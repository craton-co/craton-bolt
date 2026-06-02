// SPDX-License-Identifier: Apache-2.0

//! DISTINCT executor — host-side deduplication of a RecordBatch.
//!
//! Strategy: build an owned, typed key per row (`RowKey` — a `Vec` of
//! `RowKeyValue` entries, one per column) and accumulate a
//! `HashSet<RowKey>`. Because the set keys carry the actual column values
//! (not just a 64-bit hash digest), `HashSet::insert` only reports a row
//! as a duplicate when every column value matches an already-seen row.
//!
//! This mirrors the `JoinKey` / `JoinKeyValue` pattern in `join.rs`,
//! which has the same correctness requirement: equi-join lookups must
//! compare the real values, not just a hash.
//!
//! Historical note: the original implementation used `HashSet<u64>` keyed
//! on a `DefaultHasher` digest of the row bytes. That is silently wrong:
//! two distinct rows that hash to the same `u64` would be deduped to one,
//! and the birthday-paradox collision probability becomes non-negligible
//! around 16M rows. Worse, `DefaultHasher`'s collisions can be coerced by
//! a chosen-input adversary, so the bug had a (small) security angle as
//! well as the obvious correctness one. The fix carries the values in the
//! set, so the only way two rows collapse is if they are genuinely equal.
//!
//! Float semantics (review C12 alignment):
//!   * `+0.0` and `-0.0` are CANONICALISED to a single representation
//!     (`+0.0`) before hashing, so they dedupe to one row. This matches
//!     SQL/IEEE comparison semantics (`+0.0 == -0.0`) and what DuckDB
//!     does, and lines up with the `groupby` and host-side `join`
//!     executors which apply the same canonicalisation. See
//!     `canonicalise_f32` / `canonicalise_f64` below.
//!   * `NaN` bit patterns are CANONICALISED to a single quiet NaN (F3).
//!     Every NaN — any payload, either sign, quiet or signalling — folds
//!     to the same key, so all NaN values collapse into one DISTINCT row.
//!     This matches DuckDB, which treats all NaN as a single GROUP BY /
//!     DISTINCT key, and keeps DISTINCT and GROUP BY on one float
//!     equivalence relation. See `canonicalise_f32` / `canonicalise_f64`.
//!
//! Allocation strategy (review H9): the inner loop pre-downcasts each
//! column ONCE into a typed `ColumnReader` enum (a struct-of-arrays view
//! of the batch), then walks rows pulling values through the readers.
//! This avoids the per-row `Array::as_any` + `downcast_ref` vtable
//! shuffle that the old `extract_value(&dyn Array, row)` shape paid on
//! every (row, column) pair — for an N-row × K-column batch that is N·K
//! vtable lookups dropped to K. The `Vec<RowKeyValue>` per row is
//! preallocated with `n_cols` capacity (no growth re-allocs); the freshly
//! built key is moved into `HashSet::insert`, so on a miss it lives in
//! the set and on a hit it is dropped — same allocation count as before
//! but the per-row dtype dispatch is now branch-predictor friendly
//! (constant variant per column) instead of an `Array::data_type()`
//! match in the inner loop.
//!
//! Dispatch: two paths.
//!
//!   * **Host** (default, always correct): the `HashSet<RowKey>` dedup
//!     described above. Handles every dtype, any column count, and preserves
//!     first-occurrence order.
//!   * **GPU sort-based** (opt-in via `BOLT_GPU_DISTINCT=1`): sort the single
//!     key column on the device (reusing
//!     [`crate::exec::gpu_sort::sort_record_batch_on_gpu_multi`]), then mark
//!     adjacent-distinct rows and filter the survivors. See
//!     [`try_gpu_distinct`] for the gates and [`adjacent_distinct_mask`] for
//!     the (host-testable) dedup-after-sort core. The device "adjacent-distinct
//!     flag" kernel lives in [`crate::jit::distinct_kernel`]; the executor
//!     increment here computes the equivalent mask on the host after the GPU
//!     sort returns the reordered batch (the per-row flag is trivially cheap;
//!     the expensive sort is on-device). On ANY unsupported case — Utf8, wide
//!     multi-key, an unsupported dtype, the GPU sort declining (`Ok(None)`),
//!     or a `BoltError::GpuCapacity` decline — it returns `Ok(None)` and the
//!     caller runs the host path. **Ordering note:** the GPU path returns rows
//!     in sorted-key order, not first-occurrence order. SQL `DISTINCT` is an
//!     unordered set operation, so this is correct; callers that need a
//!     specific order must `ORDER BY` explicitly. The host path's
//!     first-occurrence order is therefore not a contract, just an artefact.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::OnceLock;

use arrow_array::{
    Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray,
};
use arrow_schema::DataType;

use crate::error::{BoltError, BoltResult};
use crate::exec::QueryHandle;

/// Maximum number of distinct rows the host-side DISTINCT may accumulate
/// before bailing. ~10M rows × ~30 bytes/key = ~300 MiB peak; deliberately
/// conservative for the host fallback. The GPU sort-based DISTINCT will
/// replace this; tracked as follow-up (see module doc-comment + ROADMAP.md).
///
/// Rationale: without this bound, `SELECT DISTINCT col FROM big_table` on a
/// high-cardinality column allocates `n_rows × n_cols × ~24 B` of host RAM
/// with no upper limit — a memory-DoS surface on user-controlled inputs.
/// The cap converts that into a clean `BoltError::Other(...)` long before
/// the OOM killer gets involved.
///
/// Overridable at runtime via the `CRATON_DISTINCT_HOST_MAX_ROWS` env var
/// (parsed once on first call; see [`distinct_host_max_rows`]).
pub(crate) const DISTINCT_HOST_MAX_ROWS: usize = 10_000_000;

/// Environment variable that overrides [`DISTINCT_HOST_MAX_ROWS`] at runtime.
/// Parsed as a base-10 `usize`; values of `0` are rejected (would disable
/// the cap entirely and reintroduce the unbounded-growth bug). On any parse
/// failure a `log::warn!` is emitted and the default is used.
const DISTINCT_HOST_MAX_ROWS_ENV: &str = "CRATON_DISTINCT_HOST_MAX_ROWS";

/// Latch for the per-process DISTINCT host-row cap. First call resolves
/// the env var; subsequent calls hit the cached `usize`. Mirrors the
/// `HASH_TABLE_BYTE_CAP_CACHE` pattern in `gpu_join.rs`.
static DISTINCT_HOST_MAX_ROWS_CACHE: OnceLock<usize> = OnceLock::new();

/// Resolve the per-process DISTINCT host-row cap. First call performs the
/// env-var lookup; subsequent calls hit the latch. On any parse failure a
/// one-time `log::warn!` is emitted and the compile-time default
/// [`DISTINCT_HOST_MAX_ROWS`] is used.
fn distinct_host_max_rows() -> usize {
    *DISTINCT_HOST_MAX_ROWS_CACHE.get_or_init(parse_distinct_host_max_rows_env)
}

/// Pure parser for `CRATON_DISTINCT_HOST_MAX_ROWS`. Extracted from the
/// OnceLock so callers (and tests) can exercise the parsing rules without
/// touching the latch. Returns the compile-time default on unset / empty /
/// unparseable / zero values, logging a warning in the unparseable / zero
/// cases.
fn parse_distinct_host_max_rows_env() -> usize {
    let raw = match std::env::var(DISTINCT_HOST_MAX_ROWS_ENV) {
        Ok(v) => v,
        Err(_) => return DISTINCT_HOST_MAX_ROWS,
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return DISTINCT_HOST_MAX_ROWS;
    }
    match trimmed.parse::<usize>() {
        Ok(0) => {
            log::warn!(
                "distinct: {DISTINCT_HOST_MAX_ROWS_ENV}='0' would disable the host-side cap; \
                 using default of {DISTINCT_HOST_MAX_ROWS}"
            );
            DISTINCT_HOST_MAX_ROWS
        }
        Ok(v) => v,
        Err(e) => {
            log::warn!(
                "distinct: {DISTINCT_HOST_MAX_ROWS_ENV}='{trimmed}' is not a valid usize ({e}); \
                 using default of {DISTINCT_HOST_MAX_ROWS}"
            );
            DISTINCT_HOST_MAX_ROWS
        }
    }
}

/// A column value inside a row key. Variants cover every primitive dtype
/// the engine produces; float variants store the raw bit pattern so that
/// `PartialEq + Eq + Hash` are bit-wise (see the module doc-comment for
/// the NaN / signed-zero implications).
///
/// Exposed `pub(crate)` so the EXCEPT / INTERSECT executor
/// ([`crate::exec::setops`]) can build the *same* row keys and therefore
/// share one row-equality / NULL-canonicalisation relation with DISTINCT.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum RowKeyValue {
    /// Column is NULL at this row. Two NULLs in the same column position
    /// compare equal, which matches the engine-wide "two NULLs dedupe to
    /// one row" convention used by the SQL `DISTINCT` operator.
    Null,
    I32(i32),
    I64(i64),
    /// `f32` reinterpreted via `to_bits`, after canonicalisation via
    /// [`canonicalise_f32`]: `-0.0 → +0.0`, and every NaN → one canonical
    /// quiet NaN so all NaN collapse to one key (F3). See module doc-comment.
    F32(u32),
    /// `f64` reinterpreted via `to_bits`. Same bit-wise semantics as `F32`.
    F64(u64),
    Bool(bool),
    Utf8(String),
}

/// Inline capacity for [`RowKey`]: keys with this many columns or fewer
/// store their values in a stack array and never touch the heap. Sized for
/// the common `n_cols <= 4` DISTINCT / set-op shape (see the H9 allocation
/// note in the module doc-comment). Wider keys spill to a heap `Vec`.
const ROW_KEY_INLINE: usize = 4;

/// A row's full key — one `RowKeyValue` per column, in column order.
///
/// This is an inline-capable small-vector: up to [`ROW_KEY_INLINE`] column
/// values live in the `Inline` array (no heap allocation — the common
/// single-/few-column case), and anything wider spills to `Heap(Vec<_>)`.
/// This replaces the old `type RowKey = Vec<RowKeyValue>` alias, whose
/// per-row heap `Vec` dominated allocator traffic on narrow DISTINCT / set
/// ops.
///
/// `pub(crate)` so [`crate::exec::setops`] can build matching keys.
///
/// # Hash / Eq invariant (load-bearing)
///
/// `Hash`, `PartialEq`, and `Eq` are all defined to operate over
/// `self.as_slice()`, i.e. the ordered sequence of `RowKeyValue`s. This is
/// byte-for-byte identical to the previous `Vec<RowKeyValue>` semantics
/// (the old alias derived these the same way and was likewise never `Ord`,
/// as `RowKeyValue` is not `Ord`):
///
///   * `Vec<T>: Hash` writes `len` then hashes each element in order; so
///     does `[T]: Hash`. Delegating `Hash` to `self.as_slice()` (a `&[T]`)
///     therefore produces the *exact same* hash stream as the old `Vec`
///     key, regardless of whether the values sit inline or on the heap.
///   * `Vec<T>: PartialEq`/`Eq` compare element-by-element in order,
///     exactly as slice comparison does — so two `RowKey`s compare equal
///     iff their value sequences are equal, independent of inline-vs-heap
///     storage. A 3-column inline key and a (hypothetical) 3-column heap
///     key with equal values hash and compare equal.
///
/// The variant (Inline vs Heap) is an internal storage detail and is NEVER
/// observed by Hash/Eq, so the equivalence relation is unchanged from the
/// `Vec` alias.
#[derive(Debug, Clone)]
pub(crate) enum RowKey {
    Inline {
        len: usize,
        vals: [RowKeyValue; ROW_KEY_INLINE],
    },
    Heap(Vec<RowKeyValue>),
}

impl RowKey {
    /// Build a `RowKey` from an exact-length iterator of column values.
    /// Mirrors `Vec::with_capacity` + `push` in a loop, but keeps the
    /// values inline when `len <= ROW_KEY_INLINE`.
    #[inline]
    pub(crate) fn from_values<I: IntoIterator<Item = RowKeyValue>>(
        n_cols: usize,
        values: I,
    ) -> Self {
        let mut it = values.into_iter();
        if n_cols <= ROW_KEY_INLINE {
            // Fill the inline array; unused tail slots hold `Null` and are
            // never read (Hash/Eq only see the first `len` entries).
            let mut vals = [
                RowKeyValue::Null,
                RowKeyValue::Null,
                RowKeyValue::Null,
                RowKeyValue::Null,
            ];
            let mut len = 0usize;
            for slot in vals.iter_mut().take(n_cols) {
                match it.next() {
                    Some(v) => {
                        *slot = v;
                        len += 1;
                    }
                    None => break,
                }
            }
            RowKey::Inline { len, vals }
        } else {
            let mut v: Vec<RowKeyValue> = Vec::with_capacity(n_cols);
            v.extend(it);
            RowKey::Heap(v)
        }
    }

    /// The ordered column values. All of `Hash`/`Eq` go through this, so
    /// the inline-vs-heap split is invisible to the key relation.
    #[inline]
    pub(crate) fn as_slice(&self) -> &[RowKeyValue] {
        match self {
            RowKey::Inline { len, vals } => &vals[..*len],
            RowKey::Heap(v) => v.as_slice(),
        }
    }
}

impl PartialEq for RowKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for RowKey {}

impl std::hash::Hash for RowKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Delegates to `[RowKeyValue]: Hash`, which writes the length then
        // each element — byte-for-byte identical to the old
        // `Vec<RowKeyValue>: Hash`.
        self.as_slice().hash(state);
    }
}

/// Pre-downcast column reader: a typed, zero-cost view into one column of
/// the input batch. Built once per column up-front so the inner row loop
/// no longer pays `Array::as_any` + `downcast_ref` per (row, column).
///
/// `pub(crate)` so [`crate::exec::setops`] can build the same typed readers
/// (and thus the same canonicalised row keys) without re-implementing the
/// per-dtype downcast.
pub(crate) enum ColumnReader<'a> {
    I32(&'a Int32Array),
    I64(&'a Int64Array),
    F32(&'a Float32Array),
    F64(&'a Float64Array),
    Bool(&'a BooleanArray),
    Utf8(&'a StringArray),
}

impl<'a> ColumnReader<'a> {
    pub(crate) fn new(array: &'a dyn Array) -> BoltResult<Self> {
        Ok(match array.data_type() {
            DataType::Int32 => ColumnReader::I32(array.as_any().downcast_ref().unwrap()),
            DataType::Int64 => ColumnReader::I64(array.as_any().downcast_ref().unwrap()),
            DataType::Float32 => ColumnReader::F32(array.as_any().downcast_ref().unwrap()),
            DataType::Float64 => ColumnReader::F64(array.as_any().downcast_ref().unwrap()),
            DataType::Boolean => ColumnReader::Bool(array.as_any().downcast_ref().unwrap()),
            DataType::Utf8 => ColumnReader::Utf8(array.as_any().downcast_ref().unwrap()),
            other => {
                return Err(BoltError::Type(format!(
                    "DISTINCT: unsupported dtype {other:?} — should have been caught by the planner"
                )))
            }
        })
    }

    /// Pull the value at `row` out as an owned `RowKeyValue`. NULL handling
    /// is uniform: any column variant returns `RowKeyValue::Null` for a
    /// null row. The only path that allocates is `Utf8`, which clones the
    /// underlying `&str` into a `String`.
    #[inline]
    pub(crate) fn value_at(&self, row: usize) -> RowKeyValue {
        match self {
            ColumnReader::I32(a) => {
                if a.is_null(row) { RowKeyValue::Null } else { RowKeyValue::I32(a.value(row)) }
            }
            ColumnReader::I64(a) => {
                if a.is_null(row) { RowKeyValue::Null } else { RowKeyValue::I64(a.value(row)) }
            }
            ColumnReader::F32(a) => {
                if a.is_null(row) {
                    RowKeyValue::Null
                } else {
                    // Canonicalise +0.0/-0.0 → +0.0 (review C12).
                    RowKeyValue::F32(canonicalise_f32(a.value(row)).to_bits())
                }
            }
            ColumnReader::F64(a) => {
                if a.is_null(row) {
                    RowKeyValue::Null
                } else {
                    // Canonicalise +0.0/-0.0 → +0.0 (review C12).
                    RowKeyValue::F64(canonicalise_f64(a.value(row)).to_bits())
                }
            }
            ColumnReader::Bool(a) => {
                if a.is_null(row) { RowKeyValue::Null } else { RowKeyValue::Bool(a.value(row)) }
            }
            ColumnReader::Utf8(a) => {
                if a.is_null(row) {
                    RowKeyValue::Null
                } else {
                    // String allocation is unavoidable in the owned-key
                    // shape; the win from H9 is that we no longer redo
                    // the downcast per row, only the clone.
                    RowKeyValue::Utf8(a.value(row).to_string())
                }
            }
        }
    }
}

/// Apply DISTINCT to the input handle, returning a new handle whose
/// RecordBatch has duplicate rows removed.
///
/// Two implementations (see the module doc-comment for the full dispatch
/// rules):
///   * **GPU sort-based** ([`try_gpu_distinct`]), opt-in via
///     `BOLT_GPU_DISTINCT=1`, for a single fixed-width primitive key column.
///     Returns rows in sorted-key order (DISTINCT is an unordered set
///     operation, so this is correct).
///   * **Host** (default), the `HashSet<RowKey>` dedup, which handles every
///     dtype and column count and preserves first-occurrence order.
///
/// The GPU path degrades to the host path on any unsupported case or a
/// `BoltError::GpuCapacity` decline, so it can never regress correctness.
///
/// Host-side note: no async opportunity here. The upstream executor that
/// produced `input` has already done its own pinned/async D2H, so the
/// `RecordBatch` we receive is already settled in host memory.
pub fn execute_distinct(input: QueryHandle) -> BoltResult<QueryHandle> {
    // GPU sort-based DISTINCT is opt-in (`BOLT_GPU_DISTINCT=1`) and degrades
    // to the host path on any unsupported case. We peek at the batch without
    // consuming the handle's ownership semantics: `try_gpu_distinct` borrows
    // the batch and returns `Some(out)` only when it produced a complete
    // result, so on `None` we fall through to the host path with the same
    // batch. On a `GpuCapacity` decline (the engine-wide "GPU path declined,
    // retry on host" marker — see `gpu_join.rs` / `gpu_compact.rs`) we also
    // fall back rather than propagate the error.
    let batch = input.into_record_batch();
    if gpu_distinct_enabled() {
        match try_gpu_distinct(&batch) {
            Ok(Some(out)) => return Ok(QueryHandle::from_record_batch(out)),
            Ok(None) => { /* fall through to host */ }
            Err(BoltError::GpuCapacity(_)) => { /* decline → host */ }
            Err(e) => return Err(e),
        }
    }
    let max_rows = distinct_host_max_rows();
    execute_distinct_with_cap(QueryHandle::from_record_batch(batch), max_rows)
}

/// Env gate for the GPU sort-based DISTINCT path. `BOLT_GPU_DISTINCT=1`
/// (or `true`/`yes`, case-insensitive) opts in; default OFF so the host
/// path stays the production default until the device round-trip has soak
/// time on real hardware. Mirrors the `BOLT_GPU_SORT` gate convention in
/// `gpu_sort.rs`.
fn gpu_distinct_enabled() -> bool {
    match std::env::var("BOLT_GPU_DISTINCT") {
        Ok(v) => {
            let t = v.trim();
            t == "1" || t.eq_ignore_ascii_case("true") || t.eq_ignore_ascii_case("yes")
        }
        Err(_) => false,
    }
}

/// Engine-internal dtypes the GPU sort-based DISTINCT path supports as the
/// single key column. Utf8 (and wide multi-key) fall back to the host path.
///
/// Operates on [`crate::plan::logical_plan::DataType`] (the *internal* dtype
/// produced by [`crate::exec::gpu_sort::arrow_dtype_to_internal`]), not the
/// Arrow `DataType` in scope at the top of this module.
fn gpu_distinct_supported_dtype(d: &crate::plan::logical_plan::DataType) -> bool {
    use crate::plan::logical_plan::DataType as Idt;
    matches!(d, Idt::Int32 | Idt::Int64 | Idt::Float32 | Idt::Float64)
}

/// GPU sort-based DISTINCT for the single, fixed-width primitive-key case.
///
/// Algorithm:
///   1. **Sort** the whole batch on the single key column using the existing
///      device sort ([`crate::exec::gpu_sort::sort_record_batch_on_gpu_multi`]).
///      NULLs are routed to one contiguous end (we ask for `nulls_first`), so
///      every NULL forms one adjacent run — the SQL "all NULLs are equal"
///      rule. The sort returns `Ok(None)` if it declines (e.g. row count below
///      its own GPU threshold, or an unsupported shape); we propagate that as
///      a host fallback.
///   2. **Mark** adjacent-distinct rows with [`adjacent_distinct_mask`]: a row
///      survives iff it is the first row or its key differs from the previous
///      row's key (with `-0.0`/NaN canonicalisation and NULL-vs-NULL collapse).
///      This is the host-side equivalent of the device flag kernel in
///      [`crate::jit::distinct_kernel`]; the per-row pass is trivially cheap
///      compared with the on-device sort.
///   3. **Filter** every column by the mask (Arrow `filter`).
///
/// Returns `Ok(None)` (→ host fallback) when:
///   * the batch is not exactly one column,
///   * the key column dtype is not one of Int32/Int64/Float32/Float64,
///   * the key column is a float that may contain `NaN` (the device sort's
///     bare IEEE compare is false on any NaN operand, so NaN rows would not
///     collapse to one DISTINCT row — host fallback canonicalises them
///     correctly; mirrors the Tier-2 float MIN/MAX NaN deferral),
///   * the GPU sort declines (`Ok(None)`).
///
/// Returns `Err(BoltError::GpuCapacity(_))` is *not* produced here directly,
/// but a `GpuCapacity` bubbling up from the sort path is caught by the caller
/// and turned into a host fallback.
///
/// **GPU-execution caveat (could not verify without a device):** the on-device
/// sort + reorder is exercised only under `cuda-stub` here; the correctness of
/// the sort itself is covered by `gpu_sort`'s own `#[ignore = "gpu:..."]`
/// round-trips. The host-testable part of THIS path — the post-sort
/// adjacent-distinct masking — is unit-tested directly via
/// [`adjacent_distinct_mask`].
fn try_gpu_distinct(batch: &RecordBatch) -> BoltResult<Option<RecordBatch>> {
    // Gate: single key column only (multi-key is a follow-up; see module doc).
    if batch.num_columns() != 1 {
        return Ok(None);
    }
    let n_rows = batch.num_rows();
    if n_rows == 0 {
        // Trivial: an empty batch is already distinct.
        return Ok(Some(batch.clone()));
    }

    let col = batch.column(0);
    let arrow_dt = col.data_type();
    // Restrict to the fixed-width primitive Arrow dtypes we can compare
    // adjacently with `adjacent_distinct_mask`. We gate on the *Arrow* dtype
    // directly rather than the mapped internal dtype because
    // `arrow_dtype_to_internal` folds plain `Utf8` (and `Dictionary(_, Utf8)`)
    // down to an integer index dtype via an inline dictionary — sorting those
    // indices is correct, but the post-sort adjacency compare would then run
    // on dictionary indices while the mask is applied to the original string
    // column. Rather than reason about that index/value split, we exclude
    // Utf8/Dictionary here and let them take the host path. Multi-key, Bool,
    // and the temporal/decimal dtypes likewise fall back.
    if !matches!(
        arrow_dt,
        DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64
    ) {
        return Ok(None);
    }
    let internal = match crate::exec::gpu_sort::arrow_dtype_to_internal(arrow_dt) {
        Some(d) if gpu_distinct_supported_dtype(&d) => d,
        _ => return Ok(None),
    };

    // NaN guard (mirror of the Tier-2 float MIN/MAX deferral in
    // `groupby_tier2_minmax_float_exec::try_execute`): the on-device sort
    // orders keys with a bare IEEE `setp.gt/lt.f`, which is false on ANY NaN
    // operand. NaN rows therefore do NOT form one contiguous adjacent run after
    // the sort, so `adjacent_distinct_mask` can emit MULTIPLE NaN output rows —
    // diverging from the documented / host "all NaN collapse to one DISTINCT
    // row" semantics (see the module doc-comment and `canonicalise_f{32,64}`).
    //
    // The fix matches the sibling float MIN/MAX executor exactly: DECLINE
    // (return `Ok(None)`) for any float key column that may contain a NaN, so
    // the caller takes the correct host path, where `canonicalise_f{32,64}`
    // fold every NaN bit pattern to one key. We scan the raw values via
    // `.values().iter().any(|v| v.is_nan())` (the same shape as the sibling); a
    // NaN sitting under a NULL slot only makes us decline conservatively, which
    // is safe. Once the device sort + flag kernel implement a NaN-as-equal
    // total order this guard can be dropped and NaN columns can take the GPU
    // path directly.
    let has_nan = match arrow_dt {
        DataType::Float32 => col
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|a| a.values().iter().any(|v| v.is_nan()))
            .unwrap_or(false),
        DataType::Float64 => col
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| a.values().iter().any(|v| v.is_nan()))
            .unwrap_or(false),
        // Int32 / Int64 keys cannot carry NaN.
        _ => false,
    };
    if has_nan {
        return Ok(None);
    }

    // Sort the batch on the single key column. nulls_first=true groups NULLs
    // at the front as one adjacent run; ASC direction is arbitrary for a set
    // operation (order is not a DISTINCT contract). The sort path declines
    // with Ok(None) below its own GPU row threshold, which we forward.
    let nullable = col.null_count() > 0;
    let keys = [(
        0usize,
        internal,
        crate::jit::sort_kernel::SortDirection::Asc,
        /* nulls_first */ true,
    )];
    let sorted = match crate::exec::gpu_sort::sort_record_batch_on_gpu_multi(batch, &keys)? {
        Some(s) => s,
        None => return Ok(None),
    };

    // Mark adjacent-distinct rows on the sorted column, then filter.
    let sorted_col = sorted.column(0);
    let mask = adjacent_distinct_mask(sorted_col.as_ref(), nullable)?;
    let filtered_cols: Vec<Arc<dyn Array>> = sorted
        .columns()
        .iter()
        .map(|c| arrow::compute::filter(c.as_ref(), &mask).map_err(arrow_err))
        .collect::<BoltResult<Vec<_>>>()?;
    let out = RecordBatch::try_new(sorted.schema(), filtered_cols).map_err(arrow_err)?;
    Ok(Some(out))
}

/// Compute the adjacent-distinct keep-mask over a **sorted** single key
/// column: `keep[i] = (i == 0) || key[i] != key[i-1]`, with SQL DISTINCT NULL
/// semantics (two adjacent NULLs collapse; NULL vs non-NULL is a boundary) and
/// float canonicalisation (`-0.0 == +0.0`, all NaN collapse to one).
///
/// This is the host-side mirror of the device flag kernel
/// ([`crate::jit::distinct_kernel::compile_distinct_flag_kernel`]). It is the
/// load-bearing, GPU-free part of the sort-based DISTINCT path, so it carries
/// the bulk of the unit tests: given a sorted column it must reproduce exactly
/// what the kernel would write to the `u8` mask.
///
/// `expect_nulls` is a hint from the caller's `null_count() > 0` check; the
/// function is correct regardless (it always consults `is_null`), the flag
/// only documents whether the validity-aware branches are expected to fire.
fn adjacent_distinct_mask(sorted: &dyn Array, expect_nulls: bool) -> BoltResult<BooleanArray> {
    let _ = expect_nulls; // documented hint only; correctness uses is_null.
    let n = sorted.len();
    let mut keep: Vec<bool> = Vec::with_capacity(n);

    // Pull a typed reader once (reuse the existing per-dtype downcast). Only
    // the fixed-width primitive variants are valid here — the GPU gate
    // guarantees that, but we re-validate so a misuse is a clean error.
    let reader = ColumnReader::new(sorted)?;
    if matches!(reader, ColumnReader::Utf8(_)) {
        return Err(BoltError::Type(
            "adjacent_distinct_mask: Utf8 is not a sort-based DISTINCT key (host fallback)".into(),
        ));
    }

    for i in 0..n {
        if i == 0 {
            keep.push(true);
            continue;
        }
        // `value_at` already applies float canonicalisation and maps NULL to
        // `RowKeyValue::Null`, so equality of two `RowKeyValue`s is exactly the
        // DISTINCT equivalence relation (Null == Null, +0.0 == -0.0, NaN ==
        // NaN). A row begins a new run iff its canonicalised value differs
        // from the previous row's.
        let cur = reader.value_at(i);
        let prev = reader.value_at(i - 1);
        keep.push(cur != prev);
    }
    Ok(BooleanArray::from(keep))
}

/// Internal entry point that lets callers (and tests) inject a cap directly,
/// bypassing the `OnceLock`-latched env-var resolution. Production code goes
/// through [`execute_distinct`]; tests use this to exercise the bound-exceeded
/// path without poisoning the global latch for other tests running in the
/// same process.
fn execute_distinct_with_cap(input: QueryHandle, max_rows: usize) -> BoltResult<QueryHandle> {
    let batch = input.into_record_batch();
    let n_rows = batch.num_rows();
    if n_rows == 0 {
        // Trivial: re-wrap.
        return Ok(QueryHandle::from_record_batch(batch));
    }

    // Pre-downcast every column ONCE (review H9). For an N-row × K-column
    // input this turns N·K vtable lookups into K.
    let n_cols = batch.num_columns();
    let readers: Vec<ColumnReader<'_>> = batch
        .columns()
        .iter()
        .map(|c| ColumnReader::new(c.as_ref()))
        .collect::<BoltResult<Vec<_>>>()?;

    // Pre-allocate the seen-set with at most `max_rows` slots so a giant
    // `n_rows` cannot drive a multi-GiB single allocation up front (the
    // unbounded-growth bug this guard exists to close). The set will still
    // grow on demand if `unique_rows` exceeds the initial capacity, but
    // the loop body's per-row cap check fires long before that becomes a
    // memory problem.
    let initial_cap = n_rows.min(max_rows);

    // Build an owned, typed key per row and check membership against the
    // set of already-seen keys. `HashSet::insert` returns `true` iff the
    // key was not already present — i.e. iff the row is a first occurrence.
    // The freshly-built `key` is *moved* into `insert`, so on a miss it
    // lives in the set and on a hit it is dropped. For the common
    // `n_cols <= ROW_KEY_INLINE` case [`RowKey`] keeps the column values
    // inline (zero heap allocations per row); wider keys spill to a single
    // heap `Vec` exactly as before.
    let mut seen: HashSet<RowKey> = HashSet::with_capacity(initial_cap);
    let mut mask_bits: Vec<bool> = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let key = RowKey::from_values(n_cols, readers.iter().map(|r| r.value_at(row)));
        let was_new = seen.insert(key);
        mask_bits.push(was_new);
        // Resource bound: keep the set from growing without limit on
        // high-cardinality inputs. The check is on `seen.len()` (not the
        // input row count) so a long input full of duplicates still
        // completes; only the *distinct* count is bounded.
        if seen.len() > max_rows {
            return Err(BoltError::Other(format!(
                "DISTINCT exceeded host bound of {max_rows} distinct rows; \
                 use the GPU sort-based variant or LIMIT the input \
                 (override via {DISTINCT_HOST_MAX_ROWS_ENV})"
            )));
        }
    }

    let mask = BooleanArray::from(mask_bits);
    let filtered_cols: Vec<Arc<dyn Array>> = batch
        .columns()
        .iter()
        .map(|c| arrow::compute::filter(c.as_ref(), &mask).map_err(arrow_err))
        .collect::<BoltResult<Vec<_>>>()?;
    let out = RecordBatch::try_new(batch.schema(), filtered_cols).map_err(arrow_err)?;
    Ok(QueryHandle::from_record_batch(out))
}

/// DISTINCT ON / "first row per key" (Postgres `SELECT DISTINCT ON`).
///
/// Walks `batch` in row order and keeps the FIRST row of each group of rows
/// that share the same values in the leading `n_keys` columns. "First" is
/// whatever order `batch` already carries — the caller (the engine's DISTINCT
/// ON executor) has applied the query's ORDER BY before calling here, so a row
/// kept is the ORDER BY-first row of its key group (Postgres semantics). When
/// ORDER BY is absent or does not lead with the keys, the kept row is the
/// input-first one — deterministic, one-per-group, matching Postgres's
/// "arbitrary but one" allowance.
///
/// Key equality reuses the engine-wide [`RowKey`] / [`RowKeyValue`]
/// canonicalisation: a NULL key compares equal to another NULL key (so NULL
/// keys form their own group, exactly like GROUP BY), and float keys fold
/// `-0.0`/`+0.0` and all-NaN together. The output preserves the input column
/// schema (all columns, including the key columns); the caller slices off the
/// leading key columns to restore the user projection.
///
/// Bounded by [`distinct_host_max_rows`] on the number of distinct keys, same
/// as [`execute_distinct`], so a high-cardinality key cannot grow the seen-set
/// without limit.
///
/// Pure (no GPU / engine state) so it is host-testable directly.
pub fn distinct_on_first_per_key(batch: &RecordBatch, n_keys: usize) -> BoltResult<RecordBatch> {
    if n_keys == 0 {
        return Err(BoltError::Other(
            "DISTINCT ON requires at least one key column".into(),
        ));
    }
    if n_keys > batch.num_columns() {
        return Err(BoltError::Other(format!(
            "DISTINCT ON: {n_keys} key columns requested but the batch has {}",
            batch.num_columns()
        )));
    }
    let n_rows = batch.num_rows();
    if n_rows == 0 {
        return Ok(batch.clone());
    }

    // Pre-downcast only the key columns once (the value columns are filtered
    // wholesale by the keep-mask, so they need no per-row reader).
    let key_readers: Vec<ColumnReader<'_>> = (0..n_keys)
        .map(|i| ColumnReader::new(batch.column(i).as_ref()))
        .collect::<BoltResult<Vec<_>>>()?;

    let max_rows = distinct_host_max_rows();
    let mut seen: HashSet<RowKey> = HashSet::with_capacity(n_rows.min(max_rows));
    let mut keep: Vec<bool> = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let key = RowKey::from_values(n_keys, key_readers.iter().map(|r| r.value_at(row)));
        let is_new = seen.insert(key);
        keep.push(is_new);
        if seen.len() > max_rows {
            return Err(BoltError::Other(format!(
                "DISTINCT ON exceeded host bound of {max_rows} distinct keys \
                 (override via {DISTINCT_HOST_MAX_ROWS_ENV})"
            )));
        }
    }

    let mask = BooleanArray::from(keep);
    let filtered_cols: Vec<Arc<dyn Array>> = batch
        .columns()
        .iter()
        .map(|c| arrow::compute::filter(c.as_ref(), &mask).map_err(arrow_err))
        .collect::<BoltResult<Vec<_>>>()?;
    RecordBatch::try_new(batch.schema(), filtered_cols).map_err(arrow_err)
}

/// Collapse `-0.0` to `+0.0` so that signed-zero pairs hash identically
/// under DISTINCT. F3: also fold every `NaN` bit pattern (any payload, either
/// sign) to a single canonical quiet NaN so all NaN values dedupe to one
/// DISTINCT row, matching DuckDB. Mirrors the host-side canonicalisation
/// applied in `groupby_common::canonicalise_f64` (and the GROUP BY key path)
/// so DISTINCT and GROUP BY share one equivalence relation for floats.
#[inline]
pub(crate) fn canonicalise_f64(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// `f32` analogue of [`canonicalise_f64`]; same shape, same rationale.
#[inline]
pub(crate) fn canonicalise_f32(x: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

fn arrow_err(e: arrow::error::ArrowError) -> BoltError {
    BoltError::Other(format!("arrow: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{
        BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    };
    use arrow_schema::{DataType, Field, Schema};

    /// Build a one-column Int32 batch from the given values.
    fn int32_batch(values: Vec<Option<i32>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let arr: Arc<dyn Array> = Arc::new(Int32Array::from(values));
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    /// Extract the Int32 column at index 0 as a Vec<Option<i32>>.
    fn col_to_vec(batch: &RecordBatch, col: usize) -> Vec<Option<i32>> {
        let arr = batch
            .column(col)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("expected Int32 column");
        (0..arr.len())
            .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
            .collect()
    }

    #[test]
    fn distinct_int32_no_dups_returns_all_rows() {
        let batch = int32_batch(vec![Some(1), Some(2), Some(3), Some(4)]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(out_batch.num_rows(), 4);
        assert_eq!(
            col_to_vec(&out_batch, 0),
            vec![Some(1), Some(2), Some(3), Some(4)]
        );
    }

    #[test]
    fn distinct_int32_with_dups_drops_duplicates() {
        let batch = int32_batch(vec![Some(1), Some(2), Some(1), Some(3), Some(2)]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(out_batch.num_rows(), 3);
        // first-occurrence wins
        assert_eq!(
            col_to_vec(&out_batch, 0),
            vec![Some(1), Some(2), Some(3)]
        );
    }

    #[test]
    fn distinct_preserves_first_occurrence_order() {
        let batch = int32_batch(vec![Some(7), Some(3), Some(5), Some(3), Some(7), Some(9)]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(
            col_to_vec(&out_batch, 0),
            vec![Some(7), Some(3), Some(5), Some(9)]
        );
    }

    #[test]
    fn distinct_empty_input_is_empty() {
        let batch = int32_batch(vec![]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        assert_eq!(out.num_rows(), 0);
    }

    #[test]
    fn distinct_handles_nulls() {
        // Two NULLs in the same column should compare equal and dedupe to one.
        let batch = int32_batch(vec![None, Some(1), None, Some(1), Some(2)]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(out_batch.num_rows(), 3);
        assert_eq!(col_to_vec(&out_batch, 0), vec![None, Some(1), Some(2)]);
    }

    /// Review C12: `+0.0` and `-0.0` belong to the same equivalence
    /// class for DISTINCT (matches SQL/IEEE and DuckDB). Two rows
    /// holding signed-zero pairs must dedupe to one row.
    #[test]
    fn distinct_signed_zero_dedupes_to_one_row() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "f",
            DataType::Float64,
            false,
        )]));
        let arr: Arc<dyn Array> =
            Arc::new(Float64Array::from(vec![0.0_f64, -0.0_f64, 0.0_f64]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        // {+0.0, -0.0, +0.0} all collapse to the canonical +0.0 key, so
        // only the first row survives.
        assert_eq!(out.into_record_batch().num_rows(), 1);
    }

    /// F3: every `NaN` bit pattern (any payload, either sign, quiet or
    /// signalling) is folded to a single canonical quiet NaN so that all NaN
    /// values collapse into one DISTINCT key, matching DuckDB. Signed-zero
    /// canonicalisation is also preserved.
    #[test]
    fn distinct_nan_canonicalisation_collapses_all_nan() {
        // Distinct NaN payloads + both NaN signs all canonicalise to the
        // SAME bit pattern.
        let payload_nan = f64::from_bits(0x7ff8_0000_0000_0001); // quiet NaN, payload 1
        let sign_nan = f64::from_bits(0xfff8_0000_0000_0000); // negative NaN
        let plain_nan = f64::NAN;
        let canon = canonicalise_f64(f64::NAN).to_bits();
        assert!(canonicalise_f64(payload_nan).is_nan());
        assert_eq!(canonicalise_f64(payload_nan).to_bits(), canon);
        assert_eq!(canonicalise_f64(sign_nan).to_bits(), canon);
        assert_eq!(canonicalise_f64(plain_nan).to_bits(), canon);
        // f32 analogue.
        let f32_canon = canonicalise_f32(f32::NAN).to_bits();
        assert_eq!(
            canonicalise_f32(f32::from_bits(0x7fc0_0001)).to_bits(),
            f32_canon
        );
        assert_eq!(
            canonicalise_f32(f32::from_bits(0xffc0_0000)).to_bits(),
            f32_canon
        );
        // Signed-zero canonicalisation still happens.
        assert_eq!(canonicalise_f64(-0.0_f64).to_bits(), 0.0_f64.to_bits());
        assert_eq!(canonicalise_f32(-0.0_f32).to_bits(), 0.0_f32.to_bits());
    }

    /// F3 end-to-end: a DISTINCT over a Float64 column containing several
    /// different NaN payloads collapses them all into one output row.
    #[test]
    fn distinct_collapses_multiple_nan_payloads() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "f",
            DataType::Float64,
            false,
        )]));
        let arr: Arc<dyn Array> = Arc::new(Float64Array::from(vec![
            f64::from_bits(0x7ff8_0000_0000_0001),
            f64::from_bits(0x7ff8_0000_0000_0002),
            f64::from_bits(0xfff8_0000_0000_0000),
            f64::NAN,
        ]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        assert_eq!(
            out.into_record_batch().num_rows(),
            1,
            "all NaN payloads must collapse to one DISTINCT row"
        );
    }

    #[test]
    fn distinct_multi_column_utf8_and_int() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, false),
            Field::new("n", DataType::Int32, false),
        ]));
        let s: Arc<dyn Array> =
            Arc::new(StringArray::from(vec!["a", "b", "a", "a", "b"]));
        let n: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 2, 1, 3, 2]));
        let batch = RecordBatch::try_new(schema, vec![s, n]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        let out_batch = out.into_record_batch();
        // ("a",1), ("b",2), ("a",3) — three uniques.
        assert_eq!(out_batch.num_rows(), 3);
    }

    /// Multi-column input where two rows differ in exactly one column —
    /// they must not collapse. Regression test for the hash-only shape,
    /// where a `u64` collision could silently drop one of them.
    #[test]
    fn distinct_multi_column_differs_in_one_column_kept_separate() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int64, false),
            Field::new("c", DataType::Utf8, false),
        ]));
        // Rows 0 and 1 differ only in column `b`. Rows 0 and 2 differ only
        // in column `c`. Row 3 is an exact duplicate of row 0. Expected:
        // three unique rows (0, 1, 2); row 3 deduped away.
        let a: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 1, 1, 1]));
        let b: Arc<dyn Array> = Arc::new(Int64Array::from(vec![10_i64, 20, 10, 10]));
        let c: Arc<dyn Array> = Arc::new(StringArray::from(vec!["x", "x", "y", "x"]));
        let batch = RecordBatch::try_new(schema, vec![a, b, c]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(out_batch.num_rows(), 3);
    }

    /// Synthetic-collision regression: two genuinely different rows whose
    /// row-byte hashes coincide must still be kept as two separate rows.
    /// We simulate a "hash collision" by checking on a small input where
    /// the unique count is known; the old `HashSet<u64>` would have lost a
    /// row only on a probabilistic collision (rare on tiny inputs), so the
    /// strongest signal we can give in a unit test is to verify that the
    /// output preserves every truly-distinct row across a mix of dtypes.
    #[test]
    fn distinct_keeps_all_distinct_rows_across_dtypes() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int64, false),
            Field::new("f", DataType::Float64, false),
            Field::new("b", DataType::Boolean, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        // 5 genuinely distinct rows.
        let i: Arc<dyn Array> =
            Arc::new(Int64Array::from(vec![1_i64, 2, 3, 4, 5]));
        let f: Arc<dyn Array> = Arc::new(Float64Array::from(vec![
            1.5_f64, 2.5, 3.5, 4.5, 5.5,
        ]));
        let b: Arc<dyn Array> =
            Arc::new(BooleanArray::from(vec![true, false, true, false, true]));
        let s: Arc<dyn Array> =
            Arc::new(StringArray::from(vec!["a", "b", "c", "d", "e"]));
        let batch = RecordBatch::try_new(schema, vec![i, f, b, s]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        assert_eq!(out.into_record_batch().num_rows(), 5);
    }

    /// NaN-vs-NaN: two `f64::NAN`s with the same bit pattern dedupe to one
    /// row. This is the documented engine-wide stance; see module doc.
    #[test]
    fn distinct_nan_dedupes_to_one_row() {
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
        let arr: Arc<dyn Array> = Arc::new(Float64Array::from(vec![
            f64::NAN,
            1.0,
            f64::NAN,
            2.0,
        ]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        let out_batch = out.into_record_batch();
        // [NaN, 1.0, 2.0] — three rows.
        assert_eq!(out_batch.num_rows(), 3);
    }

    /// Signed-zero: `+0.0` and `-0.0` collapse to a single equivalence class
    /// per `canonicalise_f64` — DISTINCT, GROUP BY, and JOIN all share this
    /// rule. The three input rows therefore reduce to a single output row.
    #[test]
    fn distinct_zero_signs() {
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
        let arr: Arc<dyn Array> =
            Arc::new(Float64Array::from(vec![0.0_f64, -0.0_f64, 0.0_f64]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        let out_batch = out.into_record_batch();
        // All three rows are equivalent under the shared canonicalisation
        // rule; expected: 1.
        assert_eq!(out_batch.num_rows(), 1);
    }

    /// `f32::NAN` path: same bit-pattern dedup rule as `f64::NAN`.
    #[test]
    fn distinct_f32_nan_dedupes_to_one_row() {
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float32, false)]));
        let arr: Arc<dyn Array> = Arc::new(Float32Array::from(vec![
            f32::NAN,
            f32::NAN,
            1.0_f32,
        ]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        assert_eq!(out.into_record_batch().num_rows(), 2);
    }

    /// All-NULL column: every row's key is `[Null]`, so they collapse to a
    /// single output row.
    #[test]
    fn distinct_all_null_column_yields_one_row() {
        let batch = int32_batch(vec![None, None, None, None, None]);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(out_batch.num_rows(), 1);
        assert_eq!(col_to_vec(&out_batch, 0), vec![None]);
    }

    /// Resource bound, happy path: a 100-row, 1-col input with all unique
    /// values produces the correct unique count (100) and does NOT trip the
    /// host cap. Locks in that the cap check is `>`-not-`>=` (off-by-one
    /// regression guard).
    #[test]
    fn distinct_100_row_1_col_unique_count_is_correct() {
        let values: Vec<Option<i32>> = (0..100).map(Some).collect();
        let batch = int32_batch(values);
        let input = QueryHandle::from_record_batch(batch);
        let out = execute_distinct(input).unwrap();
        let out_batch = out.into_record_batch();
        assert_eq!(out_batch.num_rows(), 100);
        // First-occurrence order is the natural 0..100 sequence.
        let expected: Vec<Option<i32>> = (0..100).map(Some).collect();
        assert_eq!(col_to_vec(&out_batch, 0), expected);
    }

    /// Resource bound, exceeded path: with the cap set to a small value
    /// (3), an input whose distinct count overflows the cap must fail with
    /// the "DISTINCT exceeded host bound" error rather than allocating
    /// without limit. Goes through the cap-injection helper so the global
    /// `OnceLock` is untouched (and concurrent tests are unaffected).
    #[test]
    fn distinct_bound_exceeded_returns_clear_error() {
        // 5 distinct values, cap of 3 — must error after the 4th unique.
        let batch = int32_batch(vec![Some(1), Some(2), Some(3), Some(4), Some(5)]);
        let input = QueryHandle::from_record_batch(batch);
        let result = execute_distinct_with_cap(input, 3);
        let err = match result {
            Ok(_) => panic!("expected DISTINCT bound to be exceeded with cap=3 and 5 distinct rows"),
            Err(e) => e,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("DISTINCT exceeded host bound"),
            "error did not mention the bound: {msg}"
        );
        assert!(
            msg.contains("3"),
            "error did not include the configured cap: {msg}"
        );
        assert!(
            msg.contains(DISTINCT_HOST_MAX_ROWS_ENV),
            "error did not mention the override env var: {msg}"
        );
    }

    /// Bound check operates on the *distinct* count, not the input row
    /// count: an input full of duplicates well past `max_rows` total rows
    /// but with `distinct <= max_rows` must still complete successfully.
    /// Regression guard for the obvious off-by-one of testing `n_rows`
    /// instead of `seen.len()`.
    #[test]
    fn distinct_bound_counts_uniques_not_input_rows() {
        // 10 input rows, all duplicates of two values — only 2 uniques.
        let batch = int32_batch(vec![
            Some(1),
            Some(2),
            Some(1),
            Some(2),
            Some(1),
            Some(2),
            Some(1),
            Some(2),
            Some(1),
            Some(2),
        ]);
        let input = QueryHandle::from_record_batch(batch);
        // Cap of 2 is enough — `seen.len()` never exceeds 2.
        let out = execute_distinct_with_cap(input, 2).unwrap();
        assert_eq!(out.into_record_batch().num_rows(), 2);
    }

    /// Env-var parser: unset / empty / unparseable / zero all fall back to
    /// the compile-time default; a valid positive integer wins. Exercised
    /// against the pure parser ([`parse_distinct_host_max_rows_env`]) so
    /// the `OnceLock` latch is not poisoned.
    ///
    /// Serialised on a local lock — `std::env` is process-global and other
    /// tests in this module do not touch `CRATON_DISTINCT_HOST_MAX_ROWS`,
    /// so a single lock here is sufficient.
    #[test]
    fn distinct_env_var_parser_handles_all_paths() {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();

        // Helper: set/clear the env var around a single parser call.
        let with_env = |val: Option<&str>, expected: usize, why: &str| {
            match val {
                Some(v) => std::env::set_var(DISTINCT_HOST_MAX_ROWS_ENV, v),
                None => std::env::remove_var(DISTINCT_HOST_MAX_ROWS_ENV),
            }
            let got = parse_distinct_host_max_rows_env();
            std::env::remove_var(DISTINCT_HOST_MAX_ROWS_ENV);
            assert_eq!(got, expected, "{why}");
        };

        // Unset → default.
        with_env(None, DISTINCT_HOST_MAX_ROWS, "unset should fall back to default");
        // Empty / whitespace → default.
        with_env(Some(""), DISTINCT_HOST_MAX_ROWS, "empty should fall back to default");
        with_env(Some("   "), DISTINCT_HOST_MAX_ROWS, "whitespace should fall back to default");
        // Unparseable → default (warn fires; we don't assert on the log).
        with_env(Some("not-a-number"), DISTINCT_HOST_MAX_ROWS, "unparseable should fall back");
        with_env(Some("-5"), DISTINCT_HOST_MAX_ROWS, "negative should fall back (usize parse fails)");
        // Zero is explicitly rejected — would disable the cap entirely.
        with_env(Some("0"), DISTINCT_HOST_MAX_ROWS, "zero should fall back to default");
        // Valid positive integer wins.
        with_env(Some("42"), 42, "valid positive integer should win");
        with_env(Some("  100  "), 100, "leading/trailing whitespace should be trimmed");
    }

    /// Boolean dtype: only two possible values, plus optional NULLs.
    #[test]
    fn distinct_boolean_dedupes() {
        let schema = Arc::new(Schema::new(vec![Field::new("b", DataType::Boolean, true)]));
        let arr: Arc<dyn Array> = Arc::new(BooleanArray::from(vec![
            Some(true),
            Some(false),
            Some(true),
            None,
            Some(false),
            None,
        ]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        let out = execute_distinct(QueryHandle::from_record_batch(batch)).unwrap();
        // {true, false, NULL} — three rows.
        assert_eq!(out.into_record_batch().num_rows(), 3);
    }

    /// Hash a single value through `std::hash::Hash` with the default hasher.
    fn hash_of<T: std::hash::Hash>(t: &T) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut h = DefaultHasher::new();
        t.hash(&mut h);
        h.finish()
    }

    /// The inline `RowKey` must hash/compare byte-for-byte identically to the
    /// old `Vec<RowKeyValue>` shape it replaced. We pin this by hashing the
    /// `RowKey` against a plain `Vec<RowKeyValue>` carrying the same values:
    /// `Vec<T>: Hash` writes the length then each element, and the inline
    /// key delegates `Hash` to `self.as_slice()` (`[T]: Hash`) which does the
    /// same, so the two hash streams MUST coincide.
    #[test]
    fn row_key_inline_matches_vec_hash_and_eq() {
        // Single-column (inline) key vs the equivalent Vec.
        let vals = vec![RowKeyValue::I32(42)];
        let key = RowKey::from_values(1, vals.iter().cloned());
        assert_eq!(key.as_slice(), vals.as_slice());
        assert_eq!(hash_of(&key), hash_of(&vals));

        // Multi-column inline key (within ROW_KEY_INLINE) vs Vec.
        let vals3 = vec![
            RowKeyValue::I64(7),
            RowKeyValue::Null,
            RowKeyValue::Utf8("x".to_string()),
        ];
        let key3 = RowKey::from_values(3, vals3.iter().cloned());
        assert!(matches!(key3, RowKey::Inline { .. }));
        assert_eq!(key3.as_slice(), vals3.as_slice());
        assert_eq!(hash_of(&key3), hash_of(&vals3));

        // Two equal-valued keys compare equal and hash equal.
        let key3b = RowKey::from_values(3, vals3.iter().cloned());
        assert_eq!(key3, key3b);
        assert_eq!(hash_of(&key3), hash_of(&key3b));

        // Differing in one slot ⇒ not equal.
        let diff = RowKey::from_values(
            3,
            vec![
                RowKeyValue::I64(7),
                RowKeyValue::Null,
                RowKeyValue::Utf8("y".to_string()),
            ],
        );
        assert_ne!(key3, diff);

        // Wide key spills to heap but still matches the Vec relation.
        let wide_vals: Vec<RowKeyValue> =
            (0..ROW_KEY_INLINE as i32 + 2).map(RowKeyValue::I32).collect();
        let wide = RowKey::from_values(wide_vals.len(), wide_vals.iter().cloned());
        assert!(matches!(wide, RowKey::Heap(_)));
        assert_eq!(wide.as_slice(), wide_vals.as_slice());
        assert_eq!(hash_of(&wide), hash_of(&wide_vals));
    }

    // =====================================================================
    // GPU sort-based DISTINCT — host-testable parts.
    //
    // The on-device sort + reorder is exercised by `gpu_sort`'s own
    // `#[ignore = "gpu:..."]` round-trips. Here we unit-test the two
    // GPU-free pieces of `try_gpu_distinct`: the env gate and the
    // post-sort adjacent-distinct masking (`adjacent_distinct_mask`),
    // which is the host mirror of `jit::distinct_kernel`.
    // =====================================================================

    /// Collect a `BooleanArray` mask into a `Vec<bool>` for assertions.
    fn mask_to_vec(m: &BooleanArray) -> Vec<bool> {
        (0..m.len()).map(|i| m.value(i)).collect()
    }

    /// On a SORTED i32 column, a row is kept iff it differs from its
    /// predecessor; runs of equal values collapse to their first row.
    #[test]
    fn adjacent_mask_i32_collapses_runs() {
        let arr = Int32Array::from(vec![1, 1, 2, 3, 3, 3, 4]);
        let mask = adjacent_distinct_mask(&arr, false).unwrap();
        assert_eq!(
            mask_to_vec(&mask),
            vec![true, false, true, true, false, false, true]
        );
    }

    /// Single-element column: the lone row is always kept (it is row 0).
    #[test]
    fn adjacent_mask_single_row_kept() {
        let arr = Int64Array::from(vec![42_i64]);
        let mask = adjacent_distinct_mask(&arr, false).unwrap();
        assert_eq!(mask_to_vec(&mask), vec![true]);
    }

    /// NULLs sorted into one adjacent run collapse to a single kept row; the
    /// NULL↔non-NULL boundary is a keep. Layout mirrors what the sort emits
    /// with `nulls_first=true`: [NULL, NULL, 5, 5, 7].
    #[test]
    fn adjacent_mask_nulls_collapse_and_boundary_kept() {
        let arr = Int32Array::from(vec![None, None, Some(5), Some(5), Some(7)]);
        let mask = adjacent_distinct_mask(&arr, true).unwrap();
        // row0: keep (first). row1: NULL==NULL -> drop. row2: NULL->5 boundary
        // keep. row3: 5==5 drop. row4: 5->7 keep.
        assert_eq!(mask_to_vec(&mask), vec![true, false, true, false, true]);
    }

    /// Float canonicalisation: a sorted run containing `+0.0` then `-0.0`
    /// collapses (they are one DISTINCT equivalence class), and adjacent
    /// canonical NaNs collapse too. We hand a column already in sorted order
    /// (what the device sort would produce; +0.0/-0.0 are bit-distinct but
    /// the mask must still treat them equal).
    #[test]
    fn adjacent_mask_f64_signed_zero_and_nan_collapse() {
        // Two signed zeros adjacent -> collapse; two NaNs adjacent -> collapse.
        let arr = Float64Array::from(vec![0.0_f64, -0.0_f64, 1.5, f64::NAN, f64::NAN]);
        let mask = adjacent_distinct_mask(&arr, false).unwrap();
        // row0 keep; row1 -0.0==+0.0 drop; row2 1.5 keep; row3 NaN keep
        // (1.5 != NaN); row4 NaN==NaN drop.
        assert_eq!(mask_to_vec(&mask), vec![true, false, true, true, false]);
    }

    /// Utf8 is rejected by the masking helper (the GPU gate also excludes it,
    /// but the helper guards independently so a misuse is a clean error).
    #[test]
    fn adjacent_mask_rejects_utf8() {
        let arr = StringArray::from(vec!["a", "a", "b"]);
        assert!(adjacent_distinct_mask(&arr, false).is_err());
    }

    /// End-to-end masking + filter on a sorted column reproduces the unique
    /// set (order is sorted-key order, which DISTINCT does not constrain).
    #[test]
    fn adjacent_mask_then_filter_yields_uniques() {
        let arr: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 1, 2, 2, 2, 5]));
        let mask = adjacent_distinct_mask(arr.as_ref(), false).unwrap();
        let filtered = arrow::compute::filter(arr.as_ref(), &mask).unwrap();
        let out = filtered.as_any().downcast_ref::<Int32Array>().unwrap();
        let got: Vec<i32> = (0..out.len()).map(|i| out.value(i)).collect();
        assert_eq!(got, vec![1, 2, 5]);
    }

    /// dtype gate: only the fixed-width primitives are accepted by the GPU
    /// sort-based DISTINCT path; everything else falls back to host.
    #[test]
    fn gpu_distinct_dtype_gate() {
        use crate::plan::logical_plan::DataType as Idt;
        assert!(gpu_distinct_supported_dtype(&Idt::Int32));
        assert!(gpu_distinct_supported_dtype(&Idt::Int64));
        assert!(gpu_distinct_supported_dtype(&Idt::Float32));
        assert!(gpu_distinct_supported_dtype(&Idt::Float64));
        assert!(!gpu_distinct_supported_dtype(&Idt::Utf8));
        assert!(!gpu_distinct_supported_dtype(&Idt::Bool));
    }

    /// `try_gpu_distinct` declines (Ok(None)) on shapes it does not handle —
    /// multi-column and Utf8 — so the caller falls back to the host path.
    /// These checks don't touch the GPU: the gate fires before any sort.
    #[test]
    fn gpu_distinct_declines_unsupported_shapes() {
        // Multi-column: declined regardless of dtype.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 2, 1]));
        let b: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 2, 1]));
        let batch = RecordBatch::try_new(schema, vec![a, b]).unwrap();
        assert!(matches!(try_gpu_distinct(&batch), Ok(None)));

        // Single Utf8 column: declined (Utf8 is host-only here).
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let s: Arc<dyn Array> = Arc::new(StringArray::from(vec!["a", "a", "b"]));
        let batch = RecordBatch::try_new(schema, vec![s]).unwrap();
        assert!(matches!(try_gpu_distinct(&batch), Ok(None)));
    }

    /// NaN guard: a single float key column containing a NaN must DECLINE
    /// (`Ok(None)`) so the caller takes the host path, where the all-NaN
    /// values collapse to one DISTINCT row. The decline fires before any sort,
    /// so this is GPU-free. Mirrors the Tier-2 float MIN/MAX NaN deferral.
    #[test]
    fn gpu_distinct_declines_nan_float_columns() {
        // Float64 with a NaN -> declined.
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
        let arr: Arc<dyn Array> = Arc::new(Float64Array::from(vec![1.0_f64, f64::NAN, 2.0]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        assert!(matches!(try_gpu_distinct(&batch), Ok(None)));

        // Float32 with a NaN -> declined.
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float32, false)]));
        let arr: Arc<dyn Array> = Arc::new(Float32Array::from(vec![1.0_f32, f32::NAN]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        assert!(matches!(try_gpu_distinct(&batch), Ok(None)));

        // All-NaN Float64 column -> declined (it would otherwise emit multiple
        // NaN rows after the bare-IEEE device sort; the host path collapses
        // them to one).
        let schema = Arc::new(Schema::new(vec![Field::new("f", DataType::Float64, false)]));
        let arr: Arc<dyn Array> =
            Arc::new(Float64Array::from(vec![f64::NAN, f64::NAN, f64::NAN]));
        let batch = RecordBatch::try_new(schema, vec![arr]).unwrap();
        assert!(matches!(try_gpu_distinct(&batch), Ok(None)));
    }

    /// Empty single-column batch is trivially distinct (no sort needed).
    #[test]
    fn gpu_distinct_empty_batch_is_passthrough() {
        let batch = int32_batch(vec![]);
        let out = try_gpu_distinct(&batch).unwrap();
        assert!(out.is_some());
        assert_eq!(out.unwrap().num_rows(), 0);
    }

    /// Env gate parses `1`/`true`/`yes` as enabled, everything else disabled.
    /// Serialised on a local lock because `std::env` is process-global.
    #[test]
    fn gpu_distinct_env_gate() {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _g = ENV_LOCK.lock().unwrap();
        let probe = |val: Option<&str>| {
            match val {
                Some(v) => std::env::set_var("BOLT_GPU_DISTINCT", v),
                None => std::env::remove_var("BOLT_GPU_DISTINCT"),
            }
            let got = gpu_distinct_enabled();
            std::env::remove_var("BOLT_GPU_DISTINCT");
            got
        };
        assert!(!probe(None));
        assert!(!probe(Some("0")));
        assert!(!probe(Some("off")));
        assert!(probe(Some("1")));
        assert!(probe(Some("true")));
        assert!(probe(Some("YES")));
    }

    /// Real device round-trip for the GPU sort-based DISTINCT path. Ignored
    /// without a GPU per the repo convention; requires `BOLT_GPU_DISTINCT=1`
    /// and a batch large enough to clear the sort path's own GPU row
    /// threshold (`GPU_SORT_MIN_ROWS`). Verifies that the deduped output is
    /// the set of unique input values (sorted order is acceptable).
    #[test]
    #[ignore = "gpu:distinct"]
    fn gpu_distinct_roundtrip_i32() {
        // 32K rows of two repeating values -> two uniques. Well above the
        // sort path's GPU threshold so the device path actually engages.
        let n = 32_768usize;
        let vals: Vec<Option<i32>> = (0..n).map(|i| Some((i % 2) as i32)).collect();
        let batch = int32_batch(vals);
        std::env::set_var("BOLT_GPU_DISTINCT", "1");
        let out = try_gpu_distinct(&batch).unwrap();
        std::env::remove_var("BOLT_GPU_DISTINCT");
        let out = out.expect("GPU distinct should engage above the sort threshold");
        let mut got = col_to_vec(&out, 0);
        got.sort();
        assert_eq!(got, vec![Some(0), Some(1)]);
    }

    // ---- DISTINCT ON / first-per-key -------------------------------------

    /// Build a two-column (key Int32, value Int32) batch.
    fn key_val_batch(keys: Vec<Option<i32>>, vals: Vec<Option<i32>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("v", DataType::Int32, true),
        ]));
        let k: Arc<dyn Array> = Arc::new(Int32Array::from(keys));
        let v: Arc<dyn Array> = Arc::new(Int32Array::from(vals));
        RecordBatch::try_new(schema, vec![k, v]).unwrap()
    }

    /// DISTINCT ON keeps exactly the FIRST row per key, in input (ORDER BY)
    /// order. Input pre-sorted by (k, v) ascending — so the first row of each
    /// key group is the one with the smallest v.
    #[test]
    fn distinct_on_keeps_first_row_per_key() {
        // (1,10) (1,11) (2,20) (2,21) (3,30) — pre-sorted by (k,v).
        let batch = key_val_batch(
            vec![Some(1), Some(1), Some(2), Some(2), Some(3)],
            vec![Some(10), Some(11), Some(20), Some(21), Some(30)],
        );
        let out = distinct_on_first_per_key(&batch, 1).unwrap();
        assert_eq!(out.num_rows(), 3);
        // The kept value column is the leading (smallest-v) row of each key.
        assert_eq!(col_to_vec(&out, 1), vec![Some(10), Some(20), Some(30)]);
    }

    /// A NULL key forms its own group (like GROUP BY): the first NULL-key row
    /// is kept, later NULL-key rows are dropped.
    #[test]
    fn distinct_on_null_key_is_its_own_group() {
        // keys: NULL, 1, NULL, 1 — first NULL row and first key-1 row kept.
        let batch = key_val_batch(
            vec![None, Some(1), None, Some(1)],
            vec![Some(100), Some(200), Some(101), Some(201)],
        );
        let out = distinct_on_first_per_key(&batch, 1).unwrap();
        assert_eq!(out.num_rows(), 2);
        assert_eq!(col_to_vec(&out, 0), vec![None, Some(1)]);
        assert_eq!(col_to_vec(&out, 1), vec![Some(100), Some(200)]);
    }

    /// Multi-key DISTINCT ON: dedup on the leading two key columns.
    #[test]
    fn distinct_on_multi_key() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k1", DataType::Int32, false),
            Field::new("k2", DataType::Int32, false),
            Field::new("v", DataType::Int32, false),
        ]));
        let k1: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 1, 1, 2]));
        let k2: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 1, 2, 1]));
        let v: Arc<dyn Array> = Arc::new(Int32Array::from(vec![10, 11, 12, 13]));
        let batch = RecordBatch::try_new(schema, vec![k1, k2, v]).unwrap();
        // Groups: (1,1) (1,2) (2,1) — three kept.
        let out = distinct_on_first_per_key(&batch, 2).unwrap();
        assert_eq!(out.num_rows(), 3);
        assert_eq!(col_to_vec(&out, 2), vec![Some(10), Some(12), Some(13)]);
    }

    /// Empty input yields empty output.
    #[test]
    fn distinct_on_empty_input() {
        let batch = key_val_batch(vec![], vec![]);
        let out = distinct_on_first_per_key(&batch, 1).unwrap();
        assert_eq!(out.num_rows(), 0);
    }

    /// Zero key columns is a clean error (the engine never builds this).
    #[test]
    fn distinct_on_zero_keys_errors() {
        let batch = key_val_batch(vec![Some(1)], vec![Some(2)]);
        assert!(distinct_on_first_per_key(&batch, 0).is_err());
    }
}
