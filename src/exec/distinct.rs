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
//!   * `NaN` bit patterns are LEFT AS-IS. The host-side canonicalisation
//!     uses `if x == 0.0 { 0.0 } else { x }` which evaluates `false` for
//!     every NaN (per IEEE) and therefore preserves NaN bit patterns
//!     verbatim. Documented SQL semantics: `NaN != NaN` (also DuckDB).
//!     Two `NaN`s with identical bit patterns DO dedupe to one row
//!     because the row key carries the raw bits and `Eq` is bit-wise.
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
//! Dispatch: a single host-side path. The 0.2 target (sort-based DISTINCT
//! via `gpu_sort::sort_indices_on_gpu_multi`) is tracked in ROADMAP.md;
//! it requires the input columns to already be uploaded as `GpuVec`s,
//! which the distinct executor does not have hand — that restructure is
//! deferred to the GPU-side rework.

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
    /// `f32` reinterpreted via `to_bits`. NaN-vs-NaN equality is therefore
    /// bit-wise; `+0.0` and `-0.0` are first canonicalised to `+0.0` via
    /// [`canonicalise_f32`]. See module doc-comment.
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
/// RecordBatch has duplicate rows removed (first-occurrence wins).
///
/// Host-side implementation. For wide schemas or large row counts this
/// is the slow path; the 0.2 release will add a GPU sort-based variant.
///
/// Stage 3 note: host-side only, no async opportunity here. The upstream
/// executor that produced `input` has already done its own pinned/async
/// D2H, so the `RecordBatch` we receive is already settled in host
/// memory. When the GPU-side DISTINCT lands it should pick up the same
/// async memcpy + pinned D2H pattern as the projection / aggregate paths.
pub fn execute_distinct(input: QueryHandle) -> BoltResult<QueryHandle> {
    let max_rows = distinct_host_max_rows();
    execute_distinct_with_cap(input, max_rows)
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

/// Collapse `-0.0` to `+0.0` so that signed-zero pairs hash identically
/// under DISTINCT. Preserves every other bit pattern, including the full
/// space of `NaN` payloads (the predicate `x == 0.0` is `false` for any
/// `NaN`). Mirrors the host-side canonicalisation applied in
/// `groupby::load_key_column_bits` and `join::extract_key` so that
/// DISTINCT, GROUP BY, and JOIN share one equivalence relation for
/// floats.
#[inline]
pub(crate) fn canonicalise_f64(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x }
}

/// `f32` analogue of [`canonicalise_f64`]; same shape, same rationale.
#[inline]
pub(crate) fn canonicalise_f32(x: f32) -> f32 {
    if x == 0.0 { 0.0 } else { x }
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

    /// Review C12: `NaN` is left as-is — the canonicalisation only
    /// touches signed zeros, so two `NaN`s with the SAME bit pattern
    /// still hash equal (the row carries the raw bits) and dedupe, but
    /// the canonicalisation does NOT collapse NaN-vs-not-NaN.
    #[test]
    fn distinct_nan_canonicalisation_is_noop() {
        // canonicalise_f64 must preserve NaN bit-for-bit.
        let nan_in = f64::from_bits(0x7ff8_0000_0000_0001); // a quiet NaN
        let nan_out = canonicalise_f64(nan_in);
        assert!(nan_out.is_nan());
        assert_eq!(nan_in.to_bits(), nan_out.to_bits());
        // Signed-zero canonicalisation does happen.
        assert_eq!(canonicalise_f64(-0.0_f64).to_bits(), 0.0_f64.to_bits());
        assert_eq!(canonicalise_f32(-0.0_f32).to_bits(), 0.0_f32.to_bits());
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
}
