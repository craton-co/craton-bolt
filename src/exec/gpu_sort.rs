// SPDX-License-Identifier: Apache-2.0

//! GPU-side ORDER BY: upload a key column, bitonic-sort on the device, gather
//! every other column on the host.
//!
//! Pairs with [`crate::jit::sort_kernel`], which emits the PTX. The flow:
//!
//! ```text
//!  key column (host, n_rows)
//!     │
//!     ├─ pad to n_pow2 with sentinel  ───►  keys_pow2 (host)
//!     ├─ identity 0..n_pow2 indices   ───►  idx_pow2  (host)
//!     │
//!     ▼ h2d
//!  keys_pow2 (device, GpuVec<T>) + idx_pow2 (device, GpuVec<u32>)
//!     │
//!     ▼  log2(n_pow2) * (log2(n_pow2)+1) / 2 kernel launches
//!  keys_sorted + idx_sorted (device)
//!     │
//!     ▼ d2h indices, drop padded suffix
//!  permutation: Vec<u32> of length n_rows
//!     │
//!     ▼ arrow::compute::take per column
//!  sorted RecordBatch
//! ```
//!
//! ## Scope (Stage 1 / 2 / 3 / 4)
//!
//! - Sort keys: up to [`MAX_SORT_KEYS`] (the soft cap; the real ceiling is
//!   the sm_70 register budget, validated by `compile_sort_kernel_spec`).
//!   Single-key sorts are a [`SortKernelSpec`] with `keys.len() == 1`.
//! - Dtype: Int32, Int64, Float32, Float64, Bool, Utf8,
//!   Dictionary(Int32|Int64, Utf8). Utf8 (Stage 4) routes through an
//!   inline dictionary builder; Dictionary(_, Utf8) reads the dictionary's
//!   index column.
//! - ASC and DESC, per-key. NULLs handled by a per-key validity bitmap +
//!   `nulls_first` flag (Stage 2).
//! - Padded-row routing via an explicit `is_padded` bitmap (Stage 3) so
//!   real values colliding with the sentinel survive the truncation step.
//! - `n_rows <= u32::MAX` and the padded `n_pow2` must fit too (so practical
//!   limit is `n_rows <= 2^31` since `n_pow2 = next_pow2(n_rows) <= 2^32`).
//!
//! Stage 4 also retired the legacy Stage-1 single-key entry points
//! (`sort_indices_on_gpu`, `sort_record_batch_on_gpu`, `run_bitonic_passes`)
//! and the matching PTX module (`compile_sort_kernel`). The multi-key
//! driver subsumes them — single-key sorts are now expressed as a
//! `SortKernelSpec` with one entry.
//!
//! ## Scope (Stage 5)
//!
//! - Adaptive Utf8 gate: sample the column for distinct-value density and
//!   abort the GPU path when the inline dictionary builder would do more
//!   work than the host sort. See [`HIGH_CARDINALITY_THRESHOLD`] for the
//!   threshold and [`host_values_for_key`]'s Utf8 arm for the sampler.
//!
//! ## Padding strategy
//!
//! Bitonic sort requires `n_pow2 = 2^k` elements. We pad with a sentinel that
//! makes the padded entries land at the **end** of the sort result, so we can
//! truncate them off after gathering indices:
//!
//! - ASC : pad with `+INF`-equivalent (`i32::MAX`, `i64::MAX`,
//!   `f32::INFINITY`, `f64::INFINITY`).
//! - DESC: pad with `-INF`-equivalent (`i32::MIN`, `i64::MIN`,
//!   `f32::NEG_INFINITY`, `f64::NEG_INFINITY`).
//!
//! Real-data ties with the sentinel value are not a correctness issue: the
//! padded indices (>= n_rows) are filtered out by the final truncation step,
//! never returned to the caller.
//!
//! ## Stability (EXEC-H1)
//!
//! The bitonic sort is **stable**: the comparator breaks ties on the original
//! row index (see `crate::jit::sort_kernel` — the index is an implicit final
//! ASCending key, applied for both ASC and DESC key orders so equal-key rows
//! keep their input order). This matches the host fallback
//! `arrow::compute::lexsort_to_indices` in `crate::exec::sort`, so an
//! `ORDER BY non_unique_key [LIMIT k]` returns the same rows in the same order
//! regardless of whether the GPU or host path runs and regardless of input
//! size.

use std::collections::HashSet;
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;

use arrow_array::{
    Array, ArrayRef, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array,
    Int64Array, RecordBatch, StringArray, UInt32Array,
};
use arrow_array::types::{Int32Type, Int64Type};
use arrow::compute::take;
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::cuda::cuda_sys::{self, CUdeviceptr, CUgraphExec};
use crate::cuda::GpuVec;
use crate::cuda::PinnedHostBuffer;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;
use crate::exec::module_cache;
use crate::exec::n_rows_to_u32;
use crate::jit::sort_kernel::{
    compile_sort_kernel_spec, sort_kernel_entry_spec, KeyDesc, SortDirection,
    SortKernelSpec, SortLayout, MAX_SORT_KEYS, SORT_BLOCK_SIZE,
};
use crate::jit::sort_kernel_radix::{
    compile_radix_histogram, compile_radix_scatter_with_indices,
    radix_histogram_entry, radix_scatter_with_indices_entry, radix_steps_for,
    radix_supports_dtype, RADIX_BLOCK_SIZE, RADIX_BUCKETS,
};
use crate::plan::logical_plan::DataType;
use crate::plan::physical_plan::{RadixSortKernelSpec, RadixSortPass};

// Entry-point name constants for the radix kernels we wire through the
// `RadixSortKernelSpec` cache layer. They must match the strings emitted by
// `radix_histogram_entry(dtype)` / `radix_scatter_with_indices_entry(dtype)`
// in `crate::jit::sort_kernel_radix` — kept as `&'static str` consts here
// so we can hand them to `get_or_build_module_for_radix_sort` (whose
// `entry: &'static str` participates in the cache key). Splitting the
// keys-only and with-indices scatter into separate constants pins the
// "distinct cache slots" invariant for the two scatter ABIs at compile
// time; if someone re-points one of these to the wrong PTX entry, the
// `module.function(...)` lookup further down panics rather than silently
// loading the wrong kernel.
const RADIX_HISTOGRAM_I32_ENTRY: &str = "bolt_radix_histogram_i32";
const RADIX_HISTOGRAM_I64_ENTRY: &str = "bolt_radix_histogram_i64";
const RADIX_SCATTER_WI_I32_ENTRY: &str = "bolt_radix_scatter_i32_with_indices";
const RADIX_SCATTER_WI_I64_ENTRY: &str = "bolt_radix_scatter_i64_with_indices";

/// Compute `next_power_of_two(n)` returning `Err` if the result would overflow
/// `u32` (i.e. `n > 2^31`). `n == 0` returns `1` — bitonic sort needs at least
/// one launchable element, and the truncation step drops the padded entry.
fn next_pow2_u32(n: usize) -> BoltResult<u32> {
    if n == 0 {
        return Ok(1);
    }
    // Strict: n_pow2 must fit in u32 because the kernel takes a u32 n_pow2.
    let n_u64 = n as u64;
    if n_u64 > (u32::MAX as u64) {
        return Err(BoltError::Other(format!(
            "gpu_sort: n_rows {} exceeds u32::MAX",
            n
        )));
    }
    // next_power_of_two on u32 saturates; check the bound BEFORE calling.
    let n_u32 = n as u32;
    if n_u32 > (1u32 << 31) {
        return Err(BoltError::Other(format!(
            "gpu_sort: n_rows {} exceeds 2^31 — bitonic padding would overflow u32",
            n
        )));
    }
    Ok(n_u32.next_power_of_two())
}

/// log2(n) for an exact power of two. Panics if `n` is not a power of two —
/// only called on values that came out of [`next_pow2_u32`].
fn log2_pow2(n: u32) -> u32 {
    debug_assert!(n.is_power_of_two(), "log2_pow2 requires a power of two");
    n.trailing_zeros()
}

/// Pad `values` to `n_pow2` entries by repeating `sentinel`. Returns a Vec
/// of length `n_pow2`. The original entries occupy positions `0..n_rows`; the
/// padded entries occupy `n_rows..n_pow2`.
fn pad_to_pow2<T: Copy>(values: &[T], n_pow2: usize, sentinel: T) -> Vec<T> {
    let mut out = Vec::with_capacity(n_pow2);
    out.extend_from_slice(values);
    out.resize(n_pow2, sentinel);
    out
}

/// Arrow `DataType` -> our internal `DataType`. Returns `None` if the column
/// type isn't one of the GPU-sortable kinds (the caller falls through to
/// the host-side sort).
///
/// **Stage 3** additions:
///   - `Boolean` -> `Bool` (loaded as u8, compared as s32 0/1).
///   - `Dictionary(I32 | I64, Utf8)` -> the index dtype (Int32 / Int64). The
///     raw dictionary *indices* are NOT usable as sort keys: Arrow assigns
///     them in first-seen / insertion order, so a numeric sort of the raw
///     indices does not reproduce the lexicographic string order. The dict
///     path therefore remaps each row's index to the *lex rank* of its
///     string value before sorting (see `build_dict_lex_rank_indices`),
///     mirroring the plain-Utf8 path's `build_inline_dict_indices`.
///
/// **Stage 4** addition:
///   - `Utf8` -> `Int32`. The GPU sort path now builds an inline dictionary
///     on the fly inside `host_values_for_key` and feeds the i32 numeric
///     kernel with the per-row dictionary indices. After the sort the
///     original `StringArray` is gathered via `arrow::compute::take` like any
///     other column, so the output schema is unchanged.
pub fn arrow_dtype_to_internal(d: &arrow_schema::DataType) -> Option<DataType> {
    use arrow_schema::DataType as A;
    match d {
        A::Int32 => Some(DataType::Int32),
        A::Int64 => Some(DataType::Int64),
        A::Float32 => Some(DataType::Float32),
        A::Float64 => Some(DataType::Float64),
        A::Boolean => Some(DataType::Bool),
        // Stage 4: plain `Utf8` flows through the inline-dictionary builder
        // in `host_values_for_key` and ends up driving the i32 numeric
        // kernel. Cost note: O(n) hash + alloc per *distinct* string; for
        // very-low-cardinality columns this is a clear win, for
        // high-cardinality (every row unique) the dict-build is the
        // dominant cost and a Stage-5 path could detect that and skip the
        // GPU. See `host_values_for_key`'s Utf8 arm for the full comment.
        A::Utf8 => Some(DataType::Int32),
        A::Dictionary(key_ty, value_ty) => {
            // Only string-valued dictionaries are accepted for the Stage 3
            // adapter. The numeric values would already match one of the
            // direct dtypes above, no need for the dict path.
            if !matches!(value_ty.as_ref(), A::Utf8) {
                return None;
            }
            match key_ty.as_ref() {
                A::Int32 => Some(DataType::Int32),
                A::Int64 => Some(DataType::Int64),
                _ => None,
            }
        }
        _ => None,
    }
}

// =============================================================================
// Stage 2: multi-key + NULL-aware + shmem-variant host driver.
// =============================================================================

/// Build an Arrow-format packed-bit validity bitmap (1 byte per 8 elements,
/// LSB-first) covering positions `0..n_pow2`. Positions in `0..n_rows` come
/// from `arr.is_null(i)`; padded positions are marked VALID so that NULL
/// handling stays orthogonal to padded-row routing (Stage 3 split: NULL
/// semantics now drive `nulls_first` only, padding semantics drive
/// `is_padded`).
fn build_validity_padded(arr: &dyn Array, n_pow2: usize) -> Vec<u8> {
    let n_rows = arr.len();
    let bytes = (n_pow2 + 7) / 8;
    let mut out = vec![0u8; bytes];
    for i in 0..n_rows {
        if !arr.is_null(i) {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    // Stage 3: padded positions are marked VALID. Their fate is decided by
    // the is_padded bitmap (which routes them to the global end regardless
    // of value or null). Marking them valid means a NULLS FIRST query
    // doesn't accidentally lump padded rows in with real-NULL rows.
    for i in n_rows..n_pow2 {
        out[i / 8] |= 1 << (i % 8);
    }
    out
}

/// **Stage 3** — build the `is_padded` packed-bit bitmap. Bit `i` is 1 iff
/// `i >= n_rows`, i.e. the row at position `i` in the padded buffer is one
/// of the synthetic pad slots. The kernel uses this to route padded rows
/// past every real row regardless of sentinel-value collisions.
///
/// This is the load-bearing fix for the Stage-2 silent-row-drop bug: if a
/// real row's key equals the sentinel (e.g. `i32::MAX` as legit data with
/// the ASC `+INF`-style padding), the value compare ties and previously
/// the real row could end up at an index >= n_rows and get truncated. With
/// the explicit padded-bit, padded rows always lose the tiebreak.
fn build_is_padded(n_rows: usize, n_pow2: usize) -> Vec<u8> {
    let bytes = (n_pow2 + 7) / 8;
    let mut out = vec![0u8; bytes];
    for i in n_rows..n_pow2 {
        out[i / 8] |= 1 << (i % 8);
    }
    out
}

/// **Stage 5** — sample-and-fallback gate for the plain-Utf8 path. When the
/// fraction of distinct values in a uniform sample of the column exceeds
/// this threshold we abort the GPU path and let the host `lexsort_to_indices`
/// handle the sort.
///
/// ## Rationale
///
/// The Stage-4 inline-dictionary builder costs O(n) hash probes + O(d) string
/// clones + O(d log d) lex remap, where `d` is the number of distinct
/// values. When `d ≈ n` (every row unique) the dict build itself dominates
/// the GPU launch + h2d/d2h round trip — and worse, it does so on the host
/// side serially, leaving the device idle. At a ~0.6 sample-distinct ratio,
/// the full dict will allocate ~60% as many strings as it rejects, so the
/// host build work dominates the GPU launch overhead.
///
/// 0.6 is a defensive guess, not a measured break-even. A real benchmark
/// would refine this — see Stage 6 follow-up. Tunable so callers / tests
/// can reason about the dispatch decision without observing it through a
/// counter.
pub(crate) const HIGH_CARDINALITY_THRESHOLD: f64 = 0.6;

/// **Stage 5** — sample size for the high-cardinality gate. We sample
/// `min(MAX, n_rows / STRIDE)` rows at uniform strides across the column,
/// then count distinct values in the sample. Bounded above by `1024` so the
/// sample cost stays trivial relative to the full dict build it gates.
pub(crate) const HIGH_CARDINALITY_SAMPLE_MAX: usize = 1024;

/// **Stage 5** — sampling stride divisor. We sample `n_rows / 16` rows,
/// capped at [`HIGH_CARDINALITY_SAMPLE_MAX`]. Picked so the sample covers
/// at least 6.25% of the column — enough to catch common-case low-cardinality
/// patterns (enum columns, country codes, status flags) without scanning the
/// whole column.
pub(crate) const HIGH_CARDINALITY_SAMPLE_STRIDE_DIV: usize = 16;

/// Sample `min(HIGH_CARDINALITY_SAMPLE_MAX, n_rows / HIGH_CARDINALITY_SAMPLE_STRIDE_DIV)`
/// rows from `sa` at uniform strides, returning `(sample_size, distinct_in_sample)`.
/// NULL rows are sampled but contribute a single distinct slot (their value
/// byte is not compared — they all collapse into one "NULL" bucket). For
/// very small `n_rows` (smaller than `HIGH_CARDINALITY_SAMPLE_STRIDE_DIV`)
/// the `.max(1)` clamp shrinks the target to a single sample — the GPU
/// path's row-count gate (`GPU_SORT_MIN_ROWS` in `crate::exec::sort`)
/// already rules out small columns, so this branch is effectively
/// unreachable from the executor; the clamp is defensive only.
fn sample_distinct_utf8(sa: &StringArray) -> (usize, usize) {
    let n = sa.len();
    if n == 0 {
        return (0, 0);
    }
    let target = (n / HIGH_CARDINALITY_SAMPLE_STRIDE_DIV)
        .max(1)
        .min(HIGH_CARDINALITY_SAMPLE_MAX);
    // Stride: distribute `target` samples across `n` rows. `step` is at least
    // 1 (target <= n implied by the .min(n) clamp in the caller — but be
    // defensive in case `target == 0`).
    let step = (n / target).max(1);
    let mut seen: HashSet<&str> = HashSet::with_capacity(target);
    let mut had_null = false;
    let mut sample_size = 0usize;
    let mut i = 0usize;
    while i < n && sample_size < target {
        if sa.is_null(i) {
            had_null = true;
        } else {
            seen.insert(sa.value(i));
        }
        sample_size += 1;
        i += step;
    }
    let distinct = seen.len() + if had_null { 1 } else { 0 };
    (sample_size, distinct)
}

/// Build per-row dictionary indices for a plain `StringArray` such that the
/// integer order of the indices matches the lexicographic order of the
/// underlying strings. Used by the Stage-4 plain-Utf8 GPU sort adapter.
///
/// Two-pass: first pass populates `dict_of_first_seen` (Vec<String>) and
/// `tmp_idx` (per-row index into `dict_of_first_seen`); second pass
/// computes the lex-order remap and rewrites each row's index. NULL rows
/// contribute index 0 (the same as any other value — the value byte is
/// don't-care because the validity bitmap controls NULL routing in the
/// kernel; the caller is responsible for setting `nullable: true` on the
/// `GpuSortKey` if the column has nulls so the kernel reads validity).
///
/// Returns a `Vec<i32>` of length `sa.len()`.
fn build_inline_dict_indices(sa: &StringArray) -> Vec<i32> {
    let n = sa.len();
    // Pass 1: gather distinct strings in first-seen order, record each row's
    // first-seen index.
    //
    // We keep the dict as `Vec<String>` so the keys outlive the borrow on
    // `sa.value(i)` (which is a `&str` borrowed from the StringArray). The
    // HashMap interns strings via the index into `dict_seen`.
    let mut dict_seen: Vec<String> = Vec::new();
    let mut lookup: HashMap<String, i32> = HashMap::new();
    let mut tmp_idx: Vec<i32> = Vec::with_capacity(n);
    for i in 0..n {
        if sa.is_null(i) {
            tmp_idx.push(0);
            continue;
        }
        let v = sa.value(i);
        // Use a direct `get` then `insert` rather than `entry().or_insert`
        // so we only clone the string on the cold (new-value) path.
        if let Some(&idx) = lookup.get(v) {
            tmp_idx.push(idx);
        } else {
            let new_idx = dict_seen.len() as i32;
            dict_seen.push(v.to_string());
            lookup.insert(v.to_string(), new_idx);
            tmp_idx.push(new_idx);
        }
    }
    // Pass 2: compute the lex-order remap. `order[k]` = the rank of
    // `dict_seen[k]` in lex order. After the remap, sorting the i32
    // indices ASCending is equivalent to sorting the strings ASC.
    let d = dict_seen.len();
    let mut perm: Vec<usize> = (0..d).collect();
    perm.sort_by(|&a, &b| dict_seen[a].cmp(&dict_seen[b]));
    let mut remap = vec![0i32; d];
    for (rank, original) in perm.into_iter().enumerate() {
        remap[original] = rank as i32;
    }
    // Apply remap. NULL rows kept their `0` placeholder above — the kernel's
    // validity bitmap (built separately by `build_validity_padded`) will
    // route NULLs by the `nulls_first` flag, so the value byte is unused
    // for those rows.
    for ix in tmp_idx.iter_mut() {
        if !dict_seen.is_empty() {
            // Guard against the all-null case (d==0): tmp_idx is all 0s and
            // there's nothing to remap.
            *ix = remap[*ix as usize];
        }
    }
    tmp_idx
}

/// Build per-row **lex-rank** indices for a pre-encoded `DictionaryArray<K>`
/// whose values are a `StringArray`. Mirrors [`build_inline_dict_indices`]
/// but consumes an existing Arrow dictionary instead of building one from a
/// plain `StringArray`.
///
/// Arrow dictionary keys are assigned in *first-seen* / insertion order, so
/// the raw key column does NOT induce lexicographic order over the strings
/// (dict `["b","a","c"]` → keys b=0,a=1,c=2, and a numeric ASC sort of the
/// keys yields b,a,c — not the lex order a,b,c). This remaps every row's raw
/// dict key to the *lex rank* of its string value, so feeding the remapped
/// indices to the numeric kernel makes a numeric ASC sort == lexicographic
/// string ASC sort.
///
/// NULL rows contribute the placeholder `0` (exactly as
/// [`build_inline_dict_indices`] does): the value byte is don't-care because
/// the kernel's validity bitmap — built separately by `build_validity_padded`
/// — routes NULLs by the `nulls_first` flag. We do NOT collapse nulls into
/// lex-rank 0, which would corrupt the nulls-first/last contract; the
/// placeholder is only read for non-null rows.
///
/// Returns a `Vec<i32>` of length `da.len()`. i32 ranks are sufficient even
/// for an `Int64`-keyed dictionary: the rank space is bounded by the number
/// of *distinct* dictionary values, which is itself bounded by `n_rows`, and
/// the GPU sort caps `n_rows <= 2^31`.
fn build_dict_lex_rank_indices<K>(da: &DictionaryArray<K>) -> BoltResult<Vec<i32>>
where
    K: arrow_array::types::ArrowDictionaryKeyType,
{
    let values = da
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| {
            BoltError::Other("gpu_sort: dictionary values are not Utf8".into())
        })?;

    // Lex-rank remap over the distinct dictionary values. `remap[k]` is the
    // rank of `values[k]` in lex order, so a numeric ASC sort of the ranks
    // reproduces the lexicographic string order. A dictionary value slot can
    // itself be null (rare, but legal); such slots are unreachable for any
    // non-null row's key, so their rank is immaterial — we still assign one
    // to keep the table dense.
    let d = values.len();
    let mut perm: Vec<usize> = (0..d).collect();
    perm.sort_by(|&a, &b| values.value(a).cmp(values.value(b)));
    let mut remap = vec![0i32; d];
    for (rank, original) in perm.into_iter().enumerate() {
        remap[original] = rank as i32;
    }

    // Map each row's raw dict key to its value's lex rank. `key(i)` returns
    // `None` for a null row (placeholder 0, routed by the validity bitmap,
    // not the value) and `Some(k)` with the dictionary slot for a valid row.
    let n = da.len();
    let mut out: Vec<i32> = Vec::with_capacity(n);
    for i in 0..n {
        match da.key(i) {
            Some(k) => out.push(remap[k]),
            None => out.push(0),
        }
    }
    Ok(out)
}

/// Extract a numeric "host view" from a sortable Arrow column. Stage 3
/// addition: handles Bool (-> u8 0/1 widened to i32) and dictionary-encoded
/// Utf8 (-> index column as i32 or i64). Stage 4 addition: handles plain
/// Utf8 by building an inline dictionary on the fly (see
/// `build_inline_dict_indices`). For everything else this is a straight
/// `.values().to_vec()`.
///
/// **Stage 5** changes the signature to `BoltResult<Option<HostKeyValues>>`.
/// `Ok(None)` is the "fall through to the host sort" signal: today it fires
/// only on the high-cardinality plain-Utf8 path (see
/// [`HIGH_CARDINALITY_THRESHOLD`]), but the option is the natural extension
/// point for any future per-column gate that wants to abort the GPU launch
/// without erroring (e.g. an mm-detector for runs already in sort order).
fn host_values_for_key(arr: &dyn Array, dtype: DataType) -> BoltResult<Option<HostKeyValues>> {
    use arrow_schema::DataType as A;
    Ok(Some(match (dtype, arr.data_type()) {
        (DataType::Int32, A::Int32) => HostKeyValues::I32(
            arr.as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| BoltError::Other("gpu_sort: i32 downcast failed".into()))?
                .values()
                .as_ref()
                .to_vec(),
        ),
        (DataType::Int64, A::Int64) => HostKeyValues::I64(
            arr.as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| BoltError::Other("gpu_sort: i64 downcast failed".into()))?
                .values()
                .as_ref()
                .to_vec(),
        ),
        (DataType::Float32, _) => HostKeyValues::F32(
            arr.as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| BoltError::Other("gpu_sort: f32 downcast failed".into()))?
                .values()
                .as_ref()
                .to_vec(),
        ),
        (DataType::Float64, _) => HostKeyValues::F64(
            arr.as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| BoltError::Other("gpu_sort: f64 downcast failed".into()))?
                .values()
                .as_ref()
                .to_vec(),
        ),
        (DataType::Bool, A::Boolean) => {
            // Widen each bit to a u8 of 0/1; the kernel loads via ld.global.u8
            // into a b32 register and compares as s32. Length matches `arr.len()`.
            let ba = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| BoltError::Other("gpu_sort: bool downcast failed".into()))?;
            let mut out: Vec<u8> = Vec::with_capacity(ba.len());
            for i in 0..ba.len() {
                // is_null rows still contribute a value byte; the validity
                // bitmap is the source of truth — pick 0 as a no-op.
                if ba.is_null(i) {
                    out.push(0);
                } else {
                    out.push(if ba.value(i) { 1 } else { 0 });
                }
            }
            HostKeyValues::Bool(out)
        }
        // Stage 4 plain-Utf8 adapter: build an inline dictionary on the fly
        // (HashMap<&str, i32> + Vec<String>), assign each row its dict
        // index, and feed those indices through the i32 numeric kernel.
        // Because the dictionary is constructed in *first-seen* order the
        // resulting integer order is not the lexicographic order of the
        // strings — so before returning we *remap* the indices so that
        // `idx[i] < idx[j]` iff `s[i] < s[j]` lex (this is what makes a
        // numeric ASC sort produce a Utf8 ASC sort). The remap is a single
        // sort over the distinct values: O(d log d) where d = #distinct.
        //
        // Cost model (host-side, per call):
        //   - O(n) hash probes      (each row looked up + maybe inserted)
        //   - O(d) string clones    (one per distinct value, into the dict)
        //   - O(d log d) remap sort (lex order over the distinct strings)
        //   - O(n) i32 writes       (the final per-row index buffer)
        //
        // Low-cardinality columns (e.g. a country code, an enum) are a clear
        // win: d << n keeps the dict build at near-O(n) and the i32 kernel
        // crushes the sort. High-cardinality columns (d ≈ n, every row
        // unique) push the dict build itself to O(n log n) — at that point
        // the host's `lexsort_to_indices` is competitive.
        //
        // **Stage 5** wires this gate: we sample the column up-front (see
        // `sample_distinct_utf8`) and return `Ok(None)` when the distinct-
        // value density exceeds `HIGH_CARDINALITY_THRESHOLD`. The caller
        // (`sort_indices_on_gpu_multi`) propagates the None up to
        // `try_gpu_sort`, which then falls through to the host's
        // `lexsort_to_indices`. No partial allocation: the sampler stops
        // before any string is cloned into the inline dictionary.
        //
        // The Utf8 column doesn't reach the kernel — only the indices do —
        // so the final `take` over the original StringArray (in
        // `sort_record_batch_on_gpu_multi`) preserves the column's storage
        // exactly. No re-encoding into a dictionary array.
        (DataType::Int32, A::Utf8) => {
            let sa = arr
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| BoltError::Other("gpu_sort: utf8 downcast failed".into()))?;
            // Stage 5 high-cardinality gate. Sample first; if the column is
            // too distinct-heavy, abort the GPU path entirely.
            let (sample_size, distinct) = sample_distinct_utf8(sa);
            if sample_size > 0 {
                let ratio = (distinct as f64) / (sample_size as f64);
                if ratio > HIGH_CARDINALITY_THRESHOLD {
                    return Ok(None);
                }
            }
            HostKeyValues::I32(build_inline_dict_indices(sa))
        }
        // Stage 3 dictionary-Utf8 adapter: the dictionary's *values* never
        // reach the GPU sort, but the raw index column cannot be fed directly
        // — Arrow dictionary keys are assigned in first-seen / insertion
        // order, so a numeric sort of the raw keys does NOT reproduce the
        // lexicographic string order (dict `["b","a","c"]` → keys b=0,a=1,c=2,
        // numeric ASC → b,a,c, not lex a,b,c). We therefore remap every row's
        // key to the *lex rank* of its string value (see
        // `build_dict_lex_rank_indices`), exactly as the plain-Utf8 path does
        // via `build_inline_dict_indices`. The output permutation is then
        // applied (host-side) to the dictionary-encoded column intact, which
        // keeps the values-dictionary edge alive without re-encoding.
        (DataType::Int32, A::Dictionary(key_ty, _)) if matches!(key_ty.as_ref(), A::Int32) => {
            let da = arr
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_sort: dict<i32,utf8> downcast failed".into())
                })?;
            HostKeyValues::I32(build_dict_lex_rank_indices(da)?)
        }
        (DataType::Int64, A::Dictionary(key_ty, _)) if matches!(key_ty.as_ref(), A::Int64) => {
            let da = arr
                .as_any()
                .downcast_ref::<DictionaryArray<Int64Type>>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_sort: dict<i64,utf8> downcast failed".into())
                })?;
            // Lex ranks fit in i32 (bounded by #distinct <= n_rows <= 2^31),
            // but the declared key dtype here is Int64 — the kernel loads
            // 8-byte keys and `upload_padded_key_for_dtype` pads with an i64
            // sentinel — so widen the i32 ranks to i64 to keep the device
            // buffer width matching `k.dtype`.
            let ranks = build_dict_lex_rank_indices(da)?;
            HostKeyValues::I64(ranks.into_iter().map(|r| r as i64).collect())
        }
        (dt, arrow_dt) => {
            return Err(BoltError::Other(format!(
                "gpu_sort: dtype/array mismatch ({:?} vs Arrow {:?})",
                dt, arrow_dt
            )))
        }
    }))
}

/// Heterogeneous host-side key buffer pre-upload. Existed inline before
/// Stage 3; now factored out so the Bool + Dict-Utf8 adapters can build it
/// without re-implementing the per-dtype branches twice.
enum HostKeyValues {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<u8>),
}

/// Upload a typed key column into a padded device buffer. Sentinel choice
/// follows the Stage-1 convention (`+INF` for ASC, `-INF` for DESC) so
/// padded entries land past real data in the comparator's value sense.
///
/// **Stage 3** — the padded-row routing in the kernel uses an explicit
/// is_padded bitmap (built separately, see `build_is_padded`), so the
/// sentinel choice is only a "soft hint" that still helps real data
/// converge faster (sentinel still beats real values on average). When a
/// real value legitimately ties the sentinel, the is_padded bit wins the
/// tiebreak — no row drop.
fn upload_padded_key_for_dtype(
    arr: &dyn Array,
    dtype: DataType,
    dir: SortDirection,
    n_pow2: usize,
) -> BoltResult<Option<KeyDeviceBuf>> {
    let values = match host_values_for_key(arr, dtype)? {
        Some(v) => v,
        // Stage 5: `host_values_for_key` returned the fall-through signal
        // (e.g. plain Utf8 sample looks too distinct-heavy). Propagate.
        None => return Ok(None),
    };
    Ok(Some(match values {
        HostKeyValues::I32(host) => {
            let sentinel = match dir {
                SortDirection::Asc => i32::MAX,
                SortDirection::Desc => i32::MIN,
            };
            let padded = pad_to_pow2(&host, n_pow2, sentinel);
            KeyDeviceBuf::I32(GpuVec::<i32>::from_slice(&padded)?)
        }
        HostKeyValues::I64(host) => {
            let sentinel = match dir {
                SortDirection::Asc => i64::MAX,
                SortDirection::Desc => i64::MIN,
            };
            let padded = pad_to_pow2(&host, n_pow2, sentinel);
            KeyDeviceBuf::I64(GpuVec::<i64>::from_slice(&padded)?)
        }
        HostKeyValues::F32(host) => {
            let sentinel = match dir {
                SortDirection::Asc => f32::INFINITY,
                SortDirection::Desc => f32::NEG_INFINITY,
            };
            let padded = pad_to_pow2(&host, n_pow2, sentinel);
            KeyDeviceBuf::F32(GpuVec::<f32>::from_slice(&padded)?)
        }
        HostKeyValues::F64(host) => {
            let sentinel = match dir {
                SortDirection::Asc => f64::INFINITY,
                SortDirection::Desc => f64::NEG_INFINITY,
            };
            let padded = pad_to_pow2(&host, n_pow2, sentinel);
            KeyDeviceBuf::F64(GpuVec::<f64>::from_slice(&padded)?)
        }
        HostKeyValues::Bool(host) => {
            // Bool uses 1-byte slots on device; sentinel 1=true for ASC pad
            // (lands trues at the end), 0 for DESC. Real-tie collisions are
            // still solved by is_padded bitmap routing.
            let sentinel: u8 = match dir {
                SortDirection::Asc => 1,
                SortDirection::Desc => 0,
            };
            let padded = pad_to_pow2(&host, n_pow2, sentinel);
            KeyDeviceBuf::Bool(GpuVec::<u8>::from_slice(&padded)?)
        }
    }))
}

/// Type-erased wrapper around a GpuVec of the key dtype. Lets the multi-key
/// driver hold heterogeneous key buffers in a single Vec without unsafe
/// dyn-dispatch tricks.
///
/// **Stage 3** adds the `Bool` variant: a `u8`-typed buffer the kernel reads
/// with `ld.global.u8` into a b32 register.
enum KeyDeviceBuf {
    I32(GpuVec<i32>),
    I64(GpuVec<i64>),
    F32(GpuVec<f32>),
    F64(GpuVec<f64>),
    Bool(GpuVec<u8>),
}

impl KeyDeviceBuf {
    fn device_ptr(&self) -> CUdeviceptr {
        match self {
            KeyDeviceBuf::I32(v) => v.device_ptr(),
            KeyDeviceBuf::I64(v) => v.device_ptr(),
            KeyDeviceBuf::F32(v) => v.device_ptr(),
            KeyDeviceBuf::F64(v) => v.device_ptr(),
            KeyDeviceBuf::Bool(v) => v.device_ptr(),
        }
    }
}

/// One key column ready to feed the multi-key sort kernel.
pub struct GpuSortKey<'a> {
    /// Underlying Arrow column.
    pub column: &'a dyn Array,
    /// Engine-internal dtype (must be one of the GPU-sortable set).
    pub dtype: DataType,
    /// Per-key direction.
    pub direction: SortDirection,
    /// Per-key NULLS placement.
    pub nulls_first: bool,
}

// =============================================================================
// v0.7: GPU radix-sort path. Single-key (#15) and multi-key + DESC (#19).
// =============================================================================
//
// Pairs with [`crate::jit::sort_kernel_radix`]. The single-key flow:
//
// ```text
//  key column (host, n_rows)
//     │
//     ▼ host-side direction transform (XOR signed-MSB for ASC, bit-not for DESC)
//     ▼ h2d (no padding — radix has no power-of-two requirement)
//  keys_ping (device)        idx_ping (device, 0..n_rows or starting-perm)
//     │                            │
//     ▼                            ▼
//  for pass in 0..radix_steps:
//      zero hist (16 u32)
//      histogram kernel (reads keys_ping, atom-adds hist[digit])
//      D2H hist → host exclusive scan → H2D as offsets
//      scatter_with_indices kernel
//          (keys_ping, idx_ping)  →  (keys_pong, idx_pong)
//      swap ping ↔ pong
//
//     ▼ d2h idx_ping
//  permutation: Vec<u32> of length n_rows
// ```
//
// ## Multi-key extension (#19)
//
// Radix is LSD-stable, so an N-key lex sort is `N` sequential single-key
// passes from the LAST key back to the FIRST. Each pass:
//   1. Gather the new key's host values via the running permutation
//      (`gathered[i] = key_M.host[ running_perm[i] ]`), applying the
//      per-key direction transform during the gather.
//   2. Upload as the fresh `keys_ping` buffer.
//   3. Seed `idx_ping = running_perm` (identity for the first key).
//   4. Run radix passes; the kernel carries `idx_ping` in lock-step with
//      the keys, so the final `idx_ping` is the NEW running permutation.
//
// ## DESC strategy (#19)
//
// We pre-transform the host keys per-direction so the device kernel
// always does an unsigned ASC bit-pattern sort:
//   * **ASC, signed**: XOR with the dtype's MSB constant (`0x80000000`
//     for i32 / `0x8000000000000000` for i64) — makes signed bit-patterns
//     unsigned-ordered.
//   * **DESC, signed**: XOR with all-ones (i.e. `!val`) — inverts the
//     bit-pattern order so the unsigned ASC radix sort produces the
//     value-DESC order. Mixes cleanly with the MSB-flip's algebra
//     (`!v = v ^ 0xFF…F`, both are XOR isomorphisms).
//
// The kernel-side MSB-flip (`bolt_radix_msb_flip_<dty>`) stays available
// but is no longer called from this path; the per-key host transform
// during gather is strictly cheaper than a one-shot kernel pre-pass and
// is the only point where DESC's bit-not can be applied per-key in a
// multi-key chain.
//
// ## Scope
//
// - **Up to [`MAX_SORT_KEYS`] keys.** Each key Int32 or Int64 — float
//   radix needs the IEEE-monotonic transform (deferred); Bool / Utf8
//   fall through. Mixed ASC/DESC per-key is supported.
// - **No NULLs in any key column.** Radix sort has no validity-bitmap
//   routing; we reject nullable columns up front.
// - **Env-gated.** `BOLT_GPU_SORT=1` opts in; default OFF.
//
// The dispatch decision in [`try_gpu_sort_radix`] returns `Ok(None)` on
// any precondition miss so the caller can fall through to the bitonic
// path or the host `lexsort_to_indices`.

/// Test-side dispatch counter. Bumps once per successful predicate match —
/// i.e. every time the executor decides to take the radix path for a
/// given sort. Production callers ignore it; the gpu_sort_radix tests
/// observe dispatch by comparing the counter delta.
///
/// We use this instead of trying to instrument launches under `cuda-stub`
/// because the predicate check happens *before* any kernel touches the
/// device, so a pure counter is the cheapest observability hook and the
/// only one that can fire under stub builds.
#[doc(hidden)]
pub(crate) static RADIX_DISPATCH_COUNT: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// **Pure** predicate: does this single-key `(arr, dtype, direction, nulls_first)`
/// tuple satisfy every gate the radix sort needs?
///
/// Splitting the predicate out of [`sort_indices_on_gpu_radix`] gives the
/// `sort.rs` dispatcher a host-only check (no GPU touch, no allocation)
/// for routing decisions, AND lets the unit tests verify the gates
/// deterministically under `cuda-stub` without ever launching a kernel.
///
/// The gates, in evaluation order:
///   1. `radix_supports_dtype(dtype)` — currently Int32 / Int64 only.
///   2. **Direction**: ASC and DESC both accepted (#19; the multi-key
///      driver applies the direction transform host-side during gather).
///   3. `arr.null_count() == 0` — radix sort has no validity routing.
///   4. `arr.len() <= u32::MAX` — the indices payload is `u32`.
///
/// `nulls_first` is meaningless on a column with no NULLs (gate 3 already
/// guarantees that), so we accept any value for it. The gate is kept in the
/// signature so a caller that wants to pin the parameter explicitly can
/// continue to do so.
pub(crate) fn radix_dispatch_predicate(
    arr: &dyn Array,
    dtype: DataType,
    direction: SortDirection,
    nulls_first: bool,
) -> bool {
    if !radix_supports_dtype(dtype) {
        return false;
    }
    // #19: both ASC and DESC are handled by the host-side per-direction
    // pre-transform (signed-MSB XOR for ASC, bit-not for DESC).
    let _ = direction;
    // No NULL routing in the radix kernel today.
    if arr.null_count() > 0 {
        return false;
    }
    // `nulls_first` is decoupled from routing (no NULLs by gate 3) — kept
    // in the parameter list so callers can still pass it through verbatim
    // from the `SortExpr`.
    let _ = nulls_first;
    // u32 indices payload.
    if arr.len() > (u32::MAX as usize) {
        return false;
    }
    true
}

/// **Pure** predicate for a multi-key radix dispatch.
///
/// Returns true iff every key independently passes
/// [`radix_dispatch_predicate`] AND the key list is non-empty and within
/// `MAX_SORT_KEYS`. Used by [`try_gpu_sort_radix`] in `sort.rs` to gate
/// the multi-key path without touching the GPU.
///
/// Each key's `direction` and `nulls_first` are evaluated independently,
/// so mixed ASC/DESC across keys is fully supported.
pub(crate) fn radix_dispatch_predicate_multi(keys: &[GpuSortKey<'_>]) -> bool {
    if keys.is_empty() {
        return false;
    }
    if keys.len() > MAX_SORT_KEYS {
        return false;
    }
    // All keys must agree on row count — the per-key predicate covers the
    // per-column dtype/null gates, but row-count equality is a multi-key
    // invariant we enforce here.
    let n_rows = keys[0].column.len();
    for k in keys {
        if k.column.len() != n_rows {
            return false;
        }
        if !radix_dispatch_predicate(k.column, k.dtype, k.direction, k.nulls_first) {
            return false;
        }
    }
    true
}

/// GPU radix sort for a single Int32/Int64 key column.
///
/// Thin compatibility wrapper around [`sort_indices_on_gpu_radix_multi`]
/// — the single-key path is just a multi-key sort with a 1-element key
/// list, and unifying the two avoids two parallel drivers diverging.
///
/// Returns `Ok(Some(perm))` with the row permutation on success, or
/// `Ok(None)` if the input doesn't match the radix gate (caller falls
/// through to bitonic / host). Returns `Err` only on hard GPU failures
/// (OOM, kernel launch failure).
///
/// **#19 widening:** previously this path was ASC-only. The
/// [`radix_dispatch_predicate`] now accepts both ASC and DESC; the
/// direction transform is applied host-side inside
/// [`sort_indices_on_gpu_radix_multi`] (XOR signed-MSB for ASC, bit-not
/// for DESC).
///
/// Retained as a single-key convenience wrapper around
/// [`sort_indices_on_gpu_radix_multi`]: production call sites in
/// `src/exec/sort.rs` now drive the multi-key entry directly (it handles
/// the single-key case), but the round-trip unit tests in this module
/// exercise the wrapper to keep the narrow single-key contract pinned.
/// `#[allow(dead_code)]` because those callers are all `#[cfg(test)]`.
#[allow(dead_code)]
pub fn sort_indices_on_gpu_radix(
    arr: &dyn Array,
    dtype: DataType,
    direction: SortDirection,
    nulls_first: bool,
) -> BoltResult<Option<UInt32Array>> {
    if !radix_dispatch_predicate(arr, dtype, direction, nulls_first) {
        return Ok(None);
    }
    // Single-key sort is a 1-element multi-key sort. We don't bump the
    // dispatch counter here — `_multi` does, so we don't double-count.
    let key = GpuSortKey {
        column: arr,
        dtype,
        direction,
        nulls_first,
    };
    sort_indices_on_gpu_radix_multi(&[key])
}

/// GPU radix sort for an arbitrary lexicographic key list (up to
/// [`MAX_SORT_KEYS`] keys, mixed ASC/DESC fine).
///
/// The kernel side ([`crate::jit::sort_kernel_radix`]) emits the PTX
/// entry points per dtype:
///   - `bolt_radix_histogram_<dty>` — per-pass 4-bit digit bucketing.
///   - `bolt_radix_scatter_<dty>_with_indices` — per-pass scatter that
///     carries the u32 row-index payload in lock-step with the key.
///   - `bolt_radix_msb_flip_<dty>` — kept available for the existing
///     ABI but **no longer called from this path**. The direction-aware
///     host-side pre-transform (see [`radix_pre_transform_i32`] /
///     [`radix_pre_transform_i64`]) subsumes the signed-int MSB-flip and
///     adds the DESC bit-not in the same XOR — strictly cheaper than a
///     one-shot kernel pre-pass since we already touch every key on the
///     host during gather.
///
/// ## LSD multi-key driver
///
/// Radix is LSD-stable, so an N-key lex sort is N sequential single-key
/// passes from the LAST key back to the FIRST. Each per-key pass:
///   1. Gather the new key's host values via the running permutation
///      (`gathered[i] = host[key_M][running_perm[i]]`), applying the
///      per-key direction transform during the gather.
///   2. Upload as the fresh `keys_ping` buffer.
///   3. Seed `idx_ping = running_perm` (identity on the first key).
///   4. Run 8 (Int32) or 16 (Int64) radix passes. The kernel carries
///      `idx_ping` in lock-step with the keys; final `idx_ping` is the
///      NEW running permutation.
///
/// After every key is processed, the final permutation is the lexico-
/// graphic sort across all keys with each key's per-direction order.
pub fn sort_indices_on_gpu_radix_multi(
    keys: &[GpuSortKey<'_>],
) -> BoltResult<Option<UInt32Array>> {
    if !radix_dispatch_predicate_multi(keys) {
        return Ok(None);
    }
    // Bump the dispatch counter as soon as we commit to the radix path.
    // Tests observe this; production callers ignore it. We bump once per
    // sort decision, regardless of key count — the counter measures
    // *dispatch* events, not kernel launches.
    RADIX_DISPATCH_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

    let n_rows = keys[0].column.len();
    if n_rows == 0 {
        return Ok(Some(UInt32Array::from(Vec::<u32>::new())));
    }
    let n_rows_u32 = n_rows_to_u32(n_rows)?;

    // Extract every key column once into typed host vectors. We don't apply
    // the direction transform here — that happens inside the gather step,
    // because the gather order (via the running permutation) is what feeds
    // the next radix pass's keys_ping buffer.
    let mut host_keys: Vec<HostKeyValues> = Vec::with_capacity(keys.len());
    for k in keys {
        match host_values_for_key(k.column, k.dtype)? {
            Some(v) => host_keys.push(v),
            // Predicate gates dtype to Int32/Int64, so the Utf8 fall-through
            // signal from `host_values_for_key` can't fire — defensive None.
            None => return Ok(None),
        }
    }

    // Running permutation: identity at start, replaced with each per-key
    // pass's output. By the LSD invariant, after all N passes this is the
    // lex permutation across all N keys.
    let mut running_perm: Vec<u32> = (0..n_rows_u32).collect();

    // Iterate keys in REVERSE order — LSD radix wants the last key first,
    // so its order is the least significant in the final lex ordering.
    for (k, host) in keys.iter().zip(host_keys.iter()).rev() {
        running_perm = match host {
            HostKeyValues::I32(host_vals) => {
                run_radix_pipeline_i32(host_vals, &running_perm, n_rows_u32, k.dtype, k.direction)?
            }
            HostKeyValues::I64(host_vals) => {
                run_radix_pipeline_i64(host_vals, &running_perm, n_rows_u32, k.dtype, k.direction)?
            }
            // The predicate above gates dtype to Int32/Int64; other arms
            // are unreachable. Defensive return so a future predicate
            // widening doesn't crash the engine.
            _ => return Ok(None),
        };
    }

    Ok(Some(UInt32Array::from(running_perm)))
}

/// Host-side per-direction pre-transform for an i32 key.
///
/// The device kernel does an **unsigned** ASC bit-pattern sort over the
/// key buffer. To turn that into value-ASC over signed ints, we XOR the
/// MSB so the negative bit patterns sort below the positives. To turn it
/// into value-DESC, we XOR every bit (`!val`) so the bit-pattern order
/// flips. Both are XOR isomorphisms, so the transform is invertible if
/// we ever needed to recover the original values (we don't — only the
/// idx payload is read back).
#[inline]
fn radix_pre_transform_i32(val: i32, dir: SortDirection) -> i32 {
    match dir {
        // i32::MIN as i32 is 0x8000_0000 — the canonical signed-MSB
        // constant. `wrapping_neg` would also work for the magnitude but
        // XOR makes the bit-pattern intent explicit.
        SortDirection::Asc => val ^ i32::MIN,
        // DESC = ASC-then-bitwise-NOT. Bit-NOT alone is wrong for signed
        // ints because it doesn't normalise the MSB first — `!INT_MIN ==
        // INT_MAX` would sort *below* `!0 == -1` under u32 order, which
        // is value-ASC of the inputs, not value-DESC. The correct
        // transform is `!(val ^ MIN)` so unsigned-ASC of the result =
        // signed-DESC of `val`.
        SortDirection::Desc => !(val ^ i32::MIN),
    }
}

/// Host-side per-direction pre-transform for an i64 key. Same algebra as
/// [`radix_pre_transform_i32`], scaled to 64 bits.
#[inline]
fn radix_pre_transform_i64(val: i64, dir: SortDirection) -> i64 {
    match dir {
        SortDirection::Asc => val ^ i64::MIN,
        SortDirection::Desc => !(val ^ i64::MIN),
    }
}

/// IEEE-monotonic key transform for an `f32`, producing a `u32` bit-blob whose
/// **unsigned** ascending order equals the floating-point value order.
///
/// Radix sort compares keys as unsigned bit patterns, but IEEE-754 floats do
/// not sort that way directly: negative floats have the sign bit set (so they
/// would sort *after* positives), and within the negatives larger magnitudes
/// have larger bit patterns (so they would sort backwards). The standard fix
/// (Thrust / CUB `radix_sort`): if the sign bit is set, flip **every** bit;
/// otherwise flip **only** the sign bit. After this, unsigned ascending order
/// over the transformed bits is exactly value-ascending over the floats, with
/// `-0.0` and `+0.0` adjacent and NaNs sorting to one end.
///
/// `dir` composes exactly as the integer pre-transform does: DESC is the
/// bitwise-NOT of the ASC key, which inverts the unsigned order.
///
/// This is the float counterpart of [`radix_pre_transform_i32`]. It is unit
/// tested below; wiring it into the dispatch predicate is the remaining F2
/// step (see the `RadixFlavour::for_dtype` float arm in
/// `crate::jit::sort_kernel_radix`).
// Implemented + unit-tested but not yet called by the dispatch path (floats
// still fall back to bitonic/host); silence the unused warning until wired.
#[allow(dead_code)]
#[inline]
pub(crate) fn radix_float_key_f32(f: f32, dir: SortDirection) -> u32 {
    let b = f.to_bits();
    // Arithmetic shift of the sign bit gives 0xFFFF_FFFF for negatives, else 0;
    // OR-in the sign bit so positives still flip just their sign.
    let mask = ((b as i32) >> 31) as u32 | 0x8000_0000;
    let asc = b ^ mask;
    match dir {
        SortDirection::Asc => asc,
        SortDirection::Desc => !asc,
    }
}

/// IEEE-monotonic key transform for an `f64`. Same algebra as
/// [`radix_float_key_f32`], scaled to 64 bits.
#[allow(dead_code)]
#[inline]
pub(crate) fn radix_float_key_f64(f: f64, dir: SortDirection) -> u64 {
    let b = f.to_bits();
    let mask = ((b as i64) >> 63) as u64 | 0x8000_0000_0000_0000;
    let asc = b ^ mask;
    match dir {
        SortDirection::Asc => asc,
        SortDirection::Desc => !asc,
    }
}

/// Turn a per-block-per-digit histogram into per-block-per-digit **exclusive
/// output offsets** for the stable multi-block LSD radix scatter.
///
/// `block_hist` is laid out block-major: `block_hist[b * RADIX_BUCKETS + d]` is
/// the count of digit `d` in block `b`. `num_blocks` is the launch grid size.
///
/// The output `block_offsets` (same layout) gives, for each `(block, digit)`,
/// the global output position where that block's run of that digit begins. The
/// scatter kernel reads `block_offsets[blockIdx * 16 + digit]` directly (no
/// atomic) and lays its elements down at `base + per_block_stable_rank`.
///
/// Construction (the standard stable-radix prefix sum):
///   1. `digit_total[d]` = sum over all blocks of `count(b, d)`.
///   2. `digit_base[d]`  = exclusive prefix sum of `digit_total` — the start of
///      digit `d`'s contiguous region in the output array.
///   3. For each digit `d`, walk blocks in ascending order accumulating a
///      running offset seeded at `digit_base[d]`:
///        `block_offsets[b*16 + d] = digit_base[d] + sum_{b'<b} count(b', d)`.
///
/// Because the per-digit runs are laid out in ascending `blockIdx` order and
/// each block then fills its run in ascending `threadIdx` order (the kernel's
/// stable per-block rank), equal-digit elements keep their input order globally.
///
/// Errors if any running offset would overflow `u32` — it cannot in practice
/// (the grand total is exactly `n_rows <= u32::MAX`), but we make the invariant
/// load-bearing rather than silently wrapping a garbage readback.
fn compute_block_offsets(
    block_hist: &[u32],
    num_blocks: usize,
    block_offsets: &mut [u32],
) -> BoltResult<()> {
    let buckets = RADIX_BUCKETS as usize;
    debug_assert_eq!(block_hist.len(), num_blocks * buckets);
    debug_assert_eq!(block_offsets.len(), num_blocks * buckets);

    // Step 1+2: per-digit totals, then exclusive prefix sum into digit_base.
    // We fold the two loops together: first accumulate totals, then scan.
    let mut digit_total = [0u32; RADIX_BUCKETS as usize];
    for b in 0..num_blocks {
        let row = &block_hist[b * buckets..b * buckets + buckets];
        for d in 0..buckets {
            digit_total[d] = digit_total[d].checked_add(row[d]).ok_or_else(|| {
                BoltError::Other(
                    "radix block histogram digit total overflowed u32".into(),
                )
            })?;
        }
    }
    let mut digit_base = [0u32; RADIX_BUCKETS as usize];
    let mut running_base: u32 = 0;
    for d in 0..buckets {
        digit_base[d] = running_base;
        running_base = running_base.checked_add(digit_total[d]).ok_or_else(|| {
            BoltError::Other(
                "radix digit base prefix overflowed u32 (invariant: sum == n_rows)".into(),
            )
        })?;
    }

    // Step 3: per digit, accumulate a running offset across blocks. We iterate
    // digit-outer / block-inner so the running counter resets per digit; the
    // writes still target the block-major layout `[b*16 + d]`.
    for d in 0..buckets {
        let mut running = digit_base[d];
        for b in 0..num_blocks {
            block_offsets[b * buckets + d] = running;
            running = running.checked_add(block_hist[b * buckets + d]).ok_or_else(|| {
                BoltError::Other(
                    "radix per-block offset prefix overflowed u32".into(),
                )
            })?;
        }
    }
    Ok(())
}

/// Inner pipeline for Int32 keys. Separated from the Int64 variant so the
/// monomorphic GpuVec types stay statically typed throughout the loop.
///
/// `running_perm` is the running permutation from previous keys (or the
/// identity for the first key). We gather the host key values via this
/// permutation, apply the per-direction pre-transform, and upload the
/// result as `keys_ping`. `idx_ping` is seeded with `running_perm` so
/// the kernel carries it through in lock-step with the keys; the final
/// `idx_ping` after all radix passes is the NEW running permutation.
fn run_radix_pipeline_i32(
    host_keys: &[i32],
    running_perm: &[u32],
    n_rows_u32: u32,
    dtype: DataType,
    direction: SortDirection,
) -> BoltResult<Vec<u32>> {
    debug_assert!(matches!(dtype, DataType::Int32));
    debug_assert_eq!(running_perm.len(), n_rows_u32 as usize);
    let radix_steps = radix_steps_for(dtype)?;

    // ----- host-side gather + direction pre-transform ------------------
    //
    // For multi-key chains: `gathered[i] = host_keys[running_perm[i]]`
    // reorders the key column to match the running ordering from previous
    // passes. For the first key, running_perm is identity, so this
    // collapses to `gathered[i] = host_keys[i]`.
    //
    // The per-direction transform (XOR signed-MSB for ASC, bit-not for
    // DESC) subsumes the kernel's `bolt_radix_msb_flip_i32` and adds DESC
    // support in the same XOR — see [`radix_pre_transform_i32`].
    let mut gathered: Vec<i32> = Vec::with_capacity(running_perm.len());
    for &row in running_perm {
        let v = host_keys[row as usize];
        gathered.push(radix_pre_transform_i32(v, direction));
    }

    // ----- device allocation: ping-pong key + index buffers ------------
    //
    // Both ping and pong are allocated via `from_slice` so each
    // GpuVec's internal `len` bookkeeping is populated up front. The
    // pong buffer's initial contents don't matter (the first scatter
    // overwrites every slot), but the populated len is what lets the
    // final `to_vec()` D2H read all n_rows entries after the last
    // ping↔pong swap.
    let mut keys_ping: GpuVec<i32> = GpuVec::<i32>::from_slice(&gathered)?;
    let mut keys_pong: GpuVec<i32> = GpuVec::<i32>::from_slice(&gathered)?;
    // Seed idx_ping with the running permutation (identity for the first
    // key in the LSD chain). The kernel reorders idx_ping in lock-step
    // with keys_ping, so after all radix passes idx_ping holds the
    // NEW running permutation.
    let mut idx_ping: GpuVec<u32> = GpuVec::<u32>::from_slice(running_perm)?;
    let mut idx_pong: GpuVec<u32> = GpuVec::<u32>::from_slice(running_perm)?;

    // Launch shape. `grid_x` (= num_blocks) sizes the per-block-per-digit
    // histogram / offsets buffers below, so we compute it before allocating.
    let block_size = RADIX_BLOCK_SIZE;
    let grid_x = crate::exec::launch::grid_x_for(n_rows_u32, block_size);
    let block_hist_len = (grid_x as usize) * (RADIX_BUCKETS as usize);

    // Per-block-per-digit histogram + offsets buffers (num_blocks * 16 u32
    // entries each, reused per pass). The stable multi-block radix keeps every
    // block's 16 digit counts separate so the host can build deterministic,
    // blockIdx-ordered output offsets. `zeros(...)` populates the GpuVec's
    // bookkeeping len for the memset_d8 / memcpy paths below.
    let hist_dev: GpuVec<u32> = GpuVec::<u32>::zeros(block_hist_len)?;
    let offsets_dev: GpuVec<u32> = GpuVec::<u32>::zeros(block_hist_len)?;

    // ----- modules + entry points -------------------------------------
    //
    // Each radix kernel goes through the v0.7 `RadixSortKernelSpec`-keyed
    // process-wide cache (see `module_cache::get_or_build_module_for_radix_sort`).
    // On a warm hit the entire codegen + PTX-load round-trip is skipped —
    // the cached `CudaModule` clone is returned in sub-microsecond time.
    // The keys-only scatter and the with-indices scatter share the same
    // `RadixSortPass::Scatter` / `ScatterWithIndices` variants but occupy
    // distinct cache slots because the `entry` argument participates in
    // the key.
    let stream = CudaStream::null();
    let hist_spec = RadixSortKernelSpec {
        pass: RadixSortPass::Histogram,
        dtype,
    };
    let hist_module = module_cache::get_or_build_module_for_radix_sort(
        &hist_spec,
        RADIX_HISTOGRAM_I32_ENTRY,
        |spec| compile_radix_histogram(spec.dtype),
    )?;
    let hist_fn = hist_module.function(&radix_histogram_entry(dtype)?)?;

    let scatter_spec = RadixSortKernelSpec {
        pass: RadixSortPass::ScatterWithIndices,
        dtype,
    };
    let scatter_module = module_cache::get_or_build_module_for_radix_sort(
        &scatter_spec,
        RADIX_SCATTER_WI_I32_ENTRY,
        |spec| compile_radix_scatter_with_indices(spec.dtype),
    )?;
    let scatter_fn = scatter_module.function(&radix_scatter_with_indices_entry(dtype)?)?;

    // #19: the kernel-side MSB-flip (`bolt_radix_msb_flip_i32`) is no
    // longer invoked from this path. The host-side `radix_pre_transform_i32`
    // applied during gather above subsumes the signed-MSB XOR and adds
    // the per-key DESC bit-not in the same pass — strictly cheaper than
    // a one-shot kernel pre-pass since we already touch every key on the
    // host during gather (for multi-key) or upload (for single-key).

    // PERF (radix round-trip): hoist the two (num_blocks * 16) host scratch
    // buffers out of the per-pass loop so we allocate them exactly once for the
    // whole sort instead of churning a fresh Vec on every pass. They are
    // page-locked (`PinnedHostBuffer`) so the per-pass D2H of the per-block
    // histogram and H2D of the per-block offsets can run as *real* async DMAs
    // on the sort stream (`cuMemcpy*Async` requires pinned host memory to
    // overlap — a pageable copy silently degrades to a synchronizing staging
    // copy). The contents are fully overwritten each pass (the D2H fills
    // `hist_host`, `compute_block_offsets` rewrites `offsets_host` from index
    // 0), so carrying stale bytes across passes is benign.
    let mut hist_host: PinnedHostBuffer<u32> =
        PinnedHostBuffer::<u32>::new(block_hist_len)?;
    let mut offsets_host: PinnedHostBuffer<u32> =
        PinnedHostBuffer::<u32>::new(block_hist_len)?;

    // ----- per-pass loop: histogram → host-scan → scatter -------------
    for step in 0..radix_steps {
        let shift = step * crate::jit::sort_kernel_radix::RADIX_BITS;

        // Zero the per-block histogram buffer for this pass. The same GpuVec is
        // reused across passes — `memset_d8` directly on the device pointer
        // avoids allocation churn. The buffer's GpuVec `len` is `block_hist_len`
        // (allocated via `zeros`) so the bytes we zero match the bytes we later
        // D2H out. Zeroing matters: the histogram kernel only writes the 16
        // slots for blocks that actually launched, and a partial final block
        // still writes all 16 of its slots, so a stale buffer would not corrupt
        // results — but zeroing keeps the readback unambiguous.
        let hist_bytes = block_hist_len * std::mem::size_of::<u32>();
        // SAFETY: hist_dev was allocated with `zeros(block_hist_len)`, so the
        // underlying allocation has at least block_hist_len u32 entries; we zero
        // exactly that many bytes.
        unsafe { cuda_sys::memset_d8(hist_dev.device_ptr(), 0, hist_bytes)?; }

        launch_radix_histogram(
            &hist_fn,
            &keys_ping,
            &hist_dev,
            n_rows_u32,
            shift,
            grid_x,
            block_size,
            &stream,
        )?;

        // PERF (radix round-trip): D2H the per-block-per-digit histogram as an
        // *async* copy on the sort stream into the hoisted pinned buffer, then
        // take exactly ONE synchronize — the single unavoidable serialize point,
        // because the host offset computation below reads `hist_host`.
        // SAFETY: hist_dev is a u32 allocation of block_hist_len entries and
        // hist_host is a pinned buffer of block_hist_len u32s; we copy exactly
        // that many elements. The pinned host pages stay live until the sync
        // below retires the DMA.
        unsafe {
            cuda_sys::memcpy_d2h_async::<u32>(
                hist_host.as_mut_ptr(),
                hist_dev.device_ptr(),
                block_hist_len,
                stream.raw(),
            )?;
        }
        stream.synchronize()?;

        // Build per-block-per-digit exclusive output offsets on the host (cheap
        // — `num_blocks * 16` u32s). This is what makes the scatter stable
        // across blocks: each block's run for a digit starts where the previous
        // block's run for that digit ended, in ascending blockIdx order. A
        // device-side scan kernel would remove this host leg but needs a new
        // JIT kernel + GPU validation and is out of scope here.
        compute_block_offsets(
            hist_host.as_slice(),
            grid_x as usize,
            offsets_host.as_mut_slice(),
        )?;

        // PERF (radix round-trip): H2D the offsets as an *async* copy on the
        // same stream. We deliberately do NOT synchronize here — the scatter
        // kernel is enqueued on `stream` immediately below, and same-stream
        // ordering guarantees it observes the completed H2D. The final
        // `stream.synchronize()` after the loop fences the last pass's pinned
        // source pages before they are reused / dropped.
        // SAFETY: offsets_dev has block_hist_len u32 entries; offsets_host is a
        // pinned buffer of block_hist_len u32s. The pinned source stays live
        // until the post-loop synchronize (the buffer outlives the loop).
        unsafe {
            cuda_sys::memcpy_h2d_async::<u32>(
                offsets_dev.device_ptr(),
                offsets_host.as_ptr(),
                block_hist_len,
                stream.raw(),
            )?;
        }

        launch_radix_scatter_with_indices(
            &scatter_fn,
            &keys_ping,
            &keys_pong,
            &idx_ping,
            &idx_pong,
            &offsets_dev,
            n_rows_u32,
            shift,
            grid_x,
            block_size,
            &stream,
        )?;

        // Swap ping ↔ pong for the next pass.
        std::mem::swap(&mut keys_ping, &mut keys_pong);
        std::mem::swap(&mut idx_ping, &mut idx_pong);
    }

    stream.synchronize()?;

    // ----- D2H the permutation ----------------------------------------
    // idx_ping holds the final permutation. GpuVec::to_vec needs self.len
    // to be populated; we constructed it from a slice of n_rows entries
    // so the bookkeeping len is already correct.
    let perm = idx_ping.to_vec()?;

    // Buffers drop here.
    drop(keys_ping);
    drop(keys_pong);
    drop(idx_pong);
    drop(hist_dev);
    drop(offsets_dev);

    Ok(perm)
}

/// Inner pipeline for Int64 keys. Mirror of `run_radix_pipeline_i32`
/// except the key buffer is `GpuVec<i64>` and the pass count is 16.
///
/// See [`run_radix_pipeline_i32`] for the running-permutation + direction
/// transform mechanics.
fn run_radix_pipeline_i64(
    host_keys: &[i64],
    running_perm: &[u32],
    n_rows_u32: u32,
    dtype: DataType,
    direction: SortDirection,
) -> BoltResult<Vec<u32>> {
    debug_assert!(matches!(dtype, DataType::Int64));
    debug_assert_eq!(running_perm.len(), n_rows_u32 as usize);
    let radix_steps = radix_steps_for(dtype)?;

    // Host-side gather + per-direction pre-transform. See the i32 variant
    // for the algebra — same shape, scaled to 64 bits.
    let mut gathered: Vec<i64> = Vec::with_capacity(running_perm.len());
    for &row in running_perm {
        let v = host_keys[row as usize];
        gathered.push(radix_pre_transform_i64(v, direction));
    }

    // See the i32 driver for the rationale behind initialising both
    // ping and pong via `from_slice` — `to_vec()` after the final swap
    // needs the destination buffer's len bookkeeping to be n_rows.
    let mut keys_ping: GpuVec<i64> = GpuVec::<i64>::from_slice(&gathered)?;
    let mut keys_pong: GpuVec<i64> = GpuVec::<i64>::from_slice(&gathered)?;
    let mut idx_ping: GpuVec<u32> = GpuVec::<u32>::from_slice(running_perm)?;
    let mut idx_pong: GpuVec<u32> = GpuVec::<u32>::from_slice(running_perm)?;

    // Launch shape + per-block-per-digit buffer sizing — see the i32 mirror.
    let block_size = RADIX_BLOCK_SIZE;
    let grid_x = crate::exec::launch::grid_x_for(n_rows_u32, block_size);
    let block_hist_len = (grid_x as usize) * (RADIX_BUCKETS as usize);

    let hist_dev: GpuVec<u32> = GpuVec::<u32>::zeros(block_hist_len)?;
    let offsets_dev: GpuVec<u32> = GpuVec::<u32>::zeros(block_hist_len)?;

    let stream = CudaStream::null();
    // See `run_radix_pipeline_i32` for the cache-layer rationale; this is
    // the i64 mirror.
    let hist_spec = RadixSortKernelSpec {
        pass: RadixSortPass::Histogram,
        dtype,
    };
    let hist_module = module_cache::get_or_build_module_for_radix_sort(
        &hist_spec,
        RADIX_HISTOGRAM_I64_ENTRY,
        |spec| compile_radix_histogram(spec.dtype),
    )?;
    let hist_fn = hist_module.function(&radix_histogram_entry(dtype)?)?;
    let scatter_spec = RadixSortKernelSpec {
        pass: RadixSortPass::ScatterWithIndices,
        dtype,
    };
    let scatter_module = module_cache::get_or_build_module_for_radix_sort(
        &scatter_spec,
        RADIX_SCATTER_WI_I64_ENTRY,
        |spec| compile_radix_scatter_with_indices(spec.dtype),
    )?;
    let scatter_fn = scatter_module.function(&radix_scatter_with_indices_entry(dtype)?)?;

    // #19: kernel-side MSB-flip is no longer invoked from this path
    // (see the i32 variant for the algebra). The host pre-transform
    // applied during gather above subsumes both signed-MSB and DESC
    // bit-not in a single XOR.

    // PERF (radix round-trip): hoist the two (num_blocks * 16) host scratch
    // buffers out of the per-pass loop (i64 mirror of the i32 driver) —
    // allocate the pinned histogram / offsets scratch once for all 16 passes,
    // and use page-locked memory so the per-pass D2H/H2D run as real async DMAs
    // on the sort stream. Contents are fully overwritten each pass, so stale
    // carry-over is benign.
    let mut hist_host: PinnedHostBuffer<u32> =
        PinnedHostBuffer::<u32>::new(block_hist_len)?;
    let mut offsets_host: PinnedHostBuffer<u32> =
        PinnedHostBuffer::<u32>::new(block_hist_len)?;

    for step in 0..radix_steps {
        let shift = step * crate::jit::sort_kernel_radix::RADIX_BITS;

        let hist_bytes = block_hist_len * std::mem::size_of::<u32>();
        // SAFETY: hist_dev was allocated with block_hist_len u32 entries.
        unsafe { cuda_sys::memset_d8(hist_dev.device_ptr(), 0, hist_bytes)?; }
        let _ = hist_bytes;

        launch_radix_histogram_i64(
            &hist_fn,
            &keys_ping,
            &hist_dev,
            n_rows_u32,
            shift,
            grid_x,
            block_size,
            &stream,
        )?;

        // PERF (radix round-trip): async D2H of the per-block histogram +
        // exactly one synchronize (the host offset computation depends on it),
        // mirroring the i32 driver.
        // SAFETY: hist_dev has block_hist_len u32 elements; hist_host is a
        // pinned buffer of block_hist_len u32s. Pinned pages stay live until
        // the sync below retires the DMA.
        unsafe {
            cuda_sys::memcpy_d2h_async::<u32>(
                hist_host.as_mut_ptr(),
                hist_dev.device_ptr(),
                block_hist_len,
                stream.raw(),
            )?;
        }
        stream.synchronize()?;

        // Per-block-per-digit exclusive output offsets (cross-block stable).
        // See the i32 mirror and `compute_block_offsets` for the construction.
        compute_block_offsets(
            hist_host.as_slice(),
            grid_x as usize,
            offsets_host.as_mut_slice(),
        )?;

        // PERF (radix round-trip): async H2D of the offsets on the same
        // stream — no synchronize; same-stream ordering makes the scatter
        // below observe the completed copy. The post-loop synchronize fences
        // the last pass's pinned source before reuse / drop.
        // SAFETY: offsets_dev has block_hist_len u32 entries; offsets_host is a
        // pinned buffer of block_hist_len u32s that outlives the loop.
        unsafe {
            cuda_sys::memcpy_h2d_async::<u32>(
                offsets_dev.device_ptr(),
                offsets_host.as_ptr(),
                block_hist_len,
                stream.raw(),
            )?;
        }

        launch_radix_scatter_with_indices_i64(
            &scatter_fn,
            &keys_ping,
            &keys_pong,
            &idx_ping,
            &idx_pong,
            &offsets_dev,
            n_rows_u32,
            shift,
            grid_x,
            block_size,
            &stream,
        )?;

        std::mem::swap(&mut keys_ping, &mut keys_pong);
        std::mem::swap(&mut idx_ping, &mut idx_pong);
    }

    stream.synchronize()?;

    let perm = idx_ping.to_vec()?;
    drop(keys_ping);
    drop(keys_pong);
    drop(idx_pong);
    drop(hist_dev);
    drop(offsets_dev);
    Ok(perm)
}

// Note: the `bolt_radix_msb_flip_<dty>` kernel-side helpers are still
// emitted by `crate::jit::sort_kernel_radix` (the ABI is frozen per #18),
// but #19's driver no longer invokes them — the host-side
// `radix_pre_transform_i32` / `radix_pre_transform_i64` applied during
// gather subsumes both the signed-MSB XOR and the DESC bit-not in a
// single pass. The kernel-side flip is still reachable via the JIT
// module compile / entry-name functions for any future caller that
// needs it.

/// Launch `bolt_radix_histogram_i32` for one radix step. The histogram
/// buffer is atomically updated.
#[allow(clippy::too_many_arguments)]
fn launch_radix_histogram(
    f: &crate::jit::CudaFunction<'_>,
    keys: &GpuVec<i32>,
    hist: &GpuVec<u32>,
    n_rows_u32: u32,
    shift: u32,
    grid_x: u32,
    block_size: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    let mut keys_ptr: CUdeviceptr = keys.device_ptr();
    let mut hist_ptr: CUdeviceptr = hist.device_ptr();
    let mut p_n_rows: u32 = n_rows_u32;
    let mut p_shift: u32 = shift;
    let mut params: Vec<*mut c_void> = vec![
        &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut hist_ptr as *mut CUdeviceptr as *mut c_void,
        &mut p_n_rows as *mut u32 as *mut c_void,
        &mut p_shift as *mut u32 as *mut c_void,
    ];
    // SAFETY: every param slot points at a stack local that outlives the
    // synchronous launch + sync at the end of the driver.
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            f.raw(),
            grid_x, 1, 1,
            block_size, 1, 1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    Ok(())
}

/// Int64 variant of [`launch_radix_histogram`].
#[allow(clippy::too_many_arguments)]
fn launch_radix_histogram_i64(
    f: &crate::jit::CudaFunction<'_>,
    keys: &GpuVec<i64>,
    hist: &GpuVec<u32>,
    n_rows_u32: u32,
    shift: u32,
    grid_x: u32,
    block_size: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    let mut keys_ptr: CUdeviceptr = keys.device_ptr();
    let mut hist_ptr: CUdeviceptr = hist.device_ptr();
    let mut p_n_rows: u32 = n_rows_u32;
    let mut p_shift: u32 = shift;
    let mut params: Vec<*mut c_void> = vec![
        &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut hist_ptr as *mut CUdeviceptr as *mut c_void,
        &mut p_n_rows as *mut u32 as *mut c_void,
        &mut p_shift as *mut u32 as *mut c_void,
    ];
    // SAFETY: same as the i32 variant.
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            f.raw(),
            grid_x, 1, 1,
            block_size, 1, 1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    Ok(())
}

/// Launch `bolt_radix_scatter_i32_with_indices` for one radix step. The
/// `offsets` buffer is atomically updated as the kernel runs.
#[allow(clippy::too_many_arguments)]
fn launch_radix_scatter_with_indices(
    f: &crate::jit::CudaFunction<'_>,
    keys_in: &GpuVec<i32>,
    keys_out: &GpuVec<i32>,
    vals_in: &GpuVec<u32>,
    vals_out: &GpuVec<u32>,
    offsets: &GpuVec<u32>,
    n_rows_u32: u32,
    shift: u32,
    grid_x: u32,
    block_size: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    let mut k_in_ptr: CUdeviceptr = keys_in.device_ptr();
    let mut k_out_ptr: CUdeviceptr = keys_out.device_ptr();
    let mut v_in_ptr: CUdeviceptr = vals_in.device_ptr();
    let mut v_out_ptr: CUdeviceptr = vals_out.device_ptr();
    let mut off_ptr: CUdeviceptr = offsets.device_ptr();
    let mut p_n_rows: u32 = n_rows_u32;
    let mut p_shift: u32 = shift;
    let mut params: Vec<*mut c_void> = vec![
        &mut k_in_ptr as *mut CUdeviceptr as *mut c_void,
        &mut k_out_ptr as *mut CUdeviceptr as *mut c_void,
        &mut v_in_ptr as *mut CUdeviceptr as *mut c_void,
        &mut v_out_ptr as *mut CUdeviceptr as *mut c_void,
        &mut off_ptr as *mut CUdeviceptr as *mut c_void,
        &mut p_n_rows as *mut u32 as *mut c_void,
        &mut p_shift as *mut u32 as *mut c_void,
    ];
    // SAFETY: every param slot points at a stack local that outlives the
    // synchronous launch + sync at the end of the driver.
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            f.raw(),
            grid_x, 1, 1,
            block_size, 1, 1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    Ok(())
}

/// Int64 variant of [`launch_radix_scatter_with_indices`].
#[allow(clippy::too_many_arguments)]
fn launch_radix_scatter_with_indices_i64(
    f: &crate::jit::CudaFunction<'_>,
    keys_in: &GpuVec<i64>,
    keys_out: &GpuVec<i64>,
    vals_in: &GpuVec<u32>,
    vals_out: &GpuVec<u32>,
    offsets: &GpuVec<u32>,
    n_rows_u32: u32,
    shift: u32,
    grid_x: u32,
    block_size: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    let mut k_in_ptr: CUdeviceptr = keys_in.device_ptr();
    let mut k_out_ptr: CUdeviceptr = keys_out.device_ptr();
    let mut v_in_ptr: CUdeviceptr = vals_in.device_ptr();
    let mut v_out_ptr: CUdeviceptr = vals_out.device_ptr();
    let mut off_ptr: CUdeviceptr = offsets.device_ptr();
    let mut p_n_rows: u32 = n_rows_u32;
    let mut p_shift: u32 = shift;
    let mut params: Vec<*mut c_void> = vec![
        &mut k_in_ptr as *mut CUdeviceptr as *mut c_void,
        &mut k_out_ptr as *mut CUdeviceptr as *mut c_void,
        &mut v_in_ptr as *mut CUdeviceptr as *mut c_void,
        &mut v_out_ptr as *mut CUdeviceptr as *mut c_void,
        &mut off_ptr as *mut CUdeviceptr as *mut c_void,
        &mut p_n_rows as *mut u32 as *mut c_void,
        &mut p_shift as *mut u32 as *mut c_void,
    ];
    // SAFETY: same as the i32 variant.
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            f.raw(),
            grid_x, 1, 1,
            block_size, 1, 1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    Ok(())
}

/// Threshold below which we emit the in-block shmem variant. Equal to
/// `SORT_BLOCK_SIZE` because the shmem variant requires `n_pow2 <=
/// block_size` (each thread owns one element). Anything bigger goes through
/// the multi-launch loop.
const SHMEM_VARIANT_MAX_NPOW2: u32 = SORT_BLOCK_SIZE;

// ============================================================================
// Batch 6: CUDA Graph capture for the bitonic-sort `O(log^2 n)` launch
// sequence.
//
// The MultiLaunch path already dropped per-iteration stream syncs in Batch 5
// (one terminal sync after the whole sequence). The next-level win is to
// **capture** the entire launch sequence into a `cuGraph` once per shape so
// subsequent sorts of the same `(n_pow2, dtype, idx_dev_ptr)` can re-launch
// the instantiated graph in a single FFI hop instead of replaying every
// `cuLaunchKernel`. Eliminates O(log^2 n) per-launch driver overhead for the
// steady-state path.
//
// ## Why a *device-pointer* cache key?
//
// `cuStreamBeginCapture_v2` bakes the kernel-arg pointers passed to each
// `cuLaunchKernel` directly into the recorded graph nodes. The instantiated
// `CUgraphExec` is therefore valid only as long as every captured pointer
// keeps pointing at the same live device allocation. The bitonic sort
// writes through one `idx_dev` `GpuVec<u32>` and a set of typed key
// buffers; the indices buffer is the load-bearing one (the sort runs
// in-place on it), so the cache key includes `idx_dev.device_ptr()` AND
// every key buffer's pointer. A different caller — passing a fresh
// `GpuVec` — will MISS the cache and capture a new graph.
//
// `cuGraphExecUpdate` could in principle patch the captured args on a hit
// from a different pointer, but the bookkeeping is intricate (mapping
// `CUgraphNode` handles back to argument indices) and out of scope for
// Batch 6. The pointer-keyed cache is the simpler win for the common case
// where the caller re-uses the same buffers across queries.
//
// ## Lifetime
//
// `CUgraphExec` handles are stored process-wide and intentionally LEAKED at
// process exit — destroying graph execs during context teardown races on
// some drivers and we have no signal that says "the engine is shutting
// down cleanly" before `Drop` runs on the global cache. The cache is
// monotone-grow during a process's lifetime; in practice the number of
// distinct `(n_pow2, dtype, ptr)` triples is small (one per registered
// table per dtype) and the per-entry footprint is the size of a
// `CUgraphExec` (one pointer).
//
// ## Opt-in
//
// Gated by env var `BOLT_SORT_USE_GRAPH=1`. Default OFF: the existing
// per-launch-with-one-terminal-sync path runs unchanged for anyone who
// doesn't flip the flag. Once the path is validated in benchmarks it
// becomes the default; until then this is opt-in.
// ============================================================================

/// Compact dtype tag for the graph cache key. Mirrors [`KeyDeviceBuf`]'s
/// variants — every dtype the bitonic sort accepts maps to a unique byte
/// so two different-typed sorts of the same `n_pow2` and pointer can't
/// alias each other's cached graph.
fn dtype_tag(dt: DataType) -> u8 {
    match dt {
        DataType::Int32 => 1,
        DataType::Int64 => 2,
        DataType::Float32 => 3,
        DataType::Float64 => 4,
        DataType::Bool => 5,
        // Any future GPU-sortable dtype must add a unique tag here; the
        // catch-all returns 255 so a misconfigured path errs on the side
        // of cache-miss-then-rebuild rather than alias.
        _ => 255,
    }
}

/// Process-wide cache of instantiated bitonic-sort graphs.
///
/// Key tuple (in order):
///   - `n_pow2`           — recorded grid size baked into the graph.
///   - `dtype_tag`        — keeps different-typed sorts on the same
///                          buffer from aliasing.
///   - `idx_dev_ptr`      — the indices buffer (`idx_dev.device_ptr()`).
///                          Bitonic sort writes through this in place.
///   - `keys_fingerprint` — XOR of every key buffer's device pointer. Two
///                          launches with different key buffers must MISS
///                          and re-capture; XOR is the cheapest collision-
///                          resistant fingerprint that's cheap to compute
///                          and doesn't need a Vec in the key.
///
/// The value is a `CUgraphExec` — a raw pointer that's wrapped in
/// `GraphExecHandle` so we can `Send` it across threads via the `static`.
type GraphCacheKey = (u32, u8, u64, u64);

/// `Send + Sync` wrapper around `CUgraphExec`. The driver permits using
/// a `CUgraphExec` from any thread once its context is current; we only
/// dereference the handle while we already hold the engine's `CudaContext`.
struct GraphExecHandle(CUgraphExec);
// SAFETY: a `CUgraphExec` is a raw opaque handle the driver dereferences;
// the engine serialises capture and launch through `GRAPH_CACHE.lock()`
// and only launches with a live context bound to the calling thread.
unsafe impl Send for GraphExecHandle {}
unsafe impl Sync for GraphExecHandle {}

/// Lazily-initialised cache. Allocated on first lookup, monotone-grow,
/// leaked at process exit (see module-level rationale).
static GRAPH_CACHE: Lazy<Mutex<HashMap<GraphCacheKey, GraphExecHandle>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Env-var name controlling the graph-capture opt-in. Exposed as a
/// constant so the tests can verify the gate logic without string-typing
/// the name.
pub(crate) const BOLT_SORT_USE_GRAPH_ENV: &str = "BOLT_SORT_USE_GRAPH";

/// Parse the `BOLT_SORT_USE_GRAPH` env var into a bool. Returns `true`
/// only when set to exactly `"1"`. Any other value (including `"0"`,
/// `"true"`, unparseable garbage, or unset) returns `false` — the strict
/// `=="1"` semantics keep the gate from accidentally tripping on shell
/// quoting or boolean-style strings the user might assume "feel right".
pub(crate) fn sort_uses_graph() -> bool {
    match std::env::var(BOLT_SORT_USE_GRAPH_ENV) {
        Ok(v) => v == "1",
        Err(_) => false,
    }
}

/// Test-only: clear the graph cache so cache-hit/-miss tests start from
/// a known-empty state. Production code never invokes this.
#[cfg(test)]
fn _test_clear_graph_cache() {
    GRAPH_CACHE.lock().clear();
}

/// Sort `keys` (up to [`MAX_SORT_KEYS`]) lexicographically on the GPU and
/// return the row permutation as a `UInt32Array` of length `n_rows`.
///
/// Picks between the multi-launch and shmem-variant kernels based on
/// `n_pow2`. For `n_pow2 <= SORT_BLOCK_SIZE` the shmem variant runs as a
/// single launch; above that we fall back to one launch per substage.
///
/// Returns the (taken_path, permutation) pair so callers/tests can verify
/// the dispatch decision without observing it through a counter.
///
/// **Stage 5** — the return type wraps an `Option` so a per-key
/// `host_values_for_key` fall-through signal (currently the high-cardinality
/// Utf8 gate) can bubble out without erroring. `Ok(None)` means "the caller
/// should run the host sort instead"; callers (`sort_record_batch_on_gpu_multi`,
/// `try_gpu_sort`) propagate `None` upward.
pub fn sort_indices_on_gpu_multi<'a>(
    keys: &[GpuSortKey<'a>],
) -> BoltResult<Option<(SortLayout, UInt32Array)>> {
    if keys.is_empty() {
        return Err(BoltError::Other(
            "gpu_sort: sort_indices_on_gpu_multi needs at least 1 key".into(),
        ));
    }
    if keys.len() > MAX_SORT_KEYS {
        return Err(BoltError::Other(format!(
            "gpu_sort: too many keys ({}); hard cap is {}",
            keys.len(),
            MAX_SORT_KEYS
        )));
    }
    // Stage 3: also enforce the register-pressure budget up-front so we
    // fail before allocating GPU buffers.
    let reg_tally: u32 = keys
        .iter()
        .map(|k| crate::jit::sort_kernel::key_reg_cost(k.dtype))
        .sum();
    if reg_tally > crate::jit::sort_kernel::SM70_KEY_REG_BUDGET {
        return Err(BoltError::Other(format!(
            "gpu_sort: keys would consume {} b32-register equivalents; sm_70 budget \
             is {} (drop a key or split into multi-pass)",
            reg_tally,
            crate::jit::sort_kernel::SM70_KEY_REG_BUDGET
        )));
    }

    let n_rows = keys[0].column.len();
    for k in keys {
        if k.column.len() != n_rows {
            return Err(BoltError::Other(format!(
                "gpu_sort: key column length mismatch ({} vs {})",
                k.column.len(),
                n_rows
            )));
        }
    }
    if n_rows == 0 {
        return Ok(Some((
            SortLayout::MultiLaunch,
            UInt32Array::from(Vec::<u32>::new()),
        )));
    }
    let n_pow2 = next_pow2_u32(n_rows)?;
    let n_pow2_usize = n_pow2 as usize;
    let stream = CudaStream::null();

    // Decide layout. Shmem variant when the whole padded sort fits in a
    // single block's worth of shared memory.
    let layout = if n_pow2 <= SHMEM_VARIANT_MAX_NPOW2 {
        SortLayout::Shmem
    } else {
        SortLayout::MultiLaunch
    };

    // Build the spec.
    let key_descs: Vec<KeyDesc> = keys
        .iter()
        .map(|k| KeyDesc {
            dtype: k.dtype,
            direction: k.direction,
            nulls_first: k.nulls_first,
            nullable: k.column.null_count() > 0,
        })
        .collect();
    let spec = SortKernelSpec {
        keys: key_descs.clone(),
        layout,
        shmem_n_pow2: if matches!(layout, SortLayout::Shmem) {
            n_pow2
        } else {
            0
        },
    };

    // Upload each key's padded values + (if nullable) its validity bitmap.
    // Buffers live for the duration of the kernel launches.
    let mut key_bufs: Vec<KeyDeviceBuf> = Vec::with_capacity(keys.len());
    let mut validity_bufs: Vec<Option<GpuVec<u8>>> = Vec::with_capacity(keys.len());
    for (k, kd) in keys.iter().zip(key_descs.iter()) {
        let kb = match upload_padded_key_for_dtype(k.column, k.dtype, k.direction, n_pow2_usize)? {
            Some(buf) => buf,
            // Stage 5: per-key fall-through (high-cardinality Utf8). Drop any
            // device buffers we've already allocated and tell the caller to
            // take the host path. RAII via the existing GpuVec drops on
            // function exit handles cleanup — we just return here.
            None => return Ok(None),
        };
        key_bufs.push(kb);
        if kd.nullable {
            let bm = build_validity_padded(k.column, n_pow2_usize);
            validity_bufs.push(Some(GpuVec::<u8>::from_slice(&bm)?));
        } else {
            validity_bufs.push(None);
        }
    }
    // Indices buffer (identity).
    let idx_host: Vec<u32> = (0..n_pow2).collect();
    let idx_dev = GpuVec::<u32>::from_slice(&idx_host)?;

    // Stage 3: is_padded packed-bit bitmap, uploaded as a u8 buffer.
    let is_padded_host = build_is_padded(n_rows, n_pow2_usize);
    let is_padded_dev = GpuVec::<u8>::from_slice(&is_padded_host)?;

    // Compile + load the module via the consolidated `exec::module_cache`.
    // The sort PTX is a pure function of `spec`, so a Debug-formatted spec
    // is a stable cache id.
    let module = module_cache::get_or_build_module(
        module_path!(),
        format!("sort_kernel:{:?}", spec),
        None,
        || compile_sort_kernel_spec(&spec),
    )?;
    let entry = sort_kernel_entry_spec(&spec)?;
    let function = module.function(&entry)?;

    // Build the param array. ABI is constant across MultiLaunch / Shmem
    // except for the trailing stage/mask pair.
    //
    // Slots 0..2*MAX_SORT_KEYS : alternating (key_ptr, validity_ptr).
    //   Used keys point to real buffers; unused slots point to null.
    // Slot 2*MAX_SORT_KEYS     : indices_ptr.
    // Slot 2*MAX_SORT_KEYS+1   : n_pow2 (u32).
    // Slots .. + 2/+3          : stage / substage_mask (MultiLaunch only).
    let null_ptr: CUdeviceptr = 0;
    let mut key_ptrs: [CUdeviceptr; MAX_SORT_KEYS] = [null_ptr; MAX_SORT_KEYS];
    let mut val_ptrs: [CUdeviceptr; MAX_SORT_KEYS] = [null_ptr; MAX_SORT_KEYS];
    for (i, kb) in key_bufs.iter().enumerate() {
        key_ptrs[i] = kb.device_ptr();
        val_ptrs[i] = validity_bufs[i]
            .as_ref()
            .map(|v| v.device_ptr())
            .unwrap_or(null_ptr);
    }
    let mut indices_ptr = idx_dev.device_ptr();
    let mut is_padded_ptr = is_padded_dev.device_ptr();
    let mut p_n_pow2: u32 = n_pow2;

    match layout {
        SortLayout::Shmem => {
            // Single launch. block_size = n_pow2 (one thread per element),
            // grid = 1.
            let block_size: u32 = n_pow2.max(1);
            let grid_x: u32 = 1;

            let mut kp = key_ptrs;
            let mut vp = val_ptrs;
            // Interleave (k0, v0, k1, v1, ..., kN, vN, indices, is_padded, n_pow2).
            let mut params: Vec<*mut c_void> = Vec::with_capacity(MAX_SORT_KEYS * 2 + 3);
            for i in 0..MAX_SORT_KEYS {
                params.push(&mut kp[i] as *mut CUdeviceptr as *mut c_void);
                params.push(&mut vp[i] as *mut CUdeviceptr as *mut c_void);
            }
            params.push(&mut indices_ptr as *mut CUdeviceptr as *mut c_void);
            params.push(&mut is_padded_ptr as *mut CUdeviceptr as *mut c_void);
            params.push(&mut p_n_pow2 as *mut u32 as *mut c_void);

            // SAFETY: every entry of `params` points at a stack local that
            // outlives the launch+synchronize below.
            unsafe {
                cuda_sys::check(cuda_sys::cuLaunchKernel(
                    function.raw(),
                    grid_x,
                    1,
                    1,
                    block_size,
                    1,
                    1,
                    0,
                    stream.raw(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ))?;
            }
            stream.synchronize()?;
        }
        SortLayout::MultiLaunch => {
            let block_size: u32 = SORT_BLOCK_SIZE;
            let grid_x: u32 = n_pow2.div_ceil(block_size);
            let log2_n = log2_pow2(n_pow2);

            // Batch 6: when BOLT_SORT_USE_GRAPH=1, route through the
            // CUDA-Graph cache. The default path falls through to the
            // existing per-launch loop with one terminal sync. Both
            // branches end with a single sync; only the
            // record-vs-execute method changes.
            if sort_uses_graph() {
                // Build the cache key. `keys_fingerprint` is the XOR of every
                // captured pointer; that's enough to detect "different
                // buffers" without storing a Vec in the key.
                let mut keys_fp: u64 = 0;
                for p in key_ptrs.iter() {
                    keys_fp ^= *p;
                }
                for p in val_ptrs.iter() {
                    keys_fp ^= *p;
                }
                keys_fp ^= is_padded_ptr;
                // First key's dtype dominates the cache tag — every key in a
                // single launch shares the same kernel ABI, but the first key
                // is the load-bearing one for grid sizing.
                let tag = dtype_tag(key_descs[0].dtype);
                let cache_key: GraphCacheKey = (n_pow2, tag, indices_ptr, keys_fp);

                // Cache lookup. We hold the mutex for the entire capture /
                // instantiate critical section to serialise concurrent first
                // launches of the same shape; the per-shape work is
                // amortised over every subsequent launch so the contention
                // is a non-issue.
                let mut cache = GRAPH_CACHE.lock();
                let exec_handle: CUgraphExec = match cache.get(&cache_key) {
                    Some(h) => h.0,
                    None => {
                        // MISS: capture the launch sequence on a dedicated
                        // stream (cuStreamBeginCapture rejects the NULL
                        // stream), instantiate, cache.
                        let capture_stream = CudaStream::new()?;
                        cuda_sys::stream_begin_capture(
                            capture_stream.raw(),
                            cuda_sys::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                        )?;
                        // Issue every kernel launch on the capture stream.
                        // No per-launch sync — stream capture forbids it and
                        // we don't need it.
                        let mut capture_ok = true;
                        let mut capture_err: Option<BoltError> = None;
                        'capture: for stage in 1..=log2_n {
                            let mut substage = stage;
                            loop {
                                let substage_mask: u32 = 1u32 << (substage - 1);
                                let mut kp = key_ptrs;
                                let mut vp = val_ptrs;
                                let mut p_stage: u32 = stage;
                                let mut p_mask: u32 = substage_mask;
                                let mut params: Vec<*mut c_void> =
                                    Vec::with_capacity(MAX_SORT_KEYS * 2 + 5);
                                for i in 0..MAX_SORT_KEYS {
                                    params.push(
                                        &mut kp[i] as *mut CUdeviceptr as *mut c_void,
                                    );
                                    params.push(
                                        &mut vp[i] as *mut CUdeviceptr as *mut c_void,
                                    );
                                }
                                params.push(
                                    &mut indices_ptr as *mut CUdeviceptr as *mut c_void,
                                );
                                params.push(
                                    &mut is_padded_ptr as *mut CUdeviceptr as *mut c_void,
                                );
                                params.push(&mut p_n_pow2 as *mut u32 as *mut c_void);
                                params.push(&mut p_stage as *mut u32 as *mut c_void);
                                params.push(&mut p_mask as *mut u32 as *mut c_void);

                                // SAFETY: every param points at a stack local
                                // that outlives this iteration; stream
                                // capture only records the pointer values,
                                // not the storage behind them.
                                let rc = unsafe {
                                    cuda_sys::check(cuda_sys::cuLaunchKernel(
                                        function.raw(),
                                        grid_x,
                                        1,
                                        1,
                                        block_size,
                                        1,
                                        1,
                                        0,
                                        capture_stream.raw(),
                                        params.as_mut_ptr(),
                                        ptr::null_mut(),
                                    ))
                                };
                                if let Err(e) = rc {
                                    capture_err = Some(e);
                                    capture_ok = false;
                                    break 'capture;
                                }
                                if substage == 1 {
                                    break;
                                }
                                substage -= 1;
                            }
                        }

                        // Always end capture so the stream isn't left in
                        // capture state, even on the error path. The
                        // returned graph is dropped on error.
                        let graph = cuda_sys::stream_end_capture(capture_stream.raw())?;
                        if !capture_ok {
                            let _ = cuda_sys::graph_destroy(graph);
                            return Err(capture_err.unwrap_or_else(|| {
                                BoltError::Other(
                                    "gpu_sort: stream-capture failed without a specific error"
                                        .into(),
                                )
                            }));
                        }
                        // Instantiate, then destroy the source `CUgraph`
                        // (the CUgraphExec keeps its own copy).
                        let exec = cuda_sys::graph_instantiate(graph)?;
                        let _ = cuda_sys::graph_destroy(graph);
                        cache.insert(cache_key, GraphExecHandle(exec));
                        exec
                    }
                };
                // Release the cache lock before launching — the launch
                // itself is concurrent-safe across threads as long as each
                // call uses a separate stream (we use the NULL stream here;
                // callers that need overlap should mint their own).
                drop(cache);
                cuda_sys::graph_launch(exec_handle, stream.raw())?;
                stream.synchronize()?;
            } else {
                // Default path (Batch 5 behaviour): one launch per substage,
                // terminal sync at the end. Per-launch sync was dropped in
                // Batch 5; the only sync is the one after the loop.
                for stage in 1..=log2_n {
                    let mut substage = stage;
                    loop {
                        let substage_mask: u32 = 1u32 << (substage - 1);
                        let mut kp = key_ptrs;
                        let mut vp = val_ptrs;
                        let mut p_stage: u32 = stage;
                        let mut p_mask: u32 = substage_mask;
                        let mut params: Vec<*mut c_void> =
                            Vec::with_capacity(MAX_SORT_KEYS * 2 + 5);
                        for i in 0..MAX_SORT_KEYS {
                            params.push(&mut kp[i] as *mut CUdeviceptr as *mut c_void);
                            params.push(&mut vp[i] as *mut CUdeviceptr as *mut c_void);
                        }
                        params.push(&mut indices_ptr as *mut CUdeviceptr as *mut c_void);
                        params.push(&mut is_padded_ptr as *mut CUdeviceptr as *mut c_void);
                        params.push(&mut p_n_pow2 as *mut u32 as *mut c_void);
                        params.push(&mut p_stage as *mut u32 as *mut c_void);
                        params.push(&mut p_mask as *mut u32 as *mut c_void);

                        // SAFETY: same as Shmem branch — every param points
                        // at a stack local that outlives the synchronous
                        // launch.
                        unsafe {
                            cuda_sys::check(cuda_sys::cuLaunchKernel(
                                function.raw(),
                                grid_x,
                                1,
                                1,
                                block_size,
                                1,
                                1,
                                0,
                                stream.raw(),
                                params.as_mut_ptr(),
                                ptr::null_mut(),
                            ))?;
                        }
                        stream.synchronize()?;
                        if substage == 1 {
                            break;
                        }
                        substage -= 1;
                    }
                }
            }
            // Single sync after the whole bitonic dispatch — required
            // because `idx_dev.to_vec()` below issues a blocking D2H on
            // the default (null) stream, and we must ensure the
            // sort-stream launches have completed before that read.
            stream.synchronize()?;
        }
    }

    // Download indices and truncate.
    //
    // Stage 3: with the is_padded routing in the kernel, real rows are now
    // guaranteed to live in `0..n_rows` and padded rows in `n_rows..n_pow2`
    // (modulo direction). The truncation just takes the first `n_rows`
    // entries with index < n_rows; the defensive filter is still kept in
    // case a future kernel regression slips a padded index in.
    let idx_host_sorted: Vec<u32> = idx_dev.to_vec()?;
    let n_rows_u32 = n_rows_to_u32(n_rows)?;
    let mut out: Vec<u32> = Vec::with_capacity(n_rows);
    for v in &idx_host_sorted {
        if *v < n_rows_u32 {
            out.push(*v);
        }
        if out.len() == n_rows {
            break;
        }
    }
    if out.len() != n_rows {
        return Err(BoltError::Other(format!(
            "gpu_sort multi-key: recovered only {} indices for {} real rows \
             (padded-bit routing should prevent this)",
            out.len(),
            n_rows
        )));
    }

    // Buffers (key + validity + idx_dev + is_padded) drop here.
    drop(key_bufs);
    drop(validity_bufs);
    drop(idx_dev);
    drop(is_padded_dev);

    Ok(Some((layout, UInt32Array::from(out))))
}

/// Sort an entire `RecordBatch` by `keys` on the GPU, gather every column.
///
/// **Stage 5** — returns `Ok(None)` if any key column triggers the
/// per-key fall-through gate (today: the high-cardinality plain-Utf8
/// sampler). The caller (`try_gpu_sort`) propagates `None` upward and the
/// host `lexsort_to_indices` handles the sort.
pub fn sort_record_batch_on_gpu_multi(
    batch: &RecordBatch,
    keys: &[(usize, DataType, SortDirection, bool /*nulls_first*/)],
) -> BoltResult<Option<RecordBatch>> {
    let sort_keys: Vec<GpuSortKey> = keys
        .iter()
        .map(|(idx, dtype, dir, nf)| GpuSortKey {
            column: batch.column(*idx).as_ref(),
            dtype: *dtype,
            direction: *dir,
            nulls_first: *nf,
        })
        .collect();
    let (_layout, perm) = match sort_indices_on_gpu_multi(&sort_keys)? {
        Some(pair) => pair,
        None => return Ok(None),
    };
    let new_cols: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| {
            take(c.as_ref(), &perm, None).map_err(|e| {
                BoltError::Other(format!("gpu_sort multi-key: arrow take failed: {e}"))
            })
        })
        .collect::<BoltResult<Vec<_>>>()?;
    RecordBatch::try_new(batch.schema(), new_cols)
        .map(Some)
        .map_err(|e| BoltError::Other(format!("gpu_sort multi-key: RecordBatch build failed: {e}")))
}

/// Threshold (in n_pow2 terms) at which the multi-key driver switches to the
/// shmem variant. Exposed for tests that want to verify the dispatch
/// decision without running on a CUDA device.
#[allow(dead_code)]
pub fn shmem_variant_threshold_n_pow2() -> u32 {
    SHMEM_VARIANT_MAX_NPOW2
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType as ArrowDataType, Field, Schema};

    // -- pure-host helpers --

    #[test]
    fn next_pow2_basics() {
        assert_eq!(next_pow2_u32(0).unwrap(), 1);
        assert_eq!(next_pow2_u32(1).unwrap(), 1);
        assert_eq!(next_pow2_u32(2).unwrap(), 2);
        assert_eq!(next_pow2_u32(3).unwrap(), 4);
        assert_eq!(next_pow2_u32(100).unwrap(), 128);
        assert_eq!(next_pow2_u32(1 << 20).unwrap(), 1 << 20);
        assert_eq!(next_pow2_u32((1 << 20) + 1).unwrap(), 1 << 21);
    }

    #[test]
    fn next_pow2_rejects_overflow() {
        // 2^31 + 1 would round up to 2^32, which doesn't fit in u32.
        let oversized = (1usize << 31) + 1;
        assert!(next_pow2_u32(oversized).is_err());
    }

    #[test]
    fn log2_pow2_correct() {
        assert_eq!(log2_pow2(1), 0);
        assert_eq!(log2_pow2(2), 1);
        assert_eq!(log2_pow2(4), 2);
        assert_eq!(log2_pow2(1024), 10);
        assert_eq!(log2_pow2(1 << 20), 20);
    }

    #[test]
    fn pad_to_pow2_appends_sentinel() {
        let padded = pad_to_pow2(&[3i32, 1, 4], 8, i32::MAX);
        assert_eq!(padded.len(), 8);
        assert_eq!(&padded[..3], &[3, 1, 4]);
        for v in &padded[3..] {
            assert_eq!(*v, i32::MAX);
        }
    }

    #[test]
    fn arrow_dtype_mapping() {
        assert_eq!(
            arrow_dtype_to_internal(&ArrowDataType::Int32),
            Some(DataType::Int32)
        );
        assert_eq!(
            arrow_dtype_to_internal(&ArrowDataType::Int64),
            Some(DataType::Int64)
        );
        assert_eq!(
            arrow_dtype_to_internal(&ArrowDataType::Float32),
            Some(DataType::Float32)
        );
        assert_eq!(
            arrow_dtype_to_internal(&ArrowDataType::Float64),
            Some(DataType::Float64)
        );
        // Stage 3 additions: Boolean -> Bool; Dictionary(I32|I64, Utf8) ->
        // the index dtype.
        // Stage 4 addition: plain Utf8 -> Int32 (inline-dictionary on the
        // fly inside `host_values_for_key`).
        assert_eq!(
            arrow_dtype_to_internal(&ArrowDataType::Utf8),
            Some(DataType::Int32)
        );
        assert_eq!(
            arrow_dtype_to_internal(&ArrowDataType::Boolean),
            Some(DataType::Bool)
        );
        assert_eq!(
            arrow_dtype_to_internal(&ArrowDataType::Dictionary(
                Box::new(ArrowDataType::Int32),
                Box::new(ArrowDataType::Utf8),
            )),
            Some(DataType::Int32)
        );
        assert_eq!(
            arrow_dtype_to_internal(&ArrowDataType::Dictionary(
                Box::new(ArrowDataType::Int64),
                Box::new(ArrowDataType::Utf8),
            )),
            Some(DataType::Int64)
        );
        // Non-string-valued dict (e.g. dict<i32, i64>): reject — caller
        // should hand the inner i64 directly through the numeric path.
        assert_eq!(
            arrow_dtype_to_internal(&ArrowDataType::Dictionary(
                Box::new(ArrowDataType::Int32),
                Box::new(ArrowDataType::Int64),
            )),
            None
        );
    }

    /// Stage 4 inline-dictionary builder: each row gets an i32 index such
    /// that the integer ASC order matches the lex ASC order of the strings.
    #[test]
    fn build_inline_dict_indices_assigns_lex_order() {
        let sa = StringArray::from(vec!["bravo", "alpha", "charlie", "alpha", "bravo"]);
        let idx = build_inline_dict_indices(&sa);
        // Distinct strings are {alpha, bravo, charlie} -> lex ranks 0,1,2.
        assert_eq!(idx, vec![1, 0, 2, 0, 1]);
    }

    /// Sorting the inline-dict indices ASC must produce the same row order
    /// as sorting the strings ASC.
    #[test]
    fn build_inline_dict_indices_matches_string_sort() {
        let inputs = vec!["zebra", "apple", "mango", "apple", "banana", "zebra"];
        let sa = StringArray::from(inputs.clone());
        let idx = build_inline_dict_indices(&sa);
        // Pair (idx, original_position) and sort by idx; the resulting row
        // order must match the row order from sorting `inputs` directly.
        let mut by_idx: Vec<(i32, usize)> =
            idx.iter().copied().zip(0..inputs.len()).collect();
        by_idx.sort_by_key(|p| p.0);
        let mut by_str: Vec<(&str, usize)> =
            inputs.iter().copied().zip(0..inputs.len()).collect();
        by_str.sort_by_key(|p| p.0);
        assert_eq!(
            by_idx.iter().map(|(_, i)| *i).collect::<Vec<_>>(),
            by_str.iter().map(|(_, i)| *i).collect::<Vec<_>>()
        );
    }

    /// Empty StringArray and all-null StringArray are degenerate but mustn't
    /// panic — they hit the `dict_seen.is_empty()` guard.
    #[test]
    fn build_inline_dict_indices_empty_and_all_null() {
        let sa_empty = StringArray::from(Vec::<&str>::new());
        let idx = build_inline_dict_indices(&sa_empty);
        assert!(idx.is_empty());

        // All-null: every row contributes the placeholder 0.
        let sa_nulls = StringArray::from(vec![None::<&str>, None, None]);
        let idx = build_inline_dict_indices(&sa_nulls);
        assert_eq!(idx, vec![0, 0, 0]);
    }

    /// Stage 3 padded-bit bitmap layout: padded slots at indices >= n_rows
    /// get bit=1; real rows get bit=0. Length is `ceil(n_pow2 / 8)`.
    #[test]
    fn build_is_padded_marks_only_pad_slots() {
        let n_rows = 5;
        let n_pow2 = 8;
        let bm = build_is_padded(n_rows, n_pow2);
        assert_eq!(bm.len(), 1); // ceil(8/8) = 1 byte
        // bits 0..5 = 0 (real); bits 5..8 = 1 (padded). 0b1110_0000 = 0xE0.
        assert_eq!(bm[0], 0xE0);
    }

    #[test]
    fn build_is_padded_handles_no_padding() {
        let n_rows = 8;
        let n_pow2 = 8;
        let bm = build_is_padded(n_rows, n_pow2);
        assert_eq!(bm.len(), 1);
        assert_eq!(bm[0], 0x00, "no slots padded when n_rows == n_pow2");
    }

    // -- GPU round-trip (ignored on hostless CI) --
    //
    // Stage 4: migrated from the (now-deleted) Stage-1 `sort_indices_on_gpu`
    // single-key entry point to the multi-key driver. The Stage-1 PTX module
    // was unreachable after Stage 3 routed every supported case through the
    // multi-key driver; deleting it removes a parallel code path that was
    // exercising the same `compile_sort_kernel_spec` PTX surface in a
    // narrower form.

    /// DIAGNOSTIC: triangulate the bitonic-sort bug by isolating one dimension
    /// at a time (DESC, 64-bit, float, padding). The shipped tests confound
    /// these (int32/asc/no-pad passes; int64/desc/PAD and float64/asc/PAD
    /// fail). Each case below varies exactly ONE axis off the known-good
    /// int32/asc/no-pad baseline. Collects all failures so one run names every
    /// broken dimension.
    #[test]
    #[ignore = "gpu:sort"]
    fn gpu_sort_dimension_isolation() {
        fn scramble_idx(n: usize, seed: u64) -> Vec<usize> {
            let mut idx: Vec<usize> = (0..n).collect();
            let mut s = seed;
            for i in (1..n).rev() {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = (s as usize) % (i + 1);
                idx.swap(i, j);
            }
            idx
        }
        let mut failures: Vec<String> = Vec::new();

        // (1) int32 DESC, no padding (n = 2^14) — isolates DESC.
        {
            let n = 16_384usize;
            let perm_idx = scramble_idx(n, 0x1111);
            let values: Vec<i32> = perm_idx.iter().map(|&i| i as i32).collect();
            let arr = Int32Array::from(values.clone());
            let keys = vec![GpuSortKey {
                column: &arr,
                dtype: DataType::Int32,
                direction: SortDirection::Desc,
                nulls_first: false,
            }];
            match sort_indices_on_gpu_multi(&keys) {
                Ok(Some((_l, perm))) => {
                    let sorted: Vec<i32> =
                        (0..n).map(|i| values[perm.value(i) as usize]).collect();
                    if sorted.windows(2).any(|w| w[0] < w[1]) {
                        failures.push("int32/DESC/no-pad: non-monotonic".into());
                    }
                }
                other => failures.push(format!("int32/DESC/no-pad: {other:?}")),
            }
        }
        // (2) int64 ASC, no padding (n = 2^14) — isolates 64-bit.
        {
            let n = 16_384usize;
            let perm_idx = scramble_idx(n, 0x2222);
            let values: Vec<i64> = perm_idx.iter().map(|&i| i as i64).collect();
            let arr = Int64Array::from(values.clone());
            let keys = vec![GpuSortKey {
                column: &arr,
                dtype: DataType::Int64,
                direction: SortDirection::Asc,
                nulls_first: false,
            }];
            match sort_indices_on_gpu_multi(&keys) {
                Ok(Some((_l, perm))) => {
                    let sorted: Vec<i64> =
                        (0..n).map(|i| values[perm.value(i) as usize]).collect();
                    if sorted.windows(2).any(|w| w[0] > w[1]) {
                        failures.push("int64/ASC/no-pad: non-monotonic".into());
                    }
                }
                other => failures.push(format!("int64/ASC/no-pad: {other:?}")),
            }
        }
        // (3) float64 ASC, no padding (n = 2^14) — isolates float (incl. negatives).
        {
            let n = 16_384usize;
            let perm_idx = scramble_idx(n, 0x3333);
            let values: Vec<f64> = perm_idx.iter().map(|&i| (i as f64) - 8192.0).collect();
            let arr = Float64Array::from(values.clone());
            let keys = vec![GpuSortKey {
                column: &arr,
                dtype: DataType::Float64,
                direction: SortDirection::Asc,
                nulls_first: false,
            }];
            match sort_indices_on_gpu_multi(&keys) {
                Ok(Some((_l, perm))) => {
                    let sorted: Vec<f64> =
                        (0..n).map(|i| values[perm.value(i) as usize]).collect();
                    if sorted.windows(2).any(|w| w[0] > w[1]) {
                        failures.push("float64/ASC/no-pad: non-monotonic".into());
                    }
                }
                other => failures.push(format!("float64/ASC/no-pad: {other:?}")),
            }
        }
        // (4) int32 ASC, WITH padding (n = 20000, not pow2) — isolates padding.
        {
            let n = 20_000usize;
            let perm_idx = scramble_idx(n, 0x4444);
            let values: Vec<i32> = perm_idx.iter().map(|&i| i as i32).collect();
            let arr = Int32Array::from(values.clone());
            let keys = vec![GpuSortKey {
                column: &arr,
                dtype: DataType::Int32,
                direction: SortDirection::Asc,
                nulls_first: false,
            }];
            match sort_indices_on_gpu_multi(&keys) {
                Ok(Some((_l, perm))) => {
                    if perm.len() != n {
                        failures.push(format!("int32/ASC/pad: len {} != {n}", perm.len()));
                    } else {
                        let sorted: Vec<i32> =
                            (0..n).map(|i| values[perm.value(i) as usize]).collect();
                        if sorted.windows(2).any(|w| w[0] > w[1]) {
                            failures.push("int32/ASC/pad: non-monotonic".into());
                        }
                    }
                }
                other => failures.push(format!("int32/ASC/pad: {other:?}")),
            }
        }

        assert!(
            failures.is_empty(),
            "bitonic dimension isolation — failing axes:\n  {}",
            failures.join("\n  ")
        );
    }

    /// End-to-end ASC int32 sort. Builds a 16k-row scrambled column, runs it
    /// through `sort_indices_on_gpu_multi` (single-key spec), gathers, and
    /// asserts strictly ascending output.
    #[test]
    #[ignore = "gpu:sort"]
    fn gpu_sort_int32_asc_round_trip() {
        // 16384 = 2^14, exact power of two: no padding required, exercises the
        // happy path without truncation noise.
        let n = 16_384usize;
        // Build a scrambled column: deterministic linear-congruential perm
        // of 0..n, easy to recompute the expected sorted order.
        let mut values: Vec<i32> = (0..n as i32).collect();
        // simple Fisher-Yates with a fixed seed
        let mut rng_state: u64 = 0xdeadbeef;
        for i in (1..n).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng_state as usize) % (i + 1);
            values.swap(i, j);
        }
        let arr = Int32Array::from(values.clone());

        let keys = vec![GpuSortKey {
            column: &arr,
            dtype: DataType::Int32,
            direction: SortDirection::Asc,
            nulls_first: false,
        }];
        let (_layout, perm) = sort_indices_on_gpu_multi(&keys)
            .expect("gpu sort")
            .expect("non-fallback path on int32");

        // Apply the permutation host-side and verify ASC order.
        let sorted: Vec<i32> = (0..n).map(|i| values[perm.value(i) as usize]).collect();
        for w in sorted.windows(2) {
            assert!(w[0] <= w[1], "non-monotonic: {} > {}", w[0], w[1]);
        }
        // And the result is a true permutation of the input.
        let mut expected: Vec<i32> = values.clone();
        expected.sort();
        assert_eq!(sorted, expected, "sorted output must equal sorted(input)");
    }

    /// 16385-row non-power-of-two sort exercises the padding path: n_pow2 =
    /// 32768, with 16383 sentinel entries that must be truncated cleanly.
    #[test]
    #[ignore = "gpu:sort"]
    fn gpu_sort_int64_desc_with_padding() {
        let n = 16_385usize;
        let values: Vec<i64> = (0..n as i64).map(|i| (i * 7919) % 1_000_000).collect();
        let arr = Int64Array::from(values.clone());

        let keys = vec![GpuSortKey {
            column: &arr,
            dtype: DataType::Int64,
            direction: SortDirection::Desc,
            nulls_first: false,
        }];
        let (_layout, perm) = sort_indices_on_gpu_multi(&keys)
            .expect("gpu sort")
            .expect("non-fallback path on int64");
        assert_eq!(perm.len(), n, "output length must equal n_rows");

        let sorted: Vec<i64> = (0..n).map(|i| values[perm.value(i) as usize]).collect();
        for w in sorted.windows(2) {
            assert!(w[0] >= w[1], "DESC non-monotonic: {} < {}", w[0], w[1]);
        }
        // And the output must be a true permutation of the input.
        let mut expected: Vec<i64> = values.clone();
        expected.sort_by(|a, b| b.cmp(a));
        assert_eq!(sorted, expected);
    }

    /// Float64 ASC round trip with a non-power-of-two size.
    #[test]
    #[ignore = "gpu:sort"]
    fn gpu_sort_float64_asc_with_padding() {
        let n = 20_000usize;
        let values: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.61803398875).sin()).collect();
        let arr = Float64Array::from(values.clone());

        let keys = vec![GpuSortKey {
            column: &arr,
            dtype: DataType::Float64,
            direction: SortDirection::Asc,
            nulls_first: false,
        }];
        let (_layout, perm) = sort_indices_on_gpu_multi(&keys)
            .expect("gpu sort")
            .expect("non-fallback path on float64");
        assert_eq!(perm.len(), n);

        let sorted: Vec<f64> = (0..n).map(|i| values[perm.value(i) as usize]).collect();
        for w in sorted.windows(2) {
            assert!(
                w[0] <= w[1],
                "ASC float non-monotonic: {} > {}",
                w[0],
                w[1]
            );
        }
    }

    /// `sort_record_batch_on_gpu_multi` glues the index sort to the
    /// full-batch gather. Build a two-column batch (key + payload), sort by
    /// the key, and confirm the payload tracks.
    #[test]
    #[ignore = "gpu:sort"]
    fn gpu_sort_record_batch_keeps_columns_in_sync() {
        let n = 16_384usize;
        // Key = scrambled 0..n; payload = 100 + key. After sorting by key,
        // payload[i] should equal sorted_key[i] + 100.
        let mut keys: Vec<i32> = (0..n as i32).collect();
        let mut rng_state: u64 = 0xcafef00d;
        for i in (1..n).rev() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (rng_state as usize) % (i + 1);
            keys.swap(i, j);
        }
        let payload: Vec<i32> = keys.iter().map(|k| k + 100).collect();

        let schema = std::sync::Arc::new(Schema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("v", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                std::sync::Arc::new(Int32Array::from(keys.clone())),
                std::sync::Arc::new(Int32Array::from(payload.clone())),
            ],
        )
        .unwrap();

        let out = sort_record_batch_on_gpu_multi(
            &batch,
            &[(0, DataType::Int32, SortDirection::Asc, false)],
        )
        .expect("gpu sort batch")
        .expect("non-fallback path on int32 batch");
        assert_eq!(out.num_rows(), n);

        let k_sorted = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let v_sorted = out
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();

        for i in 0..n {
            assert_eq!(
                v_sorted.value(i),
                k_sorted.value(i) + 100,
                "payload must track key after sort"
            );
        }
        for i in 1..n {
            assert!(k_sorted.value(i - 1) <= k_sorted.value(i));
        }
    }

    // -- v0.7 GPU radix sort round-trips (single-key Int32/Int64 ASC). --
    //
    // These mirror the bitonic round-trip tests above but go through
    // `sort_indices_on_gpu_radix` instead of the bitonic multi-key
    // driver. Tagged `gpu:sort_radix` so an existing `gpu:sort` skip
    // doesn't accidentally hide a radix-only regression.

    /// End-to-end Int32 ASC sort via the radix path. Builds a 16k-row
    /// scrambled column, runs the radix driver, and asserts the
    /// permutation is correct.
    #[test]
    #[ignore = "gpu:sort_radix"]
    fn gpu_sort_radix_int32_asc_round_trip() {
        let n = 16_384usize;
        let mut values: Vec<i32> = (0..n as i32).collect();
        let mut rng_state: u64 = 0xfeedface;
        for i in (1..n).rev() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng_state as usize) % (i + 1);
            values.swap(i, j);
        }
        let arr = Int32Array::from(values.clone());

        let perm = sort_indices_on_gpu_radix(
            &arr,
            DataType::Int32,
            SortDirection::Asc,
            false,
        )
        .expect("gpu radix sort")
        .expect("non-fallback on int32 asc no-null");
        assert_eq!(perm.len(), n);

        let sorted: Vec<i32> = (0..n).map(|i| values[perm.value(i) as usize]).collect();
        for w in sorted.windows(2) {
            assert!(w[0] <= w[1], "radix non-monotonic: {} > {}", w[0], w[1]);
        }
        let mut expected = values.clone();
        expected.sort();
        assert_eq!(sorted, expected);
    }

    /// End-to-end Int64 ASC sort via the radix path. Same shape as the
    /// i32 test but with 16-pass coverage of the 64-bit key.
    #[test]
    #[ignore = "gpu:sort_radix"]
    fn gpu_sort_radix_int64_asc_round_trip() {
        let n = 16_384usize;
        // Use a range that includes negatives so the MSB-flip path is
        // exercised — without it the negatives would sort after positives.
        let values: Vec<i64> = (0..n as i64)
            .map(|i| ((i * 7919) % 200_000) - 100_000)
            .collect();
        let arr = Int64Array::from(values.clone());

        let perm = sort_indices_on_gpu_radix(
            &arr,
            DataType::Int64,
            SortDirection::Asc,
            false,
        )
        .expect("gpu radix sort i64")
        .expect("non-fallback on int64 asc no-null");
        assert_eq!(perm.len(), n);

        let sorted: Vec<i64> = (0..n).map(|i| values[perm.value(i) as usize]).collect();
        for w in sorted.windows(2) {
            assert!(
                w[0] <= w[1],
                "radix i64 non-monotonic: {} > {}",
                w[0],
                w[1]
            );
        }
        let mut expected = values.clone();
        expected.sort();
        assert_eq!(sorted, expected);
    }

    // -- #19 GPU radix sort round-trips — DESC + multi-key. --

    /// End-to-end Int32 DESC sort via the radix path. The host pre-transform
    /// applies `!val` so the device's unsigned ASC bit-pattern sort yields
    /// value-DESC.
    #[test]
    #[ignore = "gpu:sort_radix"]
    fn gpu_sort_radix_int32_desc_round_trip() {
        let n = 16_384usize;
        let mut values: Vec<i32> = (0..n as i32).map(|i| i - 8_192).collect(); // include negatives
        let mut rng_state: u64 = 0xabad_cafe;
        for i in (1..n).rev() {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (rng_state as usize) % (i + 1);
            values.swap(i, j);
        }
        let arr = Int32Array::from(values.clone());

        let perm = sort_indices_on_gpu_radix(
            &arr,
            DataType::Int32,
            SortDirection::Desc,
            false,
        )
        .expect("gpu radix sort desc")
        .expect("non-fallback on int32 desc no-null");
        assert_eq!(perm.len(), n);

        let sorted: Vec<i32> = (0..n).map(|i| values[perm.value(i) as usize]).collect();
        for w in sorted.windows(2) {
            assert!(
                w[0] >= w[1],
                "radix DESC non-monotonic: {} < {}",
                w[0],
                w[1]
            );
        }
        let mut expected = values.clone();
        expected.sort_by(|a, b| b.cmp(a));
        assert_eq!(sorted, expected);
    }

    /// Two-key ASC ASC lex sort via the radix path. Key A is the primary
    /// order, key B breaks ties. The LSD chain runs key B first (so its
    /// relative order is preserved by the stable key A pass).
    #[test]
    #[ignore = "gpu:sort_radix"]
    fn gpu_sort_radix_two_key_asc_asc_round_trip() {
        let n = 16_384usize;
        // Force lots of ties on key A so key B's tie-break is observable.
        let key_a: Vec<i32> = (0..n as i32).map(|i| i % 64).collect();
        let key_b: Vec<i32> = (0..n as i32).map(|i| (i * 31) % 1024).collect();
        let a_arr = Int32Array::from(key_a.clone());
        let b_arr = Int32Array::from(key_b.clone());

        let keys = [
            GpuSortKey {
                column: &a_arr,
                dtype: DataType::Int32,
                direction: SortDirection::Asc,
                nulls_first: false,
            },
            GpuSortKey {
                column: &b_arr,
                dtype: DataType::Int32,
                direction: SortDirection::Asc,
                nulls_first: false,
            },
        ];
        let perm = sort_indices_on_gpu_radix_multi(&keys)
            .expect("gpu radix multi")
            .expect("non-fallback on two-key int32 asc asc");
        assert_eq!(perm.len(), n);

        // Verify lex ASC ASC: (a[i], b[i]) <= (a[i+1], b[i+1]).
        for w in 0..(n - 1) {
            let ia = perm.value(w) as usize;
            let ib = perm.value(w + 1) as usize;
            let pa = (key_a[ia], key_b[ia]);
            let pb = (key_a[ib], key_b[ib]);
            assert!(pa <= pb, "lex non-monotonic: {:?} > {:?}", pa, pb);
        }
    }

    /// Two-key ASC DESC lex sort via the radix path. Key A is primary
    /// ASC; key B is secondary DESC. Exercises the per-key direction
    /// transform composition.
    #[test]
    #[ignore = "gpu:sort_radix"]
    fn gpu_sort_radix_two_key_asc_desc_round_trip() {
        let n = 16_384usize;
        let key_a: Vec<i32> = (0..n as i32).map(|i| i % 32).collect();
        let key_b: Vec<i32> = (0..n as i32).map(|i| (i * 17) % 1000).collect();
        let a_arr = Int32Array::from(key_a.clone());
        let b_arr = Int32Array::from(key_b.clone());

        let keys = [
            GpuSortKey {
                column: &a_arr,
                dtype: DataType::Int32,
                direction: SortDirection::Asc,
                nulls_first: false,
            },
            GpuSortKey {
                column: &b_arr,
                dtype: DataType::Int32,
                direction: SortDirection::Desc,
                nulls_first: false,
            },
        ];
        let perm = sort_indices_on_gpu_radix_multi(&keys)
            .expect("gpu radix multi")
            .expect("non-fallback on two-key int32 asc desc");
        assert_eq!(perm.len(), n);

        // For each adjacent pair: either a[i] < a[i+1] (primary ASC), or
        // a[i] == a[i+1] AND b[i] >= b[i+1] (secondary DESC tie-break).
        for w in 0..(n - 1) {
            let ia = perm.value(w) as usize;
            let ib = perm.value(w + 1) as usize;
            if key_a[ia] != key_a[ib] {
                assert!(
                    key_a[ia] < key_a[ib],
                    "primary ASC violated: a[{}]={} > a[{}]={}",
                    w, key_a[ia], w + 1, key_a[ib],
                );
            } else {
                assert!(
                    key_b[ia] >= key_b[ib],
                    "secondary DESC violated on tie a={}: b[{}]={} < b[{}]={}",
                    key_a[ia], w, key_b[ia], w + 1, key_b[ib],
                );
            }
        }
    }

    // -- #19 pure-host predicate / pre-transform tests --

    /// `radix_pre_transform_i32` ASC = `val ^ MSB`; bit-pattern unsigned
    /// ASC of the transformed values must match value-ASC of the originals.
    #[test]
    fn radix_pre_transform_i32_asc_orders_signed_int() {
        let mut samples: Vec<i32> = vec![i32::MIN, -5, -1, 0, 1, 5, i32::MAX];
        let mut transformed: Vec<u32> = samples
            .iter()
            .map(|v| radix_pre_transform_i32(*v, SortDirection::Asc) as u32)
            .collect();
        samples.sort();
        transformed.sort();
        // Apply the transform to the sorted samples and assert it equals
        // the sorted transformed bit-patterns — i.e. the transform is
        // order-preserving for ASC.
        let from_sorted: Vec<u32> = samples
            .iter()
            .map(|v| radix_pre_transform_i32(*v, SortDirection::Asc) as u32)
            .collect();
        assert_eq!(from_sorted, transformed);
    }

    /// `radix_pre_transform_i32` DESC = `!val`; bit-pattern unsigned ASC
    /// of the transformed values must match value-DESC of the originals.
    #[test]
    fn radix_pre_transform_i32_desc_orders_signed_int_descending() {
        let mut samples: Vec<i32> = vec![i32::MIN, -5, -1, 0, 1, 5, i32::MAX];
        let mut transformed: Vec<u32> = samples
            .iter()
            .map(|v| radix_pre_transform_i32(*v, SortDirection::Desc) as u32)
            .collect();
        // Sort samples DESC, then transform; result should match sorted
        // ASC of transformed (because the bit-not flips the order).
        samples.sort_by(|a, b| b.cmp(a));
        transformed.sort();
        let from_desc_sorted: Vec<u32> = samples
            .iter()
            .map(|v| radix_pre_transform_i32(*v, SortDirection::Desc) as u32)
            .collect();
        assert_eq!(from_desc_sorted, transformed);
    }

    /// Same property for i64.
    #[test]
    fn radix_pre_transform_i64_round_trips_for_both_directions() {
        let mut samples: Vec<i64> = vec![i64::MIN, -100_000, -1, 0, 1, 100_000, i64::MAX];
        // ASC.
        let mut transformed_asc: Vec<u64> = samples
            .iter()
            .map(|v| radix_pre_transform_i64(*v, SortDirection::Asc) as u64)
            .collect();
        transformed_asc.sort();
        let mut asc_samples = samples.clone();
        asc_samples.sort();
        let from_asc_sorted: Vec<u64> = asc_samples
            .iter()
            .map(|v| radix_pre_transform_i64(*v, SortDirection::Asc) as u64)
            .collect();
        assert_eq!(from_asc_sorted, transformed_asc);

        // DESC.
        let mut transformed_desc: Vec<u64> = samples
            .iter()
            .map(|v| radix_pre_transform_i64(*v, SortDirection::Desc) as u64)
            .collect();
        transformed_desc.sort();
        samples.sort_by(|a, b| b.cmp(a));
        let from_desc_sorted: Vec<u64> = samples
            .iter()
            .map(|v| radix_pre_transform_i64(*v, SortDirection::Desc) as u64)
            .collect();
        assert_eq!(from_desc_sorted, transformed_desc);
    }

    // -- F2 — IEEE-monotonic float radix key transform. Pure host. --

    /// `radix_float_key_f32` ASC: unsigned ascending order of the transformed
    /// keys must equal value-ascending order of the original floats, including
    /// negatives, signed zeros and infinities.
    #[test]
    fn radix_float_key_f32_asc_orders_floats() {
        let mut samples: Vec<f32> = vec![
            f32::NEG_INFINITY,
            -1e30,
            -1.5,
            -0.0,
            0.0,
            1.5,
            1e30,
            f32::INFINITY,
        ];
        let mut keys: Vec<u32> = samples
            .iter()
            .map(|&f| radix_float_key_f32(f, SortDirection::Asc))
            .collect();
        // Sort both independently; the transform must preserve the relation.
        keys.sort();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let from_sorted: Vec<u32> = samples
            .iter()
            .map(|&f| radix_float_key_f32(f, SortDirection::Asc))
            .collect();
        assert_eq!(from_sorted, keys);
        // The keys must be strictly increasing (no two distinct values collide;
        // -0.0/+0.0 differ in one ulp here so they stay distinct, which is fine
        // for a stable sort — equal *values* are not in this sample).
        for w in from_sorted.windows(2) {
            assert!(w[0] < w[1], "f32 asc keys not strictly increasing");
        }
    }

    /// `radix_float_key_f32` DESC is the bitwise-NOT of the ASC key, so its
    /// unsigned ascending order equals value-descending order.
    #[test]
    fn radix_float_key_f32_desc_reverses_order() {
        let samples: Vec<f32> = vec![-2.0, -0.5, 0.0, 0.5, 2.0];
        for w in samples.windows(2) {
            // w[0] < w[1] in value, so DESC key of w[0] must be GREATER.
            let ka = radix_float_key_f32(w[0], SortDirection::Desc);
            let kb = radix_float_key_f32(w[1], SortDirection::Desc);
            assert!(ka > kb, "f32 desc must invert order: {} vs {}", w[0], w[1]);
        }
    }

    /// Same ASC ordering property for f64.
    #[test]
    fn radix_float_key_f64_asc_orders_floats() {
        let mut samples: Vec<f64> = vec![
            f64::NEG_INFINITY,
            -1e300,
            -3.25,
            -0.0,
            0.0,
            3.25,
            1e300,
            f64::INFINITY,
        ];
        let mut keys: Vec<u64> = samples
            .iter()
            .map(|&f| radix_float_key_f64(f, SortDirection::Asc))
            .collect();
        keys.sort();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let from_sorted: Vec<u64> = samples
            .iter()
            .map(|&f| radix_float_key_f64(f, SortDirection::Asc))
            .collect();
        assert_eq!(from_sorted, keys);
        for w in from_sorted.windows(2) {
            assert!(w[0] < w[1], "f64 asc keys not strictly increasing");
        }
    }

    // -- F2 — stable multi-block radix offset construction. Pure host. --
    //
    // `compute_block_offsets` is the load-bearing piece of the cross-block
    // stable radix scatter: it turns a per-block-per-digit histogram into the
    // per-block-per-digit *exclusive* output offsets the scatter kernel reads.
    // These tests simulate the scatter deterministically on the host (each
    // block lays its elements down at `base[digit] + per_block_rank` in tid
    // order) and assert the result is a stable counting-sort by digit — the
    // exact invariant LSD radix needs each pass to satisfy.

    /// A single block degenerates to a plain stable counting sort: offsets are
    /// the exclusive prefix sum of the digit totals.
    #[test]
    fn compute_block_offsets_single_block_is_exclusive_scan() {
        let buckets = RADIX_BUCKETS as usize;
        // One block, counts: digit 0 -> 3, digit 1 -> 0, digit 2 -> 5, rest 0.
        let mut hist = vec![0u32; buckets];
        hist[0] = 3;
        hist[2] = 5;
        let mut offs = vec![0u32; buckets];
        compute_block_offsets(&hist, 1, &mut offs).unwrap();
        // Exclusive scan: 0 starts at 0, 1 starts at 3, 2 starts at 3, 3+ at 8.
        assert_eq!(offs[0], 0);
        assert_eq!(offs[1], 3);
        assert_eq!(offs[2], 3);
        for d in 3..buckets {
            assert_eq!(offs[d], 8, "empty trailing digits all start past the data");
        }
    }

    /// Two blocks: block 0's run for a digit must precede block 1's run for the
    /// same digit (cross-block stability), and the per-digit regions must be
    /// laid out in digit order with no gaps or overlaps.
    #[test]
    fn compute_block_offsets_two_blocks_orders_runs_by_block_then_digit() {
        let buckets = RADIX_BUCKETS as usize;
        // block 0: digit0=2, digit1=1 ; block 1: digit0=1, digit1=3.
        let mut hist = vec![0u32; 2 * buckets];
        hist[0 * buckets + 0] = 2;
        hist[0 * buckets + 1] = 1;
        hist[1 * buckets + 0] = 1;
        hist[1 * buckets + 1] = 3;
        let mut offs = vec![0u32; 2 * buckets];
        compute_block_offsets(&hist, 2, &mut offs).unwrap();
        // digit 0 region [0,3): block0 at 0 (len 2), block1 at 2 (len 1).
        assert_eq!(offs[0 * buckets + 0], 0);
        assert_eq!(offs[1 * buckets + 0], 2);
        // digit 1 region [3,7): block0 at 3 (len 1), block1 at 4 (len 3).
        assert_eq!(offs[0 * buckets + 1], 3);
        assert_eq!(offs[1 * buckets + 1], 4);
    }

    /// End-to-end host simulation: drive `compute_block_offsets` exactly as the
    /// kernel would, then verify the scatter is a *stable* sort by digit across
    /// many blocks with heavy digit collisions. This is the property the GPU
    /// kernel relies on for LSD correctness.
    #[test]
    fn compute_block_offsets_simulated_scatter_is_stable() {
        let buckets = RADIX_BUCKETS as usize;
        let block = RADIX_BLOCK_SIZE as usize;
        let n = 3 * block + 37; // 3 full blocks + a partial tail.
        let num_blocks = (n + block - 1) / block;

        // Deterministic pseudo-random digits in 0..16.
        let digits: Vec<u32> = (0..n)
            .map(|i| (((i as u64).wrapping_mul(2654435761) >> 13) & 0xF) as u32)
            .collect();

        // Build the per-block-per-digit histogram exactly like the kernel.
        let mut hist = vec![0u32; num_blocks * buckets];
        for (i, &d) in digits.iter().enumerate() {
            let b = i / block;
            hist[b * buckets + d as usize] += 1;
        }

        let mut offs = vec![0u32; num_blocks * buckets];
        compute_block_offsets(&hist, num_blocks, &mut offs).unwrap();

        // Simulate the scatter: each block places its rows at
        // base[digit] + (count of same-digit rows earlier in the block).
        let mut out = vec![u32::MAX; n]; // out[pos] = source row index
        for b in 0..num_blocks {
            let mut local_rank = [0u32; RADIX_BUCKETS as usize];
            let lo = b * block;
            let hi = (lo + block).min(n);
            for i in lo..hi {
                let d = digits[i] as usize;
                let pos = offs[b * buckets + d] + local_rank[d];
                assert_eq!(out[pos as usize], u32::MAX, "two rows claimed slot {pos}");
                out[pos as usize] = i as u32;
                local_rank[d] += 1;
            }
        }

        // Every slot filled exactly once.
        assert!(out.iter().all(|&x| x != u32::MAX), "scatter left a hole");

        // Output digits are non-decreasing (counting-sorted by digit).
        for w in out.windows(2) {
            assert!(
                digits[w[0] as usize] <= digits[w[1] as usize],
                "scatter not sorted by digit",
            );
        }
        // Stability: within each digit, source row indices stay ascending.
        let mut last_for_digit = [None::<u32>; RADIX_BUCKETS as usize];
        for &src in &out {
            let d = digits[src as usize] as usize;
            if let Some(prev) = last_for_digit[d] {
                assert!(prev < src, "unstable: digit {d} saw {src} after {prev}");
            }
            last_for_digit[d] = Some(src);
        }
    }

    // -- Stage 5 — high-cardinality sampling gate. Pure host, no CUDA. --

    /// Smoke test for the threshold constants. Keeps the documented values
    /// (and their ordering) under review in case someone tunes one without
    /// the other.
    #[test]
    fn stage5_threshold_constants_within_documented_range() {
        assert!(
            HIGH_CARDINALITY_THRESHOLD > 0.0 && HIGH_CARDINALITY_THRESHOLD < 1.0,
            "threshold must be a strict fraction"
        );
        assert!(
            HIGH_CARDINALITY_SAMPLE_MAX >= 64,
            "sample cap must be large enough that ratio noise stays low"
        );
        assert!(
            HIGH_CARDINALITY_SAMPLE_STRIDE_DIV >= 2,
            "stride divisor of 1 would scan the whole column — defeats the sampler"
        );
    }

    /// `sample_distinct_utf8` on a low-cardinality column reports few
    /// distinct values. 1024 rows cycling through 16 strings — every other
    /// stride lands on the same residue class, so distinct count stays
    /// bounded by 16 regardless of stride.
    #[test]
    fn sample_distinct_utf8_low_cardinality() {
        let n = 1024usize;
        let alphabet = [
            "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        ];
        let strings: Vec<&str> = (0..n).map(|i| alphabet[i % alphabet.len()]).collect();
        let sa = StringArray::from(strings);
        let (sample, distinct) = sample_distinct_utf8(&sa);
        assert!(sample > 0, "sample must be non-empty for n_rows=1024");
        assert!(
            distinct <= 16,
            "low-cardinality column should sample at most 16 distinct values; got {distinct}"
        );
        let ratio = (distinct as f64) / (sample as f64);
        assert!(
            ratio < HIGH_CARDINALITY_THRESHOLD,
            "low-cardinality ratio {ratio} must be under the gate threshold {}",
            HIGH_CARDINALITY_THRESHOLD
        );
    }

    /// `sample_distinct_utf8` on an all-unique column reports near-100%
    /// distinct ratio. Every row a distinct string — sampler picks
    /// `sample_size` rows and finds `sample_size` distinct values.
    #[test]
    fn sample_distinct_utf8_high_cardinality() {
        let n = 1024usize;
        let strings: Vec<String> = (0..n).map(|i| format!("row_{i:08}")).collect();
        let sa = StringArray::from(strings);
        let (sample, distinct) = sample_distinct_utf8(&sa);
        assert!(sample > 0);
        assert_eq!(
            distinct, sample,
            "all-unique column: every sampled row should be a distinct value"
        );
        let ratio = (distinct as f64) / (sample as f64);
        assert!(
            ratio > HIGH_CARDINALITY_THRESHOLD,
            "all-unique ratio {ratio} must exceed the gate threshold {}",
            HIGH_CARDINALITY_THRESHOLD
        );
    }

    /// `host_values_for_key` returns `Ok(None)` for a high-cardinality Utf8
    /// column. The 1024-distinct-row fixture pushes the sampler well past
    /// `HIGH_CARDINALITY_THRESHOLD`, so the GPU path must abort.
    #[test]
    fn high_cardinality_utf8_falls_through_to_host() {
        let n = 1024usize;
        let strings: Vec<String> = (0..n).map(|i| format!("payload_{i:010}")).collect();
        let sa = StringArray::from(strings);
        let res = host_values_for_key(&sa, DataType::Int32)
            .expect("host_values_for_key must not error on plain Utf8");
        assert!(
            res.is_none(),
            "high-cardinality Utf8 (1024 distinct strings) must trigger the Stage-5 fall-through"
        );
    }

    /// `host_values_for_key` returns `Ok(Some(_))` for a low-cardinality
    /// Utf8 column — the inline dictionary builder is still the right path.
    #[test]
    fn low_cardinality_utf8_takes_gpu_path() {
        let n = 1024usize;
        // 16 distinct strings cycled. The sampler steps by `n/target = 16`
        // (target = n/stride_div = 64), so it lands on the same residue
        // class — distinct comes out as 1, ratio ≈ 0.016, well under the
        // threshold. Other low-cardinality layouts (random shuffle) would
        // sample a few distinct values; either way the ratio stays low.
        let alphabet = [
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
            "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
        ];
        let strings: Vec<&str> = (0..n).map(|i| alphabet[i % alphabet.len()]).collect();
        let sa = StringArray::from(strings);
        let res = host_values_for_key(&sa, DataType::Int32)
            .expect("host_values_for_key must not error on plain Utf8");
        let values = res.expect("low-cardinality Utf8 must take the GPU path");
        // Sanity-check the variant: low-cardinality Utf8 routes through the
        // i32 numeric kernel.
        assert!(matches!(values, HostKeyValues::I32(_)));
        // And the indices buffer has one entry per row.
        if let HostKeyValues::I32(v) = values {
            assert_eq!(v.len(), n);
        }
    }

    /// Edge case: empty StringArray. `sample_distinct_utf8` returns `(0, 0)`
    /// and the gate must NOT fire (no rows to sort means no dictionary work
    /// to gate — return the empty `Some(I32(empty))`).
    #[test]
    fn empty_utf8_does_not_trip_gate() {
        let sa = StringArray::from(Vec::<&str>::new());
        let res = host_values_for_key(&sa, DataType::Int32)
            .expect("empty Utf8 must not error");
        let values = res.expect("empty Utf8 must NOT trip the high-cardinality gate");
        assert!(matches!(values, HostKeyValues::I32(_)));
    }
}

// ============================================================================
// Batch 6 — CUDA Graph cache tests (host-only, no GPU).
//
// We exercise the cache-key shape, the env-var parser, and the dtype-tag
// helper directly. The GRAPH_CACHE itself stores raw `CUgraphExec`
// pointers, but for these host tests we forge non-null sentinel pointers
// and treat the cache as a plain map — none of the test cases dereference
// the handles, so the driver is never touched.
// ============================================================================
#[cfg(test)]
mod graph_cache_tests {
    use super::*;
    use parking_lot::Mutex;

    /// Process-wide gate so concurrent tests don't race on the shared
    /// GRAPH_CACHE / env-var state. Mirrors the pattern used by
    /// `init_cache_tests` in `cuda_sys.rs`.
    static TEST_GATE: Mutex<()> = Mutex::new(());

    /// `dtype_tag` must produce distinct tags for every GPU-sortable
    /// dtype so two different-typed sorts on the same buffer pointer can
    /// never alias their cached graphs.
    #[test]
    fn dtype_tag_is_injective_over_sortable_set() {
        let tags = [
            dtype_tag(DataType::Int32),
            dtype_tag(DataType::Int64),
            dtype_tag(DataType::Float32),
            dtype_tag(DataType::Float64),
            dtype_tag(DataType::Bool),
        ];
        let mut seen: std::collections::HashSet<u8> = std::collections::HashSet::new();
        for t in tags.iter() {
            assert!(
                seen.insert(*t),
                "dtype_tag must be injective; collision on tag={t}"
            );
            assert_ne!(*t, 255, "unknown-dtype catch-all must not alias a real tag");
        }
    }

    /// `sort_uses_graph()` returns `true` *only* for the exact string
    /// "1". Any other value (including "0", "true", garbage, unset)
    /// returns false — strict equality keeps the gate from tripping on
    /// shell-quoting surprises.
    #[test]
    fn env_var_parsing_strict_equals_one() {
        let _g = TEST_GATE.lock();
        // SAFETY: we hold TEST_GATE; no concurrent test mutates this env var.
        std::env::remove_var(BOLT_SORT_USE_GRAPH_ENV);
        assert!(!sort_uses_graph(), "unset must be off");

        std::env::set_var(BOLT_SORT_USE_GRAPH_ENV, "1");
        assert!(sort_uses_graph(), "\"1\" must be on");

        std::env::set_var(BOLT_SORT_USE_GRAPH_ENV, "0");
        assert!(!sort_uses_graph(), "\"0\" must be off");

        std::env::set_var(BOLT_SORT_USE_GRAPH_ENV, "true");
        assert!(!sort_uses_graph(), "\"true\" must be off — only \"1\" wins");

        std::env::set_var(BOLT_SORT_USE_GRAPH_ENV, "yes");
        assert!(!sort_uses_graph(), "\"yes\" must be off");

        // Windows API rejects NUL bytes in env-var values; use a pure
        // garbage string instead — both POSIX and Windows accept it and
        // it exercises the same "not a recognised truthy value" branch.
        std::env::set_var(BOLT_SORT_USE_GRAPH_ENV, "garbage-bytes");
        assert!(!sort_uses_graph(), "unparseable garbage must be off");

        // Restore env to a known-empty state for any follow-up tests.
        std::env::remove_var(BOLT_SORT_USE_GRAPH_ENV);
    }

    /// Synthetic insert/lookup against the GRAPH_CACHE. Confirms keys
    /// differ across `(n_pow2, dtype, idx_ptr, keys_fp)` permutations and
    /// that lookups round-trip.
    ///
    /// The cache stores raw `CUgraphExec` pointers; we forge non-null
    /// sentinels here and never dereference them. The test clears the
    /// cache on entry and exit so it doesn't disturb any concurrent
    /// (real-GPU, `#[ignore]`-gated) tests.
    #[test]
    fn cache_hit_miss_accounting_synthetic() {
        let _g = TEST_GATE.lock();
        _test_clear_graph_cache();

        // Forge three distinct fake CUgraphExec values. The cache only
        // sees them as opaque pointers — never dereferenced in this test.
        let fake_a: CUgraphExec = 0xA000_0000_usize as CUgraphExec;
        let fake_b: CUgraphExec = 0xB000_0000_usize as CUgraphExec;
        let fake_c: CUgraphExec = 0xC000_0000_usize as CUgraphExec;

        let key_a: GraphCacheKey = (1024, dtype_tag(DataType::Int32), 0x1111, 0x2222);
        let key_b: GraphCacheKey = (1024, dtype_tag(DataType::Int64), 0x1111, 0x2222);
        let key_c: GraphCacheKey = (2048, dtype_tag(DataType::Int32), 0x1111, 0x2222);

        // Miss on every key (cache was just cleared).
        {
            let cache = GRAPH_CACHE.lock();
            assert!(cache.get(&key_a).is_none(), "key_a must miss initially");
            assert!(cache.get(&key_b).is_none(), "key_b must miss initially");
            assert!(cache.get(&key_c).is_none(), "key_c must miss initially");
        }

        // Insert.
        {
            let mut cache = GRAPH_CACHE.lock();
            cache.insert(key_a, GraphExecHandle(fake_a));
            cache.insert(key_b, GraphExecHandle(fake_b));
            cache.insert(key_c, GraphExecHandle(fake_c));
        }

        // Lookup must return the same handle we inserted, keyed
        // independently — dtype_tag and n_pow2 must NOT alias.
        {
            let cache = GRAPH_CACHE.lock();
            assert_eq!(cache.get(&key_a).map(|h| h.0), Some(fake_a));
            assert_eq!(cache.get(&key_b).map(|h| h.0), Some(fake_b));
            assert_eq!(cache.get(&key_c).map(|h| h.0), Some(fake_c));
            assert_eq!(cache.len(), 3, "three distinct keys → three entries");
        }

        // Distinct idx_ptr must miss even when (n_pow2, dtype, keys_fp)
        // match an existing entry — this is the load-bearing property
        // that protects us from the "caller passes a different GpuVec"
        // failure mode documented at the top of the module.
        let key_other_ptr: GraphCacheKey = (1024, dtype_tag(DataType::Int32), 0x3333, 0x2222);
        {
            let cache = GRAPH_CACHE.lock();
            assert!(
                cache.get(&key_other_ptr).is_none(),
                "different idx_ptr must NOT hit an existing entry"
            );
        }

        // Distinct keys_fp must miss too — same n_pow2/dtype/idx_ptr but
        // a different key-buffer fingerprint means the captured key
        // pointers no longer match what the caller will pass to the
        // graph at launch time.
        let key_other_fp: GraphCacheKey = (1024, dtype_tag(DataType::Int32), 0x1111, 0xDEAD);
        {
            let cache = GRAPH_CACHE.lock();
            assert!(
                cache.get(&key_other_fp).is_none(),
                "different keys fingerprint must NOT hit an existing entry"
            );
        }

        _test_clear_graph_cache();
    }

    /// The `BOLT_SORT_USE_GRAPH` gate determines whether the MultiLaunch
    /// path even consults the cache. With the gate off, the cache stays
    /// untouched and the legacy code path runs — confirm via the env-var
    /// parser since we can't observe the launch sequence without a GPU.
    #[test]
    fn gate_off_leaves_cache_untouched() {
        let _g = TEST_GATE.lock();
        _test_clear_graph_cache();
        std::env::remove_var(BOLT_SORT_USE_GRAPH_ENV);
        assert!(!sort_uses_graph(), "gate must be off when env unset");
        // Cache stays empty: no test path inserts into it without a real
        // launch, which the host-only test deliberately avoids.
        let len = GRAPH_CACHE.lock().len();
        assert_eq!(len, 0, "no GPU launch happened; cache must be empty");
    }
}
