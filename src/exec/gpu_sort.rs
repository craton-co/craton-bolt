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
//! ## Stage 1 scope
//!
//! - Single sort key. Multi-key (lexicographic) is `TODO(s1-stage2)`.
//! - Dtype: Int32, Int64, Float32, Float64. (Bool / Utf8 stay host-side.)
//! - ASC and DESC. NULLs `null_count() == 0` only — `TODO(s1-stage2)` to plumb
//!   a parallel validity buffer through the comparator.
//! - `n_rows <= u32::MAX` and the padded `n_pow2` must fit too (so practical
//!   limit is `n_rows <= 2^31` since `n_pow2 = next_pow2(n_rows) <= 2^32`).
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
//! Real-data ties with the sentinel value are not a correctness issue: bitonic
//! sort is stable enough for our purposes — equal keys may end up in any
//! order relative to each other, but the padded indices (>= n_rows) are
//! filtered out by the final truncation step, never returned to the caller.

use std::ffi::c_void;
use std::ptr;

use arrow_array::{
    Array, ArrayRef, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array,
    Int64Array, RecordBatch, UInt32Array,
};
use arrow_array::types::{Int32Type, Int64Type};
use arrow::compute::take;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;
use crate::exec::n_rows_to_u32;
use crate::jit::jit_compiler::CudaModule;
use crate::jit::sort_kernel::{
    compile_sort_kernel, compile_sort_kernel_spec, sort_kernel_entry,
    sort_kernel_entry_spec, KeyDesc, SortDirection, SortKernelSpec, SortLayout,
    MAX_SORT_KEYS, SORT_BLOCK_SIZE,
};
use crate::plan::logical_plan::DataType;

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

/// Launch the `log2(n_pow2) * (log2(n_pow2)+1) / 2` bitonic substages over
/// `(keys_dev, idx_dev)`. After this call, `keys_dev` and `idx_dev` are both
/// permuted such that `keys_dev` is sorted in `dir` order and `idx_dev` is
/// the corresponding permutation of the identity 0..n_pow2.
///
/// The two device buffers must have exactly `n_pow2` elements each.
#[allow(dead_code)] // reason: Stage-1 single-key bitonic kept for golden-test surface; multi-key driver now dispatches all sorts.
fn run_bitonic_passes(
    keys_dev_ptr: CUdeviceptr,
    idx_dev_ptr: CUdeviceptr,
    n_pow2: u32,
    dtype: DataType,
    dir: SortDirection,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_pow2 <= 1 {
        // A single-element sort is a no-op. The padded buffer of length 1 is
        // already trivially sorted.
        return Ok(());
    }

    let ptx = compile_sort_kernel(dtype, dir)?;
    let module = CudaModule::from_ptx(&ptx)?;
    let entry = sort_kernel_entry(dtype, dir)?;
    let function = module.function(entry)?;

    let log2_n = log2_pow2(n_pow2);

    // Launch grid: one thread per element, block size 256 (matches the rest
    // of the engine).
    let block_size: u32 = SORT_BLOCK_SIZE;
    let grid_x: u32 = n_pow2.div_ceil(block_size);

    // TODO(s1-stage2): coalesce in-block substages into a single shared-mem
    // kernel launch (~5-10x fewer launches for n_pow2 <= block_size pairs).

    // Pre-bind the device-pointer args; only stage + substage_mask change
    // per launch.
    let mut p_keys: CUdeviceptr = keys_dev_ptr;
    let mut p_idx: CUdeviceptr = idx_dev_ptr;
    let mut p_n_pow2: u32 = n_pow2;

    // Outer loop: stages 1..=log2(n).
    for stage in 1..=log2_n {
        // Inner loop: substages stage..=1, in decreasing order. We pass the
        // substage MASK (1 << (substage - 1)) to the kernel so it can XOR
        // without an extra shift.
        let mut substage = stage;
        loop {
            let substage_mask: u32 = 1u32 << (substage - 1);
            let mut p_stage: u32 = stage;
            let mut p_mask: u32 = substage_mask;

            let mut kernel_params: [*mut c_void; 5] = [
                &mut p_keys as *mut CUdeviceptr as *mut c_void,
                &mut p_idx as *mut CUdeviceptr as *mut c_void,
                &mut p_n_pow2 as *mut u32 as *mut c_void,
                &mut p_stage as *mut u32 as *mut c_void,
                &mut p_mask as *mut u32 as *mut c_void,
            ];

            // SAFETY: every entry in `kernel_params` points at a stack local
            // that outlives the launch+synchronize below; `function` is borrowed
            // from a live `CudaModule`; the keys and indices buffers are owned
            // by the caller and outlive every launch in this loop.
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
                    kernel_params.as_mut_ptr(),
                    ptr::null_mut(),
                ))?;
            }
            // Synchronise between substages: the next substage reads what
            // this substage wrote. (Stage 1: one global sync per substage.
            // Stage 2: amortise via shared-memory in-block bitonic.)
            stream.synchronize()?;

            if substage == 1 {
                break;
            }
            substage -= 1;
        }
    }

    Ok(())
}

/// Upload a typed key column, sort it on the GPU, and download the resulting
/// permutation indices (truncated to `n_rows`).
///
/// This is the heart of the GPU ORDER BY fast path. The four type parameters
/// have explicit branches so the compiler can monomorphise the host-side
/// upload/download — the actual sort kernel is dtype-aware on the device.
#[allow(dead_code)] // reason: Stage-1 single-key entry kept for golden-test surface; sort_indices_on_gpu_multi now dispatches.
pub fn sort_indices_on_gpu(
    column: &dyn Array,
    dtype: DataType,
    dir: SortDirection,
) -> BoltResult<UInt32Array> {
    let n_rows = column.len();
    if n_rows == 0 {
        return Ok(UInt32Array::from(Vec::<u32>::new()));
    }
    if n_rows > (u32::MAX as usize) {
        return Err(BoltError::Other(format!(
            "gpu_sort: n_rows {} exceeds u32::MAX",
            n_rows
        )));
    }

    let n_pow2 = next_pow2_u32(n_rows)?;
    let n_pow2_usize = n_pow2 as usize;

    let stream = CudaStream::null();

    // Build the identity index vector and upload it; same for the (padded)
    // keys vector. The per-dtype branches differ only in the sentinel and
    // the GpuVec element type.
    let idx_host: Vec<u32> = (0..n_pow2).collect();
    let idx_dev = GpuVec::<u32>::from_slice(&idx_host)?;

    match dtype {
        DataType::Int32 => {
            let arr = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_sort: column dtype said Int32 but downcast failed".into())
                })?;
            let host: Vec<i32> = arr.values().as_ref().to_vec();
            let sentinel = match dir {
                SortDirection::Asc => i32::MAX,
                SortDirection::Desc => i32::MIN,
            };
            let padded = pad_to_pow2(&host, n_pow2_usize, sentinel);
            let keys_dev = GpuVec::<i32>::from_slice(&padded)?;
            run_bitonic_passes(
                keys_dev.device_ptr(),
                idx_dev.device_ptr(),
                n_pow2,
                dtype,
                dir,
                &stream,
            )?;
            drop(keys_dev);
        }
        DataType::Int64 => {
            let arr = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_sort: column dtype said Int64 but downcast failed".into())
                })?;
            let host: Vec<i64> = arr.values().as_ref().to_vec();
            let sentinel = match dir {
                SortDirection::Asc => i64::MAX,
                SortDirection::Desc => i64::MIN,
            };
            let padded = pad_to_pow2(&host, n_pow2_usize, sentinel);
            let keys_dev = GpuVec::<i64>::from_slice(&padded)?;
            run_bitonic_passes(
                keys_dev.device_ptr(),
                idx_dev.device_ptr(),
                n_pow2,
                dtype,
                dir,
                &stream,
            )?;
            drop(keys_dev);
        }
        DataType::Float32 => {
            let arr = column
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    BoltError::Other(
                        "gpu_sort: column dtype said Float32 but downcast failed".into(),
                    )
                })?;
            let host: Vec<f32> = arr.values().as_ref().to_vec();
            let sentinel = match dir {
                SortDirection::Asc => f32::INFINITY,
                SortDirection::Desc => f32::NEG_INFINITY,
            };
            let padded = pad_to_pow2(&host, n_pow2_usize, sentinel);
            let keys_dev = GpuVec::<f32>::from_slice(&padded)?;
            run_bitonic_passes(
                keys_dev.device_ptr(),
                idx_dev.device_ptr(),
                n_pow2,
                dtype,
                dir,
                &stream,
            )?;
            drop(keys_dev);
        }
        DataType::Float64 => {
            let arr = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    BoltError::Other(
                        "gpu_sort: column dtype said Float64 but downcast failed".into(),
                    )
                })?;
            let host: Vec<f64> = arr.values().as_ref().to_vec();
            let sentinel = match dir {
                SortDirection::Asc => f64::INFINITY,
                SortDirection::Desc => f64::NEG_INFINITY,
            };
            let padded = pad_to_pow2(&host, n_pow2_usize, sentinel);
            let keys_dev = GpuVec::<f64>::from_slice(&padded)?;
            run_bitonic_passes(
                keys_dev.device_ptr(),
                idx_dev.device_ptr(),
                n_pow2,
                dtype,
                dir,
                &stream,
            )?;
            drop(keys_dev);
        }
        _ => {
            return Err(BoltError::Other(format!(
                "gpu_sort: dtype {:?} not supported (Stage 1: Int32/Int64/Float32/Float64)",
                dtype
            )))
        }
    }

    // Download the sorted indices and truncate to n_rows.
    //
    // After the bitonic sort the layout is:
    //   - Real rows (in sorted order)         : the n_rows entries whose index < n_rows
    //   - Padded rows (sentinel-valued)       : the (n_pow2 - n_rows) entries whose index >= n_rows
    //
    // For ASC: real rows < sentinel = MAX, so real rows are at positions
    //   0..n_rows and padded rows are at positions n_rows..n_pow2. Truncate
    //   the tail.
    // For DESC: real rows > sentinel = MIN (treating -INF as "smallest"),
    //   so real rows are at positions 0..n_rows. Same truncation.
    //
    // Either way the first n_rows indices are the answer. Filter any index
    // that's >= n_rows as a defensive safety net (would only happen if a
    // real key tied the sentinel, which is a precision-loss corner case
    // for ASC + i32::MAX-valued real data; in that case we silently drop
    // that row from the output — TODO(s1-stage2): handle sentinel ties
    // via an explicit "is_padded" parallel mask).
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
            "gpu_sort: post-sort recovered only {} indices for {} real rows \
             (real keys probably collided with the padding sentinel)",
            out.len(),
            n_rows
        )));
    }

    Ok(UInt32Array::from(out))
}

/// Sort `batch` by a single key column on the GPU, returning a freshly built
/// `RecordBatch` whose rows are permuted accordingly.
///
/// `key_idx` is the column index of the sort key within `batch`. The caller
/// is expected to have already validated the precondition gates (single key,
/// supported dtype, n_rows threshold, no NULLs); this entry point just
/// executes the sort and the gather.
#[allow(dead_code)] // reason: Stage-1 single-key entry kept for golden-test surface; sort_record_batch_on_gpu_multi now dispatches.
pub fn sort_record_batch_on_gpu(
    batch: &RecordBatch,
    key_idx: usize,
    dtype: DataType,
    dir: SortDirection,
) -> BoltResult<RecordBatch> {
    let key_col = batch.column(key_idx);
    let perm = sort_indices_on_gpu(key_col.as_ref(), dtype, dir)?;

    let new_cols: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| {
            take(c.as_ref(), &perm, None).map_err(|e| {
                BoltError::Other(format!("gpu_sort: arrow take failed: {e}"))
            })
        })
        .collect::<BoltResult<Vec<_>>>()?;

    RecordBatch::try_new(batch.schema(), new_cols)
        .map_err(|e| BoltError::Other(format!("gpu_sort: building RecordBatch failed: {e}")))
}

/// Arrow `DataType` -> our internal `DataType`. Returns `None` if the column
/// type isn't one of the GPU-sortable kinds (the caller falls through to
/// the host-side sort).
///
/// **Stage 3** additions:
///   - `Boolean` -> `Bool` (loaded as u8, compared as s32 0/1).
///   - `Dictionary(I32 | I64, Utf8)` -> the index dtype (Int32 / Int64). The
///     dictionary's *values* are immaterial for the sort: the indices alone
///     induce the lex order the dictionary was built with. Non-dict Utf8
///     keys stay on the host path; encoder is responsible for converting if
///     they want the GPU win — see the Stage 4 follow-up note.
pub fn arrow_dtype_to_internal(d: &arrow_schema::DataType) -> Option<DataType> {
    use arrow_schema::DataType as A;
    match d {
        A::Int32 => Some(DataType::Int32),
        A::Int64 => Some(DataType::Int64),
        A::Float32 => Some(DataType::Float32),
        A::Float64 => Some(DataType::Float64),
        A::Boolean => Some(DataType::Bool),
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

/// Extract a numeric "host view" from a sortable Arrow column. Stage 3
/// addition: handles Bool (-> u8 0/1 widened to i32) and dictionary-encoded
/// Utf8 (-> index column as i32 or i64). For everything else this is a
/// straight `.values().to_vec()`.
fn host_values_for_key(arr: &dyn Array, dtype: DataType) -> BoltResult<HostKeyValues> {
    use arrow_schema::DataType as A;
    Ok(match (dtype, arr.data_type()) {
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
        // Stage 3 dictionary-Utf8 adapter: read the dictionary's index column
        // (`Int32` or `Int64`); the dictionary values themselves never reach
        // the GPU sort. The output permutation is then applied (host-side)
        // to the dictionary-encoded column intact, which keeps the
        // values-dictionary edge alive without re-encoding.
        (DataType::Int32, A::Dictionary(key_ty, _)) if matches!(key_ty.as_ref(), A::Int32) => {
            let da = arr
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_sort: dict<i32,utf8> downcast failed".into())
                })?;
            HostKeyValues::I32(da.keys().values().as_ref().to_vec())
        }
        (DataType::Int64, A::Dictionary(key_ty, _)) if matches!(key_ty.as_ref(), A::Int64) => {
            let da = arr
                .as_any()
                .downcast_ref::<DictionaryArray<Int64Type>>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_sort: dict<i64,utf8> downcast failed".into())
                })?;
            HostKeyValues::I64(da.keys().values().as_ref().to_vec())
        }
        (dt, arrow_dt) => {
            return Err(BoltError::Other(format!(
                "gpu_sort: dtype/array mismatch ({:?} vs Arrow {:?})",
                dt, arrow_dt
            )))
        }
    })
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
) -> BoltResult<KeyDeviceBuf> {
    let values = host_values_for_key(arr, dtype)?;
    Ok(match values {
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
    })
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

/// Threshold below which we emit the in-block shmem variant. Equal to
/// `SORT_BLOCK_SIZE` because the shmem variant requires `n_pow2 <=
/// block_size` (each thread owns one element). Anything bigger goes through
/// the multi-launch loop.
const SHMEM_VARIANT_MAX_NPOW2: u32 = SORT_BLOCK_SIZE;

/// Sort `keys` (up to [`MAX_SORT_KEYS`]) lexicographically on the GPU and
/// return the row permutation as a `UInt32Array` of length `n_rows`.
///
/// Picks between the multi-launch and shmem-variant kernels based on
/// `n_pow2`. For `n_pow2 <= SORT_BLOCK_SIZE` the shmem variant runs as a
/// single launch; above that we fall back to one launch per substage.
///
/// Returns the (taken_path, permutation) pair so callers/tests can verify
/// the dispatch decision without observing it through a counter.
pub fn sort_indices_on_gpu_multi<'a>(
    keys: &[GpuSortKey<'a>],
) -> BoltResult<(SortLayout, UInt32Array)> {
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
        return Ok((SortLayout::MultiLaunch, UInt32Array::from(Vec::<u32>::new())));
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
        let kb = upload_padded_key_for_dtype(k.column, k.dtype, k.direction, n_pow2_usize)?;
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

    // Compile + load the module.
    let ptx = compile_sort_kernel_spec(&spec)?;
    let module = CudaModule::from_ptx(&ptx)?;
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

                    // SAFETY: same as Shmem branch — every param points at
                    // a stack local that outlives the synchronous launch.
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

    Ok((layout, UInt32Array::from(out)))
}

/// Sort an entire `RecordBatch` by `keys` on the GPU, gather every column.
pub fn sort_record_batch_on_gpu_multi(
    batch: &RecordBatch,
    keys: &[(usize, DataType, SortDirection, bool /*nulls_first*/)],
) -> BoltResult<RecordBatch> {
    let sort_keys: Vec<GpuSortKey> = keys
        .iter()
        .map(|(idx, dtype, dir, nf)| GpuSortKey {
            column: batch.column(*idx).as_ref(),
            dtype: *dtype,
            direction: *dir,
            nulls_first: *nf,
        })
        .collect();
    let (_layout, perm) = sort_indices_on_gpu_multi(&sort_keys)?;
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
        // the index dtype. Plain Utf8 still falls through to host.
        assert_eq!(arrow_dtype_to_internal(&ArrowDataType::Utf8), None);
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

    /// End-to-end ASC int32 sort. Builds a 16k-row scrambled column, runs it
    /// through `sort_indices_on_gpu`, gathers, and asserts strictly
    /// ascending output.
    #[test]
    #[ignore = "requires CUDA toolkit + driver at runtime"]
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

        let perm =
            sort_indices_on_gpu(&arr, DataType::Int32, SortDirection::Asc).expect("gpu sort");

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
    #[ignore = "requires CUDA toolkit + driver at runtime"]
    fn gpu_sort_int64_desc_with_padding() {
        let n = 16_385usize;
        let values: Vec<i64> = (0..n as i64).map(|i| (i * 7919) % 1_000_000).collect();
        let arr = Int64Array::from(values.clone());

        let perm =
            sort_indices_on_gpu(&arr, DataType::Int64, SortDirection::Desc).expect("gpu sort");
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
    #[ignore = "requires CUDA toolkit + driver at runtime"]
    fn gpu_sort_float64_asc_with_padding() {
        let n = 20_000usize;
        let values: Vec<f64> = (0..n).map(|i| ((i as f64) * 1.61803398875).sin()).collect();
        let arr = Float64Array::from(values.clone());

        let perm =
            sort_indices_on_gpu(&arr, DataType::Float64, SortDirection::Asc).expect("gpu sort");
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

    /// `sort_record_batch_on_gpu` glues the index sort to the full-batch
    /// gather. Build a two-column batch (key + payload), sort by the key,
    /// and confirm the payload tracks.
    #[test]
    #[ignore = "requires CUDA toolkit + driver at runtime"]
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

        let out = sort_record_batch_on_gpu(&batch, 0, DataType::Int32, SortDirection::Asc)
            .expect("gpu sort batch");
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
}
