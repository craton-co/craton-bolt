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
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
    UInt32Array,
};
use arrow::compute::take;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;
use crate::exec::n_rows_to_u32;
use crate::jit::jit_compiler::CudaModule;
use crate::jit::sort_kernel::{
    compile_sort_kernel, sort_kernel_entry, SortDirection, SORT_BLOCK_SIZE,
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
pub fn arrow_dtype_to_internal(d: &arrow_schema::DataType) -> Option<DataType> {
    use arrow_schema::DataType as A;
    match d {
        A::Int32 => Some(DataType::Int32),
        A::Int64 => Some(DataType::Int64),
        A::Float32 => Some(DataType::Float32),
        A::Float64 => Some(DataType::Float64),
        _ => None,
    }
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
        assert_eq!(arrow_dtype_to_internal(&ArrowDataType::Utf8), None);
        assert_eq!(arrow_dtype_to_internal(&ArrowDataType::Boolean), None);
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
