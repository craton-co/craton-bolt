// SPDX-License-Identifier: Apache-2.0

//! Host-side filter compaction.
//!
//! Today's projection-with-filter kernel (see `ptx_gen.rs`) gates its `Store`
//! ops on the predicate, but masked-out rows still occupy slots in the output
//! buffer — they just hold the zero-initialized value from allocation. The
//! engine wants a `RecordBatch` containing exactly the surviving rows.
//!
//! This module implements the simplest correct compaction strategy:
//!
//! 1. The engine launches a small "predicate-only" kernel (see
//!    `crate::jit::scan_kernel`) that writes a `u8` per row to a device mask
//!    buffer (`1` = keep, `0` = drop).
//! 2. [`download_mask`] copies that mask back to the host as a `Vec<bool>`.
//! 3. After downloading the projected columns to Arrow arrays, the engine
//!    feeds them through [`compact_arrays`], which calls Arrow's
//!    `arrow::compute::filter` per column to drop masked-out rows.
//!
//! A GPU-side prefix-scan + gather pipeline is a future optimization; the
//! mask transfer (one byte per row) is small and `arrow::compute::filter` is
//! already well-tuned.

use arrow::compute::filter;
use arrow_array::{ArrayRef, BooleanArray};

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{grid_x_for, CudaStream};
use crate::jit::CudaFunction;

/// Default 1D block size for the predicate-eval launch. Matches the
/// projection kernel's launch shape so the engine can reuse its grid math.
const PREDICATE_BLOCK_SIZE: u32 = 256;

/// Copy a device-side `u8` mask of length `n_rows` back to a host `Vec<bool>`.
///
/// The device mask is the output written by the predicate-only kernel: one
/// byte per row, `1` for "keep" and `0` for "drop". Any non-zero byte is
/// treated as `true` to be tolerant of mask kernels that emit other truthy
/// values.
pub fn download_mask(mask_device_ptr: CUdeviceptr, n_rows: usize) -> BoltResult<Vec<bool>> {
    if n_rows == 0 {
        return Ok(Vec::new());
    }
    let mut host = vec![0u8; n_rows];
    // SAFETY: `host` is valid for `n_rows` byte writes; the caller asserts
    // `mask_device_ptr` is a live device allocation of at least that size.
    unsafe {
        cuda_sys::memcpy_d2h::<u8>(host.as_mut_ptr(), mask_device_ptr, n_rows)?;
    }
    Ok(host.into_iter().map(|b| b != 0).collect())
}

/// Apply `mask` to an Arrow array, returning a new array with only the kept
/// rows in their original order.
///
/// Internally delegates to `arrow::compute::filter`, the canonical Arrow
/// compaction kernel.
///
/// ## Nullability preservation (W7A8)
///
/// `arrow::compute::filter` is documented to preserve per-row null-ness: for
/// every kept row it copies BOTH the value and the validity bit from the
/// source array. That means a nullable `BooleanArray` (the host-side
/// reconstruction of `engine.rs::DeviceCol::BoolNullable`) round-trips
/// through this function with its validity bitmap intact — null rows that
/// pass the predicate stay null on the output side, and non-null rows
/// retain their concrete `true` / `false` value.
///
/// The predicate `BooleanArray` we build from `mask` has `null_count() == 0`
/// by construction (every entry comes from a `Vec<bool>`), so the predicate
/// itself never injects nullness — only the source array's nulls are
/// preserved.
///
/// This is verified by the `compact_bool_with_nulls_preserves_validity`
/// test in this file's `#[cfg(test)]` module.
pub fn apply_mask(arr: &ArrayRef, mask: &[bool]) -> BoltResult<ArrayRef> {
    if arr.len() != mask.len() {
        return Err(BoltError::Other(format!(
            "compact::apply_mask length mismatch: array={}, mask={}",
            arr.len(),
            mask.len()
        )));
    }
    // `BooleanArray::from(Vec<bool>)` is the standard constructor; cloning the
    // slice into an owned Vec is cheap relative to the d2h copy upstream.
    // The resulting predicate has no nulls of its own — every entry is a
    // definite keep / drop — so `filter` only ever propagates nulls that
    // already lived in `arr`.
    let predicate = BooleanArray::from(mask.to_vec());
    let filtered = filter(arr.as_ref(), &predicate).map_err(|e| {
        BoltError::Other(format!("arrow::compute::filter failed: {e}"))
    })?;
    // `filter` returns `ArrayRef` (== `Arc<dyn Array>`) directly in arrow 53;
    // no extra wrap needed.
    Ok(filtered)
}

/// Apply the same mask to every array in `arrays`, preserving order.
///
/// Convenience for the engine: a projected `RecordBatch` is `Vec<ArrayRef>`
/// plus a schema, and the schema is unchanged by compaction.
pub fn compact_arrays(arrays: &[ArrayRef], mask: &[bool]) -> BoltResult<Vec<ArrayRef>> {
    arrays.iter().map(|a| apply_mask(a, mask)).collect()
}

/// Allocate an `n`-byte device mask buffer, zero-initialized.
///
/// The returned `GpuVec<u8>` owns the allocation and frees it on drop, so the
/// engine should keep it alive until after the predicate kernel launch and
/// d2h copy complete.
pub fn alloc_mask_buffer(n: usize) -> BoltResult<GpuVec<u8>> {
    GpuVec::<u8>::zeros(n)
}

/// Launch a predicate-only kernel.
///
/// The kernel ABI (matching the PTX emitted by
/// `crate::jit::scan_kernel::compile_predicate_kernel`) is:
///
/// ```text
/// (input_col_0_ptr: .u64,
///  ...,
///  input_col_{N-1}_ptr: .u64,
///  mask_output_ptr: .u64,
///  input_validity_ptr_a: .u64,           // when input_validity_ptrs is non-empty
///  ...,
///  input_validity_ptr_K-1: .u64,
///  n_rows: .u32)
/// ```
///
/// where `N == input_ptrs.len()` and `K == input_validity_ptrs.len()`. The
/// validity pointers correspond to the flagged inputs in
/// `KernelSpec::input_has_validity` (in input-slot order, skipping
/// non-flagged slots), matching the param walk in
/// `crate::jit::scan_kernel::compile_predicate_kernel`. The launch is one
/// thread per row, `PREDICATE_BLOCK_SIZE` threads per block, and the call
/// synchronizes the stream before returning — so by the time `Ok(())` comes
/// back, the mask is safe to download with [`download_mask`].
pub fn launch_predicate_kernel(
    function: CudaFunction<'_>,
    input_ptrs: &[CUdeviceptr],
    mask_ptr: CUdeviceptr,
    input_validity_ptrs: &[CUdeviceptr],
    n_rows: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_rows == 0 {
        // Nothing to do; an empty launch is wasted work and would have a
        // grid_x of 1 with all threads OOB. Be explicit.
        return Ok(());
    }

    // Assemble the device-pointer argument list: inputs..., mask,
    // input_validity_ptrs... . The pointers must live at stable addresses
    // until after the launch, so we copy them into an owned local Vec.
    let mut device_ptrs: Vec<CUdeviceptr> =
        Vec::with_capacity(input_ptrs.len() + 1 + input_validity_ptrs.len());
    device_ptrs.extend_from_slice(input_ptrs);
    device_ptrs.push(mask_ptr);
    device_ptrs.extend_from_slice(input_validity_ptrs);

    let mut n_rows_local: u32 = n_rows;

    // Build the *mut c_void array the driver expects. Each entry points at
    // the storage of one kernel argument (a CUdeviceptr or the n_rows u32).
    let mut kernel_params: Vec<*mut std::ffi::c_void> =
        Vec::with_capacity(device_ptrs.len() + 1);
    for p in device_ptrs.iter_mut() {
        kernel_params.push(p as *mut CUdeviceptr as *mut std::ffi::c_void);
    }
    kernel_params.push(&mut n_rows_local as *mut u32 as *mut std::ffi::c_void);

    let grid_x = grid_x_for(n_rows, PREDICATE_BLOCK_SIZE);

    // SAFETY: `function` is borrowed from a live `CudaModule`; every entry of
    // `kernel_params` points into `device_ptrs` or `n_rows_local`, both of
    // which outlive the launch + synchronize below.
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            PREDICATE_BLOCK_SIZE,
            1,
            1,
            0,
            stream.raw(),
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        ))?;
    }
    stream.synchronize()?;

    // Keep the borrowed locals alive past the synchronize for clarity; the
    // compiler can also prove this, but an explicit drop documents intent.
    drop(kernel_params);
    drop(device_ptrs);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Array, BooleanArray, Int32Array};
    use std::sync::Arc;

    /// Sanity check: `apply_mask` over a plain non-nullable Int32 column drops
    /// the masked-out rows and preserves source order, without introducing
    /// any nullability. This is the baseline behavior every other test in
    /// this module builds on.
    #[test]
    fn compact_int32_no_nulls() {
        let src: ArrayRef = Arc::new(Int32Array::from(vec![10, 20, 30, 40, 50]));
        let mask = vec![true, false, true, false, true];
        let out = apply_mask(&src, &mask).expect("apply_mask");
        let out_i32 = out
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32Array");
        assert_eq!(out_i32.len(), 3);
        assert_eq!(out_i32.value(0), 10);
        assert_eq!(out_i32.value(1), 30);
        assert_eq!(out_i32.value(2), 50);
        assert_eq!(out_i32.null_count(), 0);
    }

    /// W7A8 core test: filter compaction over a nullable `BooleanArray` —
    /// the host-side reconstruction of `engine.rs::DeviceCol::BoolNullable`
    /// after download — must preserve per-row validity for every surviving
    /// row. Concretely:
    ///
    ///   * `true`  rows that pass the predicate stay `Some(true)`.
    ///   * `false` rows that pass stay `Some(false)`.
    ///   * `null`  rows that pass stay `None` (the bug we're guarding
    ///     against was that the validity bitmap was dropped during
    ///     compaction, collapsing nulls into `Some(false)`).
    ///
    /// The source layout below covers all three states crossed with both
    /// mask polarities, so any regression — to "all nulls → false", "no
    /// nulls at all", or a wrong row count — fails a specific assertion.
    #[test]
    fn compact_bool_with_nulls_preserves_validity() {
        // 6 rows: true, null, false, true, null, false.
        // Mask:    keep, keep, drop, keep, keep, drop.
        // Expected output: Some(true), None, Some(true), None  (4 rows).
        let src: ArrayRef = Arc::new(BooleanArray::from(vec![
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            Some(false),
        ]));
        assert_eq!(src.null_count(), 2, "source must have 2 nulls");

        let mask = vec![true, true, false, true, true, false];
        let out = apply_mask(&src, &mask).expect("apply_mask");

        let out_bool = out
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("filter must return a BooleanArray for a BooleanArray source");
        assert_eq!(out_bool.len(), 4, "kept 4 of 6 rows");

        // Walk every surviving row; per-row validity must survive verbatim.
        // We compare to an `Option<bool>` list rather than poking
        // `is_null` / `value` separately so the assertion failure
        // message names the offending row directly.
        let expected: Vec<Option<bool>> =
            vec![Some(true), None, Some(true), None];
        let actual: Vec<Option<bool>> = (0..out_bool.len())
            .map(|i| {
                if out_bool.is_null(i) {
                    None
                } else {
                    Some(out_bool.value(i))
                }
            })
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(out_bool.null_count(), 2, "both surviving nulls preserved");
    }

    /// All-null nullable bool input under an all-pass mask must round-trip
    /// to an all-null output of the same length. Complements the mixed
    /// test above by isolating the validity-only path: every `value` byte
    /// is undefined-or-zero on the upload side, so a regression that
    /// dropped validity would visibly produce all-`false` rather than
    /// all-`null`.
    #[test]
    fn compact_bool_all_nulls_stay_null() {
        let src: ArrayRef =
            Arc::new(BooleanArray::from(vec![None, None, None]));
        let mask = vec![true, true, true];
        let out = apply_mask(&src, &mask).expect("apply_mask");
        let out_bool = out
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("BooleanArray");
        assert_eq!(out_bool.len(), 3);
        assert_eq!(out_bool.null_count(), 3);
        for i in 0..3 {
            assert!(out_bool.is_null(i), "row {i} should remain null");
        }
    }

    /// Length-mismatch between array and mask must surface as a clean
    /// `BoltError::Other`, not a panic. Catches a regression where the
    /// guard was removed in favor of `arrow::compute::filter`'s own check
    /// (which produces a less actionable message).
    #[test]
    fn compact_length_mismatch_errors() {
        let src: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let mask = vec![true, false];
        let err = apply_mask(&src, &mask).expect_err("must reject mismatch");
        let msg = format!("{err}");
        assert!(
            msg.contains("length mismatch"),
            "expected length-mismatch error, got: {msg}"
        );
    }
}
