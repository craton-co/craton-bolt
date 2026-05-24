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
use crate::error::{JavelinError, JavelinResult};
use crate::exec::launch::CudaStream;
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
pub fn download_mask(mask_device_ptr: CUdeviceptr, n_rows: usize) -> JavelinResult<Vec<bool>> {
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
pub fn apply_mask(arr: &ArrayRef, mask: &[bool]) -> JavelinResult<ArrayRef> {
    if arr.len() != mask.len() {
        return Err(JavelinError::Other(format!(
            "compact::apply_mask length mismatch: array={}, mask={}",
            arr.len(),
            mask.len()
        )));
    }
    // `BooleanArray::from(Vec<bool>)` is the standard constructor; cloning the
    // slice into an owned Vec is cheap relative to the d2h copy upstream.
    let predicate = BooleanArray::from(mask.to_vec());
    let filtered = filter(arr.as_ref(), &predicate).map_err(|e| {
        JavelinError::Other(format!("arrow::compute::filter failed: {e}"))
    })?;
    // `filter` returns `ArrayRef` (== `Arc<dyn Array>`) directly in arrow 53;
    // no extra wrap needed.
    Ok(filtered)
}

/// Apply the same mask to every array in `arrays`, preserving order.
///
/// Convenience for the engine: a projected `RecordBatch` is `Vec<ArrayRef>`
/// plus a schema, and the schema is unchanged by compaction.
pub fn compact_arrays(arrays: &[ArrayRef], mask: &[bool]) -> JavelinResult<Vec<ArrayRef>> {
    arrays.iter().map(|a| apply_mask(a, mask)).collect()
}

/// Allocate an `n`-byte device mask buffer, zero-initialized.
///
/// The returned `GpuVec<u8>` owns the allocation and frees it on drop, so the
/// engine should keep it alive until after the predicate kernel launch and
/// d2h copy complete.
pub fn alloc_mask_buffer(n: usize) -> JavelinResult<GpuVec<u8>> {
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
///  n_rows: .u32)
/// ```
///
/// where `N == input_ptrs.len()`. The launch is one thread per row,
/// `PREDICATE_BLOCK_SIZE` threads per block, and the call synchronizes the
/// stream before returning — so by the time `Ok(())` comes back, the mask is
/// safe to download with [`download_mask`].
pub fn launch_predicate_kernel(
    function: CudaFunction<'_>,
    input_ptrs: &[CUdeviceptr],
    mask_ptr: CUdeviceptr,
    n_rows: u32,
    stream: &CudaStream,
) -> JavelinResult<()> {
    if n_rows == 0 {
        // Nothing to do; an empty launch is wasted work and would have a
        // grid_x of 1 with all threads OOB. Be explicit.
        return Ok(());
    }

    // Assemble the device-pointer argument list: inputs..., mask. The pointers
    // must live at stable addresses until after the launch, so we copy them
    // into an owned local Vec.
    let mut device_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(input_ptrs.len() + 1);
    device_ptrs.extend_from_slice(input_ptrs);
    device_ptrs.push(mask_ptr);

    let mut n_rows_local: u32 = n_rows;

    // Build the *mut c_void array the driver expects. Each entry points at
    // the storage of one kernel argument (a CUdeviceptr or the n_rows u32).
    let mut kernel_params: Vec<*mut std::ffi::c_void> =
        Vec::with_capacity(device_ptrs.len() + 1);
    for p in device_ptrs.iter_mut() {
        kernel_params.push(p as *mut CUdeviceptr as *mut std::ffi::c_void);
    }
    kernel_params.push(&mut n_rows_local as *mut u32 as *mut std::ffi::c_void);

    let grid_x = ((n_rows + PREDICATE_BLOCK_SIZE - 1) / PREDICATE_BLOCK_SIZE).max(1);

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
