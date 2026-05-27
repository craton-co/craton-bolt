// SPDX-License-Identifier: Apache-2.0

//! **cudarc-backed CUDA Driver API spike** (Stage 1).
//!
//! This module is gated on the `cudarc` feature. When the feature is
//! enabled, a handful of low-level memory primitives in `cuda_sys.rs`
//! delegate here instead of into the hand-rolled `extern "C"` FFI.
//! Everything else (context create, kernel launch, module load, …)
//! continues to use the existing path during Stage 1 — the goal of
//! this spike is to **prove a feature-flagged cudarc backend builds
//! and runs** without yet committing to a full migration. See
//! [`docs/CUDARC_ADOPTION.md`](../../docs/CUDARC_ADOPTION.md) for the
//! end-state plan.
//!
//! ## Surface area covered
//!
//! Stage 1 routes these four functions through cudarc:
//!
//!   * `mem_alloc(bytes)` — allocates `bytes` bytes of device memory.
//!   * `mem_free(ptr)` — frees a previously-allocated pointer.
//!   * `memcpy_h2d::<T>(dst, src, count)` — host → device copy.
//!   * `memcpy_d2h::<T>(dst, src, count)` — device → host copy.
//!
//! cudarc 0.13's API differs from our raw FFI in one important way:
//! cudarc *owns* its allocations via `CudaSlice<T>`, which `Drop`s
//! itself. To keep `GpuBuffer<T>` and `GpuVec<T>` working with the
//! existing call sites (which expect to free via an explicit
//! `mem_free(ptr)`), we use cudarc's raw alloc/free escape hatch —
//! `result::malloc_sync` / `result::free_sync` — which returns / takes
//! a raw `CUdeviceptr` exactly like our FFI does.
//!
//! When Stage 2 lands we'll switch to `CudaSlice<T>` ownership and
//! delete the raw-pointer helpers.

use cudarc::driver::{result, CudaDevice};
use std::sync::Arc;

use crate::cuda::cuda_sys::CUdeviceptr;
use crate::error::{BoltError, BoltResult};

/// Per-process cudarc device cache. `CudaDevice::new` returns an
/// `Arc<CudaDevice>` and binds a primary context; we keep one around
/// for the chosen ordinal. Multi-GPU is a Stage 2+ concern — the
/// current backend wires every alloc through device 0 once it's
/// latched here, but `ensure_device(n)` lets `CudaContext::new`
/// initialise the cell with a non-default ordinal on a single-GPU
/// system as a transitional step.
static GLOBAL_DEVICE: once_cell::sync::OnceCell<Arc<CudaDevice>> =
    once_cell::sync::OnceCell::new();

/// Initialise the cudarc primary context on `ordinal` if it isn't
/// already. This is the canonical entry point for `CudaContext::new`
/// under `--features cudarc` — calling it makes the cudarc-owned
/// context current on the calling thread and ensures every subsequent
/// `mem_alloc` / `mem_free` / `memcpy_*` routes through the SAME
/// context (fixing the historical two-context bug where
/// `cuCtxCreate_v2` minted a parallel context that the pool's
/// pointers did not belong to).
///
/// If the cell is already latched, this returns the existing device
/// — subsequent calls with a different `ordinal` are silently ignored
/// (single-GPU only for now; tracked in
/// `docs/CUDARC_ADOPTION.md` Stage 2).
pub(crate) fn ensure_device(ordinal: i32) -> BoltResult<()> {
    GLOBAL_DEVICE
        .get_or_try_init(|| {
            CudaDevice::new(ordinal as usize).map_err(|e| {
                BoltError::Cuda(format!(
                    "cudarc CudaDevice::new({ordinal}) failed: {e:?}"
                ))
            })
        })
        .map(|_| ())
}

fn device() -> BoltResult<Arc<CudaDevice>> {
    // Lazily initialise on device 0 if nobody called `ensure_device`
    // first. This preserves the original spike behaviour for callers
    // that go directly through this module's `mem_alloc` etc.
    GLOBAL_DEVICE
        .get_or_try_init(|| {
            CudaDevice::new(0).map_err(|e| {
                BoltError::Cuda(format!("cudarc CudaDevice::new failed: {e:?}"))
            })
        })
        .map(Arc::clone)
}

/// Allocate `bytes` of device memory via cudarc's raw `malloc_sync`.
/// Returned pointer is bit-compatible with `cuda_sys::mem_alloc`.
pub fn mem_alloc(bytes: usize) -> BoltResult<CUdeviceptr> {
    // Ensure the primary context is current on this thread.
    let _dev = device()?;
    unsafe {
        result::malloc_sync(bytes)
            .map(|p| p as CUdeviceptr)
            .map_err(|e| BoltError::Cuda(format!("cudarc malloc_sync: {e:?}")))
    }
}

/// Free a device pointer. Mirrors `cuda_sys::mem_free`.
///
/// # Safety
/// `ptr` must have been returned by `mem_alloc` (or by
/// `cuda_sys::mem_alloc` — both call into the same `cuMemAlloc_v2`).
pub unsafe fn mem_free(ptr: CUdeviceptr) -> BoltResult<()> {
    result::free_sync(ptr as cudarc::driver::sys::CUdeviceptr)
        .map_err(|e| BoltError::Cuda(format!("cudarc free_sync: {e:?}")))
}

/// Copy `count` elements of `T` from host to device.
///
/// # Safety
/// `src` must be valid for `count * size_of::<T>()` bytes of reads;
/// `dst` must point to a device allocation of at least that size.
pub unsafe fn memcpy_h2d<T>(
    dst: CUdeviceptr,
    src: *const T,
    count: usize,
) -> BoltResult<()> {
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        BoltError::Memory(format!(
            "cudarc memcpy_h2d size overflow: {count} * {}",
            std::mem::size_of::<T>()
        ))
    })?;
    let src_bytes = std::slice::from_raw_parts(src as *const u8, bytes);
    result::memcpy_htod_sync(dst as cudarc::driver::sys::CUdeviceptr, src_bytes)
        .map_err(|e| BoltError::Cuda(format!("cudarc memcpy_htod_sync: {e:?}")))
}

/// Copy `count` elements of `T` from device to host.
///
/// # Safety
/// `dst` must be valid for `count * size_of::<T>()` bytes of writes;
/// `src` must point to a live device allocation of at least that size.
pub unsafe fn memcpy_d2h<T>(
    dst: *mut T,
    src: CUdeviceptr,
    count: usize,
) -> BoltResult<()> {
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        BoltError::Memory(format!(
            "cudarc memcpy_d2h size overflow: {count} * {}",
            std::mem::size_of::<T>()
        ))
    })?;
    let dst_bytes = std::slice::from_raw_parts_mut(dst as *mut u8, bytes);
    result::memcpy_dtoh_sync(dst_bytes, src as cudarc::driver::sys::CUdeviceptr)
        .map_err(|e| BoltError::Cuda(format!("cudarc memcpy_dtoh_sync: {e:?}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test that the cudarc context comes up at all. Gated on
    /// `BOLT_BENCH_GPU=1` for the same reason as the engine tests —
    /// we can't acquire a context on a non-CUDA host.
    #[test]
    #[ignore = "requires CUDA device (set BOLT_BENCH_GPU=1 + run with --ignored)"]
    fn cudarc_alloc_roundtrip() {
        let host_in: Vec<i32> = (0..1024).collect();
        let dev_ptr = mem_alloc(host_in.len() * 4).expect("alloc");
        unsafe {
            memcpy_h2d::<i32>(dev_ptr, host_in.as_ptr(), host_in.len()).expect("h2d");
        }
        let mut host_out: Vec<i32> = vec![0; host_in.len()];
        unsafe {
            memcpy_d2h::<i32>(host_out.as_mut_ptr(), dev_ptr, host_out.len()).expect("d2h");
            mem_free(dev_ptr).expect("free");
        }
        assert_eq!(host_in, host_out);
    }
}
