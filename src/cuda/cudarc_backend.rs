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

use crate::cuda::cuda_sys::{CUdeviceptr, CUstream};
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
            CudaDevice::new(ordinal as usize).map_err(|e| cudarc_err(
                &format!("cudarc CudaDevice::new({ordinal})"),
                e,
            ))
        })
        .map(|_| ())
}

fn device() -> BoltResult<Arc<CudaDevice>> {
    // Lazily initialise on device 0 if nobody called `ensure_device`
    // first. This preserves the original spike behaviour for callers
    // that go directly through this module's `mem_alloc` etc.
    // PERF-NOTE: Arc::clone in hot memcpy path; defer optimisation to Stage 2.
    GLOBAL_DEVICE
        .get_or_try_init(|| {
            CudaDevice::new(0)
                .map_err(|e| cudarc_err("cudarc CudaDevice::new", e))
        })
        .map(Arc::clone)
}

/// Stage 5 (M3L5): translate a cudarc `DriverError` into the typed
/// [`BoltError::CudaWithCode`] variant so downstream code (notably
/// `mem_pool::is_oom_error`) can pattern-match the raw `CUresult`
/// integer instead of scraping a formatted string.
///
/// cudarc represents driver errors as `DriverError(pub sys::CUresult)`,
/// where `sys::CUresult` is a `#[repr(u32)]` enum. Casting the inner
/// value to `i32` yields the same integer that the hand-rolled FFI's
/// `cuda_sys::check` would have produced for the equivalent driver
/// error. The Debug rendering preserves the original `{e:?}` text so
/// log output stays familiar.
fn cudarc_err(context: &str, e: cudarc::driver::DriverError) -> BoltError {
    let code = e.0 as i32;
    BoltError::CudaWithCode {
        code,
        message: format!("{context}: {e:?}"),
    }
}

/// Allocate `bytes` of device memory via cudarc's raw `malloc_sync`.
/// Returned pointer is bit-compatible with `cuda_sys::mem_alloc`.
///
/// `bytes == 0` is rejected at the wrapper boundary: the CUDA driver's
/// behaviour for zero-byte allocations is implementation-defined (some
/// versions return `CUDA_ERROR_INVALID_VALUE`, others a non-null sentinel
/// pointer that cannot be freed by `cuMemFree_v2`). We refuse the call so
/// callers get a deterministic, typed error instead of a context-dependent
/// driver failure.
pub fn mem_alloc(bytes: usize) -> BoltResult<CUdeviceptr> {
    if bytes == 0 {
        return Err(BoltError::Other(
            "mem_alloc: zero-byte allocation not allowed".into(),
        ));
    }
    // Ensure the primary context is current on this thread.
    let _dev = device()?;
    unsafe {
        result::malloc_sync(bytes)
            .map(|p| p as CUdeviceptr)
            .map_err(|e| cudarc_err("cudarc malloc_sync", e))
    }
}

/// Free a device pointer. Mirrors `cuda_sys::mem_free`.
///
/// # Context-currency invariant
/// Every freed pointer belongs to cudarc's primary context, so
/// `cuMemFree_v2` must run with that context current on the *calling*
/// thread. We therefore establish currency by calling [`device()`]
/// first — exactly as [`mem_alloc`] does before `malloc_sync` — rather
/// than assuming the caller already made the context current. This
/// matters because frees originate from threads that never touched the
/// alloc path: a worker thread dropping a `GpuBuffer`, or the
/// process-wide pool drain in `CudaContext::Drop` (see
/// `cuda_sys.rs`). Without this guard a free on such a thread would hit
/// `cuMemFree_v2` against the wrong (or no) context, producing either a
/// swallowed error (a silent leak) or a context-mismatch free.
/// Self-guarding here keeps the alloc/free pair symmetric and makes the
/// `Drop`-drain path correct regardless of which thread runs it.
///
/// `get_or_try_init` makes the `device()` call idempotent and cheap on
/// the common path (the cell is already latched after the first
/// `ensure_device`/`mem_alloc`), so this adds no FFI — only an
/// `Arc::clone` that is dropped immediately.
///
/// # Safety
/// `ptr` must have been returned by `mem_alloc` (or by
/// `cuda_sys::mem_alloc` — both call into the same `cuMemAlloc_v2`).
pub unsafe fn mem_free(ptr: CUdeviceptr) -> BoltResult<()> {
    // Ensure the primary context is current on this thread before the free.
    // Mirrors `mem_alloc`'s `let _dev = device()?;` guard.
    // PERF-NOTE: Arc::clone on the free path; matches mem_alloc, defer to Stage 2.
    let _dev = match device() {
        Ok(dev) => dev,
        Err(e) => {
            // The device could not be made current — the pointer cannot be
            // freed safely against the right context. Surface the error
            // rather than calling `free_sync` against the wrong/no context.
            log::debug!(
                "cudarc mem_free: primary context unavailable on this thread \
                 ({e:?}); skipping free to avoid a context-mismatch free"
            );
            return Err(e);
        }
    };
    result::free_sync(ptr as cudarc::driver::sys::CUdeviceptr)
        .map_err(|e| cudarc_err("cudarc free_sync", e))
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
    // Zero-length copies must short-circuit BEFORE synthesising a slice from
    // `src`: `std::slice::from_raw_parts` requires a non-null, aligned, and
    // dereferenceable pointer regardless of the requested length, so a caller
    // passing `count = 0` with `src = null` would otherwise hit immediate UB.
    if count == 0 {
        return Ok(());
    }
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        BoltError::Memory(format!(
            "cudarc memcpy_h2d size overflow: {count} * {}",
            std::mem::size_of::<T>()
        ))
    })?;
    if bytes == 0 {
        return Ok(());
    }
    debug_assert!(
        !src.is_null(),
        "memcpy_h2d: src is null with non-zero count"
    );
    let src_bytes = std::slice::from_raw_parts(src as *const u8, bytes);
    result::memcpy_htod_sync(dst as cudarc::driver::sys::CUdeviceptr, src_bytes)
        .map_err(|e| cudarc_err("cudarc memcpy_htod_sync", e))
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
    // Zero-length copies must short-circuit BEFORE synthesising a slice from
    // `dst`: `std::slice::from_raw_parts_mut` requires a non-null, aligned,
    // dereferenceable pointer regardless of the requested length, so a caller
    // passing `count = 0` with `dst = null` would otherwise hit immediate UB.
    if count == 0 {
        return Ok(());
    }
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        BoltError::Memory(format!(
            "cudarc memcpy_d2h size overflow: {count} * {}",
            std::mem::size_of::<T>()
        ))
    })?;
    if bytes == 0 {
        return Ok(());
    }
    debug_assert!(
        !dst.is_null(),
        "memcpy_d2h: dst is null with non-zero count"
    );
    let dst_bytes = std::slice::from_raw_parts_mut(dst as *mut u8, bytes);
    result::memcpy_dtoh_sync(dst_bytes, src as cudarc::driver::sys::CUdeviceptr)
        .map_err(|e| cudarc_err("cudarc memcpy_dtoh_sync", e))
}

// ---------------------------------------------------------------------------
// Stage 2 (review C3): real async memcpy/memset through cudarc's raw
// driver::sys bindings.
//
// cudarc 0.13's safe `result::memcpy_*_async` wrappers take Rust slices,
// which is awkward to feed from our raw-pointer-based `cuda_sys` surface
// (the caller passes `*const T` / `*mut T` + count, and we cannot safely
// synthesize a slice over device-bound memory or over potentially-unaligned
// host pointers without imposing extra invariants on every call site).
//
// We therefore drop one level lower and invoke the dynamically-loaded
// `cudarc::driver::sys::lib()` methods directly. These are the same FFI
// symbols our hand-rolled `extern "C"` block exposes — the only difference
// is the type alias for `CUstream` (cudarc uses `*mut CUstream_st`, we use
// `*mut c_void`), so we cast at the boundary. The driver itself sees the
// identical bit pattern.
//
// Error mapping goes through `cudarc_err` so OOM and other failures surface
// as `BoltError::CudaWithCode` exactly like every other cudarc-backed call.
// ---------------------------------------------------------------------------

/// Asynchronously copy `count` elements of `T` from host `src` to device
/// `dst` on `stream`. Cudarc-backed counterpart to
/// `cuda_sys::memcpy_h2d_async`.
///
/// # Safety
/// `src` must be valid for `count * size_of::<T>()` bytes of reads for the
/// duration of the async copy (until the stream is synchronized); `dst`
/// must point to a live device allocation of at least that size in the
/// currently-bound context.
pub(crate) unsafe fn memcpy_h2d_async<T>(
    dst: CUdeviceptr,
    src: *const T,
    count: usize,
    stream: CUstream,
) -> BoltResult<()> {
    // Defensive short-circuit on zero-length copies. Unlike the sync sibling
    // this path does not synthesise a `&[u8]`, but the driver's behaviour on
    // a zero-byte async copy with a null host pointer is implementation-
    // defined, and the `debug_assert` below documents the non-zero contract.
    if count == 0 {
        return Ok(());
    }
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        BoltError::Memory(format!(
            "cudarc memcpy_h2d_async size overflow: {count} * {}",
            std::mem::size_of::<T>()
        ))
    })?;
    if bytes == 0 {
        return Ok(());
    }
    debug_assert!(
        !src.is_null(),
        "memcpy_h2d_async: src is null with non-zero count"
    );
    // Ensure the primary context is current on this thread before any FFI.
    let _dev = device()?;
    cudarc::driver::sys::lib()
        .cuMemcpyHtoDAsync_v2(
            dst as cudarc::driver::sys::CUdeviceptr,
            src as *const core::ffi::c_void,
            bytes,
            stream as cudarc::driver::sys::CUstream,
        )
        .result()
        .map_err(|e| cudarc_err("cudarc cuMemcpyHtoDAsync_v2", e))
}

/// Asynchronously copy `count` elements of `T` from device `src` to host
/// `dst` on `stream`. Cudarc-backed counterpart to
/// `cuda_sys::memcpy_d2h_async`.
///
/// # Safety
/// `dst` must be valid for `count * size_of::<T>()` bytes of writes for the
/// duration of the async copy (until the stream is synchronized); `src`
/// must point to a live device allocation of at least that size in the
/// currently-bound context.
pub(crate) unsafe fn memcpy_d2h_async<T>(
    dst: *mut T,
    src: CUdeviceptr,
    count: usize,
    stream: CUstream,
) -> BoltResult<()> {
    // Defensive short-circuit on zero-length copies; mirrors the sync path
    // and keeps the non-null contract on `dst` explicit via `debug_assert`.
    if count == 0 {
        return Ok(());
    }
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        BoltError::Memory(format!(
            "cudarc memcpy_d2h_async size overflow: {count} * {}",
            std::mem::size_of::<T>()
        ))
    })?;
    if bytes == 0 {
        return Ok(());
    }
    debug_assert!(
        !dst.is_null(),
        "memcpy_d2h_async: dst is null with non-zero count"
    );
    let _dev = device()?;
    cudarc::driver::sys::lib()
        .cuMemcpyDtoHAsync_v2(
            dst as *mut core::ffi::c_void,
            src as cudarc::driver::sys::CUdeviceptr,
            bytes,
            stream as cudarc::driver::sys::CUstream,
        )
        .result()
        .map_err(|e| cudarc_err("cudarc cuMemcpyDtoHAsync_v2", e))
}

/// Asynchronously fill `n_bytes` at device pointer `ptr` with the byte
/// `value`, on `stream`. Cudarc-backed counterpart to
/// `cuda_sys::memset_d8_async`.
///
/// # Safety
/// `ptr` must point to a live device allocation of at least `n_bytes`
/// bytes in the currently-bound context; the memory must not be freed or
/// concurrently mutated until the stream is synchronized.
pub(crate) unsafe fn memset_d8_async(
    ptr: CUdeviceptr,
    value: u8,
    n_bytes: usize,
    stream: CUstream,
) -> BoltResult<()> {
    let _dev = device()?;
    cudarc::driver::sys::lib()
        .cuMemsetD8Async(
            ptr as cudarc::driver::sys::CUdeviceptr,
            value,
            n_bytes,
            stream as cudarc::driver::sys::CUstream,
        )
        .result()
        .map_err(|e| cudarc_err("cudarc cuMemsetD8Async", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stage 5 (M3L5): every cudarc backend call site now surfaces driver
    /// errors as [`BoltError::CudaWithCode`] with the raw `CUresult` integer
    /// extracted from `DriverError(pub sys::CUresult)`. We don't have a way
    /// to trigger a real driver error from a host-only test (the FFI is
    /// stubbed under `cuda-stub` to `CUDA_ERROR_STUB`, which gets mapped to
    /// `BoltError::Other` upstream), but we can directly invoke the
    /// `cudarc_err` helper to assert its shape.
    #[test]
    fn cudarc_err_translates_to_cuda_with_code() {
        // cudarc::driver::sys::CUresult is a `#[repr(u32)]` enum whose first
        // non-success variant is `CUDA_ERROR_INVALID_VALUE = 1`. We build a
        // `DriverError` for it through the public `result()` shim on
        // `CUresult` so this test compiles against every cudarc-supported
        // CUDA version.
        let drv_err = cudarc::driver::sys::CUresult::CUDA_ERROR_INVALID_VALUE
            .result()
            .unwrap_err();
        let translated = cudarc_err("test ctx", drv_err);
        assert!(
            matches!(translated, BoltError::CudaWithCode { code: 1, .. }),
            "cudarc DriverError(CUDA_ERROR_INVALID_VALUE) must translate to \
             CudaWithCode {{ code: 1, .. }}, got: {translated:?}"
        );
    }

    /// Smoke test that the cudarc context comes up at all. Gated on
    /// `BOLT_BENCH_GPU=1` for the same reason as the engine tests —
    /// we can't acquire a context on a non-CUDA host.
    #[test]
    #[ignore = "gpu:mempool — set BOLT_BENCH_GPU=1 + run with --ignored"]
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
