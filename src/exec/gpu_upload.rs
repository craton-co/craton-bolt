// SPDX-License-Identifier: Apache-2.0

//! Shared async-memcpy / pinned-host-buffer helpers for executors.
//!
//! v0.7 promotion: the scalar-aggregate executor (`exec::aggregate`) shipped a
//! local `upload_primitive_values_async` helper as the v0.6 async-memcpy
//! pilot. As subsequent executors (filter, GROUP BY, joins, …) migrate off
//! the synchronous `GpuVec::from_slice` / `GpuVec::to_vec` calls, every site
//! wants the same `(slice, &CudaStream) -> GpuVec<T>` shape with the same
//! `--features cuda-stub` graceful degradation. Re-implementing the shim in
//! each module is a recipe for drift, so the helper has been lifted here.
//!
//! The semantics are unchanged from the aggregate-local version:
//!
//!   * Under real CUDA, [`upload_primitive_values_async`] issues
//!     `cuMemcpyHtoDAsync_v2` on the caller's `stream`, so subsequent kernel
//!     launches and D2H copies on the same stream serialize correctly without
//!     an explicit barrier. Pairing this with a [`PinnedHostBuffer`]-backed
//!     source unlocks true DMA overlap; the current Arrow value buffer is
//!     pageable, so the driver may still stage the copy internally, but the
//!     stream-scoped dispatch still avoids the implicit `cuCtxSynchronize`
//!     the legacy sync `from_slice` path took.
//!   * Under `--features cuda-stub`, the async FFI shim returns
//!     `CUDA_ERROR_STUB`, so the helper routes through the synchronous
//!     wrapper instead — both paths surface the same error at the FFI
//!     boundary in stub mode, but going through the sync wrapper means the
//!     failure happens in the same place it did before the pilot. This is
//!     the documented graceful degradation for the stub backend.
//!
//! [`PinnedHostBuffer`]: crate::cuda::buffer::PinnedHostBuffer

use bytemuck::Pod;

use crate::cuda::GpuVec;
use crate::error::BoltResult;
use crate::exec::launch::CudaStream;

/// Upload a host slice to the GPU on `stream` via the async wrappers,
/// falling back to the synchronous `from_slice` path under
/// `--features cuda-stub`.
///
/// See the module docs for the full rationale. This is the canonical
/// async-upload entry point for primitive (`Pod`) column data; every
/// executor that crosses the host→device boundary on a per-call stream
/// should funnel through here rather than rolling its own
/// `cfg(feature = "cuda-stub")` branch.
///
/// The `stream` is borrowed (not consumed); the caller still owns it and
/// is responsible for the final `stream.synchronize()` before any
/// downstream host-visible reads.
#[inline]
pub(crate) fn upload_primitive_values_async<T: Pod>(
    values: &[T],
    stream: &CudaStream,
) -> BoltResult<GpuVec<T>> {
    #[cfg(feature = "cuda-stub")]
    {
        // Stub backend has no real CUDA: prefer the sync path so the
        // call shape matches what existed before the async pilot. The
        // sync FFI shim itself still returns `CUDA_ERROR_STUB`, so this
        // is a graceful no-op routing change.
        let _ = stream;
        GpuVec::<T>::from_slice(values)
    }
    #[cfg(not(feature = "cuda-stub"))]
    {
        GpuVec::<T>::from_slice_async(values, stream.raw())
    }
}

/// Download a `GpuVec<T>` into a host `Vec<T>` via a pinned host buffer
/// and a single stream-synchronize. Mirrors the `to_pinned_async +
/// synchronize + to_vec` shape used in the scalar-aggregate executor.
///
/// `stream` is expected to already carry any prior work (the kernel
/// launch that wrote `dev`). This helper enqueues the D2H on the same
/// stream, synchronizes once, then copies the pinned bytes into a fresh
/// owned `Vec` for downstream host-side consumers. The pinned hop lets
/// the driver DMA directly without staging through a bounce buffer.
///
/// Under `--features cuda-stub` the pinned-async path is unavailable, so
/// we fall back to `GpuVec::to_vec()` which itself surfaces
/// `CUDA_ERROR_STUB` at the same FFI boundary.
#[inline]
pub(crate) fn download_to_host_pinned<T: Pod>(
    dev: &GpuVec<T>,
    stream: &CudaStream,
) -> BoltResult<Vec<T>> {
    #[cfg(feature = "cuda-stub")]
    {
        let _ = stream;
        dev.to_vec()
    }
    #[cfg(not(feature = "cuda-stub"))]
    {
        let pinned = dev.to_pinned_async(stream.raw())?;
        stream.synchronize()?;
        Ok(pinned.as_slice().to_vec())
    }
}

#[cfg(test)]
mod tests {
    // `use super::*` is gated on cuda-stub because every test in this
    // module is itself gated on `cuda-stub` — without that feature the
    // module is empty and an unconditional `use super::*` would warn as
    // unused.
    #[cfg(feature = "cuda-stub")]
    use super::*;

    /// Under `--features cuda-stub` the async helper still routes to the
    /// sync FFI shim, which returns `CUDA_ERROR_STUB` for any non-empty
    /// upload. The contract pinned here is: the helper does NOT panic, it
    /// surfaces a structured `BoltError` instead — exactly the same
    /// behaviour the synchronous `from_slice` path had before the
    /// promotion. Other executors (filter, GROUP BY) rely on this so they
    /// can use the helper unconditionally and let the stub backend
    /// degrade gracefully.
    ///
    /// The zero-length path doesn't touch the driver at all and so
    /// succeeds under cuda-stub too — pin that as well so a future
    /// change that always allocates (even on len == 0) doesn't silently
    /// regress empty-batch handling.
    #[cfg(feature = "cuda-stub")]
    #[test]
    fn stub_upload_zero_len_succeeds() {
        let stream = CudaStream::null_or_default();
        let xs: [i32; 0] = [];
        let v = upload_primitive_values_async::<i32>(&xs, &stream)
            .expect("zero-length upload never touches the driver");
        assert_eq!(v.len(), 0);
    }

    #[cfg(feature = "cuda-stub")]
    #[test]
    fn stub_upload_non_empty_returns_error_not_panic() {
        // Non-empty: the sync FFI shim returns CUDA_ERROR_STUB. The
        // helper must surface that as a `BoltResult::Err` rather than
        // panicking — every executor that adopts the helper depends on
        // this so the stub backend's error semantics stay regular.
        let stream = CudaStream::null_or_default();
        let xs: [i64; 4] = [1, 2, 3, 4];
        let r = upload_primitive_values_async::<i64>(&xs, &stream);
        assert!(
            r.is_err(),
            "cuda-stub backend must return Err for a real upload (got Ok)"
        );
    }
}
