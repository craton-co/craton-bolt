// SPDX-License-Identifier: Apache-2.0

//! Process-global pool of owned CUDA streams.
//!
//! ## Why this exists — the destroyed-stream use-after-free
//!
//! Executor entry points mint a *per-call* owned stream via
//! [`CudaStream::null_or_default`](crate::exec::launch::CudaStream::null_or_default)
//! so their H2D upload, kernel launches, and D2H download share one ordering
//! domain and can overlap with unrelated work on the NULL stream. The hazard:
//! a stream's raw handle **outlives** the `CudaStream` wrapper that minted it.
//! Every [`GpuBuffer`](crate::cuda::buffer::GpuBuffer) /
//! [`PinnedHostBuffer`](crate::cuda::buffer::PinnedHostBuffer) read or written
//! on the stream records the handle in its `used_streams` set and only fences
//! it (`cuStreamSynchronize`) or records a deferred-free event on it
//! (`cuEventRecord`) at `Drop` — which can be far later than the query that
//! created the stream (e.g. a *resident* `GpuTable` buffer freed by a later
//! `replace_table`, or a per-query input buffer dropped after a sub-orchestrator
//! that minted its own stream already returned).
//!
//! If `CudaStream::Drop` destroyed the stream (`cuStreamDestroy`) per call, that
//! later `cuEventRecord` / `cuStreamSynchronize` would run against a **dangling**
//! handle. That is undefined behaviour — and on the drivers we target it does
//! **not** reliably return `CUDA_ERROR_INVALID_HANDLE`: once the driver recycles
//! the freed stream object, the call dereferences freed driver memory and
//! **faults the host process** (`STATUS_ACCESS_VIOLATION`). Because the fault is
//! *inside* the FFI call, no Rust-side error check can intercept it — the
//! drop-time `cuCtxSynchronize` escalation in
//! [`fence_all_streams`](crate::cuda::buffer) only helps in the lucky case where
//! the call happens to return an error first.
//!
//! ## The fix — pool, never destroy-per-call
//!
//! We never destroy a per-call owned stream while a buffer might still name its
//! handle. `CudaStream::Drop` [`release`]s the handle back to this pool for
//! reuse; [`acquire`] hands it to the next `null_or_default`. Handles therefore
//! stay valid for the whole life of the CUDA context, so any `used_streams`
//! reference is always a live handle — `cuEventRecord` / `cuStreamSynchronize`
//! on it can never fault. The entire pool is destroyed exactly once, by
//! [`drain`] from [`CudaContext::Drop`](crate::cuda::cuda_sys::CudaContext),
//! which runs while the context is still current and *after* every `GpuBuffer`
//! has dropped (the `Engine` drops `_ctx` last), so at drain time no
//! `used_streams` set can still reference a pooled handle.
//!
//! ## Bounded size
//!
//! A released handle is reused by the next `acquire`, so the pool never holds
//! more than the *peak number of streams simultaneously alive* — bounded by
//! query concurrency, not by the number of queries run. Sequential queries
//! recycle a single handle.
//!
//! Handles are stored as `usize` (not `CUstream`, a raw pointer) purely so the
//! backing `Vec` is `Send` for the `static Mutex`; the value is the exact
//! `CUstream` bit pattern and is cast back on the way out. Nothing is ever
//! dereferenced here — handles are opaque to this module.

use std::sync::Mutex;

use crate::cuda::cuda_sys::{self, CUstream};

/// The free list of reusable owned-stream handles. See the module docs for the
/// safety argument (no handle is destroyed until [`drain`] at context teardown).
static STREAM_POOL: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// Take a reusable stream handle from the pool, or `None` if the pool is empty
/// (the caller then mints a fresh one via `cuStreamCreate`).
///
/// The returned handle is now exclusively owned by the caller until it is
/// [`release`]d, so two concurrent queries can never share one stream.
pub(crate) fn acquire() -> Option<CUstream> {
    // Tolerate a poisoned lock: the pool is a plain `Vec` of opaque handles
    // with no invariant a panicking holder could have left half-updated, so
    // recovering the guard is sound and avoids cascading a panic into every
    // future stream acquisition.
    STREAM_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .pop()
        .map(|raw| raw as CUstream)
}

/// Return an owned stream handle to the pool for reuse.
///
/// MUST NOT destroy the handle: a `GpuBuffer`'s `used_streams` may still name
/// it and will `cuStreamSynchronize` / `cuEventRecord` it at `Drop`, which is UB
/// (and can fault the host) on a destroyed stream. Pooled handles are destroyed
/// only by [`drain`] at context teardown. A null handle (the NULL stream, never
/// owned) is ignored.
pub(crate) fn release(raw: CUstream) {
    if raw.is_null() {
        return;
    }
    STREAM_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(raw as usize);
}

/// Destroy every pooled stream and empty the pool.
///
/// Called from [`CudaContext::Drop`](crate::cuda::cuda_sys::CudaContext) while
/// the context is still alive/current and after all `GpuBuffer`s have dropped,
/// so no `used_streams` set can still reference a pooled handle. Mirrors
/// `mem_pool::POOL.drain()`'s teardown role for device memory. Idempotent and a
/// cheap no-op when the pool is empty (e.g. the `cuda-stub` backend, where
/// `null_or_default` always falls back to the never-owned NULL stream).
pub fn drain() {
    let mut pool = STREAM_POOL.lock().unwrap_or_else(|e| e.into_inner());
    for raw in pool.drain(..) {
        let s = raw as CUstream;
        // SAFETY: `s` came from `cuStreamCreate` (via `CudaStream::new`) and is
        // not referenced by any live buffer at teardown (all `GpuBuffer`s have
        // dropped before the context tears the pool down). Destroying it now,
        // while the context is still current, reclaims it cleanly.
        let rc = unsafe { cuda_sys::cuStreamDestroy_v2(s) };
        if rc != cuda_sys::CUDA_SUCCESS {
            log::warn!(
                "craton-bolt: stream_pool::drain cuStreamDestroy_v2 returned {} \
                 (stream leaked at context teardown)",
                rc
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Host-only `acquire`/`release` semantics, driven on synthetic handles so
    /// nothing touches the driver. Kept as ONE test because the pool is a
    /// process-global `static`; splitting would let the default multi-threaded
    /// test harness race two bodies against the same shared free list.
    #[test]
    fn acquire_release_semantics() {
        // Start from a known-empty pool. We only ever inject synthetic handles
        // below and pop them ourselves, so this never calls the driver.
        while acquire().is_some() {}

        // A released handle is handed straight back by the next acquire...
        let fake = 0xABCD_usize as CUstream;
        release(fake);
        assert_eq!(acquire(), Some(fake), "released handle must be reused");
        assert_eq!(acquire(), None, "pool must be empty after the single reuse");

        // ...and a null handle (the never-owned NULL stream) is not pooled.
        release(std::ptr::null_mut());
        assert_eq!(acquire(), None, "null handle must not enter the pool");
    }
}
