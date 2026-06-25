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
//!
//! ## Context tagging — the multi-engine / multi-GPU use-after-free (W03)
//!
//! A `CUstream` is created *inside the CUDA context current at*
//! `cuStreamCreate` time and is valid only there; `cuStreamDestroy` reclaims it
//! against the context that owns it. This pool is process-global, so it can
//! simultaneously hold streams minted under **different** live contexts — e.g.
//! two `Engine`s, each with its own per-`Engine` `CudaContext`, possibly on
//! different GPUs. Two hazards follow if the pool is context-blind:
//!
//!   * [`drain`] (run from one context's teardown) would `cuStreamDestroy`
//!     *every* pooled handle, including streams still owned by another, live
//!     context — destroying another engine's streams out from under it. The
//!     next use of such a stream then faults the host exactly the way a
//!     destroyed per-call stream did before this pool existed.
//!   * [`acquire`] could hand a caller bound to context A a stream minted under
//!     context B — a wrong-context handle that launches into the wrong device.
//!
//! We close both by tagging each pooled entry with its owning `CUcontext`
//! (captured from `cuCtxGetCurrent` when the handle is [`release`]d, which runs
//! on the engine thread with the owning context current). [`acquire`] hands out
//! only a stream whose tag matches the active context; [`drain`] destroys only
//! the entries whose tag matches the context being torn down and leaves the
//! rest in place. Entries whose owning context could not be determined at
//! release time (the driver reported no current context — e.g. the `cuda-stub`
//! backend, where the pool is empty anyway) are tagged "unknown": [`acquire`]
//! never hands them out (it can't prove they match) and [`drain`] leaves them
//! (it can't prove they belong to the dying context), so they are conservatively
//! retained rather than risk a wrong-context destroy. Single-engine and the
//! `--features cudarc` shared-primary-context path are unaffected: every handle
//! is tagged with the one live context, so `acquire`/`drain` behave exactly as
//! the context-blind versions did.

use std::sync::Mutex;

use crate::cuda::cuda_sys::{self, CUstream};

/// One pooled entry: the reusable stream handle plus the `CUcontext` it was
/// minted under, both stored as `usize` so the `Vec` is `Send` for the static
/// `Mutex`. `ctx == 0` means the owning context was unknown at release time
/// (driver reported no current context); such entries are never handed out and
/// never destroyed (see the module docs).
#[derive(Clone, Copy)]
struct PooledStream {
    raw: usize,
    ctx: usize,
}

/// The free list of reusable owned-stream handles, each tagged with its owning
/// context. See the module docs for the safety argument (no handle is destroyed
/// until [`drain`] at context teardown, and then only entries owned by the
/// context being torn down).
static STREAM_POOL: Mutex<Vec<PooledStream>> = Mutex::new(Vec::new());

/// The `CUcontext` currently bound to the calling thread as a `usize` tag, or
/// `0` ("unknown") when no context is current or the driver query fails. `0` is
/// not a valid `CUcontext` (the driver never hands out a null context as the
/// *current* one — `cuCtxGetCurrent` reports null as "none"), so it is a safe
/// sentinel for "owning context could not be determined".
fn active_ctx_tag() -> usize {
    match cuda_sys::ctx_get_current() {
        Ok(Some(ctx)) => ctx as usize,
        // No context current, or the query failed (e.g. cuda-stub): unknown.
        Ok(None) | Err(_) => 0,
    }
}

/// Take a reusable stream handle owned by the *currently bound context* from
/// the pool, or `None` if no matching entry is free (the caller then mints a
/// fresh one via `cuStreamCreate`, which binds it to the active context).
///
/// Only an entry whose tagged context equals the active context is handed back,
/// so a caller bound to context A can never receive a stream minted under
/// context B (a wrong-context handle). Entries tagged "unknown" (`ctx == 0`)
/// and entries owned by other contexts are left in place. When the active
/// context itself cannot be determined we hand back nothing, forcing a fresh
/// mint — strictly safe.
///
/// The returned handle is now exclusively owned by the caller until it is
/// [`release`]d, so two concurrent queries can never share one stream.
pub(crate) fn acquire() -> Option<CUstream> {
    let want = active_ctx_tag();
    if want == 0 {
        // Active context unknown: can't prove any pooled entry matches, so
        // mint fresh rather than risk handing out a wrong-context stream.
        return None;
    }
    // Tolerate a poisoned lock: the pool is a plain `Vec` of opaque handles
    // with no invariant a panicking holder could have left half-updated, so
    // recovering the guard is sound and avoids cascading a panic into every
    // future stream acquisition.
    let mut pool = STREAM_POOL.lock().unwrap_or_else(|e| e.into_inner());
    // Search from the back (cheap removal, and the most-recently-released
    // handle is the most cache-warm) for an entry owned by the active context.
    if let Some(pos) = pool.iter().rposition(|e| e.ctx == want) {
        return Some(pool.swap_remove(pos).raw as CUstream);
    }
    None
}

/// Return an owned stream handle to the pool for reuse, tagging it with the
/// context currently bound to the calling thread.
///
/// MUST NOT destroy the handle: a `GpuBuffer`'s `used_streams` may still name
/// it and will `cuStreamSynchronize` / `cuEventRecord` it at `Drop`, which is UB
/// (and can fault the host) on a destroyed stream. Pooled handles are destroyed
/// only by [`drain`] at context teardown. A null handle (the NULL stream, never
/// owned) is ignored.
///
/// `release` runs from `CudaStream::Drop` on the engine thread, which has the
/// stream's owning context current, so `cuCtxGetCurrent` captures the correct
/// owner. If the driver reports no current context the entry is tagged
/// "unknown" (`ctx == 0`) and will be conservatively retained — never handed
/// out by [`acquire`], never destroyed by [`drain`] (see the module docs).
pub(crate) fn release(raw: CUstream) {
    if raw.is_null() {
        return;
    }
    let ctx = active_ctx_tag();
    STREAM_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(PooledStream {
            raw: raw as usize,
            ctx,
        });
}

/// Destroy the pooled streams owned by the **currently bound context** and
/// remove them from the pool, leaving every other context's entries intact.
///
/// Called from [`CudaContext::Drop`](crate::cuda::cuda_sys::CudaContext) while
/// the context being torn down is still alive/current and after all
/// `GpuBuffer`s have dropped, so no `used_streams` set can still reference a
/// pooled handle owned by *this* context. Mirrors `mem_pool::POOL.drain()`'s
/// teardown role for device memory.
///
/// Crucially this destroys ONLY the entries whose tag matches the active
/// context: a process-global pool may also hold streams owned by another, still
/// live context (a second `Engine`, possibly on another GPU), and destroying
/// those here would be a use-after-free the moment that engine next touches
/// one. Entries tagged "unknown" (`ctx == 0`) are also left in place — we can't
/// prove they belong to the dying context. Idempotent and a cheap no-op when no
/// matching entry is present (e.g. the `cuda-stub` backend, where
/// `null_or_default` always falls back to the never-owned NULL stream so the
/// pool stays empty).
pub fn drain() {
    let me = active_ctx_tag();
    let mut pool = STREAM_POOL.lock().unwrap_or_else(|e| e.into_inner());
    // If we can't identify the dying context, destroy nothing — better to leak
    // for the remainder of the process than to free another context's streams.
    if me == 0 {
        return;
    }
    // Partition: retain entries we do NOT own; destroy the ones we do.
    let mut retained: Vec<PooledStream> = Vec::with_capacity(pool.len());
    for entry in pool.drain(..) {
        if entry.ctx != me {
            // Owned by another (live) context or unknown — must not destroy.
            retained.push(entry);
            continue;
        }
        let s = entry.raw as CUstream;
        // SAFETY: `s` came from `cuStreamCreate` (via `CudaStream::new`) under
        // the context now being torn down (matching `ctx` tag) and is not
        // referenced by any live buffer at teardown (all `GpuBuffer`s have
        // dropped before the context tears the pool down). Destroying it now,
        // while that context is still current, reclaims it cleanly.
        let rc = unsafe { cuda_sys::cuStreamDestroy_v2(s) };
        if rc != cuda_sys::CUDA_SUCCESS {
            log::warn!(
                "craton-bolt: stream_pool::drain cuStreamDestroy_v2 returned {} \
                 (stream leaked at context teardown)",
                rc
            );
        }
    }
    *pool = retained;
}

/// Number of entries currently parked in the pool (test/observability hook).
#[cfg(test)]
fn pool_len() -> usize {
    STREAM_POOL.lock().unwrap_or_else(|e| e.into_inner()).len()
}

/// Test seam: push a synthetic `(raw, ctx)` entry without touching the driver.
#[cfg(test)]
fn push_tagged(raw: usize, ctx: usize) {
    STREAM_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(PooledStream { raw, ctx });
}

/// Test seam: drop every entry whose tag matches `ctx` *without* calling the
/// driver's `cuStreamDestroy_v2`, so the context-partitioning logic of
/// [`drain`] can be exercised on synthetic handles. Returns how many entries it
/// removed. Mirrors `drain`'s "only my context" partition exactly.
#[cfg(test)]
fn drain_tagged_no_driver(ctx: usize) -> usize {
    let mut pool = STREAM_POOL.lock().unwrap_or_else(|e| e.into_inner());
    let before = pool.len();
    pool.retain(|e| e.ctx != ctx);
    before - pool.len()
}

/// Test seam: empty the pool without touching the driver.
#[cfg(test)]
fn clear_no_driver() {
    STREAM_POOL
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// Process-wide gate so the tests below don't race each other on the
    /// shared `STREAM_POOL` static. Each test takes this lock for its whole
    /// body and starts from a known-empty pool.
    static TEST_GATE: StdMutex<()> = StdMutex::new(());

    /// Context tagging: `acquire` hands back only a stream whose tag matches
    /// the requested context, leaves other contexts' (and unknown-tagged)
    /// entries in place, and `acquire`/`release` round-trip via the public
    /// (real-context) helpers can't run host-only, so we drive the matching
    /// logic on synthetic tags through the test seams.
    ///
    /// We can't call the production `acquire`/`release` here because they query
    /// `cuCtxGetCurrent`, which has no current context under `cuda-stub`. The
    /// seams (`push_tagged` / `drain_tagged_no_driver`) reproduce the exact
    /// tag-matching that `acquire`/`drain` perform, minus the driver calls.
    #[test]
    fn drain_only_destroys_its_own_context() {
        let _g = TEST_GATE.lock().unwrap_or_else(|e| e.into_inner());
        clear_no_driver();

        const CTX_A: usize = 0xA000;
        const CTX_B: usize = 0xB000;
        const UNKNOWN: usize = 0; // owning context could not be determined

        // Two streams under context A, one under B, one unknown.
        push_tagged(0x1111, CTX_A);
        push_tagged(0x2222, CTX_A);
        push_tagged(0x3333, CTX_B);
        push_tagged(0x4444, UNKNOWN);
        assert_eq!(pool_len(), 4);

        // Tearing down context A removes ONLY A's two streams; B's and the
        // unknown one survive (destroying them would be a UAF / a guess).
        let removed_a = drain_tagged_no_driver(CTX_A);
        assert_eq!(
            removed_a, 2,
            "drain must reclaim exactly context A's streams"
        );
        assert_eq!(
            pool_len(),
            2,
            "context B's stream and the unknown-tagged one must survive A's teardown"
        );

        // Tearing down B removes B's; the unknown one still survives (drain
        // with a real `me == 0` returns early and touches nothing — modelled
        // here by simply never calling drain with ctx 0).
        let removed_b = drain_tagged_no_driver(CTX_B);
        assert_eq!(
            removed_b, 1,
            "drain must reclaim exactly context B's stream"
        );
        assert_eq!(
            pool_len(),
            1,
            "the unknown-tagged entry is conservatively retained"
        );

        clear_no_driver();
    }

    /// `release(null)` (the never-owned NULL stream) is dropped, not pooled.
    #[test]
    fn release_ignores_null_handle() {
        let _g = TEST_GATE.lock().unwrap_or_else(|e| e.into_inner());
        clear_no_driver();

        release(std::ptr::null_mut());
        assert_eq!(pool_len(), 0, "null handle must not enter the pool");

        clear_no_driver();
    }

    /// `acquire` returns `None` when the active context is unknown (the
    /// `cuda-stub` reality), so the caller mints a fresh stream rather than
    /// risk being handed a wrong-context handle. We also confirm it never hands
    /// back an entry tagged for a different context.
    #[test]
    fn acquire_refuses_unknown_and_foreign_context() {
        let _g = TEST_GATE.lock().unwrap_or_else(|e| e.into_inner());
        clear_no_driver();

        // Inject a stream tagged for some specific context.
        push_tagged(0x5555, 0xC000);

        // Under cuda-stub `active_ctx_tag()` is 0 (no context), so `acquire`
        // must refuse and leave the foreign entry untouched.
        #[cfg(feature = "cuda-stub")]
        {
            assert_eq!(active_ctx_tag(), 0, "no context current under cuda-stub");
            assert!(
                acquire().is_none(),
                "acquire must refuse when the context is unknown"
            );
            assert_eq!(
                pool_len(),
                1,
                "the foreign-context entry must remain pooled"
            );
        }

        clear_no_driver();
    }
}
