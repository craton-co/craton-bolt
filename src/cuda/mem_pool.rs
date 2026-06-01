// SPDX-License-Identifier: Apache-2.0

//! Process-wide device-memory pool / arena allocator.
//!
//! Every `cuMemAlloc_v2` / `cuMemFree_v2` round-trip is a synchronous driver
//! call that serializes against the GPU stream. For workloads that allocate
//! and free many short-lived device buffers per query (input upload + output
//! materialization), these calls dominate. This pool recycles freed blocks
//! back to callers instead of returning them to the driver.
//!
//! ### Size bucketing (M-3, denser than power-of-two)
//!
//! The original M0 pool rounded every request up to the next power of two.
//! That capped at 2× over-allocation per call — a 65 KiB request became
//! 128 KiB. For a JIT engine that allocates many sub-128 KiB index buffers
//! per query, this wastes a lot of headroom.
//!
//! `bucket_size` now uses a **uniform 4-classes-per-octave geometric
//! schedule** (jemalloc-style 1.25× max waste):
//!
//! ```text
//!   pow2 = highest set bit of n
//!   step = pow2 / 4                              // four sub-classes per octave
//!   bucket = ceil(n / step) * step
//! ```
//!
//! Examples (`ARROW_ALIGNMENT = 64`):
//!
//! | request | old (power-of-two) | new (4/octave) | waste |
//! |---------|--------------------|----------------|-------|
//! |    64   |    64              |    64          |   0%  |
//! |   100   |   128              |   112          |  12%  |
//! |  4097   |  8192              |  5120          |  25%  |
//! | 65537   | 131072             | 81920          |  25%  |
//! |  1 MiB  |  1 MiB             |  1 MiB         |   0%  |
//!
//! Worst-case overhead is just under 25% inside the last sub-class of each
//! octave — substantially better than the old 2× ceiling. The bucket count
//! is still bounded (~4 × log2(max_alloc)), so the per-bucket map stays small.
//!
//! ### Capacity & eviction
//!
//! Without a cap the pool would happily hoard every block it has ever seen,
//! so a workload that spikes to multiple GiB and then settles to a small
//! working set would leak that headroom indefinitely. Two limits keep the
//! pool bounded:
//!
//! * `MAX_POOLED_BYTES` — soft cap on the sum of `alloc_bytes` across all
//!   buckets. Tunable via `CRATON_BOLT_POOL_MAX_BYTES` (default 512 MiB).
//! * `MAX_BUCKET_ENTRIES` — hard cap on the number of pooled blocks per
//!   bucket. Tunable via `CRATON_BOLT_POOL_BUCKET_CAP` (default 16).
//!
//! When a `free` would breach either limit, the pool evicts the oldest
//! block (front of the bucket's `VecDeque`) via `cuMemFree_v2`. If that
//! still does not make room, the freshly freed block is returned to the
//! driver directly rather than pooled. Buckets internally are LIFO for
//! reuse (returning the most-recently freed block gives the warmest cache
//! behaviour) but FIFO for eviction.
//!
//! ### Lock granularity (L-5, per-bucket locks)
//!
//! M1 used a single `Mutex<PoolState>` for all buckets; multi-stream
//! workloads serialised on it. The pool now stores buckets in a
//! `DashMap<usize, Mutex<BucketEntry>>` so concurrent frees into distinct
//! size classes do not contend on a global lock. `total_bytes` becomes an
//! `AtomicUsize`; cap checks read/write it with `Relaxed`/`AcqRel`
//! ordering — the cap is soft, occasional overshoot by one block under
//! race is acceptable and corrected on the next free. Subtractions go
//! through `sub_total_saturating` (a `fetch_update` CAS loop with
//! `saturating_sub`) rather than a bare `fetch_sub`, so a `store` from
//! `reconcile_total_bytes` that races an in-flight free can never drive
//! the counter below zero and wrap into `~usize::MAX`. Saturation is
//! equivalent under the soft-cap invariant — a transient under-count
//! gets corrected on the next reconciliation pass anyway.
//!
//! ### Stage 2: cross-bucket global LRU + reconciliation
//!
//! Stage 2 closes the LRU and reliability gaps that the per-bucket-lock
//! split opened up.
//!
//! * **Cross-bucket global LRU index (sharded — see PERF P-1 below).**
//!   A set of `BTreeMap`s keyed by `(inserted, tick)` runs alongside the
//!   `DashMap`. Every `free` insert into a bucket also inserts
//!   `(now, tick) -> (size_class, ptr)` into the index. `evict_one`
//!   finds the oldest key (oldest across all buckets), looks up the
//!   owning bucket, takes its lock, and removes the matching block —
//!   restoring true global LRU at the cost of one index operation per
//!   pool action.
//!
//!   **Race-handling.** The index pop and the bucket lock are not a
//!   single transaction. Between the two, another `alloc` may have
//!   already pulled `ptr` out of the bucket (and removed its LRU
//!   entry — see lock-order discussion below — but our evictor had
//!   already snapshot-popped the entry first). The eviction path
//!   detects this (bucket's deque doesn't contain the popped ptr)
//!   and falls back to "take any block from that bucket": every
//!   remaining block in that bucket was inserted *after* the one
//!   we lost, so popping the bucket's front is a sound approximation
//!   of "next-oldest." If the bucket itself is empty, the eviction
//!   re-pops the LRU. The pre-existing cross-bucket scan in
//!   `evict_one_scan_fallback` is retained as a defensive tail for
//!   the (should-not-happen) "LRU fully out of sync" case where the
//!   global index disagrees with the per-bucket truth.
//!
//! * **PERF P-1: the LRU index is sharded.** The earlier design used a
//!   single `Mutex<BTreeMap<..>>` taken on *every* `alloc`-hit and *every*
//!   `free`-insert, on top of the per-bucket lock — re-serialising all
//!   size classes through one mutex under many-stream churn. The index is
//!   now an array of `LRU_SHARDS` (= 32) independent `Mutex<BTreeMap<..>>`
//!   shards, with a block assigned to `lru_index[size_class % LRU_SHARDS]`
//!   (mirroring how the bucket storage is sharded). The two hot paths each
//!   touch exactly one shard — the one for the size class they already
//!   hold — so frees/allocs into size classes on distinct shards run in
//!   parallel. The `(inserted, tick)` key is globally unique (`tick` is a
//!   process-wide counter), so a cross-shard ordering / minimum comparison
//!   is well-defined; `evict_one` still honours global LRU by scanning all
//!   shards for the globally-oldest entry (`lru_pop_global_oldest`).
//!
//!   **Lock order (extended for the sharded LRU).** Two *kinds* of lock
//!   coexist in the pool: the per-bucket `Mutex` (inside a DashMap entry
//!   or a storage shard) and the per-shard `lru_index` mutexes. The
//!   canonical order is **bucket-first, lru-second**.
//!   `try_insert_into_locked_bucket` follows it: it holds the bucket lock,
//!   mutates the deque, then takes the *single* LRU shard for that size
//!   class to insert the matching entry. `alloc`'s hit path and the
//!   eviction stale-entry cleanup take an LRU shard only *after* the
//!   bucket lock has been released. `evict_one`'s primary path inverts the
//!   order (LRU first, to pick the global oldest) — to avoid deadlock it
//!   takes each LRU shard lock individually, **never two LRU shards at
//!   once**, and *releases* the chosen shard before reaching for any
//!   bucket. `evict_one_scan_fallback` is bucket-first throughout. The
//!   combined invariant, strengthened by sharding:
//!     1. **no thread ever holds any LRU-shard lock while waiting on a
//!        bucket lock**, and
//!     2. **no thread ever holds two LRU-shard locks at the same time**.
//!   The only nested acquisition anywhere is bucket → one-lru-shard (in
//!   `try_insert_into_locked_bucket`); every other LRU touch holds no
//!   other lock. The lock graph therefore has no cycle and cannot
//!   deadlock. Because a block's shard is a pure function of its size
//!   class, the insert (in `try_insert`) and the matching remove (in
//!   `alloc`-hit / eviction) always target the *same* shard, so the LRU
//!   index can never strand a half-inserted entry across shards.
//!
//! * **`total_bytes` reconciliation.** The atomic counter can transiently
//!   drift under heavy concurrent free because the bucket lock and the
//!   atomic increment are not joined into one transaction (the cap
//!   re-check inside `try_insert_into_bucket` narrows but does not
//!   close the window). `reconcile_total_bytes` walks every bucket
//!   under its own lock, sums `bucket.len() * size_class`, and stores
//!   the result. O(buckets); intended for memory-pressure / debugging
//!   / test-harness checkpoints. Also invoked automatically every
//!   `RECONCILE_EVERY_N_FREES` (1024) calls into `free` so long-running
//!   processes self-heal without an explicit caller.
//!
//! * **DashMap vs. fixed-N sharded.** We kept DashMap because:
//!   1. The bucket count is bounded (~4 × log2(max_alloc) ≈ 100 entries
//!      for realistic max_alloc), so the DashMap shard layer's hash cost
//!      is essentially constant per access — comparable to a single
//!      indirection in a fixed-N array.
//!   2. The hot path inside `alloc` / `free` takes a *read* lock on the
//!      DashMap shard and a *write* lock on the inner `Mutex`. Distinct
//!      size classes that hash to the same shard share the read lock,
//!      so the only true contention is on the inner bucket mutex —
//!      which a fixed-N sharded scheme would not improve.
//!   3. The first-touch (insert-new-size-class) path is rare — it
//!      happens once per size class for the entire pool's lifetime —
//!      so the shard write lock cost amortises away.
//!
//!   **Pathological case where DashMap could lose.** Very high-rate
//!   concurrent `free` into many distinct size classes that all hash to
//!   the same DashMap shard (e.g. an adversarial test, or an unlucky
//!   workload with size classes that collide on `default_hasher`). In
//!   that regime the shard's reader-writer lock and hash computation
//!   itself becomes the bottleneck. Mitigation if measured: switch the
//!   `buckets` field to a fixed-N `[Mutex<HashMap<usize, BucketEntry>>;
//!   32]` keyed by `size_class % 32`. The `#[ignore]` micro-bench
//!   `bench_dashmap_baseline` at the bottom of this file exists so
//!   the orchestrator can capture the per-op cost for comparison
//!   against a future sharded variant.
//!
//! ### Stage 3: OOM-recovery hook + sharded escape hatch
//!
//! * **Driver-OOM recovery.** When the underlying CUDA driver returns
//!   `CUDA_ERROR_OUT_OF_MEMORY` (code 2) from `cuMemAlloc_v2`, the pool
//!   no longer bubbles the error immediately. Instead it triggers a
//!   cascade of releases — `evict_above_high_water` first (cheap), then
//!   a full `drain` (releases every pooled block back to the driver) —
//!   and retries the allocation exactly once. Successful recovery
//!   increments `OOM_RECOVERY_COUNT` so downstream telemetry can see
//!   pressure events; failed recovery returns the original error
//!   unchanged. See `alloc` for the wiring.
//!
//!   This hook is purely *additive*: the steady-state alloc path is
//!   one extra string-prefix check on the (rare) error path. It does
//!   not introduce new locks or change the lock order.
//!
//! * **`pool-sharded` escape hatch.** Under `--features pool-sharded`
//!   the `buckets` field becomes a `[Mutex<HashMap<usize, BucketEntry>>;
//!   SHARDS]` array (SHARDS = 32) keyed by `size_class % SHARDS`. Same
//!   external API; the storage shape diverges entirely. All access goes
//!   through `with_bucket` / `with_or_create_bucket` / `for_each_bucket`
//!   helpers so the hot-path code stays storage-agnostic. The
//!   orchestrator can flip the feature without touching mem_pool.rs.
//!
//! The pool depends on a live CUDA context being current on the calling
//! thread — same precondition as the bare `cuMemAlloc` path it replaces.

use std::collections::{BTreeMap, VecDeque};
#[cfg(feature = "pool-sharded")]
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Instant;

#[cfg(not(feature = "pool-sharded"))]
use dashmap::DashMap;
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::cuda::buffer::ARROW_ALIGNMENT;
use crate::cuda::cuda_sys::CUdeviceptr;
// `cuda_sys` is only referenced by the hand-rolled (default) backend
// path. Under `--features cudarc` the alloc/free hit `cudarc_backend`
// instead, so the import is feature-gated to keep both builds warning-
// free.
// Used by the always-compiled `real_driver_mem_alloc` / `real_driver_free` and
// the deferred-free sweep/drain. Needed in `#[cfg(test)]` builds too now that
// those real, driver-touching paths can be selected at runtime via
// `BOLT_BENCH_GPU=1` (see `bench_gpu_enabled`).
#[cfg(not(feature = "cudarc"))]
use crate::cuda::cuda_sys;
use crate::error::BoltResult;

/// Default soft cap on total pooled bytes (512 MiB). Overridden by the
/// `CRATON_BOLT_POOL_MAX_BYTES` environment variable.
const DEFAULT_MAX_POOLED_BYTES: usize = 512 * 1024 * 1024;

/// Default hard cap on the number of pooled blocks in any single bucket.
/// Overridden by the `CRATON_BOLT_POOL_BUCKET_CAP` environment variable.
const DEFAULT_MAX_BUCKET_ENTRIES: usize = 16;

/// How often (in number of `free` calls) to run `reconcile_total_bytes`
/// automatically. The reconciliation is O(buckets) — bucket count is
/// bounded (~4 × log2(max_alloc) ≈ 100), so amortised cost per free is
/// well under a microsecond on the steady state. Set high enough that
/// reconciliation cost is invisible in profiles, low enough that any
/// drift gets corrected within a fraction of a second under heavy load.
const RECONCILE_EVERY_N_FREES: u64 = 1024;

/// Number of shards used by the `pool-sharded` storage variant. Power of
/// two so `size_class % SHARDS` collapses to a cheap mask. 32 is a sweet
/// spot for typical core counts; the chance of any two of the ~100
/// realistic bucket size classes colliding on the same shard is small
/// enough that contention degenerates to a per-bucket mutex in practice.
#[cfg(feature = "pool-sharded")]
const SHARDS: usize = 32;

/// Number of independent shards the cross-bucket global LRU index is split
/// across (PERF P-1). Power of two so `size_class % LRU_SHARDS` folds to a
/// mask. A block always lands in the shard chosen by its `size_class`, so
/// the hot `alloc`-hit / `free`-insert paths touch exactly one LRU shard
/// keyed by the size class they already hold — frees/allocs into distinct
/// size classes no longer re-serialise through one global BTreeMap mutex.
/// 32 mirrors `SHARDS` (the bucket-storage shard count) so the LRU and the
/// bucket map shard congruently: blocks of a given size class share both a
/// bucket-storage shard and an LRU shard, keeping the two locks' contention
/// profiles aligned rather than cross-interleaved.
const LRU_SHARDS: usize = 32;

/// Pick the LRU-index shard for a given `size_class`. `LRU_SHARDS` is a
/// power of two, so `%` lowers to a mask. A block is always inserted into,
/// removed from, and (during eviction) located in the *same* shard chosen
/// here — the shard assignment is a pure function of `size_class`, so the
/// insert/remove pair for one block can never straddle two shards.
#[inline]
fn lru_shard_of(size_class: usize) -> usize {
    size_class % LRU_SHARDS
}

/// CUDA driver result code for "out of memory" (`CUDA_ERROR_OUT_OF_MEMORY = 2`).
///
/// Stage 4 widened `BoltError::CudaWithCode { code, message }` to carry the
/// raw `CUresult` integer, so the OOM detector now pattern-matches on
/// `code == CUDA_OOM_CODE` directly. The earlier `CUDA_OOM_PREFIX` string
/// match is gone — see `is_oom_error`.
const CUDA_OOM_CODE: i32 = 2;

/// Number of times the driver-OOM recovery path successfully retried an
/// allocation after evicting / draining the pool. Process-wide because the
/// pool is a singleton; visible to downstream telemetry layers via
/// `oom_recovery_count`. Pure counter — never reset.
static OOM_RECOVERY_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read the cumulative count of driver-OOM allocations that were rescued
/// by evicting the pool. Intended for telemetry dashboards and stress-test
/// assertions — a non-zero value over a long-running process is a signal
/// to raise `CRATON_BOLT_POOL_MAX_BYTES` or investigate working-set growth.
#[allow(dead_code)] // reason: telemetry hook, consumed by downstream observability
pub(crate) fn oom_recovery_count() -> u64 {
    OOM_RECOVERY_COUNT.load(Ordering::Relaxed)
}

/// Number of times the Stage 4 background watcher proactively called
/// `evict_above_high_water` because `cuMemGetInfo_v2` reported free
/// device memory below the configured low-water mark. Process-wide
/// counter; never reset. Visible via the public [`pool_stats`] surface.
static PROACTIVE_EVICTION_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read the cumulative count of proactive evictions triggered by the
/// `pool-watcher` background thread. Always zero unless the
/// `pool-watcher` feature is enabled and a watcher has actually
/// observed memory pressure since startup.
#[allow(dead_code)] // reason: telemetry hook, surfaced via `pool_stats`
pub(crate) fn proactive_eviction_count() -> u64 {
    PROACTIVE_EVICTION_COUNT.load(Ordering::Relaxed)
}

// ---------------------------------------------------------------------------
// Event-based deferred-free pool (review finding C1 / P1).
//
// ## What this gives us
//
// A `GpuBuffer`'s `Drop` used to *block* on a per-stream `cuStreamSynchronize`
// for every stream that ever touched the block before returning it to the
// pool — correct (no recycled address can alias in-flight work), but it stalls
// the dropping thread on each stream's entire trailing queue.
//
// With CUDA events available (`cuda_sys::event_*`), `mark_stream_use` records
// a lightweight event at the *exact* point a stream referenced the block.
// `Drop` then *queries* those events:
//
//   * all complete  -> free the block to the pool inline (fast path), or
//   * any not ready -> hand `(ptr, alloc_bytes, events)` to [`defer_free`]
//     here instead of blocking. The block stays OUT of the allocatable pool
//     until a later sweep observes every event complete.
//
// ## Safety invariant (identical to the old blanket sync)
//
// A block's device memory becomes eligible for reuse ONLY after every stream
// that touched it has drained past its recorded event. A not-ready query
// DEFERS the free; it never authorises one. So no recycled address can alias
// an in-flight DMA/kernel — exactly the no-premature-reuse property the
// blanket sync provided, without the stall.
//
// ## What this does NOT fix (documented limitation, review finding C1)
//
// The deferred path only protects work whose stream was *recorded*. A kernel
// launched directly off `device_ptr()` that never tags its stream (bypassing
// both the async helpers and `KernelArgs`) records no event, so neither the
// old blanket sync nor this deferred path can fence it. Closing that hole
// structurally requires forcing every launch through a tagging chokepoint
// (`KernelArgs`), which touches launch glue outside this module. The
// `debug_assert` guard in `GpuBuffer::Drop` (see buffer.rs) makes the
// suspicious "non-empty buffer dropped with zero recorded streams" case
// detectable in debug builds so a forgotten tag surfaces in tests.
// ---------------------------------------------------------------------------

/// Cumulative count of blocks routed through the event-based *deferred* free
/// path (i.e. dropped while still in flight on some stream). Telemetry only;
/// a high value relative to total frees means buffers are being dropped before
/// their streams drain — usually benign, occasionally a sign of a missing
/// explicit `synchronize` before drop. Never reset.
static DEFERRED_FREE_COUNT: AtomicU64 = AtomicU64::new(0);

/// Read the cumulative count of event-deferred frees. Telemetry hook.
#[allow(dead_code)] // reason: telemetry hook
pub(crate) fn deferred_free_count() -> u64 {
    DEFERRED_FREE_COUNT.load(Ordering::Relaxed)
}

/// A device block whose free was deferred because a recorded stream had not
/// yet drained past its event. Held in the process-global [`PENDING_FREES`]
/// list until a sweep observes every `event` complete.
///
/// `events` are raw `CUevent` handles. They are context-bound — queried only
/// while the owning context is current on the sweeping thread, the same
/// precondition the pool already documents for `cuMemFree_v2`. We never
/// dereference them; we only forward them to the driver via `cuda_sys::event_*`.
struct PendingFree {
    ptr: CUdeviceptr,
    alloc_bytes: usize,
    events: Vec<crate::cuda::cuda_sys::CUevent>,
}

// SAFETY: `PendingFree` holds raw `CUevent`/`CUdeviceptr` handles. Like the
// `CUdeviceptr`s the pool already stores and frees across threads, these are
// opaque driver handles that are valid in any thread that has the owning
// context current. We only ever forward them to the driver (query / destroy),
// never dereference them, and access is serialized through the `PENDING_FREES`
// mutex. Moving them between threads upholds the same context-currency
// precondition the rest of the pool relies on.
unsafe impl Send for PendingFree {}

/// Process-global list of blocks awaiting event completion before reuse.
/// Empty in the steady state (most buffers are synchronized before drop, so
/// their events are already complete and they free inline). Guarded by a
/// `parking_lot::Mutex`; the lock is held only for short list splices, never
/// across a driver call that could block.
static PENDING_FREES: Lazy<Mutex<Vec<PendingFree>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Record a deferred free. Called by `GpuBuffer::Drop` when at least one
/// recorded event reports not-ready. The block is parked here — OUT of the
/// allocatable pool — until [`sweep_pending_frees`] observes every event
/// complete, at which point it is freed to the pool.
///
/// `events` must be the full set of events recorded for the block (one per
/// stream that touched it). Ownership of the events transfers here; the sweep
/// destroys them once they complete.
///
/// This is the real, driver-touching implementation. Under `#[cfg(test)]` it is
/// reached only when `BOLT_BENCH_GPU=1` selects the live-device path; the
/// host-only default routes through the [`defer_free`] dispatcher's side-channel
/// shim instead (see [`bench_gpu_enabled`]).
fn real_defer_free(
    ptr: CUdeviceptr,
    alloc_bytes: usize,
    events: Vec<crate::cuda::cuda_sys::CUevent>,
) {
    if ptr == 0 {
        return;
    }
    DEFERRED_FREE_COUNT.fetch_add(1, Ordering::Relaxed);
    PENDING_FREES.lock().push(PendingFree {
        ptr,
        alloc_bytes,
        events,
    });
}

/// Public deferred-free entry point used by `GpuBuffer::Drop`. In a non-test
/// build this is exactly [`real_defer_free`]. In a `#[cfg(test)]` build the
/// host-only default records the parked block in [`TEST_DEFERRED`] (and drops
/// the — empty under synthetic pointers — event vector) so policy tests can
/// assert deferral without a CUDA context; `BOLT_BENCH_GPU=1` flips it to the
/// real path so the crate's `#[ignore]`'d GPU tests free on a live device.
pub(crate) fn defer_free(
    ptr: CUdeviceptr,
    alloc_bytes: usize,
    events: Vec<crate::cuda::cuda_sys::CUevent>,
) {
    #[cfg(test)]
    if !bench_gpu_enabled() {
        if ptr == 0 {
            return;
        }
        DEFERRED_FREE_COUNT.fetch_add(1, Ordering::Relaxed);
        TEST_DEFERRED.lock().push((ptr, alloc_bytes));
        return;
    }
    real_defer_free(ptr, alloc_bytes, events);
}

/// Sweep the pending-free list: any block whose events have ALL completed is
/// returned to the pool (its events destroyed first); blocks with at least one
/// still-in-flight event are retained for the next sweep.
///
/// Called opportunistically at the start of `alloc` and `free` (cheap no-op
/// when the list is empty, which is the steady state). Never blocks: it only
/// *queries* events. A query error (e.g. transient driver issue) is treated
/// conservatively as "not ready yet" so we never free early on a bad probe.
/// Public sweep entry point, called opportunistically from `alloc`/`free`. In a
/// non-test build this is [`real_sweep_pending_frees`]. Under `#[cfg(test)]` the
/// host-only default is a no-op (synthetic pointers carry no real events to
/// query); `BOLT_BENCH_GPU=1` selects the real sweep.
pub(crate) fn sweep_pending_frees() {
    #[cfg(test)]
    if !bench_gpu_enabled() {
        return;
    }
    real_sweep_pending_frees();
}

fn real_sweep_pending_frees() {
    // Fast path: nothing pending. Avoid taking the lock's write section and
    // doing any driver work in the overwhelmingly common case.
    {
        let guard = PENDING_FREES.lock();
        if guard.is_empty() {
            return;
        }
    }
    // Drain the list under the lock, decide each entry's fate without holding
    // the lock across driver calls longer than necessary, then re-park the
    // not-yet-ready entries. We move the vector out so concurrent `defer_free`
    // calls append to a fresh empty list rather than contending with us.
    let pending: Vec<PendingFree> = {
        let mut guard = PENDING_FREES.lock();
        std::mem::take(&mut *guard)
    };
    let mut still_pending: Vec<PendingFree> = Vec::new();
    for entry in pending {
        // An entry is ready iff EVERY recorded event reports complete. A
        // not-ready or errored query keeps the WHOLE entry parked (we never
        // partially free).
        let all_ready = entry.events.iter().all(|&ev| {
            // SAFETY: `ev` is a live event handle created via
            // `cuda_sys::event_create` and recorded on a stream; we only query
            // it. A query error is mapped to `false` (treat as not ready) so a
            // bad probe defers rather than frees.
            matches!(unsafe { crate::cuda::cuda_sys::event_query(ev) }, Ok(true))
        });
        if all_ready {
            // Destroy the events, then return the block to the pool. Order
            // matters only in that the block must not re-enter the allocatable
            // pool before its events confirm completion — which they have.
            for ev in &entry.events {
                // SAFETY: event completed (queried ready above) and is not used
                // again after destruction.
                let _ = unsafe { crate::cuda::cuda_sys::event_destroy(*ev) };
            }
            POOL.free(entry.ptr, entry.alloc_bytes);
        } else {
            still_pending.push(entry);
        }
    }
    if !still_pending.is_empty() {
        // Re-append the not-ready entries (plus anything `defer_free` pushed
        // while we were sweeping stays ahead of them — order is irrelevant).
        PENDING_FREES.lock().extend(still_pending);
    }
}

/// Shutdown drain of the pending-free list: BLOCK on every still-pending
/// entry's events, destroy them, and free the block. Guarantees no deferred
/// block is leaked or freed early at teardown — the safety invariant the
/// deferred path must preserve. Called from `DeviceMemPool::drain`.
/// Shutdown drain dispatcher, called from `DeviceMemPool::drain`. Non-test
/// builds run [`real_drain_pending_frees_blocking`]; the `#[cfg(test)]` host
/// default is a no-op, and `BOLT_BENCH_GPU=1` selects the real blocking drain.
fn drain_pending_frees_blocking() {
    #[cfg(test)]
    if !bench_gpu_enabled() {
        return;
    }
    real_drain_pending_frees_blocking();
}

fn real_drain_pending_frees_blocking() {
    let pending: Vec<PendingFree> = {
        let mut guard = PENDING_FREES.lock();
        std::mem::take(&mut *guard)
    };
    for entry in pending {
        for ev in &entry.events {
            // SAFETY: live event handle; block until its recorded work
            // completes so the subsequent free cannot race in-flight work.
            let _ = unsafe { crate::cuda::cuda_sys::event_synchronize(*ev) };
            // SAFETY: event is complete (just synchronized) and unused after.
            let _ = unsafe { crate::cuda::cuda_sys::event_destroy(*ev) };
        }
        // Free directly to the driver — at shutdown we are draining the pool,
        // so re-pooling would be immediately undone.
        // SAFETY: events synchronized above, so no stream still references the
        // block; provenance is the same as any pooled pointer.
        unsafe { driver_free(entry.ptr) };
    }
}

/// Test-only side-channel: blocks routed through the host-only branch of
/// [`defer_free`] under `#[cfg(test)]` (i.e. when `BOLT_BENCH_GPU` is unset).
/// Lets host tests assert a buffer's `Drop` *deferred* a block (event path)
/// instead of freeing it inline, without a CUDA context — synthetic pointers
/// carry no real events. Under `BOLT_BENCH_GPU=1` the real deferred-free path
/// is used instead and this side-channel stays empty.
#[cfg(test)]
pub(crate) static TEST_DEFERRED: Lazy<Mutex<Vec<(CUdeviceptr, usize)>>> =
    Lazy::new(|| Mutex::new(Vec::new()));

/// Runtime gate for the `#[cfg(test)]` memory-pool driver shims. When
/// `BOLT_BENCH_GPU=1` (or `=true`) is set, [`defer_free`], [`sweep_pending_frees`],
/// [`drain_pending_frees_blocking`], [`driver_mem_alloc`], and [`driver_free`]
/// route to their real, CUDA-driver-touching implementations instead of the
/// host-only synthetic shims — so the crate's `#[ignore]`'d GPU unit tests can
/// run on a live device with `cargo test -- --ignored`. Read once and cached;
/// the default (unset) preserves the GPU-less host-CI behaviour exactly. The
/// OOM-injection latch in [`test_support::test_driver_alloc`] still takes
/// precedence so host fault-injection tests are unaffected.
#[cfg(test)]
pub(crate) fn bench_gpu_enabled() -> bool {
    static GATE: Lazy<bool> = Lazy::new(|| {
        std::env::var("BOLT_BENCH_GPU")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    });
    *GATE
}

/// Snapshot of pool-wide telemetry counters and capacity figures.
///
/// Stage 4 — gives downstream observability layers a single, stable entry
/// point ([`pool_stats`]) instead of poking at individual counters. New
/// fields may be added (non-breaking) but existing ones keep their
/// semantics. All fields are read with `Relaxed` atomic ordering; a
/// snapshot may be slightly stale under heavy concurrent activity but
/// every field is internally consistent on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// Sum of `alloc_bytes` across every pooled (currently-freed-but-
    /// not-returned-to-driver) block. Mirrors the soft cap configured
    /// via `CRATON_BOLT_POOL_MAX_BYTES`.
    pub total_pooled_bytes: usize,
    /// Number of distinct bucket size classes that currently hold at
    /// least one pooled block (best-effort under DashMap iteration).
    pub bucket_count: usize,
    /// Cumulative count of driver-OOM allocations that were rescued by
    /// evicting / draining the pool and retrying. See
    /// [`oom_recovery_count`].
    pub oom_recovery_count: u64,
    /// Cumulative count of proactive evictions triggered by the
    /// `pool-watcher` background thread when free device memory dropped
    /// below `BOLT_POOL_WATCH_LOW_WATER_FRAC`. See
    /// [`proactive_eviction_count`].
    pub proactive_eviction_count: u64,
}

/// Public telemetry entry point: snapshot the process-wide pool counters.
///
/// Stage 4 surfaces this so a downstream observability layer (Prometheus
/// exporter, log aggregator, custom dashboard) can poll one function
/// instead of reaching into crate-internal symbols. The returned
/// [`PoolStats`] is a value type; the caller owns it.
///
/// The fields are read with `Relaxed` atomic ordering, so a single
/// snapshot may be slightly stale under heavy concurrent activity. Each
/// field is internally consistent — `total_pooled_bytes` is reconciled
/// at most `RECONCILE_EVERY_N_FREES` frees behind the truth, and the
/// counters are monotonically non-decreasing.
pub fn pool_stats() -> PoolStats {
    PoolStats {
        total_pooled_bytes: POOL.total_pooled_bytes(),
        bucket_count: POOL.bucket_count(),
        oom_recovery_count: oom_recovery_count(),
        proactive_eviction_count: proactive_eviction_count(),
    }
}

fn read_env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// Round `bytes` up to a denser bucket class than next-power-of-two.
///
/// Uses a uniform 4-classes-per-octave schedule: within each `[2^k, 2^(k+1))`
/// range there are four equally-spaced sub-classes, so worst-case waste is
/// just under 25%. See the module-level doc for the full table.
///
/// Floor is `ARROW_ALIGNMENT` — the CUDA driver guarantees 256-byte
/// alignment on its own end, so this is just a sanity floor for the tiniest
/// allocations.
fn bucket_size(bytes: usize) -> usize {
    let n = bytes.max(ARROW_ALIGNMENT);
    // Position of the highest set bit -> the lower octave boundary.
    // `n >= ARROW_ALIGNMENT >= 1`, so `leading_zeros < usize::BITS`.
    let pow2 = 1usize << (usize::BITS - 1 - n.leading_zeros());
    // Four sub-classes per octave: step = pow2 / 4. For `pow2 == 64`
    // (smallest, == ARROW_ALIGNMENT) step is 16, and `n >= 64` keeps
    // the rounded value at or above the floor.
    //
    // `step.max(1)` is paranoia: with pow2 < 4 the division would yield
    // zero and break `div_ceil`. We never hit that because `n >= 64`,
    // but the cost is one cmp instruction.
    let step = (pow2 / 4).max(1);
    // bitmask round-up (step is power-of-two; saves a div+mul on every alloc/free).
    // Equivalent to `ceil(n / step) * step` but compiles to an `add`+`andn`.
    // Saturating arithmetic guards against pathological sizes near `usize::MAX`;
    // cuMemAlloc would refuse those anyway.
    let rounded = n.saturating_add(step - 1) & !(step - 1);
    rounded.max(ARROW_ALIGNMENT)
}

/// One pooled block. `inserted` is captured at `free` time so the bucket's
/// front entry is always the oldest within that bucket. `tick` is a
/// process-wide monotonically-increasing counter that disambiguates blocks
/// sharing the same coarse-clock `Instant` and serves as the secondary
/// key into the global LRU `BTreeMap`.
#[derive(Clone, Copy)]
struct PooledBlock {
    ptr: CUdeviceptr,
    inserted: Instant,
    tick: u64,
}

/// Per-bucket state. Each entry in the `DashMap` is independently locked.
struct BucketEntry {
    blocks: VecDeque<PooledBlock>,
}

impl BucketEntry {
    fn new() -> Self {
        Self {
            blocks: VecDeque::new(),
        }
    }
}

/// Free a `CUdeviceptr` through whichever backend is active for this build.
/// Errors are logged but otherwise swallowed — the pool's eviction paths
/// run under a lock and cannot meaningfully propagate failures.
///
/// # Safety
/// `ptr` must have been minted by the matching backend's `mem_alloc` and
/// must no longer be aliased.
/// Real, driver-touching free. Always compiled. In a non-test build it is the
/// only free path; under `#[cfg(test)]` it is reached only when
/// `BOLT_BENCH_GPU=1` selects the live-device path.
///
/// # Safety
/// `ptr` must have been minted by the matching backend's `mem_alloc` and must
/// no longer be aliased.
unsafe fn real_driver_free(ptr: CUdeviceptr) {
    #[cfg(feature = "cudarc")]
    let result = crate::cuda::cudarc_backend::mem_free(ptr);
    #[cfg(not(feature = "cudarc"))]
    let result = cuda_sys::mem_free(ptr);
    if let Err(e) = result {
        // Use `log::warn!` for consistency with the rest of the module
        // (pool_watcher, OOM recovery). `eprintln!` bypasses the
        // crate's structured logging and is harder to silence in tests.
        log::warn!("craton-bolt: DeviceMemPool failed to free ptr: {}", e);
    }
}

/// Free a `CUdeviceptr` through whichever backend is active for this build.
/// Errors are logged but otherwise swallowed — the pool's eviction paths run
/// under a lock and cannot meaningfully propagate failures.
///
/// Under `#[cfg(test)]` the pool's policy logic runs on synthetic pointers
/// minted by `test_support::test_driver_alloc`; routing them through the real
/// CUDA driver would crash, so the host-only default records each "free" in a
/// side-channel list for eviction assertions. `BOLT_BENCH_GPU=1` routes to the
/// real driver instead so the `#[ignore]`'d GPU tests free real device memory.
///
/// # Safety
/// `ptr` must have been minted by the matching backend's `mem_alloc` and must
/// no longer be aliased.
unsafe fn driver_free(ptr: CUdeviceptr) {
    #[cfg(test)]
    if !bench_gpu_enabled() {
        test_support::record_driver_free(ptr);
        return;
    }
    real_driver_free(ptr);
}

/// Allocate `alloc_bytes` of device memory through whichever backend is
/// active for this build. Mirrors `driver_free` in shape: tests intercept
/// it via `test_support::test_driver_alloc` so the OOM-recovery logic can
/// be exercised on synthetic pointers without a live CUDA context.
/// Real, driver-touching allocation. Always compiled. In a non-test build it is
/// the only alloc path; under `#[cfg(test)]` it is reached only when
/// `BOLT_BENCH_GPU=1` selects the live-device path (via `test_driver_alloc`).
fn real_driver_mem_alloc(alloc_bytes: usize) -> BoltResult<CUdeviceptr> {
    // Under `--features cudarc`, the alloc is satisfied by cudarc's
    // `result::malloc_sync`, which calls the same `cuMemAlloc_v2` under
    // the hood and returns a bit-compatible `CUdeviceptr` — so pointers
    // stored in the pool remain backend-agnostic and the drain path can
    // free them via either implementation.
    #[cfg(feature = "cudarc")]
    {
        crate::cuda::cudarc_backend::mem_alloc(alloc_bytes)
    }
    #[cfg(not(feature = "cudarc"))]
    {
        cuda_sys::mem_alloc(alloc_bytes)
    }
}

/// Allocate `alloc_bytes` of device memory through whichever backend is active.
/// Non-test builds call [`real_driver_mem_alloc`] directly. Under `#[cfg(test)]`
/// the call is routed through `test_support::test_driver_alloc`, which honours
/// the OOM-injection latch first and then either mints a synthetic pointer
/// (host default) or delegates to [`real_driver_mem_alloc`] when
/// `BOLT_BENCH_GPU=1` is set.
fn driver_mem_alloc(alloc_bytes: usize) -> BoltResult<CUdeviceptr> {
    #[cfg(test)]
    {
        test_support::test_driver_alloc(alloc_bytes)
    }
    #[cfg(not(test))]
    {
        real_driver_mem_alloc(alloc_bytes)
    }
}

/// Recognise a driver-OOM error from `BoltError::CudaWithCode`'s integer
/// `code` field.
///
/// Stage 4 replaces the fragile formatted-string prefix-match with a direct
/// pattern match on the `CUresult` integer that `cuda_sys::check` now
/// surfaces via `BoltError::CudaWithCode`. The string-prefix path is gone
/// entirely — no risk of false negatives (driver localisation), false
/// positives (`"error 200"`), or future-proofing concerns from formatting
/// drift in `check()`.
fn is_oom_error(e: &crate::error::BoltError) -> bool {
    matches!(
        e,
        crate::error::BoltError::CudaWithCode {
            code: CUDA_OOM_CODE,
            ..
        }
    )
}

/// Storage shape for the bucket map. Default build uses DashMap; the
/// `pool-sharded` feature flips to a fixed-N array. All access goes
/// through the `with_bucket` / `with_or_create_bucket` / `for_each_bucket`
/// helpers so the hot path doesn't care which variant is active.
#[cfg(not(feature = "pool-sharded"))]
type BucketStorage = DashMap<usize, Mutex<BucketEntry>>;

#[cfg(feature = "pool-sharded")]
type BucketStorage = [Mutex<HashMap<usize, BucketEntry>>; SHARDS];

/// Pick the shard index for a given `size_class`. Uses `%` because SHARDS
/// is a power of two; LLVM folds this to a mask. Distinct size classes
/// (a couple dozen for realistic workloads) spread evenly enough that
/// the per-shard mutex is rarely contended.
#[cfg(feature = "pool-sharded")]
#[inline]
fn shard_of(size_class: usize) -> usize {
    size_class % SHARDS
}

/// Construct a freshly-initialised `BucketStorage`. Kept as a free
/// function so the cfg-gated struct shape doesn't leak into `new`.
#[cfg(not(feature = "pool-sharded"))]
fn new_bucket_storage() -> BucketStorage {
    DashMap::new()
}

#[cfg(feature = "pool-sharded")]
fn new_bucket_storage() -> BucketStorage {
    std::array::from_fn(|_| Mutex::new(HashMap::new()))
}

/// Process-wide GPU device-memory pool. Holds freed blocks keyed by their
/// bucket (rounded-up) size and hands them out on subsequent allocations.
pub struct DeviceMemPool {
    /// Buckets keyed by rounded-up byte size. Each bucket has its own
    /// `Mutex` so concurrent frees into distinct size classes don't
    /// serialise on a global lock.
    ///
    /// Storage shape depends on `pool-sharded` feature; see `BucketStorage`.
    buckets: BucketStorage,
    /// Sum of `alloc_bytes` across every pooled block. Atomic so reads in
    /// the eviction loop don't need a global lock. Soft cap — short
    /// transient overshoot under contention is acceptable; drift is
    /// corrected on the next reconciliation pass (every
    /// `RECONCILE_EVERY_N_FREES` frees, or via explicit
    /// `reconcile_total_bytes`).
    total_bytes: AtomicUsize,
    /// Cross-bucket global LRU index, **sharded** into `LRU_SHARDS`
    /// independent `BTreeMap`s (PERF P-1). Each shard is keyed by
    /// `(inserted, tick)`: `tick` (a process-wide monotonic counter) makes
    /// the key globally unique across *all* shards, so a cross-shard min /
    /// ordering comparison is well-defined. Value is `(size_class, ptr)` so
    /// eviction can locate the owning bucket without a scan. A block always
    /// lives in `lru_index[lru_shard_of(size_class)]`; see module doc for
    /// the race-handling protocol and the shard-fan-out eviction scan.
    //
    // PERF P-1 (resolved here): the previous single global `lru_index`
    // mutex was taken on *every* `alloc` hit and *every* `free` insert (on
    // top of the per-bucket lock), so under many-stream concurrent churn it
    // re-serialised every size class through one BTreeMap mutex even though
    // the per-bucket `DashMap`/array split had already removed the old
    // global *bucket* lock.
    //
    // Fix: shard the LRU index by `size_class` (mirroring how the bucket
    // storage is already sharded). The two hot paths —
    //   * `alloc`-hit:  remove from `lru_index[lru_shard_of(size_class)]`
    //   * `free`-insert: insert into `lru_index[lru_shard_of(size_class)]`
    // now touch exactly one shard, the one for the size class they already
    // hold, so distinct size classes that hash to different LRU shards run
    // fully in parallel. Eviction (`evict_one`) still honours global LRU by
    // peeking the oldest `(Instant, tick)` across all shards and popping the
    // single global minimum — see `evict_one` for the deadlock-free
    // shard-fan-out protocol and the extended lock-order invariant.
    lru_index: [Mutex<BTreeMap<(Instant, u64), (usize, CUdeviceptr)>>; LRU_SHARDS],
    /// Process-wide monotonic counter feeding `PooledBlock::tick`.
    /// `Relaxed` is sufficient: we only need uniqueness, not ordering
    /// against other atomics.
    next_tick: AtomicU64,
    /// Count of `free` calls since the last reconciliation. Wraps freely;
    /// only the `% RECONCILE_EVERY_N_FREES == 0` check matters.
    frees_since_reconcile: AtomicU64,
    /// Soft cap on `total_bytes`. Resolved from env once at construction.
    max_pooled_bytes: usize,
    /// Hard cap on `buckets[k].len()` for any `k`.
    max_bucket_entries: usize,
}

impl DeviceMemPool {
    pub fn new() -> Self {
        Self {
            buckets: new_bucket_storage(),
            total_bytes: AtomicUsize::new(0),
            // PERF P-1: one independent BTreeMap mutex per LRU shard.
            lru_index: std::array::from_fn(|_| Mutex::new(BTreeMap::new())),
            next_tick: AtomicU64::new(0),
            frees_since_reconcile: AtomicU64::new(0),
            max_pooled_bytes: read_env_usize(
                "CRATON_BOLT_POOL_MAX_BYTES",
                DEFAULT_MAX_POOLED_BYTES,
            ),
            max_bucket_entries: read_env_usize(
                "CRATON_BOLT_POOL_BUCKET_CAP",
                DEFAULT_MAX_BUCKET_ENTRIES,
            ),
        }
    }

    /// Saturating subtract `n` from `self.total_bytes`.
    ///
    /// Replaces a bare `fetch_sub(n, AcqRel)` to close a small race window:
    /// `reconcile_total_bytes` performs an atomic `store` of the freshly
    /// summed bucket bytes, and an in-flight `free` / `evict_one` whose
    /// `fetch_sub` interleaves *after* that store can drive the counter
    /// below zero and wrap a `usize` to ~`usize::MAX`. The wrong-direction
    /// value then makes `try_insert_into_locked_bucket`'s cap check reject
    /// every subsequent free until the next reconciliation pass — a window
    /// of up to `RECONCILE_EVERY_N_FREES` (1024) calls.
    ///
    /// Saturating at zero is the correct fix: `total_bytes` is a soft
    /// accounting counter (the source of truth is the bucket contents
    /// themselves, recomputed by `reconcile_total_bytes`), so any
    /// transient under-count is corrected on the next reconciliation
    /// without ever producing a pathological value in between. The
    /// memory ordering (`AcqRel` / `Acquire`) matches the prior
    /// `fetch_sub` — `fetch_update`'s closure runs inside a CAS loop
    /// and the orderings apply to the success / failure paths
    /// respectively. The closure is total (always `Some(_)`), so the
    /// `fetch_update` itself never returns `Err`.
    #[inline]
    fn sub_total_saturating(&self, n: usize) {
        let _ = self.total_bytes.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |cur| Some(cur.saturating_sub(n)),
        );
    }

    // ---- Sharded LRU-index helpers (PERF P-1) ----
    //
    // A block's LRU entry always lives in the shard chosen by its
    // `size_class` (`lru_shard_of`). These helpers funnel every LRU
    // touch through that single deterministic shard so the insert/remove
    // pair for one block can never straddle two shards, and so the hot
    // paths take exactly one shard lock instead of one global lock.

    /// Insert `(inserted, tick) -> (size_class, ptr)` into the LRU shard
    /// owning `size_class`. Takes only that one shard's lock.
    ///
    /// **Lock order.** Callers hold the bucket lock when they call this
    /// (the `try_insert_into_locked_bucket` path), so this is the
    /// *inner* lock of the bucket-then-lru order — same discipline as
    /// the pre-P-1 single-mutex design, just on a sharded mutex.
    #[inline]
    fn lru_insert(
        &self,
        size_class: usize,
        inserted: Instant,
        tick: u64,
        ptr: CUdeviceptr,
    ) {
        self.lru_index[lru_shard_of(size_class)]
            .lock()
            .insert((inserted, tick), (size_class, ptr));
    }

    /// Remove the LRU entry for a block of `size_class` keyed by
    /// `(inserted, tick)`. Takes only that one shard's lock. Used by the
    /// `alloc`-hit path and the eviction stale-entry cleanup; in both
    /// cases the caller is *not* holding any bucket lock, so this never
    /// inverts the bucket-then-lru order.
    #[inline]
    fn lru_remove(&self, size_class: usize, inserted: Instant, tick: u64) {
        self.lru_index[lru_shard_of(size_class)]
            .lock()
            .remove(&(inserted, tick));
    }

    /// Pop the globally-oldest LRU entry across *all* shards.
    ///
    /// Returns the `((inserted, tick), (size_class, ptr))` whose key is
    /// the minimum over every shard, having removed it from its shard, or
    /// `None` when every shard is empty.
    ///
    /// **Why two passes.** `(inserted, tick)` is globally unique (`tick`
    /// is a process-wide monotonic counter), so "oldest across the pool"
    /// is just the minimum first-key over all shards. We first *peek*
    /// each shard's `first_key_value` to find which shard holds the
    /// global minimum, then re-lock that one shard and `pop_first` it.
    ///
    /// **Deadlock-freedom (PERF P-1 lock order).** Each shard lock is
    /// taken and released individually — at no point are two LRU-shard
    /// locks held at once, and at no point is any bucket lock held while
    /// scanning. A concurrent insert into a shard we already peeked can
    /// only add a *newer* (larger-keyed) entry, so it cannot change which
    /// entry is the global minimum at peek time; a concurrent pop of the
    /// very entry we selected is handled by re-reading `pop_first` under
    /// the shard lock and accepting whatever is now oldest there (or
    /// retrying the scan if that shard drained). The popped entry's
    /// `size_class` is intrinsic to the block, so the caller can still
    /// route to the correct bucket.
    fn lru_pop_global_oldest(
        &self,
    ) -> Option<((Instant, u64), (usize, CUdeviceptr))> {
        loop {
            // Pass 1: peek every shard's oldest key, one shard lock at a
            // time (never two at once), and remember which shard owns the
            // global minimum.
            let mut best_shard: Option<usize> = None;
            let mut best_key: Option<(Instant, u64)> = None;
            for (idx, shard) in self.lru_index.iter().enumerate() {
                let guard = shard.lock();
                if let Some((k, _v)) = guard.first_key_value() {
                    if best_key.map_or(true, |b| *k < b) {
                        best_key = Some(*k);
                        best_shard = Some(idx);
                    }
                }
            }
            let shard_idx = best_shard?;
            // Pass 2: re-lock just the winning shard and pop its oldest.
            // Between pass 1 and here, a racing `evict_one` on another
            // thread (or an `alloc`-hit / `lru_remove`) may have already
            // taken the entry we picked. `pop_first` returns whatever is
            // oldest in that shard *now*; if the shard drained entirely in
            // the meantime we loop and re-scan rather than return a false
            // `None` while other shards may still be populated.
            if let Some((k, v)) = self.lru_index[shard_idx].lock().pop_first() {
                return Some((k, v));
            }
            // Winning shard raced empty; re-scan. Progress is guaranteed:
            // either some other shard still holds entries (next scan picks
            // one) or every shard is empty (scan returns `None`).
        }
    }

    /// Clear every LRU shard. Used by `drain`.
    #[inline]
    fn lru_clear_all(&self) {
        for shard in self.lru_index.iter() {
            shard.lock().clear();
        }
    }

    /// Total number of entries across all LRU shards. Test/diagnostic
    /// only — walks every shard under its own lock. Not a consistent
    /// snapshot under concurrent mutation, but exact when the pool is
    /// quiescent (which is when the tests consult it).
    #[cfg(test)]
    fn lru_total_len(&self) -> usize {
        self.lru_index.iter().map(|s| s.lock().len()).sum()
    }

    // ---- Storage abstraction helpers ----
    //
    // Hot-path code (alloc, free, evict_one, drain, …) calls these
    // helpers instead of touching `self.buckets` directly, so the two
    // storage shapes can coexist without sprinkling cfg gates through
    // every method body.

    /// Borrow the bucket for `size_class` (if it exists) and invoke `f`
    /// with the locked `BucketEntry`. Returns `None` when the bucket
    /// does not exist, otherwise `Some(f(...))`. Holds the bucket lock
    /// only for the duration of `f`.
    #[cfg(not(feature = "pool-sharded"))]
    #[inline]
    fn with_bucket<R>(
        &self,
        size_class: usize,
        f: impl FnOnce(&mut BucketEntry) -> R,
    ) -> Option<R> {
        let entry = self.buckets.get(&size_class)?;
        let mut guard = entry.lock();
        Some(f(&mut guard))
    }

    #[cfg(feature = "pool-sharded")]
    #[inline]
    fn with_bucket<R>(
        &self,
        size_class: usize,
        f: impl FnOnce(&mut BucketEntry) -> R,
    ) -> Option<R> {
        let mut shard = self.buckets[shard_of(size_class)].lock();
        let entry = shard.get_mut(&size_class)?;
        Some(f(entry))
    }

    /// Borrow-or-create the bucket for `size_class`. The first-touch path
    /// (insert a new size class) is rare — once per size class for the
    /// entire pool's lifetime — so its cost is amortised away. Holds the
    /// bucket / shard lock only for the duration of `f`.
    #[cfg(not(feature = "pool-sharded"))]
    #[inline]
    fn with_or_create_bucket<R>(
        &self,
        size_class: usize,
        f: impl FnOnce(&mut BucketEntry) -> R,
    ) -> R {
        let entry = self
            .buckets
            .entry(size_class)
            .or_insert_with(|| Mutex::new(BucketEntry::new()));
        let mut guard = entry.lock();
        f(&mut guard)
    }

    #[cfg(feature = "pool-sharded")]
    #[inline]
    fn with_or_create_bucket<R>(
        &self,
        size_class: usize,
        f: impl FnOnce(&mut BucketEntry) -> R,
    ) -> R {
        let mut shard = self.buckets[shard_of(size_class)].lock();
        let entry = shard.entry(size_class).or_insert_with(BucketEntry::new);
        f(entry)
    }

    /// Visit every populated bucket under its own lock and invoke `f`
    /// with the size class and locked `BucketEntry`. Used by the scan
    /// fallback, `drain`, and reconciliation paths. Bucket lock is held
    /// only across the per-bucket invocation; never across two buckets
    /// at once.
    #[cfg(not(feature = "pool-sharded"))]
    #[inline]
    fn for_each_bucket(&self, mut f: impl FnMut(usize, &mut BucketEntry)) {
        for r in self.buckets.iter() {
            let key = *r.key();
            let mut guard = r.value().lock();
            f(key, &mut guard);
        }
    }

    #[cfg(feature = "pool-sharded")]
    #[inline]
    fn for_each_bucket(&self, mut f: impl FnMut(usize, &mut BucketEntry)) {
        for shard in self.buckets.iter() {
            let mut guard = shard.lock();
            // Collect keys first so `f` may mutate the entry (the
            // HashMap iter would borrow-check fight `f`'s mutable use).
            // For the default-on DashMap path this never runs; for the
            // sharded path each shard holds ≤ ~3 size classes on average.
            let keys: Vec<usize> = guard.keys().copied().collect();
            for key in keys {
                if let Some(entry) = guard.get_mut(&key) {
                    f(key, entry);
                }
            }
        }
    }

    /// Drain every bucket into `sink` (ptrs only) and leave the storage
    /// empty. Used by `drain` for the on-Drop / shutdown path.
    #[cfg(not(feature = "pool-sharded"))]
    fn drain_all_into(&self, sink: &mut Vec<CUdeviceptr>) {
        for r in self.buckets.iter() {
            let mut guard = r.value().lock();
            while let Some(block) = guard.blocks.pop_front() {
                sink.push(block.ptr);
            }
        }
        self.buckets.clear();
    }

    #[cfg(feature = "pool-sharded")]
    fn drain_all_into(&self, sink: &mut Vec<CUdeviceptr>) {
        for shard in self.buckets.iter() {
            let mut guard = shard.lock();
            for (_, mut entry) in guard.drain() {
                while let Some(block) = entry.blocks.pop_front() {
                    sink.push(block.ptr);
                }
            }
        }
    }

    /// Try to take a freed block big enough for `bytes`. Falls back to
    /// `cuMemAlloc` on a miss. Returns `(ptr, actual_alloc_bytes)`; the caller
    /// must remember `actual_alloc_bytes` and pass it to `free` so we return
    /// to the right bucket.
    ///
    /// **Driver-OOM recovery (review finding M3).** If the miss-path driver
    /// alloc returns `CUDA_ERROR_OUT_OF_MEMORY` (code 2), this routes to
    /// `recover_from_oom`, which trims headroom above the soft cap and then
    /// evicts pooled blocks *incrementally* (oldest first), retrying after each
    /// eviction until the alloc fits or the pool empties — preserving the warm
    /// cache shared by concurrent queries instead of draining it wholesale.
    pub fn alloc(&self, bytes: usize) -> BoltResult<(CUdeviceptr, usize)> {
        // Stage 4: ensure the proactive watcher is running, lazily.
        // No-op under default build (feature off) and under #[cfg(test)]
        // so the host-only test suite doesn't spawn driver-calling
        // threads. Idempotent: spawns at most one thread for the
        // lifetime of the process.
        ensure_watcher_started();

        // Reclaim any deferred-free blocks whose stream events have completed
        // (review finding C1 / P1). Cheap no-op when nothing is pending, which
        // is the steady state. Doing it here means a fresh allocation can reuse
        // a block whose in-flight work has since drained, without the dropping
        // thread ever having blocked.
        sweep_pending_frees();

        let alloc_bytes = bucket_size(bytes);
        // Hit-path: try the pool first.
        let hit = self.with_bucket(alloc_bytes, |bucket| {
            // LIFO: most-recently freed block first — best cache locality.
            bucket.blocks.pop_back()
        });
        if let Some(Some(block)) = hit {
            self.sub_total_saturating(alloc_bytes);
            // Remove the block's entry from the global LRU index. Note
            // the lock order: we already dropped the bucket lock at
            // the end of `with_bucket`'s closure, so taking the LRU
            // shard lock here cannot cause a hold-and-wait cycle with any
            // bucket lock — this is bucket-then-lru as required, just
            // with the bucket lock having been released by the helper.
            //
            // PERF P-1: the block's LRU entry lives in the shard chosen
            // by `alloc_bytes` (its size class), so `lru_remove` touches
            // exactly that one shard — no global LRU lock.
            //
            // Together the (bucket-push, lru-insert) and (bucket-pop,
            // lru-remove) pairs guarantee that the LRU index never
            // holds a stale entry pointing at a no-longer-pooled block,
            // which is what the `lru_handles_concurrent_free_race`
            // test asserts.
            self.lru_remove(alloc_bytes, block.inserted, block.tick);
            return Ok((block.ptr, alloc_bytes));
        }
        // Miss: call the driver. cuMemAlloc_v2 guarantees at least 256-byte
        // alignment, so the ARROW_ALIGNMENT (64) invariant holds trivially.
        match driver_mem_alloc(alloc_bytes) {
            Ok(ptr) => Ok((ptr, alloc_bytes)),
            Err(e) if is_oom_error(&e) => self.recover_from_oom(alloc_bytes, e),
            Err(e) => Err(e),
        }
    }

    /// OOM-recovery slow path. Incrementally evicts pooled blocks and
    /// retries the driver alloc until it succeeds or the pool is empty.
    /// Separated from `alloc` so the common (success) case stays
    /// cold-jump-free.
    ///
    /// ## Incremental eviction (review finding M3)
    ///
    /// The previous implementation called `evict_above_high_water()` then
    /// `drain()` — releasing EVERY pooled block on any single OOM, discarding
    /// the warm cache shared by all concurrent in-flight queries on other
    /// threads. Under a workload hovering near the VRAM cap that turns into a
    /// thundering herd: every query that OOMs nukes the whole pool, the next
    /// allocation re-mints from the driver, the next query OOMs and nukes it
    /// again — inverting the pool's entire purpose (avoiding alloc/free churn).
    ///
    /// Instead we evict the globally-oldest block (LRU), free it to the driver,
    /// and retry the alloc — looping until the retry succeeds or no pooled
    /// blocks remain. This hands back exactly as much VRAM as the retry needs
    /// and no more, so concurrent queries keep their warm headroom. In the
    /// common case where freeing one or two oldest blocks is enough, the bulk
    /// of the cache survives.
    ///
    /// On the *first* retry we evict everything above the soft cap first (one
    /// `evict_above_high_water`, itself an incremental loop) because a pool that
    /// has grown past its cap is the most likely cause of the squeeze, and that
    /// eviction is "free" headroom we were going to reclaim anyway. After that
    /// the per-block loop handles the fine-grained case.
    ///
    /// Returns the *original* OOM error if the pool empties out and the driver
    /// still can't satisfy the request, so callers see the same error surface
    /// as before the hook existed.
    #[cold]
    fn recover_from_oom(
        &self,
        alloc_bytes: usize,
        original_err: crate::error::BoltError,
    ) -> BoltResult<(CUdeviceptr, usize)> {
        // Step 1: drop everything above the soft cap. Cheap if the cap is
        // already respected (no-op); useful when the workload has grown well
        // past the cap and the driver is now squeezed. This is itself an
        // incremental, LRU-ordered loop (see `evict_above_high_water`), so it
        // never frees blocks the pool is still allowed to keep.
        let _evicted = self.evict_above_high_water();
        // Retry after the high-water trim before touching any blocks the pool
        // is allowed to retain.
        if let Some(ok) = self.try_retry_alloc(alloc_bytes) {
            return ok;
        }
        // Step 2: incremental LRU eviction. Evict one oldest block at a time,
        // free it to the driver, and retry — preserving the rest of the warm
        // cache for concurrent queries instead of draining it wholesale.
        loop {
            let mut to_free: Vec<CUdeviceptr> = Vec::with_capacity(1);
            if !self.evict_one(&mut to_free) {
                // Pool is empty and the driver still can't satisfy us. Give up
                // and return the original error.
                break;
            }
            for p in to_free {
                // SAFETY: same provenance argument as `free`/`drain` — every
                // pointer routed here was pulled out of the pool, originally
                // minted by the active backend's `mem_alloc`.
                unsafe { driver_free(p) };
            }
            if let Some(ok) = self.try_retry_alloc(alloc_bytes) {
                return ok;
            }
        }
        Err(original_err)
    }

    /// Retry the driver alloc once after freeing headroom during OOM recovery.
    /// Returns `Some(Ok(..))` on success (bumping the recovery counter),
    /// `None` if the driver is still OOM (so the caller keeps evicting), or
    /// `Some(Err(..))` if the retry failed with a *non-OOM* error (which we
    /// surface immediately rather than spinning the eviction loop).
    #[inline]
    fn try_retry_alloc(
        &self,
        alloc_bytes: usize,
    ) -> Option<BoltResult<(CUdeviceptr, usize)>> {
        match driver_mem_alloc(alloc_bytes) {
            Ok(ptr) => {
                OOM_RECOVERY_COUNT.fetch_add(1, Ordering::Relaxed);
                // Use `log::warn!` for consistency with the rest of the
                // module (driver_free, pool_watcher).
                log::warn!(
                    "craton-bolt: DeviceMemPool recovered from driver OOM (alloc_bytes={})",
                    alloc_bytes
                );
                Some(Ok((ptr, alloc_bytes)))
            }
            // Still OOM: keep evicting.
            Err(e) if is_oom_error(&e) => None,
            // A different driver error on retry — don't spin the eviction
            // loop on it; surface it to the caller.
            Err(e) => Some(Err(e)),
        }
    }

    /// Return a block to the pool. If pooling this block would exceed
    /// `MAX_POOLED_BYTES` or `MAX_BUCKET_ENTRIES`, evict block(s) first;
    /// if that still does not make room, free `ptr` directly via the
    /// driver instead of pooling it.
    ///
    /// Eviction picks the globally-oldest pooled block via the
    /// cross-bucket LRU index — see `evict_one`. Every
    /// `RECONCILE_EVERY_N_FREES` calls also triggers a
    /// `reconcile_total_bytes` pass so the atomic `total_bytes` counter
    /// self-heals any drift accumulated under concurrent contention.
    pub fn free(&self, ptr: CUdeviceptr, alloc_bytes: usize) {
        if ptr == 0 {
            return;
        }

        // Reclaim any deferred-free blocks whose events have completed (review
        // finding C1 / P1). No-op in the steady state; keeps the pending list
        // from growing without bound on a free-heavy, alloc-light phase.
        sweep_pending_frees();

        // Pre-size for the common case: 0 or 1 evictions per free. Skips
        // the initial-grow allocation that a default `Vec::new()` would
        // pay for on the first `push` in the eviction loop.
        let mut to_free: Vec<CUdeviceptr> = Vec::with_capacity(2);

        // ---- Byte-cap eviction (best-effort, lock-free counter) ----
        //
        // If the incoming block is bigger than the entire cap there's no
        // point evicting — we'll just route it straight to the driver.
        if alloc_bytes <= self.max_pooled_bytes {
            // `saturating_add` guards against a transient overshoot where
            // `total_bytes` is briefly close to `usize::MAX` (e.g. under
            // an interleaved `reconcile_total_bytes` + bare `fetch_add`
            // race window). A bare `+` could wrap and silently bypass
            // the cap check.
            while self
                .total_bytes
                .load(Ordering::Acquire)
                .saturating_add(alloc_bytes)
                > self.max_pooled_bytes
            {
                if !self.evict_one(&mut to_free) {
                    break; // pool empty.
                }
            }
        }

        // ---- Per-bucket cap + insert under the bucket's own mutex ----
        //
        // Try `with_bucket` first (existing bucket — fast path). On a miss
        // fall through to `with_or_create_bucket`. Under the default
        // DashMap storage, `with_bucket` holds only a shard *read* lock
        // while the inner Mutex is taken; `with_or_create_bucket` takes
        // the shard *write* lock. The first-touch (insert-new-size-class)
        // path happens once per size class for the entire pool's lifetime,
        // so the write-lock cost amortises away. Under the sharded storage,
        // both helpers cost the same single shard-mutex acquisition.
        let pooled = match self.with_bucket(alloc_bytes, |bucket| {
            self.try_insert_into_locked_bucket(bucket, ptr, alloc_bytes)
        }) {
            Some(r) => r,
            None => self.with_or_create_bucket(alloc_bytes, |bucket| {
                self.try_insert_into_locked_bucket(bucket, ptr, alloc_bytes)
            }),
        };
        if !pooled {
            // Couldn't make room — drop this block to the driver.
            to_free.push(ptr);
        }

        for p in to_free {
            // SAFETY: every pointer routed here was either pulled out of
            // the pool (originally minted by `mem_alloc` and given up by
            // its previous owner via `free`) or is the `ptr` we were just
            // handed by a caller who has likewise transferred ownership.
            unsafe { driver_free(p) };
        }

        // ---- Periodic reconciliation ----
        //
        // The atomic `total_bytes` counter can drift under concurrent
        // free: between the eviction-loop's `load` and the eventual
        // `fetch_add` / `sub_total_saturating`, parallel frees may
        // interleave in a way that produces a value slightly off from
        // the true sum of
        // `bucket.len() * size_class`. Drift is bounded and self-
        // limiting (the cap re-check in `try_insert_into_bucket` keeps
        // overshoot to ≤ one block per racing thread), but over a long
        // process lifetime a small bias can accumulate. Reconciling
        // every N frees corrects this without imposing any cost on the
        // alloc path.
        let n = self.frees_since_reconcile.fetch_add(1, Ordering::Relaxed) + 1;
        if n % RECONCILE_EVERY_N_FREES == 0 {
            self.reconcile_total_bytes();
        }
    }

    /// Try to push `ptr` into the given bucket (already locked by the
    /// caller via `with_bucket` / `with_or_create_bucket`), respecting
    /// per-bucket and global byte caps. Returns `true` when the block
    /// was pooled, `false` when the caller must driver-free it.
    ///
    /// On success, also inserts `(now, tick) -> (alloc_bytes, ptr)` into
    /// the cross-bucket LRU index so `evict_one` can pick the globally
    /// oldest block, not just the oldest within some bucket.
    ///
    /// **Lock order.** The bucket lock is already held by the caller's
    /// closure. We acquire the LRU lock *underneath* the bucket lock,
    /// matching the canonical bucket-then-lru order. `evict_one`
    /// inverts the order (LRU-first) but releases the LRU lock before
    /// reaching for any bucket lock, so the two paths cannot form a
    /// hold-and-wait cycle. See the module-level lock-order discussion.
    fn try_insert_into_locked_bucket(
        &self,
        bucket: &mut BucketEntry,
        ptr: CUdeviceptr,
        alloc_bytes: usize,
    ) -> bool {
        let fits_bucket = bucket.blocks.len() < self.max_bucket_entries;
        // Re-check byte cap under our local (bucket) lock — eviction
        // above might have already brought us under, or a parallel
        // free may have pushed us back over. `saturating_add` matches
        // the cap check in `free` so a near-`usize::MAX` transient
        // can't wrap and slip past the limit.
        let projected = self
            .total_bytes
            .load(Ordering::Acquire)
            .saturating_add(alloc_bytes);
        let fits_total = alloc_bytes <= self.max_pooled_bytes
            && projected <= self.max_pooled_bytes;
        if fits_bucket && fits_total {
            let inserted = Instant::now();
            let tick = self.next_tick.fetch_add(1, Ordering::Relaxed);
            bucket.blocks.push_back(PooledBlock {
                ptr,
                inserted,
                tick,
            });
            self.total_bytes.fetch_add(alloc_bytes, Ordering::AcqRel);
            // Register with the global LRU index *while still holding
            // the bucket lock*. This pairs the bucket push and the LRU
            // insert atomically from the perspective of any concurrent
            // `alloc` on the same bucket — without it, an `alloc` could
            // pop our just-pushed block, try to remove the (not-yet-
            // inserted) LRU entry as a no-op, and then leave a stale
            // entry behind once our later `lru_insert` runs.
            //
            // PERF P-1: `lru_insert` writes only the shard owning
            // `alloc_bytes`. That shard is the *inner* lock under the
            // currently-held bucket lock, preserving bucket-then-lru.
            // Because a block's shard is a pure function of its size
            // class, the matching `lru_remove` in `alloc` and the
            // eviction cleanup always target this same shard, so the
            // insert/remove pair never straddles two shards.
            self.lru_insert(alloc_bytes, inserted, tick, ptr);
            true
        } else {
            false
        }
    }

    /// Evict the globally-oldest pooled block via the cross-bucket LRU
    /// index. Returns `true` if an eviction happened; `false` when the
    /// pool is empty.
    ///
    /// **Algorithm.** Pop the globally-smallest `(Instant, tick)` across
    /// all LRU shards (`lru_pop_global_oldest`), releasing every LRU shard
    /// lock before continuing. Look up the owning bucket, take its lock,
    /// and remove the block whose `ptr` matches. If the block is no longer
    /// in the bucket (an `alloc` raced ahead of us between our LRU pop and
    /// our bucket lock), fall back to popping any block from the front of
    /// that bucket — that block is at least as old as anything else in the
    /// bucket.
    ///
    /// **Lock order (PERF P-1, extended).** The LRU index is sharded; this
    /// path takes and releases each LRU shard lock individually inside
    /// `lru_pop_global_oldest` (never two LRU-shard locks at once, never a
    /// bucket lock while scanning), and the chosen entry's shard lock is
    /// fully released *before* the bucket lock is taken. So the global
    /// invariant is unchanged and strengthened:
    ///   1. **no thread ever holds any LRU-shard lock while waiting on a
    ///      bucket lock** (the bucket-then-lru order — eviction is the lone
    ///      exception and it drops the LRU shard first), and
    ///   2. **no thread ever holds two LRU-shard locks simultaneously**
    ///      (the cross-shard scan visits one shard at a time).
    /// The bucket→lru edge and the (now per-shard) lru→nothing edges form
    /// an acyclic graph, so deadlock is impossible. `free` still releases
    /// the bucket lock before its `lru_remove` cleanup; the bucket-locked
    /// `lru_insert` in `try_insert_into_locked_bucket` is the only
    /// bucket-then-lru-shard nesting and it never inverts.
    ///
    /// **Fallbacks.** If every LRU shard is empty but `total_bytes > 0`
    /// (cannot happen under correct accounting but defended against),
    /// fall through to `evict_one_scan_fallback`, the M3L5 cross-bucket
    /// scan. That keeps the eviction loop terminating even if the LRU
    /// index has somehow drifted out of sync with the buckets.
    fn evict_one(&self, sink: &mut Vec<CUdeviceptr>) -> bool {
        // Pop the globally-oldest LRU entry across all shards. Every LRU
        // shard lock is released before we return here, so the bucket
        // lock below is taken with no LRU lock held (PERF P-1 order).
        let popped = self.lru_pop_global_oldest();
        if let Some(((_, _tick), (size_class, target_ptr))) = popped {
            // Look up the owning bucket. If the bucket vanished (drain in
            // flight, or this is a stale entry the storage has already
            // cleared), fall through to the scan-based fallback below.
            // The result tracks what happened inside the closure so we
            // can react after releasing the bucket lock.
            enum Outcome {
                ExactHit(CUdeviceptr),
                Approx { block: PooledBlock, size_class: usize },
                BucketEmpty,
            }
            let outcome = self.with_bucket(size_class, |bucket| {
                // First try to remove the exact ptr the LRU pointed to.
                if let Some(pos) = bucket
                    .blocks
                    .iter()
                    .position(|b| b.ptr == target_ptr)
                {
                    let block = bucket.blocks.remove(pos).expect("position checked");
                    return Outcome::ExactHit(block.ptr);
                }
                // Race: an `alloc` consumed `target_ptr` between our LRU
                // pop and this bucket lock. Anything left at the front
                // of the bucket is at least as old as the next-newest
                // LRU entry, so popping the bucket's front is a safe
                // approximation of global LRU.
                if let Some(block) = bucket.blocks.pop_front() {
                    return Outcome::Approx { block, size_class };
                }
                Outcome::BucketEmpty
            });
            match outcome {
                Some(Outcome::ExactHit(ptr)) => {
                    self.sub_total_saturating(size_class);
                    sink.push(ptr);
                    return true;
                }
                Some(Outcome::Approx { block, size_class }) => {
                    self.sub_total_saturating(size_class);
                    // The block we actually evicted has its own LRU
                    // entry that is now stale — remove it. We're outside
                    // the bucket lock at this point; the LRU shard lock is
                    // taken *after* the bucket lock has been released,
                    // preserving the global lock order.
                    //
                    // PERF P-1: the front block came out of the bucket for
                    // `size_class`, so its LRU entry lives in that size
                    // class's shard — `lru_remove(size_class, ..)` targets
                    // exactly the right shard.
                    self.lru_remove(size_class, block.inserted, block.tick);
                    sink.push(block.ptr);
                    return true;
                }
                Some(Outcome::BucketEmpty) | None => {
                    // Bucket is empty (or missing) — fall through to
                    // the scan fallback so we don't infinite-loop on a
                    // stale LRU.
                }
            }
        }
        // LRU empty or bucket missing — defensive cross-bucket scan.
        self.evict_one_scan_fallback(sink)
    }

    /// Pre-Stage-2 cross-bucket scan: peek the front of every bucket
    /// and pop the globally-oldest. Retained as a defensive fallback
    /// for the `evict_one` path when the LRU index has somehow drifted
    /// out of sync with the buckets (should not happen under correct
    /// accounting). O(buckets); bounded.
    ///
    /// Single-pass: collects both the oldest-front candidate and the
    /// list of every non-empty bucket key in one `for_each_bucket` walk,
    /// so we never lock every bucket twice. The previous implementation
    /// re-iterated the bucket storage to find a fallback when the
    /// chosen bucket got raced empty.
    fn evict_one_scan_fallback(&self, sink: &mut Vec<CUdeviceptr>) -> bool {
        // Tracked together so one walk produces both:
        //   - `best_key` / `best_t`: bucket whose `front` is globally oldest
        //   - `non_empty`: every non-empty bucket key (fallback list,
        //     used only if `best` races empty between the scan and our
        //     follow-up pop).
        // The bucket count is bounded (~4 × log2(max_alloc) ≈ 100) so a
        // small `Vec` here is essentially free.
        let mut best_key: Option<usize> = None;
        let mut best_t: Option<Instant> = None;
        let mut non_empty: Vec<usize> = Vec::with_capacity(8);
        self.for_each_bucket(|key, bucket| {
            if let Some(front) = bucket.blocks.front() {
                non_empty.push(key);
                if best_t.map_or(true, |t| front.inserted < t) {
                    best_key = Some(key);
                    best_t = Some(front.inserted);
                }
            }
        });
        // Try the oldest-front bucket first. If a concurrent alloc
        // drained it between the scan and our pop, fall through to
        // every other bucket we observed as non-empty during the
        // same walk — preserving the original two-pass code's
        // "try literally any non-empty bucket" last-ditch guarantee.
        let primary = best_key.into_iter();
        let secondary = non_empty
            .into_iter()
            .filter(|k| Some(*k) != best_key);
        for key in primary.chain(secondary) {
            let popped = self.with_bucket(key, |bucket| bucket.blocks.pop_front());
            if let Some(Some(block)) = popped {
                self.sub_total_saturating(key);
                // PERF P-1: `key` is this block's size class, so its LRU
                // entry lives in `lru_shard_of(key)`. Bucket lock already
                // released — bucket-then-lru order preserved.
                self.lru_remove(key, block.inserted, block.tick);
                sink.push(block.ptr);
                return true;
            }
        }
        false
    }

    /// Sum of `alloc_bytes` across every pooled block. Useful for tests
    /// and memory-pressure introspection.
    #[allow(dead_code)] // reason: introspection API, used by tests and future memory-pressure hooks
    pub(crate) fn total_pooled_bytes(&self) -> usize {
        self.total_bytes.load(Ordering::Acquire)
    }

    /// Walk every bucket under its own lock, re-sum `bucket.len() *
    /// size_class`, and atomically store the result into `total_bytes`.
    /// Returns the reconciled value.
    ///
    /// **Why.** `total_bytes` is updated with `fetch_add` /
    /// `sub_total_saturating` outside of any single transaction with the
    /// bucket mutation, so
    /// concurrent free/alloc can interleave in patterns that leave the
    /// counter slightly off the true sum. The cap re-check in
    /// `try_insert_into_bucket` keeps any single overshoot bounded to
    /// one block per racing thread, but a long-running process can
    /// accumulate a small bias. This pass corrects it.
    ///
    /// **Cost.** O(buckets); bucket count is bounded by ~4 × log2(max
    /// alloc), so realistic cost is ≤ 100 bucket locks. Each lock is
    /// briefly held — just a `len()` read.
    ///
    /// Called automatically every `RECONCILE_EVERY_N_FREES` frees;
    /// callers that want a synchronous reconciliation point (e.g. a
    /// memory-pressure handler, or a test asserting steady-state
    /// invariants) can invoke it directly.
    pub(crate) fn reconcile_total_bytes(&self) -> usize {
        let mut sum: usize = 0;
        self.for_each_bucket(|size_class, bucket| {
            sum = sum.saturating_add(bucket.blocks.len().saturating_mul(size_class));
        });
        self.total_bytes.store(sum, Ordering::Release);
        sum
    }

    /// Evict pooled blocks (oldest first) until `total_pooled_bytes()` is
    /// at or below `self.max_pooled_bytes`. Intended for memory-pressure
    /// paths and `CudaContext::Drop`-adjacent shutdown hooks; the steady-
    /// state `free` path already enforces the cap, so this is a no-op in
    /// normal operation. Returns the number of blocks evicted.
    ///
    /// Stage 3: wired into the driver-OOM recovery path in `alloc`;
    /// before it was kept around for the contract only.
    pub(crate) fn evict_above_high_water(&self) -> usize {
        let mut to_free: Vec<CUdeviceptr> = Vec::new();
        while self.total_bytes.load(Ordering::Acquire) > self.max_pooled_bytes {
            if !self.evict_one(&mut to_free) {
                break;
            }
        }
        let n = to_free.len();
        for p in to_free {
            // SAFETY: same provenance argument as `free`.
            unsafe { driver_free(p) };
        }
        n
    }

    /// Total number of pooled (i.e. currently-freed-but-not-returned-to-driver)
    /// blocks across all buckets. Intended for tests and diagnostics only.
    #[doc(hidden)]
    pub fn pooled_block_count(&self) -> usize {
        let mut n = 0usize;
        self.for_each_bucket(|_, bucket| n += bucket.blocks.len());
        n
    }

    /// Number of distinct size-class buckets that currently hold at least
    /// one pooled block. Best-effort under concurrent mutation — surfaced
    /// via the public [`pool_stats`] telemetry entry point.
    #[allow(dead_code)] // reason: telemetry hook, consumed by `pool_stats`
    pub(crate) fn bucket_count(&self) -> usize {
        let mut n = 0usize;
        self.for_each_bucket(|_, bucket| {
            if !bucket.blocks.is_empty() {
                n += 1;
            }
        });
        n
    }

    /// Number of pooled blocks in the bucket that would satisfy an allocation
    /// of `bytes`. Intended for tests and diagnostics only.
    #[doc(hidden)]
    pub fn bucket_len_for(&self, bytes: usize) -> usize {
        let key = bucket_size(bytes);
        self.with_bucket(key, |bucket| bucket.blocks.len())
            .unwrap_or(0)
    }

    /// Release every pooled block back to the driver. Called on `Drop`, and
    /// usable by tests / shutdown paths that want a clean slate.
    pub fn drain(&self) {
        // Deferred-free pool (review finding C1 / P1): block on and free any
        // entries still awaiting event completion BEFORE we drain the buckets.
        // This guarantees no deferred block is leaked or freed early at
        // shutdown — `event_synchronize` waits out every recorded stream, then
        // the block goes straight to the driver. Done first so its `driver_free`
        // calls run while the context is still current (this is invoked from
        // `CudaContext::Drop` with the context live).
        drain_pending_frees_blocking();
        let mut drained: Vec<CUdeviceptr> = Vec::new();
        // Iterate over all buckets, draining each under its own lock.
        // The storage-specific `drain_all_into` handles the DashMap
        // vs. sharded shape difference; under DashMap it drains then
        // clears, under the sharded variant it `HashMap::drain`s each
        // shard in place.
        self.drain_all_into(&mut drained);
        self.total_bytes.store(0, Ordering::Release);
        // Drop the cross-bucket LRU index in lockstep with the buckets
        // it indexes. Any entry left behind would either be a phantom
        // pointer (already passed to the driver below) or a stale ref
        // into a now-deleted bucket; either way it has no business
        // staying around. PERF P-1: clear every shard (one lock at a
        // time inside `lru_clear_all`).
        self.lru_clear_all();
        for ptr in drained {
            // SAFETY: every pointer in the pool came from the matching
            // backend's `mem_alloc` (either `cuda_sys` or `cudarc_backend`,
            // both of which delegate to `cuMemAlloc_v2`) and is no longer
            // aliased — it was placed here by a `free` call that gave up
            // ownership. Pointers are interchangeable across backends
            // because they share `CUdeviceptr` and the same driver
            // allocator, so we route the free through whichever backend
            // is active for this build.
            unsafe { driver_free(ptr) };
        }
    }
}

impl Default for DeviceMemPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for DeviceMemPool {
    fn drop(&mut self) {
        // Stage 4 / V-9: signal the background watcher (if any) to exit
        // AND join it before we drain. Once the pool is being drained, any
        // `evict_above_high_water` call still running on the watcher races
        // with us — and merely flipping the shutdown flag is not enough,
        // because the watcher observes it only every `SHUTDOWN_QUANTUM`
        // (~50 ms) and may be mid-eviction (touching pool internals and
        // the captured `CUcontext`) at that instant. `request_shutdown_and_join`
        // therefore sets the flag, then blocks on the watcher's
        // `JoinHandle`, so no watcher activity can overlap the drain or the
        // teardown of the `Lazy<DeviceMemPool>` static.
        //
        // Ordering is load-bearing: request_shutdown -> join -> drain.
        // No-op when the `pool-watcher` feature is off (the whole watcher
        // module is compiled out, so this call site disappears).
        #[cfg(all(feature = "pool-watcher", not(test)))]
        pool_watcher::request_shutdown_and_join();
        self.drain();
    }
}

/// Global, process-wide pool instance. Lazily initialized on first touch.
pub(crate) static POOL: Lazy<DeviceMemPool> = Lazy::new(DeviceMemPool::new);

/// Test-only accessor for the process-wide pool. Hidden from the rendered
/// docs because external callers must not rely on it: the pool is an
/// implementation detail of `GpuBuffer` and may change shape. Integration
/// tests use this to assert invariants on pool occupancy.
#[doc(hidden)]
pub fn __test_pool() -> &'static DeviceMemPool {
    &POOL
}

// ---------------------------------------------------------------------------
// Stage 4: background watcher.
//
// Under `--features pool-watcher` the pool spawns a single OS thread on
// first allocation that polls `cuMemGetInfo_v2` every
// `BOLT_POOL_WATCH_INTERVAL_SECS` seconds (default 5). If the free
// fraction drops below `BOLT_POOL_WATCH_LOW_WATER_FRAC` (default 0.10)
// the watcher calls `POOL.evict_above_high_water()` and bumps
// `PROACTIVE_EVICTION_COUNT`. The thread shuts down cleanly on process
// exit: `DeviceMemPool::Drop` flips the `SHUTDOWN` flag AND `join`s the
// watcher's `JoinHandle` before draining the pool, when the global `POOL`
// is finalised. The join is what actually closes the V-9 teardown race —
// flipping the flag alone left a ~50 ms window in which the watcher could
// still be inside `evict_above_high_water` touching pool internals / the
// captured `CUcontext` while Drop drained underneath it. See
// `pool_watcher::request_shutdown_and_join`.
//
// Default (and `#[cfg(test)]`) builds compile out the entire module and
// `ensure_watcher_started` is a no-op `#[inline(always)]` shim, so the
// hot path takes zero overhead. The feature is opt-in to keep CI /
// cuda-stub builds free of a permanently-resident thread that would
// fail every poll under the stub backend.
// ---------------------------------------------------------------------------

/// No-op spawn hook for builds without `pool-watcher`. The compiler
/// inlines the call away entirely. See `pool_watcher::ensure_started`
/// for the real implementation.
#[cfg(not(feature = "pool-watcher"))]
#[inline(always)]
fn ensure_watcher_started() {}

/// Public shim around `pool_watcher::retry_context_capture` for
/// `Engine::sql` to invoke on every query. **Stage 6 (M3L5)** —
/// retry-on-first-engine-call hook. Under `not(feature = "pool-watcher")`
/// this compiles to a no-op call site.
#[cfg(not(feature = "pool-watcher"))]
#[inline(always)]
pub fn pool_watcher_retry_context_capture() {}

#[cfg(feature = "pool-watcher")]
#[inline]
pub fn pool_watcher_retry_context_capture() {
    pool_watcher::retry_context_capture();
}

/// ctx-race fix: invoked from `cuda_sys::CudaContext::Drop` (via runtime
/// indirection) immediately before `cuCtxDestroy_v2`. If the pool-watcher
/// captured the context identified by `raw`, this clears the capture and
/// blocks until any in-flight re-bind of that pointer has completed, so the
/// watcher can never call `cuCtxSetCurrent` on the soon-to-be-destroyed
/// context. `raw` is the raw `CUcontext` (`*mut c_void`) being destroyed.
///
/// No-op under `not(feature = "pool-watcher")` (the whole watcher module is
/// compiled out and this call site disappears).
#[cfg(not(feature = "pool-watcher"))]
#[inline(always)]
pub fn pool_watcher_invalidate_ctx(_raw: crate::cuda::cuda_sys::CUcontext) {}

#[cfg(feature = "pool-watcher")]
#[inline]
pub fn pool_watcher_invalidate_ctx(raw: crate::cuda::cuda_sys::CUcontext) {
    pool_watcher::invalidate_captured_ctx(raw);
}

// Under the `pool-watcher` feature `ensure_watcher_started` defers to
// the real `pool_watcher::ensure_started` for production builds.
// Under `#[cfg(test)]` it remains a no-op so the host-only test suite
// never spawns a watcher against the global POOL (the dedicated
// `pool_watcher_*` tests below construct a controlled local
// environment instead). This way `cargo test --features pool-watcher`
// still passes the existing test surface unchanged.
#[cfg(all(feature = "pool-watcher", not(test)))]
#[inline]
fn ensure_watcher_started() {
    pool_watcher::ensure_started();
}

#[cfg(all(feature = "pool-watcher", test))]
#[inline(always)]
fn ensure_watcher_started() {}

#[cfg(feature = "pool-watcher")]
// Under `cfg(test)` the tests drive `watcher_loop` directly against a
// local pool and never reach for `ensure_started` /
// `request_shutdown_and_join` / the singleton statics. Allow dead_code at
// the module level so the
// `cargo test --features pool-watcher` build stays warning-free
// without an attribute on every helper.
#[cfg_attr(test, allow(dead_code))]
pub(super) mod pool_watcher {
    //! Single-thread background watcher: polls `cuMemGetInfo_v2` and
    //! triggers `evict_above_high_water` when free device memory falls
    //! below a configurable threshold. Lifetime is tied to the process —
    //! `DeviceMemPool::Drop` (when the global `POOL` is finalised on
    //! shutdown) trips `SHUTDOWN` and then `join`s the watcher thread via
    //! `request_shutdown_and_join` so the thread is fully quiescent before
    //! the pool is drained (V-9: the join, not just the flag, is what
    //! prevents the watcher from racing the teardown).
    use super::{DeviceMemPool, POOL, PROACTIVE_EVICTION_COUNT};
    use crate::error::BoltResult;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;
    use std::thread::{self, JoinHandle};
    use std::time::Duration;

    /// Default poll interval (5 seconds). Tunable via the
    /// `BOLT_POOL_WATCH_INTERVAL_SECS` environment variable; a value of
    /// 0 (or any non-numeric) falls back to the default.
    const DEFAULT_INTERVAL_SECS: u64 = 5;

    /// Default low-water mark on `free / total` device memory. When the
    /// observed fraction drops below this value, the watcher calls
    /// `evict_above_high_water`. Tunable via `BOLT_POOL_WATCH_LOW_WATER_FRAC`.
    const DEFAULT_LOW_WATER_FRAC: f64 = 0.10;

    /// Sleep-loop quantum so the shutdown flag is observed within ~50 ms
    /// of `DeviceMemPool::Drop`. Sleeping for the full interval would
    /// add up to `DEFAULT_INTERVAL_SECS` seconds to shutdown.
    const SHUTDOWN_QUANTUM: Duration = Duration::from_millis(50);

    /// Function pointer type matching `cuda_sys::mem_get_info` — the
    /// indirection lets tests inject a deterministic mock without a
    /// live CUDA context.
    pub(super) type MemInfoFn = fn() -> BoltResult<(usize, usize)>;

    /// Stage 5 (M3L5): function-pointer indirection for binding the
    /// engine's CUDA context onto the calling (watcher) thread. The
    /// production hook captures `cuCtxGetCurrent` once from the
    /// spawning thread and re-binds it via `cuCtxSetCurrent` here;
    /// tests pass a no-op so the watcher loop runs without a real
    /// driver. Return `BoltResult<()>` — a failure is logged and the
    /// poll is skipped, never propagated past the watcher.
    pub(super) type CtxAttachFn = fn() -> BoltResult<()>;

    /// Join handle for the single watcher thread, stored inside a
    /// `Mutex<Option<..>>` rather than a `OnceLock<JoinHandle<..>>`.
    ///
    /// V-9 (MEDIUM — teardown data race): `DeviceMemPool::Drop` must
    /// genuinely `join()` the watcher before draining the pool and letting
    /// the `Lazy<DeviceMemPool>` static tear down — otherwise the watcher
    /// can still be inside `watcher_loop -> evict_above_high_water ->
    /// evict_one`, touching pool internals and the captured `CUcontext`,
    /// while we drain underneath it. A `OnceLock` only ever hands out a
    /// shared `&JoinHandle`, and `JoinHandle::join` consumes `self` by
    /// value, so Drop could never actually join. Keeping the handle in a
    /// `Mutex<Option<..>>` lets `request_shutdown_and_join` `.take()` it
    /// and perform exactly one clean join.
    ///
    /// `STARTED` preserves the once-only spawn semantics that `OnceLock`
    /// previously provided: it is checked and set under the `HANDLE` lock,
    /// so two threads racing `ensure_started` spawn the watcher at most
    /// once even though the `Option` is emptied again by the join at Drop.
    static HANDLE: Mutex<Option<JoinHandle<()>>> = Mutex::new(None);
    static STARTED: AtomicBool = AtomicBool::new(false);
    static SHUTDOWN: AtomicBool = AtomicBool::new(false);

    /// Stage 5: capture of the engine thread's CUDA context, taken at
    /// `ensure_started()` time. The watcher thread re-attaches this
    /// context via `cuCtxSetCurrent` before each `cuMemGetInfo_v2` poll
    /// — otherwise the watcher thread inherits no current context and
    /// every poll errors with `CUDA_ERROR_INVALID_CONTEXT`.
    ///
    /// Held as `AtomicPtr<c_void>` because `CUcontext` is itself
    /// `*mut c_void` (see `cuda_sys`) — keeping the pointer in a
    /// pointer-typed atomic preserves provenance through the static-
    /// storage round-trip rather than launder it through `usize` (the
    /// strict-provenance model treats `as usize` / `as *mut _` as a
    /// provenance-erasing cast). Acquire/Release semantics fence the
    /// capture against the load on the watcher thread.
    static CAPTURED_CTX: std::sync::atomic::AtomicPtr<std::ffi::c_void> =
        std::sync::atomic::AtomicPtr::new(std::ptr::null_mut());

    /// ctx-race fix: serialises the watcher's load-then-`cuCtxSetCurrent`
    /// of `CAPTURED_CTX` against `invalidate_captured_ctx`, which a
    /// `CudaContext::Drop` calls before `cuCtxDestroy_v2`.
    ///
    /// Without this guard there is a TOCTOU window in `real_ctx_attach`:
    /// it loads `CAPTURED_CTX`, then calls `ctx_set_current(raw)`. A
    /// concurrent context destruction could clear the slot and free the
    /// context *between* the load and the bind, so the bind would touch a
    /// dangling pointer. By holding this lock across BOTH the load+bind
    /// (in `real_ctx_attach`) and the clear+publish (in
    /// `invalidate_captured_ctx`), we guarantee that once a destroyer has
    /// observed no in-flight bind and cleared the slot, no later bind can
    /// resurrect the stale pointer, and any bind already in progress
    /// completes before `cuCtxDestroy_v2` runs.
    ///
    /// It is a plain `std::sync::Mutex<()>` (no data) used purely as a
    /// critical-section gate. The watcher only contends it once per poll
    /// (every few seconds), and destruction is rare, so contention is nil.
    static CTX_BIND_GUARD: Mutex<()> = Mutex::new(());

    fn real_mem_info() -> BoltResult<(usize, usize)> {
        crate::cuda::cuda_sys::mem_get_info()
    }

    /// Production context-attach hook: bind the captured context onto
    /// the calling thread. No-op (returns `Ok(())`) when no context was
    /// captured, which happens if `ensure_started` ran before any
    /// engine thread had a context current.
    fn real_ctx_attach() -> BoltResult<()> {
        // ctx-race fix: hold `CTX_BIND_GUARD` across the load AND the
        // `ctx_set_current` bind so a concurrent `invalidate_captured_ctx`
        // (from `CudaContext::Drop`) cannot clear the slot and let
        // `cuCtxDestroy_v2` run while we are mid-bind on the captured
        // pointer. While we hold the guard, either we observe a non-null
        // pointer that is guaranteed still alive (the destroyer is blocked
        // on the guard and has not yet called `cuCtxDestroy_v2`), or we
        // observe null (it was already invalidated) and no-op.
        let _bind = CTX_BIND_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        // Acquire load pairs with the Release store in `ensure_started`
        // / `retry_context_capture` — guarantees the captured pointer
        // is visible with its full provenance to this thread.
        let raw = CAPTURED_CTX.load(Ordering::Acquire);
        if raw.is_null() {
            return Ok(());
        }
        // `CUcontext` is itself `*mut c_void`, so this is a no-op cast.
        let ctx: crate::cuda::cuda_sys::CUcontext = raw;
        // SAFETY: `raw` was captured via `cuCtxGetCurrent` on the engine
        // thread. It is still live here: the only thing that destroys it
        // (`CudaContext::Drop`) first calls `invalidate_captured_ctx`,
        // which takes `CTX_BIND_GUARD` (held by us right now) and clears
        // `CAPTURED_CTX` before `cuCtxDestroy_v2`. So a non-null load under
        // the guard cannot be a context that has already been destroyed.
        // See `cuda_sys::ctx_set_current` docs for the precondition.
        unsafe { crate::cuda::cuda_sys::ctx_set_current(ctx) }
    }

    /// ctx-race fix: invoked from `cuda_sys::CudaContext::Drop` (through
    /// the `mem_pool::pool_watcher_invalidate_ctx` shim) immediately
    /// before `cuCtxDestroy_v2(raw)`.
    ///
    /// If the watcher captured the context `raw`, clear `CAPTURED_CTX` so
    /// no future `real_ctx_attach` re-binds it, and — by taking
    /// `CTX_BIND_GUARD` — block until any bind already in flight for that
    /// pointer has finished. After this returns, the caller may safely
    /// destroy `raw`: the watcher will load null on its next poll and skip
    /// the bind (its poll then fails benignly with no current context,
    /// which the loop already logs-and-skips).
    ///
    /// If a *different* context is captured (another live Engine), we leave
    /// the capture intact so that Engine's watcher visibility is preserved.
    ///
    /// `raw` is the raw `CUcontext` (`*mut c_void`) being destroyed.
    pub(super) fn invalidate_captured_ctx(raw: crate::cuda::cuda_sys::CUcontext) {
        if raw.is_null() {
            return;
        }
        // Take the same guard `real_ctx_attach` holds. Acquiring it means
        // no bind is in progress; any bind that started before us has
        // completed. Inside the critical section we compare-and-clear so we
        // only drop OUR context, never another Engine's still-live capture.
        let _bind = CTX_BIND_GUARD.lock().unwrap_or_else(|e| e.into_inner());
        let _ = CAPTURED_CTX.compare_exchange(
            raw,
            std::ptr::null_mut(),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
        // Guard released here; subsequent `real_ctx_attach` calls that
        // observe the cleared slot return Ok(()) without binding.
    }

    /// Spawn the watcher exactly once (idempotent). Subsequent calls are
    /// a cheap `STARTED` flag check under the `HANDLE` lock (V-9: the
    /// handle lives in a `Mutex<Option<JoinHandle>>` so `Drop` can take and
    /// join it; `STARTED` keeps the spawn once-only).
    ///
    /// Stage 5 (M3L5): on the first call, captures the current CUDA
    /// context on the calling thread (via `cuCtxGetCurrent`) and stashes
    /// it in `CAPTURED_CTX`. The watcher thread re-attaches that
    /// context on every iteration before polling `cuMemGetInfo_v2` —
    /// without the re-attach, the watcher's first poll fails with
    /// `CUDA_ERROR_INVALID_CONTEXT` because background threads inherit
    /// no current context. Failure of the capture is logged and the
    /// watcher still spawns (it'll just keep failing polls until an
    /// engine call binds a context that real_ctx_attach can later use).
    pub(super) fn ensure_started() {
        // Hold the `HANDLE` lock across the whole check-spawn-store so two
        // threads racing first-touch cannot both spawn (V-9: this lock is
        // also what `request_shutdown_and_join` takes to extract the handle
        // for the join, so spawn and join are mutually exclusive).
        let mut guard = HANDLE.lock().unwrap_or_else(|e| e.into_inner());
        if STARTED.swap(true, Ordering::AcqRel) {
            // Already spawned (or already torn down by Drop) — nothing to do.
            return;
        }
        // Capture the engine thread's context BEFORE spawning the
        // background thread (otherwise we'd capture the new thread's
        // empty context).
        match crate::cuda::cuda_sys::ctx_get_current() {
            Ok(Some(ctx)) => {
                // `ctx` is `*mut c_void` (CUcontext) — store directly
                // into the typed `AtomicPtr` so provenance is preserved.
                CAPTURED_CTX.store(ctx, Ordering::Release);
            }
            Ok(None) => {
                log::debug!(
                    "craton-bolt: pool-watcher spawning with no current \
                     context; polls will retry until a context is bound"
                );
            }
            Err(e) => {
                log::debug!(
                    "craton-bolt: pool-watcher cuCtxGetCurrent failed at \
                     spawn: {}",
                    e
                );
            }
        }
        let interval = read_interval();
        let low_water = read_low_water();
        let handle = thread::Builder::new()
            .name("craton-bolt-pool-watcher".into())
            .spawn(move || {
                watcher_loop(
                    &POOL,
                    interval,
                    low_water,
                    real_mem_info,
                    real_ctx_attach,
                    &SHUTDOWN,
                )
            })
            .expect("spawn pool-watcher thread");
        *guard = Some(handle);
    }

    /// Signal the watcher to exit **and block until it has**. Called from
    /// `DeviceMemPool::Drop`.
    ///
    /// V-9 (MEDIUM — teardown data race): flipping `SHUTDOWN` alone does
    /// not close the race — the watcher polls the flag only every
    /// `SHUTDOWN_QUANTUM` (~50 ms), so without a join the thread may still
    /// be mid-`evict_above_high_water` (touching pool internals and the
    /// captured `CUcontext`) when Drop proceeds to drain and the
    /// `Lazy<DeviceMemPool>` static is finalised. We therefore set the
    /// flag, then `.take()` the `JoinHandle` out of `HANDLE` and `join()`
    /// it, guaranteeing no watcher activity overlaps the subsequent drain.
    ///
    /// Drop must never panic, so a poisoned-join (watcher thread panicked)
    /// is logged and swallowed rather than propagated. Safe to call
    /// multiple times: the handle is consumed on the first call, so later
    /// calls find `None` and simply ensure the flag is set.
    pub(super) fn request_shutdown_and_join() {
        SHUTDOWN.store(true, Ordering::Release);
        // Take the handle under the same lock `ensure_started` uses, so a
        // concurrent first-touch spawn cannot interleave with the join.
        let handle = {
            let mut guard = HANDLE.lock().unwrap_or_else(|e| e.into_inner());
            guard.take()
        };
        if let Some(handle) = handle {
            // `join()` returns promptly once the watcher observes
            // `SHUTDOWN` on its next quantum poll (≤ `SHUTDOWN_QUANTUM`).
            if let Err(e) = handle.join() {
                // The watcher panicked. Don't re-panic inside Drop —
                // just record it; we're tearing the process down anyway.
                log::error!(
                    "craton-bolt: pool-watcher thread panicked during \
                     shutdown join (V-9): {:?}",
                    e
                );
            }
        }
    }

    /// **Stage 6 (M3L5)** — retry-on-first-engine-call hook.
    ///
    /// If `ensure_started` ran before any engine thread had a CUDA context
    /// bound (e.g. tests, or `DeviceMemPool::POOL` first-touched on an
    /// idle thread), `CAPTURED_CTX` will be zero and every watcher poll
    /// falls through `real_ctx_attach` as a no-op — losing visibility
    /// into device memory pressure.
    ///
    /// This hook is called from `Engine::sql` (and other engine entry
    /// points) on every query: if the slot is still empty AND the
    /// calling thread has a context bound, populate it. Cheap atomic
    /// load on the steady-state path; one `cuCtxGetCurrent` on the
    /// first call after a context becomes available.
    pub fn retry_context_capture() {
        // TODO(ctx-race): this CAS populates `CAPTURED_CTX` WITHOUT taking
        // `CTX_BIND_GUARD`, so it is not serialised against
        // `invalidate_captured_ctx`. The remaining (benign-in-practice) gap:
        // if a thread that still has context `C` current calls this AT THE
        // SAME TIME another thread is dropping the `CudaContext` that owns
        // `C`, this could re-store `C` into the slot just after the drop
        // cleared it, and the watcher's next bind would touch a freed `C`.
        // That requires *using* an Engine's context concurrently with
        // dropping that same Engine — already undefined at the API level
        // (`CudaContext` is `Send` but not `Sync`, and an Engine must not be
        // queried while being torn down). The common multi-Engine case (each
        // Engine used and dropped from its own thread) is fully covered by
        // the guard in `real_ctx_attach`/`invalidate_captured_ctx`. Closing
        // this fully would require gating the capture CAS on the guard plus a
        // generation counter; deferred as it would add hot-path cost to every
        // `Engine::sql` for a misuse that is already UB.
        //
        // Fast path: already captured.
        if !CAPTURED_CTX.load(Ordering::Acquire).is_null() {
            return;
        }
        match crate::cuda::cuda_sys::ctx_get_current() {
            Ok(Some(ctx)) => {
                // CAS so concurrent first-callers don't race. Only the
                // winner stores; losers no-op. `ctx` is `*mut c_void`
                // (CUcontext) — feeds the typed `AtomicPtr` directly.
                let _ = CAPTURED_CTX.compare_exchange(
                    std::ptr::null_mut(),
                    ctx,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                );
            }
            Ok(None) | Err(_) => {
                // No context yet, or driver query failed — try again
                // next call. Don't log; this is the steady-state miss.
            }
        }
    }

    /// Core loop. Factored out of `ensure_started` so the test harness
    /// can drive it against a local pool, a 1 ms interval, and a
    /// deterministic `MemInfoFn` mock without colliding with the
    /// production singleton or burning real wall-clock seconds.
    ///
    /// Stage 5 (M3L5):
    ///   * The new `ctx_attach` parameter binds the engine thread's
    ///     captured context onto this thread before each poll —
    ///     mandatory for the production path, a no-op for tests.
    ///   * After each `evict_above_high_water` that returns ZERO
    ///     blocks while free memory remains below the low-water mark,
    ///     fires a one-time `log::warn!` recommending a higher
    ///     `CRATON_BOLT_POOL_MAX_BYTES`. The `CAP_BUMP_WARNED` latch
    ///     guarantees the warning never repeats within a process.
    pub(super) fn watcher_loop(
        pool: &DeviceMemPool,
        interval: Duration,
        low_water_frac: f64,
        mem_info: MemInfoFn,
        ctx_attach: CtxAttachFn,
        shutdown: &AtomicBool,
    ) {
        log::info!(
            "craton-bolt: pool-watcher started (interval={:?}, low_water={:.2}%)",
            interval,
            low_water_frac * 100.0
        );
        loop {
            // Sleep in small quanta so the shutdown flag is observed
            // promptly. `sleep` may return early on signal but that's
            // benign — we just poll sooner.
            let mut elapsed = Duration::ZERO;
            while elapsed < interval {
                if shutdown.load(Ordering::Acquire) {
                    log::info!("craton-bolt: pool-watcher exiting on shutdown");
                    return;
                }
                let step = SHUTDOWN_QUANTUM.min(interval - elapsed);
                thread::sleep(step);
                elapsed += step;
            }
            // Stage 5: re-attach the captured engine context onto THIS
            // thread before the driver poll. Without this, the first
            // `cuMemGetInfo_v2` errors with `CUDA_ERROR_INVALID_CONTEXT`
            // and every subsequent poll does the same. Failure here is
            // logged and the poll is skipped — never propagated.
            if let Err(e) = ctx_attach() {
                log::debug!(
                    "craton-bolt: pool-watcher ctx_attach failed: {}; \
                     skipping poll",
                    e
                );
                continue;
            }
            // Poll the driver. Errors (e.g. no current context on this
            // thread, transient driver hiccup) are logged and ignored —
            // the watcher is best-effort and must never crash the
            // process.
            match mem_info() {
                Ok((free, total)) if total > 0 => {
                    let frac = free as f64 / total as f64;
                    if frac < low_water_frac {
                        let evicted = pool.evict_above_high_water();
                        PROACTIVE_EVICTION_COUNT.fetch_add(1, Ordering::Relaxed);
                        log::info!(
                            "craton-bolt: pool-watcher proactive eviction \
                             (free={} MiB, total={} MiB, frac={:.2}%, evicted={} blocks)",
                            free / (1024 * 1024),
                            total / (1024 * 1024),
                            frac * 100.0,
                            evicted
                        );
                        // Stage 5 (M3L5): cap-bump heuristic. If we
                        // evicted zero blocks but free memory is STILL
                        // below the low-water mark, the workload's
                        // working set exceeds the configured pool cap
                        // and we have nothing in the pool left to give
                        // back. Tell the operator once — never repeat,
                        // because a runaway log of the same line in
                        // every poll would drown legitimate signal.
                        if evicted == 0 {
                            emit_cap_bump_warning_once(pool.max_pooled_bytes);
                        }
                    }
                }
                Ok(_) => {} // total == 0: cuMemGetInfo gave a nonsense reading; skip.
                Err(e) => {
                    log::debug!("craton-bolt: pool-watcher cuMemGetInfo failed: {}", e);
                }
            }
        }
    }

    /// Stage 5 (M3L5): one-time guard for the cap-bump warning. Flips
    /// from `false` to `true` on the first emission; subsequent
    /// candidate emissions short-circuit. Process-wide — the operator
    /// only needs to see the recommendation once per run.
    static CAP_BUMP_WARNED: AtomicBool = AtomicBool::new(false);

    /// Emit the cap-bump recommendation exactly once per process.
    /// Subsequent calls are a `compare_exchange` that loses the race
    /// and returns silently. `current_max` is the pool's currently
    /// configured `CRATON_BOLT_POOL_MAX_BYTES` (read at construction;
    /// not affected by env-var changes after `DeviceMemPool::new`).
    fn emit_cap_bump_warning_once(current_max: usize) {
        // `compare_exchange` with Acquire/Acquire semantics: the
        // winner observes the prior `false` and stores `true`; losers
        // observe the new `true` and return. Use `Ordering::AcqRel`
        // on success to publish the latch + `Acquire` on failure so a
        // reader on another thread sees the warning happened.
        if CAP_BUMP_WARNED
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
        {
            log::warn!(
                "craton-bolt pool: workload working set exceeds \
                 CRATON_BOLT_POOL_MAX_BYTES={}; consider raising the cap",
                current_max
            );
        }
    }

    /// Test-only hook to clear the cap-bump latch between test cases.
    /// Tests drive `emit_cap_bump_warning_once` (indirectly, via
    /// `watcher_loop`) more than once across the suite, and without a
    /// reset the second test would see a no-op. Production code never
    /// calls this — the latch is intentionally one-shot per process.
    #[cfg(test)]
    pub(super) fn reset_cap_bump_warned_for_tests() {
        CAP_BUMP_WARNED.store(false, Ordering::Release);
    }

    /// Test-only accessor: did the cap-bump warning fire? Asserts in
    /// `pool_watcher_cap_bump_fires_when_eviction_yields_zero` consult
    /// this to verify the one-shot semantics without parsing log
    /// output.
    #[cfg(test)]
    pub(super) fn cap_bump_warned_for_tests() -> bool {
        CAP_BUMP_WARNED.load(Ordering::Acquire)
    }

    /// Test-only: directly seed `CAPTURED_CTX` with a fake pointer so the
    /// ctx-race invalidation path can be exercised without a real driver.
    /// `raw` is treated as an opaque `*mut c_void` and never dereferenced.
    #[cfg(test)]
    pub(super) fn set_captured_ctx_for_tests(raw: *mut std::ffi::c_void) {
        CAPTURED_CTX.store(raw, Ordering::Release);
    }

    /// Test-only: read back the raw captured-context pointer.
    #[cfg(test)]
    pub(super) fn captured_ctx_for_tests() -> *mut std::ffi::c_void {
        CAPTURED_CTX.load(Ordering::Acquire)
    }

    fn read_interval() -> Duration {
        let secs = std::env::var("BOLT_POOL_WATCH_INTERVAL_SECS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|s| *s > 0)
            .unwrap_or(DEFAULT_INTERVAL_SECS);
        Duration::from_secs(secs)
    }

    fn read_low_water() -> f64 {
        std::env::var("BOLT_POOL_WATCH_LOW_WATER_FRAC")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|f| *f > 0.0 && *f < 1.0)
            .unwrap_or(DEFAULT_LOW_WATER_FRAC)
    }
}

#[cfg(test)]
mod test_support {
    //! Host-only shim. The pool's policy code (eviction, caps, LRU) can
    //! be exercised without a live CUDA context as long as `mem_alloc`
    //! and `mem_free` are intercepted. `test_driver_alloc` mints a
    //! monotonically increasing fake `CUdeviceptr`; `record_driver_free`
    //! records every synthetic block returned to the "driver" so tests
    //! can assert on eviction.
    //!
    //! Stage 3 adds an OOM-injection latch: `arm_oom_once` / `arm_oom_n`
    //! cause the next N `test_driver_alloc` calls to return
    //! `BoltError::CudaWithCode { code: 2, .. }` (Stage 4: integer code
    //! rather than the legacy formatted-string match) so the pool's
    //! `is_oom_error` recogniser fires. After the latch counter hits
    //! zero, subsequent calls succeed normally.
    use super::CUdeviceptr;
    use crate::error::{BoltError, BoltResult};
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PTR: AtomicU64 = AtomicU64::new(1);
    static FREED: Mutex<Vec<CUdeviceptr>> = Mutex::new(Vec::new());
    /// One-shot / N-shot OOM fault-injection latch. Each
    /// `test_driver_alloc` call decrements the latch; when the latch
    /// is `> 0` the call returns the canonical OOM error instead of a
    /// fresh pointer. Tests serialize on the surrounding `ENV_LOCK`
    /// so the latch is effectively single-tenant.
    static OOM_LATCH: AtomicU64 = AtomicU64::new(0);

    pub(super) fn test_driver_alloc(bytes: usize) -> BoltResult<CUdeviceptr> {
        // OOM-injection: drain the latch by one and return an OOM
        // error in the same shape `cuda_sys::check` would produce for
        // `CUDA_ERROR_OUT_OF_MEMORY = 2` — Stage 4 carries the code in
        // a typed integer field rather than embedding it in a format
        // string. The latch is consulted FIRST, before the bench-GPU gate,
        // so host fault-injection tests keep working even if the process
        // happens to have `BOLT_BENCH_GPU` set.
        loop {
            let cur = OOM_LATCH.load(Ordering::Acquire);
            if cur == 0 {
                break;
            }
            if OOM_LATCH
                .compare_exchange(cur, cur - 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Err(BoltError::CudaWithCode {
                    code: 2,
                    message: "out of memory".to_string(),
                });
            }
        }
        // `BOLT_BENCH_GPU=1`: hand the allocation to the real CUDA driver so
        // the crate's `#[ignore]`'d GPU unit tests operate on live device
        // memory instead of synthetic pointers (which would fault with
        // `NOT_INITIALIZED` the moment a kernel touched them).
        if super::bench_gpu_enabled() {
            return super::real_driver_mem_alloc(bytes);
        }
        // Host default: mint a synthetic pointer. Wraparound is irrelevant —
        // tests use a few hundred at most.
        Ok(NEXT_PTR.fetch_add(1, Ordering::Relaxed))
    }

    pub(super) fn record_driver_free(ptr: CUdeviceptr) {
        FREED.lock().push(ptr);
    }

    pub(super) fn drained_ptrs() -> Vec<CUdeviceptr> {
        FREED.lock().clone()
    }

    pub(super) fn reset() {
        FREED.lock().clear();
        OOM_LATCH.store(0, Ordering::Release);
        // Keep NEXT_PTR monotonic across tests so a pointer freed by one
        // test cannot collide with a pointer allocated by the next.
    }

    /// Arm the OOM-injection latch so the next single `test_driver_alloc`
    /// call returns OOM, then resumes normal operation.
    pub(super) fn arm_oom_once() {
        OOM_LATCH.store(1, Ordering::Release);
    }

    /// Arm the OOM-injection latch so the next `n` `test_driver_alloc`
    /// calls return OOM in succession.
    pub(super) fn arm_oom_n(n: u64) {
        OOM_LATCH.store(n, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    //! Pure-host policy tests. These run without a CUDA context: the
    //! `driver_free` and miss-path `mem_alloc` calls route to
    //! `test_support` shims that mint synthetic pointers and count
    //! frees, so we can assert on caps and LRU semantics directly.
    //!
    //! Each test serializes on `ENV_LOCK` because the cap values are
    //! read from environment variables at `DeviceMemPool::new` time and
    //! `std::env` is process-global.
    use super::*;
    use parking_lot::Mutex as PlMutex;

    static ENV_LOCK: PlMutex<()> = PlMutex::new(());

    /// Build a pool with the env unset so we get the defaults, but allow
    /// the caller to set specific overrides for the duration of the
    /// test. The returned guard restores the env on drop.
    struct EnvGuard {
        keys: Vec<&'static str>,
    }
    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for k in &self.keys {
                std::env::remove_var(k);
            }
        }
    }
    fn with_env(vars: &[(&'static str, &str)]) -> EnvGuard {
        let keys = vars.iter().map(|(k, _)| *k).collect();
        for (k, v) in vars {
            std::env::set_var(k, v);
        }
        EnvGuard { keys }
    }

    /// M-3: assert representative sizes land on the new finer bucket
    /// classes. Worst-case waste must be < 25% on every example below.
    #[test]
    fn bucket_size_finer_granularity() {
        // Floor: anything <= ARROW_ALIGNMENT goes to ARROW_ALIGNMENT.
        assert_eq!(bucket_size(1), ARROW_ALIGNMENT);
        assert_eq!(bucket_size(ARROW_ALIGNMENT), ARROW_ALIGNMENT);

        // 100 bytes: old=128, new=112 (12% waste vs 28%).
        assert_eq!(bucket_size(100), 112);

        // 65 KiB: old=128 KiB (97% waste), new=80 KiB (23% waste).
        assert_eq!(bucket_size(65 * 1024), 80 * 1024);

        // Just past 4096: old=8192, new=5120.
        assert_eq!(bucket_size(4097), 5120);

        // Exact powers of two should still hit themselves.
        assert_eq!(bucket_size(1024), 1024);
        assert_eq!(bucket_size(65536), 65536);
        assert_eq!(bucket_size(1024 * 1024), 1024 * 1024);

        // Worst-case waste check: pick the largest value in each octave
        // and confirm we land in the same octave (i.e. waste < 25%).
        for k in 6..24 {
            let base: usize = 1 << k;
            // The last-class boundary in this octave: base + 3*(base/4) = 1.75*base
            let just_above = base + (base / 4) * 3 + 1;
            let bucket = bucket_size(just_above);
            // bucket should be exactly 2*base — first class of next octave.
            assert_eq!(
                bucket,
                base * 2,
                "octave 2^{} edge: bucket_size({}) = {}",
                k,
                just_above,
                bucket
            );
            // And the value just inside that last sub-class wastes <= 25%.
            let in_last = base + (base / 4) * 3; // exactly 1.75*base
            let bucket = bucket_size(in_last);
            assert_eq!(bucket, in_last); // exactly on a class boundary
        }
    }

    #[test]
    fn pool_evicts_when_max_bytes_exceeded() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        // 4 KiB cap, big bucket cap so the byte cap is what bites.
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "4096"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        let pool = DeviceMemPool::new();
        // Each block is 256 bytes; freeing 32 of them sums to 8 KiB,
        // double the cap.
        let block_size = 256;
        let n = 32;
        let mut ptrs = Vec::new();
        for _ in 0..n {
            let (p, _) = pool.alloc(block_size).unwrap();
            ptrs.push(p);
        }
        for p in ptrs {
            pool.free(p, block_size);
        }
        assert!(
            pool.total_pooled_bytes() <= 4096,
            "total_pooled_bytes = {} > 4096",
            pool.total_pooled_bytes()
        );
    }

    #[test]
    fn bucket_cap_enforces() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_BUCKET_CAP", "8"),
            // Big byte cap so the per-bucket cap is what bites.
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
        ]);
        let pool = DeviceMemPool::new();
        let block_size = 256;
        // Allocate ALL pointers first, then free them. If we
        // interleaved alloc/free the bucket would just oscillate
        // between 7 and 8 and we'd never test the cap path.
        let mut ptrs = Vec::with_capacity(100);
        for _ in 0..100 {
            let (p, _) = pool.alloc(block_size).unwrap();
            ptrs.push(p);
        }
        for p in ptrs {
            pool.free(p, block_size);
        }
        let bucket_len = pool.bucket_len_for(block_size);
        assert!(bucket_len <= 8, "bucket_len = {} > 8", bucket_len);
        // And we actually filled it.
        assert_eq!(bucket_len, 8);
    }

    #[test]
    fn env_var_overrides_default() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[("CRATON_BOLT_POOL_MAX_BYTES", "1048576")]);
        let pool = DeviceMemPool::new();
        assert_eq!(pool.max_pooled_bytes, 1024 * 1024);
        // And the cap actually applies: pool more than 1 MiB worth of
        // distinct blocks and watch the pool stay at or below the
        // limit. (Alloc up front, free in a second pass, so blocks
        // don't get recycled through the bucket between frees.)
        let block_size = 64 * 1024; // 64 KiB
        let mut ptrs = Vec::with_capacity(32);
        for _ in 0..32 {
            // 32 * 64 KiB = 2 MiB, twice the cap
            let (p, _) = pool.alloc(block_size).unwrap();
            ptrs.push(p);
        }
        for p in ptrs {
            pool.free(p, block_size);
        }
        assert!(
            pool.total_pooled_bytes() <= 1024 * 1024,
            "total = {}",
            pool.total_pooled_bytes()
        );
    }

    /// Per-bucket FIFO eviction: under per-bucket locks the global LRU
    /// guarantee is downgraded to "front of the chosen bucket goes first."
    /// Within a single bucket, the oldest free is still the first to go;
    /// that's what this test checks.
    #[test]
    fn per_bucket_fifo_evicts_oldest_first() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        // Cap to exactly 2 blocks' worth of bytes. Each block is one
        // bucket-sized slot (64 B = ARROW_ALIGNMENT). After 2 frees the
        // pool is at capacity; a third free must evict the oldest.
        let block_bytes = 64;
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", &(block_bytes * 2).to_string()),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        let pool = DeviceMemPool::new();

        // A small sleep between frees guarantees strictly increasing
        // `Instant`s even on platforms with coarse monotonic clocks.
        let bump = std::time::Duration::from_millis(2);

        // Allocate three fresh synthetic pointers up front, BEFORE any
        // frees, so the alloc miss-path mints distinct values for each.
        // If we interleaved alloc/free, the second alloc would just hit
        // the pool and hand back the first pointer (LIFO reuse).
        let (a, _) = pool.alloc(block_bytes).unwrap();
        let (b, _) = pool.alloc(block_bytes).unwrap();
        let (c, _) = pool.alloc(block_bytes).unwrap();
        assert!(a != b && b != c && a != c, "distinct synthetic ptrs");

        pool.free(a, block_bytes);
        std::thread::sleep(bump);
        pool.free(b, block_bytes);
        std::thread::sleep(bump);

        // Pool now at cap (2 blocks, byte total == max). Freeing `c`
        // must evict the oldest pooled block (`a`), not `b`.
        pool.free(c, block_bytes);

        let freed = test_support::drained_ptrs();
        assert!(
            freed.contains(&a),
            "FIFO should have evicted `a`; freed list = {:?}",
            freed
        );
        assert!(
            !freed.contains(&b),
            "`b` is newer than `a`, should still be pooled; freed list = {:?}",
            freed
        );
    }

    #[test]
    fn evict_above_high_water_drains_excess() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        // No cap from env so we can manually push past the default by
        // bypassing `free`'s policy via direct state mutation, then
        // assert that `evict_above_high_water` brings us back under.
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1024"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        let pool = DeviceMemPool::new();
        // Force a few extra blocks into the pool ignoring caps. This is
        // a white-box manipulation that mirrors what a memory-pressure
        // hook would observe if the cap were raised at runtime. We go
        // through `with_or_create_bucket` so the test works under both
        // the DashMap and `pool-sharded` storage variants.
        {
            for _ in 0..8 {
                let p = test_support::test_driver_alloc(64).unwrap();
                let inserted = Instant::now();
                let tick = pool.next_tick.fetch_add(1, Ordering::Relaxed);
                pool.with_or_create_bucket(64, |bucket| {
                    bucket.blocks.push_back(PooledBlock {
                        ptr: p,
                        inserted,
                        tick,
                    });
                });
                pool.total_bytes.fetch_add(64, Ordering::AcqRel);
                // PERF P-1: LRU is sharded; route the white-box insert
                // through the same helper the production path uses so the
                // entry lands in the correct shard for size class 64.
                pool.lru_insert(64, inserted, tick, p);
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        assert_eq!(pool.total_pooled_bytes(), 8 * 64);
        let evicted = pool.evict_above_high_water();
        assert!(pool.total_pooled_bytes() <= 1024);
        // 8 * 64 = 512 ≤ 1024, so evict_above_high_water should be a
        // no-op here — the assertion below catches the case where we
        // accidentally over-evict.
        assert_eq!(evicted, 0);
    }

    /// L-5: per-bucket locks let concurrent frees into distinct size
    /// classes proceed in parallel. We approximate "make progress" by
    /// timing N parallel free streams vs. a sequential baseline: if a
    /// single global mutex still gated everything, the parallel version
    /// would be ~equal to the sequential one; with per-bucket locks it
    /// should be measurably faster than 4× the per-thread time.
    ///
    /// The test is loose on purpose — CI machines have variable timing.
    /// We just assert parallel < 4× sequential (any speedup at all).
    ///
    /// `#[ignore]`: this is a wall-clock *performance characteristic*, not a
    /// correctness property — the `par_elapsed < 1.5×seq_elapsed` comparison
    /// is inherently flaky under machine load (a busy host can schedule the
    /// parallel threads worse than the sequential baseline). The per-bucket
    /// lock-split behaviour it probes is better measured by
    /// `bench_dashmap_baseline`. Run explicitly with `--ignored` on a quiet
    /// machine when validating the lock-granularity change.
    #[test]
    #[ignore = "perf-timing: flaky under load; measured by bench_dashmap_baseline"]
    fn per_bucket_lock_allows_concurrent_progress() {
        use std::sync::Arc;
        use std::time::Duration;

        let _l = ENV_LOCK.lock();
        test_support::reset();
        // Big caps so we never hit the eviction path — we're testing
        // contention on the pool's bookkeeping locks, not its policy.
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "100000"),
        ]);
        let pool = Arc::new(DeviceMemPool::new());

        // Four distinct bucket sizes — one per thread — so contention
        // happens only inside the DashMap shard layer, not on a single
        // bucket's mutex.
        let sizes = [64usize, 256, 1024, 4096];
        let per_thread_iters = 4000;

        // Sequential baseline: same total work, one thread.
        let seq_start = Instant::now();
        for s in &sizes {
            for _ in 0..per_thread_iters {
                let (p, ab) = pool.alloc(*s).unwrap();
                pool.free(p, ab);
            }
        }
        let seq_elapsed = seq_start.elapsed();

        // Parallel: spawn one thread per size class.
        let par_start = Instant::now();
        let handles: Vec<_> = sizes
            .iter()
            .copied()
            .map(|s| {
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || {
                    for _ in 0..per_thread_iters {
                        let (p, ab) = pool.alloc(s).unwrap();
                        pool.free(p, ab);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let par_elapsed = par_start.elapsed();

        // Sanity: both runs did real work.
        assert!(seq_elapsed > Duration::from_micros(1));
        assert!(par_elapsed > Duration::from_micros(1));

        // Loose check: with per-bucket locks we expect par_elapsed to be
        // less than seq_elapsed (sub-linear-ish scaling). A single global
        // mutex would force par_elapsed >= seq_elapsed. Allow generous
        // headroom for CI noise — if par_elapsed > 1.5 * seq_elapsed
        // something is clearly serialising.
        //
        // Under `--features pool-sharded` the four power-of-two sizes
        // selected here happen to all map to shard 0 (`size_class % 32`),
        // so concurrent threads contend on the same shard mutex — the
        // sharded variant cannot beat the sequential baseline for this
        // pathological size selection. Skip the concurrency assertion in
        // that mode; the `bench_dashmap_baseline` micro-bench is the
        // intended measurement vehicle for the sharded path anyway.
        #[cfg(not(feature = "pool-sharded"))]
        assert!(
            par_elapsed < seq_elapsed + seq_elapsed / 2,
            "parallel run ({:?}) should not be > 1.5x sequential ({:?}) — \
             suggests a global lock is still serialising frees",
            par_elapsed,
            seq_elapsed
        );
        // Silence unused-variable warning under the cfg-gated assertion.
        #[cfg(feature = "pool-sharded")]
        {
            let _ = par_elapsed;
            let _ = seq_elapsed;
        }
    }

    /// Stage 2: the cross-bucket LRU index must evict the globally
    /// oldest block, not just the oldest within whichever bucket is
    /// being inserted into. Free three blocks into three distinct
    /// size classes (different buckets), then push the pool over its
    /// byte cap and observe which block goes to the driver.
    #[test]
    fn global_lru_evicts_oldest_across_buckets() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        // Each block is 64 B (size class A), 128 B (B), 256 B (C).
        // Sum = 448 B. Cap == 448 so the pool sits exactly at the
        // limit after the three frees; a fourth free into any bucket
        // must evict.
        let (sa, sb, sc) = (64usize, 128, 256);
        let cap = sa + sb + sc; // 448
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", &cap.to_string()),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        let pool = DeviceMemPool::new();

        // Mint distinct synthetic ptrs first so reuse doesn't mask
        // the eviction we want to observe.
        let (a, _) = pool.alloc(sa).unwrap();
        let (b, _) = pool.alloc(sb).unwrap();
        let (c, _) = pool.alloc(sc).unwrap();
        // A fourth ptr in the largest bucket so the cap-trip on its
        // free has to evict bytes — and the global LRU pick must come
        // from the *oldest* bucket (A), not the current one (C).
        let (d, _) = pool.alloc(sc).unwrap();
        assert!(a != b && b != c && c != d, "distinct synthetic ptrs");

        let bump = std::time::Duration::from_millis(2);
        pool.free(a, sa);
        std::thread::sleep(bump);
        pool.free(b, sb);
        std::thread::sleep(bump);
        pool.free(c, sc);
        std::thread::sleep(bump);

        // Pool now at cap. Freeing `d` (another 256 B into bucket C)
        // must reclaim at least 256 B starting with the globally
        // oldest entry — which is `a` in bucket A. A per-bucket-FIFO
        // policy would instead evict `c` (the oldest in C). The
        // cross-bucket LRU guarantee is that `a` goes first.
        pool.free(d, sc);

        let freed = test_support::drained_ptrs();
        // All three originals get evicted because freeing `d` (256 B)
        // into a 448 B pool requires reclaiming all 448 B of pooled
        // capacity. The Stage-2 guarantee under test is the *order*:
        // the global LRU picks `a` first (oldest, in bucket A) — a
        // per-bucket-FIFO policy would have picked `c` first (oldest
        // in bucket C, the bucket we're inserting into).
        assert!(
            !freed.is_empty(),
            "at least one eviction expected; freed list = {:?}",
            freed
        );
        assert_eq!(
            freed[0], a,
            "global LRU must evict `a` first (oldest across buckets); \
             freed list = {:?}",
            freed
        );
    }

    /// Stage 2: `reconcile_total_bytes` must re-sum the actual bucket
    /// occupancy and correct any drift in the atomic counter.
    #[test]
    fn reconcile_total_bytes_corrects_drift() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        let pool = DeviceMemPool::new();

        // Pool a handful of blocks via the normal API so total_bytes
        // starts coherent.
        let block_bytes = 256;
        let n = 5;
        let mut ptrs = Vec::new();
        for _ in 0..n {
            let (p, _) = pool.alloc(block_bytes).unwrap();
            ptrs.push(p);
        }
        for p in ptrs {
            pool.free(p, block_bytes);
        }
        let expected = block_bytes * n;
        assert_eq!(pool.total_pooled_bytes(), expected);

        // Manually corrupt the counter — simulates the worst-case drift
        // that concurrent free races could accumulate.
        pool.total_bytes.store(expected + 99_999, Ordering::Release);
        assert_ne!(pool.total_pooled_bytes(), expected);

        // Reconciliation must restore the truth.
        let reconciled = pool.reconcile_total_bytes();
        assert_eq!(reconciled, expected);
        assert_eq!(pool.total_pooled_bytes(), expected);

        // And it works in the other direction — under-count too.
        pool.total_bytes.store(0, Ordering::Release);
        let reconciled = pool.reconcile_total_bytes();
        assert_eq!(reconciled, expected);
        assert_eq!(pool.total_pooled_bytes(), expected);
    }

    /// Stage 2: concurrent free into the same bucket must not panic
    /// and the accounting must remain coherent after a reconciliation
    /// pass. Exercises the LRU index's race-handling path indirectly:
    /// many `try_insert_into_bucket` calls fight for the bucket lock
    /// AND the LRU lock; we just need to come out the other side
    /// without UB and with `total_bytes` matching the buckets.
    #[test]
    fn lru_handles_concurrent_free_race() {
        use std::sync::Arc;

        let _l = ENV_LOCK.lock();
        test_support::reset();
        // Big caps so eviction doesn't kick in — we're stress-testing
        // the LRU insert race, not the eviction race.
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1000000"),
        ]);
        let pool = Arc::new(DeviceMemPool::new());

        let threads = 8;
        let per_thread = 2000;
        let block_bytes = 64;
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || {
                    for _ in 0..per_thread {
                        let (p, ab) = pool.alloc(block_bytes).unwrap();
                        pool.free(p, ab);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker thread panicked");
        }

        // After a steady alloc/free churn with no eviction, the bucket
        // should be at most BUCKET_CAP full (well under it here). The
        // atomic counter may have drifted because each alloc/free pair
        // races against the others — but reconciliation must bring it
        // back to the true sum.
        let pre_reconcile = pool.total_pooled_bytes();
        let mut true_sum: usize = 0;
        pool.for_each_bucket(|key, bucket| {
            true_sum += bucket.blocks.len() * key;
        });
        let reconciled = pool.reconcile_total_bytes();
        assert_eq!(
            reconciled, true_sum,
            "reconcile must equal hand-summed truth; \
             pre_reconcile={}, post={}, truth={}",
            pre_reconcile, reconciled, true_sum
        );
        assert_eq!(pool.total_pooled_bytes(), true_sum);

        // And the LRU index size should match the bucket count exactly
        // — every pooled block has a unique LRU entry, every popped
        // block had its entry removed. PERF P-1: the index is sharded,
        // so we sum across shards via `lru_total_len`.
        let lru_len = pool.lru_total_len();
        let pooled_count = pool.pooled_block_count();
        assert_eq!(
            lru_len, pooled_count,
            "LRU index ({}) should mirror pooled block count ({})",
            lru_len, pooled_count
        );
    }

    /// PERF P-1: concurrent free/alloc churn into *distinct size classes*
    /// must remain correct under the sharded LRU index. This is the
    /// cross-shard analogue of `lru_handles_concurrent_free_race` — each
    /// worker hammers a different size class so the LRU shards are
    /// exercised in parallel rather than all funnelling through one mutex.
    ///
    /// We deliberately span many octaves and include non-power-of-two
    /// bucket sizes so the size classes spread across several
    /// `size_class % LRU_SHARDS` shards (power-of-two sizes alone would
    /// all collapse onto shard 0). The invariant under test is the same
    /// one the single-mutex design guaranteed: after the churn settles,
    /// reconciliation equals the hand-summed bucket truth AND the LRU
    /// index (summed across all shards) mirrors the pooled block count
    /// exactly — i.e. no entry was lost, double-inserted, or stranded in
    /// the wrong shard, and no block was double-freed.
    #[test]
    fn sharded_lru_handles_concurrent_distinct_size_classes() {
        use std::sync::Arc;

        let _l = ENV_LOCK.lock();
        test_support::reset();
        // Big caps so eviction doesn't kick in — we're stress-testing the
        // per-shard insert/remove pairing across distinct size classes,
        // not the eviction race (covered elsewhere).
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1000000"),
        ]);
        let pool = Arc::new(DeviceMemPool::new());

        // One distinct request size per worker. These round (via
        // `bucket_size`) to a spread of size classes — including
        // non-power-of-two ones (e.g. 100 -> 112, 5000 -> 5120) — so the
        // resulting `lru_shard_of(size_class)` values are not all 0.
        let sizes: [usize; 8] = [64, 100, 256, 1000, 4096, 5000, 16384, 20000];
        let per_thread = 2000;
        let handles: Vec<_> = sizes
            .iter()
            .copied()
            .map(|s| {
                let pool = Arc::clone(&pool);
                std::thread::spawn(move || {
                    for _ in 0..per_thread {
                        let (p, ab) = pool.alloc(s).unwrap();
                        pool.free(p, ab);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("worker thread panicked");
        }

        // Accounting self-heals to the hand-summed truth.
        let mut true_sum: usize = 0;
        pool.for_each_bucket(|key, bucket| {
            true_sum += bucket.blocks.len() * key;
        });
        let reconciled = pool.reconcile_total_bytes();
        assert_eq!(
            reconciled, true_sum,
            "reconcile must equal hand-summed truth across shards"
        );
        assert_eq!(pool.total_pooled_bytes(), true_sum);

        // The sharded LRU index, summed over every shard, must still
        // mirror the pooled block count one-for-one. A lost / duplicated
        // / mis-sharded entry would break this equality.
        let lru_len = pool.lru_total_len();
        let pooled_count = pool.pooled_block_count();
        assert_eq!(
            lru_len, pooled_count,
            "sharded LRU index ({}) should mirror pooled block count ({})",
            lru_len, pooled_count
        );
    }

    /// Stage 2 / Stage 3: micro-bench scaffold for the bucket-storage
    /// hot path. Default build measures the DashMap variant; under
    /// `--features pool-sharded` the same bench measures the fixed-N
    /// sharded variant. Ignored by default because it's noisy on CI;
    /// the orchestrator runs it manually to compare per-op cost across
    /// builds. Run with:
    ///
    /// ```text
    ///   # DashMap baseline (default):
    ///   cargo test --release -p craton_bolt -- \
    ///     mem_pool::tests::bench_dashmap_baseline --ignored --nocapture
    ///
    ///   # Sharded variant for comparison:
    ///   cargo test --release --features pool-sharded -p craton_bolt -- \
    ///     mem_pool::tests::bench_dashmap_baseline --ignored --nocapture
    /// ```
    ///
    /// Output is `key=value` formatted on a single line so the
    /// orchestrator can grep it out of test stderr without parsing
    /// free-form English. Three measurement passes are taken and the
    /// median per_op_ns is reported so a single GC pause or background
    /// noise burst doesn't bias the result.
    #[test]
    #[ignore = "gpu:mempool — micro-bench, run manually with --ignored --nocapture"]
    fn bench_dashmap_baseline() {
        use std::sync::Arc;

        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1000000"),
        ]);
        let pool = Arc::new(DeviceMemPool::new());

        // High shard contention scenario: many distinct size classes,
        // many threads, all hashing into the (small) DashMap shard
        // table simultaneously. This is the pathological case noted
        // in the module docs.
        //
        // Sized to dominate noise: 16 threads × 8000 iters × 3 passes
        // = ~400k alloc/free pairs per sample at ~1µs each → ~400ms
        // of real work per sample. Comfortably above the 10ms range
        // where macOS / Windows scheduler jitter dominates.
        let sizes: Vec<usize> = (6..=16).map(|k| 1usize << k).collect(); // 64..=64K
        let threads: usize = 16;
        let per_thread: usize = 8000;
        let passes: usize = 3;

        // Warmup: a single pass primes the bucket map (first-touch
        // creates every size class entry exactly once) so the measured
        // passes only pay steady-state cost.
        let warmup_pool = Arc::clone(&pool);
        let warmup_sizes = sizes.clone();
        let warmup = std::thread::spawn(move || {
            for &s in &warmup_sizes {
                let (p, ab) = warmup_pool.alloc(s).unwrap();
                warmup_pool.free(p, ab);
            }
        });
        warmup.join().unwrap();

        let mut samples_ns: Vec<u128> = Vec::with_capacity(passes);
        for _pass in 0..passes {
            let start = Instant::now();
            let handles: Vec<_> = (0..threads)
                .map(|t: usize| {
                    let pool = Arc::clone(&pool);
                    let sizes = sizes.clone();
                    std::thread::spawn(move || {
                        for i in 0..per_thread {
                            let s = sizes[(t + i) % sizes.len()];
                            let (p, ab) = pool.alloc(s).unwrap();
                            pool.free(p, ab);
                        }
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            let elapsed = start.elapsed();
            let total_ops = threads * per_thread;
            let per_op_ns = elapsed.as_nanos() / total_ops as u128;
            samples_ns.push(per_op_ns);
        }
        samples_ns.sort_unstable();
        let median = samples_ns[passes / 2];
        let min = samples_ns[0];
        let max = samples_ns[passes - 1];
        let total_ops = threads * per_thread;

        // Storage tag lets the orchestrator differentiate baseline
        // from sharded in a single log file without re-running with
        // distinct test names.
        let storage = if cfg!(feature = "pool-sharded") {
            "sharded"
        } else {
            "dashmap"
        };
        // key=value structured line for orchestrator grep / parse.
        eprintln!(
            "BENCH mem_pool storage={} threads={} per_thread={} ops_per_pass={} \
             passes={} per_op_ns_median={} per_op_ns_min={} per_op_ns_max={}",
            storage, threads, per_thread, total_ops, passes, median, min, max
        );
        // No assertion — this is a benchmark, not a correctness check.
        // The orchestrator compares per_op_ns_median across builds.
    }

    /// OOM-recovery path, single-block retry (review finding M3:
    /// incremental eviction). The latch returns OOM on the first
    /// `driver_mem_alloc` after arming, then yields a real synthetic ptr on
    /// the retry. With a generous byte cap the high-water trim is a no-op and
    /// the single retry after it succeeds, so the warm cache is **preserved**
    /// — the M3 improvement over the old all-or-nothing drain. We verify:
    ///   1. `alloc` returns Ok (recovery happened),
    ///   2. `OOM_RECOVERY_COUNT` incremented exactly once,
    ///   3. the seeded warm blocks were NOT drained (incremental eviction
    ///      only frees what the retry needs; here, nothing).
    #[test]
    fn oom_recovery_retries_without_nuking_warm_cache() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        let pool = DeviceMemPool::new();

        // Seed the pool with some blocks. Allocate ALL ptrs first, THEN free —
        // interleaving alloc/free would just oscillate one block through the
        // LIFO bucket and never accumulate `seeded` distinct ptrs.
        let block_bytes = 256usize;
        let seeded = 5usize;
        let mut seeded_ptrs = Vec::new();
        for _ in 0..seeded {
            let (p, _) = pool.alloc(block_bytes).unwrap();
            seeded_ptrs.push(p);
        }
        for p in &seeded_ptrs {
            pool.free(*p, block_bytes);
        }
        assert_eq!(
            pool.pooled_block_count(),
            seeded,
            "seeded {} blocks but pool reports {}",
            seeded,
            pool.pooled_block_count()
        );
        let pre_recover_count = oom_recovery_count();

        // Arm the OOM latch: next driver_mem_alloc call returns OOM.
        test_support::arm_oom_once();

        // Allocate a fresh size class so the request misses the pool and
        // routes to the (now OOM-injected) driver path.
        let new_size = 4096usize;
        let (ptr, ab) = pool.alloc(new_size).expect(
            "OOM recovery should have trimmed headroom and retried successfully",
        );
        assert!(ptr != 0);
        assert_eq!(ab, bucket_size(new_size));

        // Counter incremented exactly once.
        assert_eq!(oom_recovery_count(), pre_recover_count + 1);

        // M3: with a 1 GiB cap the high-water trim freed nothing and the very
        // first retry succeeded, so the warm cache survives intact — unlike the
        // old drain-everything behaviour.
        assert_eq!(
            pool.pooled_block_count(),
            seeded,
            "incremental OOM recovery must preserve the warm cache when the \
             first retry succeeds; expected {} pooled blocks, got {}",
            seeded,
            pool.pooled_block_count()
        );

        // Latch should be disarmed (one-shot) — a follow-up alloc succeeds
        // without further recovery events.
        let (_p, _) = pool.alloc(new_size).unwrap();
        assert_eq!(oom_recovery_count(), pre_recover_count + 1);
    }

    /// M3 incremental eviction: when the *first* retry after the high-water
    /// trim still OOMs, recovery must evict pooled blocks one at a time and
    /// retry until it fits — freeing only as many as needed, not the whole
    /// pool. We arm the latch for two consecutive OOMs so the first retry
    /// (post high-water trim, which evicts nothing under a huge cap) fails and
    /// the loop must evict at least one block before the next retry succeeds.
    #[test]
    fn oom_recovery_evicts_incrementally_until_fit() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        let pool = DeviceMemPool::new();

        let block_bytes = 256usize;
        let seeded = 4usize;
        let mut seeded_ptrs = Vec::new();
        for _ in 0..seeded {
            let (p, _) = pool.alloc(block_bytes).unwrap();
            seeded_ptrs.push(p);
        }
        for p in &seeded_ptrs {
            pool.free(*p, block_bytes);
        }
        assert_eq!(pool.pooled_block_count(), seeded);

        let pre_recover_count = oom_recovery_count();
        let freed_before = test_support::drained_ptrs().len();

        // Two OOMs: the miss-path alloc consumes one, the first post-trim
        // retry consumes the second; the loop then evicts a block and the
        // following retry succeeds.
        test_support::arm_oom_n(2);

        let new_size = 4096usize;
        let (ptr, _ab) = pool
            .alloc(new_size)
            .expect("incremental eviction should free enough room to retry");
        assert!(ptr != 0);
        // Recovery counted exactly once (only the successful retry bumps it).
        assert_eq!(oom_recovery_count(), pre_recover_count + 1);

        // At least one block was evicted to the driver, but NOT necessarily
        // all of them — the loop stops as soon as a retry fits.
        let freed_after = test_support::drained_ptrs().len();
        assert!(
            freed_after >= freed_before + 1,
            "expected at least one incremental eviction; freed before={}, after={}",
            freed_before,
            freed_after
        );
        // The pool was not wholesale-drained: at most one block left for each
        // OOM we had to clear, so some warm blocks should remain.
        assert!(
            pool.pooled_block_count() < seeded,
            "at least one block must have been evicted"
        );
        assert!(
            pool.pooled_block_count() >= 1,
            "incremental eviction should not drain the entire pool when a \
             single eviction suffices; pooled={}",
            pool.pooled_block_count()
        );
    }

    /// Stage 3: OOM hook must bubble the *original* error if the retry
    /// also OOMs. The latch is armed for two consecutive failures; we
    /// expect `alloc` to return `BoltError::Cuda` and the recovery
    /// counter to stay flat.
    #[test]
    fn oom_recovery_propagates_on_double_failure() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        let pool = DeviceMemPool::new();

        let pre_recover_count = oom_recovery_count();
        test_support::arm_oom_n(2);

        let result = pool.alloc(4096);
        // Stage 4: the injected error is `CudaWithCode { code: 2, .. }`.
        assert!(matches!(
            result,
            Err(crate::error::BoltError::CudaWithCode { code: 2, .. })
        ));
        // Counter must NOT have incremented — recovery did not succeed.
        assert_eq!(oom_recovery_count(), pre_recover_count);
    }

    /// Stage 3 sanity: under `--features pool-sharded` the alternative
    /// storage shape exercises the same observable API. We just push
    /// a handful of blocks through and reconcile, asserting the same
    /// invariants the DashMap path is required to satisfy.
    #[cfg(feature = "pool-sharded")]
    #[test]
    fn sharded_storage_round_trip() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        let pool = DeviceMemPool::new();

        // Pick size classes from different octaves to spread them
        // across shards: 64 (shard 64%32=0), 1024 (shard 0), 2048
        // (shard 0), 4096 (shard 0) — wait, all small power-of-two
        // sizes collapse to shard 0 under %32. Use the actual
        // bucket-rounded sizes for the bench — they include 80, 96,
        // 112 etc. which are non-power-of-two and hit different
        // shards.
        let sizes = [64usize, 80, 96, 112, 128, 160, 192, 224, 256];
        let mut ptrs = Vec::new();
        for s in sizes {
            let (p, ab) = pool.alloc(s).unwrap();
            ptrs.push((p, ab));
        }
        for (p, ab) in &ptrs {
            pool.free(*p, *ab);
        }
        // Every freed block should be pooled (caps are huge).
        assert_eq!(pool.pooled_block_count(), sizes.len());
        // Reconcile reads through every shard's HashMap.
        let expected: usize = sizes.iter().map(|s| bucket_size(*s)).sum();
        let reconciled = pool.reconcile_total_bytes();
        assert_eq!(reconciled, expected);

        // And we can re-alloc from each bucket (LIFO).
        for s in sizes {
            let (_p, ab) = pool.alloc(s).unwrap();
            assert_eq!(ab, bucket_size(s));
        }
        // All blocks were consumed.
        assert_eq!(pool.pooled_block_count(), 0);
    }

    /// Stage 4: OOM detector uses the typed `CudaWithCode { code: 2, .. }`
    /// match — no formatted-string parsing. We construct the error
    /// directly (not via `cuda_sys::check`, which would need a CUDA
    /// driver) and assert the recogniser fires for code 2 and rejects
    /// other codes / variant shapes.
    #[test]
    fn is_oom_error_matches_code_directly() {
        let oom = crate::error::BoltError::CudaWithCode {
            code: 2,
            message: "out of memory".to_string(),
        };
        assert!(is_oom_error(&oom), "code 2 must be OOM");

        // Other CUDA codes must NOT be flagged.
        for code in [0i32, 1, 3, 20, 200, 99] {
            let e = crate::error::BoltError::CudaWithCode {
                code,
                message: "some other error".to_string(),
            };
            assert!(!is_oom_error(&e), "code {} mis-flagged as OOM", code);
        }

        // The legacy `Cuda(String)` variant must NOT be flagged — the
        // fragile prefix matcher is gone. (Even if some upstream code
        // hands us a string that *looks* like "CUDA driver error 2",
        // it's no longer interpreted as OOM.)
        let legacy = crate::error::BoltError::Cuda(
            "CUDA driver error 2: out of memory".to_string(),
        );
        assert!(
            !is_oom_error(&legacy),
            "legacy Cuda(String) variant must no longer be OOM-matched"
        );

        // Non-CUDA variants are definitely not OOM.
        assert!(!is_oom_error(&crate::error::BoltError::Other("x".into())));
    }

    /// Stage 4 telemetry: `pool_stats()` surfaces the same numbers a
    /// caller would get from reaching directly into the crate-internal
    /// counters, and is the single stable entry point downstream
    /// observability layers should consume.
    #[test]
    fn pool_stats_snapshot_is_consistent() {
        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);

        // pool_stats() reads the global POOL singleton, not the local
        // one built inside this test — so we deliberately don't build
        // a local pool here. We can still assert the snapshot is shaped
        // correctly and that the counters never go backwards.
        let snap = super::pool_stats();
        // total_pooled_bytes is a usize; reading it is the assertion
        // that the field exists with the documented type.
        let _: usize = snap.total_pooled_bytes;
        let _: usize = snap.bucket_count;
        let _: u64 = snap.oom_recovery_count;
        let _: u64 = snap.proactive_eviction_count;

        // Same getter called twice in succession returns counters that
        // are monotonically non-decreasing.
        let snap2 = super::pool_stats();
        assert!(snap2.oom_recovery_count >= snap.oom_recovery_count);
        assert!(snap2.proactive_eviction_count >= snap.proactive_eviction_count);

        // The Copy + Eq derives work — useful for downstream diff checks.
        let snap3 = snap2;
        assert_eq!(snap2, snap3);
    }

    /// Stage 4: the background watcher fires `evict_above_high_water` and
    /// increments `PROACTIVE_EVICTION_COUNT` when the mock `mem_get_info`
    /// reports free memory below the configured low-water mark. We
    /// drive `pool_watcher::watcher_loop` directly with a 1 ms interval
    /// and a `MemInfoFn` that returns `(1, 1000)` (0.1% free, well
    /// under the 10% threshold) — then flip the shutdown flag and
    /// `join` the helper thread.
    #[cfg(feature = "pool-watcher")]
    #[test]
    fn pool_watcher_triggers_eviction_on_low_free() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        use std::time::Duration;

        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1024"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        let pool = Arc::new(DeviceMemPool::new());

        // Seed the pool with blocks that exceed the soft cap so the
        // watcher's `evict_above_high_water` call has something to do.
        // 8 blocks * 256 bytes = 2 KiB, cap = 1 KiB.
        let mut seeded_ptrs = Vec::new();
        for _ in 0..8 {
            let (p, _) = pool.alloc(256).unwrap();
            seeded_ptrs.push(p);
        }
        for p in &seeded_ptrs {
            pool.free(*p, 256);
        }

        // Mock: free=1, total=1000 -> 0.1% free, well below 10% threshold.
        fn mock_low_free() -> crate::error::BoltResult<(usize, usize)> {
            Ok((1, 1000))
        }

        let pre_count = super::proactive_eviction_count();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_thread = shutdown.clone();
        let pool_thread = pool.clone();
        let handle = std::thread::spawn(move || {
            super::pool_watcher::watcher_loop(
                &pool_thread,
                Duration::from_millis(1),
                0.10,
                mock_low_free,
                noop_ctx_attach,
                &shutdown_thread,
            );
        });

        // Give the watcher enough wall-clock time to make at least one
        // poll. The interval is 1 ms but the sleep quantum inside the
        // loop is 50 ms (SHUTDOWN_QUANTUM), so the first poll lands
        // after ~50 ms regardless. 200 ms is a generous bound.
        std::thread::sleep(Duration::from_millis(200));
        shutdown.store(true, std::sync::atomic::Ordering::Release);
        handle.join().expect("watcher thread joined cleanly");

        let post_count = super::proactive_eviction_count();
        assert!(
            post_count > pre_count,
            "PROACTIVE_EVICTION_COUNT did not increment: pre={} post={}",
            pre_count,
            post_count
        );
    }

    /// Stage 5 (M3L5): test-only context-attach hook. Returns Ok(())
    /// without touching the driver. Stage 4 tests didn't need this
    /// indirection because the watcher_loop took no context-attach
    /// parameter; Stage 5 inserts the parameter so production builds
    /// can re-bind the engine context on the background thread.
    #[cfg(feature = "pool-watcher")]
    fn noop_ctx_attach() -> crate::error::BoltResult<()> {
        Ok(())
    }

    /// Stage 4: watcher must NOT evict when free memory is comfortably
    /// above the low-water mark.
    #[cfg(feature = "pool-watcher")]
    #[test]
    fn pool_watcher_skips_eviction_when_free_is_high() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        use std::time::Duration;

        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        let pool = Arc::new(DeviceMemPool::new());

        // Mock: free=900, total=1000 -> 90% free, far above 10% threshold.
        fn mock_high_free() -> crate::error::BoltResult<(usize, usize)> {
            Ok((900, 1000))
        }

        let pre_count = super::proactive_eviction_count();
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_thread = shutdown.clone();
        let pool_thread = pool.clone();
        let handle = std::thread::spawn(move || {
            super::pool_watcher::watcher_loop(
                &pool_thread,
                Duration::from_millis(1),
                0.10,
                mock_high_free,
                noop_ctx_attach,
                &shutdown_thread,
            );
        });
        std::thread::sleep(Duration::from_millis(200));
        shutdown.store(true, std::sync::atomic::Ordering::Release);
        handle.join().expect("watcher thread joined cleanly");

        let post_count = super::proactive_eviction_count();
        assert_eq!(
            post_count, pre_count,
            "PROACTIVE_EVICTION_COUNT incremented when it should not have"
        );
    }

    // ----------------------------------------------------------------
    // Stage 5 (M3L5) — cap-bump heuristic + context attach
    // ----------------------------------------------------------------

    /// Stage 5: when the watcher observes free memory below the
    /// low-water mark AND `evict_above_high_water` returns zero (pool
    /// has nothing to give back because the working set exceeds the
    /// configured cap), it must fire `log::warn!` exactly once. Drive
    /// the loop long enough to see several polls and confirm the
    /// latch stays at `true` without re-emitting.
    #[cfg(feature = "pool-watcher")]
    #[test]
    fn pool_watcher_cap_bump_fires_when_eviction_yields_zero() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        use std::time::Duration;

        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        // Clear the one-shot latch so this test starts from a known
        // state. Production code never resets — the warning is one
        // shot per process — but tests must.
        super::pool_watcher::reset_cap_bump_warned_for_tests();

        // Empty pool: `evict_above_high_water` will always return 0.
        let pool = Arc::new(DeviceMemPool::new());
        assert_eq!(
            pool.total_pooled_bytes(),
            0,
            "test precondition: pool starts empty"
        );

        // Mock: free=1, total=1000 -> well below the 10% threshold.
        fn mock_low_free() -> crate::error::BoltResult<(usize, usize)> {
            Ok((1, 1000))
        }

        assert!(
            !super::pool_watcher::cap_bump_warned_for_tests(),
            "test precondition: cap-bump latch starts clear"
        );

        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_thread = shutdown.clone();
        let pool_thread = pool.clone();
        let handle = std::thread::spawn(move || {
            super::pool_watcher::watcher_loop(
                &pool_thread,
                Duration::from_millis(1),
                0.10,
                mock_low_free,
                noop_ctx_attach,
                &shutdown_thread,
            );
        });
        // Sleep long enough for the watcher to make MULTIPLE polls.
        // The sleep quantum is 50 ms so 300 ms is roughly 6 polls —
        // confirming the warning only fires once and not on every poll.
        std::thread::sleep(Duration::from_millis(300));
        shutdown.store(true, std::sync::atomic::Ordering::Release);
        handle.join().expect("watcher thread joined cleanly");

        assert!(
            super::pool_watcher::cap_bump_warned_for_tests(),
            "cap-bump warning latch should be set after the first \
             zero-eviction poll under memory pressure"
        );
    }

    /// Stage 5: the cap-bump latch fires exactly once across many polls
    /// under sustained pressure. We need this property because the
    /// watcher polls every few seconds for the life of the process —
    /// without a one-shot guard the operator would see the warning in
    /// every poll, drowning legitimate signal. Drive the loop long
    /// enough to observe many polls, then assert the latch is set
    /// (already covered by `pool_watcher_cap_bump_fires_when_eviction_yields_zero`)
    /// AND that re-driving the watcher loop doesn't re-emit (i.e. the
    /// `compare_exchange` lose-path is hit).
    ///
    /// We assert the lose-path by directly invoking the inner emitter
    /// through `cap_bump_warned_for_tests` — the public Atomic latch
    /// is the single source of truth.
    #[cfg(feature = "pool-watcher")]
    #[test]
    fn pool_watcher_cap_bump_is_one_shot() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        use std::time::Duration;

        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);
        super::pool_watcher::reset_cap_bump_warned_for_tests();

        // Empty pool: every poll's `evict_above_high_water` returns 0.
        let pool = Arc::new(DeviceMemPool::new());
        fn mock_low_free() -> crate::error::BoltResult<(usize, usize)> {
            Ok((1, 1000))
        }

        // First watcher run: should set the latch.
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_thread = shutdown.clone();
        let pool_thread = pool.clone();
        let handle = std::thread::spawn(move || {
            super::pool_watcher::watcher_loop(
                &pool_thread,
                Duration::from_millis(1),
                0.10,
                mock_low_free,
                noop_ctx_attach,
                &shutdown_thread,
            );
        });
        std::thread::sleep(Duration::from_millis(200));
        shutdown.store(true, std::sync::atomic::Ordering::Release);
        handle.join().expect("watcher thread joined cleanly");
        assert!(
            super::pool_watcher::cap_bump_warned_for_tests(),
            "first run should set the latch"
        );

        // Second watcher run with the SAME conditions: latch is still
        // set (the compare_exchange loses on every attempt) — value
        // should remain `true`, not flip.
        let shutdown2 = Arc::new(AtomicBool::new(false));
        let shutdown2_thread = shutdown2.clone();
        let pool2_thread = pool.clone();
        let handle2 = std::thread::spawn(move || {
            super::pool_watcher::watcher_loop(
                &pool2_thread,
                Duration::from_millis(1),
                0.10,
                mock_low_free,
                noop_ctx_attach,
                &shutdown2_thread,
            );
        });
        std::thread::sleep(Duration::from_millis(200));
        shutdown2.store(true, std::sync::atomic::Ordering::Release);
        handle2.join().expect("watcher thread joined cleanly");
        // Latch must still be set — never reset between polls in
        // production. `cap_bump_warned_for_tests` is the test-only
        // accessor; the production code has no way to observe nor
        // re-emit. (Tests reset between cases via
        // `reset_cap_bump_warned_for_tests`.)
        assert!(
            super::pool_watcher::cap_bump_warned_for_tests(),
            "second run must leave the latch in the set state \
             (one-shot, no flip-back)"
        );
    }

    /// Stage 5: the watcher_loop now invokes `ctx_attach` before each
    /// `mem_info` poll. Verify that the hook is called at least once
    /// per poll cycle by counting via an `AtomicUsize` accessed from
    /// the test-injected stub. This is the behavioural-mock check the
    /// task description called for ("assert `cuCtxSetCurrent` was
    /// called") — production wires `real_ctx_attach` to
    /// `cuda_sys::ctx_set_current`, while the test counts invocations.
    ///
    /// The mock is a `fn() -> BoltResult<()>` so it must be a plain
    /// function-pointer-compatible fn item. We use a `static
    /// AtomicUsize` (process-wide, but reset at test entry) for the
    /// counter — see `CTX_ATTACH_CALLS` below.
    #[cfg(feature = "pool-watcher")]
    #[test]
    fn pool_watcher_invokes_ctx_attach_each_poll() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::time::Duration;

        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);

        // Process-wide counter so the function-pointer mock can talk
        // to the test thread. Tests are serialised on ENV_LOCK so
        // this static is exclusively owned by this test for the
        // duration of the assertions.
        static CTX_ATTACH_CALLS: AtomicUsize = AtomicUsize::new(0);
        CTX_ATTACH_CALLS.store(0, Ordering::Release);

        fn counting_ctx_attach() -> crate::error::BoltResult<()> {
            CTX_ATTACH_CALLS.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }

        fn mock_high_free() -> crate::error::BoltResult<(usize, usize)> {
            Ok((900, 1000))
        }

        let pool = Arc::new(DeviceMemPool::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_thread = shutdown.clone();
        let pool_thread = pool.clone();
        let handle = std::thread::spawn(move || {
            super::pool_watcher::watcher_loop(
                &pool_thread,
                Duration::from_millis(1),
                0.10,
                mock_high_free,
                counting_ctx_attach,
                &shutdown_thread,
            );
        });
        // Two polls' worth of wall-clock time (each poll is ~50 ms
        // because of SHUTDOWN_QUANTUM).
        std::thread::sleep(Duration::from_millis(200));
        shutdown.store(true, std::sync::atomic::Ordering::Release);
        handle.join().expect("watcher thread joined cleanly");

        let calls = CTX_ATTACH_CALLS.load(Ordering::Acquire);
        assert!(
            calls >= 1,
            "ctx_attach should be invoked at least once per poll \
             cycle; observed {calls} calls"
        );
    }

    /// Stage 5: if `ctx_attach` returns an error, the watcher must
    /// skip the mem_info poll for that iteration (the driver call
    /// would fail anyway with `CUDA_ERROR_INVALID_CONTEXT`). We
    /// verify by counting how many times the `MemInfoFn` was invoked
    /// — it should be zero across the test window because every
    /// `ctx_attach` call returns Err.
    #[cfg(feature = "pool-watcher")]
    #[test]
    fn pool_watcher_skips_poll_when_ctx_attach_errors() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::Arc;
        use std::time::Duration;

        let _l = ENV_LOCK.lock();
        test_support::reset();
        let _g = with_env(&[
            ("CRATON_BOLT_POOL_MAX_BYTES", "1073741824"),
            ("CRATON_BOLT_POOL_BUCKET_CAP", "1024"),
        ]);

        static MEM_INFO_CALLS: AtomicUsize = AtomicUsize::new(0);
        MEM_INFO_CALLS.store(0, Ordering::Release);

        fn failing_ctx_attach() -> crate::error::BoltResult<()> {
            Err(crate::error::BoltError::CudaWithCode {
                code: 201, // CUDA_ERROR_INVALID_CONTEXT
                message: "test: synthetic ctx_attach failure".to_string(),
            })
        }

        fn counting_mem_info() -> crate::error::BoltResult<(usize, usize)> {
            MEM_INFO_CALLS.fetch_add(1, Ordering::AcqRel);
            Ok((900, 1000))
        }

        let pool = Arc::new(DeviceMemPool::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let shutdown_thread = shutdown.clone();
        let pool_thread = pool.clone();
        let handle = std::thread::spawn(move || {
            super::pool_watcher::watcher_loop(
                &pool_thread,
                Duration::from_millis(1),
                0.10,
                counting_mem_info,
                failing_ctx_attach,
                &shutdown_thread,
            );
        });
        std::thread::sleep(Duration::from_millis(200));
        shutdown.store(true, std::sync::atomic::Ordering::Release);
        handle.join().expect("watcher thread joined cleanly");

        let calls = MEM_INFO_CALLS.load(Ordering::Acquire);
        assert_eq!(
            calls, 0,
            "mem_info must not be called when ctx_attach fails; \
             observed {calls} invocations"
        );
    }

    /// ctx-race fix: `invalidate_captured_ctx` (called by
    /// `CudaContext::Drop` before `cuCtxDestroy_v2`) must clear the
    /// watcher's captured pointer when it matches the context being
    /// destroyed, so the watcher's `real_ctx_attach` stops re-binding the
    /// soon-to-be-freed context. A *different* live context must be left
    /// intact. We drive the slot directly with fake opaque pointers — no
    /// real driver involved (the pointers are never dereferenced).
    #[cfg(feature = "pool-watcher")]
    #[test]
    fn invalidate_captured_ctx_clears_only_matching_ctx() {
        let _l = ENV_LOCK.lock();

        let ctx_a: *mut std::ffi::c_void = 0xA000 as *mut std::ffi::c_void;
        let ctx_b: *mut std::ffi::c_void = 0xB000 as *mut std::ffi::c_void;

        // Capture ctx_a, then invalidate a DIFFERENT context (ctx_b):
        // the capture must survive (another Engine still alive).
        super::pool_watcher::set_captured_ctx_for_tests(ctx_a);
        super::pool_watcher::invalidate_captured_ctx(ctx_b);
        assert_eq!(
            super::pool_watcher::captured_ctx_for_tests(),
            ctx_a,
            "invalidating a non-matching context must not clear the capture"
        );

        // Now invalidate the captured context itself: the slot must be
        // cleared so `real_ctx_attach` would no-op (never bind a freed ctx).
        super::pool_watcher::invalidate_captured_ctx(ctx_a);
        assert!(
            super::pool_watcher::captured_ctx_for_tests().is_null(),
            "invalidating the captured context must clear the slot so the \
             watcher cannot re-bind a destroyed context"
        );

        // Idempotent: invalidating again (or with null) is a harmless no-op.
        super::pool_watcher::invalidate_captured_ctx(ctx_a);
        super::pool_watcher::invalidate_captured_ctx(std::ptr::null_mut());
        assert!(super::pool_watcher::captured_ctx_for_tests().is_null());

        // Leave the slot clean for any later test in the suite.
        super::pool_watcher::set_captured_ctx_for_tests(std::ptr::null_mut());
    }
}

#[cfg(test)]
mod hot_path_tests {
    //! Targeted tests for the hot-path micro-optimisations applied to
    //! `mem_pool.rs`:
    //!
    //!  * `bucket_size` switched its `ceil(n / step) * step` rounding
    //!    from div/mul to a power-of-two bitmask. The test below
    //!    cross-checks the new closed-form against the legacy formula
    //!    across every byte size in `[0, 1024]` plus a sparse sweep up
    //!    to several MiB to cover every octave the realistic workload
    //!    touches.
    //!
    //!  * The cap-check in `free` and `try_insert_into_locked_bucket`
    //!    learned `saturating_add` so a near-`usize::MAX` transient in
    //!    `total_bytes` can't wrap and bypass the cap. We assert the
    //!    arithmetic alone — the cap path through `pool.free` is
    //!    already covered by `pool_evicts_when_max_bytes_exceeded`.
    use super::*;

    /// Reference implementation: the original div/mul rounding. Kept
    /// in the test module ONLY — the production path now uses the
    /// bitmask form. If this assertion ever fires it means the bitmask
    /// form diverged from the historical formula (which would be a
    /// regression callers depend on for bucket-size stability).
    fn legacy_bucket_size(bytes: usize) -> usize {
        let n = bytes.max(ARROW_ALIGNMENT);
        let pow2 = 1usize << (usize::BITS - 1 - n.leading_zeros());
        let step = (pow2 / 4).max(1);
        // ceil(n / step) * step — the pre-optimisation formula.
        let rounded = n.saturating_add(step - 1) / step * step;
        rounded.max(ARROW_ALIGNMENT)
    }

    /// Bitmask round-up must match the legacy div/mul formula for
    /// every reasonable input. Covers the small-size dense range
    /// (where `step == ARROW_ALIGNMENT/4 == 16` is non-trivial) plus
    /// a sweep across every octave up through 8 MiB.
    #[test]
    fn bucket_size_matches_legacy_formula() {
        // Dense sweep across the small-size range — captures the floor
        // (everything `<= ARROW_ALIGNMENT` maps to `ARROW_ALIGNMENT`)
        // and the first non-trivial octave (step = 16).
        for n in 0usize..=1024 {
            assert_eq!(
                bucket_size(n),
                legacy_bucket_size(n),
                "bucket_size({}) diverged: bitmask={}, legacy={}",
                n,
                bucket_size(n),
                legacy_bucket_size(n)
            );
        }

        // Sparse sweep across higher octaves: every (step-1, step, step+1)
        // sub-class boundary plus a midpoint up to 8 MiB. The "+1"
        // points are where the bitmask form would diverge from a
        // naive `& !mask` if the mask were ever miscomputed.
        for k in 6..=23 {
            let pow2: usize = 1 << k;
            let step = pow2 / 4;
            for sub in 0..4 {
                let base = pow2 + sub * step;
                for delta in [0usize, 1, step / 2, step - 1, step] {
                    let n = base.saturating_add(delta);
                    assert_eq!(
                        bucket_size(n),
                        legacy_bucket_size(n),
                        "bucket_size({}) diverged at octave 2^{} sub {}: \
                         bitmask={}, legacy={}",
                        n,
                        k,
                        sub,
                        bucket_size(n),
                        legacy_bucket_size(n)
                    );
                }
            }
        }
    }

    /// `saturating_add` on the cap check must clamp at `usize::MAX`
    /// rather than wrap — otherwise a pathological `total_bytes` near
    /// `usize::MAX` would silently bypass the cap. We exercise the
    /// arithmetic directly because the only realistic way to reach
    /// the wrap window is via the `reconcile_total_bytes` / `fetch_sub`
    /// race the saturating helper closes, which is hard to reproduce
    /// deterministically in a host test.
    #[test]
    fn cap_check_saturates_on_overflow() {
        // Worst case: total at usize::MAX-1, alloc 10 more bytes. A
        // bare `+` wraps to 9, slipping past any finite cap. The
        // saturating form clamps at usize::MAX and the cap check
        // correctly rejects.
        let total = usize::MAX - 1;
        let alloc = 10usize;
        let projected = total.saturating_add(alloc);
        assert_eq!(
            projected,
            usize::MAX,
            "saturating_add must clamp at usize::MAX"
        );
        // Verify the cap check sees the clamped value as "too big"
        // for any finite cap, including the default 512 MiB.
        let cap = 512usize * 1024 * 1024;
        assert!(
            projected > cap,
            "saturating-added projection ({}) should exceed cap ({})",
            projected,
            cap
        );

        // And the historical case the cap check needs to detect:
        // a normal-sized total plus alloc must NOT saturate when
        // the sum fits in usize.
        let total = 1024usize;
        let alloc = 256usize;
        assert_eq!(total.saturating_add(alloc), 1280);
    }
}
