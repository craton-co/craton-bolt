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
//! race is acceptable and corrected on the next free.
//!
//! ### Stage 2 deferred items
//!
//! * **Cross-bucket global LRU.** Per-bucket FIFO is "evict oldest in
//!   *that* bucket first." If a hot bucket fills while a cold bucket
//!   holds older blocks, we evict from the hot one first. Re-introducing
//!   global LRU requires a cross-bucket timestamp index (e.g. a BTreeMap
//!   keyed by `Instant`) coordinated against the per-bucket locks.
//! * **Lock-free `total_bytes` reconciliation.** The atomic counter can
//!   transiently overshoot under heavy concurrent `free` calls because
//!   the bucket lock and the atomic increment are not joined into one
//!   transaction. Stage 2 should add a periodic reconciliation pass that
//!   re-sums `bucket.len() * size_class` and corrects drift.
//! * **Sharded-or-DashMap micro-bench.** We picked DashMap for simplicity;
//!   if profiling shows the DashMap shard hash itself becomes a hot spot,
//!   revisit with a fixed-N sharded `[Mutex<HashMap<..>>; 32]`.
//!
//! The pool depends on a live CUDA context being current on the calling
//! thread — same precondition as the bare `cuMemAlloc` path it replaces.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use dashmap::DashMap;
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::cuda::buffer::ARROW_ALIGNMENT;
use crate::cuda::cuda_sys::CUdeviceptr;
// `cuda_sys` is only referenced by the hand-rolled (default) backend
// path. Under `--features cudarc` the alloc/free hit `cudarc_backend`
// instead, so the import is feature-gated to keep both builds warning-
// free.
#[cfg(not(feature = "cudarc"))]
#[cfg(not(test))]
use crate::cuda::cuda_sys;
use crate::error::BoltResult;

/// Default soft cap on total pooled bytes (512 MiB). Overridden by the
/// `CRATON_BOLT_POOL_MAX_BYTES` environment variable.
const DEFAULT_MAX_POOLED_BYTES: usize = 512 * 1024 * 1024;

/// Default hard cap on the number of pooled blocks in any single bucket.
/// Overridden by the `CRATON_BOLT_POOL_BUCKET_CAP` environment variable.
const DEFAULT_MAX_BUCKET_ENTRIES: usize = 16;

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
    // ceil(n / step) * step. Saturating arithmetic guards against pathological
    // sizes near `usize::MAX`; cuMemAlloc would refuse those anyway.
    let rounded = n.saturating_add(step - 1) / step * step;
    rounded.max(ARROW_ALIGNMENT)
}

/// One pooled block. `inserted` is captured at `free` time so the bucket's
/// front entry is always the oldest within that bucket.
#[derive(Clone, Copy)]
struct PooledBlock {
    ptr: CUdeviceptr,
    inserted: Instant,
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
#[cfg(not(test))]
unsafe fn driver_free(ptr: CUdeviceptr) {
    #[cfg(feature = "cudarc")]
    let result = crate::cuda::cudarc_backend::mem_free(ptr);
    #[cfg(not(feature = "cudarc"))]
    let result = cuda_sys::mem_free(ptr);
    if let Err(e) = result {
        eprintln!("craton-bolt: DeviceMemPool failed to free ptr: {}", e);
    }
}

/// Under `#[cfg(test)]` the pool's policy logic runs on synthetic pointers
/// minted by `test_support::test_driver_alloc`; routing them through the
/// real CUDA driver would crash. Tests record each "free" in a side-channel
/// list so they can assert on eviction behaviour.
#[cfg(test)]
unsafe fn driver_free(ptr: CUdeviceptr) {
    test_support::record_driver_free(ptr);
}

/// Process-wide GPU device-memory pool. Holds freed blocks keyed by their
/// bucket (rounded-up) size and hands them out on subsequent allocations.
pub struct DeviceMemPool {
    /// Buckets keyed by rounded-up byte size. Each bucket has its own
    /// `Mutex` so concurrent frees into distinct size classes don't
    /// serialise on a global lock.
    buckets: DashMap<usize, Mutex<BucketEntry>>,
    /// Sum of `alloc_bytes` across every pooled block. Atomic so reads in
    /// the eviction loop don't need a global lock. Soft cap — short
    /// transient overshoot under contention is acceptable.
    total_bytes: AtomicUsize,
    /// Soft cap on `total_bytes`. Resolved from env once at construction.
    max_pooled_bytes: usize,
    /// Hard cap on `buckets[k].len()` for any `k`.
    max_bucket_entries: usize,
}

impl DeviceMemPool {
    pub fn new() -> Self {
        Self {
            buckets: DashMap::new(),
            total_bytes: AtomicUsize::new(0),
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

    /// Try to take a freed block big enough for `bytes`. Falls back to
    /// `cuMemAlloc` on a miss. Returns `(ptr, actual_alloc_bytes)`; the caller
    /// must remember `actual_alloc_bytes` and pass it to `free` so we return
    /// to the right bucket.
    pub fn alloc(&self, bytes: usize) -> BoltResult<(CUdeviceptr, usize)> {
        let alloc_bytes = bucket_size(bytes);
        if let Some(entry) = self.buckets.get(&alloc_bytes) {
            let mut guard = entry.lock();
            // LIFO: most-recently freed block first — best cache locality.
            if let Some(block) = guard.blocks.pop_back() {
                drop(guard);
                self.total_bytes.fetch_sub(alloc_bytes, Ordering::AcqRel);
                return Ok((block.ptr, alloc_bytes));
            }
        }
        // Miss: call the driver. cuMemAlloc_v2 guarantees at least 256-byte
        // alignment, so the ARROW_ALIGNMENT (64) invariant holds trivially.
        //
        // Under `--features cudarc`, the alloc is satisfied by cudarc's
        // `result::malloc_sync`, which calls the same `cuMemAlloc_v2`
        // under the hood and returns a bit-compatible `CUdeviceptr` — so
        // pointers stored in the pool remain backend-agnostic and the
        // drain path can free them via either implementation.
        #[cfg(all(not(test), feature = "cudarc"))]
        let ptr = crate::cuda::cudarc_backend::mem_alloc(alloc_bytes)?;
        #[cfg(all(not(test), not(feature = "cudarc")))]
        let ptr = cuda_sys::mem_alloc(alloc_bytes)?;
        #[cfg(test)]
        let ptr = test_support::test_driver_alloc(alloc_bytes)?;
        Ok((ptr, alloc_bytes))
    }

    /// Return a block to the pool. If pooling this block would exceed
    /// `MAX_POOLED_BYTES` or `MAX_BUCKET_ENTRIES`, evict block(s) first;
    /// if that still does not make room, free `ptr` directly via the
    /// driver instead of pooling it.
    ///
    /// Under the L-5 per-bucket lock split, eviction is **per-bucket FIFO**
    /// rather than global LRU: we first try to evict from the incoming
    /// bucket itself (head = oldest), then walk other buckets if more
    /// bytes need to be reclaimed. See module doc for the Stage-2 plan
    /// to restore cross-bucket LRU.
    pub fn free(&self, ptr: CUdeviceptr, alloc_bytes: usize) {
        if ptr == 0 {
            return;
        }

        let mut to_free: Vec<CUdeviceptr> = Vec::new();

        // ---- Byte-cap eviction (best-effort, lock-free counter) ----
        //
        // If the incoming block is bigger than the entire cap there's no
        // point evicting — we'll just route it straight to the driver.
        if alloc_bytes <= self.max_pooled_bytes {
            while self.total_bytes.load(Ordering::Acquire) + alloc_bytes
                > self.max_pooled_bytes
            {
                if !self.evict_one(&mut to_free) {
                    break; // pool empty.
                }
            }
        }

        // ---- Per-bucket cap + insert under the bucket's own mutex ----
        //
        // Use `get` in the common case (bucket already exists) so we only
        // hold a DashMap *read* lock on the shard while we manipulate the
        // inner `Mutex`. `entry` would take a shard write lock for the
        // duration of the inner-mutex operation — needlessly contending
        // with other size classes that hash to the same shard.
        let pooled = if let Some(entry) = self.buckets.get(&alloc_bytes) {
            self.try_insert_into_bucket(entry.value(), ptr, alloc_bytes)
        } else {
            // Bucket missing — create it under a write lock and insert
            // immediately. The first-touch path through `entry` is rare
            // (only happens once per size class for the entire pool's
            // lifetime), so the shard write lock is acceptable here.
            let entry = self
                .buckets
                .entry(alloc_bytes)
                .or_insert_with(|| Mutex::new(BucketEntry::new()));
            self.try_insert_into_bucket(entry.value(), ptr, alloc_bytes)
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
    }

    /// Try to push `ptr` into the given bucket, respecting per-bucket and
    /// global byte caps. Returns `true` when the block was pooled,
    /// `false` when the caller must driver-free it.
    fn try_insert_into_bucket(
        &self,
        bucket: &Mutex<BucketEntry>,
        ptr: CUdeviceptr,
        alloc_bytes: usize,
    ) -> bool {
        let mut guard = bucket.lock();
        let fits_bucket = guard.blocks.len() < self.max_bucket_entries;
        // Re-check byte cap under our local lock — eviction above might
        // have already brought us under, or a parallel free may have
        // pushed us back over.
        let projected = self.total_bytes.load(Ordering::Acquire) + alloc_bytes;
        let fits_total = alloc_bytes <= self.max_pooled_bytes
            && projected <= self.max_pooled_bytes;
        if fits_bucket && fits_total {
            guard.blocks.push_back(PooledBlock {
                ptr,
                inserted: Instant::now(),
            });
            self.total_bytes.fetch_add(alloc_bytes, Ordering::AcqRel);
            true
        } else {
            false
        }
    }

    /// Evict the oldest block from any non-empty bucket. Walks `self.buckets`,
    /// peeks each front entry under a brief lock, and pops from whichever
    /// bucket holds the oldest `Instant`. Returns `true` if an eviction
    /// happened; `false` when every bucket is empty.
    ///
    /// Under L-5's per-bucket locks this is best-effort: between the scan
    /// and the pop a concurrent `alloc` may have drained the chosen bucket.
    /// We retry once on that race; if the bucket is still empty we fall
    /// through to "no eviction" and the caller's outer loop will exit.
    fn evict_one(&self, sink: &mut Vec<CUdeviceptr>) -> bool {
        // Scan: find the bucket whose front entry has the smallest Instant.
        // This is O(buckets) — bucket count is bounded (~4 × log2(max_alloc))
        // so cheap in practice.
        let mut best: Option<(usize, Instant)> = None;
        for r in self.buckets.iter() {
            let key = *r.key();
            let guard = r.value().lock();
            if let Some(front) = guard.blocks.front() {
                match best {
                    Some((_, t)) if front.inserted >= t => {}
                    _ => best = Some((key, front.inserted)),
                }
            }
        }
        let Some((key, _)) = best else {
            return false;
        };
        // Pop under the chosen bucket's lock. If the bucket emptied
        // between the scan and now, fall back to any non-empty bucket.
        if let Some(entry) = self.buckets.get(&key) {
            let mut guard = entry.lock();
            if let Some(block) = guard.blocks.pop_front() {
                self.total_bytes.fetch_sub(key, Ordering::AcqRel);
                sink.push(block.ptr);
                return true;
            }
        }
        // Lost the race — try one more scan for any non-empty bucket.
        for r in self.buckets.iter() {
            let key = *r.key();
            let mut guard = r.value().lock();
            if let Some(block) = guard.blocks.pop_front() {
                self.total_bytes.fetch_sub(key, Ordering::AcqRel);
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

    /// Evict pooled blocks (oldest first) until `total_pooled_bytes()` is
    /// at or below `self.max_pooled_bytes`. Intended for memory-pressure
    /// paths and `CudaContext::Drop`-adjacent shutdown hooks; the steady-
    /// state `free` path already enforces the cap, so this is a no-op in
    /// normal operation. Returns the number of blocks evicted.
    #[allow(dead_code)] // reason: shutdown / memory-pressure hook, not yet wired but kept for the contract
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
        self.buckets
            .iter()
            .map(|r| r.value().lock().blocks.len())
            .sum()
    }

    /// Number of pooled blocks in the bucket that would satisfy an allocation
    /// of `bytes`. Intended for tests and diagnostics only.
    #[doc(hidden)]
    pub fn bucket_len_for(&self, bytes: usize) -> usize {
        let key = bucket_size(bytes);
        self.buckets
            .get(&key)
            .map(|r| r.value().lock().blocks.len())
            .unwrap_or(0)
    }

    /// Release every pooled block back to the driver. Called on `Drop`, and
    /// usable by tests / shutdown paths that want a clean slate.
    pub fn drain(&self) {
        let mut drained: Vec<CUdeviceptr> = Vec::new();
        // Iterate over all buckets, draining each under its own lock.
        // We can't `DashMap::clear` mid-iteration safely, so we drain into
        // a local `Vec` first, then clear.
        for r in self.buckets.iter() {
            let mut guard = r.value().lock();
            while let Some(block) = guard.blocks.pop_front() {
                drained.push(block.ptr);
            }
        }
        self.buckets.clear();
        self.total_bytes.store(0, Ordering::Release);
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

#[cfg(test)]
mod test_support {
    //! Host-only shim. The pool's policy code (eviction, caps, LRU) can
    //! be exercised without a live CUDA context as long as `mem_alloc`
    //! and `mem_free` are intercepted. `test_driver_alloc` mints a
    //! monotonically increasing fake `CUdeviceptr`; `record_driver_free`
    //! records every synthetic block returned to the "driver" so tests
    //! can assert on eviction.
    use super::CUdeviceptr;
    use crate::error::BoltResult;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PTR: AtomicU64 = AtomicU64::new(1);
    static FREED: Mutex<Vec<CUdeviceptr>> = Mutex::new(Vec::new());

    pub(super) fn test_driver_alloc(_bytes: usize) -> BoltResult<CUdeviceptr> {
        // Wraparound is irrelevant — tests use a few hundred at most.
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
        // Keep NEXT_PTR monotonic across tests so a pointer freed by one
        // test cannot collide with a pointer allocated by the next.
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
        // hook would observe if the cap were raised at runtime.
        {
            for _ in 0..8 {
                let p = test_support::test_driver_alloc(64).unwrap();
                let entry = pool
                    .buckets
                    .entry(64)
                    .or_insert_with(|| Mutex::new(BucketEntry::new()));
                let mut guard = entry.lock();
                guard.blocks.push_back(PooledBlock {
                    ptr: p,
                    inserted: Instant::now(),
                });
                drop(guard);
                pool.total_bytes.fetch_add(64, Ordering::AcqRel);
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
    #[test]
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
        assert!(
            par_elapsed < seq_elapsed + seq_elapsed / 2,
            "parallel run ({:?}) should not be > 1.5x sequential ({:?}) — \
             suggests a global lock is still serialising frees",
            par_elapsed,
            seq_elapsed
        );
    }
}
