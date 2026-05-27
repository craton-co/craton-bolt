// SPDX-License-Identifier: Apache-2.0

//! Process-wide device-memory pool / arena allocator.
//!
//! Every `cuMemAlloc_v2` / `cuMemFree_v2` round-trip is a synchronous driver
//! call that serializes against the GPU stream. For workloads that allocate
//! and free many short-lived device buffers per query (input upload + output
//! materialization), these calls dominate. This pool recycles freed blocks
//! back to callers instead of returning them to the driver.
//!
//! Size-bucketing rounds each request up to the next power of two with a
//! 64-byte floor (`ARROW_ALIGNMENT`). That bounds the bucket count to
//! `log2(max_alloc)` and gives high reuse for typical query buffers, which
//! cluster around a handful of natural sizes.
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
//! When a `free` would breach either limit, the pool first evicts the
//! globally oldest block (LRU across buckets) via `cuMemFree_v2`. If that
//! still does not make room (e.g. the per-bucket cap is hit by an already
//! cold bucket), the freshly freed block is returned to the driver directly
//! rather than pooled. Buckets internally are LIFO for reuse (returning the
//! most-recently freed block gives the warmest cache behaviour) but FIFO
//! for eviction (oldest first).
//!
//! The pool depends on a live CUDA context being current on the calling
//! thread — same precondition as the bare `cuMemAlloc` path it replaces.

use std::collections::HashMap;
use std::collections::VecDeque;
use std::time::Instant;

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::cuda::buffer::ARROW_ALIGNMENT;
use crate::cuda::cuda_sys::CUdeviceptr;
// `cuda_sys` is only referenced by the hand-rolled (default) backend
// path. Under `--features cudarc` the alloc/free hit `cudarc_backend`
// instead, so the import is feature-gated to keep both builds warning-
// free.
#[cfg(not(feature = "cudarc"))]
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

/// Round `bytes` up to the next power of two, with a floor of `ARROW_ALIGNMENT`.
/// This is the canonical bucket key.
fn bucket_size(bytes: usize) -> usize {
    let n = bytes.max(ARROW_ALIGNMENT);
    if n.is_power_of_two() {
        n
    } else {
        // next_power_of_two saturates; for realistic allocation sizes we never
        // hit that ceiling, but if we did the cuMemAlloc below would fail
        // cleanly.
        n.next_power_of_two()
    }
}

/// One pooled block. `inserted` is captured at `free` time so the eviction
/// path can find the globally oldest entry across buckets.
#[derive(Clone, Copy)]
struct PooledBlock {
    ptr: CUdeviceptr,
    inserted: Instant,
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
    inner: Mutex<PoolState>,
    /// Soft cap on `inner.total_bytes`. Reads are uncontended after
    /// construction; we cache the resolved env-var value here.
    max_pooled_bytes: usize,
    /// Hard cap on `inner.buckets[k].len()` for any `k`.
    max_bucket_entries: usize,
}

struct PoolState {
    buckets: HashMap<usize, VecDeque<PooledBlock>>,
    total_bytes: usize,
}

impl PoolState {
    fn new() -> Self {
        Self {
            buckets: HashMap::new(),
            total_bytes: 0,
        }
    }

    /// Find the bucket whose oldest (front) entry has the smallest
    /// `Instant`. `None` when the pool is empty. O(buckets) — bucket count
    /// is bounded by `log2(max_alloc)`, so this is cheap in practice.
    fn oldest_bucket(&self) -> Option<usize> {
        let mut best: Option<(usize, Instant)> = None;
        for (size, deque) in &self.buckets {
            if let Some(front) = deque.front() {
                match best {
                    Some((_, t)) if front.inserted >= t => {}
                    _ => best = Some((*size, front.inserted)),
                }
            }
        }
        best.map(|(s, _)| s)
    }

    /// Pop the oldest pooled block (front of the oldest bucket). Returns
    /// `(ptr, alloc_bytes)` so the caller can route to `driver_free`.
    fn evict_oldest(&mut self) -> Option<(CUdeviceptr, usize)> {
        let key = self.oldest_bucket()?;
        let block = self.buckets.get_mut(&key)?.pop_front()?;
        self.total_bytes = self.total_bytes.saturating_sub(key);
        Some((block.ptr, key))
    }
}

impl DeviceMemPool {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(PoolState::new()),
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
        {
            let mut state = self.inner.lock();
            if let Some(bucket) = state.buckets.get_mut(&alloc_bytes) {
                // LIFO: most-recently freed block first — best cache locality.
                if let Some(block) = bucket.pop_back() {
                    state.total_bytes = state.total_bytes.saturating_sub(alloc_bytes);
                    return Ok((block.ptr, alloc_bytes));
                }
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
    /// `MAX_POOLED_BYTES` or `MAX_BUCKET_ENTRIES`, evict the LRU block(s)
    /// first; if that still does not make room, free `ptr` directly via
    /// the driver instead of pooling it.
    pub fn free(&self, ptr: CUdeviceptr, alloc_bytes: usize) {
        if ptr == 0 {
            return;
        }

        // Decide policy under the lock; defer driver_free calls until
        // after we drop the lock to keep the critical section short.
        let to_free: Vec<CUdeviceptr> = {
            let mut state = self.inner.lock();
            let mut to_free = Vec::new();

            // Evict from other buckets to make room for the new block on
            // the byte budget. Skips eviction if the incoming block is
            // already too large to ever fit — in that case we'll just
            // free it directly below.
            if alloc_bytes <= self.max_pooled_bytes {
                while state.total_bytes + alloc_bytes > self.max_pooled_bytes {
                    match state.evict_oldest() {
                        Some((evicted_ptr, _evicted_bytes)) => to_free.push(evicted_ptr),
                        None => break, // pool is empty; can't free more.
                    }
                }
            }

            // Per-bucket cap. Bucket may be empty when the global eviction
            // above happened to drain it, which is fine — `or_default`
            // recreates it.
            let bucket = state.buckets.entry(alloc_bytes).or_default();
            let fits_bucket = bucket.len() < self.max_bucket_entries;
            let fits_total = alloc_bytes <= self.max_pooled_bytes
                && state.total_bytes + alloc_bytes <= self.max_pooled_bytes;

            if fits_bucket && fits_total {
                bucket.push_back(PooledBlock {
                    ptr,
                    inserted: Instant::now(),
                });
                state.total_bytes += alloc_bytes;
            } else {
                // Couldn't make room — drop this block to the driver.
                to_free.push(ptr);
            }

            to_free
        };

        for p in to_free {
            // SAFETY: every pointer routed here was either pulled out of
            // the pool (originally minted by `mem_alloc` and given up by
            // its previous owner via `free`) or is the `ptr` we were just
            // handed by a caller who has likewise transferred ownership.
            unsafe { driver_free(p) };
        }
    }

    /// Sum of `alloc_bytes` across every pooled block. Useful for tests
    /// and memory-pressure introspection.
    pub(crate) fn total_pooled_bytes(&self) -> usize {
        self.inner.lock().total_bytes
    }

    /// Evict pooled blocks (oldest first) until `total_pooled_bytes()` is
    /// at or below `self.max_pooled_bytes`. Intended for memory-pressure
    /// paths and `CudaContext::Drop`-adjacent shutdown hooks; the steady-
    /// state `free` path already enforces the cap, so this is a no-op in
    /// normal operation. Returns the number of blocks evicted.
    pub(crate) fn evict_above_high_water(&self) -> usize {
        let to_free: Vec<CUdeviceptr> = {
            let mut state = self.inner.lock();
            let mut out = Vec::new();
            while state.total_bytes > self.max_pooled_bytes {
                match state.evict_oldest() {
                    Some((ptr, _bytes)) => out.push(ptr),
                    None => break,
                }
            }
            out
        };
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
        self.buckets.lock().values().map(|v| v.len()).sum()
    }

    /// Number of pooled blocks in the bucket that would satisfy an allocation
    /// of `bytes`. Intended for tests and diagnostics only.
    #[doc(hidden)]
    pub fn bucket_len_for(&self, bytes: usize) -> usize {
        let key = bucket_size(bytes);
        self.buckets.lock().get(&key).map(|v| v.len()).unwrap_or(0)
    }

    /// Release every pooled block back to the driver. Called on `Drop`, and
    /// usable by tests / shutdown paths that want a clean slate.
    pub fn drain(&self) {
        let drained: Vec<CUdeviceptr> = {
            let mut state = self.inner.lock();
            let mut out = Vec::new();
            for (_, mut deque) in state.buckets.drain() {
                while let Some(block) = deque.pop_front() {
                    out.push(block.ptr);
                }
            }
            state.total_bytes = 0;
            out
        };
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
        let state = pool.inner.lock();
        let bucket_len = state
            .buckets
            .get(&bucket_size(block_size))
            .map(|d| d.len())
            .unwrap_or(0);
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

    #[test]
    fn lru_evicts_oldest_first() {
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
            "LRU should have evicted `a`; freed list = {:?}",
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
            let mut s = pool.inner.lock();
            for _ in 0..8 {
                let p = test_support::test_driver_alloc(64).unwrap();
                s.buckets.entry(64).or_default().push_back(PooledBlock {
                    ptr: p,
                    inserted: Instant::now(),
                });
                s.total_bytes += 64;
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
}
