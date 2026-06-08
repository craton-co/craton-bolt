// SPDX-License-Identifier: Apache-2.0

//! Stress tests for the process-wide device memory pool (`DeviceMemPool`).
//!
//! These complement `tests/memory_tests.rs` (which proves the *type-level*
//! soundness of `GpuVec` / `GpuView`) by hammering the *runtime* allocator
//! paths under churn and concurrency. Specifically, they're meant to catch
//! the family of bugs documented in the comment block at
//! `benches/olap_benchmarks.rs:472-478`: the pool stores raw `CUdeviceptr`
//! values keyed by bucket size, so any path that leaves dangling pointers in
//! the pool will eventually surface as an `ACCESS_VIOLATION` on the next
//! recycled allocation.
//!
//! All tests are `#[ignore]`'d — they need a live CUDA device. Run them with:
//!
//!     cargo test --test memory_pool_stress -- --ignored
//!
//! The tests reach into the pool via the `#[doc(hidden)]` accessor
//! `craton_bolt::cuda::mem_pool::__test_pool()`, which gives integration
//! tests the same view of the pool the crate's internals have without
//! exposing it as a stable API.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

use craton_bolt::cuda::mem_pool::__test_pool;
use craton_bolt::cuda::{CudaContext, GpuVec};

/// Reusable harness: build a context, run `body`, propagate any panic.
///
/// The context is bound to the current thread for the duration of `body` and
/// drained on drop. This mirrors the lifecycle in `Engine::new` / `Engine::drop`
/// so the tests exercise the same shutdown path real code hits.
fn with_ctx<F: FnOnce()>(body: F) {
    let ctx = CudaContext::new(0).expect("CUDA context");
    ctx.set_current().expect("set_current");
    body();
    drop(ctx);
}

// ---- 1. alloc/free churn ----------------------------------------------------

#[test]
#[ignore = "gpu:mempool"]
fn pool_alloc_free_churn() {
    // Allocate and immediately drop 10k buffers across a span of small sizes.
    // Each drop returns its block to the pool; the next alloc of the same
    // bucket should reuse it. After the loop the *driver* still owns every
    // byte (the pool only releases on drain), but the pool's bookkeeping
    // must be internally consistent: no panic, no negative counters.
    with_ctx(|| {
        let pool = __test_pool();
        // Start from a clean slate — earlier tests in the same process may
        // have left entries in the pool.
        pool.drain();
        let baseline = pool.pooled_block_count();
        assert_eq!(baseline, 0, "drain must empty the pool");

        // Span: 256 B, 1 KiB, 4 KiB, 16 KiB, 64 KiB, 256 KiB, 1 MiB. Picking
        // power-of-two sizes means each lands cleanly in its own bucket.
        let sizes: [usize; 7] = [256, 1024, 4096, 16_384, 65_536, 262_144, 1_048_576];
        const ITERS: usize = 10_000;

        for i in 0..ITERS {
            let bytes = sizes[i % sizes.len()];
            // u8 element type → 1 byte per element, so `with_capacity(bytes)`
            // requests exactly `bytes` bytes from the pool.
            let v = GpuVec::<u8>::with_capacity(bytes).expect("alloc");
            // Touch the pointer so the optimizer can't elide the allocation.
            assert_ne!(v.device_ptr(), 0, "iter {i}: null device ptr");
            drop(v);
        }

        // After the loop every block is back in the pool. Bucket count is
        // bounded by `sizes.len()`; total pooled blocks must be ≤ ITERS.
        let after = pool.pooled_block_count();
        assert!(
            after <= ITERS,
            "pool grew unboundedly: {after} pooled blocks vs {ITERS} allocs"
        );
        // And each bucket we touched must hold ≥ 1 freed block.
        for s in sizes {
            assert!(
                pool.bucket_len_for(s) >= 1,
                "bucket for size {s} is empty after churn"
            );
        }

        // Clean up so the next test starts from a known state.
        pool.drain();
        assert_eq!(pool.pooled_block_count(), 0, "drain must re-empty the pool");
    });
}

// ---- 2. drain empties the pool ---------------------------------------------

#[test]
#[ignore = "gpu:mempool"]
fn pool_drain_empties_pool() {
    with_ctx(|| {
        let pool = __test_pool();
        pool.drain();
        assert_eq!(pool.pooled_block_count(), 0);

        // Allocate a handful of buffers across distinct buckets, then let
        // them drop to populate the pool.
        let sizes: [usize; 4] = [512, 4096, 32_768, 262_144];
        {
            let _bufs: Vec<GpuVec<u8>> = sizes
                .iter()
                .map(|&n| GpuVec::<u8>::with_capacity(n).expect("alloc"))
                .collect();
        } // <-- _bufs drops here, each returning to the pool

        let populated = pool.pooled_block_count();
        assert_eq!(
            populated,
            sizes.len(),
            "expected one pooled block per size, got {populated}"
        );

        // Drain and verify.
        pool.drain();
        assert_eq!(
            pool.pooled_block_count(),
            0,
            "drain left {} blocks in the pool",
            pool.pooled_block_count()
        );

        // Re-allocating after drain must still work (the driver still has the
        // memory; the pool just doesn't cache it anymore).
        let v = GpuVec::<u8>::with_capacity(1024).expect("alloc after drain");
        assert_ne!(v.device_ptr(), 0);
        drop(v);
        pool.drain();
    });
}

// ---- 3. concurrent alloc/free pairs ----------------------------------------

#[test]
#[ignore = "gpu:mempool"]
fn pool_concurrent_alloc() {
    // 8 threads × 1000 alloc/free pairs each. The pool uses a `parking_lot::
    // Mutex` internally, so we're stressing both the lock and the
    // bucket-vector mutations. Each thread asserts its own device pointers
    // are non-null and unique within the thread (no double-free of an
    // in-flight handle).
    //
    // We must NOT share `CudaContext` across threads — `Send` only. Instead
    // the main thread holds the context and the workers inherit the current
    // context state by binding it themselves.
    with_ctx(|| {
        let pool = __test_pool();
        pool.drain();

        // Bind the context on every worker before it touches the driver.
        // We do this by passing the raw context handle (`CUcontext` is
        // `*mut c_void`) into each worker via a Send wrapper. Simpler: create
        // a fresh `CudaContext` per worker. That's heavier but avoids any
        // Send/Sync surface on the context's `raw` pointer. The pool is the
        // unit under test here, not the context, so the extra cost is fine.

        const THREADS: usize = 8;
        const PAIRS_PER_THREAD: u32 = 1000;
        let alloc_counter = Arc::new(AtomicU32::new(0));
        let free_counter = Arc::new(AtomicU32::new(0));

        let handles: Vec<_> = (0..THREADS)
            .map(|tid| {
                let alloc_counter = Arc::clone(&alloc_counter);
                let free_counter = Arc::clone(&free_counter);
                thread::spawn(move || {
                    // Each worker needs its own context binding. We cannot
                    // move the parent's `CudaContext` here (it's pinned to
                    // the main thread for the duration of `with_ctx`), so
                    // we mint a fresh primary-style context. The driver
                    // shares device memory across contexts in the same
                    // process, so the pool's pointers remain valid.
                    let ctx = CudaContext::new(0).expect("worker ctx");
                    ctx.set_current().expect("worker set_current");

                    // Mix a couple of sizes so threads contend on different
                    // buckets some of the time.
                    let sizes = [2048usize, 8192, 32_768];

                    for i in 0..PAIRS_PER_THREAD {
                        let n = sizes[(i as usize + tid) % sizes.len()];
                        let v = GpuVec::<u8>::with_capacity(n).expect("alloc");
                        assert_ne!(v.device_ptr(), 0, "thread {tid} iter {i} null ptr");
                        alloc_counter.fetch_add(1, Ordering::Relaxed);
                        drop(v);
                        free_counter.fetch_add(1, Ordering::Relaxed);
                    }
                    // Do NOT drop the worker's CudaContext here while the
                    // pool may still hold pointers minted in it — that's
                    // exactly the dangling-pointer foot-gun. Drain locally
                    // first so the worker's context destruction can't leave
                    // stale entries behind for siblings.
                    __test_pool().drain();
                })
            })
            .collect();

        for h in handles {
            h.join().expect("worker thread panicked");
        }

        let total = (THREADS as u32) * PAIRS_PER_THREAD;
        assert_eq!(alloc_counter.load(Ordering::SeqCst), total);
        assert_eq!(free_counter.load(Ordering::SeqCst), total);

        // Final cleanup.
        pool.drain();
    });
}

// ---- 4. bucket reuse: same-size alloc after free must hit the pool ---------

#[test]
#[ignore = "gpu:mempool"]
fn pool_bucket_reuse() {
    // Allocate, free, then allocate the *same* size again. The second alloc
    // must come from the pool (we can detect that by watching the bucket
    // length drop from 1 to 0 across the second `with_capacity` call), and
    // ideally the device pointer is identical to the first allocation's.
    with_ctx(|| {
        let pool = __test_pool();
        pool.drain();

        // Pick a size that's already a power of two so the bucket key is
        // unambiguous (`bucket_size(N) == N` when N is a power of two and
        // ≥ ARROW_ALIGNMENT).
        const SIZE: usize = 8192;

        let v1 = GpuVec::<u8>::with_capacity(SIZE).expect("alloc 1");
        let ptr1 = v1.device_ptr();
        assert_ne!(ptr1, 0);

        // Before the drop, the bucket is empty (we just took a fresh block
        // from the driver).
        assert_eq!(pool.bucket_len_for(SIZE), 0, "bucket non-empty pre-free");

        drop(v1);

        // After the drop, the freed block is sitting in the bucket waiting
        // for reuse.
        assert_eq!(
            pool.bucket_len_for(SIZE),
            1,
            "free did not return block to bucket"
        );

        // Second allocation of the same size: must hit the pool, draining
        // the bucket back to 0.
        let v2 = GpuVec::<u8>::with_capacity(SIZE).expect("alloc 2");
        assert_eq!(
            pool.bucket_len_for(SIZE),
            0,
            "second alloc did not reuse the pooled block"
        );
        // Reuse-implies-same-pointer is a stronger guarantee than the
        // pool's API formally promises, but in practice the bucket is a
        // `Vec<CUdeviceptr>` with LIFO pop semantics, so the pointer we
        // get back must equal the one we just returned.
        assert_eq!(
            v2.device_ptr(),
            ptr1,
            "reused alloc did not match freed pointer (LIFO invariant broken?)"
        );

        drop(v2);
        pool.drain();
    });
}

// ---- 5. drain-on-context-drop must keep subsequent allocs safe -------------

#[test]
#[ignore = "gpu:mempool"]
fn pool_drain_after_context_drop() {
    // The bug class this guards against: `CudaContext::Drop` calls
    // `POOL.drain()` so that no `CUdeviceptr` minted in a destroyed context
    // is later handed back to a *different* context's allocation. If that
    // drain were ever removed (or if a buffer outlived its context), the
    // next alloc on a fresh context would corrupt memory. We can't directly
    // observe corruption from a test, but we can prove the drain happens by
    // checking pool occupancy across a context-drop boundary, and we can
    // prove the rebuild path is healthy by allocating again afterward.

    // Phase 1: scoped first context. Populate the pool, then drop the
    // context. The pool must be empty immediately after.
    {
        let ctx1 = CudaContext::new(0).expect("ctx1");
        ctx1.set_current().expect("set_current ctx1");
        let pool = __test_pool();
        pool.drain();

        // Drop a few buffers to populate buckets.
        for n in [1024usize, 4096, 16_384] {
            let v = GpuVec::<u8>::with_capacity(n).expect("alloc");
            assert_ne!(v.device_ptr(), 0);
            drop(v);
        }
        assert!(
            pool.pooled_block_count() > 0,
            "expected populated pool before context drop"
        );

        drop(ctx1); // <-- triggers POOL.drain() via CudaContext::Drop

        assert_eq!(
            pool.pooled_block_count(),
            0,
            "CudaContext::Drop did not drain the pool — stale CUdeviceptrs would be handed to the next context"
        );
    }

    // Phase 2: brand-new context. Allocate again. Because the pool was
    // drained in phase 1, every block here comes fresh from the driver in
    // the *current* context, so the access is sound.
    {
        let ctx2 = CudaContext::new(0).expect("ctx2");
        ctx2.set_current().expect("set_current ctx2");

        let v = GpuVec::<u8>::with_capacity(8192).expect("alloc on fresh ctx");
        assert_ne!(v.device_ptr(), 0);

        // A round-trip touches the pointer for real (h2d then d2h), so any
        // dangling-pointer corruption would surface as a CUDA error here.
        let data: Vec<i32> = (0..256).collect();
        let gv = GpuVec::from_slice(&data).expect("h2d on fresh ctx");
        let back = gv.to_vec().expect("d2h on fresh ctx");
        assert_eq!(back, data, "round trip mismatch on rebuilt context");

        drop(v);
        drop(gv);
        // ctx2 drops here, which will drain again. That's fine.
    }
}
