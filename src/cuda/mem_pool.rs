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
//! The pool depends on a live CUDA context being current on the calling
//! thread — same precondition as the bare `cuMemAlloc` path it replaces.

use std::collections::HashMap;

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
use crate::error::JavelinResult;

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

/// Process-wide GPU device-memory pool. Holds freed blocks keyed by their
/// bucket (rounded-up) size and hands them out on subsequent allocations.
pub struct DeviceMemPool {
    /// Buckets keyed by rounded-up byte size. Each bucket holds freed ptrs.
    buckets: Mutex<HashMap<usize, Vec<CUdeviceptr>>>,
}

impl DeviceMemPool {
    pub fn new() -> Self {
        Self {
            buckets: Mutex::new(HashMap::new()),
        }
    }

    /// Try to take a freed block big enough for `bytes`. Falls back to
    /// `cuMemAlloc` on a miss. Returns `(ptr, actual_alloc_bytes)`; the caller
    /// must remember `actual_alloc_bytes` and pass it to `free` so we return
    /// to the right bucket.
    pub fn alloc(&self, bytes: usize) -> JavelinResult<(CUdeviceptr, usize)> {
        let alloc_bytes = bucket_size(bytes);
        {
            let mut buckets = self.buckets.lock();
            if let Some(bucket) = buckets.get_mut(&alloc_bytes) {
                if let Some(ptr) = bucket.pop() {
                    return Ok((ptr, alloc_bytes));
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
        #[cfg(feature = "cudarc")]
        let ptr = crate::cuda::cudarc_backend::mem_alloc(alloc_bytes)?;
        #[cfg(not(feature = "cudarc"))]
        let ptr = cuda_sys::mem_alloc(alloc_bytes)?;
        Ok((ptr, alloc_bytes))
    }

    /// Return a block to the pool. Does NOT call `cuMemFree`.
    pub fn free(&self, ptr: CUdeviceptr, alloc_bytes: usize) {
        if ptr == 0 {
            return;
        }
        let mut buckets = self.buckets.lock();
        buckets.entry(alloc_bytes).or_default().push(ptr);
    }

    /// Release every pooled block back to the driver. Called on `Drop`, and
    /// usable by tests / shutdown paths that want a clean slate.
    pub fn drain(&self) {
        let drained: Vec<CUdeviceptr> = {
            let mut buckets = self.buckets.lock();
            let mut out = Vec::new();
            for (_, mut ptrs) in buckets.drain() {
                out.append(&mut ptrs);
            }
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
            unsafe {
                #[cfg(feature = "cudarc")]
                let result = crate::cuda::cudarc_backend::mem_free(ptr);
                #[cfg(not(feature = "cudarc"))]
                let result = cuda_sys::mem_free(ptr);
                if let Err(e) = result {
                    eprintln!("javelin: DeviceMemPool drain failed to free ptr: {}", e);
                }
            }
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
