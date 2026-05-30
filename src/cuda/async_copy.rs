// SPDX-License-Identifier: Apache-2.0

//! Pinned-host-buffer + async H2D/D2H copy helpers (M1 perf foundation).
//!
//! This module is the on-ramp for overlapping PCIe transfers with kernel
//! execution. It provides:
//!
//! * [`PinnedBuffer<T>`] — a page-locked host allocation made via
//!   `cuMemHostAlloc` (the flagged sibling of `cuMemAllocHost_v2` used by
//!   [`crate::cuda::buffer::PinnedHostBuffer`]). Page-locked memory lets the
//!   driver DMA straight in / out of the host buffer instead of bouncing
//!   through an internal staging copy, which is what makes
//!   `cuMemcpyHtoDAsync_v2` / `cuMemcpyDtoHAsync_v2` actually overlap with
//!   compute. Under the `cuda-stub` feature there is no driver, so the
//!   buffer falls back to a plain heap [`Vec<T>`] — the host-side accounting
//!   (len / byte_len / `Deref`) behaves identically, only the allocation is
//!   pageable.
//!
//! * [`upload_async`] / [`download_async`] — thin, borrow-checked wrappers
//!   around the `*_Async_v2` memcpy entry points that move bytes between a
//!   [`PinnedBuffer<T>`] and a [`GpuView`] / [`GpuViewMut`]. They are NOT
//!   `async fn`s — "async" here refers to the CUDA stream semantics: the call
//!   returns once the copy is *issued*, and the caller must [`sync`] the
//!   stream before touching the destination.
//!
//! * [`sync`] — block until all prior work on a stream completes, surfacing
//!   the driver error as a [`BoltResult`].
//!
//! # Stream wrapper
//!
//! There is already a [`CudaStream`](crate::exec::launch::CudaStream) RAII
//! wrapper over `cuStreamCreate` / `cuStreamDestroy_v2` /
//! `cuStreamSynchronize` in `crate::exec::launch`, so this module reuses it
//! rather than minting a second one. The helpers below take a `&CudaStream`.
//!
//! # Lifetime / safety contract (read before using)
//!
//! Like the rest of the async surface in this crate, these helpers issue
//! stream-ordered work and return immediately. The host [`PinnedBuffer`] and
//! the device allocation behind the [`GpuView`]/[`GpuViewMut`] **must both**
//! outlive the stream's completion. Call [`sync`] (or
//! [`CudaStream::synchronize`](crate::exec::launch::CudaStream::synchronize))
//! before reading a download destination or before dropping either operand.
//! After enqueueing, call [`PinnedBuffer::mark_stream_use`] so the buffer's
//! `Drop` fences the right stream before `cuMemFreeHost` reclaims the pages —
//! mirroring the discipline documented on
//! [`crate::cuda::buffer::PinnedHostBuffer`].

use std::cell::RefCell;
// `Cell` is only used by the test-only `DROP_FENCE_OVERRIDE` thread-local
// (the production multi-stream tracking uses `RefCell<StreamSet>` after the
// single-stream `Cell<Option<CUstream>>` was removed); gate the import so a
// non-test build doesn't warn on an unused import. Mirrors `cuda::buffer`.
#[cfg(test)]
use std::cell::Cell;
use std::mem::size_of;
use std::ops::{Deref, DerefMut};

use bytemuck::Pod;

use crate::cuda::cuda_sys::{self, CUstream};
use crate::cuda::smart_ptrs::{GpuView, GpuViewMut};
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;

/// Deduplicated set of stream handles a [`PinnedBuffer`] has been enqueued on.
///
/// ## Why this is a local type, not [`crate::cuda::buffer::StreamSet`]
///
/// The sibling [`crate::cuda::buffer::PinnedHostBuffer`] fixes the identical
/// host-side multi-stream use-after-free with a `RefCell<StreamSet>` and we
/// would prefer to reuse that exact type. But while `buffer::StreamSet` is
/// `pub(crate)` (nameable here), its `insert` / `len` / `is_empty` methods and
/// its inner `streams` field are *module-private* to `buffer`, so a sibling
/// module cannot actually operate on one. Exposing them would mean editing
/// `buffer.rs`, which is out of scope for this fix. We therefore keep a
/// minimal, behaviourally-identical local accumulator (deduped `Vec<CUstream>`,
/// linear dedup — n is ~1–2 in practice, see the rationale on
/// `buffer::StreamSet`). If `buffer::StreamSet`'s API is ever made
/// `pub(crate)`, this type should be deleted in favour of importing it.
///
/// `CUstream` is a raw pointer handle; we never dereference it here, only
/// compare handles for equality and forward them to `cuStreamSynchronize`.
#[derive(Default)]
struct StreamSet {
    /// Distinct stream handles, in first-seen order. Never contains
    /// duplicates (enforced by [`StreamSet::insert`]).
    streams: Vec<CUstream>,
}

impl StreamSet {
    /// Record `stream` if not already present, so `Drop` issues at most one
    /// `cuStreamSynchronize` per distinct stream.
    #[inline]
    fn insert(&mut self, stream: CUstream) {
        if !self.streams.contains(&stream) {
            self.streams.push(stream);
        }
    }

    /// Number of distinct streams recorded. Test/diagnostic hook.
    #[inline]
    fn len(&self) -> usize {
        self.streams.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }
}

/// Owned page-locked (pinned) host buffer, allocated via `cuMemHostAlloc`.
///
/// Pinned host memory is what makes async H2D / D2H copies overlap with
/// kernel execution: the DMA engine reads / writes the host pages directly
/// rather than bouncing through a driver-internal staging copy. The trade-off
/// is that the OS cannot page these out, so allocate them for the *final*
/// upload sources / download targets, not every scratch buffer.
///
/// # Two backends, one type
///
/// * Default build: `ptr` points at memory from `cuMemHostAlloc`, freed by
///   `cuMemFreeHost` in `Drop`.
/// * `--features cuda-stub`: there is no driver, so the buffer is backed by a
///   plain heap [`Vec<T>`]. Every host-visible method (`len`, `byte_len`,
///   `as_slice`, `Deref`, …) behaves exactly the same; only the underlying
///   pages are pageable rather than pinned. This keeps the crate compiling and
///   unit-testable on hosts with no CUDA toolkit (and on docs.rs).
///
/// Borrow rules mirror `Vec<T>` / `[T]`: [`as_slice`](Self::as_slice) lends a
/// shared `&[T]`, [`as_mut_slice`](Self::as_mut_slice) an exclusive
/// `&mut [T]`, and the type also `Deref`s to `[T]`. `PinnedBuffer` is `Send`
/// (the allocation may move between threads) but **not** `Sync` — concurrent
/// mutation through shared references would race the same way `&mut [T]`
/// does, and would race in-flight DMA besides.
pub struct PinnedBuffer<T: Pod> {
    /// Live-build pinned host pointer (from `cuMemHostAlloc`). Null when
    /// `len == 0`. Absent under `cuda-stub` (the `Vec` owns the storage).
    #[cfg(not(feature = "cuda-stub"))]
    ptr: *mut T,
    /// Stub-build backing store. A regular heap allocation — pageable, but
    /// indistinguishable from the host's point of view.
    #[cfg(feature = "cuda-stub")]
    storage: Vec<T>,
    /// Logical element count.
    len: usize,
    /// Bytes the allocator handed back (`len * size_of::<T>()` at
    /// construction). Cached so `Drop` / `byte_len` don't recompute and so a
    /// future bucketed pinned pool can hook in cleanly.
    byte_len: usize,
    /// The set of streams this buffer's pinned pages have been enqueued on
    /// (via [`mark_stream_use`](Self::mark_stream_use)). `Drop` fences **every**
    /// recorded stream before freeing the pinned pages so an in-flight
    /// `cuMemcpy*Async` on any of them cannot DMA into reclaimed memory.
    ///
    /// ## Why a *set*, not a single "last stream"
    ///
    /// The previous design tracked only the most-recently-used stream
    /// (`Cell<Option<CUstream>>`) and fenced just that one at `Drop`. A pinned
    /// buffer uploaded on stream A then used on stream B fenced only B, so an
    /// independent transfer still draining on A could read/write the
    /// page-locked region *after* `cuMemFreeHost` handed the pages back — a
    /// host-side use-after-free the DMA engine has no way to detect. This is
    /// the exact analogue of the multi-stream race the sibling
    /// [`crate::cuda::buffer::PinnedHostBuffer`] already fixed, so we apply the
    /// same accumulate-every-stream-and-fence-all-at-`Drop` pattern here.
    ///
    /// `RefCell<StreamSet>` (mirroring `PinnedHostBuffer::used_streams`) so the
    /// shared-borrow async helpers can tag the stream without forcing every
    /// call site onto `&mut self`; sound because the type is `!Sync`, so there
    /// is never a concurrent borrow from another thread.
    used_streams: RefCell<StreamSet>,
}

impl<T: Pod> PinnedBuffer<T> {
    /// Allocate `len` page-locked elements of `T` with default flags
    /// (`flags = 0`, equivalent to `cuMemAllocHost_v2`).
    ///
    /// `len == 0` is allowed and returns an empty buffer without touching the
    /// driver; [`as_slice`](Self::as_slice) then returns `&[]`.
    pub fn new(len: usize) -> BoltResult<Self> {
        Self::with_flags(len, 0)
    }

    /// Allocate `len` page-locked elements of `T`, forwarding `flags`
    /// (a bitwise-OR of the `CU_MEMHOSTALLOC_*` constants in
    /// [`crate::cuda::cuda_sys`]) to `cuMemHostAlloc`.
    ///
    /// Under `cuda-stub` the `flags` are accepted but ignored — the heap
    /// fallback has no notion of portable / write-combined pages.
    ///
    /// `len == 0` returns an empty buffer without an allocation.
    pub fn with_flags(len: usize, flags: u32) -> BoltResult<Self> {
        if len == 0 {
            return Ok(Self::empty_inner());
        }
        let byte_len = len
            .checked_mul(core::mem::size_of::<T>())
            .ok_or_else(|| BoltError::Memory("PinnedBuffer: byte length overflow".into()))?;

        #[cfg(not(feature = "cuda-stub"))]
        {
            // SAFETY: `byte_len > 0` (len > 0 and `size_of::<T>() >= 1` for any
            // `Pod`). `cuMemHostAlloc` returns a non-null, suitably-aligned
            // host pointer on success; we hand it to `cuMemFreeHost` in `Drop`.
            let raw = unsafe { cuda_sys::mem_host_alloc(byte_len, flags as libc::c_uint)? };
            Ok(Self {
                ptr: raw as *mut T,
                len,
                byte_len,
                used_streams: RefCell::new(StreamSet::default()),
            })
        }

        #[cfg(feature = "cuda-stub")]
        {
            // Stub fallback: a plain zero-initialized heap allocation. `flags`
            // is meaningless without a driver, so it is intentionally ignored.
            let _ = flags;
            // `T: Pod` => the all-zero bit pattern is a valid `T`
            // (`Pod: Zeroable`). `bytemuck::zeroed()` is the free-function
            // form so we don't need the `Zeroable` trait in scope, and
            // `Pod: Copy: Clone` satisfies the `vec!` repeat requirement.
            let storage: Vec<T> = vec![<T as bytemuck::Zeroable>::zeroed(); len];
            Ok(Self {
                storage,
                len,
                byte_len,
                used_streams: RefCell::new(StreamSet::default()),
            })
        }
    }

    /// Construct the empty (`len == 0`) buffer for the current backend.
    fn empty_inner() -> Self {
        Self {
            #[cfg(not(feature = "cuda-stub"))]
            ptr: std::ptr::null_mut(),
            #[cfg(feature = "cuda-stub")]
            storage: Vec::new(),
            len: 0,
            byte_len: 0,
            used_streams: RefCell::new(StreamSet::default()),
        }
    }

    /// Number of `T` elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Byte length (`len * size_of::<T>()`), computed from the *logical*
    /// element count so it tracks [`set_len`](Self::set_len) truncations.
    ///
    /// # Panics
    /// Panics if `len * size_of::<T>()` overflows `usize` — a bug we want to
    /// surface rather than a number to silently wrap. Cannot happen for a
    /// buffer produced by [`new`](Self::new) (the multiply was already
    /// checked at construction).
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len
            .checked_mul(size_of::<T>())
            .expect("PinnedBuffer::byte_len overflow")
    }

    /// Original pinned allocation size in bytes (the value the allocator
    /// returned at construction), independent of any later
    /// [`set_len`](Self::set_len) truncation.
    #[inline]
    pub fn capacity_bytes(&self) -> usize {
        self.byte_len
    }

    /// Raw host pointer for async-memcpy FFI. May be null when `len == 0`.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        #[cfg(not(feature = "cuda-stub"))]
        {
            self.ptr
        }
        #[cfg(feature = "cuda-stub")]
        {
            self.storage.as_ptr()
        }
    }

    /// Mutable raw host pointer. May be null when `len == 0`.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        #[cfg(not(feature = "cuda-stub"))]
        {
            self.ptr
        }
        #[cfg(feature = "cuda-stub")]
        {
            self.storage.as_mut_ptr()
        }
    }

    /// Borrow as a shared slice.
    ///
    /// Safe: `T: Pod` so any bit pattern is a valid value, and the buffer
    /// guarantees `len` readable elements.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            // Stable empty slice. Never `from_raw_parts(null, 0)` — UB even
            // for length 0.
            return &[];
        }
        #[cfg(not(feature = "cuda-stub"))]
        {
            // SAFETY: `self.ptr` is a valid host VA for `self.len` `T`s
            // (from `cuMemHostAlloc`), it outlives the borrow, and `T: Pod`
            // accepts any bit pattern.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
        #[cfg(feature = "cuda-stub")]
        {
            &self.storage[..self.len]
        }
    }

    /// Borrow as an exclusive slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            return <&mut [T]>::default();
        }
        #[cfg(not(feature = "cuda-stub"))]
        {
            // SAFETY: same as `as_slice`, plus the `&mut self` receiver
            // statically rules out aliasing.
            unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
        }
        #[cfg(feature = "cuda-stub")]
        {
            let n = self.len;
            &mut self.storage[..n]
        }
    }

    /// Record that `stream` has enqueued (or is about to enqueue) async work
    /// that references this buffer's host pages. `Drop` will
    /// `cuStreamSynchronize` against **every** recorded stream before
    /// `cuMemFreeHost` reclaims the pages, so the DMA engine can't be reading
    /// / writing freed memory on any stream that ever touched them.
    ///
    /// Mirrors the `mark_stream_use` contract on
    /// [`crate::cuda::buffer::PinnedHostBuffer`]. Every distinct stream is
    /// accumulated (not just the last one), so a pinned buffer used across
    /// multiple streams fences all of them at `Drop` — closing the host-side
    /// multi-stream use-after-free. Recording the same stream twice is a no-op
    /// (the set dedups), so over-calling is always safe.
    ///
    /// Takes `&self` (interior mutability via `RefCell`) so an async helper
    /// that only reads `as_ptr()` through a shared borrow doesn't have to be
    /// rewritten to `&mut self`. Sound because the type is `!Sync`, and the
    /// borrow is held only for the `insert` call so it cannot overlap the
    /// `borrow()` taken in `Drop`.
    // `stream` is an opaque CUstream handle that we merely store — no deref,
    // no FFI — so the outer fn stays safe.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    pub fn mark_stream_use(&self, stream: CUstream) {
        self.used_streams.borrow_mut().insert(stream);
    }

    /// Number of distinct streams currently recorded for this buffer.
    ///
    /// Test / diagnostic hook (mirrors
    /// [`crate::cuda::buffer::PinnedHostBuffer::recorded_stream_count`]): lets
    /// host-only tests assert the stream-set bookkeeping (dedup, multi-stream
    /// accumulation) without a GPU. Replaces the former single-stream
    /// `last_use_stream()` accessor, which no longer has a well-defined
    /// meaning now that the buffer tracks the full set.
    #[doc(hidden)]
    #[inline]
    pub fn recorded_stream_count(&self) -> usize {
        self.used_streams.borrow().len()
    }

    /// Override the logical length — used after an async D2H that filled fewer
    /// than the buffer's allocated length.
    ///
    /// # Safety
    /// The first `new_len` elements must be initialized (e.g. by a completed
    /// async D2H). The upper bound (`new_len * size_of::<T>() <=
    /// capacity_bytes()`) is enforced unconditionally by an `assert!` (release
    /// *and* debug) so an out-of-range `new_len` panics rather than exposing
    /// uninitialized memory through [`as_slice`](Self::as_slice).
    pub unsafe fn set_len(&mut self, new_len: usize) {
        let new_bytes = new_len
            .checked_mul(size_of::<T>())
            .expect("PinnedBuffer::set_len: new_len * size_of::<T>() overflowed usize");
        assert!(
            new_bytes <= self.byte_len,
            "PinnedBuffer::set_len: requested {new_len} elements ({new_bytes} bytes) \
             exceeds capacity {} bytes",
            self.byte_len
        );
        self.len = new_len;
    }
}

impl<T: Pod> Deref for PinnedBuffer<T> {
    type Target = [T];
    #[inline]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T: Pod> DerefMut for PinnedBuffer<T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

/// Signature of the per-stream fence used by [`PinnedBuffer`]'s `Drop`.
/// Aliased so a host-only test can swap in a recording stub via
/// [`drop_fence_with`], mirroring the same mockable seam on
/// [`crate::cuda::buffer::PinnedHostBuffer`] / `GpuBuffer`. Returns the raw
/// `CUresult` so the production hook can warn on failure.
///
/// Like [`fence_all_streams`], this is unreferenced in a non-test `cuda-stub`
/// build, so gate it to avoid a dead-code warning.
#[cfg(any(not(feature = "cuda-stub"), test))]
type StreamFenceFn = fn(CUstream) -> crate::cuda::cuda_sys::CUresult;

/// Production stream fence: forwards to `cuStreamSynchronize`.
///
/// SAFETY: `stream` is an opaque handle previously recorded via
/// `mark_stream_use` (exactly what the caller passed to the async FFI). It is
/// never dereferenced here, only handed to the driver, which tolerates a
/// synchronize-on-completed-stream as a cheap no-op.
#[cfg(not(feature = "cuda-stub"))]
#[allow(clippy::not_unsafe_ptr_arg_deref)] // handle forwarded, not deref'd
fn real_stream_fence(stream: CUstream) -> crate::cuda::cuda_sys::CUresult {
    unsafe { cuda_sys::cuStreamSynchronize(stream) }
}

/// Fence every stream in `streams` via `fence`, warning on any non-success rc.
/// Factored out of `Drop` so it can be unit-tested with a recording stub
/// (counting how many distinct streams get fenced) on a host with no GPU.
///
/// The set is already deduped by [`StreamSet::insert`], so each distinct
/// stream is fenced exactly once.
///
/// Compiled for the live build (where `Drop` calls it) and for any test build
/// (where the host-only fence tests call it through the recording stub). It is
/// unreferenced in a *non-test* `cuda-stub` build — there is no driver and the
/// stub `Drop` path never fences — so gate it to avoid a dead-code warning.
#[cfg(any(not(feature = "cuda-stub"), test))]
fn fence_all_streams(streams: &StreamSet, fence: StreamFenceFn) {
    for &stream in &streams.streams {
        let rc = fence(stream);
        if rc != cuda_sys::CUDA_SUCCESS {
            log::warn!(
                "craton-bolt: cuStreamSynchronize before PinnedBuffer free returned {} \
                 (buffer dropped while a pending DMA may still reference it; the pinned \
                 pages may be reclaimed before the driver is done with them)",
                rc
            );
        }
    }
}

/// Test seam: when set, `Drop` fences through this stub instead of the real
/// `cuStreamSynchronize`. Host-only tests install a recorder here to assert
/// the *number of distinct streams fenced* without a GPU, then clear it.
/// Mirrors `DROP_FENCE_OVERRIDE` in `crate::cuda::buffer`.
#[cfg(test)]
thread_local! {
    static DROP_FENCE_OVERRIDE: Cell<Option<StreamFenceFn>> = const { Cell::new(None) };
}

/// Install `fence` as the `Drop`-time stream fence for the current thread,
/// run `body`, then restore the previous hook. Test-only.
#[cfg(test)]
fn drop_fence_with<R>(fence: StreamFenceFn, body: impl FnOnce() -> R) -> R {
    let prev = DROP_FENCE_OVERRIDE.with(|c| c.replace(Some(fence)));
    let out = body();
    DROP_FENCE_OVERRIDE.with(|c| c.set(prev));
    out
}

/// Resolve the fence the current `Drop` should use: the test override if
/// installed, else the real `cuStreamSynchronize`. Under `cuda-stub` there is
/// no driver, so only the test override path exists.
#[cfg(not(feature = "cuda-stub"))]
#[inline]
fn current_drop_fence() -> StreamFenceFn {
    #[cfg(test)]
    {
        if let Some(f) = DROP_FENCE_OVERRIDE.with(|c| c.get()) {
            return f;
        }
    }
    real_stream_fence
}

impl<T: Pod> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        // Stub backend: the `Vec` frees itself; nothing to fence (no DMA can
        // be in flight without a driver). Recording a stream on a stub buffer
        // is meaningless, so we ignore `used_streams` here.
        #[cfg(feature = "cuda-stub")]
        {
            // Explicitly touch the field so a future refactor that drops it
            // trips here rather than silently. (`storage` drops on its own.)
            // Reading both accessors also keeps `StreamSet::is_empty` / `len`
            // exercised in a non-test stub build, so neither warns as dead.
            let set = self.used_streams.borrow();
            let _ = (set.is_empty(), set.len());
        }

        #[cfg(not(feature = "cuda-stub"))]
        {
            if self.ptr.is_null() {
                // Zero-length / never-allocated: nothing to free, nothing to
                // fence.
                return;
            }
            // Fence EVERY stream this buffer was enqueued on before returning
            // the pinned pages to the driver. Without this, an outstanding
            // `cuMemcpyHtoDAsync_v2` / `cuMemcpyDtoHAsync_v2` whose host operand
            // is `self.ptr` on ANY recorded stream would keep reading / writing
            // the page-locked region after `cuMemFreeHost` released it — a
            // host-side use-after-free the DMA engine has no way to detect.
            //
            // Fencing the whole set (not just the last stream) is what closes
            // the multi-stream race: a buffer uploaded on stream A then used on
            // stream B fenced only B under the old design, leaving an
            // independent transfer still draining on A to scribble the freed
            // pages. The set is deduped, so each stream is fenced at most once.
            //
            // The borrow is short-lived and cannot alias any other borrow:
            // `Drop` runs with exclusive ownership of `self`, so no `&self`
            // method (which is what takes `borrow_mut` in `mark_stream_use`)
            // can run concurrently.
            let streams = self.used_streams.borrow();
            if !streams.is_empty() {
                fence_all_streams(&streams, current_drop_fence());
            }
            drop(streams);
            // SAFETY: `self.ptr` came from `cuMemHostAlloc` and we have unique
            // ownership (move-only, `!Sync`). Cast through `c_void` so we
            // don't need `libc` in the `use` list just for this.
            let rc = unsafe { cuda_sys::mem_free_host(self.ptr as *mut std::ffi::c_void) };
            if let Err(e) = rc {
                log::warn!("craton-bolt: cuMemFreeHost failed ({e:?}); pinned host buffer leaked");
            }
        }
    }
}

// SAFETY: ownership of a pinned host buffer may move between threads — both
// `cuMemHostAlloc` pointers and a `Vec`'s heap allocation are readable from
// any thread. Cross-thread *sharing* without external sync is unsound (it
// would race in-flight DMA and the borrow checker), so we do NOT implement
// Sync. `*mut T` is `!Send` by default, hence the explicit opt-in.
unsafe impl<T: Pod> Send for PinnedBuffer<T> {}
// `!Sync` is implicit.

/// Asynchronously upload `src` (pinned host) into the device region behind
/// `dst` on `stream`. Returns once the copy is *issued*; call [`sync`] before
/// reading `dst` from another stream or the host.
///
/// `src.len()` must equal `dst.len()` — the copy is sized to the full view so
/// the caller can't accidentally leave the tail of `dst` stale. After issuing,
/// `src` is tagged via [`PinnedBuffer::mark_stream_use`] so its `Drop` fences
/// `stream`.
///
/// # Lifetime contract
/// Both `src` and the device allocation backing `dst` must remain live and
/// untouched until `stream` is synchronized. Synchronize before dropping
/// either, or before reading the destination. See the module docs.
pub fn upload_async<T: Pod>(
    stream: &CudaStream,
    dst: &mut GpuViewMut<'_, T>,
    src: &PinnedBuffer<T>,
) -> BoltResult<()> {
    if src.len() != dst.len() {
        return Err(BoltError::Memory(format!(
            "upload_async length mismatch: src(pinned)={}, dst(device)={}",
            src.len(),
            dst.len()
        )));
    }
    let stream_raw = stream.raw();
    if !src.is_empty() {
        // SAFETY: `src` is a valid host region of `src.len()` `T`s; `dst`'s
        // device pointer is a live allocation of the same element count
        // (lengths equal, checked above). The caller's lifetime contract
        // (sync before reuse / drop) is documented above.
        unsafe {
            cuda_sys::memcpy_h2d_async::<T>(dst.device_ptr(), src.as_ptr(), src.len(), stream_raw)?;
        }
    }
    // Tag the pinned source so its `Drop` fences the stream before freeing
    // the page-locked pages out from under an in-flight H2D.
    src.mark_stream_use(stream_raw);
    Ok(())
}

/// Asynchronously download the device region behind `src` into `dst` (pinned
/// host) on `stream`. Returns once the copy is *issued*; call [`sync`] before
/// reading `dst`.
///
/// `dst.len()` must equal `src.len()`. After issuing, `dst` is tagged via
/// [`PinnedBuffer::mark_stream_use`] so its `Drop` fences `stream`.
///
/// # Lifetime contract
/// Both `dst` and the device allocation backing `src` must remain live and
/// untouched until `stream` is synchronized; `dst` must not be read until
/// then. See the module docs.
pub fn download_async<T: Pod>(
    stream: &CudaStream,
    dst: &mut PinnedBuffer<T>,
    src: &GpuView<'_, T>,
) -> BoltResult<()> {
    if dst.len() != src.len() {
        return Err(BoltError::Memory(format!(
            "download_async length mismatch: dst(pinned)={}, src(device)={}",
            dst.len(),
            src.len()
        )));
    }
    let stream_raw = stream.raw();
    let n = src.len();
    // Tag before borrowing `dst` mutably for the pointer: `mark_stream_use`
    // takes `&self`, and we want the tag recorded even though the FFI uses an
    // exclusive borrow of the host pointer.
    dst.mark_stream_use(stream_raw);
    if n != 0 {
        let host_ptr = dst.as_mut_ptr();
        // SAFETY: `dst` is valid for writes of `n` `T`s (lengths equal,
        // checked above); `src`'s device pointer is a live allocation of the
        // same element count. The caller must sync `stream` before reading
        // `dst` (documented above).
        unsafe {
            cuda_sys::memcpy_d2h_async::<T>(host_ptr, src.device_ptr(), n, stream_raw)?;
        }
    }
    Ok(())
}

/// Block until all prior work enqueued on `stream` has completed, surfacing
/// any driver error. Thin convenience over
/// [`CudaStream::synchronize`](crate::exec::launch::CudaStream::synchronize)
/// so call sites importing this module don't also need to reach for the
/// stream type's inherent method.
#[inline]
pub fn sync(stream: &CudaStream) -> BoltResult<()> {
    stream.synchronize()
}

#[cfg(test)]
mod tests {
    //! Host-only tests. Nothing here may touch the CUDA driver: they run on
    //! machines without a GPU and on docs.rs. Under `cuda-stub` the
    //! `PinnedBuffer` is `Vec`-backed, so allocation succeeds and the
    //! sizing / `Deref` accounting is fully exercisable. The one test that
    //! needs a live driver is `#[ignore]`-gated under the crate's
    //! `BOLT_BENCH_GPU=1 + --ignored` convention.
    use super::*;

    // ---- sizing / len / byte_len ----------------------------------------

    #[test]
    fn zero_len_is_empty_and_allocation_free() {
        // `new(0)` must not touch the driver and must report an empty,
        // null/empty buffer. Holds on both backends.
        let buf: PinnedBuffer<u32> = PinnedBuffer::new(0).expect("zero-len alloc");
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.byte_len(), 0);
        assert_eq!(buf.capacity_bytes(), 0);
        assert_eq!(buf.as_slice(), &[] as &[u32]);
    }

    #[test]
    fn zero_len_mut_slice_is_empty() {
        // Mutable side: must produce a valid empty `&mut [T]`, never
        // `from_raw_parts_mut(null, 0)` (UB even at length 0).
        let mut buf: PinnedBuffer<u64> = PinnedBuffer::new(0).expect("zero-len alloc");
        let s: &mut [u64] = buf.as_mut_slice();
        assert!(s.is_empty());
    }

    #[test]
    fn byte_len_scales_with_element_size() {
        // `len * size_of::<T>()` accounting must be correct independent of the
        // backend. Under `cuda-stub` this allocates a real `Vec`; under the
        // live build it would hit the driver, so gate the allocation on the
        // stub feature and assert the math directly.
        #[cfg(feature = "cuda-stub")]
        {
            let buf: PinnedBuffer<u32> = PinnedBuffer::new(10).expect("alloc");
            assert_eq!(buf.len(), 10);
            assert_eq!(buf.byte_len(), 10 * size_of::<u32>());
            assert_eq!(buf.capacity_bytes(), 10 * size_of::<u32>());

            let buf64: PinnedBuffer<u64> = PinnedBuffer::new(7).expect("alloc");
            assert_eq!(buf64.byte_len(), 7 * size_of::<u64>());
        }
    }

    // ---- stub allocation path: Deref / round-trippable host buffer -------

    #[cfg(feature = "cuda-stub")]
    #[test]
    fn stub_allocation_is_readable_and_writable() {
        // The stub fallback must hand out a genuine, mutable host buffer so
        // the rest of the engine can stage data into it exactly as it would a
        // pinned buffer on a real GPU.
        let n = 256usize;
        let mut buf: PinnedBuffer<u32> = PinnedBuffer::new(n).expect("stub alloc");
        assert_eq!(buf.len(), n);

        // Fresh stub allocations are zero-initialized.
        assert!(buf.as_slice().iter().all(|&x| x == 0));

        // Write through the exclusive slice...
        for (i, slot) in buf.as_mut_slice().iter_mut().enumerate() {
            *slot = (i as u32).wrapping_mul(0x9E37_79B1);
        }
        // ...and read it back through the shared slice.
        for (i, &v) in buf.as_slice().iter().enumerate() {
            assert_eq!(v, (i as u32).wrapping_mul(0x9E37_79B1));
        }
    }

    #[cfg(feature = "cuda-stub")]
    #[test]
    fn deref_matches_as_slice() {
        // `Deref` / `DerefMut` must expose exactly the same elements as
        // `as_slice` / `as_mut_slice` so callers can treat a `PinnedBuffer`
        // like a `[T]`.
        let mut buf: PinnedBuffer<i64> = PinnedBuffer::new(4).expect("stub alloc");
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = -(i as i64);
        }
        // `&*buf` goes through `Deref`; compare to the explicit accessor.
        let via_deref: &[i64] = &buf;
        assert_eq!(via_deref, buf.as_slice());
        assert_eq!(buf.len(), via_deref.len());
        assert_eq!(via_deref, &[0, -1, -2, -3]);
    }

    #[cfg(feature = "cuda-stub")]
    #[test]
    fn set_len_truncates_logical_view() {
        // After a (hypothetical) partial D2H, `set_len` must shrink the
        // logical view while leaving the original capacity intact.
        let mut buf: PinnedBuffer<u16> = PinnedBuffer::new(8).expect("stub alloc");
        let cap = buf.capacity_bytes();
        // SAFETY (test): first 3 elements are initialized (zeroed at alloc).
        unsafe { buf.set_len(3) };
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.byte_len(), 3 * size_of::<u16>());
        // Capacity is unchanged by a logical truncation.
        assert_eq!(buf.capacity_bytes(), cap);
        assert_eq!(buf.as_slice().len(), 3);
    }

    #[cfg(feature = "cuda-stub")]
    #[test]
    #[should_panic(expected = "exceeds capacity")]
    fn set_len_beyond_capacity_panics() {
        // The bounds assert must fire (in any build profile) so an
        // out-of-range `new_len` can't expose memory past the allocation.
        let mut buf: PinnedBuffer<u8> = PinnedBuffer::new(4).expect("stub alloc");
        // SAFETY (test): deliberately out of range to trip the assert.
        unsafe { buf.set_len(5) };
    }

    // ---- stream-tracking state ------------------------------------------

    fn fake_stream(bits: usize) -> CUstream {
        bits as CUstream
    }

    #[test]
    fn mark_stream_use_accumulates_distinct_streams() {
        // The stream set must round-trip fabricated handles, dedup repeats, and
        // accumulate distinct streams (NOT overwrite to "last wins"). We never
        // deref the pointers — they're only stored. An empty buffer keeps the
        // test driver-free on both backends (Drop skips the fence / free).
        let buf: PinnedBuffer<u32> = PinnedBuffer::new(0).expect("zero-len alloc");
        assert_eq!(buf.recorded_stream_count(), 0);

        let a = fake_stream(0xDEAD_BEEF);
        buf.mark_stream_use(a);
        buf.mark_stream_use(a); // duplicate — must dedup
        assert_eq!(buf.recorded_stream_count(), 1);

        // Shared-borrow methods must not clobber the recorded set.
        let _ = buf.len();
        let _ = buf.as_slice();
        assert_eq!(buf.recorded_stream_count(), 1);

        // A second distinct stream accumulates rather than replacing the first
        // — this is the multi-stream fix. Null is a legal value (the default
        // stream is the null handle) and counts as a distinct member.
        let b = fake_stream(0xCAFE_F00D);
        buf.mark_stream_use(b);
        assert_eq!(
            buf.recorded_stream_count(),
            2,
            "distinct streams must accumulate, not overwrite"
        );
        buf.mark_stream_use(std::ptr::null_mut());
        assert_eq!(buf.recorded_stream_count(), 3);
    }

    // ---- Drop fences ALL recorded streams, via the mock seam -------------

    thread_local! {
        static FENCED: std::cell::RefCell<Vec<CUstream>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    /// Recording stub installed via `drop_fence_with`: logs each stream it is
    /// asked to fence and returns success. Never calls the driver.
    fn recording_fence(stream: CUstream) -> crate::cuda::cuda_sys::CUresult {
        FENCED.with(|f| f.borrow_mut().push(stream));
        crate::cuda::cuda_sys::CUDA_SUCCESS
    }

    #[test]
    fn drop_fences_every_recorded_stream_exactly_once() {
        // Mirror `cuda::buffer`'s multi-stream fence test: drive the exact
        // code `Drop` runs (`fence_all_streams` over the borrowed set) through
        // the recording stub, asserting every distinct recorded stream is
        // fenced exactly once. We operate on an empty buffer so no FFI free
        // runs — this isolates the "fence all, deduped" guarantee from the
        // `cuMemFreeHost`, keeping it driver-free on both backends.
        FENCED.with(|f| f.borrow_mut().clear());

        let buf: PinnedBuffer<u8> = PinnedBuffer::new(0).expect("zero-len alloc");
        let a = fake_stream(0x11);
        let b = fake_stream(0x22);
        buf.mark_stream_use(a);
        buf.mark_stream_use(b);
        buf.mark_stream_use(a); // duplicate — must not produce a 3rd fence

        drop_fence_with(recording_fence, || {
            let set = buf.used_streams.borrow();
            fence_all_streams(&set, recording_fence);
        });

        FENCED.with(|f| {
            let fenced = f.borrow();
            assert_eq!(
                fenced.len(),
                2,
                "Drop must fence each distinct recorded stream exactly once"
            );
            assert!(fenced.contains(&a) && fenced.contains(&b));
        });
    }

    #[test]
    fn empty_stream_set_fences_nothing() {
        // A buffer that was never tagged must not fence anything at Drop.
        FENCED.with(|f| f.borrow_mut().clear());
        let buf: PinnedBuffer<u8> = PinnedBuffer::new(0).expect("zero-len alloc");
        drop_fence_with(recording_fence, || {
            let set = buf.used_streams.borrow();
            if !set.is_empty() {
                fence_all_streams(&set, recording_fence);
            }
        });
        FENCED.with(|f| assert!(f.borrow().is_empty()));
    }

    #[test]
    fn stream_set_dedups_and_treats_null_as_real_handle() {
        // Pure bookkeeping check on the local accumulator: repeats dedup and
        // the null handle (default stream) is a distinct member like any other.
        let mut s = StreamSet::default();
        assert!(s.is_empty());
        let a = fake_stream(0x1000);
        s.insert(a);
        s.insert(a);
        s.insert(std::ptr::null_mut());
        s.insert(std::ptr::null_mut());
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn pinned_buffer_is_send() {
        // Lock down the documented `Send` (and *not* `Sync`) contract so a
        // future refactor adding a `!Send` field breaks here, not at a
        // distant call site.
        fn assert_send<T: Send>() {}
        assert_send::<PinnedBuffer<i32>>();
        assert_send::<PinnedBuffer<u8>>();
    }

    // ---- copy-helper signatures + length-guard (no GPU) ------------------

    #[test]
    fn copy_helpers_have_expected_signatures() {
        // Bind each helper to a function pointer of the expected shape so any
        // signature drift becomes a compile error here. Never calls them, so
        // it's safe under every feature configuration.
        let _up: fn(&CudaStream, &mut GpuViewMut<'_, u32>, &PinnedBuffer<u32>) -> BoltResult<()> =
            upload_async::<u32>;
        let _down: fn(&CudaStream, &mut PinnedBuffer<u32>, &GpuView<'_, u32>) -> BoltResult<()> =
            download_async::<u32>;
        let _sync: fn(&CudaStream) -> BoltResult<()> = sync;
    }

    #[test]
    fn upload_length_mismatch_is_rejected_before_ffi() {
        // A length mismatch must be caught by the host-side guard *before* any
        // FFI call, so this is exercisable without a GPU on both backends.
        // Use the NULL stream and empty device views so nothing is allocated.
        use crate::cuda::smart_ptrs::GpuVec;

        let stream = CudaStream::null();
        let mut dev: GpuVec<u32> = GpuVec::empty(); // len 0
        let mut dst = dev.view_mut();
        let src: PinnedBuffer<u32> = PinnedBuffer::new(0).expect("zero-len pinned");

        // Equal (both zero) -> Ok, and the empty fast-path issues no FFI.
        assert!(upload_async(&stream, &mut dst, &src).is_ok());
    }

    #[test]
    fn download_length_mismatch_is_rejected_before_ffi() {
        use crate::cuda::smart_ptrs::GpuVec;

        let stream = CudaStream::null();
        let dev: GpuVec<u32> = GpuVec::empty(); // len 0
        let src = dev.view();
        let mut dst: PinnedBuffer<u32> = PinnedBuffer::new(0).expect("zero-len pinned");

        // Equal (both zero) -> Ok.
        assert!(download_async(&stream, &mut dst, &src).is_ok());
    }

    /// End-to-end round-trip: pinned host -> device (async) -> pinned host
    /// (async), syncing the stream in between. Verifies the helpers actually
    /// move bytes through `cuMemcpy*Async_v2` and that `PinnedBuffer` hands
    /// out a DMA-able host region. GPU-gated like the rest of the crate.
    #[test]
    #[ignore = "gpu:mempool — set BOLT_BENCH_GPU=1 + run with --ignored"]
    fn pinned_buffer_async_roundtrip() {
        use crate::cuda::cuda_sys::CudaContext;
        use crate::cuda::smart_ptrs::GpuVec;

        // `CudaContext::new` calls `cuInit(0)`, so this is order-independent.
        let ctx = CudaContext::new(0).expect("create CUDA context");
        ctx.set_current().expect("set context current");

        let n = 4096usize;
        let mut host_in: PinnedBuffer<u32> = PinnedBuffer::new(n).expect("pinned in");
        let mut host_out: PinnedBuffer<u32> = PinnedBuffer::new(n).expect("pinned out");
        for (i, slot) in host_in.as_mut_slice().iter_mut().enumerate() {
            *slot = (i as u32).wrapping_mul(0x9E37_79B1);
        }
        for slot in host_out.as_mut_slice().iter_mut() {
            *slot = 0;
        }

        let stream = CudaStream::new().expect("create stream");
        // `upload_async` sizes the copy to `dst.len()`, so the device view must
        // already have a logical length of `n`. `with_capacity` yields len 0,
        // hence a zeroed vec of `n` here.
        let mut dev = GpuVec::<u32>::zeros(n).expect("alloc+zero device");

        {
            let mut dst = dev.view_mut();
            upload_async(&stream, &mut dst, &host_in).expect("async H2D");
        }
        sync(&stream).expect("sync after H2D");

        {
            let src = dev.view();
            download_async(&stream, &mut host_out, &src).expect("async D2H");
        }
        sync(&stream).expect("sync after D2H");

        assert_eq!(host_in.as_slice(), host_out.as_slice());
    }
}
