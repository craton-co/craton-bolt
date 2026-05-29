// SPDX-License-Identifier: Apache-2.0

//! Lifetime-tracked typed handles to GPU memory ("CUDA-Oxide").
//!
//! `GpuVec<T>` owns a `GpuBuffer<T>`; `GpuView<'a, T>` and `GpuViewMut<'a, T>`
//! borrow from it the same way `&[T]` and `&mut [T]` borrow from a `Vec<T>`.
//! The borrow checker, not runtime asserts, makes GPU aliasing a compile error.
//!
//! Examples of code the compiler will REJECT (do not paste these into a doctest
//! block — they are intentionally non-compiling):
//!
//!   // 1. kernel-uses-borrowed-vec: cannot drop while a view is live
//!   //    let v = GpuVec::<i32>::from_slice(&xs)?;
//!   //    let view = v.view();
//!   //    drop(v);                 // ERROR: cannot move out, `view` borrows v
//!   //    launch(&view);
//!
//!   // 2. double-free: cannot construct two owners from one allocation
//!   //    let a = GpuVec::<i32>::from_buffer(buf);
//!   //    let b = GpuVec::<i32>::from_buffer(buf); // ERROR: use of moved value
//!
//!   // 3. mut+shared overlap: cannot mix `&` and `&mut` borrows
//!   //    let shared = v.view();
//!   //    let exclusive = v.view_mut();            // ERROR: already borrowed
//!   //    use_both(&shared, &mut exclusive);

use std::marker::PhantomData;

use bytemuck::Pod;

use crate::cuda::buffer::{tag_stream_set, GpuBuffer, PinnedHostBuffer, StreamSetRef};
use crate::cuda::cuda_sys::{CUdeviceptr, CUstream};
use crate::error::BoltResult;

/// Owned, typed handle to a column of `T` on the GPU.
pub struct GpuVec<T: Pod> {
    buffer: GpuBuffer<T>,
}

impl<T: Pod> GpuVec<T> {
    /// Create an empty vector with a null device pointer and zero capacity.
    /// Does not allocate GPU memory and does not require CUDA at runtime.
    pub fn empty() -> Self {
        Self {
            buffer: GpuBuffer::empty(),
        }
    }

    /// Allocate a device vector and copy `slice` into it.
    pub fn from_slice(slice: &[T]) -> BoltResult<Self> {
        Ok(Self {
            buffer: GpuBuffer::<T>::from_slice(slice)?,
        })
    }

    /// Allocate `len` zero-initialized elements on the device.
    pub fn zeros(len: usize) -> BoltResult<Self> {
        Ok(Self {
            buffer: GpuBuffer::<T>::zeros(len)?,
        })
    }

    /// Allocate room for `cap` elements with logical length zero.
    pub fn with_capacity(cap: usize) -> BoltResult<Self> {
        Ok(Self {
            buffer: GpuBuffer::<T>::with_capacity(cap)?,
        })
    }

    /// Take ownership of an existing `GpuBuffer` without copying.
    pub fn from_buffer(buffer: GpuBuffer<T>) -> Self {
        Self { buffer }
    }

    /// Batch-5 incremental-cache helper. Allocates a new GpuVec sized for
    /// `total_len` elements, DtoD-copies the leading `prefix_len`
    /// elements from `self`'s buffer, then HtoD-uploads `tail` into the
    /// trailing rows. `self` is consumed so the old device allocation
    /// drops back to the pool at the end of the call (its memory has
    /// already been copied off the device by then, since the DtoD copy
    /// is synchronous).
    ///
    /// Returns an error if `prefix_len > self.len()` or if
    /// `prefix_len + tail.len() != total_len`.
    pub fn extended_with_prefix(
        self,
        total_len: usize,
        prefix_len: usize,
        tail: &[T],
    ) -> BoltResult<Self> {
        if prefix_len > self.buffer.len() {
            return Err(crate::error::BoltError::Memory(format!(
                "GpuVec::extended_with_prefix: prefix_len ({}) exceeds self.len ({})",
                prefix_len,
                self.buffer.len()
            )));
        }
        // SAFETY: `self.buffer` is a live device allocation of `self.len()`
        // `T`s and `prefix_len <= self.len()` (checked above). The
        // `from_prefix_and_tail` constructor allocates a fresh device
        // pointer distinct from `self.buffer`'s, so the non-overlap
        // requirement of `cuMemcpyDtoD_v2` holds.
        let buf = unsafe {
            GpuBuffer::<T>::from_prefix_and_tail(
                total_len,
                self.buffer.device_ptr(),
                prefix_len,
                tail,
            )?
        };
        // `self` drops here; its `GpuBuffer` returns to the pool.
        Ok(Self { buffer: buf })
    }

    /// Number of valid `T` elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the vec holds zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Allocated capacity in `T` elements.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Raw device pointer (for FFI / kernel-launch glue).
    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.buffer.device_ptr()
    }

    /// Borrow as a shared GPU view; many such views may coexist.
    ///
    /// The view carries a back-reference to this vec's buffer stream set so
    /// that a kernel launch driven off the view can tag the launch stream
    /// via [`GpuView::mark_launch_use`] — closing C-1 (see that method and
    /// the `Drop` invariant on [`GpuBuffer`]).
    #[inline]
    pub fn view(&self) -> GpuView<'_, T> {
        GpuView {
            ptr: self.buffer.device_ptr(),
            len: self.buffer.len(),
            streams: self.buffer.used_streams_cell(),
            _marker: PhantomData,
        }
    }

    /// Borrow as an exclusive GPU view; only one such view may exist.
    ///
    /// Carries the same stream-set back-reference as [`view`](Self::view);
    /// kernel launches that write through the view should tag the launch
    /// stream with [`GpuViewMut::mark_launch_use`].
    #[inline]
    pub fn view_mut(&mut self) -> GpuViewMut<'_, T> {
        GpuViewMut {
            ptr: self.buffer.device_ptr(),
            len: self.buffer.len(),
            streams: self.buffer.used_streams_cell(),
            _marker: PhantomData,
        }
    }

    /// Copy the vec back to a host `Vec<T>` (synchronous).
    pub fn to_vec(&self) -> BoltResult<Vec<T>> {
        self.buffer.to_vec()
    }

    // ----- Stage 2 / Stage 3 async memcpy entry points --------------------

    /// Allocate `len` elements and async-zero them on `stream`. Kernels
    /// enqueued on the same stream after this call observe a zeroed
    /// buffer without an explicit synchronize.
    pub fn zeros_async(len: usize, stream: CUstream) -> BoltResult<Self> {
        Ok(Self {
            buffer: GpuBuffer::<T>::zeros_async(len, stream)?,
        })
    }

    /// Allocate a device vec and enqueue an async H2D from `slice` on `stream`.
    /// Returns immediately; the device contents are not valid until `stream`
    /// is synchronized. `slice` must remain live and unmodified until then.
    /// Pair with a [`PinnedHostBuffer`] source for a true DMA transfer.
    pub fn from_slice_async(slice: &[T], stream: CUstream) -> BoltResult<Self> {
        let mut buffer = GpuBuffer::<T>::with_capacity(slice.len())?;
        buffer.copy_from_async(slice, stream)?;
        Ok(Self { buffer })
    }

    /// Enqueue an async D2H copy of this vec into `dst` on `stream`. `dst.len()`
    /// must equal `self.len()`. The host slice is not safe to read until the
    /// stream has been synchronized.
    pub fn copy_to_async(&self, dst: &mut [T], stream: CUstream) -> BoltResult<()> {
        self.buffer.copy_to_async(dst, stream)
    }

    /// Async D2H into a freshly-allocated [`PinnedHostBuffer<T>`] of
    /// `self.len()` elements. The caller MUST call
    /// `stream.synchronize()` before reading the pinned buffer.
    ///
    /// Use when you intend to feed the downloaded data straight into an
    /// Arrow `Vec<T>` — the DMA lands in pinned memory, then the host
    /// copies it into a regular `Vec` once. That is still strictly
    /// faster than the synchronous `to_vec()` path when paired with
    /// concurrent kernel work on the same stream.
    pub fn to_pinned_async(&self, stream: CUstream) -> BoltResult<PinnedHostBuffer<T>> {
        let mut pinned = PinnedHostBuffer::<T>::new(self.len())?;
        self.buffer.copy_to_slice_async(pinned.as_mut_slice(), stream)?;
        // V-2: the pinned buffer is the *host* destination of this async D2H,
        // so its page-locked pages are read/written by `stream` until the copy
        // completes. Record the stream in the pinned buffer's `StreamSet` so
        // its `Drop` fences `stream` before `cuMemFreeHost` — otherwise an
        // in-flight DMA could land in freed pages (host-side use-after-free).
        // (The `GpuBuffer` source is tagged separately by
        // `copy_to_slice_async`.) `mark_stream_use` dedups, so multi-stream
        // reuse of the same pinned buffer accumulates rather than clobbers.
        pinned.mark_stream_use(stream);
        Ok(pinned)
    }
}

/// Shared (immutable) GPU view; mirrors `&[T]` semantics.
///
/// `Send` only — `!Sync` because a kernel launch reads through this view
/// while a concurrent thread could launch a writer kernel against the parent
/// `GpuVec`. Kernels that need write access must take `GpuViewMut` instead.
#[derive(Copy, Clone)]
pub struct GpuView<'a, T: Pod> {
    ptr: CUdeviceptr,
    len: usize,
    /// Back-reference to the parent buffer's stream set, so a launch driven
    /// off this view can tag the launch stream (C-1). `null` for views over
    /// empty / placeholder buffers, in which case tagging is a no-op. The
    /// `'a` borrow keeps the parent buffer alive for the view's lifetime, so
    /// this raw pointer is always valid to dereference while the view lives.
    streams: StreamSetRef,
    _marker: PhantomData<(&'a [T], std::cell::Cell<()>)>,
}

impl<'a, T: Pod> GpuView<'a, T> {
    /// Number of `T` elements in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view spans zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer for FFI / kernel launches.
    ///
    /// See [`mark_launch_use`](Self::mark_launch_use): a caller that forwards
    /// this pointer into a kernel launch MUST tag the launch stream so the
    /// parent buffer's `Drop` fences it (review finding C-1).
    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Byte length of the view (`len * size_of::<T>()`).
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Record that `stream` has a kernel launch (or other async op) in
    /// flight that reads through this view, forwarding into the parent
    /// buffer's stream set. The parent's `Drop` then fences `stream` before
    /// the block returns to the pool.
    ///
    /// ## Why this is the C-1 closure
    ///
    /// A `GpuView` is a detached `Copy` snapshot (pointer + length); it does
    /// not borrow the `GpuVec` mutably, so it cannot call
    /// `GpuBuffer::mark_stream_use` directly. Instead it carries a raw
    /// back-reference (`streams`) to the parent buffer's stream-set cell.
    /// Kernel-launch glue that pushes `self.device_ptr()` into a launch MUST
    /// call this immediately after enqueueing on `stream`. Over-calling is
    /// safe — the set dedups.
    ///
    /// Idempotent and cheap; a no-op for a view over an empty buffer (null
    /// back-reference).
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // CUstream forwarded, not deref'd
    #[inline]
    pub fn mark_launch_use(&self, stream: CUstream) {
        // SAFETY: `self.streams` was minted by `GpuVec::view` from the live
        // parent buffer, whose lifetime is `>= 'a` (the view borrows it), so
        // the cell is valid for the whole life of `self`. `tag_stream_set`
        // null-checks and only mutates a `!Sync` cell from this one thread.
        unsafe { tag_stream_set(self.streams, stream) }
    }

    /// Hand the parent buffer's stream-set back-reference to the launch glue
    /// so it can centrally tag the launch stream at `cuLaunchKernel` time
    /// (review finding V-1). This is the same `StreamSetRef` that backs
    /// [`mark_launch_use`](Self::mark_launch_use); exposing it lets
    /// [`KernelArgs::push_input`](crate::exec::launch::KernelArgs::push_input)
    /// retain it after the view's device pointer has been copied out, so the
    /// launch entry point — not every one of the ~250 call sites — becomes
    /// the single tagging point. Crate-internal; never handed to FFI.
    #[inline]
    pub(crate) fn stream_set_ref(&self) -> StreamSetRef {
        self.streams
    }
}

// SAFETY: a `GpuView` is a device pointer, a length, and a raw back-pointer
// to the parent buffer's stream-set cell. Like `&[u8]` over opaque memory,
// moving it across threads cannot race on host state. The added raw pointer
// (`streams`) makes the auto-derived `Send` go away, hence this explicit
// impl. Soundness of moving it: the pointee outlives the view (the `'a`
// borrow keeps the parent buffer alive), and `mark_launch_use` only ever
// `borrow_mut`s the cell from the current thread. Craton Bolt serializes GPU
// launches per thread, so two threads never tag the same buffer's set at the
// same instant; even if they did, `RefCell` would panic rather than UB.
unsafe impl<'a, T: Pod> Send for GpuView<'a, T> {}
// Intentionally NOT `Sync`: under Craton Bolt's launch model a kernel can write
// through the parent `GpuVec` while another thread reads through the view
// across kernel boundaries. The `Cell<()>` in `_marker` makes this `!Sync`.

/// Exclusive (mutable) GPU view; mirrors `&mut [T]` semantics.
pub struct GpuViewMut<'a, T: Pod> {
    ptr: CUdeviceptr,
    len: usize,
    /// Back-reference to the parent buffer's stream set; see the field of
    /// the same name on [`GpuView`] and [`GpuViewMut::mark_launch_use`].
    streams: StreamSetRef,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T: Pod> GpuViewMut<'a, T> {
    /// Number of `T` elements in the view.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view spans zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer for FFI / kernel launches.
    ///
    /// See [`mark_launch_use`](Self::mark_launch_use): a caller that forwards
    /// this pointer into a kernel launch MUST tag the launch stream so the
    /// parent buffer's `Drop` fences it (review finding C-1).
    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Byte length of the view (`len * size_of::<T>()`).
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Record that `stream` has a kernel launch (or other async op) writing
    /// through this view, forwarding into the parent buffer's stream set so
    /// `Drop` fences it before recycling the block. See
    /// [`GpuView::mark_launch_use`] for the full C-1 rationale; this is the
    /// mutable-view counterpart. Idempotent; no-op for an empty buffer.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // CUstream forwarded, not deref'd
    #[inline]
    pub fn mark_launch_use(&self, stream: CUstream) {
        // SAFETY: identical to `GpuView::mark_launch_use` — the parent
        // buffer outlives `'a`, so `self.streams` is valid for the view's
        // life; `tag_stream_set` null-checks and only touches a `!Sync`
        // cell on this thread.
        unsafe { tag_stream_set(self.streams, stream) }
    }

    /// Mutable-view counterpart to [`GpuView::stream_set_ref`]: hand the
    /// parent buffer's stream-set back-reference to the launch glue so
    /// [`KernelArgs::push_output`](crate::exec::launch::KernelArgs::push_output)
    /// can retain it and the launch entry point can centrally tag the launch
    /// stream (review finding V-1). Crate-internal; never handed to FFI.
    #[inline]
    pub(crate) fn stream_set_ref(&self) -> StreamSetRef {
        self.streams
    }

    /// Re-borrow the exclusive view as a shared view for the remaining scope.
    /// The shared view inherits the same stream-set back-reference, so a
    /// launch driven off it still tags the parent buffer.
    #[inline]
    pub fn as_view(&self) -> GpuView<'_, T> {
        GpuView {
            ptr: self.ptr,
            len: self.len,
            streams: self.streams,
            _marker: PhantomData,
        }
    }
}

// SAFETY: ownership of a `GpuViewMut` may move between threads; the underlying
// device memory is reachable only via this single handle for its lifetime.
// The `streams` raw back-pointer carries the same soundness argument as the
// `GpuView` `Send` impl above (pointee outlives the view; `mark_launch_use`
// touches the `!Sync` cell only from the current thread).
unsafe impl<'a, T: Pod> Send for GpuViewMut<'a, T> {}
// Intentionally NOT `Sync`: concurrent mutation through shared references
// would race on device memory just as `&mut [T]` would on host memory.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_vec_empty_invariants() {
        let v = GpuVec::<i32>::empty();
        assert_eq!(v.len(), 0);
        assert!(v.is_empty());
        assert_eq!(v.device_ptr(), 0);
        // GpuVec doesn't expose byte_len directly, but its view does and must agree.
        assert_eq!(v.view().byte_len(), 0);
    }

    #[test]
    fn gpu_view_of_empty_vec() {
        let v = GpuVec::<i32>::empty();
        let view = v.view();
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
        assert_eq!(view.device_ptr(), 0);
        assert_eq!(view.byte_len(), 0);
    }

    #[test]
    fn gpu_view_mut_of_empty_vec() {
        let mut v = GpuVec::<i32>::empty();
        let view = v.view_mut();
        assert_eq!(view.len(), 0);
        assert!(view.is_empty());
        assert_eq!(view.device_ptr(), 0);
        assert_eq!(view.byte_len(), 0);
    }

    #[test]
    fn as_view_reborrows_as_shared() {
        let mut v = GpuVec::<i32>::empty();
        let m = v.view_mut();
        let s: GpuView<'_, i32> = m.as_view();
        assert_eq!(s.len(), 0);
        assert_eq!(s.device_ptr(), 0);
        // Compile-time check: explicitly bind the inferred type so a future
        // refactor that changes `as_view`'s return type breaks this test.
        let _type_check: GpuView<'_, i32> = s;
    }

    #[test]
    fn gpu_view_send_compile_check() {
        fn assert_send<T: Send>() {}
        assert_send::<GpuView<'static, i32>>();
        assert_send::<GpuViewMut<'static, i32>>();
        assert_send::<GpuVec<i32>>();
    }

    // gpu_view_not_sync_compile_check:
    //
    // GpuView and GpuViewMut are intentionally `!Sync` — the `Cell<()>` (resp.
    // `&mut [T]`) in their `PhantomData` enforces this at the type level. There
    // is no stable-Rust positive assertion for "does NOT implement Sync"
    // (negative trait bounds are unstable), so the invariant is exercised by
    // the `compile_fail` doctest in `tests/memory_tests.rs` rather than by a
    // runtime test here. Don't fight the type system.

    #[test]
    fn gpu_buffer_view_byte_len_matches_t() {
        let v = GpuVec::<u64>::empty();
        let view = v.view();
        assert_eq!(view.byte_len(), view.len() * std::mem::size_of::<u64>());
        // And specifically for the empty case:
        assert_eq!(view.byte_len(), 0);

        // The same identity must hold for the mutable view.
        let mut v2 = GpuVec::<u64>::empty();
        let vm = v2.view_mut();
        assert_eq!(vm.byte_len(), vm.len() * std::mem::size_of::<u64>());
    }

    // ---- C-1: view-level launch tagging forwards to the parent buffer ----
    //
    // These host-only tests verify that tagging through a view's
    // `mark_launch_use` lands in the *parent buffer's* stream set — the
    // mechanism by which a kernel launch keyed off `view.device_ptr()` makes
    // the parent's `Drop` fence the launch stream (closing C-1). They use
    // `GpuVec::empty()` (null device ptr, no driver calls) and reach the
    // private `buffer` field, which is in scope inside this child module.

    fn fake_stream(bits: usize) -> CUstream {
        bits as CUstream
    }

    #[test]
    fn shared_view_mark_launch_use_tags_parent_buffer() {
        let v = GpuVec::<i32>::empty();
        assert_eq!(v.buffer.recorded_stream_count(), 0);
        let view = v.view();
        let s = fake_stream(0x5151);
        view.mark_launch_use(s);
        view.mark_launch_use(s); // dedup through the parent set
        assert_eq!(
            v.buffer.recorded_stream_count(),
            1,
            "C-1: a launch tagged via the shared view must register on the \
             parent buffer's stream set (deduped)"
        );
    }

    #[test]
    fn mut_view_and_reborrow_share_one_parent_set() {
        let mut v = GpuVec::<i32>::empty();
        let a = fake_stream(0xAA);
        let b = fake_stream(0xBB);
        {
            let vm = v.view_mut();
            vm.mark_launch_use(a);
            // A reborrow as a shared view must point at the SAME parent set,
            // so tagging through it accumulates rather than forking.
            let shared = vm.as_view();
            shared.mark_launch_use(b);
            shared.mark_launch_use(a); // dedup across view + reborrow
        }
        assert_eq!(
            v.buffer.recorded_stream_count(),
            2,
            "C-1: mutable view and its `as_view` reborrow must feed the same \
             parent stream set"
        );
    }

    #[test]
    fn views_send_compile_check_still_holds_with_back_ref() {
        // The added raw back-pointer must not have silently dropped `Send`
        // (we re-assert the existing explicit impls cover it).
        fn assert_send<T: Send>() {}
        assert_send::<GpuView<'static, i32>>();
        assert_send::<GpuViewMut<'static, i32>>();
    }
}
