// SPDX-License-Identifier: Apache-2.0

//! Arrow-compatible columnar GPU storage primitives.
//!
//! `GpuBuffer<T>` is a raw, untyped (modulo `T`'s element size) device
//! allocation that mirrors Arrow's 64-byte CPU alignment. It is the low-level
//! primitive on which the typed, lifetime-tracked `GpuVec<T>` (Step 3) will be
//! built.

use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::size_of;

use bytemuck::Pod;

use crate::cuda::cuda_sys::{self, CUdeviceptr, CUstream};
use crate::error::{BoltError, BoltResult};

/// Arrow's mandated minimum buffer alignment, in bytes.
pub const ARROW_ALIGNMENT: usize = 64;

/// Raw, untyped GPU memory region. Arrow-aligned. Owns the allocation.
///
/// This is a low-level primitive. `GpuVec<T>` (Step 3) is the typed,
/// lifetime-tracked wrapper users should reach for.
pub struct GpuBuffer<T: Pod> {
    ptr: CUdeviceptr,
    /// Number of T elements.
    len: usize,
    /// Allocated capacity in T elements (>= len).
    capacity: usize,
    /// Rounded-up byte size we actually own. Needed so `Drop` returns the
    /// block to the correct pool bucket.
    alloc_bytes: usize,
    _t: PhantomData<T>,
}

impl<T: Pod> GpuBuffer<T> {
    /// Create an empty buffer with a null device pointer and zero capacity.
    /// Does not allocate GPU memory and does not require CUDA at runtime.
    pub fn empty() -> Self {
        Self {
            ptr: 0,
            len: 0,
            capacity: 0,
            alloc_bytes: 0,
            _t: PhantomData,
        }
    }

    /// Allocate an empty buffer with room for `capacity` elements of `T`.
    pub fn with_capacity(capacity: usize) -> BoltResult<Self> {
        let elem_size = size_of::<T>();
        let raw_bytes = capacity.checked_mul(elem_size).ok_or_else(|| {
            BoltError::Memory(format!(
                "GpuBuffer::with_capacity size overflow: {} * {}",
                capacity, elem_size
            ))
        })?;

        // Round up to ARROW_ALIGNMENT so even small buffers reserve an aligned
        // tail. Zero-sized requests still allocate one aligned chunk so we have
        // a stable, non-null device pointer to hand out.
        let requested = round_up_to_alignment(raw_bytes.max(ARROW_ALIGNMENT), ARROW_ALIGNMENT)
            .ok_or_else(|| {
                BoltError::Memory(format!(
                    "GpuBuffer::with_capacity alignment overflow for {} bytes",
                    raw_bytes
                ))
            })?;

        // Pool buckets round further up to the next power of two; we receive
        // the actual allocation size back so Drop returns to the right bucket.
        // The 64-byte alignment invariant is preserved transitively:
        // `cuMemAlloc_v2` guarantees ≥256-byte alignment, and the pool only
        // ever stores pointers minted there.
        //
        // Backend routing: under `--features cudarc` the pool's miss path
        // calls `cudarc_backend::mem_alloc`; otherwise it calls
        // `cuda_sys::mem_alloc`. Both wrap `cuMemAlloc_v2` and return a
        // bit-compatible `CUdeviceptr`, so callers of `GpuBuffer` are
        // backend-agnostic. See `mem_pool::DeviceMemPool::alloc`.
        let (ptr, alloc_bytes) = crate::cuda::mem_pool::POOL.alloc(requested)?;

        Ok(Self {
            ptr,
            len: 0,
            capacity,
            alloc_bytes,
            _t: PhantomData,
        })
    }

    /// Allocate `len` elements and zero them via `cuMemsetD8`.
    pub fn zeros(len: usize) -> BoltResult<Self> {
        let mut buf = Self::with_capacity(len)?;
        if len > 0 {
            let byte_len = len.checked_mul(size_of::<T>()).ok_or_else(|| {
                BoltError::Memory(format!(
                    "GpuBuffer::zeros size overflow: {} * {}",
                    len,
                    size_of::<T>()
                ))
            })?;
            // SAFETY: `buf.ptr` was just allocated with at least `byte_len`
            // bytes of capacity (rounded up to ARROW_ALIGNMENT).
            // `buf`'s `Drop` will free the allocation if the memset errors.
            unsafe {
                cuda_sys::memset_d8(buf.ptr, 0, byte_len)?;
            }
        }
        buf.len = len;
        Ok(buf)
    }

    /// Allocate and copy `slice` from host to device.
    pub fn from_slice(slice: &[T]) -> BoltResult<Self> {
        let mut buf = Self::with_capacity(slice.len())?;
        if !slice.is_empty() {
            // SAFETY: `buf.ptr` was allocated with capacity for `slice.len()`
            // elements of `T`, and `slice` is a valid read source for that many.
            unsafe {
                cuda_sys::memcpy_h2d::<T>(buf.ptr, slice.as_ptr(), slice.len())?;
            }
        }
        buf.len = slice.len();
        Ok(buf)
    }

    /// Number of valid `T` elements currently in the buffer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Valid byte length (`len * size_of::<T>()`).
    pub fn byte_len(&self) -> usize {
        self.len
            .checked_mul(size_of::<T>())
            .expect("byte_len overflow")
    }

    /// Allocated capacity in `T` elements.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Raw device pointer for kernel launches and FFI handoff.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Copy the buffer's contents back to a fresh host `Vec<T>`.
    pub fn to_vec(&self) -> BoltResult<Vec<T>> {
        let mut out: Vec<T> = Vec::with_capacity(self.len);
        if self.len > 0 {
            // SAFETY: `out` has capacity for `self.len` elements; we copy
            // exactly that many bytes from a live device allocation, then set
            // the logical length to match.
            unsafe {
                cuda_sys::memcpy_d2h::<T>(out.as_mut_ptr(), self.ptr, self.len)?;
                out.set_len(self.len);
            }
        }
        Ok(out)
    }

    /// Asynchronously upload `src` from host into this buffer on `stream`.
    ///
    /// On success the buffer's logical length is set to `src.len()` and the
    /// copy is *issued* on the stream — the caller must
    /// `cuStreamSynchronize` (or [`crate::exec::launch::CudaStream::synchronize`])
    /// before reading the destination from another stream or the host.
    ///
    /// For real H2D / kernel overlap, pass a [`PinnedHostBuffer`]-backed
    /// slice; pageable host memory still works but the driver synthesizes
    /// a staging copy and the call effectively serializes.
    ///
    /// Errors if `src.len()` exceeds the buffer's allocated capacity.
    ///
    /// # Safety / Lifetime contract
    ///
    /// **The buffer (NOT just the host source) must outlive the stream's
    /// completion.** There is no fence in [`GpuBuffer`]'s `Drop` impl;
    /// dropping `self` while the DMA is still in flight recycles the
    /// device pointer into the pool while the driver is still writing
    /// it, which is undefined behaviour and can corrupt unrelated
    /// allocations.
    ///
    /// Callers **MUST** call `cuStreamSynchronize` (or
    /// [`crate::exec::launch::CudaStream::synchronize`]) on `stream`
    /// before dropping `self`, or knowingly accept the risk (e.g. the
    /// stream is destroyed before drop, which itself synchronizes).
    /// The same rule applies to `src`: the host pages it points to must
    /// remain valid and unmodified until the stream completes.
    ///
    /// ## Buffer lifetime
    ///
    /// See the SAFETY rustdoc on [`impl Drop for GpuBuffer`](GpuBuffer#impl-Drop-for-GpuBuffer<T>)
    /// (review finding C13). `GpuBuffer::Drop` does **not** fence the
    /// stream — it returns the device pointer to the pool unconditionally,
    /// so dropping `self` while this H2D copy is still in flight will let
    /// the pool re-issue the same address to an unrelated allocation while
    /// the driver is still writing to it. That is silent data corruption,
    /// not an error. Synchronize before drop.
    // `stream` is an opaque CUstream handle, not actually dereferenced in this
    // function — only forwarded into a genuinely-unsafe FFI call. Marking the
    // outer fn `unsafe` would be a major API break for no real safety win.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn copy_from_async(&mut self, src: &[T], stream: CUstream) -> BoltResult<()> {
        if src.len() > self.capacity {
            return Err(BoltError::Memory(format!(
                "GpuBuffer::copy_from_async length exceeds capacity: src={}, cap={}",
                src.len(),
                self.capacity
            )));
        }
        if !src.is_empty() {
            // SAFETY: `src` is a valid host slice of `src.len()` `T`s; the
            // buffer has capacity for at least that many (checked above).
            // The caller is responsible for synchronizing `stream` before
            // freeing `src` or reading the destination.
            unsafe {
                cuda_sys::memcpy_h2d_async::<T>(self.ptr, src.as_ptr(), src.len(), stream)?;
            }
        }
        self.len = src.len();
        Ok(())
    }

    /// Asynchronously download this buffer into `dst` on `stream`.
    ///
    /// Errors if `dst.len()` differs from the buffer's logical length —
    /// matches the existing [`copy_to_slice`](Self::copy_to_slice)
    /// contract so an executor switching from sync to async only needs to
    /// thread a stream through, not rethink sizing.
    ///
    /// For real D2H / kernel overlap, pass a [`PinnedHostBuffer`]-backed
    /// slice. The caller must `cuStreamSynchronize` before reading `dst`.
    ///
    /// # Safety / Lifetime contract
    ///
    /// **The buffer (NOT just the host destination) must outlive the
    /// stream's completion.** There is no fence in [`GpuBuffer`]'s
    /// `Drop` impl; dropping `self` while the DMA is still in flight
    /// recycles the device pointer into the pool while the driver is
    /// still reading it, which is undefined behaviour and can corrupt
    /// unrelated allocations (the next pool consumer of that block
    /// will see torn writes from this copy).
    ///
    /// Callers **MUST** call `cuStreamSynchronize` (or
    /// [`crate::exec::launch::CudaStream::synchronize`]) on `stream`
    /// before dropping `self`, or knowingly accept the risk. The same
    /// rule applies to `dst`: the host pages it points to must remain
    /// valid until the stream completes.
    ///
    /// ## Buffer lifetime
    ///
    /// See the SAFETY rustdoc on [`impl Drop for GpuBuffer`](GpuBuffer#impl-Drop-for-GpuBuffer<T>)
    /// (review finding C13). `GpuBuffer::Drop` does **not** fence the
    /// stream — it returns the device pointer to the pool unconditionally,
    /// so dropping `self` while this D2H copy is still draining will let
    /// the pool re-issue the same address to an unrelated allocation. The
    /// in-flight read will then race with whatever the new owner writes
    /// there, and `dst` will receive a mixture of the two. Synchronize
    /// before drop.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // CUstream is forwarded, not deref'd
    pub fn copy_to_async(&self, dst: &mut [T], stream: CUstream) -> BoltResult<()> {
        if dst.len() != self.len {
            return Err(BoltError::Memory(format!(
                "GpuBuffer::copy_to_async length mismatch: dst={}, buffer={}",
                dst.len(),
                self.len
            )));
        }
        if self.len > 0 {
            // SAFETY: `dst` is valid for writes of `self.len` elements
            // (checked above), `self.ptr` is a live device allocation of at
            // least that many `T`s. Caller is responsible for synchronizing
            // `stream` before reading `dst`.
            unsafe {
                cuda_sys::memcpy_d2h_async::<T>(dst.as_mut_ptr(), self.ptr, self.len, stream)?;
            }
        }
        Ok(())
    }

    /// Alias for [`copy_to_async`](Self::copy_to_async). Stage-3 callers
    /// use this name to make the pinned-host pairing intent explicit at
    /// the call site.
    ///
    /// ## Buffer lifetime
    ///
    /// Inherits the same `Drop`-vs-in-flight-DMA hazard as
    /// [`copy_to_async`](Self::copy_to_async); see the SAFETY rustdoc on
    /// [`impl Drop for GpuBuffer`](GpuBuffer#impl-Drop-for-GpuBuffer<T>)
    /// (review finding C13). Synchronize the stream before dropping `self`.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // CUstream is forwarded, not deref'd
    pub fn copy_to_slice_async(&self, dst: &mut [T], stream: CUstream) -> BoltResult<()> {
        self.copy_to_async(dst, stream)
    }

    /// Allocate `len` elements and zero them via `cuMemsetD8Async` on `stream`.
    ///
    /// Stage-3 counterpart to [`zeros`]. The memset is enqueued on `stream`
    /// and the caller must synchronize before launching any kernel that
    /// reads these bytes on a *different* stream. Kernels enqueued on the
    /// same `stream` after this call see a fully-zeroed buffer without an
    /// explicit sync.
    ///
    /// ## Buffer lifetime
    ///
    /// See the SAFETY rustdoc on [`impl Drop for GpuBuffer`](GpuBuffer#impl-Drop-for-GpuBuffer<T>)
    /// (review finding C13). The returned buffer must outlive `stream`'s
    /// completion: dropping it while the `cuMemsetD8Async` is still in
    /// flight recycles the device pointer back into the pool while the
    /// driver is still zero-filling it, which will silently scribble over
    /// the next allocation that lands on the same block. Synchronize
    /// before drop.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // CUstream forwarded to FFI, not deref'd here
    pub fn zeros_async(len: usize, stream: CUstream) -> BoltResult<Self> {
        let mut buf = Self::with_capacity(len)?;
        if len > 0 {
            let byte_len = len.checked_mul(size_of::<T>()).ok_or_else(|| {
                BoltError::Memory(format!(
                    "GpuBuffer::zeros_async size overflow: {} * {}",
                    len,
                    size_of::<T>()
                ))
            })?;
            // `memset_d8_async` is a safe wrapper; its safety contract is
            // documented on the wrapper itself (caller must keep the range
            // live and uncontended until the stream is synchronized).
            // `buf.ptr` was just allocated with at least `byte_len` bytes
            // of capacity (rounded up to ARROW_ALIGNMENT).
            cuda_sys::memset_d8_async(buf.ptr, 0, byte_len, stream)?;
        }
        buf.len = len;
        Ok(buf)
    }

    /// Copy the buffer's contents into `dst`. Errors if lengths differ.
    pub fn copy_to_slice(&self, dst: &mut [T]) -> BoltResult<()> {
        if dst.len() != self.len {
            return Err(BoltError::Memory(format!(
                "GpuBuffer::copy_to_slice length mismatch: dst={}, buffer={}",
                dst.len(),
                self.len
            )));
        }
        if self.len > 0 {
            // SAFETY: `dst` is valid for writes of `self.len` elements (checked
            // above), and `self.ptr` is a live device allocation of at least
            // that many `T`s.
            unsafe {
                cuda_sys::memcpy_d2h::<T>(dst.as_mut_ptr(), self.ptr, self.len)?;
            }
        }
        Ok(())
    }

}

impl GpuBuffer<u8> {
    /// Copy the raw bytes of an Arrow CPU buffer into a fresh device buffer.
    pub fn from_arrow_bytes(buf: &arrow_buffer::Buffer) -> BoltResult<GpuBuffer<u8>> {
        GpuBuffer::<u8>::from_slice(buf.as_slice())
    }
}

/// # SAFETY — async DMA lifetime hazard (review finding C13)
///
/// **There is NO stream-completion fence in this `Drop` impl.** The device
/// block is returned directly to [`crate::cuda::mem_pool::POOL`] for reuse
/// the moment the owning `GpuBuffer` goes out of scope, regardless of
/// whether any asynchronous transfer or memset is still in flight on a
/// CUDA stream that touches `self.ptr`.
///
/// ## Why this matters
///
/// If a [`GpuBuffer`] participated in any of:
///
/// * [`GpuBuffer::copy_from_async`] (host→device DMA)
/// * [`GpuBuffer::copy_to_async`] / [`GpuBuffer::copy_to_slice_async`]
///   (device→host DMA)
/// * [`GpuBuffer::zeros_async`] (device-side `cuMemsetD8Async`)
/// * any external `cuMemcpy*Async` / `cuMemset*Async` keyed off
///   [`GpuBuffer::device_ptr`]
///
/// …and the buffer is dropped *before* its stream is synchronized, the
/// pool will hand the recycled `CUdeviceptr` to the next allocator. The
/// in-flight DMA continues chasing the same physical address and will
/// silently corrupt whatever unrelated `GpuBuffer` happens to receive that
/// pool block next. There is no driver-level error: the copy completes
/// "successfully" into someone else's data.
///
/// ## Required caller discipline
///
/// **Callers MUST `cuStreamSynchronize` (or
/// [`crate::exec::launch::CudaStream::synchronize`]) on every stream that
/// references this buffer's device pointer before letting the buffer drop.**
///
/// Equivalently, the stream itself may be destroyed first — destroying a
/// CUDA stream implicitly synchronizes it — but relying on that ordering
/// is fragile and discouraged.
///
/// See the matching `# Safety / Lifetime contract` sections on
/// [`GpuBuffer::copy_from_async`] and [`GpuBuffer::copy_to_async`] for the
/// per-method statement of this rule.
///
/// ## Why we don't fence here
///
/// A blanket `cuCtxSynchronize` (or per-buffer event-record + wait) in
/// `Drop` would serialize every buffer release against every stream in
/// the context, which would obliterate the H2D / kernel / D2H overlap
/// that the async API exists to enable. The architectural decision is
/// to push the fence to the caller, who knows which stream(s) actually
/// touched this buffer.
impl<T: Pod> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        // Return the block to the pool rather than the driver. The pool's
        // `drain` (run on process shutdown via `Drop` on the static) will
        // eventually hand it back to `cuMemFree_v2` — via the cudarc
        // backend when `--features cudarc` is active, or via the
        // hand-rolled FFI otherwise. Either way the call site here is
        // backend-agnostic because the pool stores raw `CUdeviceptr`s.
        //
        // NOTE (review C13): no stream-completion fence here. See the
        // SAFETY rustdoc on this `impl` block above.
        crate::cuda::mem_pool::POOL.free(self.ptr, self.alloc_bytes);
    }
}

// SAFETY: device pointers remain valid across threads provided the owning
// `CudaContext` is current on the using thread; moving a `GpuBuffer` between
// threads does not by itself violate any driver invariant.
unsafe impl<T: Pod> Send for GpuBuffer<T> {}

// Intentionally NOT `Sync`: all GPU operations should be serialized through a
// stream rather than racing through shared references.

/// Upload an Arrow primitive array's value buffer to the GPU as a typed buffer.
pub fn primitive_to_gpu<P>(
    arr: &arrow_array::PrimitiveArray<P>,
) -> BoltResult<GpuBuffer<P::Native>>
where
    P: arrow_array::types::ArrowPrimitiveType,
    P::Native: Pod,
{
    GpuBuffer::<P::Native>::from_slice(arr.values().as_ref())
}

// PinnedHostBuffer<T>
//
// Page-locked host allocation, RAII'd over `cuMemAllocHost_v2` /
// `cuMemFreeHost`. The point is to give async H2D / D2H a host source /
// destination the driver can DMA directly out of, rather than the
// staging-copy fall-back the driver synthesizes for pageable memory.
//
// Allocations are limited (the kernel's pinned-pool is global), so use only
// for the *final* H2D upload sources and D2H download targets — not for
// every intermediate buffer.
// ---------------------------------------------------------------------------

/// Owned page-locked (pinned) host buffer. Backed by `cuMemAllocHost_v2`.
///
/// Pinned host memory enables real H2D / D2H overlap with kernel
/// execution: the driver can DMA straight in / out of `as_slice()` without
/// allocating a staging buffer behind your back. The price is that pinned
/// pages cannot be paged out, so don't allocate gigabytes of these and
/// forget them.
///
/// Borrow rules mirror `Vec<T>`: `as_slice` lends a shared `&[T]`,
/// `as_mut_slice` an exclusive `&mut [T]`. The buffer is `Send` so it can
/// move between threads, but **not** `Sync` — concurrent mutation through
/// shared references would race the same way `&mut [T]` does on host
/// memory (and would race in-flight DMA besides).
pub struct PinnedHostBuffer<T: Pod> {
    /// Pinned host pointer (valid host VA, page-locked).
    ptr: *mut T,
    /// Logical element count.
    len: usize,
    /// Number of bytes the driver returned to us. Cached so `Drop` doesn't
    /// have to recompute (and so a future power-of-two-bucketed pool can
    /// hook in cleanly later).
    byte_len: usize,
    /// Last stream this buffer was used on, if any. Callers that enqueue
    /// async work referencing `self.ptr` (H2D / D2H DMA, kernels that
    /// touch the pinned region, etc.) should call
    /// [`PinnedHostBuffer::mark_stream_use`] after enqueueing so `Drop`
    /// can fence against an in-flight transfer before `cuMemFreeHost`
    /// reclaims the pages. See the `Drop` impl below for the rationale.
    ///
    /// `Cell<_>` rather than a plain field so `mark_stream_use` can take
    /// `&self`: typical async copy helpers borrow the pinned buffer
    /// shared (`as_ptr()` reads through `&self`), and forcing them to
    /// `&mut self` here would ripple needlessly through the call sites.
    /// The cell is single-threaded (the struct is `!Sync`), so no atomic
    /// is required.
    last_use_stream: Cell<Option<CUstream>>,
}

impl<T: Pod> PinnedHostBuffer<T> {
    /// Allocate `len` page-locked elements of `T` via `cuMemAllocHost_v2`.
    ///
    /// `len == 0` is allowed and returns a buffer with a null pointer; no
    /// FFI call is made. `as_slice` on a zero-length buffer returns `&[]`.
    pub fn new(len: usize) -> BoltResult<Self> {
        if len == 0 {
            // Zero-length fast path: don't touch the driver, don't hand
            // out a null pointer through `as_slice()` (the slice methods
            // below short-circuit instead — see `as_slice` / `as_mut_slice`).
            return Ok(Self {
                ptr: std::ptr::null_mut(),
                len: 0,
                byte_len: 0,
                last_use_stream: Cell::new(None),
            });
        }
        let byte_len = len.checked_mul(size_of::<T>()).ok_or_else(|| {
            BoltError::Memory(format!(
                "PinnedHostBuffer::new size overflow: {} * {}",
                len,
                size_of::<T>()
            ))
        })?;
        // SAFETY: `byte_len > 0` by construction (len > 0 and size_of::<T>
        // is at least 1 for any `Pod`). Driver returns a non-null pointer
        // on success.
        //
        // Note: this path goes through the raw FFI even under
        // `--features cudarc` because cudarc 0.13's `driver` feature does
        // not expose `cuMemAllocHost_v2`. That's fine — the pointer is
        // bit-compatible with the driver, just like our other shared FFI
        // boundary points.
        let raw = unsafe { cuda_sys::mem_alloc_host(byte_len)? };
        Ok(Self {
            ptr: raw as *mut T,
            len,
            byte_len,
            last_use_stream: Cell::new(None),
        })
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

    /// Byte length (`len * size_of::<T>()`).
    ///
    /// Computed on the fly from the *logical* element count so it tracks
    /// `set_len` truncations. To recover the original pinned allocation
    /// size, read `self.byte_len` (the cached field) directly — not exposed
    /// because no caller currently needs it.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * size_of::<T>()
    }

    /// Raw host pointer (for async memcpy FFI). May be null when `len == 0`.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Mutable raw host pointer. May be null when `len == 0`.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Borrow as a shared slice. Safe: `T: Pod` so any bit pattern is a
    /// valid value, and the buffer guarantees `len` initialized elements
    /// (post-allocation the contents are driver-defined but readable —
    /// which is fine for `Pod`).
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            // Empty buffer: return a stable empty slice. Don't call
            // `from_raw_parts` with a null pointer — that's UB even for
            // a zero length.
            return &[];
        }
        // SAFETY: `self.ptr` is a valid host VA for `self.len` `T`s
        // (allocated by `cuMemAllocHost_v2`), the buffer outlives the
        // borrow, and `T: Pod` accepts any bit pattern.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Borrow as an exclusive slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            // Stable empty mutable slice without going through a null
            // pointer in `from_raw_parts_mut`.
            return <&mut [T]>::default();
        }
        // SAFETY: same as `as_slice`, plus the `&mut self` receiver
        // statically prevents aliasing.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Record that `stream` has enqueued (or is about to enqueue) work
    /// that references this buffer's pinned host pages. The recorded
    /// stream is the one `Drop` will `cuStreamSynchronize` against
    /// before calling `cuMemFreeHost`, so the driver can't be still
    /// DMA-ing into freed pages.
    ///
    /// Mirrors the `mark_stream_use` contract on `GpuBuffer` (review
    /// finding C13). Callers MUST invoke this on the pinned source of
    /// `cuMemcpyHtoDAsync_v2`, the pinned destination of
    /// `cuMemcpyDtoHAsync_v2`, or any kernel parameter that holds
    /// `as_ptr()` / `as_mut_ptr()`, after enqueueing the async work.
    ///
    /// Only the most recent stream is remembered. If a buffer is used
    /// across multiple streams, the caller is responsible for the
    /// cross-stream barriers (events, joins) that make a single
    /// final-stream sync sufficient — or for explicitly synchronising
    /// the buffer before drop.
    ///
    /// Takes `&self` (interior mutability via `Cell`) so call sites
    /// holding a shared borrow — typical for an async helper that only
    /// reads `as_ptr()` — don't have to be rewritten to `&mut self`.
    /// The cell is single-threaded (the struct is `!Sync`) so no atomic
    /// is required.
    // `stream` is an opaque CUstream handle; we just store it. No deref,
    // no FFI. The outer fn stays safe to keep the call-site ergonomics
    // identical to the analogous helper on `GpuBuffer`.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    pub fn mark_stream_use(&self, stream: CUstream) {
        self.last_use_stream.set(Some(stream));
    }

    /// Read back the stream most recently recorded via
    /// [`mark_stream_use`](Self::mark_stream_use), or `None` if the
    /// buffer has never participated in async work.
    ///
    /// Primarily for tests; production code rarely needs to query this
    /// because `Drop` consumes it directly.
    #[inline]
    pub fn last_use_stream(&self) -> Option<CUstream> {
        self.last_use_stream.get()
    }

    /// Override the logical length — used after an async D2H that filled
    /// fewer than the buffer's allocated length.
    ///
    /// # Safety
    /// The first `new_len` elements must be initialised (e.g. by a
    /// completed async D2H). The upper bound (`new_len * size_of::<T>()
    /// <= self.byte_len`) is enforced unconditionally by an `assert!`
    /// below — in release as well as debug — so an out-of-range
    /// `new_len` will panic rather than silently expose uninitialised
    /// pinned memory through `as_slice()`. The cost is an integer
    /// compare; the safety win is that miscalculated lengths cannot
    /// turn into UB.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        assert!(
            new_len * size_of::<T>() <= self.byte_len,
            "PinnedHostBuffer::set_len({}) exceeds allocation ({} bytes)",
            new_len,
            self.byte_len,
        );
        self.len = new_len;
    }
}

impl<T: Pod> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            // Zero-length / never-allocated path: nothing to free and
            // nothing to fence against. `last_use_stream` is meaningless
            // here (callers shouldn't `mark_stream_use` on an empty
            // buffer, but if they do we silently ignore it).
            return;
        }
        // Fence against any in-flight DMA before returning the pinned
        // pages to the driver. Without this, an outstanding
        // `cuMemcpyHtoDAsync_v2` / `cuMemcpyDtoHAsync_v2` whose host
        // operand is `self.ptr` would continue to read or write the
        // page-locked region after `cuMemFreeHost` released it back to
        // the kernel — classic use-after-free on the host side, and a
        // particularly nasty one because the DMA engine has no notion
        // of Rust ownership.
        //
        // TODO(perf): a blanket sync is the safe default but it
        // serialises every pinned-buffer release against the recorded
        // stream's full queue, which can stall pipelined H2D / kernel /
        // D2H overlap if the caller has already arranged a
        // cross-stream barrier (event-record + wait) elsewhere. A
        // future optimisation could replace this with a per-buffer
        // `cuEventRecord` + `cuEventSynchronize` on a tiny event
        // attached at `mark_stream_use` time, so we wait only on the
        // specific point in the stream's history that actually touched
        // this buffer, not the entire trailing tail of the stream.
        if let Some(stream) = self.last_use_stream.get() {
            // SAFETY: `stream` is an opaque CUstream handle handed to
            // us by the caller; we just forward it. If the stream has
            // already been destroyed, `cuStreamSynchronize` returns an
            // error which we surface as a warning — we still attempt
            // the free so we don't leak the pinned pages.
            let sync_rc =
                unsafe { cuda_sys::check(cuda_sys::cuStreamSynchronize(stream)) };
            if let Err(e) = sync_rc {
                log::warn!(
                    "craton-bolt: cuStreamSynchronize before pinned-host free failed ({:?}); proceeding with cuMemFreeHost but in-flight DMA may have UB'd",
                    e
                );
            }
        }
        // SAFETY: `self.ptr` came from `cuMemAllocHost_v2` and we have
        // unique ownership (move-only, `!Sync`). The stream sync above
        // is best-effort; the caller is still responsible for the
        // overall lifetime contract on `memcpy_*_async` (see
        // `mark_stream_use`).
        // Cast via `*mut std::ffi::c_void` so we don't pull `libc` into the
        // module's `use` list just for this one type — `std::ffi::c_void`
        // and `libc::c_void` are the same alias.
        let rc =
            unsafe { cuda_sys::mem_free_host(self.ptr as *mut std::ffi::c_void) };
        if let Err(e) = rc {
            log::warn!(
                "craton-bolt: cuMemFreeHost failed ({:?}); pinned host buffer leaked",
                e
            );
        }
    }
}

// SAFETY: ownership of a pinned host buffer may move between threads —
// `cuMemAllocHost_v2` returns a host pointer the OS can read from any
// thread. Cross-thread *sharing* without external sync is unsound (it
// would race in-flight DMA and the borrow checker), so we do NOT
// implement Sync.
unsafe impl<T: Pod> Send for PinnedHostBuffer<T> {}
// `!Sync` is implicit — `*mut T` is `!Send + !Sync`, the explicit
// `Send` impl above opts back in to `Send` only.

/// Round `n` up to the next multiple of `align` (which must be a power of two).
/// Returns `None` on overflow.
fn round_up_to_alignment(n: usize, align: usize) -> Option<usize> {
    debug_assert!(align.is_power_of_two());
    let mask = align - 1;
    n.checked_add(mask).map(|v| v & !mask)
}

#[cfg(test)]
mod tests {
    //! Host-only tests for the pure helpers in this module.
    //!
    //! Nothing in here may touch CUDA: these run on machines without a GPU
    //! and on docs.rs. Anything that needs the driver belongs in an
    //! integration test with `#[ignore]` (see `dictionary_i64.rs` for the
    //! pattern).
    use super::*;
    use std::mem::size_of;

    // ---- round_up_to_alignment -------------------------------------------

    #[test]
    fn round_up_zero_is_zero() {
        // Zero is already aligned to anything; the function must not promote
        // it to a non-zero multiple.
        assert_eq!(round_up_to_alignment(0, ARROW_ALIGNMENT), Some(0));
    }

    #[test]
    fn round_up_one_byte_promotes_to_full_alignment() {
        assert_eq!(round_up_to_alignment(1, ARROW_ALIGNMENT), Some(64));
    }

    #[test]
    fn round_up_just_under_alignment() {
        // 63 -> 64: one byte short of the alignment boundary still rounds up.
        assert_eq!(round_up_to_alignment(63, ARROW_ALIGNMENT), Some(64));
    }

    #[test]
    fn round_up_exact_multiple_is_idempotent() {
        // Exact multiples must be left alone (no spurious second-bump).
        assert_eq!(round_up_to_alignment(64, ARROW_ALIGNMENT), Some(64));
        assert_eq!(round_up_to_alignment(128, ARROW_ALIGNMENT), Some(128));
        assert_eq!(round_up_to_alignment(64 * 1024, ARROW_ALIGNMENT), Some(64 * 1024));
    }

    #[test]
    fn round_up_just_over_alignment() {
        // 65 -> 128: one byte past a boundary jumps to the next chunk.
        assert_eq!(round_up_to_alignment(65, ARROW_ALIGNMENT), Some(128));
    }

    #[test]
    fn round_up_with_smaller_alignments() {
        // Function must work for any power-of-two alignment, not just 64.
        assert_eq!(round_up_to_alignment(0, 8), Some(0));
        assert_eq!(round_up_to_alignment(1, 8), Some(8));
        assert_eq!(round_up_to_alignment(7, 8), Some(8));
        assert_eq!(round_up_to_alignment(8, 8), Some(8));
        assert_eq!(round_up_to_alignment(9, 8), Some(16));
        // Alignment of 1 is a no-op (1 is a power of two).
        assert_eq!(round_up_to_alignment(0, 1), Some(0));
        assert_eq!(round_up_to_alignment(1, 1), Some(1));
        assert_eq!(round_up_to_alignment(42, 1), Some(42));
    }

    #[test]
    fn round_up_overflow_returns_none() {
        // `usize::MAX` cannot fit a 64-byte tail; the checked_add must trip
        // and we surface `None` rather than silently wrapping.
        assert_eq!(round_up_to_alignment(usize::MAX, ARROW_ALIGNMENT), None);

        // The largest value that *does* fit is `usize::MAX - (ARROW_ALIGNMENT - 1)`,
        // which rounds to a value with the alignment bits cleared.
        let max_fitting = usize::MAX - (ARROW_ALIGNMENT - 1);
        let expected = usize::MAX & !(ARROW_ALIGNMENT - 1);
        assert_eq!(round_up_to_alignment(max_fitting, ARROW_ALIGNMENT), Some(expected));

        // One byte past that limit overflows.
        assert_eq!(
            round_up_to_alignment(max_fitting + 1, ARROW_ALIGNMENT),
            None
        );
    }

    #[test]
    fn round_up_handles_half_of_usize_max() {
        // `usize::MAX / 2` plus the 63-byte mask cannot overflow on any
        // realistic 64-bit host, so the call must succeed and produce an
        // aligned value.
        let n = usize::MAX / 2;
        let result = round_up_to_alignment(n, ARROW_ALIGNMENT).expect("must not overflow");
        assert_eq!(result % ARROW_ALIGNMENT, 0);
        assert!(result >= n);
        assert!(result - n < ARROW_ALIGNMENT);
    }

    // ---- GpuBuffer::empty + host-only field invariants -------------------

    #[test]
    fn empty_buffer_has_null_ptr_and_zero_len() {
        // `empty()` is explicitly documented as not requiring CUDA. Verify
        // the host-visible invariants the rest of the module relies on:
        // a null device pointer (so `Drop` skips `mem_free`) and zero len /
        // capacity.
        let buf: GpuBuffer<i32> = GpuBuffer::empty();
        assert_eq!(buf.device_ptr(), 0);
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.capacity(), 0);
        assert!(buf.is_empty());
        // `byte_len` on an empty buffer is trivially zero, regardless of `T`.
        assert_eq!(buf.byte_len(), 0);
    }

    #[test]
    fn empty_buffer_byte_len_scales_with_t_when_len_is_set() {
        // Reach past the public API to fabricate a buffer with a non-zero
        // logical length but no allocation. This lets us test `byte_len`'s
        // multiplication without touching the GPU. Because the test lives in
        // a child module of `buffer`, the private `len` field is in scope.
        let mut buf: GpuBuffer<u32> = GpuBuffer::empty();
        buf.len = 4;
        // 4 elements * 4 bytes/u32 = 16 bytes.
        assert_eq!(buf.byte_len(), 4 * size_of::<u32>());
    }

    #[test]
    #[should_panic(expected = "byte_len overflow")]
    fn byte_len_panics_on_overflow() {
        // Pick an element whose `size_of` is > 1 so `len * size_of::<T>()`
        // can overflow even though `len` itself fits in `usize`. A 4-byte
        // `u32` with `len = usize::MAX` multiplies to 4 * usize::MAX, which
        // overflows and must trip the `expect("byte_len overflow")` guard.
        //
        // We never allocate this buffer on the device — we synthesize it
        // with `empty()` and rewrite `len`. The buffer's `Drop` is a no-op
        // because `ptr == 0`, so this is safe to leak through panic.
        let mut buf: GpuBuffer<u32> = GpuBuffer::empty();
        buf.len = usize::MAX;
        let _ = buf.byte_len();
    }

    #[test]
    fn empty_buffer_drop_is_noop_for_null_ptr() {
        // Building, dropping, and rebuilding empty buffers must not panic or
        // call into the CUDA driver. This is the host-only path the rest of
        // the codebase relies on for placeholder/test-only `GpuVec`s.
        for _ in 0..16 {
            let _b: GpuBuffer<u8> = GpuBuffer::empty();
        }
    }

    // ---- PinnedHostBuffer (host-only zero-length path) -------------------
    //
    // The GPU round-trip lives in `cuda_sys::tests::pinned_host_buffer_roundtrip`
    // behind `#[ignore]`. These host-only tests cover the zero-length fast
    // path which never touches the driver and is the only thing exercisable
    // on a non-CUDA host.

    #[test]
    fn pinned_host_buffer_zero_len_is_noop() {
        // `new(0)` returns a null-pointer buffer without calling
        // `cuMemAllocHost_v2` — verify that contract so we don't regress
        // it and start requiring CUDA for empty allocs.
        let buf: PinnedHostBuffer<u32> = PinnedHostBuffer::new(0).expect("zero-len alloc");
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.byte_len(), 0);
        assert!(buf.as_ptr().is_null());
        assert_eq!(buf.as_slice(), &[] as &[u32]);
    }

    #[test]
    fn pinned_host_buffer_zero_len_mut_slice_is_empty() {
        // Same path, mutable side: `as_mut_slice()` must produce a valid
        // empty `&mut [T]` (NOT call `from_raw_parts_mut` on a null
        // pointer, which would be UB even for length 0).
        let mut buf: PinnedHostBuffer<u64> = PinnedHostBuffer::new(0).expect("zero-len alloc");
        let s: &mut [u64] = buf.as_mut_slice();
        assert!(s.is_empty());
    }

    #[test]
    fn pinned_host_buffer_send_compile_check() {
        // PinnedHostBuffer is documented `Send`; lock that down so a
        // future refactor that adds a `Rc` or similar !Send field breaks
        // here rather than at a distant call site.
        fn assert_send<T: Send>() {}
        assert_send::<PinnedHostBuffer<i32>>();
        assert_send::<PinnedHostBuffer<u8>>();
    }
}

#[cfg(test)]
mod pinned_safety_tests {
    //! Host-only safety tests for `PinnedHostBuffer`'s zero-length path,
    //! stream-tracking state, and `Drop` discipline. None of these tests
    //! may call into the CUDA driver — they run on hosts without a GPU
    //! and on docs.rs. The companion round-trip test that actually
    //! exercises pinned DMA lives in `cuda_sys::tests` behind
    //! `#[ignore]` per the existing module conventions.
    use super::*;
    use crate::cuda::cuda_sys::CUstream;
    use std::ptr;

    #[test]
    fn empty_pinned_host_buffer_slice_is_zero_length() {
        // Symmetric with the existing `pinned_host_buffer_zero_len_is_noop`
        // test but written with the explicit `.len() == 0` form the safety
        // hardening contract calls out. A `from_raw_parts(null, 0)` here
        // would be UB even though the slice is empty — the production path
        // must short-circuit before reaching that call.
        let buf: PinnedHostBuffer<u8> = PinnedHostBuffer::new(0)
            .expect("zero-length pinned alloc must not error");
        let s = buf.as_slice();
        assert_eq!(s.len(), 0);
        // `as_ptr()` is allowed to be null for an empty buffer, but the
        // slice itself must be a valid empty borrow; `iter().count()`
        // forces the compiler to actually walk it.
        assert_eq!(s.iter().count(), 0);
    }

    #[test]
    fn empty_pinned_host_buffer_drop_without_stream_use_is_noop() {
        // Constructing and dropping an empty buffer must not panic, must
        // not call into the driver, and must not attempt to
        // `cuStreamSynchronize` against a stream that was never recorded.
        // We exercise the path by going out of scope at the end of the
        // block: if `Drop` tried to fence with a `last_use_stream` of
        // `None`, the `if let Some(stream) = ...` guard catches it; if
        // the ptr-null short-circuit failed and we reached the FFI, the
        // cuda-stub build would panic with `CUDA_ERROR_STUB`.
        for _ in 0..8 {
            let _b: PinnedHostBuffer<i64> =
                PinnedHostBuffer::new(0).expect("zero-len alloc");
        }
    }

    #[test]
    fn mark_stream_use_is_preserved_across_method_calls() {
        // The stream-tracking cell must survive arbitrary `&self` method
        // calls — only an explicit second `mark_stream_use` (or Drop)
        // should replace it. We use a fabricated non-null sentinel
        // pointer for the stream handle; we never deref it.
        //
        // This buffer is empty, so its `Drop` will skip the FFI free and
        // also skip the stream sync (the ptr-null short-circuit runs
        // first). That keeps the test driver-free on hosts without CUDA.
        let buf: PinnedHostBuffer<u32> =
            PinnedHostBuffer::new(0).expect("zero-len alloc");
        let fake_stream: CUstream = 0xDEAD_BEEF_usize as CUstream;
        assert!(buf.last_use_stream().is_none());
        buf.mark_stream_use(fake_stream);
        assert_eq!(buf.last_use_stream(), Some(fake_stream));

        // Hit a few `&self`-receiver methods and re-check.
        let _ = buf.len();
        let _ = buf.is_empty();
        let _ = buf.byte_len();
        let _ = buf.as_ptr();
        let _ = buf.as_slice();
        assert_eq!(
            buf.last_use_stream(),
            Some(fake_stream),
            "method calls must not clobber the recorded stream"
        );

        // A second `mark_stream_use` replaces the recorded stream
        // (last-wins, matching the documented contract).
        let other_stream: CUstream = 0xCAFE_F00D_usize as CUstream;
        buf.mark_stream_use(other_stream);
        assert_eq!(buf.last_use_stream(), Some(other_stream));

        // And a null stream is a perfectly legal value (the default
        // stream is the null handle).
        buf.mark_stream_use(ptr::null_mut());
        assert_eq!(buf.last_use_stream(), Some(ptr::null_mut::<()>() as CUstream));
    }
}
