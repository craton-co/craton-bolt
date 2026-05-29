// SPDX-License-Identifier: Apache-2.0

//! Arrow-compatible columnar GPU storage primitives.
//!
//! `GpuBuffer<T>` is a raw, untyped (modulo `T`'s element size) device
//! allocation that mirrors Arrow's 64-byte CPU alignment. It is the low-level
//! primitive on which the typed, lifetime-tracked `GpuVec<T>` (Step 3) will be
//! built.

use std::cell::RefCell;
// `Cell` is only used by the test-only `DROP_FENCE_OVERRIDE` thread-local
// (the production `StreamSet` tracking on both `GpuBuffer` and
// `PinnedHostBuffer` uses `RefCell<StreamSet>` after review finding V-2
// removed the single-stream `Cell<Option<CUstream>>`); gate the import so a
// non-test build doesn't warn on an unused import.
#[cfg(test)]
use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::size_of;

use bytemuck::Pod;

use crate::cuda::cuda_sys::{self, CUdeviceptr, CUstream};
use crate::error::{BoltError, BoltResult};

/// Arrow's mandated minimum buffer alignment, in bytes.
pub const ARROW_ALIGNMENT: usize = 64;

/// Deduplicated set of stream handles a [`GpuBuffer`] has been enqueued on.
///
/// ## Why a *set*, not a single "last stream" (review finding C-2)
///
/// The pre-C-2 design tracked only the most-recently-tagged stream and
/// fenced just that one at `Drop`. A buffer enqueued on stream A and then
/// stream B would fence only B — so an *independent* op still running on A
/// could race the recycled pool block after the buffer's address was handed
/// to the next allocator. Tracking the full set and fencing **every**
/// recorded stream at `Drop` closes that hole: no stream that ever touched
/// the block can still be in flight once the block returns to the pool.
///
/// ## Representation
///
/// In practice a buffer is touched by one, occasionally two, streams, so the
/// set is tiny. We store handles in a `Vec<CUstream>` and dedup linearly on
/// insert. Linear dedup is O(n) per insert but n is ~1–2 in every realistic
/// workload, so this is cheaper than the allocation/hashing overhead of a
/// `HashSet` and keeps the type trivially `!Sync`.
///
/// `CUstream` is a raw pointer handle; we never dereference it here, we only
/// compare handles for equality and forward them to `cuStreamSynchronize`.
#[derive(Default)]
pub(crate) struct StreamSet {
    /// Distinct stream handles, in first-seen order. Never contains
    /// duplicates (enforced by [`StreamSet::insert`]).
    streams: Vec<CUstream>,
}

impl StreamSet {
    /// Record `stream` if not already present. Dedups so `Drop` issues at
    /// most one `cuStreamSynchronize` per distinct stream.
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

/// Opaque back-reference a [`GpuView`](crate::cuda::smart_ptrs::GpuView) /
/// [`GpuViewMut`](crate::cuda::smart_ptrs::GpuViewMut) carries so it can tag
/// the parent buffer's stream set at kernel-launch time (closing C-1 without
/// the view needing a `&GpuVec`). It is a raw pointer to the parent buffer's
/// private stream-set cell; `null` means "no parent" (empty/placeholder
/// view) and tagging is a no-op. Always paired with the view's lifetime, so
/// the pointee outlives every dereference. See [`tag_stream_set`].
pub(crate) type StreamSetRef = *const RefCell<StreamSet>;

/// Tag `stream` into the stream set at `cell`, used by the view-level
/// `mark_launch_use` back-reference (see [`crate::cuda::smart_ptrs`]).
///
/// # Safety
///
/// `cell` must point to a live `RefCell<StreamSet>` owned by a `GpuBuffer`
/// that outlives this call. In practice the only caller is `GpuView` /
/// `GpuViewMut`, whose lifetime is bounded by a borrow of the parent
/// `GpuVec` (and hence the parent buffer), so the cell is always live for
/// the duration of the view — the borrow checker guarantees the buffer is
/// not dropped while a view exists. A null `cell` (a view over an empty /
/// placeholder buffer) is a no-op.
///
/// No other thread can hold a borrow concurrently: `GpuBuffer` is `!Sync`,
/// so the cell is only ever touched from the single owning thread.
pub(crate) unsafe fn tag_stream_set(cell: StreamSetRef, stream: CUstream) {
    if cell.is_null() {
        return;
    }
    // SAFETY: caller contract — `cell` is a live cell owned by a buffer
    // that outlives this call.
    (*cell).borrow_mut().insert(stream);
}

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
    /// The set of streams this buffer has been enqueued on (via
    /// [`mark_stream_use`](GpuBuffer::mark_stream_use)). Empty if the buffer
    /// has only ever been touched synchronously, in which case `Drop` skips
    /// the fence. Otherwise `Drop` synchronizes **every** recorded stream
    /// before returning the device pointer to the pool, so no async DMA /
    /// kernel referencing the block can outlive the allocation.
    ///
    /// ## Invariant (closes review findings C-1 and C-2)
    ///
    /// * **C-2 (multi-stream):** we track the full *set*, not just the last
    ///   stream, and fence all of them — see [`StreamSet`].
    /// * **C-1 (kernel launches):** any path that hands `device_ptr()` to a
    ///   kernel launch must record the launch stream here (directly via
    ///   `mark_stream_use`, or through a view's
    ///   [`GpuView::mark_launch_use`](crate::cuda::smart_ptrs::GpuView::mark_launch_use)
    ///   which forwards into this same set). The async DMA helpers on this
    ///   type tag automatically.
    ///
    /// `RefCell` (interior mutability) so the read-only async helpers
    /// (`copy_to_async`, taking `&self`) and the view back-reference can
    /// tag the buffer without forcing every call site onto `&mut self`.
    /// Sound because `GpuBuffer` is `!Sync`, so there is never a concurrent
    /// borrow from another thread.
    used_streams: RefCell<StreamSet>,
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
            used_streams: RefCell::new(StreamSet::default()),
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
            used_streams: RefCell::new(StreamSet::default()),
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
        let _span = tracing::info_span!(
            "transfer",
            direction = "h2d",
            bytes = slice.len() * size_of::<T>(),
        )
        .entered();
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

    /// Allocate a new buffer of `total_len` `T`s, DtoD-copy `prefix_len`
    /// elements from `prefix_src` into the leading rows, then HtoD-upload
    /// `tail` into the trailing rows. Used by the incremental `GpuTable`
    /// cache (batch 5): when `register_batch` appends rows to a table,
    /// the unchanged prefix never re-crosses PCIe — only the new tail does.
    ///
    /// `prefix_len + tail.len()` must equal `total_len`.
    ///
    /// # Safety
    /// `prefix_src` must point to a live device allocation of at least
    /// `prefix_len * size_of::<T>()` bytes. The caller must guarantee
    /// that `prefix_src` is distinct from the (not-yet-existing) new
    /// buffer's pointer.
    pub unsafe fn from_prefix_and_tail(
        total_len: usize,
        prefix_src: crate::cuda::cuda_sys::CUdeviceptr,
        prefix_len: usize,
        tail: &[T],
    ) -> BoltResult<Self> {
        if prefix_len.checked_add(tail.len()) != Some(total_len) {
            return Err(BoltError::Memory(format!(
                "GpuBuffer::from_prefix_and_tail: prefix_len ({}) + tail.len ({}) != total_len ({})",
                prefix_len,
                tail.len(),
                total_len
            )));
        }
        let mut buf = Self::with_capacity(total_len)?;
        // 1. Device-to-device copy of the prefix.
        if prefix_len > 0 {
            // SAFETY: `buf.ptr` was just allocated with capacity for
            // `total_len >= prefix_len` elements; `prefix_src` is a live
            // allocation of at least `prefix_len` `T`s per the caller's
            // contract; the two allocations are distinct (just allocated
            // vs. caller-supplied) so the non-overlap requirement of
            // `cuMemcpyDtoD_v2` holds.
            cuda_sys::memcpy_d2d::<T>(buf.ptr, prefix_src, prefix_len)?;
        }
        // 2. Host-to-device copy of the tail directly into the offset
        //    `prefix_len * size_of::<T>()` bytes.
        if !tail.is_empty() {
            let byte_offset = prefix_len
                .checked_mul(size_of::<T>())
                .ok_or_else(|| {
                    BoltError::Memory(format!(
                        "GpuBuffer::from_prefix_and_tail: offset overflow: {} * {}",
                        prefix_len,
                        size_of::<T>()
                    ))
                })?;
            let tail_dst: cuda_sys::CUdeviceptr = buf.ptr.wrapping_add(byte_offset as u64);
            // SAFETY: `tail_dst` is `buf.ptr + prefix_len` elements in; the
            // buffer has room for `total_len = prefix_len + tail.len()`
            // elements; `tail` is a host slice of `tail.len()` `T`s.
            cuda_sys::memcpy_h2d::<T>(tail_dst, tail.as_ptr(), tail.len())?;
        }
        buf.len = total_len;
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
    ///
    /// # Stream-fence invariant (review findings C-1 / C-2)
    ///
    /// Handing this pointer to a kernel launch (`cuLaunchKernel`) or any
    /// `cuMemcpy*Async` / `cuMemset*Async` makes the launch stream a
    /// *user* of this allocation. If the buffer is then dropped while that
    /// async work is still in flight, `Drop` must fence the stream before
    /// the block returns to the pool — otherwise the pool can re-issue the
    /// same physical address to an unrelated allocation mid-flight (silent
    /// use-after-free / data corruption).
    ///
    /// `device_ptr()` itself **cannot** record the stream because it does
    /// not know which stream the launch will target. Recording therefore
    /// happens at the point that *does* know the stream:
    ///
    /// * **Async DMA helpers on this type** (`copy_from_async`,
    ///   `copy_to_async`, `copy_to_slice_async`, `zeros_async`) call
    ///   [`mark_stream_use`](Self::mark_stream_use) automatically.
    /// * **Kernel launches** go through the launch glue in
    ///   [`crate::exec::launch`]. As of review finding V-1 the launch entry
    ///   points ([`launch_1d`](crate::exec::launch::launch_1d) /
    ///   [`launch_with_geometry`](crate::exec::launch::launch_with_geometry))
    ///   tag the launch stream into every buffer whose view was pushed into
    ///   the `KernelArgs`, *centrally* — so a kernel launched off
    ///   `device_ptr()` / a view's `device_ptr()` records its stream by
    ///   construction, with no per-call-site bookkeeping. The view's
    ///   [`mark_launch_use`](crate::cuda::smart_ptrs::GpuView::mark_launch_use)
    ///   remains available for any launch path that bypasses `KernelArgs`.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Record that this buffer was enqueued on `stream`. `Drop` will
    /// synchronize **every** recorded stream before returning the device
    /// pointer to the pool, so async DMA / kernels referencing
    /// `device_ptr()` cannot outlive the allocation.
    ///
    /// ## Invariant (review findings C-1 / C-2; supersedes the C13 hook)
    ///
    /// This is the single tagging hook that keeps the pool block alive long
    /// enough. It is idempotent per stream — recording the same stream
    /// twice is a no-op (the set dedups), so over-calling is always safe.
    ///
    /// * **C-2:** every distinct stream is remembered and fenced (not just
    ///   the last one), so an independent op on an earlier stream can no
    ///   longer race the recycled block.
    /// * **C-1:** the async helpers on this type (`copy_from_async`,
    ///   `copy_to_async`, `copy_to_slice_async`, `zeros_async`) call this
    ///   automatically after every `cuMemcpy*Async` / `cuMemset*Async`.
    ///   `cuLaunchKernel` is tagged centrally at the launch entry point
    ///   (review finding V-1): the `KernelArgs` machinery in
    ///   [`crate::exec::launch`] retains each pushed view's stream-set
    ///   back-reference and tags the launch stream into it after the launch,
    ///   so launch sites need no explicit call. The view-level
    ///   `mark_launch_use` remains for launch paths that bypass `KernelArgs`.
    #[allow(clippy::not_unsafe_ptr_arg_deref)] // CUstream is forwarded, not deref'd
    pub fn mark_stream_use(&self, stream: CUstream) {
        // `&self` (not `&mut self`) because the buffer's read-only async
        // helpers like `copy_to_async`, and the view back-reference used by
        // `GpuView::mark_launch_use`, need to tag the stream too — and the
        // `RefCell` makes that sound (`GpuBuffer: !Sync`). The borrow is
        // held only for the duration of this `insert` call, so it cannot
        // overlap the `borrow()` taken in `Drop` (the buffer is not being
        // dropped concurrently with its own methods).
        self.used_streams.borrow_mut().insert(stream);
    }

    /// Number of distinct streams currently recorded for this buffer.
    ///
    /// Test / diagnostic hook: lets host-only tests assert the stream-set
    /// bookkeeping (dedup, multi-stream accumulation) without a GPU.
    #[doc(hidden)]
    pub fn recorded_stream_count(&self) -> usize {
        self.used_streams.borrow().len()
    }

    /// Raw pointer to this buffer's stream-set cell, for the view
    /// back-reference (see [`crate::cuda::smart_ptrs`]). The pointer is
    /// only ever used to call [`mark_stream_use`](Self::mark_stream_use)-
    /// equivalent tagging through a `GpuView` whose lifetime is bounded by
    /// a borrow of this buffer, so the cell is guaranteed to outlive any
    /// dereference. Crate-internal; never handed to FFI.
    pub(crate) fn used_streams_cell(&self) -> StreamSetRef {
        &self.used_streams as StreamSetRef
    }

    /// Copy the buffer's contents back to a fresh host `Vec<T>`.
    pub fn to_vec(&self) -> BoltResult<Vec<T>> {
        let _span = tracing::info_span!(
            "transfer",
            direction = "d2h",
            bytes = self.len * size_of::<T>(),
        )
        .entered();
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
        let _span = tracing::info_span!(
            "transfer",
            direction = "h2d",
            bytes = src.len() * size_of::<T>(),
        )
        .entered();
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
        // Tag the stream so `Drop` fences before recycling our block.
        self.mark_stream_use(stream);
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
        let _span = tracing::info_span!(
            "transfer",
            direction = "d2h",
            bytes = self.len * size_of::<T>(),
        )
        .entered();
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
        // Tag the stream so `Drop` fences before recycling our block.
        self.mark_stream_use(stream);
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
            // Tag the stream so `Drop` fences before recycling the block —
            // a `cuMemsetD8Async` that is still in flight when `buf` drops
            // would otherwise scribble onto the next pool consumer.
            buf.mark_stream_use(stream);
        }
        buf.len = len;
        Ok(buf)
    }

    /// Copy the buffer's contents into `dst`. Errors if lengths differ.
    pub fn copy_to_slice(&self, dst: &mut [T]) -> BoltResult<()> {
        let _span = tracing::info_span!(
            "transfer",
            direction = "d2h",
            bytes = self.len * size_of::<T>(),
        )
        .entered();
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

/// Signature of the per-stream fence used by [`GpuBuffer`]'s `Drop`. Aliased
/// so a host-only test can swap in a recording stub via [`drop_fence_with`],
/// mirroring the `init_with` / `CuInitFn` mockable pattern in `cuda_sys`.
/// Returns the raw `CUresult` so the production hook can warn on failure.
type StreamFenceFn = fn(CUstream) -> crate::cuda::cuda_sys::CUresult;

/// Production stream fence: forwards to `cuStreamSynchronize`.
///
/// SAFETY: `stream` is an opaque handle previously recorded via
/// `mark_stream_use` (which forwards exactly what the caller passed to the
/// async FFI). It is never dereferenced here, only handed to the driver.
/// The driver tolerates synchronize-on-a-completed-stream as a cheap no-op,
/// so calling it even after the work has finished is safe.
///
/// Under `--features cuda-stub` `cuStreamSynchronize` is the stub shim that
/// returns `CUDA_ERROR_STUB`; the caller below treats any non-success rc as
/// "could not fence" and logs, which is the correct conservative behaviour
/// for a build with no GPU.
#[allow(clippy::not_unsafe_ptr_arg_deref)] // handle forwarded, not deref'd
fn real_stream_fence(stream: CUstream) -> crate::cuda::cuda_sys::CUresult {
    unsafe { cuda_sys::cuStreamSynchronize(stream) }
}

/// Fence every stream in `streams` via `fence`, logging any non-success rc.
/// Factored out of `Drop` so it can be unit-tested with a recording stub
/// (counting how many distinct streams get fenced) on a host with no GPU.
///
/// The set is already deduped by [`StreamSet::insert`], so each distinct
/// stream is fenced exactly once.
fn fence_all_streams(streams: &StreamSet, fence: StreamFenceFn) {
    for &stream in &streams.streams {
        let rc = fence(stream);
        if rc != cuda_sys::CUDA_SUCCESS {
            log::warn!(
                "craton-bolt: GpuBuffer::Drop stream fence returned {} \
                 (buffer dropped while a pending op may still reference it; \
                 pool block may be recycled before the driver is done with it)",
                rc
            );
        }
    }
}

/// Test seam: when set, `Drop` fences through this stub instead of the real
/// `cuStreamSynchronize`. Host-only tests install a recorder here to assert
/// the *number of distinct streams fenced* without a GPU, then clear it.
///
/// `thread_local` + `Cell` keeps it `!Sync` and avoids any cross-test
/// interference under the default single-threaded test harness path that
/// touches it. Production never sets it, so the hot path is one `Cell`
/// read returning `None`.
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
/// installed, else the real `cuStreamSynchronize`.
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

/// # SAFETY — async DMA / kernel-launch lifetime hazard (review findings C-1, C-2)
///
/// `Drop` fences **every** stream recorded in `used_streams` (a deduped
/// [`StreamSet`]) before returning the device pointer to the pool. The
/// fence is per-stream (`cuStreamSynchronize`), not a blanket
/// `cuCtxSynchronize`, so unrelated streams keep running.
///
/// ## Why a set, and why this closes C-1 and C-2
///
/// * **C-2 (multi-stream race).** The previous design fenced only the
///   *last* tagged stream. A buffer enqueued on stream A then stream B
///   fenced only B; an independent op still running on A could then race
///   the recycled pool block once the address was re-issued. Fencing the
///   whole set removes that race: no stream that ever touched the block
///   can still be in flight when it returns to the pool.
///
/// * **C-1 (kernel launches went untracked).** `device_ptr()` /
///   `GpuView::device_ptr()` hand the raw pointer to kernel-launch FFI.
///   Nothing in `device_ptr()` can know the launch stream, so the launch
///   *entry point* tags it. As of review finding V-1 this is done centrally
///   in [`launch_1d`](crate::exec::launch::launch_1d) /
///   [`launch_with_geometry`](crate::exec::launch::launch_with_geometry):
///   `KernelArgs` retains each pushed view's stream-set back-reference and
///   the launch tags the stream into all of them after `cuLaunchKernel`, so
///   every buffer a kernel touches records that kernel's stream by
///   construction (the view's `mark_launch_use` remains for launch paths
///   that bypass `KernelArgs`). With the launch stream in the set, dropping
///   a buffer that a kernel is still reading fences that kernel's stream
///   before recycling the block.
///
/// ## What is tagged automatically
///
/// The async helpers on this type call `mark_stream_use` after enqueueing:
///
/// * [`GpuBuffer::copy_from_async`] (host→device DMA)
/// * [`GpuBuffer::copy_to_async`] / [`GpuBuffer::copy_to_slice_async`]
///   (device→host DMA)
/// * [`GpuBuffer::zeros_async`] (device-side `cuMemsetD8Async`)
///
/// Kernel launches are tagged centrally at the launch entry point (review
/// finding V-1): [`launch_1d`](crate::exec::launch::launch_1d) and
/// [`launch_with_geometry`](crate::exec::launch::launch_with_geometry) record
/// the launch stream into the set of every buffer whose view was pushed into
/// the `KernelArgs`, so a kernel that reads/writes `device_ptr()` records its
/// stream by construction — no per-call-site bookkeeping, and the guarantee
/// does not rely on the post-launch `synchronize()` staying in place. Launch
/// paths that bypass `KernelArgs` should call the view's `mark_launch_use`.
/// Over-tagging is harmless — the set dedups.
///
/// ## Caller discipline (still preferred)
///
/// Even with the fence, calling `cuStreamSynchronize` (or
/// [`crate::exec::launch::CudaStream::synchronize`]) before the buffer goes
/// out of scope is cheaper than letting `Drop` discover the fence is needed
/// — the synchronize on a *completed* stream is a fast no-op, but the
/// synchronize on an *in-flight* one stalls the dropping thread.
///
/// ## Follow-up (NOT implemented here — reviewer's recommended design)
///
/// The fully-structural fix the reviewer recommended is an event-based
/// *pending-free* pool: on `mark_stream_use` record a `cuEventRecord` on the
/// stream, and on `Drop` hand `(ptr, event)` to the pool's deferred-free
/// list instead of fencing inline. The pool then reclaims the block only
/// once `cuEventQuery` reports the event complete (polled on the next
/// `alloc`/`free`), so a buffer release never *stalls* the dropping thread —
/// it only defers reuse. That is a larger change (new pool state machine,
/// event lifecycle management) and is deliberately left as a TODO; the
/// blanket per-stream sync here is the conservative, correctness-first
/// interim that needs no new pool machinery.
impl<T: Pod> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        // Fence every stream this buffer was enqueued on (DMA *or* kernel
        // launch) before the block goes back to the pool. Skipping this
        // would let the pool re-issue the same physical address to an
        // unrelated allocation while a `cuMemcpy*Async` / `cuMemset*Async`
        // / kernel was still touching it — silent corruption, not an error.
        //
        // The borrow is short-lived and cannot alias any other borrow:
        // `Drop` runs with exclusive ownership of `self`, so no other
        // `&self` method (which is what takes `borrow_mut` in
        // `mark_stream_use`) can be executing concurrently.
        let streams = self.used_streams.borrow();
        if !streams.is_empty() {
            fence_all_streams(&streams, current_drop_fence());
        }
        drop(streams);
        // Return the block to the pool rather than the driver. The pool's
        // `drain` (run on process shutdown via `Drop` on the static) will
        // eventually hand it back to `cuMemFree_v2` — via the cudarc
        // backend when `--features cudarc` is active, or via the
        // hand-rolled FFI otherwise. Either way the call site here is
        // backend-agnostic because the pool stores raw `CUdeviceptr`s.
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
    /// The set of streams this buffer's pinned pages have been enqueued on
    /// (via [`PinnedHostBuffer::mark_stream_use`]). Callers that enqueue
    /// async work referencing `self.ptr` (H2D / D2H DMA, kernels that touch
    /// the pinned region, etc.) call that helper after enqueueing so `Drop`
    /// can fence against every in-flight transfer before `cuMemFreeHost`
    /// reclaims the pages. See the `Drop` impl below for the rationale.
    ///
    /// ## Why a *set*, not a single "last stream" (review finding V-2)
    ///
    /// The pre-V-2 design tracked only the most-recently-used stream
    /// (`Cell<Option<CUstream>>`) and fenced just that one at `Drop`. A
    /// pinned buffer DMA'd on stream A and then stream B fenced only B, so an
    /// independent transfer still draining on A could read/write the
    /// page-locked region *after* `cuMemFreeHost` handed the pages back to
    /// the kernel — a host-side use-after-free. This is the exact host-side
    /// analogue of the device-side multi-stream race C-2 that `GpuBuffer`
    /// already fixed, so we reuse the same [`StreamSet`] machinery here:
    /// track every distinct stream and fence all of them at `Drop`.
    ///
    /// `RefCell<StreamSet>` (mirroring `GpuBuffer::used_streams`) so
    /// `mark_stream_use` can take `&self`: typical async copy helpers borrow
    /// the pinned buffer shared (`as_ptr()` reads through `&self`), and
    /// forcing them to `&mut self` here would ripple needlessly through the
    /// call sites. The cell is single-threaded (the struct is `!Sync`), so
    /// no atomic is required.
    used_streams: RefCell<StreamSet>,
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
                used_streams: RefCell::new(StreamSet::default()),
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
            used_streams: RefCell::new(StreamSet::default()),
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
        // `checked_mul` mirrors `GpuBuffer::byte_len` — never wrap silently;
        // if `self.len * size_of::<T>()` cannot fit `usize`, that's a bug we
        // want to surface, not a number we want to return.
        self.len
            .checked_mul(size_of::<T>())
            .expect("PinnedHostBuffer::byte_len overflow")
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
    /// that references this buffer's pinned host pages. `Drop` will
    /// `cuStreamSynchronize` against **every** recorded stream before
    /// calling `cuMemFreeHost`, so the driver can't still be DMA-ing into
    /// freed pages on any stream that ever touched them.
    ///
    /// Mirrors the `mark_stream_use` contract on `GpuBuffer` (review
    /// findings C-2 / V-2). Callers MUST invoke this on the pinned source of
    /// `cuMemcpyHtoDAsync_v2`, the pinned destination of
    /// `cuMemcpyDtoHAsync_v2`, or any kernel parameter that holds
    /// `as_ptr()` / `as_mut_ptr()`, after enqueueing the async work.
    ///
    /// ## Every distinct stream is remembered (review finding V-2)
    ///
    /// Unlike the pre-V-2 design (which kept only the last stream), this
    /// accumulates the full deduped [`StreamSet`], so a pinned buffer used
    /// across multiple streams fences all of them at `Drop`. The caller no
    /// longer has to arrange cross-stream barriers purely to make a single
    /// final-stream sync sufficient — though doing so is still cheaper, as
    /// the eventual per-stream sync on a completed stream is a no-op.
    /// Recording the same stream twice is a no-op (the set dedups).
    ///
    /// Takes `&self` (interior mutability via `RefCell`) so call sites
    /// holding a shared borrow — typical for an async helper that only
    /// reads `as_ptr()` — don't have to be rewritten to `&mut self`.
    /// The cell is single-threaded (the struct is `!Sync`) so no atomic
    /// is required, and the borrow is held only for the `insert` call so it
    /// cannot overlap the `borrow()` taken in `Drop`.
    // `stream` is an opaque CUstream handle; we just store it. No deref,
    // no FFI. The outer fn stays safe to keep the call-site ergonomics
    // identical to the analogous helper on `GpuBuffer`.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    #[inline]
    pub fn mark_stream_use(&self, stream: CUstream) {
        self.used_streams.borrow_mut().insert(stream);
    }

    /// Number of distinct streams currently recorded for this buffer.
    ///
    /// Test / diagnostic hook (mirrors
    /// [`GpuBuffer::recorded_stream_count`]): lets host-only tests assert the
    /// stream-set bookkeeping (dedup, multi-stream accumulation) without a
    /// GPU. Production code rarely needs it because `Drop` consumes the set
    /// directly.
    #[doc(hidden)]
    #[inline]
    pub fn recorded_stream_count(&self) -> usize {
        self.used_streams.borrow().len()
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
        // `checked_mul` so an adversarial `new_len` cannot wrap the product
        // before the bounds check — otherwise `as_slice()` would `from_raw_parts`
        // over arbitrary memory beyond our pinned allocation.
        let new_bytes = new_len
            .checked_mul(std::mem::size_of::<T>())
            .expect("set_len: new_len * size_of::<T>() overflowed usize");
        assert!(
            new_bytes <= self.byte_len,
            "set_len: requested {new_len} elements ({new_bytes} bytes) exceeds buffer byte_len {}",
            self.byte_len
        );
        self.len = new_len;
    }
}

impl<T: Pod> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            // Zero-length / never-allocated path: nothing to free and
            // nothing to fence against. The stream set is meaningless
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
        // ## Fence EVERY recorded stream (review finding V-2)
        //
        // We synchronize each distinct stream in `used_streams`, not just
        // the last one. A pinned buffer DMA'd on stream A then stream B
        // could otherwise have an independent transfer still draining on A
        // when `cuMemFreeHost` runs; fencing the whole set closes that
        // host-side multi-stream UAF, exactly as `GpuBuffer`'s `Drop` does
        // for the device side. The set is deduped, so each stream is fenced
        // at most once.
        //
        // The borrow is short-lived and cannot alias: `Drop` runs with
        // exclusive ownership of `self`, so no `&self` method (which is what
        // takes `borrow_mut` in `mark_stream_use`) can run concurrently.
        //
        // TODO(perf): a blanket per-stream sync is the safe default but it
        // serialises every pinned-buffer release against each recorded
        // stream's full queue, which can stall pipelined H2D / kernel /
        // D2H overlap if the caller has already arranged a cross-stream
        // barrier (event-record + wait) elsewhere. A future optimisation
        // could replace this with a per-buffer `cuEventRecord` +
        // `cuEventSynchronize` on a tiny event attached at
        // `mark_stream_use` time, so we wait only on the specific point in
        // each stream's history that actually touched this buffer, not the
        // entire trailing tail of the stream.
        let streams = self.used_streams.borrow();
        for &stream in &streams.streams {
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
        drop(streams);
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
        // block: the null-ptr short-circuit in `Drop` returns before the
        // fence loop, so an empty `used_streams` set is never even reached;
        // if that short-circuit failed and we hit the FFI, the cuda-stub
        // build would panic with `CUDA_ERROR_STUB`.
        for _ in 0..8 {
            let _b: PinnedHostBuffer<i64> =
                PinnedHostBuffer::new(0).expect("zero-len alloc");
        }
    }

    #[test]
    fn mark_stream_use_is_preserved_across_method_calls() {
        // The stream-tracking set must survive arbitrary `&self` method
        // calls — only an explicit `mark_stream_use` (or Drop) should
        // touch it. We use fabricated non-null sentinel pointers for the
        // stream handles; we never deref them.
        //
        // This buffer is empty, so its `Drop` will skip the FFI free and
        // also skip the stream sync (the ptr-null short-circuit runs
        // first). That keeps the test driver-free on hosts without CUDA.
        let buf: PinnedHostBuffer<u32> =
            PinnedHostBuffer::new(0).expect("zero-len alloc");
        let fake_stream: CUstream = 0xDEAD_BEEF_usize as CUstream;
        assert_eq!(buf.recorded_stream_count(), 0);
        buf.mark_stream_use(fake_stream);
        assert_eq!(buf.recorded_stream_count(), 1);

        // Hit a few `&self`-receiver methods and re-check the set is intact.
        let _ = buf.len();
        let _ = buf.is_empty();
        let _ = buf.byte_len();
        let _ = buf.as_ptr();
        let _ = buf.as_slice();
        assert_eq!(
            buf.recorded_stream_count(),
            1,
            "method calls must not clobber the recorded stream set"
        );

        // V-2: re-recording the same stream dedups (no growth), and a new
        // stream accumulates rather than replacing the previous one.
        buf.mark_stream_use(fake_stream);
        assert_eq!(buf.recorded_stream_count(), 1, "re-record must dedup");
        let other_stream: CUstream = 0xCAFE_F00D_usize as CUstream;
        buf.mark_stream_use(other_stream);
        assert_eq!(
            buf.recorded_stream_count(),
            2,
            "V-2: distinct streams must accumulate, not overwrite"
        );

        // A null stream is a perfectly legal value (the default stream is
        // the null handle) and counts as a distinct member of the set.
        buf.mark_stream_use(ptr::null_mut());
        assert_eq!(buf.recorded_stream_count(), 3);
    }
}

#[cfg(test)]
mod stream_set_tests {
    //! Host-only tests for the C-1 / C-2 stream-set bookkeeping and the
    //! `Drop`-time fence-all-streams logic. None of these touch the CUDA
    //! driver: they exercise the pure bookkeeping (`StreamSet` dedup,
    //! `GpuBuffer::mark_stream_use` accumulation) and the fence dispatch via
    //! the mockable `drop_fence_with` seam, which records how many streams
    //! were fenced without calling `cuStreamSynchronize`. They run on hosts
    //! with no GPU and under `--features cuda-stub`.
    use super::*;
    use crate::cuda::cuda_sys::{CUresult, CUstream, CUDA_SUCCESS};
    use std::cell::RefCell as StdRefCell;
    use std::ptr;

    fn fake_stream(bits: usize) -> CUstream {
        bits as CUstream
    }

    // ---- StreamSet pure logic --------------------------------------------

    #[test]
    fn stream_set_dedups_repeated_handles() {
        let mut s = StreamSet::default();
        assert!(s.is_empty());
        let a = fake_stream(0x1000);
        let b = fake_stream(0x2000);
        s.insert(a);
        s.insert(a); // duplicate — must not grow
        s.insert(b);
        s.insert(a); // duplicate again
        assert_eq!(s.len(), 2, "set must hold exactly the two distinct streams");
    }

    #[test]
    fn stream_set_treats_null_as_a_real_handle() {
        // The default CUDA stream is the null handle; it must be tracked
        // like any other (and deduped against itself).
        let mut s = StreamSet::default();
        s.insert(ptr::null_mut());
        s.insert(ptr::null_mut());
        assert_eq!(s.len(), 1);
    }

    // ---- GpuBuffer::mark_stream_use accumulation (C-2) -------------------

    #[test]
    fn mark_stream_use_accumulates_distinct_streams() {
        // Synthesize a buffer without touching the driver: `empty()` has a
        // null `ptr`, so its `Drop` skips both the fence and the pool free.
        let buf: GpuBuffer<u32> = GpuBuffer::empty();
        assert_eq!(buf.recorded_stream_count(), 0);

        let a = fake_stream(0xA);
        let b = fake_stream(0xB);
        buf.mark_stream_use(a);
        buf.mark_stream_use(a); // dedup
        assert_eq!(buf.recorded_stream_count(), 1);
        buf.mark_stream_use(b);
        assert_eq!(
            buf.recorded_stream_count(),
            2,
            "C-2: every distinct stream must be remembered, not just the last"
        );
    }

    // ---- Drop fences ALL recorded streams (C-2), via the mock seam -------

    thread_local! {
        static FENCED: StdRefCell<Vec<CUstream>> = const { StdRefCell::new(Vec::new()) };
    }

    /// Recording stub installed via `drop_fence_with`: logs each stream it
    /// is asked to fence and returns success. Never calls the driver.
    fn recording_fence(stream: CUstream) -> CUresult {
        FENCED.with(|f| f.borrow_mut().push(stream));
        CUDA_SUCCESS
    }

    #[test]
    fn drop_fences_every_recorded_stream_exactly_once() {
        FENCED.with(|f| f.borrow_mut().clear());

        // A buffer with a non-null ptr would try to `POOL.free` on drop; to
        // keep this host-only we operate on `empty()` (null ptr) but drive
        // the fence path directly through `fence_all_streams`, which is the
        // exact code `Drop` runs once it has the borrowed set. This isolates
        // the "fence all, deduped" guarantee from the pool free.
        let buf: GpuBuffer<u8> = GpuBuffer::empty();
        let a = fake_stream(0x11);
        let b = fake_stream(0x22);
        buf.mark_stream_use(a);
        buf.mark_stream_use(b);
        buf.mark_stream_use(a); // duplicate — must not produce a 3rd fence

        drop_fence_with(recording_fence, || {
            let set = buf.used_streams.borrow();
            fence_all_streams(&set, current_drop_fence());
        });

        FENCED.with(|f| {
            let fenced = f.borrow();
            assert_eq!(
                fenced.len(),
                2,
                "C-2: Drop must fence each distinct recorded stream exactly once"
            );
            assert!(fenced.contains(&a) && fenced.contains(&b));
        });
    }

    #[test]
    fn empty_stream_set_fences_nothing() {
        FENCED.with(|f| f.borrow_mut().clear());
        let buf: GpuBuffer<u8> = GpuBuffer::empty();
        drop_fence_with(recording_fence, || {
            let set = buf.used_streams.borrow();
            if !set.is_empty() {
                fence_all_streams(&set, current_drop_fence());
            }
        });
        FENCED.with(|f| assert!(f.borrow().is_empty()));
    }

    // ---- C-1: view back-reference forwards into the parent's set ---------

    #[test]
    fn tag_stream_set_forwards_into_parent_cell() {
        // Emulate what `GpuView::mark_launch_use` does: tag the parent
        // buffer's stream set through the raw back-reference. After tagging,
        // the parent buffer must see the launch stream in its set — that is
        // exactly what makes `Drop` fence a kernel's stream (C-1).
        let buf: GpuBuffer<i64> = GpuBuffer::empty();
        let cell = buf.used_streams_cell();
        let launch_stream = fake_stream(0xC1);
        // SAFETY: `cell` points at `buf`'s live stream-set cell; `buf`
        // outlives this call.
        unsafe {
            tag_stream_set(cell, launch_stream);
            tag_stream_set(cell, launch_stream); // dedup through the cell
        }
        assert_eq!(
            buf.recorded_stream_count(),
            1,
            "C-1: a launch tagged via the view back-reference must land in \
             the parent buffer's stream set (deduped)"
        );
    }

    #[test]
    fn tag_stream_set_null_cell_is_noop() {
        // A view over an empty/placeholder buffer carries a null back-ref;
        // tagging it must be a silent no-op, not a deref.
        unsafe {
            tag_stream_set(ptr::null(), fake_stream(0x1));
        }
        // Reaching here without UB / panic is the assertion.
    }
}
