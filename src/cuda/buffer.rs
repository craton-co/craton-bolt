// SPDX-License-Identifier: Apache-2.0

//! Arrow-compatible columnar GPU storage primitives.
//!
//! `GpuBuffer<T>` is a raw, untyped (modulo `T`'s element size) device
//! allocation that mirrors Arrow's 64-byte CPU alignment. It is the low-level
//! primitive on which the typed, lifetime-tracked `GpuVec<T>` (Step 3) will be
//! built.

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

    /// Async H2D upload from `slice` into the existing buffer on `stream`.
    ///
    /// The buffer must already have `capacity() >= slice.len()`. Updates
    /// `len` to `slice.len()`. Does NOT synchronize — the caller must
    /// synchronize `stream` before dropping (or aliasing) `slice`.
    ///
    /// Stage-3 overlap path: pair this with a [`PinnedHostBuffer`] source
    /// to get a true DMA transfer that overlaps with kernel work on the
    /// same stream.
    pub fn copy_from_async(&mut self, slice: &[T], stream: CUstream) -> BoltResult<()> {
        if slice.len() > self.capacity {
            return Err(BoltError::Memory(format!(
                "GpuBuffer::copy_from_async: source length {} > capacity {}",
                slice.len(),
                self.capacity
            )));
        }
        if !slice.is_empty() {
            // SAFETY: `self.ptr` was allocated with capacity for `self.capacity`
            // elements of `T` (>= slice.len() checked above), and `slice` is a
            // valid read source for that many. The caller is responsible for
            // keeping `slice` alive until `stream` is synchronized.
            unsafe {
                cuda_sys::memcpy_h2d_async::<T>(self.ptr, slice.as_ptr(), slice.len(), stream)?;
            }
        }
        self.len = slice.len();
        Ok(())
    }

    /// Async D2H download into `dst` on `stream`.
    ///
    /// Errors if `dst.len() != self.len`. Does NOT synchronize — the caller
    /// must synchronize `stream` before reading `dst`. Pair with a
    /// [`PinnedHostBuffer`] destination for a true DMA transfer.
    pub fn copy_to_slice_async(&self, dst: &mut [T], stream: CUstream) -> BoltResult<()> {
        if dst.len() != self.len {
            return Err(BoltError::Memory(format!(
                "GpuBuffer::copy_to_slice_async length mismatch: dst={}, buffer={}",
                dst.len(),
                self.len
            )));
        }
        if self.len > 0 {
            // SAFETY: `dst` is valid for writes of `self.len` elements (checked
            // above), and `self.ptr` is a live device allocation of at least
            // that many `T`s. Caller is responsible for keeping `dst` alive
            // until `stream` is synchronized.
            unsafe {
                cuda_sys::memcpy_d2h_async::<T>(dst.as_mut_ptr(), self.ptr, self.len, stream)?;
            }
        }
        Ok(())
    }

    /// Allocate `len` elements and zero them via `cuMemsetD8Async` on `stream`.
    ///
    /// Stage-3 counterpart to [`zeros`]. The memset is enqueued on `stream`
    /// and the caller must synchronize before launching any kernel that
    /// reads these bytes on a *different* stream. Kernels enqueued on the
    /// same `stream` after this call see a fully-zeroed buffer without an
    /// explicit sync.
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
            // SAFETY: `buf.ptr` was just allocated with at least `byte_len`
            // bytes of capacity (rounded up to ARROW_ALIGNMENT).
            unsafe {
                cuda_sys::memset_d8_async(buf.ptr, 0, byte_len, stream)?;
            }
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

// ---------------------------------------------------------------------------
// Stage 3: PinnedHostBuffer<T>
//
// Owns a page-locked (pinned) host allocation. Pinned memory is required to
// get a *true* asynchronous DMA H2D/D2H — pageable host memory forces the
// driver to stage through a hidden pinned bounce buffer, which serialises
// the transfer with the launch.
//
// The driver-side allocation comes from `cuMemAllocHost_v2` / is freed by
// `cuMemFreeHost`. Allocations are limited (the kernel's pinned-pool is
// global), so use only for the *final* H2D upload sources and D2H download
// targets — not for every intermediate buffer.
// ---------------------------------------------------------------------------

/// Page-locked (pinned) host buffer of `T`. Used as the host side of an
/// async H2D / D2H so the driver can DMA directly without a bounce copy.
///
/// `len()` is the number of valid `T` elements (set by the caller via
/// [`PinnedHostBuffer::set_len`] after an async D2H, or by
/// [`PinnedHostBuffer::new`] which initialises it to the full capacity).
///
/// `Drop` returns the pinned pages to the driver. Do NOT call drop while
/// an async copy that targets this buffer is still in flight — the caller
/// must `stream.synchronize()` first.
pub struct PinnedHostBuffer<T: Pod> {
    /// Raw host pointer returned by `cuMemAllocHost_v2`.
    ptr: *mut T,
    /// Logical element count (`<= capacity`).
    len: usize,
    /// Allocated capacity in `T` elements.
    capacity: usize,
}

impl<T: Pod> PinnedHostBuffer<T> {
    /// Allocate `capacity` elements of pinned host memory. `len()` is
    /// initialised to `capacity` — the buffer is treated as a fully-sized
    /// slice the moment it's allocated (its contents are still uninitialised
    /// bytes, but `T: Pod` makes that defined behaviour for read).
    pub fn new(capacity: usize) -> BoltResult<Self> {
        if capacity == 0 {
            // Zero-element buffer: hand out a non-null but unique sentinel
            // so `as_ptr` is well-defined. We round to one byte's worth so
            // `cuMemFreeHost` has something real to release; the alternative
            // (skipping the alloc) would force a special case in `Drop`.
            let bytes = size_of::<T>().max(1);
            // SAFETY: we forward a single, non-aliased allocation request
            // to the driver and store the returned pointer in `self`.
            // `Drop` is the unique releaser.
            let raw = unsafe { cuda_sys::mem_alloc_host(bytes)? };
            return Ok(Self {
                ptr: raw as *mut T,
                len: 0,
                capacity: 0,
            });
        }
        let bytes = capacity.checked_mul(size_of::<T>()).ok_or_else(|| {
            BoltError::Memory(format!(
                "PinnedHostBuffer::new size overflow: {} * {}",
                capacity,
                size_of::<T>()
            ))
        })?;
        // SAFETY: same as above — we own this allocation outright.
        let raw = unsafe { cuda_sys::mem_alloc_host(bytes)? };
        Ok(Self {
            ptr: raw as *mut T,
            len: capacity,
            capacity,
        })
    }

    /// Number of valid `T` elements.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Allocated capacity in `T` elements.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Raw host pointer; valid for `len()` `T`s.
    #[inline]
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Raw mutable host pointer; valid for `len()` `T`s.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Borrow as a `&[T]` slice of `len()` elements.
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: `self.ptr` is a live pinned allocation of at least `len`
        // elements; lifetime is tied to `&self`. `T: Pod` makes the bytes
        // safely readable even if uninitialised.
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
    }

    /// Borrow as a `&mut [T]` slice of `len()` elements.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 {
            return &mut [];
        }
        // SAFETY: see `as_slice`; the `&mut` borrow ensures no aliasing.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Override the logical length — used after an async D2H that filled
    /// fewer than `capacity()` elements.
    ///
    /// # Safety
    /// `new_len` must be `<= capacity()` and the first `new_len` elements
    /// must be initialised (e.g. by a completed async D2H).
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity);
        self.len = new_len;
    }
}

impl<T: Pod> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            return;
        }
        // SAFETY: `self.ptr` was returned by `cuMemAllocHost_v2` in `new()`
        // and has not been freed since. The caller's API contract requires
        // synchronising any in-flight async copy before dropping.
        unsafe {
            if let Err(e) = cuda_sys::mem_free_host(self.ptr as *mut libc::c_void) {
                log::warn!(
                    "craton-bolt: PinnedHostBuffer drop: cuMemFreeHost failed: {}",
                    e
                );
            }
        }
    }
}

// SAFETY: a pinned host pointer is just a u64 worth of address; sending it
// between threads is no more dangerous than sending a `Box<[u8]>`. The
// driver does not require thread affinity on pinned memory.
unsafe impl<T: Pod> Send for PinnedHostBuffer<T> {}
// Intentionally NOT `Sync`: concurrent mutation through shared references
// would race on host memory.

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

    // -------- Stage-3 PinnedHostBuffer + async API ------------------------

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn pinned_host_buffer_round_trip_via_async() {
        // End-to-end exercise of the Stage-3 path: pinned source ->
        // async H2D -> async D2H -> pinned sink, with the same bytes
        // surviving the round trip. Uses a fresh stream so the test
        // doesn't accidentally lean on the default-stream serialization.
        use crate::cuda::cuda_sys::{cuStreamCreate, cuStreamDestroy_v2, cuStreamSynchronize, CUstream, check, CUDA_SUCCESS};
        crate::cuda::cuda_sys::init().expect("init cuda");
        let _ctx = crate::cuda::cuda_sys::CudaContext::new(0).expect("ctx");

        let mut stream: CUstream = std::ptr::null_mut();
        unsafe {
            check(cuStreamCreate(&mut stream, 0)).expect("stream create");
        }

        let n = 257usize; // odd size that isn't a clean alignment multiple
        let mut src = PinnedHostBuffer::<i64>::new(n).expect("alloc src");
        for (i, slot) in src.as_mut_slice().iter_mut().enumerate() {
            *slot = (i as i64) * 3 - 7;
        }

        let mut dev = GpuBuffer::<i64>::with_capacity(n).expect("dev alloc");
        dev.copy_from_async(src.as_slice(), stream).expect("h2d");

        let mut dst = PinnedHostBuffer::<i64>::new(n).expect("alloc dst");
        dev.copy_to_slice_async(dst.as_mut_slice(), stream).expect("d2h");

        unsafe {
            assert_eq!(cuStreamSynchronize(stream), CUDA_SUCCESS, "sync");
        }

        // The pinned destination should now exactly mirror the source.
        for (i, (&a, &b)) in src.as_slice().iter().zip(dst.as_slice().iter()).enumerate() {
            assert_eq!(a, b, "mismatch at index {}", i);
        }

        unsafe {
            let _ = cuStreamDestroy_v2(stream);
        }
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn zeros_async_matches_zeros() {
        // `zeros_async` on the NULL stream must produce the same all-zero
        // contents as the synchronous `zeros`. We compare via `to_vec`
        // (sync D2H) so the test exercises the Stage-3 alloc but not the
        // Stage-3 D2H — that's covered separately above.
        crate::cuda::cuda_sys::init().expect("init cuda");
        let _ctx = crate::cuda::cuda_sys::CudaContext::new(0).expect("ctx");

        let n = 128usize;
        // NULL stream — the synchronize is implicit when we call to_vec
        // (cuMemcpyDtoH_v2 on the NULL stream waits for all prior work).
        let buf = GpuBuffer::<i32>::zeros_async(n, std::ptr::null_mut()).expect("async zeros");
        let host = buf.to_vec().expect("d2h");
        assert_eq!(host.len(), n);
        assert!(host.iter().all(|&x| x == 0), "expected all-zero buffer");
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn pinned_host_buffer_set_len_round_trip() {
        crate::cuda::cuda_sys::init().expect("init cuda");
        let _ctx = crate::cuda::cuda_sys::CudaContext::new(0).expect("ctx");

        let mut p = PinnedHostBuffer::<u32>::new(8).expect("alloc");
        assert_eq!(p.len(), 8);
        // SAFETY: shrinking; the bytes 0..4 are still valid (uninit u32 is
        // safe for read because u32: Pod).
        unsafe { p.set_len(4) };
        assert_eq!(p.len(), 4);
        assert_eq!(p.as_slice().len(), 4);
    }
}
