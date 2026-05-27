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

    // -----------------------------------------------------------------------
    // Async memcpy entry points (Stage 2).
    //
    // These queue the copy on `stream` and update `self.len` synchronously so
    // that subsequent kernel launches on the same stream see the right logical
    // length. They do NOT synchronize the stream — the caller does that once
    // per query after all enqueued work is in flight.
    // -----------------------------------------------------------------------

    /// Async H2D copy: enqueue `slice` onto `stream` and bump `self.len`.
    ///
    /// The destination capacity must already cover `slice.len()`. The host
    /// `slice` must remain live and unmodified until `stream` is synchronized.
    /// For peak bandwidth `slice` should sit in a [`PinnedHostBuffer`], but
    /// pageable memory still works (the driver bounces internally).
    pub fn copy_from_async(&mut self, slice: &[T], stream: CUstream) -> BoltResult<()> {
        if slice.len() > self.capacity {
            return Err(BoltError::Memory(format!(
                "GpuBuffer::copy_from_async: src len {} exceeds capacity {}",
                slice.len(),
                self.capacity
            )));
        }
        if !slice.is_empty() {
            // SAFETY: capacity check above guarantees the destination owns at
            // least `slice.len()` elements; the host slice is borrowed for the
            // entire call and the caller pledges to keep it live until the
            // owning stream is synchronized.
            unsafe {
                cuda_sys::memcpy_h2d_async::<T>(self.ptr, slice.as_ptr(), slice.len(), stream)?;
            }
        }
        self.len = slice.len();
        Ok(())
    }

    /// Async D2H copy: enqueue a copy of `self` into `dst` on `stream`.
    ///
    /// `dst.len()` MUST equal `self.len()`. The host buffer is not safe to
    /// read until `stream` is synchronized. Best performance requires `dst`
    /// to be page-locked memory.
    pub fn copy_to_async(&self, dst: &mut [T], stream: CUstream) -> BoltResult<()> {
        if dst.len() != self.len {
            return Err(BoltError::Memory(format!(
                "GpuBuffer::copy_to_async length mismatch: dst={}, buffer={}",
                dst.len(),
                self.len
            )));
        }
        if self.len > 0 {
            // SAFETY: dst is valid for writes of self.len elements (checked
            // above); src is a live device allocation of at least the same
            // size. The caller will sync `stream` before reading `dst`.
            unsafe {
                cuda_sys::memcpy_d2h_async::<T>(dst.as_mut_ptr(), self.ptr, self.len, stream)?;
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

/// Page-locked (pinned) host buffer typed at `T`.
///
/// Pinned host memory is required for true overlap of `cuMemcpy*Async` with
/// kernel work — pageable memory forces the driver to stage through an
/// internal bounce buffer, eliminating the win. This is a thin RAII wrapper
/// over `cuMemAllocHost_v2` / `cuMemFreeHost`.
///
/// The capacity is fixed at construction; `len` tracks how many elements the
/// caller has logically initialised so [`as_slice`] / [`as_mut_slice`] return
/// a correctly-sized view.
///
/// Not Stage 2 hot-path — kept here so Stage 3 (pinned result buffers) can
/// pick it up without another visibility tour.
pub struct PinnedHostBuffer<T: Pod> {
    ptr: *mut T,
    /// Logical element count visible via the slice accessors.
    len: usize,
    /// Allocated element capacity (>= len).
    capacity: usize,
    /// Bytes the driver actually allocated for us — needed for `cuMemFreeHost`.
    alloc_bytes: usize,
    _t: PhantomData<T>,
}

impl<T: Pod> PinnedHostBuffer<T> {
    /// Allocate a pinned buffer with room for `capacity` elements (logical
    /// length zero until `set_len` / `fill_from_slice` is called).
    pub fn with_capacity(capacity: usize) -> BoltResult<Self> {
        let bytes = capacity.checked_mul(size_of::<T>()).ok_or_else(|| {
            BoltError::Memory(format!(
                "PinnedHostBuffer::with_capacity overflow: {} * {}",
                capacity,
                size_of::<T>()
            ))
        })?;
        // Zero-byte allocations are not portable across CUDA driver versions;
        // round up to one element so we always own a stable, non-null pointer.
        let alloc_bytes = bytes.max(size_of::<T>().max(1));
        // SAFETY: requested size is non-zero; the driver returns NULL on
        // failure which `check` surfaces as an error.
        let raw = unsafe { cuda_sys::mem_alloc_host(alloc_bytes)? };
        Ok(Self {
            ptr: raw as *mut T,
            len: 0,
            capacity,
            alloc_bytes,
            _t: PhantomData,
        })
    }

    /// Allocate a pinned buffer of `len` elements logically initialised to
    /// `T::zeroed()`. Useful as a destination for D2H async copies.
    pub fn zeros(len: usize) -> BoltResult<Self> {
        let mut buf = Self::with_capacity(len)?;
        if len > 0 {
            // SAFETY: ptr is valid for writes of `len` elements (just
            // allocated with that capacity). Zero is a valid bit pattern for
            // any `Pod` type.
            unsafe {
                std::ptr::write_bytes(buf.ptr, 0u8, len);
            }
        }
        buf.len = len;
        Ok(buf)
    }

    /// Logical element count.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is logically empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Allocated element capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Set the logical length. Caller asserts that the first `new_len`
    /// elements have been initialised (e.g. by an async D2H copy that has
    /// since been synchronized).
    ///
    /// # Safety
    /// `new_len <= capacity` AND the prefix `[0..new_len]` must be initialised.
    pub unsafe fn set_len(&mut self, new_len: usize) {
        debug_assert!(new_len <= self.capacity);
        self.len = new_len;
    }

    /// Read-only host view of the initialised prefix.
    pub fn as_slice(&self) -> &[T] {
        if self.len == 0 || self.ptr.is_null() {
            return &[];
        }
        // SAFETY: `ptr` is non-null and valid for `len` elements; the buffer
        // owns the allocation for the lifetime of `&self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Mutable host view of the initialised prefix.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if self.len == 0 || self.ptr.is_null() {
            return &mut [];
        }
        // SAFETY: see `as_slice`; `&mut self` excludes aliasing.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Raw host pointer (for use as the `src`/`dst` of an async memcpy).
    pub fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    /// Raw mutable host pointer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T: Pod> Drop for PinnedHostBuffer<T> {
    fn drop(&mut self) {
        if self.ptr.is_null() || self.alloc_bytes == 0 {
            return;
        }
        // SAFETY: ptr came from `cuMemAllocHost_v2` and we have unique
        // ownership; no in-flight async copy is still using it because the
        // caller is responsible for synchronizing before drop.
        unsafe {
            if let Err(e) = cuda_sys::mem_free_host(self.ptr as *mut libc::c_void) {
                log::warn!("craton-bolt: cuMemFreeHost failed in PinnedHostBuffer::drop: {}", e);
            }
        }
        self.ptr = std::ptr::null_mut();
    }
}

// SAFETY: a pinned host pointer is valid from any thread once allocated; the
// driver pins it system-wide. Like `Vec<T>`, moving ownership across threads
// is fine, but shared `&mut` access across threads still races on host bytes.
unsafe impl<T: Pod + Send> Send for PinnedHostBuffer<T> {}

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
}
