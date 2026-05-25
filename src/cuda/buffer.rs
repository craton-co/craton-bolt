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

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::error::{PatinaError, PatinaResult};

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
    pub fn with_capacity(capacity: usize) -> PatinaResult<Self> {
        let elem_size = size_of::<T>();
        let raw_bytes = capacity.checked_mul(elem_size).ok_or_else(|| {
            PatinaError::Memory(format!(
                "GpuBuffer::with_capacity size overflow: {} * {}",
                capacity, elem_size
            ))
        })?;

        // Round up to ARROW_ALIGNMENT so even small buffers reserve an aligned
        // tail. Zero-sized requests still allocate one aligned chunk so we have
        // a stable, non-null device pointer to hand out.
        let requested = round_up_to_alignment(raw_bytes.max(ARROW_ALIGNMENT), ARROW_ALIGNMENT)
            .ok_or_else(|| {
                PatinaError::Memory(format!(
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
    pub fn zeros(len: usize) -> PatinaResult<Self> {
        let mut buf = Self::with_capacity(len)?;
        if len > 0 {
            let byte_len = len.checked_mul(size_of::<T>()).ok_or_else(|| {
                PatinaError::Memory(format!(
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
    pub fn from_slice(slice: &[T]) -> PatinaResult<Self> {
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
    pub fn to_vec(&self) -> PatinaResult<Vec<T>> {
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
    pub fn copy_to_slice(&self, dst: &mut [T]) -> PatinaResult<()> {
        if dst.len() != self.len {
            return Err(PatinaError::Memory(format!(
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
    pub fn from_arrow_bytes(buf: &arrow_buffer::Buffer) -> PatinaResult<GpuBuffer<u8>> {
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
) -> PatinaResult<GpuBuffer<P::Native>>
where
    P: arrow_array::types::ArrowPrimitiveType,
    P::Native: Pod,
{
    GpuBuffer::<P::Native>::from_slice(arr.values().as_ref())
}

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
