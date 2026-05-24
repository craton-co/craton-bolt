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
use crate::error::{JavelinError, JavelinResult};

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
            _t: PhantomData,
        }
    }

    /// Allocate an empty buffer with room for `capacity` elements of `T`.
    pub fn with_capacity(capacity: usize) -> JavelinResult<Self> {
        let elem_size = size_of::<T>();
        let raw_bytes = capacity.checked_mul(elem_size).ok_or_else(|| {
            JavelinError::Memory(format!(
                "GpuBuffer::with_capacity size overflow: {} * {}",
                capacity, elem_size
            ))
        })?;

        // Round up to ARROW_ALIGNMENT so even small buffers reserve an aligned
        // tail. Zero-sized requests still allocate one aligned chunk so we have
        // a stable, non-null device pointer to hand out.
        let alloc_bytes = round_up_to_alignment(raw_bytes.max(ARROW_ALIGNMENT), ARROW_ALIGNMENT)
            .ok_or_else(|| {
                JavelinError::Memory(format!(
                    "GpuBuffer::with_capacity alignment overflow for {} bytes",
                    raw_bytes
                ))
            })?;

        let ptr = cuda_sys::mem_alloc(alloc_bytes)?;

        if (ptr % ARROW_ALIGNMENT as u64) != 0 {
            // Hand it back to the driver before bailing.
            unsafe {
                let _ = cuda_sys::mem_free(ptr);
            }
            return Err(JavelinError::Memory(format!(
                "cuMemAlloc returned ptr 0x{:x} which is not {}-byte aligned",
                ptr, ARROW_ALIGNMENT
            )));
        }

        Ok(Self {
            ptr,
            len: 0,
            capacity,
            _t: PhantomData,
        })
    }

    /// Allocate `len` elements and zero them.
    pub fn zeros(len: usize) -> JavelinResult<Self> {
        let mut buf = Self::with_capacity(len)?;
        if len > 0 {
            // No cuMemsetD8 in the FFI yet; route through an h2d copy from a
            // zeroed host vector. Slow but correct; we can optimize later.
            let zeros: Vec<u8> = vec![0u8; len.checked_mul(size_of::<T>()).ok_or_else(|| {
                JavelinError::Memory(format!(
                    "GpuBuffer::zeros size overflow: {} * {}",
                    len,
                    size_of::<T>()
                ))
            })?];
            // SAFETY: `buf.ptr` was just allocated with at least `zeros.len()`
            // bytes of capacity, and `zeros` is valid for that many reads.
            unsafe {
                cuda_sys::memcpy_h2d::<u8>(buf.ptr, zeros.as_ptr(), zeros.len())?;
            }
        }
        buf.len = len;
        Ok(buf)
    }

    /// Allocate and copy `slice` from host to device.
    pub fn from_slice(slice: &[T]) -> JavelinResult<Self> {
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
    pub fn to_vec(&self) -> JavelinResult<Vec<T>> {
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
    pub fn copy_to_slice(&self, dst: &mut [T]) -> JavelinResult<()> {
        if dst.len() != self.len {
            return Err(JavelinError::Memory(format!(
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
    pub fn from_arrow_bytes(buf: &arrow_buffer::Buffer) -> JavelinResult<GpuBuffer<u8>> {
        GpuBuffer::<u8>::from_slice(buf.as_slice())
    }
}

impl<T: Pod> Drop for GpuBuffer<T> {
    fn drop(&mut self) {
        if self.ptr == 0 {
            return;
        }
        // SAFETY: `self.ptr` came from `cuda_sys::mem_alloc` in a constructor,
        // is not aliased (we own it), and `Drop` runs after any borrows end.
        let result = unsafe { cuda_sys::mem_free(self.ptr) };
        if let Err(e) = result {
            log::warn!("javelin: GpuBuffer drop failed to free device memory: {}", e);
        }
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
) -> JavelinResult<GpuBuffer<P::Native>>
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
