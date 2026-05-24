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

use crate::cuda::buffer::GpuBuffer;
use crate::cuda::cuda_sys::CUdeviceptr;
use crate::error::JavelinResult;

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
    pub fn from_slice(slice: &[T]) -> JavelinResult<Self> {
        Ok(Self {
            buffer: GpuBuffer::<T>::from_slice(slice)?,
        })
    }

    /// Allocate `len` zero-initialized elements on the device.
    pub fn zeros(len: usize) -> JavelinResult<Self> {
        Ok(Self {
            buffer: GpuBuffer::<T>::zeros(len)?,
        })
    }

    /// Allocate room for `cap` elements with logical length zero.
    pub fn with_capacity(cap: usize) -> JavelinResult<Self> {
        Ok(Self {
            buffer: GpuBuffer::<T>::with_capacity(cap)?,
        })
    }

    /// Take ownership of an existing `GpuBuffer` without copying.
    pub fn from_buffer(buffer: GpuBuffer<T>) -> Self {
        Self { buffer }
    }

    /// Number of valid `T` elements.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Whether the vec holds zero elements.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Allocated capacity in `T` elements.
    pub fn capacity(&self) -> usize {
        self.buffer.capacity()
    }

    /// Raw device pointer (for FFI / kernel-launch glue).
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.buffer.device_ptr()
    }

    /// Borrow as a shared GPU view; many such views may coexist.
    pub fn view(&self) -> GpuView<'_, T> {
        GpuView {
            ptr: self.buffer.device_ptr(),
            len: self.buffer.len(),
            _marker: PhantomData,
        }
    }

    /// Borrow as an exclusive GPU view; only one such view may exist.
    pub fn view_mut(&mut self) -> GpuViewMut<'_, T> {
        GpuViewMut {
            ptr: self.buffer.device_ptr(),
            len: self.buffer.len(),
            _marker: PhantomData,
        }
    }

    /// Copy the vec back to a host `Vec<T>` (synchronous).
    pub fn to_vec(&self) -> JavelinResult<Vec<T>> {
        self.buffer.to_vec()
    }
}

/// Shared (immutable) GPU view; mirrors `&[T]` semantics.
#[derive(Copy, Clone)]
pub struct GpuView<'a, T: Pod> {
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<&'a [T]>,
}

impl<'a, T: Pod> GpuView<'a, T> {
    /// Number of `T` elements in the view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view spans zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer for FFI / kernel launches.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Byte length of the view (`len * size_of::<T>()`).
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

// SAFETY: a `GpuView` is a device pointer plus a length; like `&[u8]` over
// opaque memory, sharing or moving it across threads cannot race on host state.
unsafe impl<'a, T: Pod> Send for GpuView<'a, T> {}
// SAFETY: shared views are read-only and the underlying GPU bytes are opaque
// to Rust; concurrent `&GpuView` access mirrors `&&[u8]` and is race-free.
unsafe impl<'a, T: Pod> Sync for GpuView<'a, T> {}

/// Exclusive (mutable) GPU view; mirrors `&mut [T]` semantics.
pub struct GpuViewMut<'a, T: Pod> {
    ptr: CUdeviceptr,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
}

impl<'a, T: Pod> GpuViewMut<'a, T> {
    /// Number of `T` elements in the view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view spans zero elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Raw device pointer for FFI / kernel launches.
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Byte length of the view (`len * size_of::<T>()`).
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Re-borrow the exclusive view as a shared view for the remaining scope.
    pub fn as_view(&self) -> GpuView<'_, T> {
        GpuView {
            ptr: self.ptr,
            len: self.len,
            _marker: PhantomData,
        }
    }
}

// SAFETY: ownership of a `GpuViewMut` may move between threads; the underlying
// device memory is reachable only via this single handle for its lifetime.
unsafe impl<'a, T: Pod> Send for GpuViewMut<'a, T> {}
// Intentionally NOT `Sync`: concurrent mutation through shared references
// would race on device memory just as `&mut [T]` would on host memory.
