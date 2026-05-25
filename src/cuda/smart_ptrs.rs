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
use crate::error::PatinaResult;

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
    pub fn from_slice(slice: &[T]) -> PatinaResult<Self> {
        Ok(Self {
            buffer: GpuBuffer::<T>::from_slice(slice)?,
        })
    }

    /// Allocate `len` zero-initialized elements on the device.
    pub fn zeros(len: usize) -> PatinaResult<Self> {
        Ok(Self {
            buffer: GpuBuffer::<T>::zeros(len)?,
        })
    }

    /// Allocate room for `cap` elements with logical length zero.
    pub fn with_capacity(cap: usize) -> PatinaResult<Self> {
        Ok(Self {
            buffer: GpuBuffer::<T>::with_capacity(cap)?,
        })
    }

    /// Take ownership of an existing `GpuBuffer` without copying.
    pub fn from_buffer(buffer: GpuBuffer<T>) -> Self {
        Self { buffer }
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
    #[inline]
    pub fn view(&self) -> GpuView<'_, T> {
        GpuView {
            ptr: self.buffer.device_ptr(),
            len: self.buffer.len(),
            _marker: PhantomData,
        }
    }

    /// Borrow as an exclusive GPU view; only one such view may exist.
    #[inline]
    pub fn view_mut(&mut self) -> GpuViewMut<'_, T> {
        GpuViewMut {
            ptr: self.buffer.device_ptr(),
            len: self.buffer.len(),
            _marker: PhantomData,
        }
    }

    /// Copy the vec back to a host `Vec<T>` (synchronous).
    pub fn to_vec(&self) -> PatinaResult<Vec<T>> {
        self.buffer.to_vec()
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
    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Byte length of the view (`len * size_of::<T>()`).
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}

// SAFETY: a `GpuView` is a device pointer plus a length; like `&[u8]` over
// opaque memory, sharing or moving it across threads cannot race on host state.
unsafe impl<'a, T: Pod> Send for GpuView<'a, T> {}
// Intentionally NOT `Sync`: under Craton Patina's launch model a kernel can write
// through the parent `GpuVec` while another thread reads through the view
// across kernel boundaries. The `Cell<()>` in `_marker` makes this `!Sync`.

/// Exclusive (mutable) GPU view; mirrors `&mut [T]` semantics.
pub struct GpuViewMut<'a, T: Pod> {
    ptr: CUdeviceptr,
    len: usize,
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
    #[inline]
    pub fn device_ptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Byte length of the view (`len * size_of::<T>()`).
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Re-borrow the exclusive view as a shared view for the remaining scope.
    #[inline]
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
}
