// SPDX-License-Identifier: Apache-2.0

//! Kernel launch glue: turns `CudaFunction` + `GpuView`s into a 1D grid launch.
//!
//! The kernel ABI matches the PTX emitted in Step 8: one `.u64` device pointer
//! per input column, then one per output column, then a final `.u32` row count.

use std::ffi::c_void;
use std::ptr;

use crate::cuda::cuda_sys::{self, CUdeviceptr, CUstream};
use crate::cuda::{GpuView, GpuViewMut};
use crate::error::PatinaResult;
use crate::jit::CudaFunction;

/// Threads per block for the 1D launch (one thread per row).
const DEFAULT_BLOCK_SIZE: u32 = 256;

/// CUDA stream wrapper. NULL stream by default; create explicit ones for overlap.
pub struct CudaStream {
    raw: CUstream,
    owned: bool,
}

impl CudaStream {
    /// The default (NULL) stream — synchronous w.r.t. the device.
    pub fn null() -> Self {
        Self {
            raw: ptr::null_mut(),
            owned: false,
        }
    }

    /// Create a new non-blocking stream.
    pub fn new() -> PatinaResult<Self> {
        let mut s: CUstream = ptr::null_mut();
        unsafe {
            cuda_sys::check(cuda_sys::cuStreamCreate(&mut s, 0))?;
        }
        Ok(Self {
            raw: s,
            owned: true,
        })
    }

    /// Raw handle accessor for the driver FFI.
    pub fn raw(&self) -> CUstream {
        self.raw
    }

    /// Block the caller until all prior work on this stream completes.
    pub fn synchronize(&self) -> PatinaResult<()> {
        unsafe { cuda_sys::check(cuda_sys::cuStreamSynchronize(self.raw)) }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        if self.owned && !self.raw.is_null() {
            unsafe {
                let rc = cuda_sys::cuStreamDestroy_v2(self.raw);
                if rc != cuda_sys::CUDA_SUCCESS {
                    log::warn!("craton-patina: cuStreamDestroy failed ({})", rc);
                }
            }
        }
    }
}

// SAFETY: a CUstream may be used from any thread once its context is current.
unsafe impl Send for CudaStream {}

/// Kernel-argument list: input device pointers, then output device pointers, then n_rows.
///
/// The `'a` lifetime ties this struct to the borrowed views, so the underlying
/// `GpuVec` cannot be dropped while a launch is in flight.
pub struct KernelArgs<'a> {
    /// Device pointers, kept alive (and at stable addresses) for the launch.
    ptrs: Vec<CUdeviceptr>,
    /// Row count passed as the final `.u32` kernel parameter.
    ///
    /// Used by [`launch_1d`] only — kernels that take additional trailing
    /// scalars push them via [`KernelArgs::push_scalar_u32`] and launch via
    /// [`launch_with_geometry`].
    n_rows: u32,
    /// Extra trailing `.u32` scalar parameters, pushed in kernel-order.
    /// Used by kernels with more than one trailing scalar (e.g. the
    /// shared-memory GROUP BY kernels which take both `n_rows` AND
    /// `n_groups`).
    scalars: Vec<u32>,
    _marker: std::marker::PhantomData<(&'a (), &'a mut ())>,
}

impl<'a> KernelArgs<'a> {
    /// Construct an empty arg list for a launch over `n_rows` rows.
    pub fn new(n_rows: u32) -> Self {
        Self {
            ptrs: Vec::new(),
            n_rows,
            scalars: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Construct an empty arg list with no implicit trailing `n_rows`.
    /// Used by callers that drive their own launch geometry and pass
    /// `n_rows` (and any other scalars) explicitly via
    /// [`KernelArgs::push_scalar_u32`].
    pub fn empty() -> Self {
        Self {
            ptrs: Vec::new(),
            n_rows: 0,
            scalars: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Append an input column's device pointer to the arg list.
    ///
    /// The view's *inner* lifetime (its borrow of the underlying
    /// `GpuVec`) must outlive `'a` so dropping the vec mid-launch is a
    /// compile error. The *outer* borrow (the `&` in front of `view`)
    /// only needs to outlive the call — separated as `'b` so a Vec of
    /// views can be iterated and pushed without unification problems.
    pub fn push_input<'b, T: bytemuck::Pod>(&mut self, view: &'b GpuView<'a, T>)
    where
        'a: 'b,
    {
        self.ptrs.push(view.device_ptr());
    }

    /// Append an output column's device pointer to the arg list.
    ///
    /// Same lifetime split as `push_input`. The `&mut` requirement on
    /// the outer reference keeps the borrow checker enforcing that no
    /// shared alias can exist for the duration of the launch.
    pub fn push_output<'b, T: bytemuck::Pod>(&mut self, view: &'b mut GpuViewMut<'a, T>)
    where
        'a: 'b,
    {
        self.ptrs.push(view.device_ptr());
    }

    /// Append a `u32` scalar to the arg list. Pushed after all device-ptr
    /// args by [`launch_with_geometry`], in the order they were registered.
    pub fn push_scalar_u32(&mut self, value: u32) {
        self.scalars.push(value);
    }
}

/// Launch with caller-controlled grid + block geometry and any number of
/// trailing `u32` scalars (registered via
/// [`KernelArgs::push_scalar_u32`]). Synchronizes the stream before
/// returning.
///
/// This is the entry point for kernels whose launch geometry isn't
/// "one thread per row" — shared-memory GROUP BY, partition kernels,
/// scatter, per-partition reduce — and which take more than one trailing
/// scalar.
pub fn launch_with_geometry(
    function: CudaFunction<'_>,
    grid_x: u32,
    block_x: u32,
    shared_bytes: u32,
    stream: &CudaStream,
    args: &mut KernelArgs<'_>,
) -> PatinaResult<()> {
    let mut kernel_params: Vec<*mut c_void> =
        Vec::with_capacity(args.ptrs.len() + args.scalars.len());
    for p in args.ptrs.iter_mut() {
        kernel_params.push(p as *mut CUdeviceptr as *mut c_void);
    }
    for s in args.scalars.iter_mut() {
        kernel_params.push(s as *mut u32 as *mut c_void);
    }

    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block_x,
            1,
            1,
            shared_bytes,
            stream.raw(),
            kernel_params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }

    stream.synchronize()?;
    Ok(())
}

/// Launch the kernel with one thread per row, block size 256, on `stream`.
/// Synchronizes before returning.
pub fn launch_1d(
    function: CudaFunction<'_>,
    stream: &CudaStream,
    args: &mut KernelArgs<'_>,
) -> PatinaResult<()> {
    let grid_x = ((args.n_rows + DEFAULT_BLOCK_SIZE - 1) / DEFAULT_BLOCK_SIZE).max(1);

    // Build the kernel-parameter pointer array. Each entry is a *mut c_void
    // pointing at the storage of one kernel argument (a CUdeviceptr or n_rows).
    let mut kernel_params: Vec<*mut c_void> = Vec::with_capacity(args.ptrs.len() + 1);
    for p in args.ptrs.iter_mut() {
        kernel_params.push(p as *mut CUdeviceptr as *mut c_void);
    }
    kernel_params.push(&mut args.n_rows as *mut u32 as *mut c_void);

    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            DEFAULT_BLOCK_SIZE,
            1,
            1,
            0,
            stream.raw(),
            kernel_params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }

    stream.synchronize()?;
    Ok(())
}
