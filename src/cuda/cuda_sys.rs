// SPDX-License-Identifier: Apache-2.0

//! Raw FFI bindings and thin safe wrappers around the CUDA Driver API.
//!
//! The orchestrator assumes the CUDA Toolkit is installed at link time; we
//! link `cuda` directly with no feature gates.

use std::ffi::CStr;
use std::sync::Once;

use libc::{c_char, c_int, c_uint, c_void};

use crate::error::{JavelinError, JavelinResult};

/// CUDA driver error code (0 == success).
pub type CUresult = i32;
/// Opaque device handle.
pub type CUdevice = i32;
/// Opaque context handle.
pub type CUcontext = *mut c_void;
/// Device pointer (always 64-bit in the v2 ABI).
pub type CUdeviceptr = u64;
/// Device attribute enum value.
pub type CUdevice_attribute = i32;
/// Opaque module handle (loaded PTX/cubin).
pub type CUmodule = *mut c_void;
/// Opaque kernel entry-point handle within a module.
pub type CUfunction = *mut c_void;
/// Opaque stream handle (NULL == default/legacy stream).
pub type CUstream = *mut c_void;

/// Driver "no error" sentinel.
pub const CUDA_SUCCESS: CUresult = 0;

#[cfg_attr(target_os = "windows", link(name = "cuda", kind = "static"))]
#[cfg_attr(not(target_os = "windows"), link(name = "cuda"))]
extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(
        pi: *mut c_int,
        attrib: CUdevice_attribute,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    pub fn cuGetErrorName(error: CUresult, str: *mut *const c_char) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, str: *mut *const c_char) -> CUresult;
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleLoadDataEx(
        module: *mut CUmodule,
        image: *const c_void,
        num_options: c_uint,
        options: *mut i32,
        option_values: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;
    pub fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;
    pub fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: c_uint, grid_dim_y: c_uint, grid_dim_z: c_uint,
        block_dim_x: c_uint, block_dim_y: c_uint, block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
}

/// Convert a `CUresult` into a `JavelinResult`, attaching the driver's message.
pub fn check(code: CUresult) -> JavelinResult<()> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }
    let msg = unsafe {
        let mut ptr: *const c_char = std::ptr::null();
        if cuGetErrorString(code, &mut ptr) == CUDA_SUCCESS && !ptr.is_null() {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        } else {
            format!("unknown CUDA error {}", code)
        }
    };
    Err(JavelinError::Cuda { code, msg })
}

static INIT: Once = Once::new();
static mut INIT_RESULT: CUresult = CUDA_SUCCESS;

/// Idempotently call `cuInit(0)`. Safe to invoke from any thread.
pub fn init() -> JavelinResult<()> {
    INIT.call_once(|| unsafe {
        INIT_RESULT = cuInit(0);
    });
    // Safe: INIT_RESULT is written exactly once inside call_once and only read after.
    check(unsafe { INIT_RESULT })
}

/// Number of CUDA-capable devices visible to the driver.
pub fn device_count() -> JavelinResult<i32> {
    let mut n: c_int = 0;
    check(unsafe { cuDeviceGetCount(&mut n) })?;
    Ok(n as i32)
}

/// Resolve the `ordinal`-th device handle.
pub fn device_get(ordinal: i32) -> JavelinResult<CUdevice> {
    let mut dev: CUdevice = 0;
    check(unsafe { cuDeviceGet(&mut dev, ordinal as c_int) })?;
    Ok(dev)
}

/// Human-readable device name (e.g. "NVIDIA GeForce RTX 4090").
pub fn device_name(dev: CUdevice) -> JavelinResult<String> {
    const LEN: usize = 256;
    let mut buf = [0i8; LEN];
    check(unsafe { cuDeviceGetName(buf.as_mut_ptr() as *mut c_char, LEN as c_int, dev) })?;
    // Buffer is NUL-terminated by the driver; find the terminator ourselves to be safe.
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr() as *const c_char) };
    Ok(cstr.to_string_lossy().trim_end_matches('\0').to_string())
}

/// Owned CUDA context. Thread-pinned by the driver: `Send` but not `Sync`.
pub struct CudaContext {
    raw: CUcontext,
}

// A context handle may be moved between threads (the driver allows
// cuCtxSetCurrent on any thread), but concurrent use from multiple threads
// without external synchronization is undefined.
unsafe impl Send for CudaContext {}

impl CudaContext {
    /// Initialize the driver and create a primary-style context on `device_ordinal`.
    pub fn new(device_ordinal: i32) -> JavelinResult<Self> {
        init()?;
        let dev = device_get(device_ordinal)?;
        let mut raw: CUcontext = std::ptr::null_mut();
        check(unsafe { cuCtxCreate_v2(&mut raw, 0, dev) })?;
        Ok(Self { raw })
    }

    /// Bind this context to the calling thread.
    pub fn set_current(&self) -> JavelinResult<()> {
        check(unsafe { cuCtxSetCurrent(self.raw) })
    }

    /// Raw handle accessor for downstream submodules.
    pub fn raw(&self) -> CUcontext {
        self.raw
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let code = unsafe { cuCtxDestroy_v2(self.raw) };
        if code != CUDA_SUCCESS {
            eprintln!(
                "javelin: cuCtxDestroy_v2 failed with code {} (context leaked)",
                code
            );
        }
    }
}

/// Allocate `bytes` of device memory and return the raw pointer.
pub fn mem_alloc(bytes: usize) -> JavelinResult<CUdeviceptr> {
    let mut ptr: CUdeviceptr = 0;
    check(unsafe { cuMemAlloc_v2(&mut ptr, bytes) })?;
    Ok(ptr)
}

/// Free a device allocation previously returned by `mem_alloc`.
///
/// # Safety
/// Caller must guarantee `ptr` is live, came from `mem_alloc`, is not aliased,
/// and that no in-flight kernel still references it.
pub unsafe fn mem_free(ptr: CUdeviceptr) -> JavelinResult<()> {
    check(cuMemFree_v2(ptr))
}

/// Copy `count` elements of `T` from host `src` to device `dst`.
///
/// # Safety
/// `src` must be valid for reads of `count * size_of::<T>()` bytes and `dst`
/// must point to a device allocation of at least the same size.
pub unsafe fn memcpy_h2d<T>(dst: CUdeviceptr, src: *const T, count: usize) -> JavelinResult<()> {
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        JavelinError::Memory(format!(
            "memcpy_h2d size overflow: {} * {}",
            count,
            std::mem::size_of::<T>()
        ))
    })?;
    check(cuMemcpyHtoD_v2(dst, src as *const c_void, bytes))
}

/// Copy `count` elements of `T` from device `src` to host `dst`.
///
/// # Safety
/// `dst` must be valid for writes of `count * size_of::<T>()` bytes and `src`
/// must point to a live device allocation of at least the same size.
pub unsafe fn memcpy_d2h<T>(dst: *mut T, src: CUdeviceptr, count: usize) -> JavelinResult<()> {
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        JavelinError::Memory(format!(
            "memcpy_d2h size overflow: {} * {}",
            count,
            std::mem::size_of::<T>()
        ))
    })?;
    check(cuMemcpyDtoH_v2(dst as *mut c_void, src, bytes))
}
