// SPDX-License-Identifier: Apache-2.0

//! Raw FFI bindings and thin safe wrappers around the CUDA Driver API.
//!
//! Real builds link `cuda` from the installed CUDA Toolkit. When the
//! `cuda-stub` feature is enabled, the `#[link]` block is omitted and every
//! FFI entry point is replaced by a Rust shim that returns
//! [`CUDA_ERROR_STUB`]; [`check`] converts that into
//! `JavelinError::Other("cuda-stub mode: no GPU support compiled in")`.
//! Stub mode lets the crate compile on hosts without the CUDA toolkit and on
//! docs.rs.

use std::ffi::CStr;
use std::sync::OnceLock;

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

/// Sentinel error code returned by every FFI shim when the crate is built with
/// the `cuda-stub` feature. Chosen well above the real CUDA driver error range
/// (currently < 1000) so it cannot collide with a legitimate `CUresult`.
pub const CUDA_ERROR_STUB: CUresult = 100_000;

#[cfg(not(feature = "cuda-stub"))]
#[link(name = "cuda")]
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

// ---------------------------------------------------------------------------
// `cuda-stub` feature: stand-in implementations so the crate compiles on hosts
// without the CUDA toolkit (including docs.rs). Every shim returns
// `CUDA_ERROR_STUB`, which `check()` maps to `JavelinError::Other(...)`.
// ---------------------------------------------------------------------------
#[cfg(feature = "cuda-stub")]
#[allow(non_snake_case, unused_variables)]
mod stubs {
    use super::*;

    pub unsafe fn cuInit(_flags: c_uint) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuDeviceGetCount(_count: *mut c_int) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuDeviceGet(_device: *mut CUdevice, _ordinal: c_int) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuDeviceGetName(_name: *mut c_char, _len: c_int, _dev: CUdevice) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuDeviceGetAttribute(
        _pi: *mut c_int,
        _attrib: CUdevice_attribute,
        _dev: CUdevice,
    ) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuCtxCreate_v2(_pctx: *mut CUcontext, _flags: c_uint, _dev: CUdevice) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuCtxDestroy_v2(_ctx: CUcontext) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuCtxSetCurrent(_ctx: CUcontext) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemAlloc_v2(_dptr: *mut CUdeviceptr, _bytesize: usize) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemFree_v2(_dptr: CUdeviceptr) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemcpyHtoD_v2(_dst: CUdeviceptr, _src: *const c_void, _bytes: usize) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemcpyDtoH_v2(_dst: *mut c_void, _src: CUdeviceptr, _bytes: usize) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuGetErrorName(_error: CUresult, _str: *mut *const c_char) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuGetErrorString(_error: CUresult, _str: *mut *const c_char) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuModuleLoadData(_module: *mut CUmodule, _image: *const c_void) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuModuleLoadDataEx(
        _module: *mut CUmodule,
        _image: *const c_void,
        _num_options: c_uint,
        _options: *mut i32,
        _option_values: *mut *mut c_void,
    ) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuModuleUnload(_module: CUmodule) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuModuleGetFunction(
        _func: *mut CUfunction,
        _module: CUmodule,
        _name: *const c_char,
    ) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuStreamCreate(_stream: *mut CUstream, _flags: c_uint) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuStreamDestroy_v2(_stream: CUstream) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuStreamSynchronize(_stream: CUstream) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuCtxSynchronize() -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuLaunchKernel(
        _f: CUfunction,
        _grid_dim_x: c_uint, _grid_dim_y: c_uint, _grid_dim_z: c_uint,
        _block_dim_x: c_uint, _block_dim_y: c_uint, _block_dim_z: c_uint,
        _shared_mem_bytes: c_uint,
        _stream: CUstream,
        _kernel_params: *mut *mut c_void,
        _extra: *mut *mut c_void,
    ) -> CUresult { CUDA_ERROR_STUB }
}

#[cfg(feature = "cuda-stub")]
pub use stubs::*;

/// Convert a `CUresult` into a `JavelinResult`, attaching the driver's message.
pub fn check(code: CUresult) -> JavelinResult<()> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }
    if code == CUDA_ERROR_STUB {
        return Err(JavelinError::Other(
            "cuda-stub mode: no GPU support compiled in".into(),
        ));
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

static INIT: OnceLock<CUresult> = OnceLock::new();

/// Idempotently call `cuInit(0)`. Safe to invoke from any thread.
pub fn init() -> JavelinResult<()> {
    let code = *INIT.get_or_init(|| unsafe { cuInit(0) });
    check(code)
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
    let mut buf: [c_char; LEN] = [0; LEN];
    check(unsafe { cuDeviceGetName(buf.as_mut_ptr(), LEN as c_int, dev) })?;
    // Buffer is NUL-terminated by the driver; find the terminator ourselves to be safe.
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
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
