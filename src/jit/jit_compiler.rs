// SPDX-License-Identifier: Apache-2.0

//! PTX → loaded CUDA module via the driver's in-process assembler.
//!
//! Despite the name, no separate NVRTC dependency is involved: we hand the
//! PTX text to `cuModuleLoadData`, which performs the PTX → SASS step inside
//! the CUDA driver and returns a ready-to-launch module.

use std::ffi::CString;
use std::marker::PhantomData;
use std::ptr;

use crate::cuda::cuda_sys::{self, CUfunction, CUmodule};
use crate::error::{JavelinError, JavelinResult};

/// Loaded GPU module — owns one or more CUfunctions.
pub struct CudaModule {
    raw: CUmodule,
}

impl CudaModule {
    /// Load PTX source into a module. The PTX must be a complete, valid module.
    pub fn from_ptx(ptx: &str) -> JavelinResult<Self> {
        let ptx_cstr = CString::new(ptx).map_err(|e| {
            JavelinError::Nvrtc(format!("PTX source contains interior NUL byte: {}", e))
        })?;
        let mut module: CUmodule = ptr::null_mut();
        let code = unsafe {
            cuda_sys::cuModuleLoadData(
                &mut module,
                ptx_cstr.as_ptr() as *const libc::c_void,
            )
        };
        cuda_sys::check(code).map_err(|e| {
            JavelinError::Nvrtc(format!("cuModuleLoadData failed: {}", inner_msg(&e)))
        })?;
        Ok(Self { raw: module })
    }

    /// Look up an entry point by name.
    pub fn function(&self, name: &str) -> JavelinResult<CudaFunction<'_>> {
        let name_cstr = CString::new(name).map_err(|e| {
            JavelinError::Nvrtc(format!(
                "kernel name contains interior NUL byte: {}",
                e
            ))
        })?;
        let mut f: CUfunction = ptr::null_mut();
        let code = unsafe {
            cuda_sys::cuModuleGetFunction(&mut f, self.raw, name_cstr.as_ptr())
        };
        cuda_sys::check(code).map_err(|e| {
            JavelinError::Nvrtc(format!(
                "cuModuleGetFunction({}) failed: {}",
                name,
                inner_msg(&e)
            ))
        })?;
        Ok(CudaFunction {
            raw: f,
            _marker: PhantomData,
        })
    }

    /// Raw handle accessor for downstream submodules.
    pub fn raw(&self) -> CUmodule {
        self.raw
    }
}

impl Drop for CudaModule {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let code = unsafe { cuda_sys::cuModuleUnload(self.raw) };
        if code != cuda_sys::CUDA_SUCCESS {
            eprintln!(
                "javelin: cuModuleUnload failed with code {} (module leaked)",
                code
            );
        }
    }
}

// SAFETY: CUmodule is a global handle valid in any thread once the context is current.
unsafe impl Send for CudaModule {}
// Not Sync — we don't want concurrent mutation across threads.

/// Borrowed handle to a kernel within a `CudaModule`. Lifetime tied to the module.
#[derive(Clone, Copy)]
pub struct CudaFunction<'a> {
    raw: CUfunction,
    _marker: PhantomData<&'a CudaModule>,
}

impl<'a> CudaFunction<'a> {
    /// Raw handle accessor for downstream submodules (e.g. kernel launch).
    pub fn raw(&self) -> CUfunction {
        self.raw
    }
}

/// One-shot: load PTX and return the module. Caller invokes `.function(entry)`.
pub fn compile_and_load(ptx: &str, _entry: &str) -> JavelinResult<CudaModule> {
    CudaModule::from_ptx(ptx)
}

/// Extract the human-readable portion of a `JavelinError::Cuda` for wrapping.
fn inner_msg(e: &JavelinError) -> String {
    match e {
        JavelinError::Cuda { code, msg } => format!("[{}] {}", code, msg),
        other => other.to_string(),
    }
}
