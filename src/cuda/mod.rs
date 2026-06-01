// SPDX-License-Identifier: Apache-2.0

//! CUDA layer: raw driver bindings plus higher-level RAII wrappers.

pub mod cuda_sys;
pub mod buffer;
pub mod async_copy;
pub mod smart_ptrs;
pub mod dictionary;
pub mod dictionary_i64;
pub mod dictionary_any;
pub mod mem_pool;
pub mod stream_pool;
#[cfg(feature = "cudarc")]
pub mod cudarc_backend;

pub use async_copy::{download_async, sync, upload_async, PinnedBuffer};
pub use buffer::{primitive_to_gpu, GpuBuffer, PinnedHostBuffer};
pub use cuda_sys::{CudaContext, CUdevice, CUdeviceptr, CUfunction, CUmodule, CUresult, CUstream};
pub use smart_ptrs::{GpuVec, GpuView, GpuViewMut};

