// SPDX-License-Identifier: Apache-2.0

//! CUDA layer: raw driver bindings plus higher-level RAII wrappers.

pub mod async_copy;
pub mod buffer;
pub mod cuda_sys;
#[cfg(feature = "cudarc")]
pub mod cudarc_backend;
pub mod dictionary;
pub mod dictionary_any;
pub mod dictionary_i64;
pub mod mem_pool;
pub mod smart_ptrs;
pub mod stream_pool;

pub use async_copy::{download_async, sync, upload_async, PinnedBuffer};
pub use buffer::{primitive_to_gpu, GpuBuffer, PinnedHostBuffer};
pub use cuda_sys::{CUdevice, CUdeviceptr, CUfunction, CUmodule, CUresult, CUstream, CudaContext};
pub use smart_ptrs::{GpuVec, GpuView, GpuViewMut};
