// SPDX-License-Identifier: Apache-2.0

//! CUDA layer: raw driver bindings plus higher-level RAII wrappers.

pub mod cuda_sys;
pub mod buffer;
pub mod smart_ptrs;
pub mod dictionary;
pub mod dictionary_i64;
pub mod dictionary_any;
pub mod mem_pool;

pub use buffer::{primitive_to_gpu, GpuBuffer};
pub use cuda_sys::{CudaContext, CUdevice, CUdeviceptr, CUfunction, CUmodule, CUresult, CUstream};
pub use smart_ptrs::{GpuVec, GpuView, GpuViewMut};

