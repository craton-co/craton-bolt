// SPDX-License-Identifier: Apache-2.0

//! Craton Bolt — JIT-compiled GPU SQL engine.
//!
//! Pipeline: SQL string → Logical Plan → Physical Plan → IR → PTX string →
//! NVRTC-compiled cubin → CUDA launch → result Arrow array.
//!
//! Memory safety: GPU allocations are owned by `GpuVec<T>` and borrowed as
//! `GpuView<T>`. Kernel launches that need write access require
//! `GpuViewMut<'_, T>` (a `!Sync`, `!Copy` exclusive handle); read-only kernels
//! accept `GpuView<'_, T>`, so the Rust borrow checker forbids concurrent CPU
//! read/write while a kernel executes.

pub mod cuda;
pub mod plan;
pub mod jit;
pub mod exec;

mod error;
pub use error::{BoltError, BoltResult};

pub use cuda::{GpuBuffer, GpuVec, GpuView, GpuViewMut};
pub use plan::{DataFrame, LogicalPlan, PhysicalPlan, Expr};
pub use exec::Engine;

/// Test-only re-exports of the multi-key GPU sort entry points. NOT a stable
/// API surface — exists so the E2E test in `tests/sort_e2e.rs` can drive the
/// shmem-variant dispatch directly (the public SQL path has a 16k-row gate
/// that wouldn't reach the n=128 shmem case).
#[doc(hidden)]
pub mod __test_only_gpu_sort {
    pub use crate::exec::gpu_sort::{
        sort_indices_on_gpu_multi, GpuSortKey,
    };
    pub use crate::jit::sort_kernel::SortLayout;
}

/// Test-only re-export of sort-direction + key-spec types.
#[doc(hidden)]
pub mod __test_only_sort_kernel {
    pub use crate::jit::sort_kernel::{KeyDesc, SortDirection, SortKernelSpec};
}

/// Test-only re-export of the engine-internal DataType.
#[doc(hidden)]
pub mod __test_only_logical_plan {
    pub use crate::plan::logical_plan::DataType;
}
