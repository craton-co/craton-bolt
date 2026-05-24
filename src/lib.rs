// SPDX-License-Identifier: Apache-2.0

//! Javelin — JIT-compiled GPU SQL engine.
//!
//! Pipeline: SQL string → Logical Plan → Physical Plan → IR → PTX string →
//! NVRTC-compiled cubin → CUDA launch → result Arrow array.
//!
//! Memory safety: GPU allocations are owned by `GpuVec<T>` and borrowed as
//! `GpuView<T>`. Kernel launches require `&mut GpuView<T>`, so the Rust borrow
//! checker forbids concurrent CPU read/write while a kernel executes.

pub mod cuda;
pub mod plan;
pub mod jit;
pub mod exec;

mod error;
pub use error::{JavelinError, JavelinResult};

pub use cuda::{GpuBuffer, GpuVec, GpuView};
pub use plan::{DataFrame, LogicalPlan, PhysicalPlan, Expr};
pub use exec::Engine;

/// Convenience entry point: parse and execute a SQL string against a context.
pub fn sql(query: &str) -> JavelinResult<exec::QueryHandle> {
    Engine::new()?.sql(query)
}
