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
//!
//! ## PV-stage-d: validity propagation
//!
//! [`plan::TableProvider`] gained two methods —
//! [`has_nulls`](plan::TableProvider::has_nulls) and
//! [`null_count`](plan::TableProvider::null_count) — that let providers
//! advertise per-column null-bearing at plan time. The default safe-`false`
//! / `None` implementations preserve every existing provider's behaviour;
//! providers that override the methods unlock the native-validity codegen
//! path in [`jit::valid_flag_kernels`] (specifically the
//! `*_with_validity` companions).

pub mod cuda;
pub mod plan;
pub mod jit;
pub mod exec;

mod error;
pub use error::{BoltError, BoltResult};

pub use cuda::{GpuBuffer, GpuVec, GpuView, GpuViewMut};
pub use plan::{DataFrame, LogicalPlan, PhysicalPlan, Expr};
pub use exec::Engine;
