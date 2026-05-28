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

/// Test-only re-export of the live Tier-2 partition-count constant.
///
/// Integration tests under `tests/` need the same `NUM_PARTITIONS` value
/// the GPU kernels use to build their host-side oracles (e.g. the
/// `partition_of(key)` mirror in `tests/tier2_groupby_e2e.rs`). Without
/// this re-export each test would hard-code the value and silently drift
/// when the kernel constant changes — exactly the bug review C1 caught.
///
/// Importing through this module guarantees a drift now becomes a
/// compile error (or a value mismatch) instead of silently miscomputing
/// the partition oracle. Not part of the public API — `#[doc(hidden)]`.
#[doc(hidden)]
pub mod __test_only_partition_offsets {
    pub use crate::exec::partition_offsets::NUM_PARTITIONS;
}
