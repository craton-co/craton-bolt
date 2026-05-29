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
pub mod observability;

mod error;
pub use error::{BoltError, BoltResult};

/// Re-export of the [`tracing`] crate so downstream users can install a
/// subscriber and reference span/event APIs without having to add `tracing`
/// to their own `Cargo.toml`. Stable: the major version of `tracing` is part
/// of our SemVer contract.
///
/// See [`observability`] for the catalogue of span names emitted across the
/// query pipeline and the recommended subscriber-installation snippet.
pub use ::tracing;

pub use cuda::{GpuBuffer, GpuVec, GpuView, GpuViewMut};
pub use plan::{DataFrame, LogicalPlan, PhysicalPlan, Expr};
pub use exec::{Engine, EngineBuilder};
pub use exec::streaming::{BatchProducer, BatchStream, MorselPlan, PinnedBudget, TableSource};

/// Stage 4 (pool telemetry): public re-exports for downstream
/// observability. [`pool_stats`] returns a [`PoolStats`] snapshot of
/// the process-wide device-memory pool — total pooled bytes, bucket
/// count, OOM-recovery count, and proactive-eviction count.
///
/// Wire this into a Prometheus exporter, a periodic log line, or a
/// custom dashboard. The fields are documented on [`PoolStats`]; new
/// fields may be added (non-breaking) but existing ones keep their
/// semantics across point releases.
///
/// ```ignore
/// use craton_bolt::pool_stats;
/// let s = pool_stats();
/// println!(
///     "pool: {} bytes across {} buckets ({} OOM rescues, {} proactive evictions)",
///     s.total_pooled_bytes,
///     s.bucket_count,
///     s.oom_recovery_count,
///     s.proactive_eviction_count
/// );
/// ```
pub use cuda::mem_pool::{pool_stats, PoolStats};

/// Stage 7 (P1b): pool-stats observability hooks.
///
/// [`Engine::sql`] emits a periodic `log::info!` line containing a
/// [`PoolStats`] snapshot — see the engine docs for the default
/// 60-second interval and the `BOLT_POOL_STATS_INTERVAL_SECS` override.
/// The default log-only path covers most observability stacks.
///
/// For Prometheus / OpenTelemetry / custom dashboards that need
/// structured data, [`install_pool_stats_observer`] registers a
/// `Fn(PoolStats)` closure that the engine calls on every periodic
/// snapshot (in addition to the log line). Exactly one observer is
/// installed process-wide; a second install overwrites the first.
///
/// ```ignore
/// use craton_bolt::{install_pool_stats_observer, PoolStats};
/// install_pool_stats_observer(Box::new(|s: PoolStats| {
///     // forward to your metrics layer of choice
///     prometheus_gauge!("bolt_pool_bytes").set(s.total_pooled_bytes as f64);
/// }));
/// ```
pub use observability::install_pool_stats_observer;

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

/// Default relative-tolerance constant for the test + bench harness.
///
/// Mirrored from `tests/common::REL_TOL`. Integration tests under `tests/`
/// share that copy via `mod common;`; benches under `benches/` are compiled
/// as their own crate and cannot reach into the test binary's modules, so
/// they import this re-export instead. Keeping both definitions
/// numerically identical is the whole point of centralising the constant —
/// when you change one, change the other (and `git grep REL_TOL` will
/// surface both).
///
/// Not a stable public API surface (`#[doc(hidden)]`); benches and dev-deps
/// only.
#[doc(hidden)]
pub const REL_TOL_TEST: f64 = 1e-9;

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

/// Test-only re-exports of opt-in env-var parser helpers.
///
/// The integration test `tests/env_var_smoke.rs` round-trips each of the
/// engine's opt-in env vars through its parser (or dispatch-flag) helper
/// to lock the toggle semantics in place — empty / "0" / "false" / unset
/// must all keep the default path active, and a positive parse must land
/// on the configured value.
///
/// Marked `#[doc(hidden)]` to signal that the names are an internal
/// test surface, not part of the public API. Cannot be `#[cfg(test)]`-
/// gated because integration tests under `tests/` link against the
/// non-test build of the library — the gate would hide the module
/// from exactly the consumers that need it. (Unit tests inside `src/`
/// reach the underlying helpers directly via `super::`.)
#[doc(hidden)]
pub mod __test_only_env_vars {
    pub use crate::exec::engine::pool_stats_interval_from_env;
    pub use crate::exec::engine::POOL_STATS_ENV;
    pub use crate::exec::gpu_join::parse_env_cap;
    pub use crate::exec::gpu_join::streaming_intern_enabled;
    pub use crate::exec::gpu_join::CAP_ENV_VAR;
    pub use crate::exec::gpu_join::STREAMING_INTERN_ENV_VAR;
    pub use crate::jit::jit_compiler::parse_cap as parse_ptx_cache_cap;
    pub use crate::jit::jit_compiler::PTX_CACHE_CAP_ENV;
    // v0.6 / M6 disk-PTX-cache opt-in env var. Re-exported so the
    // env-var smoke test can assert the canonical name from one place.
    pub use crate::jit::disk_cache::DISK_PTX_CACHE_ENV;
}
