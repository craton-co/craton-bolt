// SPDX-License-Identifier: Apache-2.0

//! Observability surface — pool-stats observer + tracing span catalogue.
//!
//! # Tracing spans (v0.6 / M5)
//!
//! Each query phase is wrapped in a `tracing` span at `info` level so a
//! downstream subscriber can attribute latency to the correct slice of the
//! pipeline. Spans are **off by default**: nothing is emitted until the
//! caller installs a subscriber.
//!
//! ## Opting in
//!
//! The most common case — pretty-printed phase timings on stderr — is two
//! lines:
//!
//! ```ignore
//! use craton_bolt::tracing; // re-exported from the crate root for convenience
//!
//! tracing_subscriber::fmt()
//!     .with_max_level(tracing::Level::INFO)
//!     .init();
//! ```
//!
//! Once a subscriber is installed every subsequent `Engine::sql` /
//! `Engine::execute` call emits spans whose names match the table below.
//! Forward to OpenTelemetry, Tokio Console, or any other subscriber that
//! consumes the standard `tracing` event stream.
//!
//! ## Span catalogue
//!
//! | Span name      | Source                              | Phase                          |
//! |----------------|-------------------------------------|--------------------------------|
//! | `parse`        | `plan::sql_frontend::parse`         | SQL → `LogicalPlan`            |
//! | `plan`         | `Engine::sql`                       | Logical-plan rewrite pipeline  |
//! | `lower`        | `plan::physical_plan::lower`        | `LogicalPlan` → `PhysicalPlan` |
//! | `codegen`      | `jit::ptx_gen::compile`             | PhysicalPlan → PTX text        |
//! | `ptx_load`     | `jit::CudaModule::from_ptx`         | PTX → loaded driver module     |
//! | `launch`       | `exec::launch::launch_*`            | Kernel grid launch + sync      |
//! | `transfer`     | `cuda::GpuBuffer::*`                | H2D / D2H memcpy               |
//! | `materialize`  | `exec::aggregate::execute_aggregate`| Arrow array packing            |
//!
//! Span names are stable across patch releases. New phases may be added
//! (non-breaking); existing names keep their semantics.
//!
//! # Pool-stats observer
//!
//! Independent of the tracing surface, [`install_pool_stats_observer`]
//! registers a callback that the engine invokes with a [`PoolStats`]
//! snapshot on every periodic emit (default 60s, override via
//! `BOLT_POOL_STATS_INTERVAL_SECS`). Useful for forwarding pool-occupancy
//! gauges into Prometheus / OTel meters without parsing the log line.

use std::sync::{Mutex, OnceLock};

use crate::PoolStats;

/// Type of a pool-stats observer callback. `Send + Sync` so it can
/// be invoked from any engine thread; `'static` so it outlives the
/// process.
pub type PoolStatsObserver = Box<dyn Fn(PoolStats) + Send + Sync + 'static>;

/// Single-slot observer registry. Replacing the observer is allowed
/// (the typical install-once-on-startup pattern is the default, but
/// integration-test code may want to swap collectors mid-process).
/// The mutex is contended only on install — `notify_observers` reads
/// the slot via `lock().ok().and_then(...)` and never blocks on
/// itself.
static REGISTRY: OnceLock<Mutex<Option<PoolStatsObserver>>> = OnceLock::new();

fn registry() -> &'static Mutex<Option<PoolStatsObserver>> {
    REGISTRY.get_or_init(|| Mutex::new(None))
}

/// Install (or replace) the process-wide pool-stats observer.
///
/// Called by downstream observability layers that want structured
/// access to the periodic pool snapshots — Prometheus exporters,
/// OTel meters, custom dashboards. The engine invokes the registered
/// observer once per periodic emit, AFTER the default log line.
///
/// Pass `Box::new(|_| ())` to install a no-op observer (effectively
/// uninstalling the previous one — there's no separate
/// `uninstall_pool_stats_observer` because the single-slot design
/// makes "install no-op" semantically identical and keeps the
/// surface minimal).
///
/// The argument type is spelled as the full `Box<dyn Fn ... + 'static>`
/// trait object (rather than the crate-internal `PoolStatsObserver`
/// alias) to keep the public signature self-describing and avoid
/// leaking an alias through a `pub(crate)` module boundary.
pub fn install_pool_stats_observer(
    f: Box<dyn Fn(PoolStats) + Send + Sync + 'static>,
) {
    if let Ok(mut slot) = registry().lock() {
        *slot = Some(f);
    }
}

/// Invoke the registered observer with `stats`, if any. Silently
/// drops the call if the mutex is poisoned — an observer that
/// panicked once should not stop subsequent engine work.
pub(crate) fn notify_observers(stats: PoolStats) {
    if let Ok(slot) = registry().lock() {
        if let Some(observer) = slot.as_ref() {
            // We intentionally hold the lock across the observer
            // call: it's a `Send + Sync` `Fn`, no re-entrant install
            // is expected, and serialising notifications is the
            // simpler contract. Heavy observer work is the caller's
            // problem to offload (e.g. via a channel).
            observer(stats);
        }
    }
}
