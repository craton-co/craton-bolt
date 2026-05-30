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

use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use crate::PoolStats;

/// Type of a pool-stats observer callback. `Send + Sync` so it can
/// be invoked from any engine thread; `'static` so it outlives the
/// process.
pub type PoolStatsObserver = Box<dyn Fn(PoolStats) + Send + Sync + 'static>;

/// Reference-counted handle to a registered observer. Cloning the
/// `Arc` (cheap) lets `notify_observers` lift the callback out of the
/// registry slot and DROP the lock before invoking it — see
/// [`notify_observers`].
type ObserverHandle = Arc<dyn Fn(PoolStats) + Send + Sync + 'static>;

/// Single-slot observer registry. Replacing the observer is allowed
/// (the typical install-once-on-startup pattern is the default, but
/// integration-test code may want to swap collectors mid-process).
///
/// The mutex is a `parking_lot::Mutex`: it does not poison, so a
/// panicking observer can never permanently disable the surface (the
/// old `std::sync::Mutex` + `if let Ok(..)` pattern silently no-op'd
/// every later call once poisoned). The lock is contended only on
/// install — `notify_observers` clones the handle out and never holds
/// the lock across the callback.
static REGISTRY: OnceLock<Mutex<Option<ObserverHandle>>> = OnceLock::new();

fn registry() -> &'static Mutex<Option<ObserverHandle>> {
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
    // `Box<dyn Fn>` → `Arc<dyn Fn>` so `notify_observers` can clone the
    // handle out of the slot and release the lock before invoking it.
    let handle: ObserverHandle = Arc::from(f);
    *registry().lock() = Some(handle);
}

/// Invoke the registered observer with `stats`, if any.
///
/// The registry lock is **not** held while the observer callback runs:
/// we acquire the lock, clone the observer handle (a cheap `Arc` bump)
/// into a local, DROP the guard, and only then invoke the callback.
/// This matters for three reasons:
///
/// * **Re-entrancy** — an observer is free to call
///   [`install_pool_stats_observer`] (or otherwise touch the registry)
///   without self-deadlocking on a non-reentrant mutex.
/// * **Poisoning** — the `parking_lot::Mutex` never poisons, so a
///   panicking observer cannot permanently disable the surface: the
///   next `notify_observers` / `install_pool_stats_observer` still
///   works.
/// * **Never-escalate** — the observer is a *user* callback invoked from
///   the engine's periodic-emit path, which runs AFTER a query has
///   already produced its result. A panic in that callback must never
///   turn a successful query into a panic (the contract documented at
///   `Engine::maybe_emit_pool_stats` / `Engine::sql`). We therefore run
///   the callback inside [`std::panic::catch_unwind`] and log + swallow
///   any panic here, at the single choke point every emit path funnels
///   through.
///
/// The lock is already released before the callback runs (see above), so
/// the caught unwind crosses no held lock and the
/// [`AssertUnwindSafe`](std::panic::AssertUnwindSafe) wrapper is sound:
/// nothing this function still owns is observed after a panic.
pub(crate) fn notify_observers(stats: PoolStats) {
    // Clone the handle out under the lock, then release it.
    let observer = registry().lock().clone();
    if let Some(observer) = observer {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            observer(stats);
        }));
        if result.is_err() {
            // A user pool-stats observer panicked. Swallow it: this runs on
            // the post-query emit path and must never escalate into the query
            // result. The `parking_lot` registry is unpoisoned, so the next
            // emit still fires.
            log::warn!(
                "craton-bolt: pool-stats observer panicked; \
                 panic swallowed (observer callbacks must not escalate)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Serializes the tests that drive the process-wide single-slot
    /// observer registry. The default test harness is multi-threaded, so
    /// without this guard two registry-touching tests stomp each other's
    /// installed observer (one overwrites the other's slot mid-run),
    /// making call-count assertions flaky. `parking_lot::Mutex` is used
    /// because it never poisons even if a test body unwinds while holding
    /// the guard.
    static TEST_SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    fn dummy_stats() -> PoolStats {
        PoolStats {
            total_pooled_bytes: 0,
            bucket_count: 0,
            oom_recovery_count: 0,
            proactive_eviction_count: 0,
        }
    }

    /// A panicking observer must not poison/disable the surface AND must
    /// not escalate: `notify_observers` swallows the panic (it runs on the
    /// post-query emit path), and a subsequently installed observer must
    /// still receive notifications.
    #[test]
    fn panicking_observer_does_not_disable_surface() {
        let _serial = TEST_SERIAL.lock();
        // Install an observer that always panics.
        install_pool_stats_observer(Box::new(|_| panic!("boom")));

        // Triggering notify must NOT unwind out of `notify_observers`: the
        // observer panic is caught and swallowed inside, honouring the
        // documented "never escalate" contract. `catch_unwind` here only
        // guards the test harness; we assert it caught nothing.
        let result = std::panic::catch_unwind(|| notify_observers(dummy_stats()));
        assert!(
            result.is_ok(),
            "observer panic must be swallowed by notify_observers, not propagated"
        );

        // The surface must still be usable: installing a fresh
        // observer must succeed and it must receive notifications.
        static HITS: AtomicUsize = AtomicUsize::new(0);
        HITS.store(0, Ordering::SeqCst);
        install_pool_stats_observer(Box::new(|_| {
            HITS.fetch_add(1, Ordering::SeqCst);
        }));

        notify_observers(dummy_stats());
        notify_observers(dummy_stats());

        assert_eq!(
            HITS.load(Ordering::SeqCst),
            2,
            "observer installed after a panic must still be notified"
        );

        // Reset the single-slot registry so we don't leak a live
        // observer into other tests in the process.
        install_pool_stats_observer(Box::new(|_| ()));
    }

    /// A panicking observer that stays installed must keep being invoked on
    /// every subsequent emit (the swallow must not unregister it) and must
    /// never escalate across repeated emits — modelling the engine's periodic
    /// emit firing more than once over a long-lived process.
    #[test]
    fn panicking_observer_swallowed_on_every_emit() {
        let _serial = TEST_SERIAL.lock();
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        CALLS.store(0, Ordering::SeqCst);
        install_pool_stats_observer(Box::new(|_| {
            CALLS.fetch_add(1, Ordering::SeqCst);
            panic!("boom");
        }));

        // None of these emits may unwind out of `notify_observers`.
        for _ in 0..3 {
            notify_observers(dummy_stats());
        }

        assert_eq!(
            CALLS.load(Ordering::SeqCst),
            3,
            "panicking observer must still be invoked on each emit (swallow \
             must not unregister it)"
        );

        // Reset the single-slot registry so we don't leak a live observer.
        install_pool_stats_observer(Box::new(|_| ()));
    }
}
