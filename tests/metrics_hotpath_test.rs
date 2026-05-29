// SPDX-License-Identifier: Apache-2.0

//! M5 — tests that the metrics registry is wired into the query hot path.
//!
//! Two flavours, mirroring `engine_builder_test.rs`:
//!
//! 1. **Host-only** — exercise the process-wide [`metrics`] registry through
//!    the same public surface `Engine::sql` bumps (`metrics().inc(...)` +
//!    `metrics_snapshot()`), with no CUDA context. These run anywhere.
//! 2. **GPU e2e** (`#[ignore = "gpu:e2e"]`) — actually drive `Engine::sql`
//!    and assert the live registry's `QueriesTotal` advances across the call.
//!    Run with `cargo test --test metrics_hotpath_test -- --ignored` on a
//!    CUDA host.

use craton_bolt::{metrics, metrics_snapshot, Counter};

/// The crate re-exports the registry handle and snapshot, and a `QueriesTotal`
/// bump through that handle is observable in a fresh snapshot. This is the
/// exact mechanism `Engine::sql`'s entry increment relies on — the host path
/// pins it without needing a device.
#[test]
fn queries_total_increments_through_public_surface() {
    // The registry is a process-wide singleton; never assume a zero start
    // (other tests in the same binary may have bumped it). Assert a *delta*.
    let before = metrics_snapshot().counter(Counter::QueriesTotal);
    metrics().inc(Counter::QueriesTotal);
    let after = metrics_snapshot().counter(Counter::QueriesTotal);
    assert_eq!(
        after,
        before + 1,
        "QueriesTotal must advance by exactly one per bump"
    );
}

/// GPU e2e: a real `Engine::sql` *query attempt* advances the live
/// `QueriesTotal` counter (the entry-point increment). We assert a delta of at
/// least one so the test is robust to the singleton's shared state and to any
/// internal sub-queries the engine might issue.
#[test]
#[ignore = "gpu:e2e"]
fn engine_sql_advances_queries_total() {
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use craton_bolt::Engine;
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as _],
    )
    .expect("build record batch");

    let mut engine = Engine::builder().device(0).build().expect("build engine");
    engine.register_table("t", batch).expect("register table");

    let before = metrics_snapshot().counter(Counter::QueriesTotal);
    // The increment lives at `Engine::sql` entry, so it fires whether the
    // query ultimately succeeds or fails — hence "query attempt".
    let _ = engine.sql("SELECT x FROM t");
    let after = metrics_snapshot().counter(Counter::QueriesTotal);

    assert!(
        after >= before + 1,
        "a query attempt must advance QueriesTotal (before={before}, after={after})"
    );
}
