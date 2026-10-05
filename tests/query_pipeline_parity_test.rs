// SPDX-License-Identifier: Apache-2.0

//! Engine-level regression tests for the unified query pipeline.
//!
//! Each test pins one previously-divergent behaviour between the SQL,
//! DataFrame, EXPLAIN and host-orchestrated paths:
//!
//! * lazy streams drain only the tables a query references;
//! * lazy streams enforce the declared schema like the eager path;
//! * `run_logical_plan` (DataFrame) applies user `PlanRewrite`s like `sql()`;
//! * host-orchestrated shapes (VALUES, COUNT(DISTINCT) GROUP BY, ...) apply
//!   user rewrites to their component subplans;
//! * `explain_sql` renders the optimized plan that would execute, and labels
//!   a plan whose subqueries were not executed;
//! * `QueriesFailed` counts planning failures, not only execution failures;
//! * sequential engines never observe the previous engine's context-bound
//!   resources.
//!
//! All tests construct an `Engine` (a CUDA context) and so are
//! `#[ignore = "gpu:..."]`; run them on a GPU host with
//! `cargo test --test query_pipeline_parity_test -- --ignored --test-threads=1`
//! (single-threaded because the metrics assertions read process-global
//! counters).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{col, lit, DataFrame, DataType, Field, LogicalPlan, PlanRewrite, Schema};
use craton_bolt::{metrics, BatchProducer, BoltResult, Counter, Engine};

fn x_schema() -> Schema {
    Schema::new(vec![Field::new("x", DataType::Int64, false)])
}

fn int64_batch(start: i64, n: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        ArrowDataType::Int64,
        false,
    )]));
    let col: Int64Array = (start..start + n as i64).collect();
    RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
}

fn float64_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        ArrowDataType::Float64,
        false,
    )]));
    let col: Float64Array = (0..n).map(|i| i as f64).collect();
    RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
}

/// `(x Int64, g Int64)` with `g = x % 4`.
fn keyed_batch(n: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("x", ArrowDataType::Int64, false),
        ArrowField::new("g", ArrowDataType::Int64, false),
    ]));
    let x: Int64Array = (0..n as i64).collect();
    let g: Int64Array = (0..n as i64).map(|v| v % 4).collect();
    RecordBatch::try_new(schema, vec![Arc::new(x), Arc::new(g)]).unwrap()
}

/// A replayable producer that counts how many times it was invoked.
fn counting_producer(calls: Arc<AtomicUsize>, start: i64, n: usize) -> BatchProducer {
    Box::new(move || {
        calls.fetch_add(1, Ordering::SeqCst);
        Box::new(std::iter::once(Ok(int64_batch(start, n))))
    })
}

fn int64_column(batch: &RecordBatch, idx: usize) -> Vec<i64> {
    let a = batch
        .column(idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 column");
    (0..a.len()).map(|i| a.value(i)).collect()
}

/// Counts invocations and returns the plan unchanged.
struct CountingRewrite(Arc<AtomicUsize>);

impl PlanRewrite for CountingRewrite {
    fn name(&self) -> &str {
        "counting"
    }
    fn rewrite(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(plan)
    }
}

/// Wraps the plan in `Filter(x > 1)` — a semantics-changing rewrite whose
/// effect is visible in the result, proving the hook actually ran on the
/// executed plan (not merely that it was called).
struct DropRowsAtOrBelowOne;

impl PlanRewrite for DropRowsAtOrBelowOne {
    fn name(&self) -> &str {
        "drop-rows-at-or-below-one"
    }
    fn rewrite(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
        Ok(LogicalPlan::Filter {
            input: Box::new(plan),
            predicate: col("x").gt(lit(1i64)),
        })
    }
}

// ---- lazy streams ---------------------------------------------------------

#[test]
#[ignore = "gpu:e2e — lazy streams drain only referenced tables"]
fn lazy_stream_only_referenced_table_is_drained() {
    let t1_calls = Arc::new(AtomicUsize::new(0));
    let t2_calls = Arc::new(AtomicUsize::new(0));
    let mut engine = Engine::new().expect("ctx");
    engine
        .register_table_stream_lazy("t1", x_schema(), counting_producer(t1_calls.clone(), 0, 8))
        .unwrap();
    engine
        .register_table_stream_lazy("t2", x_schema(), counting_producer(t2_calls.clone(), 0, 8))
        .unwrap();
    assert_eq!(t1_calls.load(Ordering::SeqCst), 0, "registration is O(1)");

    let out = engine.sql("SELECT x FROM t1").unwrap().into_record_batch();
    assert_eq!(out.num_rows(), 8);
    assert_eq!(t1_calls.load(Ordering::SeqCst), 1, "t1 drained once");
    assert_eq!(
        t2_calls.load(Ordering::SeqCst),
        0,
        "t2 is not referenced and must not be drained"
    );

    // A repeat query does not re-drain t1; a query on t2 drains t2 only.
    engine.sql("SELECT x FROM t1").unwrap();
    engine.sql("SELECT x FROM t2").unwrap();
    assert_eq!(t1_calls.load(Ordering::SeqCst), 1);
    assert_eq!(t2_calls.load(Ordering::SeqCst), 1);
}

#[test]
#[ignore = "gpu:e2e — lazy streams drained through a subquery"]
fn lazy_stream_referenced_only_by_subquery_is_drained() {
    let t_calls = Arc::new(AtomicUsize::new(0));
    let u_calls = Arc::new(AtomicUsize::new(0));
    let other_calls = Arc::new(AtomicUsize::new(0));
    let mut engine = Engine::new().expect("ctx");
    engine
        .register_table_stream_lazy("t", x_schema(), counting_producer(t_calls.clone(), 0, 10))
        .unwrap();
    engine
        .register_table_stream_lazy("u", x_schema(), counting_producer(u_calls.clone(), 5, 1))
        .unwrap();
    engine
        .register_table_stream_lazy(
            "other",
            x_schema(),
            counting_producer(other_calls.clone(), 0, 1),
        )
        .unwrap();
    let out = engine
        .sql("SELECT x FROM t WHERE x > (SELECT MAX(x) FROM u)")
        .unwrap()
        .into_record_batch();
    assert_eq!(int64_column(&out, 0), vec![6, 7, 8, 9]);
    assert_eq!(t_calls.load(Ordering::SeqCst), 1);
    assert_eq!(u_calls.load(Ordering::SeqCst), 1);
    assert_eq!(other_calls.load(Ordering::SeqCst), 0);
}

#[test]
#[ignore = "gpu:e2e — lazy stream schema validation"]
fn lazy_stream_schema_mismatch_fails_deterministically() {
    let mut engine = Engine::new().expect("ctx");
    let producer: BatchProducer = Box::new(|| Box::new(std::iter::once(Ok(float64_batch(4)))));
    engine
        .register_table_stream_lazy("t", x_schema(), producer)
        .expect("lazy registration defers validation");
    for attempt in 0..2 {
        let err = engine
            .sql("SELECT x + 1 FROM t")
            .expect_err("Float64 batches against a declared Int64 schema must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("register_table_stream_lazy") && msg.contains("mismatch"),
            "attempt {attempt}: {msg}"
        );
    }
}

#[test]
#[ignore = "gpu:e2e — lazy stream column-count validation"]
fn lazy_stream_column_count_mismatch_fails() {
    let mut engine = Engine::new().expect("ctx");
    let two_cols = Schema::new(vec![
        Field::new("x", DataType::Int64, false),
        Field::new("y", DataType::Int64, false),
    ]);
    let producer: BatchProducer = Box::new(|| Box::new(std::iter::once(Ok(int64_batch(0, 3)))));
    engine
        .register_table_stream_lazy("t", two_cols, producer)
        .unwrap();
    let err = engine
        .sql("SELECT x FROM t")
        .expect_err("1 column vs 2 declared");
    assert!(err.to_string().contains("columns"), "{err}");
}

// ---- user rewrites --------------------------------------------------------

#[test]
#[ignore = "gpu:e2e — DataFrame path applies user rewrites"]
fn run_logical_plan_applies_user_rewrites_like_sql() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut engine = Engine::new()
        .expect("ctx")
        .with_rewrite(Box::new(CountingRewrite(calls.clone())))
        .with_rewrite(Box::new(DropRowsAtOrBelowOne));
    engine.register_table("t", int64_batch(0, 5)).unwrap();

    let via_sql = engine.sql("SELECT x FROM t").unwrap().into_record_batch();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(int64_column(&via_sql, 0), vec![2, 3, 4]);

    let via_df = DataFrame::scan("t", x_schema())
        .select([col("x")])
        .collect(&mut engine)
        .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "DataFrame path must run hooks"
    );
    assert_eq!(int64_column(&via_df, 0), vec![2, 3, 4]);
}

#[test]
#[ignore = "gpu:e2e — host-orchestrated shapes apply user rewrites"]
fn special_shapes_apply_user_rewrites_to_component_subplans() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut engine = Engine::new()
        .expect("ctx")
        .with_rewrite(Box::new(CountingRewrite(calls.clone())));
    engine.register_table("t", int64_batch(0, 6)).unwrap();

    for sql in [
        "SELECT x FROM (VALUES (1), (2)) AS v(x)",
        "SELECT x % 2 AS g, COUNT(DISTINCT x) FROM t GROUP BY x % 2",
        "SELECT DISTINCT ON (x) x FROM t ORDER BY x",
    ] {
        let before = calls.load(Ordering::SeqCst);
        engine.sql(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
        assert!(
            calls.load(Ordering::SeqCst) > before,
            "{sql}: user rewrites must see the component subplans"
        );
    }
}

#[test]
#[ignore = "gpu:e2e — subqueries see user rewrites exactly once"]
fn subquery_is_rewritten_with_its_enclosing_plan_only() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut engine = Engine::new()
        .expect("ctx")
        .with_rewrite(Box::new(CountingRewrite(calls.clone())));
    engine.register_table("t", int64_batch(0, 10)).unwrap();
    engine
        .sql("SELECT x FROM t WHERE x > (SELECT MIN(x) FROM t)")
        .unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        1,
        "the enclosing plan (subquery embedded) is rewritten once; the \
         extracted subquery is not rewritten a second time"
    );
}

// ---- EXPLAIN ----------------------------------------------------------------

#[test]
#[ignore = "gpu:e2e — EXPLAIN renders the optimized plan"]
fn explain_renders_the_optimized_plan() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut engine = Engine::new()
        .expect("ctx")
        .with_rewrite(Box::new(CountingRewrite(calls.clone())));
    engine.register_table("t", int64_batch(0, 4)).unwrap();
    let optimized = engine
        .explain_sql("SELECT x FROM t WHERE 1 = 1 AND x > 2")
        .unwrap();
    let reference = engine.explain_sql("SELECT x FROM t WHERE x > 2").unwrap();
    assert_eq!(
        optimized, reference,
        "constant folding must be reflected in EXPLAIN exactly as executed"
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "EXPLAIN runs user rewrites"
    );
    assert!(!optimized.contains("pre-subquery-resolution"));
}

#[test]
#[ignore = "gpu:e2e — EXPLAIN labels deferred subqueries and never drains"]
fn explain_labels_deferred_subqueries_and_does_not_execute() {
    let calls = Arc::new(AtomicUsize::new(0));
    let mut engine = Engine::new().expect("ctx");
    engine
        .register_table_stream_lazy("t", x_schema(), counting_producer(calls.clone(), 0, 4))
        .unwrap();
    let out = engine
        .explain_sql("SELECT x FROM t WHERE x > (SELECT MIN(x) FROM t)")
        .unwrap();
    assert!(out.contains("pre-subquery-resolution"), "{out}");
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "EXPLAIN must not drain producers"
    );
}

// ---- metrics ------------------------------------------------------------------

fn counters() -> (u64, u64) {
    let m = metrics();
    (
        m.counter(Counter::QueriesTotal).get(),
        m.counter(Counter::QueriesFailed).get(),
    )
}

#[test]
#[ignore = "gpu:e2e — planning failures bump QueriesFailed"]
fn planning_failures_bump_queries_failed() {
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", int64_batch(0, 4)).unwrap();
    for bad in [
        "SELLECT broken",
        "SELECT x FROM missing_table",
        "SELECT x FROM t; SELECT x FROM t",
    ] {
        let (t0, f0) = counters();
        assert!(engine.sql(bad).is_err(), "{bad}");
        let (t1, f1) = counters();
        assert_eq!(t1, t0 + 1, "{bad}: counted once");
        assert_eq!(f1, f0 + 1, "{bad}: failure counted");
    }
    // A successful query bumps total only.
    let (t0, f0) = counters();
    engine.sql("SELECT x FROM t").unwrap();
    let (t1, f1) = counters();
    assert_eq!((t1, f1), (t0 + 1, f0));
}

#[test]
#[ignore = "gpu:e2e — DataFrame planning failures bump QueriesFailed"]
fn run_logical_plan_planning_failure_bumps_queries_failed() {
    struct Refuse;
    impl PlanRewrite for Refuse {
        fn name(&self) -> &str {
            "refuse"
        }
        fn rewrite(&self, _: LogicalPlan) -> BoltResult<LogicalPlan> {
            Err(craton_bolt::BoltError::Plan("refused".into()))
        }
    }
    let mut engine = Engine::new().expect("ctx").with_rewrite(Box::new(Refuse));
    engine.register_table("t", int64_batch(0, 4)).unwrap();
    let (t0, f0) = counters();
    let plan = DataFrame::scan("t", x_schema()).into_plan();
    assert!(engine.run_logical_plan(&plan).is_err());
    let (t1, f1) = counters();
    assert_eq!((t1, f1), (t0 + 1, f0 + 1));
}

// ---- process-global resources -------------------------------------------------

/// Sequential engines must never observe a pointer / module / stream that
/// belongs to a previous (destroyed) context. Iteration count defaults to
/// 200 and can be raised with `BOLT_SEQUENTIAL_ENGINE_ITERS` for a long soak.
#[test]
#[ignore = "gpu:stress — sequential context lifetime"]
fn sequential_engines_do_not_reuse_context_bound_resources() {
    let iters: usize = std::env::var("BOLT_SEQUENTIAL_ENGINE_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    for i in 0..iters {
        let mut e = Engine::new().unwrap_or_else(|err| panic!("iteration {i}: {err}"));
        e.register_table("t", keyed_batch(1024)).unwrap();
        let sum = e
            .sql("SELECT SUM(x) FROM t")
            .unwrap_or_else(|err| panic!("iteration {i}: {err}"))
            .into_record_batch();
        assert_eq!(
            int64_column(&sum, 0),
            vec![1023 * 1024 / 2],
            "iteration {i}"
        );
        let grouped = e
            .sql("SELECT g, COUNT(*) FROM t GROUP BY g ORDER BY g")
            .unwrap_or_else(|err| panic!("iteration {i}: {err}"))
            .into_record_batch();
        assert_eq!(grouped.num_rows(), 4, "iteration {i}");
    }
}
