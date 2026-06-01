// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the v0.6 [`DataFrame::collect`] materialising
//! terminal.
//!
//! These tests cover the contract documented on `DataFrame::collect`:
//!   1. The builder lowers and executes through the same pipeline as
//!      `Engine::sql` (rewrite → lower → populate-validity → execute).
//!   2. Builder-time validation errors are surfaced as `BoltError::Plan`
//!      from `collect()` rather than being silently dropped.
//!   3. `into_plan()` still works on the same builder shapes, so callers
//!      that want the plan without executing keep their entry point.
//!
//! Online tests (those that actually launch CUDA kernels) live behind
//! `#[ignore = "gpu:e2e"]` so a non-GPU `cargo test` run stays green;
//! invoke them on a GPU host with `cargo test -- --ignored`.

use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{col, count, sum, DataFrame, DataType, Field, LogicalPlan, Schema};
use craton_bolt::{BoltError, Engine};

// ---- fixtures ---------------------------------------------------------------

fn sales_schema() -> Schema {
    Schema::new(vec![
        Field::new("region_id", DataType::Int32, false),
        Field::new("price", DataType::Float64, false),
        Field::new("tax", DataType::Float64, false),
    ])
}

fn sales_batch(n: usize) -> RecordBatch {
    let region: Int32Array = (0..n as i32).map(|i| i % 4).collect();
    let price: Float64Array = (0..n).map(|i| (i + 1) as f64).collect();
    let tax: Float64Array = (0..n).map(|_| 0.1_f64).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("region_id", ArrowDataType::Int32, false),
        ArrowField::new("price", ArrowDataType::Float64, false),
        ArrowField::new("tax", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(region), Arc::new(price), Arc::new(tax)]).unwrap()
}

// ---- offline tests (no GPU) -------------------------------------------------

/// `into_plan()` is still a valid terminal: it returns the underlying
/// `LogicalPlan` without execution, so plan-shape assertions and the
/// low-level lowering entry points keep working.
#[test]
fn into_plan_returns_plan_without_executing() {
    let df = DataFrame::scan("sales", sales_schema()).select(vec![col("price")]);
    let plan = df.into_plan();
    assert!(
        matches!(plan, LogicalPlan::Project { .. }),
        "expected Project at the top of the plan, got {plan:?}"
    );
}

/// Builder-time validation errors (unknown column in `.select(...)`) MUST
/// surface as a `BoltError::Plan` from `collect()`. The historical
/// `into_plan()` path drops the error on the floor (it returns a bare
/// `LogicalPlan`); the new `collect()` is the place to put fail-fast
/// behaviour.
///
/// `collect()` short-circuits on `first_error` *before* touching the
/// engine, so a CUDA-less host still exercises the error path — we
/// only need an `Engine` value to satisfy the signature. The test
/// becomes a `gpu:e2e`-tagged ignore so a no-GPU `cargo test` run
/// stays green; on a GPU host with `--ignored` it covers the
/// fail-fast contract end to end.
#[test]
#[ignore = "gpu:e2e"]
fn collect_surfaces_builder_validation_error() {
    let mut engine = Engine::new().expect("engine");
    let batch = sales_batch(8);
    engine.register_table("sales", batch).expect("register");

    // `nope` is not in the schema — `select` records a first_error.
    let df = DataFrame::scan("sales", sales_schema()).select(vec![col("nope")]);
    let err = df.collect(&mut engine).expect_err("expected Plan error");
    match err {
        BoltError::Plan(msg) => {
            assert!(
                msg.contains("nope"),
                "error message should reference the offending column, got: {msg}"
            );
        }
        other => panic!("expected BoltError::Plan, got {other:?}"),
    }
}

// ---- online tests (require CUDA device) -------------------------------------

/// Smoke: a bare `scan().select()` collected via `DataFrame::collect`
/// returns the projected column with the right row count and values.
#[test]
#[ignore = "gpu:e2e"]
fn collect_projection_returns_record_batch() {
    let mut engine = Engine::new().expect("engine");
    let batch = sales_batch(1024);
    engine.register_table("sales", batch.clone()).expect("register");

    let df = DataFrame::scan("sales", sales_schema()).select(vec![col("price")]);
    let out = df.collect(&mut engine).expect("collect");

    assert_eq!(out.num_rows(), 1024);
    let actual = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64");
    let expected = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..1024 {
        assert_eq!(actual.value(i), expected.value(i), "row {i}");
    }
}

/// `select(price * tax)` should materialise the row-wise product —
/// confirms that lowering of a `Binary` expression runs end to end
/// through the DataFrame path, not just the SQL path.
#[test]
#[ignore = "gpu:e2e"]
fn collect_arithmetic_projection_matches_host() {
    let mut engine = Engine::new().expect("engine");
    let batch = sales_batch(4096);
    engine.register_table("sales", batch.clone()).expect("register");

    let df = DataFrame::scan("sales", sales_schema())
        .select(vec![col("price").mul(col("tax")).alias("revenue")]);
    let out = df.collect(&mut engine).expect("collect");

    assert_eq!(out.num_rows(), 4096);
    let actual = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64");
    let price = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let tax = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..4096 {
        let want = price.value(i) * tax.value(i);
        let got = actual.value(i);
        assert!(
            (got - want).abs() < 1e-9,
            "row {i}: got {got}, want {want}"
        );
    }
}

/// `group_by(...).agg(sum(...), count(...))` collected via the DataFrame
/// path. The aggregate executor lives behind `PhysicalPlan::Aggregate`
/// and is fed by `Engine::execute`; this confirms the same dispatch
/// fires when the entry point is `DataFrame::collect` rather than
/// `Engine::sql`.
#[test]
#[ignore = "gpu:e2e"]
fn collect_group_by_aggregate_runs_through_engine() {
    let mut engine = Engine::new().expect("engine");
    let batch = sales_batch(2048);
    engine.register_table("sales", batch).expect("register");

    let df = DataFrame::scan("sales", sales_schema())
        .group_by(vec![col("region_id")])
        .agg(vec![sum(col("price")), count(col("price"))]);
    let out = df.collect(&mut engine).expect("collect");

    // 4 distinct regions (region_id = i % 4).
    assert_eq!(out.num_rows(), 4, "expected 4 groups, got {out:?}");
    assert!(
        out.num_columns() >= 3,
        "expected at least key + sum + count columns, got {} ({:?})",
        out.num_columns(),
        out.schema()
    );
    // The schema names tell us where the sum and count columns landed —
    // ordering is up to the executor and the planner, so look them up by
    // a name prefix rather than a hard-coded column index.
    let schema = out.schema();
    let sum_idx = schema
        .fields()
        .iter()
        .position(|f| f.name().to_ascii_lowercase().contains("sum"))
        .expect("sum column in output");
    let count_idx = schema
        .fields()
        .iter()
        .position(|f| f.name().to_ascii_lowercase().contains("count"))
        .expect("count column in output");
    let sum_col = out
        .column(sum_idx)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 sum");
    let total: f64 = (0..out.num_rows()).map(|i| sum_col.value(i)).sum();
    let expected: f64 = (1..=2048).map(|x| x as f64).sum();
    assert!(
        (total - expected).abs() < 1e-6,
        "group sums total {total} should equal {expected}"
    );
    let count_col = out
        .column(count_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 count");
    let count_total: i64 = (0..out.num_rows()).map(|i| count_col.value(i)).sum();
    assert_eq!(count_total, 2048, "group counts should sum to input rows");
}

/// `filter()` chained with `select()` — exercises predicate lowering on
/// the DataFrame path. The engine's projection-with-filter pipeline COMPACTS
/// its output (only matching rows survive, in original order), so the
/// assertion mirrors the updated `e2e_filtered_select` shape.
#[test]
#[ignore = "gpu:e2e"]
fn collect_filter_then_select_runs_predicate() {
    let mut engine = Engine::new().expect("engine");
    let batch = sales_batch(2048);
    engine.register_table("sales", batch.clone()).expect("register");

    let df = DataFrame::scan("sales", sales_schema())
        .filter(col("region_id").eq(craton_bolt::plan::lit(1_i32)))
        .select(vec![col("price")]);
    let out = df.collect(&mut engine).expect("collect");

    let actual = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64");
    let region = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let price = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    // `region_id = i % 4` → 512 of 2048 rows match `region_id = 1`.
    let expected: Vec<f64> = (0..2048)
        .filter(|&i| region.value(i) == 1)
        .map(|i| price.value(i))
        .collect();
    assert_eq!(out.num_rows(), expected.len(), "compacted row count");
    assert_eq!(out.num_rows(), 512, "region_id == 1 matches 2048/4 rows");
    for (k, want) in expected.iter().enumerate() {
        assert_eq!(actual.value(k), *want, "compacted row {k}");
    }
}
