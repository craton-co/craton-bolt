// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for `STDDEV_POP` / `STDDEV_SAMP` scalar aggregates.
//!
//! Mirrors the offline parse-and-lower style used by the AVG / COUNT
//! tests in `e2e_tests.rs` plus the online round-trips that go through
//! `Engine::sql`. The hand-computed reference values pin the Welford
//! reduction's numerics against the SQL-standard formulas:
//!
//! * STDDEV_POP(x)  = sqrt( Σ (x_i - mean)^2 / N )
//! * STDDEV_SAMP(x) = sqrt( Σ (x_i - mean)^2 / (N - 1) )   for N > 1
//!                  = NULL                                  for N <= 1
//!
//! Each test uses a small fixture (`[1, 2, 3, 4, 5]`) where the deviations
//! cancel cleanly: mean = 3, Σ deviations^2 = 10, so σ_pop = √2 and
//! σ_samp = √2.5 — no floating-point drama, no chunking aliasing.

use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{lower_physical, parse_sql, DataType, Field, MemTableProvider, Schema};

mod common;

// ---------------------------------------------------------------------------
// Fixtures.
// ---------------------------------------------------------------------------

fn stats_int_schema() -> Schema {
    Schema::new(vec![Field {
        name: "v".into(),
        dtype: DataType::Int32,
        nullable: false,
    }])
}

fn stats_int_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("t", stats_int_schema())
}

fn stats_float_schema() -> Schema {
    Schema::new(vec![Field {
        name: "v".into(),
        dtype: DataType::Float64,
        nullable: false,
    }])
}

fn stats_float_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("t", stats_float_schema())
}

fn one_col_batch_int32(values: Vec<i32>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "v",
        ArrowDataType::Int32,
        false,
    )]));
    let col: Arc<dyn Array> = Arc::new(Int32Array::from(values));
    RecordBatch::try_new(schema, vec![col]).unwrap()
}

fn one_col_batch_f64(values: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "v",
        ArrowDataType::Float64,
        false,
    )]));
    let col: Arc<dyn Array> = Arc::new(Float64Array::from(values));
    RecordBatch::try_new(schema, vec![col]).unwrap()
}

/// Reference hand-computation: mean and `Σ (x_i - mean)^2` over a slice.
/// Used by the offline planner-shape tests AND the online execution tests
/// so a future change to the engine's Welford lowering can't silently
/// drift away from the standard formula.
fn ref_mean_m2(xs: &[f64]) -> (f64, f64) {
    if xs.is_empty() {
        return (0.0, 0.0);
    }
    let mean: f64 = xs.iter().copied().sum::<f64>() / (xs.len() as f64);
    let m2: f64 = xs.iter().map(|x| (x - mean).powi(2)).sum();
    (mean, m2)
}

// ---------------------------------------------------------------------------
// Offline: parse + plan output schema.
// ---------------------------------------------------------------------------

/// `STDDEV_POP(v)` and `STDDEV_SAMP(v)` both lower with a single Float64
/// output column whose name follows the `stddev_pop_<col>` /
/// `stddev_samp_<col>` convention documented on `AggregateExpr::output_name`.
#[test]
fn stddev_pop_output_schema_is_float64() {
    let provider = stats_int_provider();
    let plan = parse_sql("SELECT STDDEV_POP(v) FROM t", &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let out = phys.output_schema();
    assert_eq!(out.fields.len(), 1);
    assert_eq!(out.fields[0].name, "stddev_pop_v");
    assert_eq!(out.fields[0].dtype, DataType::Float64);
}

#[test]
fn stddev_samp_output_schema_is_float64() {
    let provider = stats_float_provider();
    let plan = parse_sql("SELECT STDDEV_SAMP(v) FROM t", &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let out = phys.output_schema();
    assert_eq!(out.fields.len(), 1);
    assert_eq!(out.fields[0].name, "stddev_samp_v");
    assert_eq!(out.fields[0].dtype, DataType::Float64);
}

/// Plain `STDDEV(x)` is the SQL-standard alias for `STDDEV_SAMP(x)`; the
/// SQL frontend folds it to the canonical name so the planner only ever
/// sees one shape. The output column name follows STDDEV_SAMP — *not*
/// STDDEV — because the SQL frontend rewrites before naming.
#[test]
fn bare_stddev_maps_to_stddev_samp_by_default() {
    let provider = stats_float_provider();
    let plan = parse_sql("SELECT STDDEV(v) FROM t", &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let out = phys.output_schema();
    assert_eq!(out.fields[0].name, "stddev_samp_v");
    assert_eq!(out.fields[0].dtype, DataType::Float64);
}

/// `STDDEV_POP` over a Utf8 column must be rejected at type-check time —
/// the operand has to be numeric. The error surfaces from the plan
/// schema-check (logical_plan.rs), not from the SQL frontend.
#[test]
fn stddev_over_non_numeric_column_is_rejected() {
    let schema = Schema::new(vec![Field {
        name: "name".into(),
        dtype: DataType::Utf8,
        nullable: true,
    }]);
    let provider = MemTableProvider::new().with_table("t", schema);
    // The numeric-operand check now fires at parse/plan time (the frontend
    // resolves the aggregate's operand dtype eagerly), surfacing a clear
    // `BoltError::Type` like "STDDEV requires a numeric operand, got Utf8"
    // rather than deferring to `lower_physical`. Either layer rejecting is
    // acceptable; accept whichever surfaces it first.
    let err = match parse_sql("SELECT STDDEV_POP(name) FROM t", &provider) {
        Ok(plan) => {
            lower_physical(&plan).expect_err("STDDEV over Utf8 must error at parse or lowering")
        }
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("STDDEV") || msg.contains("numeric"),
        "expected a STDDEV/numeric type error, got: {msg}"
    );
}

/// GROUP BY + STDDEV is intentionally rejected in v0.5 (scalar-aggregate
/// only). The parser accepts it; the executor surfaces a clear error.
/// We pin the offline plan-build shape here — the actual GROUP BY rejection
/// happens at execution time and is covered by the GPU-online tests below.
#[test]
fn stddev_with_group_by_lowers_to_aggregate_plan() {
    let schema = Schema::new(vec![
        Field {
            name: "k".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "v".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
    ]);
    let provider = MemTableProvider::new().with_table("t", schema);
    // The frontend + planner accept the syntax (so existing GROUP BY
    // queries that learn about STDDEV later in their authoring flow get a
    // useful error from the engine instead of a syntax surprise from the
    // parser).
    let plan = parse_sql("SELECT k, STDDEV_POP(v) FROM t GROUP BY k", &provider)
        .expect("parse should succeed; rejection is at exec time");
    let phys = lower_physical(&plan).expect("lower should succeed");
    // The lowered plan IS an Aggregate; we just verify it has the STDDEV
    // output column so the executor's rejection has something to fire on.
    let out = phys.output_schema();
    let names: Vec<&str> = out.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(
        names.contains(&"stddev_pop_v"),
        "expected stddev_pop_v in output, got {names:?}"
    );
}

// ---------------------------------------------------------------------------
// Online (GPU): end-to-end execution.
// ---------------------------------------------------------------------------

/// `SELECT STDDEV_POP(v) FROM t` over the `i32` sequence `[1, 2, 3, 4, 5]`.
/// Hand-computed: mean = 3, Σ (x-mean)^2 = 10, σ_pop = sqrt(10/5) = √2.
/// We compare against the reference helper rather than the literal √2 so
/// the test reads as "engine matches the SQL-standard formula" instead of
/// "engine matches a magic constant".
#[test]
#[ignore = "gpu:tier1"]
fn e2e_stddev_pop_int32_matches_hand_computed() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let values: Vec<i32> = vec![1, 2, 3, 4, 5];
    let batch = one_col_batch_int32(values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine.sql("SELECT STDDEV_POP(v) FROM t").expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1);
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("STDDEV_POP output is Float64");

    let xs_f64: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    let (_mean, m2) = ref_mean_m2(&xs_f64);
    let expected = (m2 / (xs_f64.len() as f64)).sqrt();

    assert!(!col.is_null(0), "STDDEV_POP must be non-NULL for N=5");
    assert!(
        (col.value(0) - expected).abs() < common::REL_TOL,
        "got {}, expected {}",
        col.value(0),
        expected
    );
}

/// `SELECT STDDEV_SAMP(v) FROM t` over the same `i32` sequence.
/// σ_samp = sqrt(10/4) = √2.5. The non-NULL assertion is load-bearing —
/// STDDEV_SAMP only returns NULL when N <= 1, and N = 5 here.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_stddev_samp_int32_matches_hand_computed() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let values: Vec<i32> = vec![1, 2, 3, 4, 5];
    let batch = one_col_batch_int32(values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine.sql("SELECT STDDEV_SAMP(v) FROM t").expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1);
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("STDDEV_SAMP output is Float64");

    let xs_f64: Vec<f64> = values.iter().map(|&v| v as f64).collect();
    let (_mean, m2) = ref_mean_m2(&xs_f64);
    let expected = (m2 / ((xs_f64.len() - 1) as f64)).sqrt();

    assert!(!col.is_null(0), "STDDEV_SAMP must be non-NULL for N=5");
    assert!(
        (col.value(0) - expected).abs() < common::REL_TOL,
        "got {}, expected {}",
        col.value(0),
        expected
    );
}

/// `SELECT STDDEV_POP(v) FROM t` over a Float64 column.
/// `[1.5, 2.5, 3.5, 4.5, 5.5]`: mean = 3.5, deviations = [-2,-1,0,1,2],
/// Σ squared deviations = 10 (same as the int case after the shift). The
/// f64 path runs through the same Welford accumulator but with no
/// integer-promotion step, so this catches a "we forgot to dispatch the
/// Float64 dtype" regression independently from the int path.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_stddev_pop_float64_matches_hand_computed() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let values: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let batch = one_col_batch_f64(values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine.sql("SELECT STDDEV_POP(v) FROM t").expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1);
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("STDDEV_POP output is Float64");

    let (_mean, m2) = ref_mean_m2(&values);
    let expected = (m2 / (values.len() as f64)).sqrt();

    assert!(!col.is_null(0));
    assert!(
        (col.value(0) - expected).abs() < common::REL_TOL,
        "got {}, expected {}",
        col.value(0),
        expected
    );
}

/// Same input on the Float64 path, STDDEV_SAMP semantics.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_stddev_samp_float64_matches_hand_computed() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let values: Vec<f64> = vec![1.5, 2.5, 3.5, 4.5, 5.5];
    let batch = one_col_batch_f64(values.clone());
    engine.register_table("t", batch).unwrap();

    let h = engine.sql("SELECT STDDEV_SAMP(v) FROM t").expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1);
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("STDDEV_SAMP output is Float64");

    let (_mean, m2) = ref_mean_m2(&values);
    let expected = (m2 / ((values.len() - 1) as f64)).sqrt();

    assert!(!col.is_null(0));
    assert!(
        (col.value(0) - expected).abs() < common::REL_TOL,
        "got {}, expected {}",
        col.value(0),
        expected
    );
}

/// Single-row input: STDDEV_POP must return 0 (variance over one value
/// is 0 by definition), STDDEV_SAMP must return SQL NULL (the divisor
/// `N - 1` is zero — the SQL standard says undefined → NULL). The output
/// schema field is nullable by `LogicalPlan::Aggregate` construction, so
/// the NULL packs cleanly.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_stddev_samp_single_row_is_null() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let batch = one_col_batch_f64(vec![42.0]);
    engine.register_table("t", batch).unwrap();

    let h_pop = engine
        .sql("SELECT STDDEV_POP(v) FROM t")
        .expect("execute pop");
    let pop = h_pop.record_batch();
    let pop_col = pop
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64");
    assert!(!pop_col.is_null(0), "STDDEV_POP defined for N=1");
    assert_eq!(pop_col.value(0), 0.0);

    let h_samp = engine
        .sql("SELECT STDDEV_SAMP(v) FROM t")
        .expect("execute samp");
    let samp = h_samp.record_batch();
    let samp_col = samp
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64");
    assert!(
        samp_col.is_null(0),
        "STDDEV_SAMP must be SQL NULL for N=1 (divisor would be zero)"
    );
}

/// GROUP BY + STDDEV is out of scope for v0.5; the engine must surface a
/// clear error rather than a silent wrong result. We assert on the error
/// path the executor emits (`STDDEV_POP / STDDEV_SAMP are not yet
/// supported with GROUP BY`).
#[test]
#[ignore = "gpu:tier1"]
fn e2e_stddev_with_group_by_is_rejected() {
    use craton_bolt::Engine;

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Float64, false),
    ]));
    let k: Arc<dyn Array> = Arc::new(Int32Array::from(vec![1, 1, 2, 2, 2]));
    let v: Arc<dyn Array> = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]));
    let batch = RecordBatch::try_new(schema, vec![k, v]).unwrap();

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", batch).unwrap();

    // `QueryHandle` is not `Debug`, so match rather than `.expect_err()`.
    let err = match engine.sql("SELECT k, STDDEV_POP(v) FROM t GROUP BY k") {
        Ok(_) => panic!("STDDEV with GROUP BY must error in v0.5"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("STDDEV") && msg.contains("GROUP BY"),
        "expected STDDEV / GROUP BY rejection, got: {msg}"
    );
}
