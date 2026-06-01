// SPDX-License-Identifier: Apache-2.0

//! v0.5 / M1: integration tests for primitive scalar-aggregate NULL handling.
//!
//! These tests cover the `pre = None` shape — i.e. `SELECT <agg>(col) FROM t`
//! with a bare-column aggregate input and no WHERE clause — which is dispatched
//! into `crate::exec::aggregate::execute_aggregate`. Tests for the
//! pre-projection (`agg_with_pre`) and GROUP BY paths live in
//! `tests/e2e_tests.rs` (`e2e_sum_price_times_tax_with_nulls_in_price`,
//! `e2e_groupby_sum_with_nulls_in_value_column`, etc.).
//!
//! Each per-dtype × per-aggregate matrix entry:
//!
//!   - constructs a one-column batch with explicit `None` entries,
//!   - registers it through `Engine::register_table`,
//!   - runs `SELECT <AGG>(v) FROM t`,
//!   - asserts the result matches the host-computed null-skipping value.
//!
//! Garbage in NULL positions of the underlying values buffer is irrelevant
//! because `filter_primitive_to_vec` in `aggregate.rs` reads `pa.values()` only
//! for indices where `pa.is_null(i)` is false. We don't synthesize specific
//! garbage values here — Arrow's `Int32Array::from(vec![Some(_), None, ...])`
//! constructor zeroes the underlying values buffer for None positions, but the
//! contract under test is that the engine doesn't depend on that zero. The
//! diff-vs-DuckDB regression case `diff_agg_nulls_min_max_avg_count` in
//! `tests/diff_duckdb.rs` covers the broader oracle-shaped behaviour.
//!
//! `#[ignore = "gpu:tier1"]` on every test: the scalar reduction kernels need
//! a real CUDA device. Run with `cargo test --test aggregate_nulls_e2e -- --ignored`
//! on a GPU host.

use std::sync::Arc;

use arrow_array::{
    ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::Engine;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a one-column batch named `v` from a typed Arrow array. The column is
/// marked nullable so the engine's planner accepts NULLs in the bitmap.
fn one_col_batch(arr: ArrayRef) -> RecordBatch {
    let dt = arr.data_type().clone();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new("v", dt, true)]));
    RecordBatch::try_new(schema, vec![arr]).expect("one-col batch")
}

/// Generic helper: run `sql` against a fresh engine with `batch` registered as
/// table `t`, returning the one-row output `RecordBatch`. Every aggregate query
/// in this file produces a single output row, so we factor the boilerplate out.
fn run_single_row_query(sql: &str, batch: RecordBatch) -> RecordBatch {
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", batch).expect("register");
    let handle = engine.sql(sql).expect("execute");
    let out = handle.record_batch();
    assert_eq!(
        out.num_rows(),
        1,
        "scalar aggregate must produce exactly one row; got {} for `{}`",
        out.num_rows(),
        sql
    );
    out.clone()
}

fn out_i64(out: &RecordBatch) -> i64 {
    out.column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 output")
        .value(0)
}

fn out_f64(out: &RecordBatch) -> f64 {
    out.column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64 output")
        .value(0)
}

/// Assert the single output cell is SQL NULL, dtype-agnostically. SUM/MIN/MAX
/// preserve the input dtype while AVG promotes to Float64, so we check the
/// Arrow null bitmap on the array itself (`is_null(0)` / `null_count`) rather
/// than downcasting to a concrete array type and reading a sentinel value.
fn assert_out_null(out: &RecordBatch) {
    let col = out.column(0);
    assert_eq!(
        col.null_count(),
        1,
        "expected the single aggregate output row to be NULL; null_count = {}",
        col.null_count()
    );
    assert!(
        col.is_null(0),
        "expected the single aggregate output row to be NULL (is_null(0) was false)"
    );
}

// ---------------------------------------------------------------------------
// Int32 column with NULLs — COUNT / SUM / MIN / MAX / AVG
// ---------------------------------------------------------------------------
//
// Fixture: v = [10, NULL, -5, 30, NULL, 7]
//   non-null count = 4
//   SUM           = 10 + (-5) + 30 + 7 = 42 (i64 — SUM(Int32) widens)
//   MIN           = -5
//   MAX           = 30
//   AVG           = 42 / 4 = 10.5
//
// The NULL positions sit at indices 1 and 4 so the surviving rows aren't
// contiguous; this exercises the gather path inside `filter_primitive_to_vec`
// rather than just a prefix.

fn i32_batch_with_nulls() -> RecordBatch {
    let arr = Int32Array::from(vec![Some(10), None, Some(-5), Some(30), None, Some(7)]);
    one_col_batch(Arc::new(arr) as ArrayRef)
}

#[test]
#[ignore = "gpu:tier1"]
fn count_i32_excludes_nulls() {
    // COUNT(col) reads the Arrow null bitmap via `non_null_count_for_input`;
    // a regression that defaults to `n_rows` would return 6 instead of 4.
    let out = run_single_row_query("SELECT COUNT(v) FROM t", i32_batch_with_nulls());
    assert_eq!(out_i64(&out), 4);
}

#[test]
#[ignore = "gpu:tier1"]
fn sum_i32_excludes_nulls() {
    // SUM(Int32) widens to Int64; the strip path uploads the dense filtered
    // slice and runs `reduce_gpu_vec_widened`. A regression that summed the
    // raw values buffer (including the zeroed-but-NULL slots) would still
    // return 42 by accident because Arrow zeroes NULL slots on construction;
    // the diff-vs-DuckDB test pairs with this one to catch the broader case.
    let out = run_single_row_query("SELECT SUM(v) FROM t", i32_batch_with_nulls());
    assert_eq!(out_i64(&out), 42);
}

#[test]
#[ignore = "gpu:tier1"]
fn min_i32_excludes_nulls() {
    // MIN(Int32) keeps the input dtype; the strip path runs the standard
    // i32 reduction kernel on the dense survivors. The expected -5 is the
    // minimum of the non-NULL values; if NULLs leaked as 0, the answer
    // would be -5 (matching), but if they leaked as `i32::MIN`, the answer
    // would jump to that — so a clean separation matters.
    let out = run_single_row_query("SELECT MIN(v) FROM t", i32_batch_with_nulls());
    let got = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 output")
        .value(0);
    assert_eq!(got, -5);
}

#[test]
#[ignore = "gpu:tier1"]
fn max_i32_excludes_nulls() {
    let out = run_single_row_query("SELECT MAX(v) FROM t", i32_batch_with_nulls());
    let got = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 output")
        .value(0);
    assert_eq!(got, 30);
}

#[test]
#[ignore = "gpu:tier1"]
fn avg_i32_excludes_nulls() {
    // AVG denominator is the non-null count (4); numerator the non-null
    // sum (42). A regression that divided by the full row count (6) would
    // return 7.0 instead of 10.5.
    let out = run_single_row_query("SELECT AVG(v) FROM t", i32_batch_with_nulls());
    let got = out_f64(&out);
    assert!((got - 10.5).abs() < 1e-12, "AVG(Int32 nulls): got {got}");
}

// ---------------------------------------------------------------------------
// Int64 column with NULLs
// ---------------------------------------------------------------------------
//
// Fixture: v = [100, NULL, NULL, 50, 25]
//   non-null = 3, sum = 175, min = 25, max = 100, avg = 175/3 ≈ 58.3333.

fn i64_batch_with_nulls() -> RecordBatch {
    let arr = Int64Array::from(vec![Some(100i64), None, None, Some(50), Some(25)]);
    one_col_batch(Arc::new(arr) as ArrayRef)
}

#[test]
#[ignore = "gpu:tier1"]
fn count_i64_excludes_nulls() {
    let out = run_single_row_query("SELECT COUNT(v) FROM t", i64_batch_with_nulls());
    assert_eq!(out_i64(&out), 3);
}

#[test]
#[ignore = "gpu:tier1"]
fn sum_min_max_avg_i64_excludes_nulls() {
    // Bundle SUM/MIN/MAX/AVG in one test to keep CUDA launch overhead
    // down; each is computed independently inside the engine so a
    // regression on any one would still trip the matching assert.
    let sum = out_i64(&run_single_row_query(
        "SELECT SUM(v) FROM t",
        i64_batch_with_nulls(),
    ));
    assert_eq!(sum, 175);

    let min_out = run_single_row_query("SELECT MIN(v) FROM t", i64_batch_with_nulls());
    let min = min_out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 output")
        .value(0);
    assert_eq!(min, 25);

    let max_out = run_single_row_query("SELECT MAX(v) FROM t", i64_batch_with_nulls());
    let max = max_out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 output")
        .value(0);
    assert_eq!(max, 100);

    let avg = out_f64(&run_single_row_query(
        "SELECT AVG(v) FROM t",
        i64_batch_with_nulls(),
    ));
    let expected = 175.0_f64 / 3.0_f64;
    assert!(
        (avg - expected).abs() < 1e-12,
        "AVG(Int64 nulls): got {avg}, want {expected}"
    );
}

// ---------------------------------------------------------------------------
// Float32 column with NULLs
// ---------------------------------------------------------------------------
//
// Fixture: v = [1.5, NULL, 2.5, NULL, -3.0, 4.0]
//   non-null = 4, sum = 5.0, min = -3.0, max = 4.0, avg = 5.0/4 = 1.25.

fn f32_batch_with_nulls() -> RecordBatch {
    let arr = Float32Array::from(vec![
        Some(1.5_f32),
        None,
        Some(2.5),
        None,
        Some(-3.0),
        Some(4.0),
    ]);
    one_col_batch(Arc::new(arr) as ArrayRef)
}

#[test]
#[ignore = "gpu:tier1"]
fn count_f32_excludes_nulls() {
    let out = run_single_row_query("SELECT COUNT(v) FROM t", f32_batch_with_nulls());
    assert_eq!(out_i64(&out), 4);
}

#[test]
#[ignore = "gpu:tier1"]
fn sum_min_max_avg_f32_excludes_nulls() {
    let sum_out = run_single_row_query("SELECT SUM(v) FROM t", f32_batch_with_nulls());
    let sum = sum_out
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("Float32 output")
        .value(0);
    assert!(
        (sum - 5.0_f32).abs() < 1e-6,
        "SUM(Float32 nulls): got {sum}"
    );

    let min_out = run_single_row_query("SELECT MIN(v) FROM t", f32_batch_with_nulls());
    let min = min_out
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("Float32 output")
        .value(0);
    assert!(
        (min - (-3.0_f32)).abs() < 1e-6,
        "MIN(Float32 nulls): got {min}"
    );

    let max_out = run_single_row_query("SELECT MAX(v) FROM t", f32_batch_with_nulls());
    let max = max_out
        .column(0)
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("Float32 output")
        .value(0);
    assert!(
        (max - 4.0_f32).abs() < 1e-6,
        "MAX(Float32 nulls): got {max}"
    );

    // AVG always promotes to Float64.
    let avg = out_f64(&run_single_row_query(
        "SELECT AVG(v) FROM t",
        f32_batch_with_nulls(),
    ));
    assert!(
        (avg - 1.25).abs() < 1e-12,
        "AVG(Float32 nulls): got {avg}, want 1.25"
    );
}

// ---------------------------------------------------------------------------
// Float64 column with NULLs
// ---------------------------------------------------------------------------
//
// Fixture: v = [10.0, NULL, 20.0, 30.0, NULL]
//   non-null = 3, sum = 60.0, min = 10.0, max = 30.0, avg = 20.0.

fn f64_batch_with_nulls() -> RecordBatch {
    let arr = Float64Array::from(vec![Some(10.0_f64), None, Some(20.0), Some(30.0), None]);
    one_col_batch(Arc::new(arr) as ArrayRef)
}

#[test]
#[ignore = "gpu:tier1"]
fn count_f64_excludes_nulls() {
    let out = run_single_row_query("SELECT COUNT(v) FROM t", f64_batch_with_nulls());
    assert_eq!(out_i64(&out), 3);
}

#[test]
#[ignore = "gpu:tier1"]
fn sum_min_max_avg_f64_excludes_nulls() {
    let sum = out_f64(&run_single_row_query(
        "SELECT SUM(v) FROM t",
        f64_batch_with_nulls(),
    ));
    assert!(
        (sum - 60.0).abs() < 1e-12,
        "SUM(Float64 nulls): got {sum}"
    );

    let min = out_f64(&run_single_row_query(
        "SELECT MIN(v) FROM t",
        f64_batch_with_nulls(),
    ));
    assert!(
        (min - 10.0).abs() < 1e-12,
        "MIN(Float64 nulls): got {min}"
    );

    let max = out_f64(&run_single_row_query(
        "SELECT MAX(v) FROM t",
        f64_batch_with_nulls(),
    ));
    assert!(
        (max - 30.0).abs() < 1e-12,
        "MAX(Float64 nulls): got {max}"
    );

    let avg = out_f64(&run_single_row_query(
        "SELECT AVG(v) FROM t",
        f64_batch_with_nulls(),
    ));
    assert!(
        (avg - 20.0).abs() < 1e-12,
        "AVG(Float64 nulls): got {avg}"
    );
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

/// COUNT(*) ignores any column's null bitmap and always equals `n_rows`.
/// Pins the behaviour that `COUNT(*)` does NOT route through
/// `non_null_count_for_input` — the planner currently lowers it to a
/// `Count(Literal(...))` whose inner expression doesn't resolve to a column.
#[test]
#[ignore = "gpu:tier1"]
fn count_star_includes_null_rows() {
    let out = run_single_row_query("SELECT COUNT(*) FROM t", i32_batch_with_nulls());
    // Fixture has 6 rows total; 2 are NULL. COUNT(*) returns 6.
    assert_eq!(out_i64(&out), 6);
}

/// No-NULLs fast path (`null_count == 0`) goes through the zero-copy
/// `primitive_to_gpu` upload and the standard GPU reduce kernel. Pin that
/// the answer matches a hand-computed reference so a regression that
/// silently triggered the host-strip path (or vice versa) shows up here
/// even when both paths would otherwise agree.
#[test]
#[ignore = "gpu:tier1"]
fn primitive_aggregates_no_nulls_fast_path() {
    let arr = Int64Array::from(vec![1_i64, 2, 3, 4, 5]);
    let batch = one_col_batch(Arc::new(arr) as ArrayRef);

    // COUNT(v) on a no-NULL column equals the row count.
    let out = run_single_row_query("SELECT COUNT(v) FROM t", batch.clone());
    assert_eq!(out_i64(&out), 5);

    let sum = out_i64(&run_single_row_query("SELECT SUM(v) FROM t", batch.clone()));
    assert_eq!(sum, 15);

    let min_out = run_single_row_query("SELECT MIN(v) FROM t", batch.clone());
    let min = min_out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 output")
        .value(0);
    assert_eq!(min, 1);

    let max_out = run_single_row_query("SELECT MAX(v) FROM t", batch.clone());
    let max = max_out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 output")
        .value(0);
    assert_eq!(max, 5);

    let avg = out_f64(&run_single_row_query("SELECT AVG(v) FROM t", batch));
    assert!((avg - 3.0).abs() < 1e-12, "AVG no-nulls: got {avg}");
}

/// All-NULL column: COUNT returns 0; SUM/MIN/MAX/AVG return SQL NULL.
///
/// SQL semantics says SUM/MIN/MAX/AVG over an all-NULL (or empty) input are
/// all NULL — only COUNT returns 0. The host strip yields an empty slice and
/// the GPU launch is skipped via the `n_rows == 0` short-circuit in
/// `reduce_gpu_vec` / `fused_avg_gpu_vec`; the aggregate output column is then
/// emitted as a single NULL row rather than the accumulator identity.
///
/// This pins the corrected public contract (matches DuckDB / standard SQL):
/// the result array's single row must be NULL, not a sentinel
/// (i64::MAX/MIN, ±inf) or 0.
#[test]
#[ignore = "gpu:tier1"]
fn primitive_aggregates_all_null_returns_sql_null() {
    // --- Int64 all-NULL ---
    let arr = Int64Array::from(vec![Option::<i64>::None, None, None]);
    let batch = one_col_batch(Arc::new(arr) as ArrayRef);

    // COUNT(v): 0 non-null rows — still 0, never NULL.
    let out = run_single_row_query("SELECT COUNT(v) FROM t", batch.clone());
    assert_eq!(out_i64(&out), 0);

    // SUM / MIN / MAX over all-NULL Int64 are SQL NULL.
    assert_out_null(&run_single_row_query("SELECT SUM(v) FROM t", batch.clone()));
    assert_out_null(&run_single_row_query("SELECT MIN(v) FROM t", batch.clone()));
    assert_out_null(&run_single_row_query("SELECT MAX(v) FROM t", batch.clone()));
    // AVG over all-NULL is SQL NULL (not 0.0).
    assert_out_null(&run_single_row_query("SELECT AVG(v) FROM t", batch));

    // --- Float64 all-NULL ---
    let arr = Float64Array::from(vec![Option::<f64>::None, None, None]);
    let batch = one_col_batch(Arc::new(arr) as ArrayRef);

    let out = run_single_row_query("SELECT COUNT(v) FROM t", batch.clone());
    assert_eq!(out_i64(&out), 0);

    assert_out_null(&run_single_row_query("SELECT SUM(v) FROM t", batch.clone()));
    assert_out_null(&run_single_row_query("SELECT MIN(v) FROM t", batch.clone()));
    assert_out_null(&run_single_row_query("SELECT MAX(v) FROM t", batch.clone()));
    assert_out_null(&run_single_row_query("SELECT AVG(v) FROM t", batch));
}

/// First-row and last-row NULL positions stress the strip path's loop bounds:
/// a regression that emitted `0..n-1` or `1..n` instead of `0..n` would drop
/// the last or first valid row.
#[test]
#[ignore = "gpu:tier1"]
fn nulls_at_boundary_positions() {
    let arr = Int64Array::from(vec![
        None,
        Some(7_i64),
        Some(11),
        Some(13),
        None,
    ]);
    let batch = one_col_batch(Arc::new(arr) as ArrayRef);

    // count = 3, sum = 31, min = 7, max = 13.
    let count = out_i64(&run_single_row_query("SELECT COUNT(v) FROM t", batch.clone()));
    assert_eq!(count, 3);
    let sum = out_i64(&run_single_row_query("SELECT SUM(v) FROM t", batch.clone()));
    assert_eq!(sum, 31);
    let min_out = run_single_row_query("SELECT MIN(v) FROM t", batch.clone());
    let min = min_out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(min, 7);
    let max_out = run_single_row_query("SELECT MAX(v) FROM t", batch);
    let max = max_out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0);
    assert_eq!(max, 13);
}

// ---------------------------------------------------------------------------
// SQL-NULL semantics for empty / all-NULL inputs (MIN/MAX/SUM/AVG → NULL)
// ---------------------------------------------------------------------------
//
// MIN/MAX over an all-NULL or empty primitive column is SQL NULL — previously
// this leaked the reduction identity sentinel (i64::MAX for MIN, i64::MIN for
// MAX, ±inf for floats). SUM/AVG over the same inputs are likewise NULL
// (previously SUM returned 0). These pin the corrected, DuckDB-matching
// behaviour by checking the Arrow null bitmap (`is_null(0)` / `null_count`)
// rather than a concrete value, so no sentinel can sneak through.

#[test]
#[ignore = "gpu:tier1"]
fn min_max_all_null_int64_returns_null() {
    let arr = Int64Array::from(vec![Option::<i64>::None, None, None, None]);
    let batch = one_col_batch(Arc::new(arr) as ArrayRef);

    // Previously MIN leaked i64::MAX and MAX leaked i64::MIN as sentinels.
    assert_out_null(&run_single_row_query("SELECT MIN(v) FROM t", batch.clone()));
    assert_out_null(&run_single_row_query("SELECT MAX(v) FROM t", batch));
}

#[test]
#[ignore = "gpu:tier1"]
fn min_max_all_null_float64_returns_null() {
    let arr = Float64Array::from(vec![Option::<f64>::None, None, None]);
    let batch = one_col_batch(Arc::new(arr) as ArrayRef);

    // Previously MIN leaked +inf and MAX leaked -inf as sentinels.
    assert_out_null(&run_single_row_query("SELECT MIN(v) FROM t", batch.clone()));
    assert_out_null(&run_single_row_query("SELECT MAX(v) FROM t", batch));
}

#[test]
#[ignore = "gpu:tier1"]
fn sum_empty_table_returns_null() {
    // Zero-row table: SUM over an empty input is SQL NULL (not 0). The strip
    // path produces an empty slice exactly as for all-NULL, exercising the
    // `n_rows == 0` short-circuit. COUNT over the same input stays 0.
    let arr = Int64Array::from(Vec::<Option<i64>>::new());
    let batch = one_col_batch(Arc::new(arr) as ArrayRef);

    let count = out_i64(&run_single_row_query("SELECT COUNT(v) FROM t", batch.clone()));
    assert_eq!(count, 0);

    assert_out_null(&run_single_row_query("SELECT SUM(v) FROM t", batch));
}

#[test]
#[ignore = "gpu:tier1"]
fn sum_all_null_returns_null() {
    // All-NULL (non-empty) input: SUM is SQL NULL, never the 0 identity.
    let arr = Int64Array::from(vec![Option::<i64>::None, None, None, None, None]);
    let batch = one_col_batch(Arc::new(arr) as ArrayRef);

    assert_out_null(&run_single_row_query("SELECT SUM(v) FROM t", batch));
}
