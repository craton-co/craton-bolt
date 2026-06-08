// SPDX-License-Identifier: Apache-2.0

//! v0.6 / M3 stretch: end-to-end tests for the nested-loop non-equi join
//! fallback.
//!
//! The SQL frontend now accepts non-equi ON predicates (`<`, `>`,
//! `BETWEEN`, etc.) and routes them through the residual `filter` slot on
//! `LogicalPlan::Join` / `PhysicalPlan::Join`. The executor then dispatches
//! to [`craton_bolt::exec::join::execute_nested_loop_join`], which:
//!
//!   1. caps the smaller side at `MAX_NESTED_LOOP_INNER_ROWS` (1024);
//!   2. builds the full cartesian batch via `execute_cross_join`;
//!   3. applies the residual predicate host-side via `execute_filter`.
//!
//! Plan-shape tests run offline (no CUDA). The full-execution tests are
//! `#[ignore = "gpu:join"]`-gated to match `tests/joins_e2e.rs` — `Engine`
//! still needs a CUDA context to ingest tables even though the nested-loop
//! path itself never touches the GPU. Run with:
//!   `cargo test --test non_equi_join_test -- --ignored`.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Field, LogicalPlan, MemTableProvider, PhysicalPlan, Schema,
};

// ---- Fixture ----------------------------------------------------------------

/// Small two-table fixture (~10 rows total):
///   `t1` (left):  a Int32 (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
///   `t2` (right): lo Int32 (3, 5, 8),  hi Int32 (5, 7, 9)
///
/// Used by every test below. Sizes are deliberately small (well under the
/// 1024-row inner-side cap) so the cartesian (10 × 3 = 30 rows) is easy to
/// reason about row-by-row.
fn provider_and_batches() -> (MemTableProvider, RecordBatch, RecordBatch) {
    let t1_schema = Schema::new(vec![Field {
        name: "a".into(),
        dtype: DataType::Int32,
        nullable: false,
    }]);
    let t2_schema = Schema::new(vec![
        Field {
            name: "lo".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "hi".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    let provider = MemTableProvider::new()
        .with_table("t1", t1_schema)
        .with_table("t2", t2_schema);

    let t1_arrow = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "a",
        ArrowDataType::Int32,
        false,
    )]));
    let t1_batch = RecordBatch::try_new(
        t1_arrow,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10])) as ArrayRef],
    )
    .unwrap();

    let t2_arrow = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("lo", ArrowDataType::Int32, false),
        ArrowField::new("hi", ArrowDataType::Int32, false),
    ]));
    let t2_batch = RecordBatch::try_new(
        t2_arrow,
        vec![
            Arc::new(Int32Array::from(vec![3, 5, 8])) as ArrayRef,
            Arc::new(Int32Array::from(vec![5, 7, 9])) as ArrayRef,
        ],
    )
    .unwrap();

    (provider, t1_batch, t2_batch)
}

// ---- Offline: plan-shape sanity --------------------------------------------

/// Walk past wildcard `Project`/`Filter` wrappers to surface the underlying
/// `LogicalPlan::Join`. Panics if no Join is found.
fn find_join_logical(p: &LogicalPlan) -> &LogicalPlan {
    match p {
        LogicalPlan::Join { .. } => p,
        LogicalPlan::Project { input, .. }
        | LogicalPlan::Filter { input, .. }
        | LogicalPlan::Distinct { input }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. } => find_join_logical(input),
        other => panic!("expected to find a Join under {other:?}"),
    }
}

#[test]
fn lt_join_lowers_to_join_with_filter() {
    let (provider, _, _) = provider_and_batches();
    let plan =
        parse_sql("SELECT * FROM t1 INNER JOIN t2 ON t1.a < t2.lo", &provider).expect("parse");
    match find_join_logical(&plan) {
        LogicalPlan::Join { on, filter, .. } => {
            assert!(on.is_empty(), "pure non-equi ON yields no equi pairs");
            assert!(
                filter.is_some(),
                "non-equi predicate must populate filter slot"
            );
        }
        other => panic!("expected Join, got {other:?}"),
    }
    // Physical lowering must succeed and carry the filter through.
    let phys = lower_physical(&plan).expect("lower");
    fn find_phys(p: &PhysicalPlan) -> &PhysicalPlan {
        match p {
            PhysicalPlan::Join { .. } => p,
            PhysicalPlan::Project { input, .. }
            | PhysicalPlan::Filter { input, .. }
            | PhysicalPlan::Distinct { input }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::Sort { input, .. } => find_phys(input),
            other => panic!("expected to find a Join under {other:?}"),
        }
    }
    match find_phys(&phys) {
        PhysicalPlan::Join { filter, .. } => {
            assert!(filter.is_some(), "physical Join must carry the filter");
        }
        other => panic!("expected PhysicalPlan::Join, got {other:?}"),
    }
}

#[test]
fn gt_join_lowers_to_join_with_filter() {
    let (provider, _, _) = provider_and_batches();
    let plan =
        parse_sql("SELECT * FROM t1 INNER JOIN t2 ON t1.a > t2.lo", &provider).expect("parse");
    match find_join_logical(&plan) {
        LogicalPlan::Join { on, filter, .. } => {
            assert!(on.is_empty(), "pure non-equi ON yields no equi pairs");
            assert!(filter.is_some(), ">' predicate must populate filter slot");
        }
        other => panic!("expected Join, got {other:?}"),
    }
}

#[test]
fn between_join_lowers_to_join_with_filter() {
    let (provider, _, _) = provider_and_batches();
    let plan = parse_sql(
        "SELECT * FROM t1 INNER JOIN t2 ON t1.a BETWEEN t2.lo AND t2.hi",
        &provider,
    )
    .expect("parse");
    match find_join_logical(&plan) {
        LogicalPlan::Join { on, filter, .. } => {
            assert!(on.is_empty(), "BETWEEN yields no equi pairs");
            assert!(
                filter.is_some(),
                "BETWEEN must populate filter slot via low <= x AND x <= high"
            );
        }
        other => panic!("expected Join, got {other:?}"),
    }
}

#[test]
fn outer_non_equi_join_lowers_then_executor_rejects() {
    // The planner accepts LEFT/RIGHT/FULL + non-equi syntactically (the
    // ON-clause lowerer doesn't know about join_type). The nested-loop
    // executor surfaces a clear "not yet supported" error at run time;
    // that's a follow-up, not a v0.6 deliverable. This test just pins the
    // plan-time half — the executor rejection is exercised in the gated
    // online section below if a CUDA device is available.
    let (provider, _, _) = provider_and_batches();
    let plan = parse_sql("SELECT * FROM t1 LEFT JOIN t2 ON t1.a < t2.lo", &provider)
        .expect("LEFT + non-equi must parse");
    match find_join_logical(&plan) {
        LogicalPlan::Join { filter, .. } => {
            assert!(filter.is_some(), "filter must carry non-equi predicate");
        }
        other => panic!("expected Join, got {other:?}"),
    }
}

// ---- Online (require CUDA device) ------------------------------------------
//
// These tests register two tables, run a SQL non-equi JOIN through the
// engine, and validate the result row-by-row. Run with:
//   cargo test --test non_equi_join_test -- --ignored

#[test]
#[ignore = "gpu:join"]
fn e2e_nested_loop_lt_join() {
    use craton_bolt::Engine;

    let (_, t1, t2) = provider_and_batches();
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    // `t1.a < t2.lo`. t2.lo ∈ {3, 5, 8}; for each (a, lo) pair we keep
    // those where a < lo. Expected counts per t2 row:
    //   lo=3 → a ∈ {1, 2}             → 2 rows
    //   lo=5 → a ∈ {1, 2, 3, 4}       → 4 rows
    //   lo=8 → a ∈ {1, 2, 3, 4, 5, 6, 7} → 7 rows
    // Total: 13 rows.
    let h = engine
        .sql("SELECT * FROM t1 INNER JOIN t2 ON t1.a < t2.lo")
        .expect("execute");
    let out = h.record_batch();
    assert_eq!(
        out.num_rows(),
        13,
        "`a < lo` should produce 13 surviving pairs (got {})",
        out.num_rows()
    );
    // Sanity-check the column shape: 1 left column (a) + 2 right columns
    // (lo, hi). Right collide-free so no renames.
    assert_eq!(out.num_columns(), 3, "output columns: a, lo, hi");
}

#[test]
#[ignore = "gpu:join"]
fn e2e_nested_loop_gt_join() {
    use craton_bolt::Engine;

    let (_, t1, t2) = provider_and_batches();
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    // `t1.a > t2.hi`. t2.hi ∈ {5, 7, 9}.
    //   hi=5 → a ∈ {6, 7, 8, 9, 10}  → 5 rows
    //   hi=7 → a ∈ {8, 9, 10}        → 3 rows
    //   hi=9 → a ∈ {10}              → 1 row
    // Total: 9 rows.
    let h = engine
        .sql("SELECT * FROM t1 INNER JOIN t2 ON t1.a > t2.hi")
        .expect("execute");
    let out = h.record_batch();
    assert_eq!(
        out.num_rows(),
        9,
        "`a > hi` should produce 9 surviving pairs (got {})",
        out.num_rows()
    );
}

#[test]
#[ignore = "gpu:join"]
fn e2e_nested_loop_between_join() {
    use craton_bolt::Engine;

    let (_, t1, t2) = provider_and_batches();
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    // `t1.a BETWEEN t2.lo AND t2.hi` (inclusive on both ends).
    //   (lo=3, hi=5) → a ∈ {3, 4, 5}          → 3 rows
    //   (lo=5, hi=7) → a ∈ {5, 6, 7}          → 3 rows
    //   (lo=8, hi=9) → a ∈ {8, 9}             → 2 rows
    // Total: 8 rows.
    let h = engine
        .sql("SELECT * FROM t1 INNER JOIN t2 ON t1.a BETWEEN t2.lo AND t2.hi")
        .expect("execute");
    let out = h.record_batch();
    assert_eq!(
        out.num_rows(),
        8,
        "`a BETWEEN lo AND hi` should produce 8 surviving pairs (got {})",
        out.num_rows()
    );
    // Verify the surviving `a` values are exactly the expected set.
    let a_idx = out
        .schema()
        .index_of("a")
        .expect("'a' column in output schema");
    let a_col = out
        .column(a_idx)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 'a' column");
    let mut got: Vec<i32> = (0..a_col.len()).map(|i| a_col.value(i)).collect();
    got.sort();
    assert_eq!(got, vec![3, 4, 5, 5, 6, 7, 8, 9]);
}

#[test]
#[ignore = "gpu:join"]
fn e2e_nested_loop_inner_cap_exceeded_errors() {
    // The cap is on the SMALLER side. Build a fixture where both sides
    // exceed MAX_NESTED_LOOP_INNER_ROWS (1024) so the smaller side is
    // also > 1024 and the executor must refuse with the documented message.
    use craton_bolt::Engine;

    const N: i32 = 1500; // exceeds the 1024 cap on the smaller side
    let arrow_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "a",
        ArrowDataType::Int32,
        false,
    )]));
    let values: Vec<i32> = (0..N).collect();
    let t1 = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![Arc::new(Int32Array::from(values.clone())) as ArrayRef],
    )
    .unwrap();
    let t2 = RecordBatch::try_new(
        arrow_schema,
        vec![Arc::new(Int32Array::from(values)) as ArrayRef],
    )
    .unwrap();

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    // Both sides have 1500 rows > 1024 cap, so the inner-side check
    // refuses the plan with the documented hint.
    // `QueryHandle` is not `Debug`, so match rather than `.expect_err()`.
    let err = match engine.sql("SELECT * FROM t1 INNER JOIN t2 ON t1.a < t2.a") {
        Ok(_) => panic!("oversized non-equi join must be rejected"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("non-equi join inner side"),
        "error must mention the cap; got: {msg}"
    );
    assert!(
        msg.contains("1024"),
        "error must surface the configured cap; got: {msg}"
    );
    assert!(
        msg.contains("rewrite as equi"),
        "error must hint at the equi-join rewrite; got: {msg}"
    );
}

#[test]
#[ignore = "gpu:join"]
fn e2e_left_outer_non_equi_is_rejected() {
    // OUTER + non-equi is a v0.6 follow-up; the executor must surface a
    // clear "not yet supported" message rather than silently dropping the
    // preserved-side semantics.
    use craton_bolt::Engine;

    let (_, t1, t2) = provider_and_batches();
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    // `QueryHandle` is not `Debug`, so match rather than `.expect_err()`.
    let err = match engine.sql("SELECT * FROM t1 LEFT JOIN t2 ON t1.a < t2.lo") {
        Ok(_) => panic!("LEFT + non-equi must surface a clear error"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("not yet supported"),
        "error must mention 'not yet supported'; got: {msg}"
    );
}
