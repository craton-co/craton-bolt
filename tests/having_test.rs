// SPDX-License-Identifier: Apache-2.0

//! Regression tests for the HAVING-dropped-during-lowering bug.
//!
//! Before the fix, `lower()` matched `LogicalPlan::Filter` over a
//! non-scan-chain input (e.g. `Filter { Project { Aggregate { .. } } }`,
//! which is what `plan_select` emits for `HAVING`) by silently dropping the
//! predicate and returning the unfiltered inner plan. Queries with HAVING
//! therefore returned every group instead of the surviving ones.
//!
//! The fix:
//!   - Adds a `PhysicalPlan::Filter { input, predicate }` variant.
//!   - Wraps the inner plan in that variant in `lower()` when the Filter
//!     can't be folded into a fused projection kernel.
//!   - Adds a host-side executor that evaluates the predicate against the
//!     inner plan's `RecordBatch` and applies `arrow::compute::filter`.
//!
//! These tests assert both the plan-level shape (no GPU needed) and the
//! end-to-end behaviour (gated behind `#[ignore]` so it doesn't run on
//! CPU-only CI).

use std::sync::Arc;

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{
    lower_physical, parse_sql, BinaryOp, DataType, Expr, Field, Literal, LogicalPlan,
    MemTableProvider, PhysicalPlan, Schema,
};

// ---- Fixtures --------------------------------------------------------------

fn t_schema() -> Schema {
    Schema::new(vec![
        Field {
            name: "k".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "v".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ])
}

fn t_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("t", t_schema())
}

/// Build a small batch with four groups:
///   k=1, v=1
///   k=1, v=2     -> SUM(v) for k=1 is 3
///   k=2, v=10
///   k=2, v=20    -> SUM(v) for k=2 is 30
///   k=3, v=4     -> SUM(v) for k=3 is 4
///   k=4, v=100   -> SUM(v) for k=4 is 100
fn t_batch() -> RecordBatch {
    let k = Int32Array::from(vec![1, 1, 2, 2, 3, 4]);
    let v = Int32Array::from(vec![1, 2, 10, 20, 4, 100]);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Int32, false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(v)]).unwrap()
}

// ---- Offline: predicate survives lowering ----------------------------------

/// `lower_physical` must retain the HAVING predicate as a
/// `PhysicalPlan::Filter` wrapper around the aggregate result. Before the
/// fix this test would fail: the lowerer returned a bare
/// `PhysicalPlan::Aggregate` (or a `Project` over it), with the predicate
/// silently discarded.
#[test]
fn having_predicate_survives_lowering() {
    let sql = "SELECT k, SUM(v) FROM t GROUP BY k HAVING SUM(v) > 5";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let phys = lower_physical(&plan).expect("lower");

    // The top of the physical plan must be a Filter wrapping the
    // (Project over) Aggregate.
    let (predicate, inner) = match phys {
        PhysicalPlan::Filter { input, predicate } => (predicate, input),
        other => panic!(
            "expected PhysicalPlan::Filter at top (HAVING predicate must survive), got {other:?}"
        ),
    };

    // The predicate should be `sum_v > 5` (the SQL frontend rewrites
    // SUM(v) into a reference to the aggregate output column `sum_v`).
    let Expr::Binary { op, left, right } = &predicate else {
        panic!("HAVING predicate must be a Binary expr, got {predicate:?}");
    };
    assert!(matches!(op, BinaryOp::Gt), "HAVING op should be `>`, got {op:?}");
    assert!(
        matches!(left.as_ref(), Expr::Column(n) if n == "sum_v"),
        "HAVING lhs should reference `sum_v`, got {left:?}"
    );
    assert!(
        matches!(right.as_ref(), Expr::Literal(Literal::Int64(5))),
        "HAVING rhs should be Int64(5), got {right:?}"
    );

    // The inner plan must be the lowered aggregate (possibly with a
    // SELECT-order Project on top).
    match *inner {
        PhysicalPlan::Aggregate { .. } => {}
        PhysicalPlan::Project { input: inner_inner, .. } => {
            assert!(
                matches!(*inner_inner, PhysicalPlan::Aggregate { .. }),
                "expected Aggregate under Project inside HAVING Filter"
            );
        }
        other => panic!(
            "expected Aggregate (or Project(Aggregate)) under HAVING Filter, got {other:?}"
        ),
    }
}

/// Hand-built logical plan that mirrors `plan_select`'s `HAVING` shape
/// exactly: `Filter { Project { Aggregate { Scan } } }`. Lowering must
/// still surface a `PhysicalPlan::Filter`. This test guards the path even
/// if the SQL frontend's output shape drifts in the future.
#[test]
fn having_logical_plan_shape_lowers_to_filter() {
    use craton_bolt::plan::AggregateExpr;

    let scan = LogicalPlan::Scan {
        table: "t".into(),
        projection: None,
        schema: t_schema(),
    };
    let aggregate = LogicalPlan::Aggregate {
        input: Box::new(scan),
        group_by: vec![Expr::Column("k".into())],
        aggregates: vec![AggregateExpr::Sum(Expr::Column("v".into()))],
    };
    let project = LogicalPlan::Project {
        input: Box::new(aggregate),
        exprs: vec![Expr::Column("k".into()), Expr::Column("sum_v".into())],
    };
    let filter = LogicalPlan::Filter {
        input: Box::new(project),
        predicate: Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(Expr::Column("sum_v".into())),
            right: Box::new(Expr::Literal(Literal::Int64(5))),
        },
    };

    let phys = lower_physical(&filter).expect("lower");
    assert!(
        matches!(phys, PhysicalPlan::Filter { .. }),
        "expected PhysicalPlan::Filter at top, got {phys:?}"
    );
}

// ---- Online: end-to-end correctness ----------------------------------------

#[test]
#[ignore = "requires CUDA device - run with `cargo test -- --ignored`"]
fn e2e_having_filters_out_groups() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", t_batch()).unwrap();

    // Group sums:
    //   k=1 -> 3
    //   k=2 -> 30
    //   k=3 -> 4
    //   k=4 -> 100
    // HAVING SUM(v) > 5 should keep k=2 (30) and k=4 (100); drop k=1 (3) and k=3 (4).
    let h = engine
        .sql("SELECT k, SUM(v) FROM t GROUP BY k HAVING SUM(v) > 5")
        .expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 2, "exactly two groups survive HAVING");

    let k_arr = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("k Int32");
    let sum_arr = out
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("SUM(v) widens Int32 -> Int64");

    // Collect (k, sum) pairs into a set so we don't depend on group order.
    let mut pairs: Vec<(i32, i64)> = (0..out.num_rows())
        .map(|i| (k_arr.value(i), sum_arr.value(i)))
        .collect();
    pairs.sort();
    assert_eq!(pairs, vec![(2, 30), (4, 100)]);
}
