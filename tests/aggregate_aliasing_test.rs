// SPDX-License-Identifier: Apache-2.0

//! v0.5 ergonomics: `SELECT SUM(x) AS total FROM t` and friends.
//!
//! Before v0.5 the SQL frontend rejected aliases on aggregate SELECT items —
//! the plan auto-named aggregates (e.g. `sum_x`) and the SELECT-list lowerer
//! refused to carry a user alias through. The fix attaches the alias to the
//! post-Aggregate Project (`Expr::Alias(Column("sum_x"), "total")`); the
//! Project's output schema then names the column `total`, so downstream
//! `HAVING` / `ORDER BY` (and the row caller) see the user-friendly name.
//!
//! These tests pin three shapes:
//!   1. Bare `SUM(x) AS total` — Project schema names it `total`.
//!   2. `ORDER BY total` over an aliased aggregate — the Sort sits above the
//!      Project and refers to the post-projection column.
//!   3. `HAVING total > N` over an aliased aggregate — the HAVING-aware
//!      lowerer recognises the alias via `agg_aliases` and the predicate
//!      lowers to `Column("total")`.

use craton_bolt::plan::{
    parse_sql, BinaryOp, DataType, Expr, Field, LogicalPlan, MemTableProvider, Schema,
};

// ---- Fixture ---------------------------------------------------------------

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

// ---- Helpers ---------------------------------------------------------------

/// Parse & lower a SQL string against the `t` provider, panicking with the
/// underlying error on failure. Mirrors the helper pattern used in
/// `having_test.rs` so each test reads as a single shape assertion.
fn lp(sql: &str) -> LogicalPlan {
    parse_sql(sql, &t_provider()).unwrap_or_else(|e| panic!("parse failed for {sql:?}: {e}"))
}

/// Recursively walk a `LogicalPlan` and return the first non-`Project` /
/// non-`Filter` / non-`Sort` / non-`Limit` / non-`Distinct` node — i.e. the
/// underlying body of the plan, useful for asserting that an `Aggregate`
/// sits at the expected depth.
fn unwrap_to_aggregate(plan: &LogicalPlan) -> &LogicalPlan {
    let mut cur = plan;
    loop {
        match cur {
            LogicalPlan::Project { input, .. }
            | LogicalPlan::Filter { input, .. }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Distinct { input } => cur = input,
            _ => return cur,
        }
    }
}

// ---- Test 1: simple SUM(x) AS total ---------------------------------------

/// `SELECT SUM(v) AS total FROM t` (no GROUP BY) — the top of the plan must
/// be a `Project` whose output schema renames the underlying `sum_v` column
/// to `total`. The `Aggregate` below it still produces `sum_v` (the
/// plan-assigned name); only the Project carries the user alias.
#[test]
fn simple_aggregate_alias_renames_output() {
    let plan = lp("SELECT SUM(v) AS total FROM t");
    let LogicalPlan::Project { input, exprs } = &plan else {
        panic!("expected Project at top, got {plan:?}");
    };
    // The Project must reference the aggregate's plan-assigned `sum_v` column
    // under the alias `total`. We accept either the explicit
    // `Expr::Alias(Column("sum_v"), "total")` shape OR a bare
    // `Expr::Column("total")` — the user-visible contract is the alias being
    // honoured, not the exact AST shape.
    assert_eq!(exprs.len(), 1, "one SELECT item -> one Project expr");
    match &exprs[0] {
        Expr::Alias(inner, name) => {
            assert_eq!(name, "total", "Project must rename to user alias");
            assert!(
                matches!(inner.as_ref(), Expr::Column(c) if c == "sum_v"),
                "alias must wrap the plan-assigned `sum_v` column, got {inner:?}"
            );
        }
        other => {
            panic!("expected Expr::Alias(Column(\"sum_v\"), \"total\") in Project, got {other:?}")
        }
    }
    // The Project's input must be the underlying Aggregate (no other layer).
    assert!(
        matches!(input.as_ref(), LogicalPlan::Aggregate { .. }),
        "expected Aggregate under aliasing Project, got {input:?}"
    );
    // Schema sanity: the Project's output column is named `total`.
    let schema = plan.schema().expect("schema");
    assert_eq!(schema.fields.len(), 1);
    assert_eq!(
        schema.fields[0].name, "total",
        "alias must appear in schema"
    );
}

// ---- Test 2: alias usable in ORDER BY -------------------------------------

/// `ORDER BY total` after `SUM(v) AS total` — the Sort node sits above the
/// SELECT-list Project, so its sort expression must resolve against the
/// post-projection schema (where `total` is a real column).
#[test]
fn alias_usable_in_order_by() {
    let plan = lp("SELECT k, SUM(v) AS total FROM t GROUP BY k ORDER BY total DESC");
    let LogicalPlan::Sort { input, sort_exprs } = &plan else {
        panic!("expected Sort at top, got {plan:?}");
    };
    assert_eq!(sort_exprs.len(), 1, "one ORDER BY key");
    assert!(
        matches!(&sort_exprs[0].expr, Expr::Column(n) if n == "total"),
        "ORDER BY must lower to Column(\"total\"), got {:?}",
        sort_exprs[0].expr
    );
    assert!(
        sort_exprs[0].descending,
        "ORDER BY total DESC must set descending=true"
    );
    // The Sort's input is the SELECT-list Project, which exposes `total`.
    assert!(
        matches!(input.as_ref(), LogicalPlan::Project { .. }),
        "expected Project under ORDER BY Sort, got {input:?}"
    );
    // And under that Project, an Aggregate.
    assert!(
        matches!(unwrap_to_aggregate(input), LogicalPlan::Aggregate { .. }),
        "expected Aggregate under SELECT Project, got {input:?}"
    );
    // Schema sanity: the top-level schema still surfaces `total`.
    let schema = plan.schema().expect("schema");
    let names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(
        names.contains(&"total"),
        "schema must expose `total`, got {names:?}"
    );
}

// ---- Test 3: alias usable in HAVING ---------------------------------------

/// `HAVING total > 10` after `SUM(v) AS total` — the HAVING-aware lowerer
/// must accept the alias (not just the underlying aggregate call). The plan
/// shape is `Filter { Project { Aggregate { .. } } }` and the predicate
/// references `Column("total")` so it lines up with the Project's output
/// schema.
#[test]
fn alias_usable_in_having() {
    let plan = lp("SELECT SUM(v) AS total FROM t GROUP BY k HAVING total > 10");
    let LogicalPlan::Filter { input, predicate } = &plan else {
        panic!("expected Filter (HAVING) at top, got {plan:?}");
    };
    // The HAVING predicate must reference the alias, not `sum_v`.
    let Expr::Binary { op, left, .. } = predicate else {
        panic!("HAVING predicate must be Binary, got {predicate:?}");
    };
    assert!(matches!(op, BinaryOp::Gt), "HAVING op should be `>`");
    assert!(
        matches!(left.as_ref(), Expr::Column(n) if n == "total"),
        "HAVING LHS must reference alias `total`, got {left:?}"
    );
    // The Filter's input must be the SELECT-list Project.
    assert!(
        matches!(input.as_ref(), LogicalPlan::Project { .. }),
        "expected Project under HAVING Filter, got {input:?}"
    );
}

/// Symmetry check: writing the *aggregate call* in HAVING (instead of the
/// alias) must still work — the HAVING-aware lowerer rewrites `SUM(v)` to
/// the underlying aggregate output name, and the Project keeps that name
/// reachable through the alias map. This guards against a regression where
/// adding alias support might have inadvertently dropped the original
/// SUM-call form.
#[test]
fn aggregate_call_in_having_still_works_with_alias() {
    let plan = lp("SELECT SUM(v) AS total FROM t GROUP BY k HAVING SUM(v) > 10");
    let LogicalPlan::Filter { input, predicate } = &plan else {
        panic!("expected Filter (HAVING) at top, got {plan:?}");
    };
    let Expr::Binary { left, .. } = predicate else {
        panic!("HAVING predicate must be Binary, got {predicate:?}");
    };
    // Either resolution is valid as long as the Project below carries the
    // matching column. `sum_v` is the plan-assigned name; `total` is the
    // alias the SELECT exposes.
    match left.as_ref() {
        Expr::Column(n) if n == "sum_v" || n == "total" => {}
        other => panic!("HAVING SUM(v) should resolve to either `sum_v` or `total`; got {other:?}"),
    }
    assert!(
        matches!(input.as_ref(), LogicalPlan::Project { .. }),
        "expected Project under HAVING Filter, got {input:?}"
    );
}

// ---- Test 4: end-to-end correctness (GPU-only) -----------------------------

/// Online: register a small batch and confirm the aliased aggregate produces
/// a column literally named `total`. Gated behind `#[ignore]` so it doesn't
/// run on CPU-only CI; matches the convention used by `having_test.rs`.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_aggregate_alias_round_trips() {
    use std::sync::Arc;

    use arrow_array::{Int32Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use craton_bolt::Engine;

    let k = Int32Array::from(vec![1, 1, 2, 2]);
    let v = Int32Array::from(vec![10, 20, 30, 40]);
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(arrow_schema, vec![Arc::new(k), Arc::new(v)]).unwrap();

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", batch).unwrap();

    let h = engine
        .sql("SELECT k, SUM(v) AS total FROM t GROUP BY k ORDER BY total")
        .expect("execute");
    let out = h.record_batch();
    // The schema must expose the alias literally.
    // Bind the schema so the `&str` borrows outlive the statement (the
    // `SchemaRef` returned by `schema()` is otherwise a dropped temporary).
    let schema = out.schema();
    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert!(
        field_names.contains(&"total"),
        "output schema must expose alias `total`, got {field_names:?}"
    );
    // Two groups; SUM(v) per group is 30 (k=1) and 70 (k=2). ORDER BY total
    // ASC means k=1 row comes first.
    assert_eq!(out.num_rows(), 2);
    let k_idx = field_names.iter().position(|n| *n == "k").unwrap();
    let total_idx = field_names.iter().position(|n| *n == "total").unwrap();
    let k_arr = out
        .column(k_idx)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("k Int32");
    let total_arr = out
        .column(total_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("SUM(Int32) widens to Int64");
    assert_eq!((k_arr.value(0), total_arr.value(0)), (1, 30));
    assert_eq!((k_arr.value(1), total_arr.value(1)), (2, 70));
}
