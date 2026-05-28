// SPDX-License-Identifier: Apache-2.0

//! Post-aggregate scalar expressions in SELECT.
//!
//! v0.5: lift the documented "post-aggregate expressions (SUM(price) + 1)
//! are not yet supported" limitation. The SQL frontend now extracts each
//! aggregate call nested inside a scalar SELECT expression as a feed
//! input on the `Aggregate` plan node, and rewrites the surface
//! expression with `Column("<agg_out>")` at each aggregate position.
//! The rewritten expression becomes a computed projection in the
//! post-Aggregate `Project`.
//!
//! Shapes covered:
//!   * `SUM(price) + 1`            — aggregate then literal
//!   * `AVG(qty) * 2`              — Float64 aggregate then literal
//!   * `(SUM(a) + SUM(b)) / 2`     — two aggregates in one expression
//!   * `SUM(x) + 1 AS total`       — alias on the computed projection
//!   * mixed bare + computed in a single SELECT list
//!   * deduplication: `SUM(x) + SUM(x)` only emits one aggregate
//!   * HAVING still works alongside a post-aggregate scalar
//!
//! All tests are offline (logical-plan only) so they don't require a GPU.
//! End-to-end execution is exercised by the broader e2e harness via the
//! same plan shape.

use craton_bolt::plan::{
    parse_sql, AggregateExpr, BinaryOp, DataType, Expr, Field, Literal, LogicalPlan,
    MemTableProvider, Schema,
};

// ---- Fixtures --------------------------------------------------------------

fn t_schema() -> Schema {
    Schema::new(vec![
        Field {
            name: "price".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
        Field {
            name: "qty".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "a".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "b".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "x".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "k".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ])
}

fn provider() -> MemTableProvider {
    MemTableProvider::new().with_table("t", t_schema())
}

// ---- Helpers ----------------------------------------------------------------

/// Walk down a `LogicalPlan` looking for the first (and expected only)
/// `Aggregate` node and return its aggregates list. Panics if the plan
/// has no aggregate at the top of the stack.
fn aggregates_of(plan: &LogicalPlan) -> Vec<AggregateExpr> {
    fn walk(plan: &LogicalPlan) -> Option<&Vec<AggregateExpr>> {
        match plan {
            LogicalPlan::Aggregate { aggregates, .. } => Some(aggregates),
            LogicalPlan::Project { input, .. }
            | LogicalPlan::Filter { input, .. }
            | LogicalPlan::Distinct { input }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Sort { input, .. } => walk(input),
            _ => None,
        }
    }
    walk(plan).expect("plan contains an Aggregate node").clone()
}

/// Return the `Project { exprs }` immediately above the Aggregate.
fn project_exprs(plan: &LogicalPlan) -> Vec<Expr> {
    match plan {
        LogicalPlan::Project { exprs, input } => match input.as_ref() {
            LogicalPlan::Aggregate { .. } => exprs.clone(),
            // Outer wrappers (Filter/Distinct/Limit/Sort) sit above the
            // Project for HAVING / ORDER BY etc.
            _ => panic!("expected Project directly above Aggregate"),
        },
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Distinct { input }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. } => project_exprs(input),
        other => panic!("expected Project above Aggregate, got {other:?}"),
    }
}

// ---- 1. SUM(price) + 1 -----------------------------------------------------

#[test]
fn sum_plus_literal() {
    let plan =
        parse_sql("SELECT SUM(price) + 1 FROM t", &provider()).expect("plan");

    // Single feed aggregate `SUM(price)`.
    let aggs = aggregates_of(&plan);
    assert_eq!(aggs.len(), 1, "only one feed aggregate expected");
    assert!(
        matches!(&aggs[0], AggregateExpr::Sum(Expr::Column(n)) if n == "price"),
        "expected Sum(Column(\"price\")), got {:?}",
        aggs[0]
    );

    // Project should be a single computed expression: sum_price + 1.
    let proj = project_exprs(&plan);
    assert_eq!(proj.len(), 1);
    match &proj[0] {
        Expr::Binary { op: BinaryOp::Add, left, right } => {
            assert!(
                matches!(left.as_ref(), Expr::Column(n) if n == "sum_price"),
                "left side should be Column(sum_price), got {left:?}"
            );
            assert!(
                matches!(right.as_ref(), Expr::Literal(Literal::Int64(1))),
                "right side should be Int64(1), got {right:?}"
            );
        }
        other => panic!("expected Binary(Add, ...) computed proj, got {other:?}"),
    }

    // The overall plan must type-check (the schema check runs implicitly
    // every time we call `.schema()`).
    plan.schema().expect("plan type-checks");
}

// ---- 2. AVG(qty) * 2 -------------------------------------------------------

#[test]
fn avg_times_literal() {
    let plan =
        parse_sql("SELECT AVG(qty) * 2 FROM t", &provider()).expect("plan");

    let aggs = aggregates_of(&plan);
    assert_eq!(aggs.len(), 1);
    assert!(
        matches!(&aggs[0], AggregateExpr::Avg(Expr::Column(n)) if n == "qty"),
        "expected Avg(Column(\"qty\")), got {:?}",
        aggs[0]
    );

    let proj = project_exprs(&plan);
    assert_eq!(proj.len(), 1);
    match &proj[0] {
        Expr::Binary { op: BinaryOp::Mul, left, right } => {
            assert!(
                matches!(left.as_ref(), Expr::Column(n) if n == "avg_qty"),
                "left should be Column(avg_qty), got {left:?}"
            );
            assert!(
                matches!(right.as_ref(), Expr::Literal(Literal::Int64(2))),
                "right should be Int64(2), got {right:?}"
            );
        }
        other => panic!("expected Binary(Mul, ...), got {other:?}"),
    }

    plan.schema().expect("plan type-checks");
}

// ---- 3. (SUM(a) + SUM(b)) / 2 ----------------------------------------------

#[test]
fn two_aggregates_in_one_expression() {
    let plan =
        parse_sql("SELECT (SUM(a) + SUM(b)) / 2 FROM t", &provider()).expect("plan");

    let aggs = aggregates_of(&plan);
    assert_eq!(
        aggs.len(),
        2,
        "expected two feed aggregates (SUM(a) and SUM(b))"
    );
    assert!(matches!(&aggs[0], AggregateExpr::Sum(Expr::Column(n)) if n == "a"));
    assert!(matches!(&aggs[1], AggregateExpr::Sum(Expr::Column(n)) if n == "b"));

    let proj = project_exprs(&plan);
    assert_eq!(proj.len(), 1);
    // Top-level: (Column(sum_a) + Column(sum_b)) / 2.
    match &proj[0] {
        Expr::Binary {
            op: BinaryOp::Div,
            left,
            right,
        } => {
            assert!(matches!(
                right.as_ref(),
                Expr::Literal(Literal::Int64(2))
            ));
            match left.as_ref() {
                Expr::Binary {
                    op: BinaryOp::Add,
                    left: ll,
                    right: lr,
                } => {
                    assert!(
                        matches!(ll.as_ref(), Expr::Column(n) if n == "sum_a"),
                        "inner left should be Column(sum_a), got {ll:?}"
                    );
                    assert!(
                        matches!(lr.as_ref(), Expr::Column(n) if n == "sum_b"),
                        "inner right should be Column(sum_b), got {lr:?}"
                    );
                }
                other => panic!("expected Add under Div, got {other:?}"),
            }
        }
        other => panic!("expected Div at top, got {other:?}"),
    }

    plan.schema().expect("plan type-checks");
}

// ---- 4. Alias on a computed projection -------------------------------------

#[test]
fn computed_projection_with_alias() {
    // `SUM(x) + 1 AS total` — the alias attaches to the whole computed
    // expression (sqlparser parses this as ExprWithAlias { expr: BinaryOp,
    // alias: total }). The output column must be named `total`, not
    // `__expr_0`.
    let plan =
        parse_sql("SELECT SUM(x) + 1 AS total FROM t", &provider()).expect("plan");

    let proj = project_exprs(&plan);
    assert_eq!(proj.len(), 1);
    match &proj[0] {
        Expr::Alias(inner, name) => {
            assert_eq!(name, "total");
            // Inner is the SUM(x) + 1 binary tree.
            assert!(matches!(
                inner.as_ref(),
                Expr::Binary { op: BinaryOp::Add, .. }
            ));
        }
        other => panic!("expected Alias(.., \"total\"), got {other:?}"),
    }

    // Project schema must expose the alias name (not the default placeholder).
    let schema = plan.schema().expect("plan type-checks");
    assert!(
        schema.fields.iter().any(|f| f.name == "total"),
        "post-project schema must expose alias 'total', got {schema:?}"
    );
}

// ---- 5. Mixed bare aggregate + computed expression -------------------------

#[test]
fn mixed_bare_and_computed_in_select() {
    // `SELECT SUM(x), SUM(x) + 1 FROM t` — the bare aggregate appends a
    // SUM(x) feed; the computed expression dedups against the existing
    // sum_x output (so we get exactly one feed aggregate).
    let plan =
        parse_sql("SELECT SUM(x), SUM(x) + 1 FROM t", &provider()).expect("plan");

    let aggs = aggregates_of(&plan);
    assert_eq!(
        aggs.len(),
        1,
        "computed expression must dedup against the bare SUM(x); got {aggs:?}"
    );

    let proj = project_exprs(&plan);
    assert_eq!(proj.len(), 2, "two SELECT items -> two projections");
    // First projection: bare reference to sum_x.
    assert!(
        matches!(&proj[0], Expr::Column(n) if n == "sum_x"),
        "first proj should be Column(sum_x), got {:?}",
        proj[0]
    );
    // Second projection: sum_x + 1.
    match &proj[1] {
        Expr::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            assert!(matches!(left.as_ref(), Expr::Column(n) if n == "sum_x"));
            assert!(matches!(
                right.as_ref(),
                Expr::Literal(Literal::Int64(1))
            ));
        }
        other => panic!("expected Add for second proj, got {other:?}"),
    }
}

// ---- 6. Deduplication across multiple computed projections -----------------

#[test]
fn deduplicates_repeated_aggregate() {
    // `SUM(x) + SUM(x)` — only one feed aggregate emitted.
    let plan =
        parse_sql("SELECT SUM(x) + SUM(x) FROM t", &provider()).expect("plan");

    let aggs = aggregates_of(&plan);
    assert_eq!(
        aggs.len(),
        1,
        "duplicate aggregates in one expression must dedup; got {aggs:?}"
    );

    let proj = project_exprs(&plan);
    assert_eq!(proj.len(), 1);
    match &proj[0] {
        Expr::Binary {
            op: BinaryOp::Add,
            left,
            right,
        } => {
            assert!(
                matches!(left.as_ref(), Expr::Column(n) if n == "sum_x"),
                "both sides should reference the single feed sum_x"
            );
            assert!(
                matches!(right.as_ref(), Expr::Column(n) if n == "sum_x"),
                "both sides should reference the single feed sum_x"
            );
        }
        other => panic!("expected Add, got {other:?}"),
    }
}

// ---- 7. HAVING coexists with a post-aggregate scalar -----------------------

#[test]
fn having_works_with_post_aggregate_select() {
    // SELECT exposes the aggregate both as a bare reference (so the
    // Project keeps `sum_x` visible) and inside a computed expression.
    // HAVING references the bare aggregate — its column ref resolves
    // against the Project schema.
    //
    // (When *every* aggregate in the SELECT list is buried inside a
    // computed expression, the Project no longer exposes the raw
    // aggregate column. HAVING that references those buried aggregates
    // gets a clean "unknown column" error from `validate_having_columns`.
    // See the gotchas note in the commit message.)
    let plan = parse_sql(
        "SELECT k, SUM(x), SUM(x) + 1 FROM t GROUP BY k HAVING SUM(x) > 10",
        &provider(),
    )
    .expect("plan");

    // Aggregate feed list: bare SUM(x) appends; the computed expression
    // dedups against it. Exactly one feed aggregate.
    let aggs = aggregates_of(&plan);
    assert_eq!(aggs.len(), 1);
    assert!(matches!(&aggs[0], AggregateExpr::Sum(Expr::Column(n)) if n == "x"));

    // The top of the plan must be the HAVING Filter; its predicate
    // references the post-aggregate column `sum_x`.
    let predicate = match &plan {
        LogicalPlan::Filter { predicate, .. } => predicate.clone(),
        other => panic!("expected Filter (HAVING) at top, got {other:?}"),
    };
    match predicate {
        Expr::Binary {
            op: BinaryOp::Gt,
            left,
            right,
        } => {
            assert!(matches!(left.as_ref(), Expr::Column(n) if n == "sum_x"));
            assert!(matches!(
                right.as_ref(),
                Expr::Literal(Literal::Int64(10))
            ));
        }
        other => panic!("expected `sum_x > 10` HAVING predicate, got {other:?}"),
    }
}

// ---- 8. Group-by + post-aggregate computed projection ----------------------

#[test]
fn group_by_with_post_aggregate_scalar() {
    // The group key passes through unchanged; the aggregate is extracted
    // and the surface expression becomes a computed projection.
    let plan = parse_sql(
        "SELECT k, SUM(x) * 2 AS doubled FROM t GROUP BY k",
        &provider(),
    )
    .expect("plan");

    let aggs = aggregates_of(&plan);
    assert_eq!(aggs.len(), 1);
    assert!(matches!(&aggs[0], AggregateExpr::Sum(Expr::Column(n)) if n == "x"));

    let proj = project_exprs(&plan);
    assert_eq!(proj.len(), 2, "two SELECT items: k and the computed expr");
    // First projection: group key passthrough.
    assert!(
        matches!(&proj[0], Expr::Column(n) if n == "k"),
        "first proj should be group key k, got {:?}",
        proj[0]
    );
    // Second projection: Alias(sum_x * 2, "doubled").
    match &proj[1] {
        Expr::Alias(inner, name) => {
            assert_eq!(name, "doubled");
            match inner.as_ref() {
                Expr::Binary {
                    op: BinaryOp::Mul,
                    left,
                    right,
                } => {
                    assert!(matches!(left.as_ref(), Expr::Column(n) if n == "sum_x"));
                    assert!(matches!(
                        right.as_ref(),
                        Expr::Literal(Literal::Int64(2))
                    ));
                }
                other => panic!("expected Mul inside Alias, got {other:?}"),
            }
        }
        other => panic!("expected Alias(.., \"doubled\"), got {other:?}"),
    }

    // Plan must type-check.
    let schema = plan.schema().expect("plan type-checks");
    assert_eq!(schema.fields.len(), 2);
    assert_eq!(schema.fields[0].name, "k");
    assert_eq!(schema.fields[1].name, "doubled");
}

// ---- 9. Negative: bare aggregate-free expression still requires GROUP BY ----

#[test]
fn non_aggregate_non_groupby_still_rejected() {
    // Regression guard: even though we now accept *aggregated* expressions,
    // a SELECT item with no aggregate that isn't a group key must still
    // be rejected. (Otherwise the v0.4 GROUP BY contract silently breaks.)
    let res = parse_sql("SELECT x, SUM(qty) FROM t GROUP BY k", &provider());
    assert!(res.is_err(), "x is neither a group key nor an aggregate");
    let msg = format!("{}", res.unwrap_err());
    assert!(
        msg.contains("GROUP BY"),
        "error message should mention GROUP BY; got: {msg}"
    );
}
