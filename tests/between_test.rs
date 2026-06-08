// SPDX-License-Identifier: Apache-2.0

//! Integration tests for SQL `BETWEEN` / `NOT BETWEEN`.
//!
//! The frontend desugars `expr BETWEEN low AND high` into
//! `(expr >= low) AND (expr <= high)`, and the negated form into
//! `(expr < low) OR (expr > high)` (DeMorgan). Neither form introduces
//! any new IR node — the planner and executor see plain
//! `BinaryOp::{And, Or, Lt, LtEq, Gt, GtEq}` trees — so these tests pin
//! the lowered *shape* (no GPU device required) and confirm the
//! end-to-end plan lowers without error against both int and float
//! columns.

use craton_bolt::plan::{
    lower_physical, parse_sql, BinaryOp, DataType, Expr, Field, Literal, LogicalPlan,
    MemTableProvider, Schema,
};

// ---- Fixtures --------------------------------------------------------------

fn t_schema() -> Schema {
    Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "qty".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "price".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
    ])
}

fn t_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("t", t_schema())
}

// ---- Helpers ---------------------------------------------------------------

/// Descend through optional wrapper layers (Project / Limit / Sort / ...)
/// until we hit a `Filter` and return a reference to its predicate.
fn find_filter_predicate(plan: &LogicalPlan) -> Option<&Expr> {
    match plan {
        LogicalPlan::Filter { predicate, .. } => Some(predicate),
        LogicalPlan::Project { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Distinct { input, .. }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Window { input, .. }
        | LogicalPlan::Sort { input, .. } => find_filter_predicate(input),
        LogicalPlan::Union { inputs } => inputs.iter().find_map(find_filter_predicate),
        LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => {
            find_filter_predicate(left).or_else(|| find_filter_predicate(right))
        }
        LogicalPlan::Scan { .. } => None,
    }
}

/// Peel away `Expr::Alias` wrappers at the root.
fn strip_alias(e: &Expr) -> &Expr {
    let mut cur = e;
    while let Expr::Alias(inner, _) = cur {
        cur = inner.as_ref();
    }
    cur
}

/// Assert `e` is `Expr::Binary { op, .. }` and return its (left, right).
fn unwrap_binary<'a>(e: &'a Expr, op: BinaryOp) -> (&'a Expr, &'a Expr) {
    match strip_alias(e) {
        Expr::Binary {
            op: actual,
            left,
            right,
        } if *actual == op => (left.as_ref(), right.as_ref()),
        other => panic!("expected Expr::Binary {{ op: {op:?}, .. }}, got {other:?}"),
    }
}

/// Assert that `e` is `Expr::Column(name)`.
fn assert_column(e: &Expr, name: &str) {
    match strip_alias(e) {
        Expr::Column(n) if n == name => {}
        other => panic!("expected Expr::Column({name:?}), got {other:?}"),
    }
}

/// Assert that `e` is `Expr::Literal(Int64(v))`.
fn assert_int_literal(e: &Expr, v: i64) {
    match strip_alias(e) {
        Expr::Literal(Literal::Int64(actual)) if *actual == v => {}
        other => panic!("expected Int64 literal {v}, got {other:?}"),
    }
}

/// Assert that `e` is `Expr::Literal(Float64(v))` (bitwise-equal).
fn assert_float_literal(e: &Expr, v: f64) {
    match strip_alias(e) {
        Expr::Literal(Literal::Float64(actual)) if actual.to_bits() == v.to_bits() => {}
        other => panic!("expected Float64 literal {v}, got {other:?}"),
    }
}

// ---- Plain BETWEEN (int) ---------------------------------------------------

/// `WHERE qty BETWEEN 5 AND 10` must lower to
/// `(qty >= 5) AND (qty <= 10)`.
#[test]
fn between_int_lowers_to_ge_and_le() {
    let sql = "SELECT id FROM t WHERE qty BETWEEN 5 AND 10";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let pred = find_filter_predicate(&plan).expect("Filter present");

    // Top-level AND.
    let (left, right) = unwrap_binary(pred, BinaryOp::And);

    // Left side: qty >= 5
    let (ge_l, ge_r) = unwrap_binary(left, BinaryOp::GtEq);
    assert_column(ge_l, "qty");
    assert_int_literal(ge_r, 5);

    // Right side: qty <= 10
    let (le_l, le_r) = unwrap_binary(right, BinaryOp::LtEq);
    assert_column(le_l, "qty");
    assert_int_literal(le_r, 10);

    // And the whole plan must lower without error.
    lower_physical(&plan).expect("lower BETWEEN int");
}

// ---- NOT BETWEEN (int) -----------------------------------------------------

/// `WHERE qty NOT BETWEEN 5 AND 10` must lower (via DeMorgan) to
/// `(qty < 5) OR (qty > 10)`.
#[test]
fn not_between_int_lowers_to_lt_or_gt() {
    let sql = "SELECT id FROM t WHERE qty NOT BETWEEN 5 AND 10";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let pred = find_filter_predicate(&plan).expect("Filter present");

    // Top-level OR.
    let (left, right) = unwrap_binary(pred, BinaryOp::Or);

    // Left side: qty < 5
    let (lt_l, lt_r) = unwrap_binary(left, BinaryOp::Lt);
    assert_column(lt_l, "qty");
    assert_int_literal(lt_r, 5);

    // Right side: qty > 10
    let (gt_l, gt_r) = unwrap_binary(right, BinaryOp::Gt);
    assert_column(gt_l, "qty");
    assert_int_literal(gt_r, 10);

    lower_physical(&plan).expect("lower NOT BETWEEN int");
}

// ---- Plain BETWEEN (float) -------------------------------------------------

/// Float column variant: `WHERE price BETWEEN 1.5 AND 9.75` desugars the
/// same way and lowers cleanly. The literals must arrive as Float64.
#[test]
fn between_float_lowers_to_ge_and_le() {
    let sql = "SELECT id FROM t WHERE price BETWEEN 1.5 AND 9.75";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let pred = find_filter_predicate(&plan).expect("Filter present");

    let (left, right) = unwrap_binary(pred, BinaryOp::And);

    let (ge_l, ge_r) = unwrap_binary(left, BinaryOp::GtEq);
    assert_column(ge_l, "price");
    assert_float_literal(ge_r, 1.5);

    let (le_l, le_r) = unwrap_binary(right, BinaryOp::LtEq);
    assert_column(le_l, "price");
    assert_float_literal(le_r, 9.75);

    lower_physical(&plan).expect("lower BETWEEN float");
}

// ---- NOT BETWEEN (float) ---------------------------------------------------

/// Float `NOT BETWEEN`: same DeMorgan rewrite.
#[test]
fn not_between_float_lowers_to_lt_or_gt() {
    let sql = "SELECT id FROM t WHERE price NOT BETWEEN 1.5 AND 9.75";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let pred = find_filter_predicate(&plan).expect("Filter present");

    let (left, right) = unwrap_binary(pred, BinaryOp::Or);

    let (lt_l, lt_r) = unwrap_binary(left, BinaryOp::Lt);
    assert_column(lt_l, "price");
    assert_float_literal(lt_r, 1.5);

    let (gt_l, gt_r) = unwrap_binary(right, BinaryOp::Gt);
    assert_column(gt_l, "price");
    assert_float_literal(gt_r, 9.75);

    lower_physical(&plan).expect("lower NOT BETWEEN float");
}
