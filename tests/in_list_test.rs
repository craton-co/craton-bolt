// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the SQL `IN (...)` list operator.
//!
//! The SQL frontend desugars `<probe> [NOT] IN (v1, v2, ..., vN)` into a
//! chain of element-wise comparisons reusing existing binary operators:
//!
//!   * `IN`     → `(probe = v1) OR  (probe = v2) OR  ...`
//!   * `NOT IN` → `(probe <> v1) AND (probe <> v2) AND ...`
//!
//! These tests pin the lowered plan shape (no GPU needed) and the
//! cap / empty-list behaviour. They live alongside the other plan-
//! shape tests (`having_test.rs`, `is_null_test.rs`).

use craton_bolt::plan::{
    parse_sql, BinaryOp, DataType, Expr, Field, Literal, LogicalPlan, MemTableProvider, Schema,
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

/// Walk a logical plan looking for the topmost `Filter` and return its
/// predicate. Tests that pin the IN-list desugaring all look at this
/// single predicate slot.
fn filter_predicate(plan: &LogicalPlan) -> &Expr {
    fn find<'a>(p: &'a LogicalPlan) -> Option<&'a Expr> {
        match p {
            LogicalPlan::Filter { predicate, .. } => Some(predicate),
            LogicalPlan::Project { input, .. }
            | LogicalPlan::Distinct { input }
            | LogicalPlan::Sort { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Aggregate { input, .. } => find(input),
            _ => None,
        }
    }
    find(plan).expect("expected a Filter node somewhere in the plan")
}

/// Count the number of `BinaryOp::Eq` (or `NotEq`) leaves in the predicate
/// tree. Used to assert that an IN-list of N values produces exactly N
/// element-wise comparisons.
fn count_op(e: &Expr, target: BinaryOp) -> usize {
    match e {
        Expr::Binary { op, left, right } => {
            let here = if *op == target { 1 } else { 0 };
            here + count_op(left, target) + count_op(right, target)
        }
        Expr::Unary { operand, .. } => count_op(operand, target),
        Expr::Alias(inner, _) => count_op(inner, target),
        _ => 0,
    }
}

/// True if any node in the expression tree matches `op`.
fn contains_op(e: &Expr, target: BinaryOp) -> bool {
    match e {
        Expr::Binary { op, left, right } => {
            *op == target || contains_op(left, target) || contains_op(right, target)
        }
        Expr::Unary { operand, .. } => contains_op(operand, target),
        Expr::Alias(inner, _) => contains_op(inner, target),
        _ => false,
    }
}

// ---- IN (...) — plain OR-chain --------------------------------------------

/// `SELECT v FROM t WHERE k IN (1, 2, 3)` must lower to a predicate
/// shaped as `((k = 1) OR (k = 2)) OR (k = 3)` — three Eq leaves combined
/// by OR. We don't pin the exact associativity (left- vs right-leaning)
/// because that's an implementation detail of the chain builder; what
/// matters is that the operator counts and the operand identity (column
/// `k`, literals 1/2/3) are correct.
#[test]
fn in_list_three_values_lowers_to_or_chain_of_eq() {
    let sql = "SELECT v FROM t WHERE k IN (1, 2, 3)";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let pred = filter_predicate(&plan);

    assert_eq!(
        count_op(pred, BinaryOp::Eq),
        3,
        "expected 3 Eq leaves (one per IN value), got predicate: {pred:?}",
    );
    assert!(
        contains_op(pred, BinaryOp::Or),
        "expected at least one OR combiner in predicate: {pred:?}",
    );
    assert_eq!(
        count_op(pred, BinaryOp::NotEq),
        0,
        "plain IN must not produce any NotEq leaves: {pred:?}",
    );
    assert!(
        !contains_op(pred, BinaryOp::And),
        "plain IN must not produce any AND combiners: {pred:?}",
    );
}

/// A five-value list — exercises the chain builder past the trivial 1-2
/// element edge cases and confirms it scales linearly. Also verifies that
/// the probe expression (here a bare column ref `k`) appears on the left
/// of every Eq leaf, not folded into the literal side.
#[test]
fn in_list_five_values_produces_five_eq_leaves() {
    let sql = "SELECT v FROM t WHERE k IN (10, 20, 30, 40, 50)";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let pred = filter_predicate(&plan);

    assert_eq!(
        count_op(pred, BinaryOp::Eq),
        5,
        "expected 5 Eq leaves, predicate: {pred:?}",
    );
    // Walk every Eq leaf and confirm its left side is `Column("k")`. The
    // probe expression is cloned into each branch; the desugaring must
    // not commute the operands.
    let mut eq_leaves: Vec<(&Expr, &Expr)> = Vec::new();
    fn collect<'a>(e: &'a Expr, out: &mut Vec<(&'a Expr, &'a Expr)>) {
        match e {
            Expr::Binary {
                op: BinaryOp::Eq,
                left,
                right,
            } => out.push((left, right)),
            Expr::Binary { left, right, .. } => {
                collect(left, out);
                collect(right, out);
            }
            _ => {}
        }
    }
    collect(pred, &mut eq_leaves);
    assert_eq!(eq_leaves.len(), 5);
    for (i, (lhs, rhs)) in eq_leaves.iter().enumerate() {
        assert!(
            matches!(lhs, Expr::Column(n) if n == "k"),
            "Eq leaf #{i} left side should be Column(\"k\"), got {lhs:?}",
        );
        assert!(
            matches!(rhs, Expr::Literal(Literal::Int64(_))),
            "Eq leaf #{i} right side should be an Int64 literal, got {rhs:?}",
        );
    }
}

// ---- NOT IN (...) — AND-chain via De Morgan -------------------------------

/// `NOT IN` must desugar to an AND-chain of `<>` per De Morgan, *not* to
/// a logical NOT over the OR-of-Eq form (the planner does not yet expose
/// a NOT node). The shape is `((k <> 1) AND (k <> 2)) AND (k <> 3)`.
#[test]
fn not_in_list_lowers_to_and_chain_of_neq() {
    let sql = "SELECT v FROM t WHERE k NOT IN (1, 2, 3)";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let pred = filter_predicate(&plan);

    assert_eq!(
        count_op(pred, BinaryOp::NotEq),
        3,
        "expected 3 NotEq leaves, got predicate: {pred:?}",
    );
    assert!(
        contains_op(pred, BinaryOp::And),
        "NOT IN must produce at least one AND combiner: {pred:?}",
    );
    assert_eq!(
        count_op(pred, BinaryOp::Eq),
        0,
        "NOT IN must not produce any Eq leaves (De Morgan path): {pred:?}",
    );
    assert!(
        !contains_op(pred, BinaryOp::Or),
        "NOT IN must not produce any OR combiners (De Morgan path): {pred:?}",
    );
}

// ---- Single-element list — chain builder edge case ------------------------

/// A one-element `IN` should collapse to a bare `(k = v1)` with no OR
/// combiner. This pins the chain-builder's no-prior-acc branch.
#[test]
fn in_list_single_value_collapses_to_single_eq() {
    let sql = "SELECT v FROM t WHERE k IN (42)";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let pred = filter_predicate(&plan);

    assert_eq!(count_op(pred, BinaryOp::Eq), 1);
    assert!(
        !contains_op(pred, BinaryOp::Or),
        "single-element IN should not introduce an OR: {pred:?}",
    );
    // Bare Eq at the top.
    let Expr::Binary {
        op: BinaryOp::Eq,
        left,
        right,
    } = pred
    else {
        panic!("expected a bare Eq, got {pred:?}");
    };
    assert!(matches!(left.as_ref(), Expr::Column(n) if n == "k"));
    assert!(matches!(right.as_ref(), Expr::Literal(Literal::Int64(42))));
}

// ---- Empty list -----------------------------------------------------------

/// `GenericDialect::supports_in_empty_list` is `false` in sqlparser 0.52,
/// so `WHERE k IN ()` is rejected at parse time before our lowerer ever
/// sees it. We assert that path here so the test pins the user-facing
/// behaviour: someone reading the test should expect an error message,
/// not a constant-folded plan. (The lowerer *does* still constant-fold
/// empty lists internally for robustness; that's the safety-net path
/// covered by inspection of `lower_in_list`.)
#[test]
fn empty_in_list_rejected_at_parse_time() {
    let sql = "SELECT v FROM t WHERE k IN ()";
    let result = parse_sql(sql, &t_provider());
    assert!(
        result.is_err(),
        "empty IN () should be rejected (parse error), got: {result:?}",
    );
}

// ---- Cap rejection ---------------------------------------------------------

/// Lists longer than the 64-value cap must produce a clear error message
/// pointing the user at a JOIN. The exact cap is implementation detail,
/// but the "use a JOIN" guidance is load-bearing.
#[test]
fn in_list_over_cap_rejected_with_join_hint() {
    // 65 values: one past the documented MAX_IN_LIST_VALUES = 64.
    let values: Vec<String> = (0..65).map(|i| i.to_string()).collect();
    let sql = format!("SELECT v FROM t WHERE k IN ({})", values.join(", "));

    let err = match parse_sql(&sql, &t_provider()) {
        Ok(plan) => panic!("expected over-cap IN list to error, got plan: {plan:?}"),
        Err(e) => format!("{e}"),
    };
    let msg = err.to_ascii_lowercase();
    assert!(
        msg.contains("in with") || msg.contains("> 64") || msg.contains("64 values"),
        "error should reference the cap, got: {err}",
    );
    assert!(
        msg.contains("join"),
        "error should suggest a JOIN, got: {err}",
    );
}

/// Exactly at the 64-value cap must still succeed.
#[test]
fn in_list_at_cap_succeeds() {
    let values: Vec<String> = (0..64).map(|i| i.to_string()).collect();
    let sql = format!("SELECT v FROM t WHERE k IN ({})", values.join(", "));
    let plan = parse_sql(&sql, &t_provider()).expect("at-cap IN list should lower");
    let pred = filter_predicate(&plan);
    assert_eq!(count_op(pred, BinaryOp::Eq), 64);
}

// ---- Interaction with other expressions ------------------------------------

/// IN inside a larger boolean expression should compose cleanly with the
/// surrounding `AND` — i.e. `k IN (1, 2) AND v > 0` lowers to a normal
/// Binary(And, IN-chain, Gt(v, 0)) tree. This pins that the desugaring
/// returns an `Expr` (not some sentinel) and lives inside whatever
/// boolean context the caller built.
#[test]
fn in_list_composes_with_outer_and() {
    let sql = "SELECT v FROM t WHERE k IN (1, 2) AND v > 0";
    let plan = parse_sql(sql, &t_provider()).expect("parse");
    let pred = filter_predicate(&plan);

    // Top-level AND between the IN-chain (Eq + OR) and the `v > 0` test (Gt).
    let Expr::Binary {
        op: BinaryOp::And,
        left,
        right,
    } = pred
    else {
        panic!("expected top-level AND, got {pred:?}");
    };
    // The IN-chain side has two Eqs and at least one OR.
    let in_side = if count_op(left, BinaryOp::Eq) == 2 {
        left
    } else {
        right
    };
    assert_eq!(count_op(in_side, BinaryOp::Eq), 2);
    assert!(contains_op(in_side, BinaryOp::Or));
}
