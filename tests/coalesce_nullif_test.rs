// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `COALESCE(a, b, c, ...)` and `NULLIF(a, b)`.
//!
//! v0.5 / M2 SQL scalar completeness: both functions desugar at the SQL
//! frontend into the existing `Expr::Case` IR variant so no new IR node
//! or executor path is required. The physical-plan boundary still rejects
//! CASE (and therefore COALESCE / NULLIF) with the "CASE not yet lowered
//! to GPU; coming in a follow-up" `Plan` error — execution-level
//! coverage will join this file when CASE codegen lands.
//!
//! The tests below pin the *parse-only* contract:
//!
//!   * `COALESCE(a, b, c)` lowers to `CASE WHEN a IS NOT NULL THEN a
//!     WHEN b IS NOT NULL THEN b ELSE c END`,
//!   * `COALESCE(a, b)` lowers to a single WHEN branch + ELSE,
//!   * `COALESCE(a)` collapses to `a` itself (no vestigial CASE),
//!   * `COALESCE()` is rejected,
//!   * `NULLIF(a, b)` lowers to `CASE WHEN a = b THEN NULL ELSE a END`,
//!   * `NULLIF` with a non-2 arity is rejected,
//!   * The function name match is case-insensitive (`coalesce`, `Nullif`,
//!     etc. all work).

use craton_bolt::plan::{
    parse_sql, BinaryOp, DataType, Expr, Field, Literal, LogicalPlan, MemTableProvider, Schema,
    UnaryOp,
};

// ---- Fixture ----------------------------------------------------------------

fn t_schema() -> Schema {
    Schema::new(vec![
        Field {
            name: "a".into(),
            dtype: DataType::Int32,
            nullable: true,
        },
        Field {
            name: "b".into(),
            dtype: DataType::Int32,
            nullable: true,
        },
        Field {
            name: "c".into(),
            dtype: DataType::Int32,
            nullable: true,
        },
        Field {
            name: "d".into(),
            dtype: DataType::Int32,
            nullable: true,
        },
    ])
}

fn t_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("t", t_schema())
}

/// Walk a logical plan down to the first `Project` node and return its
/// expression list. Every query in this file lowers to `Project { exprs,
/// input: Scan { .. } }`, so this is the central inspection hook.
fn first_project_exprs(plan: &LogicalPlan) -> &[Expr] {
    match plan {
        LogicalPlan::Project { exprs, .. } => exprs,
        LogicalPlan::Distinct { input }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::Filter { input, .. } => first_project_exprs(input),
        other => panic!("expected a Project somewhere in the plan, got {other:?}"),
    }
}

/// Peel any SELECT-list alias to get at the inner expression. COALESCE /
/// NULLIF tests routinely `AS name` the projection to make the alias
/// path easy to follow; the inner shape is what we want to assert on.
fn strip_alias(e: &Expr) -> &Expr {
    match e {
        Expr::Alias(inner, _) => strip_alias(inner),
        other => other,
    }
}

// ---- COALESCE: shape -------------------------------------------------------

/// `COALESCE(a, b, c)` desugars to a 2-branch CASE with `c` as ELSE.
/// Each branch's condition is `arg IS NOT NULL`, the THEN is the same
/// arg expression (cloned), and source order is preserved.
#[test]
fn coalesce_three_args_desugars_to_two_branch_case_with_else() {
    let sql = "SELECT COALESCE(a, b, c) AS r FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let exprs = first_project_exprs(&plan);
    assert_eq!(exprs.len(), 1);

    let case = strip_alias(&exprs[0]);
    let (branches, else_branch) = match case {
        Expr::Case {
            branches,
            else_branch,
        } => (branches, else_branch),
        other => panic!("expected Expr::Case, got {other:?}"),
    };
    assert_eq!(branches.len(), 2, "two non-last args become WHEN branches");

    // Branch 0: `a IS NOT NULL THEN a`
    assert_branch_is_not_null_arm(&branches[0], "a");
    // Branch 1: `b IS NOT NULL THEN b`
    assert_branch_is_not_null_arm(&branches[1], "b");

    // ELSE: the last argument `c`, unwrapped.
    let else_expr = else_branch
        .as_deref()
        .expect("3-arg COALESCE must populate ELSE with the last argument");
    match else_expr {
        Expr::Column(n) => assert_eq!(n, "c", "ELSE arm is the last argument"),
        other => panic!("expected ELSE = Column(c), got {other:?}"),
    }
}

/// `COALESCE(a, b)` is the binary form: one WHEN branch + ELSE. The
/// branch condition is `a IS NOT NULL`, THEN = `a`, ELSE = `b`.
#[test]
fn coalesce_two_args_desugars_to_one_branch_case() {
    let sql = "SELECT COALESCE(a, b) AS r FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let exprs = first_project_exprs(&plan);
    let case = strip_alias(&exprs[0]);
    let (branches, else_branch) = match case {
        Expr::Case {
            branches,
            else_branch,
        } => (branches, else_branch),
        other => panic!("expected Expr::Case, got {other:?}"),
    };
    assert_eq!(branches.len(), 1, "2-arg COALESCE has one WHEN branch");
    assert_branch_is_not_null_arm(&branches[0], "a");
    let else_expr = else_branch
        .as_deref()
        .expect("2-arg COALESCE must populate ELSE");
    match else_expr {
        Expr::Column(n) => assert_eq!(n, "b"),
        other => panic!("expected ELSE = Column(b), got {other:?}"),
    }
}

/// `COALESCE(a)` with a single argument is identity: the lowered form is
/// just `a` itself, with no surrounding CASE. This keeps the IR
/// tractable for downstream rewrites that would otherwise have to
/// special-case a trivial one-branch CASE.
#[test]
fn coalesce_single_arg_collapses_to_operand() {
    let sql = "SELECT COALESCE(a) AS r FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let exprs = first_project_exprs(&plan);
    let inner = strip_alias(&exprs[0]);
    match inner {
        Expr::Column(n) => assert_eq!(n, "a", "single-arg COALESCE is identity"),
        other => panic!("expected Column(a), got {other:?}"),
    }
}

/// Four-argument COALESCE: three WHEN branches + ELSE. Pins that the
/// desugar scales linearly with `n` and that branch order is preserved
/// past `n == 3`.
#[test]
fn coalesce_four_args_desugars_to_three_branch_case() {
    let sql = "SELECT COALESCE(a, b, c, d) AS r FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let exprs = first_project_exprs(&plan);
    let case = strip_alias(&exprs[0]);
    let (branches, else_branch) = match case {
        Expr::Case {
            branches,
            else_branch,
        } => (branches, else_branch),
        other => panic!("expected Expr::Case, got {other:?}"),
    };
    assert_eq!(branches.len(), 3);
    assert_branch_is_not_null_arm(&branches[0], "a");
    assert_branch_is_not_null_arm(&branches[1], "b");
    assert_branch_is_not_null_arm(&branches[2], "c");
    match else_branch.as_deref() {
        Some(Expr::Column(n)) => assert_eq!(n, "d"),
        other => panic!("expected ELSE = Column(d), got {other:?}"),
    }
}

/// `COALESCE()` with zero arguments is a SQL error; the frontend must
/// reject it before lowering reaches CASE construction. The exact error
/// message comes either from our desugar layer ("COALESCE requires …")
/// or from sqlparser itself if it refuses the empty-arg form at parse
/// time — both are acceptable, the contract is just "this does not
/// silently produce a plan".
#[test]
fn coalesce_zero_args_rejected() {
    let sql = "SELECT COALESCE() AS r FROM t";
    let _err = parse_sql(sql, &t_provider())
        .expect_err("zero-arg COALESCE must be rejected at parse time");
}

/// The function name match is case-insensitive. `coalesce`, `Coalesce`,
/// and `COALESCE` all lower to the same CASE shape. We pin the lowered
/// branch count to confirm the interception path fired (vs falling
/// through to the "scalar function calls are not supported" branch).
#[test]
fn coalesce_lowercase_name_recognised() {
    for spelling in ["coalesce", "Coalesce", "COALESCE", "cOaLeScE"] {
        let sql = format!("SELECT {spelling}(a, b) AS r FROM t");
        let plan = parse_sql(&sql, &t_provider())
            .unwrap_or_else(|e| panic!("parse must succeed for '{spelling}': {e}"));
        let exprs = first_project_exprs(&plan);
        match strip_alias(&exprs[0]) {
            Expr::Case { branches, .. } => assert_eq!(
                branches.len(),
                1,
                "2-arg COALESCE always lowers to a 1-branch CASE (spelling: {spelling})"
            ),
            other => panic!(
                "expected Expr::Case for spelling '{spelling}', got {other:?}"
            ),
        }
    }
}

// ---- NULLIF: shape ---------------------------------------------------------

/// `NULLIF(a, b)` desugars to `CASE WHEN a = b THEN NULL ELSE a END`.
/// The WHEN is a single Eq comparison, the THEN is `Literal::Null`, and
/// the ELSE is the first argument.
#[test]
fn nullif_desugars_to_case_eq_then_null_else_first_arg() {
    let sql = "SELECT NULLIF(a, b) AS r FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let exprs = first_project_exprs(&plan);
    let case = strip_alias(&exprs[0]);
    let (branches, else_branch) = match case {
        Expr::Case {
            branches,
            else_branch,
        } => (branches, else_branch),
        other => panic!("expected Expr::Case, got {other:?}"),
    };
    assert_eq!(branches.len(), 1, "NULLIF always lowers to a 1-branch CASE");

    // WHEN: `a = b`.
    match &branches[0].0 {
        Expr::Binary {
            op: BinaryOp::Eq,
            left,
            right,
        } => {
            match left.as_ref() {
                Expr::Column(n) => assert_eq!(n, "a"),
                other => panic!("expected LHS = Column(a), got {other:?}"),
            }
            match right.as_ref() {
                Expr::Column(n) => assert_eq!(n, "b"),
                other => panic!("expected RHS = Column(b), got {other:?}"),
            }
        }
        other => panic!("expected Binary Eq on WHEN, got {other:?}"),
    }

    // THEN: NULL literal.
    match &branches[0].1 {
        Expr::Literal(Literal::Null) => {}
        other => panic!("expected THEN = Literal::Null, got {other:?}"),
    }

    // ELSE: first argument, unwrapped.
    match else_branch.as_deref() {
        Some(Expr::Column(n)) => assert_eq!(n, "a"),
        other => panic!("expected ELSE = Column(a), got {other:?}"),
    }
}

/// `NULLIF` is strictly binary; one-arg / three-arg / zero-arg forms are
/// rejected at parse time. We assert the *rejection* rather than the
/// exact message text: depending on the input, the rejection can come
/// from sqlparser itself or from our arity guard in `lower_nullif`,
/// and either is acceptable — the user-visible contract is "this does
/// not silently produce a plan".
#[test]
fn nullif_arity_rejected_outside_two() {
    for sql in [
        "SELECT NULLIF() AS r FROM t",
        "SELECT NULLIF(a) AS r FROM t",
        "SELECT NULLIF(a, b, c) AS r FROM t",
    ] {
        let _err = parse_sql(sql, &t_provider())
            .err()
            .unwrap_or_else(|| panic!("non-binary NULLIF must be rejected: {sql}"));
    }
}

/// The one-arg / three-arg NULLIF rejections specifically go through our
/// frontend (they parse just fine as a Function call but fail the
/// arity check in `lower_nullif`). Pin the human-readable error message
/// from *that* path so a regression that drops the arity check
/// produces a clear test failure rather than silently lowering.
#[test]
fn nullif_three_arg_error_mentions_function_name() {
    let sql = "SELECT NULLIF(a, b, c) AS r FROM t";
    let err = parse_sql(sql, &t_provider())
        .expect_err("3-arg NULLIF must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.to_ascii_uppercase().contains("NULLIF"),
        "frontend arity rejection should mention NULLIF, got: {msg}"
    );
}

/// Case-insensitive name match for NULLIF as well.
#[test]
fn nullif_lowercase_name_recognised() {
    for spelling in ["nullif", "NullIf", "NULLIF"] {
        let sql = format!("SELECT {spelling}(a, b) AS r FROM t");
        let plan = parse_sql(&sql, &t_provider())
            .unwrap_or_else(|e| panic!("parse must succeed for '{spelling}': {e}"));
        let exprs = first_project_exprs(&plan);
        match strip_alias(&exprs[0]) {
            Expr::Case { branches, .. } => assert_eq!(branches.len(), 1, "spelling: {spelling}"),
            other => panic!(
                "expected Expr::Case for spelling '{spelling}', got {other:?}"
            ),
        }
    }
}

// ---- Nested / interaction tests --------------------------------------------

/// COALESCE arguments may themselves be expressions, not just bare
/// columns. `COALESCE(a + 1, b)` desugars to a CASE whose first WHEN
/// tests `a + 1 IS NOT NULL` and whose THEN is the same `a + 1` sum.
#[test]
fn coalesce_accepts_expression_arguments() {
    let sql = "SELECT COALESCE(a + 1, b) AS r FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let exprs = first_project_exprs(&plan);
    let case = strip_alias(&exprs[0]);
    match case {
        Expr::Case {
            branches,
            else_branch,
        } => {
            assert_eq!(branches.len(), 1);
            // WHEN: (a + 1) IS NOT NULL
            match &branches[0].0 {
                Expr::Unary {
                    op: UnaryOp::IsNotNull,
                    operand,
                } => match operand.as_ref() {
                    Expr::Binary {
                        op: BinaryOp::Add, ..
                    } => {}
                    other => panic!("expected operand = (a + 1), got {other:?}"),
                },
                other => panic!("expected `IS NOT NULL` on branch condition, got {other:?}"),
            }
            // THEN: a + 1 (cloned)
            match &branches[0].1 {
                Expr::Binary {
                    op: BinaryOp::Add, ..
                } => {}
                other => panic!("expected THEN = (a + 1), got {other:?}"),
            }
            // ELSE: column b
            match else_branch.as_deref() {
                Some(Expr::Column(n)) => assert_eq!(n, "b"),
                other => panic!("expected ELSE = Column(b), got {other:?}"),
            }
        }
        other => panic!("expected Expr::Case, got {other:?}"),
    }
}

/// `COALESCE(NULL, a)` is the canonical "default-when-null" idiom. The
/// `NULL` literal lowers as `Literal::Null`, sits in the WHEN-test
/// position, and the ELSE collapses to the second argument. We pin the
/// shape so a future literal-fold pass that constants-out a `NULL IS
/// NOT NULL` test would surface its rewrite here as a test failure.
#[test]
fn coalesce_with_null_literal_arg_preserves_shape() {
    let sql = "SELECT COALESCE(NULL, a) AS r FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let exprs = first_project_exprs(&plan);
    let case = strip_alias(&exprs[0]);
    match case {
        Expr::Case {
            branches,
            else_branch,
        } => {
            assert_eq!(branches.len(), 1);
            // WHEN: NULL IS NOT NULL
            match &branches[0].0 {
                Expr::Unary {
                    op: UnaryOp::IsNotNull,
                    operand,
                } => match operand.as_ref() {
                    Expr::Literal(Literal::Null) => {}
                    other => panic!("expected operand = NULL literal, got {other:?}"),
                },
                other => panic!("expected `IS NOT NULL`, got {other:?}"),
            }
            // ELSE: a
            match else_branch.as_deref() {
                Some(Expr::Column(n)) => assert_eq!(n, "a"),
                other => panic!("expected ELSE = Column(a), got {other:?}"),
            }
        }
        other => panic!("expected Expr::Case, got {other:?}"),
    }
}

// ---- Helpers ---------------------------------------------------------------

/// Assert a CASE branch is shaped `<column> IS NOT NULL THEN <same column>`.
/// Centralised here because three of the COALESCE shape tests share this
/// pattern; factoring it out keeps the individual asserts readable.
fn assert_branch_is_not_null_arm(branch: &(Expr, Expr), col: &str) {
    // WHEN side
    match &branch.0 {
        Expr::Unary {
            op: UnaryOp::IsNotNull,
            operand,
        } => match operand.as_ref() {
            Expr::Column(n) => assert_eq!(
                n, col,
                "expected `{col} IS NOT NULL` branch condition, got Column({n})"
            ),
            other => panic!(
                "expected operand = Column({col}) in IS NOT NULL, got {other:?}"
            ),
        },
        other => panic!(
            "expected branch condition = `{col} IS NOT NULL`, got {other:?}"
        ),
    }
    // THEN side
    match &branch.1 {
        Expr::Column(n) => assert_eq!(
            n, col,
            "branch THEN must be the same column as the IS NOT NULL test"
        ),
        other => panic!("expected branch THEN = Column({col}), got {other:?}"),
    }
}
