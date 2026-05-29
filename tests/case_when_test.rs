// SPDX-License-Identifier: Apache-2.0

//! Integration tests for `CASE WHEN ... THEN ... [ELSE ...] END`.
//!
//! v0.5 / M2 SQL scalar completeness: the SQL frontend lowers SQL `CASE`
//! expressions — both the plain and the simple form — into the
//! `Expr::Case { branches, else_branch }` IR variant. This test suite pins
//! the parse-time contract: SQL -> `LogicalPlan` lowering succeeds,
//! type-checks correctly, and the lowered shape matches what downstream
//! stages expect.
//!
//! v0.7 GPU lowering: CASE over numeric / Bool / Date32 / Timestamp
//! branches is now lowered to GPU via the `Op::Select` IR op (see
//! `src/plan/physical_plan.rs` and `src/jit/ptx_gen.rs`). Date32 (i32
//! storage) folds to `selp.b32` and Timestamp (i64 storage) to `selp.b64`,
//! exactly like Int32 / Int64. CASE over Utf8 / Decimal128 is still rejected
//! at the codegen boundary with a tighter dtype-specific message (Decimal128
//! is i128 — there is no `selp.b128`); see the
//! `case_over_strings_rejected_with_tighter_message` test for the contract.

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Expr, Field, Literal, LogicalPlan, MemTableProvider,
    Schema,
};

// ---- Fixture ----------------------------------------------------------------

fn t_schema() -> Schema {
    Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "x".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "y".into(),
            dtype: DataType::Float64,
            nullable: true,
        },
    ])
}

fn t_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("t", t_schema())
}

/// Walk a logical plan down to the first `Project` node and return its
/// expression list. Every CASE-bearing query in this file lowers to
/// `Project { exprs: [...], input: Scan { .. } }`, so this helper is
/// the central inspection hook for the assertions below.
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

// ---- Parse-only tests -------------------------------------------------------

/// Plain CASE without ELSE: lowering must succeed and produce a
/// `Expr::Case` with exactly one branch and no `else_branch`. The
/// implicit-NULL-on-no-match contract lives at the type-checker / future
/// executor; this test pins the IR shape.
#[test]
fn plain_case_without_else_parses() {
    let sql = "SELECT CASE WHEN x > 0 THEN 1 END AS y FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let exprs = first_project_exprs(&plan);
    assert_eq!(exprs.len(), 1, "expected one SELECT-list expression");
    // The SELECT alias wraps the CASE; peel through it.
    let case = match &exprs[0] {
        Expr::Alias(inner, name) => {
            assert_eq!(name, "y", "alias should preserve SELECT name");
            inner.as_ref()
        }
        other => panic!("expected aliased CASE, got {other:?}"),
    };
    match case {
        Expr::Case {
            branches,
            else_branch,
        } => {
            assert_eq!(branches.len(), 1, "exactly one WHEN/THEN branch");
            assert!(else_branch.is_none(), "no ELSE branch");
            // The THEN must be the integer literal 1 — pin the structural
            // shape so a future refactor that changes literal lowering
            // surfaces the failure here rather than silently misbehaving.
            match &branches[0].1 {
                Expr::Literal(Literal::Int64(1)) => {}
                other => panic!("expected THEN = Int64(1), got {other:?}"),
            }
        }
        other => panic!("expected Expr::Case, got {other:?}"),
    }
}

/// Plain CASE with ELSE: lowering must produce a `Expr::Case` with the
/// ELSE branch populated. Unification of THEN (Int64) and ELSE (Int64)
/// must succeed without a cast pair.
#[test]
fn plain_case_with_else_parses() {
    let sql = "SELECT CASE WHEN x > 0 THEN 1 ELSE 0 END AS y FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let exprs = first_project_exprs(&plan);
    let case = match &exprs[0] {
        Expr::Alias(inner, _) => inner.as_ref(),
        other => panic!("expected aliased CASE, got {other:?}"),
    };
    match case {
        Expr::Case {
            branches,
            else_branch,
        } => {
            assert_eq!(branches.len(), 1);
            let else_expr = else_branch
                .as_deref()
                .expect("ELSE branch must be present");
            assert!(
                matches!(else_expr, Expr::Literal(Literal::Int64(0))),
                "expected ELSE = Int64(0), got {else_expr:?}",
            );
        }
        other => panic!("expected Expr::Case, got {other:?}"),
    }
}

/// Multi-branch plain CASE: `WHEN ... THEN ... WHEN ... THEN ... ELSE ... END`.
/// Branch order is preserved and the lowered IR carries every branch
/// in source order so the executor (when it lands) can evaluate them
/// top-down per SQL semantics.
#[test]
fn plain_case_multi_branch_with_else_parses() {
    let sql = "SELECT CASE \
               WHEN x < 0 THEN -1 \
               WHEN x > 0 THEN 1 \
               ELSE 0 \
               END AS sign FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let exprs = first_project_exprs(&plan);
    let case = match &exprs[0] {
        Expr::Alias(inner, _) => inner.as_ref(),
        other => panic!("expected aliased CASE, got {other:?}"),
    };
    match case {
        Expr::Case {
            branches,
            else_branch,
        } => {
            assert_eq!(branches.len(), 2, "two WHEN/THEN branches");
            assert!(else_branch.is_some(), "ELSE present");
            // The first WHEN/THEN compares against -1, the second against
            // 1. Pin the THEN literals so branch ordering can't silently
            // reverse.
            match &branches[0].1 {
                // sqlparser lowers `-1` as a UnaryOp::Minus over a positive
                // literal; our `negate_expr` helper folds that into a single
                // signed Int64 literal at lower time.
                Expr::Literal(Literal::Int64(-1)) => {}
                other => panic!("expected branch 0 THEN = Int64(-1), got {other:?}"),
            }
            match &branches[1].1 {
                Expr::Literal(Literal::Int64(1)) => {}
                other => panic!("expected branch 1 THEN = Int64(1), got {other:?}"),
            }
        }
        other => panic!("expected Expr::Case, got {other:?}"),
    }
}

/// Simple CASE (with an operand) desugars per branch into
/// `operand = condition`. The lowered IR carries the plain `Expr::Case`
/// shape — i.e. no Simple-CASE operand survives lowering, every branch's
/// condition is a Bool expression.
#[test]
fn simple_case_desugars_to_equality_per_branch() {
    let sql = "SELECT CASE x WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END \
               AS label FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let exprs = first_project_exprs(&plan);
    let case = match &exprs[0] {
        Expr::Alias(inner, _) => inner.as_ref(),
        other => panic!("expected aliased CASE, got {other:?}"),
    };
    match case {
        Expr::Case {
            branches,
            else_branch,
        } => {
            assert_eq!(branches.len(), 2, "two WHEN/THEN branches");
            assert!(else_branch.is_some(), "ELSE 'other' present");
            // Each branch's condition must be `x = <value>`. The lowered
            // operand was `Expr::Column("x")` and the WHEN values were
            // `1` and `2`.
            use craton_bolt::plan::BinaryOp;
            for (i, (when, _)) in branches.iter().enumerate() {
                match when {
                    Expr::Binary {
                        op: BinaryOp::Eq,
                        left,
                        right,
                    } => {
                        match left.as_ref() {
                            Expr::Column(n) => assert_eq!(
                                n, "x",
                                "Simple-CASE operand should be Column(x), got branch {i}: {left:?}"
                            ),
                            other => panic!(
                                "expected Column(x) on LHS of branch {i}, got {other:?}"
                            ),
                        }
                        match right.as_ref() {
                            Expr::Literal(Literal::Int64(v)) => {
                                assert_eq!(*v, (i as i64) + 1, "WHEN values are 1, 2 in order")
                            }
                            other => panic!(
                                "expected Int64 literal on RHS of branch {i}, got {other:?}"
                            ),
                        }
                    }
                    other => {
                        panic!("expected `x = lit` equality on branch {i}, got {other:?}")
                    }
                }
            }
        }
        other => panic!("expected Expr::Case, got {other:?}"),
    }
}

// ---- Type-checker tests -----------------------------------------------------

/// Calling `schema()` on the lowered plan must succeed and infer the
/// CASE's result dtype from its arms. `THEN 1 ELSE 0` unifies to Int64.
#[test]
fn case_typechecks_int64_uniform_arms() {
    let sql = "SELECT CASE WHEN x > 0 THEN 1 ELSE 0 END AS s FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let schema = plan.schema().expect("schema/type-check must succeed");
    let f = schema
        .fields
        .iter()
        .find(|f| f.name == "s")
        .expect("output column `s` must exist");
    assert_eq!(f.dtype, DataType::Int64, "uniform Int64 arms infer Int64");
}

/// Mixed numeric arms widen via the same rules as `Expr::Binary`: an
/// `Int64` THEN and a `Float64` ELSE unify to `Float64`.
#[test]
fn case_typechecks_widens_int_and_float_arms_to_float64() {
    let sql = "SELECT CASE WHEN x > 0 THEN 1 ELSE y END AS s FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let schema = plan.schema().expect("schema/type-check must succeed");
    let f = schema
        .fields
        .iter()
        .find(|f| f.name == "s")
        .expect("output column `s` must exist");
    assert_eq!(
        f.dtype,
        DataType::Float64,
        "Int64 + Float64 arms widen to Float64",
    );
}

/// CASE without an ELSE branch takes its dtype from the THEN arms alone.
#[test]
fn case_typechecks_without_else_takes_then_dtype() {
    let sql = "SELECT CASE WHEN x > 0 THEN y END AS s FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let schema = plan.schema().expect("schema/type-check must succeed");
    let f = schema
        .fields
        .iter()
        .find(|f| f.name == "s")
        .expect("output column `s` must exist");
    assert_eq!(f.dtype, DataType::Float64);
}

/// A non-Bool WHEN condition must be rejected by the type-checker with a
/// clear error message naming the offending branch.
#[test]
fn case_rejects_non_bool_when_condition() {
    // `x` is Int32 (not Bool); a CASE that puts it in WHEN position
    // must error at type-check time, not at execution time.
    let sql = "SELECT CASE WHEN x THEN 1 ELSE 0 END AS s FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let err = plan.schema().expect_err("non-Bool WHEN must surface a Type error");
    let msg = format!("{err}");
    assert!(
        msg.contains("CASE WHEN condition"),
        "error should mention the offending CASE branch, got: {msg}"
    );
    assert!(
        msg.contains("Bool"),
        "error should mention the required Bool dtype, got: {msg}"
    );
}

/// Incompatible non-numeric THEN dtypes (Utf8 vs Bool) must be rejected
/// with a clear "incompatible dtype" message at type-check time.
#[test]
fn case_rejects_incompatible_non_numeric_arms() {
    // Two Utf8 arms unify fine; mixing Utf8 with Bool does not.
    let sql = "SELECT CASE WHEN x > 0 THEN 'yes' ELSE true END AS s FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    let err = plan
        .schema()
        .expect_err("Utf8 + Bool arms must error at type-check");
    let msg = format!("{err}");
    assert!(
        msg.contains("CASE") && msg.contains("incompatible"),
        "error should mention CASE and incompatibility, got: {msg}"
    );
}

// ---- Physical-plan lowering ------------------------------------------------

/// v0.7: CASE WHEN over numeric branches is now lowered to GPU via
/// `Op::Select`. Scan-chain SELECT lists with a CASE expression must
/// lower successfully — the previous "CASE not yet lowered to GPU"
/// rejection at the physical-plan boundary has been retired. The host
/// evaluator still surfaces a clear error for CASE in post-aggregate /
/// HAVING positions; that path is covered separately.
#[test]
fn case_in_scan_chain_select_lowers_successfully() {
    let sql = "SELECT CASE WHEN x > 0 THEN 1 ELSE 0 END AS s FROM t";
    let plan = parse_sql(sql, &t_provider()).expect("parse must succeed");
    lower_physical(&plan).expect("physical lowering must accept CASE in scan-chain SELECT");
}

/// CASE over Utf8 branches is still rejected at the codegen boundary —
/// the v0.7 envelope is numeric / Bool only. The error message must be
/// tighter than the legacy "CASE not yet lowered to GPU" so the user
/// knows the path forward (e.g. host-side projection of strings).
#[test]
fn case_over_strings_rejected_with_tighter_message() {
    // Build a fixture with a Utf8 column so a CASE over it survives the
    // type-checker and reaches the GPU codegen, which then rejects it.
    let schema = Schema::new(vec![
        Field {
            name: "x".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "name".into(),
            dtype: DataType::Utf8,
            nullable: false,
        },
    ]);
    let provider = MemTableProvider::new().with_table("t", schema);
    let sql = "SELECT CASE WHEN x > 0 THEN name ELSE name END AS s FROM t";
    let plan = parse_sql(sql, &provider).expect("parse must succeed");
    let err = lower_physical(&plan).expect_err("CASE over Utf8 must still be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("CASE over string")
            || msg.contains("string (Utf8)")
            || msg.contains("Utf8"),
        "rejection should mention strings/Utf8 explicitly, got: {msg}"
    );
}
