// SPDX-License-Identifier: Apache-2.0

//! Constant folding and algebraic simplification of [`Expr`] trees.
//!
//! This pass walks every expression in the plan bottom-up and:
//!
//! * folds binary arithmetic / comparison / logical operators over two
//!   literal operands into a single literal (`2 + 3` -> `5`, `1 < 2` ->
//!   `true`);
//! * applies boolean identities that hold for *any* operand without changing
//!   semantics: `x AND true` -> `x`, `x AND false` -> `false`,
//!   `x OR false` -> `x`, `x OR true` -> `true`;
//! * collapses double negation `NOT (NOT x)` -> `x`;
//! * folds `NOT true` / `NOT false`.
//!
//! Folding is intentionally conservative: it only fires on like-typed integer
//! / float / bool literal pairs. Mixed-width numeric folding is skipped (the
//! type-checker's numeric promotion rules apply at plan-validation time, and
//! re-implementing them here risks subtle divergence). String concat and any
//! NULL operand are left untouched so the executor keeps SQL three-valued
//! logic semantics.

use crate::error::BoltResult;
use crate::plan::logical_plan::{
    AggregateExpr, BinaryOp, Expr, Literal, LogicalPlan, UnaryOp,
};
use crate::plan::rewrite::PlanRewrite;

use super::plan_util::map_plan_exprs;

/// Constant-folding / boolean-simplification pass. See module docs.
#[derive(Debug, Default)]
pub struct ConstantFold;

impl PlanRewrite for ConstantFold {
    fn name(&self) -> &str {
        "constant-fold"
    }

    fn rewrite(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
        Ok(fold_plan(plan))
    }
}

/// Recursively fold every expression in `plan`, preserving structure.
fn fold_plan(plan: LogicalPlan) -> LogicalPlan {
    map_plan_exprs(plan, &fold_expr, &fold_agg)
}

/// Fold an aggregate's inner expression(s).
fn fold_agg(agg: AggregateExpr) -> AggregateExpr {
    match agg {
        AggregateExpr::Count(e) => AggregateExpr::Count(fold_expr(e)),
        AggregateExpr::Sum(e) => AggregateExpr::Sum(fold_expr(e)),
        AggregateExpr::Min(e) => AggregateExpr::Min(fold_expr(e)),
        AggregateExpr::Max(e) => AggregateExpr::Max(fold_expr(e)),
        AggregateExpr::Avg(e) => AggregateExpr::Avg(fold_expr(e)),
        AggregateExpr::VarPop(e) => AggregateExpr::VarPop(Box::new(fold_expr(*e))),
        AggregateExpr::VarSamp(e) => AggregateExpr::VarSamp(Box::new(fold_expr(*e))),
        AggregateExpr::StddevPop(e) => AggregateExpr::StddevPop(Box::new(fold_expr(*e))),
        AggregateExpr::StddevSamp(e) => AggregateExpr::StddevSamp(Box::new(fold_expr(*e))),
    }
}

/// Bottom-up fold of a single expression tree.
pub fn fold_expr(expr: Expr) -> Expr {
    match expr {
        Expr::Extract { .. } | Expr::DateTrunc { .. } | Expr::ScalarSubquery(_) | Expr::InSubquery { .. } => expr,
        // Leaves are already folded.
        Expr::Column(_) | Expr::Literal(_) => expr,
        Expr::Binary { op, left, right } => {
            let l = fold_expr(*left);
            let r = fold_expr(*right);
            fold_binary(op, l, r)
        }
        Expr::Unary { op, operand } => {
            let inner = fold_expr(*operand);
            fold_unary(op, inner)
        }
        Expr::Case {
            branches,
            else_branch,
        } => Expr::Case {
            branches: branches
                .into_iter()
                .map(|(w, t)| (fold_expr(w), fold_expr(t)))
                .collect(),
            else_branch: else_branch.map(|e| Box::new(fold_expr(*e))),
        },
        Expr::Like {
            expr,
            pattern,
            escape,
            negated,
        } => Expr::Like {
            expr: Box::new(fold_expr(*expr)),
            pattern,
            escape,
            negated,
        },
        Expr::Cast { expr, target } => Expr::Cast {
            expr: Box::new(fold_expr(*expr)),
            target,
        },
        Expr::ScalarFn { kind, args } => Expr::ScalarFn {
            kind,
            args: args.into_iter().map(fold_expr).collect(),
        },
        Expr::Alias(inner, name) => Expr::Alias(Box::new(fold_expr(*inner)), name),
    }
}

/// Fold a binary node whose children are already folded.
fn fold_binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    // Boolean identities that hold for any (non-NULL-sensitive) operand.
    // `x AND true` => x, `x AND false` => false, etc. We only apply these
    // when exactly one side is a bool literal; folding two bool literals
    // falls through to the literal-pair arithmetic below.
    if op == BinaryOp::And {
        if let Some(simplified) = simplify_and(&left, &right) {
            return simplified;
        }
    }
    if op == BinaryOp::Or {
        if let Some(simplified) = simplify_or(&left, &right) {
            return simplified;
        }
    }

    if let (Expr::Literal(l), Expr::Literal(r)) = (&left, &right) {
        if let Some(folded) = fold_literal_binary(op, l, r) {
            return Expr::Literal(folded);
        }
    }

    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// `x AND true` -> `x`; `x AND false` -> `false`; symmetric. Returns `None`
/// when neither side is a bool literal (let the literal-pair path handle
/// `true AND false`). Cloning the surviving operand is unavoidable because we
/// borrow both sides to inspect them.
fn simplify_and(left: &Expr, right: &Expr) -> Option<Expr> {
    if super::expr_util::is_bool_literal(left, true) {
        return Some(right.clone());
    }
    if super::expr_util::is_bool_literal(right, true) {
        return Some(left.clone());
    }
    if super::expr_util::is_bool_literal(left, false)
        || super::expr_util::is_bool_literal(right, false)
    {
        return Some(Expr::Literal(Literal::Bool(false)));
    }
    None
}

/// `x OR false` -> `x`; `x OR true` -> `true`; symmetric.
fn simplify_or(left: &Expr, right: &Expr) -> Option<Expr> {
    if super::expr_util::is_bool_literal(left, false) {
        return Some(right.clone());
    }
    if super::expr_util::is_bool_literal(right, false) {
        return Some(left.clone());
    }
    if super::expr_util::is_bool_literal(left, true)
        || super::expr_util::is_bool_literal(right, true)
    {
        return Some(Expr::Literal(Literal::Bool(true)));
    }
    None
}

/// Fold a unary node whose child is already folded.
fn fold_unary(op: UnaryOp, operand: Expr) -> Expr {
    match op {
        UnaryOp::Not => {
            // NOT (NOT x) => x.
            if let Expr::Unary {
                op: UnaryOp::Not,
                operand: inner,
            } = operand
            {
                return *inner;
            }
            // NOT true => false; NOT false => true.
            if let Expr::Literal(Literal::Bool(b)) = operand {
                return Expr::Literal(Literal::Bool(!b));
            }
            Expr::Unary {
                op,
                operand: Box::new(operand),
            }
        }
        // IS NULL / IS NOT NULL over a non-NULL literal fold to a constant.
        UnaryOp::IsNull => match &operand {
            Expr::Literal(Literal::Null) => Expr::Literal(Literal::Bool(true)),
            Expr::Literal(_) => Expr::Literal(Literal::Bool(false)),
            _ => Expr::Unary {
                op,
                operand: Box::new(operand),
            },
        },
        UnaryOp::IsNotNull => match &operand {
            Expr::Literal(Literal::Null) => Expr::Literal(Literal::Bool(false)),
            Expr::Literal(_) => Expr::Literal(Literal::Bool(true)),
            _ => Expr::Unary {
                op,
                operand: Box::new(operand),
            },
        },
    }
}

/// Fold a binary op over two concrete literals. Returns `None` when the
/// operand pair is not a like-typed numeric / bool pair we know how to fold
/// (mixed-width numerics, strings, decimals, dates, timestamps, and any NULL
/// are deliberately left for the executor).
fn fold_literal_binary(op: BinaryOp, l: &Literal, r: &Literal) -> Option<Literal> {
    use BinaryOp::*;
    match (l, r) {
        (Literal::Int32(a), Literal::Int32(b)) => fold_int(op, *a as i64, *b as i64)
            .map(|v| match v {
                FoldVal::Int(n) => Literal::Int32(n as i32),
                FoldVal::Bool(x) => Literal::Bool(x),
            }),
        (Literal::Int64(a), Literal::Int64(b)) => fold_int(op, *a, *b).map(|v| match v {
            FoldVal::Int(n) => Literal::Int64(n),
            FoldVal::Bool(x) => Literal::Bool(x),
        }),
        (Literal::Float64(a), Literal::Float64(b)) => fold_float(op, *a, *b),
        (Literal::Float32(a), Literal::Float32(b)) => match fold_float(op, *a as f64, *b as f64) {
            Some(Literal::Float64(v)) => Some(Literal::Float32(v as f32)),
            other => other,
        },
        (Literal::Bool(a), Literal::Bool(b)) => match op {
            And => Some(Literal::Bool(*a && *b)),
            Or => Some(Literal::Bool(*a || *b)),
            Eq => Some(Literal::Bool(a == b)),
            NotEq => Some(Literal::Bool(a != b)),
            _ => None,
        },
        _ => None,
    }
}

/// Intermediate fold result: either a numeric value (same integer family as
/// the inputs) or a boolean (from a comparison).
enum FoldVal {
    Int(i64),
    Bool(bool),
}

/// Fold an integer binary op. Division by zero returns `None` (left for the
/// runtime so the engine's existing div-by-zero behaviour is unchanged).
/// Overflow on `+ - *` returns `None` (folding must not change the observable
/// result vs. evaluating at runtime).
fn fold_int(op: BinaryOp, a: i64, b: i64) -> Option<FoldVal> {
    use BinaryOp::*;
    Some(match op {
        Add => FoldVal::Int(a.checked_add(b)?),
        Sub => FoldVal::Int(a.checked_sub(b)?),
        Mul => FoldVal::Int(a.checked_mul(b)?),
        Div => {
            if b == 0 {
                return None;
            }
            FoldVal::Int(a.checked_div(b)?)
        }
        Eq => FoldVal::Bool(a == b),
        NotEq => FoldVal::Bool(a != b),
        Lt => FoldVal::Bool(a < b),
        LtEq => FoldVal::Bool(a <= b),
        Gt => FoldVal::Bool(a > b),
        GtEq => FoldVal::Bool(a >= b),
        And | Or | Concat => return None,
    })
}

/// Fold a float binary op. Comparisons fold to bool; arithmetic folds to a
/// `Float64` literal (the caller narrows back to `Float32` when both inputs
/// were `Float32`). NaN / inf are produced by ordinary IEEE-754 arithmetic,
/// matching what the GPU kernel would compute.
fn fold_float(op: BinaryOp, a: f64, b: f64) -> Option<Literal> {
    use BinaryOp::*;
    Some(match op {
        Add => Literal::Float64(a + b),
        Sub => Literal::Float64(a - b),
        Mul => Literal::Float64(a * b),
        Div => Literal::Float64(a / b),
        Eq => Literal::Bool(a == b),
        NotEq => Literal::Bool(a != b),
        Lt => Literal::Bool(a < b),
        LtEq => Literal::Bool(a <= b),
        Gt => Literal::Bool(a > b),
        GtEq => Literal::Bool(a >= b),
        And | Or | Concat => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{DataType, Field, Schema};
    use crate::plan::{col, lit};

    fn scan(fields: Vec<Field>) -> LogicalPlan {
        LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(fields),
        }
    }

    fn b(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }

    #[test]
    fn folds_integer_arithmetic() {
        let e = b(BinaryOp::Add, lit(2_i64), lit(3_i64));
        assert!(matches!(fold_expr(e), Expr::Literal(Literal::Int64(5))));
    }

    #[test]
    fn folds_integer_comparison_to_bool() {
        let e = b(BinaryOp::Lt, lit(1_i64), lit(2_i64));
        assert!(matches!(fold_expr(e), Expr::Literal(Literal::Bool(true))));
    }

    #[test]
    fn does_not_fold_div_by_zero() {
        let e = b(BinaryOp::Div, lit(1_i64), lit(0_i64));
        // Stays a Binary so runtime semantics are preserved.
        assert!(matches!(fold_expr(e), Expr::Binary { .. }));
    }

    #[test]
    fn does_not_fold_overflow() {
        let e = b(BinaryOp::Add, lit(i64::MAX), lit(1_i64));
        assert!(matches!(fold_expr(e), Expr::Binary { .. }));
    }

    #[test]
    fn simplifies_and_true() {
        // col(a) AND true => col(a)
        let e = b(BinaryOp::And, col("a"), lit(true));
        assert!(matches!(fold_expr(e), Expr::Column(n) if n == "a"));
    }

    #[test]
    fn simplifies_and_false() {
        let e = b(BinaryOp::And, col("a"), lit(false));
        assert!(matches!(fold_expr(e), Expr::Literal(Literal::Bool(false))));
    }

    #[test]
    fn simplifies_or_false() {
        let e = b(BinaryOp::Or, col("a"), lit(false));
        assert!(matches!(fold_expr(e), Expr::Column(n) if n == "a"));
    }

    #[test]
    fn simplifies_or_true() {
        let e = b(BinaryOp::Or, col("a"), lit(true));
        assert!(matches!(fold_expr(e), Expr::Literal(Literal::Bool(true))));
    }

    #[test]
    fn collapses_double_negation() {
        let e = col("a").eq(lit(1_i64)).not().not();
        // NOT NOT (a = 1) => (a = 1)
        assert!(matches!(fold_expr(e), Expr::Binary { op: BinaryOp::Eq, .. }));
    }

    #[test]
    fn folds_not_literal() {
        assert!(matches!(
            fold_expr(lit(true).not()),
            Expr::Literal(Literal::Bool(false))
        ));
    }

    #[test]
    fn nested_fold_inside_filter_preserves_schema() {
        let plan = LogicalPlan::Filter {
            input: Box::new(scan(vec![Field::new("a", DataType::Int64, false)])),
            // (1 + 1 = 2) AND (a > 0)  =>  true AND (a > 0)  =>  (a > 0)
            predicate: b(
                BinaryOp::And,
                b(BinaryOp::Eq, b(BinaryOp::Add, lit(1_i64), lit(1_i64)), lit(2_i64)),
                col("a").gt(lit(0_i64)),
            ),
        };
        let before = plan.schema().expect("typecheck");
        let out = ConstantFold.rewrite(plan).expect("fold");
        let after = out.schema().expect("typecheck after");
        assert_eq!(before.fields.len(), after.fields.len());
        match out {
            LogicalPlan::Filter { predicate, .. } => {
                // Collapsed to just `a > 0`.
                assert!(matches!(predicate, Expr::Binary { op: BinaryOp::Gt, .. }));
            }
            other => panic!("expected Filter, got {other:?}"),
        }
    }

    #[test]
    fn folds_float_arithmetic_keeps_width() {
        let e = b(BinaryOp::Mul, lit(2.0_f32), lit(3.0_f32));
        match fold_expr(e) {
            Expr::Literal(Literal::Float32(v)) => assert_eq!(v, 6.0),
            other => panic!("expected Float32 literal, got {other:?}"),
        }
    }
}
