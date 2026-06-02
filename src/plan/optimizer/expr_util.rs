// SPDX-License-Identifier: Apache-2.0

//! Shared expression helpers for the logical optimizer passes.
//!
//! These are pure, allocation-light walkers over [`Expr`] used by more than
//! one pass (column-reference collection, conjunction split/rebuild, and a
//! generic bottom-up rewrite combinator). Keeping them here avoids each pass
//! re-implementing the same recursion and risking divergence.

use crate::plan::logical_plan::{AggregateExpr, BinaryOp, Expr, Literal};

/// Collect every column name referenced anywhere in `expr` into `out`.
///
/// `Alias(inner, _)` is descended into (the alias name is an *output* name,
/// not an input reference). Duplicates are appended; callers that need a set
/// should dedup. Bounded recursion is not enforced here — the planner has
/// already type-checked the expression against
/// [`crate::plan::sql_frontend::MAX_RECURSION_DEPTH`], so any expression
/// reaching the optimizer is within depth.
pub fn collect_columns(expr: &Expr, out: &mut Vec<String>) {
    match expr {
        Expr::Extract { expr, .. } | Expr::DateTrunc { expr, .. } => collect_columns(expr, out),
        Expr::InSubquery { expr, .. } => collect_columns(expr, out),
        Expr::ScalarSubquery(_) => {}
        Expr::Column(name) => out.push(name.clone()),
        Expr::Literal(_) => {}
        Expr::Binary { left, right, .. } => {
            collect_columns(left, out);
            collect_columns(right, out);
        }
        Expr::Unary { operand, .. } => collect_columns(operand, out),
        Expr::Case {
            branches,
            else_branch,
        } => {
            for (w, t) in branches {
                collect_columns(w, out);
                collect_columns(t, out);
            }
            if let Some(e) = else_branch {
                collect_columns(e, out);
            }
        }
        Expr::Like { expr, .. } => collect_columns(expr, out),
        Expr::Cast { expr, .. } | Expr::CastFormat { expr, .. } => collect_columns(expr, out),
        Expr::ScalarFn { args, .. } => {
            for a in args {
                collect_columns(a, out);
            }
        }
        Expr::Alias(inner, _) => collect_columns(inner, out),
    }
}

/// Collect every column referenced by an [`AggregateExpr`] into `out`.
pub fn collect_agg_columns(agg: &AggregateExpr, out: &mut Vec<String>) {
    match agg {
        AggregateExpr::Count(e)
        | AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e) => collect_columns(e, out),
        AggregateExpr::VarPop(e)
        | AggregateExpr::VarSamp(e)
        | AggregateExpr::StddevPop(e)
        | AggregateExpr::StddevSamp(e) => collect_columns(e.as_ref(), out),
    }
}

/// Split a conjunction (`a AND b AND c`) into its top-level conjuncts,
/// appending each leaf into `out`. Non-`AND` expressions append as a single
/// conjunct. The split is *structural only* — it does not recurse below the
/// first non-`AND` node, matching SQL `WHERE` semantics where the whole
/// predicate is one boolean expression and each top-level `AND` operand can
/// be evaluated independently.
pub fn split_conjuncts(expr: Expr, out: &mut Vec<Expr>) {
    match expr {
        Expr::Binary {
            op: BinaryOp::And,
            left,
            right,
        } => {
            split_conjuncts(*left, out);
            split_conjuncts(*right, out);
        }
        other => out.push(other),
    }
}

/// Rebuild a single boolean predicate from a list of conjuncts by folding
/// them with `AND`. Returns `None` for an empty list (no predicate).
pub fn combine_conjuncts(conjuncts: Vec<Expr>) -> Option<Expr> {
    let mut iter = conjuncts.into_iter();
    let first = iter.next()?;
    Some(iter.fold(first, |acc, c| Expr::Binary {
        op: BinaryOp::And,
        left: Box::new(acc),
        right: Box::new(c),
    }))
}

/// True when `expr` is a literal boolean equal to `value`. Used by the
/// constant-folding and pushdown passes to detect trivially-true / -false
/// predicates.
pub fn is_bool_literal(expr: &Expr, value: bool) -> bool {
    matches!(expr, Expr::Literal(Literal::Bool(b)) if *b == value)
}
