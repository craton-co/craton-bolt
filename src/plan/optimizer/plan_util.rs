// SPDX-License-Identifier: AGPL-3.0-only

//! Structural plan-walking helpers shared by the optimizer passes.

use crate::plan::logical_plan::{AggregateExpr, Expr, LogicalPlan};

/// Recursively apply `map_expr` to every scalar [`Expr`] and `map_agg` to
/// every [`AggregateExpr`] in `plan`, rebuilding an otherwise structurally
/// identical plan. The closures are applied **after** child plans are
/// recursed into, so a pass that folds expressions sees already-rewritten
/// inputs but the per-node expressions are transformed in source order.
///
/// This is the single recursion every "expression-only" pass (e.g. constant
/// folding) shares, so the set of plan variants only needs to be enumerated
/// once. Passes that restructure the plan tree itself (pushdown, reordering)
/// recurse directly rather than going through this helper.
pub fn map_plan_exprs<E, A>(plan: LogicalPlan, map_expr: &E, map_agg: &A) -> LogicalPlan
where
    E: Fn(Expr) -> Expr,
    A: Fn(AggregateExpr) -> AggregateExpr,
{
    map_plan_exprs_and_predicates(plan, map_expr, map_expr, map_agg)
}

/// [`map_plan_exprs`], except that a `Filter` predicate is mapped by
/// `map_predicate` instead of `map_expr` (a filter keeps only the rows where
/// it is TRUE, so a pass may treat UNKNOWN there as FALSE).
pub(crate) fn map_plan_exprs_and_predicates<E, P, A>(
    plan: LogicalPlan,
    map_expr: &E,
    map_predicate: &P,
    map_agg: &A,
) -> LogicalPlan
where
    E: Fn(Expr) -> Expr,
    P: Fn(Expr) -> Expr,
    A: Fn(AggregateExpr) -> AggregateExpr,
{
    let rec = |p: LogicalPlan| map_plan_exprs_and_predicates(p, map_expr, map_predicate, map_agg);
    match plan {
        LogicalPlan::Window {
            input,
            window_exprs,
            partition_by,
            order_by,
        } => LogicalPlan::Window {
            input: Box::new(rec(*input)),
            window_exprs,
            partition_by: partition_by.into_iter().map(map_expr).collect(),
            order_by,
        },
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(rec(*input)),
            predicate: map_predicate(predicate),
        },
        LogicalPlan::Project { input, exprs } => LogicalPlan::Project {
            input: Box::new(rec(*input)),
            exprs: exprs.into_iter().map(map_expr).collect(),
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalPlan::Aggregate {
            input: Box::new(rec(*input)),
            group_by: group_by.into_iter().map(map_expr).collect(),
            aggregates: aggregates.into_iter().map(map_agg).collect(),
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(rec(*input)),
        },
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => LogicalPlan::Limit {
            input: Box::new(rec(*input)),
            limit,
            offset,
        },
        LogicalPlan::Sort { input, sort_exprs } => LogicalPlan::Sort {
            input: Box::new(rec(*input)),
            sort_exprs: sort_exprs
                .into_iter()
                .map(|mut se| {
                    se.expr = map_expr(se.expr);
                    se
                })
                .collect(),
        },
        LogicalPlan::Union { inputs } => LogicalPlan::Union {
            inputs: inputs.into_iter().map(rec).collect(),
        },
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => LogicalPlan::Join {
            left: Box::new(rec(*left)),
            right: Box::new(rec(*right)),
            join_type,
            on: on
                .into_iter()
                .map(|(l, r)| (map_expr(l), map_expr(r)))
                .collect(),
            filter: filter.map(map_expr),
        },
        // EXCEPT / INTERSECT carry no scalar / aggregate expressions of their
        // own — just recurse into both inputs.
        LogicalPlan::SetOp {
            left,
            right,
            op,
            all,
        } => LogicalPlan::SetOp {
            left: Box::new(rec(*left)),
            right: Box::new(rec(*right)),
            op,
            all,
        },
    }
}
