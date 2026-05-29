// SPDX-License-Identifier: Apache-2.0

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
    match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(map_plan_exprs(*input, map_expr, map_agg)),
            predicate: map_expr(predicate),
        },
        LogicalPlan::Project { input, exprs } => LogicalPlan::Project {
            input: Box::new(map_plan_exprs(*input, map_expr, map_agg)),
            exprs: exprs.into_iter().map(map_expr).collect(),
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalPlan::Aggregate {
            input: Box::new(map_plan_exprs(*input, map_expr, map_agg)),
            group_by: group_by.into_iter().map(map_expr).collect(),
            aggregates: aggregates.into_iter().map(map_agg).collect(),
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(map_plan_exprs(*input, map_expr, map_agg)),
        },
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => LogicalPlan::Limit {
            input: Box::new(map_plan_exprs(*input, map_expr, map_agg)),
            limit,
            offset,
        },
        LogicalPlan::Sort { input, sort_exprs } => LogicalPlan::Sort {
            input: Box::new(map_plan_exprs(*input, map_expr, map_agg)),
            sort_exprs: sort_exprs
                .into_iter()
                .map(|mut se| {
                    se.expr = map_expr(se.expr);
                    se
                })
                .collect(),
        },
        LogicalPlan::Union { inputs } => LogicalPlan::Union {
            inputs: inputs
                .into_iter()
                .map(|i| map_plan_exprs(i, map_expr, map_agg))
                .collect(),
        },
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => LogicalPlan::Join {
            left: Box::new(map_plan_exprs(*left, map_expr, map_agg)),
            right: Box::new(map_plan_exprs(*right, map_expr, map_agg)),
            join_type,
            on: on
                .into_iter()
                .map(|(l, r)| (map_expr(l), map_expr(r)))
                .collect(),
            filter: filter.map(map_expr),
        },
    }
}
