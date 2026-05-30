// SPDX-License-Identifier: Apache-2.0

//! Filter-into-join: fold a post-join `Filter` that references *both* join
//! inputs into the join's residual `filter` slot.
//!
//! After [`crate::plan::optimizer::predicate_pushdown`] has sunk every
//! single-side conjunct into the owning input, what remains above a join are
//! conjuncts that reference columns from both sides (e.g. `l.x > r.y`). The
//! join node already has a `filter: Option<Expr>` slot — the *residual
//! non-equi predicate* evaluated against the join's combined left++right
//! schema (see [`LogicalPlan::Join`] docs) — so we can fold those conjuncts in
//! there, letting the nested-loop join evaluate them inline instead of
//! materialising the full product and then filtering.
//!
//! ## Correctness
//!
//! This is only valid for `INNER` and `CROSS` joins. For those, a post-join
//! `WHERE p` is exactly equivalent to a join with residual predicate `p`
//! AND-ed onto the existing residual: both keep precisely the combined rows
//! satisfying every equi-pair and every residual conjunct.
//!
//! For OUTER joins it is **not** valid: a post-join `WHERE` over the
//! NULL-padded side acts on the *padded* output rows (and can re-introduce
//! INNER-like semantics), whereas a join residual is evaluated *before*
//! NULL-padding. We therefore leave `Filter(OuterJoin, _)` untouched.
//!
//! The post-join `Filter` predicate is already expressed against the join's
//! combined output schema (it was type-checked there), so no column renaming
//! is needed when moving it into the residual slot — both are evaluated
//! against the same combined schema.

use crate::error::BoltResult;
use crate::plan::logical_plan::{JoinType, LogicalPlan};
use crate::plan::rewrite::PlanRewrite;

use super::expr_util::{combine_conjuncts, split_conjuncts};

/// Filter-into-join pass. See module docs.
#[derive(Debug, Default)]
pub struct FilterIntoJoin;

impl PlanRewrite for FilterIntoJoin {
    fn name(&self) -> &str {
        "filter-into-join"
    }

    fn rewrite(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
        Ok(rewrite_plan(plan))
    }
}

/// Recursively rewrite `plan`, folding eligible post-join filters into joins.
fn rewrite_plan(plan: LogicalPlan) -> LogicalPlan {
    match plan {
        LogicalPlan::Window { input, window_exprs, partition_by, order_by } => LogicalPlan::Window {
            input: Box::new(rewrite_plan(*input)),
            window_exprs,
            partition_by,
            order_by,
        },
        LogicalPlan::Filter { input, predicate } => {
            let input = rewrite_plan(*input);
            match input {
                LogicalPlan::Join {
                    left,
                    right,
                    join_type,
                    on,
                    filter,
                } if eligible_join(join_type) => {
                    // Merge the post-join predicate's conjuncts with any
                    // existing residual filter into one residual.
                    let mut conjuncts = Vec::new();
                    if let Some(existing) = filter {
                        split_conjuncts(existing, &mut conjuncts);
                    }
                    split_conjuncts(predicate, &mut conjuncts);
                    LogicalPlan::Join {
                        left,
                        right,
                        join_type,
                        on,
                        filter: combine_conjuncts(conjuncts),
                    }
                }
                other => LogicalPlan::Filter {
                    input: Box::new(other),
                    predicate,
                },
            }
        }
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Project { input, exprs } => LogicalPlan::Project {
            input: Box::new(rewrite_plan(*input)),
            exprs,
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalPlan::Aggregate {
            input: Box::new(rewrite_plan(*input)),
            group_by,
            aggregates,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(rewrite_plan(*input)),
        },
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => LogicalPlan::Limit {
            input: Box::new(rewrite_plan(*input)),
            limit,
            offset,
        },
        LogicalPlan::Sort { input, sort_exprs } => LogicalPlan::Sort {
            input: Box::new(rewrite_plan(*input)),
            sort_exprs,
        },
        LogicalPlan::Union { inputs } => LogicalPlan::Union {
            inputs: inputs.into_iter().map(rewrite_plan).collect(),
        },
        LogicalPlan::SetOp { left, right, op, all } => LogicalPlan::SetOp {
            left: Box::new(rewrite_plan(*left)),
            right: Box::new(rewrite_plan(*right)),
            op,
            all,
        },
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => LogicalPlan::Join {
            left: Box::new(rewrite_plan(*left)),
            right: Box::new(rewrite_plan(*right)),
            join_type,
            on,
            filter,
        },
    }
}

/// A residual filter may only be folded into INNER / CROSS joins. See module
/// docs for why OUTER joins are excluded.
fn eligible_join(join_type: JoinType) -> bool {
    matches!(join_type, JoinType::Inner | JoinType::Cross)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{BinaryOp, DataType, Expr, Field, Schema};
    use crate::plan::{col, lit};

    fn scan(name: &str, field: &str) -> LogicalPlan {
        LogicalPlan::Scan {
            table: name.into(),
            projection: None,
            schema: Schema::new(vec![Field::new(field, DataType::Int64, false)]),
        }
    }

    fn inner_join() -> LogicalPlan {
        LogicalPlan::Join {
            left: Box::new(scan("l", "a")),
            right: Box::new(scan("r", "b")),
            join_type: JoinType::Inner,
            on: vec![(col("a"), col("b"))],
            filter: None,
        }
    }

    #[test]
    fn folds_both_side_filter_into_inner_join() {
        let plan = LogicalPlan::Filter {
            input: Box::new(inner_join()),
            predicate: Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(col("a")),
                right: Box::new(col("b")),
            },
        };
        let before = plan.schema().expect("typecheck");
        let out = FilterIntoJoin.rewrite(plan).expect("fold");
        let after = out.schema().expect("typecheck after");
        assert_eq!(before.fields.len(), after.fields.len());
        match out {
            LogicalPlan::Join { filter, .. } => {
                assert!(filter.is_some(), "residual should carry the folded predicate");
            }
            other => panic!("expected Join (no Filter above), got {other:?}"),
        }
    }

    #[test]
    fn merges_with_existing_residual() {
        let mut join = inner_join();
        if let LogicalPlan::Join { filter, .. } = &mut join {
            *filter = Some(col("a").lt(lit(100_i64)));
        }
        let plan = LogicalPlan::Filter {
            input: Box::new(join),
            predicate: Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(col("a")),
                right: Box::new(col("b")),
            },
        };
        let out = FilterIntoJoin.rewrite(plan).expect("fold");
        match out {
            LogicalPlan::Join { filter: Some(f), .. } => {
                // Two conjuncts AND-ed together.
                assert!(matches!(f, Expr::Binary { op: BinaryOp::And, .. }));
            }
            other => panic!("expected Join with merged residual, got {other:?}"),
        }
    }

    #[test]
    fn leaves_outer_join_filter_alone() {
        let join = LogicalPlan::Join {
            left: Box::new(scan("l", "a")),
            right: Box::new(scan("r", "b")),
            join_type: JoinType::LeftOuter,
            on: vec![(col("a"), col("b"))],
            filter: None,
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(join),
            predicate: Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(col("a")),
                right: Box::new(col("b")),
            },
        };
        let out = FilterIntoJoin.rewrite(plan).expect("noop");
        assert!(matches!(out, LogicalPlan::Filter { .. }),
            "outer-join post-filter must not be folded into the residual");
    }
}
