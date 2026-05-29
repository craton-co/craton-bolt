// SPDX-License-Identifier: Apache-2.0

//! Built-in logical-plan optimizer: a fixed pipeline of [`PlanRewrite`] passes
//! the engine runs by default before lowering to the physical plan.
//!
//! Each pass is an independent [`PlanRewrite`] implementation (one per
//! submodule) that takes a [`LogicalPlan`] and returns a structurally valid,
//! schema-preserving rewrite. The engine threads them in
//! [`default_passes`] order, then runs any user-registered rewrites, then
//! lowers. None of the passes are GPU-aware — they operate purely on the
//! logical IR.
//!
//! ## Pipeline order
//!
//! The default order is chosen so each pass feeds the next:
//!
//! 1. [`ConstantFold`] — fold literal arithmetic and simplify boolean
//!    expressions first, so later passes see canonical predicates (e.g. a
//!    folded `Bool(true)` conjunct that pushdown can then drop).
//! 2. [`PredicatePushdown`] — split `WHERE` conjunctions and sink each conjunct
//!    toward the scan that owns its columns, through projections and into the
//!    correct side of joins.
//! 3. [`FilterIntoJoin`] — fold the both-sides conjuncts that pushdown left
//!    sitting above INNER/CROSS joins into the join residual `filter`.
//! 4. [`JoinReorder`] — conservatively reorder left-deep INNER-join chains
//!    smallest-input-first (a no-op without row statistics).
//! 5. [`ProjectionPruning`] — last, so it sees the final column references
//!    after pushdown/reorder and narrows each scan to the live columns.
//!
//! ## Contract
//!
//! Every pass preserves the plan's external output schema (field set, dtypes,
//! and — except for the documented column-*order* caveat on [`JoinReorder`] —
//! order) and semantics, per the [`PlanRewrite`] trait contract. Passes are
//! single-pass (no internal fixpoint loop); the pipeline is run once.

use crate::plan::rewrite::PlanRewrite;

pub mod const_fold;
pub mod expr_util;
pub mod filter_into_join;
pub mod join_reorder;
pub mod plan_util;
pub mod predicate_pushdown;
pub mod projection_pruning;

pub use const_fold::ConstantFold;
pub use filter_into_join::FilterIntoJoin;
pub use join_reorder::{JoinReorder, NoStats, RowEstimator};
pub use predicate_pushdown::PredicatePushdown;
pub use projection_pruning::ProjectionPruning;

/// Build the default optimizer pass pipeline, in execution order.
///
/// The engine prepends these to any user-registered [`PlanRewrite`]s and runs
/// the whole chain before lowering. See the module docs for the rationale
/// behind the ordering. The returned passes are stateless / use the default
/// (`NoStats`) row estimator, so join reordering is a no-op until a
/// statistics-backed estimator is wired in.
pub fn default_passes() -> Vec<Box<dyn PlanRewrite>> {
    vec![
        Box::new(ConstantFold),
        Box::new(PredicatePushdown),
        Box::new(FilterIntoJoin),
        Box::new(JoinReorder::default()),
        Box::new(ProjectionPruning),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{
        BinaryOp, DataType, Field, JoinType, LogicalPlan, Schema,
    };
    use crate::plan::{col, lit};

    /// Run the full default pipeline over `plan`.
    fn run(plan: LogicalPlan) -> LogicalPlan {
        let mut p = plan;
        for pass in default_passes() {
            p = pass.rewrite(p).expect("pass must succeed");
        }
        p
    }

    #[test]
    fn pipeline_has_all_five_passes_in_order() {
        let names: Vec<String> = default_passes().iter().map(|p| p.name().to_string()).collect();
        assert_eq!(
            names,
            vec![
                "constant-fold",
                "predicate-pushdown",
                "filter-into-join",
                "join-reorder",
                "projection-pruning",
            ]
        );
    }

    #[test]
    fn end_to_end_preserves_output_schema() {
        // SELECT a FROM t WHERE (1 = 1) AND b > 0
        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Int64, false),
                Field::new("c", DataType::Int64, false),
            ]),
        };
        let plan = LogicalPlan::Project {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(scan),
                predicate: lit(1_i64).eq(lit(1_i64)).and(col("b").gt(lit(0_i64))),
            }),
            exprs: vec![col("a")],
        };
        let before = plan.schema().expect("typecheck");
        let out = run(plan);
        let after = out.schema().expect("typecheck after pipeline");
        assert_eq!(
            before.fields.iter().map(|f| &f.name).collect::<Vec<_>>(),
            after.fields.iter().map(|f| &f.name).collect::<Vec<_>>(),
        );
        assert_eq!(
            before.fields.iter().map(|f| f.dtype).collect::<Vec<_>>(),
            after.fields.iter().map(|f| f.dtype).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn end_to_end_folds_pushes_and_prunes() {
        // `1 = 1` folds to true and drops out of the predicate; `b > 0` pushes
        // below the project to the scan; the scan prunes to [a, b].
        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Int64, false),
                Field::new("c", DataType::Int64, false),
            ]),
        };
        let plan = LogicalPlan::Project {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(scan),
                predicate: lit(1_i64).eq(lit(1_i64)).and(col("b").gt(lit(0_i64))),
            }),
            exprs: vec![col("a")],
        };
        let out = run(plan);
        // Top: Project([a]); below it a Filter(b > 0); below that the pruned scan.
        let filter = match out {
            LogicalPlan::Project { input, .. } => *input,
            other => panic!("expected Project on top, got {other:?}"),
        };
        let scan = match filter {
            LogicalPlan::Filter { input, predicate } => {
                // The `1 = 1` conjunct folded away; only `b > 0` remains.
                assert!(
                    matches!(predicate, crate::plan::Expr::Binary { op: BinaryOp::Gt, .. }),
                    "expected the surviving predicate to be `b > 0`"
                );
                *input
            }
            other => panic!("expected Filter below project, got {other:?}"),
        };
        match scan {
            LogicalPlan::Scan { projection, .. } => {
                let p = projection.expect("scan should be pruned");
                assert_eq!(p, vec!["a".to_string(), "b".to_string()]);
            }
            other => panic!("expected pruned Scan, got {other:?}"),
        }
    }

    #[test]
    fn end_to_end_join_filter_routing() {
        // l JOIN r ON a=b WHERE a > 0 AND a > b
        // => a>0 pushes to left scan; a>b folds into the join residual.
        let left = LogicalPlan::Scan {
            table: "l".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("a", DataType::Int64, false)]),
        };
        let right = LogicalPlan::Scan {
            table: "r".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("b", DataType::Int64, false)]),
        };
        let join = LogicalPlan::Join {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            on: vec![(col("a"), col("b"))],
            filter: None,
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(join),
            predicate: col("a").gt(lit(0_i64)).and(gt(col("a"), col("b"))),
        };
        let out = run(plan);
        match out {
            LogicalPlan::Join { left, filter, .. } => {
                assert!(filter.is_some(), "a > b should fold into the residual");
                assert!(matches!(*left, LogicalPlan::Filter { .. }),
                    "a > 0 should land on the left input");
            }
            other => panic!("expected Join at the root, got {other:?}"),
        }
    }

    fn gt(l: crate::plan::Expr, r: crate::plan::Expr) -> crate::plan::Expr {
        crate::plan::Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(l),
            right: Box::new(r),
        }
    }
}
