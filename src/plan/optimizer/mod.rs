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

use std::sync::Arc;

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
pub use join_reorder::{JoinReorder, NoStats, RowEstimator, StatsEstimator};
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
    default_passes_with_estimator(Arc::new(NoStats))
}

/// Build the default optimizer pipeline with a caller-supplied row estimator
/// driving the [`JoinReorder`] pass.
///
/// Identical to [`default_passes`] in every pass *but* join reordering, which
/// is constructed with `estimator` instead of the no-op [`NoStats`]. The
/// engine calls this from `Engine::sql` / `Engine::run_logical_plan` with a
/// statistics-backed estimator (see [`StatsEstimator`]) so left-deep INNER
/// chains are reordered smallest-input-first when base-table row counts are
/// known, while still degrading to the conservative no-op for any chain whose
/// leaves the estimator cannot fully cost.
///
/// `default_passes()` itself is the `NoStats` special case, preserved so
/// callers (and tests) that don't have statistics keep their exact previous
/// behaviour.
pub fn default_passes_with_estimator(
    estimator: Arc<dyn RowEstimator>,
) -> Vec<Box<dyn PlanRewrite>> {
    vec![
        Box::new(ConstantFold),
        Box::new(PredicatePushdown),
        Box::new(FilterIntoJoin),
        Box::new(JoinReorder::with_estimator(estimator)),
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
    fn estimator_pipeline_has_same_pass_order() {
        // `default_passes_with_estimator` differs from `default_passes` only
        // in which estimator drives join-reorder — the pass *list* is identical.
        let names: Vec<String> = default_passes_with_estimator(Arc::new(NoStats))
            .iter()
            .map(|p| p.name().to_string())
            .collect();
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
    fn estimator_pipeline_reorders_inner_chain() {
        use crate::plan::statistics::{StatsProvider, TableStats};
        use std::collections::HashMap as Map;

        /// Row-count-only `StatsProvider`.
        struct Stats(Map<String, usize>);
        impl StatsProvider for Stats {
            fn table_stats(&self, name: &str) -> Option<TableStats> {
                self.0.get(name).map(|&n| TableStats::new(n))
            }
        }

        fn scan(table: &str, c: &str) -> LogicalPlan {
            LogicalPlan::Scan {
                table: table.into(),
                projection: None,
                schema: Schema::new(vec![Field::new(c, DataType::Int64, false)]),
            }
        }

        // a(k) JOIN b(k2,m) JOIN c(m2): a=1000, b=10, c=5.
        let b = LogicalPlan::Scan {
            table: "b".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("k2", DataType::Int64, false),
                Field::new("m", DataType::Int64, false),
            ]),
        };
        let ab = LogicalPlan::Join {
            left: Box::new(scan("a", "k")),
            right: Box::new(b),
            join_type: JoinType::Inner,
            on: vec![(col("k"), col("k2"))],
            filter: None,
        };
        let plan = LogicalPlan::Join {
            left: Box::new(ab),
            right: Box::new(scan("c", "m2")),
            join_type: JoinType::Inner,
            on: vec![(col("m"), col("m2"))],
            filter: None,
        };

        let mut counts = Map::new();
        counts.insert("a".to_string(), 1000usize);
        counts.insert("b".to_string(), 10usize);
        counts.insert("c".to_string(), 5usize);
        let est = std::sync::Arc::new(StatsEstimator::new(Stats(counts)));

        let before = plan.schema().expect("typecheck");
        let mut p = plan;
        for pass in default_passes_with_estimator(est) {
            p = pass.rewrite(p).expect("pass must succeed");
        }
        let after = p.schema().expect("typecheck after");

        // Output column *set* preserved across the reorder.
        let bset: std::collections::HashSet<_> =
            before.fields.iter().map(|f| f.name.clone()).collect();
        let aset: std::collections::HashSet<_> =
            after.fields.iter().map(|f| f.name.clone()).collect();
        assert_eq!(bset, aset);

        // The smallest table 'c' should now be the deepest-left leaf.
        fn deepest_left_scan(plan: &LogicalPlan) -> Option<&str> {
            match plan {
                LogicalPlan::Join { left, .. } => deepest_left_scan(left),
                LogicalPlan::Scan { table, .. } => Some(table.as_str()),
                _ => None,
            }
        }
        assert_eq!(
            deepest_left_scan(&p),
            Some("c"),
            "smallest input must sink to the build-side leaf"
        );
    }

    #[test]
    fn estimator_pipeline_noop_without_stats() {
        // With NoStats the pipeline must leave a reorderable chain's join order
        // structurally identical to `default_passes()`'s output.
        fn scan(table: &str, c: &str) -> LogicalPlan {
            LogicalPlan::Scan {
                table: table.into(),
                projection: None,
                schema: Schema::new(vec![Field::new(c, DataType::Int64, false)]),
            }
        }
        let b = LogicalPlan::Scan {
            table: "b".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("k2", DataType::Int64, false),
                Field::new("m", DataType::Int64, false),
            ]),
        };
        let make = || {
            let ab = LogicalPlan::Join {
                left: Box::new(scan("a", "k")),
                right: Box::new(b.clone()),
                join_type: JoinType::Inner,
                on: vec![(col("k"), col("k2"))],
                filter: None,
            };
            LogicalPlan::Join {
                left: Box::new(ab),
                right: Box::new(scan("c", "m2")),
                join_type: JoinType::Inner,
                on: vec![(col("m"), col("m2"))],
                filter: None,
            }
        };

        let mut a = make();
        for pass in default_passes() {
            a = pass.rewrite(a).expect("pass");
        }
        let mut b2 = make();
        for pass in default_passes_with_estimator(Arc::new(NoStats)) {
            b2 = pass.rewrite(b2).expect("pass");
        }
        assert_eq!(format!("{a:?}"), format!("{b2:?}"));
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
