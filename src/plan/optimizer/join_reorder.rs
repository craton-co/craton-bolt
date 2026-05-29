// SPDX-License-Identifier: Apache-2.0

//! Conservative join reordering for chained left-deep INNER joins.
//!
//! A query like `a JOIN b JOIN c` parses into a left-deep tree
//! `Join(Join(a, b), c)`. When every join in the chain is a commutative
//! `INNER` join with pure equi-`on` pairs (and no residual `filter`), the
//! relative order of the *leaf* inputs does not affect the result set, only
//! the work the executor does — putting smaller inputs first shrinks the
//! intermediate hash tables.
//!
//! This pass:
//!
//! 1. flattens a maximal run of such joins into the ordered list of leaf
//!    inputs and the multiset of equi-`on` pairs;
//! 2. estimates each leaf's row count via [`RowEstimator`];
//! 3. if **every** leaf has an estimate, reorders the leaves smallest-first
//!    and rebuilds a left-deep tree, re-deriving each join's `on` pairs from
//!    the equi-pairs whose two columns are now both available;
//! 4. otherwise leaves the chain exactly as it was (the conservative default
//!    — without statistics, the original order is preserved).
//!
//! Correctness guards (any failure => leave the chain untouched):
//!
//! * only `INNER` joins with non-empty `on` and `filter == None` participate;
//! * a reordered tree is only accepted if every original equi-pair can be
//!   re-placed at a join level where both of its columns are in scope, so the
//!   rebuilt plan is structurally valid;
//! * the leaf *set* is identical, so the output schema's field set is
//!   preserved (column *order* in the combined schema may change — see the
//!   note on [`JoinReorder`]).
//!
//! Because the default [`RowEstimator`] (`NoStats`) returns `None` for every
//! leaf, the pass is a structural no-op in the absence of statistics, which is
//! exactly the conservative behaviour the engine wires in by default. Tests
//! supply a stub estimator to exercise the reordering path.

use std::collections::HashSet;
use std::sync::Arc;

use crate::error::BoltResult;
use crate::plan::logical_plan::{Expr, JoinType, LogicalPlan};
use crate::plan::rewrite::PlanRewrite;

use super::expr_util::collect_columns;

/// Row-count estimator for join leaves. Implementations return `Some(rows)`
/// when a cardinality hint is available for the given leaf plan, or `None`
/// when unknown. The optimizer only reorders a chain when *every* leaf has an
/// estimate, so a partial estimator is safe (it simply disables reordering for
/// chains it can't fully cost).
pub trait RowEstimator: Send + Sync {
    /// Estimated row count of `plan`, or `None` if unknown.
    fn estimate(&self, plan: &LogicalPlan) -> Option<u64>;
}

/// Default estimator: no statistics available, so every estimate is `None`
/// and join reordering becomes a structural no-op. This is what the engine
/// installs by default — reordering only kicks in once a richer estimator
/// (e.g. one backed by registered-table row counts) is supplied.
#[derive(Debug, Default)]
pub struct NoStats;

impl RowEstimator for NoStats {
    fn estimate(&self, _plan: &LogicalPlan) -> Option<u64> {
        None
    }
}

/// Join-reordering pass. See module docs.
///
/// **Schema note:** reordering the leaves of an INNER-join chain preserves the
/// *set* of output columns but may change their left-to-right *order* in the
/// combined schema. Downstream nodes reference columns by name, and the SQL
/// frontend always wraps a join in an explicit SELECT-list `Project`, so the
/// user-visible output order is unaffected. The pass still only fires when an
/// estimator is supplied; the default `NoStats` keeps order stable.
pub struct JoinReorder {
    estimator: Arc<dyn RowEstimator>,
}

impl Default for JoinReorder {
    fn default() -> Self {
        Self {
            estimator: Arc::new(NoStats),
        }
    }
}

impl std::fmt::Debug for JoinReorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JoinReorder").finish_non_exhaustive()
    }
}

impl JoinReorder {
    /// Construct with a custom row estimator (e.g. backed by table stats).
    pub fn with_estimator(estimator: Arc<dyn RowEstimator>) -> Self {
        Self { estimator }
    }
}

impl PlanRewrite for JoinReorder {
    fn name(&self) -> &str {
        "join-reorder"
    }

    fn rewrite(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
        Ok(self.rewrite_plan(plan))
    }
}

impl JoinReorder {
    fn rewrite_plan(&self, plan: LogicalPlan) -> LogicalPlan {
        // Recurse into children first.
        let plan = self.recurse_children(plan);
        // Then attempt to reorder a chain rooted here.
        if let LogicalPlan::Join { .. } = &plan {
            if let Some(reordered) = self.try_reorder_chain(&plan) {
                return reordered;
            }
        }
        plan
    }

    fn recurse_children(&self, plan: LogicalPlan) -> LogicalPlan {
        match plan {
            LogicalPlan::Scan { .. } => plan,
            LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
                input: Box::new(self.rewrite_plan(*input)),
                predicate,
            },
            LogicalPlan::Project { input, exprs } => LogicalPlan::Project {
                input: Box::new(self.rewrite_plan(*input)),
                exprs,
            },
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregates,
            } => LogicalPlan::Aggregate {
                input: Box::new(self.rewrite_plan(*input)),
                group_by,
                aggregates,
            },
            LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
                input: Box::new(self.rewrite_plan(*input)),
            },
            LogicalPlan::Limit {
                input,
                limit,
                offset,
            } => LogicalPlan::Limit {
                input: Box::new(self.rewrite_plan(*input)),
                limit,
                offset,
            },
            LogicalPlan::Sort { input, sort_exprs } => LogicalPlan::Sort {
                input: Box::new(self.rewrite_plan(*input)),
                sort_exprs,
            },
            LogicalPlan::Union { inputs } => LogicalPlan::Union {
                inputs: inputs.into_iter().map(|i| self.rewrite_plan(i)).collect(),
            },
            LogicalPlan::Join {
                left,
                right,
                join_type,
                on,
                filter,
            } => LogicalPlan::Join {
                left: Box::new(self.rewrite_plan(*left)),
                right: Box::new(self.rewrite_plan(*right)),
                join_type,
                on,
                filter,
            },
        }
    }

    /// Try to flatten + reorder a maximal left-deep INNER-join chain rooted at
    /// `plan`. Returns `None` (caller keeps the original) when the chain isn't
    /// reorderable or any leaf lacks an estimate.
    fn try_reorder_chain(&self, plan: &LogicalPlan) -> Option<LogicalPlan> {
        let mut leaves: Vec<LogicalPlan> = Vec::new();
        let mut equi_pairs: Vec<(Expr, Expr)> = Vec::new();
        if !flatten_chain(plan, &mut leaves, &mut equi_pairs) {
            return None;
        }
        if leaves.len() < 3 {
            // A 2-way join is already a single ordering decision the executor
            // makes itself (build vs probe side); nothing to reorder here.
            return None;
        }

        // Every leaf must have an estimate, else stay conservative.
        let mut estimated: Vec<(u64, LogicalPlan)> = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            let rows = self.estimator.estimate(&leaf)?;
            estimated.push((rows, leaf));
        }

        // Stable sort smallest-first; ties keep original relative order.
        estimated.sort_by_key(|(rows, _)| *rows);
        let ordered: Vec<LogicalPlan> = estimated.into_iter().map(|(_, p)| p).collect();

        rebuild_left_deep(ordered, equi_pairs)
    }
}

/// Flatten a left-deep chain of reorderable INNER joins into `leaves` (in
/// left-to-right order) and the collected equi-`on` pairs. Returns `false`
/// when the chain contains a non-reorderable join (then the whole attempt is
/// abandoned by the caller).
///
/// Only the *left spine* is followed: `Join(Join(a, b), c)` flattens to
/// `[a, b, c]`. A right child that is itself a join is treated as a single
/// opaque leaf (it has already been recursed into and reordered on its own).
fn flatten_chain(
    plan: &LogicalPlan,
    leaves: &mut Vec<LogicalPlan>,
    equi_pairs: &mut Vec<(Expr, Expr)>,
) -> bool {
    match plan {
        LogicalPlan::Join {
            left,
            right,
            join_type: JoinType::Inner,
            on,
            filter: None,
        } if !on.is_empty() => {
            for pair in on {
                equi_pairs.push(pair.clone());
            }
            // Recurse down the left spine; the right child is a leaf.
            if !flatten_chain(left, leaves, equi_pairs) {
                return false;
            }
            leaves.push((**right).clone());
            true
        }
        // Any non-reorderable node terminates the spine as a single leaf.
        other => {
            leaves.push(other.clone());
            true
        }
    }
}

/// Rebuild a left-deep INNER-join tree from `ordered` leaves, distributing
/// `equi_pairs` to the join level at which both of their columns first become
/// available. Returns `None` if any equi-pair cannot be placed (then the
/// caller keeps the original chain).
fn rebuild_left_deep(
    ordered: Vec<LogicalPlan>,
    equi_pairs: Vec<(Expr, Expr)>,
) -> Option<LogicalPlan> {
    let mut iter = ordered.into_iter();
    let first = iter.next()?;
    // Columns currently in scope on the accumulated left subtree.
    let mut scope: HashSet<String> = column_names(&first);
    let mut acc = first;
    // Remaining equi-pairs not yet attached.
    let mut remaining: Vec<(Expr, Expr)> = equi_pairs;

    for right in iter {
        let right_cols = column_names(&right);
        // Pull every pair whose columns are split across (scope, right_cols)
        // — i.e. one side resolves in the accumulated left, the other in the
        // new right input.
        let mut here: Vec<(Expr, Expr)> = Vec::new();
        let mut keep: Vec<(Expr, Expr)> = Vec::new();
        for (l, r) in remaining.into_iter() {
            if pair_spans(&l, &r, &scope, &right_cols) {
                here.push((l, r));
            } else {
                keep.push((l, r));
            }
        }
        remaining = keep;
        if here.is_empty() {
            // No join condition connects this leaf to the accumulated tree
            // at this position — would produce a cross product the original
            // plan did not have. Bail out and keep the original order.
            return None;
        }
        // Merge the new leaf's columns into scope.
        scope.extend(right_cols);
        acc = LogicalPlan::Join {
            left: Box::new(acc),
            right: Box::new(right),
            join_type: JoinType::Inner,
            on: here,
            filter: None,
        };
    }

    // Every equi-pair must have been consumed; a leftover means the rebuilt
    // tree would silently drop a join condition.
    if remaining.is_empty() {
        Some(acc)
    } else {
        None
    }
}

/// True if the two sides of an equi-pair resolve across the boundary: one side
/// in `left_scope`, the other in `right_cols` (in either orientation).
fn pair_spans(
    l: &Expr,
    r: &Expr,
    left_scope: &HashSet<String>,
    right_cols: &HashSet<String>,
) -> bool {
    let l_in_left = expr_cols_within(l, left_scope);
    let l_in_right = expr_cols_within(l, right_cols);
    let r_in_left = expr_cols_within(r, left_scope);
    let r_in_right = expr_cols_within(r, right_cols);
    (l_in_left && r_in_right) || (l_in_right && r_in_left)
}

/// True if every column referenced by `expr` is contained in `scope`.
fn expr_cols_within(expr: &Expr, scope: &HashSet<String>) -> bool {
    let mut cols = Vec::new();
    collect_columns(expr, &mut cols);
    !cols.is_empty() && cols.iter().all(|c| scope.contains(c))
}

/// Output column names of `plan`. On a type-check failure (shouldn't happen on
/// a valid plan) returns an empty set, which makes [`rebuild_left_deep`] bail
/// out conservatively.
fn column_names(plan: &LogicalPlan) -> HashSet<String> {
    match plan.schema() {
        Ok(s) => s.fields.into_iter().map(|f| f.name).collect(),
        Err(_) => HashSet::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{DataType, Field, Schema};
    use crate::plan::col;

    fn leaf(table: &str, col_name: &str) -> LogicalPlan {
        LogicalPlan::Scan {
            table: table.into(),
            projection: None,
            schema: Schema::new(vec![Field::new(col_name, DataType::Int64, false)]),
        }
    }

    /// Stub estimator keyed by table name -> row count.
    struct ByTable(std::collections::HashMap<String, u64>);
    impl RowEstimator for ByTable {
        fn estimate(&self, plan: &LogicalPlan) -> Option<u64> {
            match plan {
                LogicalPlan::Scan { table, .. } => self.0.get(table).copied(),
                _ => None,
            }
        }
    }

    /// a(k) JOIN b(k,m) ON a.k=b.k JOIN c(m) ON b.m=c.m
    fn three_way() -> LogicalPlan {
        let a = leaf("a", "k");
        let b = LogicalPlan::Scan {
            table: "b".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("k2", DataType::Int64, false),
                Field::new("m", DataType::Int64, false),
            ]),
        };
        let c = leaf("c", "m2");
        let ab = LogicalPlan::Join {
            left: Box::new(a),
            right: Box::new(b),
            join_type: JoinType::Inner,
            on: vec![(col("k"), col("k2"))],
            filter: None,
        };
        LogicalPlan::Join {
            left: Box::new(ab),
            right: Box::new(c),
            join_type: JoinType::Inner,
            on: vec![(col("m"), col("m2"))],
            filter: None,
        }
    }

    #[test]
    fn nostats_is_noop() {
        let plan = three_way();
        let before = format!("{:?}", plan);
        let out = JoinReorder::default().rewrite(plan).expect("noop");
        assert_eq!(before, format!("{:?}", out), "NoStats must leave order stable");
    }

    #[test]
    fn reorders_smallest_first_with_stats() {
        // a=1000, b=10, c=5. The chain is left-deep [a, b, c]; smallest-first
        // ordering is [c, b, a]. But reordering must still be able to place
        // every equi-pair, so the result connects c-b (m=m), then a (k=k2).
        let mut stats = std::collections::HashMap::new();
        stats.insert("a".to_string(), 1000);
        stats.insert("b".to_string(), 10);
        stats.insert("c".to_string(), 5);
        let est = Arc::new(ByTable(stats));
        let pass = JoinReorder::with_estimator(est);
        let plan = three_way();
        let before = plan.schema().expect("typecheck");
        let out = pass.rewrite(plan).expect("reorder");
        let after = out.schema().expect("typecheck after");
        // Same set of output columns (order may differ).
        let bset: HashSet<_> = before.fields.iter().map(|f| f.name.clone()).collect();
        let aset: HashSet<_> = after.fields.iter().map(|f| f.name.clone()).collect();
        assert_eq!(bset, aset, "reorder must preserve the output column set");
        // It should still be a 2-level left-deep INNER join tree.
        match &out {
            LogicalPlan::Join { left, join_type, .. } => {
                assert_eq!(*join_type, JoinType::Inner);
                assert!(matches!(**left, LogicalPlan::Join { .. }));
            }
            other => panic!("expected left-deep join tree, got {other:?}"),
        }
    }

    #[test]
    fn outer_join_chain_not_reordered() {
        let a = leaf("a", "k");
        let b = leaf("b", "k2");
        let c = leaf("c", "k3");
        let ab = LogicalPlan::Join {
            left: Box::new(a),
            right: Box::new(b),
            join_type: JoinType::LeftOuter,
            on: vec![(col("k"), col("k2"))],
            filter: None,
        };
        let abc = LogicalPlan::Join {
            left: Box::new(ab),
            right: Box::new(c),
            join_type: JoinType::Inner,
            on: vec![(col("k2"), col("k3"))],
            filter: None,
        };
        let mut stats = std::collections::HashMap::new();
        stats.insert("a".to_string(), 100);
        stats.insert("b".to_string(), 1);
        stats.insert("c".to_string(), 1);
        let pass = JoinReorder::with_estimator(Arc::new(ByTable(stats)));
        let before = format!("{:?}", abc);
        let out = pass.rewrite(abc).expect("noop");
        // The presence of a LEFT join in the spine makes the chain
        // non-reorderable; it terminates the spine as an opaque leaf, leaving
        // a 2-leaf chain (< 3) that the pass declines to touch.
        assert_eq!(before, format!("{:?}", out));
    }
}
