// SPDX-License-Identifier: Apache-2.0

//! Cost-based join reordering for chained INNER equi-joins.
//!
//! A query like `a JOIN b JOIN c` parses into a left-deep tree
//! `Join(Join(a, b), c)`. When every join in the chain is a commutative
//! `INNER` join with pure equi-`on` pairs (and no residual `filter`), the
//! relative order *and shape* of the leaf inputs does not affect the result
//! set, only the work the executor does — joining small inputs early shrinks
//! the intermediate hash tables that flow through the pipeline.
//!
//! This pass:
//!
//! 1. flattens a maximal run of such joins into the list of leaf inputs and the
//!    multiset of equi-`on` pairs;
//! 2. estimates each leaf's row count via [`RowEstimator`] (every leaf must
//!    have an estimate, else the pass stays a no-op);
//! 3. maps each equi-pair to the pair of leaves it connects, building a
//!    cardinality + connectivity model ([`super::cost::CardModel`]);
//! 4. asks the cost enumerator ([`super::cost::optimize`]) for the cheapest
//!    join *shape* — Selinger-style DP for small chains, a greedy fallback past
//!    [`super::cost::MAX_DP_RELATIONS`] — minimising the sum of intermediate
//!    cardinalities. The result may be **bushy**, not just left-deep;
//! 5. rebuilds a `LogicalPlan` from that shape, re-attaching each original
//!    equi-pair at the join level where both of its columns are in scope;
//! 6. otherwise (no estimate, disconnected graph, unplaceable pair) leaves the
//!    chain exactly as it was — the conservative default.
//!
//! Correctness guards (any failure => leave the chain untouched):
//!
//! * only `INNER` joins with non-empty `on` and `filter == None` participate;
//! * every equi-pair must resolve to exactly two distinct leaves, and the leaf
//!   set must be a single connected component over the equi-key graph — a
//!   reorder never invents a cross product the original chain lacked;
//! * a reordered tree is only accepted if every original equi-pair can be
//!   re-placed at a join level where both of its columns are in scope and every
//!   pair is consumed exactly once, so no `on` condition is dropped or
//!   duplicated and the rebuilt plan is structurally valid;
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
use crate::plan::statistics::{estimate_rows, StatsProvider};

use super::cost::{optimize, CardModel, JoinShape};
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

/// Cost-based [`RowEstimator`] bridging the join-reorder pass to the
/// standalone cardinality model in [`crate::plan::statistics`].
///
/// This is the adapter that makes join reordering *actually* cost-based: it
/// owns a [`StatsProvider`] (base-table statistics) and answers
/// [`RowEstimator::estimate`] by running
/// [`estimate_rows`](crate::plan::statistics::estimate_rows) over the leaf
/// plan. The two modules are otherwise decoupled — `statistics.rs` knows
/// nothing about the optimizer's `RowEstimator` trait, and the optimizer knows
/// nothing about how statistics are sourced — so this newtype is the single
/// seam that joins them.
///
/// The provider is owned (not borrowed) so the estimator can live behind the
/// `Arc<dyn RowEstimator>` the pass holds, which outlives any single rewrite.
/// `estimate_rows` returns a `usize` clamped to `>= 1`; we widen it to the
/// `u64` the trait speaks. A leaf whose base table has no stats entry yields
/// `None`, which (per the pass contract) disables reordering for that whole
/// chain — exactly the conservative fallback we want.
pub struct StatsEstimator<P: StatsProvider + Send + Sync> {
    provider: P,
}

impl<P: StatsProvider + Send + Sync> StatsEstimator<P> {
    /// Construct an estimator backed by `provider`.
    pub fn new(provider: P) -> Self {
        Self { provider }
    }
}

impl<P: StatsProvider + Send + Sync> RowEstimator for StatsEstimator<P> {
    fn estimate(&self, plan: &LogicalPlan) -> Option<u64> {
        // `estimate_rows` clamps its result into `[1, usize::MAX]`, so the
        // `as u64` cast is lossless on 64-bit targets and saturating-by-clamp
        // on 32-bit ones (a row count above `u64::MAX` is unreachable anyway).
        estimate_rows(plan, &self.provider).map(|rows| rows as u64)
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
            LogicalPlan::Window { input, window_exprs, partition_by, order_by } => LogicalPlan::Window {
                input: Box::new(self.rewrite_plan(*input)),
                window_exprs,
                partition_by,
                order_by,
            },
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
            LogicalPlan::SetOp { left, right, op, all } => LogicalPlan::SetOp {
                left: Box::new(self.rewrite_plan(*left)),
                right: Box::new(self.rewrite_plan(*right)),
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
        let mut leaf_rows: Vec<u64> = Vec::with_capacity(leaves.len());
        for leaf in &leaves {
            leaf_rows.push(self.estimator.estimate(leaf)?);
        }

        // Per-leaf output column sets, used both to map each equi-pair to the
        // pair of leaves it connects (the cost model's edges) and, during
        // rebuild, to place each pair at the join level where both columns are
        // in scope. An untypeable leaf yields an empty set and aborts below.
        let leaf_cols: Vec<HashSet<String>> = leaves.iter().map(column_names).collect();

        // Map each equi-pair to the (i, j) leaf indices it connects. A pair
        // whose two sides do not resolve to exactly two distinct leaves makes
        // the chain unsafe to re-derive, so bail to the conservative no-op.
        let mut edges: Vec<(usize, usize)> = Vec::with_capacity(equi_pairs.len());
        for (l, r) in &equi_pairs {
            let li = leaf_for_expr(l, &leaf_cols)?;
            let ri = leaf_for_expr(r, &leaf_cols)?;
            if li == ri {
                return None;
            }
            edges.push((li, ri));
        }

        // Cost-based enumeration: pick the cheapest (possibly bushy) join shape.
        let model = CardModel::new(leaf_rows, &edges);
        let plan = optimize(&model)?;

        rebuild_from_shape(&plan.shape, &leaves, &equi_pairs)
    }
}

/// Resolve the single leaf index whose output columns contain *every* column
/// referenced by `expr`. Returns `None` when `expr` references no columns, or
/// when its columns are not all contained in exactly one leaf (a key spanning
/// two leaves, or referencing an unknown column, makes the rewrite unsafe).
fn leaf_for_expr(expr: &Expr, leaf_cols: &[HashSet<String>]) -> Option<usize> {
    let mut cols = Vec::new();
    collect_columns(expr, &mut cols);
    if cols.is_empty() {
        return None;
    }
    let mut found: Option<usize> = None;
    for (idx, set) in leaf_cols.iter().enumerate() {
        if cols.iter().all(|c| set.contains(c)) {
            if found.is_some() {
                // Columns satisfiable by more than one leaf is ambiguous.
                return None;
            }
            found = Some(idx);
        }
    }
    found
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

/// Materialise an abstract [`JoinShape`] (over leaf indices) back into a
/// `LogicalPlan`, cloning the chosen `leaves` and re-attaching every original
/// equi-pair at the join level where both of its columns first become in scope.
///
/// Returns `None` (caller keeps the original chain) if the shape's leaf set
/// does not cover every leaf exactly once, if any join level finds no equi-pair
/// connecting its two children (a cross product the original plan lacked), or
/// if any equi-pair is left unplaced (a dropped join condition). These guards
/// make the rewrite a strict no-op whenever it cannot reproduce the original
/// semantics.
fn rebuild_from_shape(
    shape: &JoinShape,
    leaves: &[LogicalPlan],
    equi_pairs: &[(Expr, Expr)],
) -> Option<LogicalPlan> {
    // Sanity: the shape must cover every leaf exactly once.
    let covered = shape.leaves();
    if covered.len() != leaves.len() || (0..leaves.len()).any(|i| !covered.contains(&i)) {
        return None;
    }
    let mut remaining: Vec<(Expr, Expr)> = equi_pairs.to_vec();
    let (plan, _scope) = build_node(shape, leaves, &mut remaining)?;
    // Every equi-pair must have been consumed; a leftover means the rebuilt
    // tree would silently drop a join condition.
    if remaining.is_empty() {
        Some(plan)
    } else {
        None
    }
}

/// Recursively build the `LogicalPlan` for one [`JoinShape`] node, returning the
/// node and the set of column names in its output scope. Equi-pairs consumed at
/// a join level are removed from `remaining`.
fn build_node(
    shape: &JoinShape,
    leaves: &[LogicalPlan],
    remaining: &mut Vec<(Expr, Expr)>,
) -> Option<(LogicalPlan, HashSet<String>)> {
    match shape {
        JoinShape::Leaf(i) => {
            let leaf = leaves.get(*i)?.clone();
            let cols = column_names(&leaf);
            Some((leaf, cols))
        }
        JoinShape::Join { left, right } => {
            let (left_plan, left_cols) = build_node(left, leaves, remaining)?;
            let (right_plan, right_cols) = build_node(right, leaves, remaining)?;
            // Pull every still-unplaced pair whose two columns straddle the
            // (left_cols, right_cols) boundary — those are this join's `on`.
            let mut here: Vec<(Expr, Expr)> = Vec::new();
            let mut keep: Vec<(Expr, Expr)> = Vec::new();
            for (l, r) in remaining.drain(..) {
                if pair_spans(&l, &r, &left_cols, &right_cols) {
                    here.push((l, r));
                } else {
                    keep.push((l, r));
                }
            }
            *remaining = keep;
            if here.is_empty() {
                // No condition connects the two subtrees here: this would be a
                // cross product the original plan did not have. Bail.
                return None;
            }
            let mut scope = left_cols;
            scope.extend(right_cols);
            let node = LogicalPlan::Join {
                left: Box::new(left_plan),
                right: Box::new(right_plan),
                join_type: JoinType::Inner,
                on: here,
                filter: None,
            };
            Some((node, scope))
        }
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
/// a valid plan) returns an empty set, which makes [`rebuild_from_shape`] bail
/// out conservatively (an empty scope places no equi-pair, so a join level
/// finds no condition and the rewrite aborts to a no-op).
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

    // ----- StatsEstimator bridge tests -------------------------------------
    //
    // These exercise the cost-based path end-to-end: a `StatsProvider` of
    // base-table row counts, bridged through `StatsEstimator` into the pass.

    use crate::plan::statistics::{StatsProvider, TableStats};

    /// In-memory `StatsProvider` keyed by table name → row count.
    #[derive(Default)]
    struct MockStats(std::collections::HashMap<String, usize>);

    impl MockStats {
        fn with(mut self, name: &str, rows: usize) -> Self {
            self.0.insert(name.to_string(), rows);
            self
        }
    }

    impl StatsProvider for MockStats {
        fn table_stats(&self, name: &str) -> Option<TableStats> {
            self.0.get(name).map(|&n| TableStats::new(n))
        }
    }

    /// Walk a left-deep join tree collecting the leaf scans' table names in
    /// the order they appear along the spine (deepest-left first, then each
    /// right input from the bottom up).
    fn leaf_tables_in_order(plan: &LogicalPlan) -> Vec<String> {
        fn go(plan: &LogicalPlan, out: &mut Vec<String>) {
            match plan {
                LogicalPlan::Join { left, right, .. } => {
                    go(left, out);
                    go(right, out);
                }
                LogicalPlan::Scan { table, .. } => out.push(table.clone()),
                _ => {}
            }
        }
        let mut out = Vec::new();
        go(plan, &mut out);
        out
    }

    #[test]
    fn stats_estimator_reorders_three_way_smallest_first() {
        // a=1000, b=10, c=5. Original spine order is [a, b, c]; costing every
        // leaf via the StatsEstimator must reorder smallest-first. The
        // equi-pairs are a.k=b.k2 and b.m=c.m2, so the only valid
        // smallest-first rebuild that still places every pair is [c, b, a]:
        // c-b connects on m/m2, then a connects on k/k2.
        let stats = MockStats::default()
            .with("a", 1000)
            .with("b", 10)
            .with("c", 5);
        let est = Arc::new(StatsEstimator::new(stats));
        let pass = JoinReorder::with_estimator(est);

        let plan = three_way();
        let out = pass.rewrite(plan).expect("reorder");

        // Leaves now ordered smallest-first along the rebuilt spine.
        let order = leaf_tables_in_order(&out);
        assert_eq!(
            order,
            vec!["c".to_string(), "b".to_string(), "a".to_string()],
            "leaves must be reordered smallest-row-count first"
        );
    }

    #[test]
    fn stats_estimator_noop_without_stats() {
        // Empty provider: every leaf estimate is None, so the pass must leave
        // the chain byte-for-byte unchanged (the conservative default).
        let est = Arc::new(StatsEstimator::new(MockStats::default()));
        let pass = JoinReorder::with_estimator(est);

        let plan = three_way();
        let before = format!("{:?}", plan);
        let out = pass.rewrite(plan).expect("noop");
        assert_eq!(
            before,
            format!("{:?}", out),
            "missing stats must leave the join order unchanged"
        );
    }

    #[test]
    fn stats_estimator_noop_when_one_leaf_unknown() {
        // 'b' has no stats entry → its leaf can't be costed → the whole chain
        // stays conservative even though a and c are known.
        let stats = MockStats::default().with("a", 1000).with("c", 5);
        let est = Arc::new(StatsEstimator::new(stats));
        let pass = JoinReorder::with_estimator(est);

        let plan = three_way();
        let before = format!("{:?}", plan);
        let out = pass.rewrite(plan).expect("noop");
        assert_eq!(
            before,
            format!("{:?}", out),
            "a partially-costed chain must not be reordered"
        );
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

    /// Collect every join's `on` pairs across a whole tree, rendered as a
    /// sorted multiset of `"l=r"` strings (orientation-normalised so `a=b` and
    /// `b=a` compare equal). Used to prove no predicate is dropped, duplicated,
    /// or invented by a reorder.
    fn collect_on_pairs(plan: &LogicalPlan) -> Vec<String> {
        fn pair_key(l: &Expr, r: &Expr) -> String {
            let a = format!("{l:?}");
            let b = format!("{r:?}");
            // Normalise orientation so a=b and b=a hash the same.
            if a <= b {
                format!("{a}={b}")
            } else {
                format!("{b}={a}")
            }
        }
        fn go(plan: &LogicalPlan, out: &mut Vec<String>) {
            if let LogicalPlan::Join { left, right, on, filter, .. } = plan {
                assert!(filter.is_none(), "reorder must not introduce a residual filter");
                for (l, r) in on {
                    out.push(pair_key(l, r));
                }
                go(left, out);
                go(right, out);
            }
        }
        let mut out = Vec::new();
        go(plan, &mut out);
        out.sort();
        out
    }

    #[test]
    fn reorder_preserves_all_join_predicates() {
        // Every original equi-pair must survive the reorder exactly once — no
        // dropped ON condition, no invented cross product.
        let stats = MockStats::default()
            .with("a", 1000)
            .with("b", 10)
            .with("c", 5);
        let est = Arc::new(StatsEstimator::new(stats));
        let pass = JoinReorder::with_estimator(est);

        let plan = three_way();
        let before_pairs = collect_on_pairs(&plan);
        let out = pass.rewrite(plan).expect("reorder");
        let after_pairs = collect_on_pairs(&out);

        assert_eq!(
            before_pairs, after_pairs,
            "the set of equi-join predicates must be identical after reordering"
        );
        // And concretely: both original keys are still present.
        assert_eq!(after_pairs.len(), 2, "exactly the two original ON pairs remain");
    }

    /// Build a 4-leaf chain a-b-c-d connected as two cheap pairs (a-b, c-d)
    /// bridged by a single b-c key, where a bushy `(a⋈b) ⋈ (c⋈d)` plan is
    /// cheapest. Schema: a(ka), b(kb1,kb2), c(kc1,kc2), d(kd).
    fn four_way_bushy() -> LogicalPlan {
        let a = leaf("a", "ka");
        let b = LogicalPlan::Scan {
            table: "b".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("kb1", DataType::Int64, false),
                Field::new("kb2", DataType::Int64, false),
            ]),
        };
        let c = LogicalPlan::Scan {
            table: "c".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("kc1", DataType::Int64, false),
                Field::new("kc2", DataType::Int64, false),
            ]),
        };
        let d = leaf("d", "kd");
        // Original left-deep order: ((a JOIN b) JOIN c) JOIN d
        let ab = LogicalPlan::Join {
            left: Box::new(a),
            right: Box::new(b),
            join_type: JoinType::Inner,
            on: vec![(col("ka"), col("kb1"))],
            filter: None,
        };
        let abc = LogicalPlan::Join {
            left: Box::new(ab),
            right: Box::new(c),
            join_type: JoinType::Inner,
            on: vec![(col("kb2"), col("kc1"))],
            filter: None,
        };
        LogicalPlan::Join {
            left: Box::new(abc),
            right: Box::new(d),
            join_type: JoinType::Inner,
            on: vec![(col("kc2"), col("kd"))],
            filter: None,
        }
    }

    /// Count the leaf scans on the deepest-left chain of a (possibly bushy)
    /// tree — helps assert structural validity without over-constraining shape.
    fn total_leaves(plan: &LogicalPlan) -> usize {
        match plan {
            LogicalPlan::Join { left, right, .. } => total_leaves(left) + total_leaves(right),
            _ => 1,
        }
    }

    #[test]
    fn four_way_reorder_is_valid_and_predicate_preserving() {
        // A 4-way INNER equi-chain reorders into *some* cheapest cross-product-
        // free tree. We don't pin the exact shape (the containment cost model's
        // optimum on a path is implementation-defined up to ties), but the
        // rewrite MUST: (a) cover all four leaves, (b) preserve every ON pair
        // exactly once, (c) introduce no residual filter, and (d) keep the
        // output column set. This is the core semantics-preservation contract.
        let stats = MockStats::default()
            .with("a", 1000)
            .with("b", 5)
            .with("c", 5)
            .with("d", 1000);
        let est = Arc::new(StatsEstimator::new(stats));
        let pass = JoinReorder::with_estimator(est);

        let plan = four_way_bushy();
        let before_pairs = collect_on_pairs(&plan);
        let before = plan.schema().expect("typecheck");
        let out = pass.rewrite(plan).expect("reorder");

        // (a) all four leaves present.
        assert_eq!(total_leaves(&out), 4, "all four input relations must survive");
        // (b) + (c) every predicate preserved, no residual filter.
        assert_eq!(
            before_pairs,
            collect_on_pairs(&out),
            "all three ON pairs must be preserved exactly"
        );
        // (d) output column set preserved.
        let after = out.schema().expect("typecheck after");
        let bset: HashSet<_> = before.fields.iter().map(|f| f.name.clone()).collect();
        let aset: HashSet<_> = after.fields.iter().map(|f| f.name.clone()).collect();
        assert_eq!(bset, aset);
    }

    #[test]
    fn four_way_sinks_a_small_relation_to_a_build_side() {
        // With a=1000, b=5, c=6, d=1000 over the path a-b-c-d, the cheapest
        // plan threads the two large relations around the small b-c core. We
        // prove the cost model steers the order by checking that *some* small
        // relation (b or c) lands at a deepest-left (build) position — i.e. it
        // is not the case that every build leaf is a 1000-row table.
        let stats = MockStats::default()
            .with("a", 1000)
            .with("b", 5)
            .with("c", 6)
            .with("d", 1000);
        let est = Arc::new(StatsEstimator::new(stats));
        let pass = JoinReorder::with_estimator(est);

        let out = pass.rewrite(four_way_bushy()).expect("reorder");

        // Collect every deepest-left leaf reachable by descending `left` from
        // each Join node (one per join level).
        fn build_side_leaves<'a>(plan: &'a LogicalPlan, out: &mut Vec<&'a str>) {
            if let LogicalPlan::Join { left, right, .. } = plan {
                // The deepest-left scan under this node is a build leaf.
                let mut cur = &**left;
                while let LogicalPlan::Join { left: l, .. } = cur {
                    cur = l;
                }
                if let LogicalPlan::Scan { table, .. } = cur {
                    out.push(table.as_str());
                }
                build_side_leaves(left, out);
                build_side_leaves(right, out);
            }
        }
        let mut builds = Vec::new();
        build_side_leaves(&out, &mut builds);
        assert!(
            builds.iter().any(|t| *t == "b" || *t == "c"),
            "a small relation must occupy a build-side leaf, got builds={builds:?}"
        );
    }

    #[test]
    fn cross_product_chain_is_noop() {
        // Three leaves but only a-b is connected; c is unjoined (a cartesian
        // product the planner would never have produced from a pure equi
        // chain). The model reports a disconnected component, so the pass must
        // leave the plan untouched rather than invent a cross product.
        //
        // We build this by hand as a chain whose second join's `on` references
        // only columns already in the left subtree — which `leaf_for_expr`
        // cannot resolve to two distinct leaves, forcing the no-op.
        let a = leaf("a", "k");
        let b = leaf("b", "k2");
        let c = leaf("c", "k3");
        let ab = LogicalPlan::Join {
            left: Box::new(a),
            right: Box::new(b),
            join_type: JoinType::Inner,
            on: vec![(col("k"), col("k2"))],
            filter: None,
        };
        // c is joined on (k = k2): both columns live in the {a,b} subtree, so
        // there is no edge to leaf c — disconnected.
        let abc = LogicalPlan::Join {
            left: Box::new(ab),
            right: Box::new(c),
            join_type: JoinType::Inner,
            on: vec![(col("k"), col("k2"))],
            filter: None,
        };
        let stats = MockStats::default()
            .with("a", 1000)
            .with("b", 10)
            .with("c", 5);
        let pass = JoinReorder::with_estimator(Arc::new(StatsEstimator::new(stats)));
        let before = format!("{:?}", abc);
        let out = pass.rewrite(abc).expect("noop");
        assert_eq!(
            before,
            format!("{:?}", out),
            "a disconnected (cross-product) chain must not be reordered"
        );
    }
}
