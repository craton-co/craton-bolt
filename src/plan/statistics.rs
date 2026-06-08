// SPDX-License-Identifier: Apache-2.0

//! Cardinality / statistics estimation for the cost-based optimizer.
//!
//! This module provides the *standalone* statistics surface the cost-based
//! optimizer (CBO) needs to estimate the output cardinality of a
//! [`LogicalPlan`]. It is intentionally self-contained: it depends only on
//! the logical-plan AST ([`LogicalPlan`], [`Expr`], etc.) and a caller-
//! supplied [`StatsProvider`] for base-table statistics. It does **not**
//! depend on the logical optimizer module (`src/plan/optimizer/`) — that
//! module's `RowEstimator` trait lives on a separate track and an
//! orchestrator bridges the two via [`StatsRowEstimator`].
//!
//! # Estimation model
//!
//! The estimator walks the plan bottom-up and assigns each node an estimated
//! output row count. The base case is [`LogicalPlan::Scan`], whose row count
//! comes straight from the [`StatsProvider`]. Every other node applies a
//! textbook selectivity / cardinality rule on top of its input estimate(s):
//!
//! | Node        | Rule                                                       |
//! |-------------|------------------------------------------------------------|
//! | `Scan`      | base table `row_count`                                      |
//! | `Filter`    | input × predicate selectivity (see [`estimate_selectivity`]) |
//! | `Project`   | passthrough (projection never changes row count)           |
//! | `Aggregate` | NDV of the group keys (or `sqrt(rows)` heuristic); 1 for a global aggregate |
//! | `Join`      | `|L|·|R| / max(ndv_l, ndv_r)` for equi, `|L|·|R|` for cross |
//! | `Limit`     | `min(limit, input)`                                         |
//! | `Distinct`  | NDV estimate of the input (product of group-key NDVs, capped) |
//! | `Union`     | sum of branch estimates                                     |
//! | `Sort`      | passthrough                                                 |
//!
//! All arithmetic is performed in `f64` and rounded / clamped back to
//! `usize` at node boundaries so a long chain of multiplicative selectivities
//! does not silently truncate to zero. An estimate of zero rows is clamped to
//! one (the optimizer reasons about *relative* sizes; a literal-zero estimate
//! would make every downstream product collapse).
//!
//! # Missing statistics
//!
//! Estimation is best-effort. When a base table has no entry in the
//! [`StatsProvider`], [`estimate_rows`] returns `None` and the absence
//! propagates up the tree — callers treat `None` as "no estimate available"
//! and fall back to their own heuristics.

use std::collections::HashMap;

use crate::plan::logical_plan::{BinaryOp, Expr, LogicalPlan, UnaryOp};

/// Default selectivity applied to a single equality conjunct (`col = const`)
/// when no NDV statistic is available to refine it. Mirrors the canonical
/// "1/10" textbook constant used by most cost-based optimizers.
pub const DEFAULT_EQ_SELECTIVITY: f64 = 0.1;

/// Default selectivity applied to a single range conjunct (`<`, `<=`, `>`,
/// `>=`, `BETWEEN`, `LIKE`) when min/max histograms are unavailable.
pub const DEFAULT_RANGE_SELECTIVITY: f64 = 0.3;

/// Fallback selectivity for any predicate shape the estimator does not model
/// explicitly (e.g. `IS NULL`, arbitrary boolean expressions). Deliberately
/// mid-range so an unmodelled predicate neither vanishes nor passes through
/// unchanged.
pub const DEFAULT_OTHER_SELECTIVITY: f64 = 0.25;

/// A simple scalar value used for column min/max bounds.
///
/// Deliberately a small, self-contained enum rather than a reuse of
/// [`Literal`]: statistics only ever need ordered scalar bounds, and keeping
/// the type local means the `StatsProvider` surface does not leak the much
/// larger expression-literal vocabulary (decimals, timestamps with interned
/// timezones, etc.). Callers building stats from those richer types lower
/// them into one of these variants.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    /// Boolean bound.
    Bool(bool),
    /// 64-bit signed integer bound (Int32 columns widen into this).
    Int(i64),
    /// 64-bit float bound (Float32 columns widen into this).
    Float(f64),
    /// UTF-8 string bound.
    Str(String),
}

/// Per-column statistics for a base table.
#[derive(Debug, Clone, Default)]
pub struct ColumnStats {
    /// Number of NULL values in the column.
    pub null_count: usize,
    /// Number of distinct (non-NULL) values, if known. `None` means the
    /// estimator falls back to the default selectivity constants for this
    /// column.
    pub ndv: Option<usize>,
    /// Minimum value, if known.
    pub min: Option<ScalarValue>,
    /// Maximum value, if known.
    pub max: Option<ScalarValue>,
}

impl ColumnStats {
    /// Construct column stats carrying only an NDV (the most common case for
    /// driving join / equality selectivity). All other fields default.
    pub fn with_ndv(ndv: usize) -> Self {
        Self {
            null_count: 0,
            ndv: Some(ndv),
            min: None,
            max: None,
        }
    }
}

/// Statistics for a single base table: a total row count plus optional
/// per-column detail keyed by column name.
#[derive(Debug, Clone, Default)]
pub struct TableStats {
    /// Total number of rows in the table.
    pub row_count: usize,
    /// Per-column statistics, keyed by column name. Columns absent from the
    /// map simply have no detailed stats (the estimator uses defaults).
    pub per_column: HashMap<String, ColumnStats>,
}

impl TableStats {
    /// Construct table stats with a row count and no per-column detail.
    pub fn new(row_count: usize) -> Self {
        Self {
            row_count,
            per_column: HashMap::new(),
        }
    }

    /// Builder-style insert of a column's stats. Returns `self` for chaining.
    pub fn with_column(mut self, name: impl Into<String>, stats: ColumnStats) -> Self {
        self.per_column.insert(name.into(), stats);
        self
    }

    /// NDV for `column`, if both the column and its NDV statistic are known.
    pub fn ndv(&self, column: &str) -> Option<usize> {
        self.per_column.get(column).and_then(|c| c.ndv)
    }
}

/// Source of base-table statistics.
///
/// The optimizer (and this module's [`estimate_rows`]) is parameterised over
/// this trait so the actual provenance of statistics — a catalog, sampled
/// scan, hard-coded test fixture — is decoupled from the estimation logic.
pub trait StatsProvider {
    /// Statistics for the table named `name`, or `None` if the provider has
    /// no entry for it.
    fn table_stats(&self, name: &str) -> Option<TableStats>;
}

/// Recursively estimate the output cardinality of `plan`.
///
/// Returns `None` when the estimate cannot be computed because a base table
/// referenced by the plan has no entry in `stats`. A returned `Some(n)` is
/// always `>= 1` (see the module docs on zero-clamping).
///
/// See the module-level documentation for the per-node estimation rules.
pub fn estimate_rows(plan: &LogicalPlan, stats: &dyn StatsProvider) -> Option<usize> {
    estimate_rows_f64(plan, stats).map(clamp_rows)
}

/// Inner recursion returning a continuous `f64` estimate so multiplicative
/// selectivity chains do not lose precision to intermediate rounding. The
/// public [`estimate_rows`] clamps the final value back to `usize`.
fn estimate_rows_f64(plan: &LogicalPlan, stats: &dyn StatsProvider) -> Option<f64> {
    match plan {
        LogicalPlan::Window { input, .. } => estimate_rows_f64(input, stats),
        LogicalPlan::Scan { table, .. } => {
            let ts = stats.table_stats(table)?;
            Some(ts.row_count as f64)
        }
        LogicalPlan::Filter { input, predicate } => {
            let rows = estimate_rows_f64(input, stats)?;
            let sel = estimate_selectivity(predicate, input, stats);
            Some((rows * sel).max(1.0))
        }
        // Projection and Sort never change the row count.
        LogicalPlan::Project { input, .. } | LogicalPlan::Sort { input, .. } => {
            estimate_rows_f64(input, stats)
        }
        LogicalPlan::Aggregate {
            input, group_by, ..
        } => {
            let rows = estimate_rows_f64(input, stats)?;
            // A global aggregate (no GROUP BY) always emits exactly one row.
            if group_by.is_empty() {
                return Some(1.0);
            }
            Some(estimate_group_count(group_by, input, stats, rows))
        }
        LogicalPlan::Distinct { input } => {
            let rows = estimate_rows_f64(input, stats)?;
            // DISTINCT over the whole row shape: estimate the number of
            // distinct rows. We reuse the group-count heuristic treating the
            // input's columns as the grouping keys is overkill (we don't have
            // the projected expr list here), so fall back to the sqrt(rows)
            // heuristic, which is the same shape the aggregate path uses when
            // no per-key NDV is available.
            Some(distinct_rows_heuristic(rows))
        }
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            let rows = estimate_rows_f64(input, stats)?;
            // OFFSET first discards `offset` rows, then LIMIT caps the rest.
            let after_offset = (rows - *offset as f64).max(0.0);
            Some(after_offset.min(*limit as f64))
        }
        LogicalPlan::Union { inputs } => {
            // UNION ALL concatenates: the row count is the sum of branches.
            // If any branch has no estimate the whole union is unknown.
            let mut total = 0.0;
            for branch in inputs {
                total += estimate_rows_f64(branch, stats)?;
            }
            Some(total)
        }
        LogicalPlan::SetOp {
            left,
            right,
            op,
            all,
        } => {
            use crate::plan::logical_plan::SetOpKind;
            let l = estimate_rows_f64(left, stats)?;
            let r = estimate_rows_f64(right, stats)?;
            // Coarse cardinality heuristics for the set operators:
            //   * EXCEPT       — at most |L|; assume roughly half survive.
            //   * EXCEPT ALL   — at most |L|; same upper bound, kept as |L|
            //     since multiplicities are only reduced, not eliminated.
            //   * INTERSECT(/ALL) — at most min(|L|, |R|).
            // These are deliberately simple (no per-column NDV modelling);
            // they only need to be monotone and bounded for the cost model.
            let est = match (op, all) {
                (SetOpKind::Except, false) => (l * 0.5).max(1.0),
                (SetOpKind::Except, true) => l,
                (SetOpKind::Intersect, _) => l.min(r),
            };
            Some(est.max(1.0))
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            ..
        } => {
            let l = estimate_rows_f64(left, stats)?;
            let r = estimate_rows_f64(right, stats)?;
            Some(estimate_join_rows(l, r, on, left, right, *join_type, stats))
        }
    }
}

/// Estimate the number of output groups for a GROUP BY over `group_by` keys.
///
/// When every key is a bare column with a known NDV, the group count is the
/// product of the per-key NDVs (capped at the input row count — you cannot
/// have more groups than input rows). When any key lacks an NDV, fall back to
/// the `sqrt(rows)` heuristic, a standard rule-of-thumb for an unknown number
/// of groups.
fn estimate_group_count(
    group_by: &[Expr],
    input: &LogicalPlan,
    stats: &dyn StatsProvider,
    input_rows: f64,
) -> f64 {
    let mut product: f64 = 1.0;
    let mut all_known = true;
    for key in group_by {
        match column_ndv(key, input, stats) {
            Some(ndv) => product *= ndv as f64,
            None => {
                all_known = false;
                break;
            }
        }
    }
    let estimate = if all_known {
        product
    } else {
        distinct_rows_heuristic(input_rows)
    };
    // Can never exceed the number of input rows, and is at least one group.
    estimate.clamp(1.0, input_rows.max(1.0))
}

/// `sqrt(rows)` distinct-count heuristic, floored at one row.
fn distinct_rows_heuristic(rows: f64) -> f64 {
    rows.max(1.0).sqrt().max(1.0)
}

/// Estimate the output cardinality of a join.
///
/// * `Cross` (and any join with no equi `on` pairs) → cartesian product
///   `|L| · |R|`.
/// * Equi-join → standard `|L| · |R| / max(ndv_l, ndv_r)` over the first
///   equi-key pair, where `ndv_l` / `ndv_r` are the distinct-value counts of
///   the left / right key columns. When neither side has an NDV we fall back
///   to treating the larger input's size as the join-key cardinality (the
///   "containment"-style assumption that the smaller side's keys are a subset
///   of the larger side's).
///
/// Multi-predicate equi-joins multiply the additional pairs' selectivities so
/// each extra equality further reduces the estimate.
fn estimate_join_rows(
    l: f64,
    r: f64,
    on: &[(Expr, Expr)],
    left: &LogicalPlan,
    right: &LogicalPlan,
    join_type: crate::plan::logical_plan::JoinType,
    stats: &dyn StatsProvider,
) -> f64 {
    use crate::plan::logical_plan::JoinType;

    // Cross join or a join with no equi pairs: cartesian product.
    if matches!(join_type, JoinType::Cross) || on.is_empty() {
        return (l * r).max(1.0);
    }

    // Equi-join. The classic estimate divides the cartesian product by the
    // join-key cardinality. With multiple equality conjuncts each additional
    // pair multiplies in another `1 / max(ndv)` selectivity factor.
    let mut result = l * r;
    for (left_key, right_key) in on {
        let ndv_l = column_ndv(left_key, left, stats);
        let ndv_r = column_ndv(right_key, right, stats);
        let denom = join_key_cardinality(ndv_l, ndv_r, l, r);
        result /= denom.max(1.0);
    }
    result.max(1.0)
}

/// Estimate the output cardinality of an INNER equi-join given the two input
/// cardinalities, the equi-`on` pairs, and the two input subtrees (used to
/// resolve per-column NDVs through [`column_ndv`]).
///
/// This is the public entry point the cost-based join enumerator
/// ([`crate::plan::optimizer::cost`]) calls when combining two subplans: it
/// shares the exact textbook formula used by [`estimate_join_rows`]
/// (`|L|·|R| / ∏ max(ndv_l, ndv_r)`), so the enumeration cost model and the
/// whole-plan estimator never diverge.
///
/// When `on` is empty this returns the cartesian product `|L|·|R|` — a caller
/// that wants to forbid cross products should check connectivity itself.
/// NDV resolution degrades gracefully: a key whose column has no NDV stat
/// falls back to the larger input's size as the key cardinality (the
/// containment assumption), which never inflates the estimate above the
/// cartesian product.
pub fn estimate_equijoin_rows(
    left_rows: u64,
    right_rows: u64,
    on: &[(Expr, Expr)],
    left: &LogicalPlan,
    right: &LogicalPlan,
    stats: &dyn StatsProvider,
) -> u64 {
    let l = left_rows as f64;
    let r = right_rows as f64;
    if on.is_empty() {
        return clamp_rows((l * r).max(1.0)) as u64;
    }
    let mut result = l * r;
    for (left_key, right_key) in on {
        let ndv_l = column_ndv(left_key, left, stats);
        let ndv_r = column_ndv(right_key, right, stats);
        let denom = join_key_cardinality(ndv_l, ndv_r, l, r);
        result /= denom.max(1.0);
    }
    clamp_rows(result.max(1.0)) as u64
}

/// Pick the divisor for an equi-join key pair from the two sides' NDVs.
///
/// Standard rule: divide by `max(ndv_l, ndv_r)`. When only one side has an
/// NDV, use it. When neither does, assume the join key ranges over the larger
/// input (`max(|L|, |R|)`), the most conservative non-explosive default.
fn join_key_cardinality(ndv_l: Option<usize>, ndv_r: Option<usize>, l: f64, r: f64) -> f64 {
    match (ndv_l, ndv_r) {
        (Some(a), Some(b)) => a.max(b) as f64,
        (Some(a), None) => a as f64,
        (None, Some(b)) => b as f64,
        (None, None) => l.max(r),
    }
}

/// Estimate the selectivity (a fraction in `[0, 1]`) of a boolean predicate
/// evaluated against `input`.
///
/// Rules:
/// * Equality `col = const` → `1 / ndv` when the column NDV is known,
///   otherwise [`DEFAULT_EQ_SELECTIVITY`].
/// * Range comparison (`<`, `<=`, `>`, `>=`) and `LIKE` →
///   [`DEFAULT_RANGE_SELECTIVITY`].
/// * `a AND b` → `sel(a) · sel(b)` (independence assumption).
/// * `a OR b` → `sel(a) + sel(b) − sel(a)·sel(b)`, capped at 1.0
///   (inclusion–exclusion under independence).
/// * `NOT a` → `1 − sel(a)`.
/// * Anything else → [`DEFAULT_OTHER_SELECTIVITY`].
pub fn estimate_selectivity(
    predicate: &Expr,
    input: &LogicalPlan,
    stats: &dyn StatsProvider,
) -> f64 {
    match predicate {
        Expr::Binary { op, left, right } => match op {
            BinaryOp::And => {
                let a = estimate_selectivity(left, input, stats);
                let b = estimate_selectivity(right, input, stats);
                (a * b).clamp(0.0, 1.0)
            }
            BinaryOp::Or => {
                let a = estimate_selectivity(left, input, stats);
                let b = estimate_selectivity(right, input, stats);
                // Inclusion-exclusion under independence; capped at 1.0.
                (a + b - a * b).clamp(0.0, 1.0)
            }
            BinaryOp::Eq => eq_selectivity(left, right, input, stats),
            BinaryOp::NotEq => {
                // `col <> const` is the complement of equality.
                1.0 - eq_selectivity(left, right, input, stats)
            }
            BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq => {
                DEFAULT_RANGE_SELECTIVITY
            }
            // Arithmetic / string-concat operators are not boolean predicates;
            // a malformed plan that uses one here gets the generic fallback.
            _ => DEFAULT_OTHER_SELECTIVITY,
        },
        Expr::Unary { op, operand } => match op {
            UnaryOp::Not => 1.0 - estimate_selectivity(operand, input, stats),
            // IS NULL / IS NOT NULL: no histogram, use the generic default.
            UnaryOp::IsNull | UnaryOp::IsNotNull => DEFAULT_OTHER_SELECTIVITY,
        },
        // `col LIKE 'pattern'` behaves like a range scan for estimation.
        Expr::Like { .. } => DEFAULT_RANGE_SELECTIVITY,
        // Unwrap aliases transparently.
        Expr::Alias(inner, _) => estimate_selectivity(inner, input, stats),
        _ => DEFAULT_OTHER_SELECTIVITY,
    }
}

/// Selectivity of an `=` comparison. Refined to `1 / ndv` when exactly one
/// side is a column with a known NDV and the other is a constant; otherwise
/// the default equality constant.
fn eq_selectivity(
    left: &Expr,
    right: &Expr,
    input: &LogicalPlan,
    stats: &dyn StatsProvider,
) -> f64 {
    // `col = const` (either ordering) refines via the column's NDV.
    let col_side = if is_constant(right) {
        Some(left)
    } else if is_constant(left) {
        Some(right)
    } else {
        // `col = col` (e.g. a residual equi-join predicate): use the
        // smaller-NDV side, mirroring the join formula's intuition. Fall back
        // to the default if neither column has an NDV.
        return col_eq_col_selectivity(left, right, input, stats);
    };

    if let Some(col) = col_side {
        if let Some(ndv) = column_ndv(col, input, stats) {
            if ndv > 0 {
                return 1.0 / ndv as f64;
            }
        }
    }
    DEFAULT_EQ_SELECTIVITY
}

/// Selectivity of `col_a = col_b` where both sides are columns: `1 / max(ndv)`
/// when at least one NDV is known, else the default equality constant.
fn col_eq_col_selectivity(
    left: &Expr,
    right: &Expr,
    input: &LogicalPlan,
    stats: &dyn StatsProvider,
) -> f64 {
    let a = column_ndv(left, input, stats);
    let b = column_ndv(right, input, stats);
    match (a, b) {
        (Some(x), Some(y)) => 1.0 / x.max(y).max(1) as f64,
        (Some(x), None) | (None, Some(x)) => 1.0 / x.max(1) as f64,
        (None, None) => DEFAULT_EQ_SELECTIVITY,
    }
}

/// Resolve the NDV for an expression that is (or wraps) a bare column
/// reference, by locating the base [`LogicalPlan::Scan`] that supplies the
/// column and looking the column up in its table stats.
///
/// Returns `None` for non-column expressions, for columns whose source table
/// has no stats, or for columns with no NDV recorded.
fn column_ndv(expr: &Expr, input: &LogicalPlan, stats: &dyn StatsProvider) -> Option<usize> {
    let name = column_name(expr)?;
    scan_ndv_for_column(input, name, stats)
}

/// Search the plan subtree `plan` for a base `Scan` whose table stats record
/// an NDV for `column`, returning the first match found in a pre-order walk.
///
/// This is a deliberately simple resolution strategy: it does not track
/// column renames through projections or join disambiguation. It is good
/// enough to drive selectivity for the common shape — a filter / aggregate
/// directly over (a chain of row-preserving wrappers above) a scan — which is
/// what the cost-based optimizer cares about most. Unknown columns simply
/// yield `None` and the estimator falls back to its defaults.
fn scan_ndv_for_column(
    plan: &LogicalPlan,
    column: &str,
    stats: &dyn StatsProvider,
) -> Option<usize> {
    match plan {
        LogicalPlan::Window { input, .. } => scan_ndv_for_column(input, column, stats),
        LogicalPlan::Scan { table, .. } => stats.table_stats(table)?.ndv(column),
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Project { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Distinct { input }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. } => scan_ndv_for_column(input, column, stats),
        LogicalPlan::Union { inputs } => inputs
            .iter()
            .find_map(|b| scan_ndv_for_column(b, column, stats)),
        LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => {
            scan_ndv_for_column(left, column, stats)
                .or_else(|| scan_ndv_for_column(right, column, stats))
        }
    }
}

/// Extract the column name from an expression that is a bare column reference
/// (optionally wrapped in an `Alias`). Returns `None` for anything else.
fn column_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(name) => Some(name.as_str()),
        Expr::Alias(inner, _) => column_name(inner),
        _ => None,
    }
}

/// True if `expr` is a constant scalar (a literal, possibly aliased).
fn is_constant(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(_) => true,
        Expr::Alias(inner, _) => is_constant(inner),
        _ => false,
    }
}

/// Clamp a continuous row estimate to a non-negative `usize`, flooring at one
/// row (see the module docs on zero-clamping).
fn clamp_rows(rows: f64) -> usize {
    if !rows.is_finite() || rows < 1.0 {
        return 1;
    }
    // Round to nearest; saturating cast guards against absurd cartesian
    // products overflowing `usize`.
    let rounded = rows.round();
    if rounded >= usize::MAX as f64 {
        usize::MAX
    } else {
        rounded as usize
    }
}

/// A `RowEstimator`-style adapter over a [`StatsProvider`].
///
/// The logical optimizer's join-reorder pass needs to sort join leaves
/// smallest-first, for which it asks "how many rows does this leaf produce?".
/// This adapter answers that question from base-table statistics without the
/// optimizer module having to know about [`estimate_rows`] directly — the
/// orchestrator constructs a `StatsRowEstimator` and bridges it to whatever
/// `RowEstimator` trait the optimizer defines.
///
/// It is a thin newtype around a `&dyn StatsProvider` so it is cheap to
/// construct per pass and borrows (rather than owns) the provider.
pub struct StatsRowEstimator<'a>(pub &'a dyn StatsProvider);

impl<'a> StatsRowEstimator<'a> {
    /// Construct an estimator backed by `provider`.
    pub fn new(provider: &'a dyn StatsProvider) -> Self {
        StatsRowEstimator(provider)
    }

    /// Estimate the output row count of `plan`, intended for the join-reorder
    /// pass to order leaves smallest-first. Returns `None` when no estimate
    /// is available (the caller should leave that leaf's order undecided or
    /// fall back to a default).
    ///
    /// This is the same computation as [`estimate_rows`]; the method exists so
    /// the adapter presents the exact shape the optimizer's `RowEstimator`
    /// trait expects.
    pub fn estimate_leaf_rows(&self, plan: &LogicalPlan) -> Option<usize> {
        estimate_rows(plan, self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{
        col, lit, DataType, Expr, Field, JoinType, LogicalPlan, Schema,
    };

    /// A mock [`StatsProvider`] backed by an in-memory table → stats map.
    #[derive(Default)]
    struct MockStats {
        tables: HashMap<String, TableStats>,
    }

    impl MockStats {
        fn with(mut self, name: &str, stats: TableStats) -> Self {
            self.tables.insert(name.to_string(), stats);
            self
        }
    }

    impl StatsProvider for MockStats {
        fn table_stats(&self, name: &str) -> Option<TableStats> {
            self.tables.get(name).cloned()
        }
    }

    /// Build a trivial single-column scan over `table`.
    fn scan(table: &str, column: &str, dtype: DataType) -> LogicalPlan {
        LogicalPlan::Scan {
            table: table.to_string(),
            projection: None,
            schema: Schema::new(vec![Field::new(column, dtype, true)]),
        }
    }

    fn scan_two(table: &str, c0: &str, c1: &str) -> LogicalPlan {
        LogicalPlan::Scan {
            table: table.to_string(),
            projection: None,
            schema: Schema::new(vec![
                Field::new(c0, DataType::Int64, true),
                Field::new(c1, DataType::Int64, true),
            ]),
        }
    }

    #[test]
    fn scan_returns_table_row_count() {
        let stats = MockStats::default().with("t", TableStats::new(1_000));
        let plan = scan("t", "a", DataType::Int64);
        assert_eq!(estimate_rows(&plan, &stats), Some(1_000));
    }

    #[test]
    fn unknown_table_yields_none() {
        let stats = MockStats::default();
        let plan = scan("missing", "a", DataType::Int64);
        assert_eq!(estimate_rows(&plan, &stats), None);
    }

    #[test]
    fn filter_equality_uses_default_when_no_ndv() {
        let stats = MockStats::default().with("t", TableStats::new(1_000));
        // WHERE a = 5  →  1000 * 0.1 = 100
        let plan = LogicalPlan::Filter {
            input: Box::new(scan("t", "a", DataType::Int64)),
            predicate: col("a").eq(lit(5i64)),
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(100));
    }

    #[test]
    fn filter_equality_uses_ndv_when_available() {
        // ndv(a) = 4  →  selectivity 1/4  →  1000 * 0.25 = 250
        let stats = MockStats::default().with(
            "t",
            TableStats::new(1_000).with_column("a", ColumnStats::with_ndv(4)),
        );
        let plan = LogicalPlan::Filter {
            input: Box::new(scan("t", "a", DataType::Int64)),
            predicate: col("a").eq(lit(5i64)),
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(250));
    }

    #[test]
    fn filter_range_uses_range_selectivity() {
        let stats = MockStats::default().with("t", TableStats::new(1_000));
        // WHERE a > 5  →  1000 * 0.3 = 300
        let plan = LogicalPlan::Filter {
            input: Box::new(scan("t", "a", DataType::Int64)),
            predicate: col("a").gt(lit(5i64)),
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(300));
    }

    #[test]
    fn filter_and_multiplies_selectivities() {
        let stats = MockStats::default().with("t", TableStats::new(1_000));
        // (a = 5) AND (b > 3)  →  0.1 * 0.3 = 0.03  →  1000 * 0.03 = 30
        let pred = col("a").eq(lit(5i64)).and(col("b").gt(lit(3i64)));
        let plan = LogicalPlan::Filter {
            input: Box::new(scan_two("t", "a", "b")),
            predicate: pred,
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(30));
    }

    #[test]
    fn filter_or_combines_with_inclusion_exclusion() {
        let stats = MockStats::default().with("t", TableStats::new(1_000));
        // (a = 5) OR (b = 3)  →  0.1 + 0.1 - 0.01 = 0.19  →  190
        let pred = col("a").eq(lit(5i64)).or(col("b").eq(lit(3i64)));
        let plan = LogicalPlan::Filter {
            input: Box::new(scan_two("t", "a", "b")),
            predicate: pred,
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(190));
    }

    #[test]
    fn projection_is_passthrough() {
        let stats = MockStats::default().with("t", TableStats::new(777));
        let plan = LogicalPlan::Project {
            input: Box::new(scan("t", "a", DataType::Int64)),
            exprs: vec![col("a")],
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(777));
    }

    #[test]
    fn equi_join_divides_by_max_ndv() {
        // |L| = 1000 (ndv l.id = 100), |R| = 200 (ndv r.id = 50)
        // estimate = 1000 * 200 / max(100, 50) = 200000 / 100 = 2000
        let left_stats = TableStats::new(1_000).with_column("id", ColumnStats::with_ndv(100));
        let right_stats = TableStats::new(200).with_column("rid", ColumnStats::with_ndv(50));
        let stats = MockStats::default()
            .with("l", left_stats)
            .with("r", right_stats);

        let plan = LogicalPlan::Join {
            left: Box::new(scan("l", "id", DataType::Int64)),
            right: Box::new(scan("r", "rid", DataType::Int64)),
            join_type: JoinType::Inner,
            on: vec![(col("id"), col("rid"))],
            filter: None,
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(2_000));
    }

    #[test]
    fn cross_join_is_cartesian_product() {
        let stats = MockStats::default()
            .with("l", TableStats::new(30))
            .with("r", TableStats::new(20));
        let plan = LogicalPlan::Join {
            left: Box::new(scan("l", "a", DataType::Int64)),
            right: Box::new(scan("r", "b", DataType::Int64)),
            join_type: JoinType::Cross,
            on: vec![],
            filter: None,
        };
        // 30 * 20 = 600
        assert_eq!(estimate_rows(&plan, &stats), Some(600));
    }

    #[test]
    fn equi_join_without_ndv_uses_larger_side() {
        // No NDV anywhere: denom = max(|L|, |R|) = 1000.
        // estimate = 1000 * 200 / 1000 = 200
        let stats = MockStats::default()
            .with("l", TableStats::new(1_000))
            .with("r", TableStats::new(200));
        let plan = LogicalPlan::Join {
            left: Box::new(scan("l", "id", DataType::Int64)),
            right: Box::new(scan("r", "rid", DataType::Int64)),
            join_type: JoinType::Inner,
            on: vec![(col("id"), col("rid"))],
            filter: None,
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(200));
    }

    #[test]
    fn aggregate_group_count_uses_key_ndv() {
        // GROUP BY g, ndv(g) = 7  →  7 groups.
        let stats = MockStats::default().with(
            "t",
            TableStats::new(1_000).with_column("g", ColumnStats::with_ndv(7)),
        );
        let plan = LogicalPlan::Aggregate {
            input: Box::new(scan("t", "g", DataType::Int64)),
            group_by: vec![col("g")],
            aggregates: vec![],
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(7));
    }

    #[test]
    fn aggregate_group_count_caps_at_input_rows() {
        // ndv(g) = 5000 but only 1000 input rows → capped at 1000.
        let stats = MockStats::default().with(
            "t",
            TableStats::new(1_000).with_column("g", ColumnStats::with_ndv(5_000)),
        );
        let plan = LogicalPlan::Aggregate {
            input: Box::new(scan("t", "g", DataType::Int64)),
            group_by: vec![col("g")],
            aggregates: vec![],
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(1_000));
    }

    #[test]
    fn aggregate_group_count_sqrt_heuristic_without_ndv() {
        // No NDV → sqrt(900) = 30.
        let stats = MockStats::default().with("t", TableStats::new(900));
        let plan = LogicalPlan::Aggregate {
            input: Box::new(scan("t", "g", DataType::Int64)),
            group_by: vec![col("g")],
            aggregates: vec![],
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(30));
    }

    #[test]
    fn global_aggregate_is_single_row() {
        let stats = MockStats::default().with("t", TableStats::new(1_000));
        let plan = LogicalPlan::Aggregate {
            input: Box::new(scan("t", "a", DataType::Int64)),
            group_by: vec![],
            aggregates: vec![],
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(1));
    }

    #[test]
    fn multi_key_group_count_multiplies_ndvs() {
        // ndv(a)=4, ndv(b)=5 → 20 groups (< 1000 rows, no cap).
        let stats = MockStats::default().with(
            "t",
            TableStats::new(1_000)
                .with_column("a", ColumnStats::with_ndv(4))
                .with_column("b", ColumnStats::with_ndv(5)),
        );
        let plan = LogicalPlan::Aggregate {
            input: Box::new(scan_two("t", "a", "b")),
            group_by: vec![col("a"), col("b")],
            aggregates: vec![],
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(20));
    }

    #[test]
    fn limit_caps_input() {
        let stats = MockStats::default().with("t", TableStats::new(1_000));
        let plan = LogicalPlan::Limit {
            input: Box::new(scan("t", "a", DataType::Int64)),
            limit: 10,
            offset: 0,
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(10));
    }

    #[test]
    fn limit_does_not_inflate_small_input() {
        let stats = MockStats::default().with("t", TableStats::new(5));
        let plan = LogicalPlan::Limit {
            input: Box::new(scan("t", "a", DataType::Int64)),
            limit: 100,
            offset: 0,
        };
        // min(100, 5) = 5
        assert_eq!(estimate_rows(&plan, &stats), Some(5));
    }

    #[test]
    fn limit_with_offset_subtracts_first() {
        let stats = MockStats::default().with("t", TableStats::new(1_000));
        let plan = LogicalPlan::Limit {
            input: Box::new(scan("t", "a", DataType::Int64)),
            limit: 10,
            offset: 995,
        };
        // after_offset = 5, min(10, 5) = 5
        assert_eq!(estimate_rows(&plan, &stats), Some(5));
    }

    #[test]
    fn distinct_uses_sqrt_heuristic() {
        let stats = MockStats::default().with("t", TableStats::new(400));
        let plan = LogicalPlan::Distinct {
            input: Box::new(scan("t", "a", DataType::Int64)),
        };
        // sqrt(400) = 20
        assert_eq!(estimate_rows(&plan, &stats), Some(20));
    }

    #[test]
    fn union_sums_branches() {
        let stats = MockStats::default()
            .with("a", TableStats::new(100))
            .with("b", TableStats::new(250));
        let plan = LogicalPlan::Union {
            inputs: vec![
                scan("a", "x", DataType::Int64),
                scan("b", "x", DataType::Int64),
            ],
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(350));
    }

    #[test]
    fn union_missing_branch_yields_none() {
        let stats = MockStats::default().with("a", TableStats::new(100));
        let plan = LogicalPlan::Union {
            inputs: vec![
                scan("a", "x", DataType::Int64),
                scan("missing", "x", DataType::Int64),
            ],
        };
        assert_eq!(estimate_rows(&plan, &stats), None);
    }

    #[test]
    fn not_predicate_complements_selectivity() {
        let stats = MockStats::default().with("t", TableStats::new(1_000));
        // NOT (a = 5)  →  1 - 0.1 = 0.9  →  900
        let pred = Expr::Unary {
            op: UnaryOp::Not,
            operand: Box::new(col("a").eq(lit(5i64))),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(scan("t", "a", DataType::Int64)),
            predicate: pred,
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(900));
    }

    #[test]
    fn not_eq_complements_equality() {
        let stats = MockStats::default().with("t", TableStats::new(1_000));
        // a <> 5  →  1 - 0.1 = 0.9  →  900
        let plan = LogicalPlan::Filter {
            input: Box::new(scan("t", "a", DataType::Int64)),
            predicate: col("a").neq(lit(5i64)),
        };
        assert_eq!(estimate_rows(&plan, &stats), Some(900));
    }

    #[test]
    fn nested_filter_resolves_ndv_through_wrapper() {
        // Filter over a Project over a Scan: NDV resolution must walk past
        // the projection to reach the scan's stats.
        let stats = MockStats::default().with(
            "t",
            TableStats::new(1_000).with_column("a", ColumnStats::with_ndv(4)),
        );
        let inner = LogicalPlan::Project {
            input: Box::new(scan("t", "a", DataType::Int64)),
            exprs: vec![col("a")],
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(inner),
            predicate: col("a").eq(lit(5i64)),
        };
        // ndv 4 → 0.25 → 250
        assert_eq!(estimate_rows(&plan, &stats), Some(250));
    }

    #[test]
    fn row_estimator_adapter_matches_estimate_rows() {
        let stats = MockStats::default().with("t", TableStats::new(1_000));
        let plan = scan("t", "a", DataType::Int64);
        let est = StatsRowEstimator::new(&stats);
        assert_eq!(est.estimate_leaf_rows(&plan), estimate_rows(&plan, &stats));
        assert_eq!(est.estimate_leaf_rows(&plan), Some(1_000));
    }

    #[test]
    fn estimate_equijoin_rows_matches_join_node() {
        // The standalone equijoin entry point must agree with the whole-plan
        // Join estimate for the same inputs/keys.
        let left_stats = TableStats::new(1_000).with_column("id", ColumnStats::with_ndv(100));
        let right_stats = TableStats::new(200).with_column("rid", ColumnStats::with_ndv(50));
        let stats = MockStats::default()
            .with("l", left_stats)
            .with("r", right_stats);
        let left = scan("l", "id", DataType::Int64);
        let right = scan("r", "rid", DataType::Int64);
        let on = vec![(col("id"), col("rid"))];
        // 1000 * 200 / max(100, 50) = 2000.
        let direct = estimate_equijoin_rows(1_000, 200, &on, &left, &right, &stats);
        assert_eq!(direct, 2_000);

        // Same as building the Join node and calling estimate_rows.
        let join = LogicalPlan::Join {
            left: Box::new(left),
            right: Box::new(right),
            join_type: JoinType::Inner,
            on,
            filter: None,
        };
        assert_eq!(estimate_rows(&join, &stats), Some(direct as usize));
    }

    #[test]
    fn estimate_equijoin_rows_no_key_is_cartesian() {
        let stats = MockStats::default();
        let l = scan("l", "a", DataType::Int64);
        let r = scan("r", "b", DataType::Int64);
        // No keys: 30 * 20 = 600.
        assert_eq!(estimate_equijoin_rows(30, 20, &[], &l, &r, &stats), 600);
    }

    #[test]
    fn estimate_equijoin_rows_no_ndv_uses_containment() {
        let stats = MockStats::default();
        let l = scan("l", "id", DataType::Int64);
        let r = scan("r", "rid", DataType::Int64);
        let on = vec![(col("id"), col("rid"))];
        // No NDV anywhere: denom = max(|L|, |R|) = 1000 → 1000*200/1000 = 200.
        assert_eq!(estimate_equijoin_rows(1_000, 200, &on, &l, &r, &stats), 200);
    }

    #[test]
    fn scalar_value_ord_variants_construct() {
        // Smoke test that the ScalarValue surface is usable for min/max.
        let cs = ColumnStats {
            null_count: 3,
            ndv: Some(10),
            min: Some(ScalarValue::Int(1)),
            max: Some(ScalarValue::Int(100)),
        };
        assert_eq!(cs.null_count, 3);
        assert_eq!(cs.min, Some(ScalarValue::Int(1)));
        assert_eq!(cs.max, Some(ScalarValue::Int(100)));
        let _ = ScalarValue::Bool(true);
        let _ = ScalarValue::Float(2.5);
        let _ = ScalarValue::Str("z".to_string());
    }
}
