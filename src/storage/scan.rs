// SPDX-License-Identifier: Apache-2.0
//! Scan requests and zone-map pruning.
//!
//! A [`ScanRequest`] tells a [`StorageTable`](crate::storage::StorageTable)
//! which columns a query reads and which simple predicates its rows must
//! satisfy. Pruning is purely an optimisation: the query plan keeps every
//! filter, so a table may always return *more* rows or columns than asked
//! for, never fewer.
//!
//! Predicates are kept in disjunctive form — a chunk is read when *any*
//! [`ScanRequest::filters`] conjunction may match it — so one request can
//! describe a table that a statement scans several times (a self-join, a
//! `UNION` arm) with different filters.

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};

use crate::plan::logical_plan::{BinaryOp, Expr, Literal, LogicalPlan, Schema, UnaryOp};
use crate::storage::stats::{compare_literals, ColumnStatistics};

/// Comparison in a [`PrunePredicate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruneOp {
    /// `col = v`
    Eq,
    /// `col <> v`
    NotEq,
    /// `col < v`
    Lt,
    /// `col <= v`
    LtEq,
    /// `col > v`
    Gt,
    /// `col >= v`
    GtEq,
}

impl PruneOp {
    fn flip(self) -> Self {
        match self {
            PruneOp::Lt => PruneOp::Gt,
            PruneOp::LtEq => PruneOp::GtEq,
            PruneOp::Gt => PruneOp::Lt,
            PruneOp::GtEq => PruneOp::LtEq,
            other => other,
        }
    }
}

/// One predicate on a single column that a zone map can refute.
#[derive(Debug, Clone, PartialEq)]
pub enum PrunePredicate {
    /// `column <op> value`.
    Compare {
        /// Column ordinal in the table schema.
        column: usize,
        /// Comparison.
        op: PruneOp,
        /// Constant operand.
        value: Literal,
    },
    /// `column IN (values)` (also produced from an `OR` of equalities).
    InList {
        /// Column ordinal in the table schema.
        column: usize,
        /// Candidate values.
        values: Vec<Literal>,
    },
    /// `column IS NULL`.
    IsNull {
        /// Column ordinal in the table schema.
        column: usize,
    },
    /// `column IS NOT NULL`.
    IsNotNull {
        /// Column ordinal in the table schema.
        column: usize,
    },
}

impl PrunePredicate {
    /// `false` only when `stats` (for a chunk of `rows` rows) prove no row
    /// of the chunk can satisfy the predicate. Unknown statistics, or values
    /// that cannot be compared, keep the chunk.
    pub fn may_match(&self, stats: &[ColumnStatistics], rows: u64) -> bool {
        let col = |c: usize| stats.get(c);
        match self {
            PrunePredicate::IsNull { column } => col(*column)
                .and_then(|s| s.null_count)
                .is_none_or(|n| n > 0),
            PrunePredicate::IsNotNull { column } => col(*column)
                .and_then(|s| s.null_count)
                .is_none_or(|n| n < rows),
            PrunePredicate::Compare { column, op, value } => match col(*column) {
                Some(s) => range_may_match(s, *op, value, rows),
                None => true,
            },
            PrunePredicate::InList { column, values } => match col(*column) {
                Some(s) => values
                    .iter()
                    .any(|v| range_may_match(s, PruneOp::Eq, v, rows)),
                None => true,
            },
        }
    }

    fn column(&self) -> usize {
        match self {
            PrunePredicate::Compare { column, .. }
            | PrunePredicate::InList { column, .. }
            | PrunePredicate::IsNull { column }
            | PrunePredicate::IsNotNull { column } => *column,
        }
    }
}

fn range_may_match(s: &ColumnStatistics, op: PruneOp, v: &Literal, rows: u64) -> bool {
    // An all-NULL chunk satisfies no comparison.
    if s.null_count == Some(rows) && rows > 0 {
        return false;
    }
    let lo = s.min.as_ref().and_then(|m| compare_literals(m, v));
    let hi = s.max.as_ref().and_then(|m| compare_literals(m, v));
    use Ordering::*;
    match op {
        PruneOp::Eq => !(matches!(lo, Some(Greater)) || matches!(hi, Some(Less))),
        PruneOp::NotEq => !(matches!(lo, Some(Equal)) && matches!(hi, Some(Equal))),
        PruneOp::Lt => !matches!(lo, Some(Greater | Equal)),
        PruneOp::LtEq => !matches!(lo, Some(Greater)),
        PruneOp::Gt => !matches!(hi, Some(Less | Equal)),
        PruneOp::GtEq => !matches!(hi, Some(Less)),
    }
}

/// What a query needs from one table.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScanRequest {
    /// Column ordinals to read; `None` reads every column. Columns outside
    /// the set may come back as placeholder values (the plan never reads
    /// them).
    pub columns: Option<BTreeSet<usize>>,
    /// Disjunction of conjunctions. Empty means "no row filter"; an empty
    /// inner conjunction also matches everything.
    pub filters: Vec<Vec<PrunePredicate>>,
}

impl ScanRequest {
    /// A request for every row and column.
    pub fn full() -> Self {
        Self::default()
    }

    /// `true` when a chunk with these column statistics must be read.
    pub fn chunk_may_match(&self, stats: &[ColumnStatistics], rows: u64) -> bool {
        self.filters.is_empty()
            || self
                .filters
                .iter()
                .any(|conj| conj.iter().all(|p| p.may_match(stats, rows)))
    }

    /// Request that covers both `self` and `other` (union of rows and
    /// columns).
    pub fn union(&self, other: &ScanRequest) -> ScanRequest {
        let columns = match (&self.columns, &other.columns) {
            (Some(a), Some(b)) => Some(a.union(b).copied().collect()),
            _ => None,
        };
        let filters = if self.filters.is_empty() || other.filters.is_empty() {
            Vec::new()
        } else {
            let mut f = self.filters.clone();
            f.extend(other.filters.iter().cloned());
            f
        };
        ScanRequest { columns, filters }
    }

    /// Every column ordinal a predicate references (must be read to keep the
    /// plan's filter correct — they are part of the scan projection anyway).
    fn predicate_columns(&self) -> impl Iterator<Item = usize> + '_ {
        self.filters.iter().flatten().map(PrunePredicate::column)
    }
}

/// Derive the per-table scan requests of an (optimized) logical plan.
///
/// Only tables in `tables` are analysed. A table scanned under a filter that
/// sits directly on its scan (where the optimizer's predicate pushdown puts
/// it) gets that filter's prunable conjuncts; a scan anywhere else reads all
/// rows. Columns come from the scan's projection.
pub fn scan_requests(plan: &LogicalPlan, tables: &[&str]) -> HashMap<String, ScanRequest> {
    let mut out: HashMap<String, ScanRequest> = HashMap::new();
    walk(plan, None, tables, &mut out);
    // Predicate columns must be readable even if the projection pruned them
    // (it never does — the filter consumes them — but stay defensive).
    for req in out.values_mut() {
        let extra: Vec<usize> = req.predicate_columns().collect();
        if let Some(cols) = req.columns.as_mut() {
            cols.extend(extra);
        }
    }
    out
}

fn walk<'a>(
    plan: &'a LogicalPlan,
    filter_above: Option<&'a Expr>,
    tables: &[&str],
    out: &mut HashMap<String, ScanRequest>,
) {
    match plan {
        LogicalPlan::Scan {
            table,
            projection,
            schema,
        } => {
            if !tables.contains(&table.as_str()) {
                return;
            }
            let columns = projection.as_ref().map(|cols| {
                cols.iter()
                    .filter_map(|c| resolve_column(schema, table, c))
                    .collect::<BTreeSet<usize>>()
            });
            // An unresolvable projection name means we cannot trust the set.
            let columns = match (projection, columns) {
                (Some(p), Some(c)) if c.len() == p.len() => Some(c),
                _ => None,
            };
            let filters = match filter_above {
                Some(pred) => {
                    let conj = prunable_conjuncts(pred, schema, table);
                    if conj.is_empty() {
                        Vec::new()
                    } else {
                        vec![conj]
                    }
                }
                None => Vec::new(),
            };
            let req = ScanRequest { columns, filters };
            out.entry(table.clone())
                .and_modify(|r| *r = r.union(&req))
                .or_insert(req);
        }
        LogicalPlan::Filter { input, predicate } => {
            if matches!(input.as_ref(), LogicalPlan::Scan { .. }) {
                walk(input, Some(predicate), tables, out);
            } else {
                walk(input, None, tables, out);
            }
        }
        other => {
            for child in other.children() {
                walk(child, None, tables, out);
            }
        }
    }
}

/// Resolve a column reference against a scan's table schema: exact name, or
/// a `table.column` qualified name.
fn resolve_column(schema: &Schema, table: &str, name: &str) -> Option<usize> {
    if let Ok(i) = schema.index_of(name) {
        return Some(i);
    }
    let (q, c) = name.rsplit_once('.')?;
    if q == table || q.ends_with(&format!(".{table}")) {
        schema.index_of(c).ok()
    } else {
        None
    }
}

fn prunable_conjuncts(pred: &Expr, schema: &Schema, table: &str) -> Vec<PrunePredicate> {
    let mut conjuncts = Vec::new();
    split_and(pred, &mut conjuncts);
    conjuncts
        .into_iter()
        .filter_map(|e| to_prune(e, schema, table))
        .collect()
}

fn split_and<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
    match e {
        Expr::Binary {
            op: BinaryOp::And,
            left,
            right,
        } => {
            split_and(left, out);
            split_and(right, out);
        }
        other => match three_valued_condition(other) {
            Some(c) => split_and(c, out),
            None => out.push(other),
        },
    }
}

/// `c` for the frontend's three-valued wrapper `CASE WHEN c THEN true WHEN
/// NOT c THEN false ELSE NULL END` (an `IN` list): as a filter it keeps
/// exactly the rows where `c` is TRUE.
fn three_valued_condition(e: &Expr) -> Option<&Expr> {
    let Expr::Case {
        branches,
        else_branch,
    } = e
    else {
        return None;
    };
    let [(c, Expr::Literal(Literal::Bool(true))), (_, Expr::Literal(Literal::Bool(false)))] =
        branches.as_slice()
    else {
        return None;
    };
    match else_branch.as_deref() {
        None | Some(Expr::Literal(Literal::Null)) => Some(c),
        _ => None,
    }
}

fn to_prune(e: &Expr, schema: &Schema, table: &str) -> Option<PrunePredicate> {
    let column_of = |e: &Expr| match e {
        Expr::Column(c) => resolve_column(schema, table, c),
        _ => None,
    };
    let literal_of = |e: &Expr| match e {
        Expr::Literal(l) if !matches!(l, Literal::Null) => Some(l.clone()),
        _ => None,
    };
    match e {
        Expr::Binary { op, left, right } => {
            let pop = match op {
                BinaryOp::Eq => PruneOp::Eq,
                BinaryOp::NotEq => PruneOp::NotEq,
                BinaryOp::Lt => PruneOp::Lt,
                BinaryOp::LtEq => PruneOp::LtEq,
                BinaryOp::Gt => PruneOp::Gt,
                BinaryOp::GtEq => PruneOp::GtEq,
                BinaryOp::Or => return or_of_equalities(e, schema, table),
                _ => return None,
            };
            if let (Some(column), Some(value)) = (column_of(left), literal_of(right)) {
                Some(PrunePredicate::Compare {
                    column,
                    op: pop,
                    value,
                })
            } else if let (Some(value), Some(column)) = (literal_of(left), column_of(right)) {
                Some(PrunePredicate::Compare {
                    column,
                    op: pop.flip(),
                    value,
                })
            } else {
                None
            }
        }
        Expr::Unary { op, operand } => {
            let column = column_of(operand)?;
            match op {
                UnaryOp::IsNull => Some(PrunePredicate::IsNull { column }),
                UnaryOp::IsNotNull => Some(PrunePredicate::IsNotNull { column }),
                #[allow(unreachable_patterns)]
                _ => None,
            }
        }
        _ => None,
    }
}

/// `c = a OR c = b OR ...` (all on one column) → `c IN (a, b, ...)`.
fn or_of_equalities(e: &Expr, schema: &Schema, table: &str) -> Option<PrunePredicate> {
    fn collect<'a>(e: &'a Expr, out: &mut Vec<&'a Expr>) {
        match e {
            Expr::Binary {
                op: BinaryOp::Or,
                left,
                right,
            } => {
                collect(left, out);
                collect(right, out);
            }
            other => out.push(other),
        }
    }
    let mut arms = Vec::new();
    collect(e, &mut arms);
    let mut column = None;
    let mut values = Vec::new();
    for arm in arms {
        match to_prune(arm, schema, table)? {
            PrunePredicate::Compare {
                column: c,
                op: PruneOp::Eq,
                value,
            } => {
                if column.is_some_and(|k| k != c) {
                    return None;
                }
                column = Some(c);
                values.push(value);
            }
            PrunePredicate::InList {
                column: c,
                values: v,
            } => {
                if column.is_some_and(|k| k != c) {
                    return None;
                }
                column = Some(c);
                values.extend(v);
            }
            _ => return None,
        }
    }
    Some(PrunePredicate::InList {
        column: column?,
        values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{DataType, Field};

    fn stats(min: i64, max: i64, nulls: u64) -> ColumnStatistics {
        ColumnStatistics {
            null_count: Some(nulls),
            min: Some(Literal::Int64(min)),
            max: Some(Literal::Int64(max)),
            distinct_estimate: None,
        }
    }

    fn cmp(op: PruneOp, v: i64) -> PrunePredicate {
        PrunePredicate::Compare {
            column: 0,
            op,
            value: Literal::Int64(v),
        }
    }

    #[test]
    fn zone_map_refutes_out_of_range_comparisons() {
        let s = [stats(10, 20, 0)];
        assert!(!cmp(PruneOp::Eq, 5).may_match(&s, 8));
        assert!(cmp(PruneOp::Eq, 10).may_match(&s, 8));
        assert!(!cmp(PruneOp::Lt, 10).may_match(&s, 8));
        assert!(cmp(PruneOp::LtEq, 10).may_match(&s, 8));
        assert!(!cmp(PruneOp::Gt, 20).may_match(&s, 8));
        assert!(cmp(PruneOp::GtEq, 20).may_match(&s, 8));
        assert!(cmp(PruneOp::NotEq, 10).may_match(&s, 8));
        let single = [stats(7, 7, 0)];
        assert!(!cmp(PruneOp::NotEq, 7).may_match(&single, 8));
    }

    #[test]
    fn null_predicates_use_null_counts() {
        let none = [stats(1, 2, 0)];
        let all = [stats(1, 2, 8)];
        assert!(!PrunePredicate::IsNull { column: 0 }.may_match(&none, 8));
        assert!(!PrunePredicate::IsNotNull { column: 0 }.may_match(&all, 8));
        assert!(!cmp(PruneOp::GtEq, 0).may_match(&all, 8));
    }

    #[test]
    fn unknown_or_incomparable_stats_keep_the_chunk() {
        let unknown = [ColumnStatistics::default()];
        assert!(cmp(PruneOp::Eq, 5).may_match(&unknown, 8));
        let p = PrunePredicate::Compare {
            column: 0,
            op: PruneOp::Eq,
            value: Literal::Utf8("x".into()),
        };
        assert!(p.may_match(&[stats(1, 2, 0)], 8));
        assert!(cmp(PruneOp::Eq, 5).may_match(&[], 8));
    }

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, true),
            Field::new("c", DataType::Int64, false),
        ])
    }

    fn scan(projection: Option<Vec<&str>>) -> LogicalPlan {
        LogicalPlan::Scan {
            table: "t".into(),
            projection: projection.map(|p| p.into_iter().map(String::from).collect()),
            schema: schema(),
        }
    }

    fn col(n: &str) -> Expr {
        Expr::Column(n.into())
    }

    fn lit(v: i64) -> Expr {
        Expr::Literal(Literal::Int64(v))
    }

    fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }

    #[test]
    fn extracts_conjuncts_and_columns_from_filter_over_scan() {
        let pred = bin(
            BinaryOp::And,
            bin(BinaryOp::Gt, col("a"), lit(5)),
            bin(
                BinaryOp::And,
                bin(BinaryOp::LtEq, lit(9), col("t.c")),
                bin(BinaryOp::Eq, bin(BinaryOp::Add, col("a"), lit(1)), lit(3)),
            ),
        );
        let plan = LogicalPlan::Filter {
            input: Box::new(scan(Some(vec!["a", "c"]))),
            predicate: pred,
        };
        let reqs = scan_requests(&plan, &["t"]);
        let r = &reqs["t"];
        assert_eq!(r.columns, Some([0, 2].into_iter().collect()));
        assert_eq!(
            r.filters,
            vec![vec![
                cmp(PruneOp::Gt, 5),
                PrunePredicate::Compare {
                    column: 2,
                    op: PruneOp::GtEq,
                    value: Literal::Int64(9)
                },
            ]]
        );
    }

    #[test]
    fn or_of_equalities_becomes_in_list() {
        let pred = bin(
            BinaryOp::Or,
            bin(BinaryOp::Eq, col("a"), lit(1)),
            bin(BinaryOp::Eq, lit(4), col("a")),
        );
        let plan = LogicalPlan::Filter {
            input: Box::new(scan(None)),
            predicate: pred,
        };
        let r = &scan_requests(&plan, &["t"])["t"];
        assert_eq!(r.columns, None);
        assert_eq!(
            r.filters,
            vec![vec![PrunePredicate::InList {
                column: 0,
                values: vec![Literal::Int64(1), Literal::Int64(4)]
            }]]
        );
        let mixed = bin(
            BinaryOp::Or,
            bin(BinaryOp::Eq, col("a"), lit(1)),
            bin(BinaryOp::Eq, col("c"), lit(4)),
        );
        let plan = LogicalPlan::Filter {
            input: Box::new(scan(None)),
            predicate: mixed,
        };
        assert!(scan_requests(&plan, &["t"])["t"].filters.is_empty());
    }

    #[test]
    fn three_valued_in_list_wrapper_is_prunable() {
        // The frontend's strict-3VL form of `a IN (1, 4)`.
        let c = bin(
            BinaryOp::Or,
            bin(BinaryOp::Eq, col("a"), lit(1)),
            bin(BinaryOp::Eq, col("a"), lit(4)),
        );
        let wrapped = Expr::Case {
            branches: vec![
                (c.clone(), Expr::Literal(Literal::Bool(true))),
                (
                    Expr::Unary {
                        op: UnaryOp::Not,
                        operand: Box::new(c),
                    },
                    Expr::Literal(Literal::Bool(false)),
                ),
            ],
            else_branch: Some(Box::new(Expr::Literal(Literal::Null))),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(scan(None)),
            predicate: wrapped,
        };
        assert_eq!(
            scan_requests(&plan, &["t"])["t"].filters,
            vec![vec![PrunePredicate::InList {
                column: 0,
                values: vec![Literal::Int64(1), Literal::Int64(4)]
            }]]
        );
    }

    #[test]
    fn repeated_scans_union_their_requests() {
        let filtered = LogicalPlan::Filter {
            input: Box::new(scan(Some(vec!["a"]))),
            predicate: bin(BinaryOp::Gt, col("a"), lit(5)),
        };
        let plan = LogicalPlan::Union {
            inputs: vec![filtered.clone(), scan(Some(vec!["c"]))],
        };
        let r = &scan_requests(&plan, &["t"])["t"];
        assert_eq!(r.columns, Some([0, 2].into_iter().collect()));
        assert!(r.filters.is_empty(), "the unfiltered arm needs every row");
        let both = LogicalPlan::Union {
            inputs: vec![filtered.clone(), filtered],
        };
        assert_eq!(scan_requests(&both, &["t"])["t"].filters.len(), 2);
    }

    #[test]
    fn disjunctive_request_keeps_chunk_if_any_arm_matches() {
        let req = ScanRequest {
            columns: None,
            filters: vec![vec![cmp(PruneOp::Lt, 0)], vec![cmp(PruneOp::Gt, 100)]],
        };
        assert!(!req.chunk_may_match(&[stats(10, 20, 0)], 4));
        assert!(req.chunk_may_match(&[stats(50, 150, 0)], 4));
        assert!(ScanRequest::full().chunk_may_match(&[stats(10, 20, 0)], 4));
    }
}
