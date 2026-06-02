// SPDX-License-Identifier: Apache-2.0

//! Plan introspection / `EXPLAIN`.
//!
//! Human-readable, tree-indented renderings of a [`LogicalPlan`] and a
//! [`PhysicalPlan`]. These are intentionally read-only: they walk an
//! already-built plan and pretty-print it without touching the GPU,
//! re-type-checking, or otherwise mutating anything. The output is meant for
//! diagnostics (`EXPLAIN`-style) and tests, so the format is stable and
//! line-oriented:
//!
//! * one node per line,
//! * two spaces of indentation per tree depth,
//! * each line starts with the node kind followed by its key attributes.
//!
//! Expressions are rendered compactly by [`format_expr`] (a small
//! `Expr -> String` pretty-printer) so a node line stays on one line.

use std::fmt::Write as _;

use crate::plan::logical_plan::{
    AggregateExpr, BinaryOp, DataType, Expr, JoinType, Literal, LogicalPlan, ScalarFnKind,
    SortExpr, UnaryOp,
};
use crate::plan::physical_plan::{KernelSpec, PhysicalPlan};

/// Width, in spaces, of one tree-indentation level.
const INDENT: usize = 2;

/// Push `depth` levels of indentation onto `out`.
fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth * INDENT {
        out.push(' ');
    }
}

// ---------------------------------------------------------------------------
// Expression pretty-printer
// ---------------------------------------------------------------------------

/// Compact, single-line rendering of a binary operator as SQL-ish text.
fn binary_op_str(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "+",
        BinaryOp::Sub => "-",
        BinaryOp::Mul => "*",
        BinaryOp::Div => "/",
        BinaryOp::Eq => "=",
        BinaryOp::NotEq => "<>",
        BinaryOp::Lt => "<",
        BinaryOp::LtEq => "<=",
        BinaryOp::Gt => ">",
        BinaryOp::GtEq => ">=",
        BinaryOp::And => "AND",
        BinaryOp::Or => "OR",
        BinaryOp::Concat => "||",
    }
}

/// Compact rendering of a [`Literal`] constant.
fn format_literal(lit: &Literal) -> String {
    match lit {
        Literal::Null => "NULL".to_string(),
        Literal::Bool(b) => b.to_string(),
        Literal::Int32(v) => v.to_string(),
        Literal::Int64(v) => v.to_string(),
        Literal::Float32(v) => v.to_string(),
        Literal::Float64(v) => v.to_string(),
        Literal::Utf8(s) => format!("'{s}'"),
        Literal::Decimal128(v, p, s) => format!("{v}::Decimal128({p},{s})"),
        Literal::Date32(d) => format!("Date32({d})"),
        Literal::Timestamp(t, unit, tz) => match tz {
            Some(z) => format!("Timestamp({t}, {unit:?}, {z})"),
            None => format!("Timestamp({t}, {unit:?})"),
        },
    }
}

/// Compact rendering of a [`DataType`] for `CAST` targets.
fn format_dtype(dt: DataType) -> String {
    format!("{dt:?}")
}

/// Compact, single-line pretty-printer for a scalar [`Expr`].
///
/// Output is SQL-flavoured but not guaranteed to round-trip through the
/// parser — it exists for human inspection (`EXPLAIN`) and tests. Binary
/// expressions are always fully parenthesised so precedence is unambiguous.
pub fn format_expr(expr: &Expr) -> String {
    match expr {
        Expr::Extract { .. } | Expr::DateTrunc { .. } | Expr::ScalarSubquery(_) | Expr::InSubquery { .. } => format!("{expr:?}"),
        Expr::Column(name) => name.clone(),
        Expr::Literal(lit) => format_literal(lit),
        Expr::Binary { op, left, right } => {
            format!(
                "({} {} {})",
                format_expr(left),
                binary_op_str(*op),
                format_expr(right)
            )
        }
        Expr::Unary { op, operand } => {
            let inner = format_expr(operand);
            match op {
                UnaryOp::IsNull => format!("({inner} IS NULL)"),
                UnaryOp::IsNotNull => format!("({inner} IS NOT NULL)"),
                UnaryOp::Not => format!("(NOT {inner})"),
            }
        }
        Expr::Case {
            branches,
            else_branch,
        } => {
            let mut s = String::from("CASE");
            for (when, then) in branches {
                let _ = write!(s, " WHEN {} THEN {}", format_expr(when), format_expr(then));
            }
            if let Some(e) = else_branch {
                let _ = write!(s, " ELSE {}", format_expr(e));
            }
            s.push_str(" END");
            s
        }
        Expr::Like {
            expr,
            pattern,
            negated,
            case_insensitive,
            ..
        } => {
            let base = if *case_insensitive { "ILIKE" } else { "LIKE" };
            let kw = if *negated {
                if *case_insensitive {
                    "NOT ILIKE"
                } else {
                    "NOT LIKE"
                }
            } else {
                base
            };
            format!("({} {} '{}')", format_expr(expr), kw, pattern)
        }
        Expr::Cast { expr, target, .. } => {
            format!("CAST({} AS {})", format_expr(expr), format_dtype(*target))
        }
        Expr::CastFormat {
            expr,
            target,
            pattern,
            ..
        } => {
            // Render the validated tokens back to a readable pattern string.
            let mut pat = String::new();
            for tok in pattern {
                use crate::plan::logical_plan::FormatToken::*;
                match tok {
                    Year4 => pat.push_str("YYYY"),
                    Month => pat.push_str("MM"),
                    Day => pat.push_str("DD"),
                    Hour24 => pat.push_str("HH24"),
                    Minute => pat.push_str("MI"),
                    Second => pat.push_str("SS"),
                    Literal(c) => pat.push(*c),
                }
            }
            format!(
                "CAST({} AS {} FORMAT '{}')",
                format_expr(expr),
                format_dtype(*target),
                pat
            )
        }
        Expr::ScalarFn { kind, args } => {
            let rendered: Vec<String> = args.iter().map(format_expr).collect();
            format!("{}({})", scalar_fn_name(*kind), rendered.join(", "))
        }
        Expr::Alias(inner, name) => format!("{} AS {}", format_expr(inner), name),
    }
}

/// SQL name of a scalar function for the pretty-printer.
fn scalar_fn_name(kind: ScalarFnKind) -> &'static str {
    kind.sql_name()
}

/// Compact rendering of an [`AggregateExpr`] like `SUM(x)` / `COUNT(y)`.
fn format_aggregate(agg: &AggregateExpr) -> String {
    match agg {
        AggregateExpr::Count(e) => format!("COUNT({})", format_expr(e)),
        AggregateExpr::Sum(e) => format!("SUM({})", format_expr(e)),
        AggregateExpr::Min(e) => format!("MIN({})", format_expr(e)),
        AggregateExpr::Max(e) => format!("MAX({})", format_expr(e)),
        AggregateExpr::Avg(e) => format!("AVG({})", format_expr(e)),
        AggregateExpr::VarPop(e) => format!("VAR_POP({})", format_expr(e)),
        AggregateExpr::VarSamp(e) => format!("VAR_SAMP({})", format_expr(e)),
        AggregateExpr::StddevPop(e) => format!("STDDEV_POP({})", format_expr(e)),
        AggregateExpr::StddevSamp(e) => format!("STDDEV_SAMP({})", format_expr(e)),
    }
}

/// Compact rendering of a single ORDER BY key.
fn format_sort_expr(s: &SortExpr) -> String {
    let dir = if s.descending { "DESC" } else { "ASC" };
    let nulls = if s.nulls_first {
        "NULLS FIRST"
    } else {
        "NULLS LAST"
    };
    format!("{} {} {}", format_expr(&s.expr), dir, nulls)
}

/// Human-readable name for a [`JoinType`].
fn join_type_str(jt: JoinType) -> &'static str {
    match jt {
        JoinType::Inner => "Inner",
        JoinType::LeftOuter => "LeftOuter",
        JoinType::RightOuter => "RightOuter",
        JoinType::FullOuter => "FullOuter",
        JoinType::Cross => "Cross",
    }
}

/// Render the equi-join `on` pairs as `l = r, ...`.
fn format_join_on(on: &[(Expr, Expr)]) -> String {
    on.iter()
        .map(|(l, r)| format!("{} = {}", format_expr(l), format_expr(r)))
        .collect::<Vec<_>>()
        .join(", ")
}

// ---------------------------------------------------------------------------
// Logical plan
// ---------------------------------------------------------------------------

/// Render `plan` as a tree-indented, human-readable string.
///
/// Each node occupies one line; children are indented two spaces deeper than
/// their parent. The first token of each line is the node kind followed by
/// its key attributes (table + projection for `Scan`, predicate for
/// `Filter`, the SELECT list for `Projection`, etc.). Expressions are
/// rendered compactly by [`format_expr`].
///
/// This is a pure, read-only walk: it never re-type-checks the plan or
/// touches the GPU, so it works on partially-built fixtures too.
pub fn format_logical(plan: &LogicalPlan) -> String {
    let mut out = String::new();
    format_logical_into(plan, 0, &mut out);
    out
}

/// Recursive worker for [`format_logical`].
fn format_logical_into(plan: &LogicalPlan, depth: usize, out: &mut String) {
    indent(out, depth);
    match plan {
        LogicalPlan::Window { input, .. } => {
            let _ = writeln!(out, "Window");
            format_logical_into(input, depth + 1, out);
        }
        LogicalPlan::Scan {
            table, projection, ..
        } => {
            let proj = match projection {
                Some(cols) => format!(" projection=[{}]", cols.join(", ")),
                None => String::new(),
            };
            let _ = writeln!(out, "Scan: table={table}{proj}");
        }
        LogicalPlan::Filter { input, predicate } => {
            let _ = writeln!(out, "Filter: {}", format_expr(predicate));
            format_logical_into(input, depth + 1, out);
        }
        LogicalPlan::Project { input, exprs } => {
            let rendered: Vec<String> = exprs.iter().map(format_expr).collect();
            let _ = writeln!(out, "Projection: [{}]", rendered.join(", "));
            format_logical_into(input, depth + 1, out);
        }
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            let gb: Vec<String> = group_by.iter().map(format_expr).collect();
            let aggs: Vec<String> = aggregates.iter().map(format_aggregate).collect();
            let _ = writeln!(
                out,
                "Aggregate: group_by=[{}] aggs=[{}]",
                gb.join(", "),
                aggs.join(", ")
            );
            format_logical_into(input, depth + 1, out);
        }
        LogicalPlan::Distinct { input } => {
            let _ = writeln!(out, "Distinct");
            format_logical_into(input, depth + 1, out);
        }
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            let _ = writeln!(out, "Limit: limit={limit} offset={offset}");
            format_logical_into(input, depth + 1, out);
        }
        LogicalPlan::Sort { input, sort_exprs } => {
            let keys: Vec<String> = sort_exprs.iter().map(format_sort_expr).collect();
            let _ = writeln!(out, "Sort: [{}]", keys.join(", "));
            format_logical_into(input, depth + 1, out);
        }
        LogicalPlan::Union { inputs } => {
            let _ = writeln!(out, "Union");
            for child in inputs {
                format_logical_into(child, depth + 1, out);
            }
        }
        LogicalPlan::SetOp { left, right, op, all } => {
            let all_str = if *all { " all" } else { "" };
            let _ = writeln!(out, "SetOp: op={}{all_str}", op.keyword());
            format_logical_into(left, depth + 1, out);
            format_logical_into(right, depth + 1, out);
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => {
            let on_str = if on.is_empty() {
                String::new()
            } else {
                format!(" on=[{}]", format_join_on(on))
            };
            let filt = match filter {
                Some(f) => format!(" filter={}", format_expr(f)),
                None => String::new(),
            };
            let _ = writeln!(out, "Join: type={}{on_str}{filt}", join_type_str(*join_type));
            format_logical_into(left, depth + 1, out);
            format_logical_into(right, depth + 1, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Recursive CTE (feature F1)
// ---------------------------------------------------------------------------

/// Render a [`RecursiveCtePlan`](crate::plan::sql_frontend::RecursiveCtePlan)
/// as a tree-indented, human-readable string for `EXPLAIN`.
///
/// A `WITH RECURSIVE` query is not a single [`LogicalPlan`] — the engine
/// orchestrates a host-side fixpoint over three standalone subplans — so it
/// gets its own renderer. The output mirrors [`format_logical`]'s shape: a
/// `RecursiveCte` header line naming the CTE (and `UNION` vs `UNION ALL`),
/// then the `Anchor`, `Recursive`, and `Main` subplans each rendered one
/// indent level deeper under a labelled sub-header.
pub fn format_recursive_cte(rec: &crate::plan::sql_frontend::RecursiveCtePlan) -> String {
    let mut out = String::new();
    let cols: Vec<&str> = rec.cte_schema.fields.iter().map(|f| f.name.as_str()).collect();
    let union = if rec.all { "UNION ALL" } else { "UNION" };
    // A non-linear (self-join) recursive term is evaluated naively (the full
    // accumulated relation is re-bound each iteration); surface that in the
    // header so EXPLAIN distinguishes it from the linear/semi-naive path.
    let eval = if rec.naive { " naive" } else { "" };
    let _ = writeln!(
        out,
        "RecursiveCte: name={} ({}) {union}{eval}",
        rec.name,
        cols.join(", ")
    );
    indent(&mut out, 1);
    let _ = writeln!(out, "Anchor:");
    format_logical_into(&rec.anchor, 2, &mut out);
    indent(&mut out, 1);
    let _ = writeln!(out, "Recursive:");
    format_logical_into(&rec.recursive, 2, &mut out);
    indent(&mut out, 1);
    let _ = writeln!(out, "Main:");
    format_logical_into(&rec.main, 2, &mut out);
    out
}

/// Render a
/// [`MutualRecursiveCtePlan`](crate::plan::sql_frontend::MutualRecursiveCtePlan)
/// — a system of mutually-recursive CTEs advanced in lockstep — for `EXPLAIN`.
///
/// Emits a `MutualRecursiveCte` header naming the member count, then one
/// `Cte: <name> ...` block per member (each with its `Anchor:` and, when
/// recursive, `Recursive:` subplans), and finally the shared `Main:` subplan.
pub fn format_mutual_recursive_cte(
    rec: &crate::plan::sql_frontend::MutualRecursiveCtePlan,
) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "MutualRecursiveCte: {} CTEs", rec.ctes.len());
    for term in &rec.ctes {
        let cols: Vec<&str> = term.cte_schema.fields.iter().map(|f| f.name.as_str()).collect();
        indent(&mut out, 1);
        let kind = match &term.recursive {
            Some(_) if term.all => "recursive UNION ALL",
            Some(_) => "recursive UNION",
            None => "non-recursive",
        };
        let _ = writeln!(out, "Cte: {} ({}) {kind}", term.name, cols.join(", "));
        indent(&mut out, 2);
        let _ = writeln!(out, "Anchor:");
        format_logical_into(&term.anchor, 3, &mut out);
        if let Some(recursive) = &term.recursive {
            indent(&mut out, 2);
            let _ = writeln!(out, "Recursive:");
            format_logical_into(recursive, 3, &mut out);
        }
    }
    indent(&mut out, 1);
    let _ = writeln!(out, "Main:");
    format_logical_into(&rec.main, 2, &mut out);
    out
}

// ---------------------------------------------------------------------------
// LATERAL apply (feature F3 — LATERAL)
// ---------------------------------------------------------------------------

/// Render a [`LateralApplyPlan`](crate::plan::sql_frontend::LateralApplyPlan) —
/// a host nested-loop Apply for a LATERAL derived table — for `EXPLAIN`.
///
/// A LATERAL apply is not a single [`LogicalPlan`] (the engine re-runs the
/// correlated subquery per left row; see `Engine::execute_lateral_apply`), so
/// it gets its own renderer. The header names the apply kind (INNER vs LEFT),
/// the subquery alias columns, and the correlated outer columns it threads in;
/// the `Left:` sub-header renders the LEFT relation, `Lateral:` the per-row
/// (correlation-rewritten) subplan, and `Outer:` the OUTER query template run
/// over the applied relation.
pub fn format_lateral_apply(la: &crate::plan::sql_frontend::LateralApplyPlan) -> String {
    let mut out = String::new();
    let kind = if la.left_join { "LEFT" } else { "INNER" };
    let sub_cols: Vec<&str> = la
        .subquery_schema
        .fields
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    let corr_cols: Vec<&str> = la
        .outer_schema
        .fields
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    let _ = writeln!(
        out,
        "LateralApply: {kind} subquery=[{}] correlations=[{}]",
        sub_cols.join(", "),
        corr_cols.join(", ")
    );
    indent(&mut out, 1);
    let _ = writeln!(out, "Left:");
    format_logical_into(&la.left, 2, &mut out);
    indent(&mut out, 1);
    let _ = writeln!(out, "Lateral:");
    format_logical_into(&la.lateral_subplan, 2, &mut out);
    indent(&mut out, 1);
    let _ = writeln!(out, "Outer:");
    format_logical_into(&la.post, 2, &mut out);
    out
}

// ---------------------------------------------------------------------------
// COUNT(DISTINCT col) with GROUP BY (feature F3-finish)
// ---------------------------------------------------------------------------

/// Render a
/// [`CountDistinctGroupByPlan`](crate::plan::sql_frontend::CountDistinctGroupByPlan)
/// as a tree-indented, human-readable string for `EXPLAIN`.
///
/// A sole `COUNT(DISTINCT col)` with `GROUP BY` is not a single
/// [`LogicalPlan`] — the engine orchestrates a host-side per-group distinct
/// count (see `Engine::execute_count_distinct_groupby`) — so it gets its own
/// renderer. The header names the group keys + the count alias; the `Base:`
/// sub-header renders the subplan that materialises `[group_keys..., col]`,
/// and (when present) a `Post:` sub-header renders the HAVING/ORDER BY/LIMIT
/// plan applied to the count result.
pub fn format_count_distinct_groupby(
    cd: &crate::plan::sql_frontend::CountDistinctGroupByPlan,
) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "CountDistinctGroupBy: group_keys=[{}] count=COUNT(DISTINCT) AS {}",
        cd.group_key_names.join(", "),
        cd.count_alias
    );
    indent(&mut out, 1);
    let _ = writeln!(out, "Base:");
    format_logical_into(&cd.base, 2, &mut out);
    if let Some(post) = &cd.post {
        indent(&mut out, 1);
        let _ = writeln!(out, "Post:");
        format_logical_into(post, 2, &mut out);
    }
    out
}

/// Render a
/// [`MultiAggGroupByPlan`](crate::plan::sql_frontend::MultiAggGroupByPlan) for
/// `EXPLAIN`: the generalized multi / mixed COUNT(DISTINCT) + GROUP BY shape.
///
/// The header names the group keys and the per-group aggregate kinds (in
/// output order); the `Base:` sub-header renders the subplan that materialises
/// `[group_keys..., agg_inputs...]`, and a `Post:` sub-header (when present)
/// renders the HAVING / ORDER BY / LIMIT plan.
pub fn format_multi_agg_groupby(
    cd: &crate::plan::sql_frontend::MultiAggGroupByPlan,
) -> String {
    use crate::plan::sql_frontend::CdAgg;
    let mut out = String::new();
    let aggs: Vec<&str> = cd
        .aggs
        .iter()
        .map(|a| match a {
            CdAgg::CountDistinct { .. } => "COUNT(DISTINCT)",
            CdAgg::Count { .. } => "COUNT",
            CdAgg::CountStar { .. } => "COUNT(*)",
            CdAgg::Sum { .. } => "SUM",
            CdAgg::Min { .. } => "MIN",
            CdAgg::Max { .. } => "MAX",
            CdAgg::Avg { .. } => "AVG",
        })
        .collect();
    let _ = writeln!(
        out,
        "MultiAggGroupBy: group_keys=[{}] aggs=[{}]",
        cd.group_key_names.join(", "),
        aggs.join(", ")
    );
    indent(&mut out, 1);
    let _ = writeln!(out, "Base:");
    format_logical_into(&cd.base, 2, &mut out);
    if let Some(post) = &cd.post {
        indent(&mut out, 1);
        let _ = writeln!(out, "Post:");
        format_logical_into(post, 2, &mut out);
    }
    out
}

// ---------------------------------------------------------------------------
// Physical plan
// ---------------------------------------------------------------------------

/// Render `plan` as a tree-indented, human-readable string.
///
/// Mirrors [`format_logical`]'s shape for the physical IR: one node per
/// line, two-space indentation per depth, node kind first followed by a
/// compact [`KernelSpec`] summary (op count + input/output column counts)
/// where the variant carries one. Pure and read-only — no GPU, no codegen.
pub fn format_physical(plan: &PhysicalPlan) -> String {
    let mut out = String::new();
    format_physical_into(plan, 0, &mut out);
    out
}

/// One-line summary of a [`KernelSpec`]: op count, input/output column counts,
/// and whether it carries a filter predicate.
fn format_kernel(kernel: &KernelSpec) -> String {
    let pred = if kernel.predicate.is_some() {
        " predicate=yes"
    } else {
        ""
    };
    format!(
        "ops={} inputs={} outputs={}{pred}",
        kernel.ops.len(),
        kernel.inputs.len(),
        kernel.outputs.len()
    )
}

/// Recursive worker for [`format_physical`].
fn format_physical_into(plan: &PhysicalPlan, depth: usize, out: &mut String) {
    indent(out, depth);
    match plan {
        PhysicalPlan::StringLength { table, .. } => {
            let _ = writeln!(out, "StringLength: table={table}");
        }
        PhysicalPlan::StringProject { table, .. } => {
            let _ = writeln!(out, "StringProject: table={table}");
        }
        PhysicalPlan::StringLikeFilter {
            input,
            table,
            column,
            mode,
            negated,
            ..
        } => {
            let kw = if *negated { "NOT LIKE" } else { "LIKE" };
            let _ = writeln!(
                out,
                "StringLikeFilter: table={table} {column} {kw} [{mode:?}] (GPU, unvalidated)"
            );
            format_physical_into(input, depth + 1, out);
        }
        PhysicalPlan::Window { input, .. } => {
            let _ = writeln!(out, "Window");
            format_physical_into(input, depth + 1, out);
        }
        PhysicalPlan::Projection { table, kernel, .. } => {
            let _ = writeln!(
                out,
                "Projection: table={table} kernel({})",
                format_kernel(kernel)
            );
        }
        PhysicalPlan::Aggregate {
            table,
            pre,
            aggregate,
        } => {
            let pre_str = match pre {
                Some(k) => format!(" pre=kernel({})", format_kernel(k)),
                None => String::new(),
            };
            let aggs: Vec<String> = aggregate
                .aggregates
                .iter()
                .map(format_aggregate)
                .collect();
            let _ = writeln!(
                out,
                "Aggregate: table={table} group_keys={} aggs=[{}]{pre_str}",
                aggregate.group_by.len(),
                aggs.join(", ")
            );
        }
        PhysicalPlan::Distinct { input } => {
            let _ = writeln!(out, "Distinct");
            format_physical_into(input, depth + 1, out);
        }
        PhysicalPlan::CountRows { input, .. } => {
            let _ = writeln!(out, "CountRows");
            format_physical_into(input, depth + 1, out);
        }
        PhysicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            let _ = writeln!(out, "Limit: limit={limit} offset={offset}");
            format_physical_into(input, depth + 1, out);
        }
        PhysicalPlan::Sort { input, sort_exprs } => {
            let keys: Vec<String> = sort_exprs.iter().map(format_sort_expr).collect();
            let _ = writeln!(out, "Sort: [{}]", keys.join(", "));
            format_physical_into(input, depth + 1, out);
        }
        PhysicalPlan::Union { inputs } => {
            let _ = writeln!(out, "Union");
            for child in inputs {
                format_physical_into(child, depth + 1, out);
            }
        }
        PhysicalPlan::SetOp { left, right, op, all } => {
            let all_str = if *all { " all" } else { "" };
            let _ = writeln!(out, "SetOp: op={}{all_str}", op.keyword());
            format_physical_into(left, depth + 1, out);
            format_physical_into(right, depth + 1, out);
        }
        PhysicalPlan::Project {
            input, exprs, ..
        } => {
            let rendered: Vec<String> = exprs.iter().map(format_expr).collect();
            let _ = writeln!(out, "Project: [{}]", rendered.join(", "));
            format_physical_into(input, depth + 1, out);
        }
        PhysicalPlan::Filter { input, predicate } => {
            let _ = writeln!(out, "Filter: {}", format_expr(predicate));
            format_physical_into(input, depth + 1, out);
        }
        PhysicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
            ..
        } => {
            let on_str = if on.is_empty() {
                String::new()
            } else {
                format!(" on=[{}]", format_join_on(on))
            };
            let filt = match filter {
                Some(f) => format!(" filter={}", format_expr(f)),
                None => String::new(),
            };
            let _ = writeln!(out, "Join: type={}{on_str}{filt}", join_type_str(*join_type));
            format_physical_into(left, depth + 1, out);
            format_physical_into(right, depth + 1, out);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{col, lit, Field, Schema};

    /// Two-column scan fixture: `t(a Int64, b Int64)`.
    fn scan_t() -> LogicalPlan {
        LogicalPlan::Scan {
            table: "t".to_string(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("a", DataType::Int64, false),
                Field::new("b", DataType::Int64, true),
            ]),
        }
    }

    #[test]
    fn expr_binary_is_parenthesised() {
        let e = col("a").add(lit(1i64)).gt(lit(10i64));
        assert_eq!(format_expr(&e), "((a + 1) > 10)");
    }

    #[test]
    fn expr_unary_and_alias() {
        assert_eq!(format_expr(&col("a").is_null()), "(a IS NULL)");
        assert_eq!(format_expr(&col("a").is_not_null()), "(a IS NOT NULL)");
        assert_eq!(
            format_expr(&col("a").add(lit(1i64)).alias("x")),
            "(a + 1) AS x"
        );
    }

    #[test]
    fn expr_string_literal_quoted() {
        assert_eq!(format_expr(&lit("hi")), "'hi'");
    }

    #[test]
    fn logical_scan_with_projection() {
        let plan = LogicalPlan::Scan {
            table: "t".to_string(),
            projection: Some(vec!["a".to_string(), "b".to_string()]),
            schema: Schema::new(vec![Field::new("a", DataType::Int64, false)]),
        };
        let out = format_logical(&plan);
        assert_eq!(out, "Scan: table=t projection=[a, b]\n");
    }

    #[test]
    fn logical_filter_project_scan_indentation() {
        let plan = LogicalPlan::Project {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(scan_t()),
                predicate: col("a").gt(lit(5i64)),
            }),
            exprs: vec![col("a"), col("b").alias("bb")],
        };
        let out = format_logical(&plan);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "Projection: [a, b AS bb]");
        assert_eq!(lines[1], "  Filter: (a > 5)");
        assert_eq!(lines[2], "    Scan: table=t");
    }

    #[test]
    fn logical_aggregate_renders_group_by_and_aggs() {
        let plan = LogicalPlan::Aggregate {
            input: Box::new(scan_t()),
            group_by: vec![col("a")],
            aggregates: vec![
                AggregateExpr::Count(col("b")),
                AggregateExpr::Sum(col("a")),
            ],
        };
        let out = format_logical(&plan);
        let first = out.lines().next().unwrap();
        assert_eq!(
            first,
            "Aggregate: group_by=[a] aggs=[COUNT(b), SUM(a)]"
        );
        // child scan is indented one level.
        assert!(out.contains("\n  Scan: table=t"));
    }

    #[test]
    fn logical_sort_limit_distinct() {
        let plan = LogicalPlan::Limit {
            input: Box::new(LogicalPlan::Sort {
                input: Box::new(LogicalPlan::Distinct {
                    input: Box::new(scan_t()),
                }),
                sort_exprs: vec![SortExpr {
                    expr: col("a"),
                    descending: true,
                    nulls_first: false,
                }],
            }),
            limit: 10,
            offset: 5,
        };
        let out = format_logical(&plan);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "Limit: limit=10 offset=5");
        assert_eq!(lines[1], "  Sort: [a DESC NULLS LAST]");
        assert_eq!(lines[2], "    Distinct");
        assert_eq!(lines[3], "      Scan: table=t");
    }

    #[test]
    fn logical_join_renders_type_and_on() {
        let plan = LogicalPlan::Join {
            left: Box::new(scan_t()),
            right: Box::new(scan_t()),
            join_type: JoinType::Inner,
            on: vec![(col("a"), col("a"))],
            filter: None,
        };
        let out = format_logical(&plan);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "Join: type=Inner on=[a = a]");
        assert_eq!(lines[1], "  Scan: table=t");
        assert_eq!(lines[2], "  Scan: table=t");
    }

    #[test]
    fn logical_union_renders_branches() {
        let plan = LogicalPlan::Union {
            inputs: vec![scan_t(), scan_t()],
        };
        let out = format_logical(&plan);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "Union");
        assert_eq!(lines[1], "  Scan: table=t");
        assert_eq!(lines[2], "  Scan: table=t");
    }

    #[test]
    fn physical_projection_summary() {
        let plan = crate::plan::physical_plan::lower(&LogicalPlan::Project {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(scan_t()),
                predicate: col("a").gt(lit(5i64)),
            }),
            exprs: vec![col("a"), col("b")],
        })
        .expect("lowering a simple filter+project should succeed");
        let out = format_physical(&plan);
        // Single fused Projection kernel; carries a predicate from the filter.
        assert!(out.starts_with("Projection: table=t kernel("));
        assert!(out.contains("predicate=yes"), "got: {out}");
        assert!(out.contains("inputs="));
        assert!(out.contains("outputs="));
    }

    #[test]
    fn recursive_cte_renders_anchor_recursive_main() {
        use crate::plan::logical_plan::Field;
        use crate::plan::sql_frontend::RecursiveCtePlan;
        let cte_schema = Schema::new(vec![Field::new("n", DataType::Int64, true)]);
        // Minimal stand-in subplans (the renderer only walks them structurally).
        let scan = LogicalPlan::Scan {
            table: "seq".to_string(),
            projection: None,
            schema: cte_schema.clone(),
        };
        let rec = RecursiveCtePlan {
            name: "seq".to_string(),
            cte_schema,
            anchor: scan.clone(),
            recursive: scan.clone(),
            all: true,
            naive: false,
            main: scan,
        };
        let out = format_recursive_cte(&rec);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "RecursiveCte: name=seq (n) UNION ALL");
        assert_eq!(lines[1], "  Anchor:");
        assert_eq!(lines[2], "    Scan: table=seq");
        assert_eq!(lines[3], "  Recursive:");
        assert_eq!(lines[5], "  Main:");
    }

    #[test]
    fn mutual_recursive_cte_renders_each_member() {
        use crate::plan::logical_plan::Field;
        use crate::plan::sql_frontend::{MutualRecursiveCtePlan, RecursiveCteTerm};
        let schema = Schema::new(vec![Field::new("n", DataType::Int64, true)]);
        let scan = LogicalPlan::Scan {
            table: "a".to_string(),
            projection: None,
            schema: schema.clone(),
        };
        let rec = MutualRecursiveCtePlan {
            ctes: vec![
                RecursiveCteTerm {
                    name: "a".to_string(),
                    cte_schema: schema.clone(),
                    anchor: scan.clone(),
                    recursive: Some(scan.clone()),
                    all: false,
                },
                RecursiveCteTerm {
                    name: "b".to_string(),
                    cte_schema: schema.clone(),
                    anchor: scan.clone(),
                    recursive: None,
                    all: false,
                },
            ],
            main: scan,
        };
        let out = format_mutual_recursive_cte(&rec);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "MutualRecursiveCte: 2 CTEs");
        assert_eq!(lines[1], "  Cte: a (n) recursive UNION");
        assert!(lines.iter().any(|l| *l == "  Cte: b (n) non-recursive"));
        assert!(lines.iter().any(|l| l.trim() == "Main:"));
    }

    #[test]
    fn count_distinct_groupby_renders_base_and_post() {
        use crate::plan::logical_plan::Field;
        use crate::plan::sql_frontend::CountDistinctGroupByPlan;
        let result_schema = Schema::new(vec![
            Field::new("region", DataType::Int64, true),
            Field::new("cnt", DataType::Int64, false),
        ]);
        let base = LogicalPlan::Scan {
            table: "sales".to_string(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("region", DataType::Int64, true),
                Field::new("customer", DataType::Int64, true),
            ]),
        };
        let cd = CountDistinctGroupByPlan {
            base,
            group_key_names: vec!["region".to_string()],
            count_alias: "cnt".to_string(),
            result_schema: result_schema.clone(),
            post: Some(LogicalPlan::Limit {
                input: Box::new(LogicalPlan::Scan {
                    table: "__count_distinct_groupby_result".to_string(),
                    projection: None,
                    schema: result_schema,
                }),
                limit: 5,
                offset: 0,
            }),
        };
        let out = format_count_distinct_groupby(&cd);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[0],
            "CountDistinctGroupBy: group_keys=[region] count=COUNT(DISTINCT) AS cnt"
        );
        assert_eq!(lines[1], "  Base:");
        assert_eq!(lines[2], "    Scan: table=sales");
        assert_eq!(lines[3], "  Post:");
        assert_eq!(lines[4], "    Limit: limit=5 offset=0");
    }

    #[test]
    fn physical_aggregate_summary() {
        let plan = crate::plan::physical_plan::lower(&LogicalPlan::Aggregate {
            input: Box::new(scan_t()),
            group_by: vec![col("a")],
            aggregates: vec![AggregateExpr::Sum(col("b"))],
        })
        .expect("lowering a group-by aggregate should succeed");
        let out = format_physical(&plan);
        assert!(out.starts_with("Aggregate: table=t"));
        assert!(out.contains("group_keys=1"));
        assert!(out.contains("SUM(b)"), "got: {out}");
    }
}
