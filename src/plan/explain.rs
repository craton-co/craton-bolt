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
            ..
        } => {
            let kw = if *negated { "NOT LIKE" } else { "LIKE" };
            format!("({} {} '{}')", format_expr(expr), kw, pattern)
        }
        Expr::Cast { expr, target } => {
            format!("CAST({} AS {})", format_expr(expr), format_dtype(*target))
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
