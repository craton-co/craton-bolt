// SPDX-License-Identifier: AGPL-3.0-only

//! Engine-wide SQL semantics mode and the plan transforms behind its `Standard` mode.
//!
//! The mode is chosen once with
//! [`EngineBuilder::sql_semantics`](crate::EngineBuilder::sql_semantics).
//! [`SqlSemantics::Standard`] is implemented on top of the IEEE / wrapping
//! kernels every tier already has:
//!
//! * [`total_order_comparisons`] rewrites every float comparison into IEEE
//!   comparisons that give the total order (NaN equal to every NaN and
//!   greater than every other value). `isnan(x)` is spelled `NOT (x = x)`,
//!   which is UNKNOWN for a NULL `x`, so every rewrite keeps SQL's
//!   three-valued logic: a comparison with a NULL operand stays UNKNOWN.
//!   Because the output uses only `=`, `<`, `AND`, `OR` and `NOT`, it runs
//!   unchanged on the GPU kernels and the host evaluator.
//! * [`overflow_checks`] derives, for every plan node whose integer `+ - *`
//!   or narrowing `CAST` may overflow (interval analysis over the operand
//!   types and literals cannot rule it out), a check plan
//!   `LIMIT 1 (FILTER <overflow condition> (<node input>))`. The engine runs
//!   the checks before the statement and fails it with an "out of range"
//!   error when any check returns a row. The conditions are written so that
//!   they do not overflow themselves (or only in sub-terms whose result is
//!   discarded by an `AND` with a FALSE guard), so they evaluate correctly
//!   under the wrapping arithmetic of every tier.
//!
//! The default-NULL-order part of the mode is applied by the SQL frontend
//! (see [`TableProvider::default_nulls_last`](crate::plan::TableProvider::default_nulls_last)).
//!
//! # Stability
//!
//! Experimental until 1.0.

use crate::error::BoltResult;
use crate::plan::explain::format_expr;
use crate::plan::logical_plan::{
    join_combined_schema, AggregateExpr, BinaryOp, DataType, Expr, JoinType, Literal, LogicalPlan,
    Schema, UnaryOp, WindowFunc,
};

/// Which semantics the engine gives float comparisons, integer overflow and
/// the default `NULL` placement of `ORDER BY`.
///
/// See `docs/SQL_REFERENCE.md` ("Semantics modes") for the exact rules.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SqlSemantics {
    /// The historical behaviour (the default): IEEE-754 float comparisons
    /// (every comparison with NaN is FALSE, `!=` included), integer `+ - *`
    /// and narrowing `CAST` wrap on overflow (two's complement), and `ASC`
    /// sorts NULLs first.
    #[default]
    Ieee,
    /// DuckDB / PostgreSQL compatible: floats are totally ordered (every NaN
    /// equals every NaN and is greater than all other values, `!=` is the
    /// negation of `=`), integer `+ - *` and narrowing `CAST` raise an
    /// "out of range" error on overflow, and `ORDER BY` puts NULLs last
    /// unless `NULLS FIRST` is written.
    Standard,
}

// ---------------------------------------------------------------------------
// Plan walking
// ---------------------------------------------------------------------------

/// Rebuild `plan` with `f(expr, schema)` applied to every scalar expression
/// of every node, where `schema` is the schema the expression is evaluated
/// against. Subquery plans embedded in expressions are left alone (they are
/// compiled, and transformed, on their own when they are resolved).
pub(crate) fn map_exprs_with_schema(
    plan: &LogicalPlan,
    f: &dyn Fn(&Expr, &Schema) -> BoltResult<Expr>,
) -> BoltResult<LogicalPlan> {
    let rec = |p: &LogicalPlan| map_exprs_with_schema(p, f).map(Box::new);
    let map_agg = |a: &AggregateExpr, s: &Schema| -> BoltResult<AggregateExpr> {
        Ok(match a {
            AggregateExpr::Count(e) => AggregateExpr::Count(f(e, s)?),
            AggregateExpr::Sum(e) => AggregateExpr::Sum(f(e, s)?),
            AggregateExpr::Min(e) => AggregateExpr::Min(f(e, s)?),
            AggregateExpr::Max(e) => AggregateExpr::Max(f(e, s)?),
            AggregateExpr::Avg(e) => AggregateExpr::Avg(f(e, s)?),
            AggregateExpr::VarPop(e) => AggregateExpr::VarPop(Box::new(f(e, s)?)),
            AggregateExpr::VarSamp(e) => AggregateExpr::VarSamp(Box::new(f(e, s)?)),
            AggregateExpr::StddevPop(e) => AggregateExpr::StddevPop(Box::new(f(e, s)?)),
            AggregateExpr::StddevSamp(e) => AggregateExpr::StddevSamp(Box::new(f(e, s)?)),
        })
    };
    let map_sort = |v: &[crate::plan::logical_plan::SortExpr], s: &Schema| {
        v.iter()
            .map(|se| {
                let mut se = se.clone();
                se.expr = f(&se.expr, s)?;
                Ok(se)
            })
            .collect::<BoltResult<Vec<_>>>()
    };
    Ok(match plan {
        LogicalPlan::Scan { .. } => plan.clone(),
        LogicalPlan::Filter { input, predicate } => {
            let s = input.schema()?;
            LogicalPlan::Filter {
                input: rec(input)?,
                predicate: f(predicate, &s)?,
            }
        }
        LogicalPlan::Project { input, exprs } => {
            let s = input.schema()?;
            LogicalPlan::Project {
                input: rec(input)?,
                exprs: exprs.iter().map(|e| f(e, &s)).collect::<BoltResult<_>>()?,
            }
        }
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            let s = input.schema()?;
            LogicalPlan::Aggregate {
                input: rec(input)?,
                group_by: group_by
                    .iter()
                    .map(|e| f(e, &s))
                    .collect::<BoltResult<_>>()?,
                aggregates: aggregates
                    .iter()
                    .map(|a| map_agg(a, &s))
                    .collect::<BoltResult<_>>()?,
            }
        }
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct { input: rec(input)? },
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => LogicalPlan::Limit {
            input: rec(input)?,
            limit: *limit,
            offset: *offset,
        },
        LogicalPlan::Sort { input, sort_exprs } => {
            let s = input.schema()?;
            LogicalPlan::Sort {
                input: rec(input)?,
                sort_exprs: map_sort(sort_exprs, &s)?,
            }
        }
        LogicalPlan::Window {
            input,
            window_exprs,
            partition_by,
            order_by,
        } => {
            let s = input.schema()?;
            let window_exprs = window_exprs
                .iter()
                .map(|w| {
                    let mut w = w.clone();
                    w.func = match &w.func {
                        WindowFunc::Sum(e) => WindowFunc::Sum(f(e, &s)?),
                        WindowFunc::Avg(e) => WindowFunc::Avg(f(e, &s)?),
                        WindowFunc::Min(e) => WindowFunc::Min(f(e, &s)?),
                        WindowFunc::Max(e) => WindowFunc::Max(f(e, &s)?),
                        WindowFunc::Count(e) => WindowFunc::Count(f(e, &s)?),
                        other => other.clone(),
                    };
                    Ok(w)
                })
                .collect::<BoltResult<_>>()?;
            LogicalPlan::Window {
                input: rec(input)?,
                window_exprs,
                partition_by: partition_by
                    .iter()
                    .map(|e| f(e, &s))
                    .collect::<BoltResult<_>>()?,
                order_by: map_sort(order_by, &s)?,
            }
        }
        LogicalPlan::Union { inputs } => LogicalPlan::Union {
            inputs: inputs
                .iter()
                .map(|i| map_exprs_with_schema(i, f))
                .collect::<BoltResult<_>>()?,
        },
        LogicalPlan::SetOp {
            left,
            right,
            op,
            all,
        } => LogicalPlan::SetOp {
            left: rec(left)?,
            right: rec(right)?,
            op: *op,
            all: *all,
        },
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => {
            let (ls, rs) = (left.schema()?, right.schema()?);
            let combined = join_combined_schema(&ls, &rs, *join_type);
            LogicalPlan::Join {
                left: rec(left)?,
                right: rec(right)?,
                join_type: *join_type,
                on: on
                    .iter()
                    .map(|(l, r)| Ok((f(l, &ls)?, f(r, &rs)?)))
                    .collect::<BoltResult<_>>()?,
                filter: filter.as_ref().map(|e| f(e, &combined)).transpose()?,
            }
        }
    })
}

/// Rebuild `e` with `f` applied to each direct child expression. Subquery
/// plans are not entered; the probe of an `IN (SELECT ...)` is.
pub(crate) fn map_children(
    e: &Expr,
    f: &mut dyn FnMut(&Expr) -> BoltResult<Expr>,
) -> BoltResult<Expr> {
    let bx = |f: &mut dyn FnMut(&Expr) -> BoltResult<Expr>, e: &Expr| f(e).map(Box::new);
    Ok(match e {
        Expr::Column(_) | Expr::Literal(_) | Expr::ScalarSubquery(_) => e.clone(),
        Expr::Binary { op, left, right } => Expr::Binary {
            op: *op,
            left: bx(f, left)?,
            right: bx(f, right)?,
        },
        Expr::Unary { op, operand } => Expr::Unary {
            op: *op,
            operand: bx(f, operand)?,
        },
        Expr::Case {
            branches,
            else_branch,
        } => Expr::Case {
            branches: branches
                .iter()
                .map(|(w, t)| Ok((f(w)?, f(t)?)))
                .collect::<BoltResult<_>>()?,
            else_branch: match else_branch {
                Some(x) => Some(bx(f, x)?),
                None => None,
            },
        },
        Expr::Like {
            expr,
            pattern,
            escape,
            negated,
            case_insensitive,
        } => Expr::Like {
            expr: bx(f, expr)?,
            pattern: pattern.clone(),
            escape: *escape,
            negated: *negated,
            case_insensitive: *case_insensitive,
        },
        Expr::Cast { expr, target, safe } => Expr::Cast {
            expr: bx(f, expr)?,
            target: *target,
            safe: *safe,
        },
        Expr::CastFormat {
            expr,
            target,
            pattern,
            to_text,
        } => Expr::CastFormat {
            expr: bx(f, expr)?,
            target: *target,
            pattern: pattern.clone(),
            to_text: *to_text,
        },
        Expr::ScalarFn { kind, args } => Expr::ScalarFn {
            kind: *kind,
            args: args.iter().map(&mut *f).collect::<BoltResult<_>>()?,
        },
        Expr::Extract { field, expr } => Expr::Extract {
            field: *field,
            expr: bx(f, expr)?,
        },
        Expr::DateTrunc { unit, expr } => Expr::DateTrunc {
            unit: *unit,
            expr: bx(f, expr)?,
        },
        Expr::Alias(x, name) => Expr::Alias(bx(f, x)?, name.clone()),
        Expr::InSubquery {
            expr,
            subquery,
            negated,
        } => Expr::InSubquery {
            expr: bx(f, expr)?,
            subquery: subquery.clone(),
            negated: *negated,
        },
    })
}

/// The direct child expressions of `e` (subquery plans excluded).
fn children(e: &Expr) -> Vec<&Expr> {
    match e {
        Expr::Column(_) | Expr::Literal(_) | Expr::ScalarSubquery(_) => vec![],
        Expr::Binary { left, right, .. } => vec![left, right],
        Expr::Unary { operand, .. } => vec![operand],
        Expr::Case {
            branches,
            else_branch,
        } => branches
            .iter()
            .flat_map(|(w, t)| [w, t])
            .chain(else_branch.as_deref())
            .collect(),
        Expr::Like { expr, .. }
        | Expr::Cast { expr, .. }
        | Expr::CastFormat { expr, .. }
        | Expr::Extract { expr, .. }
        | Expr::DateTrunc { expr, .. }
        | Expr::InSubquery { expr, .. } => vec![expr],
        Expr::ScalarFn { args, .. } => args.iter().collect(),
        Expr::Alias(x, _) => vec![x],
    }
}

// ---------------------------------------------------------------------------
// Expression builders
// ---------------------------------------------------------------------------

fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(l),
        right: Box::new(r),
    }
}

fn not(e: Expr) -> Expr {
    match e {
        Expr::Literal(Literal::Bool(b)) => Expr::Literal(Literal::Bool(!b)),
        e => Expr::Unary {
            op: UnaryOp::Not,
            operand: Box::new(e),
        },
    }
}

fn is_bool(e: &Expr, v: bool) -> bool {
    matches!(e, Expr::Literal(Literal::Bool(b)) if *b == v)
}

/// Kleene `AND`, simplified only where that is exact for every input
/// (`FALSE AND x` is FALSE even for an UNKNOWN `x`; `TRUE AND x` is `x`).
fn and(l: Expr, r: Expr) -> Expr {
    if is_bool(&l, false) || is_bool(&r, false) {
        return Expr::Literal(Literal::Bool(false));
    }
    if is_bool(&l, true) {
        return r;
    }
    if is_bool(&r, true) {
        return l;
    }
    bin(BinaryOp::And, l, r)
}

/// Kleene `OR`, simplified only where that is exact (`FALSE OR x` is `x`).
fn or(l: Expr, r: Expr) -> Expr {
    if is_bool(&l, true) || is_bool(&r, true) {
        return Expr::Literal(Literal::Bool(true));
    }
    if is_bool(&l, false) {
        return r;
    }
    if is_bool(&r, false) {
        return l;
    }
    bin(BinaryOp::Or, l, r)
}

fn any(v: impl IntoIterator<Item = Expr>) -> Option<Expr> {
    v.into_iter().reduce(or)
}

// ---------------------------------------------------------------------------
// Total-order float comparisons
// ---------------------------------------------------------------------------

/// Rewrite every float comparison of `plan` (`=`, `<>`, `<`, `<=`, `>`,
/// `>=` with a `Float32` / `Float64` operand) into IEEE comparisons with the
/// total-order meaning of [`SqlSemantics::Standard`]. Apply exactly once per
/// plan: the output's own `x = x` tests are IEEE tests.
pub(crate) fn total_order_comparisons(plan: &LogicalPlan) -> BoltResult<LogicalPlan> {
    map_exprs_with_schema(plan, &|e, s| total_order_expr(e, s))
}

/// [`total_order_comparisons`] for one expression evaluated against `schema`.
pub(crate) fn total_order_expr(e: &Expr, schema: &Schema) -> BoltResult<Expr> {
    let rewritten = map_children(e, &mut |c| total_order_expr(c, schema))?;
    let Expr::Binary { op, left, right } = &rewritten else {
        return Ok(rewritten);
    };
    use BinaryOp::*;
    if !matches!(op, Eq | NotEq | Lt | LtEq | Gt | GtEq) {
        return Ok(rewritten);
    }
    let Expr::Binary {
        left: orig_l,
        right: orig_r,
        ..
    } = e
    else {
        return Ok(rewritten);
    };
    let is_float = |x: &Expr| matches!(x.dtype(schema), Ok(DataType::Float32 | DataType::Float64));
    let null_lit = |x: &Expr| matches!(x, Expr::Literal(Literal::Null));
    if !(is_float(orig_l) || is_float(orig_r)) || null_lit(orig_l) || null_lit(orig_r) {
        return Ok(rewritten);
    }
    let float_l = is_float(orig_l);
    let float_r = is_float(orig_r);
    let (l, r) = ((**left).clone(), (**right).clone());
    Ok(match op {
        Eq => total_eq(l, float_l, r, float_r),
        NotEq => not(total_eq(l, float_l, r, float_r)),
        Lt => total_lt(l, r, float_r),
        Gt => total_lt(r, l, float_l),
        LtEq => not(total_lt(r, l, float_l)),
        GtEq => not(total_lt(l, r, float_r)),
        _ => unreachable!("filtered to comparisons above"),
    })
}

/// `isnan(x)` as an IEEE expression: UNKNOWN for NULL, exact for literals,
/// FALSE for a non-float operand (which can never be NaN).
fn is_nan(x: &Expr, float: bool) -> Expr {
    match x {
        Expr::Literal(Literal::Float64(v)) => Expr::Literal(Literal::Bool(v.is_nan())),
        Expr::Literal(Literal::Float32(v)) => Expr::Literal(Literal::Bool(v.is_nan())),
        _ if !float => Expr::Literal(Literal::Bool(false)),
        _ => not(bin(BinaryOp::Eq, x.clone(), x.clone())),
    }
}

/// `NOT isnan(x)` that is still UNKNOWN for a NULL `x` (so it cannot turn an
/// UNKNOWN comparison TRUE): `x = x` unless `x` is a non-NULL literal.
fn not_nan(x: &Expr) -> Expr {
    match x {
        Expr::Literal(Literal::Float64(v)) => Expr::Literal(Literal::Bool(!v.is_nan())),
        Expr::Literal(Literal::Float32(v)) => Expr::Literal(Literal::Bool(!v.is_nan())),
        Expr::Literal(_) => Expr::Literal(Literal::Bool(true)),
        _ => bin(BinaryOp::Eq, x.clone(), x.clone()),
    }
}

/// Total-order `x = y`: `(x = y) OR (isnan(x) AND isnan(y))`.
fn total_eq(x: Expr, fx: bool, y: Expr, fy: bool) -> Expr {
    let both_nan = and(is_nan(&x, fx), is_nan(&y, fy));
    or(bin(BinaryOp::Eq, x, y), both_nan)
}

/// Total-order `x < y`: `(x < y) OR (NOT isnan(x) AND isnan(y))`.
fn total_lt(x: Expr, y: Expr, fy: bool) -> Expr {
    let nan_above = and(not_nan(&x), is_nan(&y, fy));
    or(bin(BinaryOp::Lt, x, y), nan_above)
}

// ---------------------------------------------------------------------------
// Checked integer arithmetic
// ---------------------------------------------------------------------------

/// One overflow check of a statement: `plan` returns at least one row iff
/// some evaluated row overflows one of `operations`.
#[derive(Debug, Clone)]
pub(crate) struct OverflowCheck {
    /// `LIMIT 1 (FILTER <overflow> (<node input>))`.
    pub(crate) plan: LogicalPlan,
    /// The operations that may overflow, rendered for the error message.
    pub(crate) operations: Vec<String>,
}

impl OverflowCheck {
    /// The error a failing check raises.
    pub(crate) fn error(&self) -> crate::error::BoltError {
        crate::error::BoltError::Other(format!(
            "Out of Range Error: integer overflow (value out of range) in {}",
            self.operations.join(", ")
        ))
    }
}

/// The overflow checks `plan` needs under [`SqlSemantics::Standard`]: one per
/// node whose integer `+ - *` or narrowing `CAST` interval analysis cannot
/// prove in range. Empty when nothing can overflow.
pub(crate) fn overflow_checks(plan: &LogicalPlan) -> BoltResult<Vec<OverflowCheck>> {
    let mut out = Vec::new();
    collect_checks(plan, &mut out)?;
    Ok(out)
}

fn push_check(
    out: &mut Vec<OverflowCheck>,
    input: &LogicalPlan,
    exprs: &[&Expr],
) -> BoltResult<()> {
    let schema = input.schema()?;
    let mut operations = Vec::new();
    let mut conds = Vec::new();
    for e in exprs {
        if let Some(c) = overflow_condition(e, &schema, &mut operations)? {
            conds.push(c);
        }
    }
    if let Some(cond) = any(conds) {
        out.push(OverflowCheck {
            plan: LogicalPlan::Limit {
                input: Box::new(LogicalPlan::Filter {
                    input: Box::new(input.clone()),
                    predicate: cond,
                }),
                limit: 1,
                offset: 0,
            },
            operations,
        });
    }
    Ok(())
}

fn agg_arg(a: &AggregateExpr) -> &Expr {
    match a {
        AggregateExpr::Count(e)
        | AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e) => e,
        AggregateExpr::VarPop(e)
        | AggregateExpr::VarSamp(e)
        | AggregateExpr::StddevPop(e)
        | AggregateExpr::StddevSamp(e) => e,
    }
}

fn collect_checks(plan: &LogicalPlan, out: &mut Vec<OverflowCheck>) -> BoltResult<()> {
    match plan {
        LogicalPlan::Scan { .. } => {}
        LogicalPlan::Filter { input, predicate } => {
            collect_checks(input, out)?;
            push_check(out, input, &[predicate])?;
        }
        LogicalPlan::Project { input, exprs } => {
            collect_checks(input, out)?;
            push_check(out, input, &exprs.iter().collect::<Vec<_>>())?;
        }
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            collect_checks(input, out)?;
            let exprs: Vec<&Expr> = group_by
                .iter()
                .chain(aggregates.iter().map(agg_arg))
                .collect();
            push_check(out, input, &exprs)?;
        }
        LogicalPlan::Distinct { input } | LogicalPlan::Limit { input, .. } => {
            collect_checks(input, out)?;
        }
        LogicalPlan::Sort { input, sort_exprs } => {
            collect_checks(input, out)?;
            push_check(
                out,
                input,
                &sort_exprs.iter().map(|s| &s.expr).collect::<Vec<_>>(),
            )?;
        }
        LogicalPlan::Window {
            input,
            window_exprs,
            partition_by,
            order_by,
        } => {
            collect_checks(input, out)?;
            let exprs: Vec<&Expr> = window_exprs
                .iter()
                .filter_map(|w| match &w.func {
                    WindowFunc::Sum(e)
                    | WindowFunc::Avg(e)
                    | WindowFunc::Min(e)
                    | WindowFunc::Max(e)
                    | WindowFunc::Count(e) => Some(e),
                    _ => None,
                })
                .chain(partition_by.iter())
                .chain(order_by.iter().map(|s| &s.expr))
                .collect();
            push_check(out, input, &exprs)?;
        }
        LogicalPlan::Union { inputs } => {
            for i in inputs {
                collect_checks(i, out)?;
            }
        }
        LogicalPlan::SetOp { left, right, .. } => {
            collect_checks(left, out)?;
            collect_checks(right, out)?;
        }
        LogicalPlan::Join {
            left,
            right,
            on,
            filter,
            ..
        } => {
            collect_checks(left, out)?;
            collect_checks(right, out)?;
            // Equi-join keys are evaluated on every row of their side.
            push_check(out, left, &on.iter().map(|(l, _)| l).collect::<Vec<_>>())?;
            push_check(out, right, &on.iter().map(|(_, r)| r).collect::<Vec<_>>())?;
            // The residual is evaluated on the pairs the equi-keys match (all
            // pairs without keys): the inner join of the same inputs.
            if let Some(f) = filter {
                let pairs = LogicalPlan::Join {
                    left: left.clone(),
                    right: right.clone(),
                    join_type: if on.is_empty() {
                        JoinType::Cross
                    } else {
                        JoinType::Inner
                    },
                    on: on.clone(),
                    filter: None,
                };
                push_check(out, &pairs, &[f])?;
            }
        }
    }
    Ok(())
}

/// Closed integer interval, wide enough for any `i64 op i64` result.
type Interval = (i128, i128);

fn type_range(t: DataType) -> Option<Interval> {
    match t {
        DataType::Int32 => Some((i32::MIN as i128, i32::MAX as i128)),
        DataType::Int64 => Some((i64::MIN as i128, i64::MAX as i128)),
        _ => None,
    }
}

fn within(i: Interval, r: Interval) -> bool {
    i.0 >= r.0 && i.1 <= r.1
}

/// Interval arithmetic for `l op r`.
fn op_interval(op: BinaryOp, l: Interval, r: Interval) -> Interval {
    match op {
        BinaryOp::Add => (l.0 + r.0, l.1 + r.1),
        BinaryOp::Sub => (l.0 - r.1, l.1 - r.0),
        _ => {
            let p = [l.0 * r.0, l.0 * r.1, l.1 * r.0, l.1 * r.1];
            (
                *p.iter().min().expect("4 products"),
                *p.iter().max().expect("4 products"),
            )
        }
    }
}

/// A sound over-approximation of the values an integer expression can take
/// once it has been checked (so an operation that may overflow contributes
/// its full type range). `None` for a non-integer expression.
fn interval(e: &Expr, schema: &Schema) -> Option<Interval> {
    match e {
        Expr::Literal(Literal::Int32(v)) => Some((*v as i128, *v as i128)),
        Expr::Literal(Literal::Int64(v)) => Some((*v as i128, *v as i128)),
        Expr::Alias(x, _) => interval(x, schema),
        Expr::Binary { op, left, right }
            if matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul) =>
        {
            let range = type_range(e.dtype(schema).ok()?)?;
            match (interval(left, schema), interval(right, schema)) {
                (Some(l), Some(r)) => {
                    let i = op_interval(*op, l, r);
                    Some(if within(i, range) { i } else { range })
                }
                _ => Some(range),
            }
        }
        Expr::Cast {
            expr,
            target,
            safe: false,
        } => {
            let range = type_range(*target)?;
            match interval(expr, schema) {
                Some(i) => Some((i.0.max(range.0), i.1.min(range.1))),
                None => Some(range),
            }
        }
        Expr::Case {
            branches,
            else_branch,
        } => {
            let range = type_range(e.dtype(schema).ok()?)?;
            let mut acc: Option<Interval> = None;
            for v in branches
                .iter()
                .map(|(_, t)| t)
                .chain(else_branch.as_deref())
            {
                if matches!(v, Expr::Literal(Literal::Null)) {
                    continue;
                }
                let i = interval(v, schema).unwrap_or(range);
                acc = Some(match acc {
                    Some(a) => (a.0.min(i.0), a.1.max(i.1)),
                    None => i,
                });
            }
            Some(acc.unwrap_or(range))
        }
        _ => type_range(e.dtype(schema).ok()?),
    }
}

fn int_lit(v: i128, t: DataType) -> Expr {
    match t {
        DataType::Int32 => Expr::Literal(Literal::Int32(v as i32)),
        _ => Expr::Literal(Literal::Int64(v as i64)),
    }
}

fn widen(e: Expr) -> Expr {
    Expr::Cast {
        expr: Box::new(e),
        target: DataType::Int64,
        safe: false,
    }
}

/// Does `l op r` (result type `t`) overflow? Written so that it does not
/// overflow itself, except in sub-terms an `AND` with a FALSE guard discards.
fn op_overflows(op: BinaryOp, l: &Expr, r: &Expr, t: DataType) -> Expr {
    use BinaryOp::*;
    let (min, max) = type_range(t).expect("integer result");
    if t == DataType::Int32 {
        // Exact in Int64: both operands are Int32.
        let w = bin(op, widen(l.clone()), widen(r.clone()));
        return or(
            bin(Lt, w.clone(), int_lit(min, DataType::Int64)),
            bin(Gt, w, int_lit(max, DataType::Int64)),
        );
    }
    let zero = || int_lit(0, t);
    let lit = |v: i128| int_lit(v, t);
    let (l, r) = (l.clone(), r.clone());
    match op {
        // r > 0 AND l > MAX - r  |  r < 0 AND l < MIN - r
        Add => or(
            and(
                bin(Gt, r.clone(), zero()),
                bin(Gt, l.clone(), bin(Sub, lit(max), r.clone())),
            ),
            and(
                bin(Lt, r.clone(), zero()),
                bin(Lt, l, bin(Sub, lit(min), r)),
            ),
        ),
        // r < 0 AND l > MAX + r  |  r > 0 AND l < MIN + r
        Sub => or(
            and(
                bin(Lt, r.clone(), zero()),
                bin(Gt, l.clone(), bin(Add, lit(max), r.clone())),
            ),
            and(
                bin(Gt, r.clone(), zero()),
                bin(Lt, l, bin(Add, lit(min), r)),
            ),
        ),
        // (l = -1 AND r = MIN) | (r = -1 AND l = MIN)
        //   | (l <> 0 AND l <> -1 AND (l * r) / l <> r)
        _ => or(
            or(
                and(bin(Eq, l.clone(), lit(-1)), bin(Eq, r.clone(), lit(min))),
                and(bin(Eq, r.clone(), lit(-1)), bin(Eq, l.clone(), lit(min))),
            ),
            and(
                and(
                    bin(NotEq, l.clone(), zero()),
                    bin(NotEq, l.clone(), lit(-1)),
                ),
                bin(NotEq, bin(Div, bin(Mul, l.clone(), r.clone()), l), r),
            ),
        ),
    }
}

/// "This row's evaluation of `e` overflows", or `None` when interval
/// analysis proves it cannot. NULL (UNKNOWN) counts as "no". Overflow in a
/// `CASE` branch counts only when the branch is taken. The rendered risky
/// operations are appended to `ops`.
fn overflow_condition(
    e: &Expr,
    schema: &Schema,
    ops: &mut Vec<String>,
) -> BoltResult<Option<Expr>> {
    match e {
        Expr::Case {
            branches,
            else_branch,
        } => {
            // Rows reaching branch k: no earlier condition was TRUE.
            let mut reach = Expr::Literal(Literal::Bool(true));
            let mut conds = Vec::new();
            for (w, t) in branches {
                if let Some(c) = overflow_condition(w, schema, ops)? {
                    conds.push(and(reach.clone(), c));
                }
                if let Some(c) = overflow_condition(t, schema, ops)? {
                    conds.push(and(and(reach.clone(), w.clone()), c));
                }
                let not_true = or(
                    Expr::Unary {
                        op: UnaryOp::IsNull,
                        operand: Box::new(w.clone()),
                    },
                    not(w.clone()),
                );
                reach = and(reach, not_true);
            }
            if let Some(x) = else_branch {
                if let Some(c) = overflow_condition(x, schema, ops)? {
                    conds.push(and(reach, c));
                }
            }
            Ok(any(conds))
        }
        _ => {
            let mut conds = Vec::new();
            for c in children(e) {
                if let Some(x) = overflow_condition(c, schema, ops)? {
                    conds.push(x);
                }
            }
            if let Some(own) = own_overflow(e, schema)? {
                ops.push(format_expr(e));
                conds.push(own);
            }
            Ok(any(conds))
        }
    }
}

/// The overflow test of `e`'s own operation (not its children's).
fn own_overflow(e: &Expr, schema: &Schema) -> BoltResult<Option<Expr>> {
    match e {
        Expr::Binary { op, left, right }
            if matches!(op, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul) =>
        {
            let Ok(t) = e.dtype(schema) else {
                return Ok(None);
            };
            let Some(range) = type_range(t) else {
                return Ok(None);
            };
            let (Some(l), Some(r)) = (interval(left, schema), interval(right, schema)) else {
                return Ok(None);
            };
            if within(op_interval(*op, l, r), range) {
                return Ok(None);
            }
            Ok(Some(op_overflows(*op, left, right, t)))
        }
        Expr::Cast {
            expr,
            target,
            safe: false,
        } => {
            let Some(range) = type_range(*target) else {
                return Ok(None);
            };
            let from = expr.dtype(schema).ok();
            if matches!(from, Some(DataType::Float32 | DataType::Float64)) {
                // Every tier truncates toward zero. The check plan is
                // compiled under the total order, where NaN is greater than
                // every bound, so NaN and +inf trip the upper test and -inf
                // the lower one; NULL stays UNKNOWN.
                let x = (**expr).clone();
                let flit = |v: f64| Expr::Literal(Literal::Float64(v));
                return Ok(Some(if *target == DataType::Int32 {
                    or(
                        bin(BinaryOp::LtEq, x.clone(), flit(i32::MIN as f64 - 1.0)),
                        bin(BinaryOp::GtEq, x, flit(i32::MAX as f64 + 1.0)),
                    )
                } else {
                    // -2^63 is exact in f64 and no f64 lies in (-2^63 - 1, -2^63).
                    or(
                        bin(BinaryOp::Lt, x.clone(), flit(i64::MIN as f64)),
                        bin(BinaryOp::GtEq, x, flit(9_223_372_036_854_775_808.0)),
                    )
                }));
            }
            if from != Some(DataType::Int64) {
                return Ok(None);
            }
            match interval(expr, schema) {
                Some(i) if !within(i, range) => Ok(Some(or(
                    bin(
                        BinaryOp::Lt,
                        (**expr).clone(),
                        int_lit(range.0, DataType::Int64),
                    ),
                    bin(
                        BinaryOp::Gt,
                        (**expr).clone(),
                        int_lit(range.1, DataType::Int64),
                    ),
                ))),
                _ => Ok(None),
            }
        }
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{col, lit, Field};

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("d", DataType::Float64, true),
            Field::new("e", DataType::Float64, true),
            Field::new("i", DataType::Int32, true),
            Field::new("j", DataType::Int32, true),
            Field::new("x", DataType::Int64, true),
        ])
    }

    fn rw(e: Expr) -> String {
        format_expr(&total_order_expr(&e, &schema()).unwrap())
    }

    /// Kleene evaluation of a rewritten predicate over one row, `None` is
    /// UNKNOWN. Only the operators the rewrite emits are supported.
    fn eval(e: &Expr, row: &dyn Fn(&str) -> Option<f64>) -> Option<Value> {
        match e {
            Expr::Column(c) => row(c).map(Value::F),
            Expr::Literal(Literal::Float64(v)) => Some(Value::F(*v)),
            Expr::Literal(Literal::Int64(v)) => Some(Value::F(*v as f64)),
            Expr::Literal(Literal::Bool(b)) => Some(Value::B(*b)),
            Expr::Unary {
                op: UnaryOp::Not,
                operand,
            } => eval(operand, row).map(|v| Value::B(!v.b())),
            Expr::Binary { op, left, right } => {
                let (l, r) = (eval(left, row), eval(right, row));
                match op {
                    BinaryOp::And => match (l.map(|v| v.b()), r.map(|v| v.b())) {
                        (Some(false), _) | (_, Some(false)) => Some(Value::B(false)),
                        (Some(true), Some(true)) => Some(Value::B(true)),
                        _ => None,
                    },
                    BinaryOp::Or => match (l.map(|v| v.b()), r.map(|v| v.b())) {
                        (Some(true), _) | (_, Some(true)) => Some(Value::B(true)),
                        (Some(false), Some(false)) => Some(Value::B(false)),
                        _ => None,
                    },
                    _ => {
                        let (a, b) = (l?.f(), r?.f());
                        Some(Value::B(match op {
                            BinaryOp::Eq => a == b,
                            BinaryOp::NotEq => a != b,
                            BinaryOp::Lt => a < b,
                            BinaryOp::LtEq => a <= b,
                            BinaryOp::Gt => a > b,
                            BinaryOp::GtEq => a >= b,
                            _ => panic!("unexpected {op:?}"),
                        }))
                    }
                }
            }
            other => panic!("unexpected {other:?}"),
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum Value {
        F(f64),
        B(bool),
    }
    impl Value {
        fn f(self) -> f64 {
            match self {
                Value::F(v) => v,
                Value::B(_) => panic!("bool"),
            }
        }
        fn b(self) -> bool {
            match self {
                Value::B(v) => v,
                Value::F(_) => panic!("float"),
            }
        }
    }

    /// The total order: NaN greatest and equal to itself.
    fn reference(op: BinaryOp, a: Option<f64>, b: Option<f64>) -> Option<bool> {
        let (a, b) = (a?, b?);
        let key = |v: f64| if v.is_nan() { (1, 0.0) } else { (0, v) };
        let ord = key(a).partial_cmp(&key(b)).unwrap();
        use std::cmp::Ordering::*;
        Some(match op {
            BinaryOp::Eq => ord == Equal,
            BinaryOp::NotEq => ord != Equal,
            BinaryOp::Lt => ord == Less,
            BinaryOp::LtEq => ord != Greater,
            BinaryOp::Gt => ord == Greater,
            BinaryOp::GtEq => ord != Less,
            _ => unreachable!(),
        })
    }

    const OPS: [BinaryOp; 6] = [
        BinaryOp::Eq,
        BinaryOp::NotEq,
        BinaryOp::Lt,
        BinaryOp::LtEq,
        BinaryOp::Gt,
        BinaryOp::GtEq,
    ];

    /// Every comparison of every pair from {NULL, NaN, -NaN, -inf, -1, 0,
    /// 3.4, inf}, column-vs-column, column-vs-literal and with `NOT`, matches
    /// the total order under three-valued logic.
    #[test]
    fn rewrite_matches_the_total_order_with_kleene_nulls() {
        let vals = [
            None,
            Some(f64::NAN),
            Some(-f64::NAN),
            Some(f64::NEG_INFINITY),
            Some(-1.0),
            Some(0.0),
            Some(3.4),
            Some(f64::INFINITY),
        ];
        let s = schema();
        for op in OPS {
            for &a in &vals {
                for &b in &vals {
                    let row = |c: &str| if c == "d" { a } else { b };
                    let want = reference(op, a, b);
                    let cmp = bin(op, col("d"), col("e"));
                    let got = eval(&total_order_expr(&cmp, &s).unwrap(), &row).map(|v| v.b());
                    assert_eq!(got, want, "{op:?} {a:?} {b:?}");
                    let neg = not(cmp);
                    let got = eval(&total_order_expr(&neg, &s).unwrap(), &row).map(|v| v.b());
                    assert_eq!(got, want.map(|w| !w), "NOT {op:?} {a:?} {b:?}");
                    if let Some(bv) = b {
                        let cmp = bin(op, col("d"), lit(bv));
                        let got = eval(&total_order_expr(&cmp, &s).unwrap(), &row).map(|v| v.b());
                        assert_eq!(got, want, "{op:?} {a:?} literal {b:?}");
                        let cmp = bin(op, lit(bv), col("d"));
                        let got = eval(&total_order_expr(&cmp, &s).unwrap(), &|_| a).map(|v| v.b());
                        assert_eq!(got, reference(op, b, a), "literal {b:?} {op:?} {a:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn literal_operands_fold_their_nan_tests() {
        assert_eq!(rw(col("d").gt(lit(3.0))), "((3 < d) OR (NOT (d = d)))");
        assert_eq!(rw(col("d").eq(lit(3.4))), "(d = 3.4)");
        assert_eq!(rw(col("d").neq(lit(3.4))), "(NOT (d = 3.4))");
        assert_eq!(rw(col("d").lt(lit(3.0))), "(d < 3)");
    }

    #[test]
    fn non_float_comparisons_are_unchanged() {
        assert_eq!(rw(col("i").gt(lit(3i64))), "(i > 3)");
        assert_eq!(rw(col("x").neq(col("i"))), "(x <> i)");
        assert_eq!(rw(col("d").eq(Expr::Literal(Literal::Null))), "(d = NULL)");
    }

    #[test]
    fn int_vs_float_comparison_treats_the_int_side_as_never_nan() {
        // `i` can be NULL, so its not-NaN test stays `i = i` (UNKNOWN).
        assert_eq!(
            rw(col("i").lt(col("d"))),
            "((i < d) OR ((i = i) AND (NOT (d = d))))"
        );
        assert_eq!(rw(col("i").eq(col("d"))), "(i = d)");
    }

    #[test]
    fn nested_comparisons_are_rewritten_once() {
        let e = Expr::Case {
            branches: vec![(col("d").gt(lit(0.0)), lit(1i64))],
            else_branch: Some(Box::new(lit(0i64))),
        };
        assert_eq!(
            rw(e),
            "CASE WHEN ((0 < d) OR (NOT (d = d))) THEN 1 ELSE 0 END"
        );
    }

    fn checks(e: Expr) -> Vec<OverflowCheck> {
        let plan = LogicalPlan::Project {
            input: Box::new(LogicalPlan::Scan {
                table: "t".into(),
                projection: None,
                schema: schema(),
            }),
            exprs: vec![e],
        };
        overflow_checks(&plan).unwrap()
    }

    #[test]
    fn provably_safe_arithmetic_needs_no_check() {
        // Int32 + Int64 literal is Int64: cannot overflow.
        assert!(checks(col("i").add(lit(1i64))).is_empty());
        assert!(checks(col("i").mul(lit(2i64)).add(lit(1i64))).is_empty());
        assert!(checks(col("x").add(lit(0i64))).is_empty());
        assert!(checks(widen(col("i")).mul(widen(col("j")))).is_empty());
        assert!(checks(col("d").add(lit(1.0))).is_empty());
    }

    #[test]
    fn unprovable_arithmetic_gets_one_check_per_node() {
        let c = checks(col("i").add(col("j")));
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].operations, vec!["(i + j)".to_string()]);
        assert!(c[0].error().to_string().contains("out of range"));
        let LogicalPlan::Limit {
            input, limit: 1, ..
        } = &c[0].plan
        else {
            panic!("{:?}", c[0].plan);
        };
        let LogicalPlan::Filter { predicate, .. } = &**input else {
            panic!("{input:?}");
        };
        let p = format_expr(predicate);
        assert!(p.contains("Int64") && p.contains("2147483647"), "{p}");
        assert_eq!(checks(col("x").add(lit(1i64))).len(), 1);
        assert_eq!(checks(col("x").mul(col("x"))).len(), 1);
        assert_eq!(checks(lit(0i64).sub(col("x"))).len(), 1);
        let narrow = Expr::Cast {
            expr: Box::new(col("x")),
            target: DataType::Int32,
            safe: false,
        };
        assert_eq!(checks(narrow).len(), 1);
    }

    /// A float-to-integer CAST is checked: NaN (greatest under the total
    /// order the check plan is compiled with), +-inf and values whose
    /// truncation leaves the target range overflow.
    #[test]
    fn float_to_int_cast_gets_a_range_check() {
        let cast = |t| Expr::Cast {
            expr: Box::new(col("d")),
            target: t,
            safe: false,
        };
        let pred = |e| {
            let c = checks(e);
            assert_eq!(c.len(), 1);
            let LogicalPlan::Limit { input, .. } = &c[0].plan else {
                panic!()
            };
            let LogicalPlan::Filter { predicate, .. } = &**input else {
                panic!()
            };
            predicate.clone()
        };
        let s = schema();
        for (t, lo, hi) in [
            (DataType::Int32, -2_147_483_648.0, 2_147_483_647.0),
            (
                DataType::Int64,
                -9_223_372_036_854_775_808.0,
                9_223_372_036_854_774_784.0,
            ),
        ] {
            let p = total_order_expr(&pred(cast(t)), &s).unwrap();
            let flags = |v: Option<f64>| eval(&p, &|_| v).map(|b| b.b()).unwrap_or(false);
            for ok in [0.0, 2.9, -2.9, lo, hi, lo - 0.5, hi + 0.5] {
                if (DataType::Int64 == t) && (ok == lo - 0.5 || ok == hi + 0.5) {
                    continue; // not representable apart from the bound
                }
                assert!(!flags(Some(ok)), "{t:?} {ok}");
            }
            for bad in [
                f64::NAN,
                -f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                hi * 2.0,
                lo * 2.0,
            ] {
                assert!(flags(Some(bad)), "{t:?} {bad}");
            }
            assert!(!flags(None), "NULL never overflows");
        }
        // TRY_CAST never raises.
        assert!(checks(Expr::Cast {
            expr: Box::new(col("d")),
            target: DataType::Int32,
            safe: true,
        })
        .is_empty());
    }

    /// The Int64 overflow conditions, evaluated with wrapping arithmetic (as
    /// every tier does), are exact on the boundary values.
    #[test]
    fn int64_conditions_are_exact_under_wrapping_evaluation() {
        fn ev(e: &Expr, l: i64, r: i64) -> Option<i64> {
            Some(match e {
                Expr::Column(c) if c == "l" => l,
                Expr::Column(_) => r,
                Expr::Literal(Literal::Int64(v)) => *v,
                Expr::Literal(Literal::Bool(b)) => *b as i64,
                Expr::Binary { op, left, right } => {
                    let a = ev(left, l, r);
                    let b = ev(right, l, r);
                    match op {
                        BinaryOp::And => {
                            return match (a, b) {
                                (Some(0), _) | (_, Some(0)) => Some(0),
                                (Some(_), Some(_)) => Some(1),
                                _ => None,
                            }
                        }
                        BinaryOp::Or => {
                            return match (a, b) {
                                (Some(1), _) | (_, Some(1)) => Some(1),
                                (Some(_), Some(_)) => Some(0),
                                _ => None,
                            }
                        }
                        _ => {}
                    }
                    let (a, b) = (a?, b?);
                    match op {
                        BinaryOp::Add => a.wrapping_add(b),
                        BinaryOp::Sub => a.wrapping_sub(b),
                        BinaryOp::Mul => a.wrapping_mul(b),
                        // The engine's integer division is NULL on a zero
                        // divisor and on MIN / -1.
                        BinaryOp::Div => a.checked_div(b)?,
                        BinaryOp::Eq => (a == b) as i64,
                        BinaryOp::NotEq => (a != b) as i64,
                        BinaryOp::Lt => (a < b) as i64,
                        BinaryOp::Gt => (a > b) as i64,
                        _ => panic!("{op:?}"),
                    }
                }
                other => panic!("{other:?}"),
            })
        }
        let edge = [
            i64::MIN,
            i64::MIN + 1,
            -3_037_000_500,
            -3_037_000_499,
            -2,
            -1,
            0,
            1,
            2,
            3_037_000_499,
            3_037_000_500,
            i64::MAX - 1,
            i64::MAX,
        ];
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            let cond = op_overflows(op, &col("l"), &col("r"), DataType::Int64);
            for &l in &edge {
                for &r in &edge {
                    let want = match op {
                        BinaryOp::Add => l.checked_add(r),
                        BinaryOp::Sub => l.checked_sub(r),
                        _ => l.checked_mul(r),
                    }
                    .is_none();
                    let got = ev(&cond, l, r) == Some(1);
                    assert_eq!(got, want, "{op:?} {l} {r}");
                }
            }
        }
    }

    #[test]
    fn case_branch_overflow_is_guarded_by_its_condition() {
        let e = Expr::Case {
            branches: vec![(col("i").lt(lit(100i64)), col("x").mul(lit(1000i64)))],
            else_branch: Some(Box::new(lit(0i64))),
        };
        let c = checks(e);
        assert_eq!(c.len(), 1);
        let LogicalPlan::Limit { input, .. } = &c[0].plan else {
            panic!()
        };
        let LogicalPlan::Filter { predicate, .. } = &**input else {
            panic!()
        };
        assert!(
            format_expr(predicate).starts_with("((i < 100) AND "),
            "{}",
            format_expr(predicate)
        );
    }
}
