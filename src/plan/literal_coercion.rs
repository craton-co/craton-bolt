// SPDX-License-Identifier: Apache-2.0

//! Exact coercion of a constant compared with an operand of another type.
//!
//! SQL constants carry the frontend's default types: `2.50` is `Float64`,
//! `3` is `Int64`, `TIMESTAMP '...'` is `Timestamp(Nanosecond, None)` and
//! `DATE '...'` is `Date32`. When such a constant is compared with a
//! `Decimal128(p, s)`, `Timestamp(unit, tz)` or `Date32` operand, this module
//! rewrites the comparison, once, into one between same-typed values, so
//! every executor (GPU kernels, the host evaluator, storage pruning) sees a
//! plain comparison it already supports.
//!
//! The rewrite is exact. The constant is placed on the operand type's value
//! grid (decimal units of `10^-s`, or timestamp / date ticks):
//!
//! * on a grid point `v`: `x op c` becomes `x op v`;
//! * strictly between grid points `lo < c < hi`: `x < c` and `x <= c` become
//!   `x <= lo`, `x > c` and `x >= c` become `x >= hi`, `x = c` is never true
//!   and `x <> c` always is (for a non-NULL `x`);
//! * beyond every value of the type: each comparison is constant.
//!
//! "Never" and "always" are written `x <> x` and `x = x`, which are FALSE
//! and TRUE for a non-NULL `x` and NULL for a NULL one, as the original
//! comparison was (none of these types has a NaN).
//!
//! Timestamps compare as instants: a zoned and a naive value are both read
//! as UTC ticks, and the constant adopts the operand's unit and time zone.
//!
//! The type checker ([`crate::plan::logical_plan::Expr::dtype`]) accepts
//! exactly the comparisons [`coercible`] reports, and the engine runs
//! [`coerce_plan`] before any other plan rewrite.

use crate::plan::logical_plan::{
    AggregateExpr, BinaryOp, DataType, Expr, Literal, LogicalPlan, Schema, TimeUnit,
};

/// True when a comparison between `lit` and an operand of type `target` is
/// rewritten by [`coerce_plan`] (the literal's own type differs from
/// `target`, and the pair is one this module places exactly).
pub(crate) fn coercible(lit: &Literal, target: DataType) -> bool {
    lit.dtype() != Some(target) && place(lit, target).is_some()
}

/// Rewrite every comparison between an expression and a differently typed,
/// coercible constant in `plan` (subqueries included). Comparisons whose
/// operand type cannot be resolved are left for the type checker to report.
pub(crate) fn coerce_plan(mut plan: LogicalPlan) -> LogicalPlan {
    coerce_node(&mut plan);
    plan
}

/// Where a constant falls on the value grid of a target type.
#[derive(Debug, Clone, PartialEq)]
enum Placement {
    /// Exactly the grid value.
    Exact(Literal),
    /// Strictly between two adjacent grid values `(below, above)`.
    Between(Literal, Literal),
    /// Greater than every value of the type.
    AboveAll,
    /// Less than every value of the type.
    BelowAll,
}

/// The constant's position on an integer grid: `floor` and whether the
/// constant equals it.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Scaled {
    Value {
        floor: i128,
        exact: bool,
    },
    /// Beyond the `i128` range, on the side of this sign.
    Overflow {
        positive: bool,
    },
}

fn place(lit: &Literal, target: DataType) -> Option<Placement> {
    match target {
        DataType::Decimal128(p, s) => {
            let (mantissa, scale) = decimal_value(lit)?;
            let max = pow10(p as u32)? - 1;
            let make = |v: i128| Literal::Decimal128(v, p, s);
            Some(on_grid(rescale(mantissa, scale, s as i32), -max, max, make))
        }
        DataType::Timestamp(unit, tz) => {
            let ns = instant_ns(lit)?;
            let per = ns_per_tick(unit);
            let make = |v: i128| Literal::Timestamp(v as i64, unit, tz);
            Some(on_grid(
                floor_div(ns, per),
                i64::MIN as i128,
                i64::MAX as i128,
                make,
            ))
        }
        DataType::Date32 => {
            let ns = instant_ns(lit)?;
            let make = |v: i128| Literal::Date32(v as i32);
            Some(on_grid(
                floor_div(ns, NS_PER_DAY),
                i32::MIN as i128,
                i32::MAX as i128,
                make,
            ))
        }
        _ => None,
    }
}

fn on_grid(scaled: Scaled, min: i128, max: i128, make: impl Fn(i128) -> Literal) -> Placement {
    match scaled {
        Scaled::Overflow { positive: true } => Placement::AboveAll,
        Scaled::Overflow { positive: false } => Placement::BelowAll,
        Scaled::Value { floor, exact: true } => {
            if floor > max {
                Placement::AboveAll
            } else if floor < min {
                Placement::BelowAll
            } else {
                Placement::Exact(make(floor))
            }
        }
        // floor < c < floor + 1.
        Scaled::Value {
            floor,
            exact: false,
        } => {
            if floor >= max {
                Placement::AboveAll
            } else if floor < min {
                Placement::BelowAll
            } else {
                Placement::Between(make(floor), make(floor + 1))
            }
        }
    }
}

/// A numeric constant as `mantissa * 10^-scale`, exactly. A float is read
/// as its shortest round-trip decimal form, which is the text it was
/// written as whenever that text has at most 15 significant digits.
fn decimal_value(lit: &Literal) -> Option<(i128, i32)> {
    match lit {
        Literal::Int32(v) => Some((*v as i128, 0)),
        Literal::Int64(v) => Some((*v as i128, 0)),
        Literal::Decimal128(v, _, s) => Some((*v, *s as i32)),
        Literal::Float64(f) if f.is_finite() => parse_scientific(&format!("{f:e}")),
        Literal::Float32(f) if f.is_finite() => parse_scientific(&format!("{f:e}")),
        _ => None,
    }
}

/// Parse Rust's `{:e}` rendering (`-2.501e0`, `1e-7`) into
/// `(mantissa, scale)` with `value = mantissa * 10^-scale`.
fn parse_scientific(s: &str) -> Option<(i128, i32)> {
    let (digits, exp) = s.split_once('e')?;
    let exp: i32 = exp.parse().ok()?;
    let (negative, digits) = match digits.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, digits),
    };
    let (int_part, frac_part) = digits.split_once('.').unwrap_or((digits, ""));
    let mut mantissa: i128 = 0;
    for c in int_part.chars().chain(frac_part.chars()) {
        mantissa = mantissa
            .checked_mul(10)?
            .checked_add(c.to_digit(10)? as i128)?;
    }
    if negative {
        mantissa = -mantissa;
    }
    Some((mantissa, frac_part.len() as i32 - exp))
}

/// `mantissa * 10^-from` on the grid of `10^-to`.
fn rescale(mantissa: i128, from: i32, to: i32) -> Scaled {
    if mantissa == 0 {
        return Scaled::Value {
            floor: 0,
            exact: true,
        };
    }
    if to >= from {
        let factor = u32::try_from(to - from).ok().and_then(pow10);
        match factor.and_then(|f| mantissa.checked_mul(f)) {
            Some(v) => Scaled::Value {
                floor: v,
                exact: true,
            },
            None => Scaled::Overflow {
                positive: mantissa > 0,
            },
        }
    } else {
        match u32::try_from(from - to).ok().and_then(pow10) {
            Some(d) => floor_div(mantissa, d),
            // |mantissa| < 10^39 <= divisor: the value is in (-1, 1).
            None => Scaled::Value {
                floor: if mantissa > 0 { 0 } else { -1 },
                exact: false,
            },
        }
    }
}

fn floor_div(n: i128, d: i128) -> Scaled {
    Scaled::Value {
        floor: n.div_euclid(d),
        exact: n.rem_euclid(d) == 0,
    }
}

fn pow10(e: u32) -> Option<i128> {
    10i128.checked_pow(e)
}

const NS_PER_DAY: i128 = 86_400 * 1_000_000_000;

fn ns_per_tick(unit: TimeUnit) -> i128 {
    match unit {
        TimeUnit::Second => 1_000_000_000,
        TimeUnit::Millisecond => 1_000_000,
        TimeUnit::Microsecond => 1_000,
        TimeUnit::Nanosecond => 1,
    }
}

/// A temporal constant as UTC nanoseconds since the epoch.
fn instant_ns(lit: &Literal) -> Option<i128> {
    match lit {
        Literal::Timestamp(t, unit, _) => Some(*t as i128 * ns_per_tick(*unit)),
        Literal::Date32(d) => Some(*d as i128 * NS_PER_DAY),
        _ => None,
    }
}

/// `c op x` for `x op c`.
fn flip(op: BinaryOp) -> BinaryOp {
    match op {
        BinaryOp::Lt => BinaryOp::Gt,
        BinaryOp::LtEq => BinaryOp::GtEq,
        BinaryOp::Gt => BinaryOp::Lt,
        BinaryOp::GtEq => BinaryOp::LtEq,
        other => other,
    }
}

fn is_comparison(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
    )
}

fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// `operand op c` for the constant `c` at `placement`.
fn rewrite(op: BinaryOp, operand: Expr, placement: Placement) -> Expr {
    use BinaryOp::*;
    let always = |x: Expr| binary(Eq, x.clone(), x);
    let never = |x: Expr| binary(NotEq, x.clone(), x);
    match placement {
        Placement::Exact(v) => binary(op, operand, Expr::Literal(v)),
        Placement::Between(below, above) => match op {
            Lt | LtEq => binary(LtEq, operand, Expr::Literal(below)),
            Gt | GtEq => binary(GtEq, operand, Expr::Literal(above)),
            NotEq => always(operand),
            _ => never(operand),
        },
        Placement::AboveAll => match op {
            Lt | LtEq | NotEq => always(operand),
            _ => never(operand),
        },
        Placement::BelowAll => match op {
            Gt | GtEq | NotEq => always(operand),
            _ => never(operand),
        },
    }
}

/// The coerced form of `e` when it is a comparison of an expression with a
/// differently typed, coercible constant.
fn coerce_comparison(e: &Expr, schema: &Schema) -> Option<Expr> {
    let Expr::Binary { op, left, right } = e else {
        return None;
    };
    if !is_comparison(*op) {
        return None;
    }
    let (operand, lit, op) = match (left.as_ref(), right.as_ref()) {
        (Expr::Literal(_), Expr::Literal(_)) => return None,
        (operand, Expr::Literal(lit)) => (operand, lit, *op),
        (Expr::Literal(lit), operand) => (operand, lit, flip(*op)),
        _ => return None,
    };
    if matches!(lit, Literal::Null) {
        return None;
    }
    let target = operand.dtype(schema).ok()?;
    if lit.dtype() == Some(target) {
        return None;
    }
    Some(rewrite(op, operand.clone(), place(lit, target)?))
}

/// Coerce every comparison inside `root` against `schema`. Iterative, so a
/// long `OR` / `AND` chain costs no stack.
fn coerce_expr(root: &mut Expr, schema: &Schema) {
    let mut stack: Vec<&mut Expr> = vec![root];
    while let Some(e) = stack.pop() {
        if let Some(coerced) = coerce_comparison(e, schema) {
            *e = coerced;
        }
        match e {
            Expr::Column(_) | Expr::Literal(_) => {}
            Expr::Binary { left, right, .. } => {
                stack.push(left);
                stack.push(right);
            }
            Expr::Unary { operand, .. } => stack.push(operand),
            Expr::Case {
                branches,
                else_branch,
            } => {
                for (when, then) in branches {
                    stack.push(when);
                    stack.push(then);
                }
                if let Some(e) = else_branch {
                    stack.push(e);
                }
            }
            Expr::Like { expr, .. }
            | Expr::Cast { expr, .. }
            | Expr::CastFormat { expr, .. }
            | Expr::Extract { expr, .. }
            | Expr::DateTrunc { expr, .. }
            | Expr::Alias(expr, _) => stack.push(expr),
            Expr::ScalarFn { args, .. } => stack.extend(args.iter_mut()),
            Expr::ScalarSubquery(plan) => coerce_node(plan),
            Expr::InSubquery { expr, subquery, .. } => {
                coerce_node(subquery);
                stack.push(expr);
            }
        }
    }
}

fn agg_expr_mut(agg: &mut AggregateExpr) -> &mut Expr {
    match agg {
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

fn coerce_all<'a>(exprs: impl IntoIterator<Item = &'a mut Expr>, input: &LogicalPlan) {
    let mut exprs = exprs.into_iter().peekable();
    if exprs.peek().is_none() {
        return;
    }
    if let Ok(schema) = input.schema() {
        for e in exprs {
            coerce_expr(e, &schema);
        }
    }
}

fn coerce_node(plan: &mut LogicalPlan) {
    // A join's residual filter sees the join's own output columns.
    let join_schema = if matches!(
        plan,
        LogicalPlan::Join {
            filter: Some(_),
            ..
        }
    ) {
        plan.schema().ok()
    } else {
        None
    };
    match plan {
        LogicalPlan::Scan { .. } => {}
        LogicalPlan::Filter { input, predicate } => {
            coerce_node(input);
            coerce_all([predicate], input);
        }
        LogicalPlan::Project { input, exprs } => {
            coerce_node(input);
            coerce_all(exprs.iter_mut(), input);
        }
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => {
            coerce_node(input);
            coerce_all(
                group_by
                    .iter_mut()
                    .chain(aggregates.iter_mut().map(agg_expr_mut)),
                input,
            );
        }
        LogicalPlan::Window {
            input,
            partition_by,
            ..
        } => {
            coerce_node(input);
            coerce_all(partition_by.iter_mut(), input);
        }
        LogicalPlan::Sort { input, sort_exprs } => {
            coerce_node(input);
            coerce_all(sort_exprs.iter_mut().map(|s| &mut s.expr), input);
        }
        LogicalPlan::Distinct { input } | LogicalPlan::Limit { input, .. } => coerce_node(input),
        LogicalPlan::Union { inputs } => inputs.iter_mut().for_each(coerce_node),
        LogicalPlan::SetOp { left, right, .. } => {
            coerce_node(left);
            coerce_node(right);
        }
        LogicalPlan::Join {
            left,
            right,
            on,
            filter,
            ..
        } => {
            coerce_node(left);
            coerce_node(right);
            coerce_all(on.iter_mut().map(|(l, _)| l), left);
            coerce_all(on.iter_mut().map(|(_, r)| r), right);
            if let (Some(f), Some(schema)) = (filter, join_schema) {
                coerce_expr(f, &schema);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::Field;

    fn dec(v: i128) -> Literal {
        Literal::Decimal128(v, 10, 2)
    }

    const D: DataType = DataType::Decimal128(10, 2);

    #[test]
    fn numeric_constants_on_a_decimal_grid() {
        assert_eq!(
            place(&Literal::Float64(2.5), D),
            Some(Placement::Exact(dec(250)))
        );
        assert_eq!(
            place(&Literal::Int64(3), D),
            Some(Placement::Exact(dec(300)))
        );
        // 2.3 is not a double; its shortest form is, so `= 2.3` still matches.
        assert_eq!(
            place(&Literal::Float64(2.3), D),
            Some(Placement::Exact(dec(230)))
        );
        assert_eq!(
            place(&Literal::Float64(2.501), D),
            Some(Placement::Between(dec(250), dec(251)))
        );
        assert_eq!(
            place(&Literal::Float64(-0.255), D),
            Some(Placement::Between(dec(-26), dec(-25)))
        );
        assert_eq!(
            place(&Literal::Decimal128(25, 5, 1), D),
            Some(Placement::Exact(dec(250)))
        );
        assert_eq!(place(&Literal::Float64(1e8), D), Some(Placement::AboveAll));
        assert_eq!(
            place(&Literal::Float64(99_999_999.99), D),
            Some(Placement::Exact(dec(9_999_999_999)))
        );
        assert_eq!(
            place(&Literal::Float64(-1e300), D),
            Some(Placement::BelowAll)
        );
        assert_eq!(
            place(&Literal::Float64(1e-300), D),
            Some(Placement::Between(dec(0), dec(1)))
        );
        assert_eq!(place(&Literal::Float64(f64::NAN), D), None);
        assert_eq!(place(&Literal::Utf8("1".into()), D), None);
    }

    #[test]
    fn instants_on_a_coarser_grid() {
        let us = DataType::Timestamp(TimeUnit::Microsecond, Some("UTC"));
        let ns = |t| Literal::Timestamp(t, TimeUnit::Nanosecond, None);
        let at_us = |t| Literal::Timestamp(t, TimeUnit::Microsecond, Some("UTC"));
        assert_eq!(place(&ns(5_000), us), Some(Placement::Exact(at_us(5))));
        assert_eq!(
            place(&ns(5_500), us),
            Some(Placement::Between(at_us(5), at_us(6)))
        );
        assert_eq!(
            place(&ns(-1), us),
            Some(Placement::Between(at_us(-1), at_us(0)))
        );
        // A day boundary is exact on Date32; noon is between two days.
        let day = 86_400_000_000_000i64;
        assert_eq!(
            place(&ns(3 * day), DataType::Date32),
            Some(Placement::Exact(Literal::Date32(3)))
        );
        assert_eq!(
            place(&ns(3 * day + day / 2), DataType::Date32),
            Some(Placement::Between(Literal::Date32(3), Literal::Date32(4)))
        );
        // A date on a finer grid, and past the end of i64 seconds.
        assert_eq!(
            place(
                &Literal::Date32(2),
                DataType::Timestamp(TimeUnit::Second, None)
            ),
            Some(Placement::Exact(Literal::Timestamp(
                172_800,
                TimeUnit::Second,
                None
            )))
        );
        assert_eq!(
            place(
                &Literal::Timestamp(i64::MAX, TimeUnit::Second, None),
                DataType::Timestamp(TimeUnit::Nanosecond, None)
            ),
            Some(Placement::AboveAll)
        );
    }

    fn col_d() -> Expr {
        Expr::Column("d".into())
    }

    fn cmp(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        binary(op, l, r)
    }

    fn schema() -> Schema {
        Schema::new(vec![
            Field::new("d", D, true),
            Field::new("id", DataType::Int32, false),
        ])
    }

    fn coerced(e: Expr) -> String {
        let mut e = e;
        coerce_expr(&mut e, &schema());
        format!("{e:?}")
    }

    #[test]
    fn comparisons_round_toward_the_right_side() {
        let lit = |f: f64| Expr::Literal(Literal::Float64(f));
        let d = |v| Expr::Literal(dec(v));
        let f = |op, v| cmp(op, col_d(), lit(v));
        use BinaryOp::*;
        assert_eq!(
            coerced(f(Gt, 2.501)),
            format!("{:?}", cmp(GtEq, col_d(), d(251)))
        );
        assert_eq!(
            coerced(f(GtEq, 2.501)),
            format!("{:?}", cmp(GtEq, col_d(), d(251)))
        );
        assert_eq!(
            coerced(f(Lt, 2.501)),
            format!("{:?}", cmp(LtEq, col_d(), d(250)))
        );
        assert_eq!(
            coerced(f(LtEq, 2.501)),
            format!("{:?}", cmp(LtEq, col_d(), d(250)))
        );
        assert_eq!(
            coerced(f(Eq, 2.501)),
            format!("{:?}", cmp(NotEq, col_d(), col_d()))
        );
        assert_eq!(
            coerced(f(NotEq, 2.501)),
            format!("{:?}", cmp(Eq, col_d(), col_d()))
        );
        assert_eq!(
            coerced(f(Lt, 1e9)),
            format!("{:?}", cmp(Eq, col_d(), col_d()))
        );
        assert_eq!(
            coerced(f(Eq, 1e9)),
            format!("{:?}", cmp(NotEq, col_d(), col_d()))
        );
        // Constant on the left: `2.50 <= d` is `d >= 2.50`.
        assert_eq!(
            coerced(cmp(LtEq, lit(2.5), col_d())),
            format!("{:?}", cmp(GtEq, col_d(), d(250)))
        );
    }

    #[test]
    fn other_comparisons_are_untouched() {
        use BinaryOp::*;
        let int = cmp(
            Gt,
            Expr::Column("id".into()),
            Expr::Literal(Literal::Int64(3)),
        );
        assert_eq!(coerced(int.clone()), format!("{int:?}"));
        let same = cmp(Gt, col_d(), Expr::Literal(dec(250)));
        assert_eq!(coerced(same.clone()), format!("{same:?}"));
        // Nested comparisons are reached.
        let nested = cmp(
            And,
            int.clone(),
            cmp(Eq, col_d(), Expr::Literal(Literal::Int64(1))),
        );
        assert_eq!(
            coerced(nested),
            format!(
                "{:?}",
                cmp(And, int, cmp(Eq, col_d(), Expr::Literal(dec(100))))
            )
        );
    }

    #[test]
    fn coercible_matches_the_rewrite() {
        assert!(coercible(&Literal::Float64(2.5), D));
        assert!(!coercible(&dec(250), D));
        assert!(!coercible(&Literal::Float64(2.5), DataType::Float64));
        assert!(coercible(
            &Literal::Timestamp(0, TimeUnit::Nanosecond, None),
            DataType::Date32
        ));
        assert!(!coercible(&Literal::Utf8("x".into()), DataType::Date32));
    }

    #[test]
    fn plans_are_coerced_below_projections() {
        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: schema(),
        };
        let plan = LogicalPlan::Project {
            input: Box::new(LogicalPlan::Filter {
                input: Box::new(scan),
                predicate: cmp(
                    BinaryOp::GtEq,
                    col_d(),
                    Expr::Literal(Literal::Float64(2.5)),
                ),
            }),
            exprs: vec![Expr::Column("id".into())],
        };
        let plan = coerce_plan(plan);
        assert!(format!("{plan:?}").contains("Decimal128(250, 10, 2)"));
        plan.schema().expect("coerced plan type-checks");
    }
}
