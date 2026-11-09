// SPDX-License-Identifier: Apache-2.0

//! Extended column types ([`DataType::Extended`]): what each one is stored
//! as, which operations it supports, and the plan rewrite that runs a query
//! over them on the engine's native types.
//!
//! # Families
//!
//! | family | Arrow types | storage | supports |
//! |---|---|---|---|
//! | time | `Time32(s, ms)`, `Time64(us, ns)` | `Int32` / `Int64` ticks | compare (any unit), `GROUP BY`, `ORDER BY`, `DISTINCT`, join keys, `MIN` / `MAX`, `CASE` |
//! | duration | `Duration(unit)` | `Int64` ticks | as time, plus `SUM`, `TIMESTAMP ± duration` |
//! | binary | `Binary`, `LargeBinary`, `BinaryView`, `FixedSizeBinary(n)` | `Utf8`, lowercase hex | compare (byte order), `GROUP BY`, `ORDER BY`, `DISTINCT`, join keys, `CASE` |
//! | opaque | `Interval`, `Decimal256`, `Null`, `List`, `LargeList`, `FixedSizeList`, `Struct`, `Map`, `Union` | `Int64` row handle | — |
//!
//! Every family can be projected, carried through filters, sorts on other
//! keys, `LIMIT`, `UNION ALL` and joins, tested with `IS [NOT] NULL` and
//! counted. [`supports`] is the one table of these rules; both tiers run
//! the storage types natively, so the table does not depend on the tier.
//!
//! # Execution
//!
//! The engine never computes on an extended type. [`validate_plan`] rejects
//! (at plan time, for `sql`, `explain_sql` and `run_logical_plan` alike)
//! every operation the table does not allow, then [`lower_plan`] rewrites the
//! plan onto the storage types: scans read the storage columns, `TIME` /
//! `INTERVAL` / `X'..'` constants become storage constants, a comparison
//! between two time or duration units rescales the coarser side, and
//! `TIMESTAMP - TIMESTAMP` / `TIMESTAMP ± duration` become integer
//! arithmetic on the ticks. Lowercase hex keeps byte order and equality, so a
//! binary column groups, sorts and compares as its hex string. An opaque
//! column's storage is a handle `(source << 40) | row` that is NULL exactly
//! where the value is; the engine attaches the real values to the result by
//! handle (`crate::exec::extended_store`). The query's logical output schema
//! says which result columns to convert back.

use std::cell::Cell;

use arrow_schema::DataType as A;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{
    join_combined_schema, AggregateExpr, BinaryOp, DataType, Expr, Literal, LogicalPlan, Schema,
    SortExpr, TimeUnit, UnaryOp, WindowFunc,
};

/// The Arrow types [`DataType::from_arrow`] maps to [`DataType::Extended`].
pub(crate) fn is_extended_arrow(t: &A) -> bool {
    matches!(
        t,
        A::Time32(_)
            | A::Time64(_)
            | A::Duration(_)
            | A::Binary
            | A::LargeBinary
            | A::BinaryView
            | A::FixedSizeBinary(_)
            | A::Interval(_)
            | A::Decimal256(_, _)
            | A::Null
            | A::List(_)
            | A::LargeList(_)
            | A::FixedSizeList(_, _)
            | A::ListView(_)
            | A::LargeListView(_)
            | A::Struct(_)
            | A::Map(_, _)
            | A::Union(_, _)
    )
}

/// The semantic family of an extended type (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    /// Time of day in `TimeUnit` ticks since midnight.
    Time(TimeUnit),
    /// Elapsed time in `TimeUnit` ticks.
    Duration(TimeUnit),
    /// Byte strings.
    Binary,
    /// Stored and returned, never computed on.
    Opaque,
}

/// The family of an extended Arrow type.
pub(crate) fn family(t: &A) -> Family {
    let unit = crate::exec::schema_convert::arrow_time_unit_to_plan;
    match t {
        A::Time32(u) | A::Time64(u) => Family::Time(unit(u)),
        A::Duration(u) => Family::Duration(unit(u)),
        A::Binary | A::LargeBinary | A::BinaryView | A::FixedSizeBinary(_) => Family::Binary,
        _ => Family::Opaque,
    }
}

/// The family of `dt`, or `None` for a natively computed type.
pub(crate) fn family_of(dt: DataType) -> Option<Family> {
    match dt {
        DataType::Extended(t) => Some(family(t)),
        _ => None,
    }
}

/// The native type the engine stores (and computes) `dt` as.
pub(crate) fn storage_dtype(dt: DataType) -> DataType {
    match dt {
        DataType::Extended(t) => match family(t) {
            Family::Time(_) if matches!(t, A::Time32(_)) => DataType::Int32,
            Family::Time(_) | Family::Duration(_) | Family::Opaque => DataType::Int64,
            Family::Binary => DataType::Utf8,
        },
        other => other,
    }
}

/// An operation applied to a value of some type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Use {
    /// `IS [NOT] NULL`.
    IsNull,
    /// `COUNT(x)`.
    Count,
    /// A comparison operator.
    Compare(BinaryOp),
    /// An arithmetic, logical, string or bitwise operator.
    Operator(BinaryOp),
    /// `SUM(x)`.
    Sum,
    /// `MIN(x)` / `MAX(x)`.
    MinMax,
    /// A numeric aggregate (`AVG`, `VAR_*`, `STDDEV_*`), by SQL name.
    Aggregate(&'static str),
    /// `GROUP BY` / `PARTITION BY` key.
    GroupBy,
    /// `ORDER BY` key.
    OrderBy,
    /// Equi-join key.
    JoinKey,
    /// `DISTINCT` / `UNION` (deduplicating) over the column.
    Distinct,
    /// `INTERSECT` / `EXCEPT` over the column.
    SetOp,
    /// A `CASE` result.
    Case,
    /// `CAST` from or to the type.
    Cast,
    /// A scalar function or `LIKE`, by SQL name.
    Function(&'static str),
    /// The result of a scalar or `IN` subquery.
    Subquery,
}

impl Use {
    fn describe(self) -> String {
        match self {
            Use::IsNull => "IS NULL".into(),
            Use::Count => "COUNT".into(),
            Use::Compare(op) | Use::Operator(op) => format!("operator {}", op_symbol(op)),
            Use::Sum => "SUM".into(),
            Use::MinMax => "MIN / MAX".into(),
            Use::Aggregate(name) | Use::Function(name) => name.into(),
            Use::GroupBy => "GROUP BY".into(),
            Use::OrderBy => "ORDER BY".into(),
            Use::JoinKey => "a join key".into(),
            Use::Distinct => "DISTINCT / UNION".into(),
            Use::SetOp => "INTERSECT / EXCEPT".into(),
            Use::Case => "a CASE result".into(),
            Use::Cast => "CAST".into(),
            Use::Subquery => "a subquery result".into(),
        }
    }
}

fn op_symbol(op: BinaryOp) -> &'static str {
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
        BinaryOp::Mod => "%",
        BinaryOp::BitAnd => "&",
        BinaryOp::BitOr => "|",
        BinaryOp::BitXor => "^",
        BinaryOp::Shl => "<<",
        BinaryOp::Shr => ">>",
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

/// The capability table: whether a value of family `f` supports `u`.
/// Temporal arithmetic (`TIMESTAMP ± duration`) is typed by
/// `date_or_timestamp_arith_result` and admitted by [`validate_plan`] before
/// this table is consulted.
pub(crate) fn supports(f: Family, u: Use) -> bool {
    use Family::*;
    match u {
        Use::IsNull | Use::Count => true,
        Use::Compare(_)
        | Use::GroupBy
        | Use::OrderBy
        | Use::JoinKey
        | Use::Distinct
        | Use::SetOp
        | Use::Case => f != Opaque,
        Use::MinMax => matches!(f, Time(_) | Duration(_)),
        Use::Sum => matches!(f, Duration(_)),
        Use::Operator(_) | Use::Aggregate(_) | Use::Cast | Use::Function(_) | Use::Subquery => {
            false
        }
    }
}

/// `Ok` when `dt` is native or its family supports `u`; otherwise a type
/// error naming the operation and the column type.
pub(crate) fn check(dt: DataType, u: Use) -> BoltResult<()> {
    match dt {
        DataType::Extended(t) if !supports(family(t), u) => Err(BoltError::Type(format!(
            "{} is not supported on {t} values",
            u.describe()
        ))),
        _ => Ok(()),
    }
}

/// `true` when two comparison operands of these types type-check: the same
/// family, any unit.
pub(crate) fn comparable(l: DataType, r: DataType) -> bool {
    match (family_of(l), family_of(r)) {
        (Some(Family::Time(_)), Some(Family::Time(_)))
        | (Some(Family::Duration(_)), Some(Family::Duration(_)))
        | (Some(Family::Binary), Some(Family::Binary)) => true,
        (Some(Family::Opaque), Some(Family::Opaque)) => l == r,
        _ => false,
    }
}

/// The type of a constant `Cast` target: the `TIME` / `INTERVAL` / `X'..'`
/// literal forms the SQL frontend builds (`CAST(<Int64 ticks> AS Time64)`,
/// `CAST(<Int64 µs> AS Duration)`, `CAST(<hex Utf8> AS Binary)`).
pub(crate) fn literal_cast_is_supported(src: DataType, target: DataType) -> bool {
    match family_of(target) {
        Some(Family::Time(_) | Family::Duration(_)) => {
            matches!(src, DataType::Int32 | DataType::Int64)
        }
        Some(Family::Binary) => src == DataType::Utf8,
        _ => false,
    }
}

/// `TIME '<text>'` as a constant expression.
pub(crate) fn time_literal(nanos: i64) -> Expr {
    Expr::Cast {
        expr: Box::new(Expr::Literal(Literal::Int64(nanos))),
        target: DataType::from_arrow(&A::Time64(arrow_schema::TimeUnit::Nanosecond))
            .expect("Time64 is an extended type"),
        safe: false,
    }
}

/// A fixed-length `INTERVAL` as a `Duration(Microsecond)` constant.
pub(crate) fn duration_literal(micros: i64) -> Expr {
    Expr::Cast {
        expr: Box::new(Expr::Literal(Literal::Int64(micros))),
        target: DataType::from_arrow(&A::Duration(arrow_schema::TimeUnit::Microsecond))
            .expect("Duration is an extended type"),
        safe: false,
    }
}

/// `X'<hex>'` as a constant expression.
pub(crate) fn binary_literal(bytes: &[u8]) -> Expr {
    Expr::Cast {
        expr: Box::new(Expr::Literal(Literal::Utf8(hex(bytes)))),
        target: DataType::from_arrow(&A::Binary).expect("Binary is an extended type"),
        safe: false,
    }
}

/// Lowercase hex: the storage form of a binary value.
pub(crate) fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(DIGITS[(b >> 4) as usize] as char);
        s.push(DIGITS[(b & 0xf) as usize] as char);
    }
    s
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// The static type of `e`, or `None` for an untyped NULL.
fn dtype_of(e: &Expr, s: &Schema) -> BoltResult<Option<DataType>> {
    match e {
        Expr::Literal(Literal::Null) => Ok(None),
        _ => e.dtype(s).map(Some),
    }
}

fn check_opt(dt: Option<DataType>, u: Use) -> BoltResult<()> {
    dt.map_or(Ok(()), |dt| check(dt, u))
}

fn is_ticks(dt: Option<DataType>) -> bool {
    matches!(dt, Some(DataType::Timestamp(_, _)))
}

fn is_duration(dt: Option<DataType>) -> bool {
    matches!(dt.and_then(family_of), Some(Family::Duration(_)))
}

/// Reject every operation in `plan` (subqueries included) that the
/// extended-type capability table does not allow.
pub(crate) fn validate_plan(plan: &LogicalPlan) -> BoltResult<()> {
    for child in plan.children() {
        validate_plan(child)?;
    }
    let input = || -> BoltResult<Schema> {
        plan.children()
            .first()
            .map_or_else(|| Ok(Schema::default()), |c| c.schema())
    };
    match plan {
        LogicalPlan::Scan { .. } | LogicalPlan::Limit { .. } | LogicalPlan::Union { .. } => {}
        LogicalPlan::Filter { predicate, .. } => visit(predicate, &input()?)?,
        LogicalPlan::Project { exprs, .. } => {
            let s = input()?;
            for e in exprs {
                visit(e, &s)?;
            }
        }
        LogicalPlan::Aggregate {
            group_by,
            aggregates,
            ..
        } => {
            let s = input()?;
            for k in group_by {
                key(k, &s, Use::GroupBy)?;
            }
            for a in aggregates {
                aggregate(a, &s)?;
            }
        }
        LogicalPlan::Distinct { input } => {
            for f in input.schema()?.fields {
                check(f.dtype, Use::Distinct)?;
            }
        }
        LogicalPlan::SetOp { left, .. } => {
            for f in left.schema()?.fields {
                check(f.dtype, Use::SetOp)?;
            }
        }
        LogicalPlan::Sort { sort_exprs, .. } => {
            let s = input()?;
            sort_keys(sort_exprs, &s)?;
        }
        LogicalPlan::Window {
            window_exprs,
            partition_by,
            order_by,
            ..
        } => {
            let s = input()?;
            for k in partition_by {
                key(k, &s, Use::GroupBy)?;
            }
            sort_keys(order_by, &s)?;
            for w in window_exprs {
                let Some(arg) = w.func.arg() else { continue };
                visit(arg, &s)?;
                let u = match &w.func {
                    WindowFunc::Count(_) => Use::Count,
                    WindowFunc::Sum(_) => Use::Sum,
                    WindowFunc::Min(_) | WindowFunc::Max(_) => Use::MinMax,
                    _ => Use::Aggregate(w.func.sql_name()),
                };
                check_opt(dtype_of(arg, &s)?, u)?;
            }
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => {
            let (ls, rs) = (left.schema()?, right.schema()?);
            for (l, r) in on {
                key(l, &ls, Use::JoinKey)?;
                key(r, &rs, Use::JoinKey)?;
                // Each key is lowered against its own side, so the two
                // sides of a time / duration key must share a unit.
                let (lt, rt) = (dtype_of(l, &ls)?, dtype_of(r, &rs)?);
                if tick_unit(lt).is_some() && lt != rt {
                    return Err(BoltError::Type(format!(
                        "join keys of types {} and {} must have the same type; CAST is not                          supported on time and duration values",
                        lt.map_or("NULL".into(), |t| format!("{t:?}")),
                        rt.map_or("NULL".into(), |t| format!("{t:?}"))
                    )));
                }
            }
            if let Some(f) = filter {
                visit(f, &join_combined_schema(&ls, &rs, *join_type))?;
            }
        }
    }
    Ok(())
}

fn key(e: &Expr, s: &Schema, u: Use) -> BoltResult<()> {
    visit(e, s)?;
    check_opt(dtype_of(e, s)?, u)
}

fn sort_keys(keys: &[SortExpr], s: &Schema) -> BoltResult<()> {
    keys.iter().try_for_each(|k| key(&k.expr, s, Use::OrderBy))
}

fn aggregate(a: &AggregateExpr, s: &Schema) -> BoltResult<()> {
    let (e, u) = match a {
        AggregateExpr::Count(e) => (e, Use::Count),
        AggregateExpr::Sum(e) => (e, Use::Sum),
        AggregateExpr::Min(e) | AggregateExpr::Max(e) => (e, Use::MinMax),
        AggregateExpr::Avg(e) => (e, Use::Aggregate("AVG")),
        AggregateExpr::VarPop(e) => (e.as_ref(), Use::Aggregate("VAR_POP")),
        AggregateExpr::VarSamp(e) => (e.as_ref(), Use::Aggregate("VAR_SAMP")),
        AggregateExpr::StddevPop(e) => (e.as_ref(), Use::Aggregate("STDDEV_POP")),
        AggregateExpr::StddevSamp(e) => (e.as_ref(), Use::Aggregate("STDDEV_SAMP")),
    };
    key(e, s, u)
}

/// Validate the operations inside `e` (not the use of `e`'s own value,
/// which the caller checks).
fn visit(e: &Expr, s: &Schema) -> BoltResult<()> {
    match e {
        Expr::Column(_) | Expr::Literal(_) => Ok(()),
        Expr::Alias(x, _) => visit(x, s),
        Expr::Binary { op, left, right } => {
            visit(left, s)?;
            visit(right, s)?;
            let (lt, rt) = (dtype_of(left, s)?, dtype_of(right, s)?);
            if is_comparison(*op) {
                check_opt(lt, Use::Compare(*op))?;
                return check_opt(rt, Use::Compare(*op));
            }
            // TIMESTAMP ± integer ticks exists only inside lowered plans.
            let ticks_int =
                |t: Option<DataType>| matches!(t, Some(DataType::Int32 | DataType::Int64));
            if matches!(op, BinaryOp::Add | BinaryOp::Sub)
                && ((is_ticks(lt) && ticks_int(rt))
                    || (*op == BinaryOp::Add && ticks_int(lt) && is_ticks(rt)))
            {
                return Err(BoltError::Type(format!(
                    "arithmetic {op:?} on Date/Timestamp operands ({:?}, {:?}) is not \
                     supported; add or subtract an INTERVAL",
                    lt.expect("matched"),
                    rt.expect("matched")
                )));
            }
            // TIMESTAMP ± duration; TIMESTAMP - TIMESTAMP is native.
            let temporal = match op {
                BinaryOp::Add => {
                    (is_ticks(lt) && is_duration(rt)) || (is_duration(lt) && is_ticks(rt))
                }
                BinaryOp::Sub => is_ticks(lt) && is_duration(rt),
                _ => false,
            };
            if temporal {
                return Ok(());
            }
            check_opt(lt, Use::Operator(*op))?;
            check_opt(rt, Use::Operator(*op))
        }
        Expr::Unary { op, operand } => {
            visit(operand, s)?;
            match op {
                UnaryOp::IsNull | UnaryOp::IsNotNull => {
                    check_opt(dtype_of(operand, s)?, Use::IsNull)
                }
                UnaryOp::Not => check_opt(dtype_of(operand, s)?, Use::Function("NOT")),
            }
        }
        Expr::Case {
            branches,
            else_branch,
        } => {
            for (w, t) in branches {
                visit(w, s)?;
                visit(t, s)?;
            }
            if let Some(x) = else_branch {
                visit(x, s)?;
            }
            check(e.dtype(s)?, Use::Case)
        }
        Expr::Cast { expr, target, .. } => {
            let src = dtype_of(expr, s)?;
            if family_of(*target).is_some() {
                // Only the constant forms the SQL frontend builds.
                return match expr.as_ref() {
                    Expr::Literal(_) => Ok(()),
                    _ => check(*target, Use::Cast),
                };
            }
            visit(expr, s)?;
            if src.is_some_and(|t| t != *target) {
                check_opt(src, Use::Cast)?;
            }
            // `TIMESTAMP -> integer` casts exist only inside lowered plans.
            match (src, *target) {
                (Some(DataType::Timestamp(_, _)), DataType::Int32 | DataType::Int64) => {
                    Err(BoltError::Type(format!(
                        "unsupported CAST from {:?} to {target:?}",
                        src.expect("matched Some")
                    )))
                }
                _ => Ok(()),
            }
        }
        Expr::CastFormat { expr, .. } => function(expr, s, "CAST ... FORMAT"),
        Expr::Like { expr, .. } => function(expr, s, "LIKE"),
        Expr::Extract { expr, .. } => function(expr, s, "EXTRACT"),
        Expr::DateTrunc { expr, .. } => function(expr, s, "DATE_TRUNC"),
        Expr::ScalarFn { kind, args } => args
            .iter()
            .try_for_each(|a| function(a, s, kind.sql_name())),
        Expr::ScalarSubquery(p) => {
            validate_plan(p)?;
            p.schema()?
                .fields
                .iter()
                .try_for_each(|f| check(f.dtype, Use::Subquery))
        }
        Expr::InSubquery { expr, subquery, .. } => {
            visit(expr, s)?;
            validate_plan(subquery)?;
            check_opt(dtype_of(expr, s)?, Use::Subquery)
        }
    }
}

fn function(arg: &Expr, s: &Schema, name: &'static str) -> BoltResult<()> {
    visit(arg, s)?;
    check_opt(dtype_of(arg, s)?, Use::Function(name))
}

// ---------------------------------------------------------------------------
// Lowering onto storage types
// ---------------------------------------------------------------------------

/// A plan rewritten onto storage types.
#[derive(Debug)]
pub(crate) struct Lowered {
    /// The rewritten plan; it contains no extended type.
    pub(crate) plan: LogicalPlan,
    /// The original plan's output schema when it differs from the rewritten
    /// plan's (some output column is extended): the engine converts the
    /// result's storage columns back to these types.
    pub(crate) restore: Option<Schema>,
    /// `false` when the plan needed no rewrite (`plan` is a copy).
    pub(crate) changed: bool,
}

/// Validate `plan` and rewrite it onto storage types (see the module docs).
pub(crate) fn lower_plan(plan: &LogicalPlan) -> BoltResult<Lowered> {
    validate_plan(plan)?;
    let changed = Cell::new(false);
    let lowered = lower_node(plan, &changed)?;
    if !changed.get() {
        return Ok(Lowered {
            plan: lowered,
            restore: None,
            changed: false,
        });
    }
    let schema = plan.schema()?;
    let restore = schema
        .fields
        .iter()
        .any(|f| family_of(f.dtype).is_some())
        .then_some(schema);
    Ok(Lowered {
        plan: lowered,
        restore,
        changed: true,
    })
}

fn lower_node(plan: &LogicalPlan, changed: &Cell<bool>) -> BoltResult<LogicalPlan> {
    let mapped =
        crate::plan::sql_semantics::map_exprs_with_schema(plan, &|e, s| lower_expr(e, s, changed))?;
    Ok(lower_scans(mapped, changed))
}

/// Rewrite every `Scan` schema onto storage types.
fn lower_scans(plan: LogicalPlan, changed: &Cell<bool>) -> LogicalPlan {
    let rec = |p: Box<LogicalPlan>| Box::new(lower_scans(*p, changed));
    match plan {
        LogicalPlan::Scan {
            table,
            projection,
            mut schema,
        } => {
            for f in &mut schema.fields {
                let st = storage_dtype(f.dtype);
                if st != f.dtype {
                    changed.set(true);
                    f.dtype = st;
                }
            }
            LogicalPlan::Scan {
                table,
                projection,
                schema,
            }
        }
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: rec(input),
            predicate,
        },
        LogicalPlan::Project { input, exprs } => LogicalPlan::Project {
            input: rec(input),
            exprs,
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalPlan::Aggregate {
            input: rec(input),
            group_by,
            aggregates,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct { input: rec(input) },
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => LogicalPlan::Limit {
            input: rec(input),
            limit,
            offset,
        },
        LogicalPlan::Sort { input, sort_exprs } => LogicalPlan::Sort {
            input: rec(input),
            sort_exprs,
        },
        LogicalPlan::Window {
            input,
            window_exprs,
            partition_by,
            order_by,
        } => LogicalPlan::Window {
            input: rec(input),
            window_exprs,
            partition_by,
            order_by,
        },
        LogicalPlan::Union { inputs } => LogicalPlan::Union {
            inputs: inputs
                .into_iter()
                .map(|p| lower_scans(p, changed))
                .collect(),
        },
        LogicalPlan::SetOp {
            left,
            right,
            op,
            all,
        } => LogicalPlan::SetOp {
            left: rec(left),
            right: rec(right),
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
            left: rec(left),
            right: rec(right),
            join_type,
            on,
            filter,
        },
    }
}

/// Ticks of `u` per second.
fn ticks_per_second(u: TimeUnit) -> i64 {
    match u {
        TimeUnit::Second => 1,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    }
}

/// The unit of a time or duration type.
fn tick_unit(dt: Option<DataType>) -> Option<TimeUnit> {
    match dt.and_then(family_of) {
        Some(Family::Time(u) | Family::Duration(u)) => Some(u),
        _ => None,
    }
}

fn int64(e: Expr) -> Expr {
    Expr::Cast {
        expr: Box::new(e),
        target: DataType::Int64,
        safe: false,
    }
}

fn int_value(e: &Expr) -> Option<i64> {
    match e {
        Expr::Literal(Literal::Int32(v)) => Some(*v as i64),
        Expr::Literal(Literal::Int64(v)) => Some(*v),
        _ => None,
    }
}

fn binary(op: BinaryOp, left: Expr, right: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(left),
        right: Box::new(right),
    }
}

/// `e` (lowered, in `from` ticks) in `to` ticks, as an `Int64` expression.
/// Converting to a coarser unit truncates toward zero.
fn rescale(e: Expr, from: TimeUnit, to: TimeUnit) -> Expr {
    let (f, t) = (ticks_per_second(from), ticks_per_second(to));
    if f == t {
        return e;
    }
    if let Some(v) = int_value(&e) {
        let scaled = if t > f {
            v.checked_mul(t / f)
        } else {
            Some(v / (f / t))
        };
        if let Some(v) = scaled {
            return Expr::Literal(Literal::Int64(v));
        }
    }
    let wide = int64(e);
    if t > f {
        binary(BinaryOp::Mul, wide, Expr::Literal(Literal::Int64(t / f)))
    } else {
        binary(BinaryOp::Div, wide, Expr::Literal(Literal::Int64(f / t)))
    }
}

/// Align two lowered tick operands in units `lu` / `ru` for a comparison:
/// the finer unit wins, except that a constant on the finer side that is a
/// whole number of coarser ticks is converted instead, so the column keeps
/// its storage type.
fn align(l: Expr, lu: TimeUnit, r: Expr, ru: TimeUnit) -> (Expr, Expr) {
    if lu == ru {
        return (l, r);
    }
    let (lf, rf) = (ticks_per_second(lu), ticks_per_second(ru));
    // `fine` is the side with more ticks per second.
    let (coarse, cu, fine, fu, swapped) = if lf < rf {
        (l, lu, r, ru, false)
    } else {
        (r, ru, l, lu, true)
    };
    let k = ticks_per_second(fu) / ticks_per_second(cu);
    let (c, f) = match int_value(&fine) {
        Some(v) if v % k == 0 => (coarse, Expr::Literal(Literal::Int64(v / k))),
        _ => (rescale(coarse, cu, fu), int64_if_column(fine)),
    };
    if swapped {
        (f, c)
    } else {
        (c, f)
    }
}

/// Widen a non-constant lowered tick operand to `Int64` (a `Time32` column
/// is stored as `Int32`).
fn int64_if_column(e: Expr) -> Expr {
    match e {
        Expr::Literal(_) => e,
        other => int64(other),
    }
}

/// Rewrite one expression (evaluated against the logical schema `s`) onto
/// storage types.
fn lower_expr(e: &Expr, s: &Schema, changed: &Cell<bool>) -> BoltResult<Expr> {
    let low = |x: &Expr| lower_expr(x, s, changed);
    Ok(match e {
        // The `TIME` / `INTERVAL` / `X'..'` constant forms.
        Expr::Cast { expr, target, .. } if family_of(*target).is_some() => {
            changed.set(true);
            match (expr.as_ref(), storage_dtype(*target)) {
                (Expr::Literal(Literal::Int64(v)), DataType::Int32) => {
                    Expr::Literal(match i32::try_from(*v) {
                        Ok(v) => Literal::Int32(v),
                        Err(_) => Literal::Int64(*v),
                    })
                }
                (Expr::Literal(l), _) => Expr::Literal(l.clone()),
                (_, st) => Expr::Cast {
                    expr: Box::new(low(expr)?),
                    target: st,
                    safe: false,
                },
            }
        }
        Expr::Binary { op, left, right } => {
            let (lt, rt) = (dtype_of(left, s)?, dtype_of(right, s)?);
            let (l, r) = (low(left)?, low(right)?);
            match (op, lt, rt) {
                // TIMESTAMP - TIMESTAMP: a duration of the operands' unit.
                (BinaryOp::Sub, Some(DataType::Timestamp(..)), Some(DataType::Timestamp(..))) => {
                    changed.set(true);
                    binary(BinaryOp::Sub, int64(l), int64(r))
                }
                // TIMESTAMP ± duration: plus or minus ticks of the
                // timestamp's unit.
                (BinaryOp::Add | BinaryOp::Sub, Some(DataType::Timestamp(u, _)), d)
                    if is_duration(d) =>
                {
                    changed.set(true);
                    binary(*op, l, rescale(r, tick_unit(d).expect("duration"), u))
                }
                (BinaryOp::Add, d, Some(DataType::Timestamp(u, _))) if is_duration(d) => {
                    changed.set(true);
                    binary(
                        BinaryOp::Add,
                        rescale(l, tick_unit(d).expect("duration"), u),
                        r,
                    )
                }
                (op, lt, rt) if is_comparison(*op) => match (tick_unit(lt), tick_unit(rt)) {
                    (Some(lu), Some(ru)) if lu != ru => {
                        changed.set(true);
                        let (l, r) = align(l, lu, r, ru);
                        binary(*op, l, r)
                    }
                    _ => binary(*op, l, r),
                },
                _ => binary(*op, l, r),
            }
        }
        Expr::ScalarSubquery(p) => Expr::ScalarSubquery(Box::new(lower_node(p, changed)?)),
        Expr::InSubquery {
            expr,
            subquery,
            negated,
        } => Expr::InSubquery {
            expr: Box::new(low(expr)?),
            subquery: Box::new(lower_node(subquery, changed)?),
            negated: *negated,
        },
        other => crate::plan::sql_semantics::map_children(other, &mut |c| low(c))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use arrow_schema::{Field as ArrowField, IntervalUnit, TimeUnit as AU};
    use std::sync::Arc;

    fn ext(t: A) -> DataType {
        DataType::from_arrow(&t).unwrap()
    }

    #[test]
    fn from_arrow_maps_native_normalized_and_extended_types() {
        assert_eq!(DataType::from_arrow(&A::Int64).unwrap(), DataType::Int64);
        assert_eq!(DataType::from_arrow(&A::LargeUtf8).unwrap(), DataType::Utf8);
        assert_eq!(DataType::from_arrow(&A::Utf8View).unwrap(), DataType::Utf8);
        let list = A::List(Arc::new(ArrowField::new("item", A::Int32, true)));
        for t in [
            A::Time32(AU::Millisecond),
            A::Time64(AU::Nanosecond),
            A::Duration(AU::Second),
            A::Binary,
            A::LargeBinary,
            A::FixedSizeBinary(16),
            A::Interval(IntervalUnit::MonthDayNano),
            A::Decimal256(40, 2),
            A::Null,
            list.clone(),
        ] {
            match DataType::from_arrow(&t).unwrap() {
                DataType::Extended(got) => assert_eq!(got, &t),
                other => panic!("{t}: expected Extended, got {other:?}"),
            }
        }
        // Interned: equal Arrow types give identical (and equal) values.
        assert_eq!(ext(list.clone()), ext(list));
        // A dictionary of an extended value type is its value type.
        let dict = A::Dictionary(Box::new(A::Int32), Box::new(A::Binary));
        assert_eq!(DataType::from_arrow(&dict).unwrap(), ext(A::Binary));
        // Types the engine still rejects name themselves.
        let err = DataType::from_arrow(&A::UInt16).unwrap_err().to_string();
        assert!(err.contains("UInt16"), "{err}");
    }

    #[test]
    fn storage_types_follow_the_family() {
        assert_eq!(storage_dtype(ext(A::Time32(AU::Second))), DataType::Int32);
        assert_eq!(
            storage_dtype(ext(A::Time64(AU::Microsecond))),
            DataType::Int64
        );
        assert_eq!(
            storage_dtype(ext(A::Duration(AU::Millisecond))),
            DataType::Int64
        );
        assert_eq!(storage_dtype(ext(A::FixedSizeBinary(16))), DataType::Utf8);
        assert_eq!(storage_dtype(ext(A::Decimal256(40, 2))), DataType::Int64);
        assert_eq!(storage_dtype(DataType::Float32), DataType::Float32);
    }

    #[test]
    fn capability_table() {
        let t = family(&A::Time64(AU::Microsecond));
        let b = family(&A::Binary);
        let o = family(&A::Null);
        assert!(supports(t, Use::MinMax) && supports(t, Use::GroupBy));
        assert!(!supports(t, Use::Sum) && !supports(t, Use::Operator(BinaryOp::Add)));
        assert!(supports(Family::Duration(TimeUnit::Second), Use::Sum));
        assert!(supports(b, Use::Compare(BinaryOp::Eq)) && supports(b, Use::OrderBy));
        assert!(!supports(b, Use::MinMax) && !supports(b, Use::Function("UPPER")));
        for u in [Use::IsNull, Use::Count] {
            assert!(supports(o, u));
        }
        for u in [
            Use::Compare(BinaryOp::Eq),
            Use::GroupBy,
            Use::OrderBy,
            Use::Distinct,
        ] {
            assert!(!supports(o, u));
        }
        let err = check(ext(A::Decimal256(40, 2)), Use::OrderBy)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("ORDER BY") && err.contains("Decimal256"),
            "{err}"
        );
    }

    #[test]
    fn hex_keeps_byte_order() {
        let mut v: Vec<Vec<u8>> = vec![vec![0xff], vec![0, 1], vec![0x10], vec![], vec![0, 0xff]];
        let mut by_hex = v.clone();
        v.sort();
        by_hex.sort_by_key(|b| hex(b));
        assert_eq!(v, by_hex);
        assert_eq!(hex(&[0x0a, 0xff]), "0aff");
    }

    fn scan() -> LogicalPlan {
        LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("k", DataType::Int64, true),
                Field::new("tm", ext(A::Time64(AU::Microsecond)), true),
                Field::new("ts", ext(A::Time32(AU::Second)), true),
                Field::new("l", ext(A::Null), true),
            ]),
        }
    }

    fn filter(predicate: Expr) -> LogicalPlan {
        LogicalPlan::Filter {
            input: Box::new(scan()),
            predicate,
        }
    }

    fn col(n: &str) -> Expr {
        Expr::Column(n.into())
    }

    #[test]
    fn validation_rejects_unsupported_uses() {
        let sort = LogicalPlan::Sort {
            input: Box::new(scan()),
            sort_exprs: vec![SortExpr {
                expr: col("l"),
                descending: false,
                nulls_first: false,
            }],
        };
        let err = validate_plan(&sort).unwrap_err().to_string();
        assert!(err.contains("ORDER BY") && err.contains("Null"), "{err}");
        let cmp = filter(binary(BinaryOp::Eq, col("l"), col("l")));
        assert!(validate_plan(&cmp).is_err());
        let distinct = LogicalPlan::Distinct {
            input: Box::new(scan()),
        };
        assert!(validate_plan(&distinct).is_err());
        // Passthrough, IS NULL and COUNT are fine.
        let ok = LogicalPlan::Aggregate {
            input: Box::new(filter(Expr::Unary {
                op: UnaryOp::IsNull,
                operand: Box::new(col("l")),
            })),
            group_by: vec![col("tm")],
            aggregates: vec![
                AggregateExpr::Count(col("l")),
                AggregateExpr::Max(col("ts")),
            ],
        };
        validate_plan(&ok).unwrap();
    }

    #[test]
    fn lowering_rescales_time_comparisons_and_rewrites_scans() {
        // tm (µs) > TIME '00:00:01' (ns): the constant becomes µs.
        let p = filter(binary(BinaryOp::Gt, col("tm"), time_literal(1_000_000_000)));
        let low = lower_plan(&p).unwrap();
        assert!(low.changed && low.restore.is_some());
        let LogicalPlan::Filter { input, predicate } = &low.plan else {
            panic!()
        };
        assert_eq!(
            format!("{predicate:?}"),
            format!(
                "{:?}",
                binary(
                    BinaryOp::Gt,
                    col("tm"),
                    Expr::Literal(Literal::Int64(1_000_000))
                )
            )
        );
        let s = input.schema().unwrap();
        assert_eq!(s.fields[1].dtype, DataType::Int64);
        assert_eq!(s.fields[2].dtype, DataType::Int32);
        assert_eq!(low.plan.schema().unwrap().fields[3].dtype, DataType::Int64);
        // tm (µs) = ts (s): the seconds column is scaled up.
        let p = filter(binary(BinaryOp::Eq, col("tm"), col("ts")));
        let low = lower_plan(&p).unwrap();
        let LogicalPlan::Filter { predicate, .. } = &low.plan else {
            panic!()
        };
        let want = binary(
            BinaryOp::Eq,
            int64(col("tm")),
            binary(
                BinaryOp::Mul,
                int64(col("ts")),
                Expr::Literal(Literal::Int64(1_000_000)),
            ),
        );
        assert_eq!(format!("{predicate:?}"), format!("{want:?}"));
        low.plan.schema().unwrap();
    }

    #[test]
    fn plans_without_extended_types_are_unchanged() {
        let p = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("k", DataType::Int64, true)]),
        };
        let low = lower_plan(&p).unwrap();
        assert!(!low.changed && low.restore.is_none());
    }

    #[test]
    fn timestamp_difference_is_typed_duration_and_lowers_to_ticks() {
        let ts = DataType::Timestamp(TimeUnit::Microsecond, None);
        let p = LogicalPlan::Project {
            input: Box::new(LogicalPlan::Scan {
                table: "w".into(),
                projection: None,
                schema: Schema::new(vec![Field::new("a", ts, true), Field::new("b", ts, true)]),
            }),
            exprs: vec![binary(BinaryOp::Sub, col("a"), col("b"))],
        };
        let logical = p.schema().unwrap();
        assert_eq!(logical.fields[0].dtype, ext(A::Duration(AU::Microsecond)));
        let low = lower_plan(&p).unwrap();
        assert_eq!(low.plan.schema().unwrap().fields[0].dtype, DataType::Int64);
        assert_eq!(
            low.restore.unwrap().fields[0].dtype,
            logical.fields[0].dtype
        );
        // a - INTERVAL '1 second' stays a timestamp; the µs constant is used as is.
        let p = LogicalPlan::Project {
            input: Box::new(LogicalPlan::Scan {
                table: "w".into(),
                projection: None,
                schema: Schema::new(vec![Field::new("a", ts, true)]),
            }),
            exprs: vec![binary(BinaryOp::Sub, col("a"), duration_literal(1_000_000))],
        };
        assert_eq!(p.schema().unwrap().fields[0].dtype, ts);
        let low = lower_plan(&p).unwrap();
        assert_eq!(low.plan.schema().unwrap().fields[0].dtype, ts);
        assert!(low.restore.is_none());
    }
}
