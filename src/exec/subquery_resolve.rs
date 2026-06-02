// SPDX-License-Identifier: Apache-2.0

//! Pre-lowering resolution of uncorrelated subqueries.
//!
//! `Expr::ScalarSubquery` and `Expr::InSubquery` parse + type-check in the SQL
//! frontend, and correlated subqueries are rejected there — so every subquery
//! that survives to the engine is *uncorrelated*, meaning its boxed
//! [`LogicalPlan`] is a self-contained, independently-executable query that
//! references no columns from the enclosing query.
//!
//! This module turns those subqueries into plain constants *before* physical
//! lowering. It walks the plan's expressions, executes each subplan via a
//! caller-supplied executor closure, and rewrites:
//!
//! * `ScalarSubquery(subplan)` → the single produced value as an
//!   `Expr::Literal` (0 rows → SQL `NULL`; >1 row → a clean error).
//! * `InSubquery { expr, subquery, negated }` → a boolean fold of equalities
//!   over `expr` (`expr = v1 OR expr = v2 …`, or the negated `<>`/`AND` form).
//!
//! Resolution is *inner-first*: subqueries nested inside another subquery's
//! subplan are resolved when that subplan is executed (the executor closure
//! runs the full engine pipeline, which itself re-enters this pass), and
//! subqueries appearing as siblings recurse normally.
//!
//! # Why a closure rather than a direct `&Engine` dependency?
//!
//! The value-extraction and IN-list-build helpers are pure functions over an
//! Arrow [`RecordBatch`] / `&[Literal]`, with no GPU or engine state, so they
//! are unit-tested on the host. The plan walker is generic over a
//! `FnMut(LogicalPlan) -> BoltResult<RecordBatch>` executor so the engine can
//! inject its `&self` execution path without this module taking an `Engine`
//! dependency.

use arrow_array::{
    Array, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int32Array,
    Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_schema::{DataType as ArrowDataType, TimeUnit as ArrowTimeUnit};

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{
    AggregateExpr, BinaryOp, Expr, Literal, LogicalPlan, SortExpr, UnaryOp,
};

/// Extract the value at row `row` of the (single) first column of `batch` as a
/// [`Literal`]. A null at that position yields [`Literal::Null`]. Unsupported
/// Arrow dtypes are rejected with a clean [`BoltError`].
///
/// Supports the dtype set the engine can produce as a subquery output:
/// Int32 / Int64 / Float32 / Float64 / Bool / Utf8 / Date32 / Timestamp
/// (all four resolutions) / Decimal128.
fn literal_from_column(batch: &RecordBatch, row: usize) -> BoltResult<Literal> {
    let col = batch.column(0);
    if col.is_null(row) {
        return Ok(Literal::Null);
    }
    macro_rules! downcast {
        ($ty:ty, $what:literal) => {
            col.as_any().downcast_ref::<$ty>().ok_or_else(|| {
                BoltError::Other(format!(
                    "subquery result column claimed dtype {:?} but did not downcast to {}",
                    col.data_type(),
                    $what
                ))
            })?
        };
    }
    let lit = match col.data_type() {
        ArrowDataType::Int32 => Literal::Int32(downcast!(Int32Array, "Int32Array").value(row)),
        ArrowDataType::Int64 => Literal::Int64(downcast!(Int64Array, "Int64Array").value(row)),
        ArrowDataType::Float32 => {
            Literal::Float32(downcast!(Float32Array, "Float32Array").value(row))
        }
        ArrowDataType::Float64 => {
            Literal::Float64(downcast!(Float64Array, "Float64Array").value(row))
        }
        ArrowDataType::Boolean => Literal::Bool(downcast!(BooleanArray, "BooleanArray").value(row)),
        ArrowDataType::Utf8 => {
            Literal::Utf8(downcast!(StringArray, "StringArray").value(row).to_string())
        }
        ArrowDataType::Date32 => Literal::Date32(downcast!(Date32Array, "Date32Array").value(row)),
        ArrowDataType::Decimal128(p, s) => {
            let v = downcast!(Decimal128Array, "Decimal128Array").value(row);
            Literal::Decimal128(v, *p, *s)
        }
        ArrowDataType::Timestamp(unit, tz) => {
            let ticks = match unit {
                ArrowTimeUnit::Second => {
                    downcast!(TimestampSecondArray, "TimestampSecondArray").value(row)
                }
                ArrowTimeUnit::Millisecond => {
                    downcast!(TimestampMillisecondArray, "TimestampMillisecondArray").value(row)
                }
                ArrowTimeUnit::Microsecond => {
                    downcast!(TimestampMicrosecondArray, "TimestampMicrosecondArray").value(row)
                }
                ArrowTimeUnit::Nanosecond => {
                    downcast!(TimestampNanosecondArray, "TimestampNanosecondArray").value(row)
                }
            };
            let plan_unit = crate::exec::schema_convert::arrow_time_unit_to_plan(unit);
            Literal::timestamp_with_tz(ticks, plan_unit, tz.as_deref().map(|s| s.to_string()))
        }
        other => {
            return Err(BoltError::Plan(format!(
                "subquery result dtype {other:?} is not supported for constant folding"
            )))
        }
    };
    Ok(lit)
}

/// Reduce a scalar-subquery result `batch` to a single [`Literal`].
///
/// Contract (SQL scalar subquery):
/// * the batch must have **exactly one column** (the frontend already
///   type-checks this, but we re-verify defensively);
/// * **0 rows** → SQL `NULL` ([`Literal::Null`]);
/// * **1 row** → that value;
/// * **>1 row** → a clean [`BoltError`] (scalar subquery returned more than
///   one row).
pub fn scalar_value_from_batch(batch: &RecordBatch) -> BoltResult<Literal> {
    if batch.num_columns() != 1 {
        return Err(BoltError::Plan(format!(
            "scalar subquery must return exactly one column, got {}",
            batch.num_columns()
        )));
    }
    match batch.num_rows() {
        0 => Ok(Literal::Null),
        1 => literal_from_column(batch, 0),
        n => Err(BoltError::Plan(format!(
            "scalar subquery returned {n} rows; expected at most one"
        ))),
    }
}

/// Collect the **distinct** values of the (single) first column of `batch` as
/// [`Literal`]s, preserving first-seen order.
///
/// The batch must have exactly one column. `NULL`s are collected as
/// [`Literal::Null`] (at most one, deduped like any other value) so the
/// IN-list builder can reason about their presence; see
/// [`build_in_predicate`] for how SQL three-valued `NULL` membership is
/// handled.
pub fn in_set_from_batch(batch: &RecordBatch) -> BoltResult<Vec<Literal>> {
    if batch.num_columns() != 1 {
        return Err(BoltError::Plan(format!(
            "IN subquery must return exactly one column, got {}",
            batch.num_columns()
        )));
    }
    let mut out: Vec<Literal> = Vec::new();
    for row in 0..batch.num_rows() {
        let lit = literal_from_column(batch, row)?;
        if !out.iter().any(|existing| literal_eq(existing, &lit)) {
            out.push(lit);
        }
    }
    Ok(out)
}

/// Structural equality for two literals, treating two `Null`s as equal so the
/// distinct-collection step dedups them. NaN floats compare unequal (matching
/// IEEE-754 / Rust `PartialEq`), so two NaN entries are both retained — that
/// is harmless for the OR-of-equalities we build (a NaN probe never matches a
/// NaN literal under `=` anyway).
fn literal_eq(a: &Literal, b: &Literal) -> bool {
    match (a, b) {
        (Literal::Null, Literal::Null) => true,
        _ => a == b,
    }
}

/// Build the boolean expression that replaces an `expr [NOT] IN (subquery)`
/// node once the subquery's value set is known.
///
/// `values` is the distinct set produced by [`in_set_from_batch`]. The result:
///
/// * **`IN` (not negated):** `expr = v1 OR expr = v2 OR …`. An empty set →
///   `Bool(false)` (nothing is a member of the empty set).
/// * **`NOT IN` (negated):** `expr <> v1 AND expr <> v2 AND …`. An empty set →
///   `Bool(true)`.
///
/// # NULL handling (strict SQL three-valued logic — finding F-6)
///
/// Strict SQL says:
/// * `x IN (… , NULL , …)` is `TRUE` if `x` matches a non-NULL element, else
///   `NULL` (never `FALSE`) — so a row whose `x` matches nothing but where the
///   set contains a NULL evaluates to `NULL` (filtered out by `WHERE`, same as
///   `FALSE`).
/// * `x NOT IN (…, NULL, …)` is `FALSE` if `x` matches a non-NULL element,
///   else `NULL` (never `TRUE`) — so when the set contains *any* NULL **no
///   row can pass**: a match makes it `FALSE`, a non-match makes it `NULL`,
///   and both are excluded by `WHERE`.
///
/// We **drop `NULL`s from the value set** before building the fold. For the
/// non-negated `IN` form this matches SQL exactly under a `WHERE` clause: a row
/// that doesn't match any non-NULL element yields `FALSE` here vs `NULL` in
/// strict SQL, and both are filtered out.
///
/// For the negated `NOT IN` form we honour the strict semantics: if the value
/// set contains **any** NULL, the predicate can never be `TRUE` for any row, so
/// we fold straight to `Bool(false)` (no rows pass). Only when the set is
/// NULL-free do we build the per-row `<>`/`AND` fold over the non-NULL
/// elements. This closes the classic `x NOT IN (SELECT nullable_col …)` footgun
/// that previously let rows through incorrectly.
pub fn build_in_predicate(expr: &Expr, values: &[Literal], negated: bool) -> Expr {
    // Does the value set contain a NULL? Under SQL 3VL, equality / inequality
    // against a NULL literal yields UNKNOWN, never TRUE.
    let set_has_null = values.iter().any(|l| matches!(l, Literal::Null));

    // F-6: strict `NOT IN` semantics. If the set contains any NULL, the whole
    // negated predicate is UNKNOWN for every row (a match → FALSE, a non-match
    // → NULL), so no row passes. Fold to `Bool(false)`. Note this also subsumes
    // the "set of only NULLs" case for the negated form below.
    if negated && set_has_null {
        return Expr::Literal(Literal::Bool(false));
    }

    // Drop NULLs: equality / inequality against a NULL literal is never TRUE
    // in SQL, so a NULL element can only ever contribute UNKNOWN. For the
    // negated form we have already returned above when a NULL was present, so
    // by this point `negated` implies a NULL-free set.
    let non_null: Vec<&Literal> = values
        .iter()
        .filter(|l| !matches!(l, Literal::Null))
        .collect();

    if non_null.is_empty() {
        // Empty membership set: `IN` → false, `NOT IN` → true. (A set of only
        // NULLs reaches here for the non-negated `IN` form, which is also
        // `false`; the negated form was already handled above.)
        return Expr::Literal(Literal::Bool(negated));
    }

    let (cmp_op, fold_op) = if negated {
        (BinaryOp::NotEq, BinaryOp::And)
    } else {
        (BinaryOp::Eq, BinaryOp::Or)
    };

    let mut iter = non_null.into_iter().map(|v| Expr::Binary {
        op: cmp_op,
        left: Box::new(expr.clone()),
        right: Box::new(Expr::Literal(v.clone())),
    });
    // `non_null` is non-empty, so `next()` is `Some`.
    let first = iter.next().expect("non_null checked non-empty");
    let folded = iter.fold(first, |acc, eq| Expr::Binary {
        op: fold_op,
        left: Box::new(acc),
        right: Box::new(eq),
    });

    if negated {
        // SQL 3VL: `expr NOT IN (set)` evaluates to UNKNOWN (→ row excluded
        // under a WHERE clause) whenever `expr` itself is NULL, regardless of
        // the set contents. The lowered `expr <> v AND …` does NOT capture
        // this: the GPU `<>` comparator reads a NULL probe as its raw stored
        // value (e.g. 0) and would wrongly include it. AND in an explicit
        // `expr IS NOT NULL` guard so NULL probe rows are dropped. (For the
        // non-negated `IN` form a NULL probe yields UNKNOWN through the `=`
        // fold and is already excluded under WHERE, so no guard is added there.)
        Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(folded),
            right: Box::new(Expr::Unary {
                op: UnaryOp::IsNotNull,
                operand: Box::new(expr.clone()),
            }),
        }
    } else {
        folded
    }
}

/// Recursively resolve every subquery in `plan`, executing subplans via
/// `exec`.
///
/// `exec` runs a self-contained [`LogicalPlan`] end-to-end and returns its
/// result [`RecordBatch`]. The executor is expected to itself route through
/// the engine pipeline (including *this* pass), which is what makes nested
/// subqueries resolve inner-first.
pub fn resolve_plan<F>(plan: LogicalPlan, exec: &mut F) -> BoltResult<LogicalPlan>
where
    F: FnMut(LogicalPlan) -> BoltResult<RecordBatch>,
{
    Ok(match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(resolve_plan(*input, exec)?),
            predicate: resolve_expr(predicate, exec)?,
        },
        LogicalPlan::Project { input, exprs } => LogicalPlan::Project {
            input: Box::new(resolve_plan(*input, exec)?),
            exprs: resolve_exprs(exprs, exec)?,
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalPlan::Aggregate {
            input: Box::new(resolve_plan(*input, exec)?),
            group_by: resolve_exprs(group_by, exec)?,
            aggregates: aggregates
                .into_iter()
                .map(|a| resolve_aggregate(a, exec))
                .collect::<BoltResult<Vec<_>>>()?,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(resolve_plan(*input, exec)?),
        },
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => LogicalPlan::Limit {
            input: Box::new(resolve_plan(*input, exec)?),
            limit,
            offset,
        },
        LogicalPlan::Sort { input, sort_exprs } => LogicalPlan::Sort {
            input: Box::new(resolve_plan(*input, exec)?),
            sort_exprs: sort_exprs
                .into_iter()
                .map(|s| {
                    Ok::<SortExpr, BoltError>(SortExpr {
                        expr: resolve_expr(s.expr, exec)?,
                        descending: s.descending,
                        nulls_first: s.nulls_first,
                    })
                })
                .collect::<BoltResult<Vec<_>>>()?,
        },
        LogicalPlan::Window {
            input,
            window_exprs,
            partition_by,
            order_by,
        } => LogicalPlan::Window {
            input: Box::new(resolve_plan(*input, exec)?),
            // WindowExpr's inner argument is a column/expr that the SQL
            // frontend does not currently allow a subquery inside; the
            // partition/order keys are plain exprs we still walk for safety.
            window_exprs,
            partition_by: resolve_exprs(partition_by, exec)?,
            order_by: order_by
                .into_iter()
                .map(|s| {
                    Ok::<SortExpr, BoltError>(SortExpr {
                        expr: resolve_expr(s.expr, exec)?,
                        descending: s.descending,
                        nulls_first: s.nulls_first,
                    })
                })
                .collect::<BoltResult<Vec<_>>>()?,
        },
        LogicalPlan::Union { inputs } => LogicalPlan::Union {
            inputs: inputs
                .into_iter()
                .map(|p| resolve_plan(p, exec))
                .collect::<BoltResult<Vec<_>>>()?,
        },
        LogicalPlan::SetOp {
            left,
            right,
            op,
            all,
        } => LogicalPlan::SetOp {
            left: Box::new(resolve_plan(*left, exec)?),
            right: Box::new(resolve_plan(*right, exec)?),
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
            left: Box::new(resolve_plan(*left, exec)?),
            right: Box::new(resolve_plan(*right, exec)?),
            join_type,
            on: on
                .into_iter()
                .map(|(l, r)| Ok::<_, BoltError>((resolve_expr(l, exec)?, resolve_expr(r, exec)?)))
                .collect::<BoltResult<Vec<_>>>()?,
            filter: filter.map(|f| resolve_expr(f, exec)).transpose()?,
        },
    })
}

/// Resolve every expression in a `Vec`.
fn resolve_exprs<F>(exprs: Vec<Expr>, exec: &mut F) -> BoltResult<Vec<Expr>>
where
    F: FnMut(LogicalPlan) -> BoltResult<RecordBatch>,
{
    exprs.into_iter().map(|e| resolve_expr(e, exec)).collect()
}

/// Resolve the inner expression(s) of an [`AggregateExpr`].
fn resolve_aggregate<F>(agg: AggregateExpr, exec: &mut F) -> BoltResult<AggregateExpr>
where
    F: FnMut(LogicalPlan) -> BoltResult<RecordBatch>,
{
    Ok(match agg {
        AggregateExpr::Count(e) => AggregateExpr::Count(resolve_expr(e, exec)?),
        AggregateExpr::Sum(e) => AggregateExpr::Sum(resolve_expr(e, exec)?),
        AggregateExpr::Min(e) => AggregateExpr::Min(resolve_expr(e, exec)?),
        AggregateExpr::Max(e) => AggregateExpr::Max(resolve_expr(e, exec)?),
        AggregateExpr::Avg(e) => AggregateExpr::Avg(resolve_expr(e, exec)?),
        AggregateExpr::VarPop(e) => AggregateExpr::VarPop(Box::new(resolve_expr(*e, exec)?)),
        AggregateExpr::VarSamp(e) => AggregateExpr::VarSamp(Box::new(resolve_expr(*e, exec)?)),
        AggregateExpr::StddevPop(e) => AggregateExpr::StddevPop(Box::new(resolve_expr(*e, exec)?)),
        AggregateExpr::StddevSamp(e) => {
            AggregateExpr::StddevSamp(Box::new(resolve_expr(*e, exec)?))
        }
    })
}

/// Recursively resolve subqueries in a single [`Expr`].
///
/// For the two subquery variants the subplan is itself run through
/// `resolve_plan` first (inner subqueries resolve before the outer one
/// executes), then executed via `exec`, then folded to a constant.
fn resolve_expr<F>(expr: Expr, exec: &mut F) -> BoltResult<Expr>
where
    F: FnMut(LogicalPlan) -> BoltResult<RecordBatch>,
{
    Ok(match expr {
        Expr::Column(_) | Expr::Literal(_) => expr,
        Expr::Binary { op, left, right } => Expr::Binary {
            op,
            left: Box::new(resolve_expr(*left, exec)?),
            right: Box::new(resolve_expr(*right, exec)?),
        },
        Expr::Unary { op, operand } => Expr::Unary {
            op,
            operand: Box::new(resolve_expr(*operand, exec)?),
        },
        Expr::Case {
            branches,
            else_branch,
        } => Expr::Case {
            branches: branches
                .into_iter()
                .map(|(w, t)| {
                    Ok::<_, BoltError>((resolve_expr(w, exec)?, resolve_expr(t, exec)?))
                })
                .collect::<BoltResult<Vec<_>>>()?,
            else_branch: else_branch
                .map(|e| Ok::<_, BoltError>(Box::new(resolve_expr(*e, exec)?)))
                .transpose()?,
        },
        Expr::Like {
            expr,
            pattern,
            escape,
            negated,
            case_insensitive,
        } => Expr::Like {
            expr: Box::new(resolve_expr(*expr, exec)?),
            pattern,
            escape,
            negated,
            case_insensitive,
        },
        Expr::Cast { expr, target, safe } => Expr::Cast {
            expr: Box::new(resolve_expr(*expr, exec)?),
            target,
            safe,
        },
        Expr::CastFormat { expr, target, pattern, to_text } => Expr::CastFormat {
            expr: Box::new(resolve_expr(*expr, exec)?),
            target,
            pattern,
            to_text,
        },
        Expr::ScalarFn { kind, args } => Expr::ScalarFn {
            kind,
            args: resolve_exprs(args, exec)?,
        },
        Expr::Extract { field, expr } => Expr::Extract {
            field,
            expr: Box::new(resolve_expr(*expr, exec)?),
        },
        Expr::DateTrunc { unit, expr } => Expr::DateTrunc {
            unit,
            expr: Box::new(resolve_expr(*expr, exec)?),
        },
        Expr::Alias(inner, name) => {
            Expr::Alias(Box::new(resolve_expr(*inner, exec)?), name)
        }
        Expr::ScalarSubquery(subplan) => {
            // Resolve inner subqueries first, then execute, then fold.
            let resolved = resolve_plan(*subplan, exec)?;
            let batch = exec(resolved)?;
            let lit = scalar_value_from_batch(&batch)?;
            Expr::Literal(lit)
        }
        Expr::InSubquery {
            expr,
            subquery,
            negated,
        } => {
            // The probe `expr` lives in the *outer* query's schema and may
            // itself contain a subquery — resolve it too.
            let probe = resolve_expr(*expr, exec)?;
            let resolved_sub = resolve_plan(*subquery, exec)?;
            let batch = exec(resolved_sub)?;
            let values = in_set_from_batch(&batch)?;
            build_in_predicate(&probe, &values, negated)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{Int32Array, Int64Array, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    fn single_col_batch(arr: arrow_array::ArrayRef) -> RecordBatch {
        let field = ArrowField::new("c", arr.data_type().clone(), true);
        let schema = Arc::new(ArrowSchema::new(vec![field]));
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    #[test]
    fn scalar_zero_rows_is_null() {
        let arr = Arc::new(Int64Array::from(Vec::<i64>::new())) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        assert_eq!(scalar_value_from_batch(&b).unwrap(), Literal::Null);
    }

    #[test]
    fn scalar_one_row_int64() {
        let arr = Arc::new(Int64Array::from(vec![42_i64])) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        assert_eq!(scalar_value_from_batch(&b).unwrap(), Literal::Int64(42));
    }

    #[test]
    fn scalar_one_row_null_value() {
        let arr = Arc::new(Int32Array::from(vec![None::<i32>])) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        assert_eq!(scalar_value_from_batch(&b).unwrap(), Literal::Null);
    }

    #[test]
    fn scalar_many_rows_errors() {
        let arr = Arc::new(Int64Array::from(vec![1_i64, 2])) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        let err = scalar_value_from_batch(&b).unwrap_err();
        assert!(format!("{err}").contains("returned 2 rows"), "{err}");
    }

    #[test]
    fn scalar_rejects_multi_column() {
        let a = Arc::new(Int64Array::from(vec![1_i64])) as arrow_array::ArrayRef;
        let b = Arc::new(Int64Array::from(vec![2_i64])) as arrow_array::ArrayRef;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int64, true),
            ArrowField::new("b", ArrowDataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![a, b]).unwrap();
        assert!(scalar_value_from_batch(&batch).is_err());
    }

    #[test]
    fn in_set_dedups_preserving_order() {
        let arr = Arc::new(Int32Array::from(vec![3, 1, 3, 2, 1])) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        let set = in_set_from_batch(&b).unwrap();
        assert_eq!(
            set,
            vec![Literal::Int32(3), Literal::Int32(1), Literal::Int32(2)]
        );
    }

    #[test]
    fn in_set_utf8() {
        let arr = Arc::new(StringArray::from(vec!["x", "y", "x"])) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        let set = in_set_from_batch(&b).unwrap();
        assert_eq!(
            set,
            vec![Literal::Utf8("x".into()), Literal::Utf8("y".into())]
        );
    }

    #[test]
    fn build_in_empty_set() {
        let probe = Expr::Column("x".into());
        assert!(matches!(
            build_in_predicate(&probe, &[], false),
            Expr::Literal(Literal::Bool(false))
        ));
        assert!(matches!(
            build_in_predicate(&probe, &[], true),
            Expr::Literal(Literal::Bool(true))
        ));
    }

    #[test]
    fn build_in_only_nulls_set() {
        let probe = Expr::Column("x".into());
        // A set of only NULLs collapses to the empty non-null case.
        assert!(matches!(
            build_in_predicate(&probe, &[Literal::Null], false),
            Expr::Literal(Literal::Bool(false))
        ));
    }

    #[test]
    fn build_in_or_of_equalities() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(&probe, &[Literal::Int32(1), Literal::Int32(2)], false);
        // `Expr` doesn't implement `PartialEq`, so destructure and compare the
        // structure / scalar leaves (which do) instead of `assert_eq!`.
        match got {
            Expr::Binary { op: BinaryOp::Or, left, right } => {
                check_cmp(&left, "x", BinaryOp::Eq, Literal::Int32(1));
                check_cmp(&right, "x", BinaryOp::Eq, Literal::Int32(2));
            }
            other => panic!("expected OR of equalities, got {other:?}"),
        }
    }

    /// Asserts `e` is `Binary { op, Column(col), Literal(lit) }`.
    fn check_cmp(e: &Expr, col: &str, op: BinaryOp, lit: Literal) {
        match e {
            Expr::Binary { op: got_op, left, right } => {
                assert_eq!(*got_op, op, "binary op");
                match (&**left, &**right) {
                    (Expr::Column(name), Expr::Literal(got_lit)) => {
                        assert_eq!(name.as_str(), col, "column name");
                        assert_eq!(*got_lit, lit, "literal");
                    }
                    other => panic!("expected Column op Literal, got {other:?}"),
                }
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    #[test]
    fn build_not_in_and_of_inequalities() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(&probe, &[Literal::Int32(1), Literal::Int32(2)], true);
        // NOT IN lowers to `(x <> 1 AND x <> 2) AND x IS NOT NULL` — the trailing
        // IS NOT NULL guard drops NULL probe rows (SQL 3VL: NULL NOT IN ... is
        // UNKNOWN → excluded under WHERE).
        match got {
            Expr::Binary { op: BinaryOp::And, left, right } => {
                // right-hand operand is the IS NOT NULL guard over the probe.
                match &*right {
                    Expr::Unary { op: UnaryOp::IsNotNull, operand } => match &**operand {
                        Expr::Column(name) => assert_eq!(name.as_str(), "x"),
                        other => panic!("expected Column in IS NOT NULL, got {other:?}"),
                    },
                    other => panic!("expected IS NOT NULL guard, got {other:?}"),
                }
                // left-hand operand is the AND-of-inequalities.
                match &*left {
                    Expr::Binary { op: BinaryOp::And, left: l2, right: r2 } => {
                        check_cmp(l2, "x", BinaryOp::NotEq, Literal::Int32(1));
                        check_cmp(r2, "x", BinaryOp::NotEq, Literal::Int32(2));
                    }
                    other => panic!("expected AND of inequalities, got {other:?}"),
                }
            }
            other => panic!("expected AND with IS NOT NULL guard, got {other:?}"),
        }
    }

    #[test]
    fn in_predicate_drops_nulls_keeps_non_null() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(
            &probe,
            &[Literal::Int32(7), Literal::Null],
            false,
        );
        // Single non-null element → bare equality, no OR fold.
        check_cmp(&got, "x", BinaryOp::Eq, Literal::Int32(7));
    }

    // ---- F-6: strict SQL 3VL for `NOT IN (subquery)` ----------------------

    /// `x NOT IN (… , NULL , …)`: with a NULL anywhere in the set, the strict
    /// SQL semantics make the predicate UNKNOWN for every row, so NO row
    /// passes. We must fold to `Bool(false)` — never build an `AND` of `<>`
    /// that would let rows through.
    #[test]
    fn not_in_with_null_in_set_excludes_all_rows() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(
            &probe,
            &[Literal::Int32(1), Literal::Int32(2), Literal::Null],
            true,
        );
        assert!(
            matches!(got, Expr::Literal(Literal::Bool(false))),
            "NOT IN with a NULL in the set must yield Bool(false) (no rows), got {got:?}"
        );
    }

    /// A set of *only* NULLs under `NOT IN` is still UNKNOWN for every row →
    /// `Bool(false)` (this is the same SQL footgun as a set containing one
    /// non-NULL plus a NULL).
    #[test]
    fn not_in_with_only_null_set_excludes_all_rows() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(&probe, &[Literal::Null], true);
        assert!(
            matches!(got, Expr::Literal(Literal::Bool(false))),
            "NOT IN over an all-NULL set must yield Bool(false), got {got:?}"
        );
    }

    /// `x NOT IN (1, 2)` with NO NULL in the set keeps the normal strict
    /// `<>`/`AND` fold over the non-NULL elements.
    #[test]
    fn not_in_without_null_builds_and_of_inequalities() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(
            &probe,
            &[Literal::Int32(1), Literal::Int32(2)],
            true,
        );
        // `(x <> 1 AND x <> 2) AND x IS NOT NULL` — the IS NOT NULL guard drops
        // NULL probe rows (SQL 3VL); the inequality fold is over the non-NULL set.
        match got {
            Expr::Binary { op: BinaryOp::And, left, right } => {
                assert!(
                    matches!(&*right, Expr::Unary { op: UnaryOp::IsNotNull, .. }),
                    "expected trailing IS NOT NULL guard, got {right:?}"
                );
                match &*left {
                    Expr::Binary { op: BinaryOp::And, left: l2, right: r2 } => {
                        check_cmp(l2, "x", BinaryOp::NotEq, Literal::Int32(1));
                        check_cmp(r2, "x", BinaryOp::NotEq, Literal::Int32(2));
                    }
                    other => panic!("expected AND of inequalities, got {other:?}"),
                }
            }
            other => panic!("expected AND with IS NOT NULL guard, got {other:?}"),
        }
    }

    /// `x IN (… , NULL , …)` (NON-negated) is unaffected by F-6: the NULL is
    /// dropped and the row matches iff it equals a non-NULL element. A NULL in
    /// the set must NOT collapse the IN form to a constant.
    #[test]
    fn in_with_null_in_set_keeps_non_null_membership() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(
            &probe,
            &[Literal::Int32(7), Literal::Null],
            false,
        );
        // Single non-null element → bare equality (the NULL is dropped).
        check_cmp(&got, "x", BinaryOp::Eq, Literal::Int32(7));
    }

    /// Probe value being NULL is orthogonal to the *set's* NULLs: the predicate
    /// structure is built over the probe expression as-is. A probe `Column`
    /// that resolves to NULL at runtime is handled by the downstream `=`/`<>`
    /// 3VL evaluation, not by `build_in_predicate`. Here we assert the builder
    /// faithfully embeds the (possibly-NULL-valued) probe expression and does
    /// not special-case it, for a NULL-free set.
    #[test]
    fn probe_expr_preserved_for_null_free_set() {
        // A probe that is itself a literal NULL — the builder must still emit
        // the equality fold; runtime 3VL (NULL = v → UNKNOWN) handles exclusion.
        let probe = Expr::Literal(Literal::Null);
        let got = build_in_predicate(&probe, &[Literal::Int32(3)], false);
        match got {
            Expr::Binary { op: BinaryOp::Eq, left, right } => {
                match (&*left, &*right) {
                    (Expr::Literal(Literal::Null), Expr::Literal(Literal::Int32(3))) => {}
                    other => panic!("expected (NULL = 3), got {other:?}"),
                }
            }
            other => panic!("expected Eq fold over probe, got {other:?}"),
        }
    }

    #[test]
    fn resolve_plan_replaces_scalar_subquery() {
        // Outer plan: Filter(Scan, x = ScalarSubquery(inner)). The executor
        // closure returns a one-row Int32 batch holding 99.
        let inner = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: crate::plan::Schema::new(vec![crate::plan::Field::new(
                "v",
                crate::plan::DataType::Int32,
                false,
            )]),
        };
        let outer = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Scan {
                table: "s".into(),
                projection: None,
                schema: crate::plan::Schema::new(vec![crate::plan::Field::new(
                    "x",
                    crate::plan::DataType::Int32,
                    false,
                )]),
            }),
            predicate: Expr::Binary {
                op: BinaryOp::Eq,
                left: Box::new(Expr::Column("x".into())),
                right: Box::new(Expr::ScalarSubquery(Box::new(inner))),
            },
        };
        let mut exec = |_p: LogicalPlan| -> BoltResult<RecordBatch> {
            let arr = Arc::new(Int32Array::from(vec![99])) as arrow_array::ArrayRef;
            Ok(single_col_batch(arr))
        };
        let resolved = resolve_plan(outer, &mut exec).unwrap();
        match resolved {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::Binary { right, .. } => match *right {
                    Expr::Literal(lit) => assert_eq!(lit, Literal::Int32(99)),
                    other => panic!("expected folded literal, got {other:?}"),
                },
                other => panic!("unexpected predicate {other:?}"),
            },
            other => panic!("unexpected plan {other:?}"),
        }
    }
}
