// SPDX-License-Identifier: Apache-2.0

//! Host-side aggregate execution for the dtypes the GPU reduction kernels
//! don't cover: `Bool` (any aggregate) and `Utf8` (COUNT/MIN/MAX only).
//!
//! Semantics, mirrored from the SQL standard:
//!
//! | op    | input | output  | notes                                         |
//! |-------|-------|---------|-----------------------------------------------|
//! | SUM   | Bool  | Int64   | count of TRUE rows (TRUE=1, FALSE=0)          |
//! | AVG   | Bool  | Float64 | fraction of TRUE rows; NULL if all-null group |
//! | MIN   | Bool  | Bool    | FALSE < TRUE; NULL if all-null group          |
//! | MAX   | Bool  | Bool    | FALSE < TRUE; NULL if all-null group          |
//! | COUNT | Bool  | Int64   | non-null row count                            |
//! | MIN   | Utf8  | Utf8    | lexicographic; NULL if all-null group         |
//! | MAX   | Utf8  | Utf8    | lexicographic; NULL if all-null group         |
//! | COUNT | Utf8  | Int64   | non-null row count                            |
//!
//! SUM/AVG over `Utf8` is not defined and returns `BoltError::Type`.
//!
//! The GPU reduction path covers fixed-width primitives end-to-end. For
//! `Bool`/`Utf8` the input cardinalities the planner expects to hit this
//! module are small enough that downloading the column and reducing on the
//! host wins over the cost of a per-dtype kernel.
//!
//! The orchestrator dispatches into this module by checking `handles(op, dt)`
//! and then calling either `execute_extended_scalar` (no GROUP BY, one output
//! row) or `execute_extended_grouped` (one output row per group).

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, BooleanArray, Int64Array, RecordBatch, StringArray};

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field};

/// True iff `(op, input_dtype)` is handled by this module.
///
/// Specifically:
///   * Any aggregate (`COUNT`/`SUM`/`AVG`/`MIN`/`MAX`) over `Bool`.
///   * `COUNT`/`MIN`/`MAX` over `Utf8`.
///
/// Returns `false` for `SUM`/`AVG` over `Utf8` — those are illegal SQL and the
/// caller is expected to surface a type error on its own dispatch path.
pub fn handles(op: &AggregateExpr, input_dtype: DataType) -> bool {
    match (op, input_dtype) {
        // All aggregates over Bool are host-side.
        (
            AggregateExpr::Sum(_)
            | AggregateExpr::Avg(_)
            | AggregateExpr::Min(_)
            | AggregateExpr::Max(_)
            | AggregateExpr::Count(_),
            DataType::Bool,
        ) => true,

        // Only COUNT / MIN / MAX make sense over Utf8.
        (
            AggregateExpr::Count(_) | AggregateExpr::Min(_) | AggregateExpr::Max(_),
            DataType::Utf8,
        ) => true,

        // SUM/AVG over Utf8 is intentionally rejected here so the caller
        // raises its own type error on the standard dispatch path.
        _ => false,
    }
}

/// Compute a single aggregate over a `Bool` or `Utf8` column on the host and
/// return a one-row Arrow array whose dtype matches `out_field.dtype`.
///
/// `expr` carries the aggregate variant; its inner expression must be a bare
/// column reference (or an alias wrapping one), matching the contract of the
/// scalar GPU reduction path.
pub fn execute_extended_scalar(
    expr: &AggregateExpr,
    out_field: &Field,
    table_batch: &RecordBatch,
) -> BoltResult<ArrayRef> {
    let (input_dtype, array) = resolve_input_array(expr, table_batch)?;
    validate_output_dtype(expr, input_dtype, out_field.dtype)?;

    match (expr, input_dtype) {
        (AggregateExpr::Sum(_), DataType::Bool) => {
            let bools = downcast_bool(array, "SUM")?;
            let count = bool_true_count(bools, 0..bools.len());
            Ok(Arc::new(Int64Array::from(vec![count])) as ArrayRef)
        }
        (AggregateExpr::Avg(_), DataType::Bool) => {
            let bools = downcast_bool(array, "AVG")?;
            let (trues, non_null) = bool_counts(bools, 0..bools.len());
            Ok(Arc::new(avg_bool_array(trues, non_null)) as ArrayRef)
        }
        (AggregateExpr::Min(_), DataType::Bool) => {
            let bools = downcast_bool(array, "MIN")?;
            Ok(Arc::new(bool_min_array(bools, 0..bools.len())) as ArrayRef)
        }
        (AggregateExpr::Max(_), DataType::Bool) => {
            let bools = downcast_bool(array, "MAX")?;
            Ok(Arc::new(bool_max_array(bools, 0..bools.len())) as ArrayRef)
        }
        (AggregateExpr::Count(_), DataType::Bool) => {
            let bools = downcast_bool(array, "COUNT")?;
            let count = non_null_count(bools, 0..bools.len());
            Ok(Arc::new(Int64Array::from(vec![count])) as ArrayRef)
        }
        (AggregateExpr::Count(_), DataType::Utf8) => {
            let strs = downcast_str(array, "COUNT")?;
            let count = non_null_count(strs, 0..strs.len());
            Ok(Arc::new(Int64Array::from(vec![count])) as ArrayRef)
        }
        (AggregateExpr::Min(_), DataType::Utf8) => {
            let strs = downcast_str(array, "MIN")?;
            Ok(Arc::new(utf8_min_array(strs, 0..strs.len())) as ArrayRef)
        }
        (AggregateExpr::Max(_), DataType::Utf8) => {
            let strs = downcast_str(array, "MAX")?;
            Ok(Arc::new(utf8_max_array(strs, 0..strs.len())) as ArrayRef)
        }
        (AggregateExpr::Sum(_) | AggregateExpr::Avg(_), DataType::Utf8) => Err(
            BoltError::Type("SUM/AVG over Utf8 is not defined".into()),
        ),
        (_, dt) => Err(BoltError::Other(format!(
            "extended_agg: unhandled (aggregate, input dtype) combination for {:?}",
            dt
        ))),
    }
}

/// Compute a single aggregate over a `Bool` or `Utf8` column per group.
///
/// `group_rows[g]` is the list of row indices belonging to group `g`. The
/// returned array has length `group_rows.len()` and its dtype matches
/// `out_field.dtype`. Empty groups produce zero/NULL per the dtype-specific
/// rules below.
pub fn execute_extended_grouped(
    expr: &AggregateExpr,
    out_field: &Field,
    table_batch: &RecordBatch,
    group_rows: &[Vec<usize>],
) -> BoltResult<ArrayRef> {
    let (input_dtype, array) = resolve_input_array(expr, table_batch)?;
    validate_output_dtype(expr, input_dtype, out_field.dtype)?;

    match (expr, input_dtype) {
        (AggregateExpr::Sum(_), DataType::Bool) => {
            let bools = downcast_bool(array, "SUM")?;
            let mut out: Vec<i64> = Vec::with_capacity(group_rows.len());
            for rows in group_rows {
                out.push(bool_true_count_indices(bools, rows));
            }
            Ok(Arc::new(Int64Array::from(out)) as ArrayRef)
        }
        (AggregateExpr::Avg(_), DataType::Bool) => {
            let bools = downcast_bool(array, "AVG")?;
            let mut out: Vec<Option<f64>> = Vec::with_capacity(group_rows.len());
            for rows in group_rows {
                let (trues, non_null) = bool_counts_indices(bools, rows);
                out.push(avg_bool_value(trues, non_null));
            }
            Ok(Arc::new(arrow_array::Float64Array::from(out)) as ArrayRef)
        }
        (AggregateExpr::Min(_), DataType::Bool) => {
            let bools = downcast_bool(array, "MIN")?;
            let mut out: Vec<Option<bool>> = Vec::with_capacity(group_rows.len());
            for rows in group_rows {
                out.push(bool_min_indices(bools, rows));
            }
            Ok(Arc::new(BooleanArray::from(out)) as ArrayRef)
        }
        (AggregateExpr::Max(_), DataType::Bool) => {
            let bools = downcast_bool(array, "MAX")?;
            let mut out: Vec<Option<bool>> = Vec::with_capacity(group_rows.len());
            for rows in group_rows {
                out.push(bool_max_indices(bools, rows));
            }
            Ok(Arc::new(BooleanArray::from(out)) as ArrayRef)
        }
        (AggregateExpr::Count(_), DataType::Bool) => {
            let bools = downcast_bool(array, "COUNT")?;
            let mut out: Vec<i64> = Vec::with_capacity(group_rows.len());
            for rows in group_rows {
                out.push(non_null_count_indices(bools, rows));
            }
            Ok(Arc::new(Int64Array::from(out)) as ArrayRef)
        }
        (AggregateExpr::Count(_), DataType::Utf8) => {
            let strs = downcast_str(array, "COUNT")?;
            let mut out: Vec<i64> = Vec::with_capacity(group_rows.len());
            for rows in group_rows {
                out.push(non_null_count_indices(strs, rows));
            }
            Ok(Arc::new(Int64Array::from(out)) as ArrayRef)
        }
        (AggregateExpr::Min(_), DataType::Utf8) => {
            let strs = downcast_str(array, "MIN")?;
            let mut out: Vec<Option<String>> = Vec::with_capacity(group_rows.len());
            for rows in group_rows {
                out.push(utf8_min_indices(strs, rows));
            }
            Ok(Arc::new(StringArray::from(out)) as ArrayRef)
        }
        (AggregateExpr::Max(_), DataType::Utf8) => {
            let strs = downcast_str(array, "MAX")?;
            let mut out: Vec<Option<String>> = Vec::with_capacity(group_rows.len());
            for rows in group_rows {
                out.push(utf8_max_indices(strs, rows));
            }
            Ok(Arc::new(StringArray::from(out)) as ArrayRef)
        }
        (AggregateExpr::Sum(_) | AggregateExpr::Avg(_), DataType::Utf8) => Err(
            BoltError::Type("SUM/AVG over Utf8 is not defined".into()),
        ),
        (_, dt) => Err(BoltError::Other(format!(
            "extended_agg: unhandled (aggregate, input dtype) combination for {:?}",
            dt
        ))),
    }
}

// ---------------------------------------------------------------------------
// Input resolution / validation.
// ---------------------------------------------------------------------------

/// Unwrap a single layer of `Alias(...)` and return the underlying expression.
fn unwrap_alias(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias(inner, _) => unwrap_alias(inner),
        other => other,
    }
}

/// Extract the column name from a bare-column-ref expression, transparently
/// unwrapping any aliases. Other expression shapes are rejected with `Other`.
fn bare_column_name(expr: &Expr) -> BoltResult<&str> {
    match unwrap_alias(expr) {
        Expr::Column(name) => Ok(name.as_str()),
        _ => Err(BoltError::Other(
            "extended_agg: aggregate input must be a bare column reference".into(),
        )),
    }
}

/// Inner expression of an `AggregateExpr`.
fn aggregate_inner(expr: &AggregateExpr) -> &Expr {
    match expr {
        AggregateExpr::Count(e)
        | AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e) => e,
    }
}

/// Resolve the aggregate's input column inside `table_batch`, returning its
/// plan dtype and the borrowed Arrow array.
fn resolve_input_array<'a>(
    expr: &AggregateExpr,
    table_batch: &'a RecordBatch,
) -> BoltResult<(DataType, &'a dyn Array)> {
    let col_name = bare_column_name(aggregate_inner(expr))?;
    let idx = table_batch
        .schema()
        .index_of(col_name)
        .map_err(|e| {
            BoltError::Plan(format!(
                "extended_agg input column '{}' not present in batch: {}",
                col_name, e
            ))
        })?;
    let array = table_batch.column(idx).as_ref();
    let dtype = match array.data_type() {
        arrow_schema::DataType::Boolean => DataType::Bool,
        arrow_schema::DataType::Utf8 => DataType::Utf8,
        other => {
            return Err(BoltError::Type(format!(
                "extended_agg: column '{}' has unsupported Arrow dtype {:?}",
                col_name, other
            )))
        }
    };
    Ok((dtype, array))
}

/// Check that the declared `out_dtype` matches the (op, input_dtype) table.
fn validate_output_dtype(
    expr: &AggregateExpr,
    input_dtype: DataType,
    out_dtype: DataType,
) -> BoltResult<()> {
    let expected = match (expr, input_dtype) {
        (AggregateExpr::Sum(_), DataType::Bool) => DataType::Int64,
        (AggregateExpr::Avg(_), DataType::Bool) => DataType::Float64,
        (AggregateExpr::Min(_), DataType::Bool) => DataType::Bool,
        (AggregateExpr::Max(_), DataType::Bool) => DataType::Bool,
        (AggregateExpr::Count(_), DataType::Bool) => DataType::Int64,
        (AggregateExpr::Min(_), DataType::Utf8) => DataType::Utf8,
        (AggregateExpr::Max(_), DataType::Utf8) => DataType::Utf8,
        (AggregateExpr::Count(_), DataType::Utf8) => DataType::Int64,
        (AggregateExpr::Sum(_) | AggregateExpr::Avg(_), DataType::Utf8) => {
            return Err(BoltError::Type(
                "SUM/AVG over Utf8 is not defined".into(),
            ))
        }
        (_, dt) => {
            return Err(BoltError::Other(format!(
                "extended_agg: unsupported (aggregate, input dtype) combo with input {:?}",
                dt
            )))
        }
    };
    if expected != out_dtype {
        return Err(BoltError::Other(format!(
            "extended_agg: output dtype mismatch: plan says {:?}, expected {:?}",
            out_dtype, expected
        )));
    }
    Ok(())
}

fn downcast_bool<'a>(array: &'a dyn Array, op_name: &str) -> BoltResult<&'a BooleanArray> {
    array.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
        BoltError::Type(format!(
            "extended_agg: {} expected BooleanArray, got {:?}",
            op_name,
            array.data_type()
        ))
    })
}

fn downcast_str<'a>(array: &'a dyn Array, op_name: &str) -> BoltResult<&'a StringArray> {
    array.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
        BoltError::Type(format!(
            "extended_agg: {} expected StringArray, got {:?}",
            op_name,
            array.data_type()
        ))
    })
}

// ---------------------------------------------------------------------------
// Per-aggregate reductions (range and indexed variants).
// ---------------------------------------------------------------------------

/// Count of non-null rows over a contiguous range.
fn non_null_count(array: &dyn Array, range: std::ops::Range<usize>) -> i64 {
    let mut c: i64 = 0;
    for i in range {
        if !array.is_null(i) {
            c += 1;
        }
    }
    c
}

/// Count of non-null rows over an arbitrary index list.
fn non_null_count_indices(array: &dyn Array, rows: &[usize]) -> i64 {
    let mut c: i64 = 0;
    for &i in rows {
        if !array.is_null(i) {
            c += 1;
        }
    }
    c
}

/// Count of TRUE rows (skipping nulls) over a contiguous range.
fn bool_true_count(array: &BooleanArray, range: std::ops::Range<usize>) -> i64 {
    let mut c: i64 = 0;
    for i in range {
        if !array.is_null(i) && array.value(i) {
            c += 1;
        }
    }
    c
}

/// Count of TRUE rows (skipping nulls) over an arbitrary index list.
fn bool_true_count_indices(array: &BooleanArray, rows: &[usize]) -> i64 {
    let mut c: i64 = 0;
    for &i in rows {
        if !array.is_null(i) && array.value(i) {
            c += 1;
        }
    }
    c
}

/// `(true_count, non_null_count)` over a contiguous range.
fn bool_counts(array: &BooleanArray, range: std::ops::Range<usize>) -> (i64, i64) {
    let mut trues: i64 = 0;
    let mut non_null: i64 = 0;
    for i in range {
        if !array.is_null(i) {
            non_null += 1;
            if array.value(i) {
                trues += 1;
            }
        }
    }
    (trues, non_null)
}

/// `(true_count, non_null_count)` over an arbitrary index list.
fn bool_counts_indices(array: &BooleanArray, rows: &[usize]) -> (i64, i64) {
    let mut trues: i64 = 0;
    let mut non_null: i64 = 0;
    for &i in rows {
        if !array.is_null(i) {
            non_null += 1;
            if array.value(i) {
                trues += 1;
            }
        }
    }
    (trues, non_null)
}

/// AVG of a Bool group as a nullable f64: `trues / non_null`, or `None` if
/// the group has no non-null rows (standard SQL).
fn avg_bool_value(trues: i64, non_null: i64) -> Option<f64> {
    if non_null == 0 {
        None
    } else {
        Some(trues as f64 / non_null as f64)
    }
}

/// Scalar one-row `Float64Array` for the AVG(bool) result.
fn avg_bool_array(trues: i64, non_null: i64) -> arrow_array::Float64Array {
    arrow_array::Float64Array::from(vec![avg_bool_value(trues, non_null)])
}

/// MIN(bool) over a contiguous range: FALSE if any FALSE is seen, otherwise
/// TRUE if any TRUE is seen, otherwise NULL.
fn bool_min(array: &BooleanArray, range: std::ops::Range<usize>) -> Option<bool> {
    let mut any: bool = false;
    let mut any_false: bool = false;
    for i in range {
        if array.is_null(i) {
            continue;
        }
        any = true;
        if !array.value(i) {
            any_false = true;
        }
    }
    if !any {
        None
    } else {
        Some(!any_false)
    }
}

/// MIN(bool) over an arbitrary index list.
fn bool_min_indices(array: &BooleanArray, rows: &[usize]) -> Option<bool> {
    let mut any: bool = false;
    let mut any_false: bool = false;
    for &i in rows {
        if array.is_null(i) {
            continue;
        }
        any = true;
        if !array.value(i) {
            any_false = true;
        }
    }
    if !any {
        None
    } else {
        Some(!any_false)
    }
}

/// Scalar one-row `BooleanArray` for the MIN(bool) result.
fn bool_min_array(array: &BooleanArray, range: std::ops::Range<usize>) -> BooleanArray {
    BooleanArray::from(vec![bool_min(array, range)])
}

/// MAX(bool) over a contiguous range: TRUE if any TRUE is seen, otherwise
/// FALSE if any FALSE is seen, otherwise NULL.
fn bool_max(array: &BooleanArray, range: std::ops::Range<usize>) -> Option<bool> {
    let mut any: bool = false;
    let mut any_true: bool = false;
    for i in range {
        if array.is_null(i) {
            continue;
        }
        any = true;
        if array.value(i) {
            any_true = true;
        }
    }
    if !any {
        None
    } else {
        Some(any_true)
    }
}

/// MAX(bool) over an arbitrary index list.
fn bool_max_indices(array: &BooleanArray, rows: &[usize]) -> Option<bool> {
    let mut any: bool = false;
    let mut any_true: bool = false;
    for &i in rows {
        if array.is_null(i) {
            continue;
        }
        any = true;
        if array.value(i) {
            any_true = true;
        }
    }
    if !any {
        None
    } else {
        Some(any_true)
    }
}

/// Scalar one-row `BooleanArray` for the MAX(bool) result.
fn bool_max_array(array: &BooleanArray, range: std::ops::Range<usize>) -> BooleanArray {
    BooleanArray::from(vec![bool_max(array, range)])
}

/// MIN(utf8) over a contiguous range — lexicographic min of non-null strings,
/// or `None` if the group is all-null.
fn utf8_min(array: &StringArray, range: std::ops::Range<usize>) -> Option<String> {
    let mut best: Option<&str> = None;
    for i in range {
        if array.is_null(i) {
            continue;
        }
        let s = array.value(i);
        best = Some(match best {
            None => s,
            Some(b) if s < b => s,
            Some(b) => b,
        });
    }
    best.map(|s| s.to_string())
}

/// MIN(utf8) over an arbitrary index list.
fn utf8_min_indices(array: &StringArray, rows: &[usize]) -> Option<String> {
    let mut best: Option<&str> = None;
    for &i in rows {
        if array.is_null(i) {
            continue;
        }
        let s = array.value(i);
        best = Some(match best {
            None => s,
            Some(b) if s < b => s,
            Some(b) => b,
        });
    }
    best.map(|s| s.to_string())
}

/// Scalar one-row `StringArray` for the MIN(utf8) result.
fn utf8_min_array(array: &StringArray, range: std::ops::Range<usize>) -> StringArray {
    StringArray::from(vec![utf8_min(array, range)])
}

/// MAX(utf8) over a contiguous range — lexicographic max of non-null strings.
fn utf8_max(array: &StringArray, range: std::ops::Range<usize>) -> Option<String> {
    let mut best: Option<&str> = None;
    for i in range {
        if array.is_null(i) {
            continue;
        }
        let s = array.value(i);
        best = Some(match best {
            None => s,
            Some(b) if s > b => s,
            Some(b) => b,
        });
    }
    best.map(|s| s.to_string())
}

/// MAX(utf8) over an arbitrary index list.
fn utf8_max_indices(array: &StringArray, rows: &[usize]) -> Option<String> {
    let mut best: Option<&str> = None;
    for &i in rows {
        if array.is_null(i) {
            continue;
        }
        let s = array.value(i);
        best = Some(match best {
            None => s,
            Some(b) if s > b => s,
            Some(b) => b,
        });
    }
    best.map(|s| s.to_string())
}

/// Scalar one-row `StringArray` for the MAX(utf8) result.
fn utf8_max_array(array: &StringArray, range: std::ops::Range<usize>) -> StringArray {
    StringArray::from(vec![utf8_max(array, range)])
}

// ---------------------------------------------------------------------------
// Tests — pure host code, no CUDA required.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::Float64Array;
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    /// Build a one-column `RecordBatch` from a typed Arrow array.
    fn one_col_batch(name: &str, arr: ArrayRef) -> RecordBatch {
        let dt = arr.data_type().clone();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(name, dt, true)]));
        RecordBatch::try_new(schema, vec![arr]).expect("one-col batch")
    }

    fn bool_col(name: &str, vals: Vec<Option<bool>>) -> RecordBatch {
        one_col_batch(name, Arc::new(BooleanArray::from(vals)) as ArrayRef)
    }

    fn utf8_col(name: &str, vals: Vec<Option<&str>>) -> RecordBatch {
        one_col_batch(name, Arc::new(StringArray::from(vals)) as ArrayRef)
    }

    fn col_expr(name: &str) -> Expr {
        Expr::Column(name.to_string())
    }

    fn out_field(name: &str, dt: DataType) -> Field {
        Field::new(name, dt, true)
    }

    /// 1. SUM(bool) counts TRUE values, ignoring NULLs.
    #[test]
    fn sum_bool_counts_trues() {
        let batch = bool_col(
            "b",
            vec![Some(true), Some(false), Some(true), Some(true), None],
        );
        let agg = AggregateExpr::Sum(col_expr("b"));
        let field = out_field("sum_b", DataType::Int64);
        let arr = execute_extended_scalar(&agg, &field, &batch).expect("sum ok");
        let arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr.value(0), 3);
    }

    /// 2. AVG(bool) divides TRUE count by non-null count (NULLs excluded
    ///    from the denominator).
    #[test]
    fn avg_bool_excludes_nulls() {
        let batch = bool_col(
            "b",
            vec![Some(true), Some(false), Some(true), Some(true), None],
        );
        let agg = AggregateExpr::Avg(col_expr("b"));
        let field = out_field("avg_b", DataType::Float64);
        let arr = execute_extended_scalar(&agg, &field, &batch).expect("avg ok");
        let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(arr.len(), 1);
        assert!(!arr.is_null(0));
        let got = arr.value(0);
        let expected = 3.0_f64 / 4.0_f64;
        assert!(
            (got - expected).abs() < 1e-12,
            "got {got}, expected {expected}"
        );
    }

    /// 3. MIN(bool) is FALSE if any FALSE present; MAX(bool) is TRUE if any
    ///    TRUE present.
    #[test]
    fn min_max_bool_basic() {
        let batch = bool_col("b", vec![Some(false), Some(true), Some(false)]);

        let min_arr = execute_extended_scalar(
            &AggregateExpr::Min(col_expr("b")),
            &out_field("min_b", DataType::Bool),
            &batch,
        )
        .expect("min ok");
        let min_arr = min_arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(min_arr.len(), 1);
        assert!(!min_arr.is_null(0));
        assert_eq!(min_arr.value(0), false);

        let max_arr = execute_extended_scalar(
            &AggregateExpr::Max(col_expr("b")),
            &out_field("max_b", DataType::Bool),
            &batch,
        )
        .expect("max ok");
        let max_arr = max_arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(max_arr.len(), 1);
        assert!(!max_arr.is_null(0));
        assert_eq!(max_arr.value(0), true);
    }

    /// 4. All-null input → MIN and MAX both return a single NULL.
    #[test]
    fn min_max_bool_all_null() {
        let batch = bool_col("b", vec![None, None, None]);

        let min_arr = execute_extended_scalar(
            &AggregateExpr::Min(col_expr("b")),
            &out_field("min_b", DataType::Bool),
            &batch,
        )
        .expect("min ok");
        let min_arr = min_arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(min_arr.len(), 1);
        assert!(min_arr.is_null(0));

        let max_arr = execute_extended_scalar(
            &AggregateExpr::Max(col_expr("b")),
            &out_field("max_b", DataType::Bool),
            &batch,
        )
        .expect("max ok");
        let max_arr = max_arr.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(max_arr.len(), 1);
        assert!(max_arr.is_null(0));
    }

    /// 5. COUNT(utf8) returns the non-null row count.
    #[test]
    fn count_utf8_excludes_nulls() {
        let batch = utf8_col("s", vec![Some("a"), Some("b"), None, Some("c")]);
        let arr = execute_extended_scalar(
            &AggregateExpr::Count(col_expr("s")),
            &out_field("count_s", DataType::Int64),
            &batch,
        )
        .expect("count ok");
        let arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr.value(0), 3);
    }

    /// 6. MIN/MAX over Utf8 are lexicographic.
    #[test]
    fn min_max_utf8_lex() {
        let batch = utf8_col("s", vec![Some("banana"), Some("apple"), Some("cherry")]);

        let min_arr = execute_extended_scalar(
            &AggregateExpr::Min(col_expr("s")),
            &out_field("min_s", DataType::Utf8),
            &batch,
        )
        .expect("min ok");
        let min_arr = min_arr.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(min_arr.len(), 1);
        assert!(!min_arr.is_null(0));
        assert_eq!(min_arr.value(0), "apple");

        let max_arr = execute_extended_scalar(
            &AggregateExpr::Max(col_expr("s")),
            &out_field("max_s", DataType::Utf8),
            &batch,
        )
        .expect("max ok");
        let max_arr = max_arr.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(max_arr.len(), 1);
        assert!(!max_arr.is_null(0));
        assert_eq!(max_arr.value(0), "cherry");
    }

    /// 7. SUM over Utf8 returns a `BoltError::Type`.
    #[test]
    fn sum_utf8_errors() {
        let batch = utf8_col("s", vec![Some("a"), Some("b")]);
        // The output dtype is irrelevant — validation errors out before any
        // use of `out_field`. Pass Utf8 (the column dtype) so the dispatch
        // reaches the SUM/AVG arm rather than tripping the output-dtype
        // mismatch first.
        let err = execute_extended_scalar(
            &AggregateExpr::Sum(col_expr("s")),
            &out_field("sum_s", DataType::Utf8),
            &batch,
        )
        .expect_err("should error");
        match err {
            BoltError::Type(msg) => {
                assert!(
                    msg.contains("SUM/AVG over Utf8"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected BoltError::Type, got {other:?}"),
        }
    }

    /// 8. Grouped SUM(bool): two groups [T,F] and [T,T,F] → counts [1, 2].
    #[test]
    fn grouped_sum_bool() {
        let batch = bool_col(
            "b",
            vec![
                Some(true),  // row 0 → group 0
                Some(false), // row 1 → group 0
                Some(true),  // row 2 → group 1
                Some(true),  // row 3 → group 1
                Some(false), // row 4 → group 1
            ],
        );
        let group_rows: Vec<Vec<usize>> = vec![vec![0, 1], vec![2, 3, 4]];
        let arr = execute_extended_grouped(
            &AggregateExpr::Sum(col_expr("b")),
            &out_field("sum_b", DataType::Int64),
            &batch,
            &group_rows,
        )
        .expect("grouped sum ok");
        let arr = arr.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr.value(0), 1);
        assert_eq!(arr.value(1), 2);
    }

    /// `handles` accepts the (op, dtype) pairs documented in the module
    /// header and rejects the rest.
    #[test]
    fn handles_matrix() {
        let c = AggregateExpr::Count(col_expr("x"));
        let s = AggregateExpr::Sum(col_expr("x"));
        let a = AggregateExpr::Avg(col_expr("x"));
        let mi = AggregateExpr::Min(col_expr("x"));
        let ma = AggregateExpr::Max(col_expr("x"));

        // Bool: all five aggregates are handled.
        assert!(handles(&c, DataType::Bool));
        assert!(handles(&s, DataType::Bool));
        assert!(handles(&a, DataType::Bool));
        assert!(handles(&mi, DataType::Bool));
        assert!(handles(&ma, DataType::Bool));

        // Utf8: only COUNT/MIN/MAX.
        assert!(handles(&c, DataType::Utf8));
        assert!(handles(&mi, DataType::Utf8));
        assert!(handles(&ma, DataType::Utf8));
        assert!(!handles(&s, DataType::Utf8));
        assert!(!handles(&a, DataType::Utf8));

        // Numeric inputs are never owned by this module.
        for dt in [
            DataType::Int32,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
        ] {
            for op in [&c, &s, &a, &mi, &ma] {
                assert!(!handles(op, dt), "unexpected handles({op:?}, {dt:?})");
            }
        }
    }

    /// AVG of an all-null group yields a NULL (standard SQL).
    #[test]
    fn grouped_avg_all_null_is_null() {
        let batch = bool_col("b", vec![None, None, Some(true)]);
        // Group 0 picks the two NULL rows; group 1 picks the lone TRUE row.
        let group_rows: Vec<Vec<usize>> = vec![vec![0, 1], vec![2]];
        let arr = execute_extended_grouped(
            &AggregateExpr::Avg(col_expr("b")),
            &out_field("avg_b", DataType::Float64),
            &batch,
            &group_rows,
        )
        .expect("grouped avg ok");
        let arr = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(arr.len(), 2);
        assert!(arr.is_null(0));
        assert!(!arr.is_null(1));
        assert!((arr.value(1) - 1.0).abs() < 1e-12);
    }
}
