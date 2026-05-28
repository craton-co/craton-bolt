// SPDX-License-Identifier: Apache-2.0

//! Host-side post-aggregate filter executor.
//!
//! Used by `PhysicalPlan::Filter`, which the lowerer emits when a
//! `LogicalPlan::Filter` sits above an operator that can't be folded into
//! the scan-kernel chain — most importantly `HAVING`, which the SQL
//! frontend produces as `Filter { Project { Aggregate { .. } } }`.
//!
//! Strategy: the inner plan has already been executed and its output
//! materialised as a host-side `RecordBatch`. Lift each column into an
//! [`expr_agg::HostColumn`], evaluate the predicate via
//! [`expr_agg::eval_expr`] to produce a `Bool` column, then apply
//! `arrow::compute::filter` to every column. Group-by outputs are tiny
//! (one row per group), so a host-side pass is the right cost trade-off
//! for 0.3 — pushing HAVING down to GPU kernels would buy nothing here.

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray,
};
use arrow_schema::DataType as ArrowDataType;

use crate::error::{BoltError, BoltResult};
use crate::exec::expr_agg::{self, ColumnEnv, HostColumn};
use crate::exec::QueryHandle;
use crate::plan::logical_plan::{DataType, Expr};

/// Apply a host-side boolean filter to the input handle.
///
/// `predicate` is evaluated against the input batch's schema; rows for
/// which it produces `Some(true)` are kept, `Some(false)` and `None` (SQL
/// NULL) are dropped — the standard SQL-WHERE semantic where NULL acts as
/// "not true". The output schema is identical to the input's.
pub fn execute_filter(input: QueryHandle, predicate: &Expr) -> BoltResult<QueryHandle> {
    let batch = input.into_record_batch();
    let n_rows = batch.num_rows();

    if n_rows == 0 {
        // No rows means nothing to filter; trivially rewrap to avoid
        // touching the evaluator (which would still succeed but at zero
        // marginal value).
        return Ok(QueryHandle::from_record_batch(batch));
    }

    // Lift each batch column into a HostColumn. We keep both the owned
    // HostColumns and the &-references the evaluator's ColumnEnv expects.
    let schema = batch.schema();
    let mut owned: Vec<(String, HostColumn)> = Vec::with_capacity(batch.num_columns());
    for (i, field) in schema.fields().iter().enumerate() {
        let arr = batch.column(i);
        let hc = arrow_array_to_host_column(arr.as_ref(), n_rows)?;
        owned.push((field.name().clone(), hc));
    }
    let env: ColumnEnv<'_> = owned.iter().map(|(n, c)| (n.clone(), c)).collect();

    // Evaluate the predicate, coercing to Bool. Then build a BooleanArray
    // mask — NULL predicate result drops the row (SQL "WHERE NULL" semantics).
    let bool_col = expr_agg::eval_expr(predicate, &env, DataType::Bool, n_rows)?;
    let HostColumn::Bool(mask_opts) = bool_col else {
        return Err(BoltError::Other(format!(
            "PhysicalPlan::Filter: predicate did not evaluate to Bool, got {:?}",
            bool_col.dtype()
        )));
    };
    let mask_bools: Vec<bool> = mask_opts.into_iter().map(|b| b.unwrap_or(false)).collect();
    let mask = BooleanArray::from(mask_bools);

    let filtered: Vec<ArrayRef> = batch
        .columns()
        .iter()
        .map(|c| {
            arrow::compute::filter(c.as_ref(), &mask).map_err(|e| {
                BoltError::Other(format!("arrow::compute::filter failed in PhysicalPlan::Filter: {e}"))
            })
        })
        .collect::<BoltResult<Vec<_>>>()?;
    let out = RecordBatch::try_new(batch.schema(), filtered).map_err(|e| {
        BoltError::Other(format!("failed to rebuild RecordBatch after Filter: {e}"))
    })?;
    Ok(QueryHandle::from_record_batch(out))
}

/// Lift an Arrow array into a `HostColumn`. Only the primitive dtypes the
/// engine actually surfaces above an Aggregate (Int32/Int64/Float32/Float64/
/// Bool/Utf8) are supported — same set as `engine.rs`'s `arrow_dtype_to_plan`.
fn arrow_array_to_host_column(arr: &dyn Array, n_rows: usize) -> BoltResult<HostColumn> {
    match arr.data_type() {
        ArrowDataType::Int32 => {
            let a = arr.as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
                BoltError::Type("PhysicalPlan::Filter: expected Int32 array".into())
            })?;
            let v: Vec<Option<i32>> = (0..n_rows)
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect();
            Ok(HostColumn::I32(v))
        }
        ArrowDataType::Int64 => {
            let a = arr.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                BoltError::Type("PhysicalPlan::Filter: expected Int64 array".into())
            })?;
            let v: Vec<Option<i64>> = (0..n_rows)
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect();
            Ok(HostColumn::I64(v))
        }
        ArrowDataType::Float32 => {
            let a = arr.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                BoltError::Type("PhysicalPlan::Filter: expected Float32 array".into())
            })?;
            let v: Vec<Option<f32>> = (0..n_rows)
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect();
            Ok(HostColumn::F32(v))
        }
        ArrowDataType::Float64 => {
            let a = arr.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                BoltError::Type("PhysicalPlan::Filter: expected Float64 array".into())
            })?;
            let v: Vec<Option<f64>> = (0..n_rows)
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect();
            Ok(HostColumn::F64(v))
        }
        ArrowDataType::Boolean => {
            let a = arr.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
                BoltError::Type("PhysicalPlan::Filter: expected Boolean array".into())
            })?;
            let v: Vec<Option<bool>> = (0..n_rows)
                .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                .collect();
            Ok(HostColumn::Bool(v))
        }
        ArrowDataType::Utf8 => {
            let a = arr.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                BoltError::Type("PhysicalPlan::Filter: expected Utf8 array".into())
            })?;
            let v: Vec<Option<String>> = (0..n_rows)
                .map(|i| {
                    if a.is_null(i) {
                        None
                    } else {
                        Some(a.value(i).to_string())
                    }
                })
                .collect();
            Ok(HostColumn::Utf8(v))
        }
        other => Err(BoltError::Type(format!(
            "PhysicalPlan::Filter: unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType as ArrowDataType, Field, Schema};
    use std::sync::Arc;

    use crate::plan::logical_plan::{BinaryOp, Expr, Literal, UnaryOp};

    fn two_col_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("sum_v", ArrowDataType::Int64, false),
        ]));
        let k = Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as Arc<dyn Array>;
        let s = Arc::new(Int64Array::from(vec![5_i64, 10, 20, 7])) as Arc<dyn Array>;
        RecordBatch::try_new(schema, vec![k, s]).unwrap()
    }

    #[test]
    fn filter_keeps_rows_where_predicate_is_true() {
        // HAVING sum_v > 8 — survivors are rows with sum_v ∈ {10, 20}, i.e. k ∈ {2, 3}.
        let batch = two_col_batch();
        let predicate = Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(Expr::Column("sum_v".into())),
            right: Box::new(Expr::Literal(Literal::Int64(8))),
        };
        let out =
            execute_filter(QueryHandle::from_record_batch(batch), &predicate).expect("ok");
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 2);
        let k_arr = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32");
        let s_arr = result
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64");
        let ks: Vec<i32> = (0..k_arr.len()).map(|i| k_arr.value(i)).collect();
        let ss: Vec<i64> = (0..s_arr.len()).map(|i| s_arr.value(i)).collect();
        assert_eq!(ks, vec![2, 3]);
        assert_eq!(ss, vec![10, 20]);
    }

    #[test]
    fn filter_empty_input_returns_empty() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "k",
            ArrowDataType::Int32,
            false,
        )]));
        let k = Arc::new(Int32Array::from(Vec::<i32>::new())) as Arc<dyn Array>;
        let batch = RecordBatch::try_new(schema, vec![k]).unwrap();
        let predicate = Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(Expr::Column("k".into())),
            right: Box::new(Expr::Literal(Literal::Int32(0))),
        };
        let out =
            execute_filter(QueryHandle::from_record_batch(batch), &predicate).expect("ok");
        assert_eq!(out.num_rows(), 0);
    }

    #[test]
    fn filter_all_rows_dropped_returns_empty() {
        let batch = two_col_batch();
        let predicate = Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(Expr::Column("sum_v".into())),
            right: Box::new(Expr::Literal(Literal::Int64(1_000_000))),
        };
        let out =
            execute_filter(QueryHandle::from_record_batch(batch), &predicate).expect("ok");
        assert_eq!(out.into_record_batch().num_rows(), 0);
    }

    /// Build a tiny batch with mixed nullable rows, then apply
    /// `WHERE x IS NULL` via the host filter executor. The lowering layer
    /// is exercised separately in `physical_plan.rs`; this test pins the
    /// runtime behaviour of `execute_filter` over an `Expr::Unary` predicate.
    fn nullable_int_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", ArrowDataType::Int32, false),
            Field::new("x", ArrowDataType::Int32, true),
        ]));
        // id=1,x=10 ; id=2,x=NULL ; id=3,x=30 ; id=4,x=NULL ; id=5,x=50
        let id = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as Arc<dyn Array>;
        let x = Arc::new(Int32Array::from(vec![
            Some(10),
            None,
            Some(30),
            None,
            Some(50),
        ])) as Arc<dyn Array>;
        RecordBatch::try_new(schema, vec![id, x]).unwrap()
    }

    #[test]
    fn filter_is_null_keeps_only_null_rows() {
        let batch = nullable_int_batch();
        let predicate = Expr::Unary {
            op: UnaryOp::IsNull,
            operand: Box::new(Expr::Column("x".into())),
        };
        let out = execute_filter(QueryHandle::from_record_batch(batch), &predicate)
            .expect("filter ok");
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 2, "two rows have x IS NULL (id=2, id=4)");
        let id_arr = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32 id col");
        let ids: Vec<i32> = (0..id_arr.len()).map(|i| id_arr.value(i)).collect();
        assert_eq!(ids, vec![2, 4]);
        // And the x column for those rows should be NULL.
        let x_arr = result
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32 x col");
        assert!(x_arr.is_null(0));
        assert!(x_arr.is_null(1));
    }

    #[test]
    fn filter_is_not_null_drops_only_null_rows() {
        let batch = nullable_int_batch();
        let predicate = Expr::Unary {
            op: UnaryOp::IsNotNull,
            operand: Box::new(Expr::Column("x".into())),
        };
        let out = execute_filter(QueryHandle::from_record_batch(batch), &predicate)
            .expect("filter ok");
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 3);
        let id_arr = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32 id col");
        let ids: Vec<i32> = (0..id_arr.len()).map(|i| id_arr.value(i)).collect();
        assert_eq!(ids, vec![1, 3, 5]);
    }
}
