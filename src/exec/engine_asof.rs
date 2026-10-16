// SPDX-License-Identifier: Apache-2.0
//! `ASOF JOIN` execution (see [`AsOfJoinPlan`]).
//!
//! The right relation's rows are grouped by their equality keys and sorted
//! by time; each left row binary-searches its group for the closest row that
//! satisfies the match condition. `O((L + R) log R)`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::compute::{cast, take};
use arrow::row::{RowConverter, SortField};
use arrow_array::{Array, ArrayRef, RecordBatch, UInt32Array};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::error::{BoltError, BoltResult};
use crate::exec::engine::{Engine, QueryHandle};
use crate::exec::schema_convert::plan_schema_to_arrow_schema;
use crate::exec::streaming::TableSource;
use crate::plan::sql_frontend::{AsOfJoinPlan, AsOfOp, LATERAL_APPLY_RESULT_TABLE};

fn err(e: impl std::fmt::Display) -> BoltError {
    BoltError::Other(format!("ASOF JOIN: {e}"))
}

/// A time column as totally ordered values (`None` = NULL).
enum Times {
    Int(Vec<Option<i64>>),
    Float(Vec<Option<f64>>),
}

impl Times {
    fn of(a: &ArrayRef, float: bool) -> BoltResult<Times> {
        if float {
            let c = cast(a, &ArrowDataType::Float64).map_err(err)?;
            let c = c
                .as_any()
                .downcast_ref::<arrow_array::Float64Array>()
                .expect("cast to Float64");
            return Ok(Times::Float(
                (0..c.len())
                    .map(|i| (!c.is_null(i)).then(|| c.value(i)))
                    .collect(),
            ));
        }
        // Dates / timestamps go through their integer representation.
        let ints = match a.data_type() {
            ArrowDataType::Date32 => cast(
                &cast(a, &ArrowDataType::Int32).map_err(err)?,
                &ArrowDataType::Int64,
            ),
            _ => cast(a, &ArrowDataType::Int64),
        }
        .map_err(err)?;
        let c = ints
            .as_any()
            .downcast_ref::<arrow_array::Int64Array>()
            .expect("cast to Int64");
        Ok(Times::Int(
            (0..c.len())
                .map(|i| (!c.is_null(i)).then(|| c.value(i)))
                .collect(),
        ))
    }
}

/// Index of the matching row in `sorted` (ascending by time) for time `t`.
fn search<T: PartialOrd + Copy>(sorted: &[(T, u32)], t: T, op: AsOfOp) -> Option<u32> {
    match op {
        AsOfOp::Ge => {
            let i = sorted.partition_point(|(x, _)| *x <= t);
            (i > 0).then(|| sorted[i - 1].1)
        }
        AsOfOp::Gt => {
            let i = sorted.partition_point(|(x, _)| *x < t);
            (i > 0).then(|| sorted[i - 1].1)
        }
        AsOfOp::Le => {
            let i = sorted.partition_point(|(x, _)| *x < t);
            sorted.get(i).map(|(_, r)| *r)
        }
        AsOfOp::Lt => {
            let i = sorted.partition_point(|(x, _)| *x <= t);
            sorted.get(i).map(|(_, r)| *r)
        }
    }
}

/// The right row each left row matches.
fn matches(
    p: &AsOfJoinPlan,
    left: &RecordBatch,
    right: &RecordBatch,
) -> BoltResult<Vec<Option<u32>>> {
    // Keys of both sides in the left key types, as comparable rows.
    let lk: Vec<ArrayRef> = p
        .keys
        .iter()
        .map(|(l, _)| Arc::clone(left.column(*l)))
        .collect();
    let rk: Vec<ArrayRef> = p
        .keys
        .iter()
        .map(|(l, r)| cast(right.column(*r), left.column(*l).data_type()).map_err(err))
        .collect::<BoltResult<_>>()?;
    let any_null = |cols: &[ArrayRef], i: usize| cols.iter().any(|c| c.is_null(i));
    let key_rows = |cols: &[ArrayRef]| -> BoltResult<Vec<Vec<u8>>> {
        if cols.is_empty() {
            return Ok(Vec::new());
        }
        let conv = RowConverter::new(
            cols.iter()
                .map(|c| SortField::new(c.data_type().clone()))
                .collect(),
        )
        .map_err(err)?;
        let rows = conv.convert_columns(cols).map_err(err)?;
        Ok((0..rows.num_rows())
            .map(|i| rows.row(i).as_ref().to_vec())
            .collect())
    };
    let (lrows, rrows) = (key_rows(&lk)?, key_rows(&rk)?);
    let key = |rows: &[Vec<u8>], i: usize| -> Vec<u8> { rows.get(i).cloned().unwrap_or_default() };
    let lt = left.column(p.left_time);
    let rt = right.column(p.right_time);
    let float = matches!(
        lt.data_type(),
        ArrowDataType::Float32 | ArrowDataType::Float64
    ) || matches!(
        rt.data_type(),
        ArrowDataType::Float32 | ArrowDataType::Float64
    );
    let (lt, rt) = (Times::of(lt, float)?, Times::of(rt, float)?);
    let mut out = vec![None; left.num_rows()];
    match (lt, rt) {
        (Times::Int(lt), Times::Int(rt)) => {
            let mut groups: HashMap<Vec<u8>, Vec<(i64, u32)>> = HashMap::new();
            for (r, t) in rt.iter().enumerate() {
                if let (Some(t), false) = (t, any_null(&rk, r)) {
                    groups
                        .entry(key(&rrows, r))
                        .or_default()
                        .push((*t, r as u32));
                }
            }
            for g in groups.values_mut() {
                g.sort_by_key(|(t, _)| *t);
            }
            for (l, t) in lt.iter().enumerate() {
                if let (Some(t), false) = (t, any_null(&lk, l)) {
                    out[l] = groups
                        .get(&key(&lrows, l))
                        .and_then(|g| search(g, *t, p.op));
                }
            }
        }
        (Times::Float(lt), Times::Float(rt)) => {
            let mut groups: HashMap<Vec<u8>, Vec<(f64, u32)>> = HashMap::new();
            for (r, t) in rt.iter().enumerate() {
                if let (Some(t), false) = (t, any_null(&rk, r)) {
                    if !t.is_nan() {
                        groups
                            .entry(key(&rrows, r))
                            .or_default()
                            .push((*t, r as u32));
                    }
                }
            }
            for g in groups.values_mut() {
                g.sort_by(|a, b| a.0.total_cmp(&b.0));
            }
            for (l, t) in lt.iter().enumerate() {
                if let (Some(t), false) = (t, any_null(&lk, l)) {
                    if !t.is_nan() {
                        out[l] = groups
                            .get(&key(&lrows, l))
                            .and_then(|g| search(g, *t, p.op));
                    }
                }
            }
        }
        _ => unreachable!("both sides use the same representation"),
    }
    Ok(out)
}

impl Engine {
    /// Run an [`AsOfJoinPlan`].
    pub(crate) fn execute_asof_join(&self, p: &AsOfJoinPlan) -> BoltResult<QueryHandle> {
        if self.has_table(LATERAL_APPLY_RESULT_TABLE) {
            return Err(BoltError::Plan(format!(
                "ASOF JOIN: reserved ephemeral table name '{LATERAL_APPLY_RESULT_TABLE}' \
                 collides with a registered table"
            )));
        }
        // Each side's declared columns, by name (a subplan may also carry
        // synthesised dictionary columns).
        let pick = |b: RecordBatch, s: &crate::plan::Schema| -> BoltResult<RecordBatch> {
            let cols = s
                .fields
                .iter()
                .map(|f| {
                    b.column_by_name(&f.name)
                        .cloned()
                        .ok_or_else(|| err(format!("subplan lost column '{}'", f.name)))
                })
                .collect::<BoltResult<Vec<_>>>()?;
            let fields: Vec<ArrowField> = s
                .fields
                .iter()
                .zip(&cols)
                .map(|(f, c)| ArrowField::new(f.name.clone(), c.data_type().clone(), true))
                .collect();
            RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), cols).map_err(err)
        };
        let left = pick(self.run_subplan(p.left.clone())?, &p.left_schema)?;
        let right = pick(self.run_subplan(p.right.clone())?, &p.right_schema)?;
        let picked = matches(p, &left, &right)?;
        let idx = UInt32Array::from(picked);
        // Matched relation: left ++ right, named as the post template expects.
        let declared = plan_schema_to_arrow_schema(&p.combined_schema)?;
        let mut fields = Vec::with_capacity(declared.fields().len());
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(declared.fields().len());
        for (c, f) in left.columns().iter().zip(declared.fields()) {
            fields.push(ArrowField::new(
                f.name().clone(),
                c.data_type().clone(),
                true,
            ));
            columns.push(Arc::clone(c));
        }
        for (c, f) in right
            .columns()
            .iter()
            .zip(&declared.fields()[left.num_columns()..])
        {
            let col = take(c.as_ref(), &idx, None).map_err(err)?;
            fields.push(ArrowField::new(
                f.name().clone(),
                col.data_type().clone(),
                true,
            ));
            columns.push(col);
        }
        let matched =
            RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns).map_err(err)?;
        self.gpu_tables
            .borrow_mut()
            .remove(LATERAL_APPLY_RESULT_TABLE);
        self.streaming_sources.borrow_mut().insert(
            LATERAL_APPLY_RESULT_TABLE.to_string(),
            TableSource::Materialized(vec![matched]),
        );
        let out = self.run_subplan(p.post.clone());
        self.streaming_sources
            .borrow_mut()
            .remove(LATERAL_APPLY_RESULT_TABLE);
        self.gpu_tables
            .borrow_mut()
            .remove(LATERAL_APPLY_RESULT_TABLE);
        out.map(QueryHandle::from_record_batch)
    }
}
