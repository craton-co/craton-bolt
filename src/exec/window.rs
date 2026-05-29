// SPDX-License-Identifier: Apache-2.0

//! Host-side window-function executor.
//!
//! Computes `func(...) OVER (PARTITION BY ... ORDER BY ...)` against an
//! already-materialised input `RecordBatch`, appending one column per window
//! function to the output. Lowered from [`crate::plan::logical_plan::LogicalPlan::Window`]
//! via [`crate::plan::physical_plan::PhysicalPlan::Window`] and dispatched from
//! the engine's physical executor.
//!
//! # Why host-side
//!
//! Window functions need a global partition + ordering view of the input,
//! which the engine's per-scan fused kernels can't express today. Rather than
//! block the feature on a GPU partition/scan kernel, this executor does the
//! whole thing on the host: correctness first, speed later. The path is
//! deliberately self-contained so a future GPU offload can slot in behind the
//! same `PhysicalPlan::Window` node without touching the frontend or planner.
//!
//! # Strategy
//!
//! 1. Build a permutation that orders rows by the PARTITION BY keys first,
//!    then the ORDER BY keys (via [`arrow::compute::lexsort_to_indices`]).
//!    The PARTITION BY keys are always sorted ascending/nulls-first — their
//!    direction is irrelevant, we only need rows of the same partition to be
//!    contiguous. The ORDER BY keys honour their declared direction / null
//!    placement.
//! 2. Walk the permutation. A partition boundary is a change in any PARTITION
//!    BY key value; within a partition, an *ordering peer group* is a maximal
//!    run of rows with identical ORDER BY key values.
//! 3. For each window function, accumulate over the partition and emit the
//!    per-row output (in permuted order), then scatter back to the original
//!    row order.
//!
//! # Frame
//!
//! Only the SQL default frame is implemented: `RANGE BETWEEN UNBOUNDED
//! PRECEDING AND CURRENT ROW`. Under RANGE the "current row" includes all
//! ordering peers, so every row in a peer group sees the same running
//! aggregate (the value through the *end* of its peer group). With no ORDER
//! BY the whole partition is one peer group, so aggregate windows report the
//! full-partition aggregate on every row — the standard SQL behaviour. The
//! SQL frontend rejects explicit non-default frames, so this executor never
//! sees ROWS / GROUPS or custom bounds.
//!
//! # Ranking functions
//!
//! `ROW_NUMBER` is the 1-based position within the partition (ties broken by
//! the permutation's row order). `RANK` gives tied peers the lowest rank and
//! then skips; `DENSE_RANK` gives tied peers the same rank with no gap.

use std::sync::Arc;

use arrow::compute::{lexsort_to_indices, SortColumn, SortOptions};
use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, RecordBatch, UInt32Array,
};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

use crate::error::{BoltError, BoltResult};
use crate::exec::QueryHandle;
use crate::plan::logical_plan::{
    DataType, Expr, Schema, SortExpr, WindowExpr, WindowFunc,
};

/// Execute a window node host-side, appending one column per `window_exprs`
/// entry to the input batch.
///
/// `output_schema` is the precomputed `input ++ window columns` schema the
/// physical plan carries; we use its tail fields (one per window expr) to
/// build the appended Arrow fields with the correct names / dtypes /
/// nullability.
pub fn execute_window(
    input: QueryHandle,
    window_exprs: &[WindowExpr],
    partition_by: &[Expr],
    order_by: &[SortExpr],
    output_schema: &Schema,
) -> BoltResult<QueryHandle> {
    let batch = input.into_record_batch();
    let n_rows = batch.num_rows();

    // Build the appended Arrow fields from the tail of `output_schema`. The
    // first `input.fields - 0` fields are the input's; the last
    // `window_exprs.len()` are the appended window columns.
    let total_fields = output_schema.fields.len();
    if total_fields < window_exprs.len() {
        return Err(BoltError::Other(
            "window output schema has fewer fields than window expressions".into(),
        ));
    }
    let appended_fields = &output_schema.fields[total_fields - window_exprs.len()..];

    // Empty input: append empty columns of the right type and return.
    if n_rows == 0 {
        let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
        let mut fields: Vec<ArrowField> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        for of in appended_fields {
            // Build a zero-length array of exactly the declared output dtype.
            let arr = empty_array(of.dtype);
            fields.push(ArrowField::new(
                of.name.clone(),
                arr.data_type().clone(),
                true,
            ));
            cols.push(arr);
        }
        let schema = Arc::new(ArrowSchema::new(fields));
        let out = RecordBatch::try_new(schema, cols).map_err(arrow_err)?;
        return Ok(QueryHandle::from_record_batch(out));
    }

    // 1. Compute the partition+order permutation.
    let perm = build_permutation(&batch, partition_by, order_by)?;

    // Pre-extract the partition-key cells and order-key cells per row so the
    // boundary / peer-group checks below are cheap value comparisons.
    let part_keys: Vec<KeyColumn> = partition_by
        .iter()
        .map(|e| KeyColumn::extract(&batch, e))
        .collect::<BoltResult<_>>()?;
    let order_keys: Vec<KeyColumn> = order_by
        .iter()
        .map(|se| KeyColumn::extract(&batch, &se.expr))
        .collect::<BoltResult<_>>()?;

    // 2. Compute each window column (in original row order).
    let mut appended: Vec<ArrayRef> = Vec::with_capacity(window_exprs.len());
    for (we, of) in window_exprs.iter().zip(appended_fields) {
        let arr = compute_window_column(
            &batch, &perm, &part_keys, &order_keys, &we.func, of.dtype,
        )?;
        appended.push(arr);
    }

    // 3. Assemble the output batch: input columns ++ appended window columns.
    let mut cols: Vec<ArrayRef> = batch.columns().to_vec();
    let mut fields: Vec<ArrowField> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect();
    for (arr, of) in appended.iter().zip(appended_fields) {
        fields.push(ArrowField::new(
            of.name.clone(),
            arr.data_type().clone(),
            true,
        ));
    }
    cols.extend(appended);
    let schema = Arc::new(ArrowSchema::new(fields));
    let out = RecordBatch::try_new(schema, cols).map_err(arrow_err)?;
    Ok(QueryHandle::from_record_batch(out))
}

/// Build the row permutation: partition keys (ascending, nulls-first) first,
/// then order-by keys with their declared direction / null placement. The
/// result is the order in which rows are visited so that each partition is
/// contiguous and rows within a partition follow the window ORDER BY.
fn build_permutation(
    batch: &RecordBatch,
    partition_by: &[Expr],
    order_by: &[SortExpr],
) -> BoltResult<Vec<usize>> {
    let n_rows = batch.num_rows();
    if partition_by.is_empty() && order_by.is_empty() {
        // No partitioning and no ordering: physical row order is fine.
        return Ok((0..n_rows).collect());
    }

    let mut sort_cols: Vec<SortColumn> = Vec::with_capacity(partition_by.len() + order_by.len());
    for e in partition_by {
        let idx = column_index(batch, e)?;
        sort_cols.push(SortColumn {
            values: batch.column(idx).clone(),
            options: Some(SortOptions {
                descending: false,
                nulls_first: true,
            }),
        });
    }
    for se in order_by {
        let idx = column_index(batch, &se.expr)?;
        sort_cols.push(SortColumn {
            values: batch.column(idx).clone(),
            options: Some(SortOptions {
                descending: se.descending,
                nulls_first: se.nulls_first,
            }),
        });
    }

    let indices: UInt32Array = lexsort_to_indices(&sort_cols, None).map_err(arrow_err)?;
    Ok(indices.values().iter().map(|&i| i as usize).collect())
}

/// Compute one window column over the permuted rows and scatter results back
/// to original row positions.
fn compute_window_column(
    batch: &RecordBatch,
    perm: &[usize],
    part_keys: &[KeyColumn],
    order_keys: &[KeyColumn],
    func: &WindowFunc,
    out_dtype: DataType,
) -> BoltResult<ArrayRef> {
    let n_rows = batch.num_rows();

    // Optional aggregate-input values (f64 view + validity), only for the
    // aggregate window family.
    let agg_input: Option<NumericColumn> = match func.arg() {
        Some(e) => Some(NumericColumn::extract(batch, e)?),
        None => None,
    };

    // Results indexed by original row position.
    let mut out = ResultBuilder::new(out_dtype, n_rows);

    let mut i = 0usize;
    while i < perm.len() {
        // Find the extent of this partition: [i, part_end).
        let mut part_end = i + 1;
        while part_end < perm.len()
            && part_keys.iter().all(|k| k.eq_rows(perm[i], perm[part_end]))
        {
            part_end += 1;
        }

        compute_partition(
            &perm[i..part_end],
            order_keys,
            agg_input.as_ref(),
            func,
            &mut out,
        )?;

        i = part_end;
    }

    out.finish()
}

/// Compute window outputs for a single partition (`rows` is the slice of the
/// permutation belonging to this partition). Writes into `out` at each row's
/// original position.
fn compute_partition(
    rows: &[usize],
    order_keys: &[KeyColumn],
    agg_input: Option<&NumericColumn>,
    func: &WindowFunc,
    out: &mut ResultBuilder,
) -> BoltResult<()> {
    match func {
        WindowFunc::RowNumber => {
            for (offset, &row) in rows.iter().enumerate() {
                out.set_i64(row, (offset as i64) + 1);
            }
        }
        WindowFunc::Rank | WindowFunc::DenseRank => {
            let dense = matches!(func, WindowFunc::DenseRank);
            let mut dense_rank: i64 = 0;
            let mut idx = 0usize;
            while idx < rows.len() {
                // Extent of this ordering peer group.
                let mut peer_end = idx + 1;
                while peer_end < rows.len()
                    && order_keys
                        .iter()
                        .all(|k| k.eq_rows(rows[idx], rows[peer_end]))
                {
                    peer_end += 1;
                }
                // RANK = 1 + number of rows strictly before this group (so
                // ties share the lowest rank and the next group skips);
                // DENSE_RANK = 1 + number of distinct groups before it.
                dense_rank += 1;
                let value = if dense { dense_rank } else { (idx as i64) + 1 };
                for &row in &rows[idx..peer_end] {
                    out.set_i64(row, value);
                }
                idx = peer_end;
            }
        }
        WindowFunc::Count(_)
        | WindowFunc::Sum(_)
        | WindowFunc::Avg(_)
        | WindowFunc::Min(_)
        | WindowFunc::Max(_) => {
            let input = agg_input.ok_or_else(|| {
                BoltError::Other(format!(
                    "{} window function requires an argument",
                    func.sql_name()
                ))
            })?;
            compute_running_aggregate(rows, order_keys, input, func, out)?;
        }
    }
    Ok(())
}

/// Running-aggregate window under the default RANGE frame: every row sees the
/// aggregate of all peer-or-earlier rows. Because RANGE includes all ordering
/// peers, we accumulate a whole peer group before emitting, then assign the
/// same value to every row in the group.
fn compute_running_aggregate(
    rows: &[usize],
    order_keys: &[KeyColumn],
    input: &NumericColumn,
    func: &WindowFunc,
    out: &mut ResultBuilder,
) -> BoltResult<()> {
    let mut acc = Accumulator::new(func);
    let mut idx = 0usize;
    while idx < rows.len() {
        // Peer group extent.
        let mut peer_end = idx + 1;
        while peer_end < rows.len()
            && order_keys
                .iter()
                .all(|k| k.eq_rows(rows[idx], rows[peer_end]))
        {
            peer_end += 1;
        }
        // Fold every peer row into the running accumulator.
        for &row in &rows[idx..peer_end] {
            acc.push(input.get(row));
        }
        // Emit the running value (through this peer group) for every peer.
        let value = acc.value();
        for &row in &rows[idx..peer_end] {
            out.set_from_accumulator(row, value)?;
        }
        idx = peer_end;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Accumulator
// ---------------------------------------------------------------------------

/// What an aggregate window emits per row. `Int` carries COUNT and integer
/// SUM/MIN/MAX; `Float` carries AVG and float SUM/MIN/MAX; `Null` is a SQL
/// NULL (no non-NULL inputs yet).
#[derive(Clone, Copy)]
enum AggValue {
    Null,
    Int(i64),
    Float(f64),
}

/// Running accumulator over the aggregate-input column's f64 view. SUM / AVG
/// accumulate as f64; the [`ResultBuilder`] coerces to the declared output
/// dtype (rounding to i64 for integer-typed outputs). MIN / MAX track the
/// running extreme; COUNT tracks the number of non-NULL inputs.
struct Accumulator {
    kind: AggKind,
    /// True once at least one non-NULL value has been folded in.
    seen: bool,
    sum: f64,
    count: i64,
    extreme: f64,
}

#[derive(Clone, Copy)]
enum AggKind {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

impl Accumulator {
    fn new(func: &WindowFunc) -> Self {
        let kind = match func {
            WindowFunc::Count(_) => AggKind::Count,
            WindowFunc::Sum(_) => AggKind::Sum,
            WindowFunc::Avg(_) => AggKind::Avg,
            WindowFunc::Min(_) => AggKind::Min,
            WindowFunc::Max(_) => AggKind::Max,
            _ => unreachable!("Accumulator only constructed for aggregate windows"),
        };
        Accumulator {
            kind,
            seen: false,
            sum: 0.0,
            count: 0,
            extreme: 0.0,
        }
    }

    /// Fold one input cell (`None` = SQL NULL, skipped for every aggregate
    /// except its effect on COUNT, which only counts non-NULLs anyway).
    fn push(&mut self, v: Option<f64>) {
        let Some(x) = v else { return };
        self.count += 1;
        self.sum += x;
        if !self.seen {
            self.extreme = x;
            self.seen = true;
        } else {
            match self.kind {
                AggKind::Min => {
                    if x < self.extreme {
                        self.extreme = x;
                    }
                }
                AggKind::Max => {
                    if x > self.extreme {
                        self.extreme = x;
                    }
                }
                _ => {}
            }
        }
    }

    /// The current running value.
    fn value(&self) -> AggValue {
        match self.kind {
            AggKind::Count => AggValue::Int(self.count),
            AggKind::Sum => {
                if !self.seen {
                    // SUM over zero non-NULL rows is SQL NULL.
                    AggValue::Null
                } else {
                    // Carry the exact f64; ResultBuilder rounds for Int outputs.
                    AggValue::Float(self.sum)
                }
            }
            AggKind::Avg => {
                if self.count == 0 {
                    AggValue::Null
                } else {
                    AggValue::Float(self.sum / self.count as f64)
                }
            }
            AggKind::Min | AggKind::Max => {
                if !self.seen {
                    AggValue::Null
                } else {
                    AggValue::Float(self.extreme)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Result builder
// ---------------------------------------------------------------------------

/// Accumulates per-row output values keyed by original row index, then packs
/// them into an Arrow array of the declared output dtype.
///
/// Values are accumulated as `i64` (integer-typed outputs: ranking, COUNT,
/// integer SUM/MIN/MAX) or `f64` (float-typed outputs: AVG, float
/// SUM/MIN/MAX), then narrowed to the declared `out_dtype` at [`finish`].
struct ResultBuilder {
    out_dtype: DataType,
    repr: Repr,
}

/// Internal accumulation representation, chosen from `out_dtype`.
enum Repr {
    Int { values: Vec<i64>, valid: Vec<bool> },
    Float { values: Vec<f64>, valid: Vec<bool> },
}

impl ResultBuilder {
    fn new(out_dtype: DataType, n_rows: usize) -> Self {
        let repr = match out_dtype {
            DataType::Float64 | DataType::Float32 => Repr::Float {
                values: vec![0.0; n_rows],
                valid: vec![false; n_rows],
            },
            // Int64 / Int32 (ranking, COUNT, integer SUM/MIN/MAX). Window
            // outputs are never anything else; default to the integer repr to
            // keep this total.
            _ => Repr::Int {
                values: vec![0; n_rows],
                valid: vec![false; n_rows],
            },
        };
        ResultBuilder { out_dtype, repr }
    }

    /// Set a non-null integer value at `row` (ranking / ROW_NUMBER path).
    fn set_i64(&mut self, row: usize, v: i64) {
        match &mut self.repr {
            Repr::Int { values, valid } => {
                values[row] = v;
                valid[row] = true;
            }
            Repr::Float { values, valid } => {
                values[row] = v as f64;
                valid[row] = true;
            }
        }
    }

    /// Set an aggregate value at `row`, coercing to the builder's repr.
    fn set_from_accumulator(&mut self, row: usize, v: AggValue) -> BoltResult<()> {
        match (&mut self.repr, v) {
            (Repr::Int { valid, .. }, AggValue::Null)
            | (Repr::Float { valid, .. }, AggValue::Null) => {
                valid[row] = false;
            }
            (Repr::Int { values, valid }, AggValue::Int(x)) => {
                values[row] = x;
                valid[row] = true;
            }
            (Repr::Int { values, valid }, AggValue::Float(x)) => {
                values[row] = x as i64;
                valid[row] = true;
            }
            (Repr::Float { values, valid }, AggValue::Int(x)) => {
                values[row] = x as f64;
                valid[row] = true;
            }
            (Repr::Float { values, valid }, AggValue::Float(x)) => {
                values[row] = x;
                valid[row] = true;
            }
        }
        Ok(())
    }

    /// Pack the accumulated values into an Arrow array of exactly the declared
    /// output dtype (narrowing i64→i32 / f64→f32 where the plan asked for the
    /// narrow type — e.g. `MIN(Int32)` preserves `Int32`).
    fn finish(self) -> BoltResult<ArrayRef> {
        use arrow_array::{Float32Array, Int32Array};
        match self.repr {
            Repr::Int { values, valid } => {
                let cells = values.into_iter().zip(valid);
                match self.out_dtype {
                    DataType::Int32 => Ok(Arc::new(Int32Array::from_iter(
                        cells.map(|(v, ok)| if ok { Some(v as i32) } else { None }),
                    ))),
                    _ => Ok(Arc::new(Int64Array::from_iter(
                        cells.map(|(v, ok)| if ok { Some(v) } else { None }),
                    ))),
                }
            }
            Repr::Float { values, valid } => {
                let cells = values.into_iter().zip(valid);
                match self.out_dtype {
                    DataType::Float32 => Ok(Arc::new(Float32Array::from_iter(
                        cells.map(|(v, ok)| if ok { Some(v as f32) } else { None }),
                    ))),
                    _ => Ok(Arc::new(Float64Array::from_iter(
                        cells.map(|(v, ok)| if ok { Some(v) } else { None }),
                    ))),
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Key / numeric column views
// ---------------------------------------------------------------------------

/// An owned, comparable per-row view of a key column (partition or order).
/// Used only for *equality* checks between two row indices (peer-group and
/// partition-boundary detection). Ordering is delegated to Arrow's lexsort.
enum KeyColumn {
    Int64(Vec<Option<i64>>),
    Float64(Vec<Option<f64>>),
    Bool(Vec<Option<bool>>),
    Utf8(Vec<Option<String>>),
}

impl KeyColumn {
    fn extract(batch: &RecordBatch, e: &Expr) -> BoltResult<Self> {
        use arrow_array::{
            BooleanArray, Date32Array, Float32Array, Float64Array, Int32Array, Int64Array,
            StringArray, TimestampNanosecondArray,
        };
        use arrow_schema::DataType as A;

        let idx = column_index(batch, e)?;
        let arr = batch.column(idx);
        let n = arr.len();
        let key = match arr.data_type() {
            A::Int32 => {
                let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
                KeyColumn::Int64(
                    (0..n)
                        .map(|i| if a.is_null(i) { None } else { Some(a.value(i) as i64) })
                        .collect(),
                )
            }
            A::Int64 => {
                let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
                KeyColumn::Int64(
                    (0..n)
                        .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                        .collect(),
                )
            }
            A::Date32 => {
                let a = arr.as_any().downcast_ref::<Date32Array>().unwrap();
                KeyColumn::Int64(
                    (0..n)
                        .map(|i| if a.is_null(i) { None } else { Some(a.value(i) as i64) })
                        .collect(),
                )
            }
            A::Timestamp(arrow_schema::TimeUnit::Nanosecond, _) => {
                let a = arr
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .unwrap();
                KeyColumn::Int64(
                    (0..n)
                        .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                        .collect(),
                )
            }
            A::Float32 => {
                let a = arr.as_any().downcast_ref::<Float32Array>().unwrap();
                KeyColumn::Float64(
                    (0..n)
                        .map(|i| if a.is_null(i) { None } else { Some(a.value(i) as f64) })
                        .collect(),
                )
            }
            A::Float64 => {
                let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
                KeyColumn::Float64(
                    (0..n)
                        .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                        .collect(),
                )
            }
            A::Boolean => {
                let a = arr.as_any().downcast_ref::<BooleanArray>().unwrap();
                KeyColumn::Bool(
                    (0..n)
                        .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                        .collect(),
                )
            }
            A::Utf8 => {
                let a = arr.as_any().downcast_ref::<StringArray>().unwrap();
                KeyColumn::Utf8(
                    (0..n)
                        .map(|i| {
                            if a.is_null(i) {
                                None
                            } else {
                                Some(a.value(i).to_string())
                            }
                        })
                        .collect(),
                )
            }
            other => {
                return Err(BoltError::Other(format!(
                    "window PARTITION BY / ORDER BY key dtype {other:?} is not supported \
                     host-side (supported: Int32/Int64/Float32/Float64/Bool/Utf8/Date32/\
                     Timestamp(ns))"
                )));
            }
        };
        Ok(key)
    }

    /// True if rows `a` and `b` carry the same key value. Two NULLs compare
    /// equal here (they belong to the same partition / peer group, matching
    /// SQL window semantics where NULLs are grouped together). For floats we
    /// treat bitwise-equal NaNs as equal so a partition keyed on NaN doesn't
    /// fragment.
    fn eq_rows(&self, a: usize, b: usize) -> bool {
        match self {
            KeyColumn::Int64(v) => v[a] == v[b],
            KeyColumn::Bool(v) => v[a] == v[b],
            KeyColumn::Utf8(v) => v[a] == v[b],
            KeyColumn::Float64(v) => match (v[a], v[b]) {
                (None, None) => true,
                (Some(x), Some(y)) => x.to_bits() == y.to_bits() || x == y,
                _ => false,
            },
        }
    }
}

/// A per-row f64 view of an aggregate-input column, with validity. NULLs are
/// `None`. Only numeric dtypes are accepted (the type-checker guarantees the
/// aggregate inner is numeric).
struct NumericColumn {
    values: Vec<Option<f64>>,
}

impl NumericColumn {
    fn extract(batch: &RecordBatch, e: &Expr) -> BoltResult<Self> {
        use arrow_array::{Float32Array, Float64Array, Int32Array, Int64Array};
        use arrow_schema::DataType as A;
        use crate::plan::logical_plan::Literal;

        // Aggregate-input literals (e.g. the `COUNT(*)` sentinel `1`) are
        // broadcast to every row. A NULL literal broadcasts to all-NULL.
        if let Expr::Literal(lit) = unwrap_alias(e) {
            let n = batch.num_rows();
            let v = match lit {
                Literal::Null => None,
                Literal::Int32(x) => Some(*x as f64),
                Literal::Int64(x) => Some(*x as f64),
                Literal::Float32(x) => Some(*x as f64),
                Literal::Float64(x) => Some(*x),
                other => {
                    return Err(BoltError::Other(format!(
                        "window aggregate literal input {other:?} is not numeric"
                    )));
                }
            };
            return Ok(NumericColumn {
                values: vec![v; n],
            });
        }

        let idx = column_index(batch, e)?;
        let arr = batch.column(idx);
        let n = arr.len();
        let values: Vec<Option<f64>> = match arr.data_type() {
            A::Int32 => {
                let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
                (0..n)
                    .map(|i| if a.is_null(i) { None } else { Some(a.value(i) as f64) })
                    .collect()
            }
            A::Int64 => {
                let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
                (0..n)
                    .map(|i| if a.is_null(i) { None } else { Some(a.value(i) as f64) })
                    .collect()
            }
            A::Float32 => {
                let a = arr.as_any().downcast_ref::<Float32Array>().unwrap();
                (0..n)
                    .map(|i| if a.is_null(i) { None } else { Some(a.value(i) as f64) })
                    .collect()
            }
            A::Float64 => {
                let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
                (0..n)
                    .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                    .collect()
            }
            other => {
                return Err(BoltError::Other(format!(
                    "window aggregate input dtype {other:?} is not numeric (supported: \
                     Int32/Int64/Float32/Float64)"
                )));
            }
        };
        Ok(NumericColumn { values })
    }

    #[inline]
    fn get(&self, row: usize) -> Option<f64> {
        self.values[row]
    }
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Resolve a window key / aggregate-input expression to an input column
/// index. The host-side window executor only supports bare column references
/// (optionally wrapped in an `Alias`); computed keys / inputs are rejected
/// cleanly so the user gets a clear message instead of a wrong answer.
fn column_index(batch: &RecordBatch, e: &Expr) -> BoltResult<usize> {
    let name = bare_column_name(e)?;
    batch
        .schema()
        .index_of(&name)
        .map_err(|_| BoltError::Other(format!("window key column '{name}' not found in input")))
}

/// Peel any `Alias` wrappers off an expression.
fn unwrap_alias(e: &Expr) -> &Expr {
    match e {
        Expr::Alias(inner, _) => unwrap_alias(inner),
        other => other,
    }
}

/// Extract the bare column name from `Column` / `Alias(Column, _)`. Errors on
/// any other shape.
fn bare_column_name(e: &Expr) -> BoltResult<String> {
    match e {
        Expr::Column(n) => Ok(n.clone()),
        Expr::Alias(inner, _) => bare_column_name(inner),
        other => Err(BoltError::Other(format!(
            "window PARTITION BY / ORDER BY / argument must be a bare column \
             reference host-side, got {other:?}"
        ))),
    }
}

/// Build a zero-length Arrow array of the given output dtype (empty-input path).
fn empty_array(dtype: DataType) -> ArrayRef {
    use arrow_array::{Float32Array, Int32Array};
    match dtype {
        DataType::Float64 => Arc::new(Float64Array::from(Vec::<f64>::new())),
        DataType::Float32 => Arc::new(Float32Array::from(Vec::<f32>::new())),
        DataType::Int32 => Arc::new(Int32Array::from(Vec::<i32>::new())),
        _ => Arc::new(Int64Array::from(Vec::<i64>::new())),
    }
}

fn arrow_err(e: arrow::error::ArrowError) -> BoltError {
    BoltError::Other(format!("arrow: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, Int64Array, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field, Schema as AwSchema};
    use crate::plan::logical_plan::{Field as PField, Schema as PSchema};

    /// Build a handle from columns: (name, Int32 values).
    fn int_col(name: &str, values: Vec<i32>) -> (ArrowField, ArrayRef) {
        (
            ArrowField::new(name, ArrowDataType::Int32, true),
            Arc::new(Int32Array::from(values)) as ArrayRef,
        )
    }
    fn str_col(name: &str, values: Vec<&str>) -> (ArrowField, ArrayRef) {
        (
            ArrowField::new(name, ArrowDataType::Utf8, true),
            Arc::new(StringArray::from(values)) as ArrayRef,
        )
    }

    fn handle(cols: Vec<(ArrowField, ArrayRef)>) -> QueryHandle {
        let fields: Vec<Field> = cols.iter().map(|(f, _)| f.clone()).collect();
        let arrays: Vec<ArrayRef> = cols.into_iter().map(|(_, a)| a).collect();
        let schema = Arc::new(AwSchema::new(fields));
        QueryHandle::from_record_batch(RecordBatch::try_new(schema, arrays).unwrap())
    }

    /// Build a plan Schema for `input_fields ++ window_outputs`.
    fn out_schema(input: &[(&str, DataType)], windows: &[(&str, DataType)]) -> PSchema {
        let mut fields: Vec<PField> = input
            .iter()
            .map(|(n, d)| PField::new(*n, *d, true))
            .collect();
        for (n, d) in windows {
            fields.push(PField::new(*n, *d, true));
        }
        PSchema::new(fields)
    }

    fn col(name: &str) -> Expr {
        Expr::Column(name.to_string())
    }

    fn order(name: &str) -> SortExpr {
        SortExpr {
            expr: col(name),
            descending: false,
            nulls_first: true,
        }
    }

    fn as_i64(batch: &RecordBatch, name: &str) -> Vec<Option<i64>> {
        let idx = batch.schema().index_of(name).unwrap();
        let arr = batch
            .column(idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        (0..arr.len())
            .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
            .collect()
    }

    fn as_f64(batch: &RecordBatch, name: &str) -> Vec<Option<f64>> {
        let idx = batch.schema().index_of(name).unwrap();
        let arr = batch
            .column(idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        (0..arr.len())
            .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
            .collect()
    }

    fn as_str(batch: &RecordBatch, name: &str) -> Vec<String> {
        let idx = batch.schema().index_of(name).unwrap();
        let arr = batch
            .column(idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        (0..arr.len()).map(|i| arr.value(i).to_string()).collect()
    }

    /// ROW_NUMBER() OVER (PARTITION BY k ORDER BY v).
    #[test]
    fn row_number_over_partition() {
        // Rows (k, v):
        //   a,30  a,10  b,20  a,20  b,10
        // Partition a ordered by v: 10,20,30 -> row_number 1,2,3
        // Partition b ordered by v: 10,20    -> row_number 1,2
        let h = handle(vec![
            str_col("k", vec!["a", "a", "b", "a", "b"]),
            int_col("v", vec![30, 10, 20, 20, 10]),
        ]);
        let wexprs = vec![WindowExpr {
            func: WindowFunc::RowNumber,
            output_name: "rn".into(),
        }];
        let os = out_schema(
            &[("k", DataType::Utf8), ("v", DataType::Int32)],
            &[("rn", DataType::Int64)],
        );
        let out = execute_window(h, &wexprs, &[col("k")], &[order("v")], &os)
            .unwrap()
            .into_record_batch();

        // Match output back against the original (k, v) row order.
        let ks = as_str(&out, "k");
        let vs: Vec<Option<i64>> = {
            let idx = out.schema().index_of("v").unwrap();
            let a = out.column(idx).as_any().downcast_ref::<Int32Array>().unwrap();
            (0..a.len()).map(|i| Some(a.value(i) as i64)).collect()
        };
        let rn = as_i64(&out, "rn");
        // Re-key by (k,v) to expected row_number.
        for i in 0..ks.len() {
            let expected = match (ks[i].as_str(), vs[i].unwrap()) {
                ("a", 10) => 1,
                ("a", 20) => 2,
                ("a", 30) => 3,
                ("b", 10) => 1,
                ("b", 20) => 2,
                _ => unreachable!(),
            };
            assert_eq!(rn[i], Some(expected), "row {i} (k={}, v={:?})", ks[i], vs[i]);
        }
    }

    /// RANK() with ties: tied rows share the lowest rank, then a gap.
    #[test]
    fn rank_with_ties() {
        // Single partition, ORDER BY v: 10,10,20,30
        // RANK:       1,1,3,4
        // DENSE_RANK: 1,1,2,3
        let h = handle(vec![int_col("v", vec![10, 30, 10, 20])]);
        let wexprs = vec![
            WindowExpr {
                func: WindowFunc::Rank,
                output_name: "rk".into(),
            },
            WindowExpr {
                func: WindowFunc::DenseRank,
                output_name: "dr".into(),
            },
        ];
        let os = out_schema(
            &[("v", DataType::Int32)],
            &[("rk", DataType::Int64), ("dr", DataType::Int64)],
        );
        let out = execute_window(h, &wexprs, &[], &[order("v")], &os)
            .unwrap()
            .into_record_batch();

        let idx = out.schema().index_of("v").unwrap();
        let va = out.column(idx).as_any().downcast_ref::<Int32Array>().unwrap();
        let vs: Vec<i32> = (0..va.len()).map(|i| va.value(i)).collect();
        let rk = as_i64(&out, "rk");
        let dr = as_i64(&out, "dr");
        for i in 0..vs.len() {
            let (erk, edr) = match vs[i] {
                10 => (1, 1),
                20 => (3, 2),
                30 => (4, 3),
                _ => unreachable!(),
            };
            assert_eq!(rk[i], Some(erk), "rank row {i} v={}", vs[i]);
            assert_eq!(dr[i], Some(edr), "dense_rank row {i} v={}", vs[i]);
        }
    }

    /// SUM(v) OVER (PARTITION BY k) — no ORDER BY, so every row sees the
    /// full-partition sum.
    #[test]
    fn sum_over_partition_no_order() {
        // k=a: v=1,2,3 -> sum 6 on every a row
        // k=b: v=10,20 -> sum 30 on every b row
        let h = handle(vec![
            str_col("k", vec!["a", "b", "a", "b", "a"]),
            int_col("v", vec![1, 10, 2, 20, 3]),
        ]);
        let wexprs = vec![WindowExpr {
            func: WindowFunc::Sum(col("v")),
            output_name: "sum_v".into(),
        }];
        // SUM(Int32) widens to Int64.
        let os = out_schema(
            &[("k", DataType::Utf8), ("v", DataType::Int32)],
            &[("sum_v", DataType::Int64)],
        );
        let out = execute_window(h, &wexprs, &[col("k")], &[], &os)
            .unwrap()
            .into_record_batch();

        let ks = as_str(&out, "k");
        let sums = as_i64(&out, "sum_v");
        for i in 0..ks.len() {
            let expected = match ks[i].as_str() {
                "a" => 6,
                "b" => 30,
                _ => unreachable!(),
            };
            assert_eq!(sums[i], Some(expected), "row {i} k={}", ks[i]);
        }
    }

    /// Running SUM with ORDER BY: cumulative within partition, peers share the
    /// value (RANGE frame).
    #[test]
    fn running_sum_with_order_and_ties() {
        // Single partition. ORDER BY v: rows v=10,10,20.
        // RANGE frame: the two v=10 peers both see sum=20 (10+10);
        // the v=20 row sees 40.
        let h = handle(vec![int_col("v", vec![20, 10, 10])]);
        let wexprs = vec![WindowExpr {
            func: WindowFunc::Sum(col("v")),
            output_name: "rs".into(),
        }];
        let os = out_schema(
            &[("v", DataType::Int32)],
            &[("rs", DataType::Int64)],
        );
        let out = execute_window(h, &wexprs, &[], &[order("v")], &os)
            .unwrap()
            .into_record_batch();

        let idx = out.schema().index_of("v").unwrap();
        let va = out.column(idx).as_any().downcast_ref::<Int32Array>().unwrap();
        let vs: Vec<i32> = (0..va.len()).map(|i| va.value(i)).collect();
        let rs = as_i64(&out, "rs");
        for i in 0..vs.len() {
            let expected = match vs[i] {
                10 => 20, // both peers see the peer-group-inclusive running sum
                20 => 40,
                _ => unreachable!(),
            };
            assert_eq!(rs[i], Some(expected), "row {i} v={}", vs[i]);
        }
    }

    /// AVG(v) OVER (PARTITION BY k) yields Float64 full-partition averages.
    #[test]
    fn avg_over_partition() {
        let h = handle(vec![
            str_col("k", vec!["a", "a", "b"]),
            int_col("v", vec![1, 3, 10]),
        ]);
        let wexprs = vec![WindowExpr {
            func: WindowFunc::Avg(col("v")),
            output_name: "avg_v".into(),
        }];
        let os = out_schema(
            &[("k", DataType::Utf8), ("v", DataType::Int32)],
            &[("avg_v", DataType::Float64)],
        );
        let out = execute_window(h, &wexprs, &[col("k")], &[], &os)
            .unwrap()
            .into_record_batch();
        let ks = as_str(&out, "k");
        let avgs = as_f64(&out, "avg_v");
        for i in 0..ks.len() {
            let expected = match ks[i].as_str() {
                "a" => 2.0,
                "b" => 10.0,
                _ => unreachable!(),
            };
            assert_eq!(avgs[i], Some(expected));
        }
    }

    /// COUNT over partition ignores NULL inputs.
    #[test]
    fn count_over_partition_skips_nulls() {
        let schema = Arc::new(AwSchema::new(vec![
            Field::new("k", ArrowDataType::Utf8, true),
            Field::new("v", ArrowDataType::Int64, true),
        ]));
        let kcol = Arc::new(StringArray::from(vec!["a", "a", "a"])) as ArrayRef;
        let vcol = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;
        let h = QueryHandle::from_record_batch(
            RecordBatch::try_new(schema, vec![kcol, vcol]).unwrap(),
        );
        let wexprs = vec![WindowExpr {
            func: WindowFunc::Count(col("v")),
            output_name: "cnt".into(),
        }];
        let os = out_schema(
            &[("k", DataType::Utf8), ("v", DataType::Int64)],
            &[("cnt", DataType::Int64)],
        );
        let out = execute_window(h, &wexprs, &[col("k")], &[], &os)
            .unwrap()
            .into_record_batch();
        // 2 non-NULL values in the single partition.
        let cnt = as_i64(&out, "cnt");
        assert_eq!(cnt, vec![Some(2), Some(2), Some(2)]);
    }

    /// Empty input produces an empty output with the appended column present.
    #[test]
    fn empty_input() {
        let schema = Arc::new(AwSchema::new(vec![Field::new(
            "v",
            ArrowDataType::Int32,
            true,
        )]));
        let v = Arc::new(Int32Array::from(Vec::<i32>::new())) as ArrayRef;
        let h = QueryHandle::from_record_batch(RecordBatch::try_new(schema, vec![v]).unwrap());
        let wexprs = vec![WindowExpr {
            func: WindowFunc::RowNumber,
            output_name: "rn".into(),
        }];
        let os = out_schema(&[("v", DataType::Int32)], &[("rn", DataType::Int64)]);
        let out = execute_window(h, &wexprs, &[], &[order("v")], &os)
            .unwrap()
            .into_record_batch();
        assert_eq!(out.num_rows(), 0);
        assert!(out.schema().index_of("rn").is_ok());
    }
}
