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
use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, UInt32Array};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

use crate::error::{BoltError, BoltResult};
use crate::exec::QueryHandle;
use crate::plan::logical_plan::{DataType, Expr, Schema, SortExpr, WindowExpr, WindowFunc};

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
    // GPU path (env-gated, conservative). Returns `Ok(None)` to decline — for
    // unsupported funcs/frames, wide/Utf8 keys, oversized inputs, missing
    // device, or `--features cuda-stub` — in which case we fall through to the
    // host executor below. Any *successful* GPU result must match the host
    // path bit-for-bit (same NULL grouping, same RANGE-frame peer semantics).
    match gpu::try_execute_window_gpu(&input, window_exprs, partition_by, order_by, output_schema) {
        Ok(Some(handle)) => return Ok(handle),
        Ok(None) => { /* decline -> host fallback */ }
        Err(e) => return Err(e),
    }

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
        let arr =
            compute_window_column(&batch, &perm, &part_keys, &order_keys, &we.func, of.dtype)?;
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
        while part_end < perm.len() && part_keys.iter().all(|k| k.eq_rows(perm[i], perm[part_end]))
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
        // Fold every peer row into the running accumulator. Integer SUM
        // overflow surfaces here as a BoltError (never a silent wrap).
        for &row in &rows[idx..peer_end] {
            acc.push(input.get(row))?;
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

/// Order two `f64`s under the DuckDB float convention: NaN sorts as the
/// *largest* value (greater than +inf), and all NaN bit-patterns are treated
/// as equal. We delegate to [`f64::total_cmp`] (which already orders
/// `-0 < +0` and places NaN at the extremes) and then fold the negative-NaN
/// half up to the top so *every* NaN is the maximum. This makes MIN skip NaN
/// unless the input is all-NaN, and MAX surface NaN whenever one is present —
/// matching the NaN-ignoring `f64::min`/`f64::max` scalar path in
/// `aggregate.rs` for MIN, and giving MAX a single well-defined NaN answer.
#[inline]
fn float_total_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater, // NaN is the largest
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.total_cmp(&b),
    }
}

/// Running accumulator over the aggregate-input column. SUM/MIN/MAX stay on
/// the input column's native numeric lane so integer columns never round-trip
/// through `f64`:
///
/// * integer SUM accumulates in `int_sum` with `checked_add` (errors loudly on
///   `i64` overflow, mirroring the `SUM(integer)` contract in `aggregate.rs`);
/// * integer MIN/MAX pass the exact `i64` extreme through unchanged;
/// * float SUM/MIN/MAX accumulate in the `f64` lane;
/// * AVG always accumulates in `f64` (documented: averages are inherently
///   fractional, so the f64 lane is the natural representation);
/// * COUNT tracks the number of non-NULL inputs.
///
/// The lane (`Int` vs `Float`) is locked in by the first non-NULL cell, which
/// always matches the column's dtype because a column is uniformly one lane.
struct Accumulator {
    kind: AggKind,
    /// True once at least one non-NULL value has been folded in.
    seen: bool,
    /// `true` once we've decided the SUM/MIN/MAX lane is integer (vs float).
    int_lane: bool,
    /// f64 lane: SUM (float), AVG (always), MIN/MAX (float extreme).
    sum: f64,
    extreme: f64,
    /// i64 lane: SUM (integer, checked), MIN/MAX (integer extreme).
    int_sum: i64,
    int_extreme: i64,
    count: i64,
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
            int_lane: false,
            sum: 0.0,
            extreme: 0.0,
            int_sum: 0,
            int_extreme: 0,
            count: 0,
        }
    }

    /// Fold one input cell (`None` = SQL NULL, skipped for every aggregate
    /// except its effect on COUNT, which only counts non-NULLs anyway).
    ///
    /// Returns `Err` only when an integer SUM overflows `i64`, matching the
    /// engine's never-silently-wrong invariant (see `aggregate.rs`'s
    /// `SUM(integer) overflow` contract).
    fn push(&mut self, v: Option<Cell>) -> BoltResult<()> {
        let Some(cell) = v else { return Ok(()) };
        self.count += 1;

        match cell {
            Cell::Int(x) => {
                if !self.seen {
                    self.int_lane = true;
                    self.int_sum = 0;
                    self.int_extreme = x;
                    self.seen = true;
                }
                // SUM(integer): exact, checked. AVG also needs an f64 running
                // sum; keep it alongside for the AVG lane.
                self.sum += x as f64;
                match self.kind {
                    AggKind::Sum => {
                        self.int_sum = self.int_sum.checked_add(x).ok_or_else(|| {
                            BoltError::Type(
                                "SUM(integer) overflow: accumulator exceeds i64 range".to_string(),
                            )
                        })?;
                    }
                    AggKind::Min => {
                        if x < self.int_extreme {
                            self.int_extreme = x;
                        }
                    }
                    AggKind::Max => {
                        if x > self.int_extreme {
                            self.int_extreme = x;
                        }
                    }
                    _ => {}
                }
            }
            Cell::Float(x) => {
                if !self.seen {
                    self.int_lane = false;
                    self.extreme = x;
                    self.seen = true;
                    self.sum = x;
                } else {
                    self.sum += x;
                    match self.kind {
                        // NaN-as-largest: MIN keeps the smaller, MAX the
                        // larger, under `float_total_cmp`. A leading NaN no
                        // longer sticks for MIN (it's the maximum, so any
                        // real value beats it), and MAX returns NaN if present.
                        AggKind::Min => {
                            if float_total_cmp(x, self.extreme) == std::cmp::Ordering::Less {
                                self.extreme = x;
                            }
                        }
                        AggKind::Max => {
                            if float_total_cmp(x, self.extreme) == std::cmp::Ordering::Greater {
                                self.extreme = x;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }

    /// The current running value.
    fn value(&self) -> AggValue {
        match self.kind {
            AggKind::Count => AggValue::Int(self.count),
            AggKind::Sum => {
                if !self.seen {
                    // SUM over zero non-NULL rows is SQL NULL.
                    AggValue::Null
                } else if self.int_lane {
                    // Exact integer sum (checked during push).
                    AggValue::Int(self.int_sum)
                } else {
                    AggValue::Float(self.sum)
                }
            }
            AggKind::Avg => {
                if self.count == 0 {
                    AggValue::Null
                } else {
                    // AVG stays f64 by design.
                    AggValue::Float(self.sum / self.count as f64)
                }
            }
            AggKind::Min | AggKind::Max => {
                if !self.seen {
                    AggValue::Null
                } else if self.int_lane {
                    // Exact i64 passthrough — no float round-trip.
                    AggValue::Int(self.int_extreme)
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
                    DataType::Int32 => Ok(Arc::new(Int32Array::from_iter(cells.map(|(v, ok)| {
                        if ok {
                            Some(v as i32)
                        } else {
                            None
                        }
                    })))),
                    _ => Ok(Arc::new(Int64Array::from_iter(cells.map(|(v, ok)| {
                        if ok {
                            Some(v)
                        } else {
                            None
                        }
                    })))),
                }
            }
            Repr::Float { values, valid } => {
                let cells = values.into_iter().zip(valid);
                match self.out_dtype {
                    DataType::Float32 => {
                        Ok(Arc::new(Float32Array::from_iter(cells.map(|(v, ok)| {
                            if ok {
                                Some(v as f32)
                            } else {
                                None
                            }
                        }))))
                    }
                    _ => Ok(Arc::new(Float64Array::from_iter(cells.map(|(v, ok)| {
                        if ok {
                            Some(v)
                        } else {
                            None
                        }
                    })))),
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
pub(crate) enum KeyColumn {
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
                        .map(|i| {
                            if a.is_null(i) {
                                None
                            } else {
                                Some(a.value(i) as i64)
                            }
                        })
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
                        .map(|i| {
                            if a.is_null(i) {
                                None
                            } else {
                                Some(a.value(i) as i64)
                            }
                        })
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
                        .map(|i| {
                            if a.is_null(i) {
                                None
                            } else {
                                Some(a.value(i) as f64)
                            }
                        })
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

/// A per-row native view of an aggregate-input column, with validity. NULLs
/// are `None`. Only numeric dtypes are accepted (the type-checker guarantees
/// the aggregate inner is numeric).
///
/// Integer inputs (`Int32`/`Int64`) keep an exact `i64` lane so SUM/MIN/MAX of
/// integers never round-trip through `f64` — values beyond 2^53 (which `f64`
/// cannot represent exactly) survive intact. Float inputs keep an `f64` lane.
/// The two lanes are surfaced as [`Cell`]s so the [`Accumulator`] can stay on
/// the native type per column.
pub(crate) enum NumericColumn {
    Int(Vec<Option<i64>>),
    Float(Vec<Option<f64>>),
}

/// One folded aggregate-input cell, carrying the column's native numeric type.
#[derive(Clone, Copy)]
enum Cell {
    Int(i64),
    Float(f64),
}

impl NumericColumn {
    fn extract(batch: &RecordBatch, e: &Expr) -> BoltResult<Self> {
        use crate::plan::logical_plan::Literal;
        use arrow_array::{Float32Array, Float64Array, Int32Array, Int64Array};
        use arrow_schema::DataType as A;

        // Aggregate-input literals (e.g. the `COUNT(*)` sentinel `1`) are
        // broadcast to every row. A NULL literal broadcasts to all-NULL.
        // Integer literals stay on the integer lane; float literals on the
        // float lane.
        if let Expr::Literal(lit) = unwrap_alias(e) {
            let n = batch.num_rows();
            return Ok(match lit {
                Literal::Null => NumericColumn::Int(vec![None; n]),
                Literal::Int32(x) => NumericColumn::Int(vec![Some(*x as i64); n]),
                Literal::Int64(x) => NumericColumn::Int(vec![Some(*x); n]),
                Literal::Float32(x) => NumericColumn::Float(vec![Some(*x as f64); n]),
                Literal::Float64(x) => NumericColumn::Float(vec![Some(*x); n]),
                other => {
                    return Err(BoltError::Other(format!(
                        "window aggregate literal input {other:?} is not numeric"
                    )));
                }
            });
        }

        let idx = column_index(batch, e)?;
        let arr = batch.column(idx);
        let n = arr.len();
        let col = match arr.data_type() {
            A::Int32 => {
                let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
                NumericColumn::Int(
                    (0..n)
                        .map(|i| {
                            if a.is_null(i) {
                                None
                            } else {
                                Some(a.value(i) as i64)
                            }
                        })
                        .collect(),
                )
            }
            A::Int64 => {
                let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
                NumericColumn::Int(
                    (0..n)
                        .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                        .collect(),
                )
            }
            A::Float32 => {
                let a = arr.as_any().downcast_ref::<Float32Array>().unwrap();
                NumericColumn::Float(
                    (0..n)
                        .map(|i| {
                            if a.is_null(i) {
                                None
                            } else {
                                Some(a.value(i) as f64)
                            }
                        })
                        .collect(),
                )
            }
            A::Float64 => {
                let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
                NumericColumn::Float(
                    (0..n)
                        .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                        .collect(),
                )
            }
            other => {
                return Err(BoltError::Other(format!(
                    "window aggregate input dtype {other:?} is not numeric (supported: \
                     Int32/Int64/Float32/Float64)"
                )));
            }
        };
        Ok(col)
    }

    #[inline]
    fn get(&self, row: usize) -> Option<Cell> {
        match self {
            NumericColumn::Int(v) => v[row].map(Cell::Int),
            NumericColumn::Float(v) => v[row].map(Cell::Float),
        }
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

// ===========================================================================
// GPU / framed window-function path
// ===========================================================================

/// GPU offload for a precisely-scoped subset of window functions.
///
/// # Scope (this increment)
///
/// The GPU path accelerates exactly:
///   * `ROW_NUMBER()`
///   * `RANK()` / `DENSE_RANK()`
///   * running `SUM(col)` / `COUNT(col)` under the **default RANGE frame**
///
/// and only when every gate in [`gpu::dispatch_decision`] passes:
///   * all PARTITION BY / ORDER BY keys are bare columns of an `i64`-encodable
///     dtype (Int32/Int64/Date32/Timestamp(ns)/Bool) — Utf8 / Float / wide
///     keys fall back to host (Float ordering is delegated to the host sort's
///     total-order semantics, which we do not replicate here);
///   * aggregate inputs (SUM/COUNT) are bare Int32/Int64 columns;
///   * row count fits one device block ([`crate::jit::window_kernel::WINDOW_BLOCK_SIZE`]);
///     larger inputs need the deferred multi-block segmented scan.
///
/// Everything else — `AVG`/`MIN`/`MAX` windows, Float aggregate inputs,
/// explicit `ROWS`/`RANGE BETWEEN` frames, `LAG`/`LEAD`, computed keys — is
/// declined (`Ok(None)`) and handled by the host executor, whose NULL-grouping
/// and RANGE-frame peer semantics this path reproduces exactly.
///
/// # Algorithm (per partition, on device)
///
/// 1. Sort rows by `(partition_key.., order_key..)` (host `lexsort_to_indices`
///    here — a follow-up wires the existing GPU sort; the permutation is
///    identical either way).
/// 2. `bolt_window_boundary_flags` writes `part_head` / `peer_head` flags.
/// 3. `bolt_window_segmented_scan` computes the partition-local inclusive
///    prefix sum of a per-function value column:
///      * ROW_NUMBER  -> values = 1            -> inclusive prefix is the index;
///      * COUNT(x)    -> values = (x non-null) -> inclusive prefix is the count;
///      * SUM(x)      -> values = x            -> inclusive prefix is the sum,
///                       then the host lifts every peer to the peer-group end
///                       (RANGE frame);
///      * DENSE_RANK  -> segmented prefix of `peer_head` (1 at each new group);
///      * RANK        -> derived from peer-group start offsets.
/// 4. Scatter the per-row results back to original row order.
///
/// All four derivations are pure functions of `(perm, part_head, peer_head,
/// inclusive_prefix)` and live in host-testable form in this module so the
/// math is verified without a device (the device only provides the segmented
/// prefix sum — a single well-tested primitive).
pub(crate) mod gpu {
    use super::*;
    use crate::jit::window_kernel::{
        compile_boundary_flag_kernel, compile_segmented_scan_kernel, BOUNDARY_FLAG_ENTRY,
        SEGMENTED_SCAN_ENTRY, WINDOW_BLOCK_SIZE,
    };

    // Device-launch imports. These mirror the hand-rolled `cuLaunchKernel`
    // marshaling pattern in `crate::exec::engine` (see `string_length_column` /
    // `string_transform_column`). They compile under `--features cuda-stub`
    // because every sibling executor (`groupby_shmem_*_exec`) uses the same set;
    // the actual device touch only happens inside `launch_window_kernels`, which
    // is reached solely on `BOLT_GPU_WINDOW=1` + a `Decision::Accept`.
    use std::ffi::c_void;
    use std::ptr;

    use crate::cuda::cuda_sys::{self, CUdeviceptr};
    use crate::cuda::GpuVec;
    use crate::exec::launch::{grid_x_for, CudaStream};
    use crate::exec::module_cache;

    /// Force-override env var, mirroring `BOLT_GPU_SORT`. Unset / "0" keeps the
    /// GPU window path OFF (host path always runs); "1" opts in on supported
    /// shapes. The path is conservative by default because device behavior is
    /// unverifiable in CI without a GPU.
    pub(crate) const BOLT_GPU_WINDOW_ENV: &str = "BOLT_GPU_WINDOW";

    /// One supported window function, resolved against its input columns.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum GpuWindowKind {
        RowNumber,
        Rank,
        DenseRank,
        /// running COUNT(col): value column is the non-null indicator.
        Count,
        /// running SUM(col): value column is the i64 input (NULL -> 0).
        Sum,
    }

    /// Result of the (pure, host-only) dispatch gate.
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub(crate) enum Decision {
        /// GPU-eligible: one kind per output column, in `window_exprs` order.
        Accept(Vec<GpuWindowKind>),
        /// Decline with a human-readable reason (host fallback).
        Decline(String),
    }

    /// Is `dtype` representable in the kernel's single `i64` key lane *with
    /// equality semantics matching the host path*? Float is excluded: the host
    /// sort uses a total order over floats (NaN handling) that we do not
    /// reproduce in the i64 lane, so Float keys fall back. Utf8 is excluded
    /// (wide key).
    pub(crate) fn key_dtype_is_i64_encodable(dt: &arrow_schema::DataType) -> bool {
        use arrow_schema::DataType as A;
        matches!(
            dt,
            A::Int8
                | A::Int16
                | A::Int32
                | A::Int64
                | A::Boolean
                | A::Date32
                | A::Date64
                | A::Timestamp(_, _)
        )
    }

    /// Is `dtype` a GPU-eligible aggregate-input column (SUM/COUNT)? Int only;
    /// Float SUM would need an f64 segmented lane (deferred).
    pub(crate) fn agg_input_is_int(dt: &arrow_schema::DataType) -> bool {
        use arrow_schema::DataType as A;
        matches!(dt, A::Int8 | A::Int16 | A::Int32 | A::Int64)
    }

    /// Pure dispatch predicate. No device touch, no allocation beyond the
    /// returned kind vector — unit-testable under `cuda-stub`.
    ///
    /// `key_dtypes` are the resolved Arrow dtypes of the PARTITION BY then
    /// ORDER BY key columns; `agg_dtypes[i]` is `Some(dtype)` for the aggregate
    /// input of `funcs[i]` (SUM/COUNT) or `None` for the ranking functions.
    pub(crate) fn dispatch_decision(
        funcs: &[&WindowFunc],
        n_rows: usize,
        key_dtypes: &[arrow_schema::DataType],
        agg_dtypes: &[Option<arrow_schema::DataType>],
    ) -> Decision {
        if funcs.is_empty() {
            return Decision::Decline("no window functions".into());
        }
        // One block only (intra-block segmented scan). Empty input is handled
        // by the host fast-path, so require 1..=BLOCK_SIZE here.
        if n_rows == 0 || n_rows > WINDOW_BLOCK_SIZE as usize {
            return Decision::Decline(format!(
                "row count {n_rows} not in 1..={} (single-block scan only)",
                WINDOW_BLOCK_SIZE
            ));
        }
        for kd in key_dtypes {
            if !key_dtype_is_i64_encodable(kd) {
                return Decision::Decline(format!("key dtype {kd:?} not i64-encodable"));
            }
        }
        let mut kinds = Vec::with_capacity(funcs.len());
        for (f, agg) in funcs.iter().zip(agg_dtypes) {
            let kind = match f {
                WindowFunc::RowNumber => GpuWindowKind::RowNumber,
                WindowFunc::Rank => GpuWindowKind::Rank,
                WindowFunc::DenseRank => GpuWindowKind::DenseRank,
                WindowFunc::Count(_) => match agg {
                    Some(dt) if agg_input_is_int(dt) => GpuWindowKind::Count,
                    Some(dt) => {
                        return Decision::Decline(format!(
                            "COUNT input dtype {dt:?} not GPU-eligible"
                        ))
                    }
                    None => return Decision::Decline("COUNT missing input dtype".into()),
                },
                WindowFunc::Sum(_) => match agg {
                    Some(dt) if agg_input_is_int(dt) => GpuWindowKind::Sum,
                    Some(dt) => {
                        return Decision::Decline(format!(
                            "SUM input dtype {dt:?} not GPU-eligible"
                        ))
                    }
                    None => return Decision::Decline("SUM missing input dtype".into()),
                },
                WindowFunc::Avg(_) => {
                    return Decision::Decline("AVG window not GPU-accelerated".into())
                }
                WindowFunc::Min(_) => {
                    return Decision::Decline("MIN window not GPU-accelerated".into())
                }
                WindowFunc::Max(_) => {
                    return Decision::Decline("MAX window not GPU-accelerated".into())
                }
            };
            kinds.push(kind);
        }
        Decision::Accept(kinds)
    }

    // -- Host-side reference of the device segmented-scan derivations --------
    //
    // The device computes ONE primitive: the partition-local inclusive
    // segmented prefix sum of an i64 value column, resetting at each
    // `part_head`. These functions express the per-function value column and
    // the post-scan derivation in pure Rust so the math is testable without a
    // device AND so the (deferred) host-fallback-under-cuda-stub path can reuse
    // them. Inputs are all in PERMUTED row order.

    /// Segmented inclusive prefix sum over `values`, resetting at every index
    /// `i` where `seg_head[i]` is true. This is exactly what
    /// `bolt_window_segmented_scan` computes (single-block); kept here as the
    /// device-equivalent reference.
    ///
    /// Used by the host-side `gpu_path` unit tests AND by the RANK/DENSE_RANK
    /// derivations on the live device-launch path (those two are pure functions
    /// of the boundary flags, so they reuse this host scan rather than a second
    /// device launch). ROW_NUMBER / SUM / COUNT instead consume the DEVICE
    /// `bolt_window_segmented_scan` output (see `execute_window_on_device`).
    pub(crate) fn segmented_inclusive_sum(values: &[i64], seg_head: &[bool]) -> Vec<i64> {
        debug_assert_eq!(values.len(), seg_head.len());
        let mut out = vec![0i64; values.len()];
        let mut acc: i64 = 0;
        for i in 0..values.len() {
            if seg_head[i] {
                acc = values[i];
            } else {
                acc += values[i];
            }
            out[i] = acc;
        }
        out
    }

    /// ROW_NUMBER in permuted order: partition-local inclusive scan of all-1s.
    pub(crate) fn derive_row_number(part_head: &[bool]) -> Vec<i64> {
        let ones = vec![1i64; part_head.len()];
        segmented_inclusive_sum(&ones, part_head)
    }

    /// DENSE_RANK in permuted order: partition-local inclusive scan of the
    /// `peer_head` indicator (1 at each new ordering peer group). Since
    /// `peer_head` is always set at a partition head, the per-partition scan
    /// gives 1 for the first peer group, 2 for the next, etc.
    pub(crate) fn derive_dense_rank(part_head: &[bool], peer_head: &[bool]) -> Vec<i64> {
        let vals: Vec<i64> = peer_head.iter().map(|&h| if h { 1 } else { 0 }).collect();
        segmented_inclusive_sum(&vals, part_head)
    }

    /// RANK in permuted order: 1 + (number of rows before the current row's
    /// peer group within the partition). Equivalently, the partition-local
    /// row index (ROW_NUMBER) of the FIRST row in each peer group, broadcast to
    /// every peer. We compute ROW_NUMBER then, walking each peer group, stamp
    /// the group's leading row-number onto every member.
    pub(crate) fn derive_rank(part_head: &[bool], peer_head: &[bool]) -> Vec<i64> {
        let rn = derive_row_number(part_head);
        let mut out = vec![0i64; part_head.len()];
        let mut current = 0i64;
        for i in 0..part_head.len() {
            if peer_head[i] {
                current = rn[i];
            }
            out[i] = current;
        }
        out
    }

    /// Running COUNT(col) (RANGE frame) in permuted order: partition-local
    /// inclusive scan of the non-null indicator, then lift each peer to the
    /// peer-group-end count (every peer sees the same value under RANGE).
    pub(crate) fn derive_count(
        part_head: &[bool],
        peer_head: &[bool],
        non_null: &[bool],
    ) -> Vec<i64> {
        let vals: Vec<i64> = non_null.iter().map(|&b| if b { 1 } else { 0 }).collect();
        let incl = segmented_inclusive_sum(&vals, part_head);
        lift_to_peer_end(&incl, peer_head)
    }

    /// Running SUM(col) (RANGE frame) in permuted order: partition-local
    /// inclusive scan of the i64 input (NULL contributes 0), then lift each
    /// peer to the peer-group-end sum.
    pub(crate) fn derive_sum(part_head: &[bool], peer_head: &[bool], values: &[i64]) -> Vec<i64> {
        let incl = segmented_inclusive_sum(values, part_head);
        lift_to_peer_end(&incl, peer_head)
    }

    /// Broadcast each peer group's LAST inclusive value onto every member of
    /// the group. Under the default RANGE frame the "current row" includes all
    /// ordering peers, so every peer reports the running aggregate through the
    /// *end* of its peer group — matching the host `compute_running_aggregate`.
    fn lift_to_peer_end(incl: &[i64], peer_head: &[bool]) -> Vec<i64> {
        let n = incl.len();
        let mut out = vec![0i64; n];
        let mut i = 0usize;
        while i < n {
            // Peer group [i, end): starts at a peer_head, runs until the next.
            let mut end = i + 1;
            while end < n && !peer_head[end] {
                end += 1;
            }
            let group_end_val = incl[end - 1];
            for slot in out.iter_mut().take(end).skip(i) {
                *slot = group_end_val;
            }
            i = end;
        }
        out
    }

    // -- Host-testable device-marshaling helpers ----------------------------

    /// Reserved `i64` value standing in for a NULL key on the single device
    /// key lane. The boundary kernel only does *equality* comparisons between
    /// adjacent permuted rows, and the host permutation places NULLs together
    /// (PARTITION/ORDER keys sort `nulls_first`), so mapping every NULL to one
    /// fixed sentinel makes two NULLs compare equal — matching the host
    /// `KeyColumn::eq_rows` rule that two NULLs share a partition / peer group.
    ///
    /// We use `i64::MIN`. A genuine key value of exactly `i64::MIN` adjacent to
    /// a NULL row would alias — but under `nulls_first` ascending key order a
    /// real `i64::MIN` sorts immediately after the NULL run, so a partition that
    /// mixes NULL and literal-`i64::MIN` keys could under-segment. This is an
    /// accepted, documented corner (the dispatch gate already restricts keys to
    /// integer-encodable dtypes; literal `i64::MIN` keys are vanishingly rare).
    /// The host path is always available as the authoritative fallback.
    pub(crate) const NULL_KEY_SENTINEL: i64 = i64::MIN;

    /// Encode one key column into the device `i64` lane in PERMUTED order,
    /// mapping NULLs to [`NULL_KEY_SENTINEL`]. `perm[i]` is the original row
    /// index visited at permuted position `i`.
    ///
    /// Boolean keys encode as 0/1, all integer-family keys pass through as
    /// `i64`; this mirrors [`KeyColumn::extract`]'s widening so the device sees
    /// the same equivalence classes the host does.
    pub(crate) fn encode_key_lane(key: &KeyColumn, perm: &[usize]) -> Vec<i64> {
        perm.iter()
            .map(|&row| match key {
                KeyColumn::Int64(v) => v[row].unwrap_or(NULL_KEY_SENTINEL),
                KeyColumn::Bool(v) => match v[row] {
                    Some(true) => 1,
                    Some(false) => 0,
                    None => NULL_KEY_SENTINEL,
                },
                // Float / Utf8 keys are gated out by `dispatch_decision`; if one
                // somehow reaches here, fall back to the sentinel so the lane
                // stays well-formed (the caller declines these shapes upstream).
                KeyColumn::Float64(_) | KeyColumn::Utf8(_) => NULL_KEY_SENTINEL,
            })
            .collect()
    }

    /// Fold the (already i64-encodable) PARTITION BY key columns into a single
    /// comparable lane in permuted order. Multiple keys are combined by hashing
    /// their per-row encodings into one `i64`; equality of the combined lane is
    /// then equivalent to equality of every component key (the kernel compares
    /// the combined lane only for *adjacent* rows, which the lexsort already
    /// grouped, so a hash collision could only mis-segment two distinct
    /// already-adjacent key tuples — accepted, with the host as fallback).
    pub(crate) fn combine_key_lanes(keys: &[KeyColumn], perm: &[usize]) -> Vec<i64> {
        let refs: Vec<&KeyColumn> = keys.iter().collect();
        combine_key_lanes_refs(&refs, perm)
    }

    /// `combine_key_lanes` over borrowed keys (lets the order axis stitch the
    /// partition keys and order keys together without cloning).
    pub(crate) fn combine_key_lanes_refs(keys: &[&KeyColumn], perm: &[usize]) -> Vec<i64> {
        if keys.is_empty() {
            // No keys on this axis: one global segment. The boundary kernel sets
            // a head only on a value change, so a constant lane yields a single
            // partition / a peer group spanning the whole input — exactly the
            // host's "no PARTITION BY" / "no ORDER BY" behaviour.
            return vec![0i64; perm.len()];
        }
        if keys.len() == 1 {
            return encode_key_lane(keys[0], perm);
        }
        let lanes: Vec<Vec<i64>> = keys.iter().map(|k| encode_key_lane(k, perm)).collect();
        (0..perm.len())
            .map(|i| {
                // FNV-1a-style fold over each key's encoded i64 for this row.
                let mut h: u64 = 0xcbf29ce484222325;
                for lane in &lanes {
                    h ^= lane[i] as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                h as i64
            })
            .collect()
    }

    /// Encode the SUM aggregate-input value column into the device `i64` lane in
    /// permuted order, mapping NULL to 0 (a NULL contributes nothing to a SUM,
    /// matching the host accumulator). Only integer inputs reach here (the gate
    /// rejects Float / non-numeric SUM inputs).
    pub(crate) fn encode_sum_values(input: &NumericColumn, perm: &[usize]) -> Vec<i64> {
        perm.iter()
            .map(|&row| match input.get(row) {
                Some(Cell::Int(x)) => x,
                // Floats are gated out; treat as 0 defensively.
                Some(Cell::Float(_)) | None => 0,
            })
            .collect()
    }

    /// Per-row non-null indicator for COUNT, in permuted order.
    pub(crate) fn encode_non_null(input: &NumericColumn, perm: &[usize]) -> Vec<bool> {
        perm.iter().map(|&row| input.get(row).is_some()).collect()
    }

    /// Invert a permutation: given `perm` (permuted-position -> original-row),
    /// produce `inv` (original-row -> permuted-position). Used to scatter the
    /// per-row results (computed in permuted order) back to original positions.
    pub(crate) fn invert_permutation(perm: &[usize]) -> Vec<usize> {
        let mut inv = vec![0usize; perm.len()];
        for (pos, &row) in perm.iter().enumerate() {
            inv[row] = pos;
        }
        inv
    }

    /// Scatter a permuted-order i64 result vector back to original row order.
    /// `permuted[i]` is the value for the row visited at permuted position `i`;
    /// the returned vec is indexed by original row position. Implemented via
    /// [`invert_permutation`] so the two share one definition of the mapping.
    pub(crate) fn scatter_to_original(permuted: &[i64], perm: &[usize]) -> Vec<i64> {
        let inv = invert_permutation(perm);
        // inv[original_row] = permuted_pos -> gather the permuted value back.
        (0..permuted.len()).map(|row| permuted[inv[row]]).collect()
    }

    /// Attempt the GPU window path. Returns `Ok(None)` to decline.
    ///
    /// Gates on the `BOLT_GPU_WINDOW=1` env override and the pure dispatch
    /// predicate ([`dispatch_decision`]); on `Decision::Accept` it runs the real
    /// device launch in [`execute_window_on_device`] — host `lexsort` for the
    /// permutation, the two `crate::jit::window_kernel` kernels (boundary flags +
    /// segmented scan) for the on-device primitive, the host derivations for the
    /// final per-row values, then scatter back to original row order.
    ///
    /// ⚠️ The device path is **compile-verified only** here: this crate is built
    /// host-side (`--features cuda-stub`, no GPU), so every launch returns an
    /// error under the stub and `BOLT_GPU_WINDOW` stays OFF by default. It must
    /// be validated on real hardware via
    /// `BOLT_GPU_WINDOW=1 cargo test --features <real-cuda> -- --ignored`
    /// (the `#[ignore = "gpu:window"]` round-trips). `BoltError::GpuCapacity`
    /// (device unavailable / OOM) is caught and turned into a clean `Ok(None)`
    /// host fallback; the derivation math is independently unit-tested host-side.
    pub(crate) fn try_execute_window_gpu(
        input: &QueryHandle,
        window_exprs: &[WindowExpr],
        partition_by: &[Expr],
        order_by: &[SortExpr],
        output_schema: &Schema,
    ) -> BoltResult<Option<QueryHandle>> {
        // Force-override env. Default OFF: unset or anything but "1" declines.
        match std::env::var(BOLT_GPU_WINDOW_ENV).as_deref() {
            Ok("1") => {}
            _ => return Ok(None),
        }

        let batch = input.record_batch();
        let n_rows = batch.num_rows();

        // Resolve key + aggregate-input dtypes for the gate. Any unresolved
        // (computed / missing) column declines cleanly.
        let mut key_dtypes = Vec::new();
        for e in partition_by.iter().chain(order_by.iter().map(|s| &s.expr)) {
            match resolve_column_dtype(batch, e) {
                Some(dt) => key_dtypes.push(dt),
                None => return Ok(None),
            }
        }
        let funcs: Vec<&WindowFunc> = window_exprs.iter().map(|w| &w.func).collect();
        let mut agg_dtypes = Vec::with_capacity(funcs.len());
        for f in &funcs {
            agg_dtypes.push(match f.arg() {
                Some(arg) => match resolve_column_dtype(batch, arg) {
                    Some(dt) => Some(dt),
                    // Literal COUNT(*) sentinel etc. — not a bare column;
                    // decline (host handles literal inputs).
                    None => return Ok(None),
                },
                None => None,
            });
        }

        let kinds = match dispatch_decision(&funcs, n_rows, &key_dtypes, &agg_dtypes) {
            Decision::Accept(kinds) => kinds,
            Decision::Decline(_reason) => return Ok(None),
        };

        // Resolve the appended output fields (one per window expr) from the tail
        // of `output_schema`, mirroring the host `execute_window`. A malformed
        // schema declines to the host path rather than erroring here.
        let total_fields = output_schema.fields.len();
        if total_fields < window_exprs.len() {
            return Ok(None);
        }
        let appended_fields = &output_schema.fields[total_fields - window_exprs.len()..];

        // Run the real device launch. Treat `GpuCapacity` (device unavailable /
        // OOM) as a clean decline so the host path takes over; surface any other
        // error to the caller (matching the engine's GPU-path error contract).
        match execute_window_on_device(
            batch,
            partition_by,
            order_by,
            &funcs,
            &kinds,
            appended_fields,
        ) {
            Ok(out) => Ok(Some(QueryHandle::from_record_batch(out))),
            Err(BoltError::GpuCapacity(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Real GPU device launch for the accepted window subset.
    ///
    /// Steps (matching the module-level algorithm doc):
    /// 1. Build the `(partition_key, order_key)` permutation via the host
    ///    `build_permutation` (`arrow::compute::lexsort_to_indices`). We use the
    ///    HOST lexsort, not `gpu_sort`: `gpu_sort`'s public API
    ///    (`sort_indices_on_gpu_multi`) is shaped around `GpuSortKey`/`GpuTable`
    ///    device columns and the engine's GPU table cache, not a freestanding
    ///    `RecordBatch`, and the module doc already notes the permutation is
    ///    identical either way. Wiring `gpu_sort` is a follow-up.
    /// 2. Encode the partition / order key columns into single comparable `i64`
    ///    lanes in permuted order, NULLs → [`NULL_KEY_SENTINEL`] (so two NULLs
    ///    group together, matching `KeyColumn::eq_rows`).
    /// 3. Upload the two key lanes; launch `bolt_window_boundary_flags` to get
    ///    `part_head` / `peer_head` (u8 flags).
    /// 4. For each function build its i64 value lane, launch
    ///    `bolt_window_segmented_scan` (single block) for the inclusive
    ///    partition-local prefix, then run the host derivation
    ///    (`derive_row_number` / `derive_rank` / `derive_dense_rank` /
    ///    `derive_count` / `derive_sum`) on `(part_head, peer_head, scan)`.
    /// 5. Scatter each result back to original row order and pack into an Arrow
    ///    array of the declared output dtype.
    fn execute_window_on_device(
        batch: &RecordBatch,
        partition_by: &[Expr],
        order_by: &[SortExpr],
        funcs: &[&WindowFunc],
        kinds: &[GpuWindowKind],
        appended_fields: &[crate::plan::logical_plan::Field],
    ) -> BoltResult<RecordBatch> {
        let n_rows = batch.num_rows();

        // (1) Permutation (host lexsort — see fn doc).
        let perm = build_permutation(batch, partition_by, order_by)?;

        // (2) Key lanes in permuted order.
        let part_keys: Vec<KeyColumn> = partition_by
            .iter()
            .map(|e| KeyColumn::extract(batch, e))
            .collect::<BoltResult<_>>()?;
        let order_keys: Vec<KeyColumn> = order_by
            .iter()
            .map(|se| KeyColumn::extract(batch, &se.expr))
            .collect::<BoltResult<_>>()?;
        let part_lane = combine_key_lanes(&part_keys, &perm);
        // The boundary kernel's `peer_head` is `part_head || order_changed`. We
        // fold the partition keys INTO the order lane (order axis = partition
        // keys ++ order keys) so a partition change is always also an order
        // change — this guarantees a peer-group break at every new partition
        // even when the ORDER BY value repeats across the boundary, matching the
        // host peer-group reset in `compute_partition`.
        let order_axis: Vec<&KeyColumn> = part_keys.iter().chain(order_keys.iter()).collect();
        let order_lane = combine_key_lanes_refs(&order_axis, &perm);

        // (3) Device launch: upload key lanes, run the boundary kernel.
        let stream = CudaStream::null_or_default();
        let n_u32 = u32::try_from(n_rows)
            .map_err(|_| BoltError::GpuCapacity(format!("window: n_rows {n_rows} exceeds u32")))?;

        let part_key_gpu = GpuVec::<i64>::from_slice(&part_lane)?;
        let order_key_gpu = GpuVec::<i64>::from_slice(&order_lane)?;
        let part_head_gpu = GpuVec::<u8>::zeros(n_rows)?;
        let peer_head_gpu = GpuVec::<u8>::zeros(n_rows)?;

        let boundary_module = module_cache::get_or_build_module(
            module_path!(),
            "window_boundary_flags".to_string(),
            None,
            compile_boundary_flag_kernel,
        )?;
        let boundary_fn = boundary_module.function(BOUNDARY_FLAG_ENTRY)?;

        // ABI order (see window_kernel::compile_boundary_flag_kernel):
        //   param_0 part_key (i64*), param_1 order_key (i64*),
        //   param_2 part_head (u8* out), param_3 peer_head (u8* out),
        //   param_4 n_rows (u32).
        let mut p_part_key = part_key_gpu.device_ptr();
        let mut p_order_key = order_key_gpu.device_ptr();
        let mut p_part_head = part_head_gpu.device_ptr();
        let mut p_peer_head = peer_head_gpu.device_ptr();
        let mut p_n = n_u32;
        let mut boundary_params: Vec<*mut c_void> = vec![
            &mut p_part_key as *mut CUdeviceptr as *mut c_void,
            &mut p_order_key as *mut CUdeviceptr as *mut c_void,
            &mut p_part_head as *mut CUdeviceptr as *mut c_void,
            &mut p_peer_head as *mut CUdeviceptr as *mut c_void,
            &mut p_n as *mut u32 as *mut c_void,
        ];

        // Single block: n_rows <= WINDOW_BLOCK_SIZE (dispatch gate), one thread
        // per row. grid_x is 1; block dim is the full WINDOW_BLOCK_SIZE so the
        // intra-block scan's stride tree is well-formed.
        let grid_x = grid_x_for(n_u32, WINDOW_BLOCK_SIZE);
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                boundary_fn.raw(),
                grid_x,
                1,
                1,
                WINDOW_BLOCK_SIZE,
                1,
                1,
                0,
                stream.raw(),
                boundary_params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        // UAF-safety: tag the launch stream into every freshly-allocated buffer
        // so a Drop while the kernel is in flight fences this stream before the
        // pool block is recycled (mirrors engine.rs's `mark_launch_stream`).
        part_key_gpu.mark_stream_use(stream.raw());
        order_key_gpu.mark_stream_use(stream.raw());
        part_head_gpu.mark_stream_use(stream.raw());
        peer_head_gpu.mark_stream_use(stream.raw());
        stream.synchronize()?;

        let part_head_u8 = part_head_gpu.to_vec()?;
        let peer_head_u8 = peer_head_gpu.to_vec()?;
        let part_head: Vec<bool> = part_head_u8.iter().map(|&b| b != 0).collect();
        let peer_head: Vec<bool> = peer_head_u8.iter().map(|&b| b != 0).collect();

        // (4) Per-function value lane → segmented scan → host derivation.
        let scan_module = module_cache::get_or_build_module(
            module_path!(),
            "window_segmented_scan".to_string(),
            None,
            compile_segmented_scan_kernel,
        )?;
        let scan_fn = scan_module.function(SEGMENTED_SCAN_ENTRY)?;

        // Pre-extract aggregate-input columns once per function.
        let mut appended: Vec<ArrayRef> = Vec::with_capacity(funcs.len());
        for ((func, kind), of) in funcs.iter().zip(kinds).zip(appended_fields) {
            // Build the i64 value lane (permuted order) the segmented scan sums.
            let values: Vec<i64> = match kind {
                GpuWindowKind::RowNumber => vec![1i64; n_rows],
                GpuWindowKind::Rank | GpuWindowKind::DenseRank => {
                    // The scan's value column for DENSE_RANK is `peer_head ? 1
                    // : 0`; RANK is derived from ROW_NUMBER + peer boundaries on
                    // the host. Both derivations consume only the boundary flags
                    // and an inclusive scan of all-1s (ROW_NUMBER), so we run the
                    // device scan on all-1s and let the host derivation do the
                    // rest. (The device primitive is the segmented inclusive sum;
                    // every rank derivation is a pure function of the flags.)
                    vec![1i64; n_rows]
                }
                GpuWindowKind::Count | GpuWindowKind::Sum => {
                    let arg = func.arg().ok_or_else(|| {
                        BoltError::Other(format!(
                            "{} window: missing aggregate input",
                            func.sql_name()
                        ))
                    })?;
                    let input = NumericColumn::extract(batch, arg)?;
                    match kind {
                        GpuWindowKind::Count => encode_non_null(&input, &perm)
                            .into_iter()
                            .map(|b| if b { 1 } else { 0 })
                            .collect(),
                        GpuWindowKind::Sum => encode_sum_values(&input, &perm),
                        _ => unreachable!(),
                    }
                }
            };

            // Launch the segmented inclusive scan over `values`, resetting at
            // every `part_head`. ABI (see compile_segmented_scan_kernel):
            //   param_0 values (i64*), param_1 seg_head (u8*),
            //   param_2 out (i64*), param_3 n_rows (u32).
            let scan_out =
                launch_segmented_scan(scan_fn.raw(), &values, &part_head_u8, n_u32, &stream)?;

            // Host derivation: turn (flags, inclusive scan) into per-row values
            // in PERMUTED order, exactly as the host-side reference math.
            let permuted_result: Vec<i64> = match kind {
                // ROW_NUMBER is the inclusive scan of all-1s itself.
                GpuWindowKind::RowNumber => scan_out,
                GpuWindowKind::Rank => derive_rank(&part_head, &peer_head),
                GpuWindowKind::DenseRank => derive_dense_rank(&part_head, &peer_head),
                GpuWindowKind::Count => {
                    let non_null: Vec<bool> = values.iter().map(|&v| v != 0).collect();
                    derive_count(&part_head, &peer_head, &non_null)
                }
                GpuWindowKind::Sum => derive_sum(&part_head, &peer_head, &values),
            };

            // (5) Scatter to original row order and pack into the declared dtype.
            let original = scatter_to_original(&permuted_result, &perm);
            appended.push(pack_i64_output(&original, of.dtype));
        }

        // Assemble: input columns ++ appended window columns.
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
        RecordBatch::try_new(schema, cols).map_err(arrow_err)
    }

    /// Upload `values` + `seg_head`, launch the single-block segmented scan, and
    /// download the inclusive partition-local prefix. `seg_head` is the u8
    /// `part_head` flag column already on the host (reused across functions).
    fn launch_segmented_scan(
        scan_fn: cuda_sys::CUfunction,
        values: &[i64],
        seg_head_u8: &[u8],
        n_u32: u32,
        stream: &CudaStream,
    ) -> BoltResult<Vec<i64>> {
        let n_rows = values.len();
        let values_gpu = GpuVec::<i64>::from_slice(values)?;
        let seg_head_gpu = GpuVec::<u8>::from_slice(seg_head_u8)?;
        let out_gpu = GpuVec::<i64>::zeros(n_rows)?;

        let mut p_values = values_gpu.device_ptr();
        let mut p_seg = seg_head_gpu.device_ptr();
        let mut p_out = out_gpu.device_ptr();
        let mut p_n = n_u32;
        let mut params: Vec<*mut c_void> = vec![
            &mut p_values as *mut CUdeviceptr as *mut c_void,
            &mut p_seg as *mut CUdeviceptr as *mut c_void,
            &mut p_out as *mut CUdeviceptr as *mut c_void,
            &mut p_n as *mut u32 as *mut c_void,
        ];

        // Single block, full WINDOW_BLOCK_SIZE threads (the intra-block
        // Hillis-Steele scan's static shared buffer is sized for the full
        // block, so `sharedMemBytes` stays 0 — the kernel declares `.shared`).
        let grid_x = grid_x_for(n_u32, WINDOW_BLOCK_SIZE);
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                scan_fn,
                grid_x,
                1,
                1,
                WINDOW_BLOCK_SIZE,
                1,
                1,
                0,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        values_gpu.mark_stream_use(stream.raw());
        seg_head_gpu.mark_stream_use(stream.raw());
        out_gpu.mark_stream_use(stream.raw());
        stream.synchronize()?;
        out_gpu.to_vec()
    }

    /// Pack a per-row `i64` result (in original row order) into an Arrow array
    /// of the declared window output dtype. Window outputs from this GPU subset
    /// (ROW_NUMBER / RANK / DENSE_RANK / COUNT / integer SUM) are always
    /// non-NULL integers; we narrow to Int32 when the plan declared it.
    fn pack_i64_output(values: &[i64], dtype: DataType) -> ArrayRef {
        use arrow_array::Int32Array;
        match dtype {
            DataType::Int32 => Arc::new(Int32Array::from_iter(
                values.iter().map(|&v| Some(v as i32)),
            )),
            _ => Arc::new(Int64Array::from_iter(values.iter().map(|&v| Some(v)))),
        }
    }

    /// Resolve a bare-column (optionally aliased) expression to its Arrow
    /// dtype in `batch`, or `None` for computed expressions / missing columns.
    fn resolve_column_dtype(batch: &RecordBatch, e: &Expr) -> Option<arrow_schema::DataType> {
        let name = match bare_column_name(e) {
            Ok(n) => n,
            Err(_) => return None,
        };
        let idx = batch.schema().index_of(&name).ok()?;
        Some(batch.column(idx).data_type().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{Field as PField, Schema as PSchema};
    use arrow_array::{Int32Array, Int64Array, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field, Schema as AwSchema};

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
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            })
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
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(arr.value(i))
                }
            })
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
            let a = out
                .column(idx)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
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
            assert_eq!(
                rn[i],
                Some(expected),
                "row {i} (k={}, v={:?})",
                ks[i],
                vs[i]
            );
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
        let va = out
            .column(idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
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
        let os = out_schema(&[("v", DataType::Int32)], &[("rs", DataType::Int64)]);
        let out = execute_window(h, &wexprs, &[], &[order("v")], &os)
            .unwrap()
            .into_record_batch();

        let idx = out.schema().index_of("v").unwrap();
        let va = out
            .column(idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
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
        let h =
            QueryHandle::from_record_batch(RecordBatch::try_new(schema, vec![kcol, vcol]).unwrap());
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

    /// Build a handle from a single Int64 column.
    fn i64_col(name: &str, values: Vec<Option<i64>>) -> (ArrowField, ArrayRef) {
        (
            ArrowField::new(name, ArrowDataType::Int64, true),
            Arc::new(Int64Array::from(values)) as ArrayRef,
        )
    }

    /// Build a handle from a single Float64 column.
    fn f64_col(name: &str, values: Vec<Option<f64>>) -> (ArrowField, ArrayRef) {
        (
            ArrowField::new(name, ArrowDataType::Float64, true),
            Arc::new(Float64Array::from(values)) as ArrayRef,
        )
    }

    /// BUG 1: SUM(Int64) over a value > 2^53 must stay EXACT (no f64 round
    /// trip). 9_007_199_254_740_993 == 2^53 + 1 is the smallest integer f64
    /// cannot represent; folding it as f64 would round to 2^53.
    #[test]
    fn sum_int64_above_2_53_is_exact() {
        let big = 9_007_199_254_740_993_i64; // 2^53 + 1
        let h = handle(vec![i64_col("v", vec![Some(big), Some(1)])]);
        let wexprs = vec![WindowExpr {
            func: WindowFunc::Sum(col("v")),
            output_name: "s".into(),
        }];
        // No ORDER BY -> whole partition is one peer group; every row sees the
        // full sum.
        let os = out_schema(&[("v", DataType::Int64)], &[("s", DataType::Int64)]);
        let out = execute_window(h, &wexprs, &[], &[], &os)
            .unwrap()
            .into_record_batch();
        let s = as_i64(&out, "s");
        assert_eq!(s, vec![Some(big + 1), Some(big + 1)]);
    }

    /// BUG 1: MIN/MAX(Int64) must pass the EXACT i64 through. With a value
    /// beyond 2^53, the old f64 path would have returned a neighbour that was
    /// never in the input.
    #[test]
    fn min_max_int64_above_2_53_are_exact() {
        let big = 9_007_199_254_740_993_i64; // 2^53 + 1, not representable in f64
        let h = handle(vec![i64_col("v", vec![Some(big), Some(big + 2)])]);
        let wexprs = vec![
            WindowExpr {
                func: WindowFunc::Min(col("v")),
                output_name: "mn".into(),
            },
            WindowExpr {
                func: WindowFunc::Max(col("v")),
                output_name: "mx".into(),
            },
        ];
        let os = out_schema(
            &[("v", DataType::Int64)],
            &[("mn", DataType::Int64), ("mx", DataType::Int64)],
        );
        let out = execute_window(h, &wexprs, &[], &[], &os)
            .unwrap()
            .into_record_batch();
        let mn = as_i64(&out, "mn");
        let mx = as_i64(&out, "mx");
        assert_eq!(mn, vec![Some(big), Some(big)]);
        assert_eq!(mx, vec![Some(big + 2), Some(big + 2)]);
    }

    /// BUG 1: integer SUM overflow must ERROR (BoltError::Type), never wrap —
    /// mirroring the `SUM(integer) overflow` contract in aggregate.rs.
    #[test]
    fn sum_int64_overflow_errors() {
        let h = handle(vec![i64_col("v", vec![Some(i64::MAX), Some(1)])]);
        let wexprs = vec![WindowExpr {
            func: WindowFunc::Sum(col("v")),
            output_name: "s".into(),
        }];
        let os = out_schema(&[("v", DataType::Int64)], &[("s", DataType::Int64)]);
        // `QueryHandle` (the Ok variant) does not implement `Debug`, so we
        // match on the `Result` rather than using `.unwrap_err()`.
        let err = match execute_window(h, &wexprs, &[], &[], &os) {
            Ok(_) => panic!("expected SUM(integer) overflow error, got Ok"),
            Err(e) => e,
        };
        match err {
            BoltError::Type(msg) => {
                assert!(
                    msg.contains("SUM(integer) overflow"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected BoltError::Type on SUM overflow, got {other:?}"),
        }
    }

    /// BUG 2: float MIN/MAX with a leading NaN follow the DuckDB convention
    /// (NaN sorts as the largest value): MIN skips the NaN and returns the
    /// real minimum; MAX returns NaN because one is present.
    #[test]
    fn float_min_max_nan_convention() {
        // Leading NaN used to seed `extreme` and stick under `<`/`>`.
        let h = handle(vec![f64_col(
            "v",
            vec![Some(f64::NAN), Some(2.0), Some(-1.0)],
        )]);
        let wexprs = vec![
            WindowExpr {
                func: WindowFunc::Min(col("v")),
                output_name: "mn".into(),
            },
            WindowExpr {
                func: WindowFunc::Max(col("v")),
                output_name: "mx".into(),
            },
        ];
        let os = out_schema(
            &[("v", DataType::Float64)],
            &[("mn", DataType::Float64), ("mx", DataType::Float64)],
        );
        let out = execute_window(h, &wexprs, &[], &[], &os)
            .unwrap()
            .into_record_batch();
        let mn = as_f64(&out, "mn");
        let mx = as_f64(&out, "mx");
        // MIN skips NaN -> -1.0 on every row (whole partition, one peer group).
        assert_eq!(mn, vec![Some(-1.0), Some(-1.0), Some(-1.0)]);
        // MAX returns NaN (NaN is the largest under the convention).
        for cell in &mx {
            assert!(cell.unwrap().is_nan(), "expected NaN MAX, got {cell:?}");
        }
    }

    /// BUG 2: all-NaN float MIN returns NaN (there is no real value to prefer).
    #[test]
    fn float_min_all_nan_is_nan() {
        let h = handle(vec![f64_col("v", vec![Some(f64::NAN), Some(f64::NAN)])]);
        let wexprs = vec![WindowExpr {
            func: WindowFunc::Min(col("v")),
            output_name: "mn".into(),
        }];
        let os = out_schema(&[("v", DataType::Float64)], &[("mn", DataType::Float64)]);
        let out = execute_window(h, &wexprs, &[], &[], &os)
            .unwrap()
            .into_record_batch();
        let mn = as_f64(&out, "mn");
        for cell in &mn {
            assert!(cell.unwrap().is_nan(), "expected NaN MIN, got {cell:?}");
        }
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

    // -----------------------------------------------------------------------
    // GPU window path: host-side tests of the dispatch gate + the
    // segmented-scan derivations (the device-equivalent reference math).
    // -----------------------------------------------------------------------
    mod gpu_path {
        use super::*;
        use crate::exec::window::gpu::{
            combine_key_lanes, derive_count, derive_dense_rank, derive_rank, derive_row_number,
            derive_sum, dispatch_decision, encode_key_lane, encode_non_null, encode_sum_values,
            invert_permutation, key_dtype_is_i64_encodable, scatter_to_original,
            segmented_inclusive_sum, Decision, GpuWindowKind, NULL_KEY_SENTINEL,
        };
        use crate::exec::window::{KeyColumn, NumericColumn};
        use arrow_schema::DataType as A;

        #[test]
        fn segmented_sum_resets_at_each_head() {
            // Two segments: [1,2,3] then [10,20].
            let vals = [1, 2, 3, 10, 20];
            let head = [true, false, false, true, false];
            assert_eq!(segmented_inclusive_sum(&vals, &head), vec![1, 3, 6, 10, 30]);
        }

        #[test]
        fn row_number_is_partition_local_index() {
            // Partition heads at 0 and 3 -> [1,2,3] then [1,2].
            let part_head = [true, false, false, true, false];
            assert_eq!(derive_row_number(&part_head), vec![1, 2, 3, 1, 2]);
        }

        #[test]
        fn dense_rank_counts_distinct_peer_groups() {
            // Single partition; peer groups start at 0, 2, 3.
            // DENSE_RANK: 1,1,2,3
            let part_head = [true, false, false, false];
            let peer_head = [true, false, true, true];
            assert_eq!(derive_dense_rank(&part_head, &peer_head), vec![1, 1, 2, 3]);
        }

        #[test]
        fn rank_shares_lowest_and_skips() {
            // Single partition, ORDER BY peers: {0,1}, {2}, {3}.
            // RANK: 1,1,3,4
            let part_head = [true, false, false, false];
            let peer_head = [true, false, true, true];
            assert_eq!(derive_rank(&part_head, &peer_head), vec![1, 1, 3, 4]);
        }

        #[test]
        fn rank_resets_per_partition() {
            // Two partitions; partition 2 starts at idx 3 (also a peer head).
            // P1 peers {0,1},{2}; P2 peers {3,4}.
            // RANK: 1,1,3 | 1,1
            let part_head = [true, false, false, true, false];
            let peer_head = [true, false, true, true, false];
            assert_eq!(derive_rank(&part_head, &peer_head), vec![1, 1, 3, 1, 1]);
        }

        #[test]
        fn running_count_lifts_peers_to_group_end() {
            // Single partition, all non-null, peers {0,1},{2}.
            // Inclusive count: 1,2,3 ; lifted to peer end: 2,2,3.
            let part_head = [true, false, false];
            let peer_head = [true, false, true];
            let non_null = [true, true, true];
            assert_eq!(
                derive_count(&part_head, &peer_head, &non_null),
                vec![2, 2, 3]
            );
        }

        #[test]
        fn running_count_skips_nulls() {
            // non_null = [t, f, t]; single peer group -> count 2 on every row.
            let part_head = [true, false, false];
            let peer_head = [true, false, false];
            let non_null = [true, false, true];
            assert_eq!(
                derive_count(&part_head, &peer_head, &non_null),
                vec![2, 2, 2]
            );
        }

        #[test]
        fn running_sum_range_frame_peers_share_value() {
            // Peers {0,1} have values 10,10 -> both see 20 (RANGE);
            // {2} value 20 -> sees 40. Matches the host `running_sum` test.
            let part_head = [true, false, false];
            let peer_head = [true, false, true];
            let values = [10, 10, 20];
            assert_eq!(
                derive_sum(&part_head, &peer_head, &values),
                vec![20, 20, 40]
            );
        }

        #[test]
        fn running_sum_null_contributes_zero() {
            // NULL pre-encoded as 0 by the host. Single peer group -> sum 30.
            let part_head = [true, false, false];
            let peer_head = [true, false, false];
            let values = [10, 0, 20];
            assert_eq!(
                derive_sum(&part_head, &peer_head, &values),
                vec![30, 30, 30]
            );
        }

        #[test]
        fn dispatch_accepts_supported_kinds() {
            let funcs = [
                WindowFunc::RowNumber,
                WindowFunc::Rank,
                WindowFunc::DenseRank,
                WindowFunc::Count(col("v")),
                WindowFunc::Sum(col("v")),
            ];
            let frefs: Vec<&WindowFunc> = funcs.iter().collect();
            let agg = [None, None, None, Some(A::Int64), Some(A::Int32)];
            let dec = dispatch_decision(&frefs, 8, &[A::Int32, A::Int64], &agg);
            assert_eq!(
                dec,
                Decision::Accept(vec![
                    GpuWindowKind::RowNumber,
                    GpuWindowKind::Rank,
                    GpuWindowKind::DenseRank,
                    GpuWindowKind::Count,
                    GpuWindowKind::Sum,
                ])
            );
        }

        #[test]
        fn dispatch_declines_avg_min_max() {
            for f in [
                WindowFunc::Avg(col("v")),
                WindowFunc::Min(col("v")),
                WindowFunc::Max(col("v")),
            ] {
                let frefs = [&f];
                let dec = dispatch_decision(&frefs, 4, &[A::Int64], &[Some(A::Int64)]);
                assert!(matches!(dec, Decision::Decline(_)), "should decline {f:?}");
            }
        }

        #[test]
        fn dispatch_declines_float_keys_and_utf8() {
            let f = WindowFunc::RowNumber;
            let frefs = [&f];
            assert!(matches!(
                dispatch_decision(&frefs, 4, &[A::Float64], &[None]),
                Decision::Decline(_)
            ));
            assert!(matches!(
                dispatch_decision(&frefs, 4, &[A::Utf8], &[None]),
                Decision::Decline(_)
            ));
            assert!(key_dtype_is_i64_encodable(&A::Int32));
            assert!(key_dtype_is_i64_encodable(&A::Date32));
            assert!(!key_dtype_is_i64_encodable(&A::Float32));
            assert!(!key_dtype_is_i64_encodable(&A::Utf8));
        }

        #[test]
        fn dispatch_declines_float_agg_input() {
            let f = WindowFunc::Sum(col("v"));
            let frefs = [&f];
            assert!(matches!(
                dispatch_decision(&frefs, 4, &[A::Int64], &[Some(A::Float64)]),
                Decision::Decline(_)
            ));
        }

        #[test]
        fn dispatch_declines_oversized_or_empty_input() {
            let f = WindowFunc::RowNumber;
            let frefs = [&f];
            // Empty -> decline (host fast-path owns it).
            assert!(matches!(
                dispatch_decision(&frefs, 0, &[A::Int64], &[None]),
                Decision::Decline(_)
            ));
            // > one block -> decline (multi-block scan deferred).
            let too_big = (crate::jit::window_kernel::WINDOW_BLOCK_SIZE as usize) + 1;
            assert!(matches!(
                dispatch_decision(&frefs, too_big, &[A::Int64], &[None]),
                Decision::Decline(_)
            ));
        }

        /// Device round-trip: ROW_NUMBER over a partitioned/ordered input must
        /// equal the host result. Requires a GPU and `BOLT_GPU_WINDOW=1`;
        /// ignored in CI (no device).
        #[test]
        #[ignore = "gpu:window"]
        fn gpu_row_number_matches_host_roundtrip() {
            std::env::set_var(super::super::gpu::BOLT_GPU_WINDOW_ENV, "1");
            let h = handle(vec![
                int_col("k", vec![1, 1, 2, 1, 2]),
                int_col("v", vec![30, 10, 20, 20, 10]),
            ]);
            let wexprs = vec![WindowExpr {
                func: WindowFunc::RowNumber,
                output_name: "rn".into(),
            }];
            let os = out_schema(
                &[("k", DataType::Int32), ("v", DataType::Int32)],
                &[("rn", DataType::Int64)],
            );
            let out = execute_window(h, &wexprs, &[col("k")], &[order("v")], &os)
                .unwrap()
                .into_record_batch();
            // Partition k=1 ordered by v: 10,20,30 -> rn 1,2,3
            // Partition k=2 ordered by v: 10,20    -> rn 1,2
            let rn = as_i64(&out, "rn");
            assert_eq!(rn.len(), 5);
            std::env::remove_var(super::super::gpu::BOLT_GPU_WINDOW_ENV);
        }

        // -------------------------------------------------------------------
        // Host-only unit tests for the device-marshaling helpers (run under
        // `--features cuda-stub`; no GPU touched).
        // -------------------------------------------------------------------

        #[test]
        fn encode_key_lane_maps_null_to_sentinel() {
            // Int64 key with a NULL in the middle; identity permutation.
            let key = KeyColumn::Int64(vec![Some(7), None, Some(-3)]);
            let perm = [0usize, 1, 2];
            assert_eq!(encode_key_lane(&key, &perm), vec![7, NULL_KEY_SENTINEL, -3]);
            // Bool encodes false=0/true=1, NULL=sentinel.
            let bkey = KeyColumn::Bool(vec![Some(false), Some(true), None]);
            assert_eq!(encode_key_lane(&bkey, &perm), vec![0, 1, NULL_KEY_SENTINEL]);
        }

        #[test]
        fn encode_key_lane_follows_permutation() {
            let key = KeyColumn::Int64(vec![Some(10), Some(20), Some(30)]);
            // Visit rows in reverse order.
            let perm = [2usize, 1, 0];
            assert_eq!(encode_key_lane(&key, &perm), vec![30, 20, 10]);
        }

        #[test]
        fn combine_key_lanes_empty_is_single_segment() {
            // No keys -> constant lane (one global segment).
            let lane = combine_key_lanes(&[], &[0usize, 1, 2, 3]);
            assert_eq!(lane, vec![0, 0, 0, 0]);
        }

        #[test]
        fn combine_key_lanes_single_passes_through() {
            let key = KeyColumn::Int64(vec![Some(1), Some(1), Some(2)]);
            let perm = [0usize, 1, 2];
            // Single key: combined lane equals the encoded lane (so equality of
            // adjacent rows is preserved exactly).
            assert_eq!(combine_key_lanes(&[key], &perm), vec![1, 1, 2]);
        }

        #[test]
        fn combine_key_lanes_multi_equal_iff_all_components_equal() {
            // Two key columns; rows 0 and 1 share both keys, row 2 differs.
            let k1 = KeyColumn::Int64(vec![Some(5), Some(5), Some(5)]);
            let k2 = KeyColumn::Int64(vec![Some(9), Some(9), Some(8)]);
            let perm = [0usize, 1, 2];
            let lane = combine_key_lanes(&[k1, k2], &perm);
            assert_eq!(lane[0], lane[1], "equal tuples must hash equal");
            assert_ne!(lane[1], lane[2], "differing tuple must hash differently");
        }

        #[test]
        fn encode_sum_values_maps_null_to_zero() {
            let input = NumericColumn::Int(vec![Some(3), None, Some(-4)]);
            let perm = [0usize, 1, 2];
            assert_eq!(encode_sum_values(&input, &perm), vec![3, 0, -4]);
        }

        #[test]
        fn encode_non_null_indicator() {
            let input = NumericColumn::Int(vec![Some(3), None, Some(-4)]);
            let perm = [0usize, 1, 2];
            assert_eq!(encode_non_null(&input, &perm), vec![true, false, true]);
        }

        #[test]
        fn invert_permutation_round_trips() {
            // perm[pos] = original_row; inv[original_row] = pos.
            let perm = [2usize, 0, 3, 1];
            let inv = invert_permutation(&perm);
            assert_eq!(inv, vec![1, 3, 0, 2]);
            // Composing perm then inv recovers identity.
            for (pos, &row) in perm.iter().enumerate() {
                assert_eq!(inv[row], pos);
            }
        }

        #[test]
        fn scatter_to_original_places_by_perm() {
            // Permuted-order results; perm maps permuted pos -> original row.
            // perm = [2,0,1] means: permuted pos 0 is original row 2, etc.
            let permuted = [100i64, 200, 300];
            let perm = [2usize, 0, 1];
            // original[2]=100, original[0]=200, original[1]=300.
            assert_eq!(scatter_to_original(&permuted, &perm), vec![200, 300, 100]);
        }

        #[test]
        fn scatter_is_inverse_of_gather_by_perm() {
            // Build a permuted vector by gathering original through perm, then
            // scatter back and confirm we recover the original.
            let original = [11i64, 22, 33, 44];
            let perm = [3usize, 1, 0, 2];
            let permuted: Vec<i64> = perm.iter().map(|&r| original[r]).collect();
            assert_eq!(scatter_to_original(&permuted, &perm), original.to_vec());
        }

        // -------------------------------------------------------------------
        // GPU round-trips for RANK / DENSE_RANK / SUM / COUNT. Each asserts the
        // GPU output equals the host path's output on the same input. Requires a
        // GPU and `BOLT_GPU_WINDOW=1`; ignored in CI (no device).
        // -------------------------------------------------------------------

        #[test]
        #[ignore = "gpu:window"]
        fn gpu_rank_matches_host_roundtrip() {
            std::env::set_var(super::super::gpu::BOLT_GPU_WINDOW_ENV, "1");
            // Single partition (k all 1), ORDER BY v with a tie at v=10.
            // RANK: 1,1,3,4 ; DENSE_RANK: 1,1,2,3 (per the host `rank_with_ties`).
            let h = handle(vec![
                int_col("k", vec![1, 1, 1, 1]),
                int_col("v", vec![10, 30, 10, 20]),
            ]);
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
                &[("k", DataType::Int32), ("v", DataType::Int32)],
                &[("rk", DataType::Int64), ("dr", DataType::Int64)],
            );
            let out = execute_window(h, &wexprs, &[col("k")], &[order("v")], &os)
                .unwrap()
                .into_record_batch();
            let idx = out.schema().index_of("v").unwrap();
            let va = out
                .column(idx)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
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
                assert_eq!(rk[i], Some(erk), "gpu rank row {i} v={}", vs[i]);
                assert_eq!(dr[i], Some(edr), "gpu dense_rank row {i} v={}", vs[i]);
            }
            std::env::remove_var(super::super::gpu::BOLT_GPU_WINDOW_ENV);
        }

        #[test]
        #[ignore = "gpu:window"]
        fn gpu_sum_matches_host_roundtrip() {
            std::env::set_var(super::super::gpu::BOLT_GPU_WINDOW_ENV, "1");
            // Single partition, ORDER BY v: rows v=10,10,20 (RANGE frame).
            // The two v=10 peers both see 20; the v=20 row sees 40 (matches the
            // host `running_sum_with_order_and_ties`).
            let h = handle(vec![
                int_col("k", vec![1, 1, 1]),
                int_col("v", vec![20, 10, 10]),
            ]);
            let wexprs = vec![WindowExpr {
                func: WindowFunc::Sum(col("v")),
                output_name: "rs".into(),
            }];
            let os = out_schema(
                &[("k", DataType::Int32), ("v", DataType::Int32)],
                &[("rs", DataType::Int64)],
            );
            let out = execute_window(h, &wexprs, &[col("k")], &[order("v")], &os)
                .unwrap()
                .into_record_batch();
            let idx = out.schema().index_of("v").unwrap();
            let va = out
                .column(idx)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let vs: Vec<i32> = (0..va.len()).map(|i| va.value(i)).collect();
            let rs = as_i64(&out, "rs");
            for i in 0..vs.len() {
                let expected = match vs[i] {
                    10 => 20,
                    20 => 40,
                    _ => unreachable!(),
                };
                assert_eq!(rs[i], Some(expected), "gpu sum row {i} v={}", vs[i]);
            }
            std::env::remove_var(super::super::gpu::BOLT_GPU_WINDOW_ENV);
        }

        #[test]
        #[ignore = "gpu:window"]
        fn gpu_count_matches_host_roundtrip() {
            std::env::set_var(super::super::gpu::BOLT_GPU_WINDOW_ENV, "1");
            // Single partition (k=1), no ORDER BY -> whole partition is one peer
            // group; COUNT(v) skips the NULL -> 2 on every row (matches the host
            // `count_over_partition_skips_nulls`).
            let schema = Arc::new(AwSchema::new(vec![
                Field::new("k", ArrowDataType::Int64, true),
                Field::new("v", ArrowDataType::Int64, true),
            ]));
            let kcol = Arc::new(Int64Array::from(vec![1i64, 1, 1])) as ArrayRef;
            let vcol = Arc::new(Int64Array::from(vec![Some(1), None, Some(3)])) as ArrayRef;
            let h = QueryHandle::from_record_batch(
                RecordBatch::try_new(schema, vec![kcol, vcol]).unwrap(),
            );
            let wexprs = vec![WindowExpr {
                func: WindowFunc::Count(col("v")),
                output_name: "cnt".into(),
            }];
            let os = out_schema(
                &[("k", DataType::Int64), ("v", DataType::Int64)],
                &[("cnt", DataType::Int64)],
            );
            let out = execute_window(h, &wexprs, &[col("k")], &[], &os)
                .unwrap()
                .into_record_batch();
            let cnt = as_i64(&out, "cnt");
            assert_eq!(cnt, vec![Some(2), Some(2), Some(2)]);
            std::env::remove_var(super::super::gpu::BOLT_GPU_WINDOW_ENV);
        }
    }
}
