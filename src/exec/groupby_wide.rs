// SPDX-License-Identifier: AGPL-3.0-only

//! GROUP BY execution for "wide" composite keys — those whose total bit
//! width exceeds 64 (e.g. `GROUP BY (a, b)` where both are `Int64`, or any
//! GROUP BY with three or more columns).
//!
//! # Why host-side reduction in v1?
//!
//! The narrow path in [`crate::exec::groupby`] losslessly packs each key
//! tuple into a single `i64`, lets the GPU build a per-slot open-addressing
//! hash table, and assembles results from the resulting tables. That design
//! relies on a bijection between the tuple and the `i64` key, which doesn't
//! exist once the tuple has more than 64 bits of payload.
//!
//! The "proper" GPU path would (a) host-side hash each tuple to an `i64`,
//! (b) use the existing GPU kernels (which don't care that the i64 is now a
//! probabilistic hash rather than a real key), then (c) walk input rows on
//! the host to detect hash collisions and split the GPU's per-slot results
//! back onto the real distinct tuples. This is complex and easy to get
//! wrong; we defer it.
//!
//! For v1 we just do the entire reduction on the host. A few-million-row
//! GROUP BY in plain Rust takes tens of milliseconds, which is acceptable
//! as a fallback. The public API ([`execute_groupby_wide`]) matches the
//! narrow path so the dispatcher in `groupby.rs` can call us directly.
//!
//! # Accumulator output dtype table
//!
//! | aggregate | input dtype | accumulator | output dtype |
//! |-----------|-------------|-------------|--------------|
//! | `SUM`     | Int32       | `i64`       | Int64        |
//! | `SUM`     | Int64       | `i64`       | Int64        |
//! | `SUM`     | Float32     | `f64`       | Float32 (cast) |
//! | `SUM`     | Float64     | `f64`       | Float64      |
//! | `MIN`/`MAX` | Int32     | `i32`       | Int32        |
//! | `MIN`/`MAX` | Int64     | `i64`       | Int64        |
//! | `MIN`/`MAX` | Float32   | `f32`       | Float32      |
//! | `MIN`/`MAX` | Float64   | `f64`       | Float64      |
//! | `COUNT`   | (any)       | `u64`       | Int64        |
//! | `AVG`     | numeric     | (`f64`, `u64`) | Float64   |
//!
//! `SUM` over `Int32` widens to `i64` internally **and on output**: the plan-
//! declared output dtype is `Int64` (see
//! [`crate::plan::logical_plan::sum_output_dtype`], the single source of truth
//! for this widening rule, and
//! [`crate::jit::agg_kernels::reduction_output_dtype`] which mirrors it for
//! kernel emission). We never narrow back to `Int32`. `SUM` over `Float32`
//! still accumulates in `f64` and is cast back to `Float32` on the way out —
//! matching the scalar reducer and the narrow GROUP BY path, which both keep
//! float output dtype unchanged.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array,
    Int32Array, Int64Array, RecordBatch, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};
#[cfg(test)]
use arrow_schema::{DataType as ArrowDataType, Schema as ArrowSchema};

use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_common::{
    arrow_dtype_to_plan, bare_column_name, build_output_arrow_schema, canonicalise_f32,
    canonicalise_f64, downcast_err, timestamp_array_from_i64,
};
#[cfg(test)]
use crate::plan::logical_plan::Schema;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Literal, TimeUnit};
use crate::plan::physical_plan::{AggregateSpec, PhysicalPlan};

// ---------------------------------------------------------------------------
// Public entry point.
// ---------------------------------------------------------------------------

/// Execute a GROUP BY plan with wide (>64 bit) keys using host-side reduction.
///
/// `plan` must be a [`PhysicalPlan::Aggregate`] with `pre = None`, a
/// non-empty `group_by`, and primitive (Int32/Int64/Float32/Float64) key and
/// aggregate-input columns. Bool/Utf8 are rejected; non-bare-column
/// aggregate inputs (`SUM(a * b)`) are rejected.
///
/// The narrow single-i64 path lives in [`crate::exec::groupby`]; this
/// fallback handles inputs that path can't pack losslessly. It is also
/// safe to call on narrow inputs — it just gives up GPU parallelism.
// Stage 3 note: this executor is host-side only — no GPU launches, no
// device buffers, no streams. The "GPU output collection" mentioned by
// the Stage-3 spec doesn't apply because the entire reduction runs on
// the host. When this path is replaced by a GPU-side wide-key hash
// table (a 0.3 goal), the new implementation should pick up the same
// async memcpy + pinned D2H pattern used in `groupby.rs` /
// `groupby_valid.rs`. Until then there's nothing to async-ify.
pub fn execute_groupby_wide(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> BoltResult<RecordBatch> {
    // 1. Validate the plan shape.
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        other => {
            return Err(BoltError::Other(format!(
                "execute_groupby_wide: expected Aggregate plan, got {:?}",
                std::mem::discriminant(other)
            )))
        }
    };
    if pre.is_some() {
        return Err(BoltError::Other(
            "wide GROUP BY with pre kernel not yet supported".into(),
        ));
    }
    if aggregate.group_by.is_empty() {
        return Err(BoltError::Other(
            "execute_groupby_wide: aggregate has no GROUP BY columns".into(),
        ));
    }

    // 2. Resolve every group-by column to a (ColumnIO, Arrow array) pair and
    //    validate dtypes up-front.
    let key_cols = resolve_key_columns(aggregate, table_batch)?;

    // 3. Resolve every aggregate to its input column (bare-column-ref only)
    //    plus the per-aggregate accumulator/output metadata.
    let agg_plan = resolve_aggregates(aggregate, table_batch)?;

    // 4. Walk the input rows once, building per-tuple accumulators.
    //
    //    Row count comes from the first key column. We've already validated
    //    that every key column shares the same Arrow array length implicitly
    //    by reading them out of the same RecordBatch, but be defensive: the
    //    RecordBatch contract is "all columns same length", so we just trust
    //    `num_rows`.
    let n_rows = table_batch.num_rows();
    // The std default HashMap uses SipHash — overkill here since we have no
    // adversarial input, but plenty fast for our v1 workload (tens of ms on
    // a few-million rows). Swapping in a faster hasher is a future tweak.
    let mut groups: HashMap<TupleKey, Vec<Accumulator>> = HashMap::new();
    // Precompute one fresh accumulator vector so we can clone it on first
    // insert per group rather than re-deriving the dtype matrix each time.
    let fresh_accs = make_initial_accumulators(&agg_plan)?;
    // Reusable scratch buffer for the per-row tuple build.
    let mut buf_key: Vec<Option<KeyValue>> = Vec::with_capacity(key_cols.len());
    // H1 NULL handling: precompute "any key column has nulls?" and "any
    // aggregate input has nulls?" once so the inner row loop stays
    // branch-light when both are null-free (the common case).
    //
    // SQL-standard NULL grouping:
    //   - NULL in any group-by key column becomes `None` at that tuple slot,
    //     so `(1, NULL)`, `(NULL, 1)`, and `(NULL, NULL)` each form their own
    //     distinct group (`NULL IS NOT DISTINCT FROM NULL` per position).
    //   - Rows feed each aggregate independently: an aggregate's update is
    //     skipped only when that aggregate's own input is NULL.
    let any_key_nullable = key_cols.iter().any(|kc| kc.arr.null_count() > 0);
    let any_agg_nullable = agg_plan
        .iter()
        .any(|p| p.arr.is_some_and(|a| a.null_count() > 0));
    for row in 0..n_rows {
        buf_key.clear();
        for kc in &key_cols {
            if any_key_nullable && kc.arr.is_null(row) {
                buf_key.push(None);
            } else {
                buf_key.push(Some(key_value_at(kc, row)?));
            }
        }
        let tuple = TupleKey(buf_key.clone());

        // Look up (or insert) the per-group accumulator vector. `entry`
        // gives us one borrow of `groups` for the whole branch, avoiding
        // the double-borrow you'd hit with a manual `get_mut` + `insert`.
        let accs = groups.entry(tuple).or_insert_with(|| fresh_accs.clone());

        // Feed each aggregate its row value, skipping per-aggregate inputs
        // that are NULL at this row. Mirrors the call-site filter in
        // `groupby.rs::run_typed_agg` (which drops NULL positions before
        // GPU upload). Since the wide path is host-side per-row, we do the
        // same thing inline:
        //   - SUM/MIN/MAX: skip update so NULLs don't contribute.
        //   - COUNT(col):  skip update so we report the non-NULL count.
        //   - AVG:         skip update so the denominator is non-NULL count
        //                  and the numerator excludes NULL contributions.
        for (i, agg_in) in agg_plan.iter().enumerate() {
            if any_agg_nullable && agg_in.arr.is_some_and(|a| a.is_null(row)) {
                continue;
            }
            let value = agg_input_at(agg_in, row)?;
            accs[i].update(value)?;
        }
    }

    // 5. Sort groups by tuple for deterministic ordering.
    let mut sorted: Vec<(TupleKey, Vec<Accumulator>)> = groups.into_iter().collect();
    sorted.sort_by(|a, b| cmp_tuple(&a.0, &b.0));

    // 6. Build the output RecordBatch.
    let m_keys = key_cols.len();
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(m_keys + aggregate.aggregates.len());

    // 6a. Key columns: one per group-by column, in group_by order.
    let key_arrays = build_key_arrays(&sorted, &key_cols)?;
    for arr in key_arrays {
        arrays.push(arr);
    }

    // 6b. Aggregate columns: each accumulator finalised into the
    //     plan-declared output dtype.
    for (i, agg) in aggregate.aggregates.iter().enumerate() {
        let out_field = aggregate
            .output_schema
            .fields
            .get(m_keys + i)
            .ok_or_else(|| {
                BoltError::Other(format!(
                    "execute_groupby_wide: output_schema missing field for aggregate index {}",
                    i
                ))
            })?;
        let arr = finalize_agg_column(agg, out_field, i, &sorted)?;
        arrays.push(arr);
    }

    let arrow_schema = build_output_arrow_schema(&aggregate.output_schema, &arrays)?;
    RecordBatch::try_new(arrow_schema, arrays)
        .map_err(|e| BoltError::Other(format!("failed to build wide GROUP BY RecordBatch: {e}")))
}

// ---------------------------------------------------------------------------
// Key column metadata + per-row tuple extraction.
// ---------------------------------------------------------------------------

/// A group-by key column resolved against the input `RecordBatch`. We hold a
/// borrowed `ArrayRef` so per-row access doesn't re-traverse the schema.
struct KeyColumn<'a> {
    /// Column name (used for output field naming).
    name: String,
    /// Plan-declared dtype (validated against the Arrow array on construction).
    dtype: DataType,
    /// The Arrow array. The outer slice lives for `'a` (the input batch's
    /// borrow); we re-downcast per-row to keep the storage simple.
    arr: &'a dyn Array,
}

/// One value in a [`TupleKey`]. Floats are stored as bit patterns so the
/// derived `Eq`/`Hash` is total (NaN equals NaN if they share a bit pattern;
/// `-0.0` and `+0.0` group separately, matching the narrow path).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum KeyValue {
    /// 32-bit signed int.
    I32(i32),
    /// 64-bit signed int.
    I64(i64),
    /// 32-bit float as bit pattern.
    F32Bits(u32),
    /// 64-bit float as bit pattern.
    F64Bits(u64),
    /// Boolean key.
    Bool(bool),
    /// Date32 days since epoch.
    Date32(i32),
    /// Timestamp ticks since epoch.
    Timestamp(i64),
    /// Fixed-point 128-bit integer value.
    Decimal128(i128),
}

/// A composite key: the full per-row tuple across every group-by column, in
/// `aggregate.group_by` order (`None` denotes SQL `NULL` at that key position).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TupleKey(Vec<Option<KeyValue>>);

/// Resolve each group-by ordinal to a [`KeyColumn`] borrowing into `batch`.
fn resolve_key_columns<'a>(
    aggregate: &AggregateSpec,
    batch: &'a RecordBatch,
) -> BoltResult<Vec<KeyColumn<'a>>> {
    let mut out: Vec<KeyColumn<'a>> = Vec::with_capacity(aggregate.group_by.len());
    for &ord in &aggregate.group_by {
        let io = aggregate.inputs.get(ord).ok_or_else(|| {
            BoltError::Plan(format!(
                "wide GROUP BY: group_by ordinal {} out of range ({} inputs)",
                ord,
                aggregate.inputs.len()
            ))
        })?;
        // Reject unsupported key dtypes up-front so we can fail fast.
        match io.dtype {
            DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Bool
            | DataType::Date32
            | DataType::Timestamp(_, _)
            | DataType::Decimal128(_, _) => {}
            DataType::Utf8 => {
                return Err(BoltError::Type(format!(
                    "Utf8 keys in wide GROUP BY not yet supported (column '{}')",
                    io.name
                )))
            }
            DataType::Extended(_) => {
                return Err(BoltError::Type(format!(
                    "Extended keys in wide GROUP BY not yet supported (column '{}')",
                    io.name
                )))
            }
        }
        let idx = batch.schema().index_of(&io.name).map_err(|e| {
            BoltError::Plan(format!(
                "wide GROUP BY: key column '{}' not in batch: {}",
                io.name, e
            ))
        })?;
        let arr = batch.column(idx).as_ref();
        // Validate the Arrow dtype matches the plan's declared dtype.
        let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
        if arr_dtype != io.dtype {
            return Err(BoltError::Type(format!(
                "wide GROUP BY: key '{}' dtype mismatch: plan says {:?}, batch has {:?}",
                io.name, io.dtype, arr_dtype
            )));
        }
        out.push(KeyColumn {
            name: io.name.clone(),
            dtype: io.dtype,
            arr,
        });
    }
    Ok(out)
}

/// Extract one [`KeyValue`] from a [`KeyColumn`] at `row`.
fn key_value_at(kc: &KeyColumn<'_>, row: usize) -> BoltResult<KeyValue> {
    match kc.dtype {
        DataType::Int32 => {
            let pa = kc
                .arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&kc.name, "Int32"))?;
            Ok(KeyValue::I32(pa.value(row)))
        }
        DataType::Int64 => {
            let pa = kc
                .arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&kc.name, "Int64"))?;
            Ok(KeyValue::I64(pa.value(row)))
        }
        DataType::Float32 => {
            let pa = kc
                .arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&kc.name, "Float32"))?;
            // Review C12: canonicalise -0.0 -> +0.0 so signed-zero pairs
            // hash into one group. F3: all NaN payloads fold to one canonical
            // quiet NaN so NaN keys collapse into a single group (DuckDB).
            Ok(KeyValue::F32Bits(canonicalise_f32(pa.value(row)).to_bits()))
        }
        DataType::Float64 => {
            let pa = kc
                .arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&kc.name, "Float64"))?;
            // Review C12: same signed-zero canonicalisation as Float32.
            Ok(KeyValue::F64Bits(canonicalise_f64(pa.value(row)).to_bits()))
        }
        DataType::Bool => {
            let ba = kc
                .arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| downcast_err(&kc.name, "Bool"))?;
            Ok(KeyValue::Bool(ba.value(row)))
        }
        DataType::Date32 => {
            let pa = kc
                .arr
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| downcast_err(&kc.name, "Date32"))?;
            Ok(KeyValue::Date32(pa.value(row)))
        }
        DataType::Timestamp(unit, _) => {
            let ticks = match unit {
                TimeUnit::Second => kc
                    .arr
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .ok_or_else(|| downcast_err(&kc.name, "TimestampSecond"))?
                    .value(row),
                TimeUnit::Millisecond => kc
                    .arr
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .ok_or_else(|| downcast_err(&kc.name, "TimestampMillisecond"))?
                    .value(row),
                TimeUnit::Microsecond => kc
                    .arr
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(|| downcast_err(&kc.name, "TimestampMicrosecond"))?
                    .value(row),
                TimeUnit::Nanosecond => kc
                    .arr
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .ok_or_else(|| downcast_err(&kc.name, "TimestampNanosecond"))?
                    .value(row),
            };
            Ok(KeyValue::Timestamp(ticks))
        }
        DataType::Decimal128(_, _) => {
            let da = kc
                .arr
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| downcast_err(&kc.name, "Decimal128"))?;
            Ok(KeyValue::Decimal128(da.value(row)))
        }
        DataType::Utf8 | DataType::Extended(_) => Err(BoltError::Type(format!(
            "wide GROUP BY: key dtype {:?} not supported",
            kc.dtype
        ))),
    }
}

/// Total ordering over `TupleKey` for deterministic output (`NULLS LAST` per
/// key position, matching `NULL_GROUP_KEY = i64::MAX` on the narrow path).
fn cmp_tuple(a: &TupleKey, b: &TupleKey) -> std::cmp::Ordering {
    debug_assert_eq!(a.0.len(), b.0.len());
    for (av, bv) in a.0.iter().zip(b.0.iter()) {
        let ord = match (av, bv) {
            (Some(x), Some(y)) => cmp_key_value(x, y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        };
        if ord != std::cmp::Ordering::Equal {
            return ord;
        }
    }
    std::cmp::Ordering::Equal
}

/// Per-variant ordering for [`KeyValue`]. Cross-variant comparisons (which
/// shouldn't occur — every position in the tuple has a fixed dtype) fall
/// back to comparing the discriminant for stability.
fn cmp_key_value(a: &KeyValue, b: &KeyValue) -> std::cmp::Ordering {
    use KeyValue::*;
    match (a, b) {
        (I32(x), I32(y)) => x.cmp(y),
        (I64(x), I64(y)) => x.cmp(y),
        (F32Bits(x), F32Bits(y)) => x.cmp(y),
        (F64Bits(x), F64Bits(y)) => x.cmp(y),
        (Bool(x), Bool(y)) => x.cmp(y),
        (Date32(x), Date32(y)) => x.cmp(y),
        (Timestamp(x), Timestamp(y)) => x.cmp(y),
        (Decimal128(x), Decimal128(y)) => x.cmp(y),
        // Mixed variants: fall back to discriminant order. In a well-formed
        // input every row produces the same variant at each position, so we
        // never reach this branch.
        _ => variant_rank(a).cmp(&variant_rank(b)),
    }
}

fn variant_rank(v: &KeyValue) -> u8 {
    match v {
        KeyValue::I32(_) => 0,
        KeyValue::I64(_) => 1,
        KeyValue::F32Bits(_) => 2,
        KeyValue::F64Bits(_) => 3,
        KeyValue::Bool(_) => 4,
        KeyValue::Date32(_) => 5,
        KeyValue::Timestamp(_) => 6,
        KeyValue::Decimal128(_) => 7,
    }
}

// ---------------------------------------------------------------------------
// Aggregate plumbing: resolve inputs, build accumulators.
// ---------------------------------------------------------------------------

/// Per-aggregate plan: which input column to read each row from, what its
/// declared dtype is, and the matching Arrow array.
struct AggInputPlan<'a> {
    /// Aggregate kind (drives accumulator selection + finalisation).
    kind: AggKind,
    /// Input column name (for diagnostics).
    name: String,
    /// Input column dtype.
    dtype: DataType,
    /// The Arrow input array, or `None` for literal `COUNT(*)`.
    arr: Option<&'a dyn Array>,
}

/// Discriminator that mirrors [`AggregateExpr`] but without carrying the
/// `Expr` payload — keeps the inner loop tight.
#[derive(Clone, Copy, Debug)]
enum AggKind {
    Sum,
    Min,
    Max,
    Count,
    Avg,
    VarPop,
    VarSamp,
    StddevPop,
    StddevSamp,
}

impl AggKind {
    /// Map an `AggregateExpr` to its `AggKind` discriminant.
    fn from_expr(e: &AggregateExpr) -> BoltResult<Self> {
        match e {
            AggregateExpr::Sum(_) => Ok(AggKind::Sum),
            AggregateExpr::Min(_) => Ok(AggKind::Min),
            AggregateExpr::Max(_) => Ok(AggKind::Max),
            AggregateExpr::Count(_) => Ok(AggKind::Count),
            AggregateExpr::Avg(_) => Ok(AggKind::Avg),
            AggregateExpr::VarPop(_) => Ok(AggKind::VarPop),
            AggregateExpr::VarSamp(_) => Ok(AggKind::VarSamp),
            AggregateExpr::StddevPop(_) => Ok(AggKind::StddevPop),
            AggregateExpr::StddevSamp(_) => Ok(AggKind::StddevSamp),
        }
    }
}

/// Resolve every aggregate's input column. Rejects non-bare-column inputs
/// and Bool/Utf8 input dtypes.
fn resolve_aggregates<'a>(
    aggregate: &AggregateSpec,
    batch: &'a RecordBatch,
) -> BoltResult<Vec<AggInputPlan<'a>>> {
    let mut out: Vec<AggInputPlan<'a>> = Vec::with_capacity(aggregate.aggregates.len());
    for agg in &aggregate.aggregates {
        let expr = match agg {
            AggregateExpr::Sum(e)
            | AggregateExpr::Min(e)
            | AggregateExpr::Max(e)
            | AggregateExpr::Count(e)
            | AggregateExpr::Avg(e) => e,
            AggregateExpr::VarPop(e) | AggregateExpr::VarSamp(e) => e.as_ref(),
            // STDDEV variants box their operand. The wide-GROUP-BY path is
            // gated by `AggKind::from_expr` to reject STDDEV up front, but
            // this match needs an exhaustive arm to compile — deref so the
            // borrow shape matches if it ever does reach here through a
            // future code path.
            AggregateExpr::StddevPop(e) | AggregateExpr::StddevSamp(e) => e.as_ref(),
        };
        if let AggregateExpr::Count(Expr::Literal(lit)) = agg {
            if !matches!(lit, Literal::Null) {
                out.push(AggInputPlan {
                    kind: AggKind::Count,
                    name: "*".into(),
                    dtype: DataType::Int64,
                    arr: None,
                });
                continue;
            }
        }
        let col_name = bare_column_name(expr)?;
        let io = aggregate
            .inputs
            .iter()
            .find(|c| c.name == col_name)
            .ok_or_else(|| {
                BoltError::Plan(format!(
                    "wide GROUP BY: aggregate input column '{}' not in plan inputs",
                    col_name
                ))
            })?;
        match io.dtype {
            DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64
            | DataType::Bool
            | DataType::Date32
            | DataType::Timestamp(_, _)
            | DataType::Decimal128(_, _) => {}
            DataType::Utf8 if matches!(agg, AggregateExpr::Count(_)) => {}
            DataType::Utf8 | DataType::Extended(_) => {
                return Err(BoltError::Type(format!(
                    "wide GROUP BY: Utf8/Extended aggregate inputs not supported (column '{}')",
                    io.name
                )))
            }
        }
        let idx = batch.schema().index_of(&io.name).map_err(|e| {
            BoltError::Plan(format!(
                "wide GROUP BY: aggregate input '{}' not in batch: {}",
                io.name, e
            ))
        })?;
        let arr = batch.column(idx).as_ref();
        let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
        if arr_dtype != io.dtype {
            return Err(BoltError::Type(format!(
                "wide GROUP BY: aggregate input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
                io.name, io.dtype, arr_dtype
            )));
        }
        out.push(AggInputPlan {
            kind: AggKind::from_expr(agg)?,
            name: io.name.clone(),
            dtype: io.dtype,
            arr: Some(arr),
        });
    }
    Ok(out)
}

/// Extract one [`AggInputValue`] from an [`AggInputPlan`] at `row`.
fn agg_input_at(plan: &AggInputPlan<'_>, row: usize) -> BoltResult<AggInputValue> {
    let Some(arr) = plan.arr else {
        return Ok(AggInputValue::CountRow);
    };
    if matches!(plan.kind, AggKind::Count) {
        return Ok(AggInputValue::CountRow);
    }
    match plan.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&plan.name, "Int32"))?;
            Ok(AggInputValue::I32(pa.value(row)))
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&plan.name, "Int64"))?;
            Ok(AggInputValue::I64(pa.value(row)))
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&plan.name, "Float32"))?;
            Ok(AggInputValue::F32(pa.value(row)))
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&plan.name, "Float64"))?;
            Ok(AggInputValue::F64(pa.value(row)))
        }
        DataType::Bool => {
            let ba = arr
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| downcast_err(&plan.name, "Bool"))?;
            Ok(AggInputValue::Bool(ba.value(row)))
        }
        DataType::Date32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Date32Array>()
                .ok_or_else(|| downcast_err(&plan.name, "Date32"))?;
            Ok(AggInputValue::I32(pa.value(row)))
        }
        DataType::Timestamp(unit, _) => {
            let ticks = match unit {
                TimeUnit::Second => arr
                    .as_any()
                    .downcast_ref::<TimestampSecondArray>()
                    .ok_or_else(|| downcast_err(&plan.name, "TimestampSecond"))?
                    .value(row),
                TimeUnit::Millisecond => arr
                    .as_any()
                    .downcast_ref::<TimestampMillisecondArray>()
                    .ok_or_else(|| downcast_err(&plan.name, "TimestampMillisecond"))?
                    .value(row),
                TimeUnit::Microsecond => arr
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(|| downcast_err(&plan.name, "TimestampMicrosecond"))?
                    .value(row),
                TimeUnit::Nanosecond => arr
                    .as_any()
                    .downcast_ref::<TimestampNanosecondArray>()
                    .ok_or_else(|| downcast_err(&plan.name, "TimestampNanosecond"))?
                    .value(row),
            };
            Ok(AggInputValue::I64(ticks))
        }
        DataType::Decimal128(_, scale) => {
            let da = arr
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| downcast_err(&plan.name, "Decimal128"))?;
            Ok(AggInputValue::Decimal128 {
                raw: da.value(row),
                scale,
            })
        }
        DataType::Utf8 | DataType::Extended(_) => Err(BoltError::Type(format!(
            "wide GROUP BY: aggregate input dtype {:?} not supported (column '{}')",
            plan.dtype, plan.name
        ))),
    }
}

/// One per-row aggregate input, typed by the input column's dtype.
#[derive(Clone, Copy, Debug)]
enum AggInputValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Decimal128 { raw: i128, scale: i8 },
    CountRow,
}

impl AggInputValue {
    /// Widen to f64 (used by AVG and float SUM, which keep f64 internally).
    fn as_f64(self) -> f64 {
        match self {
            AggInputValue::I32(v) => v as f64,
            AggInputValue::I64(v) => v as f64,
            AggInputValue::F32(v) => v as f64,
            AggInputValue::F64(v) => v,
            AggInputValue::Bool(b) => {
                if b {
                    1.0
                } else {
                    0.0
                }
            }
            AggInputValue::Decimal128 { raw, scale } => (raw as f64) / 10f64.powi(scale as i32),
            AggInputValue::CountRow => 1.0,
        }
    }

    /// Widen to i64 (used by integer SUM).
    fn as_i64(self) -> BoltResult<i64> {
        match self {
            AggInputValue::I32(v) => Ok(v as i64),
            AggInputValue::I64(v) => Ok(v),
            _ => Err(BoltError::Type(
                "wide GROUP BY: internal — tried to widen non-integer to i64".into(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-group accumulators.
// ---------------------------------------------------------------------------

/// Per-group running state for one aggregate. The variant is selected by
/// `make_initial_accumulators` based on the (agg, input-dtype) pair; once a
/// group exists, every row's update must match the same variant.
#[derive(Clone, Debug)]
enum Accumulator {
    /// `SUM` over an integer input. Widened to i64 internally, checked on overflow.
    SumI64 { total: i64, seen: bool },
    /// `SUM` over a Decimal128 input. Checked on i128 overflow.
    SumDecimal128 { total: i128, seen: bool },
    /// `SUM` over a float input. Widened to f64.
    SumF64 { total: f64, seen: bool },
    /// `MIN` over an Int32 / Date32 input.
    MinI32 { v: i32, seen: bool },
    /// `MIN` over an Int64 / Timestamp input.
    MinI64 { v: i64, seen: bool },
    /// `MIN` over a Float32 input.
    MinF32 { v: f32, seen: bool },
    /// `MIN` over a Float64 input.
    MinF64 { v: f64, seen: bool },
    /// `MIN` over a Bool input.
    MinBool { v: bool, seen: bool },
    /// `MIN` over a Decimal128 input.
    MinDecimal128 { v: i128, seen: bool },
    /// `MAX` over an Int32 / Date32 input.
    MaxI32 { v: i32, seen: bool },
    /// `MAX` over an Int64 / Timestamp input.
    MaxI64 { v: i64, seen: bool },
    /// `MAX` over a Float32 input.
    MaxF32 { v: f32, seen: bool },
    /// `MAX` over a Float64 input.
    MaxF64 { v: f64, seen: bool },
    /// `MAX` over a Bool input.
    MaxBool { v: bool, seen: bool },
    /// `MAX` over a Decimal128 input.
    MaxDecimal128 { v: i128, seen: bool },
    /// `COUNT(expr)` — count of non-null rows.
    Count { n: u64 },
    /// `AVG` — running sum + count.
    Avg { sum: f64, n: u64 },
    /// `VAR_POP` / `VAR_SAMP` / `STDDEV_POP` / `STDDEV_SAMP` — Welford
    /// `(count, mean, M2)` state.
    Welford {
        state: crate::exec::welford::WelfordState,
    },
}

impl Accumulator {
    /// Fold one row's value into the accumulator.
    fn update(&mut self, value: AggInputValue) -> BoltResult<()> {
        match self {
            Accumulator::SumI64 { total, seen } => {
                let v = value.as_i64()?;
                *total = total.checked_add(v).ok_or_else(|| {
                    BoltError::Type("SUM(integer) overflow: accumulator exceeds i64 range".into())
                })?;
                *seen = true;
            }
            Accumulator::SumDecimal128 { total, seen } => {
                if let AggInputValue::Decimal128 { raw, .. } = value {
                    *total = total.checked_add(raw).ok_or_else(|| {
                        BoltError::Type(
                            "SUM(Decimal128) precision overflow: accumulator exceeds i128 range"
                                .into(),
                        )
                    })?;
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("SumDecimal128"));
                }
            }
            Accumulator::SumF64 { total, seen } => {
                *total += value.as_f64();
                *seen = true;
            }
            Accumulator::MinI32 { v, seen } => {
                if let AggInputValue::I32(x) = value {
                    if !*seen || x < *v {
                        *v = x;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MinI32"));
                }
            }
            Accumulator::MinI64 { v, seen } => {
                if let AggInputValue::I64(x) = value {
                    if !*seen || x < *v {
                        *v = x;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MinI64"));
                }
            }
            Accumulator::MinF32 { v, seen } => {
                if let AggInputValue::F32(x) = value {
                    if !*seen || x < *v {
                        *v = x;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MinF32"));
                }
            }
            Accumulator::MinF64 { v, seen } => {
                if let AggInputValue::F64(x) = value {
                    if !*seen || x < *v {
                        *v = x;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MinF64"));
                }
            }
            Accumulator::MinBool { v, seen } => {
                if let AggInputValue::Bool(x) = value {
                    if !*seen || (!x && *v) {
                        *v = x;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MinBool"));
                }
            }
            Accumulator::MinDecimal128 { v, seen } => {
                if let AggInputValue::Decimal128 { raw, .. } = value {
                    if !*seen || raw < *v {
                        *v = raw;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MinDecimal128"));
                }
            }
            Accumulator::MaxI32 { v, seen } => {
                if let AggInputValue::I32(x) = value {
                    if !*seen || x > *v {
                        *v = x;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MaxI32"));
                }
            }
            Accumulator::MaxI64 { v, seen } => {
                if let AggInputValue::I64(x) = value {
                    if !*seen || x > *v {
                        *v = x;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MaxI64"));
                }
            }
            Accumulator::MaxF32 { v, seen } => {
                if let AggInputValue::F32(x) = value {
                    if !*seen || x > *v {
                        *v = x;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MaxF32"));
                }
            }
            Accumulator::MaxF64 { v, seen } => {
                if let AggInputValue::F64(x) = value {
                    if !*seen || x > *v {
                        *v = x;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MaxF64"));
                }
            }
            Accumulator::MaxBool { v, seen } => {
                if let AggInputValue::Bool(x) = value {
                    if !*seen || (x && !*v) {
                        *v = x;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MaxBool"));
                }
            }
            Accumulator::MaxDecimal128 { v, seen } => {
                if let AggInputValue::Decimal128 { raw, .. } = value {
                    if !*seen || raw > *v {
                        *v = raw;
                    }
                    *seen = true;
                } else {
                    return Err(variant_mismatch_err("MaxDecimal128"));
                }
            }
            Accumulator::Count { n } => {
                let _ = value;
                *n += 1;
            }
            Accumulator::Avg { sum, n } => {
                *sum += value.as_f64();
                *n += 1;
            }
            Accumulator::Welford { state } => {
                state.push(value.as_f64());
            }
        }
        Ok(())
    }
}

/// Wrap a "got the wrong AggInputValue variant" error.
fn variant_mismatch_err(acc_name: &str) -> BoltError {
    BoltError::Other(format!(
        "wide GROUP BY: internal — accumulator {acc_name} received a value of the wrong dtype"
    ))
}

/// Allocate the initial accumulator vector for one group (one entry per
/// aggregate, picked by the (agg-kind, input-dtype) pair).
fn make_initial_accumulators(plan: &[AggInputPlan<'_>]) -> BoltResult<Vec<Accumulator>> {
    let mut out = Vec::with_capacity(plan.len());
    for p in plan {
        let acc = match (p.kind, p.dtype) {
            (AggKind::Sum, DataType::Int32) | (AggKind::Sum, DataType::Int64) => {
                Accumulator::SumI64 {
                    total: 0,
                    seen: false,
                }
            }
            (AggKind::Sum, DataType::Decimal128(_, _)) => Accumulator::SumDecimal128 {
                total: 0,
                seen: false,
            },
            (AggKind::Sum, DataType::Float32) | (AggKind::Sum, DataType::Float64) => {
                Accumulator::SumF64 {
                    total: 0.0,
                    seen: false,
                }
            }
            (AggKind::Min, DataType::Int32) | (AggKind::Min, DataType::Date32) => {
                Accumulator::MinI32 {
                    v: i32::MAX,
                    seen: false,
                }
            }
            (AggKind::Min, DataType::Int64) | (AggKind::Min, DataType::Timestamp(_, _)) => {
                Accumulator::MinI64 {
                    v: i64::MAX,
                    seen: false,
                }
            }
            (AggKind::Min, DataType::Float32) => Accumulator::MinF32 {
                v: f32::INFINITY,
                seen: false,
            },
            (AggKind::Min, DataType::Float64) => Accumulator::MinF64 {
                v: f64::INFINITY,
                seen: false,
            },
            (AggKind::Min, DataType::Bool) => Accumulator::MinBool {
                v: true,
                seen: false,
            },
            (AggKind::Min, DataType::Decimal128(_, _)) => Accumulator::MinDecimal128 {
                v: i128::MAX,
                seen: false,
            },
            (AggKind::Max, DataType::Int32) | (AggKind::Max, DataType::Date32) => {
                Accumulator::MaxI32 {
                    v: i32::MIN,
                    seen: false,
                }
            }
            (AggKind::Max, DataType::Int64) | (AggKind::Max, DataType::Timestamp(_, _)) => {
                Accumulator::MaxI64 {
                    v: i64::MIN,
                    seen: false,
                }
            }
            (AggKind::Max, DataType::Float32) => Accumulator::MaxF32 {
                v: f32::NEG_INFINITY,
                seen: false,
            },
            (AggKind::Max, DataType::Float64) => Accumulator::MaxF64 {
                v: f64::NEG_INFINITY,
                seen: false,
            },
            (AggKind::Max, DataType::Bool) => Accumulator::MaxBool {
                v: false,
                seen: false,
            },
            (AggKind::Max, DataType::Decimal128(_, _)) => Accumulator::MaxDecimal128 {
                v: i128::MIN,
                seen: false,
            },
            (AggKind::Count, _) => Accumulator::Count { n: 0 },
            (
                AggKind::Avg,
                DataType::Int32
                | DataType::Int64
                | DataType::Float32
                | DataType::Float64
                | DataType::Bool
                | DataType::Decimal128(_, _),
            ) => Accumulator::Avg { sum: 0.0, n: 0 },
            (
                AggKind::VarPop | AggKind::VarSamp | AggKind::StddevPop | AggKind::StddevSamp,
                DataType::Int32
                | DataType::Int64
                | DataType::Float32
                | DataType::Float64
                | DataType::Decimal128(_, _),
            ) => Accumulator::Welford {
                state: crate::exec::welford::WelfordState::empty(),
            },
            _ => {
                return Err(BoltError::Type(format!(
                    "wide GROUP BY: cannot make accumulator for ({:?}, {:?}) on column '{}'",
                    p.kind, p.dtype, p.name
                )))
            }
        };
        out.push(acc);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Output array assembly.
// ---------------------------------------------------------------------------

/// Build one Arrow array per group-by column, indexed in `aggregate.group_by`
/// order. Each row in the output corresponds to one sorted group.
fn build_key_arrays(
    sorted: &[(TupleKey, Vec<Accumulator>)],
    key_cols: &[KeyColumn<'_>],
) -> BoltResult<Vec<ArrayRef>> {
    let m = key_cols.len();
    let n = sorted.len();

    enum ColBuf {
        I32(Vec<Option<i32>>, bool),
        I64(Vec<Option<i64>>, bool),
        F32(Vec<Option<f32>>, bool),
        F64(Vec<Option<f64>>, bool),
        Bool(Vec<Option<bool>>, bool),
        Date32(Vec<Option<i32>>, bool),
        Timestamp {
            vals: Vec<Option<i64>>,
            has_null: bool,
            unit: TimeUnit,
            tz: Option<&'static str>,
        },
        Decimal128 {
            vals: Vec<Option<i128>>,
            has_null: bool,
            precision: u8,
            scale: i8,
        },
    }

    let mut buffers: Vec<ColBuf> = Vec::with_capacity(m);
    for kc in key_cols {
        match kc.dtype {
            DataType::Int32 => buffers.push(ColBuf::I32(Vec::with_capacity(n), false)),
            DataType::Int64 => buffers.push(ColBuf::I64(Vec::with_capacity(n), false)),
            DataType::Float32 => buffers.push(ColBuf::F32(Vec::with_capacity(n), false)),
            DataType::Float64 => buffers.push(ColBuf::F64(Vec::with_capacity(n), false)),
            DataType::Bool => buffers.push(ColBuf::Bool(Vec::with_capacity(n), false)),
            DataType::Date32 => buffers.push(ColBuf::Date32(Vec::with_capacity(n), false)),
            DataType::Timestamp(unit, tz) => buffers.push(ColBuf::Timestamp {
                vals: Vec::with_capacity(n),
                has_null: false,
                unit,
                tz,
            }),
            DataType::Decimal128(precision, scale) => buffers.push(ColBuf::Decimal128 {
                vals: Vec::with_capacity(n),
                has_null: false,
                precision,
                scale,
            }),
            DataType::Utf8 | DataType::Extended(_) => {
                return Err(BoltError::Type(format!(
                    "wide GROUP BY: key dtype {:?} not supported on output",
                    kc.dtype
                )))
            }
        }
    }

    for (tuple, _) in sorted {
        if tuple.0.len() != m {
            return Err(BoltError::Other(format!(
                "wide GROUP BY: internal — tuple length {} != key column count {}",
                tuple.0.len(),
                m
            )));
        }
        for (buf, val) in buffers.iter_mut().zip(tuple.0.iter()) {
            match (buf, val) {
                (ColBuf::I32(v, _), Some(KeyValue::I32(x))) => v.push(Some(*x)),
                (ColBuf::I32(v, hn), None) => {
                    *hn = true;
                    v.push(None);
                }
                (ColBuf::I64(v, _), Some(KeyValue::I64(x))) => v.push(Some(*x)),
                (ColBuf::I64(v, hn), None) => {
                    *hn = true;
                    v.push(None);
                }
                (ColBuf::F32(v, _), Some(KeyValue::F32Bits(bits))) => {
                    v.push(Some(f32::from_bits(*bits)))
                }
                (ColBuf::F32(v, hn), None) => {
                    *hn = true;
                    v.push(None);
                }
                (ColBuf::F64(v, _), Some(KeyValue::F64Bits(bits))) => {
                    v.push(Some(f64::from_bits(*bits)))
                }
                (ColBuf::F64(v, hn), None) => {
                    *hn = true;
                    v.push(None);
                }
                (ColBuf::Bool(v, _), Some(KeyValue::Bool(x))) => v.push(Some(*x)),
                (ColBuf::Bool(v, hn), None) => {
                    *hn = true;
                    v.push(None);
                }
                (ColBuf::Date32(v, _), Some(KeyValue::Date32(x))) => v.push(Some(*x)),
                (ColBuf::Date32(v, hn), None) => {
                    *hn = true;
                    v.push(None);
                }
                (ColBuf::Timestamp { vals, .. }, Some(KeyValue::Timestamp(x))) => {
                    vals.push(Some(*x))
                }
                (ColBuf::Timestamp { vals, has_null, .. }, None) => {
                    *has_null = true;
                    vals.push(None);
                }
                (ColBuf::Decimal128 { vals, .. }, Some(KeyValue::Decimal128(x))) => {
                    vals.push(Some(*x))
                }
                (ColBuf::Decimal128 { vals, has_null, .. }, None) => {
                    *has_null = true;
                    vals.push(None);
                }
                _ => {
                    return Err(BoltError::Other(
                        "wide GROUP BY: internal — tuple variant mismatched key column dtype"
                            .into(),
                    ))
                }
            }
        }
    }

    let mut out: Vec<ArrayRef> = Vec::with_capacity(m);
    for buf in buffers {
        match buf {
            ColBuf::I32(v, false) => out.push(Arc::new(Int32Array::from(
                v.into_iter().map(|x| x.unwrap_or(0)).collect::<Vec<_>>(),
            )) as ArrayRef),
            ColBuf::I32(v, true) => out.push(Arc::new(Int32Array::from(v)) as ArrayRef),
            ColBuf::I64(v, false) => out.push(Arc::new(Int64Array::from(
                v.into_iter().map(|x| x.unwrap_or(0)).collect::<Vec<_>>(),
            )) as ArrayRef),
            ColBuf::I64(v, true) => out.push(Arc::new(Int64Array::from(v)) as ArrayRef),
            ColBuf::F32(v, false) => out.push(Arc::new(Float32Array::from(
                v.into_iter().map(|x| x.unwrap_or(0.0)).collect::<Vec<_>>(),
            )) as ArrayRef),
            ColBuf::F32(v, true) => out.push(Arc::new(Float32Array::from(v)) as ArrayRef),
            ColBuf::F64(v, false) => out.push(Arc::new(Float64Array::from(
                v.into_iter().map(|x| x.unwrap_or(0.0)).collect::<Vec<_>>(),
            )) as ArrayRef),
            ColBuf::F64(v, true) => out.push(Arc::new(Float64Array::from(v)) as ArrayRef),
            ColBuf::Bool(v, false) => out.push(Arc::new(BooleanArray::from(
                v.into_iter()
                    .map(|x| x.unwrap_or(false))
                    .collect::<Vec<_>>(),
            )) as ArrayRef),
            ColBuf::Bool(v, true) => out.push(Arc::new(BooleanArray::from(v)) as ArrayRef),
            ColBuf::Date32(v, false) => out.push(Arc::new(Date32Array::from(
                v.into_iter().map(|x| x.unwrap_or(0)).collect::<Vec<_>>(),
            )) as ArrayRef),
            ColBuf::Date32(v, true) => out.push(Arc::new(Date32Array::from(v)) as ArrayRef),
            ColBuf::Timestamp {
                vals,
                has_null: false,
                unit,
                tz,
            } => out.push(timestamp_array_from_i64(
                vals.into_iter().map(|x| x.unwrap_or(0)).collect(),
                unit,
                tz,
            )),
            ColBuf::Timestamp {
                vals,
                has_null: true,
                unit,
                tz,
            } => out.push(timestamp_opt_array_from_i64(vals, unit, tz)),
            ColBuf::Decimal128 {
                vals,
                has_null,
                precision,
                scale,
            } => {
                let da = if has_null {
                    Decimal128Array::from(vals)
                } else {
                    Decimal128Array::from(
                        vals.into_iter().map(|x| x.unwrap_or(0)).collect::<Vec<_>>(),
                    )
                }
                .with_precision_and_scale(precision, scale)
                .map_err(|e| {
                    BoltError::Type(format!(
                        "wide GROUP BY: Decimal128 key precision/scale ({precision}, {scale}) invalid: {e}"
                    ))
                })?;
                out.push(Arc::new(da) as ArrayRef);
            }
        }
    }
    Ok(out)
}

/// Finalise one aggregate's accumulator into an Arrow array matching
/// `out_field.dtype`. `i` indexes into each group's accumulator vector.
fn finalize_agg_column(
    agg: &AggregateExpr,
    out_field: &Field,
    i: usize,
    sorted: &[(TupleKey, Vec<Accumulator>)],
) -> BoltResult<ArrayRef> {
    match agg {
        AggregateExpr::Count(_) => {
            // COUNT always outputs Int64.
            let mut out: Vec<i64> = Vec::with_capacity(sorted.len());
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::Count { n }) => out.push(*n as i64),
                    _ => {
                        return Err(BoltError::Other(
                            "wide GROUP BY: internal — COUNT accumulator missing or wrong variant"
                                .into(),
                        ))
                    }
                }
            }
            pack_typed_array(out_field.dtype, TypedColumn::I64(out))
        }
        AggregateExpr::Avg(_) => {
            if out_field.dtype != DataType::Float64 {
                return Err(BoltError::Type(format!(
                    "wide GROUP BY AVG output dtype must be Float64, got {:?}",
                    out_field.dtype
                )));
            }
            let mut out: Vec<Option<f64>> = Vec::with_capacity(sorted.len());
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::Avg { sum, n }) => {
                        let v = (*n != 0).then(|| sum / (*n as f64));
                        out.push(v);
                    }
                    _ => {
                        return Err(BoltError::Other(
                            "wide GROUP BY: internal — AVG accumulator missing or wrong variant"
                                .into(),
                        ))
                    }
                }
            }
            Ok(Arc::new(Float64Array::from(out)) as ArrayRef)
        }
        AggregateExpr::Sum(_) | AggregateExpr::Min(_) | AggregateExpr::Max(_) => {
            // SUM keeps input dtype (via cast from the wider accumulator);
            // MIN/MAX always keep input dtype exactly.
            //
            // We collect into the accumulator's natural dtype, then defer
            // the (possibly narrowing) cast to `pack_typed_array`.
            collect_sum_min_max(sorted, i, out_field.dtype)
        }
        // VAR_POP / VAR_SAMP / STDDEV_POP / STDDEV_SAMP. Output is
        // always a nullable Float64 array — empty / single-observation
        // groups produce SQL NULL via the matching `WelfordState`
        // finaliser (see `crate::exec::welford` for the canonical
        // numerics: `var_pop` returns None for count == 0, `var_samp` and
        // `stddev_samp` return None for count <= 1).
        AggregateExpr::VarPop(_) => {
            finalize_welford_column(sorted, i, WelfordKind::VarPop, out_field)
        }
        AggregateExpr::VarSamp(_) => {
            finalize_welford_column(sorted, i, WelfordKind::VarSamp, out_field)
        }
        AggregateExpr::StddevPop(_) => {
            finalize_welford_column(sorted, i, WelfordKind::StddevPop, out_field)
        }
        AggregateExpr::StddevSamp(_) => {
            finalize_welford_column(sorted, i, WelfordKind::StddevSamp, out_field)
        }
    }
}

/// Tag for finalising a per-group `WelfordState` in the wide GROUP BY path.
#[derive(Clone, Copy, Debug)]
enum WelfordKind {
    VarPop,
    VarSamp,
    StddevPop,
    StddevSamp,
}

/// Walk `sorted`, picking out the `i`-th accumulator (which must be a
/// [`Accumulator::Welford`]) per group, and emit a nullable Float64
/// array using the matching `var_*` / `stddev_*` finaliser.
fn finalize_welford_column(
    sorted: &[(TupleKey, Vec<Accumulator>)],
    i: usize,
    kind: WelfordKind,
    out_field: &Field,
) -> BoltResult<ArrayRef> {
    if out_field.dtype != DataType::Float64 {
        return Err(BoltError::Type(format!(
            "wide GROUP BY: VAR/STDDEV output dtype must be Float64, got {:?}",
            out_field.dtype
        )));
    }
    let mut out: Vec<Option<f64>> = Vec::with_capacity(sorted.len());
    for (_, accs) in sorted {
        match accs.get(i) {
            Some(Accumulator::Welford { state }) => {
                let v = match kind {
                    WelfordKind::VarPop => state.var_pop(),
                    WelfordKind::VarSamp => state.var_samp(),
                    WelfordKind::StddevPop => state.stddev_pop(),
                    WelfordKind::StddevSamp => state.stddev_samp(),
                };
                out.push(v);
            }
            _ => {
                return Err(BoltError::Other(
                    "wide GROUP BY: internal — VAR/STDDEV accumulator missing \
                     or wrong variant"
                        .into(),
                ))
            }
        }
    }
    Ok(Arc::new(Float64Array::from(out)) as ArrayRef)
}

/// Walk `sorted`, picking out accumulator `i`, and collect into the
/// accumulator's natural Rust type. Then hand off to `pack_typed_array`
/// which casts to `out_dtype` for output.
fn collect_sum_min_max(
    sorted: &[(TupleKey, Vec<Accumulator>)],
    i: usize,
    out_dtype: DataType,
) -> BoltResult<ArrayRef> {
    // Empty-group case (every row was dropped by the H1 NULL filter): emit
    // an empty Arrow array of the plan-declared output dtype. Without this
    // shortcut the `first.unwrap()` below would error on legitimate inputs
    // whose entire batch is NULL-keyed.
    if sorted.is_empty() {
        return pack_typed_array(
            out_dtype,
            match out_dtype {
                DataType::Int32 | DataType::Date32 => TypedColumn::I32(Vec::new()),
                DataType::Int64 | DataType::Timestamp(_, _) => TypedColumn::I64(Vec::new()),
                DataType::Float32 => TypedColumn::F32(Vec::new()),
                DataType::Float64 => TypedColumn::F64(Vec::new()),
                DataType::Bool => TypedColumn::Bool(Vec::new()),
                DataType::Decimal128(_, _) => TypedColumn::Decimal128(Vec::new()),
                DataType::Utf8 | DataType::Extended(_) => {
                    return Err(BoltError::Type(format!(
                        "wide GROUP BY: cannot emit empty {:?} aggregate column",
                        out_dtype
                    )))
                }
            },
        );
    }
    // Peek the first non-empty group to choose the natural collector type.
    let first = sorted
        .first()
        .and_then(|(_, accs)| accs.get(i))
        .ok_or_else(|| {
            BoltError::Other("wide GROUP BY: internal — no groups when finalising aggregate".into())
        })?;

    let collected: TypedColumn = match first {
        Accumulator::SumI64 { .. } => {
            let mut out: Vec<Option<i64>> = Vec::with_capacity(sorted.len());
            let mut has_null = false;
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::SumI64 { total, seen }) => {
                        if *seen {
                            out.push(Some(*total));
                        } else {
                            has_null = true;
                            out.push(None);
                        }
                    }
                    _ => return Err(variant_mismatch_err("SumI64")),
                }
            }
            if has_null {
                TypedColumn::OptI64(out)
            } else {
                TypedColumn::I64(out.into_iter().map(|x| x.unwrap_or(0)).collect())
            }
        }
        Accumulator::SumDecimal128 { .. } => {
            let mut out: Vec<Option<i128>> = Vec::with_capacity(sorted.len());
            let mut has_null = false;
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::SumDecimal128 { total, seen }) => {
                        if *seen {
                            out.push(Some(*total));
                        } else {
                            has_null = true;
                            out.push(None);
                        }
                    }
                    _ => return Err(variant_mismatch_err("SumDecimal128")),
                }
            }
            if has_null {
                TypedColumn::OptDecimal128(out)
            } else {
                TypedColumn::Decimal128(out.into_iter().map(|x| x.unwrap_or(0)).collect())
            }
        }
        Accumulator::SumF64 { .. } => {
            let mut out: Vec<Option<f64>> = Vec::with_capacity(sorted.len());
            let mut has_null = false;
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::SumF64 { total, seen }) => {
                        if *seen {
                            out.push(Some(*total));
                        } else {
                            has_null = true;
                            out.push(None);
                        }
                    }
                    _ => return Err(variant_mismatch_err("SumF64")),
                }
            }
            if has_null {
                TypedColumn::OptF64(out)
            } else {
                TypedColumn::F64(out.into_iter().map(|x| x.unwrap_or(0.0)).collect())
            }
        }
        Accumulator::MinI32 { .. } | Accumulator::MaxI32 { .. } => {
            let mut out: Vec<Option<i32>> = Vec::with_capacity(sorted.len());
            let mut has_null = false;
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::MinI32 { v, seen })
                    | Some(Accumulator::MaxI32 { v, seen }) => {
                        if *seen {
                            out.push(Some(*v));
                        } else {
                            has_null = true;
                            out.push(None);
                        }
                    }
                    _ => return Err(variant_mismatch_err("MinI32/MaxI32")),
                }
            }
            if has_null {
                TypedColumn::OptI32(out)
            } else {
                TypedColumn::I32(out.into_iter().map(|x| x.unwrap_or(0)).collect())
            }
        }
        Accumulator::MinI64 { .. } | Accumulator::MaxI64 { .. } => {
            let mut out: Vec<Option<i64>> = Vec::with_capacity(sorted.len());
            let mut has_null = false;
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::MinI64 { v, seen })
                    | Some(Accumulator::MaxI64 { v, seen }) => {
                        if *seen {
                            out.push(Some(*v));
                        } else {
                            has_null = true;
                            out.push(None);
                        }
                    }
                    _ => return Err(variant_mismatch_err("MinI64/MaxI64")),
                }
            }
            if has_null {
                TypedColumn::OptI64(out)
            } else {
                TypedColumn::I64(out.into_iter().map(|x| x.unwrap_or(0)).collect())
            }
        }
        Accumulator::MinF32 { .. } | Accumulator::MaxF32 { .. } => {
            let mut out: Vec<Option<f32>> = Vec::with_capacity(sorted.len());
            let mut has_null = false;
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::MinF32 { v, seen })
                    | Some(Accumulator::MaxF32 { v, seen }) => {
                        if *seen {
                            out.push(Some(*v));
                        } else {
                            has_null = true;
                            out.push(None);
                        }
                    }
                    _ => return Err(variant_mismatch_err("MinF32/MaxF32")),
                }
            }
            if has_null {
                TypedColumn::OptF32(out)
            } else {
                TypedColumn::F32(out.into_iter().map(|x| x.unwrap_or(0.0)).collect())
            }
        }
        Accumulator::MinF64 { .. } | Accumulator::MaxF64 { .. } => {
            let mut out: Vec<Option<f64>> = Vec::with_capacity(sorted.len());
            let mut has_null = false;
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::MinF64 { v, seen })
                    | Some(Accumulator::MaxF64 { v, seen }) => {
                        if *seen {
                            out.push(Some(*v));
                        } else {
                            has_null = true;
                            out.push(None);
                        }
                    }
                    _ => return Err(variant_mismatch_err("MinF64/MaxF64")),
                }
            }
            if has_null {
                TypedColumn::OptF64(out)
            } else {
                TypedColumn::F64(out.into_iter().map(|x| x.unwrap_or(0.0)).collect())
            }
        }
        Accumulator::MinBool { .. } | Accumulator::MaxBool { .. } => {
            let mut out: Vec<Option<bool>> = Vec::with_capacity(sorted.len());
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::MinBool { v, seen })
                    | Some(Accumulator::MaxBool { v, seen }) => {
                        out.push(if *seen { Some(*v) } else { None });
                    }
                    _ => return Err(variant_mismatch_err("MinBool/MaxBool")),
                }
            }
            return Ok(Arc::new(BooleanArray::from(out)) as ArrayRef);
        }
        Accumulator::MinDecimal128 { .. } | Accumulator::MaxDecimal128 { .. } => {
            let mut out: Vec<Option<i128>> = Vec::with_capacity(sorted.len());
            let mut has_null = false;
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::MinDecimal128 { v, seen })
                    | Some(Accumulator::MaxDecimal128 { v, seen }) => {
                        if *seen {
                            out.push(Some(*v));
                        } else {
                            has_null = true;
                            out.push(None);
                        }
                    }
                    _ => return Err(variant_mismatch_err("MinDecimal128/MaxDecimal128")),
                }
            }
            if has_null {
                TypedColumn::OptDecimal128(out)
            } else {
                TypedColumn::Decimal128(out.into_iter().map(|x| x.unwrap_or(0)).collect())
            }
        }
        Accumulator::Count { .. } | Accumulator::Avg { .. } => {
            return Err(BoltError::Other(
                "wide GROUP BY: internal — COUNT/AVG variant in SUM/MIN/MAX path".into(),
            ))
        }
        Accumulator::Welford { .. } => {
            return Err(BoltError::Other(
                "wide GROUP BY: internal — Welford variant in SUM/MIN/MAX path".into(),
            ))
        }
    };
    pack_typed_array(out_dtype, collected)
}

/// A typed column ready to cast into Arrow.
enum TypedColumn {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
    Bool(Vec<bool>),
    Decimal128(Vec<i128>),
    OptI32(Vec<Option<i32>>),
    OptI64(Vec<Option<i64>>),
    OptF32(Vec<Option<f32>>),
    OptF64(Vec<Option<f64>>),
    OptDecimal128(Vec<Option<i128>>),
}

/// Cast a typed column into an Arrow array of `out_dtype`.
fn pack_typed_array(out_dtype: DataType, col: TypedColumn) -> BoltResult<ArrayRef> {
    match (col, out_dtype) {
        // Exact-match paths.
        (TypedColumn::I32(v), DataType::Int32) => Ok(Arc::new(Int32Array::from(v)) as ArrayRef),
        (TypedColumn::I64(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(v)) as ArrayRef),
        (TypedColumn::F32(v), DataType::Float32) => Ok(Arc::new(Float32Array::from(v)) as ArrayRef),
        (TypedColumn::F64(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(v)) as ArrayRef),
        (TypedColumn::Bool(v), DataType::Bool) => Ok(Arc::new(BooleanArray::from(v)) as ArrayRef),
        (TypedColumn::I32(v), DataType::Date32) => Ok(Arc::new(Date32Array::from(v)) as ArrayRef),
        (TypedColumn::I64(v), DataType::Timestamp(unit, tz)) => {
            Ok(timestamp_array_from_i64(v, unit, tz))
        }
        (TypedColumn::Decimal128(v), DataType::Decimal128(precision, scale)) => {
            let da = Decimal128Array::from(v)
                .with_precision_and_scale(precision, scale)
                .map_err(|e| {
                    BoltError::Type(format!(
                        "wide GROUP BY: Decimal128 output precision/scale ({precision}, {scale}) invalid: {e}"
                    ))
                })?;
            Ok(Arc::new(da) as ArrayRef)
        }
        (TypedColumn::OptI32(v), DataType::Int32) => Ok(Arc::new(Int32Array::from(v)) as ArrayRef),
        (TypedColumn::OptI64(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(v)) as ArrayRef),
        (TypedColumn::OptF32(v), DataType::Float32) => {
            Ok(Arc::new(Float32Array::from(v)) as ArrayRef)
        }
        (TypedColumn::OptF64(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(v)) as ArrayRef)
        }
        (TypedColumn::OptI32(v), DataType::Date32) => {
            Ok(Arc::new(Date32Array::from(v)) as ArrayRef)
        }
        (TypedColumn::OptI64(v), DataType::Timestamp(unit, tz)) => {
            Ok(timestamp_opt_array_from_i64(v, unit, tz))
        }
        (TypedColumn::OptDecimal128(v), DataType::Decimal128(precision, scale)) => {
            let da = Decimal128Array::from(v)
                .with_precision_and_scale(precision, scale)
                .map_err(|e| {
                    BoltError::Type(format!(
                        "wide GROUP BY: Decimal128 output precision/scale ({precision}, {scale}) invalid: {e}"
                    ))
                })?;
            Ok(Arc::new(da) as ArrayRef)
        }

        // Widening (matches groupby.rs::pack_array).
        (TypedColumn::I32(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(
            v.into_iter().map(|x| x as i64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (TypedColumn::I32(v), DataType::Float32) => Ok(Arc::new(Float32Array::from(
            v.into_iter().map(|x| x as f32).collect::<Vec<_>>(),
        )) as ArrayRef),
        (TypedColumn::I32(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (TypedColumn::I64(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (TypedColumn::F32(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (TypedColumn::F64(v), DataType::Float32) => Ok(Arc::new(Float32Array::from(
            v.into_iter().map(|x| x as f32).collect::<Vec<_>>(),
        )) as ArrayRef),
        (TypedColumn::OptI32(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(
            v.into_iter()
                .map(|x| x.map(|n| n as i64))
                .collect::<Vec<_>>(),
        )) as ArrayRef),
        (TypedColumn::OptI32(v), DataType::Float32) => Ok(Arc::new(Float32Array::from(
            v.into_iter()
                .map(|x| x.map(|n| n as f32))
                .collect::<Vec<_>>(),
        )) as ArrayRef),
        (TypedColumn::OptI32(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter()
                .map(|x| x.map(|n| n as f64))
                .collect::<Vec<_>>(),
        )) as ArrayRef),
        (TypedColumn::OptI64(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter()
                .map(|x| x.map(|n| n as f64))
                .collect::<Vec<_>>(),
        )) as ArrayRef),
        (TypedColumn::OptF32(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter()
                .map(|x| x.map(|n| n as f64))
                .collect::<Vec<_>>(),
        )) as ArrayRef),
        (TypedColumn::OptF64(v), DataType::Float32) => Ok(Arc::new(Float32Array::from(
            v.into_iter()
                .map(|x| x.map(|n| n as f32))
                .collect::<Vec<_>>(),
        )) as ArrayRef),

        (_, dt) => Err(BoltError::Type(format!(
            "wide GROUP BY: cannot pack scalars into output dtype {:?}",
            dt
        ))),
    }
}

fn timestamp_opt_array_from_i64(
    host: Vec<Option<i64>>,
    unit: TimeUnit,
    tz: Option<&'static str>,
) -> ArrayRef {
    let tz_owned: Option<Arc<str>> = tz.map(Arc::from);
    match unit {
        TimeUnit::Second => {
            Arc::new(TimestampSecondArray::from(host).with_timezone_opt(tz_owned)) as ArrayRef
        }
        TimeUnit::Millisecond => {
            Arc::new(TimestampMillisecondArray::from(host).with_timezone_opt(tz_owned)) as ArrayRef
        }
        TimeUnit::Microsecond => {
            Arc::new(TimestampMicrosecondArray::from(host).with_timezone_opt(tz_owned)) as ArrayRef
        }
        TimeUnit::Nanosecond => {
            Arc::new(TimestampNanosecondArray::from(host).with_timezone_opt(tz_owned)) as ArrayRef
        }
    }
}

// ---------------------------------------------------------------------------
// Tests — host-only, no GPU.
// ---------------------------------------------------------------------------
// these arrow/plan aliases are used only by the #[cfg(test)] modules
// below; the non-test schema conversion now lives in exec::schema_convert.
// cfg(test)-gated so normal builds don't see an unused import.
#[cfg(test)]
use arrow_schema::Field as ArrowField;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::physical_plan::ColumnIO;

    /// Convenience: extract the I64 array at column `idx` of `rb`.
    fn col_i64(rb: &RecordBatch, idx: usize) -> Vec<i64> {
        rb.column(idx)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 column")
            .values()
            .to_vec()
    }

    /// Convenience: extract the I32 array at column `idx` of `rb`.
    fn col_i32(rb: &RecordBatch, idx: usize) -> Vec<i32> {
        rb.column(idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32 column")
            .values()
            .to_vec()
    }

    /// Convenience: extract the F64 array at column `idx` of `rb`.
    fn col_f64(rb: &RecordBatch, idx: usize) -> Vec<f64> {
        rb.column(idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64 column")
            .values()
            .to_vec()
    }

    /// Build a 3-column (Int64, Int64, Int64) batch where the third column is
    /// the per-row "value" we'll SUM.
    fn build_two_key_batch(rows: &[(i64, i64, i64)]) -> RecordBatch {
        let a: Vec<i64> = rows.iter().map(|r| r.0).collect();
        let b: Vec<i64> = rows.iter().map(|r| r.1).collect();
        let v: Vec<i64> = rows.iter().map(|r| r.2).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int64, false),
            ArrowField::new("b", ArrowDataType::Int64, false),
            ArrowField::new("v", ArrowDataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(a)) as ArrayRef,
                Arc::new(Int64Array::from(b)) as ArrayRef,
                Arc::new(Int64Array::from(v)) as ArrayRef,
            ],
        )
        .expect("batch")
    }

    /// Build the plan for `SELECT a, b, SUM(v), COUNT(v), AVG(v) FROM t GROUP BY (a, b)`.
    fn build_plan_two_key_full() -> PhysicalPlan {
        let inputs = vec![
            ColumnIO {
                name: "a".to_string(),
                dtype: DataType::Int64,
            },
            ColumnIO {
                name: "b".to_string(),
                dtype: DataType::Int64,
            },
            ColumnIO {
                name: "v".to_string(),
                dtype: DataType::Int64,
            },
        ];
        let output_schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("sum_v", DataType::Int64, true),
            Field::new("count_v", DataType::Int64, true),
            Field::new("avg_v", DataType::Float64, true),
        ]);
        let aggregate = AggregateSpec {
            inputs,
            group_by: vec![0, 1],
            aggregates: vec![
                AggregateExpr::Sum(Expr::Column("v".into())),
                AggregateExpr::Count(Expr::Column("v".into())),
                AggregateExpr::Avg(Expr::Column("v".into())),
            ],
            output_schema,
            input_has_validity: Vec::new(),
        };
        PhysicalPlan::Aggregate {
            table: "t".to_string(),
            pre: None,
            aggregate,
        }
    }

    /// 1. Two Int64 keys + SUM + COUNT + AVG: end-to-end host-side reduction.
    #[test]
    fn two_int64_keys_sum_count_avg() {
        // Three distinct (a, b) tuples; some rows repeat.
        // (1, 10): vs = [100, 200]              → sum 300, count 2, avg 150
        // (1, 20): vs = [50]                    → sum 50,  count 1, avg 50
        // (2, 10): vs = [7, 8, 9]               → sum 24,  count 3, avg 8
        let rows: Vec<(i64, i64, i64)> = vec![
            (1, 10, 100),
            (2, 10, 7),
            (1, 20, 50),
            (1, 10, 200),
            (2, 10, 8),
            (2, 10, 9),
        ];
        let batch = build_two_key_batch(&rows);
        let plan = build_plan_two_key_full();
        let out = execute_groupby_wide(&plan, &batch).expect("groupby ok");
        assert_eq!(out.num_rows(), 3);

        // Sorted by (a, b) ascending:
        //   (1, 10), (1, 20), (2, 10)
        let a = col_i64(&out, 0);
        let b = col_i64(&out, 1);
        let s = col_i64(&out, 2);
        let c = col_i64(&out, 3);
        let avg = col_f64(&out, 4);
        assert_eq!(a, vec![1, 1, 2]);
        assert_eq!(b, vec![10, 20, 10]);
        assert_eq!(s, vec![300, 50, 24]);
        assert_eq!(c, vec![2, 1, 3]);
        assert_eq!(avg, vec![150.0, 50.0, 8.0]);
    }

    /// 2. Deterministic ordering: running the same input twice gives byte-
    ///    identical output, even though the underlying HashMap iteration is
    ///    randomised.
    #[test]
    fn deterministic_ordering() {
        // A slightly bigger batch so HashMap iteration randomisation actually
        // has something to scramble.
        let mut rows: Vec<(i64, i64, i64)> = Vec::new();
        for a in 0..20i64 {
            for b in 0..5i64 {
                rows.push((a, b, a * 100 + b));
            }
        }
        let batch = build_two_key_batch(&rows);
        let plan = build_plan_two_key_full();
        let out1 = execute_groupby_wide(&plan, &batch).expect("groupby ok 1");
        let out2 = execute_groupby_wide(&plan, &batch).expect("groupby ok 2");
        assert_eq!(out1.num_rows(), out2.num_rows());
        assert_eq!(out1.num_rows(), 100);
        for col in 0..out1.num_columns() {
            let v1 = col_i64_or_f64(&out1, col);
            let v2 = col_i64_or_f64(&out2, col);
            assert_eq!(v1, v2, "column {col} not deterministic");
        }
        // Spot-check sortedness on the first key column.
        let a_col = col_i64(&out1, 0);
        let b_col = col_i64(&out1, 1);
        for i in 1..a_col.len() {
            assert!(
                (a_col[i - 1], b_col[i - 1]) <= (a_col[i], b_col[i]),
                "row {i} out of order"
            );
        }
    }

    /// Helper for the determinism test: pull any numeric column out as a
    /// `Vec<f64>` for comparison without caring about its precise dtype.
    fn col_i64_or_f64(rb: &RecordBatch, col: usize) -> Vec<f64> {
        let arr = rb.column(col);
        match arr.data_type() {
            ArrowDataType::Int64 => arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .iter()
                .map(|&v| v as f64)
                .collect(),
            ArrowDataType::Float64 => arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap()
                .values()
                .to_vec(),
            other => panic!("unexpected dtype {other:?}"),
        }
    }

    /// per-group VAR_POP / STDDEV_POP / VAR_SAMP / STDDEV_SAMP via
    /// the wide host-side path. Constructs a 2-key GROUP BY where each
    /// group has a different sample shape — empty (no rows reach it),
    /// single-observation, and multi-observation — to cover the NULL
    /// emission cases in one go.
    #[test]
    fn two_int64_keys_var_stddev_basic() {
        // Two distinct (a, b) tuples:
        //   (1, 10): vs = [1, 2, 3]  -> n=3, var_pop = 2/3, var_samp = 1.0
        //   (1, 20): vs = [42]       -> n=1, var_pop = 0,   var_samp = NULL
        let rows: Vec<(i64, i64, i64)> = vec![(1, 10, 1), (1, 10, 2), (1, 10, 3), (1, 20, 42)];
        let batch = build_two_key_batch(&rows);

        let inputs = vec![
            ColumnIO {
                name: "a".into(),
                dtype: DataType::Int64,
            },
            ColumnIO {
                name: "b".into(),
                dtype: DataType::Int64,
            },
            ColumnIO {
                name: "v".into(),
                dtype: DataType::Int64,
            },
        ];
        let output_schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
            Field::new("var_pop_v", DataType::Float64, true),
            Field::new("var_samp_v", DataType::Float64, true),
            Field::new("stddev_pop_v", DataType::Float64, true),
            Field::new("stddev_samp_v", DataType::Float64, true),
        ]);
        let aggregate = AggregateSpec {
            inputs,
            group_by: vec![0, 1],
            aggregates: vec![
                AggregateExpr::VarPop(Box::new(Expr::Column("v".into()))),
                AggregateExpr::VarSamp(Box::new(Expr::Column("v".into()))),
                AggregateExpr::StddevPop(Box::new(Expr::Column("v".into()))),
                AggregateExpr::StddevSamp(Box::new(Expr::Column("v".into()))),
            ],
            output_schema,
            input_has_validity: Vec::new(),
        };
        let plan = PhysicalPlan::Aggregate {
            table: "t".to_string(),
            pre: None,
            aggregate,
        };
        let out = execute_groupby_wide(&plan, &batch).expect("groupby ok");
        assert_eq!(out.num_rows(), 2);

        let var_pop = out
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("var_pop f64");
        let var_samp = out
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("var_samp f64");
        let stddev_pop = out
            .column(4)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("stddev_pop f64");
        let stddev_samp = out
            .column(5)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("stddev_samp f64");

        // Sorted by (a, b): row 0 = (1, 10), row 1 = (1, 20).
        // n=3, [1,2,3]
        assert!((var_pop.value(0) - (2.0 / 3.0)).abs() < 1e-12);
        assert!((var_samp.value(0) - 1.0).abs() < 1e-12);
        assert!((stddev_pop.value(0) - (2.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((stddev_samp.value(0) - 1.0).abs() < 1e-12);
        assert!(!var_pop.is_null(0));
        assert!(!var_samp.is_null(0));

        // n=1, [42]:
        //   var_pop  = 0
        //   var_samp = NULL (count-1 == 0)
        //   stddev_pop = 0
        //   stddev_samp = NULL
        assert_eq!(var_pop.value(1), 0.0);
        assert!(!var_pop.is_null(1));
        assert!(var_samp.is_null(1));
        assert_eq!(stddev_pop.value(1), 0.0);
        assert!(stddev_samp.is_null(1));
    }

    /// 3. Three Int32 keys (which the narrow path rejects) work fine here;
    ///    this is the primary case our caller hands us. Also exercises MIN
    ///    and MAX accumulators.
    #[test]
    fn three_int32_keys_min_max() {
        // (a, b, c) → row value v
        let a: Vec<i32> = vec![1, 1, 1, 2, 2];
        let b: Vec<i32> = vec![10, 10, 11, 10, 10];
        let c: Vec<i32> = vec![100, 100, 100, 100, 200];
        let v: Vec<i32> = vec![5, 3, 99, 7, 8];

        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int32, false),
            ArrowField::new("b", ArrowDataType::Int32, false),
            ArrowField::new("c", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(a)) as ArrayRef,
                Arc::new(Int32Array::from(b)) as ArrayRef,
                Arc::new(Int32Array::from(c)) as ArrayRef,
                Arc::new(Int32Array::from(v)) as ArrayRef,
            ],
        )
        .expect("batch");

        let inputs = vec![
            ColumnIO {
                name: "a".into(),
                dtype: DataType::Int32,
            },
            ColumnIO {
                name: "b".into(),
                dtype: DataType::Int32,
            },
            ColumnIO {
                name: "c".into(),
                dtype: DataType::Int32,
            },
            ColumnIO {
                name: "v".into(),
                dtype: DataType::Int32,
            },
        ];
        let output_schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
            Field::new("c", DataType::Int32, false),
            Field::new("min_v", DataType::Int32, true),
            Field::new("max_v", DataType::Int32, true),
        ]);
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs,
                group_by: vec![0, 1, 2],
                aggregates: vec![
                    AggregateExpr::Min(Expr::Column("v".into())),
                    AggregateExpr::Max(Expr::Column("v".into())),
                ],
                output_schema,
                input_has_validity: Vec::new(),
            },
        };

        let out = execute_groupby_wide(&plan, &batch).expect("groupby ok");
        // Groups sorted: (1,10,100), (1,11,100), (2,10,100), (2,10,200)
        assert_eq!(out.num_rows(), 4);
        let a = col_i32(&out, 0);
        let b = col_i32(&out, 1);
        let c = col_i32(&out, 2);
        let min_v = col_i32(&out, 3);
        let max_v = col_i32(&out, 4);
        assert_eq!(a, vec![1, 1, 2, 2]);
        assert_eq!(b, vec![10, 11, 10, 10]);
        assert_eq!(c, vec![100, 100, 100, 200]);
        assert_eq!(min_v, vec![3, 99, 7, 8]);
        assert_eq!(max_v, vec![5, 99, 7, 8]);
    }

    /// 4. Reject: `pre` kernel present.
    #[test]
    fn rejects_pre_kernel() {
        // Build a plan with a non-None `pre` to verify we reject early. We
        // synthesize a trivial KernelSpec so the rest of the plan still has
        // the expected shape.
        use crate::plan::physical_plan::KernelSpec;
        let inputs = vec![ColumnIO {
            name: "a".into(),
            dtype: DataType::Int64,
        }];
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: Some(KernelSpec {
                inputs: vec![],
                outputs: vec![],
                ops: vec![],
                predicate: None,
                register_count: 0,
                input_has_validity: vec![],
                output_has_validity: vec![],
            }),
            aggregate: AggregateSpec {
                inputs,
                group_by: vec![0],
                aggregates: vec![],
                output_schema: Schema::new(vec![Field::new("a", DataType::Int64, false)]),
                input_has_validity: vec![],
            },
        };
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "a",
                ArrowDataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from(vec![1i64])) as ArrayRef],
        )
        .expect("batch");
        let err = execute_groupby_wide(&plan, &batch).expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("pre kernel"),
            "expected pre-kernel error, got: {msg}"
        );
    }

    /// 5. Reject: Utf8 key column dtype.
    #[test]
    fn rejects_utf8_key() {
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![ColumnIO {
                    name: "s".into(),
                    dtype: DataType::Utf8,
                }],
                group_by: vec![0],
                aggregates: vec![],
                output_schema: Schema::new(vec![Field::new("s", DataType::Utf8, false)]),
                input_has_validity: Vec::new(),
            },
        };
        // We never read the batch's `s` column for this plan because we fail
        // during `resolve_key_columns`; passing a placeholder batch is fine.
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "s",
                ArrowDataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from(vec![1i64])) as ArrayRef],
        )
        .expect("batch");
        let err = execute_groupby_wide(&plan, &batch).expect_err("should reject");
        let msg = format!("{err}");
        assert!(msg.contains("Utf8"), "expected Utf8 error, got: {msg}");
    }

    /// 6. Reject: aggregate input that isn't a bare column reference.
    #[test]
    fn rejects_non_bare_agg_input() {
        use crate::plan::logical_plan::BinaryOp;
        let inputs = vec![
            ColumnIO {
                name: "a".into(),
                dtype: DataType::Int64,
            },
            ColumnIO {
                name: "v".into(),
                dtype: DataType::Int64,
            },
        ];
        // SUM(a * v) — not a bare column.
        let agg_expr = Expr::Binary {
            op: BinaryOp::Mul,
            left: Box::new(Expr::Column("a".into())),
            right: Box::new(Expr::Column("v".into())),
        };
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs,
                group_by: vec![0],
                aggregates: vec![AggregateExpr::Sum(agg_expr)],
                output_schema: Schema::new(vec![
                    Field::new("a", DataType::Int64, false),
                    Field::new("sum_av", DataType::Int64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("a", ArrowDataType::Int64, false),
                ArrowField::new("v", ArrowDataType::Int64, false),
            ])),
            vec![
                Arc::new(Int64Array::from(vec![1i64])) as ArrayRef,
                Arc::new(Int64Array::from(vec![5i64])) as ArrayRef,
            ],
        )
        .expect("batch");
        let err = execute_groupby_wide(&plan, &batch).expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("bare column"),
            "expected bare-column error, got: {msg}"
        );
    }

    /// 7. Single-column key (narrow case) is accepted gracefully — this
    ///    module is a generic fallback and doesn't require >64 bits of key
    ///    width. The dispatcher prefers the GPU path for these cases, but
    ///    we should still produce correct results.
    #[test]
    fn single_int64_key_accepted() {
        let rows: Vec<(i64, i64, i64)> =
            vec![(0, 0, 10), (0, 0, 20), (1, 0, 30), (2, 0, 40), (1, 0, 50)];
        let batch = build_two_key_batch(&rows);

        // Group by just `a`; SUM(v).
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO {
                        name: "a".into(),
                        dtype: DataType::Int64,
                    },
                    ColumnIO {
                        name: "v".into(),
                        dtype: DataType::Int64,
                    },
                ],
                group_by: vec![0],
                aggregates: vec![AggregateExpr::Sum(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("a", DataType::Int64, false),
                    Field::new("sum_v", DataType::Int64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let out = execute_groupby_wide(&plan, &batch).expect("ok");
        assert_eq!(out.num_rows(), 3);
        let a = col_i64(&out, 0);
        let s = col_i64(&out, 1);
        assert_eq!(a, vec![0, 1, 2]);
        assert_eq!(s, vec![30, 80, 40]);
    }

    // -----------------------------------------------------------------
    // H1 NULL-handling tests for the wide-key path.
    //
    // The pre-fix per-row loop read every key column and aggregate input
    // via `pa.value(row)`, which returns garbage bytes at NULL positions
    // (Arrow's validity bitmap was ignored). The post-fix loop:
    //   * drops a row when ANY key column is NULL at that row, and
    //   * skips an aggregate's update when its own input is NULL.
    // These tests pin both behaviours on a 2-key wide-input.
    // -----------------------------------------------------------------

    /// Build a 3-column (Int64?, Int64?, Int64?) NULLable batch.
    /// `None` in any column produces a NULL at that row.
    fn build_two_key_batch_nullable(
        rows: &[(Option<i64>, Option<i64>, Option<i64>)],
    ) -> RecordBatch {
        let a: Vec<Option<i64>> = rows.iter().map(|r| r.0).collect();
        let b: Vec<Option<i64>> = rows.iter().map(|r| r.1).collect();
        let v: Vec<Option<i64>> = rows.iter().map(|r| r.2).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int64, true),
            ArrowField::new("b", ArrowDataType::Int64, true),
            ArrowField::new("v", ArrowDataType::Int64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(a)) as ArrayRef,
                Arc::new(Int64Array::from(b)) as ArrayRef,
                Arc::new(Int64Array::from(v)) as ArrayRef,
            ],
        )
        .expect("batch")
    }

    /// SQL-standard NULL group keys: a NULL in any key column forms a distinct
    /// group (`NULL IS NOT DISTINCT FROM NULL` per position), so `(1, NULL)`,
    /// `(NULL, 10)`, and `(NULL, NULL)` are distinct groups, and an all-NULL
    /// value group emits SQL NULL for SUM/MIN/MAX/AVG and 0 for COUNT(col).
    #[test]
    fn null_key_tuples_and_all_null_value_group_follow_sql_standard() {
        let rows: Vec<(Option<i64>, Option<i64>, Option<i64>)> = vec![
            (Some(1), Some(10), Some(100)),
            (Some(2), Some(10), Some(7)),
            (Some(1), None, Some(400)),
            (Some(1), Some(10), Some(200)),
            (Some(1), None, Some(599)),
            (None, Some(10), Some(50)),
            (None, None, None),
            (None, None, None),
            (Some(2), Some(10), Some(8)),
            (Some(2), Some(10), Some(9)),
        ];
        let batch = build_two_key_batch_nullable(&rows);

        let inputs = vec![
            ColumnIO {
                name: "a".into(),
                dtype: DataType::Int64,
            },
            ColumnIO {
                name: "b".into(),
                dtype: DataType::Int64,
            },
            ColumnIO {
                name: "v".into(),
                dtype: DataType::Int64,
            },
        ];
        let output_schema = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new("b", DataType::Int64, true),
            Field::new("sum_v", DataType::Int64, true),
            Field::new("min_v", DataType::Int64, true),
            Field::new("max_v", DataType::Int64, true),
            Field::new("count_v", DataType::Int64, true),
            Field::new("avg_v", DataType::Float64, true),
        ]);
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs,
                group_by: vec![0, 1],
                aggregates: vec![
                    AggregateExpr::Sum(Expr::Column("v".into())),
                    AggregateExpr::Min(Expr::Column("v".into())),
                    AggregateExpr::Max(Expr::Column("v".into())),
                    AggregateExpr::Count(Expr::Column("v".into())),
                    AggregateExpr::Avg(Expr::Column("v".into())),
                ],
                output_schema,
                input_has_validity: Vec::new(),
            },
        };
        let out = execute_groupby_wide(&plan, &batch).expect("groupby ok");

        // Sorted NULLS LAST per column:
        // 0: (Some(1), Some(10)) -> sum 300, min 100, max 200, count 2, avg 150.0
        // 1: (Some(1), None)     -> sum 999, min 400, max 599, count 2, avg 499.5
        // 2: (Some(2), Some(10)) -> sum 24,  min 7,   max 9,   count 3, avg 8.0
        // 3: (None,    Some(10)) -> sum 50,  min 50,  max 50,  count 1, avg 50.0
        // 4: (None,    None)     -> all-NULL v: sum/min/max/avg = NULL, count = 0
        assert_eq!(out.num_rows(), 5);
        let a = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let b = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let s = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        let mn = out.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
        let mx = out.column(4).as_any().downcast_ref::<Int64Array>().unwrap();
        let c = out.column(5).as_any().downcast_ref::<Int64Array>().unwrap();
        let avg = out
            .column(6)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        let keys: Vec<(Option<i64>, Option<i64>)> = (0..5)
            .map(|r| {
                (
                    (!a.is_null(r)).then(|| a.value(r)),
                    (!b.is_null(r)).then(|| b.value(r)),
                )
            })
            .collect();
        assert_eq!(
            keys,
            vec![
                (Some(1), Some(10)),
                (Some(1), None),
                (Some(2), Some(10)),
                (None, Some(10)),
                (None, None),
            ]
        );
        let sums: Vec<Option<i64>> = (0..5)
            .map(|r| (!s.is_null(r)).then(|| s.value(r)))
            .collect();
        let mins: Vec<Option<i64>> = (0..5)
            .map(|r| (!mn.is_null(r)).then(|| mn.value(r)))
            .collect();
        let maxs: Vec<Option<i64>> = (0..5)
            .map(|r| (!mx.is_null(r)).then(|| mx.value(r)))
            .collect();
        assert_eq!(sums, vec![Some(300), Some(999), Some(24), Some(50), None]);
        assert_eq!(mins, vec![Some(100), Some(400), Some(7), Some(50), None]);
        assert_eq!(maxs, vec![Some(200), Some(599), Some(9), Some(50), None]);
        assert_eq!(c.values(), &[2, 2, 3, 1, 0]);
        assert!((avg.value(0) - 150.0).abs() < 1e-12);
        assert!((avg.value(1) - 499.5).abs() < 1e-12);
        assert!((avg.value(2) - 8.0).abs() < 1e-12);
        assert!((avg.value(3) - 50.0).abs() < 1e-12);
        assert!(avg.is_null(4));
    }

    /// H1: SUM excludes NULL value inputs (lockstep filter on the value
    /// column). Row keys are all non-null here; only the value column has
    /// a NULL, which must be dropped from the group's running total.
    #[test]
    fn null_value_excluded_from_sum() {
        // (a=1, b=10): v ∈ {100, NULL, 200}          → sum 300 (NULL excluded)
        // (a=2, b=10): v ∈ {7, 8}                    → sum 15
        let rows: Vec<(Option<i64>, Option<i64>, Option<i64>)> = vec![
            (Some(1), Some(10), Some(100)),
            (Some(2), Some(10), Some(7)),
            (Some(1), Some(10), None), // NULL value: SUM must skip
            (Some(1), Some(10), Some(200)),
            (Some(2), Some(10), Some(8)),
        ];
        let batch = build_two_key_batch_nullable(&rows);

        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO {
                        name: "a".into(),
                        dtype: DataType::Int64,
                    },
                    ColumnIO {
                        name: "b".into(),
                        dtype: DataType::Int64,
                    },
                    ColumnIO {
                        name: "v".into(),
                        dtype: DataType::Int64,
                    },
                ],
                group_by: vec![0, 1],
                aggregates: vec![AggregateExpr::Sum(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("a", DataType::Int64, true),
                    Field::new("b", DataType::Int64, true),
                    Field::new("sum_v", DataType::Int64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let out = execute_groupby_wide(&plan, &batch).expect("groupby ok");
        assert_eq!(out.num_rows(), 2);
        let a = col_i64(&out, 0);
        let b = col_i64(&out, 1);
        let s = col_i64(&out, 2);
        assert_eq!(a, vec![1, 2]);
        assert_eq!(b, vec![10, 10]);
        // Non-NULL sum: 100 + 200 = 300; 7 + 8 = 15.
        assert_eq!(s, vec![300, 15]);
    }

    /// Empty input yields a clean 0-row output with the full output schema, and
    /// an all-NULL-keyed batch groups identical NULL tuples together.
    #[test]
    fn empty_input_and_all_null_keys_behave_cleanly() {
        let empty_batch = build_two_key_batch_nullable(&[]);
        let rows: Vec<(Option<i64>, Option<i64>, Option<i64>)> = vec![
            (None, Some(10), Some(1)),
            (Some(1), None, Some(2)),
            (None, None, Some(3)),
            (None, Some(10), Some(4)),
        ];
        let null_batch = build_two_key_batch_nullable(&rows);

        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO {
                        name: "a".into(),
                        dtype: DataType::Int64,
                    },
                    ColumnIO {
                        name: "b".into(),
                        dtype: DataType::Int64,
                    },
                    ColumnIO {
                        name: "v".into(),
                        dtype: DataType::Int64,
                    },
                ],
                group_by: vec![0, 1],
                aggregates: vec![
                    AggregateExpr::Sum(Expr::Column("v".into())),
                    AggregateExpr::Min(Expr::Column("v".into())),
                    AggregateExpr::Max(Expr::Column("v".into())),
                    AggregateExpr::Count(Expr::Column("v".into())),
                    AggregateExpr::Avg(Expr::Column("v".into())),
                ],
                output_schema: Schema::new(vec![
                    Field::new("a", DataType::Int64, true),
                    Field::new("b", DataType::Int64, true),
                    Field::new("sum_v", DataType::Int64, true),
                    Field::new("min_v", DataType::Int64, true),
                    Field::new("max_v", DataType::Int64, true),
                    Field::new("count_v", DataType::Int64, true),
                    Field::new("avg_v", DataType::Float64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let empty_out = execute_groupby_wide(&plan, &empty_batch).expect("empty groupby ok");
        assert_eq!(empty_out.num_rows(), 0);
        assert_eq!(empty_out.num_columns(), 7);

        let null_out = execute_groupby_wide(&plan, &null_batch).expect("null-keys groupby ok");
        // Groups sorted NULLS LAST: (1, NULL)->2, (NULL, 10)->5, (NULL, NULL)->3.
        assert_eq!(null_out.num_rows(), 3);
        assert_eq!(col_i64(&null_out, 2), vec![2, 5, 3]);
    }

    #[test]
    fn sum_int64_overflow_errors_and_decimal_date_count_star_work() {
        let ov_batch = build_two_key_batch(&[(1, 2, i64::MAX), (1, 2, 1)]);
        let ov_plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO {
                        name: "a".into(),
                        dtype: DataType::Int64,
                    },
                    ColumnIO {
                        name: "b".into(),
                        dtype: DataType::Int64,
                    },
                    ColumnIO {
                        name: "v".into(),
                        dtype: DataType::Int64,
                    },
                ],
                group_by: vec![0, 1],
                aggregates: vec![AggregateExpr::Sum(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("a", DataType::Int64, false),
                    Field::new("b", DataType::Int64, false),
                    Field::new("sum_v", DataType::Int64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        match execute_groupby_wide(&ov_plan, &ov_batch) {
            Err(BoltError::Type(msg)) => assert!(msg.contains("SUM(integer) overflow")),
            other => panic!("expected SUM(integer) overflow, got {other:?}"),
        }

        // Decimal128 + Date32 composite key with COUNT(*) literal and SUM(Decimal128).
        let dec = Decimal128Array::from(vec![1025i128, 1025, 2050])
            .with_precision_and_scale(10, 2)
            .unwrap();
        let d = Date32Array::from(vec![19_700, 19_700, 19_701]);
        let batch = RecordBatch::try_new(
            Arc::new(ArrowSchema::new(vec![
                ArrowField::new("dec", ArrowDataType::Decimal128(10, 2), false),
                ArrowField::new("d", ArrowDataType::Date32, false),
            ])),
            vec![Arc::new(dec) as ArrayRef, Arc::new(d) as ArrayRef],
        )
        .unwrap();
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO {
                        name: "dec".into(),
                        dtype: DataType::Decimal128(10, 2),
                    },
                    ColumnIO {
                        name: "d".into(),
                        dtype: DataType::Date32,
                    },
                ],
                group_by: vec![0, 1],
                aggregates: vec![
                    AggregateExpr::Sum(Expr::Column("dec".into())),
                    AggregateExpr::Count(Expr::Literal(Literal::Int64(1))),
                ],
                output_schema: Schema::new(vec![
                    Field::new("dec", DataType::Decimal128(10, 2), false),
                    Field::new("d", DataType::Date32, false),
                    Field::new("sum_dec", DataType::Decimal128(38, 2), true),
                    Field::new("cnt", DataType::Int64, false),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let out = execute_groupby_wide(&plan, &batch).expect("decimal+date wide groupby");
        assert_eq!(out.num_rows(), 2);
        let sums = out
            .column(2)
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .unwrap();
        assert_eq!(sums.value(0), 2050);
        assert_eq!(sums.value(1), 2050);
        assert_eq!(col_i64(&out, 3), vec![2, 1]);
    }
}
