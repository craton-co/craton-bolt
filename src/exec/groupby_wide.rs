// SPDX-License-Identifier: Apache-2.0

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
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{DataType as ArrowDataType, Schema as ArrowSchema};

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Schema};
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
    let mut buf_key: Vec<KeyValue> = Vec::with_capacity(key_cols.len());
    // H1 NULL handling: precompute "any key column has nulls?" and "any
    // aggregate input has nulls?" once so the inner row loop stays
    // branch-light when both are null-free (the common case). When either
    // flag is set we fall back to per-row `is_null` probes on the relevant
    // Arrow array via its validity bitmap.
    //
    // Lockstep semantics here:
    //   - A row is DROPPED ENTIRELY when ANY group-by key column is NULL.
    //     The narrow `groupby.rs` path does the same (`pack_keys`'s AND of
    //     per-column key_valid masks); here we just short-circuit before
    //     building the tuple key.
    //   - Surviving rows feed each aggregate independently: an aggregate's
    //     update is skipped only when that aggregate's own input is NULL.
    //     This keeps SUM/MIN/MAX out of NULL slots, and makes
    //     COUNT(col)/AVG denominator reflect the non-NULL count.
    let any_key_nullable = key_cols.iter().any(|kc| kc.arr.null_count() > 0);
    let any_agg_nullable = agg_plan.iter().any(|p| p.arr.null_count() > 0);
    for row in 0..n_rows {
        // Drop the row entirely if ANY group-by column is NULL at this row.
        // We probe `is_null` on the Arrow array directly — `key_value_at`
        // would otherwise read garbage at NULL slots via `pa.value(row)`,
        // which ignores the validity bitmap.
        if any_key_nullable && key_cols.iter().any(|kc| kc.arr.is_null(row)) {
            continue;
        }
        // Build the tuple key for this row.
        buf_key.clear();
        for kc in &key_cols {
            buf_key.push(key_value_at(kc, row)?);
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
            if any_agg_nullable && agg_in.arr.is_null(row) {
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

    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
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
}

/// A composite key: the full per-row tuple across every group-by column, in
/// `aggregate.group_by` order.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TupleKey(Vec<KeyValue>);

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
            DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64 => {}
            DataType::Utf8 => {
                return Err(BoltError::Type(format!(
                    "Utf8 keys in wide GROUP BY not yet supported (column '{}')",
                    io.name
                )))
            }
            DataType::Bool => {
                return Err(BoltError::Type(format!(
                    "Bool keys not supported (column '{}')",
                    io.name
                )))
            }
            DataType::Decimal128(_, _) => {
                return Err(BoltError::Plan(format!(
                    "Decimal128 not yet lowered to GPU; coming in a follow-up \
                     (column '{}' in GROUP BY)",
                    io.name
                )))
            }
            DataType::Date32 | DataType::Timestamp(_, _) => {
                return Err(BoltError::Type(format!(
                    "Date/Timestamp keys in wide GROUP BY not yet supported (column '{}')",
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
        DataType::Bool
        | DataType::Utf8
        | DataType::Decimal128(_, _)
        | DataType::Date32
        | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
            "wide GROUP BY: key dtype {:?} not supported",
            kc.dtype
        ))),
    }
}

/// Canonicalise a float GROUP BY key: `-0.0 → +0.0` so signed-zero pairs key
/// the same group (review C12), and every NaN bit pattern → a single canonical
/// quiet NaN so all NaN keys collapse into one group, matching DuckDB (F3).
/// Mirrors `groupby_common::canonicalise_f64` / `distinct.rs` so every GROUP
/// BY / DISTINCT path agrees on the float equivalence relation.
#[inline]
fn canonicalise_f64(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// `f32` analogue of [`canonicalise_f64`].
#[inline]
fn canonicalise_f32(x: f32) -> f32 {
    if x.is_nan() {
        f32::NAN
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// Total ordering over `TupleKey` for deterministic output. We compare
/// position by position using a stable per-variant ordering; floats are
/// ordered by their `u32`/`u64` bit pattern (consistent with how they hash).
fn cmp_tuple(a: &TupleKey, b: &TupleKey) -> std::cmp::Ordering {
    // Tuples in the same group-by have identical length; debug-assert.
    debug_assert_eq!(a.0.len(), b.0.len());
    for (av, bv) in a.0.iter().zip(b.0.iter()) {
        let ord = cmp_key_value(av, bv);
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
    /// Input column dtype (Int32/Int64/Float32/Float64 only — validated up
    /// front).
    dtype: DataType,
    /// The Arrow input array.
    arr: &'a dyn Array,
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
        // Reject Bool/Utf8 inputs.
        match io.dtype {
            DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64 => {}
            DataType::Bool
            | DataType::Utf8
            | DataType::Decimal128(_, _)
            | DataType::Date32
            | DataType::Timestamp(_, _) => {
                return Err(BoltError::Type(format!(
                    "wide GROUP BY: Bool/Utf8 aggregate inputs not supported (column '{}')",
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
            arr,
        });
    }
    Ok(out)
}

/// Extract one [`AggInputValue`] from an [`AggInputPlan`] at `row`.
fn agg_input_at(plan: &AggInputPlan<'_>, row: usize) -> BoltResult<AggInputValue> {
    match plan.dtype {
        DataType::Int32 => {
            let pa = plan
                .arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&plan.name, "Int32"))?;
            Ok(AggInputValue::I32(pa.value(row)))
        }
        DataType::Int64 => {
            let pa = plan
                .arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&plan.name, "Int64"))?;
            Ok(AggInputValue::I64(pa.value(row)))
        }
        DataType::Float32 => {
            let pa = plan
                .arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&plan.name, "Float32"))?;
            Ok(AggInputValue::F32(pa.value(row)))
        }
        DataType::Float64 => {
            let pa = plan
                .arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&plan.name, "Float64"))?;
            Ok(AggInputValue::F64(pa.value(row)))
        }
        DataType::Bool
        | DataType::Utf8
        | DataType::Decimal128(_, _)
        | DataType::Date32
        | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
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
}

impl AggInputValue {
    /// Widen to f64 (used by AVG and float SUM, which keep f64 internally).
    fn as_f64(self) -> f64 {
        match self {
            AggInputValue::I32(v) => v as f64,
            AggInputValue::I64(v) => v as f64,
            AggInputValue::F32(v) => v as f64,
            AggInputValue::F64(v) => v,
        }
    }

    /// Widen to i64 (used by integer SUM).
    fn as_i64(self) -> BoltResult<i64> {
        match self {
            AggInputValue::I32(v) => Ok(v as i64),
            AggInputValue::I64(v) => Ok(v),
            AggInputValue::F32(_) | AggInputValue::F64(_) => Err(BoltError::Type(
                "wide GROUP BY: internal — tried to widen float to i64".into(),
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
    /// `SUM` over an integer input. Widened to i64 internally to match the
    /// narrow path's kernel behaviour.
    SumI64 { total: i64 },
    /// `SUM` over a float input. Widened to f64.
    SumF64 { total: f64 },
    /// `MIN` over an Int32 input.
    MinI32 { v: i32, seen: bool },
    /// `MIN` over an Int64 input.
    MinI64 { v: i64, seen: bool },
    /// `MIN` over a Float32 input.
    MinF32 { v: f32, seen: bool },
    /// `MIN` over a Float64 input.
    MinF64 { v: f64, seen: bool },
    /// `MAX` over an Int32 input.
    MaxI32 { v: i32, seen: bool },
    /// `MAX` over an Int64 input.
    MaxI64 { v: i64, seen: bool },
    /// `MAX` over a Float32 input.
    MaxF32 { v: f32, seen: bool },
    /// `MAX` over a Float64 input.
    MaxF64 { v: f64, seen: bool },
    /// `COUNT(expr)` — count of non-null rows. We don't track nulls in v1,
    /// so this is row count per group (matches narrow-path behaviour).
    Count { n: u64 },
    /// `AVG` — running sum + count.
    Avg { sum: f64, n: u64 },
    /// `VAR_POP` / `VAR_SAMP` / `STDDEV_POP` / `STDDEV_SAMP` — Welford
    /// `(count, mean, M2)` state. The output finaliser
    /// (`finalize_agg_column`) picks the right `var_*` / `stddev_*`
    /// accessor on the state.
    Welford {
        state: crate::exec::welford::WelfordState,
    },
}

impl Accumulator {
    /// Fold one row's value into the accumulator.
    fn update(&mut self, value: AggInputValue) -> BoltResult<()> {
        match self {
            Accumulator::SumI64 { total } => {
                let v = value.as_i64()?;
                // Wrap on i64 overflow to match the GPU narrow path, whose
                // SUM kernel emits `atom.global.add.u64` (silently wraps on
                // overflow). Since SUM(Int32) widens to an i64 accumulator
                // AND the plan-declared output dtype is Int64 (see
                // `crate::plan::logical_plan::sum_output_dtype`), the wrap
                // semantics matter only for SUM(Int64), where they match the
                // GPU and scalar reducer.
                *total = total.wrapping_add(v);
            }
            Accumulator::SumF64 { total } => {
                *total += value.as_f64();
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
                    // `<` propagates NaN: NaN < anything is false, so NaN
                    // never replaces a real value. Matches scalar reducer.
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
                Accumulator::SumI64 { total: 0 }
            }
            (AggKind::Sum, DataType::Float32) | (AggKind::Sum, DataType::Float64) => {
                Accumulator::SumF64 { total: 0.0 }
            }
            (AggKind::Min, DataType::Int32) => Accumulator::MinI32 {
                v: i32::MAX,
                seen: false,
            },
            (AggKind::Min, DataType::Int64) => Accumulator::MinI64 {
                v: i64::MAX,
                seen: false,
            },
            (AggKind::Min, DataType::Float32) => Accumulator::MinF32 {
                v: f32::INFINITY,
                seen: false,
            },
            (AggKind::Min, DataType::Float64) => Accumulator::MinF64 {
                v: f64::INFINITY,
                seen: false,
            },
            (AggKind::Max, DataType::Int32) => Accumulator::MaxI32 {
                v: i32::MIN,
                seen: false,
            },
            (AggKind::Max, DataType::Int64) => Accumulator::MaxI64 {
                v: i64::MIN,
                seen: false,
            },
            (AggKind::Max, DataType::Float32) => Accumulator::MaxF32 {
                v: f32::NEG_INFINITY,
                seen: false,
            },
            (AggKind::Max, DataType::Float64) => Accumulator::MaxF64 {
                v: f64::NEG_INFINITY,
                seen: false,
            },
            (AggKind::Count, _) => Accumulator::Count { n: 0 },
            (AggKind::Avg, _) => Accumulator::Avg { sum: 0.0, n: 0 },
            (
                AggKind::VarPop | AggKind::VarSamp | AggKind::StddevPop | AggKind::StddevSamp,
                DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64,
            ) => Accumulator::Welford {
                state: crate::exec::welford::WelfordState::empty(),
            },
            (_, DataType::Bool)
            | (_, DataType::Utf8)
            | (_, DataType::Decimal128(_, _))
            | (_, DataType::Date32)
            | (_, DataType::Timestamp(_, _)) => {
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

    // Per-column typed buffers.
    enum ColBuf {
        I32(Vec<i32>),
        I64(Vec<i64>),
        F32(Vec<f32>),
        F64(Vec<f64>),
    }

    let mut buffers: Vec<ColBuf> = Vec::with_capacity(m);
    for kc in key_cols {
        match kc.dtype {
            DataType::Int32 => buffers.push(ColBuf::I32(Vec::with_capacity(n))),
            DataType::Int64 => buffers.push(ColBuf::I64(Vec::with_capacity(n))),
            DataType::Float32 => buffers.push(ColBuf::F32(Vec::with_capacity(n))),
            DataType::Float64 => buffers.push(ColBuf::F64(Vec::with_capacity(n))),
            DataType::Bool
            | DataType::Utf8
            | DataType::Decimal128(_, _)
            | DataType::Date32
            | DataType::Timestamp(_, _) => {
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
                (ColBuf::I32(v), KeyValue::I32(x)) => v.push(*x),
                (ColBuf::I64(v), KeyValue::I64(x)) => v.push(*x),
                (ColBuf::F32(v), KeyValue::F32Bits(bits)) => v.push(f32::from_bits(*bits)),
                (ColBuf::F64(v), KeyValue::F64Bits(bits)) => v.push(f64::from_bits(*bits)),
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
            ColBuf::I32(v) => out.push(Arc::new(Int32Array::from(v)) as ArrayRef),
            ColBuf::I64(v) => out.push(Arc::new(Int64Array::from(v)) as ArrayRef),
            ColBuf::F32(v) => out.push(Arc::new(Float32Array::from(v)) as ArrayRef),
            ColBuf::F64(v) => out.push(Arc::new(Float64Array::from(v)) as ArrayRef),
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
            // AVG always outputs Float64.
            let mut out: Vec<f64> = Vec::with_capacity(sorted.len());
            for (_, accs) in sorted {
                match accs.get(i) {
                    Some(Accumulator::Avg { sum, n }) => {
                        let v = if *n == 0 { 0.0 } else { sum / (*n as f64) };
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
            pack_typed_array(out_field.dtype, TypedColumn::F64(out))
        }
        AggregateExpr::Sum(_) | AggregateExpr::Min(_) | AggregateExpr::Max(_) => {
            // SUM keeps input dtype (via cast from the wider accumulator);
            // MIN/MAX always keep input dtype exactly.
            //
            // We collect into the accumulator's natural dtype, then defer
            // the (possibly narrowing) cast to `pack_typed_array`.
            collect_sum_min_max(sorted, i, out_field.dtype)
        }
        // v0.7: VAR_POP / VAR_SAMP / STDDEV_POP / STDDEV_SAMP. Output is
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
                DataType::Int32 => TypedColumn::I32(Vec::new()),
                DataType::Int64 => TypedColumn::I64(Vec::new()),
                DataType::Float32 => TypedColumn::F32(Vec::new()),
                DataType::Float64 => TypedColumn::F64(Vec::new()),
                DataType::Bool
                | DataType::Utf8
                | DataType::Decimal128(_, _)
                | DataType::Date32
                | DataType::Timestamp(_, _) => {
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

    let collected: TypedColumn =
        match first {
            Accumulator::SumI64 { .. } => {
                let mut out: Vec<i64> = Vec::with_capacity(sorted.len());
                for (_, accs) in sorted {
                    match accs.get(i) {
                        Some(Accumulator::SumI64 { total }) => out.push(*total),
                        _ => return Err(variant_mismatch_err("SumI64")),
                    }
                }
                TypedColumn::I64(out)
            }
            Accumulator::SumF64 { .. } => {
                let mut out: Vec<f64> = Vec::with_capacity(sorted.len());
                for (_, accs) in sorted {
                    match accs.get(i) {
                        Some(Accumulator::SumF64 { total }) => out.push(*total),
                        _ => return Err(variant_mismatch_err("SumF64")),
                    }
                }
                TypedColumn::F64(out)
            }
            Accumulator::MinI32 { .. } | Accumulator::MaxI32 { .. } => {
                let mut out: Vec<i32> = Vec::with_capacity(sorted.len());
                for (_, accs) in sorted {
                    match accs.get(i) {
                        Some(Accumulator::MinI32 { v, .. })
                        | Some(Accumulator::MaxI32 { v, .. }) => out.push(*v),
                        _ => return Err(variant_mismatch_err("MinI32/MaxI32")),
                    }
                }
                TypedColumn::I32(out)
            }
            Accumulator::MinI64 { .. } | Accumulator::MaxI64 { .. } => {
                let mut out: Vec<i64> = Vec::with_capacity(sorted.len());
                for (_, accs) in sorted {
                    match accs.get(i) {
                        Some(Accumulator::MinI64 { v, .. })
                        | Some(Accumulator::MaxI64 { v, .. }) => out.push(*v),
                        _ => return Err(variant_mismatch_err("MinI64/MaxI64")),
                    }
                }
                TypedColumn::I64(out)
            }
            Accumulator::MinF32 { .. } | Accumulator::MaxF32 { .. } => {
                let mut out: Vec<f32> = Vec::with_capacity(sorted.len());
                for (_, accs) in sorted {
                    match accs.get(i) {
                        Some(Accumulator::MinF32 { v, .. })
                        | Some(Accumulator::MaxF32 { v, .. }) => out.push(*v),
                        _ => return Err(variant_mismatch_err("MinF32/MaxF32")),
                    }
                }
                TypedColumn::F32(out)
            }
            Accumulator::MinF64 { .. } | Accumulator::MaxF64 { .. } => {
                let mut out: Vec<f64> = Vec::with_capacity(sorted.len());
                for (_, accs) in sorted {
                    match accs.get(i) {
                        Some(Accumulator::MinF64 { v, .. })
                        | Some(Accumulator::MaxF64 { v, .. }) => out.push(*v),
                        _ => return Err(variant_mismatch_err("MinF64/MaxF64")),
                    }
                }
                TypedColumn::F64(out)
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
}

/// Cast a typed column into an Arrow array of `out_dtype`. Mirrors the
/// cross-dtype matrix in `groupby.rs::pack_array`, plus a Float64→Float32
/// narrowing path needed because SUM(Float32) accumulates in `f64` while
/// the plan-declared output dtype is `Float32`. There is no analogous
/// Int64→Int32 narrowing: SUM(Int32) declares an `Int64` output dtype
/// (see [`crate::plan::logical_plan::sum_output_dtype`]), so the widened
/// accumulator is preserved end-to-end.
fn pack_typed_array(out_dtype: DataType, col: TypedColumn) -> BoltResult<ArrayRef> {
    match (col, out_dtype) {
        // Exact-match paths.
        (TypedColumn::I32(v), DataType::Int32) => Ok(Arc::new(Int32Array::from(v)) as ArrayRef),
        (TypedColumn::I64(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(v)) as ArrayRef),
        (TypedColumn::F32(v), DataType::Float32) => Ok(Arc::new(Float32Array::from(v)) as ArrayRef),
        (TypedColumn::F64(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(v)) as ArrayRef),

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

        // Narrowing path for SUM(Float32): we accumulate in f64 to track
        // partial-sum precision but the plan-declared output dtype remains
        // Float32. Mirrors the scalar reducer and narrow GROUP BY path.
        //
        // Note there is intentionally NO `(I64, Int32)` narrowing path: SUM
        // over `Int32` now declares an `Int64` output (see
        // `crate::plan::logical_plan::sum_output_dtype`); the wider
        // accumulator is preserved end-to-end.
        (TypedColumn::F64(v), DataType::Float32) => Ok(Arc::new(Float32Array::from(
            v.into_iter().map(|x| x as f32).collect::<Vec<_>>(),
        )) as ArrayRef),

        (_, dt) => Err(BoltError::Type(format!(
            "wide GROUP BY: cannot pack scalars into output dtype {:?}",
            dt
        ))),
    }
}

// ---------------------------------------------------------------------------
// Misc helpers (mirror groupby.rs).
// ---------------------------------------------------------------------------

/// Extract the column name from a bare-column-ref expression. Rejects
/// computed expressions like `SUM(a * b)`.
fn bare_column_name(expr: &Expr) -> BoltResult<&str> {
    match expr {
        Expr::Column(name) => Ok(name.as_str()),
        Expr::Alias(inner, _) => bare_column_name(inner),
        _ => Err(BoltError::Other(
            "wide GROUP BY: aggregate input must be a bare column reference".into(),
        )),
    }
}

/// `Type` error for a failed Arrow downcast on column `name`.
fn downcast_err(name: &str, expected: &str) -> BoltError {
    BoltError::Type(format!(
        "wide GROUP BY input column '{}' could not be downcast to {}",
        name, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType` (mirrors `groupby.rs`).
fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    crate::exec::schema_convert::arrow_dtype_to_plan_basic(d, "wide GROUP BY: ")
}

/// Build an Arrow `Schema` from our plan `Schema` for the output `RecordBatch`.
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal(
        s,
        "this aggregate output path",
    )
}

// ---------------------------------------------------------------------------
// Tests — host-only, no GPU.
// ---------------------------------------------------------------------------
// v0.7: these arrow/plan aliases are used only by the #[cfg(test)] modules
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

    /// v0.7: per-group VAR_POP / STDDEV_POP / VAR_SAMP / STDDEV_SAMP via
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

    /// H1: a row with a NULL in ANY key column is dropped from every group's
    /// MIN/MAX/AVG/COUNT(col). The surviving rows must aggregate normally.
    #[test]
    fn null_key_row_excluded_from_min_max_avg_count() {
        // Layout — note row 2 has b=NULL so it must be dropped entirely.
        // (a=1, b=10): v ∈ {100, 200}                → min 100, max 200,
        //                                              avg 150, count 2, sum 300
        // (a=1, b=NULL, v=999)                       → DROPPED (NULL key)
        // (a=2, b=10): v ∈ {7, 8, 9}                 → min 7,   max 9,
        //                                              avg 8,   count 3, sum 24
        let rows: Vec<(Option<i64>, Option<i64>, Option<i64>)> = vec![
            (Some(1), Some(10), Some(100)),
            (Some(2), Some(10), Some(7)),
            (Some(1), None, Some(999)), // NULL key → must drop
            (Some(1), Some(10), Some(200)),
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

        // Only two groups must survive — the NULL-keyed row is gone.
        assert_eq!(
            out.num_rows(),
            2,
            "NULL-key row leaked into output as its own group"
        );
        let a = col_i64(&out, 0);
        let b = col_i64(&out, 1);
        let s = col_i64(&out, 2);
        let mn = col_i64(&out, 3);
        let mx = col_i64(&out, 4);
        let c = col_i64(&out, 5);
        let avg = col_f64(&out, 6);
        assert_eq!(a, vec![1, 2]);
        assert_eq!(b, vec![10, 10]);
        assert_eq!(s, vec![300, 24]);
        assert_eq!(mn, vec![100, 7]);
        assert_eq!(mx, vec![200, 9]);
        assert_eq!(c, vec![2, 3]);
        assert_eq!(avg, vec![150.0, 8.0]);
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

    /// H1: when EVERY row is filtered out (here: every row has a NULL key),
    /// we still produce a clean empty output rather than panicking on the
    /// "no groups when finalising aggregate" branch.
    #[test]
    fn all_rows_filtered_yields_clean_empty_output() {
        // All four rows have at least one NULL key → all four are dropped.
        let rows: Vec<(Option<i64>, Option<i64>, Option<i64>)> = vec![
            (None, Some(10), Some(1)),
            (Some(1), None, Some(2)),
            (None, None, Some(3)),
            (None, Some(20), Some(4)),
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
        let out = execute_groupby_wide(&plan, &batch).expect("groupby ok");
        // Empty output, with the full output schema preserved.
        assert_eq!(out.num_rows(), 0);
        assert_eq!(out.num_columns(), 7);
    }
}
