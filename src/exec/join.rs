// SPDX-License-Identifier: Apache-2.0

//! INNER JOIN executor — host-side hash join over equi-join predicates.
//!
//! Strategy (0.1.x):
//!   1. Recursively execute both child plans via the engine, producing two
//!      `RecordBatch`es.
//!   2. Pick the smaller side as the build side and the larger as the probe
//!      side (smaller hash table = less memory + faster build).
//!   3. Walk the build side once, hashing each row's join-key tuple into a
//!      `HashMap<JoinKey, Vec<row_idx>>`.
//!   4. Walk the probe side once, looking each row's key tuple up in the
//!      map; for each hit, push `(build_row_idx, probe_row_idx)` into a
//!      pair buffer.
//!   5. Use `arrow::compute::take` to materialise the build-side and
//!      probe-side output columns, then concatenate them in the order
//!      dictated by `output_schema` (left-side fields first, right-side
//!      second — same as `join_combined_schema`).
//!
//! NULL semantics: SQL equi-join does NOT match NULLs against each other
//! (`NULL = NULL` is `UNKNOWN`, not `TRUE`). Build-side rows whose key
//! includes any NULL never make it into the map; probe-side rows whose key
//! includes any NULL never look up a match. Both paths effectively drop the
//! row from the join output, which matches the standard.
//!
//! GPU hash join (key-bucket + collision-list kernels) is a 0.2 target —
//! see ROADMAP.md.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringArray, UInt32Array,
};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};

use crate::error::{BoltError, BoltResult};
use crate::exec::{Engine, QueryHandle};
use crate::plan::logical_plan::{Expr, JoinType, Schema};
use crate::plan::physical_plan::PhysicalPlan;

/// Execute an INNER JOIN. Only `JoinType::Inner` is supported in 0.1.x;
/// other variants are reserved and would surface a clear error before
/// reaching this point because the parser doesn't emit them.
///
/// `output_schema` is the disambiguated combined schema produced by
/// `join_combined_schema` and stored on `PhysicalPlan::Join`; the engine
/// passes it through so this executor doesn't have to recompute it.
pub fn execute_join(
    left: &PhysicalPlan,
    right: &PhysicalPlan,
    join_type: &JoinType,
    on: &[(Expr, Expr)],
    output_schema: &Schema,
    engine: &Engine,
) -> BoltResult<QueryHandle> {
    // Wave 8 is INNER-only. Match the variant so future kinds force a
    // compile-time decision rather than a silent INNER fallback.
    match join_type {
        JoinType::Inner => {}
    }

    if on.is_empty() {
        return Err(BoltError::Plan(
            "INNER JOIN requires at least one equi-join predicate; \
             the parser should have rejected this upstream"
                .into(),
        ));
    }

    // 1. Execute both children. Mirrors the UNION dispatch in
    //    `Engine::execute`: recurse via the engine so any operator under
    //    a Join (Scan, Filter, Project, even another Join down the road)
    //    runs through its normal path.
    let lhs = engine.execute(left)?.into_record_batch();
    let rhs = engine.execute(right)?.into_record_batch();

    // 2. Extract join-key column names per side. The parser stripped any
    //    `table.` qualifier in `lower_join_side`, so every entry must be a
    //    bare `Expr::Column`. Reject anything else with a clear message —
    //    arbitrary key expressions are a 0.2 target.
    let mut left_keys: Vec<String> = Vec::with_capacity(on.len());
    let mut right_keys: Vec<String> = Vec::with_capacity(on.len());
    for (li, ri) in on {
        left_keys.push(column_name(li, "left")?);
        right_keys.push(column_name(ri, "right")?);
    }

    let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;

    // 3. If either side is empty, the inner-join output is empty.
    //    Build an empty RecordBatch with the correct schema (one empty
    //    column per field, dtypes inferred from `output_schema`).
    if lhs.num_rows() == 0 || rhs.num_rows() == 0 {
        let cols: Vec<ArrayRef> = arrow_schema
            .fields()
            .iter()
            .map(|f| empty_array_for_dtype(f.data_type()))
            .collect::<BoltResult<Vec<_>>>()?;
        let out = RecordBatch::try_new(arrow_schema, cols).map_err(arrow_err)?;
        return Ok(QueryHandle::from_record_batch(out));
    }

    // 4. Pick build side (smaller) vs probe side (larger). Tie -> left
    //    builds. The pair buffer always records `(build_row, probe_row)`;
    //    `build_is_left` lets us re-orient that when materialising.
    let build_is_left = lhs.num_rows() <= rhs.num_rows();
    let (build_batch, build_keys, probe_batch, probe_keys) = if build_is_left {
        (&lhs, &left_keys, &rhs, &right_keys)
    } else {
        (&rhs, &right_keys, &lhs, &left_keys)
    };

    // 5. Resolve key column indices once per side.
    let build_idx = lookup_columns(build_batch, build_keys)?;
    let probe_idx = lookup_columns(probe_batch, probe_keys)?;

    // The build/probe sides must agree on key dtypes for the hash lookup
    // to be meaningful. Mismatches at this layer indicate either a planner
    // bug or a non-supported cross-dtype join — surface it explicitly.
    for (b, p) in build_idx.iter().zip(probe_idx.iter()) {
        let bdt = build_batch.column(*b).data_type();
        let pdt = probe_batch.column(*p).data_type();
        if bdt != pdt {
            return Err(BoltError::Plan(format!(
                "INNER JOIN key dtype mismatch: build side {bdt:?}, probe side {pdt:?}; \
                 cross-dtype equi-join is not yet supported"
            )));
        }
    }

    // 6. Build phase. `JoinKey` is a small owned tuple of scalar values
    //    (one entry per join column). Rows whose key contains any NULL are
    //    silently skipped — they can never match under SQL semantics.
    let mut map: HashMap<JoinKey, Vec<u32>> =
        HashMap::with_capacity(build_batch.num_rows());
    for row in 0..build_batch.num_rows() {
        if let Some(key) = extract_key(build_batch, &build_idx, row)? {
            // u32 indices — `arrow::compute::take` wants a UInt32Array.
            // We already validate `n_rows_to_u32` elsewhere for kernel
            // launches; the join's row count must fit u32 for the same
            // reason (Arrow take indices are u32).
            map.entry(key).or_default().push(row_to_u32(row)?);
        }
    }

    // Empty build map: nothing on the probe side can match.
    if map.is_empty() {
        let cols: Vec<ArrayRef> = arrow_schema
            .fields()
            .iter()
            .map(|f| empty_array_for_dtype(f.data_type()))
            .collect::<BoltResult<Vec<_>>>()?;
        let out = RecordBatch::try_new(arrow_schema, cols).map_err(arrow_err)?;
        return Ok(QueryHandle::from_record_batch(out));
    }

    // 7. Probe phase. For each probe row, look up the key in the map and
    //    emit one (build_row, probe_row) pair per match.
    let mut build_pairs: Vec<u32> = Vec::new();
    let mut probe_pairs: Vec<u32> = Vec::new();
    for row in 0..probe_batch.num_rows() {
        let Some(key) = extract_key(probe_batch, &probe_idx, row)? else {
            continue; // NULL key on probe side never matches.
        };
        if let Some(matches) = map.get(&key) {
            let probe_u32 = row_to_u32(row)?;
            for &b in matches {
                build_pairs.push(b);
                probe_pairs.push(probe_u32);
            }
        }
    }

    // No matches: empty output, but with the correct schema.
    if build_pairs.is_empty() {
        let cols: Vec<ArrayRef> = arrow_schema
            .fields()
            .iter()
            .map(|f| empty_array_for_dtype(f.data_type()))
            .collect::<BoltResult<Vec<_>>>()?;
        let out = RecordBatch::try_new(arrow_schema, cols).map_err(arrow_err)?;
        return Ok(QueryHandle::from_record_batch(out));
    }

    // 8. Materialise the result. `output_schema` is left ++ right, so the
    //    LEFT side's columns come first regardless of which physical side
    //    actually built. Re-orient (build_pairs, probe_pairs) into
    //    (left_pairs, right_pairs) accordingly.
    let (left_pairs, right_pairs) = if build_is_left {
        (&build_pairs, &probe_pairs)
    } else {
        (&probe_pairs, &build_pairs)
    };
    let left_idx_arr = UInt32Array::from(left_pairs.clone());
    let right_idx_arr = UInt32Array::from(right_pairs.clone());

    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &left_idx_arr, None).map_err(arrow_err)?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &right_idx_arr, None).map_err(arrow_err)?,
        );
    }

    let out = RecordBatch::try_new(arrow_schema, output_cols).map_err(arrow_err)?;
    Ok(QueryHandle::from_record_batch(out))
}

/// A hashable join-key value: one entry per join column. Variants cover
/// every primitive dtype the engine produces. Float keys hash by their
/// raw bit pattern (matches `distinct.rs`'s `hash_array_row`); equality
/// of `NaN` is therefore bit-wise, which is the engine-wide convention
/// for these primitive types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum JoinKeyValue {
    I32(i32),
    I64(i64),
    F32(u32),
    F64(u64),
    Bool(bool),
    Utf8(String),
}

/// A row's join key — small `Vec` because the number of equi columns is
/// typically 1-3. Single-key joins (the common case) still allocate a
/// 1-element Vec; allocating a SmallVec or duplicating the type is a
/// micro-opt left for the 0.2 GPU port.
type JoinKey = Vec<JoinKeyValue>;

/// Pull the (build_idx, ...) tuple of values for `row` out of `batch`.
/// Returns `Ok(None)` if any key column is NULL at that row — the
/// SQL "NULL keys never match" rule, applied uniformly to both sides.
fn extract_key(
    batch: &RecordBatch,
    indices: &[usize],
    row: usize,
) -> BoltResult<Option<JoinKey>> {
    let mut key: JoinKey = Vec::with_capacity(indices.len());
    for &idx in indices {
        let arr = batch.column(idx);
        if arr.is_null(row) {
            return Ok(None);
        }
        let v = match arr.data_type() {
            ArrowDataType::Int32 => JoinKeyValue::I32(
                arr.as_any().downcast_ref::<Int32Array>().unwrap().value(row),
            ),
            ArrowDataType::Int64 => JoinKeyValue::I64(
                arr.as_any().downcast_ref::<Int64Array>().unwrap().value(row),
            ),
            ArrowDataType::Float32 => JoinKeyValue::F32(
                arr.as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .value(row)
                    .to_bits(),
            ),
            ArrowDataType::Float64 => JoinKeyValue::F64(
                arr.as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(row)
                    .to_bits(),
            ),
            ArrowDataType::Boolean => JoinKeyValue::Bool(
                arr.as_any().downcast_ref::<BooleanArray>().unwrap().value(row),
            ),
            ArrowDataType::Utf8 => JoinKeyValue::Utf8(
                arr.as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap()
                    .value(row)
                    .to_string(),
            ),
            other => {
                return Err(BoltError::Type(format!(
                    "INNER JOIN: unsupported key dtype {other:?}"
                )));
            }
        };
        key.push(v);
    }
    Ok(Some(key))
}

/// Look up every name in `names` in the batch's schema, returning the
/// resolved column ordinals in the same order.
fn lookup_columns(batch: &RecordBatch, names: &[String]) -> BoltResult<Vec<usize>> {
    let mut out: Vec<usize> = Vec::with_capacity(names.len());
    for n in names {
        let idx = batch.schema().index_of(n).map_err(|e| {
            BoltError::Plan(format!("INNER JOIN: key column '{n}' not in batch: {e}"))
        })?;
        out.push(idx);
    }
    Ok(out)
}

/// Reject non-column join keys with a clear message. The parser already
/// rewrites `t1.col = t2.col` into bare `Expr::Column` refs, so anything
/// else here is a logic bug (or a future computed-key extension).
fn column_name(e: &Expr, side: &str) -> BoltResult<String> {
    match e {
        Expr::Column(n) => Ok(n.clone()),
        Expr::Alias(inner, _) => column_name(inner, side),
        other => Err(BoltError::Plan(format!(
            "INNER JOIN {side}-side key must be a column reference, got {other:?}; \
             computed join keys are a 0.2 target"
        ))),
    }
}

/// Build an empty Arrow array of the given dtype, for the
/// "join produced zero rows" early-return paths.
fn empty_array_for_dtype(dt: &ArrowDataType) -> BoltResult<ArrayRef> {
    Ok(match dt {
        ArrowDataType::Int32 => Arc::new(Int32Array::from(Vec::<i32>::new())) as ArrayRef,
        ArrowDataType::Int64 => Arc::new(Int64Array::from(Vec::<i64>::new())) as ArrayRef,
        ArrowDataType::Float32 => {
            Arc::new(Float32Array::from(Vec::<f32>::new())) as ArrayRef
        }
        ArrowDataType::Float64 => {
            Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef
        }
        ArrowDataType::Boolean => {
            Arc::new(BooleanArray::from(Vec::<bool>::new())) as ArrayRef
        }
        ArrowDataType::Utf8 => Arc::new(StringArray::from(Vec::<&str>::new())) as ArrayRef,
        other => {
            return Err(BoltError::Type(format!(
                "INNER JOIN: unsupported output dtype {other:?}"
            )))
        }
    })
}

/// Convert a host-side row index to the u32 shape `arrow::compute::take`
/// requires for its indices array. Mirrors `exec::n_rows_to_u32`'s
/// rationale: silent truncation past u32::MAX would point `take` at the
/// wrong row.
fn row_to_u32(row: usize) -> BoltResult<u32> {
    u32::try_from(row).map_err(|_| {
        BoltError::Other(format!(
            "INNER JOIN row index {row} exceeds the u32 take-indices limit ({})",
            u32::MAX
        ))
    })
}

fn arrow_err(e: arrow::error::ArrowError) -> BoltError {
    BoltError::Other(format!("arrow: {}", e))
}

/// Convert our plan `Schema` to an `arrow_schema::Schema`. Inline copy of
/// the same helper in `engine.rs` — that one is `fn`-private to engine.rs
/// and duplicating it here keeps the join executor self-contained without
/// pulling that helper out into a shared module just for one call site.
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

fn plan_dtype_to_arrow(d: crate::plan::logical_plan::DataType) -> BoltResult<ArrowDataType> {
    use crate::plan::logical_plan::DataType as D;
    Ok(match d {
        D::Int32 => ArrowDataType::Int32,
        D::Int64 => ArrowDataType::Int64,
        D::Float32 => ArrowDataType::Float32,
        D::Float64 => ArrowDataType::Float64,
        D::Bool => ArrowDataType::Boolean,
        D::Utf8 => ArrowDataType::Utf8,
    })
}
