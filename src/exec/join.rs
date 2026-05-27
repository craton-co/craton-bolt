// SPDX-License-Identifier: Apache-2.0

//! JOIN executor — host-side hash join over equi-join predicates, with
//! INNER / LEFT / RIGHT / FULL / CROSS support.
//!
//! Strategy (Stage 1, 0.3.1):
//!   1. Recursively execute both child plans via the engine, producing two
//!      `RecordBatch`es.
//!   2. For equi joins (INNER / LEFT / RIGHT / FULL):
//!      * Pick a build side and a probe side. INNER and FULL use the
//!        smaller-side-builds heuristic (smaller hash table = less memory).
//!        LEFT *must* build the right (so unmatched left rows can be
//!        detected during the probe). RIGHT *must* build the left.
//!      * Walk the build side once, hashing each row's join-key tuple into
//!        a `HashMap<JoinKey, Vec<row_idx>>`.
//!      * Walk the probe side once; for each row, look up the key and
//!        record `(build_row, probe_row)` pairs. For LEFT/FULL, an
//!        unmatched probe row also emits a `(None, probe_row)` pair so
//!        the build side gets NULL-padded.
//!      * For RIGHT/FULL, a second pass over the build side emits rows
//!        whose key was never matched, with `(build_row, None)` for the
//!        probe side.
//!   3. For CROSS: cartesian product. Every left row × every right row
//!      with no key comparison.
//!   4. Use `arrow::compute::take` to materialise the build- and
//!      probe-side output columns, then concatenate them in the order
//!      dictated by `output_schema` (left-side first, right-side second
//!      — same as `join_combined_schema`). Unmatched slots become NULL
//!      via `take`'s null-handling: an index `Null` in the indices array
//!      pulls a NULL value from the source.
//!
//! NULL semantics: SQL equi-join does NOT match NULLs against each other
//! (`NULL = NULL` is `UNKNOWN`, not `TRUE`). For INNER, both sides drop
//! NULL-key rows. For OUTER joins, NULL-key rows on the preserved side
//! still emit with the opposite side NULL-padded (matches DuckDB /
//! Postgres behaviour).
//!
//! CROSS cap: the cartesian product grows as `n_left * n_right`. We
//! enforce a hard `MAX_CROSS_ROWS = 2^31` limit (the `u32` indices array
//! that backs `arrow::compute::take` cannot address more than that) and
//! surface a clear plan error past it. Users with genuinely larger
//! cartesian products should rewrite their query.
//!
//! GPU hash join (Stage 1): INNER + single equi-key + Int32/Int64 +
//! ≥ 1024 rows / side + no NULLs in keys + unique build keys runs on the
//! GPU via [`crate::exec::gpu_join`]. Any gate miss falls through to the
//! host hash-join path. OUTER + CROSS stay host-side (Stage 2 target).

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

/// Hard cap on the row count of a CROSS JOIN output. `arrow::compute::take`
/// requires a `UInt32Array` of indices, so any result larger than
/// `u32::MAX` rows cannot be materialised through that API. We reject at
/// build time rather than overflow silently.
const MAX_CROSS_ROWS: u64 = u32::MAX as u64;

/// Execute a JOIN. Dispatches per `join_type` to one of:
///   * `execute_inner_join` — existing INNER path (smaller side builds).
///   * `execute_outer_join` — LEFT / RIGHT / FULL (build side is fixed
///      by the join kind so unmatched-row tracking is straightforward).
///   * `execute_cross_join` — cartesian product with a row-count cap.
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
    // Execute both children up front. Mirrors the UNION dispatch in
    // `Engine::execute`: every operator under a Join (Scan, Filter,
    // Project, even another Join down the road) runs through its normal
    // path.
    let lhs = engine.execute(left)?.into_record_batch();
    let rhs = engine.execute(right)?.into_record_batch();

    match join_type {
        JoinType::Inner => execute_inner_join(lhs, rhs, on, output_schema),
        JoinType::LeftOuter | JoinType::RightOuter | JoinType::FullOuter => {
            execute_outer_join(lhs, rhs, *join_type, on, output_schema)
        }
        JoinType::Cross => execute_cross_join(lhs, rhs, output_schema),
    }
}

// ---------- INNER ---------------------------------------------------------

fn execute_inner_join(
    lhs: RecordBatch,
    rhs: RecordBatch,
    on: &[(Expr, Expr)],
    output_schema: &Schema,
) -> BoltResult<QueryHandle> {
    if on.is_empty() {
        return Err(BoltError::Plan(
            "INNER JOIN requires at least one equi-join predicate; \
             the parser should have rejected this upstream"
                .into(),
        ));
    }

    let (left_keys, right_keys) = split_keys(on)?;
    let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;

    // Either side empty -> empty INNER output.
    if lhs.num_rows() == 0 || rhs.num_rows() == 0 {
        return empty_output(arrow_schema);
    }

    // Smaller side builds. Pair buffer always records `(build, probe)`;
    // `build_is_left` lets us re-orient that when materialising.
    let build_is_left = lhs.num_rows() <= rhs.num_rows();
    let (build_batch, build_keys, probe_batch, probe_keys) = if build_is_left {
        (&lhs, &left_keys, &rhs, &right_keys)
    } else {
        (&rhs, &right_keys, &lhs, &left_keys)
    };

    let build_idx = lookup_columns(build_batch, build_keys)?;
    let probe_idx = lookup_columns(probe_batch, probe_keys)?;
    check_key_dtypes(build_batch, &build_idx, probe_batch, &probe_idx, "INNER JOIN")?;

    // GPU fast path — single equi-key, Int32/Int64, both sides large enough,
    // no NULLs, unique build keys. On any gate miss this returns Ok(None)
    // and we fall through to the host hash join.
    if let Some(out) = try_gpu_inner_join(
        &lhs,
        &rhs,
        build_is_left,
        build_batch,
        &build_idx,
        probe_batch,
        &probe_idx,
        arrow_schema.clone(),
    )? {
        return Ok(QueryHandle::from_record_batch(out));
    }

    // Build phase. NULL-key build rows are silently skipped.
    let map = build_hash_map(build_batch, &build_idx)?;
    if map.is_empty() {
        return empty_output(arrow_schema);
    }

    // Probe phase.
    let mut build_pairs: Vec<u32> = Vec::new();
    let mut probe_pairs: Vec<u32> = Vec::new();
    for row in 0..probe_batch.num_rows() {
        let Some(key) = extract_key(probe_batch, &probe_idx, row)? else {
            continue;
        };
        if let Some(matches) = map.get(&key) {
            let probe_u32 = row_to_u32(row)?;
            for &b in matches {
                build_pairs.push(b);
                probe_pairs.push(probe_u32);
            }
        }
    }

    if build_pairs.is_empty() {
        return empty_output(arrow_schema);
    }

    // Re-orient (build, probe) -> (left, right) so output column order
    // matches `output_schema` (which is left ++ right).
    let (left_pairs, right_pairs) = if build_is_left {
        (build_pairs, probe_pairs)
    } else {
        (probe_pairs, build_pairs)
    };

    materialise(
        &lhs,
        &rhs,
        &Indices::Some(left_pairs),
        &Indices::Some(right_pairs),
        arrow_schema,
    )
}

// ---------- LEFT / RIGHT / FULL ------------------------------------------

/// LEFT / RIGHT / FULL OUTER join via host-side hash join with explicit
/// unmatched-row tracking.
///
/// Build side is *fixed* by the join kind (not size-driven) so we can
/// track unmatched rows symmetrically without losing track of which side
/// is the preserved one:
///
///   * LEFT  → build = right, probe = left. Every probe (left) row emits
///     at least once; unmatched probe rows emit with right-side NULL.
///   * RIGHT → build = left,  probe = right. Symmetric to LEFT.
///   * FULL  → same as LEFT for the probe pass, then a second pass over
///     the build side emits build rows whose key was never matched.
fn execute_outer_join(
    lhs: RecordBatch,
    rhs: RecordBatch,
    join_type: JoinType,
    on: &[(Expr, Expr)],
    output_schema: &Schema,
) -> BoltResult<QueryHandle> {
    if on.is_empty() {
        let kind = match join_type {
            JoinType::LeftOuter => "LEFT OUTER",
            JoinType::RightOuter => "RIGHT OUTER",
            JoinType::FullOuter => "FULL OUTER",
            _ => "OUTER",
        };
        return Err(BoltError::Plan(format!(
            "{kind} JOIN requires at least one equi-join predicate; \
             the parser should have rejected this upstream"
        )));
    }
    let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
    let (left_keys, right_keys) = split_keys(on)?;

    // Decide which side builds. The probe side is the "preserved" side
    // for LEFT and FULL (every probe row emits); for RIGHT it's the
    // other way around. We always make the probe = preserved side.
    let build_is_left = matches!(join_type, JoinType::RightOuter);
    let (build_batch, build_keys, probe_batch, probe_keys) = if build_is_left {
        (&lhs, &left_keys, &rhs, &right_keys)
    } else {
        (&rhs, &right_keys, &lhs, &left_keys)
    };

    // Track whether either side is preserved separately so empty-batch
    // shortcuts are correct for every variant.
    let preserve_probe = matches!(
        join_type,
        JoinType::LeftOuter | JoinType::RightOuter | JoinType::FullOuter
    );
    let preserve_build = matches!(join_type, JoinType::FullOuter);

    // Edge case: one side empty.
    if probe_batch.num_rows() == 0 && build_batch.num_rows() == 0 {
        return empty_output(arrow_schema);
    }
    if probe_batch.num_rows() == 0 {
        // Only the preserved-build side (FULL) can emit anything.
        if preserve_build {
            let n = build_batch.num_rows();
            let build_idx: Vec<u32> = (0..u32::try_from(n).map_err(|_| {
                BoltError::Other(format!(
                    "OUTER JOIN row count {n} exceeds u32 take-indices limit"
                ))
            })?)
                .collect();
            let probe_nulls: Vec<Option<u32>> = vec![None; build_idx.len()];
            let (left_idx, right_idx) = orient_indices(
                build_is_left,
                Indices::Some(build_idx),
                Indices::Nulls(probe_nulls),
            );
            return materialise(&lhs, &rhs, &left_idx, &right_idx, arrow_schema);
        }
        return empty_output(arrow_schema);
    }
    if build_batch.num_rows() == 0 {
        // Only the preserved-probe side (LEFT/RIGHT/FULL) can emit.
        if preserve_probe {
            let n = probe_batch.num_rows();
            let probe_idx: Vec<u32> = (0..u32::try_from(n).map_err(|_| {
                BoltError::Other(format!(
                    "OUTER JOIN row count {n} exceeds u32 take-indices limit"
                ))
            })?)
                .collect();
            let build_nulls: Vec<Option<u32>> = vec![None; probe_idx.len()];
            let (left_idx, right_idx) = orient_indices(
                build_is_left,
                Indices::Nulls(build_nulls),
                Indices::Some(probe_idx),
            );
            return materialise(&lhs, &rhs, &left_idx, &right_idx, arrow_schema);
        }
        return empty_output(arrow_schema);
    }

    let build_idx = lookup_columns(build_batch, build_keys)?;
    let probe_idx = lookup_columns(probe_batch, probe_keys)?;
    check_key_dtypes(build_batch, &build_idx, probe_batch, &probe_idx, "OUTER JOIN")?;

    // Stage-2 GPU OUTER fast path. Gate-misses return Ok(None); kernel
    // failures fall through to the host path with a debug log.
    if let Some(out) = try_gpu_outer_join(
        &lhs,
        &rhs,
        join_type,
        build_is_left,
        build_batch,
        &build_idx,
        probe_batch,
        &probe_idx,
        arrow_schema.clone(),
    )? {
        return Ok(QueryHandle::from_record_batch(out));
    }

    let map = build_hash_map(build_batch, &build_idx)?;

    // First pass: probe side drives matches + NULL-padded unmatched.
    let mut build_pairs: Vec<Option<u32>> = Vec::new();
    let mut probe_pairs: Vec<Option<u32>> = Vec::new();
    // For FULL, we need to know which *build rows* were touched so the
    // post-pass can emit the rest.
    let mut build_matched: Vec<bool> = if preserve_build {
        vec![false; build_batch.num_rows()]
    } else {
        Vec::new()
    };

    for row in 0..probe_batch.num_rows() {
        let probe_u32 = row_to_u32(row)?;
        let key_opt = extract_key(probe_batch, &probe_idx, row)?;
        let matched = match key_opt {
            // SQL NULL keys never match. For preserved-probe joins the
            // probe row still emits once with the build side NULL-padded.
            None => None,
            Some(key) => map.get(&key),
        };
        match matched {
            Some(matches) if !matches.is_empty() => {
                for &b in matches {
                    build_pairs.push(Some(b));
                    probe_pairs.push(Some(probe_u32));
                    if preserve_build {
                        build_matched[b as usize] = true;
                    }
                }
            }
            _ => {
                if preserve_probe {
                    build_pairs.push(None);
                    probe_pairs.push(Some(probe_u32));
                }
                // INNER falls through here, but we wouldn't be in this
                // function for INNER. Still — covered.
            }
        }
    }

    // Second pass for FULL: emit unmatched build rows.
    if preserve_build {
        for (b, &matched) in build_matched.iter().enumerate() {
            if !matched {
                let bu = row_to_u32(b)?;
                build_pairs.push(Some(bu));
                probe_pairs.push(None);
            }
        }
    }

    if build_pairs.is_empty() {
        return empty_output(arrow_schema);
    }

    // Re-orient (build, probe) → (left, right) for materialisation.
    let (left_pairs, right_pairs) = orient_indices(
        build_is_left,
        Indices::Nulls(build_pairs),
        Indices::Nulls(probe_pairs),
    );
    materialise(&lhs, &rhs, &left_pairs, &right_pairs, arrow_schema)
}

// ---------- CROSS --------------------------------------------------------

fn execute_cross_join(
    lhs: RecordBatch,
    rhs: RecordBatch,
    output_schema: &Schema,
) -> BoltResult<QueryHandle> {
    let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
    let n_left = lhs.num_rows() as u64;
    let n_right = rhs.num_rows() as u64;

    if n_left == 0 || n_right == 0 {
        return empty_output(arrow_schema);
    }

    // Cartesian explosion cap. `arrow::compute::take` takes a
    // `UInt32Array`, so even on a 64-bit host the index space is bounded
    // by u32::MAX. Anything larger is rejected with a clear error.
    let total = n_left
        .checked_mul(n_right)
        .ok_or_else(|| {
            BoltError::Plan(format!(
                "CROSS JOIN cartesian product overflows u64: {n_left} × {n_right}"
            ))
        })?;
    if total > MAX_CROSS_ROWS {
        return Err(BoltError::Plan(format!(
            "CROSS JOIN cartesian product is too large: {n_left} × {n_right} = {total} rows \
             exceeds the {MAX_CROSS_ROWS}-row limit (arrow take-indices use u32). \
             Rewrite the query to add a join predicate or filter."
        )));
    }

    // Build paired indices: for each left row r_l, emit every right row
    // r_r in 0..n_right. That gives left = [0;n_right]+[1;n_right]+...,
    // right = (0..n_right).repeat(n_left).
    let total = total as usize;
    let mut left_pairs: Vec<u32> = Vec::with_capacity(total);
    let mut right_pairs: Vec<u32> = Vec::with_capacity(total);
    for l in 0..lhs.num_rows() {
        let l_u32 = row_to_u32(l)?;
        for r in 0..rhs.num_rows() {
            left_pairs.push(l_u32);
            right_pairs.push(row_to_u32(r)?);
        }
    }
    materialise(
        &lhs,
        &rhs,
        &Indices::Some(left_pairs),
        &Indices::Some(right_pairs),
        arrow_schema,
    )
}

// ---------- shared helpers ----------------------------------------------

/// Index buffer for one side of the join's output. `Some(v)` is a plain
/// u32 array (no nulls); `Nulls(v)` carries an optional u32 per row so
/// unmatched-row slots become NULL via `arrow::compute::take`'s null
/// handling.
enum Indices {
    Some(Vec<u32>),
    Nulls(Vec<Option<u32>>),
}

/// Pick out (left_keys, right_keys) from the `on` pair list. Both sides
/// must be bare column references; the parser strips any `table.`
/// qualifier in `lower_join_side`, so anything else here is a bug.
fn split_keys(on: &[(Expr, Expr)]) -> BoltResult<(Vec<String>, Vec<String>)> {
    let mut left_keys: Vec<String> = Vec::with_capacity(on.len());
    let mut right_keys: Vec<String> = Vec::with_capacity(on.len());
    for (li, ri) in on {
        left_keys.push(column_name(li, "left")?);
        right_keys.push(column_name(ri, "right")?);
    }
    Ok((left_keys, right_keys))
}

/// Resolve key column ordinals and verify both sides agree on dtype.
fn check_key_dtypes(
    build_batch: &RecordBatch,
    build_idx: &[usize],
    probe_batch: &RecordBatch,
    probe_idx: &[usize],
    kind: &str,
) -> BoltResult<()> {
    for (b, p) in build_idx.iter().zip(probe_idx.iter()) {
        let bdt = build_batch.column(*b).data_type();
        let pdt = probe_batch.column(*p).data_type();
        if bdt != pdt {
            return Err(BoltError::Plan(format!(
                "{kind} key dtype mismatch: build side {bdt:?}, probe side {pdt:?}; \
                 cross-dtype equi-join is not yet supported"
            )));
        }
    }
    Ok(())
}

/// Build the `JoinKey -> Vec<row>` hash map. NULL-key rows are skipped.
fn build_hash_map(
    build_batch: &RecordBatch,
    build_idx: &[usize],
) -> BoltResult<HashMap<JoinKey, Vec<u32>>> {
    let mut map: HashMap<JoinKey, Vec<u32>> = HashMap::with_capacity(build_batch.num_rows());
    for row in 0..build_batch.num_rows() {
        if let Some(key) = extract_key(build_batch, build_idx, row)? {
            map.entry(key).or_default().push(row_to_u32(row)?);
        }
    }
    Ok(map)
}

/// Re-orient (build, probe) pairs into (left, right). The Indices enum
/// keeps the null-vs-dense distinction intact through the swap.
fn orient_indices(
    build_is_left: bool,
    build: Indices,
    probe: Indices,
) -> (Indices, Indices) {
    if build_is_left {
        (build, probe)
    } else {
        (probe, build)
    }
}

/// Materialise the output batch by `take`'ing each input column with the
/// per-side indices array.
fn materialise(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    left_idx: &Indices,
    right_idx: &Indices,
    arrow_schema: Arc<ArrowSchema>,
) -> BoltResult<QueryHandle> {
    let left_arr = clone_indices(left_idx);
    let right_arr = clone_indices(right_idx);

    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &left_arr, None).map_err(arrow_err)?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &right_arr, None).map_err(arrow_err)?,
        );
    }

    let out = RecordBatch::try_new(arrow_schema, output_cols).map_err(arrow_err)?;
    Ok(QueryHandle::from_record_batch(out))
}

/// Clone an Indices into a fresh UInt32Array (`arrow::compute::take`
/// borrows the indices, but we may need the same indices for multiple
/// columns, so we materialise once).
fn clone_indices(idx: &Indices) -> UInt32Array {
    match idx {
        Indices::Some(v) => UInt32Array::from(v.clone()),
        Indices::Nulls(v) => UInt32Array::from(v.clone()),
    }
}

/// Build an empty RecordBatch with the given Arrow schema. Used for the
/// "produced zero rows" early-return paths.
fn empty_output(arrow_schema: Arc<ArrowSchema>) -> BoltResult<QueryHandle> {
    let cols: Vec<ArrayRef> = arrow_schema
        .fields()
        .iter()
        .map(|f| empty_array_for_dtype(f.data_type()))
        .collect::<BoltResult<Vec<_>>>()?;
    let out = RecordBatch::try_new(arrow_schema, cols).map_err(arrow_err)?;
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
/// micro-opt left for the 0.4 GPU port.
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
                    "JOIN: unsupported key dtype {other:?}"
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
            BoltError::Plan(format!("JOIN: key column '{n}' not in batch: {e}"))
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
            "JOIN {side}-side key must be a column reference, got {other:?}; \
             computed join keys are a 0.4 target"
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
                "JOIN: unsupported output dtype {other:?}"
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
            "JOIN row index {row} exceeds the u32 take-indices limit ({})",
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

// ---------- GPU INNER fast path -----------------------------------------

/// Try the GPU INNER-join fast path. Returns:
///
/// * `Ok(Some(batch))` — every gate passed, the GPU ran the join, and `batch`
///   is the result.
/// * `Ok(None)`        — some gate didn't match; caller falls through to the
///   host hash join (which is the correctness fallback for everything the
///   GPU path can't yet handle).
/// * `Err(e)`          — hard GPU error (kernel launch failure, OOM, etc.).
///   We deliberately surface these — they indicate a CUDA-layer bug, not a
///   "gate miss".
///
/// Two layered paths share one entry point:
///
/// * **Stage-1 fast path** — single Int32/Int64 key, unique build keys, no
///   NULLs, ≥ 1024 rows / side. Routes through the byte-stable Stage-1
///   build+probe kernels.
/// * **Stage-2 generalised path** — multi-key (TwoI32) + bool/float keys
///   + duplicate build keys + collision-list kernels. Activates only when
///   Stage-1 declines but Stage-2 still applies.
///
/// CROSS still falls through to host (no equi-predicate).
fn try_gpu_inner_join(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    build_is_left: bool,
    build_batch: &RecordBatch,
    build_idx: &[usize],
    probe_batch: &RecordBatch,
    probe_idx: &[usize],
    arrow_schema: Arc<ArrowSchema>,
) -> BoltResult<Option<RecordBatch>> {
    // ---- Shared gates (apply to Stage 1 + Stage 2) ----

    // Gate A: equal arity on both sides (always true if the planner is
    // honoured; cheap sanity check).
    if build_idx.len() != probe_idx.len() || build_idx.is_empty() {
        return Ok(None);
    }

    // Gate B: minimum row counts. Sub-1k joins are host-bound by the
    // JIT-compile + h2d round trip.
    let n_build = build_batch.num_rows();
    let n_probe = probe_batch.num_rows();
    if n_build < crate::exec::gpu_join::GPU_JOIN_MIN_ROWS
        || n_probe < crate::exec::gpu_join::GPU_JOIN_MIN_ROWS
    {
        return Ok(None);
    }

    // Gate C: no NULLs in the key columns. The kernel treats every i64 as
    // a real key; NULLs would either collide with the sentinel or produce
    // false matches.
    for &b in build_idx {
        if build_batch.column(b).null_count() > 0 {
            return Ok(None);
        }
    }
    for &p in probe_idx {
        if probe_batch.column(p).null_count() > 0 {
            return Ok(None);
        }
    }

    // ---- Stage 1: single Int32/Int64 + unique build keys ----
    if build_idx.len() == 1 {
        let b_key_idx = build_idx[0];
        let p_key_idx = probe_idx[0];
        let arrow_dtype = build_batch.column(b_key_idx).data_type();
        let dtype_s1 = match arrow_dtype {
            ArrowDataType::Int32 => Some(crate::plan::logical_plan::DataType::Int32),
            ArrowDataType::Int64 => Some(crate::plan::logical_plan::DataType::Int64),
            _ => None,
        };
        if let Some(dtype) = dtype_s1 {
            let b_key_col = build_batch.column(b_key_idx);
            // Stage 1 needs unique build keys.
            if build_keys_are_unique(b_key_col, dtype) {
                match crate::exec::gpu_join::execute_inner_join_on_gpu(
                    lhs,
                    rhs,
                    build_is_left,
                    b_key_idx,
                    p_key_idx,
                    dtype,
                    arrow_schema.clone(),
                ) {
                    Ok(batch) => return Ok(Some(batch)),
                    Err(e) => {
                        log::debug!(
                            "gpu_join: Stage-1 fast path declined ({e}); \
                             trying Stage 2"
                        );
                        // Fall through to Stage 2.
                    }
                }
            }
        }
    }

    // ---- Stage 2: multi-key + bool/float + duplicate build keys ----
    let shape = match gpu_key_shape_for(build_batch, build_idx) {
        Some(s) => s,
        None => return Ok(None),
    };
    if !shape.is_exact_in_i64() {
        // Lossy fold — host hash-join takes over to avoid false matches.
        return Ok(None);
    }
    // Both sides must have matching dtypes per column (sanity — the planner
    // already enforced this at the schema level).
    for (b, p) in build_idx.iter().zip(probe_idx.iter()) {
        if build_batch.column(*b).data_type() != probe_batch.column(*p).data_type() {
            return Ok(None);
        }
    }

    match crate::exec::gpu_join::execute_inner_join_on_gpu_with_shape(
        lhs,
        rhs,
        build_is_left,
        build_idx,
        probe_idx,
        shape,
        arrow_schema,
    ) {
        Ok(batch) => Ok(Some(batch)),
        Err(e) => {
            log::debug!("gpu_join: Stage-2 path declined ({e}); falling back to host");
            Ok(None)
        }
    }
}

/// Try the Stage-2 GPU OUTER-join path. Returns the same `Ok(Some(_)) /
/// Ok(None) / Err(_)` contract as `try_gpu_inner_join`.
///
/// Gates (all must hold):
///   * Equi-join only (caller already verified non-empty `on`).
///   * `KeyShape::is_exact_in_i64()` (no lossy fold). Multi-i64 / multi-i32
///     fall through to the host path.
///   * Both sides ≥ `GPU_JOIN_MIN_ROWS` rows.
///   * No NULLs in any key column (NULL keys never match in SQL; the
///     preserved-side rows for those still need to surface via the host
///     path because the kernel-side encoding can't distinguish NULL from a
///     legitimate sentinel-adjacent value).
///
/// For LEFT outer the GPU build side is the RIGHT table (probe = LEFT,
/// preserved). For RIGHT outer the GPU build side is the LEFT table
/// (probe = RIGHT, preserved). FULL emits both sides — same orientation as
/// LEFT, plus a second-pass kernel over the build (= right) bitmap.
#[allow(clippy::too_many_arguments)]
fn try_gpu_outer_join(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    join_type: JoinType,
    build_is_left: bool,
    build_batch: &RecordBatch,
    build_idx: &[usize],
    probe_batch: &RecordBatch,
    probe_idx: &[usize],
    arrow_schema: Arc<ArrowSchema>,
) -> BoltResult<Option<RecordBatch>> {
    if build_idx.len() != probe_idx.len() || build_idx.is_empty() {
        return Ok(None);
    }

    let n_build = build_batch.num_rows();
    let n_probe = probe_batch.num_rows();
    if n_build < crate::exec::gpu_join::GPU_JOIN_MIN_ROWS
        || n_probe < crate::exec::gpu_join::GPU_JOIN_MIN_ROWS
    {
        return Ok(None);
    }

    for &b in build_idx {
        if build_batch.column(b).null_count() > 0 {
            return Ok(None);
        }
    }
    for &p in probe_idx {
        if probe_batch.column(p).null_count() > 0 {
            return Ok(None);
        }
    }
    for (b, p) in build_idx.iter().zip(probe_idx.iter()) {
        if build_batch.column(*b).data_type() != probe_batch.column(*p).data_type() {
            return Ok(None);
        }
    }

    let shape = match gpu_key_shape_for(build_batch, build_idx) {
        Some(s) => s,
        None => return Ok(None),
    };
    if !shape.is_exact_in_i64() {
        return Ok(None);
    }

    // Translate join_type into emit flags. preserve_probe = LEFT/RIGHT/FULL
    // (probe is always the preserved side per build_is_left choice).
    // preserve_build = FULL only (second-pass kernel walks unmatched build).
    let emit_unmatched_probe = matches!(
        join_type,
        JoinType::LeftOuter | JoinType::RightOuter | JoinType::FullOuter
    );
    let emit_unmatched_build = matches!(join_type, JoinType::FullOuter);

    match crate::exec::gpu_join::execute_outer_join_on_gpu(
        lhs,
        rhs,
        build_is_left,
        build_idx,
        probe_idx,
        shape,
        emit_unmatched_probe,
        emit_unmatched_build,
        arrow_schema,
    ) {
        Ok(batch) => Ok(Some(batch)),
        Err(e) => {
            log::debug!("gpu_join: outer-join path declined ({e}); falling back to host");
            Ok(None)
        }
    }
}

/// Map the build-side key columns to a [`crate::jit::hash_join_kernel::KeyShape`]
/// that the GPU host-side encoder understands. Returns `None` if no shape
/// matches (e.g. Utf8 keys, or a Float32 column mixed with an Int64 column).
fn gpu_key_shape_for(
    batch: &RecordBatch,
    indices: &[usize],
) -> Option<crate::jit::hash_join_kernel::KeyShape> {
    use crate::jit::hash_join_kernel::KeyShape;
    match indices.len() {
        1 => match batch.column(indices[0]).data_type() {
            ArrowDataType::Int32 => Some(KeyShape::SingleI32),
            ArrowDataType::Int64 => Some(KeyShape::SingleI64),
            ArrowDataType::Boolean => Some(KeyShape::SingleBool),
            ArrowDataType::Float32 => Some(KeyShape::SingleF32),
            ArrowDataType::Float64 => Some(KeyShape::SingleF64),
            _ => None,
        },
        2 => {
            let a = batch.column(indices[0]).data_type();
            let b = batch.column(indices[1]).data_type();
            match (a, b) {
                (ArrowDataType::Int32, ArrowDataType::Int32) => Some(KeyShape::TwoI32),
                (ArrowDataType::Int64, ArrowDataType::Int64) => Some(KeyShape::TwoI64),
                _ => None,
            }
        }
        n if n >= 3 && n <= u8::MAX as usize => {
            // Only support all-Int32 tuples for now; mixed dtypes fall back.
            let all_i32 = indices
                .iter()
                .all(|&i| matches!(batch.column(i).data_type(), ArrowDataType::Int32));
            if all_i32 {
                Some(KeyShape::MultiI32(n as u8))
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Quick host-side uniqueness check on the build-side key column. Returns
/// `true` if every value is distinct (the Stage 1 GPU path's invariant).
///
/// We deliberately don't surface duplicates as an error: the host hash-join
/// path handles them correctly via `HashMap<JoinKey, Vec<u32>>`, so a
/// "duplicates" finding just routes the query to that path.
fn build_keys_are_unique(
    col: &dyn arrow_array::Array,
    dtype: crate::plan::logical_plan::DataType,
) -> bool {
    use std::collections::HashSet;
    let n = col.len();
    match dtype {
        crate::plan::logical_plan::DataType::Int32 => {
            let arr = match col.as_any().downcast_ref::<Int32Array>() {
                Some(a) => a,
                None => return false,
            };
            let mut seen: HashSet<i32> = HashSet::with_capacity(n);
            for v in arr.values().iter() {
                if !seen.insert(*v) {
                    return false;
                }
            }
            true
        }
        crate::plan::logical_plan::DataType::Int64 => {
            let arr = match col.as_any().downcast_ref::<Int64Array>() {
                Some(a) => a,
                None => return false,
            };
            let mut seen: HashSet<i64> = HashSet::with_capacity(n);
            for v in arr.values().iter() {
                if !seen.insert(*v) {
                    return false;
                }
            }
            true
        }
        // Other dtypes don't reach this function — gate 2 above rejects them.
        _ => false,
    }
}
