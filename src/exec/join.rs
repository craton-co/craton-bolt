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
//!        a `HashMap<JoinKey, BuildSlot>` (the slot stores the first row
//!        index inline and promotes to a heap `Vec<u32>` only when a
//!        second value arrives — see `BuildSlot` below).
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
//!
//! Float-key semantics (review C12): `+0.0` and `-0.0` compare equal per
//! SQL/IEEE (`+0.0 == -0.0`), so this executor CANONICALISES `-0.0` to
//! `+0.0` before building the key (see `canonicalise_f32`/`canonicalise_f64`
//! and `extract_key`). NaN bit patterns are LEFT AS-IS (`NaN != NaN`;
//! build-side NaN rows therefore never match a probe-side NaN row,
//! matching DuckDB). The same canonicalisation is applied in `distinct.rs`
//! and `groupby.rs` so DISTINCT / GROUP BY / JOIN agree on one
//! equivalence relation for floats.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::types::{Int32Type as ArrowInt32Type, Int64Type as ArrowInt64Type};
use arrow_array::{
    Array, ArrayRef, BooleanArray, DictionaryArray, Float32Array, Float64Array, Int32Array,
    Int64Array, RecordBatch, StringArray, UInt32Array,
};
use arrow_schema::{DataType as ArrowDataType, Schema as ArrowSchema};

use crate::error::{BoltError, BoltResult};
use crate::exec::{Engine, QueryHandle};
use crate::plan::logical_plan::{Expr, JoinType, Schema};
use crate::plan::physical_plan::PhysicalPlan;

/// Hard cap on the row count of a CROSS JOIN output. `arrow::compute::take`
/// requires a `UInt32Array` of indices, so any result larger than
/// `u32::MAX` rows cannot be materialised through that API. We reject at
/// build time rather than overflow silently.
const MAX_CROSS_ROWS: u64 = u32::MAX as u64;

/// Hard cap on the inner-side row count of a non-equi (nested-loop) join.
/// The nested-loop fallback walks the cartesian product chunk-by-chunk and
/// applies a host-side predicate; cost is `O(outer × inner)` time but only
/// `O(chunk × inner + output)` peak memory (see
/// [`execute_nested_loop_join`]). We cap the *inner* side at 1024 rows so
/// the per-outer-row scan is cheap and surface a clear "rewrite as equi if
/// possible" hint past that.
///
/// Sized for the v0.6 stretch goal: small-side non-equi joins (e.g. a
/// dimension table of buckets / ranges against a fact table) work cleanly,
/// while accidental cartesian explosions are caught early. The cap is a
/// `const` so it's easy to find and tune.
pub(crate) const MAX_NESTED_LOOP_INNER_ROWS: usize = 1024;

/// Target ceiling on the number of cartesian cells (`chunk × inner`)
/// materialised at once inside the nested-loop join's streaming loop.
///
/// The non-equi executor (EXEC-M3 fix) no longer materialises the whole
/// `outer × inner` product before filtering — instead it slices the larger
/// side into chunks sized so each chunk's `chunk × inner` cross stays at or
/// under this bound, filters each chunk, and concatenates the (small)
/// matched results. This keeps peak memory at `O(chunk × inner + output)`
/// rather than `O(outer × inner)`, so a large-fact / tiny-dim non-equi join
/// succeeds instead of tripping [`MAX_CROSS_ROWS`].
///
/// `1 << 20` (≈1M cells) is small enough to stay well under
/// [`MAX_CROSS_ROWS`] for any `inner ≤ MAX_NESTED_LOOP_INNER_ROWS`, yet
/// large enough that the per-chunk overhead (one cross + one filter call)
/// is amortised across many outer rows.
const NESTED_LOOP_CHUNK_CELLS: u64 = 1 << 20;

/// Execute a JOIN. Dispatches per `join_type` to one of:
///   * `execute_inner_join` — existing INNER path (smaller side builds).
///   * `execute_outer_join` — LEFT / RIGHT / FULL (build side is fixed
///      by the join kind so unmatched-row tracking is straightforward).
///   * `execute_cross_join` — cartesian product with a row-count cap.
///   * `execute_nested_loop_join` — v0.6 non-equi fallback. Activated when
///     `filter` is `Some(_)`: the larger side is STREAMED in chunks (the
///     EXEC-M3 fix), each chunk crossed against the held-whole smaller side
///     and filtered host-side via [`crate::exec::filter::execute_filter`],
///     then the matched chunks are concatenated. The inner (smaller) side
///     is capped at [`MAX_NESTED_LOOP_INNER_ROWS`]; peak memory is bounded
///     by `chunk × inner + output`, NOT `outer × inner`, so a large-fact /
///     tiny-dim non-equi join no longer trips the `MAX_CROSS_ROWS` cap.
///
/// `output_schema` is the disambiguated combined schema produced by
/// `join_combined_schema` and stored on `PhysicalPlan::Join`; the engine
/// passes it through so this executor doesn't have to recompute it.
///
/// `filter` is the optional non-equi residual predicate carried on
/// `PhysicalPlan::Join`. `None` is the equi/CROSS fast path.
pub fn execute_join(
    left: &PhysicalPlan,
    right: &PhysicalPlan,
    join_type: &JoinType,
    on: &[(Expr, Expr)],
    filter: Option<&Expr>,
    output_schema: &Schema,
    engine: &Engine,
) -> BoltResult<QueryHandle> {
    // Execute both children up front. Mirrors the UNION dispatch in
    // `Engine::execute`: every operator under a Join (Scan, Filter,
    // Project, even another Join down the road) runs through its normal
    // path.
    let lhs = engine.execute(left)?.into_record_batch();
    let rhs = engine.execute(right)?.into_record_batch();

    // Non-equi fast-bail: any filter present routes through the nested-loop
    // executor regardless of join_type, since the existing equi-join
    // executors can't honour the residual predicate. v0.6 supports the
    // INNER and CROSS shapes for non-equi; LEFT/RIGHT/FULL with a non-equi
    // predicate is a future extension (current behaviour: reject with a
    // clear message rather than silently produce wrong results).
    if let Some(pred) = filter {
        return execute_nested_loop_join(lhs, rhs, *join_type, on, pred, output_schema);
    }

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
    check_key_dtypes(
        build_batch,
        &build_idx,
        probe_batch,
        &probe_idx,
        "INNER JOIN",
    )?;

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
            for &b in matches.as_slice() {
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
    check_key_dtypes(
        build_batch,
        &build_idx,
        probe_batch,
        &probe_idx,
        "OUTER JOIN",
    )?;

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
            Some(matches) if !matches.as_slice().is_empty() => {
                for &b in matches.as_slice() {
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
    let total = n_left.checked_mul(n_right).ok_or_else(|| {
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

    // Stage-3 GPU CROSS fast path. Gates:
    //   * Total cells in [CROSS_JOIN_GPU_MIN_CELLS, CROSS_JOIN_GPU_CELL_CAP).
    //   * GPU available (errors fall through to host).
    if total >= crate::exec::gpu_join::CROSS_JOIN_GPU_MIN_CELLS
        && total < crate::exec::gpu_join::CROSS_JOIN_GPU_CELL_CAP
    {
        match crate::exec::gpu_join::execute_cross_join_on_gpu(&lhs, &rhs, arrow_schema.clone()) {
            Ok(batch) => return Ok(QueryHandle::from_record_batch(batch)),
            Err(e) => {
                log::debug!("gpu_join: CROSS GPU path declined ({e}); falling back to host");
            }
        }
    }

    // Host fallback: pair indices by row-major iteration.
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

// ---------- NESTED-LOOP (v0.6 non-equi fallback) -------------------------

/// Nested-loop join with a host-side residual predicate.
///
/// Strategy:
///   1. Cap the *inner* (smaller) side at [`MAX_NESTED_LOOP_INNER_ROWS`]
///      and refuse the join past that limit with a clear hint that the
///      user should rewrite as an equi-join if possible. The cap is on the
///      smaller side because the cartesian explosion is `outer × inner`;
///      capping the smaller side bounds the host-side work without
///      arbitrarily limiting the larger fact-table side.
///   2. STREAM the cartesian product in chunks instead of materialising the
///      whole `outer × inner` batch up front (EXEC-M3 fix). The larger side
///      is sliced into row-blocks sized so each block's `block × inner`
///      cross stays at or under [`NESTED_LOOP_CHUNK_CELLS`]; the smaller
///      (≤1024-row) side is held whole. For each block we build *only that
///      block's* cross via `execute_cross_join` (which produces a
///      `RecordBatch` over the join's combined output schema, with
///      right-side rename rules applied), filter it (step 3), and push the
///      matched rows into an accumulator. This bounds peak memory at
///      `O(block × inner + output)` rather than `O(outer × inner)`, so a
///      large-fact / tiny-dim non-equi join no longer trips the
///      [`MAX_CROSS_ROWS`] cap on the *outer* size. (That cap still guards
///      the genuine CROSS-JOIN path verbatim — see `execute_cross_join` —
///      and each per-block cross trivially satisfies it because
///      `block × inner ≤ NESTED_LOOP_CHUNK_CELLS ≪ MAX_CROSS_ROWS`.)
///   3. Apply `predicate` to each per-block cartesian batch via
///      [`crate::exec::filter::execute_filter`], which evaluates the
///      expression against the combined-schema columns and drops rows
///      where the predicate is `false` or NULL (standard SQL WHERE
///      semantics — matches the v0.6 INNER-non-equi contract). Identical
///      to the pre-fix post-cross filter, just applied per block; the
///      matched blocks are concatenated with `concat_batches`, preserving
///      output schema and column order exactly.
///   4. For OUTER variants (LEFT/RIGHT/FULL), the nested-loop path with
///      a non-equi predicate would need to track unmatched preserved-side
///      rows and emit them NULL-padded. v0.6 ships the INNER-non-equi
///      surface only; OUTER + non-equi rejects with a clear message
///      pointing at the future extension.
///
/// `on` (equi pairs) can be non-empty alongside a `filter`: the SQL
/// frontend's `lower_join_on` extracts equi conjuncts into `on` and routes
/// the remaining non-equi conjuncts into `filter` (e.g.
/// `t1.a = t2.a AND t1.b > t2.b` produces `on = [(a, a)]` and
/// `filter = Some(b > right.b)`). For correctness this executor folds
/// every equi pair into the residual predicate as a `left = right`
/// conjunct via [`fold_equi_into_predicate`] before evaluating the whole
/// thing host-side — that way both the equi and non-equi constraints are
/// enforced in one pass. A future optimisation could run the equi join
/// first (hash) and then post-filter, but the mixed shape is rare enough
/// in v0.6 that the simpler approach is preferred.
fn execute_nested_loop_join(
    lhs: RecordBatch,
    rhs: RecordBatch,
    join_type: JoinType,
    on: &[(Expr, Expr)],
    predicate: &Expr,
    output_schema: &Schema,
) -> BoltResult<QueryHandle> {
    // Production entry point: stream with the tuned chunk-cell ceiling. The
    // body lives in `execute_nested_loop_join_chunked` so tests can drive a
    // tiny `chunk_cells` and exercise the multi-chunk concatenation path
    // without building a multi-million-row fixture (behaviour is identical
    // for any positive `chunk_cells`).
    execute_nested_loop_join_chunked(
        lhs,
        rhs,
        join_type,
        on,
        predicate,
        output_schema,
        NESTED_LOOP_CHUNK_CELLS,
    )
}

fn execute_nested_loop_join_chunked(
    lhs: RecordBatch,
    rhs: RecordBatch,
    join_type: JoinType,
    on: &[(Expr, Expr)],
    predicate: &Expr,
    output_schema: &Schema,
    chunk_cells: u64,
) -> BoltResult<QueryHandle> {
    // OUTER + non-equi is a planned follow-up; surface a clear message
    // rather than silently dropping the preserved-side semantics.
    if matches!(
        join_type,
        JoinType::LeftOuter | JoinType::RightOuter | JoinType::FullOuter
    ) {
        return Err(BoltError::Plan(format!(
            "{join_type:?} JOIN with a non-equi ON predicate is not yet supported \
             in the nested-loop executor (v0.6 ships INNER non-equi only); \
             rewrite the predicate as an equi-join or use INNER + WHERE"
        )));
    }

    // Inner-side cap. We pick the *smaller* side as the inner (the side we
    // hold whole and scan per outer-block); that's the side whose row count
    // drives the per-block cross size. Pre-flight the cap before doing any
    // work so a bad shape fails fast.
    let inner_rows = lhs.num_rows().min(rhs.num_rows());
    if inner_rows > MAX_NESTED_LOOP_INNER_ROWS {
        return Err(BoltError::Plan(format!(
            "non-equi join inner side > {MAX_NESTED_LOOP_INNER_ROWS} rows; \
             rewrite as equi if possible (got smaller-side row count = {inner_rows})"
        )));
    }

    let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;

    // Empty either side ⇒ empty INNER result. (`execute_cross_join` would
    // also short-circuit, but bailing here avoids the chunk-loop setup and
    // a divide-by-zero when computing the chunk size below.)
    if lhs.num_rows() == 0 || rhs.num_rows() == 0 {
        return empty_output(arrow_schema);
    }

    // If the SQL frontend extracted equi pairs alongside a residual filter,
    // fold them into one composite predicate so they're enforced too. This
    // arm isn't reachable from v0.6's `lower_join_on` (which puts the whole
    // equality into the filter when any non-equi sibling is present), but
    // future planner improvements (e.g. extracting equi pairs even when a
    // non-equi sibling exists) need not change this executor. Computed once
    // and reused across every chunk so per-block work is just cross+filter.
    let composite: Expr;
    let pred_ref: &Expr = if on.is_empty() {
        predicate
    } else {
        composite = fold_equi_into_predicate(on, predicate);
        &composite
    };

    // Streaming (chunked) evaluation — the EXEC-M3 fix. Slice the LARGER
    // side into row-blocks sized so each block's `block × inner` cross stays
    // at or under `NESTED_LOOP_CHUNK_CELLS`, then for each block:
    //   * build only that block's cross product (left-cols ++ right-cols,
    //     preserving the combined output schema), and
    //   * filter it with `execute_filter` — identical to the pre-fix
    //     post-cross filter, just per block.
    // The matched per-block batches are concatenated at the end. Because we
    // always pass the chunk slot in its original (left, right) position to
    // `execute_cross_join`, column order is identical to the full-cross
    // path; only the row blocking changes.
    //
    // `chunk_rows` is derived from the *inner* (held-whole) side so the
    // per-block cross is bounded regardless of which physical side is the
    // larger one. `inner_rows ≥ 1` here (empty sides bailed above), and
    // `inner_rows ≤ MAX_NESTED_LOOP_INNER_ROWS`, so `chunk_rows ≥ 1` and
    // `chunk_rows × inner_rows ≤ NESTED_LOOP_CHUNK_CELLS ≪ MAX_CROSS_ROWS`.
    let chunk_rows: usize = (chunk_cells / inner_rows as u64).max(1) as usize;

    // The larger side is the one we chunk; the smaller side is held whole.
    // `lhs` and `rhs` keep their (left, right) identity throughout so the
    // cross output column order never changes.
    let chunk_left = lhs.num_rows() >= rhs.num_rows();
    let outer_rows = if chunk_left {
        lhs.num_rows()
    } else {
        rhs.num_rows()
    };

    let mut matched: Vec<RecordBatch> = Vec::new();
    let mut start = 0usize;
    while start < outer_rows {
        let len = chunk_rows.min(outer_rows - start);
        // Slice only the larger side; hold the smaller side whole. Arrow's
        // `RecordBatch::slice` is a zero-copy view, so the only allocation
        // per block is the cross materialisation itself.
        let (block_lhs, block_rhs) = if chunk_left {
            (lhs.slice(start, len), rhs.clone())
        } else {
            (lhs.clone(), rhs.slice(start, len))
        };

        // Per-block cross. `execute_cross_join` handles name disambiguation
        // and the (here trivially-satisfied) `MAX_CROSS_ROWS` cap.
        let cross_handle = execute_cross_join(block_lhs, block_rhs, output_schema)?;

        // Apply the residual predicate host-side. `execute_filter` uses
        // `expr_agg::eval_expr` against the combined-schema columns, so the
        // predicate is evaluated row-by-row in SQL three-valued logic
        // (predicate result of NULL drops the row, matching WHERE semantics).
        let filtered =
            crate::exec::filter::execute_filter(cross_handle, pred_ref)?.into_record_batch();
        if filtered.num_rows() > 0 {
            matched.push(filtered);
        }

        start += len;
    }

    if matched.is_empty() {
        return empty_output(arrow_schema);
    }

    // Concatenate the matched blocks. Every block carries the identical
    // combined schema (`arrow_schema`), so `concat_batches` simply stacks
    // them; column order is preserved exactly.
    let out = arrow::compute::concat_batches(&arrow_schema, matched.iter()).map_err(arrow_err)?;
    Ok(QueryHandle::from_record_batch(out))
}

/// Conjunct every equi pair (lowered as `l = r`) into `predicate` with AND
/// so the host-side filter can enforce both the equi pairs and the
/// residual non-equi predicate in one pass. See the dispatch logic in
/// [`execute_nested_loop_join`] for why this exists despite v0.6's planner
/// never producing mixed shapes.
fn fold_equi_into_predicate(on: &[(Expr, Expr)], predicate: &Expr) -> Expr {
    let mut acc: Expr = predicate.clone();
    for (l, r) in on.iter() {
        let eq = Expr::Binary {
            op: crate::plan::logical_plan::BinaryOp::Eq,
            left: Box::new(l.clone()),
            right: Box::new(r.clone()),
        };
        acc = Expr::Binary {
            op: crate::plan::logical_plan::BinaryOp::And,
            left: Box::new(eq),
            right: Box::new(acc),
        };
    }
    acc
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

/// One-element-inline list of u32 row indices.
///
/// Optimised for the unique-key build case where 95%+ of entries hold a
/// single row index; promotes to a heap `Vec` only when a second value
/// arrives. Saves the per-key allocation on the common path (the
/// `HashMap<JoinKey, Vec<u32>>` shape had to heap-allocate a 1-element
/// `Vec` for every distinct key, which dominated allocator traffic on
/// large unique-key builds).
enum BuildSlot {
    Inline([u32; 1]),
    Heap(Vec<u32>),
}

impl BuildSlot {
    fn new(first: u32) -> Self {
        BuildSlot::Inline([first])
    }

    fn push(&mut self, v: u32) {
        match self {
            BuildSlot::Inline([a]) => {
                let mut heap = Vec::with_capacity(2);
                heap.push(*a);
                heap.push(v);
                *self = BuildSlot::Heap(heap);
            }
            BuildSlot::Heap(vec) => vec.push(v),
        }
    }

    fn as_slice(&self) -> &[u32] {
        match self {
            BuildSlot::Inline(a) => a,
            BuildSlot::Heap(v) => v,
        }
    }

    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

/// Build the `JoinKey -> BuildSlot` hash map. NULL-key rows are skipped.
///
/// The slot is a `BuildSlot` rather than a `Vec<u32>` so the
/// unique-build-key common path (≈95% of real workloads) stores its sole
/// row index inline. Callers read row indices through `BuildSlot::as_slice`.
fn build_hash_map(
    build_batch: &RecordBatch,
    build_idx: &[usize],
) -> BoltResult<HashMap<JoinKey, BuildSlot>> {
    let mut map: HashMap<JoinKey, BuildSlot> = HashMap::with_capacity(build_batch.num_rows());
    for row in 0..build_batch.num_rows() {
        if let Some(key) = extract_key(build_batch, build_idx, row)? {
            let row_u32 = row_to_u32(row)?;
            map.entry(key)
                .and_modify(|s| s.push(row_u32))
                .or_insert_with(|| BuildSlot::new(row_u32));
        }
    }
    Ok(map)
}

/// Re-orient (build, probe) pairs into (left, right). The Indices enum
/// keeps the null-vs-dense distinction intact through the swap.
fn orient_indices(build_is_left: bool, build: Indices, probe: Indices) -> (Indices, Indices) {
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
        output_cols.push(arrow::compute::take(col.as_ref(), &left_arr, None).map_err(arrow_err)?);
    }
    for col in rhs.columns() {
        output_cols.push(arrow::compute::take(col.as_ref(), &right_arr, None).map_err(arrow_err)?);
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
/// CANONICALISED bit pattern (see `canonicalise_f32`/`canonicalise_f64`
/// below — `-0.0` is mapped to `+0.0` before `.to_bits()`). NaN bit
/// patterns are preserved verbatim, so equality of NaN keys is bit-wise:
/// two `NaN` join rows whose payloads differ never match each other,
/// matching the SQL standard's `NaN != NaN` rule (and DuckDB).
///
/// Utf8 keys ALWAYS take the `Utf8(Box<str>)` shape, whether the source
/// column is a raw `StringArray` or a `DictionaryArray<_, Utf8>`:
///
/// * `Utf8(Box<str>)` — the decoded string value. Raw `StringArray` rows
///   produce it directly; dict-encoded rows are first resolved through
///   their own dictionary to the string and then keyed the same way. This
///   unification is load-bearing for correctness (review F-7 analogue):
///   keying dict columns on the raw INDEX is WRONG when the build and probe
///   sides carry independent dictionaries (equal indices can map to
///   different strings, and vice-versa). `check_key_dtypes` only verifies
///   the Arrow datatype, so it cannot detect independent dictionaries — the
///   only safe key is the string value itself. `Box<str>` skips `String`'s
///   capacity field (~16 B header vs 24 B) and the typical over-allocation,
///   so it's roughly half the host-side footprint of a `String` key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum JoinKeyValue {
    I32(i32),
    I64(i64),
    F32(u32),
    F64(u64),
    Bool(bool),
    /// Raw (non-dict) Utf8 row value. `Box<str>` instead of `String` saves
    /// the capacity field + over-allocation overhead in the hot per-row
    /// path; `Box<str>` is `Eq + Hash`, so this is a drop-in swap.
    Utf8(Box<str>),
}

/// A row's join key — one [`JoinKeyValue`] per equi column.
///
/// The number of equi columns is typically 1 (and almost always 1-3), so
/// this is an inline-capable small-vector rather than a plain
/// `Vec<JoinKeyValue>`: the single-key case (`One`) stores its sole value
/// inline and NEVER heap-allocates, which is the overwhelming common path
/// for both the per-build-row and per-probe-row keys. Multi-column keys
/// spill to `Many(Vec<_>)` exactly as the old `Vec` alias did.
///
/// # Hash / Eq invariant (load-bearing)
///
/// `Hash`, `PartialEq`, and `Eq` are defined over `self.as_slice()`, the
/// ordered sequence of `JoinKeyValue`s — byte-for-byte identical to the old
/// `Vec<JoinKeyValue>` semantics:
///
///   * `Vec<T>: Hash` writes `len` then hashes each element in order; so
///     does `[T]: Hash`. Delegating `Hash` to `self.as_slice()` produces
///     the exact same hash stream as the old `Vec` key whether the value
///     sits inline (`One`) or on the heap (`Many`). In particular a
///     single-key `One(v)` hashes identically to `Many(vec![v])`.
///   * `PartialEq`/`Eq` compare the value sequences element-by-element, so
///     two keys compare equal iff their values match in order, independent
///     of `One`-vs-`Many` storage.
///
/// The variant is an internal storage detail never observed by Hash/Eq, so
/// the join's key equivalence relation is unchanged.
#[derive(Debug, Clone)]
enum JoinKey {
    One(JoinKeyValue),
    Many(Vec<JoinKeyValue>),
}

impl JoinKey {
    /// The ordered key values. All of `Hash`/`Eq` go through this, so the
    /// inline-vs-heap split is invisible to the key relation.
    #[inline]
    fn as_slice(&self) -> &[JoinKeyValue] {
        match self {
            JoinKey::One(v) => std::slice::from_ref(v),
            JoinKey::Many(vs) => vs.as_slice(),
        }
    }

    /// Number of key columns. Mirrors `Vec::len`.
    #[inline]
    #[allow(dead_code)]
    fn len(&self) -> usize {
        self.as_slice().len()
    }
}

impl From<Vec<JoinKeyValue>> for JoinKey {
    /// Build a `JoinKey` from a value vector, keeping the single-key case
    /// inline (no heap allocation). A 1-element vec collapses to `One`; the
    /// vec's backing allocation is dropped. The hash/eq relation is
    /// unaffected (see the type doc-comment).
    #[inline]
    fn from(mut vs: Vec<JoinKeyValue>) -> Self {
        if vs.len() == 1 {
            // `pop` moves the sole element out; the empty Vec is dropped.
            JoinKey::One(vs.pop().unwrap())
        } else {
            JoinKey::Many(vs)
        }
    }
}

impl std::ops::Index<usize> for JoinKey {
    type Output = JoinKeyValue;
    #[inline]
    fn index(&self, i: usize) -> &JoinKeyValue {
        &self.as_slice()[i]
    }
}

impl PartialEq for JoinKey {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for JoinKey {}

impl std::hash::Hash for JoinKey {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Delegates to `[JoinKeyValue]: Hash` (writes len then each element)
        // — byte-for-byte identical to the old `Vec<JoinKeyValue>: Hash`.
        self.as_slice().hash(state);
    }
}

/// Pull the (build_idx, ...) tuple of values for `row` out of `batch`.
/// Returns `Ok(None)` if any key column is NULL at that row — the
/// SQL "NULL keys never match" rule, applied uniformly to both sides.
fn extract_key(batch: &RecordBatch, indices: &[usize], row: usize) -> BoltResult<Option<JoinKey>> {
    // Single-key fast path (the overwhelming common case): build `One`
    // directly with no heap allocation. NULL ⇒ no key (SQL "NULL never
    // matches"). Multi-column keys fall through to a `Vec`-backed `Many`,
    // which `JoinKey::from` keeps as-is (a 1-elem vec can't reach here).
    if indices.len() == 1 {
        return Ok(extract_key_value(batch, indices[0], row)?.map(JoinKey::One));
    }
    let mut vs: Vec<JoinKeyValue> = Vec::with_capacity(indices.len());
    for &idx in indices {
        match extract_key_value(batch, idx, row)? {
            Some(v) => vs.push(v),
            None => return Ok(None),
        }
    }
    Ok(Some(JoinKey::Many(vs)))
}

/// Extract one key column value at `row`. Returns `Ok(None)` if the column
/// is NULL at that row (the SQL "NULL keys never match" rule). Factored out
/// of [`extract_key`] so the single-key fast path can build a `JoinKey::One`
/// without an intermediate `Vec`. The per-dtype dispatch, float
/// canonicalisation (review C12), and dict decoding (review F-7 analogue,
/// see the Dictionary arm below) are identical to the inline version it
/// replaced.
fn extract_key_value(
    batch: &RecordBatch,
    idx: usize,
    row: usize,
) -> BoltResult<Option<JoinKeyValue>> {
    let arr = batch.column(idx);
    if arr.is_null(row) {
        return Ok(None);
    }
    let v = match arr.data_type() {
        ArrowDataType::Int32 => JoinKeyValue::I32(
            arr.as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .value(row),
        ),
        ArrowDataType::Int64 => JoinKeyValue::I64(
            arr.as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(row),
        ),
        ArrowDataType::Float32 => JoinKeyValue::F32(
            // Review C12: canonicalise -0.0 -> +0.0 so that signed-zero
            // join keys match across sides (matches SQL/IEEE and
            // DuckDB). NaN bit patterns are preserved as-is, so
            // NaN-keyed rows never match (`NaN != NaN`).
            canonicalise_f32(
                arr.as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .value(row),
            )
            .to_bits(),
        ),
        ArrowDataType::Float64 => JoinKeyValue::F64(
            // Review C12: same signed-zero canonicalisation as Float32.
            canonicalise_f64(
                arr.as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap()
                    .value(row),
            )
            .to_bits(),
        ),
        ArrowDataType::Boolean => JoinKeyValue::Bool(
            arr.as_any()
                .downcast_ref::<BooleanArray>()
                .unwrap()
                .value(row),
        ),
        ArrowDataType::Utf8 => JoinKeyValue::Utf8(
            // Raw Utf8 still allocates per row, but Box<str> skips
            // String's capacity field + typical over-allocation.
            // ~50% smaller than the previous `to_string()` path.
            arr.as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .value(row)
                .into(),
        ),
        // Correctness fix (review F-7 analogue): dict-encoded Utf8 keys
        // must key on the DECODED STRING VALUE, not the raw dictionary
        // index. The previous L6 optimisation keyed on the i32 index on
        // the assumption that "one dict per batch/column ⇒ equal indices
        // ⇔ equal strings". That assumption breaks the moment a join
        // child yields a raw Arrow `DictionaryArray`: the build side and
        // the probe side then carry INDEPENDENT dictionaries, so the
        // SAME index can map to DIFFERENT strings (and the same string
        // can sit at DIFFERENT indices). `check_key_dtypes` only checks
        // the Arrow datatype — both sides are `Dictionary(Int32, Utf8)`
        // — so it cannot catch this. Keying on the index produced silent
        // WRONG join results.
        //
        // We therefore resolve each row's index through ITS OWN
        // dictionary to the actual string and key on `Utf8(Box<str>)` —
        // the exact same variant the raw-`StringArray` path produces. A
        // dict-encoded side and a raw-string side (or two independently
        // dict-encoded sides) now compare on string identity, which is
        // correct regardless of how each side encoded its dictionary.
        // This costs one string copy per row on the dict path (the raw
        // Utf8 fast path below is untouched).
        ArrowDataType::Dictionary(key_ty, value_ty)
            if matches!(value_ty.as_ref(), ArrowDataType::Utf8) =>
        {
            // Resolve the dict index for `row` to a (values_array,
            // position) pair, dispatching on the index width. NULL was
            // already handled by the `arr.is_null(row)` guard at the top
            // of this fn (Arrow dict NULL lives in the keys array's
            // validity). We borrow the typed dictionary's `values()`
            // (the dictionary value array) so we can decode the index to
            // the actual string.
            let (values_ref, value_pos): (&ArrayRef, usize) = match key_ty.as_ref() {
                ArrowDataType::Int32 => {
                    let da = arr
                        .as_any()
                        .downcast_ref::<DictionaryArray<ArrowInt32Type>>()
                        .ok_or_else(|| {
                            BoltError::Type("JOIN: dict<i32,utf8> downcast failed".into())
                        })?;
                    let pos = usize::try_from(da.keys().value(row)).map_err(|_| {
                        BoltError::Type("JOIN: dict<i32,utf8> negative key index".into())
                    })?;
                    (da.values(), pos)
                }
                ArrowDataType::Int64 => {
                    let da = arr
                        .as_any()
                        .downcast_ref::<DictionaryArray<ArrowInt64Type>>()
                        .ok_or_else(|| {
                            BoltError::Type("JOIN: dict<i64,utf8> downcast failed".into())
                        })?;
                    let pos = usize::try_from(da.keys().value(row)).map_err(|_| {
                        BoltError::Type("JOIN: dict<i64,utf8> negative key index".into())
                    })?;
                    (da.values(), pos)
                }
                other => {
                    return Err(BoltError::Type(format!(
                        "JOIN: unsupported dict key dtype {other:?} (expected Int32/Int64)"
                    )));
                }
            };
            // Decode the index to the string via the dictionary's value
            // array. Downcast to StringArray — guaranteed Utf8 by the
            // `value_ty` guard above.
            let str_values = values_ref
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| {
                    BoltError::Type(
                        "JOIN: dict values not a StringArray despite Utf8 value type".into(),
                    )
                })?;
            JoinKeyValue::Utf8(str_values.value(value_pos).into())
        }
        other => {
            return Err(BoltError::Type(format!(
                "JOIN: unsupported key dtype {other:?}"
            )));
        }
    };
    Ok(Some(v))
}

/// Canonicalise `-0.0` to `+0.0` so signed-zero pairs key the same
/// hash bucket. Preserves every other bit pattern, including NaN
/// payloads (`x == 0.0` is `false` for any NaN, so NaN-keyed rows never
/// match `NaN == NaN`). Mirrors the host-side canonicalisation in
/// `distinct::canonicalise_f64` and `groupby::canonicalise_f64` so
/// DISTINCT, GROUP BY, and JOIN share one float-equality relation
/// (review C12).
#[inline]
fn canonicalise_f64(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// `f32` analogue of [`canonicalise_f64`].
#[inline]
fn canonicalise_f32(x: f32) -> f32 {
    if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// Look up every name in `names` in the batch's schema, returning the
/// resolved column ordinals in the same order.
fn lookup_columns(batch: &RecordBatch, names: &[String]) -> BoltResult<Vec<usize>> {
    let mut out: Vec<usize> = Vec::with_capacity(names.len());
    for n in names {
        let idx = batch
            .schema()
            .index_of(n)
            .map_err(|e| BoltError::Plan(format!("JOIN: key column '{n}' not in batch: {e}")))?;
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
        ArrowDataType::Float32 => Arc::new(Float32Array::from(Vec::<f32>::new())) as ArrayRef,
        ArrowDataType::Float64 => Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
        ArrowDataType::Boolean => Arc::new(BooleanArray::from(Vec::<bool>::new())) as ArrayRef,
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
    crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal(s, "join output path")
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

    // ---- Stage 2/3: multi-key + bool/float + duplicate build keys ----
    let shape = match gpu_key_shape_for(build_batch, build_idx) {
        Some(s) => s,
        None => return Ok(None),
    };
    // Both sides must have matching dtypes per column (sanity — the planner
    // already enforced this at the schema level).
    for (b, p) in build_idx.iter().zip(probe_idx.iter()) {
        if build_batch.column(*b).data_type() != probe_batch.column(*p).data_type() {
            return Ok(None);
        }
    }

    // Stage-3 Utf8 path: single string key routes through the dedicated
    // dict-interning entry point.
    if matches!(shape, crate::jit::hash_join_kernel::KeyShape::SingleUtf8) {
        match crate::exec::gpu_join::execute_utf8_inner_join_on_gpu(
            lhs,
            rhs,
            build_is_left,
            build_idx[0],
            probe_idx[0],
            arrow_schema.clone(),
        ) {
            Ok(batch) => return Ok(Some(batch)),
            Err(e) => {
                log::debug!("gpu_join: Utf8 path declined ({e}); falling back to host");
                return Ok(None);
            }
        }
    }

    // Stage-5 AoS routing: only the single-int-key exact INNER path
    // currently has an AoS build kernel. We pick AoS when the probe side
    // dwarfs the build side (>8×) — see `AOS_ROUTING_PROBE_BUILD_RATIO`
    // for the rationale. Multi-key + lossy shapes stay on SoA.
    let stage1_single_int = matches!(
        shape,
        crate::jit::hash_join_kernel::KeyShape::SingleI32
            | crate::jit::hash_join_kernel::KeyShape::SingleI64
    );
    if stage1_single_int
        && crate::exec::gpu_join::should_route_aos(n_probe, n_build)
        && build_idx.len() == 1
    {
        let b_key_idx = build_idx[0];
        let p_key_idx = probe_idx[0];
        let dtype = match build_batch.column(b_key_idx).data_type() {
            ArrowDataType::Int32 => crate::plan::logical_plan::DataType::Int32,
            ArrowDataType::Int64 => crate::plan::logical_plan::DataType::Int64,
            // Unreachable per `stage1_single_int` guard above.
            _ => return Ok(None),
        };
        // The AoS build kernel still requires unique build keys (CAS-only,
        // no collision chain on AoS yet). Honour the same gate as the
        // Stage-1 SoA fast path.
        if build_keys_are_unique(build_batch.column(b_key_idx), dtype) {
            match crate::exec::gpu_join::execute_inner_join_on_gpu_aos(
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
                        "gpu_join: AoS routing declined ({e}); trying SoA verify-aware path"
                    );
                    // Fall through to SoA.
                }
            }
        }
    }

    // Stage-2 exact + Stage-3 lossy-fold (post-verify) both go through the
    // verify-aware entry point. For exact shapes the verify is a no-op.
    match crate::exec::gpu_join::execute_inner_join_on_gpu_with_shape_and_verify(
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
            log::debug!("gpu_join: Stage-2/3 path declined ({e}); falling back to host");
            Ok(None)
        }
    }
}

/// Try the Stage-2 GPU OUTER-join path. Returns the same `Ok(Some(_)) /
/// Ok(None) / Err(_)` contract as `try_gpu_inner_join`.
///
/// Gates (all must hold):
///   * Equi-join only (caller already verified non-empty `on`).
///   * `KeyShape != SingleUtf8` — Stage 4 doesn't yet route OUTER through
///     the Utf8 dict-interning entry point (Stage 5 follow-up). Stage 4
///     DID lift the Stage-3 `is_exact_in_i64()` gate so lossy shapes
///     (`TwoI64`, `MultiI32`) now flow through the host post-verify path
///     inside `execute_outer_join_indices_on_gpu`.
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

    // Translate join_type into emit flags. preserve_probe = LEFT/RIGHT/FULL
    // (probe is always the preserved side per build_is_left choice).
    // preserve_build = FULL only (second-pass kernel walks unmatched build).
    let emit_unmatched_probe = matches!(
        join_type,
        JoinType::LeftOuter | JoinType::RightOuter | JoinType::FullOuter
    );
    let emit_unmatched_build = matches!(join_type, JoinType::FullOuter);

    // Stage-5 (GJ): SingleUtf8 OUTER now routes through the dedicated
    // dict-interning entry point. The Stage-4 byte-borrowed dict produces
    // exact i32 indices, so the GPU's `SingleI32` OUTER output is correct
    // without further host post-verify (streaming-intern + OUTER is a
    // Stage-6 follow-up).
    if matches!(shape, crate::jit::hash_join_kernel::KeyShape::SingleUtf8) {
        return match crate::exec::gpu_join::execute_utf8_outer_join_on_gpu(
            lhs,
            rhs,
            build_is_left,
            build_idx[0],
            probe_idx[0],
            emit_unmatched_probe,
            emit_unmatched_build,
            arrow_schema,
        ) {
            Ok(batch) => Ok(Some(batch)),
            Err(e) => {
                log::debug!("gpu_join: Utf8 outer-join path declined ({e}); falling back to host");
                Ok(None)
            }
        };
    }

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
            // Stage-3: Utf8 keys go through string interning to i32 dict
            // indices before reaching the kernel.
            ArrowDataType::Utf8 => Some(KeyShape::SingleUtf8),
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

#[cfg(test)]
mod tests {
    //! Unit tests for the host hash-join key extraction (review L6).
    //!
    //! The focus here is the `JoinKeyValue` shape produced by
    //! `extract_key` for the two Utf8 paths:
    //!
    //!   * Raw `Utf8` → `JoinKeyValue::Utf8(Box<str>)`. Tests assert the
    //!     equivalence relation (equal strings ⇒ equal keys; distinct
    //!     strings ⇒ distinct keys).
    //!   * `Dictionary(Int32, Utf8)` → `JoinKeyValue::Utf8(Box<str>)`. Tests
    //!     assert the dict index is DECODED to the underlying string value
    //!     (review F-7 analogue) so that two independently-encoded
    //!     dictionaries key on string identity, not raw indices.
    use super::*;
    use arrow_array::{Int32Array, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    /// Build a single-column Utf8 RecordBatch.
    fn utf8_batch(name: &str, values: Vec<Option<&str>>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            name,
            ArrowDataType::Utf8,
            true,
        )]));
        let arr: Arc<dyn Array> = Arc::new(StringArray::from(values));
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    /// Build a single-column DictionaryArray<Int32, Utf8> RecordBatch from
    /// `(keys, values)` pairs.
    fn dict_utf8_batch(name: &str, keys: Vec<Option<i32>>, values: Vec<&str>) -> RecordBatch {
        let key_arr = Int32Array::from(keys);
        let value_arr = StringArray::from(values);
        let dict =
            DictionaryArray::<ArrowInt32Type>::try_new(key_arr, Arc::new(value_arr)).unwrap();
        let dt = dict.data_type().clone();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(name, dt, true)]));
        let arr: Arc<dyn Array> = Arc::new(dict);
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    #[test]
    fn extract_key_utf8_returns_box_str_variant() {
        // Raw Utf8 — key should be `Utf8(Box<str>)`.
        let batch = utf8_batch("s", vec![Some("alpha"), Some("beta"), Some("alpha")]);
        let k0 = extract_key(&batch, &[0], 0).unwrap().unwrap();
        let k1 = extract_key(&batch, &[0], 1).unwrap().unwrap();
        let k2 = extract_key(&batch, &[0], 2).unwrap().unwrap();
        assert_eq!(k0.len(), 1);
        match &k0[0] {
            JoinKeyValue::Utf8(s) => assert_eq!(s.as_ref(), "alpha"),
            other => panic!("expected Utf8 variant, got {other:?}"),
        }
        // Equivalence: rows 0 and 2 share a string, so their keys must match.
        assert_eq!(k0, k2);
        // Distinct strings ⇒ distinct keys.
        assert_ne!(k0, k1);
    }

    #[test]
    fn extract_key_dict_utf8_decodes_to_string_value() {
        // Dictionary(Int32, Utf8): values = ["alpha", "beta"], keys =
        // [0, 1, 0, 1, 0]. The keys at rows {0,2,4} all point at "alpha",
        // and so must produce the same JoinKey.
        let batch = dict_utf8_batch(
            "s",
            vec![Some(0), Some(1), Some(0), Some(1), Some(0)],
            vec!["alpha", "beta"],
        );
        let keys: Vec<_> = (0..5)
            .map(|r| extract_key(&batch, &[0], r).unwrap().unwrap())
            .collect();
        // Variant check: dict rows are DECODED to the string value (review
        // F-7 analogue), NOT keyed on the raw index — that's what makes two
        // independently-encoded dictionaries join correctly.
        match &keys[0][0] {
            JoinKeyValue::Utf8(s) => assert_eq!(s.as_ref(), "alpha"),
            other => panic!("expected Utf8(\"alpha\"), got {other:?}"),
        }
        match &keys[1][0] {
            JoinKeyValue::Utf8(s) => assert_eq!(s.as_ref(), "beta"),
            other => panic!("expected Utf8(\"beta\"), got {other:?}"),
        }
        // Equivalence relation preserved: rows with the same string hash
        // and compare equal.
        assert_eq!(keys[0], keys[2]);
        assert_eq!(keys[0], keys[4]);
        assert_eq!(keys[1], keys[3]);
        assert_ne!(keys[0], keys[1]);
    }

    #[test]
    fn extract_key_dict_and_raw_utf8_key_equal_on_string_value() {
        // Correctness regression (review F-7 analogue): a dict-encoded side
        // and a raw-string side must key EQUAL when their strings match,
        // even though one carries an index and the other a string.
        let dict = dict_utf8_batch("s", vec![Some(1)], vec!["alpha", "beta"]);
        let raw = utf8_batch("s", vec![Some("beta")]);
        let dk = extract_key(&dict, &[0], 0).unwrap().unwrap();
        let rk = extract_key(&raw, &[0], 0).unwrap().unwrap();
        assert_eq!(dk, rk);
    }

    #[test]
    fn extract_key_independent_dicts_key_on_strings_not_indices() {
        // The core bug: two dict columns with INDEPENDENT dictionaries. The
        // same index maps to DIFFERENT strings, and the same string sits at
        // DIFFERENT indices. Keying on the index would (wrongly) match
        // build idx 0 ("alpha") with probe idx 0 ("zeta"), and would (wrongly)
        // FAIL to match the two "alpha" rows that sit at different indices.
        let build = dict_utf8_batch("s", vec![Some(0), Some(1)], vec!["alpha", "beta"]);
        let probe = dict_utf8_batch("s", vec![Some(0), Some(1)], vec!["zeta", "alpha"]);
        let b_alpha = extract_key(&build, &[0], 0).unwrap().unwrap(); // idx 0 -> "alpha"
        let p_idx0 = extract_key(&probe, &[0], 0).unwrap().unwrap(); // idx 0 -> "zeta"
        let p_alpha = extract_key(&probe, &[0], 1).unwrap().unwrap(); // idx 1 -> "alpha"
                                                                      // Same index, different string => must NOT be equal.
        assert_ne!(b_alpha, p_idx0);
        // Different index, same string => MUST be equal.
        assert_eq!(b_alpha, p_alpha);
    }

    #[test]
    fn extract_key_dict_utf8_null_row_is_none() {
        // NULL dict-key rows should surface as `Ok(None)` from extract_key,
        // matching the SQL "NULL never matches" rule.
        let batch = dict_utf8_batch("s", vec![Some(0), None, Some(1)], vec!["alpha", "beta"]);
        assert!(extract_key(&batch, &[0], 0).unwrap().is_some());
        assert!(extract_key(&batch, &[0], 1).unwrap().is_none());
        assert!(extract_key(&batch, &[0], 2).unwrap().is_some());
    }

    #[test]
    fn build_hash_map_dict_utf8_groups_by_string_value() {
        // End-to-end: build_hash_map over a dict-utf8 build side. Two
        // build rows share dict index 0 ("alpha"), so the resulting bucket
        // (now keyed on the decoded string) must hold both row indices.
        let batch = dict_utf8_batch(
            "s",
            vec![Some(0), Some(1), Some(0), Some(2)],
            vec!["alpha", "beta", "gamma"],
        );
        let map = build_hash_map(&batch, &[0]).unwrap();
        // Three distinct strings => three buckets.
        assert_eq!(map.len(), 3);
        // The "alpha" bucket has both row 0 and row 2.
        let alpha_key: JoinKey = JoinKey::from(vec![JoinKeyValue::Utf8("alpha".into())]);
        let alpha_rows = map.get(&alpha_key).expect("alpha bucket present");
        let alpha_slice = alpha_rows.as_slice();
        assert_eq!(alpha_slice.len(), 2);
        assert!(alpha_slice.contains(&0));
        assert!(alpha_slice.contains(&2));
    }

    /// Hash a single value through `std::hash::Hash` with the default hasher.
    fn hash_of<T: std::hash::Hash>(t: &T) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let mut h = DefaultHasher::new();
        t.hash(&mut h);
        h.finish()
    }

    /// The inline single-key `JoinKey::One` must hash/compare byte-for-byte
    /// identically to the old `Vec<JoinKeyValue>` shape (and to the `Many`
    /// fallback carrying the same value). `Vec<T>: Hash` writes len + each
    /// element; `JoinKey` delegates `Hash` to `self.as_slice()` (`[T]: Hash`)
    /// which does the same, so the hash streams MUST coincide.
    #[test]
    fn join_key_one_matches_vec_hash_and_eq() {
        let v = vec![JoinKeyValue::I64(99)];
        let one = JoinKey::from(v.clone()); // collapses to One
        assert!(matches!(one, JoinKey::One(_)));
        // Same hash as the plain Vec it replaced.
        assert_eq!(hash_of(&one), hash_of(&v));
        // And the same as an explicit Many carrying the identical value.
        let many = JoinKey::Many(v.clone());
        assert_eq!(one, many);
        assert_eq!(hash_of(&one), hash_of(&many));

        // Multi-key path stays a Vec and matches the Vec hash too.
        let mv = vec![JoinKeyValue::I32(1), JoinKeyValue::Utf8("x".into())];
        let mk = JoinKey::from(mv.clone());
        assert!(matches!(mk, JoinKey::Many(_)));
        assert_eq!(hash_of(&mk), hash_of(&mv));
        assert_eq!(mk.len(), 2);
        assert_eq!(mk[0], JoinKeyValue::I32(1));
        assert_eq!(mk[1], JoinKeyValue::Utf8("x".into()));

        // Distinct single keys differ.
        let other = JoinKey::One(JoinKeyValue::I64(100));
        assert_ne!(one, other);
    }
}

#[cfg(test)]
mod build_slot_tests {
    //! Unit tests for the inline-then-heap `BuildSlot` (review Batch-3 PERF).
    //!
    //! `BuildSlot` is a 2-variant local enum that stores the first u32 row
    //! index inline (the common unique-build-key case, ≈95% of real
    //! workloads) and promotes to a heap `Vec<u32>` only when a second
    //! value arrives. These tests pin the promotion semantics so the
    //! single-key fast path stays allocation-free.
    use super::BuildSlot;

    #[test]
    fn inline_single_stays_inline() {
        // The unique-build-key fast path: one push, no promotion.
        let s = BuildSlot::new(5);
        assert_eq!(s.as_slice(), &[5]);
        assert_eq!(s.len(), 1);
        assert!(matches!(s, BuildSlot::Inline(_)));
    }

    #[test]
    fn inline_then_heap_promotion() {
        // First push is via `new`, the second triggers promotion to heap,
        // the third extends the heap vec in place.
        let mut s = BuildSlot::new(1);
        s.push(2);
        s.push(3);
        assert_eq!(s.len(), 3);
        assert_eq!(s.as_slice(), &[1, 2, 3]);
        assert!(matches!(s, BuildSlot::Heap(_)));
    }
}

#[cfg(test)]
mod nested_loop_streaming_tests {
    //! Tests for the EXEC-M3 streaming (chunked) non-equi join fix.
    //!
    //! The bug: the old `execute_nested_loop_join` materialised the WHOLE
    //! `outer × inner` cartesian product via `execute_cross_join` before
    //! filtering, so a large fact table joined to a tiny dimension table
    //! tripped `MAX_CROSS_ROWS` (u32::MAX) and ERRORED — even though the
    //! inner side is tiny and the result is small. The fix chunks the larger
    //! side so peak memory is `O(chunk × inner + output)`, never the full
    //! product. These tests pin: (a) a large-outer / tiny-inner non-equi
    //! join now succeeds with the correct matched count across multiple
    //! chunks, (b) empty matches yield an empty (schema-preserving) result,
    //! and (c) the genuine CROSS path still honours its row-count cap.
    use super::*;
    use crate::plan::logical_plan::{BinaryOp, DataType, Field, Literal};
    use arrow_array::Int32Array;
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    /// Single Int32 column batch.
    fn i32_batch(name: &str, values: Vec<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            name,
            ArrowDataType::Int32,
            false,
        )]));
        let arr: Arc<dyn Array> = Arc::new(Int32Array::from(values));
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    /// Two-Int32-column batch (dimension table: `lo`, `hi`).
    fn dim_batch(los: Vec<i32>, his: Vec<i32>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("lo", ArrowDataType::Int32, false),
            ArrowField::new("hi", ArrowDataType::Int32, false),
        ]));
        let lo: Arc<dyn Array> = Arc::new(Int32Array::from(los));
        let hi: Arc<dyn Array> = Arc::new(Int32Array::from(his));
        RecordBatch::try_new(schema, vec![lo, hi]).unwrap()
    }

    /// Plan-level combined output schema: `[fx, lo, hi]` — left (fact) then
    /// right (dim), matching the cross-join column order.
    fn combined_schema() -> Schema {
        Schema::new(vec![
            Field::new("fx", DataType::Int32, false),
            Field::new("lo", DataType::Int32, false),
            Field::new("hi", DataType::Int32, false),
        ])
    }

    fn col(name: &str) -> Expr {
        Expr::Column(name.to_string())
    }
    fn lit_i32(v: i32) -> Expr {
        Expr::Literal(Literal::Int32(v))
    }
    fn bin(op: BinaryOp, l: Expr, r: Expr) -> Expr {
        Expr::Binary {
            op,
            left: Box::new(l),
            right: Box::new(r),
        }
    }

    /// `fx BETWEEN lo AND hi` == `fx >= lo AND fx <= hi`.
    fn between_predicate() -> Expr {
        bin(
            BinaryOp::And,
            bin(BinaryOp::GtEq, col("fx"), col("lo")),
            bin(BinaryOp::LtEq, col("fx"), col("hi")),
        )
    }

    /// Pull the single output `fx` column out as a Vec for assertions.
    fn fx_values(h: QueryHandle) -> Vec<i32> {
        let batch = h.into_record_batch();
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        (0..arr.len()).map(|i| arr.value(i)).collect()
    }

    #[test]
    fn large_outer_tiny_inner_succeeds_with_correct_count() {
        // Fact table: fx = 0..N. Dim: one bucket [10, 19] inclusive.
        // Expected matches: fx in 10..=19 => exactly 10 rows.
        //
        // We drive a *tiny* chunk_cells (4) so the N-row outer is split into
        // many blocks — exercising the streaming concatenation path that the
        // old full-materialise code lacked. With the old code this whole
        // shape only worked because N is small here; the point of the tiny
        // chunk_cells is to prove the chunk loop stitches blocks correctly.
        const N: i32 = 1000;
        let fact = i32_batch("fx", (0..N).collect());
        let dim = dim_batch(vec![10], vec![19]);
        let schema = combined_schema();

        let handle = execute_nested_loop_join_chunked(
            fact,
            dim,
            JoinType::Inner,
            &[],
            &between_predicate(),
            &schema,
            4, // tiny chunk-cell ceiling: forces many chunks
        )
        .expect("large-outer / tiny-inner non-equi join must succeed");

        let mut got = fx_values(handle);
        got.sort_unstable();
        let expected: Vec<i32> = (10..=19).collect();
        assert_eq!(got, expected, "matched rows must be fx in [10, 19]");
    }

    #[test]
    fn empty_matches_yield_empty_schema_preserving_result() {
        // Dim bucket [10_000, 20_000] cannot match any fx in 0..1000.
        let fact = i32_batch("fx", (0..1000).collect());
        let dim = dim_batch(vec![10_000], vec![20_000]);
        let schema = combined_schema();

        let handle = execute_nested_loop_join_chunked(
            fact,
            dim,
            JoinType::Inner,
            &[],
            &between_predicate(),
            &schema,
            8,
        )
        .expect("non-equi join with no matches must still succeed");

        let batch = handle.into_record_batch();
        assert_eq!(batch.num_rows(), 0, "no rows should match");
        // Schema preserved exactly: 3 columns in [fx, lo, hi] order.
        assert_eq!(batch.num_columns(), 3);
        assert_eq!(batch.schema().field(0).name().as_str(), "fx");
        assert_eq!(batch.schema().field(1).name().as_str(), "lo");
        assert_eq!(batch.schema().field(2).name().as_str(), "hi");
    }

    #[test]
    fn small_inner_on_the_left_side_also_streams() {
        // Symmetry check: put the tiny side on the LEFT and the large side on
        // the RIGHT. The chunker must chunk the RIGHT (larger) side while
        // holding the LEFT whole, and column order (left ++ right) must be
        // preserved. Here left = dim-as-fact is awkward, so instead we keep
        // the combined schema [fx, lo, hi] but make the *dim* (right) larger
        // than the *fact* (left): fact has 3 rows, dim has 300 buckets.
        //
        // fact fx = [5, 15, 25]. Dim bucket i covers [i*10, i*10+9]:
        //   fx=5  -> bucket 0  [0,9]
        //   fx=15 -> bucket 1  [10,19]
        //   fx=25 -> bucket 2  [20,29]
        // Each fact row matches exactly one bucket => 3 matched rows.
        let fact = i32_batch("fx", vec![5, 15, 25]);
        let los: Vec<i32> = (0..300).map(|i| i * 10).collect();
        let his: Vec<i32> = (0..300).map(|i| i * 10 + 9).collect();
        let dim = dim_batch(los, his);
        let schema = combined_schema();

        let handle = execute_nested_loop_join_chunked(
            fact,
            dim,
            JoinType::Inner,
            &[],
            &between_predicate(),
            &schema,
            4, // forces the larger (right) side to chunk
        )
        .expect("tiny-left / large-right non-equi join must succeed");

        let mut got = fx_values(handle);
        got.sort_unstable();
        assert_eq!(got, vec![5, 15, 25]);
    }

    #[test]
    fn cross_join_still_respects_max_cross_rows_cap() {
        // The genuine CROSS path must still reject products over u32::MAX.
        // 70_000 × 70_000 = 4.9e9 > u32::MAX (4.29e9). The cap is checked
        // BEFORE materialisation, so this allocates only two 70k-row Int32
        // columns (~280 KB each) and errors fast.
        const SIDE: i32 = 70_000;
        let left = i32_batch("fx", (0..SIDE).collect());
        let right = i32_batch("rx", (0..SIDE).collect());
        let out_schema = Schema::new(vec![
            Field::new("fx", DataType::Int32, false),
            Field::new("rx", DataType::Int32, false),
        ]);

        // Use a match rather than `.expect_err()` so the Ok type need not be
        // `Debug`.
        let msg = match execute_cross_join(left, right, &out_schema) {
            Ok(_) => panic!("CROSS product over u32::MAX must error"),
            Err(e) => format!("{e:?}"),
        };
        assert!(
            msg.contains("too large") || msg.contains("limit"),
            "expected a cross-cap error, got: {msg}"
        );
    }
}
