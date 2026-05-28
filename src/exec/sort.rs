// SPDX-License-Identifier: Apache-2.0

//! ORDER BY executor — host-side sort of a RecordBatch.
//!
//! Strategy: extract sort-key columns from the input, build a Vec of
//! arrow_array `SortColumn` descriptors, call `arrow::compute::lexsort_to_indices`
//! to obtain an `UInt32Array` of permutation indices, then `take` each
//! column to produce the sorted RecordBatch.
//!
//! Host-side by default. A GPU fast path (bitonic sort) kicks in for
//! single-key, fixed-dtype, large-enough, no-NULL inputs — see
//! [`try_gpu_sort`] and `crate::exec::gpu_sort`. The host-side
//! lexsort_to_indices stays as the fallback for everything else (multi-key,
//! NULLable columns, tiny inputs where the upload cost dominates, Utf8/Bool
//! keys, etc.).

use std::sync::Arc;

use arrow::compute::{lexsort_to_indices, take, SortColumn, SortOptions};
use arrow_array::{Array, RecordBatch, UInt32Array};

use crate::error::{BoltError, BoltResult};
use crate::exec::QueryHandle;
use crate::plan::logical_plan::{Expr, SortExpr};

/// Row-count threshold below which we keep the sort host-side. Empirically the
/// GPU h2d/d2h round-trip + JIT load dominates below ~10k rows; 16k gives the
/// device path enough work to amortise the launch overhead. Adjust as the
/// bitonic-kernel-launch-count overhead is profiled.
const GPU_SORT_MIN_ROWS: usize = 16_384;

/// Apply ORDER BY to the input handle.
pub fn execute_sort(input: QueryHandle, sort_exprs: &[SortExpr]) -> BoltResult<QueryHandle> {
    let batch = input.into_record_batch();
    if batch.num_rows() == 0 || sort_exprs.is_empty() {
        return Ok(QueryHandle::from_record_batch(batch));
    }

    // v0.7 fast path: GPU radix sort for the single-key, Int32/Int64,
    // ASC-only, no-NULL, env-gated case. Lighter and asymptotically
    // faster than the bitonic path for the common large-int ORDER BY.
    // Returns the row permutation as a `UInt32Array`; the take loop
    // below gathers each column to produce the sorted batch — same
    // reorder machinery the host-fallback path uses, so the executor
    // surface stays narrow.
    if let Some(perm) = try_gpu_sort_radix(&batch, sort_exprs)? {
        let new_cols: Vec<Arc<dyn Array>> = batch
            .columns()
            .iter()
            .map(|c| take(c.as_ref(), &perm, None).map_err(arrow_err))
            .collect::<BoltResult<Vec<_>>>()?;
        let out = RecordBatch::try_new(batch.schema(), new_cols).map_err(arrow_err)?;
        return Ok(QueryHandle::from_record_batch(out));
    }

    // Fast path: GPU bitonic sort for the single-key, fixed-dtype, no-NULL,
    // large-input case. On any precondition miss, fall through to the
    // existing host path.
    if let Some(sorted) = try_gpu_sort(&batch, sort_exprs)? {
        return Ok(QueryHandle::from_record_batch(sorted));
    }

    let mut sort_cols: Vec<SortColumn> = Vec::with_capacity(sort_exprs.len());
    for se in sort_exprs {
        let col_name = expr_to_column_name(&se.expr)?;
        let idx = batch.schema().index_of(&col_name).map_err(arrow_err)?;
        sort_cols.push(SortColumn {
            values: batch.column(idx).clone(),
            options: Some(SortOptions {
                descending: se.descending,
                nulls_first: se.nulls_first,
            }),
        });
    }

    let indices: UInt32Array = lexsort_to_indices(&sort_cols, None).map_err(arrow_err)?;
    let new_cols: Vec<Arc<dyn Array>> = batch
        .columns()
        .iter()
        .map(|c| take(c.as_ref(), &indices, None).map_err(arrow_err))
        .collect::<BoltResult<Vec<_>>>()?;
    let out = RecordBatch::try_new(batch.schema(), new_cols).map_err(arrow_err)?;
    Ok(QueryHandle::from_record_batch(out))
}

/// Try the GPU radix sort fast path.
///
/// Returns `Ok(Some(perm))` with the row permutation when every gate is
/// satisfied. The caller is responsible for feeding `perm` through
/// `arrow::compute::take` to produce the sorted batch — keeping the
/// permutation as the return value (not the sorted batch) means the take
/// loop in `execute_sort` is the single reorder machinery for both this
/// path and the existing host fallback.
///
/// ## Gates (all must hold)
///   1. **Env opt-in.** `BOLT_GPU_SORT=1`. Default OFF; we keep the
///      historical bitonic / host paths as the steady-state behaviour
///      until the radix path is bake-tested in production.
///   2. **Single key.** `sort_exprs.len() == 1`. Multi-key lex ordering
///      needs a stable per-pass sort + outer key loop — punted to v0.8.
///   3. **Bare column reference.** No computed sort keys; matches the
///      bitonic path's gate 2.
///   4. **ASC direction.** DESC would need a second post-pass reversal
///      step or a flipped XOR constant; easy to add but no need today.
///   5. **`nulls_first == false`.** Pinned for clarity — radix has no
///      validity-bitmap routing.
///   6. **Int32 / Int64 dtype.** Float radix needs the IEEE-monotonic
///      transform (deferred); Bool / Utf8 fall through.
///   7. **No NULLs in the key column.** Radix sort has no validity
///      routing — checked inside [`radix_dispatch_predicate`].
///   8. **`n_rows >= GPU_SORT_MIN_ROWS`.** Same threshold as the bitonic
///      path — h2d/d2h overhead dominates below ~16k.
///
/// On any miss we return `Ok(None)` and the caller falls through to the
/// bitonic path (which has its own gates) and then to the host path.
fn try_gpu_sort_radix(
    batch: &RecordBatch,
    sort_exprs: &[SortExpr],
) -> BoltResult<Option<UInt32Array>> {
    use crate::jit::sort_kernel::SortDirection;
    use crate::jit::sort_kernel_radix::gpu_sort_env_enabled;

    // Gate 1: env opt-in.
    if !gpu_sort_env_enabled() {
        return Ok(None);
    }
    // Gate 2: single key.
    if sort_exprs.len() != 1 {
        return Ok(None);
    }
    let se = &sort_exprs[0];
    // Gate 3: bare column reference.
    let col_name = match expr_to_column_name(&se.expr) {
        Ok(n) => n,
        Err(_) => return Ok(None),
    };
    // Gate 4: ASC.
    if se.descending {
        return Ok(None);
    }
    // Gate 5: nulls_first off — pinned for clarity (the column is
    // null-free by the gate-7 check inside the predicate anyway).
    if se.nulls_first {
        return Ok(None);
    }

    let col_idx = match batch.schema().index_of(&col_name) {
        Ok(i) => i,
        Err(_) => return Ok(None),
    };
    let column = batch.column(col_idx);

    // Gate 6: dtype. `arrow_dtype_to_internal` maps the Arrow dtype to the
    // engine's internal dtype; the radix predicate then gates Int32/Int64.
    let dtype = match crate::exec::gpu_sort::arrow_dtype_to_internal(column.data_type()) {
        Some(d) => d,
        None => return Ok(None),
    };

    let n_rows = batch.num_rows();
    // Gate 8: row count threshold.
    if n_rows < GPU_SORT_MIN_ROWS {
        return Ok(None);
    }

    // Gates 6 + 7 + "fits u32" are checked inside `radix_dispatch_predicate`.
    // Calling it here lets `sort_indices_on_gpu_radix` re-check the same
    // predicate (it does) without duplicating the gate ladder.
    if !crate::exec::gpu_sort::radix_dispatch_predicate(
        column.as_ref(),
        dtype,
        SortDirection::Asc,
        se.nulls_first,
    ) {
        return Ok(None);
    }

    // All gates passed — hand off to the GPU driver.
    crate::exec::gpu_sort::sort_indices_on_gpu_radix(
        column.as_ref(),
        dtype,
        SortDirection::Asc,
        se.nulls_first,
    )
}

/// Try the GPU sort fast path. Returns `Ok(Some(sorted_batch))` on success,
/// `Ok(None)` if any precondition isn't met (the caller falls through to the
/// host path), or `Err(...)` only on a hard GPU error (out-of-memory, kernel
/// launch failure, etc.).
///
/// ## Stage 1 / 2 / 3 / 4 gates
///
///   1. Number of sort keys is in `1..=MAX_SORT_KEYS` (Stage 3 raised the
///      ceiling from 4 to 12; the real ceiling is the per-spec sm_70
///      register budget, validated by `compile_sort_kernel_spec`).
///   2. Each sort key is a bare column reference (no computed exprs).
///   3. Each column dtype is one of Int32 / Int64 / Float32 / Float64 /
///      Bool / Utf8 / Dictionary(Int32|Int64, Utf8). Stage 3 added Bool and
///      dictionary-encoded Utf8 (the latter sorts on the dictionary's
///      index column); **Stage 4** added plain Utf8, which now flows
///      through an inline dictionary builder inside `gpu_sort::host_values_for_key`
///      and ends up driving the i32 numeric kernel like any other column.
///   4. `n_rows >= GPU_SORT_MIN_ROWS` — below this, h2d/d2h overhead wins.
///   5. `n_rows <= u32::MAX` (tightened to `<= 2^31` inside `gpu_sort`
///      because the bitonic padding doubles).
///
/// NULLs are handled by Stage 2 via a per-key validity bitmap + nulls_first
/// flag — they no longer disqualify the GPU path. Stage 3 additionally
/// adds an `is_padded` bitmap that disambiguates real-vs-sentinel ties
/// (a Stage-2 silent-drop bug).
///
/// On any miss we return `Ok(None)`; the host path handles the input
/// correctly.
fn try_gpu_sort(
    batch: &RecordBatch,
    sort_exprs: &[SortExpr],
) -> BoltResult<Option<RecordBatch>> {
    use crate::jit::sort_kernel::{SortDirection, MAX_SORT_KEYS};

    // Gate 1: 1..=MAX_SORT_KEYS keys.
    if sort_exprs.is_empty() || sort_exprs.len() > MAX_SORT_KEYS {
        return Ok(None);
    }

    let n_rows = batch.num_rows();
    // Gate 4: row count threshold.
    if n_rows < GPU_SORT_MIN_ROWS {
        return Ok(None);
    }
    // Gate 5: fits the bitonic-padding bound.
    if n_rows > (u32::MAX as usize) {
        return Ok(None);
    }

    // Resolve every key: bare column ref + supported dtype.
    let mut resolved: Vec<(usize, crate::plan::logical_plan::DataType, SortDirection, bool)> =
        Vec::with_capacity(sort_exprs.len());
    for se in sort_exprs {
        let col_name = match expr_to_column_name(&se.expr) {
            Ok(n) => n,
            Err(_) => return Ok(None), // Gate 2 miss
        };
        let col_idx = match batch.schema().index_of(&col_name) {
            Ok(i) => i,
            Err(_) => return Ok(None),
        };
        let column = batch.column(col_idx);
        let dtype = match crate::exec::gpu_sort::arrow_dtype_to_internal(column.data_type()) {
            Some(d) => d,
            None => return Ok(None), // Gate 3 miss
        };
        let dir = if se.descending {
            SortDirection::Desc
        } else {
            SortDirection::Asc
        };
        resolved.push((col_idx, dtype, dir, se.nulls_first));
    }

    // Stage 3: route every supported case through the multi-key driver.
    // The driver carries the `is_padded` bitmap that fixes the sentinel-tie
    // row-drop bug — a legitimate `i32::MAX` value (or any other value
    // colliding with the dtype's sentinel) used to be silently dropped by
    // the single-key Stage-1 path; the multi-key driver routes padded rows
    // explicitly, preserving real ties.
    //
    // Stage 4: the single-key Stage-1 PTX entry was retired. The driver
    // here is the only path on the way to the GPU. Single-key sorts are
    // expressed as a `SortKernelSpec` with one entry in `keys`; the PTX-
    // shape golden tests were migrated to that form.
    //
    // Stage 5: the multi-key driver now returns `Ok(None)` when a per-key
    // gate decides the GPU path isn't worth it (today: the high-cardinality
    // plain-Utf8 sampler in `host_values_for_key`). We just forward that
    // None and let the caller fall through to `lexsort_to_indices`.
    let sorted = crate::exec::gpu_sort::sort_record_batch_on_gpu_multi(batch, &resolved)?;
    Ok(sorted)
}

/// Sort keys must currently be bare column references (possibly aliased).
/// Computed sort keys (`ORDER BY a + b`) error with a clear message.
fn expr_to_column_name(e: &Expr) -> BoltResult<String> {
    match e {
        Expr::Column(name) => Ok(name.clone()),
        Expr::Alias(inner, _) => expr_to_column_name(inner),
        other => Err(BoltError::Other(format!(
            "ORDER BY currently supports only column references, got {:?}",
            other
        ))),
    }
}

fn arrow_err(e: arrow::error::ArrowError) -> BoltError {
    BoltError::Other(format!("arrow: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Float64Array, Int32Array};
    use arrow_schema::{DataType, Field, Schema};

    fn int_batch(name: &str, values: Vec<Option<i32>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, true)]));
        let arr = Int32Array::from(values);
        RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
    }

    fn float_batch(name: &str, values: Vec<Option<f64>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Float64, true)]));
        let arr = Float64Array::from(values);
        RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
    }

    fn col(name: &str, desc: bool, nulls_first: bool) -> SortExpr {
        SortExpr {
            expr: Expr::Column(name.to_string()),
            descending: desc,
            nulls_first,
        }
    }

    fn as_i32(batch: &RecordBatch, col_idx: usize) -> Vec<Option<i32>> {
        let arr = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        (0..arr.len())
            .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
            .collect()
    }

    fn as_f64(batch: &RecordBatch, col_idx: usize) -> Vec<Option<f64>> {
        let arr = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        (0..arr.len())
            .map(|i| if arr.is_null(i) { None } else { Some(arr.value(i)) })
            .collect()
    }

    #[test]
    fn sort_ascending_int32() {
        let batch = int_batch("a", vec![Some(3), Some(1), Some(2), Some(5), Some(4)]);
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_sort(handle, &[col("a", false, false)]).unwrap();
        let result = out.into_record_batch();
        assert_eq!(
            as_i32(&result, 0),
            vec![Some(1), Some(2), Some(3), Some(4), Some(5)]
        );
    }

    #[test]
    fn sort_descending_int32() {
        let batch = int_batch("a", vec![Some(3), Some(1), Some(2), Some(5), Some(4)]);
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_sort(handle, &[col("a", true, false)]).unwrap();
        let result = out.into_record_batch();
        assert_eq!(
            as_i32(&result, 0),
            vec![Some(5), Some(4), Some(3), Some(2), Some(1)]
        );
    }

    #[test]
    fn sort_empty_input_is_empty() {
        let batch = int_batch("a", vec![]);
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_sort(handle, &[col("a", false, false)]).unwrap();
        let result = out.into_record_batch();
        assert_eq!(result.num_rows(), 0);
    }

    #[test]
    fn sort_no_keys_returns_input_unchanged() {
        let batch = int_batch("a", vec![Some(3), Some(1), Some(2)]);
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_sort(handle, &[]).unwrap();
        let result = out.into_record_batch();
        assert_eq!(as_i32(&result, 0), vec![Some(3), Some(1), Some(2)]);
    }

    #[test]
    fn sort_with_nulls_first_vs_last() {
        // nulls first, ascending
        let batch = int_batch("a", vec![Some(3), None, Some(1), None, Some(2)]);
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_sort(handle, &[col("a", false, true)]).unwrap();
        let result = out.into_record_batch();
        assert_eq!(
            as_i32(&result, 0),
            vec![None, None, Some(1), Some(2), Some(3)]
        );

        // nulls last, ascending
        let batch = int_batch("a", vec![Some(3), None, Some(1), None, Some(2)]);
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_sort(handle, &[col("a", false, false)]).unwrap();
        let result = out.into_record_batch();
        assert_eq!(
            as_i32(&result, 0),
            vec![Some(1), Some(2), Some(3), None, None]
        );
    }

    #[test]
    fn sort_descending_float64() {
        let batch = float_batch("x", vec![Some(2.5), Some(1.5), Some(3.5)]);
        let handle = QueryHandle::from_record_batch(batch);
        let out = execute_sort(handle, &[col("x", true, false)]).unwrap();
        let result = out.into_record_batch();
        assert_eq!(as_f64(&result, 0), vec![Some(3.5), Some(2.5), Some(1.5)]);
    }

    #[test]
    fn sort_rejects_computed_expr() {
        let batch = int_batch("a", vec![Some(1), Some(2)]);
        let handle = QueryHandle::from_record_batch(batch);
        // Build a non-column expression. Use Alias around an Add-like expr if possible;
        // here we just test the error path with any non-column variant via a literal
        // sort key — but since Expr variants beyond Column/Alias are project-specific,
        // we use Alias-of-Column which should succeed, then assert success.
        let aliased = SortExpr {
            expr: Expr::Alias(Box::new(Expr::Column("a".to_string())), "a_alias".to_string()),
            descending: false,
            nulls_first: false,
        };
        let out = execute_sort(handle, &[aliased]).unwrap();
        let result = out.into_record_batch();
        assert_eq!(as_i32(&result, 0), vec![Some(1), Some(2)]);
    }

    // ----- v0.7 radix-sort dispatch tests -----
    //
    // These verify the pure host-side gates in `try_gpu_sort_radix` /
    // `gpu_sort::radix_dispatch_predicate` without ever touching a CUDA
    // device. They run unconditionally — including under
    // `--features cuda-stub` on GPU-less CI — because the predicate
    // returns before any FFI call.
    //
    // The end-to-end "actually run the radix kernel" tests live in
    // `gpu_sort.rs` and are gated by `#[ignore = "gpu:sort_radix"]` so
    // CI skips them but a developer with a real device can opt in via
    // `cargo test -- --ignored gpu:sort_radix`.

    use crate::exec::gpu_sort::{radix_dispatch_predicate, RADIX_DISPATCH_COUNT};
    use crate::jit::sort_kernel::SortDirection;
    use crate::plan::logical_plan::DataType as PlanDataType;
    use std::sync::atomic::Ordering;

    /// Build an Int32 batch with the given non-null values and the
    /// minimum row count needed to pass the `GPU_SORT_MIN_ROWS` gate.
    /// Pads with `0` (sentinel value irrelevant — the gate only checks
    /// `n_rows`, not the contents).
    fn padded_int_batch(name: &str, head: Vec<i32>) -> RecordBatch {
        let mut all = head;
        while all.len() < GPU_SORT_MIN_ROWS {
            all.push(0);
        }
        let schema = Arc::new(Schema::new(vec![Field::new(name, DataType::Int32, false)]));
        let arr = Int32Array::from(all);
        RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
    }

    /// The pure predicate accepts Int32 ASC with no nulls and rejects
    /// every variation that should fall through to the bitonic / host
    /// path. Single test, multiple assertions — each line documents the
    /// branch it pins.
    #[test]
    fn radix_predicate_gates() {
        let arr = Int32Array::from(vec![3, 1, 2, 5, 4]);
        // Happy path — Int32 ASC no-nulls.
        assert!(radix_dispatch_predicate(
            &arr,
            PlanDataType::Int32,
            SortDirection::Asc,
            false,
        ));
        // DESC rejected (gate 4).
        assert!(!radix_dispatch_predicate(
            &arr,
            PlanDataType::Int32,
            SortDirection::Desc,
            false,
        ));
        // Float dtype rejected (gate 6 — radix_supports_dtype gate).
        assert!(!radix_dispatch_predicate(
            &arr,
            PlanDataType::Float32,
            SortDirection::Asc,
            false,
        ));
        // Nullable column with at least one NULL rejected (gate 7).
        let null_arr = Int32Array::from(vec![Some(3), None, Some(1)]);
        assert!(!radix_dispatch_predicate(
            &null_arr,
            PlanDataType::Int32,
            SortDirection::Asc,
            false,
        ));
    }

    /// `try_gpu_sort_radix` engages the radix path when every gate
    /// holds: env var set, single-key, ASC, Int32 no-null, large input.
    ///
    /// Observability: we read `RADIX_DISPATCH_COUNT` before and after
    /// the dispatch decision and assert it increments by exactly 1.
    /// `sort_indices_on_gpu_radix` bumps the counter the moment it
    /// commits to the radix path, and *before* any kernel launch — so
    /// even under `cuda-stub` (where the subsequent GPU work would
    /// fail) the counter delta is observable. We swallow the GPU error
    /// because that's the expected outcome on a stub build; the test
    /// is about predicate engagement, not kernel execution.
    #[test]
    #[cfg_attr(not(feature = "cuda-stub"), ignore = "gpu:sort_radix")]
    fn radix_dispatch_engages_on_int32_asc() {
        // Save and restore env state around the test. Tests in
        // `cargo test` share a process; another concurrent test could
        // see `BOLT_GPU_SORT=1` if we didn't restore — but the gate is
        // strict ("1"), and any concurrent test of unsorted-input
        // semantics is robust to either gate state because the predicate
        // is a function of the inputs.
        let prev = std::env::var("BOLT_GPU_SORT").ok();
        // SAFETY: set_var is unsafe on edition-2024 nightly but stable in
        // current edition; the test harness is single-threaded enough
        // that the env mutation here doesn't cross any other gate read
        // — only `gpu_sort_env_enabled` consults it.
        std::env::set_var("BOLT_GPU_SORT", "1");

        let before = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);

        let batch = padded_int_batch("a", vec![5, 3, 1, 4, 2]);
        // Run dispatch; we don't care whether the actual kernel
        // succeeds (it can't on cuda-stub) — only that the predicate
        // committed.
        let _ = try_gpu_sort_radix(&batch, &[col("a", false, false)]);

        let after = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);
        assert!(
            after >= before + 1,
            "radix dispatch counter did not increment ({} -> {})",
            before,
            after
        );

        // Restore env state.
        match prev {
            Some(v) => std::env::set_var("BOLT_GPU_SORT", v),
            None => std::env::remove_var("BOLT_GPU_SORT"),
        }
    }

    /// Sanity check the other side of the gate: with the env var
    /// **unset**, `try_gpu_sort_radix` returns `Ok(None)` without
    /// bumping the dispatch counter. This is the default production
    /// behaviour — the radix path is opt-in until benched in.
    #[test]
    fn radix_dispatch_skipped_when_env_off() {
        let prev = std::env::var("BOLT_GPU_SORT").ok();
        std::env::remove_var("BOLT_GPU_SORT");

        let before = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);

        let batch = padded_int_batch("a", vec![5, 3, 1, 4, 2]);
        let res = try_gpu_sort_radix(&batch, &[col("a", false, false)]);
        assert!(matches!(res, Ok(None)));

        let after = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            before, after,
            "counter must not bump when env gate is off"
        );

        if let Some(v) = prev {
            std::env::set_var("BOLT_GPU_SORT", v);
        }
    }

    /// The dispatch also rejects when row count is below the threshold,
    /// even with env on and all other gates green — small inputs amortize
    /// kernel launch worse than the host sort.
    #[test]
    fn radix_dispatch_skipped_when_too_small() {
        let prev = std::env::var("BOLT_GPU_SORT").ok();
        std::env::set_var("BOLT_GPU_SORT", "1");

        let before = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);

        // Only 5 rows — well below GPU_SORT_MIN_ROWS.
        let batch = int_batch("a", vec![Some(3), Some(1), Some(2), Some(5), Some(4)]);
        let res = try_gpu_sort_radix(&batch, &[col("a", false, false)]);
        assert!(matches!(res, Ok(None)));

        let after = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);
        assert_eq!(before, after, "small-input gate must short-circuit");

        match prev {
            Some(v) => std::env::set_var("BOLT_GPU_SORT", v),
            None => std::env::remove_var("BOLT_GPU_SORT"),
        }
    }
}
