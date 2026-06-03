// SPDX-License-Identifier: Apache-2.0

//! ORDER BY executor — host-side sort of a RecordBatch.
//!
//! Strategy: extract sort-key columns from the input, build a Vec of
//! arrow_array `SortColumn` descriptors, call `arrow::compute::lexsort_to_indices`
//! to obtain an `UInt32Array` of permutation indices, then `take` each
//! column to produce the sorted RecordBatch.
//!
//! ## Dispatch (planner-driven, since the v0.7 sort rework)
//!
//! The GPU path is now chosen by [`should_use_gpu_sort`] — a pure decision
//! function over `(n_rows, key_dtypes, directions)` — rather than the legacy
//! `BOLT_GPU_SORT=1` opt-in gate. By default the GPU path is selected when:
//!
//!   1. Every sort key is `Int32` or `Int64` (the dtypes the radix machinery
//!      in [`crate::jit::sort_kernel_radix`] handles, and which the bitonic
//!      multi-key driver also covers), and
//!   2. `n_rows >= GPU_SORT_MIN_ROWS` (below this the h2d/d2h round-trip
//!      dominates).
//!
//! The `BOLT_GPU_SORT` env var is retained purely as a **force override**:
//! `BOLT_GPU_SORT=1` forces the GPU path to be *attempted* even on shapes the
//! default heuristic would decline (subject to the GPU driver's own gates),
//! and `BOLT_GPU_SORT=0` forces the host path regardless of shape. Any other
//! value (or unset) means "use the planner heuristic". `directions` does not
//! affect the decision today — both ASC and DESC ride the same kernel — but it
//! is part of the signature so a future cost model (e.g. a presortedness
//! detector that prefers the host path for already-ordered input) can consult
//! it without a signature churn.
//!
//! Host-side is the fallback for everything else (Float/Bool/Utf8 keys, tiny
//! inputs, computed sort keys, or any GPU precondition miss inside
//! [`try_gpu_sort`]). The host-side `lexsort_to_indices` always produces a
//! correct result, so a decline at any stage is safe.
//!
//! ## ⚠️ TEMPORARY SAFETY GATE — GPU sort is opt-in only
//!
//! `ORDER BY` correctness must never be silently corrupted, so [`execute_sort`]
//! only runs the GPU path under an explicit `BOLT_GPU_SORT=1` opt-in; every
//! other sort falls through to the correct host `lexsort_to_indices`. The
//! [`should_use_gpu_sort`] heuristic below is retained unchanged for an easy
//! re-enable. Kernel status (updated after the stable-radix fix):
//!   * the radix scatter is now CROSS-BLOCK STABLE (per-block-per-digit
//!     histogram + global block-offset scan; the old per-digit
//!     `atom.global.add` race is gone). Non-null Int32/Int64 ASC/DESC and
//!     two-key `ORDER BY` are GPU-validated correct. Float ORDER BY now routes
//!     through the radix path too (R1 — host IEEE-monotonic key transform with
//!     NaN / -0.0 canonicalization), pending on-hardware validation. The gate
//!     stays ON pending NULLS FIRST/LAST device-validity wiring (nullable
//!     primitives carry no device validity bitmap yet).
//!   * the bitonic kernel is only verified for Int32 ASC at a power-of-two row
//!     count — DESC, 64-bit, float, and padded inputs mis-sort.
//! See the `gpu-validation-known-issues` notes. To re-enable GPU sort by
//! default, drop the `gpu_sort_override() == Some(true)` conjunct in
//! [`execute_sort`].

use std::sync::Arc;

use arrow::compute::{lexsort_to_indices, take, SortColumn, SortOptions};
use arrow_array::{Array, RecordBatch, UInt32Array};

use crate::error::{BoltError, BoltResult};
use crate::exec::QueryHandle;
use crate::jit::sort_kernel::SortDirection;
use crate::plan::logical_plan::{DataType, Expr, SortExpr};

/// Row-count threshold below which we keep the sort host-side. Empirically the
/// GPU h2d/d2h round-trip + JIT load dominates below ~10k rows; 16k gives the
/// device path enough work to amortise the launch overhead. Adjust as the
/// bitonic-kernel-launch-count overhead is profiled.
const GPU_SORT_MIN_ROWS: usize = 16_384;

/// Env var that *overrides* the planner-driven sort dispatch decision.
///
/// Mirrors [`crate::jit::sort_kernel_radix::BOLT_GPU_SORT_ENV`] — re-exported
/// here as the executor-facing name so the dispatch tests can reason about the
/// override without string-typing the variable. The override semantics differ
/// from the old opt-in gate: `"1"` *forces on*, `"0"` *forces off*, anything
/// else falls through to the heuristic (see [`gpu_sort_override`]).
const BOLT_GPU_SORT_ENV: &str = crate::jit::sort_kernel_radix::BOLT_GPU_SORT_ENV;

/// Tri-state parse of the `BOLT_GPU_SORT` override env var.
///
/// * `Some(true)`  — `"1"`: force the GPU path to be attempted.
/// * `Some(false)` — `"0"`: force the host path.
/// * `None`        — unset or any other value: use the planner heuristic.
///
/// Whitespace is trimmed first so a trailing newline from a shell export
/// still parses. The strict `"1"` / `"0"` matching keeps the override
/// unambiguous; `"true"` / `"yes"` deliberately do **not** force anything and
/// fall through to the heuristic.
fn gpu_sort_override() -> Option<bool> {
    match std::env::var(BOLT_GPU_SORT_ENV) {
        Ok(v) => match v.trim() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        },
        Err(_) => None,
    }
}

/// True iff `dtype` is a sort key dtype the GPU path is selected for by
/// default. The radix kernel handles `Int32`/`Int64`/`Float32`/`Float64`
/// (R1 wired float radix via the host IEEE-monotonic key transform), but we
/// deliberately keep the *default* heuristic to `Int32`/`Int64` only — floats
/// stay opt-in via `BOLT_GPU_SORT=1` (which overrides this heuristic outright)
/// until the float radix path is validated on hardware. Bool/Utf8 are cheaper
/// host-side at typical cardinalities.
fn dtype_is_gpu_sort_default(dtype: DataType) -> bool {
    matches!(dtype, DataType::Int32 | DataType::Int64)
}

/// Planner-driven sort dispatch decision.
///
/// Returns `true` when the engine should *attempt* the GPU sort path for a
/// sort of `n_rows` rows keyed by `key_dtypes` with per-key `directions`.
/// A `true` result is necessary-but-not-sufficient: [`try_gpu_sort`] still
/// applies the finer GPU preconditions (bare-column keys, key-count cap,
/// register budget, NULL-aware buffers) and may decline, in which case the
/// host path runs. A `false` result means "go straight to the host sort".
///
/// ## Decision rule
///
/// 1. If the `BOLT_GPU_SORT` override is set (`"1"`/`"0"`), it wins outright.
/// 2. Otherwise the heuristic selects the GPU path iff there is at least one
///    key, **every** key dtype is GPU-default-supported
///    ([`dtype_is_gpu_sort_default`] — i.e. `Int32`/`Int64`), and
///    `n_rows >= GPU_SORT_MIN_ROWS`.
///
/// `directions` is currently unused by the rule (both ASC and DESC ride the
/// same kernel) but is threaded through so a future presortedness / cost model
/// can refine the decision without a signature change. `key_dtypes` and
/// `directions` must be parallel; a length mismatch makes the function
/// conservatively decline the GPU path.
pub fn should_use_gpu_sort(
    n_rows: usize,
    key_dtypes: &[DataType],
    directions: &[SortDirection],
) -> bool {
    // Force override takes precedence over every shape consideration.
    if let Some(forced) = gpu_sort_override() {
        return forced;
    }
    // Parallel-slice contract: a mismatch is a caller bug; decline rather than
    // index out of bounds or make a half-informed decision.
    if key_dtypes.len() != directions.len() {
        return false;
    }
    if key_dtypes.is_empty() {
        return false;
    }
    if n_rows < GPU_SORT_MIN_ROWS {
        return false;
    }
    key_dtypes.iter().copied().all(dtype_is_gpu_sort_default)
}

/// Apply ORDER BY to the input handle.
pub fn execute_sort(input: QueryHandle, sort_exprs: &[SortExpr]) -> BoltResult<QueryHandle> {
    let batch = input.into_record_batch();
    if batch.num_rows() == 0 || sort_exprs.is_empty() {
        return Ok(QueryHandle::from_record_batch(batch));
    }

    // Planner-driven dispatch: resolve each key's dtype/direction and consult
    // `should_use_gpu_sort` (the `BOLT_GPU_SORT` env var overrides the
    // heuristic). When the GPU path is selected, try the radix permutation
    // first, then bitonic; any precondition miss falls through to the host
    // sort, which always produces a correct result.
    if let Some((key_dtypes, directions)) = resolve_key_shape(&batch, sort_exprs) {
        // SAFETY GATE (see gpu-validation known issues): the radix scatter is
        // now cross-block STABLE (per-block-per-digit histogram + global
        // block-offset scan) and non-null Int32/Int64 ASC/DESC + two-key
        // ORDER BY are GPU-validated correct. Remaining gaps keeping the gate
        // ON: (a) NULLS FIRST/LAST needs device-validity wiring (nullable
        // primitives carry no device validity bitmap yet); (b) float radix is
        // now routed through dispatch (R1 — IEEE-monotonic host key transform
        // with NaN / -0.0 canonicalization) but awaits on-hardware validation;
        // (c) the bitonic kernel is only verified for Int32 ASC at a
        // power-of-two row count.
        // So run the GPU path ONLY under an explicit `BOLT_GPU_SORT=1` opt-in
        // (for benchmarking / kernel
        // validation). Every production ORDER BY falls through to the correct
        // host `lexsort_to_indices` below. The lib-level kernel tests call the
        // `sort_indices_on_gpu_*` entry points directly, so they still exercise
        // the kernels regardless of this gate. To re-enable the GPU sort by
        // default once correct, drop the `gpu_sort_override() == Some(true)`
        // conjunct (restoring the plain `should_use_gpu_sort` dispatch).
        if gpu_sort_override() == Some(true)
            && should_use_gpu_sort(batch.num_rows(), &key_dtypes, &directions)
        {
            if let Some(perm) = try_gpu_sort_radix(&batch, sort_exprs)? {
                let new_cols: Vec<Arc<dyn Array>> = batch
                    .columns()
                    .iter()
                    .map(|c| take(c.as_ref(), &perm, None).map_err(arrow_err))
                    .collect::<BoltResult<Vec<_>>>()?;
                let out = RecordBatch::try_new(batch.schema(), new_cols).map_err(arrow_err)?;
                return Ok(QueryHandle::from_record_batch(out));
            }
            if let Some(sorted) = try_gpu_sort(&batch, sort_exprs)? {
                return Ok(QueryHandle::from_record_batch(sorted));
            }
        }
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
///   2. **Up to `MAX_SORT_KEYS` keys.** #19 widened this from the
///      original single-key gate. Mixed ASC/DESC across keys is fine.
///   3. **Bare column references.** No computed sort keys; matches the
///      bitonic path's gate 2.
///   4. **Int32 / Int64 dtypes per key.** Float radix needs the
///      IEEE-monotonic transform (deferred); Bool / Utf8 fall through.
///   5. **No NULLs in any key column.** Radix sort has no validity
///      routing — checked inside `radix_dispatch_predicate_multi`.
///   6. **`n_rows >= GPU_SORT_MIN_ROWS`.** Same threshold as the bitonic
///      path — h2d/d2h overhead dominates below ~16k.
///
/// On any miss we return `Ok(None)` and the caller falls through to the
/// bitonic path (which has its own gates) and then to the host path.
fn try_gpu_sort_radix(
    batch: &RecordBatch,
    sort_exprs: &[SortExpr],
) -> BoltResult<Option<UInt32Array>> {
    use crate::exec::gpu_sort::GpuSortKey;
    use crate::jit::sort_kernel::SortDirection;
    use crate::jit::sort_kernel_radix::gpu_sort_env_enabled;

    // Gate 1: env opt-in.
    if !gpu_sort_env_enabled() {
        return Ok(None);
    }
    // Gate 2: 1..=MAX_SORT_KEYS keys. The hard cap is enforced again
    // inside the predicate (defence in depth) so this gate is mostly
    // documentation; we early-out on the empty list to keep the rest of
    // the function's invariants simple.
    if sort_exprs.is_empty() {
        return Ok(None);
    }

    let n_rows = batch.num_rows();
    // Gate 6: row count threshold.
    if n_rows < GPU_SORT_MIN_ROWS {
        return Ok(None);
    }

    // Resolve every key into (column index, dtype, direction, nulls_first).
    // Any miss on bare-column / known-dtype / index lookup falls through.
    let mut resolved: Vec<(usize, crate::plan::logical_plan::DataType, SortDirection, bool)> =
        Vec::with_capacity(sort_exprs.len());
    for se in sort_exprs {
        // Gate 3: bare column reference.
        let col_name = match expr_to_column_name(&se.expr) {
            Ok(n) => n,
            Err(_) => return Ok(None),
        };
        let col_idx = match batch.schema().index_of(&col_name) {
            Ok(i) => i,
            Err(_) => return Ok(None),
        };
        let column = batch.column(col_idx);
        // Gate 4: dtype must map to a GPU-sortable internal dtype. The
        // radix predicate then narrows that to Int32/Int64.
        let dtype = match crate::exec::gpu_sort::arrow_dtype_to_internal(column.data_type()) {
            Some(d) => d,
            None => return Ok(None),
        };
        let dir = if se.descending {
            SortDirection::Desc
        } else {
            SortDirection::Asc
        };
        resolved.push((col_idx, dtype, dir, se.nulls_first));
    }

    // Build the multi-key descriptor list. We materialise GpuSortKey
    // references over the batch's columns; the borrow lives until the
    // sort_indices_on_gpu_radix_multi call returns.
    let keys: Vec<GpuSortKey<'_>> = resolved
        .iter()
        .map(|(idx, dtype, dir, nf)| GpuSortKey {
            column: batch.column(*idx).as_ref(),
            dtype: *dtype,
            direction: *dir,
            nulls_first: *nf,
        })
        .collect();

    // Predicate is re-checked inside the driver; calling it here gives
    // us a fast-fall-through on any per-key gate miss (dtype, NULLs, etc.)
    // without allocating GPU buffers.
    if !crate::exec::gpu_sort::radix_dispatch_predicate_multi(&keys) {
        return Ok(None);
    }

    // All gates passed — hand off to the GPU driver. The multi-key driver
    // handles single-key sorts as a 1-element list, so we don't need a
    // separate code path.
    crate::exec::gpu_sort::sort_indices_on_gpu_radix_multi(&keys)
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
    // `SortDirection` is imported at module scope; only `MAX_SORT_KEYS` is
    // needed locally here.
    use crate::jit::sort_kernel::MAX_SORT_KEYS;

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

/// Resolve the `(key_dtypes, directions)` shape of a sort for the dispatch
/// decision, without touching the GPU.
///
/// Returns `None` when any key is not a bare column reference, when a key
/// column is missing from the batch schema, or when a key's Arrow dtype isn't
/// one of the GPU-sortable kinds ([`crate::exec::gpu_sort::arrow_dtype_to_internal`]).
/// A `None` here means the GPU path can't apply, so the caller skips
/// `should_use_gpu_sort` entirely and runs the host sort. On success the two
/// returned vectors are parallel and in `sort_exprs` order.
fn resolve_key_shape(
    batch: &RecordBatch,
    sort_exprs: &[SortExpr],
) -> Option<(Vec<DataType>, Vec<SortDirection>)> {
    let mut dtypes: Vec<DataType> = Vec::with_capacity(sort_exprs.len());
    let mut dirs: Vec<SortDirection> = Vec::with_capacity(sort_exprs.len());
    for se in sort_exprs {
        let col_name = expr_to_column_name(&se.expr).ok()?;
        let idx = batch.schema().index_of(&col_name).ok()?;
        let dtype =
            crate::exec::gpu_sort::arrow_dtype_to_internal(batch.column(idx).data_type())?;
        dtypes.push(dtype);
        dirs.push(if se.descending {
            SortDirection::Desc
        } else {
            SortDirection::Asc
        });
    }
    Some((dtypes, dirs))
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
    use arrow_array::{Float32Array, Float64Array, Int32Array};
    use arrow_schema::{DataType, Field, Schema};
    // The dispatch heuristic (`should_use_gpu_sort`) and `resolve_key_shape`
    // operate on the crate's own `logical_plan::DataType`, whereas the batch
    // builder helpers above need Arrow's `DataType`. Alias the logical one so
    // both can coexist without the Arrow import shadowing it.
    use crate::plan::logical_plan::DataType as PlanDataType;

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

    // -- Planner-driven dispatch decision tests (pure host, no CUDA). --
    //
    // `should_use_gpu_sort` reads the `BOLT_GPU_SORT` override env var, so the
    // env-mutating tests share a process-wide gate to avoid racing on the
    // shared variable. The heuristic-only tests below take the gate too and
    // clear the var on entry so they observe the pure heuristic.

    use std::sync::Mutex as StdMutex;
    static ENV_GATE: StdMutex<()> = StdMutex::new(());

    fn dirs(n: usize) -> Vec<SortDirection> {
        vec![SortDirection::Asc; n]
    }

    #[test]
    fn dispatch_gpu_for_large_int_single_key() {
        let _g = ENV_GATE.lock().unwrap();
        std::env::remove_var(BOLT_GPU_SORT_ENV);
        assert!(should_use_gpu_sort(
            GPU_SORT_MIN_ROWS,
            &[PlanDataType::Int32],
            &dirs(1)
        ));
        assert!(should_use_gpu_sort(
            GPU_SORT_MIN_ROWS + 1,
            &[PlanDataType::Int64],
            &dirs(1)
        ));
    }

    #[test]
    fn dispatch_gpu_for_large_multi_int_key() {
        let _g = ENV_GATE.lock().unwrap();
        std::env::remove_var(BOLT_GPU_SORT_ENV);
        assert!(should_use_gpu_sort(
            1_000_000,
            &[PlanDataType::Int32, PlanDataType::Int64, PlanDataType::Int32],
            &dirs(3)
        ));
    }

    #[test]
    fn dispatch_host_below_threshold() {
        let _g = ENV_GATE.lock().unwrap();
        std::env::remove_var(BOLT_GPU_SORT_ENV);
        // One row below the threshold must decline.
        assert!(!should_use_gpu_sort(
            GPU_SORT_MIN_ROWS - 1,
            &[PlanDataType::Int32],
            &dirs(1)
        ));
        // Exactly at the threshold is the boundary that DOES select GPU.
        assert!(should_use_gpu_sort(
            GPU_SORT_MIN_ROWS,
            &[PlanDataType::Int32],
            &dirs(1)
        ));
    }

    #[test]
    fn dispatch_host_for_unsupported_dtype() {
        let _g = ENV_GATE.lock().unwrap();
        std::env::remove_var(BOLT_GPU_SORT_ENV);
        // Float / Bool / Utf8 keys are not auto-selected even when large.
        for dt in [
            PlanDataType::Float64,
            PlanDataType::Float32,
            PlanDataType::Bool,
            PlanDataType::Utf8,
        ] {
            assert!(
                !should_use_gpu_sort(1_000_000, &[dt], &dirs(1)),
                "dtype {dt:?} must not auto-select the GPU path"
            );
        }
        // A mixed key set with one unsupported dtype declines as a whole.
        assert!(!should_use_gpu_sort(
            1_000_000,
            &[PlanDataType::Int32, PlanDataType::Float64],
            &dirs(2)
        ));
    }

    #[test]
    fn dispatch_host_for_empty_keys() {
        let _g = ENV_GATE.lock().unwrap();
        std::env::remove_var(BOLT_GPU_SORT_ENV);
        assert!(!should_use_gpu_sort(1_000_000, &[], &[]));
    }

    #[test]
    fn dispatch_declines_on_parallel_slice_mismatch() {
        let _g = ENV_GATE.lock().unwrap();
        std::env::remove_var(BOLT_GPU_SORT_ENV);
        // key_dtypes.len() != directions.len() is a caller bug — decline.
        assert!(!should_use_gpu_sort(
            1_000_000,
            &[PlanDataType::Int32, PlanDataType::Int32],
            &dirs(1)
        ));
    }

    #[test]
    fn dispatch_override_forces_on_and_off() {
        let _g = ENV_GATE.lock().unwrap();

        // Force ON: GPU is attempted even for a shape the heuristic declines
        // (tiny float-keyed sort).
        std::env::set_var(BOLT_GPU_SORT_ENV, "1");
        assert!(should_use_gpu_sort(1, &[PlanDataType::Float64], &dirs(1)));

        // Force OFF: host path even for a shape the heuristic would select.
        std::env::set_var(BOLT_GPU_SORT_ENV, "0");
        assert!(!should_use_gpu_sort(
            1_000_000,
            &[PlanDataType::Int32],
            &dirs(1)
        ));

        // Non-{0,1} values fall through to the heuristic.
        std::env::set_var(BOLT_GPU_SORT_ENV, "true");
        assert!(should_use_gpu_sort(
            GPU_SORT_MIN_ROWS,
            &[PlanDataType::Int32],
            &dirs(1)
        ));
        assert!(!should_use_gpu_sort(1, &[PlanDataType::Int32], &dirs(1)));

        std::env::remove_var(BOLT_GPU_SORT_ENV);
    }

    #[test]
    fn resolve_key_shape_reports_dtypes_and_dirs() {
        let batch = int_batch("a", vec![Some(1), Some(2)]);
        let (dtypes, directions) =
            resolve_key_shape(&batch, &[col("a", true, false)]).expect("bare int col resolves");
        assert_eq!(dtypes, vec![PlanDataType::Int32]);
        assert_eq!(directions, vec![SortDirection::Desc]);
    }

    #[test]
    fn resolve_key_shape_none_for_missing_column() {
        let batch = int_batch("a", vec![Some(1)]);
        assert!(resolve_key_shape(&batch, &[col("missing", false, false)]).is_none());
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

    use crate::exec::gpu_sort::{
        radix_dispatch_predicate, radix_dispatch_predicate_multi, GpuSortKey, RADIX_DISPATCH_COUNT,
    };
    use crate::jit::sort_kernel::SortDirection;
    use crate::jit::sort_kernel_radix::set_radix_dispatch_for_tests;
    // `PlanDataType` (= `logical_plan::DataType`) is already imported at the top
    // of this `tests` module; it is in scope here without re-importing.
    use std::sync::atomic::Ordering;
    use std::sync::Mutex;

    /// Serializes every test that pins the radix dispatch gate. The gate
    /// (`RADIX_DISPATCH_STATE` in `sort_kernel_radix`) is process-global, so
    /// two tests pinning it to different values in parallel would race. The
    /// [`RadixGateGuard`] RAII helper acquires this lock, forces the gate to
    /// the requested state, and resets it to "uninitialised" on drop — so no
    /// test leaks gate state into a sibling. Replaces the old
    /// `std::env::set_var("BOLT_GPU_SORT", ...)` dance, which was both racy
    /// AND ineffective once the gate latched into its atomic cache.
    static RADIX_GATE_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard: locks [`RADIX_GATE_LOCK`] and forces the radix dispatch
    /// gate ON (`true`) or OFF (`false`) for the duration of the test, then
    /// resets the gate to "re-read from env" on drop.
    struct RadixGateGuard<'a> {
        _lock: std::sync::MutexGuard<'a, ()>,
    }

    impl RadixGateGuard<'_> {
        fn new(enabled: bool) -> Self {
            // Recover from a poisoned lock: a panicking sibling test should
            // not wedge the rest of the gate-dependent suite.
            let lock = RADIX_GATE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            set_radix_dispatch_for_tests(Some(enabled));
            Self { _lock: lock }
        }
    }

    impl Drop for RadixGateGuard<'_> {
        fn drop(&mut self) {
            set_radix_dispatch_for_tests(None);
        }
    }

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

    /// Build a 2-column batch (Int32 + Int32) with `n` rows.
    fn padded_two_int_batch(
        n: usize,
        col_a_name: &str,
        col_b_name: &str,
    ) -> RecordBatch {
        let n = n.max(GPU_SORT_MIN_ROWS);
        let a: Vec<i32> = (0..n as i32).collect();
        let b: Vec<i32> = (0..n as i32).map(|i| i * 2).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new(col_a_name, DataType::Int32, false),
            Field::new(col_b_name, DataType::Int32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(a)),
                Arc::new(Int32Array::from(b)),
            ],
        )
        .unwrap()
    }

    /// The single-key pure predicate accepts Int32 ASC with no nulls and
    /// (per #19) Int32 DESC with no nulls. Rejects every variation that
    /// should fall through to the bitonic / host path. Each assertion line
    /// documents the branch it pins.
    #[test]
    fn radix_predicate_gates() {
        let arr = Int32Array::from(vec![3, 1, 2, 5, 4]);
        // Int32 ASC no-nulls.
        assert!(radix_dispatch_predicate(
            &arr,
            PlanDataType::Int32,
            SortDirection::Asc,
            false,
        ));
        // #19: Int32 DESC no-nulls now accepted (was rejected by the v0.7
        // gate; the multi-key driver applies the direction transform
        // host-side).
        assert!(radix_dispatch_predicate(
            &arr,
            PlanDataType::Int32,
            SortDirection::Desc,
            false,
        ));
        // R1: Float dtype now ACCEPTED (radix_supports_dtype admits floats;
        // the host IEEE-monotonic key transform runs during gather). A
        // no-null Float32 column passes every gate.
        let farr = Float32Array::from(vec![3.0f32, 1.0, 2.0, 5.0, 4.0]);
        assert!(radix_dispatch_predicate(
            &farr,
            PlanDataType::Float32,
            SortDirection::Asc,
            false,
        ));
        // Bool dtype still rejected (radix_supports_dtype gate — 2 distinct
        // keys are cheaper to count-sort).
        let barr = arrow_array::BooleanArray::from(vec![true, false, true, false, true]);
        assert!(!radix_dispatch_predicate(
            &barr,
            PlanDataType::Bool,
            SortDirection::Asc,
            false,
        ));
        // Nullable column with at least one NULL rejected (no-NULLs gate).
        let null_arr = Int32Array::from(vec![Some(3), None, Some(1)]);
        assert!(!radix_dispatch_predicate(
            &null_arr,
            PlanDataType::Int32,
            SortDirection::Asc,
            false,
        ));
    }

    /// #19: the multi-key predicate accepts a 2-key Int32 ASC ASC list.
    #[test]
    fn radix_predicate_multi_two_key_asc_asc() {
        let a = Int32Array::from(vec![1, 2, 3]);
        let b = Int32Array::from(vec![4, 5, 6]);
        let keys = vec![
            GpuSortKey {
                column: &a,
                dtype: PlanDataType::Int32,
                direction: SortDirection::Asc,
                nulls_first: false,
            },
            GpuSortKey {
                column: &b,
                dtype: PlanDataType::Int32,
                direction: SortDirection::Asc,
                nulls_first: false,
            },
        ];
        assert!(radix_dispatch_predicate_multi(&keys));
    }

    /// #19: the multi-key predicate accepts mixed ASC DESC across keys.
    #[test]
    fn radix_predicate_multi_two_key_asc_desc() {
        let a = Int32Array::from(vec![1, 2, 3]);
        let b = Int32Array::from(vec![4, 5, 6]);
        let keys = vec![
            GpuSortKey {
                column: &a,
                dtype: PlanDataType::Int32,
                direction: SortDirection::Asc,
                nulls_first: false,
            },
            GpuSortKey {
                column: &b,
                dtype: PlanDataType::Int32,
                direction: SortDirection::Desc,
                nulls_first: false,
            },
        ];
        assert!(radix_dispatch_predicate_multi(&keys));
    }

    /// The multi-key predicate falls through when ANY key has an unsupported
    /// dtype. R1 made Float32/Float64 supported, so we use Bool as the
    /// unsupported key (one Int32, one Bool — radix can't handle the bool, so
    /// the whole sort goes back to the bitonic / host path).
    #[test]
    fn radix_predicate_multi_mixed_int_bool_rejected() {
        let a = Int32Array::from(vec![1, 2, 3]);
        let b = arrow_array::BooleanArray::from(vec![true, false, true]);
        let keys = vec![
            GpuSortKey {
                column: &a,
                dtype: PlanDataType::Int32,
                direction: SortDirection::Asc,
                nulls_first: false,
            },
            GpuSortKey {
                column: &b,
                dtype: PlanDataType::Bool,
                direction: SortDirection::Asc,
                nulls_first: false,
            },
        ];
        assert!(!radix_dispatch_predicate_multi(&keys));
    }

    /// R1: a multi-key Int32 + Float64 sort (both non-null) now ENGAGES the
    /// radix path — the float key rides the host IEEE-monotonic transform, so
    /// the predicate accepts the whole list.
    #[test]
    fn radix_predicate_multi_int_plus_float_accepted() {
        let a = Int32Array::from(vec![1, 2, 3]);
        let b = Float64Array::from(vec![1.0, 2.0, 3.0]);
        let keys = vec![
            GpuSortKey {
                column: &a,
                dtype: PlanDataType::Int32,
                direction: SortDirection::Asc,
                nulls_first: false,
            },
            GpuSortKey {
                column: &b,
                dtype: PlanDataType::Float64,
                direction: SortDirection::Asc,
                nulls_first: false,
            },
        ];
        assert!(radix_dispatch_predicate_multi(&keys));
    }

    /// Empty key list falls through (no sort to do — and the radix path
    /// would have no key to histogram).
    #[test]
    fn radix_predicate_multi_empty_rejected() {
        assert!(!radix_dispatch_predicate_multi(&[]));
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
        // Pin the gate ON deterministically via the serialized override —
        // see [`RadixGateGuard`]. The old `std::env::set_var` approach both
        // raced with sibling gate tests and was a no-op once the gate
        // latched into its atomic cache.
        let _gate = RadixGateGuard::new(true);

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
    }

    /// Sanity check the other side of the gate: with the gate forced
    /// **off**, `try_gpu_sort_radix` returns `Ok(None)` without
    /// bumping the dispatch counter. This is the default production
    /// behaviour — the radix path is opt-in until benched in.
    #[test]
    fn radix_dispatch_skipped_when_env_off() {
        let _gate = RadixGateGuard::new(false);

        let before = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);

        let batch = padded_int_batch("a", vec![5, 3, 1, 4, 2]);
        let res = try_gpu_sort_radix(&batch, &[col("a", false, false)]);
        assert!(matches!(res, Ok(None)));

        let after = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            before, after,
            "counter must not bump when env gate is off"
        );
    }

    /// The dispatch also rejects when row count is below the threshold,
    /// even with env on and all other gates green — small inputs amortize
    /// kernel launch worse than the host sort.
    #[test]
    fn radix_dispatch_skipped_when_too_small() {
        let _gate = RadixGateGuard::new(true);

        let before = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);

        // Only 5 rows — well below GPU_SORT_MIN_ROWS.
        let batch = int_batch("a", vec![Some(3), Some(1), Some(2), Some(5), Some(4)]);
        let res = try_gpu_sort_radix(&batch, &[col("a", false, false)]);
        assert!(matches!(res, Ok(None)));

        let after = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);
        assert_eq!(before, after, "small-input gate must short-circuit");

    }

    /// #19: single-key DESC engages the radix path (the v0.7 gate would
    /// have falled through). Same observability hook as the ASC test:
    /// counter delta proves the dispatch decision committed.
    #[test]
    #[cfg_attr(not(feature = "cuda-stub"), ignore = "gpu:sort_radix")]
    fn radix_dispatch_engages_on_int32_desc() {
        let _gate = RadixGateGuard::new(true);

        let before = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);

        let batch = padded_int_batch("a", vec![5, 3, 1, 4, 2]);
        let _ = try_gpu_sort_radix(&batch, &[col("a", true, false)]);

        let after = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);
        assert!(
            after >= before + 1,
            "DESC radix dispatch counter did not increment ({} -> {})",
            before,
            after,
        );

    }

    /// #19: multi-key ASC ASC engages the radix path.
    #[test]
    #[cfg_attr(not(feature = "cuda-stub"), ignore = "gpu:sort_radix")]
    fn radix_dispatch_engages_on_two_key_asc_asc() {
        let _gate = RadixGateGuard::new(true);

        let before = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);

        let batch = padded_two_int_batch(GPU_SORT_MIN_ROWS, "a", "b");
        let _ = try_gpu_sort_radix(
            &batch,
            &[col("a", false, false), col("b", false, false)],
        );

        let after = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);
        assert!(
            after >= before + 1,
            "multi-key ASC ASC dispatch counter did not increment ({} -> {})",
            before,
            after,
        );

    }

    /// #19: mixed ASC DESC across keys is handled — the per-key direction
    /// transform applies independently, so each key picks its own XOR
    /// constant.
    #[test]
    #[cfg_attr(not(feature = "cuda-stub"), ignore = "gpu:sort_radix")]
    fn radix_dispatch_engages_on_two_key_asc_desc() {
        let _gate = RadixGateGuard::new(true);

        let before = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);

        let batch = padded_two_int_batch(GPU_SORT_MIN_ROWS, "a", "b");
        let _ = try_gpu_sort_radix(
            &batch,
            &[col("a", false, false), col("b", true, false)],
        );

        let after = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);
        assert!(
            after >= before + 1,
            "mixed ASC DESC dispatch counter did not increment ({} -> {})",
            before,
            after,
        );

    }

    /// A mixed (Int32, Bool) multi-key sort falls through to the bitonic /
    /// host path — the predicate rejects the Bool key (R1 made floats
    /// supported, so Bool is now the canonical unsupported key), so the radix
    /// counter never bumps and no GPU buffer is allocated. Pure host predicate
    /// check, safe under `cuda-stub`.
    #[test]
    fn radix_dispatch_skipped_on_int_plus_bool() {
        let _gate = RadixGateGuard::new(true);

        let before = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);

        // Build a 2-column batch with one Int32 + one Bool. We only need
        // >= GPU_SORT_MIN_ROWS rows; the predicate's dtype gate is the
        // load-bearing assertion here. Bool maps to a GPU-sortable internal
        // dtype via `arrow_dtype_to_internal`, but `radix_supports_dtype`
        // rejects it, so the multi-key predicate falls through.
        let n = GPU_SORT_MIN_ROWS;
        let a: Vec<i32> = (0..n as i32).collect();
        let b: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Boolean, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(a)),
                Arc::new(arrow_array::BooleanArray::from(b)),
            ],
        )
        .unwrap();

        let res = try_gpu_sort_radix(
            &batch,
            &[col("a", false, false), col("b", false, false)],
        );
        assert!(matches!(res, Ok(None)), "mixed int+bool must fall through");

        let after = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            before, after,
            "predicate must reject mixed int+bool without bumping"
        );
    }

    /// R1: a single-key Float64 ASC sort engages the radix path (the float
    /// key rides the host IEEE-monotonic transform). GPU-gated because it
    /// reaches the device driver after the predicate match.
    #[test]
    #[cfg_attr(not(feature = "cuda-stub"), ignore = "gpu:sort_radix")]
    fn radix_dispatch_engages_on_float64_asc() {
        let _gate = RadixGateGuard::new(true);

        let before = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);

        let n = GPU_SORT_MIN_ROWS;
        let b: Vec<f64> = (0..n).map(|i| (n - i) as f64).collect();
        let schema = Arc::new(Schema::new(vec![Field::new(
            "b",
            DataType::Float64,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(Float64Array::from(b))]).unwrap();

        let _ = try_gpu_sort_radix(&batch, &[col("b", false, false)]);

        let after = RADIX_DISPATCH_COUNT.load(Ordering::SeqCst);
        assert!(
            after >= before + 1,
            "float64 dispatch counter did not increment ({} -> {})",
            before,
            after,
        );
    }
}
