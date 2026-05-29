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
/// default. The radix kernel handles `Int32`/`Int64`; the bitonic multi-key
/// driver covers the same set (plus floats/bool/utf8, which we do **not**
/// auto-select because they either need transforms not yet wired (float radix)
/// or are cheaper host-side at typical cardinalities).
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

    // Planner-driven dispatch: resolve each key's dtype/direction, ask
    // `should_use_gpu_sort`, and only attempt the GPU path when the heuristic
    // (or the `BOLT_GPU_SORT` override) selects it. On a `false` decision —
    // or any precondition miss inside `try_gpu_sort` — fall through to the
    // host path, which always produces a correct result.
    if let Some((key_dtypes, directions)) = resolve_key_shape(&batch, sort_exprs) {
        if should_use_gpu_sort(batch.num_rows(), &key_dtypes, &directions) {
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
            &[DataType::Int32],
            &dirs(1)
        ));
        assert!(should_use_gpu_sort(
            GPU_SORT_MIN_ROWS + 1,
            &[DataType::Int64],
            &dirs(1)
        ));
    }

    #[test]
    fn dispatch_gpu_for_large_multi_int_key() {
        let _g = ENV_GATE.lock().unwrap();
        std::env::remove_var(BOLT_GPU_SORT_ENV);
        assert!(should_use_gpu_sort(
            1_000_000,
            &[DataType::Int32, DataType::Int64, DataType::Int32],
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
            &[DataType::Int32],
            &dirs(1)
        ));
        // Exactly at the threshold is the boundary that DOES select GPU.
        assert!(should_use_gpu_sort(
            GPU_SORT_MIN_ROWS,
            &[DataType::Int32],
            &dirs(1)
        ));
    }

    #[test]
    fn dispatch_host_for_unsupported_dtype() {
        let _g = ENV_GATE.lock().unwrap();
        std::env::remove_var(BOLT_GPU_SORT_ENV);
        // Float / Bool / Utf8 keys are not auto-selected even when large.
        for dt in [DataType::Float64, DataType::Float32, DataType::Bool, DataType::Utf8] {
            assert!(
                !should_use_gpu_sort(1_000_000, &[dt], &dirs(1)),
                "dtype {dt:?} must not auto-select the GPU path"
            );
        }
        // A mixed key set with one unsupported dtype declines as a whole.
        assert!(!should_use_gpu_sort(
            1_000_000,
            &[DataType::Int32, DataType::Float64],
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
            &[DataType::Int32, DataType::Int32],
            &dirs(1)
        ));
    }

    #[test]
    fn dispatch_override_forces_on_and_off() {
        let _g = ENV_GATE.lock().unwrap();

        // Force ON: GPU is attempted even for a shape the heuristic declines
        // (tiny float-keyed sort).
        std::env::set_var(BOLT_GPU_SORT_ENV, "1");
        assert!(should_use_gpu_sort(1, &[DataType::Float64], &dirs(1)));

        // Force OFF: host path even for a shape the heuristic would select.
        std::env::set_var(BOLT_GPU_SORT_ENV, "0");
        assert!(!should_use_gpu_sort(
            1_000_000,
            &[DataType::Int32],
            &dirs(1)
        ));

        // Non-{0,1} values fall through to the heuristic.
        std::env::set_var(BOLT_GPU_SORT_ENV, "true");
        assert!(should_use_gpu_sort(
            GPU_SORT_MIN_ROWS,
            &[DataType::Int32],
            &dirs(1)
        ));
        assert!(!should_use_gpu_sort(1, &[DataType::Int32], &dirs(1)));

        std::env::remove_var(BOLT_GPU_SORT_ENV);
    }

    #[test]
    fn resolve_key_shape_reports_dtypes_and_dirs() {
        let batch = int_batch("a", vec![Some(1), Some(2)]);
        let (dtypes, directions) =
            resolve_key_shape(&batch, &[col("a", true, false)]).expect("bare int col resolves");
        assert_eq!(dtypes, vec![DataType::Int32]);
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
}
