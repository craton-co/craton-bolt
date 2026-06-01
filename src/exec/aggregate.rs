// SPDX-License-Identifier: Apache-2.0

//! Scalar (no GROUP BY) aggregate execution.
//!
//! For each `AggregateExpr` in the plan we:
//!   1. Upload its input column to the GPU as a `GpuVec`.
//!   2. JIT-compile a per-block reduction PTX kernel (`agg_kernels`).
//!   3. Launch `ceil(n / 256)` blocks, each writing one partial to a
//!      block-sized output buffer.
//!   4. Download the per-block partials to the host and finish the reduction
//!      sequentially.
//!   5. Pack the scalar results into a single-row Arrow `RecordBatch` whose
//!      schema matches `AggregateSpec::output_schema`.
//!
//! # v0.6 async-memcpy pilot (promoted in v0.7)
//!
//! This executor is the **pilot** for the v0.6 async memcpy + pinned host
//! buffer rollout. The whole H2D → kernel → D2H chain is issued on a single
//! per-call [`CudaStream`] (`CudaStream::null_or_default`) with the
//! [`GpuVec::from_slice_async`] / [`GpuVec::to_pinned_async`] wrappers, and
//! the executor synchronizes exactly once at the very end.
//!
//! As of v0.7 the local `upload_primitive_values_async` helper that
//! encapsulated the cuda-stub fallback was promoted to
//! [`crate::exec::gpu_upload`] so the filter / GROUP BY / join executors
//! can share the same shape. The aggregate path now imports it from
//! there; semantics are unchanged.
//!
//! Pinned host buffers (`PinnedHostBuffer<T>`, allocated by
//! `cuMemAllocHost_v2`) let the driver DMA straight in/out of host pages
//! without synthesizing a staging copy first — that is what makes the
//! H2D / kernel / D2H phases actually overlap on the same stream. The
//! latency win compounds across multiple per-aggregate column uploads in
//! a query like `SELECT SUM(a), AVG(b), MIN(c) FROM t`.
//!
//! Other executors (filter, GROUP BY, joins, …) currently still take the
//! synchronous `GpuBuffer::from_slice` / `to_vec` path on the no-null fast
//! path; they will follow this same template in subsequent PRs. Under
//! `--features cuda-stub` the async FFI shims all return `CUDA_ERROR_STUB`,
//! so this module falls back to the synchronous `from_slice` / `to_vec`
//! path — both paths fail the same way at the FFI boundary in stub mode,
//! but preferring the sync wrappers keeps the call shape closer to what
//! existed before the pilot landed.
//!
//! Scope (first cut):
//!   - No GROUP BY. `aggregate.group_by` must be empty.
//!   - No pre-aggregation kernel. `pre` must be `None`; this is the shape the
//!     physical-plan lowering produces for queries like `SELECT SUM(c) FROM t`
//!     where every aggregate input is a bare column reference and there is no
//!     filter. `pre = Some(...)` returns a "not yet implemented" error.
//!   - Primitive dtypes only (Int32, Int64, Float32, Float64).
//!   - `AVG` is computed by a **single fused kernel** that, in one pass over
//!     the input column, emits per-block `(f64 sum, u32 count)` partials; the
//!     host sums each partial array and divides. One PTX compilation, one
//!     launch, one stream-sync per AVG — and the count comes from what the
//!     GPU actually summed rather than a parallel Arrow-bitmap walk.
//!   - `COUNT(col)` is computed on the host from the Arrow null bitmap (no
//!     GPU launch); `COUNT(*)` returns the row count directly.
//!
//! # Validity (NULL) propagation
//!
//! v0.5/M1: primitive scalar aggregates now honour the Arrow validity bitmap.
//! The strategy is intentionally simple:
//!
//!   - **Fast path** (`null_count == 0`): zero-copy upload of the raw values
//!     buffer via `primitive_to_gpu` and the standard GPU reduction kernel
//!     (`bolt_reduce` / `bolt_avg_reduce`). No bitmap inspection on the GPU.
//!   - **Slow path** (`null_count > 0`): the host strips NULL positions via
//!     `filter_primitive_to_vec` into a dense `Vec<T>`, then either runs a
//!     small host-side reduce (`reduce_host_slice`) or uploads the stripped
//!     slice and runs the standard GPU kernel. The GPU never sees garbage at
//!     NULL positions, so the kernel reuses its identity at empty positions
//!     unchanged.
//!
//! Why host-strip rather than a masked GPU reduction:
//!
//!   - The fast path stays a true zero-copy launch; we don't pay any per-row
//!     branch when there are no nulls (the common case).
//!   - A masked GPU reduction would require a second kernel variant per (op,
//!     dtype), tripling the codegen surface to skip a (typically tiny)
//!     host-side strip. The Bool/Utf8 `extended_agg` path already takes the
//!     same host-fallback shape — primitives just join it on the slow path.
//!   - The 0.3.x compaction code already produces dense, post-filter inputs
//!     to `aggregate.rs` in the WHERE path; the only remaining
//!     null-on-primitive case at this entry is `pre = None` (bare-column
//!     aggregate, no filter) where the source batch's null bitmap is the
//!     only signal — exactly what `filter_primitive_to_vec` reads.
//!
//! Per-aggregate effect:
//!
//!   - `COUNT(col)`: `non_null_count_for_input` = `len - null_count` (no GPU).
//!   - `SUM(col)`: NULL rows stripped before upload; GPU sums survivors.
//!   - `MIN(col)` / `MAX(col)`: NULL rows stripped; GPU sees only valid values.
//!   - `AVG(col)`: NULL rows stripped; the fused kernel's per-block count
//!     therefore matches the non-NULL row count and the host divide is correct.
//!
//! `COUNT(*)` (no column reference) is unaffected — it always returns the
//! source-batch row count regardless of any column's null bitmap.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch,
};
use arrow_schema::{
    DataType as ArrowDataType, Schema as ArrowSchema,
};
use bytemuck::Pod;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{grid_x_for, CudaStream};
use crate::exec::module_cache;
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::{
    compile_avg_reduction_kernel, compile_reduction_kernel,
    compile_reduction_kernel_with_validity, ReduceOp, AVG_KERNEL_ENTRY, BLOCK_SIZE,
    REDUCTION_KERNEL_ENTRY, REDUCTION_KERNEL_WITH_VALIDITY_ENTRY,
};
use crate::exec::validity_audit::packed_validity_for;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Schema};

// `CudaModule` import dropped: every load site now routes through
// `exec::module_cache::get_or_build_module_for_scalar_agg` (v0.7 — keys on
// the `ScalarAggSpec` planner IR), which returns the cached module directly.
use crate::plan::physical_plan::{
    AggregateSpec, ColumnIO, PhysicalPlan, ScalarAggOp, ScalarAggSpec,
};

// v0.6 pilot helper `upload_primitive_values_async` was promoted to
// `crate::exec::gpu_upload` in v0.7 so the filter / GROUP BY / join
// executors can share the same `(slice, &stream) -> GpuVec<T>` shape with
// the identical `--features cuda-stub` graceful fallback. The aggregate
// path uses it through the canonical name so the migration is a
// drop-in import swap — no semantic change.
use crate::exec::gpu_upload::upload_primitive_values_async;

/// Build the packed-bit validity bitmap for `arr` and upload it as a
/// `GpuVec<u8>` on `stream`, ready to feed the `_with_validity` reduction
/// kernel. Mirrors [`upload_primitive_values_async`]'s `cuda-stub` fallback.
#[inline]
fn upload_validity_async(arr: &dyn Array, stream: &CudaStream) -> BoltResult<GpuVec<u8>> {
    let packed = packed_validity_for(arr);
    upload_primitive_values_async::<u8>(&packed, stream)
}

/// Execute an aggregate physical plan against a host-side RecordBatch.
///
/// `table_batch` must already be the relevant batch for `plan` (the caller
/// resolves the table name to a batch).
#[tracing::instrument(name = "materialize", level = "info", skip_all, fields(n_rows = table_batch.num_rows()))]
pub fn execute_aggregate(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> BoltResult<RecordBatch> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        other => {
            return Err(BoltError::Other(format!(
                "execute_aggregate: expected Aggregate plan, got {:?}",
                std::mem::discriminant(other)
            )))
        }
    };

    if !aggregate.group_by.is_empty() {
        return Err(BoltError::Other(
            "GROUP BY aggregate not yet implemented".into(),
        ));
    }
    if pre.is_some() {
        return Err(BoltError::Other(
            "aggregate with projection/filter not yet implemented in scalar reduction path"
                .into(),
        ));
    }

    let n_rows = table_batch.num_rows();
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    let arrays = build_scalar_aggregates(aggregate, table_batch, n_rows)?;

    RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
        BoltError::Other(format!("failed to build aggregate RecordBatch: {e}"))
    })
}

/// Build one Arrow scalar array per `AggregateExpr`, in `aggregate.aggregates`
/// order, against the host-side `table_batch`.
fn build_scalar_aggregates(
    aggregate: &AggregateSpec,
    table_batch: &RecordBatch,
    n_rows: usize,
) -> BoltResult<Vec<ArrayRef>> {
    // The output schema has one field per aggregate (no group keys), in order.
    if aggregate.output_schema.fields.len() != aggregate.aggregates.len() {
        return Err(BoltError::Other(format!(
            "internal: aggregate output schema has {} fields but plan has {} aggregates",
            aggregate.output_schema.fields.len(),
            aggregate.aggregates.len()
        )));
    }

    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(aggregate.aggregates.len());
    for (i, agg) in aggregate.aggregates.iter().enumerate() {
        let out_field = &aggregate.output_schema.fields[i];
        let array = build_one_aggregate(agg, out_field, &aggregate.inputs, table_batch, n_rows)?;
        arrays.push(array);
    }
    Ok(arrays)
}

/// Compute a single aggregate and return its single-row Arrow array.
fn build_one_aggregate(
    agg: &AggregateExpr,
    out_field: &Field,
    inputs: &[ColumnIO],
    table_batch: &RecordBatch,
    n_rows: usize,
) -> BoltResult<ArrayRef> {
    // Bool / Utf8 aggregate inputs go through the host-side extended_agg path
    // — the GPU reduction kernels only know primitives.
    if let Some(inner) = agg_inner_expr(agg) {
        if let Ok(col_name) = bare_column_name(inner) {
            if let Ok(col_io) = resolve_input(inputs, col_name) {
                if crate::exec::extended_agg::handles(agg, col_io.dtype) {
                    return crate::exec::extended_agg::execute_extended_scalar(
                        agg, out_field, table_batch,
                    );
                }
            }
        }
    }

    match agg {
        AggregateExpr::Sum(expr) | AggregateExpr::Min(expr) | AggregateExpr::Max(expr) => {
            let op = ReduceOp::from_agg(agg)?;
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;
            // SQL semantics: SUM/MIN/MAX over an empty or all-NULL input is
            // NULL, not the reduction identity (0 for SUM, ±inf / i64::MAX /
            // i64::MIN for MIN/MAX). Detect the zero-valid-rows case up front
            // and emit a single NULL row when the planner has marked the
            // output field nullable — mirroring the Decimal128 MIN/MAX fold
            // (`minmax_decimal128_from_batch`, seed `None`) and the AVG path
            // (`avg_result_array`). The GPU/host reduction below still seeds
            // from the identity for the non-empty happy path, which is only
            // reached when at least one valid row exists. (If the field were
            // somehow non-nullable we fall through to the legacy identity so
            // `RecordBatch::try_new` doesn't reject a null in a non-nullable
            // column.)
            if out_field.nullable
                && non_null_count_for_input(col_io, table_batch)? == 0
            {
                return null_scalar_array(out_field.dtype);
            }
            // Decimal128: SUM via the dedicated i128 GPU block-reduce kernel
            // (`decimal_sum_from_batch`; host-fold fallback inside it). MIN/MAX
            // via the host fold over the decoded `Decimal128Array` (raw i128
            // ordering == decimal ordering at the column's uniform scale).
            if let DataType::Decimal128(p, s) = col_io.dtype {
                match op {
                    ReduceOp::Sum => {
                        let scalar = decimal_sum_from_batch(col_io, table_batch, n_rows)?;
                        return scalar_to_array(scalar, out_field.dtype);
                    }
                    ReduceOp::Min | ReduceOp::Max => {
                        return minmax_decimal128_from_batch(
                            op, col_io, table_batch, p, s, out_field,
                        );
                    }
                    ReduceOp::Count => {}
                }
            }
            let scalar = reduce_column_from_batch(op, col_io, table_batch, n_rows)?;
            scalar_to_array(scalar, out_field.dtype)
        }
        AggregateExpr::StddevPop(expr) | AggregateExpr::StddevSamp(expr) => {
            // Welford one-pass reduction on the host. The scalar-aggregate
            // path's GPU offload is a v0.6 stretch goal — for v0.5 we
            // download (or already have) the values as a host slice and
            // fold them via `WelfordState::push`. The output dtype is
            // always Float64; STDDEV_SAMP packs SQL NULL when count <= 1
            // (so the output field's `nullable = true` is load-bearing for
            // that path). See `crate::exec::welford` for the canonical
            // numerics.
            //
            // `expr` is `&Box<Expr>` (the enum variant's payload is
            // `Box<Expr>`); explicitly deref through `.as_ref()` so the
            // borrowed-`&Expr` shape matches `bare_column_name`'s signature.
            let col_name = bare_column_name(expr.as_ref())?;
            let col_io = resolve_input(inputs, col_name)?;
            let state = welford_state_from_batch(col_io, table_batch)?;
            let kind = match agg {
                AggregateExpr::StddevPop(_) => crate::exec::welford::StddevKind::Pop,
                AggregateExpr::StddevSamp(_) => crate::exec::welford::StddevKind::Samp,
                _ => unreachable!("matched above"),
            };
            stddev_to_array(crate::exec::welford::finalize(&state, kind), agg, out_field)
        }
        AggregateExpr::Count(expr) => {
            // COUNT(col) excludes NULL inputs; COUNT(*) (with a literal-ish
            // expression that doesn't resolve to a column) returns the row
            // count. We mirror the SQL standard: if the expression is a bare
            // column reference, count non-null rows of that column; otherwise
            // count every row.
            let count: i64 = match bare_column_name(expr)
                .ok()
                .and_then(|name| resolve_input(inputs, name).ok())
            {
                Some(col_io) => non_null_count_for_input(col_io, table_batch)? as i64,
                None => n_rows as i64,
            };
            scalar_to_array(Scalar::I64(count), out_field.dtype)
        }
        AggregateExpr::Avg(expr) => {
            // AVG via the **fused** kernel: one launch produces both the
            // numerator (per-block `f64` sum) and the denominator (per-block
            // `u32` count) in a single pass. The host sums each partial
            // buffer and divides. Replaces the previous "SUM kernel + host
            // non-NULL count + host divide" shape — same number of kernel
            // launches, but the count now reflects what the GPU actually
            // saw rather than relying on the Arrow null bitmap, and the
            // implementation generalises cleanly to a future post-pre-stage
            // path where the host doesn't know the post-filter count.
            //
            // F4: SQL returns NULL for AVG over zero matching (non-NULL)
            // rows. When the planner has marked the AVG output field nullable
            // we surface that NULL directly; if the field is still
            // non-nullable (legacy contract — RecordBatch::try_new would
            // reject a null in a non-nullable column) we fall back to 0.0 to
            // preserve the build. Making the field unconditionally nullable
            // remains a planner-side change tracked separately.
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;
            let (sum_f64, count_u64) =
                fused_avg_from_batch(col_io, table_batch, n_rows)?;
            Ok(avg_result_array(sum_f64, count_u64, out_field.nullable))
        }
        AggregateExpr::VarPop(expr) | AggregateExpr::VarSamp(expr) => {
            // v0.5 scalar-aggregate path: download the column to the host
            // and run Welford's online algorithm in f64. The output is
            // nullable Float64 — SQL says VAR_POP/VAR_SAMP over an empty
            // (or all-NULL) input is NULL, and VAR_SAMP additionally
            // requires count > 1. A future patch can lower this to a GPU
            // kernel emitting per-block (count, mean, M2) partials that
            // merge in `O(blocks)` on the host; the wire-format used here
            // is the Welford state, so swapping the launcher is contract-
            // preserving.
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;
            let values_f64 = column_as_f64_no_nulls(col_io, table_batch)?;
            let is_pop = matches!(agg, AggregateExpr::VarPop(_));
            let result: Option<f64> = if is_pop {
                crate::exec::welford::var_pop_f64(&values_f64)
            } else {
                crate::exec::welford::var_samp_f64(&values_f64)
            };
            Ok(Arc::new(Float64Array::from(vec![result])) as ArrayRef)
        }
    }
}

/// Pull `col_io` out of `batch` as an `f64` vector, dropping NULL rows.
/// Used by the host-side Welford path; mirrors the NULL-filtering done
/// by `reduce_column_from_batch` for SUM/MIN/MAX but always upcasts to
/// the f64 accumulator dtype.
fn column_as_f64_no_nulls(
    col_io: &ColumnIO,
    batch: &RecordBatch,
) -> BoltResult<Vec<f64>> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);
    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != col_io.dtype {
        return Err(BoltError::Type(format!(
            "aggregate input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
            col_io.name, col_io.dtype, arr_dtype
        )));
    }
    match col_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int32"))?;
            Ok(primitive_to_f64_dropping_nulls::<arrow_array::types::Int32Type>(
                pa,
                |v| v as f64,
            ))
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            Ok(primitive_to_f64_dropping_nulls::<arrow_array::types::Int64Type>(
                pa,
                |v| v as f64,
            ))
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            Ok(primitive_to_f64_dropping_nulls::<arrow_array::types::Float32Type>(
                pa,
                |v| v as f64,
            ))
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            Ok(primitive_to_f64_dropping_nulls::<arrow_array::types::Float64Type>(
                pa,
                |v| v,
            ))
        }
        DataType::Bool
        | DataType::Utf8
        | DataType::Decimal128(_, _)
        | DataType::Date32
        | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
            "VAR_POP/VAR_SAMP over dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// Copy a primitive Arrow array's non-NULL values into a fresh `Vec<f64>`,
/// applying `cast` to widen each element. NULL positions are skipped so
/// the resulting slice can be fed straight into Welford.
fn primitive_to_f64_dropping_nulls<P>(
    pa: &arrow_array::PrimitiveArray<P>,
    cast: impl Fn(P::Native) -> f64,
) -> Vec<f64>
where
    P: arrow_array::types::ArrowPrimitiveType,
    P::Native: Copy,
{
    let n = pa.len();
    let mut out: Vec<f64> = Vec::with_capacity(n - pa.null_count());
    let vals = pa.values();
    for i in 0..n {
        if !pa.is_null(i) {
            out.push(cast(vals[i]));
        }
    }
    out
}

/// Count of non-NULL rows for `col_io` in `batch`. Used by COUNT(col) and as
/// the AVG denominator so neither includes garbage at NULL positions.
fn non_null_count_for_input(
    col_io: &ColumnIO,
    batch: &RecordBatch,
) -> BoltResult<usize> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);
    Ok(arr.len() - arr.null_count())
}

/// Borrow the inner expression of an `AggregateExpr`, regardless of variant.
/// Used by the Bool/Utf8 dispatch to peek at the input without committing to a
/// reduction op yet.
fn agg_inner_expr(agg: &AggregateExpr) -> Option<&Expr> {
    match agg {
        AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e)
        | AggregateExpr::Count(e) => Some(e),
        AggregateExpr::VarPop(e) | AggregateExpr::VarSamp(e) => Some(e),
        // STDDEV_* hold their operand boxed; deref to expose the inner Expr.
        AggregateExpr::StddevPop(e) | AggregateExpr::StddevSamp(e) => Some(e.as_ref()),
    }
}

/// Resolve `name` to its `(index, ColumnIO)` within `inputs`.
fn resolve_input<'a>(inputs: &'a [ColumnIO], name: &str) -> BoltResult<&'a ColumnIO> {
    inputs
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| {
            BoltError::Plan(format!(
                "aggregate input column '{}' not found in plan inputs",
                name
            ))
        })
}

/// Extract the column name from a bare-column-ref expression. The first cut
/// of `execute_aggregate` requires every aggregate to be over a column ref
/// (the physical-plan lowering guarantees this when `pre` is `None`).
fn bare_column_name(expr: &Expr) -> BoltResult<&str> {
    match expr {
        Expr::Column(name) => Ok(name.as_str()),
        Expr::Alias(inner, _) => bare_column_name(inner),
        _ => Err(BoltError::Other(
            "aggregate input must be a bare column reference in the scalar reduction path"
                .into(),
        )),
    }
}

/// Host-side MIN/MAX over a `Decimal128` aggregate input.
///
/// Like the `SUM(Decimal128)` host fold (`decimal_sum_from_batch` /
/// `decimal_sum_host`), this exists because the GPU reduction
/// kernels only handle 32/64-bit primitives — a 128-bit MIN/MAX kernel is a
/// follow-up optimisation. We fold over the already-decoded
/// `Decimal128Array` on the host. Because the scale is uniform across every
/// row of a column, the raw `i128` ordering is identical to the decimal
/// ordering, so we compare the underlying `i128` values directly.
///
/// NULL handling mirrors the SUM path: NULL rows are skipped (validity is
/// respected). Unlike SUM — whose empty/all-NULL identity is 0 — MIN/MAX of
/// an empty or all-NULL input is SQL NULL, so we pack a single NULL row in
/// that case (matching the `out_field.nullable` contract the planner
/// declares for MIN/MAX).
///
/// MIN/MAX preserve the input column's `(precision, scale)` (the planner's
/// `AggregateExpr::output_dtype` rule for MIN/MAX is identity — no widening,
/// unlike SUM which goes to `Decimal128(38, s)`). We pack with the input
/// `(precision, scale)` and validate that `out_field.dtype` agrees, surfacing
/// a loud Type error on any planner/executor mismatch.
fn minmax_decimal128_from_batch(
    op: ReduceOp,
    col_io: &ColumnIO,
    batch: &RecordBatch,
    precision: u8,
    scale: i8,
    out_field: &Field,
) -> BoltResult<ArrayRef> {
    debug_assert!(
        matches!(op, ReduceOp::Min | ReduceOp::Max),
        "minmax_decimal128_from_batch called with non-MIN/MAX op"
    );

    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);

    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != col_io.dtype {
        return Err(BoltError::Type(format!(
            "aggregate input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
            col_io.name, col_io.dtype, arr_dtype
        )));
    }

    let da = arr
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| downcast_err(&col_io.name, "Decimal128"))?;

    // The Arrow `Decimal128Array` carries its own (p, s); if it disagrees
    // with the column's plan-level dtype the schema is internally
    // inconsistent and any downstream consumer would mis-interpret the
    // values. Mirror the guard in `decimal_sum_from_batch`.
    if let ArrowDataType::Decimal128(ap, as_) = da.data_type() {
        if *ap != precision || *as_ != scale {
            return Err(BoltError::Type(format!(
                "MIN/MAX(Decimal128) column '{}': plan dtype Decimal128({precision}, {scale}) \
                 disagrees with Arrow dtype Decimal128({ap}, {as_})",
                col_io.name
            )));
        }
    }

    // Host-side fold: walk every row, skip NULLs, track the running
    // extremum by raw i128 value. The seed is `None` so that an empty or
    // all-NULL input yields SQL NULL (rather than a sentinel ±inf), matching
    // the MIN/MAX semantics the planner declares (`out_field.nullable`).
    let mut acc: Option<i128> = None;
    for i in 0..da.len() {
        if da.is_null(i) {
            continue;
        }
        let v: i128 = da.value(i);
        acc = Some(match acc {
            None => v,
            Some(cur) => match op {
                ReduceOp::Min => cur.min(v),
                ReduceOp::Max => cur.max(v),
                // Unreachable: guarded by the debug_assert and the caller's
                // dispatch, which only routes Min/Max here.
                _ => cur,
            },
        });
    }

    // MIN/MAX preserve the input (p, s); the planner declares the output
    // dtype as the column's own Decimal128(p, s). Validate the declared
    // output field agrees so a planner/executor mismatch is loud.
    let (out_p, out_s) = match out_field.dtype {
        DataType::Decimal128(p, s) => (p, s),
        ref other => {
            return Err(BoltError::Type(format!(
                "MIN/MAX(Decimal128) output field dtype must be Decimal128, got {:?}",
                other
            )));
        }
    };
    if out_p != precision || out_s != scale {
        return Err(BoltError::Type(format!(
            "MIN/MAX(Decimal128) output dtype Decimal128({out_p}, {out_s}) disagrees with \
             input dtype Decimal128({precision}, {scale})"
        )));
    }

    // `Decimal128Array::from(vec![Option<i128>])` packs a single row whose
    // validity bit follows the `Option`: `None` => SQL NULL.
    let arr = Decimal128Array::from(vec![acc])
        .with_precision_and_scale(out_p, out_s)
        .map_err(|e| {
            BoltError::Type(format!(
                "MIN/MAX(Decimal128) result: precision/scale ({out_p}, {out_s}) \
                 rejected by Arrow: {e}"
            ))
        })?;
    Ok(Arc::new(arr) as ArrayRef)
}

/// Pull `col_io` out of the batch and run a GPU reduction over it.
fn reduce_column_from_batch(
    op: ReduceOp,
    col_io: &ColumnIO,
    batch: &RecordBatch,
    n_rows: usize,
) -> BoltResult<Scalar> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);

    // Sanity-check the dtype matches what the plan promised.
    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != col_io.dtype {
        return Err(BoltError::Type(format!(
            "aggregate input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
            col_io.name, col_io.dtype, arr_dtype
        )));
    }

    // NULL handling: when `null_count > 0` the raw `.values()` buffer carries
    // garbage at NULL positions which would be silently included in
    // SUM/MIN/MAX. We detect that case and filter to a host vector of just the
    // non-null values before uploading; the GPU reduction then operates on a
    // post-filter prefix matching the natural identity (0 for SUM, +inf for
    // MIN, -inf for MAX) at the (zero) NULL positions. The fast path uploads
    // Arrow's value buffer directly through the async H2D wrapper when there
    // are no nulls.
    let has_nulls = arr.null_count() > 0;

    // v0.6 async-memcpy pilot: build a per-call stream and chain H2D-upload →
    // kernel-launch → D2H-partials on it, syncing exactly once at the end.
    // `null_or_default` falls back to the NULL stream under `cuda-stub` (and
    // any other host without a working `cuStreamCreate`).
    let stream = CudaStream::null_or_default();
    match col_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int32"))?;
            // SUM(Int32) widens to Int64 (see
            // `crate::plan::logical_plan::sum_output_dtype`): the GPU kernel
            // sign-extends each value and accumulates in s64, so the partials
            // buffer and host-side finalization must also be i64. MIN/MAX
            // preserve the input dtype and use the i32 path.
            if has_nulls {
                // Native-validity path: upload the RAW value buffer plus a
                // packed validity bitmap and let the kernel fold NULL rows to
                // the identity — no host strip.
                let validity_gpu = upload_validity_async(arr, &stream)?;
                let dev = upload_primitive_values_async::<i32>(pa.values(), &stream)?;
                if matches!(op, ReduceOp::Sum) {
                    reduce_gpu_vec_widened_with_validity::<i32, i64>(
                        op, col_io.dtype, &dev, &validity_gpu, n_rows, &stream,
                    )
                } else {
                    reduce_gpu_vec_with_validity::<i32>(
                        op, col_io.dtype, &dev, &validity_gpu, n_rows, &stream,
                    )
                }
            } else {
                // No-null fast path: async H2D of Arrow's value buffer on
                // `stream`, then the kernel + partials D2H ride the same
                // stream. v0.6 pilot replaced the synchronous
                // `primitive_to_gpu`+`from_buffer` pair here.
                let dev = upload_primitive_values_async::<i32>(pa.values(), &stream)?;
                if matches!(op, ReduceOp::Sum) {
                    reduce_gpu_vec_widened::<i32, i64>(
                        op, col_io.dtype, &dev, n_rows, &stream,
                    )
                } else {
                    reduce_gpu_vec::<i32>(op, col_io.dtype, &dev, n_rows, &stream)
                }
            }
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            if has_nulls {
                let validity_gpu = upload_validity_async(arr, &stream)?;
                let dev = upload_primitive_values_async::<i64>(pa.values(), &stream)?;
                reduce_gpu_vec_with_validity::<i64>(
                    op, col_io.dtype, &dev, &validity_gpu, n_rows, &stream,
                )
            } else {
                let dev = upload_primitive_values_async::<i64>(pa.values(), &stream)?;
                reduce_gpu_vec::<i64>(op, col_io.dtype, &dev, n_rows, &stream)
            }
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            if has_nulls {
                let validity_gpu = upload_validity_async(arr, &stream)?;
                let dev = upload_primitive_values_async::<f32>(pa.values(), &stream)?;
                reduce_gpu_vec_with_validity::<f32>(
                    op, col_io.dtype, &dev, &validity_gpu, n_rows, &stream,
                )
            } else {
                let dev = upload_primitive_values_async::<f32>(pa.values(), &stream)?;
                reduce_gpu_vec::<f32>(op, col_io.dtype, &dev, n_rows, &stream)
            }
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            if has_nulls {
                let validity_gpu = upload_validity_async(arr, &stream)?;
                let dev = upload_primitive_values_async::<f64>(pa.values(), &stream)?;
                reduce_gpu_vec_with_validity::<f64>(
                    op, col_io.dtype, &dev, &validity_gpu, n_rows, &stream,
                )
            } else {
                let dev = upload_primitive_values_async::<f64>(pa.values(), &stream)?;
                reduce_gpu_vec::<f64>(op, col_io.dtype, &dev, n_rows, &stream)
            }
        }
        DataType::Bool
        | DataType::Utf8
        | DataType::Decimal128(_, _)
        | DataType::Date32
        | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
            "aggregate input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// Run the **fused** AVG reduction over an aggregate input column, returning
/// `(sum_as_f64, non_null_count)`. The grand-total finalize (and divide-by-zero
/// guard) is done by the caller.
///
/// Layout: one GPU launch produces a pair of per-block partial buffers
/// (`block_sums: f64`, `block_counts: u32`); the host sums each to a single
/// F4: build the single-row AVG result array from a fused `(sum, count)`
/// partial. SQL returns NULL for AVG over zero matching (non-NULL) rows;
/// when the output field is nullable we surface that NULL, otherwise we fall
/// back to `0.0` (a null in a non-nullable column would be rejected by
/// `RecordBatch::try_new`). Shared by the scalar and pre-stage AVG paths so
/// the empty-input semantics stay identical.
pub(crate) fn avg_result_array(
    sum_f64: f64,
    count_u64: u64,
    out_nullable: bool,
) -> ArrayRef {
    if count_u64 == 0 {
        if out_nullable {
            return Arc::new(Float64Array::from(vec![Option::<f64>::None])) as ArrayRef;
        }
        return Arc::new(Float64Array::from(vec![0.0_f64])) as ArrayRef;
    }
    Arc::new(Float64Array::from(vec![sum_f64 / count_u64 as f64])) as ArrayRef
}

/// `(f64, u64)` pair. This replaces the previous "two kernels (SUM + COUNT) +
/// host divide" decomposition — one PTX compilation, one launch, one D2H
/// stream-synchronize.
///
/// NULL handling mirrors `reduce_column_from_batch`: when the input has any
/// NULLs we filter them on the host before uploading, so the GPU never sees
/// garbage values at NULL positions and `count` reflects the post-filter row
/// count.
fn fused_avg_from_batch(
    col_io: &ColumnIO,
    batch: &RecordBatch,
    n_rows: usize,
) -> BoltResult<(f64, u64)> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);

    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != col_io.dtype {
        return Err(BoltError::Type(format!(
            "aggregate input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
            col_io.name, col_io.dtype, arr_dtype
        )));
    }

    let has_nulls = arr.null_count() > 0;
    let stream = CudaStream::null_or_default();

    // Per-dtype: build a `GpuVec` (async H2D of Arrow's value buffer when
    // NULL-free, host-filtered upload otherwise) and dispatch to the fused
    // launcher. The launcher is monomorphic on the input dtype because
    // `compile_avg_reduction_kernel` emits dtype-specific PTX. v0.6 pilot:
    // every upload site routes through `upload_primitive_values_async` so
    // the `cuda-stub` graceful fallback lives in one place.
    match col_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int32"))?;
            if has_nulls {
                let host: Vec<i32> = filter_primitive_to_vec(pa);
                let len = host.len();
                let dev = upload_primitive_values_async::<i32>(&host, &stream)?;
                fused_avg_gpu_vec::<i32>(col_io.dtype, &dev, len, &stream)
            } else {
                let dev = upload_primitive_values_async::<i32>(pa.values(), &stream)?;
                fused_avg_gpu_vec::<i32>(col_io.dtype, &dev, n_rows, &stream)
            }
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            if has_nulls {
                let host: Vec<i64> = filter_primitive_to_vec(pa);
                let len = host.len();
                let dev = upload_primitive_values_async::<i64>(&host, &stream)?;
                fused_avg_gpu_vec::<i64>(col_io.dtype, &dev, len, &stream)
            } else {
                let dev = upload_primitive_values_async::<i64>(pa.values(), &stream)?;
                fused_avg_gpu_vec::<i64>(col_io.dtype, &dev, n_rows, &stream)
            }
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            if has_nulls {
                let host: Vec<f32> = filter_primitive_to_vec(pa);
                let len = host.len();
                let dev = upload_primitive_values_async::<f32>(&host, &stream)?;
                fused_avg_gpu_vec::<f32>(col_io.dtype, &dev, len, &stream)
            } else {
                let dev = upload_primitive_values_async::<f32>(pa.values(), &stream)?;
                fused_avg_gpu_vec::<f32>(col_io.dtype, &dev, n_rows, &stream)
            }
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            if has_nulls {
                let host: Vec<f64> = filter_primitive_to_vec(pa);
                let len = host.len();
                let dev = upload_primitive_values_async::<f64>(&host, &stream)?;
                fused_avg_gpu_vec::<f64>(col_io.dtype, &dev, len, &stream)
            } else {
                let dev = upload_primitive_values_async::<f64>(pa.values(), &stream)?;
                fused_avg_gpu_vec::<f64>(col_io.dtype, &dev, n_rows, &stream)
            }
        }
        DataType::Bool
        | DataType::Utf8
        | DataType::Decimal128(_, _)
        | DataType::Date32
        | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
            "AVG over dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// Launch the fused AVG reduction kernel against an already-uploaded device
/// buffer, then finalize the per-block partials on the host. Returns
/// `(sum_f64, count_u64)`. `dtype` is the *input* element dtype; the kernel
/// always emits `f64` sum and `u32` count partials regardless.
fn fused_avg_gpu_vec<TIn>(
    dtype: DataType,
    input: &GpuVec<TIn>,
    n_rows: usize,
    stream: &CudaStream,
) -> BoltResult<(f64, u64)>
where
    TIn: Pod,
{
    // 0-row degenerate case: skip the launch (and the PTX compile) entirely.
    if n_rows == 0 {
        stream.synchronize()?;
        return Ok((0.0, 0));
    }

    let block = BLOCK_SIZE;
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let grid_x = grid_x_for(n_rows_u32, block);

    // Per-block partial buffers: f64 sums + u32 counts, one element per block.
    let block_sums = GpuVec::<f64>::zeros_async(grid_x as usize, stream.raw())?;
    let block_counts = GpuVec::<u32>::zeros_async(grid_x as usize, stream.raw())?;

    // v0.7 `ScalarAggSpec` cache layer: the fused AVG kernel is keyed by
    // `(Avg, input_dtype)` and uses the `bolt_avg_reduce` entry point. Same
    // layering as the non-fused reduction path (in-memory → disk → PTX-text
    // hash inside `CudaModule::from_ptx`); separate cache slot from `Sum`
    // because the fused PTX is structurally different (two output buffers).
    let spec = ScalarAggSpec {
        op: ScalarAggOp::Avg,
        input_dtype: dtype,
    };
    let module = crate::exec::module_cache::get_or_build_module_for_scalar_agg(
        &spec,
        AVG_KERNEL_ENTRY,
        |_| compile_avg_reduction_kernel(dtype),
    )?;
    let function = module.function(AVG_KERNEL_ENTRY)?;

    let mut input_ptr: CUdeviceptr = input.device_ptr();
    let mut sums_ptr: CUdeviceptr = block_sums.device_ptr();
    let mut counts_ptr: CUdeviceptr = block_counts.device_ptr();

    let mut kernel_params: [*mut c_void; 4] = [
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut sums_ptr as *mut CUdeviceptr as *mut c_void,
        &mut counts_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
    ];

    // SAFETY: `function` is borrowed from a live `CudaModule`; every entry of
    // `kernel_params` points to a stack local that outlives the synchronize.
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            kernel_params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }

    let _ = input_ptr;
    let _ = sums_ptr;
    let _ = counts_ptr;

    // Async D2H for both partial buffers, then a single sync.
    let pinned_sums = block_sums.to_pinned_async(stream.raw())?;
    let pinned_counts = block_counts.to_pinned_async(stream.raw())?;
    stream.synchronize()?;

    // Neumaier-compensated host finalize over the per-block f64 partials so
    // the AVG numerator matches the GPU's tree-order sum to low bits and
    // tracks DuckDB's compensated summation (naive left-fold drifts).
    let total_sum: f64 = neumaier_sum_f64(pinned_sums.as_slice().iter().copied());
    let total_count: u64 = pinned_counts
        .as_slice()
        .iter()
        .copied()
        .map(u64::from)
        .sum();

    drop(pinned_sums);
    drop(pinned_counts);
    drop(block_sums);
    drop(block_counts);

    Ok((total_sum, total_count))
}

/// Compute `SUM(Decimal128)` over an aggregate input column.
///
/// `Decimal128` is a fixed-width `i128`, so the sum is a 128-bit integer
/// reduction. We dispatch to the dedicated GPU block-reduce kernel
/// (`crate::jit::decimal_agg`) which carries the accumulator as hi/lo `u64`
/// halves with a carry-chain add and writes one `i128` partial per block; the
/// host folds the partials. The GPU path is **reachable-but-skippable**: if it
/// declines (today: a degenerate zero-row input or a non-Decimal column), the
/// caller still gets a correct result via the host fold in
/// [`decimal_sum_host`].
///
/// NULL handling mirrors `reduce_column_from_batch`: NULL rows are stripped on
/// the host before upload so the kernel never sees garbage at masked positions.
/// The returned [`Scalar::Decimal128`] carries the column's `(precision,
/// scale)` so the caller can build a correctly-typed output array.
fn decimal_sum_from_batch(
    col_io: &ColumnIO,
    batch: &RecordBatch,
    n_rows: usize,
) -> BoltResult<Scalar> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);

    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != col_io.dtype {
        return Err(BoltError::Type(format!(
            "aggregate input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
            col_io.name, col_io.dtype, arr_dtype
        )));
    }
    let (precision, scale) = match col_io.dtype {
        DataType::Decimal128(p, s) => (p, s),
        other => {
            return Err(BoltError::Type(format!(
                "decimal_sum_from_batch called on non-Decimal column '{}' (dtype {:?})",
                col_io.name, other
            )))
        }
    };

    let da = arr
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| downcast_err(&col_io.name, "Decimal128"))?;

    // Strip NULLs into a dense host Vec<i128>. SUM ignores NULL inputs; the
    // dense buffer is also exactly what both the GPU upload and the host
    // fallback consume.
    let mut host: Vec<i128> = Vec::with_capacity(da.len() - da.null_count());
    for i in 0..da.len() {
        if !da.is_null(i) {
            host.push(da.value(i));
        }
    }

    let total = match decimal_sum_gpu(&host)? {
        Some(v) => v,
        // GPU path declined (e.g. empty input) — fold on the host. Correct and
        // cheap; this is the documented graceful fallback.
        None => decimal_sum_host(&host)?,
    };
    let _ = n_rows; // n_rows is the pre-strip row count; we sum the survivors.
    Ok(Scalar::Decimal128(total, precision, scale))
}

/// Host-side `i128` fold for `SUM(Decimal128)`. Uses `checked_add` and surfaces
/// a `BoltError::Type` on overflow rather than wrapping, honouring the engine's
/// never-silently-wrong invariant — the same contract the integer SUM path
/// (`SUM(integer) overflow`) and the planner doc in
/// `crate::plan::logical_plan` (see `sum_output_dtype`) describe. An empty
/// input folds to the identity 0 (SUM over no rows).
fn decimal_sum_host(host: &[i128]) -> BoltResult<i128> {
    let mut acc: i128 = 0;
    for &v in host {
        acc = acc.checked_add(v).ok_or_else(|| {
            BoltError::Type(
                "SUM(Decimal128) precision overflow: accumulator exceeds i128 range".to_string(),
            )
        })?;
    }
    Ok(acc)
}

/// Launch the Decimal128 SUM block-reduce kernel over `host` and fold the
/// per-block `i128` partials. Returns `Ok(None)` when the GPU path declines
/// (empty input — nothing to launch), in which case the caller folds on the
/// host. Any hard GPU error propagates as `Err`.
fn decimal_sum_gpu(host: &[i128]) -> BoltResult<Option<i128>> {
    use crate::jit::decimal_agg::{compile_decimal_sum_kernel, DECIMAL_SUM_KERNEL_ENTRY};

    // Empty input: skip the launch + PTX compile entirely and let the caller
    // take the trivial host fold (sum of nothing == 0).
    if host.is_empty() {
        return Ok(None);
    }

    let stream = CudaStream::null_or_default();
    let dev = upload_primitive_values_async::<i128>(host, &stream)?;

    let block = BLOCK_SIZE;
    let mut n_rows_u32: u32 = n_rows_to_u32(host.len())?;
    let grid_x = grid_x_for(n_rows_u32, block);

    // One i128 partial per block.
    let partials = GpuVec::<i128>::zeros_async(grid_x as usize, stream.raw())?;

    let module = module_cache::get_or_build_module(
        module_path!(),
        "decimal_sum_reduce".to_string(),
        None,
        compile_decimal_sum_kernel,
    )?;
    let function = module.function(DECIMAL_SUM_KERNEL_ENTRY)?;

    let mut input_ptr: CUdeviceptr = dev.device_ptr();
    let mut output_ptr: CUdeviceptr = partials.device_ptr();
    let mut kernel_params: [*mut c_void; 3] = [
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut output_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
    ];

    // SAFETY: `function` is borrowed from a live module; every param points at
    // a stack local that outlives the synchronize below.
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            kernel_params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    let _ = input_ptr;
    let _ = output_ptr;

    let pinned = partials.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_partials: Vec<i128> = pinned.as_slice().to_vec();
    drop(pinned);
    drop(partials);
    drop(dev);

    // Fold the per-block partials with the same checked add as the host path,
    // so a sum that overflows i128 errors loudly whether or not the GPU path
    // ran (mirrors the never-silently-wrong invariant).
    Ok(Some(decimal_sum_host(&host_partials)?))
}

/// Build a Welford `(count, mean, M2)` state by folding the non-NULL values
/// of `col_io` from `batch` in source order. Used by the scalar
/// `STDDEV_POP` / `STDDEV_SAMP` aggregates; the host-side fold is
/// acceptable for v0.5 (see `crate::exec::welford` module docs on host vs
/// device). All numeric input dtypes promote to `f64` at the push site so
/// the accumulator stays in double precision regardless of input width.
fn welford_state_from_batch(
    col_io: &ColumnIO,
    batch: &RecordBatch,
) -> BoltResult<crate::exec::welford::WelfordState> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "STDDEV input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);
    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != col_io.dtype {
        return Err(BoltError::Type(format!(
            "STDDEV input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
            col_io.name, col_io.dtype, arr_dtype
        )));
    }

    let mut state = crate::exec::welford::WelfordState::empty();
    match col_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int32"))?;
            for i in 0..pa.len() {
                if !pa.is_null(i) {
                    state.push(pa.value(i) as f64);
                }
            }
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            for i in 0..pa.len() {
                if !pa.is_null(i) {
                    state.push(pa.value(i) as f64);
                }
            }
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            for i in 0..pa.len() {
                if !pa.is_null(i) {
                    state.push(pa.value(i) as f64);
                }
            }
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            for i in 0..pa.len() {
                if !pa.is_null(i) {
                    state.push(pa.value(i));
                }
            }
        }
        DataType::Bool
        | DataType::Utf8
        | DataType::Decimal128(_, _)
        | DataType::Date32
        | DataType::Timestamp(_, _) => {
            return Err(BoltError::Type(format!(
                "STDDEV over dtype {:?} not supported (column '{}')",
                col_io.dtype, col_io.name
            )));
        }
    }
    Ok(state)
}

/// Pack a finalized stddev (`Some(σ)` or `None`) into a one-row Arrow
/// `Float64Array`.
///
/// * `STDDEV_POP` returns `0.0` on an empty input — mirrors the existing
///   AVG convention so the output schema field can stay non-nullable in
///   downstream consumers that don't yet handle the SQL NULL case.
/// * `STDDEV_SAMP` returns SQL NULL when `count <= 1` (the divisor is zero
///   or negative — undefined per the SQL standard); the output field is
///   nullable per `LogicalPlan::Aggregate` schema-construction.
///
/// `out_field.dtype` must be `Float64` (validated against the planner's
/// declared output schema); we surface a Type error otherwise so a
/// future plan that picks the wrong dtype fails loudly rather than
/// silently truncating.
fn stddev_to_array(
    value: Option<f64>,
    agg: &AggregateExpr,
    out_field: &Field,
) -> BoltResult<ArrayRef> {
    if out_field.dtype != DataType::Float64 {
        return Err(BoltError::Type(format!(
            "STDDEV output dtype must be Float64, got {:?}",
            out_field.dtype
        )));
    }
    match (agg, value) {
        (AggregateExpr::StddevPop(_), Some(v)) => {
            Ok(Arc::new(Float64Array::from(vec![v])) as ArrayRef)
        }
        (AggregateExpr::StddevPop(_), None) => {
            // Empty input: match the AVG-on-empty convention (return 0.0
            // rather than NULL) so the output column stays packed when the
            // SELECT-list re-projection consumes it. Documented in the
            // public stddev semantics on `welford::WelfordState`.
            Ok(Arc::new(Float64Array::from(vec![0.0_f64])) as ArrayRef)
        }
        (AggregateExpr::StddevSamp(_), Some(v)) => {
            Ok(Arc::new(Float64Array::from(vec![v])) as ArrayRef)
        }
        (AggregateExpr::StddevSamp(_), None) => {
            // count <= 1 → SQL NULL. The output field is nullable in the
            // logical plan's Aggregate schema (every aggregate output is),
            // so a single-element nullable Float64Array packs cleanly here.
            Ok(Arc::new(Float64Array::from(vec![None::<f64>])) as ArrayRef)
        }
        _ => Err(BoltError::Other(
            "stddev_to_array called with non-STDDEV aggregate".into(),
        )),
    }
}

/// Copy the non-NULL values of an Arrow primitive array into a fresh `Vec`.
/// Used in the `null_count > 0` path so the GPU reduction never sees garbage
/// at masked positions. Order is preserved.
fn filter_primitive_to_vec<P>(pa: &arrow_array::PrimitiveArray<P>) -> Vec<P::Native>
where
    P: arrow_array::types::ArrowPrimitiveType,
    P::Native: Copy,
{
    let n = pa.len();
    let mut out: Vec<P::Native> = Vec::with_capacity(n - pa.null_count());
    let vals = pa.values();
    for i in 0..n {
        if !pa.is_null(i) {
            out.push(vals[i]);
        }
    }
    out
}

/// Upload a host slice, then run the standard GPU reduction over it. Used by
/// COUNT (which synthesizes an all-ones column on the host) and by the
/// has-nulls path which filters Arrow's value buffer into a fresh `Vec` first.
///
/// v0.6 async-memcpy pilot: drives the upload + reduction on a per-call
/// stream so the H2D and the partials D2H overlap with the kernel where
/// the driver allows it. Routes through [`upload_primitive_values_async`]
/// so the `cuda-stub` graceful fallback lives in one place.
///
/// Retained as a host-strip fallback helper after the scalar SUM/MIN/MAX
/// has-nulls path migrated to the native `_with_validity` reduction kernel
/// (see [`reduce_gpu_vec_with_validity`]); kept available for callers that
/// already hold a dense, NULL-free host slice.
#[allow(dead_code)]
fn reduce_host_slice<T>(op: ReduceOp, dtype: DataType, host: &[T]) -> BoltResult<Scalar>
where
    T: Pod + ReduceScalar,
{
    let stream = CudaStream::null_or_default();
    let dev = upload_primitive_values_async::<T>(host, &stream)?;
    reduce_gpu_vec::<T>(op, dtype, &dev, host.len(), &stream)
}

/// Launch the per-block reduction kernel against `input` and finish the
/// reduction on the host. Returns the final scalar as a `Scalar`.
///
/// Stage 2 async memcpy: the caller provides the `stream` already carrying
/// the input column's H2D upload — we enqueue the kernel and the partials
/// D2H on the same stream and synchronize exactly once at the end.
fn reduce_gpu_vec<T>(
    op: ReduceOp,
    dtype: DataType,
    input: &GpuVec<T>,
    n_rows: usize,
    stream: &CudaStream,
) -> BoltResult<Scalar>
where
    T: Pod + ReduceScalar,
{
    // 0-row degenerate case: skip the launch entirely and return the identity.
    // The stream may still have a pending (empty) H2D queued — synchronize so
    // the next user of the default-device stream doesn't observe stale work.
    if n_rows == 0 {
        stream.synchronize()?;
        return T::identity_scalar(op, dtype);
    }

    let block = BLOCK_SIZE;
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let grid_x = grid_x_for(n_rows_u32, block);
    // Stage 3: async-zero the partials buffer on the caller's stream so the
    // memset/launch/D2H chain serializes correctly without an explicit
    // barrier.
    let partials = GpuVec::<T>::zeros_async(grid_x as usize, stream.raw())?;

    // Compile + load the kernel via the **v0.7 `ScalarAggSpec` cache layer**
    // in `exec::module_cache`. Cache layering, outer-to-inner:
    //
    //   1. Process-wide `ScalarAggSpec`-keyed cache (this call): on a hit
    //      we skip codegen entirely and return a cloned `CudaModule`.
    //   2. Optional disk-backed PTX cache (consulted inside the call
    //      below) — domain-separated from the projection-path entries by
    //      the `"scalar_agg::"` key prefix.
    //   3. PTX-text-hash cache inside `CudaModule::from_ptx` — short-
    //      circuits the `cuModuleLoadDataEx` step for cross-spec PTX
    //      collisions.
    //
    // The spec captures the entire PTX-template parameter surface
    // (`(op, input_dtype)`); repeat scalar reductions of the same shape
    // skip PTX generation on the warm path.
    let spec = ScalarAggSpec {
        op: reduce_op_to_scalar_agg_op(op),
        input_dtype: dtype,
    };
    let module = module_cache::get_or_build_module_for_scalar_agg(
        &spec,
        REDUCTION_KERNEL_ENTRY,
        |_| compile_reduction_kernel(op, dtype),
    )?;
    let function = module.function(REDUCTION_KERNEL_ENTRY)?;

    // Assemble the kernel parameter list (input_ptr, output_ptr, n_rows).
    let mut input_ptr: CUdeviceptr = input.device_ptr();
    let mut output_ptr: CUdeviceptr = partials.device_ptr();

    let mut kernel_params: [*mut c_void; 3] = [
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut output_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
    ];

    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            kernel_params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }

    // The `_` writes are immediately superseded; silence the lints by reading
    // them post-launch in case a future reordering relies on the values.
    let _ = input_ptr;
    let _ = output_ptr;

    // Stage 3: async D2H into a pinned host buffer, sync once, then copy
    // into a regular Vec for the host-side finalization. The pinned hop
    // lets the driver DMA directly without staging through a bounce
    // buffer.
    let pinned = partials.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_partials: Vec<T> = pinned.as_slice().to_vec();
    drop(pinned);
    drop(partials);
    T::finalize(op, dtype, &host_partials)
}

/// Adapter: collapse the JIT-layer `ReduceOp` (which has a host-only
/// `Count` variant that lowers to `Sum` in PTX) into a planner-IR
/// `ScalarAggOp`. Keeps the two enums in their respective domains and
/// makes the lossy projection (`Count` → `Sum` would key-alias) explicit
/// at the boundary.
///
/// `Count` keeps its own `ScalarAggOp::Count` variant so the cache key
/// stays distinct from a `Sum` — this matters for the future case where
/// the kernel grows a `Count`-specific shortcut and we don't want a stale
/// `Sum` PTX entry to serve it.
fn reduce_op_to_scalar_agg_op(op: ReduceOp) -> ScalarAggOp {
    match op {
        ReduceOp::Sum => ScalarAggOp::Sum,
        ReduceOp::Min => ScalarAggOp::Min,
        ReduceOp::Max => ScalarAggOp::Max,
        ReduceOp::Count => ScalarAggOp::Count,
    }
}

/// Variant of `reduce_gpu_vec` for reductions whose accumulator dtype is wider
/// than the input dtype (currently only `SUM` over a narrow signed integer,
/// per the widening contract in `crate::plan::logical_plan::sum_output_dtype`).
///
/// `TIn` is the input column element type; `TAcc` is the accumulator and
/// partial-output element type. The JIT'd kernel sign-extends each input load
/// on the GPU side; this function only has to size the output buffer at
/// `TAcc` and finalize the host-side reduction at `TAcc`.
///
/// `dtype` is the *input* dtype (what the kernel-compiler expects) — kernel
/// emission internally derives the accumulator dtype using the same rule.
fn reduce_gpu_vec_widened<TIn, TAcc>(
    op: ReduceOp,
    dtype: DataType,
    input: &GpuVec<TIn>,
    n_rows: usize,
    stream: &CudaStream,
) -> BoltResult<Scalar>
where
    TIn: Pod,
    TAcc: Pod + ReduceScalar,
{
    // 0-row degenerate case: skip the launch entirely and return the
    // accumulator's identity at the widened dtype.
    if n_rows == 0 {
        // The accumulator dtype is what the output array expects, which is
        // the kernel-internal widened dtype. Look it up explicitly.
        let acc_dtype = crate::jit::agg_kernels::reduction_output_dtype(op, dtype);
        stream.synchronize()?;
        return TAcc::identity_scalar(op, acc_dtype);
    }

    let block = BLOCK_SIZE;
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let grid_x = grid_x_for(n_rows_u32, block);
    // Partials buffer is sized in accumulator elements.
    // Stage 3: async-zero on the caller's stream.
    let partials = GpuVec::<TAcc>::zeros_async(grid_x as usize, stream.raw())?;

    // Compile + load the kernel. The kernel takes the *input* dtype; it
    // internally widens to the accumulator dtype. Routed through the
    // v0.7 `ScalarAggSpec` cache layer; the cache key is identical to the
    // one `reduce_gpu_vec` uses for the same `(op, input dtype)` pair
    // (the accumulator dtype is derived inside the JIT layer and is not
    // part of the cache key), so a SUM(Int32) widened call hits the same
    // entry a same-key non-widened call would create.
    let spec = ScalarAggSpec {
        op: reduce_op_to_scalar_agg_op(op),
        input_dtype: dtype,
    };
    let module = module_cache::get_or_build_module_for_scalar_agg(
        &spec,
        REDUCTION_KERNEL_ENTRY,
        |_| compile_reduction_kernel(op, dtype),
    )?;
    let function = module.function(REDUCTION_KERNEL_ENTRY)?;

    let mut input_ptr: CUdeviceptr = input.device_ptr();
    let mut output_ptr: CUdeviceptr = partials.device_ptr();

    let mut kernel_params: [*mut c_void; 3] = [
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut output_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
    ];

    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            kernel_params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }

    let _ = input_ptr;
    let _ = output_ptr;

    // Stage 3 pinned D2H: download partials directly into pinned memory,
    // sync, then copy into a host Vec for the widened finalize.
    let pinned = partials.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_partials: Vec<TAcc> = pinned.as_slice().to_vec();
    drop(pinned);
    drop(partials);
    // Finalize at the accumulator dtype.
    let acc_dtype = crate::jit::agg_kernels::reduction_output_dtype(op, dtype);
    TAcc::finalize(op, acc_dtype, &host_partials)
}

// ---------------------------------------------------------------------------
// Native-validity reduction path.
//
// Instead of host-stripping NULL rows into a dense `Vec` and uploading the
// stripped slice, these launchers upload the RAW Arrow value buffer plus a
// packed-bit validity bitmap and dispatch to
// `compile_reduction_kernel_with_validity`. The kernel folds NULL rows to the
// reduction identity on the GPU, so SUM/MIN/MAX skip them and a COUNT(expr)
// (driven by an all-ones value column) contributes 0 per NULL row. This keeps
// the launch zero-host-strip when the planner / runtime says the column
// carries validity, matching the native `_with_validity` dispatch the GROUP BY
// executors already use (see `crate::exec::groupby_valid`).
// ---------------------------------------------------------------------------

/// Native-validity sibling of [`reduce_gpu_vec`]: launches
/// `bolt_reduce_with_validity` against an already-uploaded RAW value buffer
/// (NULL positions still hold garbage) plus the packed validity bitmap
/// `validity_gpu`. The kernel folds NULL rows to the identity, so the host
/// finalize is identical to the no-null path.
fn reduce_gpu_vec_with_validity<T>(
    op: ReduceOp,
    dtype: DataType,
    input: &GpuVec<T>,
    validity_gpu: &GpuVec<u8>,
    n_rows: usize,
    stream: &CudaStream,
) -> BoltResult<Scalar>
where
    T: Pod + ReduceScalar,
{
    if n_rows == 0 {
        stream.synchronize()?;
        return T::identity_scalar(op, dtype);
    }

    let block = BLOCK_SIZE;
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let grid_x = grid_x_for(n_rows_u32, block);
    let partials = GpuVec::<T>::zeros_async(grid_x as usize, stream.raw())?;

    let module = module_cache::get_or_build_module(
        module_path!(),
        format!("reduction_validity:{:?}:{:?}", op, dtype),
        None,
        || compile_reduction_kernel_with_validity(op, dtype),
    )?;
    let function = module.function(REDUCTION_KERNEL_WITH_VALIDITY_ENTRY)?;

    let mut input_ptr: CUdeviceptr = input.device_ptr();
    let mut output_ptr: CUdeviceptr = partials.device_ptr();
    let mut validity_ptr: CUdeviceptr = validity_gpu.device_ptr();

    // ABI: (input_ptr, output_ptr, n_rows, validity_ptr).
    let mut kernel_params: [*mut c_void; 4] = [
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut output_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut validity_ptr as *mut CUdeviceptr as *mut c_void,
    ];

    // SAFETY: `function` is borrowed from a live module; every param points
    // at a stack local that outlives the synchronize below.
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            kernel_params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }

    let _ = input_ptr;
    let _ = output_ptr;
    let _ = validity_ptr;

    let pinned = partials.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_partials: Vec<T> = pinned.as_slice().to_vec();
    drop(pinned);
    drop(partials);
    T::finalize(op, dtype, &host_partials)
}

/// Native-validity sibling of [`reduce_gpu_vec_widened`]: the widened SUM path
/// (narrow signed integer accumulating into a wider dtype) with a packed
/// validity bitmap. The kernel sign-extends each in-range, non-NULL value and
/// folds NULL rows to the (zero) additive identity.
fn reduce_gpu_vec_widened_with_validity<TIn, TAcc>(
    op: ReduceOp,
    dtype: DataType,
    input: &GpuVec<TIn>,
    validity_gpu: &GpuVec<u8>,
    n_rows: usize,
    stream: &CudaStream,
) -> BoltResult<Scalar>
where
    TIn: Pod,
    TAcc: Pod + ReduceScalar,
{
    if n_rows == 0 {
        let acc_dtype = crate::jit::agg_kernels::reduction_output_dtype(op, dtype);
        stream.synchronize()?;
        return TAcc::identity_scalar(op, acc_dtype);
    }

    let block = BLOCK_SIZE;
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let grid_x = grid_x_for(n_rows_u32, block);
    let partials = GpuVec::<TAcc>::zeros_async(grid_x as usize, stream.raw())?;

    let module = module_cache::get_or_build_module(
        module_path!(),
        format!("reduction_validity:{:?}:{:?}", op, dtype),
        None,
        || compile_reduction_kernel_with_validity(op, dtype),
    )?;
    let function = module.function(REDUCTION_KERNEL_WITH_VALIDITY_ENTRY)?;

    let mut input_ptr: CUdeviceptr = input.device_ptr();
    let mut output_ptr: CUdeviceptr = partials.device_ptr();
    let mut validity_ptr: CUdeviceptr = validity_gpu.device_ptr();

    let mut kernel_params: [*mut c_void; 4] = [
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut output_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut validity_ptr as *mut CUdeviceptr as *mut c_void,
    ];

    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            kernel_params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }

    let _ = input_ptr;
    let _ = output_ptr;
    let _ = validity_ptr;

    let pinned = partials.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_partials: Vec<TAcc> = pinned.as_slice().to_vec();
    drop(pinned);
    drop(partials);
    let acc_dtype = crate::jit::agg_kernels::reduction_output_dtype(op, dtype);
    TAcc::finalize(op, acc_dtype, &host_partials)
}

/// A typed scalar result of a reduction.
#[derive(Debug, Clone, Copy)]
enum Scalar {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    /// `SUM(Decimal128)` accumulator: the raw `i128` plus the output column's
    /// `(precision, scale)` so `scalar_to_array` can build a `Decimal128Array`
    /// with the correct dtype. The sum of a decimal column keeps the input
    /// scale (SQL widens precision, never scale) — the caller passes the
    /// declared output precision/scale through `out_field.dtype`.
    Decimal128(i128, u8, i8),
}

/// Neumaier-compensated summation (an improved Kahan variant) over an `f64`
/// iterator. Used by the float SUM/AVG host finalize so the host-side fold of
/// GPU partials matches the device's tree-order sum to low bits and tracks
/// DuckDB's compensated summation — a naive left-fold (`fold(0.0, |a, b| a +
/// b)`) loses the low bits and drifts vs both.
///
/// Accumulation is always in `f64` (callers upcast `f32` partials), which is
/// both more accurate and the typical engine behavior. Summing nothing yields
/// `0.0` — callers gate the empty/all-NULL → SQL NULL case upstream.
///
/// Non-finite terms are handled by falling back to a plain IEEE fold: a `+Inf`
/// (or `-Inf`/`NaN`) summand makes the compensation terms evaluate `inf - inf
/// == NaN`, which would wrongly turn `SUM` over a column containing `+Inf` into
/// `NaN`. The naive fold propagates `Inf`/`NaN` exactly as the SQL/IEEE
/// contract requires, so we return it whenever any non-finite term is seen.
#[inline]
fn neumaier_sum_f64(iter: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64; // running compensation for lost low-order bits
    let mut naive = 0.0_f64; // plain IEEE fold, used as the non-finite fallback
    let mut saw_nonfinite = false;
    for v in iter {
        if !v.is_finite() {
            saw_nonfinite = true;
        }
        naive += v;
        let t = sum + v;
        if sum.abs() >= v.abs() {
            // `sum` is larger: the low-order bits of `v` are lost.
            c += (sum - t) + v;
        } else {
            // `v` is larger: the low-order bits of `sum` are lost.
            c += (v - t) + sum;
        }
        sum = t;
    }
    // Compensated summation is only valid for finite terms (see doc above).
    if saw_nonfinite {
        naive
    } else {
        sum + c
    }
}

/// Per-`T` helpers for the GPU reduction path.
trait ReduceScalar: Sized + Copy {
    /// Combine a host-side slice using `op` and wrap as a `Scalar`.
    fn finalize(op: ReduceOp, dtype: DataType, host: &[Self]) -> BoltResult<Scalar>;
    /// Identity value (used when `n_rows == 0`).
    fn identity_scalar(op: ReduceOp, dtype: DataType) -> BoltResult<Scalar>;
}

impl ReduceScalar for i32 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> BoltResult<Scalar> {
        // INVARIANT (V-10): SUM(Int32) is ALWAYS widened to an i64 accumulator
        // before it reaches a host finalize — the SUM(Int32) dispatch site in
        // `reduce_column_from_batch` routes through `reduce_gpu_vec_widened::
        // <i32, i64>`, which finalizes via the `i64` `ReduceScalar` impl (whose
        // Sum arm uses `checked_add` and errors loudly on overflow). This i32
        // finalize is therefore only ever reached for MIN/MAX (and COUNT, a
        // synthesized sum-over-ones that cannot overflow i32). The `Sum` arm
        // below is unreachable for a real SUM; we assert that here so a future
        // dispatch change that accidentally routes a native i32 SUM through this
        // (silently wrapping) fold is caught in debug builds rather than
        // producing a wrong answer.
        debug_assert!(
            !matches!(op, ReduceOp::Sum),
            "SUM(Int32) must widen to i64 before host finalize (see i64 ReduceScalar); \
             the i32 finalize Sum arm is unreachable for a real SUM"
        );
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0i32, i32::wrapping_add),
            ReduceOp::Min => host.iter().copied().fold(i32::MAX, i32::min),
            ReduceOp::Max => host.iter().copied().fold(i32::MIN, i32::max),
        };
        Ok(Scalar::I32(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> BoltResult<Scalar> {
        Ok(Scalar::I32(match op {
            ReduceOp::Sum | ReduceOp::Count => 0,
            ReduceOp::Min => i32::MAX,
            ReduceOp::Max => i32::MIN,
        }))
    }
}

impl ReduceScalar for i64 {
    /// V-10: Integer SUM overflow contract (host-side ReduceScalar finalize).
    ///
    /// This i64 finalize backs the host SUM result for BOTH native SUM(Int64)
    /// and the widened SUM(Int32->i64) path (see `reduce_gpu_vec_widened::<i32,
    /// i64>` at the SUM(Int32) dispatch site, which finalizes through this impl
    /// at the widened i64 accumulator dtype). Previously this path used
    /// `i64::wrapping_add`, so an integer SUM that exceeded `i64::MAX` silently
    /// produced a wrapped (often negative) answer.
    ///
    /// The engine's stated invariant is "never silently wrong", and the
    /// SUM(Decimal128) path already errors loudly via `BoltError::Type` on
    /// accumulator overflow (see the `checked_add` fold around line 586). To
    /// make the integer SUM contract CONSISTENT and EXPLICIT with that path,
    /// integer SUM now ERRORS on overflow rather than wrapping: it accumulates
    /// with `checked_add` and returns a `BoltError::Type` whose message mirrors
    /// the Decimal128 "accumulator exceeds <range>" form. There is no silent
    /// wrap on the host path anymore.
    ///
    /// COUNT (synthesized as a SUM over ones) deliberately retains
    /// `wrapping_add`: a row count cannot realistically exceed `i64::MAX`, and
    /// turning COUNT into a fallible operation would be a gratuitous semantic
    /// change unrelated to V-10. MIN/MAX are non-arithmetic and never overflow.
    ///
    /// NOTE: the GPU group-by SUM accumulates with `atom.add.u64` in
    /// `groupby.rs` and has its own wrapping-on-overflow behavior; that atomic
    /// path is out of this file's scope and is tracked separately (V-10).
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> BoltResult<Scalar> {
        let acc = match op {
            // V-10: integer SUM errors loudly on i64 overflow (consistent with
            // the SUM(Decimal128) checked_add path) — no silent wrap.
            ReduceOp::Sum => {
                let mut sum: i64 = 0;
                for &v in host {
                    sum = match sum.checked_add(v) {
                        Some(s) => s,
                        None => {
                            return Err(BoltError::Type(
                                "SUM(integer) overflow: accumulator exceeds i64 range"
                                    .to_string(),
                            ));
                        }
                    };
                }
                sum
            }
            // V-10: COUNT (sum-over-ones) keeps wrapping_add — a row count
            // cannot realistically overflow i64 and COUNT stays infallible.
            ReduceOp::Count => host.iter().copied().fold(0i64, i64::wrapping_add),
            ReduceOp::Min => host.iter().copied().fold(i64::MAX, i64::min),
            ReduceOp::Max => host.iter().copied().fold(i64::MIN, i64::max),
        };
        Ok(Scalar::I64(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> BoltResult<Scalar> {
        Ok(Scalar::I64(match op {
            ReduceOp::Sum | ReduceOp::Count => 0,
            ReduceOp::Min => i64::MAX,
            ReduceOp::Max => i64::MIN,
        }))
    }
}

/// Order two floats under the DuckDB convention: NaN sorts as the *largest*
/// value (greater than +inf), and all NaN bit-patterns compare equal. This is
/// the same convention `src/exec/window.rs::float_total_cmp` adopts, so the
/// scalar and window MIN/MAX agree: MIN skips NaN unless the input is all-NaN
/// (NaN is the maximum, so any real value beats it), and MAX surfaces NaN
/// whenever one is present. We delegate to the type's `total_cmp` for the
/// non-NaN case (which already orders `-0 < +0`) and fold every NaN — including
/// `total_cmp`'s negative-NaN half — up to the top.
#[inline]
fn float_total_cmp<T: FloatTotalCmp>(a: T, b: T) -> std::cmp::Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater, // NaN is the largest
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => a.total_cmp_native(&b),
    }
}

/// Bridge so `float_total_cmp` works over both `f32` and `f64` without
/// duplicating the NaN-folding logic (each delegates to its own `total_cmp`).
trait FloatTotalCmp: Copy {
    fn is_nan(self) -> bool;
    fn total_cmp_native(&self, other: &Self) -> std::cmp::Ordering;
}
impl FloatTotalCmp for f32 {
    #[inline]
    fn is_nan(self) -> bool {
        f32::is_nan(self)
    }
    #[inline]
    fn total_cmp_native(&self, other: &Self) -> std::cmp::Ordering {
        f32::total_cmp(self, other)
    }
}
impl FloatTotalCmp for f64 {
    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
    #[inline]
    fn total_cmp_native(&self, other: &Self) -> std::cmp::Ordering {
        f64::total_cmp(self, other)
    }
}

impl ReduceScalar for f32 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> BoltResult<Scalar> {
        let acc = match op {
            // SUM/COUNT: Neumaier-compensated summation in f64 (upcasting the
            // f32 partials) so the host finalize matches the GPU's tree-order
            // sum to low bits and tracks DuckDB's compensated summation; a
            // naive f32 left-fold drifts. The accumulated f64 is narrowed back
            // to f32 to preserve the SUM(Float32) output dtype. NaN/Inf
            // propagate through Neumaier unchanged.
            ReduceOp::Sum | ReduceOp::Count => {
                neumaier_sum_f64(host.iter().copied().map(f64::from)) as f32
            }
            // MIN/MAX use the DuckDB NaN-as-largest convention (see
            // `float_total_cmp`) so the scalar path agrees with window.rs.
            // We seed from the first element (via `reduce`) rather than an
            // ±inf identity so an all-NaN input yields NaN — exactly what
            // window.rs does (it seeds its `extreme` from the first value).
            // The empty slice never reaches here (n_rows == 0 takes the
            // `identity_scalar` path), but we keep the ±inf identity as a
            // total fallback.
            ReduceOp::Min => host
                .iter()
                .copied()
                .reduce(|a, b| {
                    if float_total_cmp(b, a) == std::cmp::Ordering::Less {
                        b
                    } else {
                        a
                    }
                })
                .unwrap_or(f32::INFINITY),
            ReduceOp::Max => host
                .iter()
                .copied()
                .reduce(|a, b| {
                    if float_total_cmp(b, a) == std::cmp::Ordering::Greater {
                        b
                    } else {
                        a
                    }
                })
                .unwrap_or(f32::NEG_INFINITY),
        };
        Ok(Scalar::F32(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> BoltResult<Scalar> {
        Ok(Scalar::F32(match op {
            ReduceOp::Sum | ReduceOp::Count => 0.0,
            ReduceOp::Min => f32::INFINITY,
            ReduceOp::Max => f32::NEG_INFINITY,
        }))
    }
}

impl ReduceScalar for f64 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> BoltResult<Scalar> {
        let acc = match op {
            // SUM/COUNT: Neumaier-compensated summation so the host finalize
            // matches the GPU's tree-order sum to low bits and tracks DuckDB's
            // compensated summation; a naive left-fold drifts. NaN/Inf
            // propagate through Neumaier unchanged.
            ReduceOp::Sum | ReduceOp::Count => neumaier_sum_f64(host.iter().copied()),
            // MIN/MAX use the DuckDB NaN-as-largest convention (see
            // `float_total_cmp`) so the scalar path agrees with window.rs.
            // Seed from the first element (via `reduce`) so all-NaN yields
            // NaN, matching window.rs; ±inf is only the empty-slice fallback.
            ReduceOp::Min => host
                .iter()
                .copied()
                .reduce(|a, b| {
                    if float_total_cmp(b, a) == std::cmp::Ordering::Less {
                        b
                    } else {
                        a
                    }
                })
                .unwrap_or(f64::INFINITY),
            ReduceOp::Max => host
                .iter()
                .copied()
                .reduce(|a, b| {
                    if float_total_cmp(b, a) == std::cmp::Ordering::Greater {
                        b
                    } else {
                        a
                    }
                })
                .unwrap_or(f64::NEG_INFINITY),
        };
        Ok(Scalar::F64(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> BoltResult<Scalar> {
        Ok(Scalar::F64(match op {
            ReduceOp::Sum | ReduceOp::Count => 0.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
        }))
    }
}

/// Convert a `Scalar` into a single-element Arrow array of `out_dtype`.
/// Performs the small numeric cast (e.g. `i64 -> Float64` for AVG output).
fn scalar_to_array(scalar: Scalar, out_dtype: DataType) -> BoltResult<ArrayRef> {
    match (scalar, out_dtype) {
        (Scalar::I32(v), DataType::Int32) => Ok(Arc::new(Int32Array::from(vec![v])) as ArrayRef),
        (Scalar::I64(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(vec![v])) as ArrayRef),
        (Scalar::F32(v), DataType::Float32) => {
            Ok(Arc::new(Float32Array::from(vec![v])) as ArrayRef)
        }
        (Scalar::F64(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(vec![v])) as ArrayRef)
        }

        // Common cross-dtype output paths (e.g. AVG always produces Float64).
        (Scalar::I32(v), DataType::Int64) => {
            Ok(Arc::new(Int64Array::from(vec![v as i64])) as ArrayRef)
        }
        (Scalar::I32(v), DataType::Float32) => {
            Ok(Arc::new(Float32Array::from(vec![v as f32])) as ArrayRef)
        }
        (Scalar::I32(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(vec![v as f64])) as ArrayRef)
        }
        (Scalar::I64(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(vec![v as f64])) as ArrayRef)
        }
        (Scalar::F32(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(vec![v as f64])) as ArrayRef)
        }

        // Decimal128 SUM: pack the raw i128 into a one-element Decimal128Array
        // carrying the declared output precision/scale. We tag the array with
        // the OUTPUT field's (p, s) (via `with_precision_and_scale`) so the
        // result column round-trips to Arrow's `Decimal128(p, s)` type. The
        // scale of the accumulated value already matches the input scale (a
        // plain integer sum doesn't shift the decimal point); SQL's SUM widens
        // precision only, so the carried-through `s` is correct.
        (Scalar::Decimal128(v, _ps, _ss), DataType::Decimal128(p, s)) => {
            let arr = Decimal128Array::from(vec![v])
                .with_precision_and_scale(p, s)
                .map_err(|e| {
                    BoltError::Type(format!(
                        "aggregate: SUM(Decimal128) output precision/scale {:?} invalid: {e}",
                        (p, s)
                    ))
                })?;
            Ok(Arc::new(arr) as ArrayRef)
        }

        (s, dt) => Err(BoltError::Type(format!(
            "aggregate: cannot pack scalar {:?} into output dtype {:?}",
            s, dt
        ))),
    }
}

/// Build a single-row, all-NULL Arrow array of `out_dtype`. Used by the
/// scalar SUM/MIN/MAX path when the input is empty or all-NULL: SQL says the
/// result is NULL, not the reduction identity. The output field must be
/// nullable (the caller gates on `out_field.nullable`) — a NULL in a
/// non-nullable column would be rejected by `RecordBatch::try_new`. Mirrors
/// the `None`-seeded packing in `minmax_decimal128_from_batch` and the
/// nullable branch of `avg_result_array`.
fn null_scalar_array(out_dtype: DataType) -> BoltResult<ArrayRef> {
    match out_dtype {
        DataType::Int32 => Ok(Arc::new(Int32Array::from(vec![Option::<i32>::None])) as ArrayRef),
        DataType::Int64 => Ok(Arc::new(Int64Array::from(vec![Option::<i64>::None])) as ArrayRef),
        DataType::Float32 => {
            Ok(Arc::new(Float32Array::from(vec![Option::<f32>::None])) as ArrayRef)
        }
        DataType::Float64 => {
            Ok(Arc::new(Float64Array::from(vec![Option::<f64>::None])) as ArrayRef)
        }
        // SUM(Decimal128) widens to Decimal128(38, s); MIN/MAX preserve the
        // input (p, s). A `None` validity bit packs SQL NULL; the (p, s) tag
        // still has to satisfy Arrow so we carry the declared output scale.
        DataType::Decimal128(p, s) => {
            let arr = Decimal128Array::from(vec![Option::<i128>::None])
                .with_precision_and_scale(p, s)
                .map_err(|e| {
                    BoltError::Type(format!(
                        "aggregate NULL result: precision/scale {:?} rejected by Arrow: {e}",
                        (p, s)
                    ))
                })?;
            Ok(Arc::new(arr) as ArrayRef)
        }
        other => Err(BoltError::Type(format!(
            "aggregate: cannot build NULL scalar for output dtype {:?}",
            other
        ))),
    }
}

/// Build a `Type` error for a failed Arrow downcast on column `name`.
fn downcast_err(name: &str, expected: &str) -> BoltError {
    BoltError::Type(format!(
        "aggregate input column '{}' could not be downcast to {}",
        name, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType`. Mirror of `engine.rs`'s
/// private helper, copied here so we don't reach across module privacy.
fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    crate::exec::schema_convert::arrow_dtype_to_plan_basic(d, "")
}

/// Build an Arrow `Schema` from our plan `Schema` for the output RecordBatch.
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal(s, "this aggregate output path")
}

// ---------------------------------------------------------------------------
// Host-only tests for the NULL-handling helpers. The full
// `execute_aggregate` path needs the GPU and is exercised by the integration
// suite; what we pin here is exactly the NULL bookkeeping that landed with
// the H1 fix: COUNT(col) excludes nulls, AVG denominator is the non-null
// count, and the pre-GPU filter keeps the raw values buffer's garbage bytes
// out of the reduction.
// ---------------------------------------------------------------------------
// v0.7: these arrow/plan aliases are used only by the #[cfg(test)] modules
// below; the non-test schema conversion now lives in exec::schema_convert.
// cfg(test)-gated so normal builds don't see an unused import.
#[cfg(test)]
use arrow_schema::{Field as ArrowField};

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: minimal single-column batch.
    fn batch_one(name: &str, arr: ArrayRef) -> RecordBatch {
        let dt = arr.data_type().clone();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(name, dt, true)]));
        RecordBatch::try_new(schema, vec![arr]).expect("batch")
    }

    /// `filter_primitive_to_vec` drops NULL positions and preserves order for
    /// the surviving values. Garbage at NULL positions in the underlying
    /// values buffer (which Arrow doesn't zero) must not appear in the output.
    #[test]
    fn filter_primitive_drops_null_positions_i32() {
        // The underlying values buffer for a NULL position is arbitrary; here
        // it's `i32::MAX`, a value that would visibly corrupt MIN/SUM if it
        // leaked through.
        let arr = Int32Array::from(vec![
            Some(1i32),
            None,
            Some(2),
            None,
            Some(3),
            None,
        ]);
        let host = filter_primitive_to_vec::<arrow_array::types::Int32Type>(&arr);
        assert_eq!(host, vec![1, 2, 3]);
    }

    #[test]
    fn filter_primitive_drops_null_positions_f64() {
        let arr = Float64Array::from(vec![Some(1.5f64), None, Some(2.5), Some(-3.0)]);
        let host = filter_primitive_to_vec::<arrow_array::types::Float64Type>(&arr);
        assert_eq!(host, vec![1.5, 2.5, -3.0]);
    }

    #[test]
    fn filter_primitive_no_nulls_returns_full_vec() {
        let arr = Int64Array::from(vec![10i64, 20, 30]);
        let host = filter_primitive_to_vec::<arrow_array::types::Int64Type>(&arr);
        assert_eq!(host, vec![10, 20, 30]);
    }

    /// F4: AVG over zero matching rows yields SQL NULL when the output field
    /// is nullable.
    #[test]
    fn avg_empty_input_is_null_when_nullable() {
        let arr = avg_result_array(0.0, 0, /* out_nullable = */ true);
        let fa = arr
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64Array");
        assert_eq!(fa.len(), 1);
        assert!(fa.is_null(0), "empty AVG over nullable field must be NULL");
    }

    /// F4: when the output field is (legacy) non-nullable, AVG over zero rows
    /// falls back to 0.0 so `RecordBatch::try_new` does not reject a null in a
    /// non-nullable column.
    #[test]
    fn avg_empty_input_is_zero_when_non_nullable() {
        let arr = avg_result_array(0.0, 0, /* out_nullable = */ false);
        let fa = arr
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Float64Array");
        assert_eq!(fa.len(), 1);
        assert!(!fa.is_null(0));
        assert_eq!(fa.value(0), 0.0);
    }

    /// F4: a non-empty AVG still divides sum by count regardless of
    /// nullability.
    #[test]
    fn avg_non_empty_divides() {
        for nullable in [true, false] {
            let arr = avg_result_array(10.0, 4, nullable);
            let fa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .expect("Float64Array");
            assert!(!fa.is_null(0));
            assert_eq!(fa.value(0), 2.5);
        }
    }

    /// V-10: integer SUM that overflows `i64::MAX` must error loudly with a
    /// `BoltError::Type` (consistent with the SUM(Decimal128) checked path),
    /// never silently wrap to a negative value.
    #[test]
    fn i64_sum_finalize_overflow_errors_not_wraps() {
        // Two halves of i64::MAX plus a final +2 tips the accumulator past
        // i64::MAX. A wrapping_add fold would have produced a negative result.
        let host = vec![i64::MAX, 1i64, 1i64];
        let res = <i64 as ReduceScalar>::finalize(ReduceOp::Sum, DataType::Int64, &host);
        match res {
            Err(BoltError::Type(msg)) => {
                assert!(
                    msg.contains("SUM(integer) overflow"),
                    "unexpected error message: {msg}"
                );
            }
            other => panic!("expected BoltError::Type on SUM overflow, got {other:?}"),
        }
    }

    /// V-10: a normal in-range integer SUM still finalizes to the correct
    /// (non-wrapped) total.
    #[test]
    fn i64_sum_finalize_in_range_is_correct() {
        let host = vec![10i64, 20, 30, -5];
        let res = <i64 as ReduceScalar>::finalize(ReduceOp::Sum, DataType::Int64, &host)
            .expect("in-range SUM must succeed");
        match res {
            Scalar::I64(v) => assert_eq!(v, 55),
            other => panic!("expected Scalar::I64, got {other:?}"),
        }
    }

    /// TASK 1: scalar float MIN/MAX follow the DuckDB NaN-as-largest convention
    /// (the same one `src/exec/window.rs` uses), so the scalar and window paths
    /// agree. With a NaN present, MIN skips it (returns the real minimum) and
    /// MAX returns NaN. Covers both f32 and f64 lanes.
    #[test]
    fn f64_min_max_finalize_nan_convention() {
        let host = vec![f64::NAN, 2.0, -1.0];
        let min = <f64 as ReduceScalar>::finalize(ReduceOp::Min, DataType::Float64, &host)
            .expect("min ok");
        match min {
            // MIN skips NaN -> the real minimum.
            Scalar::F64(v) => assert_eq!(v, -1.0),
            other => panic!("expected Scalar::F64, got {other:?}"),
        }
        let max = <f64 as ReduceScalar>::finalize(ReduceOp::Max, DataType::Float64, &host)
            .expect("max ok");
        match max {
            // MAX surfaces NaN (NaN is the largest under the convention).
            Scalar::F64(v) => assert!(v.is_nan(), "expected NaN MAX, got {v}"),
            other => panic!("expected Scalar::F64, got {other:?}"),
        }
    }

    #[test]
    fn f32_min_max_finalize_nan_convention() {
        let host = vec![f32::NAN, 2.0f32, -1.0];
        let min = <f32 as ReduceScalar>::finalize(ReduceOp::Min, DataType::Float32, &host)
            .expect("min ok");
        match min {
            Scalar::F32(v) => assert_eq!(v, -1.0),
            other => panic!("expected Scalar::F32, got {other:?}"),
        }
        let max = <f32 as ReduceScalar>::finalize(ReduceOp::Max, DataType::Float32, &host)
            .expect("max ok");
        match max {
            Scalar::F32(v) => assert!(v.is_nan(), "expected NaN MAX, got {v}"),
            other => panic!("expected Scalar::F32, got {other:?}"),
        }
    }

    /// TASK 1: an all-NaN float MIN returns NaN (no real value to prefer), and
    /// MAX also returns NaN — matching window.rs's all-NaN behaviour.
    #[test]
    fn f64_min_max_finalize_all_nan_is_nan() {
        let host = vec![f64::NAN, f64::NAN];
        let min = <f64 as ReduceScalar>::finalize(ReduceOp::Min, DataType::Float64, &host)
            .expect("min ok");
        match min {
            Scalar::F64(v) => assert!(v.is_nan(), "expected NaN MIN, got {v}"),
            other => panic!("expected Scalar::F64, got {other:?}"),
        }
        let max = <f64 as ReduceScalar>::finalize(ReduceOp::Max, DataType::Float64, &host)
            .expect("max ok");
        match max {
            Scalar::F64(v) => assert!(v.is_nan(), "expected NaN MAX, got {v}"),
            other => panic!("expected Scalar::F64, got {other:?}"),
        }
    }

    /// TASK 1: with no NaN present, float MIN/MAX behave exactly as before
    /// (ordinary numeric extremes), so the convention change is a no-op for the
    /// common case.
    #[test]
    fn f64_min_max_finalize_no_nan_unchanged() {
        let host = vec![3.0f64, -2.0, 7.5, 0.0];
        let min = <f64 as ReduceScalar>::finalize(ReduceOp::Min, DataType::Float64, &host)
            .expect("min ok");
        let max = <f64 as ReduceScalar>::finalize(ReduceOp::Max, DataType::Float64, &host)
            .expect("max ok");
        assert!(matches!(min, Scalar::F64(v) if v == -2.0));
        assert!(matches!(max, Scalar::F64(v) if v == 7.5));
    }

    /// `non_null_count_for_input` returns the count of non-null cells. This
    /// drives both COUNT(col) and the AVG denominator.
    #[test]
    fn non_null_count_for_input_counts_only_valid_rows() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(1),
            None,
            Some(3),
            None,
            Some(5),
        ]));
        let batch = batch_one("v", arr);
        let col_io = ColumnIO {
            name: "v".to_string(),
            dtype: DataType::Int32,
        };
        let c = non_null_count_for_input(&col_io, &batch).expect("count ok");
        assert_eq!(c, 3);
    }

    #[test]
    fn non_null_count_for_input_all_nulls_is_zero() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![
            Option::<i64>::None,
            None,
            None,
        ]));
        let batch = batch_one("v", arr);
        let col_io = ColumnIO {
            name: "v".to_string(),
            dtype: DataType::Int64,
        };
        assert_eq!(non_null_count_for_input(&col_io, &batch).unwrap(), 0);
    }

    /// `decimal_sum_host` folds an `i128` slice with a CHECKED add: it returns
    /// the exact sum for non-overflowing inputs (including across the 2^64
    /// boundary the GPU carry-chain kernel exercises) and errors on i128
    /// overflow rather than wrapping.
    #[test]
    fn decimal_sum_host_folds_i128() {
        assert_eq!(decimal_sum_host(&[]).unwrap(), 0);
        assert_eq!(decimal_sum_host(&[1, 2, 3]).unwrap(), 6);
        // Values that individually fit in 64 bits but whose sum crosses 2^64,
        // exercising the carry path the kernel emits (`add.cc`/`addc`).
        let big = u64::MAX as i128; // 2^64 - 1
        assert_eq!(decimal_sum_host(&[big, big, big]).unwrap(), big * 3);
        // Negative + positive decimal raw values.
        assert_eq!(decimal_sum_host(&[-5, 10, -2]).unwrap(), 3);
    }

    /// TASK 2: `SUM(Decimal128)` must ERROR loudly on i128 overflow, never
    /// wrap — matching the `SUM(integer) overflow` contract and the
    /// never-silently-wrong invariant.
    #[test]
    fn decimal_sum_host_overflow_errors() {
        let err = decimal_sum_host(&[i128::MAX, 1]).unwrap_err();
        match err {
            BoltError::Type(msg) => {
                assert!(
                    msg.contains("SUM(Decimal128) precision overflow"),
                    "unexpected message: {msg}"
                );
            }
            other => panic!("expected BoltError::Type on Decimal128 SUM overflow, got {other:?}"),
        }
    }

    /// TASK 2: a normal (non-overflowing) Decimal128 SUM via the live host fold
    /// still returns the exact sum.
    #[test]
    fn decimal_sum_host_normal_ok() {
        assert_eq!(decimal_sum_host(&[100, 250, 50]).unwrap(), 400);
    }

    /// `decimal_sum_from_batch` strips NULL rows and returns the surviving
    /// sum tagged with the column's precision/scale. Runs the GPU path when a
    /// device is present and the host fold otherwise — either way the
    /// `Scalar::Decimal128` payload must carry the dense sum and (p, s).
    /// Gated on a real GPU because `decimal_sum_gpu` launches a kernel.
    #[test]
    #[ignore = "gpu:tier1"]
    fn decimal_sum_from_batch_strips_nulls() {
        let arr: ArrayRef = Arc::new(
            Decimal128Array::from(vec![Some(100i128), None, Some(250), None, Some(50)])
                .with_precision_and_scale(20, 2)
                .unwrap(),
        );
        let batch = batch_one("d", arr);
        let col_io = ColumnIO {
            name: "d".to_string(),
            dtype: DataType::Decimal128(20, 2),
        };
        let scalar = decimal_sum_from_batch(&col_io, &batch, 5).expect("decimal sum ok");
        match scalar {
            Scalar::Decimal128(v, p, s) => {
                assert_eq!(v, 400);
                assert_eq!((p, s), (20, 2));
            }
            other => panic!("expected Scalar::Decimal128, got {other:?}"),
        }
    }

    /// `scalar_to_array` packs a `Scalar::Decimal128` into a one-row
    /// `Decimal128Array` carrying the declared output precision/scale.
    #[test]
    fn scalar_to_array_packs_decimal128() {
        let arr = scalar_to_array(
            Scalar::Decimal128(12345, 20, 2),
            DataType::Decimal128(20, 2),
        )
        .expect("pack ok");
        let da = arr
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("Decimal128Array");
        assert_eq!(da.len(), 1);
        assert_eq!(da.value(0), 12345);
        assert_eq!(da.precision(), 20);
        assert_eq!(da.scale(), 2);
    }

    #[test]
    fn non_null_count_for_input_no_nulls_is_full() {
        let arr: ArrayRef = Arc::new(Float32Array::from(vec![1.0f32, 2.0, 3.0, 4.0]));
        let batch = batch_one("v", arr);
        let col_io = ColumnIO {
            name: "v".to_string(),
            dtype: DataType::Float32,
        };
        assert_eq!(non_null_count_for_input(&col_io, &batch).unwrap(), 4);
    }

    // -------- MIN/MAX(Decimal128) host-side fold (no GPU) --------
    //
    // These exercise `minmax_decimal128_from_batch` directly: NULL-skipping,
    // the all-NULL/empty => NULL identity, negatives, single value, and that
    // the output preserves the input (precision, scale).

    /// Build a single-column `Decimal128(p, s)` batch from `Option<i128>`
    /// values, mirroring the construction used elsewhere for the SUM path.
    fn dec128_batch(name: &str, p: u8, s: i8, values: Vec<Option<i128>>) -> RecordBatch {
        let arr = Decimal128Array::from(values)
            .with_precision_and_scale(p, s)
            .expect("valid decimal128 array");
        batch_one(name, Arc::new(arr) as ArrayRef)
    }

    fn dec128_col(name: &str, p: u8, s: i8) -> ColumnIO {
        ColumnIO {
            name: name.to_string(),
            dtype: DataType::Decimal128(p, s),
        }
    }

    /// Pull the (validity, value) of the single result row out of an
    /// aggregate output array for assertions.
    fn single_dec128(out: &ArrayRef) -> (bool, i128, u8, i8) {
        let da = out
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("decimal128 result");
        assert_eq!(da.len(), 1, "scalar aggregate emits exactly one row");
        let (p, s) = match da.data_type() {
            ArrowDataType::Decimal128(p, s) => (*p, *s),
            other => panic!("expected Decimal128 result dtype, got {other:?}"),
        };
        if da.is_null(0) {
            (false, 0, p, s)
        } else {
            (true, da.value(0), p, s)
        }
    }

    #[test]
    fn minmax_decimal128_with_nulls() {
        // Underlying NULL positions must be skipped, not folded in.
        let batch = dec128_batch(
            "price",
            10,
            2,
            vec![Some(500i128), None, Some(125), None, Some(999)],
        );
        let col = dec128_col("price", 10, 2);
        let out_field = Field::new("m", DataType::Decimal128(10, 2), true);

        let min_out =
            minmax_decimal128_from_batch(ReduceOp::Min, &col, &batch, 10, 2, &out_field).unwrap();
        let (valid, v, p, s) = single_dec128(&min_out);
        assert!(valid);
        assert_eq!(v, 125);
        // Precision/scale preserved on the MIN output.
        assert_eq!((p, s), (10, 2));

        let max_out =
            minmax_decimal128_from_batch(ReduceOp::Max, &col, &batch, 10, 2, &out_field).unwrap();
        let (valid, v, p, s) = single_dec128(&max_out);
        assert!(valid);
        assert_eq!(v, 999);
        assert_eq!((p, s), (10, 2));
    }

    #[test]
    fn minmax_decimal128_all_null_is_null() {
        let batch = dec128_batch("price", 10, 2, vec![None, None, None]);
        let col = dec128_col("price", 10, 2);
        let out_field = Field::new("m", DataType::Decimal128(10, 2), true);

        let min_out =
            minmax_decimal128_from_batch(ReduceOp::Min, &col, &batch, 10, 2, &out_field).unwrap();
        let (valid, _, _, _) = single_dec128(&min_out);
        assert!(!valid, "MIN over all-NULL input is SQL NULL");

        let max_out =
            minmax_decimal128_from_batch(ReduceOp::Max, &col, &batch, 10, 2, &out_field).unwrap();
        let (valid, _, _, _) = single_dec128(&max_out);
        assert!(!valid, "MAX over all-NULL input is SQL NULL");
    }

    #[test]
    fn minmax_decimal128_empty_is_null() {
        let batch = dec128_batch("price", 10, 2, Vec::<Option<i128>>::new());
        let col = dec128_col("price", 10, 2);
        let out_field = Field::new("m", DataType::Decimal128(10, 2), true);

        let min_out =
            minmax_decimal128_from_batch(ReduceOp::Min, &col, &batch, 10, 2, &out_field).unwrap();
        assert!(!single_dec128(&min_out).0, "MIN over empty input is NULL");

        let max_out =
            minmax_decimal128_from_batch(ReduceOp::Max, &col, &batch, 10, 2, &out_field).unwrap();
        assert!(!single_dec128(&max_out).0, "MAX over empty input is NULL");
    }

    #[test]
    fn minmax_decimal128_single_value() {
        let batch = dec128_batch("price", 12, 4, vec![Some(42_0000i128)]);
        let col = dec128_col("price", 12, 4);
        let out_field = Field::new("m", DataType::Decimal128(12, 4), true);

        let min_out =
            minmax_decimal128_from_batch(ReduceOp::Min, &col, &batch, 12, 4, &out_field).unwrap();
        let (valid, v, p, s) = single_dec128(&min_out);
        assert!(valid);
        assert_eq!(v, 42_0000);
        assert_eq!((p, s), (12, 4));

        let max_out =
            minmax_decimal128_from_batch(ReduceOp::Max, &col, &batch, 12, 4, &out_field).unwrap();
        assert_eq!(single_dec128(&max_out).1, 42_0000);
    }

    #[test]
    fn minmax_decimal128_negative_values() {
        // Raw i128 ordering must equal decimal ordering across the sign
        // boundary: MIN picks the most-negative, MAX the most-positive.
        let batch = dec128_batch(
            "bal",
            10,
            2,
            vec![Some(-300i128), Some(50), Some(-1000), Some(0)],
        );
        let col = dec128_col("bal", 10, 2);
        let out_field = Field::new("m", DataType::Decimal128(10, 2), true);

        let min_out =
            minmax_decimal128_from_batch(ReduceOp::Min, &col, &batch, 10, 2, &out_field).unwrap();
        assert_eq!(single_dec128(&min_out).1, -1000);

        let max_out =
            minmax_decimal128_from_batch(ReduceOp::Max, &col, &batch, 10, 2, &out_field).unwrap();
        assert_eq!(single_dec128(&max_out).1, 50);
    }

    #[test]
    fn minmax_decimal128_rejects_non_decimal_output_field() {
        let batch = dec128_batch("price", 10, 2, vec![Some(1i128)]);
        let col = dec128_col("price", 10, 2);
        // Wrong declared output dtype must surface a loud Type error.
        let bad_field = Field::new("m", DataType::Int64, true);
        let err =
            minmax_decimal128_from_batch(ReduceOp::Min, &col, &batch, 10, 2, &bad_field).unwrap_err();
        assert!(matches!(err, BoltError::Type(_)));
    }

    /// `packed_validity_for` produces the Arrow-LE packed-bit bitmap the
    /// `_with_validity` reduction kernel reads: bit `i` of byte `i/8` is 1
    /// iff row `i` is present (non-NULL).
    #[test]
    fn packed_validity_for_matches_arrow_le_bits() {
        // Rows: present, null, present, null, present, null, present, null
        // -> bits 0,2,4,6 set, 1,3,5,7 clear -> 0b0101_0101 = 0x55.
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(10),
            None,
            Some(20),
            None,
            Some(30),
            None,
            Some(40),
            None,
        ]));
        let packed = packed_validity_for(arr.as_ref());
        assert_eq!(packed, vec![0x55u8]);

        // A NULL-free column packs to all-ones in every full byte.
        let arr2: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3, 4, 5, 6, 7, 8]));
        assert_eq!(packed_validity_for(arr2.as_ref()), vec![0xFFu8]);
    }

    /// COUNT(col) on the primitive scalar path counts ONLY non-NULL rows —
    /// the same SQL semantic the Bool/Utf8 path honours. Exercised host-only
    /// through `build_one_aggregate` (the COUNT arm computes the result from
    /// the Arrow null bitmap without a GPU launch).
    #[test]
    fn count_col_primitive_excludes_nulls() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![
            Some(1),
            None,
            Some(3),
            None,
            Some(5),
        ]));
        let batch = batch_one("v", arr);
        let inputs = vec![ColumnIO {
            name: "v".to_string(),
            dtype: DataType::Int32,
        }];
        let agg = AggregateExpr::Count(Expr::Column("v".to_string()));
        let out_field = Field {
            name: "cnt".to_string(),
            dtype: DataType::Int64,
            nullable: false,
        };
        let arr_out =
            build_one_aggregate(&agg, &out_field, &inputs, &batch, batch.num_rows())
                .expect("count ok");
        let col = arr_out
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("COUNT output is Int64");
        assert_eq!(col.len(), 1);
        // 3 non-NULL rows out of 5.
        assert_eq!(col.value(0), 3);
    }

    /// COUNT(*) (an expression that is not a bare column reference) counts
    /// EVERY row, NULLs included — the complement of `count_col_primitive_*`.
    #[test]
    fn count_star_counts_all_rows_including_nulls() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![Some(1), None, Some(3)]));
        let batch = batch_one("v", arr);
        let inputs = vec![ColumnIO {
            name: "v".to_string(),
            dtype: DataType::Int32,
        }];
        // COUNT(*) lowers to a literal-ish expression that doesn't resolve to
        // a column, so the COUNT arm returns the full row count.
        let agg = AggregateExpr::Count(Expr::Literal(
            crate::plan::logical_plan::Literal::Int64(1),
        ));
        let out_field = Field {
            name: "cnt".to_string(),
            dtype: DataType::Int64,
            nullable: false,
        };
        let arr_out =
            build_one_aggregate(&agg, &out_field, &inputs, &batch, batch.num_rows())
                .expect("count ok");
        let col = arr_out
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("COUNT output is Int64");
        assert_eq!(col.value(0), 3);
    }

    // -------- Stage-3 round-trip tests (require GPU) --------
    //
    // These tests confirm that the async memcpy + pinned D2H plumbing
    // produces bit-identical results to the sync path it replaced.

    fn one_col_batch_i64(values: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "v",
            arrow_schema::DataType::Int64,
            false,
        )]));
        let col: ArrayRef = Arc::new(Int64Array::from(values));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    #[test]
    #[ignore = "gpu:tier1"]
    fn async_sum_int64_matches_host_sum() {
        // Round-trip: build a small table, run SUM through the engine
        // (which uses the Stage-3 async-memcpy reduction path here), and
        // compare against an iterator sum.
        use crate::Engine;

        let mut engine = Engine::new().expect("ctx");
        let xs: Vec<i64> = (0..10_000i64).collect();
        let expected: i64 = xs.iter().sum();
        let batch = one_col_batch_i64(xs);
        engine.register_table("t", batch).unwrap();
        let h = engine.sql("SELECT SUM(v) FROM t").expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 1);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(col.value(0), expected);
    }

    /// Helper: build a single-column Float64 batch.
    fn one_col_batch_f64(values: Vec<f64>) -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "v",
            arrow_schema::DataType::Float64,
            false,
        )]));
        let col: ArrayRef = Arc::new(Float64Array::from(values));
        RecordBatch::try_new(schema, vec![col]).unwrap()
    }

    /// Fused-AVG happy path: `[1.0, 2.0, 3.0]` -> `2.0`. Goes through the
    /// engine (which dispatches into `fused_avg_from_batch` -> the single
    /// `bolt_avg_reduce` PTX kernel) and checks that the host-side finalize
    /// of the per-block `(sum, count)` partials matches the simple average.
    #[test]
    #[ignore = "gpu:tier1"]
    fn fused_avg_matches_simple_average() {
        use crate::Engine;

        let mut engine = Engine::new().expect("ctx");
        let batch = one_col_batch_f64(vec![1.0, 2.0, 3.0]);
        engine.register_table("t", batch).unwrap();
        let h = engine.sql("SELECT AVG(v) FROM t").expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 1);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("AVG output is Float64");
        // `[1,2,3]` -> mean = 2.0 exactly; the fused kernel sums in f64 so
        // there is no rounding error to chase here.
        assert!((col.value(0) - 2.0).abs() < 1e-12, "got {}", col.value(0));
    }

    /// Empty-input AVG: with zero rows the kernel skips the launch entirely
    /// and we fall back to the documented `0.0` semantics (see the
    /// `TODO(null)` in `build_one_aggregate::Avg`). This test pins that
    /// behaviour so the public contract doesn't drift silently.
    #[test]
    #[ignore = "gpu:tier1"]
    fn fused_avg_empty_input_returns_zero() {
        use crate::Engine;

        let mut engine = Engine::new().expect("ctx");
        let batch = one_col_batch_f64(vec![]);
        engine.register_table("t", batch).unwrap();
        let h = engine.sql("SELECT AVG(v) FROM t").expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 1);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("AVG output is Float64");
        // TODO(null): SQL standard says NULL; we currently return 0.0 to
        // keep the AVG output field non-nullable.
        assert_eq!(col.value(0), 0.0);
    }

    /// Host-only PTX-shape regression test: verify the fused kernel emits a
    /// single entry point with three pointer params and one u32, plus stores
    /// to both `block_sums` (f64) and `block_counts` (b32). Catches the
    /// trivial case where someone forgets to wire the second output buffer.
    #[test]
    fn fused_avg_kernel_emits_both_partials() {
        use crate::jit::agg_kernels::{compile_avg_reduction_kernel, AVG_KERNEL_ENTRY};

        let ptx = compile_avg_reduction_kernel(DataType::Float64)
            .expect("AVG PTX should compile");
        // One entry, four params (input, sums, counts, n_rows).
        assert!(
            ptx.contains(&format!(".visible .entry {}(", AVG_KERNEL_ENTRY)),
            "PTX missing entry point"
        );
        assert!(ptx.contains("param_0"), "missing input param");
        assert!(ptx.contains("param_1"), "missing block_sums param");
        assert!(ptx.contains("param_2"), "missing block_counts param");
        assert!(ptx.contains("param_3"), "missing n_rows param");
        // Both partial stores must be present.
        assert!(
            ptx.contains("st.global.f64"),
            "PTX must store block sums as f64"
        );
        assert!(
            ptx.contains("st.global.b32"),
            "PTX must store block counts as b32"
        );
    }

    #[test]
    #[ignore = "gpu:tier1"]
    fn pinned_d2h_matches_sync_d2h() {
        // Allocate identical GPU buffers via `GpuVec::from_slice`, then
        // pull one through the legacy sync path (`to_vec`) and one
        // through the Stage-3 pinned path (`to_pinned_async` + sync +
        // copy). The byte-for-byte equality check guards against a
        // future regression where the pinned path drops or reorders
        // elements.
        use crate::cuda::GpuVec;
        use crate::exec::launch::CudaStream;

        let stream = CudaStream::null_or_default();
        let xs: Vec<i32> = (0..1024i32).map(|i| i * 3 - 7).collect();
        let v = GpuVec::<i32>::from_slice(&xs).expect("upload");

        let via_sync = v.to_vec().expect("sync d2h");
        let pinned = v.to_pinned_async(stream.raw()).expect("async d2h");
        stream.synchronize().expect("sync");
        let via_pinned: Vec<i32> = pinned.as_slice().to_vec();

        assert_eq!(via_sync, via_pinned);
    }

    /// Neumaier compensated summation recovers an ill-conditioned sequence whose
    /// low-order terms a naive left-fold drops. For `[1e16, 1.0, -1e16, 1.0]`
    /// the naive fold loses the FIRST `+1.0` (below 1e16's ULP) and keeps only
    /// the last, yielding `1.0`; Neumaier recovers both and yields `2.0`.
    #[test]
    fn neumaier_sum_is_accurate_on_ill_conditioned_input() {
        let data = [1e16_f64, 1.0, -1e16, 1.0];
        // Sanity: confirm the naive left-fold actually drifts here (drops the
        // first +1.0 -> 1.0), so the Neumaier assertion below is meaningful.
        let naive = data.iter().copied().fold(0.0_f64, |a, b| a + b);
        assert_eq!(naive, 1.0, "precondition: naive fold drops the first +1.0");
        let neumaier = neumaier_sum_f64(data.iter().copied());
        assert_eq!(neumaier, 2.0, "Neumaier must recover the exact sum");
    }

    /// NaN and Inf propagate through Neumaier summation (IEEE arithmetic):
    /// any NaN summand makes the whole sum NaN, and a lone +Inf among finite
    /// values yields +Inf.
    #[test]
    fn neumaier_sum_propagates_nan_and_inf() {
        // NaN poisons the running sum.
        let with_nan = [1.0_f64, f64::NAN, 2.0];
        assert!(
            neumaier_sum_f64(with_nan.iter().copied()).is_nan(),
            "NaN summand must produce NaN"
        );
        // +Inf with finite values stays +Inf.
        let with_inf = [1.0_f64, f64::INFINITY, 2.0];
        assert_eq!(
            neumaier_sum_f64(with_inf.iter().copied()),
            f64::INFINITY,
            "+Inf summand must produce +Inf"
        );
        // +Inf and -Inf together produce NaN (Inf - Inf), as IEEE dictates.
        let mixed_inf = [f64::INFINITY, f64::NEG_INFINITY];
        assert!(
            neumaier_sum_f64(mixed_inf.iter().copied()).is_nan(),
            "+Inf and -Inf together must produce NaN"
        );
    }

    /// An empty iterator sums to the identity `0.0` (the empty/all-NULL → SQL
    /// NULL case is gated upstream of the finalize, so this must not panic).
    #[test]
    fn neumaier_sum_empty_is_zero() {
        let empty: [f64; 0] = [];
        assert_eq!(neumaier_sum_f64(empty.iter().copied()), 0.0);
    }
}
