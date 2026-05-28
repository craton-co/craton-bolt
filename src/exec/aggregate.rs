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

use arrow_array::{Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};
use bytemuck::Pod;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{grid_x_for, CudaStream};
use crate::exec::module_cache;
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::{
    compile_avg_reduction_kernel, compile_reduction_kernel, ReduceOp, AVG_KERNEL_ENTRY,
    BLOCK_SIZE, REDUCTION_KERNEL_ENTRY,
};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Schema};

// `CudaModule` import dropped: every load site now routes through
// `exec::module_cache::get_or_build_module`, which returns the cached module
// directly.
use crate::plan::physical_plan::{AggregateSpec, ColumnIO, PhysicalPlan};

// v0.6 pilot helper `upload_primitive_values_async` was promoted to
// `crate::exec::gpu_upload` in v0.7 so the filter / GROUP BY / join
// executors can share the same `(slice, &stream) -> GpuVec<T>` shape with
// the identical `--features cuda-stub` graceful fallback. The aggregate
// path uses it through the canonical name so the migration is a
// drop-in import swap — no semantic change.
use crate::exec::gpu_upload::upload_primitive_values_async;

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
            // TODO(null): SQL standard returns NULL when COUNT == 0; we
            // currently return 0.0 to preserve the public AVG return-type
            // contract (non-nullable Float64). Surfacing NULL would require
            // making the AVG output schema field nullable across the planner
            // and downstream consumers — out of scope for this fusion.
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;
            let (sum_f64, count_u64) =
                fused_avg_from_batch(col_io, table_batch, n_rows)?;
            let avg = if count_u64 == 0 {
                0.0
            } else {
                sum_f64 / count_u64 as f64
            };
            scalar_to_array(Scalar::F64(avg), out_field.dtype)
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
                let host: Vec<i32> = filter_primitive_to_vec(pa);
                let len = host.len();
                if matches!(op, ReduceOp::Sum) {
                    // Widened SUM path: replicate the upload-async shape that
                    // `reduce_host_slice` would have used, since that helper is
                    // monomorphic over a single accumulator type.
                    let dev = upload_primitive_values_async::<i32>(&host, &stream)?;
                    reduce_gpu_vec_widened::<i32, i64>(
                        op, col_io.dtype, &dev, len, &stream,
                    )
                } else {
                    reduce_host_slice::<i32>(op, col_io.dtype, &host)
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
                let host: Vec<i64> = filter_primitive_to_vec(pa);
                reduce_host_slice::<i64>(op, col_io.dtype, &host)
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
                let host: Vec<f32> = filter_primitive_to_vec(pa);
                reduce_host_slice::<f32>(op, col_io.dtype, &host)
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
                let host: Vec<f64> = filter_primitive_to_vec(pa);
                reduce_host_slice::<f64>(op, col_io.dtype, &host)
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

    let module = crate::exec::module_cache::get_or_build_module(
        module_path!(),
        format!("avg_reduce_{:?}", dtype),
        None,
        || compile_avg_reduction_kernel(dtype),
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

    let total_sum: f64 = pinned_sums.as_slice().iter().copied().sum();
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

    // Compile + load the kernel via the consolidated `exec::module_cache`.
    // The reduction kernel is keyed by `(op, dtype)` — that's the entire
    // PTX-template parameter surface. Repeat scalar reductions skip PTX gen.
    let module = module_cache::get_or_build_module(
        module_path!(),
        format!("reduction:{:?}:{:?}", op, dtype),
        None,
        || compile_reduction_kernel(op, dtype),
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
    // consolidated cache; same key as `reduce_gpu_vec` since the PTX
    // template only depends on `(op, input dtype)`.
    let module = module_cache::get_or_build_module(
        module_path!(),
        format!("reduction:{:?}:{:?}", op, dtype),
        None,
        || compile_reduction_kernel(op, dtype),
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

/// A typed scalar result of a reduction.
#[derive(Debug, Clone, Copy)]
enum Scalar {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
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
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> BoltResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0i64, i64::wrapping_add),
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

impl ReduceScalar for f32 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> BoltResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0.0f32, |a, b| a + b),
            ReduceOp::Min => host.iter().copied().fold(f32::INFINITY, f32::min),
            ReduceOp::Max => host.iter().copied().fold(f32::NEG_INFINITY, f32::max),
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
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0.0f64, |a, b| a + b),
            ReduceOp::Min => host.iter().copied().fold(f64::INFINITY, f64::min),
            ReduceOp::Max => host.iter().copied().fold(f64::NEG_INFINITY, f64::max),
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

        (s, dt) => Err(BoltError::Type(format!(
            "aggregate: cannot pack scalar {:?} into output dtype {:?}",
            s, dt
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
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        ArrowDataType::Decimal128(precision, scale) => {
            Ok(DataType::Decimal128(*precision, *scale))
        }
        other => Err(BoltError::Type(format!(
            "unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

/// Map our plan `DataType` to Arrow `DataType`.
fn plan_dtype_to_arrow(d: DataType) -> BoltResult<ArrowDataType> {
    match d {
        DataType::Int32 => Ok(ArrowDataType::Int32),
        DataType::Int64 => Ok(ArrowDataType::Int64),
        DataType::Float32 => Ok(ArrowDataType::Float32),
        DataType::Float64 => Ok(ArrowDataType::Float64),
        DataType::Bool => Ok(ArrowDataType::Boolean),
        DataType::Utf8 => Ok(ArrowDataType::Utf8),
        DataType::Decimal128(p, s) => Ok(ArrowDataType::Decimal128(p, s)),
        // v0.6 / M4: Date/Timestamp not yet wired through this aggregate
        // output helper. Reject so a regression is loud.
        DataType::Date32 | DataType::Timestamp(_, _) => Err(crate::error::BoltError::Type(
            format!("Date/Timestamp not yet supported in this aggregate output path: {:?}", d),
        )),
    }
}

/// Build an Arrow `Schema` from our plan `Schema` for the output RecordBatch.
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

// ---------------------------------------------------------------------------
// Host-only tests for the NULL-handling helpers. The full
// `execute_aggregate` path needs the GPU and is exercised by the integration
// suite; what we pin here is exactly the NULL bookkeeping that landed with
// the H1 fix: COUNT(col) excludes nulls, AVG denominator is the non-null
// count, and the pre-GPU filter keeps the raw values buffer's garbage bytes
// out of the reduction.
// ---------------------------------------------------------------------------
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
}
