// SPDX-License-Identifier: Apache-2.0

//! Aggregate execution with a pre-projection / filter pass.
//!
//! The companion to `aggregate.rs`. That module handles the trivial scalar case
//! where every aggregate input is a bare column reference and there is no
//! filter — the physical plan then carries `pre = None` and the aggregator can
//! read columns straight off the scan. This module handles the other case:
//! `pre = Some(KernelSpec)`, used when an aggregate input is a non-trivial
//! expression (e.g. `SUM(price * tax)`) or a filter is present (e.g.
//! `... WHERE region_id = 1`).
//!
//! Pipeline:
//!   1. JIT-compile `pre` as a projection kernel (potentially with a filter)
//!      and launch it to materialise the pre-aggregation columns as device
//!      buffers, exactly the way `engine.rs::execute_projection` does.
//!   2. If `pre.predicate` is set, run a separate predicate-only kernel to
//!      materialise a `u8` keep-mask. The projection kernel leaves zeros in
//!      masked slots (see `engine.rs` TODO), so we just download each column
//!      to the host and compact via the mask.
//!   3. For each `AggregateExpr`, reduce the matching compacted column via
//!      the existing per-block GPU reduction kernel from `agg_kernels`.
//!      `COUNT(_)` uses the post-mask row count directly; `AVG` is decomposed
//!      into `SUM` and `COUNT` and divided on the host.
//!   4. Pack the resulting scalars into a single-row Arrow `RecordBatch`
//!      matching `aggregate.output_schema`.
//!
//! Scope (first cut):
//!   - No GROUP BY. `aggregate.group_by` must be empty — `groupby.rs` handles
//!     that case separately.
//!   - Primitive dtypes only (Int32, Int64, Float32, Float64). Bool and Utf8
//!     are rejected; aggregate-input expressions never carry those through
//!     `pre`.
//!   - One-batch-per-launch: `table_batch` is the entire source.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};
use bytemuck::Pod;

use crate::cuda::buffer::primitive_to_gpu;
use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::expr_agg;
use crate::exec::launch::CudaStream;
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::{
    compile_reduction_kernel, ReduceOp, BLOCK_SIZE, REDUCTION_KERNEL_ENTRY,
};
use crate::jit::{compile_ptx, CudaModule};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Schema};
use crate::plan::physical_plan::{AggregateSpec, KernelSpec, PhysicalPlan};

/// PTX entry-point name for the pre-projection kernel.
const PRE_KERNEL_ENTRY: &str = "bolt_pre_kernel";

/// PTX entry-point name for the predicate-only kernel that materialises the mask.
const PRE_PREDICATE_ENTRY: &str = "bolt_pre_predicate";

/// Threads per block for the pre-projection / predicate launches.
const PRE_BLOCK_SIZE: u32 = 256;

/// Execute an aggregate plan whose `pre` is `Some(_)`.
///
/// Pipeline:
///   1. Run `pre` as a projection (possibly with filter) kernel to materialise
///      pre-aggregation columns as `GpuVec`s.
///   2. (If `pre.predicate`) build a `u8` mask via the predicate-only kernel,
///      then compact each materialised column on host into a contiguous prefix.
///   3. Reduce each compacted column via the existing scalar reduction path.
///   4. Pack scalar results into a single-row Arrow `RecordBatch` matching
///      `aggregate.output_schema`.
///
/// Errors if `aggregate.group_by` is non-empty (handled elsewhere) or if `pre`
/// is `None` (caller should use the scalar reduction path in `aggregate.rs`).
pub fn execute_aggregate_with_pre(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> BoltResult<RecordBatch> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        other => {
            return Err(BoltError::Other(format!(
                "execute_aggregate_with_pre: expected Aggregate plan, got {:?}",
                std::mem::discriminant(other)
            )))
        }
    };

    let pre_spec = pre.as_ref().ok_or_else(|| {
        BoltError::Other(
            "execute_aggregate_with_pre: pre kernel is None; use aggregate::execute_aggregate"
                .into(),
        )
    })?;

    if !aggregate.group_by.is_empty() {
        return Err(BoltError::Other(
            "agg_with_pre: GROUP BY handled separately".into(),
        ));
    }

    let n_rows = table_batch.num_rows();

    // 1+2. Run the pre kernel (and, if present, the predicate-only kernel),
    //      then download + host-compact each output column.
    let compacted = run_pre_stage(pre_spec, table_batch, n_rows)?;

    // 3+4. Reduce each materialised column and assemble the output batch.
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    let arrays = build_scalar_aggregates(aggregate, pre_spec, &compacted)?;

    RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
        BoltError::Other(format!("failed to build aggregate RecordBatch: {e}"))
    })
}

/// Materialised, host-side, post-filter output of the pre kernel: one host
/// vector per `pre.outputs`, parallel to `pre.outputs` by index.
struct CompactedPreOutputs {
    /// One host-side compacted column per `pre.outputs[i]`.
    cols: Vec<HostCol>,
}

impl CompactedPreOutputs {
    /// Post-filter row count. Identical across all columns; we use the first
    /// for the canonical value (the COUNT path leans on this).
    fn n_rows(&self) -> usize {
        self.cols.first().map(|c| c.len()).unwrap_or(0)
    }
}

/// Step 1+2: upload pre-kernel inputs, launch the projection (and optional
/// predicate) kernels, download the outputs, host-compact via the mask if any.
fn run_pre_stage(
    spec: &KernelSpec,
    table_batch: &RecordBatch,
    n_rows: usize,
) -> BoltResult<CompactedPreOutputs> {
    // -- Upload inputs.
    let mut input_cols: Vec<PreCol> = Vec::with_capacity(spec.inputs.len());
    for io in &spec.inputs {
        let idx = table_batch.schema().index_of(&io.name).map_err(|e| {
            BoltError::Plan(format!(
                "pre kernel input column '{}' not present in table batch: {}",
                io.name, e
            ))
        })?;
        let arr = table_batch.column(idx);
        let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
        if arr_dtype != io.dtype {
            return Err(BoltError::Type(format!(
                "pre kernel input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
                io.name, io.dtype, arr_dtype
            )));
        }
        input_cols.push(PreCol::upload(arr.as_ref(), io.dtype)?);
    }

    // -- Allocate zero-initialised output buffers.
    let mut output_cols: Vec<PreCol> = Vec::with_capacity(spec.outputs.len());
    for io in &spec.outputs {
        output_cols.push(PreCol::alloc_zeros(io.dtype, n_rows)?);
    }

    // -- JIT-compile the projection PTX and load it.
    let ptx = compile_ptx(spec, PRE_KERNEL_ENTRY)?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(PRE_KERNEL_ENTRY)?;

    // -- Assemble kernel parameters: inputs..., outputs..., n_rows u32.
    let mut device_ptrs: Vec<CUdeviceptr> =
        Vec::with_capacity(input_cols.len() + output_cols.len());
    for c in &input_cols {
        device_ptrs.push(c.device_ptr());
    }
    for c in &output_cols {
        device_ptrs.push(c.device_ptr());
    }
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;

    let mut kernel_params: Vec<*mut c_void> = Vec::with_capacity(device_ptrs.len() + 1);
    for p in device_ptrs.iter_mut() {
        kernel_params.push(p as *mut CUdeviceptr as *mut c_void);
    }
    kernel_params.push(&mut n_rows_u32 as *mut u32 as *mut c_void);

    // -- Launch one thread per row.
    let stream = CudaStream::null();
    if n_rows > 0 {
        let grid_x = ((n_rows_u32 + PRE_BLOCK_SIZE - 1) / PRE_BLOCK_SIZE).max(1);
        // SAFETY: `function` is borrowed from a live `CudaModule`; every entry
        // of `kernel_params` points into `device_ptrs` or `n_rows_u32`, both of
        // which outlive the launch + synchronize below.
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                function.raw(),
                grid_x,
                1,
                1,
                PRE_BLOCK_SIZE,
                1,
                1,
                0,
                stream.raw(),
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        stream.synchronize()?;
    }

    // -- Optional predicate kernel: builds a u8 keep-mask we download.
    let host_mask: Option<Vec<bool>> = if spec.predicate.is_some() {
        let pred_ptx =
            crate::jit::scan_kernel::compile_predicate_kernel(spec, PRE_PREDICATE_ENTRY)?;
        let pred_module = CudaModule::from_ptx(&pred_ptx)?;
        let pred_function = pred_module.function(PRE_PREDICATE_ENTRY)?;

        let mask = crate::exec::compact::alloc_mask_buffer(n_rows)?;
        let input_ptrs: Vec<CUdeviceptr> =
            input_cols.iter().map(|c| c.device_ptr()).collect();
        crate::exec::compact::launch_predicate_kernel(
            pred_function,
            &input_ptrs,
            mask.device_ptr(),
            n_rows_to_u32(n_rows)?,
            &stream,
        )?;
        Some(crate::exec::compact::download_mask(mask.device_ptr(), n_rows)?)
    } else {
        None
    };

    // Inputs no longer needed past this point.
    drop(input_cols);

    // -- Download each pre output, then host-compact via the mask if any.
    let mut cols: Vec<HostCol> = Vec::with_capacity(output_cols.len());
    for col in output_cols {
        let host = col.to_host_col(n_rows)?;
        let compact = match &host_mask {
            Some(mask) => host.compact(mask)?,
            None => host,
        };
        cols.push(compact);
    }

    Ok(CompactedPreOutputs { cols })
}

/// Build one Arrow scalar array per `AggregateExpr`, in `aggregate.aggregates`
/// order, against the post-filter host columns produced by the pre kernel.
fn build_scalar_aggregates(
    aggregate: &AggregateSpec,
    pre_spec: &KernelSpec,
    compacted: &CompactedPreOutputs,
) -> BoltResult<Vec<ArrayRef>> {
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
        let array = build_one_aggregate(agg, out_field, pre_spec, compacted)?;
        arrays.push(array);
    }
    Ok(arrays)
}

/// Compute a single aggregate and return its single-row Arrow array.
fn build_one_aggregate(
    agg: &AggregateExpr,
    out_field: &Field,
    pre_spec: &KernelSpec,
    compacted: &CompactedPreOutputs,
) -> BoltResult<ArrayRef> {
    match agg {
        AggregateExpr::Sum(expr) | AggregateExpr::Min(expr) | AggregateExpr::Max(expr) => {
            let op = ReduceOp::from_agg(agg)?;
            let resolved =
                resolve_agg_input_col(expr, pre_spec, compacted, out_field.dtype)?;
            let scalar = reduce_host_col(op, resolved.as_ref())?;
            scalar_to_array(scalar, out_field.dtype)
        }
        AggregateExpr::Count(expr) => {
            // SQL COUNT(col) counts non-NULL rows; COUNT(*) (planner-emitted
            // as `Count(Literal(1))`) counts every surviving row. The
            // distinction falls out naturally: `resolve_agg_input_col`
            // returns a column whose `non_null_count` already excludes
            // NULLs from expression evaluation, and a literal can never
            // produce NULLs so it matches the full row count.
            //
            // We materialise at Int64 since that's the COUNT result dtype.
            let resolved =
                resolve_agg_input_col(expr, pre_spec, compacted, DataType::Int64)?;
            let count = resolved.non_null_count() as i64;
            scalar_to_array(Scalar::I64(count), out_field.dtype)
        }
        AggregateExpr::Avg(expr) => {
            // AVG = SUM(expr) / COUNT(expr) — SQL semantics, NULLs ignored
            // in both numerator and denominator. The output is always
            // Float64. For AVG the natural materialisation dtype is Float64
            // (the reduction's accumulator + the final division both work
            // in f64). NULL inputs are filtered out by `from_expr_host`
            // before the reduction, and the denominator below uses the
            // resolved column's `non_null_count` so the average reflects
            // only the rows that contributed to the sum.
            let resolved =
                resolve_agg_input_col(expr, pre_spec, compacted, DataType::Float64)?;
            let sum_scalar = reduce_host_col(ReduceOp::Sum, resolved.as_ref())?;
            let sum_f64 = scalar_to_f64(sum_scalar)?;
            let count_f64 = resolved.non_null_count() as f64;
            let avg = if count_f64 == 0.0 { 0.0 } else { sum_f64 / count_f64 };
            scalar_to_array(Scalar::F64(avg), out_field.dtype)
        }
    }
}

/// Resolve an aggregate input expression to a host column ready for reduction.
///
/// Fast path: when `expr` (after stripping aliases) is a bare column ref whose
/// name matches one of `pre.outputs`, return a borrowed view of that compacted
/// column — no extra allocation.
///
/// Slow path: when `expr` is anything else (e.g. `Sum(price * tax)` where the
/// planner didn't pre-materialise the product), build a [`expr_agg::ColumnEnv`]
/// over the already-compacted pre outputs and use [`expr_agg::eval_expr`] to
/// materialise the value column. NULL rows produced by the host-side
/// evaluator are filtered out of the returned column so the GPU reduction
/// sees only valid values; the count of surviving rows is reported via
/// [`ResolvedHostCol::non_null_count`] (used by AVG and COUNT(col)). The
/// result is owned. `expected_dtype` is the caller-chosen materialisation
/// dtype: SUM/MIN/MAX use `out_field.dtype`, AVG uses `Float64` (the
/// reduction accumulator), and COUNT uses `Int64` (only the non-NULL count
/// is consumed, not the values).
fn resolve_agg_input_col<'a>(
    expr: &Expr,
    pre_spec: &KernelSpec,
    compacted: &'a CompactedPreOutputs,
    expected_dtype: DataType,
) -> BoltResult<ResolvedHostCol<'a>> {
    if let Some(name) = expr_agg::try_bare_column(expr) {
        let idx = pre_spec
            .outputs
            .iter()
            .position(|o| o.name == name)
            .ok_or_else(|| {
                BoltError::Plan(format!(
                    "aggregate input '{}' not found among pre kernel outputs",
                    name
                ))
            })?;
        if idx >= compacted.cols.len() {
            return Err(BoltError::Other(format!(
                "internal: pre output ordinal {} out of range (have {} compacted cols)",
                idx,
                compacted.cols.len()
            )));
        }
        let col = &compacted.cols[idx];
        // Fast path: the pre kernel output has no NULL bitmap, so every
        // row is a "non-NULL" row for aggregate purposes.
        let n = col.len();
        return Ok(ResolvedHostCol::Borrowed { col, non_null: n });
    }

    // Slow path: materialise via the host-side evaluator over the compacted
    // pre outputs. Each compacted column is wrapped lazily (lifting to
    // `Option`, which never carries a None on this path) so the evaluator can
    // run unchanged. The evaluator itself can introduce NULLs (e.g. integer
    // division by zero, or any operand of a binary op being NULL), which we
    // then filter out below so the reduction sees only valid rows.
    let n_rows = compacted.n_rows();
    let wrapped: Vec<(String, expr_agg::HostColumn)> = pre_spec
        .outputs
        .iter()
        .enumerate()
        .map(|(j, io)| (io.name.clone(), to_expr_host(&compacted.cols[j])))
        .collect();
    let env: expr_agg::ColumnEnv<'_> = wrapped.iter().map(|(n, c)| (n.clone(), c)).collect();
    let materialised = expr_agg::eval_expr(expr, &env, expected_dtype, n_rows)?;
    let (filtered, non_null) = from_expr_host(materialised)?;
    Ok(ResolvedHostCol::Owned {
        col: filtered,
        non_null,
    })
}

/// Borrowed or owned host column. Lets the fast path return `&HostCol` from
/// the compacted store while the slow path returns a freshly-materialised
/// value from `expr_agg`, with a single `as_ref()` method to feed either into
/// `reduce_host_col`.
///
/// `non_null` is the number of rows that were not SQL NULL in the source
/// column (after pre-filtering and expression evaluation). SQL aggregate
/// semantics — `SUM`, `MIN`, `MAX`, `AVG`, `COUNT(col)` — all ignore NULL
/// rows, so the executor exposes this count to callers that need it for
/// the AVG denominator and the COUNT result.
enum ResolvedHostCol<'a> {
    /// Borrowed view of a pre-stage output (no NULLs possible here).
    Borrowed { col: &'a HostCol, non_null: usize },
    /// Freshly materialised + NULL-filtered slow-path column.
    Owned { col: HostCol, non_null: usize },
}

impl<'a> ResolvedHostCol<'a> {
    fn as_ref(&self) -> &HostCol {
        match self {
            ResolvedHostCol::Borrowed { col, .. } => *col,
            ResolvedHostCol::Owned { col, .. } => col,
        }
    }

    /// Number of non-NULL rows in the original (pre-filter) input. Used by
    /// AVG (denominator) and COUNT(col) (result). For the fast path this
    /// equals the column length; for the slow path it equals the length
    /// of the filtered column (since NULL rows have been dropped).
    fn non_null_count(&self) -> usize {
        match self {
            ResolvedHostCol::Borrowed { non_null, .. } => *non_null,
            ResolvedHostCol::Owned { non_null, .. } => *non_null,
        }
    }
}

/// Lift a local primitive [`HostCol`] into the `Option`-carrying
/// [`expr_agg::HostColumn`] shape consumed by the host-side evaluator. The
/// `pre`-stage output path has no NULL bitmap support, so every cell is
/// `Some(_)`.
fn to_expr_host(c: &HostCol) -> expr_agg::HostColumn {
    match c {
        HostCol::I32(v) => expr_agg::HostColumn::I32(v.iter().map(|x| Some(*x)).collect()),
        HostCol::I64(v) => expr_agg::HostColumn::I64(v.iter().map(|x| Some(*x)).collect()),
        HostCol::F32(v) => expr_agg::HostColumn::F32(v.iter().map(|x| Some(*x)).collect()),
        HostCol::F64(v) => expr_agg::HostColumn::F64(v.iter().map(|x| Some(*x)).collect()),
    }
}

/// Convert a materialised [`expr_agg::HostColumn`] back into the local
/// primitive [`HostCol`] shape consumed by the reduction path.
///
/// **NULL handling**: SQL aggregate semantics require `SUM`/`MIN`/`MAX`/`AVG`/
/// `COUNT(col)` to ignore NULL rows. This function drops every `None` entry
/// from the input column and returns the filtered values plus the count of
/// surviving (non-NULL) rows. Callers use that count as the AVG denominator
/// and the COUNT(col) result.
///
/// A previous version of this function coerced NULLs to zero, which was a
/// correctness bug: `MIN([NULL, 5])` returned `0` instead of `5`,
/// `MAX([NULL, -3])` returned `0` instead of `-3`, and `AVG([NULL, 4])`
/// returned `2.0` (4/2) instead of `4.0` (4/1).
///
/// Bool / Utf8 materialisations are still rejected — the reduction kernels
/// only accept primitive numeric inputs.
fn from_expr_host(c: expr_agg::HostColumn) -> BoltResult<(HostCol, usize)> {
    match c {
        expr_agg::HostColumn::I32(v) => {
            let filtered: Vec<i32> = v.into_iter().flatten().collect();
            let n = filtered.len();
            Ok((HostCol::I32(filtered), n))
        }
        expr_agg::HostColumn::I64(v) => {
            let filtered: Vec<i64> = v.into_iter().flatten().collect();
            let n = filtered.len();
            Ok((HostCol::I64(filtered), n))
        }
        expr_agg::HostColumn::F32(v) => {
            let filtered: Vec<f32> = v.into_iter().flatten().collect();
            let n = filtered.len();
            Ok((HostCol::F32(filtered), n))
        }
        expr_agg::HostColumn::F64(v) => {
            let filtered: Vec<f64> = v.into_iter().flatten().collect();
            let n = filtered.len();
            Ok((HostCol::F64(filtered), n))
        }
        expr_agg::HostColumn::Bool(_) | expr_agg::HostColumn::Utf8(_) => {
            Err(BoltError::Type(
                "agg_with_pre: Bool/Utf8 aggregate inputs not supported by the \
                 primitive reduction path"
                    .into(),
            ))
        }
    }
}

/// Run a GPU reduction over `col` and return the scalar result.
fn reduce_host_col(op: ReduceOp, col: &HostCol) -> BoltResult<Scalar> {
    match col {
        HostCol::I32(v) => reduce_host_slice::<i32>(op, DataType::Int32, v),
        HostCol::I64(v) => reduce_host_slice::<i64>(op, DataType::Int64, v),
        HostCol::F32(v) => reduce_host_slice::<f32>(op, DataType::Float32, v),
        HostCol::F64(v) => reduce_host_slice::<f64>(op, DataType::Float64, v),
    }
}

/// Upload a host slice, then run the standard GPU reduction over it.
fn reduce_host_slice<T>(op: ReduceOp, dtype: DataType, host: &[T]) -> BoltResult<Scalar>
where
    T: Pod + ReduceScalar,
{
    let dev = GpuVec::<T>::from_slice(host)?;
    reduce_gpu_vec::<T>(op, dtype, &dev, host.len())
}

/// Launch the per-block reduction kernel against `input` and finish the
/// reduction on the host. Returns the final scalar as a `Scalar`.
fn reduce_gpu_vec<T>(
    op: ReduceOp,
    dtype: DataType,
    input: &GpuVec<T>,
    n_rows: usize,
) -> BoltResult<Scalar>
where
    T: Pod + ReduceScalar,
{
    // 0-row degenerate case: skip the launch entirely and return the identity.
    if n_rows == 0 {
        return T::identity_scalar(op, dtype);
    }

    let block = BLOCK_SIZE;
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let grid_x = ((n_rows_u32 + block - 1) / block).max(1);
    let partials = GpuVec::<T>::zeros(grid_x as usize)?;

    let ptx = compile_reduction_kernel(op, dtype)?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(REDUCTION_KERNEL_ENTRY)?;

    let mut input_ptr: CUdeviceptr = input.device_ptr();
    let mut output_ptr: CUdeviceptr = partials.device_ptr();

    let mut kernel_params: [*mut c_void; 3] = [
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut output_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
    ];

    let stream = CudaStream::null();
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
    stream.synchronize()?;

    let _ = input_ptr;
    let _ = output_ptr;

    let host_partials = partials.to_vec()?;
    drop(partials);
    T::finalize(op, dtype, &host_partials)
}

/// Heterogenous owned device column for the pre kernel's inputs and outputs.
/// Only primitive numeric dtypes are reachable here — Bool / Utf8 are rejected
/// because aggregate inputs (and the expressions feeding them) never carry
/// those types.
enum PreCol {
    I32(GpuVec<i32>),
    I64(GpuVec<i64>),
    F32(GpuVec<f32>),
    F64(GpuVec<f64>),
}

impl PreCol {
    /// Upload an Arrow array to the GPU, downcasting per `dtype`.
    ///
    /// **NULL handling**: rejects any input array with `null_count() > 0`.
    /// The pre-stage GPU kernels operate on raw value buffers and do not
    /// (yet) carry a validity bitmap, so a NULL would silently feed garbage
    /// into multiplications / additions. The clean error is the safe
    /// surgical fix; full propagation is tracked below.
    ///
    /// TODO(pre-stage-nulls): propagate `arr.nulls()` into a parallel
    /// `valid_mask: GpuBuffer<u8>` per `PreCol`, then have the JIT kernel
    /// emit `value if valid else identity` per op so NULL semantics survive
    /// the pre-stage. Until then, planners that introduce a pre kernel
    /// over a NULL-bearing source column must ensure either (a) the source
    /// has been filtered upstream or (b) the column is unconditionally
    /// non-null.
    fn upload(arr: &dyn Array, dtype: DataType) -> BoltResult<Self> {
        if arr.null_count() > 0 {
            return Err(BoltError::Type(format!(
                "agg_with_pre: pre-stage kernels do not yet propagate NULL \
                 validity; input column with dtype {:?} has {} NULL(s). \
                 See TODO(pre-stage-nulls) in agg_with_pre.rs.",
                dtype,
                arr.null_count()
            )));
        }
        match dtype {
            DataType::Int32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| downcast_err("input", "Int32"))?;
                Ok(PreCol::I32(GpuVec::from_buffer(primitive_to_gpu(pa)?)))
            }
            DataType::Int64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| downcast_err("input", "Int64"))?;
                Ok(PreCol::I64(GpuVec::from_buffer(primitive_to_gpu(pa)?)))
            }
            DataType::Float32 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| downcast_err("input", "Float32"))?;
                Ok(PreCol::F32(GpuVec::from_buffer(primitive_to_gpu(pa)?)))
            }
            DataType::Float64 => {
                let pa = arr
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| downcast_err("input", "Float64"))?;
                Ok(PreCol::F64(GpuVec::from_buffer(primitive_to_gpu(pa)?)))
            }
            DataType::Bool | DataType::Utf8 => Err(BoltError::Type(format!(
                "agg_with_pre: pre kernel column dtype {:?} not supported",
                dtype
            ))),
        }
    }

    /// Allocate a zero-initialised device column of `n` rows.
    fn alloc_zeros(dtype: DataType, n: usize) -> BoltResult<Self> {
        match dtype {
            DataType::Int32 => Ok(PreCol::I32(GpuVec::<i32>::zeros(n)?)),
            DataType::Int64 => Ok(PreCol::I64(GpuVec::<i64>::zeros(n)?)),
            DataType::Float32 => Ok(PreCol::F32(GpuVec::<f32>::zeros(n)?)),
            DataType::Float64 => Ok(PreCol::F64(GpuVec::<f64>::zeros(n)?)),
            DataType::Bool | DataType::Utf8 => Err(BoltError::Type(format!(
                "agg_with_pre: pre kernel output dtype {:?} not supported",
                dtype
            ))),
        }
    }

    /// Raw device pointer for kernel-parameter assembly.
    fn device_ptr(&self) -> CUdeviceptr {
        match self {
            PreCol::I32(v) => v.device_ptr(),
            PreCol::I64(v) => v.device_ptr(),
            PreCol::F32(v) => v.device_ptr(),
            PreCol::F64(v) => v.device_ptr(),
        }
    }

    /// Download the column to host and verify the length matches `n_rows`.
    fn to_host_col(self, n_rows: usize) -> BoltResult<HostCol> {
        match self {
            PreCol::I32(v) => Ok(HostCol::I32(copy_back::<i32>(&v, n_rows)?)),
            PreCol::I64(v) => Ok(HostCol::I64(copy_back::<i64>(&v, n_rows)?)),
            PreCol::F32(v) => Ok(HostCol::F32(copy_back::<f32>(&v, n_rows)?)),
            PreCol::F64(v) => Ok(HostCol::F64(copy_back::<f64>(&v, n_rows)?)),
        }
    }
}

/// Host-side typed column produced by downloading a `PreCol`.
enum HostCol {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl HostCol {
    /// Number of elements in the column.
    fn len(&self) -> usize {
        match self {
            HostCol::I32(v) => v.len(),
            HostCol::I64(v) => v.len(),
            HostCol::F32(v) => v.len(),
            HostCol::F64(v) => v.len(),
        }
    }

    /// Return a new column containing only positions where `mask[i]` is true.
    /// The pre-projection kernel leaves zeros in masked slots, so we drop those
    /// positions and keep the rest in original order.
    fn compact(self, mask: &[bool]) -> BoltResult<HostCol> {
        if mask.len() != self.len() {
            return Err(BoltError::Other(format!(
                "agg_with_pre: mask length {} != column length {}",
                mask.len(),
                self.len()
            )));
        }
        Ok(match self {
            HostCol::I32(v) => HostCol::I32(filter_vec(v, mask)),
            HostCol::I64(v) => HostCol::I64(filter_vec(v, mask)),
            HostCol::F32(v) => HostCol::F32(filter_vec(v, mask)),
            HostCol::F64(v) => HostCol::F64(filter_vec(v, mask)),
        })
    }
}

/// Keep `v[i]` iff `mask[i]`; returns a fresh Vec.
fn filter_vec<T: Copy>(v: Vec<T>, mask: &[bool]) -> Vec<T> {
    v.into_iter()
        .zip(mask.iter())
        .filter_map(|(x, &k)| if k { Some(x) } else { None })
        .collect()
}

/// Copy back a `GpuVec<T>` into a host `Vec<T>` of length `n_rows`.
fn copy_back<T>(v: &GpuVec<T>, n_rows: usize) -> BoltResult<Vec<T>>
where
    T: Pod,
{
    let host = v.to_vec()?;
    if host.len() != n_rows {
        return Err(BoltError::Other(format!(
            "internal: device buffer length {} did not match expected {}",
            host.len(),
            n_rows
        )));
    }
    Ok(host)
}

/// A typed scalar result of a reduction.
#[derive(Debug, Clone, Copy)]
enum Scalar {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

/// Per-`T` helpers for the GPU reduction path. Mirrors the trait in
/// `aggregate.rs` exactly so this module stays self-contained.
trait ReduceScalar: Sized + Copy {
    fn finalize(op: ReduceOp, dtype: DataType, host: &[Self]) -> BoltResult<Scalar>;
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

        // Common cross-dtype output paths (AVG -> Float64, COUNT -> Int64).
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
            "agg_with_pre: cannot pack scalar {:?} into output dtype {:?}",
            s, dt
        ))),
    }
}

/// Cast a scalar to f64 (used by AVG).
fn scalar_to_f64(s: Scalar) -> BoltResult<f64> {
    Ok(match s {
        Scalar::I32(v) => v as f64,
        Scalar::I64(v) => v as f64,
        Scalar::F32(v) => v as f64,
        Scalar::F64(v) => v,
    })
}

/// Build a `Type` error for a failed Arrow downcast.
fn downcast_err(role: &str, expected: &str) -> BoltError {
    BoltError::Type(format!(
        "agg_with_pre: pre kernel {} could not be downcast to {}",
        role, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType`. Mirror of the helper in
/// `aggregate.rs` and `engine.rs`, copied here to avoid reaching across module
/// privacy.
fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
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
    }
}

/// Build an Arrow `Schema` from our plan `Schema` for the output `RecordBatch`.
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// These tests cover the host-only NULL-filtering contract of
// `from_expr_host`, which is the patch point for a correctness bug: NULLs in
// aggregate inputs used to be coerced to dtype-zero, which broke
// `MIN`/`MAX`/`AVG`/`COUNT(col)`. The tests cannot exercise the GPU
// reduction path directly (it requires CUDA), so they verify that
// `from_expr_host` drops NULL entries and reports the surviving count, then
// emulate what each GPU aggregate kernel would compute on the filtered
// values to assert the end-to-end answer matches SQL semantics.

#[cfg(test)]
mod tests {
    use super::*;

    /// Emulate the host-side finalization that `reduce_host_slice` would
    /// perform after the GPU returns its per-block partials. Lets us assert
    /// the end-to-end SQL semantics without a CUDA context.
    fn host_reduce_i32(op: ReduceOp, vs: &[i32]) -> i32 {
        match op {
            ReduceOp::Sum | ReduceOp::Count => vs.iter().copied().fold(0, i32::wrapping_add),
            ReduceOp::Min => vs.iter().copied().fold(i32::MAX, i32::min),
            ReduceOp::Max => vs.iter().copied().fold(i32::MIN, i32::max),
        }
    }

    fn host_reduce_f64(op: ReduceOp, vs: &[f64]) -> f64 {
        match op {
            ReduceOp::Sum | ReduceOp::Count => vs.iter().copied().fold(0.0, |a, b| a + b),
            ReduceOp::Min => vs.iter().copied().fold(f64::INFINITY, f64::min),
            ReduceOp::Max => vs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }

    /// `from_expr_host` drops NULL entries and reports the count of
    /// non-NULL values for an `I32` column. Covers the I32 arm of the bug
    /// fix.
    #[test]
    fn from_expr_host_filters_i32_nulls() {
        let col = expr_agg::HostColumn::I32(vec![None, Some(5), None, Some(7), Some(-3)]);
        let (host, non_null) = from_expr_host(col).expect("ok");
        assert_eq!(non_null, 3);
        match host {
            HostCol::I32(v) => assert_eq!(v, vec![5, 7, -3]),
            _ => panic!("expected I32 variant"),
        }
    }

    /// Same as above for `I64`.
    #[test]
    fn from_expr_host_filters_i64_nulls() {
        let col = expr_agg::HostColumn::I64(vec![Some(10), None, Some(20)]);
        let (host, non_null) = from_expr_host(col).expect("ok");
        assert_eq!(non_null, 2);
        match host {
            HostCol::I64(v) => assert_eq!(v, vec![10, 20]),
            _ => panic!("expected I64"),
        }
    }

    /// Same as above for `F32`.
    #[test]
    fn from_expr_host_filters_f32_nulls() {
        let col = expr_agg::HostColumn::F32(vec![None, Some(1.5), Some(2.5), None]);
        let (host, non_null) = from_expr_host(col).expect("ok");
        assert_eq!(non_null, 2);
        match host {
            HostCol::F32(v) => assert_eq!(v, vec![1.5, 2.5]),
            _ => panic!("expected F32"),
        }
    }

    /// Same as above for `F64`.
    #[test]
    fn from_expr_host_filters_f64_nulls() {
        let col = expr_agg::HostColumn::F64(vec![Some(4.0), None]);
        let (host, non_null) = from_expr_host(col).expect("ok");
        assert_eq!(non_null, 1);
        match host {
            HostCol::F64(v) => assert_eq!(v, vec![4.0]),
            _ => panic!("expected F64"),
        }
    }

    /// `MIN` over a column with NULLs and one non-NULL must return that
    /// non-NULL value, not the dtype's zero. Pre-fix this returned 0.
    #[test]
    fn min_ignores_nulls() {
        let col = expr_agg::HostColumn::I32(vec![None, Some(5)]);
        let (host, non_null) = from_expr_host(col).expect("ok");
        assert_eq!(non_null, 1);
        let values = match host {
            HostCol::I32(v) => v,
            _ => panic!("expected I32"),
        };
        assert_eq!(host_reduce_i32(ReduceOp::Min, &values), 5);
    }

    /// `MAX([NULL, -3])` must be `-3`, not `0`. Pre-fix it was `0` because
    /// NULL coerced to zero and `max(0, -3) == 0`.
    #[test]
    fn max_ignores_nulls() {
        let col = expr_agg::HostColumn::I32(vec![None, Some(-3)]);
        let (host, non_null) = from_expr_host(col).expect("ok");
        assert_eq!(non_null, 1);
        let values = match host {
            HostCol::I32(v) => v,
            _ => panic!("expected I32"),
        };
        assert_eq!(host_reduce_i32(ReduceOp::Max, &values), -3);
    }

    /// `AVG([NULL, 4])` must be `4.0` (sum=4, count=1), not `2.0`
    /// (sum=4 with NULL-as-zero, count=2 with NULL included). The
    /// denominator comes from `non_null`, not the original row count.
    #[test]
    fn avg_ignores_nulls_in_numerator_and_denominator() {
        // The slow path materialises AVG inputs at Float64.
        let col = expr_agg::HostColumn::F64(vec![None, Some(4.0)]);
        let (host, non_null) = from_expr_host(col).expect("ok");
        assert_eq!(non_null, 1);
        let values = match host {
            HostCol::F64(v) => v,
            _ => panic!("expected F64"),
        };
        let sum = host_reduce_f64(ReduceOp::Sum, &values);
        let avg = sum / (non_null as f64);
        assert!((avg - 4.0).abs() < 1e-12, "avg was {avg}, expected 4.0");
    }

    /// `SUM` was previously "accidentally correct" because 0 is the SUM
    /// identity. Verify it stays correct under the new filter-then-reduce
    /// pipeline.
    #[test]
    fn sum_matches_non_null_sum() {
        let col = expr_agg::HostColumn::I32(vec![None, Some(5), None, Some(7)]);
        let (host, non_null) = from_expr_host(col).expect("ok");
        assert_eq!(non_null, 2);
        let values = match host {
            HostCol::I32(v) => v,
            _ => panic!("expected I32"),
        };
        assert_eq!(host_reduce_i32(ReduceOp::Sum, &values), 12);
    }

    /// `COUNT(col)` excludes NULLs: a column with two NULLs and two
    /// non-NULLs has count 2. Pre-fix `build_one_aggregate` used the
    /// pre-stage row count (4 here), which double-counted the NULLs.
    #[test]
    fn count_col_excludes_nulls() {
        let col = expr_agg::HostColumn::I32(vec![None, Some(5), None, Some(7)]);
        let (_host, non_null) = from_expr_host(col).expect("ok");
        assert_eq!(non_null, 2);
    }

    /// All-NULL input: `from_expr_host` returns an empty column and
    /// `non_null == 0`. Downstream, `reduce_host_slice` short-circuits
    /// the GPU launch (see `reduce_gpu_vec`'s `n_rows == 0` guard) and
    /// returns the reduction's identity. SQL would say MIN/MAX/SUM/AVG
    /// over an all-NULL column should be NULL, but the existing
    /// non-nullable scalar output path cannot express that — out of
    /// scope for this fix. `COUNT(col)` correctly yields 0.
    #[test]
    fn all_null_yields_empty_and_zero_count() {
        let col = expr_agg::HostColumn::I32(vec![None, None, None]);
        let (host, non_null) = from_expr_host(col).expect("ok");
        assert_eq!(non_null, 0);
        match host {
            HostCol::I32(v) => assert!(v.is_empty(), "expected empty vec, got {v:?}"),
            _ => panic!("expected I32"),
        }
    }

    /// Bool / Utf8 are still rejected — the slow-path reduction kernels
    /// only accept primitive numeric inputs.
    #[test]
    fn from_expr_host_rejects_bool_and_utf8() {
        let bool_col = expr_agg::HostColumn::Bool(vec![Some(true)]);
        assert!(matches!(
            from_expr_host(bool_col),
            Err(BoltError::Type(_))
        ));

        let utf8_col = expr_agg::HostColumn::Utf8(vec![Some("x".to_string())]);
        assert!(matches!(
            from_expr_host(utf8_col),
            Err(BoltError::Type(_))
        ));
    }
}

#[cfg(test)]
mod null_handling_tests {
    //! Host-only tests for the `PreCol::upload` NULL-rejection gate added in
    //! the pre-stage: Arrow arrays whose `null_count() > 0` are rejected
    //! before any device allocation because the pre-stage GPU kernels do not
    //! yet carry a validity bitmap and would otherwise multiply garbage at
    //! NULL positions.
    //!
    //! No CUDA touched: the error path short-circuits before any device
    //! allocation.
    use super::*;

    #[test]
    fn pre_col_upload_rejects_null_bearing_input() {
        // Arrow `Int32Array::from` with `Option`s carries a real validity
        // bitmap, so `null_count()` returns > 0 and we should reject this
        // before reaching CUDA.
        let arr = Int32Array::from(vec![Some(1i32), None, Some(3i32)]);
        assert!(arr.null_count() > 0, "arrow array should carry a NULL");
        // `PreCol` doesn't implement `Debug`, so we match the Result manually
        // rather than calling `expect_err`.
        let msg = match PreCol::upload(&arr, DataType::Int32) {
            Ok(_) => panic!("PreCol::upload should reject NULL-bearing input"),
            Err(e) => format!("{e}"),
        };
        assert!(
            msg.contains("NULL"),
            "error should mention NULL validity, got: {msg}"
        );
        assert!(
            msg.contains("pre-stage"),
            "error should mention pre-stage, got: {msg}"
        );
    }
}
