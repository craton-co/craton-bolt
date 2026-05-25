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
use crate::error::{PatinaError, PatinaResult};
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
const PRE_KERNEL_ENTRY: &str = "patina_pre_kernel";

/// PTX entry-point name for the predicate-only kernel that materialises the mask.
const PRE_PREDICATE_ENTRY: &str = "patina_pre_predicate";

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
) -> PatinaResult<RecordBatch> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        other => {
            return Err(PatinaError::Other(format!(
                "execute_aggregate_with_pre: expected Aggregate plan, got {:?}",
                std::mem::discriminant(other)
            )))
        }
    };

    let pre_spec = pre.as_ref().ok_or_else(|| {
        PatinaError::Other(
            "execute_aggregate_with_pre: pre kernel is None; use aggregate::execute_aggregate"
                .into(),
        )
    })?;

    if !aggregate.group_by.is_empty() {
        return Err(PatinaError::Other(
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
        PatinaError::Other(format!("failed to build aggregate RecordBatch: {e}"))
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
) -> PatinaResult<CompactedPreOutputs> {
    // -- Upload inputs.
    let mut input_cols: Vec<PreCol> = Vec::with_capacity(spec.inputs.len());
    for io in &spec.inputs {
        let idx = table_batch.schema().index_of(&io.name).map_err(|e| {
            PatinaError::Plan(format!(
                "pre kernel input column '{}' not present in table batch: {}",
                io.name, e
            ))
        })?;
        let arr = table_batch.column(idx);
        let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
        if arr_dtype != io.dtype {
            return Err(PatinaError::Type(format!(
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
) -> PatinaResult<Vec<ArrayRef>> {
    if aggregate.output_schema.fields.len() != aggregate.aggregates.len() {
        return Err(PatinaError::Other(format!(
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
) -> PatinaResult<ArrayRef> {
    match agg {
        AggregateExpr::Sum(expr) | AggregateExpr::Min(expr) | AggregateExpr::Max(expr) => {
            let op = ReduceOp::from_agg(agg)?;
            let resolved =
                resolve_agg_input_col(expr, pre_spec, compacted, out_field.dtype)?;
            let scalar = reduce_host_col(op, resolved.as_ref())?;
            scalar_to_array(scalar, out_field.dtype)
        }
        AggregateExpr::Count(_) => {
            // COUNT with no NULL handling: the number of surviving rows.
            let count = compacted.n_rows() as i64;
            scalar_to_array(Scalar::I64(count), out_field.dtype)
        }
        AggregateExpr::Avg(expr) => {
            // AVG = SUM(expr) / COUNT(expr). SUM is computed on the GPU; COUNT
            // is the post-mask row count. The output is always Float64.
            // For AVG the natural materialisation dtype is Float64 (the
            // reduction's accumulator + the final division both work in f64).
            let resolved =
                resolve_agg_input_col(expr, pre_spec, compacted, DataType::Float64)?;
            let sum_scalar = reduce_host_col(ReduceOp::Sum, resolved.as_ref())?;
            let sum_f64 = scalar_to_f64(sum_scalar)?;
            let count_f64 = compacted.n_rows() as f64;
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
/// materialise the value column. The result is owned. `expected_dtype` is the
/// caller-chosen materialisation dtype: for SUM/MIN/MAX use `out_field.dtype`;
/// for AVG use `Float64` (the reduction accumulator).
fn resolve_agg_input_col<'a>(
    expr: &Expr,
    pre_spec: &KernelSpec,
    compacted: &'a CompactedPreOutputs,
    expected_dtype: DataType,
) -> PatinaResult<ResolvedHostCol<'a>> {
    if let Some(name) = expr_agg::try_bare_column(expr) {
        let idx = pre_spec
            .outputs
            .iter()
            .position(|o| o.name == name)
            .ok_or_else(|| {
                PatinaError::Plan(format!(
                    "aggregate input '{}' not found among pre kernel outputs",
                    name
                ))
            })?;
        if idx >= compacted.cols.len() {
            return Err(PatinaError::Other(format!(
                "internal: pre output ordinal {} out of range (have {} compacted cols)",
                idx,
                compacted.cols.len()
            )));
        }
        return Ok(ResolvedHostCol::Borrowed(&compacted.cols[idx]));
    }

    // Slow path: materialise via the host-side evaluator over the compacted
    // pre outputs. Each compacted column is wrapped lazily (lifting to
    // `Option`, which never carries a None on this path) so the evaluator can
    // run unchanged.
    let n_rows = compacted.n_rows();
    let wrapped: Vec<(String, expr_agg::HostColumn)> = pre_spec
        .outputs
        .iter()
        .enumerate()
        .map(|(j, io)| (io.name.clone(), to_expr_host(&compacted.cols[j])))
        .collect();
    let env: expr_agg::ColumnEnv<'_> = wrapped.iter().map(|(n, c)| (n.clone(), c)).collect();
    let materialised = expr_agg::eval_expr(expr, &env, expected_dtype, n_rows)?;
    Ok(ResolvedHostCol::Owned(from_expr_host(materialised)?))
}

/// Borrowed or owned host column. Lets the fast path return `&HostCol` from
/// the compacted store while the slow path returns a freshly-materialised
/// value from `expr_agg`, with a single `as_ref()` method to feed either into
/// `reduce_host_col`.
enum ResolvedHostCol<'a> {
    Borrowed(&'a HostCol),
    Owned(HostCol),
}

impl<'a> ResolvedHostCol<'a> {
    fn as_ref(&self) -> &HostCol {
        match self {
            ResolvedHostCol::Borrowed(c) => *c,
            ResolvedHostCol::Owned(c) => c,
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
/// primitive [`HostCol`] shape consumed by the reduction path. NULLs collapse
/// to the dtype's zero, matching the zero-initialised slots that the pre
/// kernel writes for masked-out rows (the reduction's identity is also zero
/// for SUM/COUNT, so this preserves the existing semantics). Bool / Utf8
/// materialisations are rejected — the reduction kernels only accept
/// primitive numeric inputs.
fn from_expr_host(c: expr_agg::HostColumn) -> PatinaResult<HostCol> {
    match c {
        expr_agg::HostColumn::I32(v) => {
            Ok(HostCol::I32(v.into_iter().map(|x| x.unwrap_or(0)).collect()))
        }
        expr_agg::HostColumn::I64(v) => {
            Ok(HostCol::I64(v.into_iter().map(|x| x.unwrap_or(0)).collect()))
        }
        expr_agg::HostColumn::F32(v) => Ok(HostCol::F32(
            v.into_iter().map(|x| x.unwrap_or(0.0)).collect(),
        )),
        expr_agg::HostColumn::F64(v) => Ok(HostCol::F64(
            v.into_iter().map(|x| x.unwrap_or(0.0)).collect(),
        )),
        expr_agg::HostColumn::Bool(_) | expr_agg::HostColumn::Utf8(_) => {
            Err(PatinaError::Type(
                "agg_with_pre: Bool/Utf8 aggregate inputs not supported by the \
                 primitive reduction path"
                    .into(),
            ))
        }
    }
}

/// Run a GPU reduction over `col` and return the scalar result.
fn reduce_host_col(op: ReduceOp, col: &HostCol) -> PatinaResult<Scalar> {
    match col {
        HostCol::I32(v) => reduce_host_slice::<i32>(op, DataType::Int32, v),
        HostCol::I64(v) => reduce_host_slice::<i64>(op, DataType::Int64, v),
        HostCol::F32(v) => reduce_host_slice::<f32>(op, DataType::Float32, v),
        HostCol::F64(v) => reduce_host_slice::<f64>(op, DataType::Float64, v),
    }
}

/// Upload a host slice, then run the standard GPU reduction over it.
fn reduce_host_slice<T>(op: ReduceOp, dtype: DataType, host: &[T]) -> PatinaResult<Scalar>
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
) -> PatinaResult<Scalar>
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
    fn upload(arr: &dyn Array, dtype: DataType) -> PatinaResult<Self> {
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
            DataType::Bool | DataType::Utf8 => Err(PatinaError::Type(format!(
                "agg_with_pre: pre kernel column dtype {:?} not supported",
                dtype
            ))),
        }
    }

    /// Allocate a zero-initialised device column of `n` rows.
    fn alloc_zeros(dtype: DataType, n: usize) -> PatinaResult<Self> {
        match dtype {
            DataType::Int32 => Ok(PreCol::I32(GpuVec::<i32>::zeros(n)?)),
            DataType::Int64 => Ok(PreCol::I64(GpuVec::<i64>::zeros(n)?)),
            DataType::Float32 => Ok(PreCol::F32(GpuVec::<f32>::zeros(n)?)),
            DataType::Float64 => Ok(PreCol::F64(GpuVec::<f64>::zeros(n)?)),
            DataType::Bool | DataType::Utf8 => Err(PatinaError::Type(format!(
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
    fn to_host_col(self, n_rows: usize) -> PatinaResult<HostCol> {
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
    fn compact(self, mask: &[bool]) -> PatinaResult<HostCol> {
        if mask.len() != self.len() {
            return Err(PatinaError::Other(format!(
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
fn copy_back<T>(v: &GpuVec<T>, n_rows: usize) -> PatinaResult<Vec<T>>
where
    T: Pod,
{
    let host = v.to_vec()?;
    if host.len() != n_rows {
        return Err(PatinaError::Other(format!(
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
    fn finalize(op: ReduceOp, dtype: DataType, host: &[Self]) -> PatinaResult<Scalar>;
    fn identity_scalar(op: ReduceOp, dtype: DataType) -> PatinaResult<Scalar>;
}

impl ReduceScalar for i32 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> PatinaResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0i32, i32::wrapping_add),
            ReduceOp::Min => host.iter().copied().fold(i32::MAX, i32::min),
            ReduceOp::Max => host.iter().copied().fold(i32::MIN, i32::max),
        };
        Ok(Scalar::I32(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> PatinaResult<Scalar> {
        Ok(Scalar::I32(match op {
            ReduceOp::Sum | ReduceOp::Count => 0,
            ReduceOp::Min => i32::MAX,
            ReduceOp::Max => i32::MIN,
        }))
    }
}

impl ReduceScalar for i64 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> PatinaResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0i64, i64::wrapping_add),
            ReduceOp::Min => host.iter().copied().fold(i64::MAX, i64::min),
            ReduceOp::Max => host.iter().copied().fold(i64::MIN, i64::max),
        };
        Ok(Scalar::I64(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> PatinaResult<Scalar> {
        Ok(Scalar::I64(match op {
            ReduceOp::Sum | ReduceOp::Count => 0,
            ReduceOp::Min => i64::MAX,
            ReduceOp::Max => i64::MIN,
        }))
    }
}

impl ReduceScalar for f32 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> PatinaResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0.0f32, |a, b| a + b),
            ReduceOp::Min => host.iter().copied().fold(f32::INFINITY, f32::min),
            ReduceOp::Max => host.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        };
        Ok(Scalar::F32(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> PatinaResult<Scalar> {
        Ok(Scalar::F32(match op {
            ReduceOp::Sum | ReduceOp::Count => 0.0,
            ReduceOp::Min => f32::INFINITY,
            ReduceOp::Max => f32::NEG_INFINITY,
        }))
    }
}

impl ReduceScalar for f64 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> PatinaResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0.0f64, |a, b| a + b),
            ReduceOp::Min => host.iter().copied().fold(f64::INFINITY, f64::min),
            ReduceOp::Max => host.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        };
        Ok(Scalar::F64(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> PatinaResult<Scalar> {
        Ok(Scalar::F64(match op {
            ReduceOp::Sum | ReduceOp::Count => 0.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
        }))
    }
}

/// Convert a `Scalar` into a single-element Arrow array of `out_dtype`.
fn scalar_to_array(scalar: Scalar, out_dtype: DataType) -> PatinaResult<ArrayRef> {
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

        (s, dt) => Err(PatinaError::Type(format!(
            "agg_with_pre: cannot pack scalar {:?} into output dtype {:?}",
            s, dt
        ))),
    }
}

/// Cast a scalar to f64 (used by AVG).
fn scalar_to_f64(s: Scalar) -> PatinaResult<f64> {
    Ok(match s {
        Scalar::I32(v) => v as f64,
        Scalar::I64(v) => v as f64,
        Scalar::F32(v) => v as f64,
        Scalar::F64(v) => v,
    })
}

/// Build a `Type` error for a failed Arrow downcast.
fn downcast_err(role: &str, expected: &str) -> PatinaError {
    PatinaError::Type(format!(
        "agg_with_pre: pre kernel {} could not be downcast to {}",
        role, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType`. Mirror of the helper in
/// `aggregate.rs` and `engine.rs`, copied here to avoid reaching across module
/// privacy.
fn arrow_dtype_to_plan(d: &ArrowDataType) -> PatinaResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        other => Err(PatinaError::Type(format!(
            "unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

/// Map our plan `DataType` to Arrow `DataType`.
fn plan_dtype_to_arrow(d: DataType) -> PatinaResult<ArrowDataType> {
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
fn plan_schema_to_arrow_schema(s: &Schema) -> PatinaResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}
