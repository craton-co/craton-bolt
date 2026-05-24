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
//! Scope (first cut):
//!   - No GROUP BY. `aggregate.group_by` must be empty.
//!   - No pre-aggregation kernel. `pre` must be `None`; this is the shape the
//!     physical-plan lowering produces for queries like `SELECT SUM(c) FROM t`
//!     where every aggregate input is a bare column reference and there is no
//!     filter. `pre = Some(...)` returns a "not yet implemented" error.
//!   - Primitive dtypes only (Int32, Int64, Float32, Float64).
//!   - `AVG` is decomposed into a `SUM` and a `COUNT` kernel and divided on
//!     the host. `COUNT` is implemented as a `SUM` over a synthetic all-ones
//!     `Int64` column; correct but extra-allocation-heavy, fine for v1.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
};
use bytemuck::Pod;

use crate::cuda::buffer::primitive_to_gpu;
use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};
use crate::exec::launch::CudaStream;
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::{
    compile_reduction_kernel, ReduceOp, BLOCK_SIZE, REDUCTION_KERNEL_ENTRY,
};
use crate::jit::CudaModule;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Schema};
use crate::plan::physical_plan::{AggregateSpec, ColumnIO, PhysicalPlan};

/// Execute an aggregate physical plan against a host-side RecordBatch.
///
/// `table_batch` must already be the relevant batch for `plan` (the caller
/// resolves the table name to a batch).
pub fn execute_aggregate(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> JavelinResult<RecordBatch> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        PhysicalPlan::Projection { .. } => {
            return Err(JavelinError::Other(
                "execute_aggregate: expected Aggregate plan, got Projection".into(),
            ))
        }
    };

    if !aggregate.group_by.is_empty() {
        return Err(JavelinError::Other(
            "GROUP BY aggregate not yet implemented".into(),
        ));
    }
    if pre.is_some() {
        return Err(JavelinError::Other(
            "aggregate with projection/filter not yet implemented in scalar reduction path"
                .into(),
        ));
    }

    let n_rows = table_batch.num_rows();
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    let arrays = build_scalar_aggregates(aggregate, table_batch, n_rows)?;

    RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
        JavelinError::Other(format!("failed to build aggregate RecordBatch: {e}"))
    })
}

/// Build one Arrow scalar array per `AggregateExpr`, in `aggregate.aggregates`
/// order, against the host-side `table_batch`.
fn build_scalar_aggregates(
    aggregate: &AggregateSpec,
    table_batch: &RecordBatch,
    n_rows: usize,
) -> JavelinResult<Vec<ArrayRef>> {
    // The output schema has one field per aggregate (no group keys), in order.
    if aggregate.output_schema.fields.len() != aggregate.aggregates.len() {
        return Err(JavelinError::Other(format!(
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
) -> JavelinResult<ArrayRef> {
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
        AggregateExpr::Count(_) => {
            // COUNT(*) / COUNT(expr) with no NULL handling: just the row count.
            // Implement via a SUM over a synthesized all-ones Int64 column.
            // Inefficient but correct; will become a real path once NULL
            // handling lands.
            let ones: Vec<i64> = vec![1i64; n_rows];
            let scalar =
                reduce_host_slice::<i64>(ReduceOp::Sum, DataType::Int64, &ones)?;
            scalar_to_array(scalar, out_field.dtype)
        }
        AggregateExpr::Avg(expr) => {
            // AVG = SUM(expr) / COUNT(expr), both computed on the GPU then
            // divided on the host. The output is always Float64.
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;
            let sum_scalar =
                reduce_column_from_batch(ReduceOp::Sum, col_io, table_batch, n_rows)?;
            let sum_f64 = scalar_to_f64(sum_scalar)?;

            let ones: Vec<i64> = vec![1i64; n_rows];
            let count_scalar =
                reduce_host_slice::<i64>(ReduceOp::Sum, DataType::Int64, &ones)?;
            let count_f64 = scalar_to_f64(count_scalar)?;

            let avg = if count_f64 == 0.0 { 0.0 } else { sum_f64 / count_f64 };
            scalar_to_array(Scalar::F64(avg), out_field.dtype)
        }
    }
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
    }
}

/// Resolve `name` to its `(index, ColumnIO)` within `inputs`.
fn resolve_input<'a>(inputs: &'a [ColumnIO], name: &str) -> JavelinResult<&'a ColumnIO> {
    inputs
        .iter()
        .find(|c| c.name == name)
        .ok_or_else(|| {
            JavelinError::Plan(format!(
                "aggregate input column '{}' not found in plan inputs",
                name
            ))
        })
}

/// Extract the column name from a bare-column-ref expression. The first cut
/// of `execute_aggregate` requires every aggregate to be over a column ref
/// (the physical-plan lowering guarantees this when `pre` is `None`).
fn bare_column_name(expr: &Expr) -> JavelinResult<&str> {
    match expr {
        Expr::Column(name) => Ok(name.as_str()),
        Expr::Alias(inner, _) => bare_column_name(inner),
        _ => Err(JavelinError::Other(
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
) -> JavelinResult<Scalar> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        JavelinError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);

    // Sanity-check the dtype matches what the plan promised.
    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != col_io.dtype {
        return Err(JavelinError::Type(format!(
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
            let dev = GpuVec::from_buffer(primitive_to_gpu(pa)?);
            reduce_gpu_vec::<i32>(op, col_io.dtype, &dev, n_rows)
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            let dev = GpuVec::from_buffer(primitive_to_gpu(pa)?);
            reduce_gpu_vec::<i64>(op, col_io.dtype, &dev, n_rows)
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            let dev = GpuVec::from_buffer(primitive_to_gpu(pa)?);
            reduce_gpu_vec::<f32>(op, col_io.dtype, &dev, n_rows)
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            let dev = GpuVec::from_buffer(primitive_to_gpu(pa)?);
            reduce_gpu_vec::<f64>(op, col_io.dtype, &dev, n_rows)
        }
        DataType::Bool | DataType::Utf8 => Err(JavelinError::Type(format!(
            "aggregate input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// Upload a host slice, then run the standard GPU reduction over it. Used by
/// COUNT (which synthesizes an all-ones column on the host).
fn reduce_host_slice<T>(op: ReduceOp, dtype: DataType, host: &[T]) -> JavelinResult<Scalar>
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
) -> JavelinResult<Scalar>
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

    // Compile + load the kernel.
    let ptx = compile_reduction_kernel(op, dtype)?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(REDUCTION_KERNEL_ENTRY)?;

    // Assemble the kernel parameter list (input_ptr, output_ptr, n_rows).
    let mut input_ptr: CUdeviceptr = input.device_ptr();
    let mut output_ptr: CUdeviceptr = partials.device_ptr();

    let mut kernel_params: [*mut c_void; 3] = [
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut output_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
    ];

    let stream = CudaStream::null();
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

    // The `_` writes are immediately superseded; silence the lints by reading
    // them post-launch in case a future reordering relies on the values.
    let _ = input_ptr;
    let _ = output_ptr;

    // Download the per-block partials and finish the reduction on the host.
    let host_partials = partials.to_vec()?;
    drop(partials);
    T::finalize(op, dtype, &host_partials)
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
    fn finalize(op: ReduceOp, dtype: DataType, host: &[Self]) -> JavelinResult<Scalar>;
    /// Identity value (used when `n_rows == 0`).
    fn identity_scalar(op: ReduceOp, dtype: DataType) -> JavelinResult<Scalar>;
}

impl ReduceScalar for i32 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> JavelinResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0i32, i32::wrapping_add),
            ReduceOp::Min => host.iter().copied().fold(i32::MAX, i32::min),
            ReduceOp::Max => host.iter().copied().fold(i32::MIN, i32::max),
        };
        Ok(Scalar::I32(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> JavelinResult<Scalar> {
        Ok(Scalar::I32(match op {
            ReduceOp::Sum | ReduceOp::Count => 0,
            ReduceOp::Min => i32::MAX,
            ReduceOp::Max => i32::MIN,
        }))
    }
}

impl ReduceScalar for i64 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> JavelinResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0i64, i64::wrapping_add),
            ReduceOp::Min => host.iter().copied().fold(i64::MAX, i64::min),
            ReduceOp::Max => host.iter().copied().fold(i64::MIN, i64::max),
        };
        Ok(Scalar::I64(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> JavelinResult<Scalar> {
        Ok(Scalar::I64(match op {
            ReduceOp::Sum | ReduceOp::Count => 0,
            ReduceOp::Min => i64::MAX,
            ReduceOp::Max => i64::MIN,
        }))
    }
}

impl ReduceScalar for f32 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> JavelinResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0.0f32, |a, b| a + b),
            ReduceOp::Min => host.iter().copied().fold(f32::INFINITY, f32::min),
            ReduceOp::Max => host.iter().copied().fold(f32::NEG_INFINITY, f32::max),
        };
        Ok(Scalar::F32(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> JavelinResult<Scalar> {
        Ok(Scalar::F32(match op {
            ReduceOp::Sum | ReduceOp::Count => 0.0,
            ReduceOp::Min => f32::INFINITY,
            ReduceOp::Max => f32::NEG_INFINITY,
        }))
    }
}

impl ReduceScalar for f64 {
    fn finalize(op: ReduceOp, _dtype: DataType, host: &[Self]) -> JavelinResult<Scalar> {
        let acc = match op {
            ReduceOp::Sum | ReduceOp::Count => host.iter().copied().fold(0.0f64, |a, b| a + b),
            ReduceOp::Min => host.iter().copied().fold(f64::INFINITY, f64::min),
            ReduceOp::Max => host.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        };
        Ok(Scalar::F64(acc))
    }
    fn identity_scalar(op: ReduceOp, _dtype: DataType) -> JavelinResult<Scalar> {
        Ok(Scalar::F64(match op {
            ReduceOp::Sum | ReduceOp::Count => 0.0,
            ReduceOp::Min => f64::INFINITY,
            ReduceOp::Max => f64::NEG_INFINITY,
        }))
    }
}

/// Convert a `Scalar` into a single-element Arrow array of `out_dtype`.
/// Performs the small numeric cast (e.g. `i64 -> Float64` for AVG output).
fn scalar_to_array(scalar: Scalar, out_dtype: DataType) -> JavelinResult<ArrayRef> {
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

        (s, dt) => Err(JavelinError::Type(format!(
            "aggregate: cannot pack scalar {:?} into output dtype {:?}",
            s, dt
        ))),
    }
}

/// Cast a scalar to f64 (used by AVG).
fn scalar_to_f64(s: Scalar) -> JavelinResult<f64> {
    Ok(match s {
        Scalar::I32(v) => v as f64,
        Scalar::I64(v) => v as f64,
        Scalar::F32(v) => v as f64,
        Scalar::F64(v) => v,
    })
}

/// Build a `Type` error for a failed Arrow downcast on column `name`.
fn downcast_err(name: &str, expected: &str) -> JavelinError {
    JavelinError::Type(format!(
        "aggregate input column '{}' could not be downcast to {}",
        name, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType`. Mirror of `engine.rs`'s
/// private helper, copied here so we don't reach across module privacy.
fn arrow_dtype_to_plan(d: &ArrowDataType) -> JavelinResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        other => Err(JavelinError::Type(format!(
            "unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

/// Map our plan `DataType` to Arrow `DataType`.
fn plan_dtype_to_arrow(d: DataType) -> JavelinResult<ArrowDataType> {
    match d {
        DataType::Int32 => Ok(ArrowDataType::Int32),
        DataType::Int64 => Ok(ArrowDataType::Int64),
        DataType::Float32 => Ok(ArrowDataType::Float32),
        DataType::Float64 => Ok(ArrowDataType::Float64),
        DataType::Bool => Ok(ArrowDataType::Boolean),
        DataType::Utf8 => Ok(ArrowDataType::Utf8),
    }
}

/// Build an Arrow `Schema` from our plan `Schema` for the output RecordBatch.
fn plan_schema_to_arrow_schema(s: &Schema) -> JavelinResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}
