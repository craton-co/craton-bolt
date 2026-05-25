// SPDX-License-Identifier: Apache-2.0

//! Tier-2.1 **MIN / MAX over Float32 / Float64** executor.
//!
//! Sibling of [`crate::exec::groupby_tier2_minmax_exec`] (which handles
//! Int32 / Int64 values). The float variant has to route through a
//! different kernel — [`crate::jit::partition_reduce_kernel_minmax_float`]
//! — because PTX has no native `atom.shared.{min,max}.f{32,64}` on
//! sm_70. The float kernel emits an `atom.shared.cas.b{32,64}` retry
//! loop instead.
//!
//! v0 supports Float64 only. Float32 promotion is a one-line addition
//! once a workload demands it; the kernel handles both widths already.

use std::sync::Arc;

use arrow_array::{Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::partition_offsets;
use crate::jit::partition_reduce_kernel_minmax::MinMaxOp;
use crate::jit::partition_reduce_kernel_minmax_float::{
    compile_partition_reduce_kernel_minmax_float, kernel_entry as minmax_float_entry,
    FloatDtype, BLOCK_GROUPS, BLOCK_THREADS as REDUCE_BLOCK_THREADS,
};
use crate::jit::{partition_kernel, scatter_kernel, CudaModule};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

pub fn try_execute(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<JavelinResult<RecordBatch>> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        _ => return None,
    };
    if pre.is_some() {
        return None;
    }
    if aggregate.group_by.len() != 1 || aggregate.aggregates.len() != 1 {
        return None;
    }

    let (op, val_col_name) = match &aggregate.aggregates[0] {
        AggregateExpr::Min(Expr::Column(n)) => (MinMaxOp::Min, n.as_str()),
        AggregateExpr::Max(Expr::Column(n)) => (MinMaxOp::Max, n.as_str()),
        _ => return None,
    };

    // Single Int32 key.
    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };
    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;

    // Value: Float64 only in v0. Float32 promotion → Float64 host-side
    // is the obvious extension; defer until a workload asks.
    let val_col = batch.column_by_name(val_col_name)?;
    let float_dtype = match val_col.data_type() {
        ArrowDataType::Float64 => FloatDtype::Float64,
        _ => return None,
    };

    if key_arr.len() != val_col.len() {
        return None;
    }
    let n_rows = key_arr.len();
    if n_rows < 256 * 1024 {
        return None;
    }

    let mut max_key: i32 = -1;
    for &k in key_arr.values() {
        if k < 0 {
            return None;
        }
        if k > max_key {
            max_key = k;
        }
    }
    if max_key < 0 {
        return None;
    }
    let n_groups_est = (max_key as u32).saturating_add(1);
    if n_groups_est <= BLOCK_GROUPS {
        // Tier-1 MIN/MAX (integer-only today) wouldn't catch float
        // anyway; let the global-atomic fallback handle low-card
        // float min/max until a Tier-1 float path lands.
        return None;
    }
    if n_groups_est >= 100_000_000 {
        return None;
    }

    Some(execute_inner(plan, key_arr, val_col, op, float_dtype))
}

fn execute_inner(
    plan: &PhysicalPlan,
    key_arr: &Int32Array,
    val_col: &dyn arrow_array::Array,
    op: MinMaxOp,
    float_dtype: FloatDtype,
) -> JavelinResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;
    let keys_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice(key_arr.values())?;

    // Upload values. Float64 path only for v0.
    let val_arr = val_col
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| JavelinError::Other("expected Float64Array".into()))?;
    let vals_gpu: GpuVec<f64> = GpuVec::<f64>::from_slice(val_arr.values())?;

    let num_partitions = partition_kernel::NUM_PARTITIONS;

    // --- Partition pass ---
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros(n_rows as usize)?;
    {
        let ptx = partition_kernel::compile_partition_kernel()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(partition_kernel::KERNEL_ENTRY)?;

        let view_keys = keys_gpu.view();
        let mut view_pids = partition_ids.view_mut();
        let mut view_counts = counts.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_output(&mut view_pids);
        args.push_output(&mut view_counts);
        args.push_scalar_u32(n_rows);

        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);
        let stream = CudaStream::null();
        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    let offsets: Vec<u32> = partition_offsets::compute_partition_offsets(&counts)?;
    let offsets_gpu: GpuVec<u32> = partition_offsets::upload_offsets(&offsets)?;

    // --- Scatter (f64 vals — no conversion needed) ---
    let mut scatter_keys: GpuVec<i32> = GpuVec::<i32>::zeros(n_rows as usize)?;
    let mut scatter_vals: GpuVec<f64> = GpuVec::<f64>::zeros(n_rows as usize)?;
    {
        let ptx = scatter_kernel::compile_scatter_kernel()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(scatter_kernel::KERNEL_ENTRY)?;
        let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;

        let view_keys = keys_gpu.view();
        let view_vals = vals_gpu.view();
        let view_pids = partition_ids.view();
        let view_offsets = offsets_gpu.view();
        let mut view_cursors = cursors.view_mut();
        let mut view_sk = scatter_keys.view_mut();
        let mut view_sv = scatter_vals.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_input(&view_vals);
        args.push_input(&view_pids);
        args.push_input(&view_offsets);
        args.push_output(&mut view_cursors);
        args.push_output(&mut view_sk);
        args.push_output(&mut view_sv);
        args.push_scalar_u32(n_rows);

        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);
        let stream = CudaStream::null();
        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    // --- Reduce (CAS-loop float MIN/MAX) ---
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice(&offsets)?;
    let block_groups = BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros(n_out_slots)?;
    let mut out_vals_gpu: GpuVec<f64> = GpuVec::<f64>::zeros(n_out_slots)?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;

    {
        let ptx =
            compile_partition_reduce_kernel_minmax_float(op, float_dtype)?;
        let module = CudaModule::from_ptx(&ptx)?;
        let entry = minmax_float_entry(op, float_dtype);
        let func = module.function(&entry)?;

        let view_pk = scatter_keys.view();
        let view_pv = scatter_vals.view();
        let view_po = offsets_kp1_gpu.view();
        let mut view_ok = out_keys_gpu.view_mut();
        let mut view_ov = out_vals_gpu.view_mut();
        let mut view_os = out_set_gpu.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_pk);
        args.push_input(&view_pv);
        args.push_input(&view_po);
        args.push_output(&mut view_ok);
        args.push_output(&mut view_ov);
        args.push_output(&mut view_os);

        let stream = CudaStream::null();
        launch_with_geometry(
            func,
            num_partitions,
            REDUCE_BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    let host_out_keys: Vec<i32> = out_keys_gpu.to_vec()?;
    let host_out_vals: Vec<f64> = out_vals_gpu.to_vec()?;
    let host_out_set: Vec<u8> = out_set_gpu.to_vec()?;

    let mut pairs: Vec<(i32, f64)> = Vec::new();
    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] != 0 {
                pairs.push((host_out_keys[idx], host_out_vals[idx]));
            }
        }
    }
    pairs.sort_by_key(|(k, _)| *k);
    let keys: Vec<i32> = pairs.iter().map(|(k, _)| *k).collect();
    let vals: Vec<f64> = pairs.iter().map(|(_, v)| *v).collect();

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!(),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int32Array::from(keys)),
            Arc::new(Float64Array::from(vals)),
        ],
    )
    .map_err(|e| {
        JavelinError::Other(format!(
            "groupby_tier2_minmax_float_exec: build error: {e}"
        ))
    })
}

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

fn plan_schema_to_arrow_schema(s: &Schema) -> JavelinResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}
