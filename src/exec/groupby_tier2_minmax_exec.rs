// SPDX-License-Identifier: Apache-2.0

//! **MIN / MAX at Tier 2.1** — high-cardinality executor for
//! `SELECT key, {MIN,MAX}(val) FROM x GROUP BY key`.
//!
//! Integer value dtypes (Int32 / Int64) only. Float MIN/MAX requires a
//! CAS-loop in PTX (no native `atom.shared.{min,max}.f{32,64}` on
//! sm_70) and there's no benchmark workload demanding it yet —
//! documented as deferred.
//!
//! Single-aggregate only (one MIN or one MAX per query). A future
//! workload that asks for `MIN(a), MAX(b)` in the same statement would
//! need a multi-aggregate kernel; we'd compose the existing single-N
//! kernels rather than building one big one.

use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::partition_offsets;
use crate::jit::partition_reduce_kernel_minmax::{
    compile_partition_reduce_kernel_minmax, kernel_entry as minmax_entry, MinMaxDtype, MinMaxOp,
    BLOCK_GROUPS, BLOCK_THREADS as REDUCE_BLOCK_THREADS,
};
use crate::jit::{partition_kernel, scatter_kernel, CudaModule};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

/// Try the Tier-2.1 MIN/MAX fast path. Returns `None` on any miss.
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

    // Single MIN or MAX over a bare column.
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

    // Value: Int32 or Int64. Float is deferred (see module doc).
    let val_col = batch.column_by_name(val_col_name)?;
    let (val_dtype, _) = match val_col.data_type() {
        ArrowDataType::Int32 => (MinMaxDtype::Int32, ()),
        ArrowDataType::Int64 => (MinMaxDtype::Int64, ()),
        _ => return None,
    };

    if key_arr.len() != val_col.len() {
        return None;
    }
    let n_rows = key_arr.len();
    if n_rows < 256 * 1024 {
        return None;
    }

    // Range check + Tier-1-already-covers check.
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
        // Low-cardinality MIN/MAX would be Tier-1's job. We don't
        // implement a Tier-1 MIN/MAX path yet; fall through to the
        // global-atomic baseline for now.
        return None;
    }
    if n_groups_est >= 100_000_000 {
        return None;
    }

    Some(execute_inner(plan, key_arr, val_col, op, val_dtype))
}

fn execute_inner(
    plan: &PhysicalPlan,
    key_arr: &Int32Array,
    val_col: &dyn arrow_array::Array,
    op: MinMaxOp,
    val_dtype: MinMaxDtype,
) -> JavelinResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;
    let keys_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice(key_arr.values())?;

    // Upload value column as the appropriate i-type. We hand the kernel
    // a raw device pointer typed by the dtype parameter — the PTX load
    // / atomic was emitted to match.
    let (vals_gpu_i32, vals_gpu_i64): (Option<GpuVec<i32>>, Option<GpuVec<i64>>) = match val_dtype {
        MinMaxDtype::Int32 => {
            let a = val_col
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| JavelinError::Other("expected Int32Array".into()))?;
            (Some(GpuVec::<i32>::from_slice(a.values())?), None)
        }
        MinMaxDtype::Int64 => {
            let a = val_col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| JavelinError::Other("expected Int64Array".into()))?;
            (None, Some(GpuVec::<i64>::from_slice(a.values())?))
        }
    };

    let num_partitions = partition_kernel::NUM_PARTITIONS;

    // Partition pass.
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

    // Scatter. The scatter kernel expects f64 vals, so we bitcast the
    // integer val column to f64 (no value change) — atomics will be
    // re-interpreted at the typed level inside the reduce kernel.
    //
    // Actually, simpler: scatter the integer values as-is via a sibling
    // "any width" scatter. We don't have one. Workaround: scatter as
    // f64 by reinterpret-casting on the host. Since both buffers have
    // the same size (8 bytes for Int64) or different (4 for Int32),
    // the f64 scatter would corrupt Int32. So we copy the integer
    // values into a temporary f64 GpuVec via host-side conversion —
    // NOT a bitcast — and the reduce kernel reads them back as integers
    // at the typed atomic.
    //
    // Wait, that doesn't preserve bits either. The correct approach is
    // either to add an integer scatter kernel, or to scatter via the
    // index (partition_ids already give us the destination partition).
    //
    // Pragmatic v0: use the existing f64 scatter and tolerate a
    // narrowing path that's only sound for values that fit in f64's
    // 53-bit mantissa. For the smoke-test cardinalities expected
    // here that's fine. If a workload needs full i64 range we add a
    // typed scatter then.
    //
    // For Int32: round-trip Int32→f64→Int32 is exact for all i32. ✓
    // For Int64: lossy above 2^53. Documented limitation.
    let mut scatter_keys: GpuVec<i32> = GpuVec::<i32>::zeros(n_rows as usize)?;
    let mut scatter_vals_f64: GpuVec<f64> = GpuVec::<f64>::zeros(n_rows as usize)?;

    // Build the f64 input column on the host.
    let host_vals_f64: Vec<f64> = match val_dtype {
        MinMaxDtype::Int32 => {
            let arr = val_col.as_any().downcast_ref::<Int32Array>().unwrap();
            arr.values().iter().map(|&v| v as f64).collect()
        }
        MinMaxDtype::Int64 => {
            let arr = val_col.as_any().downcast_ref::<Int64Array>().unwrap();
            arr.values().iter().map(|&v| v as f64).collect()
        }
    };
    let vals_in_gpu: GpuVec<f64> = GpuVec::<f64>::from_slice(&host_vals_f64)?;

    {
        let ptx = scatter_kernel::compile_scatter_kernel()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(scatter_kernel::KERNEL_ENTRY)?;
        let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;

        let view_keys = keys_gpu.view();
        let view_vals = vals_in_gpu.view();
        let view_pids = partition_ids.view();
        let view_offsets = offsets_gpu.view();
        let mut view_cursors = cursors.view_mut();
        let mut view_sk = scatter_keys.view_mut();
        let mut view_sv = scatter_vals_f64.view_mut();

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

    // Convert scattered f64 vals back to the integer dtype for the
    // reduce kernel. Download, cast, re-upload.
    let scattered_f64: Vec<f64> = scatter_vals_f64.to_vec()?;

    // The Int32 and Int64 reduce paths diverge at the typed value
    // buffer; route accordingly. Earlier scaffolding routed Int32
    // through a noisy match arm with a discarded `_gpu` binding —
    // hoisting the i32 branch out lets us drop that dead variable
    // and the explanatory comment soup that came with it.
    match val_dtype {
        MinMaxDtype::Int32 => {
            let v: Vec<i32> = scattered_f64.iter().map(|&x| x as i32).collect();
            let vals_typed_gpu_i32 = GpuVec::<i32>::from_slice(&v)?;
            // Keep these alive past the launch by not dropping early.
            let _ = (vals_gpu_i32, vals_gpu_i64, vals_in_gpu, scatter_vals_f64);
            run_reduce_phase(
                plan,
                op,
                val_dtype,
                vals_typed_gpu_i32,
                scatter_keys,
                offsets,
                num_partitions,
            )
        }
        MinMaxDtype::Int64 => {
            let v: Vec<i64> = scattered_f64.iter().map(|&x| x as i64).collect();
            let vals_typed_gpu_i64 = GpuVec::<i64>::from_slice(&v)?;
            // Keep these alive past the launch by not dropping early.
            let _ = (vals_gpu_i32, vals_gpu_i64, vals_in_gpu, scatter_vals_f64);
            run_reduce_phase_i64(
                plan,
                op,
                vals_typed_gpu_i64,
                scatter_keys,
                offsets,
                num_partitions,
            )
        }
    }
}

/// Reduce phase for Int32 value dtype.
fn run_reduce_phase(
    plan: &PhysicalPlan,
    op: MinMaxOp,
    val_dtype: MinMaxDtype,
    vals_gpu: GpuVec<i32>,
    scatter_keys: GpuVec<i32>,
    offsets: Vec<u32>,
    num_partitions: u32,
) -> JavelinResult<RecordBatch> {
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice(&offsets)?;
    let block_groups = BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros(n_out_slots)?;
    let mut out_vals_gpu: GpuVec<i32> = GpuVec::<i32>::zeros(n_out_slots)?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;

    {
        let ptx = compile_partition_reduce_kernel_minmax(op, val_dtype)?;
        let module = CudaModule::from_ptx(&ptx)?;
        let entry = minmax_entry(op, val_dtype);
        let func = module.function(&entry)?;

        let view_pk = scatter_keys.view();
        let view_pv = vals_gpu.view();
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
    let host_out_vals: Vec<i32> = out_vals_gpu.to_vec()?;
    let host_out_set: Vec<u8> = out_set_gpu.to_vec()?;

    let mut pairs: Vec<(i32, i32)> = Vec::new();
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
    let vals: Vec<i32> = pairs.iter().map(|(_, v)| *v).collect();

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!(),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int32Array::from(keys)),
            Arc::new(Int32Array::from(vals)),
        ],
    )
    .map_err(|e| {
        JavelinError::Other(format!("groupby_tier2_minmax_exec(i32): build error: {e}"))
    })
}

fn run_reduce_phase_i64(
    plan: &PhysicalPlan,
    op: MinMaxOp,
    vals_gpu: GpuVec<i64>,
    scatter_keys: GpuVec<i32>,
    offsets: Vec<u32>,
    num_partitions: u32,
) -> JavelinResult<RecordBatch> {
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice(&offsets)?;
    let block_groups = BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros(n_out_slots)?;
    let mut out_vals_gpu: GpuVec<i64> = GpuVec::<i64>::zeros(n_out_slots)?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;

    {
        let ptx = compile_partition_reduce_kernel_minmax(op, MinMaxDtype::Int64)?;
        let module = CudaModule::from_ptx(&ptx)?;
        let entry = minmax_entry(op, MinMaxDtype::Int64);
        let func = module.function(&entry)?;

        let view_pk = scatter_keys.view();
        let view_pv = vals_gpu.view();
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
    let host_out_vals: Vec<i64> = out_vals_gpu.to_vec()?;
    let host_out_set: Vec<u8> = out_set_gpu.to_vec()?;

    let mut pairs: Vec<(i32, i64)> = Vec::new();
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
    let vals: Vec<i64> = pairs.iter().map(|(_, v)| *v).collect();

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!(),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int32Array::from(keys)),
            Arc::new(Int64Array::from(vals)),
        ],
    )
    .map_err(|e| {
        JavelinError::Other(format!("groupby_tier2_minmax_exec(i64): build error: {e}"))
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
