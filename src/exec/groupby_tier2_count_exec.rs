// SPDX-License-Identifier: Apache-2.0

//! **COUNT(*) at Tier 2.1** — high-cardinality `SELECT key, COUNT(*) FROM x
//! GROUP BY key` executor.
//!
//! Companion to the AVG-at-Tier-2.1 executor. The AVG path uses the same
//! COUNT kernel internally for its denominator; this executor exposes
//! that primitive on its own for queries that only ask for counts.
//!
//! ## Algorithm
//!
//! 1. Partition + scatter (keys only — no value column).
//! 2. Per-partition reduce via `partition_reduce_kernel_count` → per-group
//!    `u64` counts.
//! 3. Walk slots, push `(key, count)` into the output (skipping empty
//!    slots). Sort by key ASC.
//!
//! ## Scope (v0)
//!
//! - GROUP BY exactly one Int32 column
//! - Exactly one aggregate, `COUNT(*)` (which the planner represents as
//!   `AggregateExpr::Count(Expr::Literal(_))` or similar — we match by
//!   variant)
//! - `n_rows >= 256 K`
//! - `max(key) >= BLOCK_GROUPS` (Tier-1 single-aggregate path would
//!   handle the low-cardinality case if/when it grows a COUNT branch)
//! - `max(key) < 100 M`

use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{PatinaError, PatinaResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::partition_offsets;
use crate::jit::{
    partition_kernel, partition_reduce_kernel_count, scatter_kernel, CudaModule,
};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

/// Try the Tier-2.1 COUNT(*) fast path. `None` on any precondition miss.
pub fn try_execute(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<PatinaResult<RecordBatch>> {
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

    // Exactly one COUNT aggregate. We accept COUNT(<anything>) by
    // semantics: SQL COUNT(*) and COUNT(non_null_col) on a NOT NULL
    // schema produce the same result. The kernel doesn't read a value
    // column anyway, so the argument is decorative.
    match &aggregate.aggregates[0] {
        AggregateExpr::Count(_) => {}
        _ => return None,
    }

    // Single Int32 key.
    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;

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
    if n_groups_est <= partition_reduce_kernel_count::BLOCK_GROUPS {
        // Low cardinality — let the global-atomic path handle COUNT(*).
        // (We don't yet have a Tier-1 COUNT shortcut; not chasing it
        // until we see a workload that wants it.)
        return None;
    }
    if n_groups_est >= 100_000_000 {
        return None;
    }

    Some(execute_inner(plan, key_arr))
}

fn execute_inner(plan: &PhysicalPlan, key_arr: &Int32Array) -> PatinaResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;
    let keys_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice(key_arr.values())?;

    let num_partitions = partition_kernel::NUM_PARTITIONS;
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros(n_rows as usize)?;

    // Partition pass — CUDA-Oxide typed launch.
    // Kernel ABI: keys_ptr, pids_ptr, counts_ptr, n_rows
    {
        let ptx = partition_kernel::compile_partition_kernel()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(partition_kernel::KERNEL_ENTRY)?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);
        let stream = CudaStream::null();

        let view_keys = keys_gpu.view();
        let mut view_pids = partition_ids.view_mut();
        let mut view_counts = counts.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_output(&mut view_pids);
        args.push_output(&mut view_counts);
        args.push_scalar_u32(n_rows);

        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    // Offsets.
    let offsets: Vec<u32> = partition_offsets::compute_partition_offsets(&counts)?;
    let offsets_gpu: GpuVec<u32> = partition_offsets::upload_offsets(&offsets)?;

    // Scatter keys only. We still use the scatter kernel; it requires a
    // value column input, but for COUNT we have no meaningful value —
    // pass a zero-filled f64 buffer of the same length. The dummy
    // out_vals buffer is written but never read.
    let mut scatter_keys: GpuVec<i32> = GpuVec::<i32>::zeros(n_rows as usize)?;
    let dummy_vals_in: GpuVec<f64> = GpuVec::<f64>::zeros(n_rows as usize)?;
    let mut scatter_vals: GpuVec<f64> = GpuVec::<f64>::zeros(n_rows as usize)?;

    // Scatter pass — CUDA-Oxide typed launch.
    // Kernel ABI: keys, vals, pids, offsets, cursors, out_keys, out_vals, n_rows
    {
        let ptx = scatter_kernel::compile_scatter_kernel()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(scatter_kernel::KERNEL_ENTRY)?;
        let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);
        let stream = CudaStream::null();

        let view_keys = keys_gpu.view();
        let view_vals = dummy_vals_in.view();
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

        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }
    let _ = (dummy_vals_in, scatter_vals); // keep alive until end of launch

    // COUNT reduce pass.
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice(&offsets)?;
    let block_groups = partition_reduce_kernel_count::BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;
    let mut out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros(n_out_slots)?;
    let mut out_counts_gpu: GpuVec<u64> = GpuVec::<u64>::zeros(n_out_slots)?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;

    // CUDA-Oxide typed launch.
    // Kernel ABI: scatter_keys, offsets, out_keys, out_counts, out_set
    {
        let ptx = partition_reduce_kernel_count::compile_partition_reduce_kernel_count()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(partition_reduce_kernel_count::KERNEL_ENTRY)?;
        let stream = CudaStream::null();

        let view_keys = scatter_keys.view();
        let view_offsets = offsets_kp1_gpu.view();
        let mut view_ok = out_keys_gpu.view_mut();
        let mut view_oc = out_counts_gpu.view_mut();
        let mut view_os = out_set_gpu.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_input(&view_offsets);
        args.push_output(&mut view_ok);
        args.push_output(&mut view_oc);
        args.push_output(&mut view_os);

        launch_with_geometry(
            func,
            num_partitions,
            partition_reduce_kernel_count::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // Download + assemble.
    let host_out_keys: Vec<i32> = out_keys_gpu.to_vec()?;
    let host_out_counts: Vec<u64> = out_counts_gpu.to_vec()?;
    let host_out_set: Vec<u8> = out_set_gpu.to_vec()?;

    let mut pairs: Vec<(i32, i64)> = Vec::new();
    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] == 0 {
                continue;
            }
            let c = host_out_counts[idx];
            // The output schema for COUNT is Int64 (SQL semantics, the
            // planner widens it). Cast u64 → i64; in practice the count
            // is bounded by n_rows which fits in i64 fine for any input
            // size we care about.
            pairs.push((host_out_keys[idx], c as i64));
        }
    }
    pairs.sort_by_key(|(k, _)| *k);

    let keys_out: Vec<i32> = pairs.iter().map(|(k, _)| *k).collect();
    let counts_out: Vec<i64> = pairs.iter().map(|(_, c)| *c).collect();

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int32Array::from(keys_out)),
            Arc::new(Int64Array::from(counts_out)),
        ],
    )
    .map_err(|e| {
        PatinaError::Other(format!(
            "groupby_tier2_count_exec: failed to build RecordBatch: {e}"
        ))
    })
}

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

fn plan_schema_to_arrow_schema(s: &Schema) -> PatinaResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}
