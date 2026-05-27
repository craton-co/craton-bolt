// SPDX-License-Identifier: Apache-2.0

//! **AVG at Tier 2.1** — high-cardinality multi-AVG executor.
//!
//! The Tier-1 AVG executor (`groupby_shmem_avg_exec.rs`) handles
//! `n_groups ≤ 1024`. For higher-cardinality workloads (e.g. a future
//! `SELECT id3, AVG(v1), AVG(v2) FROM x GROUP BY id3` where id3 has 1 M
//! distinct values), the Tier-2 partitioning approach is the better
//! algorithm — exactly as it is for SUM (q3, q5).
//!
//! ## Algorithm
//!
//! 1. **Partition + scatter**: identical to `groupby_tier2_multi_
//!    orchestrator`. One partition kernel produces (partition_ids,
//!    counts); host-side prefix-sum gives the offsets; ONE atomic-claim
//!    pass writes the per-row `dest_idx` map + the scattered key column;
//!    N atomic-free indexed-scatter passes scatter each value column to
//!    the slots `dest_idx` specifies. This guarantees alignment between
//!    the key column and every value column by construction — independent
//!    of any `atomicAdd` ordering assumptions.
//! 2. **Pass 2 — SUMs**: one launch of `partition_reduce_kernel_multi`
//!    (n_vals = N) reduces each partition into N per-group SUMs.
//! 3. **Pass 2 — COUNT**: one launch of `partition_reduce_kernel_count`
//!    against the *same* scatter_keys buffer reduces each partition into
//!    per-group `u64` counts. No extra partitioning / scatter cost.
//! 4. **Compose**: walk the two output buffers in lockstep. For each
//!    populated slot push `(key, [sum_i / count_i for i in 0..N])` into
//!    the result. Slots with `count == 0` are omitted (SQL semantics for
//!    empty groups).
//!
//! Net cost vs the multi-SUM path: +1 reduce kernel launch (~10 ms) and
//! +8 MiB of D2H for the count output. Both well-amortised at any size
//! that selects this path in the first place.
//!
//! ## Scope (v0)
//!
//! - GROUP BY exactly one Int32 column
//! - 1..=`MAX_VALS` aggregates, ALL `AVG(<bare Float64 column>)`
//! - `n_rows >= 256 K` (matches `TIER2_MIN_ROWS`)
//! - `max(key) >= BLOCK_GROUPS` so Tier-1 AVG doesn't already win this
//! - `max(key) < 100 M` (Tier-2 dispatcher cap)

use std::sync::Arc;

use arrow_array::{Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::partition_offsets;
use crate::jit::{
    partition_kernel, partition_reduce_kernel_count, partition_reduce_kernel_multi,
    scatter_values_by_dest_idx_kernel, scatter_with_dest_idx_kernel, CudaModule,
};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

/// Try to execute `plan` against `batch` via the Tier-2.1 AVG fast path.
/// `None` on any miss — caller falls through to the next strategy.
pub fn try_execute(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<BoltResult<RecordBatch>> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        _ => return None,
    };
    if pre.is_some() {
        return None;
    }
    if aggregate.group_by.len() != 1 {
        return None;
    }
    let n_vals = aggregate.aggregates.len();
    if n_vals == 0 || n_vals > partition_reduce_kernel_multi::MAX_VALS as usize {
        return None;
    }

    // Single Int32 key.
    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    // All aggregates must be AVG(<bare Float64 column>).
    let mut val_col_names: Vec<&str> = Vec::with_capacity(n_vals);
    for agg in &aggregate.aggregates {
        let name = match agg {
            AggregateExpr::Avg(Expr::Column(n)) => n.as_str(),
            _ => return None,
        };
        val_col_names.push(name);
    }

    // Look up key + value arrays. Every value must be Float64.
    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    let mut val_arrs: Vec<&Float64Array> = Vec::with_capacity(n_vals);
    for name in &val_col_names {
        let arr = batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())?;
        if arr.len() != key_arr.len() {
            return None;
        }
        val_arrs.push(arr);
    }

    let n_rows = key_arr.len();
    if n_rows < 256 * 1024 {
        return None;
    }

    // n_groups estimator via max key. Reject Tier-1's territory and the
    // Tier-2 dispatcher's cap.
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
    if n_groups_est <= partition_reduce_kernel_multi::BLOCK_GROUPS {
        // Tier-1 AVG owns this.
        return None;
    }
    if n_groups_est >= 100_000_000 {
        return None;
    }

    Some(execute_inner(plan, key_arr, val_arrs, n_vals))
}

fn execute_inner(
    plan: &PhysicalPlan,
    key_arr: &Int32Array,
    val_arrs: Vec<&Float64Array>,
    n_vals: usize,
) -> BoltResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;

    // ---- Upload inputs --------------------------------------------------
    let keys_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice(key_arr.values())?;
    let mut vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for arr in &val_arrs {
        vals_gpu.push(GpuVec::<f64>::from_slice(arr.values())?);
    }

    let num_partitions = partition_kernel::NUM_PARTITIONS;

    // ---- Partition pass --------------------------------------------------
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

    // ---- Offsets ---------------------------------------------------------
    let offsets: Vec<u32> = partition_offsets::compute_partition_offsets(&counts)?;
    let offsets_gpu: GpuVec<u32> = partition_offsets::upload_offsets(&offsets)?;

    // ---- Scatter (deterministic dest_idx + indexed value passes) ---------
    //
    // Correctness note: the previous design called the atomic-claim scatter
    // kernel once per value column, relying on identical `atomicAdd`
    // orderings across launches to keep `(key, v1, v2, …)` aligned. That
    // ordering is NOT a CUDA contract, so a driver/scheduler change could
    // silently misalign `SUM(v_j)` with the wrong key.
    //
    // We now run the atomic-claim pass exactly ONCE
    // (`scatter_with_dest_idx_kernel`), capturing the per-row destination
    // slot in `dest_idx[n_rows]`. Each subsequent value column is scattered
    // by an atomic-free kernel that reads `dest_idx[i]` and writes
    // `out_vals[dest_idx[i]] = vals[i]`. Alignment is guaranteed by
    // construction. The COUNT reduce below also reads `scatter_keys` (the
    // claim pass's output), so the SUM-side / COUNT-side slot-population
    // agreement that the historical comment relied on is now a structural
    // property of the pipeline rather than an unsubstantiated assumption.
    let mut scatter_keys: GpuVec<i32> = GpuVec::<i32>::zeros(n_rows as usize)?;
    let mut scatter_vals: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        scatter_vals.push(GpuVec::<f64>::zeros(n_rows as usize)?);
    }
    let mut dest_idx: GpuVec<u32> = GpuVec::<u32>::zeros(n_rows as usize)?;

    // Atomic-claim pass.
    {
        let ptx =
            scatter_with_dest_idx_kernel::compile_scatter_with_dest_idx_kernel()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(scatter_with_dest_idx_kernel::KERNEL_ENTRY)?;

        let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);

        let view_keys = keys_gpu.view();
        let view_pids = partition_ids.view();
        let view_offsets = offsets_gpu.view();
        let mut view_cursors = cursors.view_mut();
        let mut view_sk = scatter_keys.view_mut();
        let mut view_di = dest_idx.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_input(&view_pids);
        args.push_input(&view_offsets);
        args.push_output(&mut view_cursors);
        args.push_output(&mut view_sk);
        args.push_output(&mut view_di);
        args.push_scalar_u32(n_rows);

        let stream = CudaStream::null();
        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    // Indexed value scatter — one launch per value column, no atomics.
    {
        let ptx =
            scatter_values_by_dest_idx_kernel::compile_scatter_values_by_dest_idx_kernel()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);

        for j in 0..n_vals {
            let func = module.function(scatter_values_by_dest_idx_kernel::KERNEL_ENTRY)?;

            let view_vals = vals_gpu[j].view();
            let view_dest = dest_idx.view();
            let mut view_sv = scatter_vals[j].view_mut();

            let mut args = KernelArgs::empty();
            args.push_input(&view_vals);
            args.push_input(&view_dest);
            args.push_output(&mut view_sv);
            args.push_scalar_u32(n_rows);

            let stream = CudaStream::null();
            launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
        }
    }

    // Reduce kernels need the FULL K+1 offsets buffer.
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice(&offsets)?;

    let block_groups = partition_reduce_kernel_multi::BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros(n_out_slots)?;
    let mut out_vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        out_vals_gpu.push(GpuVec::<f64>::zeros(n_out_slots)?);
    }
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;
    let mut out_counts_gpu: GpuVec<u64> = GpuVec::<u64>::zeros(n_out_slots)?;
    // The count kernel writes its own out_keys + out_set; we only consume
    // its out_counts. Output dedup keys / set come from the SUM reduce.
    let mut count_out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros(n_out_slots)?;
    let mut count_out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;

    // ---- Multi-SUM reduce -----------------------------------------------
    {
        let ptx = partition_reduce_kernel_multi::compile_partition_reduce_kernel_multi(
            n_vals as u32,
        )?;
        let module = CudaModule::from_ptx(&ptx)?;
        let entry = partition_reduce_kernel_multi::kernel_entry(n_vals as u32);
        let func = module.function(&entry)?;

        let view_sk = scatter_keys.view();
        let views_sv: Vec<_> = scatter_vals.iter().map(|g| g.view()).collect();
        let view_offsets = offsets_kp1_gpu.view();
        let mut view_ok = out_keys_gpu.view_mut();
        let mut views_ov: Vec<_> = out_vals_gpu.iter_mut().map(|g| g.view_mut()).collect();
        let mut view_os = out_set_gpu.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_sk);
        for v in &views_sv {
            args.push_input(v);
        }
        args.push_input(&view_offsets);
        args.push_output(&mut view_ok);
        for v in views_ov.iter_mut() {
            args.push_output(v);
        }
        args.push_output(&mut view_os);

        let stream = CudaStream::null();
        launch_with_geometry(
            func,
            num_partitions,
            partition_reduce_kernel_multi::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // ---- COUNT reduce ----------------------------------------------------
    {
        let ptx = partition_reduce_kernel_count::compile_partition_reduce_kernel_count()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(partition_reduce_kernel_count::KERNEL_ENTRY)?;

        let view_keys = scatter_keys.view();
        let view_offsets = offsets_kp1_gpu.view();
        let mut view_ok = count_out_keys_gpu.view_mut();
        let mut view_oc = out_counts_gpu.view_mut();
        let mut view_os = count_out_set_gpu.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_input(&view_offsets);
        args.push_output(&mut view_ok);
        args.push_output(&mut view_oc);
        args.push_output(&mut view_os);

        let stream = CudaStream::null();
        launch_with_geometry(
            func,
            num_partitions,
            partition_reduce_kernel_count::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // ---- Download everything --------------------------------------------
    //
    // The SUM reduce and COUNT reduce both consume `scatter_keys` (written
    // by the single atomic-claim pass above) and hash with the same slot
    // function, so for a given (partition, slot) both kernels write either
    // both populated or both empty, and both populate with the same key.
    // We use the SUM-side out_keys / out_set and the COUNT-side
    // out_counts. (Strictly speaking the count_out_keys / count_out_set
    // are redundant, but allocating them is cheaper than special-casing
    // the kernel signature.)
    let host_out_keys: Vec<i32> = out_keys_gpu.to_vec()?;
    let mut host_out_vals: Vec<Vec<f64>> = Vec::with_capacity(n_vals);
    for ov in &out_vals_gpu {
        host_out_vals.push(ov.to_vec()?);
    }
    let host_out_set: Vec<u8> = out_set_gpu.to_vec()?;
    let host_out_counts: Vec<u64> = out_counts_gpu.to_vec()?;

    // ---- Walk slots, divide host-side, build output ---------------------
    let mut out_keys_final: Vec<i32> = Vec::new();
    let mut out_avgs_final: Vec<Vec<f64>> =
        (0..n_vals).map(|_| Vec::new()).collect();

    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] == 0 {
                continue;
            }
            let c = host_out_counts[idx];
            if c == 0 {
                // Defensive: set==1 but count==0 means the two kernels
                // disagreed on slot population. With the deterministic
                // dest_idx scatter both kernels consume the same
                // scatter_keys buffer with the same slot function, so
                // this branch should be unreachable; we keep it as a
                // belt-and-suspenders skip rather than panicking, to
                // match SQL "no rows → no output" semantics.
                continue;
            }
            let cf = c as f64;
            out_keys_final.push(host_out_keys[idx]);
            for j in 0..n_vals {
                out_avgs_final[j].push(host_out_vals[j][idx] / cf);
            }
        }
    }

    // Sort by key (ASC) to match SQL canonical / what the equivalence
    // check expects.
    let mut idx: Vec<usize> = (0..out_keys_final.len()).collect();
    idx.sort_by_key(|&i| out_keys_final[i]);
    let sorted_keys: Vec<i32> = idx.iter().map(|&i| out_keys_final[i]).collect();
    let sorted_avgs: Vec<Vec<f64>> = (0..n_vals)
        .map(|j| idx.iter().map(|&i| out_avgs_final[j][i]).collect())
        .collect();

    // ---- Build the output RecordBatch -----------------------------------
    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    let mut cols: Vec<arrow_array::ArrayRef> = Vec::with_capacity(1 + n_vals);
    cols.push(Arc::new(Int32Array::from(sorted_keys)));
    for v in sorted_avgs {
        cols.push(Arc::new(Float64Array::from(v)));
    }
    RecordBatch::try_new(arrow_schema, cols).map_err(|e| {
        BoltError::Other(format!(
            "groupby_tier2_avg_exec: failed to build RecordBatch: {e}"
        ))
    })
}

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

fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}
