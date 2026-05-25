// SPDX-License-Identifier: Apache-2.0

//! **Two-key multi-AVG at Tier 2.1** — high-cardinality
//! `SELECT a, b, AVG(v1), AVG(v2), … FROM x GROUP BY a, b` executor.
//!
//! Mirror of [`crate::exec::groupby_tier2_avg_exec`] adapted for the
//! i64-packed-two-key path. Both group-by columns are Int32 and packed
//! losslessly into a single i64 host-side (matching the convention in
//! `groupby.rs::pack_keys`); the on-device chain then treats them as a
//! single dense key column.
//!
//! ## Algorithm
//!
//! 1. Pack `(k1, k2)` → `i64` host-side.
//! 2. **Partition + scatter**: one i64-key partition pass; N scatter
//!    launches (one per value column, all sharing one i64 key column).
//! 3. **Pass 2 — SUMs**: one launch of
//!    `partition_reduce_kernel_multi_i64` (n_vals = N) reduces each
//!    partition into N per-group SUMs.
//! 4. **Pass 2 — COUNT**: one launch of
//!    `partition_reduce_kernel_count_i64` against the same scatter_keys
//!    buffer reduces each partition into per-group `u64` counts.
//! 5. **Compose**: walk the two output buffers in lockstep, divide
//!    host-side, unpack `(key_hi, key_lo)`, push results out.
//!
//! ## Scope (v0)
//!
//! - GROUP BY exactly two Int32 columns
//! - 1..=`MAX_VALS` aggregates, ALL `AVG(<bare Float64 column>)`
//! - `n_rows >= 256 K`
//! - Combined (packed) key cardinality < 100 M (Tier-2 dispatcher cap).
//!   The lower-bound `BLOCK_GROUPS` check is implicit at this row
//!   threshold for two-key plans — no Tier-1 two-key AVG exists.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{PatinaError, PatinaResult};
use crate::exec::launch::CudaStream;
use crate::exec::partition_offsets;
use crate::jit::{
    partition_kernel_i64, partition_reduce_kernel_count_i64,
    partition_reduce_kernel_multi_i64, scatter_kernel_i64, CudaModule,
};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

/// Try the two-key Tier-2.1 multi-AVG fast path. `None` on any
/// precondition miss so the caller falls through to the next strategy.
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
    if aggregate.group_by.len() != 2 {
        return None;
    }
    let n_vals = aggregate.aggregates.len();
    if n_vals == 0 || n_vals > partition_reduce_kernel_multi_i64::MAX_VALS as usize {
        return None;
    }

    // Both keys must be Int32.
    let k1_io = aggregate.inputs.get(aggregate.group_by[0])?;
    let k2_io = aggregate.inputs.get(aggregate.group_by[1])?;
    if k1_io.dtype != DataType::Int32 || k2_io.dtype != DataType::Int32 {
        return None;
    }

    // Every aggregate must be AVG(bare Float64 column).
    let mut val_col_names: Vec<&str> = Vec::with_capacity(n_vals);
    for agg in &aggregate.aggregates {
        match agg {
            AggregateExpr::Avg(Expr::Column(n)) => val_col_names.push(n.as_str()),
            _ => return None,
        }
    }

    let k1 = batch
        .column_by_name(&k1_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    let k2 = batch
        .column_by_name(&k2_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    if k1.len() != k2.len() {
        return None;
    }
    let mut val_arrs: Vec<&Float64Array> = Vec::with_capacity(n_vals);
    for name in &val_col_names {
        let a = batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())?;
        if a.len() != k1.len() {
            return None;
        }
        val_arrs.push(a);
    }

    let n_rows = k1.len();
    if n_rows < 256 * 1024 {
        return None;
    }
    if n_rows >= 100_000_000 {
        return None;
    }

    Some(execute_inner(plan, k1, k2, val_arrs, n_vals))
}

fn execute_inner(
    plan: &PhysicalPlan,
    k1: &Int32Array,
    k2: &Int32Array,
    val_arrs: Vec<&Float64Array>,
    n_vals: usize,
) -> PatinaResult<RecordBatch> {
    let n_rows = k1.len() as u32;

    // ---- Host-side pack ------------------------------------------------
    // `(k1 << 32) | (k2 & 0xFFFF_FFFF)`. Matches `groupby.rs::pack_keys`
    // for the (Int32, Int32) shape.
    let packed: Vec<i64> = k1
        .values()
        .iter()
        .zip(k2.values().iter())
        .map(|(&a, &b)| ((a as u32 as u64) << 32 | (b as u32 as u64)) as i64)
        .collect();
    let keys_gpu: GpuVec<i64> = GpuVec::<i64>::from_slice(&packed)?;

    let mut vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for arr in &val_arrs {
        vals_gpu.push(GpuVec::<f64>::from_slice(arr.values())?);
    }

    let num_partitions = partition_kernel_i64::NUM_PARTITIONS;

    // ---- Partition pass (i64) ------------------------------------------
    let counts: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;
    let partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros(n_rows as usize)?;
    {
        let ptx = partition_kernel_i64::compile_partition_kernel_i64()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(partition_kernel_i64::KERNEL_ENTRY)?;
        let mut keys_ptr = keys_gpu.device_ptr();
        let mut pids_ptr = partition_ids.device_ptr();
        let mut counts_ptr = counts.device_ptr();
        let mut n_rows_u32 = n_rows;
        let mut params: [*mut c_void; 4] = [
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut pids_ptr as *mut CUdeviceptr as *mut c_void,
            &mut counts_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
        ];
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);
        let stream = CudaStream::null();
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                func.raw(),
                grid,
                1,
                1,
                BLOCK_THREADS,
                1,
                1,
                0,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        stream.synchronize()?;
    }

    // ---- Offsets -------------------------------------------------------
    let offsets: Vec<u32> = partition_offsets::compute_partition_offsets(&counts)?;
    let offsets_gpu: GpuVec<u32> = partition_offsets::upload_offsets(&offsets)?;

    // ---- Scatter (keys shared + N value passes) ------------------------
    let scatter_keys: GpuVec<i64> = GpuVec::<i64>::zeros(n_rows as usize)?;
    let mut scatter_vals: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        scatter_vals.push(GpuVec::<f64>::zeros(n_rows as usize)?);
    }
    {
        let ptx = scatter_kernel_i64::compile_scatter_kernel_i64()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(scatter_kernel_i64::KERNEL_ENTRY)?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);

        for j in 0..n_vals {
            let cursors: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;
            let mut keys_ptr = keys_gpu.device_ptr();
            let mut vals_ptr = vals_gpu[j].device_ptr();
            let mut pids_ptr = partition_ids.device_ptr();
            let mut offsets_ptr = offsets_gpu.device_ptr();
            let mut cursors_ptr = cursors.device_ptr();
            let mut sk_ptr = scatter_keys.device_ptr();
            let mut sv_ptr = scatter_vals[j].device_ptr();
            let mut n_rows_u32 = n_rows;
            let mut params: [*mut c_void; 8] = [
                &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
                &mut vals_ptr as *mut CUdeviceptr as *mut c_void,
                &mut pids_ptr as *mut CUdeviceptr as *mut c_void,
                &mut offsets_ptr as *mut CUdeviceptr as *mut c_void,
                &mut cursors_ptr as *mut CUdeviceptr as *mut c_void,
                &mut sk_ptr as *mut CUdeviceptr as *mut c_void,
                &mut sv_ptr as *mut CUdeviceptr as *mut c_void,
                &mut n_rows_u32 as *mut u32 as *mut c_void,
            ];
            let stream = CudaStream::null();
            unsafe {
                cuda_sys::check(cuda_sys::cuLaunchKernel(
                    func.raw(),
                    grid,
                    1,
                    1,
                    BLOCK_THREADS,
                    1,
                    1,
                    0,
                    stream.raw(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ))?;
            }
            stream.synchronize()?;
        }
    }

    // Reduce kernels need the FULL K+1 offsets buffer.
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice(&offsets)?;

    let block_groups = partition_reduce_kernel_multi_i64::BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros(n_out_slots)?;
    let mut out_vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        out_vals_gpu.push(GpuVec::<f64>::zeros(n_out_slots)?);
    }
    let out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;
    let out_counts_gpu: GpuVec<u64> = GpuVec::<u64>::zeros(n_out_slots)?;
    // COUNT kernel writes its own out_keys / out_set buffers; we only
    // consume its out_counts. SUM-side out_keys/out_set are the canonical
    // dedup output. (Strictly speaking the count_out_keys/set are
    // redundant; allocating them is cheaper than special-casing the
    // kernel signature.)
    let count_out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros(n_out_slots)?;
    let count_out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;

    // ---- Multi-SUM reduce (i64-key) ------------------------------------
    {
        let ptx = partition_reduce_kernel_multi_i64::compile_partition_reduce_kernel_multi_i64(
            n_vals as u32,
        )?;
        let module = CudaModule::from_ptx(&ptx)?;
        let entry = partition_reduce_kernel_multi_i64::kernel_entry(n_vals as u32);
        let func = module.function(&entry)?;

        let mut storage: Vec<CUdeviceptr> = Vec::with_capacity(4 + 2 * n_vals);
        storage.push(scatter_keys.device_ptr());
        for sv in &scatter_vals {
            storage.push(sv.device_ptr());
        }
        storage.push(offsets_kp1_gpu.device_ptr());
        storage.push(out_keys_gpu.device_ptr());
        for ov in &out_vals_gpu {
            storage.push(ov.device_ptr());
        }
        storage.push(out_set_gpu.device_ptr());

        let mut params: Vec<*mut c_void> = storage
            .iter_mut()
            .map(|p| p as *mut CUdeviceptr as *mut c_void)
            .collect();
        let stream = CudaStream::null();
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                func.raw(),
                num_partitions,
                1,
                1,
                partition_reduce_kernel_multi_i64::BLOCK_THREADS,
                1,
                1,
                0,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        stream.synchronize()?;
    }

    // ---- COUNT reduce (i64-key) ----------------------------------------
    {
        let ptx = partition_reduce_kernel_count_i64::compile_partition_reduce_kernel_count_i64()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(partition_reduce_kernel_count_i64::KERNEL_ENTRY)?;
        let mut keys_ptr = scatter_keys.device_ptr();
        let mut offsets_ptr = offsets_kp1_gpu.device_ptr();
        let mut ok_ptr = count_out_keys_gpu.device_ptr();
        let mut oc_ptr = out_counts_gpu.device_ptr();
        let mut os_ptr = count_out_set_gpu.device_ptr();
        let mut params: [*mut c_void; 5] = [
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut offsets_ptr as *mut CUdeviceptr as *mut c_void,
            &mut ok_ptr as *mut CUdeviceptr as *mut c_void,
            &mut oc_ptr as *mut CUdeviceptr as *mut c_void,
            &mut os_ptr as *mut CUdeviceptr as *mut c_void,
        ];
        let stream = CudaStream::null();
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                func.raw(),
                num_partitions,
                1,
                1,
                partition_reduce_kernel_count_i64::BLOCK_THREADS,
                1,
                1,
                0,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        stream.synchronize()?;
    }

    // ---- Download everything -------------------------------------------
    let host_out_keys: Vec<i64> = out_keys_gpu.to_vec()?;
    let mut host_out_vals: Vec<Vec<f64>> = Vec::with_capacity(n_vals);
    for ov in &out_vals_gpu {
        host_out_vals.push(ov.to_vec()?);
    }
    let host_out_set: Vec<u8> = out_set_gpu.to_vec()?;
    let host_out_counts: Vec<u64> = out_counts_gpu.to_vec()?;

    // ---- Walk slots, divide host-side, sort by packed key --------------
    let mut rows: Vec<(i64, Vec<f64>)> = Vec::new();
    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] == 0 {
                continue;
            }
            let c = host_out_counts[idx];
            if c == 0 {
                // Defensive: set==1 but count==0 — skip to match SQL
                // "no rows → no output" semantics.
                continue;
            }
            let cf = c as f64;
            let avgs: Vec<f64> = (0..n_vals)
                .map(|j| host_out_vals[j][idx] / cf)
                .collect();
            rows.push((host_out_keys[idx], avgs));
        }
    }
    rows.sort_by_key(|(k, _)| *k);

    // ---- Unpack and build output ---------------------------------------
    let mut out_k1: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_k2: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_avgs: Vec<Vec<f64>> = (0..n_vals)
        .map(|_| Vec::with_capacity(rows.len()))
        .collect();
    for (k, avgs) in rows {
        let u = k as u64;
        out_k1.push((u >> 32) as u32 as i32);
        out_k2.push((u & 0xFFFF_FFFF) as u32 as i32);
        for (j, a) in avgs.into_iter().enumerate() {
            out_avgs[j].push(a);
        }
    }

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(2 + n_vals);
    cols.push(Arc::new(Int32Array::from(out_k1)));
    cols.push(Arc::new(Int32Array::from(out_k2)));
    for v in out_avgs {
        cols.push(Arc::new(Float64Array::from(v)));
    }
    RecordBatch::try_new(arrow_schema, cols).map_err(|e| {
        PatinaError::Other(format!(
            "groupby_tier2_twokey_avg_exec: failed to build RecordBatch: {e}"
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
