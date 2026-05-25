// SPDX-License-Identifier: Apache-2.0

//! **Two-key COUNT(*) at Tier 2.1** — high-cardinality
//! `SELECT a, b, COUNT(*) FROM x GROUP BY a, b` executor.
//!
//! Mirror of [`crate::exec::groupby_tier2_count_exec`] adapted for the
//! i64-packed-two-key path. Both group-by columns are Int32 and packed
//! losslessly into a single i64 host-side (matching the convention in
//! `groupby.rs::pack_keys`); the on-device chain then treats them as a
//! single dense key column.
//!
//! ## Algorithm
//!
//! 1. Pack `(k1, k2)` → `i64` host-side via `(k1 << 32) | (k2 & 0xFFFF_FFFF)`.
//! 2. Partition + scatter (keys only — no value column).
//! 3. Per-partition reduce via `partition_reduce_kernel_count_i64` →
//!    per-group `u64` counts.
//! 4. Walk slots, unpack `(key_hi, key_lo)`, push `(key1, key2, count)`
//!    into the output (skipping empty slots). Sort by `key_i64` ASC so
//!    the ordering is deterministic and matches the sibling SUM/MULTI
//!    two-key executors.
//!
//! ## Scope (v0)
//!
//! - GROUP BY exactly two Int32 columns
//! - Exactly one aggregate, `COUNT(*)` (any argument — the kernel
//!   ignores it, mirroring the single-key COUNT executor)
//! - `n_rows >= 256 K`
//! - Combined (packed) key cardinality estimator > `BLOCK_GROUPS`
//!   (Tier-1 territory) and < 100 M (Tier-2 dispatcher cap). The
//!   single-key path estimates via `max(key)`; for the two-key path we
//!   conservatively use `n_rows` as an upper bound on n_groups (the
//!   true cardinality is at most n_rows).

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};
use crate::exec::launch::CudaStream;
use crate::exec::partition_offsets;
use crate::jit::{
    partition_kernel_i64, partition_reduce_kernel_count_i64, scatter_kernel_i64, CudaModule,
};
use crate::plan::logical_plan::{AggregateExpr, DataType, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

/// Try the two-key Tier-2.1 COUNT(*) fast path. `None` on any precondition
/// miss so the caller falls through to the next strategy.
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
    if aggregate.group_by.len() != 2 || aggregate.aggregates.len() != 1 {
        return None;
    }

    // Exactly one COUNT aggregate. Argument is decorative — see the
    // single-key COUNT executor for the rationale.
    match &aggregate.aggregates[0] {
        AggregateExpr::Count(_) => {}
        _ => return None,
    }

    // Both keys must be Int32.
    let k1_io = aggregate.inputs.get(aggregate.group_by[0])?;
    let k2_io = aggregate.inputs.get(aggregate.group_by[1])?;
    if k1_io.dtype != DataType::Int32 || k2_io.dtype != DataType::Int32 {
        return None;
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

    let n_rows = k1.len();
    if n_rows < 256 * 1024 {
        return None;
    }

    // Cardinality cap: at most n_rows distinct groups. Tier-2 dispatcher
    // caps at 100 M. The lower-bound gate (vs `BLOCK_GROUPS`) is implicit
    // — at n_rows >= 256K the single-key Tier-1 path doesn't kick in for
    // two-key plans anyway, so there's no Tier-1 sibling to defer to.
    if n_rows >= 100_000_000 {
        return None;
    }

    Some(execute_inner(plan, k1, k2))
}

fn execute_inner(
    plan: &PhysicalPlan,
    k1: &Int32Array,
    k2: &Int32Array,
) -> JavelinResult<RecordBatch> {
    let n_rows = k1.len() as u32;

    // Host-side pack: `(k1 << 32) | (k2 & 0xFFFF_FFFF)`. Matches
    // `groupby.rs::pack_keys` for the (Int32, Int32) shape — high half
    // is column 0, low half is column 1, both zero-extended via u32.
    let packed: Vec<i64> = k1
        .values()
        .iter()
        .zip(k2.values().iter())
        .map(|(&a, &b)| ((a as u32 as u64) << 32 | (b as u32 as u64)) as i64)
        .collect();
    let keys_gpu: GpuVec<i64> = GpuVec::<i64>::from_slice(&packed)?;

    let num_partitions = partition_kernel_i64::NUM_PARTITIONS;
    let counts: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;
    let partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros(n_rows as usize)?;

    // -------- Partition pass (i64) --------
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

    // -------- Offsets --------
    let offsets: Vec<u32> = partition_offsets::compute_partition_offsets(&counts)?;
    let offsets_gpu: GpuVec<u32> = partition_offsets::upload_offsets(&offsets)?;

    // -------- Scatter (keys only; dummy value column to satisfy ABI) --------
    //
    // scatter_kernel_i64 requires a value-column input/output. COUNT has
    // no meaningful value — pass a zero-filled f64 buffer of the same
    // length, exactly as the single-key COUNT executor does. The dummy
    // out_vals buffer is written but never read.
    let scatter_keys: GpuVec<i64> = GpuVec::<i64>::zeros(n_rows as usize)?;
    let dummy_vals_in: GpuVec<f64> = GpuVec::<f64>::zeros(n_rows as usize)?;
    let scatter_vals: GpuVec<f64> = GpuVec::<f64>::zeros(n_rows as usize)?;
    {
        let ptx = scatter_kernel_i64::compile_scatter_kernel_i64()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(scatter_kernel_i64::KERNEL_ENTRY)?;
        let cursors: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;
        let mut keys_ptr = keys_gpu.device_ptr();
        let mut vals_ptr = dummy_vals_in.device_ptr();
        let mut pids_ptr = partition_ids.device_ptr();
        let mut offsets_ptr = offsets_gpu.device_ptr();
        let mut cursors_ptr = cursors.device_ptr();
        let mut sk_ptr = scatter_keys.device_ptr();
        let mut sv_ptr = scatter_vals.device_ptr();
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
    let _ = (dummy_vals_in, scatter_vals); // keep alive until end of launch

    // -------- COUNT reduce (i64-key) --------
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice(&offsets)?;
    let block_groups = partition_reduce_kernel_count_i64::BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;
    let out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros(n_out_slots)?;
    let out_counts_gpu: GpuVec<u64> = GpuVec::<u64>::zeros(n_out_slots)?;
    let out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;
    {
        let ptx = partition_reduce_kernel_count_i64::compile_partition_reduce_kernel_count_i64()?;
        let module = CudaModule::from_ptx(&ptx)?;
        let func = module.function(partition_reduce_kernel_count_i64::KERNEL_ENTRY)?;
        let mut keys_ptr = scatter_keys.device_ptr();
        let mut offsets_ptr = offsets_kp1_gpu.device_ptr();
        let mut ok_ptr = out_keys_gpu.device_ptr();
        let mut oc_ptr = out_counts_gpu.device_ptr();
        let mut os_ptr = out_set_gpu.device_ptr();
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

    // -------- Download + unpack + build output --------
    let host_out_keys: Vec<i64> = out_keys_gpu.to_vec()?;
    let host_out_counts: Vec<u64> = out_counts_gpu.to_vec()?;
    let host_out_set: Vec<u8> = out_set_gpu.to_vec()?;

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] == 0 {
                continue;
            }
            // u64 → i64 cast is safe: count is bounded by n_rows.
            rows.push((host_out_keys[idx], host_out_counts[idx] as i64));
        }
    }
    rows.sort_by_key(|(k, _)| *k);

    let mut out_k1: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_k2: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_counts_v: Vec<i64> = Vec::with_capacity(rows.len());
    for (k, c) in rows {
        let u = k as u64;
        out_k1.push((u >> 32) as u32 as i32);
        out_k2.push((u & 0xFFFF_FFFF) as u32 as i32);
        out_counts_v.push(c);
    }

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int32Array::from(out_k1)),
            Arc::new(Int32Array::from(out_k2)),
            Arc::new(Int64Array::from(out_counts_v)),
        ],
    )
    .map_err(|e| {
        JavelinError::Other(format!(
            "groupby_tier2_twokey_count_exec: failed to build RecordBatch: {e}"
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
