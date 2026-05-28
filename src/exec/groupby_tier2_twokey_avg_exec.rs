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

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;
use crate::exec::partition_offsets;
use crate::jit::{
    partition_kernel_i64, partition_reduce_kernel_count_i64,
    partition_reduce_kernel_multi_i64, scatter_kernel_i64, CudaModule,
};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Per-executor module cache. See `groupby_tier2_count_exec.rs` for the
// motivation and concurrency notes — the design is identical, but over the
// i64-key kernel variants with the multi-SUM reduce parameterised on
// `n_vals`.
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum KernelSpec {
    PartitionI64,
    ScatterI64,
    ReduceMultiI64 { n_vals: u32 },
    ReduceCountI64,
}

static MODULE_CACHE: Lazy<Mutex<HashMap<KernelSpec, CudaModule>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[cfg(test)]
static LOAD_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn get_or_build_module(spec: &KernelSpec) -> BoltResult<CudaModule> {
    if let Some(m) = MODULE_CACHE.lock().get(spec) {
        return Ok(m.clone());
    }
    let ptx = match spec {
        KernelSpec::PartitionI64 => partition_kernel_i64::compile_partition_kernel_i64()?,
        KernelSpec::ScatterI64 => scatter_kernel_i64::compile_scatter_kernel_i64()?,
        KernelSpec::ReduceMultiI64 { n_vals } => {
            partition_reduce_kernel_multi_i64::compile_partition_reduce_kernel_multi_i64_with_spill(
                *n_vals,
            )?
        }
        KernelSpec::ReduceCountI64 => {
            partition_reduce_kernel_count_i64::compile_partition_reduce_kernel_count_i64_with_spill()?
        }
    };
    let module = CudaModule::from_ptx(&ptx)?;
    #[cfg(test)]
    LOAD_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut cache = MODULE_CACHE.lock();
    Ok(cache.entry(spec.clone()).or_insert(module).clone())
}

/// Try the two-key Tier-2.1 multi-AVG fast path. `None` on any
/// precondition miss so the caller falls through to the next strategy.
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

    // PV-stage-f: NULL handling — the partition_reduce_kernel_multi_i64
    // family has no `_with_validity` companion, and the host-side i64
    // pack reads `.values()` directly. Defer NULL-bearing batches back
    // through the no-pre single-key paths which handle validity
    // correctly. Stage G follow-up: validity-aware partition+reduce.
    if k1.null_count() > 0 || k2.null_count() > 0 {
        return None;
    }
    if val_arrs.iter().any(|a| a.null_count() > 0) {
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
) -> BoltResult<RecordBatch> {
    let n_rows = k1.len() as u32;

    // Stage-4 (P1b): per-call stream shared across every H2D, kernel
    // launch, and final D2H.
    let stream = CudaStream::null_or_default();

    // ---- Host-side pack ------------------------------------------------
    // `(k1 << 32) | (k2 & 0xFFFF_FFFF)`. Matches `groupby.rs::pack_keys`
    // for the (Int32, Int32) shape.
    let packed: Vec<i64> = k1
        .values()
        .iter()
        .zip(k2.values().iter())
        .map(|(&a, &b)| ((a as u32 as u64) << 32 | (b as u32 as u64)) as i64)
        .collect();
    let keys_gpu: GpuVec<i64> = GpuVec::<i64>::from_slice_async(&packed, stream.raw())?;

    let mut vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for arr in &val_arrs {
        vals_gpu.push(GpuVec::<f64>::from_slice_async(arr.values(), stream.raw())?);
    }

    let num_partitions = partition_kernel_i64::NUM_PARTITIONS;

    // ---- Partition pass (i64) ------------------------------------------
    let counts: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
    let partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;
    let partition_module = get_or_build_module(&KernelSpec::PartitionI64)?;
    {
        let func = partition_module.function(partition_kernel_i64::KERNEL_ENTRY)?;
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

    // ---- Offsets (P1b-stage8: joint helper, 2 syncs → 1) --------------
    let (offsets, offsets_gpu): (Vec<u32>, GpuVec<u32>) =
        partition_offsets::compute_and_upload_partition_offsets_async(&counts, stream.raw())?;

    // ---- Scatter (keys shared + N value passes) ------------------------
    let scatter_keys: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        scatter_vals.push(GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?);
    }
    {
        let scatter_module = get_or_build_module(&KernelSpec::ScatterI64)?;
        let func = scatter_module.function(scatter_kernel_i64::KERNEL_ENTRY)?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);

        for j in 0..n_vals {
            let cursors: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
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
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;

    let block_groups = partition_reduce_kernel_multi_i64::BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        out_vals_gpu.push(GpuVec::<f64>::zeros_async(n_out_slots, stream.raw())?);
    }
    let out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;
    let out_counts_gpu: GpuVec<u64> = GpuVec::<u64>::zeros_async(n_out_slots, stream.raw())?;
    // COUNT kernel writes its own out_keys / out_set buffers; we only
    // consume its out_counts. SUM-side out_keys/out_set are the canonical
    // dedup output. (Strictly speaking the count_out_keys/set are
    // redundant; allocating them is cheaper than special-casing the
    // kernel signature.)
    let count_out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_out_slots, stream.raw())?;
    let count_out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;
    // Two independent spill counters — the multi-SUM reduce and the
    // COUNT reduce both bump on probe overflow; either non-zero means
    // the result is silently truncated.
    let spill_multi: GpuVec<u32> = GpuVec::<u32>::zeros_async(1, stream.raw())?;
    let spill_count_buf: GpuVec<u32> = GpuVec::<u32>::zeros_async(1, stream.raw())?;

    // ---- Multi-SUM reduce (i64-key) ------------------------------------
    let reduce_multi_module = get_or_build_module(&KernelSpec::ReduceMultiI64 {
        n_vals: n_vals as u32,
    })?;
    {
        let entry =
            partition_reduce_kernel_multi_i64::kernel_entry_with_spill(n_vals as u32);
        let func = reduce_multi_module.function(&entry)?;

        let mut storage: Vec<CUdeviceptr> = Vec::with_capacity(4 + 2 * n_vals + 1);
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
        storage.push(spill_multi.device_ptr());

        let mut params: Vec<*mut c_void> = storage
            .iter_mut()
            .map(|p| p as *mut CUdeviceptr as *mut c_void)
            .collect();
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
    let reduce_count_module = get_or_build_module(&KernelSpec::ReduceCountI64)?;
    {
        let func = reduce_count_module
            .function(partition_reduce_kernel_count_i64::KERNEL_ENTRY_WITH_SPILL)?;
        let mut keys_ptr = scatter_keys.device_ptr();
        let mut offsets_ptr = offsets_kp1_gpu.device_ptr();
        let mut ok_ptr = count_out_keys_gpu.device_ptr();
        let mut oc_ptr = out_counts_gpu.device_ptr();
        let mut os_ptr = count_out_set_gpu.device_ptr();
        let mut sp_ptr = spill_count_buf.device_ptr();
        let mut params: [*mut c_void; 6] = [
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut offsets_ptr as *mut CUdeviceptr as *mut c_void,
            &mut ok_ptr as *mut CUdeviceptr as *mut c_void,
            &mut oc_ptr as *mut CUdeviceptr as *mut c_void,
            &mut os_ptr as *mut CUdeviceptr as *mut c_void,
            &mut sp_ptr as *mut CUdeviceptr as *mut c_void,
        ];
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

    // ---- Stage-4 (P1b): pinned D2H for every output buffer; sync once.
    let pinned_keys = out_keys_gpu.to_pinned_async(stream.raw())?;
    let mut pinned_vals: Vec<crate::cuda::PinnedHostBuffer<f64>> =
        Vec::with_capacity(n_vals);
    for ov in &out_vals_gpu {
        pinned_vals.push(ov.to_pinned_async(stream.raw())?);
    }
    let pinned_set = out_set_gpu.to_pinned_async(stream.raw())?;
    let pinned_counts = out_counts_gpu.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let multi_spill = spill_multi.to_vec()?[0];
    let count_spill = spill_count_buf.to_vec()?[0];
    if multi_spill > 0 || count_spill > 0 {
        return Err(BoltError::Other(format!(
            "partition_reduce spill: multi-sum={} count={} rows exceeded MAX_PROBES; result may be incorrect",
            multi_spill, count_spill
        )));
    }
    let host_out_keys: Vec<i64> = pinned_keys.as_slice().to_vec();
    let host_out_vals: Vec<Vec<f64>> = pinned_vals
        .iter()
        .map(|p| p.as_slice().to_vec())
        .collect();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();
    let host_out_counts: Vec<u64> = pinned_counts.as_slice().to_vec();

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
        BoltError::Other(format!(
            "groupby_tier2_twokey_avg_exec: failed to build RecordBatch: {e}"
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

// ---------------------------------------------------------------------------
// Host-only eligibility-gate tests for the two-key Tier-2.1 multi-AVG exec.
//
// We only exercise `try_execute`'s shape-gate logic; the kernel launch path
// is GPU-bound and covered by the dedicated e2e suite.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    /// Plan for `SELECT k1, k2, AVG(v) FROM t GROUP BY k1, k2`.
    fn build_twokey_avg_plan(n_vals: usize) -> PhysicalPlan {
        let mut inputs = vec![
            ColumnIO {
                name: "k1".into(),
                dtype: DataType::Int32,
            },
            ColumnIO {
                name: "k2".into(),
                dtype: DataType::Int32,
            },
        ];
        let mut aggregates = Vec::with_capacity(n_vals);
        let mut out_fields = vec![
            Field::new("k1", DataType::Int32, false),
            Field::new("k2", DataType::Int32, false),
        ];
        for i in 0..n_vals {
            let name = format!("v{i}");
            inputs.push(ColumnIO {
                name: name.clone(),
                dtype: DataType::Float64,
            });
            aggregates.push(AggregateExpr::Avg(Expr::Column(name.clone())));
            out_fields.push(Field::new(format!("avg_{name}"), DataType::Float64, true));
        }
        PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs,
                group_by: vec![0, 1],
                aggregates,
                output_schema: Schema::new(out_fields),
                input_has_validity: Vec::new(),
            },
        }
    }

    /// Build a matching `(k1: Int32, k2: Int32, v0..vN-1: Float64)` batch.
    fn twokey_avg_batch(n: usize, n_vals: usize) -> RecordBatch {
        let k1: Vec<i32> = (0..n as i32).collect();
        let k2: Vec<i32> = (0..n as i32).map(|i| i + 1).collect();
        let mut fields: Vec<ArrowField> = vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
        ];
        let mut cols: Vec<arrow_array::ArrayRef> = vec![
            Arc::new(Int32Array::from(k1)) as arrow_array::ArrayRef,
            Arc::new(Int32Array::from(k2)) as arrow_array::ArrayRef,
        ];
        for i in 0..n_vals {
            fields.push(ArrowField::new(
                format!("v{i}"),
                ArrowDataType::Float64,
                false,
            ));
            let v: Vec<f64> = (0..n).map(|j| j as f64).collect();
            cols.push(Arc::new(Float64Array::from(v)) as arrow_array::ArrayRef);
        }
        let schema = Arc::new(ArrowSchema::new(fields));
        RecordBatch::try_new(schema, cols).unwrap()
    }

    /// Non-Aggregate plans are not our business.
    #[test]
    fn rejects_non_aggregate_plan() {
        let plan = PhysicalPlan::Union { inputs: vec![] };
        let batch = twokey_avg_batch(0, 1);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Single-key plans → single-key sibling.
    #[test]
    fn rejects_single_key_plan() {
        let mut plan = build_twokey_avg_plan(1);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.group_by = vec![0];
        }
        let batch = twokey_avg_batch(300_000, 1);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// SUM mixed in → fall through (we want pure AVG).
    #[test]
    fn rejects_mixed_aggregates() {
        let mut plan = build_twokey_avg_plan(1);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.aggregates =
                vec![AggregateExpr::Sum(Expr::Column("v0".into()))];
        }
        let batch = twokey_avg_batch(300_000, 1);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Zero aggregates → reject.
    #[test]
    fn rejects_zero_aggregates() {
        let mut plan = build_twokey_avg_plan(1);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.aggregates.clear();
        }
        let batch = twokey_avg_batch(300_000, 1);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Below-threshold rows → defer.
    #[test]
    fn rejects_below_row_threshold() {
        let plan = build_twokey_avg_plan(2);
        let batch = twokey_avg_batch(1_024, 2);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Non-Float64 value column → reject (we only accept f64 today).
    #[test]
    fn rejects_int64_value_column() {
        let plan = build_twokey_avg_plan(1);
        let n = 300_000;
        let k1: Vec<i32> = (0..n as i32).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v0", ArrowDataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k1.clone())) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(k1)) as arrow_array::ArrayRef,
                Arc::new(arrow_array::Int64Array::from(
                    (0..n).map(|i| i as i64).collect::<Vec<_>>(),
                )) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// `pre` kernel present → defer.
    #[test]
    fn rejects_plan_with_pre_kernel() {
        use crate::plan::physical_plan::KernelSpec;
        let mut plan = build_twokey_avg_plan(1);
        if let PhysicalPlan::Aggregate { pre, .. } = &mut plan {
            *pre = Some(KernelSpec {
                inputs: vec![],
                outputs: vec![],
                ops: vec![],
                predicate: None,
                register_count: 0,
                input_has_validity: vec![],
                output_has_validity: vec![],
            });
        }
        let batch = twokey_avg_batch(300_000, 1);
        assert!(try_execute(&plan, &batch).is_none());
    }
}

// ---------------------------------------------------------------------------
// Module-cache mechanics tests. Skip on CPU-only hosts.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn second_call_same_spec_is_cache_hit() {
        let m1 = match get_or_build_module(&KernelSpec::PartitionI64) {
            Ok(m) => m,
            Err(_) => return,
        };
        let after_first = LOAD_COUNT.load(Ordering::SeqCst);
        let m2 = get_or_build_module(&KernelSpec::PartitionI64).expect("hit");
        assert_eq!(LOAD_COUNT.load(Ordering::SeqCst), after_first);
        assert_eq!(m1.raw(), m2.raw());
    }

    #[test]
    fn different_n_vals_are_distinct_cache_keys() {
        let _ = match get_or_build_module(&KernelSpec::ReduceMultiI64 { n_vals: 1 }) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ReduceMultiI64 { n_vals: 2 })
            .expect("n_vals=2 build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::ReduceMultiI64 { n_vals: 1 })
            .expect("n_vals=1 hit");
        let _ = get_or_build_module(&KernelSpec::ReduceMultiI64 { n_vals: 2 })
            .expect("n_vals=2 hit");
        assert_eq!(LOAD_COUNT.load(Ordering::SeqCst), baseline);
    }
}

// ---------------------------------------------------------------------------
// Stage-4 (P1b) async smoke test. Just confirms `try_execute` doesn't
// panic and returns *some* output on a small-but-eligible input.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage4_tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn async_tier2_twokey_avg_round_trip() {
        let n: usize = 300_000;
        let k1: Vec<i32> = (0..n as i32).map(|i| i % 64).collect();
        let k2: Vec<i32> = (0..n as i32).map(|i| (i / 64) % 64).collect();
        let v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO { name: "k1".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "k2".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "v".into(), dtype: DataType::Float64 },
                ],
                group_by: vec![0, 1],
                aggregates: vec![AggregateExpr::Avg(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("k1", DataType::Int32, false),
                    Field::new("k2", DataType::Int32, false),
                    Field::new("avg_v", DataType::Float64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k1)) as ArrayRef,
                Arc::new(Int32Array::from(k2)) as ArrayRef,
                Arc::new(Float64Array::from(v)) as ArrayRef,
            ],
        )
        .unwrap();
        let _ = try_execute(&plan, &batch);
    }
}
