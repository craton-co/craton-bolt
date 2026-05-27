// SPDX-License-Identifier: Apache-2.0

//! **Two-key multi-aggregate Tier-2.1** executor.
//!
//! Covers `SELECT a, b, SUM(v1), SUM(v2), … FROM x GROUP BY a, b` where
//! `a, b` are both `Int32` and 1..=4 SUM aggregates run over `Float64`
//! columns. Combines the i64-key partitioning machinery
//! ([`crate::jit::partition_kernel_i64`], [`crate::jit::scatter_kernel_i64`])
//! with the multi-value reduce kernel
//! ([`crate::jit::partition_reduce_kernel_multi_i64`]). All three kernels
//! already exist; this file just orchestrates them.
//!
//! ## Scope (v0)
//!
//! - GROUP BY exactly two Int32 columns
//! - 1..=4 aggregates, ALL `SUM(<bare Float64 column>)`
//! - `n_rows >= 256 K`
//! - Total distinct (packed) keys ≤ 100 M (Tier-2 dispatcher cap)
//!
//! Mixed agg ops (e.g. `SUM(v1), AVG(v2)`) deserve a dedicated executor;
//! we reject mixed shapes cleanly and let the global-atomic fallback
//! handle them. Single-SUM two-key goes through the existing
//! `groupby_tier2_twokey_exec`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::partition_offsets;
use crate::jit::{
    partition_kernel_i64, partition_reduce_kernel_multi_i64, scatter_kernel_i64, CudaModule,
};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Per-executor module cache. See `groupby_tier2_count_exec.rs` for the
// motivation and concurrency notes. The multi-SUM reduce kernel here is
// parameterised on `n_vals`.
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum KernelSpec {
    PartitionI64,
    ScatterI64,
    ReduceMultiI64 { n_vals: u32 },
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
            partition_reduce_kernel_multi_i64::compile_partition_reduce_kernel_multi_i64(*n_vals)?
        }
    };
    let module = CudaModule::from_ptx(&ptx)?;
    #[cfg(test)]
    LOAD_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut cache = MODULE_CACHE.lock();
    Ok(cache.entry(spec.clone()).or_insert(module).clone())
}

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
    if n_vals < 2 || n_vals > partition_reduce_kernel_multi_i64::MAX_VALS as usize {
        // n_vals < 2: the single-agg two-key path (groupby_tier2_twokey_exec)
        // would have caught this earlier; reject here to avoid shadowing.
        return None;
    }

    // Both keys must be Int32.
    let key1_io = aggregate.inputs.get(aggregate.group_by[0])?;
    let key2_io = aggregate.inputs.get(aggregate.group_by[1])?;
    if key1_io.dtype != DataType::Int32 || key2_io.dtype != DataType::Int32 {
        return None;
    }

    // Every aggregate must be SUM(bare Float64 column).
    let mut sum_col_names: Vec<&str> = Vec::with_capacity(n_vals);
    for agg in &aggregate.aggregates {
        match agg {
            AggregateExpr::Sum(Expr::Column(n)) => sum_col_names.push(n.as_str()),
            _ => return None,
        }
    }

    // Look up arrays.
    let k1 = batch
        .column_by_name(&key1_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    let k2 = batch
        .column_by_name(&key2_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    if k1.len() != k2.len() {
        return None;
    }
    let mut vals: Vec<&Float64Array> = Vec::with_capacity(n_vals);
    for name in &sum_col_names {
        let a = batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())?;
        if a.len() != k1.len() {
            return None;
        }
        vals.push(a);
    }
    let n_rows = k1.len();
    if n_rows < 256 * 1024 {
        return None;
    }

    Some(execute_inner(plan, k1, k2, vals, n_vals))
}

fn execute_inner(
    plan: &PhysicalPlan,
    k1: &Int32Array,
    k2: &Int32Array,
    val_arrs: Vec<&Float64Array>,
    n_vals: usize,
) -> BoltResult<RecordBatch> {
    let n_rows = k1.len() as u32;

    // Host-side pack: (k1 << 32) | (k2 & 0xFFFF_FFFF). Matches the
    // convention in `groupby.rs::pack_keys`.
    let packed: Vec<i64> = k1
        .values()
        .iter()
        .zip(k2.values().iter())
        .map(|(&a, &b)| ((a as i64) << 32) | (b as i64 & 0xFFFF_FFFF))
        .collect();
    let keys_gpu: GpuVec<i64> = GpuVec::<i64>::from_slice(&packed)?;

    let mut vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for arr in &val_arrs {
        vals_gpu.push(GpuVec::<f64>::from_slice(arr.values())?);
    }

    let num_partitions = partition_kernel_i64::NUM_PARTITIONS;

    // -------- Partition pass (i64) --------
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros(n_rows as usize)?;
    let partition_module = get_or_build_module(&KernelSpec::PartitionI64)?;
    {
        let func = partition_module.function(partition_kernel_i64::KERNEL_ENTRY)?;

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

    // -------- Scatter (keys + each value column) --------
    // scatter_kernel_i64 takes i64 keys + f64 vals. Reuse for each val
    // column with a fresh per-partition cursor buffer.
    let mut scatter_keys: GpuVec<i64> = GpuVec::<i64>::zeros(n_rows as usize)?;
    let mut scatter_vals: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        scatter_vals.push(GpuVec::<f64>::zeros(n_rows as usize)?);
    }
    {
        let scatter_module = get_or_build_module(&KernelSpec::ScatterI64)?;
        let func = scatter_module.function(scatter_kernel_i64::KERNEL_ENTRY)?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);

        for j in 0..n_vals {
            // Fresh per-iteration cursor — must be re-zeroed so each
            // scatter call writes into slots [0..m_k) of each partition.
            let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;

            // Split-borrow on `scatter_vals` to hold scatter_keys mutably
            // alongside scatter_vals[j] mutably.
            let (sv_j_slice, _) = scatter_vals.split_at_mut(j + 1);
            let scatter_vals_j = &mut sv_j_slice[j];

            let view_keys = keys_gpu.view();
            let view_vals = vals_gpu[j].view();
            let view_pids = partition_ids.view();
            let view_offsets = offsets_gpu.view();
            let mut view_cursors = cursors.view_mut();
            let mut view_sk = scatter_keys.view_mut();
            let mut view_sv = scatter_vals_j.view_mut();

            let mut args = KernelArgs::empty();
            args.push_input(&view_keys);
            args.push_input(&view_vals);
            args.push_input(&view_pids);
            args.push_input(&view_offsets);
            args.push_output(&mut view_cursors);
            args.push_output(&mut view_sk);
            args.push_output(&mut view_sv);
            args.push_scalar_u32(n_rows);

            let stream = CudaStream::null();
            launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
        }
    }

    // -------- Reduce (i64-key multi-value) --------
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice(&offsets)?;
    let block_groups = partition_reduce_kernel_multi_i64::BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros(n_out_slots)?;
    let mut out_vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        out_vals_gpu.push(GpuVec::<f64>::zeros(n_out_slots)?);
    }
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;

    let reduce_module = get_or_build_module(&KernelSpec::ReduceMultiI64 {
        n_vals: n_vals as u32,
    })?;
    {
        let entry = partition_reduce_kernel_multi_i64::kernel_entry(n_vals as u32);
        let func = reduce_module.function(&entry)?;

        // Kernel param order:
        //   partition_keys, partition_vals_0..={N-1},
        //   partition_offsets, out_keys,
        //   out_vals_0..={N-1}, out_set
        //
        // Collect iterated views eagerly so they outlive `args`.
        let view_pk = scatter_keys.view();
        let views_sv: Vec<_> = scatter_vals.iter().map(|g| g.view()).collect();
        let view_po = offsets_kp1_gpu.view();
        let mut view_ok = out_keys_gpu.view_mut();
        let mut views_ov: Vec<_> =
            out_vals_gpu.iter_mut().map(|g| g.view_mut()).collect();
        let mut view_os = out_set_gpu.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_pk);
        for v in &views_sv {
            args.push_input(v);
        }
        args.push_input(&view_po);
        args.push_output(&mut view_ok);
        for v in views_ov.iter_mut() {
            args.push_output(v);
        }
        args.push_output(&mut view_os);

        let stream = CudaStream::null();
        launch_with_geometry(
            func,
            num_partitions,
            partition_reduce_kernel_multi_i64::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // -------- Download + unpack + build output --------
    let host_out_keys: Vec<i64> = out_keys_gpu.to_vec()?;
    let mut host_out_vals: Vec<Vec<f64>> = Vec::with_capacity(n_vals);
    for ov in &out_vals_gpu {
        host_out_vals.push(ov.to_vec()?);
    }
    let host_out_set: Vec<u8> = out_set_gpu.to_vec()?;

    // (key_i64, [sum_0, sum_1, ..., sum_{N-1}])
    let mut rows: Vec<(i64, Vec<f64>)> = Vec::new();
    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] == 0 {
                continue;
            }
            let sums: Vec<f64> = (0..n_vals).map(|j| host_out_vals[j][idx]).collect();
            rows.push((host_out_keys[idx], sums));
        }
    }
    rows.sort_by_key(|(k, _)| *k);

    let mut out_k1: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_k2: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_sums: Vec<Vec<f64>> = (0..n_vals)
        .map(|_| Vec::with_capacity(rows.len()))
        .collect();
    for (k, sums) in rows {
        out_k1.push((k >> 32) as i32);
        out_k2.push((k & 0xFFFF_FFFF) as i32);
        for (j, s) in sums.into_iter().enumerate() {
            out_sums[j].push(s);
        }
    }

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!(),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(2 + n_vals);
    cols.push(Arc::new(Int32Array::from(out_k1)));
    cols.push(Arc::new(Int32Array::from(out_k2)));
    for v in out_sums {
        cols.push(Arc::new(Float64Array::from(v)));
    }
    RecordBatch::try_new(arrow_schema, cols).map_err(|e| {
        BoltError::Other(format!(
            "groupby_tier2_twokey_multi_exec: failed to build RecordBatch: {e}"
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
        let _ = match get_or_build_module(&KernelSpec::ReduceMultiI64 { n_vals: 2 }) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ReduceMultiI64 { n_vals: 3 })
            .expect("n_vals=3 build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::ReduceMultiI64 { n_vals: 2 })
            .expect("n_vals=2 hit");
        let _ = get_or_build_module(&KernelSpec::ReduceMultiI64 { n_vals: 3 })
            .expect("n_vals=3 hit");
        assert_eq!(LOAD_COUNT.load(Ordering::SeqCst), baseline);
    }
}
