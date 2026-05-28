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

use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, RecordBatch};
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

    // PV-stage-f: NULL handling — `partition_reduce_kernel_multi_i64` has
    // no `_with_validity` companion. Defer NULL-bearing batches back to
    // the no-pre single-key paths which properly handle validity.
    // Stage G follow-up: validity-aware multi-reduce.
    if k1.null_count() > 0 || k2.null_count() > 0 {
        return None;
    }
    if vals.iter().any(|a| a.null_count() > 0) {
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

    // Stage-4 (P1b): per-call stream shared across every H2D / kernel / D2H.
    let stream = CudaStream::null_or_default();

    // Host-side pack: (k1 << 32) | (k2 & 0xFFFF_FFFF). Matches the
    // convention in `groupby.rs::pack_keys`.
    let packed: Vec<i64> = k1
        .values()
        .iter()
        .zip(k2.values().iter())
        .map(|(&a, &b)| ((a as i64) << 32) | (b as i64 & 0xFFFF_FFFF))
        .collect();
    let keys_gpu: GpuVec<i64> = GpuVec::<i64>::from_slice_async(&packed, stream.raw())?;

    let mut vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for arr in &val_arrs {
        vals_gpu.push(GpuVec::<f64>::from_slice_async(arr.values(), stream.raw())?);
    }

    let num_partitions = partition_kernel_i64::NUM_PARTITIONS;

    // -------- Partition pass (i64) --------
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;
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
        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    // P1b-stage8: joint helper, 2 syncs → 1.
    let (offsets, offsets_gpu): (Vec<u32>, GpuVec<u32>) =
        partition_offsets::compute_and_upload_partition_offsets_async(&counts, stream.raw())?;

    // -------- Scatter (keys + each value column) --------
    // scatter_kernel_i64 takes i64 keys + f64 vals. Reuse for each val
    // column with a fresh per-partition cursor buffer.
    let mut scatter_keys: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        scatter_vals.push(GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?);
    }
    {
        let scatter_module = get_or_build_module(&KernelSpec::ScatterI64)?;
        let func = scatter_module.function(scatter_kernel_i64::KERNEL_ENTRY)?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);

        for j in 0..n_vals {
            // Fresh per-iteration cursor — must be re-zeroed so each
            // scatter call writes into slots [0..m_k) of each partition.
            let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;

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

            launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
        }
    }

    // -------- Reduce (i64-key multi-value) --------
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;
    let block_groups = partition_reduce_kernel_multi_i64::BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        out_vals_gpu.push(GpuVec::<f64>::zeros_async(n_out_slots, stream.raw())?);
    }
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;

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

        launch_with_geometry(
            func,
            num_partitions,
            partition_reduce_kernel_multi_i64::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // -------- Stage-4 (P1b): pinned D2H; sync once --------
    let pinned_keys = out_keys_gpu.to_pinned_async(stream.raw())?;
    let mut pinned_vals: Vec<crate::cuda::PinnedHostBuffer<f64>> =
        Vec::with_capacity(n_vals);
    for ov in &out_vals_gpu {
        pinned_vals.push(ov.to_pinned_async(stream.raw())?);
    }
    let pinned_set = out_set_gpu.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_out_keys: Vec<i64> = pinned_keys.as_slice().to_vec();
    let host_out_vals: Vec<Vec<f64>> = pinned_vals
        .iter()
        .map(|p| p.as_slice().to_vec())
        .collect();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

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
// Host-only eligibility-gate tests for the two-key multi-SUM exec.
//
// Note: this exec rejects `n_vals < 2` deliberately (single-SUM two-key has
// its own faster shim, `groupby_tier2_twokey_exec`). Tests below pin that
// boundary explicitly so a future refactor doesn't accidentally widen the
// scope and shadow the single-agg path.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    /// Plan for `SELECT k1, k2, SUM(v0), SUM(v1), … FROM t GROUP BY k1, k2`.
    fn build_twokey_multi_plan(n_vals: usize) -> PhysicalPlan {
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
            aggregates.push(AggregateExpr::Sum(Expr::Column(name.clone())));
            out_fields.push(Field::new(format!("sum_{name}"), DataType::Float64, true));
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

    fn twokey_multi_batch(n: usize, n_vals: usize) -> RecordBatch {
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

    /// Non-Aggregate plan: reject.
    #[test]
    fn rejects_non_aggregate_plan() {
        let plan = PhysicalPlan::Union { inputs: vec![] };
        let batch = twokey_multi_batch(0, 2);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Single-key plan: reject (single-key multi exec owns it).
    #[test]
    fn rejects_single_key_plan() {
        let mut plan = build_twokey_multi_plan(2);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.group_by = vec![0];
        }
        let batch = twokey_multi_batch(300_000, 2);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Single SUM (n_vals == 1) belongs to the single-agg two-key shim.
    /// This exec must NOT shadow it.
    #[test]
    fn rejects_single_aggregate() {
        let plan = build_twokey_multi_plan(1);
        let batch = twokey_multi_batch(300_000, 1);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Above MAX_VALS → reject (kernel doesn't compile beyond that).
    #[test]
    fn rejects_above_max_vals() {
        let n_vals = (partition_reduce_kernel_multi_i64::MAX_VALS as usize) + 1;
        let plan = build_twokey_multi_plan(n_vals);
        let batch = twokey_multi_batch(300_000, n_vals);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// AVG/MIN/MAX mixed in → not our shape.
    #[test]
    fn rejects_avg_aggregate() {
        let mut plan = build_twokey_multi_plan(2);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            // Second agg is AVG, not SUM.
            aggregate.aggregates[1] = AggregateExpr::Avg(Expr::Column("v1".into()));
        }
        let batch = twokey_multi_batch(300_000, 2);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Below the row threshold → defer.
    #[test]
    fn rejects_below_row_threshold() {
        let plan = build_twokey_multi_plan(2);
        let batch = twokey_multi_batch(2_048, 2);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// `pre` kernel present → defer.
    #[test]
    fn rejects_plan_with_pre_kernel() {
        use crate::plan::physical_plan::KernelSpec;
        let mut plan = build_twokey_multi_plan(2);
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
        let batch = twokey_multi_batch(300_000, 2);
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

// ---------------------------------------------------------------------------
// Stage-4 (P1b) async smoke test.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage4_tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn async_tier2_twokey_multi_round_trip() {
        let n: usize = 300_000;
        let k1: Vec<i32> = (0..n as i32).map(|i| i % 64).collect();
        let k2: Vec<i32> = (0..n as i32).map(|i| (i / 64) % 64).collect();
        let v1: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let v2: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO { name: "k1".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "k2".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "v1".into(), dtype: DataType::Float64 },
                    ColumnIO { name: "v2".into(), dtype: DataType::Float64 },
                ],
                group_by: vec![0, 1],
                aggregates: vec![
                    AggregateExpr::Sum(Expr::Column("v1".into())),
                    AggregateExpr::Sum(Expr::Column("v2".into())),
                ],
                output_schema: Schema::new(vec![
                    Field::new("k1", DataType::Int32, false),
                    Field::new("k2", DataType::Int32, false),
                    Field::new("sum_v1", DataType::Float64, true),
                    Field::new("sum_v2", DataType::Float64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v1", ArrowDataType::Float64, false),
            ArrowField::new("v2", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k1)) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(k2)) as arrow_array::ArrayRef,
                Arc::new(Float64Array::from(v1)) as arrow_array::ArrayRef,
                Arc::new(Float64Array::from(v2)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        let _ = try_execute(&plan, &batch);
    }
}
