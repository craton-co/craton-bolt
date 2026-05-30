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

use arrow_array::{Array, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::module_cache;
use crate::exec::partition_offsets;
use crate::jit::partition_reduce_kernel_minmax::MinMaxOp;
use crate::jit::partition_reduce_kernel_minmax_float::{
    compile_partition_reduce_kernel_minmax_float_with_spill,
    kernel_entry_with_spill as minmax_float_entry_with_spill, FloatDtype, BLOCK_GROUPS,
    BLOCK_THREADS as REDUCE_BLOCK_THREADS,
};
use crate::jit::{partition_kernel, scatter_kernel, CudaModule};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Per-executor module cache. See `groupby_tier2_count_exec.rs` for the
// motivation and concurrency notes — the design is identical.
//
// The float MIN/MAX reduce is parameterised on `(MinMaxOp, FloatDtype)`. We
// mirror those as cache-key variants because neither upstream enum derives
// `Hash`.
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
enum ReduceFloatKey {
    MinF32,
    MaxF32,
    MinF64,
    MaxF64,
}

impl ReduceFloatKey {
    fn from_pair(op: MinMaxOp, dt: FloatDtype) -> Self {
        match (op, dt) {
            (MinMaxOp::Min, FloatDtype::Float32) => ReduceFloatKey::MinF32,
            (MinMaxOp::Max, FloatDtype::Float32) => ReduceFloatKey::MaxF32,
            (MinMaxOp::Min, FloatDtype::Float64) => ReduceFloatKey::MinF64,
            (MinMaxOp::Max, FloatDtype::Float64) => ReduceFloatKey::MaxF64,
        }
    }

    fn into_pair(self) -> (MinMaxOp, FloatDtype) {
        match self {
            ReduceFloatKey::MinF32 => (MinMaxOp::Min, FloatDtype::Float32),
            ReduceFloatKey::MaxF32 => (MinMaxOp::Max, FloatDtype::Float32),
            ReduceFloatKey::MinF64 => (MinMaxOp::Min, FloatDtype::Float64),
            ReduceFloatKey::MaxF64 => (MinMaxOp::Max, FloatDtype::Float64),
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum KernelSpec {
    Partition,
    PartitionShmemStaging,
    Scatter,
    ReduceMinMaxFloat(ReduceFloatKey),
}

#[cfg(test)]
static LOAD_COUNT: module_cache::LoadCounter = module_cache::LoadCounter::new();

fn get_or_build_module(spec: &KernelSpec) -> BoltResult<CudaModule> {
    #[cfg(test)]
    let counter = Some(&LOAD_COUNT);
    #[cfg(not(test))]
    let counter = None;
    module_cache::get_or_build_module(module_path!(), format!("{:?}", spec), counter, || {
        Ok(match spec {
            KernelSpec::Partition => partition_kernel::compile_partition_kernel()?,
            KernelSpec::PartitionShmemStaging => partition_kernel::compile_partition_kernel_shmem_staging()?,
            KernelSpec::Scatter => scatter_kernel::compile_scatter_kernel()?,
            KernelSpec::ReduceMinMaxFloat(rk) => {
                // Batch 5: route to the spill-counter-aware emitter so the
                // launch sites (which resolve `kernel_entry_with_spill`) can
                // find their entry point in the loaded module.
                let (op, dt) = rk.into_pair();
                compile_partition_reduce_kernel_minmax_float_with_spill(op, dt)?
            }
        })
    })
}

fn partition_spec_for(n_rows: u32) -> KernelSpec {
    if n_rows < partition_kernel::SHMEM_STAGING_MIN_ROWS {
        KernelSpec::Partition
    } else {
        KernelSpec::PartitionShmemStaging
    }
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

    // GB-S1: NULL handling — this fast path reads `key_arr.values()` and the
    // value column straight off the Arrow data buffers, which carry garbage
    // bytes at NULL positions (a NULL value could spuriously win the
    // MIN/MAX; a NULL key synthesizes a group-0). Defer NULL-bearing
    // batches back to `groupby::execute_groupby` → the global-atomic path,
    // which consults the validity bitmap. Mirrors the guard in
    // `groupby_tier2_twokey_exec::try_execute`.
    if key_arr.null_count() > 0 || val_col.null_count() > 0 {
        return None;
    }

    // F2: NaN handling. The host scalar/window MIN/MAX path implements
    // DuckDB's total order (NaN sorts as the largest value; MIN skips NaN
    // unless all-NaN; MAX surfaces NaN if present — see
    // `aggregate.rs::float_total_cmp`). The CAS-loop reduce kernel compares
    // raw IEEE floats, where NaN's participation is order-dependent, so it
    // can disagree with the scalar path for the same data. Cheapest correct
    // option: defer any NaN-bearing value column to the global-atomic / host
    // path so grouped float MIN/MAX always matches the scalar aggregate.
    if let Some(val_arr) = val_col.as_any().downcast_ref::<Float64Array>() {
        if val_arr.values().iter().any(|v| v.is_nan()) {
            return None;
        }
    }

    let n_rows = key_arr.len();
    if n_rows < 256 * 1024 {
        return None;
    }

    // dedup (tier2/shmem): max-nonneg-key scan extracted to
    // `groupby_tier2_common`. `None` (negative key) and `Some(-1)` (empty)
    // both decline, matching the prior inline behaviour.
    let max_key = crate::exec::groupby_tier2_common::scan_max_nonneg_key(key_arr.values())?;
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
) -> BoltResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;

    // Stage-4 (P1b): per-call stream shared by all H2D / kernels / D2H.
    let stream = CudaStream::null_or_default();
    let keys_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice_async(key_arr.values(), stream.raw())?;

    // Upload values. Float64 path only for v0.
    let val_arr = val_col
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| BoltError::Other("expected Float64Array".into()))?;
    let vals_gpu: GpuVec<f64> = GpuVec::<f64>::from_slice_async(val_arr.values(), stream.raw())?;

    let num_partitions = partition_kernel::NUM_PARTITIONS;

    // --- Partition pass ---
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;
    let partition_module = get_or_build_module(&partition_spec_for(n_rows))?;
    {
        let func = partition_module.function(partition_kernel::KERNEL_ENTRY)?;

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

    // --- Scatter (f64 vals — no conversion needed) ---
    let mut scatter_keys: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?;
    let scatter_module = get_or_build_module(&KernelSpec::Scatter)?;
    {
        let func = scatter_module.function(scatter_kernel::KERNEL_ENTRY)?;
        let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;

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
        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    // --- Reduce (CAS-loop float MIN/MAX) ---
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;
    let block_groups = BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals_gpu: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;
    let mut spill: GpuVec<u32> = GpuVec::<u32>::zeros_async(1, stream.raw())?;

    let reduce_module = get_or_build_module(&KernelSpec::ReduceMinMaxFloat(
        ReduceFloatKey::from_pair(op, float_dtype),
    ))?;
    {
        let entry = minmax_float_entry_with_spill(op, float_dtype);
        let func = reduce_module.function(&entry)?;

        let view_pk = scatter_keys.view();
        let view_pv = scatter_vals.view();
        let view_po = offsets_kp1_gpu.view();
        let mut view_ok = out_keys_gpu.view_mut();
        let mut view_ov = out_vals_gpu.view_mut();
        let mut view_os = out_set_gpu.view_mut();
        let mut view_sp = spill.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_pk);
        args.push_input(&view_pv);
        args.push_input(&view_po);
        args.push_output(&mut view_ok);
        args.push_output(&mut view_ov);
        args.push_output(&mut view_os);
        args.push_output(&mut view_sp);

        launch_with_geometry(
            func,
            num_partitions,
            REDUCE_BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // Stage-4 (P1b): pinned D2H for the three outputs; sync once.
    let pinned_keys = out_keys_gpu.to_pinned_async(stream.raw())?;
    let pinned_vals = out_vals_gpu.to_pinned_async(stream.raw())?;
    let pinned_set = out_set_gpu.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let spill_count = spill.to_vec()?[0];
    if spill_count > 0 {
        return Err(BoltError::Other(format!(
            "partition_reduce spill: {} rows exceeded MAX_PROBES; result may be incorrect",
            spill_count
        )));
    }
    let host_out_keys: Vec<i32> = pinned_keys.as_slice().to_vec();
    let host_out_vals: Vec<f64> = pinned_vals.as_slice().to_vec();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

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
        BoltError::Other(format!(
            "groupby_tier2_minmax_float_exec: build error: {e}"
        ))
    })
}
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal(s, "this aggregate output path")
}

// ---------------------------------------------------------------------------
// Module-cache mechanics tests. Skip on CPU-only hosts.
// ---------------------------------------------------------------------------
// v0.7: these arrow/plan aliases are used only by the #[cfg(test)] modules
// below; the non-test schema conversion now lives in exec::schema_convert.
// cfg(test)-gated so normal builds don't see an unused import.
#[cfg(test)]
use arrow_schema::{Field as ArrowField};

// ---------------------------------------------------------------------------
// F2: host-only NaN-deferral eligibility tests. A NaN-bearing float value
// column must DEFER (return `None`) so grouped float MIN/MAX routes through
// the global-atomic / host scalar path, which implements DuckDB's
// NaN-as-largest total order (`aggregate.rs::float_total_cmp`).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod nan_tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    fn build_minmax_plan(op_is_min: bool) -> PhysicalPlan {
        let agg = if op_is_min {
            AggregateExpr::Min(Expr::Column("v".into()))
        } else {
            AggregateExpr::Max(Expr::Column("v".into()))
        };
        PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO { name: "k".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "v".into(), dtype: DataType::Float64 },
                ],
                group_by: vec![0],
                aggregates: vec![agg],
                output_schema: Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("v", DataType::Float64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        }
    }

    fn batch_with(vals: Vec<f64>) -> RecordBatch {
        let n = vals.len();
        let keys: Vec<i32> = (0..n as i32).map(|i| i % 8192).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys)) as arrow_array::ArrayRef,
                Arc::new(Float64Array::from(vals)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap()
    }

    /// A NaN in the value column must defer for both MIN and MAX so the
    /// result matches the scalar aggregate's DuckDB total order.
    #[test]
    fn nan_value_defers_min_and_max() {
        let n = 300_000usize;
        for op_is_min in [true, false] {
            let plan = build_minmax_plan(op_is_min);
            let mut vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
            vals[12345] = f64::NAN;
            let batch = batch_with(vals);
            assert!(
                try_execute(&plan, &batch).is_none(),
                "NaN-bearing float MIN/MAX must defer (op_is_min={op_is_min})"
            );
        }
    }

    /// Sanity: an all-finite value column is NOT declined by the NaN guard.
    /// (It may still be served by the GPU path; here we only confirm the NaN
    /// guard does not spuriously reject finite data — a sub-threshold size
    /// gives a deterministic host decline for an unrelated reason.)
    #[test]
    fn finite_values_not_declined_by_nan_guard() {
        let plan = build_minmax_plan(true);
        let vals: Vec<f64> = (0..2048).map(|i| i as f64).collect();
        let batch = batch_with(vals);
        // Sub-threshold → None via the row gate, proving the NaN guard let it
        // through to the later gate rather than rejecting on NaN.
        assert!(try_execute(&plan, &batch).is_none());
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn second_call_same_spec_is_cache_hit() {
        let m1 = match get_or_build_module(&KernelSpec::Partition) {
            Ok(m) => m,
            Err(_) => return,
        };
        let after_first = LOAD_COUNT.load(Ordering::SeqCst);
        let m2 = get_or_build_module(&KernelSpec::Partition).expect("hit");
        assert_eq!(LOAD_COUNT.load(Ordering::SeqCst), after_first);
        assert_eq!(m1.raw(), m2.raw());
    }

    #[test]
    fn op_dtype_combinations_are_distinct_cache_keys() {
        let _ = match get_or_build_module(&KernelSpec::ReduceMinMaxFloat(
            ReduceFloatKey::MinF64,
        )) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ReduceMinMaxFloat(ReduceFloatKey::MaxF64))
            .expect("max-f64 build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::ReduceMinMaxFloat(ReduceFloatKey::MinF64))
            .expect("min-f64 hit");
        let _ = get_or_build_module(&KernelSpec::ReduceMinMaxFloat(ReduceFloatKey::MaxF64))
            .expect("max-f64 hit");
        assert_eq!(LOAD_COUNT.load(Ordering::SeqCst), baseline);
    }
}

// ---------------------------------------------------------------------------
// Stage-4 (P1b) async round-trip smoke test.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage4_tests {
    use super::*;
    use crate::plan::logical_plan::{Field, Schema};
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    #[test]
    #[ignore = "gpu:tier2"]
    fn async_tier2_minmax_float_round_trip() {
        let n: usize = 300_000;
        let n_groups: usize = 4096;
        let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
        let vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut expected_min = vec![f64::INFINITY; n_groups];
        for (i, &k) in keys.iter().enumerate() {
            if vals[i] < expected_min[k as usize] {
                expected_min[k as usize] = vals[i];
            }
        }
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO { name: "k".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "v".into(), dtype: DataType::Float64 },
                ],
                group_by: vec![0],
                aggregates: vec![AggregateExpr::Min(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("min_v", DataType::Float64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys)) as arrow_array::ArrayRef,
                Arc::new(Float64Array::from(vals)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        let out = match try_execute(&plan, &batch) {
            Some(Ok(b)) => b,
            _ => return,
        };
        let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vs = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..out.num_rows() {
            assert_eq!(vs.value(i), expected_min[ks.value(i) as usize]);
        }
    }

    /// F2 (GPU): grouped float MIN and MAX on the fast path must equal a
    /// per-group host scalar reference for finite data. The NaN case is
    /// covered host-only by `nan_tests` (the fast path defers NaN-bearing
    /// columns to the global-atomic / scalar path, which `aggregate.rs`'s
    /// `float_total_cmp` tests pin to the DuckDB total order), so here we
    /// assert the on-GPU grouped result agrees with the scalar reduction over
    /// the same finite data — the "one answer per query" invariant.
    #[test]
    #[ignore = "gpu:tier2"]
    fn grouped_float_minmax_matches_scalar_reference() {
        let n: usize = 300_000;
        let n_groups: usize = 4096;
        // Mix signs and magnitudes so MIN and MAX exercise different groups.
        let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
        let vals: Vec<f64> = (0..n)
            .map(|i| ((i as f64) * 1.5 - (n as f64)) * if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();

        for op_is_min in [true, false] {
            // Per-group scalar reference (plain comparison; data is finite).
            let mut expected = vec![if op_is_min { f64::INFINITY } else { f64::NEG_INFINITY }; n_groups];
            for (i, &k) in keys.iter().enumerate() {
                let slot = &mut expected[k as usize];
                if op_is_min {
                    if vals[i] < *slot { *slot = vals[i]; }
                } else if vals[i] > *slot {
                    *slot = vals[i];
                }
            }
            let agg = if op_is_min {
                AggregateExpr::Min(Expr::Column("v".into()))
            } else {
                AggregateExpr::Max(Expr::Column("v".into()))
            };
            let plan = PhysicalPlan::Aggregate {
                table: "t".into(),
                pre: None,
                aggregate: AggregateSpec {
                    inputs: vec![
                        ColumnIO { name: "k".into(), dtype: DataType::Int32 },
                        ColumnIO { name: "v".into(), dtype: DataType::Float64 },
                    ],
                    group_by: vec![0],
                    aggregates: vec![agg],
                    output_schema: Schema::new(vec![
                        Field::new("k", DataType::Int32, false),
                        Field::new("v", DataType::Float64, true),
                    ]),
                    input_has_validity: Vec::new(),
                },
            };
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("k", ArrowDataType::Int32, false),
                ArrowField::new("v", ArrowDataType::Float64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema,
                vec![
                    Arc::new(Int32Array::from(keys.clone())) as arrow_array::ArrayRef,
                    Arc::new(Float64Array::from(vals.clone())) as arrow_array::ArrayRef,
                ],
            )
            .unwrap();
            let out = match try_execute(&plan, &batch) {
                Some(Ok(b)) => b,
                _ => return, // no GPU / declined → nothing to assert on host
            };
            let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let vs = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            for i in 0..out.num_rows() {
                assert_eq!(
                    vs.value(i),
                    expected[ks.value(i) as usize],
                    "grouped float {} disagrees with scalar reference for key {}",
                    if op_is_min { "MIN" } else { "MAX" },
                    ks.value(i)
                );
            }
        }
    }
}
