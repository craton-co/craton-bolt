// SPDX-License-Identifier: Apache-2.0

//! **Two-key MIN / MAX at Tier 2.1** — high-cardinality executor for
//! `SELECT a, b, {MIN,MAX}(v) FROM x GROUP BY a, b` over **integer** value
//! columns (Int32 / Int64).
//!
//! Mirror of [`crate::exec::groupby_tier2_minmax_exec`] adapted for the
//! i64-packed-two-key path. Both group-by columns are Int32 and packed
//! losslessly into a single i64 host-side (matching the convention in
//! `groupby.rs::pack_keys`); the on-device chain then treats them as a
//! single dense key column.
//!
//! Float MIN/MAX over two keys is handled by the sibling executor
//! [`crate::exec::groupby_tier2_twokey_minmax_float_exec`] — PTX has no
//! native `atom.shared.{min,max}.f{32,64}` on sm_70 and the float kernel
//! emits a CAS-loop instead.
//!
//! ## Algorithm
//!
//! 1. Pack `(k1, k2)` → `i64` host-side.
//! 2. Run `partition_kernel_i64` over the packed keys.
//! 3. Scatter pass — dtype-specialised:
//!    * Int32 vals: round-trip through `f64` via `scatter_kernel_i64`
//!      (`Int32 -> f64 -> Int32` is bit-exact for every i32).
//!    * Int64 vals: stay in 64-bit integer registers end-to-end via
//!      `scatter_kernel_i64_to_i64`, so values >2^53 round-trip
//!      losslessly — no narrowing.
//! 4. Run `partition_reduce_kernel_minmax_i64` with the integer vals.
//! 5. Walk slots, unpack `(key_hi, key_lo)`, sort by packed-i64 ASC,
//!    build the output RecordBatch.
//!
//! ## Scope (v0)
//!
//! - GROUP BY exactly two Int32 columns
//! - Exactly one aggregate, `MIN(<bare col>)` or `MAX(<bare col>)`
//! - Value dtype Int32 or Int64
//! - `n_rows >= 256 K`
//! - Combined key cardinality < 100 M (Tier-2 dispatcher cap)

use std::sync::Arc;

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::module_cache;
use crate::exec::partition_offsets;
use crate::jit::partition_reduce_kernel_minmax::{MinMaxDtype, MinMaxOp};
use crate::jit::partition_reduce_kernel_minmax_i64::{
    compile_partition_reduce_kernel_minmax_i64_with_spill,
    kernel_entry_with_spill as minmax_i64_entry, BLOCK_GROUPS,
    BLOCK_THREADS as REDUCE_BLOCK_THREADS,
};
use crate::jit::{partition_kernel_i64, scatter_kernel_i64, CudaModule};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Per-executor module cache. Mirror of `groupby_tier2_minmax_exec.rs` over
// the i64-key kernel variants.
//
// Scatter has two variants: the f64-val sibling (`KernelSpec::ScatterI64`,
// used by the Int32 path — `i32 -> f64 -> i32` is bit-exact) and the
// typed i64-key + i64-val variant (`KernelSpec::ScatterI64ToI64`, used by
// the Int64 path so values >2^53 round-trip losslessly).
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy)]
enum ReduceKey {
    MinI32,
    MaxI32,
    MinI64,
    MaxI64,
}

impl ReduceKey {
    fn from_pair(op: MinMaxOp, dt: MinMaxDtype) -> Self {
        match (op, dt) {
            (MinMaxOp::Min, MinMaxDtype::Int32) => ReduceKey::MinI32,
            (MinMaxOp::Max, MinMaxDtype::Int32) => ReduceKey::MaxI32,
            (MinMaxOp::Min, MinMaxDtype::Int64) => ReduceKey::MinI64,
            (MinMaxOp::Max, MinMaxDtype::Int64) => ReduceKey::MaxI64,
        }
    }

    fn into_pair(self) -> (MinMaxOp, MinMaxDtype) {
        match self {
            ReduceKey::MinI32 => (MinMaxOp::Min, MinMaxDtype::Int32),
            ReduceKey::MaxI32 => (MinMaxOp::Max, MinMaxDtype::Int32),
            ReduceKey::MinI64 => (MinMaxOp::Min, MinMaxDtype::Int64),
            ReduceKey::MaxI64 => (MinMaxOp::Max, MinMaxDtype::Int64),
        }
    }
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum KernelSpec {
    PartitionI64,
    PartitionI64ShmemStaging,
    ScatterI64,
    ScatterI64ToI64,
    ReduceMinMaxI64(ReduceKey),
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
            KernelSpec::PartitionI64 => partition_kernel_i64::compile_partition_kernel_i64()?,
            KernelSpec::PartitionI64ShmemStaging => partition_kernel_i64::compile_partition_kernel_i64_shmem_staging()?,
            KernelSpec::ScatterI64 => scatter_kernel_i64::compile_scatter_kernel_i64()?,
            KernelSpec::ScatterI64ToI64 => {
                scatter_kernel_i64::compile_scatter_kernel_i64_to_i64()?
            }
            KernelSpec::ReduceMinMaxI64(rk) => {
                // Batch 5: spill-counter-aware variant. The launch sites
                // resolve `kernel_entry_with_spill(op, dt)` and push a u32
                // spill counter as the trailing kernel arg.
                let (op, dt) = rk.into_pair();
                compile_partition_reduce_kernel_minmax_i64_with_spill(op, dt)?
            }
        })
    })
}

fn partition_i64_spec_for(n_rows: u32) -> KernelSpec {
    if n_rows < partition_kernel_i64::SHMEM_STAGING_MIN_ROWS {
        KernelSpec::PartitionI64
    } else {
        KernelSpec::PartitionI64ShmemStaging
    }
}


/// Try the two-key Tier-2.1 integer MIN/MAX fast path. `None` on any
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
    if aggregate.group_by.len() != 2 || aggregate.aggregates.len() != 1 {
        return None;
    }

    // Single MIN or MAX over a bare column.
    let (op, val_col_name) = match &aggregate.aggregates[0] {
        AggregateExpr::Min(Expr::Column(n)) => (MinMaxOp::Min, n.as_str()),
        AggregateExpr::Max(Expr::Column(n)) => (MinMaxOp::Max, n.as_str()),
        _ => return None,
    };

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

    // Value: Int32 or Int64. Float routes through the sibling float exec.
    let val_col = batch.column_by_name(val_col_name)?;
    let val_dtype = match val_col.data_type() {
        ArrowDataType::Int32 => MinMaxDtype::Int32,
        ArrowDataType::Int64 => MinMaxDtype::Int64,
        _ => return None,
    };

    if val_col.len() != k1.len() {
        return None;
    }
    let n_rows = k1.len();
    if n_rows < 256 * 1024 {
        return None;
    }
    if n_rows >= 100_000_000 {
        return None;
    }

    // PV-stage-f: NULL handling — the partition_reduce_kernel_minmax_i64
    // family has no `_with_validity` companion. Defer NULL-bearing
    // batches back to the no-pre single-key paths. Stage G follow-up.
    if k1.null_count() > 0 || k2.null_count() > 0 || val_col.null_count() > 0 {
        return None;
    }

    Some(execute_inner(plan, k1, k2, val_col, op, val_dtype))
}

fn execute_inner(
    plan: &PhysicalPlan,
    k1: &Int32Array,
    k2: &Int32Array,
    val_col: &dyn arrow_array::Array,
    op: MinMaxOp,
    val_dtype: MinMaxDtype,
) -> BoltResult<RecordBatch> {
    let n_rows = k1.len() as u32;

    // Stage-4 (P1b): per-call stream threaded through every helper.
    let stream = CudaStream::null_or_default();

    // ---- Host-side pack ----
    // `(k1 << 32) | (k2 & 0xFFFF_FFFF)`. Matches `groupby.rs::pack_keys`.
    let packed: Vec<i64> = k1
        .values()
        .iter()
        .zip(k2.values().iter())
        .map(|(&a, &b)| ((a as u32 as u64) << 32 | (b as u32 as u64)) as i64)
        .collect();
    let keys_gpu: GpuVec<i64> = GpuVec::<i64>::from_slice_async(&packed, stream.raw())?;

    let num_partitions = partition_kernel_i64::NUM_PARTITIONS;

    // ---- Partition pass (i64) ----
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;
    let partition_module = get_or_build_module(&partition_i64_spec_for(n_rows))?;
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

    // ---- Scatter pass — dtype-specialised ----
    //
    // Int32 vals route through the f64-val scatter (round-trip is exact
    // for every i32). Int64 vals route through the typed
    // `bolt_scatter_i64_to_i64` kernel — vals stay in 64-bit integer
    // registers end-to-end, so values >2^53 round-trip losslessly. The
    // previous f64 round-trip narrowed Int64 above the f64 mantissa
    // boundary; the typed path is exact for the full i64 range.
    match val_dtype {
        MinMaxDtype::Int32 => {
            let arr = val_col
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| BoltError::Other("expected Int32Array".into()))?;
            let host_vals_f64: Vec<f64> = arr.values().iter().map(|&v| v as f64).collect();
            let vals_in_gpu: GpuVec<f64> =
                GpuVec::<f64>::from_slice_async(&host_vals_f64, stream.raw())?;

            let (scatter_keys, scattered_f64) = run_scatter_f64(
                &keys_gpu,
                &vals_in_gpu,
                &partition_ids,
                &offsets_gpu,
                n_rows,
                num_partitions,
                &stream,
            )?;
            let v: Vec<i32> = scattered_f64.iter().map(|&x| x as i32).collect();
            let vals_typed_gpu = GpuVec::<i32>::from_slice_async(&v, stream.raw())?;
            run_reduce_phase_i32(plan, op, vals_typed_gpu, scatter_keys, offsets, num_partitions, &stream)
        }
        MinMaxDtype::Int64 => {
            let arr = val_col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| BoltError::Other("expected Int64Array".into()))?;
            let vals_in_gpu: GpuVec<i64> =
                GpuVec::<i64>::from_slice_async(arr.values(), stream.raw())?;

            let (scatter_keys, scatter_vals_i64) = run_scatter_i64_to_i64(
                &keys_gpu,
                &vals_in_gpu,
                &partition_ids,
                &offsets_gpu,
                n_rows,
                num_partitions,
                &stream,
            )?;
            run_reduce_phase_i64(plan, op, scatter_vals_i64, scatter_keys, offsets, num_partitions, &stream)
        }
    }
}

/// Scatter via the f64-val sibling. Returns the scattered i64 keys plus
/// a host-side copy of the scattered f64 vals so the caller can re-cast
/// to the integer dtype before the reduce. Module is fetched from the
/// per-executor cache.
fn run_scatter_f64(
    keys_gpu: &GpuVec<i64>,
    vals_in_gpu: &GpuVec<f64>,
    partition_ids: &GpuVec<u32>,
    offsets_gpu: &GpuVec<u32>,
    n_rows: u32,
    num_partitions: u32,
    stream: &CudaStream,
) -> BoltResult<(GpuVec<i64>, Vec<f64>)> {
    let mut scatter_keys: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals_f64: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?;

    let scatter_module = get_or_build_module(&KernelSpec::ScatterI64)?;
    let func = scatter_module.function(scatter_kernel_i64::KERNEL_ENTRY)?;
    let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;

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
    launch_with_geometry(func, grid, BLOCK_THREADS, 0, stream, &mut args)?;

    // Stage-4 (P1b): pinned D2H for the scattered f64s.
    let pinned = scatter_vals_f64.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let scattered_f64: Vec<f64> = pinned.as_slice().to_vec();
    Ok((scatter_keys, scattered_f64))
}

/// Typed i64-key + i64-val scatter. Returns scattered i64 keys and the
/// scattered i64 vals (on device). No f64 round-trip; values >2^53 are
/// preserved exactly. Module is fetched from the per-executor cache.
fn run_scatter_i64_to_i64(
    keys_gpu: &GpuVec<i64>,
    vals_in_gpu: &GpuVec<i64>,
    partition_ids: &GpuVec<u32>,
    offsets_gpu: &GpuVec<u32>,
    n_rows: u32,
    num_partitions: u32,
    stream: &CudaStream,
) -> BoltResult<(GpuVec<i64>, GpuVec<i64>)> {
    let mut scatter_keys: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals_i64: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_rows as usize, stream.raw())?;

    let scatter_module = get_or_build_module(&KernelSpec::ScatterI64ToI64)?;
    let func = scatter_module.function(scatter_kernel_i64::KERNEL_ENTRY_I64_TO_I64)?;
    let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;

    let view_keys = keys_gpu.view();
    let view_vals = vals_in_gpu.view();
    let view_pids = partition_ids.view();
    let view_offsets = offsets_gpu.view();
    let mut view_cursors = cursors.view_mut();
    let mut view_sk = scatter_keys.view_mut();
    let mut view_sv = scatter_vals_i64.view_mut();

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
    launch_with_geometry(func, grid, BLOCK_THREADS, 0, stream, &mut args)?;

    Ok((scatter_keys, scatter_vals_i64))
}

/// Reduce phase for Int32 value dtype (i64-key).
fn run_reduce_phase_i32(
    plan: &PhysicalPlan,
    op: MinMaxOp,
    vals_gpu: GpuVec<i32>,
    scatter_keys: GpuVec<i64>,
    offsets: Vec<u32>,
    num_partitions: u32,
    stream: &CudaStream,
) -> BoltResult<RecordBatch> {
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;
    let block_groups = BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals_gpu: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;
    let mut spill: GpuVec<u32> = GpuVec::<u32>::zeros_async(1, stream.raw())?;

    let reduce_module = get_or_build_module(&KernelSpec::ReduceMinMaxI64(
        ReduceKey::from_pair(op, MinMaxDtype::Int32),
    ))?;
    {
        let entry = minmax_i64_entry(op, MinMaxDtype::Int32);
        let func = reduce_module.function(&entry)?;

        let view_pk = scatter_keys.view();
        let view_pv = vals_gpu.view();
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

        launch_with_geometry(func, num_partitions, REDUCE_BLOCK_THREADS, 0, stream, &mut args)?;
    }

    // Stage-4 (P1b): pinned D2H; sync once.
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
    let host_out_keys: Vec<i64> = pinned_keys.as_slice().to_vec();
    let host_out_vals: Vec<i32> = pinned_vals.as_slice().to_vec();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

    let mut rows: Vec<(i64, i32)> = Vec::new();
    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] != 0 {
                rows.push((host_out_keys[idx], host_out_vals[idx]));
            }
        }
    }
    rows.sort_by_key(|(k, _)| *k);

    let mut out_k1: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_k2: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_v: Vec<i32> = Vec::with_capacity(rows.len());
    for (k, v) in rows {
        let u = k as u64;
        out_k1.push((u >> 32) as u32 as i32);
        out_k2.push((u & 0xFFFF_FFFF) as u32 as i32);
        out_v.push(v);
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
            Arc::new(Int32Array::from(out_v)),
        ],
    )
    .map_err(|e| {
        BoltError::Other(format!(
            "groupby_tier2_twokey_minmax_exec(i32): build error: {e}"
        ))
    })
}

/// Reduce phase for Int64 value dtype (i64-key).
fn run_reduce_phase_i64(
    plan: &PhysicalPlan,
    op: MinMaxOp,
    vals_gpu: GpuVec<i64>,
    scatter_keys: GpuVec<i64>,
    offsets: Vec<u32>,
    num_partitions: u32,
    stream: &CudaStream,
) -> BoltResult<RecordBatch> {
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;
    let block_groups = BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals_gpu: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;
    let mut spill: GpuVec<u32> = GpuVec::<u32>::zeros_async(1, stream.raw())?;

    let reduce_module = get_or_build_module(&KernelSpec::ReduceMinMaxI64(
        ReduceKey::from_pair(op, MinMaxDtype::Int64),
    ))?;
    {
        let entry = minmax_i64_entry(op, MinMaxDtype::Int64);
        let func = reduce_module.function(&entry)?;

        let view_pk = scatter_keys.view();
        let view_pv = vals_gpu.view();
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

        launch_with_geometry(func, num_partitions, REDUCE_BLOCK_THREADS, 0, stream, &mut args)?;
    }

    // Stage-4 (P1b): pinned D2H; sync once.
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
    let host_out_keys: Vec<i64> = pinned_keys.as_slice().to_vec();
    let host_out_vals: Vec<i64> = pinned_vals.as_slice().to_vec();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

    let mut rows: Vec<(i64, i64)> = Vec::new();
    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] != 0 {
                rows.push((host_out_keys[idx], host_out_vals[idx]));
            }
        }
    }
    rows.sort_by_key(|(k, _)| *k);

    let mut out_k1: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_k2: Vec<i32> = Vec::with_capacity(rows.len());
    let mut out_v: Vec<i64> = Vec::with_capacity(rows.len());
    for (k, v) in rows {
        let u = k as u64;
        out_k1.push((u >> 32) as u32 as i32);
        out_k2.push((u & 0xFFFF_FFFF) as u32 as i32);
        out_v.push(v);
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
            Arc::new(Int64Array::from(out_v)),
        ],
    )
    .map_err(|e| {
        BoltError::Other(format!(
            "groupby_tier2_twokey_minmax_exec(i64): build error: {e}"
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
        // v0.6 / M4: Date/Timestamp not yet wired through this aggregate
        // output helper. Reject so a regression is loud.
        DataType::Date32 | DataType::Timestamp(_, _) => Err(crate::error::BoltError::Type(
            format!("Date/Timestamp not yet supported in this aggregate output path: {:?}", d),
        )),
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
// Host-only wiring tests. End-to-end CUDA correctness lives in the
// integration suite. These tests guard the JIT call sites so a regression
// that re-routes Int64 through the lossy f64 sibling is caught early.
//
// The earlier f64-mantissa decline guard (and its inline tests) is deleted:
// the typed `ScatterI64ToI64` kernel is exact across the full i64 range,
// so the guard is no longer load-bearing.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use crate::jit::scatter_kernel_i64;

    /// The Int64 two-key path MUST be wired to the typed i64-val
    /// scatter — the kernel PTX must contain no `.f64` references.
    #[test]
    fn int64_twokey_scatter_kernel_has_no_f64() {
        let ptx = scatter_kernel_i64::compile_scatter_kernel_i64_to_i64()
            .expect("typed kernel compiles");
        assert!(
            !ptx.contains(".f64") && !ptx.contains("%fd"),
            "Int64 two-key fast path must avoid f64:\n{ptx}"
        );
        // Two .s64 loads (one for key, one for val).
        assert!(
            ptx.matches("ld.global.s64").count() >= 2,
            "typed kernel must load both key and val via ld.global.s64:\n{ptx}"
        );
        // Two .u64 stores (one for key, one for val).
        assert!(
            ptx.matches("st.global.u64").count() >= 2,
            "typed kernel must store both key and val via st.global.u64:\n{ptx}"
        );
    }

    /// The Int32 two-key path still uses the f64-val sibling — round-trip
    /// is bit-exact for i32.
    #[test]
    fn int32_twokey_scatter_kernel_remains_f64() {
        let ptx =
            scatter_kernel_i64::compile_scatter_kernel_i64().expect("kernel compiles");
        assert!(
            ptx.contains("st.global.f64"),
            "Int32 two-key path still uses the f64-val scatter:\n{ptx}"
        );
    }

    /// Sanity: the typed kernel's entry symbol matches the literal the
    /// executor passes to `module.function(...)`.
    #[test]
    fn typed_twokey_kernel_entry_symbol_matches() {
        assert_eq!(
            scatter_kernel_i64::KERNEL_ENTRY_I64_TO_I64,
            "bolt_scatter_i64_to_i64"
        );
    }
}

// ---------------------------------------------------------------------------
// Host-only eligibility-gate tests for the two-key integer MIN/MAX exec.
//
// Pure host tests; the actual MIN/MAX numerical correctness lives in the
// e2e suite where a real CUDA context is available.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod eligibility_tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    /// Plan for `SELECT k1, k2, MIN(v) FROM t GROUP BY k1, k2`. The
    /// op argument lets a test swap to MAX without rebuilding.
    fn build_twokey_minmax_plan(op_is_min: bool, val_dtype: DataType) -> PhysicalPlan {
        let inputs = vec![
            ColumnIO {
                name: "k1".into(),
                dtype: DataType::Int32,
            },
            ColumnIO {
                name: "k2".into(),
                dtype: DataType::Int32,
            },
            ColumnIO {
                name: "v".into(),
                dtype: val_dtype,
            },
        ];
        let agg = if op_is_min {
            AggregateExpr::Min(Expr::Column("v".into()))
        } else {
            AggregateExpr::Max(Expr::Column("v".into()))
        };
        let output_schema = Schema::new(vec![
            Field::new("k1", DataType::Int32, false),
            Field::new("k2", DataType::Int32, false),
            Field::new("v", val_dtype, true),
        ]);
        PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs,
                group_by: vec![0, 1],
                aggregates: vec![agg],
                output_schema,
                input_has_validity: Vec::new(),
            },
        }
    }

    /// `(k1, k2, v)` batch with `n` rows.
    fn twokey_minmax_batch_i32(n: usize) -> RecordBatch {
        let k1: Vec<i32> = (0..n as i32).collect();
        let k2: Vec<i32> = (0..n as i32).map(|i| i + 1).collect();
        let v: Vec<i32> = (0..n as i32).map(|i| i * 2).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Int32, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k1)) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(k2)) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(v)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap()
    }

    /// Non-Aggregate plan: reject.
    #[test]
    fn rejects_non_aggregate_plan() {
        let plan = PhysicalPlan::Union { inputs: vec![] };
        let batch = twokey_minmax_batch_i32(0);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Single-key plan belongs to the single-key sibling.
    #[test]
    fn rejects_single_key_plan() {
        let mut plan = build_twokey_minmax_plan(true, DataType::Int32);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.group_by = vec![0];
        }
        let batch = twokey_minmax_batch_i32(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Two aggregates → multi-agg territory.
    #[test]
    fn rejects_two_aggregates() {
        let mut plan = build_twokey_minmax_plan(true, DataType::Int32);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.aggregates.push(AggregateExpr::Max(Expr::Column("v".into())));
        }
        let batch = twokey_minmax_batch_i32(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// SUM / COUNT / AVG aggregates → reject.
    #[test]
    fn rejects_sum_aggregate() {
        let mut plan = build_twokey_minmax_plan(true, DataType::Int32);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.aggregates = vec![AggregateExpr::Sum(Expr::Column("v".into()))];
        }
        let batch = twokey_minmax_batch_i32(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// MIN over Float64 → goes to the float-minmax sibling, not this exec.
    #[test]
    fn rejects_float_value_column() {
        let plan = build_twokey_minmax_plan(true, DataType::Float64);
        let n = 300_000;
        let k: Vec<i32> = (0..n as i32).collect();
        let v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k.clone())) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(k)) as arrow_array::ArrayRef,
                Arc::new(arrow_array::Float64Array::from(v)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Below the row threshold → defer.
    #[test]
    fn rejects_below_row_threshold() {
        let plan = build_twokey_minmax_plan(false, DataType::Int32);
        let batch = twokey_minmax_batch_i32(2_048);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// `pre` kernel present → with-pre executor handles this.
    #[test]
    fn rejects_plan_with_pre_kernel() {
        use crate::plan::physical_plan::KernelSpec;
        let mut plan = build_twokey_minmax_plan(true, DataType::Int32);
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
        let batch = twokey_minmax_batch_i32(300_000);
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
    fn op_dtype_combinations_are_distinct_cache_keys() {
        let _ = match get_or_build_module(&KernelSpec::ReduceMinMaxI64(ReduceKey::MinI32)) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ReduceMinMaxI64(ReduceKey::MaxI64))
            .expect("max-i64 build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::ReduceMinMaxI64(ReduceKey::MinI32))
            .expect("min-i32 hit");
        let _ = get_or_build_module(&KernelSpec::ReduceMinMaxI64(ReduceKey::MaxI64))
            .expect("max-i64 hit");
        assert_eq!(LOAD_COUNT.load(Ordering::SeqCst), baseline);
    }

    /// The typed i64-key + i64-val scatter has its own cache slot,
    /// distinct from the f64-val scatter. Verify both can co-exist and
    /// neither recompiles on a repeat lookup.
    #[test]
    fn typed_i64_to_i64_scatter_is_distinct_cache_key() {
        let _ = match get_or_build_module(&KernelSpec::ScatterI64) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ScatterI64ToI64)
            .expect("typed i64-to-i64 scatter build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::ScatterI64).expect("f64 scatter hit");
        let _ = get_or_build_module(&KernelSpec::ScatterI64ToI64)
            .expect("typed i64-to-i64 scatter hit");
        assert_eq!(
            LOAD_COUNT.load(Ordering::SeqCst),
            baseline,
            "the two scatter variants must occupy distinct cache slots"
        );
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
    fn async_tier2_twokey_minmax_round_trip() {
        let n: usize = 300_000;
        let k1: Vec<i32> = (0..n as i32).map(|i| i % 64).collect();
        let k2: Vec<i32> = (0..n as i32).map(|i| (i / 64) % 64).collect();
        let v: Vec<i32> = (0..n as i32).collect();
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO { name: "k1".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "k2".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "v".into(), dtype: DataType::Int32 },
                ],
                group_by: vec![0, 1],
                aggregates: vec![AggregateExpr::Min(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("k1", DataType::Int32, false),
                    Field::new("k2", DataType::Int32, false),
                    Field::new("min_v", DataType::Int32, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(k1)) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(k2)) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(v)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        let _ = try_execute(&plan, &batch);
    }
}
