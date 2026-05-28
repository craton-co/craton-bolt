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
//!
//! ## Scatter dispatch
//!
//! The scatter pass is dtype-specialised:
//!
//! * **Int32 vals** route through the f64-val scatter
//!   (`compile_scatter_kernel`). `i32 -> f64 -> i32` is bit-exact for
//!   every i32 (i32::MAX < 2^53), so there's no precision loss.
//! * **Int64 vals** route through the typed
//!   `compile_scatter_kernel_i32_to_i64` kernel. Vals stay in 64-bit
//!   integer registers end-to-end, so values with `|v| > 2^53`
//!   round-trip losslessly. This replaces the earlier C4 host-side
//!   `i64_values_exceed_f64_mantissa` guard — the fast lane is now
//!   exact across the full i64 range.

use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::module_cache;
use crate::exec::partition_offsets;
use crate::jit::partition_reduce_kernel_minmax::{
    compile_partition_reduce_kernel_minmax, kernel_entry as minmax_entry, MinMaxDtype, MinMaxOp,
    BLOCK_GROUPS, BLOCK_THREADS as REDUCE_BLOCK_THREADS,
};
use crate::jit::{partition_kernel, scatter_kernel, CudaModule};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Per-executor module cache. See `groupby_tier2_count_exec.rs` for the
// motivation and concurrency notes — the design is identical.
//
// The reduce kernel here is parameterised on `(MinMaxOp, MinMaxDtype)`. We
// mirror those as cache-key variants (rather than using the upstream types
// directly, which don't derive `Hash`).
//
// Scatter has two variants: the original f64-val sibling
// (`KernelSpec::Scatter`, used by the Int32 path) and the typed
// i32-key + i64-val variant (`KernelSpec::ScatterI32ToI64`, used by the
// Int64 path).
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
    Partition,
    Scatter,
    ScatterI32ToI64,
    ReduceMinMax(ReduceKey),
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
            KernelSpec::Scatter => scatter_kernel::compile_scatter_kernel()?,
            KernelSpec::ScatterI32ToI64 => scatter_kernel::compile_scatter_kernel_i32_to_i64()?,
            KernelSpec::ReduceMinMax(rk) => {
                let (op, dt) = rk.into_pair();
                compile_partition_reduce_kernel_minmax(op, dt)?
            }
        })
    })
}

/// Try the Tier-2.1 MIN/MAX fast path. Returns `None` on any miss.
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
) -> BoltResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;
    // Stage-4 (P1b): single per-call stream threaded through every
    // helper so all H2D / kernels / D2H share one ordering domain.
    let stream = CudaStream::null_or_default();
    let keys_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice_async(key_arr.values(), stream.raw())?;

    let num_partitions = partition_kernel::NUM_PARTITIONS;

    // Partition pass.
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;
    let partition_module = get_or_build_module(&KernelSpec::Partition)?;
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

    // ---- Scatter pass — dtype-specialised ----
    //
    // Int32 vals route through the f64-val scatter: `i32 -> f64 -> i32`
    // is exact for every i32 (i32::MAX < 2^53), so there's no precision
    // loss. Int64 vals route through the typed
    // `compile_scatter_kernel_i32_to_i64` kernel — vals stay in 64-bit
    // integer registers end-to-end, so values >2^53 round-trip
    // losslessly. This replaces the earlier f64 round-trip guard.
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
            // i32 -> f64 -> i32 is bit-exact for all i32.
            let v: Vec<i32> = scattered_f64.iter().map(|&x| x as i32).collect();
            let vals_typed_gpu_i32 = GpuVec::<i32>::from_slice_async(&v, stream.raw())?;
            run_reduce_phase(
                plan,
                op,
                val_dtype,
                vals_typed_gpu_i32,
                scatter_keys,
                offsets,
                num_partitions,
                &stream,
            )
        }
        MinMaxDtype::Int64 => {
            let arr = val_col
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| BoltError::Other("expected Int64Array".into()))?;
            let vals_in_gpu: GpuVec<i64> =
                GpuVec::<i64>::from_slice_async(arr.values(), stream.raw())?;

            let (scatter_keys, scatter_vals_i64) = run_scatter_i32_to_i64(
                &keys_gpu,
                &vals_in_gpu,
                &partition_ids,
                &offsets_gpu,
                n_rows,
                num_partitions,
                &stream,
            )?;
            run_reduce_phase_i64(
                plan,
                op,
                scatter_vals_i64,
                scatter_keys,
                offsets,
                num_partitions,
                &stream,
            )
        }
    }
}

/// Run the f64-val scatter kernel and return the scattered i32 keys +
/// host-side f64 vals. Used by the Int32 value path (round-trip is exact
/// for all i32). Module is fetched from the per-executor cache.
fn run_scatter_f64(
    keys_gpu: &GpuVec<i32>,
    vals_in_gpu: &GpuVec<f64>,
    partition_ids: &GpuVec<u32>,
    offsets_gpu: &GpuVec<u32>,
    n_rows: u32,
    num_partitions: u32,
    stream: &CudaStream,
) -> BoltResult<(GpuVec<i32>, Vec<f64>)> {
    let mut scatter_keys: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals_f64: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?;

    let scatter_module = get_or_build_module(&KernelSpec::Scatter)?;
    let func = scatter_module.function(scatter_kernel::KERNEL_ENTRY)?;
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

/// Run the typed i32-key + i64-val scatter kernel. Vals stay in 64-bit
/// integer registers end-to-end; no f64 round-trip. Used by the Int64
/// value path so values >2^53 are preserved exactly. Module is fetched
/// from the per-executor cache.
fn run_scatter_i32_to_i64(
    keys_gpu: &GpuVec<i32>,
    vals_in_gpu: &GpuVec<i64>,
    partition_ids: &GpuVec<u32>,
    offsets_gpu: &GpuVec<u32>,
    n_rows: u32,
    num_partitions: u32,
    stream: &CudaStream,
) -> BoltResult<(GpuVec<i32>, GpuVec<i64>)> {
    let mut scatter_keys: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals_i64: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_rows as usize, stream.raw())?;

    let scatter_module = get_or_build_module(&KernelSpec::ScatterI32ToI64)?;
    let func = scatter_module.function(scatter_kernel::KERNEL_ENTRY_I32_TO_I64)?;
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

/// Reduce phase for Int32 value dtype.
fn run_reduce_phase(
    plan: &PhysicalPlan,
    op: MinMaxOp,
    val_dtype: MinMaxDtype,
    vals_gpu: GpuVec<i32>,
    scatter_keys: GpuVec<i32>,
    offsets: Vec<u32>,
    num_partitions: u32,
    stream: &CudaStream,
) -> BoltResult<RecordBatch> {
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;
    let block_groups = BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals_gpu: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;

    let reduce_module =
        get_or_build_module(&KernelSpec::ReduceMinMax(ReduceKey::from_pair(op, val_dtype)))?;
    {
        let entry = minmax_entry(op, val_dtype);
        let func = reduce_module.function(&entry)?;

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

        launch_with_geometry(
            func,
            num_partitions,
            REDUCE_BLOCK_THREADS,
            0,
            stream,
            &mut args,
        )?;
    }

    // Stage-4 (P1b): pinned D2H for the three output buffers; sync once.
    let pinned_keys = out_keys_gpu.to_pinned_async(stream.raw())?;
    let pinned_vals = out_vals_gpu.to_pinned_async(stream.raw())?;
    let pinned_set = out_set_gpu.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_out_keys: Vec<i32> = pinned_keys.as_slice().to_vec();
    let host_out_vals: Vec<i32> = pinned_vals.as_slice().to_vec();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

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
        BoltError::Other(format!("groupby_tier2_minmax_exec(i32): build error: {e}"))
    })
}

fn run_reduce_phase_i64(
    plan: &PhysicalPlan,
    op: MinMaxOp,
    vals_gpu: GpuVec<i64>,
    scatter_keys: GpuVec<i32>,
    offsets: Vec<u32>,
    num_partitions: u32,
    stream: &CudaStream,
) -> BoltResult<RecordBatch> {
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;
    let block_groups = BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals_gpu: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;

    let reduce_module = get_or_build_module(&KernelSpec::ReduceMinMax(ReduceKey::from_pair(
        op,
        MinMaxDtype::Int64,
    )))?;
    {
        let entry = minmax_entry(op, MinMaxDtype::Int64);
        let func = reduce_module.function(&entry)?;

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

        launch_with_geometry(
            func,
            num_partitions,
            REDUCE_BLOCK_THREADS,
            0,
            stream,
            &mut args,
        )?;
    }

    // Stage-4 (P1b): pinned D2H for the three output buffers; sync once.
    let pinned_keys = out_keys_gpu.to_pinned_async(stream.raw())?;
    let pinned_vals = out_vals_gpu.to_pinned_async(stream.raw())?;
    let pinned_set = out_set_gpu.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_out_keys: Vec<i32> = pinned_keys.as_slice().to_vec();
    let host_out_vals: Vec<i64> = pinned_vals.as_slice().to_vec();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

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
        BoltError::Other(format!("groupby_tier2_minmax_exec(i64): build error: {e}"))
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
// Host-only wiring tests. End-to-end CUDA correctness is exercised in the
// integration suite; here we only guard the JIT call sites so a regression
// that re-routes Int64 through the lossy f64 sibling is caught early.
//
// The earlier `i64_values_exceed_f64_mantissa` predicate (and its
// inline tests) is deleted: it was a host-side decline gate that traded
// correctness for slowness on Int64 columns with values >2^53. The typed
// `ScatterI32ToI64` kernel is now exact across the full i64 range, so the
// gate is no longer load-bearing.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use crate::jit::scatter_kernel;

    /// The Int64 value path MUST be wired to the typed scatter kernel
    /// — i.e. `compile_scatter_kernel_i32_to_i64` produces PTX free of
    /// `.f64` instructions. A regression that re-routed Int64 through
    /// the f64 sibling would silently narrow any value with bit 53 set.
    #[test]
    fn int64_scatter_kernel_has_no_f64() {
        let ptx = scatter_kernel::compile_scatter_kernel_i32_to_i64()
            .expect("typed kernel compiles");
        assert!(
            !ptx.contains(".f64") && !ptx.contains("%fd"),
            "Int64 fast path must avoid f64: kernel PTX contains f64 references:\n{ptx}"
        );
        assert!(
            ptx.contains("ld.global.s64") && ptx.contains("st.global.u64"),
            "Int64 fast path must load/store vals via .s64/.u64:\n{ptx}"
        );
    }

    /// The Int32 value path still uses the f64-val sibling — i32 -> f64
    /// -> i32 round-trip is bit-exact for all i32 and we don't need a
    /// typed i32-val kernel for correctness.
    #[test]
    fn int32_scatter_kernel_remains_f64() {
        let ptx = scatter_kernel::compile_scatter_kernel().expect("kernel compiles");
        assert!(
            ptx.contains("st.global.f64"),
            "Int32 path still uses the f64-val scatter (bit-exact for i32 -> f64):\n{ptx}"
        );
    }

    /// Sanity: the typed kernel's entry symbol matches the literal the
    /// executor passes to `module.function(...)`. A typo here would
    /// surface as a launch-time `CUDA_ERROR_NOT_FOUND` rather than a
    /// compile error.
    #[test]
    fn typed_kernel_entry_symbol_matches() {
        assert_eq!(
            scatter_kernel::KERNEL_ENTRY_I32_TO_I64,
            "bolt_scatter_i32_to_i64"
        );
    }
}

// ---------------------------------------------------------------------------
// Module-cache mechanics tests. Skip on CPU-only hosts (no CUDA context).
// ---------------------------------------------------------------------------
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
        assert_eq!(
            LOAD_COUNT.load(Ordering::SeqCst),
            after_first,
            "repeat call must not increment LOAD_COUNT"
        );
        assert_eq!(m1.raw(), m2.raw(), "clones must share the same CUmodule");
    }

    #[test]
    fn op_dtype_combinations_are_distinct_cache_keys() {
        // Warm two distinct (op, dtype) reduce specs and verify subsequent
        // lookups don't recompile either.
        let _ = match get_or_build_module(&KernelSpec::ReduceMinMax(ReduceKey::MinI32)) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ReduceMinMax(ReduceKey::MaxI64))
            .expect("max-i64 build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::ReduceMinMax(ReduceKey::MinI32))
            .expect("min-i32 hit");
        let _ = get_or_build_module(&KernelSpec::ReduceMinMax(ReduceKey::MaxI64))
            .expect("max-i64 hit");
        assert_eq!(
            LOAD_COUNT.load(Ordering::SeqCst),
            baseline,
            "repeated lookups across distinct specs must hit the cache"
        );
    }

    /// The typed i32-key + i64-val scatter has its own cache slot,
    /// distinct from the f64-val scatter. Verify both can co-exist and
    /// neither recompiles on a repeat lookup.
    #[test]
    fn typed_i64_scatter_is_distinct_cache_key() {
        let _ = match get_or_build_module(&KernelSpec::Scatter) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ScatterI32ToI64)
            .expect("typed i64 scatter build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::Scatter).expect("f64 scatter hit");
        let _ = get_or_build_module(&KernelSpec::ScatterI32ToI64)
            .expect("typed i64 scatter hit");
        assert_eq!(
            LOAD_COUNT.load(Ordering::SeqCst),
            baseline,
            "the two scatter variants must occupy distinct cache slots"
        );
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
    #[ignore = "requires CUDA toolkit at runtime"]
    fn async_tier2_minmax_int32_round_trip() {
        let n: usize = 300_000;
        let n_groups: usize = 4096;
        let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
        let vals: Vec<i32> = (0..n).map(|i| i as i32).collect();
        let mut expected_min = vec![i32::MAX; n_groups];
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
                    ColumnIO { name: "v".into(), dtype: DataType::Int32 },
                ],
                group_by: vec![0],
                aggregates: vec![AggregateExpr::Min(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("min_v", DataType::Int32, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys)) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(vals)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        let out = match try_execute(&plan, &batch) {
            Some(Ok(b)) => b,
            _ => return,
        };
        let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vs = out.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..out.num_rows() {
            assert_eq!(vs.value(i), expected_min[ks.value(i) as usize]);
        }
    }
}
