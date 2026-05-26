// SPDX-License-Identifier: Apache-2.0

//! Per-block shared-memory pre-aggregation **executor** (Tier 1 fast path).
//!
//! This module wires together three sibling-agent slices:
//!
//! - `crate::jit::shmem_sum_kernel`     — PTX emitter (kernel + entry point)
//! - `crate::exec::groupby_shmem_dispatch` — eligibility decision
//! - `crate::exec::groupby_shmem_launch`   — block/grid/shared-mem auto-tuner
//!
//! and turns them into a working `try_execute(plan, batch) -> Option<...>`
//! entry point. Returns `Some(Ok(batch))` on the fast path, `None` when the
//! query doesn't meet the (deliberately narrow) v0 preconditions so the
//! caller can fall through to the existing global-atomic GROUP BY path.
//!
//! v0 scope (matches the dispatcher):
//!   - GROUP BY exactly one Int32 column
//!   - exactly one aggregate: `SUM(<bare-column>)` where the column is Float64
//!   - no `pre` kernel (no upstream filter / projection)
//!   - `max(key) + 1 <= BLOCK_GROUPS` (1024)
//!
//! Anything wider, anything aliased, anything that materialises an
//! expression as the SUM input, anything with a filter: fall through.

use std::sync::Arc;

use arrow_array::{Float64Array, Int32Array, RecordBatch};

use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_shmem_dispatch::{
    dispatch, AggOp, DispatchInputs, GroupByStrategy,
};
use crate::exec::groupby_shmem_launch::{tune, TuneInputs};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::jit::shmem_sum_kernel::{
    compile_shmem_sum_kernel, BLOCK_GROUPS, BLOCK_THREADS, KERNEL_ENTRY,
};
use crate::jit::CudaModule;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

/// Try to execute `plan` against `batch` via the per-block shared-mem
/// fast path. Returns `None` if any precondition fails — the caller MUST
/// fall through to the safe path. Returns `Some(Err)` only for genuine GPU
/// failures encountered after we committed to the fast path.
pub fn try_execute(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<BoltResult<RecordBatch>> {
    // --- Plan-shape eligibility ------------------------------------------
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

    // The single group-by column must be Int32.
    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    // The single aggregate must be SUM over a bare Float64 column.
    let sum_col_name = match &aggregate.aggregates[0] {
        AggregateExpr::Sum(Expr::Column(name)) => name.as_str(),
        _ => return None,
    };

    // Look up both columns in the record batch (must be present + correct dtype).
    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    let val_arr = batch
        .column_by_name(sum_col_name)
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())?;
    if key_arr.len() != val_arr.len() {
        return None;
    }
    let n_rows = key_arr.len();

    // --- Range check on keys ---------------------------------------------
    //
    // The kernel handles overflow keys (>= BLOCK_GROUPS) via a direct
    // global atomic, but the cost benefit of the fast path is only there
    // if the bulk of keys hit the shared-mem accumulator. We additionally
    // demand `max(key) < BLOCK_GROUPS` so the output slot count is bounded
    // and we can build the result with the slot index == key.
    //
    // Host-side scan: ~5 ms for 10 M Int32 with the default loop. Fine.
    let mut max_key: i32 = -1;
    for &k in key_arr.values() {
        if k < 0 {
            return None; // negative keys never hash to a valid slot
        }
        if k > max_key {
            max_key = k;
        }
    }
    if max_key < 0 {
        // Empty input: emit empty output to match SQL semantics.
        return Some(build_empty_result(plan));
    }
    let n_groups = max_key as u32 + 1;

    // --- Final dispatcher gate -------------------------------------------
    let inputs = DispatchInputs {
        n_groups,
        n_rows: n_rows as u32,
        n_key_cols: 1,
        op: AggOp::Sum,
        value_dtype: DataType::Float64,
        key_dtype: DataType::Int32,
    };
    if dispatch(inputs) != GroupByStrategy::SharedMemPreAgg {
        return None;
    }

    // --- Commit to the fast path -----------------------------------------
    Some(execute_inner(plan, batch, key_arr, val_arr, n_groups))
}

/// All the fallible work, factored out so `try_execute` can return
/// `Option<Result<_>>` cleanly: anything past dispatch eligibility is an
/// honest engine failure, not a "preconditions not met".
fn execute_inner(
    plan: &PhysicalPlan,
    _batch: &RecordBatch,
    key_arr: &Int32Array,
    val_arr: &Float64Array,
    n_groups: u32,
) -> BoltResult<RecordBatch> {
    let n_rows = key_arr.len();

    // --- Upload inputs ----------------------------------------------------
    // We don't go through GpuTable here because this fast-path is currently
    // invoked from `execute_groupby` which takes a host RecordBatch. A future
    // refactor can short-circuit to on-device inputs.
    let keys_gpu = GpuVec::<i32>::from_slice(key_arr.values())?;
    let vals_gpu = GpuVec::<f64>::from_slice(val_arr.values())?;

    // Output buffer sized to slot count (== n_groups since we already
    // gated on max_key < BLOCK_GROUPS, and slot index == key value).
    let mut out_gpu = GpuVec::<f64>::zeros(n_groups as usize)?;

    // --- JIT + load the kernel (PTX cache hits after first run) -----------
    let ptx = compile_shmem_sum_kernel()?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(KERNEL_ENTRY)?;

    // --- Launch params (Agent 3's tuner) ---------------------------------
    let tune_in = TuneInputs {
        n_rows: n_rows as u32,
        // The kernel allocates BLOCK_GROUPS slots regardless of n_groups,
        // so the shared-mem footprint is fixed. We pass BLOCK_GROUPS so the
        // tuner sees the actual shared-mem requirement.
        n_groups: BLOCK_GROUPS,
        bytes_per_acc_slot: 8, // f64
        max_shared_per_block: None,
    };
    let params = tune(tune_in).map_err(|e| {
        BoltError::Other(format!(
            "shmem_exec: launch-param tuner refused: {e} \
             (n_rows={n_rows}, n_groups={n_groups})"
        ))
    })?;

    // --- Build kernel argument list — CUDA-Oxide typed path ---------------
    //
    // Borrow the GpuVecs as typed views. The view lifetimes tie the kernel
    // args (and therefore the launch) to the underlying allocations: the
    // borrow checker rejects dropping `keys_gpu` / `vals_gpu` / `out_gpu`
    // while `args` is live. This is the CUDA-Oxide discipline (see
    // `docs/ARCHITECTURE.md#memory-safety-cuda-oxide`) applied to a
    // kernel-launch site; the prior version of this function passed raw
    // `CUdeviceptr`s and relied on us not making a mistake.
    //
    // Kernel ABI:
    //   .param .u64 .ptr .global keys
    //   .param .u64 .ptr .global vals
    //   .param .u64 .ptr .global out
    //   .param .u32                n_rows
    //   .param .u32                n_groups
    let view_keys = keys_gpu.view();
    let view_vals = vals_gpu.view();
    let mut view_out = out_gpu.view_mut();

    let mut args = KernelArgs::empty();
    args.push_input(&view_keys);
    args.push_input(&view_vals);
    args.push_output(&mut view_out);
    args.push_scalar_u32(n_rows as u32);
    args.push_scalar_u32(n_groups);

    // --- Launch + sync ----------------------------------------------------
    //
    // shared-mem allocations are STATIC (declared at the PTX module scope),
    // so we pass 0 here regardless of `params.shared_bytes`. The tuner's
    // `shared_bytes` becomes load-bearing only if we switch to dynamic
    // shared memory.
    let stream = CudaStream::null();
    launch_with_geometry(
        function,
        params.grid_blocks,
        params.block_threads,
        0,
        &stream,
        &mut args,
    )?;

    // --- Download + build the output RecordBatch -------------------------
    let host_sums: Vec<f64> = out_gpu.to_vec()?;

    // The fast path only filled slots [0, n_groups). Build a presence map
    // by host-scanning the keys (cheap on a single Int32 column); a slot
    // with no rows must be omitted from the output to match SQL semantics
    // (SUM over an empty group is NULL / absent, not 0).
    let mut present = vec![false; n_groups as usize];
    for &k in key_arr.values() {
        present[k as usize] = true;
    }

    let mut out_keys: Vec<i32> = Vec::with_capacity(n_groups as usize);
    let mut out_sums: Vec<f64> = Vec::with_capacity(n_groups as usize);
    for (slot, &is_present) in present.iter().enumerate() {
        if is_present {
            out_keys.push(slot as i32);
            out_sums.push(host_sums[slot]);
        }
    }

    // --- Match the plan's output_schema ----------------------------------
    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;

    let key_array = Arc::new(Int32Array::from(out_keys));
    let sum_array = Arc::new(Float64Array::from(out_sums));

    RecordBatch::try_new(arrow_schema, vec![key_array, sum_array]).map_err(|e| {
        BoltError::Other(format!(
            "shmem_exec: failed to build output RecordBatch: {e}"
        ))
    })
}

/// Build a 0-row output `RecordBatch` matching the plan's output schema.
/// Used when the input has 0 rows or only negative keys (we treat the
/// latter as "no eligible groups" rather than as a hard error).
fn build_empty_result(plan: &PhysicalPlan) -> BoltResult<RecordBatch> {
    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => {
            return Err(BoltError::Other(
                "shmem_exec::build_empty_result: non-Aggregate plan".into(),
            ))
        }
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    let key_array = Arc::new(Int32Array::from(Vec::<i32>::new()));
    let sum_array = Arc::new(Float64Array::from(Vec::<f64>::new()));
    RecordBatch::try_new(arrow_schema, vec![key_array, sum_array])
        .map_err(|e| BoltError::Other(format!("empty result build failed: {e}")))
}

// Silence "unused import" if BLOCK_THREADS ever stops being referenced
// inline; keeping the import documents the contract with the kernel.
#[allow(dead_code)]
const _BLOCK_THREADS_REF: u32 = BLOCK_THREADS;

// Local copy of the plan-schema → Arrow-schema conversion. Every executor
// in this crate carries its own copy; consolidating them is a separate
// refactor.
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
