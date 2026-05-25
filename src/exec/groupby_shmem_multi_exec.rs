// SPDX-License-Identifier: Apache-2.0

//! Per-block shared-memory pre-aggregation **executor** with multiple SUM
//! aggregates folded into a single kernel launch (Tier-1 extension).
//!
//! Sibling of [`crate::exec::groupby_shmem_exec`]: same shape, same
//! eligibility logic, but loosens the "exactly one SUM" rule to
//! "1..=`MAX_VALS` SUMs over distinct Float64 columns" and routes to the
//! multi-aggregate kernel emitter in [`crate::jit::shmem_multi_sum_kernel`].
//!
//! Why this exists: h2o.ai q2 (`SELECT id2, SUM(v1), SUM(v2) FROM x GROUP BY
//! id2`) and friends want N independent sums over the *same* GROUP BY. The
//! single-SUM executor would launch the kernel N times and rescan the key
//! column N times; this executor pays for one launch + one key load and
//! amortises the per-block shared-mem zero/merge cost across every output.
//!
//! v0 scope:
//!   - GROUP BY exactly one Int32 column
//!   - 1..=`MAX_VALS` aggregates, ALL `SUM(<bare-column>)` over distinct
//!     Float64 columns (aliasing the same column twice would still work
//!     correctness-wise, but adds zero value and falls back)
//!   - `max(key) + 1 <= BLOCK_GROUPS` (1024)
//!   - no `pre` kernel (no upstream filter / projection)
//!
//! This module is **not** wired into [`crate::exec::groupby::execute_groupby`]
//! — that's an integration step performed by another agent.

use std::sync::Arc;

use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{JavelinError, JavelinResult};
use crate::exec::groupby_shmem_dispatch::{
    dispatch, AggOp, DispatchInputs, GroupByStrategy,
};
use crate::exec::groupby_shmem_launch::{tune, TuneInputs};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::jit::shmem_multi_sum_kernel::{
    compile_shmem_multi_sum_kernel, kernel_entry, BLOCK_GROUPS, BLOCK_THREADS, MAX_VALS,
};
use crate::jit::CudaModule;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

/// Try to execute `plan` against `batch` via the per-block multi-SUM
/// fast path.
///
/// Returns `None` when any precondition fails — the caller MUST fall through
/// to the safe global-atomic path. Returns `Some(Err)` only on genuine GPU
/// failures encountered after we committed to the fast path.
pub fn try_execute(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<JavelinResult<RecordBatch>> {
    // --- Plan-shape eligibility ------------------------------------------
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
    if n_vals == 0 || n_vals as u32 > MAX_VALS {
        return None;
    }

    // The single group-by column must be Int32.
    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    // Every aggregate must be SUM(<bare column>) — and the column name must
    // resolve to a Float64 array in the input batch. We collect the column
    // names up-front so we can both look up arrays and validate dtypes in
    // one pass.
    let mut sum_col_names: Vec<&str> = Vec::with_capacity(n_vals);
    for agg in &aggregate.aggregates {
        let name = match agg {
            AggregateExpr::Sum(Expr::Column(n)) => n.as_str(),
            _ => return None,
        };
        sum_col_names.push(name);
    }

    // Look up the key column.
    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;

    // Look up each value column. Every aggregate must be Float64; the
    // multi-sum kernel only emits f64 loads/atomics. We resolve all of them
    // before touching the GPU so a missing/wrong-typed column fails fast on
    // the host.
    let mut val_arrs: Vec<&Float64Array> = Vec::with_capacity(n_vals);
    for name in &sum_col_names {
        let arr = batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())?;
        if arr.len() != key_arr.len() {
            return None;
        }
        val_arrs.push(arr);
    }

    let n_rows = key_arr.len();

    // --- Range check on keys ---------------------------------------------
    //
    // Same host-side scan as the single-SUM executor — cheap (~5 ms for
    // 10 M Int32) and bounds the kernel's output slot count to n_groups so
    // we can build the result row-by-row with slot index == key value.
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
        return Some(build_empty_result(plan));
    }
    let n_groups = max_key as u32 + 1;

    // --- Final dispatcher gate -------------------------------------------
    //
    // The shared dispatcher accepts a single AggOp; that's fine here
    // because every aggregate has the same op (SUM) and dtype (Float64),
    // so a single dispatch call covers them all.
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
    Some(execute_inner(plan, batch, key_arr, &val_arrs, n_groups))
}

/// All the fallible launch + result-marshal work, factored out so
/// `try_execute` cleanly returns `Option<Result<_>>`.
fn execute_inner(
    plan: &PhysicalPlan,
    _batch: &RecordBatch,
    key_arr: &Int32Array,
    val_arrs: &[&Float64Array],
    n_groups: u32,
) -> JavelinResult<RecordBatch> {
    let n_rows = key_arr.len();
    let n_vals = val_arrs.len() as u32;

    // --- Upload inputs ----------------------------------------------------
    //
    // We upload each value column independently; sharing a single buffer
    // would require concatenation (extra copy) for no kernel benefit since
    // the PTX issues independent `ld.global.f64` per aggregate anyway.
    let keys_gpu = GpuVec::<i32>::from_slice(key_arr.values())?;
    let mut vals_gpus: Vec<GpuVec<f64>> = Vec::with_capacity(val_arrs.len());
    for v in val_arrs {
        vals_gpus.push(GpuVec::<f64>::from_slice(v.values())?);
    }

    // One output buffer per aggregate, all sized to n_groups.
    let mut out_gpus: Vec<GpuVec<f64>> = Vec::with_capacity(val_arrs.len());
    for _ in 0..n_vals {
        out_gpus.push(GpuVec::<f64>::zeros(n_groups as usize)?);
    }

    // --- JIT + load the kernel --------------------------------------------
    let ptx = compile_shmem_multi_sum_kernel(n_vals)?;
    let module = CudaModule::from_ptx(&ptx)?;
    let entry = kernel_entry(n_vals);
    let function = module.function(&entry)?;

    // --- Launch params ----------------------------------------------------
    //
    // Multi-sum shared-mem footprint is `n_vals * BLOCK_GROUPS * 8 +
    // BLOCK_GROUPS` bytes (33 KiB at the cap), still under the portable
    // 48 KiB sm_70 budget the kernel emits as static `.shared` decls. The
    // tuner sees `bytes_per_acc_slot = n_vals * 8` so its informational
    // shared-bytes accounting matches the kernel.
    let tune_in = TuneInputs {
        n_rows: n_rows as u32,
        n_groups: BLOCK_GROUPS,
        bytes_per_acc_slot: 8 * n_vals,
        max_shared_per_block: None,
    };
    let params = tune(tune_in).map_err(|e| {
        JavelinError::Other(format!(
            "shmem_multi_exec: launch-param tuner refused: {e} \
             (n_rows={n_rows}, n_groups={n_groups}, n_vals={n_vals})"
        ))
    })?;

    // --- Build kernel argument list — CUDA-Oxide typed path -------------
    //
    // Param order matches the PTX emitter:
    //   keys_ptr, vals_0_ptr..vals_{N-1}_ptr,
    //   out_0_ptr..out_{N-1}_ptr, n_rows, n_groups
    //
    // Iterated `view()` / `view_mut()` calls on `Vec<GpuVec<T>>` work
    // because `KernelArgs::push_input/output` was relaxed to take
    // `&'b GpuView<'a, T>` where `'a: 'b` — the outer borrows can
    // come from anywhere as long as the inner GpuVec borrows outlive
    // the args list.
    let view_keys = keys_gpu.view();
    let views_vals: Vec<_> = vals_gpus.iter().map(|g| g.view()).collect();
    let mut views_out: Vec<_> = out_gpus.iter_mut().map(|g| g.view_mut()).collect();

    let mut args = KernelArgs::empty();
    args.push_input(&view_keys);
    for v in &views_vals {
        args.push_input(v);
    }
    for v in views_out.iter_mut() {
        args.push_output(v);
    }
    args.push_scalar_u32(n_rows as u32);
    args.push_scalar_u32(n_groups);

    let stream = CudaStream::null();
    launch_with_geometry(
        function,
        params.grid_blocks,
        params.block_threads,
        // Static shared-mem decls in the PTX — 0 for dynamic.
        0,
        &stream,
        &mut args,
    )?;

    // --- Download + build the output RecordBatch -------------------------
    //
    // One f64 vector per aggregate, plus a presence mask so empty groups
    // are omitted (matches SQL semantics: SUM over empty group = absent,
    // not 0).
    let mut host_sums_per_agg: Vec<Vec<f64>> = Vec::with_capacity(out_gpus.len());
    for og in &out_gpus {
        host_sums_per_agg.push(og.to_vec()?);
    }

    let mut present = vec![false; n_groups as usize];
    for &k in key_arr.values() {
        present[k as usize] = true;
    }

    // Build present-only output columns.
    let live_slots: Vec<usize> = present
        .iter()
        .enumerate()
        .filter_map(|(i, &p)| if p { Some(i) } else { None })
        .collect();

    let out_keys: Vec<i32> = live_slots.iter().map(|&s| s as i32).collect();
    let mut out_sum_cols: Vec<Vec<f64>> =
        (0..n_vals).map(|_| Vec::with_capacity(live_slots.len())).collect();
    for &slot in &live_slots {
        for (j, agg_vec) in host_sums_per_agg.iter().enumerate() {
            out_sum_cols[j].push(agg_vec[slot]);
        }
    }

    // --- Match the plan's output_schema ----------------------------------
    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;

    // Assemble columns: key first, then each SUM in the order they
    // appeared in `aggregate.aggregates`.
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(1 + out_sum_cols.len());
    columns.push(Arc::new(Int32Array::from(out_keys)) as ArrayRef);
    for col in out_sum_cols {
        columns.push(Arc::new(Float64Array::from(col)) as ArrayRef);
    }

    RecordBatch::try_new(arrow_schema, columns).map_err(|e| {
        JavelinError::Other(format!(
            "shmem_multi_exec: failed to build output RecordBatch: {e}"
        ))
    })
}

/// Build a 0-row output `RecordBatch` matching the plan's output schema.
/// Used when the input has 0 rows or only negative keys.
fn build_empty_result(plan: &PhysicalPlan) -> JavelinResult<RecordBatch> {
    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => {
            return Err(JavelinError::Other(
                "shmem_multi_exec::build_empty_result: non-Aggregate plan".into(),
            ))
        }
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    // The schema dictates one Int32 key column + n_vals Float64 sum
    // columns; emit an empty array per declared field rather than hard-
    // coding two arrays.
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(arrow_schema.fields().len());
    for field in arrow_schema.fields() {
        let arr: ArrayRef = match field.data_type() {
            ArrowDataType::Int32 => Arc::new(Int32Array::from(Vec::<i32>::new())),
            ArrowDataType::Float64 => Arc::new(Float64Array::from(Vec::<f64>::new())),
            other => {
                return Err(JavelinError::Other(format!(
                    "shmem_multi_exec::build_empty_result: unsupported output dtype {:?}",
                    other
                )));
            }
        };
        columns.push(arr);
    }
    RecordBatch::try_new(arrow_schema, columns)
        .map_err(|e| JavelinError::Other(format!("empty result build failed: {e}")))
}

// Silence "unused import" if BLOCK_THREADS ever stops being referenced
// inline; keeping the import documents the contract with the kernel.
#[allow(dead_code)]
const _BLOCK_THREADS_REF: u32 = BLOCK_THREADS;

// Local copy of the plan-schema -> Arrow-schema conversion. Every executor
// in this crate carries its own copy; consolidating them is a separate
// refactor.
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
