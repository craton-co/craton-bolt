// SPDX-License-Identifier: Apache-2.0

//! Per-block shared-memory **AVG** executor (Tier 1 fast path, AVG flavour).
//!
//! `AVG(col)` decomposes into `SUM(col) / COUNT(col)`. On the GPU we already
//! have a tuned shared-mem SUM kernel
//! ([`crate::jit::shmem_sum_kernel`]); this module adds a sibling COUNT
//! kernel ([`crate::jit::shmem_count_kernel`]) and orchestrates them
//! together so an h2o.ai-q4-shaped query
//!
//! ```sql
//! SELECT id1, AVG(v1), AVG(v2), AVG(v3) FROM x GROUP BY id1
//! ```
//!
//! gets the same per-block-shared-mem speedup as q1/q4-sum.
//!
//! ## Why a separate executor (rather than extending `groupby_shmem_exec`)
//!
//! The single-SUM fast path is intentionally narrow — one aggregate, no
//! division, output schema `(key, sum)`. Bolting AVG into it would have
//! meant two divergent return shapes from one function and a bigger blast
//! radius if either path regressed. The AVG-specific executor lets the
//! dispatch (in `execute_groupby`) try AVG-only first, then SUM-only,
//! then fall through to the global-atomic safe path — each gate is a
//! clean rejection that costs at most a couple of `match` arms.
//!
//! ## Scope (v0)
//!
//! Accepts iff **all** of:
//!
//! 1. The plan is `Aggregate { pre: None, .. }` over a single Int32 key.
//! 2. `1 <= aggregate.aggregates.len() <= 4`, and *every* aggregate is
//!    `AVG(<bare Float64 column>)`. Mixed AVG/SUM/COUNT is rejected (it
//!    deserves a dedicated multi-agg executor; we don't paper over it
//!    here).
//! 3. `max(key) < BLOCK_GROUPS (1024)` and `n_rows >= 64K`.
//!
//! ## Algorithm
//!
//! * For each AVG aggregate `i`: launch the SUM kernel against `vals_i`
//!   to produce `sums_i[g]` for `g in 0..n_groups`.
//! * Exactly **once**: launch the COUNT kernel (using just the key column;
//!   COUNT(*) doesn't need the values) to produce `counts[g]`.
//! * Host-side: build the output by walking present groups and computing
//!   `avg_i[g] = sums_i[g] / counts[g] as f64`. Groups with `counts[g] == 0`
//!   are *omitted entirely* — same SQL semantics the single-SUM executor
//!   already follows (an empty group has no row in the result).
//!
//! Sharing one COUNT pass across all N AVGs is the load-bearing optimisation
//! here: q4 has three AVGs and we only count keys once.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::GpuVec;
use crate::error::{PatinaError, PatinaResult};
use crate::exec::groupby_shmem_launch::{tune, TuneInputs};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::jit::shmem_count_kernel::{
    compile_shmem_count_kernel, KERNEL_ENTRY as COUNT_ENTRY,
};
use crate::jit::shmem_sum_kernel::{
    compile_shmem_sum_kernel, BLOCK_GROUPS, KERNEL_ENTRY as SUM_ENTRY,
};
use crate::jit::CudaModule;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Schema};
use crate::plan::physical_plan::PhysicalPlan;

/// Minimum input-row count to take the fast path. Matches
/// [`crate::exec::groupby_shmem_dispatch::SHARED_MEM_MIN_ROWS`] so the
/// AVG and SUM paths flip on/off at the same threshold.
const MIN_ROWS_FAST_PATH: usize = 64 * 1024;

/// Hard cap on the number of AVG aggregates per query. With four AVGs we
/// already need four SUM passes plus the (one) COUNT pass — five kernel
/// launches. Beyond that the per-kernel overhead starts to eat into the
/// shared-mem-vs-global-atomic win, and a multi-aggregate kernel becomes
/// the right answer (Tier 2).
const MAX_AVG_AGGS: usize = 4;

/// Try to execute `plan` against `batch` via the per-block shared-mem AVG
/// fast path. Returns `None` on any precondition miss — the caller MUST
/// fall through to a safe path. `Some(Err(_))` is reserved for genuine
/// GPU failures encountered *after* eligibility was committed to.
pub fn try_execute(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<PatinaResult<RecordBatch>> {
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
    let n_aggs = aggregate.aggregates.len();
    if n_aggs == 0 || n_aggs > MAX_AVG_AGGS {
        return None;
    }

    // The single group-by column must be Int32.
    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    // Every aggregate must be `AVG(<bare Float64 column>)`. Mixed agg
    // kinds OR a non-Float64 input fall through (same defensive bar the
    // single-SUM path uses).
    let mut avg_col_names: Vec<&str> = Vec::with_capacity(n_aggs);
    for agg in &aggregate.aggregates {
        match agg {
            AggregateExpr::Avg(Expr::Column(name)) => avg_col_names.push(name.as_str()),
            _ => return None,
        }
    }

    // Look up the key column once; reject if missing / wrong type / etc.
    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    let n_rows = key_arr.len();

    // Bail before allocating anything if the input is too small.
    if n_rows < MIN_ROWS_FAST_PATH {
        return None;
    }

    // Look up each AVG value column. They must all be Float64 *and* the
    // same length as the key column.
    let mut val_arrs: Vec<&Float64Array> = Vec::with_capacity(n_aggs);
    for col_name in &avg_col_names {
        let arr = batch
            .column_by_name(col_name)
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())?;
        if arr.len() != n_rows {
            return None;
        }
        val_arrs.push(arr);
    }

    // --- Key range check (matches groupby_shmem_exec) --------------------
    // Reject negative keys and any key >= BLOCK_GROUPS: the latter would
    // route through the kernel's overflow path which still works but
    // erodes the AVG-fast-path's reason for existing. The threshold is
    // identical to the SUM executor's so behaviour stays predictable.
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
        // Empty / no rows: produce an empty matching-schema batch.
        return Some(build_empty_result(plan));
    }
    let n_groups = max_key as u32 + 1;
    if n_groups > BLOCK_GROUPS {
        return None;
    }

    // --- Commit ----------------------------------------------------------
    Some(execute_inner(
        plan, key_arr, &val_arrs, n_groups,
    ))
}

/// Run the AVG fast path. Everything past this point is "we promised";
/// errors here are real GPU failures, not eligibility misses.
fn execute_inner(
    plan: &PhysicalPlan,
    key_arr: &Int32Array,
    val_arrs: &[&Float64Array],
    n_groups: u32,
) -> PatinaResult<RecordBatch> {
    let n_rows = key_arr.len();
    let n_aggs = val_arrs.len();

    // --- Upload key column ONCE; reused by every SUM + the one COUNT. ----
    let keys_gpu = GpuVec::<i32>::from_slice(key_arr.values())?;

    // --- JIT both kernels (PTX cache hits after first run) ---------------
    let sum_ptx = compile_shmem_sum_kernel()?;
    let sum_module = CudaModule::from_ptx(&sum_ptx)?;
    let sum_function = sum_module.function(SUM_ENTRY)?;

    let count_ptx = compile_shmem_count_kernel()?;
    let count_module = CudaModule::from_ptx(&count_ptx)?;
    let count_function = count_module.function(COUNT_ENTRY)?;

    // --- Launch params (shared between SUM and COUNT — same shape) -------
    //
    // The SUM and COUNT kernels both use BLOCK_GROUPS slots of static
    // shared memory; sizing the tuner with `bytes_per_acc_slot = 8` covers
    // both (the COUNT accumulator is u64 = 8 B per slot, same as f64).
    let tune_in = TuneInputs {
        n_rows: n_rows as u32,
        n_groups: BLOCK_GROUPS,
        bytes_per_acc_slot: 8,
        max_shared_per_block: None,
    };
    let params = tune(tune_in).map_err(|e| {
        PatinaError::Other(format!(
            "shmem_avg_exec: launch-param tuner refused: {e} \
             (n_rows={n_rows}, n_groups={n_groups})"
        ))
    })?;

    let stream = CudaStream::null();
    let n_rows_u32 = n_rows as u32;
    let n_groups_u32 = n_groups;

    // --- Pass 1..N: SUM kernel, one launch per AVG aggregate -------------
    //
    // We hold the GpuVecs for SUM outputs in this vector; they're moved
    // into the host-side `host_sums` Vec<Vec<f64>> after sync. Allocating
    // and uploading the value column inside the loop keeps peak GPU
    // memory bounded at `keys + 1 vals + N sums + 1 counts` instead of
    // `keys + N vals + N sums + 1 counts`.
    let mut host_sums: Vec<Vec<f64>> = Vec::with_capacity(n_aggs);
    for val_arr in val_arrs.iter() {
        let vals_gpu = GpuVec::<f64>::from_slice(val_arr.values())?;
        let mut sum_out_gpu = GpuVec::<f64>::zeros(n_groups as usize)?;

        // CUDA-Oxide typed launch path. Kernel ABI:
        //   keys_ptr, vals_ptr, sum_ptr, n_rows, n_groups
        // The view borrows scope to the inner block so `sum_out_gpu` is
        // free to be read via `.to_vec()` afterwards.
        {
            let view_keys = keys_gpu.view();
            let view_vals = vals_gpu.view();
            let mut view_sum = sum_out_gpu.view_mut();

            let mut args = KernelArgs::empty();
            args.push_input(&view_keys);
            args.push_input(&view_vals);
            args.push_output(&mut view_sum);
            args.push_scalar_u32(n_rows_u32);
            args.push_scalar_u32(n_groups_u32);

            // Kernel uses static shared-mem; dynamic shmem param = 0.
            launch_with_geometry(
                sum_function,
                params.grid_blocks,
                params.block_threads,
                0,
                &stream,
                &mut args,
            )?;
        }

        host_sums.push(sum_out_gpu.to_vec()?);
        // `vals_gpu` and `sum_out_gpu` are dropped at the end of the
        // iteration, freeing the device memory before the next AVG's
        // value column is uploaded.
    }

    // --- Pass N+1: COUNT kernel, single launch ---------------------------
    //
    // The COUNT kernel reads ONLY the key column — it produces
    // `count[g] = #rows with key == g`. We share this across every AVG.
    //
    // CUDA-Oxide typed launch path. Kernel ABI:
    //   keys_ptr, count_ptr, n_rows, n_groups
    let mut count_gpu = GpuVec::<u64>::zeros(n_groups as usize)?;
    {
        let view_keys = keys_gpu.view();
        let mut view_count = count_gpu.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_output(&mut view_count);
        args.push_scalar_u32(n_rows_u32);
        args.push_scalar_u32(n_groups_u32);

        launch_with_geometry(
            count_function,
            params.grid_blocks,
            params.block_threads,
            0,
            &stream,
            &mut args,
        )?;
    }
    let host_counts: Vec<u64> = count_gpu.to_vec()?;

    // --- Host-side: AVG = SUM / COUNT, build output ----------------------
    //
    // Walk the count vector once. `counts[g] == 0` means "group g has no
    // rows" — omit it from the output to match the single-SUM executor's
    // semantics (and SQL's: an aggregate over an empty group is absent).
    let n_present = host_counts.iter().filter(|&&c| c > 0).count();
    let mut out_keys: Vec<i32> = Vec::with_capacity(n_present);
    let mut out_avg_cols: Vec<Vec<f64>> =
        (0..n_aggs).map(|_| Vec::with_capacity(n_present)).collect();

    for g in 0..(n_groups as usize) {
        let c = host_counts[g];
        if c == 0 {
            continue;
        }
        out_keys.push(g as i32);
        // `as f64` from u64 is exact for counts up to 2^53; far beyond
        // any realistic per-group cardinality at our 64K row floor.
        let c_f64 = c as f64;
        for (i, sum_col) in host_sums.iter().enumerate() {
            out_avg_cols[i].push(sum_col[g] / c_f64);
        }
    }

    // --- Build the output RecordBatch matching the plan's schema --------
    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(1 + n_aggs);
    columns.push(Arc::new(Int32Array::from(out_keys)) as ArrayRef);
    for col in out_avg_cols {
        columns.push(Arc::new(Float64Array::from(col)) as ArrayRef);
    }

    RecordBatch::try_new(arrow_schema, columns).map_err(|e| {
        PatinaError::Other(format!(
            "shmem_avg_exec: failed to build output RecordBatch: {e}"
        ))
    })
}

/// Build a 0-row output matching the plan's output schema. Used when the
/// input has 0 rows (or only negative keys, which we treat the same way
/// as "no groups").
fn build_empty_result(plan: &PhysicalPlan) -> PatinaResult<RecordBatch> {
    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => {
            return Err(PatinaError::Other(
                "shmem_avg_exec::build_empty_result: non-Aggregate plan".into(),
            ))
        }
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    let n_aggs = aggregate.aggregates.len();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(1 + n_aggs);
    columns.push(Arc::new(Int32Array::from(Vec::<i32>::new())) as ArrayRef);
    for _ in 0..n_aggs {
        columns.push(Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef);
    }
    RecordBatch::try_new(arrow_schema, columns)
        .map_err(|e| PatinaError::Other(format!("empty result build failed: {e}")))
}

// Local copy of the plan-schema -> Arrow-schema conversion. Each
// shared-mem executor carries its own copy; consolidating them is a
// separate refactor (see `groupby_shmem_exec.rs` for the matching one).
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
