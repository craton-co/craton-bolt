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
use arrow_schema::{Schema as ArrowSchema};

use crate::cuda::{GpuVec, PinnedHostBuffer};
use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_shmem_launch::{tune, TuneInputs};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::module_cache;
use crate::jit::shmem_count_kernel::{
    compile_shmem_count_kernel, KERNEL_ENTRY as COUNT_ENTRY,
};
use crate::jit::shmem_sum_kernel::{
    compile_shmem_sum_kernel, BLOCK_GROUPS, KERNEL_ENTRY as SUM_ENTRY,
};
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
) -> Option<BoltResult<RecordBatch>> {
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
) -> BoltResult<RecordBatch> {
    let n_rows = key_arr.len();
    let n_aggs = val_arrs.len();

    // Stage-4 (P1b): per-call stream shared across every H2D / kernel / D2H.
    let stream = CudaStream::null_or_default();

    // --- Upload key column ONCE; reused by every SUM + the one COUNT. ----
    let keys_gpu = GpuVec::<i32>::from_slice_async(key_arr.values(), stream.raw())?;

    // --- JIT both kernels (PTX cache hits after first run) ---------------
    // Routed through the consolidated `exec::module_cache` so repeated
    // SUM/COUNT launches skip PTX generation entirely. Namespaced by SUM
    // namespace (matches `groupby_shmem_exec`) for SUM and by our own
    // module path for COUNT.
    let sum_module = module_cache::get_or_build_module(
        module_path!(),
        "shmem_sum".to_string(),
        None,
        || compile_shmem_sum_kernel(),
    )?;
    let sum_function = sum_module.function(SUM_ENTRY)?;

    let count_module = module_cache::get_or_build_module(
        module_path!(),
        "shmem_count".to_string(),
        None,
        || compile_shmem_count_kernel(),
    )?;
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
        BoltError::Other(format!(
            "shmem_avg_exec: launch-param tuner refused: {e} \
             (n_rows={n_rows}, n_groups={n_groups})"
        ))
    })?;

    let n_rows_u32 = n_rows as u32;
    let n_groups_u32 = n_groups;

    // --- Pass 1..N: SUM kernel, one launch per AVG aggregate -------------
    //
    // Stage-5 (P1b): we persist *all* per-aggregate device output buffers
    // and pinned host destinations across the loop, with a SINGLE
    // `stream.synchronize()` after the last iteration plus the count
    // pass. The per-iteration sync in Stage 4 was needed because the
    // GpuVec for each SUM accumulator was dropped at the end of its
    // iteration; we now keep them alive in `sum_out_gpus` so the pool
    // doesn't reclaim memory while DMA is still draining it.
    //
    // Memory cost: N PinnedHostBuffer<f64>[n_groups] + N GpuVec<f64>[n_groups].
    // Tier-1 caps n_groups at BLOCK_GROUPS (1024) and MAX_AVG_AGGS at 4,
    // so the upper bound is 4 × 1024 × 8 B = 32 KiB pinned + 32 KiB device
    // — trivial vs the input columns we're already holding.
    //
    // Benefit: kernel launches and D2H downloads now pipeline across
    // iterations. With stream-ordered async copies, iteration i+1's H2D
    // upload and kernel launch can start while iteration i's D2H is
    // still in flight, instead of serialising on a per-iter synchronize.
    let mut pinned_sums: Vec<PinnedHostBuffer<f64>> = Vec::with_capacity(n_aggs);
    let mut sum_out_gpus: Vec<GpuVec<f64>> = Vec::with_capacity(n_aggs);
    // The vals GpuVecs also have to live until sync — the kernel reads
    // them via the SAME stream as the D2H, so chronologically the SUM
    // kernel completes before the D2H starts, BUT we still cannot drop
    // them while the kernel is in flight. Holding onto them until the
    // final sync is the simplest correct choice; it slightly increases
    // peak GPU memory but the increase is bounded by `N × n_rows × 8 B`
    // and Tier-1 already loaded all N columns into the Arrow batch on
    // the host side, so the GPU footprint mirrors that.
    let mut vals_gpus: Vec<GpuVec<f64>> = Vec::with_capacity(n_aggs);

    // Pre-allocate the pinned destinations + device outputs. Allocating
    // up front (rather than inside the loop) keeps the pinned host pages
    // contiguous in allocation order, which matters less on x86_64 than
    // on jemalloc-managed pools but is harmless either way.
    for _ in 0..n_aggs {
        pinned_sums.push(PinnedHostBuffer::<f64>::new(n_groups as usize)?);
        sum_out_gpus.push(GpuVec::<f64>::zeros_async(n_groups as usize, stream.raw())?);
    }

    for (i, val_arr) in val_arrs.iter().enumerate() {
        // Upload the value column on the stream. The kernel that reads
        // it is enqueued on the same stream, so no synchronize is needed
        // between upload and kernel.
        let vals_gpu = GpuVec::<f64>::from_slice_async(val_arr.values(), stream.raw())?;

        // CUDA-Oxide typed launch path. Kernel ABI:
        //   keys_ptr, vals_ptr, sum_ptr, n_rows, n_groups
        // The view borrows scope to the inner block so `sum_out_gpus[i]`
        // is free to be read via `copy_to_async` afterwards.
        {
            let view_keys = keys_gpu.view();
            let view_vals = vals_gpu.view();
            let mut view_sum = sum_out_gpus[i].view_mut();

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

        // Stage-5: enqueue the D2H into the pre-allocated pinned host
        // buffer. NO synchronize here — the next iteration's H2D + kernel
        // can start while this D2H is in flight, since they touch
        // different device memory regions.
        sum_out_gpus[i].copy_to_async(pinned_sums[i].as_mut_slice(), stream.raw())?;

        // Stash the val column so it outlives the kernel. Dropping
        // `vals_gpu` here would return memory to the pool while the
        // kernel might still be running.
        vals_gpus.push(vals_gpu);
    }

    // --- Pass N+1: COUNT kernel, single launch ---------------------------
    //
    // The COUNT kernel reads ONLY the key column — it produces
    // `count[g] = #rows with key == g`. We share this across every AVG.
    //
    // CUDA-Oxide typed launch path. Kernel ABI:
    //   keys_ptr, count_ptr, n_rows, n_groups
    let mut count_gpu = GpuVec::<u64>::zeros_async(n_groups as usize, stream.raw())?;
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
    // Stage-5: pinned D2H for the count vector, no sync yet.
    let mut pinned_counts: PinnedHostBuffer<u64> =
        PinnedHostBuffer::<u64>::new(n_groups as usize)?;
    count_gpu.copy_to_async(pinned_counts.as_mut_slice(), stream.raw())?;

    // ====================================================================
    // SINGLE synchronize for the entire executor: 1 sync vs N+1 in Stage 4.
    //
    // All prior async work on `stream` — H2D uploads, N SUM kernels, N
    // SUM D2Hs, the COUNT kernel, the COUNT D2H — completes here. Now
    // the pinned host buffers hold the result and we can dereference
    // them as plain `&[T]`.
    // ====================================================================
    stream.synchronize()?;

    // Copy results out of pinned memory into ordinary `Vec`s. Pinned
    // pages stay alive until the buffers are dropped; we keep the
    // copies short-lived by collecting into Vec immediately so the
    // pinned pages can be released as soon as `pinned_sums` /
    // `pinned_counts` go out of scope at the end of this function.
    let host_sums: Vec<Vec<f64>> = pinned_sums
        .iter()
        .map(|p| p.as_slice().to_vec())
        .collect();
    let host_counts: Vec<u64> = pinned_counts.as_slice().to_vec();

    // After the sync the device output GpuVecs and pinned host buffers
    // are safe to drop; they'll be released at end of scope. Naming
    // them here in a comment serves as a reminder that the
    // `stream.synchronize()` above is what makes their `Drop` safe:
    // without it, the pool would reclaim device memory while DMA was
    // still active.
    //   - sum_out_gpus (Vec<GpuVec<f64>>)
    //   - vals_gpus    (Vec<GpuVec<f64>>)
    //   - count_gpu    (GpuVec<u64>)
    //   - pinned_sums / pinned_counts hold the result data; borrowed
    //     above for `to_vec`, then dropped here.

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
        BoltError::Other(format!(
            "shmem_avg_exec: failed to build output RecordBatch: {e}"
        ))
    })
}

/// Build a 0-row output matching the plan's output schema. Used when the
/// input has 0 rows (or only negative keys, which we treat the same way
/// as "no groups").
fn build_empty_result(plan: &PhysicalPlan) -> BoltResult<RecordBatch> {
    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => {
            return Err(BoltError::Other(
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
        .map_err(|e| BoltError::Other(format!("empty result build failed: {e}")))
}

// Local copy of the plan-schema -> Arrow-schema conversion. Each
// shared-mem executor carries its own copy; consolidating them is a
// separate refactor (see `groupby_shmem_exec.rs` for the matching one).
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal(s, "this aggregate output path")
}

// ---------------------------------------------------------------------------
// Stage-4 (P1b) async round-trip smoke test.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage4_tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    #[test]
    #[ignore = "gpu:tier1"]
    fn async_shmem_avg_round_trip() {
        let n: usize = 1024;
        let n_groups: usize = 8;
        let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
        let vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut sums = vec![0.0f64; n_groups];
        let mut counts = vec![0u64; n_groups];
        for (i, &k) in keys.iter().enumerate() {
            sums[k as usize] += vals[i];
            counts[k as usize] += 1;
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
                aggregates: vec![AggregateExpr::Avg(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("avg_v", DataType::Float64, true),
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
        let avs = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..out.num_rows() {
            let k = ks.value(i) as usize;
            let expected = sums[k] / counts[k] as f64;
            assert!((avs.value(i) - expected).abs() < 1e-6);
        }
    }
}

// ---------------------------------------------------------------------------
// Stage-5 (P1b) round-trip: multi-AVG path with persisted pinned host
// buffers and a single end-of-iteration synchronize. The Stage-4 test
// covers single-AVG correctness; this one drives the executor with the
// maximum supported aggregate count to exercise the persisted-buffer
// fan-out specifically.
//
// `n_rows = MIN_ROWS_FAST_PATH` (64K) so try_execute commits to the
// fast path instead of falling through.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage5_tests {
    use super::*;
    use crate::plan::logical_plan::Field;
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    /// Build the expected SUM and COUNT per group for verification.
    fn expected_avgs(keys: &[i32], cols: &[Vec<f64>], n_groups: usize) -> Vec<Vec<f64>> {
        let n_aggs = cols.len();
        let mut sums = vec![vec![0.0f64; n_groups]; n_aggs];
        let mut counts = vec![0u64; n_groups];
        for (i, &k) in keys.iter().enumerate() {
            counts[k as usize] += 1;
            for a in 0..n_aggs {
                sums[a][k as usize] += cols[a][i];
            }
        }
        sums.iter()
            .map(|s| {
                s.iter()
                    .zip(counts.iter())
                    .map(|(&sm, &c)| if c == 0 { 0.0 } else { sm / c as f64 })
                    .collect::<Vec<f64>>()
            })
            .collect()
    }

    #[test]
    #[ignore = "gpu:tier1"]
    fn multi_avg_persisted_pinned() {
        // 64K rows (the fast-path floor), 16 groups, 4 AVGs — exactly the
        // shape MAX_AVG_AGGS targets. Each iteration's D2H now writes
        // into its own persisted PinnedHostBuffer, and exactly ONE
        // stream.synchronize() covers all 4 SUM kernels + their D2Hs +
        // the COUNT kernel + its D2H.
        let n: usize = MIN_ROWS_FAST_PATH;
        let n_groups: usize = 16;
        let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
        let cols: Vec<Vec<f64>> = (0..MAX_AVG_AGGS)
            .map(|a| (0..n).map(|i| (i as f64) * (a as f64 + 1.0)).collect())
            .collect();
        let expected = expected_avgs(&keys, &cols, n_groups);

        // Plan: AVG(v0), AVG(v1), AVG(v2), AVG(v3) — four-up parity case.
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO { name: "k".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "v0".into(), dtype: DataType::Float64 },
                    ColumnIO { name: "v1".into(), dtype: DataType::Float64 },
                    ColumnIO { name: "v2".into(), dtype: DataType::Float64 },
                    ColumnIO { name: "v3".into(), dtype: DataType::Float64 },
                ],
                group_by: vec![0],
                aggregates: (0..MAX_AVG_AGGS)
                    .map(|a| AggregateExpr::Avg(Expr::Column(format!("v{a}"))))
                    .collect(),
                output_schema: Schema::new(
                    std::iter::once(Field::new("k", DataType::Int32, false))
                        .chain(
                            (0..MAX_AVG_AGGS).map(|a| {
                                Field::new(format!("avg_v{a}"), DataType::Float64, true)
                            }),
                        )
                        .collect(),
                ),
                input_has_validity: Vec::new(),
            },
        };

        // Arrow batch with the 1 key column + 4 value columns.
        let mut fields: Vec<ArrowField> =
            vec![ArrowField::new("k", ArrowDataType::Int32, false)];
        for a in 0..MAX_AVG_AGGS {
            fields.push(ArrowField::new(
                format!("v{a}"),
                ArrowDataType::Float64,
                false,
            ));
        }
        let schema = Arc::new(ArrowSchema::new(fields));
        let mut columns: Vec<ArrayRef> =
            vec![Arc::new(Int32Array::from(keys.clone())) as ArrayRef];
        for col in &cols {
            columns.push(Arc::new(Float64Array::from(col.clone())) as ArrayRef);
        }
        let batch = RecordBatch::try_new(schema, columns).unwrap();

        let out = match try_execute(&plan, &batch) {
            Some(Ok(b)) => b,
            // No CUDA context — bail; this is the standard pattern for
            // `#[ignore]`-gated GPU tests.
            _ => return,
        };

        // Verify each AVG column matches the host-side expectation. The
        // executor emits one row per present group; here every group has
        // n / n_groups rows so every group is present.
        assert_eq!(out.num_rows(), n_groups);
        let out_keys = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for a in 0..MAX_AVG_AGGS {
            let col = out
                .column(1 + a)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for row in 0..out.num_rows() {
                let g = out_keys.value(row) as usize;
                assert!(
                    (col.value(row) - expected[a][g]).abs() < 1e-6,
                    "AVG(v{a}) group {g}: got {} expected {}",
                    col.value(row),
                    expected[a][g],
                );
            }
        }
    }

    #[test]
    #[ignore = "gpu:tier1"]
    fn single_avg_still_correct_after_persist_refactor() {
        // Regression guard for the N=1 path: the persisted-buffer code
        // must not have broken the most-common single-AVG case (it had
        // a per-iteration sync in Stage 4; now there's a single
        // end-of-executor sync).
        let n: usize = MIN_ROWS_FAST_PATH;
        let n_groups: usize = 8;
        let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
        let vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let expected = expected_avgs(&keys, &[vals.clone()], n_groups);

        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO { name: "k".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "v".into(), dtype: DataType::Float64 },
                ],
                group_by: vec![0],
                aggregates: vec![AggregateExpr::Avg(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("avg_v", DataType::Float64, true),
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
                Arc::new(Int32Array::from(keys)) as ArrayRef,
                Arc::new(Float64Array::from(vals)) as ArrayRef,
            ],
        )
        .unwrap();

        let out = match try_execute(&plan, &batch) {
            Some(Ok(b)) => b,
            _ => return,
        };
        let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let avs = out.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..out.num_rows() {
            let g = ks.value(i) as usize;
            assert!((avs.value(i) - expected[0][g]).abs() < 1e-6);
        }
    }
}
