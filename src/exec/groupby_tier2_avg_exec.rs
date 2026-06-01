// SPDX-License-Identifier: Apache-2.0

//! **AVG at Tier 2.1** — high-cardinality multi-AVG executor.
//!
//! The Tier-1 AVG executor (`groupby_shmem_avg_exec.rs`) handles
//! `n_groups ≤ 1024`. For higher-cardinality workloads (e.g. a future
//! `SELECT id3, AVG(v1), AVG(v2) FROM x GROUP BY id3` where id3 has 1 M
//! distinct values), the Tier-2 partitioning approach is the better
//! algorithm — exactly as it is for SUM (q3, q5).
//!
//! ## Algorithm
//!
//! 1. **Partition + scatter**: identical to `groupby_tier2_multi_
//!    orchestrator`. One partition kernel produces (partition_ids,
//!    counts); host-side prefix-sum gives the offsets; ONE atomic-claim
//!    pass writes the per-row `dest_idx` map + the scattered key column;
//!    N atomic-free indexed-scatter passes scatter each value column to
//!    the slots `dest_idx` specifies. This guarantees alignment between
//!    the key column and every value column by construction — independent
//!    of any `atomicAdd` ordering assumptions.
//! 2. **Pass 2 — SUMs**: one launch of `partition_reduce_kernel_multi`
//!    (n_vals = N) reduces each partition into N per-group SUMs.
//! 3. **Pass 2 — COUNT**: one launch of `partition_reduce_kernel_count`
//!    against the *same* scatter_keys buffer reduces each partition into
//!    per-group `u64` counts. No extra partitioning / scatter cost.
//! 4. **Compose**: walk the two output buffers in lockstep. For each
//!    populated slot push `(key, [sum_i / count_i for i in 0..N])` into
//!    the result. Slots with `count == 0` are omitted (SQL semantics for
//!    empty groups).
//!
//! Net cost vs the multi-SUM path: +1 reduce kernel launch (~10 ms) and
//! +8 MiB of D2H for the count output. Both well-amortised at any size
//! that selects this path in the first place.
//!
//! ## Scope (v0)
//!
//! - GROUP BY exactly one Int32 column
//! - 1..=`MAX_VALS` aggregates, ALL `AVG(<bare Float64 column>)`
//! - `n_rows >= 256 K` (matches `TIER2_MIN_ROWS`)
//! - `max(key) >= BLOCK_GROUPS` so Tier-1 AVG doesn't already win this
//! - `max(key) < 100 M` (Tier-2 dispatcher cap)

use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int32Array, RecordBatch};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::module_cache;
use crate::exec::partition_offsets;
use crate::jit::{
    partition_kernel, partition_reduce_kernel_count, partition_reduce_kernel_multi,
    scatter_values_by_dest_idx_kernel, scatter_with_dest_idx_kernel, CudaModule,
};
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Module-cache wrapper.
//
// `CudaModule::from_ptx` (in `src/jit/jit_compiler.rs`) already deduplicates
// PTX → SASS by hashing the PTX text, but the AVG path issues *four* distinct
// kernel launches per query (partition, scatter, multi-SUM-reduce, count-
// reduce) and each one used to rebuild a kilobyte-scale PTX string only to
// hit the cache. We route every lookup through the process-wide
// `exec::module_cache` so all sibling executors share one consolidated table;
// the local `enum KernelSpec` stays private (its variants don't fit the
// engine's projection-path `ModuleCacheKey`) and the `module_path!()`
// namespace keeps our cache slots disjoint from siblings'. See
// `module_cache`'s docs for the multi-GPU caveat and the migration TODO.
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum KernelSpec {
    /// `partition_kernel::compile_partition_kernel()`.
    Partition,
    PartitionShmemStaging,
    /// `scatter_with_dest_idx_kernel::compile_scatter_with_dest_idx_kernel()`.
    /// Atomic-claim scatter pass that produces the per-row `dest_idx`.
    ScatterWithDestIdx,
    /// `scatter_values_by_dest_idx_kernel::compile_scatter_values_by_dest_idx_kernel()`.
    /// Atomic-free per-value-column scatter using a pre-computed `dest_idx`.
    ScatterValuesByDestIdx,
    /// `partition_reduce_kernel_multi::compile_partition_reduce_kernel_multi_with_spill(n_vals)`.
    /// `n_vals` is the per-row fan-out (1..=`MAX_VALS`). Batch-5 spill-aware
    /// variant — launches resolve `kernel_entry_with_spill(n_vals)`.
    ReduceMulti { n_vals: u32 },
    /// `partition_reduce_kernel_count::compile_partition_reduce_kernel_count_with_spill()`.
    /// Batch-5 spill-aware variant — launches resolve `KERNEL_ENTRY_WITH_SPILL`.
    ReduceCount,
}

/// Test-only counter of cache-miss compile passes serviced via THIS executor.
#[cfg(test)]
static LOAD_COUNT: module_cache::LoadCounter = module_cache::LoadCounter::new();

/// Cache-aware module loader. See module-cache comment above.
fn get_or_build_module(spec: &KernelSpec) -> BoltResult<CudaModule> {
    #[cfg(test)]
    let counter = Some(&LOAD_COUNT);
    #[cfg(not(test))]
    let counter = None;
    module_cache::get_or_build_module(module_path!(), format!("{:?}", spec), counter, || {
        Ok(match spec {
            KernelSpec::Partition => partition_kernel::compile_partition_kernel()?,
            KernelSpec::PartitionShmemStaging => partition_kernel::compile_partition_kernel_shmem_staging()?,
            KernelSpec::ScatterWithDestIdx => {
                scatter_with_dest_idx_kernel::compile_scatter_with_dest_idx_kernel()?
            }
            KernelSpec::ScatterValuesByDestIdx => {
                scatter_values_by_dest_idx_kernel::compile_scatter_values_by_dest_idx_kernel()?
            }
            KernelSpec::ReduceMulti { n_vals } => {
                // Batch 5: spill-counter-aware variant — the launch site
                // resolves `kernel_entry_with_spill(n_vals)` and pushes a u32
                // spill counter as the trailing kernel arg.
                partition_reduce_kernel_multi::compile_partition_reduce_kernel_multi_with_spill(
                    *n_vals,
                )?
            }
            KernelSpec::ReduceCount => {
                // Batch 5: spill-counter-aware variant. AVG drives two reduce
                // kernels — the multi-SUM above and this COUNT — and both must
                // surface MAX_PROBES overflow as a structured error rather
                // than silently corrupting the per-key denominator.
                partition_reduce_kernel_count::compile_partition_reduce_kernel_count_with_spill()?
            }
        })
    })
}

fn partition_spec_for(n_rows: u32) -> KernelSpec {
    // dedup (tier2): threshold test shared via
    // `groupby_tier2_common::use_shmem_staging_partition`.
    if crate::exec::groupby_tier2_common::use_shmem_staging_partition(n_rows) {
        KernelSpec::PartitionShmemStaging
    } else {
        KernelSpec::Partition
    }
}


/// Try to execute `plan` against `batch` via the Tier-2.1 AVG fast path.
/// `None` on any miss — caller falls through to the next strategy.
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
    if aggregate.group_by.len() != 1 {
        return None;
    }
    let n_vals = aggregate.aggregates.len();
    if n_vals == 0 || n_vals > partition_reduce_kernel_multi::MAX_VALS as usize {
        return None;
    }

    // Single Int32 key.
    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    // All aggregates must be AVG(<bare Float64 column>).
    let mut val_col_names: Vec<&str> = Vec::with_capacity(n_vals);
    for agg in &aggregate.aggregates {
        let name = match agg {
            AggregateExpr::Avg(Expr::Column(n)) => n.as_str(),
            _ => return None,
        };
        val_col_names.push(name);
    }

    // Look up key + value arrays. Every value must be Float64.
    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    let mut val_arrs: Vec<&Float64Array> = Vec::with_capacity(n_vals);
    for name in &val_col_names {
        let arr = batch
            .column_by_name(name)
            .and_then(|c| c.as_any().downcast_ref::<Float64Array>())?;
        if arr.len() != key_arr.len() {
            return None;
        }
        val_arrs.push(arr);
    }

    // GB-S1: NULL handling — this fast path reads `key_arr.values()` /
    // `arr.values()` straight off the Arrow data buffers, which carry
    // garbage bytes at NULL positions (NULL values fold in as 0 and inflate
    // the AVG denominator; NULL keys synthesize a group-0). Defer
    // NULL-bearing batches back to `groupby::execute_groupby` → the
    // global-atomic path, which consults the validity bitmap. Mirrors the
    // guard in `groupby_tier2_twokey_exec::try_execute`.
    if key_arr.null_count() > 0 || val_arrs.iter().any(|a| a.null_count() > 0) {
        return None;
    }

    let n_rows = key_arr.len();
    if n_rows < 256 * 1024 {
        return None;
    }

    // n_groups estimator via max key. Reject Tier-1's territory and the
    // Tier-2 dispatcher's cap.
    // dedup (tier2/shmem): max-nonneg-key scan extracted to
    // `groupby_tier2_common`. `None` (negative key) and `Some(-1)` (empty)
    // both decline, matching the prior inline behaviour.
    let max_key = crate::exec::groupby_tier2_common::scan_max_nonneg_key(key_arr.values())?;
    if max_key < 0 {
        return None;
    }
    let n_groups_est = (max_key as u32).saturating_add(1);
    if n_groups_est <= partition_reduce_kernel_multi::BLOCK_GROUPS {
        // Tier-1 AVG owns this.
        return None;
    }
    if n_groups_est >= 100_000_000 {
        return None;
    }

    Some(execute_inner(plan, key_arr, val_arrs, n_vals))
}

fn execute_inner(
    plan: &PhysicalPlan,
    key_arr: &Int32Array,
    val_arrs: Vec<&Float64Array>,
    n_vals: usize,
) -> BoltResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;

    // Stage-4 (P1b): per-call stream so every H2D upload, kernel
    // launch, and final D2H share one ordering domain.
    let stream = CudaStream::null();

    // ---- Upload inputs --------------------------------------------------
    let keys_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice_async(key_arr.values(), stream.raw())?;
    let mut vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for arr in &val_arrs {
        vals_gpu.push(GpuVec::<f64>::from_slice_async(arr.values(), stream.raw())?);
    }

    let num_partitions = partition_kernel::NUM_PARTITIONS;

    // ---- Partition pass --------------------------------------------------
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

    // ---- Offsets (P1b-stage8: joint helper, 2 syncs → 1) ----------------
    let (offsets, offsets_gpu): (Vec<u32>, GpuVec<u32>) =
        partition_offsets::compute_and_upload_partition_offsets_async(&counts, stream.raw())?;

    // ---- Scatter (deterministic dest_idx + indexed value passes) ---------
    //
    // Correctness note: the previous design called the atomic-claim scatter
    // kernel once per value column, relying on identical `atomicAdd`
    // orderings across launches to keep `(key, v1, v2, …)` aligned. That
    // ordering is NOT a CUDA contract, so a driver/scheduler change could
    // silently misalign `SUM(v_j)` with the wrong key.
    //
    // We now run the atomic-claim pass exactly ONCE
    // (`scatter_with_dest_idx_kernel`), capturing the per-row destination
    // slot in `dest_idx[n_rows]`. Each subsequent value column is scattered
    // by an atomic-free kernel that reads `dest_idx[i]` and writes
    // `out_vals[dest_idx[i]] = vals[i]`. Alignment is guaranteed by
    // construction. The COUNT reduce below also reads `scatter_keys` (the
    // claim pass's output), so the SUM-side / COUNT-side slot-population
    // agreement that the historical comment relied on is now a structural
    // property of the pipeline rather than an unsubstantiated assumption.
    let mut scatter_keys: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        scatter_vals.push(GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?);
    }
    let mut dest_idx: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;

    // Atomic-claim pass.
    {
        let claim_module = get_or_build_module(&KernelSpec::ScatterWithDestIdx)?;
        let func = claim_module.function(scatter_with_dest_idx_kernel::KERNEL_ENTRY)?;

        let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);

        let view_keys = keys_gpu.view();
        let view_pids = partition_ids.view();
        let view_offsets = offsets_gpu.view();
        let mut view_cursors = cursors.view_mut();
        let mut view_sk = scatter_keys.view_mut();
        let mut view_di = dest_idx.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_input(&view_pids);
        args.push_input(&view_offsets);
        args.push_output(&mut view_cursors);
        args.push_output(&mut view_sk);
        args.push_output(&mut view_di);
        args.push_scalar_u32(n_rows);

        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    // Indexed value scatter — one launch per value column, no atomics.
    {
        // Reuse the cached module across all N value-column scatters; obtain
        // a fresh `CudaFunction` handle per launch since it borrows the
        // module for the duration of the kernel args.
        let scatter_module = get_or_build_module(&KernelSpec::ScatterValuesByDestIdx)?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);

        for j in 0..n_vals {
            let func = scatter_module.function(scatter_values_by_dest_idx_kernel::KERNEL_ENTRY)?;

            let view_vals = vals_gpu[j].view();
            let view_dest = dest_idx.view();
            let mut view_sv = scatter_vals[j].view_mut();

            let mut args = KernelArgs::empty();
            args.push_input(&view_vals);
            args.push_input(&view_dest);
            args.push_output(&mut view_sv);
            args.push_scalar_u32(n_rows);

            launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
        }
    }

    // Reduce kernels need the FULL K+1 offsets buffer.
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;

    let block_groups = partition_reduce_kernel_multi::BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;

    let mut out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        out_vals_gpu.push(GpuVec::<f64>::zeros_async(n_out_slots, stream.raw())?);
    }
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_counts_gpu: GpuVec<u64> = GpuVec::<u64>::zeros_async(n_out_slots, stream.raw())?;
    // Perf (redundant-buffer elision): the COUNT reduce's kernel signature
    // mandates an `out_keys` and an `out_set` output, but we only consume its
    // `out_counts`. The SUM and COUNT reduces both consume the SAME
    // `scatter_keys` buffer with the SAME `slot = key & mask` hash and the
    // same deterministic probe sequence over the same per-partition row range,
    // so for every (partition, slot) they agree byte-for-byte on whether the
    // slot is populated and on the key it holds. We therefore point the COUNT
    // launch's out_keys / out_set params straight at the SUM reduce's
    // `out_keys_gpu` / `out_set_gpu`: COUNT re-stores the identical key/set
    // bytes SUM already wrote (a value-preserving overwrite, ordered after the
    // SUM launch on this single stream), and the final result is unchanged.
    // This drops two zero-initialised `n_out_slots` scratch buffers
    // (`count_out_keys` / `count_out_set`) that the old code allocated only to
    // discard. See the post-launch download comment for why this is sound.
    //
    // TODO(perf, cross-file): the truly minimal pass would be a fused
    // sum+count reduce kernel emitting per-group SUMs and the u64 COUNT in one
    // launch over `scatter_keys` (one hash/probe pass instead of two). That
    // requires adding an entry point in `jit::partition_reduce_kernel_multi`
    // (extra `out_counts` output + an `atom.shared.add.u64` on the shared
    // count slot alongside the existing value adds) and is out of scope here.
    //
    // Spill counters for the multi-SUM and COUNT reduces. AVG = sum/count,
    // so EITHER reduce dropping a row on MAX_PROBES overflow silently
    // corrupts the per-key average. We allocate one counter per kernel and
    // OR-merge the post-sync check below.
    let mut spill_multi: GpuVec<u32> = GpuVec::<u32>::zeros_async(1, stream.raw())?;
    let mut spill_count: GpuVec<u32> = GpuVec::<u32>::zeros_async(1, stream.raw())?;

    // ---- Multi-SUM reduce -----------------------------------------------
    let reduce_multi_module =
        get_or_build_module(&KernelSpec::ReduceMulti { n_vals: n_vals as u32 })?;
    {
        let entry = partition_reduce_kernel_multi::kernel_entry_with_spill(n_vals as u32);
        let func = reduce_multi_module.function(&entry)?;

        let view_sk = scatter_keys.view();
        let views_sv: Vec<_> = scatter_vals.iter().map(|g| g.view()).collect();
        let view_offsets = offsets_kp1_gpu.view();
        let mut view_ok = out_keys_gpu.view_mut();
        let mut views_ov: Vec<_> = out_vals_gpu.iter_mut().map(|g| g.view_mut()).collect();
        let mut view_os = out_set_gpu.view_mut();
        let mut view_sp = spill_multi.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_sk);
        for v in &views_sv {
            args.push_input(v);
        }
        args.push_input(&view_offsets);
        args.push_output(&mut view_ok);
        for v in views_ov.iter_mut() {
            args.push_output(v);
        }
        args.push_output(&mut view_os);
        args.push_output(&mut view_sp);

        launch_with_geometry(
            func,
            num_partitions,
            partition_reduce_kernel_multi::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // ---- COUNT reduce ----------------------------------------------------
    let reduce_count_module = get_or_build_module(&KernelSpec::ReduceCount)?;
    {
        let func = reduce_count_module
            .function(partition_reduce_kernel_count::KERNEL_ENTRY_WITH_SPILL)?;

        let view_keys = scatter_keys.view();
        let view_offsets = offsets_kp1_gpu.view();
        // Reuse the SUM reduce's key/set buffers (see allocation comment): the
        // COUNT kernel re-writes the identical key/set bytes into the same
        // slots, so this overwrite is value-preserving and saves two buffers.
        let mut view_ok = out_keys_gpu.view_mut();
        let mut view_oc = out_counts_gpu.view_mut();
        let mut view_os = out_set_gpu.view_mut();
        let mut view_sp = spill_count.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_input(&view_offsets);
        args.push_output(&mut view_ok);
        args.push_output(&mut view_oc);
        args.push_output(&mut view_os);
        args.push_output(&mut view_sp);

        launch_with_geometry(
            func,
            num_partitions,
            partition_reduce_kernel_count::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // ---- Download everything --------------------------------------------
    //
    // The SUM reduce and COUNT reduce both consume `scatter_keys` (written
    // by the single atomic-claim pass above) and hash with the same slot
    // function, so for a given (partition, slot) both kernels write either
    // both populated or both empty, and both populate with the same key.
    // We use the SUM-side out_keys / out_set and the COUNT-side out_counts.
    // Because that key/set agreement is byte-exact, the COUNT launch wrote
    // its (identical) keys / set DIRECTLY into `out_keys_gpu` / `out_set_gpu`
    // (see the allocation comment) — the separate count_out_keys /
    // count_out_set scratch buffers no longer exist. `out_keys_gpu` /
    // `out_set_gpu` therefore carry the SUM kernel's values, last overwritten
    // by the COUNT kernel with the same bytes.
    // Stage-4 (P1b): pinned D2H for every output buffer; sync once
    // after all are queued so the driver overlaps them.
    let pinned_keys = out_keys_gpu.to_pinned_async(stream.raw())?;
    let mut pinned_vals: Vec<crate::cuda::PinnedHostBuffer<f64>> =
        Vec::with_capacity(n_vals);
    for ov in &out_vals_gpu {
        pinned_vals.push(ov.to_pinned_async(stream.raw())?);
    }
    let pinned_set = out_set_gpu.to_pinned_async(stream.raw())?;
    let pinned_counts = out_counts_gpu.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    // AVG = SUM/COUNT — both reduces must succeed for the per-key average to
    // be correct. OR-merge the two counters and surface a structured error
    // mentioning each component so the failure mode is unambiguous.
    let spill_multi_count = spill_multi.to_vec()?[0];
    let spill_count_count = spill_count.to_vec()?[0];
    if spill_multi_count > 0 || spill_count_count > 0 {
        return Err(BoltError::Other(format!(
            "partition_reduce spill: multi={} count={} rows exceeded MAX_PROBES; result may be incorrect",
            spill_multi_count, spill_count_count
        )));
    }
    let host_out_keys: Vec<i32> = pinned_keys.as_slice().to_vec();
    let host_out_vals: Vec<Vec<f64>> = pinned_vals
        .iter()
        .map(|p| p.as_slice().to_vec())
        .collect();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();
    let host_out_counts: Vec<u64> = pinned_counts.as_slice().to_vec();

    // ---- Walk slots, divide host-side, build output ---------------------
    let mut out_keys_final: Vec<i32> = Vec::new();
    let mut out_avgs_final: Vec<Vec<f64>> =
        (0..n_vals).map(|_| Vec::new()).collect();

    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] == 0 {
                continue;
            }
            let c = host_out_counts[idx];
            if c == 0 {
                // Defensive: set==1 but count==0 means the two kernels
                // disagreed on slot population. With the deterministic
                // dest_idx scatter both kernels consume the same
                // scatter_keys buffer with the same slot function, so
                // this branch should be unreachable; we keep it as a
                // belt-and-suspenders skip rather than panicking, to
                // match SQL "no rows → no output" semantics.
                continue;
            }
            let cf = c as f64;
            out_keys_final.push(host_out_keys[idx]);
            for j in 0..n_vals {
                out_avgs_final[j].push(host_out_vals[j][idx] / cf);
            }
        }
    }

    // Sort by key (ASC) to match SQL canonical / what the equivalence
    // check expects.
    let mut idx: Vec<usize> = (0..out_keys_final.len()).collect();
    idx.sort_by_key(|&i| out_keys_final[i]);
    let sorted_keys: Vec<i32> = idx.iter().map(|&i| out_keys_final[i]).collect();
    let sorted_avgs: Vec<Vec<f64>> = (0..n_vals)
        .map(|j| idx.iter().map(|&i| out_avgs_final[j][i]).collect())
        .collect();

    // ---- Build the output RecordBatch -----------------------------------
    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema =
        crate::exec::groupby_tier2_common::plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    let mut cols: Vec<arrow_array::ArrayRef> = Vec::with_capacity(1 + n_vals);
    cols.push(Arc::new(Int32Array::from(sorted_keys)));
    for v in sorted_avgs {
        cols.push(Arc::new(Float64Array::from(v)));
    }
    RecordBatch::try_new(arrow_schema, cols).map_err(|e| {
        BoltError::Other(format!(
            "groupby_tier2_avg_exec: failed to build RecordBatch: {e}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Module-cache mechanics tests. Skip gracefully on CPU-only hosts (no CUDA
// context, so `from_ptx` errors). Verify:
//   * a repeat call with the same `KernelSpec` does not re-compile;
//   * two specs that differ only in `n_vals` are distinct cache keys
//     (separate miss + subsequent hit).
// ---------------------------------------------------------------------------
// v0.7: these arrow/plan aliases are used only by the #[cfg(test)] modules
// below; the non-test schema conversion now lives in exec::schema_convert.
// cfg(test)-gated so normal builds don't see an unused import.
#[cfg(test)]
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
#[cfg(test)]
use crate::plan::logical_plan::Schema;

#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn second_call_same_spec_is_cache_hit() {
        let m1 = match get_or_build_module(&KernelSpec::Partition) {
            Ok(m) => m,
            Err(_) => return, // no CUDA context — skip.
        };
        let after_first = LOAD_COUNT.load(Ordering::SeqCst);
        let m2 = get_or_build_module(&KernelSpec::Partition)
            .expect("second lookup must succeed");
        let after_second = LOAD_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            after_second, after_first,
            "repeat call must not increment LOAD_COUNT (was {} -> {})",
            after_first, after_second
        );
        assert_eq!(m1.raw(), m2.raw(), "clones must share the same CUmodule");
    }

    #[test]
    fn different_n_vals_are_distinct_cache_keys() {
        // Warm two different reduce-multi specs and confirm a follow-up
        // lookup hits each without recompiling.
        let _ = match get_or_build_module(&KernelSpec::ReduceMulti { n_vals: 1 }) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ReduceMulti { n_vals: 2 })
            .expect("n_vals=2 build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::ReduceMulti { n_vals: 1 })
            .expect("n_vals=1 hit");
        let _ = get_or_build_module(&KernelSpec::ReduceMulti { n_vals: 2 })
            .expect("n_vals=2 hit");
        assert_eq!(
            LOAD_COUNT.load(Ordering::SeqCst),
            baseline,
            "both warm specs must be cache hits on the second lookup"
        );
    }
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
    #[ignore = "gpu:tier2"]
    fn async_tier2_avg_round_trip() {
        let n: usize = 300_000;
        let n_groups: usize = 4096;
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
            assert!((avs.value(i) - expected).abs() < 1e-6, "key={} expected={} got={}", k, expected, avs.value(i));
        }
    }
}
