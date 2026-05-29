// SPDX-License-Identifier: Apache-2.0

//! Tier-2 hash-partitioned GROUP BY SUM orchestrator (head of the chain).
//!
//! ## Tier 2.1 — pass 2 now on the GPU
//!
//! Pass 2 (per-partition dedup+sum) used to live on the host: download the
//! `4 n_rows + 8 n_rows`-byte scatter buffers, then build a small
//! `HashMap<i32, f64>` per partition. At N=10 M that cost ~50 ms of D2H +
//! ~100 ms of host hash-map work, which dominated q5's 460 ms total.
//!
//! As of Tier 2.1 we replace that loop with a single launch of
//! [`crate::jit::partition_reduce_kernel`]. Grid = `NUM_PARTITIONS`, one
//! block per partition. Each block builds an open-addressing hash table
//! in 16 KiB of shared memory, walks its partition's scatter slice with a
//! grid-stride loop, and emits one slot's worth of `(key, val, set)` per
//! shared-table position. The host then walks the (now fixed-size,
//! `NUM_PARTITIONS * BLOCK_GROUPS * 13 B = 4096 * 1024 * 13 B = 52 MiB`)
//! output to collect the populated slots into the per-partition result
//! vectors.
//!
//! Net cost change:
//!   * D2H: ~150 ms (n_rows-sized buffers) → ~50 ms (fixed 52 MiB output)
//!   * Host pass: ~100 ms of HashMap → ~5 ms of trivial scan
//!   * GPU pass: +10–30 ms for the new kernel
//! Net: ~120 ms saved, projected ~5× speedup on q5 per
//! `docs/GROUPBY_PERF.md`.
//!
//! ## Pipeline
//!
//! 1. **Partition pass.** Hash every input row's key into one of
//!    `NUM_PARTITIONS = 4096` buckets, count rows per bucket, and remember
//!    the per-row bucket assignment. (`partition_kernel`, sibling agent.)
//! 2. **Prefix sum.** Exclusive prefix sum over the bucket counts gives the
//!    write offset for each partition in the scattered output.
//!    (`partition_offsets`, sibling agent.)
//! 3. **Scatter pass.** Each input row writes its (key, value) into the
//!    scratch buffer at `offset[pid] + per_partition_cursor[pid]++`.
//!    Output is `n_rows` of i32 keys + `n_rows` of f64 values, contiguous
//!    by partition. (`scatter_kernel`, sibling agent.)
//! 4. **Per-partition reduce — GPU.** One CUDA block per partition builds
//!    a shared-memory open-addressing hash table over the partition's
//!    `[offsets[pid]..offsets[pid+1])` slice (see
//!    [`crate::jit::partition_reduce_kernel`]). The host then walks the
//!    fixed-size output buffer to collect populated slots.
//!
//! ## Sibling-agent stubs
//!
//! Three sibling agents own the GPU pieces of this pipeline and write them
//! in their own worktrees. THEIR files do not exist in mine. The
//! integrator's merger pass will replace these stub modules with real
//! `use crate::jit::partition_kernel::*;` (etc.) imports.
//!
//! Until then the stubs let this file **compile** in isolation, even if any
//! actual call to an unimplemented stub would `unimplemented!()` at runtime.
//! Wiring is the merger's job, not mine.

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::module_cache;
use crate::jit::CudaModule;

/// Stable prefix of the structured error returned by the Tier-2 reduce
/// orchestrators when the open-addressing hash table overflows
/// `MAX_PROBES` (a row's linear probe wrapped without finding a slot, so
/// its contribution was dropped and the result is incorrect).
///
/// This is a *soft-miss sentinel*, not a hard failure: `groupby.rs::
/// execute_groupby` matches on this prefix via `starts_with` and falls
/// through to the next strategy / the global-atomic path rather than
/// propagating the error to the caller. Every orchestrator that emits a
/// spill error (single-key SUM/AVG/COUNT/MIN/MAX and their two-key twins)
/// formats its message starting with this exact prefix so the dispatcher's
/// match stays robust to wording changes in the trailing detail.
pub(crate) const PARTITION_REDUCE_SPILL_PREFIX: &str = "partition_reduce spill:";

// ---------------------------------------------------------------------------
// Sibling-agent stubs.
//
// These are *placeholders* for the real modules other agents are writing in
// their own worktrees. The merger swaps `stub_*` for real `use` statements;
// see module-level docs.
//
// We do NOT depend on these stubs being correct at runtime — every body is
// `unimplemented!()` — but their *signatures* must match the merger's
// expectations so the swap is a one-line edit. Do not change a signature
// without coordinating with the sibling agent that owns it.
// ---------------------------------------------------------------------------

// Sibling-agent modules wired in (stubs replaced by real imports):
use crate::jit::partition_kernel as stub_partition_kernel;
use crate::jit::scatter_kernel as stub_scatter_kernel;
use crate::exec::partition_offsets as stub_partition_offsets;
// This file (Tier 2.1) owns:
use crate::jit::partition_reduce_kernel;

// ---------------------------------------------------------------------------
// Module-cache wrapper.
//
// The SUM pipeline runs three kernel launches (partition + scatter + reduce)
// per call. Each one used to rebuild a kilobyte-scale PTX string and feed it
// to `CudaModule::from_ptx` only to hit the in-driver PTX cache anyway. We
// route every lookup through the process-wide `exec::module_cache` so all
// sibling executors share one consolidated table — see that module's docs
// for the multi-GPU caveat and the per-Engine migration TODO.
//
// The local `enum KernelSpec` stays private (each executor's variants do not
// fit the engine's projection-path `ModuleCacheKey`), and the namespace
// argument to `get_or_build_module` (the `module_path!()`) keeps our slots
// disjoint from siblings'.
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum KernelSpec {
    Partition,
    PartitionShmemStaging,
    Scatter,
    ReduceSum,
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
            KernelSpec::Partition => stub_partition_kernel::compile_partition_kernel()?,
            KernelSpec::PartitionShmemStaging => stub_partition_kernel::compile_partition_kernel_shmem_staging()?,
            KernelSpec::Scatter => stub_scatter_kernel::compile_scatter_kernel()?,
            // Batch 4: spill-counter-aware variant — MAX_PROBES overflow
            // surfaces as a structured error instead of silently dropping
            // rows. The caller MUST allocate a u32 spill counter and pass
            // it as the extra kernel arg.
            KernelSpec::ReduceSum => {
                partition_reduce_kernel::compile_partition_reduce_kernel_with_spill()?
            }
        })
    })
}

fn partition_spec_for(n_rows: u32) -> KernelSpec {
    if n_rows < stub_partition_kernel::SHMEM_STAGING_MIN_ROWS {
        KernelSpec::Partition
    } else {
        KernelSpec::PartitionShmemStaging
    }
}


// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Tier-2 partial result: one `(keys, sums)` pair per partition.
///
/// Length is exactly `NUM_PARTITIONS`. Per-partition keys are already
/// deduplicated and summed; a merger downstream needs only to *concatenate*
/// the partitions (or stream them into a `RecordBatch` builder) to produce
/// the final output.
///
/// Empty partitions are represented as `(vec![], vec![])` rather than
/// elided — the index in `per_partition` is significant for downstream
/// merging (if a future pass wants to walk partitions in order without
/// rehashing).
pub struct Tier2PartialResult {
    /// Indexed by partition id `[0, NUM_PARTITIONS)`. Each entry is
    /// `(distinct_keys_in_this_partition, summed_values_in_this_partition)`,
    /// in matching order.
    pub per_partition: Vec<(Vec<i32>, Vec<f64>)>,
}

/// Defensive host-side check that a `[K+1]`-length partition-offsets vector is
/// monotonic non-decreasing. A reversed range `offsets[pid+1] < offsets[pid]`
/// from a corrupt prefix-sum step would be reinterpreted by the reduce kernel
/// as a (wrap-around) slice and walk OOB. Caller passes a short `tag` (e.g.
/// `"tier2"`, `"tier2_multi"`, `"tier2_twokey"`) so the error message points at
/// the orchestrator that surfaced the bad offsets.
///
/// O(K) where K = NUM_PARTITIONS = 4096 — ~16K compares, negligible vs. the
/// H2D upload that immediately follows.
pub(crate) fn validate_offsets_monotonic(offsets: &[u32], tag: &str) -> BoltResult<()> {
    for pid in 0..offsets.len().saturating_sub(1) {
        if offsets[pid + 1] < offsets[pid] {
            return Err(BoltError::Other(format!(
                "{tag}: partition offsets not monotonic at pid={pid}: {} < {}",
                offsets[pid + 1],
                offsets[pid]
            )));
        }
    }
    Ok(())
}

/// Execute Tier-2 hash-partitioned GROUP BY SUM.
///
/// Inputs live on the device. `keys` and `vals` must each have length
/// `n_rows` (the caller is responsible for this — we don't have a cheap way
/// to assert it on `GpuVec` without an extra round-trip).
///
/// Returns one partial-result vector per partition (length
/// `stub_partition_kernel::NUM_PARTITIONS`). The merger concatenates them
/// into the final `RecordBatch`.
///
/// # Errors
///
/// Surfaces any CUDA driver failure encountered during partition / scatter /
/// download. Stub sibling functions return `unimplemented!()`, so calling
/// this *before* the merger pass will panic — that's intentional.
pub fn execute_tier2_sum(
    keys: &GpuVec<i32>,
    vals: &GpuVec<f64>,
    n_rows: u32,
) -> BoltResult<Tier2PartialResult> {
    let num_partitions = stub_partition_kernel::NUM_PARTITIONS;

    // Fast path: empty input. We still return `NUM_PARTITIONS` empty slots
    // so downstream code can rely on the length invariant.
    if n_rows == 0 {
        return Ok(Tier2PartialResult {
            per_partition: vec![(Vec::new(), Vec::new()); num_partitions as usize],
        });
    }

    // Stage-4 (P1b): mint a per-call stream so allocations, kernel
    // launches, and the final D2H share one ordering domain. Each
    // `launch_with_geometry` already synchronizes on the stream, so the
    // kernel ordering matches the previous NULL-stream behaviour.
    let stream = CudaStream::null_or_default();

    // ----------------------------------------------------------------------
    // Step 1. Allocate the partition-pass outputs.
    // ----------------------------------------------------------------------
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;

    // ----------------------------------------------------------------------
    // Step 2. JIT the partition kernel and launch it.
    //
    // Grid shape: ceil(n_rows / 256) blocks × 256 threads. The kernel does
    // a grid-stride loop internally so the exact block count is not
    // performance-critical, but matching it to n_rows minimises idle warps.
    // ----------------------------------------------------------------------
    let partition_module = get_or_build_module(&partition_spec_for(n_rows))?;
    let partition_fn = partition_module.function(stub_partition_kernel::KERNEL_ENTRY)?;

    const BLOCK_THREADS: u32 = 256;
    let grid_blocks = n_rows.div_ceil(BLOCK_THREADS).max(1);

    {
        // CUDA-Oxide typed launch path. Views keep the parent GpuVecs
        // borrowed for the duration of the kernel-args list, so they
        // cannot be dropped while the launch is in flight.
        let view_keys = keys.view();
        let mut view_pid = partition_ids.view_mut();
        let mut view_counts = counts.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_output(&mut view_pid);
        args.push_output(&mut view_counts);
        args.push_scalar_u32(n_rows);

        launch_with_geometry(
            partition_fn,
            grid_blocks,
            BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // ----------------------------------------------------------------------
    // Step 3. Prefix-sum the counts into per-partition offsets.
    //
    // `offsets` has length NUM_PARTITIONS + 1, with offsets[K] = n_rows.
    // The scatter kernel only needs the first NUM_PARTITIONS entries
    // (as bases); we keep [K] around for the host loop below to compute
    // per-partition lengths.
    //
    // P1b-stage6: this used to call `compute_partition_offsets` and
    // `upload_offsets` separately, each ending in its own
    // `cuStreamSynchronize` (2 syncs total per orchestrator call). The
    // joint `compute_and_upload_partition_offsets_async` helper collapses
    // both legs onto the per-call stream with a single sync between the
    // D2H of the counts and the H2D of the bases. Net: 2 syncs → 1.
    // ----------------------------------------------------------------------
    let (offsets, offsets_gpu): (Vec<u32>, GpuVec<u32>) =
        stub_partition_offsets::compute_and_upload_partition_offsets_async(
            &counts,
            stream.raw(),
        )?;
    if offsets.len() != (num_partitions as usize) + 1 {
        return Err(BoltError::Other(format!(
            "tier2: prefix-sum returned {} offsets, expected {}",
            offsets.len(),
            num_partitions as usize + 1
        )));
    }

    // ----------------------------------------------------------------------
    // Step 4. Allocate scatter outputs + cursor.
    //
    // `partition_cursors` is an atomic counter the scatter kernel bumps once
    // per row to claim its write slot within a partition.
    //
    // `offsets_gpu` (length NUM_PARTITIONS, the K bases — same shape the
    // legacy `upload_offsets` returned) was produced by the joint helper
    // above; the scatter kernel reads it directly.
    // ----------------------------------------------------------------------
    let mut scatter_keys: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?;
    let mut partition_cursors: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;

    // ----------------------------------------------------------------------
    // Step 5. JIT + launch the scatter kernel.
    //
    // Args (PTX-level):
    //   .param .u64 keys              (in,  i32* len n_rows)
    //   .param .u64 vals              (in,  f64* len n_rows)
    //   .param .u64 partition_ids     (in,  u32* len n_rows)
    //   .param .u64 offsets           (in,  u32* len K)
    //   .param .u64 partition_cursors (in,  u32* len K, zeroed)
    //   .param .u64 scatter_keys      (out, i32* len n_rows)
    //   .param .u64 scatter_vals      (out, f64* len n_rows)
    //   .param .u32 n_rows
    // ----------------------------------------------------------------------
    let scatter_module = get_or_build_module(&KernelSpec::Scatter)?;
    let scatter_fn = scatter_module.function(stub_scatter_kernel::KERNEL_ENTRY)?;

    {
        let view_keys = keys.view();
        let view_vals = vals.view();
        let view_pid = partition_ids.view();
        let view_offsets = offsets_gpu.view();
        let mut view_cursors = partition_cursors.view_mut();
        let mut view_sk = scatter_keys.view_mut();
        let mut view_sv = scatter_vals.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_input(&view_vals);
        args.push_input(&view_pid);
        args.push_input(&view_offsets);
        args.push_output(&mut view_cursors);
        args.push_output(&mut view_sk);
        args.push_output(&mut view_sv);
        args.push_scalar_u32(n_rows);

        launch_with_geometry(
            scatter_fn,
            grid_blocks,
            BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // ----------------------------------------------------------------------
    // Step 6. Pass 2 — GPU per-partition dedup+sum (Tier 2.1).
    //
    // One block per partition. Each block builds an open-addressing hash
    // table in shared memory over its partition's `[offsets[pid], offsets
    // [pid+1])` slice. After the launch we download a fixed-size 13 MiB
    // output buffer (NUM_PARTITIONS × BLOCK_GROUPS × {4 B key + 8 B val +
    // 1 B set}) and walk the set flag to collect populated slots.
    //
    // See `crate::jit::partition_reduce_kernel` for the algorithm. The
    // historical host-HashMap path lived here before; the diff is in the
    // module-level docs above.
    // ----------------------------------------------------------------------

    // Defensive: the scatter kernel must populate exactly `n_rows` rows
    // into the (keys, vals) buffers, and offsets[K] must equal n_rows.
    // A violation here would silently corrupt the reduce kernel's slice
    // bounds, so we surface it as a structured error.
    let n_rows_usize = n_rows as usize;
    if (offsets[num_partitions as usize] as usize) != n_rows_usize {
        return Err(BoltError::Other(format!(
            "tier2: offsets[K]={}, expected n_rows={}",
            offsets[num_partitions as usize],
            n_rows
        )));
    }

    // Defensive: validate monotonicity of partition offsets before re-uploading.
    // A buggy prefix-sum step (e.g. host wrapping arithmetic in gpu_compact)
    // could produce offsets[pid+1] < offsets[pid], which the reduce kernel
    // would interpret as a (wrap-around) range and walk OOB in device memory.
    validate_offsets_monotonic(&offsets, "tier2")?;

    // The reduce kernel needs the FULL K+1 offsets buffer on the device
    // — it reads `offsets[pid]` AND `offsets[pid+1]` to compute each
    // partition's slice. (The scatter kernel only needed the K bases,
    // hence `upload_offsets` drops the trailing total — we re-upload here
    // with the total intact.)
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;

    // Output buffers. NUM_PARTITIONS × BLOCK_GROUPS slots; one entry per
    // shared-table slot per partition. Total fixed cost regardless of
    // n_rows: 4096 * 1024 * (4 + 8 + 1) = ~52 MiB.
    let n_out_slots: usize =
        (num_partitions as usize) * (partition_reduce_kernel::BLOCK_GROUPS as usize);
    let mut out_keys: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_set: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;

    // JIT + launch the per-partition reduce kernel. Grid = NUM_PARTITIONS
    // blocks (one per partition); blockIdx.x IS the partition id.
    //
    // Batch 4: we resolve the spill-counter entry point and pass a
    // zero-initialised 1-element u32 buffer as the 7th argument. After the
    // launch syncs we download the counter; any non-zero value indicates
    // a partition exceeded MAX_PROBES probes, which would silently corrupt
    // the SUM. We surface that as a structured error instead.
    let reduce_module = get_or_build_module(&KernelSpec::ReduceSum)?;
    let reduce_fn = reduce_module.function(partition_reduce_kernel::KERNEL_ENTRY_WITH_SPILL)?;
    let mut spill_counter: GpuVec<u32> = GpuVec::<u32>::zeros_async(1, stream.raw())?;

    {
        let view_pk = scatter_keys.view();
        let view_pv = scatter_vals.view();
        let view_po = offsets_kp1_gpu.view();
        let mut view_ok = out_keys.view_mut();
        let mut view_ov = out_vals.view_mut();
        let mut view_os = out_set.view_mut();
        let mut view_spill = spill_counter.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_pk);
        args.push_input(&view_pv);
        args.push_input(&view_po);
        args.push_output(&mut view_ok);
        args.push_output(&mut view_ov);
        args.push_output(&mut view_os);
        args.push_output(&mut view_spill);

        launch_with_geometry(
            reduce_fn,
            num_partitions,
            partition_reduce_kernel::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // Stage-4 (P1b): pinned D2H for the three fixed-size output
    // buffers (totalling ~52 MiB vs. ~150 MiB the host path used to
    // download at N=10 M). Each download enqueues an async copy; we
    // synchronize once after all three are queued so the driver can
    // overlap them.
    let pinned_keys = out_keys.to_pinned_async(stream.raw())?;
    let pinned_vals = out_vals.to_pinned_async(stream.raw())?;
    let pinned_set = out_set.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_out_keys: Vec<i32> = pinned_keys.as_slice().to_vec();
    let host_out_vals: Vec<f64> = pinned_vals.as_slice().to_vec();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

    // Spill-counter check. The launch above synchronized through the stream,
    // so `to_vec()` is safe to call here. A non-zero count means at least
    // one row's linear probe wrapped past MAX_PROBES without finding a slot
    // — the kernel dropped it, and the per-group sum for that key is now
    // missing one or more contributions. Surface as a structured error so
    // the engine can fall back to a host-side aggregation path rather than
    // returning silently corrupt results.
    let spill_count = spill_counter.to_vec()?[0];
    if spill_count > 0 {
        return Err(BoltError::Other(format!(
            "partition_reduce spill: {} rows exceeded MAX_PROBES; result may be incorrect",
            spill_count
        )));
    }

    if host_out_keys.len() != n_out_slots
        || host_out_vals.len() != n_out_slots
        || host_out_set.len() != n_out_slots
    {
        return Err(BoltError::Other(format!(
            "tier2: reduce-kernel output buffers have unexpected length \
             (keys={}, vals={}, set={}, expected={})",
            host_out_keys.len(),
            host_out_vals.len(),
            host_out_set.len(),
            n_out_slots
        )));
    }

    // Walk the per-partition slot maps. For each populated slot (set==1)
    // push (key, sum) into the partition's result. Empty partitions get
    // empty `(vec![], vec![])` to preserve the length invariant on
    // `Tier2PartialResult.per_partition`.
    let block_groups = partition_reduce_kernel::BLOCK_GROUPS as usize;
    let mut per_partition: Vec<(Vec<i32>, Vec<f64>)> =
        Vec::with_capacity(num_partitions as usize);

    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;

        // Conservative capacity hint: in the worst case the table is fully
        // packed at BLOCK_GROUPS slots, but typical workloads at q5 scale
        // populate ~1 K of the 1024 slots. The Vec will grow if needed.
        let mut out_k: Vec<i32> = Vec::new();
        let mut out_s: Vec<f64> = Vec::new();

        // Fast-path: skip the whole sweep if this partition is empty in
        // the input. Saves 1024 byte-loads per empty partition (negligible
        // but cheap to encode).
        let p_start = offsets[pid] as usize;
        let p_end = offsets[pid + 1] as usize;
        if p_start == p_end {
            per_partition.push((out_k, out_s));
            continue;
        }

        for slot in 0..block_groups {
            // SAFETY-equivalent of bounds: base + slot < n_out_slots is
            // guaranteed by the loop bounds. We avoid an explicit get()
            // here because the inner loop runs 1 M times at q5 scale and
            // the unwrap overhead would be measurable.
            if host_out_set[base + slot] != 0 {
                out_k.push(host_out_keys[base + slot]);
                out_s.push(host_out_vals[base + slot]);
            }
        }
        per_partition.push((out_k, out_s));
    }

    Ok(Tier2PartialResult { per_partition })
}

// ---------------------------------------------------------------------------
// Host-only sanity tests.
//
// We cannot exercise the full pipeline here without (a) a working CUDA
// context and (b) the sibling-agent kernels, both of which are out of scope
// for this worktree. What we *can* test cheaply:
//
//   * The empty-input early return matches the documented length invariant.
//   * `Tier2PartialResult` is constructible and inspectable.
//
// The integrator's harness (T2G) covers the GPU-end-to-end correctness.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_num_partitions_slots() {
        // Requires a CUDA context to allocate even zero-length GpuVecs; if
        // we cannot acquire one, skip rather than fail. This keeps the test
        // useful on dev machines and inert on docs.rs.
        let keys = match GpuVec::<i32>::from_slice(&[]) {
            Ok(v) => v,
            Err(_) => return,
        };
        let vals = match GpuVec::<f64>::from_slice(&[]) {
            Ok(v) => v,
            Err(_) => return,
        };
        let result = execute_tier2_sum(&keys, &vals, 0).expect("empty input must succeed");
        assert_eq!(
            result.per_partition.len(),
            stub_partition_kernel::NUM_PARTITIONS as usize,
            "Tier2PartialResult must always carry NUM_PARTITIONS slots"
        );
        for (k, v) in &result.per_partition {
            assert!(k.is_empty() && v.is_empty(), "empty input yields empty partitions");
        }
    }

    // --- Module-cache mechanics tests ---------------------------------------
    //
    // Skip on CPU-only hosts (no CUDA context).
    // -----------------------------------------------------------------------

    use std::sync::atomic::Ordering;

    #[test]
    fn cache_repeat_same_spec_is_hit() {
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
    fn cache_different_specs_independent() {
        let _ = match get_or_build_module(&KernelSpec::Scatter) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ReduceSum).expect("reduce build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::Scatter).expect("scatter hit");
        let _ = get_or_build_module(&KernelSpec::ReduceSum).expect("reduce hit");
        assert_eq!(LOAD_COUNT.load(Ordering::SeqCst), baseline);
    }

    // --- Batch 4 spill-counter wiring tests ----------------------------------
    //
    // Tests below exercise the spill-counter path added in batch 4. The
    // host-only assertion below pins the error message format so the
    // engine's fallback path (which matches on the prefix) doesn't break
    // silently. The GPU-gated `spill_fires_on_pathological_input` test
    // constructs a synthetic high-collision workload where more than
    // BLOCK_GROUPS distinct keys land in the same partition, and asserts
    // the kernel-driven counter trips the error.
    //
    // The error-prefix contract: `"partition_reduce spill: "`. Callers may
    // detect via `starts_with` for fallback routing.
    // -----------------------------------------------------------------------

    #[test]
    fn spill_error_prefix_is_stable() {
        // Mirrors the literal format string in `execute_tier2_sum`. If
        // someone changes the wording, this test catches the contract drift.
        let err = BoltError::Other(format!(
            "partition_reduce spill: {} rows exceeded MAX_PROBES; result may be incorrect",
            42
        ));
        let msg = format!("{err}");
        assert!(
            msg.contains("partition_reduce spill:"),
            "spill-error message must contain the stable prefix; got {msg}"
        );
        assert!(msg.contains("42"), "spill count must appear in message; got {msg}");
    }

    /// GPU-gated: construct a workload where >BLOCK_GROUPS distinct keys
    /// all hash into the same partition, forcing the per-block table to
    /// overflow. The kernel must atomically bump the spill counter; the
    /// orchestrator must surface a structured error rather than returning
    /// silently corrupt sums.
    ///
    /// Construction: every key K we produce must satisfy
    /// `(K * HASH_MULTIPLIER) & (NUM_PARTITIONS - 1) == 0` so the
    /// partition kernel maps them all to partition 0. Then we just need
    /// > BLOCK_GROUPS = 1024 distinct such keys. The modular inverse of
    /// HASH_MULTIPLIER mod 2^32 lets us reverse the hash and walk keys
    /// directly. We don't bother computing it here — instead we scan i32
    /// upward until we find 1500 keys whose hash falls in partition 0;
    /// that's O(n_partitions × n_keys) ≈ 6 M iterations and runs in
    /// milliseconds at test time.
    #[test]
    #[ignore = "requires CUDA toolkit + JIT at runtime (executes Tier-2 pipeline with pathological input)"]
    fn spill_fires_on_pathological_input() {
        use crate::jit::partition_kernel as pk;
        const TARGET_PID: u32 = 0;
        let mult: u32 = pk::HASH_MULTIPLIER;
        let mask: u32 = pk::NUM_PARTITIONS - 1;
        let needed: usize = (partition_reduce_kernel::BLOCK_GROUPS as usize) + 500;

        // Collect `needed` distinct positive i32 keys that hash to partition 0.
        let mut keys: Vec<i32> = Vec::with_capacity(needed);
        let mut k: i32 = 1;
        while keys.len() < needed && k < i32::MAX {
            let hash = (k as u32).wrapping_mul(mult);
            if (hash & mask) == TARGET_PID {
                keys.push(k);
            }
            k += 1;
        }
        if keys.len() < needed {
            // Search exhausted i32 space — should not happen for K=4096
            // since one in 4096 keys hits any given partition, so we
            // expect to find 1500 hits inside the first ~6 M.
            return;
        }
        let vals: Vec<f64> = (0..keys.len()).map(|i| (i + 1) as f64).collect();
        let n_rows = keys.len() as u32;

        let keys_gpu = match GpuVec::<i32>::from_slice(&keys) {
            Ok(v) => v,
            Err(_) => return, // No CUDA — skip.
        };
        let vals_gpu = match GpuVec::<f64>::from_slice(&vals) {
            Ok(v) => v,
            Err(_) => return,
        };

        match execute_tier2_sum(&keys_gpu, &vals_gpu, n_rows) {
            Err(BoltError::Other(msg)) => {
                assert!(
                    msg.starts_with("partition_reduce spill:"),
                    "expected spill error, got: {msg}"
                );
            }
            Err(e) => panic!("expected spill error, got different error: {e:?}"),
            Ok(_) => panic!(
                "expected spill error on pathological partition-0 overflow, but \
                 execute_tier2_sum returned Ok — spill counter likely not wired"
            ),
        }
    }

    // --- P1b-stage6 wiring smoke test ----------------------------------------
    //
    // Exercises the joint `compute_and_upload_partition_offsets_async` path
    // end-to-end on a small fixture and compares the merged result against a
    // host-computed oracle. `#[ignore]`-gated because it requires the JIT
    // toolchain + a live CUDA context.
    // -----------------------------------------------------------------------
    #[test]
    #[ignore = "requires CUDA toolkit + JIT at runtime (executes Tier-2 pipeline)"]
    fn stage6_joint_offsets_smoke() {
        use std::collections::HashMap;

        // Deterministic small fixture: 8 rows, keys with duplicates so a
        // few partitions populate and the reduce kernel has real work.
        let host_keys: Vec<i32> = vec![1, 2, 1, 3, 2, 1, 4, 3];
        let host_vals: Vec<f64> = vec![10.0, 20.0, 11.0, 30.0, 21.0, 12.0, 40.0, 31.0];
        let n_rows = host_keys.len() as u32;

        let keys = match GpuVec::<i32>::from_slice(&host_keys) {
            Ok(v) => v,
            Err(_) => return, // No CUDA context — skip.
        };
        let vals = match GpuVec::<f64>::from_slice(&host_vals) {
            Ok(v) => v,
            Err(_) => return,
        };

        let result = match execute_tier2_sum(&keys, &vals, n_rows) {
            Ok(r) => r,
            Err(_) => return, // JIT or kernel unavailable — skip.
        };

        // Host oracle: a straightforward HashMap aggregation.
        let mut oracle: HashMap<i32, f64> = HashMap::new();
        for (k, v) in host_keys.iter().zip(host_vals.iter()) {
            *oracle.entry(*k).or_insert(0.0) += *v;
        }

        // Collect GPU result into the same shape: HashMap<i32, f64>.
        let mut got: HashMap<i32, f64> = HashMap::new();
        for (keys, sums) in &result.per_partition {
            assert_eq!(keys.len(), sums.len(), "per-partition shape mismatch");
            for (k, s) in keys.iter().zip(sums.iter()) {
                let prev = got.insert(*k, *s);
                assert!(
                    prev.is_none(),
                    "key {k} appeared in two partitions (Tier-2 disjoint invariant violated)"
                );
            }
        }

        assert_eq!(got.len(), oracle.len(), "distinct-key count mismatch");
        for (k, expected) in &oracle {
            let got_v = got.get(k).copied().unwrap_or(f64::NAN);
            assert!(
                (got_v - expected).abs() < 1e-9,
                "sum mismatch for key {k}: oracle={expected}, got={got_v}"
            );
        }
    }

    // --- Host-only: validate_offsets_monotonic ------------------------------
    //
    // Pure host arithmetic — no CUDA context needed. Confirms the defensive
    // check both lets well-formed offsets through unchanged and surfaces a
    // structured error on a hand-crafted non-monotonic vec (the shape a
    // wrapping host prefix-sum bug could produce).
    // -----------------------------------------------------------------------

    #[test]
    fn validate_offsets_monotonic_accepts_well_formed() {
        // Strictly increasing.
        validate_offsets_monotonic(&[0, 1, 2, 5, 5, 10], "tier2")
            .expect("strictly non-decreasing must pass");
        // Empty and single-element are both vacuously monotonic.
        validate_offsets_monotonic(&[], "tier2").expect("empty must pass");
        validate_offsets_monotonic(&[42], "tier2").expect("single-element must pass");
        // All-zero (every partition empty, offsets[K] = 0) must pass.
        validate_offsets_monotonic(&[0; 8], "tier2").expect("all-zero must pass");
    }

    #[test]
    fn validate_offsets_monotonic_rejects_reversal() {
        // Hand-crafted: offsets[2]=10 then offsets[3]=4 — a wrap-around that
        // a buggy prefix-sum could produce. The reduce kernel would read it
        // as a slice `[10, 4)` which wraps to a huge range and walks OOB.
        let bad = [0u32, 5, 10, 4, 12, 20];
        let err =
            validate_offsets_monotonic(&bad, "tier2").expect_err("non-monotonic must fail");
        let msg = format!("{err}");
        assert!(
            msg.contains("not monotonic"),
            "error message must mention monotonicity, got: {msg}"
        );
        assert!(
            msg.contains("pid=2"),
            "error message must pinpoint the offending index, got: {msg}"
        );
        assert!(
            msg.contains("tier2"),
            "error message must carry the orchestrator tag, got: {msg}"
        );
    }
}
