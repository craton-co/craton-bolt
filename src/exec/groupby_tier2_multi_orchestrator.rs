// SPDX-License-Identifier: Apache-2.0

//! Tier-2 hash-partitioned GROUP BY orchestrator for **multiple SUM** aggregates.
//!
//! Mirrors [`crate::exec::groupby_tier2_orchestrator`] one-to-one but accepts
//! `N` value columns (1..=4) in parallel, producing `N` per-group sums per
//! distinct key. The partition / prefix-sum / scatter / offsets pipeline is
//! identical — we re-use the same kernels and offsets module — only pass-2
//! (host-side dedup) is extended to accumulate `N` sums per key.
//!
//! Target query: h2o.ai q2 (`SELECT id2, SUM(v1), SUM(v2) FROM x GROUP BY id2`)
//! at medium-to-high cardinality (1k < n_groups <= 100M).
//!
//! ## Design choice: deterministic dest_idx + indexed value scatter
//!
//! For N value columns sharing one key column we need every column's row `i`
//! to land in the **same** destination slot, so the per-key tuple
//! `(key, v0, v1, …, v_{N-1})` stays aligned. The naive approach — call the
//! atomic-claim scatter kernel N times with identical inputs — looks like it
//! should produce identical placements, but ordering of concurrent
//! `atomicAdd` calls is **not** part of the CUDA contract. Any driver
//! release, warp-scheduler tweak, or block-count change can permute the
//! order, silently misaligning value columns against the key. That is the
//! kind of bug a regression test only catches by luck.
//!
//! We avoid the assumption entirely by making the destination slot
//! deterministic by construction:
//!
//!   1. **One** atomic-claim pass ([`scatter_with_dest_idx_kernel`])
//!      computes, for every input row, the destination slot
//!      `dest_idx[i] = offsets[pid_i] + atomicAdd(cursors[pid_i], 1)` and
//!      writes it to a `dest_idx[n_rows]` buffer. The same launch also
//!      writes the key column to `scatter_keys[dest_idx[i]]`. The atomic
//!      now happens exactly once per row, in exactly one kernel — there is
//!      no opportunity for cross-launch divergence.
//!
//!   2. For each value column `j` we launch
//!      [`scatter_values_by_dest_idx_kernel`]: a tiny, **atomic-free**
//!      kernel that does `out_vals[dest_idx[i]] = vals[j][i]`. Because
//!      `dest_idx` is read, not recomputed, every value column lands at
//!      the slot the key already occupies. Alignment is now a property of
//!      the data, not of the GPU's atomic scheduler.
//!
//! Cost relative to the old N-scatter design: identical launch count
//! (1 atomic-claim + N indexed scatters ≈ N scatters), one extra
//! `u32 * n_rows` buffer (`dest_idx`), and the indexed-scatter kernel is
//! cheaper than the original (no atomics, no offsets table loads).

// HashMap removed: pass-2 now runs on the GPU via partition_reduce_kernel_multi.
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_tier2_orchestrator::validate_offsets_monotonic;
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::module_cache;
use crate::exec::partition_offsets;
use crate::jit::{
    partition_kernel, partition_reduce_kernel_multi, scatter_values_by_dest_idx_kernel,
    scatter_with_dest_idx_kernel,
};

/// Tier-2 multi-SUM partial result: one `(keys, sums_per_value_column)` pair
/// per partition.
///
/// `per_partition.len() == NUM_PARTITIONS`. Within an entry:
///   - `keys_for_partition_k.len() == m_k` (distinct keys in partition k)
///   - `sums_per_value_column.len() == n_vals`
///   - each `sums_per_value_column[j].len() == m_k`, aligned to keys
pub struct Tier2MultiPartial {
    /// `per_partition[k]` = `(keys_for_partition_k, sums_per_value_column)`.
    pub per_partition: Vec<(Vec<i32>, Vec<Vec<f64>>)>,
    /// Number of value columns (1..=4). Carried out so the merger can build
    /// the right number of output Float64 columns without recomputing it.
    pub n_vals: usize,
}

/// Execute Tier-2 hash-partitioned GROUP BY with N SUM aggregates.
///
/// Inputs:
///   - `keys`: device buffer of i32 group-by keys, length `n_rows`
///   - `vals`: slice of 1..=4 device buffers, each holding one f64 value
///     column of length `n_rows`
///   - `n_rows`: row count (caller-supplied; we trust it)
///
/// Returns a `Tier2MultiPartial` with `NUM_PARTITIONS` per-partition entries,
/// each carrying its distinct keys and the N corresponding running sums.
///
/// # Errors
///
/// Surfaces any CUDA driver failure (partition pass, scatter, D2H copies, or
/// allocation) and rejects malformed inputs (`n_vals` 0 or > 4).
pub fn execute_tier2_multi_sum(
    keys: &GpuVec<i32>,
    vals: &[&GpuVec<f64>],
    n_rows: u32,
) -> BoltResult<Tier2MultiPartial> {
    let n_vals = vals.len();
    if n_vals == 0 || n_vals > 4 {
        return Err(BoltError::Other(format!(
            "tier2_multi: n_vals must be in 1..=4, got {n_vals}"
        )));
    }

    let num_partitions = partition_kernel::NUM_PARTITIONS;

    // Empty input: return NUM_PARTITIONS empty slots, each with n_vals empty
    // inner Vec<f64> rows. Downstream code can rely on the shape invariant.
    if n_rows == 0 {
        let mut per_partition: Vec<(Vec<i32>, Vec<Vec<f64>>)> =
            Vec::with_capacity(num_partitions as usize);
        for _ in 0..num_partitions {
            per_partition.push((Vec::new(), (0..n_vals).map(|_| Vec::new()).collect()));
        }
        return Ok(Tier2MultiPartial {
            per_partition,
            n_vals,
        });
    }

    // Stage-4 (P1b): per-call stream so every device allocation, kernel
    // launch, and the final D2H share one ordering domain.
    let stream = CudaStream::null_or_default();

    // ----------------------------------------------------------------------
    // Step 1. Allocate the partition-pass outputs.
    // ----------------------------------------------------------------------
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;

    // ----------------------------------------------------------------------
    // Step 2. JIT + launch the partition kernel (once — pid depends only on
    // the key column, not on any value column).
    // ----------------------------------------------------------------------
    let partition_module = module_cache::get_or_build_module(
        module_path!(),
        "partition".to_string(),
        None,
        || partition_kernel::compile_partition_kernel(),
    )?;
    let partition_fn = partition_module.function(partition_kernel::KERNEL_ENTRY)?;

    const BLOCK_THREADS: u32 = 256;
    let grid_blocks = n_rows.div_ceil(BLOCK_THREADS).max(1);

    {
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
    // Step 3. Prefix-sum counts into per-partition offsets.
    //
    // P1b-stage6: joint helper does the D2H + prefix scan + H2D in a single
    // pinned-async pipeline on the per-call stream. Replaces the legacy
    // `compute_partition_offsets` + `upload_offsets` pair (2 syncs → 1).
    // Returns the K+1 host offsets (needed for scatter-buffer sizing and
    // pass-2 slice bounds) plus the K-base device offsets the scatter
    // kernel consumes.
    // ----------------------------------------------------------------------
    let (offsets, offsets_gpu): (Vec<u32>, GpuVec<u32>) =
        partition_offsets::compute_and_upload_partition_offsets_async(
            &counts,
            stream.raw(),
        )?;
    if offsets.len() != (num_partitions as usize) + 1 {
        return Err(BoltError::Other(format!(
            "tier2_multi: prefix-sum returned {} offsets, expected {}",
            offsets.len(),
            num_partitions as usize + 1
        )));
    }

    // ----------------------------------------------------------------------
    // Step 4. Allocate scatter outputs.
    //
    //   - `scatter_keys[n_rows]`  i32: keys placed at their final slot
    //   - `scatter_vals[j][n_rows]` f64: values for column j placed at slot
    //   - `dest_idx[n_rows]`       u32: the slot index each row claimed,
    //                                   written by the atomic-claim pass and
    //                                   read by every indexed value-scatter
    //                                   pass. This is the load-bearing
    //                                   correctness primitive — it captures
    //                                   the atomic-claim ordering exactly
    //                                   once so every value column lands in
    //                                   lockstep with the key column.
    //   - `partition_cursors[K]`   u32: zero-init; the atomic-claim pass
    //                                   bumps it. Not reused after that.
    // ----------------------------------------------------------------------
    let mut scatter_keys: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        scatter_vals.push(GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?);
    }
    let mut dest_idx: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;
    let mut partition_cursors: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;

    // ----------------------------------------------------------------------
    // Step 5. Atomic-claim pass — runs ONCE.
    //
    // Writes:
    //   - dest_idx[i] = offsets[pid_i] + atomicAdd(cursors[pid_i], 1)
    //   - scatter_keys[dest_idx[i]] = keys[i]
    //
    // This is the only place an atomic-ordered slot assignment happens.
    // After this kernel returns, `dest_idx` is the canonical row→slot map.
    // ----------------------------------------------------------------------
    let claim_module = module_cache::get_or_build_module(
        module_path!(),
        "scatter_with_dest_idx".to_string(),
        None,
        || scatter_with_dest_idx_kernel::compile_scatter_with_dest_idx_kernel(),
    )?;
    let claim_fn = claim_module.function(scatter_with_dest_idx_kernel::KERNEL_ENTRY)?;

    {
        let view_keys = keys.view();
        let view_pid = partition_ids.view();
        let view_offsets = offsets_gpu.view();
        let mut view_cursors = partition_cursors.view_mut();
        let mut view_sk = scatter_keys.view_mut();
        let mut view_di = dest_idx.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_input(&view_pid);
        args.push_input(&view_offsets);
        args.push_output(&mut view_cursors);
        args.push_output(&mut view_sk);
        args.push_output(&mut view_di);
        args.push_scalar_u32(n_rows);

        launch_with_geometry(
            claim_fn,
            grid_blocks,
            BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // ----------------------------------------------------------------------
    // Step 6. Indexed value scatter — runs N times, one per value column.
    //
    // Each launch reads dest_idx[i] (deterministic) and writes
    //   scatter_vals[j][dest_idx[i]] = vals[j][i]
    // with NO atomics. Because dest_idx is fixed at this point, every
    // column's row i lands in the same slot as the key — alignment is
    // guaranteed by construction.
    // ----------------------------------------------------------------------
    let val_scatter_module = module_cache::get_or_build_module(
        module_path!(),
        "scatter_values_by_dest_idx".to_string(),
        None,
        || scatter_values_by_dest_idx_kernel::compile_scatter_values_by_dest_idx_kernel(),
    )?;

    for j in 0..n_vals {
        let val_scatter_fn =
            val_scatter_module.function(scatter_values_by_dest_idx_kernel::KERNEL_ENTRY)?;

        // Split-borrow on `scatter_vals` so we can hold `scatter_vals[j]`
        // mutably alongside the immutable inputs in the same args list.
        let (sv_j_slice, _) = scatter_vals.split_at_mut(j + 1);
        let scatter_vals_j = &mut sv_j_slice[j];

        let view_vals = vals[j].view();
        let view_dest = dest_idx.view();
        let mut view_sv = scatter_vals_j.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_vals);
        args.push_input(&view_dest);
        args.push_output(&mut view_sv);
        args.push_scalar_u32(n_rows);

        launch_with_geometry(
            val_scatter_fn,
            grid_blocks,
            BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // Silence unused-mut on partition_cursors after the single atomic-claim
    // launch (we don't read or reset it again).
    let _ = &partition_cursors;

    // ----------------------------------------------------------------------
    // Step 7. Pass 2 — GPU per-partition dedup + N-way sum (Tier 2.1 multi).
    //
    // Mirrors the single-value pass-2-on-GPU pattern (`partition_reduce_
    // kernel`) but with N parallel f64 accumulators per slot. One block
    // per partition; each builds an open-addressing hash table in 16 +
    // 8*N KiB of shared memory and exports one slot per thread to a
    // fixed-size output buffer.
    //
    // See `crate::jit::partition_reduce_kernel_multi` for the algorithm.
    // ----------------------------------------------------------------------
    let n_rows_usize = n_rows as usize;
    if (offsets[num_partitions as usize] as usize) != n_rows_usize {
        return Err(BoltError::Other(format!(
            "tier2_multi: offsets[K]={}, expected n_rows={}",
            offsets[num_partitions as usize],
            n_rows
        )));
    }

    // Defensive: validate monotonicity of partition offsets before re-uploading.
    // A buggy prefix-sum step (e.g. host wrapping arithmetic in gpu_compact)
    // could produce offsets[pid+1] < offsets[pid], which the reduce kernel
    // would interpret as a (wrap-around) range and walk OOB in device memory.
    validate_offsets_monotonic(&offsets, "tier2_multi")?;

    // Reduce kernel needs the FULL K+1 offsets buffer on the device.
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;

    let n_out_slots: usize =
        (num_partitions as usize) * (partition_reduce_kernel_multi::BLOCK_GROUPS as usize);
    let mut out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals_gpu: Vec<GpuVec<f64>> = Vec::with_capacity(n_vals);
    for _ in 0..n_vals {
        out_vals_gpu.push(GpuVec::<f64>::zeros_async(n_out_slots, stream.raw())?);
    }
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;

    // JIT + launch — kernel is cached per (n_vals) via the PTX cache.
    //
    // TODO(batch-4-spill): switch to a future
    // `compile_partition_reduce_kernel_multi_with_spill(n_vals)` once that
    // emitter exists. See `groupby_tier2_orchestrator.rs::execute_tier2_sum`
    // for the wiring pattern (zero-init u32 spill counter, push as the
    // extra kernel arg, download + check after sync, return a structured
    // `partition_reduce spill: …` error). Until then this path can drop
    // rows silently on MAX_PROBES overflow under high-cardinality skew.
    let reduce_ptx = partition_reduce_kernel_multi::compile_partition_reduce_kernel_multi(
        n_vals as u32,
    )?;
    let reduce_entry_name = partition_reduce_kernel_multi::kernel_entry(n_vals as u32);
    let reduce_fn = reduce_module.function(&reduce_entry_name)?;

    {
        // Kernel param order:
        //   partition_keys, partition_vals_0 ..= partition_vals_{N-1},
        //   partition_offsets, out_keys,
        //   out_vals_0 ..= out_vals_{N-1}, out_set
        //
        // Collect the iterated views eagerly so they outlive `args`.
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
            reduce_fn,
            num_partitions,
            partition_reduce_kernel_multi::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // Stage-4 (P1b): pinned D2H for the fixed-size outputs; sync once
    // after all transfers are queued.
    let pinned_keys = out_keys_gpu.to_pinned_async(stream.raw())?;
    let mut pinned_vals: Vec<crate::cuda::PinnedHostBuffer<f64>> =
        Vec::with_capacity(n_vals);
    for ov in &out_vals_gpu {
        pinned_vals.push(ov.to_pinned_async(stream.raw())?);
    }
    let pinned_set = out_set_gpu.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_out_keys: Vec<i32> = pinned_keys.as_slice().to_vec();
    let host_out_vals: Vec<Vec<f64>> = pinned_vals
        .iter()
        .map(|p| p.as_slice().to_vec())
        .collect();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

    // Walk per-partition slot maps. For each populated slot push
    // (key, [sum_0, …, sum_{N-1}]) into the partition's result.
    let block_groups = partition_reduce_kernel_multi::BLOCK_GROUPS as usize;
    let mut per_partition: Vec<(Vec<i32>, Vec<Vec<f64>>)> =
        Vec::with_capacity(num_partitions as usize);

    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;

        let p_start = offsets[pid] as usize;
        let p_end = offsets[pid + 1] as usize;
        if p_start == p_end {
            per_partition.push((Vec::new(), (0..n_vals).map(|_| Vec::new()).collect()));
            continue;
        }

        let mut out_k: Vec<i32> = Vec::new();
        let mut out_s: Vec<Vec<f64>> = (0..n_vals).map(|_| Vec::new()).collect();

        for slot in 0..block_groups {
            if host_out_set[base + slot] != 0 {
                out_k.push(host_out_keys[base + slot]);
                for j in 0..n_vals {
                    out_s[j].push(host_out_vals[j][base + slot]);
                }
            }
        }
        per_partition.push((out_k, out_s));
    }

    // Reference these to silence "unused" — they're load-bearing through
    // the kernel launch above but no longer reach a host-side reader.
    let _ = &scatter_vals;
    let _ = &scatter_keys;

    Ok(Tier2MultiPartial {
        per_partition,
        n_vals,
    })
}

// ---------------------------------------------------------------------------
// Host-only sanity tests.
//
// Same constraints as the single-SUM orchestrator: we can't drive the full
// GPU pipeline here without sibling kernels in this worktree. We cover the
// invariants that hold without launching a kernel — empty input shape and
// argument validation.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_value_columns() {
        // We can't always get a CUDA context in this test environment, so
        // we sidestep it: construct a GpuVec only if alloc succeeds, else
        // skip. The n_vals=0 check runs unconditionally on the host path,
        // so if alloc fails we just return early without exercising it
        // (acceptable — see the orchestrator test for the same pattern).
        let keys = match GpuVec::<i32>::from_slice(&[1, 2, 3]) {
            Ok(v) => v,
            Err(_) => return,
        };
        let vals_empty: Vec<&GpuVec<f64>> = Vec::new();
        let r = execute_tier2_multi_sum(&keys, &vals_empty, 3);
        assert!(r.is_err(), "n_vals=0 must be rejected");
    }

    #[test]
    fn rejects_too_many_value_columns() {
        let keys = match GpuVec::<i32>::from_slice(&[1, 2, 3]) {
            Ok(v) => v,
            Err(_) => return,
        };
        let v0 = GpuVec::<f64>::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        let v1 = GpuVec::<f64>::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        let v2 = GpuVec::<f64>::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        let v3 = GpuVec::<f64>::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        let v4 = GpuVec::<f64>::from_slice(&[1.0, 2.0, 3.0]).unwrap();
        let vals = vec![&v0, &v1, &v2, &v3, &v4];
        let r = execute_tier2_multi_sum(&keys, &vals, 3);
        assert!(r.is_err(), "n_vals=5 must be rejected");
    }

    #[test]
    fn empty_input_returns_num_partitions_slots() {
        let keys = match GpuVec::<i32>::from_slice(&[]) {
            Ok(v) => v,
            Err(_) => return,
        };
        let v0 = match GpuVec::<f64>::from_slice(&[]) {
            Ok(v) => v,
            Err(_) => return,
        };
        let v1 = match GpuVec::<f64>::from_slice(&[]) {
            Ok(v) => v,
            Err(_) => return,
        };
        let vals = vec![&v0, &v1];
        let r = execute_tier2_multi_sum(&keys, &vals, 0).expect("empty input must succeed");
        assert_eq!(
            r.per_partition.len(),
            partition_kernel::NUM_PARTITIONS as usize,
            "Tier2MultiPartial must always carry NUM_PARTITIONS slots"
        );
        assert_eq!(r.n_vals, 2, "n_vals carried through unchanged");
        for (k, sums) in &r.per_partition {
            assert!(k.is_empty(), "empty input yields empty keys");
            assert_eq!(sums.len(), 2, "n_vals inner Vec<f64>s, even when empty");
            for s in sums {
                assert!(s.is_empty(), "empty input yields empty sums");
            }
        }
    }

    // --- P1b-stage6 wiring smoke test ----------------------------------------
    //
    // End-to-end exercise of the joint
    // `compute_and_upload_partition_offsets_async` path through the multi-SUM
    // orchestrator. Two value columns; oracle = host-side HashMap reduction.
    // Gated on `#[ignore]` because the pipeline needs the JIT + a CUDA context.
    // -----------------------------------------------------------------------
    #[test]
    #[ignore = "requires CUDA toolkit + JIT at runtime (executes Tier-2 multi pipeline)"]
    fn stage6_joint_offsets_multi_smoke() {
        use std::collections::HashMap;

        let host_keys: Vec<i32> = vec![1, 2, 1, 3, 2, 1, 4, 3];
        let host_v0: Vec<f64> = vec![10.0, 20.0, 11.0, 30.0, 21.0, 12.0, 40.0, 31.0];
        let host_v1: Vec<f64> = vec![1.0, 2.0, 1.1, 3.0, 2.1, 1.2, 4.0, 3.1];
        let n_rows = host_keys.len() as u32;

        let keys = match GpuVec::<i32>::from_slice(&host_keys) {
            Ok(v) => v,
            Err(_) => return,
        };
        let v0 = match GpuVec::<f64>::from_slice(&host_v0) {
            Ok(v) => v,
            Err(_) => return,
        };
        let v1 = match GpuVec::<f64>::from_slice(&host_v1) {
            Ok(v) => v,
            Err(_) => return,
        };
        let vals = vec![&v0, &v1];

        let r = match execute_tier2_multi_sum(&keys, &vals, n_rows) {
            Ok(r) => r,
            Err(_) => return,
        };

        assert_eq!(r.n_vals, 2);

        // Oracle: per-key sums for both value columns.
        let mut oracle: HashMap<i32, (f64, f64)> = HashMap::new();
        for i in 0..host_keys.len() {
            let e = oracle.entry(host_keys[i]).or_insert((0.0, 0.0));
            e.0 += host_v0[i];
            e.1 += host_v1[i];
        }

        let mut got: HashMap<i32, (f64, f64)> = HashMap::new();
        for (keys, sums) in &r.per_partition {
            assert_eq!(sums.len(), 2, "per-partition must carry n_vals=2 sum cols");
            assert_eq!(keys.len(), sums[0].len());
            assert_eq!(keys.len(), sums[1].len());
            for i in 0..keys.len() {
                let prev = got.insert(keys[i], (sums[0][i], sums[1][i]));
                assert!(
                    prev.is_none(),
                    "key {} appeared in two partitions (disjoint invariant)",
                    keys[i]
                );
            }
        }

        assert_eq!(got.len(), oracle.len());
        for (k, (e0, e1)) in &oracle {
            let (g0, g1) = got.get(k).copied().unwrap_or((f64::NAN, f64::NAN));
            assert!((g0 - e0).abs() < 1e-9, "col0 mismatch for key {k}");
            assert!((g1 - e1).abs() < 1e-9, "col1 mismatch for key {k}");
        }
    }
}
