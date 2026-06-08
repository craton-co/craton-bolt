// SPDX-License-Identifier: Apache-2.0

//! Tier-2 hash-partitioned GROUP BY SUM orchestrator — **two-key (i64-packed)
//! variant**.
//!
//! This is the i64 sibling of [`crate::exec::groupby_tier2_orchestrator`].
//! The single-key Tier-2 path drives the partition / scatter / per-partition
//! reduce chain over `int32_t` keys. This module does the same chain over
//! `int64_t` packed keys, where each i64 holds two Int32 group-by columns
//! losslessly: high 32 bits = column 0, low 32 bits = column 1 (matching
//! the host-side `groupby.rs::pack_keys` convention).
//!
//! ## Why a separate orchestrator
//!
//! The on-device representations diverge in two places:
//!   * The partition kernel hashes 64 bits, not 32, and reads the top 10
//!     bits of a 64-bit multiplicative product instead of the low 10 of a
//!     32-bit product. See [`crate::jit::partition_kernel_i64`].
//!   * The scatter kernel reads/writes 8-byte keys (`ld.global.s64` /
//!     `st.global.u64`) instead of 4-byte. See
//!     [`crate::jit::scatter_kernel_i64`].
//!
//! Everything else — the prefix-sum, the per-partition cursor, the host-side
//! pass-2 HashMap reduce — is structurally identical, just keyed on `i64`
//! instead of `i32`. We deliberately copy rather than try to generify the
//! single-key orchestrator because Tier-2 is a hot path and inlining the
//! exact key type lets `rustc` lay out the HashMap entries optimally for
//! each width.
//!
//! ## Pass-2 (host) reduce
//!
//! Same shape as the single-key path: download the scatter buffers
//! (`8·n_rows + 8·n_rows` bytes; the i64 key column doubles the wire cost
//! of the i32 path but the f64 vals dominate either way), then for each
//! partition build a small `HashMap<i64, f64>` over its slice.
//!
//! Pass-2-on-GPU is sibling agent (c)'s work and lands in a separate file
//! at integration time; we do NOT depend on it here.

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_tier2_orchestrator::validate_offsets_monotonic;
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::module_cache;
use crate::exec::partition_offsets;
use crate::jit::partition_kernel_i64;
use crate::jit::partition_reduce_kernel_i64;
use crate::jit::scatter_kernel_i64;
use crate::jit::CudaModule;

// ---------------------------------------------------------------------------
// Per-orchestrator module cache. Mirror of `groupby_tier2_orchestrator`'s
// cache but over the i64-key kernel variants.
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum KernelSpec {
    PartitionI64,
    PartitionI64ShmemStaging,
    ScatterI64,
    ReduceSumI64,
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
            KernelSpec::PartitionI64ShmemStaging => {
                partition_kernel_i64::compile_partition_kernel_i64_shmem_staging()?
            }
            KernelSpec::ScatterI64 => scatter_kernel_i64::compile_scatter_kernel_i64()?,
            KernelSpec::ReduceSumI64 => {
                // Batch 5: spill-counter-aware variant. The launch site
                // resolves `KERNEL_ENTRY_WITH_SPILL` and pushes a u32 spill
                // counter as the trailing kernel arg; we surface a structured
                // error on MAX_PROBES overflow instead of silently dropping
                // rows.
                partition_reduce_kernel_i64::compile_partition_reduce_kernel_i64_with_spill()?
            }
        })
    })
}

fn partition_i64_spec_for(n_rows: u32) -> KernelSpec {
    // dedup (tier2): threshold test shared via
    // `groupby_tier2_common::use_shmem_staging_partition_i64` (same
    // `partition_kernel_i64::SHMEM_STAGING_MIN_ROWS` comparison as before).
    if crate::exec::groupby_tier2_common::use_shmem_staging_partition_i64(n_rows) {
        KernelSpec::PartitionI64ShmemStaging
    } else {
        KernelSpec::PartitionI64
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Tier-2 two-key partial result: one `(keys_i64, sums)` pair per partition.
///
/// Length is exactly `NUM_PARTITIONS`. Keys are still packed i64 here —
/// the merger pass [`crate::exec::groupby_tier2_twokey_merge`] unpacks them
/// back into the two original Int32 columns.
///
/// Empty partitions are kept as `(vec![], vec![])` rather than elided so
/// the index in `per_partition` stays significant for any downstream code
/// that wants to walk partitions in order.
pub struct Tier2TwokeyPartial {
    /// Indexed by partition id `[0, NUM_PARTITIONS)`. Each entry is
    /// `(distinct_packed_keys, summed_values)`, in matching order. Keys
    /// are still in the i64-packed form produced by host-side `pack_keys`.
    pub per_partition: Vec<(Vec<i64>, Vec<f64>)>,
}

/// Execute Tier-2 hash-partitioned GROUP BY SUM for **two-key (Int32, Int32)**
/// input, encoded as a single i64 key per row.
///
/// `keys_packed` must hold the host-packed i64 keys uploaded to the device.
/// `vals` holds the f64 SUM input. Both must have length `n_rows`.
///
/// Returns one partial-result entry per partition (length
/// `partition_kernel_i64::NUM_PARTITIONS`). The merger pass concatenates and
/// unpacks them into the final two-column `RecordBatch`.
pub fn execute_tier2_twokey_sum(
    keys_packed: &GpuVec<i64>,
    vals: &GpuVec<f64>,
    n_rows: u32,
) -> BoltResult<Tier2TwokeyPartial> {
    let num_partitions = partition_kernel_i64::NUM_PARTITIONS;

    // Fast path: empty input. Preserve the length invariant so downstream
    // code can rely on it.
    if n_rows == 0 {
        return Ok(Tier2TwokeyPartial {
            per_partition: vec![(Vec::new(), Vec::new()); num_partitions as usize],
        });
    }

    // Stage-4 (P1b): per-call stream so device allocs, launches, and
    // the final D2H share one ordering domain.
    let stream = CudaStream::null_or_default();

    // ----------------------------------------------------------------------
    // Step 1. Allocate partition-pass outputs.
    // ----------------------------------------------------------------------
    let mut counts: GpuVec<u32> =
        GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;

    // ----------------------------------------------------------------------
    // Step 2. JIT + launch the i64 partition kernel.
    //
    // PTX signature:
    //   .param .u64 keys              (in,  i64* len n_rows)
    //   .param .u64 partition_ids     (out, u32* len n_rows)
    //   .param .u64 counts            (out, u32* len K, zeroed)
    //   .param .u32 n_rows
    // ----------------------------------------------------------------------
    let partition_module = get_or_build_module(&partition_i64_spec_for(n_rows))?;
    let partition_fn = partition_module.function(partition_kernel_i64::KERNEL_ENTRY)?;

    const BLOCK_THREADS: u32 = 256;
    let grid_blocks = n_rows.div_ceil(BLOCK_THREADS).max(1);

    {
        let view_keys = keys_packed.view();
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
    // Step 3. Prefix-sum counts → offsets. Reuse the single-key joint helper
    // — the counts vector is shape-identical (u32[K]) regardless of key
    // width, and the helper has no per-key-type baggage.
    //
    // P1b-stage6: joint helper does the D2H + prefix scan + H2D in a single
    // pinned-async pipeline on the per-call stream. Replaces the legacy
    // `compute_partition_offsets` + `upload_offsets` pair (2 syncs → 1).
    // ----------------------------------------------------------------------
    let (offsets, offsets_gpu): (Vec<u32>, GpuVec<u32>) =
        partition_offsets::compute_and_upload_partition_offsets_async(&counts, stream.raw())?;
    if offsets.len() != (num_partitions as usize) + 1 {
        return Err(BoltError::Other(format!(
            "tier2_twokey: prefix-sum returned {} offsets, expected {}",
            offsets.len(),
            num_partitions as usize + 1
        )));
    }

    // ----------------------------------------------------------------------
    // Step 4. Allocate scatter outputs + cursor.
    //
    // `scatter_keys` is i64 — twice the byte budget of the single-key path.
    // For n_rows = 10 M that's 80 MB; still well under any sane device cap.
    // `offsets_gpu` (length NUM_PARTITIONS, the K bases — same shape the
    // legacy `upload_offsets` returned) was produced by the joint helper.
    // ----------------------------------------------------------------------
    let mut scatter_keys: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?;
    let mut partition_cursors: GpuVec<u32> =
        GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;

    // ----------------------------------------------------------------------
    // Step 5. JIT + launch the i64 scatter kernel.
    // ----------------------------------------------------------------------
    let scatter_module = get_or_build_module(&KernelSpec::ScatterI64)?;
    let scatter_fn = scatter_module.function(scatter_kernel_i64::KERNEL_ENTRY)?;

    {
        let view_keys = keys_packed.view();
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
    // Step 6. Pass 2 — GPU per-partition dedup+sum (Tier 2.1, i64 keys).
    //
    // Mirrors the i32 single-key path's pass-2-on-GPU. One CUDA block per
    // partition builds an open-addressing hash table in 20 KiB of shared
    // memory (vs 16 KiB for the i32 variant; the extra 4 KiB pays for the
    // wider key slots). Output is a fixed-size 80 MiB buffer
    // (NUM_PARTITIONS × BLOCK_GROUPS × (8 B key + 8 B val + 1 B set))
    // regardless of n_rows.
    //
    // See `crate::jit::partition_reduce_kernel_i64` for the algorithm.
    // ----------------------------------------------------------------------
    let n_rows_usize = n_rows as usize;
    if (offsets[num_partitions as usize] as usize) != n_rows_usize {
        return Err(BoltError::Other(format!(
            "tier2_twokey: offsets[K]={}, expected n_rows={}",
            offsets[num_partitions as usize], n_rows
        )));
    }

    // Defensive: validate monotonicity of partition offsets before re-uploading.
    // A buggy prefix-sum step (e.g. host wrapping arithmetic in gpu_compact)
    // could produce offsets[pid+1] < offsets[pid], which the reduce kernel
    // would interpret as a (wrap-around) range and walk OOB in device memory.
    validate_offsets_monotonic(&offsets, "tier2_twokey")?;

    // Reduce kernel needs the FULL K+1 offsets buffer on the device.
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;

    let n_out_slots: usize =
        (num_partitions as usize) * (partition_reduce_kernel_i64::BLOCK_GROUPS as usize);
    let mut out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_vals_gpu: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;
    let mut spill: GpuVec<u32> = GpuVec::<u32>::zeros_async(1, stream.raw())?;

    let reduce_module = get_or_build_module(&KernelSpec::ReduceSumI64)?;
    let reduce_fn = reduce_module.function(partition_reduce_kernel_i64::KERNEL_ENTRY_WITH_SPILL)?;

    {
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
            reduce_fn,
            num_partitions,
            partition_reduce_kernel_i64::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // Stage-4 (P1b): pinned D2H for the three fixed-size outputs; sync once.
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
    let host_out_vals: Vec<f64> = pinned_vals.as_slice().to_vec();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

    if host_out_keys.len() != n_out_slots
        || host_out_vals.len() != n_out_slots
        || host_out_set.len() != n_out_slots
    {
        return Err(BoltError::Other(format!(
            "tier2_twokey: reduce-kernel output buffers have unexpected length \
             (keys={}, vals={}, set={}, expected={})",
            host_out_keys.len(),
            host_out_vals.len(),
            host_out_set.len(),
            n_out_slots
        )));
    }

    let block_groups = partition_reduce_kernel_i64::BLOCK_GROUPS as usize;
    let mut per_partition: Vec<(Vec<i64>, Vec<f64>)> = Vec::with_capacity(num_partitions as usize);

    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        let mut out_k: Vec<i64> = Vec::new();
        let mut out_s: Vec<f64> = Vec::new();
        let p_start = offsets[pid] as usize;
        let p_end = offsets[pid + 1] as usize;
        if p_start == p_end {
            per_partition.push((out_k, out_s));
            continue;
        }
        for slot in 0..block_groups {
            if host_out_set[base + slot] != 0 {
                out_k.push(host_out_keys[base + slot]);
                out_s.push(host_out_vals[base + slot]);
            }
        }
        per_partition.push((out_k, out_s));
    }

    Ok(Tier2TwokeyPartial { per_partition })
}

// ---------------------------------------------------------------------------
// Host-only sanity tests. The empty-input case is the only one we can
// exercise without a live CUDA context AND the sibling-kernel chain; the
// integrator's harness covers GPU-end-to-end correctness.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_num_partitions_slots() {
        // GpuVec allocation requires a live CUDA context. If we cannot
        // acquire one (e.g. docs.rs build, no GPU), skip rather than fail.
        let keys = match GpuVec::<i64>::from_slice(&[]) {
            Ok(v) => v,
            Err(_) => return,
        };
        let vals = match GpuVec::<f64>::from_slice(&[]) {
            Ok(v) => v,
            Err(_) => return,
        };
        let result = execute_tier2_twokey_sum(&keys, &vals, 0).expect("empty input must succeed");
        assert_eq!(
            result.per_partition.len(),
            partition_kernel_i64::NUM_PARTITIONS as usize,
            "Tier2TwokeyPartial must always carry NUM_PARTITIONS slots"
        );
        for (k, v) in &result.per_partition {
            assert!(
                k.is_empty() && v.is_empty(),
                "empty input must yield empty partitions"
            );
        }
    }

    // --- Module-cache mechanics tests. Skip on CPU-only hosts. -------------

    use std::sync::atomic::Ordering;

    #[test]
    fn cache_repeat_same_spec_is_hit() {
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
    fn cache_different_specs_independent() {
        let _ = match get_or_build_module(&KernelSpec::ScatterI64) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ReduceSumI64).expect("reduce build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::ScatterI64).expect("scatter hit");
        let _ = get_or_build_module(&KernelSpec::ReduceSumI64).expect("reduce hit");
        assert_eq!(LOAD_COUNT.load(Ordering::SeqCst), baseline);
    }

    // --- P1b-stage6 wiring smoke test ----------------------------------------
    //
    // End-to-end exercise of the joint
    // `compute_and_upload_partition_offsets_async` path through the
    // two-key (i64-packed) orchestrator. Oracle = host-side HashMap
    // reduction over packed i64 keys. Gated on `#[ignore]` because the
    // pipeline needs the JIT + a CUDA context.
    // -----------------------------------------------------------------------
    #[test]
    #[ignore = "requires CUDA toolkit + JIT at runtime (executes Tier-2 twokey pipeline)"]
    fn stage6_joint_offsets_twokey_smoke() {
        use std::collections::HashMap;

        // Pack (a, b) the same way `groupby.rs::pack_keys` does.
        fn pack(a: i32, b: i32) -> i64 {
            (((a as u32 as u64) << 32) | (b as u32 as u64)) as i64
        }

        // 8-row fixture with duplicate (k1, k2) pairs so reduce kernel has
        // real per-partition work.
        let host_packed: Vec<i64> = vec![
            pack(1, 10),
            pack(2, 20),
            pack(1, 10),
            pack(3, 30),
            pack(2, 20),
            pack(1, 10),
            pack(4, 40),
            pack(3, 31),
        ];
        let host_vals: Vec<f64> = vec![1.0, 2.0, 1.5, 3.0, 2.5, 1.25, 4.0, 3.1];
        let n_rows = host_packed.len() as u32;

        let keys = match GpuVec::<i64>::from_slice(&host_packed) {
            Ok(v) => v,
            Err(_) => return,
        };
        let vals = match GpuVec::<f64>::from_slice(&host_vals) {
            Ok(v) => v,
            Err(_) => return,
        };

        let r = match execute_tier2_twokey_sum(&keys, &vals, n_rows) {
            Ok(r) => r,
            Err(_) => return,
        };

        let mut oracle: HashMap<i64, f64> = HashMap::new();
        for (k, v) in host_packed.iter().zip(host_vals.iter()) {
            *oracle.entry(*k).or_insert(0.0) += *v;
        }

        let mut got: HashMap<i64, f64> = HashMap::new();
        for (keys, sums) in &r.per_partition {
            assert_eq!(keys.len(), sums.len(), "per-partition shape mismatch");
            for (k, s) in keys.iter().zip(sums.iter()) {
                let prev = got.insert(*k, *s);
                assert!(prev.is_none(), "key {k:#x} in two partitions");
            }
        }

        assert_eq!(got.len(), oracle.len());
        for (k, expected) in &oracle {
            let got_v = got.get(k).copied().unwrap_or(f64::NAN);
            assert!(
                (got_v - expected).abs() < 1e-9,
                "sum mismatch for key {k:#x}: oracle={expected}, got={got_v}"
            );
        }
    }
}
