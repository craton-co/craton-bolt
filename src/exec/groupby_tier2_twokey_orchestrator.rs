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
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::partition_offsets;
use crate::jit::partition_kernel_i64;
use crate::jit::partition_reduce_kernel_i64;
use crate::jit::scatter_kernel_i64;
use crate::jit::CudaModule;

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

    // ----------------------------------------------------------------------
    // Step 1. Allocate partition-pass outputs.
    // ----------------------------------------------------------------------
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros(n_rows as usize)?;

    // ----------------------------------------------------------------------
    // Step 2. JIT + launch the i64 partition kernel.
    //
    // PTX signature:
    //   .param .u64 keys              (in,  i64* len n_rows)
    //   .param .u64 partition_ids     (out, u32* len n_rows)
    //   .param .u64 counts            (out, u32* len K, zeroed)
    //   .param .u32 n_rows
    // ----------------------------------------------------------------------
    let partition_ptx = partition_kernel_i64::compile_partition_kernel_i64()?;
    let partition_module = CudaModule::from_ptx(&partition_ptx)?;
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

        let stream = CudaStream::null();
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
    // Step 3. Prefix-sum counts → offsets. Reuse the single-key helper —
    // the counts vector is shape-identical (u32[K]).
    // ----------------------------------------------------------------------
    let offsets: Vec<u32> = partition_offsets::compute_partition_offsets(&counts)?;
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
    // ----------------------------------------------------------------------
    let mut scatter_keys: GpuVec<i64> = GpuVec::<i64>::zeros(n_rows as usize)?;
    let mut scatter_vals: GpuVec<f64> = GpuVec::<f64>::zeros(n_rows as usize)?;
    let mut partition_cursors: GpuVec<u32> = GpuVec::<u32>::zeros(num_partitions as usize)?;

    // Upload the K bases (drop the trailing total — `upload_offsets` slices
    // internally and would reject the K-length form).
    let offsets_gpu: GpuVec<u32> = partition_offsets::upload_offsets(&offsets)?;

    // ----------------------------------------------------------------------
    // Step 5. JIT + launch the i64 scatter kernel.
    // ----------------------------------------------------------------------
    let scatter_ptx = scatter_kernel_i64::compile_scatter_kernel_i64()?;
    let scatter_module = CudaModule::from_ptx(&scatter_ptx)?;
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

        let stream = CudaStream::null();
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

    // Reduce kernel needs the FULL K+1 offsets buffer on the device.
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice(&offsets)?;

    let n_out_slots: usize =
        (num_partitions as usize) * (partition_reduce_kernel_i64::BLOCK_GROUPS as usize);
    let mut out_keys_gpu: GpuVec<i64> = GpuVec::<i64>::zeros(n_out_slots)?;
    let mut out_vals_gpu: GpuVec<f64> = GpuVec::<f64>::zeros(n_out_slots)?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros(n_out_slots)?;

    let reduce_ptx = partition_reduce_kernel_i64::compile_partition_reduce_kernel_i64()?;
    let reduce_module = CudaModule::from_ptx(&reduce_ptx)?;
    let reduce_fn = reduce_module.function(partition_reduce_kernel_i64::KERNEL_ENTRY)?;

    {
        let view_pk = scatter_keys.view();
        let view_pv = scatter_vals.view();
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

        let stream = CudaStream::null();
        launch_with_geometry(
            reduce_fn,
            num_partitions,
            partition_reduce_kernel_i64::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    let host_out_keys: Vec<i64> = out_keys_gpu.to_vec()?;
    let host_out_vals: Vec<f64> = out_vals_gpu.to_vec()?;
    let host_out_set: Vec<u8> = out_set_gpu.to_vec()?;

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
    let mut per_partition: Vec<(Vec<i64>, Vec<f64>)> =
        Vec::with_capacity(num_partitions as usize);

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
        let result =
            execute_tier2_twokey_sum(&keys, &vals, 0).expect("empty input must succeed");
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
}
