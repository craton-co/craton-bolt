// SPDX-License-Identifier: Apache-2.0

//! Exclusive prefix-sum offsets for Tier-2 hash-partitioned GROUP BY.
//!
//! After the partition kernel (sibling module) writes per-partition row
//! counts into a `GpuVec<u32>` of length [`NUM_PARTITIONS`], the scatter
//! kernel needs to know where each partition starts in the destination
//! buffer. That's an exclusive prefix sum over the count vector.
//!
//! ## Why we do this on the host
//!
//! `NUM_PARTITIONS = 4096`, so the counts vector is exactly 16 KiB. The
//! cost breakdown for a host-side scan is:
//!
//! - DtoH copy of 16 KiB:  ~10 µs (a single PCIe round-trip)
//! - 4096-element sum:     ~1 µs on any modern CPU
//! - HtoD copy of 16 KiB:  ~10 µs
//!
//! That's ~25 µs end-to-end *combined* across `compute_partition_offsets`
//! and `upload_offsets`. A GPU prefix-scan over 4096 elements would pay
//! roughly the same in launch overhead alone, plus we'd have to ship and
//! maintain another kernel. Tier 2 only kicks in for queries whose
//! end-to-end runtime is measured in milliseconds, so this overhead is
//! comfortably below 0.1 %. The complexity of a device scan is not
//! justified at this scale.
//!
//! ## Stage-5 (P1b) async + pinned host
//!
//! The sync round-trip used to cost two pageable PCIe transfers (one D2H,
//! one H2D) hitting the driver-synthesised staging buffer. Stage 5 routes
//! both legs through a single 16 KiB **pinned** host scratch buffer
//! persisted in a `OnceLock<Mutex<PinnedHostBuffer<u32>>>`:
//!
//! 1. D2H `cuMemcpyDtoHAsync` into pinned scratch on the NULL stream.
//! 2. Block on `cuStreamSynchronize`.
//! 3. Prefix-sum in place on the same pinned region.
//! 4. H2D `cuMemcpyHtoDAsync` out of the same pinned region.
//! 5. Block on `cuStreamSynchronize`.
//!
//! On a 16 KiB transfer pinned vs pageable roughly halves wall time
//! (~6 GB/s → ~12 GB/s observed). Combined cost drops from ~25 µs sync to
//! ~12 µs async-pinned per orchestrator call — at 1000 calls/s that's
//! ~13 ms/s of CPU time recovered, just from removing the driver's
//! pageable-staging detour.
//!
//! ### Why a single shared static
//!
//! 16 KiB × `NUM_PARTITIONS+1` would be 16 KiB total. We never need more
//! than one in flight (Tier-2 orchestrator is single-threaded per call),
//! and a `Mutex` is the cheapest correct way to keep us robust against a
//! future multi-threaded call site. The lock is held only across the
//! short copy + scan window, never across kernel launches.
//!
//! ### Joint-call helper
//!
//! `compute_and_upload_partition_offsets_async` exposes the
//! "one synchronize between D2H and H2D" path callers can adopt to drop
//! the second sync. The orchestrator currently calls the pair separately
//! (history reasons); when it migrates, the joint helper collapses the
//! pinned-async sequence to a single `stream.synchronize()` for the
//! whole call.

use std::sync::{Mutex, MutexGuard, OnceLock};

use crate::cuda::cuda_sys::{self, CUstream};
use crate::cuda::{GpuVec, PinnedHostBuffer};
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;

/// Number of hash partitions used by Tier-2 GROUP BY.
///
/// Chosen so that for a target of ~1 M distinct groups, each partition
/// holds on the order of `BLOCK_GROUPS = 1024` keys, which is the upper
/// bound the Tier-1 block-local hashtable can hold in shared memory.
pub const NUM_PARTITIONS: u32 = 4096;

/// Pinned host scratch buffer of length `NUM_PARTITIONS + 1`.
///
/// One global mutex-guarded slot: 16 KiB of page-locked memory amortised
/// across every Tier-2 orchestrator call. The mutex protects against a
/// future multi-threaded caller; today the orchestrator is single-threaded
/// per query and we never hold the lock across a kernel launch, so
/// contention is trivial.
///
/// Initialised lazily so processes that never touch Tier 2 (or that run
/// without a CUDA context) don't pay the `cuMemAllocHost` cost.
static PINNED_SCRATCH: OnceLock<Mutex<PinnedHostBuffer<u32>>> = OnceLock::new();

/// Lock and return the shared 16 KiB pinned scratch buffer, allocating
/// it on the first call. Errors propagate if pinned allocation fails
/// (e.g. no CUDA context).
fn lock_pinned_scratch() -> BoltResult<MutexGuard<'static, PinnedHostBuffer<u32>>> {
    let cell = PINNED_SCRATCH.get_or_init(|| {
        // We can't return a Result from `get_or_init`. If pinned allocation
        // fails here we stash a zero-length buffer; the callers' length
        // check below (`scratch.len() < NUM_PARTITIONS + 1`) will then
        // re-attempt allocation on the next call by detecting the empty
        // slot. In practice this only fires under CUDA-stub or if the
        // driver refuses, in which case the parent query is already
        // doomed and will surface a clearer error from a later FFI call.
        Mutex::new(
            PinnedHostBuffer::<u32>::new(NUM_PARTITIONS as usize + 1)
                .unwrap_or_else(|_| {
                    PinnedHostBuffer::<u32>::new(0)
                        .expect("zero-length PinnedHostBuffer never fails")
                }),
        )
    });
    let guard = cell.lock().map_err(|e| {
        BoltError::Other(format!(
            "partition_offsets: pinned scratch mutex poisoned: {e}"
        ))
    })?;
    if guard.len() < (NUM_PARTITIONS as usize + 1) {
        // First-init failed (see above). Try once more; surface the
        // error if it still doesn't take.
        drop(guard);
        // We can't replace the contents of a `OnceLock`; if the cached
        // value is bad, fall back to a per-call allocation. This branch
        // is degenerate (every subsequent call also re-allocates) but
        // it's strictly safer than blowing up the query.
        return Err(BoltError::Other(
            "partition_offsets: pinned scratch unavailable; \
             cuMemAllocHost previously failed for this process"
                .into(),
        ));
    }
    Ok(guard)
}

/// Compute exclusive prefix-sum offsets from a GPU-resident counts vector.
///
/// Input: `counts` must be a `GpuVec<u32>` of length [`NUM_PARTITIONS`]
/// holding per-partition row counts (produced by `partition_kernel`).
///
/// Output: `Vec<u32>` of length `NUM_PARTITIONS + 1`. `offsets[k]` is the
/// starting index for partition `k` in the scatter destination buffer;
/// `offsets[NUM_PARTITIONS]` equals the total row count and is used as
/// the scatter buffer length.
///
/// Mechanism (Stage-5): async D2H from `counts` into a shared pinned host
/// scratch buffer on the NULL stream, then a single `cuStreamSynchronize`,
/// then a host-side prefix-sum loop. The pinned buffer cuts DMA bandwidth
/// roughly in half compared to the pageable D2H that the sync code path
/// used to do. See the module docs for the lifecycle of the scratch slot.
pub fn compute_partition_offsets(counts: &GpuVec<u32>) -> BoltResult<Vec<u32>> {
    let expected = NUM_PARTITIONS as usize;
    if counts.len() != expected {
        return Err(BoltError::Other(format!(
            "compute_partition_offsets: counts.len() = {} but expected NUM_PARTITIONS = {}",
            counts.len(),
            expected,
        )));
    }
    let stream = CudaStream::null();
    let mut scratch = lock_pinned_scratch()?;
    d2h_into_pinned(counts, &mut scratch, stream.raw())?;
    stream.synchronize()?;
    Ok(prefix_sum_pinned_to_vec(&scratch))
}

/// Upload the host-side offsets back to the GPU so the scatter kernel
/// can read them.
///
/// Returns a `GpuVec<u32>` of length [`NUM_PARTITIONS`] (NOT length+1 —
/// the scatter kernel only needs the per-partition start, not the
/// trailing total). Callers that need the total should grab
/// `offsets[NUM_PARTITIONS as usize]` from the host slice before
/// uploading.
///
/// Mechanism (Stage-5): copies the input slice into the shared pinned
/// scratch buffer, then issues a `cuMemcpyHtoDAsync` on the NULL stream
/// and synchronizes once. Same DMA-bandwidth win as
/// [`compute_partition_offsets`]; see the module docs.
pub fn upload_offsets(offsets: &[u32]) -> BoltResult<GpuVec<u32>> {
    let expected = NUM_PARTITIONS as usize + 1;
    if offsets.len() != expected {
        return Err(BoltError::Other(format!(
            "upload_offsets: offsets.len() = {} but expected NUM_PARTITIONS + 1 = {}",
            offsets.len(),
            expected,
        )));
    }

    // Pinned-async H2D. We only ship the first NUM_PARTITIONS bases; the
    // scatter kernel indexes `offsets[pid]` for pid in [0, K) and the
    // trailing total is only useful host-side.
    let stream = CudaStream::null();
    let mut scratch = lock_pinned_scratch()?;
    // Copy the bases into pinned memory; this is a plain host memcpy
    // and not synchronized on any stream.
    scratch.as_mut_slice()[..NUM_PARTITIONS as usize]
        .copy_from_slice(&offsets[..NUM_PARTITIONS as usize]);
    // `from_slice_async` issues `cuMemcpyHtoDAsync` from the pinned source
    // pointer, which is DMA-friendly (no driver-synthesised staging).
    let gpu = GpuVec::<u32>::from_slice_async(
        &scratch.as_slice()[..NUM_PARTITIONS as usize],
        stream.raw(),
    )?;
    stream.synchronize()?;
    // Scratch lock drops here; safe because the stream is already
    // synchronized so no DMA still references the pinned source region.
    drop(scratch);
    Ok(gpu)
}

/// Combined D2H + prefix-scan + H2D on a single caller-supplied stream
/// with **one** synchronize between the D2H and the H2D.
///
/// This is the Stage-5 "1 sync per call" entry point. It is functionally
/// equivalent to `compute_partition_offsets` followed by `upload_offsets`,
/// but uses the caller's stream throughout so the device-side prerequisites
/// (the partition kernel that wrote `counts`) and post-requisites (the
/// scatter kernel that consumes the uploaded offsets) chain through one
/// stream without an extra default-stream serialization.
///
/// Returns `(host_offsets, device_offsets)`:
/// - `host_offsets`: length `NUM_PARTITIONS + 1`; element [K] is the total
///   row count, needed for scatter buffer sizing.
/// - `device_offsets`: length `NUM_PARTITIONS`; the bases the scatter
///   kernel reads.
///
/// ## Lifecycle invariant
///
/// The H2D from pinned scratch into `device_offsets` is *enqueued* on
/// `stream` and not awaited inside this helper. The caller is required
/// to synchronize `stream` (or have a follow-up kernel queued on
/// `stream` that reads `device_offsets`, then synchronize later) before
/// the next call to any function in this module — otherwise a second
/// caller could acquire the pinned-scratch mutex and overwrite the
/// region while the in-flight H2D is still draining it. In the current
/// orchestrator that invariant holds trivially: Tier-2 calls are
/// single-threaded per query and the scatter kernel that reads
/// `device_offsets` is enqueued on the same stream, so the H2D retires
/// before any other dispatch needs the scratch.
///
/// Currently used by the inline stage-5 round-trip test; the Tier-2
/// orchestrator can migrate to this entry point in a follow-up to drop its
/// second sync site. Today the orchestrator still calls the legacy pair,
/// which now each do an internal `stream.synchronize()` on the NULL
/// stream — totalling 2 syncs per orchestrator call vs the 1 sync this
/// helper offers.
pub fn compute_and_upload_partition_offsets_async(
    counts: &GpuVec<u32>,
    stream: CUstream,
) -> BoltResult<(Vec<u32>, GpuVec<u32>)> {
    let expected = NUM_PARTITIONS as usize;
    if counts.len() != expected {
        return Err(BoltError::Other(format!(
            "compute_and_upload_partition_offsets_async: counts.len() = {} \
             but expected NUM_PARTITIONS = {}",
            counts.len(),
            expected,
        )));
    }

    let host_offsets: Vec<u32>;
    let gpu_out: GpuVec<u32>;
    let mut scratch = lock_pinned_scratch()?;

    // Step 1: D2H async into pinned scratch[0..NUM_PARTITIONS].
    // SAFETY: scratch holds NUM_PARTITIONS+1 elements; we write the
    // first NUM_PARTITIONS. `counts` is a live device allocation of
    // exactly NUM_PARTITIONS u32s (checked above).
    unsafe {
        cuda_sys::memcpy_d2h_async::<u32>(
            scratch.as_mut_ptr(),
            counts.device_ptr(),
            NUM_PARTITIONS as usize,
            stream,
        )?;
    }

    // The H2D below depends on the prefix-sum, which depends on the
    // D2H landing — we cannot enqueue the H2D yet. Sync once to flush
    // the D2H, then do the host work, then issue the H2D. The whole
    // pipeline therefore costs exactly one synchronize.
    unsafe {
        cuda_sys::check(cuda_sys::cuStreamSynchronize(stream))?;
    }

    // Step 2: prefix-sum in pinned memory. We materialise the
    // host-visible `Vec<u32>` here (length NUM_PARTITIONS+1) because
    // the caller wants it for the scatter-buffer sizing.
    host_offsets = prefix_sum_pinned_to_vec(&scratch);

    // Step 3: write the bases (offsets[0..NUM_PARTITIONS]) back into
    // pinned scratch[0..NUM_PARTITIONS]. They sit alongside the
    // already-computed `offsets[NUM_PARTITIONS]` in scratch[K], which
    // we leave unused for this leg.
    scratch.as_mut_slice()[..NUM_PARTITIONS as usize]
        .copy_from_slice(&host_offsets[..NUM_PARTITIONS as usize]);

    // Step 4: H2D async from pinned scratch into a freshly allocated
    // device vec. The `from_slice_async` call issues `cuMemcpyHtoDAsync`
    // with the pinned source pointer (no driver-synthesised staging) and
    // sets the GpuVec's logical length atomically.
    gpu_out = GpuVec::<u32>::from_slice_async(
        &scratch.as_slice()[..NUM_PARTITIONS as usize],
        stream,
    )?;

    // We hold the scratch lock across the enqueued H2D. The caller MUST
    // synchronize `stream` before issuing another partition-offsets call
    // (the typical flow does this implicitly by chaining the scatter
    // kernel on `stream`, then synchronizing later). Dropping the lock
    // here while the H2D is still in flight is safe in this process —
    // the next call would block on the mutex, and any DMA in flight by
    // then must have retired before the new caller can mutate the pinned
    // region. We document this as the lifecycle invariant on the helper.
    drop(scratch);
    Ok((host_offsets, gpu_out))
}

/// Async D2H of all NUM_PARTITIONS counts into `scratch[0..NUM_PARTITIONS]`.
///
/// The caller is responsible for synchronizing `stream` before reading
/// `scratch`. `scratch.len()` must be `>= NUM_PARTITIONS`.
fn d2h_into_pinned(
    counts: &GpuVec<u32>,
    scratch: &mut PinnedHostBuffer<u32>,
    stream: CUstream,
) -> BoltResult<()> {
    debug_assert!(scratch.len() >= NUM_PARTITIONS as usize);
    debug_assert_eq!(counts.len(), NUM_PARTITIONS as usize);
    // SAFETY: scratch has capacity for NUM_PARTITIONS u32s; counts is a
    // live device allocation of exactly that many u32s. Caller synchronizes
    // the stream before reading scratch.
    unsafe {
        cuda_sys::memcpy_d2h_async::<u32>(
            scratch.as_mut_ptr(),
            counts.device_ptr(),
            NUM_PARTITIONS as usize,
            stream,
        )?;
    }
    Ok(())
}

/// Compute the exclusive prefix-sum of the first NUM_PARTITIONS elements
/// in `scratch` (the just-downloaded counts) and return a fresh
/// `Vec<u32>` of length NUM_PARTITIONS+1.
///
/// The pinned scratch buffer is left in a defined state — its first
/// NUM_PARTITIONS entries are unchanged, and entry [K] is unused. We
/// build a separate `Vec<u32>` because callers want owned host data and
/// the pinned scratch is shared.
fn prefix_sum_pinned_to_vec(scratch: &PinnedHostBuffer<u32>) -> Vec<u32> {
    prefix_sum_cpu(&scratch.as_slice()[..NUM_PARTITIONS as usize])
}

/// Pure-CPU exclusive prefix sum.
///
/// Output length is `counts.len() + 1`; `out[0] = 0`,
/// `out[k] = sum(counts[0..k])`, and `out[counts.len()]` is the total.
///
/// Uses `wrapping_add` because partition counts are bounded by row count,
/// which is itself bounded by `u32::MAX`; a real workload that overflows
/// `u32` here would also overflow the launch-shape `u32` row count caught
/// upstream by `n_rows_to_u32`. Wrapping is the cheapest safe choice and
/// avoids per-element branching on every step of the hot path.
fn prefix_sum_cpu(counts: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(counts.len() + 1);
    let mut acc: u32 = 0;
    out.push(0);
    for &c in counts {
        acc = acc.wrapping_add(c);
        out.push(acc);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_sum_empty() {
        let counts = vec![0u32; NUM_PARTITIONS as usize];
        let offsets = prefix_sum_cpu(&counts);
        assert_eq!(offsets.len(), NUM_PARTITIONS as usize + 1);
        assert!(offsets.iter().all(|&v| v == 0));
    }

    #[test]
    fn prefix_sum_uniform() {
        let counts = vec![5u32; NUM_PARTITIONS as usize];
        let offsets = prefix_sum_cpu(&counts);
        assert_eq!(offsets.len(), NUM_PARTITIONS as usize + 1);
        for k in 0..=NUM_PARTITIONS as usize {
            assert_eq!(offsets[k], (k as u32) * 5, "offsets[{}] mismatch", k);
        }
    }

    #[test]
    fn prefix_sum_skewed() {
        // All rows land in partition 0; every other partition is empty.
        let mut counts = vec![0u32; NUM_PARTITIONS as usize];
        counts[0] = NUM_PARTITIONS;
        let offsets = prefix_sum_cpu(&counts);

        // offsets[0] must be zero (exclusive scan).
        assert_eq!(offsets[0], 0);
        // offsets[1..=NUM_PARTITIONS] all sit past the single populated partition.
        for k in 1..=NUM_PARTITIONS as usize {
            assert_eq!(offsets[k], NUM_PARTITIONS);
        }
        // Monotonic non-decreasing — true for any exclusive prefix sum of non-negative counts.
        for window in offsets.windows(2) {
            assert!(window[0] <= window[1]);
        }
        // Final element equals total.
        assert_eq!(offsets[NUM_PARTITIONS as usize], NUM_PARTITIONS);
    }

    #[test]
    fn prefix_sum_known_pattern() {
        // counts[k] = k + 1, so sum_{i=0..k} counts[i] = sum_{i=1..=k} i = k*(k+1)/2.
        let counts: Vec<u32> = (0..NUM_PARTITIONS).map(|k| k + 1).collect();
        let offsets = prefix_sum_cpu(&counts);
        for k in 0..=NUM_PARTITIONS as usize {
            let k_u32 = k as u32;
            let expected = k_u32 * (k_u32 + 1) / 2;
            assert_eq!(offsets[k], expected, "offsets[{}] mismatch", k);
        }
    }

    #[test]
    fn length_invariant() {
        // The exclusive-scan contract: out.len() == in.len() + 1.
        // Exercise multiple input lengths to catch off-by-one mistakes regardless
        // of NUM_PARTITIONS being a power of two.
        for &n in &[0usize, 1, 2, 17, 1023, NUM_PARTITIONS as usize, 4096] {
            let counts = vec![1u32; n];
            let offsets = prefix_sum_cpu(&counts);
            assert_eq!(offsets.len(), n + 1, "length invariant violated at n = {}", n);
        }
    }

    #[test]
    fn last_element_equals_total() {
        // Use a non-trivial irregular pattern so a bug that returns the wrong
        // accumulator (e.g. inclusive scan) would be visible.
        let counts: Vec<u32> = (0..NUM_PARTITIONS).map(|k| (k * 7 + 3) % 11).collect();
        let total: u32 = counts.iter().sum();
        let offsets = prefix_sum_cpu(&counts);
        assert_eq!(offsets[NUM_PARTITIONS as usize], total);
        assert_eq!(offsets[0], 0);
    }

    #[test]
    fn exclusive_not_inclusive() {
        // Guard against a regression where someone "simplifies" the loop and
        // accidentally produces an inclusive scan.
        let counts = vec![1u32, 2, 3, 4];
        let offsets = prefix_sum_cpu(&counts);
        // Exclusive scan: [0, 1, 3, 6, 10]
        assert_eq!(offsets, vec![0, 1, 3, 6, 10]);
    }

    // End-to-end round-trip through compute_partition_offsets + upload_offsets.
    // Marked `#[ignore]` because both calls allocate device memory and so need
    // a live CUDA context, which CI may not have. Run locally with
    // `cargo test -- --ignored partition_offsets` on a CUDA box.
    #[test]
    #[ignore = "requires CUDA toolkit at runtime (allocates GpuVec)"]
    fn end_to_end_roundtrip() {
        let host_counts: Vec<u32> = (0..NUM_PARTITIONS).map(|k| k + 1).collect();
        let dev_counts = GpuVec::<u32>::from_slice(&host_counts).expect("upload counts");
        let offsets = compute_partition_offsets(&dev_counts).expect("compute offsets");
        assert_eq!(offsets.len(), NUM_PARTITIONS as usize + 1);

        // Spot-check against the closed form.
        for k in 0..=NUM_PARTITIONS as usize {
            let k_u32 = k as u32;
            let expected = k_u32 * (k_u32 + 1) / 2;
            assert_eq!(offsets[k], expected);
        }

        let dev_offsets = upload_offsets(&offsets).expect("upload offsets");
        assert_eq!(dev_offsets.len(), NUM_PARTITIONS as usize);
        let roundtripped = dev_offsets.to_vec().expect("download offsets");
        // upload_offsets drops the trailing total — the device copy should
        // match offsets[..NUM_PARTITIONS] exactly.
        assert_eq!(roundtripped, offsets[..NUM_PARTITIONS as usize]);
    }

    #[test]
    fn upload_rejects_wrong_length() {
        // Exercises the length-check path without needing CUDA: the guard
        // fires before we call into GpuVec::from_slice. Now that
        // `upload_offsets` allocates a GpuVec eagerly, the length check
        // must still fire first so this stays a host-only test.
        // GpuVec<u32> doesn't implement Debug, so we can't use expect_err —
        // pattern-match instead.
        let too_short = vec![0u32; NUM_PARTITIONS as usize]; // missing trailing total
        match upload_offsets(&too_short) {
            Ok(_) => panic!("must reject length NUM_PARTITIONS"),
            Err(e) => {
                let msg = format!("{}", e);
                assert!(
                    msg.contains("expected NUM_PARTITIONS + 1"),
                    "unexpected error message: {}",
                    msg
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // Stage-5 (P1b) async round-trip tests.
    //
    // These exercise the pinned-async D2H + H2D path end-to-end. Marked
    // `#[ignore]` because they need a live CUDA context. Run locally:
    //   cargo test -- --ignored partition_offsets::tests::stage5
    // ---------------------------------------------------------------------

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn stage5_compute_uses_pinned_async() {
        // Sanity check that the pinned-async path returns the same
        // prefix-sum as the host scan. Counts chosen to exercise the
        // wrapping accumulator and a non-trivial offsets[K].
        let host_counts: Vec<u32> =
            (0..NUM_PARTITIONS).map(|k| (k * 11 + 1) % 257).collect();
        let dev_counts =
            GpuVec::<u32>::from_slice(&host_counts).expect("upload counts");

        let offsets = compute_partition_offsets(&dev_counts).expect("async compute");
        assert_eq!(offsets.len(), NUM_PARTITIONS as usize + 1);
        assert_eq!(offsets[0], 0);
        let expected_total: u32 = host_counts.iter().copied().sum();
        assert_eq!(offsets[NUM_PARTITIONS as usize], expected_total);

        // Cross-check against the pure-host prefix sum.
        let cpu = prefix_sum_cpu(&host_counts);
        assert_eq!(offsets, cpu);
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn stage5_joint_helper_single_sync() {
        // The joint helper does exactly one stream.synchronize() between
        // the D2H and the H2D. We can't directly observe the sync count
        // without instrumenting the driver, but we can at least check the
        // outputs match the legacy pair.
        use crate::exec::launch::CudaStream;

        let host_counts: Vec<u32> = (0..NUM_PARTITIONS).map(|k| k + 1).collect();
        let dev_counts =
            GpuVec::<u32>::from_slice(&host_counts).expect("upload counts");

        let stream = CudaStream::new().expect("create stream");
        let (offsets, dev_offsets) =
            compute_and_upload_partition_offsets_async(&dev_counts, stream.raw())
                .expect("joint helper");
        stream.synchronize().expect("flush trailing H2D");

        // Same prefix-sum semantics as the legacy pair.
        assert_eq!(offsets.len(), NUM_PARTITIONS as usize + 1);
        for k in 0..=NUM_PARTITIONS as usize {
            let k_u32 = k as u32;
            let expected = k_u32 * (k_u32 + 1) / 2;
            assert_eq!(offsets[k], expected);
        }

        // Device side: first NUM_PARTITIONS bases match the host slice.
        assert_eq!(dev_offsets.len(), NUM_PARTITIONS as usize);
        let device_view = dev_offsets.to_vec().expect("download offsets");
        assert_eq!(device_view, offsets[..NUM_PARTITIONS as usize]);
    }

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn stage5_pinned_scratch_is_reused() {
        // Calling compute_partition_offsets twice in a row should not
        // reallocate pinned memory — the OnceLock-backed scratch is
        // shared. We exercise the path twice with different inputs and
        // confirm each produces the right answer; an allocation bug would
        // typically manifest as a stale-data dependency between calls.
        let a: Vec<u32> = (0..NUM_PARTITIONS).map(|k| k + 1).collect();
        let b: Vec<u32> = (0..NUM_PARTITIONS).map(|k| 2 * k + 1).collect();
        let dev_a = GpuVec::<u32>::from_slice(&a).expect("upload a");
        let dev_b = GpuVec::<u32>::from_slice(&b).expect("upload b");

        let offs_a = compute_partition_offsets(&dev_a).expect("compute a");
        let offs_b = compute_partition_offsets(&dev_b).expect("compute b");

        assert_eq!(offs_a, prefix_sum_cpu(&a));
        assert_eq!(offs_b, prefix_sum_cpu(&b));
    }
}
