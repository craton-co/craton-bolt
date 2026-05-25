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
//! `NUM_PARTITIONS = 1024`, so the counts vector is exactly 4 KiB. The
//! cost breakdown for a host-side scan is:
//!
//! - DtoH copy of 4 KiB:   ~10 µs (a single PCIe round-trip)
//! - 1024-element sum:     ~1 µs on any modern CPU
//! - HtoD copy of 4 KiB:   ~10 µs
//!
//! That's ~25 µs end-to-end. A GPU prefix-scan over 1024 elements would
//! pay roughly the same in launch overhead alone, plus we'd have to ship
//! and maintain another kernel. Tier 2 only kicks in for queries whose
//! end-to-end runtime is measured in milliseconds, so this overhead is
//! comfortably below 0.1 %. The complexity of a device scan is not
//! justified at this scale.
//!
//! If the partition count ever grows past ~16 K we should revisit, but
//! 1024 is the right choice for q5-class workloads (~1 M groups, ~1 K
//! groups per partition) and there's no plausible path to making it
//! larger without also blowing up the per-partition hashtable budget.

use crate::cuda::GpuVec;
use crate::error::PatinaResult;

/// Number of hash partitions used by Tier-2 GROUP BY.
///
/// Chosen so that for a target of ~1 M distinct groups, each partition
/// holds on the order of `BLOCK_GROUPS = 1024` keys, which is the upper
/// bound the Tier-1 block-local hashtable can hold in shared memory.
pub const NUM_PARTITIONS: u32 = 4096;

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
/// Mechanism: downloads the 1024 `u32`s (4 KiB) over PCIe, prefix-sums
/// on the host with one tight loop, returns. See the module docs for
/// the cost rationale.
pub fn compute_partition_offsets(counts: &GpuVec<u32>) -> PatinaResult<Vec<u32>> {
    let expected = NUM_PARTITIONS as usize;
    if counts.len() != expected {
        return Err(crate::error::PatinaError::Other(format!(
            "compute_partition_offsets: counts.len() = {} but expected NUM_PARTITIONS = {}",
            counts.len(),
            expected,
        )));
    }
    let host = counts.to_vec()?;
    Ok(prefix_sum_cpu(&host))
}

/// Upload the host-side offsets back to the GPU so the scatter kernel
/// can read them.
///
/// Returns a `GpuVec<u32>` of length [`NUM_PARTITIONS`] (NOT length+1 —
/// the scatter kernel only needs the per-partition start, not the
/// trailing total). Callers that need the total should grab
/// `offsets[NUM_PARTITIONS as usize]` from the host slice before
/// uploading.
pub fn upload_offsets(offsets: &[u32]) -> PatinaResult<GpuVec<u32>> {
    let expected = NUM_PARTITIONS as usize + 1;
    if offsets.len() != expected {
        return Err(crate::error::PatinaError::Other(format!(
            "upload_offsets: offsets.len() = {} but expected NUM_PARTITIONS + 1 = {}",
            offsets.len(),
            expected,
        )));
    }
    // Drop the trailing total; the scatter kernel indexes only [0, NUM_PARTITIONS).
    GpuVec::<u32>::from_slice(&offsets[..NUM_PARTITIONS as usize])
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
        // fires before we call into GpuVec::from_slice.
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
}
