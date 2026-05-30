// SPDX-License-Identifier: Apache-2.0

//! dedup (tier2/shmem): single home for the genuinely-identical host-side
//! eligibility-gate helpers shared by the Tier-1 (shmem) and Tier-2
//! GROUP BY executors.
//!
//! ## Scope — extraction-only, behaviour-preserving
//!
//! The 20+ `try_execute` variants under `groupby_tier2_*` / `groupby_shmem_*`
//! share *some* boilerplate, but most of it is adapted per variant (kernel
//! ABIs, dtype-specific atomics, accumulator fan-out, the spill-counter
//! error string — which is a cross-module sentinel matched by
//! `groupby.rs`'s GB-S2 soft-fallback contract and exported as
//! `groupby_tier2_orchestrator::PARTITION_REDUCE_SPILL_PREFIX`). Those are
//! deliberately left local to each executor; consolidating them behind
//! flags would be unsafe without GPU-runtime verification.
//!
//! What *is* safe to share is the pure host-side key-range scan: a loop
//! that finds `max(key)` while rejecting negative keys. It appears
//! byte-identically in every single-key executor and touches no launch
//! parameters or dispatch decisions. Only that loop lives here; each call
//! site keeps its own divergent handling of the empty-input case
//! (`None` vs an empty-schema result batch) and its own `n_groups`
//! derivation.

/// Scan a single-Int32-key column for its maximum value while rejecting any
/// negative key.
///
/// dedup (tier2/shmem): replaces the byte-identical
///
/// ```ignore
/// let mut max_key: i32 = -1;
/// for &k in key_arr.values() {
///     if k < 0 { return None; }       // negative keys never hash to a slot
///     if k > max_key { max_key = k; }
/// }
/// ```
///
/// loop that every single-key Tier-1 / Tier-2 executor ran inline before
/// computing its `n_groups` estimate.
///
/// Returns:
/// * `None` — at least one key is negative. Negative keys never hash to a
///   valid dense slot, so the caller must decline the fast path (the
///   pre-extraction code did `return None;` from the loop on the first
///   negative key).
/// * `Some(-1)` — the input is empty (no keys). The caller decides what an
///   empty input means: most executors decline (`return None;`), while the
///   shmem-SUM family emit an empty-schema result. The historical sentinel
///   value was exactly `max_key == -1` after the loop, so this preserves
///   each site's existing `if max_key < 0 { … }` branch verbatim.
/// * `Some(max)` with `max >= 0` — the largest key seen. Callers derive
///   their `n_groups` estimate from this exactly as before (e.g.
///   `(max as u32).saturating_add(1)`), keeping the per-variant arithmetic
///   local.
///
/// Pure host-side computation: no GPU calls, no I/O, and crucially no
/// launch-parameter or dispatch input is produced here — so this extraction
/// cannot change any execution behaviour.
#[inline]
pub(crate) fn scan_max_nonneg_key(keys: &[i32]) -> Option<i32> {
    let mut max_key: i32 = -1;
    for &k in keys {
        if k < 0 {
            return None;
        }
        if k > max_key {
            max_key = k;
        }
    }
    Some(max_key)
}

use std::sync::Arc;

use arrow_schema::Schema as ArrowSchema;

use crate::error::BoltResult;
use crate::plan::logical_plan::Schema;

/// dedup (tier2): single home for the per-file `plan_schema_to_arrow_schema`
/// wrappers that every Tier-2 executor / merger carried verbatim.
///
/// Each `groupby_tier2_*` file previously declared a private, byte-identical
///
/// ```ignore
/// fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
///     crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal(
///         s, "this aggregate output path")
/// }
/// ```
///
/// that did nothing but forward to the shared
/// [`crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal`]
/// with the same `ctx` string. Those wrappers are deleted in favour of this
/// single delegation. The forwarded `ctx` ("this aggregate output path") and
/// the underlying conversion are preserved byte-for-byte, so the produced
/// Arrow schema and any error text are identical to the pre-extraction code.
#[inline]
pub(crate) fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal(
        s,
        "this aggregate output path",
    )
}

/// dedup (tier2): the single-key partition-kernel selection predicate that
/// every single-Int32-key Tier-2 executor / orchestrator carried verbatim
/// inside its private `partition_spec_for`.
///
/// Each `partition_spec_for` ran the byte-identical comparison
///
/// ```ignore
/// if n_rows < partition_kernel::SHMEM_STAGING_MIN_ROWS {
///     KernelSpec::Partition
/// } else {
///     KernelSpec::PartitionShmemStaging
/// }
/// ```
///
/// The branch *value* is the per-file `KernelSpec` enum (which differs by
/// executor and carries the ABI-bearing reduce variant), so the function
/// itself stays local; only the threshold test is shared here. Returns
/// `true` when the shmem-staging partition kernel should be selected (i.e.
/// `n_rows >= SHMEM_STAGING_MIN_ROWS`), `false` for the plain global-atomics
/// partition kernel. References the exact same
/// [`crate::jit::partition_kernel::SHMEM_STAGING_MIN_ROWS`] constant as the
/// inlined comparisons did, so the selection is identical for every `n_rows`.
#[inline]
pub(crate) fn use_shmem_staging_partition(n_rows: u32) -> bool {
    n_rows >= crate::jit::partition_kernel::SHMEM_STAGING_MIN_ROWS
}

/// dedup (tier2): the two-key (packed-i64 key) analogue of
/// [`use_shmem_staging_partition`].
///
/// Every two-key Tier-2 executor / orchestrator carried a byte-identical
/// private `partition_i64_spec_for` whose only varying piece is the per-file
/// `KernelSpec` enum it returns; the threshold comparison
/// `n_rows < partition_kernel_i64::SHMEM_STAGING_MIN_ROWS` was the same in
/// every copy. This shares only that comparison (against the **i64**
/// partition kernel's own `SHMEM_STAGING_MIN_ROWS`, distinct from the
/// single-key constant used by [`use_shmem_staging_partition`]). Returns
/// `true` when the i64 shmem-staging partition kernel should be selected.
#[inline]
pub(crate) fn use_shmem_staging_partition_i64(n_rows: u32) -> bool {
    n_rows >= crate::jit::partition_kernel_i64::SHMEM_STAGING_MIN_ROWS
}

/// dedup (tier2): the host slot-walk that collects populated `(key, value)`
/// pairs out of the fixed-size `NUM_PARTITIONS * BLOCK_GROUPS` reduce output
/// and sorts them by key.
///
/// Every single-key reduce executor downloaded three parallel host buffers —
/// `host_keys`, `host_vals` and a `host_set` present-map — and then ran the
/// byte-identical walk
///
/// ```ignore
/// let mut pairs: Vec<(i32, T)> = Vec::new();
/// for pid in 0..num_partitions {
///     let base = pid * block_groups;
///     for slot in 0..block_groups {
///         let idx = base + slot;
///         if host_set[idx] != 0 {
///             pairs.push((host_keys[idx], host_vals[idx]));
///         }
///     }
/// }
/// pairs.sort_by_key(|(k, _)| *k);
/// ```
///
/// (the COUNT executor wrote the equivalent `if host_set[idx] == 0 { continue }`
/// guard, which selects exactly the same slots). The per-executor value type
/// is the type parameter `T` (`i32` / `i64` / `u64` / `f64`); callers that need
/// a *different* output type than the buffer type (e.g. COUNT downloads `u64`
/// but emits `i64`) keep their post-walk cast at the call site, so this helper
/// is a pure, behaviour-identical extraction of the selection + key-sort.
///
/// Because each distinct key hashes to exactly one partition and is
/// deduplicated within that partition's slot table, the produced keys are
/// unique; the `sort_by_key` therefore yields the same order regardless of
/// stability, matching the pre-extraction code exactly.
///
/// Pure host computation — no GPU calls, no launch parameters produced.
#[inline]
pub(crate) fn collect_populated_slots_sorted<T: Copy>(
    host_keys: &[i32],
    host_vals: &[T],
    host_set: &[u8],
    num_partitions: usize,
    block_groups: usize,
) -> Vec<(i32, T)> {
    let mut pairs: Vec<(i32, T)> = Vec::new();
    for pid in 0..num_partitions {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_set[idx] != 0 {
                pairs.push((host_keys[idx], host_vals[idx]));
            }
        }
    }
    pairs.sort_by_key(|(k, _)| *k);
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_sentinel_minus_one() {
        // Mirrors the pre-extraction `max_key == -1` empty-input sentinel.
        assert_eq!(scan_max_nonneg_key(&[]), Some(-1));
    }

    #[test]
    fn negative_key_declines() {
        assert_eq!(scan_max_nonneg_key(&[0, 1, -1, 2]), None);
        // First-element negative also declines.
        assert_eq!(scan_max_nonneg_key(&[-5, 3]), None);
    }

    #[test]
    fn finds_max_of_nonneg_keys() {
        assert_eq!(scan_max_nonneg_key(&[0, 7, 3, 7, 1]), Some(7));
        assert_eq!(scan_max_nonneg_key(&[0]), Some(0));
        assert_eq!(scan_max_nonneg_key(&[i32::MAX, 0]), Some(i32::MAX));
    }

    #[test]
    fn matches_inline_loop_semantics() {
        // Differential check against the exact inline loop the executors
        // used to run, over a mix of shapes including the empty + negative
        // edge cases.
        let cases: &[&[i32]] = &[
            &[],
            &[0],
            &[5, 5, 5],
            &[0, 1, 2, 3, 1024, 1025],
            &[3, -1, 9],
            &[-2],
        ];
        for case in cases {
            let mut max_key: i32 = -1;
            let mut declined = false;
            for &k in case.iter() {
                if k < 0 {
                    declined = true;
                    break;
                }
                if k > max_key {
                    max_key = k;
                }
            }
            let expected = if declined { None } else { Some(max_key) };
            assert_eq!(scan_max_nonneg_key(case), expected, "case={case:?}");
        }
    }

    #[test]
    fn shmem_staging_predicate_matches_threshold() {
        // The predicate must agree exactly with the inlined
        // `n_rows < SHMEM_STAGING_MIN_ROWS` comparison (negated, since the
        // shared fn returns `use shmem staging` = the `else` branch).
        let t = crate::jit::partition_kernel::SHMEM_STAGING_MIN_ROWS;
        assert!(!use_shmem_staging_partition(0));
        assert!(!use_shmem_staging_partition(t - 1));
        assert!(use_shmem_staging_partition(t));
        assert!(use_shmem_staging_partition(t + 1));

        let t64 = crate::jit::partition_kernel_i64::SHMEM_STAGING_MIN_ROWS;
        assert!(!use_shmem_staging_partition_i64(0));
        assert!(!use_shmem_staging_partition_i64(t64 - 1));
        assert!(use_shmem_staging_partition_i64(t64));
        assert!(use_shmem_staging_partition_i64(t64 + 1));
    }

    #[test]
    fn collect_populated_slots_selects_set_and_sorts_by_key() {
        // Two partitions, two slots each. Mark some present, out of key
        // order, and confirm only present slots survive, sorted by key.
        let num_partitions = 2usize;
        let block_groups = 2usize;
        // pid 0: slot0 (key=5, present), slot1 (key=2, absent)
        // pid 1: slot0 (key=1, present), slot1 (key=9, present)
        let host_keys = [5, 2, 1, 9];
        let host_vals = [50i64, 20, 10, 90];
        let host_set = [1u8, 0, 1, 1];
        let pairs = collect_populated_slots_sorted::<i64>(
            &host_keys,
            &host_vals,
            &host_set,
            num_partitions,
            block_groups,
        );
        assert_eq!(pairs, vec![(1, 10), (5, 50), (9, 90)]);
    }

    #[test]
    fn collect_populated_slots_matches_inline_walk() {
        // Differential check against the exact inline walk every single-key
        // executor used to run, for an f64 value buffer (NaN in an absent
        // slot must not surface).
        let num_partitions = 3usize;
        let block_groups = 4usize;
        let n = num_partitions * block_groups;
        let host_keys: Vec<i32> = (0..n as i32).map(|i| (n as i32) - i).collect();
        let host_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut host_set = vec![0u8; n];
        for (i, s) in host_set.iter_mut().enumerate() {
            *s = (i % 2) as u8;
        }
        // Reference inline walk.
        let mut expected: Vec<(i32, f64)> = Vec::new();
        for pid in 0..num_partitions {
            let base = pid * block_groups;
            for slot in 0..block_groups {
                let idx = base + slot;
                if host_set[idx] != 0 {
                    expected.push((host_keys[idx], host_vals[idx]));
                }
            }
        }
        expected.sort_by_key(|(k, _)| *k);
        let got = collect_populated_slots_sorted::<f64>(
            &host_keys,
            &host_vals,
            &host_set,
            num_partitions,
            block_groups,
        );
        assert_eq!(got, expected);
    }
}
