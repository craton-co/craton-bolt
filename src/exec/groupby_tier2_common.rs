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
}
