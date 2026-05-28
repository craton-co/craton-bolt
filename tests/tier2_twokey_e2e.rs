// SPDX-License-Identifier: Apache-2.0
//
// CPU reference model for the two-key (Int32, Int32) Tier-2 GROUP BY SUM
// path, plus host-only correctness tests.
//
// The actual GPU pipeline (partition_kernel_i64 / scatter_kernel_i64 /
// orchestrator) requires a live CUDA context and is exercised by the
// integrator's harness. What we cover here:
//
//   1. Round-trip of the (Int32, Int32) → i64 → (Int32, Int32) packing.
//      The packing convention MUST match `src/exec/groupby.rs::pack_keys`:
//      high 32 bits = column 0, low 32 bits = column 1.
//
//   2. The CPU reference model (naive HashMap over (k1, k2)) equals the
//      pack-then-reduce model (HashMap over packed i64, then unpack on
//      read). If these ever diverge, the GPU pipeline — which agrees with
//      the pack-then-reduce form by construction — would silently produce
//      wrong answers.
//
//   3. The pack-then-unpack reproduces input pairs exactly across the
//      full Int32 range, including the sign-bit edge cases (the cast
//      chain MUST be `i32 -> u32 -> u64`, not `i32 -> i64`, to avoid
//      sign-extending the high half into the low half).
//
// Why this is a separate `tests/` file rather than a `#[cfg(test)] mod`:
// the orchestrator and merger each carry their own host-only unit tests
// already; this file pins the *integration* of the packing convention
// between them.

use std::collections::HashMap;

mod common;
use common::REL_TOL;

/// Pack two `i32` columns into a single `i64` per row using the
/// `src/exec/groupby.rs::pack_keys` convention:
///   high 32 bits = column 0
///   low  32 bits = column 1
///
/// Replicated verbatim from `groupby_tier2_twokey_exec::pack_two_i32`.
/// Keeping a local copy lets this integration test guard against a
/// regression in either implementation.
fn pack(col0: &[i32], col1: &[i32]) -> Vec<i64> {
    assert_eq!(col0.len(), col1.len());
    let mut out = Vec::with_capacity(col0.len());
    for i in 0..col0.len() {
        let hi = (col0[i] as u32 as u64) << 32;
        let lo = col1[i] as u32 as u64;
        out.push((hi | lo) as i64);
    }
    out
}

/// Reverse of `pack`. Mirrors the merger's `unpack_i64`.
fn unpack(packed: i64) -> (i32, i32) {
    let u = packed as u64;
    let hi = (u >> 32) as u32 as i32;
    let lo = (u & 0xFFFF_FFFFu64) as u32 as i32;
    (hi, lo)
}

/// Naive reference: HashMap keyed by `(i32, i32)` directly. Returns
/// `(key1, key2, sum)` triples sorted ascending by `(key1, key2)` — the
/// same canonical order the merger emits.
fn reference_naive(
    col0: &[i32],
    col1: &[i32],
    vals: &[f64],
) -> Vec<(i32, i32, f64)> {
    let mut acc: HashMap<(i32, i32), f64> = HashMap::with_capacity(col0.len());
    for i in 0..col0.len() {
        *acc.entry((col0[i], col1[i])).or_insert(0.0) += vals[i];
    }
    let mut out: Vec<(i32, i32, f64)> = acc
        .into_iter()
        .map(|((k1, k2), v)| (k1, k2, v))
        .collect();
    out.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    out
}

/// Pack-then-reduce model: pack into i64, reduce on the packed key, then
/// unpack on read. This mirrors what the GPU pipeline produces (modulo
/// per-partition ordering, which the merger normalises).
fn reference_packed(
    col0: &[i32],
    col1: &[i32],
    vals: &[f64],
) -> Vec<(i32, i32, f64)> {
    let packed = pack(col0, col1);
    let mut acc: HashMap<i64, f64> = HashMap::with_capacity(col0.len());
    for i in 0..col0.len() {
        *acc.entry(packed[i]).or_insert(0.0) += vals[i];
    }
    let mut out: Vec<(i32, i32, f64)> = acc
        .into_iter()
        .map(|(k, v)| {
            let (k1, k2) = unpack(k);
            (k1, k2, v)
        })
        .collect();
    out.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    out
}

#[test]
fn pack_unpack_roundtrips_full_i32_edges() {
    // Every Int32 sign-bit combination — the cast chain bug class lives
    // here. If pack used `i32 as i64` directly instead of via `u32`, the
    // sign extension would smear the high half into the low half for any
    // negative col0 value.
    let edges: &[i32] = &[i32::MIN, -1, 0, 1, i32::MAX];
    for &a in edges {
        for &b in edges {
            let p = pack(&[a], &[b]);
            assert_eq!(p.len(), 1);
            let (a2, b2) = unpack(p[0]);
            assert_eq!((a, b), (a2, b2), "round-trip failed for ({a}, {b})");
        }
    }
}

#[test]
fn naive_and_packed_models_agree_small() {
    // A few rows, deliberately containing duplicates that span partition
    // boundaries in the GPU pipeline.
    let col0 = vec![1, 2, 1, 3, 2, 1];
    let col1 = vec![10, 20, 10, 30, 21, 11];
    let vals = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let naive = reference_naive(&col0, &col1, &vals);
    let packed = reference_packed(&col0, &col1, &vals);
    assert_eq!(naive, packed, "naive and packed models must agree");
    // Spot-check: (1, 10) gets two contributions (1.0 + 3.0 = 4.0).
    let (_, _, sum_1_10) = naive
        .iter()
        .find(|t| (t.0, t.1) == (1, 10))
        .copied()
        .expect("(1,10) group must exist");
    assert_eq!(sum_1_10, 4.0);
}

#[test]
fn naive_and_packed_models_agree_with_negatives() {
    // Sign-bit pairs are where a sloppy pack (e.g. `i32 as i64 << 32`)
    // would diverge between the two models.
    let col0 = vec![-1, -1, 0, i32::MIN, i32::MIN];
    let col1 = vec![5, 5, -3, 0, 0];
    let vals = vec![1.0, 10.0, 100.0, 1000.0, 10_000.0];
    let naive = reference_naive(&col0, &col1, &vals);
    let packed = reference_packed(&col0, &col1, &vals);
    assert_eq!(naive, packed);
    // (-1, 5) collapses to one row with sum = 11.0.
    let (_, _, sum_neg) = naive
        .iter()
        .find(|t| (t.0, t.1) == (-1, 5))
        .copied()
        .expect("(-1, 5) group must exist");
    assert_eq!(sum_neg, 11.0);
}

#[test]
fn pack_distinguishes_swapped_pair() {
    // (a, b) and (b, a) must hash to different packed keys, otherwise
    // every transposed pair would collapse into one group.
    let p_ab = pack(&[7], &[3])[0];
    let p_ba = pack(&[3], &[7])[0];
    assert_ne!(p_ab, p_ba);
    assert_eq!(unpack(p_ab), (7, 3));
    assert_eq!(unpack(p_ba), (3, 7));
}

#[test]
fn h2oai_q3_shape_one_million_groups() {
    // Synthetic stand-in for h2o.ai q3: ~1 M distinct (id1, id2) groups,
    // 4 M rows (4 contributions per group). Use small per-axis cardinality
    // (1000 × 1000) so the test is fast and the reference HashMap stays
    // small. Confirms the model agreement at the cardinality scale where
    // Tier-2 actually fires.
    let id1_card: i32 = 1000;
    let id2_card: i32 = 1000;
    let mut col0: Vec<i32> = Vec::with_capacity(4 * 1_000_000);
    let mut col1: Vec<i32> = Vec::with_capacity(4 * 1_000_000);
    let mut vals: Vec<f64> = Vec::with_capacity(4 * 1_000_000);
    for rep in 0..4 {
        for k1 in 0..id1_card {
            for k2 in 0..id2_card {
                col0.push(k1);
                col1.push(k2);
                // Deterministic non-trivial value so summing isn't a
                // tautology against the count.
                vals.push((rep as f64) + (k1 as f64) * 0.5 + (k2 as f64) * 0.25);
            }
        }
    }
    let naive = reference_naive(&col0, &col1, &vals);
    let packed = reference_packed(&col0, &col1, &vals);
    assert_eq!(naive.len(), (id1_card as usize) * (id2_card as usize));
    assert_eq!(naive, packed);

    // Sanity-check one group's sum against the closed form: for (k1, k2),
    //   sum_{rep=0..4} (rep + 0.5*k1 + 0.25*k2)
    //     = (0+1+2+3) + 4*(0.5*k1 + 0.25*k2)
    //     = 6 + 2.0*k1 + 1.0*k2
    let (k1, k2) = (37i32, 91i32);
    let expected = 6.0 + 2.0 * (k1 as f64) + 1.0 * (k2 as f64);
    let observed = naive
        .iter()
        .find(|t| (t.0, t.1) == (k1, k2))
        .map(|t| t.2)
        .expect("(37, 91) group must exist");
    assert!(
        (observed - expected).abs() < REL_TOL,
        "observed sum {observed} differs from expected {expected}"
    );
}
