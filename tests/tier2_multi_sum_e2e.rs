// SPDX-License-Identifier: Apache-2.0

//! CPU reference model + regression hook for Tier-2 hash-partitioned GROUP BY
//! with **N SUM aggregates** (1..=4).
//!
//! Mirrors `tests/tier2_groupby_e2e.rs` but extended to N value columns. The
//! orchestrator's host pass-2 builds `HashMap<i32, [f64; N]>` per partition;
//! this file reproduces that algorithm in plain Rust so we can cross-validate
//! both against a naive single-pass groupby.
//!
//! The Tier-2 reordering of float adds means bit-exact equality with the
//! naive sum is not guaranteed; we test numerical equivalence within a tight
//! relative tolerance (1e-9), per the existing Tier-2 contract.

use std::collections::HashMap;

// ---- Tier-2 constants (mirror the kernel) ----------------------------------

const NUM_PARTITIONS: u32 = 1024;
const HASH_MULTIPLIER: u32 = 0x9E37_79B1;

#[inline]
fn partition_of(key: i32) -> u32 {
    (key as u32).wrapping_mul(HASH_MULTIPLIER) & (NUM_PARTITIONS - 1)
}

// ---- CPU reference models --------------------------------------------------

/// CPU model of the Tier-2 multi-SUM orchestrator's algorithm with N value
/// columns. Output is `(key, [sum_v0, ..., sum_v{N-1}])` tuples sorted by key.
fn cpu_tier2_multi_sum_model(
    keys: &[i32],
    vals: &[Vec<f64>], // shape: [n_vals][n_rows]
) -> Vec<(i32, Vec<f64>)> {
    let n_vals = vals.len();
    assert!(n_vals >= 1 && n_vals <= 4, "n_vals must be 1..=4, got {n_vals}");
    for (j, v) in vals.iter().enumerate() {
        assert_eq!(
            v.len(),
            keys.len(),
            "vals[{j}] length mismatch with keys ({} vs {})",
            v.len(),
            keys.len()
        );
    }

    // Pass 1: scatter row indices into per-partition buckets. We scatter the
    // *row index* (not the (key, vals...) tuple), so the bucket walk only
    // pays one pointer dereference per row regardless of N. This matches the
    // GPU pipeline's layout: the partition pass writes per-row partition
    // ids; the scatter pass writes per-row keys + values aligned by slot.
    let mut buckets: Vec<Vec<usize>> = (0..NUM_PARTITIONS).map(|_| Vec::new()).collect();
    for i in 0..keys.len() {
        let pid = partition_of(keys[i]) as usize;
        buckets[pid].push(i);
    }

    // Pass 2: per-partition HashMap<i32, [f64; N]>. Each partition's table is
    // tiny (~n_rows/K entries on average) so allocation/insert is fast.
    let mut flat: Vec<(i32, Vec<f64>)> = Vec::new();
    for bucket in &buckets {
        let mut table: HashMap<i32, Vec<f64>> = HashMap::with_capacity(bucket.len());
        for &i in bucket {
            let slot = table.entry(keys[i]).or_insert_with(|| vec![0.0; n_vals]);
            for j in 0..n_vals {
                slot[j] += vals[j][i];
            }
        }
        flat.extend(table.into_iter());
    }

    // Sort by key ASC for deterministic comparison.
    flat.sort_by_key(|(k, _)| *k);
    flat
}

/// Single-pass naive SUM-by-key over N value columns. Reference oracle.
fn cpu_naive_multi_sum_groupby(
    keys: &[i32],
    vals: &[Vec<f64>],
) -> Vec<(i32, Vec<f64>)> {
    let n_vals = vals.len();
    let mut table: HashMap<i32, Vec<f64>> = HashMap::with_capacity(keys.len().min(1 << 20));
    for i in 0..keys.len() {
        let slot = table.entry(keys[i]).or_insert_with(|| vec![0.0; n_vals]);
        for j in 0..n_vals {
            slot[j] += vals[j][i];
        }
    }
    let mut flat: Vec<(i32, Vec<f64>)> = table.into_iter().collect();
    flat.sort_by_key(|(k, _)| *k);
    flat
}

// ---- Fixture ---------------------------------------------------------------

/// Generate `(keys, vals[n_vals])` for unit tests. Deterministic from seed;
/// same xorshift64* pattern as the single-SUM Tier-2 test. Values land in
/// [-1.0, 1.0) so SUMs across millions of rows stay numerically interesting.
fn fixture(
    n_rows: usize,
    n_distinct_keys: i32,
    n_vals: usize,
    seed: u64,
) -> (Vec<i32>, Vec<Vec<f64>>) {
    assert!(n_distinct_keys > 0, "n_distinct_keys must be positive");
    assert!(n_vals >= 1 && n_vals <= 4, "n_vals must be 1..=4");
    let modulus = n_distinct_keys as u64;
    let mut state: u64 = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    if state == 0 {
        state = 0xDEAD_BEEF_CAFE_BABE;
    }
    let mut next = || -> u64 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };

    let mut keys = Vec::with_capacity(n_rows);
    let mut vals: Vec<Vec<f64>> = (0..n_vals).map(|_| Vec::with_capacity(n_rows)).collect();
    for _ in 0..n_rows {
        let r = next();
        let k = (r % modulus) as i32;
        keys.push(k);
        for j in 0..n_vals {
            let r2 = next();
            let unit = ((r2 >> 11) as f64) * (1.0_f64 / ((1_u64 << 53) as f64));
            // Vary scale slightly per column so a column-misalignment bug
            // shows up as wrong sums rather than coincidentally-matching
            // values across columns.
            let v = (unit * 2.0 - 1.0) * (1.0 + j as f64 * 0.5);
            vals[j].push(v);
        }
    }
    (keys, vals)
}

// ---- Helpers ---------------------------------------------------------------

/// Max relative error across N value columns. Both inputs must be sorted by
/// key and contain the same key set; we assert both. Absolute floor of 1.0
/// in the denominator so near-zero sums don't blow up the ratio.
fn max_relative_error_multi(
    a: &[(i32, Vec<f64>)],
    b: &[(i32, Vec<f64>)],
) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "result length mismatch: model={} naive={}",
        a.len(),
        b.len()
    );
    let mut worst = 0.0_f64;
    for i in 0..a.len() {
        assert_eq!(
            a[i].0, b[i].0,
            "key mismatch at index {i}: {} vs {}",
            a[i].0, b[i].0
        );
        assert_eq!(
            a[i].1.len(),
            b[i].1.len(),
            "n_vals mismatch at key {}",
            a[i].0
        );
        for j in 0..a[i].1.len() {
            let x = a[i].1[j];
            let y = b[i].1[j];
            let denom = x.abs().max(y.abs()).max(1.0);
            let rel = (x - y).abs() / denom;
            if rel > worst {
                worst = rel;
            }
        }
    }
    worst
}

// ---- CPU unit tests --------------------------------------------------------

#[test]
fn empty_input_n2() {
    let keys: Vec<i32> = Vec::new();
    let vals: Vec<Vec<f64>> = vec![Vec::new(), Vec::new()];
    let model = cpu_tier2_multi_sum_model(&keys, &vals);
    let naive = cpu_naive_multi_sum_groupby(&keys, &vals);
    assert!(model.is_empty());
    assert!(naive.is_empty());
}

#[test]
fn model_agrees_with_naive_n2_medium() {
    // h2o.ai q2-style: 1M rows over 100K keys, n_vals=2.
    let (keys, vals) = fixture(1_000_000, 100_000, 2, 0xA1);
    let model = cpu_tier2_multi_sum_model(&keys, &vals);
    let naive = cpu_naive_multi_sum_groupby(&keys, &vals);
    let err = max_relative_error_multi(&model, &naive);
    assert!(err < 1e-9, "max rel err {err:e} exceeded 1e-9");
}

#[test]
fn model_agrees_with_naive_n2_high_card() {
    // 10M rows over 1M keys, n_vals=2 — the q2 stress case.
    let (keys, vals) = fixture(10_000_000, 1_000_000, 2, 0xB2);
    let model = cpu_tier2_multi_sum_model(&keys, &vals);
    let naive = cpu_naive_multi_sum_groupby(&keys, &vals);
    let err = max_relative_error_multi(&model, &naive);
    assert!(err < 1e-9, "max rel err {err:e} exceeded 1e-9");
}

#[test]
fn model_agrees_with_naive_n4_medium() {
    // n_vals=4 (max supported by the v0 fast path). Stresses the per-key
    // [f64; N] accumulator allocation/walk.
    let (keys, vals) = fixture(1_000_000, 50_000, 4, 0xC3);
    let model = cpu_tier2_multi_sum_model(&keys, &vals);
    let naive = cpu_naive_multi_sum_groupby(&keys, &vals);
    let err = max_relative_error_multi(&model, &naive);
    assert!(err < 1e-9, "max rel err {err:e} exceeded 1e-9");
}

#[test]
fn model_agrees_with_naive_n1() {
    // Sanity: with n_vals=1 we should still match the existing single-SUM
    // Tier-2 behaviour bit-by-bit on small inputs (no big sums to drift).
    let (keys, vals) = fixture(10_000, 100, 1, 0xD4);
    let model = cpu_tier2_multi_sum_model(&keys, &vals);
    let naive = cpu_naive_multi_sum_groupby(&keys, &vals);
    let err = max_relative_error_multi(&model, &naive);
    assert!(err < 1e-10, "max rel err {err:e} exceeded 1e-10");
}

#[test]
fn deterministic_fixture() {
    let (k1, v1) = fixture(50_000, 64, 3, 0x1234_5678);
    let (k2, v2) = fixture(50_000, 64, 3, 0x1234_5678);
    assert_eq!(k1, k2, "keys must be deterministic from seed");
    assert_eq!(v1.len(), v2.len(), "vals shape must be deterministic");
    for j in 0..v1.len() {
        assert_eq!(v1[j], v2[j], "vals[{j}] must be deterministic from seed");
    }
}

#[test]
fn single_key_all_rows_n3() {
    // All rows map to the same key — every row should accumulate into one
    // output tuple, with three independent SUM totals.
    let n_rows = 50_000;
    let (_throwaway_keys, sample_vals) = fixture(n_rows, 1, 3, 0xE5);
    let keys: Vec<i32> = vec![7; n_rows];
    let model = cpu_tier2_multi_sum_model(&keys, &sample_vals);
    let naive = cpu_naive_multi_sum_groupby(&keys, &sample_vals);
    assert_eq!(model.len(), 1);
    assert_eq!(model[0].0, 7);
    assert_eq!(model[0].1.len(), 3);
    let err = max_relative_error_multi(&model, &naive);
    assert!(err < 1e-12, "max rel err {err:e} exceeded 1e-12");
}

#[test]
fn negative_keys_round_trip_n2() {
    // Keys in [-1000, 1000] (2001 distinct values). Partition function must
    // round-trip negative i32 through the (k as u32) cast in the hash.
    let n_rows = 50_000;
    let (raw_keys, vals) = fixture(n_rows, 2001, 2, 0xF6);
    let keys: Vec<i32> = raw_keys.iter().map(|&k| k - 1000).collect();
    let model = cpu_tier2_multi_sum_model(&keys, &vals);
    let naive = cpu_naive_multi_sum_groupby(&keys, &vals);
    assert!(
        model.iter().any(|(k, _)| *k < 0),
        "expected at least one negative key in output"
    );
    assert_eq!(model.len(), naive.len(), "distinct-key count mismatch");
    let err = max_relative_error_multi(&model, &naive);
    assert!(err < 1e-10, "max rel err {err:e} exceeded 1e-10");
}

// ---- GPU-gated integration test --------------------------------------------
//
// Regression hook for the Tier-2 multi-SUM GPU pipeline. Once the dispatcher
// is wired into `execute_groupby` (separate integration step), drop
// `#[ignore]` and finish the body.

#[test]
#[ignore = "requires CUDA + integration"]
fn tier2_multi_pipeline_matches_cpu_model() {
    // q2-style fixture: 10M rows over 1M distinct keys, n_vals=2.
    let n_rows: usize = 10_000_000;
    let n_distinct_keys: i32 = 1_000_000;
    let n_vals: usize = 2;
    let (keys, vals) = fixture(n_rows, n_distinct_keys, n_vals, 42);

    // Oracle: the CPU multi-SUM model. The orchestrator's host pass-2 walks
    // the same shape, so this is the tighter contract than the naive
    // single-pass groupby (which differs by ~1e-12 due to reordered adds).
    let expected = cpu_tier2_multi_sum_model(&keys, &vals);

    // Wire-up:
    //   1. Build a RecordBatch with columns: `id2` (Int32) from `keys`,
    //      `v1` (Float64) from `vals[0]`, `v2` (Float64) from `vals[1]`.
    //   2. let mut engine = craton_patina::Engine::new().unwrap();
    //      engine.register_table("x", batch).unwrap();
    //   3. let h = engine.sql("SELECT id2, SUM(v1), SUM(v2) FROM x GROUP BY id2").unwrap();
    //      let out = h.record_batch();
    //   4. Extract (id2, sum_v1, sum_v2), sort by key, compare with
    //      `max_relative_error_multi(..) < 1e-9`.
    let _ = expected.len();
    unimplemented!(
        "wire engine.sql -> sort by key -> compare against expected with max_relative_error_multi"
    );
}
