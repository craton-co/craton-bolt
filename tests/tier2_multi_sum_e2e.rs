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

mod common;
use common::Xorshift64Star;

use std::collections::HashMap;

// ---- Tier-2 constants (mirror the kernel) ----------------------------------
//
// `NUM_PARTITIONS` is imported from the crate's `__test_only_partition_offsets`
// re-export so the oracle cannot drift away from the GPU kernel constant
// (review C1).
use craton_bolt::__test_only_partition_offsets::NUM_PARTITIONS;

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
/// uses the shared `Xorshift64Star` PRNG, same stream as the single-SUM
/// Tier-2 test. Values land in `[-1.0, 1.0)` so SUMs across millions of
/// rows stay numerically interesting.
fn fixture(
    n_rows: usize,
    n_distinct_keys: i32,
    n_vals: usize,
    seed: u64,
) -> (Vec<i32>, Vec<Vec<f64>>) {
    assert!(n_distinct_keys > 0, "n_distinct_keys must be positive");
    assert!(n_vals >= 1 && n_vals <= 4, "n_vals must be 1..=4");
    let modulus = n_distinct_keys as u64;
    let mut rng = Xorshift64Star::new(seed);

    let mut keys = Vec::with_capacity(n_rows);
    let mut vals: Vec<Vec<f64>> = (0..n_vals).map(|_| Vec::with_capacity(n_rows)).collect();
    for _ in 0..n_rows {
        let k = (rng.next_u64() % modulus) as i32;
        keys.push(k);
        for j in 0..n_vals {
            // Vary scale slightly per column so a column-misalignment bug
            // shows up as wrong sums rather than coincidentally-matching
            // values across columns.
            let v = rng.next_signed_unit_f64() * (1.0 + j as f64 * 0.5);
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

// ---- Per-key alignment tests -----------------------------------------------
//
// These tests directly attack the kind of bug the old multi-call scatter
// design could produce: a value column j ending up paired with the wrong
// key. We build inputs where each (key, v0, v1[, v2]) tuple has a unique
// arithmetic relationship between its value columns, so a misalignment
// shows up as an impossible per-group sum that no permutation of values
// against the right key could produce.

/// Fixture where, for every row, `v1 = 1000 * key` and `v2 = 1_000_000 * key`.
/// After GROUP BY, `SUM(v1) / SUM(v2) = 1/1000` and both sums are exact
/// integer multiples of the row count per group. A misalignment of v1
/// vs v2 against the key column would produce sums whose ratios are not
/// 1/1000 — easy to detect.
fn fixture_aligned_multiples(
    n_rows: usize,
    n_distinct_keys: i32,
    n_vals: usize,
    seed: u64,
) -> (Vec<i32>, Vec<Vec<f64>>) {
    assert!(n_vals >= 1 && n_vals <= 4);
    let modulus = n_distinct_keys as u64;
    let mut state: u64 = seed.wrapping_add(0xDEAD_BEEF_CAFE_BABE);
    if state == 0 {
        state = 0xA1B2_C3D4_E5F6_0718;
    }
    let mut next = || -> u64 {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state.wrapping_mul(0x2545_F491_4F6C_DD1D)
    };

    let mut keys = Vec::with_capacity(n_rows);
    let mut vals: Vec<Vec<f64>> = (0..n_vals).map(|_| Vec::with_capacity(n_rows)).collect();
    let per_col_scale: [f64; 4] = [1.0, 1_000.0, 1_000_000.0, 1_000_000_000.0];
    for _ in 0..n_rows {
        let k = (next() % modulus) as i32;
        keys.push(k);
        for j in 0..n_vals {
            // Each row's value in column j is `scale_j * key`. Sums per
            // group preserve this ratio exactly: SUM(v_j) over a group is
            // `scale_j * key * count_for_key`. A scatter that paired v_j
            // for row i with the wrong key would break the ratio.
            vals[j].push(per_col_scale[j] * (k as f64));
        }
    }
    (keys, vals)
}

#[test]
fn aligned_multiples_n2_ratio_holds() {
    // Small N so the CPU model is fast; exercises the alignment invariant
    // without needing a GPU.
    let (keys, vals) = fixture_aligned_multiples(50_000, 500, 2, 0xABCD);
    let model = cpu_tier2_multi_sum_model(&keys, &vals);
    let naive = cpu_naive_multi_sum_groupby(&keys, &vals);

    // The model is the algorithm the orchestrator runs host-side; the naive
    // path is a true oracle. They must agree.
    let err = max_relative_error_multi(&model, &naive);
    assert!(err < 1e-10, "model vs naive: max rel err {err:e}");

    // Now check the alignment invariant directly: for every key, the
    // returned SUM(v0) and SUM(v1) must satisfy v1_sum == 1000 * v0_sum.
    // Even at f64 precision this is an exact integer relationship for
    // small keys + row counts.
    for (k, sums) in &model {
        assert_eq!(sums.len(), 2);
        let v0 = sums[0];
        let v1 = sums[1];
        // SUM(v0) for key k = k * count_k; SUM(v1) = 1000 * k * count_k.
        // So v1 should be exactly 1000 * v0.
        let expected_v1 = v0 * 1000.0;
        let denom = expected_v1.abs().max(1.0);
        let rel = (v1 - expected_v1).abs() / denom;
        assert!(
            rel < 1e-12,
            "alignment broken at key {k}: SUM(v0)={v0} SUM(v1)={v1} (expected {expected_v1}, rel err {rel:e})"
        );
    }
}

#[test]
fn aligned_multiples_n3_ratio_holds() {
    let (keys, vals) = fixture_aligned_multiples(100_000, 1_000, 3, 0xBEEF);
    let model = cpu_tier2_multi_sum_model(&keys, &vals);
    let naive = cpu_naive_multi_sum_groupby(&keys, &vals);

    let err = max_relative_error_multi(&model, &naive);
    assert!(err < 1e-10, "model vs naive: max rel err {err:e}");

    // SUM(v0) for key k = k * count_k.
    // SUM(v1) = 1_000 * SUM(v0).
    // SUM(v2) = 1_000_000 * SUM(v0).
    for (k, sums) in &model {
        assert_eq!(sums.len(), 3);
        let v0 = sums[0];
        let v1 = sums[1];
        let v2 = sums[2];

        let expected_v1 = v0 * 1_000.0;
        let denom1 = expected_v1.abs().max(1.0);
        let rel1 = (v1 - expected_v1).abs() / denom1;
        assert!(
            rel1 < 1e-12,
            "alignment broken (v1) at key {k}: v0={v0} v1={v1} expected {expected_v1} (rel {rel1:e})"
        );

        let expected_v2 = v0 * 1_000_000.0;
        let denom2 = expected_v2.abs().max(1.0);
        let rel2 = (v2 - expected_v2).abs() / denom2;
        assert!(
            rel2 < 1e-12,
            "alignment broken (v2) at key {k}: v0={v0} v2={v2} expected {expected_v2} (rel {rel2:e})"
        );
    }
}

#[test]
fn known_sums_per_group_n2() {
    // Hand-built tiny fixture with known sums per group. If alignment between
    // v0 and v1 ever drifts in the orchestrator's host or GPU paths, this
    // test will fail with sums that do not match the hand-computed totals.
    //   key=10:  v0 = 1+2+3 = 6,    v1 = 10+20+30 = 60
    //   key=20:  v0 = 4+5   = 9,    v1 = 40+50    = 90
    //   key=30:  v0 = 6     = 6,    v1 = 60       = 60
    //   key=-7:  v0 = 7+8   = 15,   v1 = 70+80    = 150
    let keys: Vec<i32> = vec![10, 20, 10, 20, 10, 30, -7, -7];
    let v0: Vec<f64> = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 7.0, 8.0];
    let v1: Vec<f64> = vec![10.0, 40.0, 20.0, 50.0, 30.0, 60.0, 70.0, 80.0];
    let vals = vec![v0, v1];

    let model = cpu_tier2_multi_sum_model(&keys, &vals);

    // Build a lookup keyed by group key.
    let mut got: HashMap<i32, Vec<f64>> = HashMap::new();
    for (k, sums) in model {
        got.insert(k, sums);
    }

    let expected: &[(i32, [f64; 2])] = &[
        (-7, [15.0, 150.0]),
        (10, [6.0, 60.0]),
        (20, [9.0, 90.0]),
        (30, [6.0, 60.0]),
    ];
    for (k, exp) in expected {
        let g = got.get(k).unwrap_or_else(|| panic!("missing key {k}"));
        assert_eq!(g.len(), 2);
        assert!(
            (g[0] - exp[0]).abs() < 1e-12,
            "SUM(v0) for key {k}: got {} expected {}",
            g[0],
            exp[0]
        );
        assert!(
            (g[1] - exp[1]).abs() < 1e-12,
            "SUM(v1) for key {k}: got {} expected {}",
            g[1],
            exp[1]
        );
        // Most importantly: v1 should be exactly 10 * v0 in this fixture.
        // A misalignment would break this ratio.
        let expected_ratio = g[0] * 10.0;
        assert!(
            (g[1] - expected_ratio).abs() < 1e-12,
            "alignment broken at key {k}: v0={} v1={} (v1 should be 10*v0 = {expected_ratio})",
            g[0],
            g[1]
        );
    }
}

#[test]
fn no_value_column_swap_under_permutation_n2() {
    // Build a fixture where swapping the v0 and v1 results for any single
    // key would yield different (non-physical) per-group sums. We then
    // verify the model produces the physical sums, not the swapped ones.
    //
    // Each key gets a unique constant for v0 and a unique-but-different
    // constant for v1, with row counts that vary per key. This guarantees
    // that no two keys could share the same (SUM(v0), SUM(v1)) tuple,
    // so a misalignment that paired v1 with the wrong key would surface.
    let mut keys: Vec<i32> = Vec::new();
    let mut v0: Vec<f64> = Vec::new();
    let mut v1: Vec<f64> = Vec::new();
    // key k has k+1 rows, v0=k, v1=100+k.
    for k in 0..50_i32 {
        let n = (k + 1) as usize;
        for _ in 0..n {
            keys.push(k);
            v0.push(k as f64);
            v1.push(100.0 + k as f64);
        }
    }

    let vals = vec![v0, v1];
    let model = cpu_tier2_multi_sum_model(&keys, &vals);

    for (k, sums) in &model {
        let n = (*k + 1) as f64;
        let expected_v0 = (*k as f64) * n;
        let expected_v1 = (100.0 + *k as f64) * n;
        assert!(
            (sums[0] - expected_v0).abs() < 1e-9,
            "key {k}: SUM(v0) got {} expected {}",
            sums[0],
            expected_v0
        );
        assert!(
            (sums[1] - expected_v1).abs() < 1e-9,
            "key {k}: SUM(v1) got {} expected {}",
            sums[1],
            expected_v1
        );
    }
}

// ---- GPU-gated integration test --------------------------------------------
//
// Regression hook for the Tier-2 multi-SUM GPU pipeline. Once the dispatcher
// is wired into `execute_groupby` (separate integration step), drop
// `#[ignore]` and finish the body.

#[test]
#[ignore = "requires CUDA + integration"]
fn tier2_multi_pipeline_matches_cpu_model() {
    use std::sync::Arc;

    use arrow_array::{Array, Float64Array, Int32Array, RecordBatch};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    // q2-style fixture: 10M rows over 1M distinct keys, n_vals=2.
    let n_rows: usize = 10_000_000;
    let n_distinct_keys: i32 = 1_000_000;
    let n_vals: usize = 2;
    let (keys, vals) = fixture(n_rows, n_distinct_keys, n_vals, 42);

    // Oracle: the CPU multi-SUM model. The orchestrator's host pass-2 walks
    // the same shape, so this is the tighter contract than the naive
    // single-pass groupby (which differs by ~1e-12 due to reordered adds).
    let expected = cpu_tier2_multi_sum_model(&keys, &vals);

    // Build a RecordBatch with columns `id2` (Int32), `v1` (Float64), `v2` (Float64).
    let id2: Int32Array = keys.iter().copied().collect();
    let v1: Float64Array = vals[0].iter().copied().collect();
    let v2: Float64Array = vals[1].iter().copied().collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id2", ArrowDataType::Int32, false),
        ArrowField::new("v1", ArrowDataType::Float64, false),
        ArrowField::new("v2", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(id2), Arc::new(v1), Arc::new(v2)],
    )
    .expect("build RecordBatch");

    // Stand up the engine on the default CUDA device. Mirrors the convention
    // in `tests/memory_tests.rs`: `.expect()` is fine because the test is
    // `#[ignore]`'d, so it only runs on a GPU host.
    let mut engine = craton_bolt::Engine::new().expect("CUDA engine");
    engine
        .register_table("x", batch)
        .expect("register table");

    let h = engine
        .sql("SELECT id2, SUM(v1), SUM(v2) FROM x GROUP BY id2")
        .expect("execute multi-sum groupby");
    let out = h.record_batch();

    // The output schema is SELECT-ordered: [id2, sum_v1, sum_v2].
    let id_col = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id2 must be Int32");
    let sum_v1_col = out
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("SUM(v1) must be Float64");
    let sum_v2_col = out
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("SUM(v2) must be Float64");

    // Flatten into the (key, [sums]) shape the oracle uses and sort by key.
    let mut actual: Vec<(i32, Vec<f64>)> = (0..out.num_rows())
        .map(|i| {
            (
                id_col.value(i),
                vec![sum_v1_col.value(i), sum_v2_col.value(i)],
            )
        })
        .collect();
    actual.sort_by_key(|(k, _)| *k);

    let err = max_relative_error_multi(&actual, &expected);
    assert!(
        err < 1e-9,
        "GPU multi-SUM pipeline vs CPU model: max rel err {err:e} exceeded 1e-9"
    );
}

/// GPU-side alignment regression. Uses the aligned-multiples fixture so a
/// (key, v_j) misalignment in the orchestrator's scatter path surfaces as
/// a broken `SUM(v1) == 1000 * SUM(v0)` ratio — independent of any
/// floating-point reordering. The CPU model already runs the same fixture
/// above; this is the live-GPU end of the regression.
#[test]
#[ignore = "requires CUDA + integration"]
fn tier2_multi_pipeline_preserves_value_column_alignment() {
    let n_rows: usize = 10_000_000;
    let n_distinct_keys: i32 = 100_000;
    let n_vals: usize = 3;
    let (keys, vals) = fixture_aligned_multiples(n_rows, n_distinct_keys, n_vals, 0xFEED);
    let expected = cpu_tier2_multi_sum_model(&keys, &vals);

    // Wire-up (mirrors the previous test):
    //   1. RecordBatch with `id2` (Int32) from keys, `v1`/`v2`/`v3` (Float64)
    //      from vals[0..3].
    //   2. engine.sql("SELECT id2, SUM(v1), SUM(v2), SUM(v3) FROM x GROUP BY id2")
    //   3. Sort by key; per-key, assert
    //        SUM(v2) == 1000 * SUM(v1) within 1e-12 relative
    //        SUM(v3) == 1_000_000 * SUM(v1) within 1e-12 relative
    //      and that the per-key totals match `expected` within 1e-9
    //      relative via `max_relative_error_multi`.
    let _ = expected.len();
    unimplemented!(
        "wire engine.sql -> per-key ratio + max_relative_error_multi against expected"
    );
}
