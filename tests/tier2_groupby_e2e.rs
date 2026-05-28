// SPDX-License-Identifier: Apache-2.0

//! Test scaffolding for Tier 2 of the GROUP BY perf plan: hash-partitioned
//! two-pass aggregation, designed for the high-cardinality regime
//! (n_groups >> shared-memory capacity) where the Tier-1 block-local kernel
//! can no longer hold a per-block accumulator.
//!
//! The Tier-2 pipeline is:
//!   1. compute `partition_id = hash(key) % NUM_PARTITIONS` per row
//!   2. scatter rows into per-partition buffers
//!   3. for each partition, reduce keys -> SUM with a small hash table that
//!      fits in L2 (or the Tier-1 shared-mem kernel, once n_groups/K is small
//!      enough)
//!   4. concatenate per-partition results
//!
//! Step 3 reorders float adds, so bit-exact equality with a naive single-pass
//! reduction is not guaranteed; numerical equivalence within a tight relative
//! tolerance is the contract we test.
//!
//! This file ships:
//!   1. `cpu_tier2_sum_model` — pure-Rust mirror of the Tier-2 algorithm,
//!      using std `HashMap<i32, f64>` per partition. This is the oracle the
//!      GPU pipeline will be compared against.
//!   2. `cpu_naive_sum_groupby` — single-HashMap reference, used to
//!      cross-validate that the tier2 model is itself correct.
//!   3. Deterministic fixture builders.
//!   4. CPU-only unit tests (no `#[ignore]`) covering the cardinality
//!      envelope Tier 2 must handle.
//!   5. A `#[ignore]`'d integration test (`tier2_pipeline_matches_cpu_model`)
//!      that is the regression hook for the GPU pipeline landing — drop
//!      `#[ignore]` and fill in the body once a sibling worktree merges.
//!
//! Algorithm context: see `docs/GROUPBY_PERF.md` Tier 2.

mod common;
use common::{Xorshift64Star, REL_TOL};

use std::collections::HashMap;

// ---- Tier-2 constants -------------------------------------------------------
//
// These MUST match the GPU partition kernel exactly — a mismatch would mean
// the oracle is checking the wrong reduction order and the cross-validation
// would silently let real bugs through. We import `NUM_PARTITIONS` from the
// crate's `__test_only_partition_offsets` re-export rather than hard-coding
// the value so a future change to the kernel constant cannot drift the
// oracle out from under us (review C1).
use craton_bolt::__test_only_partition_offsets::NUM_PARTITIONS;

/// Knuth-style multiplicative constant (golden ratio fraction of 2^32).
/// This is the same constant used by the partition kernel.
const HASH_MULTIPLIER: u32 = 0x9E37_79B1;

/// `partition_id` for a given key. Centralised so the oracle and the
/// evenness test cannot drift apart from the kernel by accident.
#[inline]
fn partition_of(key: i32) -> u32 {
    (key as u32).wrapping_mul(HASH_MULTIPLIER) & (NUM_PARTITIONS - 1)
}

// ---- CPU references ---------------------------------------------------------

/// CPU model of the Tier-2 hash-partitioned GROUP BY SUM.
///
/// Mirrors what the GPU pipeline will do:
///   1. `partition[i] = hash(keys[i]) % NUM_PARTITIONS`
///   2. scatter rows into per-partition buckets
///   3. for each partition: `HashMap<i32, f64>` reduce keys -> sum
///   4. concat per-partition results into a flat `(key, sum)` list
///
/// Output is sorted by key ASC so callers can index it deterministically
/// and so the partition-driven reordering does not leak into the API.
fn cpu_tier2_sum_model(keys: &[i32], vals: &[f64]) -> Vec<(i32, f64)> {
    assert_eq!(keys.len(), vals.len(), "keys/vals length mismatch");

    // Pass 1: scatter row indices into per-partition buckets. We scatter
    // (key, val) pairs directly rather than indices because the per-partition
    // reducer in step 2 only needs the pair, and indirecting through indices
    // adds a layer of cache misses that would skew the comparison.
    let mut buckets: Vec<Vec<(i32, f64)>> = (0..NUM_PARTITIONS)
        .map(|_| Vec::new())
        .collect();
    for i in 0..keys.len() {
        let pid = partition_of(keys[i]) as usize;
        buckets[pid].push((keys[i], vals[i]));
    }

    // Pass 2: per-partition HashMap reduce. Each partition's HashMap holds
    // only the subset of distinct keys whose hash lands in that partition,
    // which is the entire point of the algorithm: small per-partition tables
    // fit in L2 (on the GPU) / L1 (on the CPU).
    //
    // Pass 3: concat. We collect into a flat (key, sum) vec; partitions are
    // disjoint by construction so there's no second-level merge to do.
    let mut flat: Vec<(i32, f64)> = Vec::new();
    for bucket in &buckets {
        let mut table: HashMap<i32, f64> = HashMap::with_capacity(bucket.len());
        for &(k, v) in bucket {
            *table.entry(k).or_insert(0.0) += v;
        }
        flat.extend(table.into_iter());
    }

    // Sort by key ASC so the caller can compare deterministically.
    flat.sort_by_key(|&(k, _)| k);
    flat
}

/// Single-pass naive SUM-by-key. Uses one `HashMap<i32, f64>` for the entire
/// input. The reference the tier2 model is cross-validated against.
fn cpu_naive_sum_groupby(keys: &[i32], vals: &[f64]) -> Vec<(i32, f64)> {
    assert_eq!(keys.len(), vals.len(), "keys/vals length mismatch");
    let mut table: HashMap<i32, f64> = HashMap::with_capacity(keys.len().min(1 << 20));
    for i in 0..keys.len() {
        *table.entry(keys[i]).or_insert(0.0) += vals[i];
    }
    let mut flat: Vec<(i32, f64)> = table.into_iter().collect();
    flat.sort_by_key(|&(k, _)| k);
    flat
}

// ---- Fixture ----------------------------------------------------------------

/// Generate `(keys, vals)` for the unit tests. Deterministic from a seed so
/// tests reproduce across runs and across the nine sibling worktrees.
///
/// `n_distinct_keys` controls cardinality: keys are drawn uniformly from
/// `[0, n_distinct_keys)`. Note that the actual *distinct* count in the
/// output is `min(n_rows, n_distinct_keys)` once collisions take effect —
/// e.g. n_rows=10M, n_distinct_keys=5M will surface ~3.9M distinct keys
/// after birthday-paradox collisions, which is what the
/// `model_agrees_with_naive_super_high` test exercises.
///
/// Uses the shared `Xorshift64Star` from `tests/common/mod.rs` — no extra
/// dev-deps. Values are in `[-1.0, 1.0)` so SUMs across millions of rows
/// stay in a numerically interesting range and the reordered partition sums
/// diverge from the naive single-pass sum by an amount visible to a tight
/// `1e-9` tolerance.
fn fixture(n_rows: usize, n_distinct_keys: i32, seed: u64) -> (Vec<i32>, Vec<f64>) {
    assert!(n_distinct_keys > 0, "n_distinct_keys must be positive");
    let modulus = n_distinct_keys as u64;
    let mut rng = Xorshift64Star::new(seed);

    let mut keys = Vec::with_capacity(n_rows);
    let mut vals = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let k = (rng.next_u64() % modulus) as i32;
        let v = rng.next_signed_unit_f64();
        keys.push(k);
        vals.push(v);
    }
    (keys, vals)
}

// ---- Helpers ----------------------------------------------------------------

/// Max relative error between two `(key, sum)` lists. Both must be sorted by
/// key ASC and contain the same key set; we assert both. Uses an absolute
/// floor of 1.0 in the denominator so near-zero sums don't blow up the ratio.
fn max_relative_error(a: &[(i32, f64)], b: &[(i32, f64)]) -> f64 {
    assert_eq!(
        a.len(),
        b.len(),
        "result length mismatch: model={} naive={}",
        a.len(),
        b.len()
    );
    let mut worst = 0.0_f64;
    for i in 0..a.len() {
        assert_eq!(a[i].0, b[i].0, "key mismatch at index {i}: {} vs {}", a[i].0, b[i].0);
        let x = a[i].1;
        let y = b[i].1;
        let denom = x.abs().max(y.abs()).max(1.0);
        let rel = (x - y).abs() / denom;
        if rel > worst {
            worst = rel;
        }
    }
    worst
}

// ---- CPU unit tests (no `#[ignore]`) ---------------------------------------

#[test]
fn model_agrees_with_naive_low_card() {
    // 10k rows over 100 keys — the q1-style low-cardinality case.
    let (keys, vals) = fixture(10_000, 100, 0xA1);
    let model = cpu_tier2_sum_model(&keys, &vals);
    let naive = cpu_naive_sum_groupby(&keys, &vals);
    let err = max_relative_error(&model, &naive);
    assert!(err < 1e-10, "max rel err {err:e} exceeded 1e-10");
}

#[test]
fn model_agrees_with_naive_medium() {
    // 100k rows over 10k keys — the q2-style medium-cardinality case.
    let (keys, vals) = fixture(100_000, 10_000, 0xB2);
    let model = cpu_tier2_sum_model(&keys, &vals);
    let naive = cpu_naive_sum_groupby(&keys, &vals);
    let err = max_relative_error(&model, &naive);
    assert!(err < REL_TOL, "max rel err {err:e} exceeded {REL_TOL:e}");
}

#[test]
fn model_agrees_with_naive_high_card() {
    // 10M rows over 1M keys — the q5-style high-cardinality case that
    // Tier 2 was designed for.
    let (keys, vals) = fixture(10_000_000, 1_000_000, 0xC3);
    let model = cpu_tier2_sum_model(&keys, &vals);
    let naive = cpu_naive_sum_groupby(&keys, &vals);
    let err = max_relative_error(&model, &naive);
    assert!(err < REL_TOL, "max rel err {err:e} exceeded {REL_TOL:e}");
}

#[test]
fn model_agrees_with_naive_super_high() {
    // 10M rows over 5M nominal keys — birthday-paradox collisions yield
    // ~3.9M actual distinct keys. Stresses the per-partition table sizing
    // and exercises the regime where most groups have very few rows.
    let (keys, vals) = fixture(10_000_000, 5_000_000, 0xD4);
    let model = cpu_tier2_sum_model(&keys, &vals);
    let naive = cpu_naive_sum_groupby(&keys, &vals);
    let err = max_relative_error(&model, &naive);
    assert!(err < REL_TOL, "max rel err {err:e} exceeded {REL_TOL:e}");
}

#[test]
fn model_handles_single_key() {
    // Degenerate cardinality: every row maps to a single partition. Output
    // must be a single (42, sum) tuple.
    let n_rows = 100_000;
    let key = 42_i32;
    let mut keys = Vec::with_capacity(n_rows);
    let mut vals = Vec::with_capacity(n_rows);
    let (_throwaway_keys, sample_vals) = fixture(n_rows, 1, 0xE5);
    for v in &sample_vals {
        keys.push(key);
        vals.push(*v);
    }
    let model = cpu_tier2_sum_model(&keys, &vals);
    let naive = cpu_naive_sum_groupby(&keys, &vals);
    assert_eq!(model.len(), 1, "expected exactly one output row, got {}", model.len());
    assert_eq!(model[0].0, key, "expected key=42, got key={}", model[0].0);
    let err = max_relative_error(&model, &naive);
    assert!(err < 1e-12, "max rel err {err:e} exceeded 1e-12");
}

#[test]
fn model_handles_negative_keys() {
    // Keys in [-1000, 1000] (2001 distinct values). The partition function
    // must handle negative i32 correctly via the `as u32` cast — negative
    // keys partition to high-bit residues and must round-trip back.
    let n_rows = 50_000;
    let (raw_keys, vals) = fixture(n_rows, 2001, 0xF6);
    let keys: Vec<i32> = raw_keys.iter().map(|&k| k - 1000).collect();

    let model = cpu_tier2_sum_model(&keys, &vals);
    let naive = cpu_naive_sum_groupby(&keys, &vals);

    // Sanity: at least one negative key should show up given the wide range.
    assert!(
        model.iter().any(|&(k, _)| k < 0),
        "expected at least one negative key in output"
    );
    // Every key present in `naive` (the ground truth) must also appear in
    // the model's output — partition-by-hash must not drop negative keys.
    assert_eq!(model.len(), naive.len(), "distinct-key count mismatch");
    let err = max_relative_error(&model, &naive);
    assert!(err < 1e-10, "max rel err {err:e} exceeded 1e-10");
}

#[test]
fn fixture_is_deterministic() {
    let (k1, v1) = fixture(50_000, 64, 0x1234_5678);
    let (k2, v2) = fixture(50_000, 64, 0x1234_5678);
    assert_eq!(k1, k2, "keys must be deterministic from seed");
    assert_eq!(v1, v2, "vals must be deterministic from seed");
}

#[test]
fn partition_function_distributes_evenly() {
    // Hash 1M sequential keys. The multiplicative-by-odd hash is a bijection
    // on the low log2(NUM_PARTITIONS) bits, so the ideal load is
    // `N / NUM_PARTITIONS` per partition. We assert a generous +/-15 % band
    // around the ideal so any future change to the hash constant that
    // broke distribution would still fail here without false alarms on
    // legitimate variance. The band is computed from `NUM_PARTITIONS` so
    // it auto-tracks the kernel constant (review C1 — see
    // `__test_only_partition_offsets`).
    const N: u32 = 1_000_000;
    let mut counts = vec![0_u32; NUM_PARTITIONS as usize];
    for k in 0..N as i32 {
        let pid = partition_of(k) as usize;
        counts[pid] += 1;
    }

    let min = *counts.iter().min().unwrap();
    let max = *counts.iter().max().unwrap();
    let total: u64 = counts.iter().map(|&c| c as u64).sum();
    assert_eq!(total, N as u64, "every key must land in exactly one partition");

    let ideal = N / NUM_PARTITIONS;
    let lo = ideal - ideal / 7; // ~ -14 %
    let hi = ideal + ideal / 7; // ~ +14 %
    assert!(
        min >= lo && max <= hi,
        "partition load out of bounds: min={min} max={max} (expected [{lo}, {hi}], ideal={ideal})"
    );
}

// ---- GPU-gated integration test --------------------------------------------
//
// Regression hook for the Tier-2 GPU pipeline. Once the sibling worktrees
// merge the partition + per-partition-reduce kernels and the dispatch
// heuristic, drop `#[ignore]` and finish the body. The intent is documented
// inline so the integrator can wire it up mechanically.

#[test]
#[ignore = "gpu:tier2"]
fn tier2_pipeline_matches_cpu_model() {
    use std::sync::Arc;

    use arrow_array::{Array, Float64Array, Int32Array, RecordBatch};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    // Build a 10M-row fixture with 1M distinct keys — the q5-style stress
    // case Tier 2 was designed for. Same seed as the CPU test above so
    // mismatches can be debugged side-by-side.
    let n_rows: usize = 10_000_000;
    let n_distinct_keys: i32 = 1_000_000;
    let (keys, vals) = fixture(n_rows, n_distinct_keys, 42);

    // Expected (key, sum) output, sorted by key ASC.
    let expected = cpu_tier2_sum_model(&keys, &vals);

    // Build a RecordBatch with columns `id1` (Int32) and `v1` (Float64).
    let id1: Int32Array = keys.iter().copied().collect();
    let v1: Float64Array = vals.iter().copied().collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id1", ArrowDataType::Int32, false),
        ArrowField::new("v1", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(id1), Arc::new(v1)])
        .expect("build RecordBatch");

    // Stand up the engine on the default CUDA device. Mirrors the convention
    // in `tests/memory_tests.rs`: `.expect()` is fine because the test is
    // `#[ignore]`'d, so it only runs on a GPU host.
    let mut engine = craton_bolt::Engine::new().expect("CUDA engine");
    engine
        .register_table("x", batch)
        .expect("register table");

    let h = engine
        .sql("SELECT id1, SUM(v1) FROM x GROUP BY id1")
        .expect("execute groupby");
    let out = h.record_batch();

    // The output schema is SELECT-ordered: [id1, sum_v1]. The dispatcher
    // (and the Tier-2 pipeline once landed) emits rows in an
    // implementation-defined order; sort by key for the oracle comparison.
    let id_col = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id1 must be Int32");
    let sum_col = out
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("SUM(v1) must be Float64");

    let mut actual: Vec<(i32, f64)> = (0..out.num_rows())
        .map(|i| (id_col.value(i), sum_col.value(i)))
        .collect();
    actual.sort_by_key(|&(k, _)| k);

    // Compare against the CPU oracle with the same tolerance as the CPU
    // model-vs-naive tests above.
    let err = max_relative_error(&actual, &expected);
    assert!(
        err < REL_TOL,
        "GPU pipeline vs CPU tier-2 model: max rel err {err:e} exceeded {REL_TOL:e}"
    );
}
