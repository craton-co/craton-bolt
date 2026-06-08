// SPDX-License-Identifier: Apache-2.0

//! Test scaffolding for Tier 1 of the GROUP BY perf plan: a per-block
//! shared-memory pre-aggregation kernel.
//!
//! The new kernel re-orders the float adds (it sums per-block first, then
//! reduces block partials), so bit-exact equality with the old single-pass
//! kernel is not realistic; numerical *equivalence within a tight relative
//! tolerance* is. This file builds the oracle for that check:
//!
//! 1. `cpu_shmem_sum_model` — a pure-Rust mirror of the kernel's add order.
//!    This is the function the GPU output is compared against.
//! 2. `cpu_naive_sum` — the obvious single-pass reference. Used here to
//!    cross-validate that the reordered model is itself correct.
//! 3. Deterministic fixture builders.
//! 4. CPU-only unit tests (no `#[ignore]`) that prove the model matches the
//!    naive reference across the full cardinality envelope of Tier 1.
//! 5. A `#[ignore]`'d integration test (`shmem_kernel_matches_cpu_model`) that
//!    is a regression hook for the kernel landing — fill in the body and
//!    un-ignore once a sibling worktree merges the kernel.
//!
//! Algorithm context: see `docs/GROUPBY_PERF.md` Tier 1.

mod common;
use common::{Xorshift64Star, REL_TOL};

// ---- CPU references ---------------------------------------------------------

/// CPU model of the per-block shared-mem pre-aggregation kernel — used to
/// generate expected outputs the GPU kernel must match within REL_TOL.
///
/// Mirrors the algorithm sketched in docs/GROUPBY_PERF.md Tier 1: bin rows
/// into block-slices, sum each slice's contribution per-key into a block-
/// local accumulator, then sum those block accumulators into the final
/// per-group result. Float reordering matches what the GPU does; result is
/// numerically *closer* to the GPU's output than the naive single-pass
/// reference, so this is the right oracle for a tight tolerance test.
///
/// Grid-stride layout (matching the GPU): thread `t` in block `b` touches
/// row indices
///   `b * block_threads * rows_per_thread + t`,
///   `b * block_threads * rows_per_thread + t + block_threads`,
///   ...
/// up to `rows_per_thread` rows per thread before the next block starts.
/// Rows past `keys.len()` are simply ignored (tail).
fn cpu_shmem_sum_model(
    keys: &[i32],
    vals: &[f64],
    n_groups: u32,
    block_threads: usize,
    rows_per_thread: usize,
) -> Vec<f64> {
    assert_eq!(keys.len(), vals.len(), "keys/vals length mismatch");
    assert!(block_threads > 0, "block_threads must be positive");
    assert!(rows_per_thread > 0, "rows_per_thread must be positive");

    let n_rows = keys.len();
    let n_groups_usize = n_groups as usize;
    let rows_per_block = block_threads * rows_per_thread;
    if n_rows == 0 || n_groups_usize == 0 {
        return vec![0.0; n_groups_usize];
    }
    let n_blocks = (n_rows + rows_per_block - 1) / rows_per_block;

    // Final result; we fold block-local partials into this.
    let mut result = vec![0.0_f64; n_groups_usize];

    // One reusable block-partial buffer. Cleared at the top of each block so
    // we avoid a fresh allocation per block on the hot path.
    let mut block_partial = vec![0.0_f64; n_groups_usize];

    for b in 0..n_blocks {
        // Reset block-partial.
        for slot in block_partial.iter_mut() {
            *slot = 0.0;
        }

        let block_base = b * rows_per_block;
        // Process rows in grid-stride order to mirror the GPU's add ordering.
        for stride_step in 0..rows_per_thread {
            for t in 0..block_threads {
                let row = block_base + stride_step * block_threads + t;
                if row >= n_rows {
                    continue;
                }
                let k = keys[row];
                if k < 0 || (k as u32) >= n_groups {
                    // Out-of-range key — kernel will mask these; the oracle does the same.
                    continue;
                }
                let v = vals[row];
                block_partial[k as usize] += v;
            }
        }

        // Fold this block's partial into the global result.
        for g in 0..n_groups_usize {
            result[g] += block_partial[g];
        }
    }

    result
}

/// Single-pass naive SUM-by-key. Used to verify that `cpu_shmem_sum_model`
/// agrees with the obvious reference within tight tolerance — i.e. the
/// fancy reordered model is itself correct.
fn cpu_naive_sum(keys: &[i32], vals: &[f64], n_groups: u32) -> Vec<f64> {
    assert_eq!(keys.len(), vals.len(), "keys/vals length mismatch");
    let mut out = vec![0.0_f64; n_groups as usize];
    for i in 0..keys.len() {
        let k = keys[i];
        if k < 0 || (k as u32) >= n_groups {
            continue;
        }
        out[k as usize] += vals[i];
    }
    out
}

// ---- Fixture ----------------------------------------------------------------

/// Generate `(keys, vals)` for the unit tests. Deterministic from a seed so
/// tests are reproducible across runs and across the four sibling worktrees.
///
/// Uses the shared `Xorshift64Star` PRNG from `tests/common/mod.rs` — keeps
/// the test self-contained with no extra dev-deps. Keys are spread roughly
/// uniformly across `[0, n_groups)`; values are in `[-1.0, 1.0)` so SUMs
/// across 10M rows stay in a numerically interesting (non-degenerate) range
/// and exercise float reordering sensitivity.
fn fixture(n_rows: usize, n_groups: u32, seed: u64) -> (Vec<i32>, Vec<f64>) {
    assert!(n_groups > 0, "n_groups must be positive");
    let mut rng = Xorshift64Star::new(seed);

    let mut keys = Vec::with_capacity(n_rows);
    let mut vals = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let k = (rng.next_u64() % n_groups as u64) as i32;
        let v = rng.next_signed_unit_f64();
        keys.push(k);
        vals.push(v);
    }
    (keys, vals)
}

// ---- Helpers ----------------------------------------------------------------

/// Max relative error between two equal-length result vectors, with an
/// absolute floor to keep near-zero comparisons sane.
fn max_relative_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "result length mismatch");
    let mut worst = 0.0_f64;
    for i in 0..a.len() {
        let x = a[i];
        let y = b[i];
        let denom = x.abs().max(y.abs()).max(1.0);
        let rel = (x - y).abs() / denom;
        if rel > worst {
            worst = rel;
        }
    }
    worst
}

// ---- CPU unit tests (no `#[ignore]`) ---------------------------------------

const BLOCK_THREADS: usize = 256;
const ROWS_PER_THREAD: usize = 4;

#[test]
fn model_agrees_with_naive_small() {
    let (keys, vals) = fixture(1024, 10, 0xA1);
    let model = cpu_shmem_sum_model(&keys, &vals, 10, BLOCK_THREADS, ROWS_PER_THREAD);
    let naive = cpu_naive_sum(&keys, &vals, 10);
    let err = max_relative_error(&model, &naive);
    assert!(err < 1e-10, "max rel err {err:e} exceeded 1e-10");
}

#[test]
fn model_agrees_with_naive_low_card_10m() {
    let (keys, vals) = fixture(10_000_000, 100, 0xB2);
    let model = cpu_shmem_sum_model(&keys, &vals, 100, BLOCK_THREADS, ROWS_PER_THREAD);
    let naive = cpu_naive_sum(&keys, &vals, 100);
    let err = max_relative_error(&model, &naive);
    // Empirical max relative error on the current fixture: ~2.5e-13.
    assert!(err < REL_TOL, "max rel err {err:e} exceeded {REL_TOL:e}");
}

#[test]
fn model_agrees_with_naive_med_card() {
    let (keys, vals) = fixture(10_000_000, 1000, 0xC3);
    let model = cpu_shmem_sum_model(&keys, &vals, 1000, BLOCK_THREADS, ROWS_PER_THREAD);
    let naive = cpu_naive_sum(&keys, &vals, 1000);
    let err = max_relative_error(&model, &naive);
    assert!(err < REL_TOL, "max rel err {err:e} exceeded {REL_TOL:e}");
}

#[test]
fn model_agrees_with_naive_at_block_groups_limit() {
    // n_groups exactly 1024 — the shared-mem cap for Tier 1. Cardinality at
    // the boundary stresses the per-block accumulator sizing.
    let (keys, vals) = fixture(2_000_000, 1024, 0xD4);
    let model = cpu_shmem_sum_model(&keys, &vals, 1024, BLOCK_THREADS, ROWS_PER_THREAD);
    let naive = cpu_naive_sum(&keys, &vals, 1024);
    let err = max_relative_error(&model, &naive);
    assert!(err < REL_TOL, "max rel err {err:e} exceeded {REL_TOL:e}");
}

#[test]
fn model_handles_n_groups_lt_block_groups() {
    // Tiny cardinality (n_groups=5) — each block's partial is densely packed.
    let (keys, vals) = fixture(10_000, 5, 0xE5);
    let model = cpu_shmem_sum_model(&keys, &vals, 5, BLOCK_THREADS, ROWS_PER_THREAD);
    let naive = cpu_naive_sum(&keys, &vals, 5);
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

// ---- GPU-gated integration test --------------------------------------------
//
// This test is the regression hook for the Tier 1 kernel landing. Once the
// sibling worktree merges the kernel + dispatch heuristic, drop the
// `#[ignore]` and finish the body. The intent is documented inline so the
// next agent can fill it in mechanically.

#[test]
#[ignore = "gpu:tier1"]
fn shmem_kernel_matches_cpu_model() {
    use std::sync::Arc;

    use arrow_array::{Array, Float64Array, Int32Array, RecordBatch};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    // Build a 10M-row fixture with n_groups=100. Same seed as the CPU test
    // above so debugging mismatches is straightforward.
    let n_rows: usize = 10_000_000;
    let n_groups: u32 = 100;
    let (keys, vals) = fixture(n_rows, n_groups, 0xB2);

    // Expected per-group result, computed with the reordered CPU oracle that
    // mirrors the kernel's float-add order. Use the same launch params as
    // the live kernel will pick — these are the Tier 1 defaults; if the
    // launch tuner sibling lands different defaults, update both sides.
    let expected = cpu_shmem_sum_model(&keys, &vals, n_groups, BLOCK_THREADS, ROWS_PER_THREAD);

    // Build a RecordBatch with columns `id1` (Int32) and `v1` (Float64).
    let id1: Int32Array = keys.iter().copied().collect();
    let v1: Float64Array = vals.iter().copied().collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id1", ArrowDataType::Int32, false),
        ArrowField::new("v1", ArrowDataType::Float64, false),
    ]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(id1), Arc::new(v1)]).expect("build RecordBatch");

    // Stand up the engine on the default CUDA device. Mirrors the convention
    // in `tests/memory_tests.rs`: `.expect()` is fine because the test is
    // `#[ignore]`'d, so it only runs when the operator explicitly opted in
    // with `-- --ignored` on a GPU host.
    let mut engine = craton_bolt::Engine::new().expect("CUDA engine");
    engine.register_table("x", batch).expect("register table");

    let h = engine
        .sql("SELECT id1, SUM(v1) FROM x GROUP BY id1")
        .expect("execute groupby");
    let out = h.record_batch();

    // The output schema is SELECT-ordered: [id1, sum_v1].
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

    let n_out = out.num_rows();
    assert_eq!(
        n_out, n_groups as usize,
        "expected {} output rows (one per group), got {}",
        n_groups, n_out
    );

    // For every output row, look up the corresponding expected sum by key
    // and check relative error. Use the same REL_TOL contract as the CPU
    // model tests above (shared `tests/common::REL_TOL`).
    for i in 0..n_out {
        let key = id_col.value(i);
        assert!(
            key >= 0 && (key as u32) < n_groups,
            "row {i}: key {key} out of range [0, {n_groups})"
        );
        let got = sum_col.value(i);
        let want = expected[key as usize];
        let denom = got.abs().max(want.abs()).max(1.0);
        let rel = (got - want).abs() / denom;
        assert!(
            rel < REL_TOL,
            "row {i} key={key}: got {got}, want {want}, rel err {rel:e} exceeded {REL_TOL:e}"
        );
    }
}
