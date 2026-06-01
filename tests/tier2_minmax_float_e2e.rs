// SPDX-License-Identifier: Apache-2.0

//! GPU e2e regression baseline for **high-cardinality Tier-2 GROUP BY
//! MIN / MAX over FLOAT values**.
//!
//! ## Why this file exists
//!
//! The Tier-2 float MIN/MAX path is served by two executors:
//!
//!   * `src/exec/groupby_tier2_minmax_float_exec.rs` — single Int32 key,
//!     reduces with `partition_reduce_kernel_minmax_float` (i32 key).
//!   * `src/exec/groupby_tier2_twokey_minmax_float_exec.rs` — two Int32
//!     keys packed into one i64, reduces with
//!     `partition_reduce_kernel_minmax_float_i64`.
//!
//! Both kernels do their atomic MIN/MAX via an intricate
//! `atom.shared.cas.b{32,64}` retry loop (PTX has no
//! `atom.shared.{min,max}.f{32,64}` on sm_70). That CAS loop is about to
//! be refactored, so this file pins the *current* behaviour: it runs the
//! real `Engine::sql` pipeline on a GPU host and compares the grouped
//! float MIN/MAX against a pure-CPU per-group reference. If a refactor
//! regresses the reduction, these tests fail.
//!
//! ## Exactness contract (unlike the SUM e2e)
//!
//! MIN and MAX *select* an actual input element — they never compute a
//! new value via arithmetic — so the reordering the partition pipeline
//! introduces cannot perturb the result. The selected min/max is bit-for-bit
//! one of the input f64s. We therefore assert **EXACT** f64 equality of the
//! per-group (min, max), not a relative tolerance. To stay robust against
//! the implementation-defined output row order we compare the
//! `(key, min, max)` **multiset** with exact float equality (keys sorted
//! ASC; floats compared by `to_bits()` so the comparison is total and a
//! stray `-0.0`/`+0.0` or NaN would surface rather than silently pass).
//!
//! ## NaN handling choice
//!
//! The baseline fixture contains **NO NaN**. Both float MIN/MAX executors
//! explicitly *defer* (return `None`, routing to the global-atomic / host
//! scalar path) whenever the value column contains a NaN — see the `F2`
//! guards in both exec files. So a NaN-bearing column would not even
//! exercise the Tier-2 CAS kernel this file is meant to pin. Keeping the
//! fixture all-finite means MIN/MAX are well-defined under the plain IEEE
//! `setp.lt/gt` the kernel uses, and the CPU reference (plain `<` / `>`
//! over finite f64) is an exact oracle. Values span positive and negative
//! magnitudes so MIN and MAX land on genuinely different elements.
//!
//! ## Value dtype: Float64 only
//!
//! Both Tier-2 float MIN/MAX executors accept **Float64 only** in v0
//! (`ArrowDataType::Float64 => ...; _ => return None`). A `Float32` value
//! column is explicitly rejected and would fall back off the Tier-2 path,
//! so this file does NOT build an f32 value column — there is no Tier-2 f32
//! MIN/MAX path to regression-test today. See the report / module docs in
//! `groupby_tier2_minmax_float_exec.rs` ("v0 supports Float64 only").
//!
//! All correctness tests are `#[ignore = "gpu:tier2"]`; the orchestrator
//! runs them on a GPU host.

mod common;
use common::Xorshift64Star;

use std::collections::HashMap;

// ---- CPU reference ----------------------------------------------------------

/// Per-group (min, max) over finite f64 values, keyed by a single i32.
///
/// Plain `<` / `>` selection — valid because the fixture is all-finite, so
/// there is no NaN to make the IEEE comparison non-total. Returns a vec of
/// `(key, min, max)` sorted by key ASC.
fn cpu_minmax_single(keys: &[i32], vals: &[f64]) -> Vec<(i32, f64, f64)> {
    assert_eq!(keys.len(), vals.len(), "keys/vals length mismatch");
    let mut table: HashMap<i32, (f64, f64)> = HashMap::with_capacity(keys.len().min(1 << 21));
    for i in 0..keys.len() {
        let v = vals[i];
        let e = table.entry(keys[i]).or_insert((v, v));
        if v < e.0 {
            e.0 = v;
        }
        if v > e.1 {
            e.1 = v;
        }
    }
    let mut flat: Vec<(i32, f64, f64)> =
        table.into_iter().map(|(k, (mn, mx))| (k, mn, mx)).collect();
    flat.sort_by_key(|&(k, _, _)| k);
    flat
}

/// Per-group (min, max) over finite f64 values, keyed by a packed (i32, i32)
/// pair. The pack must match the executor's host-side pack exactly
/// (`(a as u32 as u64) << 32 | (b as u32 as u64)`) so the sort order and key
/// identity line up. Returns `(k1, k2, min, max)` sorted by packed-i64 key
/// ASC (the order the two-key executor emits before unpacking).
fn cpu_minmax_two(
    k1: &[i32],
    k2: &[i32],
    vals: &[f64],
) -> Vec<(i32, i32, f64, f64)> {
    assert_eq!(k1.len(), k2.len(), "k1/k2 length mismatch");
    assert_eq!(k1.len(), vals.len(), "key/vals length mismatch");
    let mut table: HashMap<i64, (f64, f64)> = HashMap::with_capacity(k1.len().min(1 << 21));
    for i in 0..k1.len() {
        let packed = ((k1[i] as u32 as u64) << 32 | (k2[i] as u32 as u64)) as i64;
        let v = vals[i];
        let e = table.entry(packed).or_insert((v, v));
        if v < e.0 {
            e.0 = v;
        }
        if v > e.1 {
            e.1 = v;
        }
    }
    let mut flat: Vec<(i64, f64, f64)> =
        table.into_iter().map(|(k, (mn, mx))| (k, mn, mx)).collect();
    flat.sort_by_key(|&(k, _, _)| k);
    flat.into_iter()
        .map(|(packed, mn, mx)| {
            let u = packed as u64;
            (((u >> 32) as u32) as i32, ((u & 0xFFFF_FFFF) as u32) as i32, mn, mx)
        })
        .collect()
}

// ---- Exact float comparison helpers ----------------------------------------

/// Total, exact f64 equality via bit pattern. The fixture has no NaN so this
/// is just exact equality, but `to_bits` keeps the comparison total and would
/// surface an unexpected `-0.0`/NaN drift instead of letting `==` paper over
/// it.
#[allow(dead_code)]
fn f64_bits_eq(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits()
}

// ---- Fixture ----------------------------------------------------------------

/// High-cardinality single-key fixture: `id3` ∈ [0, `n_distinct`) drawn
/// uniformly (≈ `n_distinct` distinct groups once `n_rows >> n_distinct`),
/// and a `v1: f64` value spanning positive and negative magnitudes.
///
/// Keys are kept non-negative because the single-key float MIN/MAX executor
/// declines negative keys (`scan_max_nonneg_key`). `n_distinct` ≈ 1_000_000
/// puts `n_groups_est` (= max(id3)+1) comfortably inside the executor's
/// `(BLOCK_GROUPS, 100M)` Tier-2 window so the query takes the Tier-2 float
/// path rather than the low-card fallback.
///
/// Deterministic from `seed`. Values are scaled to ~[-1e6, 1e6) so MIN and
/// MAX per group are well separated and distinct from each other.
fn fixture_single(n_rows: usize, n_distinct: i32, seed: u64) -> (Vec<i32>, Vec<f64>) {
    assert!(n_distinct > 0, "n_distinct must be positive");
    let modulus = n_distinct as u64;
    let mut rng = Xorshift64Star::new(seed);
    let mut keys = Vec::with_capacity(n_rows);
    let mut vals = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let k = (rng.next_u64() % modulus) as i32;
        // Signed value in ~[-1e6, 1e6). next_signed_unit_f64() ∈ [-1, 1).
        let v = rng.next_signed_unit_f64() * 1.0e6;
        keys.push(k);
        vals.push(v);
    }
    (keys, vals)
}

/// High-cardinality two-key fixture: `id1` ∈ [0, `n1`), `id2` ∈ [0, `n2`),
/// independent uniform draws, so the combined cardinality approaches
/// `min(n_rows, n1 * n2)` distinct `(id1, id2)` pairs. With `n1 = 100`,
/// `n2 = 10_000` that is up to 1_000_000 pairs — high-card Tier-2 territory,
/// well under the 100M dispatcher cap. Both keys non-negative; value is the
/// same signed-f64 spread as the single-key fixture.
fn fixture_two(
    n_rows: usize,
    n1: i32,
    n2: i32,
    seed: u64,
) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    assert!(n1 > 0 && n2 > 0, "key moduli must be positive");
    let m1 = n1 as u64;
    let m2 = n2 as u64;
    let mut rng = Xorshift64Star::new(seed);
    let mut k1 = Vec::with_capacity(n_rows);
    let mut k2 = Vec::with_capacity(n_rows);
    let mut vals = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        k1.push((rng.next_u64() % m1) as i32);
        k2.push((rng.next_u64() % m2) as i32);
        vals.push(rng.next_signed_unit_f64() * 1.0e6);
    }
    (k1, k2, vals)
}

// ---- CPU-only sanity tests (no `#[ignore]`) --------------------------------
//
// These do not touch the GPU; they pin the reference + fixture so a failure
// here points at the oracle, not the kernel.

#[test]
fn fixtures_are_deterministic() {
    let (k1a, v1a) = fixture_single(100_000, 50_000, 0x51);
    let (k1b, v1b) = fixture_single(100_000, 50_000, 0x51);
    assert_eq!(k1a, k1b, "single-key keys must be deterministic");
    assert_eq!(v1a, v1b, "single-key vals must be deterministic");

    let (a1, a2, av) = fixture_two(100_000, 100, 10_000, 0x52);
    let (b1, b2, bv) = fixture_two(100_000, 100, 10_000, 0x52);
    assert_eq!(a1, b1);
    assert_eq!(a2, b2);
    assert_eq!(av, bv);
}

#[test]
fn cpu_reference_picks_actual_elements_single() {
    // Tiny hand-checkable case: two groups, known min/max.
    let keys = vec![7, 7, 7, 3, 3];
    let vals = vec![2.0, -5.0, 9.5, 100.0, -100.0];
    let r = cpu_minmax_single(&keys, &vals);
    assert_eq!(r, vec![(3, -100.0, 100.0), (7, -5.0, 9.5)]);
    // Min/max are actual inputs (no arithmetic): they appear in `vals`.
    for &(_, mn, mx) in &r {
        assert!(vals.iter().any(|&v| f64_bits_eq(v, mn)));
        assert!(vals.iter().any(|&v| f64_bits_eq(v, mx)));
    }
}

#[test]
fn cpu_reference_packs_two_key_like_executor() {
    let k1 = vec![1, 1, 2];
    let k2 = vec![5, 5, 9];
    let vals = vec![3.0, -3.0, 42.0];
    let r = cpu_minmax_two(&k1, &k2, &vals);
    // (1,5) → min -3, max 3 ; (2,9) → min/max 42. Packed (1,5) < (2,9).
    assert_eq!(r, vec![(1, 5, -3.0, 3.0), (2, 9, 42.0, 42.0)]);
}

// ---- GPU-gated e2e tests ----------------------------------------------------

/// SINGLE-KEY high-cardinality Tier-2 float MIN/MAX (i32 key).
///
/// Query: `SELECT id3, MIN(v1), MAX(v1) FROM x GROUP BY id3`
///
/// Routes through `groupby_tier2_minmax_float_exec` →
/// `partition_reduce_kernel_minmax_float` (i32 key, CAS-loop atomic).
/// Decoded column types: `id3` → Int32, `MIN(v1)`/`MAX(v1)` → Float64
/// (MIN/MAX preserve the input dtype). Asserts EXACT (key, min, max)
/// multiset equality vs the CPU reference.
///
/// Expectation on CURRENT code: PASS.
#[test]
#[ignore = "gpu:tier2"]
fn tier2_single_key_minmax_float_matches_cpu() {
    use std::sync::Arc;

    use arrow_array::{Array, Float64Array, Int32Array, RecordBatch};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    // ~2M rows, ~1M distinct id3 → high-card Tier-2.
    let n_rows: usize = 2_000_000;
    let n_distinct: i32 = 1_000_000;
    let (id3, v1) = fixture_single(n_rows, n_distinct, 0xF10A7);

    let expected = cpu_minmax_single(&id3, &v1);

    let id3_arr: Int32Array = id3.iter().copied().collect();
    let v1_arr: Float64Array = v1.iter().copied().collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id3", ArrowDataType::Int32, false),
        ArrowField::new("v1", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(id3_arr), Arc::new(v1_arr)])
        .expect("build single-key RecordBatch");

    let mut engine = craton_bolt::Engine::new().expect("CUDA engine");
    engine.register_table("x", batch).expect("register table");

    let h = engine
        .sql("SELECT id3, MIN(v1), MAX(v1) FROM x GROUP BY id3")
        .expect("execute single-key minmax-float groupby");
    let out = h.record_batch();

    // SELECT-ordered schema: [id3, min_v1, max_v1]. MIN/MAX preserve the f64
    // input dtype, so both aggregate columns decode as Float64.
    let id_col = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id3 must be Int32");
    let min_col = out
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("MIN(v1) must be Float64");
    let max_col = out
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("MAX(v1) must be Float64");

    assert_eq!(
        out.num_rows(),
        expected.len(),
        "distinct-group count mismatch: gpu={} cpu={}",
        out.num_rows(),
        expected.len()
    );

    let mut actual: Vec<(i32, f64, f64)> = (0..out.num_rows())
        .map(|i| (id_col.value(i), min_col.value(i), max_col.value(i)))
        .collect();
    actual.sort_by_key(|&(k, _, _)| k);

    // EXACT multiset comparison: MIN/MAX select an actual input element, so
    // bit-exact equality is the contract (no tolerance).
    for (i, (got, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(got.0, exp.0, "key mismatch at row {i}: gpu={} cpu={}", got.0, exp.0);
        assert!(
            f64_bits_eq(got.1, exp.1),
            "MIN mismatch for key {}: gpu={:?} cpu={:?}",
            got.0,
            got.1,
            exp.1
        );
        assert!(
            f64_bits_eq(got.2, exp.2),
            "MAX mismatch for key {}: gpu={:?} cpu={:?}",
            got.0,
            got.2,
            exp.2
        );
    }
}

/// TWO-KEY high-cardinality Tier-2 float MIN/MAX (i64 packed key).
///
/// Query: `SELECT id1, id2, MIN(v1), MAX(v1) FROM x GROUP BY id1, id2`
///
/// Routes through `groupby_tier2_twokey_minmax_float_exec` →
/// `partition_reduce_kernel_minmax_float_i64` (two i32 keys packed to one
/// i64, CAS-loop atomic). Decoded column types: `id1`,`id2` → Int32,
/// `MIN(v1)`/`MAX(v1)` → Float64. Asserts EXACT (k1, k2, min, max) multiset
/// equality vs the CPU reference.
///
/// Expectation on CURRENT code: PASS. The two-key float path IS wired into
/// the dispatcher (`groupby.rs` routes to
/// `groupby_tier2_twokey_minmax_float_exec::try_execute`) and accepts
/// Float64 value columns with ≥256K rows and <100M combined cardinality —
/// all satisfied here. If this variant ever STOPS taking the Tier-2 path
/// (e.g. a future deferral), the group count / values would still have to
/// match because the fallback path computes the same MIN/MAX; an actual
/// FAILURE here therefore flags a real kernel regression.
#[test]
#[ignore = "gpu:tier2"]
fn tier2_two_key_minmax_float_matches_cpu() {
    use std::sync::Arc;

    use arrow_array::{Array, Float64Array, Int32Array, RecordBatch};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    // ~2M rows; id1∈[0,100), id2∈[0,10_000) → up to 1M distinct pairs.
    let n_rows: usize = 2_000_000;
    let (id1, id2, v1) = fixture_two(n_rows, 100, 10_000, 0xF207E);

    let expected = cpu_minmax_two(&id1, &id2, &v1);

    let id1_arr: Int32Array = id1.iter().copied().collect();
    let id2_arr: Int32Array = id2.iter().copied().collect();
    let v1_arr: Float64Array = v1.iter().copied().collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id1", ArrowDataType::Int32, false),
        ArrowField::new("id2", ArrowDataType::Int32, false),
        ArrowField::new("v1", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(id1_arr), Arc::new(id2_arr), Arc::new(v1_arr)],
    )
    .expect("build two-key RecordBatch");

    let mut engine = craton_bolt::Engine::new().expect("CUDA engine");
    engine.register_table("x", batch).expect("register table");

    let h = engine
        .sql("SELECT id1, id2, MIN(v1), MAX(v1) FROM x GROUP BY id1, id2")
        .expect("execute two-key minmax-float groupby");
    let out = h.record_batch();

    // SELECT-ordered schema: [id1, id2, min_v1, max_v1].
    let k1_col = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id1 must be Int32");
    let k2_col = out
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id2 must be Int32");
    let min_col = out
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("MIN(v1) must be Float64");
    let max_col = out
        .column(3)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("MAX(v1) must be Float64");

    assert_eq!(
        out.num_rows(),
        expected.len(),
        "distinct-pair count mismatch: gpu={} cpu={}",
        out.num_rows(),
        expected.len()
    );

    // Sort actual by the same packed-i64 key order the reference used.
    let mut actual: Vec<(i32, i32, f64, f64)> = (0..out.num_rows())
        .map(|i| (k1_col.value(i), k2_col.value(i), min_col.value(i), max_col.value(i)))
        .collect();
    actual.sort_by_key(|&(a, b, _, _)| ((a as u32 as u64) << 32 | (b as u32 as u64)) as i64);

    for (i, (got, exp)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(
            (got.0, got.1),
            (exp.0, exp.1),
            "key pair mismatch at row {i}: gpu=({},{}) cpu=({},{})",
            got.0,
            got.1,
            exp.0,
            exp.1
        );
        assert!(
            f64_bits_eq(got.2, exp.2),
            "MIN mismatch for ({}, {}): gpu={:?} cpu={:?}",
            got.0,
            got.1,
            got.2,
            exp.2
        );
        assert!(
            f64_bits_eq(got.3, exp.3),
            "MAX mismatch for ({}, {}): gpu={:?} cpu={:?}",
            got.0,
            got.1,
            got.3,
            exp.3
        );
    }
}
