// SPDX-License-Identifier: Apache-2.0

//! GPU e2e regression baseline for **high-cardinality Tier-2 GROUP BY
//! MIN / MAX over INTEGER values**.
//!
//! This file is the regression net for an imminent refactor of the Tier-2
//! integer MIN/MAX reduce kernels:
//!
//!   * single-key i32  -> `partition_reduce_kernel_minmax`
//!   * two-key  i64    -> `partition_reduce_kernel_minmax_i64` (via the
//!                        host-side `(k1<<32)|(k2&0xFFFF_FFFF)` two-key pack)
//!
//! No existing GPU e2e exercises these paths over a *high-cardinality*
//! INTEGER value column, so the refactor would otherwise land uncovered.
//! Each integration test below builds a deterministic fixture, runs the
//! query through the real `Engine` (which dispatches to the GPU Tier-2
//! executors), computes a pure-CPU naive `(min, max)`-per-group oracle, and
//! asserts EXACT integer equality of the sorted result multiset. MIN/MAX are
//! order-independent and lossless, so there is no float tolerance here —
//! any drift is a real bug.
//!
//! ## What pins the GPU path (verified against the executors on `dev`)
//!
//! Single-key i32 minmax (`groupby_tier2_minmax_exec::try_execute`):
//!   * exactly one Int32 group-by column, exactly one `MIN`/`MAX` of a bare
//!     column, value dtype Int32 **or** Int64, no NULLs;
//!   * `n_rows >= 256 * 1024`;
//!   * the key is treated as DENSE: `n_groups_est = max_key + 1` must be
//!     `> BLOCK_GROUPS (1024)` and `< 100_000_000`, and the key must be
//!     **non-negative** (a negative key declines the fast path). Hence the
//!     NEGATIVE-values requirement is satisfied via the VALUE column, never
//!     the key.
//!
//! Two-key i64 minmax (`groupby_tier2_twokey_minmax_exec::try_execute`):
//!   * exactly two Int32 group-by columns, one `MIN`/`MAX` of a bare column,
//!     value dtype Int32 **or** Int64, no NULLs;
//!   * `n_rows >= 256 * 1024` and `< 100_000_000`.
//!
//! Output dtype: MIN/MAX PRESERVE the input value dtype. The fixtures here
//! use an `Int64` value column, so the aggregate result column decodes as
//! `Int64Array` in every variant (verified: the i64 reduce phase builds an
//! `Int64Array`, the schema comes from `plan_schema_to_arrow_schema`).
//!
//! Both group keys are non-negative, so the packed-i64 sort order used by the
//! two-key executor coincides with lexical `(id1, id2)` order — the oracle
//! sorts on the same `(id1, id2)` tuple, so the comparison is apples-to-apples.

mod common;
use common::Xorshift64Star;

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

// ---- Fixture ---------------------------------------------------------------

/// Columns for the high-cardinality integer MIN/MAX fixture, h2o-shaped:
///   * `id1`  — Int32, ~100 distinct          (two-key high bits)
///   * `id2`  — Int32, ~10_000 distinct        (two-key low bits)
///   * `id3`  — Int32, ~1_000_000 distinct      (single-key, dense in [0, N))
///   * `ival` — Int64, signed spread            (the MIN/MAX target)
///
/// `id3` is drawn from `[0, n_id3)` so the dense `max_key + 1` cardinality
/// estimate the executor uses lands well above `BLOCK_GROUPS` and below the
/// 100M cap. `ival` spans a wide signed range (positive AND negative, with
/// magnitudes beyond the f64 mantissa boundary on some rows) so MIN and MAX
/// are distinct, meaningful, and would expose a signed-comparison bug.
struct Fixture {
    id1: Vec<i32>,
    id2: Vec<i32>,
    id3: Vec<i32>,
    ival: Vec<i64>,
}

/// Deterministic from `seed` so the GPU result and the CPU oracle are built
/// from identical inputs and reproduce across runs.
fn fixture(n_rows: usize, n_id1: i32, n_id2: i32, n_id3: i32, seed: u64) -> Fixture {
    assert!(n_id1 > 0 && n_id2 > 0 && n_id3 > 0, "cardinalities must be positive");
    let mut rng = Xorshift64Star::new(seed);

    let mut id1 = Vec::with_capacity(n_rows);
    let mut id2 = Vec::with_capacity(n_rows);
    let mut id3 = Vec::with_capacity(n_rows);
    let mut ival = Vec::with_capacity(n_rows);

    for _ in 0..n_rows {
        id1.push((rng.next_u64() % n_id1 as u64) as i32);
        id2.push((rng.next_u64() % n_id2 as u64) as i32);
        id3.push((rng.next_u64() % n_id3 as u64) as i32);

        // Signed value with a wide spread. Take a 33-bit magnitude (so some
        // |v| exceed 2^31 and a few approach 2^32, well within i64 but past
        // the i32 range) and a sign bit. This guarantees both signs occur and
        // that MIN lands clearly negative while MAX lands clearly positive.
        let mag = (rng.next_u64() & 0x1_FFFF_FFFF) as i64; // [0, 2^33)
        let v = if rng.next_u64() & 1 == 0 { mag } else { -mag - 1 };
        ival.push(v);
    }

    Fixture { id1, id2, id3, ival }
}

// ---- CPU naive references --------------------------------------------------

/// Naive single-key `(min, max)` per group over integer values. Sorted by key
/// ASC for a deterministic, index-comparable result.
fn cpu_minmax_one_key(keys: &[i32], vals: &[i64]) -> Vec<(i32, i64, i64)> {
    assert_eq!(keys.len(), vals.len(), "keys/vals length mismatch");
    let mut table: HashMap<i32, (i64, i64)> = HashMap::with_capacity(keys.len().min(1 << 20));
    for i in 0..keys.len() {
        let e = table.entry(keys[i]).or_insert((vals[i], vals[i]));
        if vals[i] < e.0 {
            e.0 = vals[i];
        }
        if vals[i] > e.1 {
            e.1 = vals[i];
        }
    }
    let mut flat: Vec<(i32, i64, i64)> =
        table.into_iter().map(|(k, (mn, mx))| (k, mn, mx)).collect();
    flat.sort_by_key(|&(k, _, _)| k);
    flat
}

/// Naive two-key `(min, max)` per `(k1, k2)` group. Sorted by `(k1, k2)` ASC.
fn cpu_minmax_two_key(k1: &[i32], k2: &[i32], vals: &[i64]) -> Vec<(i32, i32, i64, i64)> {
    assert_eq!(k1.len(), k2.len(), "k1/k2 length mismatch");
    assert_eq!(k1.len(), vals.len(), "keys/vals length mismatch");
    let mut table: HashMap<(i32, i32), (i64, i64)> =
        HashMap::with_capacity(k1.len().min(1 << 20));
    for i in 0..k1.len() {
        let e = table.entry((k1[i], k2[i])).or_insert((vals[i], vals[i]));
        if vals[i] < e.0 {
            e.0 = vals[i];
        }
        if vals[i] > e.1 {
            e.1 = vals[i];
        }
    }
    let mut flat: Vec<(i32, i32, i64, i64)> = table
        .into_iter()
        .map(|((a, b), (mn, mx))| (a, b, mn, mx))
        .collect();
    flat.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)));
    flat
}

// ---- Decode helpers --------------------------------------------------------

/// Downcast a column to `Int32Array`, with a descriptive panic on dtype miss.
fn i32_col<'a>(batch: &'a RecordBatch, idx: usize, what: &str) -> &'a Int32Array {
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap_or_else(|| {
            panic!(
                "{what}: expected Int32 at col {idx}, got {:?}",
                batch.column(idx).data_type()
            )
        })
}

/// Downcast a column to `Int64Array`. MIN/MAX over an Int64 value column must
/// preserve the Int64 dtype — this is the assertion that verifies it.
fn i64_col<'a>(batch: &'a RecordBatch, idx: usize, what: &str) -> &'a Int64Array {
    batch
        .column(idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap_or_else(|| {
            panic!(
                "{what}: expected Int64 at col {idx}, got {:?}",
                batch.column(idx).data_type()
            )
        })
}

// ---- GPU-gated integration tests -------------------------------------------

/// SINGLE-KEY high-cardinality MIN+MAX over Int64 values.
///
/// Query: `SELECT id3, MIN(ival), MAX(ival) FROM x GROUP BY id3`.
///
/// NOTE: the single-key integer-minmax executor handles exactly ONE aggregate
/// per query (see `groupby_tier2_minmax_exec` module docs: "Single-aggregate
/// only"). Asking for `MIN(ival), MAX(ival)` together in one statement would
/// decline the Tier-2 fast path and fall through to the global-atomic
/// baseline — still correct, but it would NOT exercise the kernels under
/// refactor. So we issue MIN and MAX as two separate single-aggregate queries
/// (each pins the Tier-2 i32 path) and stitch the results together for the
/// exact-equality check against the CPU oracle.
///
/// Decoded columns: `id3` -> Int32Array, `MIN/MAX(ival)` -> Int64Array
/// (dtype preserved from the Int64 input value column).
#[test]
#[ignore = "gpu:tier2"]
fn tier2_single_key_minmax_i32_int_values() {
    // ~2M rows, id3 dense in [0, 1_000_000): max_key+1 ~ 1e6, comfortably in
    // the (1024, 100_000_000) Tier-2 single-key window.
    let n_rows: usize = 2_000_000;
    let f = fixture(n_rows, 100, 10_000, 1_000_000, 0x5117);

    // CPU oracle: (key, min, max) sorted by key.
    let expected = cpu_minmax_one_key(&f.id3, &f.ival);
    // Sanity: high cardinality and a genuine signed spread, so the variant is
    // actually meaningful (not a degenerate all-positive / single-group case).
    assert!(expected.len() > 1024, "fixture must exceed BLOCK_GROUPS distinct keys");
    assert!(
        expected.iter().any(|&(_, mn, _)| mn < 0),
        "expected at least one negative per-group MIN"
    );
    assert!(
        expected.iter().any(|&(_, _, mx)| mx > 0),
        "expected at least one positive per-group MAX"
    );

    let batch = single_key_batch(&f);

    let mut engine = craton_bolt::Engine::new().expect("CUDA engine");
    engine.register_table("x", batch).expect("register table");

    // --- MIN(ival) ---
    let min_h = engine
        .sql("SELECT id3, MIN(ival) FROM x GROUP BY id3")
        .expect("execute single-key MIN");
    let min_out = min_h.record_batch();
    let min_keys = i32_col(min_out, 0, "single-key MIN: id3");
    let min_vals = i64_col(min_out, 1, "single-key MIN: MIN(ival) dtype");
    let mut min_actual: Vec<(i32, i64)> = (0..min_out.num_rows())
        .map(|i| (min_keys.value(i), min_vals.value(i)))
        .collect();
    min_actual.sort_by_key(|&(k, _)| k);

    // --- MAX(ival) ---
    let max_h = engine
        .sql("SELECT id3, MAX(ival) FROM x GROUP BY id3")
        .expect("execute single-key MAX");
    let max_out = max_h.record_batch();
    let max_keys = i32_col(max_out, 0, "single-key MAX: id3");
    let max_vals = i64_col(max_out, 1, "single-key MAX: MAX(ival) dtype");
    let mut max_actual: Vec<(i32, i64)> = (0..max_out.num_rows())
        .map(|i| (max_keys.value(i), max_vals.value(i)))
        .collect();
    max_actual.sort_by_key(|&(k, _)| k);

    // Exact equality of the (key, min) and (key, max) multisets.
    assert_eq!(
        min_actual.len(),
        expected.len(),
        "single-key MIN row count: gpu={} cpu={}",
        min_actual.len(),
        expected.len()
    );
    assert_eq!(
        max_actual.len(),
        expected.len(),
        "single-key MAX row count: gpu={} cpu={}",
        max_actual.len(),
        expected.len()
    );
    for (i, &(k, mn, mx)) in expected.iter().enumerate() {
        assert_eq!(min_actual[i].0, k, "single-key MIN key mismatch at {i}");
        assert_eq!(
            min_actual[i].1, mn,
            "single-key MIN value mismatch for key {k}: gpu={} cpu={}",
            min_actual[i].1, mn
        );
        assert_eq!(max_actual[i].0, k, "single-key MAX key mismatch at {i}");
        assert_eq!(
            max_actual[i].1, mx,
            "single-key MAX value mismatch for key {k}: gpu={} cpu={}",
            max_actual[i].1, mx
        );
    }
}

/// TWO-KEY high-cardinality MIN+MAX over Int64 values (i64-packed path).
///
/// Query: `SELECT id1, id2, MIN(ival), MAX(ival) FROM x GROUP BY id1, id2`.
///
/// Same single-aggregate constraint as the single-key path, so MIN and MAX
/// are issued separately to keep each on the Tier-2 two-key kernel under
/// refactor.
///
/// Decoded columns: `id1`,`id2` -> Int32Array, `MIN/MAX(ival)` -> Int64Array.
#[test]
#[ignore = "gpu:tier2"]
fn tier2_two_key_minmax_i64_int_values() {
    // ~2M rows, id1 ~100 x id2 ~10_000 nominal => up to ~1M combined groups,
    // birthday-collisioned down to a few hundred K distinct (id1,id2) pairs.
    let n_rows: usize = 2_000_000;
    let f = fixture(n_rows, 100, 10_000, 1_000_000, 0x7C0D);

    let expected = cpu_minmax_two_key(&f.id1, &f.id2, &f.ival);
    assert!(expected.len() > 1024, "fixture must produce many (id1,id2) groups");
    assert!(
        expected.iter().any(|&(_, _, mn, _)| mn < 0),
        "expected at least one negative per-group MIN"
    );
    assert!(
        expected.iter().any(|&(_, _, _, mx)| mx > 0),
        "expected at least one positive per-group MAX"
    );

    let batch = two_key_batch(&f);

    let mut engine = craton_bolt::Engine::new().expect("CUDA engine");
    engine.register_table("x", batch).expect("register table");

    // --- MIN(ival) ---
    let min_h = engine
        .sql("SELECT id1, id2, MIN(ival) FROM x GROUP BY id1, id2")
        .expect("execute two-key MIN");
    let min_out = min_h.record_batch();
    let min_k1 = i32_col(min_out, 0, "two-key MIN: id1");
    let min_k2 = i32_col(min_out, 1, "two-key MIN: id2");
    let min_vals = i64_col(min_out, 2, "two-key MIN: MIN(ival) dtype");
    let mut min_actual: Vec<(i32, i32, i64)> = (0..min_out.num_rows())
        .map(|i| (min_k1.value(i), min_k2.value(i), min_vals.value(i)))
        .collect();
    min_actual.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    // --- MAX(ival) ---
    let max_h = engine
        .sql("SELECT id1, id2, MAX(ival) FROM x GROUP BY id1, id2")
        .expect("execute two-key MAX");
    let max_out = max_h.record_batch();
    let max_k1 = i32_col(max_out, 0, "two-key MAX: id1");
    let max_k2 = i32_col(max_out, 1, "two-key MAX: id2");
    let max_vals = i64_col(max_out, 2, "two-key MAX: MAX(ival) dtype");
    let mut max_actual: Vec<(i32, i32, i64)> = (0..max_out.num_rows())
        .map(|i| (max_k1.value(i), max_k2.value(i), max_vals.value(i)))
        .collect();
    max_actual.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));

    assert_eq!(
        min_actual.len(),
        expected.len(),
        "two-key MIN row count: gpu={} cpu={}",
        min_actual.len(),
        expected.len()
    );
    assert_eq!(
        max_actual.len(),
        expected.len(),
        "two-key MAX row count: gpu={} cpu={}",
        max_actual.len(),
        expected.len()
    );
    for (i, &(k1, k2, mn, mx)) in expected.iter().enumerate() {
        assert_eq!((min_actual[i].0, min_actual[i].1), (k1, k2), "two-key MIN key mismatch at {i}");
        assert_eq!(
            min_actual[i].2, mn,
            "two-key MIN value mismatch for ({k1},{k2}): gpu={} cpu={}",
            min_actual[i].2, mn
        );
        assert_eq!((max_actual[i].0, max_actual[i].1), (k1, k2), "two-key MAX key mismatch at {i}");
        assert_eq!(
            max_actual[i].2, mx,
            "two-key MAX value mismatch for ({k1},{k2}): gpu={} cpu={}",
            max_actual[i].2, mx
        );
    }
}

// ---- Batch builders --------------------------------------------------------

fn single_key_batch(f: &Fixture) -> RecordBatch {
    let id3: Int32Array = f.id3.iter().copied().collect();
    let ival: Int64Array = f.ival.iter().copied().collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id3", ArrowDataType::Int32, false),
        ArrowField::new("ival", ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(id3), Arc::new(ival)])
        .expect("build single-key RecordBatch")
}

fn two_key_batch(f: &Fixture) -> RecordBatch {
    let id1: Int32Array = f.id1.iter().copied().collect();
    let id2: Int32Array = f.id2.iter().copied().collect();
    let ival: Int64Array = f.ival.iter().copied().collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id1", ArrowDataType::Int32, false),
        ArrowField::new("id2", ArrowDataType::Int32, false),
        ArrowField::new("ival", ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(id1), Arc::new(id2), Arc::new(ival)])
        .expect("build two-key RecordBatch")
}

// ---- CPU-only self-check (no GPU) ------------------------------------------
//
// Not `#[ignore]`'d: runs on every `cargo test` to guard the oracle itself
// (the part that does NOT need a GPU). If the naive references regress, this
// fails loudly on CI without a GPU host, independent of the gated e2e tests.

#[test]
fn oracle_self_check_minmax_small() {
    // Hand-checkable: keys {0,1}, values straddling zero.
    let keys = vec![0_i32, 0, 1, 1, 0];
    let vals = vec![5_i64, -3, 7, 7, -10];
    let one = cpu_minmax_one_key(&keys, &vals);
    assert_eq!(one, vec![(0, -10, 5), (1, 7, 7)]);

    let k1 = vec![0_i32, 0, 0, 1];
    let k2 = vec![0_i32, 0, 1, 0];
    let v = vec![4_i64, -8, 2, -1];
    let two = cpu_minmax_two_key(&k1, &k2, &v);
    assert_eq!(two, vec![(0, 0, -8, 4), (0, 1, 2, 2), (1, 0, -1, -1)]);

    // Determinism of the fixture.
    let a = fixture(10_000, 8, 64, 512, 0xABCD);
    let b = fixture(10_000, 8, 64, 512, 0xABCD);
    assert_eq!(a.id1, b.id1);
    assert_eq!(a.id3, b.id3);
    assert_eq!(a.ival, b.ival);
    // Signed spread really does straddle zero.
    assert!(a.ival.iter().any(|&v| v < 0) && a.ival.iter().any(|&v| v >= 0));
}
