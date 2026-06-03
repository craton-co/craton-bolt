// SPDX-License-Identifier: Apache-2.0

//! GPU e2e regression coverage for the **Tier-2 high-cardinality GROUP BY
//! COUNT** path.
//!
//! There is already GPU e2e coverage for Tier-2 GROUP BY *SUM*
//! (`tests/tier2_groupby_e2e.rs`), but none for COUNT. The Tier-2 COUNT path
//! dispatches through two distinct kernels depending on the group-key width:
//!
//!   * single i32 key  -> `partition_reduce_kernel_count`
//!   * two keys packed into one i64 -> `partition_reduce_kernel_count_i64`
//!
//! Both kernels are about to be refactored, so this file is the regression
//! baseline that must pass on the *current* code (run on a GPU host by the
//! orchestrator). If a variant is actually broken today, the assertion message
//! is written to make the failure self-explanatory.
//!
//! Unlike SUM, COUNT is exact: it counts rows (integers), so there is no
//! float-reordering tolerance to budget for. We therefore assert **exact**
//! equality of the multiset of `(key, count)` pairs against a pure-Rust CPU
//! reference, sorting both sides by key first so the implementation-defined GPU
//! emission order does not leak into the comparison.
//!
//! Cardinality envelope (mirrors the SUM file's `fixture(...)` shape, scaled to
//! trip Tier-2 dispatch):
//!   * `id3` — ~1,000,000 distinct values over ~2,000,000 rows -> the i32
//!     single-key count path.
//!   * `id1` (~100) x `id2` (~10,000) -> up to ~1,000,000 distinct
//!     (id1, id2) composite groups -> the two-key (i64-packed) count path.
//!
//! COUNT output column type: the engine types `COUNT(...)` as Arrow `Int64`
//! (verified against `tests/aggregate_nulls_e2e.rs`, which downcasts every
//! COUNT result to `Int64Array`). Group-key columns preserve their input Arrow
//! type, so `id1/id2/id3` come back as `Int32`.

mod common;
use common::Xorshift64Star;

use std::collections::HashMap;

// ---- Fixture ----------------------------------------------------------------

/// Deterministic h2o-shaped fixture. Mirrors the `fixture(...)` convention in
/// `tests/tier2_groupby_e2e.rs` (same `Xorshift64Star`, no extra dev-deps) but
/// emits the columns the COUNT paths need:
///
///   * `id1`  — low cardinality  (`[0, n_id1)`),  drives one half of the
///              two-key composite group.
///   * `id2`  — medium cardinality (`[0, n_id2)`), the other half.
///   * `id3`  — high cardinality  (`[0, n_id3)`), the single i32 key.
///   * `v1`   — a non-null value column. COUNT(v1) therefore equals the row
///              count per group (no NULLs to exclude), which keeps the CPU
///              reference a pure row-count and makes COUNT(v1) == COUNT(*).
///
/// Values are returned as parallel vectors so the test can both build the Arrow
/// batch and feed the CPU reference from the exact same data.
struct Fixture {
    id1: Vec<i32>,
    id2: Vec<i32>,
    id3: Vec<i32>,
    v1: Vec<f64>,
}

fn fixture(n_rows: usize, n_id1: i32, n_id2: i32, n_id3: i32, seed: u64) -> Fixture {
    assert!(
        n_id1 > 0 && n_id2 > 0 && n_id3 > 0,
        "cardinalities must be positive"
    );
    let m1 = n_id1 as u64;
    let m2 = n_id2 as u64;
    let m3 = n_id3 as u64;
    let mut rng = Xorshift64Star::new(seed);

    let mut id1 = Vec::with_capacity(n_rows);
    let mut id2 = Vec::with_capacity(n_rows);
    let mut id3 = Vec::with_capacity(n_rows);
    let mut v1 = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        id1.push((rng.next_u64() % m1) as i32);
        id2.push((rng.next_u64() % m2) as i32);
        id3.push((rng.next_u64() % m3) as i32);
        v1.push(rng.next_signed_unit_f64());
    }
    Fixture { id1, id2, id3, v1 }
}

// ---- CPU references ---------------------------------------------------------

/// Naive single-key COUNT: group rows by `keys[i]`, count rows per group.
/// Because the fixture's value column has no NULLs, this is exactly the
/// expected `COUNT(v1)` (and `COUNT(*)`) per group. Output sorted by key ASC so
/// the caller can compare deterministically against the (re-sorted) GPU output.
fn cpu_naive_count_groupby(keys: &[i32]) -> Vec<(i32, i64)> {
    let mut table: HashMap<i32, i64> = HashMap::with_capacity(keys.len().min(1 << 20));
    for &k in keys {
        *table.entry(k).or_insert(0) += 1;
    }
    let mut flat: Vec<(i32, i64)> = table.into_iter().collect();
    flat.sort_by_key(|&(k, _)| k);
    flat
}

/// Naive two-key COUNT: group rows by the `(a[i], b[i])` composite key, count
/// rows per group. Output sorted by `(a, b)` ASC.
fn cpu_naive_count_groupby2(a: &[i32], b: &[i32]) -> Vec<(i32, i32, i64)> {
    assert_eq!(a.len(), b.len(), "key column length mismatch");
    let mut table: HashMap<(i32, i32), i64> = HashMap::with_capacity(a.len().min(1 << 20));
    for i in 0..a.len() {
        *table.entry((a[i], b[i])).or_insert(0) += 1;
    }
    let mut flat: Vec<(i32, i32, i64)> = table.into_iter().map(|((x, y), c)| (x, y, c)).collect();
    flat.sort_by(|l, r| (l.0, l.1).cmp(&(r.0, r.1)));
    flat
}

// ---- CPU unit tests (no `#[ignore]`) ---------------------------------------
//
// These guard the CPU reference + fixture themselves so a GPU-host failure can
// be attributed to the kernel, not the oracle.

#[test]
fn fixture_is_deterministic() {
    let a = fixture(50_000, 100, 10_000, 40_000, 0x1234_5678);
    let b = fixture(50_000, 100, 10_000, 40_000, 0x1234_5678);
    assert_eq!(a.id1, b.id1, "id1 must be deterministic from seed");
    assert_eq!(a.id2, b.id2, "id2 must be deterministic from seed");
    assert_eq!(a.id3, b.id3, "id3 must be deterministic from seed");
    assert_eq!(a.v1, b.v1, "v1 must be deterministic from seed");
}

#[test]
fn cpu_count_totals_are_consistent() {
    // The sum of per-group counts must equal n_rows for both groupings: every
    // row lands in exactly one group, nothing is dropped or double-counted.
    let n_rows = 200_000;
    let f = fixture(n_rows, 100, 10_000, 100_000, 0xC0);

    let single = cpu_naive_count_groupby(&f.id3);
    let total_single: i64 = single.iter().map(|&(_, c)| c).sum();
    assert_eq!(
        total_single as usize, n_rows,
        "single-key counts must total n_rows"
    );

    let two = cpu_naive_count_groupby2(&f.id1, &f.id2);
    let total_two: i64 = two.iter().map(|&(_, _, c)| c).sum();
    assert_eq!(
        total_two as usize, n_rows,
        "two-key counts must total n_rows"
    );

    // id3 cardinality should genuinely be high (Tier-2 regime), not collapsed.
    assert!(
        single.len() > 50_000,
        "expected high distinct-key count for id3, got {}",
        single.len()
    );
}

// ---- GPU-gated integration tests -------------------------------------------
//
// `#[ignore = "gpu:tier2"]` matches the bucket convention in
// `tests/common/mod.rs` and `tests/tier2_groupby_e2e.rs`. Run on a GPU host
// via: `cargo test -- --ignored --filter gpu:tier2`.

/// Build the Arrow batch for table `x` from a fixture. Columns are
/// non-nullable, matching the SUM e2e file's `nullable = false` convention so
/// COUNT(v1) == row count per group.
#[cfg(test)]
fn build_batch(f: &Fixture) -> arrow_array::RecordBatch {
    use arrow_array::{Float64Array, Int32Array, RecordBatch};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    let id1: Int32Array = f.id1.iter().copied().collect();
    let id2: Int32Array = f.id2.iter().copied().collect();
    let id3: Int32Array = f.id3.iter().copied().collect();
    let v1: Float64Array = f.v1.iter().copied().collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id1", ArrowDataType::Int32, false),
        ArrowField::new("id2", ArrowDataType::Int32, false),
        ArrowField::new("id3", ArrowDataType::Int32, false),
        ArrowField::new("v1", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(id1), Arc::new(id2), Arc::new(id3), Arc::new(v1)],
    )
    .expect("build RecordBatch")
}

/// SINGLE-KEY Tier-2 COUNT (exercises `partition_reduce_kernel_count`, i32 key).
///
/// `SELECT id3, COUNT(v1) FROM x GROUP BY id3` over ~2M rows / ~1M distinct
/// keys. Asserts the GPU result's `(key, count)` multiset is **exactly** equal
/// to the CPU reference (COUNT is integer-exact — no tolerance).
#[test]
#[ignore = "gpu:tier2"]
fn tier2_count_single_key_i32() {
    use arrow_array::{Array, Int32Array, Int64Array};

    let n_rows: usize = 2_000_000;
    let f = fixture(n_rows, 100, 10_000, 1_000_000, 42);

    // CPU reference: COUNT(v1) per id3 == row count per id3 (no NULLs).
    let expected = cpu_naive_count_groupby(&f.id3);

    let batch = build_batch(&f);
    let mut engine = craton_bolt::Engine::new().expect("CUDA engine");
    engine.register_table("x", batch).expect("register table");

    let h = engine
        .sql("SELECT id3, COUNT(v1) FROM x GROUP BY id3")
        .expect("execute single-key COUNT groupby");
    let out = h.record_batch();

    // Output schema is SELECT-ordered: [id3 (Int32), count (Int64)].
    let key_col = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id3 must be Int32");
    let cnt_col = out
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("COUNT(v1) must be Int64");

    let mut actual: Vec<(i32, i64)> = (0..out.num_rows())
        .map(|i| (key_col.value(i), cnt_col.value(i)))
        .collect();
    actual.sort_by_key(|&(k, _)| k);

    assert_eq!(
        actual.len(),
        expected.len(),
        "distinct-group count mismatch: gpu={} cpu={}",
        actual.len(),
        expected.len()
    );
    // Exact multiset equality (both sorted by key).
    assert_eq!(
        actual, expected,
        "Tier-2 single-key COUNT(i32) mismatch vs CPU reference"
    );

    // Sanity: total counted rows == n_rows.
    let total: i64 = actual.iter().map(|&(_, c)| c).sum();
    assert_eq!(total as usize, n_rows, "total COUNT must equal n_rows");
}

/// TWO-KEY Tier-2 COUNT (exercises `partition_reduce_kernel_count_i64`, two i32
/// keys packed into one i64).
///
/// `SELECT id1, id2, COUNT(v1) FROM x GROUP BY id1, id2` over ~2M rows. With
/// id1 in [0,100) and id2 in [0,10_000), the composite group space is up to
/// ~1M distinct (id1, id2) pairs — the high-cardinality two-key regime that
/// triggers the i64 count kernel. Asserts exact `(id1, id2, count)` equality.
#[test]
#[ignore = "gpu:tier2"]
fn tier2_count_two_key_i64() {
    use arrow_array::{Array, Int32Array, Int64Array};

    let n_rows: usize = 2_000_000;
    let f = fixture(n_rows, 100, 10_000, 1_000_000, 7);

    let expected = cpu_naive_count_groupby2(&f.id1, &f.id2);

    let batch = build_batch(&f);
    let mut engine = craton_bolt::Engine::new().expect("CUDA engine");
    engine.register_table("x", batch).expect("register table");

    let h = engine
        .sql("SELECT id1, id2, COUNT(v1) FROM x GROUP BY id1, id2")
        .expect("execute two-key COUNT groupby");
    let out = h.record_batch();

    // Output schema is SELECT-ordered: [id1 (Int32), id2 (Int32), count (Int64)].
    let k1 = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id1 must be Int32");
    let k2 = out
        .column(1)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id2 must be Int32");
    let cnt = out
        .column(2)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("COUNT(v1) must be Int64");

    let mut actual: Vec<(i32, i32, i64)> = (0..out.num_rows())
        .map(|i| (k1.value(i), k2.value(i), cnt.value(i)))
        .collect();
    actual.sort_by(|l, r| (l.0, l.1).cmp(&(r.0, r.1)));

    assert_eq!(
        actual.len(),
        expected.len(),
        "distinct (id1,id2) group count mismatch: gpu={} cpu={}",
        actual.len(),
        expected.len()
    );
    assert_eq!(
        actual, expected,
        "Tier-2 two-key COUNT(i64-packed) mismatch vs CPU reference"
    );

    let total: i64 = actual.iter().map(|&(_, _, c)| c).sum();
    assert_eq!(total as usize, n_rows, "total COUNT must equal n_rows");
}

/// SINGLE-KEY `COUNT(*)` variant. `COUNT(*)` does not route through the
/// null-bitmap path (see `tests/aggregate_nulls_e2e.rs`) but must still produce
/// the same per-group row counts as `COUNT(v1)` here, since v1 has no NULLs.
///
/// If the planner rejects `COUNT(*)` with GROUP BY, `engine.sql(...)` returns
/// an `Err` and the `.expect(...)` below makes that explicit rather than
/// silently passing — report it back to the orchestrator as unsupported.
#[test]
#[ignore = "gpu:tier2"]
fn tier2_count_star_single_key() {
    use arrow_array::{Array, Int32Array, Int64Array};

    let n_rows: usize = 2_000_000;
    let f = fixture(n_rows, 100, 10_000, 1_000_000, 99);
    let expected = cpu_naive_count_groupby(&f.id3);

    let batch = build_batch(&f);
    let mut engine = craton_bolt::Engine::new().expect("CUDA engine");
    engine.register_table("x", batch).expect("register table");

    // EXPECTATION: COUNT(*) with GROUP BY is supported and equals COUNT(v1)
    // here (no NULLs in v1). If this `.expect` fires, the variant is
    // unsupported on the current code path — report it.
    let h = engine
        .sql("SELECT id3, COUNT(*) FROM x GROUP BY id3")
        .expect("execute single-key COUNT(*) groupby (report if unsupported)");
    let out = h.record_batch();

    let key_col = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id3 must be Int32");
    let cnt_col = out
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("COUNT(*) must be Int64");

    let mut actual: Vec<(i32, i64)> = (0..out.num_rows())
        .map(|i| (key_col.value(i), cnt_col.value(i)))
        .collect();
    actual.sort_by_key(|&(k, _)| k);

    assert_eq!(
        actual, expected,
        "Tier-2 single-key COUNT(*) mismatch vs CPU reference"
    );
}
