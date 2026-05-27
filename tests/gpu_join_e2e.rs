// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the GPU JOIN fast paths.
//!
//! These tests run JOIN queries through the full `Engine::sql` pipeline at
//! sizes that trip the row-count gate in `crate::exec::join::try_gpu_*_join`,
//! exercising the GPU build + probe kernels in `crate::exec::gpu_join`.
//!
//! Every test is `#[ignore]`'d so non-GPU CI passes. Run with
//! `cargo test --test gpu_join_e2e -- --ignored` on a CUDA host.
//!
//! Stage 1 (GJ): single-key Int32/Int64 INNER, unique build keys.
//! Stage 2 (GJ-2): multi-key (TwoI32), LEFT/RIGHT/FULL OUTER, duplicate
//! build keys via collision lists, Bool + Float keys with NaN
//! canonicalisation.

use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::Engine;

/// Above the `GPU_JOIN_MIN_ROWS` (1024) threshold so the GPU path is taken.
const N_BUILD: usize = 4096;
const N_PROBE: usize = 8192;

/// Build a two-column Int32 batch: (k, v) where v depends on k.
fn int32_batch(name_k: &str, name_v: &str, keys: Vec<i32>, vals: Vec<i32>) -> RecordBatch {
    assert_eq!(keys.len(), vals.len());
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(name_k, ArrowDataType::Int32, false),
        ArrowField::new(name_v, ArrowDataType::Int32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(keys)) as ArrayRef,
            Arc::new(Int32Array::from(vals)) as ArrayRef,
        ],
    )
    .unwrap()
}

/// Build a two-column Int64 batch.
fn int64_batch(name_k: &str, name_v: &str, keys: Vec<i64>, vals: Vec<i64>) -> RecordBatch {
    assert_eq!(keys.len(), vals.len());
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(name_k, ArrowDataType::Int64, false),
        ArrowField::new(name_v, ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys)) as ArrayRef,
            Arc::new(Int64Array::from(vals)) as ArrayRef,
        ],
    )
    .unwrap()
}

/// 4k-build × 8k-probe Int32 INNER join through `Engine::sql`. The fixture is
/// arranged so the expected match count is computable host-side: build keys
/// 0..N_BUILD are unique; probe keys cycle 0..(N_BUILD * 2), so exactly
/// N_PROBE / 2 = 4096 probe rows land on a build key.
#[test]
#[ignore = "requires CUDA device - run with `cargo test --test gpu_join_e2e -- --ignored`"]
fn e2e_gpu_inner_join_int32_basic() {
    let mut engine = Engine::new().expect("ctx");

    // Build: unique keys 0..N_BUILD with payload = 1000 + k.
    let build_keys: Vec<i32> = (0..N_BUILD as i32).collect();
    let build_payload: Vec<i32> = build_keys.iter().map(|k| 1000 + k).collect();
    // Probe: keys cycle 0..(N_BUILD*2) so half match.
    let probe_keys: Vec<i32> = (0..N_PROBE as i32).map(|i| i % (N_BUILD as i32 * 2)).collect();
    let probe_payload: Vec<i32> = (0..N_PROBE as i32).map(|i| 10_000 + i).collect();

    let t1 = int32_batch("k", "bv", build_keys.clone(), build_payload.clone());
    let t2 = int32_batch("k", "pv", probe_keys.clone(), probe_payload.clone());
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 INNER JOIN t2 ON t1.k = t2.k")
        .expect("INNER JOIN");
    let out = h.record_batch();

    // Expected match count: probe rows whose key < N_BUILD.
    let expected: usize = probe_keys.iter().filter(|k| (**k as usize) < N_BUILD).count();
    assert_eq!(
        out.num_rows(),
        expected,
        "GPU INNER JOIN: row count mismatch (expected={expected})"
    );

    // Every output row must satisfy the equi-join invariant: bv = 1000 + pv_key.
    // We don't know column ordinals exactly (planner may add disambiguation),
    // so look up by name.
    let bv_idx = out
        .schema()
        .index_of("bv")
        .expect("output schema must include 'bv'");
    let pv_idx = out
        .schema()
        .index_of("pv")
        .expect("output schema must include 'pv'");
    let bv = out
        .column(bv_idx)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("bv must be Int32");
    let pv = out
        .column(pv_idx)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("pv must be Int32");

    // For matched rows, recover probe_key from pv (pv = 10_000 + probe_row),
    // then assert bv equals (1000 + probe_key) using the inverse mapping.
    // probe_row = pv - 10_000, probe_key = probe_row % (N_BUILD * 2).
    for i in 0..out.num_rows() {
        let probe_row = (pv.value(i) - 10_000) as usize;
        let probe_key = probe_keys[probe_row];
        assert_eq!(
            bv.value(i),
            1000 + probe_key,
            "row {i}: bv must equal 1000 + probe_key (got bv={}, expected={})",
            bv.value(i),
            1000 + probe_key
        );
    }
}

/// 4k × 8k Int64 INNER join — exercises the Int64 path through the same
/// fast path.
#[test]
#[ignore = "requires CUDA device"]
fn e2e_gpu_inner_join_int64_basic() {
    let mut engine = Engine::new().expect("ctx");

    let build_keys: Vec<i64> = (0..N_BUILD as i64).collect();
    let build_payload: Vec<i64> = build_keys.iter().map(|k| 1000 + k).collect();
    let probe_keys: Vec<i64> = (0..N_PROBE as i64).map(|i| i % (N_BUILD as i64 * 2)).collect();
    let probe_payload: Vec<i64> = (0..N_PROBE as i64).map(|i| 10_000 + i).collect();

    let t1 = int64_batch("k", "bv", build_keys.clone(), build_payload.clone());
    let t2 = int64_batch("k", "pv", probe_keys.clone(), probe_payload.clone());
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 INNER JOIN t2 ON t1.k = t2.k")
        .expect("INNER JOIN");
    let out = h.record_batch();

    let expected: usize = probe_keys.iter().filter(|k| (**k as usize) < N_BUILD).count();
    assert_eq!(
        out.num_rows(),
        expected,
        "GPU INNER JOIN Int64: row count mismatch (expected={expected})"
    );

    let bv_idx = out.schema().index_of("bv").expect("'bv' in output schema");
    let pv_idx = out.schema().index_of("pv").expect("'pv' in output schema");
    let bv = out
        .column(bv_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("bv must be Int64");
    let pv = out
        .column(pv_idx)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("pv must be Int64");

    for i in 0..out.num_rows() {
        let probe_row = (pv.value(i) - 10_000) as usize;
        let probe_key = probe_keys[probe_row];
        assert_eq!(
            bv.value(i),
            1000 + probe_key,
            "row {i}: bv must equal 1000 + probe_key"
        );
    }
}

/// Below-threshold INNER join must still produce correct results — it just
/// goes through the host path. Sanity-check that the fall-through doesn't
/// break correctness when the GPU gate rejects.
#[test]
#[ignore = "requires CUDA device"]
fn e2e_gpu_inner_join_small_falls_through_to_host() {
    let mut engine = Engine::new().expect("ctx");

    // Below GPU_JOIN_MIN_ROWS=1024 — host path takes this.
    let build_keys: Vec<i32> = (0..64).collect();
    let build_payload: Vec<i32> = build_keys.iter().map(|k| k * 10).collect();
    let probe_keys: Vec<i32> = (0..128).map(|i| i % 80).collect();
    let probe_payload: Vec<i32> = (0..128).map(|i| 100 + i).collect();

    let t1 = int32_batch("k", "bv", build_keys.clone(), build_payload.clone());
    let t2 = int32_batch("k", "pv", probe_keys.clone(), probe_payload.clone());
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 INNER JOIN t2 ON t1.k = t2.k")
        .expect("INNER JOIN");
    let out = h.record_batch();

    let expected: usize = probe_keys.iter().filter(|k| (**k as i32) < 64).count();
    assert_eq!(
        out.num_rows(),
        expected,
        "small INNER JOIN (host fallback): row count mismatch"
    );
}

/// Build side larger than probe side: the host path picks the smaller
/// (probe) side as the build, so the GPU executor flips orientation. This
/// test catches the build_is_left=false branch of the orient logic.
#[test]
#[ignore = "requires CUDA device"]
fn e2e_gpu_inner_join_build_larger_than_probe() {
    let mut engine = Engine::new().expect("ctx");

    // Bigger "left" side = bigger physical lhs. The host picks the smaller
    // physical side as the build, so this exercises build_is_left=false.
    let big_keys: Vec<i32> = (0..N_PROBE as i32).collect();
    let big_payload: Vec<i32> = big_keys.iter().map(|k| 200 + k).collect();
    let small_keys: Vec<i32> = (0..N_BUILD as i32).collect();
    let small_payload: Vec<i32> = small_keys.iter().map(|k| 500 + k).collect();

    let t1 = int32_batch("k", "av", big_keys.clone(), big_payload.clone());
    let t2 = int32_batch("k", "bv", small_keys.clone(), small_payload.clone());
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 INNER JOIN t2 ON t1.k = t2.k")
        .expect("INNER JOIN");
    let out = h.record_batch();

    // small is fully contained in big, so match count = small.len() = N_BUILD.
    assert_eq!(
        out.num_rows(),
        N_BUILD,
        "INNER JOIN: smaller-side-on-right -> rows == |smaller|"
    );

    // Spot-check the equi-join invariant.
    let av_idx = out.schema().index_of("av").unwrap();
    let bv_idx = out.schema().index_of("bv").unwrap();
    let av = out.column(av_idx).as_any().downcast_ref::<Int32Array>().unwrap();
    let bv = out.column(bv_idx).as_any().downcast_ref::<Int32Array>().unwrap();
    for i in 0..out.num_rows() {
        // av = 200 + k, bv = 500 + k -> bv - av = 300 for every matched row.
        assert_eq!(
            bv.value(i) - av.value(i),
            300,
            "row {i}: bv - av must equal 300 across the equi-join"
        );
    }
}

// ============================================================================
// Stage 2 (GJ-2): multi-key, OUTER joins, duplicate build keys, float keys.
// ============================================================================

/// Helper: build a three-Int32-column batch (k1, k2, payload).
fn int32x3_batch(
    name_k1: &str,
    name_k2: &str,
    name_v: &str,
    k1: Vec<i32>,
    k2: Vec<i32>,
    vals: Vec<i32>,
) -> RecordBatch {
    assert_eq!(k1.len(), k2.len());
    assert_eq!(k1.len(), vals.len());
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new(name_k1, ArrowDataType::Int32, false),
        ArrowField::new(name_k2, ArrowDataType::Int32, false),
        ArrowField::new(name_v, ArrowDataType::Int32, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(k1)) as ArrayRef,
            Arc::new(Int32Array::from(k2)) as ArrayRef,
            Arc::new(Int32Array::from(vals)) as ArrayRef,
        ],
    )
    .unwrap()
}

/// Two-key INNER JOIN: `t1 JOIN t2 ON t1.a = t2.a AND t1.b = t2.b`. Build
/// side has unique (a,b) tuples; probe contains exactly half-matching rows.
/// Exercises the Stage-2 [`KeyShape::TwoI32`] composite-pack path.
#[test]
#[ignore = "requires CUDA device"]
fn two_key_inner_join() {
    let mut engine = Engine::new().expect("ctx");
    // Build: 4096 rows, (a, b) = (i / 64, i % 64), payload = i + 1000.
    let n_build = N_BUILD;
    let n_probe = N_PROBE;
    let b_a: Vec<i32> = (0..n_build as i32).map(|i| i / 64).collect();
    let b_b: Vec<i32> = (0..n_build as i32).map(|i| i % 64).collect();
    let b_v: Vec<i32> = (0..n_build as i32).map(|i| 1000 + i).collect();

    // Probe: 8192 rows; half land on the build (when probe index < n_build),
    // the rest use disjoint (a, b) tuples (a += 1000 to push them out of
    // range).
    let p_a: Vec<i32> = (0..n_probe as i32)
        .map(|i| if (i as usize) < n_build { i / 64 } else { 1000 + i / 64 })
        .collect();
    let p_b: Vec<i32> = (0..n_probe as i32)
        .map(|i| if (i as usize) < n_build { i % 64 } else { i % 64 })
        .collect();
    let p_v: Vec<i32> = (0..n_probe as i32).map(|i| 50_000 + i).collect();

    let t1 = int32x3_batch("a", "b", "bv", b_a.clone(), b_b.clone(), b_v.clone());
    let t2 = int32x3_batch("a", "b", "pv", p_a.clone(), p_b.clone(), p_v.clone());
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 INNER JOIN t2 ON t1.a = t2.a AND t1.b = t2.b")
        .expect("two-key INNER JOIN");
    let out = h.record_batch();

    // Expected: exactly the first n_build probe rows match (each on a
    // unique (a,b) build tuple).
    assert_eq!(out.num_rows(), n_build, "two-key INNER: half-match expected");
}

/// LEFT OUTER JOIN: every left row appears at least once. Probe rows
/// whose key doesn't appear in the build side surface with the build
/// (right-side) columns NULL-padded.
#[test]
#[ignore = "requires CUDA device"]
fn left_outer_with_unmatched() {
    let mut engine = Engine::new().expect("ctx");

    // Left (probe for LEFT OUTER): 4096 rows, keys 0..4096.
    let l_keys: Vec<i32> = (0..N_BUILD as i32).collect();
    let l_vals: Vec<i32> = l_keys.iter().map(|k| 100 + k).collect();
    // Right (build for LEFT OUTER): 4096 rows, keys 2048..(2048+4096) =
    // 2048..6144. So left rows with key 0..2047 are unmatched.
    let r_keys: Vec<i32> = (2048..(2048 + N_BUILD as i32)).collect();
    let r_vals: Vec<i32> = r_keys.iter().map(|k| 500 + k).collect();

    let t1 = int32_batch("k", "lv", l_keys.clone(), l_vals.clone());
    let t2 = int32_batch("k", "rv", r_keys.clone(), r_vals.clone());
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 LEFT OUTER JOIN t2 ON t1.k = t2.k")
        .expect("LEFT OUTER JOIN");
    let out = h.record_batch();

    // Every left row appears exactly once.
    assert_eq!(
        out.num_rows(),
        N_BUILD,
        "LEFT OUTER: row count must equal left.len()"
    );

    // Unmatched rows (left key < 2048) have rv NULL; matched rows have rv
    // set to 500 + left_key.
    let rv_idx = out.schema().index_of("rv").expect("'rv' in output");
    let rv = out.column(rv_idx);
    let n_nulls = rv.null_count();
    assert_eq!(
        n_nulls, 2048,
        "LEFT OUTER: exactly 2048 rows must have rv NULL (left key 0..2047)"
    );
}

/// RIGHT OUTER JOIN: every right row appears at least once. Symmetric to
/// LEFT — the build side is now the LEFT table.
#[test]
#[ignore = "requires CUDA device"]
fn right_outer_with_unmatched_build() {
    let mut engine = Engine::new().expect("ctx");
    let l_keys: Vec<i32> = (0..N_BUILD as i32).collect();
    let l_vals: Vec<i32> = l_keys.iter().map(|k| 100 + k).collect();
    // Right has more rows + non-overlapping tail.
    let r_keys: Vec<i32> = (2048..(2048 + N_BUILD as i32)).collect();
    let r_vals: Vec<i32> = r_keys.iter().map(|k| 500 + k).collect();

    let t1 = int32_batch("k", "lv", l_keys, l_vals);
    let t2 = int32_batch("k", "rv", r_keys, r_vals);
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 RIGHT OUTER JOIN t2 ON t1.k = t2.k")
        .expect("RIGHT OUTER JOIN");
    let out = h.record_batch();

    // Every right row appears exactly once.
    assert_eq!(out.num_rows(), N_BUILD);
    // Unmatched rights (key 4096..6144) have lv NULL.
    let lv_idx = out.schema().index_of("lv").expect("'lv' in output");
    assert_eq!(
        out.column(lv_idx).null_count(),
        2048,
        "RIGHT OUTER: 2048 rights with key 4096..6144 must NULL-pad lv"
    );
}

/// FULL OUTER JOIN: union of LEFT + RIGHT semantics. Both unmatched sets
/// surface — one with the right NULL'd, the other with the left NULL'd.
#[test]
#[ignore = "requires CUDA device"]
fn full_outer_emits_both_sides() {
    let mut engine = Engine::new().expect("ctx");
    let l_keys: Vec<i32> = (0..N_BUILD as i32).collect();
    let l_vals: Vec<i32> = l_keys.iter().map(|k| 100 + k).collect();
    let r_keys: Vec<i32> = (2048..(2048 + N_BUILD as i32)).collect();
    let r_vals: Vec<i32> = r_keys.iter().map(|k| 500 + k).collect();

    let t1 = int32_batch("k", "lv", l_keys, l_vals);
    let t2 = int32_batch("k", "rv", r_keys, r_vals);
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 FULL OUTER JOIN t2 ON t1.k = t2.k")
        .expect("FULL OUTER JOIN");
    let out = h.record_batch();

    // 4096 left rows total: 2048 matched + 2048 left-only.
    // 4096 right rows total: 2048 matched (already counted) + 2048 right-only.
    // Output: 2048 (matched) + 2048 (left-only) + 2048 (right-only) = 6144.
    assert_eq!(
        out.num_rows(),
        6144,
        "FULL OUTER: matched + left-only + right-only = 6144"
    );
    // 2048 rows have lv NULL (right-only); 2048 rows have rv NULL (left-only).
    let lv_idx = out.schema().index_of("lv").unwrap();
    let rv_idx = out.schema().index_of("rv").unwrap();
    assert_eq!(out.column(lv_idx).null_count(), 2048);
    assert_eq!(out.column(rv_idx).null_count(), 2048);
}

/// INNER JOIN with duplicate build keys. Build has three rows with key=K,
/// probe has two rows with key=K → expect 6 output rows. Exercises the
/// collision-list build + chain-walking probe kernels.
#[test]
#[ignore = "requires CUDA device"]
fn duplicate_build_keys_emit_all_matches() {
    let mut engine = Engine::new().expect("ctx");

    // Build: 4096 rows, every key in 0..1024 appears 4 times (rotating
    // through value).
    let n_build = N_BUILD;
    let n_probe = N_PROBE;
    let b_keys: Vec<i32> = (0..n_build as i32).map(|i| i % 1024).collect();
    let b_vals: Vec<i32> = (0..n_build as i32).map(|i| 1000 + i).collect();
    // Probe: 8192 rows, every key in 0..1024 appears 8 times.
    let p_keys: Vec<i32> = (0..n_probe as i32).map(|i| i % 1024).collect();
    let p_vals: Vec<i32> = (0..n_probe as i32).map(|i| 5000 + i).collect();

    let t1 = int32_batch("k", "bv", b_keys, b_vals);
    let t2 = int32_batch("k", "pv", p_keys, p_vals);
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 INNER JOIN t2 ON t1.k = t2.k")
        .expect("dup-key INNER JOIN");
    let out = h.record_batch();

    // Each of 1024 distinct keys has 4 build rows × 8 probe rows = 32
    // matches. Total = 1024 * 32 = 32_768 output rows.
    assert_eq!(
        out.num_rows(),
        32_768,
        "duplicate-key INNER: 4*8 matches per key * 1024 keys = 32768"
    );
}

/// Float64 INNER JOIN with NaN canonicalisation. Build has a row with NaN
/// as its key; probe has multiple rows whose NaN bit-pattern differs from
/// `f64::NAN.to_bits()`. After canonicalisation they must all match.
#[test]
#[ignore = "requires CUDA device"]
fn float_key_join() {
    let mut engine = Engine::new().expect("ctx");

    // Build: 4096 unique-keyed rows (key = 1.0, 2.0, 3.0, ..., 4096.0), plus
    // one NaN at the end.
    let mut b_keys: Vec<f64> = (1..=N_BUILD).map(|i| i as f64).collect();
    b_keys[N_BUILD - 1] = f64::NAN;
    let b_vals: Vec<f64> = (0..N_BUILD as i32).map(|i| 100.0 + i as f64).collect();

    // Probe: 8192 rows. Half hit the build (key = 1..4095), the rest use a
    // NaN with a non-canonical bit pattern.
    let weird_nan = f64::from_bits(f64::NAN.to_bits() ^ 0x1);
    assert!(weird_nan.is_nan());
    let p_keys: Vec<f64> = (0..N_PROBE as i32)
        .map(|i| {
            if (i as usize) < N_BUILD - 1 {
                (1 + i) as f64
            } else {
                weird_nan
            }
        })
        .collect();
    let p_vals: Vec<f64> = (0..N_PROBE as i32).map(|i| 1000.0 + i as f64).collect();

    let build_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Float64, false),
        ArrowField::new("bv", ArrowDataType::Float64, false),
    ]));
    let t1 = RecordBatch::try_new(
        build_schema,
        vec![
            Arc::new(Float64Array::from(b_keys.clone())) as ArrayRef,
            Arc::new(Float64Array::from(b_vals)) as ArrayRef,
        ],
    )
    .unwrap();

    let probe_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Float64, false),
        ArrowField::new("pv", ArrowDataType::Float64, false),
    ]));
    let t2 = RecordBatch::try_new(
        probe_schema,
        vec![
            Arc::new(Float64Array::from(p_keys.clone())) as ArrayRef,
            Arc::new(Float64Array::from(p_vals)) as ArrayRef,
        ],
    )
    .unwrap();
    engine.register_table("t1", t1).unwrap();
    engine.register_table("t2", t2).unwrap();

    let h = engine
        .sql("SELECT * FROM t1 INNER JOIN t2 ON t1.k = t2.k")
        .expect("float INNER JOIN");
    let out = h.record_batch();

    // Matched rows = (N_BUILD - 1) finite-value matches (probe rows
    // 0..(N_BUILD - 1)) + (N_PROBE - (N_BUILD - 1)) NaN-canonicalised
    // matches (each maps to the single NaN build row).
    let expected = (N_BUILD - 1) + (N_PROBE - (N_BUILD - 1));
    assert_eq!(
        out.num_rows(),
        expected,
        "float INNER: expected NaN canonicalisation to merge all NaN probes onto the NaN build row"
    );
}
