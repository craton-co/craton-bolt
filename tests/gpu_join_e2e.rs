// SPDX-License-Identifier: Apache-2.0

//! End-to-end tests for the GPU INNER JOIN fast path.
//!
//! These tests run `SELECT * FROM t1 INNER JOIN t2 ON t1.k = t2.k` through
//! the full `Engine::sql` pipeline at sizes that trip the row-count gate in
//! `crate::exec::join::try_gpu_inner_join`, exercising the GPU build + probe
//! kernels in `crate::exec::gpu_join`.
//!
//! Every test is `#[ignore]`'d so non-GPU CI passes. Run with
//! `cargo test --test gpu_join_e2e -- --ignored` on a CUDA host.

use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Int32Array, Int64Array, RecordBatch};
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
