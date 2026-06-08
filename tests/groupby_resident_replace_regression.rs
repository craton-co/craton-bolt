// SPDX-License-Identifier: Apache-2.0
//
//! Regression: the resident GROUP BY fast path must not associate the
//! long-lived resident `GpuTable` buffers with a per-query owned CUDA stream.
//!
//! Before the fix, `groupby_shmem_exec::try_execute_resident` ran its kernels
//! on a `CudaStream::null_or_default()` (a freshly-created, owned stream that is
//! destroyed when the query returns). Because that path reads the *resident*
//! key/value buffers — which outlive the query in `gpu_tables` — the memory
//! pool's stream-aware async-free later recorded an event on the now-destroyed
//! stream when the table was freed by `replace_table`, producing
//! `CUDA_ERROR_INVALID_HANDLE` (cuEventRecord / cuStreamSynchronize) and a
//! process segfault. The minimal trigger is: register → run the resident query
//! → `replace_table` → run it again. This test reproduces exactly that and
//! asserts the second query both succeeds and returns the correct result.
//!
//! GPU-gated: run with `BOLT_BENCH_GPU=1 cargo test --release
//! --no-default-features --features cudarc --test
//! groupby_resident_replace_regression -- --ignored`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

const ID1_CARD: i32 = 100;

// h2o generators (match benches/olap_benchmarks.rs exactly).
const ID2_CARD: i32 = 10_000;
const ID3_CARD: i32 = 1_000_000;
fn id1(i: usize) -> i32 {
    ((i.wrapping_mul(2_654_435_761)) as i32).rem_euclid(ID1_CARD)
}
fn id2(i: usize) -> i32 {
    ((i.wrapping_mul(40_503)) as i32).rem_euclid(ID2_CARD)
}
fn id3(i: usize) -> i32 {
    ((i.wrapping_mul(11_400_714_819_323_198_485_u64 as usize)) as i32).rem_euclid(ID3_CARD)
}
fn v1(i: usize) -> f64 {
    ((i.wrapping_mul(7) as i32).rem_euclid(5) + 1) as f64
}
fn v2(i: usize) -> f64 {
    ((i.wrapping_mul(13) as i32).rem_euclid(15) + 1) as f64
}
fn v3(i: usize) -> f64 {
    ((i.wrapping_mul(17) as i32).rem_euclid(10_000)) as f64 / 100.0
}

fn batch(n: usize) -> RecordBatch {
    let s = Arc::new(Schema::new(vec![
        Field::new("id1", DataType::Int32, false),
        Field::new("v1", DataType::Float64, false),
    ]));
    let id1c: Int32Array = (0..n).map(id1).collect();
    let v1c: Float64Array = (0..n).map(v1).collect();
    RecordBatch::try_new(s, vec![Arc::new(id1c), Arc::new(v1c)]).unwrap()
}

/// Full h2o schema (id1/id2/id3 Int32, v1/v2/v3 Float64) so the q1–q5
/// sequence — multi-SUM, two-key, AVG, high-card — all have their columns.
fn full_batch(n: usize) -> RecordBatch {
    let s = Arc::new(Schema::new(vec![
        Field::new("id1", DataType::Int32, false),
        Field::new("id2", DataType::Int32, false),
        Field::new("id3", DataType::Int32, false),
        Field::new("v1", DataType::Float64, false),
        Field::new("v2", DataType::Float64, false),
        Field::new("v3", DataType::Float64, false),
    ]));
    let id1c: Int32Array = (0..n).map(id1).collect();
    let id2c: Int32Array = (0..n).map(id2).collect();
    let id3c: Int32Array = (0..n).map(id3).collect();
    let v1c: Float64Array = (0..n).map(v1).collect();
    let v2c: Float64Array = (0..n).map(v2).collect();
    let v3c: Float64Array = (0..n).map(v3).collect();
    RecordBatch::try_new(
        s,
        vec![
            Arc::new(id1c),
            Arc::new(id2c),
            Arc::new(id3c),
            Arc::new(v1c),
            Arc::new(v2c),
            Arc::new(v3c),
        ],
    )
    .unwrap()
}

/// Host reference: SUM(v1) GROUP BY id1.
fn host_ref(n: usize) -> HashMap<i32, f64> {
    let mut m = HashMap::new();
    for i in 0..n {
        *m.entry(id1(i)).or_insert(0.0) += v1(i);
    }
    m
}

fn assert_q1(engine: &craton_bolt::Engine, n: usize) {
    let h = engine
        .sql("SELECT id1, SUM(v1) FROM x GROUP BY id1")
        .expect("q1");
    let b = h.record_batch();
    let keys = b
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id1");
    let sums = b
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("sum");
    let mut got = HashMap::new();
    for i in 0..b.num_rows() {
        got.insert(keys.value(i), sums.value(i));
    }
    let want = host_ref(n);
    assert_eq!(got.len(), want.len(), "group count mismatch at n={n}");
    for (k, &w) in &want {
        let g = *got
            .get(k)
            .unwrap_or_else(|| panic!("missing group {k} at n={n}"));
        let rel = (g - w).abs() / w.abs().max(1.0);
        assert!(
            rel < 1e-9,
            "group {k}: got {g} want {w} (rel {rel:e}) at n={n}"
        );
    }
}

#[test]
#[ignore = "requires a CUDA device; run with BOLT_BENCH_GPU=1 -- --ignored"]
fn resident_groupby_survives_replace_table() {
    if std::env::var("BOLT_BENCH_GPU").ok().as_deref() != Some("1") {
        eprintln!("skipping: set BOLT_BENCH_GPU=1");
        return;
    }
    let small = 100_000;
    let big = 2_000_000;

    let mut engine = craton_bolt::Engine::new().expect("engine");

    // 1. Register small, run the resident q1 (this previously tagged the
    //    resident buffers with a per-query stream that gets destroyed).
    engine.register_table("x", batch(small)).expect("register");
    assert_q1(&engine, small);

    // 2. replace_table frees the small resident buffers — the moment the
    //    dangling-stream async-free used to fire CUDA_ERROR_INVALID_HANDLE.
    engine.replace_table("x", batch(big)).expect("replace");

    // 3. Run the resident q1 again on the new data. Must not crash, must be
    //    correct. Run a few times to exercise repeated alloc/free cycles.
    for _ in 0..3 {
        assert_q1(&engine, big);
    }

    // 4. A second replace + query, to be thorough.
    engine.replace_table("x", batch(small)).expect("replace2");
    assert_q1(&engine, small);
}

/// The exact olap_benchmarks-shaped sequence that previously segfaulted:
/// register a small fixture, run ALL five h2o GROUP BY shapes (q1 resident
/// single-SUM, q2 multi-SUM, q3 two-key, q4 AVG, q5 high-card), then
/// `replace_table` and run the resident q1 again. The mix matters because the
/// bolt equivalence step in the bench runs every shape before the table swap;
/// the resident-q1 stream-lifetime bug fired on the post-replace q1. Asserts no
/// crash and a correct q1 result.
#[test]
#[ignore = "requires a CUDA device; run with BOLT_BENCH_GPU=1 -- --ignored"]
fn resident_groupby_survives_mixed_shape_sequence_then_replace() {
    if std::env::var("BOLT_BENCH_GPU").ok().as_deref() != Some("1") {
        eprintln!("skipping: set BOLT_BENCH_GPU=1");
        return;
    }
    let small = 100_000;
    let big = 2_000_000;

    let mut engine = craton_bolt::Engine::new().expect("engine");
    engine
        .register_table("x", full_batch(small))
        .expect("register");

    // Run all five shapes at the small size (the bench's equivalence step). We
    // don't assert their values here (other tests cover correctness); the point
    // is that running them must not leave the context in a state that crashes
    // the post-replace resident q1.
    for q in [
        "SELECT id1, SUM(v1) FROM x GROUP BY id1",
        "SELECT id2, SUM(v1), SUM(v2) FROM x GROUP BY id2",
        "SELECT id1, id2, SUM(v1) FROM x GROUP BY id1, id2",
        "SELECT id1, AVG(v1), AVG(v2), AVG(v3) FROM x GROUP BY id1",
        "SELECT id3, SUM(v1) FROM x GROUP BY id3",
    ] {
        engine
            .sql(q)
            .unwrap_or_else(|e| panic!("query failed: {q}: {e}"));
    }

    // Swap in the big dataset (frees the small resident buffers) and re-run the
    // resident q1 — this is the post-replace query that used to segfault.
    engine.replace_table("x", full_batch(big)).expect("replace");
    assert_q1(&engine, big);
}

/// Repeated-invocation stress: the Tier-2 GROUP BY orchestrators previously ran
/// on a per-query OWNED CUDA stream (`null_or_default`). Across many calls the
/// memory pool's stream-aware async-free referenced a now-destroyed per-query
/// stream, accumulating to a CUDA_ERROR_INVALID_HANDLE / use-after-free segfault
/// after ~9 iterations (it surfaced as a crash in the olap bench's criterion
/// warmup loop, originally mis-diagnosed as a high-cardinality kernel TDR — the
/// reduce kernel is actually ~1-4 ms). The fix runs the orchestrators on the
/// long-lived NULL stream. This loops the high-cardinality two-key (q3) and
/// high-card single-key (q5) Tier-2 paths well past the old ~9-iteration crash
/// point.
#[test]
#[ignore = "requires a CUDA device; run with BOLT_BENCH_GPU=1 -- --ignored"]
fn tier2_high_card_groupby_survives_repeated_invocation() {
    if std::env::var("BOLT_BENCH_GPU").ok().as_deref() != Some("1") {
        eprintln!("skipping: set BOLT_BENCH_GPU=1");
        return;
    }
    // 2M rows > TWOKEY_MIN_ROWS (256K) so q3 uses the two-key Tier-2 path; the
    // accumulation bug is iteration-count-dependent, not size-dependent.
    let n = 2_000_000;
    let mut engine = craton_bolt::Engine::new().expect("engine");
    engine.register_table("x", full_batch(n)).expect("register");

    // q3: two-key Tier-2 (id1×id2). q5: high-card single-key Tier-2 (id3).
    for i in 0..20 {
        let h3 = engine
            .sql("SELECT id1, id2, SUM(v1) FROM x GROUP BY id1, id2")
            .unwrap_or_else(|e| panic!("q3 iter {i}: {e}"));
        assert!(h3.record_batch().num_rows() > 0, "q3 iter {i}: empty");
        let h5 = engine
            .sql("SELECT id3, SUM(v1) FROM x GROUP BY id3")
            .unwrap_or_else(|e| panic!("q5 iter {i}: {e}"));
        assert!(h5.record_batch().num_rows() > 0, "q5 iter {i}: empty");
    }
}
