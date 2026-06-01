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

fn id1(i: usize) -> i32 {
    ((i.wrapping_mul(2_654_435_761)) as i32).rem_euclid(ID1_CARD)
}
fn v1(i: usize) -> f64 {
    ((i.wrapping_mul(7) as i32).rem_euclid(5) + 1) as f64
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

/// Host reference: SUM(v1) GROUP BY id1.
fn host_ref(n: usize) -> HashMap<i32, f64> {
    let mut m = HashMap::new();
    for i in 0..n {
        *m.entry(id1(i)).or_insert(0.0) += v1(i);
    }
    m
}

fn assert_q1(engine: &craton_bolt::Engine, n: usize) {
    let h = engine.sql("SELECT id1, SUM(v1) FROM x GROUP BY id1").expect("q1");
    let b = h.record_batch();
    let keys = b.column(0).as_any().downcast_ref::<Int32Array>().expect("id1");
    let sums = b.column(1).as_any().downcast_ref::<Float64Array>().expect("sum");
    let mut got = HashMap::new();
    for i in 0..b.num_rows() {
        got.insert(keys.value(i), sums.value(i));
    }
    let want = host_ref(n);
    assert_eq!(got.len(), want.len(), "group count mismatch at n={n}");
    for (k, &w) in &want {
        let g = *got.get(k).unwrap_or_else(|| panic!("missing group {k} at n={n}"));
        let rel = (g - w).abs() / w.abs().max(1.0);
        assert!(rel < 1e-9, "group {k}: got {g} want {w} (rel {rel:e}) at n={n}");
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
