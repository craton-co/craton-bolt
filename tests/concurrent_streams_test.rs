// SPDX-License-Identifier: Apache-2.0

//! Concurrency regression for the CUDA stream pool (`src/cuda/stream_pool.rs`).
//!
//! The stream-pool fix guarantees that two concurrent queries can never share
//! a single CUDA stream: `acquire` hands out a stream with *exclusive*
//! ownership and `release` returns it for reuse only after the query is done.
//! Every other repeated-invocation test in this suite is single-threaded
//! (`groupby_resident_replace_regression.rs`,
//! `tier2_high_card_groupby_survives_repeated_invocation`, ...), so the
//! concurrency invariant — no pool corruption, no shared-stream aliasing, no
//! wrong results under parallel `acquire`/`release` churn — is otherwise
//! UNTESTED. This file exercises it directly.
//!
//! # Concurrency model: one `Engine` PER thread (REQUIREMENT 4)
//!
//! `craton_bolt::Engine` is `!Sync`: it holds `RefCell<…>` fields
//! (`streaming_sources`, `gpu_tables`) for interior mutability, and the
//! struct's own field docs state outright that "the underlying engine is not
//! yet `Send + Sync` because of `RefCell`". `Engine::sql` takes `&self`, but
//! that `&self` cannot legally be shared across threads while those `RefCell`s
//! exist — `&Engine` is not `Send`. So the shared-`Arc<Engine>` model in
//! REQUIREMENT 3 is impossible against the current public API.
//!
//! Per REQUIREMENT 4 we therefore build one `Engine` per thread. Each thread
//! constructs its own engine on the default CUDA device, registers the SAME
//! deterministic fixture, and runs the SAME mix of GROUP BY / aggregate / join
//! queries in a loop. The CUDA *stream pool* the fix protects is a
//! process-global resource (`cuda::stream_pool`), so N engines querying
//! concurrently still drive concurrent `acquire`/`release` against that one
//! shared pool — which is exactly the contended path the fix guards. Any pool
//! corruption, shared-stream aliasing, or use-after-free surfaces as a panic,
//! a hang, or a wrong result, all of which this test catches.
//!
//! # What is asserted
//!
//! Reference results for every query are computed SINGLE-THREADED before the
//! parallel section (the queries are deterministic). Each of the ~8 worker
//! threads then runs all queries 20 times and asserts every result is
//! bit/relative-equal to the reference. A divergence under concurrency — but
//! not single-threaded — is the signature of stream aliasing / pool
//! corruption. A panic or hang likewise fails the test (a hung thread blocks
//! the join and the test runner times out).
//!
//! GPU-gated with the suite's `#[ignore = "gpu:*"]` convention (see
//! `tests/common/mod.rs` for the bucket scheme); this adds a `gpu:concurrent`
//! bucket. It will not run on non-GPU CI but MUST compile and link as part of
//! the test suite. Run on a CUDA host with:
//! `cargo test --test concurrent_streams_test -- --ignored`.

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

use arrow_array::{Array, ArrayRef, Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

use craton_bolt::Engine;

mod common;
use common::REL_TOL;

// ---- Fixture ----------------------------------------------------------------
//
// h2o-shaped generators, matching `groupby_resident_replace_regression.rs` /
// `benches/olap_benchmarks.rs` so the Tier-2 dispatch thresholds are tripped
// for the high-cardinality keys.

const ID1_CARD: i32 = 100;
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

/// Full h2o-shaped fact table: id1/id2/id3 Int32 keys + v1/v2 Float64 values.
fn fact_batch(n: usize) -> RecordBatch {
    let s = Arc::new(Schema::new(vec![
        Field::new("id1", DataType::Int32, false),
        Field::new("id2", DataType::Int32, false),
        Field::new("id3", DataType::Int32, false),
        Field::new("v1", DataType::Float64, false),
        Field::new("v2", DataType::Float64, false),
    ]));
    let id1c: Int32Array = (0..n).map(id1).collect();
    let id2c: Int32Array = (0..n).map(id2).collect();
    let id3c: Int32Array = (0..n).map(id3).collect();
    let v1c: Float64Array = (0..n).map(v1).collect();
    let v2c: Float64Array = (0..n).map(v2).collect();
    RecordBatch::try_new(
        s,
        vec![
            Arc::new(id1c) as ArrayRef,
            Arc::new(id2c) as ArrayRef,
            Arc::new(id3c) as ArrayRef,
            Arc::new(v1c) as ArrayRef,
            Arc::new(v2c) as ArrayRef,
        ],
    )
    .unwrap()
}

/// Small dimension table keyed by id1 (0..ID1_CARD), used for the JOIN query.
/// `dval = 1000 + id1` so a matched row's value is recoverable host-side.
fn dim_batch() -> RecordBatch {
    let s = Arc::new(Schema::new(vec![
        Field::new("id1", DataType::Int32, false),
        Field::new("dval", DataType::Int32, false),
    ]));
    let keys: Int32Array = (0..ID1_CARD).collect();
    let vals: Int32Array = (0..ID1_CARD).map(|k| 1000 + k).collect();
    RecordBatch::try_new(s, vec![Arc::new(keys) as ArrayRef, Arc::new(vals) as ArrayRef])
        .unwrap()
}

// ---- Result extraction ------------------------------------------------------
//
// A query result is normalised into a sorted `Vec<(i32, f64)>` keyed by the
// group key (or, for the join, by id1) so it can be compared deterministically
// regardless of the engine's row emission order.

/// Run a single-key `SELECT k, SUM(v) GROUP BY k` style query and collect a
/// sorted `(key, agg)` vector. Column 0 must be Int32, column 1 Float64.
fn collect_key_agg(engine: &Engine, sql: &str) -> Vec<(i32, f64)> {
    let h = engine.sql(sql).unwrap_or_else(|e| panic!("query failed: {sql}: {e}"));
    let b = h.record_batch();
    let keys = b
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap_or_else(|| panic!("col0 not Int32 for: {sql}"));
    let aggs = b
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap_or_else(|| panic!("col1 not Float64 for: {sql}"));
    let mut out: Vec<(i32, f64)> = (0..b.num_rows())
        .map(|i| (keys.value(i), aggs.value(i)))
        .collect();
    out.sort_by_key(|&(k, _)| k);
    out
}

/// The full deterministic query mix every thread runs. Returns a stable,
/// comparable digest of all results so a single `assert_eq!`/`rel-eq` per
/// query covers the whole mix.
///
/// Mix rationale:
/// - q_t1: low-cardinality single-key GROUP BY (Tier-1 shmem path, id1).
/// - q_t2: high-cardinality single-key GROUP BY (Tier-2 path, id3) — the
///   path whose per-query-stream UAF the pool fix replaced.
/// - q_two: two-key GROUP BY (id1, id2) — two-key Tier-2 above TWOKEY_MIN_ROWS.
/// - q_agg: scalar aggregate (no GROUP BY) — the resident on-device reduce.
/// - q_join: GROUP BY over an INNER JOIN — exercises the join build/probe
///   kernels' stream usage alongside the aggregate.
struct QuerySet {
    t1: Vec<(i32, f64)>,
    t2: Vec<(i32, f64)>,
    two: Vec<(i32, f64)>,
    agg: f64,
    join: Vec<(i32, f64)>,
}

const Q_T1: &str = "SELECT id1, SUM(v1) FROM fact GROUP BY id1";
const Q_T2: &str = "SELECT id3, SUM(v1) FROM fact GROUP BY id3";
const Q_TWO: &str = "SELECT id1, id2, SUM(v1) FROM fact GROUP BY id1, id2";
const Q_AGG: &str = "SELECT SUM(v2) FROM fact";
const Q_JOIN: &str =
    "SELECT fact.id1, SUM(fact.v1) FROM fact INNER JOIN dim ON fact.id1 = dim.id1 \
     GROUP BY fact.id1";

/// Collect the two-key result, folding (id1, id2) into a single i32 surrogate
/// key so it fits the `(i32, f64)` comparison vector. Column layout: id1, id2,
/// SUM(v1).
fn collect_two_key(engine: &Engine) -> Vec<(i32, f64)> {
    let h = engine.sql(Q_TWO).unwrap_or_else(|e| panic!("q_two failed: {e}"));
    let b = h.record_batch();
    let k1 = b.column(0).as_any().downcast_ref::<Int32Array>().expect("id1 Int32");
    let k2 = b.column(1).as_any().downcast_ref::<Int32Array>().expect("id2 Int32");
    let s = b.column(2).as_any().downcast_ref::<Float64Array>().expect("sum Float64");
    let mut out: Vec<(i32, f64)> = (0..b.num_rows())
        .map(|i| {
            // id2 < ID2_CARD (10_000) so (id1 * 100_000 + id2) is collision-free
            // and stays within i32 range for the i32 keys here.
            let surrogate = k1.value(i).wrapping_mul(100_000).wrapping_add(k2.value(i));
            (surrogate, s.value(i))
        })
        .collect();
    out.sort_by_key(|&(k, _)| k);
    out
}

/// Run a scalar `SELECT SUM(v2) FROM fact` and return the single f64 cell.
fn collect_scalar(engine: &Engine) -> f64 {
    let h = engine.sql(Q_AGG).unwrap_or_else(|e| panic!("q_agg failed: {e}"));
    let b = h.record_batch();
    let c = b
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("scalar SUM must be Float64");
    assert_eq!(b.num_rows(), 1, "scalar aggregate must return exactly one row");
    c.value(0)
}

/// Run the full query mix against an already-populated engine.
fn run_all(engine: &Engine) -> QuerySet {
    QuerySet {
        t1: collect_key_agg(engine, Q_T1),
        t2: collect_key_agg(engine, Q_T2),
        two: collect_two_key(engine),
        agg: collect_scalar(engine),
        join: collect_key_agg(engine, Q_JOIN),
    }
}

/// Build a fresh engine on the default device and register the standard
/// `fact` + `dim` fixture. Factored so the reference pass and every worker
/// thread stand up identical state.
fn build_engine(n: usize) -> Engine {
    let mut engine = Engine::new().expect("CUDA engine");
    engine.register_table("fact", fact_batch(n)).expect("register fact");
    engine.register_table("dim", dim_batch()).expect("register dim");
    engine
}

// ---- Comparison -------------------------------------------------------------

fn assert_key_agg_eq(label: &str, got: &[(i32, f64)], want: &[(i32, f64)]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: group-count mismatch (got {}, want {})",
        got.len(),
        want.len()
    );
    for (g, w) in got.iter().zip(want.iter()) {
        assert_eq!(g.0, w.0, "{label}: key mismatch {} vs {}", g.0, w.0);
        let denom = g.1.abs().max(w.1.abs()).max(1.0);
        let rel = (g.1 - w.1).abs() / denom;
        assert!(
            rel < REL_TOL,
            "{label}: key {} value diverged under concurrency: got {} want {} (rel {rel:e})",
            g.0, g.1, w.1
        );
    }
}

fn assert_queryset_eq(label: &str, got: &QuerySet, want: &QuerySet) {
    assert_key_agg_eq(&format!("{label}/q_t1"), &got.t1, &want.t1);
    assert_key_agg_eq(&format!("{label}/q_t2"), &got.t2, &want.t2);
    assert_key_agg_eq(&format!("{label}/q_two"), &got.two, &want.two);
    let denom = got.agg.abs().max(want.agg.abs()).max(1.0);
    let rel = (got.agg - want.agg).abs() / denom;
    assert!(
        rel < REL_TOL,
        "{label}/q_agg: scalar SUM diverged under concurrency: got {} want {} (rel {rel:e})",
        got.agg, want.agg
    );
    assert_key_agg_eq(&format!("{label}/q_join"), &got.join, &want.join);
}

// ---- The concurrency test ---------------------------------------------------

#[test]
#[ignore = "gpu:concurrent"]
fn concurrent_queries_do_not_alias_streams() {
    // 2M rows: above TWOKEY_MIN_ROWS (256K) so the two-key path uses Tier-2,
    // and large enough that id3 (1M-cardinality) exercises the high-card
    // Tier-2 single-key path. The same size the single-threaded
    // repeated-invocation regression uses.
    const N_ROWS: usize = 2_000_000;
    const N_THREADS: usize = 8;
    const ITERS: usize = 20;

    // --- Reference pass (single-threaded) -----------------------------------
    // Compute the ground-truth result for every query with zero concurrency.
    // Any later divergence is then attributable to the parallel section.
    let reference = {
        let engine = build_engine(N_ROWS);
        run_all(&engine)
    };
    // Wrap read-only so every worker shares the immutable reference cheaply.
    let reference = Arc::new(reference);

    // --- Parallel section ---------------------------------------------------
    // Spawn N worker threads; each builds its OWN engine (Engine is !Sync) and
    // hammers the global CUDA stream pool with the full query mix for ITERS
    // iterations, asserting every result equals the single-threaded reference.
    let mut handles = Vec::with_capacity(N_THREADS);
    for tid in 0..N_THREADS {
        let reference = Arc::clone(&reference);
        handles.push(thread::spawn(move || {
            // Per-thread engine + fixture: drives concurrent acquire/release
            // against the shared process-global stream pool.
            let engine = build_engine(N_ROWS);
            for iter in 0..ITERS {
                let got = run_all(&engine);
                assert_queryset_eq(&format!("thread {tid} iter {iter}"), &got, &reference);
            }
            tid
        }));
    }

    // Join all workers. A panic in any thread (wrong result / CUDA fault)
    // propagates here and fails the test; a hung thread (deadlocked pool)
    // blocks the join until the harness times the test out.
    let mut joined = HashMap::new();
    for h in handles {
        let tid = h.join().expect("worker thread panicked — stream-pool corruption suspected");
        joined.insert(tid, ());
    }
    assert_eq!(
        joined.len(),
        N_THREADS,
        "every worker thread must complete exactly once"
    );
}
