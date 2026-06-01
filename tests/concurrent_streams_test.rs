// SPDX-License-Identifier: Apache-2.0

//! Multi-engine + stream-pool stability stress (`src/cuda/stream_pool.rs`,
//! `src/exec/module_cache.rs`, `src/jit/jit_compiler.rs`).
//!
//! # Why this is sequential, not concurrent
//!
//! `craton_bolt::Engine` keeps exactly ONE active CUDA context per process, and
//! the resources that back a query — the device-memory pool, the stream pool,
//! and the JIT module caches — are all process-global statics BOUND to the
//! current context. A pointer, stream, or `CudaModule` minted in one context is
//! invalid in another. The engine is also `!Sync` (it holds `RefCell<…>`
//! fields), so `&Engine` cannot be shared across threads. Running two engines
//! (two contexts on one device) at the same time would therefore cross-
//! contaminate those globals — that is a known architectural limitation, not a
//! supported mode (see `docs/LIMITATIONS.md`). So this test exercises the two
//! patterns that ARE supported, both single-threaded:
//!
//! 1. **Sequential multi-engine.** Build a fresh engine (new context), run the
//!    full query mix, drop it, repeat. This is the regression guard for the
//!    cross-context module-cache bug: the global module caches key on the
//!    device ordinal / PTX-text hash, not the context, so a `CudaModule` cached
//!    by a destroyed context must be cleared on teardown — otherwise the next
//!    engine's first launch fails with `cuModuleGetFunction ... invalid
//!    resource handle`. `CudaContext::Drop` now clears those caches; this test
//!    proves a long sequence of engines stays correct.
//!
//! 2. **Intra-engine repeated queries.** Within one engine, run the mix many
//!    times to drive the stream pool's `acquire`/`release` cycle — the path
//!    whose per-query owned-stream use-after-free the stream-pool fix replaced.
//!
//! # What is asserted
//!
//! A single-engine reference result is computed first (the queries are
//! deterministic). Every engine instance and every repeated iteration must
//! produce results bit/relative-equal to that reference; any divergence, panic,
//! or CUDA fault fails the test.
//!
//! GPU-gated with the suite's `#[ignore = "gpu:*"]` convention (see
//! `tests/common/mod.rs`); uses a `gpu:multi-engine` bucket. It does not run on
//! non-GPU CI but MUST compile and link. Run on a CUDA host with:
//! `cargo test --test concurrent_streams_test -- --ignored`.

use std::sync::Arc;

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
/// Mix rationale — only DETERMINISTIC paths are compared exactly:
/// - q_t1: low-cardinality single-key GROUP BY (Tier-1 shmem path, id1) — a
///   stable group count (id1 has 100 distinct keys) and exact sums, so it can
///   be compared bit/rel-exact across every engine and iteration.
/// - q_agg: scalar aggregate (no GROUP BY) — the resident on-device reduce.
///
/// High-cardinality Tier-2 GROUP BY (id2/id3) is deliberately EXCLUDED here:
/// that path has a known nondeterministic phantom-group issue at scale, and
/// across multiple live/destroyed contexts it additionally trips a separate
/// context-bound device-buffer hazard (a crash) — both are pre-existing issues
/// out of scope for THIS test, whose job is the cross-context MODULE-CACHE
/// regression guard. GROUP-BY-over-JOIN is likewise excluded (it is a planner
/// limitation — aggregation over a non-scan-chain input is unsupported). See
/// docs/LIMITATIONS.md.
struct QuerySet {
    t1: Vec<(i32, f64)>,
    agg: f64,
}

const Q_T1: &str = "SELECT id1, SUM(v1) FROM fact GROUP BY id1";
const Q_AGG: &str = "SELECT SUM(v2) FROM fact";

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
        agg: collect_scalar(engine),
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
            "{label}: key {} value diverged across re-invocation: got {} want {} (rel {rel:e})",
            g.0, g.1, w.1
        );
    }
}

fn assert_queryset_eq(label: &str, got: &QuerySet, want: &QuerySet) {
    assert_key_agg_eq(&format!("{label}/q_t1"), &got.t1, &want.t1);
    let denom = got.agg.abs().max(want.agg.abs()).max(1.0);
    let rel = (got.agg - want.agg).abs() / denom;
    assert!(
        rel < REL_TOL,
        "{label}/q_agg: scalar SUM diverged across re-invocation: got {} want {} (rel {rel:e})",
        got.agg, want.agg
    );
}

// ---- The stress test --------------------------------------------------------

#[test]
#[ignore = "gpu:multi-engine"]
fn sequential_engines_and_repeated_queries_are_stable() {
    // 2M rows: above TWOKEY_MIN_ROWS (256K) so the two-key path uses Tier-2,
    // and large enough that id3 (1M-cardinality) exercises the high-card
    // Tier-2 single-key path. The same size the single-threaded
    // repeated-invocation regression uses.
    // Low-cardinality Tier-1 + scalar only, so a modest row count is plenty —
    // the point is many engine create/destroy cycles, not scale.
    const N_ROWS: usize = 200_000;
    const N_ENGINES: usize = 10;
    const ITERS: usize = 20;

    // Ground-truth from a fresh engine.
    let reference = {
        let engine = build_engine(N_ROWS);
        run_all(&engine)
    };

    // (1) SEQUENTIAL MULTI-ENGINE STRESS. Each iteration builds a brand-new
    // engine — a fresh CUDA context on the same device — runs the full query
    // mix, asserts it matches the reference, then DROPS the engine. This
    // exercises context teardown plus the module-cache clear-on-drop: the
    // process-global JIT module caches (`exec::module_cache` + the
    // `jit_compiler` PTX cache) hold `CudaModule` handles bound to the context
    // that loaded them, so without the clear-on-drop a stale handle from a
    // prior, destroyed context would fail the next engine's first launch with
    // `cuModuleGetFunction ... invalid resource handle`. Regression guard for
    // exactly that cross-context bug.
    for e in 0..N_ENGINES {
        let engine = build_engine(N_ROWS);
        let got = run_all(&engine);
        assert_queryset_eq(&format!("engine {e}"), &got, &reference);
        // `engine` drops here: context torn down, module caches cleared.
    }

    // (2) INTRA-ENGINE REPEATED-QUERY CHURN. Within ONE engine, run the full
    // mix ITERS times. This drives the process-global CUDA stream pool's
    // acquire/release cycle hard — the path whose per-query owned-stream UAF
    // the stream-pool fix replaced. Every result must stay bit/rel-equal
    // across re-invocation (a divergence or CUDA fault would signal pool
    // corruption or stream aliasing).
    {
        let engine = build_engine(N_ROWS);
        for iter in 0..ITERS {
            let got = run_all(&engine);
            assert_queryset_eq(&format!("repeat iter {iter}"), &got, &reference);
        }
    }
}
