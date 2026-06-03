// SPDX-License-Identifier: Apache-2.0

//! High-cardinality threshold microbench for `ORDER BY <utf8>`.
//!
//! Background
//! ----------
//! Stage 5 of the GPU sort feature gates the dictionary-encoded sort path
//! behind a `HIGH_CARDINALITY_THRESHOLD = 0.6` sample: if more than 60 % of
//! the sampled rows are distinct, the executor falls back to a host-side
//! sort instead of building a dictionary that's nearly as wide as the
//! source column. The 0.6 figure is a heuristic — this bench exists to put
//! a measured-throughput table next to it so a future tuner can pick a
//! smarter cutoff.
//!
//! Shape
//! -----
//! For each cardinality ratio in [0.01, 0.10, 0.30, 0.60, 0.90], synthesise
//! a 1 M-row Utf8 column with that many distinct values, then time a
//! `SELECT col FROM t ORDER BY col` end-to-end through the engine.
//!
//! Run policy
//! ----------
//! The bench is `#[ignore]`d at the criterion-group level: it spins up a
//! CUDA context and uploads ~50 MB per cardinality, which is heavy enough
//! that you don't want it firing on every `cargo bench`. Invoke explicitly:
//!
//! ```text
//! BOLT_BENCH_THRESHOLD=1 cargo bench --bench utf8_sort_bench
//! ```
//!
//! When `BOLT_BENCH_THRESHOLD` is unset the bench body short-circuits to a
//! single zero-work `b.iter(|| ())` so criterion's HTML report still
//! renders without GPU work.

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

/// Row count per bench iteration. 1 M is large enough that the per-iter
/// throughput dominates host-side fixture overhead but small enough to run
/// in seconds on a single GPU.
const N_ROWS: usize = 1_000_000;

/// Cardinality ratios sampled by the bench. The 0.6 entry sits right on the
/// Stage-5 `HIGH_CARDINALITY_THRESHOLD` so the cliff (if any) is visible in
/// the report.
const CARDINALITY_RATIOS: &[f64] = &[0.01, 0.10, 0.30, 0.60, 0.90];

/// Measurement window per case. Kept short relative to `query_benchmarks`
/// because each iteration uploads ~50 MB of column data — we want enough
/// samples for criterion's CI to be tight but not enough to wedge a CI run.
const MEASUREMENT_SECS: u64 = 8;

/// Generate a `StringArray` of `n_rows` synthetic strings with approximately
/// `ratio * n_rows` distinct values. The values are picked round-robin from
/// a fixed pool so the histogram is uniform — the worst case for the
/// fall-back gate's sampling step.
fn synth_utf8(n_rows: usize, ratio: f64) -> StringArray {
    let distinct = ((n_rows as f64 * ratio).round() as usize).max(1);
    let pool: Vec<String> = (0..distinct).map(|i| format!("v{i:08}")).collect();
    let strs: Vec<&str> = (0..n_rows).map(|i| pool[i % distinct].as_str()).collect();
    StringArray::from(strs)
}

/// Build a single-column `RecordBatch` named `col` of `n_rows` Utf8 values
/// with the requested cardinality ratio.
fn synth_batch(n_rows: usize, ratio: f64) -> RecordBatch {
    let arr = synth_utf8(n_rows, ratio);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "col",
        ArrowDataType::Utf8,
        false,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("build batch")
}

/// Whether the bench should actually exercise the engine. Off by default
/// so `cargo bench` on a host without a GPU still completes (criterion
/// will record a near-zero baseline rather than panic).
fn bench_enabled() -> bool {
    std::env::var("BOLT_BENCH_THRESHOLD")
        .map(|v| !v.is_empty() && v != "0")
        .unwrap_or(false)
}

fn bench_utf8_threshold(c: &mut Criterion) {
    let mut g = c.benchmark_group("utf8_sort_threshold");
    g.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    g.throughput(Throughput::Elements(N_ROWS as u64));

    let enabled = bench_enabled();

    for &ratio in CARDINALITY_RATIOS {
        let id = BenchmarkId::new("order_by_col", format!("ratio_{:.2}", ratio));
        g.bench_with_input(id, &ratio, |b, &ratio| {
            if !enabled {
                // Short-circuit: keeps `cargo bench` cheap on hosts without
                // a GPU. Criterion still records a (~0 ns) row in the HTML
                // report so the harness shape is visible end-to-end.
                b.iter(|| black_box(()));
                return;
            }

            // GPU-enabled path. Imports are inside the bench closure so
            // that a `cargo bench` on a host that compiled without the
            // CUDA toolchain still gets through the harness — the symbols
            // resolve at link time regardless of whether they ever run.
            use craton_bolt::exec::Engine;

            let batch = synth_batch(N_ROWS, ratio);
            let mut engine = match Engine::new() {
                Ok(e) => e,
                Err(_) => {
                    // No CUDA device available — fall back to the zero-work
                    // path rather than fail the whole bench binary.
                    b.iter(|| black_box(()));
                    return;
                }
            };
            engine.register_table("t", batch).expect("register");

            b.iter(|| {
                let h = engine.sql("SELECT col FROM t ORDER BY col").expect("sql");
                black_box(h);
            });
        });
    }

    g.finish();
}

criterion_group!(benches, bench_utf8_threshold);
criterion_main!(benches);
