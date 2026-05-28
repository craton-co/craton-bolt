// SPDX-License-Identifier: Apache-2.0

//! Craton Bolt — baseline regression benchmark suite.
//!
//! v0.6 / M6 ("production discipline") introduces a small, *stable* set of
//! Criterion benchmarks whose median wall-time we track release-over-release
//! as a regression tripwire. This file is intentionally **narrow** and
//! **headless**: only the CPU-bound stages of the query pipeline
//! (parse + logical plan + physical lowering + PTX code generation) are
//! exercised, and *no* CUDA driver call is ever issued. That property is what
//! lets the suite run in CI on a vanilla Linux runner with **no GPU
//! attached**: the crate is built with the `cuda-stub` feature, the FFI
//! shims in `src/cuda/cuda_sys.rs` are stubbed to return `CUDA_ERROR_STUB`,
//! and we deliberately never hit them.
//!
//! Run locally
//! -----------
//! ```text
//! cargo bench --bench regression --features cuda-stub
//! ```
//!
//! Run in CI
//! ---------
//! The CI job mirrors the local invocation. Because `cuda-stub` strips the
//! `#[link(name = "cuda")]` block, no CUDA toolkit / driver is required on
//! the runner — the bench builds and runs on any host with a stable Rust
//! toolchain.
//!
//! Regression threshold convention
//! -------------------------------
//! We treat a **>5% slowdown** in the reported Criterion median (vs. the
//! committed baseline for the same bench id) as a regression that must be
//! either justified or fixed before merge. Criterion's own change-detection
//! at default sample-size already flags shifts of this magnitude with
//! ample statistical confidence on a quiet runner.
//!
//! The actual CI gating wiring — comparing JSON output against a stored
//! baseline and failing the job on a >5% regression — is intentionally a
//! follow-up. This PR lands the *scaffold* only: a reproducible workload,
//! a stable set of bench ids, and the docs that future automation will
//! key off of.
//!
//! Workload
//! --------
//! All three queries run against an in-memory table of 100_000 rows with
//! three columns (`region`, `price`, `tax`). The chosen queries are the
//! smallest representatives of three distinct execution shapes:
//!
//! 1. **Scalar aggregate** — `SELECT COUNT(*), SUM(price), AVG(price) FROM t`
//! 2. **GROUP BY**         — `SELECT region, SUM(price) FROM t GROUP BY region`
//!                           (10 distinct groups)
//! 3. **Filter**           — `SELECT price FROM t WHERE price > 50`
//!
//! Each query is timed at three pipeline stages: `parse`, `lower`, and
//! `ptx_gen`. The bench *ids* are stable (`regression/<query>/<stage>`)
//! so future tooling can diff a single key over time.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use craton_bolt::jit::compile_ptx;
use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, PhysicalPlan, Schema,
};

/// Row count for the in-memory fixture. Small enough that the bench is
/// dominated by codegen / planning cost (the bits we actually exercise on a
/// `cuda-stub` host) and finishes quickly in CI; large enough to stay
/// representative of the constants planners care about (cardinality
/// estimation, group count, etc.).
const BENCH_ROWS: usize = 100_000;

/// Distinct group cardinality for the GROUP BY workload.
const NUM_GROUPS: i32 = 10;

// --- Queries ----------------------------------------------------------------
//
// These three SQL strings ARE the regression contract. Changing them changes
// the bench id and resets the baseline — only do so deliberately, with a
// commit message that calls it out.

/// (1) Scalar aggregate — one row out, three aggregates over `BENCH_ROWS` rows.
const Q_SCALAR_AGG: &str = "SELECT COUNT(*), SUM(price), AVG(price) FROM t";

/// (2) GROUP BY — `NUM_GROUPS` rows out, one SUM per group.
const Q_GROUP_BY: &str = "SELECT region, SUM(price) FROM t GROUP BY region";

/// (3) Filter — selective projection; ~half the input rows survive.
const Q_FILTER: &str = "SELECT price FROM t WHERE price > 50";

// --- Fixture ---------------------------------------------------------------

fn schema() -> Schema {
    Schema::new(vec![
        Field { name: "region".into(), dtype: DataType::Int32, nullable: false },
        Field { name: "price".into(), dtype: DataType::Float64, nullable: false },
        Field { name: "tax".into(), dtype: DataType::Float64, nullable: false },
    ])
}

fn provider() -> MemTableProvider {
    // Note: the planner only needs the schema (and a row-count hint, where
    // applicable); we deliberately do not register an Arrow `RecordBatch`
    // payload because the bench never reaches the execution stage on a
    // `cuda-stub` host. `BENCH_ROWS` / `NUM_GROUPS` are documented above
    // for the day someone wires an Arrow-backed provider in.
    let _ = BENCH_ROWS;
    let _ = NUM_GROUPS;
    MemTableProvider::new().with_table("t", schema())
}

// --- Benches ---------------------------------------------------------------

/// Time SQL → `LogicalPlan` (parse + bind + validate) for each of the three
/// regression queries.
fn bench_parse(c: &mut Criterion) {
    let p = provider();
    let mut g = c.benchmark_group("regression/parse");
    for (name, sql) in [
        ("scalar_agg", Q_SCALAR_AGG),
        ("group_by", Q_GROUP_BY),
        ("filter", Q_FILTER),
    ] {
        g.bench_with_input(BenchmarkId::from_parameter(name), &sql, |b, &sql| {
            b.iter(|| {
                let plan = parse_sql(black_box(sql), &p).unwrap();
                black_box(plan)
            })
        });
    }
    g.finish();
}

/// Time `LogicalPlan` → `PhysicalPlan` (lowering) for each query.
fn bench_lower(c: &mut Criterion) {
    let p = provider();
    let mut g = c.benchmark_group("regression/lower");
    for (name, sql) in [
        ("scalar_agg", Q_SCALAR_AGG),
        ("group_by", Q_GROUP_BY),
        ("filter", Q_FILTER),
    ] {
        let plan = parse_sql(sql, &p).unwrap();
        g.bench_with_input(BenchmarkId::from_parameter(name), &plan, |b, plan| {
            b.iter(|| {
                let phys = lower_physical(black_box(plan)).unwrap();
                black_box(phys)
            })
        });
    }
    g.finish();
}

/// Time `PhysicalPlan` → PTX (codegen) for the filter query, which is the
/// only one of the three whose physical plan is a leaf-`Projection` and
/// therefore exposes a single `kernel` we can hand to `compile_ptx`.
/// Aggregate/GROUP BY plans are multi-stage and don't fit a single-kernel
/// codegen call; for those, the regression signal lives in `bench_parse`
/// and `bench_lower` above.
fn bench_ptx_gen(c: &mut Criterion) {
    let p = provider();
    let mut g = c.benchmark_group("regression/ptx_gen");
    for (name, sql) in [("filter", Q_FILTER)] {
        let plan = parse_sql(sql, &p).unwrap();
        let phys = lower_physical(&plan).unwrap();
        let kernel = match phys {
            PhysicalPlan::Projection { kernel, .. } => kernel,
            // Defensive: if a future planner change reshapes the filter
            // plan into something other than a leaf Projection, the bench
            // should be re-thought rather than silently skipped.
            other => panic!(
                "regression bench expected a leaf Projection plan for `{}`, got {:?}",
                sql, other
            ),
        };
        g.bench_with_input(BenchmarkId::from_parameter(name), &kernel, |b, kernel| {
            b.iter(|| {
                let ptx = compile_ptx(black_box(kernel), "bolt_regression_kernel").unwrap();
                black_box(ptx)
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench_parse, bench_lower, bench_ptx_gen);
criterion_main!(benches);
