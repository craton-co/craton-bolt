// SPDX-License-Identifier: Apache-2.0

//! Craton Patina micro-benchmarks. Measures plan + codegen latency on CPU; gated
//! end-to-end GPU execution behind `PATINA_BENCH_GPU=1`.
//!
//! The `bench_polars` group provides a head-to-head comparison against Polars
//! using the same three queries and row count as `bench_engine_execute`. Polars
//! runs on its default rayon-based thread pool (all CPU cores), so this is a
//! fair fight: Craton Patina-on-GPU vs Polars-on-all-CPU-cores — exactly the
//! comparison we want to publish.
//!
//! Workload sizing
//! ---------------
//! `BENCH_ROWS = 50_000_000` rows × 3 Float64 columns is ~1.2 GB of dataset
//! material per fixture. The heavy arithmetic queries chain ~20 binary
//! operations per row, which lifts each criterion iteration well into the
//! hundreds-of-milliseconds range — far past the per-iter floor where
//! warm-up jitter and timer resolution dominate. Combined with a 20 s
//! `measurement_time` per benchmark, criterion collects enough samples for
//! a tight (<5 % CI) reported median.

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use craton_patina::jit::compile_ptx;
use craton_patina::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, PhysicalPlan, Schema,
};

/// Row count for the data-bearing benchmarks. Sized so a heavy arithmetic
/// query takes hundreds of ms per iteration even on a Polars/CPU baseline.
const BENCH_ROWS: usize = 50_000_000;

/// Criterion measurement window per data-bearing benchmark. Long enough to
/// average over ~10–50 iterations at this workload size.
const MEASUREMENT_SECS: u64 = 20;

// ---- Fixtures --------------------------------------------------------------

fn schema() -> Schema {
    Schema::new(vec![
        Field { name: "region_id".into(), dtype: DataType::Int32, nullable: false },
        Field { name: "price".into(), dtype: DataType::Float64, nullable: false },
        Field { name: "tax".into(), dtype: DataType::Float64, nullable: false },
    ])
}

fn provider() -> MemTableProvider {
    MemTableProvider::new().with_table("sales", schema())
}

fn batch(n: usize) -> RecordBatch {
    let region: Int32Array = (0..n as i32).map(|i| i % 4).collect();
    let price: Float64Array = (0..n).map(|i| (i + 1) as f64).collect();
    let tax: Float64Array = (0..n).map(|_| 0.0825_f64).collect();
    let s = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("region_id", ArrowDataType::Int32, false),
        ArrowField::new("price", ArrowDataType::Float64, false),
        ArrowField::new("tax", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(s, vec![Arc::new(region), Arc::new(price), Arc::new(tax)]).unwrap()
}

// ---- Query texts -----------------------------------------------------------
//
// Light queries are kept for plan / lower / ptx_gen micro-benches (they don't
// touch GPU memory and don't depend on workload size).
const Q_PROJ_LIGHT: &str = "SELECT price FROM sales";
const Q_ARITH_LIGHT: &str = "SELECT price * tax FROM sales";
const Q_FILTERED_LIGHT: &str = "SELECT price FROM sales WHERE region_id = 1";

/// Heavy projection: trivially passthrough — included so the projection
/// case still measures pure orchestration cost at 50 M rows.
const Q_PROJ_HEAVY: &str = "SELECT price FROM sales";

/// Heavy arithmetic: 20 binary ops per row (10 multiplies + 10 adds/subs),
/// all folded into one expression. Chosen so the per-row FLOPs dominate
/// per-row launch overhead and PCIe-D2H of a single output column.
const Q_ARITH_HEAVY: &str =
    "SELECT \
        price * tax \
        + price * 0.01 \
        + tax * 0.02 \
        + price * 1.001 \
        - tax * 0.999 \
        + price * 1.05 \
        - tax * 0.05 \
        + price * 0.5 \
        + tax * 0.7 \
        + price * tax * 1.1 \
        - price * 0.25 \
     FROM sales";

/// Heavy filter: select ~25 % of rows AND apply a multi-op arithmetic
/// expression on the surviving column so the post-compaction kernel and the
/// arithmetic kernel both do real work.
const Q_FILTERED_HEAVY: &str =
    "SELECT \
        price * tax \
        + price * 1.01 \
        - tax * 0.5 \
        + price * 0.7 \
     FROM sales WHERE region_id = 1";

fn polars_df(n: usize) -> polars::prelude::DataFrame {
    use polars::prelude::*;
    let region: Vec<i32> = (0..n as i32).map(|i| i % 4).collect();
    let price: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let tax: Vec<f64> = (0..n).map(|_| 0.0825_f64).collect();
    df!(
        "region_id" => region,
        "price" => price,
        "tax" => tax,
    )
    .expect("polars df")
}

// ---- Plan / lower / ptx_gen micro-benches (no GPU, no big batch) -----------

fn bench_plan(c: &mut Criterion) {
    let p = provider();
    let mut g = c.benchmark_group("plan");
    for (name, sql) in [
        ("proj", Q_PROJ_LIGHT),
        ("arith", Q_ARITH_LIGHT),
        ("filtered", Q_FILTERED_LIGHT),
    ] {
        g.bench_function(name, |b| {
            b.iter(|| {
                let plan = parse_sql(black_box(sql), &p).unwrap();
                black_box(plan)
            })
        });
    }
    g.finish();
}

fn bench_lower(c: &mut Criterion) {
    let p = provider();
    let mut g = c.benchmark_group("lower");
    for (name, sql) in [
        ("proj", Q_PROJ_LIGHT),
        ("arith", Q_ARITH_LIGHT),
        ("filtered", Q_FILTERED_LIGHT),
    ] {
        let plan = parse_sql(sql, &p).unwrap();
        g.bench_function(name, |b| {
            b.iter(|| {
                let phys = lower_physical(black_box(&plan)).unwrap();
                black_box(phys)
            })
        });
    }
    g.finish();
}

fn bench_ptx_gen(c: &mut Criterion) {
    let p = provider();
    let mut g = c.benchmark_group("ptx_gen");
    for (name, sql) in [
        ("proj", Q_PROJ_LIGHT),
        ("arith", Q_ARITH_LIGHT),
        ("filtered", Q_FILTERED_LIGHT),
    ] {
        let plan = parse_sql(sql, &p).unwrap();
        let phys = lower_physical(&plan).unwrap();
        let PhysicalPlan::Projection { kernel, .. } = phys else { panic!() };
        g.bench_function(name, |b| {
            b.iter(|| {
                let ptx = compile_ptx(black_box(&kernel), "patina_kernel").unwrap();
                black_box(ptx)
            })
        });
    }
    g.finish();
}

/// CPU reference baseline — measures the same heavy arithmetic in plain Rust.
/// Single-threaded, written as a tight loop the optimiser can autovectorise.
fn bench_cpu_reference(c: &mut Criterion) {
    let batch = batch(BENCH_ROWS);
    let price = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .clone();
    let tax = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .clone();

    let mut g = c.benchmark_group("cpu_reference");
    g.throughput(Throughput::Elements(BENCH_ROWS as u64));
    g.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    g.bench_function("heavy_arith", |b| {
        b.iter(|| {
            let mut out: Vec<f64> = Vec::with_capacity(BENCH_ROWS);
            let prices = price.values();
            let taxes = tax.values();
            for i in 0..BENCH_ROWS {
                let p = prices[i];
                let t = taxes[i];
                let v = p * t
                    + p * 0.01
                    + t * 0.02
                    + p * 1.001
                    - t * 0.999
                    + p * 1.05
                    - t * 0.05
                    + p * 0.5
                    + t * 0.7
                    + p * t * 1.1
                    - p * 0.25;
                out.push(v);
            }
            black_box(out)
        })
    });

    g.finish();
}

/// Polars baseline — same three heavy queries as `bench_engine_execute`,
/// expressed in Polars' lazy DataFrame API. Multi-threaded by default.
fn bench_polars(c: &mut Criterion) {
    use polars::prelude::*;
    let df = polars_df(BENCH_ROWS);

    let mut g = c.benchmark_group("polars");
    g.throughput(Throughput::Elements(BENCH_ROWS as u64));
    g.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    // Query 1: SELECT price FROM sales  (passthrough)
    g.bench_function("proj", |b| {
        b.iter(|| {
            let out = df
                .clone()
                .lazy()
                .select([col("price")])
                .collect()
                .expect("polars collect");
            black_box(out)
        })
    });

    // Query 2: heavy arithmetic chain mirroring Q_ARITH_HEAVY.
    g.bench_function("arith", |b| {
        b.iter(|| {
            let p = col("price");
            let t = col("tax");
            let expr = p.clone() * t.clone()
                + p.clone() * lit(0.01)
                + t.clone() * lit(0.02)
                + p.clone() * lit(1.001)
                - t.clone() * lit(0.999)
                + p.clone() * lit(1.05)
                - t.clone() * lit(0.05)
                + p.clone() * lit(0.5)
                + t.clone() * lit(0.7)
                + p.clone() * t.clone() * lit(1.1)
                - p.clone() * lit(0.25);
            let out = df
                .clone()
                .lazy()
                .select([expr.alias("y")])
                .collect()
                .expect("polars collect");
            black_box(out)
        })
    });

    // Query 3: filter + heavy arithmetic mirroring Q_FILTERED_HEAVY.
    g.bench_function("filtered", |b| {
        b.iter(|| {
            let p = col("price");
            let t = col("tax");
            let expr = p.clone() * t.clone()
                + p.clone() * lit(1.01)
                - t.clone() * lit(0.5)
                + p.clone() * lit(0.7);
            let out = df
                .clone()
                .lazy()
                .filter(col("region_id").eq(lit(1i32)))
                .select([expr.alias("y")])
                .collect()
                .expect("polars collect");
            black_box(out)
        })
    });

    g.finish();
}

/// End-to-end GPU benchmark. Skipped unless `PATINA_BENCH_GPU=1`.
fn bench_engine_execute(c: &mut Criterion) {
    if std::env::var("PATINA_BENCH_GPU").ok().as_deref() != Some("1") {
        eprintln!("skipping engine_execute (set PATINA_BENCH_GPU=1 to enable)");
        return;
    }
    use craton_patina::Engine;
    let mut engine = match Engine::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("skipping engine_execute: CUDA context init failed: {err}");
            return;
        }
    };
    let b = batch(BENCH_ROWS);
    engine.register_table("sales", b).unwrap();

    let mut g = c.benchmark_group("engine_execute");
    g.throughput(Throughput::Elements(BENCH_ROWS as u64));
    g.measurement_time(Duration::from_secs(MEASUREMENT_SECS));

    for (name, sql) in [
        ("proj", Q_PROJ_HEAVY),
        ("arith", Q_ARITH_HEAVY),
        ("filtered", Q_FILTERED_HEAVY),
    ] {
        g.bench_function(name, |b| {
            b.iter_batched(
                || sql,
                |sql| {
                    let h = engine.sql(sql).unwrap();
                    black_box(h)
                },
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

criterion_group!(
    benches,
    bench_plan,
    bench_lower,
    bench_ptx_gen,
    bench_cpu_reference,
    bench_polars,
    bench_engine_execute,
);
criterion_main!(benches);
