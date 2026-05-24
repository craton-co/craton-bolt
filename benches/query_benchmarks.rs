// SPDX-License-Identifier: Apache-2.0

//! Javelin micro-benchmarks. Measures plan + codegen latency on CPU; gated
//! end-to-end GPU execution behind `JAVELIN_BENCH_GPU=1`.
//!
//! The `bench_polars` group provides a head-to-head comparison against Polars
//! using the same three queries and row count as `bench_engine_execute`. Polars
//! runs on its default rayon-based thread pool (all CPU cores), so this is a
//! fair fight: Javelin-on-GPU vs Polars-on-all-CPU-cores — exactly the
//! comparison we want to publish.

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use javelin::jit::compile_ptx;
use javelin::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, PhysicalPlan, Schema,
};

const BENCH_ROWS: usize = 1_000_000;

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

const Q_PROJ: &str = "SELECT price FROM sales";
const Q_ARITH: &str = "SELECT price * tax FROM sales";
const Q_FILTERED: &str = "SELECT price FROM sales WHERE region_id = 1";

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

// ---- Bench groups ----------------------------------------------------------

fn bench_plan(c: &mut Criterion) {
    let p = provider();
    let mut g = c.benchmark_group("plan");
    for (name, sql) in [("proj", Q_PROJ), ("arith", Q_ARITH), ("filtered", Q_FILTERED)] {
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
    for (name, sql) in [("proj", Q_PROJ), ("arith", Q_ARITH), ("filtered", Q_FILTERED)] {
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
    for (name, sql) in [("proj", Q_PROJ), ("arith", Q_ARITH), ("filtered", Q_FILTERED)] {
        let plan = parse_sql(sql, &p).unwrap();
        let phys = lower_physical(&plan).unwrap();
        let PhysicalPlan::Projection { kernel, .. } = phys else { panic!() };
        g.bench_function(name, |b| {
            b.iter(|| {
                let ptx = compile_ptx(black_box(&kernel), "javelin_kernel").unwrap();
                black_box(ptx)
            })
        });
    }
    g.finish();
}

/// CPU reference baseline — measures the same arithmetic in plain Rust.
/// Useful as a sanity-check ceiling for what hand-tuned CPU code can do.
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

    g.bench_function("price_times_tax", |b| {
        b.iter(|| {
            let mut out = Vec::with_capacity(BENCH_ROWS);
            for i in 0..BENCH_ROWS {
                out.push(price.value(i) * tax.value(i));
            }
            black_box(out)
        })
    });

    g.finish();
}

/// Polars baseline — same three queries as `bench_engine_execute`, against the
/// same row count, built with identical column shapes. Polars uses its default
/// thread pool (rayon, all CPU cores), so this is a head-to-head between
/// Javelin-on-GPU and Polars-on-all-CPU-cores.
fn bench_polars(c: &mut Criterion) {
    use polars::prelude::*;
    let df = polars_df(BENCH_ROWS);

    let mut g = c.benchmark_group("polars");
    g.throughput(Throughput::Elements(BENCH_ROWS as u64));

    // Query 1: SELECT price FROM sales
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

    // Query 2: SELECT price * tax FROM sales
    g.bench_function("arith", |b| {
        b.iter(|| {
            let out = df
                .clone()
                .lazy()
                .select([(col("price") * col("tax")).alias("price_tax")])
                .collect()
                .expect("polars collect");
            black_box(out)
        })
    });

    // Query 3: SELECT price FROM sales WHERE region_id = 1
    // `lit(1i32)` keeps the literal as Int32 so it matches `region_id`'s dtype
    // and avoids a needless upcast — keeps the comparison fair.
    g.bench_function("filtered", |b| {
        b.iter(|| {
            let out = df
                .clone()
                .lazy()
                .filter(col("region_id").eq(lit(1i32)))
                .select([col("price")])
                .collect()
                .expect("polars collect");
            black_box(out)
        })
    });

    g.finish();
}

/// End-to-end GPU benchmark. Skipped unless `JAVELIN_BENCH_GPU=1`.
fn bench_engine_execute(c: &mut Criterion) {
    if std::env::var("JAVELIN_BENCH_GPU").ok().as_deref() != Some("1") {
        eprintln!("skipping engine_execute (set JAVELIN_BENCH_GPU=1 to enable)");
        return;
    }
    use javelin::Engine;
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
    g.measurement_time(Duration::from_secs(8));

    for (name, sql) in [("proj", Q_PROJ), ("arith", Q_ARITH), ("filtered", Q_FILTERED)] {
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
