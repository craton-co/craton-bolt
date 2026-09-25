// SPDX-License-Identifier: Apache-2.0

//! Real-device row-scaling and VRAM-pressure benchmark.
//!
//! The default sweep is intentionally bounded. Set
//! `BOLT_SCALING_ROWS=1000,10000,100000,1000000,10000000,100000000` to
//! override it, and set `BOLT_VRAM_PRESSURE_ROWS=<rows>` to add one
//! near-capacity point chosen for the installed GPU. No device work runs
//! unless `BOLT_BENCH_GPU=1`.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};
use craton_bolt::Engine;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn configured_rows() -> Vec<usize> {
    let mut rows = std::env::var("BOLT_SCALING_ROWS")
        .ok()
        .map(|raw| {
            raw.split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|&n| n > 0)
                .collect::<Vec<_>>()
        })
        .filter(|rows| !rows.is_empty())
        .unwrap_or_else(|| vec![1_000, 10_000, 100_000, 1_000_000, 10_000_000]);
    if let Ok(n) = std::env::var("BOLT_VRAM_PRESSURE_ROWS")
        .unwrap_or_default()
        .parse::<usize>()
    {
        if n > 0 {
            rows.push(n);
        }
    }
    rows.sort_unstable();
    rows.dedup();
    rows
}

fn fixture(n: usize) -> RecordBatch {
    let region = Int32Array::from_iter_values((0..n).map(|i| (i % 1024) as i32));
    let price = Float64Array::from_iter_values((0..n).map(|i| i as f64 + 1.0));
    let tax = Float64Array::from_iter_values((0..n).map(|i| 0.05 + (i % 7) as f64 * 0.001));
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("region_id", DataType::Int32, false),
            Field::new("price", DataType::Float64, false),
            Field::new("tax", DataType::Float64, false),
        ])),
        vec![Arc::new(region), Arc::new(price), Arc::new(tax)],
    )
    .expect("scaling fixture")
}

fn scaling(c: &mut Criterion) {
    if std::env::var("BOLT_BENCH_GPU").as_deref() != Ok("1") {
        eprintln!("scaling_benchmarks: set BOLT_BENCH_GPU=1 to run device measurements");
        return;
    }

    let mut group = c.benchmark_group("gpu_scaling");
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);

    for n in configured_rows() {
        let mut engine = Engine::new().expect("CUDA engine");
        engine
            .register_table("scale", fixture(n))
            .expect("register scaling fixture");
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("fused_filter_projection", n),
            &n,
            |b, _| {
                b.iter(|| {
                    let result = engine
                        .sql(
                            "SELECT price * tax + 1.0 FROM scale \
                         WHERE region_id >= 128 AND region_id < 896",
                        )
                        .expect("scaling query");
                    black_box(result.num_rows())
                })
            },
        );
        drop(engine);
    }
    group.finish();
}

criterion_group!(benches, scaling);
criterion_main!(benches);
