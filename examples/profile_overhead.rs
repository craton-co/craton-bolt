// SPDX-License-Identifier: Apache-2.0
//
//! Per-query overhead profiler for the compute_benchmarks queries.
//!
//! `compute_benchmarks` measured Craton Bolt at ~70-100 ms/query on a resident
//! 10M-row table — ~100x below the GPU's bandwidth roofline — which means the
//! wall-clock is dominated by a fixed per-query overhead, not GPU compute. This
//! binary attributes that floor by timing, on a table uploaded ONCE:
//!
//!   * `explain_sql(q)` — parse + plan + lower (+ render), NO device work.
//!     A proxy for the CPU frontend cost.
//!   * `sql(q)`         — the full path including execute (launch + sync + D2H
//!     + any per-query device re-marshaling).
//!
//! `execute ≈ full - frontend`. Run with the same feature set as the bench:
//!   BOLT_BENCH_GPU=1 cargo run --release --no-default-features --features cudarc \
//!       --example profile_overhead

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Float64Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

const ROWS: usize = 10_000_000;
const WARMUP: usize = 3;
const ITERS: usize = 30;

const QUERIES: &[(&str, &str)] = &[
    ("c1_fma_sum", "SELECT SUM(a*b + c*d) FROM x"),
    ("c2_poly4_sum", "SELECT SUM(a*a*a*a - a*a*a + a*a) FROM x"),
    ("c3_sphere_filter", "SELECT SUM(a) FROM x WHERE a*a + b*b + c*c < 1.0"),
    ("c4_weighted_sum", "SELECT SUM(a*0.4 + b*0.3 + c*0.2 + d*0.1) FROM x"),
    ("c5_filtered_fma", "SELECT SUM(a*b) FROM x WHERE c*c + d*d < 0.5"),
];

#[inline]
fn unit(i: usize, mult: u64) -> f64 {
    let h = (i as u64).wrapping_mul(mult) as u32;
    (h as f64) / (u32::MAX as f64 + 1.0)
}

fn arrow_batch(n: usize) -> RecordBatch {
    let s = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Float64, false),
        Field::new("b", DataType::Float64, false),
        Field::new("c", DataType::Float64, false),
        Field::new("d", DataType::Float64, false),
    ]));
    let ac: Float64Array = (0..n).map(|i| unit(i, 0x9E37_79B1)).collect();
    let bc: Float64Array = (0..n).map(|i| unit(i, 0x0001_9E27)).collect();
    let cc: Float64Array = (0..n).map(|i| unit(i, 0x85EB_CA77)).collect();
    let dc: Float64Array = (0..n).map(|i| unit(i, 0xC2B2_AE3D)).collect();
    RecordBatch::try_new(
        s,
        vec![Arc::new(ac), Arc::new(bc), Arc::new(cc), Arc::new(dc)],
    )
    .unwrap()
}

fn mean_ms<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP {
        f();
    }
    let t = Instant::now();
    for _ in 0..ITERS {
        f();
    }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn main() {
    if std::env::var("BOLT_BENCH_GPU").ok().as_deref() != Some("1") {
        eprintln!("set BOLT_BENCH_GPU=1 to run the device path");
        return;
    }
    let mut engine = craton_bolt::Engine::new().expect("engine");
    eprintln!("[profile] uploading {ROWS} rows (one-time H2D)…");
    engine
        .register_table("x", arrow_batch(ROWS))
        .expect("register");

    // Sanity: one query, print its scalar so we know execution is real.
    let h = engine.sql(QUERIES[0].1).expect("sql");
    eprintln!(
        "[profile] warm result {} = {:?}",
        QUERIES[0].0,
        h.record_batch().column(0)
    );

    println!(
        "\n{:<18} {:>12} {:>12} {:>12}",
        "query", "frontend_ms", "full_ms", "execute_ms"
    );
    println!("{}", "-".repeat(56));
    for (name, q) in QUERIES {
        let frontend = mean_ms(|| {
            let _ = engine.explain_sql(q).expect("explain");
        });
        let full = mean_ms(|| {
            let _ = engine.sql(q).expect("sql");
        });
        let execute = full - frontend;
        println!("{name:<18} {frontend:>12.3} {full:>12.3} {execute:>12.3}");
    }
}
