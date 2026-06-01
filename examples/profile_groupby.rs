// SPDX-License-Identifier: Apache-2.0
//
//! GROUP BY per-query overhead profiler (h2o.ai groupby shapes).
//!
//! The scalar-aggregate path was just moved on-device (`try_execute_resident`),
//! eliminating a ~480 MB/query host round-trip. The GROUP BY executors still
//! re-upload key+value columns from the host batch every query. Before
//! refactoring ~18 tier executors, this binary quantifies HOW MUCH of GROUP BY
//! time is that re-upload vs the hash-aggregation kernels themselves — the
//! h2o groupby is memory-bound hashing (real GPU work), unlike the scalar case
//! where the round-trip was ~100% of the time.
//!
//! Method: time `explain_sql` (frontend) vs `sql` (full) per query on a
//! resident 10M table, and measure this machine's H2D bandwidth via a timed
//! `replace_table`, so per-query upload bytes can be turned into an estimated
//! upload-ms and an upload fraction of execute time.
//!
//!   BOLT_BENCH_GPU=1 cargo run --release --no-default-features --features cudarc \
//!       --example profile_groupby

use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType, Field, Schema};

const ROWS: usize = 10_000_000;
const WARMUP: usize = 3;
const ITERS: usize = 20;
const ID1_CARD: i32 = 100;
const ID2_CARD: i32 = 10_000;
const ID3_CARD: i32 = 1_000_000;

// (name, sql, est. upload MB/query = sum of key+value column bytes at 10M)
const QUERIES: &[(&str, &str, f64)] = &[
    ("q1_low_card_sum", "SELECT id1, SUM(v1) FROM x GROUP BY id1", 120.0),
    ("q2_med_card_2sum", "SELECT id2, SUM(v1), SUM(v2) FROM x GROUP BY id2", 200.0),
    ("q3_two_key_sum", "SELECT id1, id2, SUM(v1) FROM x GROUP BY id1, id2", 160.0),
    ("q4_low_card_3avg", "SELECT id1, AVG(v1), AVG(v2), AVG(v3) FROM x GROUP BY id1", 280.0),
    ("q5_high_card_sum", "SELECT id3, SUM(v1) FROM x GROUP BY id3", 120.0),
];

fn id1(i: usize) -> i32 { ((i.wrapping_mul(2_654_435_761)) as i32).rem_euclid(ID1_CARD) }
fn id2(i: usize) -> i32 { ((i.wrapping_mul(40_503)) as i32).rem_euclid(ID2_CARD) }
fn id3(i: usize) -> i32 {
    ((i.wrapping_mul(11_400_714_819_323_198_485_u64 as usize)) as i32).rem_euclid(ID3_CARD)
}
fn v1(i: usize) -> f64 { ((i.wrapping_mul(7) as i32).rem_euclid(5) + 1) as f64 }
fn v2(i: usize) -> f64 { ((i.wrapping_mul(13) as i32).rem_euclid(15) + 1) as f64 }
fn v3(i: usize) -> f64 { ((i.wrapping_mul(17) as i32).rem_euclid(10_000)) as f64 / 100.0 }

fn arrow_batch(n: usize) -> RecordBatch {
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
            Arc::new(id1c), Arc::new(id2c), Arc::new(id3c),
            Arc::new(v1c), Arc::new(v2c), Arc::new(v3c),
        ],
    )
    .unwrap()
}

fn mean_ms<F: FnMut()>(mut f: F) -> f64 {
    for _ in 0..WARMUP { f(); }
    let t = Instant::now();
    for _ in 0..ITERS { f(); }
    t.elapsed().as_secs_f64() * 1e3 / ITERS as f64
}

fn main() {
    if std::env::var("BOLT_BENCH_GPU").ok().as_deref() != Some("1") {
        eprintln!("set BOLT_BENCH_GPU=1 to run the device path");
        return;
    }
    let mut engine = craton_bolt::Engine::new().expect("engine");
    eprintln!("[gb-profile] uploading {ROWS} rows…");
    engine.register_table("x", arrow_batch(ROWS)).expect("register");

    // --- Correctness gate: q1 (SUM(v1) GROUP BY id1) through engine.sql (the
    //     resident dispatch) vs a host HashMap reference over the same data. ---
    {
        let h = engine.sql(QUERIES[0].1).expect("q1");
        let batch = h.record_batch();
        let keys = batch.column(0).as_any().downcast_ref::<Int32Array>().expect("id1 col");
        let sums = batch.column(1).as_any().downcast_ref::<Float64Array>().expect("sum col");
        let mut got = std::collections::HashMap::<i32, f64>::new();
        for i in 0..batch.num_rows() {
            got.insert(keys.value(i), sums.value(i));
        }
        let mut want = std::collections::HashMap::<i32, f64>::new();
        for i in 0..ROWS {
            *want.entry(id1(i)).or_insert(0.0) += v1(i);
        }
        assert_eq!(got.len(), want.len(), "q1 group count mismatch: {} vs {}", got.len(), want.len());
        let mut max_rel = 0.0f64;
        for (k, &w) in &want {
            let g = *got.get(k).unwrap_or_else(|| panic!("q1 missing group {k}"));
            let rel = (g - w).abs() / w.abs().max(1.0);
            max_rel = max_rel.max(rel);
        }
        assert!(max_rel < 1e-9, "q1 resident result diverges from host ref: max_rel={max_rel:e}");
        eprintln!("[gb-profile] q1 correctness ✓ ({} groups, max_rel={max_rel:e})", got.len());
    }

    // warm
    let _ = engine.sql(QUERIES[0].1).expect("warm");

    // Per-query execute_ms. Each query measured in isolation. Upload cost is
    // reported separately by the BOLT_GB_PROFILE=1 instrumentation inside the
    // executors (exact H2D ms, not a bandwidth estimate). The `up_mb` column
    // is the theoretical key+value bytes for reference.
    println!(
        "{:<18} {:>10} {:>10} {:>10} {:>9}",
        "query", "front_ms", "full_ms", "exec_ms", "up_mb"
    );
    println!("{}", "-".repeat(62));
    let only = std::env::var("GB_ONLY").ok();
    for (name, q, up_mb) in QUERIES {
        if let Some(o) = &only {
            if !name.contains(o.as_str()) { continue; }
        }
        let frontend = mean_ms(|| { let _ = engine.explain_sql(q).expect("explain"); });
        let full = mean_ms(|| { let _ = engine.sql(q).expect("sql"); });
        let exec = full - frontend;
        println!("{name:<18} {frontend:>10.3} {full:>10.3} {exec:>10.3} {up_mb:>8.0}MB");
    }
}
