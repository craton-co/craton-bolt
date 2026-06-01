// SPDX-License-Identifier: Apache-2.0

//! **Compute-bound analytics — three-engine comparison (GPU-favorable).**
//!
//! Companion to `olap_benchmarks.rs`. The h2o.ai groupby suite measures hash
//! aggregation — irregular memory access + atomic contention with ~1 FLOP per
//! row — which is a GPU's *weakest* analytic path and a CPU engine's most
//! heavily tuned one. This suite instead measures the workloads a GPU is built
//! for:
//!
//!   * **Compute-bound reductions / projections** — many FMAs per row, so the
//!     device's thousands of FP64 lanes dominate (scalar SUMs of arithmetic
//!     expressions, polynomials, weighted sums).
//!   * **Compound-predicate filters** — per-row arithmetic predicates evaluated
//!     in parallel, then a masked reduce.
//!
//! Crucially every query is a SCALAR aggregate (or filtered scalar): there is
//! **no GROUP BY**, so no hash table, no scatter, no atomic contention — the
//! kernel is a dense, fully-coalesced scan over GPU-resident data. This is where
//! the device's ~10-20x memory-bandwidth and FLOP-throughput advantage shows.
//!
//! Same discipline as the groupby suite: every query is verified equal across
//! Craton Bolt, DuckDB, and Polars on a small fixture before any timed run
//! (cross-engine disagreement panics), then timed at h2o's "small" scale.
//! Gated on `BOLT_BENCH_GPU=1` for the device path.

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Float64Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

use craton_bolt::REL_TOL_TEST as REL_TOL;

const BENCH_ROWS: usize = 10_000_000;
const VERIFY_ROWS: usize = 100_000;
const MEASUREMENT_SECS: u64 = 20;

// --- Queries (no GROUP BY → no hash table; pure compute/bandwidth) ---------

/// 2 muls + 1 add + accumulate per row over two column pairs — classic FMA.
const C1: &str = "SELECT SUM(a*b + c*d) FROM x";
/// Degree-4 polynomial of a single column: highest arithmetic-intensity case
/// (6 muls + 2 add/sub per 8 bytes read).
const C2: &str = "SELECT SUM(a*a*a*a - a*a*a + a*a) FROM x";
/// Compound arithmetic predicate (point inside the unit sphere) + masked SUM:
/// 3 muls + 2 adds + compare per row, then reduce the survivors.
const C3: &str = "SELECT SUM(a) FROM x WHERE a*a + b*b + c*c < 1.0";
/// Weighted sum / dot-product-with-constants — 4 muls + 3 adds per row.
const C4: &str = "SELECT SUM(a*0.4 + b*0.3 + c*0.2 + d*0.1) FROM x";
/// Filtered FMA: arithmetic predicate gate + a product reduce.
const C5: &str = "SELECT SUM(a*b) FROM x WHERE c*c + d*d < 0.5";
/// **Compute-bound extreme**: a degree-12 Horner polynomial of a SINGLE
/// column — 12 muls + 12 adds (24 FLOPs) over just 8 bytes read per row, an
/// arithmetic intensity ~4x higher than `c2` and an order above `c1`/`c4`.
/// This is the case where the device's FP throughput, not memory bandwidth,
/// sets the pace — the clearest test of raw GPU compute. The coefficients
/// here are kept in lockstep with [`POLY12_COEFFS`] (Horner order: high
/// degree first), so the SQL string and the Polars expression evaluate the
/// identical polynomial.
const C6: &str = "SELECT SUM(((((((((((((0.13 * a + 0.11) * a + 0.17) * a + 0.19) \
    * a + 0.23) * a + 0.29) * a + 0.31) * a + 0.37) * a + 0.41) * a + 0.43) \
    * a + 0.47) * a + 0.53) * a + 0.59)) FROM x";

/// Coefficients for `c6`'s degree-12 Horner polynomial, highest degree first.
/// `p(a) = ((…((c0*a + c1)*a + c2)…)*a + c12)`.
const POLY12_COEFFS: [f64; 13] = [
    0.13, 0.11, 0.17, 0.19, 0.23, 0.29, 0.31, 0.37, 0.41, 0.43, 0.47, 0.53, 0.59,
];

const QUERIES: &[(&str, &str)] = &[
    ("c1_fma_sum", C1),
    ("c2_poly4_sum", C2),
    ("c3_sphere_filter_sum", C3),
    ("c4_weighted_sum", C4),
    ("c5_filtered_fma_sum", C5),
    ("c6_poly12_sum", C6),
];

// --- Deterministic data: 4 Float64 columns in [0, 1), byte-identical across
//     engines (no PRNG-seed plumbing across the C++/Rust boundary). --------

#[inline]
fn unit(i: usize, mult: u64) -> f64 {
    // Map a multiplicative hash into [0, 1) via the low 32 bits.
    let h = (i as u64).wrapping_mul(mult) as u32;
    (h as f64) / (u32::MAX as f64 + 1.0)
}
fn a(i: usize) -> f64 {
    unit(i, 0x9E37_79B1)
}
fn b(i: usize) -> f64 {
    unit(i, 0x0001_9E27)
}
fn c(i: usize) -> f64 {
    unit(i, 0x85EB_CA77)
}
fn d(i: usize) -> f64 {
    unit(i, 0xC2B2_AE3D)
}

fn arrow_batch(n: usize) -> RecordBatch {
    let s = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("a", ArrowDataType::Float64, false),
        ArrowField::new("b", ArrowDataType::Float64, false),
        ArrowField::new("c", ArrowDataType::Float64, false),
        ArrowField::new("d", ArrowDataType::Float64, false),
    ]));
    let ac: Float64Array = (0..n).map(a).collect();
    let bc: Float64Array = (0..n).map(b).collect();
    let cc: Float64Array = (0..n).map(c).collect();
    let dc: Float64Array = (0..n).map(d).collect();
    RecordBatch::try_new(
        s,
        vec![Arc::new(ac), Arc::new(bc), Arc::new(cc), Arc::new(dc)],
    )
    .unwrap()
}

fn polars_df(n: usize) -> polars::prelude::DataFrame {
    use polars::prelude::*;
    let ac: Vec<f64> = (0..n).map(a).collect();
    let bc: Vec<f64> = (0..n).map(b).collect();
    let cc: Vec<f64> = (0..n).map(c).collect();
    let dc: Vec<f64> = (0..n).map(d).collect();
    df!("a" => ac, "b" => bc, "c" => cc, "d" => dc).expect("polars df")
}

fn duckdb_conn(n: usize) -> duckdb::Connection {
    let conn = duckdb::Connection::open_in_memory().expect("duckdb open");
    conn.execute_batch(
        "CREATE TABLE x ( a DOUBLE NOT NULL, b DOUBLE NOT NULL, \
                          c DOUBLE NOT NULL, d DOUBLE NOT NULL );",
    )
    .expect("create");
    {
        let mut app = conn.appender("x").expect("appender");
        for i in 0..n {
            app.append_row(duckdb::params![a(i), b(i), c(i), d(i)])
                .expect("append");
        }
        app.flush().expect("flush");
    }
    conn
}

// --- Per-engine scalar executors ------------------------------------------

fn duckdb_scalar(conn: &duckdb::Connection, q: &str) -> f64 {
    let mut stmt = conn.prepare(q).expect("duckdb prep");
    let mut rows = stmt.query([]).expect("duckdb query");
    let row = rows.next().expect("duckdb row").expect("one row");
    row.get::<usize, f64>(0).expect("scalar f64")
}

fn polars_scalar(df: &polars::prelude::DataFrame, q: &str) -> f64 {
    use polars::prelude::*;
    let lf = df.clone().lazy();
    let out = match q {
        C1 => lf.select([(col("a") * col("b") + col("c") * col("d")).sum()]),
        C2 => lf.select([(col("a") * col("a") * col("a") * col("a")
            - col("a") * col("a") * col("a")
            + col("a") * col("a"))
        .sum()]),
        C3 => lf
            .filter((col("a") * col("a") + col("b") * col("b") + col("c") * col("c")).lt(lit(1.0)))
            .select([col("a").sum()]),
        C4 => lf.select([(col("a") * lit(0.4)
            + col("b") * lit(0.3)
            + col("c") * lit(0.2)
            + col("d") * lit(0.1))
        .sum()]),
        C5 => lf
            .filter((col("c") * col("c") + col("d") * col("d")).lt(lit(0.5)))
            .select([(col("a") * col("b")).sum()]),
        C6 => {
            // Horner: acc = c0; acc = acc*a + c_i. Same op order as the SQL
            // string and the engine, so all three evaluate the identical poly.
            let a = col("a");
            let mut acc = lit(POLY12_COEFFS[0]);
            for &k in &POLY12_COEFFS[1..] {
                acc = acc * a.clone() + lit(k);
            }
            lf.select([acc.sum()])
        }
        _ => panic!("unknown query"),
    }
    .collect()
    .expect("polars collect");
    out.column(out.get_column_names()[0])
        .unwrap()
        .f64()
        .expect("f64 scalar")
        .get(0)
        .expect("one value")
}

fn bolt_scalar(engine: &craton_bolt::Engine, q: &str) -> f64 {
    let h = engine.sql(q).expect("craton-bolt sql");
    let batch = h.record_batch();
    assert_eq!(batch.num_rows(), 1, "scalar aggregate must produce one row");
    let col = batch.column(0);
    if let Some(f) = col.as_any().downcast_ref::<Float64Array>() {
        f.value(0)
    } else if let Some(i) = col.as_any().downcast_ref::<Int64Array>() {
        i.value(0) as f64
    } else {
        panic!("unexpected scalar output dtype: {:?}", col.data_type());
    }
}

// --- Equivalence ----------------------------------------------------------

fn close_enough(x: f64, y: f64) -> bool {
    let diff = (x - y).abs();
    let mag = x.abs().max(y.abs()).max(1.0);
    diff / mag <= REL_TOL
}

fn verify_polars_vs_duckdb() {
    eprintln!("[compute-bench] verifying Polars ⇄ DuckDB on {VERIFY_ROWS}-row fixture…");
    let p = polars_df(VERIFY_ROWS);
    let dk = duckdb_conn(VERIFY_ROWS);
    for (name, sql) in QUERIES {
        let pv = polars_scalar(&p, sql);
        let dv = duckdb_scalar(&dk, sql);
        assert!(
            close_enough(pv, dv),
            "{name}: Polars {pv} vs DuckDB {dv} disagree"
        );
        eprintln!("[compute-bench]   {name} ✓ ({dv:.6})");
    }
}

// --- Criterion groups -----------------------------------------------------

fn bench_polars_group(c: &mut Criterion) {
    eprintln!("[compute-bench] loading {BENCH_ROWS} rows into Polars…");
    let df = polars_df(BENCH_ROWS);
    let mut g = c.benchmark_group("polars");
    g.throughput(Throughput::Elements(BENCH_ROWS as u64));
    g.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    for (name, sql) in QUERIES {
        g.bench_function(*name, |b| b.iter(|| black_box(polars_scalar(&df, sql))));
    }
    g.finish();
}

fn bench_duckdb_group(c: &mut Criterion) {
    eprintln!("[compute-bench] loading {BENCH_ROWS} rows into DuckDB…");
    let conn = duckdb_conn(BENCH_ROWS);
    let mut g = c.benchmark_group("duckdb");
    g.throughput(Throughput::Elements(BENCH_ROWS as u64));
    g.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    for (name, sql) in QUERIES {
        g.bench_function(*name, |b| b.iter(|| black_box(duckdb_scalar(&conn, sql))));
    }
    g.finish();
}

fn bench_bolt_group(c: &mut Criterion) {
    if std::env::var("BOLT_BENCH_GPU").ok().as_deref() != Some("1") {
        eprintln!("[compute-bench] skipping craton-bolt (set BOLT_BENCH_GPU=1)");
        return;
    }
    let mut engine = match craton_bolt::Engine::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("[compute-bench] skipping craton-bolt: CUDA init failed: {err}");
            return;
        }
    };
    // Equivalence vs DuckDB inside the long-lived bench engine (same pool/context).
    eprintln!("[compute-bench] craton-bolt equivalence vs DuckDB on {VERIFY_ROWS}-row fixture…");
    let verify_duck = duckdb_conn(VERIFY_ROWS);
    engine
        .register_table("x", arrow_batch(VERIFY_ROWS))
        .expect("register verify");
    for (name, sql) in QUERIES {
        let dv = duckdb_scalar(&verify_duck, sql);
        let bv = bolt_scalar(&engine, sql);
        assert!(
            close_enough(dv, bv),
            "{name}: DuckDB {dv} vs Craton Bolt {bv} disagree"
        );
        eprintln!("[compute-bench]   {name} ✓ ({bv:.6})");
    }
    drop(verify_duck);

    eprintln!("[compute-bench] loading {BENCH_ROWS} rows into Craton Bolt (one-time H2D)…");
    engine
        .replace_table("x", arrow_batch(BENCH_ROWS))
        .expect("replace bench");
    let mut g = c.benchmark_group("craton-bolt");
    g.throughput(Throughput::Elements(BENCH_ROWS as u64));
    g.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    for (name, sql) in QUERIES {
        g.bench_function(*name, |b| {
            b.iter_batched(
                || (),
                |_| black_box(bolt_scalar(&engine, sql)),
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

fn bench_entry(c: &mut Criterion) {
    use std::sync::Once;
    static VERIFY_ONCE: Once = Once::new();
    VERIFY_ONCE.call_once(verify_polars_vs_duckdb);
    bench_polars_group(c);
    bench_duckdb_group(c);
    bench_bolt_group(c);
}

criterion_group!(benches, bench_entry);
criterion_main!(benches);
