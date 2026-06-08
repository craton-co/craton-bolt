// SPDX-License-Identifier: Apache-2.0

//! **h2o.ai db-benchmark — groupby subset, three-engine comparison.**
//!
//! Bench protocol
//! --------------
//! This is a faithful port of the GROUP-BY portion of the h2o.ai
//! db-benchmark, the community-recognised standard that Polars, DuckDB,
//! Pandas, ClickHouse, and others use for embedded-OLAP comparisons. The
//! schema and query shapes match the original spec; the only deviation is
//! that grouping keys are `Int32` instead of categorical strings, so that
//! Craton Bolt's GPU GROUP-BY (which does not yet hash string keys) can run the
//! same SQL the CPU engines do.
//!
//! Engines
//! -------
//! - **Craton Bolt** — GPU SQL engine under test. Gated on `BOLT_BENCH_GPU=1`.
//! - **Polars 0.42** — Rust-native, Rayon-threaded.
//! - **DuckDB 1.2** — embedded, bundled, multi-threaded CPU.
//!
//! Discipline
//! ----------
//! 1. Every query is run through every engine on a small fixture *before*
//!    the timed runs, and results are compared with a strict floating-point
//!    tolerance. Cross-engine disagreement panics — we will not publish
//!    numbers from engines that disagree on the answer.
//! 2. The timed run uses h2o.ai's "small" scale (`N = 10_000_000`).
//! 3. Criterion `measurement_time = 20 s` per benchmark keeps the CI tight.
//!
//! Reference for the queries:
//! <https://h2oai.github.io/db-benchmark/> — original benchmark.
//! <https://duckdblabs.github.io/db-benchmark/> — DuckDB Labs fork (current
//! maintained version with up-to-date numbers).

use std::sync::Arc;
use std::time::Duration;

use arrow_array::{Float64Array, Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion, Throughput};

// Shared cross-suite relative-tolerance constant. See the doc comment on
// `craton_bolt::REL_TOL_TEST` in `src/lib.rs`; the same constant lives in
// `tests/common::REL_TOL` for integration tests (which can't import bench
// crates and vice-versa).
use craton_bolt::REL_TOL_TEST as REL_TOL;

// --- h2o.ai-style data spec ------------------------------------------------
//
// The original h2o.ai benchmark generates N rows with:
//   id1, id2, id3 — high-/medium-/low-cardinality grouping columns
//   v1, v2, v3    — numeric value columns to aggregate
// Cardinalities scale with N; below we pin to h2o.ai's "small" scale (N=1e7).

/// Row count for the timed runs. h2o.ai's "small" scale.
const BENCH_ROWS: usize = 10_000_000;

/// Row count for the result-equivalence check at startup.
const VERIFY_ROWS: usize = 100_000;

/// Cardinalities (distinct values) for the grouping columns at the
/// `BENCH_ROWS` scale. Matches the h2o.ai shape: low / medium / high.
const ID1_CARD: i32 = 100;
const ID2_CARD: i32 = 10_000;
const ID3_CARD: i32 = 1_000_000;

/// Criterion measurement window per query.
const MEASUREMENT_SECS: u64 = 20;

// --- The five queries we benchmark ----------------------------------------
//
// Names match the h2o.ai numbering (q1..q5 of the groupby section). We
// pick a representative subset that exercises both low- and high-cardinality
// group-by paths.

/// h2o.ai q1: low-cardinality SUM. The simplest groupby.
const Q1: &str = "SELECT id1, SUM(v1) FROM x GROUP BY id1";

/// h2o.ai q2: medium-cardinality multi-aggregate. Two SUMs per group.
const Q2: &str = "SELECT id2, SUM(v1), SUM(v2) FROM x GROUP BY id2";

/// h2o.ai q3 (adapted): two grouping keys, medium overall cardinality.
const Q3: &str = "SELECT id1, id2, SUM(v1) FROM x GROUP BY id1, id2";

/// h2o.ai q4: low-cardinality multi-aggregate AVG. Tests AVG codegen.
/// Re-enabled after the `atom.add.s64` → `atom.add.u64` codegen fix.
const Q4: &str = "SELECT id1, AVG(v1), AVG(v2), AVG(v3) FROM x GROUP BY id1";

/// h2o.ai q5: very-high-cardinality SUM. Stresses the hash-table size.
const Q5: &str = "SELECT id3, SUM(v1) FROM x GROUP BY id3";

const QUERIES: &[(&str, &str)] = &[
    ("q1_low_card_sum", Q1),
    ("q2_med_card_2sum", Q2),
    ("q3_two_key_sum", Q3),
    ("q4_low_card_3avg", Q4),
    ("q5_high_card_sum", Q5),
];

// --- Deterministic synthetic data generator --------------------------------
//
// h2o.ai's generator is random; we use a deterministic hash so every engine
// sees byte-identical input (no PRNG seed plumbing needed across language
// boundaries — DuckDB is C++, Polars/Craton Bolt are Rust).

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
fn v3(i: usize) -> f64 {
    ((i.wrapping_mul(17) as i32).rem_euclid(10_000)) as f64 / 100.0
}

// --- Per-engine fixture builders ------------------------------------------

fn arrow_batch(n: usize) -> RecordBatch {
    let s = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id1", ArrowDataType::Int32, false),
        ArrowField::new("id2", ArrowDataType::Int32, false),
        ArrowField::new("id3", ArrowDataType::Int32, false),
        ArrowField::new("v1", ArrowDataType::Float64, false),
        ArrowField::new("v2", ArrowDataType::Float64, false),
        ArrowField::new("v3", ArrowDataType::Float64, false),
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
            Arc::new(id1c),
            Arc::new(id2c),
            Arc::new(id3c),
            Arc::new(v1c),
            Arc::new(v2c),
            Arc::new(v3c),
        ],
    )
    .unwrap()
}

fn polars_df(n: usize) -> polars::prelude::DataFrame {
    use polars::prelude::*;
    let id1c: Vec<i32> = (0..n).map(id1).collect();
    let id2c: Vec<i32> = (0..n).map(id2).collect();
    let id3c: Vec<i32> = (0..n).map(id3).collect();
    let v1c: Vec<f64> = (0..n).map(v1).collect();
    let v2c: Vec<f64> = (0..n).map(v2).collect();
    let v3c: Vec<f64> = (0..n).map(v3).collect();
    df!(
        "id1" => id1c,
        "id2" => id2c,
        "id3" => id3c,
        "v1" => v1c,
        "v2" => v2c,
        "v3" => v3c,
    )
    .expect("polars df")
}

fn duckdb_conn(n: usize) -> duckdb::Connection {
    let conn = duckdb::Connection::open_in_memory().expect("duckdb open");
    conn.execute_batch(
        "CREATE TABLE x ( \
            id1 INTEGER NOT NULL, \
            id2 INTEGER NOT NULL, \
            id3 INTEGER NOT NULL, \
            v1  DOUBLE  NOT NULL, \
            v2  DOUBLE  NOT NULL, \
            v3  DOUBLE  NOT NULL \
        );",
    )
    .expect("create");
    {
        let mut app = conn.appender("x").expect("appender");
        for i in 0..n {
            app.append_row(duckdb::params![id1(i), id2(i), id3(i), v1(i), v2(i), v3(i)])
                .expect("append");
        }
        app.flush().expect("flush");
    }
    conn
}

// --- Per-engine, per-query executors --------------------------------------
//
// Every executor returns a normalised `QueryResult` so the equivalence check
// can compare them with a single helper.

#[derive(Debug, Clone)]
struct QueryResult {
    /// Group keys, one per group; for two-key queries we pack `(id1<<32) | id2`
    /// into an i64 to keep the comparison logic flat.
    keys: Vec<i64>,
    /// Per-group aggregate output; one Vec per output aggregate column.
    aggs: Vec<Vec<f64>>,
}

impl QueryResult {
    /// Sort groups by key so order-independent engines compare equal.
    fn canonicalise(mut self) -> Self {
        let mut idx: Vec<usize> = (0..self.keys.len()).collect();
        idx.sort_by_key(|&i| self.keys[i]);
        let keys = idx.iter().map(|&i| self.keys[i]).collect();
        let aggs = self
            .aggs
            .into_iter()
            .map(|col| idx.iter().map(|&i| col[i]).collect())
            .collect();
        self.keys = keys;
        self.aggs = aggs;
        self
    }
}

fn pack2(a: i32, b: i32) -> i64 {
    ((a as i64) << 32) | (b as i64 & 0xFFFF_FFFF)
}

// --- Polars

fn polars_q(df: &polars::prelude::DataFrame, q: &str) -> QueryResult {
    use polars::prelude::*;
    let lf = df.clone().lazy();
    let r = match q {
        Q1 => lf.group_by([col("id1")]).agg([col("v1").sum().alias("s1")]),
        Q2 => lf
            .group_by([col("id2")])
            .agg([col("v1").sum().alias("s1"), col("v2").sum().alias("s2")]),
        Q3 => lf
            .group_by([col("id1"), col("id2")])
            .agg([col("v1").sum().alias("s1")]),
        Q4 => lf.group_by([col("id1")]).agg([
            col("v1").mean().alias("a1"),
            col("v2").mean().alias("a2"),
            col("v3").mean().alias("a3"),
        ]),
        Q5 => lf.group_by([col("id3")]).agg([col("v1").sum().alias("s1")]),
        _ => panic!("unknown query"),
    };
    let out = r.collect().expect("polars groupby");
    polars_to_result(&out, q)
}

fn polars_to_result(out: &polars::prelude::DataFrame, q: &str) -> QueryResult {
    let h = out.height();
    let (keys, aggs) = match q {
        Q1 | Q5 => {
            let kc = if q == Q1 { "id1" } else { "id3" };
            let k = out.column(kc).unwrap().i32().unwrap();
            let s = out.column("s1").unwrap().f64().unwrap();
            let mut keys = Vec::with_capacity(h);
            let mut a = Vec::with_capacity(h);
            for i in 0..h {
                keys.push(k.get(i).unwrap() as i64);
                a.push(s.get(i).unwrap());
            }
            (keys, vec![a])
        }
        Q2 => {
            let k = out.column("id2").unwrap().i32().unwrap();
            let s1 = out.column("s1").unwrap().f64().unwrap();
            let s2 = out.column("s2").unwrap().f64().unwrap();
            let mut keys = Vec::with_capacity(h);
            let mut a1 = Vec::with_capacity(h);
            let mut a2 = Vec::with_capacity(h);
            for i in 0..h {
                keys.push(k.get(i).unwrap() as i64);
                a1.push(s1.get(i).unwrap());
                a2.push(s2.get(i).unwrap());
            }
            (keys, vec![a1, a2])
        }
        Q3 => {
            let k1 = out.column("id1").unwrap().i32().unwrap();
            let k2 = out.column("id2").unwrap().i32().unwrap();
            let s = out.column("s1").unwrap().f64().unwrap();
            let mut keys = Vec::with_capacity(h);
            let mut a = Vec::with_capacity(h);
            for i in 0..h {
                keys.push(pack2(k1.get(i).unwrap(), k2.get(i).unwrap()));
                a.push(s.get(i).unwrap());
            }
            (keys, vec![a])
        }
        Q4 => {
            let k = out.column("id1").unwrap().i32().unwrap();
            let a1 = out.column("a1").unwrap().f64().unwrap();
            let a2 = out.column("a2").unwrap().f64().unwrap();
            let a3 = out.column("a3").unwrap().f64().unwrap();
            let mut keys = Vec::with_capacity(h);
            let mut c1 = Vec::with_capacity(h);
            let mut c2 = Vec::with_capacity(h);
            let mut c3 = Vec::with_capacity(h);
            for i in 0..h {
                keys.push(k.get(i).unwrap() as i64);
                c1.push(a1.get(i).unwrap());
                c2.push(a2.get(i).unwrap());
                c3.push(a3.get(i).unwrap());
            }
            (keys, vec![c1, c2, c3])
        }
        _ => panic!(),
    };
    QueryResult { keys, aggs }.canonicalise()
}

// --- DuckDB

fn duckdb_q(conn: &duckdb::Connection, q: &str) -> QueryResult {
    let sql = format!("{} ORDER BY 1", q);
    let mut stmt = conn.prepare(&sql).expect("duckdb prep");
    let mut rows = stmt.query([]).expect("duckdb query");

    let (mut keys, mut aggs): (Vec<i64>, Vec<Vec<f64>>) = (Vec::new(), Vec::new());
    let n_aggs = match q {
        Q1 | Q3 | Q5 => 1,
        Q2 => 2,
        Q4 => 3,
        _ => panic!(),
    };
    for _ in 0..n_aggs {
        aggs.push(Vec::new());
    }
    while let Some(row) = rows.next().expect("duckdb row") {
        let k = if q == Q3 {
            let k1: i32 = row.get(0).expect("Q3 k1");
            let k2: i32 = row.get(1).expect("Q3 k2");
            pack2(k1, k2)
        } else {
            let k: i32 = row.get(0).expect("k");
            k as i64
        };
        keys.push(k);
        let agg_start = if q == Q3 { 2 } else { 1 };
        for ai in 0..n_aggs {
            let v: f64 = row.get(agg_start + ai).expect("agg");
            aggs[ai].push(v);
        }
    }
    QueryResult { keys, aggs }.canonicalise()
}

// --- Craton Bolt (GPU)

fn bolt_q(engine: &craton_bolt::Engine, q: &str) -> QueryResult {
    let h = engine.sql(q).expect("craton-bolt sql");
    bolt_decode(&h, q)
}

/// Decode a Craton Bolt `QueryHandle` into the normalised `QueryResult` shape.
/// Takes the original query *constant* `q` (one of `Q1..Q5`) to determine the
/// expected number of key / agg columns — the actual SQL that produced `h`
/// may target a verification table name (e.g. `x_verify`) rather than `x`.
fn bolt_decode(h: &craton_bolt::exec::QueryHandle, q: &str) -> QueryResult {
    let batch = h.record_batch();
    let n = batch.num_rows();
    let n_aggs = match q {
        Q1 | Q3 | Q5 => 1,
        Q2 => 2,
        Q4 => 3,
        _ => panic!(),
    };
    let key_cols = if q == Q3 { 2 } else { 1 };
    let mut keys = Vec::with_capacity(n);
    if key_cols == 1 {
        let k = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("group key Int32");
        for i in 0..n {
            keys.push(k.value(i) as i64);
        }
    } else {
        let k1 = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Q3 k1 Int32");
        let k2 = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Q3 k2 Int32");
        for i in 0..n {
            keys.push(pack2(k1.value(i), k2.value(i)));
        }
    }
    let mut aggs: Vec<Vec<f64>> = (0..n_aggs).map(|_| Vec::with_capacity(n)).collect();
    for ai in 0..n_aggs {
        // SUM(Int)→Int64 widening: handle both Int64 and Float64 result columns.
        let col = batch.column(key_cols + ai);
        if let Some(f) = col.as_any().downcast_ref::<Float64Array>() {
            for i in 0..n {
                aggs[ai].push(f.value(i));
            }
        } else if let Some(i64a) = col.as_any().downcast_ref::<arrow_array::Int64Array>() {
            for i in 0..n {
                aggs[ai].push(i64a.value(i) as f64);
            }
        } else {
            panic!(
                "craton-bolt agg column {} has unexpected type {:?}",
                ai,
                col.data_type()
            );
        }
    }
    QueryResult { keys, aggs }.canonicalise()
}

// --- Equivalence check ----------------------------------------------------

// Relative tolerance: SUMs over 10 M Float64 values accumulate up to ~10
// ULPs of rounding when summed in different orders — 1e-9 relative covers
// it with margin to spare. The actual constant `REL_TOL` is imported at
// the top of the file from `craton_bolt::REL_TOL_TEST` (the doc-hidden
// re-export of `tests/common::REL_TOL`); the bench crate can't reach
// into the integration-test binary, so the shared constant lives on the
// library side.

fn close_enough(a: f64, b: f64) -> bool {
    let diff = (a - b).abs();
    let mag = a.abs().max(b.abs()).max(1.0);
    diff / mag <= REL_TOL
}

fn assert_results_match(label: &str, expected: &QueryResult, actual: &QueryResult) {
    assert_eq!(
        expected.keys.len(),
        actual.keys.len(),
        "{label}: group-count mismatch (expected {}, got {})",
        expected.keys.len(),
        actual.keys.len()
    );
    for (i, (e, a)) in expected.keys.iter().zip(actual.keys.iter()).enumerate() {
        assert_eq!(e, a, "{label}: group key #{i} differs ({e} vs {a})");
    }
    assert_eq!(
        expected.aggs.len(),
        actual.aggs.len(),
        "{label}: agg-column-count mismatch"
    );
    for (ci, (ec, ac)) in expected.aggs.iter().zip(actual.aggs.iter()).enumerate() {
        for (i, (e, a)) in ec.iter().zip(ac.iter()).enumerate() {
            assert!(
                close_enough(*e, *a),
                "{label}: agg col {ci}, row {i}, key {}: expected {e}, got {a} (rel diff {})",
                expected.keys[i],
                (e - a).abs() / e.abs().max(a.abs()).max(1.0),
            );
        }
    }
}

/// Cross-engine equivalence check that does NOT touch Craton Bolt. The process-
/// wide device memory pool stores its entries by raw `CUdeviceptr` against the
/// context they were minted in; if we created a verification engine here and
/// dropped it before the bench, the pool would later hand the 10 M-row engine
/// dangling pointers from the destroyed context. So we keep this check
/// CPU-only (Polars vs DuckDB) — the Craton Bolt cross-check happens later inside
/// `bench_bolt_group`, where it shares a single long-lived engine with the
/// timed runs.
fn verify_polars_vs_duckdb() {
    eprintln!("[h2o-bench] verifying Polars ⇄ DuckDB equivalence on {VERIFY_ROWS}-row fixture…");
    let polars = polars_df(VERIFY_ROWS);
    let duck = duckdb_conn(VERIFY_ROWS);
    for (name, sql) in QUERIES {
        let p = polars_q(&polars, sql);
        let d = duckdb_q(&duck, sql);
        assert_results_match(&format!("{name}: Polars vs DuckDB"), &p, &d);
        eprintln!("[h2o-bench]   {name} ✓");
    }
    eprintln!("[h2o-bench] Polars and DuckDB agree on every query");
}

// --- Criterion groups -----------------------------------------------------

fn bench_polars_group(c: &mut Criterion) {
    eprintln!("[h2o-bench] loading {BENCH_ROWS} rows into Polars…");
    let df = polars_df(BENCH_ROWS);
    let mut g = c.benchmark_group("polars");
    g.throughput(Throughput::Elements(BENCH_ROWS as u64));
    g.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    for (name, sql) in QUERIES {
        g.bench_function(*name, |b| b.iter(|| black_box(polars_q(&df, sql))));
    }
    g.finish();
}

fn bench_duckdb_group(c: &mut Criterion) {
    eprintln!("[h2o-bench] loading {BENCH_ROWS} rows into DuckDB (slow first-time)…");
    let conn = duckdb_conn(BENCH_ROWS);
    eprintln!("[h2o-bench] DuckDB load complete");
    let mut g = c.benchmark_group("duckdb");
    g.throughput(Throughput::Elements(BENCH_ROWS as u64));
    g.measurement_time(Duration::from_secs(MEASUREMENT_SECS));
    for (name, sql) in QUERIES {
        g.bench_function(*name, |b| b.iter(|| black_box(duckdb_q(&conn, sql))));
    }
    g.finish();
}

fn bench_bolt_group(c: &mut Criterion) {
    if std::env::var("BOLT_BENCH_GPU").ok().as_deref() != Some("1") {
        eprintln!("[h2o-bench] skipping craton-bolt (set BOLT_BENCH_GPU=1 to enable)");
        return;
    }
    let mut engine = match craton_bolt::Engine::new() {
        Ok(e) => e,
        Err(err) => {
            eprintln!("[h2o-bench] skipping craton-bolt: CUDA init failed: {err}");
            return;
        }
    };

    // Step 1 — equivalence check against DuckDB at the small fixture size,
    // inside the same long-lived engine that will run the timed bench.
    // Keeping it in one engine matters because the device-memory pool's
    // entries are tied to the CUDA context that allocated them; dropping a
    // verification engine and creating a fresh one for the bench would leave
    // the pool's free-list dangling (now fixed at `CudaContext::drop` time,
    // but using one engine is still the cleaner pattern).
    eprintln!(
        "[h2o-bench] craton-bolt equivalence check on {VERIFY_ROWS}-row fixture (vs DuckDB)…"
    );
    let verify_duck = duckdb_conn(VERIFY_ROWS);
    engine
        .register_table("x", arrow_batch(VERIFY_ROWS))
        .expect("register verify");
    for (name, sql) in QUERIES {
        let d = duckdb_q(&verify_duck, sql);
        let j = bolt_q(&engine, sql);
        assert_results_match(&format!("{name}: DuckDB vs Craton Bolt"), &d, &j);
        eprintln!("[h2o-bench]   {name} ✓");
    }
    drop(verify_duck);

    // Step 2 — swap in the BENCH_ROWS dataset via the new `replace_table`
    // entry point. This drops the old GpuTable's allocations back into the
    // memory pool (where the new upload can recycle them) and rebuilds
    // the dictionary registry / provider schema atomically.
    eprintln!("[h2o-bench] loading {BENCH_ROWS} rows into Craton Bolt (one-time upload to GPU)…");
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
                |_| black_box(bolt_q(&engine, sql)),
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
