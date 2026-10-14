// SPDX-License-Identifier: Apache-2.0

//! Out-of-core execution: partitioned GROUP BY / DISTINCT, grace hash join,
//! external sort, host-budget scans, and device-residency eviction. Results
//! are checked against values computed directly in Rust.
//!
//! `cargo test --test out_of_core_test -- --ignored --test-threads=1`

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::Engine;

const N: i64 = 200_000;
const BUDGET: usize = 1 << 20;

/// `(k, g = k % 97, v = k % 1000, s = "s{k % 13}")`.
fn fact(n: i64) -> RecordBatch {
    let k: Vec<i64> = (0..n).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int64, false),
        ArrowField::new("g", ArrowDataType::Int64, false),
        ArrowField::new("v", ArrowDataType::Int64, false),
        ArrowField::new("s", ArrowDataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(k.clone())),
            Arc::new(Int64Array::from(
                k.iter().map(|x| x % 97).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                k.iter().map(|x| x % 1000).collect::<Vec<_>>(),
            )),
            Arc::new(StringArray::from(
                k.iter().map(|x| format!("s{}", x % 13)).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// `(k2 = 2k, tag = k % 5)` for k in 0..n.
fn fact2(n: i64) -> RecordBatch {
    let k: Vec<i64> = (0..n).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k2", ArrowDataType::Int64, false),
        ArrowField::new("tag", ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(
                k.iter().map(|x| x * 2).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                k.iter().map(|x| x % 5).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// `(g 0..97, w = 3g)`.
fn dim() -> RecordBatch {
    let g: Vec<i64> = (0..97).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("g", ArrowDataType::Int64, false),
        ArrowField::new("w", ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(g.clone())),
            Arc::new(Int64Array::from(
                g.iter().map(|x| x * 3).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn i64s(b: &RecordBatch, c: usize) -> Vec<i64> {
    b.column(c).as_primitive::<Int64Type>().values().to_vec()
}

fn budget_engine() -> Engine {
    Engine::builder()
        .memory_budget(BUDGET)
        .build()
        .expect("ctx")
}

#[test]
#[ignore = "gpu:e2e — partitioned GROUP BY over an in-memory table and through a join"]
fn partitioned_group_by_is_exact() {
    let mut e = budget_engine();
    e.register_table("fact", fact(N)).unwrap();
    e.register_table("dim", dim()).unwrap();
    let before = e.memory_stats();
    let plan = e
        .explain_sql("SELECT g, SUM(v) AS sv, COUNT(*) AS n FROM fact GROUP BY g ORDER BY g")
        .unwrap();
    assert!(
        plan.contains("Execution: out of core — partitioned GROUP BY on 'fact' (g)"),
        "{plan}"
    );
    assert_eq!(
        e.memory_stats().out_of_core_statements,
        before.out_of_core_statements
    );

    let out = e
        .sql("SELECT g, SUM(v) AS sv, COUNT(*) AS n FROM fact GROUP BY g ORDER BY g")
        .unwrap()
        .into_record_batch();
    let mut sv = BTreeMap::new();
    for k in 0..N {
        *sv.entry(k % 97).or_insert(0i64) += k % 1000;
    }
    assert_eq!(i64s(&out, 0), sv.keys().copied().collect::<Vec<_>>());
    assert_eq!(i64s(&out, 1), sv.values().copied().collect::<Vec<_>>());
    assert_eq!(i64s(&out, 2).iter().sum::<i64>(), N);
    let after = e.memory_stats();
    assert!(after.out_of_core_statements > before.out_of_core_statements);
    assert!(after.spill_files > before.spill_files);
    assert!(after.spilled_bytes > 0);

    // Grouped on the large side of a join: each partition joins the small
    // side whole.
    let ooc = e.memory_stats().out_of_core_statements;
    let j = e
        .sql(
            "SELECT fact.g AS fg, SUM(dim.w) AS sw FROM fact JOIN dim ON fact.g = dim.g \
             WHERE fact.v < 500 GROUP BY fact.g ORDER BY fg",
        )
        .unwrap()
        .into_record_batch();
    let mut sw = BTreeMap::new();
    for k in (0..N).filter(|k| k % 1000 < 500) {
        *sw.entry(k % 97).or_insert(0i64) += (k % 97) * 3;
    }
    assert_eq!(i64s(&j, 0), sw.keys().copied().collect::<Vec<_>>());
    assert_eq!(i64s(&j, 1), sw.values().copied().collect::<Vec<_>>());
    assert_eq!(e.memory_stats().out_of_core_statements, ooc + 1);
}

#[test]
#[ignore = "gpu:e2e — grace hash join (inner and outer)"]
fn grace_join_matches_in_memory_semantics() {
    let mut e = budget_engine();
    e.register_table("a", fact(N)).unwrap();
    e.register_table("b", fact2(N)).unwrap();
    let ooc = e.memory_stats().out_of_core_statements;
    let plan = e
        .explain_sql("SELECT COUNT(*) FROM a JOIN b ON a.k = b.k2")
        .unwrap();
    assert!(
        plan.contains("grace hash join of 'a' (k) with 'b' (k2)"),
        "{plan}"
    );
    let inner = e
        .sql("SELECT COUNT(*) AS n, SUM(b.tag) AS t FROM a JOIN b ON a.k = b.k2")
        .unwrap()
        .into_record_batch();
    // a.k = 2j for j < N/2.
    let expect_t: i64 = (0..N / 2).map(|j| j % 5).sum();
    assert_eq!(i64s(&inner, 0), vec![N / 2]);
    assert_eq!(i64s(&inner, 1), vec![expect_t]);
    let left = e
        .sql("SELECT COUNT(*) FROM a LEFT JOIN b ON a.k = b.k2")
        .unwrap()
        .into_record_batch();
    assert_eq!(i64s(&left, 0), vec![N]);
    let right = e
        .sql("SELECT COUNT(*) FROM a RIGHT JOIN b ON a.k = b.k2")
        .unwrap()
        .into_record_batch();
    assert_eq!(i64s(&right, 0), vec![N]);
    let full = e
        .sql("SELECT COUNT(*) FROM a FULL JOIN b ON a.k = b.k2")
        .unwrap()
        .into_record_batch();
    // N left rows; right rows k2 >= N (j >= N/2) are unmatched.
    assert_eq!(i64s(&full, 0), vec![N + N / 2]);
    assert_eq!(e.memory_stats().out_of_core_statements, ooc + 4);
}

#[test]
#[ignore = "gpu:e2e — partitioned DISTINCT and external sort"]
fn distinct_and_external_sort() {
    let mut e = budget_engine();
    e.register_table("fact", fact(N)).unwrap();
    let ooc = e.memory_stats().out_of_core_statements;
    let d = e
        .sql("SELECT DISTINCT g, s FROM fact")
        .unwrap()
        .into_record_batch();
    let expect: BTreeSet<(i64, i64)> = (0..N).map(|k| (k % 97, k % 13)).collect();
    assert_eq!(d.num_rows(), expect.len());

    let plan = e
        .explain_sql("SELECT k, v FROM fact WHERE v < 10 ORDER BY v DESC, k LIMIT 5")
        .unwrap();
    assert!(
        plan.contains("external sort of 'fact'") && plan.contains("(top-k runs)"),
        "{plan}"
    );
    let top = e
        .sql("SELECT k, v FROM fact WHERE v < 10 ORDER BY v DESC, k LIMIT 5")
        .unwrap()
        .into_record_batch();
    let mut rows: Vec<(i64, i64)> = (0..N)
        .filter(|k| k % 1000 < 10)
        .map(|k| (k, k % 1000))
        .collect();
    rows.sort_by(|x, y| y.1.cmp(&x.1).then(x.0.cmp(&y.0)));
    let want: Vec<(i64, i64)> = rows.into_iter().take(5).collect();
    let got: Vec<(i64, i64)> = i64s(&top, 0).into_iter().zip(i64s(&top, 1)).collect();
    assert_eq!(got, want);

    let all = e
        .sql("SELECT k FROM fact WHERE g = 3 ORDER BY k DESC")
        .unwrap()
        .into_record_batch();
    let want: Vec<i64> = (0..N).rev().filter(|k| k % 97 == 3).collect();
    assert_eq!(i64s(&all, 0), want);
    assert_eq!(e.memory_stats().out_of_core_statements, ooc + 3);
}

#[cfg(feature = "storage")]
#[test]
#[ignore = "gpu:e2e — Parquet aggregate under a host budget, scan eviction"]
fn host_budget_bounds_storage_scans() {
    let dir = std::env::temp_dir().join(format!("bolt-ooc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let opts = craton_bolt::storage::ParquetWriteOptions {
        row_group_rows: 20_000,
        ..Default::default()
    };
    craton_bolt::storage::write_parquet(&dir.join("f.parquet"), &[fact(N)], &opts).unwrap();
    craton_bolt::storage::write_parquet(&dir.join("small.parquet"), &[dim()], &opts).unwrap();
    let mut e = Engine::builder()
        .host_memory_budget(BUDGET)
        .spill_dir(dir.join("spill"))
        .build()
        .expect("ctx");
    e.register_parquet("pq", dir.join("f.parquet")).unwrap();
    e.register_parquet("d1", dir.join("small.parquet")).unwrap();
    let out = e
        .sql("SELECT g, COUNT(*) AS n FROM pq GROUP BY g ORDER BY g")
        .unwrap()
        .into_record_batch();
    assert_eq!(out.num_rows(), 97);
    assert_eq!(i64s(&out, 1).iter().sum::<i64>(), N);
    assert!(e.memory_stats().out_of_core_statements >= 1);
    // The spill directory is cleaned as partitions complete.
    let leftover = std::fs::read_dir(dir.join("spill"))
        .map(|d| d.count())
        .unwrap_or(0);
    assert_eq!(leftover, 0);
    // Small scans stay under the budget and are reused.
    let s = e.sql("SELECT SUM(w) FROM d1").unwrap().into_record_batch();
    assert_eq!(i64s(&s, 0), vec![(0..97).map(|g| g * 3).sum::<i64>()]);
    assert!(e.memory_stats().host_reserved <= BUDGET);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "gpu:e2e — device budget evicts idle resident tables between statements"]
fn device_budget_evicts_idle_tables() {
    let mut e = Engine::builder()
        .memory_budget(64 * 1024)
        .build()
        .expect("ctx");
    e.register_table("t1", fact(20_000)).unwrap();
    e.register_table("t2", fact(20_000)).unwrap();
    let ev0 = e.memory_stats().evictions;
    let a = e.sql("SELECT SUM(v) FROM t1").unwrap().into_record_batch();
    let b = e.sql("SELECT SUM(v) FROM t2").unwrap().into_record_batch();
    assert_eq!(i64s(&a, 0), i64s(&b, 0));
    assert!(
        e.memory_stats().evictions > ev0,
        "t1's device copy is evicted for t2"
    );
    // Results stay correct after re-upload.
    let a2 = e.sql("SELECT SUM(v) FROM t1").unwrap().into_record_batch();
    assert_eq!(i64s(&a2, 0), i64s(&a, 0));
}

/// `(h, v)`: 70% of the rows share `h = 0`, the rest are distinct.
fn skewed(n: i64) -> RecordBatch {
    let h: Vec<i64> = (0..n).map(|i| if i % 10 < 7 { 0 } else { i }).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("h", ArrowDataType::Int64, false),
        ArrowField::new("v", ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(h)),
            Arc::new(Int64Array::from((0..n).collect::<Vec<_>>())),
        ],
    )
    .unwrap()
}

#[test]
#[ignore = "gpu:e2e — over-budget partitions are split again (key skew)"]
fn skewed_partitions_are_split_again() {
    let mut e = budget_engine();
    e.register_table("s", skewed(N)).unwrap();
    e.register_table("b", fact2(N)).unwrap();
    let before = e.memory_stats();
    let out = e
        .sql("SELECT h, COUNT(*) AS n, SUM(v) AS sv FROM s GROUP BY h ORDER BY h")
        .unwrap()
        .into_record_batch();
    let mut expect: BTreeMap<i64, (i64, i64)> = BTreeMap::new();
    for i in 0..N {
        let h = if i % 10 < 7 { 0 } else { i };
        let x = expect.entry(h).or_insert((0, 0));
        x.0 += 1;
        x.1 += i;
    }
    assert_eq!(i64s(&out, 0), expect.keys().copied().collect::<Vec<_>>());
    assert_eq!(
        i64s(&out, 1),
        expect.values().map(|x| x.0).collect::<Vec<_>>()
    );
    assert_eq!(
        i64s(&out, 2),
        expect.values().map(|x| x.1).collect::<Vec<_>>()
    );
    let mid = e.memory_stats();
    assert!(mid.out_of_core_statements > before.out_of_core_statements);
    assert!(mid.repartitions > before.repartitions, "{mid:?}");

    // Grace join with the skewed side: b.k2 = 0 matches every hot row.
    let j = e
        .sql("SELECT COUNT(*) AS n, SUM(b.tag) AS t FROM s JOIN b ON s.h = b.k2")
        .unwrap()
        .into_record_batch();
    let mut n = 0i64;
    let mut t = 0i64;
    for i in 0..N {
        let h = if i % 10 < 7 { 0 } else { i };
        // b.k2 = 2k (k < N), tag = k % 5.
        if h % 2 == 0 && h / 2 < N {
            n += 1;
            t += (h / 2) % 5;
        }
    }
    assert_eq!(i64s(&j, 0), vec![n]);
    assert_eq!(i64s(&j, 1), vec![t]);
    assert!(e.memory_stats().repartitions > mid.repartitions);
}

#[test]
#[ignore = "gpu:e2e — row-wise scans over the host budget, and sql_stream"]
fn row_wise_scans_run_in_chunks_and_stream() {
    let dir = std::env::temp_dir().join(format!("bolt-ooc-rw-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let opts = craton_bolt::storage::ParquetWriteOptions {
        row_group_rows: 10_000,
        ..Default::default()
    };
    craton_bolt::storage::write_parquet(&dir.join("f.parquet"), &[fact(N)], &opts).unwrap();
    let mut e = Engine::builder()
        .host_memory_budget(BUDGET)
        .spill_dir(dir.join("spill"))
        .build()
        .expect("ctx");
    e.register_parquet("pq", dir.join("f.parquet")).unwrap();
    let plan = e
        .explain_sql("SELECT k FROM pq WHERE v >= 0 LIMIT 9")
        .unwrap();
    assert!(plan.contains("chunked row-wise scan of 'pq'"), "{plan}");
    assert!(plan.contains("stops at the LIMIT"), "{plan}");
    let plan = e.explain_sql("SELECT SUM(v), AVG(v) FROM pq").unwrap();
    assert!(plan.contains("partial aggregates of 'pq'"), "{plan}");
    // Every row passes: far more than the budget at once.
    let all = e
        .sql("SELECT k, v + 1 AS w FROM pq WHERE v >= 0")
        .unwrap()
        .into_record_batch();
    assert_eq!(all.num_rows() as i64, N);
    assert_eq!(
        i64s(&all, 1).iter().sum::<i64>(),
        (0..N).map(|k| k % 1000 + 1).sum::<i64>()
    );
    assert!(e.memory_stats().host_reserved <= BUDGET);
    let lim = e
        .sql("SELECT k FROM pq WHERE g = 3 LIMIT 5 OFFSET 2")
        .unwrap()
        .into_record_batch();
    assert_eq!(i64s(&lim, 0), vec![197, 294, 391, 488, 585]);

    // The same rows as a stream of bounded batches.
    let stream = e.sql_stream("SELECT k, s FROM pq WHERE g < 10").unwrap();
    assert!(stream.is_incremental());
    let mut batches = 0;
    let mut rows = 0i64;
    let mut sum = 0i64;
    for b in stream {
        let b = b.unwrap();
        batches += 1;
        rows += b.num_rows() as i64;
        sum += i64s(&b, 0).iter().sum::<i64>();
    }
    let expect: Vec<i64> = (0..N).filter(|k| k % 97 < 10).collect();
    assert!(batches > 1, "{batches}");
    assert_eq!(rows, expect.len() as i64);
    assert_eq!(sum, expect.iter().sum::<i64>());
    // LIMIT stops early; non-row-wise queries arrive as one batch.
    let first: Vec<RecordBatch> = e
        .sql_stream("SELECT k FROM pq LIMIT 3")
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(first.iter().map(|b| b.num_rows()).sum::<usize>(), 3);
    let agg = e.sql_stream("SELECT COUNT(*) FROM pq").unwrap();
    assert!(!agg.is_incremental());
    let agg: Vec<RecordBatch> = agg.collect::<Result<_, _>>().unwrap();
    assert_eq!(i64s(&agg[0], 0), vec![N]);
    // Scalar aggregates over the over-budget scan merge per-chunk partials.
    let s = e
        .sql("SELECT COUNT(*) AS n, SUM(v) AS sv, MIN(k) AS lo, MAX(k) AS hi, AVG(v) AS av               FROM pq WHERE g < 50")
        .unwrap()
        .into_record_batch();
    let rows: Vec<i64> = (0..N).filter(|k| k % 97 < 50).collect();
    let sv: i64 = rows.iter().map(|k| k % 1000).sum();
    assert_eq!(i64s(&s, 0), vec![rows.len() as i64]);
    assert_eq!(i64s(&s, 1), vec![sv]);
    assert_eq!(i64s(&s, 2), vec![rows[0]]);
    assert_eq!(i64s(&s, 3), vec![*rows.last().unwrap()]);
    let av = s
        .column(4)
        .as_primitive::<arrow_array::types::Float64Type>()
        .value(0);
    assert!((av - sv as f64 / rows.len() as f64).abs() < 1e-9, "{av}");
    // No chunk can match.
    let z = e
        .sql("SELECT COUNT(*) AS n FROM pq WHERE v > 5000")
        .unwrap()
        .into_record_batch();
    assert_eq!(i64s(&z, 0), vec![0]);
    std::fs::remove_dir_all(&dir).ok();
}
