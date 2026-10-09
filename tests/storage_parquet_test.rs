// SPDX-License-Identifier: Apache-2.0

//! Parquet / storage-table scans through the engine.
//!
//! Host-only tests (statistics, pruning decisions) run everywhere; the
//! `#[ignore = "gpu:..."]` ones execute queries and need a CUDA device:
//! `cargo test --test storage_parquet_test -- --ignored --test-threads=1`.

#![cfg(feature = "storage")]

use std::path::PathBuf;
use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::{Float64Type, Int64Type};
use arrow_array::{Float64Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::logical_plan::Literal;
use craton_bolt::storage::{
    write_parquet, MemoryTable, ParquetTable, ParquetWriteOptions, StorageTable,
};
use craton_bolt::{metrics, Counter, Engine};

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "bolt-storage-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// `(k Int64 = lo.., g Int64 = k % 7, v Float64 = k / 2, s Utf8 = "c{k % 3}")`.
fn batch(lo: i64, n: usize) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int64, false),
        ArrowField::new("g", ArrowDataType::Int64, false),
        ArrowField::new("v", ArrowDataType::Float64, false),
        ArrowField::new("s", ArrowDataType::Utf8, false),
    ]));
    let k: Vec<i64> = (lo..lo + n as i64).collect();
    let g: Vec<i64> = k.iter().map(|x| x % 7).collect();
    let v: Vec<f64> = k.iter().map(|x| *x as f64 / 2.0).collect();
    let s: Vec<String> = k.iter().map(|x| format!("c{}", x % 3)).collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(k)),
            Arc::new(Int64Array::from(g)),
            Arc::new(Float64Array::from(v)),
            Arc::new(StringArray::from(s)),
        ],
    )
    .unwrap()
}

/// 10 000 rows in 10 row groups of 1 000 (k ascending → tight zone maps).
fn write_fixture(dir: &std::path::Path) -> PathBuf {
    let p = dir.join("t.parquet");
    let opts = ParquetWriteOptions {
        row_group_rows: 1000,
        ..Default::default()
    };
    write_parquet(&p, &[batch(0, 10_000)], &opts).unwrap();
    p
}

fn i64_at(b: &RecordBatch, col: usize, row: usize) -> i64 {
    b.column(col).as_primitive::<Int64Type>().value(row)
}

#[test]
fn parquet_statistics_come_from_footers() {
    let dir = temp_dir("stats");
    let t = ParquetTable::open(write_fixture(&dir)).unwrap();
    assert_eq!(t.chunk_count(), 10);
    let s = t.statistics();
    assert_eq!(s.row_count, 10_000);
    assert_eq!(s.columns[0].min, Some(Literal::Int64(0)));
    assert_eq!(s.columns[0].max, Some(Literal::Int64(9_999)));
    assert_eq!(s.columns[3].min, Some(Literal::Utf8("c0".into())));
    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[ignore = "gpu:e2e — pruned Parquet scans answer like a full scan"]
fn parquet_filter_prunes_row_groups_and_columns() {
    let dir = temp_dir("prune");
    let path = write_fixture(&dir);
    let mut e = Engine::new().expect("ctx");
    e.register_parquet("t", &path).unwrap();
    let pruned0 = metrics().counter(Counter::StorageChunksPruned).get();
    // EXPLAIN reports the pruned scan without reading data.
    let plan = e
        .explain_sql("SELECT COUNT(*) AS n, SUM(k) AS sk FROM t WHERE k >= 7500")
        .unwrap();
    assert!(
        plan.contains(
            "Storage scans:
  t: 3 of 10 chunks (~3000 rows), 1 of 4 columns"
        ),
        "{plan}"
    );
    assert!(e.last_scan("t").is_none(), "explain must not scan");

    let out = e
        .sql("SELECT COUNT(*) AS n, SUM(k) AS sk FROM t WHERE k >= 7500")
        .unwrap()
        .into_record_batch();
    assert_eq!(i64_at(&out, 0, 0), 2500);
    assert_eq!(i64_at(&out, 1, 0), (7500..10_000).sum::<i64>());
    let r = e.last_scan("t").unwrap();
    assert_eq!((r.chunks_total, r.chunks_read), (10, 3));
    assert_eq!(r.columns_read, Some(1), "only k is read");
    assert!(!r.reused);
    assert!(metrics().counter(Counter::StorageChunksPruned).get() >= pruned0 + 7);

    // Same needs → the installed scan is reused.
    let again = e
        .sql("SELECT COUNT(*) FROM t WHERE k >= 8000")
        .unwrap()
        .into_record_batch();
    assert_eq!(i64_at(&again, 0, 0), 2000);
    assert!(e.last_scan("t").unwrap().reused);

    // Wider needs (another column, other rows) → re-scan, still exact.
    let wide = e
        .sql("SELECT SUM(v) FROM t WHERE k < 1500")
        .unwrap()
        .into_record_batch();
    let expect: f64 = (0..1500).map(|k| k as f64 / 2.0).sum();
    assert!((wide.column(0).as_primitive::<Float64Type>().value(0) - expect).abs() < 1e-6);
    let r = e.last_scan("t").unwrap();
    assert_eq!(r.chunks_read, 2);
    assert!(!r.reused);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[ignore = "gpu:e2e — every row group pruned"]
fn fully_pruned_scan_returns_empty_results() {
    let dir = temp_dir("empty");
    let mut e = Engine::new().expect("ctx");
    e.register_parquet("t", write_fixture(&dir)).unwrap();
    let rows = e
        .sql("SELECT k FROM t WHERE k > 100000")
        .unwrap()
        .into_record_batch();
    assert_eq!(rows.num_rows(), 0);
    let n = e
        .sql("SELECT COUNT(*) FROM t WHERE k < -5")
        .unwrap()
        .into_record_batch();
    assert_eq!(i64_at(&n, 0, 0), 0);
    assert_eq!(e.last_scan("t").unwrap().chunks_read, 0);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[ignore = "gpu:e2e — GROUP BY, string predicates, joins and subqueries over Parquet"]
fn parquet_tables_support_the_full_query_surface() {
    let dir = temp_dir("surface");
    let mut e = Engine::new().expect("ctx");
    e.register_parquet("t", write_fixture(&dir)).unwrap();

    let g = e
        .sql("SELECT g, COUNT(*) AS n FROM t WHERE s = 'c1' GROUP BY g ORDER BY g")
        .unwrap()
        .into_record_batch();
    assert_eq!(g.num_rows(), 7);
    let total: i64 = (0..7).map(|r| i64_at(&g, 1, r)).sum();
    assert_eq!(total, (0..10_000).filter(|k| k % 3 == 1).count() as i64);

    // Join with an in-memory dimension table.
    let dim = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("g", ArrowDataType::Int64, false),
            ArrowField::new("w", ArrowDataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from((0..7).collect::<Vec<i64>>())),
            Arc::new(Int64Array::from(
                (0..7).map(|x| x * 10).collect::<Vec<i64>>(),
            )),
        ],
    )
    .unwrap();
    e.register_table("dim", dim).unwrap();
    let j = e
        .sql("SELECT SUM(dim.w) FROM t JOIN dim ON t.g = dim.g WHERE t.k < 70")
        .unwrap()
        .into_record_batch();
    let expect: i64 = (0..70).map(|k| (k % 7) * 10).sum();
    assert_eq!(i64_at(&j, 0, 0), expect);

    // A subquery over the same table sees every row (k > AVG(k) = 4999.5).
    let sq = e
        .sql("SELECT COUNT(*) FROM t WHERE k > (SELECT AVG(k) FROM t) AND k < 6000")
        .unwrap()
        .into_record_batch();
    assert_eq!(i64_at(&sq, 0, 0), 1000);
    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[ignore = "gpu:e2e — Parquet aggregates under a memory budget"]
fn parquet_scan_runs_in_morsels_under_a_budget() {
    let dir = temp_dir("budget");
    let mut e = Engine::builder()
        .memory_budget(16 * 1024)
        .build()
        .expect("ctx");
    e.register_parquet("t", write_fixture(&dir)).unwrap();
    let out = e
        .sql("SELECT g, SUM(k) AS sk FROM t WHERE k >= 3000 GROUP BY g ORDER BY g")
        .unwrap()
        .into_record_batch();
    for row in 0..7 {
        let g = i64_at(&out, 0, row);
        let expect: i64 = (3000..10_000).filter(|k| k % 7 == g).sum();
        assert_eq!(i64_at(&out, 1, row), expect, "g={g}");
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[ignore = "gpu:e2e — Hive-partitioned directories prune whole partitions"]
fn hive_partitioned_tables_prune_partitions() {
    let dir = temp_dir("hive");
    for region in ["eu", "us", "apac"] {
        for day in 1..=3 {
            let d = dir
                .join(format!("region={region}"))
                .join(format!("day={day}"));
            std::fs::create_dir_all(&d).unwrap();
            write_parquet(
                &d.join("part-0.parquet"),
                &[batch(day * 1000, 100)],
                &Default::default(),
            )
            .unwrap();
        }
    }
    let mut e = Engine::new().expect("ctx");
    e.execute_sql(&format!(
        "CREATE EXTERNAL TABLE ev STORED AS PARQUET LOCATION '{}'",
        dir.display()
    ))
    .unwrap();
    let out = e
        .sql("SELECT COUNT(*) AS n, SUM(day) AS d FROM ev WHERE region = 'us' AND day >= 2")
        .unwrap()
        .into_record_batch();
    assert_eq!(i64_at(&out, 0, 0), 200);
    assert_eq!(i64_at(&out, 1, 0), 2 * 100 + 3 * 100);
    let r = e.last_scan("ev").unwrap();
    assert_eq!(
        (r.chunks_total, r.chunks_read),
        (9, 2),
        "two of nine partitions read"
    );
    let g = e
        .sql("SELECT region, COUNT(*) AS n FROM ev GROUP BY region ORDER BY region")
        .unwrap()
        .into_record_batch();
    assert_eq!(g.num_rows(), 3);
    assert_eq!(g.column(0).as_string::<i32>().value(0), "apac");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[ignore = "gpu:e2e — table statistics and storage-table registration"]
fn statistics_and_registration_contract() {
    let dir = temp_dir("reg");
    let mut e = Engine::new().expect("ctx");
    let path = write_fixture(&dir);
    e.register_parquet("t", &path).unwrap();
    assert!(e.register_parquet("t", &path).is_err(), "duplicate name");
    assert!(
        e.register_table("t", batch(0, 3)).is_err(),
        "taken by a storage table"
    );
    e.register_table("mem", batch(0, 100)).unwrap();
    assert!(e.register_parquet("mem", &path).is_err());

    let st = e.table_statistics("t").unwrap();
    assert_eq!(st.row_count, 10_000);
    let sm = e.table_statistics("mem").unwrap();
    assert_eq!(sm.row_count, 100);
    assert_eq!(sm.columns[1].distinct_estimate, Some(7));
    assert_eq!(sm.columns[0].max, Some(Literal::Int64(99)));
    assert_eq!(e.table_names(), vec!["mem".to_string(), "t".to_string()]);

    // A MemoryTable registers through the same contract.
    let mt = MemoryTable::try_new(vec![batch(0, 4000)], Some(500)).unwrap();
    e.register_storage_table("chunks", Arc::new(mt)).unwrap();
    let out = e
        .sql("SELECT MAX(k) FROM chunks WHERE k < 1200")
        .unwrap()
        .into_record_batch();
    assert_eq!(i64_at(&out, 0, 0), 1199);
    let r = e.last_scan("chunks").unwrap();
    assert_eq!((r.chunks_total, r.chunks_read), (8, 3));
    std::fs::remove_dir_all(dir).ok();
}
