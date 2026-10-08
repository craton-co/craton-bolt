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
