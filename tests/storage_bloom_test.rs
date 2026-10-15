// SPDX-License-Identifier: Apache-2.0

//! Bloom-filter pruning: equality and `IN` predicates skip Parquet row
//! groups whose min / max cannot rule the value out.
//!
//! `cargo test --test storage_bloom_test -- --ignored --test-threads=1`

#![cfg(feature = "storage")]

use std::sync::Arc;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::storage::{write_parquet, ParquetWriteOptions};
use craton_bolt::Engine;

const ROWS: i64 = 20_000;
const GROUP: usize = 1_000;

/// Ids are spread over every row group (`i * 7919 mod 40000`), so each row
/// group's min / max spans nearly the whole range.
fn batch() -> RecordBatch {
    let ids: Vec<i64> = (0..ROWS).map(|i| (i * 7919) % 40_000).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int64, false),
        ArrowField::new("s", ArrowDataType::Utf8, false),
        ArrowField::new("cat", ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(ids.clone())),
            Arc::new(StringArray::from(
                ids.iter().map(|i| format!("u{i}")).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                (0..ROWS).map(|i| i % 3).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

fn count(e: &Engine, sql: &str) -> i64 {
    e.sql(sql)
        .unwrap()
        .into_record_batch()
        .column(0)
        .as_primitive::<Int64Type>()
        .value(0)
}

fn dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("bolt-bloom-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[test]
#[ignore = "gpu:e2e — bloom filters skip row groups for equality predicates"]
fn bloom_filters_prune_equality_predicates() {
    let d = dir("pq");
    let opts = ParquetWriteOptions {
        row_group_rows: GROUP,
        ..Default::default()
    };
    write_parquet(&d.join("b.parquet"), &[batch()], &opts).unwrap();
    let plain = ParquetWriteOptions {
        row_group_rows: GROUP,
        bloom_filters: false,
        ..Default::default()
    };
    write_parquet(&d.join("plain.parquet"), &[batch()], &plain).unwrap();
    let mut e = Engine::new().expect("ctx");
    e.register_parquet("b", d.join("b.parquet")).unwrap();
    e.register_parquet("plain", d.join("plain.parquet"))
        .unwrap();
    let groups = (ROWS as usize).div_ceil(GROUP);

    // 7919 * 5 % 40000 = 39595: present once.
    assert_eq!(count(&e, "SELECT COUNT(*) FROM b WHERE id = 39595"), 1);
    let read = e.last_scan("b").unwrap().chunks_read;
    assert!(read <= 2, "read {read} of {groups} row groups");
    assert_eq!(count(&e, "SELECT COUNT(*) FROM plain WHERE id = 39595"), 1);
    assert_eq!(e.last_scan("plain").unwrap().chunks_read, groups);

    // A value inside every zone map but absent: (almost) nothing is read.
    assert_eq!(count(&e, "SELECT COUNT(*) FROM b WHERE id = 2"), 0);
    assert!(e.last_scan("b").unwrap().chunks_read <= 1);

    // Strings and IN lists.
    assert_eq!(count(&e, "SELECT COUNT(*) FROM b WHERE s = 'u39595'"), 1);
    assert!(e.last_scan("b").unwrap().chunks_read <= 2);
    assert_eq!(
        count(&e, "SELECT COUNT(*) FROM b WHERE id IN (39595, 7919, 2)"),
        2
    );
    assert!(e.last_scan("b").unwrap().chunks_read <= 4);

    // Low-cardinality columns carry no filter; results stay exact.
    assert_eq!(
        count(&e, "SELECT COUNT(*) FROM b WHERE cat = 1"),
        (0..ROWS).filter(|i| i % 3 == 1).count() as i64
    );
    assert_eq!(e.last_scan("b").unwrap().chunks_read, groups);
    std::fs::remove_dir_all(&d).ok();
}

#[test]
#[ignore = "gpu:e2e — catalog files carry bloom filters"]
fn catalog_tables_get_bloom_filters() {
    let d = dir("cat");
    let mut e = Engine::builder().database(&d).build().expect("ctx");
    e.execute_sql("CREATE TABLE t (id BIGINT, s VARCHAR, cat BIGINT)")
        .unwrap();
    // Several appends: one data file each.
    for part in 0..4 {
        let b = batch();
        let n = b.num_rows();
        let piece = b.slice(part * n / 4, n / 4);
        e.insert_batch("t", piece).unwrap();
    }
    assert_eq!(count(&e, "SELECT COUNT(*) FROM t WHERE id = 39595"), 1);
    let scan = e.last_scan("t").unwrap();
    assert!(
        scan.chunks_read < scan.chunks_total,
        "{scan:?}: bloom filters should skip files"
    );
    std::fs::remove_dir_all(&d).ok();
}
