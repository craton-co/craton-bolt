// SPDX-License-Identifier: Apache-2.0

//! SQL DDL / DML, COPY, the persistent catalog, and materialized views.
//!
//! Every test builds an `Engine` (a CUDA context):
//! `cargo test --test database_test -- --ignored --test-threads=1`.

#![cfg(feature = "storage")]

use std::path::PathBuf;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::RecordBatch;

use craton_bolt::storage::Catalog;
use craton_bolt::{Engine, QueryHandle, RefreshMode};

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "bolt-db-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn batch(h: QueryHandle) -> RecordBatch {
    h.into_record_batch()
}

fn scalar_i64(e: &Engine, sql: &str) -> i64 {
    let b = batch(e.sql(sql).unwrap());
    assert_eq!(b.num_rows(), 1, "{sql}");
    b.column(0).as_primitive::<Int64Type>().value(0)
}

fn affected(h: QueryHandle) -> i64 {
    batch(h).column(0).as_primitive::<Int64Type>().value(0)
}

#[test]
#[ignore = "gpu:e2e — in-memory DDL and DML"]
fn in_memory_ddl_and_dml() {
    let mut e = Engine::new().expect("ctx");
    e.execute_sql("CREATE TABLE t (a BIGINT NOT NULL, s VARCHAR(10), d DOUBLE)")
        .unwrap();
    assert_eq!(scalar_i64(&e, "SELECT COUNT(*) FROM t"), 0);
    let n = affected(
        e.execute_sql("INSERT INTO t VALUES (1, 'x', 1.5), (2, NULL, 2.5), (3, 'y', NULL)")
            .unwrap(),
    );
    assert_eq!(n, 3);
    let n = affected(
        e.execute_sql("INSERT INTO t (a, s) SELECT a + 10, s FROM t WHERE a < 3")
            .unwrap(),
    );
    assert_eq!(n, 2);
    assert_eq!(scalar_i64(&e, "SELECT SUM(a) FROM t"), 1 + 2 + 3 + 11 + 12);
    assert_eq!(scalar_i64(&e, "SELECT COUNT(*) FROM t WHERE s = 'x'"), 2);

    // NOT NULL is enforced.
    assert!(e.execute_sql("INSERT INTO t (s) VALUES ('z')").is_err());
    // sql() refuses statements and points at execute_sql.
    let err = e
        .sql("INSERT INTO t VALUES (9, 'q', 0.0)")
        .unwrap_err()
        .to_string();
    assert!(err.contains("execute_sql"), "{err}");

    // CTAS, multi-statement scripts, SHOW / DESCRIBE.
    let last = e
        .execute_sql(
            "CREATE TABLE big AS SELECT a, d FROM t WHERE a > 2; \
             CREATE TABLE IF NOT EXISTS big (z INT); \
             SELECT COUNT(*) FROM big",
        )
        .unwrap();
    assert_eq!(
        batch(last).column(0).as_primitive::<Int64Type>().value(0),
        3
    );
    let tables = batch(e.execute_sql("SHOW TABLES").unwrap());
    let names: Vec<&str> = (0..tables.num_rows())
        .map(|i| tables.column(0).as_string::<i32>().value(i))
        .collect();
    assert_eq!(names, vec!["big", "t"]);
    let desc = batch(e.execute_sql("DESCRIBE t").unwrap());
    assert_eq!(desc.num_rows(), 3);
    assert_eq!(desc.column(1).as_string::<i32>().value(0), "BIGINT");

    // ANALYZE: one row per column with exact statistics.
    let st = batch(e.execute_sql("ANALYZE t").unwrap());
    assert_eq!(st.num_rows(), 3);
    assert_eq!(st.column(1).as_primitive::<Int64Type>().value(0), 5); // rows
    assert_eq!(st.column(3).as_string::<i32>().value(0), "1"); // min(a)
    assert_eq!(st.column(4).as_string::<i32>().value(0), "12"); // max(a)
    assert_eq!(st.column(2).as_primitive::<Int64Type>().value(1), 2); // nulls(s)

    e.execute_sql("TRUNCATE TABLE big").unwrap();
    assert_eq!(scalar_i64(&e, "SELECT COUNT(*) FROM big"), 0);
    e.execute_sql("CREATE OR REPLACE TABLE big AS SELECT a FROM t")
        .unwrap();
    assert_eq!(scalar_i64(&e, "SELECT COUNT(*) FROM big"), 5);
    e.execute_sql("DROP TABLE big").unwrap();
    assert!(e.sql("SELECT * FROM big").is_err());
    e.execute_sql("DROP TABLE IF EXISTS big").unwrap();
    assert!(e.execute_sql("DROP TABLE big").is_err());
}

#[test]
#[ignore = "gpu:e2e — COPY to and from Parquet and CSV"]
fn copy_round_trips() {
    let dir = temp_dir("copy");
    let mut e = Engine::new().expect("ctx");
    e.execute_sql(
        "CREATE TABLE src AS SELECT * FROM (VALUES (1, 'a'), (2, 'b'), (3, 'c')) AS v(k, s)",
    )
    .unwrap();
    let pq = dir.join("out.parquet");
    let csv = dir.join("out.csv");
    let n = affected(
        e.execute_sql(&format!("COPY src TO '{}'", pq.display()))
            .unwrap(),
    );
    assert_eq!(n, 3);
    let n = affected(
        e.execute_sql(&format!(
            "COPY (SELECT k, s FROM src WHERE k > 1) TO '{}'",
            csv.display()
        ))
        .unwrap(),
    );
    assert_eq!(n, 2);
    let text = std::fs::read_to_string(&csv).unwrap();
    assert!(text.starts_with("k,s\n"), "{text}");

    e.execute_sql("CREATE TABLE dst (k BIGINT, s VARCHAR)")
        .unwrap();
    e.execute_sql(&format!("COPY dst FROM '{}'", pq.display()))
        .unwrap();
    e.execute_sql(&format!(
        "COPY dst FROM '{}' (FORMAT csv, HEADER true)",
        csv.display()
    ))
    .unwrap();
    assert_eq!(scalar_i64(&e, "SELECT COUNT(*) FROM dst"), 5);
    assert_eq!(scalar_i64(&e, "SELECT SUM(k) FROM dst"), 1 + 2 + 3 + 2 + 3);

    // An external Parquet table over the exported file.
    e.execute_sql(&format!(
        "CREATE EXTERNAL TABLE ext STORED AS PARQUET LOCATION '{}'",
        pq.display()
    ))
    .unwrap();
    assert_eq!(scalar_i64(&e, "SELECT MAX(k) FROM ext"), 3);
    assert!(
        e.execute_sql("INSERT INTO ext VALUES (4, 'd')").is_err(),
        "external tables are read-only"
    );
    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[ignore = "gpu:e2e — durable catalog: reopen, snapshots, compaction, vacuum"]
fn catalog_survives_restart_and_time_travels() {
    let dir = temp_dir("catalog");
    let v_after_first_insert;
    {
        let mut e = Engine::builder().database(&dir).build().expect("ctx");
        e.execute_sql("CREATE TABLE sales (id BIGINT NOT NULL, region VARCHAR, amount BIGINT)")
            .unwrap();
        e.execute_sql("INSERT INTO sales VALUES (1, 'eu', 10), (2, 'us', 20), (3, 'eu', 30)")
            .unwrap();
        v_after_first_insert = e.catalog().unwrap().version();
        e.execute_sql("INSERT INTO sales VALUES (4, 'apac', 40)")
            .unwrap();
        e.execute_sql("CREATE TEMP TABLE scratch AS SELECT * FROM (VALUES (1)) AS v(x)")
            .unwrap();
        assert_eq!(scalar_i64(&e, "SELECT SUM(amount) FROM sales"), 100);
        // The writer lock refuses a second writer.
        assert!(Catalog::open(&dir).is_err());
    }
    // Reopen: data, files and schema persist; temp tables do not.
    {
        let mut e = Engine::builder().database(&dir).build().expect("ctx");
        assert_eq!(scalar_i64(&e, "SELECT SUM(amount) FROM sales"), 100);
        assert_eq!(
            scalar_i64(&e, "SELECT COUNT(*) FROM sales WHERE region = 'eu'"),
            2
        );
        assert!(e.sql("SELECT * FROM scratch").is_err());
        // Zone maps prune the newest file for an id range it cannot hold.
        assert_eq!(
            scalar_i64(&e, "SELECT COUNT(*) FROM sales WHERE id <= 3"),
            3
        );
        let r = e.last_scan("sales").unwrap();
        assert_eq!((r.chunks_total, r.chunks_read), (2, 1));
        assert_eq!(e.catalog().unwrap().table("sales").unwrap().files.len(), 2);
        assert_eq!(affected(e.execute_sql("COMPACT TABLE sales").unwrap()), 1);
        assert_eq!(scalar_i64(&e, "SELECT SUM(amount) FROM sales"), 100);
        e.execute_sql("ANALYZE sales").unwrap();
        let st = e.table_statistics("sales").unwrap();
        assert_eq!(st.columns[1].distinct_estimate, Some(3));
    }
    // Time travel: the snapshot after the first insert, read-only.
    {
        let mut e = Engine::builder()
            .database_snapshot(&dir, v_after_first_insert)
            .build()
            .expect("ctx");
        assert_eq!(scalar_i64(&e, "SELECT SUM(amount) FROM sales"), 60);
        assert!(e
            .execute_sql("INSERT INTO sales VALUES (9, 'x', 1)")
            .is_err());
        e.execute_sql("CREATE TEMP TABLE ok AS SELECT * FROM (VALUES (1)) AS v(x)")
            .unwrap();
    }
    // Vacuum keeps only the current snapshot and its files.
    {
        let mut e = Engine::builder().database(&dir).build().expect("ctx");
        let removed = affected(e.execute_sql("VACUUM").unwrap());
        assert_eq!(removed, 2, "the two pre-compaction files");
        assert_eq!(e.catalog().unwrap().snapshots().unwrap().len(), 1);
        assert_eq!(scalar_i64(&e, "SELECT SUM(amount) FROM sales"), 100);
        e.execute_sql("DROP TABLE sales").unwrap();
    }
    {
        let e = Engine::builder().database(&dir).build().expect("ctx");
        assert!(e.table_names().is_empty());
    }
    std::fs::remove_dir_all(dir).ok();
}

#[test]
#[ignore = "gpu:e2e — in-memory materialized views"]
fn materialized_views_track_staleness_and_refresh() {
    let mut e = Engine::new().expect("ctx");
    e.execute_sql("CREATE TABLE f (g BIGINT, v BIGINT)")
        .unwrap();
    e.execute_sql("INSERT INTO f VALUES (1, 10), (1, 5), (2, 7)")
        .unwrap();
    e.execute_sql("CREATE MATERIALIZED VIEW agg AS SELECT g, SUM(v) AS total FROM f GROUP BY g")
        .unwrap();
    let st = e.view_status("agg").unwrap();
    assert!(st.fresh);
    assert_eq!(st.dependencies, vec!["f".to_string()]);
    assert_eq!(st.refresh, RefreshMode::Manual);
    assert_eq!(scalar_i64(&e, "SELECT SUM(total) FROM agg"), 22);

    // Manual: a base change leaves the view stale until REFRESH.
    e.execute_sql("INSERT INTO f VALUES (3, 100)").unwrap();
    let st = e.view_status("agg").unwrap();
    assert!(!st.fresh);
    assert_eq!(st.stale_dependencies, vec!["f".to_string()]);
    assert_eq!(scalar_i64(&e, "SELECT SUM(total) FROM agg"), 22);
    assert_eq!(
        affected(e.execute_sql("REFRESH MATERIALIZED VIEW agg").unwrap()),
        3
    );
    assert!(e.view_status("agg").unwrap().fresh);
    assert_eq!(scalar_i64(&e, "SELECT SUM(total) FROM agg"), 122);
    assert!(e.execute_sql("INSERT INTO agg VALUES (9, 9)").is_err());

    // On commit: refreshed by the changing statement itself; a view over the
    // view follows.
    e.execute_sql(
        "CREATE MATERIALIZED VIEW live WITH (refresh = 'on_commit') AS \
         SELECT COUNT(*) AS n FROM f",
    )
    .unwrap();
    e.execute_sql("CREATE MATERIALIZED VIEW live2 WITH (refresh = 'on_commit') AS SELECT n * 2 AS n2 FROM live")
        .unwrap();
    e.execute_sql("INSERT INTO f VALUES (4, 1)").unwrap();
    assert_eq!(scalar_i64(&e, "SELECT n FROM live"), 5);
    assert_eq!(scalar_i64(&e, "SELECT n2 FROM live2"), 10);
    // A change through the Rust API is picked up before the next statement.
    let extra = batch(e.sql("SELECT g, v FROM f WHERE g = 4").unwrap());
    e.register_batch("f", extra).unwrap();
    assert!(!e.view_status("live").unwrap().fresh);
    e.execute_sql("SELECT COUNT(*) FROM f").unwrap();
    assert_eq!(scalar_i64(&e, "SELECT n FROM live"), 6);
    e.execute_sql("DROP MATERIALIZED VIEW live2").unwrap();
    e.execute_sql("DROP MATERIALIZED VIEW IF EXISTS live2")
        .unwrap();
    assert!(!e.view_exists("live2"));
    assert!(
        e.execute_sql("DROP TABLE live").is_err(),
        "views drop as views"
    );
}

#[test]
#[ignore = "gpu:e2e — durable materialized views"]
fn durable_views_persist_definitions_and_freshness() {
    let dir = temp_dir("views");
    {
        let mut e = Engine::builder().database(&dir).build().expect("ctx");
        e.execute_sql("CREATE TABLE f (g BIGINT, v BIGINT)")
            .unwrap();
        e.execute_sql("INSERT INTO f VALUES (1, 10), (2, 20)")
            .unwrap();
        e.execute_sql(
            "CREATE MATERIALIZED VIEW s WITH (refresh = 'on_commit') AS SELECT SUM(v) AS t FROM f",
        )
        .unwrap();
        e.execute_sql("CREATE MATERIALIZED VIEW m AS SELECT COUNT(*) AS n FROM f")
            .unwrap();
        e.execute_sql("CREATE TEMP TABLE tmp AS SELECT * FROM (VALUES (1)) AS v(x)")
            .unwrap();
        assert!(
            e.execute_sql("CREATE MATERIALIZED VIEW bad AS SELECT x FROM tmp")
                .is_err(),
            "a durable view cannot read a temp table"
        );
        e.execute_sql("INSERT INTO f VALUES (3, 30)").unwrap();
        assert_eq!(scalar_i64(&e, "SELECT t FROM s"), 60);
    }
    {
        let mut e = Engine::builder().database(&dir).build().expect("ctx");
        assert_eq!(e.view_names(), vec!["m".to_string(), "s".to_string()]);
        assert!(e.view_status("s").unwrap().fresh);
        // The manual view's staleness survived the restart.
        assert!(!e.view_status("m").unwrap().fresh);
        assert_eq!(scalar_i64(&e, "SELECT n FROM m"), 2);
        e.execute_sql("REFRESH MATERIALIZED VIEW m").unwrap();
        assert_eq!(scalar_i64(&e, "SELECT n FROM m"), 3);
        e.execute_sql("INSERT INTO f VALUES (4, 40)").unwrap();
        assert_eq!(scalar_i64(&e, "SELECT t FROM s"), 100);
        e.execute_sql("DROP MATERIALIZED VIEW s").unwrap();
    }
    {
        let e = Engine::builder().database(&dir).build().expect("ctx");
        assert_eq!(e.view_names(), vec!["m".to_string()]);
        assert!(!e.has_view_table_named("s"));
    }
    std::fs::remove_dir_all(dir).ok();
}

trait HasTable {
    fn has_view_table_named(&self, name: &str) -> bool;
}

impl HasTable for Engine {
    fn has_view_table_named(&self, name: &str) -> bool {
        self.table_names().iter().any(|t| t == name)
    }
}

#[test]
fn catalog_lock_is_released_on_close() {
    let dir = temp_dir("lock");
    {
        let _a = Catalog::open(&dir).unwrap();
        assert!(Catalog::open(&dir).is_err());
    }
    let _b = Catalog::open(&dir).unwrap();
    drop(_b);
    std::fs::remove_dir_all(dir).ok();
}
