// SPDX-License-Identifier: Apache-2.0

//! `DELETE`, `UPDATE`, `ALTER TABLE` and `BEGIN` / `COMMIT` / `ROLLBACK`,
//! over in-memory and catalog tables, and transactions across sessions.
//!
//! Every test builds an `Engine` (a CUDA context):
//! `cargo test --test dml_test -- --ignored --test-threads=1`.

#![cfg(feature = "storage")]

use std::path::PathBuf;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};

use craton_bolt::{Database, Engine, QueryHandle};

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "bolt-dml-{tag}-{}-{}",
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

/// `(a, s)` rows of `SELECT a, s FROM t ORDER BY a`.
fn rows(e: &Engine, sql: &str) -> Vec<(i64, Option<String>)> {
    let b = batch(e.sql(sql).unwrap());
    let a = b.column(0).as_primitive::<Int64Type>();
    let s = b.column(1).as_string::<i32>();
    (0..b.num_rows())
        .map(|i| (a.value(i), (!s.is_null(i)).then(|| s.value(i).to_string())))
        .collect()
}

fn seed(e: &mut Engine) {
    e.execute_sql(
        "CREATE TABLE t (a BIGINT NOT NULL, s VARCHAR, v BIGINT); \
         INSERT INTO t VALUES (1, 'x', 10), (2, NULL, 20), (3, 'y', NULL), (4, 'x', 40)",
    )
    .unwrap();
}

fn delete_and_update(e: &mut Engine) {
    seed(e);
    // NULL predicate rows are not deleted.
    assert_eq!(
        affected(e.execute_sql("DELETE FROM t WHERE v > 15").unwrap()),
        2
    );
    assert_eq!(
        rows(e, "SELECT a, s FROM t ORDER BY a"),
        vec![(1, Some("x".into())), (3, Some("y".into()))]
    );
    assert_eq!(
        affected(e.execute_sql("DELETE FROM t WHERE a = 99").unwrap()),
        0
    );

    e.execute_sql("INSERT INTO t VALUES (5, 'z', 50), (6, NULL, 60)")
        .unwrap();
    assert_eq!(
        affected(
            e.execute_sql("UPDATE t SET v = v * 2, s = 'w' WHERE a >= 5")
                .unwrap()
        ),
        2
    );
    assert_eq!(scalar_i64(e, "SELECT SUM(v) FROM t"), 10 + 100 + 120);
    assert_eq!(scalar_i64(e, "SELECT COUNT(*) FROM t WHERE s = 'w'"), 2);
    // Subqueries see the table as it was before the statement.
    assert_eq!(
        affected(
            e.execute_sql("UPDATE t AS x SET v = 0 WHERE x.v < (SELECT MAX(v) FROM t)")
                .unwrap()
        ),
        2
    );
    assert_eq!(scalar_i64(e, "SELECT SUM(v) FROM t"), 120);
    // No WHERE: every row.
    assert_eq!(affected(e.execute_sql("UPDATE t SET v = a").unwrap()), 4);
    assert_eq!(scalar_i64(e, "SELECT SUM(v) FROM t"), 1 + 3 + 5 + 6);
    // NOT NULL holds for UPDATE.
    assert!(e.execute_sql("UPDATE t SET a = NULL WHERE a = 1").is_err());
    assert_eq!(scalar_i64(e, "SELECT COUNT(*) FROM t"), 4);
    assert_eq!(affected(e.execute_sql("DELETE FROM t").unwrap()), 4);
    assert_eq!(scalar_i64(e, "SELECT COUNT(*) FROM t"), 0);
    // Unknown columns are rejected.
    assert!(e.execute_sql("UPDATE t SET nope = 1").is_err());
}

#[test]
#[ignore = "gpu:e2e — DELETE / UPDATE over an in-memory table"]
fn in_memory_delete_and_update() {
    let mut e = Engine::new().expect("ctx");
    delete_and_update(&mut e);
}

#[test]
#[ignore = "gpu:e2e — DELETE / UPDATE over a catalog table"]
fn catalog_delete_and_update() {
    let dir = temp_dir("cat");
    {
        let mut e = Engine::builder().database(&dir).build().expect("ctx");
        delete_and_update(&mut e);
        e.execute_sql("INSERT INTO t VALUES (7, 'k', 70)").unwrap();
    }
    let e = Engine::builder().database(&dir).build().expect("ctx");
    assert_eq!(scalar_i64(&e, "SELECT SUM(v) FROM t"), 70);
}

#[test]
#[ignore = "gpu:e2e — DELETE rewrites only the files its zone maps may match"]
fn catalog_delete_rewrites_only_touched_files() {
    let dir = temp_dir("files");
    let mut e = Engine::builder().database(&dir).build().expect("ctx");
    e.execute_sql("CREATE TABLE f (k BIGINT, v BIGINT)")
        .unwrap();
    for lo in [0, 100, 200] {
        e.execute_sql(&format!(
            "INSERT INTO f SELECT k, k FROM (VALUES ({lo}), ({}), ({})) AS x(k)",
            lo + 1,
            lo + 2
        ))
        .unwrap();
    }
    let files_before: Vec<String> = e
        .catalog()
        .unwrap()
        .table("f")
        .unwrap()
        .files
        .iter()
        .map(|f| f.path.clone())
        .collect();
    assert_eq!(files_before.len(), 3);
    assert_eq!(
        affected(e.execute_sql("DELETE FROM f WHERE k = 101").unwrap()),
        1
    );
    let files_after: Vec<String> = e
        .catalog()
        .unwrap()
        .table("f")
        .unwrap()
        .files
        .iter()
        .map(|f| f.path.clone())
        .collect();
    assert_eq!(files_after.len(), 3);
    assert!(files_after.contains(&files_before[0]));
    assert!(files_after.contains(&files_before[2]));
    assert!(!files_after.contains(&files_before[1]));
    assert_eq!(scalar_i64(&e, "SELECT SUM(v) FROM f"), 3 + 303 + 603 - 101);
    // A predicate no file can match commits nothing.
    let v = e.catalog().unwrap().version();
    assert_eq!(
        affected(e.execute_sql("DELETE FROM f WHERE k > 1000").unwrap()),
        0
    );
    assert_eq!(e.catalog().unwrap().version(), v);
}

fn alter(e: &mut Engine) {
    seed(e);
    e.execute_sql("ALTER TABLE t ADD COLUMN w BIGINT DEFAULT 7")
        .unwrap();
    assert_eq!(scalar_i64(e, "SELECT SUM(w) FROM t"), 28);
    e.execute_sql("ALTER TABLE t ADD COLUMN IF NOT EXISTS w BIGINT")
        .unwrap();
    e.execute_sql("ALTER TABLE t ADD COLUMN n DOUBLE").unwrap();
    assert_eq!(scalar_i64(e, "SELECT COUNT(*) FROM t WHERE n IS NULL"), 4);
    assert!(e
        .execute_sql("ALTER TABLE t ADD COLUMN m BIGINT NOT NULL")
        .is_err());
    e.execute_sql("ALTER TABLE t DROP COLUMN n").unwrap();
    e.execute_sql("ALTER TABLE t RENAME COLUMN w TO weight")
        .unwrap();
    assert_eq!(scalar_i64(e, "SELECT SUM(weight) FROM t"), 28);
    assert!(e.sql("SELECT w FROM t").is_err());
    e.execute_sql("ALTER TABLE t ALTER COLUMN v SET DATA TYPE DOUBLE")
        .unwrap();
    let b = batch(e.sql("SELECT v FROM t").unwrap());
    assert_eq!(
        b.schema().field(0).data_type(),
        &arrow_schema::DataType::Float64
    );
    // `v` has a NULL.
    assert!(e
        .execute_sql("ALTER TABLE t ALTER COLUMN v SET NOT NULL")
        .is_err());
    e.execute_sql("ALTER TABLE t ALTER COLUMN a DROP NOT NULL")
        .unwrap();
    e.execute_sql("INSERT INTO t (s) VALUES ('nil')").unwrap();
    e.execute_sql("ALTER TABLE t RENAME TO t2").unwrap();
    assert!(e.sql("SELECT COUNT(*) FROM t").is_err());
    assert_eq!(scalar_i64(e, "SELECT COUNT(*) FROM t2"), 5);
    let d = batch(e.execute_sql("DESCRIBE t2").unwrap());
    let names: Vec<String> = (0..d.num_rows())
        .map(|i| d.column(0).as_string::<i32>().value(i).to_string())
        .collect();
    assert_eq!(names, vec!["a", "s", "v", "weight"]);
}

#[test]
#[ignore = "gpu:e2e — ALTER TABLE over in-memory and catalog tables"]
fn alter_table() {
    {
        let mut e = Engine::new().expect("ctx");
        alter(&mut e);
    }
    let dir = temp_dir("alter");
    {
        let mut e = Engine::builder().database(&dir).build().expect("ctx");
        alter(&mut e);
    }
    let e = Engine::builder().database(&dir).build().expect("ctx");
    assert_eq!(scalar_i64(&e, "SELECT SUM(weight) FROM t2"), 28);
}

#[test]
#[ignore = "gpu:e2e — views block column changes of their base tables"]
fn alter_respects_materialized_views() {
    let mut e = Engine::new().expect("ctx");
    seed(&mut e);
    e.execute_sql("CREATE MATERIALIZED VIEW mv AS SELECT SUM(v) AS s FROM t")
        .unwrap();
    assert!(e.execute_sql("ALTER TABLE t DROP COLUMN s").is_err());
    assert!(e.execute_sql("ALTER TABLE t RENAME TO u").is_err());
    e.execute_sql("ALTER TABLE t ADD COLUMN extra BIGINT")
        .unwrap();
    assert!(!e.view_status("mv").unwrap().fresh);
    assert!(e.execute_sql("DELETE FROM mv").is_err());
}

fn transactions(e: &mut Engine) {
    seed(e);
    e.execute_sql("BEGIN; INSERT INTO t VALUES (5, 'a', 50); DELETE FROM t WHERE a = 1")
        .unwrap();
    assert!(e.in_transaction());
    // The transaction sees its own changes.
    assert_eq!(scalar_i64(e, "SELECT COUNT(*) FROM t"), 4);
    e.execute_sql("CREATE TABLE extra AS SELECT a FROM t; DROP TABLE t")
        .unwrap();
    e.execute_sql("ROLLBACK").unwrap();
    assert!(!e.in_transaction());
    assert_eq!(scalar_i64(e, "SELECT SUM(a) FROM t"), 10);
    assert!(e.sql("SELECT COUNT(*) FROM extra").is_err());

    e.execute_sql("START TRANSACTION; UPDATE t SET v = 1; COMMIT")
        .unwrap();
    assert_eq!(scalar_i64(e, "SELECT SUM(v) FROM t"), 4);

    // A failing statement aborts the transaction.
    e.execute_sql("BEGIN").unwrap();
    e.execute_sql("DELETE FROM t WHERE a = 4").unwrap();
    assert!(e.execute_sql("INSERT INTO nope VALUES (1)").is_err());
    let err = e.execute_sql("SELECT 1 FROM t").unwrap_err().to_string();
    assert!(err.contains("aborted"), "{err}");
    assert!(e.execute_sql("COMMIT").is_err());
    assert!(!e.in_transaction());
    assert_eq!(scalar_i64(e, "SELECT COUNT(*) FROM t"), 4);

    assert!(e.execute_sql("COMMIT").is_err());
    e.execute_sql("BEGIN").unwrap();
    assert!(e.execute_sql("BEGIN").is_err());
    e.execute_sql("ROLLBACK").unwrap();
}

#[test]
#[ignore = "gpu:e2e — transactions over in-memory tables"]
fn in_memory_transactions() {
    let mut e = Engine::new().expect("ctx");
    transactions(&mut e);
}

#[test]
#[ignore = "gpu:e2e — catalog transactions publish one snapshot"]
fn catalog_transactions() {
    let dir = temp_dir("txn");
    {
        let mut e = Engine::builder().database(&dir).build().expect("ctx");
        transactions(&mut e);
        let v = e.catalog().unwrap().version();
        e.execute_sql(
            "BEGIN; INSERT INTO t VALUES (10, 'p', 1); INSERT INTO t VALUES (11, 'q', 1); \
             CREATE TABLE side (z BIGINT); INSERT INTO side VALUES (5)",
        )
        .unwrap();
        // Nothing is published before COMMIT.
        assert!(craton_bolt::storage::Catalog::open_snapshot(&dir, v + 1).is_err());
        e.execute_sql("COMMIT").unwrap();
        assert_eq!(e.catalog().unwrap().version(), v + 1);
        // VACUUM is refused inside a transaction.
        e.execute_sql("BEGIN").unwrap();
        assert!(e.execute_sql("VACUUM").is_err());
        e.execute_sql("ROLLBACK").unwrap();
        // An uncommitted transaction is lost when the engine drops.
        e.execute_sql("BEGIN; DELETE FROM t").unwrap();
    }
    let mut e = Engine::builder().database(&dir).build().expect("ctx");
    assert_eq!(scalar_i64(&e, "SELECT COUNT(*) FROM t"), 6);
    assert_eq!(scalar_i64(&e, "SELECT SUM(z) FROM side"), 5);
    e.execute_sql("VACUUM").unwrap();
}

#[test]
#[ignore = "gpu:e2e — a session's transaction holds other sessions back"]
fn session_transactions_are_serializable() {
    let db = Database::open(Engine::builder()).expect("db");
    let a = db.session();
    let b = db.session();
    a.sql("CREATE TABLE t (x BIGINT); INSERT INTO t VALUES (1)")
        .unwrap();
    a.sql("BEGIN; INSERT INTO t VALUES (2)").unwrap();
    // `b` waits for `a`'s transaction, then sees its outcome.
    let pending = b.submit("SELECT SUM(x) FROM t");
    std::thread::sleep(Duration::from_millis(100));
    a.sql("INSERT INTO t VALUES (3); COMMIT").unwrap();
    let r = pending.wait().unwrap();
    assert_eq!(r.column(0).as_primitive::<Int64Type>().value(0), 6);

    // A waiting statement times out.
    db.set_transaction_wait(Duration::from_millis(50));
    a.sql("BEGIN").unwrap();
    let err = b.sql("SELECT COUNT(*) FROM t").unwrap_err().to_string();
    assert!(err.contains("timed out"), "{err}");
    a.sql("ROLLBACK").unwrap();
    db.set_transaction_wait(Duration::from_secs(30));

    // Dropping the session rolls its transaction back.
    let c = db.session();
    c.sql("BEGIN; DELETE FROM t").unwrap();
    drop(c);
    let r = b.sql("SELECT COUNT(*) FROM t").unwrap();
    assert_eq!(r.column(0).as_primitive::<Int64Type>().value(0), 3);
}
