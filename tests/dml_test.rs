// SPDX-License-Identifier: Apache-2.0

//! `DELETE`, `UPDATE`, `MERGE`, `ALTER TABLE`, `BEGIN` / `COMMIT` /
//! `ROLLBACK` and savepoints, over in-memory and catalog tables, and
//! concurrent transactions across sessions.
//!
//! Every test builds an `Engine` (a CUDA context):
//! `cargo test --test dml_test -- --ignored --test-threads=1`.

#![cfg(feature = "storage")]

use std::path::PathBuf;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};

use craton_bolt::{Database, Engine, EngineBuilder, IsolationLevel, QueryHandle, Session};

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
#[ignore = "gpu:e2e — serializable isolation holds other sessions back"]
fn session_transactions_are_serializable() {
    let db = Database::open(Engine::builder()).expect("db");
    db.set_isolation(IsolationLevel::Serializable);
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

fn session_scalar(s: &Session, sql: &str) -> i64 {
    let b = s.sql(sql).unwrap_or_else(|e| panic!("{sql}: {e}"));
    b.column(0).as_primitive::<Int64Type>().value(0)
}

/// Concurrent sessions under snapshot isolation: no waiting, no dirty
/// reads, private snapshots, merged commits, first committer wins.
fn concurrent_transactions(builder: EngineBuilder) {
    let db = Database::open(builder).expect("db");
    let (a, b, c) = (db.session(), db.session(), db.session());
    a.sql(
        "CREATE TABLE t (x BIGINT); INSERT INTO t VALUES (1); \
         CREATE TABLE u (y BIGINT); INSERT INTO u VALUES (10)",
    )
    .unwrap();
    a.sql("BEGIN; INSERT INTO t VALUES (2)").unwrap();
    // `b` runs at once and does not see `a`'s uncommitted row.
    assert_eq!(session_scalar(&b, "SELECT SUM(x) FROM t"), 1);
    b.sql("BEGIN; INSERT INTO u VALUES (20)").unwrap();
    assert_eq!(session_scalar(&a, "SELECT SUM(y) FROM u"), 10);
    assert_eq!(session_scalar(&a, "SELECT SUM(x) FROM t"), 3);
    b.sql("INSERT INTO t VALUES (100)").unwrap();
    a.sql("COMMIT").unwrap();
    // `b` keeps its snapshot: `a`'s commit is invisible to it.
    assert_eq!(session_scalar(&b, "SELECT SUM(x) FROM t"), 101);
    assert_eq!(session_scalar(&c, "SELECT SUM(x) FROM t"), 3);
    // Both changed `t`: `a` committed first, so `b` fails and rolls back.
    let err = b.sql("COMMIT").unwrap_err().to_string();
    assert!(err.contains("could not serialize"), "{err}");
    assert_eq!(session_scalar(&c, "SELECT SUM(y) FROM u"), 10);
    assert_eq!(session_scalar(&b, "SELECT SUM(x) FROM t"), 3);

    // Disjoint changes merge, in either commit order.
    a.sql("BEGIN; INSERT INTO t VALUES (3)").unwrap();
    b.sql("BEGIN; INSERT INTO u VALUES (30); CREATE TABLE w (z BIGINT); INSERT INTO w VALUES (7)")
        .unwrap();
    b.sql("COMMIT").unwrap();
    // `a` still sees `u` as of its BEGIN.
    assert_eq!(session_scalar(&a, "SELECT SUM(y) FROM u"), 10);
    a.sql("COMMIT").unwrap();
    assert_eq!(session_scalar(&c, "SELECT SUM(x) FROM t"), 6);
    assert_eq!(session_scalar(&c, "SELECT SUM(y) FROM u"), 40);
    assert_eq!(session_scalar(&c, "SELECT SUM(z) FROM w"), 7);

    // A single statement outside a transaction commits first.
    a.sql("BEGIN; DELETE FROM u").unwrap();
    c.sql("INSERT INTO u VALUES (1)").unwrap();
    assert_eq!(session_scalar(&a, "SELECT COUNT(*) FROM u"), 0);
    assert!(a.sql("COMMIT").is_err());
    assert_eq!(session_scalar(&c, "SELECT SUM(y) FROM u"), 41);

    // VACUUM waits for every transaction to end; dropping a session rolls
    // its transaction back.
    let d = db.session();
    d.sql("BEGIN; DELETE FROM t").unwrap();
    assert!(c.sql("VACUUM").is_err());
    drop(d);
    assert_eq!(session_scalar(&c, "SELECT SUM(x) FROM t"), 6);
}

#[test]
#[ignore = "gpu:e2e — concurrent snapshot-isolated transactions over in-memory tables"]
fn concurrent_in_memory_transactions() {
    concurrent_transactions(Engine::builder());
}

#[test]
#[ignore = "gpu:e2e — concurrent snapshot-isolated transactions over a catalog"]
fn concurrent_catalog_transactions() {
    let dir = temp_dir("concurrent");
    concurrent_transactions(Engine::builder().database(&dir));
    // Everything committed is durable.
    let e = Engine::builder().database(&dir).build().expect("ctx");
    assert_eq!(scalar_i64(&e, "SELECT SUM(x) FROM t"), 6);
    assert_eq!(scalar_i64(&e, "SELECT SUM(y) FROM u"), 41);
    assert_eq!(scalar_i64(&e, "SELECT SUM(z) FROM w"), 7);
}

fn savepoints(e: &mut Engine) {
    seed(e);
    e.execute_sql("BEGIN; INSERT INTO t VALUES (5, 'z', 50); SAVEPOINT s1; DELETE FROM t")
        .unwrap();
    assert_eq!(scalar_i64(e, "SELECT COUNT(*) FROM t"), 0);
    e.execute_sql("ROLLBACK TO SAVEPOINT s1").unwrap();
    assert_eq!(scalar_i64(e, "SELECT COUNT(*) FROM t"), 5);
    // The savepoint survives a rollback to it; later ones are released.
    e.execute_sql("SAVEPOINT s2; UPDATE t SET v = 0; RELEASE SAVEPOINT s2")
        .unwrap();
    assert!(e.execute_sql("ROLLBACK TO s2").is_err());
    assert_eq!(scalar_i64(e, "SELECT SUM(v) FROM t"), 0);
    // An error aborts the transaction; rolling back to a savepoint resumes it.
    assert!(e.execute_sql("INSERT INTO nope VALUES (1)").is_err());
    assert!(e.execute_sql("SELECT 1 FROM t").is_err());
    e.execute_sql("ROLLBACK TO s1").unwrap();
    e.execute_sql("CREATE TABLE side AS SELECT a FROM t; SAVEPOINT s3; DROP TABLE side")
        .unwrap();
    e.execute_sql("ROLLBACK TO SAVEPOINT s3").unwrap();
    assert_eq!(scalar_i64(e, "SELECT COUNT(*) FROM side"), 5);
    e.execute_sql("COMMIT").unwrap();
    assert_eq!(scalar_i64(e, "SELECT SUM(a) FROM t"), 15);
    assert_eq!(scalar_i64(e, "SELECT SUM(v) FROM t"), 120);
    assert_eq!(scalar_i64(e, "SELECT COUNT(*) FROM side"), 5);
    assert!(e.execute_sql("SAVEPOINT outside").is_err());
}

#[test]
#[ignore = "gpu:e2e — savepoints over in-memory and catalog tables"]
fn savepoints_roll_back_part_of_a_transaction() {
    let mut e = Engine::new().expect("ctx");
    savepoints(&mut e);
    drop(e);
    let dir = temp_dir("savepoint");
    let mut e = Engine::builder().database(&dir).build().expect("ctx");
    savepoints(&mut e);
    drop(e);
    let e = Engine::builder().database(&dir).build().expect("ctx");
    assert_eq!(scalar_i64(&e, "SELECT SUM(v) FROM t"), 120);
}

/// `(a, s, v)` rows of `t`, ordered by `a`.
fn all_rows(e: &Engine) -> Vec<(i64, Option<String>, Option<i64>)> {
    let b = batch(e.sql("SELECT a, s, v FROM t ORDER BY a").unwrap());
    let a = b.column(0).as_primitive::<Int64Type>();
    let s = b.column(1).as_string::<i32>();
    let v = b.column(2).as_primitive::<Int64Type>();
    (0..b.num_rows())
        .map(|i| {
            (
                a.value(i),
                (!s.is_null(i)).then(|| s.value(i).to_string()),
                (!v.is_null(i)).then(|| v.value(i)),
            )
        })
        .collect()
}

fn merge(e: &mut Engine) {
    seed(e);
    e.execute_sql(
        "CREATE TABLE src (a BIGINT, s VARCHAR, v BIGINT); \
         INSERT INTO src VALUES (2, 'b', 200), (3, 'c', 300), (4, 'd', 400), (7, 'g', 700), \
         (8, 'h', NULL)",
    )
    .unwrap();
    let n = affected(
        e.execute_sql(
            "MERGE INTO t USING src ON t.a = src.a \
             WHEN MATCHED AND src.a = 4 THEN DELETE \
             WHEN MATCHED THEN UPDATE SET v = src.v + t.a, s = src.s \
             WHEN NOT MATCHED AND src.v IS NOT NULL THEN INSERT (a, s, v) VALUES (src.a, src.s, src.v) \
             WHEN NOT MATCHED THEN INSERT VALUES (src.a, 'none', 0)",
        )
        .unwrap(),
    );
    assert_eq!(n, 5);
    let s = |x: &str| Some(x.to_string());
    assert_eq!(
        all_rows(e),
        vec![
            (1, s("x"), Some(10)),
            (2, s("b"), Some(202)),
            (3, s("c"), Some(303)),
            (7, s("g"), Some(700)),
            (8, s("none"), Some(0)),
        ]
    );
    // Target rows without a source row, with an alias on each side.
    e.execute_sql("INSERT INTO t VALUES (9, 'q', 9)").unwrap();
    let n = affected(
        e.execute_sql(
            "MERGE INTO t AS tt USING src AS z ON tt.a = z.a \
             WHEN NOT MATCHED BY SOURCE AND tt.a = 1 THEN UPDATE SET v = -1 \
             WHEN NOT MATCHED BY SOURCE THEN DELETE",
        )
        .unwrap(),
    );
    assert_eq!(n, 2);
    assert_eq!(all_rows(e)[0], (1, s("x"), Some(-1)));
    assert_eq!(all_rows(e).len(), 5);
    // A source subquery; nothing matches a condition: nothing changes.
    let n = affected(
        e.execute_sql(
            "MERGE INTO t USING (SELECT a FROM src WHERE a > 100) AS q ON t.a = q.a \
             WHEN MATCHED THEN DELETE",
        )
        .unwrap(),
    );
    assert_eq!(n, 0);
    // A target row matched twice is an error; the table is unchanged.
    e.execute_sql("INSERT INTO src VALUES (2, 'dup', 1)")
        .unwrap();
    let err = e
        .execute_sql("MERGE INTO t USING src ON t.a = src.a WHEN MATCHED THEN UPDATE SET v = 0")
        .unwrap_err()
        .to_string();
    assert!(err.contains("more than one source row"), "{err}");
    assert_eq!(all_rows(e)[1], (2, s("b"), Some(202)));
    // Inside a transaction, rolled back.
    let n = affected(
        e.execute_sql(
            "BEGIN; MERGE INTO t USING src ON t.a = src.a AND src.s = 'dup' \
             WHEN MATCHED THEN UPDATE SET v = 0",
        )
        .unwrap(),
    );
    assert_eq!(n, 1);
    assert_eq!(all_rows(e)[1].2, Some(0));
    e.execute_sql("ROLLBACK").unwrap();
    assert_eq!(all_rows(e)[1].2, Some(202));
    // Malformed clauses.
    assert!(e
        .execute_sql(
            "MERGE INTO t USING src ON t.a = src.a WHEN MATCHED THEN INSERT VALUES (1, 'x', 1)"
        )
        .is_err());
    assert!(e
        .execute_sql("MERGE INTO t USING src ON t.a = src.a WHEN MATCHED THEN UPDATE SET nope = 1")
        .is_err());
}

#[test]
#[ignore = "gpu:e2e — MERGE over in-memory and catalog tables"]
fn merge_into() {
    let mut e = Engine::new().expect("ctx");
    merge(&mut e);
    drop(e);
    let dir = temp_dir("merge");
    let mut e = Engine::builder().database(&dir).build().expect("ctx");
    let v = {
        merge(&mut e);
        e.catalog().unwrap().version()
    };
    drop(e);
    let e = Engine::builder().database(&dir).build().expect("ctx");
    assert_eq!(e.catalog().unwrap().version(), v);
    assert_eq!(all_rows(&e)[0].2, Some(-1));
}
