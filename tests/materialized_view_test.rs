// SPDX-License-Identifier: Apache-2.0

//! Materialized views: incremental refresh from appended catalog files,
//! scheduled refresh, and answering queries from fresh views.
//!
//! `cargo test --test materialized_view_test -- --ignored --test-threads=1`

#![cfg(feature = "storage")]

use std::path::PathBuf;
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Array, RecordBatch};

use craton_bolt::metrics::{metrics, Counter};
use craton_bolt::{Database, Engine, RefreshMode};

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "bolt-mv-{tag}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn rows(e: &Engine, sql: &str) -> Vec<Vec<i64>> {
    let b: RecordBatch = e.sql(sql).unwrap().into_record_batch();
    (0..b.num_rows())
        .map(|r| {
            (0..b.num_columns())
                .map(|c| {
                    let col =
                        arrow::compute::cast(b.column(c), &arrow_schema::DataType::Int64).unwrap();
                    let a = col.as_primitive::<Int64Type>();
                    if a.is_null(r) {
                        i64::MIN
                    } else {
                        a.value(r)
                    }
                })
                .collect()
        })
        .collect()
}

const VIEW: &str = "SELECT g, SUM(v) AS s, COUNT(*) AS n, MIN(v) AS lo, MAX(v) AS hi \
                    FROM t WHERE v > 0 GROUP BY g";
const DIRECT: &str = "SELECT g, SUM(v) AS s, COUNT(*) AS n, MIN(v) AS lo, MAX(v) AS hi \
                      FROM t WHERE v > 0 GROUP BY g ORDER BY g";

fn insert(e: &mut Engine, lo: i64, hi: i64) {
    let values: Vec<String> = (lo..hi)
        .map(|k| format!("({k}, {}, {})", k % 4, k % 9 - 2))
        .collect();
    e.execute_sql(&format!("INSERT INTO t VALUES {}", values.join(", ")))
        .unwrap();
}

#[test]
#[ignore = "gpu:e2e — incremental refresh from appended catalog files"]
fn incremental_refresh_merges_appended_files() {
    let dir = temp_dir("inc");
    {
        let mut e = Engine::builder().database(&dir).build().expect("ctx");
        e.execute_sql("CREATE TABLE t (k BIGINT, g BIGINT, v BIGINT)")
            .unwrap();
        insert(&mut e, 0, 40);
        e.execute_sql(&format!("CREATE MATERIALIZED VIEW mv AS {VIEW}"))
            .unwrap();
        insert(&mut e, 40, 90);
        let st = e.view_status("mv").unwrap();
        assert!(!st.fresh && st.incremental, "{st:?}");
        e.execute_sql("REFRESH MATERIALIZED VIEW mv").unwrap();
        let st = e.view_status("mv").unwrap();
        assert!(st.fresh && st.last_refresh_incremental, "{st:?}");
        assert_eq!(rows(&e, "SELECT * FROM mv ORDER BY g"), rows(&e, DIRECT));
        // A new group arrives through the delta.
        e.execute_sql("INSERT INTO t VALUES (1000, 7, 5)").unwrap();
        e.execute_sql("REFRESH MATERIALIZED VIEW mv").unwrap();
        assert_eq!(rows(&e, "SELECT * FROM mv ORDER BY g"), rows(&e, DIRECT));
        // A rewrite (not an append) refreshes in full.
        e.execute_sql("DELETE FROM t WHERE k < 10").unwrap();
        let st = e.view_status("mv").unwrap();
        assert!(!st.incremental, "{st:?}");
        e.execute_sql("REFRESH MATERIALIZED VIEW mv").unwrap();
        assert!(!e.view_status("mv").unwrap().last_refresh_incremental);
        assert_eq!(rows(&e, "SELECT * FROM mv ORDER BY g"), rows(&e, DIRECT));
    }
    // The recorded base files survive a restart.
    let mut e = Engine::builder().database(&dir).build().expect("ctx");
    insert(&mut e, 200, 230);
    assert!(e.view_status("mv").unwrap().incremental);
    e.execute_sql("REFRESH MATERIALIZED VIEW mv").unwrap();
    assert!(e.view_status("mv").unwrap().last_refresh_incremental);
    assert_eq!(rows(&e, "SELECT * FROM mv ORDER BY g"), rows(&e, DIRECT));
}

#[test]
#[ignore = "gpu:e2e — views outside the incremental shape refresh in full"]
fn non_incremental_shapes_refresh_in_full() {
    let dir = temp_dir("full");
    let mut e = Engine::builder().database(&dir).build().expect("ctx");
    e.execute_sql("CREATE TABLE t (k BIGINT, g BIGINT, v BIGINT)")
        .unwrap();
    insert(&mut e, 0, 20);
    // AVG does not merge; a computed output does not either.
    e.execute_sql("CREATE MATERIALIZED VIEW a AS SELECT g, AVG(v) AS m FROM t GROUP BY g")
        .unwrap();
    e.execute_sql("CREATE MATERIALIZED VIEW b AS SELECT g, SUM(v) * 2 AS s2 FROM t GROUP BY g")
        .unwrap();
    insert(&mut e, 20, 30);
    for v in ["a", "b"] {
        assert!(!e.view_status(v).unwrap().incremental, "{v}");
        e.refresh_view(v).unwrap();
        assert!(e.view_status(v).unwrap().fresh, "{v}");
    }
    assert_eq!(
        rows(&e, "SELECT * FROM b ORDER BY g"),
        rows(
            &e,
            "SELECT g, SUM(v) * 2 AS s2 FROM t GROUP BY g ORDER BY g"
        )
    );
}

#[test]
#[ignore = "gpu:e2e — scheduled refresh before statements and on an idle Database"]
fn scheduled_views_refresh_when_due() {
    let mut e = Engine::new().expect("ctx");
    e.execute_sql("CREATE TABLE t (k BIGINT, g BIGINT, v BIGINT); INSERT INTO t VALUES (1, 1, 1)")
        .unwrap();
    e.execute_sql(
        "CREATE MATERIALIZED VIEW s WITH (refresh = 'every 1 second') AS \
         SELECT COUNT(*) AS n FROM t",
    )
    .unwrap();
    assert_eq!(
        e.view_status("s").unwrap().refresh,
        RefreshMode::Scheduled(Duration::from_secs(1))
    );
    insert(&mut e, 10, 12);
    // Not yet due.
    assert!(!e.view_status("s").unwrap().fresh);
    std::thread::sleep(Duration::from_millis(1100));
    assert_eq!(e.refresh_due_views().unwrap(), 1);
    assert_eq!(rows(&e, "SELECT n FROM s"), vec![vec![3]]);
    insert(&mut e, 20, 21);
    std::thread::sleep(Duration::from_millis(1100));
    // Any execute_sql statement catches it up.
    e.execute_sql("SELECT n FROM s").unwrap();
    assert!(e.view_status("s").unwrap().fresh);
    assert!(e
        .execute_sql(
            "CREATE MATERIALIZED VIEW x WITH (refresh = 'every so often') AS SELECT 1 FROM t"
        )
        .is_err());
    drop(e);

    // An idle Database refreshes on its own.
    let db = Database::open(Engine::builder()).expect("db");
    let s = db.session();
    s.sql("CREATE TABLE t (k BIGINT, g BIGINT, v BIGINT); INSERT INTO t VALUES (1, 1, 1)")
        .unwrap();
    s.sql(
        "CREATE MATERIALIZED VIEW s WITH (refresh = 'every 1 second') AS \
         SELECT COUNT(*) AS n FROM t",
    )
    .unwrap();
    db.with_engine(|e| {
        e.insert_batch(
            "t",
            e.sql("SELECT k, g, v FROM t").unwrap().into_record_batch(),
        )
    })
    .unwrap()
    .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        let fresh = db
            .with_engine(|e| e.view_status("s").map(|st| st.fresh))
            .unwrap()
            .unwrap();
        if fresh {
            break;
        }
        assert!(std::time::Instant::now() < deadline, "never refreshed");
        std::thread::sleep(Duration::from_millis(200));
    }
    let n = db.with_engine(|e| rows(e, "SELECT n FROM s")).unwrap();
    assert_eq!(n, vec![vec![2]]);
}

#[test]
#[ignore = "gpu:e2e — queries are answered from fresh views with the same plan"]
fn fresh_views_answer_matching_queries() {
    let mut e = Engine::new().expect("ctx");
    e.execute_sql("CREATE TABLE t (k BIGINT, g BIGINT, v BIGINT)")
        .unwrap();
    insert(&mut e, 0, 50);
    let q = "SELECT g, SUM(v) AS s FROM t GROUP BY g";
    e.execute_sql(&format!("CREATE MATERIALIZED VIEW mv AS {q}"))
        .unwrap();
    let plan = e.explain_sql(q).unwrap();
    assert!(
        plan.contains("Answered from materialized view 'mv'"),
        "{plan}"
    );
    let before = metrics().counter(Counter::ViewRewrites).get();
    let mut from_view = rows(&e, q);
    assert_eq!(metrics().counter(Counter::ViewRewrites).get(), before + 1);
    let mut direct = rows(&e, &format!("{q} ORDER BY g"));
    from_view.sort();
    direct.sort();
    assert_eq!(from_view, direct);
    // A stale view is not used.
    insert(&mut e, 50, 51);
    let before = metrics().counter(Counter::ViewRewrites).get();
    rows(&e, q);
    assert_eq!(metrics().counter(Counter::ViewRewrites).get(), before);
    assert!(!e.explain_sql(q).unwrap().contains("Answered from"));
    drop(e);

    let mut e = Engine::builder().view_rewrite(false).build().expect("ctx");
    e.execute_sql("CREATE TABLE t (k BIGINT, g BIGINT, v BIGINT)")
        .unwrap();
    insert(&mut e, 0, 5);
    e.execute_sql(&format!("CREATE MATERIALIZED VIEW mv AS {q}"))
        .unwrap();
    assert!(!e.explain_sql(q).unwrap().contains("Answered from"));
}
