// SPDX-License-Identifier: Apache-2.0

//! `Database` / `Session`: one shared GPU runtime serving many clients.
//!
//! `cargo test --test session_test -- --ignored --test-threads=1`

use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};
use std::time::Duration;

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::RecordBatch;

use craton_bolt::{Database, Engine};

fn scalar(b: &RecordBatch) -> i64 {
    b.column(0).as_primitive::<Int64Type>().value(0)
}

/// Minimal `block_on`: park the thread until the waker fires.
fn block_on<F: Future>(f: F) -> F::Output {
    struct ThreadWaker(std::thread::Thread);
    impl Wake for ThreadWaker {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(ThreadWaker(std::thread::current())));
    let mut cx = Context::from_waker(&waker);
    let mut f = pin!(f);
    loop {
        if let Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::park_timeout(Duration::from_millis(50));
    }
}

#[test]
#[ignore = "gpu:e2e — concurrent sessions share one runtime"]
fn concurrent_sessions_get_correct_results() {
    let db = Database::open(Engine::builder()).expect("ctx");
    let admin = db.session();
    admin
        .sql("CREATE TABLE t AS SELECT * FROM (VALUES (1, 10), (2, 20), (3, 30)) AS v(k, x)")
        .unwrap();
    let handles: Vec<_> = (0..8)
        .map(|i| {
            let s = db.session();
            std::thread::spawn(move || {
                let mut sum = 0;
                for _ in 0..10 {
                    let b = s.sql(&format!("SELECT SUM(x) + {i} FROM t")).unwrap();
                    sum += scalar(&b) - i;
                }
                (s.id(), s.statement_count(), sum)
            })
        })
        .collect();
    let mut ids = std::collections::BTreeSet::new();
    for h in handles {
        let (id, n, sum) = h.join().unwrap();
        assert!(ids.insert(id), "session ids are unique");
        assert_eq!(n, 10);
        assert_eq!(sum, 600);
    }
    let st = db.stats();
    assert_eq!(st.sessions, 9);
    assert_eq!(st.statements, 81);
    assert_eq!(st.failed, 0);
}

#[test]
#[ignore = "gpu:e2e — read-only sessions, async results, cancellation, panic isolation"]
fn session_controls() {
    let db = Database::open(Engine::builder()).expect("ctx");
    let s = db.session();
    s.sql("CREATE TABLE t AS SELECT * FROM (VALUES (5)) AS v(x)")
        .unwrap();

    let ro = db.session();
    ro.set_read_only(true);
    assert!(ro.settings().read_only);
    assert!(ro.sql("DROP TABLE t").is_err());
    assert_eq!(scalar(&ro.sql("SELECT x FROM t").unwrap()), 5);

    // Awaitable results.
    let fut = s.submit("SELECT x * 3 FROM t");
    assert_eq!(scalar(&block_on(fut).unwrap()), 15);

    // A statement dropped before it starts is cancelled. Hold the runtime
    // busy, queue one, drop it.
    let blocker = {
        let db = db.clone();
        std::thread::spawn(move || {
            db.with_engine(|_| std::thread::sleep(Duration::from_millis(400)))
        })
    };
    std::thread::sleep(Duration::from_millis(100));
    drop(s.submit("SELECT x FROM t"));
    blocker.join().unwrap().unwrap();
    s.sql("SELECT x FROM t").unwrap();
    assert_eq!(db.stats().cancelled, 1);

    // A panic fails only its own task.
    let p = db.with_engine(|_| -> i32 { panic!("boom") });
    assert!(p.unwrap_err().to_string().contains("boom"));
    assert_eq!(scalar(&s.sql("SELECT x FROM t").unwrap()), 5);

    // Engine access beyond SQL.
    let names = db.with_engine(|e| e.table_names()).unwrap();
    assert_eq!(names, vec!["t".to_string()]);

    db.close();
    assert!(s
        .sql("SELECT x FROM t")
        .unwrap_err()
        .to_string()
        .contains("closed"));
}

#[cfg(feature = "storage")]
#[test]
#[ignore = "gpu:e2e — a durable catalog behind a Database"]
fn database_over_a_catalog_persists() {
    let dir = std::env::temp_dir().join(format!("bolt-session-{}", std::process::id()));
    {
        let db = Database::open(Engine::builder().database(&dir)).expect("ctx");
        let s = db.session();
        s.sql("CREATE TABLE events (id BIGINT, kind VARCHAR)")
            .unwrap();
        let writers: Vec<_> = (0..4)
            .map(|w| {
                let s = db.session();
                std::thread::spawn(move || {
                    for i in 0..5 {
                        s.sql(&format!(
                            "INSERT INTO events VALUES ({}, 'w{w}')",
                            w * 100 + i
                        ))
                        .unwrap();
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        assert_eq!(scalar(&s.sql("SELECT COUNT(*) FROM events").unwrap()), 20);
    }
    // Every session's inserts were committed; a second database reopens it.
    let db = Database::open(Engine::builder().database(&dir)).expect("ctx");
    let s = db.session();
    assert_eq!(scalar(&s.sql("SELECT COUNT(*) FROM events").unwrap()), 20);
    assert_eq!(
        scalar(
            &s.sql("SELECT COUNT(*) FROM events WHERE kind = 'w2'")
                .unwrap()
        ),
        5
    );
    drop(s);
    drop(db);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
#[ignore = "gpu:e2e — a failed open releases the engine slot"]
fn failed_open_reports_the_error() {
    let err = Database::open(Engine::builder().device(999)).unwrap_err();
    assert!(err.to_string().contains("out of range"), "{err}");
    // The slot is free again.
    let db = Database::open(Engine::builder()).expect("ctx");
    drop(db);
}
