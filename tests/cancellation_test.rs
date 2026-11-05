// SPDX-License-Identifier: Apache-2.0

//! Query cancellation and statement timeouts (`CancellationToken`): a
//! cancelled or timed-out statement stops promptly with
//! `BoltError::Cancelled`, releases its device memory, leaves data
//! unchanged, and the engine runs the next statement normally.
//!
//! `cargo test --test cancellation_test -- --ignored --test-threads=1`

use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{DataFrame, DataType, Field, Schema};
use craton_bolt::{
    BoltError, BoltResult, CancelReason, CancellationToken, Database, Engine, MemoryKind,
    TierPolicy,
};

/// Long multi-partition out-of-core GROUP BY (device budget 1 MiB).
const OOC_ROWS: i64 = 3_000_000;
const OOC_QUERY: &str = "SELECT g, SUM(v) AS sv, COUNT(*) AS n FROM fact GROUP BY g ORDER BY g";

/// Long nested-loop (non-equi) join: `BIG × SMALL` cells in 1M-cell blocks.
const BIG_ROWS: i64 = 25_000;
const SMALL_ROWS: i64 = 1_000;
const JOIN_QUERY: &str = "SELECT COUNT(*) AS n FROM big JOIN small ON big.v < small.w";

const POLICIES: [TierPolicy; 2] = [TierPolicy::GpuFirst, TierPolicy::CostBased];

/// `(k, g = k % 50_000, v = k % 1000)`.
fn fact(n: i64) -> RecordBatch {
    let k: Vec<i64> = (0..n).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int64, false),
        ArrowField::new("g", ArrowDataType::Int64, false),
        ArrowField::new("v", ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(k.clone())),
            Arc::new(Int64Array::from(
                k.iter().map(|x| x % 50_000).collect::<Vec<_>>(),
            )),
            Arc::new(Int64Array::from(
                k.iter().map(|x| x % 1000).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap()
}

/// One Int64 column `name` holding `f(0..n)`.
fn column(name: &str, n: i64, f: impl Fn(i64) -> i64) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        name,
        ArrowDataType::Int64,
        false,
    )]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(
            (0..n).map(f).collect::<Vec<_>>(),
        ))],
    )
    .unwrap()
}

/// `big(k, v = k % 1000)`, `small(w = i)`.
fn register_join_tables(e: &mut Engine) {
    let k: Vec<i64> = (0..BIG_ROWS).collect();
    let big = RecordBatch::try_new(
        Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int64, false),
            ArrowField::new("v", ArrowDataType::Int64, false),
        ])),
        vec![
            Arc::new(Int64Array::from(k.clone())),
            Arc::new(Int64Array::from(
                k.iter().map(|x| x % 1000).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    e.register_table("big", big).unwrap();
    e.register_table("small", column("w", SMALL_ROWS, |i| i))
        .unwrap();
}

fn join_engine(policy: TierPolicy) -> Engine {
    let mut e = Engine::builder().tier_policy(policy).build().expect("ctx");
    register_join_tables(&mut e);
    e
}

fn ooc_engine(policy: TierPolicy) -> Engine {
    let mut e = Engine::builder()
        .memory_budget(1 << 20)
        .tier_policy(policy)
        .build()
        .expect("ctx");
    e.register_table("fact", fact(OOC_ROWS)).unwrap();
    e
}

fn i64s(b: &RecordBatch, c: usize) -> Vec<i64> {
    b.column(c).as_primitive::<Int64Type>().values().to_vec()
}

fn assert_cancelled<T: std::fmt::Debug>(r: BoltResult<T>, want: CancelReason) {
    match r {
        Err(BoltError::Cancelled(got)) => assert_eq!(got, want),
        other => panic!("expected Cancelled({want:?}), got {other:?}"),
    }
}

/// Device bytes in use with the allocation pool drained, so only live
/// allocations (resident tables, kernels' modules, leaks) count.
fn device_used() -> usize {
    craton_bolt::cuda::mem_pool::__test_pool().drain();
    let (free, total) = craton_bolt::cuda::cuda_sys::mem_get_info().expect("mem info");
    total - free
}

/// Run `f` with a token a timer thread cancels after `delay`; returns the
/// result and how long `f` took.
fn cancel_after<T>(
    delay: Duration,
    f: impl FnOnce(&CancellationToken) -> BoltResult<T>,
) -> (BoltResult<T>, Duration) {
    let token = CancellationToken::new();
    let timer = {
        let t = token.clone();
        std::thread::spawn(move || {
            std::thread::sleep(delay);
            t.cancel();
        })
    };
    let start = Instant::now();
    let r = f(&token);
    let took = start.elapsed();
    timer.join().unwrap();
    (r, took)
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, Duration) {
    let start = Instant::now();
    let r = f();
    (r, start.elapsed())
}

/// Shared shape of the "cancel a running query" tests: an uncancelled run
/// gives the reference result and time `full`; a run cancelled at `full / 4`
/// stops well before `full`; the engine's memory is back to where it was and
/// the same query then returns the reference result.
fn cancel_mid_query(e: &Engine, sql: &str) {
    let (reference, full) = timed(|| e.sql(sql).expect("uncancelled run").into_record_batch());
    assert!(
        full >= Duration::from_millis(400),
        "the query must run long enough to cancel mid-way (took {full:?}); grow the inputs"
    );
    let baseline = device_used();

    let delay = full / 4;
    let (r, took) = cancel_after(delay, |t| e.sql_with_cancel(sql, t));
    assert_cancelled(r, CancelReason::Cancelled);
    eprintln!("{sql}: uncancelled {full:?}, cancelled at {delay:?} -> stopped after {took:?}");
    assert!(
        took < delay + full / 2,
        "cancelled at {delay:?}, stopped after {took:?} (uncancelled run: {full:?})"
    );

    // Nothing leaked: device memory, governor reservations.
    let after = device_used();
    assert!(
        after <= baseline + (16 << 20),
        "device memory in use grew from {baseline} to {after} bytes after a cancelled query"
    );
    assert_eq!(e.memory_governor().reserved(MemoryKind::Device), 0);
    assert_eq!(e.memory_governor().reserved(MemoryKind::Host), 0);

    // The engine is fully usable: same result as before.
    let again = e.sql(sql).expect("rerun after cancel").into_record_batch();
    assert_eq!(again, reference);
}

#[test]
#[ignore = "gpu:e2e — a cancelled token stops every entry point before it runs"]
fn pre_cancelled_token_returns_cancelled_without_running() {
    for policy in POLICIES {
        let mut e = ooc_engine(policy);
        e.register_table("dst", column("k", 3, |i| i)).unwrap();
        let token = CancellationToken::new();
        token.cancel();
        let before = e.memory_stats().out_of_core_statements;

        let (r, took) = timed(|| e.sql_with_cancel(OOC_QUERY, &token));
        assert_cancelled(r, CancelReason::Cancelled);
        assert!(took < Duration::from_millis(500), "{took:?}");
        assert_eq!(
            e.memory_stats().out_of_core_statements,
            before,
            "nothing ran"
        );

        assert_cancelled(
            e.execute_sql_with_cancel("INSERT INTO dst VALUES (7)", &token),
            CancelReason::Cancelled,
        );
        assert_cancelled(
            e.execute_sql_with_cancel("DELETE FROM dst", &token),
            CancelReason::Cancelled,
        );
        assert_cancelled(
            e.sql_stream_with_cancel("SELECT k FROM dst", &token)
                .map(|_| ()),
            CancelReason::Cancelled,
        );
        let df = DataFrame::scan(
            "dst",
            Schema::new(vec![Field::new("k", DataType::Int64, false)]),
        );
        assert_cancelled(
            df.collect_with_cancel(&mut e, &token),
            CancelReason::Cancelled,
        );

        // Data unchanged, and an uncancelled token / plain call runs.
        let rows = e
            .execute_sql_with_cancel("SELECT COUNT(*) FROM dst", &CancellationToken::new())
            .unwrap()
            .into_record_batch();
        assert_eq!(i64s(&rows, 0), vec![3], "{policy:?}");
    }
}

#[test]
#[ignore = "gpu:e2e — cancel a multi-partition out-of-core GROUP BY from another thread"]
fn cancel_from_another_thread_stops_out_of_core_group_by() {
    for policy in POLICIES {
        let e = ooc_engine(policy);
        cancel_mid_query(&e, OOC_QUERY);
        let out = e.sql(OOC_QUERY).unwrap().into_record_batch();
        assert_eq!(out.num_rows(), 50_000);
        assert_eq!(i64s(&out, 2).iter().sum::<i64>(), OOC_ROWS);
        assert!(e.memory_stats().out_of_core_statements > 0);
    }
}

#[test]
#[ignore = "gpu:e2e — cancel a long nested-loop join from another thread"]
fn cancel_from_another_thread_stops_nested_loop_join() {
    for policy in POLICIES {
        let e = join_engine(policy);
        cancel_mid_query(&e, JOIN_QUERY);
    }
}

#[test]
#[ignore = "gpu:e2e — a deadline token times the statement out"]
fn deadline_times_out_and_engine_recovers() {
    for policy in POLICIES {
        let e = join_engine(policy);
        let (reference, full) = timed(|| e.sql(JOIN_QUERY).unwrap().into_record_batch());
        let timeout = full / 4;
        let token = CancellationToken::new().with_timeout(timeout);
        let (r, took) = timed(|| e.sql_with_cancel(JOIN_QUERY, &token));
        assert_cancelled(r, CancelReason::DeadlineExceeded);
        assert!(
            took < timeout + full / 2,
            "timeout {timeout:?}, took {took:?}"
        );
        // A generous deadline does not get in the way.
        let roomy = CancellationToken::new().with_timeout(Duration::from_secs(3600));
        let out = e
            .sql_with_cancel(JOIN_QUERY, &roomy)
            .unwrap()
            .into_record_batch();
        assert_eq!(out, reference);
    }
}

#[test]
#[ignore = "gpu:e2e — cancelled DML leaves the table unchanged"]
fn cancelled_dml_leaves_data_unchanged() {
    const LONG: &str = "SELECT big.k AS k FROM big JOIN small ON big.v < small.w";
    for policy in POLICIES {
        let mut e = join_engine(policy);
        e.register_table("dst", column("k", 3, |i| i)).unwrap();
        let (_, full) = timed(|| e.sql(JOIN_QUERY).unwrap());
        let delay = full / 4;
        let count = |e: &mut Engine, sql: &str| -> i64 {
            i64s(&e.execute_sql(sql).unwrap().into_record_batch(), 0)[0]
        };

        // INSERT ... SELECT, CREATE TABLE AS, DELETE / UPDATE with a long
        // subquery predicate: each cancelled mid-way.
        let statements = [
            format!("INSERT INTO dst {LONG}"),
            format!("CREATE TABLE made AS {LONG}"),
            "DELETE FROM dst WHERE k < (SELECT COUNT(*) FROM big JOIN small ON big.v < small.w)"
                .to_string(),
            "UPDATE dst SET k = k + (SELECT COUNT(*) FROM big JOIN small ON big.v < small.w)"
                .to_string(),
        ];
        for stmt in &statements {
            let (r, _) = cancel_after(delay, |t| e.execute_sql_with_cancel(stmt, t));
            assert_cancelled(r, CancelReason::Cancelled);
            assert_eq!(count(&mut e, "SELECT COUNT(*) FROM dst"), 3, "{stmt}");
            assert_eq!(count(&mut e, "SELECT SUM(k) FROM dst"), 3, "{stmt}");
        }
        assert!(
            e.execute_sql("SELECT * FROM made").is_err(),
            "CTAS left no table"
        );

        // Inside a transaction: the cancelled statement changes nothing and
        // the transaction stays usable; even a cancelled token can ROLLBACK.
        e.execute_sql("BEGIN").unwrap();
        e.execute_sql("INSERT INTO dst VALUES (100)").unwrap();
        let stmt = format!("INSERT INTO dst {LONG}");
        let (r, _) = cancel_after(delay, |t| e.execute_sql_with_cancel(&stmt, t));
        assert_cancelled(r, CancelReason::Cancelled);
        assert!(e.in_transaction());
        assert_eq!(count(&mut e, "SELECT COUNT(*) FROM dst"), 4);
        e.execute_sql("COMMIT").unwrap();
        assert_eq!(count(&mut e, "SELECT SUM(k) FROM dst"), 103);

        let cancelled = CancellationToken::new();
        cancelled.cancel();
        e.execute_sql("BEGIN").unwrap();
        e.execute_sql("INSERT INTO dst VALUES (1000)").unwrap();
        e.execute_sql_with_cancel("ROLLBACK", &cancelled).unwrap();
        assert!(!e.in_transaction());
        assert_eq!(count(&mut e, "SELECT SUM(k) FROM dst"), 103);

        // And the long statement still works afterwards.
        let n = count(&mut e, JOIN_QUERY);
        e.execute_sql(&format!("INSERT INTO dst {LONG}")).unwrap();
        assert_eq!(count(&mut e, "SELECT COUNT(*) FROM dst"), 4 + n);
    }
}

#[test]
#[ignore = "gpu:e2e — an incremental result stream polls its token per chunk group"]
fn cancelled_stream_stops_between_chunks() {
    let mut e = Engine::builder()
        .host_memory_budget(1 << 20)
        .build()
        .expect("ctx");
    e.register_table("fact", fact(400_000)).unwrap();
    let token = CancellationToken::new();
    let mut stream = e
        .sql_stream_with_cancel("SELECT k, v FROM fact WHERE v < 900", &token)
        .unwrap();
    assert!(stream.is_incremental());
    let first = stream.next().expect("a first chunk").unwrap();
    assert!(first.num_rows() > 0);
    token.cancel();
    assert_cancelled(
        stream.next().expect("an error item"),
        CancelReason::Cancelled,
    );
    assert!(stream.next().is_none(), "the stream ends after the error");
    drop(stream);
    // A fresh stream reads everything.
    let rows: usize = e
        .sql_stream("SELECT k, v FROM fact WHERE v < 900")
        .unwrap()
        .map(|b| b.unwrap().num_rows())
        .sum();
    assert_eq!(rows, 360_000);
}

#[test]
#[ignore = "gpu:e2e — sessions forward a token to the runtime thread"]
fn session_statement_is_cancellable() {
    let db = Database::open(Engine::builder()).expect("open");
    let session = db.session();
    db.with_engine(register_join_tables).unwrap();
    let (reference, full) = timed(|| session.sql(JOIN_QUERY).unwrap());
    let (r, took) = cancel_after(full / 4, |t| session.sql_with_cancel(JOIN_QUERY, t));
    assert_cancelled(r, CancelReason::Cancelled);
    assert!(took < full * 3 / 4, "{took:?} vs {full:?}");
    assert_eq!(session.sql(JOIN_QUERY).unwrap(), reference);
}
