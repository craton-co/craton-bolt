// SPDX-License-Identifier: Apache-2.0

//! Integration tests for SQL window functions (`func(...) OVER (...)`).
//!
//! The window executor (`src/exec/window.rs`) is host-side, but the only
//! PUBLIC driver that reaches it is [`craton_bolt::Engine::sql`], and `Engine`
//! owns a live `CudaContext` (opened in `Engine::new`). So every end-to-end
//! value check here needs a GPU and is `#[ignore = "gpu:e2e"]`'d — the project's
//! standard bucket for "generic e2e SQL needing GPU" (see `tests/common/mod.rs`).
//! They still COMPILE and link under `--no-default-features --features
//! cuda-stub`; run them on a GPU host with:
//!
//! ```text
//! cargo test --test window_functions_test -- --ignored
//! ```
//!
//! What we ALSO pin, runnable on CI without a GPU, is the *frontend contract*
//! for frames: the SQL parser (`craton_bolt::plan::parse_sql`) is a pure host
//! path, so the "unsupported frame is rejected" tests run everywhere (no
//! `#[ignore]`): `GROUPS`, `RANGE` with a value offset, and a start bound
//! after the end bound (`sql_frontend::lower_window_frame`).
//!
//! ## Truth table (standard SQL / DuckDB semantics asserted)
//!
//! - `ROW_NUMBER() OVER (ORDER BY x)` → 1..N, ties broken deterministically.
//! - `RANK()` skips after a tie group; `DENSE_RANK()` does not — the
//!   peer-group distinction.
//! - `SUM(v) OVER (ORDER BY k)` is a running total under the default RANGE
//!   frame: equal-key peers all see the SAME running value (through the end of
//!   their peer group).
//! - `PARTITION BY` resets ranks and running totals per partition; a singleton
//!   partition gets ROW_NUMBER/RANK = 1 and a running SUM equal to its own row.
//! - NULL ordering follows the engine's documented default
//!   (`nulls_first = !descending`): ASC puts NULLs first, DESC puts NULLs last
//!   (`lower_window_order_by` in the SQL frontend).

use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::{parse_sql, DataType, Field, MemTableProvider, Schema};
use craton_bolt::{BoltError, Engine};

mod common;

// ---------------------------------------------------------------------------
// Column-decoding helpers (each integration binary is its own crate).
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn col_i32(batch: &RecordBatch, c: usize) -> &Int32Array {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("column is Int32")
}

#[allow(dead_code)]
fn col_i64(batch: &RecordBatch, c: usize) -> &Int64Array {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("column is Int64")
}

#[allow(dead_code)]
fn col_f64(batch: &RecordBatch, c: usize) -> &Float64Array {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("column is Float64")
}

#[allow(dead_code)]
fn col_str(batch: &RecordBatch, c: usize) -> &StringArray {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("column is Utf8")
}

/// Read an `Int64` window-output column into a `Vec<i64>` (asserts non-null).
#[allow(dead_code)]
fn i64_vec(batch: &RecordBatch, c: usize) -> Vec<i64> {
    let a = col_i64(batch, c);
    (0..a.len())
        .map(|i| {
            assert!(!a.is_null(i), "unexpected NULL at col {c} row {i}");
            a.value(i)
        })
        .collect()
}

/// Read a `Float64` window-output column into a `Vec<f64>` (asserts non-null).
#[allow(dead_code)]
fn f64_vec(batch: &RecordBatch, c: usize) -> Vec<f64> {
    let a = col_f64(batch, c);
    (0..a.len())
        .map(|i| {
            assert!(!a.is_null(i), "unexpected NULL at col {c} row {i}");
            a.value(i)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Fixture builders (e2e, GPU-gated).
// ---------------------------------------------------------------------------

/// Register `t(k: Utf8, v: Int32)` and return the engine. Both columns
/// nullable so ORDER BY / PARTITION BY NULL behaviour is exercisable.
#[allow(dead_code)]
fn engine_kv(name: &str, k: Vec<Option<&str>>, v: Vec<Option<i32>>) -> Engine {
    let mut engine = Engine::new().expect("CUDA ctx");
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Utf8, true),
        ArrowField::new("v", ArrowDataType::Int32, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(k)),
            Arc::new(Int32Array::from(v)),
        ],
    )
    .expect("batch");
    engine.register_table(name, batch).expect("register");
    engine
}

/// Register a single nullable `Int32` column table `t(v)`.
#[allow(dead_code)]
fn engine_v(name: &str, v: Vec<Option<i32>>) -> Engine {
    let mut engine = Engine::new().expect("CUDA ctx");
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "v",
        ArrowDataType::Int32,
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(v))]).expect("batch");
    engine.register_table(name, batch).expect("register");
    engine
}

// ===========================================================================
// ROW_NUMBER — 1..N, ties broken deterministically.
// ===========================================================================

/// `ROW_NUMBER() OVER (ORDER BY v)` is a dense 1..N sequence. We project the
/// row number alongside `v` and ORDER the final result by the row number so the
/// output order is deterministic, then assert the sequence is exactly 1..N and
/// that `v` is non-decreasing along it (the window ORDER BY honoured).
#[test]
#[ignore = "gpu:e2e"]
fn row_number_orders_one_to_n() {
    // v with a tie at 20: {10, 20, 20, 40}. ROW_NUMBER must still be 1..4.
    let engine = engine_v("t", vec![Some(40), Some(20), Some(10), Some(20)]);
    let h = engine
        .sql(
            "SELECT v, ROW_NUMBER() OVER (ORDER BY v) AS rn \
             FROM t ORDER BY rn",
        )
        .expect("ROW_NUMBER over ORDER BY");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 4);

    let rn = i64_vec(out, 1);
    assert_eq!(rn, vec![1, 2, 3, 4], "ROW_NUMBER must be a dense 1..N");

    // Along the row-number order, v is non-decreasing (window ORDER BY honoured).
    let vs = col_i32(out, 0);
    for i in 1..vs.len() {
        assert!(
            vs.value(i - 1) <= vs.value(i),
            "v must be non-decreasing along ROW_NUMBER order"
        );
    }
    // The tied pair occupies the two middle row numbers (2,3) with v=20.
    assert_eq!(vs.value(0), 10);
    assert_eq!(vs.value(1), 20);
    assert_eq!(vs.value(2), 20);
    assert_eq!(vs.value(3), 40);
}

// ===========================================================================
// RANK vs DENSE_RANK — peer-group / tie behaviour.
// ===========================================================================

/// The defining difference: over ORDER BY keys `{10, 10, 20, 30}`,
/// `RANK()` = `{1, 1, 3, 4}` (ties share the lowest rank, then a GAP) while
/// `DENSE_RANK()` = `{1, 1, 2, 3}` (no gap). We assert the value per distinct
/// `v` so the check is independent of intra-tie row ordering.
#[test]
#[ignore = "gpu:e2e"]
fn rank_skips_dense_rank_does_not() {
    let engine = engine_v("t", vec![Some(30), Some(10), Some(20), Some(10)]);
    let h = engine
        .sql(
            "SELECT v, \
                    RANK() OVER (ORDER BY v) AS rk, \
                    DENSE_RANK() OVER (ORDER BY v) AS dr \
             FROM t ORDER BY v",
        )
        .expect("RANK / DENSE_RANK");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 4);

    let vs = col_i32(out, 0);
    let rk = i64_vec(out, 1);
    let dr = i64_vec(out, 2);
    for i in 0..rk.len() {
        let (erk, edr) = match vs.value(i) {
            10 => (1, 1), // tie group of two -> rank 1, dense 1
            20 => (3, 2), // RANK skips to 3 (two rows preceded); DENSE = 2
            30 => (4, 3), // RANK 4; DENSE 3
            other => panic!("unexpected v={other}"),
        };
        assert_eq!(rk[i], erk, "RANK at v={}", vs.value(i));
        assert_eq!(dr[i], edr, "DENSE_RANK at v={}", vs.value(i));
    }
}

// ===========================================================================
// Running SUM — default RANGE frame, peers share the running value.
// ===========================================================================

/// `SUM(v) OVER (ORDER BY v)` is a running total. Under the default RANGE
/// frame, the two `v=10` peers BOTH see the sum through the end of their peer
/// group (10+10 = 20), and the `v=20` row sees 40. This is the key RANGE
/// peer-group semantic the executor implements (`compute_running_aggregate`).
#[test]
#[ignore = "gpu:e2e"]
fn running_sum_range_peers_share_value() {
    let engine = engine_v("t", vec![Some(20), Some(10), Some(10)]);
    let h = engine
        .sql(
            "SELECT v, SUM(v) OVER (ORDER BY v) AS rs \
             FROM t ORDER BY v",
        )
        .expect("running SUM");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 3);

    let vs = col_i32(out, 0);
    let rs = i64_vec(out, 1);
    for (i, got) in rs.iter().enumerate() {
        let expected = match vs.value(i) {
            10 => 20, // both peers see the peer-group-inclusive running sum
            20 => 40,
            other => panic!("unexpected v={other}"),
        };
        assert_eq!(*got, expected, "running SUM at v={}", vs.value(i));
    }
}

/// Strictly-increasing ORDER BY keys give a textbook prefix-sum (no peer
/// sharing): `SUM(v) OVER (ORDER BY v)` over `{1,2,3,4}` is `{1,3,6,10}`.
#[test]
#[ignore = "gpu:e2e"]
fn running_sum_strictly_increasing_is_prefix_sum() {
    let engine = engine_v("t", vec![Some(3), Some(1), Some(4), Some(2)]);
    let h = engine
        .sql(
            "SELECT v, SUM(v) OVER (ORDER BY v) AS rs \
             FROM t ORDER BY v",
        )
        .expect("prefix SUM");
    let out = h.record_batch();
    let rs = i64_vec(out, 1);
    assert_eq!(rs, vec![1, 3, 6, 10], "prefix sum of 1,2,3,4");
}

// ===========================================================================
// PARTITION BY — ranks / running totals reset per partition; singleton.
// ===========================================================================

/// `ROW_NUMBER` and a running `SUM` both RESET at each partition boundary.
/// Partitions: `a = {10, 20, 30}`, `b = {100}` (a SINGLETON partition). For
/// partition `a`, row numbers are 1,2,3 and the running sum is 10,30,60. The
/// singleton `b` gets row_number 1 and running sum 100 (its own value only).
#[test]
#[ignore = "gpu:e2e"]
fn partition_by_resets_rank_and_running_sum() {
    let engine = engine_kv(
        "t",
        vec![Some("a"), Some("a"), Some("b"), Some("a")],
        vec![Some(20), Some(10), Some(100), Some(30)],
    );
    let h = engine
        .sql(
            "SELECT k, v, \
                    ROW_NUMBER() OVER (PARTITION BY k ORDER BY v) AS rn, \
                    SUM(v) OVER (PARTITION BY k ORDER BY v) AS rs \
             FROM t ORDER BY k, v",
        )
        .expect("PARTITION BY ROW_NUMBER + running SUM");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 4);

    let k = col_str(out, 0);
    let v = col_i32(out, 1);
    let rn = i64_vec(out, 2);
    let rs = i64_vec(out, 3);

    // ORDER BY k, v makes the row order deterministic:
    //   (a,10) (a,20) (a,30) (b,100)
    let expected = [
        ("a", 10, 1, 10),
        ("a", 20, 2, 30),
        ("a", 30, 3, 60),
        ("b", 100, 1, 100), // singleton partition: rn=1, running sum = own value
    ];
    for (i, &(ek, ev, ern, ers)) in expected.iter().enumerate() {
        assert_eq!(k.value(i), ek, "row {i} k");
        assert_eq!(v.value(i), ev, "row {i} v");
        assert_eq!(rn[i], ern, "row {i} ROW_NUMBER (resets per partition)");
        assert_eq!(rs[i], ers, "row {i} running SUM (resets per partition)");
    }
}

/// `AVG(v) OVER (PARTITION BY k)` with NO window ORDER BY: the whole partition
/// is one peer group, so every row sees the FULL-partition average. Partition
/// `a = {1,3}` → 2.0 on every a-row; singleton `b = {10}` → 10.0.
#[test]
#[ignore = "gpu:e2e"]
fn avg_over_partition_no_order_is_full_partition() {
    let engine = engine_kv(
        "t",
        vec![Some("a"), Some("a"), Some("b")],
        vec![Some(1), Some(3), Some(10)],
    );
    let h = engine
        .sql(
            "SELECT k, AVG(v) OVER (PARTITION BY k) AS av \
             FROM t ORDER BY k",
        )
        .expect("AVG over partition");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 3);

    let k = col_str(out, 0);
    let av = f64_vec(out, 1);
    for (i, got) in av.iter().enumerate() {
        let expected = match k.value(i) {
            "a" => 2.0, // (1+3)/2
            "b" => 10.0,
            other => panic!("unexpected k={other}"),
        };
        assert_eq!(*got, expected, "AVG for partition {}", k.value(i));
    }
}

// ===========================================================================
// NULL ordering in window ORDER BY (engine default: nulls_first = !descending).
// ===========================================================================

/// ASC window ORDER BY places NULLs FIRST (engine default `nulls_first =
/// !descending`, see `lower_window_order_by`). So `ROW_NUMBER() OVER (ORDER BY
/// v)` over `{NULL, 10, 20}` assigns row number 1 to the NULL row. NULL keys
/// form their own peer group, so a running `SUM(v)` over them is the SUM of
/// non-NULL inputs in that group — here a lone NULL contributes nothing, making
/// `SUM(v)` over the NULL peer group SQL NULL.
#[test]
#[ignore = "gpu:e2e"]
fn null_orders_first_under_asc() {
    let engine = engine_v("t", vec![Some(20), None, Some(10)]);
    let h = engine
        .sql(
            "SELECT v, \
                    ROW_NUMBER() OVER (ORDER BY v) AS rn, \
                    SUM(v) OVER (ORDER BY v) AS rs \
             FROM t ORDER BY rn",
        )
        .expect("NULL-first ASC ordering");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 3);

    let v = col_i32(out, 0);
    let rn = i64_vec(out, 1);
    let rs = col_i64(out, 2);

    // rn=1 is the NULL row (NULLs first under ASC).
    assert_eq!(rn, vec![1, 2, 3]);
    assert!(v.is_null(0), "ASC NULLs-first: row 1 must be the NULL v");
    assert_eq!(v.value(1), 10);
    assert_eq!(v.value(2), 20);

    // The NULL peer group has no non-NULL input, so its running SUM is SQL NULL.
    assert!(rs.is_null(0), "SUM over the lone-NULL peer group is NULL");
    // After the NULL group, the running SUM accumulates the real values.
    assert_eq!(rs.value(1), 10, "running SUM through v=10");
    assert_eq!(rs.value(2), 30, "running SUM through v=20 (10+20)");
}

/// DESC window ORDER BY places NULLs LAST (default `nulls_first = !descending`
/// → `false` for DESC). Over `{NULL, 10, 20}` ordered DESC, the row numbers are
/// 20→1, 10→2, NULL→3.
#[test]
#[ignore = "gpu:e2e"]
fn null_orders_last_under_desc() {
    let engine = engine_v("t", vec![Some(20), None, Some(10)]);
    let h = engine
        .sql(
            "SELECT v, ROW_NUMBER() OVER (ORDER BY v DESC) AS rn \
             FROM t ORDER BY rn",
        )
        .expect("NULL-last DESC ordering");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 3);

    let v = col_i32(out, 0);
    let rn = i64_vec(out, 1);
    assert_eq!(rn, vec![1, 2, 3]);
    assert_eq!(v.value(0), 20, "DESC: largest first");
    assert_eq!(v.value(1), 10);
    assert!(v.is_null(2), "DESC NULLs-last: the NULL v is row 3");
}

// ===========================================================================
// Frontend contract: unsupported frames are rejected (HOST-reachable, no GPU).
// ===========================================================================
//
// `ROWS` frames with literal offsets and `RANGE` frames bounded by UNBOUNDED /
// CURRENT ROW are supported; the rest is rejected in
// `sql_frontend::lower_window_frame`, reached purely through `parse_sql` (no
// CUDA context), so these run on CI without a GPU.

/// A single-column provider for the host-only parse checks.
fn frame_provider() -> MemTableProvider {
    let schema = Schema::new(vec![Field::new("v", DataType::Int32, false)]);
    MemTableProvider::new().with_table("t", schema)
}

/// A `RANGE` frame with a value offset (`RANGE BETWEEN 1 PRECEDING AND
/// CURRENT ROW`) and an inverted frame are rejected with a clear SQL error;
/// `ROWS` frames with offsets parse.
#[test]
fn unsupported_frames_are_rejected() {
    let provider = frame_provider();
    for sql in [
        "SELECT SUM(v) OVER (ORDER BY v RANGE BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
        "SELECT SUM(v) OVER (ORDER BY v ROWS BETWEEN 1 FOLLOWING AND 1 PRECEDING) FROM t",
        "SELECT SUM(v) OVER (ORDER BY v ROWS BETWEEN UNBOUNDED FOLLOWING AND CURRENT ROW) FROM t",
        "SELECT SUM(v) OVER (ORDER BY v ROWS BETWEEN v PRECEDING AND CURRENT ROW) FROM t",
    ] {
        match parse_sql(sql, &provider) {
            Ok(_) => panic!("{sql}: must be rejected"),
            Err(BoltError::Sql(msg)) => assert!(msg.contains("frame"), "{sql}: {msg}"),
            Err(other) => panic!("{sql}: expected BoltError::Sql, got {other:?}"),
        }
    }
    for sql in [
        "SELECT SUM(v) OVER (ORDER BY v ROWS BETWEEN 1 PRECEDING AND CURRENT ROW) FROM t",
        "SELECT AVG(v) OVER (ORDER BY v ROWS BETWEEN 2 PRECEDING AND 3 FOLLOWING) FROM t",
        "SELECT MAX(v) OVER (ORDER BY v RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) FROM t",
        "SELECT COUNT(v) OVER (ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) FROM t",
    ] {
        parse_sql(sql, &provider).unwrap_or_else(|e| panic!("{sql}: {e}"));
    }
}

/// A `GROUPS` frame is explicitly rejected by `reject_non_default_frame`.
#[test]
fn groups_frame_is_rejected() {
    let provider = frame_provider();
    let err = match parse_sql(
        "SELECT SUM(v) OVER (ORDER BY v GROUPS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) \
         FROM t",
        &provider,
    ) {
        Ok(_) => panic!("GROUPS frame must be rejected, got Ok"),
        Err(e) => e,
    };
    match err {
        BoltError::Sql(msg) => assert!(
            msg.contains("GROUPS") || msg.contains("frame"),
            "rejection must name the unsupported GROUPS frame; got: {msg}"
        ),
        other => panic!("expected BoltError::Sql for GROUPS frame, got {other:?}"),
    }
}

/// Positive control: the IMPLICIT default frame (no frame clause at all) parses
/// cleanly. Guards against an over-broad "reject all frames" regression that
/// would break ordinary running aggregates.
#[test]
fn default_frame_parses_cleanly() {
    let provider = frame_provider();
    parse_sql("SELECT SUM(v) OVER (ORDER BY v) FROM t", &provider)
        .expect("default-frame window must parse");
    parse_sql("SELECT ROW_NUMBER() OVER (ORDER BY v) FROM t", &provider)
        .expect("ROW_NUMBER default-frame must parse");
}

// ===========================================================================
// Explicit frames (GPU-built engine; the window itself runs on the host).
// ===========================================================================

/// `(g, k, v)` rows ordered by `k` within `g`; `v` has a NULL and ties in `k`.
fn framed_engine() -> Engine {
    let mut e = Engine::new().expect("ctx");
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("g", ArrowDataType::Int32, false),
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Int32, true),
    ]));
    let g = vec![1, 1, 1, 1, 1, 2, 2, 2];
    let k = vec![1, 2, 2, 3, 4, 1, 2, 3];
    let v = vec![
        Some(10),
        Some(20),
        None,
        Some(40),
        Some(50),
        Some(1),
        Some(2),
        Some(3),
    ];
    let b = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(g)),
            Arc::new(Int32Array::from(k)),
            Arc::new(Int32Array::from(v)),
        ],
    )
    .unwrap();
    e.register_table("t", b).unwrap();
    e
}

/// Values of column `c` as `Option<f64>`, rows ordered by `(g, k, v)`.
fn ordered(e: &Engine, select: &str) -> Vec<Option<f64>> {
    let b = e
        .sql(&format!("{select} FROM t ORDER BY g, k, v NULLS LAST"))
        .unwrap()
        .into_record_batch();
    let c = arrow::compute::cast(b.column(b.num_columns() - 1), &ArrowDataType::Float64).unwrap();
    let a = c.as_any().downcast_ref::<Float64Array>().unwrap();
    (0..a.len())
        .map(|i| (!a.is_null(i)).then(|| a.value(i)))
        .collect()
}

#[test]
#[ignore = "gpu:e2e"]
fn rows_frames_slide_within_partitions() {
    let e = framed_engine();
    // Partition 1 values in order: 10, 20, NULL, 40, 50 (the NULL sorts
    // after 20 inside the k = 2 tie); partition 2: 1, 2, 3.
    let w =
        "OVER (PARTITION BY g ORDER BY k, v NULLS LAST ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING)";
    let s = ordered(&e, &format!("SELECT g, k, v, SUM(v) {w} AS s"));
    assert_eq!(
        s,
        vec![
            Some(30.0),
            Some(30.0),
            Some(60.0),
            Some(90.0),
            Some(90.0),
            Some(3.0),
            Some(6.0),
            Some(5.0)
        ]
    );
    let c = ordered(&e, &format!("SELECT g, k, v, COUNT(v) {w} AS c"));
    assert_eq!(
        c,
        [2, 2, 2, 2, 2, 2, 3, 2]
            .iter()
            .map(|&x| Some(x as f64))
            .collect::<Vec<_>>()
    );
    // Trailing frame; the first row's frame is just itself.
    let a = ordered(
        &e,
        "SELECT g, k, v, AVG(v) OVER (PARTITION BY g ORDER BY k, v NULLS LAST \
         ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS a",
    );
    assert_eq!(
        a,
        vec![
            Some(10.0),
            Some(15.0),
            Some(15.0),
            Some(30.0),
            Some(45.0),
            Some(1.0),
            Some(1.5),
            Some(2.0)
        ]
    );
    // A frame entirely before the partition is empty: SUM NULL, COUNT 0.
    let p = ordered(
        &e,
        "SELECT g, k, v, SUM(v) OVER (PARTITION BY g ORDER BY k, v NULLS LAST \
         ROWS BETWEEN 3 PRECEDING AND 2 PRECEDING) AS p",
    );
    assert_eq!(
        p,
        vec![
            None,
            None,
            Some(10.0),
            Some(30.0),
            Some(20.0),
            None,
            None,
            Some(1.0)
        ]
    );
}

#[test]
#[ignore = "gpu:e2e"]
fn unbounded_and_range_frames() {
    let e = framed_engine();
    let whole = ordered(
        &e,
        "SELECT g, k, v, MAX(v) OVER (PARTITION BY g ROWS BETWEEN UNBOUNDED PRECEDING \
         AND UNBOUNDED FOLLOWING) AS m",
    );
    assert_eq!(
        whole,
        [50, 50, 50, 50, 50, 3, 3, 3]
            .iter()
            .map(|&x| Some(x as f64))
            .collect::<Vec<_>>()
    );
    // Reverse running minimum.
    let tail = ordered(
        &e,
        "SELECT g, k, v, MIN(v) OVER (PARTITION BY g ORDER BY k, v NULLS LAST \
         ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS m",
    );
    assert_eq!(
        tail,
        vec![
            Some(10.0),
            Some(20.0),
            Some(40.0),
            Some(40.0),
            Some(50.0),
            Some(1.0),
            Some(2.0),
            Some(3.0)
        ]
    );
    // RANGE CURRENT ROW .. UNBOUNDED FOLLOWING: the k = 2 peers share a frame.
    let r = ordered(
        &e,
        "SELECT g, k, v, SUM(v) OVER (PARTITION BY g ORDER BY k \
         RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS r",
    );
    assert_eq!(
        r,
        vec![
            Some(120.0),
            Some(110.0),
            Some(110.0),
            Some(90.0),
            Some(50.0),
            Some(6.0),
            Some(5.0),
            Some(3.0)
        ]
    );
    // ROWS UNBOUNDED PRECEDING .. CURRENT ROW excludes later peers, unlike
    // the default RANGE frame.
    let rows = ordered(
        &e,
        "SELECT g, k, v, COUNT(*) OVER (PARTITION BY g ORDER BY k \
         ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS c",
    );
    let range = ordered(
        &e,
        "SELECT g, k, v, COUNT(*) OVER (PARTITION BY g ORDER BY k) AS c",
    );
    assert_eq!(rows[1].unwrap() + rows[2].unwrap(), 2.0 + 3.0);
    assert_eq!(range[1], Some(3.0));
    assert_eq!(range[2], Some(3.0));
}
