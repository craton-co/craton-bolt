// SPDX-License-Identifier: Apache-2.0

//! End-to-end guard for the `IN (SELECT …)` host-set size cap.
//!
//! # Why this test needs its own binary
//!
//! `CRATON_IN_SET_MAX_ROWS` is resolved through a process-wide `OnceLock`
//! latch (the convention every `CRATON_*` size guard in this crate follows),
//! so the cap freezes at whatever the environment said when the *first*
//! `IN`-subquery in the process was resolved.
//!
//! This test therefore cannot live beside the other subquery tests: Rust
//! compiles each `tests/*.rs` into one binary and runs **all** of that
//! binary's tests in a **single process**, so any earlier test that resolves
//! an `IN` subquery latches the compile-time default first and the lowered
//! value this test exports is silently ignored. It previously sat in
//! `tests/subquery_e2e_test.rs` on exactly that mistaken assumption, and
//! failed the moment the live-GPU lane started running the ignored tests.
//!
//! A dedicated binary is its own process, so the latch resolves to the value
//! set below. Do not add further tests to this file.
//!
//! ```text
//! cargo test --test in_subquery_cap_e2e -- --ignored
//! ```

use std::sync::Arc;

use arrow_array::{Int32Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::Engine;

/// Register `t(k)` (the probe) and `other(id)` (the subquery source).
fn engine_with_probe_and_set(probe: Vec<Option<i32>>, set: Vec<Option<i32>>) -> Engine {
    let mut engine = Engine::new().expect("CUDA ctx");

    let t_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "k",
        ArrowDataType::Int32,
        true,
    )]));
    let t =
        RecordBatch::try_new(t_schema, vec![Arc::new(Int32Array::from(probe))]).expect("t batch");
    engine.register_table("t", t).expect("register t");

    let o_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        ArrowDataType::Int32,
        true,
    )]));
    let o =
        RecordBatch::try_new(o_schema, vec![Arc::new(Int32Array::from(set))]).expect("other batch");
    engine.register_table("other", o).expect("register other");
    engine
}

/// A high-cardinality `IN (SELECT …)` must be rejected with a clean
/// `BoltError` once the DISTINCT set exceeds the host cap, instead of building
/// an unbounded membership set / deeply-nested expression tree (which risked an
/// OOM or a stack overflow during the later recursive lower/JIT walks).
///
/// The cap is lowered to a tiny value via `CRATON_IN_SET_MAX_ROWS` so the test
/// stays cheap. The env var is set before the `Engine` is constructed so the
/// process-wide latch resolves to it — see the module docs for why this test
/// owns its binary.
#[test]
#[ignore = "gpu:e2e"]
fn in_subquery_oversized_set_is_capped_cleanly() {
    std::env::set_var("CRATON_IN_SET_MAX_ROWS", "8");

    // `other` has 20 DISTINCT ids → exceeds the cap of 8.
    let probe: Vec<Option<i32>> = (0..5).map(Some).collect();
    let set: Vec<Option<i32>> = (0..20).map(Some).collect();
    let engine = engine_with_probe_and_set(probe, set);

    let err = engine
        .sql("SELECT k FROM t WHERE k IN (SELECT id FROM other)")
        .expect_err("oversized IN subquery must be rejected, not OOM / stack-overflow");
    let msg = format!("{err}");
    assert!(
        msg.contains("distinct values") || msg.contains("more than 8"),
        "cap error should name the distinct-value bound, got: {msg}",
    );

    std::env::remove_var("CRATON_IN_SET_MAX_ROWS");
}
