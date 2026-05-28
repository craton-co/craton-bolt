// SPDX-License-Identifier: Apache-2.0

//! v0.6 / M7 — end-to-end tests for `Engine::builder()` / [`EngineBuilder`].
//!
//! Two flavours of test live here:
//!
//! 1. **Host-only smoke** — exercises the builder's pure type-level
//!    contract: defaults, knob propagation through getters, `Default`
//!    derivation. These run anywhere; no CUDA context is created because
//!    `build()` is NOT called. Driven by inspecting the builder's
//!    method chain via the public surface.
//!
//! 2. **GPU e2e** (`#[ignore = "gpu:e2e"]`) — actually calls `.build()`,
//!    drives a SQL query through the resulting engine, and asserts the
//!    builder-set knobs survive on the constructed `Engine`. Run with
//!    `cargo test --test engine_builder_test -- --ignored` on a CUDA
//!    host.
//!
//! These tests pin the v0.6 public API for `EngineBuilder`. If a knob's
//! name or default changes, this file must be updated in lockstep —
//! intentional friction for what is now a frozen public-API surface.

use std::path::PathBuf;

use craton_bolt::{Engine, EngineBuilder};

// ---- Host-only: the builder API surface itself ------------------------------

/// A fresh builder is `Default` and `Clone` — we rely on both for ergonomic
/// "stash a default then tweak per test" patterns in downstream code.
#[test]
fn builder_is_default_and_clone() {
    let b1: EngineBuilder = EngineBuilder::default();
    let _b2 = b1.clone(); // must compile
    let _b3 = EngineBuilder::new(); // explicit ::new() also available
}

/// `Engine::builder()` returns an [`EngineBuilder`] identical to
/// [`EngineBuilder::new()`]. The two entry points are equivalent — the
/// inherent associated function is just an ergonomic shortcut.
#[test]
fn engine_builder_entry_point_matches_engine_builder_new() {
    // Both expressions must type-check to `EngineBuilder`. Use a generic
    // identity to assert the type without naming it twice.
    fn assert_engine_builder(_b: EngineBuilder) {}
    assert_engine_builder(Engine::builder());
    assert_engine_builder(EngineBuilder::new());
}

/// Builder methods are chainable and consume-self. Compile-checking the
/// chain locks the v0.6 fluent contract.
#[test]
fn builder_methods_chain() {
    let _: EngineBuilder = Engine::builder()
        .device(0)
        .memory_budget(64 * 1024 * 1024)
        .persistent_cache(PathBuf::from("./does-not-exist"))
        .enable_tracing();
}

// ---- GPU e2e: build() actually constructs an engine -------------------------

/// Defaults flow: builder with no knobs set is observationally equivalent to
/// `Engine::new()`. Device ordinal `0`, no memory budget, no persistent
/// cache, tracing off.
#[test]
#[ignore = "gpu:e2e"]
fn build_with_defaults_matches_engine_new() {
    let engine = Engine::builder().build().expect("build default engine");
    assert_eq!(engine.device(), 0, "default device should be 0");
    assert_eq!(
        engine.memory_budget_bytes(),
        None,
        "default memory budget should be uncapped"
    );
    assert!(
        engine.persistent_cache_path().is_none(),
        "default persistent cache should be None"
    );
    assert!(
        !engine.tracing_enabled(),
        "tracing should default to disabled"
    );
}

/// `Engine::new()` is now a thin wrapper around the builder — assert
/// observable behaviour matches the explicit `builder().build()` path.
#[test]
#[ignore = "gpu:e2e"]
fn engine_new_routes_through_builder() {
    let engine = Engine::new().expect("Engine::new() still works");
    assert_eq!(engine.device(), 0);
    assert_eq!(engine.memory_budget_bytes(), None);
    assert!(engine.persistent_cache_path().is_none());
    assert!(!engine.tracing_enabled());
}

/// `Engine::new_with_device(0)` still routes through the builder and lands
/// at the same defaults except for the explicitly-set device knob.
#[test]
#[ignore = "gpu:e2e"]
fn engine_new_with_device_routes_through_builder() {
    let engine = Engine::new_with_device(0).expect("device 0 must exist");
    assert_eq!(engine.device(), 0);
}

/// Out-of-range device on the builder surfaces the same error shape as the
/// legacy `Engine::new_with_device` path (the wrapper does NOT add extra
/// validation — the builder owns the check).
#[test]
#[ignore = "gpu:e2e"]
fn builder_rejects_out_of_range_device() {
    // Negative is unconditionally invalid — no CUDA driver call needed
    // to reject. (Positive-but-too-large would require the driver to
    // know `cuDeviceGetCount`; we test the cheap branch.)
    let err = Engine::builder()
        .device(-1)
        .build()
        .expect_err("negative device index must be rejected");
    let msg = format!("{err}");
    assert!(
        msg.contains("out of range") || msg.contains("-1"),
        "error message should mention the invalid index: {msg}"
    );
}

/// Knob propagation: every builder setter is reflected on the constructed
/// `Engine` through the matching getter.
#[test]
#[ignore = "gpu:e2e"]
fn builder_knobs_propagate_to_engine() {
    let cache_dir = PathBuf::from(".bolt-ptx-cache-test");
    let engine = Engine::builder()
        .device(0)
        .memory_budget(128 * 1024 * 1024)
        .persistent_cache(cache_dir.clone())
        .enable_tracing()
        .build()
        .expect("build with knobs");

    assert_eq!(engine.device(), 0);
    assert_eq!(engine.memory_budget_bytes(), Some(128 * 1024 * 1024));
    assert_eq!(
        engine.persistent_cache_path(),
        Some(cache_dir.as_path()),
        "persistent cache path must round-trip verbatim"
    );
    assert!(engine.tracing_enabled());
}

/// End-to-end: a builder-constructed engine actually runs SQL. This is the
/// "the wrapper doesn't break execution" smoke. Uses a single-batch
/// in-memory table to keep the test trivial; correctness of SQL execution
/// is covered exhaustively in `tests/e2e_tests.rs`.
#[test]
#[ignore = "gpu:e2e"]
fn builder_constructed_engine_can_run_sql() {
    use arrow_array::{Int32Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use std::sync::Arc;

    let schema = Arc::new(Schema::new(vec![Field::new(
        "x",
        DataType::Int32,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])) as _],
    )
    .expect("build record batch");

    let mut engine = Engine::builder()
        .device(0)
        .build()
        .expect("build engine");
    engine.register_table("t", batch).expect("register table");
    let handle = engine
        .sql("SELECT x FROM t")
        .expect("query the builder-constructed engine");
    assert_eq!(handle.record_batch().num_rows(), 5);
}
