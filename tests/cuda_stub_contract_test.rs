// SPDX-License-Identifier: Apache-2.0

//! Contract test for the `cuda-stub` build configuration.
//!
//! `--features cuda-stub` is the configuration CI actually runs: a CPU-only
//! build with no CUDA toolkit and no GPU. Every FFI entry in
//! `src/cuda/cuda_sys.rs` is replaced by a Rust shim that returns
//! `CUDA_ERROR_STUB`, and `cuda_sys::check()` maps that sentinel to a single
//! typed error. The whole point of the stub is that GPU work must fail with a
//! CLEAN, TYPED error rather than panicking, aborting, or faulting the host —
//! yet this is the only code path CI exercises and it was previously untested.
//!
//! This module is gated on `feature = "cuda-stub"` so it compiles and runs
//! under `--features cuda-stub` and is simply absent in a real-GPU build (where
//! these calls would instead succeed or hit the driver). It is intentionally
//! NOT `#[ignore]`d: it must run in CI.
//!
//! # Which variant does the stub return?
//!
//! As of the current tree, `cuda_sys::check(CUDA_ERROR_STUB)` returns
//! `BoltError::Other("cuda-stub mode: no GPU support compiled in")`
//! (see `src/cuda/cuda_sys.rs`, the `check()` function). A dedicated
//! `BoltError::Unsupported(String)` variant exists in `src/error.rs` and is the
//! intended long-term home for this "no GPU / not compiled in" case, but the
//! stub has not migrated to it yet. These tests therefore assert against the
//! variant the stub *actually returns now* (`Other`), while the panic-freedom
//! and `Display` checks are written to survive the eventual migration to
//! `Unsupported` (they accept either typed "no-GPU" variant). When the stub is
//! migrated, flip `assert_is_stub_no_gpu_error` to require `Unsupported` and the
//! contract stays green.
#![cfg(feature = "cuda-stub")]

use craton_bolt::cuda::cuda_sys;
use craton_bolt::{BoltError, Engine};

/// Assert that `err` is the typed "no GPU in this build" error the stub is
/// contracted to produce, and that its rendered message names the stub /
/// no-GPU condition.
///
/// Accepts either `BoltError::Other` (what the stub returns today) or
/// `BoltError::Unsupported` (the intended target variant) so this contract
/// keeps holding across the planned migration. It deliberately does NOT accept
/// `BoltError::CudaWithCode` / `BoltError::Cuda`: the stub short-circuits on the
/// `CUDA_ERROR_STUB` sentinel precisely so callers get a recognisable typed
/// "unsupported build" signal, not a generic driver-error code.
#[track_caller]
fn assert_is_stub_no_gpu_error(err: &BoltError) {
    assert!(
        matches!(err, BoltError::Other(_) | BoltError::Unsupported(_)),
        "stub GPU work must surface a typed no-GPU error \
         (Other(_) today, Unsupported(_) after migration), got: {err:?}"
    );

    // The Display string must be non-empty and identify the stub / no-GPU
    // condition — downstream logs and users rely on this being actionable.
    let rendered = err.to_string();
    assert!(
        !rendered.trim().is_empty(),
        "stub error Display must be non-empty, got empty string"
    );
    let lower = rendered.to_lowercase();
    assert!(
        lower.contains("stub") || lower.contains("no gpu") || lower.contains("gpu support"),
        "stub error Display should mention the stub / no-GPU condition, got: {rendered:?}"
    );
}

/// Lowest-level boundary: `cuda_sys::init()` calls the `cuInit` shim, which
/// returns `CUDA_ERROR_STUB`; `check()` converts that into the typed no-GPU
/// error. This is the first thing every GPU code path touches, so locking its
/// contract down here covers the broadest surface.
#[test]
fn cuda_init_returns_typed_no_gpu_error() {
    let err = cuda_sys::init().expect_err("cuda_sys::init() must fail under cuda-stub");
    assert_is_stub_no_gpu_error(&err);
}

/// `device_count()` is the public entry the engine builder uses right after
/// `init()`. Under the stub the `cuDeviceGetCount` shim returns
/// `CUDA_ERROR_STUB`, so this must also be a clean typed error, not a panic.
#[test]
fn cuda_device_count_returns_typed_no_gpu_error() {
    let err =
        cuda_sys::device_count().expect_err("device_count() must fail under cuda-stub");
    assert_is_stub_no_gpu_error(&err);
}

/// Direct CUDA-context initialization — the boundary the task calls out
/// explicitly. `CudaContext::new(0)` runs `init()` → `device_get()` →
/// `cuCtxCreate_v2`, every one of which is stubbed. It must return `Err`, never
/// panic.
#[test]
fn cuda_context_new_returns_typed_no_gpu_error() {
    let err = cuda_sys::CudaContext::new(0)
        .err()
        .expect("CudaContext::new(0) must fail under cuda-stub");
    assert_is_stub_no_gpu_error(&err);
}

/// Highest-level public boundary: building an `Engine`. `Engine::new()` ->
/// `EngineBuilder::build()` calls `cuda_sys::init()?` up front, so engine
/// construction itself fails under the stub (the failure is at construction,
/// not deferred to query execution). Assert the typed error and, critically,
/// that we did not panic getting here.
#[test]
fn engine_new_returns_typed_no_gpu_error() {
    let err = Engine::new().err().expect("Engine::new() must fail under cuda-stub");
    assert_is_stub_no_gpu_error(&err);
}

/// Dedicated `Display` contract: the rendered error string must be non-empty
/// and mention the stub / no-GPU condition. Requirement (3): an explicit,
/// standalone assertion on the human-readable message, independent of which
/// boundary produced it.
#[test]
fn stub_error_display_is_non_empty_and_mentions_no_gpu() {
    let err = cuda_sys::init().expect_err("cuda_sys::init() must fail under cuda-stub");

    let rendered = err.to_string();
    assert!(
        !rendered.is_empty(),
        "stub error Display must be non-empty"
    );

    // Exact current wording. If the stub is reworded, update this single
    // assertion; the looser `assert_is_stub_no_gpu_error` substring check above
    // is the migration-tolerant guard, this one pins today's exact contract.
    assert_eq!(
        rendered, "cuda-stub mode: no GPU support compiled in",
        "stub error Display wording drifted from the documented contract"
    );
}

/// The whole motivation for the stub is "fail cleanly, do not panic". This test
/// drives the lowest-level GPU entry point inside `catch_unwind` and asserts the
/// call returned an `Err` value rather than unwinding. (Belt-and-suspenders:
/// the `expect_err` calls above already prove the others return `Err`; this
/// makes the no-panic guarantee an explicit, named contract.)
#[test]
fn stub_gpu_entry_does_not_panic() {
    let outcome = std::panic::catch_unwind(|| {
        // Returns `BoltResult<()>`; we only care that calling it does not
        // unwind. Map to a bool so the closure is `RefUnwindSafe`.
        cuda_sys::init().is_err()
    });

    match outcome {
        Ok(returned_err) => assert!(
            returned_err,
            "cuda_sys::init() under cuda-stub returned Ok — it must return a typed Err"
        ),
        Err(_) => panic!(
            "cuda_sys::init() panicked under cuda-stub; the stub contract is to \
             return a clean typed error, never to panic"
        ),
    }
}
