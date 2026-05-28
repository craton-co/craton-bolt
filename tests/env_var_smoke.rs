// SPDX-License-Identifier: Apache-2.0

//! Smoke tests for opt-in env-var-gated code paths.
//!
//! Each test SETs the env var, exercises the parse / dispatch helper, then
//! RESTOREs the env var. Tests serialise on a process-wide mutex so they
//! don't race against each other (Rust's test harness runs tests in
//! parallel by default and `std::env::set_var` mutates global state).
//!
//! ## Env vars covered (against the live codebase)
//!
//! The task spec asked for tests of `BOLT_PREFIX_SCAN_ALGO=blelloch`,
//! `BOLT_HASH_ALGO=robin_hood`, `BOLT_HASH_PROBE_TILED=1`, and
//! `BOLT_SORT_USE_GRAPH=1`. None of those env vars or their matching
//! `compile_*_blelloch` / `compile_*_dispatched` / `compile_*_tiled`
//! entry points exist in this branch — the algorithm-selection work
//! (Robin Hood, lookback scan, tiled probe, sort-via-CUDA-Graph) has
//! shipped its kernels but not yet a runtime dispatcher gated on an env
//! var. When the dispatcher work lands the test file should grow a
//! matching block per env var; the [`ENV_LOCK`] + [`restore`] pattern
//! below is the template.
//!
//! What this file does cover is every env var that **does** read through
//! `std::env::var(` in `src/` today (grep target: `BOLT_*` and
//! `CRATON_BOLT_*` constants):
//!
//! - `BOLT_GPU_JOIN_TABLE_CAP_MB` — hash-table byte cap parser, clamped
//!   to `[64, 4096]` MiB.
//! - `BOLT_GPU_JOIN_STREAMING_INTERN` — streaming-intern toggle, truthy
//!   semantics (`"1"`/`"true"` enable; `""`/`"0"`/`"false"`/unset disable).
//! - `CRATON_BOLT_PTX_CACHE_CAP` — PTX module-cache capacity parser.
//! - `BOLT_POOL_STATS_INTERVAL_SECS` — engine periodic pool-stats
//!   interval (`0` → disabled, `n` → `Duration::from_secs(n)`).
//!
//! ## Env vars not exercised here (parser is `pub(crate)`-internal and
//! not reachable through a `parse_*` shim)
//!
//! - `CRATON_BOLT_POOL_MAX_BYTES`, `CRATON_BOLT_POOL_BUCKET_CAP` —
//!   consumed by `mem_pool::read_env_usize`, which is currently
//!   `fn` (private). No `pub fn parse_*` shim exists; the existing
//!   `#[cfg(test)] mod tests` inside `mem_pool.rs` already round-trips
//!   them via `with_env` and a fresh `Pool`, so a separate smoke here
//!   would duplicate coverage without a clearer signal.
//! - `BOLT_POOL_WATCH_INTERVAL_SECS`, `BOLT_POOL_WATCH_LOW_WATER_FRAC`
//!   — read inline inside `cuda::mem_pool::pool_watcher_thread`; no
//!   parser shim. Documented for future expansion.

use std::sync::Mutex;
use std::time::Duration;

use craton_bolt::__test_only_env_vars::{
    parse_env_cap, parse_ptx_cache_cap, pool_stats_interval_from_env, streaming_intern_enabled,
    CAP_ENV_VAR, POOL_STATS_ENV, PTX_CACHE_CAP_ENV, STREAMING_INTERN_ENV_VAR,
};

/// Process-wide mutex serialising `std::env::set_var` calls across tests.
///
/// Cargo runs integration tests in parallel by default, and `set_var`
/// mutates global process state — without this lock two tests racing
/// against the same env var would see each other's writes and
/// intermittently fail. The body of every test in this file MUST take
/// `ENV_LOCK.lock()` before mutating the environment.
static ENV_LOCK: Mutex<()> = Mutex::new(());

/// Restore an env var to its captured prior value (or remove it if
/// the prior value was `None`).
///
/// Factored out so every test reads identically: capture `prev`, set,
/// assert, then `restore(name, prev)`. Centralising the "remove vs.
/// set" branch keeps individual tests focused on the assertion.
fn restore(name: &str, prev: Option<String>) {
    match prev {
        Some(v) => std::env::set_var(name, v),
        None => std::env::remove_var(name),
    }
}

// ---------------------------------------------------------------------------
// BOLT_GPU_JOIN_TABLE_CAP_MB
// ---------------------------------------------------------------------------

/// Setting `BOLT_GPU_JOIN_TABLE_CAP_MB` to a valid integer routes through
/// `parse_env_cap` and clamps within `[64, 4096]` MiB. Mirrors the in-
/// module unit test in `gpu_join.rs::tests::parse_env_cap_clamps_to_range`
/// but pins the *constant name* (`CAP_ENV_VAR`) from a downstream
/// vantage point: if the env-var name ever drifts, the link error here
/// surfaces immediately at integration-test build time.
#[test]
fn bolt_gpu_join_table_cap_mb_parses_valid_integer() {
    let _guard = ENV_LOCK.lock().unwrap();
    let prev = std::env::var(CAP_ENV_VAR).ok();

    // In-range value: parsed as MiB, returned as bytes.
    std::env::set_var(CAP_ENV_VAR, "128");
    let cap = parse_env_cap().expect("128 MiB parses");
    assert_eq!(cap, 128 * 1024 * 1024, "in-range cap must round-trip MiB");

    // Below the 64 MiB floor → clamped up.
    std::env::set_var(CAP_ENV_VAR, "1");
    let cap = parse_env_cap().expect("1 MiB parses");
    assert_eq!(cap, 64 * 1024 * 1024, "sub-floor input must clamp to 64 MiB");

    // Above the 4 GiB ceiling → clamped down.
    std::env::set_var(CAP_ENV_VAR, "99999");
    let cap = parse_env_cap().expect("99999 MiB parses");
    assert_eq!(cap, 4096 * 1024 * 1024, "super-ceiling input must clamp to 4 GiB");

    restore(CAP_ENV_VAR, prev);
}

/// Unset / empty / garbage values must return `None` — the dispatcher
/// then falls back to the driver-detected cap. This guards against a
/// regression where a parse failure routed to a misleading default
/// (e.g. zero) instead of "no override".
#[test]
fn bolt_gpu_join_table_cap_mb_rejects_invalid_input() {
    let _guard = ENV_LOCK.lock().unwrap();
    let prev = std::env::var(CAP_ENV_VAR).ok();

    std::env::remove_var(CAP_ENV_VAR);
    assert!(parse_env_cap().is_none(), "unset env: None");

    std::env::set_var(CAP_ENV_VAR, "");
    assert!(parse_env_cap().is_none(), "empty env: None");

    std::env::set_var(CAP_ENV_VAR, "   ");
    assert!(parse_env_cap().is_none(), "whitespace-only env: None");

    std::env::set_var(CAP_ENV_VAR, "not-a-number");
    assert!(parse_env_cap().is_none(), "non-numeric env: None");

    restore(CAP_ENV_VAR, prev);
}

// ---------------------------------------------------------------------------
// BOLT_GPU_JOIN_STREAMING_INTERN
// ---------------------------------------------------------------------------

/// The streaming-intern toggle MUST default off (unset / empty / "0" /
/// "false" all map to false). Pins the falsy-input contract that
/// downstream tooling (benchmark harnesses, env-passthrough wrappers)
/// quietly relies on.
#[test]
fn bolt_gpu_join_streaming_intern_defaults_off() {
    let _guard = ENV_LOCK.lock().unwrap();
    let prev = std::env::var(STREAMING_INTERN_ENV_VAR).ok();

    std::env::remove_var(STREAMING_INTERN_ENV_VAR);
    assert!(!streaming_intern_enabled(), "unset env: streaming OFF");

    std::env::set_var(STREAMING_INTERN_ENV_VAR, "");
    assert!(!streaming_intern_enabled(), "empty env: streaming OFF");

    std::env::set_var(STREAMING_INTERN_ENV_VAR, "0");
    assert!(!streaming_intern_enabled(), "'0' env: streaming OFF");

    std::env::set_var(STREAMING_INTERN_ENV_VAR, "false");
    assert!(!streaming_intern_enabled(), "'false' env: streaming OFF");

    // Case-insensitive: `False`, `FALSE`, mixed-case all map to false.
    std::env::set_var(STREAMING_INTERN_ENV_VAR, "False");
    assert!(!streaming_intern_enabled(), "'False' env: streaming OFF");

    restore(STREAMING_INTERN_ENV_VAR, prev);
}

/// Truthy values (`"1"`, `"true"`, any non-empty non-falsy string)
/// flip the toggle on. Companion to `_defaults_off`.
#[test]
fn bolt_gpu_join_streaming_intern_flips_on_truthy() {
    let _guard = ENV_LOCK.lock().unwrap();
    let prev = std::env::var(STREAMING_INTERN_ENV_VAR).ok();

    std::env::set_var(STREAMING_INTERN_ENV_VAR, "1");
    assert!(streaming_intern_enabled(), "'1' env: streaming ON");

    std::env::set_var(STREAMING_INTERN_ENV_VAR, "true");
    assert!(streaming_intern_enabled(), "'true' env: streaming ON");

    std::env::set_var(STREAMING_INTERN_ENV_VAR, "yes");
    assert!(streaming_intern_enabled(), "'yes' env: streaming ON (non-falsy)");

    restore(STREAMING_INTERN_ENV_VAR, prev);
}

// ---------------------------------------------------------------------------
// CRATON_BOLT_PTX_CACHE_CAP
// ---------------------------------------------------------------------------

/// `CRATON_BOLT_PTX_CACHE_CAP` must round-trip through `parse_cap`
/// with the documented "zero / non-numeric → default" semantics.
///
/// Notes:
/// - The `ptx_cache_cap()` resolver in `jit_compiler.rs` latches via
///   `OnceLock` on first call, so a smoke test against the resolver
///   would only be meaningful in the first test to run in a fresh
///   process. We drive `parse_cap` directly instead, mirroring the
///   in-module `parse_cap_picks_up_env_var` unit test.
/// - We still set+restore the live env var so a regression where the
///   constant `PTX_CACHE_CAP_ENV` is renamed surfaces here (the read
///   happens via `std::env::var(PTX_CACHE_CAP_ENV)` below).
#[test]
fn craton_bolt_ptx_cache_cap_parser_round_trip() {
    let _guard = ENV_LOCK.lock().unwrap();
    let prev = std::env::var(PTX_CACHE_CAP_ENV).ok();

    // Valid positive integer → parsed.
    std::env::set_var(PTX_CACHE_CAP_ENV, "32");
    let raw = std::env::var(PTX_CACHE_CAP_ENV).ok();
    assert_eq!(
        parse_ptx_cache_cap(raw.as_deref(), 256),
        32,
        "valid positive integer must override default",
    );

    // Zero → default (zero-sized cache is pathological).
    std::env::set_var(PTX_CACHE_CAP_ENV, "0");
    let raw = std::env::var(PTX_CACHE_CAP_ENV).ok();
    assert_eq!(
        parse_ptx_cache_cap(raw.as_deref(), 256),
        256,
        "'0' must fall back to default",
    );

    // Non-numeric → default.
    std::env::set_var(PTX_CACHE_CAP_ENV, "not-a-number");
    let raw = std::env::var(PTX_CACHE_CAP_ENV).ok();
    assert_eq!(
        parse_ptx_cache_cap(raw.as_deref(), 256),
        256,
        "non-numeric must fall back to default",
    );

    // Negative (parse-fails as `usize`) → default.
    std::env::set_var(PTX_CACHE_CAP_ENV, "-1");
    let raw = std::env::var(PTX_CACHE_CAP_ENV).ok();
    assert_eq!(
        parse_ptx_cache_cap(raw.as_deref(), 256),
        256,
        "negative must fall back to default",
    );

    // Unset → default.
    std::env::remove_var(PTX_CACHE_CAP_ENV);
    let raw = std::env::var(PTX_CACHE_CAP_ENV).ok();
    assert_eq!(
        parse_ptx_cache_cap(raw.as_deref(), 256),
        256,
        "unset must fall back to default",
    );

    restore(PTX_CACHE_CAP_ENV, prev);
}

// ---------------------------------------------------------------------------
// BOLT_POOL_STATS_INTERVAL_SECS
// ---------------------------------------------------------------------------

/// `BOLT_POOL_STATS_INTERVAL_SECS=0` disables the periodic emit
/// (signalled by `Duration::ZERO`), a positive integer maps to
/// `Duration::from_secs(n)`, and anything unparseable returns the
/// 60-second default. Pins the "operator can silence pool stats with
/// =0" contract that benchmark scripts rely on.
#[test]
fn bolt_pool_stats_interval_secs_parser() {
    let _guard = ENV_LOCK.lock().unwrap();
    let prev = std::env::var(POOL_STATS_ENV).ok();

    // Zero → disabled.
    std::env::set_var(POOL_STATS_ENV, "0");
    assert_eq!(
        pool_stats_interval_from_env(),
        Duration::ZERO,
        "'0' must disable periodic emission",
    );

    // Positive integer → that many seconds.
    std::env::set_var(POOL_STATS_ENV, "7");
    assert_eq!(
        pool_stats_interval_from_env(),
        Duration::from_secs(7),
        "'7' must map to Duration::from_secs(7)",
    );

    std::env::set_var(POOL_STATS_ENV, "3600");
    assert_eq!(
        pool_stats_interval_from_env(),
        Duration::from_secs(3600),
        "'3600' must map to one-hour interval",
    );

    // Garbage → 60s default.
    std::env::set_var(POOL_STATS_ENV, "not-a-number");
    assert_eq!(
        pool_stats_interval_from_env(),
        Duration::from_secs(60),
        "non-numeric must fall back to default 60s",
    );

    // Unset → 60s default.
    std::env::remove_var(POOL_STATS_ENV);
    assert_eq!(
        pool_stats_interval_from_env(),
        Duration::from_secs(60),
        "unset must fall back to default 60s",
    );

    restore(POOL_STATS_ENV, prev);
}

// ---------------------------------------------------------------------------
// Sanity: env-var constants are byte-identical to their documented spellings
// ---------------------------------------------------------------------------

/// Pin the canonical env-var names. Drift between the code constants and
/// the operator-facing names (in docs, benchmark scripts, deployment
/// configs) silently disables every override; an explicit assertion
/// here turns a rename into a test failure.
#[test]
fn env_var_constant_names_are_stable() {
    assert_eq!(CAP_ENV_VAR, "BOLT_GPU_JOIN_TABLE_CAP_MB");
    assert_eq!(STREAMING_INTERN_ENV_VAR, "BOLT_GPU_JOIN_STREAMING_INTERN");
    assert_eq!(PTX_CACHE_CAP_ENV, "CRATON_BOLT_PTX_CACHE_CAP");
    assert_eq!(POOL_STATS_ENV, "BOLT_POOL_STATS_INTERVAL_SECS");
}
