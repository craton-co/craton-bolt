# Bolt Core / Infrastructure Review

Scope: `src/lib.rs`, `src/error.rs`, `src/metrics.rs`, `src/observability.rs`,
`build.rs`, `kernels/Cargo.toml`, `kernels/rust-toolchain.toml`,
`kernels/src/lib.rs`.

Overall this is a mature, well-documented infrastructure layer. The error type
is `#[non_exhaustive]` with a clean `Display`, the metrics registry is lock-free
and allocation-free, the observer surface correctly handles panics/poisoning,
and `build.rs` has thought through CUDA-less fallback. The findings below are
mostly polish, dead surface, and documentation drift — no memory-safety or
injection vulnerabilities were found in the reviewed files. The one item worth
flagging in CI terms is the build-time toolchain download under `rust-cuda`
(already documented in-repo).

---

## 1. CODE REVIEW

### Critical
None.

### High

**H1 — `kernels/` pins a nightly feature (`register_attr`) that no longer exists; the `rust-cuda` path is effectively non-buildable on a current toolchain.**
`kernels/src/lib.rs:24` does `#![cfg_attr(target_arch = "nvptx64", feature(register_attr))]`. `register_attr` was removed from rustc in 2021–2022. `kernels/rust-toolchain.toml:13` pins `nightly-2026-04-02`, which will not have this feature. Combined with `cuda_std = "0.2.2"` (a Feb-2022 cut, `kernels/Cargo.toml:36`) being pinned against a 2026 nightly that links `rustc-dev`/`rustc_codegen_nvvm` internals, the `rust-cuda` feature almost certainly fails to compile. It is off-by-default and explicitly a "Wave A spike", so this is a latent/experimental-path issue rather than a shipping bug — but the toolchain pin and the feature flag are mutually inconsistent and should be reconciled (or the spike marked clearly unbuildable). Severity High because anyone enabling the documented feature hits a hard failure.

### Medium

**M1 — Dead metrics surface: 3 of 8 counters and 5 of 8 phase histograms are never written.**
`BytesUploaded`, `BytesDownloaded`, `GpuLaunchesTotal` (`src/metrics.rs:115-121`) have no producers anywhere in `src/` (verified by grep). Likewise the `Codegen`, `PtxLoad`, `Launch`, `Transfer`, `Materialize` phase histograms (`src/metrics.rs:166-176`) are never `observe`d — only `Phase::Parse` (`src/exec/engine.rs:2085`) and `Phase::Lower` (`engine.rs:2131`) are. A scraper following the documented Prometheus recipe (`metrics.rs:38-65`) will emit permanently-zero series for PCIe bytes, launches, and most phase latencies, which is misleading (looks like "no GPU traffic ever"). Either wire these up at the transfer/launch/codegen sites or document them as reserved/not-yet-populated.

**M2 — `Counter::PtxCacheHits`/`Misses` double-counting risk vs. module cache.**
`PtxCacheHits`/`Misses` are bumped in `src/exec/module_cache.rs:261,266`, and the same crate also exposes a separate `module_cache` miss counter (`engine.rs:642`, `module_cache.rs:304`). Two parallel hit/miss accounting systems for the "PTX/module cache" concept invite drift. Worth a doc note clarifying which is authoritative for which layer (disk PTX cache vs. in-process module cache).

**M3 — `build.rs` Linux path emits multiple `-L` search dirs and never `break`s, unlike Windows.**
`build.rs:115-128` iterates the candidate Linux/macOS lib dirs and pushes a `rustc-link-search` for every one that exists, whereas the Windows branch (`build.rs:100-107`) breaks on first hit. Emitting `/usr/lib/x86_64-linux-gnu` *and* a CUDA `lib64` simultaneously can pick up a system `libcuda` stub ahead of the real toolkit one depending on linker order. Functionally usually fine, but reproducibility/ordering is less deterministic than the Windows branch. Consider first-match-wins for parity.

**M4 — `Metrics` snapshot is `O(buckets)` per phase and copies the whole histogram array.**
`Metrics::snapshot` (`src/metrics.rs:383-396`) does `8 phases × 24 buckets` relaxed loads + the array copy on every call. This is the correct "pull" design and the cost is trivial at human scrape intervals, but `snapshot()` is `pub` and the docs invite calling it from a `/metrics` handler — if a caller wired it per-request on a hot HTTP path it would be a (small) needless cost. Not a bug; flagging for the docs to note it is intended for periodic, not per-request, scraping.

### Low

**L1 — Doc/value drift in metrics overflow-bucket description.**
`src/metrics.rs:95` says the final bucket "therefore means `≥ 2^22 µs`". With `HISTOGRAM_BUCKETS = 24` the last finite bound is `2^23` µs and `bucket_index` saturates at index 23 for inputs `> 2^23` (verified: `bucket_index(1<<23) == 23`). The text should read `≥ 2^23 µs` (the "2^22" appears to be stale from a 23-bucket era). The accompanying tests (`metrics.rs:551-559`) are correct; only the prose is wrong.

**L2 — `BoltError::Cuda(String)` legacy variant is soft-deprecated only in prose.**
`src/error.rs:37-38` documents that new code "SHOULD prefer `CudaWithCode`" and deliberately is *not* `#[deprecated]`. That's a defensible call, but there's no compile-time guard preventing a new `CUresult`-bearing site from regressing onto `Cuda(String)`. A clippy lint or a `#[doc(hidden)]` constructor wrapper would make the intent enforceable. Low — the design note is explicit.

**L3 — `Io` variant via `#[from] std::io::Error` loses path/context.**
`src/error.rs:101-102` blanket-converts any `io::Error`. The disk PTX cache (`DISK_PTX_CACHE_ENV`, re-exported at `lib.rs:208`) does filesystem I/O; a bare "IO error: No such file or directory" with no path is hard to action. Consider a context-carrying variant for the cache layer (out of scope for this file but the error shape lives here).

**L4 — Large `#[doc(hidden)] pub mod __test_only_*` surface in `lib.rs`.**
`src/lib.rs:131-209` exposes six `__test_only_*` modules plus `REL_TOL_TEST`. They are `#[doc(hidden)]` and well-justified (integration tests link the non-test build), but they are genuinely `pub` and reachable by any downstream crate. Each one is a small unstable-API leak. The rationale is documented; the residual risk is that `__test_only_gpu_sort` / `__test_only_sort_kernel` expose internal kernel types (`SortLayout`, `KeyDesc`, `SortKernelSpec`) on the stable build. Acceptable, but worth periodically pruning as the public test helpers consolidate.

**L5 — `install_pool_stats_observer` is single-slot; second install silently drops the first.**
`src/observability.rs:104-111` overwrites the slot. Documented (`lib.rs:83`, `observability.rs:71-73`), but there is no return value indicating a prior observer was replaced, and no multiplexing. For a library that may be embedded alongside other instrumentation this can cause "my metrics stopped" surprises. Low; the design is intentional and documented.

**L6 — `kernels/Cargo.toml` pins `cuda_std = "0.2.2"` with an inline note that it may be "too old".**
`kernels/Cargo.toml:29-37` acknowledges the crates.io pin may need to flip to a git rev. This is honest but means the committed manifest is known-fragile. Tie-in with H1.

### Security review of `build.rs`
- **No command injection / shell-out.** `build.rs` never spawns a shell; it only emits `cargo:` directives and (under `rust-cuda`) calls the `cuda_builder` API in-process (`build.rs:200-204`). Good.
- **`CUDA_PATH` handling** (`build.rs:47-69`) is used verbatim as a `-L` search path. This is the standard `*-sys` contract and the in-file V-16 note correctly frames it as trusted build input; `exists()`-gating prevents emitting bogus paths. No path traversal concern because the value is only handed to the linker, not opened/read.
- **Version parsing** (`build.rs:84-94`) correctly sorts numerically (not lexicographically) and degrades unparseable dirs to `(0,0)` instead of panicking. Good.
- **CPU-only fallback** is graceful: `cuda-stub` early-returns (`build.rs:24-26`); a real build with no CUDA dir emits two actionable `cargo:warning=` lines (`build.rs:137-151`) instead of dying with a cryptic linker error. Solid.
- **Build-time network fetch (V-4)** under `rust-cuda`: `cuda_builder`/`rustc_codegen_nvvm` download and execute an LLVM/libNVVM toolchain with **no in-repo checksum/signature pinning** (documented at `build.rs:166-183`). This is the single most security-relevant item: enabling the feature on an uncontrolled runner trusts whatever upstream pulls. It is off-by-default and the hardening guidance is written down, so the residual risk is acceptable *as documented*, but real integrity verification is still a TODO.
- **Reproducibility:** the empty `partition.ptx` stub write (`build.rs:217-219`) is idempotent (only writes if absent), which is correct.

### Stubs / unimplemented
- `compile_rust_cuda_kernels` off-path writes an empty PTX stub (`build.rs:207-220`) — intentional, documented.
- `register_table_stream` is documented as eager-consume now, lazy "v0.7+" later (see `tests/register_stream_test.rs:6-10`) — a known forward-looking stub, not a defect.

---

## 2. TESTS

Coverage is good for the *contract* surfaces and notably honest about gaps.

**Strengths**
- `metrics.rs` in-module tests (`metrics.rs:487-668`) are thorough: bucket boundary inclusivity, overflow saturation, snapshot fidelity, copy-not-view, singleton stability.
- `observability.rs` tests (`:160-253`) cover the two hardest behaviours — panic-swallow and re-use-after-panic across repeated emits — with a serialising mutex to defeat the multi-threaded harness.
- `error.rs` tests pin `Display` wire-compat and `span()` for every variant.
- `env_var_smoke.rs` pins env-var *constant names* (`:319-325`) so a rename becomes a test failure — excellent drift guard — and exercises clamp/falsy/truthy/default semantics.
- `tracing_test.rs` captures real span names through a custom layer over the offline pipeline.
- `engine_builder_test.rs` cleanly splits host-only (compile/chain) from `gpu:e2e` ignored tests.

**Gaps / enhancements**
1. **No test asserts the dead counters/histograms are populated (M1).** Add a `gpu:e2e` test that runs a real query and asserts `BytesUploaded`/`GpuLaunchesTotal` and the `Launch`/`Transfer` histograms advance — this would have caught that they are never wired.
2. **`metrics_hotpath_test.rs` only covers `QueriesTotal`.** `QueriesFailed`, `HostFallbacksTotal`, `PtxCacheHits/Misses` have no hot-path delta test. Add a failing-query test asserting `QueriesFailed` advances (host-only feasible: a parse error path increments at `engine.rs:2161/2255`).
3. **No concurrency/stress test on `Metrics`.** The whole point is lock-free multi-thread bumps; a test spawning N threads each doing M increments and asserting the exact total would lock the Relaxed-atomic contract.
4. **No test for `install_pool_stats_observer` replacement semantics (L5)** — that a second install supersedes the first and the first stops being called.
5. **`build.rs` has zero test coverage.** `parse_cuda_version` (`build.rs:84`) is a pure, testable function with a documented sort-order bug class (lexicographic vs numeric) — extract it or add a `#[cfg(test)]` block so `v9.0 < v12.0` is pinned.
6. **`env_var_smoke.rs` self-documents (`:14-22`) that 4 spec-requested env vars and several `mem_pool`/`pool-watcher` vars are NOT covered.** When the dispatcher work lands, the template is ready; flagging so it isn't forgotten.
7. **No negative test for `bucket_index` doc claim (L1)** at the exact `2^23` boundary tied to the prose — the existing test uses `HISTOGRAM_BUCKETS-1` symbolically, so the "2^22" prose error slipped through.

---

## 3. NEW FEATURES / DIRECTIONS

- **Wire the unused metrics (M1)** at the transfer (`cuda::GpuBuffer::*`), launch (`exec::launch`), codegen, ptx_load, and materialize sites so the per-phase histograms and PCIe-byte counters reflect reality. This is the highest-value observability win and the surface already exists.
- **Multi-observer / layered pool-stats observers.** Replace the single-slot registry with a `Vec<ObserverHandle>` (or an `install` returning a guard that unregisters on drop) so embedders don't clobber each other.
- **Feature-gate the metrics registry.** A `metrics` cargo feature (default-on) would let extremely-minimal embedders compile it out, paralleling the existing `cuda-stub`/`pool-watcher` granularity.
- **Build-integrity hardening for `rust-cuda` (V-4).** Add checksum pinning / vendored toolchain support before the spike graduates from experimental; reconcile the `register_attr` + nightly + `cuda_std 0.2.2` triple (H1).
- **Richer CPU fallback telemetry.** `HostFallbacksTotal` exists but there's no breakdown of *why* (capacity vs. unsupported op vs. driver error). A small labelled counter set, or reusing the `BoltError::GpuCapacity` typed marker (`error.rs:114`) as the fallback reason, would make fallbacks diagnosable.
- **`no_std`-ish core.** `metrics.rs` is already dependency-light (only `once_cell` + `core` atomics); a `core`-only split of the counter/histogram primitives would let them be reused in the nvptx64 kernel crate or other constrained contexts.
- **Structured error context.** Add path/source context to `Io` and a span to `Plan` (the `error.rs:9-18` note already anticipates "a richer `Plan` shape carrying its own span") to round out the editor/IDE story begun by `SqlWithSpan`.

---

## Severity summary

| ID | Sev | File:line | Issue |
|----|-----|-----------|-------|
| H1 | High | kernels/src/lib.rs:24; rust-toolchain.toml:13; kernels/Cargo.toml:36 | `register_attr` feature + nightly + cuda_std 0.2.2 pin are mutually inconsistent; `rust-cuda` likely won't build |
| M1 | Medium | src/metrics.rs:115-121,166-176 | 3 counters + 5 phase histograms declared but never written |
| M2 | Medium | src/exec/module_cache.rs:261,266 vs engine.rs:642 | Two parallel PTX/module cache hit/miss accounting systems |
| M3 | Medium | build.rs:115-128 | Linux link-search emits all matches (no first-match break) vs Windows |
| M4 | Medium | src/metrics.rs:383-396 | `snapshot()` is pull-only by design; doc should warn against per-request use |
| L1 | Low | src/metrics.rs:95 | Overflow-bucket prose says `2^22`, should be `2^23` |
| L2 | Low | src/error.rs:37 | Legacy `Cuda(String)` soft-deprecated in prose only |
| L3 | Low | src/error.rs:101 | `#[from] io::Error` drops path/context |
| L4 | Low | src/lib.rs:131-209 | Large `pub #[doc(hidden)]` test-only surface leaks internal types |
| L5 | Low | src/observability.rs:104 | Single-slot observer silently replaces prior |
| L6 | Low | kernels/Cargo.toml:29-37 | `cuda_std` pin self-documented as fragile |

No critical issues; no command-injection / path-traversal / memory-safety defects in the reviewed files. `build.rs` security posture is sound and the one real supply-chain concern (V-4 build-time toolchain download) is off-by-default and documented.
