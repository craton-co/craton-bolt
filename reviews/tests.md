# Test-Suite Review — `craton-bolt`

**Reviewer:** senior test-engineering audit
**Date:** 2026-05-30
**Scope:** `tests/` (41 integration files + `tests/common/`), `benches/` (4 files), `examples/` (2 files), cross-referenced against `src/` (~143 K LOC) and `.github/workflows/ci.yml`.

---

## TL;DR — the one number that matters

**GPU correctness in CI is entirely unverified.** CI runs on github-hosted runners with **no GPU**, building with `--no-default-features --features cuda-stub` (`.github/workflows/ci.yml:71`). Every test that drives a real kernel is `#[ignore]`-gated, so:

- **369 / 2337** test functions across `src/` + `tests/` carry `#[ignore]` (~16%).
- The ignored set is **exactly the GPU correctness/e2e tests** — joins, sort, group-by, string ops, mempool, decimal, and the entire DuckDB differential harness.
- What runs in CI is overwhelmingly **host-side**: SQL parse/plan/lower, PTX *shape* (substring) checks, and CPU "model mirror" unit tests.

So while a coarse "tests pass / N tests" metric looks healthy, **verified GPU-execution coverage in automation is ~0%**. The 85% target is not plausibly met for the engine's actual job (executing SQL on a GPU). Host-side correctness (frontend/planner/optimizer) is reasonably well covered; device-side correctness is asserted only by tests no machine in CI ever runs.

---

## 1. Adequacy & Coverage

### 1.1 Raw inventory

| Area | Count |
|---|---|
| Integration test files (`tests/*.rs`) | 41 (+ `tests/common/mod.rs`) |
| Integration test LOC | 20 630 |
| `src/` LOC | 143 416 |
| `src/` inline unit tests (`#[test]`/`#[tokio::test]`) | 1 738 across 130 `#[cfg(test)]` modules |
| Total test fns (src + tests) | **2 337** |
| Total `#[ignore]` (src + tests) | **369** |
| `#[ignore]` in `tests/` alone | 216 / 599 attribute sites (~36%) |
| `#[ignore]` in `src/` alone | 153 |
| `proptest!` blocks | 2 (in `tests/sql_proptest.rs`) |

`#[ignore]` reason buckets in `tests/` (`grep -ho '#[ignore = "…"'`):

```
 43 gpu:tier1     33 gpu:join      32 gpu:e2e      27 gpu:string
 19 gpu:sort      10 gpu:mempool    9 bootstrap      3 gpu:tier2
  1 gpu:proptest-semantic   1 gpu:projection   1 gpu:like
```

Every non-`bootstrap` bucket is a GPU gate. `bootstrap` (9 sites, all in `ptx_golden_tests.rs`) gates `insta` snapshot creation.

### 1.2 What actually runs in CI (CPU / `cuda-stub`)

`ci.yml` `build` job steps that execute tests:
- `cargo test --lib --tests --features cuda-stub --no-default-features` (`ci.yml:71`)
- `cargo test --doc …` (ubuntu/stable only, `ci.yml:75`)

This runs **only the non-`#[ignore]`d** tests. Those are genuinely useful but host-bound:

- **Frontend / planner / optimizer** — `parser_tests.rs` (1158 LOC), `sql_proptest.rs` properties 1–3, `wave3_regression_test.rs` (constant-fold overflow, decimal widening at the public API), and ~1 700 inline `src/` unit tests for `plan`, `exec` host mirrors, dictionary logic, etc. **This layer is well covered.**
- **PTX codegen shape** — `ptx_golden_tests.rs` (1758 LOC) runs ~202 `ptx.contains("…")` substring assertions on host-side `compile_ptx` / `compile_*_kernel` output. These run in CI and catch dropped-mnemonic regressions, but they assert *text presence*, not execution correctness (see §2.2).
- **CPU "model mirror" tests** — `shmem_groupby_e2e.rs`, `tier2_groupby_e2e.rs`, `tier2_multi_sum_e2e.rs` each ship CPU-only unit tests that reimplement the kernel's intended semantics and check the *model*, with the actual GPU pipeline test `#[ignore]`d (e.g. `shmem_groupby_e2e.rs:165`, `:235`; `tier2_groupby_e2e.rs:182`, `:319`). Useful as a spec, but the model and the kernel can diverge silently.

### 1.3 What only runs on a real GPU (dark in CI)

Effectively **all execution-correctness assets**:

- `diff_duckdb.rs` — the DuckDB differential oracle. **Every test `#[ignore]`d** (`diff_duckdb.rs:567,586,607,638,666,685,705,732,787,808,835,857,885,906`) because `Engine::new()` opens a CUDA context (`diff_duckdb.rs:31`).
- `e2e_tests.rs` — online engine execute, all `#[ignore]`d (`:528…:1850`).
- `gpu_join_e2e.rs` (23 tests), `joins_e2e.rs`, `non_equi_join_test.rs` — all `gpu:join`.
- `sort_e2e.rs` — 20 tests, all `gpu:sort`.
- `aggregate_nulls_e2e.rs`, `stddev_e2e.rs`, `decimal_type_test.rs`, `register_stream_test.rs`, `metrics_hotpath_test.rs` (e2e arm), `memory_pool_stress.rs`, `memory_tests.rs` (live arm) — all GPU-gated.
- `sql_proptest.rs` property 4 (semantic diff vs DuckDB) — `#[ignore = "gpu:proptest-semantic"]` (`:707`).

### 1.4 Honest coverage estimate

- **Host-side frontend/planner/optimizer/codegen-shape: ~70–80% verified in CI.** Strong proptest + dense unit tests + PTX substring checks.
- **GPU execution (cuda, jit-runtime, exec device paths, mempool, kernels): ~0% verified in CI.** The tests exist and many look good, but no CI runner executes them. The `coverage` job (`ci.yml:92`) runs `cargo llvm-cov … --features cuda-stub`, i.e. it measures the *same dark-GPU* config — so even the reported lcov number understates nothing for GPU; those lines are simply never hit.
- **Blended "real verified coverage" of the whole engine: realistically 35–45%**, dominated by the host half. **The 85% claim is not met** once you discount `#[ignore]`d device code. CI proves "the host pipeline plans and emits plausible PTX"; it does **not** prove "queries return correct answers on a GPU."

---

## 2. Test Quality

### 2.1 Meaningful vs shallow

- **`diff_duckdb.rs` is the best asset in the repo** and it is genuinely wired: real bundled DuckDB (`Cargo.toml:77`), data loaded into both engines, results decoded into a normalized `Cell` enum with **null-aware, tolerance-based** comparison (`approx_eq`, `close_enough` at `REL_TOL = 1e-9`, `diff_duckdb.rs:79,109`), row-order canonicalization, and full both-sides dump on mismatch. Cases are chosen to trip specific historical regressions (C1 HAVING-dropped, C2 NULL→0, C3 DISTINCT collision, C5 multi-SUM misalignment — `diff_duckdb.rs:21-26`, `:608,:639`). **This is exactly the right design — and it never runs in CI.** This is the highest-leverage fix in the whole report.
- **GPU e2e tests assert real values, not just no-panic.** E.g. `register_stream_test.rs:73` checks `SUM=666`, `:85` checks `COUNT=9`; sort tests assert a true permutation + monotonic order (`sort_e2e.rs:793-802`). Quality is good *where it runs* — the problem is purely that it doesn't run.
- **Some tests self-oracle against Rust `Vec::sort()`** (`sort_e2e.rs:800,832`) rather than DuckDB. For integer sort this is fine; for UTF-8 it bakes in byte-lexicographic ordering as "correct" (see §3).

### 2.2 PTX golden tests — substring, not golden (CONFIRMED)

The jit agent's finding is **confirmed**:
- **No committed golden PTX files.** `find … -name '*.snap'` → none; no `tests/snapshots/` directory exists anywhere in the repo.
- The "register-flow" snapshot layer is **9 `#[ignore = "bootstrap"]` functions** (`ptx_golden_tests.rs:1391-1458`) that would *create* snapshots on first `cargo insta accept`. Since none are committed and CI never runs `--ignored`, **the snapshot layer is entirely dark** — `normalize_ptx` (a 100-line, non-trivial normalizer) is itself only checked by 2 tiny self-tests (`:1463`).
- The real content is **~202 `ptx.contains("…")` substring assertions**. These verify mnemonics/dtypes/markers are present (e.g. `ld.global.nc.s32`, `cvt.s64.s32` at `:277-279`). They catch "dropped instruction" regressions but **cannot catch wrong register wiring, wrong operand order, or semantically-wrong-but-textually-present PTX**. Calling them "golden" overstates them; they are shape assertions.

### 2.3 Proptest — well designed (for what it covers)

`sql_proptest.rs` is solid: a constrained SQL grammar feeding three properties — parse/schema/lower **never panic** — each under `catch_unwind` with shrink-reporting (`:614,:638,:652,:668`), 256 cases each. This is a legitimately good fuzz harness for the frontend and **runs in CI**. Its weakness is scope: properties 1–3 only assert *absence of panic*, not result correctness. The one property that checks correctness (property 4, semantic diff vs DuckDB, `:708`) is `#[ignore]`d.

### 2.4 Known-WRONG behavior locked in (CONFIRMED + worse than a single test)

The SUBSTRING mid-codepoint finding is **confirmed and is a real correctness divergence, not just a quirky test**:

- The host implementation `sql_substring` (`src/exec/string_ops_extended.rs:126`) treats `start`/`length` as **byte offsets** (`byte_start_raw = (start-1) as usize`, `:133`), then rounds endpoints **down** to char boundaries.
- The unit test `substring_unicode_boundary` (`src/exec/string_ops_extended.rs:697`) **asserts `SUBSTRING("héllo", 1, 2) == "h"`** (`:705`) and `SUBSTRING("héllo", 1, 3) == "hé"` (`:710`).
- **This is wrong by the SQL standard and disagrees with DuckDB**, which index SUBSTRING by **characters**: `SUBSTRING('héllo', 1, 2)` is `'hé'` and `SUBSTRING('héllo', 1, 3)` is `'hél'`. The test encodes byte-slicing as the contract and would block a future correctness fix.
- Crucially, **`diff_duckdb.rs` would catch this immediately** — but it's `#[ignore]`d, so the byte/char divergence is invisible to automation and actively *defended* by a green unit test.

This is the canonical failure mode of this suite: a host "mirror" test certifies behavior that the oracle (DuckDB) would reject, and the oracle never runs.

---

## 3. Gaps — untested / weakly-tested areas

Verified absences (grep across `tests/`):

1. **OOM / allocation-failure paths — NONE.** No test induces an allocation failure or asserts graceful error vs. abort. (`grep -i 'oom|out.of.memory'` → only the word "cancel" in a `stddev_e2e.rs` comment.)
2. **Eviction / VRAM pressure — NONE.** `memory_pool_stress.rs` exercises churn/reuse/drain (`:46,:100,:224`) but all `#[ignore]`d and none force eviction under a capped pool.
3. **Drop-during-kernel / in-flight cancellation — NONE.** No test drops a `QueryHandle`/engine while a kernel is running, nor cancels an in-flight query. `pool_drain_after_context_drop` (`memory_pool_stress.rs:283`) is the closest and is GPU-gated.
4. **Streaming — scaffold only, never exercised.** `register_stream_test.rs` pins the *eager-consumption* API shape, both tests `#[ignore]`d (`:44,:94`). The file itself documents lazy per-batch streaming as "v0.7+ expected" (`:5-10`) — i.e. the streaming engine path is unimplemented and untested. Confirmed.
5. **Multi-key UTF-8 ordering — NONE.** `sort_e2e.rs` multi-key tests are **int/int only** (`multi_key_int_int`, `:280`). All Utf8 sort tests use **ASCII-only** fixtures (`alphabet` of lowercase words, `:761`) and self-oracle against Rust byte-sort (`:800`). No multi-byte codepoint, no mixed-case collation, no NULL-in-multi-key-Utf8.
6. **All-null aggregates / joins — partial.** Scalar agg over nulls exists (`aggregate_nulls_e2e.rs`) but is GPU-gated; **all-null group key**, **join key entirely null**, and **all-null GROUP BY column** are not covered.
7. **Spill paths — NONE.** No test forces a hash join or group-by to spill.
8. **Concurrency / thread-safety of global caches — NONE meaningful.** The grep hits in `qualified_columns_test.rs`/`tracing_test.rs` are `Arc<Mutex>` in tracing-capture plumbing, not engine-cache stress. `pool_concurrent_alloc` (`memory_pool_stress.rs:145`) is the only concurrency probe and is GPU-gated. The JIT/kernel cache and any global `static` caches are not stress-tested from multiple threads.

### High-value tests to add (priority order)

1. **Un-gate the DuckDB differential harness in a GPU CI lane.** It already exists and is excellent; the single biggest coverage win is making `diff_duckdb.rs` + `sql_proptest.rs` property 4 *run*.
2. **Fix + re-spec SUBSTRING for characters**, then add a `diff_duckdb` case with multi-byte input (`'héllo'`, `'世界'`, emoji) asserting char semantics. Delete/invert the byte-slicing assertion at `string_ops_extended.rs:705`.
3. **Multi-key UTF-8 sort** vs DuckDB: `ORDER BY s1 ASC, s2 DESC` over multi-byte strings with NULLs and `NULLS FIRST/LAST`; assert against the oracle, not `Vec::sort`.
4. **OOM/eviction**: cap the pool, allocate past capacity, assert clean `BoltError` (not panic/abort); assert eviction reclaims and re-serves.
5. **Drop-during-kernel & cancellation**: launch a long kernel, drop the handle/engine; assert no UB / clean teardown (run under `compute-sanitizer` on the GPU lane).
6. **Concurrency**: N threads issuing distinct queries through one engine; assert JIT-cache correctness (no torn kernels, no cross-query result bleed).
7. **All-null edge matrix**: all-null group key, all-null join key (inner→empty, outer→null-extended), all-null agg input (`SUM`→NULL, `COUNT`→0) — as `diff_duckdb` cases.
8. **Commit the `insta` PTX snapshots** (run the 9 bootstrap tests once, check in `tests/snapshots/`, drop the `#[ignore]`) so the register-flow layer becomes a real regression gate in CI.

---

## 4. CI Story

### Current state
- **No GPU runner exists.** CI is `ubuntu-latest` + `windows-latest`, `cuda-stub`, no CUDA device (`ci.yml:20,53,71`). The author acknowledges this explicitly: a GPU lane running `cargo test --features cudarc -- --ignored` is described as a *comment*, not a job (`ci.yml:88-91`).
- **Coverage is measured but not gated.** The `coverage` job runs `cargo llvm-cov --features cuda-stub` and uploads lcov, with **`continue-on-error: true` and intentionally NO threshold** (`ci.yml:78-82,92-95`). The rationale given (GPU paths are dark, a blanket floor is unrealistic) is honest — but it means coverage is informational only and the dark GPU half is invisible.
- **Supply-chain gates are solid** (`cargo-deny` blocking + all-features informational, `ci.yml:149,189`) — unrelated to test coverage but worth noting as the one genuinely enforced quality gate besides fmt/clippy/host-tests.
- Net: **GPU correctness is entirely unverified by automation.** Green CI guarantees "host pipeline compiles, plans, lints, and emits plausible PTX," nothing about device results.

### Recommended path to trustworthy coverage
1. **Stand up a self-hosted GPU runner** (or scheduled cloud-GPU job; cost-bound it to nightly + pre-release if needed). Add the already-sketched lane:
   `cargo test --no-default-features --features cudarc -- --ignored`
   This single change converts the entire `diff_duckdb` / e2e / sort / join / mempool suite from dark to live.
2. **Run device tests under `compute-sanitizer`** (memcheck + racecheck) in that lane to catch the untested drop/cancel/concurrency UB.
3. **Add a separate `llvm-cov` run on the GPU lane** with `--features cudarc -- --ignored`, and only *there* introduce a coverage floor (start ~60%, ratchet). Keep the host `cuda-stub` coverage informational as today.
4. **Commit the PTX `insta` snapshots** and un-`#[ignore]` them so register-flow regressions gate on the existing CPU runners (no GPU needed).
5. **Add a nightly proptest-semantic job** (property 4) on the GPU lane with a higher case count — it is the cheapest way to broaden differential coverage beyond the 8 curated `diff_duckdb` cases.
6. **Make the GPU lane required for release tags** even if it stays non-blocking on PRs, so no release ships with the device half unverified.

---

## Appendix — key file:line citations

- CI test command (CPU/stub only): `.github/workflows/ci.yml:71`
- Coverage job, no threshold, dark GPU acknowledged: `.github/workflows/ci.yml:78-95`
- GPU lane is a comment, not a job: `.github/workflows/ci.yml:88-91`
- DuckDB differential harness, all `#[ignore]`d: `tests/diff_duckdb.rs:31`, `:567-906`
- DuckDB null-aware compare: `tests/diff_duckdb.rs:79,109`
- Proptest properties 1–3 (run in CI): `tests/sql_proptest.rs:638,652,668`
- Proptest property 4 (semantic, ignored): `tests/sql_proptest.rs:707`
- PTX substring assertions (run in CI): `tests/ptx_golden_tests.rs:260-279` (×~202)
- PTX snapshot bootstrap (ignored, no `.snap` committed): `tests/ptx_golden_tests.rs:1391-1458`
- No `.snap` files / no `tests/snapshots/` dir: verified via `find`
- SUBSTRING byte-slicing impl: `src/exec/string_ops_extended.rs:126,133`
- SUBSTRING known-wrong test locked in: `src/exec/string_ops_extended.rs:697,705,710`
- Sort: ASCII-only Utf8 fixtures, self-oracle vs `Vec::sort`: `tests/sort_e2e.rs:761,800`; int-only multi-key: `:280`
- Streaming = eager scaffold, ignored: `tests/register_stream_test.rs:5-10,44,94`
- Mempool stress, all ignored: `tests/memory_pool_stress.rs:46-283`
- Bench GPU gate (`BOLT_BENCH_GPU=1`, else skip bolt): `benches/olap_benchmarks.rs:531-539`
