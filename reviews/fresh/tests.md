# Bolt — Test Suite, Benchmarks & Examples Review

Repo: `C:\Projects\bolt` (branch `dev`). Scope: `tests/` (~22.4K LOC, 45 files),
`benches/` (4 files), `examples/` (2 files). src ~140 `.rs` files.

---

## 0. Executive summary

The suite is **broad in surface area but shallow in CI-enforced depth**. There
are 664 `#[test]` functions, but **243 (~37%) are `#[ignore]`-gated** and require
a live NVIDIA GPU. The single GitHub CI lane that would run them
(`gpu-integration`) is hard-disabled with `if: false`
(`.github/.wf/ci.yml:256`). Therefore **every test that exercises real GPU
execution — i.e. essentially all of `src/exec` and `src/cuda` runtime behavior —
never runs in CI**. What CI actually validates is the host-only frontend:
parser, planner/optimizer, type-checking, and *PTX text generation* (the JIT
emits strings without a GPU). The 85% coverage target is, on CI, unreachable and
unmeasured against the modules that matter most.

---

## 1. TEST SUITE REVIEW

### 1.1 GPU gating — the central issue

- Gating is done purely by `#[ignore = "gpu:*"]` string tags, documented in
  `tests/common/mod.rs:11-23`. There is **no runtime GPU-probe that skips
  gracefully** — GPU tests call `Engine::new().expect("CUDA ctx")` /
  `CudaContext::new(0).expect(...)` (e.g. `tests/semantics_e2e.rs:82`,
  `tests/memory_pool_stress.rs:36`, `tests/memory_tests.rs:155`). Run with
  `--ignored` on a non-GPU host, they **panic** rather than skip. This is a
  deliberate convention (the `#[ignore]` is the skip), but it means there is no
  "run everything, auto-skip GPU" mode.
- CI test job: `cargo test --lib --tests --features cuda-stub
  --no-default-features` (`.github/.wf/ci.yml:78`). `cuda-stub` stubs the FFI to
  return `CUDA_ERROR_STUB`, so `Engine::new()` fails and **only the ~421
  non-ignored host tests run**.
- The GPU lane (`gpu-integration`, lines 252-278) is `if: false` +
  `continue-on-error: true` — a documented placeholder awaiting a self-hosted
  runner. **It has never executed.**
- Coverage job (`.github/.wf/ci.yml:99-138`) runs `cargo llvm-cov
  --no-default-features --features cuda-stub --lib` — **`--lib` only, not
  `--tests`**, host-only, and explicitly `continue-on-error: true`
  ("informational, non-blocking"). There is **no enforced coverage floor** and
  the 85% target is aspirational, not gated.

**De-facto CI coverage vs local:** CI exercises the SQL→plan→PTX-string path
only. A developer with a GPU running `cargo test --features cudarc -- --ignored`
gets dramatically more, but that is a manual, unenforced ritual.

### 1.2 Honest coverage estimate vs 85% target

No coverage artifact is committed, so this is an informed estimate by
cross-referencing the src module map against which tests can run in CI:

| src area | files / size | CI-runnable coverage | Notes |
|---|---|---|---|
| `plan` + `plan/optimizer` | ~22K LOC | **High (likely 70-85%)** | parser_tests (1158 LOC), sql_proptest, did_you_mean, qualified_columns, between/case/cast/in_list all run host-side. |
| `jit` PTX **emission** | 39K LOC, 39 kernel files | **Moderate (text only)** | ptx_golden_tests (216 substring asserts) + 40 insta snapshots validate emitted PTX text, no execution. |
| `jit` compiler/cache runtime | — | **Low** | disk_cache, PTX cache eviction beyond env_var_smoke largely untested on CI. |
| `exec` core | ~70K LOC | **Very low on CI** | All real execution is `#[ignore]`. |
| `exec/groupby` (32 files) | large | **~0% on CI** | Tier1/Tier2/shmem all GPU-gated. |
| `exec/strings` | — | **~0% on CI** | string_ops_e2e / string_fns_sql all GPU-gated. |
| `cuda` (11.5K LOC) | mempool, GpuVec, context | **~0% on CI** | memory_tests, memory_pool_stress all GPU-gated. |

**Estimate:** locally-with-GPU the suite plausibly approaches the 85% line for
the breadth of SQL features; **on CI alone, whole-crate line coverage is likely
~35-45%**, concentrated in `plan` and JIT text emission. The **least-covered
areas are exactly the highest-risk ones**: `exec/groupby` (32 files),
`exec/strings`, `exec` core, and `cuda` mempool/runtime.

### 1.3 Test quality

**Golden / snapshot (`ptx_golden_tests.rs`, `ptx_golden_partition_snapshots.rs`):**
These are a **mix of behavioral and pure-text** checks. The substring assertions
are genuinely behavioral contracts on codegen — e.g. `ld.global.nc.s32` (read-only
cache load), `cvt.s64.s32` (correct widening), `setp.eq.s64` + gate-before-store
(`tests/ptx_golden_tests.rs:277-305`). These catch real codegen regressions
(wrong dtype, missing predicate gate). The 40 `assert_snapshot!` insta snapshots
(`tests/snapshots/*.snap`) are **text pinning** after `normalize_ptx`
(register-renumbering) — they detect *any* PTX drift but cannot tell a
correctness fix from a regression; a human must re-bless. **Caveat: passing PTX
text does not prove the kernel computes correctly on hardware** — only execution
(`#[ignore]`) does that, and that never runs in CI. So the golden tests give
false confidence if read as "groupby works".

**Property testing (`sql_proptest.rs`, 731 LOC):** Good. A constrained SQL
grammar (proj/where/group/having/order/limit, recursive bool exprs) with 256
cases asserts parse/schema/lower **never panic** (catch_unwind + shrinker,
`tests/sql_proptest.rs:627-678`). This is the right invariant and runs on CI.
The strong **semantic** property (Property 4, differential vs DuckDB,
line 706) is `#[ignore = "gpu:proptest-semantic"]` with only 32 cases — so the
high-value behavioral property does not run on CI.

**Differential vs DuckDB (`diff_duckdb.rs` 918 LOC, `diff_duckdb_semantics.rs`
541 LOC):** This is the strongest behavioral asset — runs identical SQL on Bolt
and bundled DuckDB and compares row-sets with order-agnostic canonicalisation
and tight float tolerance (`REL_TOL = 1e-9`, `tests/common/mod.rs:32`). Well
built (proper NULL/typed-cell handling, appender fixtures). **But all 18 tests
are `#[ignore = "gpu:e2e"]`** and panic on init failure (line 341) — so DuckDB,
the oracle, is never consulted on CI.

### 1.4 Concrete gaps

- **Spill paths:** "spill" appears **only in PTX snapshot filenames** (the
  partition-reduce spill *codegen* variant) — there is **no runtime test that
  forces a spill** (oversized group cardinality exceeding shmem) and checks
  correctness/perf. High risk given 32 groupby files.
- **NULL semantics:** reasonably covered at the SQL level (`is_null_test.rs` 685
  LOC, `aggregate_nulls_e2e.rs`, `coalesce_nullif_test.rs`) — but the e2e ones
  are GPU-gated, so on CI only the planner's null typing is checked.
- **Overflow:** only **const-fold / parse-time** overflow is tested on CI
  (`wave3_regression_test.rs:111,150` i32 fold no-wrap; `parser_tests.rs:318`;
  `e2e_tests.rs:439`). `decimal_type_test.rs:420` checks SUM(Decimal128)
  overflow but is GPU-gated. **Runtime `SUM(i64)` / `SUM(Int32)`-widening
  overflow on the GPU is untested** (grep found no saturating/wrapping
  assertions on aggregate output).
- **Error paths:** thin. `assert ... Err` appears in only ~8 files (mostly
  parser/plan rejection). Few tests for OOM, allocation failure, kernel launch
  failure, or malformed-input runtime errors.
- **Concurrency:** only `memory_pool_stress.rs::pool_concurrent_alloc`
  (`:143`) — and it spins a fresh `CudaContext` per worker (correctly noting
  `CudaContext` is `!Sync`). No concurrent *query* execution test. GPU-gated.
- **Memory-pool stress:** `memory_pool_stress.rs` (alloc/free churn, drain,
  concurrent) + `memory_tests.rs` exist and are reasonable, but **GPU-gated →
  zero CI signal** on the entire `src/cuda` allocator.

---

## 2. BENCHMARKS

Four Criterion benches; **all designed to run host-only under `cuda-stub`**, so
they are reproducible without a GPU (`benches/regression.rs:12-25`).

- **`regression.rs`** is the standout: a deliberate, narrow tripwire over
  `parse` / `lower` / `ptx_gen` stages with **stable bench ids**
  (`regression/<stage>/<query>`) and a **self-contained, opt-in regression
  guard** via env vars: `BOLT_REGRESSION_WRITE_BASELINE` to mint and
  `BOLT_REGRESSION_BASELINE` + `BOLT_REGRESSION_THRESHOLD_PCT` to enforce
  (`benches/regression.rs:56-89`). Missing/garbled baseline is skipped, never
  panics. **Weakness:** **no baseline JSON is committed** to the repo (grep
  found none) and **CI does not run the guard** — so regression detection is
  purely manual/local. It is a good mechanism with no automation wired up.
- `olap_benchmarks.rs`, `query_benchmarks.rs`, `utf8_sort_bench.rs`: meaningful
  workloads but, being execution-oriented, only measure host paths under
  `cuda-stub`; the GPU timings developers care about aren't captured in any
  reproducible CI artifact.
- **Verdict:** meaningful and reproducible host-side; **not regression-guarded
  in CI** (no committed baseline, no enforcing job).

---

## 3. EXAMPLES

- `examples/quickstart.rs` and `examples/groupby.rs` are concise, correct, and
  **degrade gracefully** when no GPU is present — they match on `Engine::new()`
  and print a clear "requires CUDA / recompile with `--features cuda-stub`"
  message then `return Ok(())` (`quickstart.rs:31-41`, `groupby.rs:34-40`).
- They **compile on CI** (`cargo check --features cuda-stub`) and serve as good,
  minimal docs (build a RecordBatch → register → SQL). `groupby.rs` even prints
  a timing.
- **Gap:** no example for the DataFrame/builder API, joins, or streaming
  register — and because they short-circuit without a GPU, CI proves they
  *compile* but not that they *run* end-to-end.

---

## 4. PRIORITIZED ENHANCEMENTS to credibly reach 85%

1. **Provision (or rent) a GPU CI lane and flip `gpu-integration` to `if: true`.**
   This is the single highest-leverage change: it would activate the 243
   already-written `#[ignore]` tests + the DuckDB differential + the semantic
   proptest. Without it, no amount of new tests changes CI coverage of `exec`/
   `cuda`. (`.github/.wf/ci.yml:256`)
2. **Make the coverage job measure `--tests` and report whole-crate, and emit a
   GPU-lane coverage artifact** (`cargo llvm-cov --features cudarc -- --ignored`)
   so the 85% target is actually measured against `exec`/`cuda`, not just
   `--lib` host paths. (`.github/.wf/ci.yml:132`)
3. **Add runtime spill tests** for `exec/groupby`: high-cardinality GROUP BY that
   forces the partition-reduce spill path, asserting correctness vs DuckDB. The
   spill *codegen* is snapshotted but never executed.
4. **Add runtime aggregate-overflow tests** (SUM(i64) near `i64::MAX`,
   SUM(Int32) widening boundary) asserting defined behavior — currently only
   const-fold overflow is covered on CI.
5. **Expand the differential-vs-DuckDB grammar** in `sql_proptest.rs` Property 4
   (joins, multi-key group by, string fns, CASE/COALESCE) and **raise its case
   count** from 32; it is the best correctness oracle and is underused.
6. **Commit a `regression-baseline.json`** and add a CI job (host, `cuda-stub`)
   that runs `BOLT_REGRESSION_BASELINE=... BOLT_REGRESSION_THRESHOLD_PCT=...` so
   the existing regression mechanism actually guards.
7. **Add error/failure-path tests:** allocation failure, kernel launch error,
   pool exhaustion (these can partly run under a fault-injecting stub without a
   GPU, decoupling them from the GPU lane).
8. **Add a graceful auto-skip helper** in `tests/common/mod.rs` (probe CUDA,
   `eprintln!` + `return` instead of `expect`-panic) so `--ignored` is runnable
   on mixed hosts and the GPU lane is robust.
9. **Add a concurrent multi-query execution test** (beyond pool alloc churn) and
   an example covering joins / DataFrame API.

---

### Key file references
- Harness: `tests/common/mod.rs` (PRNG + shuffle only; `#[ignore]` taxonomy at :11-23)
- GPU gating disabled: `.github/.wf/ci.yml:256` (`if: false`); CI test cmd `:78`; coverage `:99-138` (`--lib` only, non-blocking)
- Property tests: `tests/sql_proptest.rs:627-731` (semantic prop GPU-gated at :707)
- Differential: `tests/diff_duckdb.rs` (all `#[ignore]`, init panic `:341`), `tests/diff_duckdb_semantics.rs`
- Golden PTX: `tests/ptx_golden_tests.rs` (216 substring asserts; behavioral at :277-305), 40 snapshots in `tests/snapshots/`
- Regression bench: `benches/regression.rs:56-131` (no committed baseline, not in CI)
- Examples: `examples/quickstart.rs:31-41`, `examples/groupby.rs:34-40` (graceful GPU fallback)
