# Craton Bolt — Consolidated Review (2026-05-30)

Parallel multi-agent review, one Opus agent per module. Per-module detail in the
sibling files: `cuda.md`, `exec_core.md`, `exec_groupby.md`, `exec_strings.md`,
`jit_kernels.md`, `jit_core.md`, `plan.md`, `core.md`, `tests.md`, `docs.md`,
`oss_readiness.md`.

Scope reviewed: ~149K LOC src, ~22K LOC tests, all docs and OSS metadata.

## Headline

The codebase is mature, unusually well-documented, and most classic hazards are
already hardened. There are **no Critical memory-safety bugs**. The real risks are:
(1) a small set of correctness bugs in cold paths, (2) two release blockers that are
trivial to fix, (3) CI that exercises almost no GPU code, and (4) heavy structural
duplication that is a maintainability tax, not a bug.

## Release blockers (fix before any public release / publish)

| # | Issue | Evidence | Fix |
|---|-------|----------|-----|
| B1 | **CI does not run.** Workflows are in `.github/.wf/`; GitHub only runs `.github/workflows/`. README badge dead, DCO unenforced, cargo-deny ungated. | `.github/.wf/ci.yml`, `dco.yml` | `git mv .github/.wf .github/workflows` |
| B2 | **Internal scratch file ships in the crate.** `exclude` lists `continue_prompt.md` but the file is `bolt_continue_prompt.md`; it is git-tracked and will be packaged. It also contradicts shipping docs. | `Cargo.toml:15`, `bolt_continue_prompt.md` | Remove file + fix glob |

## Critical / High correctness bugs

| Sev | Bug | Location | Notes |
|-----|-----|----------|-------|
| Critical | **Projection pruning drops renamed right-side join column on name collision** → wrong result / schema error for `SELECT a.id FROM a JOIN b` with colliding names. An in-tree probe test exists but only `eprintln!`s instead of asserting. | `plan/optimizer/projection_pruning.rs:199-306`, `logical_plan.rs:2398` | Highest-priority real bug. |
| Critical | **JIT cache key omits GPU arch / PTX ISA / driver.** Latent today (target hardcoded `sm_70`), becomes silent cross-arch mis-routing the moment target is device-derived. | `jit/ptx_gen.rs:15`, `disk_cache.rs:236` | Fold device cap into `codegen_salt()` now. |
| Critical | **Integer DIV by zero / INT_MIN/-1** lowered straight to `div.s32/s64`, no guard → GPU UB, no SQL NULL/error semantics. | `jit/ptx_gen.rs:1534` | |
| High | **Grouped float MIN/MAX silently ignores NaN**, while scalar MIN/MAX treats NaN as largest → `MAX(v)` and `k,MAX(v) GROUP BY k` disagree on same NaN data. | `exec/groupby_tier2_minmax_float_exec.rs:179`, `jit/float_atomics.rs:40`, `exec/aggregate.rs:1836` | |
| High | **Radix GPU sort is non-stable** (racing `atom.global.add`); LSD radix requires per-pass stability → wrong sort. Gated behind `BOLT_GPU_SORT=1` (off by default). | `jit/sort_kernel_radix.rs:592-655` | |
| High | **Host JOIN on dict-encoded keys compares dictionary indices, not strings** → wrong join when both sides carry independent `DictionaryArray`s. | `exec/join.rs:1148-1191` | |
| High | **JIT NULL propagation is coarse** — ANDs all inputs into every output; precise per-output sweep exists but is unused. Wrong for multi-output / CASE; masked by current single-output shape. | `jit/ptx_gen.rs:356-389`, unused `:3668` | |
| High | **kernels/ crate won't compile** — `feature(register_attr)` (removed from rustc) + 2026 nightly pin + 2022-era `cuda_std 0.2.2`. Off-by-default, but anyone enabling `rust-cuda` hits a hard failure. | `kernels/src/lib.rs:24`, `rust-toolchain.toml:13`, `kernels/Cargo.toml:36` | |
| High | **NULL group keys silently dropped** — diverges from DuckDB/Postgres (which emit a NULL group); untested either way. | `exec/groupby.rs:431-446` | |
| High | **ILIKE `_` wildcard desyncs on expanding-lowercase codepoints** (`İ`→`i`+combining); GPU byte-fold vs host Unicode-fold disagree for non-ASCII, safe only while `dict_is_ascii` guard holds. | `exec/like.rs:165-225`, `exec/string_project.rs:340-348` | |

## Medium themes

- **Disk JIT cache has no eviction/TTL/size bound** → unbounded growth (`jit/disk_cache.rs`).
- **Dead metrics surface:** `BytesUploaded/Downloaded`, `GpuLaunchesTotal`, 5/8 phase
  histograms declared but never written → permanently-zero Prometheus series (`metrics.rs`).
- **Massive duplication** (maintainability, not bug): groupby 32 files with ~11 copies of
  module-cache plumbing + ~10 copies of the partition→scatter→reduce pipeline; JIT i32-vs-i64
  twins (~4,929 duplicated lines across 7 file pairs). Both are CPU-diff-testable to collapse
  with a generic driver / `KeyWidth` parameter at zero behavior change.
- **Unbounded key maps** in set-ops (no memory cap, unlike DISTINCT) (`exec/setops.rs`).
- **Synchronous per-column D2H** in gather/compact ignoring the existing async/pinned helper.
- **`streaming.rs` not device-wired** — oversells capability vs docs.
- **PTXAS log truncated at 4KB**, brittle pointer cast (`jit/jit_compiler.rs:557`).

## Tests

- **CI exercises almost no GPU code.** 243 of 664 tests (~37%) are `#[ignore = "gpu:*"]`;
  the `gpu-integration` lane is hard-disabled (`if: false`, `.github/.wf/ci.yml:256`).
  CI runs `--features cuda-stub`, so all of `exec`, `exec/groupby`, `exec/strings`, `cuda`
  real execution never runs. **De-facto whole-crate CI coverage ≈ 35-45%**, concentrated in
  `plan` + JIT text emission. 85% is only plausible *locally with a GPU*.
- Coverage job is `--lib`-only, host-only, `continue-on-error` → **no enforced floor**.
- Golden/snapshot PTX tests verify *text/structure*, not runtime behavior — a passing
  snapshot never proves a launched kernel computes correctly.
- DuckDB differential suites (the best correctness oracle, ~1450 LOC) are all `#[ignore]`.
- Regression bench tripwire is well-designed but no baseline committed and CI never runs it.
- **Top action:** provision a GPU CI lane and flip `gpu-integration` on — single change that
  activates the 243 already-written tests + the DuckDB oracle. Then measure `--tests` coverage
  there, add runtime spill/overflow/error-path tests, commit a regression baseline.

## Docs

- B2 (above) + dead CI badge.
- `docs/CUDARC_ADOPTION.md` referenced from 4+ files (INSTALL, USER_GUIDE, BENCHMARKS×2,
  Cargo.toml) but **does not exist**.
- `ENV_VARS.md` wrong: lists `CRATON_DISTINCT_HOST_MAX_ROWS` and `CRATON_PLAN_CACHE_SIZE`
  as absent, but both are read at runtime (`distinct.rs:90/111`, `sql_frontend.rs:818/842`).
- **String-function tier contradictions** across ARCHITECTURE / COMPETITIVE_BENCHMARKING /
  SQL_REFERENCE / LIMITATIONS (host vs GPU; byte/ASCII vs char/Unicode for LENGTH/case).
- `API_SURFACE.md` omits public `exec::streaming::*`, `metrics::*`, `register_table_stream_lazy`.
- Gap: no user-facing versioning policy. Doc set is otherwise well-factored — nothing to merge.

## OSS / crates.io readiness (Apache-2.0, Craton Software Company)

- **Good:** `LICENSE` is correct full Apache-2.0; all 140 src files carry SPDX headers;
  `NOTICE` conventional and attributes Craton + Arrow/sqlparser; year/owner consistent;
  `deny.toml` permissive allowlist, **no GPL/AGPL/LGPL** present or allowed; Cargo.lock
  all crates.io-index; `kernels/` is `publish=false` and won't block publish; templates,
  SECURITY.md, CoC, CONTRIBUTING (DCO `-s`), DCO 1.1 all present; no secrets committed.
- **Fix:** B1 + B2; `LICENSE:189` Appendix hard-codes copyright instead of bracket
  placeholder (cosmetic); `deny.toml:85` has an unresolved RUSTSEC TODO for the rust-cuda
  `xz 0.1.0`; CODEOWNERS references `@craton-co/craton-bolt-maintainers` team that must be
  created; confirm `craton-bolt` crate name is free.
- **Top actions:** move `.wf`→`workflows`; remove `bolt_continue_prompt.md`;
  `cargo publish --dry-run`; provision GH team + craton.com.ar mailboxes; claim crate name.

## Suggested features / directions (cross-module)

1. **Wire `streaming.rs` to the device** — currently the largest capability gap vs the docs.
2. **Collapse the duplication** with a `Tier2Pipeline` generic driver (groupby) and a
   `KeyWidth`-parameterized kernel generator (jit i32/i64) — unlocks new aggregates cheaply.
3. **Finish the cost-based optimizer** — statistics exist but only drive left-deep ordering;
   add subquery decorrelation, transitive equality, limit push-through, CTE column aliases.
4. **GPU DISTINCT / set-ops / window functions** via the existing sort/partition machinery.
5. **String collations + regex / `SIMILAR TO`**; validate & wire the unused SUBSTRING/TRIM
   GPU kernels.
6. **Persistent cross-version cubin/fatbin cache** (after fixing the arch-in-key bug) with
   LRU + size bound.
7. **A GPU CI lane** + ptxas assemble-only job + device round-trip numeric tests.
8. **Total-order float MIN/MAX** consistent between scalar and grouped paths.
