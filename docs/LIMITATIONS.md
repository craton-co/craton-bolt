<!-- SPDX-License-Identifier: Apache-2.0 -->
# Craton Bolt — Limitations

**Crate:** `craton-bolt` v0.7.0 · **License:** Apache-2.0

This is the 30-second, honest read on what Craton Bolt **cannot** do yet, what
it **requires**, and why it is **not production-ready**. It consolidates caveats
that are otherwise scattered across [`ROADMAP.md`](../ROADMAP.md) ("Known
limitations"), [`docs/SQL_REFERENCE.md`](SQL_REFERENCE.md) ("What's NOT
supported"), [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), and
[`docs/PATH_TO_1.0.md`](PATH_TO_1.0.md). If you only read one limitations page,
read this one.

---

## Not production ready

**Do not depend on Craton Bolt in production.** It is a pre-1.0 (0.x), actively
developed research/engineering engine. The public API is unstable, GPU code
paths are not verified in CI, and several SQL features fall back to host-side
execution or are only partially lowered to the GPU. Use it for evaluation,
experimentation, and benchmarking — not for systems where correctness,
stability, or availability matter.

---

## Hardware / toolkit requirements (hard requirements)

Running anything beyond a type-check requires an NVIDIA GPU and CUDA toolkit:

- **GPU compute capability ≥ 7.0 (`sm_70`, Volta or newer).** The emitted PTX
  declares `.target sm_70`; older GPUs (Pascal `sm_6x` and below) are not
  supported. There is no `sm_70`-downlevel fallback.
- **NVIDIA CUDA Toolkit ≥ 12** with a matching driver. The `cudarc` binding is
  pinned to the CUDA 12.x ABI (`cuda-12060`). CUDA 11.x is not supported.
  - **Windows MSVC:** `cuda.lib` must be on the linker path. See
    [`docs/INSTALL.md`](INSTALL.md) for the CUDA 13.2 / `__imp_*` workaround.
  - **Linux (x86_64):** `libcuda.so` must be on the linker path.
- **No CUDA toolkit installed?** You can still type-check / build the library
  with `--no-default-features --features cuda-stub` (this is the docs.rs and
  CUDA-less CI path), but the stub cannot execute any query on a GPU.

### Platform support

| Platform | Status |
|---|---|
| Linux x86_64 | Supported |
| Windows x86_64 (MSVC) | Supported |
| macOS (any arch) | **Not supported** — Apple dropped CUDA in 2019. `cuda-stub` type-check only. |
| ARM aarch64 (Jetson) | In theory supported; **not tested**. |

---

## Pre-1.0 API instability

- The crate is `0.7.0`. Per [SemVer](https://semver.org/), pre-1.0 the public
  API may break on any minor (`0.x`) bump. The IR, `KernelSpec`, executor
  surface, and several `Engine` methods are still moving.
- See [`docs/API_SURFACE.md`](API_SURFACE.md) for the per-item stability tiers
  and [`docs/MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md) for breaking-change
  deltas across releases.

---

## GPU paths are not verified in CI

- CI (`.github/workflows/ci.yml`) builds, tests, lints, and runs `cargo deny`
  using the **`cuda-stub`** feature only. It exercises **0 GPU code paths** —
  no GPU runner exists.
- The live-GPU integration tests are `#[ignore]`-gated and run **separately**
  on developer/maintainer hardware (`cargo test --features cudarc -- --ignored`
  on a GPU host). A scheduled, allow-failure GPU lane stub is documented in the
  CI workflow for when a GPU runner becomes available.
- Practical consequence: a green CI run validates host logic, planning, and
  codegen *shape* (PTX-string assertions), **not** end-to-end GPU execution.

---

## Known semantic gaps

These are real, code-level behaviors to be aware of:

- **Host-side fallbacks, not always GPU.** Despite the "errors instead of
  silently falling back" aspiration stated for 1.0
  ([`docs/PATH_TO_1.0.md`](PATH_TO_1.0.md) §5), the **current** engine routinely
  falls back to host-side execution for sort, some joins, set ops, window
  functions, string functions, and `DISTINCT`. "Runs" does not always mean "ran
  on the GPU." See [`docs/SQL_REFERENCE.md`](SQL_REFERENCE.md) for the
  per-feature execution tier (GPU / host-side / GPU-lowering-pending).
- **`NOT IN` / `IN` with NULL — three-valued-logic caveat (known foot-gun).**
  SQL three-valued logic around `NOT IN (... NULL ...)` has historically been a
  correctness foot-gun in GPU engines (a `NULL` in the list makes the predicate
  `UNKNOWN`/false for non-matching rows). The engine drops NULLs from a folded
  `IN`/`NOT IN` value set, which matches strict SQL for `IN` under `WHERE` but
  **diverges for `NOT IN` against a set containing NULLs**. Verify behavior
  against your reference engine before relying on `NOT IN`/`IN` over nullable
  lists.
- **Grouped integer `SUM` overflow may go undetected for streaming inputs.**
  Scalar and grouped integer `SUM` overflow is normally a hard error
  (`BoltError::Type("SUM(integer) overflow")`; see
  [`docs/SQL_REFERENCE.md`](SQL_REFERENCE.md)). For the **grouped** case the
  overflow is currently detected via a **host-side recompute** of the per-group
  sums, so an overflow may **not** be caught for streaming inputs the host
  cannot replicate (the device produced the result but the host has no way to
  re-derive it for the check). Tracked follow-up: an on-device overflow flag
  that makes the check independent of the host recompute.
- **String handling is dictionary/ASCII-oriented.** String predicates operate
  over dictionary-encoded literals, and the GPU string functions
  (`UPPER` / `LOWER` / `LENGTH`) are byte/ASCII-oriented. Treat non-ASCII /
  multi-byte UTF-8 case-folding and length-in-characters semantics as
  unverified — `LENGTH` is byte-length, and case conversion is ASCII-range.
- **Temporal types partially lower; host fallback on upload.** `Decimal128`,
  `Date32`, and `Timestamp` parse and **partially** lower to the GPU. Some
  temporal paths fall back to a host upload/compute step rather than running
  fully on-device. See the type tiers in
  [`docs/SQL_REFERENCE.md`](SQL_REFERENCE.md).
- **Env-gated experimental paths.** Several kernels (e.g. GPU radix sort,
  CUDA-graph sort, alternate hash/scan algorithms) are gated behind
  environment variables and are not the default path. See
  [`docs/ENV_VARS.md`](ENV_VARS.md).
- **Builder knobs not fully wired.** Some `EngineBuilder` knobs (e.g. the
  persistent-cache option) are not yet connected to `build()`. See
  [`ROADMAP.md`](../ROADMAP.md) "Known limitations."

---

## Rejected SQL constructs

The frontend **cleanly rejects** the following constructs with a
`BoltError::Sql(...)` (a clear parse/plan-time error), **not** a crash or a
silent wrong answer. If you depend on any of these, expect an error and
rewrite the query (see [`docs/SQL_REFERENCE.md`](SQL_REFERENCE.md) for the
supported alternatives):

- **Derived tables / subqueries in `FROM`** — use a `WITH`/CTE instead.
- **Correlated subqueries** (and `EXISTS` / `NOT EXISTS`).
- **`GROUP BY ROLLUP` / `CUBE` / `ALL`** (and `TOTALS`).
- **`WINDOW` clause / `QUALIFY` / named windows** (`OVER <named_window>`), and
  non-default window frames.
- **`VALUES` lists** (as a standalone row source).
- **`WITH RECURSIVE`** and **CTE column-list aliases** (`WITH c (a, b) AS ...`).
- **Table-valued functions.**
- **`DISTINCT ON (...)`** (Postgres extension).
- **`TOP`** (T-SQL row limit).
- **`LATERAL`.**
- **`PREWHERE`** (ClickHouse extension).
- **`GLOBAL JOIN`** (ClickHouse extension).

These return a clean error rather than crashing, so the engine gives you correct
expectations up front.

---

## Where to read more

- [`docs/SQL_REFERENCE.md`](SQL_REFERENCE.md) — authoritative supported SQL
  subset and per-feature execution tier.
- [`docs/INSTALL.md`](INSTALL.md) — prerequisites, CUDA version notes, and
  platform-specific build workarounds.
- [`ROADMAP.md`](../ROADMAP.md) — forward plan and the current "Known
  limitations" list.
- [`docs/PATH_TO_1.0.md`](PATH_TO_1.0.md) — what 1.0 is (and is not) intended
  to be.
