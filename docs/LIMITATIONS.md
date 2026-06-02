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

## Concurrency: one active `Engine` (CUDA context) per process

- `Engine` is **`!Sync`** (it holds `RefCell` state), so a single `Engine`
  cannot be shared across threads, and `&Engine` is not `Send`.
- More fundamentally, the engine keeps **one active CUDA context per process**.
  The device-memory pool, the CUDA stream pool, and the JIT module caches are
  process-global statics whose resources (pooled pointers, streams, loaded
  `CudaModule`s) are **bound to the context that created them**. Constructing
  and using **two `Engine`s at the same time** (two contexts on one device)
  cross-contaminates those globals and is **not supported** — expect invalid
  handles or faults.
- **Sequential** multi-engine use **is** supported: build an `Engine`, use it,
  drop it (its context tears down and the module caches are cleared), then build
  another. This is the right pattern for "reset" or per-job isolation.
- Practical guidance: use **one long-lived `Engine` per process**, and serialize
  queries through it. Multi-GPU / multi-context concurrency (per-context pools,
  streams, and caches) is post-1.0 work.

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
- **`NOT IN` / `IN` with NULL — three-valued logic (now strict for the
  subquery path).** SQL three-valued logic around `NOT IN (... NULL ...)` is a
  classic correctness foot-gun in GPU engines (a `NULL` in the set makes the
  predicate `UNKNOWN` for non-matching rows, so no row passes). The
  subquery-membership lowering (`build_in_predicate`) now matches strict SQL:
  - `expr NOT IN (set)` where the set contains a `NULL` folds to `Bool(false)`
    (no row passes), because every row is `UNKNOWN`/`FALSE`;
  - a `NULL` *probe* (`expr` itself is `NULL`) is excluded from `NOT IN` via an
    explicit `expr IS NOT NULL` guard ANDed onto the lowered `<>` chain — the
    raw GPU `<>` comparator would otherwise read the NULL probe as its stored
    value and wrongly include it;
  - `NULL` elements of a non-negated `IN` set are dropped (they can only
    contribute `UNKNOWN`), and an empty / NULL-only set folds to `IN` → `false`,
    `NOT IN` → `true`.

  Caveat on scope: this strict handling lives in the **`IN`/`NOT IN`
  subquery** path. The literal-list path (`WHERE x IN (v1, v2, …)`) desugars to
  a plain `=`/`<>` comparison chain and relies on three-valued evaluation of
  those comparators rather than the explicit set-NULL fold above; verify
  behavior against your reference engine if you embed a literal `NULL` directly
  in an `IN`/`NOT IN` value list.
- **Grouped integer `SUM` overflow may go undetected for streaming inputs.**
  Scalar and grouped integer `SUM` overflow is normally a hard error
  (`BoltError::Type("SUM(integer) overflow")`; see
  [`docs/SQL_REFERENCE.md`](SQL_REFERENCE.md)). For the **grouped** case the
  overflow is currently detected via a **host-side recompute** of the per-group
  sums, so an overflow may **not** be caught for streaming inputs the host
  cannot replicate (the device produced the result but the host has no way to
  re-derive it for the check). Tracked follow-up: an on-device overflow flag
  that makes the check independent of the host recompute. Where the host
  *can* re-fold (the host-materialized grouped path), the recompute uses
  `i64::checked_add` per group and raises the same hard error on overflow
  (`checked_group_sum` / `checked_group_sum_native_validity`,
  `src/exec/groupby.rs`).
- **Integer division by zero does not error — defined as `0` (deliberate).**
  The GPU integer-division codegen (`emit_int_div_guarded`,
  `src/jit/ptx_gen.rs`) defines `x / 0 => 0` rather than
  raising the standard-SQL division-by-zero error: the divisor is sanitised
  to a non-zero stand-in for the hardware `div` and the result is then
  `selp`-ed back to `0` when the divisor was zero. The two's-complement
  overflow corner `INT_MIN / -1` is likewise defined as a **wrapping**
  `INT_MIN` (the `(INT_MIN, -1)` pair is steered away from the trapping
  `div` and the result forced to `INT_MIN`), not an error. This is an
  intentional, test-pinned engine choice to keep the division kernel
  total/branch-free, and it diverges from standard SQL (and from DuckDB,
  which raises a divide-by-zero error). Integer **float** division is
  unaffected (IEEE `div.rn` semantics).
- **Grouped `AVG` of an empty / all-NULL group returns `0.0`, not NULL
  (deliberate).** Standard SQL says `AVG` over zero contributing rows is
  `NULL`; the engine instead returns `0.0` to keep the `AVG` output column
  non-nullable (`src/exec/aggregate.rs`; the empty-input behavior is pinned
  by `fused_avg_empty_input_returns_zero` and flagged in-code with a
  `TODO(null)`). This diverges from standard SQL and from DuckDB (both of
  which return `NULL`). Intentional for now; tracked as the `TODO(null)`
  follow-up.
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
- **Persistent PTX cache: opt-in, no automatic eviction.** The
  `EngineBuilder::persistent_cache(path)` knob **is** wired into `build()`:
  it installs the path as the process-wide disk PTX-cache override (via
  `install_persistent_cache_override`), so the JIT compile path reads and
  writes cubins at the configured directory. Precedence is builder-path →
  `BOLT_PTX_CACHE_DIR` env var → disabled; an unset builder knob does **not**
  clear an env-var- or otherwise-installed override (opt-in: a default-built
  engine never enables the disk cache on its own). The override is
  process-global, so it is shared across sequential engines in the same
  process. The cache directory itself has no size cap or eviction policy —
  you manage its lifetime. See [`ROADMAP.md`](../ROADMAP.md) "Known
  limitations" for remaining builder-surface gaps.

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
