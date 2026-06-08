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
  unaffected (IEEE `div.rn` semantics). **`Decimal128` division** (the
  0.7 GPU `Op::Div128` path) follows the same convention: a zero divisor
  yields a deterministic **`0`** quotient for that lane (non-trapping)
  rather than the standard-SQL error.
- **Grouped `AVG` of an empty / all-NULL group returns `0.0`, not NULL
  (deliberate).** Standard SQL says `AVG` over zero contributing rows is
  `NULL`; the engine instead returns `0.0` to keep the `AVG` output column
  non-nullable (`src/exec/aggregate.rs`; the empty-input behavior is pinned
  by `fused_avg_empty_input_returns_zero` and flagged in-code with a
  `TODO(null)`). This diverges from standard SQL and from DuckDB (both of
  which return `NULL`). Intentional for now; tracked as the `TODO(null)`
  follow-up.
- **GPU string device path is host-validated only (opt-in, off by default).**
  The GPU string device kernels — the non-dictionary `LIKE` matcher
  (`StringLikeFilter` / `compile_like_match_kernel`) and the `UPPER` / `LOWER`
  / `CONCAT` / `SUBSTRING` / `TRIM` two-pass `StringProject` producers — are
  implemented and PTX-shape-tested but have **never been executed on GPU
  hardware** as of v0.7.0 (CI builds with no CUDA device). They are therefore
  **HOST-VALIDATED ONLY** and **not enabled by default**: the byte-identical
  **host** path is the correctness path and is selected by default, and the
  device kernels are reached only when the opt-in `BOLT_GPU_STRING` env var is
  set (default OFF — see [`docs/ENV_VARS.md`](ENV_VARS.md)). Dictionary
  `Utf8` `LIKE` / equality / ordering predicates are unaffected: they fold to
  pure-integer index-membership predicates that run on the GPU and are not part
  of this string-device gate. (`LENGTH` likewise rides the integer-output
  `StringLength` path, not a string producer.)
- **String handling is dictionary/ASCII-oriented.** String predicates operate
  over dictionary-encoded literals, and the GPU case-folding functions
  (`UPPER` / `LOWER`) are byte/ASCII-oriented — treat non-ASCII / multi-byte
  UTF-8 case-folding as unverified, since case conversion is ASCII-range only
  (any dictionary entry with a non-ASCII byte falls back to the host
  transform). **`LENGTH` counts characters** (Unicode scalar values —
  `s.chars().count()`, `src/exec/string_length.rs`), **not bytes**; the
  byte-length function is **`OCTET_LENGTH`** (UTF-8 byte count,
  `src/exec/expr_agg.rs`). So for a multi-byte string the two diverge:
  `LENGTH('café') = 4` while `OCTET_LENGTH('café') = 5`.
  Utf8 ordering comparisons against a literal (`WHERE name < 'M'`) use
  **binary (UTF-8 byte) collation**, not locale-aware / ICU collation, so the
  ordering matches `memcmp` rather than any natural-language collation.
  Ordering of *two* Utf8 columns (`a < b`) now folds to a **GPU** rank
  comparison (finding F12): both columns are ranked against the de-duplicated
  union of their dictionaries under binary collation, and the predicate becomes
  a NULL-safe integer compare over the materialised per-row rank columns. A NULL
  on either side fails the `>= 0` guard, so the row is dropped (correct SQL 3VL
  projection of NULL → no row passes). Shapes the rank path can't cover (a
  non-dict or protected column) still fall back to a host string comparison.
- **Temporal / decimal types: lowered, with residual gaps.** `Decimal128`,
  `Date32`, and `Timestamp` parse and lower to the GPU. GPU gather (filter /
  compaction) and column upload are wired for all three. As of the 0.7 wave:
  `COUNT`, **`MIN`, and `MAX`** over a `Date32` / `Timestamp` column work
  end-to-end (GPU reduction on the i32 / i64 storage; the result is rebuilt
  preserving the date type, or the timestamp **unit + timezone** — scalar and
  GROUP BY alike). `Decimal128` `+`, `-`, `*`, **`/`** (including mixed
  Decimal/integer arithmetic), scale-aligned comparisons, and integer↔decimal
  / decimal-rescale **CAST** lower to the GPU. **Scalar** `SUM` / `MIN` / `MAX`
  over `Decimal128` now run on the **GPU** (dedicated i128 block-reduce kernels,
  with a host-fold fallback inside). A `CASE` whose result dtype is `Decimal128`
  now lowers to the **GPU** (a pair of `selp.b64` via `Op::Select128`). **Plain**
  `CAST` between **Float and `Decimal128`** now lowers to the **GPU**
  (`F64ToI128` / `I128ToF64`). **Grouped** `Decimal128` `SUM` / `MIN` / `MAX`
  now also run on the **GPU** (`bolt_groupby_agg_decimal`, a per-slot spin-locked
  128-bit accumulator; `SUM` raises the same overflow error as the scalar path,
  `MIN` / `MAX` preserve the input `(p, s)`). **Residual gaps:** `SUM` over a
  temporal column is rejected by design (undefined SQL);
  `TRY_CAST` / `SAFE_CAST` of Float⇄`Decimal128`
  is rejected at type-check (the host evaluator has no `Decimal128` column); and
  `CAST` to/from `Timestamp` / `String` is still rejected at the GPU lowering
  boundary. See the type tiers in
  [`docs/SQL_REFERENCE.md`](SQL_REFERENCE.md).
- **Env-gated experimental paths.** Several kernels (e.g. GPU radix sort,
  CUDA-graph sort, alternate hash/scan algorithms) are gated behind
  environment variables and are not the default path. See
  [`docs/ENV_VARS.md`](ENV_VARS.md).
- **Persistent PTX cache: opt-in, bounded by an LRU eviction policy.** The
  `EngineBuilder::persistent_cache(path)` knob **is** wired into `build()`:
  it installs the path as the process-wide disk PTX-cache override (via
  `install_persistent_cache_override`), so the JIT compile path reads and
  writes cubins at the configured directory. Precedence is builder-path →
  `BOLT_PTX_CACHE_DIR` env var → disabled; an unset builder knob does **not**
  clear an env-var- or otherwise-installed override (opt-in: a default-built
  engine never enables the disk cache on its own). The override is
  process-global, so it is shared across sequential engines in the same
  process. The cache directory **is** bounded: `enforce_bounds`
  (`src/jit/disk_cache.rs`) runs after each store and evicts least-recently-
  modified `*.ptx` entries (LRU by mtime) once it exceeds either the total-
  bytes cap (`CRATON_BOLT_PTX_CACHE_MAX_BYTES`, default 64 MiB) or the
  entry-count cap (`CRATON_BOLT_PTX_CACHE_MAX_ENTRIES`, default 4096); setting
  either var to `0` disables that one cap. See
  [`ENV_VARS.md`](ENV_VARS.md) for the knobs and [`ROADMAP.md`](../ROADMAP.md)
  "Known limitations" for remaining builder-surface gaps.

---

## Rejected SQL constructs

The frontend **cleanly rejects** the following constructs with a
`BoltError::Sql(...)` (a clear parse/plan-time error), **not** a crash or a
silent wrong answer. If you depend on any of these, expect an error and
rewrite the query (see [`docs/SQL_REFERENCE.md`](SQL_REFERENCE.md) for the
supported alternatives):

- **Correlated scalar / `EXISTS` / `NOT EXISTS` subqueries in `SELECT`.**
  (A **single** correlated scalar / `EXISTS` / `NOT EXISTS` subquery in a
  top-level `WHERE` conjunct is now **supported**, executed as a host
  nested-loop apply — see below. Residual rejections: **more than one**
  correlated subquery in `WHERE`, correlation **inside an `OR`**, and a
  **correlated `IN (SELECT ...)`** are still rejected. Non-lateral **derived
  tables** `(SELECT ...) AS alias` are **supported** — alias required;
  **`LATERAL` derived tables** are now **supported** too; column-list aliases
  `AS d(x, y)` remain rejected.)
- **Non-default window frames** (custom bounds, `GROUPS`, anything other than
  the default `RANGE UNBOUNDED PRECEDING AND CURRENT ROW`). (`QUALIFY`, the
  named `WINDOW` clause, and `OVER <named_window>` are now **supported** — see
  below.)
- **Table-valued functions** *except* `generate_series(start, stop[, step])`,
  which is now **supported** in `FROM` — see below. Other TVFs (`unnest`, etc.)
  remain rejected: they would need a function-as-table-source mechanism, and
  the engine scans only registered tables / CTEs / derived tables.
- **`CONNECT BY`**, **`CLUSTER` / `DISTRIBUTE` / `SORT BY`**, **`INTO`**.
- **`GLOBAL JOIN`** (ClickHouse extension).
- **`SUM(Date32 / Timestamp)`** — undefined in SQL; rejected by design.

The following are now **supported** (they were previously listed here as
rejected); see [`docs/SQL_REFERENCE.md`](SQL_REFERENCE.md) for the per-feature
detail and residual sub-rejections:

- **`GROUP BY ROLLUP` / `CUBE` / `GROUPING SETS` / `ALL`**, the trailing
  `WITH TOTALS` / `WITH ROLLUP` / `WITH CUBE` modifiers, and the
  `GROUPING()` / `GROUPING_ID()` indicators. Expanded host-side to a UNION-ALL
  of the per-set plans (max **12** grouping columns; `GROUPING()` without a
  grouping construct stays rejected).
- **`WITH RECURSIVE`** — linear, non-linear (self-join, naive evaluation), and
  mutual (multi-CTE lockstep) recursion, with `UNION` / `UNION ALL`, an optional
  recursive-CTE column-list alias, and an iteration cap
  (`CRATON_MAX_RECURSIVE_ITERATIONS`, default 1000). Residual rejections: a
  recursive anchor that seeds from a recursive member, a self-reference buried
  in a subquery, and `UNION BY NAME`.
- **`TRY_CAST` / `SAFE_CAST`** (NULL-on-failure, host-evaluated) and
  **`CAST(... FORMAT '<pattern>')`** (bounded temporal⇄string pattern
  vocabulary, host-evaluated). `TRY_CAST` of Float⇄`Decimal128` is rejected at
  type-check; unknown `FORMAT` tokens are rejected.
- **Multi-table `FROM a, b [WHERE ...]`** — the comma list desugars to a
  left-deep chain of `CROSS JOIN`s.
- **A single correlated scalar / `EXISTS` / `NOT EXISTS` subquery in `WHERE`** —
  detected at plan time and executed as a host nested-loop apply (semi-join for
  `EXISTS`, anti-join for `NOT EXISTS`, per-row scalar test otherwise) over a
  row-capped outer relation. Residuals: more than one correlated subquery in
  `WHERE`, correlation inside `OR`, and correlated `IN (SELECT ...)`.
- **`QUALIFY`**, the named **`WINDOW`** clause (`WINDOW w AS (...)`), and
  **`OVER w`** named-window references — lowered onto the host-side window
  executor. Residual: `QUALIFY` combined with `GROUP BY` / aggregates.
- **`FETCH FIRST n ROWS` / T-SQL `TOP n`** — both fold into `LIMIT`.
  **`FOR UPDATE` / `FOR SHARE`** are accepted as a no-op (read-only OLAP engine).
  **`PREWHERE`** (ClickHouse early filter) is folded into `WHERE`. Residuals:
  `FETCH` / `TOP` with `PERCENT` or `WITH TIES`, a `TOP` combined with
  `LIMIT` / `FETCH` (ambiguous), and `QUALIFY` combined with `GROUP BY`.
- **`VALUES` as a row source** — both the bare `VALUES (...), (...)` form and
  `FROM (VALUES ...) AS t(a, b)`. Common-type inference is limited to the
  numeric widenings (`Int32`↔`Int64`, `Int`↔`Float`, `Float32`↔`Float64`);
  other mixed-type columns require an exact match. Row count is capped
  (`CRATON_VALUES_MAX_ROWS`, default 1,000,000).
- **`DISTINCT ON (...)`** (Postgres extension) — host-orchestrated; keeps the
  first row per key, deterministic when the query's leading `ORDER BY` keys
  match the `DISTINCT ON` keys. Keys must be simple column references; rejected
  alongside `GROUP BY` / `HAVING`.
- **`generate_series(start, stop[, step])`** in `FROM` — an inclusive integer
  (`Int64`) series; `step` defaults to `1`, a negative `step` descends, and
  `step = 0` is an error. Row count is capped
  (`CRATON_GENERATE_SERIES_MAX_ROWS`). This is the one supported
  table-valued function.
- **Grouped `Decimal128` `SUM` / `MIN` / `MAX`** now run on the **GPU** (see the
  temporal / decimal note above), no longer host-side.

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
