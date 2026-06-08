# Path to 1.0

A proposal for how Craton Bolt gets to a stable, semver-bound 1.0. The current
baseline is **0.7.0**; this document is **strategic, not contractual** — version
numbers and dates are illustrative, milestone *ordering* and *acceptance
criteria* are the substance.

> **Baseline note.** This plan was first drafted against a 0.3.0 baseline. Much
> of the *Foundation* and *Coverage* work below has since landed across
> 0.5 → 0.7 (multi-batch tables, the full join family, GPU sort, the expanded
> scalar/type surface, `tracing`, the disk-backed PTX cache, the
> `KernelSpec`-keyed module cache, and the `EngineBuilder`). The "today" column
> in §2 therefore reflects the *original* 0.3.0 starting point and is kept for
> historical contrast — consult [`CHANGELOG.md`](../CHANGELOG.md) for what has
> already shipped. The milestone *ordering* and *acceptance criteria* remain the
> substance of the plan.

The existing [`ROADMAP.md`](../ROADMAP.md) is the index of where we are and what
we're skipping for the current 0.x line. This doc is the depth behind its "1.0"
section: what each milestone contains, what counts as done, and where we will
have to make real decisions before shipping the freeze.

## TL;DR

- **1.0 is a promise of API stability**, not a feature checklist. Everything
  below exists to support that promise: stop landing breaking changes, stop
  hiding the IR, start measuring regression-grade quality.
- **Eight milestones** between here and 1.0, grouped into four phases:
  *Foundation* (multi-batch + async + validity), *Coverage* (SQL surface, joins,
  types), *Quality* (observability, performance discipline), *Freeze* (API
  stabilization, RC, audit).
- **1.0 acceptance is measurable**, not aesthetic: regression-CI green, ClickBench
  parity-or-better vs DuckDB on the queries we support, all `#[doc(hidden)]`
  IR types either promoted or replaced.
- **Several real decisions remain** — see §11. None of them can be deferred past
  the API freeze.

---

## 1. What 1.0 means

Three concrete promises:

1. **Semver compliance.** Public API does not break between 1.x.y releases.
   Breaking changes wait for 2.0. The "public API" is enumerated explicitly in
   the freeze (§9).
2. **Production-acceptable defaults.** Out of the box, on supported hardware,
   running supported SQL on in-spec data, the engine returns correct answers,
   doesn't leak GPU memory, and doesn't crash the host on OOM.
3. **Documented limits.** Where Craton Bolt doesn't fit a workload, the limit is
   stated up front (in `SQL_REFERENCE.md`, `ROADMAP.md`, or the error message
   itself), not discovered at runtime.

What 1.0 explicitly **does not** promise (§10):

- Performance leadership across all workloads.
- Coverage of every SQL feature in the standard.
- Distributed / multi-node operation.
- Wire-protocol compatibility (PostgreSQL, MySQL, Arrow Flight SQL).
- ABI stability for the JIT'd kernels (cache layout may change).

If readers infer any of those from "1.0", we'll get angry GitHub issues
forever. Headlining the exclusions matters as much as headlining the
inclusions.

## 2. The gap, at a glance

Where the 0.3.0 baseline sat and what stands between the project and 1.0. The
"today" column is the **original 0.3.0** starting point preserved for contrast;
many of these rows have since closed across 0.5 → 0.7 (see
[`CHANGELOG.md`](../CHANGELOG.md)):

| Axis              | 0.3.0 baseline                                | 1.0 target                                              |
| ----------------- | --------------------------------------------- | ------------------------------------------------------- |
| API stability     | All IR types `#[doc(hidden)]`. Frequent breaks | Public surface enumerated; semver-bound                 |
| Table model       | Multi-batch in-memory tables; no streaming / spill | Multi-batch streaming, larger-than-VRAM via spill       |
| SQL scalars       | Arithmetic, comparison, bool                  | + `IS NULL`, `LIKE`, `IN`, `BETWEEN`, `CASE`, `CAST`, `COALESCE`, `||` |
| SQL aggregates    | `SUM`, `COUNT`, `AVG`, `MIN`, `MAX`           | + `STDDEV`, `VAR`, `PERCENTILE`, `MEDIAN`               |
| Joins             | `INNER` equi, one per `SELECT`                | `INNER` / `LEFT` / `RIGHT` / `FULL` / `CROSS`, multi-join per `SELECT`, non-equi |
| Subqueries / CTEs | None                                          | Uncorrelated subqueries, basic `WITH` (CTEs)            |
| Types             | Int32/64, Float32/64, Bool, Utf8 (dict)       | + Decimal128, Date32, Timestamp(ns), basic List         |
| NULL handling     | Bool / Utf8 round-trip; primitives drop nulls | Validity propagated through every kernel                |
| Error semantics   | Plain string errors                           | Structured errors with span info, suggestions           |
| Observability     | `eprintln!` in Drop paths only                | `tracing` spans per query phase; opt-in metrics         |
| JIT cache         | In-process PTX cache (FIFO, 256 entries)      | Optional disk-backed persistent cache                   |
| Performance       | One ad-hoc bench run                          | Regression-CI baseline, per-release published numbers   |
| Platform matrix   | Linux x86_64, Windows x86_64                  | + Linux aarch64 (Jetson)                                |
| Documentation     | Reference docs, no narrative user guide       | User guide, migration guide, complete `cargo doc`       |

Each row is a milestone or part of one. The next section sequences them.

## 3. The milestone plan

Eight milestones, four phases. Names are mnemonic; version numbers are
suggested.

### Phase A — Foundation (0.4)

#### M1: Streaming tables + async memcpy + full validity

The current in-memory multi-batch model bottoms out once a table no longer fits
in VRAM. M1 unblocks every later milestone.

**Goal**: support tables that don't fit in one batch, transfer data without
stalling, and propagate Arrow validity through every kernel.

**Deliverables**
- Stable `Engine::register_table_stream(name, schema, iterator)` API. Engine
  consumes batches lazily.
- Async H2D / D2H using the already-bound `cuMemcpyHtoDAsync_v2` /
  `cuMemcpyDtoHAsync_v2` + pinned host buffers.
- Validity bitmaps propagated through filter, primitive aggregate, GROUP BY,
  and sort kernels. `COUNT(expr)` honours nulls everywhere, not only on the
  Bool/Utf8 path.
- Larger-than-VRAM tables via spill (out-of-core to host pinned memory; not
  to disk yet).

**Acceptance**
- ClickBench `hits` at 100M rows runs to completion on a 16 GB-VRAM card.
- `cargo test --features cuda --ignored` exercises a 10-batch streaming
  table.
- `COUNT(col)` on a column with 30% nulls returns the same answer as DuckDB.

**Why for 1.0**: any production user hits the in-memory-table ceiling the first
time their dataset exceeds VRAM. Cannot ship 1.0 without this.

---

### Phase B — Coverage (0.5, 0.6, 0.7)

Three milestones, each delivering a chunk of the SQL surface. Order is by
dependency: scalars feed joins (predicate evaluation); types feed everything.

#### M2: SQL scalar completeness (0.5)

**Goal**: every common scalar expression a user might write parses and runs.

**Deliverables**
- `IS NULL` / `IS NOT NULL` — needs validity propagation from M1.
- `IN (...)` — small lists inlined as OR-chain; large lists as a hash probe.
- `BETWEEN x AND y` — desugared to `>= AND <=`.
- `CASE WHEN ... THEN ... ELSE ... END` — compiles to predicated SELECT chain.
- `CAST(x AS T)` — supported (T, U) pairs explicit; unsupported pairs error
  with a clear message.
- `COALESCE`, `NULLIF` — desugared to `CASE`.
- `LIKE` with constant pattern — compiled to character-class kernel (no
  regex). Wildcards `%` and `_` only.
- String concat `||` — fixed-width path for known-bounded results; dictionary
  re-encode for general case.
- `NOT` as a unary op in the AST.
- Aggregates: `STDDEV_POP`, `STDDEV_SAMP`, `VAR_POP`, `VAR_SAMP` (Welford on
  device).

**Acceptance**
- Every scalar in the table above has a parser test + e2e test + at least one
  microbench.
- ClickBench queries that *only* failed due to missing scalars now run.
- Geometric-mean ClickBench result within 2× of DuckDB warm-cache on a
  contemporary GPU (A100-class or better).

**Why for 1.0**: shipping without `IS NULL` / `CASE` / `CAST` would be
embarrassing. These are table stakes.

#### M3: Join expansion + GPU sort (0.6)

**Goal**: the join story stops being a 0.x asterisk.

**Deliverables**
- `LEFT JOIN`, `RIGHT JOIN`, `FULL OUTER JOIN`, `CROSS JOIN`.
- Non-equi join via nested-loop kernel (small inner side only; warned if
  cardinality > threshold).
- Multiple joins per `SELECT` (the 0.3.0 parser rejects this).
- GPU hash-join kernel replacing the host-side build/probe from wave 8.
- GPU sort kernel (radix or merge) backing `ORDER BY` and the dedup step of
  `UNION` / `DISTINCT`, eliminating the host round-trip.

**Acceptance**
- All SSB queries (joined form) run.
- TPC-H queries 1, 3, 5, 6, 10, 12 run to completion (the join-heavy subset
  most likely to expose bugs).
- GPU sort beats `[sort: host round-trip via Rayon]` on a 10M-row workload.

**Why for 1.0**: a SQL engine without proper joins isn't a SQL engine. The GPU
sort unblocks ORDER BY as a first-class operator instead of the apologetic
host-side fallback.

#### M4: Typesystem expansion (0.7)

**Goal**: real analytic workloads stop hitting "unsupported type" errors.

**Deliverables**
- `Decimal128(p, s)` with the four arithmetic ops and aggregates.
- `Date32`, `Timestamp(ns, tz?)` with `DATE_TRUNC`, `EXTRACT`, comparison.
- Basic `List<T>` — load, project, `array_length`, `unnest` if it fits.
- Implicit conversion rules documented (`SQL_REFERENCE.md`), explicit `CAST`
  for everything ambiguous.

**Acceptance**
- A ClickBench-shaped workload using `DateTime` and Decimal columns runs to
  completion.
- All `CAST` matrices have tests, including the `Err` cases.

**Why for 1.0**: Decimal is non-negotiable for finance/analytics. Date/Time is
non-negotiable for anything time-series.

---

### Phase C — Quality (0.8, 0.9)

Coverage doesn't ship without quality discipline behind it.

#### M5: Observability + ergonomics (0.8)

**Goal**: when something goes wrong, the user knows what and why without
attaching a debugger.

**Deliverables**
- `tracing` spans for every query phase: parse, plan, lower, codegen,
  PTX-load, launch, transfer, materialize. Off by default, opt-in via env or
  config.
- Structured errors: `BoltError::Parse { msg, span }`, similar for plan /
  lower / runtime. Errors carry source spans where applicable.
- "Did you mean...?" suggestions for unknown columns / tables (cheap
  edit-distance).
- A small `metrics` integration (counters + histograms) behind a feature flag.
- A user guide (`docs/USER_GUIDE.md`): "your first query," "registering data,"
  "what to do when something fails," "performance tuning."

**Acceptance**
- Every error type in `BoltError` documented with an example and the
  recovery path.
- The user guide walks a reader from `cargo add craton-bolt` to a working
  GROUP BY in under 10 minutes.

**Why for 1.0**: API stability without ergonomics is a stable bad experience.

#### M6: Performance discipline (0.9)

**Goal**: never ship a regression silently.

**Deliverables**
- Criterion-based regression CI on a GPU runner. Fail the build on a > 5%
  regression in any tracked metric.
- Published per-release benchmark numbers (ClickBench + microbench), stored
  in `docs/perf/<version>.md` and indexed in `BENCHMARKS.md`.
- `KernelSpec`-keyed cache (the existing PTX cache is keyed on emitted text;
  this skips codegen entirely on a hit).
- Optional disk-backed PTX cache (`~/.cache/craton-bolt/ptx/<hash>.cubin`),
  controlled by `Engine::Builder::persistent_cache(path)`.

**Acceptance**
- Last 4 releases have benchmark numbers published.
- A deliberately regressed PR fails CI on the perf gate.
- Warm-cache JIT cost is < 50 µs end-to-end on a representative query.

**Why for 1.0**: without measured regressions, "1.0 stable" means "stable at
whatever it happens to be when we tag it."

---

### Phase D — Freeze (0.10, 0.11, 1.0)

#### M7: API stabilization preview (0.10)

**Goal**: enumerate the public surface; force the breaking changes now, not
during the RC.

**Deliverables**
- For every `#[doc(hidden)]` type, a decision: **promote** (becomes public,
  documented, semver-bound) or **encapsulate** (replaced by a public builder
  /method that doesn't leak the internal shape).
- `Engine::Builder` (replacing the current `Engine::new` / `new_with_device`
  pair) with explicit knobs for device, memory budget, cache, tracing.
- `DataFrame::collect()` becomes a real materializing terminal (the 0.x
  tombstone is removed).
- `Reg`, `Op`, `Value`, `KernelSpec`, `AggregateSpec`, `ColumnIO`,
  `PhysicalPlan`, `LogicalPlan` — decision and implementation per the
  promote/encapsulate matrix above.
- Optimizer extension points: a `PlanRewrite` trait users can register for
  custom rewrites (e.g. predicate pushdown into a custom table provider).
- Public API enumerated in a new `docs/API_SURFACE.md` that lists every
  exported symbol, its stability tier (stable / experimental / hidden), and
  its semver contract.

**Acceptance**
- `cargo public-api` (or equivalent) output is the explicit, reviewed,
  signed-off surface.
- Every symbol marked "stable" has rustdoc with example.
- The `1.0 — public API freeze` section of `ROADMAP.md` is the diff between
  this surface and 1.0, and that diff is *empty*.

**Why for 1.0**: this is where 1.0 actually lives. Everything else is
prerequisite.

#### M8: RC, audit, freeze (0.11, 1.0)

**Goal**: ship.

**Deliverables**
- 0.11-rc.N releases until two consecutive RCs land with no API-affecting
  changes.
- External security review (focused on the FFI surface — `cuda_sys.rs`,
  buffer ownership, PTX cache poisoning).
- Soundness audit of `GpuView` / `GpuViewMut` / `GpuVec` against the latest
  Rust soundness lints (esp. Stacked Borrows / Tree Borrows under Miri where
  practicable).
- `aarch64-linux` (Jetson) added to the CI matrix.
- Migration guide for 0.x users (which are mostly internal at this point, but
  the discipline of writing one forces honest API review).
- Tag 1.0.

**Acceptance**
- Two consecutive RCs land with no API-affecting changes (`cargo public-api`
  diff empty).
- Security review report has no unresolved high/critical findings.
- All three CI platforms green.

---

## 4. 1.0 acceptance criteria (top-level)

A single checklist. If any item is unchecked, we are not at 1.0.

- [ ] All M1-M7 deliverables shipped.
- [ ] `cargo public-api` diff vs 0.11 final is empty.
- [ ] Last RC ran clean on all three CI platforms (Linux x86_64, Windows
      x86_64, Linux aarch64).
- [ ] ClickBench result published; geometric mean within 2× of DuckDB on the
      queries we support; no Craton Bolt query is > 10× slower than DuckDB on
      that suite.
- [ ] No `#[doc(hidden)]` items in the public crate root.
- [ ] Every public item has rustdoc with at least one example.
- [ ] `SECURITY.md` policy actively monitored; no unresolved high-severity
      reports.
- [ ] User guide complete; migration guide from 0.x shipped.
- [ ] No `unimplemented!()`, `todo!()`, or `panic!("…")` in any code path
      reachable from supported SQL (greps clean; CI gates on this).

## 5. What 1.0 is NOT

Explicit exclusions, to be cited at the top of any "Craton Bolt 1.0 release" post:

- **Not a wire-protocol-compatible drop-in for PostgreSQL/MySQL.** Embed via
  the Rust crate. Network exposure is post-1.0.
- **Not a distributed engine.** Single node, single process.
- **Not feature-complete vs the SQL standard.** Window functions, recursive
  CTEs, full subquery support, MERGE, etc. are post-1.0.
- **Not a CPU fallback engine.** If a query uses a feature Craton Bolt doesn't
  GPU-support, it errors — it does not silently fall back to a host pipeline.
- **Not a transactional store.** Read-only at 1.0; no `INSERT` / `UPDATE` /
  `DELETE`. Data is registered, not mutated.
- **Not a multi-tenant engine.** One `Engine` is one user's session.

The point of the list isn't to limit ambition; it's to be ruthless about what
*has to* ship in 1.0 vs what *can* ship in 1.x or 2.0.

## 6. What goes in 1.x (post-1.0, pre-2.0)

Acceptable to defer because they don't require breaking changes to add:

- Wire protocol(s): Arrow Flight SQL first, then maybe a PostgreSQL frontend.
- Window functions.
- Recursive CTEs.
- More aggregates: `APPROX_COUNT_DISTINCT` (HyperLogLog on device),
  `APPROX_QUANTILE`.
- Sliding-window operators for time-series.
- Multi-GPU within a single Engine (currently one Engine per GPU).
- User-defined functions via JIT'd CUDA C++ snippets.
- Vectorized UDFs in Rust via a stable kernel-template API.

If any of these *would* require an API break to add cleanly, they get pulled
into M7 instead.

## 7. What goes in 2.0+

Items that are likely to require breaking changes:

- Distributed execution.
- Transactional writes (would require a totally different memory model).
- Multi-statement transactions.
- Pluggable storage layer (Iceberg, Delta, custom).

These are flagged so M7's API design accounts for them — for instance, the
`Engine::Builder` shape should leave room for a `cluster_endpoint(...)` knob
even if it does nothing at 1.0.

## 8. Open decisions

These are real forks where no obvious answer exists. They cannot remain open
through M7.

### 8.1 Plan IR exposure

**Choice**: promote `LogicalPlan` and `PhysicalPlan` to public, or hide them
behind a `DataFrame`-only API.

**Trade-off**:
- Promote: power users can construct plans directly, build alternate frontends
  (e.g. Substrait → Craton Bolt). Locks us into the current shape.
- Hide: free to refactor the IR forever. Locks out plan-level integrations.

**Recommendation**: hide behind `DataFrame`, but expose a stable `Substrait`
ingestion path. Lets us refactor the IR without breaking integrations.

### 8.2 Persistent cache as default or opt-in

**Choice**: enable disk-backed PTX cache by default in 1.0?

**Trade-off**:
- Default-on: best UX (cold start is fast after first run). Risk: cache
  poisoning, stale cache after driver upgrades.
- Opt-in: safer, but most users will never enable it.

**Recommendation**: opt-in via `Engine::Builder::persistent_cache(path)`. Ship
default-on in 1.1 once we have a year of field telemetry on the invalidation
heuristics.

### 8.3 CUDA-version floor

**Choice**: minimum supported CUDA toolkit version at 1.0?

**Trade-off**:
- 11.x floor: widest hardware support. Misses ~3 years of driver-side
  improvements.
- 12.x floor: cleaner code paths, better async memcpy semantics. Excludes a
  meaningful fraction of currently-deployed cards.

**Recommendation**: 12.0 floor. Document explicitly. Older cards can stay on
0.x.

### 8.4 Error type granularity

**Choice**: single `BoltError` enum (current direction) vs typed errors per
phase (`ParseError`, `PlanError`, `RuntimeError`).

**Trade-off**:
- Single enum: easier `?`-propagation, simpler match arms. Loss of compile-time
  guarantees about which errors can come from which phase.
- Typed errors: precise contracts, but `?` requires `From` impls or a
  consolidated public wrapper anyway.

**Recommendation**: single `BoltError` with structured variants. Add
phase-specific error subtypes only if M5 user feedback demands it.

### 8.5 Async API

**Choice**: does the public `Engine::execute` become `async fn`?

**Trade-off**:
- Sync only: simpler, matches the current shape, no Tokio coupling.
- Async: composes with web servers / Flight SQL. Forces a runtime choice.

**Recommendation**: keep `execute` sync, add `execute_async` later (post-1.0)
that returns `impl Future`. The blocking implementation runs on a thread.

## 9. The API freeze in concrete terms

What "frozen" means, operationally, at 1.0:

- **Public crate root**: nothing added or removed without a 2.0.
- **Public re-exports**: same.
- **Public types**: field additions allowed only for `#[non_exhaustive]`-marked
  types. Pre-mark anything we expect to grow.
- **Function signatures**: parameter changes are breaking. Add new functions
  instead.
- **Trait methods**: cannot add required methods. Default methods are fine.
- **Error variants**: `BoltError` is `#[non_exhaustive]` from 1.0; adding
  variants is non-breaking, renaming is breaking.
- **Feature flags**: existing flags don't change meaning. New flags are fine.
- **MSRV**: bumping is breaking *in cargo's view*; flag it loudly in the
  changelog but it's not a 2.0 by itself.

The `docs/API_SURFACE.md` document from M7 enumerates this list per symbol;
1.0 lives or dies by it.

## 10. Risks and mitigations

| Risk                                                        | Mitigation                                                           |
| ----------------------------------------------------------- | -------------------------------------------------------------------- |
| API design mistake found post-freeze                        | M7 lasts as long as needed; no calendar pressure on the freeze.      |
| GPU vendor landscape shifts (AMD/Intel demand)              | Keep CUDA-specific code behind a `gpu::cuda` module from M5+; an HSA/HIP backend is a 2.0 effort, not a 1.0 blocker. |
| Performance regression below DuckDB on too many queries     | M6 perf-CI catches this per-PR; 1.0 acceptance criterion makes it a release-blocker. |
| Correctness divergence from DuckDB on edge cases (NULL, NaN, overflow) | M2 + M4 acceptance criteria include parity-vs-DuckDB tests; differences documented in `SQL_REFERENCE.md`. |
| FFI soundness regression                                    | Annual Miri sweep where practicable; security audit in M8 covers what Miri can't. |
| User adoption forces premature breaking changes pre-1.0     | Communicate "0.x is unstable" loudly; add release notes per minor.   |
| Maintainer bandwidth                                        | Phase A and Phase D are the hard parts; B and C parallelize across contributors. |

## 11. How we'll know we're done

A short list of external signals that 1.0 has actually landed, vs the
declaration of "1.0" being marketing:

- A third party ships an integration using only the public API and *does not
  reach into `#[doc(hidden)]`*.
- An issue filed against "Craton Bolt produces wrong answer for X" is the
  exception, not the norm.
- A "Craton Bolt vs DuckDB" benchmark from someone other than us appears, and we
  don't have to apologize for the result.
- A user's first 10 minutes with Craton Bolt (registration → first query →
  inspect result) happens without reading source.

## 12. See also

- [`ROADMAP.md`](../ROADMAP.md) — the high-level index this doc extends.
- [`COMPETITIVE_BENCHMARKING.md`](./COMPETITIVE_BENCHMARKING.md) — the
  methodology M6 will use to publish per-release numbers.
- [`SQL_REFERENCE.md`](./SQL_REFERENCE.md) — what works today; the deltas
  drive M2 and M4.
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — what the IR types in M7 actually
  are, and why they're currently hidden.
- [`CHANGELOG.md`](../CHANGELOG.md) — the record of how we actually got from
  one milestone to the next.
