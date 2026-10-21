<!-- SPDX-License-Identifier: Apache-2.0 -->
<!--
applies-to: main
owner: execution
version-neutral: the crate version and MSRV live in Cargo.toml; do not pin them here
-->
# Craton Bolt — Limitations

**Crate:** `craton-bolt` (version: see `Cargo.toml` / [`CHANGELOG.md`](../CHANGELOG.md)) · **License:** Apache-2.0

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
developed research/engineering engine. The public API is unstable, and several
SQL features intentionally use host or hybrid execution even though real-GPU
paths are now a blocking CI gate.
Use it for evaluation, experimentation, and benchmarking — not for systems
where correctness,
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
  and using **two `Engine`s at the same time** is unsupported.
  `EngineBuilder::build` enforces this atomically and rejects the second live
  engine before creating another context, so the unsupported shape cannot
  cross-contaminate handles or fault the driver.
- **Sequential** multi-engine use **is** supported: build an `Engine`, use it,
  drop it (its context tears down and the module caches are cleared), then build
  another. This is the right pattern for "reset" or per-job isolation.
- Practical guidance: use **one long-lived `Engine` per process**. To serve
  many threads or clients, wrap it in a `Database` and hand out `Session`s:
  statements are queued and executed **one at a time** in arrival order on
  the shared runtime (no concurrent kernel execution across queries, no
  per-session resource limits beyond the queue capacity). The Flight SQL
  server can front a `Database` (`FlightSqlServer::from_database`); it then
  serves queries only (a read-only session). Multi-GPU /
  multi-context concurrency (per-context pools, streams, and caches) is
  post-1.0 work.

---

## Pre-1.0 API instability

- The crate is pre-1.0. Per [SemVer](https://semver.org/), pre-1.0 the public
  API may break on any minor (`0.x`) bump. The IR, `KernelSpec`, executor
  surface, and several `Engine` methods are still moving.
- See [`docs/API_SURFACE.md`](API_SURFACE.md) for the per-item stability tiers
  and [`docs/MIGRATION_GUIDE.md`](MIGRATION_GUIDE.md) for breaking-change
  deltas across releases.

---

## GPU verification boundary

- Hosted jobs use `cuda-stub` for planning, host logic, and PTX shape.
- The canonical self-hosted GPU job is blocking. It runs every ignored CUDA
  test serially (`BOLT_BENCH_GPU=1`, `--test-threads=1`) and then runs the
  split DuckDB/proptest conformance shard.
- A missing, cancelled, or failing canonical GPU runner therefore prevents a
  green workflow; host-only evidence is never presented as device evidence.

---

## Query pipeline and extension hooks

- **One compile pipeline.** `Engine::sql`, `Engine::explain_sql`,
  `Engine::run_logical_plan` / `DataFrame::collect`, and every component
  subplan of a host-orchestrated SQL shape run the same sequence: dictionary
  rewrite → built-in optimizer → `PlanRewrite` hooks → subquery resolution →
  lowering. The exact plans a hook receives on each path are documented on
  `PlanRewrite`; the notable consequence is that host-orchestrated shapes
  (recursive CTE, LATERAL, correlated `WHERE`, `VALUES`, `generate_series`,
  `DISTINCT ON`, `COUNT(DISTINCT)` + `GROUP BY`) hand hooks their component
  subplans — once per compilation, e.g. once per recursive iteration —
  rather than one whole-statement plan.
- **`EXPLAIN` does not execute subqueries.** When a plan carries uncorrelated
  scalar / `IN` subqueries, `explain_sql` labels its physical section
  `Physical plan (pre-subquery-resolution ...)`: at run time each subquery is
  executed and folded to a constant first, so the executed physical plan can
  differ from the rendered one in exactly that respect. Host-orchestrated
  shapes render their descriptor (component subplans as planned).
- **Host-orchestrated shapes still exist.** They are classified once by the
  frontend and share the compile pipeline, but they are host loops over
  subplans, not physical operators (a LATERAL / correlated apply is a capped
  O(left × subquery) nested loop).

## Streaming and data size

- **Lazy streams materialise on first reference.**
  `register_table_stream_lazy` registers in O(1) and invokes the producer only
  when a query (or one of its subqueries / component subplans) scans the
  table; unrelated lazy tables are never drained, and `explain_sql` never
  drains. The drained batches are validated against the declared schema
  (column count, names, plan dtypes) — a mismatch fails every query that
  references the table — and coalesced into one host batch.
- **What bounds the data set.** Tables registered in memory (eager or
  drained lazy) live in host memory as one coalesced Arrow batch. Storage
  tables (Parquet, catalog) are read per statement — only the row groups
  whose zone maps may match and only the referenced columns. Under a budget
  (`memory_budget`, `host_memory_budget`) the morsel path bounds device
  working sets for row-wise scans and distributive / algebraic aggregates
  over lazily-held tables, and `GROUP BY` / `DISTINCT` / equi-joins /
  `ORDER BY [LIMIT]` over a larger input run **out of core** through spill
  files; a row-wise scan (`SELECT ... WHERE ... [LIMIT]`) of a storage table
  larger than the host budget runs over bounded chunk groups, and so do
  `COUNT` / `SUM` / `MIN` / `MAX` / `AVG` without `GROUP BY` (per-chunk
  partials, merged). Residual limits:
  - `Engine::sql` returns one host batch; `Engine::sql_stream` yields a
    row-wise query's result chunk group by chunk group, but any other shape
    (e.g. an `ORDER BY` without `LIMIT` over a huge input) still arrives as
    one batch, and so does each out-of-core node's intermediate result
    (e.g. a grace-join output);
  - out-of-core partitioning needs the grouping / join keys to be plain
    columns of one base table (through filters, pass-through projections
    and — for `GROUP BY` — joins on which that table is preserved); other
    shapes, windows, set operations, non-equi joins, and `VARIANCE` /
    `STDDEV` without `GROUP BY` materialise their input;
  - an over-budget partition (key skew) is split again with a new hash seed
    up to three times; a partition whose rows all share one key cannot be
    split and is executed as it is.
- **Appends rebuild the dictionary union.** Each `register_batch` append
  coalesces the table and rebuilds string dictionaries over all rows, so N
  appends cost O(N × table). Bulk-load with one `register_table_stream` call
  (a single coalesce) instead.
- **Lazy tables and string dictionaries.** A lazy table gets string
  dictionaries when it is drained, like an eager table — except under a
  memory budget (morsel execution), where its string predicates are evaluated
  host-side (correct, but without the dictionary-index GPU fold).
- **Aggregates over a non-fusable input are materialised.** `GROUP BY` /
  aggregates over a union, distinct, or a filter the fused GPU kernel
  cannot evaluate (e.g. a raw `Utf8` comparison) run the input first, bind
  its rows as an ephemeral table, and aggregate that (GPU where the
  aggregate shape allows). An inner equi-join on one integer key over
  projections / scans of numeric, boolean and temporal columns stays on the
  device instead (as does a `COUNT(*)` of one, or one returned to the
  caller); joins on several keys, string or nullable keys, outer joins,
  string or decimal columns, and every other operator still pass host
  batches between operators.

## Storage, catalog and statements

- **Changes rewrite data.** `DELETE` / `UPDATE` rewrite the catalog files
  their predicate's zone maps may match (in-memory tables are rebuilt), and
  `ALTER TABLE` (other than `RENAME TO`) rewrites the whole table: there are
  no deletion vectors or in-place updates, so a change touching a few rows
  of a large file costs a rewrite of that file; `MERGE` rewrites the whole
  target table. `UPDATE ... FROM`, `DELETE ... USING`, `RETURNING` and
  `MERGE ... INSERT ROW` are not supported; column changes are refused while
  a materialized view reads the table.
- **Transactions isolate by snapshot, at table granularity.** Concurrent
  sessions' transactions each work on a private snapshot; a commit fails if
  another commit changed one of the same tables or views since it began,
  even when the rows do not overlap (no row-level conflict detection), and
  write skew across different tables is possible (use
  `IsolationLevel::Serializable` to run transactions one at a time). One
  transaction per session; `SET TRANSACTION` and `AND CHAIN` are not
  supported. `VACUUM` is refused while any transaction is open, and dropping
  a lazy stream inside one.
- **One writer per catalog.** `_bolt/LOCK` admits a single writing engine;
  a crashed writer leaves the lock file to remove by hand. Snapshots opened
  with `database_snapshot` are read-only.
- **Parquet types.** Columns of nested, binary, unsigned-64 or other
  non-engine types are skipped (`ParquetTable::skipped_columns`); narrow
  integers widen to `Int32` / `Int64`, large / view strings and string
  dictionaries read as `Utf8`. External tables are read-only. Hive-style
  `key=value` directories become partition columns typed `Int64` or `Utf8`
  only (no date / decimal partition values), and every file must use the
  same partition keys.
- **Statistics.** Chunk statistics come from Parquet footers (or are
  computed for in-memory chunks); table-level distinct estimates merged
  from chunks are lower bounds until `ANALYZE` records exact HyperLogLog
  estimates. In-memory tables have column statistics only after `ANALYZE`
  (until their next change). Statistics drive pruning and the optimizer's
  filter / join selectivity (bounds, distinct counts, null counts → join
  order). The cost-based CPU / GPU choice (`TierPolicy::CostBased`, opt-in)
  covers row-wise statements and scalar aggregates over one table whose
  expressions the host evaluator supports; joins, `GROUP BY` and windows
  always take the GPU path where it exists.
- **Materialized views.** Incremental refresh covers `SUM` / `COUNT` /
  `MIN` / `MAX` (with or without `GROUP BY`, no `HAVING`, no computed
  outputs) over one catalog table that changed only by appends; everything
  else refreshes in full. A query is answered from a view only when its plan
  equals the view's defining query exactly (no subsumption or partial
  matching) and the view is fresh. A view in a catalog may read only catalog
  tables. `on_commit` and scheduled refresh happen on the `execute_sql`
  path (and the `Database` runtime), not inside `Engine::sql`; scheduled
  views need such activity to be refreshed.
- **Temporary tables are engine-wide**, including across the `Session`s of
  one `Database`.
- **Indexes are pruning structures only.** Scans skip data by zone maps
  (min / max / null count per row group or chunk), Hive partitions,
  clustering, and Parquet bloom filters for `=` / `IN` on integer, date and
  string columns (written by the engine for columns with at least 1024
  distinct values; read from any Parquet file that has them). There are no
  B-trees or row-level lookups. Storage is the local filesystem (no
  object-store backends).

## Known performance gaps

- **High-cardinality GROUP BY over `Int64` keys or integer sums.** The
  resident Tier-1 path covers ≤ 1024 groups for `Int32` / `Int64` keys and
  `Float64` / `Int64` / `Int32` sums; the partitioned Tier-2 paths for larger
  group counts are specialised to `Int32` keys with `Float64` sums. Other
  high-cardinality shapes take the general global-atomic path, measured
  roughly 4–7× slower at 10 K groups (`benches/cost_model.rs`,
  `groupby_cardinality`).

## Known semantic gaps

These are real, code-level behaviors to be aware of:

- **NULLs in GPU kernels.** Predicates and outputs over nullable columns are
  compiled with explicit validity checks (`plan::null_semantics`), which
  cost one validity load per nullable column read.
- **Placement is explicit.** `PhysicalPlan::planned_execution_tier()` and
  `QueryHandle::planned_execution_tier()` report `Gpu`, `Host`, or `Hybrid`.
  Runtime-dependent supported paths are classified `Hybrid` up front.
- **`IN` / `NOT IN` uses strict three-valued logic in both literal-list and
  subquery forms.** Literal lists are wrapped in an explicit CASE that returns
  TRUE, FALSE, or NULL; a NULL member therefore makes a non-match UNKNOWN.
- **Streaming grouped integer `SUM` overflow is a hard error.** Per-morsel
  device partials merge into an `i128` accumulator and are range-checked when
  rebuilt as Int32/Int64. The result cannot wrap even when the source is not
  replayed as one host batch.
- **SQL-compatible integer division/remainder is the default.** Invalid rows
  (`x / 0`, `x % 0`, and `INT_MIN / -1`) are NULL through the host evaluator.
  Aggregate-feed integer division and Decimal128 division are rejected where a
  nullable compatible path does not exist. `BOLT_LEGACY_ARITHMETIC=1` is the
  explicit opt-in to the historical zero/wrapping device convention.
- **Grouped AVG empty state is NULL.** Existing all-NULL groups emit NULL and an
  empty relation emits no groups, matching standard SQL.
- **Validated GPU string boundary.** Non-dictionary
  EXACT/PREFIX/SUFFIX/CONTAINS `LIKE` and ASCII `UPPER`/`LOWER` have direct
  CUDA-hardware coverage and are enabled by default. `BOLT_GPU_STRING=0`
  forces their host mirrors. Unicode case mapping and
  SUBSTRING/TRIM/CONCAT remain supported host implementations; their dormant
  device producers are not presented as supported.
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
  compaction) and column upload are wired for all three.
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
- **Window frame `EXCLUDE`**, `RANGE` offsets in months or years, and a
  `RANGE` offset over more than one `ORDER BY` key. (`ROWS`, `GROUPS` and
  `RANGE` frames, including numeric and `INTERVAL` offsets, are
  **supported**; so are `QUALIFY`, the named `WINDOW` clause, and
  `OVER <named_window>` — see below.)
- **Table-valued functions** *except* `generate_series(start, stop[, step])`,
  which is now **supported** in `FROM` — see below. Other TVFs (`unnest`, etc.)
  remain rejected: they would need a function-as-table-source mechanism, and
  the engine scans only registered tables / CTEs / derived tables.
- **`CONNECT BY`**, **`CLUSTER` / `DISTRIBUTE` / `SORT BY`**, **`INTO`**.
- **`GLOBAL JOIN`** (ClickHouse extension).
- **`ASOF JOIN` beyond its supported shape**: more than one per query, not
  the last join of the FROM clause, combined with `WITH`, `NATURAL`, or an
  `ON` that is not a conjunction of column equalities.
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
