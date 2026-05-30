# Craton Bolt Migration Guide

This guide covers the user-visible API deltas across the non-trivial
version jumps in Craton Bolt's history so far:

- **0.3.0 -> 0.5.0** — M2 SQL scalar completeness. `0.4` was skipped (see
  `CHANGELOG.md` for the rationale on skipped versions).
- **0.5.0 -> 0.6.0** — M1 foundation finishing plus the M3-M8 milestone
  work.
- **0.6.0 -> 0.7.0** — the v0.6 carry-overs become live GPU code paths.
  `0.7.0` is the current series.

Each section follows the same shape:

- **What changed** — the user-visible API delta in one paragraph.
- **Before / After** — code blocks where applicable.
- **Classification** — `Breaking`, `Soft-deprecated` (compiles, warns,
  scheduled removal), or `Additive` (purely new surface).

If you only need the headline:

- 0.3 -> 0.5: SQL gets noticeably more complete. Identifier resolution is
  now case-folded, which is the one change that *can* alter the behaviour
  of an existing working query against a mixed-case schema.
- 0.5 -> 0.6: prefer `Engine::Builder` over `Engine::new` /
  `Engine::new_with_device` going forward. `DataFrame::collect` is being
  re-purposed from a deprecated alias into a real materializing call —
  read the section before you upgrade.
- 0.6 -> 0.7: no API moved. Queries that parsed and type-checked in 0.6
  but rejected at the GPU lowering boundary with "not yet lowered to GPU"
  now execute — `Decimal128` / `Date` / `Timestamp` arithmetic, grouped
  `STDDEV` / `VAR`, and GPU-side `ORDER BY`. The SQL surface also widens:
  `EXCEPT` / `INTERSECT`, non-recursive CTEs, host-side window functions,
  uncorrelated subqueries, `JOIN ... USING` / `NATURAL`, and
  `COUNT(DISTINCT)` — all previously rejected at the parser — are now
  accepted (purely additive). The one change that can alter an existing
  result is the `DESC` radix-sort fix; the one change that can turn a
  previously-accepted query into an error is WHERE type-checking catching
  `LIKE` on a non-`Utf8` column.

---

## 0.3.0 -> 0.5.0 migration

The 0.5 cycle (M2) focused on closing the most painful holes in the SQL
surface: things that parsed but were rejected at plan time, things that
were silently order-rearranged, and a few new aggregate kernels. None of
the changes below remove existing API; the case-folding change is the
only one that can quietly alter the result of a previously-working query.

### Aggregate aliasing (`SUM(x) AS total`)

**What changed.** Aggregate expressions can now carry a SELECT-list
alias, and that alias is honoured end-to-end (output schema, GROUP BY
key naming, HAVING resolution). In 0.3.0 the planner rejected
`SUM(x) AS total` with a plan error; you had to project the rename in a
follow-up `SELECT` or rely on the default `sum_x` column name.

**Classification: Additive.** Existing queries that did *not* use an
alias continue to return the same column names (`sum_x`, `min_y`, ...).

```sql
-- Before (0.3.0): plan error
SELECT SUM(amount) AS total FROM orders GROUP BY user_id;

-- After (0.5.0): SELECT-list alias preserved, output column is `total`
SELECT SUM(amount) AS total FROM orders GROUP BY user_id;
```

In `HAVING`, the alias is now also resolvable alongside the raw
aggregate output name:

```sql
-- After (0.5.0): both forms work
SELECT user_id, SUM(amount) AS total FROM orders
GROUP BY user_id
HAVING total > 1000;          -- by SELECT alias

SELECT user_id, SUM(amount) AS total FROM orders
GROUP BY user_id
HAVING sum_amount > 1000;     -- by raw aggregate output name
```

### Qualified column references (`table.col`)

**What changed.** Two-part identifiers of the form `<table>.<col>` are
now accepted in `WHERE`, `SELECT`, `GROUP BY`, `HAVING`, and `JOIN ON`,
and resolve against the registered table's schema. In 0.3.0 the parser
emitted a "qualified column references not yet supported" error.

**Classification: Additive.** Bare column names (`col`) continue to
work and resolve unchanged; the qualified form is purely an opt-in
disambiguator for queries that need to spell which side of a JOIN they
came from.

```sql
-- Before (0.3.0): rejected at the SQL frontend
SELECT orders.user_id, users.name
FROM orders INNER JOIN users ON orders.user_id = users.id;

-- After (0.5.0): two-part identifiers resolve against the named table
SELECT orders.user_id, users.name
FROM orders INNER JOIN users ON orders.user_id = users.id;
```

Schema-qualified (three-part) names like `db.orders.user_id` are still
rejected — the engine does not model schemas yet. The error spells that
out explicitly.

### Case-folded identifiers

**What changed.** SQL identifiers are now resolved case-insensitively.
A column registered as `UserId` resolves equally well from
`SELECT userid`, `SELECT USERID`, or `SELECT UserId`. In 0.3.0
identifier lookup was case-sensitive, so the first two of those would
have returned a "column not found" error.

**Classification: Behavioural change — read carefully.** This is *not*
a breaking compile-time change (no API moved), but it CAN change the
runtime behaviour of a query that previously failed *or* the column
resolved against in a mixed-case schema.

The two scenarios worth checking before you upgrade:

1. **You relied on a case-sensitive failure.** If your application code
   caught "column not found" for a deliberately mis-cased column name
   as a sentinel — it will no longer fire. The lookup succeeds.
2. **You have two columns that differ only by case.** E.g. a table with
   both `userid` and `UserId`. The case-folded lookup is ambiguous; the
   engine resolves to whichever name the registered schema lists first.
   This is a corner case worth auditing if your tables come from
   case-preserving sources (e.g. legacy Excel imports).

```rust
// Before (0.3.0): exact-case match required at lookup time.
//
// engine.register_table("orders", batch_with_column_named("UserId"))?;
// engine.sql("SELECT UserId FROM orders")?;  // ok
// engine.sql("SELECT userid FROM orders")?;  // Err(Plan("column not found: userid"))

// After (0.5.0): identifiers are case-folded at lookup.
//
// engine.register_table("orders", batch_with_column_named("UserId"))?;
// engine.sql("SELECT UserId FROM orders")?;  // ok
// engine.sql("SELECT userid FROM orders")?;  // ok (resolves to UserId)
// engine.sql("SELECT USERID FROM orders")?;  // ok (resolves to UserId)
```

Output column names continue to be returned in the case the source
schema declared, so result-set consumers that key by exact column name
are unaffected.

### STDDEV / VARIANCE aggregates

**What changed.** `STDDEV_POP`, `STDDEV_SAMP`, `VAR_POP`, and
`VAR_SAMP` are now accepted as aggregate functions in `SELECT` and
`HAVING`. 0.3.0 supported only `SUM` / `MIN` / `MAX` / `COUNT` / `AVG`;
the parser rejected the variance / standard-deviation aggregates with
an "unsupported aggregate" error.

**Classification: Additive.**

```sql
-- Before (0.3.0): rejected at the SQL frontend
SELECT user_id, STDDEV_POP(latency_ms) FROM events GROUP BY user_id;

-- After (0.5.0): supported. Output dtype is Float64 regardless of
-- input dtype, matching the SUM(Int32) -> Int64 widening pattern from
-- 0.3.0 (a numeric aggregate always promotes to its widest safe form).
SELECT user_id, STDDEV_POP(latency_ms) FROM events GROUP BY user_id;
```

`STDDEV` and `VARIANCE` (no `_POP` / `_SAMP` suffix) are accepted as
aliases for the `_SAMP` variants, matching ANSI SQL.

### `IS NULL` / `IS NOT NULL`

**What changed.** The `IS NULL` and `IS NOT NULL` postfix predicates
are now first-class. They lower through the logical plan as
`Expr::Unary { op: UnaryOp::IsNull | UnaryOp::IsNotNull, operand }`
and type-check to `Bool` regardless of the operand's dtype (including
the untyped `Literal::Null` operand). In 0.3.0 the SQL frontend
rejected both with a parser error.

**Classification: Additive.**

```sql
-- Before (0.3.0): parser error
SELECT * FROM users WHERE email IS NULL;
SELECT * FROM users WHERE email IS NOT NULL;

-- After (0.5.0): supported
SELECT * FROM users WHERE email IS NULL;
SELECT * FROM users WHERE email IS NOT NULL;
```

Status note for GPU codegen: see the `Expr::Unary` doc comment in
`src/plan/logical_plan.rs`. The logical plane has accepted the syntax
since 0.5.0, and the M8 line item under 0.6 wires the GPU kernel path
through `Expr::Unary` end-to-end. On 0.5.x you may observe certain
shapes falling back to the host filter compaction path; the SQL still
works, the kernel path is what's evolving.

---

## 0.5.0 -> 0.6.0 migration

0.6 is where the public Rust surface starts to firm up for 1.0. The
headline is **prefer `Engine::Builder`** for new code — the existing
`Engine::new` and `Engine::new_with_device` constructors continue to
work and are *not* scheduled for removal in the 0.x series, but new
configuration knobs (memory pool sizing, observer wiring, streaming
table ingest) are exposed only through the builder.

### `Engine::Builder` (recommended path)

**What changed.** A fluent builder is now the recommended constructor.
`Engine::new()` and `Engine::new_with_device(idx)` keep working as
thin wrappers — they are *soft-deprecated*, not removed, because they
remain the right API for the trivial "one-GPU, default everything"
case. New configuration knobs land on the builder.

**Classification:**
- `Engine::Builder` itself: **Additive**.
- `Engine::new` / `Engine::new_with_device`: **Soft-deprecated**.
  They compile, do not warn in 0.6.x, and call into the builder under
  the hood. Plan to migrate at your leisure; we will re-evaluate the
  deprecation policy at the 1.0 boundary.

```rust
// Before (0.3 / 0.5): direct constructors.
use craton_bolt::Engine;

let engine = Engine::new()?;                   // device 0
let engine = Engine::new_with_device(1)?;      // device 1

// After (0.6): builder, recommended for new code.
use craton_bolt::Engine;

let engine = Engine::builder()
    .device(1)
    .build()?;

// Default-everything case — equivalent to Engine::new():
let engine = Engine::builder().build()?;
```

The builder is the only place that exposes knobs added in 0.6 (see
`register_table_stream` and the periodic pool-stats observer below);
the legacy constructors stay on the device-0 / defaults-only path
forever.

### `DataFrame::collect` materialises

**What changed.** In 0.3.0 / 0.5.0 `DataFrame::collect` was a
`#[doc(hidden)] #[deprecated(since = "0.1.0", note = "use into_plan()
instead")]` tombstone that returned a bare `LogicalPlan` — i.e. it
delegated to `into_plan()`. 0.6 finally makes the name do what
Polars / pandas users expect: `collect()` materialises the plan
against the engine and returns an Arrow `RecordBatch`.

`into_plan()` is unchanged and remains the way to get the raw
`LogicalPlan` out without executing it.

**Classification: Breaking** for code that was still calling
`collect()` despite the existing `#[deprecated]` warning. The
deprecation note since 0.1 said to use `into_plan()`; if your
0.3 / 0.5 build was silencing or ignoring the warning, your call
site will return a different type after the upgrade.

```rust
// Before (0.3 / 0.5): #[doc(hidden)] deprecated alias for into_plan().
//   pub fn collect(self) -> LogicalPlan { self.into_plan() }
//
// The deprecation note since 0.1.0 has been "use into_plan() instead".
use craton_bolt::{DataFrame, LogicalPlan, plan::logical_plan::Schema};

let plan: LogicalPlan = DataFrame::scan("orders", schema).collect();
// (deprecation warning since 0.1.0)

// After (0.6): collect() takes an Engine and returns RecordBatch.
// into_plan() still gives you the raw plan unchanged.
use craton_bolt::{DataFrame, Engine, plan::logical_plan::Schema};
use arrow_array::RecordBatch;

let engine = Engine::builder().build()?;
let batch: RecordBatch = DataFrame::scan("orders", schema)
    .filter(/* ... */)
    .collect(&engine)?;

// Or, unchanged, get the raw plan without executing:
let plan = DataFrame::scan("orders", schema).into_plan();
```

If you have a large number of `collect()` call sites that were really
using it as a `LogicalPlan` accessor, mechanical rename to
`into_plan()` is the migration. The compiler will catch every site
because the signature changed.

### `register_table_stream`

**What changed.** A new `register_table_stream` API on the builder
lets you register a table whose batches arrive over an iterator /
channel rather than handing the engine a single fully-materialised
`RecordBatch` (or appending with `register_batch`). This is purely
additive — `register_table` and `register_batch` continue to work as
before.

**Classification: Additive.**

```rust
// Before (0.5): register a fully-materialised RecordBatch, or append
// further batches one at a time via register_batch.
use craton_bolt::Engine;

let mut engine = Engine::new()?;
engine.register_table("orders", first_batch)?;
engine.register_batch("orders", second_batch)?;
engine.register_batch("orders", third_batch)?;

// After (0.6): stream batches in via Engine::register_table_stream.
// It is an `Engine` method (call it after build()), not an
// EngineBuilder method. The engine consumes the iterator at
// registration time; downstream queries see the same multi-batch table
// they would have built up via register_batch calls.
//
// The iterator yields `BoltResult<RecordBatch>` (so a fallible source
// can propagate errors), and a declared `Schema` is required up front.
use craton_bolt::{Engine, BoltResult};
use arrow::record_batch::RecordBatch;

let mut engine = Engine::new()?;
let schema = first_batch.schema();
let batches: Vec<BoltResult<RecordBatch>> =
    vec![Ok(first_batch), Ok(second_batch), Ok(third_batch)];
engine.register_table_stream("orders", (*schema).clone(), batches)?;
```

The classic `register_table` / `register_batch` pair stays the API
for callers that want to register tables incrementally after the
engine is built.

### `tracing` crate dependency

**What changed.** Craton Bolt now uses the `tracing` crate alongside
the existing `log` dep for internal diagnostics. The engine's
periodic pool-stats line (Stage 7 / P1b — `BOLT_POOL_STATS_INTERVAL_SECS`)
emits both a `log::info!` record (preserving the 0.3 / 0.5 wire format
for downstream log-scraping) and a `tracing` span/event so that
applications already wired into `tracing-subscriber` can capture
engine-internal telemetry structurally.

**Classification: Additive** for runtime behaviour. The dependency
itself is new, so downstream lockfiles will see new transitive crates;
this matters for security-audit tooling and offline-build environments
but does not change any API.

If you have a global `log` -> `tracing` bridge installed (e.g. via
`tracing_log::LogTracer`), make sure you are not double-counting the
pool-stats line. The `log::info!` path is unchanged; the new
`tracing` event is what's additional.

### `BoltError` is `#[non_exhaustive]`, plus did-you-mean hints

**What changed (variants).** `BoltError` carries a `#[non_exhaustive]`
marker as of 0.6. New variants can be added in future point releases
without it being a breaking change; pattern matches on `BoltError`
MUST include a `_ => ...` arm.

The existing variant set on 0.6 is:

```text
BoltError::Cuda(String)                 // free-form, no CUresult
BoltError::CudaWithCode { code, message } // driver error with CUresult
BoltError::Sql(String)
BoltError::Plan(String)
BoltError::Type(String)
BoltError::Memory(String)
BoltError::Io(std::io::Error)
BoltError::GpuCapacity(String)          // typed "fall back to host" signal
BoltError::Other(String)
```

(The pre-0.6 line-up was the same — 0.6 is adding the non-exhaustive
marker, not introducing or removing variants in this release. See
`src/error.rs`.)

**Classification: Breaking** for downstream code that exhaustively
matched `BoltError` without a wildcard arm. The recommended fix is to
add `_ => ...` to your match.

```rust
// Before (0.3 / 0.5): exhaustive match was sound.
use craton_bolt::BoltError;

match err {
    BoltError::Cuda(s) => /* ... */,
    BoltError::CudaWithCode { code, message } => /* ... */,
    BoltError::Sql(s) => /* ... */,
    BoltError::Plan(s) => /* ... */,
    BoltError::Type(s) => /* ... */,
    BoltError::Memory(s) => /* ... */,
    BoltError::Io(e) => /* ... */,
    BoltError::GpuCapacity(s) => /* ... */,
    BoltError::Other(s) => /* ... */,
}

// After (0.6): #[non_exhaustive] requires a wildcard.
use craton_bolt::BoltError;

match err {
    BoltError::Cuda(s) => /* ... */,
    BoltError::CudaWithCode { code, message } => /* ... */,
    BoltError::Sql(s) => /* ... */,
    BoltError::Plan(s) => /* ... */,
    BoltError::Type(s) => /* ... */,
    BoltError::Memory(s) => /* ... */,
    BoltError::Io(e) => /* ... */,
    BoltError::GpuCapacity(s) => /* ... */,
    BoltError::Other(s) => /* ... */,
    _ => /* future variants land here */,
}
```

**What changed (messages).** Error messages produced by the SQL
frontend, the logical-plan validator, and `DataFrame` builder-time
validation now include a *did-you-mean* hint whenever the user
referenced an unknown column or table whose name has a close edit-
distance match in the registered schema. The hint is appended to the
existing message; downstream string-matching on the error text should
still find the original substring (`column not found: foo`) but
should not assume the message ENDS there.

**Classification: Additive** for the message content, **Soft-breaking**
for callers that did `assert_eq!` on the exact error string. We
recommend `assert!(s.contains("column not found"))` style assertions.

```rust
// Before (0.3 / 0.5): "column not found: usrid"
// After  (0.6):       "column not found: usrid (did you mean: user_id?)"
```

The did-you-mean hint applies to:
- Unknown column names in `SELECT`, `WHERE`, `GROUP BY`, `HAVING`,
  `ORDER BY`, and `JOIN ON`.
- Unknown table names in `FROM` / `JOIN`.
- Unknown aggregate output names in `HAVING` (when neither the SELECT
  alias nor the raw `sum_x`-style name resolves).
- DataFrame builder-time validation errors surfaced through
  `DataFrame::validation_error()`.

CUDA / kernel / IO error messages are unchanged.

---

## 0.6.0 -> 0.7.0 migration

0.7 adds almost no new public surface. Its job is to turn the v0.6
carry-overs — the features that parsed and type-checked but rejected at
the GPU lowering boundary with a "not yet lowered to GPU" message — into
live execution paths. For most callers this is invisible: SQL you could
already write but not run now runs. The two items worth reading before you
upgrade are the `DESC` radix-sort correctness fix (it can change a result)
and the new WHERE-clause type-check (it can turn a previously-accepted
query into a clear error).

### `Decimal128` arithmetic, comparisons, and `SUM` now execute

**What changed.** `Decimal128` columns are now reachable through GPU
lowering. Arithmetic (`+`, `-`, `*`) and comparisons (`=`, `!=`, `<`,
`>`, `<=`, `>=`) lower to the GPU, the latter being usable from `WHERE`
predicates; `SUM(Decimal128)` is supported via a host-side reduction. In
0.6, `DataType::Decimal128(p, s)` plumbed through the logical plan and
Arrow round-trip and `CAST(int AS DECIMAL(p, s))` parsed, but physical
lowering rejected with `"Decimal128 not yet lowered to GPU"`.

**Classification: Additive** for the surface — no API moved, and the SQL
itself was already accepted by the parser / type-checker in 0.6.
**Behavioural** in the narrow sense that a query which previously returned
the `"Decimal128 not yet lowered to GPU"` error now executes instead.

```sql
-- Before (0.6): parses and type-checks, then errors at lowering with
--   "Decimal128 not yet lowered to GPU"
SELECT SUM(price) FROM line_items WHERE price > 0;   -- price is DECIMAL(p, s)

-- After (0.7): lowers and executes.
SELECT SUM(price) FROM line_items WHERE price > 0;
```

### `Date32` / `Timestamp` arithmetic now executes

**What changed.** `Date32` and `Timestamp` subtraction lowers to the GPU:
`Date − Date` and `Timestamp − Timestamp`, plus Day-`INTERVAL` arithmetic
only. The literal and type plumbing shipped in 0.6 (`DATE '...'` /
`TIMESTAMP '...'`); 0.7 wires the runtime path.

**Classification: Additive / Behavioural**, same shape as the
`Decimal128` item above — previously-rejected lowering now succeeds for
the supported operations.

```sql
-- Before (0.6): parses and type-checks, rejected at GPU lowering.
SELECT shipped_at - ordered_at FROM orders;          -- Timestamp - Timestamp

-- After (0.7): lowers and executes (Day-INTERVAL granularity).
SELECT shipped_at - ordered_at FROM orders;
```

Operations beyond Date−Date / Timestamp−Timestamp and Day-`INTERVAL` are
still out of scope for this release.

### Grouped `STDDEV` / `VAR` under `GROUP BY`

**What changed.** `STDDEV` / `VAR` aggregates are now supported under a
`GROUP BY`, computed with a per-group host-side Welford pass. 0.5 added
the scalar (no-`GROUP BY`) variants; the grouped form was an explicit v0.6
carry-over.

**Classification: Additive.** Scalar `STDDEV` / `VAR` queries are
unaffected; the grouped form is new surface that previously rejected.

```sql
-- Before (0.6): scalar form worked; the grouped form rejected.
SELECT user_id, STDDEV_POP(latency_ms) FROM events GROUP BY user_id;

-- After (0.7): grouped STDDEV / VAR supported.
SELECT user_id, STDDEV_POP(latency_ms) FROM events GROUP BY user_id;
```

### GPU radix sort for `ORDER BY` (plus a `DESC` correctness fix)

**What changed.** `ORDER BY` now dispatches to the GPU radix sort in
`src/exec/sort.rs` for single-key `Int32` / `Int64` ascending, with
multi-key and `DESC` support. In 0.6 the radix-sort kernel existed only as
a scaffold, gated behind `BOLT_GPU_SORT=1` and not selected by the
planner; the host-side sort was the default path. This release also fixes
the `DESC` pre-transform to use `!(val ^ MIN)` instead of a bare `!val`.

**Classification: Additive** for the dispatch (the SQL surface is
unchanged — `ORDER BY` already worked host-side). **Behavioural — read
carefully** for the `DESC` fix: a `DESC` sort that exercised the buggy
pre-transform could previously return rows in the wrong order. The result
ordering, not the API, is what changes.

The change is internal to the executor — there is no before/after code at
the SQL or Rust API level. If you have golden-result fixtures that pinned
the *previous* (incorrect) `DESC` ordering, re-baseline them after the
upgrade.

### New SQL surface (set ops, windows, CTEs, subqueries, joins)

**What changed.** A batch of SQL constructs that 0.6 rejected at the
parser / planner now lower and execute:

- **`EXCEPT [ALL]` / `INTERSECT [ALL]`** — host-side set / multiset
  operations.
- **Non-recursive CTEs** — `WITH name AS (...) SELECT ...`.
- **Window functions** — `ROW_NUMBER` / `RANK` / `DENSE_RANK` and
  `SUM` / `AVG` / `MIN` / `MAX` / `COUNT` `OVER (PARTITION BY ... ORDER
  BY ...)`, host-side, default frame only, as top-level SELECT items.
- **Uncorrelated subqueries** — scalar `(SELECT ...)` and
  `[NOT] IN (SELECT ...)` in `SELECT` / `WHERE`.
- **`JOIN ... USING (...)` / `NATURAL JOIN`** — desugared to equi-key
  joins.
- **`COUNT(DISTINCT col)`** — as the sole SELECT item.
- **GPU `LIKE`** over `Utf8`, and GPU `UPPER` / `LOWER` / `LENGTH`;
  host-side `SUBSTRING` / `TRIM` / `CONCAT`.

**Classification: Additive.** These were hard parser/planner rejections
in 0.6, so no previously-working query changes behaviour. Correlated
subqueries, `EXISTS`, derived tables in `FROM`, `WITH RECURSIVE`,
`QUALIFY`, the named `WINDOW` clause, and non-default window frames
remain rejected. See `docs/SQL_REFERENCE.md` for the exact caveats on
each.

```sql
-- Before (0.6): rejected at the SQL frontend.
-- After  (0.7): supported.
WITH recent AS (SELECT * FROM orders WHERE ts > 0)
SELECT region, COUNT(DISTINCT user_id)            -- COUNT(DISTINCT), sole item
  FROM recent;                                    -- (illustrative; see caveats)

SELECT user_id, ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY ts)
  FROM orders;

SELECT region FROM orders EXCEPT SELECT region FROM archived_orders;

SELECT * FROM orders WHERE user_id IN (SELECT id FROM vip_users);
```

### WHERE-clause predicate type-checking

**What changed.** SQL lowering now type-checks `WHERE` predicates. The
motivating fix is `LIKE` applied to a non-`Utf8` column: this previously
slipped past lowering and is now rejected with a type error at plan time.

**Classification: Soft-breaking** for queries that relied on the missing
check. A `WHERE` predicate that was malformed (e.g. `LIKE` against a
numeric column) and previously reached execution now fails earlier, with a
type error, rather than misbehaving downstream. Well-typed predicates are
unaffected.

```sql
-- Before (0.6): LIKE on a non-Utf8 column was not caught during lowering.
SELECT * FROM events WHERE latency_ms LIKE '1%';   -- latency_ms is numeric

-- After (0.7): rejected with a type error during SQL lowering.
SELECT * FROM events WHERE latency_ms LIKE '1%';   -- type error
```

---

## Quick upgrade checklist

If you are jumping straight from 0.3 -> 0.6 (skipping a 0.5 stop):

1. Search for case-sensitive column-name comparisons in your wrapper
   code (the case-folding change is the one quiet behaviour shift).
2. If you match on `BoltError` exhaustively, add a `_` arm.
3. If you call `DataFrame::collect()`, decide whether you wanted
   `into_plan()` (raw plan) or the new materializing form (Arrow
   `RecordBatch`, requires `&Engine`).
4. Migrate new code to `Engine::builder()`. Leave existing
   `Engine::new()` / `Engine::new_with_device()` call sites alone
   unless you need a builder-only knob.
5. Update any `assert_eq!` on error message strings to
   `contains(...)` to accommodate did-you-mean hints.

Additionally, if you are coming from 0.6 -> 0.7:

6. Audit any `WHERE` predicate using `LIKE` against a non-`Utf8` column —
   it now fails with a type error at plan time instead of slipping
   through lowering.
7. Re-baseline any golden-result fixtures that pinned a `DESC` ordering;
   the radix-sort `DESC` pre-transform fix can change the row order a
   previously-buggy sort produced.
