# SQL Reference

The exact subset of SQL Craton Bolt's frontend accepts. The grammar is built on top of [`sqlparser`](https://github.com/apache/datafusion-sqlparser-rs); anything outside this surface produces a clear `BoltError::Sql(...)` or `BoltError::Plan(...)` with the unsupported construct.

This document tracks the 0.7.0 release. For the JIT pipeline that lowers and executes these queries, see [`JIT_PIPELINE.md`](JIT_PIPELINE.md). For the gap to 1.0, see [`../ROADMAP.md`](../ROADMAP.md).

A note on execution tiers. The SQL surface below is wider than the set of
constructs that run end-to-end on the GPU. Throughout this document each
feature is tagged with where it actually executes:

- **GPU** — lowered to a PTX kernel and run on the device.
- **host-side** — parses, type-checks, and *executes* correctly, but on a
  host (CPU) code path rather than the GPU.
- **parses; GPU lowering pending** — accepted by the frontend and
  type-checker, but the physical layer rejects it at the GPU lowering
  boundary with a clear `"… not yet lowered to GPU"` message. The query
  fails rather than running on a slow fallback.

## Supported query shape

```
[WITH <name> AS (<query>) [, ...]]
SELECT [DISTINCT] <select_list>
  FROM <table>
       [{INNER | LEFT [OUTER] | RIGHT [OUTER] | FULL [OUTER]} JOIN <table>
            {ON <equi_predicate> | USING (<col>, ...) | NATURAL}
        | CROSS JOIN <table>] ...
 [WHERE  <bool_expr>]
 [GROUP BY <expr_list>]
 [HAVING <bool_expr>]
 [ORDER BY <expr> [ASC|DESC] [NULLS FIRST|NULLS LAST] [, ...]]
 [LIMIT <int_literal>] [OFFSET <int_literal>]
```

Window functions (`func(...) OVER (...)`) may appear as top-level SELECT items. Uncorrelated scalar and `[NOT] IN` subqueries may appear in `SELECT` / `WHERE`.

Two queries can be combined with `UNION` / `UNION ALL` / `EXCEPT [ALL]` / `INTERSECT [ALL]`; the optional `ORDER BY` / `LIMIT` / `OFFSET` then apply to the combined result. A query may be prefixed with a non-recursive `WITH` (CTE) clause.

**Hard restrictions** (everything else returns a `BoltError`):

- Exactly one base table in `FROM`, optionally widened by one or more JOINs per `SELECT`: `INNER`, `LEFT [OUTER]`, `RIGHT [OUTER]`, or `FULL [OUTER]` with an `ON` / `USING (...)` / `NATURAL` constraint, or `CROSS JOIN` (no constraint). Every shape has a gated GPU fast path with a host-side fallback (see the [JOIN](#join) section). `JOIN ... USING (...)` and `NATURAL JOIN` desugar to equi `<left.col> = <right.col>` pairs; computed join keys are still rejected. The equi `ON` predicate must be a conjunction of `<left.col> = <right.col>` equalities (a small-cardinality non-equi INNER predicate is handled by a capped host nested-loop).
- **Uncorrelated** scalar (`(SELECT ...)`) and `[NOT] IN (SELECT ...)` subqueries in `SELECT` / `WHERE` are supported (resolved to constants before lowering). An uncorrelated **scalar subquery** is also accepted in `ORDER BY`. **Correlated** subqueries and `EXISTS` / `NOT EXISTS` are rejected. **Derived tables** (a subquery in `FROM`, `(SELECT ...) AS alias`) are supported as of 0.7, with restrictions: the alias is **required**, `LATERAL` (correlated) derived tables are rejected, and a column-list alias (`AS d(x, y)`) is rejected.
- Non-recursive CTEs (`WITH name AS (...)`) are supported. `WITH RECURSIVE`, CTE column-list aliases (`WITH c (a, b) AS ...`), and the materialization hint are rejected.
- `EXCEPT [ALL]` and `INTERSECT [ALL]` are supported (host-side). `UNION BY NAME` (and `EXCEPT`/`INTERSECT BY NAME`) are rejected.
- Window functions (`OVER`) are supported host-side for a fixed function set under the default frame only (see the [Window functions](#window-functions) section). `QUALIFY`, the named `WINDOW` clause, and explicit/non-default frames are rejected. `LATERAL`, table-valued functions, `PREWHERE` (ClickHouse-ism), `CONNECT BY`, `CLUSTER / DISTRIBUTE / SORT BY`, `FETCH`, `FOR UPDATE/SHARE`, `INTO` remain rejected.
- No `GROUP BY ALL`, `ROLLUP`, `CUBE`, `TOTALS`.
- No schema-qualified *table* names in `FROM`. Single-level qualified column references (`t.col`) are supported in SELECT, WHERE, GROUP BY, HAVING, and `JOIN ... ON` predicates; the qualifier must match a FROM table and only the resolved column name survives lowering. In a `JOIN ... ON` predicate the **schema-qualified `schema.table.col`** form is also accepted (the leading single-catalog segment is dropped, so it resolves to `table.col`). Four-or-more-segment references (`catalog.db.t.col`) and struct-field access are rejected.
- `LIMIT` and `OFFSET` must be integer literals; expressions and parameters are rejected.

## Data types

The plan's `DataType` enum is intentionally small.

| Plan dtype | Arrow dtype     | Notes                                                    |
|------------|-----------------|----------------------------------------------------------|
| `Bool`     | `Boolean`       | Bitmap on host; 1 byte per row on the GPU.               |
| `Int32`    | `Int32`         |                                                          |
| `Int64`    | `Int64`         |                                                          |
| `Float32`  | `Float32`       |                                                          |
| `Float64`  | `Float64`       |                                                          |
| `Utf8`     | `Utf8`          | Dictionary-encoded on register; i32 or i64 indices.      |
| `Decimal128(p, s)` | `Decimal128(p, s)` | `+`, `-`, `*`, **`/`** all lower to GPU (dual-register 128-bit IR), including **mixed Decimal/integer** arithmetic (the integer side is auto-coerced); comparisons (`=`, `!=`, `<`, `>`, `<=`, `>=`) lower to GPU and are **scale-aligned** (differing scales are rescaled to `max(s)` before the i128 compare). `SUM`/`MIN`/`MAX` are host-side. **CAST** integer↔Decimal128 and Decimal128↔Decimal128 (rescale) lower to GPU; Float↔Decimal128 stays rejected. GPU gather (filter/compaction) + upload are wired (16-byte interleaved layout). **CASE** with a Decimal128 result: parses; GPU lowering pending. |
| `Date32`   | `Date32`        | `DATE '…'` literals; Date−Date and Day-`INTERVAL` arithmetic lower to GPU. GPU gather (filter/compaction) + upload are wired (i32 days-since-epoch layout). `COUNT(date_col)`, **`MIN(date_col)` and `MAX(date_col)`** all work end-to-end (the MIN/MAX reduction runs on the GPU over the i32 storage and the result is rebuilt as a `Date32`); `SUM` over a date is rejected by design. CAST integer→Date32 lowers via the Decimal/i128 path; CAST to/from Date32 (string, etc.): parses; GPU lowering pending. |
| `Timestamp(unit, tz)` | `Timestamp(unit, tz)` | `TIMESTAMP '…'` literals; Timestamp−Timestamp arithmetic lowers to GPU. Timezones are interned. GPU gather (filter/compaction) + upload are wired (i64 ticks-since-epoch layout, unit + tz preserved on download). `COUNT(ts_col)`, **`MIN(ts_col)` and `MAX(ts_col)`** all work end-to-end (the MIN/MAX reduction runs on the GPU over the i64 storage and the result is rebuilt preserving unit + timezone); `SUM` over a timestamp is rejected by design. CAST to/from Timestamp: parses; GPU lowering pending. |

`Decimal128`, `Date32`, and `Timestamp` arrived in 0.6 (plan + parser +
type-check) and gained their GPU lowering in 0.7 (see the per-type notes
above; the Decimal `/`, mixed Decimal/integer arithmetic, integer↔decimal
and decimal-rescale CAST, scale-aligned Decimal comparison, and temporal
`MIN`/`MAX` routing all landed in the 0.7 wave). Interval (beyond
Day-INTERVAL on dates), time-of-day, list, struct, and map are still not
modelled.

## Literals

- Integer: `42`. Parsed as `Int64` (negative-literal supported via unary minus folding; pure integer literals outside `i64` range are rejected, not silently demoted).
- Float: `3.14`. Parsed as `Float64`.
- String: `'US'`. Parsed as `Utf8`.
- Boolean: `TRUE` / `FALSE`. Case-insensitive per `sqlparser`.
- Null: `NULL`. Parsed as `Literal::Null`; semantics in expressions are limited (see Type checking).

Unary minus on a numeric literal is folded into a signed literal (`-5` becomes `Literal::Int64(-5)`, and `-9223372036854775808` is preserved as `i64::MIN`). On any other expression it's rewritten as `0 - expr`. Unary plus is a no-op. Any other unary op (including SQL `NOT`) is an error.

## Operators

### Arithmetic

`+`, `-`, `*`, `/` over Int32 / Int64 / Float32 / Float64. Numeric type promotion follows the standard SQL rules already used by the lowering (`physical_plan::unify_numeric`):

- Same dtype → same.
- `Float64` op anything → `Float64`.
- `Float32` op `Int64` (either order) → `Float64`.
- `Float32` op anything else → `Float32`.
- `Int64` op anything else → `Int64`.
- Else → `Int32`.

Integer division by zero produces `NULL` (host evaluator) or undefined behaviour (GPU kernel — IEEE follows for floats, integer div by zero is the user's problem). Float division follows IEEE-754: `1.0 / 0.0 = +inf`, `0.0 / 0.0 = NaN`.

#### Decimal128 arithmetic

As of 0.7 `+`, `-`, `*`, **and `/`** over `Decimal128(p, s)` lower to the **GPU** via the dual-register (lo/hi) 128-bit IR (`Op::Add128` / `Sub128` / `Mul128` / `Div128`). **Mixed Decimal/integer** arithmetic is supported: an `Int32` / `Int64` peer is auto-coerced to `Decimal128(_, 0)` (Float peers are rejected — CAST explicitly to avoid losing exactness). Result-dtype rules follow the SQL convention (`logical_plan::decimal128_arith_result`):

- **`+` / `-`**: operands rescaled to a common scale `s = max(s_l, s_r)`; result `Decimal128(min(max(p_l, p_r) + 1, 38), s)`.
- **`*`**: result `Decimal128(min(p_l + p_r, 38), s_l + s_s)` (the raw i128 product carries the summed scale; no operand rescale).
- **`/`**: result `Decimal128(min(max(p_l, 1), 38), max(s_l, 6))` — the quotient scale is the dividend's scale floored at **6** fractional digits, so an integer / low-scale dividend still gets fractional digits. The dividend is pre-scaled before a 128-bit truncating (toward zero) divide.

**Division by zero on the eager GPU path yields a deterministic `0`** for that lane (`Op::Div128` branches a zero divisor to a zero-quotient tail — non-trapping), consistent with the engine's integer-div-by-zero convention rather than raising the standard-SQL error. A result precision > 38 (Arrow's `Decimal128` ceiling) is a hard error rather than a silent wrap, as is an overflowing decimal `SUM`.

```sql
SELECT price * qty            FROM line_items;   -- Decimal * Decimal (GPU)
SELECT amount + 1             FROM ledger;        -- mixed Decimal + integer (GPU)
SELECT total / count          FROM ledger;        -- Decimal / Decimal, scale max(s,6) (GPU)
SELECT * FROM ledger WHERE amount > 100.00;       -- scale-aligned Decimal compare (GPU)
SELECT CAST(qty AS DECIMAL(20, 4)) FROM line_items;        -- integer -> Decimal (GPU)
SELECT CAST(price AS DECIMAL(10, 2)) FROM line_items;      -- Decimal rescale (GPU)
```

### Comparison

`=`, `<>` (also `!=`), `<`, `<=`, `>`, `>=`. Result dtype is `Bool`. Both operands must unify under the rules above. Numeric comparisons run on the **GPU**. `Decimal128` comparisons (`=`, `!=`, `<`, `>`, `<=`, `>=`) lower to the **GPU** (`Op::Cmp128`) as of 0.7 and are **scale-aligned**: two decimals with *differing* scales are rescaled to the common scale `max(s_l, s_r)` (the smaller-scale side is multiplied by `10^Δ`) before the i128 compare, and an **integer peer** is auto-coerced (`WHERE dec_col > 5` works). Precision need not match. Float/Bool/temporal peers are rejected — CAST explicitly.

For `Utf8` columns, equality (`=`, `<>`, `!=`) against string *literals* is supported — the literal rewriter folds `WHERE col = 'X'` into integer equality on the column's dictionary index at plan time (**GPU**). As of 0.7, **ordering comparisons** (`<`, `>`, `<=`, `>=`) of a `Utf8` column against a string *literal* (`WHERE name < 'M'`, either operand order) are **also** supported (**GPU**): because dictionary indices reflect insertion order rather than lex order, the rewriter partitions the dictionary entries by the literal under **binary (UTF-8 byte) collation** at plan time and emits an OR-of-equalities over the matching dictionary indices (the same index-membership form `LIKE` uses). This is byte/binary collation, **not** locale-aware / ICU collation, and a `NULL` row never satisfies an ordering predicate (correct SQL 3VL). Column-vs-column Utf8 ordering (`WHERE a < b`, two string columns) is **not** folded — there is no single literal to partition by — and stays a host string comparison. `MIN(utf8_col)` and `MAX(utf8_col)` use lex order on the host.

### IN and BETWEEN

`<expr> [NOT] IN (v1, v2, …)` is supported (0.5). It desugars to an OR/AND chain of element-wise comparisons, so it executes wherever the underlying comparisons do — on the **GPU** for numeric columns. Capped at 64 values; a large-list hash probe is a follow-up. `IN` against a `Utf8` column is still not wired through the dictionary rewriter — use an explicit `OR` chain of literal equalities.

`<expr> [NOT] BETWEEN low AND high` is supported (0.5), desugared to `(expr >= low) AND (expr <= high)` (or the DeMorgan inverse), and likewise runs on the **GPU** for numeric operands.

### Logical

`AND`, `OR`. Both operands must be `Bool`. Result is `Bool`. **GPU**.

`NOT <bool-expr>` is supported (0.5) via `UnaryOp::Not`, routed through the **host-side** filter path. GPU lowering of `NOT` is pending (it is rejected with `"NOT not yet lowered to GPU; requires host fallback"` on the GPU path).

### String concatenation

`a || b` (`BinaryOp::Concat`) is supported (0.5). In a SELECT position it runs through the **host-side** `Project` executor. In a `WHERE` predicate the whole filter is routed to the **host-side** filter path.

### CASE / CAST / COALESCE / NULLIF

- `CASE WHEN cond THEN val [WHEN…] [ELSE val] END` (both the plain and simple/with-operand forms) is supported (0.5). Execution tier by result dtype:
  - **GPU** when the unified result dtype is numeric, `Bool`, `Date32`, or `Timestamp` (emitted as a fold of PTX `selp.b32` / `selp.b64`; Date32/Timestamp ride along as plain bit-copies on their i32/i64 storage).
  - **host-side** when the result dtype is `Utf8`: a bare-`Scan` SELECT-list Utf8 `CASE` lowers to the host-realized `PhysicalPlan::StringProject` (`CaseUtf8` output, evaluated by `string_project::eval_case_utf8`). It follows SQL three-valued logic — **only a TRUE `WHEN` fires**; a `FALSE` or `NULL` condition falls through, and a row that matches no `WHEN` with no `ELSE` yields SQL `NULL`. A *nested* `CASE` inside a branch is out of scope and falls back / errors. (Utf8 CASE under a Filter/Project chain, or feeding an aggregate, is not on this host path.)
  - A `CASE` whose result dtype is **`Decimal128`** parses but **GPU lowering is pending** (rejected with `"CASE over Decimal128 … not yet lowered to GPU"`; the `selp` register classes have no `b128`, and there is no host realisation for it yet).
- `CAST(expr AS type)` is supported (0.5) for primitive numeric and `Bool` pairs, lowered to a PTX `cvt.*` on the **GPU**. As of 0.7 **integer↔`Decimal128`** and **`Decimal128`↔`Decimal128`** (rescale) also lower to the **GPU** via the 128-bit widen / rescale (`Mul128`) / truncating-divide (`Div128`) path; integer→`Date32` rides the same i128 widen. Still pending: CAST between **Float and `Decimal128`** (rejected — a correct round-to-nearest i128 scaling is not expressible on the fixed `cvt` path; CAST through an integer or an intermediate decimal), and CAST to/from `Timestamp` / `String` (parses; GPU lowering pending).
- `COALESCE(a, b, …)` and `NULLIF(a, b)` are supported (0.5), desugared to `CASE`, so they execute on the **GPU** under the same numeric/Bool result-dtype rule as CASE.

### LIKE

`<expr> [NOT] LIKE 'pattern'` is supported (0.5) for constant patterns with `%` and `_` wildcards (with prefix / suffix / contains / exact fast paths). As of 0.7 a `LIKE` / `NOT LIKE` predicate over a `Utf8` column **lowers to the GPU**: dictionary-encoded columns use a dictionary-precompute → index-membership kernel, and non-dictionary `Utf8` columns use the `StringLikeFilter` device matcher (`compile_like_match_kernel`, with EXACT/PREFIX/SUFFIX/CONTAINS specialisations). Both retain a **host-side** `host_like` fallback on a gate miss. `LIKE` with an `ESCAPE` clause is fully implemented: `<expr> [NOT] LIKE 'pattern' ESCAPE '\'` honours the escape character so a literal `%` / `_` / escape char in the pattern is matched verbatim (the escape is applied during pattern compilation, on both the GPU-lowered and host-side paths). WHERE-predicate `LIKE` is type-checked against the column dtype during lowering (must be `Utf8`).

`<expr> [NOT] ILIKE 'pattern'` (case-insensitive `LIKE`) is also supported, with the same wildcard, fast-path, `ESCAPE`, and execution-tier behaviour as `LIKE`. ILIKE performs **Unicode-aware per-character case folding** when matching (not an ASCII-only fold), so case-insensitive matching is correct for non-ASCII text.

### IS [NOT] NULL

`<expr> IS NULL` and `<expr> IS NOT NULL` are supported (`UnaryOp::IsNull` / `UnaryOp::IsNotNull`). The result dtype is `Bool` and any operand dtype is accepted. Semantics follow SQL: `IS NULL` is `TRUE` exactly for rows whose operand is NULL, and `IS NOT NULL` is its pointwise inverse.

Execution tier:

- **GPU.** When the operand is a **bare, nullable column reference**, the predicate lowers to the GPU `Op::IsNullCheck`, which reads the column's validity bitmap directly (`KernelSpec::input_has_validity` must mark the slot). This is the common `WHERE col IS NULL` / `WHERE col IS NOT NULL` case.
- **Constant-folded.** When the operand is statically known to be non-nullable (a non-nullable column or a non-null literal), the planner folds the check to a constant Bool at plan time — `IS NULL` becomes all-`FALSE`, `IS NOT NULL` all-`TRUE` — so no kernel work happens. An operand that is `Literal::Null` folds the other way.
- **Host-side.** A **compound** operand (e.g. `(x + y) IS NULL`, `(expr AS renamed) IS NULL`) takes the host-side filter fallback via `predicate_contains_unary`; the host evaluator (`src/exec/expr_agg.rs`) computes per-row nullness and applies the check.

`IS NULL` / `IS NOT NULL` are not currently wired into the aggregate-`pre` path (so they cannot yet appear as an aggregate input expression). `COUNT(DISTINCT col)` internally lowers through an `IS NOT NULL` filter to exclude NULLs from the distinct count.

## SELECT list

Each item is one of:

- `column`. Bare column reference. Carries the source column's name into the output.
- `*`. Expanded to one `Expr::Column(name)` per field of the FROM table (or, after a JOIN, the combined join schema).
- `expr AS alias`. The alias names the output column.
- `expr` (unnamed). Output column gets a synthetic name `__expr_<i>`.

If the query has any aggregate function in the SELECT list, OR a `GROUP BY` clause, the planner switches to aggregate mode:

- Every non-aggregate SELECT item must appear in `GROUP BY` (verified via structural equality).
- Aggregate inputs that aren't bare columns are routed through the `pre` kernel (which materialises the expression) then the standard reduction.
- Post-aggregate scalar expressions are accepted by the SQL frontend (0.5): `SUM(price) + 1`, `AVG(qty) * 2`, `(SUM(a) + SUM(b)) / 2`, and `SUM(x) + 1 AS total` (alias). The aggregates nested inside the expression are extracted as feed inputs (deduplicated by output name across the SELECT list), and the surface expression is rewritten with `Column("<aggregate_output_name>")` at each aggregate position — then evaluated by the post-Aggregate `Project`.

## Aggregate functions

| Function       | Output dtype                  | Notes                                                      |
|----------------|-------------------------------|------------------------------------------------------------|
| `COUNT(*)`     | `Int64`                       | Counts every row.                                          |
| `COUNT(expr)`  | `Int64`                       | Excludes NULLs via the validity bitmap (0.5).              |
| `COUNT(bool)`  | `Int64`                       | Honours nulls (host-side path).                            |
| `COUNT(utf8)`  | `Int64`                       | Honours nulls (host-side path).                            |
| `COUNT(date32 \| timestamp)` | `Int64`         | Counts non-NULL temporal rows; works end-to-end (dtype-agnostic count path). |
| `SUM(int|float)` | Same dtype as input         | `SUM(Int32) -> Int64` widening; SUM(Int64), SUM(Float*) unchanged. |
| `SUM(bool)`    | `Int64`                       | Count of `TRUE` rows.                                      |
| `MIN(int|float)` | Same dtype as input         | Float MIN via `atom.cas` loop on bit pattern (sm_70).      |
| `MIN(bool)`    | `Bool`                        | `FALSE < TRUE`. NULL if all-null group.                    |
| `MIN(utf8)`    | `Utf8`                        | Lexicographic; NULL if all-null group.                     |
| `MIN(date32 \| timestamp)` | Same dtype as input | GPU reduction on the i32/i64 storage; result rebuilt preserving the date / (unit + tz). NULL if all-null group. Scalar + GROUP BY. |
| `MIN(decimal)` | `Decimal128(p, s)`            | **Host-side** fold; preserves input `(p, s)`. NULL if all-null group. |
| `MAX(int|float)` | Same dtype as input         | Same caveats as MIN.                                       |
| `MAX(bool)`    | `Bool`                        | NULL if all-null group.                                    |
| `MAX(utf8)`    | `Utf8`                        | NULL if all-null group.                                    |
| `MAX(date32 \| timestamp)` | Same dtype as input | Same as `MIN` (temporal): GPU reduction, type/unit/tz preserved. Scalar + GROUP BY. |
| `MAX(decimal)` | `Decimal128(p, s)`            | **Host-side** fold; preserves input `(p, s)`.              |
| `AVG(numeric)` | `Float64`                     | Single fused kernel (on-device `SUM` + `COUNT` reduced together, divided on finalise); no longer split into separate host passes. |
| `AVG(bool)`    | `Float64`                     | Fraction of `TRUE` rows. NULL if all-null group.           |
| `SUM(decimal)` | `Decimal128`                  | **Host-side** reduction (0.7).                             |
| `STDDEV`, `STDDEV_POP`, `STDDEV_SAMP` | `Float64`  | **Host-side** Welford (0.5 scalar; 0.7 adds `GROUP BY` via per-group Welford). |
| `VARIANCE`, `VAR_POP`, `VAR_SAMP`     | `Float64`  | **Host-side** Welford, shared state with STDDEV. Scalar (0.5) + grouped (0.7). |

**Temporal MIN / MAX status.** `COUNT`, `MIN`, and `MAX` over a `Date32` / `Timestamp` column **all work end-to-end** as of the 0.7 wave. The reduction runs on the **GPU** over the normalised integer storage (`Date32 → Int32`, `Timestamp → Int64`), and the result is rebuilt as the original temporal type — a `Date32` for dates, and a `Timestamp` **preserving the unit and timezone** for timestamps (`src/exec/aggregate.rs`, `src/exec/groupby.rs`, and the temporal-output schema builder in `src/exec/schema_convert.rs` were all wired through). This holds in both the scalar and `GROUP BY` paths. `SUM` over a temporal column is undefined SQL and is **rejected by design** (`"SUM over Date32/Timestamp is not supported"`). (`MIN`/`MAX` over `Decimal128` are likewise supported, host-side, preserving the input precision/scale.)

**NULL / empty-input semantics.** The target behaviour is standard SQL, matching DuckDB: `MIN` / `MAX` / `SUM` / `AVG` over an all-NULL group **or an empty input** return SQL `NULL`, and `COUNT` returns `0`. This is the contract to rely on. (Historically there were two divergences the engine is converging away from: scalar `SUM` over an empty/all-NULL input returned `0` rather than `NULL`, and primitive `MIN` / `MAX` returned a type sentinel rather than `NULL`. The correct, documented behaviour is SQL `NULL`.) The Bool/Utf8 inputs (which thread validity through `extended_agg`) already return SQL `NULL` for an all-NULL group in both the scalar and GROUP BY paths. As of 0.5, scalar primitive aggregates honour validity: `COUNT(col)` excludes NULLs via the bitmap and `SUM`/`MIN`/`MAX`/`AVG` over `Int*`/`Float*` host-strip NULL positions before the GPU reduction (with a zero-copy fast path when `null_count == 0`).

`SUM` widens narrow integer inputs to the corresponding 64-bit type: `SUM(Int32) -> Int64`. `SUM(Int64)` and `SUM(Float32|Float64)` are unchanged. The widening is applied consistently in both the scalar and GROUP BY paths via `crate::plan::logical_plan::sum_output_dtype`.

Integer `SUM` overflow is a **hard error**, not silent wraparound and not undefined behaviour: if the running `i64` accumulator overflows, the query fails loudly with a `BoltError::Type("SUM(integer) overflow")`. The same applies to `SUM(Decimal128)` — an overflowing decimal sum errors rather than wrapping. (Float `SUM` follows IEEE-754 and saturates to `±inf` instead of erroring.) See [`LIMITATIONS.md`](LIMITATIONS.md) for the caveat on grouped-`SUM` overflow detection for streaming inputs the host cannot replicate.

`COUNT(DISTINCT col)` is supported as the **sole SELECT item** (no other columns or aggregates alongside it). It lowers to `COUNT(*) ∘ Distinct ∘ Project([col]) ∘ Filter(col IS NOT NULL)` (NULL-excluding distinct count, executed via the host-side `Distinct` executor). As of the 0.7 wave (F3), two combined forms over the bare sole-item distinct-count are now accepted and lower on top of the same base plan:

- **`SELECT DISTINCT COUNT(DISTINCT col)`** — the surrounding `DISTINCT` over a single-row result is a (correct) no-op, lowered to the existing `Distinct` node for plan uniformity.
- **`... HAVING <pred over COUNT(DISTINCT col)>`** with **no `GROUP BY`** — the whole table is one implicit group, so `HAVING` filters that single result row (standard SQL). The predicate may reference the distinct-count via the same `COUNT(DISTINCT col)` call it was written with (comparison / boolean / `IS [NOT] NULL` shapes).

As of the 0.7 wave (F3-finish), **`COUNT(DISTINCT col)` combined with `GROUP BY`** is also supported when it is the **sole aggregate**: `SELECT keys..., COUNT(DISTINCT col) FROM t [WHERE ...] GROUP BY keys [HAVING ...] [ORDER BY ...] [LIMIT ...]`. It runs as a host-orchestrated special-case — the engine materializes the WHERE-filtered `(keys..., col)` rows and counts the distinct non-NULL values of `col` per group-key tuple (NULL keys form their own group; an all-NULL group yields 0), then applies HAVING/ORDER BY/LIMIT. The count must be the last SELECT item and every GROUP BY key must be projected.

Still **rejected**: `COUNT(DISTINCT col)` with `GROUP BY` *alongside another aggregate* or with more than one `COUNT(DISTINCT)`, `COUNT(DISTINCT *)`, `COUNT(DISTINCT a, b)`, `COUNT(DISTINCT col) OVER (...)`, and `DISTINCT` inside any other aggregate. Aggregate aliasing (`SUM(price) AS total`) is supported in v0.5: the alias renames the aggregate's plan-assigned name (e.g. `sum_price`) in a post-Aggregate Project, and the alias is visible to `HAVING` and `ORDER BY`.

## GROUP BY

```
GROUP BY <expr_list>
```

Supported key shapes:

- **Single column**: any of `Int32`, `Int64`, `Float32`, `Float64`.
- **Two columns** whose combined width fits in 64 bits:
  - `(Int32, Int32)`, `(Int32, Float32)`, `(Float32, Float32)`. Packed into one i64 host-side.
- **Three or more columns**, or pairs wider than 64 bits (e.g. `(Int64, Int64)`): host-side reduction fallback. Correct but doesn't use the GPU hash table.

Float keys: bitwise grouping AFTER signed-zero canonicalisation (`canonicalise(f).to_bits() as i64`, where `canonicalise(x) = if x == 0.0 { 0.0 } else { x }`). `-0.0` and `+0.0` collapse to ONE group (matches SQL/IEEE and DuckDB — review C12). DISTINCT and JOIN apply the same canonicalisation so the three operators agree on float equivalence. NaN bit patterns are LEFT AS-IS (`NaN != NaN` per IEEE/SQL standard; DuckDB does the same), so different NaN payloads still group separately. The classic GPU path rejects keys that encode to `i64::MIN`; after canonicalisation `-0.0` packs to `+0.0` (bits `0`), removing the historical sentinel collision for the headline case. The sentinel-free `groupby_valid` path remains the safety net for any other dtype where the encoded bits happen to collide with `i64::MIN`.

Utf8 keys: not yet supported. Would need a dictionary-aware GROUP BY codegen path.

Output ordering: groups are sorted by encoded key for determinism. (Float bit-pattern order, which is NOT the same as numeric order — `-1.0` sorts after `+0.0` in output. Acceptable for a deterministic-but-not-ANSI v1; an explicit `ORDER BY` re-sorts the final result.)

## HAVING

`HAVING <bool_expr>` is supported when the query is in aggregate mode (i.e. has a `GROUP BY` and / or an aggregate in the SELECT list). The predicate may reference:

- Group-key columns by name.
- Aggregate function *calls* (`HAVING SUM(price) > 100`, `HAVING COUNT(*) > 1`) — the lowerer rewrites these into references to the aggregate's plan-assigned output column (`sum_price`, `count`, etc.).
- Arithmetic, comparison, and logical operators over the above.

`HAVING` without `GROUP BY` or any aggregate in the SELECT list is rejected with a clear error.

## ORDER BY

`ORDER BY <expr> [ASC|DESC] [NULLS FIRST|NULLS LAST] [, ...]` is supported and sits *above* the projection / aggregate / HAVING in the plan tree, so it sees the final output schema. Defaults:

- Direction defaults to `ASC`.
- NULL placement defaults to `NULLS FIRST` for `ASC` and `NULLS LAST` for `DESC` (SQL convention).

`ORDER BY ... WITH FILL` is rejected. As of 0.7 a **GPU** radix sort backs single-key `Int32` / `Int64` orderings (ASC, plus multi-key and `DESC`); other key shapes and dtypes fall back to the **host-side** sort executor (`src/exec/sort.rs`).

## LIMIT and OFFSET

`LIMIT <int>` and `OFFSET <int>` are both supported and fold into a single `Limit` node, so a downstream executor can implement the offset as a skip. Either clause alone is legal; `OFFSET` without `LIMIT` is represented as `Limit { limit: usize::MAX, offset }`. The argument must be a non-negative integer literal — `LIMIT -1`, `LIMIT 1.5`, and `LIMIT <expr>` are all rejected.

## Set operations (UNION / EXCEPT / INTERSECT)

`q1 UNION ALL q2 [UNION ALL q3 ...]` lowers to a single flat `Union { inputs }` node (left-recursive chains of the same quantifier are flattened, so a three-way union is one 3-input node, not nested binary trees).

`q1 UNION q2` (no `ALL`) lowers to `Distinct(Union { inputs })`, matching SQL's set-union semantics.

`q1 EXCEPT [ALL] q2` and `q1 INTERSECT [ALL] q2` are supported and lower to a binary `LogicalPlan::SetOp` node executed **host-side** by `src/exec/setops.rs`. The set forms (`EXCEPT` / `INTERSECT`, no `ALL`) return distinct left rows; the multiset forms (`EXCEPT ALL` → `max(0, lc - rc)` copies; `INTERSECT ALL` → `min(lc, rc)` copies) follow the SQL-standard multiplicity rules. Row equality reuses the `DISTINCT` executor's row-key machinery, so two NULLs in the same column position compare **equal** (the engine-wide "NULLs are not distinct" convention) and `+0.0` / `-0.0` canonicalise to one key. Chains are left-associative (`a EXCEPT b EXCEPT c` = `(a EXCEPT b) EXCEPT c`); they are not flattened.

`UNION BY NAME` (and `EXCEPT` / `INTERSECT BY NAME`) are rejected. `ORDER BY` / `LIMIT` / `OFFSET` applied to a set operation apply to the combined result, not the individual branches.

## Common table expressions (WITH)

`WITH name AS (<query>) [, name2 AS (...)] <body>` is supported for **non-recursive** CTEs. Each CTE is lowered against the scope of the CTEs that precede it (standard left-to-right visibility) and type-checked eagerly at its definition site. A CTE name is referenced from `FROM` exactly like a base table. Nested subqueries may reference an in-scope CTE.

Rejected: `WITH RECURSIVE` (only non-recursive CTEs), CTE column-list aliases (`WITH c (a, b) AS ...`), the CTE materialization hint, and a duplicate CTE name in the same `WITH` clause.

## Window functions

`func(...) OVER (PARTITION BY ... ORDER BY ...)` is supported, executed **host-side** by `src/exec/window.rs` (the executor needs a global partition + ordering view the per-scan GPU kernels can't express yet). Supported functions:

- **Ranking**: `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()` (no argument).
- **Aggregate windows**: `SUM`, `AVG`, `MIN`, `MAX`, `COUNT` over a single bare column argument.

A window function must appear as a **top-level SELECT item** (optionally aliased); a window call nested inside a larger expression is rejected with a clear message. `PARTITION BY` / `ORDER BY` / the aggregate argument must each be a bare column reference (computed keys are rejected host-side). Window partition / order key dtypes supported host-side: `Int32` / `Int64` / `Float32` / `Float64` / `Bool` / `Utf8` / `Date32` / `Timestamp(ns)`; aggregate-input columns must be numeric (`Int32` / `Int64` / `Float32` / `Float64`).

**Frame**: only the SQL default `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW` is implemented. Under this frame every ordering peer (rows with equal `ORDER BY` keys) sees the same running aggregate; with no `ORDER BY` the whole partition is one peer group, so the aggregate windows report the full-partition value on every row. Explicit non-default frames (custom bounds, `GROUPS`, anything other than `UNBOUNDED PRECEDING [AND CURRENT ROW]`) are rejected.

Rejected: `QUALIFY`, the named `WINDOW` clause, `OVER <named_window>`, `COUNT(DISTINCT ...) OVER (...)`, `FILTER` / `IGNORE NULLS` / `WITHIN GROUP` on a window function, and lead/lag/value functions (`LAG`, `LEAD`, `FIRST_VALUE`, `NTILE`, etc.).

## Subqueries

**Uncorrelated** subqueries in `SELECT` and `WHERE` are supported and resolved to constants *before* physical lowering (`src/exec/subquery_resolve.rs`):

- **Scalar** `(SELECT ...)` — the subquery must produce a single column; 0 rows folds to SQL `NULL`, 1 row to that value, and `>1` row is a clean error.
- **`<expr> [NOT] IN (SELECT ...)`** — the single-column result set is folded into an `OR`/`AND` chain of equalities over `expr` (`expr = v1 OR …` / `expr <> v1 AND …`). NULLs are dropped from the value set; this matches strict SQL exactly for `IN` under `WHERE`, and diverges only for `NOT IN` against a set containing NULLs (documented in the module).

An uncorrelated **scalar subquery** is also accepted in `ORDER BY` (it folds to a constant before physical lowering, exactly like a SELECT / WHERE scalar subquery).

Nested subqueries resolve inner-first. **Correlated** subqueries (any reference to an outer column) are detected and rejected with a precise message. `EXISTS` / `NOT EXISTS` are rejected.

### Derived tables (subquery in FROM)

A subquery may appear as a FROM item — `FROM (SELECT ...) AS alias` — as of 0.7. The subquery is planned recursively as a self-contained subtree and exposed under the alias (the same pipeline a CTE reference uses; a CTE is just a named, pre-lowered derived table), so qualified `alias.col` references resolve. Restrictions:

- The **alias is required** (standard SQL — `(SELECT ...)` with no alias is rejected).
- **`LATERAL`** derived tables are rejected: they are correlated (may reference earlier FROM items) and the engine has no correlated-execution path.
- A **column-list alias** (`AS d(x, y)`) is rejected.

## INNER JOIN

```
SELECT <select_list>
  FROM <lhs_table> INNER JOIN <rhs_table> ON <equi_predicate>
 [WHERE ...] [GROUP BY ...] ...
```

Supported:

- Multiple joins per `SELECT` are supported: the frontend folds each `JOIN` in FROM order into a left-deep chain of `Join` nodes (see `sql_frontend.rs`, the `for join in &twj.joins` loop). Note: joins execute in written order — there is no cost-based join reordering yet (a conservative reorder pass exists but is a no-op until table statistics are wired in).
- `ON` predicate must be a conjunction (`AND` only) of `<col> = <col>` equalities. Either side may be a bare, table-qualified, or **schema-qualified** column reference (`a = a`, `t1.a = t2.a`, or `schema.t1.a = schema.t2.a` — the leading single-catalog segment is dropped); only the trailing column name survives lowering and the executor matches it against each side's schema. Four-or-more-segment references have no namespace to collapse and are rejected.
- The executor runs on the GPU hash-join path (`src/exec/gpu_join.rs`): build a hash table on the smaller input on-device, probe the larger, then materialise matches via `arrow::compute::take` on the host. Multi-key joins build a tuple key. Equi-join key dtypes must match on both sides; cross-dtype keys are rejected.
- `NULL` keys never match (`NULL = NULL → UNKNOWN`, per SQL).
- The combined output schema is left's columns followed by right's columns, with collision-safe naming: a clashing right-side `c` becomes `right.c` (and gets a `__2`, `__3`, … suffix if that itself collides).

All join shapes have a GPU fast path *and* a host fallback. The dispatch lives in `src/exec/join.rs`, which tries `src/exec/gpu_join.rs` first and falls through to a host hash-join on any gate miss or kernel decline:

- `INNER` — GPU path requires a single `Int32`/`Int64` equi-key, both sides large enough, no NULL keys, unique build keys (`try_gpu_inner_join`); otherwise host hash join. A non-equi INNER predicate (small cardinality) runs through `execute_nested_loop_join` (host, inner side capped at 1024 rows).
- `LEFT` / `RIGHT` / `FULL [OUTER]` — Stage-2 GPU fast path (`try_gpu_outer_join`); host hash join on a gate miss.
- `CROSS` — Stage-3 GPU fast path for cell counts within a bounded window (`execute_cross_join_on_gpu`); host cartesian product otherwise.

`JOIN ... USING (c1, ...)` and `NATURAL JOIN` are supported: each is desugared to equi `<left.col> = <right.col>` pairs (`USING` over the named columns; `NATURAL` over every column common to both sides) and then runs the same join paths as an explicit `ON`. A `USING` column that is missing, ambiguous, or duplicated — and a `NATURAL JOIN` with no common column — is rejected with a clear message.

Still rejected: a JOIN with no `ON` / `USING` / `NATURAL` constraint (other than `CROSS`), non-equi `ON` predicates with arbitrary cardinality, computed join keys (`ON l.a + 1 = r.b`), and `GLOBAL JOIN` (ClickHouse extension).

## Dictionary-encoded Utf8 predicates

For every `Utf8` column registered on a table, the engine builds a dictionary (i32-indexed by default, i64 above the cardinality threshold) at `register_table` time. The `StringPredicateRewriter` then folds, at plan time:

- `WHERE col = 'X'`  →  `WHERE __idx_col = i32/i64(idx_of_X)`
- `WHERE col != 'X'` →  the same with `!=`

After the rewrite the predicate is pure integer equality, which the standard codegen already handles. Literals not present in the dictionary collapse to a constant-false predicate. `IN (...)` against a Utf8 column is still *not* folded through the dictionary rewriter (it defers this shape — see `src/plan/string_literal_rewrite.rs`); rewrite as an `OR` chain of literal equalities. `LIKE` on a Utf8 column *is* supported and, as of 0.7, **lowers to the GPU** (dictionary-precompute → index membership for dictionary columns, or the `StringLikeFilter` device matcher for non-dictionary `Utf8`), with a host-side `host_like` fallback (see the LIKE section above). Ordering comparisons (`<`, `>`, `<=`, `>=`) of a Utf8 column against a string *literal* are also folded as of 0.7, via a **binary (UTF-8 byte) collation** precompute that partitions the dictionary by the literal and emits the same index-membership form (**GPU**; not locale/ICU collation). Column-vs-column Utf8 ordering remains a host string comparison.

## SELECT DISTINCT

`SELECT DISTINCT <select_list> FROM ...` is supported and dedups the *output* rows (after projection, HAVING, and any aggregate work). The executor is host-side (`src/exec/distinct.rs`).

`DISTINCT ON (...)` (Postgres extension) is rejected. `COUNT(DISTINCT col)` *is* supported as the sole SELECT item (see [Aggregate functions](#aggregate-functions)).

## Expression examples

These all work today against a table with `region_id: Int32`, `price: Float64`, `tax: Float64`, `region: Utf8`, `active: Bool`:

```sql
-- Projection / filter
SELECT price FROM sales;
SELECT price, tax FROM sales;
SELECT price * tax FROM sales;
SELECT price * tax + price FROM sales;
SELECT price FROM sales WHERE region_id = 1;
SELECT price * tax FROM sales WHERE region_id = 1 AND price > 100.0;
SELECT * FROM sales;
SELECT region, price FROM sales WHERE region = 'US';
SELECT region_id, price FROM sales WHERE region_id IN (1, 2, 3);       -- desugars to OR chain (GPU)
SELECT price FROM sales WHERE price BETWEEN 10.0 AND 100.0;            -- desugars to >= AND <= (GPU)
SELECT CASE WHEN price > 100 THEN 1 ELSE 0 END FROM sales;            -- numeric CASE (GPU)
SELECT CAST(region_id AS FLOAT8) FROM sales;                         -- numeric CAST (GPU)
SELECT region || '-' || CAST(region_id AS VARCHAR) FROM sales;       -- || concat (host-side Project)
SELECT name FROM sales WHERE name LIKE 'A%';                         -- LIKE (GPU, host fallback)
SELECT region FROM sales WHERE region < 'M';                         -- Utf8 col < literal (GPU, byte collation)
SELECT region FROM sales WHERE region >= 'US';                       -- Utf8 ordering vs literal (GPU)
SELECT * FROM sales WHERE active;

-- Scalar aggregates
SELECT SUM(price) FROM sales;
SELECT SUM(price * tax) FROM sales;
SELECT SUM(price * tax) FROM sales WHERE region_id = 1;
SELECT COUNT(*) FROM sales;
SELECT COUNT(*) FROM sales WHERE region = 'US';

-- GROUP BY
SELECT region_id, SUM(price), AVG(tax) FROM sales GROUP BY region_id;
SELECT region_id, SUM(price * tax) FROM sales WHERE active GROUP BY region_id;
SELECT a, b, SUM(v) FROM events GROUP BY a, b;             -- 2-col Int32 packed key
SELECT a, b, c, SUM(v) FROM events GROUP BY a, b, c;       -- 3-col wide-key fallback (host)
SELECT MIN(temp), MAX(temp) FROM weather GROUP BY station; -- Float MIN/MAX via CAS

-- HAVING
SELECT region_id, COUNT(*) FROM sales GROUP BY region_id HAVING COUNT(*) > 10;
SELECT region_id, SUM(price) FROM sales GROUP BY region_id HAVING SUM(price) > 1000.0;

-- DISTINCT / ORDER BY / LIMIT / OFFSET
SELECT DISTINCT region FROM sales;
SELECT region_id FROM sales ORDER BY region_id;
SELECT region_id FROM sales ORDER BY region_id DESC NULLS FIRST;
SELECT * FROM sales LIMIT 100;
SELECT * FROM sales LIMIT 100 OFFSET 50;
SELECT region_id, SUM(price)
  FROM sales GROUP BY region_id ORDER BY region_id LIMIT 10;

-- Set operations
SELECT region FROM sales UNION ALL SELECT region FROM sales_archive;
SELECT region FROM sales UNION     SELECT region FROM sales_archive;  -- dedups
SELECT region FROM sales EXCEPT    SELECT region FROM sales_archive;  -- host-side
SELECT region FROM sales INTERSECT SELECT region FROM sales_archive;  -- host-side

-- CTE (WITH), non-recursive
WITH us_sales AS (SELECT * FROM sales WHERE region = 'US')
SELECT region_id, SUM(price) FROM us_sales GROUP BY region_id;

-- Uncorrelated subqueries
SELECT region_id FROM sales WHERE price > (SELECT AVG(price) FROM sales);
SELECT * FROM sales WHERE region_id IN (SELECT id FROM active_regions);

-- COUNT(DISTINCT) — sole SELECT item only
SELECT COUNT(DISTINCT region) FROM sales;
SELECT DISTINCT COUNT(DISTINCT region) FROM sales;                   -- F3: SELECT DISTINCT over the sole count (no-op)
SELECT COUNT(DISTINCT region) FROM sales HAVING COUNT(DISTINCT region) > 5;  -- F3: HAVING, no GROUP BY
-- (COUNT(DISTINCT region) ... GROUP BY region_id  is still rejected)

-- Temporal MIN/MAX (GPU; type + unit + tz preserved)
SELECT MIN(order_date), MAX(order_date) FROM orders;                  -- Date32
SELECT station, MIN(ts), MAX(ts) FROM readings GROUP BY station;      -- Timestamp, grouped

-- Window functions (host-side, default frame)
SELECT region_id, ROW_NUMBER() OVER (PARTITION BY region_id ORDER BY price) FROM sales;
SELECT region_id, SUM(price) OVER (PARTITION BY region_id) AS region_total FROM sales;

-- String functions
SELECT UPPER(region), LENGTH(region) FROM sales;          -- UPPER/LENGTH (GPU)
SELECT SUBSTRING(region FROM 1 FOR 2) FROM sales;         -- SUBSTRING (host-realized StringProject)
SELECT TRIM(region) FROM sales;                           -- single-arg TRIM (host-realized StringProject)
SELECT CONCAT(region, '-', name) FROM sales;              -- CONCAT, NULL-if-any-arg-NULL (host mirror)

-- CASE producing a string (host-realized over a bare scan; SQL 3VL)
SELECT CASE WHEN active THEN 'on' ELSE 'off' END AS state FROM sales;

-- Derived table (subquery in FROM) — alias required, non-lateral
SELECT d.region_id, d.n
  FROM (SELECT region_id, COUNT(*) AS n FROM sales GROUP BY region_id) AS d
 WHERE d.n > 10;

-- Scalar subquery in ORDER BY
SELECT region_id FROM sales ORDER BY (SELECT AVG(price) FROM sales);

-- Schema-qualified column in JOIN ON (leading catalog segment dropped)
SELECT s.region_id, c.name
  FROM sales INNER JOIN customers ON public.sales.customer_id = public.customers.id;

-- INNER JOIN (ON / USING / NATURAL)
SELECT s.region_id, c.name
  FROM sales INNER JOIN customers ON sales.customer_id = customers.id;
SELECT * FROM sales INNER JOIN customers USING (customer_id);
SELECT * FROM sales NATURAL JOIN customers;
SELECT *
  FROM orders INNER JOIN line_items
    ON orders.id = line_items.order_id AND orders.region = line_items.region;
```

## JOIN

```
SELECT ... FROM <table>
  [{INNER | LEFT [OUTER] | RIGHT [OUTER] | FULL [OUTER]} JOIN <table>
        {ON <equi_predicate> | USING (<col>, ...) | NATURAL}]
  [CROSS JOIN <table>]
  ...
```

- The ON predicate is a conjunction of `left.col = right.col` equalities. Non-equi predicates and non-conjunctive shapes are rejected. `USING (...)` and `NATURAL` desugar to the same equi-pair form.
- `CROSS JOIN` has no ON clause. The output row count is `|left| × |right|`; rewrite your query if it would exceed the engine's `u32::MAX`-row materialisation limit (an explicit `BoltError::Plan` surfaces at execute time when it would).
- For `LEFT` / `RIGHT` / `FULL [OUTER]`, columns coming from the *non-preserved* side are marked nullable in the output schema. Unmatched preserved-side rows emit with NULLs in those columns.
- Right-side column names that collide with a left-side name are prefixed with `right.` (e.g. left `id` and right `id` → output has `id` and `right.id`).
- Both sides of an equi-join key must have the same dtype; cross-dtype equi-joins (e.g. `Int32 = Int64`) are rejected.
- SQL NULL semantics on keys: `NULL = NULL` is `UNKNOWN`, so NULL-keyed rows never match. For OUTER joins they still emit on the preserved side with the opposite side NULL-padded.
- Every shape has a GPU fast path with a host fallback (`src/exec/join.rs` dispatches to `src/exec/gpu_join.rs`): `INNER` (`try_gpu_inner_join`), `LEFT`/`RIGHT`/`FULL` (`try_gpu_outer_join`), and `CROSS` (`execute_cross_join_on_gpu`). The GPU paths are gated (dtype, NULL-freedom, cardinality / cell-count windows); a gate miss or kernel decline transparently falls back to the host hash-join (or host cartesian product for CROSS).

## What's NOT supported

These produce explicit errors at parse / plan time:

### Joins beyond the supported set
- Non-equi `ON` predicates (`>`, `<`, function calls, `BETWEEN`, range joins) with arbitrary cardinality. (A small-cardinality non-equi INNER predicate runs through a capped host nested-loop.)
- Computed join keys (`ON l.a + 1 = r.b`).
- `GLOBAL JOIN` (ClickHouse extension).
- (`NATURAL JOIN`, `JOIN ... USING`, and multiple joins per `SELECT` are now **supported** — see the [JOIN](#join) section.)

### Query composition
- **Correlated** subqueries and `EXISTS` / `NOT EXISTS`. (Uncorrelated scalar and `[NOT] IN` subqueries are **supported** in SELECT / WHERE, an uncorrelated scalar subquery is also accepted in ORDER BY, and non-lateral **derived tables** `(SELECT ...) AS alias` are **supported** — see [Subqueries](#subqueries). `LATERAL` derived tables and column-list aliases `AS d(x, y)` remain rejected.)
- `WITH RECURSIVE`, CTE column-list aliases. (Non-recursive CTEs are **supported** — see [Common table expressions](#common-table-expressions-with).)
- `UNION BY NAME` (and `EXCEPT` / `INTERSECT BY NAME`). (`EXCEPT [ALL]` / `INTERSECT [ALL]` are **supported** — see [Set operations](#set-operations-union--except--intersect).)
- `QUALIFY`, the named `WINDOW` clause, `OVER <named_window>`, and non-default window frames. (`OVER (...)` with `ROW_NUMBER` / `RANK` / `DENSE_RANK` / `SUM` / `AVG` / `MIN` / `MAX` / `COUNT` under the default frame is **supported** — see [Window functions](#window-functions).)

### Expressions

The following are supported now (see the Operators section above for the
execution tier of each) and are no longer in this list: `CAST`, `CASE`,
`COALESCE`, `NULLIF`, `IN (...)`, `BETWEEN`, `LIKE`, `IS NULL` / `IS NOT
NULL`, `NOT`, the `||` concat operator, qualified column references
(`t.col`), and post-aggregate expressions (`SUM(price) + 1`,
`SUM(a) / SUM(b)`).

Still rejected (or only partially lowered):

- `IFNULL`, `IIF` — not parsed (use `COALESCE` / `CASE`).
- `IN (...)` against a `Utf8` column — not folded through the dictionary rewriter; rewrite as an `OR` chain of literal equalities.
- Ordering comparisons of **two Utf8 columns** (`WHERE a < b`). (A Utf8 column vs a string *literal* — `WHERE name < 'M'` — *is* supported as of 0.7 via byte-collation folding; see the Comparison section.)
- Multi-level qualified column references with four or more segments (`catalog.db.t.col`, struct-field access). Single-level `t.col` *and* the schema-qualified `schema.table.col` form (in SELECT/WHERE/GROUP BY/HAVING and `JOIN ... ON`, leading catalog segment dropped) *are* supported — see "Hard restrictions" above.
- The "parses; GPU lowering pending" cases below.

(`COUNT(DISTINCT col)` is now supported as the sole SELECT item — see [Aggregate functions](#aggregate-functions).)

### Parses but GPU lowering pending

These type-check but the physical layer rejects them at the GPU lowering boundary (a clear `"… not yet lowered to GPU"` error, not a silent fallback):

- `CAST` between **Float and `Decimal128`** (a correct round-to-nearest i128 scaling is not expressible on the fixed `cvt` path — CAST through an integer or an intermediate decimal), and `CAST` to or from `Timestamp` / `String`. (`CAST` integer↔`Decimal128`, `Decimal128`↔`Decimal128` rescale, and integer→`Date32` **do** lower to GPU as of 0.7 — see the CASE / CAST section.)
- `CASE` whose unified result dtype is **`Decimal128`** (no `selp.b128` register class, and no host realisation yet). (A numeric / `Bool` / `Date32` / `Timestamp` result lowers to the GPU `selp` fold; a `Utf8` result is **host-realized** via `StringProject` over a bare scan — see the CASE / CAST section.)
- GPU lowering of `NOT` in a predicate (runs host-side instead).
- `SUBSTRING`, single-arg `TRIM`, and the `CONCAT` scalar function: these **execute end-to-end** but on a **host-side** projection rather than the GPU (so this is a host-execution tier, not a hard rejection). As of the 0.7 wave, `SUBSTRING(col FROM start [FOR len])` (literal start/length) and the single-argument `TRIM` / `LTRIM` / `RTRIM` (default-whitespace) over a **bare `Utf8` scan** lower to the host-realized two-pass `PhysicalPlan::StringProject` producer; a custom trim-character set (`TRIM(chars FROM col)`) or computed `SUBSTRING` arguments fall back to the host `Project` evaluator. `CONCAT`'s dedicated GPU two-pass kernels exist and are PTX-shape-tested, but the executor uses the byte-identical host mirror for now (device launch wiring pending). The `||` concat operator likewise runs host-side. (`UPPER` / `LOWER` / `LENGTH` *do* lower to GPU as of 0.7 — see "String functions" below.)
- `Decimal128` division (`/`) **now lowers to GPU** as of 0.7 (`Op::Div128`), so it is no longer in this list — see the "Decimal128 arithmetic" subsection.

### Types and values
- Time-of-day / general interval (beyond Day-`INTERVAL` on dates) literals and arithmetic. `Date32`, `Timestamp`, and `Decimal128` *are* supported (see Data types).
- Array / list / struct / map types.

### Clauses and statements
- `LIMIT <expr>` (must be an integer literal); `LIMIT BY` (ClickHouse).
- `FETCH`, `FOR UPDATE/SHARE`, `INTO`, `LATERAL`, table-valued functions, `PREWHERE`, `CONNECT BY`, `CLUSTER / DISTRIBUTE / SORT BY`, `SETTINGS`, `FORMAT`.
- `GROUP BY ALL`, `ROLLUP`, `CUBE`, `TOTALS`.
- `SELECT AS STRUCT/VALUE`.
- DDL of any kind — no `CREATE TABLE`. Tables are registered via the Rust API (`Engine::register_table`).
- DML (`INSERT`, `UPDATE`, `DELETE`).

### Validity propagation
- Scalar primitive aggregates honour validity as of 0.5: `COUNT(col)` excludes NULLs via the bitmap, and `SUM`/`MIN`/`MAX`/`AVG` host-strip NULL positions before the GPU reduction (the zero-null fast path stays a zero-copy upload). The Bool/Utf8 `extended_agg` path also honours nulls. Full per-row NULL propagation through `CASE` branches on the GPU is still a follow-up (a CASE that fires no WHEN currently yields a deterministic zero rather than SQL NULL).

## String functions

`UPPER`, `LOWER`, `LENGTH`, `SUBSTRING`, `CONCAT`, and `TRIM` are surfaced through the SQL frontend via `Expr::ScalarFn` and **execute end-to-end** as of 0.7 (see also the "Additional scalar string functions" table below for `CHAR_LENGTH` / `OCTET_LENGTH` / `POSITION` / `REPLACE` / `LEFT` / `RIGHT` / `LPAD` / `RPAD` / `REVERSE` / `INITCAP`):

- **`UPPER` / `LOWER`** lower to the **GPU** via the two-pass `PhysicalPlan::StringProject` executor (variable-width device output).
- **`LENGTH`** lowers to the **GPU** via `PhysicalPlan::StringLength` (dictionary-gather, `Int64` output).
- **`SUBSTRING` / `TRIM`** execute **host-side** end-to-end. As of the 0.7 wave, `SUBSTRING(col FROM start [FOR len])` (with **integer-literal** start/length) and the **single-argument** `TRIM` / `LTRIM` / `RTRIM` (`TRIM BOTH` / `LEADING` / `TRAILING`, default whitespace) over a **bare `Utf8` scan** lower to the host-realized two-pass `PhysicalPlan::StringProject` producer (one device-shaped pass mirrored on the host). A **custom trim-character set** (`TRIM(chars FROM col)`) or a computed (non-literal) `SUBSTRING` argument falls back to the host `Project` evaluator. Both are byte/character-correct and NULL-propagating.
- **`CONCAT(a, b, ...)`** of `Utf8` columns executes end-to-end with **NULL-if-any-argument-NULL** semantics (standard SQL — a NULL in any source row makes the output row NULL). The dedicated N-input two-pass GPU producer kernels (`compile_concat_len_pass` / `compile_concat_write_pass`, supporting up to `CONCAT_MAX_INPUTS = 8` source columns) are implemented and PTX-shape-tested; the executor currently realises the result via the **byte-identical host mirror** (`string_project::host_concat_strings`), so results are correct and the device launch wiring is a follow-up. Arities beyond 8 source columns, computed/literal arguments, and non-Utf8 arguments take the host fallback (`string_ops_extended::concat`). The `||` concat operator likewise runs host-side.

### Additional scalar string functions

The following functions are surfaced through the SQL frontend via `Expr::ScalarFn`
and execute **host-side** (no GPU producer; same path as `SUBSTRING` / `TRIM`).
All are **character-based** (operate on Unicode codepoints, never bytes) and
**NULL-propagating** (a NULL in any argument yields a NULL result).

| Function | Result | Semantics |
|----------|--------|-----------|
| `CHAR_LENGTH(s)` / `CHARACTER_LENGTH(s)` | `Int64` | Synonym for the character-based `LENGTH`. `CHAR_LENGTH('héllo') = 5`. |
| `OCTET_LENGTH(s)` | `Int64` | UTF-8 **byte** length. `OCTET_LENGTH('héllo') = 6` (the `é` is two bytes). |
| `POSITION(substr IN s)` / `STRPOS(s, substr)` | `Int64` | 1-based **character** index of the first occurrence of `substr` in `s`, or `0` if absent. The empty substring is found at position `1`. `POSITION('llo' IN 'héllo') = 3`. |
| `REPLACE(s, from, to)` | `Utf8` | Replace every occurrence of `from` in `s` with `to`. An empty `from` returns `s` unchanged (PostgreSQL/DuckDB). |
| `LEFT(s, n)` | `Utf8` | First `n` characters of `s`. Negative `n` drops the last `|n|` characters (`LEFT('abcde', -2) = 'abc'`). |
| `RIGHT(s, n)` | `Utf8` | Last `n` characters of `s`. Negative `n` drops the first `|n|` characters (`RIGHT('abcde', -2) = 'cde'`). |
| `LPAD(s, len, pad)` | `Utf8` | Left-pad `s` to `len` characters using `pad`; if `s` is longer than `len` it is truncated to the first `len` characters. An empty `pad` only truncates. |
| `RPAD(s, len, pad)` | `Utf8` | Right-pad `s` to `len` characters using `pad`; truncation behaviour matches `LPAD`. |
| `REVERSE(s)` | `Utf8` | Reverse the characters of `s` (multibyte codepoints preserved). `REVERSE('héllo') = 'olléh'`. |
| `INITCAP(s)` | `Utf8` | Capitalise the first letter of each word and lower-case the rest. A "word" is a maximal run of alphanumeric characters. `INITCAP('hi tHERE-bob') = 'Hi There-Bob'`. |

Case folding (`UPPER` / `LOWER` / `INITCAP`) uses Unicode default case mapping and is locale-invariant.

The underlying transformations live in `src/exec/string_ops` / `src/exec/string_ops_extended` / `src/exec/string_project`.

## Not yet supported (planned)

### Wider GPU sort coverage

As of 0.7 a GPU radix sort backs `ORDER BY` for single-key `Int32` / `Int64` orderings (ASC, plus multi-key and `DESC`). Other key dtypes — and the dedup step of `UNION` / `DISTINCT` — still round-trip through the host-side executors. Extending the radix path to more dtypes and backing DISTINCT/UNION dedup with it is the natural next step.

If you need any of the above for your use case, please open an issue describing the query and the use case.
