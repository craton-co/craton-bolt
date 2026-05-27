# SQL Reference

The exact subset of SQL Craton Bolt's frontend accepts. The grammar is built on top of [`sqlparser`](https://github.com/apache/datafusion-sqlparser-rs); anything outside this surface produces a clear `BoltError::Sql(...)` or `BoltError::Plan(...)` with the unsupported construct.

This document tracks the 0.3.0 release. For the JIT pipeline that lowers and executes these queries, see [`JIT_PIPELINE.md`](JIT_PIPELINE.md). For the gap to 0.4 / 1.0, see [`../ROADMAP.md`](../ROADMAP.md).

## Supported query shape

```
SELECT [DISTINCT] <select_list>
  FROM <table> [INNER JOIN <table> ON <equi_predicate>]
 [WHERE  <bool_expr>]
 [GROUP BY <expr_list>]
 [HAVING <bool_expr>]
 [ORDER BY <expr> [ASC|DESC] [NULLS FIRST|NULLS LAST] [, ...]]
 [LIMIT <int_literal>] [OFFSET <int_literal>]
```

Two queries can be combined with `UNION` or `UNION ALL`; the optional `ORDER BY` / `LIMIT` / `OFFSET` then apply to the combined result.

**Hard restrictions** (everything else returns a `BoltError`):

- Exactly one base table in `FROM`, optionally widened by **one** JOIN per `SELECT`: `INNER`, `LEFT [OUTER]`, `RIGHT [OUTER]`, or `FULL [OUTER]` with an equi `ON` predicate, or `CROSS JOIN` (no `ON`). All joins execute host-side. `JOIN USING`, `NATURAL JOIN`, non-equi predicates, computed join keys, and chaining more than one JOIN per `SELECT` are rejected at the parser. The equi `ON` predicate must be a conjunction of `<left.col> = <right.col>` equalities.
- No CTEs (`WITH`), no subqueries in `FROM` or `WHERE`, no correlated subqueries, no `EXISTS`.
- No `EXCEPT`, `INTERSECT`, `UNION BY NAME`.
- No `WINDOW`, `OVER`, `QUALIFY`, `LATERAL`, table-valued functions, `PREWHERE` (ClickHouse-ism), `CONNECT BY`, `CLUSTER / DISTRIBUTE / SORT BY`, `FETCH`, `FOR UPDATE/SHARE`, `INTO`.
- No `GROUP BY ALL`, `ROLLUP`, `CUBE`, `TOTALS`.
- No schema-qualified table names. Qualified column references (`t.col`) are rejected everywhere *except* inside a `JOIN ... ON` predicate (where they're used for cross-side disambiguation, and only the column name survives lowering).
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

Date, time, timestamp, decimal, interval, list, struct, map — none yet.

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

### Comparison

`=`, `<>` (also `!=`), `<`, `<=`, `>`, `>=`. Result dtype is `Bool`. Both operands must unify under the rules above.

For `Utf8` columns, only equality (`=`, `<>`, `!=`) and `IN`-shaped equality against string *literals* are supported — the literal rewriter folds `WHERE col = 'X'` (or `col IN ('X', 'Y')`) into integer equality on the column's dictionary index at plan time. Ordering comparisons (`<`, `>`, `<=`, `>=`) on Utf8 columns are rejected, because dictionary indices reflect insertion order, not lex order. `MIN(utf8_col)` and `MAX(utf8_col)` use lex order on the host.

### Logical

`AND`, `OR`. Both operands must be `Bool`. Result is `Bool`.

`NOT` is not yet supported — it would need a unary op in the AST.

## SELECT list

Each item is one of:

- `column`. Bare column reference. Carries the source column's name into the output.
- `*`. Expanded to one `Expr::Column(name)` per field of the FROM table (or, after a JOIN, the combined join schema).
- `expr AS alias`. The alias names the output column.
- `expr` (unnamed). Output column gets a synthetic name `__expr_<i>`.

If the query has any aggregate function in the SELECT list, OR a `GROUP BY` clause, the planner switches to aggregate mode:

- Every non-aggregate SELECT item must appear in `GROUP BY` (verified via structural equality).
- Aggregate inputs that aren't bare columns are routed through the `pre` kernel (which materialises the expression) then the standard reduction.
- Post-aggregate expressions (`SUM(price) + 1`) are not yet supported — emits a `BoltError::Sql("post-aggregate expressions not yet supported")`.

## Aggregate functions

| Function       | Output dtype                  | Notes                                                      |
|----------------|-------------------------------|------------------------------------------------------------|
| `COUNT(*)`     | `Int64`                       | Counts every row. No NULL exclusion yet.                   |
| `COUNT(expr)`  | `Int64`                       | Currently same as `COUNT(*)` for primitive inputs.         |
| `COUNT(bool)`  | `Int64`                       | Honours nulls (host-side path).                            |
| `COUNT(utf8)`  | `Int64`                       | Honours nulls (host-side path).                            |
| `SUM(int|float)` | Same dtype as input         | `SUM(Int32) -> Int64` widening; SUM(Int64), SUM(Float*) unchanged. |
| `SUM(bool)`    | `Int64`                       | Count of `TRUE` rows.                                      |
| `MIN(int|float)` | Same dtype as input         | Float MIN via `atom.cas` loop on bit pattern (sm_70).      |
| `MIN(bool)`    | `Bool`                        | `FALSE < TRUE`. NULL if all-null group.                    |
| `MIN(utf8)`    | `Utf8`                        | Lexicographic; NULL if all-null group.                     |
| `MAX(int|float)` | Same dtype as input         | Same caveats as MIN.                                       |
| `MAX(bool)`    | `Bool`                        | NULL if all-null group.                                    |
| `MAX(utf8)`    | `Utf8`                        | NULL if all-null group.                                    |
| `AVG(numeric)` | `Float64`                     | Split into `SUM + COUNT` on host.                          |
| `AVG(bool)`    | `Float64`                     | Fraction of `TRUE` rows. NULL if all-null group.           |

Aggregates over an all-NULL group (Bool/Utf8 inputs, which thread validity through `extended_agg`) return SQL `NULL` in both the scalar and GROUP BY paths. Primitive aggregates do not yet read a validity bitmap, so `SUM`/`MIN`/`MAX`/`AVG` over `Int*`/`Float*` treat every row as non-null (see "What's NOT supported"; tracked for 0.4).

`SUM` widens narrow integer inputs to the corresponding 64-bit type to prevent silent overflow: `SUM(Int32) -> Int64`. `SUM(Int64)` and `SUM(Float32|Float64)` are unchanged. The widening is applied consistently in both the scalar and GROUP BY paths via `crate::plan::logical_plan::sum_output_dtype`.

`DISTINCT` inside an aggregate (`COUNT(DISTINCT col)`) is not supported. Aggregate aliasing (`SUM(price) AS total`) is rejected by the SQL frontend — the plan auto-names aggregates and the SQL frontend doesn't carry the alias through.

## GROUP BY

```
GROUP BY <expr_list>
```

Supported key shapes:

- **Single column**: any of `Int32`, `Int64`, `Float32`, `Float64`.
- **Two columns** whose combined width fits in 64 bits:
  - `(Int32, Int32)`, `(Int32, Float32)`, `(Float32, Float32)`. Packed into one i64 host-side.
- **Three or more columns**, or pairs wider than 64 bits (e.g. `(Int64, Int64)`): host-side reduction fallback. Correct but doesn't use the GPU hash table.

Float keys: bitwise grouping (`f.to_bits() as i64`). Different NaN bit patterns group separately; `-0.0` and `+0.0` group separately. The classic GPU path rejects keys that encode to `i64::MIN` (notably `-0.0`); the engine falls back to the sentinel-free `groupby_valid` path which has no such restriction.

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

`ORDER BY ... WITH FILL` is rejected. The 0.3.0 executor is host-side; a GPU sort kernel is a 0.4 stretch goal.

## LIMIT and OFFSET

`LIMIT <int>` and `OFFSET <int>` are both supported and fold into a single `Limit` node, so a downstream executor can implement the offset as a skip. Either clause alone is legal; `OFFSET` without `LIMIT` is represented as `Limit { limit: usize::MAX, offset }`. The argument must be a non-negative integer literal — `LIMIT -1`, `LIMIT 1.5`, and `LIMIT <expr>` are all rejected.

## UNION and UNION ALL

`q1 UNION ALL q2 [UNION ALL q3 ...]` lowers to a single flat `Union { inputs }` node (left-recursive chains of the same quantifier are flattened, so a three-way union is one 3-input node, not nested binary trees).

`q1 UNION q2` (no `ALL`) lowers to `Distinct(Union { inputs })`, matching SQL's set-union semantics.

`UNION BY NAME`, `EXCEPT`, and `INTERSECT` are rejected. `ORDER BY` / `LIMIT` / `OFFSET` applied to a `UNION` apply to the combined result, not the individual branches.

## INNER JOIN

```
SELECT <select_list>
  FROM <lhs_table> INNER JOIN <rhs_table> ON <equi_predicate>
 [WHERE ...] [GROUP BY ...] ...
```

Supported:

- Exactly one `INNER JOIN` per `SELECT` (chaining a second `INNER JOIN` is rejected — the frontend builds at most one `Join` node and complains on a second).
- `ON` predicate must be a conjunction (`AND` only) of `<col> = <col>` equalities. Either side may be a bare or qualified column reference (`t1.a = t2.a` or `a = a`); only the trailing column name survives lowering and the executor matches it against each side's schema.
- The executor is host-side: build a `HashMap<JoinKey, Vec<row_idx>>` on the smaller input, probe the larger, materialise matches via `arrow::compute::take`. Multi-key joins build a tuple key.
- `NULL` keys never match (`NULL = NULL → UNKNOWN`, per SQL).
- The combined output schema is left's columns followed by right's columns, with collision-safe naming: a clashing right-side `c` becomes `right.c` (and gets a `__2`, `__3`, … suffix if that itself collides).

Rejected: `LEFT JOIN`, `RIGHT JOIN`, `FULL OUTER JOIN`, `CROSS JOIN`, `NATURAL JOIN`, `JOIN ... USING`, `INNER JOIN` without `ON`, non-equi predicates (`>`, `<`, function calls), and any join graph wider than one join per `SELECT`. A GPU-resident hash join is a 0.4 stretch goal.

## Dictionary-encoded Utf8 predicates

For every `Utf8` column registered on a table, the engine builds a dictionary (i32-indexed by default, i64 above the cardinality threshold) at `register_table` time. The `StringPredicateRewriter` then folds, at plan time:

- `WHERE col = 'X'`  →  `WHERE __idx_col = i32/i64(idx_of_X)`
- `WHERE col != 'X'` →  the same with `!=`
- `WHERE col IN ('X', 'Y', 'Z')` → a disjunction of those equalities (or a single one for a 1-element list)

After the rewrite the predicate is pure integer equality, which the standard codegen already handles. Literals not present in the dictionary collapse to a constant-false predicate. `LIKE`, `BETWEEN`, prefix / substring matching, and ordering comparisons on Utf8 are *not* supported.

## SELECT DISTINCT

`SELECT DISTINCT <select_list> FROM ...` is supported and dedups the *output* rows (after projection, HAVING, and any aggregate work). The executor is host-side (`src/exec/distinct.rs`).

`DISTINCT ON (...)` (Postgres extension) and `COUNT(DISTINCT col)` are rejected.

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
SELECT region, price FROM sales WHERE region IN ('US', 'CA');
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

-- UNION
SELECT region FROM sales UNION ALL SELECT region FROM sales_archive;
SELECT region FROM sales UNION     SELECT region FROM sales_archive;  -- dedups

-- INNER JOIN
SELECT s.region_id, c.name
  FROM sales INNER JOIN customers ON sales.customer_id = customers.id;
SELECT *
  FROM orders INNER JOIN line_items
    ON orders.id = line_items.order_id AND orders.region = line_items.region;
```

## JOIN

```
SELECT ... FROM <table>
  [{INNER | LEFT [OUTER] | RIGHT [OUTER] | FULL [OUTER]} JOIN <table> ON <equi_predicate>]
  [CROSS JOIN <table>]
  ...
```

- The ON predicate is a conjunction of `left.col = right.col` equalities. Non-equi predicates and non-conjunctive shapes are rejected.
- `CROSS JOIN` has no ON clause. The output row count is `|left| × |right|`; rewrite your query if it would exceed the engine's `u32::MAX`-row materialisation limit (an explicit `BoltError::Plan` surfaces at execute time when it would).
- For `LEFT` / `RIGHT` / `FULL [OUTER]`, columns coming from the *non-preserved* side are marked nullable in the output schema. Unmatched preserved-side rows emit with NULLs in those columns.
- Right-side column names that collide with a left-side name are prefixed with `right.` (e.g. left `id` and right `id` → output has `id` and `right.id`).
- Both sides of an equi-join key must have the same dtype; cross-dtype equi-joins (e.g. `Int32 = Int64`) are rejected.
- SQL NULL semantics on keys: `NULL = NULL` is `UNKNOWN`, so NULL-keyed rows never match. For OUTER joins they still emit on the preserved side with the opposite side NULL-padded.
- The executor is host-side (build smaller side into a HashMap, probe larger; for CROSS, a host-side cartesian product). GPU hash join is a 0.4 target.

## What's NOT supported

These produce explicit errors at parse / plan time:

### Joins beyond the supported set
- `NATURAL JOIN`.
- Non-equi `ON` predicates (`>`, `<`, function calls, `BETWEEN`, range joins).
- `JOIN ... USING (...)` (rewrite as `ON`).
- More than one JOIN per `SELECT` (chained joins).
- Computed join keys (`ON l.a + 1 = r.b`).

### Query composition
- Subqueries anywhere (`FROM`, `WHERE`, scalar, correlated, `EXISTS`).
- CTEs (`WITH`).
- `EXCEPT`, `INTERSECT`, `UNION BY NAME`.
- Window functions (`OVER`), `QUALIFY`.

### Expressions
- `CAST(... AS ...)` — the planner does implicit numeric promotion only.
- `CASE ... WHEN`, `NULLIF`, `COALESCE`, `IFNULL`, `IIF`.
- `LIKE`, `BETWEEN`, `IS NULL`, `IS NOT NULL`.
- `IN` other than the dictionary-equality form (i.e. `col IN ('lit', 'lit', ...)` against a Utf8 column).
- `NOT` (would need a unary op in the AST).
- String concatenation operator `||`.
- Ordering comparisons on Utf8 columns (`WHERE name < 'M'`).
- Schema-qualified or table-qualified column refs (`t.col`) outside `JOIN ... ON`.
- Aggregate aliasing (`SUM(price) AS total`).
- Post-aggregate expressions (`SUM(price) + 1`, `SUM(a) / SUM(b)`).
- `COUNT(DISTINCT col)`.

### Types and values
- Date / time / timestamp / interval literals and arithmetic.
- Decimal / fixed-point arithmetic.
- Array / list / struct / map types.

### Clauses and statements
- `LIMIT <expr>` (must be an integer literal); `LIMIT BY` (ClickHouse).
- `FETCH`, `FOR UPDATE/SHARE`, `INTO`, `LATERAL`, table-valued functions, `PREWHERE`, `CONNECT BY`, `CLUSTER / DISTRIBUTE / SORT BY`, `SETTINGS`, `FORMAT`.
- `GROUP BY ALL`, `ROLLUP`, `CUBE`, `TOTALS`.
- `SELECT AS STRUCT/VALUE`.
- DDL of any kind — no `CREATE TABLE`. Tables are registered via the Rust API (`Engine::register_table`).
- DML (`INSERT`, `UPDATE`, `DELETE`).

### Validity propagation
- Primitive aggregate kernels (`SUM`/`MIN`/`MAX`/`AVG` over `Int*`/`Float*`) do not yet read a validity bitmap; every row is treated as non-null. The Bool/Utf8 `extended_agg` path *does* honour nulls. Tracked for 0.4.

## Not yet supported (planned)

### String functions

`UPPER`, `LOWER`, `LENGTH`, `CONCAT`, `SUBSTRING` are reachable only from the executor-level `src/exec/string_ops` / `src/exec/string_ops_extended` Rust API; no SQL or DataFrame surface exposes them yet. They run as pure-host dictionary transformations because variable-width device writes remain unsupported by the codegen path. Wiring them through the SQL frontend would mean teaching `sql_frontend::lower_expr` to recognise `Expr::FunctionCall` and routing it to a per-function host-side projection executor. Listed as a 0.4 stretch goal in `ROADMAP.md`.

### GPU-resident JOIN

The 0.3.0 `INNER JOIN` executor is host-side (build `HashMap`, probe). A GPU-resident hash-join probe path is the natural next step (also a 0.4 stretch goal).

### GPU sort kernel

`ORDER BY` and the dedup step of `UNION` / `DISTINCT` currently round-trip through host code. A GPU sort kernel would back all three (0.4 stretch goal).

If you need any of the above for your use case, please open an issue describing the query and the use case.
