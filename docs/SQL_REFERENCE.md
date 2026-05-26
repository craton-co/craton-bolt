# SQL Reference

The exact subset of SQL Craton Bolt's frontend accepts. The grammar is built on top of [`sqlparser`](https://github.com/apache/datafusion-sqlparser-rs); anything outside this surface produces a clear `BoltError::Sql(...)` or `BoltError::Plan(...)` with the unsupported construct.

For the JIT pipeline that lowers and executes these queries, see [`JIT_PIPELINE.md`](JIT_PIPELINE.md).

## Supported query shape

```
SELECT <select_list>
  FROM <single_table>
 [WHERE <bool_expr>]
 [GROUP BY <expr_list>]
```

**Hard restrictions** (everything else returns a `BoltError`):

- One `SELECT` per query. No UNION, INTERSECT, EXCEPT, CTE, subquery in FROM, subquery in WHERE.
- Exactly one table in `FROM`. No JOIN. No schema-qualified names.
- No `DISTINCT`, `HAVING`, `ORDER BY`, `LIMIT`, `OFFSET`, `FETCH`, `LOCK`, `INTO`, `WINDOW`, `QUALIFY`, lateral, table-valued functions, `PREWHERE` (clickhouse-ism), `CONNECT BY`, `CLUSTER / DISTRIBUTE / SORT BY`.
- No `GROUP BY ALL`, `ROLLUP`, `CUBE`, `TOTALS`.

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

- Integer: `42`. Parsed as `Int64` (negative-literal supported via unary minus folding).
- Float: `3.14`. Parsed as `Float64`.
- String: `'US'`. Parsed as `Utf8`.
- Boolean: `TRUE` / `FALSE`. Case-insensitive per `sqlparser`.
- Null: `NULL`. Parsed as `Literal::Null`; semantics in expressions are limited (see Type checking).

Unary minus on a numeric literal is folded into a signed literal (`-5` becomes `Literal::Int64(-5)`). On any other expression it's rewritten as `0 - expr`. Unary plus is a no-op. Any other unary op is an error.

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

String comparison (`<`, `>`, etc.) on Utf8 columns is rejected — dictionary indices reflect insertion order, not lex order, so the rewriter can't safely fold them. Use `=` and `<>` for string predicates. `MIN(utf8_col)` and `MAX(utf8_col)` use lex order on the host.

### Logical

`AND`, `OR`. Both operands must be `Bool`. Result is `Bool`.

No `NOT` yet — would need a unary op in the AST.

## SELECT list

Each item is one of:

- `column`. Bare column reference. Carries the source column's name into the output.
- `*`. Expanded to one `Expr::Column(name)` per field of the FROM table.
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
| `SUM(int|float)` | Same dtype as input         | Synthesises an all-ones column for COUNT internally.       |
| `SUM(bool)`    | `Int64`                       | Count of `TRUE` rows.                                      |
| `MIN(int|float)` | Same dtype as input         | Float MIN via `atom.cas` loop on bit pattern (sm_70).      |
| `MIN(bool)`    | `Bool`                        | `FALSE < TRUE`. NULL if all-null group.                    |
| `MIN(utf8)`    | `Utf8`                        | Lexicographic; NULL if all-null group.                     |
| `MAX(int|float)` | Same dtype as input         | Same caveats as MIN.                                       |
| `MAX(bool)`    | `Bool`                        | NULL if all-null group.                                    |
| `MAX(utf8)`    | `Utf8`                        | NULL if all-null group.                                    |
| `AVG(numeric)` | `Float64`                     | Split into `SUM + COUNT` on host.                          |
| `AVG(bool)`    | `Float64`                     | Fraction of `TRUE` rows. NULL if all-null group.           |

Aggregates over an all-NULL group (Bool/Utf8 inputs, which thread validity through `extended_agg`) return SQL `NULL` in both the scalar and GROUP BY paths. Earlier 0.1.x snapshots could return `0.0` / `false` for those empty groups in the GROUP BY path — fixed in the same wave that wired `extended_agg` through `groupby_with_pre`. Primitive aggregates do not yet read a validity bitmap, so `SUM`/`MIN`/`MAX`/`AVG` over `Int*`/`Float*` treat every row as non-null (see "What's NOT supported").

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

Output ordering: groups are sorted by encoded key for determinism. (Float bit-pattern order, which is NOT the same as numeric order — `-1.0` sorts after `+0.0` in output. Acceptable for a deterministic-but-not-ANSI v1.)

## Expression examples

These all work today against a table with `region_id: Int32`, `price: Float64`, `tax: Float64`, `region: Utf8`:

```sql
SELECT price FROM sales;
SELECT price, tax FROM sales;
SELECT price * tax FROM sales;
SELECT price * tax + price FROM sales;
SELECT price FROM sales WHERE region_id = 1;
SELECT price * tax FROM sales WHERE region_id = 1 AND price > 100.0;
SELECT * FROM sales;
SELECT region, price FROM sales WHERE region = 'US';
SELECT SUM(price) FROM sales;
SELECT SUM(price * tax) FROM sales;
SELECT SUM(price * tax) FROM sales WHERE region_id = 1;
SELECT region_id, SUM(price), AVG(tax) FROM sales GROUP BY region_id;
SELECT region_id, SUM(price * tax) FROM sales WHERE active GROUP BY region_id;
SELECT a, b, SUM(v) FROM events GROUP BY a, b;             -- 2-col Int32 packed key
SELECT a, b, c, SUM(v) FROM events GROUP BY a, b, c;       -- 3-col wide-key fallback (host)
SELECT COUNT(*) FROM sales;
SELECT COUNT(*) FROM sales WHERE region = 'US';
SELECT MIN(temp), MAX(temp) FROM weather GROUP BY station; -- Float MIN/MAX via CAS
```

## What's NOT supported

These produce explicit errors:

- JOIN of any kind.
- Subqueries.
- Window functions.
- `CASE ... WHEN`, `NULLIF`, `COALESCE`, `IFNULL`, `IIF`.
- `CAST(... AS ...)` — the planner does implicit numeric promotion only.
- `LIKE`, `IN`, `BETWEEN`, `IS NULL`, `IS NOT NULL`.
- `EXISTS`.
- Date / time / timestamp literals and arithmetic.
- Decimal / fixed-point arithmetic.
- Array / list / struct / map types.
- `COUNT(DISTINCT col)`.
- Post-aggregate expressions (`SUM(price) + 1`).
- Ordering on Utf8 columns (`ORDER BY name` or `WHERE name < 'M'`).
- String concatenation operator `||`.
- DDL of any kind. There's no `CREATE TABLE`. Tables are registered via the Rust API (`Engine::register_table`).

## Not yet supported (planned)

### String functions

`UPPER`, `LOWER`, `LENGTH`, `CONCAT`, `SUBSTRING` are reachable only from the executor-level `src/exec/string_ops` / `src/exec/string_ops_extended` API; no SQL or DataFrame surface exposes them yet. They run as pure-host dictionary transformations because variable-width device writes remain unsupported by the codegen path. Wiring them through the SQL frontend would mean teaching `sql_frontend::lower` to recognise `Expr::FunctionCall` and routing it to a per-function host-side projection executor.

If you need any of the above for your use case, please open an issue describing the query and the use case.
