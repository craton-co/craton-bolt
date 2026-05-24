# SQL Reference

The exact subset of SQL Javelin's frontend accepts. The grammar is built on top of [`sqlparser`](https://github.com/apache/datafusion-sqlparser-rs); anything outside this surface produces a clear `JavelinError::Sql(...)` or `JavelinError::Plan(...)` with the unsupported construct.

For the JIT pipeline that lowers and executes these queries, see [`JIT_PIPELINE.md`](JIT_PIPELINE.md).

## Supported query shape

```
SELECT <select_list>
  FROM <single_table>
 [WHERE <bool_expr>]
 [GROUP BY <expr_list>]
```

**Hard restrictions** (everything else returns a `JavelinError`):

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
- Post-aggregate expressions (`SUM(price) + 1`) are not yet supported — emits a `JavelinError::Sql("post-aggregate expressions not yet supported")`.

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
| `MAX(bool)`    | `Bool`                        |                                                            |
| `MAX(utf8)`    | `Utf8`                        |                                                            |
| `AVG(numeric)` | `Float64`                     | Split into `SUM + COUNT` on host.                          |
| `AVG(bool)`    | `Float64`                     | Fraction of `TRUE` rows. NULL if all-null group.           |

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
SELECT UPPER(region) FROM sales;             -- via Engine API, not yet wired into SQL frontend
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
- String concatenation operator `||`. (The `CONCAT(...)` host function exists in `src/exec/string_ops_extended.rs` but isn't yet wired into the SQL frontend.)
- DDL of any kind. There's no `CREATE TABLE`. Tables are registered via the Rust API (`Engine::register_table`).

If you need any of the above for your use case, please open an issue describing the query and the use case.
