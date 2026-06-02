# User Guide

A narrative walk through Craton Bolt from `cargo add` to a working `GROUP BY`.
If you read this end-to-end you should reach a successful query in under ten
minutes; if you don't, that's a bug — please file an issue.

For the exhaustive SQL surface see
[`SQL_REFERENCE.md`](SQL_REFERENCE.md). For the architectural picture
underneath the API see [`ARCHITECTURE.md`](ARCHITECTURE.md). For the env
vars referenced from this guide see [`ENV_VARS.md`](ENV_VARS.md).

---

## Quick start

Add the crate:

```toml
[dependencies]
craton-bolt = "0.7"
arrow-array  = "53"
arrow-schema = "53"
```

Or:

```sh
cargo add craton-bolt arrow-array arrow-schema
```

Craton Bolt is GPU-only at runtime. You need an NVIDIA card with compute
capability >= 7.0 and a CUDA toolkit >= 12.0 on the linker path. Hosts
without CUDA can still type-check the crate via
`--features cuda-stub --no-default-features`, but cannot execute a query.

The smallest end-to-end program builds an Arrow `RecordBatch`, registers
it as a table, and runs a `SELECT`:

```rust
use std::sync::Arc;
use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use craton_bolt::Engine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id",    DataType::Int32, false),
        Field::new("name",  DataType::Utf8,  false),
        Field::new("score", DataType::Int32, false),
    ]));

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["alice", "bob", "carol", "dave"])),
            Arc::new(Int32Array::from(vec![95, 87, 92, 78])),
        ],
    )?;

    let mut engine = Engine::new()?;
    engine.register_table("users", batch)?;

    let result = engine.sql(
        "SELECT name, score FROM users WHERE score >= 90 ORDER BY score DESC",
    )?;
    println!("rows = {}", result.num_rows());
    println!("{:?}", result.record_batch());
    Ok(())
}
```

That's the round-trip: parse the SQL → build a logical plan → lower to a
physical plan → emit PTX → load the module on the GPU → run the kernel →
download an Arrow `RecordBatch`. `Engine::new()` picks device 0; for a
specific GPU use `Engine::new_with_device(idx)`.

Now widen the example to a `GROUP BY` — the acceptance criterion for this
guide:

```rust
let result = engine.sql(
    "SELECT id, SUM(score), COUNT(*) FROM users GROUP BY id",
)?;
```

That's enough to be productive. The rest of the guide is breadth.

---

## Registering data

Three entry points cover every shape of input data Craton Bolt accepts
today. They live on `Engine` and all return `BoltResult<()>`.

### Single batch — `register_table`

The common case. Builds the table's Utf8 dictionaries, uploads every
column to the GPU, and makes it queryable.

```rust
engine.register_table("sales", batch)?;
```

Re-registering an existing name errors. To swap contents, use
`replace_table` (atomic rebuild — a failure mid-rebuild leaves the
previous table intact):

```rust
engine.replace_table("sales", new_batch)?;
```

### Multi-batch — `register_batch`

For tables whose rows arrive incrementally. The first call creates the
table; subsequent calls append additional batches under the same name.

```rust
engine.register_batch("sales", batch_0)?;
engine.register_batch("sales", batch_1)?;
engine.register_batch("sales", batch_2)?;
```

All appended batches must share the schema of batch 0 (checked eagerly;
a mismatch returns `BoltError::Plan` at append time, not at query time).
The engine concatenates batches lazily at query time and keeps a per-
column GPU cache, so a second append only re-uploads the new tail —
the cached prefix never re-crosses PCIe.

Dictionaries are **unioned across all registered batches**: a string
literal that appears only in batch 2 still resolves correctly in
`WHERE col = 'literal'`.

### Streaming — `register_table_stream`

`Engine::register_table_stream(name, schema, iter)` accepts a
`RecordBatch` iterator (0.6). The current implementation is **eager** —
it drains the iterator into the existing in-memory table representation
— but the signature is future-compatible with a truly-lazy / out-of-core
executor, so callers won't need to rewrite when the lazy path lands (see
[`PATH_TO_1.0.md`](PATH_TO_1.0.md)). The per-column delta-upload path
behind `register_batch` still applies, so you only pay for new rows on
each subsequent query.

---

## Supported SQL

Craton Bolt's frontend is built on
[`sqlparser`](https://github.com/apache/datafusion-sqlparser-rs) and
accepts a precisely-bounded subset of SQL. The authoritative list lives
in [`SQL_REFERENCE.md`](SQL_REFERENCE.md); the short version is:

- `[WITH ...] SELECT [DISTINCT] ... FROM <table> [JOIN ...] [WHERE ...] [GROUP BY ...] [HAVING ...] [ORDER BY ...] [LIMIT ...]`.
- Aggregates: `COUNT`, `SUM`, `MIN`, `MAX`, `AVG` (GPU), plus host-side
  `STDDEV` / `VAR` (scalar and grouped) and `SUM` / `MIN` / `MAX`
  over `Decimal128`. `SUM(Int32)` widens to `Int64` to prevent silent
  wraparound. `COUNT(DISTINCT col)` is supported as the sole SELECT item —
  and, as of the 0.7 wave, also with a surrounding `SELECT DISTINCT` or a
  `HAVING` over the count (no `GROUP BY`); `COUNT(DISTINCT)` *with*
  `GROUP BY` is still rejected. `COUNT`, **`MIN`, and `MAX`** over a
  `Date32` / `Timestamp` column all work end-to-end now (GPU reduction;
  the date type / timestamp unit + timezone is preserved), in both the
  scalar and `GROUP BY` paths; `SUM` over a temporal column is rejected by
  design (see `SQL_REFERENCE.md`).
- Scalar expressions: arithmetic and comparisons (GPU), including
  `Decimal128` `+` / `-` / `*` / `/` and mixed Decimal/integer arithmetic
  (GPU; scale-aligned comparisons too), `IN` / `BETWEEN` (desugar to GPU
  comparison chains), `CASE` / `CAST` / `COALESCE` / `NULLIF` (GPU for
  numeric / `Bool` / `Date32` / `Timestamp` results; a `Utf8`-result `CASE`
  over a bare scan is **host-realized** with SQL 3VL; a `Decimal128`-result
  `CASE` is still rejected at GPU lowering), `CAST` (GPU for numeric/`Bool`
  pairs and for integer↔`Decimal128` / `Decimal128` rescale / integer→`Date32`;
  Float↔`Decimal128` and CAST to/from `Timestamp` / `String` rejected at GPU
  lowering), `LIKE` (GPU over `Utf8`), `||` (host-side), `NOT` (GPU). String
  functions: `UPPER` / `LOWER` / `LENGTH` (GPU); `SUBSTRING` (literal args)
  and single-arg `TRIM` over a bare `Utf8` scan are **host-realized** via the
  `StringProject` producer (custom-chars `TRIM` and computed `SUBSTRING` args
  fall back to the host `Project`); `CONCAT` is host-side and NULL-if-any-arg-
  NULL (GPU two-pass kernels implemented but the executor uses a byte-identical
  host mirror for now).
- Joins: one or more `INNER` / `LEFT` / `RIGHT` / `FULL OUTER` / `CROSS`
  JOINs per `SELECT`, with `ON` / `USING (...)` / `NATURAL` constraints
  (equi-keys only). Each shape has a gated GPU fast path that falls back
  to a host executor on a gate miss.
- Set ops: `UNION` (dedups), `UNION ALL` (concatenates),
  `EXCEPT [ALL]` / `INTERSECT [ALL]` (host-side).
- Query composition: non-recursive CTEs (`WITH`); uncorrelated scalar
  and `[NOT] IN` subqueries in `SELECT` / `WHERE` (and an uncorrelated
  scalar subquery in `ORDER BY`); non-lateral **derived tables** in
  `FROM` (`(SELECT ...) AS alias`, alias required). (Correlated
  subqueries, `EXISTS`, `LATERAL` derived tables, and column-list aliases
  `AS d(x, y)` are rejected.)
- Window functions: `ROW_NUMBER` / `RANK` / `DENSE_RANK` /
  `SUM` / `AVG` / `MIN` / `MAX` / `COUNT` `OVER (PARTITION BY ... ORDER
  BY ...)` (host-side, default frame only).
- `ORDER BY`: single-key `Int32`/`Int64` orderings (and multi-key /
  `DESC`) run on the GPU radix sort; other shapes sort host-side.
- Types: `Bool`, `Int32`, `Int64`, `Float32`, `Float64`, dictionary-
  encoded `Utf8`, plus `Decimal128`, `Date32`, and `Timestamp` (with the
  per-type GPU-lowering caveats in `SQL_REFERENCE.md`). `Decimal128`,
  `Date32`, and `Timestamp` columns have GPU gather (filter / compaction)
  and upload wired, so they survive a filtered query end-to-end.
  `Decimal128` arithmetic (`+` / `-` / `*` / `/`, mixed with integers),
  scale-aligned comparisons, and integer↔decimal / decimal-rescale CAST now
  run on the GPU; temporal `MIN` / `MAX` run on the GPU and preserve the date
  type / timestamp unit + timezone.
- Utf8 predicates: equality / inequality against string literals (folded
  to integer comparisons on the dictionary index at plan time, GPU),
  `LIKE` (GPU as of v0.7, with a host-side fallback), and — as of v0.7 —
  **ordering comparisons against a string literal** (`WHERE name < 'M'`,
  GPU via byte/binary collation; not locale/ICU). `IN` against Utf8 and
  ordering of *two* Utf8 columns (`a < b`) are still rejected.
- Qualified column refs (`t.col`) and case-insensitive identifiers are
  supported; `JOIN ... ON` also accepts the schema-qualified
  `schema.table.col` form (leading catalog segment dropped).

A representative query that exercises most of the working surface:

```rust
let result = engine.sql(
    "SELECT region_id, SUM(price), COUNT(*) AS n \
       FROM sales \
      WHERE region = 'US' AND active \
      GROUP BY region_id \
     HAVING COUNT(*) > 10 \
      ORDER BY region_id \
      LIMIT 100",
)?;
```

Anything outside the supported surface returns a structured
`BoltError::Sql(...)` or `BoltError::Plan(...)` with the unsupported
construct quoted in the message. Some *supported* constructs execute on
a host-side code path rather than the GPU — the `||` concat operator,
`SUBSTRING` / `TRIM` / `CONCAT`, a `Utf8`-result `CASE`, `STDDEV` / `VAR`,
`SUM` / `MIN` / `MAX` over `Decimal128`, set operations
(`EXCEPT` / `INTERSECT`), window functions, and host-side join / sort
fallbacks — and a few constructs parse and type-check but are rejected at
the GPU lowering boundary with a clear `"… not yet lowered to GPU"` message
(e.g. `CAST` between Float and Decimal or to/from Timestamp/String, and a
`Decimal128`-result `CASE`). `SQL_REFERENCE.md` tags every feature with its
execution tier (GPU / host-side / GPU lowering pending).

---

## When something fails

Every fallible entry point returns
`BoltResult<T> = Result<T, BoltError>`. `BoltError` is a `thiserror`
enum (`src/error.rs`); the variants you'll see in normal use are:

| Variant            | Phase            | Typical cause                                                |
| ------------------ | ---------------- | ------------------------------------------------------------ |
| `Sql(String)`      | Parse            | Unrecognised keyword, unsupported construct, syntax error.   |
| `Plan(String)`     | Plan / lower     | Unknown column, schema mismatch, unsupported expression.     |
| `Type(String)`     | Plan / lower     | Operands won't unify (e.g. `Utf8 + Int32`).                  |
| `Memory(String)`   | Runtime          | Host-side allocator failure, pool eviction edge case.        |
| `GpuCapacity(...)` | Runtime          | GPU hash-join overshoot — caller should retry on host.       |
| `Cuda(String)`     | Runtime          | CUDA driver error without an associated `CUresult`.          |
| `CudaWithCode { code, message }` | Runtime | Driver error with a numeric `CUresult`. Pattern-match `code` to recognise specific errors (e.g. `code == 2` is `CUDA_ERROR_OUT_OF_MEMORY`). |
| `Io(io::Error)`    | Runtime          | Wraps `std::io::Error` via `#[from]`.                        |

Use the `Display` impl to render an error, or pattern-match on the
variant to dispatch retry logic:

```rust
use craton_bolt::BoltError;

match engine.sql("SELECT FOO(x) FROM t") {
    Ok(handle) => process(handle),
    Err(BoltError::Sql(msg))    => eprintln!("query rejected: {msg}"),
    Err(BoltError::Plan(msg))   => eprintln!("plan error: {msg}"),
    Err(BoltError::CudaWithCode { code: 2, .. }) => {
        // CUDA_ERROR_OUT_OF_MEMORY — drop caches and retry once.
    }
    Err(other) => eprintln!("engine error: {other}"),
}
```

A few common error shapes and what they actually mean:

- **`SQL parse error: ...`** — the query didn't make it past
  `sqlparser`. Re-read the keyword the parser flagged; the error
  message includes a column number.
- **`plan error: unknown column 'foo' on table 'sales'`** — typo or
  case mismatch. Identifiers are matched case-insensitively
  (`SELECT Score` resolves to a column named `score`), but they have
  to exist.
- **`plan error: table 'sales' is already registered`** — you called
  `register_table` twice. Use `register_batch` to append, or
  `replace_table` to swap.
- **`CUDA driver error 2: out of memory`** — the device-memory pool
  couldn't satisfy an allocation. See *Performance tuning* below for
  the pool knobs; the pool watcher (under the `pool-watcher` feature)
  proactively evicts on low-water hits.

For observability, the engine emits periodic pool-stats lines via
the [`log`](https://crates.io/crates/log) crate's `info!` macro. Wire
up any `log` backend (`env_logger`, `tracing-log`, `fern`, …) and the
lines appear automatically:

```rust
fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    // ... your engine code ...
    Ok(())
}
```

Run with `RUST_LOG=info` to see them. For structured access (Prometheus,
OpenTelemetry, a custom dashboard) install an observer:

```rust
use craton_bolt::{install_pool_stats_observer, PoolStats};

install_pool_stats_observer(Box::new(|s: PoolStats| {
    println!(
        "pool: {} bytes / {} buckets ({} OOM rescues)",
        s.total_pooled_bytes, s.bucket_count, s.oom_recovery_count,
    );
}));
```

Structured per-phase tracing spans (parse / plan / lower / codegen /
ptx_load / launch / transfer / materialize) shipped in 0.6 via the
[`tracing`](https://crates.io/crates/tracing) crate. They are off by
default and opt-in: install a `tracing_subscriber` in your binary and
the spans appear automatically (span names are catalogued in
`src/observability.rs`). The `log`-level diagnostics and the pool-stats
observer remain available alongside the spans.

---

## Performance tuning

Most workloads need no tuning. The defaults are conservative
production-safe choices. When you do need to dial something, here are
the knobs that actually exist today.

### Env vars

Full reference: [`ENV_VARS.md`](ENV_VARS.md). The ones you're most
likely to touch:

| Var                              | Default     | What                                                  |
| -------------------------------- | ----------- | ----------------------------------------------------- |
| `CRATON_BOLT_POOL_MAX_BYTES`     | 512 MiB     | Soft cap on total pooled GPU memory.                  |
| `CRATON_BOLT_PTX_CACHE_CAP`      | 256         | Compiled-PTX module cache size (FIFO eviction).       |
| `BOLT_POOL_STATS_INTERVAL_SECS`  | 60          | Pool-stats log cadence. Set to `0` to silence.        |
| `BOLT_POOL_WATCH_INTERVAL_SECS`  | 5           | `pool-watcher` poll cadence (only with that feature). |
| `BOLT_GPU_JOIN_TABLE_CAP_MB`     | driver-detected | Override the GPU hash-join byte cap, 64..=4096 MiB. |

Example — bound the engine's resident footprint on a shared GPU:

```sh
CRATON_BOLT_POOL_MAX_BYTES=$((256 * 1024 * 1024)) \
    cargo run --release
```

Most env vars are read once on first use and frozen for the process
lifetime; setting them mid-run has no effect.

### Cargo features

```toml
[dependencies]
craton-bolt = { version = "0.7", features = ["pool-sharded"] }
```

| Feature        | Default | What it does                                                            |
| -------------- | ------- | ----------------------------------------------------------------------- |
| `cuda-stub`    | off     | Build without linking CUDA. Every FFI entry returns `CUDA_ERROR_STUB`. Useful for `cargo check` on a CUDA-less host or docs.rs. Cannot execute queries. |
| `cudarc`       | off     | Route low-level driver calls through the `cudarc` crate instead of the hand-rolled FFI. Stage-1 spike; see `docs/CUDARC_ADOPTION.md`. |
| `pool-sharded` | off     | Swap the memory-pool bucket map from a `DashMap` to a fixed 32-way mutex-array shard. Turn on only if profiling shows DashMap contention. |
| `pool-watcher` | off     | Spawn a background thread that polls device memory and proactively evicts when free / total drops below `BOLT_POOL_WATCH_LOW_WATER_FRAC` (default 10%). Off by default — adds a permanently-resident thread. |

### The PTX module cache

Every distinct `KernelSpec` Craton Bolt emits is JIT-compiled exactly
once per process: the emitted PTX text is hashed and the loaded
`CudaModule` is reused on subsequent queries with the same kernel
shape. The cache is in-process, FIFO, capped at
`CRATON_BOLT_PTX_CACHE_CAP` entries (default 256). On a long-running
workload that cycles through a bounded set of query shapes, every
query after the first warm-up issues zero PTX compiles.

A **disk-backed** persistent PTX cache (so cold-start is fast after the
first run of a process) shipped in 0.6. It is opt-in via the
`BOLT_PTX_CACHE_DIR=/path` env var (or the
`Engine::Builder::persistent_cache(path)` hook); on a miss in the
in-process cache the engine reads a `.ptx` entry from disk before
re-running codegen, and writes freshly-generated PTX back atomically
(tempfile + rename) for the next process. Set it on benchmark harnesses,
CLI tools, and per-request workers that never benefit from the
in-process cache alone. See [`ENV_VARS.md`](ENV_VARS.md) for the path
conventions. The in-process cache still covers the steady-state case.

### Multi-GPU

One CUDA context per `Engine`, one `Engine` per GPU. To use multiple
cards, construct one engine per device and dispatch queries to the
appropriate one:

```rust
let gpu_0 = Engine::new_with_device(0)?;
let gpu_1 = Engine::new_with_device(1)?;
```

There is no automatic cross-GPU query planning. That's a 2.0 concern
(see [`PATH_TO_1.0.md`](PATH_TO_1.0.md) §7).

---

## See also

- [`SQL_REFERENCE.md`](SQL_REFERENCE.md) — every accepted query shape,
  every rejection reason.
- [`ENV_VARS.md`](ENV_VARS.md) — exhaustive env-var reference.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — what the engine actually does
  with your query under the hood.
- [`FAQ.md`](FAQ.md) — common questions about the design choices.
- [`PATH_TO_1.0.md`](PATH_TO_1.0.md) — what's coming, in what order,
  with what acceptance criteria.
- [`CHANGELOG.md`](../CHANGELOG.md) — what landed when.
