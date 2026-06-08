# craton-bolt — Public API Surface

This document enumerates every symbol that crosses the `craton-bolt` crate
boundary via `pub use` from `src/lib.rs` (and the modules it re-exports). It
is the v0.6 / M7 staging ground for the public-API freeze that ships with
1.0: until 1.0, the tier assignments below describe **intent**, not yet a
contract. At 1.0 each entry in the **stable** tier becomes binding under
the semver rules listed below.

> Crate version this document was enumerated against: `0.7.0` (from
> `Cargo.toml`). Reconciled against `src/lib.rs` and the modules it
> re-exports; re-run the enumeration whenever the public surface changes (and
> before the 1.0 freeze).

## Stability tiers and semver contract

The `Notes` column for each symbol records both the kind of change that
would break it and, where useful, why it lives in the tier it does. The
tiers themselves carry these contracts:

### stable

Symbols intended to be locked at 1.0. Once the freeze ships, the following
count as breaking changes (major-version bump):

- Renaming, removing, or relocating the symbol.
- For **structs / enums**: adding a new public field or variant *without*
  `#[non_exhaustive]` (none of the current items carry that attribute — see
  per-row notes). Changing an existing field's type or visibility is
  breaking.
- For **functions / methods**: changing the signature (parameters, return
  type, generic bounds, `&self` vs `&mut self`).
- For **traits**: adding a required method (without a default), changing
  any existing method signature, or tightening a supertrait bound.
- For **type aliases / `const` items**: changing the aliased type or the
  constant's type. The numeric value of a `const` may change in a minor
  release IFF the documented meaning is preserved (e.g. tuning a default
  cap); this is called out per-row where it applies.
- Tightening trait bounds on a public item is breaking; relaxing them is
  not.

Adding new fields with `#[non_exhaustive]`, adding new variants to a
`#[non_exhaustive]` enum, or adding new methods with defaults to a trait
that has no public implementors outside the crate is **non-breaking**.

### experimental

Public, but not yet covered by the 1.0 freeze. Anything in this tier MAY
change shape between minor releases. The crate will document the change in
`CHANGELOG.md` and (where reasonable) provide a deprecation window, but
downstream callers should expect to track these. Items here are
publicly-reachable today either because they are the only handle on a
real workload (e.g. observability hooks) or because the surface needs more
soak time before promotion.

### hidden

Items marked `#[doc(hidden)]` (or living behind a `#[doc(hidden)]` module
path such as `__test_only_*`). They are *not* part of the public API
surface and are not subject to semver — a patch release may rename,
relocate, or delete any of them without notice. Downstream code that
imports a hidden item is on its own. The items appear in the crate's `pub`
graph so integration tests under `tests/` and benches under `benches/`
(which compile as separate crates) can reach them; for that reason
removing them outright is also non-breaking by definition.

The public IR types (`KernelSpec`, `PhysicalPlan`, `Op`, `Reg`, `Value`,
`ColumnIO`) are publicly re-exported from `crate::plan` but each individual
type carries `#[doc(hidden)]` at its definition site. They are listed in
**hidden** below with a per-row M7 decision note: for v0.6 the default is
to leave them hidden, with the explicit option to either **promote** to
stable IR (committing to backwards-compatible evolution) or to
**encapsulate** them behind opaque builders before 1.0. No decision is
forced by this document.

---

## stable

| Symbol | Kind | Notes |
|---|---|---|
| `BoltError` | enum (re-exported from `error`) | Public error type. Carries `#[non_exhaustive]` (added v0.6 / M5), so adding a new variant is non-breaking and downstream `match`es must use a wildcard arm. Variants `Cuda(String)`, `CudaWithCode { code, message }`, `Sql(String)`, `SqlWithSpan { msg, span }`, `Plan(String)`, `Type(String)`, `Memory(String)`, `Io(std::io::Error)`, `GpuCapacity(String)`, `Other(String)`. Removing or renaming any variant, or changing the field shape of `CudaWithCode` / `SqlWithSpan`, is breaking. Inherent method `span(&self) -> Option<Range<usize>>` is part of the surface (returns the byte-range for `SqlWithSpan`, `None` otherwise). `impl From<std::io::Error>` and `impl From<sqlparser::parser::ParserError>` are also public. |
| `BoltResult<T>` | type alias | `= Result<T, BoltError>`. Changing the alias target is breaking. |
| `tracing` | crate re-export (`pub use ::tracing`) | The `tracing` crate is re-exported at the crate root so downstream users can install a subscriber and reference span/event APIs without depending on `tracing` directly. Per the lib.rs docs the *major version* of `tracing` is part of the SemVer contract; bumping it across a major is breaking. |
| `GpuBuffer<T: Pod>` | struct (`cuda::buffer`) | RAII device buffer. Fields are private; adding a new field is non-breaking. Changing the `T: Pod` bound is breaking. |
| `GpuVec<T: Pod>` | struct (`cuda::smart_ptrs`) | Owned, growable device vector. Fields are private. The `T: Pod` bound is part of the public contract. |
| `GpuView<'a, T: Pod>` | struct (`cuda::smart_ptrs`) | Shared device-side borrow handle. Lifetime parameter is part of the signature. |
| `GpuViewMut<'a, T: Pod>` | struct (`cuda::smart_ptrs`) | Exclusive `!Sync`, `!Copy` device-side borrow. Removing the auto-trait negative impls is breaking (callers rely on the borrow-checker exclusivity). |
| `DataFrame` | struct (`plan::dataframe`) | Builder-style DataFrame API. Fields are private. |
| `LogicalPlan` | enum (`plan::logical_plan`) | Logical-plan IR root. Each variant carries planner-internal payloads. **Not** `#[non_exhaustive]` — adding variants is breaking until the attribute is added. |
| `Expr` | enum (`plan::logical_plan`) | Logical scalar expression. Same caveat as `LogicalPlan` re: `#[non_exhaustive]`. |
| `Engine` | struct (`exec::engine`) | Top-level query engine. Re-exported at the crate root via `pub use exec::{Engine, EngineBuilder}`. Fields are private; new fields are non-breaking. Stable inherent methods: `new() -> BoltResult<Self>`, `new_with_device(i32) -> BoltResult<Self>`, `builder() -> EngineBuilder`, `device(&self) -> i32`, `memory_budget_bytes(&self) -> Option<usize>`, `persistent_cache_path(&self) -> Option<&Path>`, `tracing_enabled(&self) -> bool`, `with_rewrite(self, Box<dyn PlanRewrite>) -> Self`, `rewrite_count(&self) -> usize`, `register_table(...)`, `register_table_stream(...)`, `register_table_stream_lazy(&mut self, name: impl Into<String>, schema: plan::Schema, producer: exec::streaming::BatchProducer) -> BoltResult<()>` (registers a replayable, lazily-materialised streaming source; errors if the table name is already registered), `replace_table(...)`, `register_batch(&mut self, name, RecordBatch) -> BoltResult<()>`, `sql(&str) -> BoltResult<QueryHandle>`, `run_logical_plan(&mut self, &LogicalPlan) -> BoltResult<QueryHandle>`, `execute(&PhysicalPlan) -> BoltResult<QueryHandle>`. Changing any of those signatures is breaking. |
| `EngineBuilder` | struct (`exec::engine`) | Builder for `Engine`, re-exported at the crate root alongside `Engine`. Stable inherent methods: `new()`, `device(self, i32)`, `memory_budget(self, usize)`, `persistent_cache(self, PathBuf)`, `enable_tracing(self)`, `build(self) -> BoltResult<Engine>`. Fields are private; new fields are non-breaking. |
| `QueryHandle` | struct (`exec::engine`) | Query-result handle wrapping a `RecordBatch`. Reachable as `craton_bolt::exec::QueryHandle` (re-exported from `exec`, but not at the crate root). Stable inherent methods: `record_batch(&self)`, `into_record_batch(self)`, `num_rows(&self)`. |
| `PlanRewrite` | trait (`plan::rewrite`) | Re-exported at `plan::PlanRewrite`. Supertrait-bounded `Send + Sync`; consumed by `Engine::with_rewrite`. Required method(s) define the logical-plan rewrite hook; changing any existing signature or tightening the supertrait bound is breaking. |
| `plan::TableProvider` | trait (`plan::sql_frontend`) | Frontend trait for resolving table schemas and per-column null-bearing. Required method: `schema(&self, name: &str) -> BoltResult<Schema>`. Default-impl methods: `has_nulls`, `null_count`, `schema_version`. Adding a new method WITH a default is non-breaking; adding a required method is breaking. Changing any existing signature is breaking. |
| `plan::MemTableProvider` | struct (`plan::sql_frontend`) | Default in-memory `TableProvider` impl. Inherent methods `new`, `with_table`, `register`, `unregister_table`, `set_column_nullability`, `has_nulls` are part of the surface. |
| `plan::parse_sql` | fn (`plan::sql_frontend::parse`) | `fn parse(sql: &str, provider: &dyn TableProvider) -> BoltResult<LogicalPlan>`. Re-exported at `plan::parse_sql`. Changing the signature or accepted SQL dialect compatibility envelope is breaking. |
| `plan::Schema` | struct (`plan::logical_plan`) | Logical schema. Used in `TableProvider::schema`; changing field shape is breaking. |
| `plan::Field` | struct (`plan::logical_plan`) | Named column with `DataType` + nullability. |
| `plan::DataType` | enum (`plan::logical_plan`) | Engine-internal type enum. **Not** `#[non_exhaustive]` — see general caveat. (Also re-exported under `__test_only_logical_plan` for test-only consumers; the canonical path is `plan::DataType`.) |
| `plan::Literal` | enum (`plan::logical_plan`) | Scalar literal in logical expressions. |
| `plan::BinaryOp` | enum (`plan::logical_plan`) | Binary operator enum. |
| `plan::UnaryOp` | enum (`plan::logical_plan`) | Unary operator enum. |
| `plan::AggregateExpr` | enum (`plan::logical_plan`) | Aggregate expression carried in `LogicalPlan::Aggregate`. |
| `plan::ScalarFnKind` | enum (`plan::logical_plan`) | Scalar-function kind carried in `Expr`. Re-exported at `plan::ScalarFnKind`. **Not** `#[non_exhaustive]` — adding variants is breaking until the attribute is added. |
| `plan::TimeUnit` | enum (`plan::logical_plan`) | Time-unit qualifier for temporal `DataType`s. Re-exported at `plan::TimeUnit`. Same `#[non_exhaustive]` caveat. |
| `plan::col` | fn (`plan::logical_plan`) | `fn col(name: impl Into<String>) -> Expr`. |
| `plan::lit` | fn (`plan::logical_plan`) | `fn lit<T: Into<Literal>>(v: T) -> Expr`. Changing the generic bound is breaking. |
| `plan::dataframe::GroupedDataFrame` | struct | Intermediate DataFrame builder for grouped operations. |
| `plan::dataframe::count` | fn | `fn count(e: Expr) -> AggregateExpr`. |
| `plan::dataframe::sum` | fn | `fn sum(e: Expr) -> AggregateExpr`. |
| `plan::dataframe::min` | fn | `fn min(e: Expr) -> AggregateExpr`. |
| `plan::dataframe::max` | fn | `fn max(e: Expr) -> AggregateExpr`. |
| `plan::dataframe::avg` | fn | `fn avg(e: Expr) -> AggregateExpr`. |
| `plan::dataframe::var_pop` | fn | `fn var_pop(e: Expr) -> AggregateExpr`. Also re-exported at `plan::var_pop`. |
| `plan::dataframe::var_samp` | fn | `fn var_samp(e: Expr) -> AggregateExpr`. Also re-exported at `plan::var_samp`. |
| `plan::dataframe::stddev_pop` | fn | `fn stddev_pop(e: Expr) -> AggregateExpr`. Also re-exported at `plan::stddev_pop`. |
| `plan::dataframe::stddev_samp` | fn | `fn stddev_samp(e: Expr) -> AggregateExpr`. Also re-exported at `plan::stddev_samp`. |

## experimental

These symbols are public today but the team retains the right to evolve
their shape before 1.0. Each row lists the specific axis along which the
team expects to iterate.

| Symbol | Kind | Notes |
|---|---|---|
| `pool_stats` | fn (`cuda::mem_pool`) | `fn pool_stats() -> PoolStats`. Observability hook for the process-wide device-memory pool. The function signature is stable, but `PoolStats` is itself experimental (new fields expected). |
| `PoolStats` | struct (`cuda::mem_pool`) | Snapshot of pool telemetry: `total_pooled_bytes: usize`, `bucket_count: usize`, `oom_recovery_count: u64`, `proactive_eviction_count: u64`. Lib-rs docs declare new fields non-breaking ("**new fields may be added (non-breaking) but existing ones keep their semantics**"). Reaching 1.0 SHOULD either mark this `#[non_exhaustive]` or freeze the field set; until then callers must use `..` in struct patterns. |
| `install_pool_stats_observer` | fn (`observability`) | `fn install_pool_stats_observer(f: Box<dyn Fn(PoolStats) + Send + Sync + 'static>)`. Re-exported at the crate root. Single-slot process-wide observer. The single-slot semantics (second install overwrites first) are intentional but considered experimental until used by at least one downstream exporter. |
| `observability::PoolStatsObserver` | type alias (`observability`) | `= Box<dyn Fn(PoolStats) + Send + Sync + 'static>`. The boxed-callback type for the pool-stats observer. Reachable as `craton_bolt::observability::PoolStatsObserver` (not re-exported at the crate root; `install_pool_stats_observer` spells the type out inline rather than using the alias). Experimental — tied to the observer surface above. |
| `jit::ptx_cache_stats` | fn (`jit::jit_compiler`) | `fn ptx_cache_stats() -> (usize, usize, usize)`. Returns `(hits, misses, evictions)`. Tuple-return is experimental — at 1.0 this should likely be a named struct (`PtxCacheStats`) for forward-compatibility, since today adding a fourth counter is breaking. |
| `plan::sql_frontend::plan_cache_stats` | fn | `fn plan_cache_stats() -> (usize, usize, usize)`. Same shape-evolution caveat as `ptx_cache_stats` — tuple return is breaking-to-extend. Not re-exported at the crate root, only reachable as `craton_bolt::plan::sql_frontend::plan_cache_stats`. |
| `cuda::CudaContext` | struct (`cuda::cuda_sys`) | Re-exported at `craton_bolt::cuda::CudaContext`. Wraps a CUDA driver context. Public today because `Engine::new_with_device` returns one transitively, but the inherent API is in flux. |
| `cuda::CUdevice`, `cuda::CUdeviceptr`, `cuda::CUfunction`, `cuda::CUmodule`, `cuda::CUresult`, `cuda::CUstream` | type aliases (`cuda::cuda_sys`) | Raw driver-binding type aliases (`i32`, `u64`, `*mut c_void`). Re-exported at `craton_bolt::cuda::*` for callers that need to bridge to other CUDA crates. Considered experimental because their stability depends on the CUDA driver ABI we choose to track. |
| `cuda::buffer::primitive_to_gpu` | fn | `fn primitive_to_gpu<P>(...)`. Re-exported at `craton_bolt::cuda::primitive_to_gpu`. Helper for uploading primitive Arrow arrays. Experimental because the input shape will likely generalise. |
| `cuda::PinnedHostBuffer<T: Pod>` | struct (`cuda::buffer`) | Page-locked host buffer for async H2D / D2H transfers. |
| `jit::set_disk_ptx_cache_dir` | fn (`jit::disk_cache::set_override_dir`) | `fn set_override_dir(dir: Option<PathBuf>)`, re-exported at `jit::set_disk_ptx_cache_dir`. Points the process-wide disk-PTX cache at a directory, overriding `BOLT_PTX_CACHE_DIR`; `None` clears the override. Experimental builder hook (v0.6 / M6 disk-cache opt-in). |
| `jit::current_disk_ptx_cache_dir` | fn (`jit::disk_cache::current_override_dir`) | `fn current_override_dir() -> Option<PathBuf>`, re-exported at `jit::current_disk_ptx_cache_dir`. Read-back accessor mirroring `set_disk_ptx_cache_dir`. Experimental — exposed primarily for `EngineBuilder` integration tests. |
| `plan::sort_by` / sort-related `plan` items | (none stable today) | — |
| `exec::streaming::BatchProducer` | type alias (`exec::streaming`) | `= Box<dyn Fn() -> Box<dyn Iterator<Item = BoltResult<RecordBatch>>> + Send + Sync>`. Re-exported at the crate root via `pub use exec::streaming::{...}`. The replayable batch-producer factory consumed by `Engine::register_table_stream_lazy`. Experimental — the streaming registration surface is still evolving. |
| `exec::streaming::BatchStream<'a>` | struct (`exec::streaming`) | Re-exported at the crate root. Borrowed streaming-batch iterator handle. Experimental. |
| `exec::streaming::MorselPlan` | enum (`exec::streaming`) | Re-exported at the crate root. Describes how a streaming source is sliced into morsels. Experimental. |
| `exec::streaming::PinnedBudget` | struct (`exec::streaming`) | Re-exported at the crate root. Pinned-host transfer budget for the streaming path. Experimental. |
| `exec::streaming::TableSource` | enum (`exec::streaming`) | Re-exported at the crate root. How a registered table's host data is stored (eager `Vec<RecordBatch>` vs lazy `Streaming(BatchProducer)`). Experimental. |
| `metrics` | module (`pub mod metrics`) | The M5 process-wide metrics registry: atomic counters + per-phase power-of-two latency histograms, dependency-free. The re-exports below are surfaced at the crate root; the full module is reachable as `craton_bolt::metrics`. Experimental until the counter/phase enums and snapshot shape soak one release. |
| `metrics::metrics` (re-exported at crate root as `metrics`) | fn (`metrics`) | `fn metrics() -> &'static Metrics`. Hands back the process-wide registry for inline `inc(Counter)` / `observe(Phase, micros)` bumps. Experimental. |
| `metrics_snapshot` | fn (crate root; `pub use metrics::snapshot as metrics_snapshot`) | `fn snapshot() -> MetricsSnapshot`, re-exported under the disambiguated name `metrics_snapshot` (the pool surface also has a "snapshot" concept). Returns an owned, plain-data snapshot for a scraper. Experimental. |
| `render_prometheus` | fn (crate root; `pub use metrics::{..., render_prometheus, ...}`) | `fn render_prometheus(snapshot: &MetricsSnapshot) -> String`. Formats a `MetricsSnapshot` in Prometheus text-exposition format (newline-terminated `craton_`-prefixed gauge + histogram lines). Experimental — the output schema (metric names, label names, histogram bucket boundaries) may change between minor releases as the metrics surface matures. |
| `metrics::Counter` (re-exported at crate root as `Counter`) | enum (`metrics`) | Counter identifiers — the six surviving variants are `QueriesTotal`, `QueriesFailed`, `PtxCacheHits`, `PtxCacheMisses`, `GpuLaunchesTotal`, `HostFallbacksTotal` (the `BytesUploaded` / `BytesDownloaded` PCIe-byte-total counters were removed). **Not** `#[non_exhaustive]` — adding variants is breaking until the attribute is added; enumeration order is the order `MetricsSnapshot::counters` yields. Experimental. |
| `metrics::Phase` (re-exported at crate root as `Phase`) | enum (`metrics`) | Per-query-phase identifier for the latency histograms — the four surviving variants are `Parse`, `Plan`, `Lower`, `Materialize` (the earlier `Codegen` / `PtxLoad` / `Launch` / `Transfer` phases were removed). Same `#[non_exhaustive]` caveat as `Counter`. Experimental. |
| `metrics::MetricsSnapshot` (re-exported at crate root as `MetricsSnapshot`) | struct (`metrics`) | Owned, plain-data snapshot of the whole registry. Exposes `counters()` (name/value pairs) and per-phase histogram accessors. Field set is expected to grow; callers should treat it as snapshot-only. Experimental. |
| (1.0 decision note) | — | `jit::ptx_cache_stats` and `plan::sql_frontend::plan_cache_stats` return tuples; promote them to named structs (`PtxCacheStats`, `PlanCacheStats`) before the 1.0 freeze so adding a counter is non-breaking. `PoolStats` should gain `#[non_exhaustive]` or freeze its field set. `metrics::Counter` / `metrics::Phase` are not `#[non_exhaustive]` — add the attribute or freeze the variant list. The `exec::streaming::*` and `metrics::*` re-exports have now soaked through v0.7; evaluate for promotion to stable at the start of the 1.0 freeze window. |

## hidden

Items in this tier are publicly reachable but carry `#[doc(hidden)]` (or
live under a `__test_only_*` module that does). They are **NOT** subject
to semver — any patch release may rename, move, or delete them. Listed
here for completeness so the parent agent / reviewer can see what is
*currently* reachable from outside the crate without being part of the
contract.

### Public IR (`#[doc(hidden)]` today)

These types are re-exported at `craton_bolt::plan::*` but each definition
carries `#[doc(hidden)]`. The M7 decision for each is whether to
**promote** (commit to the type as a stable IR for downstream tooling) or
**encapsulate** (wrap behind opaque builders, keep the concrete shape
internal). For v0.6 the default is **hidden / experimental**.

| Symbol | Kind | Notes |
|---|---|---|
| `plan::PhysicalPlan` | enum (`plan::physical_plan`) | Internal pipeline IR. **M7 decision: default hidden** for v0.6. Promotion candidate if downstream wants to build executors; otherwise encapsulate via a `Engine::execute(handle: OpaquePlanHandle)` style API before 1.0. |
| `plan::KernelSpec` | struct (`plan::physical_plan`) | Per-kernel IR. **M7: hidden / experimental.** Same promote-or-encapsulate decision as `PhysicalPlan`. Several fields (`input_has_validity`, `output_has_validity`) themselves carry `#[doc(hidden)]` and are de-facto internal. |
| `plan::Op` | enum (`plan::physical_plan`) | Single IR instruction. Hidden — see `KernelSpec`. |
| `plan::Reg` | struct (`plan::physical_plan`) | SSA register handle. Field is `pub(crate)`; only the inherent `id() -> u32` accessor is public. Hidden until IR-promotion decision lands. |
| `plan::Value` | struct (`plan::physical_plan`) | Typed value (Reg + DataType). Hidden — see `KernelSpec`. |
| `plan::ColumnIO` | struct (`plan::physical_plan`) | Column metadata for `KernelSpec`. Hidden — see `KernelSpec`. |
| `plan::lower_physical` | fn (`plan::physical_plan::lower`) | `fn lower(plan: &LogicalPlan) -> BoltResult<PhysicalPlan>`. Lowers Logical → Physical. Hidden until the IR types are themselves promoted; otherwise this returns an opaque-but-`pub` type, which is non-useful. |

### Test/bench-only re-exports (`#[doc(hidden)]`)

Reachable by integration tests under `tests/` and benches under `benches/`
(which compile as separate crates and cannot reach `pub(crate)` items).
All non-contractual.

| Symbol | Kind | Notes |
|---|---|---|
| `REL_TOL_TEST` | `const f64` | `= 1e-9`. Shared tolerance for bench harness. Internal. |
| `__test_only_gpu_sort::sort_indices_on_gpu_multi` | fn (`exec::gpu_sort`) | Internal sort entry point for the sort E2E test. Internal. |
| `__test_only_gpu_sort::GpuSortKey<'a>` | struct (`exec::gpu_sort`) | Internal sort key descriptor. Internal. |
| `__test_only_gpu_sort::SortLayout` | enum (`jit::sort_kernel`) | Internal sort dispatch enum. Internal. |
| `__test_only_sort_kernel::KeyDesc` | struct (`jit::sort_kernel`) | Internal sort-key description. Internal. |
| `__test_only_sort_kernel::SortDirection` | enum (`jit::sort_kernel`) | Internal sort-direction enum. Internal. |
| `__test_only_sort_kernel::SortKernelSpec` | struct (`jit::sort_kernel`) | Internal sort-kernel spec. Internal. |
| `__test_only_logical_plan::DataType` | enum re-export (`plan::logical_plan`) | Mirror of `plan::DataType` reachable through the test-only module path. The canonical public path is `plan::DataType`. Internal. |
| `__test_only_partition_offsets::NUM_PARTITIONS` | `const u32` (`exec::partition_offsets`) | `= 4096`. Internal Tier-2 partition count; tests use it to mirror the kernel constant. Internal. |
| `__test_only_env_vars::pool_stats_interval_from_env` | fn (`exec::engine`) | Internal env-var parser. Internal. |
| `__test_only_env_vars::POOL_STATS_ENV` | `const &str` (`exec::engine`) | `= "BOLT_POOL_STATS_INTERVAL_SECS"`. Internal. |
| `__test_only_env_vars::parse_env_cap` | fn (`exec::gpu_join`) | Internal env-var parser. Internal. |
| `__test_only_env_vars::streaming_intern_enabled` | fn (`exec::gpu_join`) | Internal dispatch flag. Internal. |
| `__test_only_env_vars::CAP_ENV_VAR` | `const &str` (`exec::gpu_join`) | `= "BOLT_GPU_JOIN_TABLE_CAP_MB"`. Internal. |
| `__test_only_env_vars::STREAMING_INTERN_ENV_VAR` | `const &str` (`exec::gpu_join`) | `= "BOLT_GPU_JOIN_STREAMING_INTERN"`. Internal. |
| `__test_only_env_vars::parse_ptx_cache_cap` | fn (`jit::jit_compiler::parse_cap`) | Internal env-var parser. Internal. |
| `__test_only_env_vars::PTX_CACHE_CAP_ENV` | `const &str` (`jit::jit_compiler`) | `= "CRATON_BOLT_PTX_CACHE_CAP"`. Internal. |
| `__test_only_env_vars::DISK_PTX_CACHE_ENV` | `const &str` (`jit::disk_cache`) | `= "BOLT_PTX_CACHE_DIR"`. v0.6 / M6 disk-PTX-cache opt-in env var; re-exported so the env-var smoke test can assert the canonical name. Internal. |

### Other `#[doc(hidden)]` re-exports

| Symbol | Kind | Notes |
|---|---|---|
| `jit::compile_ptx` | fn (`jit::ptx_gen::compile`) | `fn compile(spec: &KernelSpec, kernel_name: &str) -> BoltResult<String>`. Hidden because `KernelSpec` itself is hidden. |
| `jit::compile_and_load` | fn (`jit::jit_compiler`) | `fn compile_and_load(ptx: &str) -> BoltResult<CudaModule>`. Hidden. |
| `jit::CudaFunction<'a>` | struct (`jit::jit_compiler`) | Hidden. JIT-loaded function handle. |
| `jit::CudaModule` | struct (`jit::jit_compiler`) | Hidden. JIT-compiled module handle. |
| `exec::launch_1d` | fn (`exec::launch`) | Hidden. 1-D launch helper. |
| `exec::CudaStream` | struct (`exec::launch`) | Hidden. RAII stream wrapper. |
| `exec::KernelArgs<'a>` | struct (`exec::launch`) | Hidden. Type-erased kernel-args builder. |

---

## Indirectly-reachable surface (not in scope for the freeze)

A handful of crate modules are `pub mod` without being re-exported at the
crate root, so their items are reachable as
`craton_bolt::<module>::<item>` paths. The most prominent ones:

- `craton_bolt::cuda::cuda_sys::*` — raw driver bindings (`init`,
  `device_count`, `device_get`, `device_total_mem`, `device_name`,
  `mem_alloc`, `mem_get_info`, `ctx_get_current`, `check`,
  `CUDA_SUCCESS`, `CUDA_ERROR_STUB`, `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`,
  CUgraph / CUgraphExec aliases, etc.).
- `craton_bolt::cuda::buffer::ARROW_ALIGNMENT` (`const usize = 64`).
- `craton_bolt::exec::gpu_join::*` — the join executor surface
  (`GpuJoinIndices`, `GpuOuterJoinIndices`, `InternedUtf8Columns`,
  `execute_inner_join_on_gpu`, `execute_outer_join_on_gpu`,
  `execute_cross_join_on_gpu`, `execute_utf8_inner_join_on_gpu`,
  `execute_utf8_outer_join_on_gpu`, `hash_join_indices_on_gpu`,
  `intern_utf8_columns`, `compute_device_string_hashes`,
  `GPU_JOIN_MIN_ROWS`, `CROSS_JOIN_GPU_CELL_CAP`, etc.). Although the
  module itself is `#[doc(hidden)]` at `exec/mod.rs`, the items inside
  are `pub`.
- `craton_bolt::exec::partition_offsets::*` —
  `compute_partition_offsets`, `upload_offsets`,
  `compute_and_upload_partition_offsets_async`, `NUM_PARTITIONS`.
- `craton_bolt::jit::*` — every sub-module under `src/jit/` is `pub mod`,
  so individual kernel-compile entry points (e.g.
  `sort_kernel::compile_sort_kernel_spec`,
  `sort_kernel::SortKernelSpec`, `sort_kernel::SortDirection`,
  `sort_kernel::SortLayout`, `sort_kernel::KeyDesc`, etc.) are
  reachable via the full path. These are not part of the public-API
  freeze; the 1.0 freeze applies only to the symbols re-exported above.

The M7 freeze deliberately scopes itself to the re-exports in
`src/lib.rs`. Items reachable only via the full module path are treated
as **hidden** for semver purposes regardless of the absence of
`#[doc(hidden)]` — they exist to support tests, benches, and the
in-progress engine internals, and may move at any release. A future
revision of this document will narrow the `pub mod` surface (changing
the relevant modules to `pub(crate) mod`) so this caveat goes away.
