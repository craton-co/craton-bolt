<!--
applies-to: main
owner: execution
generated-fields: module-inventory (tests/doc_consistency_test.rs)
version-neutral: per-release deltas live in CHANGELOG.md
-->

# Architecture

This document maps Craton Bolt's source tree to the runtime pipeline, and explains the major design decisions. For the SQL → PTX deep-dive read [`JIT_PIPELINE.md`](JIT_PIPELINE.md). For the full supported subset see [`SQL_REFERENCE.md`](SQL_REFERENCE.md); for what changed release to release see [`CHANGELOG.md`](../CHANGELOG.md).

## Layers

| Layer | Directory | Role |
|---|---|---|
| Query front door | `src/exec/engine.rs`, `src/exec/engine_query.rs` | `Engine` / `EngineBuilder`, table registration, the one compile pipeline (`QueryCompiler`), `sql` / `explain_sql` / `run_logical_plan` |
| Planning | `src/plan/` | SQL frontend (sqlparser → `LogicalPlan`, shape classification), `DataFrame` builder, logical optimizer, physical lowering to the SSA-shaped IR |
| Execution | `src/exec/` | per-shape GPU / host executors (projection, aggregates, GROUP BY tiers, joins, sort, distinct, windows, set ops, strings), dictionary registry, device tables, streaming / morsel machinery |
| Code generation | `src/jit/` | one PTX emitter per kernel family, in-process + disk PTX caches, module loading |
| CUDA | `src/cuda/` | cudarc-backed driver access, RAII device buffers and borrow-checked views, the process-wide memory and stream pools, dictionary columns |

The complete, generated module inventory is at the [end of this document](#module-inventory).

## The pipeline

Every query — `Engine::sql`, `Engine::run_logical_plan` / `DataFrame::collect`, `Engine::explain_sql` — enters one pipeline. Every arrow is a function call within the crate.

```
   SQL string                                   DataFrame / LogicalPlan
       │                                                  │
       │ sql_frontend::classify_query                     │
       │   (parse once; recognise host-orchestrated shapes)│
       ▼                                                  │
   QueryShape ── Generic(LogicalPlan) ────────────────────┤
       │                                                  │
       └─ Recursive / Lateral / CorrelatedWhere /         │
          Values / GenerateSeries / DistinctOn /          │
          CountDistinctGroupBy / MultiAggGroupBy          │
            (host loop over component subplans — each     │
             subplan re-enters the compiler below) ───────┤
                                                          ▼
                 QueryCompiler::compile  (exec/engine_query.rs)
                   1. dict_registry.rewrite_plan   (col = 'x' → __idx_col = k)
                   2. optimizer::run_to_fixpoint   (fold, pushdown, filter-into-join,
                                                    stats-driven join reorder, pruning)
                   3. user PlanRewrite hooks
                   4. subquery resolution          (execute + fold to constants;
                                                    EXPLAIN defers and says so)
                                                          │
                                                          ▼
                 physical_plan::lower → PhysicalPlan
                                                          │
                 drain + validate the lazy streams this plan scans;
                 populate per-kernel input validity
                                                          │
                                                          ▼
                 Engine::execute (per-shape dispatch, see below)
                                                          │
                                                          ▼
                 RecordBatch in a QueryHandle (+ planned ExecutionTier)
```

`run_top_level` wraps `sql` / `run_logical_plan`: it counts the statement once in `QueriesTotal`, counts *any* failure in `QueriesFailed`, and performs the throttled pool-stats emit.

### Optimizer

`crate::plan::optimizer` holds the logical passes `default_passes_with_estimator` returns — constant folding, predicate pushdown, filter-into-join, statistics-driven join reordering (row counts snapshotted from the registered tables; lazy tables that have not been drained are deliberately not costed), and projection pruning — driven to a bounded fixpoint by `run_to_fixpoint`. `EngineBuilder::without_optimizer` (test-only) disables it.

### Tables, streaming and residency

- **Eager tables** (`register_table`, `register_batch`, `register_table_stream`) are stored coalesced as **one** host `RecordBatch` per table — multi-batch ingestion concatenates once at ingest, so no query pays a per-query concat. String columns get a dictionary (`dict_registry.rs`) built from the full data.
- **Lazy tables** (`register_table_stream_lazy`) live in a `TableSource::Streaming` overlay. The compile pipeline drains exactly the tables a plan scans, validates every batch against the declared schema, and coalesces them.
- **Device residency.** `ensure_gpu_table` keeps a `GpuTable` per table, versioned by per-column host revisions: unchanged columns are reused in place, appends upload only the new tail (DtoD-preserving the prefix).
- **Morsels.** Under `EngineBuilder::memory_budget`, row-wise leaf scans and distributive aggregates over overlay tables are processed in bounded morsels (`engine_streaming_impl.rs`); everything else materialises its input.

## Memory safety: CUDA-Oxide

The single most important design idea in the crate. GPU memory in C++ is a foot-gun magnet (use-after-free, double-free, concurrent mutation across CPU and GPU). Craton Bolt lifts those problems into Rust's type system:

```
GpuVec<T>           // owned device allocation, drops via cuMemFree on Drop
   │
   │ .view() borrows immutably for the GpuVec's lifetime
   │ .view_mut() borrows mutably (exclusive)
   ▼
GpuView<'a, T>      // PhantomData<(&'a [T], Cell<()>)>, Copy + Clone + Send only — !Sync
GpuViewMut<'a, T>   // PhantomData<&'a mut [T]>, Send only (not Sync, not Copy)
```

`GpuView` was `Send + Sync` in early drafts. It was demoted to `!Sync` in wave 1 because the launch model is concurrency-unsafe at the API surface: a sibling thread holding the parent `GpuVec` can construct a `GpuViewMut` and launch a writer kernel, which would race a reader kernel launched through a shared `GpuView` on this thread. The `Cell<()>` smuggled into the `PhantomData` is what enforces the `!Sync` bound at the type level — it has no runtime cost. `Send` is preserved because moving an exclusive reader across threads is sound; only sharing it is not.

Three properties fall out:

1. **A view can't outlive its `GpuVec`.** The `'a` lifetime is tied to the vec via PhantomData. Drop the vec while a view is borrowed → compile error.
2. **No mutable + shared aliasing.** `view()` borrows `&self`; `view_mut()` borrows `&mut self`. The borrow checker is the proof.
3. **No use-after-move.** Move a `GpuVec` into another scope → the original binding is gone. Use it after the move → compile error.

The three corresponding `compile_fail` doctests live at the top of `tests/memory_tests.rs`. They're the primary correctness contract of the entire CUDA layer.

The one place we step outside this discipline is the `cuLaunchKernel` parameter assembly — the CUDA driver wants `*mut *mut c_void`, and we hand it raw pointers into the kernel-args storage. Those pointers are documented to live for the duration of the launch + synchronize, and the launch itself synchronizes before returning, so the borrow ends before the device memory can be freed. The relevant `unsafe` blocks carry `// SAFETY:` comments.

## Per-shape execution

The engine doesn't have a one-size-fits-all executor. The `Engine::execute` match in `src/exec/engine.rs` dispatches by physical-plan shape, and within `PhysicalPlan::Aggregate` it dispatches again on `(group_by.is_empty(), pre.is_some())`:

| `group_by` non-empty | `pre` is `Some` | Executor                          | Notes                                    |
|----------------------|-----------------|-----------------------------------|------------------------------------------|
| no                   | no              | `aggregate.rs`                    | Trivial scalar reductions.               |
| no                   | yes             | `agg_with_pre.rs`                 | Resident on-device path first (pre kernel reads the `GpuTable` + dictionary index columns, reduces in place); host pre-stage fallback. |
| yes                  | no              | `groupby.rs`                      | Resident Tier-1 shared-memory / Tier-2 partitioned paths; packed-i64-key hash table. |
| yes                  | no, key collides w/ sentinel | `groupby_valid.rs` (fallback in `groupby.rs`) | Sentinel-free; bounded probe + spill. |
| yes                  | no, > 64-bit composite key   | `groupby_wide.rs` (fallback in `groupby.rs`) | Host-side reduction.            |
| yes                  | yes             | `groupby_with_pre.rs`             | Combines pre + GROUP BY pipelines.       |

Two wrappers route aggregates whose input the fused path cannot read:

- `PhysicalPlan::CountRows` — scalar `COUNT` over a non-scan child (e.g. `COUNT(DISTINCT col)` over a `Distinct`).
- `PhysicalPlan::AggregateOverInput` — any other aggregate over a join / union / distinct, or over a filter the fused kernel cannot evaluate (a raw `Utf8` comparison, 3VL over a NULL literal, a `Utf8` key under a predicate): the child runs, its rows are bound as an ephemeral table, and an ordinary `Aggregate` over that table executes.

Host-batch executors that read rewriter-synthesised columns (`__idx_<col>`, `__rank_<col>`) get them from `Engine::materialize_table_for_inputs`, with exactly the values the device path reads. Within each executor, the inner aggregate-input dispatch may further route through `extended_agg.rs` (for Bool/Utf8) or `expr_agg.rs` (for non-bare-column inputs).

The intent is that every routing decision is local and explicit: a contributor adding a new shape adds a new file, registers it in `Engine::execute`, and the existing paths are untouched.

## Dictionary encoding for Utf8

Variable-width strings are a poor fit for fused-codegen GPU kernels. Every comparison would force a pointer-dereference + length check that defeats coalesced loads. Craton Bolt's answer: encode Utf8 columns as dictionaries on the host, ship only fixed-width integer indices to the device.

```
StringArray ──► DictionaryColumn { dictionary: Vec<String>, indices: GpuVec<i32 or i64> }
   │                  │                                       │
   │ (engine reads)   │ (host)                                │ (device)
   │                  │                                       │
   │                  │ shipped to GPU for col equality       │
   │                  │ used to decode results back to strings│
   │                                                          │
   └──────────────────┴──────────────────────────────────────┘
```

The registry (`dict_registry.rs`) builds a dictionary for every Utf8 column at registration (from the table's full, coalesced data — so each dictionary is known-complete and an absent literal folds to a constant), picks i32 or i64 indices based on a distinct-string estimate, and exposes the `rewrite_plan` step of the compile pipeline. The rewriter folds `WHERE region = 'US'` into `WHERE __idx_region = <i32 or i64>(idx)` — pure integer equality, which the standard codegen already handles. Every distinct string owns exactly one index (duplicate Arrow dictionary values are merged on ingest). Lazily-registered tables have no dictionaries; their string predicates are evaluated on the host.

`UPPER` / `LOWER` / `LENGTH` lower to the GPU: `UPPER` / `LOWER` use the two-pass variable-width `PhysicalPlan::StringProject` executor (`string_project.rs`), and `LENGTH` uses the dictionary-gather `PhysicalPlan::StringLength` executor (`string_length.rs`). The GPU case-fold path is ASCII-only (byte-wise) and falls back to the full-Unicode host transform in `string_ops.rs` for any column whose dictionary contains non-ASCII bytes; both paths produce identical results for ASCII data. `LENGTH` counts characters (Unicode codepoints), not bytes (`OCTET_LENGTH` covers byte length). `SUBSTRING` / `TRIM` / `CONCAT` still run as pure-host dictionary transformations (`string_ops.rs`, `string_ops_extended.rs`) — CONCAT builds a new dictionary via cross-product on the host — because their variable-width device-write paths are not yet wired through the codegen path.

## Filter compaction

Two paths coexist:

1. **GPU-side** (`gpu_compact.rs`): the projection kernel emits a u8 mask alongside its outputs; a per-block Hillis-Steele prefix scan computes per-row exclusive offsets and per-block sums; the host does a small reduction over block sums (or recursively device-scans for very large inputs via `gpu_compact_multipass.rs`); a per-dtype gather kernel writes only the surviving rows into compacted outputs. This is the default.
2. **Host-side** (`compact.rs`): downloads the mask and applies `arrow::compute::filter` per column. Used as the fallback when the output schema contains Utf8 columns the gather kernel can't move.

The dispatch lives in `engine.rs::execute_projection` and is purely dtype-based: any Utf8 output → host-side, else → GPU-side.

## GROUP BY hash table

Single-pass open-addressing on the GPU. The host estimates K (the table size, rounded up to a power of two so probe becomes a mask-and). The classic kernel uses `i64::MIN` as an empty-slot sentinel and `atom.cas.b64` for insertion. For inputs that may legitimately encode to `i64::MIN` (notably Float64 `-0.0`, which is `0x8000_0000_0000_0000`), the fallback kernel in `valid_flag_kernels.rs` uses a parallel `slot_valid: u32[]` table — slot states {0 = empty, 1 = claimed, 2 = committed} — with bounded probe and SPIN loops that spill on overflow. The host folds the spill into the final result. The validity-bitmap variant (`compile_groupby_keys_kernel_with_validity` in `hash_kernels.rs`) threads a per-row valid bit through the same machinery so NULL-tolerant GROUP BY stays on the device end-to-end.

Float MIN/MAX over `Float32` / `Float64` has no native `atom.global.min/max.f*` on sm_70, so `float_atomics.rs` and `valid_flag_float.rs` synthesise it via a CAS loop on the bit pattern.

## Multi-pass prefix scan

`prefix_scan.rs` tops out at `n_rows ≤ u32::MAX / BLOCK_SIZE ≈ 16.8M` (the host-side scan over block_sums assumes the array fits comfortably). For larger inputs `prefix_scan_multipass.rs` recursively scans the block_sums via the same kernel, building a depth-3-or-4 ladder of prefix-summed arrays. The dispatch lives at the top of `gpu_compact.rs::prefix_scan_mask` — if `n_rows > limit`, delegate.

## What's deliberately *not* in this architecture

- **No process model.** Everything runs in a single Rust process.
- **One active `Engine` per process.** The memory pool, stream pool and module caches are process-global and bound to the CUDA context; `EngineBuilder::build` rejects a second live engine. Sequential engines are supported (and stress-tested). A shared-runtime / many-session model is post-1.0 work.
- **No multi-GPU per engine.** One context, one device per `Engine`; `Engine::new_with_device(idx)` picks the device.
- **Host memory bounds the data set.** Tables are host-resident Arrow batches; the morsel path bounds *device* working sets for row-wise scans and distributive aggregates, but sorts, joins, windows, distinct and set operations materialise their inputs. Spill-capable operators are open work.
- **Non-equi joins are limited.** Non-equi predicates run through the host nested-loop fallback (`execute_nested_loop_join`), INNER-only and capped (`MAX_NESTED_LOOP_INNER_ROWS`).
- **Host-orchestrated SQL shapes are host loops.** Recursive CTEs and correlated applies share the compile pipeline but are not physical operators; their cost is bounded by `CRATON_MAX_RECURSIVE_ITERATIONS` / `CRATON_MAX_APPLY_ROWS` (see [`ENV_VARS.md`](ENV_VARS.md)).
- **Async memcpy coverage is partial.** Projection, scalar aggregates, the GROUP BY family and filter D2H use pinned async transfers; the join executor still issues most H2D uploads synchronously.
- **Caches.** The PTX cache keys on the emitted PTX hash (skips PTXAS); the `KernelSpec`-keyed module cache skips codegen and PTXAS; the SQL plan cache memoises `LogicalPlan`s per schema version, and the shape classifier memoises "ordinary query" verdicts so repeats skip shape detection.

See [`LIMITATIONS.md`](LIMITATIONS.md) for the user-facing consequences and [`ROADMAP.md`](../ROADMAP.md) for the plan.

## Stability of the IR types

`PhysicalPlan`, `KernelSpec`, `AggregateSpec`, `Op`, `Reg`, `Value`, `ColumnIO`, and the rest of the physical-plan / IR vocabulary in `src/plan/physical_plan.rs` are marked `#[doc(hidden)]`. They are implementation-internal: the codegen + executor split owns them end-to-end (the one public entry point that accepts one, `Engine::execute(&PhysicalPlan)`, is for tests and experiments), and they may change shape, gain variants, or be replaced in any pre-1.0 release without a deprecation cycle. External code that wants to drive Craton Bolt should hold to the `Engine` and `DataFrame` surface; the planner and IR are explicitly out-of-contract until the 1.0 API freeze.

## Module inventory

Generated from each file's `//!` summary line by `tests/doc_consistency_test.rs`
(`architecture_module_inventory_is_current`); refresh with
`BOLT_REGENERATE_DOCS=1 cargo test --test doc_consistency_test --no-default-features --features cuda-stub`.

<!-- module-inventory:begin (generated — do not edit by hand) -->
| Module | Summary |
|---|---|
| `src/cuda/async_copy.rs` | Pinned-host-buffer + async H2D/D2H copy helpers used by the executors to overlap transfers with compute. |
| `src/cuda/buffer.rs` | Arrow-compatible columnar GPU storage primitives. |
| `src/cuda/cuda_sys.rs` | Raw FFI bindings and thin safe wrappers around the CUDA Driver API. |
| `src/cuda/cudarc_backend.rs` | The supported CUDA driver backend, built on `cudarc` (the default `cudarc` feature). |
| `src/cuda/dictionary.rs` | Host-side string dictionary paired with on-device i32 indices. |
| `src/cuda/dictionary_any.rs` | Unified wrapper over the i32- and i64-indexed dictionary variants. |
| `src/cuda/dictionary_i64.rs` | Host-side string dictionary paired with on-device **i64** indices. |
| `src/cuda/mem_pool.rs` | Process-wide device-memory pool / arena allocator. |
| `src/cuda/mod.rs` | CUDA layer: raw driver bindings plus higher-level RAII wrappers. |
| `src/cuda/smart_ptrs.rs` | Lifetime-tracked typed handles to GPU memory ("CUDA-Oxide"). |
| `src/cuda/stream_pool.rs` | Process-global pool of owned CUDA streams. |
| `src/error.rs` | The engine-wide error type, [`BoltError`], and its [`BoltResult`] alias. |
| `src/exec/agg_with_pre.rs` | Aggregate execution with a pre-projection / filter pass. |
| `src/exec/aggregate.rs` | Scalar (no GROUP BY) aggregate execution. |
| `src/exec/compact.rs` | Host-side filter compaction. |
| `src/exec/dict_registry.rs` | Per-table dictionary registry that drives the string-literal predicate rewrite. |
| `src/exec/distinct.rs` | DISTINCT executor — host-side deduplication of a RecordBatch. |
| `src/exec/engine.rs` | Top-level engine: dispatches per-shape executors (scalar agg, GROUP BY, etc.); performs GPU prefix-scan + gather compaction for filter outputs, or a host-side `arrow::compute::filter` fallback when any output column is Utf8. |
| `src/exec/engine_agg_impl.rs` | Host-side aggregate / row-source query executors for [`Engine`]. |
| `src/exec/engine_cache_key.rs` | Module-cache key + host-revision snapshot types lifted out of `exec::engine` (pure reorganization; no behavior change). |
| `src/exec/engine_device_col.rs` | Heterogenous owned device output column ([`DeviceCol`]) and its download / staging helpers, lifted out of `exec::engine` (pure reorganization; no behavior change). |
| `src/exec/engine_join_impl.rs` | Reserved sibling module for the join execution / dispatch methods of [`crate::exec::engine::Engine`]. |
| `src/exec/engine_provider.rs` | Planner-facing adapters over the engine's registered tables, lifted out of `exec::engine` (pure reorganization; no behavior change). |
| `src/exec/engine_query.rs` | The SQL / DataFrame query front door of [`Engine`]: `sql`, `explain_sql`, `run_logical_plan` and the internal `run_subplan`, plus the single [`QueryCompiler`] pipeline every one of them shares. |
| `src/exec/engine_streaming_impl.rs` | Streaming / morsel-driven execution methods for [`Engine`]. |
| `src/exec/engine_string_impl.rs` | String-projection execution methods for [`Engine`]. |
| `src/exec/engine_support.rs` | Standalone free helper functions lifted out of `exec::engine` (pure reorganization; no behavior change). |
| `src/exec/expr_agg.rs` | Host-side expression evaluator for aggregate inputs. |
| `src/exec/extended_agg.rs` | Host-side aggregate execution for the dtypes the GPU reduction kernels don't cover: `Bool` (any aggregate) and `Utf8` (COUNT/MIN/MAX only). |
| `src/exec/filter.rs` | Host-side post-aggregate filter executor. |
| `src/exec/gpu_compact.rs` | GPU-side filter compaction: prefix-scan + gather, end-to-end. |
| `src/exec/gpu_compact_multipass.rs` | Multi-pass prefix-scan compaction. |
| `src/exec/gpu_join.rs` | GPU hash join (INNER, LEFT / RIGHT / FULL OUTER, CROSS): build a hash table on the GPU from the smaller (build) side, probe it with the larger (probe) side, materialise the result host-side via `arrow::compute::take`. |
| `src/exec/gpu_sort.rs` | GPU-side ORDER BY: sort key columns on the device (stable LSD radix for supported Int32/Int64 keys, bitonic otherwise), gather every other column on the host. |
| `src/exec/gpu_table.rs` | GPU-resident table storage: columns uploaded once and queried in place. |
| `src/exec/gpu_upload.rs` | Shared async-memcpy / pinned-host-buffer helpers for executors. |
| `src/exec/groupby.rs` | GROUP BY aggregate execution. |
| `src/exec/groupby_common.rs` | Shared host-side helpers for the GROUP BY execution family. |
| `src/exec/groupby_shmem_avg_exec.rs` | Per-block shared-memory **AVG** executor (Tier 1 fast path, AVG flavour). |
| `src/exec/groupby_shmem_count_exec.rs` | Tier-1 **COUNT(*) GROUP BY** executor — shared-memory pre-aggregation for low-cardinality `SELECT key, COUNT(*) FROM x GROUP BY key`. |
| `src/exec/groupby_shmem_dispatch.rs` | Tier-1 GROUP BY dispatch: pick between the existing global-atomic kernel and the new per-block shared-memory pre-aggregation kernel. |
| `src/exec/groupby_shmem_exec.rs` | Per-block shared-memory pre-aggregation **executor** (Tier 1 fast path). |
| `src/exec/groupby_shmem_launch.rs` | Launch-parameter auto-tuner for the per-block shared-memory GROUP BY kernel (Tier 1 of the GROUP BY performance plan). |
| `src/exec/groupby_shmem_minmax_exec.rs` | Tier-1 **MIN / MAX** GROUP BY executor (low-cardinality integer MIN/MAX over `n_groups <= BLOCK_GROUPS`). |
| `src/exec/groupby_shmem_multi_exec.rs` | Per-block shared-memory pre-aggregation **executor** with multiple SUM aggregates folded into a single kernel launch (Tier-1 extension). |
| `src/exec/groupby_tier2_avg_exec.rs` | **AVG at Tier 2.1** — high-cardinality multi-AVG executor. |
| `src/exec/groupby_tier2_common.rs` | dedup (tier2/shmem): single home for the genuinely-identical host-side eligibility-gate helpers shared by the Tier-1 (shmem) and Tier-2 GROUP BY executors. |
| `src/exec/groupby_tier2_count_exec.rs` | **COUNT(*) at Tier 2.1** — high-cardinality `SELECT key, COUNT(*) FROM x GROUP BY key` executor. |
| `src/exec/groupby_tier2_dispatch.rs` | Tier-2 GROUP BY dispatch (v2): pick between the existing global-atomic kernel, the Tier-1 per-block shared-memory pre-aggregation kernel, and the new Tier-2 hash-partitioned two-pass kernel. |
| `src/exec/groupby_tier2_exec.rs` | Tier-2 hash-partitioned GROUP BY executor (top-level shim). |
| `src/exec/groupby_tier2_merge.rs` | Tier-2 hash-partitioned GROUP BY **result merger**. |
| `src/exec/groupby_tier2_minmax_exec.rs` | **MIN / MAX at Tier 2.1** вЂ” high-cardinality executor for `SELECT key, {MIN,MAX}(val) FROM x GROUP BY key`. |
| `src/exec/groupby_tier2_minmax_float_exec.rs` | Tier-2.1 **MIN / MAX over Float32 / Float64** executor. |
| `src/exec/groupby_tier2_multi_exec.rs` | Tier-2 hash-partitioned GROUP BY executor for **multiple SUM** aggregates (top-level shim). |
| `src/exec/groupby_tier2_multi_merge.rs` | Tier-2 hash-partitioned GROUP BY **multi-SUM** result merger. |
| `src/exec/groupby_tier2_multi_orchestrator.rs` | Tier-2 hash-partitioned GROUP BY orchestrator for **multiple SUM** aggregates. |
| `src/exec/groupby_tier2_orchestrator.rs` | Tier-2 hash-partitioned GROUP BY SUM orchestrator (head of the chain). |
| `src/exec/groupby_tier2_twokey_avg_exec.rs` | **Two-key multi-AVG at Tier 2.1** — high-cardinality `SELECT a, b, AVG(v1), AVG(v2), … FROM x GROUP BY a, b` executor. |
| `src/exec/groupby_tier2_twokey_count_exec.rs` | **Two-key COUNT(*) at Tier 2.1** — high-cardinality `SELECT a, b, COUNT(*) FROM x GROUP BY a, b` executor. |
| `src/exec/groupby_tier2_twokey_exec.rs` | Tier-2 hash-partitioned GROUP BY executor — **two-key (Int32, Int32) shim**. |
| `src/exec/groupby_tier2_twokey_merge.rs` | Tier-2 hash-partitioned GROUP BY **two-key result merger**. |
| `src/exec/groupby_tier2_twokey_minmax_exec.rs` | **Two-key MIN / MAX at Tier 2.1** вЂ” high-cardinality executor for `SELECT a, b, {MIN,MAX}(v) FROM x GROUP BY a, b` over **integer** value columns (Int32 / Int64). |
| `src/exec/groupby_tier2_twokey_minmax_float_exec.rs` | **Two-key float MIN / MAX at Tier 2.1** — high-cardinality executor for `SELECT a, b, {MIN,MAX}(v) FROM x GROUP BY a, b` over `Float64` value columns. |
| `src/exec/groupby_tier2_twokey_multi_exec.rs` | **Two-key multi-aggregate Tier-2.1** executor. |
| `src/exec/groupby_tier2_twokey_orchestrator.rs` | Tier-2 hash-partitioned GROUP BY SUM orchestrator — **two-key (i64-packed) variant**. |
| `src/exec/groupby_valid.rs` | GROUP BY execution using the sentinel-free (slot-valid-flag) protocol. |
| `src/exec/groupby_wide.rs` | GROUP BY execution for "wide" composite keys — those whose total bit width exceeds 64 (e.g. |
| `src/exec/groupby_with_pre.rs` | Fused pre-projection + GROUP BY aggregate execution. |
| `src/exec/join.rs` | JOIN executor — host-side hash join over equi-join predicates, with INNER / LEFT / RIGHT / FULL / CROSS support. |
| `src/exec/launch.rs` | Kernel launch glue: turns `CudaFunction` + `GpuView`s into a 1D grid launch. |
| `src/exec/like.rs` | Host-side SQL `LIKE` evaluator. |
| `src/exec/limit.rs` | LIMIT / OFFSET executor — slice the input RecordBatch. |
| `src/exec/mod.rs` | Execution: the [`Engine`](engine::Engine) query front door, the per-shape GPU / host executors, and the device-table, dictionary and streaming machinery they share. |
| `src/exec/module_cache.rs` | Process-wide JIT module cache shared across executors. |
| `src/exec/partition_offsets.rs` | Exclusive prefix-sum offsets for Tier-2 hash-partitioned GROUP BY. |
| `src/exec/schema_convert.rs` | Single-source-of-truth conversions between the plan-layer `DataType` / `Schema` and Arrow's `DataType` / `Schema`. |
| `src/exec/setops.rs` | `EXCEPT` / `INTERSECT` executor — host-side multiset set operations. |
| `src/exec/sort.rs` | ORDER BY executor — host-side sort of a RecordBatch. |
| `src/exec/streaming.rs` | Morsel / chunk streaming abstractions for bounded, larger-than-VRAM execution. |
| `src/exec/string_col.rs` | Device columns for non-primitive dtypes: `Bool` and `Utf8`. |
| `src/exec/string_length.rs` | Executor for the fully-GPU `SELECT LENGTH(<utf8_col>)` projection ([`crate::plan::physical_plan::PhysicalPlan::StringLength`]). |
| `src/exec/string_like.rs` | Executor helpers for the GPU per-row `LIKE` matcher over **variable-width (non-dictionary) `Utf8`** columns ([`crate::plan::physical_plan::PhysicalPlan::StringLikeFilter`]). |
| `src/exec/string_ops.rs` | Dictionary-aware string operations: `UPPER`, `LOWER`, `LENGTH`, and literal-equality on `DictionaryColumn`. |
| `src/exec/string_ops_extended.rs` | Dictionary-aware extended string operations: `CONCAT`, `SUBSTRING`, `CONCAT_WS`. |
| `src/exec/string_project.rs` | Executor helpers for the GPU variable-width string projection ([`crate::plan::physical_plan::PhysicalPlan::StringProject`]) — currently `UPPER` / `LOWER` over a `Utf8` column. |
| `src/exec/subquery_resolve.rs` | Pre-lowering resolution of uncorrelated subqueries. |
| `src/exec/validity_audit.rs` | NULL / validity propagation audit + shared validity helpers. |
| `src/exec/welford.rs` | Welford's one-pass algorithm for numerically-stable mean / variance. |
| `src/exec/window.rs` | Host-side window-function executor. |
| `src/flight/encode.rs` | Arrow ↔ Arrow Flight result-encoding helpers (feature `flight`). |
| `src/flight/mod.rs` | Arrow Flight SQL service skeleton (feature `flight`). |
| `src/flight/server.rs` | `FlightService` trait implementation for [`FlightSqlServer`]. |
| `src/flight/sql.rs` | Arrow Flight SQL — command layer (feature `flight`). |
| `src/jit/agg_kernels.rs` | PTX codegen for scalar reduction kernels. |
| `src/jit/date_scalar.rs` | PTX codegen + host reference math for the date/time scalar functions `EXTRACT(field FROM ts)` and `DATE_TRUNC(unit, ts)`. |
| `src/jit/decimal_agg.rs` | PTX codegen for the `SUM(Decimal128)` reduction kernel. |
| `src/jit/disk_cache.rs` | Optional disk-backed PTX cache. |
| `src/jit/distinct_kernel.rs` | PTX codegen for the **adjacent-distinct flag** kernel used by the sort-based GPU DISTINCT path. |
| `src/jit/float_atomics.rs` | PTX codegen for GROUP BY `MIN` / `MAX` aggregates over floating-point inputs. |
| `src/jit/hash_join_kernel.rs` | PTX codegen for the GPU INNER-JOIN build + probe kernels. |
| `src/jit/hash_kernels.rs` | PTX codegen for the single-pass open-addressing GPU hash-grouping kernels. |
| `src/jit/jit_compiler.rs` | PTX → loaded CUDA module via the driver's in-process assembler. |
| `src/jit/mod.rs` | PTX code generation and module loading: one emitter per kernel family, plus the in-process and disk PTX caches. |
| `src/jit/partition_kernel.rs` | Hash-partition pass-1 kernel (Tier 2 of the GROUP BY perf plan). |
| `src/jit/partition_kernel_i64.rs` | Hash-partition pass-1 kernel for **64-bit packed keys** (Tier 2 of the GROUP BY perf plan). |
| `src/jit/partition_reduce_kernel.rs` | Per-partition shared-memory GROUP BY SUM kernel (Tier 2.1). |
| `src/jit/partition_reduce_kernel_count.rs` | Per-partition shared-memory GROUP BY **COUNT(*)** kernel — Tier 2.1 companion to `partition_reduce_kernel_multi`. |
| `src/jit/partition_reduce_kernel_count_i64.rs` | Per-partition COUNT(*) reduce kernel — **i64 key variant**. |
| `src/jit/partition_reduce_kernel_i64.rs` | Per-partition shared-memory GROUP BY SUM kernel — **Int64-key variant** (Tier 2.1 for two-key GROUP BY). |
| `src/jit/partition_reduce_kernel_minmax.rs` | Per-partition shared-memory GROUP BY **MIN / MAX** kernel (Tier 2.1). |
| `src/jit/partition_reduce_kernel_minmax_float.rs` | Per-partition shared-memory **float MIN / MAX** kernel — Tier 2.1 variant for `Float32` and `Float64` value dtypes. |
| `src/jit/partition_reduce_kernel_minmax_float_i64.rs` | Per-partition shared-memory **float MIN / MAX** kernel — i64-key variant (Tier 2.1 for two-key float MIN/MAX). |
| `src/jit/partition_reduce_kernel_minmax_i64.rs` | Per-partition shared-memory GROUP BY **MIN / MAX** kernel — i64-key variant (Tier 2.1 for two-key MIN/MAX). |
| `src/jit/partition_reduce_kernel_multi.rs` | Per-partition shared-memory GROUP BY SUM kernel — **multi-value variant** (Tier 2.1 for multi-aggregate workloads). |
| `src/jit/partition_reduce_kernel_multi_i64.rs` | Per-partition shared-memory reduce kernel — **i64-key, multi-value SUM**. |
| `src/jit/partition_reduce_kernel_spill_common.rs` | Shared PTX-emission helpers for `partition_reduce_kernel*` `_with_spill` variants. |
| `src/jit/prefix_scan.rs` | PTX codegen for GPU-side filter compaction (prefix scan + gather). |
| `src/jit/prefix_scan_multipass.rs` | PTX codegen for the multi-pass prefix-scan compaction path. |
| `src/jit/ptx_gen.rs` | PTX codegen: lower a `KernelSpec` into a complete PTX module string. |
| `src/jit/scan_kernel.rs` | PTX codegen for the predicate-only "scan" kernel used by filter compaction. |
| `src/jit/scatter_kernel.rs` | Scatter kernel for Tier-2 hash-partitioned GROUP BY. |
| `src/jit/scatter_kernel_i64.rs` | Scatter kernel for Tier-2 hash-partitioned GROUP BY — **i64-key sibling**. |
| `src/jit/scatter_values_by_dest_idx_kernel.rs` | Deterministic value-column scatter using a precomputed `dest_idx[n_rows]` buffer. |
| `src/jit/scatter_with_dest_idx_kernel.rs` | Scatter kernel **that also records the per-row destination index**, for deterministic multi-column scatter. |
| `src/jit/shmem_count_kernel.rs` | Per-block shared-memory GROUP BY **COUNT** kernel (Tier 1 fast path). |
| `src/jit/shmem_minmax_kernel.rs` | Tier-1 per-block shared-memory **MIN / MAX** kernel. |
| `src/jit/shmem_multi_sum_kernel.rs` | Per-block shared-memory GROUP BY kernel with **multiple SUM aggregates** computed in a single pass (Tier-1 extension). |
| `src/jit/shmem_sum_kernel.rs` | Per-block shared-memory GROUP BY SUM kernel (Tier 1 of the GROUP BY perf plan). |
| `src/jit/sort_kernel.rs` | PTX codegen for a **bitonic sort** kernel. |
| `src/jit/sort_kernel_radix.rs` | PTX codegen for a **single-pass 4-bit-radix sort** kernel. |
| `src/jit/string_kernel.rs` | PTX codegen for GPU-side variable-width (string / `Utf8`) scalar functions. |
| `src/jit/valid_flag_float.rs` | PTX codegen for the sentinel-free GROUP BY `MIN(float)` / `MAX(float)` aggregate kernel. |
| `src/jit/valid_flag_kernels.rs` | Sentinel-free open-addressing GROUP BY kernels. |
| `src/jit/window_kernel.rs` | PTX codegen for GPU-side **framed window-function** execution. |
| `src/lib.rs` | Craton Bolt — JIT-compiled GPU SQL engine. |
| `src/metrics.rs` | Lightweight, dependency-free metrics registry. |
| `src/observability.rs` | Observability surface — pool-stats observer + tracing span catalogue. |
| `src/plan/dataframe.rs` | Lazy DataFrame builder over [`LogicalPlan`]. |
| `src/plan/explain.rs` | Plan introspection / `EXPLAIN`. |
| `src/plan/logical_plan.rs` | Logical plan AST: schemas, expressions, and relational nodes. |
| `src/plan/mod.rs` | Logical plan IR, DataFrame builder, SQL frontend, and physical plan. |
| `src/plan/optimizer/const_fold.rs` | Constant folding and algebraic simplification of [`Expr`] trees. |
| `src/plan/optimizer/cost.rs` | Cost model and join-order enumeration for the cost-based optimizer (CBO). |
| `src/plan/optimizer/expr_util.rs` | Shared expression helpers for the logical optimizer passes. |
| `src/plan/optimizer/filter_into_join.rs` | Filter-into-join: fold a post-join `Filter` that references *both* join inputs into the join's residual `filter` slot. |
| `src/plan/optimizer/join_reorder.rs` | Cost-based join reordering for chained INNER equi-joins. |
| `src/plan/optimizer/mod.rs` | Built-in logical-plan optimizer: a fixed pipeline of [`PlanRewrite`] passes the engine runs by default before lowering to the physical plan. |
| `src/plan/optimizer/plan_util.rs` | Structural plan-walking helpers shared by the optimizer passes. |
| `src/plan/optimizer/predicate_pushdown.rs` | Predicate pushdown: move `WHERE` conjuncts as close to the data source as correctness allows. |
| `src/plan/optimizer/projection_pruning.rs` | Projection pruning: stop reading columns no operator upstream needs. |
| `src/plan/physical_plan.rs` | Physical plan: column-ordinal-resolved, register-machine IR for GPU codegen. |
| `src/plan/rewrite.rs` | Public optimizer extension surface: the [`PlanRewrite`] trait. |
| `src/plan/sql_frontend.rs` | SQL frontend: parses a SQL string into a `LogicalPlan` against a `TableProvider`. |
| `src/plan/statistics.rs` | Cardinality / statistics estimation for the cost-based optimizer. |
| `src/plan/string_literal_rewrite.rs` | String-literal predicate rewriting. |
| `src/plan/subquery.rs` | Uncorrelated-subquery support for the SQL frontend. |
| `src/plan/substrait/expr.rs` | Substrait `Expression` → engine [`Expr`] conversion (feature `substrait`). |
| `src/plan/substrait/mod.rs` | Substrait plan ingestion (feature `substrait`). |
| `src/plan/substrait/rel.rs` | Substrait `Rel` → engine [`LogicalPlan`] conversion (feature `substrait`). |
| `src/plan/suggest.rs` | "Did you mean...?" suggestion helpers for unknown SQL identifiers. |
<!-- module-inventory:end -->
