# Architecture

This document maps Craton Bolt's source tree to the runtime pipeline, and explains the major design decisions. For the SQL → PTX deep-dive read [`JIT_PIPELINE.md`](JIT_PIPELINE.md). For the full supported subset see [`SQL_REFERENCE.md`](SQL_REFERENCE.md).

> **What changed since 0.1.** 0.3.0 added `INNER JOIN`, multi-batch tables, `DISTINCT` / `LIMIT [OFFSET]` / `ORDER BY` / `HAVING` / `UNION [ALL]`, a process-wide PTX module cache, the `cuda-stub` feature, and the Tier-1 / Tier-2 hash-partitioned GROUP BY family. The layer cake and exclusion list below reflect 0.3.0; see [`CHANGELOG.md`](../CHANGELOG.md) for the full delta and [`ROADMAP.md`](../ROADMAP.md) for what's still open.

## Layer cake

```
┌─────────────────────────────────────────────────────────────────────┐
│  src/exec/engine.rs        Engine        ── public surface          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Per-shape executors (src/exec/)  ─────────────────────────         │
│    Scalar / classic GROUP BY                                        │
│      aggregate.rs                   scalar SUM/MIN/MAX/COUNT/AVG    │
│      agg_with_pre.rs                scalar agg + pre kernel         │
│      groupby.rs                     packed-i64-key GROUP BY         │
│      groupby_with_pre.rs            pre + GROUP BY                  │
│      groupby_wide.rs                host fallback for >64-bit keys  │
│      groupby_valid.rs               sentinel-free GROUP BY          │
│      extended_agg.rs                Bool/Utf8 aggregate inputs      │
│      expr_agg.rs                    host-side expr evaluator        │
│    Tier-1 shared-memory GROUP BY                                    │
│      groupby_shmem_*.rs             single-block shared-memory      │
│                                     hash GROUP BY (count / sum /    │
│                                     minmax / multi + dispatch +     │
│                                     launch wrappers)                │
│    Tier-2 hash-partitioned GROUP BY                                 │
│      groupby_tier2_*.rs             partition → per-partition       │
│                                     reduce → host merge; single-    │
│                                     and two-key variants across     │
│                                     count / sum / avg / minmax      │
│                                     (int and float) / multi-agg     │
│      partition_offsets.rs           per-partition output offsets    │
│    Standalone relational operators                                  │
│      join.rs                        host-side INNER hash-equi join  │
│      sort.rs                        host-side ORDER BY              │
│      distinct.rs                    host-side DISTINCT              │
│      limit.rs                       host-side LIMIT [OFFSET]        │
│      gpu_table.rs                   multi-batch GPU table model     │
│    Filter compaction                                                │
│      compact.rs                     host-side filter compaction     │
│      gpu_compact.rs                 GPU-side compaction             │
│      gpu_compact_multipass.rs       multi-pass scan driver          │
│    Glue                                                             │
│      launch.rs                      CudaStream, kernel launch glue  │
│      dict_registry.rs               per-table Utf8 dictionaries     │
│      string_col.rs                  Bool/Utf8 device columns        │
│      string_ops.rs                  UPPER / LOWER / LENGTH          │
│      string_ops_extended.rs         CONCAT / SUBSTRING              │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PTX codegen + module loading (src/jit/)  ─────────────────         │
│    Classic kernels                                                  │
│      ptx_gen.rs                     projection kernels              │
│      scan_kernel.rs                 predicate-only kernels          │
│      agg_kernels.rs                 per-block reductions            │
│      hash_kernels.rs                GROUP BY hash insert / agg      │
│      float_atomics.rs               float MIN/MAX via atom.cas      │
│      valid_flag_kernels.rs          sentinel-free GROUP BY kernels  │
│      valid_flag_float.rs            sentinel-free float MIN/MAX     │
│      prefix_scan.rs                 Hillis-Steele scan + gather     │
│      prefix_scan_multipass.rs       recursive scan over block sums  │
│      jit_compiler.rs                cuModuleLoadDataEx + PTX cache  │
│    Tier-1 shared-memory GROUP BY kernels                            │
│      shmem_count_kernel.rs          shared-memory COUNT             │
│      shmem_sum_kernel.rs            shared-memory SUM               │
│      shmem_minmax_kernel.rs         shared-memory MIN/MAX           │
│      shmem_multi_sum_kernel.rs      multi-aggregate SUM             │
│    Tier-2 partition + per-partition reduce kernels                  │
│      partition_kernel{,_i64}.rs     key → partition assignment      │
│      scatter_kernel{,_i64}.rs       per-partition row scatter       │
│      partition_reduce_kernel*.rs    per-partition reductions:       │
│                                     count, sum, minmax (int/float), │
│                                     multi-agg; i32 and i64 keys     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Plan & IR  ───────────────────────────────────────────────         │
│    src/plan/logical_plan.rs         LogicalPlan AST, Expr, dtypes   │
│    src/plan/dataframe.rs            lazy builder                    │
│    src/plan/sql_frontend.rs         sqlparser → LogicalPlan         │
│    src/plan/physical_plan.rs        IR + lowering                   │
│    src/plan/string_literal_rewrite.rs  predicate-literal rewrite     │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  CUDA layer  ──────────────────────────────────────────────         │
│    src/cuda/cuda_sys.rs             raw driver FFI                  │
│    src/cuda/buffer.rs               GpuBuffer<T> (Arrow-aligned)    │
│    src/cuda/smart_ptrs.rs           GpuVec<T> + borrow-checked views│
│    src/cuda/mem_pool.rs             reusable device-allocation pool │
│    src/cuda/cudarc_backend.rs       optional cudarc backend shim    │
│    src/cuda/dictionary.rs           DictionaryColumn (i32 indices)  │
│    src/cuda/dictionary_i64.rs       DictionaryColumnI64             │
│    src/cuda/dictionary_any.rs       unified enum, cardinality-driven│
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## The pipeline

A `Engine::sql(query)` call walks the following stages. Every arrow below is a function call within the crate — no extra processes, no FFI to a query engine, just Rust + the CUDA driver.

```
   user SQL string
          │
          │ sqlparser::parse_sql
          ▼
    sqlparser AST
          │
          │ sql_frontend::lower
          ▼
     LogicalPlan
          │
          │ dict_registry.rewrite_plan
          │   (col = 'X' → __idx_col = i32/i64(idx))
          ▼
     LogicalPlan          ◄── DataFrame builder lands here too
          │
          │ physical_plan::lower
          ▼
    PhysicalPlan { Projection { KernelSpec } | Aggregate { pre?, AggregateSpec } }
          │
          │ engine::execute dispatch:
          ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  Per-shape executors                                              │
   │    Projection                  → execute_projection               │
   │    Aggregate w/o GROUP BY w/o pre  → aggregate                    │
   │    Aggregate w/o GROUP BY  w/ pre  → agg_with_pre                 │
   │    Aggregate w/  GROUP BY w/o pre  → groupby (or _valid, _wide)   │
   │    Aggregate w/  GROUP BY  w/ pre  → groupby_with_pre             │
   │    Bool / Utf8 aggregate input     → extended_agg                 │
   └──────────────────────────────────────────────────────────────────┘
          │
          │  per executor:
          ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │  1. Upload host RecordBatch columns → DeviceCol::I32/I64/F32/.../Utf8 │
   │  2. JIT-compile PTX via jit::*::compile_*_kernel                  │
   │  3. CudaModule::from_ptx → cuModuleGetFunction                    │
   │  4. cuLaunchKernel (block 256, grid ceil(n/256))                  │
   │  5. cuStreamSynchronize                                           │
   │  6. (filter) prefix-scan + gather OR host-side filter             │
   │  7. Download outputs → Arrow arrays                               │
   └──────────────────────────────────────────────────────────────────┘
          │
          ▼
     RecordBatch wrapped in QueryHandle
```

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

The engine doesn't have a one-size-fits-all executor. The `Engine::execute` match in `src/exec/engine.rs` dispatches by physical-plan shape, and within `PhysicalPlan::Aggregate` it dispatches again on `(group_by.is_empty(), pre.is_some())`. There are six aggregate-side branches today:

| `group_by` non-empty | `pre` is `Some` | Executor                          | Notes                                    |
|----------------------|-----------------|-----------------------------------|------------------------------------------|
| no                   | no              | `aggregate.rs`                    | Trivial scalar reductions.               |
| no                   | yes             | `agg_with_pre.rs`                 | Run pre kernel, then reduce.             |
| yes                  | no              | `groupby.rs`                      | Packed-i64-key hash table on GPU.        |
| yes                  | no, key collides w/ sentinel | `groupby_valid.rs` (fallback in `groupby.rs`) | Sentinel-free; bounded probe + spill. |
| yes                  | no, > 64-bit composite key   | `groupby_wide.rs` (fallback in `groupby.rs`) | Host-side reduction.            |
| yes                  | yes             | `groupby_with_pre.rs`             | Combines pre + GROUP BY pipelines.       |

Within each executor, the inner aggregate-input dispatch may further route through `extended_agg.rs` (for Bool/Utf8) or `expr_agg.rs` (for non-bare-column inputs).

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

The registry (`dict_registry.rs`) builds a dictionary for every Utf8 column at `register_table` time, picks i32 or i64 indices based on a distinct-string estimate, and exposes a `rewrite_plan` method that the engine calls before lowering. The rewriter folds `WHERE region = 'US'` into `WHERE __idx_region = <i32 or i64>(idx)` — pure integer equality, which the standard codegen already handles.

UPPER / LOWER / LENGTH / SUBSTRING all run as pure-host dictionary transformations (`string_ops.rs`, `string_ops_extended.rs`). CONCAT builds a new dictionary via cross-product on the host. None of these go through the GPU because variable-width device writes remain unsupported by the codegen path.

## Filter compaction

Two paths coexist:

1. **GPU-side** (`gpu_compact.rs`): the projection kernel emits a u8 mask alongside its outputs; a per-block Hillis-Steele prefix scan computes per-row exclusive offsets and per-block sums; the host does a small reduction over block sums (or recursively device-scans for very large inputs via `gpu_compact_multipass.rs`); a per-dtype gather kernel writes only the surviving rows into compacted outputs. This is the default.
2. **Host-side** (`compact.rs`): downloads the mask and applies `arrow::compute::filter` per column. Used as the fallback when the output schema contains Utf8 columns the gather kernel can't move.

The dispatch lives in `engine.rs::execute_projection` and is purely dtype-based: any Utf8 output → host-side, else → GPU-side.

## GROUP BY hash table

Single-pass open-addressing on the GPU. The host estimates K (the table size, rounded up to a power of two so probe becomes a mask-and). The classic kernel uses `i64::MIN` as an empty-slot sentinel and `atom.cas.b64` for insertion. For inputs that may legitimately encode to `i64::MIN` (notably Float64 `-0.0`, which is `0x8000_0000_0000_0000`), the fallback kernel in `valid_flag_kernels.rs` uses a parallel `slot_valid: u32[]` table — slot states {0 = empty, 1 = claimed, 2 = committed} — with bounded probe and SPIN loops that spill on overflow. The host folds the spill into the final result.

Float MIN/MAX over `Float32` / `Float64` has no native `atom.global.min/max.f*` on sm_70, so `float_atomics.rs` and `valid_flag_float.rs` synthesise it via a CAS loop on the bit pattern.

## Multi-pass prefix scan

`prefix_scan.rs` tops out at `n_rows ≤ u32::MAX / BLOCK_SIZE ≈ 16.8M` (the host-side scan over block_sums assumes the array fits comfortably). For larger inputs `prefix_scan_multipass.rs` recursively scans the block_sums via the same kernel, building a depth-3-or-4 ladder of prefix-summed arrays. The dispatch lives at the top of `gpu_compact.rs::prefix_scan_mask` — if `n_rows > limit`, delegate.

## What's deliberately *not* in this architecture

- **No process model.** Everything runs in a single Rust process.
- **No multi-GPU per engine.** One context, one device, one default stream per `Engine`. `Engine::new_with_device(idx)` picks the device; multi-GPU means one engine per device. Multiple streams are a future change.
- **No streaming / larger-than-VRAM tables.** Multi-batch tables work (`register_table` accepts more than one `RecordBatch`), but the whole table is uploaded eagerly. A `register_table_stream` API and batched, spill-aware execution are 0.4 work.
- **No async memcpy yet.** The async H2D / D2H FFI bindings landed in 0.3.0; integration into the executors is 0.4.
- **No GPU hash join.** `INNER`, `LEFT`, `RIGHT`, `FULL OUTER`, and `CROSS` joins all work in 0.3.0, but the executors are host-side: build a `HashMap` on the smaller side, probe the larger, materialise via `arrow::compute::take`; `CROSS` is a host-side cartesian product. Non-equi predicates are still rejected. A GPU-resident build+probe path is a 0.4 stretch goal.
- **No GPU sort kernel.** `ORDER BY` (and the dedup step of `DISTINCT` / plain `UNION`) runs host-side via `sort.rs` / `distinct.rs`. A device-resident sort is 0.4 stretch.
- **No CTEs, subqueries, or window functions.** The parser rejects them outright.
- **No `KernelSpec`-keyed codegen cache.** The 0.3.0 PTX cache keys on the *emitted PTX hash*, so PTXAS reassembly is skipped on a hit but codegen itself still runs. A `KernelSpec`-level cache that also skips codegen is 0.4.
- **No optimiser passes beyond the lowering.** No predicate pushdown, no constant folding (beyond what's already in `LogicalPlan::schema` type-checking), no join reordering. The flat pipeline is the optimisation budget today.

Each of these is a deliberate scope choice for 0.3.0, not a fundamental limitation. See [`ROADMAP.md`](../ROADMAP.md) for the milestone mapping.

## Stability of the IR types

`PhysicalPlan`, `KernelSpec`, `AggregateSpec`, `Op`, `Reg`, `Value`, `ColumnIO`, and the rest of the physical-plan / IR vocabulary in `src/plan/physical_plan.rs` are marked `#[doc(hidden)]`. They are implementation-internal: the codegen + executor split owns them end-to-end, no public method takes one as a parameter, and they may change shape, gain variants, or be replaced in any pre-1.0 release without a deprecation cycle. External code that wants to drive Craton Bolt should hold to the `Engine` and `DataFrame` surface; the planner and IR are explicitly out-of-contract until the 1.0 API freeze.
