# Architecture

This document maps Craton Bolt's source tree to the runtime pipeline, and explains the major design decisions. For the SQL → PTX deep-dive read [`JIT_PIPELINE.md`](JIT_PIPELINE.md). For the full supported subset see [`SQL_REFERENCE.md`](SQL_REFERENCE.md).

> **Current as of v0.7.0.** The engine has grown well past the original 0.1 / 0.3.0 baseline. The GPU hash-join family (`INNER` / `LEFT` / `RIGHT` / `FULL OUTER`, plus `CROSS`) and the GPU sort paths (bitonic, integrated; plus an env-gated radix path) are live in the executor, with host-side fallbacks as the correctness reference. v0.6 added a small-cardinality non-equi (nested-loop) join, and v0.7 wired the `KernelSpec`-keyed module cache into the real call sites and landed the GPU radix-sort dispatch. The layer cake and the "Landed" / exclusion sections below reflect that v0.7 state; see [`CHANGELOG.md`](../CHANGELOG.md) for the full per-release delta and [`ROADMAP.md`](../ROADMAP.md) for what's still open.

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
│      join.rs                        join dispatcher; INNER / LEFT / │
│                                     RIGHT / FULL OUTER / CROSS,     │
│                                     host-side hash-equi fallback    │
│      gpu_join.rs                    GPU INNER + LEFT/RIGHT/FULL     │
│                                     OUTER hash-equi join            │
│                                     (KeyShape-aware encoding,       │
│                                     collision-list build/probe,     │
│                                     VRAM-driven hash-table cap)     │
│      sort.rs                        ORDER BY dispatcher; host-side  │
│                                     lexsort fallback                │
│      gpu_sort.rs                    GPU bitonic sort fast path      │
│                                     (single-/multi-key fixed dtype, │
│                                     NULL-aware via validity bitmap) │
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
│    GPU join + sort kernels                                          │
│      hash_join_kernel.rs            build + probe kernels for       │
│                                     INNER and LEFT/RIGHT/FULL       │
│                                     OUTER (KeyShape-aware,          │
│                                     collision-list, AoS slot, CROSS)│
│      sort_kernel.rs                 bitonic sort kernel (multi-key, │
│                                     validity-aware)                 │
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

Single-pass open-addressing on the GPU. The host estimates K (the table size, rounded up to a power of two so probe becomes a mask-and). The classic kernel uses `i64::MIN` as an empty-slot sentinel and `atom.cas.b64` for insertion. For inputs that may legitimately encode to `i64::MIN` (notably Float64 `-0.0`, which is `0x8000_0000_0000_0000`), the fallback kernel in `valid_flag_kernels.rs` uses a parallel `slot_valid: u32[]` table — slot states {0 = empty, 1 = claimed, 2 = committed} — with bounded probe and SPIN loops that spill on overflow. The host folds the spill into the final result. The validity-bitmap variant (`compile_groupby_keys_kernel_with_validity` in `hash_kernels.rs`) threads a per-row valid bit through the same machinery so NULL-tolerant GROUP BY stays on the device end-to-end.

Float MIN/MAX over `Float32` / `Float64` has no native `atom.global.min/max.f*` on sm_70, so `float_atomics.rs` and `valid_flag_float.rs` synthesise it via a CAS loop on the bit pattern.

## Multi-pass prefix scan

`prefix_scan.rs` tops out at `n_rows ≤ u32::MAX / BLOCK_SIZE ≈ 16.8M` (the host-side scan over block_sums assumes the array fits comfortably). For larger inputs `prefix_scan_multipass.rs` recursively scans the block_sums via the same kernel, building a depth-3-or-4 ladder of prefix-summed arrays. The dispatch lives at the top of `gpu_compact.rs::prefix_scan_mask` — if `n_rows > limit`, delegate.

## Landed since 0.3.0

A handful of items called out as "not yet" in early 0.3.0 drafts have since landed and are part of the current (v0.7) architecture:

- **GPU hash join.** `src/exec/gpu_join.rs` (paired with `src/jit/hash_join_kernel.rs`) implements `INNER` plus `LEFT` / `RIGHT` / `FULL OUTER` on the GPU: `KeyShape`-aware key encoding (Int32, Int64, packed two-key, Utf8 via dictionary interning, AoS for wider composites), a per-slot collision-list build/probe pair so non-unique build keys no longer fall back to the host, and a VRAM-driven hash-table cap that the executor negotiates against `cuMemGetInfo` (the byte budget scales from a 64 MiB floor up to 512 MiB on large devices). `CROSS` joins also run on the GPU via a dedicated cartesian kernel. `src/exec/join.rs` is the dispatcher: it picks the GPU path when the predicate, dtypes, and VRAM all agree, otherwise it falls through to the host-side hash join (which remains the correctness reference and the fallback for unsupported shapes and non-equi predicates).
- **GPU sort.** `src/exec/gpu_sort.rs` + `src/jit/sort_kernel.rs` implement a bitonic sort kernel with multi-key support (up to `MAX_SORT_KEYS`), per-key direction, per-key validity bitmaps for `NULL` ordering, and an `is_padded` bitmap that disambiguates real-vs-sentinel ties. The dispatcher `sort.rs::try_gpu_sort` gates the device path on key count, dtype, row count (`GPU_SORT_MIN_ROWS = 16_384`), and the `n_rows <= 2^31` bitonic-padding bound; misses fall through to `arrow::compute::lexsort_to_indices` + `take`. Utf8 keys flow through an inline dictionary builder so the kernel only sees fixed-width indices.
- **Async memcpy in the executors.** `src/cuda/cuda_sys.rs` exposes `memcpy_h2d_async` / `memcpy_d2h_async` / `memset_d8_async` over both the raw driver FFI and the optional cudarc backend (`src/cuda/cudarc_backend.rs`). The projection executor (`engine.rs::execute_projection`) downloads each output column through `StagedDownload`, which pins a host buffer, kicks off `cuMemcpyAsync` on the engine stream, and synchronizes once at the end of the projection — overlapping the D2H of column *i* with the launch wind-down of column *i+1*.
- **Sentinel-free GROUP BY with validity bitmap.** `compile_groupby_keys_kernel_with_validity` in `src/jit/hash_kernels.rs` carries a per-row validity bit through the hash-insert path so columns that legitimately encode to the sentinel value (e.g. Float64 `-0.0`) no longer have to spill to a host-side reduction; the `slot_valid` table plus bounded probe is described under "GROUP BY hash table" below.
- **Tier-2 hash-partitioned GROUP BY.** Already shipped: the `groupby_tier2_*.rs` executors plus the `partition_kernel*` / `scatter_kernel*` / `partition_reduce_kernel*` PTX families implement the partition → per-partition reduce → host merge pipeline, with i32 and i64 keys and count / sum / avg / minmax (int and float) / multi-agg variants.
- **Dictionary registry + string-literal rewrite.** `src/exec/dict_registry.rs` and `src/plan/string_literal_rewrite.rs` are first-class components, not stubs: every Utf8 column gets a dictionary at `register_table` time, and the rewrite turns `col = 'X'` into `__idx_col = i32/i64(idx)` before lowering so the standard codegen handles string equality as integer equality.

## What's deliberately *not* in this architecture

- **No process model.** Everything runs in a single Rust process.
- **No multi-GPU per engine.** One context, one device, one default stream per `Engine`. `Engine::new_with_device(idx)` picks the device; multi-GPU means one engine per device. Multiple streams are a future change.
- **No streaming / larger-than-VRAM tables.** Multi-batch tables work (`register_table` accepts more than one `RecordBatch`), but the whole table is uploaded eagerly. `register_table_stream` ships (v0.6) with an eager implementation behind a future-compatible signature; truly-lazy, batched, spill-aware execution is still open.
- **Async memcpy coverage is partial.** `execute_projection` uses pinned async D2H via `StagedDownload`, and v0.7 rolled async memcpy out to the scalar-aggregate and the GROUP BY variants (tier2 / shmem / wide / valid) plus the `WHERE`-filter D2H; the join executor still issues most of its H2D uploads synchronously. Broadening the async surface across every remaining executor is still open.
- **Non-equi join predicates are limited.** `INNER` / `LEFT` / `RIGHT` / `FULL OUTER` (on GPU when the shape qualifies, host-side otherwise) and `CROSS` are all supported as equi joins. Non-equi predicates like `a.x < b.y` run only through the v0.6 host-side nested-loop fallback (`execute_nested_loop_join`), which is **INNER-only and caps the smaller side at `MAX_NESTED_LOOP_INNER_ROWS = 1024`**; OUTER non-equi and large non-equi joins still reject with a clear message. A GPU predicate-aware probe is still open.
- **GPU sort has fast-path gates.** Multi-key bitonic sort works on the device (integrated by default); v0.7 added an env-gated (`BOLT_GPU_SORT=1`) radix path for `Int32` / `Int64` keys. Tiny inputs (`< 16_384` rows), dtypes outside the supported set, and `> 2^31` rows still drop to the host-side `lexsort_to_indices`. `DISTINCT` and plain `UNION` dedup still go through the host-side sort.
- **No CTEs, subqueries, or window functions.** The parser rejects them outright.
- **PTX cache plus a `KernelSpec`-keyed module cache.** The PTX cache keys on the *emitted PTX hash* (skips PTXAS reassembly on a hit). The `KernelSpec`-keyed module cache (built in v0.6, wired into the real call sites in v0.7 — scalar reduction, the `gpu_join` build/probe sites, the `gpu_sort` radix sites, and the compaction kernels) skips both codegen and PTXAS on a hit.
- **No optimiser passes beyond the lowering.** No predicate pushdown, no constant folding (beyond what's already in `LogicalPlan::schema` type-checking), no join reordering. The flat pipeline is the optimisation budget today.

Each of these is a deliberate scope choice, not a fundamental limitation. See [`ROADMAP.md`](../ROADMAP.md) for the milestone mapping.

## Stability of the IR types

`PhysicalPlan`, `KernelSpec`, `AggregateSpec`, `Op`, `Reg`, `Value`, `ColumnIO`, and the rest of the physical-plan / IR vocabulary in `src/plan/physical_plan.rs` are marked `#[doc(hidden)]`. They are implementation-internal: the codegen + executor split owns them end-to-end, no public method takes one as a parameter, and they may change shape, gain variants, or be replaced in any pre-1.0 release without a deprecation cycle. External code that wants to drive Craton Bolt should hold to the `Engine` and `DataFrame` surface; the planner and IR are explicitly out-of-contract until the 1.0 API freeze.
