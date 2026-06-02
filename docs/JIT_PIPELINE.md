# JIT Pipeline

This document is the deep dive into Craton Bolt's distinctive technical bet: **compile each SQL query into a fresh NVIDIA PTX kernel at runtime**, rather than chaining precompiled kernels. If you only read one design doc, read this one.

For the layer map, see [`ARCHITECTURE.md`](ARCHITECTURE.md). For the user-facing SQL surface, see [`SQL_REFERENCE.md`](SQL_REFERENCE.md).

## The thesis

GPU dataframe engines like RAPIDS / cuDF expose a library of precompiled kernels — one for `add`, one for `multiply`, one for `less_than`, and so on. A SQL query like

```sql
SELECT price * tax FROM sales WHERE region_id = 1
```

becomes (roughly):

```
load(region_id)  →  cmp_eq(1)  →  load(price)  →  load(tax)  →  multiply  →  mask_filter
```

Six kernel launches, six round trips to global memory, six chunks of intermediate output buffer. The GPU's L2 cache is fast but global memory is not; reading and writing the same row five times before the user sees a result is wasteful.

Craton Bolt's bet: **fuse all six steps into one kernel** by emitting it from the SQL query at runtime. The whole expression tree lives in registers for the duration of one thread's work on one row. Global memory is touched exactly twice: once to read the inputs, once to write the output (or not, if the predicate gates the store).

This is exactly what Polars and DataFusion do on the CPU — codegen a vectorised pipeline per query. It hasn't been done in OSS for the GPU because the cost of "compile a kernel at query time" sounds expensive. It turns out it isn't, if you skip LLVM and emit PTX directly.

## What runtime PTX actually costs

| Stage                                        | Time on a modern CPU (eyeball)   |
|----------------------------------------------|----------------------------------|
| `sqlparser::parse_sql`                       | ~20 μs                           |
| `LogicalPlan::schema` (type-check)           | < 5 μs                           |
| `physical_plan::lower`                       | ~10 μs                           |
| `ptx_gen::compile` (string-builder)          | ~50 μs                           |
| `cuModuleLoadData` (driver PTX → SASS)       | ~10–50 ms (cold), ~1–5 ms (warm) |
| `cuLaunchKernel`                             | ~10 μs                           |

The dominant cost is the driver's PTX-to-SASS assembly inside `cuModuleLoadData`. That's milliseconds, not seconds — orders of magnitude faster than invoking LLVM (which is what `rustc` or `clang` would do). For a query that processes millions of rows, the JIT cost amortises away.

A process-wide PTX module cache lives in `src/jit/jit_compiler.rs`. `CudaModule::from_ptx` hashes the emitted PTX text and, on a hit, returns a clone of the cached `Arc<CudaModuleInner>` — skipping PTXAS re-assembly and the `cuModuleLoadDataEx` driver call. The key is a **128-bit** hash (`hash_ptx` — two domain-separated `DefaultHasher` outputs packed into a `(u64, u64)` pair, ≈ 2^-64 collision bound vs the original 64-bit key's ≈ 2^-32). Default capacity is 256 entries with **LRU** eviction backed by an intrusive O(1) doubly-linked list: a hit moves the node to the head (`touch()`), a miss inserts at the head and evicts the tail when at cap; override at process start by setting the env var `CRATON_BOLT_PTX_CACHE_CAP` to a positive integer (unset / empty / zero / unparseable values fall back to 256, read once and memoized). Entries hold the PTX `String` alongside the module and the cache **re-validates the full PTX text on every hit**, so even a 128-bit hash collision is correctness-safe — it falls through to a fresh uncached load instead of returning the wrong kernel. Concurrent misses on the same PTX serialise through a per-entry `OnceCell`, so N threads racing on the same query pay exactly one PTXAS compile rather than N — unrelated keys still compile fully in parallel. As of v0.7 the `KernelSpec`-keyed cache that skips the codegen step too exists: a per-`Engine` `module_cache` plus a process-wide cache in `src/exec/module_cache.rs` key on the planner `KernelSpec` (content hash / `format!("{:?}", spec)`), so a warm hit is a sub-µs `Arc<CudaModuleInner>` clone that skips `ptx_gen::compile`, the disk cache, and the driver load entirely. `Engine::get_or_build_module` (`src/exec/engine.rs`) checks the per-engine cache first, then the global one, before falling through to codegen on a true miss.

### Disk-backed PTX cache

The caches above are all per-process: a fresh process (a CLI invocation, a benchmark-harness restart, a serverless cold start) re-runs the whole codegen pipeline and pays the driver `cuModuleLoadData` PTXAS cost from scratch, even though `PhysicalPlan → PTX` is byte-for-byte deterministic. `src/jit/disk_cache.rs` adds an **opt-in persistent layer**: a directory of `<key>.ptx` files, one per cached spec, that lets a cold process skip codegen (it still pays the one-time PTXAS load, since the module isn't resident in the new process). On a cold-process miss the caller looks up the disk cache first; on a hit it gets the PTX text without re-running codegen and hands it to `CudaModule::from_ptx`. On a disk miss it runs codegen and writes the result back through.

The cache is **disabled by default** (preserving the zero-side-effect contract of `Engine::sql`). It is enabled either by setting the env var **`BOLT_PTX_CACHE_DIR`** to a non-empty path, or via `EngineBuilder::persistent_cache(path)` (which overrides the env var). Key properties:

- **Codegen-salt key.** The on-disk key is the `KernelSpec` content hash *prefixed by a codegen salt* (`disk_key` / `codegen_salt`): `CODEGEN_VERSION` (a manual maintainer-bumped constant), the crate version (`CARGO_PKG_VERSION`), and an optional `build.rs` codegen fingerprint (`BOLT_CODEGEN_FINGERPRINT`, consumed via `option_env!` if a build script ever provides one). The `KernelSpec` hash captures *what* the kernel computes but not *how* the PTX was emitted; folding the salt in means any change to PTX emission rotates the filename, so a stale entry written by an older binary simply misses and the new binary re-runs codegen. (This is why the in-process text-hash cache needs no salt — it re-checks the full PTX text on every hit, which the disk layer can't do without reading the file.)
- **Integrity-protected.** Each file carries a `#bolt-ptx-cache v1 <digest>` header line; `lookup` recomputes the digest over the body and serves it **only** if they match. A corrupt, partially-written, tampered, or headerless (old-format) entry is treated as a miss, so the caller recompiles rather than launching untrusted PTX. The digest is a tripwire against accidental corruption and naive tampering, not a cryptographic MAC; the real integrity boundary is the cache directory's permissions (owner-only `0o700` on Unix, best-effort `icacls` ACL tightening on Windows, set in `DiskPtxCache::open`).
- **Path-traversal hardened.** The key becomes a filename, so it is validated (`valid_key`) at that boundary against a strict `^[0-9A-Za-z._-]+$` charset with no path separators, no `..`, no `:`, and no NUL. An unsafe key turns into a lookup miss / store no-op rather than a read or write outside the cache root.
- **Atomic writes.** A store writes to a temp file in the same directory (suffixed with PID + a process-monotonic counter) and `rename`s it into place — atomic on a single filesystem on every supported platform — so concurrent readers never observe a partial file. Two writers racing on the same key produce identical deterministic bytes, so last-writer-wins is harmless.

The net effect is that the disk cache amortises the cold-process codegen cost (and, on a hit, the only remaining cold tax is the unavoidable one-time `cuModuleLoadData` PTXAS assembly in the fresh process).

## The stages, in detail

### 1. SQL → `LogicalPlan`

Source: `src/plan/sql_frontend.rs`.

Uses [`sqlparser`](https://github.com/apache/datafusion-sqlparser-rs) as the lexer/parser. We don't accept the full SQL grammar — `parse_sql` walks the parser's AST and only accepts shapes Craton Bolt can execute:

- `SELECT` with optional `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, `LIMIT [OFFSET]`, `DISTINCT`, `UNION [ALL]` / `EXCEPT` / `INTERSECT`, and one or more JOINs per `SELECT` — `INNER`, `LEFT [OUTER]`, `RIGHT [OUTER]`, `FULL [OUTER]` (all with equi `ON` predicates), or `CROSS` (no `ON`). Equi-joins route through the GPU hash-join kernel (`src/jit/hash_join_kernel.rs` + `src/exec/gpu_join.rs`) when the key shape is supported, falling back to the host-side hash join otherwise; CROSS remains host-side. Non-recursive CTEs (`WITH`), uncorrelated subqueries, and window functions are **accepted** in v0.7 (correlated subqueries and `WITH RECURSIVE` are still rejected; small-cardinality non-equi INNER joins run through a host nested-loop). See [`SQL_REFERENCE.md`](SQL_REFERENCE.md) for the full surface; the rest of this section focuses on the parts that drive PTX codegen.
- One or more base tables in `FROM` (a base table plus the joined tables from the supported JOIN forms). No schema-qualified names.
- Scalar expressions: column references, integer / float / string / bool / null literals, binary arithmetic (`+ - * /`), comparison (`= <> < <= > >=`), logical (`AND OR`), parenthesised sub-expressions, unary minus on literals (folded), unary plus (no-op).
- Aggregate functions in SELECT: `COUNT(*)`, `COUNT(expr)`, `SUM`, `MIN`, `MAX`, `AVG`.
- Implicit GROUP BY validation: every non-aggregate SELECT item must appear in `GROUP BY` if the query has aggregates.

Anything outside this surface produces a clear `BoltError::Sql(...)` with the unsupported shape.

The resulting `LogicalPlan` is a small enum (`Scan`, `Filter`, `Project`, `Aggregate`) wrapping `Expr` trees. Type-checking lives in `LogicalPlan::schema` and `Expr::dtype(&schema)`.

### 2. String-literal predicate rewriting

Source: `src/plan/string_literal_rewrite.rs`. Called from `Engine::sql` right after parse, before lowering.

The codegen path doesn't speak Utf8 — variable-width strings would defeat coalesced loads. To avoid handing the codegen a string predicate it can't lower, the rewriter folds:

```text
   WHERE region = 'US'
       │
       │   region is Utf8; dictionary loaded at register_table time
       │   contains "US" at index 5
       ▼
   WHERE __idx_region = Int32(5)
```

If the literal isn't in the dictionary (no row has that string), the predicate is constant-folded to `Literal(Bool(false))` for `=` or `Literal(Bool(true))` for `<>`. Ordering comparisons (`< > <= >=`) on Utf8 columns return `BoltError::Plan` — dictionary indices reflect insertion order, not lex order.

The rewriter accepts both i32-indexed and i64-indexed dictionaries via a `LiteralIndex { I32(i32), I64(i64) }` enum; the output literal carries the matching dtype. The `__idx_<col>` column is appended to the scan's logical schema by `DictRegistry::extended_schema`, so the SQL frontend can resolve the rewriter's emitted column references at parse time.

### 3. `LogicalPlan` → `PhysicalPlan`

Source: `src/plan/physical_plan.rs`.

The lowering pass resolves names to ordinals, flattens nested expressions into a sequence of single-result `Op`s, and decides what fits in one kernel vs what needs a `pre` projection feeding an aggregate.

The IR (`Op`) is small and SSA-shaped:

```rust
pub enum Op {
    LoadColumn { dst: Reg, col_idx: usize, dtype: DataType },
    Const      { dst: Reg, lit: Literal },
    Cast       { dst: Reg, src: Reg, from: DataType, to: DataType },
    Binary     { dst: Reg, op: BinaryOp, lhs: Reg, rhs: Reg,
                 dtype: DataType, result_dtype: DataType },
    Store      { src: Reg, col_idx: usize, dtype: DataType },
}
```

Each `Op` produces at most one fresh `Reg`. A `KernelSpec` carries an ordered `Vec<Op>` plus the input/output column lists, an optional predicate register, and a register count. That's the input the codegen consumes.

For aggregates, the lowering may emit a `pre: Some(KernelSpec)` that's a regular projection (possibly with a predicate) followed by an `AggregateSpec` describing the reduction over the pre's outputs.

### 4. PTX codegen

Source: `src/jit/ptx_gen.rs` (projection kernels), with siblings in `scan_kernel.rs`, `agg_kernels.rs`, `hash_kernels.rs`, etc.

Every emitter is a Rust function that builds a `String`. The output is real PTX — no placeholders, no fake assembly, no TODO comments. A trivial projection kernel for `SELECT price FROM sales` looks like:

```ptx
.version 7.5
.target sm_70
.address_size 64

.visible .entry bolt_kernel(
	.param .u64 bolt_kernel_param_0,         // price input pointer
	.param .u64 bolt_kernel_param_1,         // price output pointer
	.param .u32 bolt_kernel_param_2_n_rows
)
{
	.reg .pred %p<1>;
	.reg .b32  %r<6>;
	.reg .b64  %rd<6>;
	.reg .f64  %fd<2>;
	mov.u32       %r0, %ctaid.x;
	mov.u32       %r1, %ntid.x;
	mov.u32       %r2, %tid.x;
	mad.lo.s32    %r3, %r0, %r1, %r2;          // global thread idx
	ld.param.u32  %r4, [bolt_kernel_param_2_n_rows];
	setp.ge.s32   %p0, %r3, %r4;
	@%p0 bra DONE;                              // bounds check
	ld.param.u64  %rd0, [bolt_kernel_param_0];
	cvta.to.global.u64 %rd0, %rd0;
	ld.param.u64  %rd1, [bolt_kernel_param_1];
	cvta.to.global.u64 %rd1, %rd1;
	mul.wide.s32  %rd2, %r3, 8;
	add.s64       %rd3, %rd0, %rd2;
	ld.global.f64 %fd0, [%rd3];                 // read price[tid]
	mul.wide.s32  %rd4, %r3, 8;
	add.s64       %rd5, %rd1, %rd4;
	st.global.f64 [%rd5], %fd0;                 // write output[tid]
DONE:
	ret;
}
```

A few conventions used throughout:

- **Target:** `sm_70` (Volta). Every kernel the JIT emits — projection, predicate, prefix-scan, gather, per-block reduction, GROUP BY insert and aggregate, sentinel-free variants, and float MIN/MAX via CAS — declares `.target sm_70` by default. sm_70+ GPUs cover everything from V100 onward, which is the realistic deployment floor for serious GPU compute.

  | Instruction                        | Min CC required | Where emitted                                                  |
  |------------------------------------|-----------------|----------------------------------------------------------------|
  | `atom.global.add.f64`              | sm_60           | scalar + group-by SUM over `Float64` (`agg_kernels.rs`, `hash_kernels.rs`) |
  | `atom.global.add.f32`              | sm_50           | scalar + group-by SUM over `Float32`                            |
  | `atom.global.add.s64` / `.min.s64` / `.max.s64` | sm_60   | group-by SUM / MIN / MAX over 64-bit signed integers            |
  | `atom.global.cas.b64`              | sm_30           | classic GROUP BY key insertion                                  |
  | `atom.global.cas.b32`              | sm_30           | sentinel-free `slot_valid` insertion (`valid_flag_kernels.rs`) and float MIN/MAX CAS-loop on `Float32` |
  | `cvta.shared.u64` / `cvta.to.global.u64` | sm_50     | every pointer parameter (`cvta.to.global.u64`) and shared-memory addressing in the reductions |
  | `shfl.sync.*`                      | sm_70           | warp-level reductions in the per-block aggregator               |

  None of these exceed sm_70, so every kernel is loadable on V100+. If you change the target floor, audit `float_atomics.rs` first — the CAS-loop pattern there is the workaround for the absence of native `atom.global.{min,max}.f*`, which is still unavailable through sm_90.
- **Address size:** 64-bit pointers. Every parameter is `.param .u64`; addresses are computed via `mul.wide.s32` + `add.s64`.
- **Globalisation:** Every parameter pointer is converted with `cvta.to.global.u64` before use.
- **Bounds check:** Each thread bails to `DONE` if `tid >= n_rows`. The last warp on the last block sees partial work.
- **Predicate gating:** When the kernel has a predicate, the emitter inserts a single `setp.eq.s32 %p, <pred_reg>, 0; @%p bra DONE` (positive form — branch when the predicate register is zero, i.e. when the row was masked out) right before the first store, so ALL stores are skipped in one branch. The predicate itself is computed early in the kernel body; the gate sits at the boundary between the load/compute prefix and the store suffix.

Per-dtype register classes:
- `Bool`, `Int32` → `%r<N>` (b32).
- `Int64` → `%rl<N>` (b64).
- `Float32` → `%f<N>` (f32).
- `Float64` → `%fd<N>` (f64).
- `Bool` mid-flight bit-pattern manipulation → `%vr<N>` (b32) on the float-atomic CAS paths.
- Pointers → `%rd<N>` (b64).
- Predicates → `%p<N>` (pred).

A `RegAlloc` table tracks which logical `Reg` maps to which physical register. The `.reg` declarations at the top of the kernel are sized at codegen-end based on the final counters.

### 5. PTX → cubin via the CUDA driver

Source: `src/jit/jit_compiler.rs`.

Common confusion: NVRTC compiles **CUDA C++** to PTX. To go PTX → cubin you use the driver's `cuModuleLoadData`, which takes a null-terminated PTX string and assembles it to SASS internally. That's what we use — no `libnvrtc` dependency, no separate `ptxas` invocation.

```rust
let module = CudaModule::from_ptx(ptx_string)?;
let function = module.function("bolt_kernel")?;
```

`CudaModule` owns the cubin; `Drop` calls `cuModuleUnload`. `CudaFunction<'a>` is a borrowed handle with a `PhantomData<&'a CudaModule>` so it can't outlive the module.

### 6. Launch

Source: `src/exec/launch.rs` for the typed `KernelArgs` API; `src/exec/engine.rs::execute_projection` for the direct `cuLaunchKernel` call (used because heterogenous columns don't fit the monomorphic `KernelArgs::push_input<T>` shape).

Launch geometry: 1D, block size 256, grid size `ceil(n_rows / 256)`. The CUDA driver wants kernel args as a `*mut *mut c_void` — each entry pointing at the storage for one argument (a `CUdeviceptr` or the `n_rows` u32). We assemble that array on the stack inside the launch function; it lives for the duration of the launch + synchronize.

```rust
let stream = CudaStream::null();
let grid_x = ((n_rows_u32 + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
unsafe {
    cuda_sys::check(cuda_sys::cuLaunchKernel(
        function.raw(),
        grid_x, 1, 1,
        256,    1, 1,
        0,                          // shared mem bytes
        stream.raw(),               // NULL stream by default
        kernel_params.as_mut_ptr(),
        ptr::null_mut(),
    ))?;
}
stream.synchronize()?;
```

After `synchronize` returns, every thread has completed. The output buffers can be downloaded or fed to the next kernel.

### 7. Filter compaction

When the kernel has a predicate, the output buffers have zeros in the masked-out slots. Compaction produces a `RecordBatch` with only the surviving rows.

Two paths:

- **GPU-side (default for non-Utf8 outputs):** a separate predicate-only kernel (`src/jit/scan_kernel.rs`) re-evaluates the predicate and writes a `u8` mask. A per-block Hillis-Steele prefix scan (`src/jit/prefix_scan.rs`) computes per-row exclusive offsets and per-block sums. The host scans the block sums (or recursively device-scans for `n_rows > 16.8M` via `src/jit/prefix_scan_multipass.rs`) and re-uploads as `block_bases`. A per-dtype gather kernel writes `output[base + local_idx] = input[gid]` only for surviving rows. Result is a tight, compacted column.
- **Host-side fallback (used when any output is Utf8):** download the mask, download the outputs as full Arrow arrays, run `arrow::compute::filter` per column.

The dispatch in `engine.rs::execute_projection` picks based on output dtype.

### 8. Aggregate reductions

Source: `src/jit/agg_kernels.rs` (per-block primitive reductions), `src/jit/hash_kernels.rs` (GROUP BY hash), `src/jit/valid_flag_kernels.rs` (sentinel-free variant), `src/jit/float_atomics.rs` / `src/jit/valid_flag_float.rs` (float MIN/MAX via CAS).

Scalar reduction is a single-kernel grid reduction: each block reduces its rows in shared memory via a Hillis-Steele tree, then thread 0 writes the block partial to `output[blockIdx.x]`. The host downloads the partials and does the final cross-block reduction in plain Rust (one O(n_blocks) loop, microseconds).

GROUP BY uses an open-addressing hash table with linear probing. The keys kernel inserts via `atom.global.cas.b64` on the keys table itself (classic, sentinel-based) or via `atom.global.cas.b32` on a parallel `slot_valid: u32[]` table (sentinel-free, dispatch fallback when float keys may collide with `i64::MIN`). The agg kernels probe the populated keys table and run a per-slot atomic update — `atom.global.add.f64` for SUM on floats (sm_70 native), `atom.global.{min,max}.s64` for integer MIN/MAX, or a CAS loop on the bit pattern for float MIN/MAX (no native instruction on sm_70).

For load factors below 0.5 (which the executor enforces via `K = next_pow2(2 * unique + 16)`), the expected probe length is well under `log2(K)`. The valid-flag variant adds bounded probe + spin counters that spill overflowing rows to a host-allocated buffer for safety against pathological warp scheduling.

### 9. Download

Source: `src/exec/engine.rs::execute_projection` and the per-shape executors.

Each output `GpuVec` is round-tripped to a host `Vec<T>` via `cuMemcpyDtoH`, wrapped in the matching Arrow primitive array (`Int32Array`, `Float64Array`, etc.) or decoded back through a `DictionaryColumn` for Utf8 outputs. The arrays plus the schema build a `RecordBatch`, which the engine wraps in a `QueryHandle` and returns to the caller.

> **Async transfer status (as of 0.7.0).** Async memcpy has landed and is
> partially wired into the executors. The safe wrappers `memcpy_h2d_async` /
> `memcpy_d2h_async` / `memset_d8_async` sit alongside a typed
> `PinnedHostBuffer<T>` and additive `GpuBuffer::copy_from_async` /
> `copy_to_async` entry points; pinned-host alloc/free itself remains
> hand-rolled `cuMemAllocHost_v2` / `cuMemFreeHost` FFI (cudarc 0.13 does not
> expose those cleanly). The async path is now live on several executors:
> `execute_projection` runs a pinned async D2H via `StagedDownload`, the
> scalar-aggregate executor uploads via `upload_primitive_values_async`
> (piloted in 0.6, see `src/exec/aggregate.rs`), and 0.7 rolled async memcpy
> out to the GROUP BY variants (tier2 / shmem / wide / valid) plus the
> `WHERE`-filter D2H in `compact::download_mask`. Coverage is still partial:
> the join executor issues most of its H2D uploads through the synchronous
> `from_slice` helpers, and broadening the async surface across every
> remaining executor — so the full Ingest → kernel → Download pipeline
> overlaps on an explicit stream rather than serializing on the NULL stream
> — is still open.

## Fusing aggregates with projections

The hardest case the planner has to handle: `SELECT SUM(price * tax) FROM sales WHERE region_id = 1`.

The aggregate input `price * tax` isn't a bare column — it's an expression. The lowering produces:

```
PhysicalPlan::Aggregate {
    pre: Some(KernelSpec {
        inputs:    [price, tax, region_id],
        outputs:   [__expr_0 (= price * tax)],
        ops:       [LoadColumn(price), LoadColumn(tax), Binary(Mul),
                    LoadColumn(region_id), Const(1), Binary(Eq), Store(__expr_0)],
        predicate: Some(<the Eq result>),
    }),
    aggregate: AggregateSpec {
        inputs:     [__expr_0],
        group_by:   [],
        aggregates: [Sum(Column("__expr_0"))],
        output_schema: ...,
    },
}
```

`execute_aggregate_with_pre` runs the pre kernel to materialise `__expr_0` (with the predicate gating stores), downloads + host-compacts via the mask, then re-uploads the compacted column and reduces it via the standard per-block kernel.

For non-bare aggregate inputs that aren't covered by the pre kernel's outputs (rare, but possible if a query is constructed by hand), `src/exec/expr_agg.rs` is a host-side expression evaluator that operates over `HostColumn` enums and produces a materialised column the reduction can consume.

## What's not codegened

- **NULL handling.** Validity-aware kernels are available — see `compile_*_with_validity` in `src/jit/hash_kernels.rs` and the sentinel-free variants in `src/jit/valid_flag_kernels.rs`. The dispatch in `src/exec/groupby.rs` auto-routes to the validity variant on null-bearing inputs (`AggregateSpec::input_has_validity`). Older reduction paths that don't yet read the bitmap fall back to host-side handling via the `extended_agg` path (Bool, Utf8), which honours nulls.
- **Variable-width string outputs.** CONCAT producing genuinely new strings works via host-side dictionary cross-product (`src/exec/string_ops_extended.rs`), not on the GPU.
- **Joins.** `INNER`, `LEFT [OUTER]`, `RIGHT [OUTER]`, `FULL [OUTER]`, and `CROSS` joins all work, and a `SELECT` may carry more than one of them. The GPU hash-join kernel landed in 0.3.x — see `src/jit/hash_join_kernel.rs` for the build / probe / collision / unmatched emitters, wired through `src/exec/gpu_join.rs`. The host-side hash-join path (build smaller side into a HashMap, probe the larger; CROSS is a host-side cartesian product) remains as the fallback for shapes the GPU path doesn't yet cover. Non-equi predicates are not rejected outright — small-cardinality non-equi INNER joins run through a host-side nested-loop fallback (`execute_nested_loop_join`); OUTER non-equi and large non-equi joins still error.
- **Window functions.** Parsed and executed as of 0.7 (host-side evaluation); not yet codegened to PTX.
- **CASE / CAST / COALESCE / NULLIF.** These **are** modeled in the AST (`Expr::Case`, `Expr::Cast`, `Expr::CastFormat`; COALESCE / NULLIF desugar to CASE) and lower to the GPU when the result dtype is numeric or `Bool` (a fold of `selp.*` for CASE, `cvt.*` for CAST). As of the 0.7 feature waves, **`Decimal128` CASE also lowers to the GPU** via the 128-bit predicated select (`Op::Select128`), and **CAST integer↔Decimal128 and Float↔Decimal128 lower to the GPU** (`Op::Div128` / `WidenToI128` / `NarrowI128ToInt` / `F64ToI128` / `I128ToF64`). A **`Utf8`-result CASE** is supported but host-realized (lowers to a `StringProject` `CaseUtf8` output), as is **`CAST … FORMAT`** (temporal↔string) and **`TRY_CAST` / `SAFE_CAST`** (NULL-on-failure host path). `Date32` / `Timestamp` CASE results still run host-side. See [`SQL_REFERENCE.md`](SQL_REFERENCE.md) §"CASE / CAST / COALESCE / NULLIF" for the per-dtype tier.

Each is a self-contained future change, not a blocker for the existing surface.

## rust-cuda alternative emitter (experimental, opt-in)

In addition to the default hand-emit PTX path (every kernel in
`src/jit/*_kernel.rs`), the crate ships a second emitter that compiles
Rust source code to PTX via the `rustc_codegen_nvvm` backend. This is
gated behind `--features rust-cuda` and is currently *narrow in scope*:
only the partition kernel (`src/jit/partition_kernel.rs` →
`kernels/src/lib.rs::bolt_partition`) has a rust-cuda implementation.

When the feature is enabled, `build.rs` invokes `cuda_builder` to
compile `kernels/` to a PTX module (`OUT_DIR/partition.ptx`), and
`compile_partition_kernel()` returns the embedded PTX bytes rather than
the hand-emit string. Constants (`NUM_PARTITIONS`, `HASH_MULTIPLIER`)
are duplicated between the two paths but verified equal via inline
inspection (see `kernels/src/lib.rs:37` ≡ `src/jit/partition_kernel.rs:79`).

Why a parallel path? The hand-emit PTX text emitter is fast,
type-aware, and produces deterministic output — but it sits below the
Rust type system. The rust-cuda path proves the same kernel can be
written as readable Rust, with the cost of a much heavier toolchain
(nightly + libNVVM + LLVM). The crate's default remains the hand-emit
path; rust-cuda is a spike toward "everything in Rust" that may or may
not become the default in a future release. The feature flag, the
`kernels/` crate, and `build.rs`'s `cuda_builder` invocation are the
authoritative reference for the current scope.

Practical impact for downstream users: zero unless you opt in via
`--features rust-cuda` AND have the rust-cuda toolchain installed (see
`kernels/rust-toolchain.toml`).
