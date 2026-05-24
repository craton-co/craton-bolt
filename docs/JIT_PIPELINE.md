# JIT Pipeline

This document is the deep dive into Javelin's distinctive technical bet: **compile each SQL query into a fresh NVIDIA PTX kernel at runtime**, rather than chaining precompiled kernels. If you only read one design doc, read this one.

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

Javelin's bet: **fuse all six steps into one kernel** by emitting it from the SQL query at runtime. The whole expression tree lives in registers for the duration of one thread's work on one row. Global memory is touched exactly twice: once to read the inputs, once to write the output (or not, if the predicate gates the store).

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

A PTX module cache keyed on `KernelSpec` would eliminate even that. It hasn't been built yet, but it's an obvious next optimization.

## The stages, in detail

### 1. SQL → `LogicalPlan`

Source: `src/plan/sql_frontend.rs`.

Uses [`sqlparser`](https://github.com/apache/datafusion-sqlparser-rs) as the lexer/parser. We don't accept the full SQL grammar — `parse_sql` walks the parser's AST and only accepts shapes Javelin can execute:

- `SELECT` with optional `WHERE`, `GROUP BY`. No UNION, no CTE, no subqueries, no JOIN, no ORDER BY, no LIMIT, no HAVING.
- A single table in `FROM`. No schema-qualified names.
- Scalar expressions: column references, integer / float / string / bool / null literals, binary arithmetic (`+ - * /`), comparison (`= <> < <= > >=`), logical (`AND OR`), parenthesised sub-expressions, unary minus on literals (folded), unary plus (no-op).
- Aggregate functions in SELECT: `COUNT(*)`, `COUNT(expr)`, `SUM`, `MIN`, `MAX`, `AVG`.
- Implicit GROUP BY validation: every non-aggregate SELECT item must appear in `GROUP BY` if the query has aggregates.

Anything outside this surface produces a clear `JavelinError::Sql(...)` with the unsupported shape.

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

If the literal isn't in the dictionary (no row has that string), the predicate is constant-folded to `Literal(Bool(false))` for `=` or `Literal(Bool(true))` for `<>`. Ordering comparisons (`< > <= >=`) on Utf8 columns return `JavelinError::Plan` — dictionary indices reflect insertion order, not lex order.

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

.visible .entry javelin_kernel(
	.param .u64 javelin_kernel_param_0,         // price input pointer
	.param .u64 javelin_kernel_param_1,         // price output pointer
	.param .u32 javelin_kernel_param_2_n_rows
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
	ld.param.u32  %r4, [javelin_kernel_param_2_n_rows];
	setp.ge.s32   %p0, %r3, %r4;
	@%p0 bra DONE;                              // bounds check
	ld.param.u64  %rd0, [javelin_kernel_param_0];
	cvta.to.global.u64 %rd0, %rd0;
	ld.param.u64  %rd1, [javelin_kernel_param_1];
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

- **Target:** `sm_70` (Volta). Inherits everything from sm_70 forward — sm_70+ GPUs cover everything from V100 onward, which is the realistic deployment floor for serious GPU compute.
- **Address size:** 64-bit pointers. Every parameter is `.param .u64`; addresses are computed via `mul.wide.s32` + `add.s64`.
- **Globalisation:** Every parameter pointer is converted with `cvta.to.global.u64` before use.
- **Bounds check:** Each thread bails to `DONE` if `tid >= n_rows`. The last warp on the last block sees partial work.
- **Predicate gating:** When the kernel has a predicate, a single `setp.ne.s32 %p, <pred_reg>, 0; @!%p bra DONE` skips ALL stores at once — the predicate is computed early, the gate sits right before the first store.

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
let function = module.function("javelin_kernel")?;
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

- **NULL handling.** The current reduction kernels don't read a validity bitmap. `COUNT(expr)` counts every row, not just non-null rows. The host-side `extended_agg` path (Bool, Utf8) does honour nulls.
- **Variable-width string outputs.** CONCAT producing genuinely new strings works via host-side dictionary cross-product (`src/exec/string_ops_extended.rs`), not on the GPU.
- **Joins.** No join algorithm yet.
- **Window functions.** Not yet.
- **CASE / NULLIF / CAST / unary ops beyond folded minus.** The expression evaluator covers the standard binary set; the AST doesn't model these.

Each is a self-contained future change, not a blocker for the existing surface.
