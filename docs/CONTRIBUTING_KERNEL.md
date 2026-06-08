<!-- SPDX-License-Identifier: Apache-2.0 -->

# Kernel Contributor Guide

This document covers everything needed to add a new PTX kernel to Craton Bolt:
where emitters live, the naming and structural conventions they follow, the
`// SAFETY:` invariants required at every launch site, and the two-tier test
pattern (PTX-shape + live-GPU) used to validate new GPU code.

Read [`docs/ARCHITECTURE.md`](ARCHITECTURE.md) first for the layer-cake context.
For the SQL → PTX deep-dive see [`docs/JIT_PIPELINE.md`](JIT_PIPELINE.md).

---

## When to add a new kernel

A new PTX kernel is warranted when:

- A SQL construct cannot be implemented by composing existing
  projection / scan / aggregate / hash-join / sort kernels without a
  host-side round-trip that would make it materially slower.
- An existing kernel is correct but the work shape (threads-per-group,
  memory access pattern, data-type representation) is wrong for the new
  use case — copy rather than extend.

Before writing a kernel, open an issue to discuss the shape. A small change to
an existing emitter is often preferable to a new file.

---

## Where kernels live

| Directory | What goes there |
|---|---|
| `src/jit/` | PTX emitters — functions that return `BoltResult<String>`. One logical kernel family per file. |
| `src/exec/` | Executors that compile kernels and launch them. Call into `src/jit/` via the compile entry point. |
| `src/cuda/` | CUDA driver FFI, buffer types, stream pool. Do not emit PTX from here. |

The naming convention is: `src/jit/<verb>_kernel.rs` for the emitter,
`src/exec/<verb>.rs` for the executor. Examples: `src/jit/sort_kernel.rs` /
`src/exec/gpu_sort.rs`, `src/jit/hash_join_kernel.rs` /
`src/exec/gpu_join.rs`.

---

## PTX emitter conventions

### Module header

Every emitted PTX module begins with the same three-line header, using the
crate-level constants from `src/jit/ptx_gen.rs`:

```rust
// src/jit/ptx_gen.rs
pub(crate) const PTX_VERSION: &str = ".version 7.5";
pub(crate) const PTX_TARGET:  &str = ".target sm_70";
const PTX_ADDRESS_SIZE:        &str = ".address_size 64";
```

Emit them verbatim at the top of every module string. Do not hard-code the
values inline — the disk-cache salt (`jit::disk_cache::codegen_salt`) folds
`PTX_VERSION` and `PTX_TARGET` into the cache key so that a GPU-architecture
change automatically invalidates all cached entries.

### Register classes

The `RegAlloc` allocator in `ptx_gen.rs` defines the mapping from `DataType`
to PTX register class. Honour it in new emitters:

| PTX class | Register prefix | Used for |
|---|---|---|
| `r`  | `%r0`, `%r1`, … | `Bool`, `Int32`, `Date32` |
| `rl` | `%rl0`, `%rl1`, … | `Int64`, `Timestamp`, one half of `Decimal128` |
| `f`  | `%f0`, `%f1`, … | `Float32` |
| `fd` | `%fd0`, `%fd1`, … | `Float64` |
| `p`  | `%p0`, `%p1`, … | Predicate registers (`setp.*` results) |
| `rd` | `%rd0`, `%rd1`, … | Device-pointer arithmetic |

Never mix classes. An `rl` value in an `r`-typed instruction is a PTX
assembler error that surfaces only at `cuModuleLoadDataEx` time.

**128-bit values (Decimal128).** There is no native 128-bit register class in
PTX. Represent each i128 as two adjacent `rl` registers: `(lo, hi)`. Use
`RegAlloc::assign_pair` so the indices stay contiguous in the `.reg` block.
Every 128-bit load, store, and arithmetic op must handle the `lo` / `hi` split
explicitly.

### The `PtxBuilder` + `emit_fmt!` pattern

Build emitters around the `PtxBuilder` struct and the `emit_fmt!` macro, both
defined in `src/jit/ptx_gen.rs`. Use `emit_fmt!` instead of `b.emit(&format!(…))` —
the macro writes directly into the body buffer without an intermediate `String`
allocation:

```rust
// BAD — allocates a throwaway String per instruction:
b.emit(&format!("ld.global.nc.u32 {}, [{}];", dst, ptr))?;

// GOOD — writes straight into b.body:
emit_fmt!(b, "ld.global.nc.u32 {}, [{}];", dst, ptr)?;
```

For labels (column zero, no leading tab), use `b.emit_label("loop_top")`.

### Emitter function signature

Every kernel emitter must:

1. Return `BoltResult<String>` (the complete PTX module text, not just the body).
2. Have a matching `pub const` for the entry-point name so callers and tests
   both import it from one place:

```rust
// src/jit/my_kernel.rs

/// PTX entry-point name for the my-op kernel.
pub const MY_OP_KERNEL: &str = "bolt_my_op";

/// Emit a PTX module implementing the my-op operation.
pub fn compile_my_op_kernel(/* spec fields */) -> BoltResult<String> {
    let mut b = PtxBuilder::new(MY_OP_KERNEL);
    // ... emit instructions ...
    Ok(build_module(MY_OP_KERNEL, b))
}
```

Entry-point names are `snake_case`, prefixed with `bolt_` to avoid symbol
collisions if the consumer links multiple modules.

### Column-pointer parameters

Column input and output pointers must carry the `.ptr .global .restrict
.align 16` qualifier in their `.param` declaration. This is what allows PTXAS
to apply alias-based optimisations across loads and stores:

```ptx
.param .u64 .ptr .global .restrict .align 16 bolt_my_op_param_0
```

Emit these through the builder's `param_name` helper so the suffix index stays
consistent with the host-side `KernelArgs` push order.

### Parameter ordering

The kernel ABI expected by `KernelArgs` and `launch_1d` is fixed:

1. Input column pointers (one `.u64 .ptr .global .restrict .align 16` per input), in input order.
2. Output column pointers, in output order.
3. Input validity-bitmap pointers (one per nullable input), if the kernel is validity-aware.
4. Output validity-bitmap pointers, if any.
5. `n_rows` as a trailing `.u32`.

Kernels that take additional trailing scalars (e.g. `n_groups`, `n_partitions`)
are launched via `launch_with_geometry` and push those scalars via
`KernelArgs::push_scalar_u32` after `n_rows`. The PTX param list must reflect
the same order.

### Row-count cap

The TID setup uses **signed 32-bit** arithmetic (`mad.lo.s32 %tid, %ctaid,
%ntid, %tid_x`), so the per-launch addressable row space is `i32::MAX`
(≈ 2.1 billion rows). All offset arithmetic widens `tid` unsigned
(`mul.wide.u32`) to address the full `u32` range of rows, but the TID itself is
signed-bounded. The executor **must** enforce this before launching:

```rust
let n_rows = n_rows_to_u32(batch.num_rows())?;  // errors if > u32::MAX
// additionally guard for the TID signed-math cap:
if n_rows as i64 > i32::MAX as i64 {
    return Err(BoltError::Plan("row count exceeds 2.1B per-kernel cap".into()));
}
```

Larger datasets require migrating the TID computation to 64-bit grid
addressing, which is tracked as a pre-1.0 follow-up.

### Literal emission

Numeric literals must be emitted as **hex bit-patterns**, never as decimal
strings that could be corrupted by locale formatting. Use the `emit_const`
pattern:

```rust
// Integer (i32 → hex):
emit_fmt!(b, "mov.s32 {}, 0x{:08X};", dst, value as u32)?;

// Float64 (bit-cast to u64 → hex):
emit_fmt!(b, "mov.f64 {}, 0d{:016X};", dst, f64::to_bits(value))?;
```

**String literals must never be interpolated into PTX source.** If a kernel
must act on a string value, encode it as a dictionary index and pass the integer
index as an `Int32` / `Int64` column. See `src/plan/string_literal_rewrite.rs`
for the host-side rewrite that makes this work for SQL string predicates.

---

## `// SAFETY:` requirements at launch sites

Every `unsafe` block in the executor layer must carry a `// SAFETY:` comment
explaining the invariants that make it sound. The kernel launch pattern
generates two specific `unsafe` blocks:

### 1. `cuLaunchKernel`

```rust
// SAFETY: kernel_params is a slice of *mut c_void that point into
// args.ptrs (CUdeviceptr values) and args.n_rows (u32). These live
// on the stack for the duration of the call. The lifetime parameter
// on KernelArgs<'a> keeps the underlying GpuVec buffers alive until
// at least after the synchronize() below, so no device pointer in
// kernel_params can dangle during or after the kernel launch.
unsafe {
    cuda_sys::check(cuda_sys::cuLaunchKernel(
        function.raw(),
        grid_x, 1, 1,
        block_x, 1, 1,
        shared_bytes,
        stream.raw(),
        kernel_params.as_mut_ptr(),
        ptr::null_mut(),
    ))?;
}
```

Required elements of the `// SAFETY:` annotation:

- What `kernel_params` points to and that it is stack-resident for the call.
- That `KernelArgs<'a>` lifetime guarantees the `GpuVec` buffers outlive the launch.
- That `synchronize()` follows, making results host-visible before any borrow ends.

### 2. After every launch: `tag_launch_stream`

Immediately after `cuLaunchKernel` and before any other work, call
`args.tag_launch_stream(stream.raw())`. This records the launch stream into
the `StreamSet` of every buffer whose view was pushed into `args`, ensuring
`GpuBuffer::Drop` fences the launch stream before returning any block to the
pool. It is the single enforcement point for the V-1 stream-safety invariant:

```rust
// V-1: record the launch stream into every buffer this launch touched.
// GpuBuffer::Drop will fence it before recycling the block, making the
// safety hold even if the synchronize() below is later removed.
args.tag_launch_stream(stream.raw());

stream.synchronize()?;
```

**Never remove `tag_launch_stream`.** The `synchronize()` is a performance
convenience (makes results host-visible, turns the later Drop-fence into a
no-op) — but it is `tag_launch_stream` that makes the use-after-free impossible
by construction.

### 3. `CudaStream::null_or_default`

Use `CudaStream::null_or_default()` in executors rather than constructing a new
stream with `CudaStream::new()`. The null-or-default path reuses a stream from
the process-wide pool (`crate::cuda::stream_pool`), which keeps stream handles
alive until context teardown. Constructing per-query streams with `::new()` and
dropping them is UB: a `GpuBuffer` that recorded the stream in its `used_streams`
will `cuEventRecord` a destroyed handle at `Drop` time, faulting the host.

### 4. `unsafe impl Send` for stream / function wrappers

If you wrap a CUDA handle type in a new struct and need to implement `Send`,
use:

```rust
// SAFETY: a CUstream (or CUfunction / CUmodule) may be used from any
// thread once the owning context is current on that thread. The engine
// ensures the context is active before transferring stream ownership
// across a thread boundary.
unsafe impl Send for MyStreamWrapper {}
```

Do not implement `Sync` — a single stream is not safe to use concurrently from
multiple threads.

---

## Testing a new kernel

There are two mandatory test tiers and one optional third.

### Tier 1: PTX-shape tests (no GPU required)

A PTX-shape test calls the emitter directly, receives the PTX string, and
asserts on substrings. These run in CI without a GPU and catch regressions in
the emitted instruction set, predicate structure, and parameter conventions.
They live in `tests/ptx_golden_tests.rs` or in a `#[cfg(test)]` module inside
the emitter file.

```rust
#[test]
fn my_op_kernel_emits_global_load() {
    let ptx = compile_my_op_kernel(/* args */).expect("compile failed");

    // Target header present.
    assert!(ptx.contains(".target sm_70"), "wrong SM target");

    // Column pointer has restrict qualifier.
    assert!(ptx.contains(".ptr .global .restrict .align 16"), "missing restrict");

    // Core instruction present.
    assert!(ptx.contains("add.s64"), "expected 64-bit add for Int64 accumulator");
}
```

Shape tests are the primary correctness mechanism for the JIT layer. Write
them for:

- Each PTX target directive (`.target sm_70`, `.version 7.5`).
- The column-pointer `.restrict` qualifier.
- Every non-obvious instruction choice (e.g. `atom.cas` for float MIN/MAX,
  `setp.lt.f32` for a float predicate, `cvt.s64.s32` for SUM widening).
- The predicate gate that bounds each thread to its row (`setp.ge.s32 %p, %tid, %n_rows; @%p ret`).
- Any `SECURITY:` invariant in the emitter (hex immediate for literals, no
  string interpolation).

### Tier 2: Live-GPU integration tests (`#[ignore]`)

End-to-end tests that actually launch the kernel on hardware go in
`tests/e2e_tests.rs` or a dedicated file under `tests/`, gated with
`#[test] #[ignore]`. They are skipped by default and run with:

```bash
cargo test --features cuda -- --ignored
```

```rust
#[test]
#[ignore = "requires CUDA GPU"]
fn my_op_kernel_produces_correct_output() {
    let mut engine = Engine::new().expect("no GPU");
    engine.register_table("t", make_test_batch()).unwrap();
    let result = engine.sql("SELECT my_op(col) FROM t").unwrap();
    assert_eq!(result.num_rows(), EXPECTED_ROWS);
    // ... column-value assertions ...
}
```

A live test is required for any kernel that modifies device memory. PTX-shape
tests alone are not sufficient to catch ABI mismatches (wrong parameter order,
off-by-one in the row bound, incorrect reduction tree).

### Tier 3: Property / fuzz tests (optional)

For kernels with complex numerical properties (e.g. Decimal128 arithmetic,
multi-key sort correctness), a `proptest`-based test in `tests/proptest_*.rs`
can automatically find corner cases. These are always-on (no `#[ignore]`) but
marked `proptest` in the test name so they can be targeted with
`-- proptest_` if needed. Reference existing examples in
`tests/proptest_groupby.rs`.

---

## Checklist: adding a new kernel

- [ ] Emitter file in `src/jit/<verb>_kernel.rs` with:
  - [ ] `pub const <VERB>_KERNEL: &str = "bolt_<verb>";`
  - [ ] `pub fn compile_<verb>_kernel(...) -> BoltResult<String>`
  - [ ] Module header via `PTX_VERSION` / `PTX_TARGET` constants (not inline strings)
  - [ ] Column pointers declared with `.ptr .global .restrict .align 16`
  - [ ] Numeric literals emitted as hex immediates; no string interpolation
  - [ ] `// SECURITY:` comment on every `emit_const` site explaining why the literal is injection-safe
- [ ] Executor in `src/exec/<verb>.rs` (or extended existing executor):
  - [ ] `// SAFETY:` on every `unsafe { cuLaunchKernel(...) }` block
  - [ ] `args.tag_launch_stream(stream.raw())` immediately after every `cuLaunchKernel` call
  - [ ] `stream.synchronize()?` before returning results
  - [ ] Row-count guard enforcing `n_rows <= i32::MAX`
  - [ ] Stream acquired via `CudaStream::null_or_default()`, not `::new()`
- [ ] Dispatch wired into `src/exec/engine.rs::execute` (or appropriate sub-dispatch)
- [ ] PTX-shape tests covering target/version header, `.restrict`, and at least one non-trivial instruction
- [ ] Live-GPU `#[ignore]` test covering at least one round-trip (correct output rows / values)
- [ ] `pub mod <verb>_kernel;` added to `src/jit/mod.rs` (or `src/exec/mod.rs`)
- [ ] `CHANGELOG.md` entry under `[Unreleased]`
