# Development

How to build, test, benchmark, and extend Javelin.

## Prerequisites

| Tool                     | Why                                                                     |
|--------------------------|-------------------------------------------------------------------------|
| Rust 1.74+               | The crate uses 2021 edition; nothing newer is required.                 |
| `cargo`                  | Standard.                                                               |
| CUDA Toolkit 12.x        | Provides `cuda.lib` (Windows) / `libcuda.so` (Linux) for the linker.    |
| NVIDIA driver matching the toolkit | Required only for running tests / benchmarks on a real GPU.   |
| NVIDIA GPU with compute capability ≥ 7.0 | Required only for `cargo test -- --ignored` and `cargo bench` with `JAVELIN_BENCH_GPU=1`. |

If you don't have CUDA installed yet:

- **Linux**: see [NVIDIA's package manager instructions](https://developer.nvidia.com/cuda-downloads). Make sure `/usr/local/cuda/lib64` is in `LD_LIBRARY_PATH` and `/usr/local/cuda/lib64/stubs` (or the equivalent) provides `libcuda.so` for the linker.
- **Windows**: install the CUDA Toolkit from the [official installer](https://developer.nvidia.com/cuda-toolkit). The installer adds `cuda.lib` to the linker path (`%CUDA_PATH%\lib\x64`) automatically when you open a fresh Developer Command Prompt.
- **macOS**: NVIDIA dropped Mac support years ago; you cannot run Javelin on a Mac with an actual GPU. `cargo check` still works.

## What works without CUDA

A surprisingly large fraction of development can happen on a CUDA-less host:

```bash
cargo check --lib --tests --benches      # full type-check
cargo build --lib                         # compiles the library
cargo doc --no-deps --open                # generates rustdoc
cargo clippy --all-targets                # lints
```

These all pass on a host that has no `cuda.lib` because the crate doesn't *invoke* CUDA at compile time — the FFI declarations are just `#[link(name = "cuda")]` symbols that the linker resolves later. The linker errors only kick in when you try to build a binary that actually exercises them: `cargo test` and `cargo bench`.

The pure-Rust unit tests (PTX-shape assertions, host-side helpers, expression evaluator, dictionary transformations) DO need to link to the CUDA library because they share the binary with code that calls into FFI. They run fine on a CUDA-equipped host without a GPU; the `#[ignore]`-marked tests are the ones that actually launch kernels.

## Build commands

```bash
# Full clean build (~7 min cold from scratch because polars pulls in a lot).
cargo build --release

# Library only.
cargo build --lib

# Quick check.
cargo check --lib --tests --benches

# Format.
cargo fmt

# Lint.
cargo clippy --all-targets
```

## Test commands

```bash
# All non-ignored tests. Requires cuda.lib on linker path.
cargo test

# Just the library's inline tests.
cargo test --lib

# Just one integration test file.
cargo test --test e2e_tests
cargo test --test memory_tests

# Live-GPU tests. Requires an actual NVIDIA GPU.
cargo test -- --ignored

# Run a single test by name (substring match).
cargo test ptx_for_trivial_select_contains
```

### What the three test flavours mean

1. **Pure-host unit tests** (`#[test]`, no `#[ignore]`). Always run. Examples: `pack_keys_two_int32`, `sql_substring_unicode_round_down_at_start`, `unify_numeric` behaviour, dictionary dedup.
2. **PTX-shape tests** (`#[test]`, no `#[ignore]`). Always run. Emit a PTX string and assert that it contains specific instructions / labels / parameter declarations. They don't need a GPU but catch JIT regressions.
3. **Live-GPU tests** (`#[test] #[ignore]`). Skipped by default. Need both `cuda.lib` AND an actual GPU. Marked with `#[ignore = "requires CUDA device — run with cargo test -- --ignored"]`.

## Benchmark commands

```bash
# CPU-only benchmarks (planner, codegen, CPU reference, Polars).
cargo bench

# Add the GPU engine path. Requires an actual NVIDIA GPU.
JAVELIN_BENCH_GPU=1 cargo bench           # bash
$env:JAVELIN_BENCH_GPU="1"; cargo bench   # PowerShell

# A single bench group.
cargo bench --bench query_benchmarks -- plan
cargo bench --bench query_benchmarks -- polars
cargo bench --bench query_benchmarks -- engine_execute   # GPU only
```

Criterion writes HTML reports to `target/criterion/`. The bench file is `benches/query_benchmarks.rs`.

## Project workflow

### Adding a new SQL feature

1. **Decide where it lowers.** New unary op? Add a variant to `Expr` in `src/plan/logical_plan.rs`. New aggregate? Add to `AggregateExpr`. New plan node? Add to `LogicalPlan`.
2. **Teach the SQL frontend.** Walk to the appropriate match in `src/plan/sql_frontend.rs` and add the case. Add a `parse_*` test in `tests/e2e_tests.rs`.
3. **Teach the lowering.** `src/plan/physical_plan.rs::lower` may need a new op or a new physical-plan shape.
4. **Teach the codegen.** `src/jit/ptx_gen.rs` (or a sibling) for new ops. Add a PTX-shape test in the same file.
5. **Teach the executor.** `src/exec/engine.rs::execute` may need a new dispatch branch, or one of the per-shape executors may need updating.
6. **Add an `#[ignore]`-gated live-GPU test** that runs the new shape end-to-end.

### Adding a new aggregate path

The pattern: write a self-contained executor in `src/exec/<your_executor>.rs`, expose a public `pub fn execute_<shape>(plan, batch) -> JavelinResult<RecordBatch>`, then wire it into `Engine::execute`'s match in `src/exec/engine.rs`.

Look at `src/exec/groupby_with_pre.rs` for a recent example. The pattern is:

1. Validate the plan shape.
2. Materialise inputs as host or device columns.
3. JIT-compile + launch any kernels.
4. Download + post-process.
5. Pack into a `RecordBatch` matching the plan's `output_schema`.

### Adding a new PTX kernel

Place it in `src/jit/<your_kernel>.rs`. Expose a `pub fn compile_<name>_kernel(...) -> JavelinResult<String>` and a `pub const <NAME>_ENTRY: &str = "..."` constant for the symbol-lookup name.

Conventions:

- Target `sm_70`, version `7.5`, 64-bit addressing.
- One thread per row (1D launch) by default.
- Bounds check at the top with `setp.ge.s32 %p, %tid, %n_rows; @%p bra DONE`.
- Globalise every pointer parameter with `cvta.to.global.u64` before use.
- Generous `.reg` declarations (PTX `.reg` only allocates names, not physical registers).
- All errors flow through `JavelinResult`. No `unwrap` in the codegen.

Add a PTX-shape test in the same file:

```rust
#[test]
fn kernel_contains_expected_instructions() {
    let ptx = compile_my_kernel(...).expect("emit");
    assert!(ptx.contains(".version 7.5"));
    assert!(ptx.contains("ld.global.s32"));
    assert!(ptx.contains("DONE:"));
}
```

### Adding a new dtype

This is a big change. You'd need:

- A variant in `src/plan/logical_plan.rs::DataType`.
- `byte_width()` and `unify_numeric` updates.
- Arrow type mapping in `src/exec/engine.rs::plan_dtype_to_arrow` and the inverse.
- A variant in `src/exec/engine.rs::DeviceCol` plus upload / alloc_zeros / download.
- PTX type-suffix tables in `src/jit/ptx_gen.rs` for `ld.global.<ty>` and `st.global.<ty>`.
- Per-dtype kernels in `src/jit/agg_kernels.rs` and `src/jit/prefix_scan.rs`.

Open an issue first.

## Recovering from common errors

### `error: linking with link.exe failed: cannot open input file 'cuda.lib'`

The CUDA Toolkit isn't installed or isn't on the linker path.

- Windows: install the toolkit and reopen the terminal. Verify `where cl` returns the MSVC compiler and `%CUDA_PATH%\lib\x64\cuda.lib` exists.
- Linux: install the toolkit and verify `ld -lcuda --verbose 2>&1 | head` finds `libcuda.so`.

Or just use `cargo check --lib` instead, which doesn't invoke the linker.

### `error[E0277]: ... doesn't implement Debug`

A test is calling `.expect_err(...)` on a result whose `T` doesn't implement `Debug`. The fix is to match on the `Result` instead:

```rust
// Before (won't compile):
let err = some_call().expect_err("must error");

// After (compiles):
match some_call() {
    Ok(_) => panic!("must error"),
    Err(e) => assert!(matches!(e, JavelinError::Other(_))),
}
```

This bites tests that touch `GpuVec`, `DeviceCol`, `GatheredCol`, `DictionaryColumn`, or any wrapper that holds a non-Debug `GpuVec` inside.

### `error: cannot find macro 'println' in this scope` in `criterion` output

Polars 0.42 pulls in a lot. Cold builds take 3–6 minutes. Be patient on the first `cargo bench`; subsequent builds are fast.

### Tests pass but `cargo bench` hangs

Criterion benchmarks need a quiet machine. Close Chrome, Slack, Spotify. The `cpu_reference` and `polars` benches in particular are sensitive to background CPU work.

## Project conventions

- **One-line `///` doc comments** on public items. Longer prose goes in `//!` module-level docs.
- **`// SAFETY:` comments** on every `unsafe` block explaining the invariant.
- **`JavelinResult<T>`** everywhere a fallible operation could happen in library code. Never `unwrap()` in `src/`.
- **`#[cfg(test)] mod tests`** at the bottom of each file for unit tests.
- **No `panic!()`** in library code. Tests can panic; benches can panic; library code returns errors.
- **`debug_assert!`** for invariants the type system can't express but you want to catch in debug builds.

## Where to ask for help

Open an issue with the [`question`] label. Include the query, the API call, and the exact error.

## Licensing

Javelin is Apache-2.0-licensed. See [`../LICENSE`](../LICENSE) for the
canonical text and [`../NOTICE`](../NOTICE) for third-party attribution.

Two practical implications for day-to-day work:

1. **Every new `.rs` file needs an SPDX header.** Put it on line 1:

   ```rust
   // SPDX-License-Identifier: Apache-2.0
   ```

   Then a blank line, then your module's existing first line (`//!` docs,
   `use` statements, whatever). The CI lint script (forthcoming) will
   reject files without this header.

2. **Vendoring third-party code requires a NOTICE update.** If you copy
   in code from another Apache-2.0 project, add an entry to
   [`../NOTICE`](../NOTICE) crediting the upstream. If the third-party
   project is under a different license, check Apache-2.0 compatibility
   first (most permissive licenses — MIT, BSD-2 / -3, ISC, Zlib — are
   fine; copyleft licenses like GPL are not).

3. **Dev-dependencies don't need NOTICE entries** unless they're shipped
   in the published artifact. Criterion and Polars (both dev-only) are
   already mentioned in NOTICE but aren't redistributed by the crate
   itself.
