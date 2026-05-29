# Installing Craton Bolt

How to install the prerequisites, build the crate in each of its supported
configurations, and recover from the common build / link failures.

For day-to-day build / test / bench commands once you're set up, see
[`DEVELOPMENT.md`](./DEVELOPMENT.md). For the SQL surface, see
[`SQL_REFERENCE.md`](./SQL_REFERENCE.md).

## Prerequisites

| Tool                                      | Why                                                                          |
|-------------------------------------------|------------------------------------------------------------------------------|
| Rust 1.74+                                | The crate uses the 2021 edition; 1.74 is the MSRV. Nothing newer is required. |
| `cargo`                                   | Standard build driver.                                                       |
| CUDA Toolkit 12.x                         | Provides `cuda.lib` (Windows) / `libcuda.so` (Linux) for the linker.         |
| NVIDIA driver matching the toolkit        | Required only to *run* kernels on a real GPU (tests / benches).              |
| NVIDIA GPU with compute capability ≥ 7.0  | Required only for live-GPU tests and `cargo bench` with `BOLT_BENCH_GPU=1`.  |

You do **not** need a GPU or the CUDA toolkit to build, type-check, or run the
offline test suite — see [Building without CUDA](#building-without-cuda)
below.

### CUDA Toolkit version

Craton Bolt targets the **CUDA 12.x** toolkit series. Specifically:

- The optional `cudarc` backend pins the `cuda-12060` API surface
  (CUDA 12.6), so 12.6 is the reference toolkit.
- The default (hand-rolled FFI) backend links `cuda.lib` / `libcuda.so` and
  works against any CUDA **12.x** install (12.0 through 12.6+).
- `build.rs` deliberately prefers the **highest-versioned 12.x** install when
  several toolkits are present on the host (e.g. it picks `v12.6` over `v12.4`
  over `v11.8`). On Windows it also prefers a `v12.x` install over a `v13.x`
  one — see the [v13.2 linker workaround](#windows-cuda-toolkit-v132-imp_-linker-error)
  below for why.

CUDA Toolkit **v13.x is not supported on Windows** out of the box because of a
stub-library regression (see the troubleshooting section). Linux v13 is
untested.

### Installing the CUDA Toolkit

- **Linux**: follow [NVIDIA's package-manager instructions](https://developer.nvidia.com/cuda-downloads).
  Ensure `/usr/local/cuda/lib64` is on `LD_LIBRARY_PATH`, and that
  `/usr/local/cuda/lib64/stubs` (or equivalent) provides a `libcuda.so` for
  the linker on hosts without a real driver.
- **Windows**: install the toolkit from the
  [official installer](https://developer.nvidia.com/cuda-toolkit). It adds
  `cuda.lib` at `%CUDA_PATH%\lib\x64` and sets `CUDA_PATH`. Open a fresh
  Developer Command Prompt afterward so MSVC's `link.exe` and the new
  environment are both on `PATH`.
- **macOS**: NVIDIA dropped Mac support years ago — you cannot run kernels on
  a Mac. `cargo check` and the `cuda-stub` build still work.

## Building

### Default build (linked CUDA path)

The default feature set is empty (`default = []`). This is the production
build: the hand-rolled `extern "C"` FFI links against the real CUDA driver.

```bash
cargo build --release
```

This requires `cuda.lib` / `libcuda.so` on the linker path (see
[prerequisites](#installing-the-cuda-toolkit)). `build.rs` discovers the
toolkit automatically from `CUDA_PATH` or the platform-default install
locations; set `CUDA_PATH` explicitly to pin a specific install (see
[`ENV_VARS.md`](./ENV_VARS.md)).

### Cargo features

| Feature        | Default | What it does |
|----------------|---------|--------------|
| *(none)*       | yes     | Production build. Links the real CUDA driver via hand-rolled FFI. |
| `cuda-stub`    | no      | Stub mode for GPU-less hosts / CI / `docs.rs`. Skips all CUDA discovery and link injection in `build.rs`; every FFI entry becomes a Rust shim returning `CUDA_ERROR_STUB`. The crate compiles, links, and runs offline tests without any toolkit. |
| `cudarc`       | no      | Stage-1 spike that routes a handful of low-level CUDA driver calls through the pure-Rust [`cudarc`](https://crates.io/crates/cudarc) crate (v0.13, `cuda-12060`) instead of the hand-rolled FFI. Uses cudarc's primary context as the only CUDA context. See `docs/CUDARC_ADOPTION.md`. |
| `rust-cuda`    | no      | Experimental: compiles the sibling `kernels/` crate to PTX at build time via `cuda_builder` (rustc_codegen_nvvm) instead of the hand-rolled string emitter for `partition_kernel.rs`. Requires the rust-cuda toolchain (nightly + libNVVM + LLVM — see `kernels/rust-toolchain.toml`). See `docs/JIT_PIPELINE.md`. |
| `pool-sharded` | no      | Stage-3 escape hatch: swaps the device-mem-pool bucket map for a fixed-size sharded array. Same API, different lock granularity. Turn on only if profiling shows the DashMap shard layer is the bottleneck. |
| `pool-watcher` | no      | Stage-4 proactive eviction: spawns a background thread that polls `cuMemGetInfo_v2` and evicts pooled blocks when free VRAM drops below a threshold. Tunable via `BOLT_POOL_WATCH_*` env vars (see `ENV_VARS.md`). |

Example invocations:

```bash
# GPU-less / CI / docs.rs build (no toolkit needed).
cargo build --no-default-features --features cuda-stub

# Opt into the cudarc backend.
cargo build --features cudarc

# Experimental rust-cuda PTX codegen (needs the rust-cuda toolchain).
cargo build --features rust-cuda
```

### Building without CUDA

The `cuda-stub` feature makes the entire crate — library, tests, and benches —
compile, link, and run on a host with no CUDA toolkit installed. Use it for CI
matrix cells without a GPU, for `docs.rs`, and on developer Macs:

```bash
# Type-check + run all offline tests (host-side helpers, PTX-shape
# snapshots, parser tests, memory-soundness compile-fail doctests).
cargo check --lib --tests --no-default-features --features cuda-stub
cargo test  --lib --tests --no-default-features --features cuda-stub

# cargo doc for docs.rs reproduction.
cargo doc   --no-deps --no-default-features --features cuda-stub
```

At runtime every FFI call in stub mode returns `CUDA_ERROR_STUB`, surfaced as
`BoltError::Other("cuda-stub mode: no GPU support compiled in")`. The
`#[ignore]`-marked tests that genuinely launch kernels need a real GPU; run
them on a CUDA-equipped host *without* `cuda-stub` so the real driver links.

## Troubleshooting

### `cannot open input file 'cuda.lib'` / `cannot find -lcuda`

The CUDA Toolkit isn't installed or isn't on the linker path.

- **Windows**: install the toolkit and reopen the terminal. Verify `where cl`
  returns the MSVC compiler and `%CUDA_PATH%\lib\x64\cuda.lib` exists.
- **Linux**: install the toolkit and verify
  `ld -lcuda --verbose 2>&1 | head` finds `libcuda.so` (check the
  `lib64/stubs/` directory on driverless CI hosts).
- Or build/check with `--no-default-features --features cuda-stub`, which
  skips CUDA discovery entirely (see [Building without CUDA](#building-without-cuda)).

### Windows CUDA Toolkit v13.2 `__imp_*` linker error

If you have CUDA Toolkit **v13.2** installed on Windows you may see unresolved
external symbols of the form `__imp_cu...` at link time. The v13.2 stub
`cuda.lib` lacks the `__imp_*` import symbols MSVC's linker expects.

**Workaround:** point the build at a **v12.6** (or any v12.x) installation:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
cargo build --release
```

`build.rs` already prefers the highest-versioned **v12.x** install over a
v13.x one when both are present, so installing v12.6 alongside v13.2 is
usually enough without setting `CUDA_PATH` by hand. (This is the same
workaround maintainers apply when running `cargo test` on Windows — see
`RELEASING.md` §8.)

### GPU-less testing

You don't need a GPU to run the bulk of the test suite. Build with
`--features cuda-stub` (see above). Only the `#[ignore]`-marked live-GPU
tests require real hardware; they're skipped by default and run with
`cargo test -- --ignored` on a CUDA-equipped host.

### Cold builds are slow

The dev-dependencies (`polars`, bundled `duckdb`) pull in a lot. A cold
`cargo build` / first `cargo bench` can take several minutes; subsequent
builds are cached and fast. This is expected, not a hang.

### `cargo build` picks the wrong CUDA toolkit

On a host with multiple toolkits, `build.rs` selects the highest-versioned
12.x install. To pin a specific one, set `CUDA_PATH` to its root (see
[`ENV_VARS.md`](./ENV_VARS.md)).
