# Installing Craton Bolt

How to install the prerequisites, build the crate in each of its supported
configurations, and recover from the common build / link failures.

For day-to-day build / test / bench commands once you're set up, see
[`DEVELOPMENT.md`](./DEVELOPMENT.md). For the SQL surface, see
[`SQL_REFERENCE.md`](./SQL_REFERENCE.md).

## Prerequisites

| Tool                                      | Why                                                                          |
|-------------------------------------------|------------------------------------------------------------------------------|
| Rust 1.85+                                | 1.85 is the MSRV, verified by a dedicated CI matrix leg. The crate's own source is 2021 edition; the floor comes from the committed `Cargo.lock`, which pins a dependency whose manifest declares `edition = "2024"` — Cargo before 1.85 cannot parse it and fails before compiling anything. |
| `cargo`                                   | Standard build driver.                                                       |
| CUDA Toolkit 12.x                         | Provides `cuda.lib` (Windows) / `libcuda.so` (Linux) for the linker.         |
| NVIDIA driver matching the toolkit        | Required only to *run* kernels on a real GPU (tests / benches).              |
| NVIDIA GPU with compute capability ≥ 7.0  | Required only for live-GPU tests and `cargo bench` with `BOLT_BENCH_GPU=1`.  |
| **MSVC build environment — Windows only** | Supplies the Windows SDK and CRT libraries. `.cargo/config.toml` uses the Rust toolchain's bundled `rust-lld` driver (see [Windows linker: rust-lld](#windows-linker-rust-lld)). |

You do **not** need a GPU or the CUDA toolkit to build, type-check, or run the
offline test suite — see [Building without CUDA](#building-without-cuda)
below.

### CUDA Toolkit version

Craton Bolt targets the **CUDA 12.x** toolkit series. Specifically:

- The default `cudarc` backend pins the `cuda-12060` API surface. It is
  validated with CUDA 12.x and CUDA 13.3 drivers/toolkits.
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

The default feature set is `default = ["cudarc"]`. This is the supported
production build: cudarc owns the primary CUDA context while the remaining
adapter calls link against the driver import library.

```bash
cargo build --release
```

This requires `cuda.lib` / `libcuda.so` on the linker path (see
[prerequisites](#installing-the-cuda-toolkit)). `build.rs` discovers the
toolkit automatically from `CUDA_PATH` or the platform-default install
locations; set `CUDA_PATH` explicitly to pin a specific install (see
[`ENV_VARS.md`](./ENV_VARS.md)).

### Windows linker: rust-lld

On Windows, `.cargo/config.toml` hard-sets the linker for the
`x86_64-pc-windows-msvc` target:

```toml
[target.x86_64-pc-windows-msvc]
linker = "rust-lld"
```

`rust-lld` ships with the Rust toolchain and accepts rustc's current
`-flavor link` driver argument. A separate LLVM installation is unnecessary.
Run from `vcvars64` (or equivalent) so SDK/CRT include and library roots exist.

Why LLD rather than MSVC's `link.exe`: the optional `reference-tests` shard
links bundled DuckDB. MSVC's
`link.exe` spawns `mspdbsrv.exe` for PDB debug info, and `mspdbsrv` enforces a
hard concurrent-session limit that linking that many DuckDB-embedding binaries
can exceed (`LNK1318: Unexpected PDB error; LIMIT (12)`). LLD generates
PDBs in-process with no `mspdbsrv` and no such limit. It only changes the
*link* step; compiled rlibs are unaffected. See the comments in
`.cargo/config.toml` for the full rationale.

**Fallback — no LLVM installed?** If you don't have (and don't want) LLVM, you
can override the linker back to MSVC `link.exe`. Open an **MSVC developer
shell** (run `vcvars64.bat`, or use the "x64 Native Tools Command Prompt") so
`link.exe` is on `PATH`, then set the per-target linker env var, which
overrides the `.cargo/config.toml` setting:

```powershell
# In an MSVC dev shell (vcvars64) so link.exe is on PATH:
$env:CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER = "link.exe"
cargo build --release
```

```cmd
:: cmd.exe equivalent:
set CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=link.exe
cargo build --release
```

This works fine for the library and a single binary; the `mspdbsrv` session
limit only bites when linking the DuckDB reference shard, so prefer the
committed `rust-lld` configuration if you intend to run it.
Do **not** edit `.cargo/config.toml` to make this change — the env-var
override keeps the committed config (which the CI/maintainer flow depends on)
intact.

### Cargo features

| Feature        | Default | What it does |
|----------------|---------|--------------|
| `cudarc`       | yes     | Supported primary-context/driver adapter. See `docs/CUDARC_ADOPTION.md`. |
| `cuda-stub`    | no      | Stub mode for GPU-less hosts / CI / `docs.rs`. Skips all CUDA discovery and link injection in `build.rs`; every FFI entry becomes a Rust shim returning `CUDA_ERROR_STUB`. The crate compiles, links, and runs offline tests without any toolkit. |
| `reference-tests` | no   | Compiles the bundled DuckDB conformance/proptest shard. Kept out of ordinary test loops. |
| `reference-benches` | no | Compiles bundled DuckDB and Polars for comparative benchmarks. |
| `pool-sharded` | no      | Stage-3 escape hatch: swaps the device-mem-pool bucket map for a fixed-size sharded array. Same API, different lock granularity. Turn on only if profiling shows the DashMap shard layer is the bottleneck. |
| `pool-watcher` | no      | Stage-4 proactive eviction: spawns a background thread that polls `cuMemGetInfo_v2` and evicts pooled blocks when free VRAM drops below a threshold. Tunable via `BOLT_POOL_WATCH_*` env vars (see `ENV_VARS.md`). |

Example invocations:

```bash
# GPU-less / CI / docs.rs build (no toolkit needed).
cargo build --no-default-features --features cuda-stub

# Split DuckDB conformance shard (real GPU required to execute ignored tests).
cargo test --features reference-tests -- --ignored --test-threads=1
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
