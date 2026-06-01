# cudarc Adoption

This document describes the optional `cudarc` backend: what it is, what it
currently does, and how it relates to the default hand-rolled FFI path. It is
referenced from [`INSTALL.md`](./INSTALL.md), [`BENCHMARKS.md`](./BENCHMARKS.md),
and the Cargo feature table.

For the broader CUDA layer this sits beneath, see
[`ARCHITECTURE.md`](ARCHITECTURE.md); for the JIT path that loads kernels, see
[`JIT_PIPELINE.md`](JIT_PIPELINE.md).

## The two backends

Craton Bolt talks to the CUDA Driver API. By default it does so through a
**hand-rolled `extern "C"` FFI** layer (`src/cuda/cuda_sys.rs`): a thin set of
`cuMemAlloc_v2` / `cuMemcpy*` / `cuModuleLoadDataEx` / `cuLaunchKernel` bindings
the crate declares itself and links against `cuda.lib` / `libcuda.so`. This is
the production path and the default feature set is empty (`default = []`).

The optional **`cudarc` backend** routes a *subset* of those low-level driver
calls through the pure-Rust [`cudarc`](https://crates.io/crates/cudarc) crate
(pinned to v0.13, `cuda-12060` — i.e. the CUDA 12.6 API surface) instead of the
hand-rolled FFI. It is gated behind a Cargo feature, off by default:

```bash
# Opt into the cudarc backend.
cargo build --features cudarc
```

The CUDA-Oxide safety layer above the driver (`GpuVec`, `GpuView`,
`GpuViewMut`, `GpuBuffer`) is **unchanged** by the feature — cudarc replaces
only what sits *below* those types, swapping which code issues the raw driver
calls. Call sites and the public API surface stay the same.

## What it does today (Stage 1 spike)

The current state is a deliberately narrow **Stage-1 spike** whose goal is to
prove that a feature-flagged cudarc backend *builds and runs* without yet
committing to a full migration. It is **not** the default and is not on the
critical path for any shipped configuration.

Concretely, when `--features cudarc` is enabled (`src/cuda/cudarc_backend.rs`):

- A handful of low-level memory primitives in `cuda_sys.rs` delegate into the
  cudarc backend instead of the hand-rolled FFI: device allocation
  (`mem_alloc`), free (`mem_free`), and the host↔device copies (`memcpy_h2d`,
  `memcpy_d2h`), plus the async memcpy / memset counterparts
  (`memcpy_h2d_async`, `memcpy_d2h_async`, `memset_d8_async`).
- Everything else — context creation, kernel launch, and module load — still
  runs through the existing hand-rolled path during Stage 1.

Because cudarc 0.13 *owns* its allocations via `CudaSlice<T>` (which frees
itself on drop), and the existing call sites expect to free explicitly via
`mem_free(ptr)`, the backend uses cudarc's raw alloc/free escape hatch
(`result::malloc_sync` / `result::free_sync`) so it returns and takes a raw
`CUdeviceptr` exactly like the FFI does. The deeper copy/memset calls likewise
drop down to cudarc's dynamically-loaded `driver::sys::lib()` symbols, which are
the same driver entry points the hand-rolled block exposes — the driver sees an
identical bit pattern, with only a `CUstream` type-alias cast at the boundary.

## The single-context invariant

The most important thing the cudarc backend buys is a **single CUDA context**.

When `cudarc` is enabled, the engine uses cudarc's **primary context**
(`CudaDevice::new`) as the *only* CUDA context; the hand-rolled
`cuCtxCreate_v2` / `cuCtxDestroy_v2` path is bypassed. This fixes a historical
two-context lifetime bug: the hand-rolled path could mint a parallel context, so
device pointers allocated in one context were freed against the other at
teardown. Routing every alloc / free / memcpy through one cudarc-owned primary
context makes the pool pointers and their frees belong to the same context.

Because the launch and module-load paths still operate on the *thread's current*
context, the backend binds cudarc's primary context to the calling thread on
each entry (`ensure_device` / `bind_current_thread` / the per-call
`bind_to_thread` in `device_ref`). This is what lets launch-only worker threads
— and the pool-drain path in `CudaContext::drop`, which can run on a thread that
never touched the alloc path — issue driver calls against the right context.
Multi-GPU is a later concern: today the cell latches one device (default
ordinal 0) for the process.

## Relationship to the default FFI path

- **Default (no feature):** the hand-rolled `extern "C"` FFI is the production
  backend. Nothing about cudarc is compiled in; the `cudarc` dependency is
  `optional` and absent from the default build.
- **`--features cudarc`:** the four memory primitives (plus async siblings)
  delegate to cudarc and the engine runs on cudarc's primary context;
  everything else is still the hand-rolled path. This is the spike — useful for
  validating the approach, not the shipping default.

The two backends are intended to produce bit-identical results: the cudarc
calls bottom out at the same CUDA driver entry points, and driver errors are
translated into the same typed `BoltError::CudaWithCode { code, message }`
shape (`cudarc_err`) the FFI path produces, so downstream code that
pattern-matches a raw `CUresult` (e.g. OOM detection in `mem_pool`) keeps
working unchanged across either backend.

A future Stage 2 would switch to `CudaSlice<T>` ownership (deleting the
raw-pointer alloc/free helpers) and widen the surface cudarc covers; until then
the hand-rolled FFI remains the default and the authoritative reference for the
crate's CUDA behaviour.
