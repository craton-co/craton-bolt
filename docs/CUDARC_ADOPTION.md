# cudarc backend (`--features cudarc`)

> **Status: optional, experimental backend + staged migration plan.** This
> document describes the `cudarc` Cargo feature and the plan for migrating
> Craton Bolt's hand-rolled CUDA Driver API FFI onto the pure-Rust
> [`cudarc`](https://github.com/coreylowman/cudarc) crate. **Only Stage 1 and
> Stage 1.5 have landed** as of v0.7.0 — they cover the low-level allocation,
> free, and sync/async memcpy/memset surface. Stages 2–4 (PTX module loading,
> context, launch path, memory pool) are *not* done, and the cudarc path has
> not been validated against a live GPU. The default build does **not** enable
> this feature. For supported runtime behaviour see
> [`ARCHITECTURE.md`](ARCHITECTURE.md) and [`../README.md`](../README.md).

## What the feature does today

`cudarc` is declared in `Cargo.toml` as an **optional** dependency, gated
behind the `cudarc` Cargo feature:

```toml
[dependencies]
cudarc = { version = "0.13", optional = true, default-features = false, features = ["driver", "cuda-12060"] }

[features]
cudarc = ["dep:cudarc"]
```

Build with `cargo build --features cudarc` to opt in. When the feature is
enabled, a handful of low-level device-memory primitives are routed through
cudarc instead of the hand-rolled `extern "C"` FFI in `src/cuda/cuda_sys.rs`.
Everything else — context creation, kernel launch, PTX module loading — still
uses the existing path. The goal of the spike was to prove a feature-flagged
cudarc backend builds, links, and serves real allocations and transfers
without committing to a full migration.

The implementation lives in `src/cuda/cudarc_backend.rs`, registered behind
`#[cfg(feature = "cudarc")]` in `src/cuda/mod.rs`.

### Surface area covered (Stage 1 + 1.5)

`cudarc_backend.rs` routes the following through cudarc:

- **Sync**: `mem_alloc(bytes)`, `mem_free(ptr)`, `memcpy_h2d`, `memcpy_d2h`.
- **Async**: `memcpy_h2d_async`, `memcpy_d2h_async`, `memset_d8_async`.

The sync alloc/free use cudarc's raw `result::malloc_sync` / `free_sync`
escape hatch, which returns/takes a raw `CUdeviceptr` bit-compatible with the
hand-rolled FFI (both ultimately call `cuMemAlloc_v2` / `cuMemFree_v2`). The
memcpy/memset paths drop one level lower and call cudarc's
dynamically-loaded `cudarc::driver::sys::lib()` bindings
(`cuMemcpyHtoD_v2`, `cuMemcpyDtoH_v2`, `cuMemcpyHtoDAsync_v2`,
`cuMemcpyDtoHAsync_v2`, `cuMemsetD8Async`) directly rather than synthesising a
host slice from an unchecked caller pointer (a deliberate soundness choice —
see the in-code "review finding H4" comments).

### Context model

When `--features cudarc` is enabled, the engine uses cudarc's **primary
context** (`CudaDevice::new`) as the only CUDA context. `CudaContext::new` in
`src/cuda/cuda_sys.rs` calls `cudarc_backend::ensure_device`, and
`CudaContext::set_current` calls `bind_current_thread`. Each backend call
re-binds the primary context to the calling thread (`dev.bind_to_thread()`)
because alloc/free/memcpy can originate from arbitrary worker threads. This
fixes the historical two-context lifetime bug where the hand-rolled
`cuCtxCreate_v2` minted a parallel context that the memory pool's pointers did
not belong to. The cached `Arc<CudaDevice>` lives in a process-wide
`OnceCell` (`GLOBAL_DEVICE`).

### Pool integration

`GpuBuffer<T>::with_capacity` allocates through the process-wide
`DeviceMemPool` in `src/cuda/mem_pool.rs`. Under `--features cudarc` the pool's
allocate-miss and free paths dispatch to `cudarc_backend::mem_alloc` /
`mem_free` (see the `#[cfg(feature = "cudarc")]` blocks around
`driver_mem_alloc` and `driver_free` in `mem_pool.rs`); otherwise they call the
hand-rolled `cuda_sys` FFI. Because both backends mint bit-compatible
`CUdeviceptr`s, the pool's bucketing and drain logic is backend-agnostic and
`GpuBuffer::drop` needed no cudarc-specific changes.

### Error mapping

cudarc's `DriverError(pub sys::CUresult)` is translated to the typed
`BoltError::CudaWithCode { code, message }` via the `cudarc_err` helper, so
downstream code (notably `mem_pool::is_oom_error`) can pattern-match the raw
`CUresult` integer exactly as it does for the hand-rolled FFI.

## What is deliberately NOT done in Stage 1 / 1.5

- **Pinned-host alloc/free stay hand-rolled even under `--features cudarc`.**
  cudarc 0.13's `driver` feature does not expose `cuMemAllocHost_v2` cleanly,
  so the `PinnedHostBuffer` path in `src/cuda/buffer.rs` continues to call
  `cuda_sys::mem_alloc_host` / `cuMemFreeHost` directly under both feature
  flags. Tracked as Stage 3 future work — revisit when cudarc 0.14+ surfaces a
  usable pinned-host API.
- **`CudaContext` and `CudaModule` still use the hand-rolled FFI.** cudarc
  exposes equivalents (`CudaDevice`, `CudaDevice::load_ptx`) but swapping them
  in is Stage 2.
- **Multi-GPU.** `ensure_device(ordinal)` accepts a non-default ordinal as a
  transitional step on single-GPU systems, but the cell latches once and later
  calls with a different ordinal are silently ignored. Real multi-GPU is a
  Stage 2+ concern.
- **No live-GPU validation.** The one `#[ignore]`d smoke test
  (`cudarc_alloc_roundtrip`, gated on `BOLT_BENCH_GPU=1`) is a template for
  that follow-up; the only host-runnable test asserts the `cudarc_err`
  translation shape.

## Where cudarc fits in the architecture

```
                    ┌──────────────────────────────────┐
                    │      Application code            │
                    │  (Engine, execute_groupby, ...)  │
                    └─────────────────┬────────────────┘
                                      │
                    ┌─────────────────▼────────────────┐
                    │   CUDA-Oxide layer (unchanged)   │
                    │   GpuVec<T>, GpuView<'a, T>,     │
                    │   GpuViewMut<'a, T>, GpuBuffer   │
                    └─────────────────┬────────────────┘
                                      │
   ─── boundary cudarc replaces ──────┼───────────────────────────────
                                      │
                    ┌─────────────────▼────────────────┐
                    │  Driver FFI (cuda_sys.rs + ...)  │
                    │  cuMemAlloc_v2, cuLaunchKernel,  │
                    │  cuMemcpyHtoD_v2, cuModuleLoad…  │
                    └─────────────────┬────────────────┘
                                      │
                              libcuda / nvcuda.dll
```

The borrow-checked CUDA-Oxide types (`GpuVec<T>`, `GpuView<'a, T>`,
`GpuViewMut<'a, T>`, `GpuBuffer`) stay the public-facing layer; cudarc only
replaces what sits *below* them. Every executor keeps calling
`GpuVec::from_slice`, `keys_gpu.view()`, etc., unchanged.

## Staged migration plan (Stages 2–4 not yet done)

### Stage 1 — `cuda_sys.rs` alloc/memcpy surface — **LANDED**

Keep `GpuBuffer<T>`, `GpuVec<T>`, and `CudaContext` as-is; route the
allocation/free/copy primitives through cudarc behind the pool's
`#[cfg(feature = "cudarc")]` dispatch. The design keeps the raw `CUdeviceptr`
representation in `GpuBuffer<T>` (rather than storing a cudarc `CudaSlice<u8>`)
so the pool's recycling stays in charge of frees and `GpuBuffer::drop` is
unchanged.

### Stage 1.5 — async memcpy/memset — **LANDED**

`memcpy_h2d_async`, `memcpy_d2h_async`, and `memset_d8_async` go through
cudarc's raw `driver::sys` bindings, so there is no sync fallback under
`--features cudarc`.

### Stage 2 — PTX module loading — *not done*

Port `jit_compiler.rs::CudaModule` to cudarc's `CudaModule` /
`CudaDevice::load_ptx`. The PTX cache key (a hash of the PTX text) is
unchanged; only the cache value type and the manual `cuModuleUnload` `Drop`
impl change (cudarc's `CudaModule` Drops itself).

### Stage 3 — launch path + streams — *not done*

Port `src/exec/launch.rs`'s hand-rolled `CudaStream` and `KernelArgs` /
`launch_with_geometry` onto cudarc's typed launch API. The recent CUDA-Oxide
refactor of `groupby_shmem_exec.rs` (see [`BENCHMARKS.md`](BENCHMARKS.md)) is a
preview of the target shape. Migrate pinned-host alloc only if cudarc 0.14+
exposes a clean `cuMemAllocHost_v2`; otherwise leave it hand-rolled.

### Stage 4 — memory pool — *not done, likely declined*

cudarc has its own bucketed memory pool, but `DeviceMemPool`'s LRU eviction and
OOM-retry behaviour are tightly tuned to this workload and cudarc's pool is
less configurable. Likely outcome: keep ours.

## Regression-safety plan (for the remaining stages)

- All existing tests must pass under **both** feature flags, including the
  `compile_fail` doctests that encode the CUDA-Oxide borrow contract — cudarc's
  lifetimes must reject the same patterns ours do.
- The h2o.ai benchmark suite must run under both flags and produce identical
  results within the existing cross-engine relative tolerance in
  `benches/olap_benchmarks.rs`.
- Performance must not regress on the fast-path queries (q1 / q4 / q5).
- Memory-leak verification via `compute-sanitizer` on a large run — both
  backends should report zero leaks.

After Stage 4 and one release cycle of `default = ["cudarc"]` with no
regressions, the hand-rolled `#[cfg(not(feature = "cudarc"))]` path can be
deleted.

## See also

- [cudarc on GitHub](https://github.com/coreylowman/cudarc) — surface-area
  overview.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — the CUDA-Oxide layer that stays intact
  across this migration.
- [`BENCHMARKS.md`](BENCHMARKS.md) — the perf baseline the migration must not
  regress.
- `src/cuda/cudarc_backend.rs` — the landed Stage 1 / 1.5 implementation.
- `src/cuda/cuda_sys.rs` — the hand-rolled FFI the later stages would replace.
