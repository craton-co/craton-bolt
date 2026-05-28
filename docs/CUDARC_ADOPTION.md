# cudarc adoption design doc

## TL;DR

Replace Craton Bolt's hand-rolled CUDA Driver API FFI in `src/cuda/cuda_sys.rs`
(~470 LOC) and `src/cuda/buffer.rs` (~250 LOC) with
[**cudarc**](https://github.com/coreylowman/cudarc), a well-maintained
pure-Rust CUDA Driver / Runtime / NVRTC binding crate. Migration is
feature-flagged, single-PR per layer, and the existing CUDA-Oxide types
(`GpuVec<T>`, `GpuView<'a, T>`, `GpuViewMut<'a, T>`) stay exactly the
same — cudarc replaces what's *below* them, not what's *above* them.

Expected wins:
- **~700 LOC of FFI maintenance burden deleted** (cuda_sys.rs, the manual
  link.rs gymnastics, the CUDA-version-specific cuda.lib mismatch workaround
  documented in BENCHMARKS.md).
- **`Send` / `Sync` correctness for free** — cudarc threads lifetimes
  through its handle types so concurrent context misuse is a compile
  error, not a runtime ACCESS_VIOLATION. (Recall the memory-pool stale-
  pointer class of bug we just fixed manually in `CudaContext::drop`.)
- **NVRTC integration** — cudarc has a clean `nvrtc::compile_ptx()` that
  could replace our `cuModuleLoadDataEx` path with a real NVRTC-driven
  one if we ever want it. Today's path is fine; this is optional headroom.
- **Stream / event ergonomics** — cudarc's `CudaStream` is `Send + Sync`
  and supports both null-stream and explicit streams uniformly. Our
  `src/exec/launch.rs::CudaStream` is a hand-rolled wrapper that mostly
  exists to avoid the manual drop and the Send/Sync impl gotchas; cudarc
  does it correctly.

Costs:
- A modest dev-dep / build-dep delta (cudarc itself is pure Rust, but
  adds 2–3 transitive deps). Compile time should be slightly *faster*
  because we drop the `link.rs` shenanigans.
- One feature-flag rebuild matrix: `--features cudarc` vs the default
  hand-rolled path during the transition. Removed once we're confident.
- About 1 week of focused engineering for the full migration. Lower
  if we accept "Stage 1 only" (cuda_sys.rs replacement).

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

Craton Bolt's design already cleanly separates the borrow-checked Rust types
(`GpuVec` and friends) from the raw FFI. The CUDA-Oxide types stay the
public-facing API; cudarc just becomes a better implementation
underneath.

## What stays the same

The public API of `src/cuda/` does NOT change:

- `GpuVec<T>::from_slice`, `::zeros`, `::with_capacity`, `::to_vec`,
  `::view`, `::view_mut`, `::device_ptr`, `::len`, …
- `GpuView<'a, T>`, `GpuViewMut<'a, T>` (the borrow-checked view types).
- `CudaContext::new`, the field-drop-order rules in `Engine`.
- `CudaModule::from_ptx` and the process-wide PTX cache in `jit_compiler`.
- `KernelArgs<'a>` and `launch_with_geometry`.

Every executor (`groupby.rs`, `groupby_shmem_exec.rs`, the Tier-2
orchestrators, …) is **untouched** by this migration. They keep calling
`GpuVec::from_slice`, `keys_gpu.view()`, etc. cudarc lives entirely below
that line.

## Stages

### Stage 1 — replace `cuda_sys.rs` (1–2 days)

The smallest possible migration: keep `GpuBuffer<T>`, `GpuVec<T>`, and
`CudaContext` exactly as they are; swap the `extern "C"` FFI bindings
for cudarc's. Map our internal helpers (`mem_alloc`, `mem_free`,
`memcpy_h2d`, `memcpy_d2h`, `check`, `init`, `device_get`, …) to cudarc
equivalents one-for-one.

Concretely, the diff is to `src/cuda/cuda_sys.rs`:

```rust
// BEFORE (current):
extern "C" {
    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    // ... ~30 more lines of extern "C" bindings
}

pub fn mem_alloc(bytes: usize) -> BoltResult<CUdeviceptr> {
    let mut ptr: CUdeviceptr = 0;
    check(unsafe { cuMemAlloc_v2(&mut ptr, bytes) })?;
    Ok(ptr)
}

// AFTER (cudarc-backed):
pub fn mem_alloc(bytes: usize) -> BoltResult<CUdeviceptr> {
    let device = current_device()?;     // cudarc's CudaDevice
    let slice = device.alloc_zeros::<u8>(bytes)?;
    Ok(slice.as_kernel_param() as CUdeviceptr)
}
```

The interesting subtlety: cudarc owns its allocations and returns a
`CudaSlice<u8>` (or generic `CudaSlice<T>`) handle. We'd either:
- Wrap it inside `GpuBuffer<T>` (`buffer.rs` already holds `(ptr, len,
  capacity)`; replace `ptr` with `Option<CudaSlice<u8>>` and adjust
  `Drop`).
- Or `.leak()` the slice to get back a raw `CUdeviceptr` and free it
  manually in `Drop` via `cudarc::driver::CudaContext::release` (less
  idiomatic but minimises behavioural change).

Recommended: **wrap the CudaSlice**, accept the small structural change
to `GpuBuffer<T>`, gain the borrow-tracked drop semantics for free.

### Stage 2 — replace `jit_compiler.rs` PTX-load path (1 day)

Today: `CudaModule::from_ptx` calls our `PTX_CACHE` hash-deduped
`cuModuleLoadDataEx`. cudarc has `CudaDevice::load_ptx()` returning a
`CudaModule` that the device tracks. The PTX cache stays — it's higher up.

Migration is just swapping `cuModuleLoadDataEx` (raw FFI) for
`CudaDevice::load_ptx_from_string()` (typed wrapper). The cache key
(64-bit hash of PTX text) stays the same; the cache value goes from
`Arc<CudaModuleInner>` to `Arc<cudarc::driver::CudaModule>`.

The `Drop` impl of `CudaModuleInner` (which manually calls
`cuModuleUnload`) is deleted — cudarc's `CudaModule` Drops itself.

### Stage 3 — replace `launch.rs::CudaStream` + KernelArgs (2–3 days)

Today: `launch_with_geometry` builds the `*mut c_void` array by hand,
inside a `unsafe { cuLaunchKernel(...) }`. cudarc has a typed
`LaunchAsync` trait that accepts kernel args via a generic tuple:

```rust
unsafe {
    function.launch(cfg, (
        &keys_slice,
        &vals_slice,
        &mut out_slice,
        n_rows_u32,
        n_groups_u32,
    ))?;
}
```

The `unsafe` is unavoidable because the PTX kernel signature isn't
known to the compiler — we're matching it by position. But the
*aliasing* and *lifetime* discipline is enforced: `&mut out_slice`
proves to the borrow checker that nothing else references it for the
launch.

Our existing `KernelArgs<'a>` is a stepping stone toward this; the
recent CUDA-Oxide refactor of `groupby_shmem_exec.rs` is a preview of
what every executor will look like under cudarc.

### Stage 4 — replace memory pool (1–2 days)

`src/cuda/mem_pool.rs::DeviceMemPool` is a process-wide
`Lazy<DeviceMemPool>` that we hand-built. cudarc has its own
`MemoryPool` type with similar bucket semantics but with the lifetime
discipline cudarc carries everywhere — pool entries are tied to the
`CudaDevice` that minted them, so the "stale pointer after context
destroy" class of bug we fixed manually in `CudaContext::drop` becomes
impossible.

Migrating: replace `POOL.alloc(bytes)` with the device's memory pool
calls. The visible difference at our call sites is none —
`GpuBuffer::with_capacity` still routes through the pool — only the
implementation below changes.

## Feature-flag plan

Add to `Cargo.toml`:

```toml
[features]
default = []                                    # keep hand-rolled until we soak cudarc
cudarc = ["dep:cudarc"]
cuda-stub = []                                  # unchanged — for non-CUDA hosts

[dependencies]
cudarc = { version = "0.13", optional = true, default-features = false, features = ["driver", "cuda-12060"] }
```

`src/cuda/cuda_sys.rs` (and friends) get `#[cfg(feature = "cudarc")]`
/ `#[cfg(not(feature = "cudarc"))]` blocks during the transition.
After Stage 4 and one release cycle of "default = cudarc" with no
regressions, delete the hand-rolled path entirely.

## Regression-safety plan

- **All existing tests pass under both feature flags.** The
  `compile_fail` doctests in `tests/memory_tests.rs` (the CUDA-Oxide
  contract) are particularly important here — cudarc's lifetimes must
  reject the same patterns ours do.
- **The h2o.ai benchmark suite runs under both flags** and produces
  identical results within 1e-9 relative tolerance via the existing
  cross-engine equivalence check in `benches/olap_benchmarks.rs`.
- **Performance must not regress > 5 %** on any of q1 / q4 / q5 (the
  three queries where we currently engage fast paths). cudarc's
  overhead is typically zero, but we measure to be sure.
- **Memory-leak verification** via `cuda-memcheck` (or its modern
  equivalent `compute-sanitizer`) on a 1 M-row run. Both the
  hand-rolled path and the cudarc path should report 0 leaks.

## Out of scope

- **Writing kernels in Rust.** That's `cust` / `rustc_codegen_nvvm`,
  a different and much larger migration. See ROADMAP for the 0.3
  milestone proposal.
- **Multi-GPU.** cudarc supports it natively; we don't have a use case
  yet. Wire it up when we do.
- **CUDA Graphs.** cudarc has `CudaGraph` support; we don't use Graphs
  today and the win is workload-specific. Defer.

## Open questions

1. **Does cudarc's `CudaSlice<T>` re-allocate on resize?** Our
   `GpuBuffer<T>` doesn't — `len` and `capacity` are decoupled. If
   cudarc's slice is fixed-size we keep `(slice, len)` in
   `GpuBuffer` and the API stays unchanged.
2. **PTX text passing.** Our cache stores PTX strings; cudarc accepts
   `&str` for `load_ptx_from_string`. Should be a no-op at the call
   site.
3. **Error type interop.** cudarc returns `cudarc::driver::result::DriverError`.
   **Decision (landed):** stringly-typed via `format!`. Every cudarc error
   becomes `BoltError::Other(format!("cudarc …: {e:?}"))` at the call site
   in `cudarc_backend.rs`. We considered a `From<DriverError>` impl, but
   the cudarc error variants are sparse and a `{:?}` debug print carries
   enough context for the error path; the trait impl would only add a
   layer of indirection without changing what the user sees.
4. **`cuda-stub` feature.** The current crate has a `cuda-stub` feature
   for docs.rs builds on hosts with no CUDA. cudarc handles this with a
   `dynamic-linking` feature flag — slightly nicer ergonomics. Stage 1
   should preserve the docs.rs build path.

## Stage 1 — concrete implementation spike

Inventory of `src/cuda/cuda_sys.rs`'s 12 public functions and their
cudarc replacements at the API level (verified against cudarc 0.13's
`CudaDevice` / `CudaSlice` / `result::driver::*` surface):

| Hand-rolled (today) | cudarc 0.13 equivalent | Notes |
| --- | --- | --- |
| `check(code: CUresult)` | `?` on cudarc results | cudarc returns `Result<_, DriverError>` directly. Add `From<DriverError> for BoltError::Cuda` (~5 LOC in `src/error.rs`). |
| `init()` | `CudaContext::new(0)?` (Lazy) | cudarc initialises the driver on first device acquisition. The current `init_once` indirection becomes a `OnceLock<Arc<CudaContext>>`. |
| `device_count()` | `cudarc::driver::result::device::get_count()` | Direct port. |
| `device_get(ordinal)` | `CudaContext::new(ordinal)?` | cudarc combines device-handle + context creation into one call — fewer footguns. |
| `device_name(dev)` | `dev.name()?` | Returns `String`, no `CStr` dance. |
| `mem_alloc(bytes)` | `device.alloc_zeros::<u8>(n)?` or `device.alloc::<T>(n)?` | cudarc owns the slice via `CudaSlice<T>`; the `device_ptr()` lives as long as the slice. We'd wrap a `CudaSlice<u8>` inside `GpuBuffer<T>` and reinterpret in our typed accessor. |
| `mem_free(ptr)` | `drop(slice)` (RAII) | Manual `mem_free` goes away — `CudaSlice::Drop` handles it. The whole `mem_pool` becomes optional (cudarc has its own internal recycling). |
| `memcpy_h2d::<T>(dst, src, n)` | `device.htod_copy(host_vec)?` or `device.htod_sync_copy_into(...)` | cudarc takes `&[T]`, returns `CudaSlice<T>`. Our existing `GpuBuffer::from_slice` path is essentially the same call. |
| `memcpy_d2h::<T>(dst, src, n)` | `device.dtoh_sync_copy(&slice)?` | Returns `Vec<T>`. Matches `GpuBuffer::to_vec`. |
| `mem_alloc_host`/`mem_free_host` | **Stays hand-rolled** even under `--features cudarc` | cudarc 0.13's `driver` feature does not expose `cuMemAllocHost_v2` cleanly, so the pinned-host alloc/free path continues to call our `cuda_sys::mem_alloc_host` raw FFI under both feature flags. See [`src/cuda/buffer.rs:396`](../src/cuda/buffer.rs) for the in-code explanation. Revisit when cudarc 0.14+ surfaces pinned-host primitives (see Stage 3 future-work). |
| `memset_d8(ptr, val, n)` | `device.memset_zeros(&mut slice)?` or `slice.fill(val)?` | Available; signature change is minor. |

### Feature-flag boilerplate (ready to drop into Cargo.toml)

```toml
[features]
default = []                    # keep hand-rolled until we soak cudarc
cudarc = ["dep:cudarc"]
cuda-stub = []                  # unchanged — for non-CUDA hosts

[dependencies]
cudarc = { version = "0.13", optional = true, default-features = false,
           features = ["driver", "cuda-12060"] }
```

### Migration sketch (`src/cuda/cuda_sys.rs`)

```rust
#[cfg(feature = "cudarc")]
mod backend {
    use cudarc::driver::{self, CudaContext};
    use std::sync::Arc;

    static GLOBAL_CTX: once_cell::sync::OnceCell<Arc<CudaContext>> =
        once_cell::sync::OnceCell::new();

    pub fn current_context(ordinal: i32) -> BoltResult<Arc<CudaContext>> {
        GLOBAL_CTX
            .get_or_try_init(|| {
                CudaContext::new(ordinal as usize)
                    .map(Arc::new)
                    .map_err(|e| BoltError::Cuda(format!("cudarc: {e}")))
            })
            .map(Arc::clone)
    }

    pub fn mem_alloc(bytes: usize) -> BoltResult<CudaSlice<u8>> {
        current_context(0)?
            .alloc_zeros::<u8>(bytes)
            .map_err(|e| BoltError::Cuda(format!("cudarc alloc: {e}")))
    }
    // … memcpy / device_count / etc.
}

#[cfg(not(feature = "cudarc"))]
mod backend { … existing extern "C" + thin wrappers … }

pub use backend::*;
```

The hand-rolled `extern "C"` block stays as the fallback. CI tests both
paths; we'd add a `cargo build --features cudarc` line to whatever
script runs.

### Stage 1 + Stage 1.5 — LANDED

Status as of this commit:

- `cudarc = "0.13"` added to `Cargo.toml` as an **optional** dependency,
  feature-gated behind `[features] cudarc = ["dep:cudarc"]`.
- `src/cuda/cudarc_backend.rs` exposes the full sync surface
  (`mem_alloc` / `mem_free` / `memcpy_h2d` / `memcpy_d2h` / `memset_d8`)
  backed by cudarc's `result::{malloc_sync, free_sync, memcpy_htod_sync,
  memcpy_dtoh_sync, memset_d8_sync}`. `CudaDevice::new(0)` is cached in
  a `OnceCell` so primary-context bootstrapping happens once.
- **Async memcpy / memset are wired** (commit `e79b568`, "review C3"):
  `memcpy_h2d_async`, `memcpy_d2h_async`, and `memset_d8_async` in
  `cudarc_backend.rs` call through `cudarc::driver::sys` raw bindings
  (`cuMemcpyHtoDAsync_v2`, `cuMemcpyDtoHAsync_v2`, `cuMemsetD8Async`).
  No more sync fallback under `--features cudarc`.
- **`GpuBuffer<T>::with_capacity` routes through the pool**, which under
  `--features cudarc` calls `cudarc_backend::mem_alloc` (see
  `mem_pool.rs:driver_mem_alloc` at line ~408 and `driver_free` at
  ~384). Pointers minted by cudarc and pointers minted by the
  hand-rolled FFI are bit-compatible `CUdeviceptr`s, so the pool's
  drain path is backend-agnostic.
- `src/cuda/mod.rs` registers the module behind `#[cfg(feature =
  "cudarc")]` so default builds remain identical.
- Both `cargo build --release` (default) and `cargo build --release
  --features cudarc` complete without errors.

What's deliberately **not** done in Stage 1 / 1.5:

- **Pinned-host alloc/free remain hand-rolled FFI even under
  `--features cudarc`** — cudarc 0.13's `driver` feature doesn't expose
  `cuMemAllocHost_v2` cleanly. The `PinnedHostBuffer` path
  (`src/cuda/buffer.rs:396`, ~line 517) continues to call
  `cuda_sys::mem_alloc_host` / `cuMemFreeHost` directly under both
  feature flags. Tracked as Stage 3 future work — revisit when cudarc
  0.14+ surfaces a usable pinned-host API.
- `CudaContext` and `CudaModule` still use the hand-rolled FFI. cudarc
  exposes equivalents (`CudaDevice`, `CudaDevice::load_ptx`) but
  swapping them in is Stage 2.
- No tests have been run under `--features cudarc` against a live
  GPU. The one `#[ignore]`'d smoke test in `cudarc_backend.rs` is a
  template for that follow-up.

Stage 1 + Stage 1.5 between them prove cudarc compiles, links, and
serves real allocations and async transfers behind the feature flag.
The next unit of work is Stage 2 (PTX module loading).

### Design choice: keep raw `CUdeviceptr` in `GpuBuffer<T>`

Rather than store a cudarc `CudaSlice<u8>` inside `GpuBuffer<T>`, the
landed Stage 1.5 keeps the raw `CUdeviceptr` representation and routes
allocation/free through the pool, which dispatches to either backend
behind `#[cfg(feature = "cudarc")]`. The cudarc and hand-rolled paths
mint bit-compatible `CUdeviceptr`s (cudarc's `malloc_sync` ultimately
calls `cuMemAlloc_v2` too), so the pool's free path is backend-agnostic
and `GpuBuffer::drop` did not need to learn about cudarc-specific drop
semantics.

The alternative — wrap `CudaSlice<u8>` inside `GpuBuffer<T>` and let
cudarc's RAII drive the free — was rejected because it would either
duplicate the pool (cudarc's `CudaSlice::Drop` calls `cuMemFree`
directly, bypassing `mem_pool::POOL`'s recycling) or require cudarc to
expose a "release without freeing" handle that 0.13 doesn't have.
Keeping the pool means keeping the raw pointer; the cost is that
`GpuBuffer::drop` still calls `mem_pool` rather than just `drop(slice)`,
which is fine.

The Stage 2/3/4 migrations (PTX module, launch path, stream lifetimes)
can adopt cudarc's typed handles independently of the buffer representation.

## Suggested execution order

1. ~~**Spike**~~ — **DONE**. `cudarc_backend.rs` ships with sync
   `mem_alloc` / `mem_free` / `memcpy_{h2d,d2h}` / `memset_d8`.
2. ~~**Stage 1 full**~~ — **DONE**. `cuda_sys` → `cudarc_backend`
   dispatch lives in `mem_pool.rs::driver_mem_alloc` /
   `driver_free`, so every `GpuBuffer<T>::with_capacity` routes through
   cudarc under `--features cudarc`. Pinned-host explicitly stays
   hand-rolled (see Stage 3 below).
3. ~~**Stage 1.5 — async memcpy**~~ — **DONE** (commit `e79b568`,
   "review C3"). `memcpy_h2d_async`, `memcpy_d2h_async`,
   `memset_d8_async` go through `cudarc::driver::sys` raw bindings.
4. **Stage 2** (1 day): port `jit_compiler.rs::CudaModule` to
   `cudarc::driver::CudaModule` / `CudaDevice::load_ptx`. The PTX
   cache key (64-bit FNV of PTX text) is unchanged; only the cache
   value type and the `Drop` impl change.
5. **Bench at end of Stage 2** — checkpoint. Numbers must match.
6. **Stage 3** (2–3 days): port `CudaStream` + launch path. Also:
   migrate pinned-host alloc to cudarc *if and only if* cudarc 0.14+
   exposes a clean `cuMemAllocHost_v2` surface; otherwise leave the
   hand-rolled FFI in place and document why.
7. **Stage 4** (1–2 days): consider migrating `DeviceMemPool` to
   cudarc's bucket-based pool. The wins are unclear: our pool's LRU
   eviction policy and OOM-retry behaviour are tightly tuned to our
   workload, and cudarc's pool is less configurable. Likely outcome:
   keep ours.
8. **Final bench + comprehensive doc update**. Flip
   `default = ["cudarc"]`. Remove the `hand-rolled` cfg blocks after
   one release cycle of soak.

Remaining realistic estimate post Stage 1.5: **3–5 engineering days**
for Stages 2–4. Stage 1 and the async-memcpy follow-up already cost
their estimated 1–2 days and shipped.

## See also

- [cudarc on GitHub](https://github.com/coreylowman/cudarc) — README
  has the surface-area overview.
- [`docs/ARCHITECTURE.md`](./ARCHITECTURE.md) — explains the CUDA-Oxide
  layer that stays intact across this migration.
- [`docs/BENCHMARKS.md`](./BENCHMARKS.md) — perf baseline that the
  migration must not regress.
- [`src/cuda/cuda_sys.rs`](../src/cuda/cuda_sys.rs) — the file being
  replaced.
