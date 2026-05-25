# cudarc adoption design doc

## TL;DR

Replace Craton Patina's hand-rolled CUDA Driver API FFI in `src/cuda/cuda_sys.rs`
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

Craton Patina's design already cleanly separates the borrow-checked Rust types
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

pub fn mem_alloc(bytes: usize) -> PatinaResult<CUdeviceptr> {
    let mut ptr: CUdeviceptr = 0;
    check(unsafe { cuMemAlloc_v2(&mut ptr, bytes) })?;
    Ok(ptr)
}

// AFTER (cudarc-backed):
pub fn mem_alloc(bytes: usize) -> PatinaResult<CUdeviceptr> {
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
default = ["cudarc"]                        # opt-in to cudarc by default
cudarc = ["dep:cudarc"]
hand-rolled = []                            # the current path, kept for now

[dependencies]
cudarc = { version = "0.13", optional = true, features = ["driver", "nvrtc"] }
```

`src/cuda/cuda_sys.rs` (and friends) get `#[cfg(feature = "cudarc")]`
/ `#[cfg(feature = "hand-rolled")]` blocks during the transition.
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
   We'd add a `From<DriverError> for PatinaError::Cuda` impl in
   `src/error.rs` (~5 lines).
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
| `check(code: CUresult)` | `?` on cudarc results | cudarc returns `Result<_, DriverError>` directly. Add `From<DriverError> for PatinaError::Cuda` (~5 LOC in `src/error.rs`). |
| `init()` | `CudaContext::new(0)?` (Lazy) | cudarc initialises the driver on first device acquisition. The current `init_once` indirection becomes a `OnceLock<Arc<CudaContext>>`. |
| `device_count()` | `cudarc::driver::result::device::get_count()` | Direct port. |
| `device_get(ordinal)` | `CudaContext::new(ordinal)?` | cudarc combines device-handle + context creation into one call — fewer footguns. |
| `device_name(dev)` | `dev.name()?` | Returns `String`, no `CStr` dance. |
| `mem_alloc(bytes)` | `device.alloc_zeros::<u8>(n)?` or `device.alloc::<T>(n)?` | cudarc owns the slice via `CudaSlice<T>`; the `device_ptr()` lives as long as the slice. We'd wrap a `CudaSlice<u8>` inside `GpuBuffer<T>` and reinterpret in our typed accessor. |
| `mem_free(ptr)` | `drop(slice)` (RAII) | Manual `mem_free` goes away — `CudaSlice::Drop` handles it. The whole `mem_pool` becomes optional (cudarc has its own internal recycling). |
| `memcpy_h2d::<T>(dst, src, n)` | `device.htod_copy(host_vec)?` or `device.htod_sync_copy_into(...)` | cudarc takes `&[T]`, returns `CudaSlice<T>`. Our existing `GpuBuffer::from_slice` path is essentially the same call. |
| `memcpy_d2h::<T>(dst, src, n)` | `device.dtoh_sync_copy(&slice)?` | Returns `Vec<T>`. Matches `GpuBuffer::to_vec`. |
| `mem_alloc_host`/`mem_free_host` | Not needed in cudarc path | cudarc's `htod_copy` accepts plain `&[T]`. Page-locked memory is exposed via `Pinned<T>` if we need it later. |
| `memset_d8(ptr, val, n)` | `device.memset_zeros(&mut slice)?` or `slice.fill(val)?` | Available; signature change is minor. |

### Feature-flag boilerplate (ready to drop into Cargo.toml)

```toml
[features]
default = []                    # keep hand-rolled until we soak cudarc
cudarc = ["dep:cudarc"]
cuda-stub = []                  # unchanged — for non-CUDA hosts

[dependencies]
cudarc = { version = "0.13", optional = true, default-features = false,
           features = ["driver"] }
```

### Migration sketch (`src/cuda/cuda_sys.rs`)

```rust
#[cfg(feature = "cudarc")]
mod backend {
    use cudarc::driver::{self, CudaContext};
    use std::sync::Arc;

    static GLOBAL_CTX: once_cell::sync::OnceCell<Arc<CudaContext>> =
        once_cell::sync::OnceCell::new();

    pub fn current_context(ordinal: i32) -> PatinaResult<Arc<CudaContext>> {
        GLOBAL_CTX
            .get_or_try_init(|| {
                CudaContext::new(ordinal as usize)
                    .map(Arc::new)
                    .map_err(|e| PatinaError::Cuda(format!("cudarc: {e}")))
            })
            .map(Arc::clone)
    }

    pub fn mem_alloc(bytes: usize) -> PatinaResult<CudaSlice<u8>> {
        current_context(0)?
            .alloc_zeros::<u8>(bytes)
            .map_err(|e| PatinaError::Cuda(format!("cudarc alloc: {e}")))
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

### Stage 1 spike — LANDED

Status as of this commit:

- `cudarc = "0.13"` added to `Cargo.toml` as an **optional** dependency,
  feature-gated behind `[features] cudarc = ["dep:cudarc"]`.
- New `src/cuda/cudarc_backend.rs` (~130 LOC) exposes
  `mem_alloc` / `mem_free` / `memcpy_h2d` / `memcpy_d2h` backed by
  cudarc's `result::{malloc_sync, free_sync, memcpy_htod_sync,
  memcpy_dtoh_sync}`. `CudaDevice::new(0)` is cached in a `OnceCell`
  so primary-context bootstrapping happens once.
- `src/cuda/mod.rs` registers the module behind `#[cfg(feature =
  "cudarc")]` so default builds remain identical.
- Both `cargo build --release` (default) and `cargo build --release
  --features cudarc` complete without errors. The two paths coexist:
  with `--features cudarc` the cudarc crate compiles in but is not
  yet *wired* into `GpuBuffer<T>` (that's Stage 1.5 — see below).

What's deliberately **not** done in this session:

- `GpuBuffer<T>::with_capacity` still calls the hand-rolled
  `cuda_sys::mem_alloc`. Switching it to `cudarc_backend::mem_alloc`
  under the feature flag is a one-line `cfg!` shim — but breaks the
  `mem_pool::POOL` invariants if the pool stores cudarc-minted
  pointers and the cleanup path uses raw cuMemFree, or vice versa.
  Resolving that interaction is Stage 1.5.
- `CudaContext` and `CudaModule` still use the hand-rolled FFI. cudarc
  exposes equivalents (`CudaDevice`, `CudaDevice::load_ptx`) but
  swapping them in is Stage 2.
- No tests have been run under `--features cudarc` against a live
  GPU. The one `#[ignore]`'d smoke test in `cudarc_backend.rs` is a
  template for that follow-up.

The spike's purpose was: prove cudarc COMPILES inside Craton Patina behind a
feature flag and exposes the right primitives for the migration. That's
done. The actual migration (call-site swap, pool reconciliation, test
sweep) is the next 1–2 day unit of work.

### Why this session does NOT execute the FULL migration

`GpuBuffer<T>` currently exposes a raw `CUdeviceptr` accessor that the
launch path (`launch_with_geometry`, `cuLaunchKernel`) feeds into a
`*mut c_void` parameter slot. cudarc's `CudaSlice<T>` does expose a
device pointer (`slice.device_ptr()` returns the same `CUdeviceptr`),
so the launch path stays unchanged — **but** every `GpuBuffer<T>`
constructor needs to swap from `cuda_sys::mem_alloc(bytes)` to
`backend::mem_alloc(bytes)`, and the resulting `CudaSlice<u8>` needs
to be stored inside the struct (replacing the raw `CUdeviceptr`).

That's about a dozen edits across `buffer.rs` plus the `mem_pool.rs`
overhaul (cudarc's slice doesn't surface a re-allocatable pool the
way we do; we'd either keep ours or delete it and rely on cudarc's
internal caching). The risky bit is `Drop` semantics: today
`GpuBuffer::drop` calls `cuda_sys::mem_free`; under cudarc it drops
the inner `CudaSlice<u8>` (which calls `cuMemFree` itself). One has
to be careful that `mem_pool.rs`'s drain doesn't double-free.

Estimated effort: **1–2 engineering days** of focused work + a full
bench + test sweep under both feature flags before flipping
`default = ["cudarc"]`. That's the right unit of work for a separate
PR, not an afterthought to a multi-task session.

## Suggested execution order

1. **Spike** (half a day): port `mem_alloc` / `mem_free` /
   `memcpy_{h2d,d2h}` to cudarc behind a feature flag. Verify a
   single bench query (q1) runs end-to-end with identical results.
2. **Stage 1 full** (1–2 days): finish the `cuda_sys.rs` migration.
3. **Stage 2** (1 day): port `jit_compiler.rs::CudaModule`.
4. **Bench at end of Stage 2** — checkpoint. Numbers must match.
5. **Stage 3** (2–3 days): port `CudaStream` + launch path.
6. **Stage 4** (1–2 days): port memory pool.
7. **Final bench + comprehensive doc update**. Flip
   `default = ["cudarc"]`. Remove the `hand-rolled` cfg blocks after
   one release cycle of soak.

Total realistic estimate: **5–8 engineering days** for the full
migration, **1–2 days** if we ship Stage 1 only.

## See also

- [cudarc on GitHub](https://github.com/coreylowman/cudarc) — README
  has the surface-area overview.
- [`docs/ARCHITECTURE.md`](./ARCHITECTURE.md) — explains the CUDA-Oxide
  layer that stays intact across this migration.
- [`docs/BENCHMARKS.md`](./BENCHMARKS.md) — perf baseline that the
  migration must not regress.
- [`src/cuda/cuda_sys.rs`](../src/cuda/cuda_sys.rs) — the file being
  replaced.
