# CUDA-Oxide refactor sweep — status

The borrow-checked GPU-memory layer (`GpuVec` / `GpuView` / `GpuViewMut`,
see [`docs/ARCHITECTURE.md`](./ARCHITECTURE.md#memory-safety-cuda-oxide))
is the project's headline design. Several executors landed with the
borrow-checker bypassed at the launch site — they call `.device_ptr()`
on a `GpuVec`, then assemble `*mut c_void` kernel-arg arrays and hand
them to `cuda_sys::cuLaunchKernel` directly inside an `unsafe` block.

This document tracks the migration of those launch sites to the typed
`KernelArgs<'a>` + `launch_with_geometry` API in
[`src/exec/launch.rs`](../src/exec/launch.rs). The pattern is the same
one [`groupby_shmem_exec.rs`](../src/exec/groupby_shmem_exec.rs) uses
as a reference implementation:

```rust
// Before:
let mut keys_ptr: CUdeviceptr = keys_gpu.device_ptr();
let mut vals_ptr: CUdeviceptr = vals_gpu.device_ptr();
let mut out_ptr:  CUdeviceptr = out_gpu.device_ptr();
let mut params: [*mut c_void; 5] = [
    &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
    &mut vals_ptr as *mut CUdeviceptr as *mut c_void,
    &mut out_ptr  as *mut CUdeviceptr as *mut c_void,
    &mut n_rows_u32 as *mut u32 as *mut c_void,
    &mut n_groups_u32 as *mut u32 as *mut c_void,
];
unsafe {
    cuda_sys::check(cuda_sys::cuLaunchKernel(
        function.raw(), grid, 1, 1, block, 1, 1, 0,
        stream.raw(), params.as_mut_ptr(), ptr::null_mut(),
    ))?;
}
stream.synchronize()?;

// After (CUDA-Oxide):
let view_keys = keys_gpu.view();
let view_vals = vals_gpu.view();
let mut view_out = out_gpu.view_mut();
let mut args = KernelArgs::empty();
args.push_input(&view_keys);
args.push_input(&view_vals);
args.push_output(&mut view_out);
args.push_scalar_u32(n_rows);
args.push_scalar_u32(n_groups);
launch_with_geometry(function, grid, block, 0, &stream, &mut args)?;
```

The win is that the borrow checker now rejects:

- dropping `keys_gpu` / `vals_gpu` / `out_gpu` while a launch is in
  flight (the view lifetime extends across the whole `args` scope);
- aliasing the output with another mutable view from the same scope
  (`view_mut()` takes `&mut self`);
- using `out_gpu` immutably elsewhere while it's bound as a `view_mut`
  here.

All of that was previously a `// SAFETY: …` comment we hoped held.

## Status table

| Executor | Status | Notes |
| --- | --- | --- |
| `groupby_shmem_exec.rs` (Tier-1 single SUM) | ✅ refactored | The reference / PoC. |
| `groupby_shmem_count_exec.rs` (Tier-1 COUNT) | ✅ refactored | Uses `KernelArgs::empty` + push pattern. |
| `groupby_shmem_minmax_exec.rs` (Tier-1 MIN/MAX) | ✅ refactored | Same pattern across i32 / i64 dispatch arms. |
| `groupby_shmem_avg_exec.rs` (Tier-1 AVG) | ✅ refactored | Each SUM launch + the single COUNT launch use `KernelArgs::empty` + push pattern. |
| `groupby_shmem_multi_exec.rs` (Tier-1 multi-SUM) | ✅ refactored | First user of the relaxed `KernelArgs` API (the `'b: 'a` split below). Iterated `view()`/`view_mut()` over `Vec<GpuVec<T>>` now compiles cleanly. |
| `groupby_tier2_exec.rs` (Tier-2 single SUM entry shim) | ✅ no launch sites — naturally CUDA-Oxide-clean | Pure shim — delegates to `execute_tier2_sum`; no `cuLaunchKernel` to lift. The orchestrator it calls is the actual refactor target. |
| `groupby_tier2_orchestrator.rs` (Tier-2.1 partition → scatter → reduce) | ⏳ todo | Three sequential launches, each independently refactorable. |
| `groupby_tier2_twokey_orchestrator.rs` (i64 sibling) | ⏳ todo | Same shape as the i32 orchestrator. |
| `groupby_tier2_multi_orchestrator.rs` (Tier-2.1 multi-SUM) | ⏳ todo | `4 + 2N` param launch; same "iterated views" issue as `groupby_shmem_multi_exec`. |
| `groupby_tier2_avg_exec.rs` (Tier-2.1 AVG) | ⏳ todo | Three launches (partition, scatter ×N, multi-SUM reduce, count reduce). |
| `groupby_tier2_count_exec.rs` (Tier-2.1 COUNT) | ✅ refactored | Partition + scatter + reduce launches all use `KernelArgs::empty` + push pattern. |
| `groupby_tier2_minmax_exec.rs` (Tier-2.1 MIN/MAX) | ⏳ todo | Two reduce-phase helper functions; refactor each independently. |
| `groupby_tier2_twokey_multi_exec.rs` (Tier-2.1 two-key + multi-agg) | ⏳ todo | Combines i64 partition + scatter + multi-value reduce. Largest launch in the tree. |

## API relaxation — landed

The previous `push_input/push_output<T>(view: &'a GpuView<'a, T>)`
signature unified the outer-borrow lifetime with the view's
inner-borrow lifetime, which made iterated views over a
`Vec<GpuVec<T>>` fight the borrow checker:

```rust
let views: Vec<GpuView<'_, f64>> = vals.iter().map(|v| v.view()).collect();
for view in &views {
    args.push_input(view);   // formerly rejected
}
```

Relaxed signature (now in `src/exec/launch.rs`):

```rust
pub fn push_input<'b, T: bytemuck::Pod>(&mut self, view: &'b GpuView<'a, T>)
where
    'a: 'b,
{ ... }
```

The view's *inner* lifetime (its borrow of the parent GpuVec) must
outlive `'a` (the `KernelArgs`'s lifetime — so the kernel-arg list
can't outlive the device allocation). The view's *outer* lifetime
`'b` only needs to outlive the call — distinct from `'a`, so an
iterated push from a Vec of views works.

This unblocked `groupby_shmem_multi_exec.rs` as the second
user. The pattern now generalises to every other executor in the
status table above.

## Recommended order to finish the sweep

1. **Single-launch executors first** (count, exec, minmax, single-SUM
   variants of orchestrators): mechanical, low-risk, ~10 min each.
2. **`KernelArgs` API relaxation** (the `'b` / `'a` separation above):
   one PR, ~30 min, unlocks the rest.
3. **Multi-aggregate executors**: each ~15 min once the API change
   lands.

Total: 4–6 engineering-hours for the remaining 10 executors. None is
blocking; they all already work — the refactor only tightens
compile-time guarantees.
