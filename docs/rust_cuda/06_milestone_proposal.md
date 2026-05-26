# 0.3 Milestone Proposal — Rust-CUDA Kernel Migration

**Milestone:** Rust-source GPU kernels via `rustc_codegen_nvvm`
**Target release:** 0.3
**Status:** Proposal (integrator-merged into `ROADMAP.md` once the other
`docs/rust_cuda/` agents land).

## Executive summary

Today every GPU kernel in Craton Bolt is emitted as PTX text from a Rust
function in `src/jit/*_kernel*.rs`. That worked through 0.1 and 0.2, but
the project is at ~18 kernels and growing — Tier-2 GROUP BY alone added
the `partition_*`, `scatter_*`, and `partition_reduce_*` families in the
last cycle. The 0.3 milestone replaces the hand-emit path with Rust
source compiled to PTX by `rustc_codegen_nvvm`. Public API does not
change. The existing `CudaModule::from_ptx` loader stays. The deliverable
is the same PTX, produced by a Rust compiler instead of a Rust function
that concatenates strings.

## Motivation

### PTX text emission is at its readability ceiling

The hand-emit kernels are now the densest unreadable code in the tree.
`src/jit/partition_reduce_kernel_minmax.rs` and
`src/jit/shmem_multi_sum_kernel.rs` mix register allocation, shared-memory
layout, and atomic-CAS loops as fmt-string concatenation. Bugs land in
the spaces between two `format!` calls. The maintainability tax has been
acceptable at the current kernel count; it will not be acceptable when
the count doubles (`COUNT DISTINCT`, decimal aggregates, hash join probe,
GPU sort all want kernels in 0.3+).

### Borrow-checked GPU code, end to end

Craton Bolt already gets compile-time safety on the host side through
`GpuVec` / `GpuView` / `GpuViewMut` — use-after-free, double-free, and
mutable/shared aliasing across kernel boundaries are rejected at
compile. Inside the kernels themselves we have none of that: the PTX
emitter has no model of which pointer is read-only and which is
write-only, no slice-length check, no aliasing rule. Rust-CUDA brings
those guarantees down into the kernel body. The same borrow checker
that catches host bugs will catch the kernel-side equivalents before
they reach PTXAS.

### Rust-CUDA ecosystem traction

`rust-cuda` was un-archived in late 2024 with renewed maintenance and a
shipping `rustc_codegen_nvvm`. The bet is to pay down the structural
debt of hand-emit PTX *before* kernel count doubles, while the upstream
project is active. Waiting another release cycle means migrating twice
the surface area against the same maintenance window.

## Scope

### In

- Rewrite every current `src/jit/*_kernel*.rs` file as a Rust kernel in
  a new `kernels/` workspace member.
- Wire the new PTX outputs through the existing `CudaModule::from_ptx`
  loader — the host-side launcher and `KernelSpec` cache are unchanged.
- Maintain feature parity with current behaviour: every executor that
  exists today, every shape dispatched by `PhysicalPlan`, every bench in
  `cargo bench`, continues to work.
- A `build.rs` in `kernels/` that emits PTX into a known path; the main
  crate consumes those artefacts.

### Out

- **Device-side allocation.** Kernels remain pre-sized by the host.
- **Dynamic kernel parameterisation.** Rust-CUDA kernels are
  compile-time monomorphic. A kernel parameterised on `n_vals: 1..=4`
  ships as four variants — that scales for small enumerations but does
  not scale to arbitrary runtime parameters. Documented as a non-goal
  for 0.3; revisited in 0.4 with const-generic kernels or a proc-macro
  specialisation pass.
- **NVRTC fallback.** We commit to `rustc_codegen_nvvm`. The hand-emit
  path stays in git history for one soak release and is then deleted.
- **Public API changes.** `Engine::sql`, `register_table`,
  `DataFrame::collect` are untouched.

## Sub-milestones

Six waves, ~9 engineering weeks for a single engineer. Wave gating is
explicit: no wave merges to `main` unless the bench gate (see Risks)
holds.

### Wave A — toolchain spike (~1 week)

Get a single Rust kernel — `partition_kernel` — compiled to PTX by
`rustc_codegen_nvvm` and loaded through the existing
`CudaModule::from_ptx` path. Bench parity is a sanity check, not a
gate: the kernel round-trips the same way the hand-emit version does.
Output: a runnable end-to-end example and a documented `rust-toolchain`
pin.

### Wave B — Cargo workspace + build pipeline (~1 week)

Split `kernels/` into its own workspace member. Wire `build.rs` so that
`cargo build` in the root produces PTX artefacts and the main crate
picks them up via `include_str!` (or equivalent). `cargo build` green
end-to-end, `cargo test` green on whatever ports from Wave A. CI on
both the `cuda` and `cuda-stub` feature paths.

### Wave C — simple kernels (~2 weeks)

Port the kernels with no shared memory and no atomics:
- `scan_kernel`
- `agg_kernels` (the per-dtype reduction tree)
- `prefix_scan` (Hillis-Steele)
- `scatter_kernel`

These are the lowest-risk migrations. Bench gate enforced from here on.

### Wave D — Tier-1 GROUP BY kernels (~2 weeks)

Port the shared-memory + integer-atomic kernels that carry the project's
GROUP BY perf wins:
- `shmem_sum_kernel`
- `shmem_count_kernel`
- `shmem_minmax_kernel` (integer paths)
- `shmem_multi_sum_kernel`

Atomic-i32 / atomic-i64 surfaces in `cuda_std` are exercised here.

### Wave E — Tier-2 GROUP BY kernels (~2 weeks)

Port the partition/scatter/reduce family:
- `partition_kernel`, `partition_kernel_i64`
- `scatter_kernel`, `scatter_kernel_i64`
- `partition_reduce_kernel`, `_i64`, `_count`, `_minmax`, `_multi`

`partition_reduce_kernel_minmax` on float keys is the highest-complexity
item — it is a `atom.cas` loop on the bit pattern (no native
`atom.global.{min,max}.f*` through sm_90). Expect this kernel to take
~3 days on its own.

### Wave F — cleanup (~1 week)

Delete `src/jit/*_kernel*.rs`. Update `docs/JIT_PIPELINE.md` and
`docs/ARCHITECTURE.md`. Run the full bench delta sweep against the 0.2
baseline. Update `ROADMAP.md`. Tag 0.3.

### Carry-over risk

If `rustc_codegen_nvvm` breaks under a mid-migration Rust release, the
estimate slips. Mitigation in the Risks section below.

## Risks + mitigations

1. **Rust-CUDA toolchain abandoned again.**
   *Mitigation:* pin to a known-good `rust-cuda` revision in
   `Cargo.lock`. Keep `src/jit/*_kernel*.rs` in git history for one
   release cycle (0.3 → 0.4) as a documented recovery path. Wave F
   deletes the files; the history remains.

2. **Performance regression from machine-generated PTX.**
   *Mitigation:* per-wave bench gate. No wave merges to `main` unless
   every affected kernel stays within ±5 % of the hand-emit baseline
   recorded at 0.2. The h2o.ai bench queries are the reference workload;
   `docs/BENCHMARKS.md` records the baseline.

3. **Build-time explosion.**
   *Mitigation:* snapshot the produced PTX into the repo. Downstream
   builds skip kernel-recompile by default; opt-in
   `--features rebuild-kernels` rebuilds from Rust source. Keeps
   `cargo build` on a developer machine in the same order of magnitude
   as today.

4. **Dynamic kernel parameterisation gap.**
   *Mitigation:* ship one variant per N for `n_vals: 1..=4` (the only
   runtime-parameterised shape we currently emit). Documented as a
   non-goal beyond that range. 0.4 revisits with const generics.

5. **Nightly Rust pinning leaks into the host crate.**
   *Mitigation:* `rust-toolchain.toml` confined to the `kernels/`
   workspace member. The main crate stays on stable. CI exercises both.

## Success criteria

0.3 is done when *all* of the following hold:

- Every h2o.ai bench query continues to verify equivalence against
  Polars and DuckDB (the existing competitive-bench gate).
- No measured query regresses > 5 % vs the 0.2 baseline recorded in
  `docs/BENCHMARKS.md`.
- `src/jit/*_kernel*.rs` is empty, or contains only loader / dispatch
  infrastructure — no PTX-emitter functions remain.
- A new contributor can add `kernels/src/foo.rs`, declare it in the
  `kernels/` workspace member, and have it picked up by the build with
  no changes elsewhere in the tree. (Loader registration on the host
  side is the one exception and is documented.)

## What this milestone is NOT

- **Not a `cudarc` adoption.** That is a separate 0.2.x stage,
  documented in `docs/CUDARC_ADOPTION.md`. Independent of this work.
- **Not a multi-GPU adoption.** One CUDA context, one device per
  `Engine` — unchanged from 0.1.
- **Not a CUDA Graphs adoption.** Stream and graph capture remain out
  of scope.
- **Not an SQL-feature expansion.** No new SQL surface lands in 0.3.
  CTE, window functions, `IS NULL`, `LIKE`, decimal — all still 0.2 /
  0.4 work as listed in `ROADMAP.md`.

## Carry-over to 0.4+

- **Const-generic kernels.** Parameterise N at compile time elegantly,
  closing the dynamic-parameterisation gap from this milestone.
- **Host/device type sharing.** A struct definition consumed by both
  the launcher and the kernel — currently impossible because the
  emitter only knows strings. Rust-CUDA makes this an obvious next
  step.
- **Maybe: device-side dynamic dispatch via `cuda_std`.** Allows
  per-row work that does not fit into a single static kernel today
  (e.g. column-typed reductions in one launch). Speculative, gated on
  what we learn during Wave E.
