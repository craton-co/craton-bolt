# 0.4 Milestone Proposal — Kernel Parameterisation, Backend Convergence

**Milestone:** Consolidate the GPU backend decision; ship const-generic kernels
**Target release:** 0.4
**Status:** Proposal (drafted before 0.3 closes; revisited at the 0.3 tag)

## Executive summary

0.3 makes a bet on `rustc_codegen_nvvm` (see
[`docs/rust_cuda/06_milestone_proposal.md`](rust_cuda/06_milestone_proposal.md)).
The bet may win cleanly, fail outright, or land somewhere in the middle.
0.4 cannot wait for the answer to be uniform — it has to plan for all
three. The headline of 0.4 is **backend convergence**: by the end of the
cycle Javelin ships exactly one device-side codegen story (rust-cuda,
hand-emit PTX, or — speculatively — NVIDIA's CUDA-Oxide), the others are
removed or feature-gated to history, and the cudarc host-side track from
`docs/CUDARC_ADOPTION.md` is fully soaked. The secondary headline is
**kernel parameterisation**: closing the dynamic-`n_vals` gap that
[`07_risk_assessment.md`](rust_cuda/07_risk_assessment.md) R5 calls out
as 0.3's structural debt.

## Pre-0.4 state — three branching scenarios

0.4 scope is keyed off which scenario obtains at the 0.3 tag. The
post-0.3 status review names one of:

- **Scenario A — clean 0.3.** All 18 PTX kernels enumerated in
  [`02_kernel_inventory.md`](rust_cuda/02_kernel_inventory.md) are
  rewritten as Rust source in the `kernels/` workspace member. Every
  bench in [`docs/BENCHMARKS.md`](BENCHMARKS.md) stays within ±5 % of
  the 0.2 baseline. `src/jit/*_kernel*.rs` is empty of emitter
  functions. `rust-cuda` is the source of truth; hand-emit lives only
  in the `v0.2-handemit` tag.
- **Scenario B — Fallback B fired.** Wave A or a later wave hit
  Risk R1 / R3 / R4 from `07_risk_assessment.md` hard enough to trigger
  the full revert documented as "Fallback B". The `kernels/` workspace
  member is shelved on a `rust-cuda-experiment` branch. Javelin is
  back on hand-emit. The recovery window between the revert and 0.3
  tag was likely spent pushing the cudarc adoption further — Stage 1
  certainly landed (already in tree as of this writing), and Stage 2
  (CudaModule load) and/or Stage 3 (launch path) may be in as well.
- **Scenario C — partial success.** Wave A through (say) Wave D
  passed; Wave E regressed on the float `partition_reduce_minmax`
  kernel or on a `n_vals > 1` shape. Fallback A was invoked per kernel:
  the simple kernels are rust-cuda, the awkward ones are still
  hand-emit. The dispatch side has a kernel-source feature knob that
  needs cleanup.

The three sub-sections below give 0.4 a scope per scenario. The
cross-cutting work in §5 fits any scenario.

## 0.4 scope by scenario

### Scenario A → 0.4 themes (3 waves × 2 weeks ≈ 6 weeks)

Rust-CUDA is the floor. 0.4 builds on top of it.

- **Wave A1 — const-generic kernels.** Replace
  `partition_reduce_kernel_multi(n_vals: usize)` (currently a runtime
  parameter that drives one PTX-per-N) with a single
  `partition_reduce_kernel_multi<const N: usize>(...)` Rust generic.
  Same for `shmem_multi_sum_kernel`. The variants-per-N macro from
  Wave E of 0.3 is deleted; the host-side dispatch picks the
  monomorphisation at launch time from a `KernelSpec` lookup. Resolves
  R5 from `07_risk_assessment.md`.
- **Wave A2 — host/device type sharing.** A `kernels-shared/` crate
  exposes `#[derive(DeviceCopy)]` structs that both the launcher and
  the kernel consume — currently impossible because the hand-emit path
  knows only strings. First use: the partition-offsets descriptor in
  `src/exec/partition_offsets.rs` is one struct on both sides instead
  of two parallel definitions that drift.
- **Wave A3 — dynamic-dispatch experiment.** Spike a single executor
  (Tier-2 multi-agg) where per-aggregate work lives behind a
  `cuda_std`-side function pointer, so one launch handles
  `SUM(a), MIN(b), COUNT(*)` instead of three. This is speculative —
  it might land as a prod path, or it might land as a documented
  carry-over to 0.5. Either is an acceptable outcome.

Non-goals in Scenario A: cudarc Stage 4 (memory pool) unless it falls
out for free; multi-GPU; CUDA Graphs.

### Scenario B → 0.4 themes (4 waves × 1.5 weeks ≈ 6 weeks)

Hand-emit is the floor. 0.4 doubles down on the host-side track and
defers the kernel-language question to 0.5.

- **Wave B1 — cudarc Stage 2.** Land `CudaModule::from_ptx` →
  `CudaDevice::load_ptx_from_string()` per `docs/CUDARC_ADOPTION.md`
  Stage 2. Deletes the manual `cuModuleUnload` Drop. Single
  release-soak cycle under `--features cudarc`.
- **Wave B2 — cudarc Stage 3.** Launch path: `KernelArgs<'a>` →
  cudarc's typed `LaunchAsync` tuple per `CUDARC_ADOPTION.md` Stage 3.
  Touches every executor listed in
  [`docs/CUDA_OXIDE_SWEEP.md`](CUDA_OXIDE_SWEEP.md). The CUDA-Oxide
  sweep finishes inside this wave — every `⏳ todo` row goes `✅`.
- **Wave B3 — cudarc Stage 4 + pool consolidation.** Memory pool
  migration. Reconcile or delete `src/cuda/mem_pool.rs`. After this
  wave `default = ["cudarc"]` flips on and the hand-rolled FFI lives
  only behind `--features hand-rolled` for one cycle.
- **Wave B4 — rust-cuda re-evaluation.** Re-run the
  `01_toolchain_audit.md` audit against whatever rust-cuda HEAD looks
  like at the start of Q3 2026. Two questions decide whether 0.5
  retries the kernel-language migration: (i) is a tagged crates.io
  release of `rustc_codegen_nvvm` out? (ii) has NVIDIA's CUDA-Oxide
  shipped a 0.2 with PTX-text loading?

Non-goals in Scenario B: const-generic kernels (no Rust kernel source
to be generic over); dynamic dispatch; the whole rust-cuda surface area
is deferred.

### Scenario C → 0.4 themes (2 waves × 1 week ≈ 2 weeks, then pivot)

Mixed-state stabilisation. Cheaper than A or B because the bulk of the
work is decision-making and cleanup, not engineering.

- **Wave C1 — freeze the split.** For each kernel: confirm whether
  the rust-cuda or hand-emit version owns it. Migration regressions
  that hand-emit recovered cleanly stay hand-emit; rust-cuda kernels
  that passed the bench gate stay rust-cuda. Document the split in a
  new section of `docs/JIT_PIPELINE.md`.
- **Wave C2 — collapse the dispatch knob.** Today's Fallback-A
  feature flags (`kernel-partition-handemit`, …) become unconditional
  source picks based on the Wave C1 decision. No runtime knob, no
  feature gate — the dispatch table just says "this kernel comes from
  here". `Cargo.toml` shrinks.

Then pivot. With ~4 weeks left in a 6-week cycle, 0.4 picks up the
highest-priority cross-cutting item from §5 (likely streaming
execution; multi-GPU is more speculative). The pivot target is named
at the end of Wave C1 once the split is known.

## NVIDIA CUDA-Oxide vs `rust-cuda`

[`01_toolchain_audit.md`](rust_cuda/01_toolchain_audit.md) §5 notes
that NVIDIA shipped **CUDA-Oxide 0.1 on 2026-05-09** — a separate
official Rust-to-PTX compiler. As of this writing it is pre-beta. 0.4
has to take a position on it.

### What is NVIDIA CUDA-Oxide?

A first-party, NVIDIA-maintained pure-Rust GPU runtime. Distinct from
the community-maintained `Rust-GPU/Rust-CUDA` umbrella that drives
`rustc_codegen_nvvm`. Same goal (compile Rust to PTX), different
delivery vehicle (vendor-led, presumably tracking the proprietary
nvcc/PTX pipeline more directly), different release cadence (no tagged
0.2 as of 2026-05-25).

### Where does it overlap with rust-cuda?

- Both target Rust source → PTX → existing CUDA driver runtime.
- Both expose a `#[kernel]`-style annotation surface.
- Both are positioned as alternatives to writing PTX-emitting Rust
  functions (Javelin's pre-0.3 path) or NVRTC-driven C++ kernels.

### Where do they diverge?

- **Maintainer.** NVIDIA's CUDA-Oxide is vendor-led; rust-cuda is
  community-led under the Rust-GPU org.
- **Codegen path.** rust-cuda is built on `rustc_codegen_nvvm`
  (libNVVM-IR). NVIDIA CUDA-Oxide's codegen path is not yet
  publicly documented in enough detail to compare; the 0.2 release
  is the moment to re-audit. Tracked in
  [`docs/CUDA_OXIDE_SWEEP.md`](CUDA_OXIDE_SWEEP.md) and the future
  `01_toolchain_audit.md` revision.
- **Ecosystem coupling.** rust-cuda pulls in `cust`, `cust_core`,
  `cust_raw`, `cuda_std`. NVIDIA CUDA-Oxide's transitive footprint is
  unknown.
- **Risk profile.** rust-cuda has high bus-factor risk (R1) and a
  high-velocity nightly-Rust dependency (R2). NVIDIA-led tooling
  carries the opposite trade: vendor longevity is durable, but vendor
  Rust efforts have a non-trivial cancellation rate.

### Decision framework: pivot to CUDA-Oxide in 0.4?

If NVIDIA ships CUDA-Oxide **0.2 before 0.4 starts**, the 0.4 scope
review asks three questions. Each needs fresh investigation — none of
the answers exist as of 2026-05-25.

1. **Does CUDA-Oxide 0.2 support PTX-text loading via the existing
   `CudaModule::from_ptx` path?** Javelin's host-side loader is
   `cuModuleLoadDataEx`-based and the cudarc track will keep it that
   way. A CUDA-Oxide pivot is only cheap if its PTX output drops into
   that loader unchanged. *(Needs investigation.)*
2. **Does CUDA-Oxide 0.2 support sm_70 (Volta) as a first-class
   target?** sm_70 is Javelin's published floor (see
   `01_toolchain_audit.md` §2). If CUDA-Oxide 0.2 ships sm_75-only or
   Blackwell-first, the pivot waits. *(Needs investigation.)*
3. **Can hand-emit PTX and CUDA-Oxide-emitted PTX coexist in one
   process?** A staged migration — like rust-cuda's Wave A through F
   — needs the answer to be yes. Two PTX sources, one loader, one
   linker, one driver. *(Needs investigation.)*

If all three answer yes, a **0.4 spike wave** (1 week) ports one
kernel — `shmem_sum_kernel` is the canonical low-risk pick — and
benchmarks against whichever of rust-cuda or hand-emit owns it at the
0.3 tag. If the spike's results beat or match the incumbent, the 0.5
scope opens to a broader pivot. 0.4 itself does **not** ship a
CUDA-Oxide-backed build path; the bar for "ship as production" is
higher than for "spike to inform 0.5".

If any answer is no, the framework's verdict is: stay the course on
whichever floor Scenario A/B/C selected.

## Cross-cutting work — fits any scenario

These four items are scenario-independent and are scheduled into 0.4
in whatever slack the scenario-specific waves leave. None of them
requires picking a kernel-source story first.

- **Multi-GPU support.** Single-node multi-card workloads. Per
  `docs/CUDARC_ADOPTION.md` "Out of scope" §, cudarc supports it
  natively; we don't have a use case in 0.3 but a 0.4 user demand
  could materialise from a partner. Scope: one `Engine` per
  device-set, work-stealing scheduler across two-to-eight cards. The
  Tier-2 partition phase is the natural unit of multi-GPU work.
- **Streaming execution.** Today a registered table materialises in
  one batch (see `ROADMAP.md` 0.1.x known limitations). 0.4 lands a
  streaming `register_table_stream` flow that batches in fixed
  chunks; the GROUP BY tier-2 orchestrator already accumulates across
  partitions in a shape that maps well to streaming inputs.
- **DataFusion executor integration.** Javelin as a DataFusion
  physical-plan executor: SQL parsing and logical planning happen in
  DataFusion, the executor offloads to Javelin's PhysicalPlan/Kernel
  pipeline. This is the route to plugging Javelin into a higher-level
  query engine (Ballista, etc.) without owning the SQL frontend.
  Scope-limited to the kernel shapes Javelin already supports —
  unsupported shapes fall back to DataFusion's CPU executor.
- **Async memcpy + pinned host buffers** (carry-over from
  `ROADMAP.md` 0.2 — FFI bound, integration deferred). Lands here
  unconditionally; pairs naturally with streaming execution.

## Out of scope for 0.4 (explicit)

- **SQL feature expansion.** Window functions, CTEs, `LIKE`, `IS NULL`
  propagation through filter/agg, decimal aggregates, date/time types.
  All 0.5 work.
- **Distributed query execution.** Multi-node query planning, network
  shuffle, distributed joins. 1.0 territory; see
  [`docs/PATH_TO_1.0.md`](PATH_TO_1.0.md).
- **Persistent storage / catalog.** Javelin remains a stateless
  engine driven by `register_table*` on every session. 1.0 territory.

## Success criteria

Minimal. By the end of 0.4:

- Whichever path 0.4 ships (A / B / C), every bench in
  [`docs/BENCHMARKS.md`](BENCHMARKS.md) retains or improves its
  post-0.2 numbers. No regression > 5 % on q1 / q4 / q5.
- The cudarc / rust-cuda / hand-emit decision is **baked**: at the
  0.4 tag exactly one is the recommended backend, the others are
  either removed from the tree or live behind a `--features
  legacy-*` flag with a deprecation note. No three-way ambiguity.
- At least one new query workload exists that exercises kernel
  parameterisation flexibility 0.3 could not — either via const
  generics (Scenario A), via cudarc-only ergonomics (Scenario B), or
  via the post-decision dispatch cleanup (Scenario C).

## Carry-over to 0.5+

- **Const generics for hand-emit** (if Scenario B): the const-generic
  work from Scenario A becomes 0.5 scope, gated on the kernel-source
  decision settling.
- **NVIDIA CUDA-Oxide pivot** (if the §4 framework spike succeeds).
  0.5 opens the broader port.
- **DataFusion plug-in** completeness: shape-by-shape parity with
  DataFusion's native executor.
- **Multi-GPU rebench**: ClickBench-style queries on 2× and 4× device
  topologies, published per `docs/COMPETITIVE_BENCHMARKING.md`.
- **SQL surface expansion** (the 0.2 stretch + 0.4 explicit
  non-goals) lands in 0.5 — window functions, decimal, `LIKE`,
  validity propagation through every kernel.
