# Agent W — cross-file wiring (done)

Branch `dev`. Only the three permitted files were edited:
`src/exec/engine.rs`, `src/jit/prefix_scan.rs`, `docs/JIT_PIPELINE.md`.
No `lib.rs`/`mod.rs`, no cuda/groupby/aggregate/test files, no `cargo` run
(verified by inspection against the real code + done_G/C/E specs).

---

## 1. (done_G F11) GpuCapacity decline → transparent host re-run of the projection

**Site:** `Engine::execute` dispatch arm `PhysicalPlan::Projection` (engine.rs
~:2364) + new method `Engine::execute_projection_host_fallback` + new free
helper `passthrough_output_sources`.

**Wiring (mirrors join.rs).** `execute_projection` now runs inside a `match`:
`Err(BoltError::GpuCapacity(reason))` is caught, the
`HostFallbacksTotal` metric is bumped, a debug line is logged, and the
projection re-runs on the host — exactly the `Err(GpuCapacity) → host` mapping
`try_gpu_inner_join`/`try_gpu_outer_join` already apply to the join gates. Every
*other* error still `?`-propagates. `BoltError::GpuCapacity(String)` is a typed
tuple variant (error.rs:115), pattern-matched, not string-parsed.

**Host re-run mechanism.** There is no host op-VM interpreter for the fused
projection IR (`KernelSpec` is a pure GPU register-VM with no `Expr`
backpointer; the planner routes anything richer than the GPU IR can express
through `PhysicalPlan::Project`/`StringProject`). The columns that actually trip
agent-G's decline — Date32 / Timestamp / Decimal128 upload+gather — reach
`execute_projection` only as **identity-passthrough** projections
(`SELECT date_col, dec_col, … FROM t`). So the fallback:

- `passthrough_output_sources(kernel)` traces the `ops`: with no predicate and
  ops that are exactly `LoadColumn`→`Store` (and the 128-bit
  `LoadColumn128`→`Store128`) pairs, it returns `Some(out_src)` mapping each
  output column to the input column ordinal that feeds it (handles output
  reordering, e.g. `SELECT b, a`). Any predicate / compute / cast / select /
  128-bit-arithmetic op ⇒ `None`.
- On `Some`, the method materializes the table on the host and picks the mapped
  column per output (casting to the declared output arrow dtype — a no-op for a
  true passthrough, but also coerces a dict-encoded source to its logical
  dtype), then rebuilds the result batch.
- On `None` (non-passthrough), it **re-raises** the `GpuCapacity` decline rather
  than risk a silently-wrong host result. This is the one case not fully wired
  to a host result — see "Couldn't fully wire" below.

**Tests (host-pure, no CUDA — `mod tests` in engine.rs):**
`passthrough_maps_outputs_to_inputs`, `passthrough_respects_output_reordering`,
`passthrough_handles_decimal128_pair`, `non_passthrough_compute_returns_none`,
`predicate_kernel_is_not_passthrough`. They exercise the load-bearing
GPU-free decision (`passthrough_output_sources`); the helper was extracted as a
free fn precisely so it is testable without constructing an `Engine` (which
needs a CUDA context).

## 2. (done_C F-3) `string_length_column` host fallback emits SQL NULL, not 0

**Site:** `Engine::string_length_column`, the `None`-layout host-fallback block
(engine.rs ~:3240), DictUtf8 branch.

The DictUtf8 NULL branch previously pushed `0` for NULL rows. It now produces
`Vec<Option<i64>>`: a NULL input row → `None` (SQL NULL, validity-carrying),
distinct from `LENGTH('') = 0`; valid rows → `Some(table[key+1])`. The
non-DictUtf8 host-gather branch (`host_gather_lengths`, no per-row validity at
that layer) maps each length through `Some`. The result is built with
`Int64Array::from(Vec<Option<i64>>)`, which carries the validity bitmap, so the
engine/GPU LENGTH path is now NULL-consistent with `string_ops::length` (agent
C's F-3). No host-pure unit test added here (the method calls `ensure_gpu_table`
⇒ needs a GPU); agent C already covers the pure NULL logic in `string_ops.rs`.

## 3. (done_E §2 / C-7) decoupled-lookback forward-progress contract

`src/jit/prefix_scan.rs` is a **PTX emitter only** — it has no `cuLaunch` /
host launch site (confirmed: the launch lives in
`gpu_compact::prefix_scan_mask_lookback`, outside my edit boundary). So within
this file:

**(a) Emitter doc-comment** — added a "Host launch contract — forward-progress /
no-deadlock (review C-7 / E §2)" section to `compile_prefix_scan_kernel_lookback`
stating the required single-wave occupancy bound
(`gridDim.x <= num_SMs * maxActiveBlocksPerSM`, every block co-resident so a
successor never spins on an unscheduled predecessor), the `n_rows <= i32::MAX`
s32-tid addressing assumption, and the `n_rows < (1<<30)` value-budget bound,
with the multipass fallback named explicitly.

**(b) Guard predicate** — added `pub const LOOKBACK_MAX_ROWS: u32 = 1 << 30` and
`pub fn lookback_launch_is_safe(grid_dim_x, max_resident_blocks, n_rows) -> bool`
encoding all three bounds (co-residency incl. `max_resident_blocks > 0`,
`n_rows <= i32::MAX`, `n_rows < LOOKBACK_MAX_ROWS`). Its rustdoc shows the exact
`debug_assert!` + `if !… { return prefix_scan_multipass(…) }` the launch site
must apply. **Tests (host-pure):** `lookback_guard_enforces_coresidency`,
`lookback_guard_enforces_row_bounds`.

## 4. docs/JIT_PIPELINE.md:40 stale-line fix

Replaced the stale "A `KernelSpec`-keyed cache … is still a future
optimization" sentence with reality: as of v0.7 the per-`Engine` `module_cache`
plus the process-wide `src/exec/module_cache.rs` key on the planner
`KernelSpec`, so a warm hit is a sub-µs `Arc<CudaModuleInner>` clone that skips
`ptx_gen::compile`, the disk cache, and the driver load — verified against
`Engine::get_or_build_module` (engine.rs:1002, "skips every layer below") and
`module_cache.rs`.

---

## Couldn't fully wire (and why)

- **3(b) host launch guard at the actual launch site.** The lookback launch is
  in `gpu_compact.rs` (`prefix_scan_mask_lookback`), which I'm not permitted to
  edit. I provided the tested predicate `prefix_scan::lookback_launch_is_safe`
  + `LOOKBACK_MAX_ROWS` + the doc'd usage; the launch-site owner must add the
  `debug_assert!` + multipass-fallback branch and confirm the dispatcher prefers
  multipass when `gridDim.x` would exceed resident capacity (done_E §2 items
  1–4). Route to the `gpu_compact.rs` owner.
- **1 non-passthrough temporal/decimal projection.** If a *computed* op over a
  declined temporal/decimal column (or a predicate-bearing such projection) ever
  reaches `execute_projection` and the GPU declines it, the fallback re-raises
  the `GpuCapacity` decline instead of host-evaluating — there is no host
  interpreter for the projection register-VM and silently mis-evaluating would
  be worse. In practice the planner does not emit GPU `Projection` kernels for
  those shapes (rich temporal/decimal expressions go through the host
  `PhysicalPlan::Project` path), so this branch is not expected to be reachable;
  it is a correctness-preserving guard, not a silent wrong answer.

## Constraints honored
- Only the 3 permitted files edited.
- No `cargo` build/check/test run; changes verified by inspection.
- Every behavior change that is host-reachable has a pure `#[cfg(test)]` unit
  test (no `#[ignore]`/GPU): the projection passthrough detector (5 tests) and
  the lookback launch guard (2 tests).
