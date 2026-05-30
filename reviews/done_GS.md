# Agent GS — full-PTX golden snapshots for `partition_reduce_kernel_*`

## Deliverable

Created **one** new file (and only this file, per the strict rules):

- `C:\Projects\bolt\tests\ptx_golden_partition_snapshots.rs`

No edits to `tests/ptx_golden_tests.rs`, `src/`, `Cargo.toml`, or `lib.rs`. No
cargo run (orchestrator will `cargo insta accept`). `normalize_ptx` +
`parse_reg_suffix` are duplicated verbatim from `ptx_golden_tests.rs` so the
file is self-contained and normalizes identically. Each test emits exactly one
explicitly-named `insta::assert_snapshot!(<stable name>, normalize_ptx(&ptx))`,
so the `.snap` filenames are deterministic. All tests are host-side (no GPU, no
`#[ignore]`).

## Verified facts (not guessed)

- Enum types live once: `MinMaxOp` + `MinMaxDtype` in
  `partition_reduce_kernel_minmax`; `FloatDtype` in
  `partition_reduce_kernel_minmax_float`. The `_i64` variant fns reuse those
  enums (confirmed via their `use` lines). My imports/signatures match.
- `compile_partition_reduce_kernel_multi*` take `n_vals: u32`, valid range
  `1..=MAX_VALS` where `MAX_VALS = 4` — so n=1 and n=2 are both legal.
- `insta = "1"` is a `[dev-dependencies]` entry — the test file compiles.
- No `#[cfg]` gating on any `partition_reduce_kernel_*` module in `mod.rs`; they
  compile under `--no-default-features --features cuda-stub` (same feature set
  the sibling golden suite already builds under).

## Generators covered (entry point -> snapshot name)

Every `pub fn compile_*` in all 11 `partition_reduce_kernel*.rs` files, base and
`_with_spill` twin:

| Entry point | Snapshot name(s) |
|---|---|
| `partition_reduce_kernel::compile_partition_reduce_kernel` | `partition_reduce` |
| `partition_reduce_kernel::compile_partition_reduce_kernel_with_spill` | `partition_reduce_spill` |
| `partition_reduce_kernel_i64::compile_partition_reduce_kernel_i64` | `partition_reduce_i64` |
| `partition_reduce_kernel_i64::compile_partition_reduce_kernel_i64_with_spill` | `partition_reduce_i64_spill` |
| `partition_reduce_kernel_count::compile_partition_reduce_kernel_count` | `partition_reduce_count` |
| `partition_reduce_kernel_count::compile_partition_reduce_kernel_count_with_spill` | `partition_reduce_count_spill` |
| `partition_reduce_kernel_count_i64::compile_partition_reduce_kernel_count_i64` | `partition_reduce_count_i64` |
| `partition_reduce_kernel_count_i64::compile_partition_reduce_kernel_count_i64_with_spill` | `partition_reduce_count_i64_spill` |
| `partition_reduce_kernel_multi::compile_partition_reduce_kernel_multi(n)` | `partition_reduce_multi_n1`, `partition_reduce_multi_n2` |
| `partition_reduce_kernel_multi::compile_partition_reduce_kernel_multi_with_spill(n)` | `partition_reduce_multi_n1_spill`, `partition_reduce_multi_n2_spill` |
| `partition_reduce_kernel_multi_i64::compile_partition_reduce_kernel_multi_i64(n)` | `partition_reduce_multi_i64_n1`, `partition_reduce_multi_i64_n2` |
| `partition_reduce_kernel_multi_i64::compile_partition_reduce_kernel_multi_i64_with_spill(n)` | `partition_reduce_multi_i64_n1_spill`, `partition_reduce_multi_i64_n2_spill` |
| `partition_reduce_kernel_minmax::compile_partition_reduce_kernel_minmax(op,dt)` | `partition_reduce_minmax_min_i32`, `_max_i32`, `_min_i64`, `_max_i64` |
| `partition_reduce_kernel_minmax::compile_partition_reduce_kernel_minmax_with_spill(op,dt)` | `partition_reduce_minmax_min_i32_spill`, `partition_reduce_minmax_max_i64_spill` |
| `partition_reduce_kernel_minmax_i64::compile_partition_reduce_kernel_minmax_i64(op,dt)` | `partition_reduce_minmax_i64_min_i32`, `_max_i32`, `_min_i64`, `_max_i64` |
| `partition_reduce_kernel_minmax_i64::compile_partition_reduce_kernel_minmax_i64_with_spill(op,dt)` | `partition_reduce_minmax_i64_min_i32_spill`, `partition_reduce_minmax_i64_max_i64_spill` |
| `partition_reduce_kernel_minmax_float::compile_partition_reduce_kernel_minmax_float(op,dt)` | `partition_reduce_minmax_float_min_f32`, `_max_f32`, `_min_f64`, `_max_f64` |
| `partition_reduce_kernel_minmax_float::compile_partition_reduce_kernel_minmax_float_with_spill(op,dt)` | `partition_reduce_minmax_float_min_f32_spill`, `partition_reduce_minmax_float_max_f64_spill` |
| `partition_reduce_kernel_minmax_float_i64::compile_partition_reduce_kernel_minmax_float_i64(op,dt)` | `partition_reduce_minmax_float_i64_min_f32`, `_max_f32`, `_min_f64`, `_max_f64` |
| `partition_reduce_kernel_minmax_float_i64::compile_partition_reduce_kernel_minmax_float_i64_with_spill(op,dt)` | `partition_reduce_minmax_float_i64_min_f32_spill`, `partition_reduce_minmax_float_i64_max_f64_spill` |

Total: **22 `compile_*` entry points**, **46 named snapshots** (44 test fns).

### Input-selection notes
- No-arg generators (base SUM, i64 SUM, COUNT, COUNT i64): one snapshot each for
  base and spill.
- `_multi` / `_multi_i64` (take `n_vals`): snapshotted at n=1 (degenerate
  scalar) and n=2 (exercises the per-value loop) for both base and spill.
- `_minmax` / `_minmax_i64` (take op x dtype): all 4 combos
  {Min,Max}x{Int32,Int64} for the base fn; 2 representative combos for spill.
- `_minmax_float` / `_minmax_float_i64` (CAS-loop path, most refactor-sensitive):
  all 4 combos {Min,Max}x{Float32,Float64} for base; 2 for spill.

## Generators NOT directly reachable (and why)

- `partition_reduce_kernel_spill_common` — declared `pub(crate) mod` in
  `src/jit/mod.rs:39`, and exposes only `pub(crate)` **helper** emitters
  (`emit_ptx_header`, `emit_thread_block_ids`, `emit_spin_backoff`,
  `emit_spill_bump_with_null_check`, `emit_spill_bump_unchecked`,
  `emit_loop_next_done`). It has **no `pub fn compile_*` full-kernel generator**,
  and `pub(crate)` items are invisible to an external integration test
  (`tests/` is a separate crate). It cannot be snapshotted directly, and there
  is nothing to snapshot — it produces PTX fragments, not whole kernels. It is
  covered **transitively**: every `*_with_spill` snapshot above embeds its
  emitted bytes verbatim, so a byte change in any spill helper will diff at
  least one spill snapshot. (The file additionally has its own in-`src` unit
  tests asserting exact byte output of each helper.)

## For the dedup agent

After `cargo insta accept` writes `tests/snapshots/*.snap`, the consolidation
refactor must keep all 46 snapshots byte-identical. Any intended PTX change
requires a reviewed re-accept; an unintended drift (reordered instr, renumbered
non-normalized item, shifted offset, dropped barrier/membar) will fail these
tests even where the substring tests in `ptx_golden_tests.rs` still pass — which
is exactly the safety net `reviews/jit.md` §2 calls for as a prerequisite for
the dedup.
