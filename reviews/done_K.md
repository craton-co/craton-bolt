# Agent K — Final consolidation of `partition_reduce_kernel_*` PTX generators

## Summary

The `partition_reduce_kernel_*` family was already substantially de-duplicated by
prior waves: `partition_reduce_kernel_spill_common.rs` already homed 5 `pub(crate)`
emission helpers (`emit_ptx_header`, `emit_thread_block_ids`, `emit_spin_backoff`,
`emit_spill_bump_with_null_check` / `emit_spill_bump_unchecked`,
`emit_loop_next_done`), each called from both the non-spill and `_with_spill`
emitters across all 11 files.

I diffed all 11 files for further **byte-identical** emission fragments not yet
extracted and found exactly one clean, safe candidate: the **per-block
partition-slice read** (`[start, end)` load from `partition_offsets`). I extracted
that one fragment and left everything else duplicated (see "Deliberately left
duplicated" below).

## Fragment extracted + helper name

New helper added to `src/jit/partition_reduce_kernel_spill_common.rs`:

```rust
pub(crate) fn emit_partition_slice_read(ptx: &mut String) -> BoltResult<()>
```

Emits exactly these 5 PTX lines:

```text
	mul.wide.u32 %rd10, %r0, 4;
	add.s64 %rd11, %rd5, %rd10;
	ld.global.u32 %r10, [%rd11];
	add.s64 %rd12, %rd11, 4;
	ld.global.u32 %r11, [%rd12];
```

A matching exact-byte unit test (`partition_slice_read_emits_expected_bytes`) was
added next to the existing helper tests.

### Call sites replaced (12 total, 2 per file — non-spill + `_with_spill`)

In the 6 files whose offsets pointer is in `%rd5`:
- `partition_reduce_kernel.rs`
- `partition_reduce_kernel_i64.rs`
- `partition_reduce_kernel_minmax.rs`
- `partition_reduce_kernel_minmax_i64.rs`
- `partition_reduce_kernel_minmax_float.rs`
- `partition_reduce_kernel_minmax_float_i64.rs`

Each inline 5-line block was replaced with a single call to
`super::partition_reduce_kernel_spill_common::emit_partition_slice_read(&mut ptx)?;`.
The surrounding blank lines and Rust `//` comments (which are not part of the
emitted PTX string) were left untouched.

## LOC removed

- 12 sites × (5 inline `writeln!` lines → 1 call line) = **48 net source lines
  removed** from the 6 generators.
- Helper + doc + test added to `spill_common.rs`: ~40 lines (one home, vs the 12
  duplicated copies it replaces).

This is intentionally a small, surgical reduction. The bulk of each file (zero-init
phase, probe/CAS/atomic body, export phase, register-decl blocks, entry-param
blocks) is genuinely per-variant and was left alone.

## No emitted PTX byte changed — and why

- The helper emits the **exact same 5 lines, in the same order, with the same
  registers (`%rd10`/`%rd11`/`%rd12`, `%rd5`, `%r10`/`%r11`) and the same `\t`
  indentation** that the inline code emitted. Verified the literal text is
  identical at all 12 sites before replacing.
- Only the 5 emission lines were swapped for the call; every adjacent
  `writeln!(ptx)` blank line was preserved in the caller, so module-level
  whitespace and ordering are unchanged.
- The trailing `// start` / `// end` Rust source comments in the base file (the
  only site that carried them) are source-only and never appeared in the emitted
  string, so dropping them changes no byte of output.
- The new unit test pins the helper's byte output, so any future drift fails fast.

The concatenated output of every one of the 12 affected `compile_*` entry points
is therefore byte-for-byte identical to before. The 40 committed golden snapshots
should pass unchanged.

## Deliberately left duplicated (why)

- **`count` / `count_i64` slice-read** — these emit `add.s64 %rd11, %rd4, %rd10`
  (offsets pointer in `%rd4`, not `%rd5`, because the COUNT kernels have one fewer
  pointer param — no separate values array). Routing them through the helper would
  change a register and drift the bytes. Left as inline `%rd4` emission.
- **`multi` / `multi_i64` slice-read** — use scratch registers `%rd80`/`%rd81`/`%rd82`
  and a *parametric* offsets register (`%rd{rd_poff}`), not the fixed `%rd5`. Not
  byte-identical; left inline.
- **Register-declaration blocks** (`.reg .b64 %rd<N>` etc.) — N differs per variant
  (`<64>` base, `<80>` i64-spill, `<128>` multi), `count` omits `.reg .f64`, float
  variants add `.reg .f32`. Not uniformly identical; left alone.
- **`.visible .entry` param blocks + cvta param-load preamble** — parameterised by
  the per-kernel `entry` string and a variable param count (5/6/7+); not a fixed
  byte sequence shared across files. Left alone.
- **Shared-memory base-address movs** (`mov.u64 %rd0, block_keys_buf;` …) — `count`
  omits the `block_vals_buf` mov, and the spill variants use `_sp`/`_csp`
  buffer-name suffixes. Fragmented; left alone.
- **Zero-init phase, probe/CAS body, atomic-merge, export phase** — encode the
  per-variant key/value widths, the reduction op, the CAS-vs-atomic-add choice, and
  the multi value count. Genuinely per-variant; left alone (matches the existing
  module's documented scope limits).

Per the mandate ("when in doubt, LEAVE IT — a smaller correct dedup beats a
byte-drifting one"), I extracted only the single fragment I could prove is
byte-identical across its callers.

## mod.rs / lib.rs changes needed

**None.** `partition_reduce_kernel_spill_common` is already declared
`pub(crate) mod` in `src/jit/mod.rs`, and the 6 caller files already use the
`super::partition_reduce_kernel_spill_common::...` path for the 5 pre-existing
helpers. The new helper is reachable through the identical path. No new module
file was created. No file outside `src/jit/partition_reduce_kernel*.rs` was
touched.

## Files modified

- `src/jit/partition_reduce_kernel_spill_common.rs` (new helper + test + doc)
- `src/jit/partition_reduce_kernel.rs`
- `src/jit/partition_reduce_kernel_i64.rs`
- `src/jit/partition_reduce_kernel_minmax.rs`
- `src/jit/partition_reduce_kernel_minmax_i64.rs`
- `src/jit/partition_reduce_kernel_minmax_float.rs`
- `src/jit/partition_reduce_kernel_minmax_float_i64.rs`
