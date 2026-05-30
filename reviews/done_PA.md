# Agent PA — `collect_populated_slots_sorted` host slot-walk optimization

**Scope:** edited ONLY `src/exec/groupby_tier2_common.rs`. No other file touched,
no new dependency, no `unsafe`, no `cargo build/check/test` run (verified by
inspection). Output is argued byte-identical to the prior serial implementation.

## What was optimized

`collect_populated_slots_sorted<T>` is the single host slot-walk collector in
this file (confirmed — no sibling collector exists here). It scanned the
~4.2M-slot (`NUM_PARTITIONS=4096 × BLOCK_GROUPS=1024`) tier2 reduce output via a
nested `pid`/`slot` loop, pushing populated `(key, val)` pairs into an
unsized-`Vec` (reallocating as it grew), then `sort_by_key`.

### Serial micro-optimizations (zero-risk, unconditional)

1. **Flattened the nested loop.** Since `idx = pid*block_groups + slot`
   enumerates `0..num_partitions*block_groups` strictly ascending with no gaps,
   the nested walk is equivalent to a single flat pass over `0..n_slots` in
   index order. Replaced with one fused loop (`collect_pairs_serial`).
2. **Pre-sized the output** with `Vec::with_capacity(n_slots)` (upper bound on
   the populated count) — the push loop now never reallocates.
3. **Hoisted bounds checks** by truncating all three input slices to the common
   `n_slots` length once up front (`&buf[..n_slots]`), after which per-element
   indexing carries no bounds check the optimizer can't elide. This also
   preserves the original's panic-on-too-short-buffer behavior identically
   (`[..n_slots]` panics on a short buffer just as `[idx]` would have).

### Parallel scan — ENABLED, threshold = `256 * 1024` slots

`PARALLEL_SLOT_SCAN_THRESHOLD = 256 * 1024`. Production input (~4.2M) is far
above it; small/test inputs stay serial (spawn/join overhead would dominate).

`collect_pairs_parallel` (only above threshold):
- Splits `0..n_slots` into `n_chunks = available_parallelism().min(n_slots)`
  contiguous, non-overlapping, ascending sub-ranges (`div_ceil` chunk length,
  last chunk absorbs the remainder).
- Each chunk scanned on its own `std::thread::scope` thread into a private
  `Vec`, mirroring the serial loop within its sub-range (ascending index order
  preserved locally). Scoped threads borrow the slices directly — no clone, no
  `'static` requirement.
- Joined **in chunk (spawn) order** into a `chunk_outputs: Vec<Vec<_>>`, then
  concatenated in that same order via `append`. Completion order is irrelevant;
  concatenation order is deterministic.

Used only `std::thread` (already in std). Both `std::thread::scope` and
`div_ceil` are already used elsewhere in this crate (e.g. `gpu_join.rs`,
`gpu_compact*.rs`), so no toolchain risk.

### Signature change

Public bound widened from `<T: Copy>` to `<T: Copy + Send>` (required to move
slice refs into worker threads). All four call sites use `u64`/`f64`/`i32`/`i64`
— all `Send` — so this cannot break any caller.

## Identical-output argument

The pre-sort element sequence is the load-bearing invariant:
- **Serial flat pass** yields populated pairs in ascending `idx` order — the
  exact same sequence the original nested loop produced (the nested loop's
  `base+slot` is just `idx` ascending).
- **Parallel path** scans contiguous ascending sub-ranges and concatenates them
  in ascending chunk order ⇒ reproduces that identical ascending-`idx`
  sequence.
- Both feed the **same** `sort_by_key(|(k,_)| *k)`, which is a *stable* sort.
  Identical input order + identical stable sort ⇒ byte-identical output. This
  holds even under (contractually impossible) duplicate keys, since stability
  preserves the identical relative order of equal keys.

Selection is identical too: both paths test `host_set[idx] != 0`, matching the
original (and the COUNT executor's equivalent `== 0 { continue }` guard).

## Tests added (all pure-host, no GPU; run under cuda-stub)

Existing small cases kept. Added in `#[cfg(test)]`:
- `reference_walk` helper: the exact pre-optimization nested walk + sort, used
  as the differential oracle.
- `collect_populated_slots_large_matches_serial_reference`: 1,048,576-slot input
  (1024×1024, above the 256K threshold ⇒ exercises the parallel path), scrambled
  unique keys + ~62.5% populated bitmask, asserts element-for-element equality
  (and length) against `reference_walk`.
- `collect_pairs_parallel_equals_serial_directly`: directly compares the two
  private collectors on `threshold + 12_345` slots, asserting the **pre-sort**
  sequences are identical (the core correctness property).

## Decision

Parallelism enabled (not skipped): the target is a multi-million-slot scan, the
threshold cleanly excludes small inputs, scoped threads add no allocation/clone
cost, and the deterministic chunk-order concatenation makes byte-identical
output provable. Risk is low and the value on the production hotspot is clear.
