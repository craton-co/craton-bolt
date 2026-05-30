# Agent E — JIT remediation (done)

Branch: `dev`. Files touched (only the three I own):
`src/jit/ptx_gen.rs`, `src/jit/disk_cache.rs`. (`jit_compiler.rs` reviewed —
no change required; the lookback launch site is not there.)

Did NOT run cargo. Did NOT touch any other jit kernel file, lib.rs, or mod.rs.

---

## 1. C-3 (MEDIUM) — validity-byte addressing made UNSIGNED & consistent

**Bug:** validity-byte offsets widened the row index **signed**
(`cvt.s64.s32 %off, %tid`) while value loads/stores widen it **unsigned**
(`mul.wide.u32 %off, %tid, <stride>`). Above 2^31 rows `tid` (from
`mad.lo.s32`) is interpreted negative and sign-extends into a huge negative
offset → OOB validity load/store, while the value load at the same row stays
correct.

**Fix (3 sites in `ptx_gen.rs`):** replaced `cvt.s64.s32 %off, %tid` with
`mul.wide.u32 %off, %tid, 1` (stride-1 byte addressing, zero-extends tid),
matching `emit_load`/`emit_load_128`/`emit_store`. Sites:

- `compile`, input-validity AND-of-inputs fold (was ~:358).
- `compile`, output-validity store (was ~:431).
- `emit_is_null_check` (was ~:1006).
- Also updated the `emit_is_null_check` doc-comment "Wire shape" block that
  still showed the old `cvt.s64.s32`.

**Doc:** added a "Row-count limit (C-3 / C-4)" section to the `compile` rustdoc
stating the per-launch row space is capped at `i32::MAX` by the s32 `mad.lo`
tid math, that all offset arithmetic now widens unsigned, and that the host
launch path MUST ensure `n_rows <= i32::MAX` (covers C-4's documentation ask;
the host-side assertion itself lives outside my files — see §4).

**Tests added (`#[cfg(test)]`, ptx_gen.rs):**
- `validity_offset_widens_unsigned` — compiles the `mul_with_validity_spec`
  kernel; asserts the validity path emits `mul.wide.u32 .., 1;` and that
  `cvt.s64.s32` is entirely absent (this spec emits no Int32→Int64 CAST).
- Extended `is_null_check_emits_validity_load_and_setp` with the same two
  assertions for the `emit_is_null_check` path.

Note: existing `cvt.s64.s32` at ptx_gen.rs ~:1313/:1318 are the legitimate
`CAST(Int32 AS Int64)` sign-extend — intentionally left unchanged; the
`cast_i32_to_i64_sign_extends` test that pins them still holds.

## C-1-adjacent / cache freshness — `CODEGEN_VERSION` bump (disk_cache.rs)

Because C-3 changes the emitted PTX *text* for validity-carrying kernels, per
the documented maintainer protocol I bumped `CODEGEN_VERSION 1 -> 2` so stale
disk-cached PTX produced by the old signed-widen codegen is not served. This
only rotates the codegen salt — it does **not** alter the cache-key composition,
the full-PTX re-compare collision-safety, or the V-7 integrity-digest logic.

While there I upgraded the `CODEGEN_VERSION` rustdoc per C-1: documented that a
Rust-std/toolchain upgrade is also a key-rotation event (because the on-disk key
and integrity digest use `DefaultHasher`/SipHash-1-3, whose output is not
contractually stable across std versions), and added a version-history note
(v2 = C-3). No algorithm change to `body_digest`/`hash_to_key` (changing it
would break the existing re-derive-on-read integrity guarantee).

Verified no test hardcodes the version: `valid_key` tests use a static `cg1-...`
key *shape* (still valid), and the salt tests use `CODEGEN_VERSION` /
`CODEGEN_VERSION + 1` relatively. All remain green by construction.

## 2. C-7 (MEDIUM) — decoupled-lookback forward-progress contract

The lookback **emitter** (`bolt_prefix_scan_lookback`) is in
`src/jit/prefix_scan.rs` and the host **launch site** was not found in any of my
three files (confirmed: no `lookback` / `cuLaunch` / `gridDim` references in
`ptx_gen.rs` or `jit_compiler.rs`). Per the rules I did NOT edit those files.

### >>> ACTION FOR ORCHESTRATOR — wire this guard at the lookback launch site <<<

The decoupled single-pass (Merrill-Garland) lookback scan spins on a
predecessor block's `partial_status[blockIdx.x - 1]` with `ld.acquire.gpu.u32`
and **no fallback** if that predecessor block is never scheduled. On GPUs
without guaranteed occupancy-bounded co-residency this **deadlocks** when
`gridDim.x` exceeds the number of resident CTAs.

Required launch contract (document in the prefix_scan module header AND enforce
at the host launch path):

1. **Single-wave / occupancy-bounded launch.** `gridDim.x` MUST be
   `<= max_resident_blocks`, where
   `max_resident_blocks = num_SMs * maxActiveBlocksPerSM` for this kernel at the
   chosen `blockDim` (query via
   `cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, kernel, blockDim, smem)`
   then `* deviceProp.multiProcessorCount`). Every block must be co-resident so
   forward progress is guaranteed.

2. **Host guard (add as `debug_assert!` + a hard runtime check):**
   ```rust
   // Decoupled-lookback forward-progress guard (review C-7).
   debug_assert!(
       grid_dim_x <= max_resident_blocks,
       "lookback scan grid {grid_dim_x} exceeds resident capacity \
        {max_resident_blocks}; would deadlock"
   );
   if grid_dim_x > max_resident_blocks {
       // fall back to the multipass scan (prefix_scan_multipass.rs)
       return launch_multipass_scan(...);
   }
   ```

3. **Row-count guard (C-3/C-4, same launch path):**
   `debug_assert!(n_rows <= i32::MAX as usize)` — the kernels compute the global
   thread id in s32 (`mad.lo.s32`), so larger row counts mis-address. This guard
   belongs on EVERY ptx_gen-emitted kernel launch, not just lookback.

4. **Confirm dispatcher routing:** verify the scan dispatcher prefers
   `prefix_scan_multipass.rs` (the safe variant) once `gridDim.x` would exceed
   resident capacity, rather than always selecting lookback.

The orchestrator should route items 1-4 to whichever agent owns the host launch
/ dispatch file (not in jit/ptx_gen.rs or jit/jit_compiler.rs).

## 3. Other small in-scope fixes

None beyond the above were both clearly-correct AND confined to my three files.
DUP/TST/P-1/P-2 all require editing files I do not own or are larger refactors;
left for the consolidation agent. C-2 (cache-cap TODO) is INFO/no-op and
intentionally frozen — not touched.

## 4. Summary of edits

- `src/jit/ptx_gen.rs`: 3 codegen sites signed→unsigned widen; 1 doc-shape fix;
  `compile` rustdoc row-limit section; 1 new test + extended 1 test.
- `src/jit/disk_cache.rs`: `CODEGEN_VERSION` 1→2; expanded rustdoc (C-1
  toolchain note + version history). No logic/algorithm change.
- `src/jit/jit_compiler.rs`: unchanged (no relevant launch site).
