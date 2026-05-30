# Code Review — `src/jit/` (PTX/CUDA codegen + JIT compiler + caches)

Reviewer: senior Rust/GPU-compiler audit
Scope: all 39 files under `C:\Projects\bolt\src\jit\` (≈39,500 LOC).
Date: 2026-05-30.

Files read in full: `mod.rs`, `jit_compiler.rs`, `disk_cache.rs`, `ptx_gen.rs`
(core sections + emitters), `prefix_scan.rs`, `sort_kernel.rs`,
`partition_reduce_kernel_spill_common.rs`, `shmem_sum_kernel.rs`,
`partition_reduce_kernel_minmax_float.rs`, `hash_join_kernel.rs` (build/probe),
`string_kernel.rs` (write pass), `scan_kernel.rs` (binary ops), `float_atomics.rs`.
Diffed/skimmed: all `partition_reduce_kernel_*` variants, `sort_kernel_radix.rs`,
`hash_kernels.rs`, `valid_flag*`, `scatter*`, `shmem_*`, `date_scalar.rs`,
`decimal_agg.rs`, `prefix_scan_multipass.rs`.

Overall assessment: **the JIT layer is mature and unusually well-documented.**
The cache machinery (in-process LRU + disk cache) is the strongest part — the
concurrency model, collision-safety, and path-traversal/integrity hardening are
correct and well-tested. The codegen emitters are mostly correct; the findings
below are a small number of latent correctness issues, one architectural
duplication problem, and a significant test-adequacy gap (no true golden PTX
snapshots).

---

## 1. CODE REVIEW

### Cache correctness (jit_compiler.rs / disk_cache.rs)

**No cache-collision or poisoning bugs found.** Specifically verified:

- **In-process PTX cache key** (`jit_compiler.rs:439-448`): 128-bit
  `(hi, lo)` from two domain-separated `DefaultHasher`s, AND the full PTX
  string is re-compared on every hit (`get_and_touch`, line 355). A hash
  collision is therefore correctness-safe — it routes to `Slot::Collision` →
  uncached load (lines 519-525). Two different kernels cannot collide into a
  wrong module. **Correct.**
- **Concurrency** (`from_ptx_with`, lines 479-548): the `Arc<OnceCell>`-per-key
  pattern releases the lock before the PTXAS compile; racing threads on the same
  PTX block in `get_or_try_init` and share one compile. A failed compile leaves
  the cell empty (not poisoned) so the next caller retries. Tests
  `from_ptx_compiles_once_under_contention` and
  `from_ptx_failed_compile_does_not_poison_cell` pin this. **Correct.**
- **Disk cache** (`disk_cache.rs`): tempfile-then-`rename` for atomicity;
  `valid_key` (line 739) blocks path traversal (`..`, `/`, `\`, `:`, NUL);
  V-7 integrity header (`#bolt-ptx-cache v1 <digest>`) is verified on read so a
  corrupt/partial/tampered body is treated as a miss; `0o700` (Unix) / `icacls`
  (Windows) dir hardening. The codegen-version salt (`codegen_salt`, line 189)
  + crate version + optional fingerprint correctly rotate the disk key so a
  stale binary's PTX cannot be served (JIT-M1). **Concurrent-safe and
  poisoning-safe.**

**Finding C-1 — LOW — disk digest/key uses `DefaultHasher` (non-portable, weak).**
`disk_cache.rs:772` (`body_digest`) and `hash_to_key`/the upstream spec hash use
`std::collections::hash_map::DefaultHasher` (SipHash-1-3). This is fine *within*
a single binary (re-derived on read, salted by crate version), and the file
documents it is not a cryptographic MAC. But two caveats deserve a code comment
upgrade to an assertion-level guarantee: (a) `DefaultHasher`'s output is **not
contractually stable across Rust std versions**, so a toolchain bump could
silently change every disk key — today this is masked by the crate-version salt,
but only because releases bump the version. (b) The integrity digest is
trivially forgeable by anyone who can write the cache dir. *Suggested fix:* note
explicitly in `CODEGEN_VERSION` docs that a Rust-std upgrade is also a
key-rotation event, or switch the disk digest to a fixed algorithm (e.g. a
vendored FNV-1a/xxhash) so it is reproducible and toolchain-independent.

**Finding C-2 — INFO — `ptx_cache_cap()` TODO is a no-op risk, not a bug.**
`jit_compiler.rs:143` carries `TODO(cache-cap): re-read on each insert`. The cap
is frozen via `OnceLock` for the process lifetime — this is intentional and
documented; the TODO should be closed or downgraded since re-reading would
reintroduce a runaway-resize hazard.

### Codegen correctness (ptx_gen.rs and emitters)

**Finding C-3 — MEDIUM — signed vs unsigned row-index widening is inconsistent.**
Value-column addressing widens the row index **unsigned** via `mul.wide.u32`
(`ptx_gen.rs:1150`, `:1180`, `:578`, `:614`), but validity-byte addressing
widens it **signed** via `cvt.s64.s32` (`ptx_gen.rs:358`, `:431`, `:1006`; same
pattern in `emit_is_null_check`). For `n_rows ≤ 2^31` the two are numerically
identical. Above that, `tid` (produced by `mad.lo.s32`, line 256) is interpreted
as negative and `cvt.s64.s32` sign-extends it into a huge negative offset →
out-of-bounds validity load/store, while the value load at the same row stays
correct. The whole kernel already implicitly caps grid size at ~2^31 via the
s32 tid math, so this is latent rather than live, but it is an internal
inconsistency that will bite the day someone lifts the row count to i64.
*Suggested fix:* use `mul.wide.u32 %off, %tid, 1` (or `cvt.u64.u32`) for the
validity offset to match the unsigned value path, and document the ≤2^31
row-count assumption in `compile`.

**Finding C-4 — LOW — `mad.lo.s32` global thread id caps usable rows at 2^31.**
`ptx_gen.rs:256` computes `tid = ctaid*ntid + tid.x` in s32. Combined with C-3,
the engine cannot address >2^31 rows in a single launch through these kernels.
This is almost certainly fine for current workloads but should be an explicit,
asserted host-side precondition rather than an emergent property. *Fix:* assert
`n_rows <= i32::MAX` on the host launch path, or migrate the index math to u32
wide ops end-to-end.

**Finding C-5 — INFO — Mul128 truncating multiply is correct.** `emit_mul_128`
(`ptx_gen.rs:735-791`) implements the 3-partial-product schoolbook truncating
multiply with plain `add.u64` on the high half (bits ≥128 discarded), matching
`i128::wrapping_mul`. Verified against the documented algebra. The Add/Sub
carry-chain (`add.cc.u64`/`addc.u64`, `sub.cc.u64`/`subc.u64`) and the Cmp128
signed-high/unsigned-low decomposition are all correct.

**Finding C-6 — INFO — Float MIN/MAX CAS loop is NaN-safe.**
`partition_reduce_kernel_minmax_float.rs:437-512` does the decision with
`setp.lt/gt.fXX` (IEEE: NaN comparisons false → candidate propagates, matching
CPU ref) but the no-op-skip and CAS-won checks with **bit** equality
(`setp.eq.b32/b64`), correctly avoiding the NaN==NaN trap. Slots are
identity-initialised to `±inf` (line 142, not zero), so the empty-slot-wins-MIN
bug is avoided. **Correct.**

### Race conditions / synchronization in shmem reductions and scans

All shared-memory barriers I inspected are correctly placed **outside**
divergent control flow:

- **Bitonic shmem sort** (`sort_kernel.rs:1160-1224`): `bar.sync 0` sits at the
  `SH_S{stage}_T{substage}_AFTER:` label that the skip-branch jumps *to*
  (lines 1181, 1200, 1217-1218), so every lane reaches the barrier each
  substage. The stable index-tiebreak and padded-row routing are correct.
  **No barrier divergence.**
- **Hillis-Steele scan** (`prefix_scan.rs:296-344`): all lanes store
  `own+neighbor` unconditionally to the pong buffer before each `bar.sync`
  (line 330-331); ping/pong pointer swap is per-lane register `mov`. Correct.
- **Decoupled-lookback scan** (`prefix_scan.rs:1055-1200`): publisher does
  `st.global` + `membar.gl` (lines 1090-1093, 1164-1165); reader spins with
  `ld.acquire.gpu.u32` (line 1117). The acquire/release pairing is correct.
  Thread-0-only lookback while peers wait at `BROADCAST`'s `bar.sync` (line
  1182) is safe.

**Finding C-7 — MEDIUM (design constraint, document loudly) — decoupled-lookback
forward-progress.** `prefix_scan.rs` lookback (`bolt_prefix_scan_lookback`) spins
on a predecessor block's status with no fallback if that block is never
scheduled. On GPUs without guaranteed occupancy-bounded co-residency this can
**deadlock** if `gridDim.x` exceeds resident-CTA capacity. The standard
single-pass-scan requirement (launch ≤ max resident blocks, or use a
dynamically-assigned block id via `atom.global.add` on a global ctr) is not
documented in the kernel header and I did not find a host-side launch guard
referenced here. *Fix:* document the occupancy-bounded launch contract in the
module header and add a host-side assertion; or fall back to the multipass scan
above a threshold. (The `prefix_scan_multipass.rs` variant exists and is the
safe path — confirm the dispatcher prefers it for large grids.)

### Stubs / unimplemented / dead code / TODOs

All located markers (verified by line):

- **`string_kernel.rs:125,151`** — `CONCAT` GPU two-pass codegen **not
  implemented**; rejected with `"not yet implemented"`. `string_kernel.rs:1034`
  — write-pass `_ =>` arm returns "write pass not implemented" for kinds other
  than UPPER/LOWER/SUBSTRING/TRIM. These are deferred features rejected at plan
  time, not dead code. **OK but track as feature gaps.**
- **Decimal128 GPU lowering deferred** — consistent `"Decimal128 not yet
  lowered to GPU"` rejections in `ptx_gen.rs` (108, 1083, 1206, 1293, 1357,
  1567, 1601), `scan_kernel.rs` (656, 732, 814, 875, 1271, 1305),
  `prefix_scan.rs:1374`, `sort_kernel_radix.rs:235`. Note: `ptx_gen` *does* lower
  Decimal128 Add/Sub/Mul/Cmp via the i128 dual-register path — these rejections
  are for the remaining ops (Div, CAST, etc.). Messaging is consistent.
- **MIN/MAX-over-float in GROUP BY** rejected in `hash_kernels.rs:1511` and
  `valid_flag_kernels.rs:657` — but a float MIN/MAX path *does* exist
  (`float_atomics.rs`, `partition_reduce_kernel_minmax_float*`). Confirm the
  dispatcher routes float MIN/MAX to those and these rejections are only the
  fallback-hash path. **Possible dead rejection / routing smell — verify.**
- **`unreachable!()`** in `date_scalar.rs` (161,209,512,528,565,604),
  `ptx_gen.rs:1462`, `scan_kernel.rs:959`, `sort_kernel.rs:393,797`,
  `sort_kernel_radix.rs:397` — all are genuinely guarded by an enclosing match
  that already narrowed the variant. **Not reachable; acceptable.**
- **`TODO(perf): exponential back-off`** appears in **11 files** (`float_atomics.rs:88`,
  `partition_reduce_kernel*.rs`, `valid_flag_kernels.rs:142`,
  `hash_kernels.rs:168`). All use a fixed `SPIN_BACKOFF_NS = 32` nanosleep. See
  Perf below.
- **`hash_join_kernel.rs:685,1328`** — `TODO(perf): ld.global.nc.v2.u64` vectorised
  probe load. Perf-only.
- **`ptx_gen.rs:1711`** — `TODO(orchestrator): golden test update`. See Tests.

No `panic!`/`todo!`/`unimplemented!`/`unwrap()`-on-fallible-codegen found in
production paths. Error handling routes through `BoltResult` cleanly.

### Massive near-duplication across `partition_reduce_kernel_*` (quantified)

There are **11 sibling files** (`partition_reduce_kernel` + 10 variants),
totalling **~7,600 LOC**, each emitting ~220–262 `writeln!` lines of PTX for one
`(key-type, value-type, op)` tuple:

| Axis | Variants |
|------|----------|
| key width | i32 (base) vs i64 (`_i64`) |
| op | SUM (base), COUNT, MIN/MAX (int), MIN/MAX (float) |
| arity | single-value vs `_multi` (parametric value count) |

Measured textual overlap (identical `writeln!` lines, ignoring the
`with_spill` twins each file also carries):

- base SUM vs `_i64`: **159 of 262 emit lines identical (~61%)**.
- `minmax` vs `minmax_i64`: **~70% identical**.
- `minmax` vs `minmax_float`: **~70% identical**.

The only shared code today is `partition_reduce_kernel_spill_common.rs` — **6
trivial helpers** (header, thread-id movs, spill-bump, loop epilogue, spin
backoff) covering perhaps 15 lines per kernel. The bulk (zero-init phase, probe
loop, CAS/atomic body, export phase) is hand-duplicated per file, and each file
additionally duplicates its own non-spill vs `_with_spill` emitter (~2× within
the file).

**Recommended consolidation strategy (MEDIUM priority):**
Replace the 11 files with **one parameterised emitter** driven by a small spec:

```rust
struct PartitionReduceSpec {
    key: KeyType,        // I32 | I64
    value: ValueType,    // I32 | I64 | F32 | F64
    op: ReduceKind,      // Sum | Count | Min | Max
    n_values: u32,       // 1 for scalar, N for multi
    with_spill: bool,
}
fn compile_partition_reduce(spec: &PartitionReduceSpec) -> BoltResult<String>
```

Factor the four phases (zero-init / grid-stride probe+accumulate / atomic-merge /
spill) into helpers that take the type widths and the per-op atomic mnemonic
(`atom.shared.add.{u32,u64,f64}` vs the `atom.shared.cas` retry loop for float
MIN/MAX). The existing substring golden tests (see below) make this refactor
*safe to verify* only if you first add full-PTX snapshots (otherwise the
substring tests will pass even if registers/offsets drift). Net: ~7,600 LOC →
~1,200 LOC, one place to fix the back-off TODO, one place to fix C-3-style
addressing bugs.

The same pattern (i32 vs i64 twins) recurs in `partition_kernel.rs` /
`partition_kernel_i64.rs`, `scatter_kernel.rs` / `scatter_kernel_i64.rs`, and
`sort_kernel.rs` / `sort_kernel_radix.rs` — a generic "width" parameter would
collapse each pair.

### Performance suggestions

- **P-1 (HIGH value, low effort):** the fixed `nanosleep.u32 32` back-off in all
  contended CAS/probe loops leaves throughput on the table under hot-key skew.
  Implement the documented exponential back-off **once** (in the consolidated
  emitter from above), capped at 256 ns, using one loop-carried register.
- **P-2:** hash-join probe loads one slot per `ld.global.nc.u64`; the documented
  `ld.global.nc.v2.u64` vectorised pair load (`hash_join_kernel.rs:685,1328`)
  would halve probe memory transactions for the SoA layout.
- **P-3:** `shmem_sum_kernel` uses `atom.shared.add.f64` per row. For high-skew
  keys a warp-aggregated pre-reduction (`__shfl`-based, or `match.any` slot
  voting on sm_70) before the shared atomic would cut shared-atomic traffic
  further. Marked as future tier in the module docs.
- **P-4:** `emit_is_null_check` / validity loads issue one `ld.global.nc.u8` per
  flagged input per row. When multiple inputs are flagged, the AND-of-validity
  fold (`ptx_gen.rs:344-370`) does N separate byte loads; packing validity into
  a bitmap word and a single load would reduce memory ops (already a noted
  Stage-C follow-up).

---

## 2. TEST ADEQUACY

**Headline gap: there are NO true golden PTX snapshots.** The suite
`tests/ptx_golden_tests.rs` (1,758 LOC, 49 test fns) is **202 `.contains()`
substring assertions** plus **only 5 `insta::assert_snapshot!` calls**, and
there are **no committed `.snap` files** under `tests/snapshots/`. In-file unit
tests across `src/jit/*.rs` add ~483 `#[test]` cases, but these too are
overwhelmingly substring/structural checks (e.g. "contains `bar.sync 0`", "≥2
`membar.gl`", reg-count sanity).

Consequences:
- A refactor that reorders instructions, renumbers registers, or shifts a byte
  offset can pass every substring test while emitting subtly wrong PTX. This is
  exactly the failure mode that makes the partition-reduce consolidation
  (above) risky.
- Off-by-one regressions in scan/sort offsets, or a flipped `add.cc`/`addc`
  carry, would not be caught unless the specific mnemonic happens to be asserted.

**Coverage estimate:** behavioral/structural coverage of emitters is **good**
(every `compile_*` entry point named in the golden file is exercised — see list
below), but **exact-output coverage is ~0%** and **on-GPU numerical coverage is
limited** (only `tests/sort_e2e.rs` runs an end-to-end sort; most kernels have no
device-execution test in this layer).

Golden-tested entry points (from `tests/ptx_golden_tests.rs`): `compile_ptx`,
`compile_build_kernel`, `compile_build_aos_kernel`, `compile_build_collision_kernel`,
`compile_unmatched_build_kernel`, `compile_probe_kernel`, `compile_probe_kernel_tiled`,
`compile_probe_aos_kernel`, `compile_probe_collision_kernel`,
`compile_groupby_agg_kernel`, `compile_groupby_float_atomic_kernel`,
`compile_groupby_keys_kernel`, `compile_reduction_kernel`,
`compile_prefix_scan_kernel`, `_blelloch`, `_lookback`,
`compile_partition_reduce_kernel` (+ `_i64`, `_count`, `_count_i64`, `_minmax`,
`_minmax_i64`, `_minmax_float`, `_minmax_float_i64`, `_multi`, `_multi_i64`),
`compile_sort_kernel_spec`, `compile_scatter_*`, `compile_length_gather_kernel`,
`compile_varwidth_len_pass`, `compile_varwidth_write_pass`.

**Kernel variants/paths that appear under-tested or untested:**
- `sort_kernel_radix.rs` (radix LSD sort) — present in golden list only via
  `compile_sort_kernel_spec`? Confirm the radix path itself has a snapshot; its
  digit-histogram + scatter is the most off-by-one-prone code in the layer.
- `decimal_agg.rs` (`SUM(Decimal128)` two-stage i128 block reduce) — not in the
  golden entry-point list; verify it has dedicated tests.
- `date_scalar.rs` `EXTRACT`/`DATE_TRUNC` — has host-reference math with 6
  `unreachable!`; needs value-equivalence tests (GPU vs host ref) per field/unit.
- The `_with_spill` twin of **every** partition-reduce kernel — confirm both the
  spill and non-spill emitter of each file are snapshotted (the golden list
  names the base fn, not obviously the spill variant).
- `prefix_scan_multipass.rs` — the safe large-grid fallback; ensure it is tested
  *and* that the dispatcher's grid-size threshold is tested (relevant to C-7).
- Decimal128 `Cmp128` six comparison shapes (eq/ne/lt/gt/le/ge) — assert each
  produces the documented signed-high/unsigned-low sequence.

**Specific tests to add:**
1. **Full-PTX `insta` snapshots** for every `compile_*` entry point (both spill
   and non-spill), with a `normalize_ptx` pass (already present at
   `tests/ptx_golden_tests.rs:85`) to absorb only intended churn. This is the
   single highest-leverage addition and a prerequisite for the dedup refactor.
2. **Device numeric equivalence tests** (gated on a real GPU / `cuda` feature):
   for each reduce/scan/sort kernel, run on random input and assert against a
   CPU reference — covers carry-chain, NaN, ±inf-identity, partial-block tail,
   and validity-AND correctness that substring tests cannot.
3. **A regression test for C-3**: a kernel with validity over a row count that
   exercises the high bit of `tid` (or simply assert the validity offset uses an
   unsigned widen) to lock down the signed/unsigned fix.
4. **A lookback deadlock/forward-progress test** (C-7): launch
   `bolt_prefix_scan_lookback` with `gridDim.x` near/over resident capacity and
   assert it completes (or that the dispatcher refuses and falls back).
5. **Cmp128 per-op golden snapshots** (6 ops × the documented instruction
   sequence).

---

## 3. FEATURES / DIRECTIONS

1. **Consolidate the `partition_reduce_kernel_*` family** into one
   spec-parameterised emitter (see §1). Highest structural ROI in the module.
2. **Finish the deferred lowerings** behind clean rejections: GPU `CONCAT`
   (`string_kernel`), Decimal128 Div/CAST (`ptx_gen`), float MIN/MAX in the
   group-by hash path (clarify routing vs the dedicated float-atomic kernel).
3. **Wire up `BOLT_CODEGEN_FINGERPRINT`** (already consumed in
   `disk_cache.rs:153` via `option_env!`) from `build.rs` as a digest over the
   codegen module tree. This turns the manual `CODEGEN_VERSION` bump into a
   defense-in-depth backstop rather than the load-bearing freshness guard, and
   resolves the C-1 toolchain-stability concern automatically.
4. **Exponential back-off + warp-aggregation** for contended atomics (P-1, P-3).
5. **Unify i32/i64 twins** (`partition_kernel`, `scatter_kernel`, sort) under a
   width parameter — same template strategy, smaller blast radius.
6. **A PTX self-validation pass**: optionally run emitted PTX through
   `cuModuleLoadDataEx` in CI on a GPU runner (or `ptxas --compile-only`) so
   syntactic/ISA regressions are caught even where snapshots only pin substrings.
7. **Lift the ≤2^31-row launch limit** (C-3/C-4) to true 64-bit grid addressing
   if/when single-launch row counts can exceed that, with an explicit host
   assertion in the interim.

---

## Severity index

| ID  | Sev    | File:line | Summary |
|-----|--------|-----------|---------|
| C-3 | MEDIUM | ptx_gen.rs:358,431,1006 | Validity offset uses signed `cvt.s64.s32` while value path uses unsigned `mul.wide.u32`; breaks >2^31 rows |
| C-7 | MEDIUM | prefix_scan.rs:1095-1146 | Decoupled-lookback forward-progress/deadlock contract undocumented + no host guard found |
| DUP | MEDIUM | partition_reduce_kernel_*.rs (11 files, ~7.6k LOC, 60-70% dup) | Consolidate into one parameterised emitter |
| TST | MEDIUM | tests/ptx_golden_tests.rs | "Golden" tests are 202 substring asserts + 5 snapshots; no true full-PTX golden files |
| C-4 | LOW    | ptx_gen.rs:256 | s32 `mad.lo` global tid caps usable rows at 2^31 (no host assertion) |
| C-1 | LOW    | disk_cache.rs:772 | Disk digest/key via `DefaultHasher` — not toolchain-stable; doc/algorithm hardening |
| P-1 | perf   | 11 files | Fixed 32 ns back-off; implement documented exponential variant |
| P-2 | perf   | hash_join_kernel.rs:685,1328 | Vectorise probe loads (`ld.global.nc.v2.u64`) |

No CRITICAL or HIGH correctness defects were found. The cache layer
(in-process + disk) is correct, concurrent-safe, and poisoning-safe; no
two-kernel PTX-cache-key collision is possible (full-string re-check on hit).
