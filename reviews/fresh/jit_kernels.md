# Review: Bolt JIT Kernel Generators (`src/jit/*`)

Scope: the PTX/kernel-source-generating files listed in the brief (35 files,
~32.5K lines). Excludes `jit_compiler.rs`, `ptx_gen.rs`, `disk_cache.rs`,
`mod.rs` (another agent's scope). All findings are about the *generated CUDA
PTX* and the Rust that emits it. No code was edited.

Overall: this is unusually high-quality, heavily-documented codegen. The
authors clearly understand sm_70 memory ordering (membar.cta / ld.acquire,
membar.gl + ld.acquire.gpu for cross-CTA), NaN semantics, partial-block
handling, warp-shuffle membermask invariants, and probe-bound safety nets. The
dominant problem is **structural duplication of the i32-vs-i64 kernel family**,
not correctness. Findings ranked by severity below.

---

## 1. CODE REVIEW

### CRITICAL

None that affect a default-on path. The one genuine correctness defect
(C-1 below) is gated behind an off-by-default env var, so it is High, not
Critical.

### HIGH

**H-1. Radix scatter is non-stable → LSD radix sort is *incorrect*, not just
"unstable ORDER BY".**
`sort_kernel_radix.rs:592-610` (keys-only) and the keys+indices variant
(`:635-655`) claim each output slot with `atom.global.add.u32` on the
per-digit offset. The doc at `:644-655` correctly notes the result is not a
*stable* sort, but frames it purely as an ORDER-BY-stability concern. It is
worse than that: an LSD radix sort **requires per-pass stability for
correctness** — if pass *p* does not preserve the relative order established
by passes `0..p`, the final fully-sorted result is wrong for any key whose
lower digits already disambiguated it. The race-based slot claim does not
preserve intra-digit order across passes, so multi-pass radix can produce a
mis-sorted output, not merely a differently-tie-broken one. Mitigated only by
the `BOLT_GPU_SORT=1` gate (default OFF, per `:13-14`). Must be fixed (stable
per-block partition + block-ordered scatter) before this path is promoted.
Recommend the doc be corrected to call this a correctness blocker, not a
stability nicety.

**H-2. Radix histogram uses global atomics with no shared-memory
privatization.** `sort_kernel_radix.rs:490` (`atom.global.add.u32` into a
16-entry global histogram). Every thread in every block atomically bumps one
of only 16 global counters. Under realistic (skewed) digit distributions this
serializes massively on a handful of cache lines — the classic reason real
radix sorts privatize the histogram in shared memory per block and reduce once
at the end. This is a large throughput left-on-the-table for the headline
"large-sort" feature. (Performance, but high-impact for the feature's purpose.)

### MEDIUM

**M-1. NaN-handling doc contradicts the emitted code in the partition float
MIN/MAX kernel.** `partition_reduce_kernel_minmax_float.rs:44-51` states "NaN
values therefore propagate into the slot if encountered." But the emitted CAS
loop (`emit_cas_loop`, `:453-473`) does `setp.{lt,gt}.fXX %p7, val, old` then
`selp newv, val, old, %p7` — i.e. it selects `val` only when the comparison is
*true*. For a NaN candidate the comparison is false, so it keeps `old` and NaN
is **ignored**, exactly like the global path (`float_atomics.rs:348-357`, whose
doc at `:28-43` correctly says "NaN inputs are silently ignored"). The two
modules now have opposite documented behaviour for the same situation while the
code does the same (ignore) thing. The doc in `partition_reduce_kernel_minmax_
float.rs` is wrong and should be fixed. (Behaviour itself is fine and matches
SQL MIN/MAX; only the doc is misleading.)

**M-2. Inconsistent acquire-load hardening between spill and non-spill float
MIN/MAX (i32 key).** Non-spill MATCH path uses `ld.acquire.cta.s32 %r35`
(`partition_reduce_kernel_minmax_float.rs:299`) but the `_with_spill` sibling's
MATCH path uses a plain `ld.shared.s32 %r35` (`:714`). Both are preceded by
`membar.cta`, so this is *not* a live correctness bug (the fence already
orders the set-CAS vs the key load); it's an inconsistency where one variant
got the explicit-acquire upgrade and its twin did not. The i64-key float
kernel applies the acquire load uniformly (`partition_reduce_kernel_minmax_
float_i64.rs:270`), so the spill i32 variant is the odd one out. Recommend
aligning for auditability.

**M-3. The i32-vs-i64 duplication problem (quantified).** Seven file pairs are
near-duplicates differing essentially in (a) key-load width `s32` vs `s64`,
(b) the `mul.wide`/stride for the key, and (c) entry-name suffix:

| pair | i32 lines | i64 lines | line-similarity* |
|------|-----------|-----------|------------------|
| `partition_kernel` / `_i64` | 853 | 770 | 0.45 |
| `partition_reduce_kernel` / `_i64` | 1063 | 744 | 0.51 |
| `partition_reduce_kernel_count` / `_i64` | 672 | 562 | 0.67 |
| `partition_reduce_kernel_minmax` / `_i64` | 829 | 776 | 0.70 |
| `partition_reduce_kernel_minmax_float` / `_i64` | 905 | 865 | 0.78 |
| `partition_reduce_kernel_multi` / `_i64` | 921 | 742 | 0.74 |
| `scatter_kernel` / `_i64` | 547 | 470 | 0.57 |

\* whole-file `difflib` ratio; *deflated* by divergent doc comments — the
actual PTX-emitting statement bodies are far closer. The i64 siblings total
**~4,929 lines** that exist almost entirely to flip a load width and a suffix.
Each also carries a `_with_spill` twin, doubling the surface again. The
`partition_reduce_kernel_spill_common.rs` module already extracts the
byte-identical fragments (header, thread-ids, spill-bump, slice-read, loop
epilogue) and is exactly the right pattern — but it deliberately stops short of
the per-kernel bodies (`:27-35`). **Recommendation:** introduce a `KeyWidth`
(I32/I64) parameter threaded through a single emitter per logical kernel
(sum / count / minmax-int / minmax-float / multi / scatter / partition),
selecting `s32`/`s64`, the stride, and the suffix from the parameter. The
golden full-PTX snapshots in `tests/ptx_golden_partition_snapshots.rs` already
exist precisely to make this collapse provably byte-identical (see the module
header at `:3-13` which explicitly calls out the planned dedup) — so the safety
net is in place; the work just hasn't been done. This is the single
highest-leverage cleanup in the whole scope.

**M-4. `partition_reduce_kernel_minmax_float.rs` carries dead computed
locals.** `cas_suffix` / `setp_dt` are computed (`:134-135`) then explicitly
discarded with `let _ = cas_suffix; let _ = setp_dt;` (`:400-401`) because the
real values are recomputed inside `emit_cas_loop`. Minor, but it signals an
incomplete refactor and should be removed.

### LOW

**L-1. Fixed 32 ns spin back-off everywhere; exponential back-off TODO never
done.** Repeated across `float_atomics.rs:93`, `partition_reduce_kernel*.rs`,
etc. Documented as deliberate (`SPIN_BACKOFF_NS`); fine for v1.

**L-2. Lookback scan forward-progress depends on a host launch contract that
the kernel cannot enforce.** `prefix_scan.rs:869-903` documents the
co-residency / no-deadlock requirement thoroughly and provides
`lookback_launch_is_safe` (`:1288`) as a tested guard. This is correctly
handled *if* the (out-of-scope) launch site actually calls it. Worth a
cross-check with the executor reviewer that the `debug_assert!` + multipass
fallback is wired up; the kernel side is sound.

**L-3. Blelloch scan has the textbook strided-shared-access pattern**
(`prefix_scan.rs:625-650`, `idx = tid*stride + stride-1`) which causes
shared-memory bank conflicts at large strides. The classic fix is conflict-free
offset padding (`CONFLICT_FREE_OFFSET`). At BLOCK_SIZE=256 the impact is small
and the Hillis-Steele variant is the default; low priority.

### Correctness items checked and found SOUND
- **Cross-CTA ordering (lookback scan):** `membar.gl` after both publishes +
  `ld.acquire.gpu.u32` on the spin read (`prefix_scan.rs:1129,1153,1201`). Correct
  for sm_70+.
- **Intra-CTA set/key publish race:** `membar.cta` between set-CAS and key
  load/store on both CLAIM and MATCH paths; pinned by a dedicated regression
  golden (`ptx_golden_tests.rs:614-686`). Correct.
- **Warp-shuffle reduction:** `shfl.sync.down.b32` with membermask `0xffffffff`
  gated on `tid < 32` (`agg_kernels.rs:527,548`). Safe because the launcher
  always launches full 256-thread blocks (out-of-range lanes seed identity in
  phase 1), so warp 0 is always fully populated even for partial final blocks.
  Backed by a compile-time block-size invariant (`:48-53`).
- **Float MIN/MAX atomics:** CAS-on-bit-pattern loop with no-op-skip and
  ±inf identity init (`float_atomics.rs`, `partition_reduce_kernel_minmax_
  float.rs:141-146`). NaN ignored (correct for SQL).
- **i128 decimal SUM:** atomic-free two-stage block reduce with
  `add.cc.u64`/`addc.u64` carry chain and double-b32-shuffle warp tail
  (`decimal_agg.rs:29-41`). Sound; avoids the nonexistent 128-bit atomic.
- **Integer overflow on SUM:** `cvt.s64.s32` widening before `add.s64`,
  regression-pinned (`ptx_golden_tests.rs:372-404`). Correct.
- **No integer division/modulo in hot paths:** slot index is `& (k-1)` with
  host-enforced power-of-two `k` (`hash_kernels.rs:314,521,...`). No div/rem.
- **Unbounded-probe hangs:** keys and agg probe loops are bounded
  (`setp.gt.u32` + `bra DONE`), pinned by goldens (`ptx_golden_tests.rs:408-484`).
- **No stubs/panics in generators:** every unsupported `(op, dtype)` returns a
  `BoltError`; no `todo!()`/`unimplemented!()` in the kernel emitters. Decimal
  gather and float GROUP BY MIN/MAX-in-`hash_kernels` are explicit errors
  (`prefix_scan.rs:1461`, `hash_kernels.rs:1511`) routed to dedicated kernels.

---

## 2. TESTS

Two suites cover the scope:

- **`tests/ptx_golden_tests.rs`** (~1,759 lines): two-layer — (1) substring
  assertions pinning the *behavioral contract* (which mnemonics / labels / dtype
  suffixes / ordering must be present, e.g. `membar.cta` before the MATCH key
  load, `bra DONE` before `FOUND`), and (2) `insta` snapshots over
  `normalize_ptx`-rewritten register names to pin *register flow* while
  absorbing allocator-counter churn.
- **`tests/ptx_golden_partition_snapshots.rs`** (494 lines + snapshot files):
  full-PTX `insta` snapshots for every `partition_reduce_kernel_*` entry point
  (base + spill, plus arity 1/2 for the multi variants), explicitly created to
  make the planned i32/i64 dedup byte-provable.

**What they verify:** text and structure, *not runtime behavior*. The header
of `ptx_golden_partition_snapshots.rs:15-18` is explicit: "HOST-SIDE only — PTX
codegen runs entirely on the CPU, needs no GPU." There is **no test that
assembles the PTX with ptxas or executes it on a device** within this scope.
So a generated kernel that is syntactically plausible but semantically wrong
(e.g. a swapped operand that still emits the right mnemonics) would pass. The
ordering/precedence assertions (`assert_appears_before`) and the dedicated
race-fence regression goldens partly compensate by encoding *semantic
intent* as text constraints, which is a good pragmatic middle ground.

**Coverage estimate: ~85% is a fair characterization for *structural* coverage.**
Nearly every public `compile_*` entry point has at least a smoke/shape golden,
and the refactor-sensitive ones (lookback, blelloch, float CAS, partition
race) have deep contract goldens. Gaps:
- No end-to-end *numeric* verification of any kernel (no ptxas-assemble step,
  no `#[ignore]`'d device round-trip in scope — the hash-join doc mentions one
  but it's host-replay of the hash, not kernel execution).
- The non-stable radix scatter (H-1) has **no test asserting sortedness** —
  precisely the property that's broken — only shape goldens.
- The M-2 acquire-load inconsistency is invisible to tests because both forms
  satisfy the existing `membar.cta`-ordering assertions.

**Enhancements:**
1. Add a CI job that runs every emitted PTX through `ptxas -arch=sm_70`
   (assemble-only) to catch malformed PTX that string tests miss — cheap, no GPU.
2. Add `#[ignore]`'d device tests (run on GPU CI) that execute representative
   kernels against a CPU reference: sum/count/minmax over random data with
   duplicate keys, the prefix-scan family, and especially a **sortedness +
   stability assertion for the radix path** before it can go default-on.
3. Property-test `normalize_ptx` itself (it is duplicated verbatim across both
   test files — extract to a shared `tests/common` module to avoid drift).

---

## 3. NEW FEATURES / DIRECTIONS

1. **Collapse the i32/i64 family (M-3)** — the single biggest maintainability
   win; the byte-identical snapshot net already exists.
2. **Shared-memory privatized radix histogram (H-2)** and a **stable radix
   scatter (H-1)** — required to make `BOLT_GPU_SORT` default-on and to make the
   large-sort feature actually fast and correct.
3. **Float radix sort** — currently deferred (`sort_kernel_radix.rs:50-56`);
   the IEEE-monotonic key transform is well understood and would remove a host
   fallback.
4. **Warp-aggregated atomics** for the GROUP BY agg and hash-join output
   counters — `match.any.sync` / ballot to coalesce same-slot or same-counter
   increments into one atomic per warp; large win under hot-key skew (the probe
   kernels already have a speculative-load guard, this is the next step).
5. **Warp-wide decoupled-lookback** — `prefix_scan.rs:862-864` already flags the
   single-thread lookback as a known optimization point.
6. **i64-key float MIN/MAX** is noted as having "no workload driver"
   (`partition_reduce_kernel_minmax_float.rs:41-43`) yet the `_i64` file exists —
   confirm whether it is reachable or dead; if dead, it is more dedup surface.
7. **Decimal128 GPU gather / GROUP BY** — currently hard-errors
   (`prefix_scan.rs:1461`, `hash_kernels.rs:1511`); a natural follow-up given the
   decimal SUM reduce already exists.
