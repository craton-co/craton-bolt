# Code Review — GROUP BY Execution Subsystem (`src/exec/groupby*.rs`)

Reviewer: automated deep review. Date: 2026-05-30. Scope: all 32 `src/exec/groupby*.rs`
files (26,220 LOC) plus the relevant integration tests under `tests/`.

## Files in scope (LOC)

| Category | Files | LOC |
|---|---|---|
| Core / fallback | `groupby.rs` (2982), `groupby_valid.rs` (2941), `groupby_wide.rs` (2066), `groupby_with_pre.rs` (2828), `groupby_common.rs` (1059) | 11,876 |
| Dispatch | `groupby_shmem_dispatch.rs` (320), `groupby_tier2_dispatch.rs` (313) | 633 |
| Tier-1 shmem execs | `groupby_shmem_{exec,multi_exec,avg_exec,count_exec,minmax_exec,launch}.rs` | ~2,725 |
| Tier-2 execs/orchestrators/merge | 17 `groupby_tier2_*` files | ~10,900 |

Overall this is a mature, careful subsystem. NULL/NaN/overflow/spill hazards are
mostly identified and handled, dispatchers are pure and well unit-tested, and the
spill sentinel contract is real and exercised. The dominant problem is **massive
structural duplication** across the ~20 Tier-1/Tier-2 executor variants, and there
is **one genuine cross-path correctness inconsistency** (grouped float MIN/MAX NaN).

---

## 1. CODE REVIEW — findings ranked by severity

### CRITICAL
None. No silently-wrong arithmetic on the default paths, no unguarded overflow, no
unhandled spill. The bugs that would be critical (wrapping SUM overflow, dropped
spill rows, NULL-key garbage, sentinel collisions) all have guards.

### HIGH

**H1 — Grouped float MIN/MAX disagrees with scalar MIN/MAX on NaN (cross-path inconsistency).**
`groupby_tier2_minmax_float_exec.rs:179-183` (and the two-key twin
`groupby_tier2_twokey_minmax_float_exec.rs:201-209`) **defer** any NaN-bearing float
value column, with the stated rationale that the fallback implements DuckDB's
"NaN-as-largest total order (`aggregate.rs::float_total_cmp`)". That rationale is
incorrect about where the query actually lands. Grouped float MIN/MAX does **not**
go through the scalar `float_total_cmp` path; it routes through the
`float_atomics` CAS kernel (`groupby.rs:205,752-781`), whose documented semantics
(`src/jit/float_atomics.rs:40`) are *"NaN inputs are silently ignored"* (SQL
"MIN/MAX ignore NaN"). Meanwhile the **scalar** path (`aggregate.rs:1833-1839`)
treats NaN as Greater, so scalar `MAX(v)` returns NaN when any NaN is present.

Result: for the same NaN-bearing data,
`SELECT MAX(v) FROM t` → **NaN**, but
`SELECT k, MAX(v) FROM t GROUP BY k` → **largest non-NaN** in the NaN group.
The Tier-2 deferral does not fix this — it just relocates the divergence from the
fast path to the global path. The "one answer per query" invariant the file's F2
comments claim to uphold is violated. Recommendation: make grouped float MIN/MAX
honour the same total order as the scalar path (either a `float_total_cmp`-equivalent
CAS comparator, or a host post-pass that re-injects NaN for groups that contained
one), and correct the misleading comments at the two deferral sites.

**H2 — NULL group-by keys are silently dropped, diverging from DuckDB/Postgres.**
`groupby.rs:431-446`: rows whose key is NULL are filtered out entirely ("NULL keys
are not a group"). Standard SQL (and DuckDB, Postgres, Polars) emits **one NULL
group**. This is documented as an intentional v1 choice, but it is a real result
divergence that will surface as wrong row counts in `diff_duckdb`-style comparisons
for any column that is both a key and nullable. No test asserts the intended
behaviour either way (see §2). Recommendation: implement the NULL-group (the
sentinel-free `groupby_valid` path already has the slot-valid machinery to host it),
or at minimum document it as a known deviation in `docs/GROUPBY_PERF.md` and add a
diff-duckdb xfail.

### MEDIUM

**M1 — Massive code duplication across the ~20 executor variants (primary maintainability risk).**
Quantified across the Tier-1/Tier-2 execs:
- `fn get_or_build_module` + `static LOAD_COUNT` + per-file `enum KernelSpec`:
  **11 near-identical copies** (`grep -l "fn get_or_build_module"` → 11;
  `static LOAD_COUNT` → 11). Each is ~30-40 LOC of module-cache plumbing differing
  only in the reduce-kernel arm.
- `fn partition_spec_for`: 5 byte-identical copies (the shared threshold helper
  `groupby_tier2_common::use_shmem_staging_partition` exists, but the wrapper is
  still pasted per file).
- The full **partition → offsets → scatter → reduce → pinned-D2H** pipeline body:
  **~10 copies** (orchestrators + count/minmax/minmax_float/twokey execs), each
  ~120-180 LOC, differing only in (a) reduce kernel entry, (b) value dtype, (c) the
  output array constructor. See `groupby_tier2_orchestrator.rs:234-535`,
  `groupby_tier2_minmax_float_exec.rs:233-377`,
  `groupby_tier2_count_exec.rs:183-330` — the partition+scatter halves are
  essentially identical.
- The spill-error literal `"partition_reduce spill: {} rows exceeded MAX_PROBES; …"`
  is hand-written in **~10 sites** (grep count above) rather than formatted by one
  helper keyed off `PARTITION_REDUCE_SPILL_PREFIX`. The orchestrator exports the
  prefix const but not a constructor, so the full string can drift in one site and
  silently break the `starts_with` fallback contract in `groupby.rs:284-291`.
- The NULL-deferral guard (`null_count() > 0 → return None`) appears in ~20 sites
  with copy-pasted block comments.

This is ~6,000-8,000 LOC of which a large fraction is mechanical repetition. The
module docs (`groupby_tier2_dispatch.rs:44-74`, `groupby_tier2_common.rs:3-24`)
pre-emptively argue *against* consolidation, claiming the ABIs are
"not interchangeable" and a blind merge "cannot be verified without GPU hardware."
That argument is valid for the *kernel-arg lists* but overstated for the
*host scaffolding*. Recommended deduplication strategy (host-side, behaviour-preserving,
unit-testable without a GPU):
  1. **Generic module-cache helper.** Replace the 11 `get_or_build_module`/`LOAD_COUNT`
     copies with one generic `module_cache::cached<K: Hash+Debug>(namespace, key, build_fn)`
     already half-present in `module_cache`; each exec keeps only its `KernelSpec`
     enum and a one-line `match` for the build closure.
  2. **A `Tier2Pipeline` driver** parameterised over a small trait
     (`ReduceKernel { entry(); push_reduce_args(); out_dtype(); build_output() }`).
     The partition+scatter+offsets+spill-check+slot-walk is identical for every
     single-key variant and is the single largest copy; the trait isolates exactly
     the 3 things that differ. This is pure host code and can be diff-tested against
     the current per-file bodies on CPU (the existing `collect_populated_slots_sorted`
     differential tests are the model).
  3. **One `spill_error(count)` constructor** in the orchestrator module next to
     `PARTITION_REDUCE_SPILL_PREFIX`, used by all ~10 sites (eliminates the
     drift risk, makes the multi vs single message variants explicit).
  4. **One `defer_if_null(&[&dyn Array]) -> Option<()>`** helper in `groupby_tier2_common`
     for the NULL guard. Conservatively this would cut the Tier-2 surface by an
     estimated 30-40% with zero behaviour change.

**M2 — `pre.is_some()` hard-errors in `execute_groupby` but `groupby_with_pre` exists and is wired only from the engine.**
`groupby.rs:374-378` returns a hard `"GROUP BY with projection/filter pre-kernel not
yet implemented"` error, while `engine.rs:2592` calls
`groupby_with_pre::execute_groupby_with_pre`. Routing depends entirely on the caller
choosing the right entry point; a call to `execute_groupby` with a `pre` plan fails
even though a working implementation exists. This is fragile — recommend
`execute_groupby` delegate to `groupby_with_pre` on `pre.is_some()` rather than
erroring, so the entry point is correct-by-construction.

**M3 — COUNT scatter wastes an n_rows×f64 dummy buffer.**
`groupby_tier2_count_exec.rs:225-257`: COUNT needs only keys, but reuses the generic
scatter kernel and allocates `dummy_vals_in` + `scatter_vals` (2 × 8·n_rows bytes,
~160 MB at n=10M) that are written and never read. A keys-only scatter kernel (or a
zero-width value param) would halve the scatter D2/allocations on the COUNT path.
Performance, not correctness.

### LOW

**L1 — `validate_offsets_monotonic` is O(K) but only catches reversal, not overflow-past-n_rows mid-array.**
`groupby_tier2_orchestrator.rs:185-196` checks non-decreasing but the separate
`offsets[K] == n_rows` check (`:383`) only validates the endpoint. A corrupt interior
offset that is monotonic but exceeds `n_rows` would still index OOB in the reduce
slice. Very unlikely given the prefix-sum source, but the guard is advertised as
defensive; tightening it to also assert `offsets[pid] <= n_rows` is cheap.

**L2 — `collect_populated_slots_sorted` parallel path spawns `available_parallelism`
threads per call** (`groupby_tier2_common.rs:289-329`). For the 4.2M-slot production
output this is fine, but every Tier-2 query pays a thread-spawn/join cycle. A shared
rayon pool (if the crate already uses rayon elsewhere) would amortise this. Minor.

**L3 — Historical `stub_` aliases** (`groupby_tier2_orchestrator.rs:85-89`) bind to
real kernels but the naming is actively misleading to a new reader ("are these
stubs?"). Pure cosmetics; rename to drop `stub_`.

**L4 — `TIER2_MAX_GROUPS = 100_000_000`** (`groupby_tier2_dispatch.rs:146`) is also
hard-coded as the literal `100_000_000` inside several execs
(`groupby_tier2_minmax_float_exec.rs:204`, count exec, etc.) rather than referencing
the const, so the dispatcher cap and the executor self-gate can drift apart.

**Positive findings (verified correct):**
- COUNT accumulates in `u64` (`groupby_tier2_count_exec.rs:264`) — no overflow at any
  realistic n_rows; widened to i64 output with a documented bound.
- Grouped integer SUM overflow is guarded host-side with `checked_add` matching the
  scalar contract verbatim (`groupby.rs:1329-1352`, `1354+`), with both positive and
  negative overflow tests (`:2409-2440`). The wrapping-atomic risk is correctly
  identified and the long-term kernel fix is TODO'd.
- Spill (hash-table MAX_PROBES overflow) is a real, wired soft-fallback: kernel bumps
  an atomic counter, orchestrator surfaces the structured error
  (`groupby_tier2_orchestrator.rs:466-479`), and `execute_groupby` matches the prefix
  and falls through to global-atomic (`groupby.rs:279-304`). Contract is unit-tested.
- i64 two-key packing is negative-safe (`as u32 as u64`,
  `groupby_tier2_twokey_exec.rs:55-65`) and the i64 partition hash uses the full
  64-bit multiplicative hash, not the low half (`partition_kernel_i64.rs:15,84`).
- EMPTY_KEY (`i64::MIN`) sentinel collisions are caught both pre- and post-encoding and
  rerouted to the sentinel-free `groupby_valid` path (`groupby.rs:387-462`).
- NaN/NULL deferral guards are present and consistent across all Tier-2 float/int
  execs (the *guards* are correct; only H1's *destination* is wrong).

---

## 2. TESTS

**Coverage is good for the host-pure logic, thin for the GPU paths and absent for the
hard hazards.** The "~85%" figure is optimistic — closer to ~85% of *host* lines but
much lower for *behavioural* coverage of spill/collision/wide paths.

Strong:
- Dispatchers: exhaustive boundary tests (`groupby_shmem_dispatch.rs:160-320`,
  `groupby_tier2_dispatch.rs:203-313`) — every threshold edge and every precondition.
- `groupby_tier2_common`: differential tests against the exact pre-extraction inline
  loops, including the parallel slot-scan path (`:339-561`). Exemplary.
- `groupby_tier2_merge`: empty/single/multi-partition + schema (`:113-216`).
- Overflow guard: positive and negative i64 overflow (`groupby.rs:2409-2440`).
- Spill sentinel prefix contract pinned (`groupby.rs:2942-2979`,
  `orchestrator:625-639`).

Gaps / enhancements needed:
- **Spill path is never exercised end-to-end.** The only spill test that runs the
  kernel is `#[ignore]`d (`orchestrator:656-706`, requires CUDA). There is no CPU-side
  test that the `execute_groupby` *fallthrough* actually recomputes correctly after a
  simulated `Some(Err(spill))`. Add a host test that injects the spill error and
  asserts the global-atomic recompute matches an oracle.
- **Hash collisions / bucket overflow** are tested only for DISTINCT
  (`diff_duckdb.rs:25`), not for GROUP BY. Add a pathological-key GROUP BY case
  (>BLOCK_GROUPS distinct keys into one partition) at least as a host oracle vs the
  global path.
- **Most e2e tests are GPU-gated** (`#[ignore]`): shmem 6/7, tier2 7/9, multi_sum 4/14.
  On CI without a GPU, the actual GPU kernels are essentially untested. The twokey
  e2e file has 0 ignores (5/5 run) — confirm those don't silently no-op when CUDA is
  absent (several execs `return` on `from_slice` error, which would make a "passing"
  test vacuous).
- **Wide groups** (`groupby_wide.rs`, 2066 LOC, >64-bit packed keys) — no dedicated
  e2e file in scope; coverage relies on `diff_duckdb`. Add explicit wide-key
  (3+ Int32 columns, or Utf8 keys) GROUP BY tests.
- **NULL-key-as-group** (H2): no test asserts the current drop behaviour *or* the
  standard behaviour. Whichever is chosen, pin it.
- **NaN grouped MIN/MAX** (H1): no test compares grouped vs scalar float MAX with NaN
  present. The `nan_tests` modules only assert the fast path *defers*; they never
  assert the *fallback's* answer matches scalar. That blind spot is exactly why H1
  went unnoticed.

---

## 3. NEW FEATURES / DIRECTIONS

1. **NULL-group support** (resolves H2) — make NULL its own group via the
   `groupby_valid` slot-valid protocol; aligns with every reference engine.
2. **Total-order float MIN/MAX** (resolves H1) — a NaN-propagating CAS comparator or
   host post-pass so grouped and scalar agree; then the Tier-2 NaN deferral can be
   removed entirely (faster, and correct).
3. **`Tier2Pipeline` generic driver** (M1) — the deduplication itself is a feature:
   it unlocks new aggregates (e.g. grouped `STDDEV`, `COUNT(DISTINCT)`,
   multi-key MIN/MAX of more dtypes) by implementing one trait instead of copying a
   600-LOC file. This is the highest-leverage investment.
4. **On-device overflow flag** (the existing `TODO(overflow-kernel)`,
   `groupby.rs:1325`) — lets the SUM overflow guard drop its host re-fold and enables
   streaming inputs never materialised host-side.
5. **Tier-1 float MIN/MAX and Tier-1 COUNT** — currently low-cardinality float
   MIN/MAX and COUNT have no shmem fast path and fall to global-atomic
   (`minmax_float_exec.rs:198-203`, `count_exec.rs:170-175`).
6. **Wide / string group keys on the GPU** — `groupby_wide` is host-side today;
   a composite-hash + on-device tuple-verify path (mentioned as deferred in
   `groupby.rs:40`) would close the perf gap for multi-column / Utf8 keys.
7. **Unified dispatch** — merge `groupby_shmem_dispatch` and `groupby_tier2_dispatch`
   (the v2 module already supersedes v1; the duplicate `AggOp`/`DispatchInputs` types
   are flagged "merged by follow-up" in their own docs).

---

## Summary table

| ID | Severity | Issue | Location |
|----|----------|-------|----------|
| H1 | High | Grouped float MIN/MAX ignores NaN; scalar treats NaN as largest → cross-path divergence; deferral rationale wrong | `groupby_tier2_minmax_float_exec.rs:179-183`; `float_atomics.rs:40`; `aggregate.rs:1833` |
| H2 | High | NULL group keys silently dropped vs DuckDB NULL-group | `groupby.rs:431-446` |
| M1 | Medium | ~10-11× duplication of module-cache, pipeline, spill-error, NULL-guard scaffolding | 20+ `groupby_tier2_*`/`groupby_shmem_*` files |
| M2 | Medium | `execute_groupby` hard-errors on `pre` though `groupby_with_pre` exists | `groupby.rs:374-378` vs `engine.rs:2592` |
| M3 | Medium | COUNT scatter allocates unused n_rows×f64 buffers | `groupby_tier2_count_exec.rs:225-257` |
| L1 | Low | Offset validation misses monotonic-but-OOB interior offsets | `groupby_tier2_orchestrator.rs:185-196,383` |
| L2 | Low | Per-call thread spawn in slot scan | `groupby_tier2_common.rs:289-329` |
| L3 | Low | Misleading `stub_` kernel aliases | `groupby_tier2_orchestrator.rs:85-89` |
| L4 | Low | `TIER2_MAX_GROUPS` cap hard-coded in execs, can drift from const | `groupby_tier2_dispatch.rs:146` + execs |
