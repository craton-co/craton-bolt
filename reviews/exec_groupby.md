# Code Review — `src/exec/` GROUPBY / AGGREGATE family

Reviewer: senior GPU-database reviewer
Scope: `agg_with_pre.rs, aggregate.rs, expr_agg.rs, extended_agg.rs, distinct.rs,
welford.rs, partition_offsets.rs, groupby*.rs`, the `groupby_shmem_*` family, and
the `groupby_tier2_*` family (common/dispatch/orchestrator/merge fully; ≥2 of each
leaf-variant family read, the rest diffed).
Engine: `craton-bolt` (Apache-2.0, Craton Software Company).

---

## 0. Executive summary

The aggregate family is unusually well documented and the host-side numeric
contracts (Welford, integer SUM overflow, signed-zero canonicalisation, two-key
i64 packing) are correct and carefully tested. The dispatch layering in
`execute_groupby` is sound, and the `partition_reduce spill:` soft-fallback
contract is consistently honoured across every Tier-2 orchestrator.

The most important finding is a **NULL-handling correctness bug in the two-key
COUNT executor** (over-counts `COUNT(col)` when the counted column has NULLs),
which every sibling COUNT executor guards against but this one does not. There is
also a **family-wide gap in GPU float MIN/MAX NaN semantics**: the host scalar
path implements DuckDB's NaN-as-largest convention, but the GPU CAS-loop float
min/max executors have no NaN handling and no NaN tests, so grouped float
MIN/MAX likely disagrees with the scalar/window path.

Duplication across the Tier-2 / shmem executors is real and large (~20 near-clone
`try_execute` bodies; ~5,000 lines), but the authors made a deliberate,
well-argued decision not to consolidate behind flags. I largely agree, with a
narrower consolidation recommendation below.

---

## 1. CODE REVIEW — findings

### 1.1 CRITICAL / HIGH — correctness

#### F1 (HIGH) — Two-key COUNT over-counts NULLs in `COUNT(col)`
**File:** `groupby_tier2_twokey_count_exec.rs:119-159` (and dispatch at
`groupby.rs:336`)

`try_execute` matches `AggregateExpr::Count(_)` ("argument is decorative") and
commits to the GPU fast path. It guards only the **key** columns for NULLs
(`k1.null_count() > 0 || k2.null_count() > 0`, line 157) — it never captures the
counted-column name and never declines a NULL-bearing `COUNT(col)`. The reduce
kernel counts **every** scattered row, so `SELECT a, b, COUNT(c) FROM t GROUP BY
a, b` where `c` contains NULLs returns counts that are too high (NULL rows of `c`
should be excluded per SQL).

This contradicts the explicit guard every sibling COUNT executor carries:
- `groupby_shmem_count_exec.rs:60-90` captures `count_col_name` and defers if
  that column has NULLs.
- `groupby_tier2_count_exec.rs:120-153` does exactly the same.

Because the bug commits to the fast path (returns `Some(Ok(_))`), the
always-correct global-atomic fallback at `groupby.rs:359+` is never reached, so
the wrong result is returned silently.

**Fix:** mirror the single-key COUNT executors:
```rust
let count_col_name: Option<&str> = match &aggregate.aggregates[0] {
    AggregateExpr::Count(Expr::Column(n)) => Some(n.as_str()),
    AggregateExpr::Count(_) => None,
    _ => return None,
};
// ... after key NULL guard:
if let Some(name) = count_col_name {
    if let Some(col) = batch.column_by_name(name) {
        if col.null_count() > 0 { return None; }
    }
}
```
Add a unit test with a NULL-bearing counted column asserting `try_execute`
returns `None`.

#### F2 (HIGH) — GPU float MIN/MAX NaN semantics undefined and untested
**Files:** `groupby_tier2_minmax_float_exec.rs`,
`groupby_tier2_twokey_minmax_float_exec.rs` (+ kernels
`jit::partition_reduce_kernel_minmax_float*`, out of scope but implicated)

The host scalar/window path implements DuckDB's convention precisely: NaN sorts
as the largest value, all NaN bit patterns compare equal, MIN skips NaN unless
all-NaN, MAX surfaces NaN if present (`aggregate.rs:1807-1896`,
`float_total_cmp`). The grouped GPU float MIN/MAX executors route through an
`atom.shared.cas.b{32,64}` retry loop and contain **zero** NaN handling or NaN
tests (confirmed: no `NaN|is_nan|INFINITY` matches in either float-minmax exec).
CAS-loop comparisons on raw IEEE floats make NaN's participation
order-dependent, so grouped `MIN(float)/MAX(float)` can disagree with the scalar
aggregate over the same data — a "two answers for one query" inconsistency, the
exact class the engine's V-10 overflow work was created to eliminate.

Note these executors do NOT defer NaN-bearing batches (only NULL-bearing ones),
so NaN data reaches the kernel.

**Fix:** decide the contract and enforce it. Cheapest correct option:
defer NaN-bearing float value columns to the global-atomic / host path (add a
`val_arr.values().iter().any(|v| v.is_nan())` decline in `try_execute`), then add
a regression test pinning grouped == scalar for NaN inputs. Longer term, make the
float CAS kernel implement the NaN-as-largest total order so the fast path stays
on-GPU.

#### F3 (MEDIUM) — NaN group keys do not collapse (GROUP BY/DISTINCT vs DuckDB)
**File:** `groupby_common.rs:248-264` (`load_key_column_bits`),
`distinct.rs:31-35`

Float group **keys** are canonicalised for signed zero (`-0.0 → +0.0`,
consistent across GROUP BY / DISTINCT / JOIN — good, F-positive) but NaN bit
patterns are preserved verbatim. Two NaNs with different payloads therefore form
distinct groups, and even identical-payload NaNs are documented to "dedupe to one
row" only by bit-pattern accident. DuckDB treats all NaN as a single GROUP BY /
DISTINCT key. This is documented as an intentional `NaN != NaN` choice, but it is
a real divergence from the reference engine the benches compare against. At
minimum it should be called out in user docs; ideally canonicalise NaN payloads
to a single quiet-NaN bit pattern in `load_key_column_bits` so GROUP BY/DISTINCT
match DuckDB.

#### F4 (LOW) — AVG returns 0.0 instead of SQL NULL for empty/all-NULL input
**Files:** `aggregate.rs:313-327`, `agg_with_pre.rs:450-461`

`AVG` over zero matching rows returns `0.0` rather than SQL NULL, to keep the AVG
output field non-nullable. Documented as `TODO(null)` in both paths. This is a
scalar-path semantics deviation (a grouped AVG omits empty groups entirely, which
is correct). Low severity but a genuine SQL-conformance gap; fixing requires
making the AVG output field nullable across the planner.

### 1.2 Correctness items verified GOOD (no action)

- **Welford** (`welford.rs`): `push` and Chan-Golub-LeVeque `combine` are
  textbook-correct; `var_samp`/`stddev_samp` correctly return `None` for
  `count <= 1`; `combine` short-circuits empty states. Well tested.
- **Integer SUM overflow** (`aggregate.rs:1745-1795`, `groupby.rs:1293-1389`):
  scalar path errors via `checked_add`; the grouped path's device atomic wraps
  (`atom.global.add.u64`) but a faithful host re-fold (`checked_group_sum`) is
  run on aligned `(values, host_keys)` and on the native-validity path
  (`checked_group_sum_native_validity`) to restore the "never silently wrong"
  invariant. The associativity argument is valid. A `TODO(overflow-kernel)`
  correctly notes the streaming case still needs an on-device flag.
- **Two-key i64 packing** (`groupby_tier2_twokey_exec.rs:55-65`,
  `groupby_tier2_twokey_merge.rs:54-59`, `groupby_common.rs:324-465`):
  hi=col0/lo=col1 via `u32 as u64`, lossless, round-trips for `i32::MIN/MAX`/-1;
  `(a,b)` vs `(b,a)` proven distinct. `pack_keys` uses `wrapping_shl` (V-17) to
  make a hypothetical shift==64 deterministic. Solid and well tested.
- **Float scalar MIN/MAX NaN** (`aggregate.rs:1807-1896`): correct DuckDB
  NaN-as-largest via `float_total_cmp`, seeded from first element so all-NaN
  yields NaN. (Contrast F2 — only the *scalar* path is correct.)
- **NULL deferral on the fast paths** (all shmem + tier2 single-key/multi
  executors): every executor reads `.values()` off the raw Arrow buffer and
  correctly declines NULL-bearing key/value batches back to the global-atomic
  path which consults the validity bitmap. Consistent and correct — **except**
  F1.
- **Tier-2 merge / disjoint-partition invariant**
  (`groupby_tier2_merge.rs`, `_multi_merge.rs`, `_twokey_merge.rs`): each input
  key hashes to exactly one partition, so concatenate-then-sort is correct;
  length-mismatch and schema-arity errors are surfaced structurally.
- **Spill / MAX_PROBES handling** (every orchestrator + count/avg/minmax execs):
  open-addressing overflow bumps a device counter; the host surfaces a
  `partition_reduce spill:` structured error which `execute_groupby` recognises
  via `PARTITION_REDUCE_SPILL_PREFIX` and treats as a *soft* miss, falling
  through to global-atomic. The contract string is centralised and tested
  (`groupby_tier2_orchestrator.rs:75, 621-635`).
- **Offset monotonicity guard** (`groupby_tier2_orchestrator.rs:181-192`,
  reused by multi/twokey via import): defends the reduce kernel against a
  wrap-around slice from a corrupt prefix-sum. Good defensive O(K) check.
- **Multi-SUM / AVG scatter alignment** (`_multi_orchestrator.rs:14-47`,
  `tier2_avg_exec.rs:282-356`): the deterministic `dest_idx` + atomic-free
  indexed value scatter correctly eliminates the *real* latent bug of relying on
  cross-launch `atomicAdd` ordering (which is not a CUDA contract). This is a
  genuinely good design fix; the COUNT/SUM key+set buffer aliasing in
  `tier2_avg_exec.rs:371-384` is justified (byte-identical re-store on one
  stream).
- **DISTINCT** (`distinct.rs`): signed-zero canonicalisation matches GROUP
  BY/JOIN; NaN preserved as-is (see F3); host distinct-count cap is enforced with
  a clear override env var.
- **Int64 MIN/MAX precision** (`groupby_tier2_minmax_exec.rs:23-28, 260-271`):
  the earlier f64-round-trip narrowing bug for `|v| > 2^53` was correctly
  replaced by a typed `scatter_kernel_i32_to_i64` path; the old mantissa decline
  guard was deleted.

### 1.3 Stubs / dead code / TODO inventory

No `todo!()`/`unimplemented!()` macros exist in the family. All `unreachable!()`
uses are genuinely guarded by the preceding `try_execute` match and are fine.
The `stub_*` aliases in `groupby_tier2_orchestrator.rs:85-91` are **misleadingly
named** — they bind to real kernel modules (the module doc says so at line 54).
Recommend renaming to drop `stub_` to avoid future readers grepping for "stub"
and flagging a non-issue.

Genuine "not yet supported" surfaces (all return clean errors, not silent wrong
answers — acceptable):
- `groupby_common.rs:350-363` — >2 key columns / >64 bits of key width.
- `groupby.rs:374-377`, `groupby_valid.rs:250` — GROUP BY with pre-kernel.
- `groupby_with_pre.rs:285-308, 1591-1615, 2361` — float/Utf8/Decimal128 keys,
  MIN/MAX over float in the pre path.
- `groupby_wide.rs:103, 285-304` — wide GROUP BY with pre / Utf8 / Decimal /
  temporal keys.
- `expr_agg.rs:328-368, 615` — CAST/EXTRACT/DATE_TRUNC/subquery/Decimal in the
  host expr evaluator.
- `groupby_shmem_minmax_exec.rs:67`, `groupby_tier2_minmax_exec.rs:170` — float
  MIN/MAX deferred to the float-specialised executors (which exist).

`groupby_tier2_count_exec.rs:204-214` keeps a `_UNUSED_IMPORT_GUARDS` const to
hold otherwise-unused `cuda_sys`/`ptr`/`c_void` imports alive — dead weight;
prune the imports instead.

### 1.4 GPU reduction races / partition-merge

No host-visible race issues found at the executor layer. The two real race
hazards are correctly handled:
1. Cross-launch `atomicAdd` ordering for multi-column scatter → solved by
   deterministic `dest_idx` (F-positive above).
2. Open-addressing table overflow dropping rows → solved by the spill counter +
   soft fallback.
The actual atomic/CAS reduction correctness lives in `src/jit/*` (out of scope);
F2 is the one place where the kernel-level contract (float NaN) is not pinned by
the executor.

### 1.5 Duplication quantification & consolidation

Measured sizes (`wc`-equivalent from directory listing): the `groupby_tier2_*`
family is ~470 KB / ~13k LOC across 23 files; the `groupby_shmem_*` family adds
~110 KB / 6 files. The ~20 `try_execute` bodies share a near-identical skeleton:
plan-shape match → key/val downcast → NULL guard → `scan_max_nonneg_key` →
cardinality gate → per-call stream → partition → offsets → scatter → reduce →
spill check → slot-walk → sort → build batch. Each file also carries its own
`plan_schema_to_arrow_schema` wrapper, its own `KernelSpec` enum + `get_or_build_
module` + `partition_spec_for`, and its own `LOAD_COUNT`/cache tests.

The authors deliberately shared exactly one helper
(`groupby_tier2_common::scan_max_nonneg_key`) and documented (in both dispatchers)
why the rest is intentionally specialized: divergent eligibility tails,
non-interchangeable kernel ABIs, and the cross-module spill-sentinel string. That
argument is sound for the kernel-ABI-bearing core.

However, several pieces are genuinely identical and **safe** to consolidate
without touching any GPU ABI:
- `partition_spec_for(n_rows)` — byte-identical in every single-key executor
  (count/minmax/minmax_float/avg/orchestrator), modulo the i64 twins. Lift to
  `groupby_tier2_common`.
- The slot-walk + `sort_by_key` + RecordBatch-build tail for the
  `(key, value, set)` 13-byte output is structurally identical across
  count/minmax/minmax_float; a generic `collect_populated_slots<T>(host_keys,
  host_vals, host_set, num_partitions, block_groups) -> Vec<(i32, T)>` would
  remove ~40 lines × ~6 sites with no ABI risk.
- The per-file `plan_schema_to_arrow_schema` wrappers already all delegate to
  `schema_convert::plan_schema_to_arrow_schema_no_temporal` — the local wrappers
  are pure boilerplate and can be deleted in favour of calling the shared fn
  directly.

Net: an estimated 600–900 lines removable with zero behavioural change and zero
GPU verification needed. I'd stop short of unifying the `KernelSpec`/cache
machinery (the per-variant kernel sets genuinely differ) — the authors' caution
there is correct.

### 1.6 Performance suggestions

- **AVG fused reduce** (already noted in `tier2_avg_exec.rs:386-391`): a single
  sum+count reduce kernel (`atom.shared.add.u64` on a count slot alongside the
  value adds) would halve the per-partition hash/probe passes for AVG. Highest-
  value perf item.
- **Tier-2 SUM host slot-walk** (`groupby_tier2_orchestrator.rs:499-529`) is
  single-threaded over `NUM_PARTITIONS * BLOCK_GROUPS = 4096*1024 ≈ 4.2M` slots
  with a `Vec::new()` (no capacity hint) per partition. For high-cardinality q5
  this is a measurable host tail; parallelise with `rayon` over partitions and
  pre-size the per-partition vecs from `offsets[pid+1]-offsets[pid]`.
- **Repeated D2H of fixed 52 MiB output** regardless of cardinality: for queries
  with few groups most of the 4.2M slots are empty yet fully downloaded. A
  device-side compaction (stream-compact populated slots before D2H) would cut
  the dominant 52 MiB transfer for low-fill partitions.
- **`shmem_avg_exec` COUNT reuse** is good; consider also reusing the COUNT
  result to early-skip the present-map scan.
- Minor: `groupby_tier2_count_exec.rs` allocates a `dummy_vals_in` f64 buffer of
  `n_rows` purely to satisfy the scatter ABI (line 224). A keys-only scatter
  kernel variant would save `8·n_rows` bytes of alloc+H2D on every COUNT(*).

---

## 2. TEST ADEQUACY

### 2.1 Current coverage

Host-only unit tests are plentiful (per-file counts gathered): dispatch logic
(`*_dispatch.rs`: 15+10 tests, exhaustive boundary coverage), mergers
(`*_merge.rs`: 7-8 each, including empty/multi-partition/sort/arity), key packing
(`groupby_common.rs`: 14, incl. signed-zero, two-key, sentinel-collision),
Welford (6, incl. empty/single/combine-associativity), eligibility-gate rejection
paths (twokey/tier2 count & minmax: 10-14 each). E2E GPU tests exist as separate
files (`tier2_groupby_e2e.rs`, `tier2_twokey_e2e.rs`, `tier2_multi_sum_e2e.rs`,
`shmem_groupby_e2e.rs`, `aggregate_nulls_e2e.rs`) and the in-module GPU round-trip
smoke tests are `#[ignore]`-gated (run only with CUDA).

**Coverage estimate:** host-side logic (dispatch, packing, merge, finalize,
overflow) ~80-85%. GPU-path correctness is almost entirely behind `#[ignore]`
gates, so on a CPU-CI run the *executed* coverage of the actual aggregation
kernels is low (~30-40% effective). Edge-case **semantics** coverage (NULL
combinations, NaN, all-null groups, spill) is the weakest area.

### 2.2 Untested combos / tests to add (specific)

Correctness-driven (tie directly to findings):
1. **`twokey COUNT(col)` with NULLs declines** — would have caught F1. Add to
   `groupby_tier2_twokey_count_exec.rs` tests (host-only, asserts `None`).
2. **grouped float MIN/MAX with NaN == scalar MIN/MAX** — would expose F2. Add to
   `tier2_groupby_e2e.rs` / a new `groupby_float_nan_e2e.rs`.
3. **GROUP BY float key with multiple NaN payloads** — pin the F3 decision
   (currently nothing asserts the group count for NaN keys).
4. **AVG over all-NULL / empty group** — pin the 0.0-vs-NULL decision (F4) so a
   future planner change is forced through review.

Combinatorial gaps (aggregate × edge-shape) with no current test:
- **All-NULL group**: a group whose value column is entirely NULL for SUM / AVG /
  MIN / MAX (should be omitted / NULL). E2E, all four ops.
- **Empty input** for every executor: only merge + a couple of execs cover it;
  none of the twokey minmax/avg/multi execs test the 0-row batch through to
  output schema.
- **Single-row group** (count==1): exercises VAR_SAMP→NULL and the `set==1,
  count==0` defensive branch in `tier2_avg_exec.rs:529-536`.
- **Spill path actually firing**: only `groupby_tier2_orchestrator` has a
  pathological-input spill test; the count/avg/minmax/twokey spill branches are
  untested end-to-end. Add a shared pathological-partition-0 fixture and run each.
- **`COUNT(*)` vs `COUNT(col)` divergence under NULLs** for single-key shmem and
  tier2 (the guards exist but no test proves they fire).
- **Negative key / `Some(-1)` empty-sentinel** behaviour per executor (the common
  helper is tested, but the per-executor decline-vs-empty-batch branch is not).
- **Int64 MIN/MAX with `|v| > 2^53`** (the precision-fix path) — assert
  losslessness end-to-end; currently only documented, not tested.
- **Two-key SUM/COUNT where packed key == 0** (i.e. `(0,0)`) collides with the
  zeroed scatter buffer sentinel — verify it is not dropped (the merge test pins
  `(0,0)->0` packing but not the full GPU round-trip).

---

## 3. FEATURES / DIRECTIONS

1. **More aggregates on the fast path.** Today only SUM/COUNT/MIN/MAX/AVG are
   GPU-accelerated for GROUP BY; VAR/STDDEV fall to the host Welford scalar path
   and there is no *grouped* var/stddev fast path at all. A grouped
   per-block Welford reduce (emit `(count, mean, M2)` partials, merge with
   `WelfordState::combine` which already exists) would be a natural, well-scoped
   addition reusing the existing wire format.
2. **Approximate aggregates.** APPROX_COUNT_DISTINCT (HyperLogLog) and
   approximate quantiles (t-digest / KLL) fit the partition-then-reduce shape
   well and would let the engine answer high-cardinality DISTINCT without the
   current host distinct-count cap (`distinct.rs:420-427`). HLL registers reduce
   trivially across partitions.
3. **Spill-to-host / multi-pass for >100M groups.** Above `TIER2_MAX_GROUPS`
   (100M) everything falls to global-atomic; a recursive re-partition (partition
   the overflowing partition again) or a host-side spill of the over-full
   partitions would extend the fast path to arbitrary cardinality and remove the
   hard cap.
4. **Native validity in the partition/reduce kernels.** The single biggest
   limiter on fast-path coverage is that *any* NULL defers to global-atomic
   (and, per F1, sometimes incorrectly does not). A `_with_validity` partition +
   reduce kernel family (noted as "Stage G follow-up" in several files) would let
   NULL-bearing data stay on the GPU and would let the COUNT executors drop their
   conservative declines.
5. **Multi-key beyond 64 bits.** The composite-hash + host-verification fallback
   for >64-bit / 3+ column keys is documented as deferred (`groupby_common.rs:
   316-317`); implementing it would remove a common "not yet supported" error.
6. **Float key NaN canonicalisation** (ties to F3) to match DuckDB GROUP
   BY/DISTINCT, behind a session flag if strict IEEE `NaN != NaN` is also wanted.

---

## 4. Finding table (file:line · severity)

| ID | File:line | Sev | Summary |
|----|-----------|-----|---------|
| F1 | groupby_tier2_twokey_count_exec.rs:119-159 | HIGH | `COUNT(col)` over-counts NULLs; missing counted-column NULL guard present in all sibling COUNT execs |
| F2 | groupby_tier2_minmax_float_exec.rs, groupby_tier2_twokey_minmax_float_exec.rs | HIGH | GPU float MIN/MAX has no NaN handling/tests; disagrees with DuckDB-convention scalar path |
| F3 | groupby_common.rs:248-264; distinct.rs:31-35 | MEDIUM | NaN group/DISTINCT keys not collapsed → diverges from DuckDB |
| F4 | aggregate.rs:313-327; agg_with_pre.rs:450-461 | LOW | AVG returns 0.0 not SQL NULL for empty/all-NULL input |
| F5 | groupby_tier2_orchestrator.rs:85-91 | LOW | `stub_*` aliases name real kernels — misleading |
| F6 | groupby_tier2_count_exec.rs:204-214,224 | LOW | dead `_UNUSED_IMPORT_GUARDS`; dummy f64 scatter buffer for COUNT(*) |
| P1 | tier2_avg_exec.rs:386-391 | perf | fuse sum+count reduce (halves probe passes for AVG) |
| P2 | groupby_tier2_orchestrator.rs:499-529 | perf | single-threaded 4.2M-slot host walk; parallelise + presize |
| P3 | (orchestrators) | perf | always-52 MiB D2H regardless of fill; device-side compact before D2H |
| C1 | groupby_tier2_common.rs | cleanup | lift identical `partition_spec_for` + slot-walk tail + drop per-file schema wrappers (~600-900 LOC, zero ABI risk) |

---

## 5. Verification notes

- F1 verified by reading the full `try_execute` (no later counted-column guard
  exists) and confirming the executor is wired and reached before the
  global-atomic fallback (`groupby.rs:336` precedes `:359`), and by contrast with
  the two sibling COUNT executors that DO guard.
- F2/F3 verified by grepping the float-minmax execs and `load_key_column_bits`
  for any NaN handling (none) and reading the host `float_total_cmp` (correct,
  scalar-only).
- Welford, overflow, packing, signed-zero, spill, offset-monotonicity, and
  multi-scatter alignment were each read in full and cross-checked against their
  in-module tests; all correct as described.
- The apparent "mojibake" in some tier2 doc-comment headers was checked at the
  byte level and is valid UTF-8 (em-dashes) — **not** a finding.
