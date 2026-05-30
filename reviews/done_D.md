# done_D.md — GROUPBY/AGGREGATE correctness fixes (F1–F4)

Agent **D2** auditing & completing agent **D**'s stalled work. Branch `dev`.
Scope limited to `src/exec/` groupby*/aggregate/agg_with_pre/distinct family.
No `cargo build/check/test` run (tree already compiles per constraints; all edits
verified by inspection against the real code paths).

D's original edits landed in commit `3a54490` ("A,D partial, compiles"). D
produced no done_D.md (stalled mid-NaN-test), so its work was unverified.

---

## 1. Audit of what D had ALREADY done (confirmed CORRECT)

### F1 (HIGH) — two-key COUNT(col) over-counts NULLs
`groupby_tier2_twokey_count_exec.rs`:
- Added `Expr` import; captured `count_col_name` from `Count(Expr::Column(n))`
  (matching the single-key tier2 / shmem COUNT execs).
- Added the post-key-NULL-guard decline: NULL-bearing counted column → `return None`.
- Added tests `rejects_null_bearing_counted_column` and
  `count_star_not_declined_by_counted_column_guard`.
- **Verified consistent** with `groupby_tier2_count_exec.rs:120-153` (same guard
  shape) and `groupby_shmem_count_exec.rs`.
- **Dispatch wiring confirmed**: `groupby.rs:336` calls the twokey count exec via
  `try_fast_path!`, which falls through on `None` to the always-correct
  global-atomic path at `:359+`. Decline is honored. (groupby.rs not edited.)

### F2 (HIGH) — grouped float MIN/MAX NaN semantics
- `groupby_tier2_minmax_float_exec.rs` (single key): D added the NaN-deferral
  guard (`Float64Array … any(is_nan) → None`) + host-only `nan_tests` module
  (`nan_value_defers_min_and_max`, `finite_values_not_declined_by_nan_guard`).
- `groupby_tier2_twokey_minmax_float_exec.rs` (two key): D added the **guard
  only** — no tests (this was the gap; see §2).
- **Variant audit (key finding):** the ONLY float-value-handling execs are these
  two `*_minmax_float_exec.rs` files. Confirmed by reading:
  - `groupby_shmem_minmax_exec.rs:64-68` — float value `=> return None` (deferred).
  - `groupby_tier2_minmax_exec.rs` — Int32/Int64 only.
  - `groupby_tier2_twokey_minmax_exec.rs:186-191` — float routes to the float
    sibling (has a `rejects_float_value_column` test).
  So F2's "all variants" = exactly these two files, both now guarded. No shmem
  float minmax executor exists to patch.

### F3 (MEDIUM) — NaN GROUP BY / DISTINCT key collapse
- `groupby_common.rs::canonicalise_f64/f32` now fold every NaN → single canonical
  quiet NaN (`f64::NAN`/`f32::NAN`); doc + `load_key_column_bits` doc updated.
- `distinct.rs::canonicalise_f64/f32` folded identically; module/RowKeyValue docs
  updated; old `distinct_nan_canonicalisation_is_noop` test REPLACED with
  `distinct_nan_canonicalisation_collapses_all_nan` and new end-to-end
  `distinct_collapses_multiple_nan_payloads`.
- `groupby_wide.rs::canonicalise_f64/f32` kept in lock-step (delegating copy; no
  tests — it's the wide-key path mirror).
- All four float key operators (GROUP BY, DISTINCT, wide GROUP BY) now share one
  equivalence relation. (JOIN's copy is out of scope.)

### F4 (LOW) — AVG empty/all-NULL → NULL
- `aggregate.rs`: new shared `pub(crate) fn avg_result_array(sum,count,nullable)`
  → NULL when `count==0 && nullable`, else 0.0 fallback (non-nullable), else
  `sum/count`. Replaces inline 0.0 logic in the scalar path.
- `agg_with_pre.rs`: pre-stage AVG path routes through the same helper.
- Tests `avg_empty_input_is_null_when_nullable`,
  `avg_empty_input_is_zero_when_non_nullable`, `avg_non_empty_divides`.

---

## 2. What I (D2) COMPLETED

D had finished F1, F3, F4 (guards + tests) and the F2 **guards** for both float
execs, but left the twokey float exec **untested** and had no grouped-vs-scalar
GPU equivalence test. I completed:

1. `groupby_tier2_twokey_minmax_float_exec.rs` — added the missing F2 tests:
   - `rejects_nan_value_column_min_and_max` (host-only; 300K-row batch with one
     NaN, both MIN and MAX → asserts `try_execute` returns `None`).
   - `finite_values_not_declined_by_nan_guard` (sub-threshold sanity).

2. `groupby_tier2_minmax_float_exec.rs` — added a GPU-gated grouped==scalar test:
   - `grouped_float_minmax_matches_scalar_reference` (`#[ignore = "gpu:tier2"]`)
     runs the fast path for finite data (mixed signs) for both MIN and MAX and
     asserts each group equals a per-group host scalar reduction — pins the
     "one answer per query" invariant. (NaN equivalence is established by the
     host deferral tests + the scalar `float_total_cmp` tests already in
     `aggregate.rs`, since NaN-bearing inputs are deferred to that scalar path.)

3. `groupby_common.rs` — added the GROUP-BY half of the F3 collapse proof
   (D had only the DISTINCT half in `distinct.rs`):
   - `pack_keys_nan_collapse_f32` and `pack_keys_nan_collapse_f64`: distinct NaN
     payloads + signalling NaN + negative-NaN + plain NaN all `pack_keys` to the
     same `keys_i64` (== `f{32,64}::NAN.to_bits()`), while a finite value stays a
     distinct key.

---

## 3. Variants touched (final state)

| File | F | Change |
|------|---|--------|
| groupby_tier2_twokey_count_exec.rs | F1 | guard + 2 tests (D) |
| groupby_tier2_count_exec.rs | F1 | pre-existing guard (audited consistent) |
| groupby_shmem_count_exec.rs | F1 | pre-existing guard (audited consistent) |
| groupby_tier2_minmax_float_exec.rs | F2 | guard + nan_tests (D) **+ GPU grouped==scalar test (D2)** |
| groupby_tier2_twokey_minmax_float_exec.rs | F2 | guard (D) **+ 2 NaN tests (D2)** |
| groupby_common.rs | F3 | canonicalise NaN-fold (D) **+ 2 pack-keys collapse tests (D2)** |
| distinct.rs | F3 | canonicalise NaN-fold + 2 tests (D) |
| groupby_wide.rs | F3 | canonicalise NaN-fold lock-step copy (D) |
| aggregate.rs | F4 | avg_result_array helper + 3 tests (D) |
| agg_with_pre.rs | F4 | route AVG through helper (D) |

float-minmax variant coverage = COMPLETE: the only two float-value execs both
defer NaN; all other minmax execs are integer-only or defer floats to these two.

---

## 4. Test list (all in-module `#[cfg(test)]`)

Host-runnable:
- twokey_count: `rejects_null_bearing_counted_column`,
  `count_star_not_declined_by_counted_column_guard`
- tier2_minmax_float: `nan_value_defers_min_and_max`,
  `finite_values_not_declined_by_nan_guard`
- twokey_minmax_float: `rejects_nan_value_column_min_and_max`,
  `finite_values_not_declined_by_nan_guard`
- groupby_common: `pack_keys_nan_collapse_f32`, `pack_keys_nan_collapse_f64`
- distinct: `distinct_nan_canonicalisation_collapses_all_nan`,
  `distinct_collapses_multiple_nan_payloads`
- aggregate: `avg_empty_input_is_null_when_nullable`,
  `avg_empty_input_is_zero_when_non_nullable`, `avg_non_empty_divides`
- (pre-existing, audited) aggregate scalar NaN: `f64_min_max_finalize_nan_convention`,
  `f32_min_max_finalize_nan_convention`, `f64_min_max_finalize_all_nan_is_nan`,
  `f64_min_max_finalize_no_nan_unchanged`

GPU-gated (`#[ignore = "gpu:tier2"]`):
- tier2_minmax_float: `grouped_float_minmax_matches_scalar_reference` (new),
  `async_tier2_minmax_float_round_trip` (pre-existing)

---

## 5. Notes / not done

- A grouped float MIN/MAX **NaN** end-to-end test through `Engine::sql` would
  live in `tests/tier2_groupby_e2e.rs` (or a new `groupby_float_nan_e2e.rs`),
  which is OUTSIDE the permitted `src/exec/` edit scope, so it was not added.
  The contract is nonetheless pinned: NaN inputs are deferred (host tests) to the
  scalar path whose NaN total order is tested in `aggregate.rs`.
- F4 remains 0.0 when the output field is non-nullable (legacy contract); full
  NULL surfacing needs a planner-side nullability change (out of scope, noted in
  the helper doc).
- Did not touch engine.rs / lib.rs / mod.rs / jit/* per constraints. groupby.rs
  read-only (dispatch confirmed, not edited).
