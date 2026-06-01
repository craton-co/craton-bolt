# Bolt Core Execution Engine Review — `src/exec/` (core, non-groupby/non-string)

Scope: engine.rs, filter.rs, join.rs, gpu_join.rs, sort.rs, gpu_sort.rs, aggregate.rs,
expr_agg.rs, extended_agg.rs, compact.rs, gpu_compact.rs, gpu_compact_multipass.rs,
limit.rs, setops.rs, distinct.rs, window.rs, streaming.rs, gpu_table.rs, gpu_upload.rs,
schema_convert.rs, subquery_resolve.rs, validity_audit.rs, welford.rs, launch.rs,
module_cache.rs, partition_offsets.rs, dict_registry.rs, agg_with_pre.rs, mod.rs.

Overall the module is in good shape: many historical correctness bugs are documented and
fixed (signed-zero/NaN canonicalisation, NOT IN NULL footgun, i64 SUM overflow, u32
launch-shape overflow, validity preservation in compaction). Findings below are ranked by
severity. No code was edited.

---

## 1. CODE REVIEW

### HIGH

**H1 — JOIN on dictionary-encoded Utf8 keys can compare across independent dictionary
index spaces (potential wrong results).**
`join.rs:1148-1191` (`extract_key_value`) keys a `Dictionary(Int32|Int64, Utf8)` column on
the *dictionary index* (`JoinKeyValue::DictIdx(i32)`). The type-level doc-comment
(`join.rs:930-947`) only argues that `DictIdx` and `Utf8` variants never collide; it does
**not** address that the build side and probe side come from two *different* RecordBatches,
each with its **own** dictionary. Equal indices across two independent dictionaries do not
imply equal strings (e.g. build dict `["US","EU"]` vs probe dict `["EU","US"]` → index 0
means "US" on one side and "EU" on the other). `check_key_dtypes` (`join.rs:771-789`) only
checks Arrow dtype equality, not dictionary-content equality, so it cannot catch this.
Reachability: the common SQL path downloads string columns as decoded `StringArray`
(`engine.rs:4229-4232`, `DeviceCol::Utf8 -> to_string_array()`), keyed as `Utf8(Box<str>)`,
which is safe. The bug is reachable only when a join child yields a *raw* Arrow
`DictionaryArray` (native dict ingest is live — `dict_registry.rs:117-139`,
`flatten_dictionary_utf8_columns` is a deprecated no-op per `engine.rs:4486`). Recommend:
decode dict keys to the underlying string for join keying, OR poison/reject `DictIdx` keys
unless both sides provably share one dictionary (mirror the F-7 poisoning already done in
`dict_registry.rs:219-291`). At minimum add a two-dict-tables join test with differing
dictionary orderings.

**H2 — EXCEPT / INTERSECT and window functions have no end-to-end execution test
coverage.** See TESTS section; flagged HIGH because both are full executors
(`setops.rs`, `window.rs`) whose only coverage is in-module unit tests + parser tests. A
lowering/dispatch regression would ship silently.

### MEDIUM

**M1 — `i32` scalar SUM finalize uses `wrapping_add`, inconsistent with the documented
"never silently wrong" SUM contract.** `aggregate.rs:1746`
(`impl ReduceScalar for i32 :: finalize`, `ReduceOp::Sum` arm) folds with
`i32::wrapping_add`, while the i64 path (`aggregate.rs:1788-1806`) correctly errors via
`checked_add`. Today this i32 arm is *dead* for SUM because `SUM(Int32)` always widens to
i64 (`aggregate.rs:673-706` routes through `reduce_gpu_vec_widened::<i32,i64>`), so the
result is correct in practice — but it is a latent landmine: any future path that finalizes
an i32 SUM directly would silently wrap. Make it `checked_add` (error) or add a
`debug_assert!(!matches!(op, Sum))` to document the dead arm.

**M2 — `GatheredCol::download` issues a synchronous per-column D2H (`to_vec`) with no
pinned/async overlap.** `gpu_compact.rs:206-255`: each variant calls `v.to_vec()`
(synchronous D2H + implicit context-sync), and `BoolNullable` does two separate
synchronous D2Hs plus a host zip (`gpu_compact.rs:231-251`). The engine calls `download()`
per output column in a loop, so an N-column projected+filtered result pays N full
host/device round trips that serialize. The async/pinned helper already exists
(`gpu_upload.rs::download_to_host_pinned`) and the filter-mask path already uses pinned D2H
(`compact.rs:61-96`). Migrating the gather download to pinned-async + one `synchronize`
across all columns would recover the overlap this module otherwise advertises.

**M3 — Window executor materialises every PARTITION/ORDER key and the aggregate input into
full owned `Vec<Option<T>>` host buffers, even for tiny frames.** `window.rs:679-893`
(`KeyColumn::extract`, `NumericColumn::extract`) build `Vec<Option<T>>` over the entire
column (and `Vec<Option<String>>` cloning every Utf8 cell) up front; the running aggregate
then re-walks per partition. For wide inputs this is O(n) extra allocation on top of the
lexsort permutation. Acceptable for the "correctness first" host path, but worth a note as
the largest allocation hot-spot in the host operators.

**M4 — `should_use_gpu_sort` / `try_gpu_sort` documentation is stale and partly
contradictory.** `sort.rs:199-213` documents Gate 1 of `try_gpu_sort` (the bitonic path) as
"Env opt-in `BOLT_GPU_SORT=1`, default OFF", but the actual dispatch
(`execute_sort` at `sort.rs:149-164`) selects the GPU path via the `should_use_gpu_sort`
heuristic and `try_gpu_sort` has **no** env gate — only `try_gpu_sort_radix`
(`sort.rs:224`) reads the env. The stale doc could mislead a future maintainer into
thinking the bitonic path is still opt-in. Doc-only, but a real hazard for the dispatch
logic.

**M5 — `extended_agg` / Bool/Utf8 scalar aggregates and the host DISTINCT/SETOP paths are
unbounded in row width but only DISTINCT has a memory cap.** `distinct.rs:84-136` has the
`CRATON_DISTINCT_HOST_MAX_ROWS` guard (good). `setops.rs` (`build_key_counts`,
`execute_setop`) builds an unbounded `HashMap<RowKey,usize>` over the right input with no
cap — a large right side is a host-memory DoS surface analogous to the one DISTINCT closed.
Consider sharing the DISTINCT cap with the set-op key map.

### LOW

**L1 — `module_cache` warm-path panics in tests only** (`module_cache.rs:1962-1963,
2834-2835`) — these are test closures, not production. No action.

**L2 — `welford::combine` is numerically fine but `stddev_*` clamps variance with
`.max(0.0)` before sqrt** (`welford.rs:125,131`). Correct (guards tiny negative round-off),
just worth knowing the population stddev can read `0.0` for a near-constant column.

**L3 — `n_rows as u64` in `execute_cross_join`** (`join.rs:459-460`) is safe (usize→u64
widening) and the product is `checked_mul`’d (`join.rs:469`). No overflow. Mentioned only
because the CROSS cap logic is load-bearing and correct.

**L4 — `streaming.rs` is pure host scaffolding not yet wired to the device.** `PinnedBudget`,
`MorselPlan`, `BatchStream` are well-tested but the device-pinned allocation is a
`TODO(cuda)` (`streaming.rs:382-387`) and `plan_upload` is not consulted by the engine’s
`materialize_table` yet. Larger-than-VRAM execution is therefore still not actually bounded
at runtime — the module name oversells current capability.

**L5 — `window.rs` aggregate `Repr` always allocates `valid: Vec<bool>` and `values` of
`n_rows`** even for ranking functions that never produce NULLs (`window.rs:574-587`). Minor.

**L6 — `KeyColumn::eq_rows` float equality is `to_bits()==to_bits() || x==y`**
(`window.rs:784-788`). This makes distinct NaN bit-patterns in a PARTITION BY key compare
*unequal* (fragmenting partitions), which is inconsistent with the DISTINCT/GROUP BY engine
convention that folds all NaN to one key (`distinct.rs:447-455`). Edge case (NaN partition
keys), but a cross-operator semantic inconsistency worth aligning.

---

## 2. TESTS

**In-module unit tests are strong** for the host operators: filter.rs, limit.rs, setops.rs,
distinct.rs, sort.rs (dispatch heuristic), window.rs, streaming.rs, welford.rs,
validity_audit.rs, compact.rs, dict_registry.rs all carry focused unit tests including
empty-input, single-row, NULL, signed-zero/NaN, and overflow cases. aggregate.rs/window.rs
explicitly pin the i64-SUM-overflow and >2^53 exactness contracts.

**Integration coverage in `tests/`:**
- JOIN: well covered — `joins_e2e.rs` (10), `gpu_join_e2e.rs` (27), `non_equi_join_test.rs`
  (12), CROSS in `joins_e2e.rs`/`gpu_join_e2e.rs`.
- SORT: `sort_e2e.rs` (27) — good.
- Scalar aggregate / NULLs / STDDEV: `aggregate_nulls_e2e.rs` (23), `stddev_e2e.rs` (18),
  `aggregate_aliasing_test.rs`, `post_aggregate_exprs_test.rs`, `having_test.rs` — good.
- Subquery: `e2e_tests.rs`, `semantics_e2e.rs`, `in_list_test.rs`, `diff_duckdb_semantics.rs`
  — good, including NOT IN.
- Semantics vs DuckDB: `diff_duckdb.rs`, `diff_duckdb_semantics.rs`, `semantics_e2e.rs`.

**Coverage GAPS (test enhancements needed):**
1. **EXCEPT / INTERSECT (`setops.rs`)** — only parser tests exist (`grep` shows
   EXCEPT/INTERSECT only in `parser_tests.rs`); no e2e SQL execution test and none in
   `diff_duckdb*`. Add ALL/distinct multiset cases, NULL-equality across inputs, and a
   `diff_duckdb` parity test.
2. **Window functions (`window.rs`)** — no `OVER` clause appears in any `tests/` file except
   `wave3_regression_test.rs`. Add e2e tests for ROW_NUMBER/RANK/DENSE_RANK with ties,
   running SUM/AVG/MIN/MAX under the RANGE frame, multi-partition, and NaN/NULL partition
   keys (covers L6).
3. **DISTINCT e2e** — exercised only inside `e2e_tests.rs`/`diff_duckdb.rs`; no dedicated
   file covering the host-cap env var or multi-column/NaN dedup at the SQL level.
4. **JOIN on dictionary-encoded keys (H1)** — no test joins two tables whose dict-encoded
   Utf8 join columns have *different* dictionary orderings. This is the test that would
   expose H1.
5. **Larger-than-VRAM / morsel path (`streaming.rs`)** — unit-tested in isolation but no
   integration test drives a query through morsel chunking (because it isn't wired yet).

Estimated module coverage: host operators ~75-85% (unit-heavy); EXCEPT/INTERSECT and window
**execution** coverage effectively ~0% at the integration level; GPU join/sort/compact paths
are gated behind `#[ignore = "gpu:*"]` so they only run on a real device.

---

## 3. NEW FEATURES / DIRECTIONS

1. **Wire `streaming.rs` to the device.** `plan_upload`/`MorselPlan`/`PinnedBudget` exist
   and are tested; the engine’s `materialize_table` still uploads whole tables
   (`streaming.rs:6-12` admits this). Implementing the `TODO(cuda)` pinned-buffer backing
   and consulting `plan_upload` in the scan/upload path would unlock larger-than-VRAM
   queries — the single biggest capability gap.
2. **GPU DISTINCT / SET-OP via sort.** Both `distinct.rs` and `setops.rs` are host-only and
   capped; the sort-based GPU variant is already referenced as deferred
   (`distinct.rs:52-56`). `gpu_sort::sort_indices_on_gpu_multi` exists, so this is mostly
   plumbing input columns as `GpuVec`s.
3. **GPU / pushed-down window functions.** `window.rs:11-18` notes the host-only design is
   intentional; a GPU partition+segmented-scan kernel behind the same `PhysicalPlan::Window`
   node is the natural next step.
4. **Non-default window frames** (`ROWS`/`GROUPS`, explicit bounds) — currently only the
   default RANGE frame is implemented and the frontend rejects others (`window.rs:37-44`).
5. **Computed join/sort/window keys.** All three reject non-bare-column keys
   (`join.rs:1235-1244`, `sort.rs:424-433`, `window.rs:921-930`); supporting `ORDER BY a+b`
   / `JOIN ON f(a)=g(b)` is a recurring limitation.
6. **OUTER + non-equi join** is explicitly rejected (`join.rs:610-619`); a streaming
   nested-loop with preserved-side tracking would complete the non-equi surface.
7. **Cross-dtype equi-join** is rejected (`join.rs:781-787`); implicit numeric widening on
   join keys would match SQL expectations.
8. **Decimal128 / temporal aggregate inputs** beyond SUM — `aggregate.rs:756-763` rejects
   Bool/Utf8/Decimal/Date/Timestamp for several reduction ops; `schema_convert`’s
   `_no_temporal` guards (`schema_convert.rs:91-102`) confirm temporal types are not wired
   through GROUP BY/join output schemas yet.
