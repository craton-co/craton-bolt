# Core Operator + Engine Review — `craton-bolt` (`src/exec/` + root src)

**Reviewer:** senior GPU-database reviewer
**Scope:** `src/exec/{engine, streaming, join, gpu_join, sort, gpu_sort, filter, limit, setops, launch, compact, gpu_compact, gpu_compact_multipass, gpu_table, gpu_upload, module_cache, mod, window}.rs` + root `src/{lib, error, metrics, observability}.rs`
**Date:** 2026-05-30
**Verdict:** The host-side operator semantics are unusually careful and well-tested (NULL=NULL handling, signed-zero/NaN canonicalisation, integer-exact window aggregates, multiset set-ops, OUTER-join unmatched tracking). The GPU paths consistently treat themselves as *accelerators with a correctness fallback* (`Ok(None)` / `GpuCapacity` → host), which is the right architecture. The findings below are mostly robustness, observability-consistency, and resource-lifetime concerns rather than correctness defects in the steady-state SQL path. A few genuine gaps (unbounded module cache, metrics double-/under-counting, `run_logical_plan` not counted) are worth fixing.

Severity legend: CRITICAL (wrong results / UB / crash on a normal path), HIGH (wrong results or resource leak on a reachable edge path), MEDIUM (robustness / observability / perf-correctness), LOW (polish / minor perf).

---

## 1. CODE REVIEW

### 1a. Operator correctness — findings

#### F1 — MEDIUM — `module_cache.rs:128` — `GLOBAL_MODULE_CACHE` is an unbounded `HashMap` with no eviction
`static GLOBAL_MODULE_CACHE: Lazy<Mutex<HashMap<Key, CudaModule>>>` (line 128) grows without bound. Every other cache family in this file (`KERNELSPEC_CACHE`, `SCALARAGG_CACHE`, `HASHJOIN_CACHE`, `RADIXSORT_CACHE`, `COMPACTION_CACHE`) uses the FIFO `SpecCache` with `cap = 256` and a `VecDeque` eviction log (see `KERNELSPEC_CACHE_CAP` at line 272 and `SpecCache::insert` at 480). The string-keyed global cache is the one exception.
- Because the key folds in `\0dev{N}` *and* the caller's `spec_id = format!("{:?}", spec)`, a workload that JITs many distinct kernel shapes (wide GROUP BY tiers, many distinct projection specs, multi-GPU) accumulates `CudaModule` handles forever. Each entry pins an `Arc<CudaModuleInner>` (a loaded cubin) on the device — a slow VRAM/host leak over a long-lived process.
- **Fix:** give `GLOBAL_MODULE_CACHE` the same bounded `SpecCache`/FIFO treatment (a `VecDeque<Key>` + cap), or migrate these call sites onto the existing `SpecCache`. At minimum document and cap it.

#### F2 — MEDIUM — `engine.rs:2078` vs `engine.rs:2213` — `run_logical_plan` / `run_subplan` bypass the query counters
`Engine::sql` bumps `Counter::QueriesTotal` (line 2078) and `QueriesFailed` (2161). `Engine::run_logical_plan` (DataFrame `collect` path, line 2213) and `run_subplan` (line 2285) do **not** bump either. Consequences:
- A DataFrame-driven workload reports `queries_total = 0` while doing real work — the dashboards documented in `lib.rs` and `metrics.rs` undercount.
- Subquery resolution runs full sub-plans through `run_subplan` → `execute`; their failures never reach `QueriesFailed`. Conversely a single `sql()` call containing N subqueries still counts as one query, which may be the intent, but the asymmetry is undocumented.
- **Fix:** decide the contract (count top-level queries only vs. all plan executions) and apply it uniformly; at least bump `QueriesTotal`/`QueriesFailed` in `run_logical_plan`.

#### F3 — LOW — `metrics.rs:93-95` / `bucket_index` overflow-bucket doc is off-by-one
The doc on `HISTOGRAM_BUCKETS` (line 95) says the final overflow bucket "means `≥ 2^22 µs`", but with `HISTOGRAM_BUCKETS = 24` the last finite bound is `2^23` and `bucket_index` saturates at index 23 for anything `> 2^23` (verified by `bucket_index_overflow_saturates`, which checks `1<<23`). The prose says `2^22`. Cosmetic, but it is load-bearing documentation for a Prometheus exporter author.
- **Fix:** change "≥ 2^22 µs" to "≥ 2^23 µs" (or "> 2^22", matching whichever bound is intended). The code is correct; only the comment is wrong.

#### F4 — LOW — `join.rs` non-equi (nested-loop) join silently drops the GPU path and is INNER-only
`execute_nested_loop_join_chunked` (join.rs:599) rejects LEFT/RIGHT/FULL + non-equi with a clear `BoltError::Plan` (good — no silent wrong results), and caps the inner side at `MAX_NESTED_LOOP_INNER_ROWS = 1024`. This is documented, but it is a real feature gap: `t1 LEFT JOIN t2 ON t1.a = t2.a AND t1.b > t2.b` errors rather than executes. Not a correctness bug; flagged as a known gap for the feature list (§3). The streaming-chunk fix (EXEC-M3) is correct and well-tested.

#### F5 — LOW — `sort.rs` — `try_gpu_sort_radix` gate 1 (env opt-in) contradicts the module-level "planner-driven" claim
The module docs (sort.rs:11-35) say the GPU path is now planner-driven via `should_use_gpu_sort`, and `BOLT_GPU_SORT` is "purely a force override". But `try_gpu_sort_radix` still has an independent `gpu_sort_env_enabled()` gate (line 224, "Gate 1: env opt-in") that is *off by default* (see the test `radix_dispatch_skipped_when_env_off`). So the radix path is opt-in, while the bitonic `try_gpu_sort` path is planner-driven. The result: with the default env, a large Int32 sort takes the bitonic kernel, never radix, even though `should_use_gpu_sort` returned true. This is intended ("opt-in until benched in") but the two override semantics (`gpu_sort_override` tri-state in sort.rs vs `gpu_sort_env_enabled` in `sort_kernel_radix`) read the *same* `BOLT_GPU_SORT` var with *different* rules, which is confusing and a latent footgun.
- **Fix:** unify on one parser, or rename the radix gate's env var so the two don't alias.

#### F6 — INFO (verified correct) — JOIN NULL=NULL semantics
`extract_key`/`extract_key_value` (join.rs:1064, 1092) return `Ok(None)` on any NULL key cell; the build map skips NULL-key rows and the probe drops them — so `NULL = NULL` never matches (INNER drops both sides; OUTER still emits the preserved side NULL-padded, lines 397-425). The GPU paths gate on `null_count() == 0` (join.rs:1348-1357, 1535-1544) and fall to host otherwise. `verify_pairs_on_host` (gpu_join.rs:3218) also returns `false` if either side is NULL. Float keys canonicalise `-0.0 → +0.0` and preserve NaN bits so NaN never self-matches. **This is correct and consistent across join/distinct/groupby/setops.** No action.

#### F7 — INFO (verified correct) — Set-op arity + dedup + NULL handling
`setops.rs:73` re-checks column count defensively, `RowKey` (shared with DISTINCT) treats two NULLs as equal (engine "NULLs are not distinct" convention), the `ALL` multiset arithmetic decrements `right_counts` in place (`EXCEPT ALL = max(0, lc-rc)`, `INTERSECT ALL = min(lc,rc)`), and left first-occurrence order is preserved via `arrow::compute::filter`. **Correct, well-tested.** One observation: type *unification* (cross-dtype set-ops, e.g. Int32 EXCEPT Int64) is delegated to the logical planner — `setops.rs` only checks column *count*, not per-column dtype. If a hand-built physical plan passes mismatched dtypes, `RowKey::from_values` / `ColumnReader::new` would mis-key or error rather than coerce. Acceptable given the planner contract, but worth an assertion (see F12).

#### F8 — INFO (verified correct) — Window frames + ranking + integer-exact aggregates
`window.rs` implements only the SQL default frame (`RANGE UNBOUNDED PRECEDING .. CURRENT ROW`), documented at line 37, and the frontend rejects explicit frames — so no silent wrong frame. `RANK`/`DENSE_RANK`/`ROW_NUMBER` peer-group logic is correct (lines 260-289). Integer SUM/MIN/MAX stay on the i64 lane (no f64 round-trip past 2^53 — tested at line 1267), SUM overflow errors rather than wraps (line 460), float MIN/MAX use the DuckDB NaN-as-largest convention. **Correct.** Limitation: window keys/inputs must be bare columns (`bare_column_name`, line 921) — computed PARTITION BY / ORDER BY errors. Flagged for §3.

#### F9 — INFO (verified correct) — LIMIT/OFFSET
`limit.rs` handles `offset >= n_rows` (empty slice via `batch.slice(n_rows, 0)`), `limit > remaining` (clamp to `n_rows - offset`), `limit == 0`, and empty input. Zero-copy `RecordBatch::slice`. **Correct and fully tested.**

### 1b. Engine orchestration — error handling, fallback, resource cleanup

#### F10 — MEDIUM — `engine.rs:2872-2950` — predicate/projection path: no explicit cleanup ordering on error, relies on `Drop`-fence; double H2D round-trip for Decimal128 validity
Two sub-findings in `execute_projection`:
- **(a) Resource cleanup on error is correct-by-construction but subtle.** The kernel launch (line 2843) uses `cuLaunchKernel` directly (not `launch_1d`), so it does **not** call `KernelArgs::tag_launch_stream` (the V-1 stream-fence enforcement point described in launch.rs:251-261). Safety here therefore *does* depend on the `stream.synchronize()` inside the downstream `download_pinned` / `gpu_compact` calls and the per-call `debug_sync_check()`. If a future edit removes those syncs (as the V-1 doc says is safe for `launch_1d` callers), this hand-rolled launch would reintroduce a use-after-free because the output `DeviceCol`s could be recycled while the kernel is in flight. The `launch_1d` path is hardened; this bespoke launch is not.
  - **Fix:** route this launch through `launch_1d`/`launch_with_geometry` (it can't today because `KernelArgs` is monomorphic per push and the columns are heterogeneous — see the comment at 2736), OR replicate the `tag_launch_stream` call for the raw-launch path so the Drop-fence invariant holds uniformly.
- **(b) Perf — redundant D2H+H2D for Decimal128 passthrough validity.** Lines 2701-2717 download the source column's validity mask to host (`src_mask.to_vec()`) and immediately re-upload it (`GpuVec::<u8>::from_slice(&bits)`) to give the output column an owned mask. That is a full mask round-trip per query for every passthrough Decimal128 column. A device-to-device copy (or sharing the source mask buffer by `Arc`/refcount) would avoid the host bounce. Same pattern flagged below for the Utf8 dictionary clone (which is at least host-side already).

#### F11 — MEDIUM — `gpu_table.rs:535` — Date32/Timestamp upload is an unimplemented gap that surfaces as a hard error, not a host fallback
`GpuColumn::from_arrow` returns `BoltError::Type("Date/Timestamp upload not yet supported")` for `Date32`/`Timestamp`. Unlike the join/sort GPU gates (which return `Ok(None)` and fall to host), a projection over a Date/Timestamp column has **no host fallback at this layer** — `execute_projection` → `ensure_gpu_table` will propagate this error and fail the whole query. Meanwhile `window.rs:706,714` *does* support Date32/Timestamp(ns) host-side, so the engine's capability surface is inconsistent: you can `... OVER (ORDER BY ts)` but cannot `SELECT ts FROM t` if it routes through the GPU projection path.
- **Fix:** either add host-side projection passthrough for temporal columns (they don't need GPU compute for a bare passthrough) or ensure the lowerer never routes a temporal-column projection to `PhysicalPlan::Projection`. Verify which path real `SELECT ts` queries take.
- Similarly `gpu_compact.rs:981,994` reject Decimal128 and Date/Timestamp gather with "coming in a follow-up" — same fallback-gap concern for filtered projections over those types.

#### F12 — LOW — `setops.rs` / `window.rs` / `join.rs` — defensive dtype asserts missing on hand-built-plan paths
Several executors document "the planner already guaranteed X" and then trust it. `setops` checks column count but not dtype (F7). `execute_join`'s `check_key_dtypes` (join.rs:771) *does* check — good. For symmetry and defence-in-depth on the SetOp path (which the module comment explicitly worries about: "so a hand-built physical plan can't silently mis-key rows"), add a per-column dtype equality check in `build_key_counts` / `execute_setop`.

#### F13 — LOW — `engine.rs:2168` — `maybe_emit_pool_stats` runs after `execute` even on the error path, but the observer can panic
`sql()` calls `self.maybe_emit_pool_stats(...)` unconditionally (line 2168), which eventually calls `observability::notify_observers`. The observability module is hardened against a panicking observer (parking_lot mutex, lock dropped before callback — observability.rs:128, tested). Good. But note `notify_observers` **propagates** the observer's panic to the caller (the test `panicking_observer_does_not_disable_surface` asserts `result.is_err()` via `catch_unwind`). In `sql()` there is no `catch_unwind`, so a panicking user observer would unwind out of `Engine::sql` *after* the query succeeded, turning a successful query into a panic. The doc at engine.rs:2163-2167 claims "internal errors in the emit path are swallowed — they must never escalate to the query result", but a panic is not swallowed.
- **Fix:** wrap the observer invocation in `std::panic::catch_unwind` inside `notify_observers` (or in `maybe_emit_pool_stats`) and log, to honour the documented "never escalate" contract.

#### F14 — INFO (verified correct) — host fallback wiring for joins/sorts/cross
The `try_gpu_*` functions return `Ok(None)` on gate miss and `Err(_)` only on hard GPU faults; `execute_inner_join`/`execute_outer_join` propagate `Err` but treat `Ok(None)` as "fall through to host hash join". The `GpuCapacity` variant (error.rs:114) is the typed marker for "output buffer overshoot → retry host", correctly emitted by the GPU join kernels (gpu_join.rs:1371, 1498, 1751, 1915) and mapped by callers. CROSS GPU decline logs at debug and falls to host (join.rs:496). **This fallback architecture is sound.** One nit: `try_gpu_inner_join` returns `Err` for hard faults but the Stage-1/AoS sub-paths *catch* their errors and fall through to Stage-2 (join.rs:1383, 1462) — so a genuine CUDA fault in Stage-1 is masked and retried on Stage-2 rather than surfaced. That is intentional (defence in depth) but means a real device fault is only surfaced if the *last* stage also fails; worth a one-line note in the docstring.

### 1c. Concurrency in metrics / observability

#### F15 — INFO (verified correct) — metrics atomics
`metrics.rs` uses `Relaxed` atomics for independent counters/histograms; the snapshot is explicitly documented as not a globally-consistent instant (standard scraping contract). `bucket_index` is overflow-safe (`leading_zeros`-based, tested across boundaries incl. `u64::MAX`). `observe_duration` saturates to `u64::MAX` rather than wrapping. The `Metrics` singleton is `Lazy` and returns a stable `&'static`. **No races; correct.**

#### F16 — LOW — `observability.rs:81` — single-slot observer registry: install-races and teardown
`REGISTRY: OnceLock<Mutex<Option<ObserverHandle>>>` is a single global slot; `install_pool_stats_observer` overwrites whatever was there. The doc acknowledges "a second install overwrites the first". Two concerns:
- **No teardown.** There is no `uninstall`; the documented idiom is "install a no-op" (the test does this at line 183 to avoid leaking an observer between tests). In a library embedded in a larger process this means an observer (and anything it captures, e.g. an `Arc` to a metrics registry) lives for the process lifetime with no way to drop it. Consider returning the previous observer from `install` (so callers can restore) or adding an explicit `uninstall`.
- **Re-entrancy is handled** (lock dropped before callback — good), and **panic propagation** is a deliberate choice but see F13.

#### F17 — INFO (verified correct) — launch.rs stream lifetime / Drop
`CudaStream::Drop` logs (not panics) on `cuStreamDestroy` failure and is `unsafe impl Send`. `null_or_default` degrades to the NULL stream on creation failure (correct: serialises but stays correct). `grid_x_for` is overflow-safe for `u32::MAX` rows (tested). The V-1 `tag_launch_stream` centralisation makes the Drop-fence sound for `launch_1d`/`launch_with_geometry` callers (but not the raw `execute_projection` launch — see F10a).

### 1d. Stubs / TODO / dead code / fallback-that-is-a-gap (consolidated)

| Location | Kind | Note |
|---|---|---|
| `streaming.rs:386` | `TODO(cuda)` | `PinnedBudget::reserve` only bumps a host counter; real `cuMemHostAlloc` pinned backing deferred. Host accounting is correct; the whole streaming/morsel orchestration (`BatchStream`, `MorselPlan`, `plan_upload`) is **scaffolding never wired into `execute`** — the engine still materialises whole tables (`materialize_table`). This is the single biggest "looks done, isn't wired" item. |
| `gpu_table.rs:535` | hard error, no fallback | Date32/Timestamp upload (F11). |
| `gpu_compact.rs:981,994` | hard error, no fallback | Decimal128 / Date / Timestamp GPU gather (F11). |
| `module_cache.rs:46` | `TODO` | per-Engine cache migration; global string cache unbounded (F1). |
| `join.rs:609` | reject | OUTER + non-equi (F4). |
| `window.rs:921` | reject | computed window keys/inputs. |
| `sort.rs:424` / `join.rs:1235` | reject | computed ORDER BY / computed join keys. |
| `filter.rs:54` | dead-ish | `upload_primitive_values_async` imported only for a cuda-stub test (`#[allow(unused_imports)]`); pre-positioned for a future GPU-lifted predicate path that doesn't exist. Harmless but is unused production surface. |

No `todo!()`/`unimplemented!()` macros in production paths; `panic!`/`unreachable!`/`unwrap` occurrences are all in `#[cfg(test)]` modules or genuinely-unreachable match arms (e.g. `Accumulator::new` line 423, gated by the dispatch in `compute_partition`). The `expect("just built")`/`expect("hit above")` in engine.rs (lines 1761, 2583) are guarded by immediately-preceding inserts/checks and are sound.

### 1e. Performance suggestions (PCIe / launch overhead / redundant transfers)

- **P1 (MEDIUM)** — `engine.rs:2701-2717` Decimal128 passthrough validity does a D2H+H2D round-trip per query (F10b). Use device-to-device copy or share the source mask.
- **P2 (MEDIUM)** — `gpu_table.rs:2699` Utf8 dictionary passthrough clones the dictionary (`src.to_vec()`) on every projection that passes a string column through; for a hot repeated query this re-clones an immutable dictionary each call. Cache/`Arc`-share it.
- **P3 (LOW)** — `execute_projection` builds a fresh per-call `CudaStream` (`null_or_default`, line 2841) and synchronises at each `download_pinned` (line 2947). Multiple output columns each trigger their own `to_pinned_async + synchronize` (gpu_upload.rs:96-98) — i.e. one stream sync **per output column** rather than one D2H batch + a single sync. Batch the D2H copies onto the stream and synchronise once.
- **P4 (LOW)** — host hash join (`build_hash_map`, join.rs:839) allocates a `Box<str>` per non-dict Utf8 build/probe key row (`JoinKeyValue::Utf8`). For large string joins that dominate allocator traffic; the dict path (`DictIdx`) avoids it. Consider interning non-dict Utf8 build sides too (the GPU path already does via `intern_utf8_columns`).
- **P5 (LOW)** — `verify_pairs_on_host` (gpu_join.rs:3178) re-downcasts both arrays *per row* inside `arrow_row_eq` (`.as_any().downcast_ref()` in the loop, lines 3230-3277). Downcast once per column outside the row loop.
- **P6 (LOW)** — launch overhead: every `execute_projection` predicate query compiles/loads two modules (`KERNEL_ENTRY` + `PREDICATE_ENTRY`) and runs two launches + a gather pipeline. The module cache amortises the JIT, but the predicate kernel re-reads all input columns the projection kernel already read. A fused predicate-eval-and-store kernel would halve the column reads; flagged as a known follow-up in `compact.rs`.

---

## 2. TEST ADEQUACY

**Overall coverage estimate:** host-side operators ~75-85% of branches (excellent unit coverage in `sort.rs`, `limit.rs`, `setops.rs`, `window.rs`, `filter.rs`, `streaming.rs`, `compact.rs`, `metrics.rs`, `error.rs`, `observability.rs`). GPU paths are predominantly covered only by *dispatch/predicate* tests that run under `cuda-stub` and `#[ignore = "gpu:..."]` integration tests that CI skips — so the actual kernel-execution branches (capacity overshoot, lossy-fold verify, OUTER second-pass, multipass recursion) have **near-zero automated coverage on CI**. Engine orchestration (`execute` dispatch) is covered indirectly by `engine.rs`'s `#[cfg(test)]` SQL round-trips but several arms are untested (Window, SetOp, non-equi Join through the public `sql` path).

### Untested / under-tested operator paths and specific tests to add

**Join (`join.rs`)**
- All-NULL join keys on both sides (INNER → empty; LEFT → all left rows NULL-padded). Tests assert `extract_key` returns None but no end-to-end `execute_inner_join`/`execute_outer_join` over an all-NULL-key batch.
- INNER join where the build hash map is non-empty but probe finds zero matches (`build_pairs.is_empty()` early-return, line 244).
- FULL OUTER with both sides non-empty and disjoint keys (exercises the second-pass unmatched-build emission, lines 428-436).
- Multi-key (`Many`) host join correctness (only `One`/hash-equivalence is unit-tested; no `execute_*` over a 2-column key).
- `MAX_NESTED_LOOP_INNER_ROWS` boundary: inner side exactly 1024 vs 1025.
- Cross-dtype key rejection (`check_key_dtypes` error path) — no test.

**Sort (`sort.rs`)**
- Multi-key host sort with mixed ASC/DESC and per-key nulls_first (only single-key host sorts and GPU *dispatch* are tested; the host `lexsort_to_indices` multi-key path has no assertion).
- UTF-8 ordering (the review brief explicitly calls this out): no test sorts a `StringArray` and asserts lexicographic/byte order, including non-ASCII multi-byte. Add one.
- Stability: lexsort is not stable; add a test documenting tie-break behaviour for equal keys so a future GPU-radix swap can't silently change row order for ties.

**Set-ops (`setops.rs`)**
- `INTERSECT ALL` / `EXCEPT ALL` where right multiplicity exceeds left.
- Empty *right* with INTERSECT (→ empty) — only EXCEPT-empty-right is tested.
- Multi-column NULL keys across inputs (single-column NULL equality is tested).

**Window (`window.rs`)**
- Single-row partition; partition of all-NULL keys (NULLs grouped together — `eq_rows` treats `None==None`).
- Multiple PARTITION BY keys + multiple ORDER BY keys together.
- Float MIN/MAX scatter when a peer group spans the partition boundary.
- `Float32`/`Int32` narrowing in `ResultBuilder::finish` (i64→i32, f64→f32) — only Int64/Float64 outputs are asserted.

**Limit (`limit.rs`)** — well covered. Add: OFFSET on a multi-batch (post-concat) input to confirm slice offsets apply to the concatenated batch, not per-batch.

**Streaming (`streaming.rs`)** — morsel logic is well covered. Missing:
- **Cancellation / partial-drain:** there is no notion of cancelling a `BatchProducer` mid-iteration; `drain_to_batches` fully drains or errors. If streaming is ever wired into `execute`, add a test for producer error on the *Nth* batch leaving no partial GPU state.
- `plan_upload` with `total_bytes` overflow / `total_rows == 0 && total_bytes > 0`.

**Engine (`engine.rs`)**
- `execute` arms with no direct test: `Window`, `SetOp`, `Union` with mismatched schemas (should error), non-equi `Join` through `sql()`.
- `run_logical_plan` not counted in metrics (F2) — add a test asserting the counter contract once decided.
- Decimal128 passthrough NULL reconstruction (the round-trip at 2701) — assert a NULL Decimal row survives a projection as NULL.
- Date/Timestamp projection: add a test pinning the *current* error (so the fallback gap F11 is intentional and visible) and flip it when fixed.

**GPU paths (CI-runnable under cuda-stub)**
- `gpu_join` capacity-overshoot → `GpuCapacity` → host fallback equivalence (host result must equal what the GPU would have produced). Currently only the `compute_capacity` math is unit-tested.
- `module_cache` eviction: add a test that inserts > cap entries into `GLOBAL_MODULE_CACHE` once it is bounded (F1) and asserts the oldest is evicted — mirrors the existing `SpecCache` eviction tests.

---

## 3. FEATURES — directions

1. **Wire the streaming/morsel scaffolding into execution.** `BatchStream`/`MorselPlan`/`PinnedBudget` are complete and tested but unused — `materialize_table` still uploads whole tables, capping the engine at VRAM size. Closing this (larger-than-VRAM joins/aggregations via morsel pipelining + the pinned-buffer TODO at streaming.rs:386) is the highest-leverage feature and is already 60% built.
2. **OUTER + non-equi join** (join.rs:609) and **computed join keys** (join.rs:1235) — the nested-loop executor already streams; extend it to track unmatched preserved-side rows.
3. **Temporal GPU support** (Date32/Timestamp upload + gather) to remove the F11 capability inconsistency, or a clean host-projection fallback for them.
4. **GPU window functions** — `window.rs` is deliberately host-only behind a stable `PhysicalPlan::Window`; a partition/scan GPU offload slots in cleanly. Computed PARTITION/ORDER keys too.
5. **Cross-dtype equi-join / set-op type unification** at the executor layer (currently delegated to the planner; add coercion or at least executor-level dtype assertions — F12).
6. **Cancellation tokens** for long-running queries (streaming producers, multi-chunk nested-loop joins) — there is no way to abort a running `execute` today.
7. **Bounded global module cache + per-Engine migration** (F1) — also enables a documented cache-reset/debug hook.

---

## Top priorities (ranked)

1. **F1 (MEDIUM)** — bound `GLOBAL_MODULE_CACHE`; it leaks loaded cubins for the process lifetime.
2. **F11 (MEDIUM)** — Date/Timestamp (and Decimal/temporal gather) are hard errors with no host fallback, inconsistent with every other GPU gate; verify real `SELECT ts` queries don't hit it.
3. **F10a (MEDIUM)** — `execute_projection`'s hand-rolled `cuLaunchKernel` skips the V-1 `tag_launch_stream` fence; safety depends on downstream syncs the V-1 work declared removable.
4. **F2 (MEDIUM)** — `run_logical_plan`/`run_subplan` don't bump query counters; metrics undercount the DataFrame path.
5. **F13 (LOW) / F16** — observer panic propagates out of `Engine::sql` despite the "never escalate" doc; wrap in `catch_unwind`.
6. **Tests:** UTF-8 sort ordering, all-NULL-key join end-to-end, multi-key host join/sort, GPU capacity→host fallback equivalence.
