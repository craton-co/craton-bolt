# CUDA Backend Module Review — Bolt

Scope: `src/cuda/{async_copy, buffer, cudarc_backend, cuda_sys, dictionary, dictionary_any, dictionary_i64, mem_pool, mod, smart_ptrs}.rs`

Overall this is a mature, heavily-iterated module. Many classic FFI/GPU foot-guns (multi-stream UAF, double-free, drop-vs-in-flight DMA, context-currency on free, OOM recovery, LRU pool) have already been identified and closed in prior review waves (C-1/C-2/V-1/V-2/H1/H4/M3/P1/V-9). The findings below are what remains.

---

## 1. CODE REVIEW

### CRITICAL

None outstanding. The previously-critical hazards (recycling a pool block while a stream still references it; host-side pinned UAF; double-free) are structurally addressed via the `StreamSet` + event-deferred-free machinery and the borrow-checked `GpuVec`/`GpuView` design.

### HIGH

**H-1. Deferred-free sweep can run with the wrong / no CUDA context current; events are context-bound.**
`mem_pool.rs:434-482` (`sweep_pending_frees`) is called at the top of `alloc` (`:1135`) and `free` (`:1291`). It calls `cuda_sys::event_query`/`event_destroy` and then `POOL.free`. Events are context-bound (acknowledged at `mem_pool.rs:372-374`). A `GpuBuffer` is `Send`, so `free`/`alloc` can be invoked from a worker thread whose current context differs from the one the events were recorded in. `event_query` against a non-current/foreign context can return a driver error; the code maps any error to "not ready" (`:461`), so the block is retained — not freed early, which is safe — but it is **never reclaimed** on that path, so the pending list can grow until shutdown drain. Worse, if a *different* context happens to be current, behavior is undefined per the driver. Unlike `cuda_sys::mem_free` (`cuda_sys.rs:697-722`) which guards on `ctx_get_current`, the sweep does no context-currency check before touching events. Recommendation: gate the sweep on a context-currency check (or capture+rebind the recording context, mirroring the pool-watcher's `CAPTURED_CTX` machinery), and/or only sweep on the engine thread.

**H-2. Doc/code mismatch: the "debug_assert guard in `GpuBuffer::Drop`" does not exist.**
`mem_pool.rs:348-351` and the C1 design note both claim a `debug_assert` in `GpuBuffer::Drop` flags "non-empty buffer dropped with zero recorded streams" so a forgotten kernel-launch stream tag surfaces in tests. No such assertion exists in `buffer.rs` (`Drop` at `:1075-1111` only branches on `streams.is_empty()`; the only `debug_assert` in the file is `round_up_to_alignment` at `:1544`). This is the documented detector for the known C-1 residual hole (a kernel launched off `device_ptr()` that bypasses both the async helpers and `KernelArgs` records no stream, so neither sync nor deferred-free fences it). The hole is real and currently has **no** runtime tripwire. Recommendation: add the `debug_assert` that the docs promise, or correct the docs. The hole itself remains the most dangerous residual soundness gap in the module (silent UAF if any launch path forgets to tag).

### MEDIUM

**M-1. `recover_from_oom` retry uses `driver_mem_alloc` directly and never re-checks the pool.**
`mem_pool.rs:1250-1271` (`try_retry_alloc`) calls `driver_mem_alloc` after eviction. That is correct for reclaiming VRAM, but a concurrent thread that freed a block of the right size into the bucket between the original miss and the retry is not consulted — the retry always goes to the driver. Minor efficiency only; not a correctness issue.

**M-2. Pinned-host deferred free is still synchronous (blanket per-stream sync).**
`buffer.rs:1497-1511` (`PinnedHostBuffer::Drop`) and `async_copy.rs:519-522` (`PinnedBuffer::Drop`) both do a blanket `cuStreamSynchronize` over every recorded stream on drop. The device-side path got the event-based non-stalling deferred free (P1); the host-pinned paths did not (the long TODO at `buffer.rs:1417-1473` explains the missing *pinned* pending-free list). Result: dropping a pinned buffer with an in-flight DMA stalls the dropping thread on each stream's *entire* trailing queue. This is correct but a real latency cost on the hot D2H/H2D path (e.g. `to_pinned_async` at `smart_ptrs.rs:224-237`). Perf opportunity, not a bug.

**M-3. `from_arrow_bytes` / `from_slice` always allocate + synchronous H2D; no pinned staging.**
`buffer.rs:710-712`, `:258-275`. The synchronous upload path uses pageable host memory, so the driver synthesizes a staging copy and the call serializes. For the input-upload hot path this is the single biggest PCIe win available. The async+pinned plumbing already exists (`PinnedBuffer`, `upload_async`); the synchronous constructors don't use it. Consider a pinned-staged bulk upload for large columns.

**M-4. `total == 0` from `cuMemGetInfo_v2` silently skips watcher poll but a tiny `total` does not.**
`mem_pool.rs:2246`/`:2272` guards `total > 0`. Fine. But the low-water fraction (`free/total`) uses raw division; if the driver ever returns `free > total` (transient under some MIG/driver states), `frac > 1.0` and eviction simply never triggers. Low impact; defensive clamp would be cleaner.

**M-5. `DictionaryColumn::dedupe`/`from_string_array` indexing `dictionary[(idx as usize) - 1]` relies on never-negative invariant.**
`dictionary.rs:121`, `:246`; `dictionary_i64.rs:147`, `:356`. `idx` here is always a freshly-minted positive value from the same loop, so it's safe in practice, but the `(idx as usize) - 1` would panic-underflow if a `0` ever entered a bucket. The buckets only ever receive positive indices, so this is robust; noting for completeness.

### LOW

**L-1. `device_name` truncation handling.** `cuda_sys.rs:466-476` forces a NUL at the last byte defensively — good. No issue.

**L-2. `mem_alloc`/`free` zero-byte asymmetry between backends.** `cudarc_backend::mem_alloc` rejects `bytes == 0` (`cudarc_backend.rs:154-159`); the hand-rolled `cuda_sys::mem_alloc` does not. The pool always passes `bucket_size(bytes) >= ARROW_ALIGNMENT (64)` so zero never reaches either in practice, but the two backends differ in contract. Harmless given the pool floor.

**L-3. `GpuView` `Send` soundness leans on a process-wide convention.** `smart_ptrs.rs:328-337` — the `Send` impl is sound only because "Bolt serializes GPU launches per thread." This is a global invariant not enforced by the type system; two threads tagging the same parent buffer's `RefCell<StreamSet>` concurrently would panic (RefCell) rather than UB, but it is a latent fragility if the launch model ever changes. Documented honestly.

**L-4. `dictionary_i64` is dead in production but carries full surface.** `dictionary_i64.rs` is `#[allow(dead_code)]`, gated off by the runtime guard in `dictionary_any.rs:71-90`. Fine and clearly documented, but it is untested end-to-end (all GPU tests `#[ignore]`'d) — re-enabling will need real coverage (the module's own TODO §4 says so).

**L-5. `is_empty`/`len` field-touch canary under `cuda-stub`.** `async_copy.rs:491-492` reads `used_streams` purely as a refactor tripwire. Harmless; slightly obscure.

---

## 2. TESTS

### What exists

* **`tests/memory_tests.rs`** — type-level `Send`/`Copy`/`Clone` assertions plus 5 `compile_fail` doctests proving the borrow-checker invariants (view-outlives-vec, mut-excludes-shared, use-after-move). Live round-trips (`round_trip_i32/f64`, `views_observe_same_buffer`, `drop_is_safe_no_double_free`) are `#[ignore]`'d (need GPU).
* **`tests/memory_pool_stress.rs`** — 5 `#[ignore]`'d GPU tests: alloc/free churn (10k), drain-empties, 8-thread concurrent alloc/free, bucket reuse (LIFO same-ptr), drain-on-context-drop.
* **In-module host-only unit tests** (run without a GPU, the bulk of real coverage):
  - `buffer.rs`: `round_up_to_alignment` exhaustively; `StreamSet` dedup/null; `mark_stream_use` accumulation (C-2); `fence_all_streams` via the `drop_fence_with` mock seam; the **event-deferred-free `Drop` decision tree** via `drop_reclaim_with`/`TEST_DEFERRED` (empty→inline, non-empty→reclaim, in-flight→defer-then-sweep, events-unavailable→fallback). Strong.
  - `cuda_sys.rs`: `init` cache retry semantics (success latches, error not cached, flaky-then-success) via the `init_with` fn-pointer seam.
  - `mem_pool.rs`: a large `#[cfg(test)]` harness with synthetic-pointer `test_driver_alloc`/`record_driver_free`, OOM fault injection, LRU-race tests, pool-watcher loop driven against a local pool + mock `MemInfoFn`, cap-bump one-shot, ctx-race invalidation.
  - `async_copy.rs`: sizing/`set_len`/`Deref` (stub backend), stream-set dedup, multi-stream fence via mock, length-guard-before-FFI, signature pins.
  - dictionaries: dedupe (1M-row redundancy), null handling, `decode_index` width-safety (V-14), i32/i64 symmetry, dispatch threshold, M6 null-dict-slot remap.

### Adequacy / ~85% coverage assessment

Host-only **logic** coverage is genuinely high — the mockable seams (`drop_fence_with`, `drop_reclaim_with`, `init_with`, `MemInfoFn`, `TEST_DEFERRED`, synthetic pool driver) let the pure decision logic run on CI without a GPU, and they're used well. For the *policy/bookkeeping* surface (pool LRU, eviction, caps, stream-set, dedupe, decode), 85% is plausible.

However, the **driver-touching surface is essentially untested except behind `#[ignore]`**: every real `cuMemAlloc`/memcpy/event/stream path, `CudaContext` lifecycle, pinned DMA round-trip, and the entire `cudarc_backend` (only `cudarc_err` shape + one ignored smoke test) run only on a GPU host that someone must remember to invoke with `--ignored`. So *line* coverage of `buffer.rs`/`cuda_sys.rs`/`cudarc_backend.rs` is well below 85% on a CI host. The `Drop` *fast paths* and FFI wrappers are not exercised by default.

Notable gaps:
* No test exercises the **deferred-free sweep with a wrong/no current context** (H-1) — the bug is invisible to the current synthetic-pointer harness because `sweep_pending_frees`/`drain_pending_frees_blocking` are `#[cfg(test)]` no-ops (`mem_pool.rs:517-520`). The deferred path's *production* event handling is therefore never executed under test.
* No tripwire test for the **untagged-launch C-1 hole** (H-2) — would be trivial to add once the missing `debug_assert` exists.
* `from_prefix_and_tail` / `extended_with_prefix` (incremental cache, DtoD) have no host-only test and no non-ignored GPU test.
* `recover_from_oom`'s incremental-eviction loop is tested via fault injection (good), but the concurrency interaction (M-1) is not.

### Recommended enhancements

1. Add the promised `debug_assert!(!streams.is_empty() || ...)` to `GpuBuffer::Drop` and a unit test that a tagged launch records its stream (host-only via the view back-reference, like the existing C-1 tests).
2. Make `sweep_pending_frees` testable: a `#[cfg(test)]` seam that injects event-ready/not-ready/error verdicts so the sweep's reclaim/retain/destroy logic runs on CI (it currently can't).
3. Add a context-currency guard test for the deferred sweep (H-1).
4. A CI lane that actually runs `-- --ignored` on a GPU runner (even nightly) — the strongest existing tests are all gated off by default.
5. Host-only dedupe/decode is solid; add a `from_prefix_and_tail` length-mismatch unit test (the error branch at `buffer.rs:296-303` is host-checkable).

---

## 3. NEW FEATURES / DIRECTIONS (CUDA layer)

* **Pinned pending-free list (close M-2).** Mirror the device `PENDING_FREES`/`drain_pending_frees_blocking` for `cuMemFreeHost` pages so pinned-buffer drops stop stalling. The TODO at `buffer.rs:1417-1473` already specifies the design; it's the highest-value perf follow-up.
* **Pinned staging for bulk synchronous uploads (M-3).** Route `from_slice`/`from_arrow_bytes` of large columns through a reusable pinned staging buffer + async copy to get true DMA overlap on the input path.
* **CUDA stream pool.** The module reuses `exec::launch::CudaStream` but each transfer takes a `&CudaStream`; a small pool of pre-created streams would enable multi-stream overlap of independent column uploads without per-call create/destroy.
* **CUDA graph capture for the transfer+launch sequence.** `cuda_sys` already binds the graph API (`cuStreamBeginCapture_v2` etc., `:213-224`) for sort; extending capture to the steady-state upload→kernel→download path would cut per-launch driver overhead.
* **`cuMemHostAlloc` flag exploitation.** `CU_MEMHOSTALLOC_WRITECOMBINED` / `_DEVICEMAP` are bound (`cuda_sys.rs:876-884`) but `PinnedBuffer::new` always passes `flags = 0`. Write-combined pinned memory for upload-only sources is a cheap PCIe win.
* **cudarc migration completion.** `cudarc_backend.rs` is an explicit Stage-1 spike (raw `malloc_sync`/`free_sync` escape hatch). Finishing the `CudaSlice<T>`-owned migration would let cudarc own context/lifetime and remove the dual-context reasoning in `CudaContext`.
* **Re-enable i64 dictionary path** once codegen accepts i64 indices (the four conditions in `dictionary_i64.rs:38-55`).
* **Telemetry surfacing.** `deferred_free_count`/`oom_recovery_count`/`proactive_eviction_count` exist but are `#[allow(dead_code)]` behind `pool_stats`; wiring them to the observability layer would make pool pressure visible in production.

---

### Severity summary

| Sev | Finding | Cite |
|-----|---------|------|
| HIGH | Deferred-free sweep touches context-bound events with no context-currency guard | `mem_pool.rs:434-482, 1135, 1291` |
| HIGH | Documented `debug_assert` C-1 tripwire missing; untagged-launch UAF hole has no detector | `mem_pool.rs:348-351`, `buffer.rs:1075-1111` |
| MED | OOM retry skips pool re-check | `mem_pool.rs:1250-1271` |
| MED | Pinned-host drop still blanket-syncs (stall) | `buffer.rs:1497-1511`, `async_copy.rs:519-522` |
| MED | Synchronous uploads not pinned-staged | `buffer.rs:258-275, 710-712` |
| MED | `free/total` fraction unclamped | `mem_pool.rs:2247` |
| LOW | dict index `as usize - 1` underflow latent | `dictionary.rs:121,246` |
| LOW | zero-byte alloc contract differs across backends | `cudarc_backend.rs:154`, `cuda_sys.rs:658` |
| LOW | `GpuView: Send` relies on global launch-serialization convention | `smart_ptrs.rs:328-337` |
| LOW | i64 dict dead/untested end-to-end | `dictionary_i64.rs` |
