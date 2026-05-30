# A2 completion report — event-based deferred-free wiring (review finding C1 / P1)

Branch `dev`, root `C:\Projects\bolt`. Edits confined to `src/cuda/` only.
No `cargo build/check/test` was run (orchestrator builds); every change is
kept compiling-correct by inspection, including the `cuda-stub` config.

## Files changed

### 1. `src/cuda/buffer.rs` — the core wiring (THE GAP A left)

**`GpuBuffer::Drop` rewritten** (was a blanket per-stream `cuStreamSynchronize`
+ `POOL.free`):
- `ptr == 0` → early return (unchanged).
- `used_streams` **empty** → `POOL.free(ptr, alloc_bytes)` inline, no fence, no
  events (the never-async-touched fast path; unchanged behaviour, clearer code).
- `used_streams` **non-empty** → dispatch to `current_drop_reclaim()(ptr,
  alloc_bytes, &streams)`. This is the new event-based path.

**New production reclaim `real_drop_reclaim(ptr, alloc_bytes, streams)`:**
1. `record_defer_events(streams)` records one `CU_EVENT_DISABLE_TIMING` event on
   every recorded stream (all-or-nothing).
2. On `Some(events)`: `event_query` all of them.
   - all ready → `event_destroy` each, then `POOL.free` inline (no stall);
   - any not-ready → `mem_pool::defer_free(ptr, alloc_bytes, events)` — the block
     is parked OUT of the allocatable pool; the lazy sweep in `alloc`/`free`
     reclaims it once every event completes. **`defer_free` is now live** (it was
     dead code before this change — the whole point of the task).
3. On `None` (events unavailable / create or record failed / `cuda-stub`):
   blanket `fence_all_streams(streams, current_drop_fence())` then `POOL.free`
   inline — i.e. exactly the old conservative behaviour.

**New helper `record_defer_events(streams) -> Option<Vec<CUevent>>`:** creates +
records an event per stream; on ANY failure it destroys every event it already
created and returns `None`. All-or-nothing is a safety requirement (a partial
event set would let the sweep reclaim after only some streams drained — so a
partial outcome routes the whole block to the blanket-sync fallback instead).

**New test seam (mirrors the existing `drop_fence_with`):** `type DropReclaimFn`,
`#[cfg(test)] DROP_RECLAIM_OVERRIDE` thread-local, `#[cfg(test)] drop_reclaim_with`,
and `current_drop_reclaim()` (returns the test override if installed, else
`real_drop_reclaim`). In non-test builds this collapses to a direct call to
`real_drop_reclaim`.

**`#[cfg(test)] GpuBuffer::test_with_raw(ptr, alloc_bytes)`** — synthesize a
buffer over a fake non-null ptr without the driver, so host tests can exercise
the Drop dispatch (the fake ptr routes through the pool's `#[cfg(test)]` free
shim).

**`PinnedHostBuffer::Drop` — intentionally left on the blanket sync** (see
"PinnedHostBuffer decision" below). Its TODO comment was updated to reflect that
the `cuEvent*` FFI now exists and that the only remaining cross-file gap is a
*pinned-specific* pending-free list (the device pool's `defer_free` parks
`CUdeviceptr` pool blocks, not `cuMemFreeHost` host pages).

**4 new host-only tests** appended to `mod stream_set_tests`:
- `empty_used_streams_frees_inline_without_reclaim_hook` — empty set ⇒ inline
  free, reclaim hook never runs.
- `nonempty_used_streams_invokes_reclaim_with_full_set` — non-empty ⇒ reclaim
  hook runs once with the full deduped stream set (2 streams from 3 marks).
- `inflight_drop_defers_block_then_sweep_reclaims` — an in-flight drop parks the
  block in `mem_pool::TEST_DEFERRED` (NOT the allocatable pool); a simulated
  sweep then drains+frees it and the pending list empties.
- `record_defer_events_falls_back_when_events_unavailable` — with no
  driver/events (host CI / cuda-stub), `record_defer_events` returns `None`
  (fallback), never a partial set.

### 2. `src/cuda/mem_pool.rs` — test-build stub for `defer_free`

`real_drop_reclaim` references `mem_pool::defer_free`, but the production
`defer_free` is `#[cfg(not(test))]`, so a `#[cfg(test)]` counterpart was required
for the tree to compile under `cargo test`. Added:
- `#[cfg(test)] pub(crate) static TEST_DEFERRED: Lazy<Mutex<Vec<(CUdeviceptr,
  usize)>>>` — side-channel recording parked blocks.
- `#[cfg(test)] pub(crate) fn defer_free(ptr, alloc_bytes, _events)` — records
  `(ptr, alloc_bytes)` in `TEST_DEFERRED`, bumps `DEFERRED_FREE_COUNT`, drops the
  (non-existent under test) events. Signature is identical to production so the
  `GpuBuffer::Drop` call site is unchanged across configs.

The existing `#[cfg(test)]` no-op `sweep_pending_frees` / `drain_pending_frees_blocking`
stubs are retained (host tests have no real events to query).

## Verification of A's earlier H1 / H4 edits (task item 3)

Both are **present and complete** — no finishing needed:

- **H1 (context-currency guard on default-backend free):** `cuda_sys::mem_free`
  (`cuda_sys.rs:684`) now calls `ctx_get_current()` and: frees via `cuMemFree_v2`
  only when a context is current; returns a descriptive `BoltError::Other` (leak,
  not unsafe free) when none is current; surfaces the error if the query itself
  fails (e.g. cuda-stub). `ctx_get_current` + `cuCtxGetCurrent` real binding
  (`:108`) and cuda-stub shim (`:248`) both exist. The cudarc backend already
  self-guards via `device_ref()` in `cudarc_backend::mem_free` (`:168`).
- **H4 (cudarc memcpy `from_raw_parts` hardening):** `cudarc_backend::memcpy_h2d`
  (`:195`) and `memcpy_d2h` (`:252`) no longer synthesize a host slice; they
  short-circuit `count == 0` before any pointer use, `checked_mul` the byte size,
  and pass the raw pointer straight to `cuMemcpyHtoD_v2` / `cuMemcpyDtoH_v2` via
  cudarc's `driver::sys::lib()` (the non-null `debug_assert` is kept only as a
  developer tripwire, explicitly no longer load-bearing). The async paths were
  already raw-pointer based.

Also confirmed A's `cuda_sys` event API is complete: `CUevent` type, real
`extern "C"` bindings (`cuEventCreate/Record/Query/Synchronize/Destroy_v2`),
cuda-stub shims returning `CUDA_ERROR_STUB`, `CUDA_ERROR_NOT_READY` (600),
`CU_EVENT_DISABLE_TIMING`, and safe wrappers `event_create/record/query/
synchronize/destroy`. `mem_pool`'s `PendingFree`, `PENDING_FREES`, `sweep_pending_frees`
(alloc/free call sites), and `drain_pending_frees_blocking` (drain call site) were
all already in place; the only missing link was a caller of `defer_free`, now wired.

## Safety argument — why no premature reuse is possible

A block's device address re-enters the allocatable pool ONLY after every stream
that ever touched it has drained past its recorded point. In all three Drop
branches:

- **inline event fast path:** the block is freed only after `event_query`
  returned ready for *every* recorded stream.
- **deferred path:** `defer_free` parks the block out of the pool; the sweep
  (`sweep_pending_frees`, run on every `alloc`/`free`) frees it only when every
  event queries complete. A not-ready OR errored query keeps the WHOLE entry
  parked (`matches!(.., Ok(true))`) — a bad probe defers, it never authorises a
  free. Shutdown `drain_pending_frees_blocking` `event_synchronize`s any
  remainder before freeing, so nothing is leaked or freed early at teardown.
- **fallback path:** identical to the old code — `cuStreamSynchronize` drains
  every recorded stream before the inline free.

In every branch the set used is the FULL deduped `StreamSet` (the C-2 fix), so —
exactly like the blanket sync — no stream that ever touched the block can be in
flight when its address is recycled. The event path strictly *tightens* the wait
(to the recorded point, not the stream's whole tail) and removes the synchronous
Drop stall; it never widens the reuse window. Net: **no less safe than the
blanket sync, and faster on the common path.** (The pre-existing C1 residual —
a kernel launched off a raw `device_ptr()` that never tags its stream — is
unchanged by this work; it was never fenced by the blanket sync either, and
closing it structurally needs launch-glue changes outside `src/cuda/`.)

## Stub-fallback behaviour (`--features cuda-stub`)

Under cuda-stub there is no GPU and `cuEventCreate` is the shim returning
`CUDA_ERROR_STUB`. So on a non-empty-streams drop: `record_defer_events`'s first
`event_create()` → `Err` → `None` → `real_drop_reclaim` takes the fallback:
`fence_all_streams` (stub `cuStreamSynchronize` returns `CUDA_ERROR_STUB`,
logged as a warning, harmless) then `POOL.free`. This is exactly the old
behaviour, so the cuda-stub build is no less safe and behaves identically to
before. No cuda-only symbol is referenced without an existing stub counterpart:
the event helpers, `cuStreamSynchronize`, and `ctx_get_current` all have shims A
added; my new code only composes those. The `defer_free` call site compiles in
every config (`#[cfg(not(test))]` real + `#[cfg(test)]` stub).

## Residual follow-ups (out of scope here)

1. **`PinnedHostBuffer::Drop` deferral** — still on the blanket sync. Deferring
   pinned host-page frees needs a *new* pinned-specific pending-free list with a
   drain-at-shutdown path (mirroring `mem_pool`'s device `PENDING_FREES` +
   `drain_pending_frees_blocking`). Without that drain, a deferred pinned free
   would risk leaking page-locked pages at process exit (strictly worse), and an
   inline query-then-free without deferral buys nothing because
   `cuStreamSynchronize` already no-ops cheaply on a drained stream. Building the
   pinned list is the right next step; the blanket sync is the correct, no-leak
   interim. Comment in `PinnedHostBuffer::Drop` updated to say exactly this.
2. **C1 launch-tagging invariant** — a raw `device_ptr()` launch that forgets to
   tag its stream is still unprotected (unchanged). The deferred path only
   protects work whose stream was *recorded*. Structural fix (force launches
   through `KernelArgs`) lives in launch glue outside `src/cuda/`.
3. **Inline fast-path cost** — `real_drop_reclaim` always creates+records events
   on a non-empty drop even when the work is already complete, then frees inline.
   This is correct but spends event create/record/query/destroy on the
   already-synced common case. A cheaper variant could `cuStreamQuery` first, but
   that is a micro-opt and not required for correctness.

## PinnedHostBuffer decision (task item 2)

`PinnedHostBuffer::Drop` *does* have the analogous blanket-sync stall, but
deferring it safely is NOT possible with the infrastructure A built (which is
device-pool-only). Per the task's "only if present and safe" guard, it was left
as the blanket sync and documented as follow-up #1 above — deferring it now would
be either unsafe (pinned-page leak at shutdown) or a no-op win.
