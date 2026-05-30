# Agent B — exec core remediation (done)

Branch `dev`. Edited ONLY: `src/exec/engine.rs`, `src/observability.rs`.
`src/metrics.rs` needed **no** code change (the F2 counting lives entirely in
engine.rs). Did NOT run cargo. Did NOT touch streaming.rs / lib.rs / mod.rs /
src/cuda.

---

## Task 1 (F2, MEDIUM) — DataFrame path query counters

**File:** `src/exec/engine.rs`

- `run_logical_plan` (~:2213) now bumps `Counter::QueriesTotal` once at the top
  (before any `?`), and `Counter::QueriesFailed` when the `execute` bind returns
  `Err` — byte-for-byte mirroring the `sql()` path (`QueriesTotal` at entry,
  `QueriesFailed` only on `execute` failure, not on the rare early
  rewrite/lower `?`-returns).
- `run_subplan` (~:2285) intentionally left **uncounted**. It is the nested
  subquery executor invoked from `resolve_subqueries` under both `sql()` and
  `run_logical_plan`; counting it would double-count. Added a doc block to
  `run_subplan` stating the "top-level queries only" contract explicitly (closes
  the "undocumented asymmetry" half of F2). Net effect: one top-level statement
  (DataFrame or SQL) = exactly one `QueriesTotal`, regardless of N subqueries.

**Contract chosen:** count top-level queries only, uniformly across `sql()` and
`run_logical_plan`. This matches the review's "at least bump in
run_logical_plan" and the existing `sql()` semantics.

**Tests added** (engine.rs test mod):
- `run_logical_plan_bumps_queries_total` `#[ignore = "gpu:metrics"]` — delta
  assertion (`>= +1`, race-tolerant) over a DataFrame scan.
- `run_logical_plan_failure_bumps_queries_failed` `#[ignore = "gpu:metrics"]` —
  scan of an unregistered table fails in `execute`; asserts both `QueriesTotal`
  and `QueriesFailed` advance.
- `query_counters_are_readable_and_monotone` — **host-only / CI-runnable**
  sanity that the `metrics().counter(..).get()` read surface the GPU tests rely
  on is wired and monotone.
- Tests serialise under a new `METRICS_TEST_LOCK` and assert monotone deltas
  (the counters are process-global; exact counts would race sibling `--ignored`
  tests).

---

## Task 2 (F13/F16, observer panic must not escalate)

**Files:** `src/observability.rs`, `src/exec/engine.rs`

- `notify_observers` now invokes the user observer inside
  `std::panic::catch_unwind(AssertUnwindSafe(..))`; a panic is **logged
  (`log::warn!`) and swallowed**. The lock is already dropped before the
  callback (pre-existing), so the caught unwind crosses no held lock and
  `AssertUnwindSafe` is sound.
- Rewrote the `notify_observers` doc to state panics are swallowed (was: "panic
  propagates"). Updated `Engine::maybe_emit_pool_stats` doc (engine.rs) to spell
  out the never-escalate guarantee, so the doc at the old engine.rs:2163-2167
  ("internal errors ... must never escalate") is now accurate for panics too.
- **Updated existing test** `panicking_observer_does_not_disable_surface`: it
  previously asserted `result.is_err()` (panic propagates); it now asserts
  `result.is_ok()` (panic swallowed by `notify_observers`).
- **Added test** `panicking_observer_swallowed_on_every_emit`: a still-installed
  panicking observer is invoked on each of 3 emits without ever unwinding (the
  swallow must not unregister it) — models the engine's repeated periodic emit.

These observability tests are host-only / CI-runnable (no GPU).

---

## Task 3 (F10a, MEDIUM) — projection launch Drop-fence tagging

**File:** `src/exec/engine.rs`

The hand-rolled `cuLaunchKernel` in `execute_projection` (~:2895) bypassed
`KernelArgs::tag_launch_stream`, so the freshly-allocated **output** buffers
weren't recorded in their `StreamSet` and their `Drop` fence depended on
downstream syncs the V-1 work declared removable.

- Added `DeviceCol::mark_launch_stream(&self, stream)` (engine.rs) that records
  the launch stream into every `GpuVec` the output column owns — including the
  Decimal128 `values` buffer **and** its optional `valid_mask` — via the
  **existing public** `GpuVec::mark_stream_use` (the documented entry point for
  callers that drive a raw `cuLaunchKernel` off `device_ptr()` instead of
  `KernelArgs`).
- Call it for every `output_col` immediately after the launch, so each output
  buffer fences this stream on `Drop` exactly as `tag_launch_stream` would.

**Coordination with agent A / cuda API:** no new cuda API needed — used the
already-public `GpuVec::mark_stream_use(CUstream)` and
`CudaStream::raw()`. Nothing in src/cuda was edited.

**Scope note for the orchestrator (NOT a regression I introduced):**
- The **input** projection columns could not be tagged from engine.rs:
  `GpuColumn`/`GpuColumnData` in `src/exec/gpu_table.rs` (not in my allowed file
  set) expose no `mark_stream_use`/`used_streams_cell` accessor. They live in the
  persistent GpuTable cache and are not recycled across the launch, so the
  load-bearing case (outputs) is covered. If full input-side parity is wanted,
  add a `pub fn mark_stream_use(&self, CUstream)` to `GpuColumn`/`GpuColumnData`
  (owner of gpu_table.rs) and call it on each `input` column in
  `execute_projection`.
- The predicate sub-path's `launch_predicate_kernel` lives in
  `src/exec/compact.rs` (not editable by me) and tags nothing, relying on its
  own `stream.synchronize()`. Pre-existing; flag for whoever owns compact.rs if
  the same V-1 hardening is wanted there.

**Test added:** `device_col_mark_launch_stream_tags_all_variants`
`#[ignore = "gpu:projection"]` — allocates each `DeviceCol` variant (incl.
Decimal128 with a mask installed) and tags a stream; must not panic/err, and is
idempotent (StreamSet dedups). Can't read `StreamSet` contents from engine.rs
(it's `pub(crate)` in cuda), so this asserts the no-panic/tagging plumbing.

---

## Task 4 (F10b/P1, MEDIUM) — Decimal128 validity round-trip

**File:** `src/exec/engine.rs` (~:2740 in `execute_projection`)

Replaced the D2H + H2D host round-trip (`src_mask.to_vec()` then
`GpuVec::from_slice(&bits)`) with a **device-to-device** copy:
allocate `GpuVec::<u8>::zeros(mask_len)` and `cuda_sys::memcpy_d2d::<u8>(dst,
src, mask_len)` (already a public `unsafe` cuda_sys fn used elsewhere for the
incremental cache prefix copy). Removes both PCIe crossings per passthrough
Decimal128 query while keeping the existing **owned-buffer** semantics (no
lifetime/refactor risk).

**Deferred (documented):** fully *eliminating* the copy by Arc/refcount-sharing
the source mask buffer would require changing `DeviceCol::Decimal128.valid_mask`
from `Option<GpuVec<u8>>` to a shared/borrowed type and threading a borrow of
the cached source column through to the download path — a non-trivial
lifetime refactor that also touches the download code. Judged not low-risk;
the D2D copy captures the PCIe win safely. Left as a follow-up.

SAFETY of the D2D: `dst` is freshly `zeros`-allocated (distinct device pointer,
non-overlapping with `src`), both are `mask_len` bytes; zero-length is
short-circuited inside `memcpy_d2d`. The new dst mask is also tagged by the
Task-3 `mark_launch_stream` (it covers the Decimal128 `valid_mask` arm).

---

## Files changed
- `src/exec/engine.rs` — Task 1 (counters in run_logical_plan + run_subplan
  doc), Task 2 (maybe_emit_pool_stats doc), Task 3 (DeviceCol::mark_launch_stream
  + call site), Task 4 (D2D mask copy), 4 new tests.
- `src/observability.rs` — Task 2 (catch_unwind in notify_observers + doc),
  updated 1 test, added 1 test.

## Wiring needed from orchestrator
- None for compilation. Optional follow-ups noted above:
  (a) public `mark_stream_use` on `GpuColumn`/`GpuColumnData` (gpu_table.rs) for
  input-side Drop-fence parity; (b) same V-1 tagging in compact.rs
  `launch_predicate_kernel`; (c) the deferred Arc-share of the Decimal128 mask.

## Out of scope but noted
- F3 (metrics.rs:95 doc "≥ 2^22 µs" should be "≥ 2^23 µs") is in my allowed
  file but outside tasks 1-4; left untouched. Trivial one-word doc fix for
  whoever picks up F3.
