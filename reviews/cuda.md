# Code Review: `src/cuda/` — GPU Memory-Safety / Device-Backend Layer

Reviewer: senior Rust/CUDA auditor
Scope: all 10 files in `C:\Projects\bolt\src\cuda\`
Date: 2026-05-30

Files reviewed in full:
`async_copy.rs`, `buffer.rs`, `cuda_sys.rs`, `cudarc_backend.rs`, `dictionary.rs`,
`dictionary_any.rs`, `dictionary_i64.rs`, `mem_pool.rs`, `mod.rs`, `smart_ptrs.rs`.

---

## Executive summary

This is a carefully written, unsafe-heavy module. The stream-set / `Drop`-fence
machinery (review findings C-1, C-2, V-2) is genuinely sound for the *single-threaded
per-buffer* model it documents, and the pool's lock-order reasoning is well argued. The
overwhelming majority of `unsafe` blocks have correct, specific SAFETY notes.

However there are real correctness gaps. The most serious is a **genuine soundness hole
in the `device_ptr()` / kernel-launch stream-tagging contract**: the entire C-1 guarantee
depends on every launch site calling `mark_launch_use` / routing through `KernelArgs`, but
nothing in this module enforces it — a single launch off `GpuVec::device_ptr()` that
forgets to tag produces a silent use-after-free with no compile error. The `Send`-but-driver-
context-bound design also has a latent unsoundness around freeing on the wrong thread under
the default (non-cudarc) backend.

Severity counts:

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH | 4 |
| MEDIUM | 8 |
| LOW | 9 |

No `todo!()`, `unimplemented!()`, or `panic!("not implemented")` exist in production paths.
Several documented cross-file TODOs (event-based deferred free) and one deliberately
runtime-guarded dead path (i64 dictionary) are present and called out below.

---

## 1. CODE REVIEW

### CRITICAL

#### C1. The C-1 stream-tagging invariant is unenforced — a forgotten `mark_launch_use` is a silent UAF
**File:** `smart_ptrs.rs:277-312` (`GpuView::device_ptr` / `mark_launch_use`), `buffer.rs:364-401`, `buffer.rs:865-893` (`Drop`)
**Severity:** CRITICAL

`GpuBuffer::Drop` only fences streams that were *recorded* in `used_streams`. For kernel
launches, recording happens only if the caller either (a) routes the view through
`KernelArgs::push_input/push_output`, or (b) manually calls `view.mark_launch_use(stream)`.
`device_ptr()` returns a raw `CUdeviceptr` that can be handed to `cuLaunchKernel` with **no
tagging at all**, and the type system does not force the tag. The doc comments are explicit
that this is the contract — but a contract enforced only by prose over ~250 call sites is
exactly the kind of invariant that regresses silently. A single launch that reads
`v.device_ptr()` directly (bypassing `KernelArgs`) and never tags will let `Drop` recycle the
block into the pool *while the kernel is still running*, producing data corruption that is
invisible in this module.

This is the inherent weakness of the "tag after launch" approach versus a structural
event-based pending-free (which the code itself acknowledges as the real fix, see
`buffer.rs:808-864`). It is CRITICAL because the failure mode is silent memory corruption,
not a panic or error.

**Suggested fix (in priority order):**
1. Make `device_ptr()` on `GpuView`/`GpuViewMut` `pub(crate)` and force all launch glue
   through `KernelArgs` (which tags centrally). Audit the codebase for any direct
   `device_ptr()` → `cuLaunchKernel` path that bypasses `KernelArgs`.
2. Land the documented event-based deferred-free pool (`buffer.rs:832-864`). That removes the
   dependence on caller discipline entirely: a block is reclaimed only when its recorded
   events report complete, regardless of whether anyone "remembered" to fence.
3. At minimum, add a debug-build assertion or `cargo` integration test that launches a
   kernel, drops the `GpuVec` mid-flight without syncing, and verifies the pool did not
   re-issue the block (currently no such test exists — see Test section).

---

### HIGH

#### H1. Default-backend free can run against the wrong CUDA context with no guard
**File:** `mem_pool.rs:433-445` (`driver_free`), `buffer.rs:891` (`GpuBuffer::Drop` → `POOL.free`), `cuda_sys.rs:621-623` (`mem_free`)
**Severity:** HIGH

Under the cudarc backend, `cudarc_backend::mem_free` (`cudarc_backend.rs:168-188`) carefully
re-establishes context currency via `device_ref()` before `free_sync`, precisely because a
`GpuBuffer` may be dropped on a worker thread that never bound the context. The **default
(non-cudarc) backend has no such guard**: `driver_free` → `cuda_sys::mem_free` →
`cuMemFree_v2(ptr)` runs against whatever context (or none) is current on the dropping
thread. `GpuBuffer` is `Send` (`buffer.rs:898`), so it can be moved to and dropped on a
thread with no current context, yielding `CUDA_ERROR_INVALID_CONTEXT` (silent leak) or a
free against the wrong context. The cudarc path explicitly documents this exact hazard; the
default path silently has it.

**Suggested fix:** In the default backend, either (a) document and enforce that all drops
happen on the context-current thread, or (b) bind the owning context in `driver_free` the way
the cudarc path does. Given `GpuBuffer: Send` is relied upon by the executor, (b) is safer.

#### H2. `GpuVec::from_slice_async` sizes capacity to `slice.len()`, so a later `set_len`/reuse cannot grow — and `copy_from_async` will silently truncate intent
**File:** `smart_ptrs.rs:202-206`
**Severity:** HIGH (correctness, latent)

`from_slice_async` calls `GpuBuffer::with_capacity(slice.len())` then `copy_from_async`.
`copy_from_async` (`buffer.rs:485-512`) checks `src.len() > self.capacity` and sets
`self.len = src.len()`. That is fine for the immediate call, but the buffer is created with
capacity *exactly* equal to the slice — there is zero headroom. Any downstream code that
treats a `from_slice_async`-produced vec as having spare capacity (e.g. for an in-place
append via `copy_from_async` with a larger slice) will hit the capacity error. This differs
from the synchronous `from_slice` only subtly, but the async path is the one wired into
overlapped pipelines where reuse is most likely. Not a memory-safety bug, but an easy source
of `BoltError::Memory` at runtime.

**Suggested fix:** Document the zero-headroom contract explicitly on `from_slice_async`, or
accept an explicit capacity argument as `zeros_async`-style callers do.

#### H3. `to_pinned_async` records the stream but a partial D2H leaves `len` unadjusted
**File:** `smart_ptrs.rs:224-237`
**Severity:** HIGH (correctness)

`to_pinned_async` allocates `PinnedHostBuffer::<T>::new(self.len())`, issues the async D2H,
and tags the stream. The pinned buffer's logical `len` equals `self.len()`. That is correct
for a full copy. But the symmetric `PinnedHostBuffer::set_len` / `PinnedBuffer::set_len`
(`buffer.rs:1151`, `async_copy.rs:401`) exist precisely to handle "an async D2H that filled
fewer than the buffer's allocated length" — and `to_pinned_async` never exposes a way to
correct the length after a short read. If a kernel produces fewer rows than `self.len()`
(common in filter/compaction outputs where `self.len()` is an upper bound), the returned
pinned buffer reports a stale length and `as_slice()` exposes uninitialized tail elements as
valid `T` (sound for `Pod`, but semantically wrong data). The doc says "DMA lands in pinned
memory" but never warns the caller about the upper-bound case.

**Suggested fix:** Document that `to_pinned_async` assumes an exact-length copy, or return the
pinned buffer plus a mechanism to `set_len` after the caller syncs and learns the true count.

#### H4. `cudarc_backend::memcpy_h2d` synthesizes a host `&[u8]` from a caller-supplied raw pointer with only a `debug_assert` on alignment/validity
**File:** `cudarc_backend.rs:220` (and `:255`, h2d/d2h)
**Severity:** HIGH (soundness under `--features cudarc`)

`std::slice::from_raw_parts(src as *const u8, bytes)` requires `src` be non-null, **properly
aligned for `u8` (trivially true), valid for `bytes` reads, and the whole range within a
single allocation**. The non-null check is only a `debug_assert!` (`cudarc_backend.rs:216`),
compiled out in release. More importantly, the function is generic over `T` but reinterprets
as `u8` — fine for reads — yet it relies entirely on the caller's `# Safety` contract with no
release-mode defense. The sync path (`memcpy_htod_sync`) takes a Rust slice, so cudarc itself
will read the whole range; if the caller passed a too-short or dangling `src`, this is
immediate UB inside `from_raw_parts` *before* any FFI. The hand-rolled `cuda_sys::memcpy_h2d`
(`cuda_sys.rs:630`) does **not** synthesize a slice — it passes the raw pointer straight to
`cuMemcpyHtoD_v2`, so the cudarc path is strictly *more* dangerous for the same inputs.

**Suggested fix:** This is inherent to cudarc 0.13's slice-based sync API. Prefer routing
sync copies through `cudarc::driver::sys::lib().cuMemcpyHtoD_v2(...)` (raw-pointer form, as
the async paths already do at `cudarc_backend.rs:319`) so no host slice is fabricated.

---

### MEDIUM

#### M1. `download_async` tags the stream *before* the FFI, so a failed enqueue still records the stream
**File:** `async_copy.rs:645-655`
**Severity:** MEDIUM

`dst.mark_stream_use(stream_raw)` runs *before* `memcpy_d2h_async`. If the FFI errors (returns
`Err`), the stream is already recorded, and `Drop` will issue an unnecessary
`cuStreamSynchronize` against it. This is safe (over-fencing is harmless and documented) but
wasteful, and it differs from `upload_async` (`async_copy.rs:602-613`) which tags *after* the
copy. The comment explains the ordering is to avoid a borrow conflict, which is legitimate,
but the asymmetry is a latent foot-gun if someone later relies on "recorded ⇒ DMA actually
issued."

**Suggested fix:** Either tag after the FFI in both (restructure the borrow), or document the
"recorded does not imply issued" semantics on `mark_stream_use`.

#### M2. `PinnedHostBuffer::Drop` proceeds to `cuMemFreeHost` even after a failed fence — documented UAF risk
**File:** `buffer.rs:1271-1296`
**Severity:** MEDIUM

When `cuStreamSynchronize` fails (e.g. the stream was already destroyed), the code logs and
**still calls `cuMemFreeHost`** while a DMA may be in flight (`buffer.rs:1275` comment admits
"in-flight DMA may have UB'd"). The rationale (don't leak pinned pages) is defensible, but
this is a deliberate choice to risk host-side UAF over a leak. The same applies to the
`PinnedBuffer` path (`async_copy.rs:561`). Choosing "leak" is the safer default here since
pinned-page UAF can corrupt unrelated allocations.

**Suggested fix:** On fence failure, consider leaking the pages (skip the free) rather than
freeing into a possibly-active DMA, or escalate to `cuCtxSynchronize` as a last resort before
the free.

#### M3. `recover_from_oom` drains the *entire* process-wide pool on any single OOM, including blocks owned by concurrent queries
**File:** `mem_pool.rs:949-983`
**Severity:** MEDIUM (correctness is fine; severe perf/availability cliff)

On one driver OOM, `recover_from_oom` calls `drain()` which returns *every* pooled block to
the driver — discarding the warm cache shared by all concurrent in-flight queries on other
threads. The doc admits this is intentional, but under a workload that hovers near the VRAM
limit this turns into a thundering-herd of drains: every query that OOMs nukes the pool, the
next allocation re-mints from the driver, and a second query OOMs and drains again. The pool's
whole purpose (avoiding `cuMemAlloc`/`cuMemFree` churn) inverts under pressure.

**Suggested fix:** Evict incrementally (call `evict_one` in a loop until the retry succeeds or
the pool is empty) rather than an all-or-nothing `drain()`. The existing `evict_above_high_water`
+ a bounded eviction loop would hand back enough room without cold-cache-ing every concurrent
query.

#### M4. `total_bytes` cap check and the bucket insert are not atomic — soft-cap overshoot is unbounded under N-way contention
**File:** `mem_pool.rs:1015-1024`, `1092-1139`
**Severity:** MEDIUM

The eviction loop reads `total_bytes`, then `try_insert_into_locked_bucket` re-checks under
the bucket lock and `fetch_add`s. Under N threads freeing into N *distinct* size classes
simultaneously, each can independently pass its own cap re-check before any of the `fetch_add`s
land, so the pool can transiently exceed `max_pooled_bytes` by up to `N × alloc_bytes`, not
"one block per racing thread" as the comment claims (the comment is true per *bucket*, but the
caps are global). Reconciliation corrects the counter, but the actual *VRAM* over-reservation
is real until the next free triggers eviction. The doc undersells the worst case.

**Suggested fix:** Acceptable for a soft cap, but correct the comment to reflect the
`O(threads)` global overshoot, and consider a global atomic CAS reservation before the bucket
insert if VRAM headroom is tight.

#### M5. `bucket_size` integer math is correct but the `step.max(1)` paranoia hides a real edge
**File:** `mem_pool.rs:380-399`
**Severity:** MEDIUM (robustness)

`pow2 = 1 << (usize::BITS - 1 - n.leading_zeros())`. For `n` near `usize::MAX`, `pow2` can be
`1 << 63`, and `n.saturating_add(step - 1) & !(step - 1)` saturates — so a request near
`usize::MAX` rounds down to a `step`-aligned value *below* `n` (because saturation caps the
add, then the mask clears low bits). The result can be **smaller than the requested `bytes`**,
which would under-allocate. In practice `cuMemAlloc` rejects such sizes and the pool never sees
them, but `bucket_size` is also used as a pure key (`bucket_len_for`, `free`), so a caller that
freed with a giant `alloc_bytes` could mis-bucket. Extremely unlikely to be hit, hence MEDIUM.

**Suggested fix:** Add a `debug_assert!(rounded >= n)` after the round-up, or clamp the input
to a sane max before bucketing.

#### M6. `from_dictionary_array` maps dictionary-level NULL entries to empty string, silently colliding with a real `""` value
**File:** `dictionary_any.rs:180-186`
**Severity:** MEDIUM (correctness)

When an Arrow `DictionaryArray` has a null *dictionary value* (not a null key), the code pushes
`String::new()` (`""`). If the same dictionary also contains a genuine empty-string entry, the
two become indistinguishable on decode, and any row pointing at the null dict slot decodes to
`""` instead of NULL. The comment calls this "rare," but it is a silent data-correctness bug,
not just an edge case.

**Suggested fix:** Either reject dictionaries with null value entries, or track the null slot
explicitly (e.g. remap null dict entries to index 0 / NULL like the key path does).

#### M7. `GpuView` is `Copy` + carries a raw `streams` back-pointer + is `Send` — a copied view can outlive reasoning about the parent
**File:** `smart_ptrs.rs:245-256`, `:328-337`
**Severity:** MEDIUM (soundness depends on an unstated invariant)

`GpuView` is `#[derive(Copy, Clone)]` and holds `streams: *const RefCell<StreamSet>`. The `'a`
lifetime keeps the parent buffer alive *as long as the view exists*, which is sound — but
because the view is `Copy` and `Send`, a copy can be moved to another thread. `mark_launch_use`
then does `(*cell).borrow_mut()` from that other thread (`buffer.rs:104-111`). The SAFETY note
(`smart_ptrs.rs:333-336`) claims "Craton Bolt serializes GPU launches per thread, so two
threads never tag the same buffer's set at the same instant; even if they did, `RefCell` would
panic rather than UB." That is **not** strictly true: two `borrow_mut` calls from two threads on
the same non-`Sync` `RefCell` is a data race (UB), and `RefCell`'s panic-on-overlap is itself
implemented with non-atomic loads/stores — racing those is UB, not a guaranteed panic. The
`unsafe impl Send for GpuView` is therefore only sound under the *unenforced* "one thread tags a
given buffer at a time" invariant.

**Suggested fix:** This is the same class of issue as C1 — the safety rests on a global
discipline the types don't encode. At minimum, strengthen the comment to stop claiming
`RefCell` "would panic rather than UB" (it would be UB). Ideally make the stream-set an atomic
structure if cross-thread tagging is genuinely possible.

#### M8. `next_tick` is `u64` `Relaxed` and assumed globally unique for LRU keys — wraps after 2^64 frees (benign) but ordering across `Instant` ties relies on it
**File:** `mem_pool.rs:702-735`, `:1112`
**Severity:** MEDIUM (theoretical)

`lru_pop_global_oldest` orders by `(Instant, tick)`. `Instant` resolution on Windows is coarse;
under burst frees many blocks share the same `Instant`, so `tick` is the real tiebreaker.
`tick` is `Relaxed` — fine for uniqueness, but two blocks freed on different threads can get
`tick` values whose order does *not* match wall-clock free order (Relaxed gives no cross-thread
ordering). The LRU is therefore "approximately oldest," not strictly oldest, under contention.
The code's own race-handling comments accept approximation, so this is consistent — but the
"global LRU" claim in the module doc is stronger than what's delivered.

**Suggested fix:** Document that LRU is best-effort under concurrency; no code change needed.

---

### LOW

- **L1. `byte_len()` uses `.expect("byte_len overflow")` — a panic in a getter.**
  `buffer.rs:324-328`, `:1037-1044`, `async_copy.rs:269-273`. A logically-impossible overflow
  (for any pool-minted buffer) still panics rather than returning a `Result`. Acceptable as a
  "this is a bug" guard, but a getter that can panic is a sharp edge. Consider returning
  `Option`/`Result` or saturating.

- **L2. `GpuView::byte_len` / `GpuViewMut::byte_len` use unchecked `len * size_of::<T>()`.**
  `smart_ptrs.rs:284`, `:378`. Unlike `GpuBuffer::byte_len` these do **not** use `checked_mul`;
  they silently wrap on overflow. Inconsistent with the rest of the module's defensive posture.
  Since `len` came from a real buffer this can't realistically overflow, but the asymmetry is a
  latent bug if a view is ever constructed with a synthetic length.

- **L3. `device_name` defensively NUL-terminates but `to_string_lossy().trim_end_matches('\0')`
  is redundant.** `cuda_sys.rs:423-433`. `CStr::from_ptr` already stops at the first NUL, so
  there are no embedded trailing NULs to trim. Harmless, slightly misleading.

- **L4. `init_with` releases the lock across the FFI `cu_init()` call, so two threads can both
  call `cuInit(0)`.** `cuda_sys.rs:371-388`. Documented as intentional (cuInit is idempotent),
  and correct — but the success store is a separate lock acquisition, so a benign double-init
  race exists. Fine, just noting it is by-design.

- **L5. `mem_alloc` (cudarc) rejects zero-byte allocations, but `GpuBuffer::with_capacity`
  always rounds up to `ARROW_ALIGNMENT` (≥64 bytes), so the zero path is unreachable from the
  pool.** `cudarc_backend.rs:125-130` vs `buffer.rs:180`. Defensive and correct, but the two
  layers both guard the same thing; the cudarc guard can never fire via the normal path.

- **L6. `graph_exec_destroy` is `#[allow(dead_code)]` and the graph cache "deliberately leaks"
  at process exit.** `cuda_sys.rs:1228-1231`. Documented intentional leak (avoids a teardown
  race). Worth a tracking issue rather than a permanent leak, but defensible.

- **L7. `memset_d8_async` is `#[allow(dead_code)]` in the non-cudarc default but used by
  `zeros_async`.** `cuda_sys.rs:1101`. The `dead_code` allow is stale-ish — `buffer.rs:627`
  calls it. The attribute may now be unnecessary; verify and remove to avoid masking real dead
  code later.

- **L8. `DictionaryColumn::index_of` is O(dict) linear scan; `index_of_many` builds a map each
  call.** `dictionary.rs:162-195`. Documented trade-off (lookups are once-per-query). Fine, but
  a repeatedly-queried dictionary re-builds the reverse map every `index_of_many` call — a
  cached reverse map would help hot literal-resolution paths.

- **L9. `DictionaryColumnI64` is entirely `#[allow(dead_code)]` / runtime-guarded-off.**
  `dictionary_i64.rs:102-113`, `dictionary_any.rs:81-86`. This is a deliberately-parked feature
  (i64 indices not wired through codegen). Clearly documented with re-enabling conditions. Not a
  bug, but it is ~500 lines of unreachable production code plus tests; flag for the orchestrator
  so it doesn't bit-rot.

---

### Performance opportunities

- **P1. Event-based deferred free (the big one).** Both `GpuBuffer::Drop` and
  `PinnedHostBuffer::Drop` do a blanket `cuStreamSynchronize` per recorded stream, stalling the
  dropping thread on each stream's *entire* trailing queue rather than just the work that
  touched the block. The code documents the full `cuEvent*`-based fix (`buffer.rs:832-864`,
  `:1199-1241`). This is the single highest-value perf+safety improvement and also closes C1.

- **P2. `recover_from_oom` all-or-nothing drain** — see M3. Incremental eviction avoids the
  cold-cache cliff.

- **P3. Pinned-memory pooling.** `PinnedHostBuffer` / `PinnedBuffer` call `cuMemAllocHost` /
  `cuMemFreeHost` per allocation with no pool. Pinned allocation is expensive (page-locking);
  the comments at `buffer.rs:946` and `async_copy.rs:150` both anticipate "a future bucketed
  pinned pool." For overlapped pipelines that allocate a pinned staging buffer per batch this
  is a measurable cost.

- **P4. `from_slice` is fully synchronous** (`buffer.rs:234-251`): `with_capacity` (which may
  hit the pool lock) then a blocking `cuMemcpyHtoD_v2`. The async path exists but isn't the
  default; pageable-memory uploads serialize. Wiring more executors onto pinned + async would
  reclaim PCIe/compute overlap.

- **P5. `StreamSet` linear dedup is fine (n≈1-2)** as documented — no action.

---

## 2. TEST ADEQUACY

### What exists (host-only, runs without a GPU)
- **`buffer.rs`:** strong host-only coverage — `round_up_to_alignment` (8 tests incl.
  overflow), `StreamSet` dedup / null-handling, `mark_stream_use` accumulation,
  `fence_all_streams` via the mockable `drop_fence_with` seam, `byte_len` overflow panic,
  empty-buffer Drop no-op, pinned zero-length paths. The mock-fence seam is a genuinely good
  pattern for testing Drop logic without a driver.
- **`async_copy.rs`:** sizing/`set_len`/`Deref` (stub-backed), stream-set dedup, multi-stream
  fence via the recording stub, length-mismatch guards before FFI, `Send` compile checks.
- **`cuda_sys.rs`:** `init` cache retry behavior (success latches, error not cached, flaky
  retry) via injected fake `cuInit`; `check()` branch coverage for success/stub/`CudaWithCode`.
- **`mem_pool.rs`:** the richest suite — byte-cap eviction, bucket-cap, env overrides, per-bucket
  FIFO, `evict_above_high_water`, `bucket_size` granularity table, OOM-injection latch, and a
  full `pool_watcher` test surface (driven against a local pool with mock `MemInfoFn`).
- **`dictionary*.rs`:** host-only dedupe (1M-row redundancy regression), null interleaving,
  width-safe `decode_index` (incl. `i32::MAX` / negative), `index_of_many` parity i32↔i64,
  dispatch-by-threshold. GPU round-trips are `#[ignore]`d.
- **`smart_ptrs.rs`:** empty-vec invariants, view/view_mut, `as_view` reborrow, C-1 view→parent
  tagging, `Send` compile checks.

### Estimated coverage
- **Pure host logic** (bucketing, dedup, stream-set bookkeeping, dictionary decode, init cache,
  pool policy): **~80-85%**. This is well-tested.
- **GPU/driver paths** (actual H2D/D2H, kernel-launch tagging end-to-end, real OOM, real
  eviction-during-kernel, Drop-during-DMA): **effectively 0% in CI** — every such test is
  `#[ignore]`d and gated on `BOLT_BENCH_GPU=1`. These are the safety-critical paths.

### Critical untested paths
1. **C1 / Drop-during-kernel** (the CRITICAL finding): no test launches a kernel, drops the
   owning `GpuVec` without syncing, and asserts the pool did not recycle the block. The mock
   fence seam *could* exercise this host-only by asserting the launch stream lands in the set.
2. **Wrong-thread free** (H1): no test drops a `GpuBuffer` on a thread with no current context.
3. **OOM recovery end-to-end** (`mem_pool`): the OOM latch tests injection at the *pool* layer,
   but the all-or-nothing drain's effect on *concurrent* queries (M3) is untested.
4. **Eviction racing alloc** (`evict_one` "Approx"/`BucketEmpty` fallback, `mem_pool.rs:1202-1238`):
   the race-handling branches are reasoned about in prose but have no concurrent stress test
   that actually drives an alloc between LRU-pop and bucket-lock.
5. **`to_pinned_async` short-read** (H3): untested.
6. **`from_dictionary_array` null dict value** (M6): untested.
7. **cudarc `from_raw_parts` with a short/misaligned `src`** (H4): untested (and would be UB).

### Specific tests to add
- Host-only: `launch_tag_via_kernel_args_lands_in_parent_set` — simulate `KernelArgs` retaining
  a `stream_set_ref` and tag; assert via `recorded_stream_count`. (Guards C1 mechanism.)
- Host-only: a `loom`-style or threaded stress test over `DeviceMemPool` (synthetic ptrs)
  driving concurrent `alloc`/`free` across distinct size classes, asserting no double-free in
  the `FREED` side-channel and `reconcile_total_bytes` converges. (Guards M4 / evict races.)
- Host-only: `from_dictionary_array_null_value_does_not_alias_empty_string`.
- `#[ignore]` GPU: `drop_during_inflight_dma_does_not_corrupt_next_alloc` — issue async H2D,
  drop without sync, alloc a new buffer in the same bucket, assert contents. (Guards C1/Drop.)
- `#[ignore]` GPU: drop a `GpuBuffer` on a spawned thread with no context (H1).

---

## 3. FEATURE / DIRECTION SUGGESTIONS

1. **Land the event-based pending-free pool.** It is already fully designed in-comment
   (`buffer.rs:832-864`). It simultaneously (a) closes the CRITICAL C1 hole structurally,
   (b) removes the per-Drop blanket-sync stall (P1), and (c) is the precondition for a pinned
   pool. This should be the top roadmap item for this module.

2. **Encode the launch-tagging invariant in the type system.** Make raw `device_ptr()`
   `pub(crate)` and require launches to go through a `LaunchToken`/`KernelArgs` that *consumes*
   the view and tags by construction, so "forgot to tag" becomes a compile error rather than a
   silent UAF.

3. **Pinned-memory pool** (P3) mirroring `DeviceMemPool` — both pinned types already cache
   `byte_len` "so a future bucketed pool can hook in cleanly."

4. **Unify the duplicated `StreamSet`.** `async_copy.rs:84-111` reimplements `buffer.rs`'s
   `StreamSet` verbatim solely because the latter's methods are module-private. Promote
   `buffer::StreamSet`'s API to `pub(crate)` and delete the copy (the comment at
   `async_copy.rs:68-80` explicitly asks for this). Reduces drift risk across the two
   safety-critical fence implementations.

5. **Multi-GPU.** `cudarc_backend` hard-wires device 0 (`GLOBAL_DEVICE` `OnceCell`,
   `cudarc_backend.rs:48-95`); `ensure_device(n)` silently ignores a differing ordinal after
   first init. The hand-rolled backend supports per-Engine contexts. A unified multi-GPU story
   (pool-per-device, view carries device affinity) is the natural next step.

6. **Telemetry surface** is good (`pool_stats`, OOM/proactive-eviction counters). Consider
   adding pinned-allocation byte totals and a histogram of bucket occupancy for VRAM tuning.

---

## Appendix: positive notes
- The `drop_fence_with` mock seam (`buffer.rs:723-749`, `async_copy.rs:483-511`) is an excellent
  way to unit-test Drop-time driver calls host-only.
- Integer-overflow discipline is strong across `checked_mul` in the FFI wrappers and dictionary
  paths (the two `byte_len` view sites at L2 are the exception).
- The pool lock-order reasoning (bucket→lru, never two lru shards) is carefully argued and the
  sharded-LRU fan-out scan is correct as described.
- `cudarc_backend`'s context-currency self-guarding on the free path (`cudarc_backend.rs:168-188`)
  is exactly the right instinct — it just needs to be mirrored on the default backend (H1).
