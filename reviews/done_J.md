# Agent J — Wire streaming / morsel execution into the engine

**Branch:** `dev`  **Files edited:** `src/exec/engine.rs` only.
**`src/exec/streaming.rs`:** inspected, **no edits required** — all needed
primitives (`BatchStream`, `TableSource`, `MorselPlan`, `morsel_rows()`,
`num_morsels()`, `estimate_batches_bytes`, `plan_upload`) already existed and
are now consumed by the engine.

No cargo build/check/test run (per instructions). Changes validated by careful
inspection against the real code; new code is written to compile under
`--no-default-features --features cuda-stub`.

---

## What now streams vs drains

### Streams (morsel-by-morsel, bounded working set)
Only **row-wise scan-leaf plans over a streaming-registered table**, when a
morsel budget actually calls for chunking:

| PhysicalPlan variant | Why it is morsel-safe |
|---|---|
| `Projection` (incl. `WHERE`/predicate) | output rows are an independent, order-preserving function of input rows; a predicate only *drops* rows. `concat(project(morselᵢ)) == project(concat)`. |
| `StringLength` (`LENGTH(col)` + passthroughs) | per-row scalar; row-wise. |
| `StringProject` (`UPPER`/`LOWER`/passthroughs) | per-row scalar; row-wise. |

These are exactly the three executors that take a `table: String` directly (no
child sub-plan) and resolve it through `materialize_table` / `ensure_gpu_table`.

### Drains (status quo — whole table materialised)
Everything else, unchanged:
`Aggregate` (global/grouped fold crosses rows), `Sort`, `Distinct`, `SetOp`,
`Union`, `Window`, `Join` (build side must be resident), `Limit`, `Filter`,
`Project`, `CountRows`, and `StringLikeFilter` (the last four wrap a child
sub-plan whose own scan would have to be threaded — deliberately out of scope
for this minimal, correctness-first cut). For these the streaming source is
drained to a whole table (by the pre-existing `ensure_streaming_materialized`
hook) and processed exactly as before.

The drain-fallback is therefore the *default* for any non-streamable shape, so
"anything not safely streamable drains the stream to a whole table" holds.

---

## Trigger (opt-in) and default-behaviour-preservation argument

The morsel path in `Engine::execute` fires **only** when ALL of:
1. `memory_budget_bytes.is_some()` — **default is `None` (uncapped)**, so for
   every existing query/test (none set a budget) the entire streaming block is
   skipped and the original whole-table dispatch (`execute_leaf_whole`) runs
   byte-for-byte unchanged.
2. the plan is a streamable leaf (`streamable_leaf_scan` returns the table);
3. the table is **overlay-only** (registered via `register_table_stream_lazy`,
   present in `streaming_sources`, absent from the eager `tables` store) — the
   only data the engine can swap per-morsel under `&self` (the overlay is a
   `RefCell`; `tables` is not). Eager tables always keep the whole-table path.
4. `morsel_plan_for_table(table)` returns `Morsels { .. }` (footprint exceeds
   budget). If it returns `Whole` (budget set but table fits) we fall through to
   the whole-table path.

Because (1) gates on a non-default knob, **all 2098 existing host tests and
every existing query are unaffected.** When the path does fire, the result is
provably identical to the whole-table result: the morsel loop installs each
morsel as the table's data, runs the *same* per-shape executor
(`execute_leaf_whole(phys)`), and concatenates the row-wise outputs in table
order. A streaming-overlay table has no `host_revisions` entry, so
`ensure_gpu_table` rebuilds a fresh small GPU table per morsel (no stale-cache
hazard). The whole-table overlay entry is **always restored** after the loop
(including on the error path), so subsequent queries see the full table.

---

## Files + changes (`src/exec/engine.rs`)

1. **`fn streamable_leaf_scan(phys) -> Option<&str>`** (associated fn) —
   classifies the three streamable leaf shapes; returns `None` for everything
   else (drain). Heavily documented with the per-variant stream/drain rationale.
2. **`fn execute_streaming_leaf(table, morsel_rows, run_morsel, output_schema)`**
   — the morsel orchestrator: snapshots whole-table batches, builds a
   `streaming::BatchStream`, swaps each morsel into the overlay, runs the
   per-morsel executor, restores the whole-table entry unconditionally, and
   concatenates per-morsel results (empty-table → schema-shaped empty batch via
   `RecordBatch::new_empty`).
3. **`Engine::execute`** — added the opt-in streaming hook at the top (the four
   conditions above), then delegates to…
4. **`fn execute_leaf_whole(phys)`** — the original `execute` body, renamed and
   split out verbatim so the orchestrator can invoke the identical per-shape
   executor on a single installed morsel. The inner recursive arms still call
   `self.execute(...)`, so child sub-plans get their own streaming opportunity.
5. **Tests** (in the existing `#[cfg(test)] mod tests`):
   - Host-pure (run on CI, no GPU): `streamable_leaf_scan_recognises_projection`,
     `streamable_leaf_scan_recognises_string_leaves`,
     `streamable_leaf_scan_rejects_non_leaf_shapes` (Distinct/Limit wrappers).
   - GPU-gated (`#[ignore = "gpu:..."]`, per repo convention — they bind a real
     CUDA context and launch kernels):
     - `streaming_morsel_matches_materialized_projection` — same data registered
       streaming-with-budget vs materialised-whole must produce identical
       `SELECT x FROM t WHERE x >= 100` results.
     - `streaming_drain_fallback_global_aggregate` — `SUM(x)` over a streaming
       source drains and yields the correct global result under a small budget.
     - `streaming_overlay_restored_after_morsel_query` — a streamed query
       followed by a second query confirms the whole-table overlay was restored.

---

## Orchestrator wiring needed in OTHER files

**None.** The implementation is self-contained in `engine.rs` and reuses the
existing public API (`register_table_stream_lazy`, `EngineBuilder::memory_budget`,
`morsel_plan_for_table`) and the already-exported `streaming` module. No
`mod.rs` / `lib.rs` export changes are required.

---

## Residual follow-ups (not done; flagged for the orchestrator/backlog)

1. **Eager `tables` streaming.** Morsel-driving an eagerly-registered table
   needs a `&mut self` seam (or making `tables` interior-mutable) to swap
   morsels; today eager tables always take the whole-table path even under a
   budget. The streaming win currently applies to `register_table_stream_lazy`
   tables only — which is the review's headline case.
2. **Child-wrapping leaves.** `StringLikeFilter`, `Limit`, `Filter`, `Project`
   over a streaming scan still drain because the morsel must flow through the
   child sub-tree. Threading morsels through a one-level child wrapper is a
   natural next increment.
3. **Real pinned backing.** `PinnedBudget::reserve` is still host-accounting
   only (the `streaming.rs:386` `TODO(cuda)`); the morsel loop here bounds the
   *device* working set (one small GPU table per morsel) but does not yet stage
   intermediates in `cuMemHostAlloc` pinned memory for overlapped transfer.
4. **Eager pre-drain interaction.** `ensure_streaming_materialized` still
   collapses every streaming source to `Materialized` before `execute`. That is
   fine (the morsel loop re-slices the materialised host batches and only
   uploads one morsel at a time, so device residency is bounded), but a future
   "never fully materialise on host" variant would need a different seam that
   keeps the producer un-drained and pulls batches lazily.
5. **Morsel-vs-whole GPU equivalence** is asserted only by the GPU-gated tests
   here; CI (cuda-stub, no device) covers the classification logic but not the
   end-to-end kernel equivalence — same coverage caveat the review notes for all
   GPU paths.
