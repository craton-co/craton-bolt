# PB — StreamSet de-duplication (cuda)

## Verdict: SAFE merge. Done.

The two `StreamSet` types were **functionally identical**, so the duplicate in
`async_copy.rs` was deleted and that module now consumes the canonical
`pub(crate) StreamSet` from `buffer.rs`. Behavior-preserving; no semantic change
to stream tracking, `Drop` fencing, or A2's deferred-free path.

## 1. Diff of the two definitions (before)

| Aspect | `buffer.rs:52` | `async_copy.rs:85` | Equivalent? |
|---|---|---|---|
| Representation | `streams: Vec<CUstream>` | `streams: Vec<CUstream>` | yes |
| Derives | `#[derive(Default)]` | `#[derive(Default)]` | yes |
| `insert` | linear `contains` dedup, push | identical | yes |
| `len` | `self.streams.len()` | identical | yes |
| `is_empty` | `self.streams.is_empty()` | identical | yes |
| Ordering / capacity | first-seen order, no pre-cap | identical | yes |

No behavioral difference (dedup semantics, ordering, empty/len) — a clean merge,
not the "leave them separate" escape hatch.

## 2. Why a tiny `buffer.rs` edit was unavoidable (and why it's safe)

`async_copy` could *name* `buffer::StreamSet` (already `pub(crate)`) but could
not *operate* on it: `insert` / `len` / `is_empty` and the `streams` field were
all **module-private** to `buffer`. async_copy's own comment anticipated exactly
this: "If `buffer::StreamSet`'s API is ever made `pub(crate)`, this type should
be deleted in favour of importing it."

Reuse (rule 2) is impossible without widening that visibility, which conflicts
with rule 3 ("keep it exactly as A2 left it"). I resolved it with the **minimal
additive** change the task explicitly permits ("...unless additive"):

- Widened `insert` / `len` / `is_empty` from private to `pub(crate)`. Widening
  private → `pub(crate)` cannot change any existing caller's behavior or break
  compilation — purely additive.
- Added one `pub(crate) fn iter()` accessor so async_copy's `Drop` fence loop can
  walk the set **without** exposing the private `streams` field (the field stays
  encapsulated; buffer.rs's own in-module `Drop`/deferred-free loops still touch
  `streams` directly, untouched).

**No logic, representation, derive, field, `StreamSetRef`, `Drop` fencing, or
deferred-free path in `buffer.rs` was changed.** A2's work is intact.

## 3. async_copy.rs changes

- Deleted the duplicate `struct StreamSet` + `impl StreamSet` (~46 lines incl.
  doc), replaced with a short comment + `use crate::cuda::buffer::StreamSet;`.
- `fence_all_streams` now iterates via `streams.iter()` instead of the
  now-private `streams.streams` field.
- Updated the stale stub-`Drop` comment about keeping `is_empty`/`len` "alive".
- Kept the host-runnable `#[cfg(test)] stream_set_dedups_and_treats_null_as_real_handle`
  test — it now exercises the canonical type from async_copy's test module,
  confirming reuse works (dedup + null-handle-as-distinct).

## 4. cfg / lint correctness (must compile under `--no-default-features --features cuda-stub`)

`iter()`'s only caller is async_copy's `fence_all_streams`, gated
`#[cfg(any(not(feature = "cuda-stub"), test))]`; `iter()` carries the identical
gate. Verified across all four configs (cuda non-test, cuda test, cuda-stub
non-test, cuda-stub test): no dead symbol, no missing symbol. `len`/`is_empty`
stay ungated and are referenced from buffer.rs regardless, so the stub-`Drop`
"keep alive" read is now just a field-touch canary.

## 5. LOC

- Removed from `async_copy.rs`: ~46 lines (the duplicate `struct` + `impl` +
  its rationale doc), replaced by ~11 lines of comment + 1 `use`.
- Net duplicate logic removed: the entire second `StreamSet` definition/impl.
- Added to `buffer.rs`: ~13 lines (one additive `iter()` accessor + doc;
  visibility keyword changes on 3 existing methods).

## Files changed (only the two permitted)
- `C:\Projects\bolt\src\cuda\buffer.rs`
- `C:\Projects\bolt\src\cuda\async_copy.rs`

No other file touched (mod.rs / lib.rs untouched). No cargo build/check/test run;
verified by inspection.
