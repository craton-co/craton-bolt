# Agent F — remediation report

Branch `dev`, root `C:\Projects\bolt`. Two findings implemented. Only the two
owned files were edited:

- `src/exec/subquery_resolve.rs`
- `src/exec/dict_registry.rs`

No `cargo build/check/test` was run (per instructions). Code written to compile
correctly; reasoning on borrow/lifetime/type points captured below.

---

## F-6 (MEDIUM) — `NOT IN (subquery)` with NULL in the set

**File:** `src/exec/subquery_resolve.rs`, fn `build_in_predicate`.

**Before:** NULLs were dropped from the value set unconditionally, then a
`<>`/`AND` fold was built over the non-NULL elements for the negated form. This
let rows pass `x NOT IN (SELECT nullable_col …)` when the set contained a NULL —
a wrong-results bug (classic SQL footgun).

**After:** strict SQL three-valued logic. The set is scanned for any NULL up
front (`set_has_null`). For the **negated** form, if the set contains any NULL
the predicate is UNKNOWN for every row (a match → FALSE, a non-match → NULL), so
it folds straight to `Expr::Literal(Bool(false))` → **no rows pass**. The
non-negated `IN` form is unchanged (NULLs dropped; row matches iff it equals a
non-NULL element). The all-NULL negated case is subsumed by the same early
return.

**Tests added** (in the existing `#[cfg(test)] mod tests`):
- `not_in_with_null_in_set_excludes_all_rows` — `{1, 2, NULL}` negated → `Bool(false)`.
- `not_in_with_only_null_set_excludes_all_rows` — `{NULL}` negated → `Bool(false)`.
- `not_in_without_null_builds_and_of_inequalities` — `{1, 2}` negated → normal `AND` of `<>`.
- `in_with_null_in_set_keeps_non_null_membership` — non-negated `{7, NULL}` → bare `x = 7` (NULL dropped, not collapsed).
- `probe_expr_preserved_for_null_free_set` — probe value NULL: builder embeds the probe expr verbatim; runtime `=`/`<>` 3VL handles exclusion.

Existing test `build_in_only_nulls_set` (non-negated all-NULL → `Bool(false)`)
still holds and was verified against the new control flow.

---

## F-7 (MEDIUM) — Dictionary registry cross-table same-name column collision

**File:** `src/exec/dict_registry.rs`, fn `rewrite_plan` (+ new helper
`dicts_conflict`, + module/struct/method doc updates).

**Root cause:** `StringPredicateRewriter` is keyed by **unqualified** column
name (the plan's `Expr::Column` references are themselves unqualified). When a
plan scans two tables exposing a same-named Utf8 column with **different**
dictionaries — reachable today via `UNION`/`SetOp` since `collect_scan_tables`
recurses into both children — the old loop registered each, last-write-wins, and
folded predicates against the wrong index space (silent wrong results).

**Fix chosen (backward-compatible, no caller signature change):** instead of
threading a qualified `(table, column)` key (which would force every caller —
including ones in files I cannot edit, e.g. `engine.rs` — to pass a table
identifier into the rewriter's unqualified-column-keyed `register`), I detect
the collision inside `rewrite_plan` using data the registry already owns. For
each Utf8 column *name* across the (deduplicated) scanned tables I compare the
actual decoded dictionary contents via `dicts_conflict` (`a.dictionary() != b.dictionary()`).
A column name whose dictionaries **conflict** (different values or order →
different fold target) is **poisoned**: dropped from the rewriter entirely, so
its predicates fall back to the always-correct host string-comparison path. This
is exactly the "at minimum, bail rather than fold against the wrong dictionary"
remedy the review names.

Behaviour preserved:
- **Single-table plans** never poison anything — the existing fast path is byte-for-byte the same fold.
- **Multi-scan plans whose same-named columns carry identical dictionaries** still fold (single well-defined index space).
- A table appearing twice in one plan (self-`UNION`) is deduped first, so it is not mistaken for a self-collision.

Comparing `dictionary()` slices is exact: the slot index a literal resolves to
is a pure function of that slice (`DictionaryColumn::index_of` is a positional
scan), so identical slices guarantee identical folding and any difference is a
genuine collision.

**Tests added** (in the existing `#[cfg(test)] mod tests`):
- `rewrite_plan_poisons_conflicting_same_name_column` (`#[ignore = "gpu:string"]`) — two tables, `region` dicts `["US","EU"]` vs `["JP","AU"]`, UNION over both scans; asserts the `region = 'US'` predicate in **both** branches stays an unfolded `Column("region") = Utf8("US")` (proves the two scans keep distinct dictionaries, no aliasing).
- `rewrite_plan_folds_identical_same_name_column` (`#[ignore]`) — control: identical dicts on both tables still fold to `__idx_region = 1`.
- `rewrite_plan_single_table_still_folds` (`#[ignore]`) — regression: lone scan still folds.
- `dicts_conflict_contract_is_slice_equality` (runs without CUDA) — pure-host assertion of the exact comparison `dicts_conflict` encodes (identical → no conflict; different value or different order → conflict). Added because constructing a real `DictionaryColumnAny` requires CUDA (`from_string_array` uploads to the device), so the end-to-end poison tests follow the file's `#[ignore = "gpu:string"]` convention while this one exercises the collision predicate on the host.

Added private helpers `utf8_batch` / `utf8_scan` in the test module.

---

## Call-sites the orchestrator should be aware of

- **No public signature changed.** `rewrite_plan(&self, &LogicalPlan)` keeps its
  exact signature; `register_table` / `register_dictionary_column` /
  `StringPredicateRewriter::register` are all untouched. No caller in any other
  file needs to be updated for these changes to compile or behave correctly.

- **Forward note (not required now):** poisoning is the safe interim remedy. The
  full fix the review's "directions" section calls for — qualifying dictionary
  lookups by `(table, column)` so same-named columns from different relations can
  *both* fold once JOIN/relation-qualified column references exist — still needs
  the rewriter to learn relation-qualified column keys and the plan to carry
  qualified `Expr::Column`s. That is a larger cross-file change (touches
  `string_literal_rewrite.rs` and the planner) and was intentionally NOT done
  here, since it requires editing files outside my allowed set. Until then,
  poisoning guarantees correctness (host fallback) for the multi-scan case.

## Verification notes
- Borrow checker: in `rewrite_plan` the dictionary comparison is resolved into a
  copied `Option<bool>` (`conflict`) *before* any mutation of `chosen`, so no
  borrow of `chosen` is held across `insert`/`remove`. `&str`/`&DictionaryColumnAny`
  keys/values inserted into the working maps all borrow from `&self.by_table`,
  which outlives the function — identical lifetime shape to the original loop.
- Did not run the compiler. The two `#[ignore = "gpu:string"]` poison/control
  tests need a CUDA runtime (same as the sibling `register_table_*` tests);
  the F-6 tests and `dicts_conflict_contract_is_slice_equality` are pure host.
