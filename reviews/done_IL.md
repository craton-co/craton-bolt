# Agent IL — ILIKE (case-insensitive LIKE)

Branch `dev`. Implemented the deferred ILIKE feature per the "ILIKE plan" at
the end of `reviews/done_M.md`. No `cargo` run (per instructions); changes are
verified by inspection and are CUDA-free / `cuda-stub`-safe.

## Scope honored
Edited ONLY the eight allowed files:
`src/plan/logical_plan.rs`, `src/plan/sql_frontend.rs`, `src/plan/explain.rs`,
`src/plan/string_literal_rewrite.rs`, `src/exec/like.rs`,
`src/exec/expr_agg.rs`, `src/plan/physical_plan.rs`, `src/exec/string_like.rs`.

## ⚠️ COMPILE BLOCKERS the orchestrator MUST apply (out of my file scope)
Adding `case_insensitive: bool` to `Expr::Like` makes three out-of-scope
files that **explicitly destructure AND reconstruct** every `Expr::Like` field
(no `..`) non-exhaustive → hard compile errors. I could not touch them. Each
needs the new field threaded (always `case_insensitive: *case_insensitive`):

1. **`src/plan/optimizer/predicate_pushdown.rs`** (~lines 575–585,
   `rename_columns`). Add `case_insensitive,` to the destructure pattern and
   `case_insensitive: *case_insensitive,` to the reconstruct.
2. **`src/plan/optimizer/const_fold.rs`** (~lines 90–100, `fold_expr`). Add
   `case_insensitive,` to the destructure and `case_insensitive,` to the
   reconstruct (this arm moves the fields by value).
3. **`src/exec/subquery_resolve.rs`** (~lines 435–445, `resolve_expr`). Add
   `case_insensitive,` to the destructure and `case_insensitive,` to the
   reconstruct (by value).

All other out-of-scope `Expr::Like` matches use `..` and are unaffected
(`optimizer/expr_util.rs`, `statistics.rs`, `dataframe.rs`,
`exec/subquery_resolve` walk arms, `plan/subquery.rs` which already handles
`SqlExpr::Like | SqlExpr::ILike` together, plus the `tests/*.rs` files —
`like_test.rs`, `is_null_test.rs` — all use `..` or the unchanged 4-arg
`host_like` signature, so they keep compiling untouched).

No mod.rs / lib.rs export changes are required: `PatternMatcher::compile_ci`
is a new method on an already-exported type, and `host_like`'s public
signature is unchanged.

## Changes by file

### `src/plan/logical_plan.rs`
- Added `case_insensitive: bool` to the `Expr::Like` struct variant (doc'd:
  `true` = ILIKE, both pattern and input are Unicode case-folded).
- `dtype_depth` arm already used `..` (ILIKE type-checks identically to LIKE →
  `Bool` over a `Utf8` operand). Updated the two `#[cfg(test)]` constructors.

### `src/plan/sql_frontend.rs`
- Existing `SqlExpr::Like` lowering arm now sets `case_insensitive: false`.
- **New `SqlExpr::ILike { .. }` arm** mirroring the LIKE arm verbatim
  (single-char ESCAPE validation, string-literal-only pattern, NOT-ILIKE via
  `negated`) and setting `case_insensitive: true`. sqlparser 0.52 surfaces
  ILIKE as a distinct `SqlExpr::ILike` with fields identical to `SqlExpr::Like`.
- `contains_aggregate` arm extended to `SqlExpr::Like | SqlExpr::ILike`.
- New parse tests: `parse_ilike_sets_case_insensitive_flag`,
  `parse_plain_like_is_case_sensitive` (regression guard),
  `parse_not_ilike_sets_negated_and_case_insensitive`.

### `src/exec/like.rs` (the matcher)
- `PatternMatcher` gained a `case_insensitive` field.
- `compile(pattern, escape)` now delegates to new
  `compile_ci(pattern, escape, case_insensitive)`. When case-insensitive,
  BOTH the pattern and the escape char are `to_lowercase`-folded at compile
  time (helper `fold_char` for the single escape char); the case-sensitive
  path is byte-identical to before.
- `matches(s)` folds the input with `to_lowercase` when case-insensitive, then
  dispatches via the unchanged shape logic (extracted into `matches_folded`).
- New tests: case-insensitive matching across all shapes, genuine-mismatch
  rejection, Unicode folding, **plain-LIKE-stays-case-sensitive** regression,
  `host_like` default is case-sensitive, ILIKE+ESCAPE composition.

### `src/exec/expr_agg.rs` (host eval)
- `Expr::Like` match arm threads `*case_insensitive` into `eval_like`.
- `eval_like` gained a `case_insensitive` param and calls
  `PatternMatcher::compile_ci(pattern, escape, case_insensitive)`. NULL 3VL
  and `negated` handling unchanged (so `NULL ILIKE 'x'` = NULL).
- New host-eval tests: `ilike_matches_across_case_and_propagates_null`,
  `not_ilike_inverts_and_preserves_null`,
  `plain_like_host_eval_is_case_sensitive`.

### `src/plan/explain.rs`
- `Expr::Like` render arm now prints `ILIKE` / `NOT ILIKE` vs `LIKE` /
  `NOT LIKE` based on `case_insensitive`.

### `src/plan/string_literal_rewrite.rs`
- Threaded `case_insensitive` through the destructure + preserved-`Expr::Like`
  reconstruct.
- **Correctness gate:** the dictionary-precompute lowering (`col LIKE 'pat'`
  over a dict column → OR-of-index-equalities via the case-SENSITIVE
  `LiteralResolver::like_match_indices`) is now skipped when
  `case_insensitive` is true. ILIKE therefore always falls through to the
  host `Expr::Like` path, never silently producing case-sensitive results.
- Updated all 11 `#[cfg(test)]` constructors + the one explicit destructure.

### `src/plan/physical_plan.rs`
- `substitute_one_depth` Like reconstruct threads `case_insensitive`.
- **`try_lower_string_like_filter`** (the GPU `StringLikeFilter` lowering):
  destructure now binds `case_insensitive`, and the function returns
  `Ok(None)` (host fallback) when it is true — the GPU per-row kernel matches
  raw bytes and has no case-folding, so routing ILIKE there would be wrong.
- The GPU-codegen reject arm (`Expr::Like { .. } => Err`) and all the
  expr-walk helper arms use `..` and are unaffected.

### `src/exec/string_like.rs` (GPU device path + host mirror)
- No functional change needed: ILIKE is gated OUT of this path upstream in
  `physical_plan::try_lower_string_like_filter`, so every pattern reaching
  `decompose_like_pattern` / the device mirror is case-sensitive `LIKE`.
  Added a doc note to `decompose_like_pattern` recording that contract.

## Where ILIKE executes — GPU vs host
**Always host.** ILIKE never runs on the GPU:
- Dict-column path: gated off in `string_literal_rewrite` → preserved as
  `Expr::Like`.
- Non-dict GPU `StringLikeFilter` path: gated off in
  `physical_plan::try_lower_string_like_filter` (returns `None`).
Both fall through to the host `Expr::Like` filter, executed via
`exec::filter::execute_filter` → `expr_agg::eval_expr` → `eval_like` →
`PatternMatcher::compile_ci(.., case_insensitive=true)`. Case-sensitive plain
`LIKE` keeps its existing GPU/dict fast paths byte-for-byte unchanged.

## Residuals / notes
- **Compile blockers (sections 1–3 above) are mandatory** before the tree
  builds. They are trivial mechanical field-threading.
- ESCAPE composes with ILIKE (folded escape char + folded pattern) and is
  covered by a unit test; the dict and GPU fast paths reject ESCAPE anyway, so
  ILIKE+ESCAPE always runs host-side.
- `to_lowercase`-based folding is simple-fold (per the plan's recommended
  approach), not full Unicode case-folding (e.g. it lowercases rather than
  case-folds ß). This matches the plan's "case-fold BOTH sides with
  to_lowercase" instruction and is consistent on both pattern and input.
- The `fold_char` helper takes the first scalar of a multi-scalar lowercase
  expansion for the (rare) cased single-char escape; escape chars are almost
  always case-neutral ASCII, so this is the identity in practice.
- `host_like`'s public 4-arg signature was intentionally left unchanged to
  avoid breaking out-of-scope callers (`tests/like_test.rs`); ILIKE flows
  through `eval_like` → `compile_ci` instead.
