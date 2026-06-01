# Bolt Query Planner / Optimizer Review — `src/plan/`

Scope: `src/plan/` (logical_plan, sql_frontend, optimizer rules, subquery, statistics,
string_literal_rewrite, explain, dataframe, rewrite, suggest) + planner tests in `tests/`.
Reviewer focus: correctness of optimizer rules (semantics preservation), SQL frontend gaps,
vulnerabilities, stubs, performance, and test coverage.

Overall the optimizer is unusually well-documented and conservative. Every rewrite pass carries an
explicit soundness argument, and most rules degrade to a safe no-op when in doubt. The one place that
argument breaks is projection pruning across a join with colliding column names.

---

## 1. CODE REVIEW

### CRITICAL

#### C1. Projection pruning drops the renamed right-side column of a join with a name collision
`src/plan/optimizer/projection_pruning.rs:199-245` (the `Join` arm), helper
`restrict_to_schema` at `:300-306`.

The join arm attributes each required column to a side by filtering the parent's `required` set
against the *child* schemas, which carry **bare** names (`a`, `k`). But the parent's `required` set
is expressed against the join's **combined output** schema, where a right-side column that collides
with a left name has been renamed to `right.<name>` by
`join_combined_schema` (`src/plan/logical_plan.rs:2398-2428`, rename rule at
`join_rename` `:2484`).

Concrete trace — `t1(k,a) JOIN t2(k,a) ON t1.k=t2.k`, then `SELECT customers.id`-style
`SELECT right.a`:
- combined schema = `[k, a, right.k, right.a]`; parent `required = {right.a}`.
- `on` uses bare names, so `join_cols = {k}`; `wanted = {right.a, k}`.
- `right_req = restrict_to_schema(wanted, rschema={k,a})` keeps only `{k}` because the string
  `right.a` is not a field of the right child and the bare `a` is not in `wanted`.
- `prune_scan` therefore narrows `t2` to `[k]`, **dropping `a`** — the very column the query selects.

Result: the optimized plan no longer produces the selected column. Depending on the executor this is
a schema mismatch error or a silently wrong result. This is a results-changing optimizer bug — the
defining "CRITICAL" category.

The in-tree probe test `review_probe_join_collision_rename`
(`projection_pruning.rs:445-491`) was clearly written to expose exactly this and only `eprintln!`s
the outcome rather than asserting — it is a latent-bug marker, not a guard. The integration test
`qualified_picks_left_side_id_in_join` (`tests/qualified_columns_test.rs:340-360`) only calls
`try_plan` (parse time); projection pruning runs later in the engine, so it does not cover this path.

Fix direction: in the join arm, translate the parent's combined-schema names back to child-local
names before restricting (invert the `right.`-prefix / `__N` suffix produced by `join_rename`, or
thread the combined schema through and map by position), so a required `right.a` maps to the right
child's bare `a`. Until fixed, the safe stopgap is to fall back to "keep both sides whole" whenever
any required name is not directly present in either child schema.

### HIGH

#### H1. No integration/e2e coverage of the optimizer on join-collision shapes
The collision rename rule is unit-tested at the schema level
(`logical_plan.rs:3081-3148`) and at parse time, but no test runs the **full optimizer pipeline +
execution** over `SELECT <right-collision-col> FROM a JOIN b ...`. This is why C1 went unnoticed.
Add an end-to-end `diff_duckdb`-style test selecting the renamed right column after a join, asserting
the executed result. (Test gap that directly masks a CRITICAL bug.)

#### H2. Correlation detector only sees the immediately-enclosing query's columns
`src/plan/subquery.rs:78-86` + `src/plan/sql_frontend.rs:371-380` (`outer_column_names`) +
`NameResolver` (`sql_frontend.rs:170-183`).

`NameResolver` holds no link to a parent resolver, so `outer_column_names()` returns only the columns
of the *immediate* enclosing query. For a doubly-nested subquery that correlates to the **outermost**
query, the grandparent's column name is absent from the middle resolver's outer set, so
`reject_if_correlated` does not classify it as correlated. The reference then falls through to the
"plain unknown column" path (`subquery.rs:302-314`) and surfaces as an `unknown column` error rather
than the precise "correlated subquery" rejection. This is **not** a wrong-results bug (the engine
errors out, it does not mis-plan), but it is a robustness/UX gap and a sharp edge if correlated
execution is ever partially wired up. Document the single-level limitation or thread a chained outer
scope.

### MEDIUM

#### M1. `flatten_chain` opaque-leaf assumption can re-introduce a cross product it then rejects — correct, but fragile
`src/plan/optimizer/join_reorder.rs:270-356`. The reorder only fires with ≥3 leaves and full
estimates, and `rebuild_left_deep` bails to `None` (keeping the original) if any equi-pair cannot be
placed or any pair is left over (`:332-336`, `:351-355`). This is sound. The fragility: a right child
that is itself a join is treated as one opaque leaf (`:290`) and costed via `RowEstimator::estimate`,
which for `StatsEstimator` returns `None` for non-`Scan` nodes — so any bushy shape disables
reordering for the whole chain. Functionally safe, but means reordering effectively only ever fires
on pure left-deep scan chains. Worth noting as a known limitation, not a bug.

#### M2. Const-fold float equality folding over NaN-producing constant subexpressions
`src/plan/optimizer/const_fold.rs:368-383`. Folding `a == b` / `a < b` etc. on `f64`/`f32` literals
uses Rust IEEE semantics, which match the GPU kernel for ordinary values and for NaN comparisons
(`NaN == NaN` → false, all orderings false). This is correct **iff** the runtime kernel uses the same
IEEE total/partial ordering. There is no test asserting a folded NaN comparison matches the executed
result. Add a fold test for `0.0/0.0`-derived NaN comparisons (or confirm the kernel) so a future
kernel change can't silently diverge from the folder. Low risk, but unverified.

#### M3. Predicate pushdown into the *preserved* side of an outer join — verify constant conjuncts
`src/plan/optimizer/predicate_pushdown.rs:460-502`. `classify_side` routes a column-free (constant)
conjunct arbitrarily to the **left** side (`:463-466`). For a `RightOuter` join, the left side is the
non-preserved side, and `can_push_into_join_side(RightOuter, Left)` correctly returns `false`
(`:498`), so a constant conjunct stays above — safe. But the comment at `:464-465` ("safe to push to
either side") is misleading given the very next guard depends on side. The behavior is correct; the
comment should be corrected so a future edit doesn't "optimize" the constant straight into a
non-preserved side. Comment-vs-code hazard.

#### M4. `string_literal_rewrite` does not rewrite GROUP BY / aggregate Utf8 literals
`src/plan/string_literal_rewrite.rs:845` (`// TODO: rewriting group_by / aggregate expressions over
Utf8 ...`). Equality/dictionary rewriting of string literals is applied in filter/project positions
but skipped for group-by/aggregate expressions. Not incorrect (it just forgoes the optimization
there), but an asymmetry that could surprise; flag as a known incomplete rewrite.

### LOW

- **L1. Plan cache caches the *unoptimized* logical plan.** `sql_frontend.rs:597-612`. `parse()`
  caches the lowered-but-not-optimized plan keyed by `(sql, schema_version)`; optimization runs
  per-call downstream. Correct (the version token is process-globally unique per mutation —
  `next_provider_version` `:418-427`, `Clone` takes a fresh token `:439-452`), but it means the
  optimizer re-runs on every cache hit. If optimization is hot, consider caching the optimized plan
  too. Performance, not correctness.
- **L2. Fixpoint change-detection via `{:?}` string compare.** `optimizer/mod.rs:141-157`. Formatting
  the whole plan to a `String` every sweep is O(plan size) allocation per iteration; fine at
  `MAX_FIXPOINT_ITERS = 4` but quadratic-ish on large plans. A structural `PartialEq`/hash would be
  cheaper. Minor.
- **L3. `EXISTS` / derived tables in FROM / `VALUES` / `WITH RECURSIVE` / CTE column-list aliases all
  unsupported** but cleanly rejected (`sql_frontend.rs:1082,1090,1288`, `:2088`, `subquery`
  EXISTS reject at `:4173`). No mis-planning; these are feature gaps, listed under §3.
- **L4. No `todo!`/`unimplemented!` in production paths** — confirmed across `src/plan`. Only doc
  TODOs (`dataframe.rs:57,203`) and test-only `panic!`s.

---

## 2. TESTS

### Current coverage (good breadth)
- **Unit tests live next to each pass** and are strong: predicate pushdown covers project/sort/union/
  setop/join/outer-join/limit (`predicate_pushdown.rs:638-936`); const-fold covers arithmetic,
  overflow guards, bool identities, all cast cases (`const_fold.rs:407-628`); join-reorder covers
  NoStats no-op, stats reorder, partial-stats no-op, outer-join exclusion
  (`join_reorder.rs:443-614`); pipeline ordering + fixpoint idempotence (`optimizer/mod.rs:159-571`).
- **Frontend / semantics**: `parser_tests.rs` (~106 cases), `semantics_e2e.rs` (26),
  `diff_duckdb.rs` (31) + `diff_duckdb_semantics.rs` (18) differential tests, `sql_proptest.rs`
  property tests, `qualified_columns_test.rs` (21), `did_you_mean_test.rs` (5).
- Subquery correlation has a dedicated detector with doc'd invariants.

Estimated coverage of `src/plan` is plausibly ~85% by line, but it is **uneven**: the per-pass unit
tests are dense, while the *interaction* of passes with real execution on adversarial schemas (join
collisions, outer-join NULL padding) is thin — exactly where C1 hides.

### Gaps / enhancements (priority order)
1. **(blocks C1) End-to-end optimizer+execution test for join name collisions** — select the renamed
   `right.<col>` after `a JOIN b` and diff against DuckDB. Convert the dormant
   `review_probe_join_collision_rename` probe into a hard assertion once C1 is fixed.
2. **Property test: every optimizer pass preserves output schema and result set.** Generate random
   valid plans (or random SQL via the existing proptest harness), run `run_to_fixpoint`, and assert
   `before.schema() == after.schema()` *and* equal executed results. A schema-equality proptest alone
   would have caught C1.
3. **Predicate-pushdown NULL-safety e2e**: LEFT/RIGHT/FULL join with a WHERE on the non-preserved
   side, asserting NULL-padded rows are handled as DuckDB does (unit test exists at the structural
   level; add an executed differential case).
4. **Correlated-subquery rejection at depth 2+** (covers H2): assert the error message and that no
   silently-wrong plan is produced.
5. **Const-fold runtime-equivalence**: NaN/inf float comparisons and int overflow boundaries folded
   vs. executed (covers M2).
6. **Join-reorder result-set invariance with stats**: currently asserts column set + structure;
   add an executed-result equality assertion for a 3+ way reorder.

---

## 3. NEW FEATURES / DIRECTIONS

- **Finish the CBO loop.** The statistics module (`statistics.rs`) and `StatsEstimator`
  (`join_reorder.rs:89-107`) are in place but only drive smallest-first leaf ordering on pure
  left-deep scan chains (M1). Next steps: cost bushy plans, feed selectivity estimates into a real
  DP/greedy join enumerator, and use `estimate_selectivity` to choose build vs. probe sides.
- **More rewrite rules**: predicate *push-up* / derivation of implied equalities for transitive join
  keys; `LIMIT` push-through `UNION ALL` and into sorted/topN; redundant `Distinct`/`Sort`
  elimination; `Project` expression pruning (the pass currently prunes scan columns but keeps all
  projection expressions — `projection_pruning.rs:98-113`); constant-propagation into `CASE`
  (const_fold leaves `Case` untouched, `const_fold.rs:80-89`).
- **CTEs**: only non-recursive, non-column-aliased CTEs are supported
  (`sql_frontend.rs:1063-1096`). Add `WITH ... (col, ...)` aliasing and inlining/materialization
  choice; eventually `WITH RECURSIVE`.
- **Correlated subqueries / EXISTS**: currently rejected (`subquery.rs`, `sql_frontend.rs:4173`).
  Decorrelation to semi/anti-joins would unlock a large class of analytic queries.
- **Window-function optimization**: window plans exist (`LogicalPlan::Window`) but `SELECT ... WINDOW`
  clause is rejected (`sql_frontend.rs:2628`) and there is no window-specific optimization (shared
  partition/sort reuse, push filters that are partition-key-only below the window).
- **Cache the optimized plan** (L1) and switch fixpoint change-detection off string formatting (L2).

---

## Severity summary

| ID | Severity | Area | Location |
|----|----------|------|----------|
| C1 | CRITICAL | Projection pruning drops renamed right-join column → wrong results | `projection_pruning.rs:199-306` |
| H1 | HIGH | No e2e optimizer test for join collisions (masks C1) | `tests/qualified_columns_test.rs:340` |
| H2 | HIGH | Correlation detector misses grandparent correlation (errors, not wrong) | `subquery.rs:78`, `sql_frontend.rs:371` |
| M1 | MEDIUM | Join reorder only fires on left-deep scan chains | `join_reorder.rs:270-356` |
| M2 | MEDIUM | Folded float NaN comparisons unverified vs. kernel | `const_fold.rs:368-383` |
| M3 | MEDIUM | Misleading "either side" comment for constant conjunct pushdown | `predicate_pushdown.rs:463-466` |
| M4 | MEDIUM | String-literal rewrite skips group_by/aggregate | `string_literal_rewrite.rs:845` |
| L1 | LOW | Plan cache stores unoptimized plan | `sql_frontend.rs:597-612` |
| L2 | LOW | Fixpoint change-detect via `{:?}` string | `optimizer/mod.rs:141-157` |
| L3 | LOW | Feature gaps cleanly rejected | various |
