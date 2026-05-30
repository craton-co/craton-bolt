# Agent H — doc remediation (done)

Branch `dev`. Only the seven owned docs were edited; no code, no README,
no Cargo.toml, no other docs. cargo was not run. Every fact was verified
against 0.7 source before writing.

## 1. docs/ENV_VARS.md
- **Quick-start matrix**: added four previously-undocumented but
  actively-read vars — `BOLT_PREFIX_SCAN_ALGO`, `BOLT_HASH_ALGO`,
  `BOLT_HASH_PROBE_TILED`, `BOLT_SORT_USE_GRAPH` — with defaults and
  accepted values.
- Replaced the false "Not present in this build" entries for those four
  with a new **"Internal / unstable kernel selectors"** section (full
  per-var semantics, accepted values, defaults, source file:line). They
  are flagged internal/unstable rather than "has no effect."
- Kept `CRATON_DISTINCT_HOST_MAX_ROWS` and `CRATON_PLAN_CACHE_SIZE` in
  "Not present" (verified they are compile-time constants, not env reads).
- Verified semantics in code:
  - `BOLT_PREFIX_SCAN_ALGO` — `gpu_compact.rs:919-938` (`blelloch` /
    `lookback` / default Hillis-Steele, case-insensitive, read per call).
  - `BOLT_HASH_ALGO` — `groupby.rs:661-666` (`robin_hood`/`rh` →
    Robin Hood keys kernel; default linear-probe).
  - `BOLT_HASH_PROBE_TILED` — `gpu_join.rs:1004-1017` (truthy: any
    non-empty value other than `0`/`false`; default off).
  - `BOLT_SORT_USE_GRAPH` — `gpu_sort.rs:1722-1734` (exactly `"1"`;
    gate at `:1942`; default off).

## 2. docs/ARCHITECTURE.md
- Removed the false bullet "No CTEs, subqueries, or window functions. The
  parser rejects them outright." Replaced with an accurate statement:
  CTEs / uncorrelated subqueries / window functions are supported in 0.7
  (host-side where no GPU lowering exists); only **correlated** subqueries
  and **recursive** CTEs remain out of scope. Cross-checked against
  SQL_REFERENCE.md:42-43,404-405.

## 3. docs/MIGRATION_GUIDE.md
- Fixed the non-compiling `register_table_stream` snippet. It is an
  `Engine` method (`engine.rs:1255`), not an `EngineBuilder` method, and
  its real signature takes `(name, schema: Schema, batches: I)` where
  `I: IntoIterator<Item = BoltResult<RecordBatch>>`. New snippet calls it
  on `&mut engine` after `build()`, passes a declared `Schema`, and uses
  `Ok(...)`-wrapped batches.

## 4. docs/JIT_PIPELINE.md
- §1 (`:48-51`): removed "CTEs, subqueries, window functions … still
  rejected at parse time" and "a single JOIN per SELECT" / "at most one
  joined table." Now states non-recursive CTEs, uncorrelated subqueries,
  window functions, and **multiple joins per SELECT** are accepted in 0.7;
  small-cardinality non-equi INNER runs host nested-loop.
- §8 "What's not codegened": updated the Joins bullet (multiple joins
  per SELECT; non-equi handled via host nested-loop, not parse-time
  reject); "Window functions" now "parsed and executed (host-side), not
  yet codegened"; the CASE/CAST/COALESCE/NULLIF bullet rewritten — they
  **are** modeled in the AST and lower to GPU for numeric/Bool result
  dtypes as of 0.7 (verified `logical_plan.rs` `Expr::CaseWhen`/`Cast`;
  SQL_REFERENCE.md:118-122).
- Removed the dangling reference to unpublished `docs/rust_cuda/`;
  repointed at the real `kernels/` crate + `build.rs` `cuda_builder`.

## 5. docs/COMPETITIVE_BENCHMARKING.md
- §1 category list: "Single-batch in-memory" → "Multi-batch in-memory"
  (resolves the self-contradiction with the `:13` TL;DR, which already
  said multi-batch). "no plan cache, no kernel cache beyond PTX" →
  documents the plan cache, KernelSpec module cache, and PTX cache.
  "INNER JOIN … <equi>" SQL list expanded to the real 0.7 surface (all
  join types, set ops, CTEs, uncorrelated subqueries, window functions).

## 6. docs/GROUPBY_PERF.md
- Reconciled the self-contradiction. Added a **"Current results
  (post-optimization)"** section reproducing the canonical BENCHMARKS.md
  §1 numbers (q1 51.4 / q2 384 / q3 219⭐ / q5 237⭐; wins on q3/q5).
- Relabelled the old headline table as **"The pre-optimization baseline
  (historical)"** (q1 269 / q2 406 / q3 704 / q5 770) and reframed the
  "loses on every query" verdict + "Why we lose" → "Why the baseline
  lost" in past tense, explicitly noting these are the pre-fast-path
  numbers, not current.
- Updated the header NOTE and the "What to do about it" paragraph so the
  doc no longer presents the baseline as current.

## 7. docs/SQL_REFERENCE.md
- Added a dedicated **"IS [NOT] NULL"** subsection under Operators (after
  LIKE). Documents result dtype (Bool), SQL semantics, and the three
  execution tiers verified in code: GPU `Op::IsNullCheck` for bare
  nullable columns (`physical_plan.rs:146-158`), plan-time constant
  folding for non-nullable / literal operands, and host-side fallback for
  compound operands (`predicate_contains_unary` → `expr_agg.rs`). Notes
  the COUNT(DISTINCT) internal IS NOT NULL filter and the aggregate-`pre`
  gap.
