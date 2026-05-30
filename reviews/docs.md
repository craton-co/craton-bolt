# Craton Bolt — Documentation Review

**Crate:** `craton-bolt` v0.7.0 · **License:** Apache-2.0 · **Owner:** Craton Software Company
**Reviewed:** 2026-05-30 · **Scope:** all root docs + `docs/*` cross-checked against `Cargo.toml`, `src/lib.rs`, env-var call sites, `examples/`.

---

## Executive summary

The documentation set is unusually large and, for the most part, unusually good: README, USER_GUIDE, SQL_REFERENCE, INSTALL, DEVELOPMENT, MIGRATION_GUIDE, API_SURFACE, and the benchmarking pair are professionally written and broadly accurate. However, the corpus is **not release-clean**. There are:

- **Two outright-wrong factual claims that will mislead users** (`ENV_VARS.md` "Not present in this build" list; `ARCHITECTURE.md` "No CTEs, subqueries, or window functions").
- **A wrong code sample in the migration guide** (`register_table_stream` shown as an `EngineBuilder` method; it is an `Engine` method).
- **Stale "rejected at parse time" claims** in `JIT_PIPELINE.md` and `COMPETITIVE_BENCHMARKING.md` contradicting the v0.7 surface.
- **An internally self-contradicting perf doc** (`GROUPBY_PERF.md`: "Status: Implemented" up top, "Craton Bolt loses on every query / current kernel" in the body, with numbers that don't match the BENCHMARKS.md they cite).
- **Two clearly-internal scratch documents** (`CUDARC_ADOPTION.md`, `CUDA_OXIDE_SWEEP.md`) that read as engineering working-notes and should not ship as-is in a public `docs/`.

Naming/contact facts are internally **consistent**: GitHub org `craton-co`, email domain `cratonsoftware.com`, copyright "Craton Software Company". The `craton.com.ar` domain referenced in the review brief does **not** appear anywhere in the repo's docs or code.

---

## 1. Per-document review

### README.md — purpose: front door. Quality: high.
- Accurate on version (0.7.0), MSRV (1.74), platform support, CUDA ≥ 12, sm_70 floor, supported SQL. Cross-checked against `Cargo.toml:3,14` and `README.md:3` badges (MSRV 1.74 ✓, version badge dynamic ✓).
- **Minor:** the "Project layout" tree (`README.md:186-202`) lists every `docs/` file **except `JIT_PIPELINE.md`**, even though the README links to `docs/JIT_PIPELINE.md` twice (`README.md:131`, and it is a core doc). Add it to the tree.
- **Minor:** badges point at `crates.io/craton-bolt`, `docs.rs/craton-bolt`, and the `craton-co/craton-bolt` CI workflow — all 404 until first publish/repo creation. Self-resolving; acceptable for pre-publish.
- The performance numbers in the README (`README.md:144-157`) match `BENCHMARKS.md §1` exactly (q1 51.4 / q3 219 / q5 237 ms) — good, no drift between README and BENCHMARKS.

### CHANGELOG.md — purpose: release history. Quality: high.
- Thorough, Keep-a-Changelog format. Explains 0.2.0 and 0.4.0 skips (`CHANGELOG.md:5-7,227-233`). Compare-links present (`:541-545`).
- Error-variant lists and feature descriptions cross-check against `src/error.rs` and `Cargo.toml`. No fabrications found.
- **Minor:** `[Unreleased]` section is empty (`:9`) — fine, but RELEASING.md step 10.1 asks to confirm it is "closed and named"; it is currently just an empty header.

### ROADMAP.md — purpose: forward plan + known gaps. Quality: high.
- Accurately reflects v0.7 carry-overs and remaining gaps. The "Known limitations … as of 0.7.0" section (`:136-153`) is honest and matches code (radix sort still `BOLT_GPU_SORT`-gated; `persistent_cache` builder knob not wired to `build()`).
- **Minor:** "ClickBench numbers published per release" (`:198`) is a 1.0 aspiration listed under "1.0 — public API freeze"; fine, but BENCHMARKS.md uses h2o.ai, not ClickBench, so a reader may expect ClickBench numbers that don't exist yet. Cosmetic.

### CONTRIBUTING.md — purpose: contributor onboarding. Quality: high.
- DCO sign-off documented; references `DCO` file. References `docs/DEVELOPMENT.md`, `SQL_REFERENCE.md`. No version-specific claims to drift.
- **Minor:** table row "CUDA driver / NVRTC tweaks" (`:22`) — the project explicitly does **not** use NVRTC (FAQ Q1, CHANGELOG "Removed `BoltError::Nvrtc`"). The word "NVRTC" here is misleading shorthand for "driver-layer"; reword to avoid implying an NVRTC path exists.

### RELEASING.md — purpose: maintainer release checklist. Quality: high.
- Realistic and detailed. The v13.2 `cuda.lib`/`__imp_*` workaround (`:104-108`) matches `INSTALL.md` and `build.rs` behavior. References `cargo deny check licenses` and a `deny.toml` "wired into CI" (`:118-119`) — verify `deny.toml` actually exists at repo root before relying on this (not confirmed in this review).
- Internally consistent with BENCHMARKS.md hardware (RTX 2060) and the Jetson/aarch64 "untested" stance.

### SECURITY.md — purpose: disclosure policy. Quality: excellent.
- Supported-version table (`0.7.x` ✓ / `<0.7` ✗), `security@cratonsoftware.com`, GitHub private advisories, 5-day ack / 90-day disclosure. Notes 0.2.0/0.4.0 skips. No issues.

### CODE_OF_CONDUCT.md — Contributor Covenant 2.1, `conduct@cratonsoftware.com`. Quality: standard/fine. No issues.

### MAINTAINERS.md / CODEOWNERS — purpose: ownership. Quality: honest about placeholder state.
- Both correctly disclose that the `@craton-co/craton-bolt-maintainers` GitHub team **does not yet exist** and the CODEOWNERS auto-review is a no-op until provisioned (`CODEOWNERS:6-12`, `MAINTAINERS.md:5-8,16-18`). This is honest but is a **release gap**: shipping a CODEOWNERS that references a non-existent team means any "require CODEOWNER review" branch rule is unsatisfiable. Either create the team or point CODEOWNERS at a real handle before public launch.

### NOTICE — purpose: third-party attribution. Quality: high.
- Lists arrow-{,array,buffer,schema}, sqlparser, and dev/build deps with licenses. Cross-checked against `Cargo.toml` dependencies — all present deps accounted for (dashmap, cudarc, criterion, polars, duckdb, proptest, insta, cuda_builder). `tracing`, `bytemuck`, `libc`, `parking_lot`, `once_cell`, `thiserror`, `log` covered. CUDA EULA disclaimer present. No omissions found.

### docs/API_SURFACE.md — purpose: public-surface stability tiers. Quality: very high.
- Enumerated against `0.7.0` (`:11`). Cross-checked key claims against `src/lib.rs` and `src/error.rs`: `BoltError` variants (`:83`) match `error.rs` exactly (Cuda, CudaWithCode, Sql, SqlWithSpan, Plan, Type, Memory, Io, GpuCapacity, Other); `#[non_exhaustive]` ✓; `span()` accessor ✓ (`error.rs:142`). Env-var constants table (`:190-197`) matches `src/lib.rs:198-208` re-exports (`POOL_STATS_ENV`, `CAP_ENV_VAR`, `STREAMING_INTERN_ENV_VAR`, `PTX_CACHE_CAP_ENV`, `DISK_PTX_CACHE_ENV`) and the constant values match the code.
- **Gap:** `Engine::register_table_stream_lazy` exists in code (`src/exec/engine.rs:1394`) but is not listed in the `Engine` stable-methods row (`:93`). Either document it or confirm it is intentionally `#[doc(hidden)]`/internal.

### docs/ENV_VARS.md — purpose: env-var reference. Quality: good layout, **but contains the single most serious factual error in the set.**
- The matrix and per-var sections for the *documented* vars are accurate and cite real call sites (`CRATON_BOLT_POOL_MAX_BYTES`, `CRATON_BOLT_POOL_BUCKET_CAP`, `CRATON_BOLT_PTX_CACHE_CAP`, `BOLT_POOL_STATS_INTERVAL_SECS`, `BOLT_POOL_WATCH_*`, `BOLT_GPU_JOIN_TABLE_CAP_MB`, `BOLT_GPU_JOIN_STREAMING_INTERN`, `BOLT_PTX_CACHE_DIR`, `BOLT_GPU_SORT`, `BOLT_BENCH_*`, `CUDA_PATH`, `CARGO_FEATURE_CUDA_STUB`). All verified to exist in code.
- **WRONG — "Not present in this build" section (`ENV_VARS.md:254-272`)** claims the following "are NOT honoured by the current codebase. Setting them has no effect." That is false for at least four of them:
  - `BOLT_PREFIX_SCAN_ALGO` — **actively read** and dispatched on in `src/exec/gpu_compact.rs:920` (`prefix_scan_algo_selection`: `blelloch` / `lookback` / default Hillis-Steele). Doc says "no runtime selector"; there is one.
  - `BOLT_HASH_ALGO` — **actively read** in `src/exec/groupby.rs:661` (`robin_hood`/`rh` selects the Robin Hood keys kernel). Doc says "selected by host policy, not by env var"; it is selected by env var.
  - `BOLT_HASH_PROBE_TILED` — **actively read** in `src/exec/gpu_join.rs:1010` (`probe_tiled_enabled`). Doc says "always on … no runtime override"; there is an override.
  - `BOLT_SORT_USE_GRAPH` — **actively read** in `src/exec/gpu_sort.rs:1730` (`BOLT_SORT_USE_GRAPH_ENV`, gate at `:1942`). Doc says "selected by the sort orchestrator on size, not by env var"; it is gated by the env var.
  These four are not only mis-listed as inert — they are **undocumented working knobs** missing from the main matrix. Fix: move them into the documented section with their real semantics, or, if they are meant to be private, say "internal/unstable, may be removed" rather than "has no effect."
  - (`CRATON_DISTINCT_HOST_MAX_ROWS` and `CRATON_PLAN_CACHE_SIZE` could not be found as env reads and do appear to be compile-time constants — those two entries look correct.)

### docs/SQL_REFERENCE.md — purpose: authoritative SQL surface. Quality: excellent, the best doc in the set.
- Tracks 0.7.0 (`:5`). Execution-tier tagging (GPU / host-side / GPU-lowering-pending) is precise and matches CHANGELOG and code behavior. Joins, set ops, CTEs, subqueries, window functions, COUNT(DISTINCT), GPU LIKE/UPPER/LOWER/LENGTH, Decimal128/Date32/Timestamp tiers all consistent with v0.7.
- **Minor:** `IS NULL` / `IS NOT NULL` is listed only inside the "supported now" prose (`:413`) but has **no dedicated section** in Operators, unlike every other operator. Given MIGRATION_GUIDE devotes a whole subsection to it (and `UnaryOp::IsNull`/`IsNotNull` exist in `logical_plan.rs:391-393`), SQL_REFERENCE under-documents a supported feature. Add an "IS [NOT] NULL" subsection.
- Otherwise no contradictions with code found.

### docs/BENCHMARKS.md — purpose: measured numbers + methodology. Quality: high, transparent.
- §1 numbers (RTX 2060, CUDA 12.6, driver 591.86) are internally consistent and match the README. Methodology, verification tolerance (1e-9), criterion window all stated. The historical appendix honestly preserves pre-fast-path baselines and explicitly labels them non-canonical (`:226-232`).
- Numbers are plausible and reproducible-in-principle (harness paths cited). **Not** fabricated-looking. One caveat: §1 q-numbers and the "Tier-2.1 (canonical numbers)" appendix table agree (q1 51.4 / q3 219 / q5 237), good.
- **Note for cross-doc:** these are the numbers GROUPBY_PERF.md *should* be citing but does not (see below).

### docs/COMPETITIVE_BENCHMARKING.md — purpose: fair-comparison methodology. Quality: high methodology, **stale "category" framing.**
- The methodology (cold/warm, geomean, correctness verification, system-control table, per-competitor invocations, pitfalls) is excellent and genuinely useful.
- **WRONG/stale:** §1 "What category is Craton Bolt in?" (`:30-38`) states Craton Bolt is **"Single-batch in-memory (no streaming, no larger-than-VRAM tables, no spill)"** and **"JIT-compiled per-query (no plan cache, no kernel cache beyond PTX)"**. Both are outdated:
  - The engine supports **multi-batch** tables (README, USER_GUIDE, ARCHITECTURE all say so; `register_batch`/`register_table_stream` exist). "Single-batch" directly contradicts the rest of the corpus.
  - There **is** a plan cache (`plan_cache_stats` in API_SURFACE `:135`) and a `KernelSpec`-keyed module cache (v0.7). "No plan cache, no kernel cache beyond PTX" is stale.
  - The TL;DR also says "single-batch" (`:13` says "multi-batch" — so the doc even contradicts *itself*: `:13` "multi-batch in-memory" vs `:32` "Single-batch in-memory").
- The SQL-surface bullet (`:36-38`) lists only "INNER JOIN" and omits the OUTER/CROSS/USING/NATURAL joins, set ops, CTEs, subqueries, windows that §3's tables later acknowledge — internally inconsistent.

### docs/CUDARC_ADOPTION.md — purpose: cudarc migration plan. Quality: detailed, **but internal scratch.**
- Explicitly self-labels "internal design note / in-progress migration spike … engineering working notes, not a stable feature description" (`:3`). Contains stage-by-stage TODO checklists, day estimates, "Open questions," `~~struck-through~~ DONE` markers, commit SHAs.
- **Recommendation: do not ship in public `docs/`** (or move to a `docs/internal/` or design-notes area, or the wiki). It is accurate to the code (Stage 1/1.5 landed, Stages 2–4 not) but is developer-facing planning material, not user/contributor documentation.

### docs/CUDA_OXIDE_SWEEP.md — purpose: refactor tracker. Quality: **clearly internal scratch.**
- Self-labels "internal refactor tracker / working notes … an engineering checklist" (`:3`). A per-executor ✅/⏳-todo status table, "Recommended order to finish the sweep," engineering-hour estimates.
- **Recommendation: do not ship in public `docs/`.** This is a sprint board in Markdown. No public-repo value; reads as half-finished internal work and signals "not production ready" in the wrong way (incomplete refactor) on the headline safety feature.

### docs/DEVELOPMENT.md — purpose: build/test/bench workflow. Quality: high.
- Commands, CI matrix (`{ubuntu,windows} × {stable,1.74}` under cuda-stub), the three test flavours, "adding a new dtype/kernel/feature" guides. Cross-checked against `Cargo.toml` features and bench names — accurate.
- **Minor:** "The CI lint script (forthcoming) will reject files without this [SPDX] header" (`:256`) and "CI lint script (forthcoming)" — flags an unbuilt thing; fine but it's a "forthcoming" that's been forthcoming since 0.3.

### docs/FAQ.md — purpose: design-rationale Q&A. Quality: high.
- Q1 (no NVRTC), Q2 (no macOS), Q11 (sm_70 floor with the instruction rationale) all match code/FAQ/JIT_PIPELINE. Q9 explains the Craton Software Company attribution. Contact email consistent. No drift.
- **Minor:** Q3 says async memcpy "shipped … 0.3.0" for the FFI and rolled out in 0.6/0.7 — consistent with CHANGELOG; fine.

### docs/GROUPBY_PERF.md — purpose: GROUP BY design analysis. Quality: good analysis, **self-contradicting status + stale numbers.**
- Header says **"Status: Implemented (Tiers 1–2 landed)"** and "For current performance numbers see BENCHMARKS.md §1" (`:3-6`).
- **But the body is written in the pre-optimization present tense** and its headline table (`:13-18`) shows **q1 = 269 ms, q2 = 406 ms, q3 = 704 ms, q5 = 770 ms** with the verdict **"Craton Bolt loses on every query"** (`:20`) and "the cost of the current kernel" (`:67`). Those numbers are the **pre-fast-path baseline** (cf. BENCHMARKS.md appendix q1≈282 ms), **not** the canonical §1 numbers (q1 51.4 / q3 219 / q5 237) the doc claims to be quoting "in BENCHMARKS.md."
- Net effect: a reader is told the optimizations are implemented, pointed at BENCHMARKS.md for current numbers, then shown a different (worse) table presented as current with a "we lose everywhere" conclusion. **This is the most confusing doc in the set.** Fix: clearly mark the table as the *pre-optimization diagnosis* and add the post-Tier-2.1 result row, or replace the verdict text.

### docs/INSTALL.md — purpose: prerequisites + build configs + troubleshooting. Quality: excellent.
- CUDA 12.x guidance, cudarc `cuda-12060` pin, the v13.2 Windows workaround, the full Cargo feature matrix (cuda-stub/cudarc/rust-cuda/pool-sharded/pool-watcher) — all match `Cargo.toml:94-135` and `build.rs`. No issues.

### docs/JIT_PIPELINE.md — purpose: SQL→PTX deep dive. Quality: high technically, **stale surface claims.**
- The PTX walkthrough, cost table, sm_70 instruction/CC table, cache description, and codegen conventions are accurate and cross-check against FAQ Q11 and `jit_compiler.rs` behavior.
- **Stale:** §1 "SQL → LogicalPlan" says **"CTEs, subqueries, window functions, and non-equi join predicates are still rejected at parse time"** (`:50`) and "a single JOIN per `SELECT`" / "at most one joined table" (`:50-51`). All four are wrong for v0.7: CTEs, uncorrelated subqueries, window functions, and **multiple joins per SELECT** are supported (SQL_REFERENCE, CHANGELOG). Small-cardinality non-equi INNER is also handled via host nested-loop.
- **Stale:** §8 "What's not codegened" → "Window functions. Not yet." (`:292`) and "CASE / NULLIF / CAST … the AST doesn't model these" (`:293`) — CASE/CAST/COALESCE/NULLIF are modeled and (numeric/Bool) lower to GPU as of 0.7. These two bullets predate 0.5–0.7.
- The async-transfer status block (`:241-257`) is, by contrast, correctly updated to 0.7.

### docs/USER_GUIDE.md — purpose: 10-minute tutorial. Quality: excellent.
- `cargo add` snippet, end-to-end example, error-handling table (matches `BoltError` variants in code), performance-tuning knobs, multi-GPU. The Supported-SQL summary (`:147-188`) is consistent with SQL_REFERENCE. Examples match `examples/quickstart.rs` and `examples/groupby.rs` (verified — the quickstart in the guide mirrors `examples/quickstart.rs` modulo the error-fallback wrapper).
- **Minor:** the env-var tuning table (`:316-322`) is a correct subset of ENV_VARS.md. Good.

### docs/MIGRATION_GUIDE.md — purpose: upgrade deltas. Quality: high, **one wrong code sample.**
- 0.3→0.5→0.6→0.7 deltas are accurate and match CHANGELOG. The `BoltError` `#[non_exhaustive]` and `DataFrame::collect` re-purposing sections are correct.
- **WRONG code:** the `register_table_stream` "After (0.6)" example shows it as a **builder** method:
  ```rust
  let engine = Engine::builder()
      .register_table_stream("orders", batches.into_iter())
      .build()?;
  ```
  (`MIGRATION_GUIDE.md:326-330`). In the actual code `register_table_stream` is an **`Engine`** method (`src/exec/engine.rs:1255`, signature `pub fn register_table_stream<I>(&mut self, ...)`), not an `EngineBuilder` method, and `EngineBuilder` has no such method (API_SURFACE `:94` lists builder methods: `new/device/memory_budget/persistent_cache/enable_tracing/build` only). This sample will not compile. The doc's own prose elsewhere (and USER_GUIDE `:134-141`) correctly treats it as an `Engine` method — so this is an isolated wrong snippet. Fix to `engine.register_table_stream("orders", batches.into_iter())?;` after `build()`.

### docs/PATH_TO_1.0.md — purpose: strategic 1.0 plan. Quality: high, honestly scoped.
- Explicitly "strategic, not contractual" with a baseline note that the "today" column reflects the original 0.3.0 starting point (`:8-16,68-71`). This framing correctly inoculates the otherwise-stale "today" table.
- **Watch item:** §5 "What 1.0 is NOT" says **"Not a CPU fallback engine … it errors — it does not silently fall back to a host pipeline"** (`:348-349`). This contradicts the *actual current* engine, which routinely falls back to host-side execution (sort, joins, set ops, window functions, string fns, DISTINCT). It is framed as a 1.0 *goal*, but a reader skimming may take it as a current statement. Consider a note that host-side execution is present today and the "errors instead of falling back" stance applies only to *unsupported-on-GPU-lowering* expression cases, not to the many features that legitimately run host-side.

---

## 2. Cross-document consistency

### Consistency matrix (key facts)

| Fact | Cargo.toml | README | USER_GUIDE | SECURITY | API_SURFACE | Verdict |
|---|---|---|---|---|---|---|
| Version | 0.7.0 (`:3`) | 0.7.0 (`:16`) | "0.7" (`:21`) | 0.7.x (`:13`) | 0.7.0 (`:11`) | ✅ consistent |
| License | Apache-2.0 (`:6`) | Apache-2.0 (`:235`) | — | — | — | ✅ |
| Repo URL | `github.com/craton-co/craton-bolt` (`:8`) | same (`:49`) | — | implied | — | ✅ |
| Contact email | `opensource@cratonsoftware.com` (`:7`) | SECURITY ref | `security@cratonsoftware.com` (FAQ) | `security@cratonsoftware.com` | — | ✅ |
| MSRV | rust-version 1.74 (`:14`) | 1.74 (`:3,33`) | — | — | — | ✅ |
| GPU floor | sm_70 / CC≥7.0 | CC≥7.0 (`:35`) | CC≥7.0 (`:31`) | — | — | ✅ |
| CUDA | 12.x (cudarc cuda-12060) | ≥12 (`:34`) | ≥12.0 (`:32`) | — | — | ✅ (INSTALL adds the v13/Windows caveat) |

Naming is **internally consistent**: org slug `craton-co`, company `Craton Software Company`, domain `cratonsoftware.com`. The `craton.com.ar` domain from the review brief is **absent from the repo** (only appears in a prior `reviews/oss_readiness.md`). The only stylistic mismatch is org-slug `craton-co` vs domain `cratonsoftware.com`, which is normal and not an error.

### Contradictions between docs (substantive)

1. **Multi-batch vs single-batch.** COMPETITIVE_BENCHMARKING `:32` ("Single-batch in-memory") contradicts README, USER_GUIDE, ARCHITECTURE, and even its own `:13` ("multi-batch in-memory"). **The engine is multi-batch.**
2. **CTEs / subqueries / windows support.**
   - SQL_REFERENCE, README, USER_GUIDE, ROADMAP, CHANGELOG, MIGRATION_GUIDE: **supported in 0.7** ✅.
   - ARCHITECTURE `:273`: "**No CTEs, subqueries, or window functions. The parser rejects them outright.**" ❌ stale.
   - JIT_PIPELINE `:50,292`: "CTEs, subqueries, window functions … still rejected at parse time"; "Window functions. Not yet." ❌ stale.
   - PATH_TO_1.0 lists window functions / recursive CTEs as *post-1.0* (`:346-347`) — fine in context, but a casual reader gets mixed signals across the three.
3. **Joins per SELECT.** SQL_REFERENCE/README/ROADMAP: multiple joins per SELECT supported. JIT_PIPELINE `:50-51`: "a single JOIN per SELECT" / "at most one joined table." ❌ stale.
4. **Plan/kernel cache.** API_SURFACE documents `plan_cache_stats` and the v0.7 KernelSpec module cache; COMPETITIVE_BENCHMARKING `:34` says "no plan cache, no kernel cache beyond PTX." ❌ stale.
5. **`register_table_stream` location.** MIGRATION_GUIDE `:326-330` (builder method) vs USER_GUIDE `:134-141` + code (Engine method). ❌ migration guide wrong.
6. **GROUPBY_PERF numbers vs BENCHMARKS §1.** Same workload, different numbers, with GROUPBY_PERF claiming to quote BENCHMARKS §1. ❌ (see per-doc).
7. **"errors, does not fall back to host" (PATH_TO_1.0 `:348`)** vs the pervasive host-side fallbacks documented everywhere else. Framed as a 1.0 goal but easy to misread.

---

## 3. Gaps and redundancy

### Gaps for an OSS release
- **No single "Limitations / Not production ready" disclaimer doc.** The "production use is not recommended" caveat exists in README `:16` and the pre-1.0/API-unstable messaging is in ROADMAP/SECURITY/API_SURFACE, but there is no consolidated **LIMITATIONS** page a prospective user can read in 30 seconds. The information is scattered across ROADMAP "Known limitations," ARCHITECTURE "What's deliberately not in," SQL_REFERENCE "What's NOT supported," and PATH_TO_1.0 §5. Consider a short top-level pointer.
- **GPU/CUDA requirements are well-covered** (INSTALL.md is strong: CC≥7.0, CUDA 12.x, the v13.2 Windows caveat, supported CUDA versions). No gap here — this is a strength.
- **CODEOWNERS references a non-existent GitHub team** — a concrete release blocker if branch protection requires CODEOWNER review.
- **`register_table_stream_lazy`** (real code) is undocumented in API_SURFACE.
- **Four real env vars** (`BOLT_PREFIX_SCAN_ALGO`, `BOLT_HASH_ALGO`, `BOLT_HASH_PROBE_TILED`, `BOLT_SORT_USE_GRAPH`) are undocumented and actively mis-described as inert.
- **`deny.toml` referenced by RELEASING/CONTRIBUTING** — confirm it exists (not verified here).

### Redundant / should be removed or merged
- **`CUDARC_ADOPTION.md` and `CUDA_OXIDE_SWEEP.md` should not ship in public `docs/`.** Both self-identify as internal working notes / trackers with TODO checklists, hour estimates, and commit SHAs. Move to `docs/internal/`, a design-notes branch, the wiki, or delete. Keeping them signals an unfinished internal refactor on the project's headline safety feature.
- **`JIT_PIPELINE.md` references `docs/rust_cuda/` (internal design notes, not published)** at `:320` — a dangling reference to material that "is not published." Remove the pointer or publish the target.
- **GROUPBY_PERF.md** overlaps heavily with BENCHMARKS.md's groupby section; at minimum reconcile the numbers so the two don't disagree. The design rationale is worth keeping; the stale numeric table is not.

### Documents reading as internal/AI-generated scratch
- **`CUDA_OXIDE_SWEEP.md`** — strongest example; a per-file refactor status board.
- **`CUDARC_ADOPTION.md`** — design spike with struck-through DONE items and "Open questions."
- **`GROUPBY_PERF.md`** — mixed: solid analysis, but the "current/we lose everywhere" framing alongside a "Status: Implemented" header reads like notes that were never reconciled after the optimizations landed.
- The Cargo.toml comments are unusually essay-like (e.g. the `[workspace]` shim explanation `:17-36`, the dashmap rationale `:51-55`) — harmless, even helpful, but worth a skim for any over-sharing before publish.

---

## Priority fix list (highest first)

1. **ENV_VARS.md `:254-272`** — remove the four real vars from "Not present"; document them (or mark internal/unstable). They currently tell users a working knob "has no effect."
2. **ARCHITECTURE.md `:273`** — delete/replace "No CTEs, subqueries, or window functions. The parser rejects them outright." (flatly false in 0.7).
3. **MIGRATION_GUIDE.md `:326-330`** — fix the `register_table_stream` sample (it's an `Engine` method, won't compile as shown on the builder).
4. **JIT_PIPELINE.md `:50,51,292,293`** — update the "rejected at parse time" / "single JOIN" / "Window functions. Not yet" / "CASE/CAST not modeled" claims to the 0.7 surface.
5. **COMPETITIVE_BENCHMARKING.md `:32-38`** — fix "Single-batch in-memory" / "no plan cache" / INNER-only SQL list; reconcile with `:13`.
6. **GROUPBY_PERF.md `:3-20,67`** — reconcile the headline table with BENCHMARKS §1 (label it pre-optimization, or update numbers); fix the "loses on every query" verdict.
7. **Remove `CUDARC_ADOPTION.md` + `CUDA_OXIDE_SWEEP.md`** from the public `docs/` tree (relocate or delete).
8. **CODEOWNERS / MAINTAINERS** — provision the team or point at a real handle before enabling required-review branch protection.
9. **README `:186-202`** — add `JIT_PIPELINE.md` to the project-layout tree.
10. **SQL_REFERENCE** — add an `IS [NOT] NULL` subsection; **API_SURFACE** — document `register_table_stream_lazy`.
