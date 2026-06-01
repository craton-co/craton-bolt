# Craton Bolt — Documentation Review

Scope: all root docs + `docs/*.md`, spot-checked against `src/`, `Cargo.toml`,
`build.rs`, `lib.rs`. Crate version under review: **0.7.0**. Reviewer date:
2026-05-30.

Overall: the documentation set is unusually thorough and internally
cross-referenced for a pre-1.0 project. The main problems are (a) a handful of
**dead cross-references** to a doc that does not exist, (b) **stale string /
LIKE execution-tier claims** that contradict each other and the code, (c) an
**API surface doc that is missing two whole public modules**, (d) an **ENV_VARS
matrix that mislabels two real env vars as "not present"**, (e) a **CI workflow
path that does not resolve**, and (f) an **internal scratch file
(`bolt_continue_prompt.md`) that should not ship**.

---

## 1. Per-document review

### Root

**README.md** — Accurate and current overall (v0.7.0, SQL surface summary at
line 16 matches code). Issues:
- Project-layout tree (lines 198–213) lists `docs/` contents but **omits
  `MIGRATION_GUIDE.md`'s sibling files inconsistently** — minor; it does list
  most. More importantly it does not mention `examples/` (which exists:
  `examples/quickstart.rs`, `examples/groupby.rs`) even though `Cargo.toml`
  declares both as `[[example]]`. Add an `examples/` line.
- Line 16 says `LIKE` "lower to GPU" is implied by "numeric/Bool results lower
  to GPU; …" — fine, but see the LIKE contradiction under §2.
- CI badge points at `actions/workflows/ci.yml` — but the workflow file is at
  `.github/.wf/ci.yml`, not `.github/workflows/` (see §3, infra finding).

**ROADMAP.md** — Strong, current as of 0.7.0. The "Known limitations (not
bugs) — as of 0.7.0" section (lines 136–153) is accurate against code
(radix sort env-gated, `persistent_cache` builder knob not wired, eager
streaming). No removals needed.

**CHANGELOG.md** — Current; `[Unreleased]` section (lines 9–32) references
`reviews/PERF_BACKLOG.md` (line 31). That file exists in `reviews/`, but
`reviews/` is `exclude`d from the package (`Cargo.toml:15`), so the published
crate's CHANGELOG points at a path not in the tarball. Cosmetic, but a reader on
docs.rs/crates.io cannot follow it. Consider inlining or dropping the pointer.

**CONTRIBUTING.md** — Accurate. References `DCO` file (exists) and `CODEOWNERS`
(exists). Note the DCO `Signed-off-by` convention here is the project's own and
is unrelated to the harness commit-trailer rules.

**RELEASING.md** — Good maintainer checklist. References `deny.toml` (exists)
and the v13.2 Windows workaround (consistent with INSTALL.md). §8 "Windows CUDA
v13" cross-refers correctly.

**SECURITY.md** — Accurate; supported-version table (0.7.x) matches current
version. Mentions skipped 0.2.0/0.4.0 correctly.

**CODE_OF_CONDUCT.md** — Contributor Covenant 2.1 adaptation, but it only
includes Pledge/Standards/Responsibilities/Scope/Enforcement; it **omits the
Enforcement Guidelines (the 4-tier consequence ladder)** that ship with
Covenant 2.1. Not wrong, just a partial adaptation. Optional to complete.

**MAINTAINERS.md** — Honest interim doc; references `CODEOWNERS` (exists).
Fine.

**NOTICE** — Accurate against `Cargo.toml` dependency set (arrow, sqlparser,
dashmap, cudarc, criterion, polars, duckdb, proptest, insta, cuda_builder all
present and correctly licensed). Copyright year 2026 matches.

**bolt_continue_prompt.md** — **Does NOT belong in the repo.** It is an internal
agent handoff/scratch file ("You are taking over on craton-bolt…", HEAD SHA,
GPU-validation TODO list). It is not referenced by any other doc, contains
forward-looking "this is unvalidated device code" warnings that **contradict**
the published docs (see §2), and would confuse external readers. **Critically,
`Cargo.toml:15`'s `exclude` list names `continue_prompt.md`, NOT
`bolt_continue_prompt.md`** — so the actual filename is NOT excluded and **will
be packaged into the published crate**. Remove the file (or at minimum fix the
exclude entry). Recommend removal.

### docs/

**API_SURFACE.md** — Mostly precise (the `BoltError` 10-variant list at line 83
matches `src/error.rs` exactly; `Engine`/`EngineBuilder`/`QueryHandle` method
lists verified). **But it is incomplete: `src/lib.rs` re-exports two public
surfaces this doc never mentions:**
- `pub use exec::streaming::{BatchProducer, BatchStream, MorselPlan,
  PinnedBudget, TableSource}` (`lib.rs:47`) — an entire streaming module
  (`src/exec/streaming.rs` exists).
- `pub use metrics::{metrics, Counter, MetricsSnapshot, Phase}` and
  `metrics_snapshot` (`lib.rs:119,125`) — the `metrics` module
  (`src/metrics.rs` exists) is public and undocumented here.
Also missing: `Engine::register_table_stream_lazy` (`engine.rs:1394`) is a
public method absent from the `Engine` method list (line 93). The doc's own
header says "re-run the enumeration whenever the public surface changes" — it is
overdue. ADD: streaming + metrics tiers; the lazy stream method.

**ARCHITECTURE.md** — Largely current (v0.7.0 header, layer cake matches module
tree). **One materially stale claim:** line 233 — *"UPPER / LOWER / LENGTH /
SUBSTRING all run as pure-host dictionary transformations … None of these go
through the GPU because variable-width device writes remain unsupported."* This
contradicts the code: `PhysicalPlan::StringProject` (UPPER/LOWER) and
`StringLength` (LENGTH) are real GPU executors (`physical_plan.rs:977,1000`;
dispatched in `engine.rs:2565,2570`) and SQL_REFERENCE/ROADMAP/CHANGELOG all
claim these lower to GPU as of 0.7. FIX line 233 to: UPPER/LOWER/LENGTH lower to
GPU (v0.7), SUBSTRING/TRIM/CONCAT remain host-side.

**JIT_PIPELINE.md** — Strong deep-dive, accurate on PTX/codegen/cache. Minor
staleness: §1 (lines 53, 58) still describes `LogicalPlan` as "a small enum
(`Scan`, `Filter`, `Project`, `Aggregate`)" and lists only the 5 base scalar
op classes / `COUNT/SUM/MIN/MAX/AVG` — predates joins, set ops, window, CTE,
subqueries, string ops. The later sections (§What's not codegened) are current,
so it's only the intro that lags. Tighten §1's "small enum" framing.

**SQL_REFERENCE.md** — The most important doc and largely excellent and current.
Two internal inconsistencies to fix (see §2): the **LIKE execution tier** and
the **LENGTH character-vs-byte** claim. The "Expression examples" block
(lines 322–323) labels `||` and `LIKE 'A%'` as host-side, while the prose
(line 126) says LIKE lowers to GPU — at minimum the `'A%'` prefix case is
exactly a GPU-lowered `StringLikeFilter` shape, so the inline comment is
misleading. Otherwise verified accurate (`COUNT(DISTINCT)` sole-item rule,
join gates, set-op semantics, GROUP BY key shapes, `LIMIT` literal rule).

**LIMITATIONS.md** — Good consolidated honest-read. **Conflicts with
SQL_REFERENCE on string semantics:** lines 96–99 say the GPU string functions
are "byte/ASCII-oriented", "`LENGTH` is byte-length", "case conversion is
ASCII-range" — but SQL_REFERENCE.md lines 483, 494 say `CHAR_LENGTH`/`LENGTH`
are **character-based** (codepoints) and case folding uses **Unicode default
case mapping**. These cannot both be true; one is stale. Needs reconciliation
against the actual `string_ops` implementation (recommend verifying
`src/exec/string_ops*.rs` and aligning both docs).

**ENV_VARS.md** — Detailed and mostly verifiable against grep of
`std::env::var`. **Two concrete errors** (see §3): the "Not present in this
build" section (lines 322–330) lists `CRATON_DISTINCT_HOST_MAX_ROWS` and
`CRATON_PLAN_CACHE_SIZE` as NOT honoured — but both ARE read at runtime
(`src/exec/distinct.rs:90,111`; `src/plan/sql_frontend.rs:818,842`). These are
real, documented-as-absent env vars: move them into the live matrix with their
source lines. Everything else (pool, PTX cache, GPU join/sort, scan/hash
selectors) checks out.

**INSTALL.md** — Accurate against `build.rs` (CUDA_PATH discovery, v12.x
preference, cuda-stub) and `Cargo.toml` feature matrix. **Dead link:** line 79
references `docs/CUDARC_ADOPTION.md`, which does not exist (see §2).

**DEVELOPMENT.md** — Accurate workflow. References CI at
`.github/workflows/ci.yml` (line 62) — see §3 path discrepancy. The "SPDX
header… CI lint script (forthcoming)" (line 256) is honest about not existing
yet. Fine otherwise.

**FAQ.md** — Accurate. Q3 says async memcpy "landed in 0.3.0"; ROADMAP/CHANGELOG
frame the async *pilot* as 0.6 and rollout as 0.7. The FFI binding may indeed
predate the executor wiring, so not strictly wrong, but the "0.3.0" date reads
as inconsistent with the 0.6/0.7 framing elsewhere. Minor.

**USER_GUIDE.md** — Good 10-minute tutorial, code compiles against the public
API. **Internal contradiction:** line 163 lists `LIKE (GPU over Utf8)` but
line 184 lists `LIKE (host-side)` in the same bullet list. Pick one (GPU, with
host fallback — matches SQL_REFERENCE prose). Also references
`docs/CUDARC_ADOPTION.md` (line 344, dead link).

**BENCHMARKS.md** — Thorough, with honest "the honest read". Numbers are
internally consistent with README's perf tables. **Dead link:**
`docs/CUDARC_ADOPTION.md` referenced at lines 391, 404.

**COMPETITIVE_BENCHMARKING.md** — Excellent methodology doc. **Stale claim:**
line 199–200 says `UPPER`/`LOWER`/`LENGTH`/`CONCAT`/`SUBSTRING` "run host-side"
— same staleness as ARCHITECTURE.md:233 (UPPER/LOWER/LENGTH are GPU as of 0.7).
Also lines 83, 208 defer TPC-DS/streaming to "0.4+" / "until 0.4+" — anachronistic
now that the project is at 0.7 and 0.4 was skipped; reword to "post-1.0" or a
concrete later milestone.

**GROUPBY_PERF.md** — Well-managed: explicitly marks itself "Status:
Implemented" and reproduces canonical numbers from BENCHMARKS.md to avoid drift.
Current. No action.

**MIGRATION_GUIDE.md** — Accurate version-jump deltas (0.3→0.5→0.6→0.7),
correctly notes 0.4 skipped and the 0.7 `DESC` radix fix / LIKE-type-check
behavior changes. Current.

**PATH_TO_1.0.md** — Explicitly framed as "strategic, not contractual" with a
baseline note that milestone version numbers are illustrative/historical (0.4
baseline). Given that framing, its 0.4/0.8/0.9/0.10 milestone numbering is
acceptable. Current.

---

## 2. Cross-document consistency

**Contradictions:**

1. **LIKE execution tier.** SQL_REFERENCE.md prose (line 126) and ROADMAP
   (line 38) and CHANGELOG say LIKE *lowers to GPU* in 0.7 (and the
   `StringLikeFilter` node is real: `physical_plan.rs:1030`). But:
   - SQL_REFERENCE.md example comment (line 323): `LIKE 'A%'` → "host-side
     filter".
   - USER_GUIDE.md line 184: "`LIKE` (host-side)" (while line 163 says GPU).
   - `bolt_continue_prompt.md` calls the GPU LIKE matcher "EXPLICITLY
     UNVALIDATED DEVICE CODE."
   Reality: there is a host `Expr::Like` fallback (`physical_plan.rs:1408`
   "LIKE requires host fallback") AND a GPU `StringLikeFilter` path; the docs
   should consistently say "lowers to GPU for the supported shapes, host
   fallback otherwise." Right now three docs disagree.

2. **String length / case semantics.** LIMITATIONS.md (byte-length, ASCII case)
   vs SQL_REFERENCE.md (character-based, Unicode case mapping). Direct conflict.

3. **String funcs on GPU vs host.** ARCHITECTURE.md:233 and
   COMPETITIVE_BENCHMARKING.md:199 say UPPER/LOWER/LENGTH run host-side; ROADMAP,
   SQL_REFERENCE, USER_GUIDE, CHANGELOG say GPU (v0.7). The latter group matches
   the code (`StringProject`/`StringLength` executors).

**Dead links / broken cross-references:**

- **`docs/CUDARC_ADOPTION.md` does not exist** but is referenced from
  INSTALL.md:79, USER_GUIDE.md:344, BENCHMARKS.md:391 & 404, and the `cudarc`
  feature comment in `Cargo.toml:79`. Either add the doc or remove the four
  references.

**Duplication (acceptable but worth noting):**
- The h2o.ai groupby results table is duplicated across README, BENCHMARKS.md,
  and GROUPBY_PERF.md. GROUPBY_PERF explicitly reproduces to "not drift" and
  they currently agree, so this is managed duplication, not a bug.
- Prerequisites/feature-matrix tables are repeated in README, INSTALL,
  DEVELOPMENT, USER_GUIDE, LIMITATIONS — all consistent. Fine for a docs set.

**Version numbers:** consistent at 0.7.0 across README, Cargo.toml, ROADMAP,
SQL_REFERENCE, LIMITATIONS, API_SURFACE, SECURITY, BENCHMARKS. Good.

---

## 3. Accuracy vs code (specific claims)

1. **ENV_VARS.md:323–330 — WRONG.** Claims `CRATON_DISTINCT_HOST_MAX_ROWS` and
   `CRATON_PLAN_CACHE_SIZE` are "NOT honoured by the current codebase." Evidence:
   `src/exec/distinct.rs:90` `const DISTINCT_HOST_MAX_ROWS_ENV: &str =
   "CRATON_DISTINCT_HOST_MAX_ROWS"` and read at `:111`; `src/plan/sql_frontend.rs:818`
   `const PLAN_CACHE_SIZE_ENV: &str = "CRATON_PLAN_CACHE_SIZE"` read at `:842`.
   Both are live env vars and should be documented in the matrix.

2. **ARCHITECTURE.md:233 — STALE.** "UPPER / LOWER / LENGTH / SUBSTRING all run
   as pure-host… None go through the GPU." Evidence: `StringProject` /
   `StringLength` physical-plan variants (`src/plan/physical_plan.rs:1000,977`)
   are GPU executors dispatched in `src/exec/engine.rs:2565,2570`.

3. **COMPETITIVE_BENCHMARKING.md:199 — STALE.** Same UPPER/LOWER/LENGTH
   host-side claim.

4. **API_SURFACE.md — INCOMPLETE.** Misses `exec::streaming::*`
   (`src/lib.rs:47`, `src/exec/streaming.rs`) and `metrics::*`
   (`src/lib.rs:119,125`, `src/metrics.rs`), plus
   `Engine::register_table_stream_lazy` (`src/exec/engine.rs:1394`).

5. **CI workflow path — INFRA / DOC MISMATCH.** README badge,
   LIMITATIONS.md:67, DEVELOPMENT.md:62 all reference
   `.github/workflows/ci.yml`. The actual file is `.github/.wf/ci.yml` (the
   `workflows/` directory does not exist). GitHub only runs workflows under
   `.github/workflows/`, so as committed **CI does not run**. Either the path is
   deliberately parked (`.wf`) and the docs overstate CI, or it's a typo that
   disables CI. Flag for the maintainer; whichever it is, the docs and the path
   disagree.

6. **`Cargo.toml:15` exclude typo — PACKAGING.** Excludes `continue_prompt.md`
   but the file present is `bolt_continue_prompt.md`; the scratch file will ship
   in the published crate.

**Verified-correct claims (sample):** `BoltError` 10 variants (API_SURFACE.md:83
≡ `src/error.rs:38–118`); stub message `BoltError::Other("cuda-stub mode: no GPU
support compiled in")` (DEVELOPMENT/INSTALL ≡ `src/cuda/cuda_sys.rs:9`); all
runtime env vars in ENV_VARS matrix found at their cited source lines; feature
matrix (INSTALL/USER_GUIDE ≡ `Cargo.toml:94–135`); MSRV 1.74 / edition 2021
(`Cargo.toml:3,14`); examples exist (`examples/quickstart.rs`,
`examples/groupby.rs`).

---

## 4. Gaps (missing docs/sections for OSS release)

- **No top-level getting-started entry obvious to a first-time visitor beyond
  README.** README + USER_GUIDE cover this well; arguably sufficient. Low
  priority.
- **No versioning / release-cadence policy for users** (as opposed to the
  maintainer-facing RELEASING.md). SECURITY.md states "latest minor only";
  a short "Versioning & stability" section (semver pre-1.0 caveat, what a minor
  bump may break) would help. README line 236 covers this in one sentence;
  could be expanded.
- **No CHANGELOG entry for the `[Unreleased]` perf/dedup work tying back to a
  user-visible effect** — internal-only, acceptable.
- **Troubleshooting** is well covered (INSTALL §Troubleshooting, USER_GUIDE
  §When something fails, DEVELOPMENT §Recovering from common errors). No gap.
- **`CUDARC_ADOPTION.md` is referenced as if it exists** — it is effectively a
  "missing doc" that 4 files promise. Either write it or delete the references.
- **`metrics` module has no narrative doc** anywhere (not in USER_GUIDE
  observability section, not in API_SURFACE). USER_GUIDE covers `log` +
  `tracing` + pool-stats observer but never the `metrics`/`Counter`/`Phase`
  surface that lib.rs exports. Add coverage.

---

## 5. Redundancy / files to remove or merge

- **REMOVE `bolt_continue_prompt.md`.** Internal agent scratch/handoff file;
  not referenced by any doc; contradicts published docs; and (per §3.6) is
  currently mis-excluded so it would ship in the crate. This is the clearest
  "should not be in the repo" item.
- **Consider trimming `reviews/` from the repo or keeping it out of any doc
  links.** `CHANGELOG.md:31` links `reviews/PERF_BACKLOG.md`, but `reviews/` is
  excluded from the package — a published-artifact dangling pointer. The
  `reviews/` tree (done_*.md, ORCHESTRATION.md, SYNTHESIS.md, etc.) is clearly
  internal working material; fine to keep in-repo but should not be linked from
  shipping docs.
- **No genuinely redundant docs to merge.** The docs/ set is well-factored:
  BENCHMARKS (numbers) vs COMPETITIVE_BENCHMARKING (methodology) vs GROUPBY_PERF
  (design) have clear, non-overlapping roles, and GROUPBY_PERF explicitly
  defers canonical numbers to BENCHMARKS. ROADMAP vs PATH_TO_1.0 vs LIMITATIONS
  overlap thematically but each has a distinct altitude (index / depth /
  honest-read) and they cross-link cleanly. Keep all.

---

## Priority fix list

1. Remove `bolt_continue_prompt.md` (and/or fix `Cargo.toml` exclude typo).
2. Resolve the CI path discrepancy (`.github/.wf/` vs `.github/workflows/`) —
   functional, not just docs.
3. Fix ENV_VARS.md: move the two real env vars out of "Not present."
4. Reconcile the LIKE tier across SQL_REFERENCE / USER_GUIDE; and the string
   GPU-vs-host claims in ARCHITECTURE:233 + COMPETITIVE:199.
5. Reconcile LENGTH/case semantics (byte/ASCII vs char/Unicode) between
   LIMITATIONS and SQL_REFERENCE against actual code.
6. Add `metrics` + `streaming` to API_SURFACE.md (and `register_table_stream_lazy`).
7. Create or delete-references-to `docs/CUDARC_ADOPTION.md`.
