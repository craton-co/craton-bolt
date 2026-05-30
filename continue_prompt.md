# Handoff: GPU validation of the engine-improvement batch on `dev`

You are taking over on **craton-bolt** (a JIT-compiled GPU SQL engine). A previous agent,
working on a machine **without a GPU**, implemented and merged a large batch of **engine
capability + SQL-surface improvements** onto the **`dev`** branch (which equals **`main`**).

**Verified green** at commit `4afd542` (2026-05-29):
`CARGO_INCREMENTAL=0 cargo build --no-default-features --features cuda-stub --lib` → exit 0,
working tree clean. **But almost none of the new GPU-execution paths have run on real
hardware.** Your machine has a GPU. **Your job:** run the GPU-gated tests, fix what fails, and
sanitize the new device code — especially the explicitly-unvalidated GPU `LIKE` matcher.

> NOTE: This supersedes the earlier security/perf (`V-#`/`P#`/`D#`) handoff. That work is older
> history; this document covers the *engine-improvement* batch and its validation needs.

---

## 1. Branch state
- Work branch: **`dev`** (== `main`), HEAD = `4afd542fc134f22b21b8e3d7fa132329d7bf3dbd`
  (`4afd542`; run `git log --oneline -50`).
- The whole batch was integrated as a series of `merge: …` / `merge R*` commits; the last is
  `4afd542 merge: GPU LIKE device matcher for non-dict Utf8 (StringLikeFilter)`.
- Worktree caveat (process note): agent worktrees fork a FIXED stale base, so each commit's
  author ran `git merge main` first. Nothing for you to do — just context if history looks odd.

## 2. What landed this session (all need GPU validation)

**GPU codegen / execution (HIGHEST validation priority — new device code):**
- **GPU `LIKE` for non-dict Utf8 — `PhysicalPlan::StringLikeFilter`** (`src/exec/string_like.rs`,
  `src/jit/string_kernel.rs::compile_like_match_kernel`, lowering in `physical_plan.rs`).
  ⚠️ **EXPLICITLY UNVALIDATED DEVICE CODE.** Per-row byte matcher for EXACT / PREFIX (`'p%'`) /
  SUFFIX (`'%p'`) / CONTAINS (`'%p%'`) constant patterns (+ `NOT LIKE`). Anything with `_`,
  `ESCAPE`, interior `%`, non-const pattern, or non-bare-Scan input stays on the host path.
  Correctness so far rests ONLY on a host mirror (`like_match_row` vs `exec::like::PatternMatcher`)
  + PTX-shape tests. **VALIDATE FIRST, under `memcheck`.**
- **GPU `UPPER`/`LOWER` — `PhysicalPlan::StringProject`** (`src/exec/string_project.rs`,
  two-pass var-width kernels). Output offsets are host-scanned today; device path is new.
- **GPU `LENGTH` — `PhysicalPlan::StringLength`** (`src/exec/string_length.rs`, per-dict-entry
  length-table gather).
- **GPU `LIKE` for dict-encoded Utf8** — rewrite to integer OR-of-equalities on `__idx_<col>`
  (lower risk: reuses the proven host matcher to build the set; only an integer predicate runs
  on device).
- **GPU `NOT` (`Op::Not`)** — `xor.b32` in `ptx_gen.rs` / `scan_kernel.rs`.
- **CASE over Date32 / Timestamp** (`selp.b32/.b64`), **GPU `SUM(Decimal128)`**, **EXTRACT /
  DATE_TRUNC** scalar fns, planner-driven **sort dispatch (radix + bitonic)**.

**Engine / SQL execution (host-side but new; need e2e checks):**
- **`COUNT(DISTINCT col)`** executes via new `PhysicalPlan::CountRows` (Distinct → count).
- **Uncorrelated scalar / IN subqueries** execute via a pre-lowering resolve pass
  (`src/exec/subquery_resolve.rs`) that runs the subplan and folds it to constants / an IN-list.
- **`EXCEPT` / `INTERSECT` (+ALL)** — host executor `src/exec/setops.rs` (`PhysicalPlan::SetOp`).
- **`SUBSTRING` / `TRIM`** — host eval (`expr_agg` + `string_ops_extended`); SUBSTRING was
  previously rejected outright, now works.
- **Window functions, CTEs, JOIN USING/NATURAL, schema-qualified names** — frontend + host exec.
- **Cost-based join reorder** (statistics → optimizer), **streaming/spill scaffolding**,
  **metrics** hot-path counters, **persistent PTX cache**, **async pinned-copy helpers**.

## 3. Build & test commands (still valid from prior handoff)
```sh
# Host stub (no GPU) — should already pass:
CARGO_INCREMENTAL=0 cargo test --lib --tests --no-default-features --features cuda-stub

# cudarc CUDA driver backend (stable Rust, CUDA 12.x):
cargo test --no-default-features --features cudarc            # <-- RUN ON GPU

# Default linked-CUDA backend (links system CUDA):
cargo test                                                    # <-- RUN ON GPU
```
Gotchas observed: Windows linker is forced to **`lld-link`** in `.cargo/config.toml` (LLVM's
`lld-link` must be on PATH; integration tests embed bundled DuckDB and won't link with MSVC
`link.exe`). Concurrent cargo invocations can corrupt the incremental cache
(`os error 3`) — use `CARGO_INCREMENTAL=0` or `cargo clean -p craton-bolt`.

## 4. The validation checklist — run the GPU-gated tests
All new execution tests are `#[ignore]` with a `gpu:<area>` reason. Run them with `--ignored`:
```sh
# Everything, on GPU:
cargo test --no-default-features --features cudarc -- --ignored

# Or per area (the new/high-risk surface this session touched):
cargo test --features cudarc -- --ignored gpu:string     # LIKE/UPPER/LOWER/LENGTH/SUBSTRING/TRIM
cargo test --features cudarc -- --ignored                # then the e2e SQL files below
```
New / updated test files to focus on (host-stub-green, GPU-unrun):
- `tests/string_fns_sql_test.rs` (UPPER/LOWER/SUBSTRING/TRIM e2e), `tests/like_test.rs`
  (LIKE shapes incl. the new GPU matcher), `tests/string_ops_e2e.rs`.
- `tests/e2e_tests.rs` — `e2e_count_distinct`, the scalar/IN-subquery e2e tests, set-ops.
- `tests/metrics_hotpath_test.rs`.
- In-crate `#[ignore = "gpu:*"]` unit tests in `src/exec/string_like.rs`, `string_project.rs`,
  `string_length.rs`, `subquery_resolve` paths, `setops.rs`.

## 5. Sanitize (catch the memory-safety classes the new device code can hit)
```sh
compute-sanitizer --tool memcheck   cargo test --features cudarc -- --ignored gpu:string
compute-sanitizer --tool racecheck  cargo test --features cudarc -- --ignored
compute-sanitizer --tool initcheck  cargo test --features cudarc -- --ignored gpu:string
```
Pay special attention to **`StringLikeFilter`** (var-width offset/byte indexing — classic
OOB-read risk: empty literal, row shorter than literal, CONTAINS scan bounds, multi-byte UTF-8)
and **`StringProject`** (two-pass output buffer sizing from host-scanned offsets).

## 6. Known NON-bugs (don't chase these)
- **`COUNT(DISTINCT)`** beyond the sole-item/no-GROUP-BY form is rejected by design.
- **Subqueries**: only UNCORRELATED scalar/IN are executed; correlated are rejected at the
  frontend. `>1`-row scalar subquery is a runtime error (correct). NOT-IN-with-NULLs has a
  documented 3VL divergence (see `subquery_resolve.rs` doc comment).
- **GPU `LIKE`/string** intentionally covers only the shapes in §2; everything else is a
  correct host fallback, not a regression.
- **SUBSTRING/TRIM** run host-side (no GPU kernel yet — by design).

## 7. Definition of done
1. `cargo test --features cudarc -- --ignored` and `cargo test -- --ignored` **green on GPU**.
2. `compute-sanitizer memcheck` clean on the `gpu:string` set (esp. `StringLikeFilter`).
3. Spot-check results vs a reference (DuckDB, see `docs/COMPETITIVE_BENCHMARKING.md`) for:
   `COUNT(DISTINCT)`, `EXCEPT`/`INTERSECT`, a scalar + an IN subquery, `UPPER`/`LOWER`/`LENGTH`,
   and each `LIKE` shape (incl. `NOT LIKE`) on a non-dict Utf8 column.
4. Report anything that passed on stub but failed on hardware — the GPU `LIKE` matcher is the
   most likely; if it misbehaves, the safe move is to gate `try_lower_string_like_filter` off
   (fall back to the host `Expr::Like` path) until fixed.
