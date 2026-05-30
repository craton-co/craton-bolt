# Craton Bolt — Consolidated Review (v0.7.0)

JIT-compiled GPU SQL engine, ~142K LOC Rust. Reviewed by 10 parallel Opus agents, one per module/area.
Per-module detail in: `cuda.md`, `jit.md`, `exec_groupby.md`, `exec_core.md`, `exec_strings.md`,
`tests.md`, `docs.md`, `oss_readiness.md`, `build_kernels.md`.

---

## 1. Code review — bugs, vulnerabilities, stubs, perf

### Correctness bugs (ranked)
| ID | Sev | Location | Issue |
|----|-----|----------|-------|
| C1 | CRITICAL | cuda/smart_ptrs.rs:277 | Kernel-launch stream-tagging is unenforced; `device_ptr()` hands out a raw ptr with no compile-time obligation to tag. One forgotten launch site → pool recycles a live block → use-after-free / silent corruption. Event-based deferred-free is designed in comments but **not implemented**. |
| H1 | HIGH | cuda/cudarc_backend.rs:168 | Default backend `driver_free` has no context-currency guard (cudarc backend does). `GpuBuffer: Send` → drop on a context-less thread leaks/frees wrong context. |
| G1 | HIGH | exec/groupby_tier2_twokey_count_exec.rs:119 | Two-key `COUNT(col)` never declines NULL-bearing columns (single-key executors do) → over-counts NULLs, silent wrong answer. |
| G2 | HIGH | exec grouped float MIN/MAX | No NaN handling/tests; disagrees with the (correct) host scalar DuckDB NaN-as-largest order. |
| S1 | HIGH | exec/string_ops_extended.rs:126 | `SUBSTRING` byte-slices and rounds mid-codepoint start DOWN, leaking bytes before the start; **byte-indexed not char-indexed** (LENGTH/SUBSTRING). A test locks in the wrong result. |
| S3 | HIGH | exec/string_ops.rs:174 | `LENGTH(NULL)` returns 0 (indistinguishable from `''`); validity plumbing now exists, excuse is stale. |
| H4 | HIGH | cuda/cudarc_backend.rs:220 | `memcpy_*` builds a host slice via `from_raw_parts` from a caller raw ptr, only `debug_assert`-guarded → UB on bad `src` before FFI. |

Lower-severity correctness: validity-byte signed-vs-unsigned addressing >2^31 rows (jit ptx_gen.rs:358); NaN GROUP BY/DISTINCT keys not collapsed; `input_eq_literal` 3VL violation (`NULL='x'`→false); `NOT IN (subquery w/ NULLs)`; dict-registry keyed by column-name only → cross-table collision reachable today via UNION/SetOp.

### Vulnerabilities / soundness
- C1/H1/H4 above are the security-relevant ones (memory-safety in unsafe CUDA FFI).
- **No injection/secret/path-traversal issues**: disk PTX cache is path-traversal-hardened, atomic, integrity-checked; `icacls` invoked with direct args (no shell); build-time toolchain download is opt-in + off by default.

### Stubs / unimplemented / not-wired
- **Streaming (exec/streaming.rs)**: complete + tested scaffolding (`BatchStream`/`MorselPlan`/`PinnedBudget`) **never called by the engine** — working set still capped at VRAM. Biggest "looks done, isn't wired" gap.
- GPU `CONCAT`, Decimal128 Div/CAST, two-arg SUBSTRING, `ILIKE` (referenced, absent), LargeUtf8.
- `kernels/` rust-cuda spike unproven (stale `feature(register_attr)`, Feb-2022 `cuda_std 0.2.2` vs 2026 nightly), excluded from crate.
- Date32/Timestamp/Decimal128 upload+gather are hard errors with no host fallback (inconsistent — window.rs supports temporal host-side).
- No `todo!()`/`unimplemented!()` in production paths.

### Performance opportunities
- cuda: land event-based pending-free pool → removes per-Drop blanket-sync stall, enables pinned-memory pool; all-or-nothing pool drain on OOM nukes other queries' warm cache.
- jit: 32 ns fixed spin back-off in 11 files; **~7,600 LOC of 60-70% duplicated `partition_reduce_kernel_*`** → single spec-parameterised emitter (~1,200 LOC target).
- exec_groupby: fuse AVG sum+count reduce (halves probe passes); parallelise single-threaded 4.2M-slot host walk; device-compact before the always-52 MiB D2H.
- exec_core: `GLOBAL_MODULE_CACHE` unbounded (no eviction — leaks cubins for process life); redundant Decimal128 validity D2H+H2D round-trip per query.

---

## 2. Tests — adequacy & coverage

**The headline: GPU correctness is entirely unverified in CI.** CI runs `--features cuda-stub` on GPU-less GitHub runners. 369/2337 test fns (~16%) are `#[ignore]`d and they are *exactly* the GPU correctness tests. A GPU lane is a *comment* in ci.yml.

- Host/frontend/planner/PTX-shape coverage: ~70-80% verified — genuinely solid.
- GPU execution: **~0% verified in CI**. Blended real coverage ≈ **35-45%. The 85% target is NOT met.**
- `diff_duckdb.rs` (the best asset — real DuckDB differential oracle) is 100% `#[ignore]`d.
- PTX "golden" tests are ~202 substring asserts; **no committed full-PTX golden files**; 9 insta snapshots are `#[ignore="bootstrap"]`.
- Tests lock in known-wrong behavior (SUBSTRING byte-slicing asserted as correct).
- Untested: OOM/eviction, drop-during-kernel, cancellation, spill, multi-key UTF-8 sort, all-null aggregates/joins, cache thread-safety.

**Path to trust:** stand up a self-hosted/scheduled GPU runner running `--features cudarc -- --ignored` (instantly lights up diff_duckdb + e2e + sort + join + mempool), add `compute-sanitizer`, commit the insta PTX snapshots (gates register-flow on existing CPU runners), gate coverage on the GPU lane, make it required for release tags.

---

## 3. Documentation & scripts

Large, mostly high-quality corpus. **Not release-clean.** Verified-wrong claims:
- `ENV_VARS.md:254-272` declares 4 **actively-read** env vars as "not present / no effect" (`BOLT_PREFIX_SCAN_ALGO`, `BOLT_HASH_ALGO`, `BOLT_HASH_PROBE_TILED`, `BOLT_SORT_USE_GRAPH`).
- `ARCHITECTURE.md:273`: "No CTEs, subqueries, or window functions" — all three are supported in 0.7.
- `MIGRATION_GUIDE.md:326`: `register_table_stream` shown on wrong type — snippet won't compile.
- `JIT_PIPELINE.md` & `COMPETITIVE_BENCHMARKING.md`: stale "not supported" claims; the latter contradicts itself (single- vs multi-batch).
- `GROUPBY_PERF.md`: "Status: Implemented" but body table shows pre-optimization baseline it claims to supersede.

**Consistency matrix is clean**: version 0.7.0, Apache-2.0, repo `craton-co/craton-bolt`, contact `*@cratonsoftware.com`, MSRV 1.74, GPU floor sm_70 — all consistent. **Note: `craton.com.ar` appears nowhere in the repo** (repo uses `cratonsoftware.com`); reconcile which is canonical.

**Remove/relocate before public**: `CUDARC_ADOPTION.md`, `CUDA_OXIDE_SWEEP.md` are self-labeled internal working notes (TODO checklists, hour estimates, commit SHAs). Add a consolidated "Limitations / not production ready" page. CODEOWNERS points at a non-existent GitHub team.

---

## 4. OSS / crates.io readiness (Apache-2.0, Craton Software Company)

**Verdict: CONDITIONAL GO.** Scaffolding is strong; one history blocker.

**BLOCKER (before GitHub):** git history exposes inconsistent/third-party author identities — `victor.bobrovskiy@ois.gold` (unrelated corporate domain, undercuts sole-copyright story), personal yandex/gmail addresses, and `orchestrator <o@local>` placeholder bot (leaks an AI-orchestration workflow). Public history is permanent → normalize via mailmap or squash to a clean import. Confirm `continue_prompt.md` is absent from rewritten history.

**Clean/strong:** Apache-2.0 LICENSE intact, NOTICE §4(d)-compliant, **100% SPDX coverage (190/190 .rs)**, consistent copyright. No GPL/copyleft/git/path deps. All crates.io metadata present, valid categories, 5 keywords, docs.rs cuda-stub config correct, builds without excluded `kernels/`.

**Must-fix before publish:**
- Add `reviews/*` (and any internal docs) to `Cargo.toml` `exclude` — they'd ship in the tarball otherwise.
- Run `cargo package --list` / `cargo publish --dry-run` (not yet verified).
- `cargo build` without CUDA toolkit hard-fails at **link** (only `cuda-stub` avoids it) — document this sharp edge; emit `cargo:warning=` when no lib dir found.
- Create the CODEOWNERS team; enable branch protection.
- Surface in README that CI exercises 0 GPU paths.

---

## 5. Suggested features / directions

1. **Wire the streaming engine** — biggest latent win; removes the VRAM working-set cap (out-of-core queries).
2. **Trustworthy GPU CI** (self-hosted runner + compute-sanitizer + committed PTX snapshots) — unlocks everything else safely.
3. **Land the event-based deferred-free pool** — closes C1 structurally, removes Drop stalls, enables pinned-memory pool.
4. **Consolidate the kernel-generator duplication** (jit partition_reduce + exec tier2 ≈ 13K+7.6K LOC) into spec-parameterised emitters — only safe *after* real golden snapshots exist.
5. **Fix UTF-8 string semantics** (char-indexed LENGTH/SUBSTRING, `CHAR_LENGTH`/`OCTET_LENGTH`), then add `POSITION`/`REPLACE`/`LEFT`/`RIGHT`/`LPAD`/`INITCAP`/`ILIKE`, variadic CONCAT.
6. Qualify dict-registry by `(table,column)` before any multi-scan plan ships.
7. More aggregates (approximate/HLL distinct, percentiles), spill-to-host for group-by, temporal-key GPU upload with host fallback.
8. Encode launch-tagging in the type system (make C1 unrepresentable).
