# Remediation Orchestration (branch: dev)

Pool size 9. Globally file-disjoint ownership so no two concurrent agents conflict.
Agents write code/docs only; never build; never edit mod.rs/lib.rs (report wiring to orchestrator).
Orchestrator builds (`cargo check --no-default-features --features cuda-stub`) after merges, fixes errors, tops up the pool.

## Ownership map (globally disjoint)
- **A** src/cuda/*  — C1(CRIT UAF deferred-free), H1 ctx-guard, H4 from_raw_parts, pool-drain, dict null-alias
- **B** src/exec/engine.rs, src/observability.rs, src/metrics.rs — QueriesTotal counting, projection launch-tag, observer catch_unwind, Decimal128 round-trip
- **C** src/exec/{string_ops,string_ops_extended,string_length,string_col,string_project,string_like,like,validity_audit}.rs — SUBSTRING/LENGTH char-indexed, LENGTH(NULL)→NULL, input_eq_literal 3VL, LENGTH Int64
- **D** src/exec/{groupby*,aggregate,agg_with_pre,expr_agg,extended_agg,welford,partition_offsets,distinct}.rs — G1 twokey COUNT NULL, G2 grouped-float MIN/MAX NaN, NaN key collapse, AVG empty→NULL
- **E** src/jit/{ptx_gen,jit_compiler,disk_cache}.rs — C-3 signed/unsigned validity addr, C-7 lookback launch-guard
- **F** src/exec/{subquery_resolve,dict_registry}.rs — NOT IN(subquery w/ NULL), dict key by (table,col)
- **G** src/exec/{module_cache,gpu_table,gpu_compact,gpu_compact_multipass,gpu_upload}.rs — cache eviction, temporal host fallback
- **H** docs/{ENV_VARS,ARCHITECTURE,MIGRATION_GUIDE,JIT_PIPELINE,COMPETITIVE_BENCHMARKING,GROUPBY_PERF,SQL_REFERENCE}.md
- **I** Cargo.toml, build.rs, CODEOWNERS, README.md, .github/workflows/ci.yml, del docs/CUDARC_ADOPTION.md + docs/CUDA_OXIDE_SWEEP.md, new docs/LIMITATIONS.md

## Wave 2 (launch as owner frees)
- **J** engine.rs+streaming.rs (after B) — wire streaming/morsel path into execute
- **N** tests/* (anytime) — un-ignore strategy, new high-value tests, commit insta PTX snapshots
- **M** string files (after C) — POSITION/REPLACE/LEFT/RIGHT/LPAD/INITCAP/ILIKE/CHAR_LENGTH/OCTET_LENGTH
- **L** groupby_tier2_*.rs (after D + snapshots) — safe shared-helper extraction
- **K** jit partition_reduce_kernel_*.rs (after E + golden snapshots) — spec-parameterised emitter

## Status
Wave 1 launched (background pool, all RUNNING):
- A=a1f0e9d5f314954ea  cuda (CRITICAL UAF)
- B=ac8112276a409730d  engine
- C=a309c97aabe0e85d4  strings
- D=a512c5adc8dbbcdbb  groupby
- E=a826e6505dab4bbd9  jit codegen
- F=a40bf07df59ac6e38  subquery/dict
- G=a3bfcaafb90e7c57f  module-cache/temporal
- H=a3e7266b73357033b  docs
- I=a71fad017eca00cb3  oss/ci

Top-up rules (launch when a slot frees AND deps met):
- B done  -> launch J (streaming wiring; engine.rs+streaming.rs)
- C done  -> launch M (string features)
- E done  -> launch N (tests + commit insta PTX golden snapshots)  [N is dep-free; launch on FIRST free slot regardless]
- D done + snapshots -> launch L (tier2 dedup)
- E done + snapshots -> launch K (jit kernel dedup)
After each batch of merges: orchestrator runs `cargo check --no-default-features --features cuda-stub`, fixes errors, then `cargo test --no-default-features --features cuda-stub` for host tests.

## Orchestrator merge-phase TODOs (cross-file, no agent owns)
1. [E] Wire C-7 lookback launch guard into src/jit/prefix_scan.rs (gridDim single-wave bound + n_rows<=i32::MAX debug_assert + fallback to prefix_scan_multipass). See done_E.md §2.
2. [H] Fix stale src/docs JIT_PIPELINE.md:40 ("KernelSpec-keyed cache … future optimization") — contradicts v0.7 module cache.
3. [G] Add `Err(GpuCapacity) => host re-run` at engine.rs execute_projection dispatch so temporal/decimal host-fallback actually routes. Apply AFTER B finishes (B owns engine.rs).
4. [E/N] After first green build: generate & `cargo insta accept` PTX golden snapshots; commit tests/snapshots/.
5. [B] Check done_B.md for any cuda tagging helper B needed but isn't public in A's src/cuda — if so, make it pub (coordinate A+B).

6. [C] engine.rs `string_length_column` host fallback still pushes 0 for NULL rows — make it emit SQL NULL like string_ops::length now does. Apply after B finishes (B owns engine.rs). Fix in done_C.md.
7. [C] string_ops::length dtype Int32->Int64: no in-tree callers, but verify after merge.

## Completed
Wave 1: A(partial→A2), B, C, D(partial→D2), E, F, G, H, I.
Wave 2: A2 (critical deferred-free LIVE), D2 (groupby F1-F4), W (cross-file wiring), N2 (tests).
  Orchestrator fixes: BinaryOp import, observability test-isolation race, lookback row-budget guard.
  Checkpoints green: 2098 passed / 0 failed / 324 GPU-ignored.
Wave 3 RUNNING: M=a524045951acf20fd (string fns), J=abf8a96806fc82b6c (streaming wiring).

## Wave 4 DONE: IL (ILIKE), L (tier2 dedup), GS (40 golden PTX snapshots).
## Wave 5 DONE: K (jit partition_reduce fragment dedup; snapshots verified byte-identical).

## FINAL STATUS: all review items implemented. dev = 8 commits over baseline.
Final: 2201 host tests passing / 0 failed / 330 GPU-ignored. Lib + all integ bins compile under cuda-stub.

## Remaining (NOT code — require human/GPU; out of agent scope)
1. GPU validation: every fix is verified under cuda-stub + host oracle only. The
   #[ignore]'d GPU correctness suite (incl. diff_duckdb) must run on the GPU CI lane
   (stubbed by agent I) before release.
2. Manual git-history scrub (mailmap/squash) for mixed author identities — orchestrator
   must NOT rewrite shared history without sign-off (oss_readiness.md blocker).
3. CODEOWNERS team creation + `cargo publish --dry-run` (manual, pre-release).
4. Optional/deferred (risky without GPU profiling, left as backlog): AVG fused reduce,
   parallelize host slot-walk, device-compact before D2H; PinnedHostBuffer Drop still
   blanket-sync; streaming eager-tables &mut seam + real PinnedBudget backing.
