# `v0.2-handemit` Pre-Tag — Recovery Floor for the Rust-CUDA Migration

Status: pre-Wave-A artefact for the 0.3 milestone. Cut **before**
any commit on the `rust-cuda-experiment` track lands on `main`.

This document specifies the annotated git tag that fixes the
recovery point for the rust-cuda migration. It is the floor the
project commits to: if anything downstream of Wave A goes wrong,
we land here.

---

## 1. Why this tag exists

The 0.3 milestone replaces ~18 hand-emitted PTX kernels with Rust
source compiled through `rustc_codegen_nvvm`
(see [`06_milestone_proposal.md`](./06_milestone_proposal.md)).
The risk assessment in
[`07_risk_assessment.md`](./07_risk_assessment.md) enumerates ten
distinct failure modes, three of which (R1 toolchain abandonment,
R3 driver-side PTX rejection, R4 large perf regression) can fire
mid-migration and force a full revert.

**Fallback B** of section 2 of `07_risk_assessment.md` is the
"revert the migration entirely" escape. That fallback assumes a
git-reachable, working snapshot of the hand-emit world to revert
*to*. This tag is that snapshot.

It exists so that, at any point between Wave A and the 0.3 release,
the team can run a mechanical revert and ship the previously known
good state. Without this tag, Fallback B becomes a forensic
exercise across commits; with it, Fallback B is one PR
([`11_fallback_b_pr.md`](./11_fallback_b_pr.md)).

---

## 2. What gets tagged

**Exactly the current `main` HEAD at the time the tag is cut.**
No staging, no cherry-picking, no rebase. The tag captures:

- Every hand-emit PTX-emitter file under `src/jit/` (see § 4).
- The dispatch stack in `src/exec/groupby.rs` and its sibling
  executors that route Tier-1 / Tier-2 GROUP BY through the
  hand-emit kernels.
- The bench infrastructure: `benches/query_benchmarks.rs` and
  `benches/olap_benchmarks.rs`, including the h2o.ai db-benchmark
  subset and the heavy-workload GPU bench.
- `Cargo.toml` / `Cargo.lock` as they stand at HEAD — the exact
  dependency closure that produces the reference numbers in
  `docs/BENCHMARKS.md`.
- `docs/BENCHMARKS.md` itself, so the recovery target's perf
  expectations are unambiguously pinned in the repo at the tag.

Nothing else. No experimental branches, no WIP. If a file is in
`main` at tag time, it's in the tag; if it isn't, it isn't.

---

## 3. How to create it

The tag is **annotated** (not lightweight) so the release notes
travel with it and `git describe` resolves cleanly.

```bash
# From a clean checkout of main at the intended commit:
git checkout main
git pull --ff-only
git status   # MUST be clean — no untracked, no modified

# Cut the tag. Message body comes from the heredoc below.
git tag -a v0.2-handemit -F - <<'EOF'
Craton Bolt v0.2 hand-emit PTX baseline — recovery floor for the
rust-cuda migration (0.3 milestone).

This tag is the Fallback B target per
docs/rust_cuda/07_risk_assessment.md. If rust-cuda regresses
mid-migration, revert to this tag.

Reference benchmark numbers at this commit (RTX 2060, driver
591.86, CUDA 12.6, see docs/BENCHMARKS.md "Heavy-workload GPU
results, 2026-05-24" for the full table):

  proj       (passthrough)       Craton Bolt   115.5  ms   (432.9 Melem/s)
  arith      (11-op chain)       Craton Bolt   124.8  ms   (400.7 Melem/s)
  filtered   (filter + 4-op)     Craton Bolt    41.8  ms   ( 1.196 Gelem/s)

h2o.ai db-benchmark groupby subset, N = 10 M rows, three-engine
verified equivalent before timing — see the "h2o.ai db-benchmark"
section of docs/BENCHMARKS.md for q1 / q2 / q3 / q4 / q5 medians.

Any Fallback B revert MUST reproduce these numbers within criterion
noise (±3 %) on the same host. If it does not, the revert itself
has gone wrong — do not ship.

Promise: this tag will not be force-moved, deleted, or rewritten.
It is the floor.
EOF

# Push the tag to origin so it survives a local-machine loss.
git push origin v0.2-handemit
```

### GitHub release

Cut a GitHub release pointing at the tag, with the same body as the
annotation. Mark it **pre-release** (it is not a published Craton Bolt
version) and **not** the latest release. Title: `v0.2-handemit
(rust-cuda migration recovery floor)`.

```bash
gh release create v0.2-handemit \
  --title "v0.2-handemit (rust-cuda migration recovery floor)" \
  --notes-file <(git tag -l --format='%(contents)' v0.2-handemit) \
  --prerelease \
  --latest=false
```

The release exists for two reasons: it gives Fallback B a stable
download URL for the hand-emit source tree, and it puts the
recovery target in front of human reviewers in the GitHub UI.

---

## 4. What's explicitly preserved

### Hand-emit kernel files (all of `src/jit/`)

Every file below must be buildable from the tag with
`cargo build --release --features cuda` on a host that has the
0.2-era toolchain. Concretely the file set is:

- `src/jit/mod.rs` — module surface, kernel registry.
- `src/jit/jit_compiler.rs` — `PtxCache`, the 256-entry FIFO cache.
- `src/jit/ptx_gen.rs` — projection / arithmetic PTX emitter.
- `src/jit/scan_kernel.rs` — element-wise scan kernel emitter.
- `src/jit/agg_kernels.rs` — per-dtype reduction tree.
- `src/jit/prefix_scan.rs` — Hillis-Steele prefix scan.
- `src/jit/prefix_scan_multipass.rs` — multi-pass scan for large N.
- `src/jit/hash_kernels.rs` — Tier-1 GROUP BY hash kernels.
- `src/jit/valid_flag_kernels.rs` — null-mask aware kernel paths.
- `src/jit/valid_flag_float.rs` — float-typed null-mask kernels.
- `src/jit/float_atomics.rs` — `atom.cas` bit-pattern loops for
  float min/max (no native `atom.global.{min,max}.f*` through
  sm_90).
- `src/jit/partition_kernel.rs`,
  `src/jit/partition_kernel_i64.rs` — Tier-2 partition emitters.
- `src/jit/scatter_kernel.rs`,
  `src/jit/scatter_kernel_i64.rs` — Tier-2 scatter emitters.
- `src/jit/partition_reduce_kernel.rs`,
  `src/jit/partition_reduce_kernel_i64.rs`,
  `src/jit/partition_reduce_kernel_count.rs`,
  `src/jit/partition_reduce_kernel_count_i64.rs`,
  `src/jit/partition_reduce_kernel_minmax.rs`,
  `src/jit/partition_reduce_kernel_minmax_float.rs`,
  `src/jit/partition_reduce_kernel_multi.rs`,
  `src/jit/partition_reduce_kernel_multi_i64.rs` — the full Tier-2
  reduce family.

That is 18 hand-emit Rust files (excluding `mod.rs`,
`jit_compiler.rs`, and `ptx_gen.rs` infrastructure), matching the
"~18 kernels" count in `06_milestone_proposal.md`.

### Bench suites

- [`benches/query_benchmarks.rs`](../../benches/query_benchmarks.rs)
  — the heavy-workload GPU bench. Reference numbers are in the
  "Heavy-workload GPU results (2026-05-24, RTX 2060)" section of
  [`docs/BENCHMARKS.md`](../BENCHMARKS.md).
- [`benches/olap_benchmarks.rs`](../../benches/olap_benchmarks.rs)
  — the h2o.ai db-benchmark groupby subset (q1 / q2 / q3 / q4 / q5),
  three-engine verified against Polars 0.42 and DuckDB 1.2.

Both bench suites must run green from the tag with no source
modifications. If they do not, the tag is broken — do not push.

### Dispatch / executor surface

- `src/exec/groupby.rs` — the dispatch entry point that selects
  between Tier-1 / Tier-2 GROUP BY paths.
- The Tier-1 / Tier-2 executor modules under `src/exec/` that the
  dispatcher fans out to (the `groupby_shmem_*` and
  `groupby_tier2_*` families).
- `src/exec/launch.rs`, `src/exec/mod.rs` — the launcher glue and
  module surface.

The dispatcher's fast-path selection logic is part of the recovery
target; a Fallback B that lands and silently regresses to a slow
generic path has failed.

---

## 5. Promise

This tag will **not** be force-pushed, deleted, moved, or rewritten.

It is the floor — the version of Craton Bolt we commit to being able
to ship if the rust-cuda migration goes wrong at any point. Anyone
with push access to `origin` who deletes or moves this tag is
defeating the entire risk framework in
[`07_risk_assessment.md`](./07_risk_assessment.md), and the
fallback ladder in particular.

If the tag ever needs to be superseded (e.g. a critical security
fix has to land in the hand-emit baseline before 0.3 ships), cut a
**new** tag — `v0.2-handemit-1`, `v0.2-handemit-2` — and update
[`11_fallback_b_pr.md`](./11_fallback_b_pr.md) and
[`07_risk_assessment.md`](./07_risk_assessment.md) to point at it.
Do not move the original.
