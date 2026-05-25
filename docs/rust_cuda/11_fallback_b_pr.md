# Fallback B — Full Revert of the Rust-CUDA Migration

This file is a **pre-written PR description**. It is not filed
yet. The Wave A executor — or any later wave's executor — can
submit it unchanged if Fallback B (full revert of the rust-cuda
migration) is invoked per section 2 of
[`07_risk_assessment.md`](./07_risk_assessment.md).

Open the PR with this body, run the commands in § 2, run the
testing checklist in § 4, and merge once green.

---

## PR title

`Revert rust-cuda migration; restore v0.2-handemit baseline`

## PR body

### Rationale

This PR invokes **Fallback B** of the rust-cuda migration risk
plan ([`docs/rust_cuda/07_risk_assessment.md`](./07_risk_assessment.md),
section 2). It reverts the rust-cuda kernel migration in full and
restores the hand-emit PTX baseline that was tagged at
[`v0.2-handemit`](./09_v02_handemit_tag.md) before Wave A began.

Trigger (fill in the specific risk that materialised — pick whichever applies):

- **R1 materialised** — the rust-cuda upstream has been silent for
  30+ consecutive days and our pinned commit is no longer
  defensible as a long-term dependency. Per the Day-30+
  decision-tree in section 4 of `07_risk_assessment.md`, we are
  not willing to take on the maintenance burden of forking
  upstream (branch (a)) and the snapshot is not stable enough for
  the remaining waves (branch (b) ruled out), so we land on
  branch (c) — Fallback B.
- **R3 materialised** — driver-side PTX rejection on the bench
  host that we cannot work around without changes inside
  `rustc_codegen_nvvm`.
- **R4 materialised** — more than 3 kernels regressed by > 10 %
  vs `v0.2-handemit` and Fallback A would leave rust-cuda owning
  less than half the migrated kernels (migration not paying for
  itself).

The recovery target is the `v0.2-handemit` tag. Its annotation
records the reference bench numbers that this revert must
reproduce. See [`09_v02_handemit_tag.md`](./09_v02_handemit_tag.md)
for the tag's contents and the "what's explicitly preserved" list.

### Commands

These commands must be run by the PR author and the resulting
state pushed to a branch named `fallback-b-revert-rust-cuda`. The
working tree at the end of these commands is what the PR diff
should contain.

```bash
# 1. Branch from current main.
git checkout main
git pull --ff-only
git checkout -b fallback-b-revert-rust-cuda

# 2. Restore every hand-emit kernel source file from the recovery tag.
git checkout v0.2-handemit -- src/jit/

# 3. Remove the rust-cuda workspace member entirely.
git rm -r kernels/
rm -rf kernels/          # belt-and-braces — kill any untracked artefacts

# 4. Restore Cargo.toml and Cargo.lock from the tag. These contain
#    the dependency closure that produced the v0.2-handemit bench
#    numbers and must match exactly.
git checkout v0.2-handemit -- Cargo.toml Cargo.lock

# 5. Restore the executor / dispatch surface that the rust-cuda
#    migration touched. Most of src/exec/ should already be at
#    v0.2-handemit state (the migration is kernel-side), but
#    explicitly restore the dispatcher and launcher to be safe.
git checkout v0.2-handemit -- src/exec/groupby.rs src/exec/launch.rs src/exec/mod.rs

# 6. Restore the cuda module surface in case rust-cuda introduced
#    a new loader path. The from_ptx loader at the tag is the
#    canonical one.
git checkout v0.2-handemit -- src/cuda/mod.rs

# 7. Restore the jit module surface (mod.rs and the PTX-cache).
git checkout v0.2-handemit -- src/jit/mod.rs src/jit/jit_compiler.rs

# 8. Remove the rust-cuda feature from any feature-gating in
#    Cargo.toml that survived steps 4 (paranoia — should already be
#    gone after step 4 since Cargo.toml was restored from the tag).
grep -n 'rust-cuda' Cargo.toml || echo "no rust-cuda feature references — clean"

# 9. Build sanity-check before committing.
cargo build --release --features cuda

# 10. Commit and push.
git add -A
git commit -m "Revert rust-cuda migration; restore v0.2-handemit baseline (Fallback B)"
git push -u origin fallback-b-revert-rust-cuda
```

If step 9 (`cargo build --release --features cuda`) fails, **do
not commit**. Something is missing from the file list above and
must be added before this PR can land. The most likely cause is
a new file under `src/` that depends on the `kernels/` member —
identify it, decide whether it needs to be deleted or restored
from the tag, and re-run the build before continuing.

### What this PR does NOT do

- Does **not** delete the `rust-cuda-experiment` branch. That
  branch survives for a future attempt — Fallback B is reversible
  in principle, just expensive in practice.
- Does **not** delete the
  [`docs/rust_cuda/`](./) directory. The migration's design
  documents stay in tree as the basis for a future retry.
- Does **not** force-move or delete the `v0.2-handemit` tag. Per
  the promise in [`09_v02_handemit_tag.md`](./09_v02_handemit_tag.md)
  section 5, that tag is the floor and stays where it is.

### Bench expectations

Same code → same perf. The merged PR's bench run **must reproduce
the `v0.2-handemit` numbers within criterion noise (±3 %)** on
the same host that produced the originals. The reference numbers
are recorded in the tag annotation
([`09_v02_handemit_tag.md`](./09_v02_handemit_tag.md) § 3) and in
the "Heavy-workload GPU results (2026-05-24, RTX 2060)" section of
[`docs/BENCHMARKS.md`](../BENCHMARKS.md):

| Query    | Expected (rerun) |
|----------|-----------------:|
| `proj`     | ~115.5 ms (432.9 Melem/s) |
| `arith`    | ~124.8 ms (400.7 Melem/s) |
| `filtered` |  ~41.8 ms (1.196 Gelem/s) |

Plus the h2o.ai db-benchmark groupby subset (q1 / q2 / q3 / q4 /
q5 at N = 10 M rows) must verify equivalent across Polars 0.42,
DuckDB 1.2 and Craton Patina on the 100 K-row fixture before timing —
the same protocol as the original measurement run.

If post-merge bench numbers do **not** match within ±3 %, the
revert itself has gone wrong. Open an issue, do not ship.

### Testing checklist

Run before requesting review:

- [ ] `cargo build --release --features cuda` — clean.
- [ ] `cargo build --release --no-default-features --features cuda-stub`
      — docs.rs path stays buildable (per R7 in
      `07_risk_assessment.md`).
- [ ] `cargo test --release --features cuda` — all tests green.
- [ ] `cargo test --release --no-default-features --features cuda-stub`
      — stub-path tests green.
- [ ] `PATINA_BENCH_GPU=1 cargo bench --bench query_benchmarks`
      — heavy-workload GPU numbers match `v0.2-handemit` within ±3 %.
- [ ] `PATINA_BENCH_GPU=1 cargo bench --bench olap_benchmarks`
      — h2o.ai subset verifies equivalent across all three engines
      on the 100 K-row fixture, then timings match `v0.2-handemit`
      within ±3 %.
- [ ] Dispatch-stack tests still find their fast paths.
      Specifically: Tier-1 GROUP BY queries with low cardinality
      (id1 / id2 columns from the h2o.ai bench) must route through
      the shared-memory hash kernels (`src/jit/hash_kernels.rs`),
      and Tier-2 GROUP BY queries with high cardinality (id3) must
      route through the partition / scatter / partition-reduce
      family (`src/jit/partition_kernel*.rs`,
      `src/jit/scatter_kernel*.rs`,
      `src/jit/partition_reduce_kernel*.rs`). Confirm by enabling
      the dispatcher's trace log and observing the kernel names in
      the per-query log lines.
- [ ] `git diff v0.2-handemit -- src/jit/` is empty.
- [ ] `git diff v0.2-handemit -- benches/` is empty.
- [ ] `ls kernels/ 2>/dev/null` returns nothing (directory deleted).

### Follow-up

After this PR merges:

1. Update [`docs/rust_cuda/08_wave_a_outcome.md`](./08_wave_a_outcome.md)
   (or the corresponding wave-outcome doc if Fallback B was
   invoked later) to record which R-number triggered the revert
   and what the post-revert bench numbers were.
2. Update `ROADMAP.md` to reflect that the 0.3 milestone has been
   redefined. Either: (a) defer the rust-cuda migration to 0.4
   and refill 0.3 with the items listed under "Carry-over to
   0.4+" in [`06_milestone_proposal.md`](./06_milestone_proposal.md),
   or (b) pivot to the Fallback C scope (`cust_raw` host-API
   modernisation, per section 2 of `07_risk_assessment.md`).
3. Cut a CHANGELOG entry noting the revert. Public-facing
   language: "the planned 0.3 kernel-language migration has been
   deferred; see `docs/rust_cuda/` for the post-mortem".
4. Do **not** delete the `docs/rust_cuda/` directory. Leave it as
   the design record for the next attempt.
