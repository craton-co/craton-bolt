# Wave A Go / No-Go Checklist — Rust-CUDA Migration

Status: executable checklist. Run at the end of Wave A (toolchain
spike, ~1 week) before any commit on the `rust-cuda-experiment`
track is permitted to land on `main`.

This document is the procedural form of the Wave A gate in section
3 of [`07_risk_assessment.md`](./07_risk_assessment.md). Every
item below is a binary check; the evaluator fills in the Result
field after running the listed test. Decision matrix at the end.

Sibling docs:
- Milestone plan: [`06_milestone_proposal.md`](./06_milestone_proposal.md)
- Risk inventory + fallback ladder: [`07_risk_assessment.md`](./07_risk_assessment.md)
- Pre-tag spec: [`09_v02_handemit_tag.md`](./09_v02_handemit_tag.md)
- Fallback B PR scaffold: [`11_fallback_b_pr.md`](./11_fallback_b_pr.md)

---

## Checklist

### 1. `partition_kernel` compiles via `rustc_codegen_nvvm`

- [ ] **Check.**
- **Test:** in the `kernels/` workspace member, run
  `cargo +<pinned-nightly> build --release` against the
  partition-kernel source. The `rust-toolchain.toml` in
  `kernels/` must pin the nightly. Output PTX is written to a
  known artefact path (see `04_build_integration.md`).
- **Expected:** clean build with no `rustc_codegen_nvvm` panics
  and a non-empty `.ptx` artefact produced. `ptxas --verbose` on
  the artefact reports no errors.
- **Result:** _Pass / Fail / Inconclusive — fill in._

### 2. Output PTX loads via `CudaModule::from_ptx` and matches hand-emit answer

- [ ] **Check.**
- **Test:** load the rust-cuda-produced PTX through the existing
  `CudaModule::from_ptx` path. Run the canonical partition-kernel
  fixture (the same input rows the hand-emit version is tested
  against in `src/jit/partition_kernel.rs`'s unit tests, lifted
  into `tests/wave_a_partition_kernel.rs`). Compare output buffer
  byte-for-byte against hand-emit output on the same input.
- **Expected:** `cuModuleLoadDataEx` returns `CUDA_SUCCESS`;
  kernel launches; bit-identical output. No driver-side `CUDA_ERROR_*`.
- **Result:** _Pass / Fail / Inconclusive — fill in._

### 3. Bench delta on the spike kernel ≤ 5 % vs hand-emit

- [ ] **Check.**
- **Test:** criterion micro-bench against the partition-kernel
  alone (not the full query). 3 runs, median of medians, same
  bench host as the `v0.2-handemit` reference numbers in
  [`docs/BENCHMARKS.md`](../BENCHMARKS.md). Compare against the
  hand-emit baseline recorded at the `v0.2-handemit` tag.
- **Expected:** rust-cuda median within ±5 % of hand-emit median.
  Criterion confidence interval ≤ ±3 %.
- **Result:** _Pass / Fail / Inconclusive — fill in._

### 4. Build-time impact on full `cargo build` ≤ +60 s

- [ ] **Check.**
- **Test:** clean target, all features:
  `cargo clean && /usr/bin/time -v cargo build --release --all-features`
  twice — once at the `v0.2-handemit` tag, once at the Wave A
  spike branch. Take the wall-time delta.
- **Expected:** spike build wall-time ≤ baseline + 60 s. Use
  `cargo build --timings` to confirm the delta lives in the
  `kernels/` member and not in unintended host-crate
  recompilation.
- **Result:** _Pass / Fail / Inconclusive — fill in._

### 5. Workspace-member toolchain pinning works — host crate stays on stable

- [ ] **Check.**
- **Test:** with the `kernels/` member's `rust-toolchain.toml`
  pinning nightly, run `cargo +stable build --release` at the
  workspace root *targeting only the host crate*
  (`cargo build -p javelin`). Then confirm the host crate's
  `Cargo.lock` and feature flags do not require nightly anywhere.
  CI matrix should exercise both `(stub, stable)` and
  `(real-cuda, nightly, rust-cuda)` rows per R7 in
  `07_risk_assessment.md`.
- **Expected:** `javelin` builds on stable; `kernels` builds on
  nightly; no `#![feature(...)]` leakage into `javelin`.
- **Result:** _Pass / Fail / Inconclusive — fill in._

### 6. Pre-tag `v0.2-handemit` exists and is reachable

- [ ] **Check.**
- **Test:** `git tag -l v0.2-handemit` returns the tag locally;
  `git ls-remote --tags origin v0.2-handemit` returns the tag on
  `origin`; `git show v0.2-handemit` prints the annotation body
  specified in [`09_v02_handemit_tag.md`](./09_v02_handemit_tag.md);
  `gh release view v0.2-handemit` shows a pre-release on GitHub.
- **Expected:** all four commands succeed; tag is annotated (not
  lightweight) and points at the intended `main` commit.
- **Result:** _Pass / Fail / Inconclusive — fill in._

### 7. Fallback B PR scaffold exists and is reviewable

- [ ] **Check.**
- **Test:** [`11_fallback_b_pr.md`](./11_fallback_b_pr.md) is
  present, lints clean, contains the exact `git` commands needed
  to revert, lists testing checklist items, and cross-references
  `07_risk_assessment.md` R-numbers.
- **Expected:** the doc reads as a submittable PR description —
  the Wave A executor could open the PR with no further editing
  and the reviewer would understand what changed and why.
- **Result:** _Pass / Fail / Inconclusive — fill in._

### 8. `docs/rust_cuda/08_wave_a_outcome.md` has been written

- [ ] **Check.**
- **Test:** the Wave A executor has produced
  `docs/rust_cuda/08_wave_a_outcome.md` recording what they did,
  what they measured, and what they recommend. The doc must
  reference the bench numbers from check 3 and the build-time
  delta from check 4 explicitly.
- **Expected:** doc exists, is not a stub, contains concrete
  numbers (not "TBD"), and ends with a recommendation: proceed
  to Wave B / re-evaluate / invoke Fallback C.
- **Result:** _Pass / Fail / Inconclusive — fill in._

---

## Decision matrix

Count Pass results across all 8 checks above.

| Pass count | Action |
|------------|--------|
| **≥ 6 / 8** | Proceed to Wave B. The toolchain spike is green; structural risk is manageable. Document any < 6 failures in `08_wave_a_outcome.md` so Wave B can address them. |
| **4 – 5 / 8** | **Stop.** Re-evaluate before continuing. Hold a go / no-go review with the maintainer. The most likely causes of a 4–5 outcome are R2 (nightly breakage) or R4 (≥ 5 % perf regression); decide whether to retry the spike with a different pinned nightly, accept the regression and continue, or pivot to Fallback C. Do not advance to Wave B without an explicit written decision. |
| **≤ 3 / 8** | **Invoke Fallback C immediately.** The NVVM codegen path is not viable on the project's terms. Pivot to the `cust_raw` host-API modernisation per section 2 of `07_risk_assessment.md`. Cut the `rust-cuda-experiment` branch, archive it, and update `ROADMAP.md` to reflect the redefined 0.3 scope. |

**Inconclusive** results count as **Fail** for the purpose of the
matrix. Wave A is short and cheap; if a check can't be evaluated
clearly, the answer is "it does not work yet". Do not paper over
an inconclusive result by re-classifying it.

---

## Notes for the evaluator

- Run the checks in order. Checks 1 → 5 are cumulative — a failure
  early on usually invalidates later checks. If check 2 fails
  (PTX won't load), checks 3 and 4 are moot.
- Use the bench host that produced the
  `v0.2-handemit` numbers in
  [`docs/BENCHMARKS.md`](../BENCHMARKS.md). Switching hosts mid-eval
  defeats the comparison.
- If check 3 (perf delta) is the only failure, do **not** flip to
  Fallback C — that's R4 territory, and Fallback A (per-kernel
  rollback flag) is the correct response, not Fallback C. Check 3
  failures get logged in `08_wave_a_outcome.md` with a Wave-B
  remediation plan; the overall Wave A gate can still pass if the
  other 7 checks pass.
- Conversely, if check 1 or check 2 fails, the migration is not
  viable as designed. Even a 7/8 score with checks 1–2 failing is
  a Fallback C trigger, because the rest of the milestone presumes
  those work.
