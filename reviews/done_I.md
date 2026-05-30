# Agent I — OSS-readiness remediation (config + limitations docs)

Branch: `dev` · Root: `C:\Projects\bolt` · Date: 2026-05-30

All changes are confined to the files agent I was authorized to touch. No `src/`
code (other than `build.rs`) and no agent-H-owned docs were modified.

---

## Files changed

### `Cargo.toml`
- Added `"reviews"` and `"reviews/*"` to the `package.exclude` list so the
  internal review/orchestration docs (`reviews/oss_readiness.md`,
  `reviews/docs.md`, `SYNTHESIS.md`, `ORCHESTRATION.md`, etc.) never ship in the
  published crate tarball. `continue_prompt.md` was already excluded (and is not
  on disk / not tracked) — left intact.
- Removed two now-dangling comment references to `docs/CUDARC_ADOPTION.md`
  (deleted below) from the `cudarc` dependency and feature comments. All
  required crates.io metadata (description, license, repository, homepage,
  documentation, readme, authors, keywords, categories, rust-version) left
  untouched.

### `build.rs`
- Added a `cuda_lib_found` flag set at every successful
  `cargo:rustc-link-search` emission (CUDA_PATH Windows/Linux branches + both
  fallback branches).
- When the build is a real (non-stub) build and no CUDA lib directory is
  discovered, emits two clear `cargo:warning=` lines: (1) that the build WILL
  FAIL at link time with an unresolved CUDA-driver symbol, and (2) that
  `--no-default-features --features cuda-stub` is the CUDA-less path (and how to
  set up a real GPU build). The `cuda-stub` early-return and all successful
  build behavior are unchanged.

### `README.md`
- Added two short notes to the Status section: (a) CI exercises **0 GPU code
  paths** (cuda-stub only) and GPU correctness is validated separately, and
  (b) a one-line "Limitations / not yet production-ready" pointer to
  `docs/LIMITATIONS.md`.
- Fixed the Project-layout tree: removed the two deleted internal docs
  (`CUDARC_ADOPTION.md`, `CUDA_OXIDE_SWEEP.md`) and added `LIMITATIONS.md`.
  Version (0.7.0), repo (`craton-co/craton-bolt`), Apache-2.0, and
  `@cratonsoftware.com` contact left accurate.
  - Note: the docs-review claim that the tree omitted `JIT_PIPELINE.md` was
    stale — the current tree already lists it, so no change was needed there.

### `CODEOWNERS`
- Added a clearly-labeled "REQUIRED MANUAL STEP (orchestrator)" comment block
  stating the `@craton-co/craton-bolt-maintainers` team must be created in the
  GitHub org BEFORE enabling required-CODEOWNER-review branch protection (such a
  rule is unsatisfiable until the team exists). No real usernames invented.

### `.github/workflows/ci.yml`
- Added `schedule` (weekly cron) and `workflow_dispatch` triggers to `on:`.
- Added a new `gpu-integration` job: clearly labeled stub, `continue-on-error:
  true` AND `if: false` (no-op on github-hosted runners), with a commented
  `runs-on: [self-hosted, linux, gpu, cuda]` and the real
  `cargo test --features cudarc --no-default-features -- --ignored` command
  documented. Existing CPU build / coverage / cargo-deny lanes are unchanged.

## Files created

### `docs/LIMITATIONS.md` (NEW)
- Consolidated, honest limitations page with SPDX tag. Covers: "not production
  ready" disclaimer; hardware/toolkit requirements (sm_70+, CUDA >= 12, 12.x ABI
  pin, platform support table); pre-1.0 API instability; GPU-paths-unverified-in-
  CI; and known semantic gaps (host-side fallbacks, NOT-IN-with-NULL,
  ASCII/byte-oriented string fns, temporal partial-lower / host-upload fallback,
  env-gated experimental kernels, unwired builder knobs). Links to the real
  scattered sources (ROADMAP, SQL_REFERENCE, INSTALL, PATH_TO_1.0, API_SURFACE).

## Files deleted

- `docs/CUDARC_ADOPTION.md` (internal scratch / migration spike) — `git rm`.
- `docs/CUDA_OXIDE_SWEEP.md` (internal refactor tracker) — `git rm`.

---

## REMINDER — manual orchestrator steps (NOT done by agent I)

These are out of agent I's scope and must be performed manually by the
orchestrator/maintainer:

1. **Git history scrub / normalize (BLOCKER before pushing to GitHub).**
   Public history currently exposes `*@ois.gold`, personal yandex/gmail
   addresses, and the `orchestrator <o@local>` bot identity. Settle on a
   canonical public author and either squash to a clean import commit or run
   `git filter-repo --mailmap`. Confirm `continue_prompt.md` is absent from the
   rewritten history. (Agent I did not and must not rewrite history.)

2. **Create the `@craton-co/craton-bolt-maintainers` GitHub team.** Required for
   CODEOWNERS auto-review and before enabling any required-CODEOWNER-review
   branch protection (see the new comment block in `CODEOWNERS`).

3. **Run `cargo publish --dry-run`** (e.g.
   `cargo publish --dry-run --no-default-features --features cuda-stub`) and
   `cargo package --list` to confirm the updated `exclude` list keeps `reviews/`
   (and the heavy test/doc trees, if further trimming is desired) out of the
   tarball and that the packaged build still compiles. (Agent I was instructed
   not to run cargo.)

4. (Optional, recommended) Enable branch protection on `main` with required
   status checks (build matrix, cargo-deny, DCO) once the team is provisioned;
   flip the `gpu-integration` lane's `if: false` to `true` and point `runs-on`
   at the self-hosted GPU runner once one exists.
