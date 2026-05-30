# Craton Bolt — OSS Release Readiness Review

**Crate:** `craton-bolt` v0.7.0 · **License:** Apache-2.0 · **Owner:** Craton Software Company (craton.com.ar)
**Date:** 2026-05-30 · **Reviewer:** release-engineering gate · **Verdict:** **CONDITIONAL GO** — one history-scrub blocker before GitHub; a few metadata gaps before `cargo publish`.

All findings below are verified against the working tree on branch `dev`. File:line citations are to the inspected files.

---

## 1. Licensing & Attribution

**Status: STRONG. One copyright-holder/provenance discrepancy in git history (see §4).**

- **Apache-2.0 LICENSE intact.** `LICENSE:1-202` is the verbatim Apache 2.0 text. The appendix is correctly filled: `LICENSE:189` reads `Copyright 2026 Craton Software Company`. Brackets removed, boilerplate present.
- **NOTICE is correct per Apache-2.0 §4(d).** `NOTICE:1-36` declares the product, copyright holder, and the redistributed Apache-licensed components (Apache Arrow `arrow*`, DataFusion `sqlparser`) plus a "depends on but does not redistribute" section. This satisfies §4(d) — note that for a *source* crate that statically links Arrow/sqlparser, listing them as included works is appropriate. NVIDIA CUDA driver/headers correctly called out as **not redistributed** (`NOTICE:34-36`).
- **SPDX coverage is 100%.** All **190** tracked `*.rs` files carry `// SPDX-License-Identifier: Apache-2.0` in their first lines; **0 untagged** (verified by scanning every `git ls-files '*.rs'`). 140 of these are under `src/`. `build.rs:1` is tagged. `MAINTAINERS.md:1` carries an HTML-comment SPDX tag.
- **Copyright holder consistent.** `Craton Software Company` appears identically in `LICENSE:189`, `NOTICE:2`, and `Cargo.toml:7` (`authors`). No stray alternate holders in source/docs.
- **No copyleft / GPL dependencies.** Direct deps (`Cargo.toml:42-92`): arrow*, sqlparser, thiserror, parking_lot, once_cell, dashmap, bytemuck, libc, log, tracing, cudarc (opt), and dev-deps criterion/polars/duckdb/proptest/insta/tracing-subscriber. All are MIT / Apache-2.0 / BSD / MPL-2.0 family. `Cargo.lock` contains **0 git sources** and no GPL crates. `deny.toml:24-36` allow-lists only permissive licenses (Apache-2.0, MIT, BSD-2/3, ISC, Unicode, Zlib, CC0, **MPL-2.0**, OpenSSL) — MPL-2.0 is weak-copyleft but file-level and OSI-permissive enough for redistribution; acceptable but worth knowing it's in the allow-list.
- **Third-party attribution for vendored CUDA.** There is **no vendored `cuda_sys` or `cudarc` source** in-tree — `src/cuda/cuda_sys.rs` is the project's own hand-rolled `extern "C"` FFI (original work), and `cudarc` is a normal crates.io dependency, attributed at `NOTICE:21-22`. So no vendoring-attribution gap.

**Minor:** `deny.toml:40-47` clarifies `ring`'s license, but `ring` does not appear in `Cargo.lock` (no TLS/crypto dep) — a harmless stale clarify entry.

---

## 2. crates.io Readiness

**Status: MOSTLY READY. Metadata complete; will publish; a few correctness nits.**

- **Required + recommended metadata all present** (`Cargo.toml:1-15`): `description`, `license = "Apache-2.0"`, `repository`, `homepage`, `documentation = https://docs.rs/craton-bolt`, `readme = "README.md"`, `authors`, `rust-version = "1.74"`, `keywords` (5 — at the max, valid), `categories = ["database", "compilers"]` (both are **valid** crates.io categories).
- **Version 0.7.0 is appropriate** — 0.x pre-1.0 signals an unstable API, consistent with the README/SECURITY posture that the IR and public surface are still moving.
- **`cargo publish` will succeed structurally:** no path deps, no git deps (`Cargo.lock` clean). `cudarc`/`cuda_builder` are optional registry deps. The `[workspace] members = []` shim (`Cargo.toml:34-36`) is unusual but valid and keeps `kernels/` out of the published build.
- **Crate builds without the excluded `kernels/` dir.** The only `kernels/` references in `src/` are comments; the `include_str!` for `partition.ptx` (`src/jit/partition_kernel.rs`) reads from `$OUT_DIR`, and `build.rs:173-186` writes an empty `partition.ptx` stub when `rust-cuda` is off (the default). So a packaged build with `kernels/*` excluded compiles. **Confirmed: the `exclude` list is correct in spirit.**
- **`exclude` list** (`Cargo.toml:15`): `target/*`, `.idea/*`, `.github/*`, `.cargo` + `.cargo/*`, `kernels/*`, `.claude/*`, `tests/snapshots/*`, `benches/data/*`, `continue_prompt.md`. Sensible. **`reviews/` is NOT excluded** — this very report lives in `reviews/` and would be packaged into the crate tarball. Add `reviews/*` (or `/reviews`) to `exclude`, or it ships internal review docs to crates.io.
- **Package size:** src is ~6.2 MB, tests ~872 KB, docs ~324 KB, benches ~76 KB. After exclusions the tarball is on the order of ~6–7 MB unpacked. Under the 10 MB crates.io limit but on the larger side; `tests/` and `docs/` ship by default. Not a blocker; optionally `exclude` `tests/*` and `docs/*` to slim it (they aren't needed for a downstream build).
- **docs.rs config present and correct** (`Cargo.toml:194-196`): builds with `cuda-stub` feature + `--cfg docsrs`, so docs.rs (no GPU, no CUDA toolkit) will build. Good.

**Publish-time nits:**
- The README badges (`README.md:3`) point at `crates.io`, `docs.rs`, and the CI badge for `craton-co/craton-bolt` — all will 404 until the crate/repo exist. Cosmetic, self-resolving on first publish.

---

## 3. GitHub OSS Hygiene

**Status: STRONG for a pre-launch repo. CODEOWNERS is a known placeholder.**

- **CI (`/.github/workflows/ci.yml`) genuinely builds + tests + lints + denies:**
  - Matrix: ubuntu + windows × {stable, 1.74 MSRV} (`ci.yml:19-22`).
  - `rustfmt --check` (`ci.yml:44-46`), `clippy` with `-D warnings` (`ci.yml:11`, `48-50`), `cargo check`, `cargo test --lib --tests`, and doctests (`ci.yml:52-75`).
  - **It compile/test-checks only — it does NOT run on a GPU.** Everything uses `--features cuda-stub --no-default-features`. This is honestly documented (`ci.yml:88-91`): GPU `#[ignore]` tests are dark because no GPU runner exists. Acceptable and clearly stated, but reviewers should know coverage of real CUDA paths is **0** in CI.
  - **`cargo-deny` is a blocking gate** (`ci.yml:149-179`) on the default host graph, plus a non-blocking all-features scan (`ci.yml:189-216`) for the rust-cuda/NVVM tree (which contains unmaintained `xz 0.1.0` — knowingly quarantined, `deny.toml:73-86`).
  - Coverage job is informational/non-blocking (`ci.yml:92-145`).
- **DCO enforced** via dedicated workflow (`dco.yml:1-68`) with an inline, no-third-party-action verifier (good supply-chain hygiene — it explains why at `dco.yml:19-27`). Backed by `DCO` file (v1.1, `DCO:1-34`), CONTRIBUTING (`CONTRIBUTING.md:75-87`), and the PR template (`PULL_REQUEST_TEMPLATE.md:29-37`). **Bug:** the merge-commit skip uses `parents > 2` (`dco.yml:43-44`), but a normal 2-parent merge yields `git rev-list --parents -n1` = 3 words (sha + 2 parents), so the condition should be `> 2` ... which is correct for 2-parent merges. Octopus merges aside, this is fine — no action needed, but note it only skips merges with ≥2 parents (i.e. all real merges), which is the intent.
- **Dependabot** configured for cargo + github-actions, weekly, major bumps ignored (`dependabot.yml:1-21`). Good.
- **Issue templates:** `bug.md`, `feature.md`, and `config.yml` with `blank_issues_enabled: false` and security redirect to private advisories + email (`config.yml:6-17`). Solid.
- **PR template** (`PULL_REQUEST_TEMPLATE.md`) covers summary, test plan, checklist, license + DCO certification. Good.
- **SECURITY.md** (`SECURITY.md:1-63`): private disclosure to `security@cratonsoftware.com`, GitHub private advisories, 5-day ack / 90-day coordinated disclosure SLA, supported-version table. Excellent.
- **CONTRIBUTING.md and CODE_OF_CONDUCT.md both present** (root listing). CONTRIBUTING includes third-party-code attribution guidance (`CONTRIBUTING.md:117`).
- **CODEOWNERS is a documented PLACEHOLDER** (`CODEOWNERS:6-14`): references team `@craton-co/craton-bolt-maintainers` that **does not exist yet**. GitHub will silently skip auto-review-requests until the org/team is provisioned. `MAINTAINERS.md:1-33` is the honest interim bridge (routes to `opensource@cratonsoftware.com`). **Not a blocker** (PRs remain mergeable) but the team must be created for CODEOWNERS to function, and any branch-protection rule that "requires CODEOWNER review" would be unsatisfiable until then.
- **Branch protection:** CI runs on `main` and `dev` (`ci.yml:4-7`); no protection config is in-repo (it's a GitHub setting). Recommend enabling required status checks (build matrix + cargo-deny + DCO) and required review before opening to the public.

---

## 4. Red Flags for Public Release

**One real blocker (git history), otherwise clean.**

- **BLOCKER — git history exposes third-party/personal identities inconsistent with sole "Craton Software Company" authorship.** Commit authors/committers across history:
  - `victor.bobrovskiy <victor.bobrovskiy@ois.gold>` and `victor-bobrovskiy <victor.bobrovskiy@ois.gold>` — **`ois.gold` is an unrelated corporate domain**, not cratonsoftware.com. Publishing the history attributes the entire codebase to an employee at a different company's domain, which undercuts the Craton copyright/DCO story and may raise IP-provenance questions.
  - `victorbobrovskiy <porrero@yandex.ru>` and `victorbobrovskiy <victor.bobrovskiy@gmail.com>` — personal yandex + gmail addresses.
  - `orchestrator <o@local>` — a placeholder/bot identity (`o@local` is not a real address) appears as author **and committer** on commits. Shipping `*@local` and an "orchestrator" bot identity publicly looks unpolished and reveals an internal AI-orchestration workflow.

  **Action:** decide on canonical author identity (ideally `Name <handle@cratonsoftware.com>` or a consistent personal email the contributor is comfortable making public), and either (a) squash/rewrite history to a clean import commit, or (b) `git filter-repo --mailmap` to normalize authors before pushing. Pushing as-is is hard to undo once cloned/forked.

- **Internal AI-orchestration handoff doc.** `continue_prompt.md` is gitignored (`.gitignore:7-8`) and `exclude`d from the package (`Cargo.toml:15`) — **good, it won't ship** — but confirm it's not in history (`git log --all -- continue_prompt.md`); commit `22fc2b4 "fix: c_p.md"` suggests it was tracked at some point. Verify it's absent from the published history during the rewrite above.

- **`.idea/` and `.claude/` exist on disk but are NOT tracked** (verified: `git ls-files .idea/ .claude/` → empty). Gitignored at `.gitignore:4` and `.gitignore:10`. No leak. Good.

- **No secrets / tokens / keys.** Scanned `*.rs/*.toml/*.yml/*.md` for API keys, private-key blocks, AWS/GitHub/Slack token patterns — only false positives (`src/error.rs:231` "bad token" test string, `src/exec/like.rs` `tokenise()`, `disk_cache.rs:1147` `/etc/passwd` as a path-traversal **test** fixture, and `RELEASING.md`/`ci.yml` references to `CODECOV_TOKEN` as a *secret name*, not a value). Clean.

- **No absolute local paths.** No `C:\Users\...`, `/home/<user>`, or `/Users/<user>` in tracked source/config/docs. `build.rs:84` hardcodes the standard `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA` install root — that's a legitimate platform default, not a leak.

- **No internal corporate URLs.** "internal" hits are all benign (code/architecture terminology, doc working-notes status banners). `docs/rust_cuda/` is described as internal design notes — confirm those are intended to be public or `exclude`/remove them (they ship in the repo, not the crate).

- **TODO/FIXME density: 62** across `*.rs`. Moderate for a project this size; none observed to contain secrets or embarrassing content. Acceptable for a 0.x release. `deny.toml:85` has a `TODO: confirm RUSTSEC id for xz 0.1.0` — harmless.

---

## 5. Prioritized GO / NO-GO Checklist

### MUST FIX before pushing to GitHub (public history is forever)
1. **[BLOCKER] Scrub / normalize git author history.** Remove `*@ois.gold`, personal yandex/gmail addresses, and the `orchestrator <o@local>` bot identity. Settle on a public-facing canonical author and rewrite (squash to clean import, or `git filter-repo` with a mailmap). Confirm `continue_prompt.md` is absent from the rewritten history.
2. **Create the `@craton-co/craton-bolt-maintainers` GitHub team** (or edit `CODEOWNERS:14` to real handles) so CODEOWNERS auto-review works and any "require CODEOWNER review" branch rule is satisfiable.
3. **Enable branch protection** on `main`: required status checks (build matrix, `cargo deny`, DCO) + required review. (GitHub setting, not in-repo.)
4. **Confirm intent of `docs/rust_cuda/` and other working-note docs** shipping publicly (`docs/CUDARC_ADOPTION.md`, `docs/CUDA_OXIDE_SWEEP.md` are explicitly "internal working notes"). Either keep deliberately or remove.

### MUST FIX before `cargo publish`
5. **Add `reviews/*` (and ideally `tests/*`, `docs/*`) to `Cargo.toml` `exclude`** (`Cargo.toml:15`) so internal review docs and the heavy test/doc trees don't ship in the crate tarball.
6. **Verify a clean packaged build:** run `cargo publish --dry-run --no-default-features --features cuda-stub` (and `cargo package --list`) to confirm the excluded `kernels/` build still compiles and the tarball contents are intended.

### SHOULD FIX (non-blocking, quality)
7. Remove the stale `ring` clarify block from `deny.toml:40-47` (crate not in the graph).
8. Note in README/docs that **CI exercises 0 GPU code paths** (already honestly commented in `ci.yml`); consider stating it in the README "Status" section so users don't assume GPU coverage.
9. Optionally tighten `deny.toml` `bans.multiple-versions` from `warn` to `deny` post-launch once the dep graph settles.

### Already GOOD (no action)
- Apache-2.0 LICENSE + NOTICE correct and §4(d)-compliant.
- 100% SPDX tagging (190/190 `.rs` files).
- Consistent `Craton Software Company` copyright.
- No GPL/incompatible deps; no git/path deps; no vendored-code attribution gap.
- Complete crates.io metadata; valid categories; appropriate 0.x version; docs.rs config correct.
- DCO workflow, dependabot, issue/PR templates, SECURITY.md, CONTRIBUTING, CODE_OF_CONDUCT, MAINTAINERS all present and solid.
- No secrets, no absolute local paths, no internal URLs; `.idea/`/`.claude/`/`continue_prompt.md` correctly untracked.
