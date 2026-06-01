# Bolt — OSS / crates.io Publication Readiness Review

Repo: `C:\Projects\bolt` · Intended license: Apache-2.0 · Copyright holder: Craton Software Company (craton.com.ar)
Package name: `craton-bolt` v0.7.0 · Review date: 2026-05-30

---

## Verdict

The project is **close to release-ready** with strong governance hygiene, but there is **one hard blocker** (CI workflows live in `.github/.wf/` instead of `.github/workflows/`, so no CI/DCO check actually runs on GitHub) and a handful of **medium** issues (an internal handoff prompt file shipping publicly + in the crate, a stale `exclude` glob, an unconfirmed/un-ignored RUSTSEC id). No copyleft/legal blocker to an Apache-2.0 release under Craton ownership was found.

---

## 1. License / Copyright / Attribution

- **LICENSE** (`LICENSE:1-201`) is the full, correct Apache-2.0 body text (Definitions through "END OF TERMS AND CONDITIONS", verbatim). **The Appendix was edited**: the canonical file keeps the boilerplate placeholder `Copyright [yyyy] [name of copyright owner]`, but `LICENSE:189` hard-codes `Copyright 2026 Craton Software Company`. This is a cosmetic deviation, not a legal defect — the Appendix is *meant* to be filled when you apply the notice to files — but strict license scanners (and `cargo-deny`'s confidence-threshold matcher) prefer the unmodified template. **Recommendation:** restore the bracket placeholders in the Appendix and keep the filled copyright only in NOTICE + per-file SPDX headers. Low priority.
- **NOTICE** (`NOTICE:1-36`) follows Apache-2.0 conventions: product name + `Copyright 2026 Craton Software Company` (`NOTICE:2`), attribution for redistributed Apache Arrow / sqlparser code, and a (non-required but tidy) survey of dev/optional deps. Correctly notes CUDA driver is © NVIDIA and not redistributed (`NOTICE:34-36`). Good.
- **SPDX headers**: **all 140 `src/**/*.rs` files** carry `// SPDX-License-Identifier: Apache-2.0` on line 1 (verified by grep — 140/140, zero misses). `MAINTAINERS.md:1` also carries it. Excellent.
- **Copyright year/owner consistency**: `2026` + "Craton Software Company" is consistent across `LICENSE:189`, `NOTICE:2`, `Cargo.toml:7` (`authors = ["Craton Software Company <opensource@cratonsoftware.com>"]`), and `README.md:242,246`. No mismatches.
- Email domains: code uses `cratonsoftware.com` (opensource@, security@, conduct@). The task brief cites the company web as `craton.com.ar` — confirm the `cratonsoftware.com` domain is owned/controlled by Craton and that those mailboxes exist before launch (see §5).

## 2. crates.io readiness (`Cargo.toml`)

All required + recommended fields present (`Cargo.toml:1-15`): `name`, `version`, `edition=2021`, `description`, `license = "Apache-2.0"`, `authors`, `repository`, `homepage`, `documentation` (docs.rs), `readme`, `keywords` (5, the max), `categories` (`database`, `compilers` — both valid crates.io slugs), `rust-version = "1.74"`. `[package.metadata.docs.rs]` (`Cargo.toml:194-196`) correctly builds docs with `cuda-stub` so docs.rs won't try to link CUDA. This is well done.

- **Path deps that block publish: NONE.** The only `path =` entries are `[lib]` (`:40`) and the two `[[example]]` blocks (`:173,177`) — all in-package, fine. The `kernels/` crate is **not** a dependency: the root declares an explicit `[workspace] members=[] exclude=["kernels", ".claude/worktrees"]` (`Cargo.toml:34-36`) and `kernels/Cargo.toml:22` sets `publish = false`. `kernels` is invoked only from `build.rs` under the optional `rust-cuda` feature. So `cargo publish` will not be blocked by the GPU-kernels spike. Cargo.lock has **no `git`/`path` sources** (verified) — all from crates.io-index.
- **Package size / contents:** `exclude` (`Cargo.toml:15`) drops `target`, `.idea`, `.github`, `.cargo`, `kernels`, `.claude`, `tests/snapshots`, `benches/data`, `reviews`. **Two gaps:**
  1. **`bolt_continue_prompt.md` ships in the crate.** The exclude lists `"continue_prompt.md"` but the actual tracked file is **`bolt_continue_prompt.md`** (name mismatch → glob does not match). This is an internal AI agent handoff doc (mentions branch internals, "previous agent", unvalidated GPU LIKE matcher, security `V-#` history). It is currently **git-tracked (public on GitHub) AND would be packaged into the published crate.** MEDIUM — remove the file (or fix the exclude to `bolt_continue_prompt.md`) and untrack it.
  2. `docs/` (~316 KB, 15 files) is **not** excluded, so it ships in the crate. That's acceptable/desirable (architecture, SQL reference). No action required.
- **Crate-name squatting risk:** package name `craton-bolt` matches repo `craton-co/craton-bolt`. Per brief, not network-checked — **flag:** verify `craton-bolt` is unclaimed on crates.io before the first `git tag` (RELEASING.md:42-44,126 already calls this out). Names cannot be transferred once taken.
- **Will `cargo publish` work?** Most likely yes with `--no-default-features` default (default features are empty, `Cargo.toml:95`). The crate must *build* during the package verify step; the default build has no CUDA link requirement only if the code compiles without the `cudarc`/real-CUDA path. RELEASING.md:17 mandates a `cargo publish --dry-run` — that gate must be green before launch. Treat dry-run as the publish acceptance test.

## 3. Dependency license compliance (`deny.toml` + `Cargo.lock`)

- `deny.toml:19-36` enforces a license **allowlist** (`version = 2`, so anything not listed is denied). The list is Apache-2.0-compatible permissive: Apache-2.0, MIT, BSD-2/3-Clause, ISC, Unicode-DFS-2016, Unicode-3.0, Zlib, CC0-1.0, MPL-2.0, OpenSSL. **No GPL/AGPL/LGPL is allowed** — good. MPL-2.0 is file-level copyleft but Apache-compatible for redistribution; fine to keep. `ring` is clarified (`deny.toml:40-47`).
- **Copyleft deps in Cargo.lock:** none of the GPL/AGPL family found. The lock's notable extras are the **rust-cuda build tree** — `rustc_codegen_nvvm 0.3.0` (`Cargo.lock:2384`), `nvvm 0.1.1` (`:1618`), `find_cuda_helper 0.2.0` (`:1063`), and `xz 0.1.0` (`:3251`). These are only reachable through the optional `rust-cuda` feature and `kernels/`; the **default host graph is clean**. `deny.toml` only scans the default graph (`all-features = false`, `:12`).
- **Supply-chain caveat (`xz 0.1.0`):** known-**unmaintained**; `advisories.unmaintained = "deny"` (`deny.toml:73`) would fail on it, so the all-features scan is run as a **non-blocking** CI job (`ci.yml:196-223`, `continue-on-error: true`). `deny.toml:85` leaves a `TODO: confirm RUSTSEC id for xz 0.1.0 before ignoring it explicitly` and the `ignore` list is intentionally empty. This is an honest, defensible posture, but it means the rust-cuda graph is never gated. LOW/MEDIUM — fine for launch (feature is off by default + experimental), but track it.

## 4. Supply-chain / CI

- **BLOCKER — workflows are in the wrong directory.** Both `ci.yml` and `dco.yml` live in **`.github/.wf/`** (`.github/.wf/ci.yml`, `.github/.wf/dco.yml`). GitHub Actions only executes workflows under **`.github/workflows/`**. As committed, **no CI runs, DCO is not enforced, and cargo-deny never gates** on GitHub — and the README CI badge (`README.md:3`, points at `actions/workflows/ci.yml`) will be perpetually "no status / unknown". The workflow *content* is otherwise solid:
  - `ci.yml`: build matrix (ubuntu+windows × stable+1.74 MSRV), `RUSTFLAGS: -D warnings`, rustfmt, clippy, `cargo check/test/doc` under `cuda-stub`, informational coverage, **blocking** `cargo deny check advisories licenses bans` (`:185-186`), non-blocking all-features deny, and a documented `if:false` GPU lane stub.
  - `dco.yml`: inline per-commit `Signed-off-by:` verification over the PR range, skips merge commits, no unpinned third-party action. Good design.
  - **Fix:** rename/move `.github/.wf/` → `.github/workflows/`. This single move flips CI + DCO from inert to enforced.
- **Dependabot** (`dependabot.yml`) sane: weekly `cargo` + `github-actions` ecosystems, PR limit 5, ignores semver-major bumps. Note: the github-actions updater only works once the workflows are in `.github/workflows/`.
- **Secrets/tokens:** none committed. No `.pem/.key/.env/id_rsa` files tracked. `RELEASING.md:69-76` documents `CRATES_IO_TOKEN`/`CODECOV_TOKEN` as repo secrets (not in-repo). Codecov upload is guarded on `env.CODECOV_TOKEN != ''` (`ci.yml:145`) so it no-ops on forks. Good.
- **`.gitignore`** (`.gitignore:1-23`) covers `/target/`, `.claude/`, IDE dirs, OS junk. Verified `.idea/` and `.claude/` are **not** tracked. Gap: `.idea` is matched without a trailing slash (fine on git), but there is **no generic ignore for secrets** (`*.env`, `*.pem`) — low risk today since none exist; consider adding defensively.
- **`.cargo/config.toml`** only sets `lld-link` on `x86_64-pc-windows-msvc` — environment-specific, harmless, but it is excluded from the package so downstream users won't inherit it.

## 5. GitHub hygiene

- **Issue/PR templates:** `ISSUE_TEMPLATE/bug.md`, `feature.md`, and `config.yml` (blank issues disabled, security routed to advisories + email) are all present and good. `PULL_REQUEST_TEMPLATE.md` includes a test-plan checklist and an explicit **License & DCO sign-off** section (`PULL_REQUEST_TEMPLATE.md:29-37`).
- **CODEOWNERS** (`CODEOWNERS:22`) references `@craton-co/craton-bolt-maintainers`, a team that **does not yet exist**. The file documents this clearly and warns NOT to enable "require Code Owner review" branch protection until the team is provisioned (else all merges block). This is a correct, self-aware placeholder — **action required before enabling that branch-protection rule**, but not a publish blocker.
- **SECURITY.md** is thorough: private disclosure via `security@cratonsoftware.com` + GitHub private advisories, 5-day ack / 10-day assessment / 90-day coordinated disclosure, supported-version table (0.7.x). Good.
- **CODE_OF_CONDUCT.md** (Contributor Covenant 2.1) with enforcement contact `conduct@cratonsoftware.com`.
- **CONTRIBUTING.md** has clear DCO `git commit -s` instructions (`:75-87`), SPDX-header requirement for new files (`:106-115`), inbound=outbound licensing (`:96-104`), code-style rules. **DCO** file is the verbatim DCO 1.1 text.
- **MAINTAINERS.md** documents the interim review process while the GH team is being created.
- **Action:** all four contact mailboxes (`opensource@`, `security@`, `conduct@`, plus the `@craton-co` org/team) must actually be provisioned and monitored before going public.

## 6. Blockers & prioritized checklist

**Hard blocker (must fix before public release):**
1. **Move `.github/.wf/` → `.github/workflows/`** so CI, DCO enforcement, and cargo-deny gating actually run, and the README CI badge resolves. (`.github/.wf/ci.yml`, `.github/.wf/dco.yml`)

**High (fix before first publish):**
2. **Remove `bolt_continue_prompt.md`** from git and from the package — it's an internal agent handoff doc that would be both public on GitHub and shipped in the crate (exclude glob says `continue_prompt.md`, file is `bolt_continue_prompt.md`). (`Cargo.toml:15`)
3. **Run `cargo publish --dry-run`** and `cargo package --list`; confirm contents and a clean default build (per `RELEASING.md:17`). Treat as the publish acceptance gate.
4. **Provision** the `@craton-co/craton-bolt-maintainers` GitHub team and the `opensource@`/`security@`/`conduct@cratonsoftware.com` mailboxes; verify `cratonsoftware.com` ownership vs the `craton.com.ar` web presence.

**Medium:**
5. **Verify `craton-bolt` crate name is unclaimed** on crates.io before tagging (no transfers after first publish).
6. **Resolve the `xz 0.1.0` advisory posture** (`deny.toml:85` TODO): either confirm the RUSTSEC id and add it to `ignore` with justification, or accept the non-blocking all-features scan and document it. Off-default feature, so not launch-critical.
7. Decide whether the git-tracked **`reviews/`** internal docs should be public (excluded from the crate via `Cargo.toml:15`, but visible on GitHub). Not a legal issue; a polish/optics one.

**Low:**
8. Restore the Apache-2.0 Appendix **bracket placeholders** in `LICENSE:189` (keep filled copyright in NOTICE + headers) for clean license-scanner matching.
9. Add defensive secret globs (`*.env`, `*.pem`, `*.key`) to `.gitignore`.

**No legal blocker** to an Apache-2.0 release under Craton ownership was found: license text is correct, attribution is in order, every source file is SPDX-tagged, the dep license allowlist excludes copyleft, and no copyleft/GPL deps appear in the default dependency graph.
