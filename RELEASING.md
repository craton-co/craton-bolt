# Releasing Craton Bolt

This document is for maintainers. It lists the steps required to cut a release,
publish to crates.io, and announce it publicly. Each step notes who has the
required credentials and what needs to be prepared beforehand.

---

## 1. Pre-release checklist

- [ ] `CHANGELOG.md` — rename `[Unreleased]` to the new version with today's
  date. Verify every entry has a section (`Added`, `Changed`, `Fixed`, `Removed`).
- [ ] `Cargo.toml` — bump `version` to the new version number.
- [ ] Run `cargo build --release` and `cargo test --lib` locally (set
  `CUDA_PATH` to the v12.x toolkit on Windows if both v12 and v13 are installed).
- [ ] Run `cargo package --list` and confirm the file list looks right.
- [ ] Run `cargo publish --dry-run` and fix any packaging errors.
- [ ] Run `cargo deny check licenses` — confirm no GPL or unknown-license
  transitive deps are present.

## 2. Tagging a release on GitHub

**Credentials needed:** `push` access to `origin`.

```bash
git tag -a vX.Y.Z -m "vX.Y.Z — <one-line release summary>"
git push origin vX.Y.Z
```

GitHub Actions will pick up the tag; confirm the CI run is green before proceeding.

## 3. Publishing to crates.io

**Credentials needed:** a crates.io account with publish rights on `craton-bolt`.

```bash
cargo publish
```

Notes:
- Run `--dry-run` first (step 1 above).
- If this is the first publish, verify the crate name `craton-bolt` is unclaimed
  before the `git tag` step — crate names cannot be transferred once taken.

## 4. Verifying the docs.rs build

The build triggers automatically on publish. Monitor:
<https://docs.rs/crate/craton-bolt/latest>

If the build fails, the `cuda-stub` feature (used by docs.rs) is the most likely
culprit. Fix locally with:

```bash
cargo build --no-default-features --features cuda-stub
```

## 5. GPG-signing release artifacts

**Credentials needed:** a GPG key registered to the maintainer email with the
secret key available locally.

```bash
gpg --armor --detach-sign craton-bolt-<version>.crate
```

Attach the `.asc` signature to the GitHub release.

## 6. GitHub Secrets (CI)

Set the following secrets in **Settings → Secrets → Actions** before automating
the publish step:

| Secret | Purpose |
|--------|---------|
| `CRATES_IO_TOKEN` | `cargo publish` in CI |
| `CODECOV_TOKEN` | Coverage upload (once wired) |

## 7. Announcing the release

Platform-specific templates can be drafted from the release notes. Suggested
channels: Hacker News ("Show HN"), `r/rust`, `r/databases`, This Week in Rust,
Mastodon/Bluesky.

## 8. Hardware-bound tasks

### Re-run benches on production-class hardware (V100 / A100 / H100)

The results in `docs/BENCHMARKS.md` were captured on a consumer RTX 2060. Numbers
on data-centre GPUs would be substantially better. The harness is ready:

```bash
BOLT_BENCH_GPU=1 cargo bench --bench olap_benchmarks
```

Update `docs/BENCHMARKS.md` with the new numbers and record the hardware details.

### Test on Jetson (aarch64-linux)

The crate should work — nothing is x86-specific — but Jetson builds against CUDA
on `aarch64` have historically had quirks. Run the full test suite and add the
platform to the CI matrix once validated.

### Windows CUDA Toolkit v13 workaround

CUDA Toolkit v13.2's stub `cuda.lib` lacks `__imp_*` symbols that MSVC's linker
expects. Workaround: set `CUDA_PATH` to the v12.6 installation when running
`cargo test` on Windows. `build.rs` already prefers the highest-versioned v12.x
over v13.x when both are present.

## 9. Legal

- **Trademark**: run a search on "Craton Bolt" via USPTO TESS / WIPO Global Brand
  before any large-scale marketing push.
- **Contributor agreement**: DCO sign-off is in place (`DCO` file,
  `CONTRIBUTING.md`). No CLA is required; a CLA would need legal review before
  adoption.
- **Third-party license audit**: run `cargo deny check licenses` before each
  publish (see step 1). A `deny.toml` is present at the repo root and is wired
  into CI to automate this check.

## 10. Things to check before any public announcement

1. Is the CHANGELOG `[Unreleased]` section closed and named?
2. Is the version in `Cargo.toml` bumped?
3. Does `cargo publish --dry-run` pass cleanly?
4. Is the `craton-bolt` crate name still available (first publish only)?
5. Have the `docs/BENCHMARKS.md` numbers been updated if a new bench run was done?
6. Is `README.md`'s performance section current?
