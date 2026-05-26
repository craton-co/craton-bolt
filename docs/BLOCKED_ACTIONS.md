# Blocked actions — what an automated coding assistant cannot do for you

This document enumerates the concrete actions a coding assistant (me, the
Claude Code agent operating in this repository) **cannot execute on its own**
and which therefore require a human with the right credentials, hardware,
or social authority. It exists so that "ship Craton Bolt to the world" can be
decomposed into a clean human-side checklist alongside the
already-completed code-side work.

The list is grouped by domain. Each item has:
- **Who can do it** — the role / credential needed.
- **What's needed from me** — what I can prepare so the human action is one-click / one-command.
- **Status** — `prepared` (ready for you to execute), `partial` (some prep done, more needed), or `not started`.

---

## 1. Release & package distribution

### 1.1 Tag a release on GitHub

- **Who can do it:** Anyone with `push` rights to `origin`.
- **What's needed from me:** A clean commit on `main`, an up-to-date `CHANGELOG.md` entry, and a `[v0.2.0]` (or whatever version is being cut) release-notes block. The `docs/rust_cuda/09_v02_handemit_tag.md` document already drafts the v0.2 tag plan.
- **Status:** `partial` — release notes drafted, tag not pushed.
- **Command for you to run:**
  ```bash
  git tag -a v0.2.0 -m "v0.2.0 — hand-emit PTX feature-complete"
  git push origin v0.2.0
  ```

### 1.2 Publish to crates.io

- **Who can do it:** A crates.io account holder with publish rights on the `craton-bolt` crate name.
- **What's needed from me:** `Cargo.toml` `[package]` fields complete, `README.md` rendering verified, `cargo package --list` clean, license fields populated.
- **Status:** `prepared` — `Cargo.toml` already has `description`, `license = "Apache-2.0"`, `repository`, `readme`, `keywords`, `categories`. Run `cargo publish --dry-run` first.
- **Blocker:** Crate name `craton-bolt` may already be taken on crates.io; you'll need to verify or pick an alternative (`craton-bolt-sql`, `craton-bolt-rs`, etc.) before publish.

### 1.3 Verify docs.rs build

- **Who can do it:** Triggered automatically by `cargo publish`; can be manually rebuilt via the docs.rs admin panel for crate owners.
- **What's needed from me:** `cuda-stub` feature working for CUDA-less environments (it does — `cargo build --no-default-features --features cuda-stub` already passes).
- **Status:** `prepared`.

### 1.4 GPG-sign release artifacts

- **Who can do it:** Anyone with a GPG key registered to the maintainer email and the `secret-key` available.
- **What's needed from me:** Nothing — I have no access to private keys and never should.
- **Status:** `not started`.

---

## 2. CI / external services

### 2.1 Set up GitHub Actions secrets

- **Who can do it:** A repository admin.
- **What's needed from me:** Workflow YAML files in `.github/workflows/`; I can write them. Secrets (e.g. `CRATES_IO_TOKEN`, `CODECOV_TOKEN`) must be added through the GitHub web UI.
- **Status:** `partial` — `.github/` directory exists in tree; specific secrets not enumerated until release infra is decided.

### 2.2 Enable Dependabot / Renovate

- **Who can do it:** Repo admin via GitHub Settings → Security → Dependabot.
- **What's needed from me:** `.github/dependabot.yml` if you want to customise update frequency / grouping. I can write that file; the toggle itself is a click.
- **Status:** `not started`.

### 2.3 Connect to Codecov / Coveralls

- **Who can do it:** Account holder on Codecov / Coveralls.
- **What's needed from me:** Coverage-collection step in CI (`cargo llvm-cov` or `cargo tarpaulin`). I can wire the workflow.
- **Status:** `not started`.

### 2.4 Set up `cargo-deny` / `cargo-audit` in CI

- **Who can do it:** Anyone editing CI.
- **What's needed from me:** Workflow YAML + `deny.toml`. I can write both.
- **Status:** `not started` — would take ~30 minutes.

---

## 3. Benchmarking & evaluation

### 3.1 Re-run benches on production-class hardware (V100 / A100 / H100)

- **Who can do it:** Anyone with access to such hardware (cloud GPU rental, university cluster, NVIDIA partner program, employer's infrastructure).
- **What's needed from me:** The benches already exist and are reproducible — `BOLT_BENCH_GPU=1 cargo bench --bench olap_benchmarks`. The numbers in [`BENCHMARKS.md`](BENCHMARKS.md) and [`MARKETING.md`](MARKETING.md) are all from an RTX 2060, which is a **consumer-grade card from 2019**. Numbers on data-center GPUs would be substantially better.
- **Status:** `prepared` — bench harness verified working on the RTX 2060 in-tree.

### 3.2 Capture cross-platform CI bench results

- **Who can do it:** Anyone with GitHub Actions GPU runner access (currently a paid feature) or a self-hosted runner with a GPU.
- **What's needed from me:** Workflow YAML and badge plumbing for README.
- **Status:** `not started`.

### 3.3 Run against TPC-H or TPC-DS at scale-factor ≥ 10

- **Who can do it:** Anyone with disk + memory headroom and either licensed dataset access or a generator that produces compatible data (`duckdb` ships `tpch-gen`).
- **What's needed from me:** Right now Craton Bolt doesn't have all the SQL features TPC-H needs (no JOIN, no ORDER BY, limited DATE handling). I can scope which queries from TPC-H subset are runnable today and add scaffolding for the rest.
- **Status:** `not started` — this is itself a roadmap item, not a one-shot bench.

---

## 4. Community & marketing

### 4.1 Post the launch / announcement

- **Who can do it:** The maintainer, on whatever channels they care about (HN, Lobsters, Reddit r/rust, r/databases, Twitter/X, LinkedIn, Mastodon, This Week in Rust).
- **What's needed from me:** [`docs/MARKETING.md`](MARKETING.md) is the marketing one-pager; I can write platform-specific submissions (300-char HN title, 280-char tweet, 500-word "Show & Tell" Lobsters post) on request.
- **Status:** `partial` — long-form marketing doc done; per-platform variants not drafted.

### 4.2 Submit a talk to a conference

- **Who can do it:** The maintainer. RustConf, FOSDEM, GTC (NVIDIA), Rust Nation, EuroRust, P99 CONF are the natural targets.
- **What's needed from me:** I can draft a CFP abstract, slide outline, and demo script. Submission and presentation are human-only.
- **Status:** `not started`.

### 4.3 Reach out to NVIDIA for Developer Program / partnership

- **Who can do it:** A human with email access and a real identity NVIDIA can verify. NVIDIA's developer program has channels for OSS projects; deeper partnership (e.g. NVIDIA Inception, hardware grants) requires direct outreach with a pitch deck.
- **What's needed from me:** [`docs/MARKETING.md`](MARKETING.md) doubles as the pitch material; I can adapt it into deck-shaped slides.
- **Status:** `partial`.

### 4.4 Engage with the Polars / DuckDB / DataFusion communities

- **Who can do it:** The maintainer (or any contributor willing to attach their name).
- **What's needed:** A blog post or GitHub Discussion that's honest about Craton Bolt's positioning vs each of these projects. Being honest about losses on q1 / q2 / q4 (we are) and clear about wins on q3 / q5 (we are) is the right tone.
- **Status:** `not started`.

### 4.5 Write a "compared to RAPIDS" blog post

- **Who can do it:** Someone with cuDF installed and willing to run the same bench harness against RAPIDS.
- **What's needed from me:** The bench is already DuckDB- and Polars-comparable; adding cuDF as a fourth engine is a ~half-day extension I can do, but **running it** requires a working RAPIDS installation, which is itself non-trivial (conda, glibc constraints, etc.).
- **Status:** `not started`.

---

## 5. Legal & IP

### 5.1 Trademark check on "Craton Bolt" / "craton-bolt"

- **Who can do it:** Anyone via a trademark database (USPTO TESS, EUIPO eSearch, WIPO Global Brand). For thoroughness: a lawyer.
- **What's needed from me:** Nothing — I cannot perform legal research with binding authority.
- **Status:** `not started`. **Likely conflict:** "Craton Bolt" is a common English word; trademark search results will be noisy. Worth doing before any large-scale marketing push.

### 5.2 Set up a CLA (or stick with DCO)

- **Who can do it:** Project maintainer.
- **What's needed from me:** The DCO is already in tree (`DCO` file, sign-off referenced in `CONTRIBUTING.md`). Switching to a CLA would require legal review and contributor agreement.
- **Status:** `prepared` — DCO is in place; CLA is a decision, not a missing artifact.

### 5.3 Third-party-license inventory

- **Who can do it:** Anyone running `cargo deny check licenses`.
- **What's needed from me:** A `deny.toml` allowing the licenses we actually depend on, blocking anything strong-copyleft. I can write that.
- **Status:** `not started` — ~30 minutes of work; recommended before crates.io publish.

---

## 6. Infrastructure & hosting

### 6.1 Set up a project website

- **Who can do it:** Anyone with domain registration access and a static-site host.
- **What's needed from me:** I can write a Mkdocs / Zola / mdBook config and adapt the existing docs/ into a website tree. Domain registration, DNS setup, and hosting bill are human-side.
- **Status:** `not started`.

### 6.2 Set up a Discord / Matrix / Zulip community channel

- **Who can do it:** The maintainer.
- **What's needed from me:** Nothing — chat channels are not artifacts I can manage.
- **Status:** `not started`.

### 6.3 Mirror the repo to GitLab / Codeberg

- **Who can do it:** Anyone with accounts on those services.
- **Status:** `not started`. Low priority.

---

## 7. Hardware / environment-bound testing

### 7.1 Test on Jetson (aarch64)

- **Who can do it:** Anyone with a Jetson dev board.
- **What's needed from me:** The crate *should* work — there's nothing x86-specific in the codebase — but `aarch64-linux` builds against CUDA on Jetson have historically had quirks.
- **Status:** `not started`.

### 7.2 Test against multiple CUDA driver versions

- **Who can do it:** A CI matrix runner or a human with multiple machines.
- **What's needed from me:** I can write a matrix-build workflow; running it requires the runners.
- **Status:** `not started`.

### 7.3 Resolve the CUDA Toolkit v13.2 `cuda.lib` linker issue (Windows)

- **Who can do it:** Me — but it's not a "blocked action", it's a known workaround that the user already knows about. Documenting here for completeness: v13.2's stub `cuda.lib` lacks `__imp_*` symbols MSVC's linker wants. Workaround: set `CUDA_PATH` to v12.6 when running tests.
- **Status:** `prepared` — `build.rs` could be patched to prefer v12.6 over v13.2 when both are present. Not yet done; one-line patch when desired.

---

## 8. Things I deliberately won't do without asking

These aren't "blocked" in the credential sense — they are decisions I should
not make unilaterally:

- **Force-push to `main`.** Even when correct, this is a decision a human owner has to authorise.
- **Delete a non-empty file or directory the user didn't explicitly ask me to remove.**
- **Bypass `--no-verify` on commit hooks.**
- **Bypass GPG signing.**
- **Commit code that introduces a new third-party dependency without surfacing it for review.**
- **Run `cargo publish` even if I had a token, because the act of publishing is a public commitment that should originate from the maintainer.**

If any of these are required, ask explicitly and I'll do them.

---

## What's actually next (human-side)

Realistic ordered list of things a human can do in the next ~hour with the
state of the repository as of this commit:

1. **Skim [`MARKETING.md`](MARKETING.md)** and decide if the positioning matches the maintainer's intent. Edit / regenerate as needed.
2. **Run `cargo build --release` and `cargo test --release --lib`** (with `CUDA_PATH` pointed at v12.6 on Windows) to confirm everything still builds clean on the maintainer's machine.
3. **Read [`docs/rust_cuda/09_v02_handemit_tag.md`](rust_cuda/09_v02_handemit_tag.md)** for the v0.2 tag checklist; verify each item.
4. **Decide** whether to:
   - Cut v0.2.0 now (recommended — the hand-emit PTX path is feature-complete).
   - Hold the tag for one more pass of `cudarc` integration polish.
5. **Trademark search** on "Craton Bolt" before any public launch.
6. **Run benches on production hardware** (cloud-rent an A100 for an hour for ~$2; numbers will be much better than the in-tree RTX 2060 results).
7. **Post** the launch wherever the maintainer prefers, using [`MARKETING.md`](MARKETING.md) as source material.

Items 1, 2, 3 are <30 minutes total. Items 4–7 are the actual launch path.
