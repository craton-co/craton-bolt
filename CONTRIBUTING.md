# Contributing to Craton Bolt

Thank you for considering a contribution. This file covers what you need to know to get a useful change into the tree.

## Ground rules

- **Be civil.** See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
- **Keep changes focused.** One PR per logical change. If you find yourself reaching for "and while I'm here…", open a second PR.
- **Add tests.** Even if the build machine can't run them. PTX-shape assertions and `#[ignore]`-gated live-GPU tests both count.
- **Document your decisions.** Inline `//` comments explaining *why*, doc comments (`///`) explaining *what*. Module-level `//!` docs for non-trivial files.

## What kinds of contributions are welcome

| Kind                       | Notes                                                                              |
|----------------------------|------------------------------------------------------------------------------------|
| Bug reports                | Include a minimal reproducer + the failing query / API call.                       |
| New SQL features           | See [`docs/SQL_REFERENCE.md`](docs/SQL_REFERENCE.md) for the current subset.       |
| New aggregate / GROUP BY paths | The executor selection in `src/exec/engine.rs` shows the existing dispatch.     |
| Performance improvements   | Benchmarks under `benches/` are the contract. Include before/after numbers.        |
| Documentation              | Yes, please. Especially examples for the SQL reference and architecture deep-dive. |
| Tests                      | We always want more.                                                               |
| CUDA driver / PTX loading  | `src/cuda/` (the cudarc-backed driver layer, pools) or `src/jit/jit_compiler.rs` (PTX → module via `cuModuleLoadDataEx`; there is no NVRTC). |

If you want to take on a big change (a new join algorithm, a new dtype, a new optimizer pass), open an issue first to talk about the shape before writing the code.

## Getting set up

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the full build and test workflow. The 30-second version:

```bash
git clone <your-fork>
cd craton-bolt
# Host-only (no CUDA toolkit needed) — the CI configuration:
cargo test --lib --tests --no-default-features --features cuda-stub
# Default (cudarc) build — needs the CUDA toolkit on the linker path:
cargo test
# Live-GPU suite — needs an NVIDIA GPU; serialised, with the real pool backend:
BOLT_BENCH_GPU=1 cargo test -- --ignored --test-threads=1
```

## Code style

- `rustfmt` defaults. Run `cargo fmt` before committing.
- `clippy` lints clean in the CI configuration: `cargo clippy --lib --tests --features cuda-stub --no-default-features -- -D warnings`. Pragmatic exceptions allowed if you explain.
- **No `unwrap()` or `panic!()` in library code.** Errors must flow through `BoltResult<T>` / `BoltError`. `unwrap` is fine in `#[cfg(test)]` modules and in benchmarks where the harness can't surface a Result.
- **No `unsafe` outside the documented FFI boundaries.** The CUDA driver layer in `src/cuda/` and the raw `cuLaunchKernel` parameter assembly in the `src/exec/` executors are the sanctioned exceptions. New `unsafe` blocks need a `// SAFETY:` comment explaining the invariant.
- **No new dependencies without discussion.** The current dep set is deliberate. If you need a new crate, justify it in the PR description.

## Module ownership

- Adding a new `pub mod foo` requires adding it to the corresponding `mod.rs`.
- New executors go in `src/exec/`. The `Engine::execute` dispatch in `src/exec/engine.rs` is the integration point. Query-path changes (parse → rewrite → optimize → hooks → subqueries → lower) belong in the shared `QueryCompiler` in `src/exec/engine_query.rs`, never in one entry point only.
- Every new source file needs a `//!` summary line: it feeds the generated module inventory in `docs/ARCHITECTURE.md`.
- New PTX kernels go in `src/jit/`. The convention is: emitters return `BoltResult<String>` and entry-point names are `pub const` symbols.
- New CUDA layer types (alternative dictionaries, buffer flavours) go in `src/cuda/`.
- Plan / IR work goes in `src/plan/`.

## Tests

| Flavour | Marker | Runs in | Covers |
|---|---|---|---|
| Pure-host unit tests | `#[test]` | every lane (`cuda-stub`) | host algorithms, planning, the `QueryCompiler` stages, frontend classification |
| PTX-shape / golden tests | `#[test]` (`tests/ptx_golden_*.rs`, `insta` snapshots) | every lane | emitted PTX text — codegen regressions without a GPU |
| Document-consistency tests | `#[test]` (`tests/doc_consistency_test.rs`) | every lane | version / MSRV / env-var parity, generated module inventory + CI matrix (`BOLT_REGENERATE_DOCS=1` refreshes) |
| Live-GPU tests | `#[test] #[ignore = "gpu:..."]` | self-hosted GPU lane, `BOLT_BENCH_GPU=1 ... -- --ignored --test-threads=1` | end-to-end pipelines on real hardware |
| Reference / property tests | `reference-tests` feature (`diff_duckdb*`, `sql_proptest`) | dedicated shard | result parity with DuckDB, randomized SQL |

PTX-shape assertions complement device evidence; they never replace it — a change to a device path needs a live-GPU test.

Don't gate behaviour behind feature flags unless there's no other way — the `#[ignore]` pattern is preferred for live-GPU work because it keeps the test discoverable.

## PR process

1. Fork, branch, commit.
2. `cargo fmt`, the CI clippy command above, and `cargo test --lib --tests --no-default-features --features cuda-stub` (or `bash ci_local.sh` for the full rehearsal).
3. If you touched a device path (`src/jit/`, `src/exec/`, `src/cuda/`), run the live-GPU suite on a CUDA host — or say in the PR that you could not.
4. If you changed the public API, regenerate `docs/PUBLIC_API_SNAPSHOT.txt` (`bash scripts/check_public_api.sh` shows the drift) and update `docs/API_SURFACE.md`.
5. Open a PR with:
   - A one-sentence summary in the title.
   - A description that says *what* and *why*, not just *how*.
   - A list of any new public API.
   - A note on test coverage.

## DCO sign-off

Every commit must include a `Signed-off-by:` trailer attesting to the
[Developer Certificate of Origin v1.1](DCO). Use `git commit -s` to add
it automatically — the trailer is just:

```
Signed-off-by: Your Name <you@example.com>
```

The DCO is a lightweight alternative to a CLA: by signing off you certify
that you have the right to submit the work under the project's Apache-2.0
license. See the [`DCO`](DCO) file at the repo root for the full text.

## What to work on

See [`ROADMAP.md`](ROADMAP.md) for milestones — the sections toward 1.0 and
the "1.0 — public API freeze" section list the work still planned. Items in
its "Known limitations (not bugs)" section and in
[`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) are intentional gaps — flag in
your PR if you want to tackle one.

## Licensing of contributions

Craton Bolt is licensed under the [Apache License, Version 2.0](LICENSE). By
submitting a pull request, issue, or patch, you confirm that you have the
right to license the contribution under those terms and that you agree to
do so. No separate Contributor License Agreement (CLA) is required —
the standard inbound = outbound model applies: contributions are licensed
to the project under the same Apache-2.0 terms that the project itself
ships under.

If you're contributing source code, please include the standard SPDX
header as the first line of any new `.rs` file:

```rust
// SPDX-License-Identifier: Apache-2.0
```

Followed by a blank line and then the existing module-level `//!` docs or
`use` statements. The header lets license-scanning tools recognise the
file's terms without parsing the LICENSE file.

If your contribution incorporates code from a third-party project, make
sure that project's license is compatible with Apache-2.0 and that any
required attribution lands in [`NOTICE`](NOTICE).

## Reporting bugs

Open an issue with:

- The exact query / API call that fails.
- The expected vs actual behaviour.
- `cargo --version`, `rustc --version`, `nvidia-smi` output, target triple.
- A minimal reproducer (ideally a single test function).

Thanks for contributing.
