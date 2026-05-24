# Contributing to Javelin

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
| CUDA driver / NVRTC tweaks | Anything in `src/cuda/cuda_sys.rs` or `src/jit/jit_compiler.rs`.                    |

If you want to take on a big change (a new join algorithm, a new dtype, a new optimizer pass), open an issue first to talk about the shape before writing the code.

## Getting set up

See [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) for the full build and test workflow. The 30-second version:

```bash
git clone <your-fork>
cd javelin
cargo check --lib --tests --benches      # works without CUDA installed
cargo test                                # needs cuda.lib on the linker path
cargo test -- --ignored                   # needs an actual GPU
```

## Code style

- `rustfmt` defaults. Run `cargo fmt` before committing.
- `clippy` lints clean. Run `cargo clippy --all-targets`. Pragmatic exceptions allowed if you explain.
- **No `unwrap()` or `panic!()` in library code.** Errors must flow through `JavelinResult<T>` / `JavelinError`. `unwrap` is fine in `#[cfg(test)]` modules and in benchmarks where the harness can't surface a Result.
- **No `unsafe` outside the documented FFI boundaries.** The CUDA driver calls in `src/cuda/cuda_sys.rs` and the raw `cuLaunchKernel` parameter assembly in `src/exec/engine.rs` are the only sanctioned exceptions. New `unsafe` blocks need a `// SAFETY:` comment explaining the invariant.
- **No new dependencies without discussion.** The current dep set is deliberate. If you need a new crate, justify it in the PR description.

## Module ownership

- Adding a new `pub mod foo` requires adding it to the corresponding `mod.rs`.
- New executors go in `src/exec/`. The `Engine::execute` dispatch in `src/exec/engine.rs` is the integration point.
- New PTX kernels go in `src/jit/`. The convention is: emitters return `JavelinResult<String>` and entry-point names are `pub const` symbols.
- New CUDA layer types (alternative dictionaries, buffer flavours) go in `src/cuda/`.
- Plan / IR work goes in `src/plan/`.

## Tests

There are three flavours:

1. **Pure-host unit tests** (`#[test]`, no `#[ignore]`). Always run. Cover host-side algorithms (packing, scan, dedup, expr eval).
2. **PTX-shape tests** (`#[test]`, no `#[ignore]`). Always run. Emit a PTX string and assert on its content (`contains("setp.lt.f32")`, etc.). The right tool for the JIT layer because they don't need a GPU but catch regressions in the codegen.
3. **Live-GPU tests** (`#[test] #[ignore]`). Skipped by default. Run with `cargo test -- --ignored` on a CUDA host. Cover end-to-end pipelines.

Don't gate behaviour behind feature flags unless there's no other way — the `#[ignore]` pattern is preferred for live-GPU work because it keeps the test discoverable.

## PR process

1. Fork, branch, commit.
2. `cargo fmt && cargo clippy --all-targets && cargo check --lib --tests --benches`.
3. If you touched anything in `src/jit/` or `src/exec/`, run `cargo test` (link errors against `cuda.lib` are expected on hosts without CUDA — say so in the PR description).
4. Open a PR with:
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

See [`ROADMAP.md`](ROADMAP.md) for milestones planned for 0.2 and 1.0.
Items in the "Known limitations" section under "0.1.x (current)" are
intentional gaps — flag in your PR if you want to tackle one.

## Licensing of contributions

Javelin is licensed under the [Apache License, Version 2.0](LICENSE). By
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
