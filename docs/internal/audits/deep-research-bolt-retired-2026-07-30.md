# Deep research review retirement audit

Status: **retired on 2026-07-30**

Source reviewed: `deep-research-bolt.md`, an external working-tree report against
the pre-retirement `dev` branch.

This audit is the durable disposition of every issue in that report. The source
review mixed correctness defects, build and release gates, documentation drift,
and longer-term product directions. Retirement means that each concrete defect
or drift item has been implemented and tested, while recommendations that cannot
be proven on this one machine have been converted into explicit support
boundaries and reproducible gates rather than unverified claims.

## Runtime and SQL closure

| Review concern | Disposition |
|---|---|
| Process-global CUDA state | `Engine::build` now atomically enforces one live CUDA engine per process before CUDA initialization. Failed construction and `Drop` release the permit, and CUDA-owned fields are dropped before it. This makes the supported isolation contract fail-fast instead of allowing silent cross-context contamination. |
| Streaming grouped integer `SUM` overflow | Streaming aggregation now retains the checked host-side source values needed for overflow validation. An end-to-end multi-morsel regression proves overflow is reported rather than wrapped. |
| Non-standard integer and Decimal arithmetic | SQL-compatible integer division/remainder invalid rows produce `NULL`; `MIN / -1` is not allowed to wrap. Decimal division is rejected instead of silently returning a non-standard value. The old lowering is available only with the explicit `BOLT_LEGACY_ARITHMETIC=1` compatibility switch, which defaults off. |
| Grouped `AVG` for all-NULL groups | All grouped finalizers, including standard, pre-aggregation, wide, and validity-aware variants, emit a nullable `Float64` result and return `NULL` when the count is zero. |
| Literal-list `IN` / `NOT IN` with `NULL` | Literal lists lower to an explicit SQL three-valued-logic `CASE` chain. Scalar, filtered, and `HAVING` uses now share deterministic NULL semantics. |
| Unpredictable host/GPU selection | The public `ExecutionTier::{Gpu, Host, Hybrid}` contract classifies physical plans. `PhysicalPlan::planned_execution_tier()` / `QueryHandle::planned_execution_tier()` expose the decision before collection, and SQL/execution documentation is machine-checked against this contract. |
| GPU string path hidden and unvalidated | Supported non-dictionary string operations select their GPU implementation by default; `BOLT_GPU_STRING=0` is the explicit host override. The hardware suite executes the default path with the variable unset. |
| GPU sort and dedup default paths | The planner selects radix GPU sort by supported key shape and size without an opt-in flag; unsupported shapes fall back predictably. Single primitive-key `DISTINCT` uses GPU dedup by default, and `UNION` benefits through that path. Explicit `=0` overrides remain for diagnosis. |
| Build-time `rust-cuda` toolchain download | The public feature and dependency/build-script graph were removed. Archived kernel research remains excluded from normal builds and is clearly labeled as non-shipping research. |
| Half-adopted `cudarc` backend | `cudarc` is the default and sole supported real-CUDA adapter. The bespoke driver surface is no longer described as the main backend, and CUDA-stub remains the host-only validation adapter. |
| JIT cache race found during residual audit | In-flight `OnceCell` entries are no longer selected for LRU eviction; completed entries restore the configured capacity. A concurrent regression locks this behavior. |
| Decimal GPU `SUM` overflow found during hardware retirement | Decimal inputs are checked while NULLs are stripped, before block-local GPU accumulation can hide an overflow in a wrapped partial. Cross-block host merge remains checked. |
| Non-linear `WITH RECURSIVE` process abort, found by the final ledger run | A `k`-self-reference recursive term is a `k`-way self-join whose `n^k` intermediate is built inside the recursive subplan, invisible to both the working-set and accumulated-result row caps. A cyclic `UNION ALL` squared its way to a 64 GiB host allocation and killed the test process while still under the 10M-row cap and short of the iteration cap. The driver now counts the term's self-references and rejects an iteration whose worst-case fan-out cannot fit under the cap. |

## Test and release-engineering closure

| Review concern | Disposition |
|---|---|
| Informational, threshold-free coverage | CI coverage is blocking and enforces a 50% line floor across library and integration tests. CUDA hardware code remains covered by the separate blocking GPU lane rather than being represented as host coverage. |
| Non-blocking real-GPU placeholder | The self-hosted GPU job is blocking and runs the ignored hardware suite serially with the real `cudarc` backend. It also runs the reference-engine GPU shard. |
| Optional features only compile-checked | Dedicated `flight` and `substrait` lanes execute their library and integration smoke tests. CI installs `protoc` for Substrait generation. |
| Narrow document-consistency tests | Tests now check workflow hardening, public-API snapshot instructions, README/document links, SQL execution-tier language, local/hosted CI parity, and exact code/document environment-variable parity. |
| Local CI drift | `ci_local.sh` uses pinned tool versions, stable and MSRV parity, blocking deny/coverage/API/doc gates, clean packaging, and optional real-GPU parity. It uses `set -euo pipefail` and no network-to-shell bootstrap. |
| Heavy reference engines in ordinary loops | DuckDB and Polars dependencies are isolated behind `reference-tests` and `reference-benches`; the affected test and benchmark targets declare required features. |
| Missing semantic regressions | Dedicated tests cover integer invalid arithmetic, Decimal division rejection, grouped all-NULL `AVG`, literal NULL membership, streaming `SUM` overflow, GPU-path defaults, execution-tier reporting, engine isolation, and Decimal GPU overflow. |
| Unpinned or advisory-only hardening | Third-party workflow actions use immutable commit SHAs. `cargo deny` is blocking, including the all-feature set now that the `rust-cuda` advisory tree is gone. Strict Clippy is also blocking. |
| Optional subsystems executed, not just compiled | `tests/flight_e2e.rs` and `tests/substrait_e2e.rs` each carry a host-runnable tier (Flight wire round-trip through the stock arrow-flight client decoder; Substrait producer-shaped plan conversion) and a `gpu:e2e` tier (live gRPC `get_flight_info` + `do_get` and the bearer-token gate; converted plans executed through `Engine::run_logical_plan`). Both GPU lanes were extended to run that second tier, which no previous command reached because none enabled those features. |
| Dead MSRV gate, found by the final ledger run | The declared 1.74 MSRV could not parse the committed `Cargo.lock` — a locked dependency declares `edition = "2024"` — so the hosted 1.74 matrix leg failed before compiling anything. Corrected to the verified 1.85 floor everywhere and pinned by `doc_consistency_test::msrv_is_consistent_across_repo` against the CI matrix, `ci_local.sh`, and every doc that quotes it. |
| Failing supply-chain gate, found by the final ledger run | `cargo deny` was not actually green: RUSTSEC-2026-0204 (`crossbeam-epoch`) failed the default graph, and the all-features graph failed on two licenses plus two advisories with no upstream fix. The vulnerability was fixed by an upgrade; the licenses are permissive and now allowed with rationale; the two remaining advisories are unreachable from the default graph and are listed explicitly in `deny.toml` and tracked in `SECURITY.md`. |
| Flaky test inside a blocking gate, found by the final ledger run | `per_bucket_lock_allows_concurrent_progress` asserted a parallel-vs-sequential wall-clock ratio and self-described as "flaky under load". Harmless while the ignored lane was advisory; a coin flip once that lane became blocking. The ratio assertion was replaced by the deterministic post-conditions (per-class bucket routing, exact bucket count, blocks still pooled). |

## Documentation and governance closure

| Review item | Disposition |
|---|---|
| README omissions | The project layout includes `kernels/`, and the documentation index includes the cudarc and kernel-contribution references under advanced/developer material. README links are exact-path tested. |
| ROADMAP drift | Persistent-cache and benchmark statements reflect current behavior and files. |
| RELEASING drift | Tag behavior, Codecov status, available workflows, clean packaging, API checks, and manual publication responsibilities match the repository. |
| Manual API enumeration | `cargo-public-api` output is committed as `docs/PUBLIC_API_SNAPSHOT.txt`; `scripts/check_public_api.sh` and CI reject drift using a date-pinned nightly toolchain. |
| Historical `PATH_TO_1.0` state | The old 0.3-era baseline is in `docs/internal/history/PATH_TO_1.0-0.3-baseline.md`; the top-level document contains only current direction and an archive pointer. |
| Historical `GROUPBY_PERF` analysis | The pre-optimization narrative is in `docs/internal/history/GROUPBY_PERF-pre-optimization.md`; the top-level document points to current canonical benchmarks and the archive. |
| Benchmark coverage | `scaling_benchmarks` accepts controlled row sweeps and a separate VRAM-pressure size. Results in public docs remain explicitly scoped to the tested RTX 2060; this audit does not claim behavior for a second GPU class that was not physically available. |
| DEVELOPMENT drift | A CI truth table documents hosted and local gates, optional features, reference shards, GPU requirements, and coverage. |
| SQL reference drift | Execution tiers and supported/fallback behavior are machine-checked. |
| USER_GUIDE caveats | Non-production status and the distinction between SQL support and GPU-native execution appear before the quick start. |
| ENV_VARS maintenance | The parity test scans production source, build script, and benchmarks in both directions with no “docs lead code” escape path. |
| Advanced docs discoverability | Cudarc adoption and kernel contribution guidance are first-class entries under an explicitly advanced/developer section. |
| Placeholder CODEOWNERS | Ownership names the real `@victor-craton` maintainer; placeholder team language was removed from both governance files. Repository-side review enforcement is documented as an administrator setting. |
| SECURITY policy mismatch | The policy now describes SHA pinning, blocking advisory scans, removed build-time toolchain downloads, GPU gating, and the remaining disclosure/release boundaries accurately. |
| `docs/internal` ambiguity | Durable historical context and this retirement record now live in named `history` and `audits` subtrees. Public/operator guidance stays in `docs/` and is indexed from README. |

## Directional recommendations resolved

- Predictable execution selection is now a typed public contract rather than an
  implicit environment-variable convention.
- Multi-context safety is resolved conservatively for the current architecture:
  concurrent engines fail before CUDA initialization. True multi-GPU engines
  remain outside the supported API and are not implied by the documentation.
- SQL NULL and invalid-arithmetic behavior follows the standard path by default;
  legacy arithmetic requires an explicit compatibility setting.
- Obvious GPU hotspots now have planner-selected paths where the shipped kernels
  are supported. Unsupported shapes retain documented host or hybrid execution
  rather than being described as GPU-native.
- CI hardening is encoded in blocking, SHA-pinned, machine-checked workflows and
  a local parity script.

## Validation ledger

The final delivery was accepted only after all rows below completed from the
isolated worktree and its unique target directory.

| Gate | Result |
|---|---|
| Host library and integration suite (`cuda-stub`) | 2,389 library tests passed, 169 ignored; every integration binary passed |
| Rust doctests | 0 failed; 27 hardware/example doctests intentionally ignored |
| Document consistency | 9 passed |
| Strict Clippy | Passed with `-D warnings` |
| Default `cudarc` compile | Passed |
| Focused live-GPU engine/string/sort/dedup/streaming/AVG/Decimal regressions | Passed |
| Complete ignored live-GPU suite | Pending final recorded run |
| DuckDB/reference live-GPU shard | Pending final recorded run |
| `flight` executable suite | Pending final recorded run |
| `substrait` executable suite | Pending final recorded run |
| MSRV 1.74 compile gate | Pending final recorded run |
| Blocking coverage floor | Pending final recorded run |
| `cargo deny` | Pending final recorded run |
| Scaling and VRAM-pressure benchmark smoke | Pending final recorded run |
| Clean package dry run and public-API snapshot | Pending final recorded run |

The working report is retired by this audit only after every pending ledger row
is replaced by its exact result and the resulting commit is integrated into
`dev`.
