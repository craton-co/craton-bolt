# Rust-CUDA Toolchain Audit (Craton Patina 0.3 milestone)

Date of audit: 2026-05-25. All cited facts checked against GitHub HEAD,
crates.io JSON API, and the `rust-gpu.github.io` blog.

## 1. Ecosystem inventory

| Crate | crates.io | GitHub HEAD | Notes |
|---|---|---|---|
| `Rust-CUDA` (umbrella) | n/a | `main` last push **2026-04-29** | 5.2k stars, 86 open issues, not archived. Rebooted under the **Rust-GPU** org. |
| `cust` | **0.3.2** (2022-02-16) | path-dep in workspace, edition 2024 | crates.io is stale; HEAD is far ahead. MSRV not declared. |
| `cust_core` | 0.1.1 (2022-02-12) | path-dep | shared CPU/GPU types. |
| `cust_derive` | 0.2.0 (2022-02-12) | path-dep | macros for `cust`. |
| `cust_raw` | 0.11.3 (2021-12-05) | path-dep, `driver` feature | raw FFI; HEAD has been re-bindgen'd against CUDA 12/13. |
| `rustc_codegen_nvvm` | **0.3.0** (2022-02-07) | path-dep, workspace `=0.3.0` | NVVM-IR (libNVVM) backend; LLVM 7 *or* LLVM 19 via the new `llvm19` feature (added 2026-04-29, commit `2dec3ab`). |
| `cuda_std` | 0.2.2 (2022-02-07) | path-dep | device-side intrinsics: shared memory, atomics, warp shuffle, `i128`, etc. Active fixes April 2026 (warp shuffle, fence scopes). |
| `cuda_builder` | n/a on crates.io | workspace `=0.3.0` | `build.rs` helper; the supported integration entry point. |
| `nvvm` | n/a | path-dep | wraps libNVVM; defines the `NvvmArch` enum. |

**Crates.io is dormant.** Every published version is from late 2021 / early 2022.
The only realistic way to consume Rust-CUDA today is a **git dependency** pinned
to a SHA in `Rust-GPU/Rust-CUDA`. A fresh crates.io release has been on the
maintainers' radar since Jan 2025 but is not announced in any of the five status
posts through Aug 2025 (no 2026 status post yet).

**Relationship to `cudarc`.** `cudarc` (0.19.7, **2026-05-15**) is a *host-side
only* safe driver-API wrapper. It does not compile Rust to PTX, it only loads
PTX/cubin and launches kernels. It is orthogonal to `rustc_codegen_nvvm` and
complementary to (not a replacement for) `cust_raw`. Craton Patina already uses
cudarc-style host plumbing; the question here is purely about the
**device-side** compiler.

**Maintainer activity.** Christian Legnitto (`@LegNeato`) leads the reboot. The
top-of-tree commit log (April 2026) shows multiple contributors landing real
fixes: ABI alignment bugs (`#388`), LLVM-19 backend (`#2dec3ab`), warp shuffle
PTX-scope correctness (`@Snehal Reddy`), cuDNN-9 mappings (`@Charry Wu`), and
nightly bumps (`@LegNeato`). This is not a one-person hobby project, but it is
small (≈4–6 active contributors).

## 2. CUDA toolkit & sm_70 compatibility

- The guide requires **CUDA 12.0 or later**; CUDA 12 and CUDA 13 are both
  exercised in CI containers (issue `#386`, May 2026 — `ubuntu24-cuda13` image
  with LLVM 19.1.7). CUDA 11.x support is dropped on `main`.
- **Craton Patina's CUDA 12.6 lands cleanly inside the supported window.**
- The `NvvmArch` enum on HEAD includes `Compute50` through `Compute121` plus
  `*a` / `*f` family variants. **`Compute70` (sm_70 / Volta) is listed and is
  not deprecated.** The default `arch` is gated by the `llvm19` cargo feature:
  pre-llvm19 defaults to a pre-Blackwell baseline; with `llvm19` it defaults
  forward to Blackwell. Either way `.arch(NvvmArch::Compute70)` is the
  supported way to lock to Volta.
- The March 2025 update flagged CUDA 12 support as "experimental, limited
  testing." Subsequent status posts (May, July, Aug 2025) and the April 2026
  llvm19 work suggest it has matured, but **there is no GPU-in-CI**, so
  regressions are caught by users, not by upstream.

## 3. Rust nightly pinning

`rust-toolchain.toml` on HEAD pins **`nightly-2026-04-02`** with components
`clippy, llvm-tools-preview, rust-src, rustc-dev, rustfmt, rust-analyzer`.
The `rustc-dev` requirement is non-negotiable: `rustc_codegen_nvvm` is a
codegen backend and links against the in-tree rustc crates, so the nightly
must match the one the backend was built against. Nightly bumps happen
roughly every 4–8 weeks based on the commit log (e.g. `nightly-2025-06-23`
→ `nightly-2026-04-02`).

**Avoiding nightly contamination of Craton Patina proper.** The nightly pin is per
package — `rust-toolchain.toml` files are tracked per directory. The supported
pattern (see `examples/vecadd/`) is:

```
craton-patina/
  Cargo.toml                  # workspace, stable rustc
  src/                        # current Craton Patina, stable
  kernels/                    # NEW: separate workspace member
    Cargo.toml                # depends on cuda_std (git)
    rust-toolchain.toml       # nightly-2026-04-02
    src/lib.rs                # #[kernel] fns
  build.rs                    # calls cuda_builder to emit PTX
```

`cuda_builder` shells out to a sub-`cargo` with the kernel crate's nightly
toolchain, so the parent crate can stay on stable. Craton Patina's `src/jit/` PTX
emitter can be kept untouched during a parallel migration.

## 4. Build-system integration

The canonical pattern from `examples/vecadd/build.rs`:

```rust
use cuda_builder::CudaBuilder;
CudaBuilder::new("kernels")        // path to the kernel crate
    .copy_to(out_dir.join("kernels.ptx"))
    .arch(NvvmArch::Compute70)     // sm_70 for Craton Patina's target
    .build()
    .unwrap();
```

Plus `cargo:rerun-if-changed=kernels` to retrigger on kernel edits and
optional `RUST_CUDA_DUMP_FINAL_MODULE` / `RUST_CUDA_EMIT_LLVM_IR` env vars
for debugging. The PTX artifact is then loaded with cudarc/cust at runtime
exactly like Craton Patina's current hand-emitted PTX.

There is **no Bazel / Buck / non-cargo integration story**. Everything is
`build.rs` + cargo workspaces.

## 5. 2026 outlook

Positive signals:

- Active development through HEAD (last commit 6 days before this audit).
- LLVM 19 backend landed April 2026 — meaningful modernization, not just
  janitorial fixes.
- ABI / alignment / fence-scope bugs are being found and fixed, which means
  someone is actually compiling real kernels with this.
- Compatible with our exact CUDA 12.6 + sm_70 targets.

Risks:

- **Zero crates.io releases since Feb 2022.** Anyone using this is pinned
  to a git SHA, with all the supply-chain and reproducibility implications.
- **No GPU CI upstream.** Correctness regressions are user-discovered.
- ~4–6 active maintainers, all unpaid. Bus factor is low.
- Backend is tied to a specific rustc nightly; that nightly bumps every
  ~2 months and historically has broken consumers each time (see issue
  `#291`, "Update to nightly-2025-08-11", still open and being rolled into
  the 2026-04-02 bump).
- NVIDIA shipped CUDA-Oxide 0.1 on 2026-05-09 — a separate official
  Rust-to-PTX compiler. It is currently pre-beta but may consolidate
  community mindshare away from `rustc_codegen_nvvm` over the next year.

## Bottom line

**Don't bet 0.3 on rust-cuda; bet 0.4.** The technical pieces Craton Patina
needs — sm_70, CUDA 12.6, shared memory, atomics, warp intrinsics — all
exist and work today, and the `build.rs` + isolated-nightly-workspace
pattern keeps the blast radius contained. But shipping 0.3 against a
git-SHA dependency with no upstream GPU CI, a 2-month nightly churn cycle,
and a freshly-landed LLVM-19 backend is taking on integration risk we
don't need while we're still landing groupby-tier2 and hash-join work.
Recommended posture: build a **throwaway proof-of-concept** rewrite of
one kernel (e.g. `shmem_sum_kernel`) in the 0.3 cycle as a separate
unmerged branch, track the project's next crates.io release as the
trigger, and plan the production migration for 0.4 when (a) a tagged
release exists, (b) NVIDIA's CUDA-Oxide direction is clearer, and (c)
the LLVM-19 path has had two more nightly bumps without breaking.

## Sources

- [Rust-GPU/Rust-CUDA on GitHub](https://github.com/Rust-GPU/Rust-CUDA)
- [Rebooting the Rust CUDA project (2025-01-27)](https://rust-gpu.github.io/blog/2025/01/27/rust-cuda-reboot/)
- [Rust CUDA March 2025 update](https://rust-gpu.github.io/blog/2025/03/18/rust-cuda-update/)
- [Rust CUDA August 2025 update](https://rust-gpu.github.io/blog/2025/08/11/rust-cuda-update/)
- [Getting Started — The Rust CUDA Guide](https://rust-gpu.github.io/rust-cuda/guide/getting_started.html)
- [Raising the baseline for the nvptx64-nvidia-cuda target (Rust Blog, 2026-05-01)](https://blog.rust-lang.org/2026/05/01/nvptx-baseline-update/)
- [cust on crates.io](https://crates.io/crates/cust)
- [rustc_codegen_nvvm on crates.io](https://crates.io/crates/rustc_codegen_nvvm)
- [cudarc on crates.io](https://crates.io/crates/cudarc)
- [Phoronix: Rust-CUDA Project Restarted](https://www.phoronix.com/news/Rust-CUDA-Project-Reboot)
