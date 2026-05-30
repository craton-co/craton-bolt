# Build / Kernels / Feature-Wiring Review — craton-bolt

Scope: `build.rs`, `Cargo.toml`, `kernels/`, `.cargo/config.toml`, `deny.toml`, plus
cross-cutting `#[cfg(feature = ...)]` / `env::var` audit and the CUDA-absent
fallback story.

Overall: the build wiring is unusually careful and well-documented. The CUDA
discovery logic, the docs.rs `cuda-stub` path, the version-sorted toolkit
selection, and the `include_str!` stub trick are all correct. The findings below
are mostly MEDIUM/LOW polish items plus a couple of coherence gaps; nothing is a
hard correctness CRITICAL in the default path.

---

## 1. BUILD CORRECTNESS

### B-1 (MEDIUM) — Default build hard-fails to LINK without a CUDA toolkit; no CPU fallback
- `Cargo.toml:95` `default = []`; `src/cuda/cuda_sys.rs:62-63`
  `#[cfg(not(feature = "cuda-stub"))] #[link(name = "cuda")]`.
- A plain `cargo build` (no features) emits `#[link(name = "cuda")]`. On any host
  without the CUDA toolkit import lib (`cuda.lib` / `libcuda.so`) on the linker
  search path, the **link step fails**. There is no host/CPU execution path — the
  crate is GPU-only by design (`src/lib.rs:3` "JIT-compiled GPU SQL engine"); the
  only CUDA-less build mode is `--features cuda-stub`, and stubs return
  `CUDA_ERROR_STUB` at runtime (every GPU op errors).
- This is a deliberate design, but it is a sharp edge for first-time consumers and
  for `cargo install`/`cargo build` on a fresh machine. `compile` (type-check)
  succeeds CUDA-less; only `build`/`test` link.
- Suggested fix: none required if intentional, but (a) document prominently in
  README that a default build requires the CUDA Toolkit, and (b) consider whether
  `cuda-stub` should be a *named* fallback that build.rs auto-enables when it
  cannot locate any CUDA lib dir, OR emit a `cargo:warning=` from build.rs when no
  link-search path was added so the eventual `LNK`/`ld` error is less mysterious.
  Today build.rs silently adds no `-L` and the failure surfaces only at link.

### B-2 (MEDIUM) — `cudarc` feature does not gate the `#[link(name = "cuda")]` block
- `src/cuda/cuda_sys.rs:62` gates the extern link block only on
  `not(feature = "cuda-stub")`; `Cargo.toml:109` `cudarc = ["dep:cudarc"]`.
- `cargo build --features cudarc` (without `cuda-stub`) still emits the hand-rolled
  `#[link(name = "cuda")]` *and* pulls cudarc. cudarc 0.13 with only the `driver`
  feature is dlopen-based (no static link), so there is no symbol/link *conflict*,
  but the build still hard-requires the `cuda` import lib even though the stated
  intent of `cudarc` is to route driver calls through cudarc. The combination
  `--features cudarc,cuda-stub` is the only CUDA-less cudarc build, and there the
  hand-rolled path is stubbed while cudarc would dlopen at runtime — coherent but
  subtle.
- Suggested fix: confirm the intended matrix. If `cudarc` is meant to *replace* the
  hand-rolled FFI, the link block should be `cfg(all(not(cuda-stub), not(cudarc)))`
  and cudarc should own linkage. If cudarc is purely additive for now, add a one-
  line comment at `cuda_sys.rs:62` stating that the link block is intentionally
  active under `cudarc` too.

### B-3 (LOW) — build.rs only adds `-L` search paths; never emits `rustc-link-lib`
- `build.rs:44/50/93/113` etc. emit `cargo:rustc-link-search=native=...` only. The
  actual `-l cuda` comes from the `#[link]` attribute in source. This is correct
  and intentional, but it means build.rs does **zero** verification that
  `cuda.lib`/`libcuda.so` actually exists in the discovered dir — it only checks
  the *directory* exists (`lib_path.exists()`). A toolkit install missing the
  driver import lib (rare, but happens with runtime-only redistributables) yields a
  search path that contains no `cuda` lib and a confusing linker error.
- Suggested fix: optionally probe for the concrete lib file before adding the
  search path, or emit `cargo:warning=` if the dir exists but the lib file does
  not. Low priority.

### B-4 (LOW) — `rerun-if-changed` coverage is good but misses `kernels/rust-toolchain.toml` in the OFF path
- `build.rs:160-162` correctly declares rerun-if-changed for `kernels/src`,
  `kernels/Cargo.toml`, `kernels/rust-toolchain.toml` — but only inside
  `compile_rust_cuda_kernels()` under `#[cfg(feature = "rust-cuda")]`. In the OFF
  variant (`build.rs:173-186`) none are declared, which is correct (kernels aren't
  consulted), and `build.rs` itself is covered at line 3. No bug; noting that the
  rerun triggers are feature-correct. The env triggers for `CARGO_FEATURE_*`
  (lines 4-5) and `CUDA_PATH` (line 28) are present and correct.

### B-5 (LOW) — Windows toolkit discovery hardcodes `C:\Program Files\...` and `lib\x64`
- `build.rs:84` literal `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA`;
  `build.rs:42/91` hardcode `lib\x64`.
- This is the standard NVIDIA layout, but it ignores localized/relocated Program
  Files (`%ProgramFiles%`, non-C: installs, `ProgramW6432`). `CUDA_PATH` (set by
  the official installer) is checked first (line 39) and covers the normal case, so
  this fallback only matters when `CUDA_PATH` is unset, which is uncommon.
- Suggested fix: read `std::env::var("ProgramFiles")` (and `ProgramW6432`) instead
  of the literal `C:\Program Files`. Low priority given `CUDA_PATH` precedence.

### B-6 (PASS) — Version sort, docs.rs, platform split are correct
- `parse_cuda_version` (build.rs:73-83) correctly parses `vMAJOR.MINOR` numerically
  and `Reverse`-sorts so v12.6 > v12.4 > v9.0 (the lexical-sort bug called out in
  the V-16 comment is genuinely avoided). Unparseable dirs fold to `(0,0)` and
  don't panic. Good.
- `[package.metadata.docs.rs] features = ["cuda-stub"]` (Cargo.toml:194-196) plus
  the early `return` on `CARGO_FEATURE_CUDA_STUB` (build.rs:24) means a docs.rs
  build skips all CUDA discovery and the `#[link]` block — docs.rs will build
  cleanly without CUDA. Verified coherent. `rustdoc-args = ["--cfg","docsrs"]` is
  set but I found no `#[cfg(docsrs)]` usage in src (no doc_cfg gating) — harmless,
  just currently unused.
- Linux `stubs/` search-path handling (build.rs:51-53, 100-115) is correct for
  driverless CI/docs builders.

### B-7 (PASS) — External-compiler invocation safety
- The only external compiler invocation is `cuda_builder` under
  `--features rust-cuda` (build.rs:166-170), which is OFF by default. The V-4
  comment (build.rs:131-149) honestly documents that this downloads + runs an
  unverified LLVM/NVVM toolchain at build time with no pinned checksum, and
  restricts it to egress-controlled CI. This is the correct disclosure; the residual
  risk is real but accepted and gated. The `CUDA_PATH` trust model (V-16 comment,
  build.rs:32-38) is also correctly reasoned (trusted build-env input).
- `harden_windows_dir` (src/jit/disk_cache.rs:637-659) spawns `icacls` with args
  passed directly (no shell), best-effort, output suppressed — safe.

---

## 2. KERNELS SPIKE

### K-1 (MEDIUM) — `kernels/` is an explicitly-labeled spike, not production; correctly excluded from publish
- `kernels/Cargo.toml:18-23` `version = "0.0.1"`, `publish = false`,
  description "Wave A spike". `kernels/src/lib.rs` implements exactly ONE kernel
  (`bolt_partition`) as a 1-for-1 rewrite of the hand-emit equivalent.
- Excluded from the published crate via `Cargo.toml:15` package `exclude`
  (`kernels/*`) AND `[workspace] exclude = ["kernels", ...]` (Cargo.toml:36). Both
  are present (belt-and-suspenders) and correct — the published crate will not ship
  `kernels/`, and `cargo build` won't descend into the nvptx64 crate.
- Consequence to be aware of: because `kernels/` is excluded from the package, a
  consumer who downloads craton-bolt from crates.io and enables `--features
  rust-cuda` will **fail** — `build.rs:157` `manifest.join("kernels")` won't exist
  in the unpacked crate. `rust-cuda` is therefore effectively a *source-tree-only*
  feature. This is defensible (it's experimental + needs a special toolchain) but
  should be documented as "git checkout only" so a crates.io user doesn't hit a
  confusing `cuda_builder failed` error. Suggested fix: note this in the
  `rust-cuda` feature comment (Cargo.toml:110-115).

### K-2 (MEDIUM) — `feature(register_attr)` in kernels may be stale against the pinned nightly
- `kernels/src/lib.rs:24` `#![cfg_attr(target_arch = "nvptx64", feature(register_attr))]`.
- `register_attr` was an unstable rustc feature that was **removed** from the
  compiler years ago (folded into `register_tool` / tool attributes). Whether it
  still exists depends entirely on the pinned `nightly-2026-04-02`
  (kernels/rust-toolchain.toml:13). The comment claims `cuda_std`'s `#[kernel]`
  requires it. I cannot compile nvptx64 in this environment to verify, so this is
  flagged as **verify-before-relying**: build `kernels/` against the pinned nightly
  and confirm `register_attr` is still accepted. If the rust-cuda fork uses a
  patched rustc the feature gate may differ. Given `rust-cuda` is OFF by default and
  spike-only, this does not affect shipped builds — but it is the single most
  likely thing to break the `rust-cuda` path.
- Suggested fix: add a CI smoke job (even allow-failure) that actually runs
  `cargo build --features rust-cuda` on the pinned toolchain so this rots loudly,
  not silently.

### K-3 (LOW) — `cuda_std = "0.2.2"` (Feb-2022 crates.io cut) pinned against a 2026 nightly
- `kernels/Cargo.toml:36`. The crate's own comment (lines 29-35) acknowledges the
  0.2.2 cut may be too old for the pinned nightly and that the documented fallback
  is to switch to a git rev. This is honest, but it means the spike is in a known-
  fragile state (old device-std crate + very new compiler). Combined with K-2, the
  `rust-cuda` path should be treated as unproven until a green build is
  demonstrated. No action needed for the default crate.

### K-4 (PASS) — Toolchain pinning mechanics are correct
- `kernels/rust-toolchain.toml` correctly pins channel + components, applies ONLY
  inside `kernels/` (cuda_builder invokes cargo there), and deliberately omits the
  nvptx64 target component because the codegen backend supplies its own target
  spec. The host crate stays on stable (rust-version 1.74). The dual
  `crate-type = ["cdylib","rlib"]` and the `cfg(target_arch="nvptx64")` no_std gate
  are correctly reasoned so host `cargo check` still sees a std constants-only lib.

---

## 3. DEPENDENCY HYGIENE

### D-1 (MEDIUM) — Many duplicate transitive versions, all from dev-deps; `multiple-versions = "warn"` so non-blocking
- `Cargo.lock` has duplicates: `arrow` 53.4.1 + 54.2.1, `ahash` 0.7 + 0.8,
  `hashbrown` 0.12/0.14/0.15/0.17, `syn` 1 + 2, `bitflags` 1 + 2, `rand` 0.8 + 0.9,
  `getrandom` 0.2 + 0.3, `windows-sys` 0.59 + 0.61, plus rand_chacha/rand_core/
  semver/strum_macros/heck.
- Source: dev-dependencies only — `polars 0.42` (pulls arrow 54 / its own arrow
  stack), `duckdb 1.2 (bundled)`, `proptest`. The library runtime graph
  (`arrow 53`, sqlparser, dashmap, etc.) is clean. `deny.toml:12-13` sets
  `all-features=false, no-default-features=false`, so the *checked* graph is the
  host lib graph; `[bans] multiple-versions = "warn"` (deny.toml:52) means even the
  dupes that are visible won't fail CI. Acceptable, but the duplication is large and
  driven entirely by heavy benchmarking dev-deps.
- Suggested fix: none strictly required. If you want the dev-graph clean too,
  consider gating duckdb/polars behind an optional `bench` feature so a normal
  `cargo test` doesn't drag two Arrow stacks. Worth a note that the published-crate
  dependency footprint is unaffected.

### D-2 (LOW) — `deny.toml` advisories/bans are reasonable but carry an unresolved TODO
- `deny.toml:73-74` `unmaintained = "deny"`, `unsound = "deny"` — strict, good.
  `yanked = "warn"`. `[bans] wildcards = "deny"` good. `[sources]` locks to
  crates.io with empty `allow-git` — good supply-chain posture.
- `deny.toml:85` `TODO: confirm RUSTSEC id for xz 0.1.0 before ignoring it`. The
  rust-cuda build tree pulls unmaintained `xz 0.1.0`; the config deliberately does
  NOT fabricate a RUSTSEC id and instead relies on the all-features cargo-deny CI
  job being `continue-on-error`. This is a defensible choice but leaves the
  rust-cuda graph permanently un-gated for advisories. Acceptable while rust-cuda is
  a spike; revisit if it graduates. The license allow-list (deny.toml:24-36) is
  broad but standard; `OpenSSL` + `ring` clarify block (40-47) is correct.

### D-3 (LOW) — `[graph] exclude = []` comment says kernels are linted separately "if/when re-included"
- `deny.toml:14-17` — correct that the nvptx64 kernels crate isn't in the cargo-deny
  graph (it's not a workspace member). No action; just confirms kernels deps are
  unscanned, consistent with D-2.

---

## 4. STUBS / GAPS / EXPERIMENTAL FLAGS

### S-1 (LOW) — Several experimental/escape-hatch features all correctly default OFF
- `Cargo.toml:94-135`: `cuda-stub`, `cudarc` (Stage-1 spike), `rust-cuda`
  (experimental), `pool-sharded` (Stage-3 escape hatch), `pool-watcher` (Stage-4).
  `default = []` pulls none of them — verified no `default` feature drags in GPU or
  experimental code. Each is documented with rationale. The env-var knobs referenced
  in comments (`BOLT_POOL_WATCH_INTERVAL_SECS` default 5,
  `BOLT_POOL_WATCH_LOW_WATER_FRAC` default 0.1) match the code
  (src/cuda/mem_pool.rs:2042/2051). No stray experimental flag left on.

### S-2 (LOW) — `rust-cuda` PTX "no-op variant" mapping is a documented gap, not a bug
- `src/jit/partition_kernel.rs:198-218`: under `rust-cuda`, all four kernel-variant
  entry points (`_for_n_rows`, `_global_atomics`, `_shmem_staging`) collapse to the
  single embedded `partition.ptx`. The shmem-staging variant has no rust-cuda
  counterpart (Wave B). This is documented and safe (falls back to global-atomics),
  but means enabling `rust-cuda` silently downgrades the shmem path. Fine for a
  spike; flag if rust-cuda ever becomes default.

### S-3 (PASS) — `include_str!` stub pattern is correct
- build.rs OFF-variant (173-186) writes an empty `partition.ptx` so the
  `include_str!(concat!(env!("OUT_DIR"), "/partition.ptx"))` at
  partition_kernel.rs:176 resolves at parse time even though it's never read in
  default builds (the consumer is itself `#[cfg(feature="rust-cuda")]`). The
  `if !ptx_out.exists()` guard avoids clobbering a real artifact. The runtime empty-
  check (partition_kernel.rs:178-188) is a reasonable defensive guard. Correct.

### S-4 (LOW) — `rustdoc-args = ["--cfg","docsrs"]` set but no `docsrs` cfg consumed
- `Cargo.toml:196`. No `#[cfg(docsrs)]` / `#![cfg_attr(docsrs, feature(doc_cfg))]`
  found in `src/`. Harmless dead config; either add `doc_cfg` feature-badge gating
  on the feature-gated public items or drop the arg.

---

## Verification notes
- Confirmed `default = []` (no GPU pulled by default).
- Confirmed docs.rs metadata enables `cuda-stub` and build.rs early-returns on it →
  docs.rs builds without CUDA. (Could not run an actual docs.rs build here.)
- Confirmed the default (non-stub) build emits `#[link(name = "cuda")]` →
  link-time CUDA requirement (B-1). Type-check (`cargo check`) succeeds CUDA-less.
- Confirmed Cargo.lock duplicates originate from dev-deps (polars/duckdb/proptest),
  not the runtime lib graph.
- Could NOT verify: that `kernels/` actually compiles under the pinned
  nightly-2026-04-02 (no nvptx64 toolchain here) — K-2/K-3 are flagged as
  verify-before-relying, not confirmed breakage.

## Top priorities
1. B-1 — make the CUDA-less link failure discoverable (README + build.rs warning).
2. B-2 — decide/document whether `cudarc` should gate the hand-rolled `#[link]`.
3. K-1 — document that `--features rust-cuda` is git-checkout-only (kernels/ is
   excluded from the published crate).
4. K-2 — add an allow-failure CI job that actually builds `--features rust-cuda`
   so the spike rots loudly.
