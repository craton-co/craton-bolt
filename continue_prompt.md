# Handoff: GPU validation of the security/perf remediation on `dev`

You are taking over a remediation job on **craton-bolt** (a JIT-compiled GPU SQL engine).
A previous agent, working on a machine **without a GPU**, implemented and committed a large
batch of security fixes, code-review cleanups, performance optimizations, and a dedup refactor
to the **`dev`** branch. Everything compiles on both the `cuda-stub` and `cudarc` feature sets
and the full host test suite passes (**1220 lib tests, 0 failed**), but **none of the
GPU-execution paths have been run on real hardware.**

**Your machine has a GPU. Your job:** validate the changes on hardware, fix anything that
fails, run the perf benchmarks to confirm the optimizations are real wins (and not
regressions), and — only if you choose to — implement the one deferred item (P4). Do **not**
trust "it compiled"; several changes alter device memory management, kernel launch ordering,
and PTX-adjacent codegen where only real execution can prove correctness.

---

## 1. Repository state

- Branch: **`dev`** (this is the integration branch — keep working here or branch off it).
- Base commit (last commit before this work): `6f13e6d`.
- Remediation commits (oldest→newest):
  ```
  6be8631 security: wave 1 (V-1..V-12, V-15, V-16)
  4b6fa71 security: wave 2 (V-13, V-14, V-17)
  4e170fc review: safe cleanups (logging, latent panic, stale comments)
  14d1990 perf: P2, P3, P5, P6, P7 + dispatch audit (P1)
  a138c35 refactor: dedup group-by helpers (D1); document codegen divergence (D2)
  6841d58 perf: P9 filtered-keys round-trip; D3 tier2 scan helper
  2e7e8a1 chore: drop unused AggregateSpec import
  ```
- Review the diff with: `git log --oneline 6f13e6d..HEAD` and
  `git diff 6f13e6d..HEAD -- <file>`.

## 2. Build & test commands

The host-without-GPU agent could only use the stub backend. **You should build the real
backends.** Feature matrix (from `.github/workflows/ci.yml` and `Cargo.toml`):

```sh
# Host stub (no GPU) — what was used so far; should still pass for you:
cargo test  --lib --tests --features cuda-stub --no-default-features
cargo clippy --lib --tests --features cuda-stub --no-default-features

# Pure-Rust cudarc CUDA driver backend (stable Rust, CUDA 12.x):
cargo check --lib --features cudarc --no-default-features
cargo test  --lib --features cudarc --no-default-features      # <-- RUN ON GPU

# Default / linked-CUDA backend (real builds use this — links system CUDA):
cargo build
cargo test                                                     # <-- RUN ON GPU

# Watcher-thread teardown path (needed to validate V-9):
cargo test --features "pool-watcher cudarc" --no-default-features

# rust-cuda (nightly + libNVVM + LLVM; downloads a toolchain — see V-4):
#   off by default, optional; only if you want to exercise kernels/ crate.
```

Notes / gotchas observed on the prior machine:
- **Windows linker:** `.cargo/config.toml` forces `lld-link` (needs LLVM's `lld-link` on PATH).
  Integration tests embed bundled DuckDB and won't link with MSVC's `link.exe`.
- **Incremental-cache corruption:** running multiple `cargo` invocations concurrently produced
  `error: unable to copy ... incremental ... (os error 3)`. If you hit it, prefix with
  `CARGO_INCREMENTAL=0` or `cargo clean -p craton-bolt`.
- Clippy currently emits ~189 **pre-existing** warnings (CI does not gate on `-D warnings`).
  Our changes add **zero** net-new warnings. Don't get lost in the pre-existing noise.

## 3. Highest-priority GPU validation (run a sanitizer)

Use **`compute-sanitizer`** (the successor to `cuda-memcheck`) — it is the single best tool to
catch the memory-safety classes these fixes touch:

```sh
compute-sanitizer --tool memcheck   cargo test  # use-after-free / OOB device access
compute-sanitizer --tool racecheck  cargo test  # shared-mem races (group-by/sort kernels)
compute-sanitizer --tool synccheck  cargo test  # missing __syncthreads
compute-sanitizer --tool initcheck  cargo test  # reads of uninitialized device memory
```

(Run against a GPU-exercising test target, e.g. the integration tests or a small example.)

### Per-finding validation checklist (what to actually exercise)

**Memory-safety fixes — the ones most likely to surface only on hardware:**

- **V-1 / V-2 (stream tracking, pinned-buffer UAF)** — `src/cuda/buffer.rs`,
  `src/cuda/smart_ptrs.rs`, `src/exec/launch.rs`.
  The fix makes `KernelArgs` auto-tag the launch stream and gives `PinnedHostBuffer` a full
  `StreamSet`. **Validate:** run multi-stream workloads (joins/sorts/group-bys that overlap)
  under `memcheck`. Specifically construct a case where a `PinnedHostBuffer` is used as a DMA
  source/dest on more than one stream, then dropped — confirm no UAF into freed pinned pages.
  There is a test override `DROP_FENCE_OVERRIDE` in `buffer.rs` you can use to assert fence
  behavior.

- **V-9 (pool watcher join on Drop)** — `src/cuda/mem_pool.rs`, **`pool-watcher` feature only.**
  Build `--features pool-watcher`, then exercise process teardown (let the `Lazy` pool finalize)
  under `synccheck`/`memcheck`. Confirm the watcher thread is genuinely joined before drain and
  there's no touch of `CUcontext` after `cuCtxDestroy`.

- **V-5 (dictionary key bounds)** — `src/exec/gpu_table.rs`, `src/exec/engine.rs`.
  **Validate:** register a `DictionaryArray<Int32|Int64, Utf8>` whose keys are out of range
  (e.g. via `DictionaryArray::new` with a bad key, or an i64 key > i32::MAX). Should now return
  `BoltError::Type`, not OOB-read / panic. Also confirm well-formed dictionaries still decode
  correctly under `memcheck`.

- **V-13 (LargeUtf8 hash cursor widened to 64-bit)** — `src/jit/hash_join_kernel.rs`.
  **Validate:** JOIN / GROUP BY on a `LargeUtf8` column. Correctness on normal-sized data is
  the must-pass; the >4 GiB-offset case is hard to construct but the logic should now be
  full-width. Confirm i32/`Utf8` path results are unchanged (it was deliberately left 32-bit).

**Performance changes — validate correctness AND benchmark the win:**

- **P2 (join sentinel via `cuMemsetD8(0xFF)`)** — `src/exec/gpu_join.rs`.
  `0xFF` bytes == `u32::MAX` sentinel. **Validate:** run the join test suite under `memcheck`
  and confirm join results match the host/DuckDB reference. The i64 key-table init was
  deliberately left as a host upload (i64::MIN is not byte-replicable). Watch for
  stream-ordering: the memset is enqueued on the same stream the build kernel reads from.

- **P3 (radix sort: async pinned histogram, 1 sync/pass)** — `src/exec/gpu_sort.rs`.
  **Validate:** sort correctness across i32/i64 and large row counts; the change relies on
  same-stream ordering for the offsets H2D feeding the scatter. **Benchmark** the per-pass
  sync reduction (it was 2 syncs/pass → 1).

- **P5 (sharded LRU in the device mem pool, 32 shards)** — `src/cuda/mem_pool.rs`.
  This is the **highest-risk concurrency change.** **Validate:** many-stream allocation churn
  (concurrent frees into distinct size classes). The new global-eviction order is *approximate
  under concurrent races* by design — it can never lose/double-free a block, but a strict-LRU
  stress harness may observe looser ordering. Watch for the theoretical livelock note in
  `lru_pop_global_oldest`. **Benchmark** allocator throughput vs. base `6f13e6d`.

- **P6 (cudarc `device_ref()` returns a borrow, no per-op `Arc::clone`)** —
  `src/cuda/cudarc_backend.rs`, **`cudarc` feature only.** Build `--features cudarc` and run a
  memory-op-heavy workload; confirm lazy device init still works and no lifetime/borrow issue
  at runtime.

- **P7 (codegen `write!`-into-buffer, PTX byte-identical)** — `src/jit/ptx_gen.rs`.
  Low risk: the 420 in-lib jit golden tests + `tests/ptx_golden_tests.rs` (insta snapshots)
  already pin the emitted PTX byte-for-byte and pass. Just run `cargo insta test` /
  `cargo test --test ptx_golden_tests` to be sure.

- **P9 (filtered-keys round-trip eliminated)** — `src/exec/groupby.rs`.
  **Validate:** multi-aggregate GROUP BY where value columns contain NULLs (e.g.
  `SELECT k, SUM(a), AVG(b) FROM t GROUP BY k` with NULLs in a/b). Results must match the
  reference; the key column is now filtered off a cached host copy instead of re-downloaded.

**Refactors — pure regression validation:**

- **D1 (`src/exec/groupby_common.rs`)** — group-by helpers consolidated. Run the full group-by
  test suite on GPU. The canonical `pack_keys` uses `wrapping_shl` (V-17).
- **D3 (`src/exec/groupby_tier2_common.rs`)** — only one scan loop extracted; low risk.

## 4. Run the benchmarks

```sh
cargo bench                 # criterion; HTML reports under target/criterion
```
Compare against base `6f13e6d` (you can `git stash`/checkout to A/B). Confirm P3 (sort), P5
(alloc), P2 (join init) show improvement or at least no regression. The competitive bench vs.
DuckDB lives per `docs/COMPETITIVE_BENCHMARKING.md`.

## 5. The one DEFERRED item — P4 (optional, your call)

**P4: GPU-accelerate Welford (STDDEV/VAR) and Decimal128 aggregation.** These currently run
**host-side** (full key download + host fold):
- Welford host fold: `src/exec/groupby.rs` (~`run_welford_aggregate`, around line 1470) and
  `src/exec/welford.rs` (the correct single-pass + Chan-Golub-LeVeque combine — reuse this math).
- Decimal128 aggregation: `src/exec/aggregate.rs` (~line 533) and the Decimal128 paths in
  `groupby.rs`.

This was **not** done by the prior agent because it requires writing **new GPU kernels** (in
`src/jit/`) plus host launch plumbing, which cannot be verified without a GPU — exactly what you
have. **Caution / precedent:** finding **P1** (fused multi-agg) revealed that a PTX *emitter*
can exist with **zero host callers and no launch driver** — i.e. "the kernel exists" does not
mean it's wired. Check `src/jit/hash_kernels.rs` (`compile_groupby_agg_kernel_multi`,
`AGG_KERNEL_MULTI_ENTRY`) before assuming anything is callable. If you implement P4: add the
kernel + a host launcher, gate it behind eligibility checks, keep the host path as the fallback,
and validate results against the host implementation (which is the correct reference) under
`memcheck`.

While there, **P1** is also still open: the fused multi-agg kernel emitter exists but has no
host launcher (see the corrected `TODO(perf, review L3)` comment in `groupby.rs` ~line 535 for
the exact (a)-(d) checklist of what's missing). Flipping the dispatch is a real perf win if you
build the launcher — same "validate against the N-launch path" discipline applies.

## 6. Working method (what worked well)

- The prior agent fanned out **one Opus sub-agent per file-disjoint finding** (pool of ~9),
  then built/tested between waves. You can do the same for P4/P1 or fix-ups, but on a GPU
  machine **you can finally run the tests** — so prefer: implement → run real GPU tests +
  sanitizer → benchmark → commit, per item.
- Keep commits scoped per finding with the `V-#`/`P#`/`D#` tags for traceability.
- The audit's full finding catalog and severities are in the git history commit messages and
  were derived from a 6-subsystem security audit (cuda/, jit/, exec aggregation, exec pipeline,
  plan/, kernels+build).

## 7. Definition of done for your handoff

1. `cargo test` (default/linked-CUDA) and `cargo test --features cudarc` **green on GPU**.
2. `compute-sanitizer --tool memcheck` clean on a representative GPU test run (esp. the V-1/V-2/
   V-5/V-9 paths and P2/P3/P5).
3. Benchmarks confirm P2/P3/P5 are wins (or documented neutral), no regressions elsewhere.
4. PTX goldens still pass (P7).
5. (Optional) P4 / P1 implemented + validated against host reference, or explicitly re-deferred
   with a note.
6. Summarize results back, noting anything that failed on hardware that passed under stub.
