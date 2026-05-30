# Performance backlog — GPU-gated items (NOT landed blind)

These perf items from the review require a GPU **and** a benchmark harness to validate.
They change device behavior or emitted PTX, so they cannot be verified under the
`cuda-stub` + host-oracle setup used for all the dev-branch work. Landing them without
measurement would (a) risk silent correctness regressions on real hardware and (b) only
lock in unverified PTX behind the golden snapshots. They are deferred ON PURPOSE.

Each should be implemented on a GPU runner, validated by `cargo test --features cudarc
-- --ignored` (incl. `diff_duckdb`) + `cargo bench` (benches/) before/after, and gated by
`compute-sanitizer`.

## 1. AVG fused sum+count reduce  (exec_groupby.md, MEDIUM)
Today AVG runs two reduce passes (sum, then count) and divides on host. Fuse into one
kernel that accumulates `(sum, count)` per group in a single probe pass — halves the
reduce passes and the D2H.
- Files: `groupby_tier2_avg_exec.rs`, `groupby_tier2_twokey_avg_exec.rs`, a new fused
  `jit` reduce kernel (or extend `partition_reduce_kernel_multi`).
- Guard: the AVG correctness tests D2 added (currently `#[ignore]`) must pass on GPU;
  add a new golden snapshot for the fused kernel.
- Risk: medium — changes the reduce ABI; verify NaN/overflow/empty-group parity vs the
  current two-pass result on hardware.

## 2. Device-compact before the 52 MiB D2H  (exec_groupby.md, MEDIUM)
`execute_tier2_sum` always copies the full `NUM_PARTITIONS × BLOCK_GROUPS × 13 B = 52 MiB`
slot buffer to host, then walks it for populated slots (see PA for the host-walk side).
Compact populated slots ON DEVICE (stream-compaction over the `set` flag, reusing
`gpu_compact`) and D2H only the compacted region — turns a fixed 52 MiB transfer into
`O(populated)`.
- Files: `groupby_tier2_orchestrator.rs` (+ multi/twokey orchestrators), reuse
  `exec::gpu_compact`.
- Guard: tier2 e2e GPU tests; measure PCIe bytes via the metrics surface before/after.
- Risk: medium — interacts with the partition layout; the host walk (PA) becomes a
  thin post-pass.

## 3. Adaptive spin back-off  (jit.md, LOW)
The fixed 32 ns spin back-off appears in 11 partition_reduce kernels. An exponential /
occupancy-aware back-off may reduce contention — but it is workload-dependent and needs
on-hardware contention measurement to know it helps rather than hurts.
- Files: `partition_reduce_kernel_spill_common.rs` (`emit_spin_backoff`) — would change
  emitted PTX, so the 40 golden snapshots must be regenerated and re-reviewed.
- Risk: low correctness, but **unmeasurable without a GPU** and easy to pessimize. Do
  ONLY with a contention benchmark.

## 4. Pinned-memory pool  (cuda.md — enabled by A2's deferred-free)
A2 landed the event-based deferred-free pool, which is the prerequisite for a pinned
host-memory pool (faster, async H2D/D2H). `PinnedHostBuffer::Drop` also still uses the
conservative blanket-sync (A2 left it, documented) — a pinned pending-free list would
remove that stall.
- Files: `cuda/mem_pool.rs`, `cuda/buffer.rs`, `cuda/async_copy.rs`.
- Risk: medium — host page-locked lifetime; needs a shutdown drain to avoid leaking
  pinned pages. Validate with `compute-sanitizer` + transfer benchmarks.

## Landed on dev instead (host-verifiable perf)
- PA: host slot-walk collector optimized (serial fast path + size-gated parallel scan,
  output byte-identical, host-tested).
- PB: `StreamSet` de-duplication (cuda).
- Wave 1–2: bounded `GLOBAL_MODULE_CACHE` eviction; Decimal128 D2H+H2D round-trip
  replaced with device-to-device copy; per-Drop blanket-sync stall removed on the common
  path via deferred-free.
