# Opt-in env vars

craton-bolt honours several env vars to switch experimental code paths on or
to configure resource limits without recompiling. All defaults are
conservative production-safe choices; nothing in this list needs to be set
for ordinary use.

The vars below were discovered by grepping `std::env::var(...)` across the
crate; the source file column is the call site that actually reads the
variable.

## Quick-start matrix

| Env var                          | Default              | Set to...   | Effect                                          |
| -------------------------------- | -------------------- | ----------- | ----------------------------------------------- |
| `CRATON_BOLT_POOL_MAX_BYTES`     | 512 MiB              | byte count  | Soft cap on total pooled GPU bytes              |
| `CRATON_BOLT_POOL_BUCKET_CAP`    | 16                   | integer     | Per-bucket max pooled blocks                    |
| `CRATON_BOLT_PTX_CACHE_CAP`      | 256                  | integer     | JIT PTX module-cache capacity (FIFO eviction)   |
| `BOLT_POOL_STATS_INTERVAL_SECS`  | 60                   | seconds / 0 | Pool-stats log emit cadence (`0` disables)      |
| `BOLT_POOL_WATCH_INTERVAL_SECS`  | 5                    | seconds     | Background watcher poll cadence                 |
| `BOLT_POOL_WATCH_LOW_WATER_FRAC` | 0.10                 | `(0, 1)`    | Watcher proactive-evict threshold (free/total)  |
| `BOLT_GPU_JOIN_TABLE_CAP_MB`     | driver-detected      | `64..=4096` | Override hash-table byte cap (MiB)              |
| `BOLT_GPU_JOIN_STREAMING_INTERN` | off                  | `1`         | Streaming Utf8 intern for high-cardinality keys |
| `BOLT_BENCH_GPU`                 | off                  | `1`         | Enable GPU paths in `cargo bench`               |
| `BOLT_BENCH_THRESHOLD`           | off                  | `1`         | Enable the Utf8-sort threshold bench            |
| `CUDA_PATH`                      | toolkit-default      | path        | Build-time CUDA toolkit location (build.rs)     |
| `CARGO_FEATURE_CUDA_STUB`        | unset                | `1`         | Build-time: skip CUDA discovery (build.rs)      |

## GPU memory pool

### `CRATON_BOLT_POOL_MAX_BYTES`
- **Default**: `536_870_912` (512 MiB)
- **Type**: positive integer (bytes), parsed as `usize`
- **What**: Soft cap on the sum of pooled (freed-but-not-returned-to-driver)
  device-memory bytes managed by `DeviceMemPool`. Allocations beyond this
  cap evict pooled blocks via the cross-bucket LRU before retrying.
- **When**: Raise on rigs where the working set is larger than 512 MiB and
  the pool's OOM-recovery counter is climbing; lower on shared GPUs where
  you want to bound the engine's resident footprint.
- **Notes**: Read once at `DeviceMemPool::new`; non-integer / zero values
  fall back to the default. Pairs with `CRATON_BOLT_POOL_BUCKET_CAP`.
- **Source**: `src/cuda/mem_pool.rs::read_env_usize` (called from
  `DeviceMemPool::new` around line 523).

### `CRATON_BOLT_POOL_BUCKET_CAP`
- **Default**: `16`
- **Type**: positive integer, parsed as `usize`
- **What**: Hard cap on the number of pooled blocks held in any single
  size-class bucket. Excess blocks are evicted to the driver at `free`
  time.
- **When**: Raise on workloads whose allocation profile is dominated by a
  single size class (e.g. fixed-width column buffers) to reduce churn.
  Lower to bound per-bucket memory if a workload thrashes one class.
- **Notes**: Read once at `DeviceMemPool::new`.
- **Source**: `src/cuda/mem_pool.rs::read_env_usize` (called from
  `DeviceMemPool::new` around line 527).

## Pool observability and watcher

### `BOLT_POOL_STATS_INTERVAL_SECS`
- **Default**: `60`
- **Type**: non-negative integer (seconds), parsed as `u64`
- **What**: Interval between pool-stats log lines emitted from
  `Engine::sql`. A value of `0` disables periodic emission entirely.
- **When**: Set to `0` for benchmark runs that don't want the log noise.
  Lower for live debugging of pool behaviour.
- **Notes**: Frozen at `Engine` construction time. Non-integer values fall
  back to the default.
- **Source**: `src/exec/engine.rs::pool_stats_interval_from_env`
  (env var name constant: `POOL_STATS_ENV`, line 112).

### `BOLT_POOL_WATCH_INTERVAL_SECS`
- **Default**: `5`
- **Type**: positive integer (seconds), parsed as `u64`
- **What**: Poll cadence of the `pool-watcher` background thread that
  calls `cuMemGetInfo_v2` and triggers `evict_above_high_water` when free
  device memory drops below the low-water mark.
- **When**: Only meaningful when the `pool-watcher` Cargo feature is
  enabled. Lower for tighter eviction reaction on memory-tight rigs;
  raise to reduce poll overhead on roomy GPUs.
- **Notes**: `0` or unparseable values fall back to the default.
- **Source**: `src/cuda/mem_pool.rs::pool_watcher::read_interval`
  (line 1583).

### `BOLT_POOL_WATCH_LOW_WATER_FRAC`
- **Default**: `0.10`
- **Type**: float in the open interval `(0, 1)`, parsed as `f64`
- **What**: Free-memory fraction (`free / total`) below which the watcher
  proactively evicts pooled blocks. `0.10` means "evict when less than
  10% of device memory is free".
- **When**: Raise (e.g. `0.20`) on shared GPUs where other processes need
  headroom. Lower (e.g. `0.05`) to defer eviction until truly tight.
- **Notes**: Values outside `(0, 1)` fall back to the default. Only
  meaningful when `pool-watcher` is enabled.
- **Source**: `src/cuda/mem_pool.rs::pool_watcher::read_low_water`
  (line 1592).

## JIT module cache

### `CRATON_BOLT_PTX_CACHE_CAP`
- **Default**: `256`
- **Type**: positive integer, parsed as `usize`
- **What**: Maximum number of compiled PTX modules retained in the JIT
  cache. Eviction is FIFO once the cap is exceeded; entries with live
  `CudaModule` clones stay loaded until the last clone is dropped.
- **When**: Raise on long-running processes that cycle through many
  distinct query shapes and you observe repeat PTXAS compiles. Lower on
  memory-tight devices where unloading sooner is preferable to keeping a
  large hot set.
- **Notes**: Read exactly once on first cache access and frozen for the
  process lifetime. Unset / empty / zero / unparseable values fall back to
  the default.
- **Source**: `src/jit/jit_compiler.rs::ptx_cache_cap`
  (env var name constant: `PTX_CACHE_CAP_ENV`, line 100).

## GPU join

### `BOLT_GPU_JOIN_TABLE_CAP_MB`
- **Default**: unset — driver-detected (64 MiB on cards with < 8 GiB total
  VRAM, 512 MiB on cards with >= 8 GiB)
- **Type**: positive integer (MiB), parsed as `usize`, clamped to
  `[64, 4096]`
- **What**: Overrides the driver-detected hash-table byte cap used by the
  GPU hash-join path. Out-of-range values are clamped (with a `log::warn`)
  to the supported range; unparseable values fall back to the
  driver-detected cap.
- **When**: Raise on cards with abundant VRAM running probe-heavy joins
  whose build side overflows the default 512 MiB cap. Lower on shared GPUs
  to bound per-join allocation.
- **Notes**: Capacity (in slots) is `cap_bytes / 12` on the default SoA
  layout; the AoS path divides by 16 instead.
- **Source**: `src/exec/gpu_join.rs::parse_env_cap`
  (env var name constant: `CAP_ENV_VAR`, line 241).

### `BOLT_GPU_JOIN_STREAMING_INTERN`
- **Default**: off
- **Type**: truthy string. Empty, `0`, or `false` (case-insensitive) is
  treated as off; any other non-empty value enables the path
- **What**: Routes `execute_utf8_inner_join_on_gpu` through
  `intern_utf8_columns_streaming`, which builds a `HashMap<u64, i32>`
  keyed by 64-bit hashes instead of the Stage-3 byte-borrowed `HashMap<&str, i32>`.
  Roughly 5-10x smaller dict footprint, at the cost of host post-verify
  for hash collisions.
- **When**: Enable on high-cardinality Utf8 joins (UUID-shaped keys,
  millions of distinct strings) where the borrowed-`&str` dict dominates
  host cost. Leave off for medium-cardinality joins (<= 100k unique
  strings) where the default path is faster.
- **Source**: `src/exec/gpu_join.rs::streaming_intern_enabled`
  (env var name constant: `STREAMING_INTERN_ENV_VAR`, line 2132).

## Benchmark gates

### `BOLT_BENCH_GPU`
- **Default**: off
- **Type**: must equal exactly `"1"` to enable
- **What**: Gate for the GPU-touching bench groups in
  `benches/olap_benchmarks.rs` and `benches/query_benchmarks.rs`. When
  unset, the bench short-circuits with a log line so `cargo bench` on a
  CUDA-less host still completes.
- **When**: Set to `1` before invoking `cargo bench` on a host with a
  working GPU + CUDA toolkit.
- **Source**: `benches/olap_benchmarks.rs` (line 531) and
  `benches/query_benchmarks.rs` (line 305).

### `BOLT_BENCH_THRESHOLD`
- **Default**: off (treats unset / empty / `"0"` as disabled)
- **Type**: any non-empty value other than `"0"` enables the bench
- **What**: Gate for `benches/utf8_sort_bench.rs`. Off by default because
  each iteration uploads ~50 MB of column data per cardinality bucket.
- **When**: Set when explicitly running the Utf8-sort threshold bench
  (`cargo bench --bench utf8_sort_bench`).
- **Source**: `benches/utf8_sort_bench.rs::bench_enabled` (line 84).

## Build-time

### `CUDA_PATH`
- **Default**: unset — falls back to platform-default toolkit locations
  (e.g. `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*` on
  Windows; highest-version install wins)
- **Type**: filesystem path
- **What**: Build-script (`build.rs`) input. Points to the CUDA toolkit
  root so the linker can find `cuda.lib` / `libcuda.so`.
- **When**: Set when the toolkit lives outside the platform defaults, or
  to pin a specific toolkit version on a host with multiple installs.
- **Notes**: Only consumed during compilation; has no runtime effect.
- **Source**: `build.rs` (line 30).

### `CARGO_FEATURE_CUDA_STUB`
- **Default**: unset (managed by Cargo when `--features cuda-stub` is
  passed)
- **Type**: presence-only (any value enables)
- **What**: Build-script gate that skips CUDA toolkit discovery and
  linker-search injection entirely. Used by docs.rs and CUDA-less CI
  hosts that exercise the host-only crate surface.
- **When**: Not for hand-setting; Cargo sets it automatically when the
  `cuda-stub` feature is selected.
- **Source**: `build.rs` (line 23).

## Not present in this build

The following vars appear in some forward-looking design notes but are
NOT honoured by the current codebase. Setting them has no effect:

- `BOLT_PREFIX_SCAN_ALGO` — only the Hillis-Steele scan is shipped today
  (see `src/jit/prefix_scan.rs`). A Blelloch variant exists but is wired
  in unconditionally; there is no runtime selector.
- `BOLT_HASH_ALGO` — the Robin Hood hashing path lives in
  `src/jit/hash_kernels` but is selected by host policy, not by env var.
- `BOLT_HASH_PROBE_TILED` — the tile-aware SoA probe is always on once
  its host gate fires; no runtime override.
- `BOLT_SORT_USE_GRAPH` — the CUDA-graph bitonic sort is selected by the
  sort orchestrator on size, not by env var.
- `CRATON_DISTINCT_HOST_MAX_ROWS` — the host DISTINCT cap is a compile-
  time constant, not an env var.
- `CRATON_PLAN_CACHE_SIZE` — the plan cache capacity is a compile-time
  constant.

If any of these graduates to a runtime knob, document it here and link
the source file that reads it.
