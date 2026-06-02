# Opt-in env vars

craton-bolt honours several env vars to switch experimental code paths on or
to configure resource limits without recompiling. All defaults are
conservative production-safe choices; nothing in this list needs to be set
for ordinary use.

The vars below were discovered by grepping `std::env::var(...)` across the
crate; the source file column is the call site that actually reads the
variable. This list aims to track every runtime/build var the crate reads,
but the codebase moves quickly — treat it as best-effort, and re-grep
`std::env::var` / `env::var_os` if you need the ground truth for a given
release. (Pure path-resolution lookups like `HOME` / `LOCALAPPDATA` /
`USERPROFILE` / `XDG_CACHE_HOME`, read only to compute the platform-default
PTX-cache directory, are not configuration knobs and are omitted.)

## Quick-start matrix

| Env var                          | Default              | Set to...   | Effect                                          |
| -------------------------------- | -------------------- | ----------- | ----------------------------------------------- |
| `CRATON_BOLT_POOL_MAX_BYTES`     | 512 MiB              | byte count  | Soft cap on total pooled GPU bytes              |
| `CRATON_BOLT_POOL_BUCKET_CAP`    | 16                   | integer     | Per-bucket max pooled blocks                    |
| `CRATON_BOLT_PTX_CACHE_CAP`      | 256                  | integer     | In-process JIT PTX module-cache capacity (FIFO) |
| `CRATON_BOLT_PTX_CACHE_MAX_BYTES`  | 64 MiB             | byte count / `0` | Disk PTX-cache total-bytes cap (`0` disables) |
| `CRATON_BOLT_PTX_CACHE_MAX_ENTRIES`| 4096               | integer / `0` | Disk PTX-cache entry-count cap (`0` disables)  |
| `CRATON_DISTINCT_HOST_MAX_ROWS`  | 10_000_000           | integer > 0 | Host-side DISTINCT input row cap                |
| `CRATON_SETOP_HOST_MAX_ROWS`     | 10_000_000           | integer > 0 | Host-side `UNION`/`EXCEPT`/`INTERSECT` row cap  |
| `CRATON_PLAN_CACHE_SIZE`         | 64                   | integer > 0 | SQL→LogicalPlan parse-cache capacity (FIFO)     |
| `CRATON_MAX_SQL_BYTES`           | 1 MiB                | integer > 0 | Pre-parse cap on SQL input length (bytes)       |
| `CRATON_MAX_SQL_TOKENS`          | 100_000              | integer > 0 | Pre-parse cap on SQL token count                |
| `CRATON_MAX_RECURSIVE_ITERATIONS`| 1000                 | integer > 0 | `WITH RECURSIVE` fixpoint iteration cap         |
| `CRATON_MAX_APPLY_ROWS`          | 100_000              | integer > 0 | LATERAL/correlated-apply left-row cap           |
| `CRATON_VALUES_MAX_ROWS`         | 1_000_000            | integer > 0 | `VALUES` literal row cap                         |
| `CRATON_GENERATE_SERIES_MAX_ROWS`| 10_000_000           | integer > 0 | `generate_series` output row cap                |
| `BOLT_POOL_STATS_INTERVAL_SECS`  | 60                   | seconds / 0 | Pool-stats log emit cadence (`0` disables)      |
| `BOLT_POOL_WATCH_INTERVAL_SECS`  | 5                    | seconds     | Background watcher poll cadence                 |
| `BOLT_POOL_WATCH_LOW_WATER_FRAC` | 0.10                 | `(0, 1)`    | Watcher proactive-evict threshold (free/total)  |
| `BOLT_GPU_JOIN_TABLE_CAP_MB`     | driver-detected      | `64..=4096` | Override hash-table byte cap (MiB)              |
| `BOLT_GPU_JOIN_STREAMING_INTERN` | off                  | `1`         | Streaming Utf8 intern for high-cardinality keys |
| `BOLT_PTX_CACHE_DIR`             | unset (disabled)     | dir path    | Opt-in disk-backed PTX cache root (v0.6 / M6)   |
| `BOLT_GPU_SORT`                  | off                  | `1`         | Opt into the GPU radix-sort path for `ORDER BY` |
| `BOLT_GPU_DISTINCT`              | off                  | `1`/`true`/`yes` | Opt into the GPU sort-based `DISTINCT` path |
| `BOLT_GPU_STRING`                | off                  | `1`/`true`/`yes` | Opt into the (host-validated-only) GPU string device kernels |
| `BOLT_GPU_WINDOW`                | off                  | `1`         | Opt into the GPU window-function path           |
| `BOLT_PREFIX_SCAN_ALGO`          | Hillis-Steele        | `blelloch` / `lookback` | Select the GPU prefix-scan kernel   |
| `BOLT_HASH_ALGO`                 | linear-probe         | `robin_hood` / `rh` | Select the GROUP BY keys hash kernel    |
| `BOLT_HASH_PROBE_TILED`          | off                  | `1`         | Opt into the tiled SoA hash-join probe kernel   |
| `BOLT_SORT_USE_GRAPH`            | off                  | `1`         | Opt into CUDA-graph capture for bitonic sort    |
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

## GPU sort

### `BOLT_GPU_SORT`
- **Default**: off (treats unset / anything other than exactly `"1"` as
  disabled; the value is trimmed before the comparison)
- **Type**: must equal exactly `"1"` to enable — `"true"` / `"yes"` / `"on"`
  are deliberately **not** accepted so the gate stays unambiguous
- **What**: Opts the `ORDER BY` executor into the GPU radix-sort path
  (v0.7). When set, the executor *may* route a sort through the radix kernel
  for supported key dtypes (`Int32` / `Int64`, ASC or DESC, including
  multi-key); when unset (the default) the historical bitonic / host sort
  paths run instead. Nullable key columns and unsupported dtypes
  (`Float*` / `Bool` / `Utf8`) always fall back regardless of this var.
- **When**: Enable to exercise or benchmark the radix path on large
  single- or multi-key integer `ORDER BY`s. Left off by default because the
  bitonic / host paths are the bake-tested steady-state until the radix path
  has more production mileage.
- **Notes**: Latched lazily on first read into a process-wide atomic, so the
  value is effectively frozen for the process lifetime once a sort runs. The
  dtype-support check is consulted before the env var, so an unsupported sort
  never even reads it.
- **Source**: `src/jit/sort_kernel_radix.rs` (env var name constant
  `BOLT_GPU_SORT_ENV`, line 150); dispatch gate in `src/exec/sort.rs`
  (`try_gpu_sort_radix`).

## JIT module cache

### `BOLT_PTX_CACHE_DIR`
- **Default**: unset (disk-backed cache disabled — in-process cache only)
- **Type**: path to a writable directory
- **What**: Enables the optional disk-backed PTX cache (v0.6 / M6). On a
  miss in the in-process module cache the engine reads
  `<dir>/<entry>-<hash>.ptx` from disk before re-running codegen; on a
  disk miss it writes the freshly-generated PTX back to disk for the
  next process. The codegen pipeline is deterministic so reuse is
  byte-identical.
- **When**: Set on benchmark harnesses, CLI tools, serverless workers,
  and any other context where the engine is constructed and torn down
  per request — those processes never benefit from the in-process cache
  alone and pay full codegen on every invocation.
- **Path conventions**: Pick any writable directory. Convenient
  platform defaults are documented in `jit::disk_cache::platform_default_dir`
  (`~/.cache/craton-bolt/ptx/` on Linux, `~/Library/Caches/craton-bolt/ptx/`
  on macOS, `%LOCALAPPDATA%\craton-bolt\ptx\` on Windows) — pass one of
  these as the env var value to opt in to the conventional location.
- **Notes**: An in-engine `Engine::Builder::persistent_cache(path)` hook
  overrides this env var when set (see `jit::set_disk_ptx_cache_dir`).
  Writes are atomic (tempfile + rename); read failures fall back to the
  codegen path silently — a corrupt cache entry never produces a wrong
  result.
- **Source**: `src/jit/disk_cache.rs` (env var name constant:
  `DISK_PTX_CACHE_ENV`).

### `CRATON_BOLT_PTX_CACHE_MAX_BYTES`
- **Default**: `67_108_864` (64 MiB)
- **Type**: non-negative integer (bytes), parsed as `u64`; `0` disables the
  byte cap (the entry-count cap still applies)
- **What**: Total-bytes cap on the **disk-backed** PTX cache directory (the
  one enabled by `BOLT_PTX_CACHE_DIR`). After each store, `enforce_bounds`
  scans the committed `*.ptx` files and, if their combined size exceeds this
  cap, evicts least-recently-modified entries (LRU by mtime) until the cache
  is back under both this cap and the entry-count cap. Tempfiles still being
  written (`*.ptx.tmp.*`) are skipped so eviction never races a `store`.
- **When**: Raise on long-lived cache directories that legitimately hold many
  large kernels; lower to bound the on-disk footprint on space-tight hosts.
- **Notes**: Re-read on each `enforce_bounds` call (cheap env lookup), so it
  takes effect without a process restart. Eviction is best-effort: a file that
  fails to delete (e.g. permission denied) is left in place and not
  double-counted. The naming mirrors the GPU pool's `CRATON_BOLT_POOL_MAX_BYTES`.
- **Source**: `src/jit/disk_cache.rs::enforce_bounds` (env var name constant:
  `DISK_PTX_CACHE_MAX_BYTES_ENV`, line 116; default constant
  `DEFAULT_MAX_CACHE_BYTES`, line 126).

### `CRATON_BOLT_PTX_CACHE_MAX_ENTRIES`
- **Default**: `4096`
- **Type**: non-negative integer, parsed as `u64`; `0` disables the
  entry-count cap (the byte cap still applies)
- **What**: Entry-count cap on the **disk-backed** PTX cache directory — a
  second, independent bound (alongside the byte cap) so a flood of tiny
  entries can't blow up the directory inode count while staying under the byte
  cap. Eviction is the same LRU-by-mtime pass in `enforce_bounds`.
- **When**: Raise on directories that cache many distinct kernel shapes; lower
  to keep the file count small.
- **Notes**: Setting both this and `CRATON_BOLT_PTX_CACHE_MAX_BYTES` to `0`
  disables disk-cache eviction entirely (unbounded growth — not recommended).
- **Source**: `src/jit/disk_cache.rs::enforce_bounds` (env var name constant:
  `DISK_PTX_CACHE_MAX_ENTRIES_ENV`, line 121; default constant
  `DEFAULT_MAX_CACHE_ENTRIES`, line 131).

## GPU DISTINCT and window paths

### `BOLT_GPU_DISTINCT`
- **Default**: off
- **Type**: truthy string — `1`, `true`, or `yes` (case-insensitive, trimmed)
  enable; anything else (including unset) is off
- **What**: Opts `DISTINCT` into the GPU sort-based dedup path for a single
  fixed-width primitive key (`Int32` / `Int64` / `Float32` / `Float64`). Utf8
  and wide multi-key shapes always fall back to the host path regardless of
  this var, as does any input below the device sort's own row threshold.
- **When**: Enable to exercise or benchmark the device `DISTINCT` path. Left
  off by default so the host path stays the production default until the
  device round-trip has soak time on real hardware. Mirrors the `BOLT_GPU_SORT`
  gate convention.
- **Source**: `src/exec/distinct.rs::gpu_distinct_enabled` (line 419).

### `BOLT_GPU_STRING`
- **Default**: off
- **Type**: truthy string — `1`, `true`, or `yes` (case-insensitive, trimmed)
  enable; anything else (including unset) is off
- **What**: Single gate for **every GPU string device path**: the per-row
  `LIKE` / `NOT LIKE` / `ILIKE` matcher (`StringLikeFilter` /
  `compile_like_match_kernel`) and the `UPPER` / `LOWER` / `CONCAT` /
  `SUBSTRING` / `TRIM` two-pass producers in `src/exec/string_project.rs`. When
  off (the default) those operations take the **host** code path, which is the
  correctness path. The device kernels are **host-validated only** — they have
  never been executed on GPU hardware as of v0.7.0 (CI builds with no CUDA
  device), so the gate exists purely so a hardware bring-up can opt the device
  kernels in for validation without editing code.
- **When**: Enable only on a GPU host doing string-kernel bring-up /
  validation. Leave off for ordinary use. Mirrors the `BOLT_GPU_SORT` /
  `BOLT_GPU_DISTINCT` gate convention.
- **Source**: `src/exec/string_like.rs::gpu_string_enabled` (env var name
  constant `BOLT_GPU_STRING_ENV`, line 60); re-exported (with
  `gpu_string_enabled`) from `src/exec/string_project.rs`.

### `BOLT_GPU_WINDOW`
- **Default**: off
- **Type**: must equal exactly `"1"` to enable; unset or anything else keeps
  the host window path
- **What**: Opts window-function execution (`ROW_NUMBER` / `RANK` /
  `DENSE_RANK` / running `COUNT` / running `SUM`) into the GPU path on
  supported key/aggregate dtypes. Unresolved or unsupported column shapes
  decline cleanly back to the host evaluator.
- **When**: Enable to exercise or benchmark the device window path. Off by
  default because device behavior is unverifiable in CI without a GPU. Mirrors
  the `BOLT_GPU_SORT` gate convention.
- **Source**: `src/exec/window.rs::try_execute_window_gpu` (env var name
  constant `BOLT_GPU_WINDOW_ENV`, line 1037; gate at line 1419).

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

## Internal / unstable kernel selectors

These vars are **read at runtime** and switch between alternative GPU
kernels. They are wired up for shake-out and benchmarking of newer kernel
variants whose defaults have not yet flipped; treat them as
**internal/unstable — they may change semantics or be removed** without a
deprecation cycle. The default (var unset) is the bake-tested steady-state
path in every case.

### `BOLT_PREFIX_SCAN_ALGO`
- **Default**: Hillis-Steele (var unset, or any unrecognised value, or
  `hillis` / `hillis-steele`)
- **Type**: case-insensitive string; accepted values `blelloch`,
  `lookback`, `hillis`, `hillis-steele`
- **What**: Selects the GPU prefix-scan kernel used by the compaction
  path. `blelloch` routes through the O(n)-work upsweep/downsweep kernel;
  `lookback` routes through the single-pass decoupled-lookback kernel
  (`SCAN_KERNEL_ENTRY_LOOKBACK`), which allocates an extra `partial_status`
  buffer and returns global exclusive prefixes directly. Anything else
  uses the default Hillis-Steele scan.
- **Notes**: Read on every scan dispatch (a cheap env lookup), so it can be
  changed without a process restart for ad-hoc benchmarking. The
  Hillis-Steele default is intentional while the alternatives are in
  shake-out.
- **Source**: `src/exec/gpu_compact.rs::prefix_scan_algo_selection`
  (line 919).

### `BOLT_HASH_ALGO`
- **Default**: classic linear-probe keys kernel (var unset or any value
  other than `robin_hood` / `rh`)
- **Type**: case-insensitive string; `robin_hood` or `rh` selects the
  Robin Hood variant
- **What**: Selects the GROUP BY keys hash kernel. The Robin Hood and
  linear-probe kernels share an identical 4-parameter ABI; only the entry
  point (and module-cache spec id) differs, so the two variants are cached
  separately.
- **Source**: `src/exec/groupby.rs::launch_keys_kernel` (env read at
  line 661).

### `BOLT_HASH_PROBE_TILED`
- **Default**: off (var unset, empty, `0`, or `false` case-insensitive)
- **Type**: truthy string — any non-empty value other than `0` / `false`
  (case-insensitive) enables the path
- **What**: Opts the GPU hash-join probe into the 2-way-unrolled tiled SoA
  kernel (`PROBE_KERNEL_TILED_ENTRY`) instead of the default probe kernel.
  The tiled kernel has an identical nine-parameter ABI, so opting in only
  switches which entry point `launch_probe_kernel` resolves at module load;
  block size, grid shape, and output sizing are unchanged.
- **Source**: `src/exec/gpu_join.rs::probe_tiled_enabled`
  (env var name constant `PROBE_TILED_ENV_VAR`, line 1004).

### `BOLT_SORT_USE_GRAPH`
- **Default**: off (var unset or anything other than exactly `"1"`;
  `"0"` / `"true"` / garbage all read as off)
- **Type**: must equal exactly `"1"` to enable — the strict comparison
  keeps the gate from tripping on shell quoting or boolean-style strings
- **What**: Opts the bitonic sort into CUDA-graph capture, replaying a
  cached `GraphExecHandle` (keyed in `GRAPH_CACHE`) instead of re-issuing
  the per-substage launches each call. Falls back to ordinary launches when
  off.
- **Source**: `src/exec/gpu_sort.rs::sort_uses_graph`
  (env var name constant `BOLT_SORT_USE_GRAPH_ENV`, line 1722;
  gate consulted at line 1942).

## Query planning and execution limits

### `CRATON_DISTINCT_HOST_MAX_ROWS`
- **Default**: `10_000_000` (ten million rows)
- **Type**: positive integer, parsed as `usize`; `0` is rejected
- **What**: Upper bound on the number of input rows the host-side
  `DISTINCT` executor (`src/exec/distinct.rs`) will buffer. (Set operations
  have their own independent cap, `CRATON_SETOP_HOST_MAX_ROWS`, below.) Without
  the cap, `SELECT DISTINCT col FROM big_table` on a high-cardinality column
  allocates `n_rows × n_cols × ~24 B` of host RAM with no ceiling — a
  memory-DoS surface on user-controlled inputs. Exceeding the cap produces a
  clean `BoltError::Other(...)` instead of an OOM.
- **When**: Raise on trusted workloads that legitimately dedup more than
  10M rows on the host; lower to bound host memory on shared / hostile
  inputs.
- **Notes**: Parsed once on first DISTINCT call and cached for the process
  lifetime (`OnceLock`). A value of `0` would disable the cap entirely and
  is rejected with a one-time `log::warn!`; empty / unparseable values also
  fall back to the default with a warning.
- **Source**: `src/exec/distinct.rs::parse_distinct_host_max_rows_env`
  (env var name constant `DISTINCT_HOST_MAX_ROWS_ENV`, line 90; default
  constant `DISTINCT_HOST_MAX_ROWS`, line 84).

### `CRATON_PLAN_CACHE_SIZE`
- **Default**: `64` entries
- **Type**: positive integer, parsed as `usize`; `0` falls back to the
  default
- **What**: Capacity of the SQL→`LogicalPlan` parse cache in the SQL
  frontend (`src/plan/sql_frontend.rs`). The cache is keyed by
  `(sql_text, schema_version)` and stores `Arc<LogicalPlan>` with FIFO
  eviction once the cap is exceeded. Sized by default for "tens of
  dashboard tiles".
- **When**: Raise on long-running processes that cycle through more than
  ~64 distinct query texts and observe repeat parses; lower to bound the
  cache footprint.
- **Notes**: Read exactly once on first parse and frozen for the process
  lifetime (`OnceLock`). Empty / zero / unparseable values fall back to the
  default of `64`.
- **Source**: `src/plan/sql_frontend.rs::plan_cache_cap` /
  `parse_plan_cache_cap` (env var name constant `PLAN_CACHE_SIZE_ENV`,
  line 818; default constant `PLAN_CACHE_CAP_DEFAULT`, line 824).

### `CRATON_SETOP_HOST_MAX_ROWS`
- **Default**: `10_000_000` (ten million rows)
- **Type**: positive integer, parsed as `usize`; `0` is rejected
- **What**: Upper bound on the input rows the host-side set-operation
  executor (`src/exec/setops.rs` — `UNION` / `EXCEPT` / `INTERSECT`, with or
  without `ALL`) will buffer. Same unbounded-growth DoS concern as the
  DISTINCT cap; exceeding it is a clean error rather than an OOM.
- **When**: Raise on trusted workloads that legitimately combine more than
  10M rows on the host; lower to bound host memory on shared / hostile inputs.
- **Notes**: Latched once per process (`OnceLock`). A value of `0` would
  disable the cap and is rejected with a one-time `log::warn!`; empty /
  unparseable values fall back to the default with a warning. Mirrors
  `CRATON_DISTINCT_HOST_MAX_ROWS`.
- **Source**: `src/exec/setops.rs::parse_setop_host_max_rows_env` (env var
  name constant `SETOP_HOST_MAX_ROWS_ENV`, line 77; default constant
  `SETOP_HOST_MAX_ROWS`, line 70).

### `CRATON_MAX_SQL_BYTES`
- **Default**: `1_048_576` (1 MiB)
- **Type**: positive integer (bytes), parsed as `usize`; `0` / empty /
  unparseable fall back to the default
- **What**: Pre-parse denial-of-service guard. The SQL frontend rejects any
  input longer than this many bytes *before* it reaches the `sqlparser`
  parser, so an adversarially huge query can't build an over-large AST whose
  recursive `Drop` would crash the process. Exceeding it is a clean
  `BoltError::Sql(...)`.
- **When**: Raise for legitimately large generated SQL; lower to tighten the
  guard on hostile inputs.
- **Notes**: Read once on first parse and frozen for the process lifetime
  (`OnceLock`). Pairs with `CRATON_MAX_SQL_TOKENS`.
- **Source**: `src/plan/sql_frontend.rs::max_sql_bytes` (env var name
  constant `MAX_SQL_BYTES_ENV`, line 79; default constant
  `MAX_SQL_BYTES_DEFAULT`, line 65).

### `CRATON_MAX_SQL_TOKENS`
- **Default**: `100_000`
- **Type**: positive integer, parsed as `usize`; `0` / empty / unparseable
  fall back to the default
- **What**: The second pre-parse DoS guard: after the cheap byte-length
  check, the frontend runs a flat (non-recursive) tokenizer scan and rejects
  inputs with more than this many tokens. The tokenizer never builds the
  recursive AST, so counting tokens here is safe even for adversarial input.
- **When**: Raise / lower alongside `CRATON_MAX_SQL_BYTES`.
- **Notes**: Read once on first parse and frozen for the process lifetime
  (`OnceLock`).
- **Source**: `src/plan/sql_frontend.rs::max_sql_tokens` (env var name
  constant `MAX_SQL_TOKENS_ENV`, line 83; default constant
  `MAX_SQL_TOKENS_DEFAULT`, line 74).

### `CRATON_MAX_RECURSIVE_ITERATIONS`
- **Default**: `1000`
- **Type**: positive integer, parsed as `usize`; missing / non-integer / `0`
  fall back to the default
- **What**: Caps the number of fixpoint iterations a `WITH RECURSIVE` CTE may
  run before the engine stops with a clean error. The recursive fixpoint is a
  host nested loop that grows the accumulated relation each round; the cap
  bounds runaway or non-terminating recursions.
- **When**: Raise for legitimately deep recursions (long graph walks, deep
  hierarchies); lower to fail fast on pathological queries.
- **Source**: `src/exec/engine.rs::max_recursive_iterations` (env var name
  constant `MAX_RECURSIVE_ITERATIONS_ENV`, line 88; default constant
  `MAX_RECURSIVE_ITERATIONS`, line 82).

### `CRATON_MAX_APPLY_ROWS`
- **Default**: `100_000`
- **Type**: positive integer, parsed as `usize`; missing / non-integer / `0`
  fall back to the default
- **What**: Hard cap on the number of LEFT rows a `LATERAL` / correlated apply
  will drive. The apply is a host nested loop that re-plans and re-runs the
  correlated subquery once per left row (`O(left_rows × subquery)`), so a huge
  left input would spin or OOM; `Engine::execute_lateral_apply` refuses more
  than this many left rows and returns a clean `BoltError`.
- **When**: Raise on trusted workloads with large correlated drivers; lower to
  bound host cost on shared / hostile inputs.
- **Source**: `src/exec/engine.rs::max_apply_left_rows` (env var name constant
  `MAX_APPLY_LEFT_ROWS_ENV`, line 112; default constant `MAX_APPLY_LEFT_ROWS`,
  line 107).

### `CRATON_VALUES_MAX_ROWS`
- **Default**: `1_000_000` (one million rows)
- **Type**: positive integer, parsed as `usize`; `0` / empty / unparseable
  fall back to the default
- **What**: Row cap on an inline `VALUES (...)` row source. Without it a giant
  literal blob (`VALUES (1),(2),...,(10^9)`) would allocate host-side without
  bound. Exceeding the cap is a clean `BoltError::Sql(...)`.
- **When**: Raise for legitimately large literal tables; lower to tighten the
  guard.
- **Notes**: Re-parsed on each use (cheap env lookup, no global latch) so the
  cap stays per-process overridable.
- **Source**: `src/plan/sql_frontend.rs::values_max_rows` (env var name
  constant `VALUES_MAX_ROWS_ENV`, line 1581; default constant
  `VALUES_MAX_ROWS`, line 1577).

### `CRATON_GENERATE_SERIES_MAX_ROWS`
- **Default**: `10_000_000` (ten million rows)
- **Type**: positive integer, parsed as `usize`; `0` / empty / unparseable
  fall back to the default
- **What**: Row cap on the `generate_series(...)` table-valued function. The
  computed row count (via checked arithmetic) is checked against this cap
  *before* any host allocation, so an unbounded series fails cleanly instead
  of attempting a runaway allocation.
- **When**: Raise for legitimately large series; lower to tighten the guard.
- **Notes**: Re-parsed on each use (cheap env lookup, no global latch).
- **Source**: `src/plan/sql_frontend.rs::generate_series_max_rows` (env var
  name constant `GENERATE_SERIES_MAX_ROWS_ENV`, line 2057; default constant
  `GENERATE_SERIES_MAX_ROWS`, line 2053).
