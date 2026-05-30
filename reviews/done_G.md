# Agent G — remediation report (exec core: module_cache + temporal GPU gates)

Branch `dev`. Only the five files in my lane were edited:
`src/exec/module_cache.rs`, `gpu_table.rs`, `gpu_compact.rs`.
(`gpu_compact_multipass.rs`, `gpu_upload.rs` reviewed — no change needed; see
"No-change files" below.) `engine.rs` / `lib.rs` / `mod.rs` untouched.

---

## 1. MEDIUM — module_cache.rs: bound the unbounded `GLOBAL_MODULE_CACHE` (F1)

### Change
`GLOBAL_MODULE_CACHE` was `Lazy<Mutex<HashMap<Key, CudaModule>>>` — the one
cache family in the file with no eviction (every sibling spec cache is a
FIFO-capped `SpecCache`). A workload that JITs many distinct kernel shapes
(wide GROUP BY tiers, many distinct projection/`spec_id` shapes, multi-GPU via
the `\0dev{N}` key suffix) accumulated `CudaModule` handles for the process
lifetime — a slow host/VRAM cubin leak.

Replaced the raw `HashMap` with a new `struct GlobalModuleCache { by_key:
HashMap<Key, CudaModule>, order: VecDeque<Key>, cap }` carrying the **same FIFO
policy and `cap = 256`** as the string-cache convention
(`KERNELSPEC_CACHE_CAP`). It mirrors `SpecCache` exactly: `by_key` for lookup,
a parallel `order` insertion-log; on insert past the cap we `pop_front()` and
remove the matching map entry. `insert` is idempotent (re-insert of an existing
key returns the incumbent and preserves its FIFO position, so a racing miss
shares one `Arc<CudaModuleInner>`). `get` is a pure read (FIFO, not LRU).

Call sites updated:
- `get_or_build_module` fast-path: `cache.get(&key)` now returns an owned
  `CudaModule` clone.
- `get_or_build_module` insert path: `cache.entry(...).or_insert(...).clone()`
  → `cache.insert((namespace, spec_id), module)`.

### Safe eviction of in-use modules (the brief's hard constraint)
`CudaModule` is `Arc<CudaModuleInner>` and the driver `cuModuleUnload` lives in
`CudaModuleInner`'s `Drop` (verified in `jit::jit_compiler`), which runs only
when the **last** `Arc` is dropped. Evicting an entry drops only *this map's*
`Arc` clone — any caller still using a previously-handed-out module holds its
own clone, so eviction can never unload an in-flight module. This is exactly
the "bounded cache that retains in-use entries" the brief asked for, achieved
with **zero cross-file unload plumbing**. Documented inline on the `static` and
on `GlobalModuleCache::insert`.

### Tests added (`mod global_module_cache_tests`, mirrors `spec_cache_tests`)
- `global_cache_evicts_in_fifo_order` — past-cap insert evicts the oldest key;
  live map never exceeds `cap`.
- `global_cache_insert_is_idempotent_and_preserves_fifo_position`.
- `global_cache_eviction_retains_in_use_module` — a caller's clone stays a
  valid handle after its key is evicted (pins the safety contract).
All three are pure-host (`CudaModule::stub_for_tests()`), no GPU needed.

---

## 2. MEDIUM — temporal/decimal hard errors → graceful host-fallback signal (F11)

### Exact fallback mechanism used: `BoltError::GpuCapacity`
I mirrored the **existing GPU join gate** mechanism, not `Ok(None)`. Reason:
the three sites are inside functions returning `BoltResult<Self>` /
`BoltResult<GatheredCol>` (no `Option` channel), exactly like the GPU join
kernels. `error.rs` documents `BoltError::GpuCapacity(String)` as the **typed
"GPU path declined — retry on host"** marker, and `gpu_join.rs` already emits it
in four places (lines 1371/1498/1751/1915) for the join overshoot path. The
join callers (`try_gpu_inner_join`/`try_gpu_outer_join` in join.rs) map
`Err(GpuCapacity(_))` → `Ok(None)` → host. I converted the temporal/decimal
*hard* errors to that same typed decline so the projection/gather path becomes
type-consistent with the join gates.

### gpu_table.rs (`GpuColumn::upload`, ~:535)
`DataType::Date32 | Timestamp(_,_)` arm: was `BoltError::Type("…not yet
supported")` (fatal, propagated through `ensure_gpu_table` → failed the query).
Now `BoltError::GpuCapacity("GPU upload of temporal column … decline to host …")`.
Decimal128 *upload* already works at this layer (line 478) and is unchanged.
Supported-dtype correctness untouched — only the unsupported arm's error *kind*
changed.

### gpu_compact.rs (`alloc_gathered`, ~:981 / :994)
- `Decimal128(_,_)`: `BoltError::Plan(…)` → `BoltError::GpuCapacity(…)`.
- `Date32 | Timestamp(_,_)`: `BoltError::Other(…)` → `BoltError::GpuCapacity(…)`.
Utf8 gather arm left as-is (it's pre-gated in the engine by `has_utf8_output`
and never reaches here). Supported gather dtypes unchanged.

### Tests added
- gpu_table.rs `temporal_upload_declines_to_host` — `upload` of Date32 /
  Timestamp returns `Err(GpuCapacity(_))`. Pure-host: the temporal arm returns
  before any downcast/device alloc, so an empty `Int32Array` placeholder works
  with no CUDA context.
- gpu_compact.rs `alloc_gathered_declines_unsupported_dtypes_to_host` —
  Decimal128 / Date32 / Timestamp all return `Err(GpuCapacity(_))`. Pure-host
  (error returns before device alloc). Asserts the variant, not just the string.

---

## Wiring required from the orchestrator (engine.rs — NOT my lane)

These three sites now emit `BoltError::GpuCapacity` from the projection/gather
path. Today the engine's `execute` dispatcher (`engine.rs`, `execute_projection`
~:2642 `ensure_gpu_table?` and ~:2969 `compact_columns_on_gpu?`) `?`-propagates
errors — there is **no caller-level catch** that turns `GpuCapacity` into a host
re-run for the projection path (unlike join.rs, which already does this for the
join gates). So for the fallback to actually take effect end-to-end, the
orchestrator agent must add, at the projection dispatch site:

> When `execute_projection` (or its `ensure_gpu_table` / `compact_columns_on_gpu`
> calls) returns `Err(BoltError::GpuCapacity(_))`, re-run the projection on the
> host (materialize_table + host filter/project) instead of failing the query —
> the same `Err(GpuCapacity) → host` mapping `execute_inner_join` /
> `execute_outer_join` already apply to the join kernels.

My change is the necessary precondition (emit the typed, routable decline marker
instead of a fatal `Type`/`Plan`/`Other`); the one-line routing lives in
engine.rs, which is outside my edit boundary. The marker is type-safe to
pattern-match (`matches!(e, BoltError::GpuCapacity(_))`), not string-parsed.

---

## No-change files (in my lane, reviewed)
- `gpu_compact_multipass.rs` — handles only the multi-pass *scan*; gather dtype
  dispatch delegates to `gpu_compact::alloc_gathered`, so my fix there covers
  the multipass path too. No temporal gate of its own.
- `gpu_upload.rs` — pinned H2D/D2H plumbing; no temporal/decimal dtype gate.

## Constraints honored
- Only the 5 permitted files edited; no engine.rs/lib.rs/mod.rs.
- No cargo build/check/test run.
- `#[cfg(test)]` unit tests added for every behavior change, all pure-host
  (no `#[ignore]` GPU requirement).
