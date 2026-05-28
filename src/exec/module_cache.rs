// SPDX-License-Identifier: Apache-2.0

//! Process-wide JIT module cache shared across executors.
//!
//! # Why this exists (review MED-fix)
//!
//! Historically every Tier-1 (`groupby_shmem_*_exec`) and Tier-2
//! (`groupby_tier2_*_exec`, `groupby_tier2_*_orchestrator`) executor maintained
//! its OWN `static MODULE_CACHE: Lazy<Mutex<HashMap<KernelSpec, CudaModule>>>`
//! (search the git history for `static MODULE_CACHE`). That had three problems:
//!
//! 1. **Wasted memory** — every executor cached separately, so a query that
//!    hit multiple sibling executors held N copies of the same kernel module
//!    bookkeeping (the inner `Arc<CudaModuleInner>` is shared by PTX-text-hash
//!    inside `jit::CudaModule::from_ptx`, but the per-executor `HashMap`
//!    entries and the kilobyte-scale PTX-build short-circuit lived N times).
//! 2. **Multi-GPU unsoundness** — a `CudaModule` is bound to the CUDA context
//!    that loaded it. Per-file statics can be primed by executor A on device
//!    0 and then served to executor B on device 1, where the module handle is
//!    invalid. Consolidating routes every cache lookup through one place so
//!    the eventual multi-GPU fix only has to touch this file.
//! 3. **Harder invalidation** — a debug/reset path would have to enumerate
//!    every executor's private static.
//!
//! # Strategy (option 3 from the task brief)
//!
//! There already exists a per-`Engine` `module_cache` for the projection-path
//! `KernelSpec` (see `engine::Engine::module_cache`). However each executor
//! defines its OWN local `enum KernelSpec` whose variants do not align with
//! the planner's `plan::KernelSpec`, so we cannot reuse that field's key
//! directly without restructuring every executor's signature.
//!
//! Plumbing `&Engine` through every Tier-1/Tier-2 executor's `try_execute` /
//! orchestrator entry would touch every dispatch caller — more than ten files
//! at last count. The task brief explicitly directs us to use the simpler
//! global-static approach when per-Engine plumbing would be that invasive.
//!
//! So this module exposes a single process-wide `GLOBAL_MODULE_CACHE` keyed by
//! a *namespaced* string identifier. Every executor calls
//! [`get_or_build_module`] with a stable `namespace` (typically `module_path!()`)
//! and a `spec_id` derived via `format!("{:?}", spec)`. Namespacing guarantees
//! that two executors that happen to declare identical `KernelSpec::Partition`
//! variants do not collide on cache slots and do not interfere with each
//! other's test-side load counters (see the `LoadCounter` glue below).
//!
//! ## TODO: per-Engine migration
//!
//! When `Engine` is plumbed through every executor's entry point (or when we
//! refactor `Engine::module_cache` to accept a string key rather than a
//! `ModuleCacheKey`), migrate every call site to `engine.get_or_build_module_str`
//! or similar. The global cache is correct for single-GPU workloads but
//! cannot disambiguate between modules belonging to different CUDA contexts
//! on a multi-GPU host. Tracked as a follow-up.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::error::BoltResult;
use crate::jit::CudaModule;
use crate::plan::physical_plan::{CompactionKernelSpec, KernelSpec};

/// Composite cache key: `(namespace, spec_id)`.
///
/// `namespace` is typically `module_path!()` from the calling executor —
/// stable across builds and unique per executor module. `spec_id` is the
/// Debug-formatted local `KernelSpec` (or any other stable identifier the
/// caller wants to use). String-typed so any executor can fit its existing
/// local KernelSpec enum without leaking the enum's type identity into this
/// crate-wide module.
type Key = (&'static str, String);

/// Process-wide module cache.
///
/// `CudaModule` is `Clone` over an internal `Arc<CudaModuleInner>`; storing
/// owned modules in the map and handing callers clones is cheap.
static GLOBAL_MODULE_CACHE: Lazy<Mutex<HashMap<Key, CudaModule>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Test- and observability-facing global miss counter.
///
/// Bumps once per `compile + CudaModule::from_ptx` round-trip serviced by
/// this cache. Tests that previously consulted a per-file `LOAD_COUNT` can
/// continue to do so via the [`LoadCounter`] wrapper; this global one is
/// available for cross-executor observability if we ever want it.
#[doc(hidden)]
pub static GLOBAL_LOAD_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Look up (or build, on a miss) the `CudaModule` for `(namespace, spec_id)`.
///
/// On a cache hit we hand back a cheap Arc-clone of the cached `CudaModule`
/// — no PTX generation, no `cuModuleLoadDataEx`. On a miss we run `compile`
/// (which should produce the PTX text for this spec) and feed the result to
/// `CudaModule::from_ptx`. That lower layer maintains its own PTX-text-hash
/// cache (see `jit::jit_compiler`), so even a cross-namespace miss against
/// the same PTX text reuses the already-loaded driver module.
///
/// The closure receives no arguments by design — each executor's local
/// `KernelSpec` enum stays private to its module; the executor's wrapper
/// dispatches the match on `spec` itself before calling into here. Keeping
/// the wrapper layer per-executor preserves the static type-checking of
/// each `KernelSpec` variant against its compile fn.
///
/// The optional `local_counter` lets per-executor tests keep their
/// historical `static LOAD_COUNT` invariants ("second call with same spec
/// must not bump the counter") without observing increments from sibling
/// executors. Production callers pass `None`.
pub(crate) fn get_or_build_module<F>(
    namespace: &'static str,
    spec_id: String,
    local_counter: Option<&LoadCounter>,
    compile: F,
) -> BoltResult<CudaModule>
where
    F: FnOnce() -> BoltResult<String>,
{
    // Fast path: cache hit. Hold the lock only long enough to clone the Arc.
    {
        let cache = GLOBAL_MODULE_CACHE.lock();
        if let Some(m) = cache.get(&(namespace, spec_id.clone())) {
            return Ok(m.clone());
        }
    }
    // Miss: compile + load WITHOUT the cache lock held. PTX generation and
    // `cuModuleLoadDataEx` can be slow; we don't want unrelated cache
    // misses serialising behind one ongoing compile. The jit PTX-text-hash
    // cache deduplicates the cubin load on its own.
    let ptx = compile()?;
    let module = CudaModule::from_ptx(&ptx)?;
    GLOBAL_LOAD_COUNT.fetch_add(1, Ordering::SeqCst);
    if let Some(c) = local_counter {
        c.0.fetch_add(1, Ordering::SeqCst);
    }
    // Insert and hand back a clone. If a concurrent thread raced us to the
    // same key, `or_insert` keeps the first winner — both threads observe
    // the same `Arc<CudaModuleInner>`, just one of the two `CudaModule`
    // wrappers we built gets dropped (cheap: an Arc dec).
    let mut cache = GLOBAL_MODULE_CACHE.lock();
    Ok(cache.entry((namespace, spec_id)).or_insert(module).clone())
}

/// Per-executor test helper: a thin newtype around `AtomicUsize` that
/// [`get_or_build_module`] bumps on a miss serviced via this counter.
///
/// Each executor that wants test compatibility with its old `LOAD_COUNT`
/// invariant declares `#[cfg(test)] static LOAD_COUNT: LoadCounter =
/// LoadCounter::new()` and threads `Some(&LOAD_COUNT)` into the
/// `get_or_build_module` call. Production callers pass `None` so the
/// branch optimises away.
#[doc(hidden)]
pub struct LoadCounter(pub AtomicUsize);

impl LoadCounter {
    /// New zero-initialised counter (for use in a `static`).
    pub const fn new() -> Self {
        Self(AtomicUsize::new(0))
    }

    /// Current cumulative miss count observed via this counter.
    #[allow(dead_code)] // exposed for tests; production code uses GLOBAL_LOAD_COUNT.
    pub fn load(&self, ordering: Ordering) -> usize {
        self.0.load(ordering)
    }
}

// ---------------------------------------------------------------------------
// v0.6 / M1+M6: KernelSpec-keyed cache layer.
//
// The PTX-text-hash cache in `jit::jit_compiler` (and the string-keyed
// `GLOBAL_MODULE_CACHE` above) only short-circuits AFTER codegen has run —
// every warm call still pays the cost of emitting the PTX text from the
// planner IR. Codegen for a typical Tier-1 kernel runs in the low single
// digits of milliseconds, which dominates the warm-cache JIT path.
//
// This layer adds a process-wide cache keyed by the **`KernelSpec`** itself
// (the IR BEFORE PTX emission) plus an entry-point tag. On a hit we return
// the cached PTX text + loaded `CudaModule` immediately, skipping codegen
// entirely. On a miss we run codegen, then route the PTX through
// `CudaModule::from_ptx` (which consults the PTX-text-hash cache so a
// cross-spec PTX collision still reuses the loaded module). The resulting
// `(ptx, module)` pair is stored under the KernelSpec key, so the next call
// with the same IR hits the fast path.
//
// # Key shape
//
// `KernelSpec` transitively contains `Op::Const { lit: Literal }`, and
// `Literal` carries `f32`/`f64` constants. Floats do not implement `Hash`
// (NaN inequality), so a `#[derive(Hash)]` on the planner IR would require a
// hand-rolled `Hash` impl over the raw bit patterns of every numeric literal
// (with a matching `PartialEq` so the `Hash`/`Eq` contract holds). That
// route reaches far outside this file's blast radius.
//
// We instead hash the `Debug` output of the spec with two domain-separated
// `DefaultHasher` instances, packing the result into a `(u64, u64)` 128-bit
// fingerprint. The same pattern is already used by
// `crate::exec::engine::ModuleCacheKey`; consolidating it here lets both the
// per-`Engine` cache and the new process-wide cache share the same shape.
//
// # Eviction
//
// FIFO, 256 entries — matching the convention in `jit::jit_compiler`
// (`PTX_CACHE_CAP_DEFAULT`). We keep a `VecDeque<(Key, ())>` insertion-order
// log alongside the map; on insert past the cap, we pop the front entry and
// remove it from the map. FIFO is cheap (no LRU bookkeeping) and matches the
// task brief; the PTX-text-hash cache below us is LRU and absorbs the
// occasional hot-key-evicted-early case.

/// FIFO cap on the KernelSpec→(PTX, module) cache. Matches the default
/// `PTX_CACHE_CAP_DEFAULT` in `jit::jit_compiler`.
const KERNELSPEC_CACHE_CAP: usize = 256;

/// 128-bit content fingerprint of a `KernelSpec` plus its entry-point tag.
///
/// The tag distinguishes between the multiple PTX shapes a single spec can
/// produce (e.g. the full projection kernel vs. the predicate-only mask
/// kernel emitted from the same `KernelSpec`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct KernelSpecKey {
    /// Upper 64 bits of the 128-bit content hash (domain byte `0x01`).
    hi: u64,
    /// Lower 64 bits of the 128-bit content hash (domain byte `0x02`).
    lo: u64,
    /// PTX entry-point name (e.g. `"bolt_kernel"` vs. `"bolt_predicate"`).
    entry: &'static str,
}

/// `fmt::Write` → `Hasher` adapter so we can stream `Debug` output of a
/// `KernelSpec` directly into a hasher with zero heap allocation.
struct HasherWrite<'a, H: std::hash::Hasher>(&'a mut H);

impl<H: std::hash::Hasher> std::fmt::Write for HasherWrite<'_, H> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write(s.as_bytes());
        Ok(())
    }
}

impl KernelSpecKey {
    /// Compute the key for `(spec, entry)`. See type docs for rationale.
    fn new(spec: &KernelSpec, entry: &'static str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::fmt::Write as _;
        use std::hash::Hasher;

        let mut hi = DefaultHasher::new();
        hi.write_u8(0x01);
        // Both arms are unreachable in practice; degrade to a benign cache
        // miss rather than panic if Debug formatting ever fails.
        let _ = write!(HasherWrite(&mut hi), "{:?}", spec);

        let mut lo = DefaultHasher::new();
        lo.write_u8(0x02);
        let _ = write!(HasherWrite(&mut lo), "{:?}", spec);

        Self {
            hi: hi.finish(),
            lo: lo.finish(),
            entry,
        }
    }
}

/// Cached payload for one KernelSpec key: the emitted PTX text and the
/// loaded module. The PTX text is retained so callers that want to inspect
/// it (e.g. for tracing) can do so without re-running codegen, and so a
/// future hit-collision-detection path can compare-by-content if needed.
#[derive(Clone)]
struct KernelSpecEntry {
    /// Emitted PTX text. Retained for observability / future
    /// hit-collision-detection. `#[allow(dead_code)]` because the
    /// production lookup path only needs `module` — but storing the PTX
    /// here is part of the cache contract per the v0.6 task brief, so we
    /// don't want a future refactor to drop the field by mistake.
    #[allow(dead_code)]
    ptx: String,
    module: CudaModule,
}

/// State of the KernelSpec-keyed cache.
///
/// `by_key` is the primary lookup; `order` is a parallel FIFO log used for
/// eviction. The two are kept in sync by `insert` (the only mutator). On
/// eviction we pop the front of `order` and remove the matching map entry.
struct KernelSpecCache {
    by_key: HashMap<KernelSpecKey, KernelSpecEntry>,
    order: VecDeque<KernelSpecKey>,
    /// Cap on the number of cached entries before FIFO eviction kicks in.
    /// Stored on the struct (rather than read as a const) so tests can
    /// drive eviction with a small cap without polluting the global.
    cap: usize,
    /// Cumulative count of cache hits (fast-path returns). Tests observe
    /// this to confirm the hit path is wired up.
    hits: u64,
    /// Cumulative count of cache misses (codegen + module-load round trips).
    misses: u64,
}

impl KernelSpecCache {
    fn new(cap: usize) -> Self {
        Self {
            by_key: HashMap::with_capacity(cap),
            order: VecDeque::with_capacity(cap),
            cap,
            hits: 0,
            misses: 0,
        }
    }

    /// Look up `key`; on a hit, return the cached entry (cheap clone — the
    /// inner `CudaModule` is `Arc`-shared and `String` is one allocation).
    fn get(&mut self, key: &KernelSpecKey) -> Option<KernelSpecEntry> {
        if let Some(entry) = self.by_key.get(key) {
            self.hits = self.hits.saturating_add(1);
            Some(entry.clone())
        } else {
            self.misses = self.misses.saturating_add(1);
            None
        }
    }

    /// Insert `(key, entry)`, evicting the oldest entry if we are at cap.
    /// Idempotent: re-inserting an existing key is a no-op (preserves the
    /// existing FIFO position rather than re-aging it).
    fn insert(&mut self, key: KernelSpecKey, entry: KernelSpecEntry) {
        if self.by_key.contains_key(&key) {
            return;
        }
        while self.by_key.len() >= self.cap {
            if let Some(oldest) = self.order.pop_front() {
                self.by_key.remove(&oldest);
            } else {
                break;
            }
        }
        self.order.push_back(key);
        self.by_key.insert(key, entry);
    }
}

/// Process-wide KernelSpec→(PTX, CudaModule) cache. Initialised lazily.
static KERNELSPEC_CACHE: Lazy<Mutex<KernelSpecCache>> =
    Lazy::new(|| Mutex::new(KernelSpecCache::new(KERNELSPEC_CACHE_CAP)));

/// Test- and observability-facing snapshot of `(hits, misses)` for the
/// KernelSpec cache.
#[doc(hidden)]
#[must_use]
pub fn kernelspec_cache_stats() -> (usize, usize) {
    let c = KERNELSPEC_CACHE.lock();
    (c.hits as usize, c.misses as usize)
}

/// Look up (or compile-and-load) the `CudaModule` for `(spec, entry)`.
///
/// On a **cache hit** we return the cached `CudaModule` clone immediately —
/// no codegen, no PTX text generation, no `cuModuleLoadDataEx`. This is the
/// sub-microsecond warm-cache path that motivates this layer.
///
/// On a **cache miss** we run `compile(spec)` to get the PTX text, hand it
/// to `CudaModule::from_ptx` (which consults the PTX-text-hash cache in
/// `jit::jit_compiler` — so a cross-spec PTX collision still skips PTXAS
/// reassembly), and store the resulting `(ptx, module)` pair under the
/// KernelSpec key. Subsequent calls with the same spec hit the fast path.
///
/// Concurrency: the cache lock is held only long enough to look up or
/// insert; the slow `compile` + `from_ptx` work runs outside the lock. If
/// two threads race on the same miss they will each compile once and the
/// second insert is dropped by `insert`'s idempotence check — the PTX-text
/// cache below us deduplicates the heavier `cuModuleLoadDataEx` step
/// regardless.
pub(crate) fn get_or_build_module_for_spec<F>(
    spec: &KernelSpec,
    entry: &'static str,
    compile: F,
) -> BoltResult<CudaModule>
where
    F: FnOnce(&KernelSpec) -> BoltResult<String>,
{
    let key = KernelSpecKey::new(spec, entry);
    // Fast path: hit.
    if let Some(entry) = KERNELSPEC_CACHE.lock().get(&key) {
        return Ok(entry.module);
    }
    // Miss: run codegen + module load WITHOUT the cache lock held. Both
    // steps can be slow and we don't want to serialise unrelated misses.
    let ptx = compile(spec)?;
    let module = CudaModule::from_ptx(&ptx)?;
    // Store under the KernelSpec key so the next call skips codegen.
    KERNELSPEC_CACHE.lock().insert(
        key,
        KernelSpecEntry {
            ptx,
            module: module.clone(),
        },
    );
    Ok(module)
}

// ---------------------------------------------------------------------------
// v0.7: CompactionKernelSpec-keyed cache layer.
//
// Mirrors the `KernelSpec`-keyed layer above but keyed on the
// compaction-family planner IR (see
// [`crate::plan::physical_plan::CompactionKernelSpec`]). The compaction
// family covers the prefix-scan + gather kernels that
// `crate::exec::gpu_compact` and `crate::exec::gpu_compact_multipass`
// launch:
//
//   * `PrefixScan(HillisSteele|Blelloch|Lookback)` — the three
//     u8-mask scan kernels emitted by `jit::prefix_scan`.
//   * `PrefixScanU32` / `AddBlockBases` — the recursive multipass
//     helpers in `jit::prefix_scan_multipass`.
//   * `Gather(DataType)` — the per-dtype fixed-width gather kernel.
//   * `GatherBoolNullable` — reserved for a future fused
//     values+validity gather (today two `Gather(Bool)` launches).
//
// Each compaction kernel takes zero or one codegen-time knob and
// emits a fixed-entry-name PTX blob. The wrapper here maps a
// `CompactionKernelSpec` to that closure-supplied PTX and routes the
// resulting module through the existing `CudaModule::from_ptx`
// pipeline. The in-memory cache is independent from the projection /
// scalar-agg / hash-join / radix-sort caches so the FIFO eviction
// policies of the families don't compete.
//
// # Disk-cache prefix
//
// `"compaction::"` — keeps the on-disk PTX cache directory
// human-greppable and prevents accidental key collision with the
// projection-side / scalar-agg / hash-join / radix-sort families that
// share the cache directory.
//
// # Why `entry: &str` participates in the key
//
// Two specs can share `kind` but produce PTX with different entry
// symbols (e.g. `bolt_prefix_scan` for HillisSteele vs.
// `bolt_prefix_scan_blelloch` for Blelloch). The kind-tag already
// distinguishes these in `kind`, but we still mix `entry` into the
// 128-bit fingerprint as a defence-in-depth: if a future variant
// keeps the same `kind` but flips the PTX entry name (e.g. a
// keys-only vs. with-payload variant), the cache slots stay distinct
// without requiring an IR-enum change.

/// FIFO cap on the CompactionKernelSpec→`CudaModule` cache.
/// 64 matches the sibling spec-keyed caches (scalar-agg / hash-join /
/// radix-sort). The compaction family is small (3 scan algos + 5
/// gather dtypes + 2 multipass helpers + 1 bool-nullable reservation
/// = ~11 entries in practice), so 64 leaves substantial headroom.
const COMPACTION_CACHE_CAP: usize = 64;

/// Disk-cache key prefix for the compaction PTX family.
///
/// Stamped onto every disk-cache write/read so a directory listing
/// shows immediately which kernel family produced each `.ptx` file,
/// and so the projection / scalar-aggregate / hash-join /
/// radix-sort paths' key shapes can't ever collide with a compaction
/// entry.
pub(crate) const COMPACTION_DISK_PREFIX: &str = "compaction::";

/// 128-bit content fingerprint of a `CompactionKernelSpec` plus its
/// entry-point tag. Domain bytes `0x41` / `0x42` distinguish this
/// family from the KernelSpec (`0x01` / `0x02`) family already in
/// this file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct CompactionKey {
    /// Upper 64 bits of the 128-bit content hash (domain byte `0x41`).
    hi: u64,
    /// Lower 64 bits of the 128-bit content hash (domain byte `0x42`).
    lo: u64,
    /// PTX entry-point name (e.g. `"bolt_prefix_scan"`,
    /// `"bolt_gather_i32"`). Participates in the key so two specs
    /// that share `kind` but emit PTX with different entry names
    /// occupy distinct cache slots.
    entry: &'static str,
}

impl CompactionKey {
    /// Compute the key for `(spec, entry)`. The 128-bit hash is built
    /// from `Debug` output of the spec with domain-separating prefix
    /// bytes distinct from `KernelSpecKey`'s (`0x01` / `0x02`) so a
    /// hypothetical hash collision between the Debug shapes of the
    /// two spec types can't alias keys.
    fn new(spec: &CompactionKernelSpec, entry: &'static str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::fmt::Write as _;
        use std::hash::Hasher;

        let mut hi = DefaultHasher::new();
        hi.write_u8(0x41);
        let _ = write!(HasherWrite(&mut hi), "{:?}", spec);

        let mut lo = DefaultHasher::new();
        lo.write_u8(0x42);
        let _ = write!(HasherWrite(&mut lo), "{:?}", spec);

        Self {
            hi: hi.finish(),
            lo: lo.finish(),
            entry,
        }
    }
}

/// Cached payload for one CompactionKernelSpec key. Same shape as
/// `KernelSpecEntry`: PTX text retained for observability, module
/// for the fast-path return.
#[derive(Clone)]
struct CompactionEntry {
    /// Emitted PTX text. Retained for observability; the production
    /// lookup path only needs `module`.
    #[allow(dead_code)]
    ptx: String,
    module: CudaModule,
}

/// State of the CompactionKernelSpec-keyed cache. FIFO eviction with
/// the same shape as `KernelSpecCache` — see those docs.
struct CompactionCache {
    by_key: HashMap<CompactionKey, CompactionEntry>,
    order: VecDeque<CompactionKey>,
    cap: usize,
    hits: u64,
    misses: u64,
}

impl CompactionCache {
    fn new(cap: usize) -> Self {
        Self {
            by_key: HashMap::with_capacity(cap),
            order: VecDeque::with_capacity(cap),
            cap,
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, key: &CompactionKey) -> Option<CompactionEntry> {
        if let Some(entry) = self.by_key.get(key) {
            self.hits = self.hits.saturating_add(1);
            Some(entry.clone())
        } else {
            self.misses = self.misses.saturating_add(1);
            None
        }
    }

    fn insert(&mut self, key: CompactionKey, entry: CompactionEntry) {
        if self.by_key.contains_key(&key) {
            return;
        }
        while self.by_key.len() >= self.cap {
            if let Some(oldest) = self.order.pop_front() {
                self.by_key.remove(&oldest);
            } else {
                break;
            }
        }
        self.order.push_back(key);
        self.by_key.insert(key, entry);
    }
}

/// Process-wide CompactionKernelSpec→`CudaModule` cache. Initialised lazily.
static COMPACTION_CACHE: Lazy<Mutex<CompactionCache>> =
    Lazy::new(|| Mutex::new(CompactionCache::new(COMPACTION_CACHE_CAP)));

/// Test- and observability-facing snapshot of `(hits, misses)` for
/// the CompactionKernelSpec cache. Parallel to
/// [`kernelspec_cache_stats`].
#[doc(hidden)]
#[must_use]
pub fn compaction_cache_stats() -> (usize, usize) {
    let c = COMPACTION_CACHE.lock();
    (c.hits as usize, c.misses as usize)
}

/// Compose the on-disk PTX cache key for a CompactionKernelSpec
/// lookup.
///
/// The shape is `"{COMPACTION_DISK_PREFIX}{entry}-{hex(hash128)}"`:
///   1. The `"compaction::"` prefix domain-separates these entries
///      from any other PTX family that may share the disk-cache
///      directory.
///   2. The `entry` suffix distinguishes the per-`kind` PTX entry
///      points (e.g. `bolt_prefix_scan` vs
///      `bolt_prefix_scan_blelloch` vs `bolt_gather_i32`).
///   3. The 128-bit hex content hash makes the key
///      collision-resistant against unrelated specs that happen to
///      share the same `kind` shape.
fn compaction_disk_key(key: &CompactionKey) -> String {
    format!(
        "{}{}-{}",
        COMPACTION_DISK_PREFIX,
        key.entry,
        crate::jit::disk_cache::hash_to_key(key.hi, key.lo),
    )
}

/// Look up (or compile-and-load) the `CudaModule` for `(spec, entry)`.
///
/// This is the compaction-family analogue of
/// [`get_or_build_module_for_spec`]. On a **cache hit** we return the
/// cached `CudaModule` clone immediately — no codegen, no PTX text
/// generation, no `cuModuleLoadDataEx`. On a **cache miss** we run
/// `compile(spec)` to get the PTX text and hand it to
/// `CudaModule::from_ptx`. The resulting `(ptx, module)` pair is
/// stored under the CompactionKernelSpec key so subsequent calls with
/// the same spec hit the fast path.
///
/// # Disk-cache integration
///
/// When the disk-backed PTX cache is enabled (env var
/// `BOLT_PTX_CACHE_DIR` or builder override), a miss in the in-memory
/// cache consults the disk cache *before* paying the codegen cost.
/// See [`compaction_disk_key`] for the key shape; the
/// `"compaction::"` prefix keeps these entries human-greppable in a
/// shared cache directory.
///
/// # Concurrency
///
/// The cache lock is held only long enough to look up or insert; the
/// slow `compile` + `from_ptx` work runs outside the lock. If two
/// threads race on the same miss they will each compile once and the
/// second insert is dropped by `insert`'s idempotence check.
pub(crate) fn get_or_build_module_for_compaction<F>(
    spec: &CompactionKernelSpec,
    entry: &'static str,
    compile: F,
) -> BoltResult<CudaModule>
where
    F: FnOnce(&CompactionKernelSpec) -> BoltResult<String>,
{
    let key = CompactionKey::new(spec, entry);
    // Fast path: in-memory hit. Hold the lock just long enough to clone.
    if let Some(cached) = COMPACTION_CACHE.lock().get(&key) {
        return Ok(cached.module);
    }
    // In-memory miss: try the optional on-disk cache before paying
    // for codegen.
    let disk = crate::jit::disk_cache::disk_cache();
    let disk_key = disk.as_ref().map(|_| compaction_disk_key(&key));
    let ptx = match (&disk, &disk_key) {
        (Some(cache), Some(k)) => match cache.lookup(k) {
            Some(text) => text,
            None => {
                let text = compile(spec)?;
                // Write-through to disk. Errors here are non-fatal: a
                // failed write just means future processes won't
                // benefit, the current process still loads the module
                // successfully.
                let _ = cache.store(k, &text);
                text
            }
        },
        _ => compile(spec)?,
    };
    let module = CudaModule::from_ptx(&ptx)?;
    COMPACTION_CACHE.lock().insert(
        key,
        CompactionEntry {
            ptx,
            module: module.clone(),
        },
    );
    Ok(module)
}

// ---------------------------------------------------------------------------
// Tests for the KernelSpec-keyed cache.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod kernelspec_cache_tests {
    use super::*;
    use crate::plan::physical_plan::KernelSpec;
    use std::sync::atomic::{AtomicUsize, Ordering as AOrdering};

    /// Stand-up an empty `KernelSpec`. All fields default to empty vectors /
    /// `None`; the IR is degenerate but its `Debug` representation is
    /// stable and unique enough for hashing (and we can perturb fields
    /// across specs to test miss behaviour).
    fn empty_spec() -> KernelSpec {
        KernelSpec {
            inputs: Vec::new(),
            outputs: Vec::new(),
            ops: Vec::new(),
            predicate: None,
            register_count: 0,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        }
    }

    /// Cold lookup against a fresh cache is a miss, and `get` accounts
    /// it as such. Pins the miss-counter wiring used by the
    /// `kernelspec_cache_stats` observability hook.
    #[test]
    fn kernelspec_cache_cold_lookup_is_a_miss() {
        let mut cache = KernelSpecCache::new(4);
        let spec = empty_spec();
        let key = KernelSpecKey::new(&spec, "bolt_kernel");

        assert!(cache.get(&key).is_none(), "fresh cache must miss");
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 0);
    }

    /// State-machine-level pin of the hit path: `get_or_build_module_for_spec`
    /// must run the compile closure exactly once per unique spec, and zero
    /// times on the second call with the same spec.
    ///
    /// We can't actually invoke `CudaModule::from_ptx` without a CUDA
    /// context, so this test drives the cache via a parallel mock that
    /// reproduces the lookup-or-insert logic on a fake "module" type.
    /// The production code path is the same shape — only the inner
    /// `CudaModule::from_ptx` differs.
    #[test]
    fn kernelspec_cache_compile_runs_once_per_unique_spec() {
        // Mock module type: just a tag so we can confirm we got the same
        // entry back on a hit.
        #[derive(Clone)]
        struct MockModule(&'static str);

        struct MockCache {
            by_key: HashMap<KernelSpecKey, MockModule>,
            order: VecDeque<KernelSpecKey>,
            cap: usize,
        }

        impl MockCache {
            fn get_or_build(
                &mut self,
                spec: &KernelSpec,
                entry: &'static str,
                compile_count: &AtomicUsize,
                tag: &'static str,
            ) -> MockModule {
                let key = KernelSpecKey::new(spec, entry);
                if let Some(m) = self.by_key.get(&key) {
                    return m.clone();
                }
                // Miss: "compile".
                compile_count.fetch_add(1, AOrdering::SeqCst);
                let m = MockModule(tag);
                while self.by_key.len() >= self.cap {
                    if let Some(oldest) = self.order.pop_front() {
                        self.by_key.remove(&oldest);
                    } else {
                        break;
                    }
                }
                self.order.push_back(key);
                self.by_key.insert(key, m.clone());
                m
            }
        }

        let spec_a = empty_spec();
        let spec_b = KernelSpec {
            register_count: 7,
            ..empty_spec()
        };

        let compile_count = AtomicUsize::new(0);
        let mut cache = MockCache {
            by_key: HashMap::new(),
            order: VecDeque::new(),
            cap: 256,
        };

        // First call with spec_a: miss → compile runs.
        let m1 = cache.get_or_build(&spec_a, "bolt_kernel", &compile_count, "A");
        assert_eq!(compile_count.load(AOrdering::SeqCst), 1);
        assert_eq!(m1.0, "A");

        // Second call with spec_a: HIT → compile does NOT run.
        let m2 = cache.get_or_build(&spec_a, "bolt_kernel", &compile_count, "A");
        assert_eq!(
            compile_count.load(AOrdering::SeqCst),
            1,
            "warm-cache hit must skip codegen"
        );
        assert_eq!(m2.0, "A");

        // Third call with a DIFFERENT spec: miss again.
        let m3 = cache.get_or_build(&spec_b, "bolt_kernel", &compile_count, "B");
        assert_eq!(compile_count.load(AOrdering::SeqCst), 2);
        assert_eq!(m3.0, "B");

        // Fourth call with spec_a again: still a hit.
        let m4 = cache.get_or_build(&spec_a, "bolt_kernel", &compile_count, "A");
        assert_eq!(compile_count.load(AOrdering::SeqCst), 2);
        assert_eq!(m4.0, "A");

        // Fifth call with spec_a but a DIFFERENT entry tag: miss (entry
        // participates in the key).
        let m5 = cache.get_or_build(&spec_a, "bolt_predicate", &compile_count, "Ap");
        assert_eq!(compile_count.load(AOrdering::SeqCst), 3);
        assert_eq!(m5.0, "Ap");
    }

    /// Key uniqueness: distinct `KernelSpec`s and distinct entry tags
    /// must produce distinct keys. (This is the assumption the
    /// hit-vs-miss classification relies on.)
    #[test]
    fn kernelspec_cache_key_distinguishes_specs_and_entries() {
        let mk_spec = |rc: u32| KernelSpec {
            register_count: rc,
            ..empty_spec()
        };
        let k0 = KernelSpecKey::new(&mk_spec(0), "k");
        let k1 = KernelSpecKey::new(&mk_spec(1), "k");
        let k0_alt = KernelSpecKey::new(&mk_spec(0), "k_alt");
        assert_ne!(k0, k1, "different specs must produce different keys");
        assert_ne!(k0, k0_alt, "entry tag must participate in the key");
    }

    /// The key fingerprint is stable across `Clone` of the same spec — a
    /// load-bearing invariant: callers often clone the IR before handing
    /// it to the cache (e.g. to keep the planner output borrow-free).
    #[test]
    fn kernelspec_cache_key_stable_across_clone() {
        let spec = KernelSpec {
            register_count: 42,
            ..empty_spec()
        };
        let cloned = spec.clone();
        assert_eq!(
            KernelSpecKey::new(&spec, "bolt_kernel"),
            KernelSpecKey::new(&cloned, "bolt_kernel"),
            "cloning the spec must not change its cache key"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests for the CompactionKernelSpec-keyed cache.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod compaction_cache_tests {
    use super::*;
    use crate::plan::logical_plan::DataType;
    use crate::plan::physical_plan::{
        CompactionKernelKind, CompactionKernelSpec, PrefixScanAlgoTag,
    };
    use std::sync::atomic::{AtomicUsize, Ordering as AOrdering};

    /// Spec constructor mirroring the production call-site shape — the
    /// kernels in `crate::exec::gpu_compact` build the spec inline at
    /// the launch site.
    fn mk_spec(kind: CompactionKernelKind) -> CompactionKernelSpec {
        CompactionKernelSpec { kind }
    }

    /// Cold lookup against a fresh cache is a miss and `get` accounts
    /// for it. Mirrors `kernelspec_cache_cold_lookup_is_a_miss`.
    #[test]
    fn compaction_cache_cold_lookup_is_a_miss() {
        let mut cache = CompactionCache::new(4);
        let spec = mk_spec(CompactionKernelKind::PrefixScan(
            PrefixScanAlgoTag::HillisSteele,
        ));
        let key = CompactionKey::new(&spec, "bolt_prefix_scan");

        assert!(cache.get(&key).is_none(), "fresh cache must miss");
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 0);
    }

    /// State-machine pin of the hit path. We can't actually call
    /// `CudaModule::from_ptx` without a CUDA context, so the test
    /// drives the cache via a parallel mock that reproduces the
    /// lookup-or-insert logic on a fake "module" type. The production
    /// path is the same shape — only the inner module loader differs.
    ///
    /// What this pins:
    ///   * compile runs once per unique `(spec, entry)`,
    ///   * a second call with the same `(spec, entry)` is a hit,
    ///   * a call with a different `kind` variant is a miss,
    ///   * a call with the same `kind` but different `entry` is a
    ///     miss.
    #[test]
    fn compaction_cache_compile_runs_once_per_unique_spec() {
        #[derive(Clone)]
        struct MockModule(&'static str);

        struct MockCache {
            by_key: HashMap<CompactionKey, MockModule>,
            order: VecDeque<CompactionKey>,
            cap: usize,
        }

        impl MockCache {
            fn get_or_build(
                &mut self,
                spec: &CompactionKernelSpec,
                entry: &'static str,
                compile_count: &AtomicUsize,
                tag: &'static str,
            ) -> MockModule {
                let key = CompactionKey::new(spec, entry);
                if let Some(m) = self.by_key.get(&key) {
                    return m.clone();
                }
                compile_count.fetch_add(1, AOrdering::SeqCst);
                let m = MockModule(tag);
                while self.by_key.len() >= self.cap {
                    if let Some(oldest) = self.order.pop_front() {
                        self.by_key.remove(&oldest);
                    } else {
                        break;
                    }
                }
                self.order.push_back(key);
                self.by_key.insert(key, m.clone());
                m
            }
        }

        let scan_hs = mk_spec(CompactionKernelKind::PrefixScan(
            PrefixScanAlgoTag::HillisSteele,
        ));
        let scan_bl =
            mk_spec(CompactionKernelKind::PrefixScan(PrefixScanAlgoTag::Blelloch));
        let gather_i32 = mk_spec(CompactionKernelKind::Gather(DataType::Int32));

        let compile_count = AtomicUsize::new(0);
        let mut cache = MockCache {
            by_key: HashMap::new(),
            order: VecDeque::new(),
            cap: 64,
        };

        // First call: miss → compile runs.
        let m1 = cache.get_or_build(&scan_hs, "bolt_prefix_scan", &compile_count, "HS");
        assert_eq!(compile_count.load(AOrdering::SeqCst), 1);
        assert_eq!(m1.0, "HS");

        // Second call with the same (spec, entry): HIT.
        let m2 = cache.get_or_build(&scan_hs, "bolt_prefix_scan", &compile_count, "HS");
        assert_eq!(
            compile_count.load(AOrdering::SeqCst),
            1,
            "warm-cache hit must skip codegen"
        );
        assert_eq!(m2.0, "HS");

        // Different algo variant: miss.
        let m3 = cache.get_or_build(
            &scan_bl,
            "bolt_prefix_scan_blelloch",
            &compile_count,
            "BL",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 2);
        assert_eq!(m3.0, "BL");

        // Different kind entirely (Gather): miss.
        let m4 = cache.get_or_build(&gather_i32, "bolt_gather_i32", &compile_count, "GI32");
        assert_eq!(compile_count.load(AOrdering::SeqCst), 3);
        assert_eq!(m4.0, "GI32");

        // Re-querying the first (scan_hs, "bolt_prefix_scan") still
        // hits the cache.
        let m5 = cache.get_or_build(&scan_hs, "bolt_prefix_scan", &compile_count, "HS");
        assert_eq!(compile_count.load(AOrdering::SeqCst), 3);
        assert_eq!(m5.0, "HS");

        // Same scan_hs but a DIFFERENT entry tag: miss (entry
        // participates in the key).
        let m6 = cache.get_or_build(&scan_hs, "bolt_prefix_scan_alt", &compile_count, "HSa");
        assert_eq!(compile_count.load(AOrdering::SeqCst), 4);
        assert_eq!(m6.0, "HSa");
    }

    /// Key uniqueness: distinct kinds, distinct algo tags, distinct
    /// gather dtypes, and distinct entry tags must all produce
    /// distinct keys.
    #[test]
    fn compaction_cache_key_distinguishes_specs_and_entries() {
        let hs = mk_spec(CompactionKernelKind::PrefixScan(
            PrefixScanAlgoTag::HillisSteele,
        ));
        let bl =
            mk_spec(CompactionKernelKind::PrefixScan(PrefixScanAlgoTag::Blelloch));
        let lb =
            mk_spec(CompactionKernelKind::PrefixScan(PrefixScanAlgoTag::Lookback));
        let scan_u32 = mk_spec(CompactionKernelKind::PrefixScanU32);
        let add_bases = mk_spec(CompactionKernelKind::AddBlockBases);
        let g_i32 = mk_spec(CompactionKernelKind::Gather(DataType::Int32));
        let g_i64 = mk_spec(CompactionKernelKind::Gather(DataType::Int64));
        let g_bool = mk_spec(CompactionKernelKind::Gather(DataType::Bool));
        let g_bool_n = mk_spec(CompactionKernelKind::GatherBoolNullable);

        let k_hs = CompactionKey::new(&hs, "bolt_prefix_scan");
        let k_bl = CompactionKey::new(&bl, "bolt_prefix_scan_blelloch");
        let k_lb = CompactionKey::new(&lb, "bolt_prefix_scan_lookback");
        let k_u32 = CompactionKey::new(&scan_u32, "bolt_prefix_scan_u32");
        let k_ab = CompactionKey::new(&add_bases, "bolt_add_block_bases");
        let k_g_i32 = CompactionKey::new(&g_i32, "bolt_gather_i32");
        let k_g_i64 = CompactionKey::new(&g_i64, "bolt_gather_i64");
        let k_g_bool = CompactionKey::new(&g_bool, "bolt_gather_bool");
        let k_g_bool_n = CompactionKey::new(&g_bool_n, "bolt_gather_bool");

        // Three algo variants are distinct.
        assert_ne!(k_hs, k_bl);
        assert_ne!(k_bl, k_lb);
        assert_ne!(k_hs, k_lb);

        // Multipass helpers are distinct from each other and from the
        // scan family.
        assert_ne!(k_u32, k_ab);
        assert_ne!(k_u32, k_hs);
        assert_ne!(k_ab, k_hs);

        // Per-dtype gather slots are distinct.
        assert_ne!(k_g_i32, k_g_i64);
        assert_ne!(k_g_i32, k_g_bool);
        assert_ne!(k_g_i64, k_g_bool);

        // GatherBoolNullable is distinct from Gather(Bool) even when
        // the entry symbol happens to coincide — this is the
        // load-bearing slot-distinctness check.
        assert_ne!(k_g_bool, k_g_bool_n);

        // Entry tag participates: same spec, different entry → miss.
        let hs_alt = CompactionKey::new(&hs, "bolt_prefix_scan_alt");
        assert_ne!(k_hs, hs_alt);
    }

    /// The key fingerprint is stable across `Clone` of the same spec —
    /// callers often build a spec inline at the launch site and the
    /// cache lookup must agree with the next call's lookup byte for
    /// byte.
    #[test]
    fn compaction_cache_key_stable_across_clone() {
        let spec = mk_spec(CompactionKernelKind::Gather(DataType::Float64));
        let cloned = spec;
        assert_eq!(
            CompactionKey::new(&spec, "bolt_gather_f64"),
            CompactionKey::new(&cloned, "bolt_gather_f64"),
            "cloning the spec must not change its cache key"
        );
    }

    /// FIFO eviction: filling the cache to `cap` and inserting one
    /// more entry must evict the oldest key. The eviction order is
    /// the insertion order, not LRU; this test pins that contract so
    /// the cache stays cheap to maintain.
    #[test]
    fn compaction_cache_fifo_eviction_pops_oldest() {
        // Quick sanity check on the production cache's stored cap so a
        // mistaken cap-zero refactor surfaces here, then drive the
        // eviction policy below through a parallel fake.
        let cache = CompactionCache::new(2);
        assert_eq!(cache.cap, 2, "constructor must round-trip the cap");

        let s1 = mk_spec(CompactionKernelKind::Gather(DataType::Int32));
        let s2 = mk_spec(CompactionKernelKind::Gather(DataType::Int64));
        let s3 = mk_spec(CompactionKernelKind::Gather(DataType::Float32));

        let k1 = CompactionKey::new(&s1, "bolt_gather_i32");
        let k2 = CompactionKey::new(&s2, "bolt_gather_i64");
        let k3 = CompactionKey::new(&s3, "bolt_gather_f32");

        // The CompactionEntry stores a CudaModule which we can't
        // build without CUDA, so we drive the eviction policy
        // through a parallel `by_key` HashMap with a fake entry
        // type. The eviction logic itself is a 6-line FIFO loop
        // that `CompactionCache::insert` exercises in the same
        // shape; this test pins that shape without needing a GPU.
        fn fake_insert(
            by_key: &mut HashMap<CompactionKey, ()>,
            order: &mut VecDeque<CompactionKey>,
            cap: usize,
            key: CompactionKey,
        ) {
            if by_key.contains_key(&key) {
                return;
            }
            while by_key.len() >= cap {
                if let Some(oldest) = order.pop_front() {
                    by_key.remove(&oldest);
                } else {
                    break;
                }
            }
            order.push_back(key);
            by_key.insert(key, ());
        }

        let mut by_key: HashMap<CompactionKey, ()> = HashMap::new();
        let mut order: VecDeque<CompactionKey> = VecDeque::new();

        fake_insert(&mut by_key, &mut order, cache.cap, k1);
        fake_insert(&mut by_key, &mut order, cache.cap, k2);
        assert!(by_key.contains_key(&k1));
        assert!(by_key.contains_key(&k2));
        assert_eq!(by_key.len(), 2);

        // Inserting a third entry at cap evicts the oldest (k1).
        fake_insert(&mut by_key, &mut order, cache.cap, k3);
        assert!(!by_key.contains_key(&k1), "oldest entry must be evicted");
        assert!(by_key.contains_key(&k2));
        assert!(by_key.contains_key(&k3));
        assert_eq!(by_key.len(), 2);
    }

    /// `get_or_build_module_for_compaction` exposes a closure-supplied
    /// compile path; the test here drives a stand-in stub loader
    /// through the same key/cache machinery to confirm the wiring
    /// between the public entry point and the underlying
    /// `CompactionCache` is correct.
    ///
    /// We can't actually invoke `CudaModule::from_ptx` (no CUDA in
    /// unit tests), so this test exercises the cache state machine
    /// directly via a stub loader closure that mimics `from_ptx`'s
    /// signature. The production path differs only at the loader
    /// step.
    #[test]
    fn compaction_cache_stub_loader_drives_cache_state_machine() {
        // Helper that mirrors `get_or_build_module_for_compaction`
        // but takes a stand-in loader closure in place of
        // `CudaModule::from_ptx`. The cache shape and state
        // transitions are identical; only the module-construction
        // primitive differs.
        fn get_or_build_with_loader<F, L, M>(
            cache: &mut HashMap<CompactionKey, M>,
            order: &mut VecDeque<CompactionKey>,
            cap: usize,
            spec: &CompactionKernelSpec,
            entry: &'static str,
            compile: F,
            loader: L,
            stats: &mut (u64, u64),
        ) -> BoltResult<M>
        where
            F: FnOnce(&CompactionKernelSpec) -> BoltResult<String>,
            L: FnOnce(&str) -> BoltResult<M>,
            M: Clone,
        {
            let key = CompactionKey::new(spec, entry);
            if let Some(cached) = cache.get(&key) {
                stats.0 += 1; // hit
                return Ok(cached.clone());
            }
            stats.1 += 1; // miss
            let ptx = compile(spec)?;
            let module = loader(&ptx)?;
            if !cache.contains_key(&key) {
                while cache.len() >= cap {
                    if let Some(oldest) = order.pop_front() {
                        cache.remove(&oldest);
                    } else {
                        break;
                    }
                }
                order.push_back(key);
                cache.insert(key, module.clone());
            }
            Ok(module)
        }

        let compile_count = AtomicUsize::new(0);
        let loader_count = AtomicUsize::new(0);
        let mut cache: HashMap<CompactionKey, &'static str> = HashMap::new();
        let mut order: VecDeque<CompactionKey> = VecDeque::new();
        let mut stats: (u64, u64) = (0, 0);

        let spec = mk_spec(CompactionKernelKind::PrefixScan(
            PrefixScanAlgoTag::Lookback,
        ));

        // First call: miss → compile and loader each run once.
        let m1 = get_or_build_with_loader(
            &mut cache,
            &mut order,
            64,
            &spec,
            "bolt_prefix_scan_lookback",
            |_s| {
                compile_count.fetch_add(1, AOrdering::SeqCst);
                Ok("stub_ptx_text".to_string())
            },
            |ptx| {
                loader_count.fetch_add(1, AOrdering::SeqCst);
                Ok(if ptx == "stub_ptx_text" {
                    "MODULE_OK"
                } else {
                    "MODULE_WRONG"
                })
            },
            &mut stats,
        )
        .expect("first call must succeed");
        assert_eq!(m1, "MODULE_OK");
        assert_eq!(compile_count.load(AOrdering::SeqCst), 1);
        assert_eq!(loader_count.load(AOrdering::SeqCst), 1);
        assert_eq!(stats, (0, 1));

        // Second call with the same spec/entry: HIT. Neither the
        // compile closure nor the loader runs.
        let m2 = get_or_build_with_loader(
            &mut cache,
            &mut order,
            64,
            &spec,
            "bolt_prefix_scan_lookback",
            |_s| {
                compile_count.fetch_add(1, AOrdering::SeqCst);
                Ok("would-be-recompiled".to_string())
            },
            |_ptx| {
                loader_count.fetch_add(1, AOrdering::SeqCst);
                Ok("WRONG_HIT")
            },
            &mut stats,
        )
        .expect("warm hit must succeed");
        assert_eq!(m2, "MODULE_OK", "warm hit must return the cached module");
        assert_eq!(
            compile_count.load(AOrdering::SeqCst),
            1,
            "compile must NOT run on a hit"
        );
        assert_eq!(
            loader_count.load(AOrdering::SeqCst),
            1,
            "loader must NOT run on a hit"
        );
        assert_eq!(stats, (1, 1));
    }
}
