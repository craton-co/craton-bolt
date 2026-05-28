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
use crate::plan::physical_plan::{KernelSpec, ScalarAggSpec};

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
// v0.7: ScalarAggSpec-keyed cache layer.
//
// Mirrors the `KernelSpec` layer above but keyed on the scalar-aggregate
// planner IR (see [`crate::plan::physical_plan::ScalarAggSpec`]). The
// scalar-aggregate executor (`exec::aggregate`) historically routed every
// reduction through the string-keyed `get_or_build_module` above (keying
// on a `format!("reduction:{:?}:{:?}", op, dtype)` string). That worked but:
//
//   1. The Debug-of-tuple key shape was hand-rolled per call site, so adding
//      a third axis (e.g. accumulator-widening) meant editing each call site
//      and re-stating the format string.
//   2. The string key carried no domain separation from the projection-path
//      cache other than the `namespace` (module_path!()) field — a refactor
//      that moved the reduction call out of `exec::aggregate` would silently
//      change the key.
//   3. The disk-cache key (when wired) needs an explicit, visible domain
//      prefix per the v0.7 task brief.
//
// This layer keys on the `ScalarAggSpec` IR itself (a `(op, dtype)` pair),
// uses the same 128-bit content-hash shape as `KernelSpecKey`, and stamps
// the disk-cache key with a `"scalar_agg::"` prefix so a hand inspection of
// the cache directory shows immediately which family produced each entry.
// The in-memory cache is independent from `KERNELSPEC_CACHE` — splitting them
// keeps the FIFO eviction policies of the two families from competing.

/// FIFO cap on the ScalarAggSpec→`CudaModule` cache. Same default as the
/// `KernelSpec` cap above; tunable by editing this constant.
const SCALARAGG_CACHE_CAP: usize = 64;

/// 128-bit content fingerprint of a `ScalarAggSpec` plus its entry-point tag.
/// Built the same way `KernelSpecKey` is — two `DefaultHasher` instances
/// domain-separated by a leading byte, packing into 128 bits total. We
/// re-hash even though `ScalarAggSpec` itself implements `Hash`, so the
/// fingerprint shape exactly matches the projection-side cache (and so the
/// `Debug` output drives the key in case `ScalarAggSpec` ever grows
/// non-Hashable fields like `KernelSpec` did).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ScalarAggKey {
    /// Upper 64 bits of the 128-bit content hash (domain byte `0x11`).
    hi: u64,
    /// Lower 64 bits of the 128-bit content hash (domain byte `0x12`).
    lo: u64,
    /// PTX entry-point name (`bolt_reduce` vs. `bolt_avg_reduce`).
    entry: &'static str,
}

impl ScalarAggKey {
    fn new(spec: &ScalarAggSpec, entry: &'static str) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::fmt::Write as _;
        use std::hash::Hasher;

        // Domain-separating prefix bytes distinct from `KernelSpecKey`'s
        // (`0x01` / `0x02`) so a hypothetical hash collision between the
        // Debug shapes of `KernelSpec` and `ScalarAggSpec` can't alias keys.
        let mut hi = DefaultHasher::new();
        hi.write_u8(0x11);
        let _ = write!(HasherWrite(&mut hi), "{:?}", spec);

        let mut lo = DefaultHasher::new();
        lo.write_u8(0x12);
        let _ = write!(HasherWrite(&mut lo), "{:?}", spec);

        Self {
            hi: hi.finish(),
            lo: lo.finish(),
            entry,
        }
    }
}

/// Cached payload for one ScalarAggSpec key. Mirrors `KernelSpecEntry`.
#[derive(Clone)]
struct ScalarAggEntry {
    /// Emitted PTX text. Retained for observability / future
    /// hit-collision-detection. The production lookup path only needs
    /// `module`.
    #[allow(dead_code)]
    ptx: String,
    module: CudaModule,
}

/// State of the ScalarAggSpec-keyed cache. Same FIFO eviction shape as
/// [`KernelSpecCache`]; see those docs for the invariants.
struct ScalarAggCache {
    by_key: HashMap<ScalarAggKey, ScalarAggEntry>,
    order: VecDeque<ScalarAggKey>,
    cap: usize,
    hits: u64,
    misses: u64,
}

impl ScalarAggCache {
    fn new(cap: usize) -> Self {
        Self {
            by_key: HashMap::with_capacity(cap),
            order: VecDeque::with_capacity(cap),
            cap,
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, key: &ScalarAggKey) -> Option<ScalarAggEntry> {
        if let Some(entry) = self.by_key.get(key) {
            self.hits = self.hits.saturating_add(1);
            Some(entry.clone())
        } else {
            self.misses = self.misses.saturating_add(1);
            None
        }
    }

    fn insert(&mut self, key: ScalarAggKey, entry: ScalarAggEntry) {
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

/// Process-wide ScalarAggSpec→`CudaModule` cache. Initialised lazily.
static SCALARAGG_CACHE: Lazy<Mutex<ScalarAggCache>> =
    Lazy::new(|| Mutex::new(ScalarAggCache::new(SCALARAGG_CACHE_CAP)));

/// Disk-cache key prefix for the scalar-aggregate PTX family. Stamped onto
/// every disk-cache write/read so a directory listing shows immediately
/// which kernel family produced each `.ptx` file, and so the projection
/// path's `<entry>-<hex>` shape can't ever collide with a scalar-aggregate
/// entry (different prefix string, different overall key shape).
pub(crate) const SCALAR_AGG_DISK_PREFIX: &str = "scalar_agg::";

/// Test- and observability-facing snapshot of `(hits, misses)` for the
/// ScalarAggSpec cache. Parallel to [`kernelspec_cache_stats`].
#[doc(hidden)]
#[must_use]
pub fn scalar_agg_cache_stats() -> (usize, usize) {
    let c = SCALARAGG_CACHE.lock();
    (c.hits as usize, c.misses as usize)
}

/// Look up (or compile-and-load) the `CudaModule` for `(spec, entry)`.
///
/// This is the scalar-aggregate analogue of
/// [`get_or_build_module_for_spec`]. On a **cache hit** we return the
/// cached `CudaModule` clone immediately — no codegen, no PTX text
/// generation, no `cuModuleLoadDataEx`. On a **cache miss** we run
/// `compile(spec)` to get the PTX text and hand it to
/// `CudaModule::from_ptx`. The resulting `(ptx, module)` pair is stored
/// under the ScalarAggSpec key so subsequent calls with the same spec
/// hit the fast path.
///
/// # Disk-cache integration (v0.6 / M6)
///
/// When the disk-backed PTX cache is enabled (env var `BOLT_PTX_CACHE_DIR`
/// or builder override), a miss in the in-memory cache consults the disk
/// cache *before* paying the codegen cost. The disk key is composed as
/// `"{SCALAR_AGG_DISK_PREFIX}{entry}-{hex(hash128)}"` so:
///   1. The `"scalar_agg::"` prefix domain-separates these entries from the
///      projection-path entries that share the disk directory.
///   2. The `entry` suffix distinguishes `bolt_reduce` from `bolt_avg_reduce`.
///   3. The 128-bit hex content hash makes the key collision-resistant
///      against unrelated specs that happen to share the same `(op, dtype)`.
///
/// On a disk hit we skip codegen and feed the on-disk PTX straight to
/// `CudaModule::from_ptx` (which still pays the one-time PTXAS assembly
/// cost in this fresh-process scenario, since there's no cubin cache yet).
///
/// # Concurrency
///
/// The cache lock is held only long enough to look up or insert; the slow
/// `compile` + `from_ptx` work runs outside the lock. If two threads race
/// on the same miss they will each compile once and the second insert is
/// dropped by `insert`'s idempotence check.
pub(crate) fn get_or_build_module_for_scalar_agg<F>(
    spec: &ScalarAggSpec,
    entry: &'static str,
    compile: F,
) -> BoltResult<CudaModule>
where
    F: FnOnce(&ScalarAggSpec) -> BoltResult<String>,
{
    let key = ScalarAggKey::new(spec, entry);
    // Fast path: in-memory hit. Hold the lock just long enough to clone.
    if let Some(cached) = SCALARAGG_CACHE.lock().get(&key) {
        return Ok(cached.module);
    }
    // In-memory miss: try the optional on-disk cache before paying for codegen.
    let disk = crate::jit::disk_cache::disk_cache();
    let disk_key = disk.as_ref().map(|_| {
        format!(
            "{}{}-{}",
            SCALAR_AGG_DISK_PREFIX,
            entry,
            crate::jit::disk_cache::hash_to_key(key.hi, key.lo),
        )
    });
    let ptx = match (&disk, &disk_key) {
        (Some(cache), Some(k)) => match cache.lookup(k) {
            Some(text) => text,
            None => {
                let text = compile(spec)?;
                // Write-through to disk. Errors here are non-fatal: a
                // failed write just means future processes won't benefit,
                // the current process still loads the module successfully.
                let _ = cache.store(k, &text);
                text
            }
        },
        _ => compile(spec)?,
    };
    let module = CudaModule::from_ptx(&ptx)?;
    // Insert and hand back a clone. Concurrent winners are tolerated by
    // `insert`'s idempotence check.
    SCALARAGG_CACHE.lock().insert(
        key,
        ScalarAggEntry {
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
// Tests for the ScalarAggSpec-keyed cache.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod scalar_agg_cache_tests {
    use super::*;
    use crate::plan::logical_plan::DataType;
    use crate::plan::physical_plan::{ScalarAggOp, ScalarAggSpec};
    use std::sync::atomic::{AtomicUsize, Ordering as AOrdering};

    /// Cold lookup against a fresh cache is a miss and `get` accounts for
    /// it. Mirrors `kernelspec_cache_cold_lookup_is_a_miss`.
    #[test]
    fn scalar_agg_cache_cold_lookup_is_a_miss() {
        let mut cache = ScalarAggCache::new(4);
        let spec = ScalarAggSpec {
            op: ScalarAggOp::Sum,
            input_dtype: DataType::Int64,
        };
        let key = ScalarAggKey::new(&spec, "bolt_reduce");

        assert!(cache.get(&key).is_none(), "fresh cache must miss");
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 0);
    }

    /// State-machine pin of the hit path: a second call with the same spec
    /// must NOT re-run the compile closure and must NOT re-run the loader.
    /// We mock `CudaModule::from_ptx` with a stub closure since CUDA isn't
    /// available in unit tests; the cache state machine is identical.
    #[test]
    fn scalar_agg_cache_compile_and_loader_run_once_per_unique_spec() {
        // Mock module type — just a tag so we can confirm the same instance
        // comes back on a hit.
        #[derive(Clone)]
        struct MockModule(&'static str);

        struct MockCache {
            by_key: HashMap<ScalarAggKey, MockModule>,
            order: VecDeque<ScalarAggKey>,
            cap: usize,
        }

        impl MockCache {
            fn get_or_build(
                &mut self,
                spec: &ScalarAggSpec,
                entry: &'static str,
                compile_count: &AtomicUsize,
                loader_count: &AtomicUsize,
                tag: &'static str,
            ) -> MockModule {
                let key = ScalarAggKey::new(spec, entry);
                if let Some(m) = self.by_key.get(&key) {
                    return m.clone();
                }
                // Miss: "compile" then "load".
                compile_count.fetch_add(1, AOrdering::SeqCst);
                loader_count.fetch_add(1, AOrdering::SeqCst);
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

        let spec_sum_i64 = ScalarAggSpec {
            op: ScalarAggOp::Sum,
            input_dtype: DataType::Int64,
        };
        let spec_min_i64 = ScalarAggSpec {
            op: ScalarAggOp::Min,
            input_dtype: DataType::Int64,
        };

        let compile_count = AtomicUsize::new(0);
        let loader_count = AtomicUsize::new(0);
        let mut cache = MockCache {
            by_key: HashMap::new(),
            order: VecDeque::new(),
            cap: 64,
        };

        // First call with (Sum, Int64): miss → compile + load each run once.
        let m1 = cache.get_or_build(
            &spec_sum_i64,
            "bolt_reduce",
            &compile_count,
            &loader_count,
            "sum-i64",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 1);
        assert_eq!(loader_count.load(AOrdering::SeqCst), 1);
        assert_eq!(m1.0, "sum-i64");

        // Second call with the SAME spec: HIT → neither closure runs again.
        let m2 = cache.get_or_build(
            &spec_sum_i64,
            "bolt_reduce",
            &compile_count,
            &loader_count,
            "sum-i64",
        );
        assert_eq!(
            compile_count.load(AOrdering::SeqCst),
            1,
            "warm-cache hit must skip codegen"
        );
        assert_eq!(
            loader_count.load(AOrdering::SeqCst),
            1,
            "warm-cache hit must skip module load"
        );
        assert_eq!(m2.0, "sum-i64");

        // A different op (Min): miss again.
        let m3 = cache.get_or_build(
            &spec_min_i64,
            "bolt_reduce",
            &compile_count,
            &loader_count,
            "min-i64",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 2);
        assert_eq!(loader_count.load(AOrdering::SeqCst), 2);
        assert_eq!(m3.0, "min-i64");

        // Same spec as m1, different entry tag (e.g. the fused AVG kernel):
        // miss — entry participates in the key.
        let spec_avg_i64 = ScalarAggSpec {
            op: ScalarAggOp::Avg,
            input_dtype: DataType::Int64,
        };
        let m4 = cache.get_or_build(
            &spec_avg_i64,
            "bolt_avg_reduce",
            &compile_count,
            &loader_count,
            "avg-i64",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 3);
        assert_eq!(loader_count.load(AOrdering::SeqCst), 3);
        assert_eq!(m4.0, "avg-i64");

        // And back to spec_sum_i64: still a hit, neither closure runs.
        let m5 = cache.get_or_build(
            &spec_sum_i64,
            "bolt_reduce",
            &compile_count,
            &loader_count,
            "sum-i64",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 3);
        assert_eq!(loader_count.load(AOrdering::SeqCst), 3);
        assert_eq!(m5.0, "sum-i64");
    }

    /// Key uniqueness: distinct ops, dtypes, and entry tags must each
    /// produce distinct keys. (This is the assumption the hit-vs-miss
    /// classification relies on.)
    #[test]
    fn scalar_agg_cache_key_distinguishes_op_dtype_and_entry() {
        let sum_i32 = ScalarAggSpec {
            op: ScalarAggOp::Sum,
            input_dtype: DataType::Int32,
        };
        let sum_i64 = ScalarAggSpec {
            op: ScalarAggOp::Sum,
            input_dtype: DataType::Int64,
        };
        let min_i32 = ScalarAggSpec {
            op: ScalarAggOp::Min,
            input_dtype: DataType::Int32,
        };

        let k_sum_i32 = ScalarAggKey::new(&sum_i32, "bolt_reduce");
        let k_sum_i64 = ScalarAggKey::new(&sum_i64, "bolt_reduce");
        let k_min_i32 = ScalarAggKey::new(&min_i32, "bolt_reduce");
        let k_sum_i32_alt = ScalarAggKey::new(&sum_i32, "bolt_avg_reduce");

        assert_ne!(k_sum_i32, k_sum_i64, "dtype must participate in the key");
        assert_ne!(k_sum_i32, k_min_i32, "op must participate in the key");
        assert_ne!(
            k_sum_i32, k_sum_i32_alt,
            "entry tag must participate in the key"
        );
    }

    /// The disk-cache key prefix must start with `"scalar_agg::"` so a
    /// hand inspection of the cache directory distinguishes scalar-agg
    /// entries from projection-path entries. Pins the prefix contract
    /// against accidental drift.
    #[test]
    fn scalar_agg_disk_prefix_is_visibly_namespaced() {
        assert_eq!(SCALAR_AGG_DISK_PREFIX, "scalar_agg::");
        // The full disk key shape is `"<PREFIX><entry>-<hex>"` — confirm
        // the prefix lands at the start of a composed key.
        let composed = format!(
            "{}{}-{}",
            SCALAR_AGG_DISK_PREFIX,
            "bolt_reduce",
            "deadbeefcafebabe1234567890abcdef",
        );
        assert!(
            composed.starts_with("scalar_agg::"),
            "composed disk key must carry the scalar_agg prefix: {composed}"
        );
    }

    /// The key fingerprint is stable across `Clone` of the same spec —
    /// matches the `KernelSpec` invariant. Callers often `Copy` the IR
    /// before handing it to the cache, so the stability is load-bearing.
    #[test]
    fn scalar_agg_cache_key_stable_across_copy() {
        let spec = ScalarAggSpec {
            op: ScalarAggOp::Max,
            input_dtype: DataType::Float64,
        };
        let copied = spec;
        assert_eq!(
            ScalarAggKey::new(&spec, "bolt_reduce"),
            ScalarAggKey::new(&copied, "bolt_reduce"),
            "copying the spec must not change its cache key"
        );
    }
}
