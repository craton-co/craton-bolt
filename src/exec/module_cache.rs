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
use crate::plan::physical_plan::{
    CompactionKernelSpec, HashJoinKernelSpec, KernelSpec, RadixSortKernelSpec, ScalarAggSpec,
};

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

/// `fmt::Write` → `Hasher` adapter so we can stream `Debug` output of a
/// spec directly into a hasher with zero heap allocation.
struct HasherWrite<'a, H: std::hash::Hasher>(&'a mut H);

impl<H: std::hash::Hasher> std::fmt::Write for HasherWrite<'_, H> {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        self.0.write(s.as_bytes());
        Ok(())
    }
}

/// Compute the 128-bit content fingerprint `(hi, lo)` of a spec's `Debug`
/// shape, domain-separated by a pair of leading bytes.
///
/// This is the shared body of every `*Key::new` below: each family passes
/// its own domain bytes (`0x01`/`0x02`, `0x11`/`0x12`, …) so that a
/// hypothetical collision between the `Debug` shapes of two different spec
/// types can never alias cache keys across families. The per-family domain
/// bytes are part of the on-disk cache key contract and MUST NOT change.
fn hash128<S: std::fmt::Debug + ?Sized>(spec: &S, hi_byte: u8, lo_byte: u8) -> (u64, u64) {
    use std::collections::hash_map::DefaultHasher;
    use std::fmt::Write as _;
    use std::hash::Hasher;

    let mut hi = DefaultHasher::new();
    hi.write_u8(hi_byte);
    // Both arms are unreachable in practice; degrade to a benign cache miss
    // rather than panic if Debug formatting ever fails.
    let _ = write!(HasherWrite(&mut hi), "{:?}", spec);

    let mut lo = DefaultHasher::new();
    lo.write_u8(lo_byte);
    let _ = write!(HasherWrite(&mut lo), "{:?}", spec);

    (hi.finish(), lo.finish())
}

/// Trait implemented by every per-family content-hash key
/// (`KernelSpecKey`, `ScalarAggKey`, …). Exposes the 128-bit fingerprint
/// and the PTX entry name so the shared disk-cache key composer can build
/// `"{codegen_salt}-{prefix}{entry}-{hex(hash128)}"` without knowing the
/// concrete family.
trait CacheKey: Copy + Eq + std::hash::Hash {
    fn hi(&self) -> u64;
    fn lo(&self) -> u64;
    fn entry(&self) -> &'static str;
}

/// Cached payload for one spec key: the emitted PTX text and the loaded
/// module. The PTX text is retained so callers that want to inspect it
/// (e.g. for tracing) can do so without re-running codegen, and so a
/// future hit-collision-detection path can compare-by-content if needed.
#[derive(Clone)]
struct CacheEntry {
    /// Emitted PTX text. Retained for observability / future
    /// hit-collision-detection. `#[allow(dead_code)]` because the
    /// production lookup path only needs `module` — but storing the PTX
    /// here is part of the cache contract, so we don't want a future
    /// refactor to drop the field by mistake.
    #[allow(dead_code)]
    ptx: String,
    module: CudaModule,
}

/// Generic process-wide spec→`(PTX, CudaModule)` cache with FIFO eviction.
///
/// One instance backs each kernel family (`KernelSpecCache`,
/// `ScalarAggCache`, …) via the per-family type aliases below; the only
/// per-family difference is the key type `K`, the capacity, and (for the
/// disk-backed families) the disk-cache prefix supplied at the call site.
///
/// `by_key` is the primary lookup; `order` is a parallel FIFO log used for
/// eviction. The two are kept in sync by `insert` (the only mutator). On
/// eviction we pop the front of `order` and remove the matching map entry.
struct SpecCache<K: CacheKey> {
    by_key: HashMap<K, CacheEntry>,
    order: VecDeque<K>,
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

impl<K: CacheKey> SpecCache<K> {
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
    fn get(&mut self, key: &K) -> Option<CacheEntry> {
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
    fn insert(&mut self, key: K, entry: CacheEntry) {
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

/// Compose the on-disk PTX cache key for a disk-backed kernel family.
///
/// Shape: `"{codegen_salt}-{disk_prefix}{entry}-{hex(hash128)}"`.
///
/// * The leading [`codegen_salt`](crate::jit::disk_cache::codegen_salt) is the
///   JIT-M1 guard: it folds in `CODEGEN_VERSION` + the crate version so any
///   codegen change rotates the key. A populated cache dir written by an older
///   binary then *misses* (forcing a recompile) instead of serving stale PTX
///   whose `KernelSpec` hash happens to be unchanged. This matches the
///   engine.rs scalar/projection path, which uses
///   [`disk_cache::disk_key`](crate::jit::disk_cache::disk_key) — the only
///   difference is that path has no per-family `disk_prefix`.
/// * The `{disk_prefix}{entry}-{hex(hash128)}` tail is the historical
///   domain-separated shape, preserved byte-for-byte: the prefix distinguishes
///   kernel families (`scalar_agg__`, `hash_join__`, …; V-3: `__` not `::`,
///   so composed keys stay inside the filename-safe charset), `entry` distinguishes
///   symbols, and the spec hash distinguishes IR. Only the salt is prepended.
///
/// The disk key string is internal to [`get_or_build_with_disk`]; no other
/// caller depends on its shape, so prepending the salt only rotates which
/// on-disk entries are considered fresh.
fn compose_disk_key(disk_prefix: &str, entry: &str, hi: u64, lo: u64) -> String {
    format!(
        "{}-{}{}-{}",
        crate::jit::disk_cache::codegen_salt(),
        disk_prefix,
        entry,
        crate::jit::disk_cache::hash_to_key(hi, lo),
    )
}

/// Shared in-memory-then-disk fall-through used by every disk-backed
/// family's `get_or_build_module_for_*`. On a hit in `cache` we return the
/// cached module; on a miss we consult the optional on-disk PTX cache
/// (keyed by [`compose_disk_key`]:
/// `"{codegen_salt}-{disk_prefix}{entry}-{hex(hash128)}"`) before paying for
/// codegen, write-through any freshly compiled PTX, load it via the real
/// `CudaModule::from_ptx`, and store the result. Behaviour (lock scope,
/// write-through, insert idempotence) is what each family open-coded before
/// this consolidation; the codegen salt was added for JIT-M1.
fn get_or_build_with_disk<K, F>(
    cache: &Lazy<Mutex<SpecCache<K>>>,
    key: K,
    disk_prefix: &str,
    compile: F,
) -> BoltResult<CudaModule>
where
    K: CacheKey,
    F: FnOnce() -> BoltResult<String>,
{
    // Fast path: in-memory hit. Hold the lock just long enough to clone.
    if let Some(cached) = cache.lock().get(&key) {
        return Ok(cached.module);
    }
    // In-memory miss: try the optional on-disk cache before paying for codegen.
    let disk = crate::jit::disk_cache::disk_cache();
    let disk_key = disk
        .as_ref()
        .map(|_| compose_disk_key(disk_prefix, key.entry(), key.hi(), key.lo()));
    let ptx = match (&disk, &disk_key) {
        (Some(dcache), Some(k)) => match dcache.lookup(k) {
            Some(text) => text,
            None => {
                let text = compile()?;
                // Write-through to disk. Errors here are non-fatal: a failed
                // write just means future processes won't benefit, the
                // current process still loads the module successfully.
                let _ = dcache.store(k, &text);
                text
            }
        },
        _ => compile()?,
    };
    let module = CudaModule::from_ptx(&ptx)?;
    // Insert and hand back a clone. Concurrent winners are tolerated by
    // `insert`'s idempotence check.
    cache.lock().insert(
        key,
        CacheEntry {
            ptx,
            module: module.clone(),
        },
    );
    Ok(module)
}

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

impl KernelSpecKey {
    /// Compute the key for `(spec, entry)`. See type docs for rationale.
    fn new(spec: &KernelSpec, entry: &'static str) -> Self {
        let (hi, lo) = hash128(spec, 0x01, 0x02);
        Self { hi, lo, entry }
    }
}

impl CacheKey for KernelSpecKey {
    fn hi(&self) -> u64 {
        self.hi
    }
    fn lo(&self) -> u64 {
        self.lo
    }
    fn entry(&self) -> &'static str {
        self.entry
    }
}

/// State of the KernelSpec-keyed cache. See [`SpecCache`] for the FIFO
/// eviction invariants.
type KernelSpecCache = SpecCache<KernelSpecKey>;

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
    get_or_build_module_for_spec_with(spec, entry, compile, CudaModule::from_ptx)
}

/// Shared cache logic, parameterised over the PTX-text → `CudaModule`
/// loader. Production code supplies `CudaModule::from_ptx` (the real
/// driver path); tests inject a stub loader that returns a fake module
/// without touching CUDA. The shape mirrors
/// `crate::jit::jit_compiler::CudaModule::from_ptx_with` so the two
/// caches stay test-symmetric.
///
/// `pub(crate)` so a future cross-module test (e.g. one that wires the
/// projection path against a stub) can drive this without a real GPU,
/// but invisible in the public API.
pub(crate) fn get_or_build_module_for_spec_with<F, L>(
    spec: &KernelSpec,
    entry: &'static str,
    compile: F,
    loader: L,
) -> BoltResult<CudaModule>
where
    F: FnOnce(&KernelSpec) -> BoltResult<String>,
    L: FnOnce(&str) -> BoltResult<CudaModule>,
{
    let key = KernelSpecKey::new(spec, entry);
    // Fast path: hit.
    if let Some(entry) = KERNELSPEC_CACHE.lock().get(&key) {
        return Ok(entry.module);
    }
    // Miss: run codegen + module load WITHOUT the cache lock held. Both
    // steps can be slow and we don't want to serialise unrelated misses.
    let ptx = compile(spec)?;
    let module = loader(&ptx)?;
    // Store under the KernelSpec key so the next call skips codegen.
    KERNELSPEC_CACHE.lock().insert(
        key,
        CacheEntry {
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
// the disk-cache key with a `"scalar_agg__"` prefix so a hand inspection of
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
        // Domain-separating prefix bytes distinct from `KernelSpecKey`'s
        // (`0x01` / `0x02`) so a hypothetical hash collision between the
        // Debug shapes of `KernelSpec` and `ScalarAggSpec` can't alias keys.
        let (hi, lo) = hash128(spec, 0x11, 0x12);
        Self { hi, lo, entry }
    }
}

impl CacheKey for ScalarAggKey {
    fn hi(&self) -> u64 {
        self.hi
    }
    fn lo(&self) -> u64 {
        self.lo
    }
    fn entry(&self) -> &'static str {
        self.entry
    }
}

/// State of the ScalarAggSpec-keyed cache. See [`SpecCache`] for the FIFO
/// eviction invariants.
type ScalarAggCache = SpecCache<ScalarAggKey>;

/// Process-wide ScalarAggSpec→`CudaModule` cache. Initialised lazily.
static SCALARAGG_CACHE: Lazy<Mutex<ScalarAggCache>> =
    Lazy::new(|| Mutex::new(ScalarAggCache::new(SCALARAGG_CACHE_CAP)));

/// Disk-cache key prefix for the scalar-aggregate PTX family. Stamped onto
/// every disk-cache write/read so a directory listing shows immediately
/// which kernel family produced each `.ptx` file, and so the projection
/// path's `<entry>-<hex>` shape can't ever collide with a scalar-aggregate
/// entry (different prefix string, different overall key shape).
///
/// V-3 (path-traversal hardening): the separator is `__` (double
/// underscore), NOT `::`. The disk-cache layer now validates every key
/// against a strict filename-safe charset (`^[0-9A-Za-z._-]+$`, see
/// `jit::disk_cache::valid_key`) and rejects anything else as a cache
/// miss / store no-op. A `:` is not in that charset (and is actively
/// dangerous on Windows — drive-letter / NTFS alternate-data-stream
/// syntax), so a `::`-separated prefix would make every scalar-agg entry
/// fail validation and silently disable this family's disk cache. `__`
/// keeps the directory listing just as human-greppable while staying
/// inside the allowed charset. Changing the separator rotates the on-disk
/// key shape, which is harmless: stale `::` entries (if any) simply miss
/// and codegen re-runs (same contract as the codegen salt).
pub(crate) const SCALAR_AGG_DISK_PREFIX: &str = "scalar_agg__";

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
/// `"{codegen_salt}-{SCALAR_AGG_DISK_PREFIX}{entry}-{hex(hash128)}"` so:
///   1. The `"scalar_agg__"` prefix domain-separates these entries from the
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
    get_or_build_with_disk(&SCALARAGG_CACHE, key, SCALAR_AGG_DISK_PREFIX, || {
        compile(spec)
    })
}

// ---------------------------------------------------------------------------
// v0.7: HashJoinKernelSpec-keyed cache layer.
//
// Mirrors the `ScalarAggSpec` layer above but keyed on the hash-join planner
// IR (see [`crate::plan::physical_plan::HashJoinKernelSpec`]). Every
// `compile_*_kernel` helper in `crate::jit::hash_join_kernel` takes no
// arguments and returns a fixed PTX string for a fixed entry symbol; the
// codegen-time knob is therefore which helper to call. The wrapper here
// maps a `HashJoinKernelSpec` (kind + key_dtype + string_hash_returns_i64)
// to the corresponding helper and routes the resulting PTX through the
// existing [`CudaModule::from_ptx`] pipeline.
//
// As with `ScalarAggSpec`, we domain-separate the disk-cache key with a
// `"hash_join__"` prefix so a hand inspection of the cache directory shows
// immediately which family produced each entry, and we keep the in-memory
// cache independent from `KERNELSPEC_CACHE` / `SCALARAGG_CACHE` so the FIFO
// eviction policies of the three families don't compete.

/// FIFO cap on the HashJoinKernelSpec→`CudaModule` cache. Matches the
/// `SCALARAGG_CACHE_CAP` default.
const HASHJOIN_CACHE_CAP: usize = 64;

/// 128-bit content fingerprint of a `HashJoinKernelSpec` plus its
/// entry-point tag. Built the same way `ScalarAggKey` is — two
/// `DefaultHasher` instances domain-separated by a leading byte, packing
/// into 128 bits total. We re-hash even though `HashJoinKernelSpec` itself
/// implements `Hash`, so the fingerprint shape exactly matches the
/// projection-side and scalar-agg caches (and so the `Debug` output drives
/// the key in case `HashJoinKernelSpec` ever grows non-Hashable fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct HashJoinKey {
    /// Upper 64 bits of the 128-bit content hash (domain byte `0x21`).
    hi: u64,
    /// Lower 64 bits of the 128-bit content hash (domain byte `0x22`).
    lo: u64,
    /// PTX entry-point name (e.g. `bolt_build` / `bolt_probe` / `bolt_string_hash`).
    entry: &'static str,
}

impl HashJoinKey {
    fn new(spec: &HashJoinKernelSpec, entry: &'static str) -> Self {
        // Domain-separating prefix bytes distinct from `KernelSpecKey`'s
        // (`0x01` / `0x02`) and `ScalarAggKey`'s (`0x11` / `0x12`) so a
        // hypothetical hash collision between the Debug shapes of the three
        // spec types can't alias keys.
        let (hi, lo) = hash128(spec, 0x21, 0x22);
        Self { hi, lo, entry }
    }
}

impl CacheKey for HashJoinKey {
    fn hi(&self) -> u64 {
        self.hi
    }
    fn lo(&self) -> u64 {
        self.lo
    }
    fn entry(&self) -> &'static str {
        self.entry
    }
}

/// State of the HashJoinKernelSpec-keyed cache. See [`SpecCache`] for the
/// FIFO eviction invariants.
type HashJoinCache = SpecCache<HashJoinKey>;

/// Process-wide HashJoinKernelSpec→`CudaModule` cache. Initialised lazily.
static HASHJOIN_CACHE: Lazy<Mutex<HashJoinCache>> =
    Lazy::new(|| Mutex::new(HashJoinCache::new(HASHJOIN_CACHE_CAP)));

/// Disk-cache key prefix for the hash-join PTX family. Stamped onto every
/// disk-cache write/read so a directory listing shows immediately which
/// kernel family produced each `.ptx` file, and so the projection /
/// scalar-aggregate paths' key shapes can't ever collide with a hash-join
/// entry.
///
/// V-3: `__` separator (not `::`) — see [`SCALAR_AGG_DISK_PREFIX`] for the
/// path-traversal-hardening rationale (`:` is outside the validated
/// filename-safe charset).
pub(crate) const HASH_JOIN_DISK_PREFIX: &str = "hash_join__";

/// Test- and observability-facing snapshot of `(hits, misses)` for the
/// HashJoinKernelSpec cache. Parallel to [`scalar_agg_cache_stats`].
#[doc(hidden)]
#[must_use]
pub fn hash_join_cache_stats() -> (usize, usize) {
    let c = HASHJOIN_CACHE.lock();
    (c.hits as usize, c.misses as usize)
}

/// Look up (or compile-and-load) the `CudaModule` for `(spec, entry)`.
///
/// This is the hash-join analogue of [`get_or_build_module_for_scalar_agg`].
/// On a **cache hit** we return the cached `CudaModule` clone immediately —
/// no codegen, no PTX text generation, no `cuModuleLoadDataEx`. On a
/// **cache miss** we run `compile(spec)` to get the PTX text and hand it to
/// `CudaModule::from_ptx`. The resulting `(ptx, module)` pair is stored
/// under the HashJoinKernelSpec key so subsequent calls with the same spec
/// hit the fast path.
///
/// # Disk-cache integration
///
/// When the disk-backed PTX cache is enabled (env var `BOLT_PTX_CACHE_DIR`
/// or builder override), a miss in the in-memory cache consults the disk
/// cache *before* paying the codegen cost. The disk key is composed as
/// `"{codegen_salt}-{HASH_JOIN_DISK_PREFIX}{entry}-{hex(hash128)}"` so:
///   1. The `"hash_join__"` prefix domain-separates these entries from the
///      projection-path and scalar-aggregate entries sharing the directory.
///   2. The `entry` suffix distinguishes `bolt_build`, `bolt_probe`,
///      `bolt_build_aos`, `bolt_string_hash`, etc.
///   3. The 128-bit hex content hash makes the key collision-resistant
///      against unrelated specs that happen to share the same kind +
///      dtype tuple.
///
/// On a disk hit we skip codegen and feed the on-disk PTX straight to
/// `CudaModule::from_ptx`.
///
/// # Concurrency
///
/// The cache lock is held only long enough to look up or insert; the slow
/// `compile` + `from_ptx` work runs outside the lock. If two threads race
/// on the same miss they will each compile once and the second insert is
/// dropped by `insert`'s idempotence check.
pub(crate) fn get_or_build_module_for_hash_join<F>(
    spec: &HashJoinKernelSpec,
    entry: &'static str,
    compile: F,
) -> BoltResult<CudaModule>
where
    F: FnOnce(&HashJoinKernelSpec) -> BoltResult<String>,
{
    let key = HashJoinKey::new(spec, entry);
    get_or_build_with_disk(&HASHJOIN_CACHE, key, HASH_JOIN_DISK_PREFIX, || compile(spec))
}

// ---------------------------------------------------------------------------
// v0.7: RadixSortKernelSpec-keyed cache layer.
//
// Mirrors the `ScalarAggSpec` / `HashJoinKernelSpec` layers above but keyed
// on the radix-sort planner IR (see
// [`crate::plan::physical_plan::RadixSortKernelSpec`]). Each per-pass radix
// kernel in `crate::jit::sort_kernel_radix` takes a `DataType` argument and
// returns a fixed PTX string for a fixed entry symbol; the codegen-time
// knobs are exactly `(pass, dtype)`. The wrapper here maps a
// `RadixSortKernelSpec` to the corresponding helper (the caller picks
// which `compile_radix_*` to invoke based on `spec.pass`) and routes the
// resulting PTX through the existing [`CudaModule::from_ptx`] pipeline.
//
// As with the sibling caches, we domain-separate the disk-cache key with
// a `"radix_sort__"` prefix so a hand inspection of the cache directory
// shows immediately which family produced each entry, and we keep the
// in-memory cache independent from `KERNELSPEC_CACHE` / `SCALARAGG_CACHE`
// / `HASHJOIN_CACHE` so the FIFO eviction policies of the four families
// don't compete.
//
// # Why `entry: &str` participates in the key
//
// One `RadixSortKernelSpec { pass: Scatter, .. }` value can map to *two*
// distinct PTX entry points in the executor: `bolt_radix_scatter_<dty>`
// (keys-only) vs `bolt_radix_scatter_<dty>_with_indices` (keys + row-index
// payload — the standard multi-column ORDER BY path). The `pass` field
// now has a dedicated `ScatterWithIndices` variant for the latter, but
// the `entry: &str` parameter is retained as the load-bearing
// disambiguator so a future ABI variant doesn't have to grow the IR enum
// just to claim its own cache slot.

/// FIFO cap on the RadixSortKernelSpec→`CudaModule` cache. Matches the
/// `SCALARAGG_CACHE_CAP` / `HASHJOIN_CACHE_CAP` default.
const RADIX_SORT_CACHE_CAP: usize = 64;

/// 128-bit content fingerprint of a `RadixSortKernelSpec` plus its
/// entry-point tag. Built the same way `HashJoinKey` is — two
/// `DefaultHasher` instances domain-separated by a leading byte, packing
/// into 128 bits total. We re-hash even though `RadixSortKernelSpec` itself
/// implements `Hash`, so the fingerprint shape exactly matches the
/// projection-side / scalar-agg / hash-join caches (and so the `Debug`
/// output drives the key in case `RadixSortKernelSpec` ever grows
/// non-Hashable fields).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RadixSortKey {
    /// Upper 64 bits of the 128-bit content hash (domain byte `0x31`).
    hi: u64,
    /// Lower 64 bits of the 128-bit content hash (domain byte `0x32`).
    lo: u64,
    /// PTX entry-point name. Drives the keys-only-vs-with-indices split
    /// (`bolt_radix_scatter_<dty>` vs `bolt_radix_scatter_<dty>_with_indices`)
    /// and the histogram/MSB-flip distinctions within the same `(pass,
    /// dtype)` value.
    entry: &'static str,
}

impl RadixSortKey {
    fn new(spec: &RadixSortKernelSpec, entry: &'static str) -> Self {
        // Domain-separating prefix bytes distinct from `KernelSpecKey`'s
        // (`0x01` / `0x02`), `ScalarAggKey`'s (`0x11` / `0x12`), and
        // `HashJoinKey`'s (`0x21` / `0x22`) so a hypothetical hash
        // collision between the Debug shapes of the four spec types
        // can't alias keys.
        let (hi, lo) = hash128(spec, 0x31, 0x32);
        Self { hi, lo, entry }
    }
}

impl CacheKey for RadixSortKey {
    fn hi(&self) -> u64 {
        self.hi
    }
    fn lo(&self) -> u64 {
        self.lo
    }
    fn entry(&self) -> &'static str {
        self.entry
    }
}

/// State of the RadixSortKernelSpec-keyed cache. See [`SpecCache`] for the
/// FIFO eviction invariants.
type RadixSortCache = SpecCache<RadixSortKey>;

/// Process-wide RadixSortKernelSpec→`CudaModule` cache. Initialised lazily.
static RADIXSORT_CACHE: Lazy<Mutex<RadixSortCache>> =
    Lazy::new(|| Mutex::new(RadixSortCache::new(RADIX_SORT_CACHE_CAP)));

/// Disk-cache key prefix for the radix-sort PTX family. Stamped onto every
/// disk-cache write/read so a directory listing shows immediately which
/// kernel family produced each `.ptx` file, and so the projection /
/// scalar-aggregate / hash-join paths' key shapes can't ever collide with
/// a radix-sort entry.
///
/// V-3: `__` separator (not `::`) — see [`SCALAR_AGG_DISK_PREFIX`] for the
/// path-traversal-hardening rationale (`:` is outside the validated
/// filename-safe charset).
pub(crate) const RADIX_SORT_DISK_PREFIX: &str = "radix_sort__";

/// Test- and observability-facing snapshot of `(hits, misses)` for the
/// RadixSortKernelSpec cache. Parallel to [`hash_join_cache_stats`].
#[doc(hidden)]
#[must_use]
pub fn radix_sort_cache_stats() -> (usize, usize) {
    let c = RADIXSORT_CACHE.lock();
    (c.hits as usize, c.misses as usize)
}

/// Look up (or compile-and-load) the `CudaModule` for `(spec, entry)`.
///
/// This is the radix-sort analogue of [`get_or_build_module_for_hash_join`].
/// On a **cache hit** we return the cached `CudaModule` clone immediately —
/// no codegen, no PTX text generation, no `cuModuleLoadDataEx`. On a
/// **cache miss** we run `compile(spec)` to get the PTX text and hand it to
/// `CudaModule::from_ptx`. The resulting `(ptx, module)` pair is stored
/// under the RadixSortKernelSpec key so subsequent calls with the same
/// spec hit the fast path.
///
/// # Disk-cache integration
///
/// When the disk-backed PTX cache is enabled (env var `BOLT_PTX_CACHE_DIR`
/// or builder override), a miss in the in-memory cache consults the disk
/// cache *before* paying the codegen cost. The disk key is composed as
/// `"{codegen_salt}-{RADIX_SORT_DISK_PREFIX}{entry}-{hex(hash128)}"` so:
///   1. The `"radix_sort__"` prefix domain-separates these entries from
///      the projection-path, scalar-aggregate, and hash-join entries that
///      share the disk directory.
///   2. The `entry` suffix distinguishes
///      `bolt_radix_histogram_<dty>`, `bolt_radix_scatter_<dty>`,
///      `bolt_radix_scatter_<dty>_with_indices`, and
///      `bolt_radix_msb_flip_<dty>` slots even when the underlying
///      `RadixSortKernelSpec.pass` value alone wouldn't.
///   3. The 128-bit hex content hash makes the key collision-resistant
///      against unrelated specs that happen to share the same
///      `(pass, dtype)` tuple.
///
/// On a disk hit we skip codegen and feed the on-disk PTX straight to
/// `CudaModule::from_ptx`.
///
/// # Concurrency
///
/// The cache lock is held only long enough to look up or insert; the slow
/// `compile` + `from_ptx` work runs outside the lock. If two threads race
/// on the same miss they will each compile once and the second insert is
/// dropped by `insert`'s idempotence check.
pub(crate) fn get_or_build_module_for_radix_sort<F>(
    spec: &RadixSortKernelSpec,
    entry: &'static str,
    compile: F,
) -> BoltResult<CudaModule>
where
    F: FnOnce(&RadixSortKernelSpec) -> BoltResult<String>,
{
    let key = RadixSortKey::new(spec, entry);
    get_or_build_with_disk(&RADIXSORT_CACHE, key, RADIX_SORT_DISK_PREFIX, || {
        compile(spec)
    })
}

/// Test-friendly variant of [`get_or_build_module_for_radix_sort`] that
/// injects the module loader (so unit tests can stub out the CUDA driver
/// dependency in `CudaModule::from_ptx`). Production callers use the
/// non-`_with` form.
#[cfg(test)]
fn get_or_build_module_for_radix_sort_with<F, L>(
    spec: &RadixSortKernelSpec,
    entry: &'static str,
    compile: F,
    loader: L,
) -> BoltResult<CudaModule>
where
    F: FnOnce(&RadixSortKernelSpec) -> BoltResult<String>,
    L: FnOnce(&str) -> BoltResult<CudaModule>,
{
    let key = RadixSortKey::new(spec, entry);
    if let Some(cached) = RADIXSORT_CACHE.lock().get(&key) {
        return Ok(cached.module);
    }
    let ptx = compile(spec)?;
    let module = loader(&ptx)?;
    RADIXSORT_CACHE.lock().insert(
        key,
        CacheEntry {
            ptx,
            module: module.clone(),
        },
    );
    Ok(module)
}
// ---------------------------------------------------------------------------
// CompactionKernelSpec-keyed cache (v0.7). Mirror of the other spec-keyed
// caches above.
// ---------------------------------------------------------------------------

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
///
/// V-3: `__` separator (not `::`) — see [`SCALAR_AGG_DISK_PREFIX`] for the
/// path-traversal-hardening rationale (`:` is outside the validated
/// filename-safe charset).
pub(crate) const COMPACTION_DISK_PREFIX: &str = "compaction__";

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
        let (hi, lo) = hash128(spec, 0x41, 0x42);
        Self { hi, lo, entry }
    }
}

impl CacheKey for CompactionKey {
    fn hi(&self) -> u64 {
        self.hi
    }
    fn lo(&self) -> u64 {
        self.lo
    }
    fn entry(&self) -> &'static str {
        self.entry
    }
}

/// State of the CompactionKernelSpec-keyed cache. See [`SpecCache`] for the
/// FIFO eviction invariants.
type CompactionCache = SpecCache<CompactionKey>;

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
/// cache consults the disk cache *before* paying the codegen cost. The
/// disk key is composed as
/// `"{codegen_salt}-{COMPACTION_DISK_PREFIX}{entry}-{hex(hash128)}"`; the
/// `"compaction__"` prefix keeps these entries human-greppable in a
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
    get_or_build_with_disk(&COMPACTION_CACHE, key, COMPACTION_DISK_PREFIX, || {
        compile(spec)
    })
}

// ---------------------------------------------------------------------------
// Tests for the generic `SpecCache<K>` (FIFO eviction + hit/miss stats).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod spec_cache_tests {
    use super::*;

    /// Minimal `CacheKey` for driving the generic cache in isolation. The
    /// `(hi, lo)` pair doubles as the identity so we can enumerate distinct
    /// keys without depending on any planner IR type.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    struct TestKey {
        hi: u64,
        lo: u64,
    }

    impl CacheKey for TestKey {
        fn hi(&self) -> u64 {
            self.hi
        }
        fn lo(&self) -> u64 {
            self.lo
        }
        fn entry(&self) -> &'static str {
            "bolt_test_entry"
        }
    }

    fn key(n: u64) -> TestKey {
        TestKey { hi: n, lo: !n }
    }

    fn entry() -> CacheEntry {
        CacheEntry {
            ptx: String::new(),
            module: crate::jit::CudaModule::stub_for_tests(),
        }
    }

    /// FIFO eviction: inserting past the cap drops the *oldest* key first
    /// (insertion order), not the most-recently-touched one, and the map
    /// never exceeds `cap` live entries.
    #[test]
    fn spec_cache_evicts_in_fifo_order() {
        let mut cache: SpecCache<TestKey> = SpecCache::new(2);

        cache.insert(key(1), entry());
        cache.insert(key(2), entry());
        // At cap. Inserting key(3) must evict key(1) (the front of the FIFO).
        cache.insert(key(3), entry());

        assert!(cache.get(&key(1)).is_none(), "oldest key must be evicted");
        assert!(cache.get(&key(2)).is_some(), "key(2) must survive");
        assert!(cache.get(&key(3)).is_some(), "freshly inserted key present");
        assert!(cache.by_key.len() <= 2, "live entries must not exceed cap");
    }

    /// Re-inserting an existing key is idempotent: it neither grows the map
    /// nor re-ages the key in the FIFO order. After re-inserting key(1) we
    /// add key(3); key(1) (still at the front) must be the one evicted.
    #[test]
    fn spec_cache_insert_is_idempotent_and_preserves_fifo_position() {
        let mut cache: SpecCache<TestKey> = SpecCache::new(2);

        cache.insert(key(1), entry());
        cache.insert(key(2), entry());
        // Re-insert key(1): no-op, must NOT move it to the back.
        cache.insert(key(1), entry());
        assert_eq!(cache.by_key.len(), 2, "idempotent insert must not grow map");

        cache.insert(key(3), entry());
        assert!(
            cache.get(&key(1)).is_none(),
            "key(1) kept its original FIFO position and was evicted first"
        );
        assert!(cache.get(&key(2)).is_some());
        assert!(cache.get(&key(3)).is_some());
    }

    /// Hit/miss accounting: `get` bumps `misses` on a cold key and `hits`
    /// on a warm one. (Note: each `get` above also mutates the counters, so
    /// this test uses a fresh cache to pin the exact semantics.)
    #[test]
    fn spec_cache_get_tracks_hits_and_misses() {
        let mut cache: SpecCache<TestKey> = SpecCache::new(4);

        // Cold lookup: miss.
        assert!(cache.get(&key(1)).is_none());
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 0);

        // Seed then warm lookup: hit.
        cache.insert(key(1), entry());
        assert!(cache.get(&key(1)).is_some());
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 1, "a hit must not bump the miss counter");
    }
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

    /// v0.7 integration test: call the **real** `get_or_build_module_for_spec_with`
    /// twice with the same spec and confirm the second call hits the cache
    /// (compile closure runs exactly once across both calls). This is the
    /// production code path — only the module loader is stubbed out, since
    /// `CudaModule::from_ptx` requires a live CUDA context the test runner
    /// doesn't have.
    ///
    /// The spec uses a distinctive `register_count` so its key cannot
    /// collide with any other test's spec in the process-wide
    /// `KERNELSPEC_CACHE` — tests in the same binary share the static.
    #[test]
    fn get_or_build_module_for_spec_runs_compile_once() {
        let spec = KernelSpec {
            // Distinctive marker so neither the cold-lookup test nor any
            // unrelated test pollutes our cache slot. (Tests in the same
            // binary all share the global `KERNELSPEC_CACHE`.)
            register_count: 0xA7B0_07F1,
            ..empty_spec()
        };
        let entry = "bolt_v07_integration_marker";
        let compile_calls = AtomicUsize::new(0);
        let loader_calls = AtomicUsize::new(0);

        // First call: cold miss. Both the compile closure and the loader run.
        let _m1 = get_or_build_module_for_spec_with(
            &spec,
            entry,
            |_spec| {
                compile_calls.fetch_add(1, AOrdering::SeqCst);
                Ok("// fake ptx — never reaches the driver".to_string())
            },
            |_ptx| {
                loader_calls.fetch_add(1, AOrdering::SeqCst);
                Ok(crate::jit::CudaModule::stub_for_tests())
            },
        )
        .expect("stub loader must succeed");
        assert_eq!(compile_calls.load(AOrdering::SeqCst), 1, "cold miss must compile");
        assert_eq!(loader_calls.load(AOrdering::SeqCst), 1, "cold miss must load");

        // Second call with the same spec: warm hit. Neither the compile
        // closure nor the loader should run.
        let _m2 = get_or_build_module_for_spec_with(
            &spec,
            entry,
            |_spec| {
                compile_calls.fetch_add(1, AOrdering::SeqCst);
                Ok("// MUST NOT RUN — warm cache hit was expected".to_string())
            },
            |_ptx| {
                loader_calls.fetch_add(1, AOrdering::SeqCst);
                Ok(crate::jit::CudaModule::stub_for_tests())
            },
        )
        .expect("stub loader must succeed");
        assert_eq!(
            compile_calls.load(AOrdering::SeqCst),
            1,
            "warm-cache hit must skip codegen — compile closure ran a second time"
        );
        assert_eq!(
            loader_calls.load(AOrdering::SeqCst),
            1,
            "warm-cache hit must skip module load — loader ran a second time"
        );
    }

    /// Companion to `get_or_build_module_for_spec_runs_compile_once`: the
    /// public `kernelspec_cache_stats()` hook must reflect the hit a
    /// successful warm call generated. We use `>=` rather than `==` because
    /// the static counter is shared across every test in the binary; this
    /// test only proves the warm path *increments* `hits`.
    #[test]
    fn kernelspec_cache_stats_reflect_warm_hit() {
        let spec = KernelSpec {
            register_count: 0xA7B0_5747,
            ..empty_spec()
        };
        let entry = "bolt_v07_stats_marker";

        let (hits_before, _) = kernelspec_cache_stats();

        // Cold miss — seeds the cache.
        let _m = get_or_build_module_for_spec_with(
            &spec,
            entry,
            |_| Ok("// fake ptx".to_string()),
            |_| Ok(crate::jit::CudaModule::stub_for_tests()),
        )
        .expect("loader must succeed");

        // Warm hit — bumps `hits` by 1.
        let _m = get_or_build_module_for_spec_with(
            &spec,
            entry,
            |_| panic!("compile must not run on the warm path"),
            |_| panic!("loader must not run on the warm path"),
        )
        .expect("warm cache hit must succeed");

        let (hits_after, _) = kernelspec_cache_stats();
        assert!(
            hits_after >= hits_before + 1,
            "expected at least one hit bump (before={}, after={}); the warm \
             call must have flowed through the hit path",
            hits_before,
            hits_after,
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

    /// The disk-cache key prefix must start with `"scalar_agg__"` so a
    /// hand inspection of the cache directory distinguishes scalar-agg
    /// entries from projection-path entries. Pins the prefix contract
    /// against accidental drift.
    ///
    /// V-3: the separator is `__` (not `::`) so the composed key stays
    /// inside the filename-safe charset enforced by
    /// `jit::disk_cache::valid_key`; this test also asserts the composed
    /// key passes that validator (otherwise the family's disk cache would
    /// silently be a no-op).
    #[test]
    fn scalar_agg_disk_prefix_is_visibly_namespaced() {
        assert_eq!(SCALAR_AGG_DISK_PREFIX, "scalar_agg__");
        // The full disk key shape is now `"{codegen_salt}-{PREFIX}{entry}-{hex}"`
        // (JIT-M1 salt prepended). Confirm the family prefix lands immediately
        // after the salt so a hand inspection of the cache dir still
        // distinguishes scalar-agg entries from projection-path entries.
        let salt = crate::jit::disk_cache::codegen_salt();
        let composed = compose_disk_key(SCALAR_AGG_DISK_PREFIX, "bolt_reduce", 0xdead_beef, 0xcafe_babe);
        assert!(
            composed.starts_with(&format!("{salt}-scalar_agg__")),
            "composed disk key must carry the salt then the scalar_agg prefix: {composed}"
        );
        // V-3: composed key must survive the disk-cache key validator,
        // otherwise this family's disk cache is dead on arrival.
        assert!(
            crate::jit::disk_cache::valid_key(&composed),
            "composed scalar_agg disk key must pass the filename-safe validator: {composed}"
        );
    }

    /// JIT-M1: `compose_disk_key` must (1) prepend the codegen salt, (2)
    /// preserve the `{prefix}{entry}-{hash}` domain-separated tail, (3) rotate
    /// when the spec hash changes, and (4) keep distinct kernel families in
    /// distinct keys. This is the disk-backed twin of the engine.rs
    /// `disk_cache::disk_key` salt fix.
    #[test]
    fn compose_disk_key_salts_and_separates_domains() {
        use crate::jit::disk_cache::{codegen_salt, hash_to_key};
        let salt = codegen_salt();
        let k = compose_disk_key("scalar_agg__", "bolt_reduce", 0xABCD, 0x1234);

        // (1) salt is the leading component.
        assert!(k.starts_with(&format!("{salt}-")), "missing salt prefix: {k}");
        // (2) the historical tail is preserved byte-for-byte after the salt
        //     (V-3: `__` separator in place of the old `::`).
        let tail = format!("scalar_agg__bolt_reduce-{}", hash_to_key(0xABCD, 0x1234));
        assert_eq!(k, format!("{salt}-{tail}"), "tail must be salt + historical shape");
        // (3) a different spec hash yields a different key.
        assert_ne!(k, compose_disk_key("scalar_agg__", "bolt_reduce", 0xABCD, 0x9999));
        // (4) a different family prefix yields a different key (no aliasing).
        assert_ne!(k, compose_disk_key("hash_join__", "bolt_reduce", 0xABCD, 0x1234));
        // ...and a different entry symbol too.
        assert_ne!(k, compose_disk_key("scalar_agg__", "bolt_other", 0xABCD, 0x1234));
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

// ---------------------------------------------------------------------------
// Tests for the HashJoinKernelSpec-keyed cache.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod hash_join_cache_tests {
    use super::*;
    use crate::plan::logical_plan::DataType;
    use crate::plan::physical_plan::{HashJoinKernelKind, HashJoinKernelSpec};
    use std::sync::atomic::{AtomicUsize, Ordering as AOrdering};

    fn mk_spec(kind: HashJoinKernelKind, key_dtype: DataType) -> HashJoinKernelSpec {
        HashJoinKernelSpec {
            kind,
            key_dtype,
            string_hash_returns_i64: false,
        }
    }

    /// Cold lookup against a fresh cache is a miss and `get` accounts for
    /// it. Mirrors `scalar_agg_cache_cold_lookup_is_a_miss`.
    #[test]
    fn hash_join_cache_cold_lookup_is_a_miss() {
        let mut cache = HashJoinCache::new(4);
        let spec = mk_spec(HashJoinKernelKind::Build, DataType::Int64);
        let key = HashJoinKey::new(&spec, "bolt_build");

        assert!(cache.get(&key).is_none(), "fresh cache must miss");
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 0);
    }

    /// State-machine pin of the hit path: a second call with the same spec
    /// must NOT re-run the compile closure and must NOT re-run the loader.
    /// We mock `CudaModule::from_ptx` with a stub closure since CUDA isn't
    /// available in unit tests; the cache state machine is identical.
    #[test]
    fn hash_join_cache_compile_and_loader_run_once_per_unique_spec() {
        // Mock module type — just a tag so we can confirm the same instance
        // comes back on a hit.
        #[derive(Clone)]
        struct MockModule(&'static str);

        struct MockCache {
            by_key: HashMap<HashJoinKey, MockModule>,
            order: VecDeque<HashJoinKey>,
            cap: usize,
        }

        impl MockCache {
            fn get_or_build(
                &mut self,
                spec: &HashJoinKernelSpec,
                entry: &'static str,
                compile_count: &AtomicUsize,
                loader_count: &AtomicUsize,
                tag: &'static str,
            ) -> MockModule {
                let key = HashJoinKey::new(spec, entry);
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

        let spec_build_i64 = mk_spec(HashJoinKernelKind::Build, DataType::Int64);
        let spec_probe_i64 = mk_spec(HashJoinKernelKind::Probe, DataType::Int64);

        let compile_count = AtomicUsize::new(0);
        let loader_count = AtomicUsize::new(0);
        let mut cache = MockCache {
            by_key: HashMap::new(),
            order: VecDeque::new(),
            cap: 64,
        };

        // First call with (Build, Int64): miss → compile + load each run once.
        let m1 = cache.get_or_build(
            &spec_build_i64,
            "bolt_build",
            &compile_count,
            &loader_count,
            "build-i64",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 1);
        assert_eq!(loader_count.load(AOrdering::SeqCst), 1);
        assert_eq!(m1.0, "build-i64");

        // Second call with the SAME spec: HIT → neither closure runs again.
        let m2 = cache.get_or_build(
            &spec_build_i64,
            "bolt_build",
            &compile_count,
            &loader_count,
            "build-i64",
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
        assert_eq!(m2.0, "build-i64");

        // A different kind (Probe): miss again.
        let m3 = cache.get_or_build(
            &spec_probe_i64,
            "bolt_probe",
            &compile_count,
            &loader_count,
            "probe-i64",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 2);
        assert_eq!(loader_count.load(AOrdering::SeqCst), 2);
        assert_eq!(m3.0, "probe-i64");

        // And back to spec_build_i64: still a hit, neither closure runs.
        let m4 = cache.get_or_build(
            &spec_build_i64,
            "bolt_build",
            &compile_count,
            &loader_count,
            "build-i64",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 2);
        assert_eq!(loader_count.load(AOrdering::SeqCst), 2);
        assert_eq!(m4.0, "build-i64");

        // string_hash_returns_i64 = true flavour vs false: must miss
        // because the bool field participates in the cache key. This pins
        // the routing that distinguishes the two string-hash kernel widths.
        let string_hash_i32 = HashJoinKernelSpec {
            kind: HashJoinKernelKind::StringHash,
            key_dtype: DataType::Utf8,
            string_hash_returns_i64: false,
        };
        let string_hash_i64 = HashJoinKernelSpec {
            kind: HashJoinKernelKind::StringHash,
            key_dtype: DataType::Utf8,
            string_hash_returns_i64: true,
        };
        let _ = cache.get_or_build(
            &string_hash_i32,
            "bolt_string_hash",
            &compile_count,
            &loader_count,
            "sh-i32",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 3);
        let _ = cache.get_or_build(
            &string_hash_i64,
            "bolt_string_hash_i64",
            &compile_count,
            &loader_count,
            "sh-i64",
        );
        assert_eq!(
            compile_count.load(AOrdering::SeqCst),
            4,
            "string_hash_returns_i64 must participate in the key"
        );
    }

    /// Key uniqueness: distinct kinds, dtypes, entry tags, and
    /// `string_hash_returns_i64` flavours must each produce distinct keys.
    #[test]
    fn hash_join_cache_key_distinguishes_axes() {
        let build_i64 = mk_spec(HashJoinKernelKind::Build, DataType::Int64);
        let build_i32 = mk_spec(HashJoinKernelKind::Build, DataType::Int32);
        let probe_i64 = mk_spec(HashJoinKernelKind::Probe, DataType::Int64);
        let string_hash_i32 = HashJoinKernelSpec {
            kind: HashJoinKernelKind::StringHash,
            key_dtype: DataType::Utf8,
            string_hash_returns_i64: false,
        };
        let string_hash_i64 = HashJoinKernelSpec {
            kind: HashJoinKernelKind::StringHash,
            key_dtype: DataType::Utf8,
            string_hash_returns_i64: true,
        };

        let k_build_i64 = HashJoinKey::new(&build_i64, "bolt_build");
        let k_build_i32 = HashJoinKey::new(&build_i32, "bolt_build");
        let k_probe_i64 = HashJoinKey::new(&probe_i64, "bolt_probe");
        let k_build_i64_alt = HashJoinKey::new(&build_i64, "bolt_build_aos");
        let k_sh_i32 = HashJoinKey::new(&string_hash_i32, "bolt_string_hash");
        let k_sh_i64 = HashJoinKey::new(&string_hash_i64, "bolt_string_hash_i64");

        assert_ne!(k_build_i64, k_build_i32, "key_dtype must participate");
        assert_ne!(k_build_i64, k_probe_i64, "kind must participate");
        assert_ne!(k_build_i64, k_build_i64_alt, "entry tag must participate");
        assert_ne!(
            k_sh_i32, k_sh_i64,
            "string_hash_returns_i64 must participate in the key"
        );
    }

    /// The disk-cache key prefix must start with `"hash_join__"` so a
    /// hand inspection of the cache directory distinguishes hash-join
    /// entries from projection-path and scalar-agg entries. Pins the
    /// prefix contract against accidental drift.
    ///
    /// V-3: `__` separator (not `::`) keeps the composed key inside the
    /// filename-safe charset enforced by `jit::disk_cache::valid_key`.
    #[test]
    fn hash_join_disk_prefix_is_visibly_namespaced() {
        assert_eq!(HASH_JOIN_DISK_PREFIX, "hash_join__");
        // Shape: `"{codegen_salt}-{PREFIX}{entry}-{hex}"` (JIT-M1 salt prepended).
        let salt = crate::jit::disk_cache::codegen_salt();
        let composed = compose_disk_key(HASH_JOIN_DISK_PREFIX, "bolt_build", 0xdead_beef, 0xcafe_babe);
        assert!(
            composed.starts_with(&format!("{salt}-hash_join__")),
            "composed disk key must carry the salt then the hash_join prefix: {composed}"
        );
        // V-3: composed key must survive the disk-cache key validator.
        assert!(
            crate::jit::disk_cache::valid_key(&composed),
            "composed hash_join disk key must pass the filename-safe validator: {composed}"
        );
    }

    /// The key fingerprint is stable across `Copy` of the same spec.
    /// Callers `Copy` the IR before handing it to the cache, so the
    /// stability is load-bearing.
    #[test]
    fn hash_join_cache_key_stable_across_copy() {
        let spec = mk_spec(HashJoinKernelKind::ProbeTiled, DataType::Int64);
        let copied = spec;
        assert_eq!(
            HashJoinKey::new(&spec, "bolt_probe_tiled"),
            HashJoinKey::new(&copied, "bolt_probe_tiled"),
            "copying the spec must not change its cache key"
        );
    }
}

// ---------------------------------------------------------------------------
// Tests for the RadixSortKernelSpec-keyed cache.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod radix_sort_cache_tests {
    use super::*;
    use crate::plan::logical_plan::DataType;
    use crate::plan::physical_plan::{RadixSortKernelSpec, RadixSortPass};
    use std::sync::atomic::{AtomicUsize, Ordering as AOrdering};

    fn mk_spec(pass: RadixSortPass, dtype: DataType) -> RadixSortKernelSpec {
        RadixSortKernelSpec { pass, dtype }
    }

    /// Cold lookup against a fresh cache is a miss and `get` accounts for
    /// it. Mirrors `hash_join_cache_cold_lookup_is_a_miss`.
    #[test]
    fn radix_sort_cache_cold_lookup_is_a_miss() {
        let mut cache = RadixSortCache::new(4);
        let spec = mk_spec(RadixSortPass::Histogram, DataType::Int32);
        let key = RadixSortKey::new(&spec, "bolt_radix_histogram_i32");

        assert!(cache.get(&key).is_none(), "fresh cache must miss");
        assert_eq!(cache.misses, 1);
        assert_eq!(cache.hits, 0);
    }

    /// State-machine pin of the hit path: a second call with the same spec
    /// must NOT re-run the compile closure and must NOT re-run the loader.
    /// We mock `CudaModule::from_ptx` with a stub closure since CUDA isn't
    /// available in unit tests; the cache state machine is identical.
    #[test]
    fn radix_sort_cache_compile_and_loader_run_once_per_unique_spec() {
        #[derive(Clone)]
        struct MockModule(&'static str);

        struct MockCache {
            by_key: HashMap<RadixSortKey, MockModule>,
            order: VecDeque<RadixSortKey>,
            cap: usize,
        }

        impl MockCache {
            fn get_or_build(
                &mut self,
                spec: &RadixSortKernelSpec,
                entry: &'static str,
                compile_count: &AtomicUsize,
                loader_count: &AtomicUsize,
                tag: &'static str,
            ) -> MockModule {
                let key = RadixSortKey::new(spec, entry);
                if let Some(m) = self.by_key.get(&key) {
                    return m.clone();
                }
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

        let hist_i32 = mk_spec(RadixSortPass::Histogram, DataType::Int32);
        let scatter_i32 = mk_spec(RadixSortPass::Scatter, DataType::Int32);

        let compile_count = AtomicUsize::new(0);
        let loader_count = AtomicUsize::new(0);
        let mut cache = MockCache {
            by_key: HashMap::new(),
            order: VecDeque::new(),
            cap: 64,
        };

        // First call with (Histogram, Int32): miss → compile + load each
        // run once.
        let m1 = cache.get_or_build(
            &hist_i32,
            "bolt_radix_histogram_i32",
            &compile_count,
            &loader_count,
            "hist-i32",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 1);
        assert_eq!(loader_count.load(AOrdering::SeqCst), 1);
        assert_eq!(m1.0, "hist-i32");

        // Second call with the SAME spec: HIT → neither closure runs again.
        let m2 = cache.get_or_build(
            &hist_i32,
            "bolt_radix_histogram_i32",
            &compile_count,
            &loader_count,
            "hist-i32",
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
        assert_eq!(m2.0, "hist-i32");

        // A different pass (Scatter) at the same dtype: miss again.
        let m3 = cache.get_or_build(
            &scatter_i32,
            "bolt_radix_scatter_i32",
            &compile_count,
            &loader_count,
            "scatter-i32",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 2);
        assert_eq!(loader_count.load(AOrdering::SeqCst), 2);
        assert_eq!(m3.0, "scatter-i32");

        // The keys-only Scatter entry and the ScatterWithIndices entry
        // both live under the Scatter→ScatterWithIndices branch but use
        // different `entry` strings — the entry tag participates in the
        // key, so the call must miss and compile again.
        let scatter_with_indices_i32 =
            mk_spec(RadixSortPass::ScatterWithIndices, DataType::Int32);
        let m4 = cache.get_or_build(
            &scatter_with_indices_i32,
            "bolt_radix_scatter_i32_with_indices",
            &compile_count,
            &loader_count,
            "scatter-wi-i32",
        );
        assert_eq!(
            compile_count.load(AOrdering::SeqCst),
            3,
            "ScatterWithIndices is a distinct pass and entry — must miss"
        );
        assert_eq!(m4.0, "scatter-wi-i32");

        // And back to the original (Histogram, Int32): still a hit, neither
        // closure runs.
        let m5 = cache.get_or_build(
            &hist_i32,
            "bolt_radix_histogram_i32",
            &compile_count,
            &loader_count,
            "hist-i32",
        );
        assert_eq!(compile_count.load(AOrdering::SeqCst), 3);
        assert_eq!(loader_count.load(AOrdering::SeqCst), 3);
        assert_eq!(m5.0, "hist-i32");
    }

    /// Key uniqueness: distinct passes, dtypes, and entry tags must each
    /// produce distinct keys. (This is the assumption the hit-vs-miss
    /// classification relies on.)
    #[test]
    fn radix_sort_cache_key_distinguishes_pass_dtype_and_entry() {
        let hist_i32 = mk_spec(RadixSortPass::Histogram, DataType::Int32);
        let hist_i64 = mk_spec(RadixSortPass::Histogram, DataType::Int64);
        let scatter_i32 = mk_spec(RadixSortPass::Scatter, DataType::Int32);
        let scatter_wi_i32 = mk_spec(RadixSortPass::ScatterWithIndices, DataType::Int32);
        let msb_flip_i32 = mk_spec(RadixSortPass::MsbFlip, DataType::Int32);

        let k_hist_i32 = RadixSortKey::new(&hist_i32, "bolt_radix_histogram_i32");
        let k_hist_i64 = RadixSortKey::new(&hist_i64, "bolt_radix_histogram_i64");
        let k_scatter_i32 = RadixSortKey::new(&scatter_i32, "bolt_radix_scatter_i32");
        let k_scatter_wi_i32 = RadixSortKey::new(
            &scatter_wi_i32,
            "bolt_radix_scatter_i32_with_indices",
        );
        let k_msb_flip_i32 = RadixSortKey::new(&msb_flip_i32, "bolt_radix_msb_flip_i32");
        let k_hist_i32_alt = RadixSortKey::new(&hist_i32, "bolt_radix_histogram_alt");

        assert_ne!(k_hist_i32, k_hist_i64, "dtype must participate in the key");
        assert_ne!(k_hist_i32, k_scatter_i32, "pass must participate in the key");
        assert_ne!(
            k_scatter_i32, k_scatter_wi_i32,
            "Scatter and ScatterWithIndices must hash distinctly"
        );
        assert_ne!(
            k_hist_i32, k_msb_flip_i32,
            "MsbFlip must hash distinctly from Histogram"
        );
        assert_ne!(
            k_hist_i32, k_hist_i32_alt,
            "entry tag must participate in the key"
        );
    }

    /// The disk-cache key prefix must start with `"radix_sort__"` so a
    /// hand inspection of the cache directory distinguishes radix-sort
    /// entries from projection-path, scalar-agg, and hash-join entries.
    /// Pins the prefix contract against accidental drift.
    ///
    /// V-3: `__` separator (not `::`) keeps the composed key inside the
    /// filename-safe charset enforced by `jit::disk_cache::valid_key`.
    #[test]
    fn radix_sort_disk_prefix_is_visibly_namespaced() {
        assert_eq!(RADIX_SORT_DISK_PREFIX, "radix_sort__");
        // Shape: `"{codegen_salt}-{PREFIX}{entry}-{hex}"` (JIT-M1 salt prepended).
        let salt = crate::jit::disk_cache::codegen_salt();
        let composed =
            compose_disk_key(RADIX_SORT_DISK_PREFIX, "bolt_radix_histogram_i32", 0xdead_beef, 0xcafe_babe);
        assert!(
            composed.starts_with(&format!("{salt}-radix_sort__")),
            "composed disk key must carry the salt then the radix_sort prefix: {composed}"
        );
        // V-3: composed key must survive the disk-cache key validator.
        assert!(
            crate::jit::disk_cache::valid_key(&composed),
            "composed radix_sort disk key must pass the filename-safe validator: {composed}"
        );
    }

    /// The key fingerprint is stable across `Copy` of the same spec.
    /// Callers `Copy` the IR before handing it to the cache, so the
    /// stability is load-bearing.
    #[test]
    fn radix_sort_cache_key_stable_across_copy() {
        let spec = mk_spec(RadixSortPass::ScatterWithIndices, DataType::Int64);
        let copied = spec;
        assert_eq!(
            RadixSortKey::new(&spec, "bolt_radix_scatter_i64_with_indices"),
            RadixSortKey::new(&copied, "bolt_radix_scatter_i64_with_indices"),
            "copying the spec must not change its cache key"
        );
    }

    /// v0.7 integration test: call the real `get_or_build_module_for_radix_sort_with`
    /// twice with the same `(spec, entry)` pair and confirm the second call
    /// hits the cache. The compile closure runs exactly once across both
    /// calls; the stub loader likewise runs exactly once. This is the
    /// production code path with only `CudaModule::from_ptx` replaced
    /// (since the test runner has no CUDA context).
    ///
    /// We use a distinctive `entry` tag so the cache slot cannot collide
    /// with any other test's spec in the process-wide `RADIXSORT_CACHE` —
    /// tests in the same binary share the static.
    #[test]
    fn get_or_build_module_for_radix_sort_with_runs_compile_once() {
        let spec = mk_spec(RadixSortPass::Histogram, DataType::Int32);
        let entry = "bolt_v07_radix_integration_marker";
        let compile_calls = AtomicUsize::new(0);
        let loader_calls = AtomicUsize::new(0);

        // First call: cold miss. Both the compile closure and the loader run.
        let _m1 = get_or_build_module_for_radix_sort_with(
            &spec,
            entry,
            |_spec| {
                compile_calls.fetch_add(1, AOrdering::SeqCst);
                Ok("// fake ptx — never reaches the driver".to_string())
            },
            |_ptx| {
                loader_calls.fetch_add(1, AOrdering::SeqCst);
                Ok(crate::jit::CudaModule::stub_for_tests())
            },
        )
        .expect("stub loader must succeed");
        assert_eq!(
            compile_calls.load(AOrdering::SeqCst),
            1,
            "cold miss must compile"
        );
        assert_eq!(
            loader_calls.load(AOrdering::SeqCst),
            1,
            "cold miss must load"
        );

        // Second call with the same spec: warm hit. Neither the compile
        // closure nor the loader should run.
        let _m2 = get_or_build_module_for_radix_sort_with(
            &spec,
            entry,
            |_spec| {
                compile_calls.fetch_add(1, AOrdering::SeqCst);
                Ok("// MUST NOT RUN — warm cache hit was expected".to_string())
            },
            |_ptx| {
                loader_calls.fetch_add(1, AOrdering::SeqCst);
                Ok(crate::jit::CudaModule::stub_for_tests())
            },
        )
        .expect("stub loader must succeed");
        assert_eq!(
            compile_calls.load(AOrdering::SeqCst),
            1,
            "warm-cache hit must skip codegen — compile closure ran a second time"
        );
        assert_eq!(
            loader_calls.load(AOrdering::SeqCst),
            1,
            "warm-cache hit must skip module load — loader ran a second time"
        );
    }

    /// Companion to `get_or_build_module_for_radix_sort_with_runs_compile_once`:
    /// the public `radix_sort_cache_stats()` hook must reflect the hit a
    /// successful warm call generated. We use `>=` rather than `==` because
    /// the static counter is shared across every test in the binary; this
    /// test only proves the warm path *increments* `hits`.
    #[test]
    fn radix_sort_cache_stats_reflect_warm_hit() {
        let spec = mk_spec(RadixSortPass::Scatter, DataType::Int64);
        let entry = "bolt_v07_radix_stats_marker";

        let (hits_before, _) = radix_sort_cache_stats();

        // Cold miss — seeds the cache.
        let _m = get_or_build_module_for_radix_sort_with(
            &spec,
            entry,
            |_| Ok("// fake ptx".to_string()),
            |_| Ok(crate::jit::CudaModule::stub_for_tests()),
        )
        .expect("loader must succeed");

        // Warm hit — bumps `hits` by 1.
        let _m = get_or_build_module_for_radix_sort_with(
            &spec,
            entry,
            |_| panic!("compile must not run on the warm path"),
            |_| panic!("loader must not run on the warm path"),
        )
        .expect("warm cache hit must succeed");

        let (hits_after, _) = radix_sort_cache_stats();
        assert!(
            hits_after >= hits_before + 1,
            "expected at least one hit bump (before={}, after={}); the warm \
             call must have flowed through the hit path",
            hits_before,
            hits_after,
        );
    }
}
