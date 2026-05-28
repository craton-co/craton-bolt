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

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::error::BoltResult;
use crate::jit::CudaModule;

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
