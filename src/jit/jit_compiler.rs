// SPDX-License-Identifier: Apache-2.0

//! PTX → loaded CUDA module via the driver's in-process assembler.
//!
//! Despite the name, no separate NVRTC dependency is involved: we hand the
//! PTX text to `cuModuleLoadDataEx`, which performs the PTX → SASS step inside
//! the CUDA driver and returns a ready-to-launch module. We use the `Ex`
//! variant so we can pass info/error log buffers and surface PTXAS diagnostics
//! (including line numbers) when the load fails.
//!
//! # Process-wide PTX cache
//!
//! PTXAS assembly inside `cuModuleLoadDataEx` is the dominant cost of
//! `Engine::sql` for short queries — typically tens of milliseconds per
//! invocation. To eliminate that cost for repeated identical queries we keep
//! a process-wide cache keyed by a 128-bit hash of the PTX text (a pair of
//! `DefaultHasher` outputs with domain-separation bytes — see [`hash_ptx`]).
//!
//! **Invariant.** The codegen pipeline is deterministic: for a given
//! `(PhysicalPlan, kernel_name)` pair the emitted PTX text is byte-identical
//! across runs within a process. Therefore hashing the PTX text and reusing
//! the loaded `CUmodule` is sound — two cache lookups that collide on the
//! hash *and* the full string match represent literally the same program.
//! The cache always re-checks the stored PTX text on a hit, so even a hash
//! collision is correctness-safe: it falls through to [`Slot::Collision`]
//! and the caller runs an uncached `load_uncached`. The hash width was
//! upgraded from the original 64 bits (birthday-paradox bound ≈ 2^-32 over
//! the lifetime of a busy process) to 128 bits (≈ 2^-64) so that the
//! Collision-fallback path is effectively unreachable rather than merely
//! rare — it remains in place as defence-in-depth.
//!
//! The cache is bounded (default 256 entries) with **LRU** eviction backed
//! by an intrusive doubly-linked list over `Vec<Option<LruNode>>`. Cache
//! hits move the node to the head (most-recently-used); a fresh miss
//! inserts at the head and evicts the tail when at cap. All operations are
//! O(1). The `HashMap` maps the 128-bit PTX-hash key `(u64, u64)` to the
//! node index.
//!
//! When an entry is evicted from the cache its `Arc` strong count drops; if
//! no `CudaModule` clones are live the underlying `CudaModuleInner::Drop`
//! runs and calls `cuModuleUnload`. If clones *are* live the module stays
//! loaded until the last clone is dropped — exactly the lifetime users
//! expect.
//!
//! The cap is overridable at process start via the environment variable
//! **`CRATON_BOLT_PTX_CACHE_CAP`** — set it to any positive integer (parsed
//! as `usize`). Unset, empty, zero, or unparseable values fall back to the
//! built-in default of 256. The value is read exactly once on first cache
//! access and frozen for the lifetime of the process.
//!
//! # Concurrency
//!
//! On a cache miss the actual PTX → SASS compile (`cuModuleLoadDataEx`,
//! tens of ms) runs *outside* the cache lock. To make that race-free we
//! store a `OnceCell` per key inside the map: the first thread to miss
//! inserts an empty `OnceCell` under the lock, releases the lock, and then
//! `get_or_try_init`s it. A second thread racing on the same PTX finds the
//! same `OnceCell` under the lock, releases the lock, and blocks inside
//! `get_or_try_init` until the first thread's compile completes — it then
//! receives the same `Arc<CudaModuleInner>` without paying the compile
//! cost. Compiles for *different* PTX keys run fully in parallel.
//!
//! # Observability
//!
//! The cache tracks three monotonically-increasing counters — `hits`,
//! `misses`, and `evictions` — accessible via the public
//! [`ptx_cache_stats`] free function. The counters are snapshot under the
//! cache lock, so callers always see a consistent triple. Suitable for
//! Prometheus exporters, benchmark scoreboards, and test assertions.

use std::collections::HashMap;
use std::ffi::CString;
// `Hasher` is needed in scope so `DefaultHasher::write` / `write_u8` /
// `finish` resolve via the trait. We no longer reach for `Hash` directly
// (the 128-bit `hash_ptx` writes raw bytes) but keep the import to mirror
// the test module's needs and avoid a churn on the import line.
use std::hash::Hasher;
use std::marker::PhantomData;
use std::ptr;
use std::sync::{Arc, OnceLock};

use libc::c_void;
use log::warn;
use once_cell::sync::OnceCell;
use parking_lot::Mutex;

use crate::cuda::cuda_sys::{self, CUfunction, CUmodule};
use crate::error::{BoltError, BoltResult};

// --- CUjit_option constants -------------------------------------------------
// Mirrored from <cuda.h> — declared here (rather than in cuda_sys.rs) per the
// orchestrator's file-ownership boundary. Values are stable ABI.
/// `CUjit_option` value type, as expected by `cuModuleLoadDataEx`.
#[allow(non_camel_case_types)] // reason: must match the CUDA C ABI name verbatim
pub type CUjit_option = i32;

/// Pointer to a buffer in which to print any info log messages from PTXAS.
pub const CU_JIT_INFO_LOG_BUFFER: CUjit_option = 3;
/// Input: size in bytes of `CU_JIT_INFO_LOG_BUFFER`. Output: bytes filled.
pub const CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES: CUjit_option = 4;
/// Pointer to a buffer in which to print any error log messages from PTXAS.
pub const CU_JIT_ERROR_LOG_BUFFER: CUjit_option = 5;
/// Input: size in bytes of `CU_JIT_ERROR_LOG_BUFFER`. Output: bytes filled.
pub const CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES: CUjit_option = 6;

const JIT_LOG_BUF_SIZE: usize = 4096;

/// Built-in default cap on the number of cached compiled modules. Override at
/// process start via `CRATON_BOLT_PTX_CACHE_CAP` (see module docs). Eviction
/// is LRU (least-recently-used) once we exceed this many entries.
const PTX_CACHE_CAP_DEFAULT: usize = 256;

/// Environment variable that overrides the PTX cache capacity. Parsed as
/// `usize`. Unset / empty / zero / unparseable → fall back to
/// `PTX_CACHE_CAP_DEFAULT`. Read once on first cache access and memoized.
///
/// `pub(crate)` so the integration test `tests/env_var_smoke.rs` can
/// reference the canonical name from one place. The in-module test
/// `parse_cap_picks_up_env_var` already drives `parse_cap` against
/// a *different* env var to side-step the OnceLock memoisation; the
/// integration test does the same against the live name.
pub const PTX_CACHE_CAP_ENV: &str = "CRATON_BOLT_PTX_CACHE_CAP";

/// Parse a candidate cache-cap string (typically from `std::env::var`).
/// `None`, empty strings, zero, and unparseable values map to `default`.
/// Factored out so the policy is testable without touching the process
/// environment or the memoized global cap.
///
/// `pub(crate)` so `tests/env_var_smoke.rs` can drive the policy with
/// borrowed `&str` inputs (mirroring the in-module unit tests) without
/// depending on the memoised `ptx_cache_cap` global.
pub fn parse_cap(raw: Option<&str>, default: usize) -> usize {
    raw.and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(default)
}

/// Resolve the effective cache cap. Reads the env var the first time it is
/// called and freezes the result for the rest of the process lifetime — we
/// do not want a runaway query to change the cap mid-execution.
fn ptx_cache_cap() -> usize {
    static CAP: OnceLock<usize> = OnceLock::new();
    // TODO(cache-cap): re-read on each insert so misconfigured envs can be
    // hot-fixed
    *CAP.get_or_init(|| {
        let raw = std::env::var(PTX_CACHE_CAP_ENV).ok();
        parse_cap(raw.as_deref(), PTX_CACHE_CAP_DEFAULT)
    })
}

/// Owns the raw `CUmodule` and is responsible for unloading on drop. Lives
/// behind an `Arc` so identical PTX queries share a single loaded module.
struct CudaModuleInner {
    raw: CUmodule,
}

impl Drop for CudaModuleInner {
    fn drop(&mut self) {
        if self.raw.is_null() {
            return;
        }
        let code = unsafe { cuda_sys::cuModuleUnload(self.raw) };
        if code != cuda_sys::CUDA_SUCCESS {
            // Non-fatal: we're in a Drop path and cannot propagate the error.
            // Log via the `log` facade so library consumers can route it.
            warn!(
                "craton-bolt: cuModuleUnload failed with code {} (module leaked)",
                code
            );
        }
    }
}

// SAFETY: CUmodule is a global handle valid in any thread once the context is
// current. We only ever read `raw` after construction; the `Drop` impl is the
// sole writer and runs at most once when the last `Arc` is dropped.
unsafe impl Send for CudaModuleInner {}
unsafe impl Sync for CudaModuleInner {}

/// One cache slot's payload: the PTX text that produced this entry and the
/// lazily-populated `OnceCell` holding the loaded module. We retain the
/// source text so a hash collision can be detected (see the module-level
/// comment).
///
/// The `Arc<OnceCell<…>>` lets us release the cache lock before doing the
/// slow PTXAS compile: the first thread to miss inserts an empty cell and
/// `get_or_try_init`s it; concurrent threads racing on the same PTX clone
/// the same `Arc<OnceCell>`, block on `get_or_try_init`, and pick up the
/// same compiled module without re-running the driver. (Bug H3.)
struct CacheEntry {
    ptx: String,
    module: Arc<OnceCell<Arc<CudaModuleInner>>>,
}

/// One node in the intrusive doubly-linked LRU list. Indices reference
/// other slots in `PtxCache::nodes`; `None` marks the head's `prev` and
/// the tail's `next`.
struct LruNode {
    prev: Option<usize>,
    next: Option<usize>,
    key: (u64, u64),
    entry: CacheEntry,
}

/// Cache state: a `HashMap` for O(1) key → index lookup plus an intrusive
/// doubly-linked list (over `Vec<Option<LruNode>>`) for O(1) LRU eviction.
///
/// The key is the 128-bit `(hi, lo)` PTX hash — see [`hash_ptx`].
struct PtxCache {
    nodes: Vec<Option<LruNode>>,
    free_list: Vec<usize>,
    by_key: HashMap<(u64, u64), usize>,
    head: Option<usize>,
    tail: Option<usize>,
    /// Cumulative count of LRU evictions since cache creation.
    evictions: u64,
    /// Cumulative count of cache hits since creation.
    hits: u64,
    /// Cumulative count of cache misses since creation.
    misses: u64,
}

impl PtxCache {
    fn new() -> Self {
        let cap = ptx_cache_cap();
        Self {
            nodes: Vec::with_capacity(cap),
            free_list: Vec::new(),
            by_key: HashMap::with_capacity(cap),
            head: None,
            tail: None,
            evictions: 0,
            hits: 0,
            misses: 0,
        }
    }

    fn len(&self) -> usize {
        self.by_key.len()
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.by_key.is_empty()
    }

    #[cfg(test)]
    fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Borrow the node at `idx`. Panics if the slot has been freed — only
    /// safe to call with indices we just read from `by_key` or `head`/
    /// `tail`, which always point at occupied slots by construction.
    fn node(&self, idx: usize) -> &LruNode {
        self.nodes[idx]
            .as_ref()
            .expect("PtxCache: node index points at a freed slot")
    }

    fn node_mut(&mut self, idx: usize) -> &mut LruNode {
        self.nodes[idx]
            .as_mut()
            .expect("PtxCache: node index points at a freed slot")
    }

    /// Detach `idx` from its position in the doubly-linked list, fixing
    /// up its neighbours and the `head`/`tail` anchors. The node itself
    /// is left in place inside `nodes`; the caller decides whether to
    /// re-insert it at the head or free it.
    fn unlink(&mut self, idx: usize) {
        let (prev, next) = {
            let n = self.node(idx);
            (n.prev, n.next)
        };
        match prev {
            Some(p) => self.node_mut(p).next = next,
            None => self.head = next,
        }
        match next {
            Some(n) => self.node_mut(n).prev = prev,
            None => self.tail = prev,
        }
        let n = self.node_mut(idx);
        n.prev = None;
        n.next = None;
    }

    /// Insert `idx` at the head of the list (most-recently-used).
    /// Assumes the node's `prev`/`next` are already `None`.
    fn push_front(&mut self, idx: usize) {
        let old_head = self.head;
        self.node_mut(idx).next = old_head;
        if let Some(h) = old_head {
            self.node_mut(h).prev = Some(idx);
        } else {
            // List was empty: this node becomes both head and tail.
            self.tail = Some(idx);
        }
        self.head = Some(idx);
    }

    /// Move an existing node to the head (mark as MRU). O(1).
    fn touch(&mut self, idx: usize) {
        if self.head == Some(idx) {
            return; // Already MRU.
        }
        self.unlink(idx);
        self.push_front(idx);
    }

    /// Allocate a node slot, reusing a freed index if available. Returns
    /// the chosen index; the caller writes the `LruNode` into it.
    fn alloc_slot(&mut self, node: LruNode) -> usize {
        if let Some(idx) = self.free_list.pop() {
            self.nodes[idx] = Some(node);
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(Some(node));
            idx
        }
    }

    /// Pop the tail (LRU) entry, removing it from the map and freeing
    /// its node slot. No-op when the cache is empty.
    fn evict_lru(&mut self) {
        let Some(tail_idx) = self.tail else {
            return;
        };
        self.unlink(tail_idx);
        let node = self.nodes[tail_idx]
            .take()
            .expect("PtxCache: tail pointed at a freed slot");
        self.by_key.remove(&node.key);
        self.free_list.push(tail_idx);
        self.evictions = self.evictions.saturating_add(1);
    }

    /// Cache-hit path: look up `key`, verify the stored PTX text matches
    /// `ptx`, and on success mark the entry as MRU and return its cell.
    /// Returns `Some(Err(()))` to signal a hash collision (the stored PTX
    /// did not match) — the caller routes those to the uncached loader,
    /// preserving the long-standing `Slot::Collision` semantics.
    fn get_and_touch(
        &mut self,
        key: (u64, u64),
        ptx: &str,
    ) -> Option<Result<Arc<OnceCell<Arc<CudaModuleInner>>>, ()>> {
        let Some(&idx) = self.by_key.get(&key) else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        let n = self.node(idx);
        if n.entry.ptx != ptx {
            // Hash collision (different PTX text). Treat as a miss.
            self.misses = self.misses.saturating_add(1);
            return Some(Err(()));
        }
        let cell = Arc::clone(&n.entry.module);
        self.touch(idx);
        self.hits = self.hits.saturating_add(1);
        Some(Ok(cell))
    }

    /// Insert a fresh (empty) cell for `key` / `ptx`, performing LRU
    /// eviction first if we are at or above `cap`. Returns the inserted
    /// `Arc<OnceCell>`. The caller is responsible for ensuring `key` is
    /// not already present in the map.
    ///
    /// Factored out so the eviction policy can be exercised in isolation
    /// from the global cache, and so the single source of truth lives in
    /// one place. Bumps `self.evictions` whenever a FIFO eviction occurs
    /// — this is the *only* place the eviction counter is updated, so
    /// `ptx_cache_stats` reflects a precise eviction count even under
    /// concurrent inserts.
    fn insert_empty(
        &mut self,
        key: (u64, u64),
        ptx: String,
        cap: usize,
    ) -> Arc<OnceCell<Arc<CudaModuleInner>>> {
        while self.len() >= cap {
            self.evict_lru();
        }
        let cell = Arc::new(OnceCell::new());
        let idx = self.alloc_slot(LruNode {
            prev: None,
            next: None,
            key,
            entry: CacheEntry {
                ptx,
                module: Arc::clone(&cell),
            },
        });
        self.by_key.insert(key, idx);
        self.push_front(idx);
        cell
    }
}

/// Process-wide cache. Initialised lazily on first PTX load.
static PTX_CACHE: OnceLock<Mutex<PtxCache>> = OnceLock::new();

fn ptx_cache() -> &'static Mutex<PtxCache> {
    PTX_CACHE.get_or_init(|| Mutex::new(PtxCache::new()))
}

/// Returns `(hits, misses, evictions)` snapshot of the process-wide PTX cache.
///
/// Observability hook — useful for benchmarking, dashboards, and tests. All
/// three counters are saturating-incremented inside the same `Mutex` critical
/// section that mutates the LRU, so the returned triple is always consistent.
#[must_use]
pub fn ptx_cache_stats() -> (usize, usize, usize) {
    let c = ptx_cache().lock();
    (c.hits as usize, c.misses as usize, c.evictions as usize)
}

/// 128-bit hash of the PTX source, packed into `(hi, lo)`.
///
/// The cache always verifies the full PTX string on a hit (the
/// `entry.ptx == ptx` check in [`CudaModule::from_ptx_with`]), so a hash
/// collision is *correctness-safe* — it triggers the [`Slot::Collision`]
/// path and falls back to an uncached compile. The cost of that fallback
/// is one extra `cuModuleLoadDataEx` (~10 ms), which is the entire reason
/// to upgrade to 128 bits here: the previous 64-bit `DefaultHasher` key
/// had a birthday-paradox collision probability of ~1 in 2^32 over the
/// distinct PTX strings a long-running process sees, which is well within
/// reach for a busy analytical workload and would manifest as sporadic
/// ~10 ms latency spikes on cached queries.
///
/// We use two `DefaultHasher` instances domain-separated by a leading
/// byte (`0x10` vs `0x20`) so the two 64-bit halves are independent.
/// `DefaultHasher` is internally SipHash-1-3; it is non-cryptographic
/// but more than adequate for collision-resistance against our own
/// deterministic PTX output. The `Slot::Collision` fallback is retained
/// as defence-in-depth — at 128 bits it is effectively unreachable.
fn hash_ptx(ptx: &str) -> (u64, u64) {
    use std::collections::hash_map::DefaultHasher;
    let mut hi = DefaultHasher::new();
    hi.write_u8(0x10);
    hi.write(ptx.as_bytes());
    let mut lo = DefaultHasher::new();
    lo.write_u8(0x20);
    lo.write(ptx.as_bytes());
    (hi.finish(), lo.finish())
}

/// Loaded GPU module — owns one or more CUfunctions.
#[derive(Clone)]
#[must_use]
pub struct CudaModule {
    inner: Arc<CudaModuleInner>,
    // Make `CudaModule: !Sync` at the type level. The `Send` impl below is
    // intentional, but we do NOT want `&CudaModule` to be shared across
    // threads (see the comment on the `unsafe impl Send` block). A
    // `PhantomData<Cell<()>>` is the standard zero-sized opt-out from `Sync`.
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

impl CudaModule {
    /// Load PTX source into a module. The PTX must be a complete, valid module.
    ///
    /// Repeated calls with identical PTX text are served from a process-wide
    /// cache (see module docs), skipping the expensive `cuModuleLoadDataEx`
    /// PTXAS assembly step entirely.
    ///
    /// On failure the driver's PTXAS error log (which usually includes line
    /// numbers for malformed instructions) is appended to the returned error.
    pub fn from_ptx(ptx: &str) -> BoltResult<Self> {
        Self::from_ptx_with(ptx, Self::load_uncached)
    }

    /// Shared cache logic, parameterised over the loader. Production code
    /// always supplies `Self::load_uncached`; tests inject a counting / stub
    /// loader so they can assert on race behaviour without a real GPU.
    fn from_ptx_with<L>(ptx: &str, loader: L) -> BoltResult<Self>
    where
        L: FnOnce(&str) -> BoltResult<Self>,
    {
        // Phase 1: under the lock, locate (or create) a `OnceCell` for this
        // PTX. We do NOT run the compile here — the lock is released as soon
        // as we have an `Arc<OnceCell>` so other threads (including ones
        // working on a different key) are not blocked behind PTXAS.
        //
        // Classify the slot via a small owned enum so the immutable borrow
        // of `cache.map` from the lookup ends before we (potentially) take
        // a mutable borrow to evict and insert.
        let key = hash_ptx(ptx);
        enum Slot {
            Reuse(Arc<OnceCell<Arc<CudaModuleInner>>>),
            Collision,
            Miss,
        }
        let cell: Arc<OnceCell<Arc<CudaModuleInner>>> = {
            let mut cache = ptx_cache().lock();
            // `get_and_touch` both classifies the slot AND, on a hit,
            // re-orders the LRU to mark this entry as most-recently-used.
            // We must do that bump *inside* the lock so concurrent threads
            // see a consistent list. The match below collapses the three
            // possible outcomes onto our existing `Slot` enum.
            let slot = match cache.get_and_touch(key, ptx) {
                Some(Ok(cell)) => Slot::Reuse(cell),
                Some(Err(())) => Slot::Collision,
                None => Slot::Miss,
            };
            // Counter updates live inside the same critical section as the
            // classification above so a concurrent reader of
            // `ptx_cache_stats` cannot observe a half-updated triple. A
            // collision still counts as a miss for stats purposes — the
            // caller pays the full compile cost on this path.
            match slot {
                Slot::Reuse(cell) => {
                    cache.hits += 1;
                    cell
                }
                Slot::Collision => {
                    cache.misses += 1;
                    // 64-bit hash collision against a different PTX string —
                    // astronomically rare at our cache sizes. Drop the lock
                    // and serve this caller from a one-shot uncached load.
                    drop(cache);
                    return loader(ptx);
                }
                Slot::Miss => {
                    // Fresh miss: insert an empty cell, LRU-evict if at cap.
                    let cap = ptx_cache_cap();
                    cache.insert_empty(key, ptx.to_owned(), cap)
                }
            }
        };

        // Phase 2: initialise the cell outside the cache lock. The first
        // thread to reach this point for a given cell runs `loader`; any
        // other thread that holds the same `Arc<OnceCell>` blocks inside
        // `get_or_try_init` until the first thread finishes, and then
        // observes the cached module. If `loader` returns Err the cell
        // stays empty, so subsequent calls retry the compile rather than
        // permanently poisoning the cache slot.
        let inner = cell
            .get_or_try_init(|| loader(ptx).map(|m| m.inner))
            .map(Arc::clone)?;
        Ok(Self {
            inner,
            _not_sync: PhantomData,
        })
    }

    /// Internal: drive `cuModuleLoadDataEx` and wrap the resulting handle in
    /// an `Arc<CudaModuleInner>`. Used only by the cache miss path.
    fn load_uncached(ptx: &str) -> BoltResult<Self> {
        let ptx_cstr = CString::new(ptx).map_err(|e| {
            BoltError::Cuda(format!("PTX source contains interior NUL byte: {}", e))
        })?;

        let mut info_buf: Vec<u8> = vec![0u8; JIT_LOG_BUF_SIZE];
        let mut error_buf: Vec<u8> = vec![0u8; JIT_LOG_BUF_SIZE];

        // Options array: keep order in sync with `values` below.
        let mut options: [CUjit_option; 4] = [
            CU_JIT_INFO_LOG_BUFFER,
            CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
            CU_JIT_ERROR_LOG_BUFFER,
            CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        ];

        // The CUDA driver reads each option value as a `void*`-sized slot. For
        // sizes we pass the integer bit-pattern in the pointer slot, which is
        // the documented contract for `*_SIZE_BYTES` options.
        // CUDA option values use pointer-sized slots; the option SEMANTICS
        // require the value to fit in u32 for *_SIZE_BYTES options, but the
        // casting through `usize` is correct.
        let info_size_slot = JIT_LOG_BUF_SIZE as usize as *mut c_void;
        let error_size_slot = JIT_LOG_BUF_SIZE as usize as *mut c_void;
        let mut values: [*mut c_void; 4] = [
            info_buf.as_mut_ptr() as *mut c_void,
            info_size_slot,
            error_buf.as_mut_ptr() as *mut c_void,
            error_size_slot,
        ];

        let mut module: CUmodule = ptr::null_mut();
        let code = unsafe {
            cuda_sys::cuModuleLoadDataEx(
                &mut module,
                ptx_cstr.as_ptr() as *const c_void,
                options.len() as libc::c_uint,
                options.as_mut_ptr(),
                values.as_mut_ptr(),
            )
        };

        if let Err(e) = cuda_sys::check(code) {
            let ptxas_msg = decode_log(&error_buf);
            let detail = if ptxas_msg.is_empty() {
                inner_msg(&e)
            } else {
                format!("{}; ptxas log: {}", inner_msg(&e), ptxas_msg)
            };
            // Stage 5 (M3L5): preserve the raw `CUresult` integer from
            // the underlying `check()` call so downstream pattern-match
            // code (`mem_pool::is_oom_error` and any future code-aware
            // wrappers) keeps working transparently through this
            // re-wrap. Falling back to `Cuda(String)` would erase the
            // code. `inner_code` returns the underlying driver code
            // when `e` is a `CudaWithCode`, or `code` (the local
            // CUresult we just checked) as a safety net.
            return Err(BoltError::CudaWithCode {
                code: inner_code(&e).unwrap_or(code),
                message: format!("cuModuleLoadDataEx failed: {}", detail),
            });
        }

        Ok(Self {
            inner: Arc::new(CudaModuleInner { raw: module }),
            _not_sync: PhantomData,
        })
    }

    /// Look up an entry point by name.
    pub fn function(&self, name: &str) -> BoltResult<CudaFunction<'_>> {
        let name_cstr = CString::new(name).map_err(|e| {
            BoltError::Cuda(format!(
                "kernel name contains interior NUL byte: {}",
                e
            ))
        })?;
        let mut f: CUfunction = ptr::null_mut();
        let code = unsafe {
            cuda_sys::cuModuleGetFunction(&mut f, self.inner.raw, name_cstr.as_ptr())
        };
        cuda_sys::check(code).map_err(|e| {
            // Stage 5 (M3L5): forward the raw `CUresult` integer through
            // the rewrap so callers can still recognise specific driver
            // errors (e.g. `CUDA_ERROR_NOT_FOUND` for a missing entry
            // point) without parsing the formatted string.
            BoltError::CudaWithCode {
                code: inner_code(&e).unwrap_or(code),
                message: format!(
                    "cuModuleGetFunction({}) failed: {}",
                    name,
                    inner_msg(&e)
                ),
            }
        })?;
        Ok(CudaFunction {
            raw: f,
            _marker: PhantomData,
        })
    }

    /// Raw handle accessor for downstream submodules.
    pub fn raw(&self) -> CUmodule {
        self.inner.raw
    }
}

// SAFETY: `CudaModule` is now just an `Arc<CudaModuleInner>`. The inner type
// already asserts `Send + Sync` (see above), and `Arc<T: Send + Sync>` is both
// `Send` and `Sync` automatically — but we keep this `Send` impl explicit to
// match the prior surface and to make the intent unambiguous to readers.
unsafe impl Send for CudaModule {}
// Not Sync — we don't want concurrent mutation across threads. (The inner
// module *is* Sync, but exposing the wrapper as Sync would invite call sites
// to share `&CudaModule` across threads, and `function()` returns a borrow
// tied to `&self` that we'd rather not have aliased across threads.)
// This is enforced at the type level by the `_not_sync: PhantomData<Cell<()>>`
// field on `CudaModule`, which makes the struct `!Sync` automatically — no
// explicit `unsafe impl !Sync` (which would require the unstable
// `negative_impls` feature) is needed.

/// Borrowed handle to a kernel within a `CudaModule`. Lifetime tied to the module.
#[derive(Clone, Copy)]
#[must_use]
pub struct CudaFunction<'a> {
    raw: CUfunction,
    _marker: PhantomData<&'a CudaModule>,
}

impl<'a> CudaFunction<'a> {
    /// Raw handle accessor for downstream submodules (e.g. kernel launch).
    pub fn raw(&self) -> CUfunction {
        self.raw
    }
}

/// One-shot: load PTX and return the module. Caller invokes `.function(entry)`.
pub fn compile_and_load(ptx: &str) -> BoltResult<CudaModule> {
    CudaModule::from_ptx(ptx)
}

/// Extract the human-readable portion of a CUDA-flavoured `BoltError`
/// for wrapping into a more specific error message.
///
/// Stage 5 (M3L5): aware of both the legacy `Cuda(String)` shape and the
/// typed `CudaWithCode { code, message }` shape introduced in Stage 4.
/// For `CudaWithCode` we surface just the inner `message` (not the
/// formatted `"CUDA driver error <code>: <msg>"`) so a subsequent
/// `format!("X failed: {}", inner_msg(...))` doesn't double-print the
/// "CUDA driver error" prefix — and so the outer wrapper can re-emit
/// `CudaWithCode` with a clean message that the Display impl renders
/// uniformly.
fn inner_msg(e: &BoltError) -> String {
    match e {
        BoltError::Cuda(msg) => msg.clone(),
        BoltError::CudaWithCode { message, .. } => message.clone(),
        other => other.to_string(),
    }
}

/// Extract the raw `CUresult` integer from a `CudaWithCode` error, if
/// the error is of that shape. Used by wrappers around `cuda_sys::check`
/// to forward the driver code through layered error contexts so a
/// downstream caller can still recognise e.g. `CUDA_ERROR_OUT_OF_MEMORY`
/// or `CUDA_ERROR_NOT_FOUND` without parsing the formatted string.
fn inner_code(e: &BoltError) -> Option<i32> {
    match e {
        BoltError::CudaWithCode { code, .. } => Some(*code),
        _ => None,
    }
}

/// Decode a NUL-terminated driver log buffer into a trimmed `String`.
fn decode_log(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).trim().to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// These tests cover the cache state machine in isolation — they do NOT invoke
// the real CUDA driver. The "loader" indirection on `from_ptx_with` lets us
// substitute a stub loader that returns a `CudaModule` whose inner handle is
// `ptr::null_mut()`; `CudaModuleInner::Drop` short-circuits on null, so no
// real GPU resource is ever allocated or freed.
//
// The tests share a process-wide `PTX_CACHE` static. To stay independent of
// run order each test uses unique PTX strings (a unique tag suffix) so cache
// entries from one test cannot satisfy a lookup in another.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;
    use std::thread;

    /// Build a stub `CudaModule` with a null inner handle. `CudaModuleInner::Drop`
    /// short-circuits on null, so this is safe to drop without a CUDA context.
    fn stub_module() -> CudaModule {
        CudaModule {
            inner: Arc::new(CudaModuleInner {
                raw: ptr::null_mut(),
            }),
            _not_sync: PhantomData,
        }
    }

    // -- parse_cap (P3, env-var parsing in isolation) ----------------------

    #[test]
    fn parse_cap_unset_returns_default() {
        assert_eq!(parse_cap(None, 256), 256);
    }

    #[test]
    fn parse_cap_empty_returns_default() {
        assert_eq!(parse_cap(Some(""), 256), 256);
    }

    #[test]
    fn parse_cap_zero_returns_default() {
        // Zero would disable the cache entirely; treat as misconfiguration.
        assert_eq!(parse_cap(Some("0"), 256), 256);
    }

    #[test]
    fn parse_cap_garbage_returns_default() {
        assert_eq!(parse_cap(Some("not-a-number"), 256), 256);
        assert_eq!(parse_cap(Some("-1"), 256), 256);
        assert_eq!(parse_cap(Some("12.5"), 256), 256);
    }

    #[test]
    fn parse_cap_valid_returns_parsed() {
        assert_eq!(parse_cap(Some("1"), 256), 1);
        assert_eq!(parse_cap(Some("8"), 256), 8);
        assert_eq!(parse_cap(Some("4096"), 256), 4096);
    }

    /// End-to-end env-var hookup: `std::env::set_var` round-tripped through
    /// `std::env::var().ok()` → `parse_cap` lands at the configured value.
    /// We test against `parse_cap` rather than the global `ptx_cache_cap()`
    /// because the latter is memoized via `OnceLock` for the process and so
    /// cannot be re-tested with different env-var values inside one binary.
    #[test]
    fn parse_cap_picks_up_env_var() {
        let key = "CRATON_BOLT_PTX_CACHE_CAP_TEST_ENV";
        // SAFETY: set_var is safe on Windows; on Unix it's documented as
        // unsound across threads, but cargo test runs each #[test] on a
        // dedicated thread and we never read this var from another thread.
        std::env::set_var(key, "8");
        let raw = std::env::var(key).ok();
        assert_eq!(parse_cap(raw.as_deref(), PTX_CACHE_CAP_DEFAULT), 8);
        std::env::remove_var(key);
    }

    // -- PtxCache eviction (LRU at the configured cap) --------------------

    /// LRU semantics with 128-bit `(u64, u64)` keys: cap=2; insert A B,
    /// access A (bumps to MRU), insert C — B should be evicted (LRU after
    /// the bump), not A. This is the load-bearing distinction vs FIFO.
    #[test]
    fn ptx_cache_evicts_oldest_at_cap() {
        let mut cache = PtxCache::new();
        let cap = 2usize;

        // Synthesise distinct (u64, u64) keys with the second half pinned to 0.
        let k = |i: u64| (i, 0u64);

        // Insert A, B → list (MRU → LRU): B, A. Cache at cap.
        cache.insert_empty(k(0), "ptx-A".to_owned(), cap);
        cache.insert_empty(k(1), "ptx-B".to_owned(), cap);
        assert_eq!(cache.len(), cap);
        assert_eq!(cache.evictions(), 0);

        // Insert C without touching A — A is LRU and gets evicted.
        cache.insert_empty(k(2), "ptx-C".to_owned(), cap);
        assert!(!cache.by_key.contains_key(&k(0)), "A should have been evicted");
        assert!(cache.by_key.contains_key(&k(1)));
        assert!(cache.by_key.contains_key(&k(2)));
        assert_eq!(cache.evictions(), 1);

        // Reset for the LRU-specific case.
        let mut cache = PtxCache::new();
        cache.insert_empty(k(0), "ptx-A".to_owned(), cap);
        cache.insert_empty(k(1), "ptx-B".to_owned(), cap);
        cache.insert_empty(k(2), "ptx-C".to_owned(), cap); // evict A
        assert!(!cache.by_key.contains_key(&k(0)));

        // ACCESS B → bump to MRU. C is now the LRU.
        let _ = cache.get_and_touch(k(1), "ptx-B").expect("B is still cached");

        // Insert D — must evict C (LRU after the bump), NOT B.
        cache.insert_empty(k(3), "ptx-D".to_owned(), cap);
        assert!(
            !cache.by_key.contains_key(&k(2)),
            "C should have been LRU-evicted after B was touched"
        );
        assert!(cache.by_key.contains_key(&k(1)));
        assert!(cache.by_key.contains_key(&k(3)));
    }

    /// Classic LRU re-ordering with 128-bit keys.
    #[test]
    fn ptx_cache_lru_reordering_keeps_touched_entries() {
        let mut cache = PtxCache::new();
        let cap = 3usize;
        let k = |i: u64| (i, 0u64);

        cache.insert_empty(k(10), "ptx-A".to_owned(), cap);
        cache.insert_empty(k(11), "ptx-B".to_owned(), cap);
        cache.insert_empty(k(12), "ptx-C".to_owned(), cap);
        assert_eq!(cache.len(), cap);

        let _ = cache.get_and_touch(k(10), "ptx-A").expect("A is cached");

        cache.insert_empty(k(13), "ptx-D".to_owned(), cap);
        assert!(cache.by_key.contains_key(&k(10)), "A must survive — just touched");
        assert!(!cache.by_key.contains_key(&k(11)), "B must be evicted as LRU");
        assert!(cache.by_key.contains_key(&k(12)));
        assert!(cache.by_key.contains_key(&k(13)));
        assert_eq!(cache.evictions(), 1);
    }

    /// get_and_touch distinguishes hit / collision / miss for (u64,u64) keys.
    #[test]
    fn ptx_cache_get_and_touch_detects_collision() {
        let mut cache = PtxCache::new();
        let cap = 4usize;
        let k = |i: u64| (i, 0u64);
        cache.insert_empty(k(42), "stored ptx".to_owned(), cap);

        let hit = cache.get_and_touch(k(42), "stored ptx");
        assert!(matches!(hit, Some(Ok(_))));

        let collision = cache.get_and_touch(k(42), "DIFFERENT ptx");
        assert!(matches!(collision, Some(Err(()))));

        let miss = cache.get_and_touch(k(43), "anything");
        assert!(miss.is_none());
    }

    // -- from_ptx_with concurrency (H3, no redundant compile on miss) ------

    /// Many threads racing on the *same* PTX must invoke the loader exactly
    /// once — the `OnceCell` in the cache entry serialises the compile so
    /// late-arriving threads block on the in-flight compile rather than
    /// kicking off a second one. Before the H3 fix this counter would
    /// typically reach `N` under contention.
    #[test]
    fn from_ptx_compiles_once_under_contention() {
        // Unique PTX so this test isn't satisfied by any other test's entry.
        let ptx = "// H3 contention test — unique tag a7c5e91b3f024d8e".to_string();
        let calls = Arc::new(AtomicUsize::new(0));
        let n_threads = 16;
        let barrier = Arc::new(Barrier::new(n_threads));

        let mut handles = Vec::with_capacity(n_threads);
        for _ in 0..n_threads {
            let ptx = ptx.clone();
            let calls = Arc::clone(&calls);
            let barrier = Arc::clone(&barrier);
            handles.push(thread::spawn(move || {
                // Wait so all threads hit the cache lookup roughly together.
                barrier.wait();
                let calls = Arc::clone(&calls);
                CudaModule::from_ptx_with(&ptx, move |_| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    // Make the "compile" non-trivial so the race window is
                    // wide enough to be meaningful on fast hardware.
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    Ok(stub_module())
                })
                .expect("stub loader cannot fail")
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "loader must run exactly once across all racing threads"
        );
    }

    /// After a successful load, subsequent lookups for the same PTX must
    /// return the cached module without invoking the loader at all.
    #[test]
    fn from_ptx_hits_cache_on_repeat() {
        let ptx = "// H3 cache-hit test — unique tag 3d92f8a17ce04b06".to_string();
        let calls = Arc::new(AtomicUsize::new(0));

        // First call: cold miss, loader fires once.
        {
            let calls = Arc::clone(&calls);
            CudaModule::from_ptx_with(&ptx, move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(stub_module())
            })
            .unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Subsequent calls: warm hit, loader must NOT fire.
        for _ in 0..5 {
            let calls = Arc::clone(&calls);
            CudaModule::from_ptx_with(&ptx, move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(stub_module())
            })
            .unwrap();
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "warm-hit lookups must not re-invoke the loader"
        );
    }

    // -- Stage 5 (M3L5) error-shape migration -------------------------------
    //
    // The PTX cache + module-load path previously surfaced driver errors as
    // `BoltError::Cuda(format!("CUDA driver error N: ..."))`. Stage 5 routes
    // them through `BoltError::CudaWithCode { code, message }` instead so
    // downstream code (`mem_pool::is_oom_error`) can pattern-match the raw
    // `CUresult` integer without parsing the formatted string.

    /// `inner_msg` extracts the message body from BOTH the legacy
    /// `Cuda(String)` variant AND the new `CudaWithCode { message, .. }`
    /// variant — without re-prepending the "CUDA driver error <code>:"
    /// prefix that the Display impl adds.
    #[test]
    fn inner_msg_handles_both_cuda_variants() {
        let legacy = BoltError::Cuda("legacy text".to_string());
        assert_eq!(inner_msg(&legacy), "legacy text");

        let typed = BoltError::CudaWithCode {
            code: 7,
            message: "typed text".to_string(),
        };
        // Just the message body — not "CUDA driver error 7: typed text".
        assert_eq!(inner_msg(&typed), "typed text");

        let other = BoltError::Other("misc".to_string());
        // Non-CUDA variants fall back to the Display rendering.
        assert_eq!(inner_msg(&other), "misc");
    }

    /// `inner_code` returns `Some(code)` for `CudaWithCode` and `None`
    /// otherwise. The from_ptx / function() wrappers rely on this to
    /// forward the underlying `CUresult` integer through their re-wrap so
    /// callers can still recognise specific driver errors after layering.
    #[test]
    fn inner_code_extracts_cuda_with_code_integer() {
        let typed = BoltError::CudaWithCode {
            code: 2,
            message: "OOM".to_string(),
        };
        assert_eq!(inner_code(&typed), Some(2));

        let legacy = BoltError::Cuda("anything".to_string());
        assert_eq!(inner_code(&legacy), None);

        let other = BoltError::Other("misc".to_string());
        assert_eq!(inner_code(&other), None);
    }

    // -- ptx_cache_stats observability hook --------------------------------

    /// `ptx_cache_stats` returns a `(hits, misses, evictions)` triple
    /// reflecting the process-wide cache counters. Because the cache is
    /// shared across tests *and* cargo runs tests concurrently, the test
    /// works in *lower-bound deltas* from a snapshot taken at the top:
    /// each of our operations must bump its counter by *at least* its
    /// contribution. Other concurrent tests can only push the absolute
    /// values higher, never lower — counters are monotonic — so a
    /// lower-bound assertion is robust to parallel test execution.
    ///
    /// Scenario:
    /// 1. Insert one fresh entry via stub loader → `misses` advances by
    ///    >= 1.
    /// 2. Look up the same PTX again → `hits` advances by >= 1, and the
    ///    loader does NOT fire (verified via a sentinel counter, which
    ///    is unaffected by other tests).
    /// 3. Insert `cap + 1` further unique entries → `evictions` advances
    ///    by >= 1 and `misses` by >= `cap + 1`.
    #[test]
    fn ptx_cache_stats_reports_hits_misses_evictions() {
        let (h0, m0, e0) = ptx_cache_stats();

        // Step 1: cold miss on a unique key.
        let ptx_a = "// ptx_cache_stats test A — unique tag 41c6b7d92f8e0a13".to_string();
        CudaModule::from_ptx_with(&ptx_a, |_| Ok(stub_module())).unwrap();
        let (h1, m1, e1) = ptx_cache_stats();
        assert!(
            m1 >= m0 + 1,
            "fresh miss must advance misses by >= 1 (m0={}, m1={})",
            m0,
            m1
        );
        assert!(h1 >= h0, "hits counter must be monotonic");
        assert!(e1 >= e0, "evictions counter must be monotonic");

        // Step 2: warm hit on the same key. The sentinel `fired` is
        // private to this test (closure-local AtomicUsize), so it is
        // unaffected by any concurrent test running on the global cache —
        // this gives us an authoritative "did the loader actually run"
        // signal independent of the global counters.
        let fired = Arc::new(AtomicUsize::new(0));
        {
            let fired = Arc::clone(&fired);
            CudaModule::from_ptx_with(&ptx_a, move |_| {
                fired.fetch_add(1, Ordering::SeqCst);
                Ok(stub_module())
            })
            .unwrap();
        }
        assert_eq!(
            fired.load(Ordering::SeqCst),
            0,
            "warm hit must not invoke the loader"
        );
        let (h2, m2, e2) = ptx_cache_stats();
        assert!(
            h2 >= h1 + 1,
            "warm hit must advance hits by >= 1 (h1={}, h2={})",
            h1,
            h2
        );
        assert!(m2 >= m1, "misses counter must be monotonic");
        assert!(e2 >= e1, "evictions counter must be monotonic");

        // Step 3: force at least one eviction by inserting `cap + 1` more
        // unique entries. The cap is process-frozen via OnceLock so this
        // matches the cap the cache itself is using. We assert lower
        // bounds (not equality) because pre-existing entries from other
        // tests and concurrent test threads may both push the actual
        // delta higher than our contribution alone.
        let cap = ptx_cache_cap();
        let burst = cap + 1;
        for i in 0..burst {
            let ptx = format!(
                "// ptx_cache_stats test eviction burst {} — unique tag bd7e54f08a2c91{:04x}",
                i, i
            );
            CudaModule::from_ptx_with(&ptx, |_| Ok(stub_module())).unwrap();
        }
        let (h3, m3, e3) = ptx_cache_stats();
        assert!(
            m3 >= m2 + burst,
            "burst of {} unique keys must advance misses by >= {} \
             (m2={}, m3={})",
            burst,
            burst,
            m2,
            m3,
        );
        assert!(
            e3 >= e2 + 1,
            "inserting cap+1 fresh entries into a cache already at \
             least at cap (we just inserted `cap+1` plus prior step's \
             entry) must produce >= 1 eviction (e2={}, e3={}, cap={})",
            e2,
            e3,
            cap,
        );
        assert!(h3 >= h2, "hits counter must be monotonic");
    }

    /// A loader that fails leaves the cell empty rather than poisoning the
    /// slot — the next caller retries the compile. Without this property a
    /// transient driver hiccup would permanently break cached-key compiles.
    #[test]
    fn from_ptx_failed_compile_does_not_poison_cell() {
        let ptx = "// H3 failure-retry test — unique tag 8e1a4f0b9d375c22".to_string();
        let calls = Arc::new(AtomicUsize::new(0));

        // First call: loader returns Err, count = 1.
        {
            let calls = Arc::clone(&calls);
            let res = CudaModule::from_ptx_with(&ptx, move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                Err(BoltError::Cuda("simulated PTXAS failure".into()))
            });
            assert!(res.is_err());
        }

        // Second call: loader fires again (cell was not poisoned), count = 2.
        {
            let calls = Arc::clone(&calls);
            CudaModule::from_ptx_with(&ptx, move |_| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(stub_module())
            })
            .unwrap();
        }
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
