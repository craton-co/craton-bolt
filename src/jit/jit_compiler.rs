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
//! a process-wide cache keyed by a 64-bit hash of the PTX text.
//!
//! **Invariant.** The codegen pipeline is deterministic: for a given
//! `(PhysicalPlan, kernel_name)` pair the emitted PTX text is byte-identical
//! across runs within a process. Therefore hashing the PTX text and reusing
//! the loaded `CUmodule` is sound — two cache lookups that collide on the
//! hash *and* the full string match represent literally the same program.
//! Hash collisions on the 64-bit DefaultHasher key are astronomically
//! unlikely for the cache sizes we use (cap = 256), and even on collision
//! the worst case is a spurious cache miss on the second program: we only
//! match the hash here, but every cached entry was inserted from a PTX
//! string we ourselves produced, so reusing a colliding module would launch
//! the wrong kernel. We therefore guard against that by *also* keying on the
//! PTX string itself: the map's `value` retains an `Arc<CudaModuleInner>`
//! and we additionally compare the stored PTX text on lookup. (See
//! `CacheEntry` below.)
//!
//! The cache is bounded at 256 entries with FIFO eviction — LRU is overkill
//! for what is essentially a hot-set of recently-issued query shapes, and
//! FIFO needs only a `VecDeque<u64>` companion to the `HashMap`. When an
//! entry is evicted from the cache its `Arc` strong count drops; if no
//! `CudaModule` clones are live the underlying `CudaModuleInner::Drop`
//! runs and calls `cuModuleUnload`. If clones *are* live the module stays
//! loaded until the last clone is dropped — exactly the lifetime users
//! expect.

use std::collections::{HashMap, VecDeque};
use std::ffi::CString;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ptr;
use std::sync::{Arc, OnceLock};

use libc::c_void;
use parking_lot::Mutex;

use crate::cuda::cuda_sys::{self, CUfunction, CUmodule};
use crate::error::{BoltError, BoltResult};

// --- CUjit_option constants -------------------------------------------------
// Mirrored from <cuda.h> — declared here (rather than in cuda_sys.rs) per the
// orchestrator's file-ownership boundary. Values are stable ABI.
/// `CUjit_option` value type, as expected by `cuModuleLoadDataEx`.
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

/// Cap on the number of cached compiled modules. Eviction is FIFO once we
/// exceed this many entries.
const PTX_CACHE_CAP: usize = 256;

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
            // FIXME(orchestrator): use tracing/log once added as dep.
            // Neither `tracing` nor `log` is in Cargo.toml today, so we still
            // route this through stderr. Library consumers will want a proper
            // logging facade — swap this `eprintln!` the moment one lands.
            eprintln!(
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

/// One cache slot: the loaded module plus the PTX text that produced it.
/// We retain the source text so a hash collision can be detected (see the
/// module-level comment).
struct CacheEntry {
    ptx: String,
    module: Arc<CudaModuleInner>,
}

/// Cache state: a `HashMap` for O(1) lookup plus a `VecDeque` of keys in
/// insertion order for FIFO eviction.
struct PtxCache {
    map: HashMap<u64, CacheEntry>,
    order: VecDeque<u64>,
}

impl PtxCache {
    fn new() -> Self {
        Self {
            map: HashMap::with_capacity(PTX_CACHE_CAP),
            order: VecDeque::with_capacity(PTX_CACHE_CAP),
        }
    }
}

/// Process-wide cache. Initialised lazily on first PTX load.
static PTX_CACHE: OnceLock<Mutex<PtxCache>> = OnceLock::new();

fn ptx_cache() -> &'static Mutex<PtxCache> {
    PTX_CACHE.get_or_init(|| Mutex::new(PtxCache::new()))
}

/// 64-bit hash of the PTX source. We use the std default hasher; it is not
/// cryptographic but the cache treats hash collisions as misses (we also
/// compare the full PTX text in the hit path), so any reasonable hash works.
fn hash_ptx(ptx: &str) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    ptx.hash(&mut h);
    h.finish()
}

/// Loaded GPU module — owns one or more CUfunctions.
#[derive(Clone)]
pub struct CudaModule {
    inner: Arc<CudaModuleInner>,
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
        // Fast path: look up by hash, confirm via full-text compare to defend
        // against the (astronomically unlikely) 64-bit collision.
        let key = hash_ptx(ptx);
        {
            let cache = ptx_cache().lock();
            if let Some(entry) = cache.map.get(&key) {
                if entry.ptx == ptx {
                    return Ok(Self {
                        inner: Arc::clone(&entry.module),
                    });
                }
                // Collision: fall through to recompile. We do not attempt to
                // store a second entry under the same key — at FIFO cap 256
                // and 64-bit hashes a collision is a once-in-the-lifetime
                // event and an extra recompile per call is acceptable.
            }
        }

        // Slow path: actually invoke the driver to assemble the PTX.
        let module = Self::load_uncached(ptx)?;

        // Insert into the cache, evicting the oldest entry if we're at cap.
        // We do not check whether another thread raced us and inserted the
        // same key concurrently: an overwrite is harmless (both `Arc`s point
        // to functionally identical modules, and the loser is unloaded when
        // its last clone drops).
        {
            let mut cache = ptx_cache().lock();
            if cache.map.len() >= PTX_CACHE_CAP && !cache.map.contains_key(&key) {
                if let Some(old_key) = cache.order.pop_front() {
                    cache.map.remove(&old_key);
                }
            }
            let entry = CacheEntry {
                ptx: ptx.to_owned(),
                module: Arc::clone(&module.inner),
            };
            if cache.map.insert(key, entry).is_none() {
                cache.order.push_back(key);
            }
        }

        Ok(module)
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
        let info_size_slot = JIT_LOG_BUF_SIZE as u32 as usize as *mut c_void;
        let error_size_slot = JIT_LOG_BUF_SIZE as u32 as usize as *mut c_void;
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
            return Err(BoltError::Cuda(format!(
                "cuModuleLoadDataEx failed: {}",
                detail
            )));
        }

        Ok(Self {
            inner: Arc::new(CudaModuleInner { raw: module }),
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
            BoltError::Cuda(format!(
                "cuModuleGetFunction({}) failed: {}",
                name,
                inner_msg(&e)
            ))
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
// `Send` and `Sync` automatically — but we keep these impls explicit to match
// the prior surface and to make the intent unambiguous to readers.
unsafe impl Send for CudaModule {}
// Not Sync — we don't want concurrent mutation across threads. (The inner
// module *is* Sync, but exposing the wrapper as Sync would invite call sites
// to share `&CudaModule` across threads, and `function()` returns a borrow
// tied to `&self` that we'd rather not have aliased across threads.)

/// Borrowed handle to a kernel within a `CudaModule`. Lifetime tied to the module.
#[derive(Clone, Copy)]
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

/// Extract the human-readable portion of a `BoltError::Cuda` for wrapping.
fn inner_msg(e: &BoltError) -> String {
    match e {
        BoltError::Cuda(msg) => msg.clone(),
        other => other.to_string(),
    }
}

/// Decode a NUL-terminated driver log buffer into a trimmed `String`.
fn decode_log(buf: &[u8]) -> String {
    let end = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    String::from_utf8_lossy(&buf[..end]).trim().to_string()
}
