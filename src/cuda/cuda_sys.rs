// SPDX-License-Identifier: Apache-2.0

//! Raw FFI bindings and thin safe wrappers around the CUDA Driver API.
//!
//! Real builds link `cuda` from the installed CUDA Toolkit. When the
//! `cuda-stub` feature is enabled, the `#[link]` block is omitted and every
//! FFI entry point is replaced by a Rust shim that returns
//! [`CUDA_ERROR_STUB`]; [`check`] converts that into
//! `BoltError::Other("cuda-stub mode: no GPU support compiled in")`.
//! Stub mode lets the crate compile on hosts without the CUDA toolkit and on
//! docs.rs.

use std::ffi::CStr;

use bytemuck::Pod;
use libc::{c_char, c_int, c_uint, c_void};

use crate::error::{BoltError, BoltResult};

/// CUDA driver error code (0 == success).
pub type CUresult = i32;
/// Opaque device handle.
pub type CUdevice = i32;
/// Opaque context handle.
pub type CUcontext = *mut c_void;
/// Device pointer (always 64-bit in the v2 ABI).
pub type CUdeviceptr = u64;
/// Device attribute enum value.
#[allow(non_camel_case_types)] // reason: must match the CUDA C ABI name verbatim
pub type CUdevice_attribute = i32;
/// Opaque module handle (loaded PTX/cubin).
pub type CUmodule = *mut c_void;
/// Opaque kernel entry-point handle within a module.
pub type CUfunction = *mut c_void;
/// Opaque stream handle (NULL == default/legacy stream).
pub type CUstream = *mut c_void;
/// Batch 6: opaque CUDA graph handle (a recorded sequence of operations,
/// before instantiation).
pub type CUgraph = *mut c_void;
/// Batch 6: opaque executable graph handle (instantiated form of a `CUgraph`,
/// the thing actually launched on a stream).
pub type CUgraphExec = *mut c_void;

/// `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`: capture mode that scopes its
/// "did anything race?" detection to operations issued from the calling
/// thread. The other two valid modes are `_GLOBAL` (0) and `_RELAXED` (1);
/// thread-local is the right default for the bitonic-sort capture because
/// every kernel launch in the capture sequence happens on the same thread
/// (the executor that called `sort_indices_on_gpu_multi`). `GLOBAL` would
/// erroneously flag concurrent unrelated CUDA work from other engine
/// threads as a capture violation.
pub const CU_STREAM_CAPTURE_MODE_THREAD_LOCAL: u32 = 2;

/// Driver "no error" sentinel.
pub const CUDA_SUCCESS: CUresult = 0;

/// Sentinel error code returned by every FFI shim when the crate is built with
/// the `cuda-stub` feature. Chosen well above the real CUDA driver error range
/// (currently < 1000) so it cannot collide with a legitimate `CUresult`.
pub const CUDA_ERROR_STUB: CUresult = 100_000;

#[cfg(not(feature = "cuda-stub"))]
#[link(name = "cuda")]
extern "C" {
    pub fn cuInit(flags: c_uint) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut c_int) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: c_int) -> CUresult;
    pub fn cuDeviceGetName(name: *mut c_char, len: c_int, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetAttribute(
        pi: *mut c_int,
        attrib: CUdevice_attribute,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    pub fn cuCtxCreate_v2(pctx: *mut CUcontext, flags: c_uint, dev: CUdevice) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    // Stage-4 (GJ): query the device handle bound to the calling thread's
    // current CUDA context. Needed by `gpu_join::resolve_byte_cap_from_driver`
    // so multi-GPU rigs detect the right card's VRAM cap.
    pub fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult;
    /// Stage 5 (M3L5): return the CUDA context currently bound to the
    /// calling thread. Used by the pool-watcher (`mem_pool::pool_watcher`)
    /// to capture the engine thread's context once at spawn time and
    /// re-bind it on the background thread before each `cuMemGetInfo_v2`
    /// poll — the watcher otherwise inherits no current context and the
    /// driver returns `CUDA_ERROR_INVALID_CONTEXT` for every call.
    ///
    /// Writes `NULL` into `*pctx` if no context is current on this
    /// thread; that is NOT an error from the driver's perspective.
    pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemAllocHost_v2(pp: *mut *mut c_void, bytesize: usize) -> CUresult;
    /// Page-locked host allocation with explicit behavior flags (e.g.
    /// `CU_MEMHOSTALLOC_PORTABLE`, `_DEVICEMAP`, `_WRITECOMBINED`). The
    /// flagless `cuMemAllocHost_v2` above is equivalent to passing
    /// `flags = 0`; this entry point is bound for
    /// [`crate::cuda::async_copy::PinnedBuffer`] so callers can opt into
    /// portable / write-combined pinned memory for async H2D/D2H DMA.
    pub fn cuMemHostAlloc(pp: *mut *mut c_void, bytesize: usize, flags: c_uint) -> CUresult;
    pub fn cuMemFreeHost(p: *mut c_void) -> CUresult;
    pub fn cuMemcpyHtoD_v2(dst: CUdeviceptr, src: *const c_void, bytes: usize) -> CUresult;
    pub fn cuMemcpyDtoH_v2(dst: *mut c_void, src: CUdeviceptr, bytes: usize) -> CUresult;
    /// Synchronous device-to-device copy. Added for the incremental
    /// `GpuTable` cache: when `register_batch` appends rows, the engine
    /// allocates a fresh GpuVec sized for old+new rows, DtoD-copies the
    /// unchanged prefix, and HtoD-uploads only the new tail — replacing
    /// the previous full re-upload.
    pub fn cuMemcpyDtoD_v2(dst: CUdeviceptr, src: CUdeviceptr, bytes: usize) -> CUresult;
    pub fn cuMemcpyHtoDAsync_v2(
        dst: CUdeviceptr,
        src: *const c_void,
        bytecount: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoHAsync_v2(
        dst: *mut c_void,
        src: CUdeviceptr,
        bytecount: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemsetD8_v2(dst: CUdeviceptr, value: u8, count: usize) -> CUresult;
    pub fn cuMemsetD8Async(
        dst: CUdeviceptr,
        value: u8,
        count: usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuGetErrorName(error: CUresult, str: *mut *const c_char) -> CUresult;
    pub fn cuGetErrorString(error: CUresult, str: *mut *const c_char) -> CUresult;
    pub fn cuModuleLoadData(module: *mut CUmodule, image: *const c_void) -> CUresult;
    pub fn cuModuleLoadDataEx(
        module: *mut CUmodule,
        image: *const c_void,
        num_options: c_uint,
        options: *mut i32,
        option_values: *mut *mut c_void,
    ) -> CUresult;
    pub fn cuModuleUnload(module: CUmodule) -> CUresult;
    pub fn cuModuleGetFunction(
        func: *mut CUfunction,
        module: CUmodule,
        name: *const c_char,
    ) -> CUresult;
    pub fn cuStreamCreate(stream: *mut CUstream, flags: c_uint) -> CUresult;
    pub fn cuStreamDestroy_v2(stream: CUstream) -> CUresult;
    pub fn cuStreamSynchronize(stream: CUstream) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;
    pub fn cuLaunchKernel(
        f: CUfunction,
        grid_dim_x: c_uint, grid_dim_y: c_uint, grid_dim_z: c_uint,
        block_dim_x: c_uint, block_dim_y: c_uint, block_dim_z: c_uint,
        shared_mem_bytes: c_uint,
        stream: CUstream,
        kernel_params: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> CUresult;
    /// Report the device's currently free and total memory, in bytes,
    /// in the active context. Stage 4 (pool watcher) calls this from a
    /// background thread to drive proactive eviction before the driver
    /// OOMs. The `_v2` suffix mirrors the rest of this block — the
    /// driver exposes both `cuMemGetInfo` and `cuMemGetInfo_v2`;
    /// the v2 form is the documented one on CUDA 12.x.
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;

    // ---------------------------------------------------------------------
    // Batch 6: CUDA Graph stream-capture API.
    //
    // Used by `crate::exec::gpu_sort` to capture the bitonic-sort
    // `O(log^2 n)` launch sequence into a `CUgraph` once per shape, then
    // re-launch the instantiated `CUgraphExec` on every subsequent sort
    // of the same `(n_pow2, dtype)` — eliminates per-launch driver
    // overhead for the steady-state path.
    //
    // All entries are gated behind the `BOLT_SORT_USE_GRAPH=1` env var
    // in `gpu_sort.rs`; the default code path is unchanged.
    // ---------------------------------------------------------------------
    pub fn cuStreamBeginCapture_v2(stream: CUstream, mode: c_uint) -> CUresult;
    pub fn cuStreamEndCapture(stream: CUstream, graph_out: *mut CUgraph) -> CUresult;
    pub fn cuGraphInstantiate_v2(
        graph_exec_out: *mut CUgraphExec,
        graph: CUgraph,
        error_node: *mut c_void,
        log_buffer: *mut c_char,
        buffer_size: usize,
    ) -> CUresult;
    pub fn cuGraphLaunch(graph_exec: CUgraphExec, stream: CUstream) -> CUresult;
    pub fn cuGraphExecDestroy(graph_exec: CUgraphExec) -> CUresult;
    pub fn cuGraphDestroy(graph: CUgraph) -> CUresult;
}

// ---------------------------------------------------------------------------
// `cuda-stub` feature: stand-in implementations so the crate compiles on hosts
// without the CUDA toolkit (including docs.rs). Every shim returns
// `CUDA_ERROR_STUB`, which `check()` maps to `BoltError::Other(...)`.
// ---------------------------------------------------------------------------
#[cfg(feature = "cuda-stub")]
#[allow(non_snake_case, unused_variables)]
mod stubs {
    use super::*;

    pub unsafe fn cuInit(_flags: c_uint) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuDeviceGetCount(_count: *mut c_int) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuDeviceGet(_device: *mut CUdevice, _ordinal: c_int) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuDeviceGetName(_name: *mut c_char, _len: c_int, _dev: CUdevice) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuDeviceGetAttribute(
        _pi: *mut c_int,
        _attrib: CUdevice_attribute,
        _dev: CUdevice,
    ) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuDeviceTotalMem_v2(_bytes: *mut usize, _dev: CUdevice) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuCtxCreate_v2(_pctx: *mut CUcontext, _flags: c_uint, _dev: CUdevice) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuCtxDestroy_v2(_ctx: CUcontext) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuCtxSetCurrent(_ctx: CUcontext) -> CUresult { CUDA_ERROR_STUB }
    // Stage-4 (GJ): mirror of the production `cuCtxGetDevice` for stub builds.
    pub unsafe fn cuCtxGetDevice(_device: *mut CUdevice) -> CUresult { CUDA_ERROR_STUB }
    // Stage 5 (M3L5): mirror of the production `cuCtxGetCurrent` for stub
    // builds. Pool-watcher only invokes this under the real CUDA backend;
    // the stub return path is exercised only by tests that explicitly
    // build `--features cuda-stub`.
    pub unsafe fn cuCtxGetCurrent(_pctx: *mut CUcontext) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemAlloc_v2(_dptr: *mut CUdeviceptr, _bytesize: usize) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemFree_v2(_dptr: CUdeviceptr) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemAllocHost_v2(_pp: *mut *mut c_void, _bytesize: usize) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemHostAlloc(_pp: *mut *mut c_void, _bytesize: usize, _flags: c_uint) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemFreeHost(_p: *mut c_void) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemcpyHtoD_v2(_dst: CUdeviceptr, _src: *const c_void, _bytes: usize) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemcpyDtoH_v2(_dst: *mut c_void, _src: CUdeviceptr, _bytes: usize) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemcpyDtoD_v2(_dst: CUdeviceptr, _src: CUdeviceptr, _bytes: usize) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemcpyHtoDAsync_v2(
        _dst: CUdeviceptr,
        _src: *const c_void,
        _bytecount: usize,
        _stream: CUstream,
    ) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemcpyDtoHAsync_v2(
        _dst: *mut c_void,
        _src: CUdeviceptr,
        _bytecount: usize,
        _stream: CUstream,
    ) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemsetD8_v2(_dst: CUdeviceptr, _value: u8, _count: usize) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemsetD8Async(
        _dst: CUdeviceptr,
        _value: u8,
        _count: usize,
        _stream: CUstream,
    ) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuGetErrorName(_error: CUresult, _str: *mut *const c_char) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuGetErrorString(_error: CUresult, _str: *mut *const c_char) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuModuleLoadData(_module: *mut CUmodule, _image: *const c_void) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuModuleLoadDataEx(
        _module: *mut CUmodule,
        _image: *const c_void,
        _num_options: c_uint,
        _options: *mut i32,
        _option_values: *mut *mut c_void,
    ) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuModuleUnload(_module: CUmodule) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuModuleGetFunction(
        _func: *mut CUfunction,
        _module: CUmodule,
        _name: *const c_char,
    ) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuStreamCreate(_stream: *mut CUstream, _flags: c_uint) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuStreamDestroy_v2(_stream: CUstream) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuStreamSynchronize(_stream: CUstream) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuCtxSynchronize() -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuLaunchKernel(
        _f: CUfunction,
        _grid_dim_x: c_uint, _grid_dim_y: c_uint, _grid_dim_z: c_uint,
        _block_dim_x: c_uint, _block_dim_y: c_uint, _block_dim_z: c_uint,
        _shared_mem_bytes: c_uint,
        _stream: CUstream,
        _kernel_params: *mut *mut c_void,
        _extra: *mut *mut c_void,
    ) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuMemGetInfo_v2(_free: *mut usize, _total: *mut usize) -> CUresult { CUDA_ERROR_STUB }

    // Batch 6: cuGraph stub mirrors. Returning the stub sentinel so
    // `check()` maps every call to `BoltError::Other("cuda-stub mode")`.
    pub unsafe fn cuStreamBeginCapture_v2(_stream: CUstream, _mode: c_uint) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuStreamEndCapture(_stream: CUstream, _graph_out: *mut CUgraph) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuGraphInstantiate_v2(
        _graph_exec_out: *mut CUgraphExec,
        _graph: CUgraph,
        _error_node: *mut c_void,
        _log_buffer: *mut c_char,
        _buffer_size: usize,
    ) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuGraphLaunch(_graph_exec: CUgraphExec, _stream: CUstream) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuGraphExecDestroy(_graph_exec: CUgraphExec) -> CUresult { CUDA_ERROR_STUB }
    pub unsafe fn cuGraphDestroy(_graph: CUgraph) -> CUresult { CUDA_ERROR_STUB }
}

#[cfg(feature = "cuda-stub")]
pub use stubs::*;

/// Convert a `CUresult` into a `BoltResult`, attaching the driver's message.
///
/// Stage 4 (pool): the error now carries the raw `CUresult` integer
/// alongside the human-readable string. Downstream consumers
/// (`mem_pool::is_oom_error`) pattern-match on
/// `BoltError::CudaWithCode { code: 2, .. }` directly instead of
/// scraping the formatted message — see `crate::error::BoltError` for
/// the variant rationale. The Display impl of `CudaWithCode` reproduces
/// the legacy `"CUDA driver error {code}: {message}"` shape, so any
/// caller that still falls through to `.to_string()` keeps working.
pub fn check(code: CUresult) -> BoltResult<()> {
    if code == CUDA_SUCCESS {
        return Ok(());
    }
    if code == CUDA_ERROR_STUB {
        return Err(BoltError::Other(
            "cuda-stub mode: no GPU support compiled in".into(),
        ));
    }
    let msg = unsafe {
        let mut ptr: *const c_char = std::ptr::null();
        if cuGetErrorString(code, &mut ptr) == CUDA_SUCCESS && !ptr.is_null() {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        } else {
            format!("unknown CUDA error {}", code)
        }
    };
    Err(BoltError::CudaWithCode {
        code,
        message: msg,
    })
}

/// Successful-init latch. Stores `true` exactly once when `cuInit(0)`
/// has returned `CUDA_SUCCESS`. Deliberately *not* an `OnceLock<CUresult>`:
/// the old design cached the first result, success or failure, so a
/// transient driver-load hiccup at startup (DSO not yet ready, driver
/// service still coming up, container with a deferred GPU mount, …)
/// poisoned every subsequent `init()` for the lifetime of the process,
/// even after the driver became usable.
///
/// We use a `parking_lot::Mutex<bool>` rather than `OnceLock<()>` so the
/// test path can clear the latch between cases — `OnceLock::take` is not
/// stabilised until Rust 1.79 and our MSRV is 1.74. The fast path stays
/// O(1) and uncontended in production: a single un-poisoned mutex
/// acquire per `init()` call, which itself is called O(1) times per
/// process.
static INIT_OK: parking_lot::Mutex<bool> = parking_lot::Mutex::new(false);

/// Indirection so the inline unit test can substitute a deterministic
/// fake for `cuInit`. Production code always uses [`real_cu_init`].
type CuInitFn = fn() -> CUresult;

fn real_cu_init() -> CUresult {
    unsafe { cuInit(0) }
}

/// Idempotently call `cuInit(0)`. Safe to invoke from any thread.
///
/// Only `CUDA_SUCCESS` is cached — on any error this returns `Err` *and*
/// leaves the cache empty, so the next call retries the driver. That
/// matters for processes where the driver/DSO is only fully usable
/// shortly after startup (deferred container GPU mounts, lazy
/// `libcuda.so` load on first kernel-module probe, etc.).
pub fn init() -> BoltResult<()> {
    init_with(real_cu_init)
}

/// Test-friendly inner: factored out so the host-side unit test can
/// inject a fake `cuInit` and exercise the cache-retry behaviour
/// without a real GPU.
fn init_with(cu_init: CuInitFn) -> BoltResult<()> {
    // Fast path: already latched.
    if *INIT_OK.lock() {
        return Ok(());
    }
    // Slow path: call the driver. We don't hold the lock across the FFI
    // call — a concurrent `init()` from another thread is fine (the
    // driver's own `cuInit(0)` is idempotent; both callers will see the
    // same outcome) and avoids tying GPU-driver latency to a global mutex.
    let code = cu_init();
    if code == CUDA_SUCCESS {
        *INIT_OK.lock() = true;
        Ok(())
    } else {
        // Deliberately do NOT cache the error: the next caller retries.
        check(code)
    }
}

/// Test-only helper: clears the success latch so a fresh `init_with`
/// call starts from a known-empty state. Not exposed outside the crate.
#[cfg(test)]
pub(crate) fn _test_reset_init_cache() {
    *INIT_OK.lock() = false;
}

/// Number of CUDA-capable devices visible to the driver.
pub fn device_count() -> BoltResult<i32> {
    let mut n: c_int = 0;
    check(unsafe { cuDeviceGetCount(&mut n) })?;
    Ok(n as i32)
}

/// Resolve the `ordinal`-th device handle.
pub fn device_get(ordinal: i32) -> BoltResult<CUdevice> {
    let mut dev: CUdevice = 0;
    check(unsafe { cuDeviceGet(&mut dev, ordinal as c_int) })?;
    Ok(dev)
}

/// Total global memory available on `dev`, in bytes. Wraps
/// `cuDeviceTotalMem_v2`. Returns the bytes the device reports — note this is
/// the total physical VRAM, NOT the free portion at the moment of the call.
///
/// Used by `gpu_join` to opt large-VRAM cards into a wider hash-table cap.
pub fn device_total_mem(dev: CUdevice) -> BoltResult<usize> {
    let mut bytes: usize = 0;
    check(unsafe { cuDeviceTotalMem_v2(&mut bytes, dev) })?;
    Ok(bytes)
}

/// Human-readable device name (e.g. "NVIDIA GeForce RTX 4090").
pub fn device_name(dev: CUdevice) -> BoltResult<String> {
    const LEN: usize = 256;
    let mut buf: [c_char; LEN] = [0; LEN];
    check(unsafe { cuDeviceGetName(buf.as_mut_ptr(), LEN as c_int, dev) })?;
    // Defensively force a NUL terminator at the final byte. The driver is
    // documented to NUL-terminate, but if it ever wrote exactly LEN bytes
    // without a terminator, `CStr::from_ptr` would read past the array.
    buf[LEN - 1] = 0;
    let cstr = unsafe { CStr::from_ptr(buf.as_ptr()) };
    Ok(cstr.to_string_lossy().trim_end_matches('\0').to_string())
}

/// Stage-4 (GJ): return the device ordinal bound to the calling thread's
/// current CUDA context. Thin safe wrapper over `cuCtxGetDevice`.
///
/// The driver pins a CUDA context to a specific device at creation time, and
/// `cuCtxSetCurrent` binds a context to the calling thread. This wrapper is
/// the cheap way to discover which physical device the engine is talking to
/// without having to thread a `CUdevice` handle through every layer.
///
/// Used by `crate::exec::gpu_join::resolve_byte_cap_from_driver` so multi-GPU
/// rigs detect the right card's VRAM cap (the Stage-2 placeholder hardcoded
/// device 0, which was correct on single-GPU rigs but wrong when the engine
/// is bound to ordinal 1+).
///
/// Returns `Err` when no context is current on the calling thread (driver
/// error `CUDA_ERROR_INVALID_CONTEXT`).
pub(crate) fn current_device() -> BoltResult<i32> {
    let mut dev: CUdevice = 0;
    check(unsafe { cuCtxGetDevice(&mut dev) })?;
    Ok(dev as i32)
}

/// Owned CUDA context. Thread-pinned by the driver: `Send` but not `Sync`.
///
/// # Two backends, one type
///
/// In the default build, `CudaContext::new` calls `cuCtxCreate_v2` and
/// `Drop` calls `cuCtxDestroy_v2` — one process-owned context per
/// `Engine` instance.
///
/// Under `--features cudarc`, the cudarc backend creates a CUDA *primary*
/// context internally on first use of `cudarc_backend::device()`. Having
/// `CudaContext::new` also call `cuCtxCreate_v2` would mint a SECOND,
/// parallel context — pointers from one backend would be invalid in the
/// other, and the process-wide `mem_pool` would alias both. To prevent
/// that two-context lifetime bug, under `cudarc` we ALIAS the cudarc
/// primary context: `CudaContext::new` just forces cudarc's
/// `GLOBAL_DEVICE` cell to initialise, stores no handle of its own
/// (`raw` stays null), and `Drop` only drains the pool — cudarc owns
/// the context teardown.
///
/// This makes `CudaContext` a thin handle whose lifetime maps to *one*
/// pool drain, regardless of backend. Multiple `Engine::new()` calls
/// each get their own `CudaContext` instance, but under cudarc they
/// share the same underlying primary context.
pub struct CudaContext {
    /// Hand-rolled context handle. Only populated when `cudarc` is OFF.
    /// Under `--features cudarc` this stays null and Drop skips
    /// `cuCtxDestroy_v2` entirely.
    raw: CUcontext,
}

// A context handle may be moved between threads (the driver allows
// cuCtxSetCurrent on any thread), but concurrent use from multiple threads
// without external synchronization is undefined.
unsafe impl Send for CudaContext {}

impl CudaContext {
    /// Initialize the driver and acquire a CUDA context on `device_ordinal`.
    ///
    /// In the default build this calls `cuCtxCreate_v2` and the resulting
    /// `CudaContext` owns the lifetime of that context.
    ///
    /// Under `--features cudarc` this instead forces cudarc's
    /// process-wide primary context to initialise (via
    /// `cudarc_backend::ensure_device(ordinal)`) and the returned
    /// `CudaContext` is a *handle* — `Drop` does not destroy the
    /// underlying context (cudarc owns that). See the type-level docs
    /// for the rationale.
    #[cfg(not(feature = "cudarc"))]
    pub fn new(device_ordinal: i32) -> BoltResult<Self> {
        init()?;
        let dev = device_get(device_ordinal)?;
        let mut raw: CUcontext = std::ptr::null_mut();
        check(unsafe { cuCtxCreate_v2(&mut raw, 0, dev) })?;
        Ok(Self { raw })
    }

    /// cudarc-flavored constructor: defers context ownership to cudarc.
    #[cfg(feature = "cudarc")]
    pub fn new(device_ordinal: i32) -> BoltResult<Self> {
        // Force cudarc's primary-context init. From this point on every
        // alloc / free / memcpy in the engine routes through that single
        // context — there is no parallel hand-rolled context to leak.
        crate::cuda::cudarc_backend::ensure_device(device_ordinal)?;
        Ok(Self { raw: std::ptr::null_mut() })
    }

    /// Bind this context to the calling thread.
    ///
    /// Under `--features cudarc` this is a no-op: cudarc primary contexts
    /// are bound automatically when the device handle is used.
    pub fn set_current(&self) -> BoltResult<()> {
        #[cfg(not(feature = "cudarc"))]
        {
            check(unsafe { cuCtxSetCurrent(self.raw) })
        }
        #[cfg(feature = "cudarc")]
        {
            Ok(())
        }
    }

    /// Raw handle accessor for downstream submodules. Returns null under
    /// `--features cudarc` (cudarc owns the context; callers should use
    /// the cudarc backend's APIs instead of poking at a raw `CUcontext`).
    pub fn raw(&self) -> CUcontext {
        self.raw
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // Drain the global device-memory pool BEFORE destroying the context.
        //
        // Why: `DeviceMemPool` is a process-wide `static`, so its entries
        // outlive any single `CudaContext`. Every pooled `CUdeviceptr` is
        // valid only inside the context that allocated it; once the context
        // is gone the pointer is dangling. If we don't drain here, a later
        // `Engine::new()` (which mints a fresh context) inherits stale pool
        // entries and the next allocation hits an `ACCESS_VIOLATION` the
        // moment a kernel touches the recycled pointer.
        //
        // Draining now — while the context is still alive and current —
        // routes each pooled block through `cuMemFree_v2` cleanly. Any
        // outstanding `GpuBuffer`s drop their pointers BEFORE this runs
        // because field-drop order in `Engine` puts `_ctx` last.
        //
        // We do this via a runtime indirection rather than a direct call so
        // the cyclic crate-internal dependency (`cuda_sys` → `mem_pool` →
        // `cuda_sys`) does not show up in the build graph.
        //
        // The drain runs on BOTH backends — under `--features cudarc` the
        // pool's free path already uses `cudarc_backend::mem_free`, so the
        // pointers are freed against cudarc's still-live primary context.
        crate::cuda::mem_pool::POOL.drain();

        // Default backend only: tear down the hand-rolled context.
        // Under `--features cudarc` we never minted one (`raw` is null);
        // cudarc owns its primary context and will release it at process
        // exit via its own static `Drop`.
        #[cfg(not(feature = "cudarc"))]
        {
            if self.raw.is_null() {
                return;
            }
            let code = unsafe { cuCtxDestroy_v2(self.raw) };
            if code != CUDA_SUCCESS {
                log::warn!(
                    "craton-bolt: cuCtxDestroy_v2 failed with code {} (context leaked)",
                    code
                );
            }
        }
    }
}

/// Allocate `bytes` of device memory and return the raw pointer.
pub fn mem_alloc(bytes: usize) -> BoltResult<CUdeviceptr> {
    let mut ptr: CUdeviceptr = 0;
    check(unsafe { cuMemAlloc_v2(&mut ptr, bytes) })?;
    Ok(ptr)
}

/// Free a device allocation previously returned by `mem_alloc`.
///
/// # Safety
/// Caller must guarantee `ptr` is live, came from `mem_alloc`, is not aliased,
/// and that no in-flight kernel still references it.
pub unsafe fn mem_free(ptr: CUdeviceptr) -> BoltResult<()> {
    check(cuMemFree_v2(ptr))
}

/// Copy `count` elements of `T` from host `src` to device `dst`.
///
/// # Safety
/// `src` must be valid for reads of `count * size_of::<T>()` bytes and `dst`
/// must point to a device allocation of at least the same size.
pub unsafe fn memcpy_h2d<T>(dst: CUdeviceptr, src: *const T, count: usize) -> BoltResult<()> {
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        BoltError::Memory(format!(
            "memcpy_h2d size overflow: {} * {}",
            count,
            std::mem::size_of::<T>()
        ))
    })?;
    check(cuMemcpyHtoD_v2(dst, src as *const c_void, bytes))
}

/// Copy `count` elements of `T` from device `src` to host `dst`.
///
/// # Safety
/// `dst` must be valid for writes of `count * size_of::<T>()` bytes and `src`
/// must point to a live device allocation of at least the same size.
pub unsafe fn memcpy_d2h<T>(dst: *mut T, src: CUdeviceptr, count: usize) -> BoltResult<()> {
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        BoltError::Memory(format!(
            "memcpy_d2h size overflow: {} * {}",
            count,
            std::mem::size_of::<T>()
        ))
    })?;
    check(cuMemcpyDtoH_v2(dst as *mut c_void, src, bytes))
}

/// Synchronously device-to-device copy `count` elements of `T` from `src`
/// to `dst`. The two device allocations may belong to the same or different
/// pool buckets; the driver does not require any specific alignment relation
/// between them (beyond each being a valid device pointer).
///
/// Used by the incremental `GpuTable` cache: when `register_batch` appends
/// rows to a table, the engine allocates a fresh GpuVec sized for the new
/// total, DtoD-copies the previously-uploaded prefix into the leading rows,
/// and HtoD-uploads only the tail. The DtoD copy stays on the device — no
/// host bounce, no PCIe traffic.
///
/// # Safety
/// Both `dst` and `src` must point to live device allocations of at least
/// `count * size_of::<T>()` bytes. `dst` and `src` may NOT overlap.
pub unsafe fn memcpy_d2d<T>(dst: CUdeviceptr, src: CUdeviceptr, count: usize) -> BoltResult<()> {
    let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        BoltError::Memory(format!(
            "memcpy_d2d size overflow: {} * {}",
            count,
            std::mem::size_of::<T>()
        ))
    })?;
    if bytes == 0 {
        return Ok(());
    }
    check(cuMemcpyDtoD_v2(dst, src, bytes))
}

/// Allocate `bytes` of page-locked (pinned) host memory via `cuMemAllocHost_v2`.
///
/// # Safety
/// The returned pointer must be freed with [`mem_free_host`]; never with
/// `free`/`Box::from_raw`/etc. The driver determines validity and alignment.
pub unsafe fn mem_alloc_host(bytes: usize) -> BoltResult<*mut c_void> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    check(cuMemAllocHost_v2(&mut ptr, bytes))?;
    Ok(ptr)
}

/// `cuMemHostAlloc` flag: the allocation is *portable*, i.e. considered
/// page-locked by every CUDA context in the process, not just the one
/// current at allocation time. Mirrors the C macro `CU_MEMHOSTALLOC_PORTABLE`.
pub const CU_MEMHOSTALLOC_PORTABLE: c_uint = 0x01;
/// `cuMemHostAlloc` flag: map the allocation into the CUDA address space
/// (unified-addressing devices can then dereference it directly). Mirrors
/// `CU_MEMHOSTALLOC_DEVICEMAP`.
pub const CU_MEMHOSTALLOC_DEVICEMAP: c_uint = 0x02;
/// `cuMemHostAlloc` flag: allocate write-combined memory — faster for the
/// CPU to *write* and for the GPU to read over PCIe, but slow for the CPU to
/// read back. Mirrors `CU_MEMHOSTALLOC_WRITECOMBINED`.
pub const CU_MEMHOSTALLOC_WRITECOMBINED: c_uint = 0x04;

/// Allocate `bytes` of page-locked (pinned) host memory via `cuMemHostAlloc`,
/// passing the driver `flags` verbatim (a bitwise-OR of the
/// `CU_MEMHOSTALLOC_*` constants, or `0` for the same behavior as
/// [`mem_alloc_host`]).
///
/// # Safety
/// The returned pointer must be freed with [`mem_free_host`]; never with
/// `free`/`Box::from_raw`/etc. The driver determines validity and alignment.
pub unsafe fn mem_host_alloc(bytes: usize, flags: c_uint) -> BoltResult<*mut c_void> {
    let mut ptr: *mut c_void = std::ptr::null_mut();
    check(cuMemHostAlloc(&mut ptr, bytes, flags))?;
    Ok(ptr)
}

/// Free a pinned host allocation previously returned by [`mem_alloc_host`].
///
/// # Safety
/// Caller must guarantee `p` came from [`mem_alloc_host`], is not aliased, and
/// is not still in use by any in-flight async copy.
pub unsafe fn mem_free_host(p: *mut c_void) -> BoltResult<()> {
    check(cuMemFreeHost(p))
}

/// Synchronously set `count` bytes at device pointer `ptr` to the byte `value`.
///
/// # Safety
/// `ptr` must point to a live device allocation of at least `count` bytes.
pub unsafe fn memset_d8(ptr: CUdeviceptr, value: u8, count: usize) -> BoltResult<()> {
    check(cuMemsetD8_v2(ptr, value, count))
}

/// Query free and total device-memory bytes in the current context.
///
/// Returns `(free_bytes, total_bytes)`. Wraps `cuMemGetInfo_v2`.
///
/// Stage 4 (pool watcher): a background thread spawned by
/// [`crate::cuda::mem_pool`] polls this and triggers
/// `evict_above_high_water` when the free fraction drops below a
/// threshold, getting ahead of driver OOMs. Requires a live CUDA
/// context on the calling thread (same precondition as
/// `cuMemAlloc_v2`).
pub fn mem_get_info() -> BoltResult<(usize, usize)> {
    let mut free: usize = 0;
    let mut total: usize = 0;
    check(unsafe { cuMemGetInfo_v2(&mut free, &mut total) })?;
    Ok((free, total))
}

/// Stage 5 (M3L5): return the CUDA context currently bound to the
/// calling thread, or `Ok(None)` if no context is current.
///
/// Wraps `cuCtxGetCurrent`. Used by the pool-watcher to capture the
/// engine thread's context at spawn time and re-bind it on the
/// background watcher thread before each `cuMemGetInfo_v2` poll —
/// otherwise the watcher thread inherits no context and every poll
/// errors with `CUDA_ERROR_INVALID_CONTEXT`.
///
/// A null context is NOT an error from the driver's standpoint
/// (it just means no context is current), so we surface that as
/// `Ok(None)` and let the caller decide how to react.
pub fn ctx_get_current() -> BoltResult<Option<CUcontext>> {
    let mut ctx: CUcontext = std::ptr::null_mut();
    check(unsafe { cuCtxGetCurrent(&mut ctx) })?;
    if ctx.is_null() {
        Ok(None)
    } else {
        Ok(Some(ctx))
    }
}

/// Stage 5 (M3L5): bind `ctx` to the calling thread. Wraps
/// `cuCtxSetCurrent`. Counterpart to [`ctx_get_current`] — the
/// pool-watcher captures a context on the engine thread via
/// `ctx_get_current` then re-attaches it on its own thread with
/// this function.
///
/// # Safety
/// `ctx` must be a live `CUcontext` returned by the driver (e.g.
/// via [`ctx_get_current`] on a thread that already had one current),
/// AND it must not have been destroyed by another thread before this
/// call returns. The watcher pairs the capture and re-bind closely
/// enough that this holds in practice — the engine's `CudaContext`
/// outlives the watcher because `DeviceMemPool::Drop` requests
/// shutdown before context teardown.
pub unsafe fn ctx_set_current(ctx: CUcontext) -> BoltResult<()> {
    check(cuCtxSetCurrent(ctx))
}

#[cfg(test)]
mod init_cache_tests {
    //! Host-only tests for the `cuInit(0)` cache (Bug #2 regression).
    //!
    //! Verifies that:
    //!   * a successful init latches and short-circuits subsequent calls,
    //!   * an error result is NOT cached (the next call retries),
    //!   * a transient error followed by a success eventually latches.
    //!
    //! Tests inject a fake `cuInit` via the function-pointer indirection
    //! `init_with(cu_init)` and reset the latch with
    //! `_test_reset_init_cache()` between cases.
    //!
    //! Cargo runs unit tests inside a single binary on N threads, so
    //! every test here must serialize through `TEST_GATE` — otherwise
    //! concurrent tests would see each other's latch state and the
    //! "calls counted" assertions would be racy. The gate is local to
    //! this module and only ever held for the duration of a single
    //! test body.
    use super::*;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Process-wide gate. Each test acquires it for its full duration.
    static TEST_GATE: Mutex<()> = Mutex::new(());

    /// `cuInit` fake that always succeeds. Counts calls.
    static OK_CALLS: AtomicUsize = AtomicUsize::new(0);
    fn fake_ok() -> CUresult {
        OK_CALLS.fetch_add(1, Ordering::SeqCst);
        CUDA_SUCCESS
    }

    /// `cuInit` fake that always returns a non-success code. Counts calls.
    static ERR_CALLS: AtomicUsize = AtomicUsize::new(0);
    fn fake_err() -> CUresult {
        ERR_CALLS.fetch_add(1, Ordering::SeqCst);
        // 999 is well below CUDA_ERROR_STUB and unlikely to collide
        // with a real driver code in this test context.
        999
    }

    /// `cuInit` fake that fails the first N times then succeeds. Counts calls.
    static FLAKY_CALLS: AtomicUsize = AtomicUsize::new(0);
    fn fake_flaky_3() -> CUresult {
        let n = FLAKY_CALLS.fetch_add(1, Ordering::SeqCst);
        if n < 2 { 999 } else { CUDA_SUCCESS }
    }

    #[test]
    fn success_latches_and_short_circuits() {
        let _g = TEST_GATE.lock();
        _test_reset_init_cache();
        OK_CALLS.store(0, Ordering::SeqCst);

        assert!(init_with(fake_ok).is_ok());
        // Subsequent calls must NOT re-invoke the driver.
        assert!(init_with(fake_ok).is_ok());
        assert!(init_with(fake_ok).is_ok());
        assert_eq!(
            OK_CALLS.load(Ordering::SeqCst),
            1,
            "fake_ok called more than once after success"
        );
    }

    #[test]
    fn error_is_not_cached() {
        let _g = TEST_GATE.lock();
        _test_reset_init_cache();
        ERR_CALLS.store(0, Ordering::SeqCst);

        // The pre-bug behaviour cached `999` forever; the fix retries.
        assert!(init_with(fake_err).is_err());
        assert!(init_with(fake_err).is_err());
        assert!(init_with(fake_err).is_err());
        assert_eq!(
            ERR_CALLS.load(Ordering::SeqCst),
            3,
            "error result was cached instead of retried"
        );
    }

    #[test]
    fn transient_error_then_success_eventually_latches() {
        let _g = TEST_GATE.lock();
        _test_reset_init_cache();
        FLAKY_CALLS.store(0, Ordering::SeqCst);

        // Two errors, then a success, then short-circuited.
        assert!(init_with(fake_flaky_3).is_err());
        assert!(init_with(fake_flaky_3).is_err());
        assert!(init_with(fake_flaky_3).is_ok());
        assert!(init_with(fake_flaky_3).is_ok());
        assert!(init_with(fake_flaky_3).is_ok());
        // Once success latches we stop calling.
        assert_eq!(
            FLAKY_CALLS.load(Ordering::SeqCst),
            3,
            "driver was called after success latched"
        );
    }

    /// Stage 5 (M3L5): `check()` must return [`BoltError::CudaWithCode`]
    /// (NOT the legacy `Cuda(String)` shape) for every non-success
    /// driver code. `check()` invokes the real `cuGetErrorString` FFI
    /// to populate the message, but the typed-variant shape is decided
    /// before that call returns. We only need to confirm `check(2)`
    /// (the OOM code that `mem_pool::is_oom_error` depends on) takes
    /// the `CudaWithCode` branch.
    ///
    /// `check(CUDA_SUCCESS)` short-circuits without an FFI call, and
    /// `check(CUDA_ERROR_STUB)` short-circuits to `BoltError::Other`;
    /// both branches are exercised here too without touching the
    /// driver.
    #[test]
    fn check_special_codes_take_documented_branches() {
        // Success short-circuits.
        assert!(check(CUDA_SUCCESS).is_ok());

        // Stub sentinel maps to `BoltError::Other`, NOT CudaWithCode —
        // documented in `check()` and depended on by the docs.rs path.
        let stub = check(CUDA_ERROR_STUB).expect_err("stub must be Err");
        assert!(
            matches!(stub, BoltError::Other(_)),
            "CUDA_ERROR_STUB must surface as Other(_), got: {stub:?}"
        );
    }

    /// Stage 5 (M3L5): explicit codepath-level assertion that for any
    /// non-success, non-stub code, `check()` returns
    /// `BoltError::CudaWithCode { code, .. }` carrying the SAME integer
    /// passed in. We avoid invoking the live FFI by going through
    /// `check()` only when a real driver is present — under
    /// `--features cuda-stub` every code becomes `CUDA_ERROR_STUB`
    /// and the test would no-op. Without `cuda-stub` the unit test
    /// still works because `cuGetErrorString` either succeeds (real
    /// driver) or returns non-success (and we use the
    /// `"unknown CUDA error N"` fallback) — either way the returned
    /// variant is `CudaWithCode`.
    ///
    /// This is the property that `mem_pool::is_oom_error` relies on —
    /// if `check(2)` ever degraded back to `Cuda(String)` the
    /// OOM-recovery hook would silently break.
    #[cfg(not(feature = "cuda-stub"))]
    #[test]
    fn check_returns_cuda_with_code_for_nonzero_codes() {
        // CUDA_ERROR_OUT_OF_MEMORY is the canonical case.
        let err = check(2).expect_err("non-zero code must produce Err");
        assert!(
            matches!(err, BoltError::CudaWithCode { code: 2, .. }),
            "check(2) must produce CudaWithCode {{ code: 2, .. }}, got: {err:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Stage 1 async memcpy / memset wrappers.
//
// These are the safe-ish parents of `cuMemcpyHtoDAsync_v2`,
// `cuMemcpyDtoHAsync_v2`, and `cuMemsetD8Async`. They issue the operation
// on the supplied stream and return *immediately* — synchronization is the
// caller's responsibility (typically a `CudaStream::synchronize` later in
// the same phase).
//
// Stage 1 ships wrappers only: executors still use the existing synchronous
// `from_slice` / `to_vec` paths and are unaffected. Stage 2 will wire these
// into the per-shape executors together with pinned-host buffer plumbing.
//
// Backend split:
//   - Default build: routes straight into the hand-rolled `extern "C"`
//     async FFI.
//   - `--features cudarc`: routes into `cudarc_backend::memcpy_{h2d,d2h}_async`
//     and `cudarc_backend::memset_d8_async`, which invoke the
//     `cuMemcpyHtoDAsync_v2` / `cuMemcpyDtoHAsync_v2` / `cuMemsetD8Async`
//     entry points via `cudarc::driver::sys::lib()`. Both backends preserve
//     stream-ordered semantics — Stage 1's synchronous fall-back has been
//     retired (review C3).
//
// All three wrappers are `pub(crate)` rather than `pub`: Stage 1 deliberately
// keeps the async surface internal until executors are wired in Stage 2.

/// Asynchronously copy `n` elements of `T` from host `src` to device `dst`
/// on `stream`. Returns once the copy is *issued* — call
/// `cuStreamSynchronize` (or use
/// [`crate::exec::launch::CudaStream::synchronize`]) before reading the
/// destination.
///
/// For correct overlap with kernel work, `src` should point at pinned host
/// memory ([`crate::cuda::buffer::PinnedHostBuffer`]). Pageable host memory
/// still works but the driver synthesizes a staging copy that serializes
/// against the calling thread.
///
/// # Safety
/// - `src` must be valid for reads of `n * size_of::<T>()` bytes for the
///   entire duration of the async copy (until the stream is synchronized).
/// - `dst` must point to a live device allocation of at least the same
///   size in the currently-bound context.
/// - The caller must not mutate or free `src` until the copy completes.
pub(crate) unsafe fn memcpy_h2d_async<T: Pod>(
    dst: CUdeviceptr,
    src: *const T,
    n: usize,
    stream: CUstream,
) -> BoltResult<()> {
    let bytes = n.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        BoltError::Memory(format!(
            "memcpy_h2d_async size overflow: {} * {}",
            n,
            std::mem::size_of::<T>()
        ))
    })?;
    if bytes == 0 {
        return Ok(());
    }

    #[cfg(feature = "cudarc")]
    {
        // Stage 2 (review C3): real async via cudarc's driver::sys raw
        // bindings. The cudarc backend retains stream-ordered semantics.
        return crate::cuda::cudarc_backend::memcpy_h2d_async::<T>(dst, src, n, stream);
    }

    #[cfg(not(feature = "cudarc"))]
    {
        check(cuMemcpyHtoDAsync_v2(
            dst,
            src as *const c_void,
            bytes,
            stream,
        ))
    }
}

/// Asynchronously copy `n` elements of `T` from device `src` to host `dst`
/// on `stream`. Returns once the copy is *issued* — call
/// `cuStreamSynchronize` (or use
/// [`crate::exec::launch::CudaStream::synchronize`]) before reading the
/// destination.
///
/// For correct overlap with kernel work, `dst` should point at pinned host
/// memory ([`crate::cuda::buffer::PinnedHostBuffer`]).
///
/// # Safety
/// - `dst` must be valid for writes of `n * size_of::<T>()` bytes for the
///   entire duration of the async copy (until the stream is synchronized).
/// - `src` must point to a live device allocation of at least the same size
///   in the currently-bound context.
/// - The caller must not read `dst` until the copy completes.
pub(crate) unsafe fn memcpy_d2h_async<T: Pod>(
    dst: *mut T,
    src: CUdeviceptr,
    n: usize,
    stream: CUstream,
) -> BoltResult<()> {
    let bytes = n.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
        BoltError::Memory(format!(
            "memcpy_d2h_async size overflow: {} * {}",
            n,
            std::mem::size_of::<T>()
        ))
    })?;
    if bytes == 0 {
        return Ok(());
    }

    #[cfg(feature = "cudarc")]
    {
        // Stage 2 (review C3): real async via cudarc's driver::sys raw
        // bindings. See `cudarc_backend::memcpy_d2h_async`.
        return crate::cuda::cudarc_backend::memcpy_d2h_async::<T>(dst, src, n, stream);
    }

    #[cfg(not(feature = "cudarc"))]
    {
        check(cuMemcpyDtoHAsync_v2(
            dst as *mut c_void,
            src,
            bytes,
            stream,
        ))
    }
}

/// Asynchronously fill `n_bytes` at device pointer `ptr` with the byte
/// `value`, on `stream`.
///
/// Unlike the typed memcpy wrappers this one does not need a `T` because
/// `cuMemsetD8Async` operates on raw bytes.
///
/// # Safety contract
/// `ptr` must point to a live device allocation of at least `n_bytes`
/// bytes in the currently-bound context; the memory must not be freed or
/// concurrently mutated until the stream is synchronized.
///
/// This wrapper itself does not deref the pointer, so it does not need to
/// be `unsafe fn` — the unsafety is moved into the caller's obligation to
/// uphold the contract above.
#[allow(dead_code)] // reason: Stage 2 will wire executors to async memset
pub(crate) fn memset_d8_async(
    ptr: CUdeviceptr,
    value: u8,
    n_bytes: usize,
    stream: CUstream,
) -> BoltResult<()> {
    if n_bytes == 0 {
        return Ok(());
    }

    #[cfg(feature = "cudarc")]
    {
        // Stage 2 (review C3): real async via cudarc's driver::sys raw
        // bindings. See `cudarc_backend::memset_d8_async`.
        // SAFETY: precondition documented on the function — caller
        // guarantees the device range is live and not in use.
        return unsafe {
            crate::cuda::cudarc_backend::memset_d8_async(ptr, value, n_bytes, stream)
        };
    }

    #[cfg(not(feature = "cudarc"))]
    {
        // SAFETY: precondition documented on the function.
        unsafe { check(cuMemsetD8Async(ptr, value, n_bytes, stream)) }
    }
}

// ---------------------------------------------------------------------------
// Batch 6: CUDA Graph wrappers.
//
// Safe-ish parents of the `cuStreamBeginCapture_v2` / `cuStreamEndCapture` /
// `cuGraphInstantiate_v2` / `cuGraphLaunch` / `cuGraph*Destroy` FFI calls.
//
// The bitonic-sort capture in `crate::exec::gpu_sort` is the only consumer
// today; these wrappers stay `pub(crate)` because the lifetimes / ownership
// of the returned handles are subtle (a `CUgraphExec` bakes in every
// kernel-arg pointer at instantiation time — see `gpu_sort.rs` for the
// cache-key discussion).
//
// Every wrapper short-circuits to `BoltError::Other("cuda-stub …")` under
// `--features cuda-stub` because the underlying FFI shim returns
// `CUDA_ERROR_STUB`. Callers that want to opt out of the graph path on
// stub builds should check the env var BEFORE calling these wrappers
// (see `gpu_sort::sort_uses_graph()` for the gate).
// ---------------------------------------------------------------------------

/// Start stream-capture on `stream`. Every subsequent kernel launch /
/// async memcpy on this stream is *recorded* into a graph instead of
/// executed; capture stops with [`stream_end_capture`].
///
/// `mode` should be [`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`] (=2) unless the
/// caller knows it wants the laxer GLOBAL/RELAXED semantics; thread-local
/// is the safer default because it scopes the driver's "did anything race
/// with capture?" detection to the calling thread (other engine threads
/// can keep issuing CUDA work without tripping the capture).
///
/// # Safety contract (callers)
/// - `stream` must NOT be the NULL stream — stream capture rejects it
///   (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). The caller must mint a
///   real `CudaStream` first.
/// - The caller must pair this with exactly one [`stream_end_capture`]
///   call on the same stream. Leaving a stream in capture state leaks
///   any operations submitted on it.
pub(crate) fn stream_begin_capture(stream: CUstream, mode: u32) -> BoltResult<()> {
    check(unsafe { cuStreamBeginCapture_v2(stream, mode as c_uint) })
}

/// End stream-capture and return the recorded `CUgraph`. Pairs with
/// [`stream_begin_capture`].
///
/// The returned handle is owned by the caller; release it with
/// [`graph_destroy`] when the graph is no longer needed. Typically the
/// caller `cuGraphInstantiate`s it once, then destroys the `CUgraph`
/// immediately (the instantiated `CUgraphExec` holds its own copy).
pub(crate) fn stream_end_capture(stream: CUstream) -> BoltResult<CUgraph> {
    let mut g: CUgraph = std::ptr::null_mut();
    check(unsafe { cuStreamEndCapture(stream, &mut g) })?;
    Ok(g)
}

/// Instantiate `graph` into an executable form. The returned
/// `CUgraphExec` is what gets re-launched on every subsequent sort of
/// matching shape.
///
/// The `error_node` and log-buffer outputs are not surfaced — passing
/// nulls / zeros is fine for the bitonic-sort use case because we know
/// the captured graph has no host-side data dependencies and every node
/// is a pure kernel launch. If a future caller needs the diagnostic
/// info, this signature can grow accessors without breaking
/// `gpu_sort.rs`.
pub(crate) fn graph_instantiate(graph: CUgraph) -> BoltResult<CUgraphExec> {
    let mut exec: CUgraphExec = std::ptr::null_mut();
    check(unsafe {
        cuGraphInstantiate_v2(
            &mut exec,
            graph,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        )
    })?;
    Ok(exec)
}

/// Launch a previously-instantiated `CUgraphExec` on `stream`. Returns
/// immediately; the caller is responsible for the `cuStreamSynchronize`
/// (or equivalent) before reading device outputs.
///
/// # Safety contract
/// `graph_exec` must have been returned by [`graph_instantiate`] in the
/// current process AND every kernel argument captured at instantiation
/// time must still point at a live device allocation. The bitonic-sort
/// path enforces this by keying its graph cache on the input device
/// pointer — see `gpu_sort::GRAPH_CACHE`.
pub(crate) fn graph_launch(graph_exec: CUgraphExec, stream: CUstream) -> BoltResult<()> {
    check(unsafe { cuGraphLaunch(graph_exec, stream) })
}

/// Destroy a `CUgraphExec` returned by [`graph_instantiate`].
///
/// # Safety
/// `graph_exec` must not be in flight on any stream. The bitonic-sort
/// cache deliberately *leaks* its entries at process exit rather than
/// running this in `Drop` — destroying graph execs during teardown
/// races against the context-destroy path on some drivers.
#[allow(dead_code)] // reason: kept for symmetry; the cache leaks intentionally
pub(crate) unsafe fn graph_exec_destroy(graph_exec: CUgraphExec) -> BoltResult<()> {
    check(cuGraphExecDestroy(graph_exec))
}

/// Destroy a `CUgraph` returned by [`stream_end_capture`]. Safe to call
/// immediately after instantiation: the resulting `CUgraphExec` keeps its
/// own copy of the recorded operations, so the source graph can be
/// released right away.
pub(crate) fn graph_destroy(graph: CUgraph) -> BoltResult<()> {
    check(unsafe { cuGraphDestroy(graph) })
}

#[cfg(test)]
mod tests {
    //! Stage 1 sanity tests for the new async wrappers and pinned host
    //! buffer. The GPU-touching test is `#[ignore]`-gated under the same
    //! `BOLT_BENCH_GPU=1 + --ignored` convention as the rest of the crate;
    //! the compile-only test runs everywhere (including `cuda-stub`).
    use super::*;

    /// Type-only check: the new async APIs exist with the documented
    /// signatures and accept a `CUstream`. Runs under every feature
    /// configuration including `cuda-stub` and `cudarc` because it never
    /// calls the functions — it only takes their address through a
    /// matching `fn` pointer.
    #[test]
    fn async_copy_apis_compile_under_cuda_stub() {
        // Bind each wrapper to a function pointer of the expected shape.
        // Any signature drift becomes a compile error here rather than
        // silently breaking Stage 2 wiring.
        let _h2d: unsafe fn(CUdeviceptr, *const u32, usize, CUstream) -> BoltResult<()> =
            memcpy_h2d_async::<u32>;
        let _d2h: unsafe fn(*mut u32, CUdeviceptr, usize, CUstream) -> BoltResult<()> =
            memcpy_d2h_async::<u32>;
        let _set: fn(CUdeviceptr, u8, usize, CUstream) -> BoltResult<()> = memset_d8_async;

        // Also check that zero-length calls are infallible without needing
        // a live CUDA context (the wrappers short-circuit before the FFI).
        // Use a NULL stream so the call is self-contained.
        let null_stream: CUstream = std::ptr::null_mut();
        unsafe {
            memcpy_h2d_async::<u32>(0, std::ptr::null(), 0, null_stream)
                .expect("zero-length h2d must be Ok");
            memcpy_d2h_async::<u32>(std::ptr::null_mut(), 0, 0, null_stream)
                .expect("zero-length d2h must be Ok");
        }
        memset_d8_async(0, 0, 0, null_stream).expect("zero-length memset must be Ok");
    }

    /// End-to-end round-trip: pinned host -> device (async) -> pinned host
    /// (async), with a stream sync in between. Verifies that the async
    /// memcpy wrappers actually move bytes and that `PinnedHostBuffer`
    /// hands out a valid host buffer.
    #[test]
    #[ignore = "gpu:mempool — set BOLT_BENCH_GPU=1 + run with --ignored"]
    fn pinned_host_buffer_roundtrip() {
        use crate::cuda::buffer::{GpuBuffer, PinnedHostBuffer};
        use crate::exec::launch::CudaStream;

        // Bring up a context on device 0. `CudaContext::new` calls
        // `cuInit(0)` for us, so this test does not depend on test
        // ordering.
        let ctx = CudaContext::new(0).expect("create CUDA context");
        ctx.set_current().expect("set context current");

        let n = 4096usize;
        let mut host_in: PinnedHostBuffer<u32> =
            PinnedHostBuffer::new(n).expect("alloc pinned host (in)");
        let mut host_out: PinnedHostBuffer<u32> =
            PinnedHostBuffer::new(n).expect("alloc pinned host (out)");

        for (i, slot) in host_in.as_mut_slice().iter_mut().enumerate() {
            *slot = (i as u32).wrapping_mul(0x9E37_79B1);
        }
        for slot in host_out.as_mut_slice().iter_mut() {
            *slot = 0;
        }

        let stream = CudaStream::new().expect("create stream");
        let mut dev: GpuBuffer<u32> = GpuBuffer::with_capacity(n).expect("alloc device");

        dev.copy_from_async(host_in.as_slice(), stream.raw())
            .expect("async H2D");
        // Synchronize once so the kernel-less round-trip is self-contained
        // and we can issue the D2H against a known-good device buffer.
        stream.synchronize().expect("sync after H2D");

        dev.copy_to_async(host_out.as_mut_slice(), stream.raw())
            .expect("async D2H");
        stream.synchronize().expect("sync after D2H");

        assert_eq!(host_in.as_slice(), host_out.as_slice());
    }
}
