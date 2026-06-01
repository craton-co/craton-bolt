// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for a **single-pass 4-bit-radix sort** kernel.
//!
//! ## Why a radix sort?
//!
//! The existing bitonic kernel ([`crate::jit::sort_kernel`]) is excellent for
//! small, multi-key, NULL-aware ORDER BY — but its work complexity is
//! `O(n log²n)`. For the v0.6 / M3 stretch goal of replacing the host
//! round-trip ORDER BY on very large single-key sorts, an `O(n · k/r)` radix
//! sort (where `k` is the key bit-width and `r` the radix bit-width) wins.
//!
//! This kernel is **gated behind the `BOLT_GPU_SORT=1` environment variable**
//! — see [`try_gpu_radix_sort`] for the hook point.
//!
//! ## Two ABI flavours: keys-only vs keys+indices
//!
//! The codegen surface emits **two** scatter variants per dtype:
//!
//! - **Keys-only** ([`compile_radix_scatter`]) — retained for the ORDER BY
//!   single-key shortcut where the executor never needs to project unrelated
//!   columns. The sorted key buffer *is* the result.
//! - **Keys + indices** ([`compile_radix_scatter_with_indices`]) — the
//!   standard path for multi-column ORDER BY. The kernel carries a parallel
//!   `u32` row-index payload through every scatter step, so the final
//!   `vals_out` buffer is the row permutation. The executor then feeds that
//!   permutation to `arrow::compute::take` to materialise every projected
//!   column in the sorted order.
//!
//! Both variants share the same histogram kernel — the histogram only counts
//! digits in the keys, so it doesn't need the index payload.
//!
//! ## Algorithm — standard histogram / scan / scatter
//!
//! For each 4-bit radix step (LSB to MSB, 8 steps for u32/i32, 16 for i64):
//!
//! 1. **Histogram.** Every thread reads its key, extracts the current 4-bit
//!    digit (`(key >> shift) & 0xF`), and `atomicAdd`s 1 to the digit's bucket
//!    in a global histogram of 16 `u32` counters.
//! 2. **Scan.** The host runs an exclusive prefix-scan over the 16-bucket
//!    histogram, producing the per-digit starting offsets in the output array.
//!    (We reuse the existing [`crate::jit::prefix_scan`] machinery for this.)
//! 3. **Scatter.** Every thread reads its key + digit, `atomicAdd`s 1 to the
//!    digit's running counter (initialised from the scan), and writes the key
//!    into `out[offset]`. After scatter, swap input/output buffers and move to
//!    the next radix step.
//!
//! Signed types (`i32`/`i64`) need a one-time MSB flip on entry and again on
//! exit so the bit-pattern compare matches the value-order compare — same
//! standard "flip top bit" trick used in Thrust's radix sort. Floats need the
//! full IEEE-monotonic transform (flip all bits if sign==1, else flip just the
//! sign bit). The float transform is **deferred**; this kernel rejects
//! `Float32`/`Float64` and the host falls back to the bitonic sort or the host
//! sort. Bool / Utf8 are likewise rejected — bools have only 2 distinct keys
//! (cheaper to count), and Utf8 needs dictionary-decoding before it reaches
//! any device-side sort.
//!
//! ## What this PTX module emits
//!
//! This file scaffolds **the codegen surface** for the radix sort. The full
//! "histogram / scan / scatter" loop is multi-kernel and multi-launch — this
//! module emits the per-step PTX that the executor will drive. Per dtype:
//!
//! - `bolt_radix_histogram_<dty>` — read each key, bump its 4-bit digit bucket
//!   in a global 16-counter histogram. Used at the start of every radix step.
//!   Keys-only and keys+indices variants share this kernel.
//! - `bolt_radix_scatter_<dty>` — keys-only scatter: read each key, look up
//!   its digit's running offset (atomic-bumped), and write the key into the
//!   output buffer. Kept for the ORDER BY single-key shortcut.
//! - `bolt_radix_scatter_<dty>_with_indices` — keys+indices scatter: read
//!   each `(key, val)` pair from `(keys_in, vals_in)`, atomic-bump the
//!   digit's offset to claim a single output slot, then write both `key` and
//!   `val` at that slot in `(keys_out, vals_out)`. `val` is a `u32`
//!   row-index; this is the standard path for multi-column ORDER BY.
//! - `bolt_radix_msb_flip_<dty>` — signed-key fixup: XOR every key with the
//!   MSB constant. Run once on the input before pass 0 and once on the final
//!   output after the last pass. See [`compile_radix_msb_flip`] for the
//!   rationale.
//!
//! The host-side scan over 16 buckets is trivial enough to run on the CPU or
//! to reuse the engine's existing prefix-scan kernel — we don't emit a
//! dedicated scan kernel here.
//!
//! ## ABI
//!
//! ```text
//! .visible .entry bolt_radix_histogram_<dty>(
//!     .param .u64 keys_ptr,       // input keys buffer
//!     .param .u64 hist_ptr,       // 16-entry u32 histogram (atomic-added)
//!     .param .u32 n_rows,         // number of valid keys
//!     .param .u32 shift           // bit-shift for the current radix step
//! )
//!
//! .visible .entry bolt_radix_scatter_<dty>(
//!     .param .u64 keys_in_ptr,    // input keys
//!     .param .u64 keys_out_ptr,   // output keys (sorted by current radix step)
//!     .param .u64 offsets_ptr,    // 16-entry u32 running offsets (atomic-bumped)
//!     .param .u32 n_rows,
//!     .param .u32 shift
//! )
//!
//! .visible .entry bolt_radix_scatter_<dty>_with_indices(
//!     .param .u64 keys_in_ptr,    // input keys
//!     .param .u64 keys_out_ptr,   // output keys (sorted by current radix step)
//!     .param .u64 vals_in_ptr,    // input row-index payload (u32 per row)
//!     .param .u64 vals_out_ptr,   // output row-index payload (lock-step with keys)
//!     .param .u64 offsets_ptr,    // 16-entry u32 running offsets (atomic-bumped)
//!     .param .u32 n_rows,
//!     .param .u32 shift
//! )
//!
//! .visible .entry bolt_radix_msb_flip_<dty>(
//!     .param .u64 keys_ptr,       // in-place XOR with the MSB constant
//!     .param .u32 n_rows
//! )
//! ```
//!
//! Grid: 1D, `n_rows` threads total, block size 256 (matches the rest of the
//! engine's per-row kernels).

use std::fmt::Write;
use std::sync::atomic::{AtomicI8, Ordering};

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::DataType;

/// PTX target metadata baked into every emitted module. Matches the rest of
/// the JIT pipeline (see `sort_kernel.rs`, `scan_kernel.rs`).
const PTX_VERSION: &str = ".version 7.5";
/// Target SM architecture string.
const PTX_TARGET: &str = ".target sm_70";
/// Address size directive (we always use 64-bit pointers).
const PTX_ADDRESS_SIZE: &str = ".address_size 64";

/// Threads per block for the radix-sort launches. Matches `BLOCK_SIZE`
/// elsewhere so occupancy tuning stays uniform across the engine's kernels.
pub const RADIX_BLOCK_SIZE: u32 = 256;

/// Radix step width in bits. 4 bits = 16 buckets per step. 8 steps cover a
/// 32-bit key, 16 steps cover a 64-bit key. The standard tradeoff: wider
/// radix means fewer passes but bigger histograms; 4 is a good sm_70 default.
pub const RADIX_BITS: u32 = 4;

/// Number of buckets per radix step (`1 << RADIX_BITS`).
pub const RADIX_BUCKETS: u32 = 1 << RADIX_BITS;

/// Environment variable that gates the GPU radix-sort path. When set to `1`
/// the executor *may* route ORDER BY through the radix kernel for supported
/// dtypes; when unset (the default) the existing host / bitonic path runs.
pub const BOLT_GPU_SORT_ENV: &str = "BOLT_GPU_SORT";

/// Per-dtype PTX details for radix-sort key handling.
///
/// `byte_width` and `ld_st_suffix` mirror the bitonic kernel's `DtypeFlavour`
/// (the file-private struct in `sort_kernel.rs`). `radix_steps` is what
/// the radix sort cares about: how many 4-bit passes are needed to cover the
/// key. `signed_msb_flip` selects whether we need the "flip top bit so
/// signed-compare and unsigned-compare agree" trick before the first pass.
#[derive(Debug, Clone, Copy)]
struct RadixFlavour {
    /// Element byte width.
    byte_width: u32,
    /// Type suffix for `ld.global.<sfx>` / `st.global.<sfx>` (e.g. `"b32"`).
    /// We use the unsigned bit-pattern suffix because the histogram step
    /// treats keys as bit-blobs after the optional MSB flip.
    ld_st_suffix: &'static str,
    /// Number of 4-bit radix passes needed to cover this key width.
    /// `(byte_width * 8) / RADIX_BITS`.
    radix_steps: u32,
    /// Whether the key needs a one-shot MSB flip on entry (to make signed
    /// values bit-pattern-comparable with unsigned). `true` for `Int32`/`Int64`,
    /// `false` for `u32`-flavoured `Int32` views (not yet exposed) — for now
    /// we conservatively flip every integer key on entry.
    signed_msb_flip: bool,
}

impl RadixFlavour {
    /// Pick the radix-flavour table for a supported dtype, or reject the dtype.
    ///
    /// `Float32`/`Float64` are deliberately rejected: a correct float radix
    /// sort needs the IEEE-monotonic transform (flip all bits if the sign bit
    /// is set, else flip just the sign bit). The transform is straightforward
    /// to add — see e.g. Thrust's `radix_sort.inl` — but deferred to v0.7 to
    /// keep the v0.6 scaffold focused.
    ///
    /// `Bool` is rejected: 2 distinct keys means a single-pass counting
    /// sort is strictly cheaper.
    ///
    /// `Utf8` is rejected: variable-width keys don't fit the fixed-radix model
    /// without dictionary-decoding first, which is the bitonic kernel's job.
    fn for_dtype(dtype: DataType) -> BoltResult<Self> {
        Ok(match dtype {
            DataType::Int32 => Self {
                byte_width: 4,
                ld_st_suffix: "b32",
                radix_steps: 8,
                signed_msb_flip: true,
            },
            DataType::Int64 => Self {
                byte_width: 8,
                ld_st_suffix: "b64",
                radix_steps: 16,
                signed_msb_flip: true,
            },
            // Float radix needs the IEEE-monotonic transform: if the sign bit
            // is set, flip every bit; else flip just the sign bit. This makes
            // the bit-pattern unsigned compare agree with the floating-point
            // value compare for normals + zeros (NaNs sort to the end either
            // way). Deferred to v0.7 — `try_gpu_radix_sort` falls back to the
            // host path when it sees a float dtype.
            DataType::Float32 | DataType::Float64 => {
                return Err(BoltError::Other(format!(
                    "sort_kernel_radix: dtype {:?} requires the IEEE-monotonic \
                     bit transform which is deferred to v0.7; \
                     fall back to host or bitonic sort",
                    dtype
                )))
            }
            DataType::Bool => {
                return Err(BoltError::Other(
                    "sort_kernel_radix: Bool keys have only 2 distinct values; \
                     a single-pass counting sort is strictly cheaper. \
                     Fall back to host or bitonic sort.".into(),
                ))
            }
            DataType::Utf8 => {
                return Err(BoltError::Other(
                    "sort_kernel_radix: Utf8 keys must be dictionary-decoded \
                     into a fixed-width index before any device-side sort. \
                     Fall back to host or bitonic sort.".into(),
                ))
            }
            DataType::Decimal128(_, _) => {
                return Err(BoltError::Other(
                    "sort_kernel_radix: Decimal128 not yet supported".into(),
                ))
            }
            DataType::Date32 | DataType::Timestamp(_, _) => {
                return Err(BoltError::Other(
                    "sort_kernel_radix: Date/Timestamp not yet supported".into(),
                ))
            }
        })
    }
}

/// Public: is this dtype handled by the radix kernel?
///
/// The executor calls this before consulting [`BOLT_GPU_SORT_ENV`]; if the
/// dtype isn't supported, we never need to touch the env var at all.
pub fn radix_supports_dtype(dtype: DataType) -> bool {
    RadixFlavour::for_dtype(dtype).is_ok()
}

/// Public: how many 4-bit radix passes does `dtype` need?
///
/// Errors if the dtype isn't supported (same set as [`radix_supports_dtype`]).
pub fn radix_steps_for(dtype: DataType) -> BoltResult<u32> {
    Ok(RadixFlavour::for_dtype(dtype)?.radix_steps)
}

/// Cached dispatch state for the radix-sort gate. Tri-state so we can
/// distinguish "not yet read" from the two terminal values:
///
///   - `-1` — never read; first reader latches from the env var.
///   - ` 0` — gate is OFF (env unset or not exactly `"1"` after trimming).
///   - ` 1` — gate is ON (`BOLT_GPU_SORT=1`).
///
/// We use an atomic (rather than a `OnceLock<bool>`) so the `#[cfg(test)]`
/// override hook [`set_radix_dispatch_for_tests`] can flip the value without
/// having to touch process-global env state. The env-var read happens lazily
/// on first call to [`gpu_sort_env_enabled`]; subsequent calls are a plain
/// relaxed atomic load.
///
/// Why not `std::env::var(...)` on every call? Two reasons:
///
/// 1. Under `cargo test --lib` the test harness runs tests in parallel and
///    `std::env::set_var` / `std::env::remove_var` mutate process-global
///    state. Tests that probed the gate by toggling the env var would flake
///    against each other. Caching the value behind an atomic plus exposing
///    a test-only override hook lets each test pin a deterministic gate
///    state without racing on `std::env`.
/// 2. The env read happens on the hot dispatch path; a cached atomic load
///    is several orders of magnitude cheaper than `std::env::var` (which
///    takes a process-wide lock on most platforms).
static RADIX_DISPATCH_STATE: AtomicI8 = AtomicI8::new(-1);

/// Lazily latch the gate from the `BOLT_GPU_SORT` env var, returning the
/// terminal `0` / `1` value. Idempotent: subsequent calls see the cached
/// state via the atomic load and skip the env read.
fn read_env_into_dispatch_state() -> i8 {
    let v = std::env::var(BOLT_GPU_SORT_ENV)
        .ok()
        .map(|s| s.trim() == "1")
        .unwrap_or(false);
    let encoded: i8 = if v { 1 } else { 0 };
    // Relaxed store: the gate is advisory; we don't need an ordering edge
    // with any other memory. A racing initialiser that lands a different
    // value would only happen if the env var changed between two threads'
    // first reads, which violates the env-var contract anyway ("read once
    // at startup").
    RADIX_DISPATCH_STATE.store(encoded, Ordering::Relaxed);
    encoded
}

/// Public: is the `BOLT_GPU_SORT` env var set to a truthy value?
///
/// "Truthy" is exactly `"1"` — we deliberately don't accept `"true"` /
/// `"yes"` / `"on"` so the gate stays unambiguous and CI is easy to drive.
/// Returns `false` if the var is unset or set to anything else. Whitespace
/// is stripped before the equality check so an accidental trailing newline
/// from a shell-script export still trips the gate.
///
/// The env var is read once and cached; tests can override the cached value
/// via [`set_radix_dispatch_for_tests`] (test-only) without touching
/// process-global env state. See [`RADIX_DISPATCH_STATE`] for the
/// rationale.
pub fn gpu_sort_env_enabled() -> bool {
    match RADIX_DISPATCH_STATE.load(Ordering::Relaxed) {
        0 => false,
        1 => true,
        // -1 (or any other sentinel) → latch from env and recurse on the
        // cached value.
        _ => read_env_into_dispatch_state() == 1,
    }
}

/// Test-only: override the cached radix-sort dispatch gate without touching
/// the `BOLT_GPU_SORT` env var. Lets parallel test cases pin the gate
/// deterministically without racing on `std::env::set_var`.
///
/// `Some(true)`  → gate forced ON.
/// `Some(false)` → gate forced OFF.
/// `None`        → reset to "uninitialised"; the next call to
///                 [`gpu_sort_env_enabled`] re-reads the env var.
#[cfg(test)]
pub fn set_radix_dispatch_for_tests(state: Option<bool>) {
    let encoded: i8 = match state {
        Some(true) => 1,
        Some(false) => 0,
        None => -1,
    };
    RADIX_DISPATCH_STATE.store(encoded, Ordering::Relaxed);
}

/// Build the entry-point name for the histogram kernel of a given dtype.
///
/// Kept content-addressed (suffix == dtype tag) so the PTX module cache keys
/// cleanly. The naming mirrors `bolt_bitonic_sort_*` from `sort_kernel.rs`.
pub fn radix_histogram_entry(dtype: DataType) -> BoltResult<String> {
    let tag = dtype_tag(dtype)?;
    Ok(format!("bolt_radix_histogram_{}", tag))
}

/// Build the entry-point name for the scatter kernel of a given dtype.
pub fn radix_scatter_entry(dtype: DataType) -> BoltResult<String> {
    let tag = dtype_tag(dtype)?;
    Ok(format!("bolt_radix_scatter_{}", tag))
}

/// Build the entry-point name for the keys+indices scatter kernel of a given
/// dtype.
///
/// This is the standard path for multi-column ORDER BY: the kernel carries a
/// parallel `u32` row-index payload through every scatter step, so the final
/// `vals_out` buffer is the row permutation the executor feeds to
/// `arrow::compute::take`.
pub fn radix_scatter_with_indices_entry(dtype: DataType) -> BoltResult<String> {
    let tag = dtype_tag(dtype)?;
    Ok(format!("bolt_radix_scatter_{}_with_indices", tag))
}

/// Build the entry-point name for the MSB-flip kernel of a given dtype.
///
/// The MSB-flip is a one-shot in-place XOR over the keys buffer: it is run
/// once on the input before pass 0 and once on the final output after the
/// last pass, so the per-pass histogram / scatter kernels can treat the
/// keys as plain unsigned bit-blobs.
pub fn radix_msb_flip_entry(dtype: DataType) -> BoltResult<String> {
    let tag = dtype_tag(dtype)?;
    Ok(format!("bolt_radix_msb_flip_{}", tag))
}

/// Map a supported dtype to its short tag used in entry-point names.
fn dtype_tag(dtype: DataType) -> BoltResult<&'static str> {
    // We deliberately use `i32` / `i64` (the source-language spelling) rather
    // than the PTX `s32` / `s64` because the kernel manipulates the key as a
    // bit-blob (`b32` / `b64`) after the MSB flip — there's no signedness on
    // the wire. The tag is for the *user-facing* dtype that fed the sort.
    Ok(match dtype {
        DataType::Int32 => "i32",
        DataType::Int64 => "i64",
        _ => {
            // Validate via the flavour table — guarantees we never silently
            // accept a dtype that has no entry-name tag.
            let _ = RadixFlavour::for_dtype(dtype)?;
            unreachable!("dtype_tag must mirror RadixFlavour::for_dtype")
        }
    })
}

/// Emit the PTX for the radix-sort **histogram** kernel for `dtype`.
///
/// Per-block logic (shared-memory **privatized** — see PERF note below):
///
/// ```text
///   __shared__ u32 s_hist[16]
///   tid   = blockIdx.x * blockDim.x + threadIdx.x
///   lane  = threadIdx.x
///   if lane < 16: s_hist[lane] = 0          // zero the private histogram
///   __syncthreads()
///   if tid < n_rows:
///       key   = keys[tid]                   // .b32 or .b64
///       digit = (key >> shift) & 0xF
///       atomicAdd(&s_hist[digit], 1u32)     // SHARED atomic — fast, per-block
///   __syncthreads()
///   if lane < 16: atomicAdd(&hist[lane], s_hist[lane])  // reduce to global
/// ```
///
/// **PERF (C / histogram privatization).** The previous version bumped the
/// 16-entry histogram directly in GLOBAL memory with `atom.global.add`. Under
/// skewed digit distributions (e.g. a column where most keys share a digit)
/// every thread serializes on the *same* global counter, which is the dominant
/// cost of the pass. Privatizing the histogram in shared memory turns the hot
/// per-element bump into a `atom.shared.add` (an order of magnitude cheaper and
/// contended only within a block) and collapses the global traffic to exactly
/// 16 `atom.global.add`s per block in the reduction step. The emitted global
/// histogram is bit-identical to the old kernel's (sum over all blocks), so the
/// host-side exclusive-scan that follows is unchanged.
///
/// **Barrier / divergence note.** Because the kernel now contains
/// `bar.sync`, out-of-range threads (`tid >= n_rows`) must NOT early-`ret`
/// before the barriers or the block would hang. We therefore guard only the
/// per-element increment with a predicate and let every thread fall through
/// both barriers and the reduction.
///
/// The MSB-flip transform is **not** done inside this kernel — the executor
/// runs the dedicated [`compile_radix_msb_flip`] kernel once before pass 0
/// and once after the last pass. Keeping the per-step kernel transform-free
/// means the scatter kernel can ride the same already-flipped buffer without
/// doing per-step work that would cancel itself.
pub fn compile_radix_histogram(dtype: DataType) -> BoltResult<String> {
    let flavour = RadixFlavour::for_dtype(dtype)?;
    let entry = radix_histogram_entry(dtype)?;

    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // -- Signature ----------------------------------------------------
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?; // keys
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?; // hist
    writeln!(p, "\t.param .u32 {entry}_param_2,").map_err(write_err)?; // n_rows
    writeln!(p, "\t.param .u32 {entry}_param_3").map_err(write_err)?; // shift
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    // -- Shared-memory private histogram (C). 16 u32 buckets per block. -----
    writeln!(p, "\t.shared .align 4 .b32 s_hist[{}];", RADIX_BUCKETS)
        .map_err(write_err)?;

    // -- Register declarations ---------------------------------------
    writeln!(p, "\t.reg .pred %p<4>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32 %r<20>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %rd<16>;").map_err(write_err)?;

    // lane = threadIdx.x ; tid = ctaid.x * ntid.x + lane
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?; // lane
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;

    // n_rows -> %r4 ; active = (tid < n_rows) in %p0. NOTE: no early `ret`
    // here — the block contains barriers below, so every thread must fall
    // through. We gate only the per-element increment on %p0.
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(p, "\tsetp.lt.u32 %p0, %r3, %r4;").map_err(write_err)?;

    // s_hist base address -> %rd5 (generic shared address of s_hist[0]).
    writeln!(p, "\tmov.u64 %rd5, s_hist;").map_err(write_err)?;

    // Zero the private histogram: lanes 0..15 each clear one bucket.
    writeln!(p, "\tsetp.lt.u32 %p1, %r2, {};", RADIX_BUCKETS).map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd6, %r2, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd7, %rd5, %rd6;").map_err(write_err)?;
    writeln!(p, "\t@%p1 st.shared.u32 [%rd7], 0;").map_err(write_err)?;
    writeln!(p, "\tbar.sync 0;").map_err(write_err)?;

    // load shift (radix step) -> %r5
    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_3];").map_err(write_err)?;

    // keys_ptr -> %rd0 (globalised)
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;

    // addr = keys_ptr + tid * byte_width
    let key_w = flavour.byte_width as i64;
    writeln!(p, "\tmul.wide.u32 %rd1, %r3, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;

    // Per-element private bump, guarded by %p0 (tid < n_rows). Inactive
    // threads skip the load+increment but still reach the barriers/reduction.
    //
    // load key, then extract digit = (key >> shift) & 0xF
    if flavour.byte_width == 4 {
        writeln!(p, "\t@%p0 ld.global.{} %r6, [%rd2];", flavour.ld_st_suffix)
            .map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r7, %r6, %r5;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r8, %r7, 15;").map_err(write_err)?;
    } else {
        // 64-bit keys: shift in b64, then narrow the digit to b32 for indexing.
        writeln!(p, "\t@%p0 ld.global.{} %rd3, [%rd2];", flavour.ld_st_suffix)
            .map_err(write_err)?;
        // `shr.u64` takes a b32 shift amount in PTX — %r5 is already b32.
        writeln!(p, "\tshr.u64 %rd4, %rd3, %r5;").map_err(write_err)?;
        writeln!(p, "\tcvt.u32.u64 %r7, %rd4;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r8, %r7, 15;").map_err(write_err)?;
    }

    // private bucket address = s_hist + digit*4 ; atom.shared.add 1.
    writeln!(p, "\tmul.wide.u32 %rd8, %r8, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd9, %rd5, %rd8;").map_err(write_err)?;
    writeln!(p, "\t@%p0 atom.shared.add.u32 %r9, [%rd9], 1;").map_err(write_err)?;
    writeln!(p, "\tbar.sync 0;").map_err(write_err)?;

    // Reduce the private histogram into the global one: lanes 0..15 each add
    // their bucket's block-local count to hist[lane]. Skip the global atomic
    // when the count is zero so empty digits cost no global traffic.
    writeln!(p, "\tld.param.u64 %rd10, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd10, %rd10;").map_err(write_err)?;
    writeln!(p, "\t@%p1 ld.shared.u32 %r10, [%rd7];").map_err(write_err)?;
    writeln!(p, "\tsetp.ne.u32 %p2, %r10, 0;").map_err(write_err)?;
    writeln!(p, "\tand.pred %p3, %p1, %p2;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd11, %rd10, %rd6;").map_err(write_err)?;
    writeln!(p, "\t@%p3 atom.global.add.u32 %r11, [%rd11], %r10;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;

    Ok(p)
}

/// Emit the PTX for the radix-sort **scatter** kernel for `dtype`.
///
/// Per-block logic (**block-stable** — see stability note):
///
/// ```text
///   __shared__ u32 s_digit[BLOCK]      // each lane's digit (0xFFFF = inactive)
///   __shared__ u32 s_hist[16]          // per-block digit counts
///   __shared__ u32 s_base[16]          // reserved global base per digit
///   tid  = blockIdx.x*blockDim.x + lane ; lane = threadIdx.x
///   active = tid < n_rows
///   digit  = active ? (keys_in[tid] >> shift) & 0xF : 0xFFFF
///   s_digit[lane] = digit
///   if lane < 16: s_hist[lane] = 0
///   __syncthreads()
///   if active: atomicAdd(&s_hist[digit], 1)       // shared, per-block count
///   __syncthreads()
///   if lane < 16 && s_hist[lane] != 0:
///       s_base[lane] = atomicAdd(&offsets[lane], s_hist[lane])  // reserve range
///   __syncthreads()
///   if active:
///       rank = #{ j < lane : s_digit[j] == digit }   // STABLE within block
///       out_idx = s_base[digit] + rank
///       keys_out[out_idx] = key
/// ```
///
/// `offsets[]` is still initialised on the host to the exclusive-scan of the
/// histogram. Each block reserves a contiguous run inside its digit's region
/// via a single `atom.global.add(offsets[digit], block_count)` (instead of one
/// global atomic *per element*), then places its own elements inside that run
/// in **input (tid) order** using a deterministic per-block rank.
///
/// **Stability — read this.** The old kernel claimed each output slot with a
/// racing `atom.global.add` per element, so two equal-digit elements landed in
/// nondeterministic relative order: every LSD pass was non-stable and the
/// multi-pass sort could be **wrong** for `ORDER BY non_unique_key`. This
/// version is **stable within a block**: the rank loop counts only lanes with a
/// strictly smaller `threadIdx.x`, so equal-digit elements in the same block
/// keep their input order deterministically.
///
/// **Residual limitation (documented, not yet fixed).** Ordering is *not* yet
/// stable **across** blocks: blocks reserve their per-digit runs in
/// `atom.global.add` arrival order, which is scheduling-dependent, so a higher-
/// `blockIdx` block can occupy an earlier run than a lower one. Full global
/// stability needs a per-block-per-digit exclusive prefix sum
/// (`block_digit_offsets[num_blocks][16]`) computed in a separate pass so each
/// block's run is ordered by `blockIdx` — that requires an extra buffer and
/// kernel launch in the executor (`gpu_sort.rs`) and is deferred. Until then
/// the radix path stays gated behind `BOLT_GPU_SORT=1` (default OFF); it must
/// reach full cross-block stability before being promoted to default-on so
/// `ORDER BY non_unique_key [LIMIT k]` matches the host `lexsort_to_indices`
/// fallback. The per-element race, however, is gone.
pub fn compile_radix_scatter(dtype: DataType) -> BoltResult<String> {
    let flavour = RadixFlavour::for_dtype(dtype)?;
    let entry = radix_scatter_entry(dtype)?;

    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // -- Signature ----------------------------------------------------
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?; // keys_in
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?; // keys_out
    writeln!(p, "\t.param .u64 {entry}_param_2,").map_err(write_err)?; // offsets
    writeln!(p, "\t.param .u32 {entry}_param_3,").map_err(write_err)?; // n_rows
    writeln!(p, "\t.param .u32 {entry}_param_4").map_err(write_err)?; // shift
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    // Block-stable prologue: computes digit (%r8), out_idx (%r9), and leaves
    // the loaded key in %r6 (b32) / %rd3 (b64). `active` predicate is %p0.
    let key_w = flavour.byte_width as i64;
    emit_block_stable_scatter_prologue(
        &mut p,
        &flavour,
        &entry,
        /* keys_in_param  */ 0,
        /* offsets_param  */ 2,
        /* n_rows_param   */ 3,
        /* shift_param    */ 4,
    )?;

    // keys_out_ptr -> %rd8; out_addr = keys_out_ptr + out_idx * byte_width.
    writeln!(p, "\tld.param.u64 %rd8, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd9, %r9, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd10, %rd8, %rd9;").map_err(write_err)?;

    if flavour.byte_width == 4 {
        writeln!(p, "\t@%p0 st.global.{} [%rd10], %r6;", flavour.ld_st_suffix)
            .map_err(write_err)?;
    } else {
        writeln!(p, "\t@%p0 st.global.{} [%rd10], %rd3;", flavour.ld_st_suffix)
            .map_err(write_err)?;
    }

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;

    Ok(p)
}

/// Emit the shared block-stable scatter prologue used by both
/// [`compile_radix_scatter`] and [`compile_radix_scatter_with_indices`].
///
/// On return, the following registers/predicates are live:
///   - `%p0`  = active (tid < n_rows)
///   - `%r3`  = tid, `%r2` = lane (threadIdx.x)
///   - `%r6`  = loaded key (b32 path) ; `%rd3` = loaded key (b64 path)
///   - `%r8`  = digit (0..15) for active threads
///   - `%r9`  = out_idx = s_base[digit] + per-block stable rank
///
/// Register budget consumed: `%r0..%r15`, `%rd0..%rd15`, `%p0..%p3`.
/// Shared symbols emitted: `s_digit`, `s_hist`, `s_base`.
///
/// Barrier-safety: the kernel contains three `bar.sync`s, so the prologue
/// never early-`ret`s on out-of-range threads — every lane falls through all
/// barriers; out-of-range lanes simply store the `0xFFFF` inactive sentinel
/// into `s_digit[lane]` and skip the increment/store via `%p0`.
#[allow(clippy::too_many_arguments)]
fn emit_block_stable_scatter_prologue(
    p: &mut String,
    flavour: &RadixFlavour,
    entry: &str,
    keys_in_param: u32,
    offsets_param: u32,
    n_rows_param: u32,
    shift_param: u32,
) -> BoltResult<()> {
    // Inactive-lane sentinel stored into s_digit (never a real 4-bit digit).
    const INACTIVE: u32 = 0xFFFF;

    // -- Shared state -------------------------------------------------
    writeln!(p, "\t.shared .align 4 .b32 s_digit[{}];", RADIX_BLOCK_SIZE)
        .map_err(write_err)?;
    writeln!(p, "\t.shared .align 4 .b32 s_hist[{}];", RADIX_BUCKETS).map_err(write_err)?;
    writeln!(p, "\t.shared .align 4 .b32 s_base[{}];", RADIX_BUCKETS).map_err(write_err)?;

    // -- Register declarations ---------------------------------------
    // The prologue itself uses %r0..%r15 / %rd0..%rd15 / %p0..%p3. We declare a
    // wider pool (r/rd up to 23) so callers can stage their own payload
    // (e.g. the with-indices scatter's u32 row-index) in %r16.. / %rd16..
    // without emitting a second (illegal) `.reg` directive in the same body.
    writeln!(p, "\t.reg .pred %p<4>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32 %r<24>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %rd<24>;").map_err(write_err)?;

    // lane = threadIdx.x ; tid = ctaid.x * ntid.x + lane
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;

    // active = tid < n_rows -> %p0 (no early ret; barriers below).
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_{n_rows_param}];").map_err(write_err)?;
    writeln!(p, "\tsetp.lt.u32 %p0, %r3, %r4;").map_err(write_err)?;

    // load shift -> %r5
    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_{shift_param}];").map_err(write_err)?;

    // keys_in_ptr -> %rd0; load key (guarded) and compute digit.
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_{keys_in_param}];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    let key_w = flavour.byte_width as i64;
    writeln!(p, "\tmul.wide.u32 %rd1, %r3, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;

    // Default digit to INACTIVE; active lanes overwrite with the real digit.
    writeln!(p, "\tmov.u32 %r8, {INACTIVE};").map_err(write_err)?;
    if flavour.byte_width == 4 {
        writeln!(p, "\t@%p0 ld.global.{} %r6, [%rd2];", flavour.ld_st_suffix)
            .map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r7, %r6, %r5;").map_err(write_err)?;
        writeln!(p, "\t@%p0 and.b32 %r8, %r7, 15;").map_err(write_err)?;
    } else {
        writeln!(p, "\t@%p0 ld.global.{} %rd3, [%rd2];", flavour.ld_st_suffix)
            .map_err(write_err)?;
        writeln!(p, "\tshr.u64 %rd4, %rd3, %r5;").map_err(write_err)?;
        writeln!(p, "\tcvt.u32.u64 %r7, %rd4;").map_err(write_err)?;
        writeln!(p, "\t@%p0 and.b32 %r8, %r7, 15;").map_err(write_err)?;
    }

    // s_digit / s_hist / s_base base addresses.
    writeln!(p, "\tmov.u64 %rd5, s_digit;").map_err(write_err)?;
    writeln!(p, "\tmov.u64 %rd6, s_hist;").map_err(write_err)?;
    writeln!(p, "\tmov.u64 %rd7, s_base;").map_err(write_err)?;

    // s_digit[lane] = digit (every lane writes; inactive lanes write INACTIVE).
    writeln!(p, "\tmul.wide.u32 %rd11, %r2, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd12, %rd5, %rd11;").map_err(write_err)?;
    writeln!(p, "\tst.shared.u32 [%rd12], %r8;").map_err(write_err)?;

    // Zero s_hist with lanes 0..15.
    writeln!(p, "\tsetp.lt.u32 %p1, %r2, {};", RADIX_BUCKETS).map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd13, %rd6, %rd11;").map_err(write_err)?; // &s_hist[lane]
    writeln!(p, "\t@%p1 st.shared.u32 [%rd13], 0;").map_err(write_err)?;
    writeln!(p, "\tbar.sync 0;").map_err(write_err)?;

    // Per-block digit count: active lanes bump s_hist[digit].
    writeln!(p, "\tmul.wide.u32 %rd14, %r8, 4;").map_err(write_err)?; // digit*4
    writeln!(p, "\tadd.s64 %rd15, %rd6, %rd14;").map_err(write_err)?; // &s_hist[digit]
    writeln!(p, "\t@%p0 atom.shared.add.u32 %r10, [%rd15], 1;").map_err(write_err)?;
    writeln!(p, "\tbar.sync 0;").map_err(write_err)?;

    // Reserve a contiguous global run per non-empty digit: lanes 0..15.
    //   s_base[lane] = atomicAdd(&offsets[lane], s_hist[lane])
    writeln!(p, "\tld.param.u64 %rd8, [{entry}_param_{offsets_param}];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(p, "\t@%p1 ld.shared.u32 %r11, [%rd13];").map_err(write_err)?; // block count
    writeln!(p, "\tsetp.ne.u32 %p2, %r11, 0;").map_err(write_err)?;
    writeln!(p, "\tand.pred %p3, %p1, %p2;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd9, %rd8, %rd11;").map_err(write_err)?; // &offsets[lane]
    // Reserve (only non-empty buckets touch global memory). %r12 = run base.
    writeln!(p, "\tmov.u32 %r12, 0;").map_err(write_err)?;
    writeln!(p, "\t@%p3 atom.global.add.u32 %r12, [%rd9], %r11;").map_err(write_err)?;
    // s_base[lane] = run base (write for all 16 lanes; empty buckets store 0,
    // and no active thread will read an empty bucket's base anyway).
    writeln!(p, "\tadd.s64 %rd10, %rd7, %rd11;").map_err(write_err)?; // &s_base[lane]
    writeln!(p, "\t@%p1 st.shared.u32 [%rd10], %r12;").map_err(write_err)?;
    writeln!(p, "\tbar.sync 0;").map_err(write_err)?;

    // Per-block STABLE rank: rank = #{ j in 0..lane : s_digit[j] == digit }.
    // Deterministic and tid-ordered → equal-digit elements keep input order
    // within the block. Loop counter %r13 = j, accumulator %r14 = rank.
    writeln!(p, "\tmov.u32 %r13, 0;").map_err(write_err)?; // j
    writeln!(p, "\tmov.u32 %r14, 0;").map_err(write_err)?; // rank
    writeln!(p, "RANK_LOOP:").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.u32 %p2, %r13, %r2;").map_err(write_err)?; // j >= lane?
    writeln!(p, "\t@%p2 bra RANK_DONE;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd11, %r13, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd12, %rd5, %rd11;").map_err(write_err)?; // &s_digit[j]
    writeln!(p, "\tld.shared.u32 %r15, [%rd12];").map_err(write_err)?;
    writeln!(p, "\tsetp.eq.u32 %p3, %r15, %r8;").map_err(write_err)?; // same digit?
    writeln!(p, "\t@%p3 add.u32 %r14, %r14, 1;").map_err(write_err)?;
    writeln!(p, "\tadd.u32 %r13, %r13, 1;").map_err(write_err)?;
    writeln!(p, "\tbra RANK_LOOP;").map_err(write_err)?;
    writeln!(p, "RANK_DONE:").map_err(write_err)?;

    // out_idx (%r9) = s_base[digit] + rank. (Meaningful only for active lanes;
    // the store at the call site is guarded by %p0 so inactive lanes — whose
    // digit is the INACTIVE sentinel and would index out of s_base — never read
    // a bad base or write. We still gate the s_base load on %p0 to avoid an
    // out-of-bounds shared read for inactive lanes.)
    writeln!(p, "\tmov.u32 %r9, 0;").map_err(write_err)?;
    writeln!(p, "\t@%p0 mul.wide.u32 %rd11, %r8, 4;").map_err(write_err)?; // digit*4
    writeln!(p, "\t@%p0 add.s64 %rd12, %rd7, %rd11;").map_err(write_err)?; // &s_base[digit]
    writeln!(p, "\t@%p0 ld.shared.u32 %r9, [%rd12];").map_err(write_err)?; // run base
    writeln!(p, "\tadd.u32 %r9, %r9, %r14;").map_err(write_err)?; // + stable rank
    Ok(())
}

/// Emit the PTX for the radix-sort **keys+indices scatter** kernel for `dtype`.
///
/// This is the standard path for multi-column ORDER BY: the kernel carries a
/// parallel `u32` row-index payload through every scatter step, so after the
/// last pass `vals_out` is the row permutation. The executor wraps that
/// permutation in a `UInt32Array` and feeds it to `arrow::compute::take` to
/// materialise every projected column in sorted order.
///
/// Per-thread logic:
///
/// ```text
///   (block-stable prologue — see emit_block_stable_scatter_prologue):
///   active = tid < n_rows
///   digit  = (keys_in[tid] >> shift) & 0xF
///   out_idx = s_base[digit] + per-block stable rank
///   // then, at the same out_idx:
///   keys_out[out_idx] = key
///   vals_out[out_idx] = vals_in[tid]      // u32 row-index, lock-step with key
/// ```
///
/// The key and its row-index payload are written to the **same** `out_idx`
/// computed once by the prologue, so the pairing is preserved.
///
/// **Stability.** This path now shares the **block-stable** scatter prologue
/// with the keys-only kernel: equal-key rows within a block keep their input
/// order via the deterministic per-block rank, eliminating the previous
/// per-element `atom.global.add` race. The same residual limitation applies —
/// ordering is not yet stable across blocks (block runs are reserved in
/// scheduling order), which requires a per-block-per-digit prefix-sum pass in
/// the executor and is deferred. The radix path stays gated behind
/// `BOLT_GPU_SORT=1` (default OFF) until that lands. See
/// [`compile_radix_scatter`] for the full discussion.
pub fn compile_radix_scatter_with_indices(dtype: DataType) -> BoltResult<String> {
    let flavour = RadixFlavour::for_dtype(dtype)?;
    let entry = radix_scatter_with_indices_entry(dtype)?;

    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // -- Signature ----------------------------------------------------
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?; // keys_in
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?; // keys_out
    writeln!(p, "\t.param .u64 {entry}_param_2,").map_err(write_err)?; // vals_in
    writeln!(p, "\t.param .u64 {entry}_param_3,").map_err(write_err)?; // vals_out
    writeln!(p, "\t.param .u64 {entry}_param_4,").map_err(write_err)?; // offsets
    writeln!(p, "\t.param .u32 {entry}_param_5,").map_err(write_err)?; // n_rows
    writeln!(p, "\t.param .u32 {entry}_param_6").map_err(write_err)?; // shift
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    // Block-stable prologue. Params: keys_in=0, offsets=4, n_rows=5, shift=6.
    // On return: %p0=active, %r3=tid, %r9=out_idx, key in %r6 (b32)/%rd3 (b64).
    // The prologue declares a register pool up to %r23 / %rd23 so we can stage
    // the row-index payload in %r16 / %rd16.. below without a second `.reg`.
    let key_w = flavour.byte_width as i64;
    emit_block_stable_scatter_prologue(
        &mut p,
        &flavour,
        &entry,
        /* keys_in_param  */ 0,
        /* offsets_param  */ 4,
        /* n_rows_param   */ 5,
        /* shift_param    */ 6,
    )?;

    // vals_in_ptr -> %rd16; load the u32 row-index payload for active lanes.
    writeln!(p, "\tld.param.u64 %rd16, [{entry}_param_2];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd16, %rd16;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd17, %r3, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd18, %rd16, %rd17;").map_err(write_err)?;
    writeln!(p, "\t@%p0 ld.global.u32 %r16, [%rd18];").map_err(write_err)?;

    // keys_out_ptr -> %rd19; out_addr = keys_out_ptr + out_idx * key_w
    writeln!(p, "\tld.param.u64 %rd19, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd19, %rd19;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd20, %r9, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd21, %rd19, %rd20;").map_err(write_err)?;

    if flavour.byte_width == 4 {
        writeln!(p, "\t@%p0 st.global.{} [%rd21], %r6;", flavour.ld_st_suffix)
            .map_err(write_err)?;
    } else {
        writeln!(p, "\t@%p0 st.global.{} [%rd21], %rd3;", flavour.ld_st_suffix)
            .map_err(write_err)?;
    }

    // vals_out_ptr -> %rd22; vout_addr = vals_out_ptr + out_idx * 4
    writeln!(p, "\tld.param.u64 %rd22, [{entry}_param_3];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd22, %rd22;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd23, %r9, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd16, %rd22, %rd23;").map_err(write_err)?; // reuse %rd16
    writeln!(p, "\t@%p0 st.global.u32 [%rd16], %r16;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;

    Ok(p)
}

/// Emit the PTX for the **MSB-flip** kernel for a signed `dtype`.
///
/// Why we need this: a signed integer's bits sort wrong as unsigned because
/// the sign bit is the most-significant bit and is inverted (negative values
/// have the sign bit set, but should sort *before* positive values that have
/// it cleared). The standard trick — used in Thrust's radix sort and CUB —
/// is to XOR the key with the MSB constant on entry, run the unsigned radix
/// sort over the transformed bits, then XOR again on exit to restore the
/// original values. After the round-trip XOR the visible output is identical
/// to the input *value*, but the intermediate per-pass histogram / scatter
/// kernels see a clean unsigned bit-pattern that sorts correctly.
///
/// The MSB constants:
/// - Int32: `0x8000_0000`
/// - Int64: `0x8000_0000_0000_0000`
///
/// Per-thread logic:
///
/// ```text
///   tid = blockIdx.x * blockDim.x + threadIdx.x
///   if tid >= n_rows: return
///   keys[tid] ^= MSB
/// ```
///
/// Run once before pass 0 over the input buffer, then once after the last
/// pass over the final output buffer. The transform is its own inverse
/// (XOR is involutive), so the same kernel does entry-flip and exit-flip.
///
/// We separate this from the per-pass kernels so the scatter kernel can ride
/// already-flipped keys without doing per-step work that would cancel itself.
/// Returns an error for dtypes that don't require the flip (today: none of
/// the supported set, since `Float32`/`Float64` need the IEEE-monotonic
/// transform instead and `Bool`/`Utf8` aren't supported at all). Callers
/// can also gate via [`radix_needs_msb_flip`].
pub fn compile_radix_msb_flip(dtype: DataType) -> BoltResult<String> {
    let flavour = RadixFlavour::for_dtype(dtype)?;
    if !flavour.signed_msb_flip {
        return Err(BoltError::Other(format!(
            "sort_kernel_radix: dtype {:?} does not require an MSB flip; \
             callers should gate via radix_needs_msb_flip first",
            dtype
        )));
    }
    let entry = radix_msb_flip_entry(dtype)?;

    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // -- Signature ----------------------------------------------------
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?; // keys
    writeln!(p, "\t.param .u32 {entry}_param_1").map_err(write_err)?; // n_rows
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    // -- Register declarations ---------------------------------------
    writeln!(p, "\t.reg .pred %p<2>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32 %r<10>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %rd<10>;").map_err(write_err)?;

    // tid
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;

    // bail if tid >= n_rows
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // keys_ptr -> %rd0; addr = keys_ptr + tid * byte_width
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    let key_w = flavour.byte_width as i64;
    writeln!(p, "\tmul.wide.u32 %rd1, %r3, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;

    // load key, XOR with the MSB constant, store back. Round-tripping
    // via the same address (involutive XOR) makes the kernel its own inverse.
    if flavour.byte_width == 4 {
        writeln!(p, "\tld.global.{} %r5, [%rd2];", flavour.ld_st_suffix).map_err(write_err)?;
        writeln!(p, "\txor.b32 %r6, %r5, 2147483648;").map_err(write_err)?; // 0x8000_0000
        writeln!(p, "\tst.global.{} [%rd2], %r6;", flavour.ld_st_suffix).map_err(write_err)?;
    } else {
        writeln!(p, "\tld.global.{} %rd3, [%rd2];", flavour.ld_st_suffix).map_err(write_err)?;
        // 0x8000_0000_0000_0000 as a literal; PTX accepts unsigned 64-bit
        // immediates for xor.b64.
        writeln!(p, "\txor.b64 %rd4, %rd3, 9223372036854775808;").map_err(write_err)?;
        writeln!(p, "\tst.global.{} [%rd2], %rd4;", flavour.ld_st_suffix).map_err(write_err)?;
    }

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;

    Ok(p)
}

/// Public: does this dtype need a one-shot MSB flip on entry and exit?
///
/// Returns `true` for signed integer dtypes (Int32, Int64) where the sign
/// bit's inverted order would otherwise break the unsigned bit-pattern
/// compare used by the per-pass histogram / scatter kernels. Returns `false`
/// for unsigned-natured dtypes once they're added. Errors if the dtype isn't
/// supported at all (same set as [`radix_supports_dtype`]).
pub fn radix_needs_msb_flip(dtype: DataType) -> BoltResult<bool> {
    Ok(RadixFlavour::for_dtype(dtype)?.signed_msb_flip)
}

/// Hook point for the executor: decide whether the GPU radix sort should run
/// for an ORDER BY with `dtype` keys.
///
/// Returns `Ok(true)` when both:
///   1. The env var `BOLT_GPU_SORT=1` is set, **and**
///   2. The dtype is supported by the radix kernel (Int32/Int64 — see
///      [`RadixFlavour::for_dtype`] for the gate; Float/Bool/Utf8 fall back
///      with a doc-comment note that float radix needs the IEEE-monotonic
///      transform — deferred).
///
/// Returns `Ok(false)` otherwise. Never errors today (the gate is purely
/// advisory) but kept `BoltResult<bool>` so future extensions can surface
/// e.g. unsupported-driver errors without breaking call sites.
///
/// **Not wired into [`crate::exec::sort`] yet.** When the executor adopts
/// this gate it will look like:
///
/// ```ignore
/// if try_gpu_radix_sort(key_dtype)? {
///     /* launch histogram / scan / scatter loop using compile_radix_* */
/// } else {
///     /* existing host or bitonic path */
/// }
/// ```
pub fn try_gpu_radix_sort(dtype: DataType) -> BoltResult<bool> {
    if !gpu_sort_env_enabled() {
        return Ok(false);
    }
    Ok(radix_supports_dtype(dtype))
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("sort_kernel_radix: write failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the kernel-name shapes. Two consumers depend on these strings:
    /// the host-side launcher (key into the PTX module cache) and the
    /// golden PTX assertions further down. Changing either tag is a
    /// breaking change to the executor wiring.
    #[test]
    fn entry_names_pin() {
        assert_eq!(
            radix_histogram_entry(DataType::Int32).unwrap(),
            "bolt_radix_histogram_i32"
        );
        assert_eq!(
            radix_histogram_entry(DataType::Int64).unwrap(),
            "bolt_radix_histogram_i64"
        );
        assert_eq!(
            radix_scatter_entry(DataType::Int32).unwrap(),
            "bolt_radix_scatter_i32"
        );
        assert_eq!(
            radix_scatter_entry(DataType::Int64).unwrap(),
            "bolt_radix_scatter_i64"
        );
    }

    /// Unsupported dtypes must reject at the entry-name layer too — not just
    /// at codegen — so the executor can branch before reaching for PTX.
    #[test]
    fn unsupported_dtypes_rejected() {
        for dty in [
            DataType::Bool,
            DataType::Float32,
            DataType::Float64,
            DataType::Utf8,
        ] {
            assert!(
                !radix_supports_dtype(dty),
                "dtype {:?} should not be supported by the radix kernel",
                dty
            );
            assert!(radix_histogram_entry(dty).is_err());
            assert!(radix_scatter_entry(dty).is_err());
            assert!(radix_scatter_with_indices_entry(dty).is_err());
            assert!(radix_msb_flip_entry(dty).is_err());
            assert!(compile_radix_histogram(dty).is_err());
            assert!(compile_radix_scatter(dty).is_err());
            assert!(compile_radix_scatter_with_indices(dty).is_err());
            assert!(compile_radix_msb_flip(dty).is_err());
            assert!(radix_steps_for(dty).is_err());
            assert!(radix_needs_msb_flip(dty).is_err());
        }
    }

    /// The new keys+indices entry names pin to the documented shape — both
    /// the executor wiring and the PTX module cache rely on this string.
    #[test]
    fn scatter_with_indices_entry_names_pin() {
        assert_eq!(
            radix_scatter_with_indices_entry(DataType::Int32).unwrap(),
            "bolt_radix_scatter_i32_with_indices"
        );
        assert_eq!(
            radix_scatter_with_indices_entry(DataType::Int64).unwrap(),
            "bolt_radix_scatter_i64_with_indices"
        );
    }

    /// MSB-flip entry names pin to the documented shape.
    #[test]
    fn msb_flip_entry_names_pin() {
        assert_eq!(
            radix_msb_flip_entry(DataType::Int32).unwrap(),
            "bolt_radix_msb_flip_i32"
        );
        assert_eq!(
            radix_msb_flip_entry(DataType::Int64).unwrap(),
            "bolt_radix_msb_flip_i64"
        );
    }

    /// Both signed integer dtypes need the MSB flip.
    #[test]
    fn signed_dtypes_need_msb_flip() {
        assert!(radix_needs_msb_flip(DataType::Int32).unwrap());
        assert!(radix_needs_msb_flip(DataType::Int64).unwrap());
    }

    /// The histogram PTX module includes the right entry name, the atomic
    /// histogram bump, and the 4-bit digit mask. Lightweight golden — we
    /// only pin the shape; the bytes can drift safely.
    #[test]
    fn histogram_ptx_shape_i32() {
        let ptx = compile_radix_histogram(DataType::Int32).unwrap();
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".visible .entry bolt_radix_histogram_i32("));
        // 4-bit digit extraction.
        assert!(ptx.contains("and.b32"));
        assert!(ptx.contains(", 15;"));
        // (C) Privatized histogram: a per-block shared histogram bumped with
        // atom.shared.add, two barriers, then a global reduction with
        // atom.global.add over the 16 buckets.
        assert!(ptx.contains(".shared .align 4 .b32 s_hist"));
        assert!(ptx.contains("atom.shared.add.u32"));
        assert_eq!(
            ptx.matches("bar.sync").count(),
            2,
            "privatized histogram needs two barriers (post-zero, post-count)",
        );
        assert!(ptx.contains("atom.global.add.u32"));
        // The b32 load suffix for an Int32 key.
        assert!(ptx.contains("ld.global.b32"));
        assert!(ptx.contains("DONE:"));
        assert!(ptx.contains("ret;"));
    }

    /// Same shape pinning for the i64 histogram — the b64 load + shift path.
    #[test]
    fn histogram_ptx_shape_i64() {
        let ptx = compile_radix_histogram(DataType::Int64).unwrap();
        assert!(ptx.contains(".visible .entry bolt_radix_histogram_i64("));
        assert!(ptx.contains("ld.global.b64"));
        assert!(ptx.contains("shr.u64"));
        assert!(ptx.contains("atom.global.add.u32"));
    }

    /// Scatter PTX shape — atomic offset bump, store of the key into
    /// `keys_out[out_idx]`.
    #[test]
    fn scatter_ptx_shape_i32() {
        let ptx = compile_radix_scatter(DataType::Int32).unwrap();
        assert!(ptx.contains(".visible .entry bolt_radix_scatter_i32("));
        // Block-stable scatter: a per-block shared histogram, three barriers,
        // a single global atomic to RESERVE the per-digit run (no longer a
        // per-element bump), and the deterministic per-block rank loop.
        assert!(ptx.contains("st.global.b32"));
        assert!(ptx.contains("ld.global.b32"));
        assert!(ptx.contains("atom.shared.add.u32"));
        assert!(ptx.contains("atom.global.add.u32"));
        assert_eq!(
            ptx.matches("bar.sync").count(),
            3,
            "block-stable scatter needs exactly three bar.sync barriers",
        );
        assert!(ptx.contains("RANK_LOOP:"));
        // Exactly one GLOBAL atomic: the per-digit run reservation.
        assert_eq!(ptx.matches("atom.global.add.u32").count(), 1);
    }

    /// Keys+indices scatter PTX shape for `i32` (block-stable):
    ///   1. The entry name and signature carry the documented seven `.param`s.
    ///   2. The key load (`ld.global.b32`) and key store (`st.global.b32`)
    ///      survive intact.
    ///   3. Exactly one GLOBAL atomic — the per-digit run reservation — so the
    ///      key and its row-index payload share the single `out_idx` the
    ///      block-stable prologue computes. The per-element race is gone.
    ///   4. The `vals_in` u32 load and `vals_out` u32 store appear — the ABI
    ///      surface the executor depends on.
    ///   5. The block-stable machinery (shared atomic, barriers, rank loop) is
    ///      present.
    #[test]
    fn scatter_with_indices_ptx_shape_i32() {
        let ptx = compile_radix_scatter_with_indices(DataType::Int32).unwrap();
        assert!(ptx.contains(".visible .entry bolt_radix_scatter_i32_with_indices("));
        // Seven params: keys_in, keys_out, vals_in, vals_out, offsets, n_rows, shift.
        for i in 0..=6 {
            assert!(
                ptx.contains(&format!("_param_{i}")),
                "missing _param_{i} in keys+indices scatter PTX",
            );
        }
        // Existing key-side ABI preserved.
        assert!(ptx.contains("ld.global.b32"));
        assert!(ptx.contains("st.global.b32"));
        // Exactly one GLOBAL atomic: the per-digit run reservation. Key and val
        // writes both use the prologue's single `out_idx`, so they stay paired.
        assert_eq!(
            ptx.matches("atom.global.add.u32").count(),
            1,
            "keys+indices scatter must reserve its per-digit run with exactly \
             one global atomicAdd so keys and vals land at the same slot",
        );
        // Block-stable machinery.
        assert!(ptx.contains("atom.shared.add.u32"));
        assert_eq!(ptx.matches("bar.sync").count(), 3);
        assert!(ptx.contains("RANK_LOOP:"));
        // The vals payload: u32 load from vals_in and u32 store to vals_out.
        assert!(
            ptx.contains("ld.global.u32"),
            "expected `ld.global.u32` for the vals_in row-index payload load",
        );
        assert!(
            ptx.contains("st.global.u32"),
            "expected `st.global.u32` for the vals_out row-index payload store",
        );
        assert!(ptx.contains("DONE:"));
        assert!(ptx.contains("ret;"));
    }

    /// Keys+indices scatter PTX shape for `i64` (block-stable):
    ///   1. Entry name carries the i64 tag.
    ///   2. Key load/store use the `b64` suffix (the 64-bit-key path).
    ///   3. The vals payload remains u32 — row indices don't grow with key
    ///      width — so we still see `ld.global.u32` / `st.global.u32`.
    ///   4. Exactly one global atomic (the per-digit run reservation).
    #[test]
    fn scatter_with_indices_ptx_shape_i64() {
        let ptx = compile_radix_scatter_with_indices(DataType::Int64).unwrap();
        assert!(ptx.contains(".visible .entry bolt_radix_scatter_i64_with_indices("));
        assert!(ptx.contains("ld.global.b64"));
        assert!(ptx.contains("st.global.b64"));
        // Row-index payload is still u32 even for 64-bit keys.
        assert!(ptx.contains("ld.global.u32"));
        assert!(ptx.contains("st.global.u32"));
        assert_eq!(
            ptx.matches("atom.global.add.u32").count(),
            1,
            "keys+indices scatter (i64) must reserve its per-digit run with \
             exactly one global atomicAdd",
        );
        assert!(ptx.contains("atom.shared.add.u32"));
        assert_eq!(ptx.matches("bar.sync").count(), 3);
        assert!(ptx.contains("RANK_LOOP:"));
    }

    /// MSB-flip PTX shape for `i32`: XOR with `0x8000_0000` (decimal
    /// `2147483648`) in-place over the keys buffer.
    #[test]
    fn msb_flip_ptx_shape_i32() {
        let ptx = compile_radix_msb_flip(DataType::Int32).unwrap();
        assert!(ptx.contains(".visible .entry bolt_radix_msb_flip_i32("));
        assert!(ptx.contains("xor.b32"));
        assert!(ptx.contains("2147483648")); // 0x8000_0000
        assert!(ptx.contains("ld.global.b32"));
        assert!(ptx.contains("st.global.b32"));
        assert!(ptx.contains("DONE:"));
        assert!(ptx.contains("ret;"));
    }

    /// MSB-flip PTX shape for `i64`: XOR with `0x8000_0000_0000_0000`
    /// (decimal `9223372036854775808`) using `xor.b64`.
    #[test]
    fn msb_flip_ptx_shape_i64() {
        let ptx = compile_radix_msb_flip(DataType::Int64).unwrap();
        assert!(ptx.contains(".visible .entry bolt_radix_msb_flip_i64("));
        assert!(ptx.contains("xor.b64"));
        assert!(ptx.contains("9223372036854775808")); // 0x8000_0000_0000_0000
        assert!(ptx.contains("ld.global.b64"));
        assert!(ptx.contains("st.global.b64"));
    }

    /// The keys-only scatter must remain unchanged in shape — neither the
    /// keys+indices variant nor the MSB-flip helper should leak into it.
    /// This guards against accidental shared-emission regressions.
    #[test]
    fn keys_only_scatter_does_not_carry_vals() {
        for dty in [DataType::Int32, DataType::Int64] {
            let ptx = compile_radix_scatter(dty).unwrap();
            assert!(!ptx.contains("_with_indices"));
            assert!(!ptx.contains("xor.b32"));
            assert!(!ptx.contains("xor.b64"));
        }
    }

    /// Radix-step count: 32-bit keys need 8 passes at 4 bits per pass;
    /// 64-bit keys need 16. The executor uses this to size the per-step
    /// launch loop.
    #[test]
    fn radix_steps_counts() {
        assert_eq!(radix_steps_for(DataType::Int32).unwrap(), 8);
        assert_eq!(radix_steps_for(DataType::Int64).unwrap(), 16);
    }

    /// **The env-var off path still works.** This is the key contract from
    /// the v0.6 scaffold task: when the radix-sort dispatch gate is off,
    /// `try_gpu_radix_sort` returns `Ok(false)` regardless of dtype, so the
    /// executor falls back to its existing host path.
    ///
    /// Implementation note: we never call `set_var`/`remove_var` here. The
    /// Rust test runner shares one process across tests and env mutations
    /// race when run in parallel. Instead we use the test-only override hook
    /// [`set_radix_dispatch_for_tests`] to pin the cached atomic gate to a
    /// known value, then restore it to "uninitialised" so any follow-up
    /// test that depends on the env-derived default sees the same view it
    /// would have had if this test never ran.
    #[test]
    fn env_off_path_falls_back() {
        // Capture the cached state so we can restore it on exit; otherwise
        // a sibling test that latched the env value would see the OFF
        // override leak in. We use `None` (sentinel "re-read on next call")
        // as the restore target — equivalent to the process-startup state.
        set_radix_dispatch_for_tests(Some(false));

        // Float dtypes always fall back regardless of gate state because
        // the radix kernel doesn't support them yet — IEEE-monotonic
        // transform deferred. This exercises the dtype gate.
        assert!(!try_gpu_radix_sort(DataType::Float32).unwrap());
        assert!(!try_gpu_radix_sort(DataType::Float64).unwrap());
        assert!(!try_gpu_radix_sort(DataType::Bool).unwrap());
        assert!(!try_gpu_radix_sort(DataType::Utf8).unwrap());

        // Gate OFF: supported dtypes also fall back.
        assert!(!try_gpu_radix_sort(DataType::Int32).unwrap());
        assert!(!try_gpu_radix_sort(DataType::Int64).unwrap());
        assert!(!gpu_sort_env_enabled());

        // Gate ON: supported dtypes engage; unsupported dtypes still fall
        // back via the dtype gate.
        set_radix_dispatch_for_tests(Some(true));
        assert!(gpu_sort_env_enabled());
        assert!(try_gpu_radix_sort(DataType::Int32).unwrap());
        assert!(try_gpu_radix_sort(DataType::Int64).unwrap());
        assert!(!try_gpu_radix_sort(DataType::Float32).unwrap());
        assert!(!try_gpu_radix_sort(DataType::Bool).unwrap());

        // Restore to "uninitialised" so the next call latches from env —
        // the same behaviour the process would have at startup if this
        // test had never run.
        set_radix_dispatch_for_tests(None);
    }
}
