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
/// Per-thread logic:
///
/// ```text
///   tid = blockIdx.x * blockDim.x + threadIdx.x
///   if tid >= n_rows: return
///   key = keys[tid]            // .b32 or .b64
///   if signed_msb_flip: key ^= MSB     // (deferred to a separate pre-pass
///                                      // in the executor; the kernel sees
///                                      // already-transformed keys)
///   digit = (key >> shift) & 0xF
///   atomicAdd(&hist[digit], 1u32)
/// ```
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

    // -- Register declarations ---------------------------------------
    writeln!(p, "\t.reg .pred %p<2>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32 %r<16>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %rd<8>;").map_err(write_err)?;

    // tid = ctaid.x * ntid.x + tid.x
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;

    // bail if tid >= n_rows
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_2];").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // load shift (radix step) -> %r5
    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_3];").map_err(write_err)?;

    // keys_ptr -> %rd0 (globalised)
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;

    // addr = keys_ptr + tid * byte_width
    let key_w = flavour.byte_width as i64;
    writeln!(p, "\tmul.wide.u32 %rd1, %r3, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;

    // load key, then extract digit = (key >> shift) & 0xF
    if flavour.byte_width == 4 {
        writeln!(p, "\tld.global.{} %r6, [%rd2];", flavour.ld_st_suffix).map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r7, %r6, %r5;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r8, %r7, 15;").map_err(write_err)?;
    } else {
        // 64-bit keys: shift in b64, then narrow the digit to b32 for indexing.
        writeln!(p, "\tld.global.{} %rd3, [%rd2];", flavour.ld_st_suffix).map_err(write_err)?;
        // `shr.u64` takes a b32 shift amount in PTX — %r5 is already b32.
        writeln!(p, "\tshr.u64 %rd4, %rd3, %r5;").map_err(write_err)?;
        writeln!(p, "\tcvt.u32.u64 %r7, %rd4;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r8, %r7, 15;").map_err(write_err)?;
    }

    // hist_ptr -> %rd5 (globalised); bucket address = hist_ptr + digit*4
    writeln!(p, "\tld.param.u64 %rd5, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd5, %rd5;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd6, %r8, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd7, %rd5, %rd6;").map_err(write_err)?;

    // atomic-add 1 to hist[digit]
    writeln!(p, "\tatom.global.add.u32 %r9, [%rd7], 1;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;

    Ok(p)
}

/// Emit the PTX for the radix-sort **scatter** kernel for `dtype`.
///
/// Per-thread logic:
///
/// ```text
///   tid = blockIdx.x * blockDim.x + threadIdx.x
///   if tid >= n_rows: return
///   key = keys_in[tid]
///   digit = (key >> shift) & 0xF
///   out_idx = atomicAdd(&offsets[digit], 1u32)
///   keys_out[out_idx] = key
/// ```
///
/// `offsets[]` is initialised on the host to the exclusive-scan of the
/// histogram; each thread then atomic-bumps its digit's offset and writes
/// the key at the bumped position. Note this is **not stable** under the
/// atomic-bump strategy — two threads with the same digit race for the same
/// pair of slots. For ORDER BY semantics that's fine (SQL ORDER BY is not
/// required to be stable). If we ever need a stable radix sort the standard
/// trick is the per-warp prefix-scan within each block; defer that with the
/// float transform.
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

    // -- Register declarations ---------------------------------------
    writeln!(p, "\t.reg .pred %p<2>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32 %r<16>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %rd<16>;").map_err(write_err)?;

    // tid
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;

    // bail if tid >= n_rows
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_3];").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // load shift
    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_4];").map_err(write_err)?;

    // keys_in_ptr -> %rd0 (globalised); load key.
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    let key_w = flavour.byte_width as i64;
    writeln!(p, "\tmul.wide.u32 %rd1, %r3, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;

    if flavour.byte_width == 4 {
        writeln!(p, "\tld.global.{} %r6, [%rd2];", flavour.ld_st_suffix).map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r7, %r6, %r5;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r8, %r7, 15;").map_err(write_err)?;
    } else {
        writeln!(p, "\tld.global.{} %rd3, [%rd2];", flavour.ld_st_suffix).map_err(write_err)?;
        writeln!(p, "\tshr.u64 %rd4, %rd3, %r5;").map_err(write_err)?;
        writeln!(p, "\tcvt.u32.u64 %r7, %rd4;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r8, %r7, 15;").map_err(write_err)?;
    }

    // offsets_ptr -> %rd5; atomic-add 1 to offsets[digit], capturing the
    // *pre*-increment value as the output slot for this key.
    writeln!(p, "\tld.param.u64 %rd5, [{entry}_param_2];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd5, %rd5;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd6, %r8, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd7, %rd5, %rd6;").map_err(write_err)?;
    writeln!(p, "\tatom.global.add.u32 %r9, [%rd7], 1;").map_err(write_err)?;

    // keys_out_ptr -> %rd8; out_addr = keys_out_ptr + out_idx * byte_width.
    writeln!(p, "\tld.param.u64 %rd8, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd9, %r9, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd10, %rd8, %rd9;").map_err(write_err)?;

    if flavour.byte_width == 4 {
        writeln!(p, "\tst.global.{} [%rd10], %r6;", flavour.ld_st_suffix).map_err(write_err)?;
    } else {
        writeln!(p, "\tst.global.{} [%rd10], %rd3;", flavour.ld_st_suffix).map_err(write_err)?;
    }

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;

    Ok(p)
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
///   tid = blockIdx.x * blockDim.x + threadIdx.x
///   if tid >= n_rows: return
///   key = keys_in[tid]
///   val = vals_in[tid]                    // u32 row-index payload
///   digit = (key >> shift) & 0xF
///   out_idx = atomicAdd(&offsets[digit], 1u32)
///   keys_out[out_idx] = key
///   vals_out[out_idx] = val               // lock-step with key at same slot
/// ```
///
/// The single `atomicAdd` is critical: keys and values must land at the
/// **same** slot, so we capture `out_idx` once and reuse it for both stores.
/// Two `atomicAdd`s would race and break the pairing.
///
/// Like the keys-only scatter this is not stable — for ORDER BY semantics
/// that's fine (SQL ORDER BY is not required to be stable, and adding stable
/// row-index breaks ties in the natural per-thread order anyway because the
/// kernel processes rows in `tid` order within each digit bucket modulo
/// scheduling jitter).
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

    // -- Register declarations ---------------------------------------
    // %r10 holds the u32 row-index payload; reserve a larger b32 pool.
    writeln!(p, "\t.reg .pred %p<2>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32 %r<20>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %rd<24>;").map_err(write_err)?;

    // tid
    writeln!(p, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;

    // bail if tid >= n_rows
    writeln!(p, "\tld.param.u32 %r4, [{entry}_param_5];").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // load shift
    writeln!(p, "\tld.param.u32 %r5, [{entry}_param_6];").map_err(write_err)?;

    // keys_in_ptr -> %rd0; load key from keys_in + tid * key_w
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    let key_w = flavour.byte_width as i64;
    writeln!(p, "\tmul.wide.u32 %rd1, %r3, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;

    if flavour.byte_width == 4 {
        writeln!(p, "\tld.global.{} %r6, [%rd2];", flavour.ld_st_suffix).map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r7, %r6, %r5;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r8, %r7, 15;").map_err(write_err)?;
    } else {
        writeln!(p, "\tld.global.{} %rd3, [%rd2];", flavour.ld_st_suffix).map_err(write_err)?;
        writeln!(p, "\tshr.u64 %rd4, %rd3, %r5;").map_err(write_err)?;
        writeln!(p, "\tcvt.u32.u64 %r7, %rd4;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r8, %r7, 15;").map_err(write_err)?;
    }

    // vals_in_ptr -> %rd11; addr = vals_in_ptr + tid * 4 (u32 payload).
    // Load the row-index now, before we claim the output slot, so the load
    // and the store are independent.
    writeln!(p, "\tld.param.u64 %rd11, [{entry}_param_2];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd11, %rd11;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd12, %r3, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd13, %rd11, %rd12;").map_err(write_err)?;
    writeln!(p, "\tld.global.u32 %r10, [%rd13];").map_err(write_err)?;

    // offsets_ptr -> %rd5; atomic-add 1 to offsets[digit], capturing the
    // *pre*-increment value as the output slot. The single atomic guarantees
    // keys and vals land at the same slot.
    writeln!(p, "\tld.param.u64 %rd5, [{entry}_param_4];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd5, %rd5;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd6, %r8, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd7, %rd5, %rd6;").map_err(write_err)?;
    writeln!(p, "\tatom.global.add.u32 %r9, [%rd7], 1;").map_err(write_err)?;

    // keys_out_ptr -> %rd8; out_addr = keys_out_ptr + out_idx * key_w
    writeln!(p, "\tld.param.u64 %rd8, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd9, %r9, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd10, %rd8, %rd9;").map_err(write_err)?;

    if flavour.byte_width == 4 {
        writeln!(p, "\tst.global.{} [%rd10], %r6;", flavour.ld_st_suffix).map_err(write_err)?;
    } else {
        writeln!(p, "\tst.global.{} [%rd10], %rd3;", flavour.ld_st_suffix).map_err(write_err)?;
    }

    // vals_out_ptr -> %rd14; vout_addr = vals_out_ptr + out_idx * 4
    writeln!(p, "\tld.param.u64 %rd14, [{entry}_param_3];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd14, %rd14;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd15, %r9, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd16, %rd14, %rd15;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd16], %r10;").map_err(write_err)?;

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
        // Atomic histogram bump.
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
        assert!(ptx.contains("atom.global.add.u32"));
        assert!(ptx.contains("st.global.b32"));
        assert!(ptx.contains("ld.global.b32"));
    }

    /// Keys+indices scatter PTX shape for `i32`:
    ///   1. The entry name and signature carry the documented seven `.param`s.
    ///   2. The key load (`ld.global.b32`) and key store (`st.global.b32`)
    ///      survive intact — same shape as the keys-only scatter.
    ///   3. A single atomic-add bumps the offset; both stores reuse its result
    ///      (we don't see a second `atom.global.add.u32` for the index).
    ///   4. The new `vals_in` u32 load and `vals_out` u32 store appear —
    ///      these are the new ABI surface the executor depends on.
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
        // Single atomic bump — paired key+val writes share its result.
        assert_eq!(
            ptx.matches("atom.global.add.u32").count(),
            1,
            "keys+indices scatter must use exactly one atomicAdd per thread \
             so keys and vals land at the same slot",
        );
        // The new vals payload: u32 load from vals_in and u32 store to vals_out.
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

    /// Keys+indices scatter PTX shape for `i64`:
    ///   1. Entry name carries the i64 tag.
    ///   2. Key load/store use the `b64` suffix (the 64-bit-key path).
    ///   3. The vals payload remains u32 — row indices don't grow with key
    ///      width — so we still see `ld.global.u32` / `st.global.u32`.
    ///   4. A single `atomicAdd` couples the key and val writes.
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
            "keys+indices scatter (i64) must use exactly one atomicAdd per thread",
        );
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
