// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for GPU-side variable-width (string / `Utf8`) scalar functions.
//!
//! Until now the JIT could not emit device writes whose per-row width is not a
//! compile-time constant, so every scalar string function (`UPPER`, `LOWER`,
//! `LENGTH`, `SUBSTRING`, `CONCAT`) and `CASE`/`CAST` over `Utf8` was rejected
//! at physical lowering with *"not yet lowered to GPU"* and string production
//! ran entirely host-side (see [`crate::exec::string_ops`]). This module is the
//! first GPU codegen for that surface. It covers two distinct shapes:
//!
//! ## 1. Fixed-output-width: `LENGTH` (fully GPU)
//!
//! On a dictionary-encoded `Utf8` column the per-row byte length is a pure
//! gather: precompute one `i32` length per dictionary entry on the host (slot
//! `0` is the NULL sentinel, `0` bytes — matching
//! [`crate::exec::string_ops::length`]), upload it, and have the kernel emit
//! `out[tid] = length_table[indices[tid]]`. The output is `Int32`, a
//! compile-time-fixed 4 bytes per row, so no offset bookkeeping is needed.
//! This is the lowest-risk end-to-end path and is wired through the projection
//! lowering in `physical_plan.rs`. See [`compile_length_gather_kernel`].
//!
//! ## 2. Variable-output-width: `UPPER` / `LOWER` / `SUBSTRING` (two-pass)
//!
//! Producing a brand-new `Utf8` array whose row widths are data-dependent is
//! the classic GPU two-pass pattern:
//!
//! 1. **Length pass** ([`compile_varwidth_len_pass`]): each thread reads its
//!    input string slice (from the source `offsets` + `bytes` buffers) and
//!    writes the *output* byte length for its row into a per-row `u32`
//!    `row_lens` buffer. For `UPPER`/`LOWER` the output length equals the input
//!    length (ASCII case folding is length-preserving — the non-ASCII / UTF-8
//!    multibyte caveat is documented on the emitter). For `SUBSTRING` the
//!    length is `clamp(input_len, start, len)`.
//! 2. **Prefix scan** ([`crate::jit::prefix_scan`]): exclusive-scan `row_lens`
//!    into output `offsets` (and the grand total = the output `bytes` buffer
//!    size the host must allocate). We reuse the existing scan kernels rather
//!    than re-emitting a scan here.
//! 3. **Write pass** ([`compile_varwidth_write_pass`]): each thread copies /
//!    transforms its input slice into `out_bytes[out_offsets[tid] ..]`.
//!
//! Passes 1 and 3 share the same source-slice address arithmetic, factored
//! into [`emit_load_src_slice`].
//!
//! ## Testing convention
//!
//! Like every other kernel emitter in `src/jit`, these functions return PTX as
//! a `String` and are unit-tested by asserting on the emitted text (see the
//! `tests` module below and `tests/ptx_golden_tests.rs`). No GPU or CUDA
//! runtime is required to exercise them, so they build and test under the
//! `cuda-stub` feature.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::ScalarFnKind;

/// PTX target metadata baked into every emitted module. Kept in sync with the
/// other `src/jit` emitters (`scan_kernel.rs`, `prefix_scan.rs`).
const PTX_VERSION: &str = ".version 7.5";
/// Target SM architecture string.
const PTX_TARGET: &str = ".target sm_70";
/// Address size directive (we always use 64-bit pointers).
const PTX_ADDRESS_SIZE: &str = ".address_size 64";

/// Threads per block for the string kernels. Matches
/// [`crate::jit::prefix_scan::BLOCK_SIZE`] so the length pass, the scan, and
/// the write pass can all be launched with the same 1-D grid geometry.
pub const BLOCK_SIZE: u32 = 256;

/// Entry-point name for the dictionary-gather `LENGTH` kernel.
pub const LENGTH_GATHER_ENTRY: &str = "bolt_str_length_gather";

/// Entry-point name prefix for the variable-width length pass. The concrete
/// name appends the lowercased op (e.g. `bolt_str_len_pass_upper`).
const LEN_PASS_PREFIX: &str = "bolt_str_len_pass";

/// Entry-point name prefix for the variable-width write pass (e.g.
/// `bolt_str_write_pass_upper`).
const WRITE_PASS_PREFIX: &str = "bolt_str_write_pass";

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("string_kernel: write failed: {}", e))
}

/// Emit the shared three-line module header (`.version` / `.target` /
/// `.address_size`) plus a trailing blank line.
fn emit_header(ptx: &mut String) -> BoltResult<()> {
    writeln!(ptx, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(ptx, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;
    Ok(())
}

/// Lowercased op tag used to mangle a variable-width pass entry-point name.
///
/// Returns an error for ops that have no two-pass variable-width producer:
/// `LENGTH` is fixed-width (use [`compile_length_gather_kernel`]) and `CONCAT`
/// is deferred (multi-input two-pass; see the module TODO and the host
/// fallback in [`crate::exec::string_ops`]).
fn varwidth_tag(kind: ScalarFnKind) -> BoltResult<&'static str> {
    match kind {
        ScalarFnKind::Upper => Ok("upper"),
        ScalarFnKind::Lower => Ok("lower"),
        ScalarFnKind::Substring => Ok("substring"),
        ScalarFnKind::Length => Err(BoltError::Plan(
            "string_kernel: LENGTH is fixed-width; use compile_length_gather_kernel".into(),
        )),
        ScalarFnKind::Concat => Err(BoltError::Plan(
            // TODO(string-concat-gpu): CONCAT is a multi-input two-pass
            // producer (sum of N input lengths per row). Deferred — the
            // host fallback in `exec::string_ops` remains the supported
            // path; see `physical_plan.rs` lowering branch.
            "string_kernel: CONCAT GPU two-pass codegen not yet implemented; \
             host fallback remains reachable"
                .into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// 1. Fixed-output-width: LENGTH via dictionary-index gather.
// ---------------------------------------------------------------------------

/// Compile the fully-GPU `LENGTH` kernel for a dictionary-encoded `Utf8`
/// column.
///
/// The kernel performs a per-row gather of a precomputed per-dictionary-entry
/// `i32` length table:
///
/// ```text
/// out[tid] = length_table[indices[tid]]   for tid < n_rows
/// ```
///
/// The host builds `length_table` exactly as
/// [`crate::exec::string_ops::length`] does its host-side table: slot `0` is
/// the NULL sentinel (`0` bytes) and slot `k` (`k >= 1`) is
/// `dictionary[k-1].len()`. Because the table is indexed by the same `i32`
/// device indices the dictionary column already stores, the gather is a single
/// read-modify-write per row with no offset bookkeeping — it is fixed-output-
/// width (`Int32`, 4 bytes) and therefore the lowest-risk string path.
///
/// ## ABI
///
/// ```text
/// .visible .entry bolt_str_length_gather(
///     .param .u64 ..._param_0,   // indices       (i32*)  -- dictionary indices, one per row
///     .param .u64 ..._param_1,   // length_table  (i32*)  -- per-dict-entry byte length (slot 0 = NULL)
///     .param .u64 ..._param_2,   // out           (i32*)  -- per-row Int32 length output
///     .param .u32 ..._param_3    // n_rows
/// )
/// ```
///
/// Grid is 1-D, one thread per row, block size [`BLOCK_SIZE`].
///
/// The `indices` and `length_table` inputs are read-only, so their loads go
/// through the read-only cache (`ld.global.nc`), matching the convention in
/// `prefix_scan.rs` / `scan_kernel.rs`.
pub fn compile_length_gather_kernel() -> BoltResult<String> {
    let mut ptx = String::new();
    emit_header(&mut ptx)?;

    writeln!(ptx, ".visible .entry {}(", LENGTH_GATHER_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", LENGTH_GATHER_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", LENGTH_GATHER_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", LENGTH_GATHER_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_3", LENGTH_GATHER_ENTRY).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<16>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // gid = ctaid.x * ntid.x + tid.x
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    // n_rows guard.
    writeln!(ptx, "\tld.param.u32 %r4, [{}_param_3];", LENGTH_GATHER_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Globalize the three pointers.
    writeln!(ptx, "\tld.param.u64 %rd0, [{}_param_0];", LENGTH_GATHER_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{}_param_1];", LENGTH_GATHER_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{}_param_2];", LENGTH_GATHER_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;

    // idx = indices[tid] (read-only cache).
    writeln!(ptx, "\tmul.wide.u32 %rd3, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd4, %rd0, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u32 %r5, [%rd4];").map_err(write_err)?;

    // len = length_table[idx] (read-only cache).
    writeln!(ptx, "\tmul.wide.u32 %rd5, %r5, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd6, %rd1, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u32 %r6, [%rd6];").map_err(write_err)?;

    // out[tid] = len.
    writeln!(ptx, "\tadd.s64 %rd7, %rd2, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd7], %r6;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// 2. Variable-output-width two-pass: UPPER / LOWER / SUBSTRING.
// ---------------------------------------------------------------------------

/// Emit the source-slice address arithmetic shared by both passes.
///
/// Given the row index in `%r3` and the globalized `src_offsets` pointer in
/// `%rd_off` and `src_bytes` pointer in `%rd_bytes`, this writes:
///
/// * `%r_begin` = `src_offsets[tid]`     (start byte offset of the row's slice)
/// * `%r_end`   = `src_offsets[tid + 1]` (end byte offset; Arrow offset arrays
///   have `n_rows + 1` entries so this read is always in-bounds for `tid <
///   n_rows`)
/// * `%r_len`   = `%r_end - %r_begin`    (input byte length of the row)
/// * `%rd_slice` = `src_bytes + %r_begin` (pointer to the first input byte)
///
/// `src_offsets` is read through the read-only cache. The offsets are `i32`
/// (Arrow `Utf8`, not `LargeUtf8`); 64-bit `LargeUtf8` is out of scope here.
fn emit_load_src_slice(
    ptx: &mut String,
    r_begin: &str,
    r_end: &str,
    r_len: &str,
    rd_slice: &str,
    rd_off: &str,
    rd_bytes: &str,
) -> BoltResult<()> {
    // begin = src_offsets[tid]
    writeln!(ptx, "\tmul.wide.u32 %rd20, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd21, {off}, %rd20;", off = rd_off).map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u32 {begin}, [%rd21];", begin = r_begin).map_err(write_err)?;
    // end = src_offsets[tid + 1]
    writeln!(ptx, "\tadd.s64 %rd22, %rd21, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u32 {end}, [%rd22];", end = r_end).map_err(write_err)?;
    // len = end - begin
    writeln!(ptx, "\tsub.s32 {len}, {end}, {begin};", len = r_len, end = r_end, begin = r_begin)
        .map_err(write_err)?;
    // slice_ptr = src_bytes + begin
    writeln!(ptx, "\tmul.wide.u32 %rd23, {begin}, 1;", begin = r_begin).map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 {slice}, {bytes}, %rd23;", slice = rd_slice, bytes = rd_bytes)
        .map_err(write_err)?;
    Ok(())
}

/// Compile **pass 1** (the length pass) of the two-pass variable-width string
/// producer for `kind`.
///
/// Each thread computes the *output* byte length of its row and writes it to
/// `row_lens[tid]` (a `u32`). The host then exclusive-scans `row_lens` (via
/// [`crate::jit::prefix_scan`]) to obtain the output `offsets` array, whose
/// final element is the total size of the output `bytes` buffer it must
/// allocate before launching pass 2.
///
/// ## Output-length rule per op
///
/// * `UPPER` / `LOWER`: output length == input length. ASCII case folding is
///   byte-length-preserving. **Caveat:** for non-ASCII UTF-8 this is only an
///   approximation (e.g. some Unicode case mappings change byte length); the
///   GPU path is correct for ASCII data and the host fallback
///   ([`crate::exec::string_ops`]) remains the supported path for full Unicode
///   — the lowering branch only routes ASCII-safe cases here.
/// * `SUBSTRING(s, start, len)`: output length == `clamp(input_len - (start-1),
///   0, len)` (1-based `start`, SQL semantics). This emitter takes `start` and
///   `len` as compile-time-unknown kernel parameters.
///
/// ## ABI (UPPER / LOWER — 2-arg shape)
///
/// ```text
/// .visible .entry bolt_str_len_pass_upper(
///     .param .u64 ..._param_0,   // src_offsets (i32*, n_rows+1 entries)
///     .param .u64 ..._param_1,   // src_bytes   (u8*)
///     .param .u64 ..._param_2,   // row_lens    (u32*) -- OUTPUT, per-row out length
///     .param .u32 ..._param_3    // n_rows
/// )
/// ```
///
/// ## ABI (SUBSTRING — 4-arg shape, two extra u32s)
///
/// Same as above but with `..._param_4 = start (u32, 1-based)` and
/// `..._param_5 = sub_len (u32)` appended before... no: appended AFTER n_rows
/// to keep the row-count at a fixed position. The concrete layout is:
///
/// ```text
///     .param .u64 src_offsets
///     .param .u64 src_bytes
///     .param .u64 row_lens
///     .param .u32 n_rows
///     .param .u32 start
///     .param .u32 sub_len
/// ```
pub fn compile_varwidth_len_pass(kind: ScalarFnKind) -> BoltResult<String> {
    let tag = varwidth_tag(kind)?;
    let entry = format!("{}_{}", LEN_PASS_PREFIX, tag);
    let is_substring = matches!(kind, ScalarFnKind::Substring);

    let mut ptx = String::new();
    emit_header(&mut ptx)?;

    writeln!(ptx, ".visible .entry {}(", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", entry).map_err(write_err)?;
    if is_substring {
        writeln!(ptx, "\t.param .u32 {}_param_3,", entry).map_err(write_err)?;
        writeln!(ptx, "\t.param .u32 {}_param_4,", entry).map_err(write_err)?;
        writeln!(ptx, "\t.param .u32 {}_param_5", entry).map_err(write_err)?;
    } else {
        writeln!(ptx, "\t.param .u32 {}_param_3", entry).map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // gid + n_rows guard.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{}_param_3];", entry).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Globalize src_offsets / src_bytes / row_lens.
    writeln!(ptx, "\tld.param.u64 %rd0, [{}_param_0];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{}_param_1];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{}_param_2];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;

    // Load the source slice metadata: %r5=begin, %r6=end, %r7=in_len, %rd10=ptr.
    emit_load_src_slice(&mut ptx, "%r5", "%r6", "%r7", "%rd10", "%rd0", "%rd1")?;

    // out_len computation.
    if is_substring {
        // start (1-based), sub_len in params 4/5.
        writeln!(ptx, "\tld.param.u32 %r8, [{}_param_4];", entry).map_err(write_err)?;
        writeln!(ptx, "\tld.param.u32 %r9, [{}_param_5];", entry).map_err(write_err)?;
        // start0 = max(start - 1, 0). start is unsigned 1-based; if start==0
        // treat as 0 offset (defensive — SQL start is >= 1).
        writeln!(ptx, "\tsub.s32 %r10, %r8, 1;").map_err(write_err)?;
        writeln!(ptx, "\tmax.s32 %r10, %r10, 0;").map_err(write_err)?;
        // avail = in_len - start0   (bytes available from the start offset)
        writeln!(ptx, "\tsub.s32 %r11, %r7, %r10;").map_err(write_err)?;
        writeln!(ptx, "\tmax.s32 %r11, %r11, 0;").map_err(write_err)?;
        // out_len = min(avail, sub_len)
        writeln!(ptx, "\tmin.s32 %r12, %r11, %r9;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u32 %r13, %r12;").map_err(write_err)?;
    } else {
        // UPPER / LOWER: out_len == in_len (ASCII case folding).
        writeln!(ptx, "\tmov.u32 %r13, %r7;").map_err(write_err)?;
    }

    // row_lens[tid] = out_len
    writeln!(ptx, "\tmul.wide.u32 %rd11, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd2, %rd11;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd12], %r13;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Compile **pass 2** (the write pass) of the two-pass variable-width string
/// producer for `kind`.
///
/// After the host has exclusive-scanned the pass-1 `row_lens` into
/// `out_offsets` and allocated the output `bytes` buffer, this kernel copies /
/// transforms each input slice into its destination region:
///
/// ```text
/// dst = out_bytes + out_offsets[tid]
/// for i in 0 .. out_len[tid]:
///     dst[i] = transform(src_slice[i])
/// ```
///
/// The per-byte transform is a tight loop with a `WRITE_LOOP:` / `WRITE_DONE:`
/// structure (so a golden test can pin the loop body). The transform itself:
///
/// * `UPPER`: ASCII upper-case — `if 'a' <= b <= 'z' { b - 32 }`.
/// * `LOWER`: ASCII lower-case — `if 'A' <= b <= 'Z' { b + 32 }`.
/// * `SUBSTRING`: byte-for-byte copy of the clamped slice (the start offset is
///   folded into the source pointer; the length into the loop bound).
///
/// ## ABI (UPPER / LOWER — 5-arg shape)
///
/// ```text
/// .visible .entry bolt_str_write_pass_upper(
///     .param .u64 ..._param_0,   // src_offsets (i32*)
///     .param .u64 ..._param_1,   // src_bytes   (u8*)
///     .param .u64 ..._param_2,   // out_offsets (i32*, exclusive scan of row_lens)
///     .param .u64 ..._param_3,   // out_bytes   (u8*) -- OUTPUT buffer
///     .param .u32 ..._param_4    // n_rows
/// )
/// ```
///
/// ## ABI (SUBSTRING — 7-arg shape)
///
/// Same plus `..._param_5 = start (u32)` and `..._param_6 = sub_len (u32)`.
pub fn compile_varwidth_write_pass(kind: ScalarFnKind) -> BoltResult<String> {
    let tag = varwidth_tag(kind)?;
    let entry = format!("{}_{}", WRITE_PASS_PREFIX, tag);
    let is_substring = matches!(kind, ScalarFnKind::Substring);

    let mut ptx = String::new();
    emit_header(&mut ptx)?;

    writeln!(ptx, ".visible .entry {}(", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_3,", entry).map_err(write_err)?;
    if is_substring {
        writeln!(ptx, "\t.param .u32 {}_param_4,", entry).map_err(write_err)?;
        writeln!(ptx, "\t.param .u32 {}_param_5,", entry).map_err(write_err)?;
        writeln!(ptx, "\t.param .u32 {}_param_6", entry).map_err(write_err)?;
    } else {
        writeln!(ptx, "\t.param .u32 {}_param_4", entry).map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<40>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // gid + n_rows guard. n_rows lives at param_4.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{}_param_4];", entry).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Globalize the four pointers.
    writeln!(ptx, "\tld.param.u64 %rd0, [{}_param_0];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{}_param_1];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{}_param_2];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd3, [{}_param_3];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    // Source slice: %r5=begin, %r6=end, %r7=in_len, %rd10=src_ptr.
    emit_load_src_slice(&mut ptx, "%r5", "%r6", "%r7", "%rd10", "%rd0", "%rd1")?;

    // dst_ptr = out_bytes + out_offsets[tid]
    writeln!(ptx, "\tmul.wide.u32 %rd13, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd14, %rd2, %rd13;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u32 %r8, [%rd14];").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd15, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd16, %rd3, %rd15;").map_err(write_err)?; // dst base

    // Determine the copy length (%r9) and adjust the source pointer for
    // SUBSTRING's start offset.
    if is_substring {
        writeln!(ptx, "\tld.param.u32 %r20, [{}_param_5];", entry).map_err(write_err)?; // start (1-based)
        writeln!(ptx, "\tld.param.u32 %r21, [{}_param_6];", entry).map_err(write_err)?; // sub_len
        // start0 = max(start - 1, 0)
        writeln!(ptx, "\tsub.s32 %r22, %r20, 1;").map_err(write_err)?;
        writeln!(ptx, "\tmax.s32 %r22, %r22, 0;").map_err(write_err)?;
        // src_ptr += start0
        writeln!(ptx, "\tmul.wide.u32 %rd17, %r22, 1;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd10, %rd10, %rd17;").map_err(write_err)?;
        // avail = max(in_len - start0, 0)
        writeln!(ptx, "\tsub.s32 %r23, %r7, %r22;").map_err(write_err)?;
        writeln!(ptx, "\tmax.s32 %r23, %r23, 0;").map_err(write_err)?;
        // copy_len = min(avail, sub_len)
        writeln!(ptx, "\tmin.s32 %r9, %r23, %r21;").map_err(write_err)?;
    } else {
        // UPPER / LOWER copy the whole (length-preserving) slice.
        writeln!(ptx, "\tmov.u32 %r9, %r7;").map_err(write_err)?;
    }

    // Per-byte copy/transform loop. i in [0, copy_len).
    writeln!(ptx, "\tmov.u32 %r10, 0;").map_err(write_err)?; // i = 0
    writeln!(ptx, "WRITE_LOOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p1, %r10, %r9;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra WRITE_DONE;").map_err(write_err)?;
    // b = src_ptr[i]
    writeln!(ptx, "\tmul.wide.u32 %rd18, %r10, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd19, %rd10, %rd18;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u8 %rs0, [%rd19];").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u16 %r11, %rs0;").map_err(write_err)?;

    match kind {
        ScalarFnKind::Upper => {
            // if 'a'(97) <= b <= 'z'(122) { b -= 32 }
            writeln!(ptx, "\tsetp.lt.u32 %p2, %r11, 97;").map_err(write_err)?;
            writeln!(ptx, "\tsetp.gt.u32 %p3, %r11, 122;").map_err(write_err)?;
            writeln!(ptx, "\tor.pred %p4, %p2, %p3;").map_err(write_err)?;
            writeln!(ptx, "\tsub.s32 %r12, %r11, 32;").map_err(write_err)?;
            writeln!(ptx, "\tselp.b32 %r13, %r11, %r12, %p4;").map_err(write_err)?;
        }
        ScalarFnKind::Lower => {
            // if 'A'(65) <= b <= 'Z'(90) { b += 32 }
            writeln!(ptx, "\tsetp.lt.u32 %p2, %r11, 65;").map_err(write_err)?;
            writeln!(ptx, "\tsetp.gt.u32 %p3, %r11, 90;").map_err(write_err)?;
            writeln!(ptx, "\tor.pred %p4, %p2, %p3;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s32 %r12, %r11, 32;").map_err(write_err)?;
            writeln!(ptx, "\tselp.b32 %r13, %r11, %r12, %p4;").map_err(write_err)?;
        }
        ScalarFnKind::Substring => {
            // Byte-for-byte copy (no case transform).
            writeln!(ptx, "\tmov.b32 %r13, %r11;").map_err(write_err)?;
        }
        // varwidth_tag already rejected Length / Concat above.
        other => {
            return Err(BoltError::Plan(format!(
                "string_kernel: write pass not implemented for {:?}",
                other
            )))
        }
    }

    // dst[i] = transformed
    writeln!(ptx, "\tadd.s64 %rd24, %rd16, %rd18;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u16.u32 %rs1, %r13;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd24], %rs1;").map_err(write_err)?;
    // i += 1; loop.
    writeln!(ptx, "\tadd.s32 %r10, %r10, 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra WRITE_LOOP;").map_err(write_err)?;
    writeln!(ptx, "WRITE_DONE:").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- LENGTH gather (fully GPU, fixed-width) ----------------------------

    #[test]
    fn length_gather_header_and_abi() {
        let ptx = compile_length_gather_kernel().expect("compile length gather");
        assert!(ptx.contains(".version 7.5"), "{ptx}");
        assert!(ptx.contains(".target sm_70"), "{ptx}");
        assert!(ptx.contains(".address_size 64"), "{ptx}");
        // 4-param ABI: indices, length_table, out, n_rows.
        assert!(ptx.contains(".visible .entry bolt_str_length_gather("), "{ptx}");
        assert!(ptx.contains(".param .u64 bolt_str_length_gather_param_0,"), "{ptx}");
        assert!(ptx.contains(".param .u64 bolt_str_length_gather_param_1,"), "{ptx}");
        assert!(ptx.contains(".param .u64 bolt_str_length_gather_param_2,"), "{ptx}");
        assert!(ptx.contains(".param .u32 bolt_str_length_gather_param_3"), "{ptx}");
    }

    #[test]
    fn length_gather_is_a_double_indirection_gather() {
        let ptx = compile_length_gather_kernel().expect("compile");
        // The load-bearing shape: read the index, then read the length table
        // at that index (two read-only-cache loads), then a single u32 store.
        let n_nc = ptx.matches("ld.global.nc.u32").count();
        assert!(
            n_nc >= 2,
            "expected >=2 read-only-cache loads (indices + length_table), got {n_nc}\n{ptx}"
        );
        assert!(ptx.contains("st.global.u32"), "missing the Int32 length store\n{ptx}");
        // n_rows guard before any work.
        let guard = ptx.find("bra DONE").expect("guard branch");
        let store = ptx.find("st.global.u32").expect("store");
        assert!(guard < store, "n_rows guard must precede the store\n{ptx}");
    }

    // ---- Variable-width length pass ---------------------------------------

    #[test]
    fn upper_len_pass_is_length_preserving() {
        let ptx = compile_varwidth_len_pass(ScalarFnKind::Upper).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_len_pass_upper("), "{ptx}");
        // 4-param ABI for UPPER (no start/len).
        assert!(ptx.contains(".param .u32 bolt_str_len_pass_upper_param_3"), "{ptx}");
        assert!(
            !ptx.contains("bolt_str_len_pass_upper_param_4"),
            "UPPER len pass must NOT have a 5th param\n{ptx}"
        );
        // out_len = in_len: the row_lens store happens; the slice end-begin
        // subtraction is the input length computation.
        assert!(ptx.contains("sub.s32"), "missing in_len = end - begin\n{ptx}");
        assert!(ptx.contains("st.global.u32"), "missing row_lens store\n{ptx}");
    }

    #[test]
    fn substring_len_pass_has_start_and_len_params_and_clamps() {
        let ptx = compile_varwidth_len_pass(ScalarFnKind::Substring).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_len_pass_substring("), "{ptx}");
        // 6-param ABI: offsets, bytes, row_lens, n_rows, start, sub_len.
        assert!(ptx.contains(".param .u32 bolt_str_len_pass_substring_param_4,"), "{ptx}");
        assert!(ptx.contains(".param .u32 bolt_str_len_pass_substring_param_5"), "{ptx}");
        // Clamp arithmetic: max for the start clamp, min for the length cap.
        assert!(ptx.contains("max.s32"), "missing clamp max\n{ptx}");
        assert!(ptx.contains("min.s32"), "missing length-cap min\n{ptx}");
    }

    #[test]
    fn len_pass_rejects_length_and_concat() {
        let e = compile_varwidth_len_pass(ScalarFnKind::Length).unwrap_err();
        assert!(format!("{e}").contains("fixed-width"), "{e}");
        let e = compile_varwidth_len_pass(ScalarFnKind::Concat).unwrap_err();
        assert!(format!("{e}").contains("CONCAT"), "{e}");
    }

    // ---- Variable-width write pass ---------------------------------------

    #[test]
    fn upper_write_pass_has_ascii_case_fold_and_loop() {
        let ptx = compile_varwidth_write_pass(ScalarFnKind::Upper).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_write_pass_upper("), "{ptx}");
        // The per-byte copy loop structure is load-bearing.
        assert!(ptx.contains("WRITE_LOOP:"), "missing loop label\n{ptx}");
        assert!(ptx.contains("WRITE_DONE:"), "missing loop exit label\n{ptx}");
        // ASCII upper fold: compare against 'a'(97) and 'z'(122), subtract 32.
        assert!(ptx.contains("97"), "missing 'a' bound\n{ptx}");
        assert!(ptx.contains("122"), "missing 'z' bound\n{ptx}");
        assert!(ptx.contains("sub.s32 %r12, %r11, 32"), "missing -32 case fold\n{ptx}");
        // Per-byte store.
        assert!(ptx.contains("st.global.u8"), "missing byte store\n{ptx}");
    }

    #[test]
    fn lower_write_pass_adds_32_within_az() {
        let ptx = compile_varwidth_write_pass(ScalarFnKind::Lower).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_write_pass_lower("), "{ptx}");
        // ASCII lower fold: 'A'(65)/'Z'(90), add 32.
        assert!(ptx.contains("65"), "missing 'A' bound\n{ptx}");
        assert!(ptx.contains("90"), "missing 'Z' bound\n{ptx}");
        assert!(ptx.contains("add.s32 %r12, %r11, 32"), "missing +32 case fold\n{ptx}");
    }

    #[test]
    fn substring_write_pass_copies_and_takes_start_len() {
        let ptx = compile_varwidth_write_pass(ScalarFnKind::Substring).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_write_pass_substring("), "{ptx}");
        // 7-param ABI.
        assert!(ptx.contains(".param .u32 bolt_str_write_pass_substring_param_5,"), "{ptx}");
        assert!(ptx.contains(".param .u32 bolt_str_write_pass_substring_param_6"), "{ptx}");
        // No case fold for substring: it's a plain byte copy via mov.b32.
        assert!(ptx.contains("mov.b32 %r13, %r11"), "substring must be a plain copy\n{ptx}");
        assert!(!ptx.contains("sub.s32 %r12, %r11, 32"), "substring must not case-fold\n{ptx}");
        assert!(ptx.contains("WRITE_LOOP:"), "{ptx}");
    }

    #[test]
    fn write_pass_rejects_length_and_concat() {
        let e = compile_varwidth_write_pass(ScalarFnKind::Length).unwrap_err();
        assert!(format!("{e}").contains("fixed-width"), "{e}");
        let e = compile_varwidth_write_pass(ScalarFnKind::Concat).unwrap_err();
        assert!(format!("{e}").contains("CONCAT"), "{e}");
    }
}
