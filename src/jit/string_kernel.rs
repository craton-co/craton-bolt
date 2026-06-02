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

/// Entry-point name for the per-row variable-width `LIKE` matcher kernel.
pub const LIKE_MATCH_ENTRY: &str = "bolt_str_like_match";

/// Entry-point name prefix for the variable-width length pass. The concrete
/// name appends the lowercased op (e.g. `bolt_str_len_pass_upper`).
const LEN_PASS_PREFIX: &str = "bolt_str_len_pass";

/// Entry-point name prefix for the variable-width write pass (e.g.
/// `bolt_str_write_pass_upper`).
const WRITE_PASS_PREFIX: &str = "bolt_str_write_pass";

/// Entry-point name prefix for the N-input CONCAT **length pass**. The concrete
/// name appends the input arity (e.g. `bolt_str_concat_len_pass_2`).
const CONCAT_LEN_PASS_PREFIX: &str = "bolt_str_concat_len_pass";

/// Entry-point name prefix for the N-input CONCAT **write pass** (e.g.
/// `bolt_str_concat_write_pass_2`).
const CONCAT_WRITE_PASS_PREFIX: &str = "bolt_str_concat_write_pass";

/// Maximum number of CONCAT input columns the GPU two-pass producer supports in
/// a single kernel. Beyond this the executor keeps the host fallback
/// ([`crate::exec::string_ops_extended::concat`]) — the register/parameter
/// budget per launch grows linearly with `N`, and very wide CONCATs are rare.
pub const CONCAT_MAX_INPUTS: usize = 8;

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
        // TRIM is a single-input, length-reducing transform whose output
        // boundaries (leading/trailing ASCII-whitespace skip) are computed
        // identically in the length pass and the write pass — structurally the
        // same shape as SUBSTRING but with data-dependent (rather than
        // parameter-driven) start/end. ASCII-whitespace-only; see
        // [`TrimMode`] and [`compile_varwidth_len_pass`] for the byte rule and
        // why it is UTF-8-safe.
        ScalarFnKind::TrimBoth => Ok("trim_both"),
        ScalarFnKind::TrimLeading => Ok("trim_leading"),
        ScalarFnKind::TrimTrailing => Ok("trim_trailing"),
        ScalarFnKind::Length => Err(BoltError::Plan(
            "string_kernel: LENGTH is fixed-width; use compile_length_gather_kernel".into(),
        )),
        ScalarFnKind::Concat => Err(BoltError::Plan(
            // CONCAT is a multi-input two-pass producer with a fundamentally
            // different ABI (N offsets+bytes descriptor pairs, not one), so it
            // does NOT share the single-input `LEN_PASS_PREFIX` mangling /
            // `compile_varwidth_{len,write}_pass` path. Its GPU producer lives
            // in the dedicated [`compile_concat_len_pass`] /
            // [`compile_concat_write_pass`] (keyed by input arity N) — call
            // those, not this helper. The host fallback in
            // `exec::string_ops_extended::concat` remains the supported path for
            // arities beyond [`CONCAT_MAX_INPUTS`] and on any kernel/launch
            // error.
            "string_kernel: CONCAT uses the dedicated N-input \
             compile_concat_len_pass / compile_concat_write_pass producers, \
             not the single-input varwidth path"
                .into(),
        )),
        // New scalar string fns (OCTET_LENGTH, POSITION, REPLACE, LEFT/RIGHT,
        // LPAD/RPAD, REVERSE, INITCAP) are host-only — they have no two-pass
        // GPU producer, so they belong with LENGTH/CONCAT as an Err here and
        // route through the host fallback (see physical_plan host whitelist).
        ScalarFnKind::OctetLength
        | ScalarFnKind::Position
        | ScalarFnKind::Replace
        | ScalarFnKind::Left
        | ScalarFnKind::Right
        | ScalarFnKind::Lpad
        | ScalarFnKind::Rpad
        | ScalarFnKind::Reverse
        | ScalarFnKind::Initcap => Err(BoltError::Plan(
            "string_kernel: this string function has no GPU producer; host fallback only".into(),
        )),
    }
}

/// Which end(s) a GPU `TRIM` strips. Mirrors
/// [`crate::exec::string_ops_extended::TrimSide`] but lives here so the codegen
/// has no dependency on the exec module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrimMode {
    /// `TRIM(BOTH ...)` — strip leading and trailing whitespace.
    Both,
    /// `TRIM(LEADING ...)` — strip leading whitespace only.
    Leading,
    /// `TRIM(TRAILING ...)` — strip trailing whitespace only.
    Trailing,
}

/// Map a `ScalarFnKind` trim variant to its [`TrimMode`]. Returns `None` for
/// non-trim ops so the two-pass emitters can branch on "is this a TRIM".
fn trim_mode(kind: ScalarFnKind) -> Option<TrimMode> {
    match kind {
        ScalarFnKind::TrimBoth => Some(TrimMode::Both),
        ScalarFnKind::TrimLeading => Some(TrimMode::Leading),
        ScalarFnKind::TrimTrailing => Some(TrimMode::Trailing),
        _ => None,
    }
}

/// Emit a predicate `%p_out` that is TRUE when the byte in `%r_byte` is an
/// ASCII whitespace character, matching the bytes Rust's `str::trim` strips
/// from ASCII text: HT/LF/VT/FF/CR (0x09..=0x0D) and SPACE (0x20).
///
/// We test `(b >= 0x09 && b <= 0x0D) || b == 0x20`. All of these are
/// single-byte ASCII; a whitespace byte can never be a UTF-8 lead or
/// continuation byte (those are all >= 0x80), so trimming on these byte
/// boundaries can never split a multi-byte codepoint — the produced slice is
/// always valid UTF-8. The `%p_a`/`%p_b` scratch predicates are clobbered.
fn emit_is_ascii_ws(
    ptx: &mut String,
    p_out: &str,
    p_a: &str,
    p_b: &str,
    r_byte: &str,
) -> BoltResult<()> {
    // ws_range = (b >= 0x09) && (b <= 0x0D)
    writeln!(ptx, "\tsetp.ge.u32 {pa}, {b}, 9;", pa = p_a, b = r_byte).map_err(write_err)?;
    writeln!(ptx, "\tsetp.le.u32 {pb}, {b}, 13;", pb = p_b, b = r_byte).map_err(write_err)?;
    writeln!(ptx, "\tand.pred {po}, {pa}, {pb};", po = p_out, pa = p_a, pb = p_b)
        .map_err(write_err)?;
    // is_space = (b == 0x20)
    writeln!(ptx, "\tsetp.eq.u32 {pa}, {b}, 32;", pa = p_a, b = r_byte).map_err(write_err)?;
    // p_out = ws_range || is_space
    writeln!(ptx, "\tor.pred {po}, {po}, {pa};", po = p_out, pa = p_a).map_err(write_err)?;
    Ok(())
}

/// Emit the TRIM boundary computation shared by the length pass and the write
/// pass, so the two passes agree byte-for-byte on the trimmed window.
///
/// Inputs (already materialised by [`emit_load_src_slice`]):
/// * `r_inlen` — input byte length (`end - begin`).
/// * `rd_slice` — pointer to the first input byte.
///
/// Outputs:
/// * `r_tbegin` — index of the first KEPT byte (0 for LEADING-noop / TRAILING).
/// * `r_outlen` — number of kept bytes (`t_end - t_begin`).
///
/// The leading scan advances `t_begin` while `slice[t_begin]` is ASCII
/// whitespace and `t_begin < in_len` (skipped for TRAILING). The trailing scan
/// retreats `t_end` while `slice[t_end-1]` is whitespace and `t_end > t_begin`
/// (skipped for LEADING). Uses fixed scratch registers `%r30..%r35`, predicates
/// `%p4..%p7`, and `%rs0`/`%rd30..%rd31`; callers must not rely on those across
/// this call.
fn emit_trim_bounds(
    ptx: &mut String,
    mode: TrimMode,
    r_inlen: &str,
    rd_slice: &str,
    r_tbegin: &str,
    r_outlen: &str,
) -> BoltResult<()> {
    // t_begin = 0; t_end = in_len.
    writeln!(ptx, "\tmov.u32 {tb}, 0;", tb = r_tbegin).map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r31, {inl};", inl = r_inlen).map_err(write_err)?; // t_end

    // ---- Leading scan: advance t_begin past whitespace. ----
    if matches!(mode, TrimMode::Both | TrimMode::Leading) {
        writeln!(ptx, "TRIM_LEAD:").map_err(write_err)?;
        // stop if t_begin >= t_end (string is all whitespace / empty).
        writeln!(ptx, "\tsetp.ge.s32 %p4, {tb}, %r31;", tb = r_tbegin).map_err(write_err)?;
        writeln!(ptx, "\t@%p4 bra TRIM_LEAD_DONE;").map_err(write_err)?;
        // b = slice[t_begin]
        writeln!(ptx, "\tmul.wide.u32 %rd30, {tb}, 1;", tb = r_tbegin).map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd31, {sl}, %rd30;", sl = rd_slice).map_err(write_err)?;
        writeln!(ptx, "\tld.global.nc.u8 %rs0, [%rd31];").map_err(write_err)?;
        writeln!(ptx, "\tcvt.u32.u16 %r30, %rs0;").map_err(write_err)?;
        emit_is_ascii_ws(ptx, "%p5", "%p6", "%p7", "%r30")?;
        // if not whitespace -> done.
        writeln!(ptx, "\t@!%p5 bra TRIM_LEAD_DONE;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s32 {tb}, {tb}, 1;", tb = r_tbegin).map_err(write_err)?;
        writeln!(ptx, "\tbra TRIM_LEAD;").map_err(write_err)?;
        writeln!(ptx, "TRIM_LEAD_DONE:").map_err(write_err)?;
    }

    // ---- Trailing scan: retreat t_end past whitespace. ----
    if matches!(mode, TrimMode::Both | TrimMode::Trailing) {
        writeln!(ptx, "TRIM_TRAIL:").map_err(write_err)?;
        // stop if t_end <= t_begin.
        writeln!(ptx, "\tsetp.le.s32 %p4, %r31, {tb};", tb = r_tbegin).map_err(write_err)?;
        writeln!(ptx, "\t@%p4 bra TRIM_TRAIL_DONE;").map_err(write_err)?;
        // b = slice[t_end - 1]
        writeln!(ptx, "\tsub.s32 %r32, %r31, 1;").map_err(write_err)?;
        writeln!(ptx, "\tmul.wide.u32 %rd30, %r32, 1;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd31, {sl}, %rd30;", sl = rd_slice).map_err(write_err)?;
        writeln!(ptx, "\tld.global.nc.u8 %rs0, [%rd31];").map_err(write_err)?;
        writeln!(ptx, "\tcvt.u32.u16 %r30, %rs0;").map_err(write_err)?;
        emit_is_ascii_ws(ptx, "%p5", "%p6", "%p7", "%r30")?;
        writeln!(ptx, "\t@!%p5 bra TRIM_TRAIL_DONE;").map_err(write_err)?;
        writeln!(ptx, "\tmov.u32 %r31, %r32;").map_err(write_err)?; // t_end -= 1
        writeln!(ptx, "\tbra TRIM_TRAIL;").map_err(write_err)?;
        writeln!(ptx, "TRIM_TRAIL_DONE:").map_err(write_err)?;
    }

    // out_len = t_end - t_begin.
    writeln!(ptx, "\tsub.s32 {ol}, %r31, {tb};", ol = r_outlen, tb = r_tbegin)
        .map_err(write_err)?;
    Ok(())
}

/// Entry-point name of the variable-width **length pass** for `kind` (e.g.
/// `bolt_str_len_pass_upper`). Errors for ops with no two-pass producer
/// (`LENGTH` / `CONCAT`), matching [`compile_varwidth_len_pass`].
///
/// Host launchers use this to look up the compiled function by name rather than
/// re-deriving the mangling, so the entry-point convention stays owned here.
pub fn len_pass_entry(kind: ScalarFnKind) -> BoltResult<String> {
    Ok(format!("{}_{}", LEN_PASS_PREFIX, varwidth_tag(kind)?))
}

/// Entry-point name of the variable-width **write pass** for `kind` (e.g.
/// `bolt_str_write_pass_upper`). See [`len_pass_entry`].
pub fn write_pass_entry(kind: ScalarFnKind) -> BoltResult<String> {
    Ok(format!("{}_{}", WRITE_PASS_PREFIX, varwidth_tag(kind)?))
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
// 1b. Per-row variable-width `LIKE` matcher (Bool output).
// ---------------------------------------------------------------------------
//
// ⚠️ UNVALIDATED DEVICE CODE ⚠️
//
// `compile_like_match_kernel` emits a real per-row device matcher for the
// constant single-literal-segment `LIKE` shapes (EXACT / PREFIX / SUFFIX /
// CONTAINS, plus `NOT LIKE` via inversion). It has NOT been executed on GPU
// hardware in CI — this engine has no GPU at build/test time. Correctness is
// established by two host-side proxies only:
//
//   * the **host mirror** [`crate::exec::string_like::like_match_row`], which
//     replicates the exact per-row byte logic the PTX emits and is asserted
//     equal to [`crate::exec::like::PatternMatcher`] over a sample set, and
//   * the **PTX-shape tests** in this module, which pin the compare / branch
//     structure each mode emits.
//
// Until a GPU hardware test pass validates it, the executor
// ([`crate::exec::string_like`]) is conservatively host-fallback-safe: any
// unsupported layout / shape encountered at run time evaluates the SAME match
// on the host via `PatternMatcher`, so a latent device bug can only ever cost
// performance, never correctness.

/// Match mode for [`compile_like_match_kernel`]. Mirrors the four supported
/// single-literal-segment `LIKE` shapes; the SQL-level `%`-decomposition lives
/// in [`crate::exec::string_like::decompose_like_pattern`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LikeMode {
    /// `'lit'` — row matches iff `row_len == L && bytes == lit`.
    Exact,
    /// `'lit%'` — row matches iff `row_len >= L && bytes[0..L] == lit`.
    Prefix,
    /// `'%lit'` — row matches iff `row_len >= L && bytes[row_len-L..] == lit`.
    Suffix,
    /// `'%lit%'` — row matches iff `lit` occurs as a substring (naive scan over
    /// the `row_len - L + 1` candidate start offsets).
    Contains,
}

impl LikeMode {
    /// Lowercased tag used in PTX comments / debugging.
    fn tag(self) -> &'static str {
        match self {
            LikeMode::Exact => "exact",
            LikeMode::Prefix => "prefix",
            LikeMode::Suffix => "suffix",
            LikeMode::Contains => "contains",
        }
    }
}

/// Compile the per-row variable-width `LIKE` matcher kernel.
///
/// One thread per row reads the row's byte slice (via the Arrow-`Utf8`-shaped
/// `offsets` + `bytes` buffers, exactly like the two-pass producers) and writes
/// a single `u8` (0 / 1) into `out_mask[tid]`. The literal to match against is
/// uploaded as a small device buffer (`lit_ptr`, `lit_len` bytes). `mode`
/// selects the comparison; `negated` inverts the final 0/1.
///
/// Per-row logic by mode (`L = lit_len`, `n = row_len`):
///
///   * `Exact`    — `match = (n == L) && bytes[i] == lit[i] for i in 0..L`.
///   * `Prefix`   — `match = (n >= L) && bytes[i] == lit[i] for i in 0..L`.
///   * `Suffix`   — `match = (n >= L) && bytes[n-L+i] == lit[i] for i in 0..L`.
///   * `Contains` — `match = exists s in 0..=(n-L) s.t. bytes[s+i]==lit[i] ∀i`.
///
/// Empty literal (`L == 0`): `Prefix` / `Suffix` / `Contains` match every row
/// (`""` is a prefix / suffix / substring of anything); `Exact` matches iff
/// `n == 0`. The kernel handles `L == 0` by short-circuiting to the right
/// constant before the per-byte loop, so no out-of-bounds read occurs.
///
/// NULL handling lives on the HOST: the row-aligned input has no validity
/// channel, so NULL rows decode to an empty slice here; the executor re-applies
/// the input column's validity bitmap to the downloaded mask so a NULL row
/// surfaces as SQL NULL (dropped by the filter), matching
/// [`crate::exec::like::host_like`]'s 3VL.
///
/// ## ABI
///
/// ```text
/// .visible .entry bolt_str_like_match(
///     .param .u64 ..._param_0,   // offsets  (i32*, n_rows+1 entries)
///     .param .u64 ..._param_1,   // bytes    (u8*)
///     .param .u64 ..._param_2,   // lit      (u8*, lit_len bytes; may be 1-byte pad if lit_len==0)
///     .param .u64 ..._param_3,   // out_mask (u8*) -- OUTPUT, 0/1 per row
///     .param .u32 ..._param_4,   // n_rows
///     .param .u32 ..._param_5    // lit_len  (L)
/// )
/// ```
///
/// `mode` and `negated` are baked into the emitted code (compile-time), so the
/// ABI is identical across all four modes. Grid is 1-D, one thread per row,
/// block size [`BLOCK_SIZE`].
pub fn compile_like_match_kernel(mode: LikeMode, negated: bool) -> BoltResult<String> {
    let entry = LIKE_MATCH_ENTRY;
    let mut ptx = String::new();
    emit_header(&mut ptx)?;

    writeln!(ptx, "// mode={} negated={}", mode.tag(), negated).map_err(write_err)?;
    writeln!(ptx, ".visible .entry {}(", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_3,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_4,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_5", entry).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<12>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<40>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<40>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // gid + n_rows guard.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{}_param_4];", entry).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // Globalize offsets / bytes / lit / out_mask.
    writeln!(ptx, "\tld.param.u64 %rd0, [{}_param_0];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd1, [{}_param_1];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd2, [{}_param_2];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd3, [{}_param_3];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;

    // L = lit_len.
    writeln!(ptx, "\tld.param.u32 %r5, [{}_param_5];", entry).map_err(write_err)?;

    // Source slice: %r6=begin, %r7=end, %r8=row_len(n), %rd10=row_ptr.
    emit_load_src_slice(&mut ptx, "%r6", "%r7", "%r8", "%rd10", "%rd0", "%rd1")?;

    // `%r9` accumulates the raw (un-negated) match as 0/1. Default 0; set to 1
    // on a confirmed match. We branch to MATCH_TRUE / MATCH_FALSE labels and
    // converge at MATCH_DONE.
    writeln!(ptx, "\tmov.u32 %r9, 0;").map_err(write_err)?;

    // ---- Empty-literal short circuit (L == 0). -----------------------------
    // Prefix/Suffix/Contains: "" matches any row → true. Exact: true iff n==0.
    writeln!(ptx, "\tsetp.ne.u32 %p1, %r5, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LIT_NONEMPTY;").map_err(write_err)?;
    match mode {
        LikeMode::Prefix | LikeMode::Suffix | LikeMode::Contains => {
            writeln!(ptx, "\tmov.u32 %r9, 1;").map_err(write_err)?;
            writeln!(ptx, "\tbra MATCH_DONE;").map_err(write_err)?;
        }
        LikeMode::Exact => {
            // match = (n == 0)
            writeln!(ptx, "\tsetp.eq.u32 %p2, %r8, 0;").map_err(write_err)?;
            writeln!(ptx, "\tselp.b32 %r9, 1, 0, %p2;").map_err(write_err)?;
            writeln!(ptx, "\tbra MATCH_DONE;").map_err(write_err)?;
        }
    }
    writeln!(ptx, "LIT_NONEMPTY:").map_err(write_err)?;

    // ---- Length precondition. ----------------------------------------------
    // Exact: n == L. Prefix/Suffix/Contains: n >= L. On failure → no match.
    match mode {
        LikeMode::Exact => {
            writeln!(ptx, "\tsetp.ne.u32 %p3, %r8, %r5;").map_err(write_err)?;
            writeln!(ptx, "\t@%p3 bra MATCH_DONE;").map_err(write_err)?;
        }
        LikeMode::Prefix | LikeMode::Suffix | LikeMode::Contains => {
            writeln!(ptx, "\tsetp.lt.u32 %p3, %r8, %r5;").map_err(write_err)?;
            writeln!(ptx, "\t@%p3 bra MATCH_DONE;").map_err(write_err)?;
        }
    }

    match mode {
        LikeMode::Exact | LikeMode::Prefix => {
            // Compare bytes[0..L] against lit[0..L]. %r10 = i.
            writeln!(ptx, "\tmov.u32 %r10, 0;").map_err(write_err)?;
            writeln!(ptx, "CMP_LOOP:").map_err(write_err)?;
            writeln!(ptx, "\tsetp.ge.u32 %p4, %r10, %r5;").map_err(write_err)?;
            writeln!(ptx, "\t@%p4 bra CMP_OK;").map_err(write_err)?;
            // a = row_ptr[i]
            writeln!(ptx, "\tmul.wide.u32 %rd11, %r10, 1;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd12, %rd10, %rd11;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.nc.u8 %rs0, [%rd12];").map_err(write_err)?;
            writeln!(ptx, "\tcvt.u32.u16 %r11, %rs0;").map_err(write_err)?;
            // b = lit[i]
            writeln!(ptx, "\tadd.s64 %rd13, %rd2, %rd11;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.nc.u8 %rs1, [%rd13];").map_err(write_err)?;
            writeln!(ptx, "\tcvt.u32.u16 %r12, %rs1;").map_err(write_err)?;
            // if a != b → mismatch → MATCH_DONE (r9 still 0).
            writeln!(ptx, "\tsetp.ne.u32 %p5, %r11, %r12;").map_err(write_err)?;
            writeln!(ptx, "\t@%p5 bra MATCH_DONE;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s32 %r10, %r10, 1;").map_err(write_err)?;
            writeln!(ptx, "\tbra CMP_LOOP;").map_err(write_err)?;
            writeln!(ptx, "CMP_OK:").map_err(write_err)?;
            writeln!(ptx, "\tmov.u32 %r9, 1;").map_err(write_err)?;
        }
        LikeMode::Suffix => {
            // base = n - L; compare bytes[base+i] against lit[i].
            writeln!(ptx, "\tsub.s32 %r14, %r8, %r5;").map_err(write_err)?; // base
            writeln!(ptx, "\tmov.u32 %r10, 0;").map_err(write_err)?;
            writeln!(ptx, "CMP_LOOP:").map_err(write_err)?;
            writeln!(ptx, "\tsetp.ge.u32 %p4, %r10, %r5;").map_err(write_err)?;
            writeln!(ptx, "\t@%p4 bra CMP_OK;").map_err(write_err)?;
            // a = row_ptr[base + i]
            writeln!(ptx, "\tadd.s32 %r15, %r14, %r10;").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd11, %r15, 1;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd12, %rd10, %rd11;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.nc.u8 %rs0, [%rd12];").map_err(write_err)?;
            writeln!(ptx, "\tcvt.u32.u16 %r11, %rs0;").map_err(write_err)?;
            // b = lit[i]
            writeln!(ptx, "\tmul.wide.u32 %rd14, %r10, 1;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd13, %rd2, %rd14;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.nc.u8 %rs1, [%rd13];").map_err(write_err)?;
            writeln!(ptx, "\tcvt.u32.u16 %r12, %rs1;").map_err(write_err)?;
            writeln!(ptx, "\tsetp.ne.u32 %p5, %r11, %r12;").map_err(write_err)?;
            writeln!(ptx, "\t@%p5 bra MATCH_DONE;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s32 %r10, %r10, 1;").map_err(write_err)?;
            writeln!(ptx, "\tbra CMP_LOOP;").map_err(write_err)?;
            writeln!(ptx, "CMP_OK:").map_err(write_err)?;
            writeln!(ptx, "\tmov.u32 %r9, 1;").map_err(write_err)?;
        }
        LikeMode::Contains => {
            // Naive substring scan. For start s in [0, n-L]:
            //   if bytes[s..s+L] == lit[0..L] → match.
            // last_start = n - L (inclusive). %r16 = s (outer), %r10 = i (inner).
            writeln!(ptx, "\tsub.s32 %r16, %r8, %r5;").map_err(write_err)?; // last_start
            writeln!(ptx, "\tmov.u32 %r17, 0;").map_err(write_err)?; // s = 0
            writeln!(ptx, "SCAN_LOOP:").map_err(write_err)?;
            // if s > last_start → no match left.
            writeln!(ptx, "\tsetp.gt.s32 %p4, %r17, %r16;").map_err(write_err)?;
            writeln!(ptx, "\t@%p4 bra MATCH_DONE;").map_err(write_err)?;
            // inner compare bytes[s+i] vs lit[i].
            writeln!(ptx, "\tmov.u32 %r10, 0;").map_err(write_err)?;
            writeln!(ptx, "CMP_LOOP:").map_err(write_err)?;
            writeln!(ptx, "\tsetp.ge.u32 %p5, %r10, %r5;").map_err(write_err)?;
            writeln!(ptx, "\t@%p5 bra CMP_OK;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s32 %r15, %r17, %r10;").map_err(write_err)?; // s + i
            writeln!(ptx, "\tmul.wide.u32 %rd11, %r15, 1;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd12, %rd10, %rd11;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.nc.u8 %rs0, [%rd12];").map_err(write_err)?;
            writeln!(ptx, "\tcvt.u32.u16 %r11, %rs0;").map_err(write_err)?;
            writeln!(ptx, "\tmul.wide.u32 %rd14, %r10, 1;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s64 %rd13, %rd2, %rd14;").map_err(write_err)?;
            writeln!(ptx, "\tld.global.nc.u8 %rs1, [%rd13];").map_err(write_err)?;
            writeln!(ptx, "\tcvt.u32.u16 %r12, %rs1;").map_err(write_err)?;
            // mismatch at this start → advance s.
            writeln!(ptx, "\tsetp.ne.u32 %p6, %r11, %r12;").map_err(write_err)?;
            writeln!(ptx, "\t@%p6 bra SCAN_NEXT;").map_err(write_err)?;
            writeln!(ptx, "\tadd.s32 %r10, %r10, 1;").map_err(write_err)?;
            writeln!(ptx, "\tbra CMP_LOOP;").map_err(write_err)?;
            writeln!(ptx, "SCAN_NEXT:").map_err(write_err)?;
            writeln!(ptx, "\tadd.s32 %r17, %r17, 1;").map_err(write_err)?;
            writeln!(ptx, "\tbra SCAN_LOOP;").map_err(write_err)?;
            writeln!(ptx, "CMP_OK:").map_err(write_err)?;
            writeln!(ptx, "\tmov.u32 %r9, 1;").map_err(write_err)?;
        }
    }

    writeln!(ptx, "MATCH_DONE:").map_err(write_err)?;
    // Apply negation (NOT LIKE) by XOR-ing the raw 0/1 with 1.
    if negated {
        writeln!(ptx, "\txor.b32 %r9, %r9, 1;").map_err(write_err)?;
    }
    // out_mask[tid] = r9 (as u8).
    writeln!(ptx, "\tmul.wide.u32 %rd15, %r3, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd16, %rd3, %rd15;").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u16.u32 %rs2, %r9;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd16], %rs2;").map_err(write_err)?;

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

/// Convert a 1-based **character** start and a **character** length into a
/// byte window `[byte_start, byte_start + byte_copy)` into a UTF-8 slice.
///
/// `SUBSTRING` is defined over Unicode characters, not bytes. A naive
/// byte-offset implementation (`src += start-1`, copy `len` bytes) splits
/// multibyte codepoints: e.g. `SUBSTRING('héllo', 2, 1)` would land at the
/// first byte of the 2-byte `é` and copy a single byte (a partial char that
/// leaks the lead byte and drops the trail), and `SUBSTRING('世界x', 2, 2)`
/// would land in the middle of `世`. This helper walks whole characters so the
/// emitted window always starts and ends on a UTF-8 boundary.
///
/// A byte begins a character iff `(b & 0xC0) != 0x80` (continuation bytes are
/// `10xxxxxx`). The input is always valid UTF-8 (dictionary entries are valid
/// Rust strings), so stepping a lead byte then its continuation bytes is safe.
///
/// Inputs (all already loaded): `r_inlen` = slice byte length, `rd_slice` =
/// pointer to the slice's first byte, `r_start` = 1-based character start,
/// `r_sublen` = character count. Outputs: `r_bstart` = byte start offset and
/// `r_bcopy` = byte length of the selected window. `label` makes the emitted
/// branch targets unique within the enclosing kernel.
///
/// Scratch: `%r24`–`%r29`, `%rs2`, `%rd25`–`%rd26` (free in the SUBSTRING
/// branch — TRIM's `emit_trim_bounds` scratch is on the mutually-exclusive
/// TRIM path).
#[allow(clippy::too_many_arguments)]
fn emit_substring_char_window(
    ptx: &mut String,
    label: &str,
    r_inlen: &str,
    rd_slice: &str,
    r_start: &str,
    r_sublen: &str,
    r_bstart: &str,
    r_bcopy: &str,
) -> BoltResult<()> {
    // start0 = max(start - 1, 0) characters to skip; want = max(sub_len, 0)
    // characters to take.
    writeln!(ptx, "\tsub.s32 %r24, {st}, 1;", st = r_start).map_err(write_err)?;
    writeln!(ptx, "\tmax.s32 %r24, %r24, 0;").map_err(write_err)?; // %r24 = start0 (chars)
    writeln!(ptx, "\tmax.s32 %r25, {sl}, 0;", sl = r_sublen).map_err(write_err)?; // %r25 = want (chars)

    // Walk `start0` whole characters → byte index %r26 = byte_start.
    // %r26 = i (byte index), %r27 = chars skipped so far.
    writeln!(ptx, "\tmov.u32 %r26, 0;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r27, 0;").map_err(write_err)?;
    writeln!(ptx, "{lbl}_SKIP:", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p1, %r26, {inl};", inl = r_inlen).map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra {lbl}_SKIP_DONE;", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p1, %r27, %r24;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra {lbl}_SKIP_DONE;", lbl = label).map_err(write_err)?;
    // Step past this character's lead byte, then its continuation bytes.
    writeln!(ptx, "\tadd.s32 %r26, %r26, 1;").map_err(write_err)?;
    writeln!(ptx, "{lbl}_SKIP_CONT:", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p1, %r26, {inl};", inl = r_inlen).map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra {lbl}_SKIP_CONT_DONE;", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd25, %r26, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd26, {sl}, %rd25;", sl = rd_slice).map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u8 %rs2, [%rd26];").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u16 %r28, %rs2;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r28, %r28, 192;").map_err(write_err)?; // b & 0xC0
    writeln!(ptx, "\tsetp.ne.s32 %p1, %r28, 128;").map_err(write_err)?; // boundary if != 0x80
    writeln!(ptx, "\t@%p1 bra {lbl}_SKIP_CONT_DONE;", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r26, %r26, 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra {lbl}_SKIP_CONT;", lbl = label).map_err(write_err)?;
    writeln!(ptx, "{lbl}_SKIP_CONT_DONE:", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r27, %r27, 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra {lbl}_SKIP;", lbl = label).map_err(write_err)?;
    writeln!(ptx, "{lbl}_SKIP_DONE:", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 {bs}, %r26;", bs = r_bstart).map_err(write_err)?; // byte_start

    // Walk `want` more whole characters from byte_start → byte_end %r26.
    // %r27 reused as chars taken so far.
    writeln!(ptx, "\tmov.u32 %r27, 0;").map_err(write_err)?;
    writeln!(ptx, "{lbl}_TAKE:", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p1, %r26, {inl};", inl = r_inlen).map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra {lbl}_TAKE_DONE;", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p1, %r27, %r25;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra {lbl}_TAKE_DONE;", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r26, %r26, 1;").map_err(write_err)?;
    writeln!(ptx, "{lbl}_TAKE_CONT:", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p1, %r26, {inl};", inl = r_inlen).map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra {lbl}_TAKE_CONT_DONE;", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd25, %r26, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd26, {sl}, %rd25;", sl = rd_slice).map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u8 %rs2, [%rd26];").map_err(write_err)?;
    writeln!(ptx, "\tcvt.u32.u16 %r28, %rs2;").map_err(write_err)?;
    writeln!(ptx, "\tand.b32 %r28, %r28, 192;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p1, %r28, 128;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra {lbl}_TAKE_CONT_DONE;", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r26, %r26, 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra {lbl}_TAKE_CONT;", lbl = label).map_err(write_err)?;
    writeln!(ptx, "{lbl}_TAKE_CONT_DONE:", lbl = label).map_err(write_err)?;
    writeln!(ptx, "\tadd.s32 %r27, %r27, 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra {lbl}_TAKE;", lbl = label).map_err(write_err)?;
    writeln!(ptx, "{lbl}_TAKE_DONE:", lbl = label).map_err(write_err)?;
    // byte_copy = byte_end - byte_start
    writeln!(ptx, "\tsub.s32 {bc}, %r26, {bs};", bc = r_bcopy, bs = r_bstart)
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
///
/// ## ABI (TRIM `BOTH`/`LEADING`/`TRAILING` — 4-arg shape, same as UPPER)
///
/// TRIM takes no extra parameters; the side is baked into the entry name /
/// emitted scan. The output length is `in_len` minus the leading and/or
/// trailing ASCII-whitespace run (see [`emit_trim_bounds`]). Restricted to the
/// ASCII-whitespace default; custom trim-character sets and Unicode whitespace
/// stay on the host fallback ([`crate::exec::string_ops_extended::trim_str`]).
pub fn compile_varwidth_len_pass(kind: ScalarFnKind) -> BoltResult<String> {
    let tag = varwidth_tag(kind)?;
    let entry = format!("{}_{}", LEN_PASS_PREFIX, tag);
    let is_substring = matches!(kind, ScalarFnKind::Substring);
    let trim = trim_mode(kind);

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

    // TRIM's scan helper (`emit_trim_bounds`) uses scratch up to %r35 / %p7;
    // the other ops stay within the original budget.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<36>;").map_err(write_err)?;
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
    if let Some(mode) = trim {
        // TRIM: out_len = (trailing-trimmed end) - (leading-trimmed begin).
        // %r14 = t_begin (unused further here), %r13 = out_len. The write pass
        // recomputes the SAME bounds, so the two passes always agree.
        emit_trim_bounds(&mut ptx, mode, "%r7", "%rd10", "%r14", "%r13")?;
    } else if is_substring {
        // start (1-based CHARACTER), sub_len (CHARACTER count) in params 4/5.
        // SUBSTRING is character-indexed: walk whole UTF-8 characters to find
        // the byte window so no multibyte codepoint is split (see
        // `emit_substring_char_window`). out_len = byte length of that window.
        writeln!(ptx, "\tld.param.u32 %r8, [{}_param_4];", entry).map_err(write_err)?;
        writeln!(ptx, "\tld.param.u32 %r9, [{}_param_5];", entry).map_err(write_err)?;
        // %r12 = byte_start (unused here), %r13 = out_len (byte copy length).
        emit_substring_char_window(
            &mut ptx, "LEN", "%r7", "%rd10", "%r8", "%r9", "%r12", "%r13",
        )?;
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
///
/// ## ABI (TRIM — 5-arg shape, identical to UPPER/LOWER)
///
/// TRIM takes no extra parameters. It recomputes the SAME leading/trailing
/// ASCII-whitespace bounds as the length pass ([`emit_trim_bounds`]), advances
/// the source pointer to the first kept byte, sets `copy_len = out_len`, and
/// then runs the shared per-byte copy loop with no case transform (a plain
/// byte copy, like SUBSTRING). Because both passes derive the window from the
/// identical byte scan, the bytes written here exactly fill the region the
/// scanned `out_offsets` reserved.
pub fn compile_varwidth_write_pass(kind: ScalarFnKind) -> BoltResult<String> {
    let tag = varwidth_tag(kind)?;
    let entry = format!("{}_{}", WRITE_PASS_PREFIX, tag);
    let is_substring = matches!(kind, ScalarFnKind::Substring);
    let trim = trim_mode(kind);

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

    // TRIM's scan helper (`emit_trim_bounds`) uses scratch up to %r35 / %p7 /
    // %rd31; bump the b32 budget accordingly (the others already fit).
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<36>;").map_err(write_err)?;
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
    // SUBSTRING's start offset / TRIM's leading-whitespace skip.
    if let Some(mode) = trim {
        // Recompute the trim window: %r24 = t_begin, %r9 = out_len (copy_len).
        // These MUST match the length pass byte-for-byte (same scan helper).
        emit_trim_bounds(&mut ptx, mode, "%r7", "%rd10", "%r24", "%r9")?;
        // src_ptr += t_begin so the copy loop starts at the first kept byte.
        writeln!(ptx, "\tmul.wide.u32 %rd17, %r24, 1;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd10, %rd10, %rd17;").map_err(write_err)?;
    } else if is_substring {
        writeln!(ptx, "\tld.param.u32 %r20, [{}_param_5];", entry).map_err(write_err)?; // start (1-based char)
        writeln!(ptx, "\tld.param.u32 %r21, [{}_param_6];", entry).map_err(write_err)?; // sub_len (char count)
        // Character-indexed window: walk whole UTF-8 characters to find the
        // byte start offset (%r22) and byte copy length (%r9). This MUST match
        // the length pass byte-for-byte (same helper) so the bytes written
        // exactly fill the region `out_offsets` reserved.
        emit_substring_char_window(
            &mut ptx, "SUB", "%r7", "%rd10", "%r20", "%r21", "%r22", "%r9",
        )?;
        // src_ptr += byte_start so the copy loop begins at the first kept byte.
        writeln!(ptx, "\tmul.wide.u32 %rd17, %r22, 1;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd10, %rd10, %rd17;").map_err(write_err)?;
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
        ScalarFnKind::Substring
        | ScalarFnKind::TrimBoth
        | ScalarFnKind::TrimLeading
        | ScalarFnKind::TrimTrailing => {
            // Byte-for-byte copy (no case transform). For TRIM the source
            // pointer and copy length already select the kept window.
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

// ---------------------------------------------------------------------------
// 3. N-input variable-output-width two-pass: CONCAT.
// ---------------------------------------------------------------------------
//
// `CONCAT(a_0, a_1, ..., a_{N-1})` glues N `Utf8` inputs per row. Unlike the
// single-input two-pass producers (UPPER/LOWER/SUBSTRING/TRIM) the ABI carries
// **N** Arrow-`Utf8`-shaped source-slice descriptors (an `offsets`+`bytes`
// pointer pair per input). The output row width is the SUM of the N input byte
// lengths, so the same two-pass pattern applies:
//
//   * Length pass: `row_lens[tid] = sum_{k} (off_k[tid+1] - off_k[tid])`.
//   * Host exclusive-scan of `row_lens` → `out_offsets` + total `out_bytes`
//     size (reuses [`crate::exec::string_project::exclusive_scan_lens`], the
//     same helper UPPER/LOWER use).
//   * Write pass: `dst = out_bytes + out_offsets[tid]`; for each input k in
//     order, byte-copy its slice into `dst` advancing a running cursor.
//
// ## NULL semantics (matches the host fallback EXACTLY)
//
// Standard SQL `CONCAT(...)` returns NULL if ANY argument is NULL — this is the
// behaviour [`crate::exec::string_ops_extended::concat`] implements (it pushes
// index 0 / NULL whenever either side's index is 0). The GPU kernels carry **no
// validity channel**: a NULL input row decodes to an EMPTY slice on the host
// (see [`crate::exec::string_project::build_row_aligned_input`]), so the kernel
// happily sums/copies a zero-length contribution for it. The NULL-if-any-arg-
// NULL rule is then re-applied HOST-SIDE by the executor when it rebuilds the
// output array (it ORs the N input validity bitmaps and marks the row NULL).
// This mirrors the LIKE / UPPER NULL convention and keeps the length pass and
// the write pass byte-identical (no NULL-conditional branch to drift between
// the two passes).
//
// ## CRITICAL: length/write byte-length agreement
//
// Both passes compute each input's per-row byte length with the IDENTICAL
// `emit_load_src_slice` arithmetic (`off_k[tid+1] - off_k[tid]`). The length
// pass sums those; the write pass copies exactly that many bytes per input.
// Because the source of truth is the same offsets array read the same way, the
// total the write pass emits can never exceed the region `out_offsets`
// reserved — no out-of-bounds write.

/// Entry-point name of the CONCAT **length pass** for `n` inputs (e.g.
/// `bolt_str_concat_len_pass_2`). Host launchers use this to look up the
/// compiled function by name.
pub fn concat_len_pass_entry(n: usize) -> String {
    format!("{}_{}", CONCAT_LEN_PASS_PREFIX, n)
}

/// Entry-point name of the CONCAT **write pass** for `n` inputs (e.g.
/// `bolt_str_concat_write_pass_2`). See [`concat_len_pass_entry`].
pub fn concat_write_pass_entry(n: usize) -> String {
    format!("{}_{}", CONCAT_WRITE_PASS_PREFIX, n)
}

/// Validate the CONCAT input arity, shared by both pass compilers.
fn check_concat_arity(n: usize) -> BoltResult<()> {
    if n < 2 {
        return Err(BoltError::Plan(format!(
            "string_kernel: CONCAT needs >= 2 inputs, got {n}"
        )));
    }
    if n > CONCAT_MAX_INPUTS {
        return Err(BoltError::Plan(format!(
            "string_kernel: CONCAT GPU producer supports at most {} inputs, got {} \
             (use the host fallback for wider CONCATs)",
            CONCAT_MAX_INPUTS, n
        )));
    }
    Ok(())
}

/// Compile **pass 1** (the length pass) of the N-input CONCAT two-pass producer.
///
/// Each thread sums the per-row byte length of all `n` input slices and writes
/// the total to `row_lens[tid]` (a `u32`). NULL inputs decode host-side to empty
/// slices and contribute `0` (see the module section above); the SQL
/// NULL-if-any-arg-NULL rule is re-applied host-side, not here.
///
/// ## ABI (`n` inputs)
///
/// ```text
/// .visible .entry bolt_str_concat_len_pass_<n>(
///     .param .u64 ..._param_0,    // src_offsets_0 (i32*, n_rows+1 entries)
///     .param .u64 ..._param_1,    // src_bytes_0   (u8*)
///     .param .u64 ..._param_2,    // src_offsets_1 (i32*)
///     .param .u64 ..._param_3,    // src_bytes_1   (u8*)
///     ...                          // (offsets, bytes) pair per input, in order
///     .param .u64 ..._param_{2n},   // row_lens (u32*) -- OUTPUT, per-row out length
///     .param .u32 ..._param_{2n+1}  // n_rows
/// )
/// ```
///
/// Grid is 1-D, one thread per row, block size [`BLOCK_SIZE`].
pub fn compile_concat_len_pass(n: usize) -> BoltResult<String> {
    check_concat_arity(n)?;
    let entry = concat_len_pass_entry(n);

    let mut ptx = String::new();
    emit_header(&mut ptx)?;

    let row_lens_param = 2 * n; // param index of row_lens
    let n_rows_param = 2 * n + 1; // param index of n_rows

    writeln!(ptx, ".visible .entry {}(", entry).map_err(write_err)?;
    // N (offsets, bytes) pointer pairs.
    for k in 0..(2 * n) {
        writeln!(ptx, "\t.param .u64 {}_param_{},", entry, k).map_err(write_err)?;
    }
    // row_lens output pointer.
    writeln!(ptx, "\t.param .u64 {}_param_{},", entry, row_lens_param).map_err(write_err)?;
    // n_rows (last, no trailing comma).
    writeln!(ptx, "\t.param .u32 {}_param_{}", entry, n_rows_param).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<40>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<40>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // gid + n_rows guard.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{}_param_{}];", entry, n_rows_param).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // total = 0 (accumulated in %r13, matching the single-input convention).
    writeln!(ptx, "\tmov.u32 %r13, 0;").map_err(write_err)?;

    // For each input k: load offsets/bytes, compute in_len, add to total.
    for k in 0..n {
        let off_param = 2 * k;
        let bytes_param = 2 * k + 1;
        // Globalize this input's offsets (%rd0) and bytes (%rd1). `emit_load_src_slice`
        // only reads offsets; %rd1 is unused in the length pass but globalized for
        // symmetry / cheap and to keep the helper's signature satisfied.
        writeln!(ptx, "\tld.param.u64 %rd0, [{}_param_{}];", entry, off_param).map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
        writeln!(ptx, "\tld.param.u64 %rd1, [{}_param_{}];", entry, bytes_param)
            .map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
        // %r5=begin, %r6=end, %r7=in_len_k, %rd10=slice_ptr (ptr unused here).
        emit_load_src_slice(&mut ptx, "%r5", "%r6", "%r7", "%rd10", "%rd0", "%rd1")?;
        // total += in_len_k
        writeln!(ptx, "\tadd.s32 %r13, %r13, %r7;").map_err(write_err)?;
    }

    // row_lens[tid] = total. row_lens pointer is param_{2n}.
    writeln!(ptx, "\tld.param.u64 %rd2, [{}_param_{}];", entry, row_lens_param)
        .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd11, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd12, %rd2, %rd11;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u32 [%rd12], %r13;").map_err(write_err)?;

    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Compile **pass 2** (the write pass) of the N-input CONCAT two-pass producer.
///
/// After the host has exclusive-scanned the pass-1 `row_lens` into `out_offsets`
/// and allocated `out_bytes`, this kernel copies each input slice, in input
/// order, into the row's destination region:
///
/// ```text
/// dst = out_bytes + out_offsets[tid]
/// cursor = 0
/// for k in 0..n:
///     for i in 0 .. in_len_k(tid):
///         dst[cursor + i] = src_slice_k[i]
///     cursor += in_len_k(tid)
/// ```
///
/// The per-input copy uses a `CONCAT_WRITE_LOOP_k:` / `CONCAT_WRITE_DONE_k:`
/// structure so golden tests can pin the loop body and confirm one copy loop is
/// emitted per input.
///
/// ## ABI (`n` inputs)
///
/// ```text
/// .visible .entry bolt_str_concat_write_pass_<n>(
///     .param .u64 ..._param_0,    // src_offsets_0 (i32*)
///     .param .u64 ..._param_1,    // src_bytes_0   (u8*)
///     ...                          // (offsets, bytes) pair per input, in order
///     .param .u64 ..._param_{2n},   // out_offsets (i32*, exclusive scan of row_lens)
///     .param .u64 ..._param_{2n+1}, // out_bytes   (u8*) -- OUTPUT buffer
///     .param .u32 ..._param_{2n+2}  // n_rows
/// )
/// ```
///
/// Grid is 1-D, one thread per row, block size [`BLOCK_SIZE`].
pub fn compile_concat_write_pass(n: usize) -> BoltResult<String> {
    check_concat_arity(n)?;
    let entry = concat_write_pass_entry(n);

    let mut ptx = String::new();
    emit_header(&mut ptx)?;

    let out_off_param = 2 * n; // out_offsets pointer
    let out_bytes_param = 2 * n + 1; // out_bytes pointer
    let n_rows_param = 2 * n + 2; // n_rows

    writeln!(ptx, ".visible .entry {}(", entry).map_err(write_err)?;
    for k in 0..(2 * n) {
        writeln!(ptx, "\t.param .u64 {}_param_{},", entry, k).map_err(write_err)?;
    }
    writeln!(ptx, "\t.param .u64 {}_param_{},", entry, out_off_param).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_{},", entry, out_bytes_param).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_{}", entry, n_rows_param).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b16   %rs<4>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<40>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // gid + n_rows guard.
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{}_param_{}];", entry, n_rows_param).map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DONE;").map_err(write_err)?;

    // dst base = out_bytes + out_offsets[tid].
    writeln!(ptx, "\tld.param.u64 %rd2, [{}_param_{}];", entry, out_off_param).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd2, %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd3, [{}_param_{}];", entry, out_bytes_param)
        .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd3, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tmul.wide.u32 %rd13, %r3, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd14, %rd2, %rd13;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u32 %r8, [%rd14];").map_err(write_err)?; // out_offset
    writeln!(ptx, "\tmul.wide.u32 %rd15, %r8, 1;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd16, %rd3, %rd15;").map_err(write_err)?; // dst base ptr

    // cursor (running byte offset within the row's output region) = 0. Held in
    // %rd30 as a 64-bit byte offset added to the dst base.
    writeln!(ptx, "\tmov.u64 %rd30, 0;").map_err(write_err)?;

    // For each input k: load its slice and append it at the running cursor.
    for k in 0..n {
        let off_param = 2 * k;
        let bytes_param = 2 * k + 1;
        // Globalize input k's offsets (%rd0) and bytes (%rd1).
        writeln!(ptx, "\tld.param.u64 %rd0, [{}_param_{}];", entry, off_param).map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
        writeln!(ptx, "\tld.param.u64 %rd1, [{}_param_{}];", entry, bytes_param)
            .map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;
        // %r5=begin, %r6=end, %r9=in_len_k (copy_len), %rd10=src_ptr.
        // NOTE: %r9 holds the copy length — the IDENTICAL `end-begin` the length
        // pass summed for this input, so the bytes written exactly fill the
        // reserved region.
        emit_load_src_slice(&mut ptx, "%r5", "%r6", "%r9", "%rd10", "%rd0", "%rd1")?;

        // Per-byte copy loop for input k. i in [0, in_len_k).
        writeln!(ptx, "\tmov.u32 %r10, 0;").map_err(write_err)?; // i = 0
        writeln!(ptx, "CONCAT_WRITE_LOOP_{}:", k).map_err(write_err)?;
        writeln!(ptx, "\tsetp.ge.s32 %p1, %r10, %r9;").map_err(write_err)?;
        writeln!(ptx, "\t@%p1 bra CONCAT_WRITE_DONE_{};", k).map_err(write_err)?;
        // b = src_ptr[i]
        writeln!(ptx, "\tmul.wide.u32 %rd18, %r10, 1;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd19, %rd10, %rd18;").map_err(write_err)?;
        writeln!(ptx, "\tld.global.nc.u8 %rs0, [%rd19];").map_err(write_err)?;
        // dst[cursor + i] = b
        writeln!(ptx, "\tadd.s64 %rd24, %rd16, %rd30;").map_err(write_err)?; // dst + cursor
        writeln!(ptx, "\tadd.s64 %rd24, %rd24, %rd18;").map_err(write_err)?; // + i
        writeln!(ptx, "\tst.global.u8 [%rd24], %rs0;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s32 %r10, %r10, 1;").map_err(write_err)?;
        writeln!(ptx, "\tbra CONCAT_WRITE_LOOP_{};", k).map_err(write_err)?;
        writeln!(ptx, "CONCAT_WRITE_DONE_{}:", k).map_err(write_err)?;
        // cursor += in_len_k (advance the running output offset).
        writeln!(ptx, "\tcvt.u64.u32 %rd25, %r9;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd30, %rd30, %rd25;").map_err(write_err)?;
    }

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
        // Character-indexed window: `max` clamps start0/want to >= 0, and the
        // start/take scan loops walk whole UTF-8 characters (a byte begins a
        // char iff `(b & 0xC0) != 0x80`, tested via `and.b32 ..., 192`).
        assert!(ptx.contains("max.s32"), "missing clamp max\n{ptx}");
        assert!(ptx.contains("LEN_SKIP:"), "missing char-skip scan\n{ptx}");
        assert!(ptx.contains("LEN_TAKE:"), "missing char-take scan\n{ptx}");
        assert!(ptx.contains("and.b32 %r28, %r28, 192"), "missing UTF-8 boundary test\n{ptx}");
    }

    #[test]
    fn len_pass_rejects_length_and_concat() {
        let e = compile_varwidth_len_pass(ScalarFnKind::Length).unwrap_err();
        assert!(format!("{e}").contains("fixed-width"), "{e}");
        let e = compile_varwidth_len_pass(ScalarFnKind::Concat).unwrap_err();
        assert!(format!("{e}").contains("CONCAT"), "{e}");
    }

    // ---- TRIM length pass (single-input, length-reducing) -----------------

    #[test]
    fn trim_both_len_pass_scans_both_ends() {
        let ptx = compile_varwidth_len_pass(ScalarFnKind::TrimBoth).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_len_pass_trim_both("), "{ptx}");
        // 4-param ABI (no start/len): TRIM takes no extra params.
        assert!(ptx.contains(".param .u32 bolt_str_len_pass_trim_both_param_3"), "{ptx}");
        assert!(
            !ptx.contains("bolt_str_len_pass_trim_both_param_4"),
            "TRIM len pass must NOT have a 5th param\n{ptx}"
        );
        // BOTH emits both the leading and the trailing scan loops.
        assert!(ptx.contains("TRIM_LEAD:"), "missing leading scan\n{ptx}");
        assert!(ptx.contains("TRIM_TRAIL:"), "missing trailing scan\n{ptx}");
        // ASCII-whitespace byte test: HT..CR range (9..=13) plus SPACE (32).
        assert!(ptx.contains("setp.ge.u32 %p6, %r30, 9"), "missing ws low bound 9\n{ptx}");
        assert!(ptx.contains("setp.le.u32 %p7, %r30, 13"), "missing ws high bound 13\n{ptx}");
        assert!(ptx.contains("setp.eq.u32 %p6, %r30, 32"), "missing SPACE test 32\n{ptx}");
        // out_len store still happens.
        assert!(ptx.contains("st.global.u32"), "missing row_lens store\n{ptx}");
    }

    #[test]
    fn trim_leading_len_pass_scans_only_lead() {
        let ptx = compile_varwidth_len_pass(ScalarFnKind::TrimLeading).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_len_pass_trim_leading("), "{ptx}");
        assert!(ptx.contains("TRIM_LEAD:"), "leading must scan front\n{ptx}");
        assert!(!ptx.contains("TRIM_TRAIL:"), "leading must NOT scan tail\n{ptx}");
    }

    #[test]
    fn trim_trailing_len_pass_scans_only_trail() {
        let ptx = compile_varwidth_len_pass(ScalarFnKind::TrimTrailing).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_len_pass_trim_trailing("), "{ptx}");
        assert!(ptx.contains("TRIM_TRAIL:"), "trailing must scan tail\n{ptx}");
        assert!(!ptx.contains("TRIM_LEAD:"), "trailing must NOT scan front\n{ptx}");
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

    // ---- TRIM write pass --------------------------------------------------

    #[test]
    fn trim_both_write_pass_copies_kept_window() {
        let ptx = compile_varwidth_write_pass(ScalarFnKind::TrimBoth).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_write_pass_trim_both("), "{ptx}");
        // 5-param ABI, identical to UPPER/LOWER (no start/len params).
        assert!(ptx.contains(".param .u32 bolt_str_write_pass_trim_both_param_4"), "{ptx}");
        assert!(
            !ptx.contains("bolt_str_write_pass_trim_both_param_5"),
            "TRIM write pass must NOT have a 6th param\n{ptx}"
        );
        // Recomputes the same window scan as the length pass.
        assert!(ptx.contains("TRIM_LEAD:"), "missing leading scan\n{ptx}");
        assert!(ptx.contains("TRIM_TRAIL:"), "missing trailing scan\n{ptx}");
        // Plain byte copy (no case fold).
        assert!(ptx.contains("mov.b32 %r13, %r11"), "TRIM must be a plain copy\n{ptx}");
        assert!(!ptx.contains("sub.s32 %r12, %r11, 32"), "TRIM must not case-fold\n{ptx}");
        // Shared per-byte copy loop.
        assert!(ptx.contains("WRITE_LOOP:"), "{ptx}");
        assert!(ptx.contains("WRITE_DONE:"), "{ptx}");
        // Source pointer is advanced by t_begin (%r24) before the copy loop.
        assert!(
            ptx.contains("mul.wide.u32 %rd17, %r24, 1"),
            "TRIM must advance src by t_begin\n{ptx}"
        );
    }

    #[test]
    fn trim_leading_write_pass_only_scans_front() {
        let ptx = compile_varwidth_write_pass(ScalarFnKind::TrimLeading).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_write_pass_trim_leading("), "{ptx}");
        assert!(ptx.contains("TRIM_LEAD:"), "{ptx}");
        assert!(!ptx.contains("TRIM_TRAIL:"), "leading must not scan tail\n{ptx}");
    }

    #[test]
    fn trim_trailing_write_pass_only_scans_tail() {
        let ptx = compile_varwidth_write_pass(ScalarFnKind::TrimTrailing).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_write_pass_trim_trailing("), "{ptx}");
        assert!(ptx.contains("TRIM_TRAIL:"), "{ptx}");
        assert!(!ptx.contains("TRIM_LEAD:"), "trailing must not scan front\n{ptx}");
    }

    #[test]
    fn trim_entry_name_helpers_match_emitted_entries() {
        for (kind, len_name, write_name) in [
            (ScalarFnKind::TrimBoth, "bolt_str_len_pass_trim_both", "bolt_str_write_pass_trim_both"),
            (
                ScalarFnKind::TrimLeading,
                "bolt_str_len_pass_trim_leading",
                "bolt_str_write_pass_trim_leading",
            ),
            (
                ScalarFnKind::TrimTrailing,
                "bolt_str_len_pass_trim_trailing",
                "bolt_str_write_pass_trim_trailing",
            ),
        ] {
            assert_eq!(len_pass_entry(kind).unwrap(), len_name);
            assert_eq!(write_pass_entry(kind).unwrap(), write_name);
            let len_ptx = compile_varwidth_len_pass(kind).unwrap();
            assert!(
                len_ptx.contains(&format!(".visible .entry {len_name}(")),
                "len pass entry mismatch for {kind:?}\n{len_ptx}"
            );
            let write_ptx = compile_varwidth_write_pass(kind).unwrap();
            assert!(
                write_ptx.contains(&format!(".visible .entry {write_name}(")),
                "write pass entry mismatch for {kind:?}\n{write_ptx}"
            );
        }
    }

    // ---- Entry-point name helpers ----------------------------------------

    #[test]
    fn entry_name_helpers_match_emitted_entries() {
        // The host launcher looks functions up by these names; they MUST equal
        // the `.visible .entry` the corresponding compiler emits.
        for (kind, len_name, write_name) in [
            (ScalarFnKind::Upper, "bolt_str_len_pass_upper", "bolt_str_write_pass_upper"),
            (ScalarFnKind::Lower, "bolt_str_len_pass_lower", "bolt_str_write_pass_lower"),
            (
                ScalarFnKind::Substring,
                "bolt_str_len_pass_substring",
                "bolt_str_write_pass_substring",
            ),
        ] {
            assert_eq!(len_pass_entry(kind).unwrap(), len_name);
            assert_eq!(write_pass_entry(kind).unwrap(), write_name);
            let len_ptx = compile_varwidth_len_pass(kind).unwrap();
            assert!(
                len_ptx.contains(&format!(".visible .entry {len_name}(")),
                "len pass entry mismatch for {kind:?}\n{len_ptx}"
            );
            let write_ptx = compile_varwidth_write_pass(kind).unwrap();
            assert!(
                write_ptx.contains(&format!(".visible .entry {write_name}(")),
                "write pass entry mismatch for {kind:?}\n{write_ptx}"
            );
        }
    }

    #[test]
    fn entry_name_helpers_reject_length_and_concat() {
        assert!(len_pass_entry(ScalarFnKind::Length).is_err());
        assert!(write_pass_entry(ScalarFnKind::Concat).is_err());
    }

    // ---- LIKE matcher kernel (UNVALIDATED device path) --------------------

    #[test]
    fn like_match_header_and_abi() {
        let ptx = compile_like_match_kernel(LikeMode::Prefix, false).expect("compile");
        assert!(ptx.contains(".version 7.5"), "{ptx}");
        assert!(ptx.contains(".target sm_70"), "{ptx}");
        // 6-param ABI: offsets, bytes, lit, out_mask, n_rows, lit_len.
        assert!(ptx.contains(".visible .entry bolt_str_like_match("), "{ptx}");
        assert!(ptx.contains(".param .u64 bolt_str_like_match_param_0,"), "{ptx}");
        assert!(ptx.contains(".param .u64 bolt_str_like_match_param_3,"), "{ptx}");
        assert!(ptx.contains(".param .u32 bolt_str_like_match_param_4,"), "{ptx}");
        assert!(ptx.contains(".param .u32 bolt_str_like_match_param_5"), "{ptx}");
        // Output is a single u8 store per row.
        assert!(ptx.contains("st.global.u8"), "missing mask store\n{ptx}");
        // n_rows guard precedes the store.
        let guard = ptx.find("bra DONE").expect("guard");
        let store = ptx.find("st.global.u8").expect("store");
        assert!(guard < store, "n_rows guard must precede store\n{ptx}");
    }

    #[test]
    fn like_exact_emits_eq_length_check() {
        let ptx = compile_like_match_kernel(LikeMode::Exact, false).expect("compile");
        // Exact requires n == L: a `setp.ne.u32 %p3, %r8, %r5` (row_len vs L)
        // that branches away on inequality.
        assert!(ptx.contains("setp.ne.u32 %p3, %r8, %r5"), "exact length-eq check\n{ptx}");
        // Byte compare loop present.
        assert!(ptx.contains("CMP_LOOP:"), "{ptx}");
        assert!(ptx.contains("CMP_OK:"), "{ptx}");
        // Exact is not a substring scan.
        assert!(!ptx.contains("SCAN_LOOP:"), "exact must not scan\n{ptx}");
    }

    #[test]
    fn like_prefix_emits_ge_length_check_and_no_scan() {
        let ptx = compile_like_match_kernel(LikeMode::Prefix, false).expect("compile");
        // Prefix requires n >= L: a `setp.lt.u32 %p3, %r8, %r5` (fail if n<L).
        assert!(ptx.contains("setp.lt.u32 %p3, %r8, %r5"), "prefix length-ge check\n{ptx}");
        assert!(ptx.contains("CMP_LOOP:"), "{ptx}");
        assert!(!ptx.contains("SCAN_LOOP:"), "prefix must not scan\n{ptx}");
        // Prefix compares from offset 0 — no suffix base subtraction of n-L.
        assert!(!ptx.contains("sub.s32 %r14, %r8, %r5"), "prefix has no suffix base\n{ptx}");
    }

    #[test]
    fn like_suffix_emits_tail_base_offset() {
        let ptx = compile_like_match_kernel(LikeMode::Suffix, false).expect("compile");
        // Suffix computes base = n - L then compares the tail.
        assert!(ptx.contains("sub.s32 %r14, %r8, %r5"), "suffix base = n - L\n{ptx}");
        assert!(ptx.contains("CMP_LOOP:"), "{ptx}");
        assert!(!ptx.contains("SCAN_LOOP:"), "suffix must not scan\n{ptx}");
    }

    #[test]
    fn like_contains_emits_substring_scan() {
        let ptx = compile_like_match_kernel(LikeMode::Contains, false).expect("compile");
        // Contains is the naive double loop: outer SCAN over start offsets,
        // inner CMP over literal bytes.
        assert!(ptx.contains("SCAN_LOOP:"), "contains outer scan\n{ptx}");
        assert!(ptx.contains("SCAN_NEXT:"), "contains advances start\n{ptx}");
        assert!(ptx.contains("CMP_LOOP:"), "contains inner compare\n{ptx}");
        // last_start = n - L.
        assert!(ptx.contains("sub.s32 %r16, %r8, %r5"), "contains last_start = n - L\n{ptx}");
    }

    #[test]
    fn like_negated_xors_the_result() {
        let plain = compile_like_match_kernel(LikeMode::Prefix, false).expect("compile");
        let negated = compile_like_match_kernel(LikeMode::Prefix, true).expect("compile");
        assert!(!plain.contains("xor.b32 %r9, %r9, 1"), "non-negated must not XOR\n{plain}");
        assert!(negated.contains("xor.b32 %r9, %r9, 1"), "NOT LIKE must XOR the 0/1\n{negated}");
    }

    // ---- N-input CONCAT two-pass producer ---------------------------------

    #[test]
    fn concat_len_pass_2_input_abi_and_sum() {
        let ptx = compile_concat_len_pass(2).expect("compile");
        assert!(ptx.contains(".version 7.5"), "{ptx}");
        assert!(ptx.contains(".visible .entry bolt_str_concat_len_pass_2("), "{ptx}");
        // 2 inputs -> 4 pointer params (off0,bytes0,off1,bytes1), then row_lens
        // at param_4 and n_rows at param_5.
        assert!(ptx.contains(".param .u64 bolt_str_concat_len_pass_2_param_0,"), "{ptx}");
        assert!(ptx.contains(".param .u64 bolt_str_concat_len_pass_2_param_3,"), "{ptx}");
        assert!(ptx.contains(".param .u64 bolt_str_concat_len_pass_2_param_4,"), "row_lens\n{ptx}");
        assert!(ptx.contains(".param .u32 bolt_str_concat_len_pass_2_param_5"), "n_rows\n{ptx}");
        assert!(
            !ptx.contains("bolt_str_concat_len_pass_2_param_6"),
            "2-input len pass must NOT have a 7th param\n{ptx}"
        );
        // The length sum: each input's in_len = end - begin (sub.s32) is added
        // into the running total (add.s32 %r13, %r13, %r7), once per input.
        assert_eq!(
            ptx.matches("add.s32 %r13, %r13, %r7").count(),
            2,
            "expected one length-accumulate per input\n{ptx}"
        );
        // A single u32 row_lens store at the end.
        assert!(ptx.contains("st.global.u32 [%rd12], %r13"), "missing row_lens store\n{ptx}");
        // n_rows guard precedes the store.
        let guard = ptx.find("bra DONE").expect("guard");
        let store = ptx.find("st.global.u32 [%rd12]").expect("store");
        assert!(guard < store, "n_rows guard must precede store\n{ptx}");
    }

    #[test]
    fn concat_len_pass_3_input_has_three_accumulates() {
        let ptx = compile_concat_len_pass(3).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_concat_len_pass_3("), "{ptx}");
        // 3 inputs -> 6 pointer params; row_lens at param_6, n_rows at param_7.
        assert!(ptx.contains(".param .u64 bolt_str_concat_len_pass_3_param_5,"), "{ptx}");
        assert!(ptx.contains(".param .u64 bolt_str_concat_len_pass_3_param_6,"), "row_lens\n{ptx}");
        assert!(ptx.contains(".param .u32 bolt_str_concat_len_pass_3_param_7"), "n_rows\n{ptx}");
        assert_eq!(
            ptx.matches("add.s32 %r13, %r13, %r7").count(),
            3,
            "expected one length-accumulate per input\n{ptx}"
        );
    }

    #[test]
    fn concat_write_pass_2_input_abi_and_loops() {
        let ptx = compile_concat_write_pass(2).expect("compile");
        assert!(ptx.contains(".visible .entry bolt_str_concat_write_pass_2("), "{ptx}");
        // 2 inputs -> 4 pointer params, then out_offsets (param_4), out_bytes
        // (param_5), n_rows (param_6).
        assert!(ptx.contains(".param .u64 bolt_str_concat_write_pass_2_param_4,"), "out_offsets\n{ptx}");
        assert!(ptx.contains(".param .u64 bolt_str_concat_write_pass_2_param_5,"), "out_bytes\n{ptx}");
        assert!(ptx.contains(".param .u32 bolt_str_concat_write_pass_2_param_6"), "n_rows\n{ptx}");
        // One copy loop per input, in order.
        assert!(ptx.contains("CONCAT_WRITE_LOOP_0:"), "missing input-0 copy loop\n{ptx}");
        assert!(ptx.contains("CONCAT_WRITE_DONE_0:"), "{ptx}");
        assert!(ptx.contains("CONCAT_WRITE_LOOP_1:"), "missing input-1 copy loop\n{ptx}");
        assert!(ptx.contains("CONCAT_WRITE_DONE_1:"), "{ptx}");
        assert!(
            !ptx.contains("CONCAT_WRITE_LOOP_2:"),
            "2-input write pass must NOT emit a third loop\n{ptx}"
        );
        // Plain byte copy (no case fold): loads u8 then stores u8.
        assert!(ptx.contains("ld.global.nc.u8 %rs0"), "{ptx}");
        assert!(ptx.contains("st.global.u8 [%rd24], %rs0"), "{ptx}");
        // Running cursor advances by the same in_len (%r9) the length pass summed.
        assert!(ptx.contains("add.s64 %rd30, %rd30, %rd25"), "cursor advance\n{ptx}");
    }

    #[test]
    fn concat_write_pass_3_input_has_three_loops() {
        let ptx = compile_concat_write_pass(3).expect("compile");
        assert!(ptx.contains("CONCAT_WRITE_LOOP_0:"), "{ptx}");
        assert!(ptx.contains("CONCAT_WRITE_LOOP_1:"), "{ptx}");
        assert!(ptx.contains("CONCAT_WRITE_LOOP_2:"), "{ptx}");
        assert!(!ptx.contains("CONCAT_WRITE_LOOP_3:"), "{ptx}");
    }

    #[test]
    fn concat_len_and_write_use_identical_per_input_length() {
        // CRITICAL OOB-safety property: both passes derive each input's byte
        // length from the SAME `end - begin` slice arithmetic via
        // `emit_load_src_slice`. The length pass sums `%r7` (its in_len reg);
        // the write pass copies `%r9` bytes (its in_len reg) per input. Both
        // come from the identical helper, so the totals agree by construction.
        let len = compile_concat_len_pass(2).unwrap();
        let write = compile_concat_write_pass(2).unwrap();
        // Both passes emit `sub.s32 <len>, %r6, %r5` (end - begin) per input.
        assert!(len.contains("sub.s32 %r7, %r6, %r5"), "len pass in_len calc\n{len}");
        assert!(write.contains("sub.s32 %r9, %r6, %r5"), "write pass in_len calc\n{write}");
        // Same number of slice loads (one per input) in each pass.
        assert_eq!(len.matches("sub.s32 %r7, %r6, %r5").count(), 2, "{len}");
        assert_eq!(write.matches("sub.s32 %r9, %r6, %r5").count(), 2, "{write}");
    }

    #[test]
    fn concat_entry_name_helpers_match_emitted_entries() {
        for n in 2..=CONCAT_MAX_INPUTS {
            let len_name = concat_len_pass_entry(n);
            let write_name = concat_write_pass_entry(n);
            assert_eq!(len_name, format!("bolt_str_concat_len_pass_{n}"));
            assert_eq!(write_name, format!("bolt_str_concat_write_pass_{n}"));
            let len_ptx = compile_concat_len_pass(n).unwrap();
            assert!(
                len_ptx.contains(&format!(".visible .entry {len_name}(")),
                "len entry mismatch for n={n}\n{len_ptx}"
            );
            let write_ptx = compile_concat_write_pass(n).unwrap();
            assert!(
                write_ptx.contains(&format!(".visible .entry {write_name}(")),
                "write entry mismatch for n={n}\n{write_ptx}"
            );
        }
    }

    #[test]
    fn concat_rejects_arity_below_two_and_above_max() {
        assert!(compile_concat_len_pass(0).is_err());
        assert!(compile_concat_len_pass(1).is_err());
        assert!(compile_concat_write_pass(1).is_err());
        let e = compile_concat_len_pass(CONCAT_MAX_INPUTS + 1).unwrap_err();
        assert!(format!("{e}").contains("at most"), "{e}");
        assert!(compile_concat_write_pass(CONCAT_MAX_INPUTS + 1).is_err());
    }

    #[test]
    fn like_all_modes_handle_empty_literal() {
        // Every mode short-circuits L==0 before the per-byte loop (no OOB read).
        for mode in [LikeMode::Exact, LikeMode::Prefix, LikeMode::Suffix, LikeMode::Contains] {
            let ptx = compile_like_match_kernel(mode, false).expect("compile");
            assert!(
                ptx.contains("setp.ne.u32 %p1, %r5, 0"),
                "mode {mode:?} must test lit_len==0 first\n{ptx}"
            );
            assert!(ptx.contains("LIT_NONEMPTY:"), "mode {mode:?}\n{ptx}");
        }
    }
}
