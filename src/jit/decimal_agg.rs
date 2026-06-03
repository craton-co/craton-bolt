// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for the `SUM(Decimal128)` reduction kernel.
//!
//! `Decimal128` is a fixed-width 16-byte (`i128`) value: precision/scale only
//! affect interpretation, not storage, so a SUM over a decimal column is a
//! plain 128-bit integer add (modulo the overflow concerns SQL leaves to the
//! caller — see the host finalize in `crate::exec::aggregate`).
//!
//! ## Why a bespoke kernel (vs. the generic `agg_kernels::compile_reduction_kernel`)
//!
//! The generic reduction kernel keys its codegen on `ptx_type_info`, which only
//! knows about 32/64-bit register classes. A 128-bit accumulator has no native
//! PTX register class on sm_70, so we carry it as **two `.u64` halves** (`lo`,
//! `hi`) and implement the add with the carry-chain pair `add.cc.u64` /
//! `addc.u64`. That doesn't fit the templated single-register reduction, hence
//! this dedicated emitter.
//!
//! ## No atomics
//!
//! sm_70 has no 128-bit atomic add, and emulating one with a `b128` CAS loop is
//! both slow and unavailable as a single instruction. Instead we use the same
//! **two-stage block-reduce** shape the rest of the aggregate path already
//! relies on: each block reduces its slice into one per-block `i128` partial in
//! shared memory and writes it to `output[blockIdx.x]`; the host sums the
//! per-block partials (a tiny `Vec<i128>` fold). This is atomic-free and
//! deterministic.
//!
//! ## Reduction shape
//!
//! Mirrors `compile_reduction_kernel`'s structure but operates on the (lo, hi)
//! pair throughout:
//!
//!   - Phase 1 (strides 128, 64, 32): shared-memory tree with `bar.sync`
//!     between halvings, gated on `tid < stride`.
//!   - Phase 2 (strides 16, 8, 4, 2, 1): warp-0-only. sm_70 has no
//!     `shfl.sync.down.b128`, so each 64-bit half is shuffled with
//!     `shfl.sync.down.b32` on its two 32-bit sub-halves, recombined, and the
//!     128-bit carry-add applied. (Same b32-shuffle trick the i64/f64 paths in
//!     `agg_kernels` already use, just done twice.)
//!
//! ## ABI
//!
//! ```text
//! .visible .entry bolt_decimal_sum_reduce(
//!     .param .u64 input_ptr,    // *const i128, n_rows elements
//!     .param .u64 output_ptr,   // *mut i128,   one per block
//!     .param .u32 n_rows
//! )
//! ```
//! Block size is [`crate::jit::agg_kernels::BLOCK_SIZE`] (256); the grid is
//! sized by the launcher to `ceil(n_rows / BLOCK_SIZE)`.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::jit::agg_kernels::BLOCK_SIZE;

/// PTX entry-point name for the Decimal128 SUM reduction kernel.
pub const DECIMAL_SUM_KERNEL_ENTRY: &str = "bolt_decimal_sum_reduce";

/// Byte width of the `i128` accumulator element (input and per-block output).
pub const DECIMAL_ELEM_BYTES: usize = 16;

/// Adapt an `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("decimal_agg: write failed: {}", e))
}

/// Generate PTX for the per-block `SUM(Decimal128)` reduction kernel.
///
/// The kernel reduces its block's slice of the input `i128` column into a
/// single `i128` partial written to `output[blockIdx.x]`. The host performs
/// the final cross-block fold (see `crate::exec::aggregate`). NULL handling is
/// the caller's responsibility (host-strip before upload), exactly as for the
/// primitive `SUM` kernel.
pub fn compile_decimal_sum_kernel() -> BoltResult<String> {
    // Two shared-memory scratch pads, one per 64-bit half, indexed by tid.
    // Splitting lo/hi into separate arrays keeps the per-half addressing a
    // simple `tid * 8` stride (vs. interleaving into a 16-byte stride and
    // computing two sub-offsets).
    let half_shared_bytes = BLOCK_SIZE as usize * 8;

    let mut ptx = String::new();
    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 8 .b8 sdata_lo[{n}];",
        n = half_shared_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 8 .b8 sdata_hi[{n}];",
        n = half_shared_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", DECIMAL_SUM_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", DECIMAL_SUM_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", DECIMAL_SUM_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_2", DECIMAL_SUM_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register declarations. Generous counts — PTX `.reg` only names registers.
    // %rd holds addresses + the two 64-bit accumulator halves; %r holds the
    // 32-bit thread-index scratch and the shfl sub-halves.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // gid = ctaid.x * ntid.x + tid.x
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_2];",
        DECIMAL_SUM_KERNEL_ENTRY
    )
    .map_err(write_err)?;

    // Load this thread's i128 value (lo into %rd1, hi into %rd2), or the
    // additive identity (0, 0) if past the end.
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        DECIMAL_SUM_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DEC_LOAD_IDENTITY;").map_err(write_err)?;
    // addr = base + tid * 16
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd3, %r3, {bytes};",
        bytes = DECIMAL_ELEM_BYTES
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd4, %rd0, %rd3;").map_err(write_err)?;
    // Two 8-byte loads: lo at [addr+0], hi at [addr+8]. Read-only input -> .nc.
    writeln!(ptx, "\tld.global.nc.u64 %rd1, [%rd4];").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u64 %rd2, [%rd4+8];").map_err(write_err)?;
    writeln!(ptx, "\tbra DEC_AFTER_LOAD;").map_err(write_err)?;
    writeln!(ptx, "DEC_LOAD_IDENTITY:").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, 0;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, 0;").map_err(write_err)?;
    writeln!(ptx, "DEC_AFTER_LOAD:").map_err(write_err)?;

    // Stash (lo, hi) into the two shared arrays at slot tid.
    writeln!(ptx, "\tmul.wide.u32 %rd5, %r2, 8;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd6, sdata_lo;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd7, %rd6, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd7], %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd8, sdata_hi;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd9, %rd8, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd9], %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // Phase 1: inter-warp shared-memory tree at strides 128, 64, 32.
    // %rd7 / %rd9 are this thread's lo / hi shared slot addresses (kept live).
    let phase1_strides: [i32; 3] = [128, 64, 32];
    let mut step = 0usize;
    for &stride in &phase1_strides {
        let neighbor_pred = format!("%p{}", 1 + (step % 4));
        let stride_bytes = stride as usize * 8;
        writeln!(
            ptx,
            "\tsetp.lt.s32 {pred}, %r2, {stride};",
            pred = neighbor_pred,
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\t@!{pred} bra DEC_SKIP_ADD_{step};",
            pred = neighbor_pred,
            step = step
        )
        .map_err(write_err)?;
        // Load own (lo,hi) and neighbour (lo,hi).
        writeln!(ptx, "\tld.shared.u64 %rd10, [%rd7];").map_err(write_err)?; // own lo
        writeln!(ptx, "\tld.shared.u64 %rd11, [%rd9];").map_err(write_err)?; // own hi
        writeln!(
            ptx,
            "\tadd.s64 %rd12, %rd7, {offset};",
            offset = stride_bytes
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tadd.s64 %rd13, %rd9, {offset};",
            offset = stride_bytes
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tld.shared.u64 %rd14, [%rd12];").map_err(write_err)?; // neigh lo
        writeln!(ptx, "\tld.shared.u64 %rd15, [%rd13];").map_err(write_err)?; // neigh hi
                                                                              // 128-bit add: lo with carry-out, hi with carry-in.
        writeln!(ptx, "\tadd.cc.u64 %rd16, %rd10, %rd14;").map_err(write_err)?;
        writeln!(ptx, "\taddc.u64 %rd17, %rd11, %rd15;").map_err(write_err)?;
        // Store the combined value back into own slot.
        writeln!(ptx, "\tst.shared.u64 [%rd7], %rd16;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u64 [%rd9], %rd17;").map_err(write_err)?;
        writeln!(ptx, "DEC_SKIP_ADD_{step}:", step = step).map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        step += 1;
    }

    // Phase 2: warp-0-only intra-warp shuffle reduction (strides 16..1).
    // Gate on tid < 32 so all 32 lanes of warp 0 stay live for shfl.sync.
    writeln!(ptx, "\tsetp.ge.u32 %p6, %r2, 32;").map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra DEC_DONE;").map_err(write_err)?;

    // Load this lane's (lo, hi) into registers once; the rest is register-only.
    writeln!(ptx, "\tld.shared.u64 %rd20, [%rd7];").map_err(write_err)?; // lo
    writeln!(ptx, "\tld.shared.u64 %rd21, [%rd9];").map_err(write_err)?; // hi

    for &stride in &[16i32, 8, 4, 2, 1] {
        // Shuffle the lo half: split into two b32, shfl each, recombine.
        writeln!(ptx, "\tmov.b64 {{%r10, %r11}}, %rd20;").map_err(write_err)?;
        writeln!(
            ptx,
            "\tshfl.sync.down.b32 %r12, %r10, {stride}, 0x1f, 0xffffffff;",
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tshfl.sync.down.b32 %r13, %r11, {stride}, 0x1f, 0xffffffff;",
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tmov.b64 %rd22, {{%r12, %r13}};").map_err(write_err)?;
        // Shuffle the hi half.
        writeln!(ptx, "\tmov.b64 {{%r14, %r15}}, %rd21;").map_err(write_err)?;
        writeln!(
            ptx,
            "\tshfl.sync.down.b32 %r16, %r14, {stride}, 0x1f, 0xffffffff;",
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tshfl.sync.down.b32 %r17, %r15, {stride}, 0x1f, 0xffffffff;",
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tmov.b64 %rd23, {{%r16, %r17}};").map_err(write_err)?;
        // 128-bit carry add of the shuffled neighbour into the running value.
        writeln!(ptx, "\tadd.cc.u64 %rd20, %rd20, %rd22;").map_err(write_err)?;
        writeln!(ptx, "\taddc.u64 %rd21, %rd21, %rd23;").map_err(write_err)?;
    }

    // Only lane 0 of warp 0 writes the block's reduced i128 to global output.
    writeln!(ptx, "\tsetp.ne.s32 %p5, %r2, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra DEC_DONE;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd30, [{}_param_1];",
        DECIMAL_SUM_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd30, %rd30;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd31, %r0, {bytes};",
        bytes = DECIMAL_ELEM_BYTES
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd32, %rd30, %rd31;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u64 [%rd32], %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u64 [%rd32+8], %rd21;").map_err(write_err)?;

    writeln!(ptx, "DEC_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// Decimal128 MIN / MAX block-reduce kernels.
// ---------------------------------------------------------------------------

/// PTX entry-point name for the Decimal128 MIN reduction kernel.
pub const DECIMAL_MIN_KERNEL_ENTRY: &str = "bolt_decimal_min_reduce";

/// PTX entry-point name for the Decimal128 MAX reduction kernel.
pub const DECIMAL_MAX_KERNEL_ENTRY: &str = "bolt_decimal_max_reduce";

/// Which extremum a [`compile_decimal_minmax_kernel`] invocation emits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecimalMinMax {
    /// `MIN(Decimal128)` — 128-bit signed minimum.
    Min,
    /// `MAX(Decimal128)` — 128-bit signed maximum.
    Max,
}

impl DecimalMinMax {
    /// The kernel entry-point name for this extremum.
    pub fn entry(self) -> &'static str {
        match self {
            DecimalMinMax::Min => DECIMAL_MIN_KERNEL_ENTRY,
            DecimalMinMax::Max => DECIMAL_MAX_KERNEL_ENTRY,
        }
    }
}

/// Emit a 128-bit signed compare-and-select that combines a *neighbour* pair
/// (`n_lo`/`n_hi`) into a *running* pair (`o_lo`/`o_hi`), writing the chosen
/// (lo, hi) back into `dst_lo`/`dst_hi`.
///
/// The neighbour "wins" (replaces the running value) when:
///   - MIN: `neighbour < running`
///   - MAX: `neighbour > running`
///
/// 128-bit signed ordering is the standard high-half / low-half decomposition
/// (signed on the high half — bit 127 carries the sign — unsigned on the low
/// half, whose raw u64 ordering IS the within-equal-high-half ordering). This
/// is the *exact* rule [`crate::jit::ptx_gen::emit_cmp_128`] uses, so a scalar
/// `Cmp128` and this reduction agree bit-for-bit on the i128 ordering.
///
/// `selp.b64` performs the predicated pick on each half: when `%p7` is true the
/// neighbour half is selected, otherwise the running half is kept. Because the
/// underlying raw `i128` ordering equals the decimal ordering at the column's
/// uniform scale (every row shares the same scale), this is a correct decimal
/// MIN/MAX — matching the host fold in
/// `crate::exec::aggregate::minmax_decimal128_from_batch`.
///
/// `pred_reg` is a `%p<..>` name reserved by the caller; `n_*`/`o_*`/`dst_*`
/// are `%rd<..>` names. `eq_reg` is a second predicate name used for the
/// equal-high-half sub-test.
fn emit_dec_minmax_combine(
    ptx: &mut String,
    which: DecimalMinMax,
    pred_reg: &str,
    eq_reg: &str,
    n_lo: &str,
    n_hi: &str,
    o_lo: &str,
    o_hi: &str,
    dst_lo: &str,
    dst_hi: &str,
) -> BoltResult<()> {
    // Choose the high/low comparison sense.
    let (hi_cmp, lo_cmp) = match which {
        // neighbour < running
        DecimalMinMax::Min => ("setp.lt.s64", "setp.lt.u64"),
        // neighbour > running
        DecimalMinMax::Max => ("setp.gt.s64", "setp.gt.u64"),
    };
    // The neighbour wins iff  (n_hi <cmp> o_hi) || (n_hi == o_hi && n_lo <cmp_u> o_lo).
    // Three predicates are consumed: `pred_reg` (the high-half / final result),
    // `eq_reg` (the equal-high-half flag, then reused for the AND term), and the
    // fixed scratch `%p7` (the low-half compare). All callers below reserve
    // `.reg .pred %p<8>;` so `%p7` is always available; `pred_reg`/`eq_reg` are
    // distinct `%p5`/`%p6` so nothing aliases.
    //
    //   pred = (n_hi <cmp> o_hi)
    writeln!(ptx, "\t{} {}, {}, {};", hi_cmp, pred_reg, n_hi, o_hi).map_err(write_err)?;
    //   eq   = (n_hi == o_hi)
    writeln!(ptx, "\tsetp.eq.s64 {}, {}, {};", eq_reg, n_hi, o_hi).map_err(write_err)?;
    //   %p7  = (n_lo <cmp_u> o_lo)
    writeln!(ptx, "\t{} %p7, {}, {};", lo_cmp, n_lo, o_lo).map_err(write_err)?;
    //   eq   = eq && %p7
    writeln!(ptx, "\tand.pred {}, {}, %p7;", eq_reg, eq_reg).map_err(write_err)?;
    //   pred = pred || eq
    writeln!(ptx, "\tor.pred {}, {}, {};", pred_reg, pred_reg, eq_reg).map_err(write_err)?;
    // Select each half: neighbour when pred, running otherwise.
    writeln!(
        ptx,
        "\tselp.b64 {}, {}, {}, {};",
        dst_lo, n_lo, o_lo, pred_reg
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tselp.b64 {}, {}, {}, {};",
        dst_hi, n_hi, o_hi, pred_reg
    )
    .map_err(write_err)?;
    Ok(())
}

/// Generate PTX for the per-block `MIN(Decimal128)` / `MAX(Decimal128)`
/// reduction kernel.
///
/// Structurally identical to [`compile_decimal_sum_kernel`] (atomic-free
/// two-stage block reduce over the `(lo, hi)` `i128` pair: a shared-memory
/// tree at strides 128/64/32, then a warp-0 `shfl.sync.down.b32` reduction at
/// strides 16..1), except the per-step *combine* is a 128-bit signed
/// compare-and-select ([`emit_dec_minmax_combine`]) rather than a carry-chain
/// add. Each block writes one `i128` partial to `output[blockIdx.x]`; the host
/// folds the per-block partials with the same min/max ordering.
///
/// ## Identity / NULL handling
///
/// Unlike SUM (whose identity 0 is harmless to fold in), MIN/MAX have no
/// in-band identity that is safe to materialise for out-of-range / NULL slots:
/// seeding `i128::MAX` (MIN) or `i128::MIN` (MAX) would be correct in theory
/// but would also become a *real* candidate if it equalled an actual data
/// value. Instead the kernel mirrors the SUM kernel's contract: the host
/// strips NULLs into a dense `Vec<i128>` before upload, and the out-of-range
/// tail threads of the final (partial) block load the identity extremum
/// (`i128::MAX` for MIN, `i128::MIN` for MAX) — which can never beat a real
/// value because the dense survivors are all genuine column values and the
/// final host fold seeds from `None` (see
/// `crate::exec::aggregate::minmax_decimal128_from_batch`). The GPU partials
/// are therefore always dominated by real values; the identity is purely a
/// "do not win" sentinel for padding lanes.
pub fn compile_decimal_minmax_kernel(which: DecimalMinMax) -> BoltResult<String> {
    let entry = which.entry();
    let half_shared_bytes = BLOCK_SIZE as usize * 8;

    // Out-of-range padding identity (lo, hi) as raw u64 halves of the i128
    // sentinel that can never beat a real value:
    //   MIN -> i128::MAX  (lo = u64::MAX, hi = 0x7FFF_FFFF_FFFF_FFFF)
    //   MAX -> i128::MIN  (lo = 0,        hi = 0x8000_0000_0000_0000)
    let (id_lo, id_hi): (u64, u64) = match which {
        DecimalMinMax::Min => {
            let v = i128::MAX as u128;
            (v as u64, (v >> 64) as u64)
        }
        DecimalMinMax::Max => {
            let v = i128::MIN as u128;
            (v as u64, (v >> 64) as u64)
        }
    };

    let mut ptx = String::new();
    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 8 .b8 sdata_lo[{n}];",
        n = half_shared_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 8 .b8 sdata_hi[{n}];",
        n = half_shared_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_2", entry).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // %p7 is reserved by `emit_dec_minmax_combine` as the low-half scratch
    // predicate; %p0..%p6 mirror the SUM kernel's gating predicates.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<48>;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // gid = ctaid.x * ntid.x + tid.x
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{}_param_2];", entry).map_err(write_err)?;

    // Load this thread's i128 value (lo into %rd1, hi into %rd2), or the
    // padding identity if past the end.
    writeln!(ptx, "\tld.param.u64 %rd0, [{}_param_0];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DEC_LOAD_IDENTITY;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd3, %r3, {bytes};",
        bytes = DECIMAL_ELEM_BYTES
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd4, %rd0, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u64 %rd1, [%rd4];").map_err(write_err)?;
    writeln!(ptx, "\tld.global.nc.u64 %rd2, [%rd4+8];").map_err(write_err)?;
    writeln!(ptx, "\tbra DEC_AFTER_LOAD;").map_err(write_err)?;
    writeln!(ptx, "DEC_LOAD_IDENTITY:").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd1, {};", id_lo).map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd2, {};", id_hi).map_err(write_err)?;
    writeln!(ptx, "DEC_AFTER_LOAD:").map_err(write_err)?;

    // Stash (lo, hi) into the two shared arrays at slot tid.
    writeln!(ptx, "\tmul.wide.u32 %rd5, %r2, 8;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd6, sdata_lo;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd7, %rd6, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd7], %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd8, sdata_hi;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd9, %rd8, %rd5;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd9], %rd2;").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // Phase 1: inter-warp shared-memory tree at strides 128, 64, 32.
    let phase1_strides: [i32; 3] = [128, 64, 32];
    let mut step = 0usize;
    for &stride in &phase1_strides {
        let neighbor_pred = format!("%p{}", 1 + (step % 4));
        let stride_bytes = stride as usize * 8;
        writeln!(
            ptx,
            "\tsetp.lt.s32 {pred}, %r2, {stride};",
            pred = neighbor_pred,
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\t@!{pred} bra DEC_SKIP_ADD_{step};",
            pred = neighbor_pred,
            step = step
        )
        .map_err(write_err)?;
        // Load own (lo,hi) and neighbour (lo,hi).
        writeln!(ptx, "\tld.shared.u64 %rd10, [%rd7];").map_err(write_err)?; // own lo
        writeln!(ptx, "\tld.shared.u64 %rd11, [%rd9];").map_err(write_err)?; // own hi
        writeln!(
            ptx,
            "\tadd.s64 %rd12, %rd7, {offset};",
            offset = stride_bytes
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tadd.s64 %rd13, %rd9, {offset};",
            offset = stride_bytes
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tld.shared.u64 %rd14, [%rd12];").map_err(write_err)?; // neigh lo
        writeln!(ptx, "\tld.shared.u64 %rd15, [%rd13];").map_err(write_err)?; // neigh hi
                                                                              // 128-bit compare-and-select: combine neighbour into own → %rd16/%rd17.
        emit_dec_minmax_combine(
            &mut ptx, which, "%p5", "%p6", "%rd14", "%rd15", "%rd10", "%rd11", "%rd16", "%rd17",
        )?;
        // Store the chosen value back into own slot.
        writeln!(ptx, "\tst.shared.u64 [%rd7], %rd16;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u64 [%rd9], %rd17;").map_err(write_err)?;
        writeln!(ptx, "DEC_SKIP_ADD_{step}:", step = step).map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        step += 1;
    }

    // Phase 2: warp-0-only intra-warp shuffle reduction (strides 16..1).
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r2, 32;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra DEC_DONE;").map_err(write_err)?;

    // Load this lane's (lo, hi) into registers once; the rest is register-only.
    writeln!(ptx, "\tld.shared.u64 %rd20, [%rd7];").map_err(write_err)?; // lo
    writeln!(ptx, "\tld.shared.u64 %rd21, [%rd9];").map_err(write_err)?; // hi

    for &stride in &[16i32, 8, 4, 2, 1] {
        // Shuffle the lo half: split into two b32, shfl each, recombine.
        writeln!(ptx, "\tmov.b64 {{%r10, %r11}}, %rd20;").map_err(write_err)?;
        writeln!(
            ptx,
            "\tshfl.sync.down.b32 %r12, %r10, {stride}, 0x1f, 0xffffffff;",
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tshfl.sync.down.b32 %r13, %r11, {stride}, 0x1f, 0xffffffff;",
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tmov.b64 %rd22, {{%r12, %r13}};").map_err(write_err)?;
        // Shuffle the hi half.
        writeln!(ptx, "\tmov.b64 {{%r14, %r15}}, %rd21;").map_err(write_err)?;
        writeln!(
            ptx,
            "\tshfl.sync.down.b32 %r16, %r14, {stride}, 0x1f, 0xffffffff;",
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tshfl.sync.down.b32 %r17, %r15, {stride}, 0x1f, 0xffffffff;",
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tmov.b64 %rd23, {{%r16, %r17}};").map_err(write_err)?;
        // 128-bit compare-and-select of the shuffled neighbour into the running
        // value (%rd20/%rd21). Result written back in place.
        emit_dec_minmax_combine(
            &mut ptx, which, "%p5", "%p6", "%rd22", "%rd23", "%rd20", "%rd21", "%rd20", "%rd21",
        )?;
    }

    // Only lane 0 of warp 0 writes the block's reduced i128 to global output.
    writeln!(ptx, "\tsetp.ne.s32 %p5, %r2, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra DEC_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u64 %rd30, [{}_param_1];", entry).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd30, %rd30;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd31, %r0, {bytes};",
        bytes = DECIMAL_ELEM_BYTES
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd32, %rd30, %rd31;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u64 [%rd32], %rd20;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u64 [%rd32+8], %rd21;").map_err(write_err)?;

    writeln!(ptx, "DEC_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernel emits a single entry point with the documented three-param
    /// ABI (input, output, n_rows).
    #[test]
    fn emits_entry_with_three_params() {
        let ptx = compile_decimal_sum_kernel().expect("decimal SUM PTX should compile");
        assert!(
            ptx.contains(&format!(".visible .entry {}(", DECIMAL_SUM_KERNEL_ENTRY)),
            "PTX missing entry point"
        );
        assert!(ptx.contains("_param_0"), "missing input param");
        assert!(ptx.contains("_param_1"), "missing output param");
        assert!(ptx.contains("_param_2"), "missing n_rows param");
    }

    /// The 128-bit add MUST use the carry-chain pair: `add.cc.u64` produces
    /// the low-half carry-out and `addc.u64` consumes it for the high half.
    /// A regression that drops either instruction silently corrupts sums that
    /// cross the 2^64 boundary.
    #[test]
    fn uses_carry_chain_add() {
        let ptx = compile_decimal_sum_kernel().unwrap();
        assert!(
            ptx.contains("add.cc.u64"),
            "128-bit add must emit add.cc.u64 (carry-out)"
        );
        assert!(
            ptx.contains("addc.u64"),
            "128-bit add must emit addc.u64 (carry-in)"
        );
    }

    /// The kernel loads / stores the i128 as two 8-byte halves with the hi
    /// half at the `+8` byte offset.
    #[test]
    fn loads_and_stores_two_halves() {
        let ptx = compile_decimal_sum_kernel().unwrap();
        assert!(
            ptx.contains("ld.global.nc.u64 %rd1, [%rd4];"),
            "lo load missing"
        );
        assert!(
            ptx.contains("ld.global.nc.u64 %rd2, [%rd4+8];"),
            "hi load (at +8) missing"
        );
        assert!(
            ptx.contains("st.global.u64 [%rd32], %rd20;"),
            "lo store missing"
        );
        assert!(
            ptx.contains("st.global.u64 [%rd32+8], %rd21;"),
            "hi store (at +8) missing"
        );
    }

    /// No atomic instruction may appear — the design is explicitly atomic-free
    /// (two-stage block reduce + host fold).
    #[test]
    fn emits_no_atomics() {
        let ptx = compile_decimal_sum_kernel().unwrap();
        assert!(
            !ptx.contains("atom."),
            "decimal SUM kernel must be atomic-free; found an atom.* instruction"
        );
        // Anchor on the instruction boundary (every instruction is emitted
        // tab-indented) so this does not false-match the `red.` substring inside
        // `st.shared.`/`ld.shared.` memory ops, which are not reduction atomics.
        assert!(
            !ptx.contains("\tred.") && !ptx.contains("red.global") && !ptx.contains("red.shared"),
            "must not use red.* reduction atomics"
        );
    }

    /// Phase-1 tree (bar.sync) + phase-2 warp shuffle must both be present.
    #[test]
    fn emits_two_phase_reduction() {
        let ptx = compile_decimal_sum_kernel().unwrap();
        assert!(ptx.contains("bar.sync 0;"), "phase-1 needs bar.sync");
        assert!(
            ptx.contains("shfl.sync.down.b32"),
            "phase-2 needs warp-shuffle"
        );
    }

    // -- MIN / MAX kernel PTX-shape tests --------------------------------

    /// Both extrema emit their own entry point with the 3-param ABI.
    #[test]
    fn minmax_emits_entry_with_three_params() {
        for which in [DecimalMinMax::Min, DecimalMinMax::Max] {
            let ptx = compile_decimal_minmax_kernel(which).expect("MIN/MAX PTX should compile");
            assert!(
                ptx.contains(&format!(".visible .entry {}(", which.entry())),
                "PTX missing entry point for {which:?}:\n{ptx}"
            );
            assert!(ptx.contains("_param_0"), "missing input param ({which:?})");
            assert!(ptx.contains("_param_1"), "missing output param ({which:?})");
            assert!(ptx.contains("_param_2"), "missing n_rows param ({which:?})");
        }
    }

    /// The combine MUST be a 128-bit signed compare-and-select, NOT an add:
    /// signed high-half compare + unsigned low-half compare + `selp.b64` on
    /// each half. A regression that emitted `add.cc.u64`/`addc.u64` would
    /// silently turn MIN/MAX into SUM.
    #[test]
    fn minmax_uses_compare_and_select_not_add() {
        let min = compile_decimal_minmax_kernel(DecimalMinMax::Min).unwrap();
        // MIN: neighbour wins when strictly less → lt on both halves.
        assert!(
            min.contains("setp.lt.s64"),
            "MIN hi-half signed-lt missing:\n{min}"
        );
        assert!(
            min.contains("setp.lt.u64"),
            "MIN lo-half unsigned-lt missing:\n{min}"
        );
        assert!(
            min.contains("selp.b64"),
            "MIN must select via selp.b64:\n{min}"
        );
        assert!(
            !min.contains("add.cc.u64") && !min.contains("addc.u64"),
            "MIN must NOT emit carry-chain add (that is SUM):\n{min}"
        );

        let max = compile_decimal_minmax_kernel(DecimalMinMax::Max).unwrap();
        // MAX: neighbour wins when strictly greater → gt on both halves.
        assert!(
            max.contains("setp.gt.s64"),
            "MAX hi-half signed-gt missing:\n{max}"
        );
        assert!(
            max.contains("setp.gt.u64"),
            "MAX lo-half unsigned-gt missing:\n{max}"
        );
        assert!(
            max.contains("selp.b64"),
            "MAX must select via selp.b64:\n{max}"
        );
        assert!(
            !max.contains("add.cc.u64") && !max.contains("addc.u64"),
            "MAX must NOT emit carry-chain add:\n{max}"
        );
    }

    /// Out-of-range padding lanes load the "can never win" identity extremum:
    /// `i128::MAX` for MIN and `i128::MIN` for MAX, carried as the two u64
    /// halves of the i128.
    #[test]
    fn minmax_padding_identity_is_dominated_sentinel() {
        let min = compile_decimal_minmax_kernel(DecimalMinMax::Min).unwrap();
        let max_v = i128::MAX as u128;
        assert!(
            min.contains(&format!("mov.u64 %rd1, {};", max_v as u64)),
            "MIN padding lo half must be i128::MAX lo:\n{min}"
        );
        assert!(
            min.contains(&format!("mov.u64 %rd2, {};", (max_v >> 64) as u64)),
            "MIN padding hi half must be i128::MAX hi:\n{min}"
        );

        let max = compile_decimal_minmax_kernel(DecimalMinMax::Max).unwrap();
        let min_v = i128::MIN as u128;
        assert!(
            max.contains(&format!("mov.u64 %rd1, {};", min_v as u64)),
            "MAX padding lo half must be i128::MIN lo:\n{max}"
        );
        assert!(
            max.contains(&format!("mov.u64 %rd2, {};", (min_v >> 64) as u64)),
            "MAX padding hi half must be i128::MIN hi:\n{max}"
        );
    }

    /// MIN/MAX kernels are atomic-free (same two-stage block-reduce + host
    /// fold contract as SUM).
    #[test]
    fn minmax_emits_no_atomics() {
        for which in [DecimalMinMax::Min, DecimalMinMax::Max] {
            let ptx = compile_decimal_minmax_kernel(which).unwrap();
            assert!(
                !ptx.contains("atom."),
                "MIN/MAX kernel must be atomic-free ({which:?}):\n{ptx}"
            );
        }
    }

    /// MIN/MAX load + store the i128 as two 8-byte halves (hi at `+8`), exactly
    /// like SUM, so the host i128 buffer round-trips bit-for-bit.
    #[test]
    fn minmax_loads_and_stores_two_halves() {
        let ptx = compile_decimal_minmax_kernel(DecimalMinMax::Min).unwrap();
        assert!(
            ptx.contains("ld.global.nc.u64 %rd1, [%rd4];"),
            "lo load missing:\n{ptx}"
        );
        assert!(
            ptx.contains("ld.global.nc.u64 %rd2, [%rd4+8];"),
            "hi load (+8) missing:\n{ptx}"
        );
        assert!(
            ptx.contains("st.global.u64 [%rd32], %rd20;"),
            "lo store missing:\n{ptx}"
        );
        assert!(
            ptx.contains("st.global.u64 [%rd32+8], %rd21;"),
            "hi store (+8) missing:\n{ptx}"
        );
    }

    /// Two-phase shape (bar.sync tree + warp shuffle) carries over to MIN/MAX.
    #[test]
    fn minmax_emits_two_phase_reduction() {
        let ptx = compile_decimal_minmax_kernel(DecimalMinMax::Max).unwrap();
        assert!(
            ptx.contains("bar.sync 0;"),
            "phase-1 needs bar.sync:\n{ptx}"
        );
        assert!(
            ptx.contains("shfl.sync.down.b32"),
            "phase-2 needs warp-shuffle:\n{ptx}"
        );
    }

    /// MIN and MAX must produce DISTINCT entry points (the host launcher picks
    /// the symbol by name) and structurally different combine senses.
    #[test]
    fn min_and_max_are_distinct() {
        let min = compile_decimal_minmax_kernel(DecimalMinMax::Min).unwrap();
        let max = compile_decimal_minmax_kernel(DecimalMinMax::Max).unwrap();
        assert_ne!(DECIMAL_MIN_KERNEL_ENTRY, DECIMAL_MAX_KERNEL_ENTRY);
        assert!(min.contains(DECIMAL_MIN_KERNEL_ENTRY));
        assert!(max.contains(DECIMAL_MAX_KERNEL_ENTRY));
        assert_ne!(min, max, "MIN and MAX PTX must differ (compare sense)");
    }
}
