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
        assert!(ptx.contains("ld.global.nc.u64 %rd1, [%rd4];"), "lo load missing");
        assert!(
            ptx.contains("ld.global.nc.u64 %rd2, [%rd4+8];"),
            "hi load (at +8) missing"
        );
        assert!(ptx.contains("st.global.u64 [%rd32], %rd20;"), "lo store missing");
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
}
