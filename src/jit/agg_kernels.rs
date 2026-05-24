// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for scalar reduction kernels.
//!
//! Each kernel performs a per-block reduction over a single primitive input
//! column and writes one partial result per block to `output[blockIdx.x]`.
//! The final cross-block reduction is performed on the host (see
//! `crate::exec::aggregate`).
//!
//! This avoids the complexity of `atom.global.*` instructions (especially
//! float min/max, which require a CAS loop on sm_70) at the cost of one extra
//! d2h copy + a tiny host-side reduction. For now that trade is fine.
//!
//! ABI of every emitted kernel:
//! ```text
//! .visible .entry javelin_reduce(
//!     .param .u64 input_ptr,
//!     .param .u64 output_ptr,
//!     .param .u32 n_rows
//! )
//! ```
//! Block size is hard-coded to `BLOCK_SIZE` (256). The grid is sized by the
//! launcher to `ceil(n_rows / BLOCK_SIZE)`.

use std::fmt::Write;

use crate::error::{JavelinError, JavelinResult};
use crate::plan::logical_plan::{sum_output_dtype, AggregateExpr, DataType};

/// Public helper: the accumulator dtype the reduction kernel emitted by
/// `compile_reduction_kernel` will write to for `(op, input_dtype)`.
///
/// For `Sum` over a narrow signed integer this is the widened (Int64)
/// accumulator dtype; for every other case it is the input dtype unchanged.
/// Callers (`crate::exec::aggregate`) use this to size the partial-output
/// buffer and to pick the matching Arrow array type. This must agree with
/// `crate::plan::logical_plan::sum_output_dtype` — that helper is the single
/// source of truth for the widening rule.
pub fn reduction_output_dtype(op: ReduceOp, input_dtype: DataType) -> DataType {
    match op {
        ReduceOp::Sum | ReduceOp::Count => sum_output_dtype(input_dtype),
        ReduceOp::Min | ReduceOp::Max => input_dtype,
    }
}

/// Threads per block for every reduction kernel.
pub const BLOCK_SIZE: u32 = 256;

/// PTX kernel entry-point name.
pub const REDUCTION_KERNEL_ENTRY: &str = "javelin_reduce";

/// Reduction operator the codegen needs to emit identity + combine for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// `SUM` — identity 0, combine `add`.
    Sum,
    /// `MIN` — identity +inf / INT_MAX, combine `min`.
    Min,
    /// `MAX` — identity -inf / INT_MIN, combine `max`.
    Max,
    /// `COUNT` — handled at a higher level (synthesized as `Sum` over ones).
    Count,
}

impl ReduceOp {
    /// Map an `AggregateExpr` to its underlying reduction op. `Avg` is not a
    /// single reduction and must be decomposed (sum + count) by the caller.
    pub fn from_agg(agg: &AggregateExpr) -> JavelinResult<Self> {
        match agg {
            AggregateExpr::Sum(_) => Ok(ReduceOp::Sum),
            AggregateExpr::Min(_) => Ok(ReduceOp::Min),
            AggregateExpr::Max(_) => Ok(ReduceOp::Max),
            AggregateExpr::Count(_) => Ok(ReduceOp::Count),
            AggregateExpr::Avg(_) => Err(JavelinError::Other(
                "agg_kernels: AVG must be decomposed into Sum + Count by the caller".into(),
            )),
        }
    }

    /// PTX literal expression for the identity value of `self` at `dtype`.
    pub fn identity_ptx(self, dtype: DataType) -> JavelinResult<String> {
        use DataType::*;
        use ReduceOp::*;
        match (self, dtype) {
            // Additive identities.
            (Sum, Int32) | (Count, Int32) => Ok("0".to_string()),
            (Sum, Int64) | (Count, Int64) => Ok("0".to_string()),
            (Sum, Float32) | (Count, Float32) => Ok(format!("0f{:08X}", 0_f32.to_bits())),
            (Sum, Float64) | (Count, Float64) => Ok(format!("0d{:016X}", 0_f64.to_bits())),

            // MIN identities — the largest representable value of dtype.
            (Min, Int32) => Ok(i32::MAX.to_string()),
            (Min, Int64) => Ok(i64::MAX.to_string()),
            (Min, Float32) => Ok(format!("0f{:08X}", f32::INFINITY.to_bits())),
            (Min, Float64) => Ok(format!("0d{:016X}", f64::INFINITY.to_bits())),

            // MAX identities — the smallest representable value of dtype.
            (Max, Int32) => Ok(i32::MIN.to_string()),
            (Max, Int64) => Ok(i64::MIN.to_string()),
            (Max, Float32) => Ok(format!("0f{:08X}", f32::NEG_INFINITY.to_bits())),
            (Max, Float64) => Ok(format!("0d{:016X}", f64::NEG_INFINITY.to_bits())),

            (_, Bool) | (_, Utf8) => Err(JavelinError::Type(format!(
                "agg_kernels: reduction over dtype {:?} is not supported",
                dtype
            ))),
        }
    }

    /// PTX combine instruction mnemonic for `self` at `dtype`.
    pub fn combine_ptx(self, dtype: DataType) -> JavelinResult<String> {
        use DataType::*;
        use ReduceOp::*;
        let s = match (self, dtype) {
            (Sum, Int32) | (Count, Int32) => "add.s32",
            (Sum, Int64) | (Count, Int64) => "add.s64",
            (Sum, Float32) | (Count, Float32) => "add.f32",
            (Sum, Float64) | (Count, Float64) => "add.f64",

            (Min, Int32) => "min.s32",
            (Min, Int64) => "min.s64",
            (Min, Float32) => "min.f32",
            (Min, Float64) => "min.f64",

            (Max, Int32) => "max.s32",
            (Max, Int64) => "max.s64",
            (Max, Float32) => "max.f32",
            (Max, Float64) => "max.f64",

            (_, Bool) | (_, Utf8) => {
                return Err(JavelinError::Type(format!(
                    "agg_kernels: reduction over dtype {:?} is not supported",
                    dtype
                )))
            }
        };
        Ok(s.to_string())
    }
}

/// Generate PTX for a per-block reduction kernel. Each block computes one
/// partial result and writes it to `output[blockIdx.x]`. The final reduction
/// across block partials is performed on the host.
///
/// `dtype` is the *input* column dtype. The kernel's accumulator (and hence
/// the output buffer element type) is `reduction_output_dtype(op, dtype)` —
/// for `SUM` over a narrow signed integer this is wider than `dtype` and the
/// kernel sign-extends each loaded value before accumulating. See
/// `crate::plan::logical_plan::sum_output_dtype` for the widening contract.
pub fn compile_reduction_kernel(op: ReduceOp, dtype: DataType) -> JavelinResult<String> {
    // Input-side PTX info governs the global load from the source column.
    let (input_load_suffix, _input_store_suffix, input_reg_class, input_reg_ty, _input_imm_ty) =
        ptx_type_info(dtype)?;
    // Accumulator-side PTX info governs identity, combine, shared-memory
    // storage, and the global output store. For SUM(Int32) these diverge:
    // input is `s32`, accumulator is `s64`.
    let acc_dtype = reduction_output_dtype(op, dtype);
    let (acc_load_suffix, acc_store_suffix, acc_reg_class, acc_reg_ty, acc_imm_ty) =
        ptx_type_info(acc_dtype)?;
    let widens = acc_dtype != dtype;

    // Identity and combine are computed against the ACCUMULATOR dtype: the
    // tree reduction in shared memory operates on widened values.
    let identity = op.identity_ptx(acc_dtype)?;
    let combine = op.combine_ptx(acc_dtype)?;

    let input_elem_bytes = dtype.byte_width().ok_or_else(|| {
        JavelinError::Other(format!(
            "agg_kernels: variable-width dtype {:?} not supported",
            dtype
        ))
    })?;
    let acc_elem_bytes = acc_dtype.byte_width().ok_or_else(|| {
        JavelinError::Other(format!(
            "agg_kernels: variable-width accumulator dtype {:?} not supported",
            acc_dtype
        ))
    })?;
    // Shared memory and per-block output are sized by the accumulator dtype.
    let elem_bytes = acc_elem_bytes;
    let shared_bytes = BLOCK_SIZE as usize * elem_bytes;

    let mut ptx = String::new();
    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Shared memory scratchpad. One element per thread, sized for the largest
    // primitive in the kernel.
    writeln!(
        ptx,
        ".shared .align 8 .b8 sdata[{shared}];",
        shared = shared_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", REDUCTION_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", REDUCTION_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", REDUCTION_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_2", REDUCTION_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register declarations. We use generous counts because PTX `.reg`
    // declarations only allocate names, not real hardware registers.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .{}   %{}<8>;", acc_reg_ty, acc_reg_class).map_err(write_err)?;
    // When the kernel widens (e.g. SUM(Int32) -> Int64), allocate a separate
    // register class for the raw input load that is then sign-extended into
    // the accumulator register. Skip when input and accumulator dtype agree
    // to avoid emitting a duplicate `.reg` declaration of the same class.
    if widens {
        writeln!(ptx, "\t.reg .{}   %{}<4>;", input_reg_ty, input_reg_class)
            .map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // Compute global thread index: gid = ctaid.x * ntid.x + tid.x
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u32 %r4, [{}_param_2];",
        REDUCTION_KERNEL_ENTRY
    )
    .map_err(write_err)?;

    // Load this thread's value (or the identity if it's past the end).
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        REDUCTION_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.s32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra LOAD_IDENTITY;").map_err(write_err)?;
    // Address arithmetic for the source-column load uses the *input* element
    // size (the source array is laid out at input dtype, not accumulator).
    writeln!(
        ptx,
        "\tmul.wide.s32 %rd1, %r3, {bytes};",
        bytes = input_elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    if widens {
        // Load the narrow value into an input-class register, then
        // sign-extend into the accumulator-class register. The acc class is
        // always wider (currently only s64 from s32), so `cvt.<acc>.<in>`
        // is the right widening cvt mnemonic.
        writeln!(
            ptx,
            "\tld.global.{} %{}0, [%rd2];",
            input_load_suffix, input_reg_class
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tcvt.{acc}.{inp} %{arc}0, %{irc}0;",
            acc = acc_imm_ty,
            inp = input_load_suffix,
            arc = acc_reg_class,
            irc = input_reg_class
        )
        .map_err(write_err)?;
    } else {
        writeln!(
            ptx,
            "\tld.global.{} %{}0, [%rd2];",
            acc_load_suffix, acc_reg_class
        )
        .map_err(write_err)?;
    }
    writeln!(ptx, "\tbra AFTER_LOAD;").map_err(write_err)?;
    writeln!(ptx, "LOAD_IDENTITY:").map_err(write_err)?;
    // Identity is in accumulator dtype.
    writeln!(ptx, "\tmov.{} %{}0, {};", acc_imm_ty, acc_reg_class, identity)
        .map_err(write_err)?;
    writeln!(ptx, "AFTER_LOAD:").map_err(write_err)?;

    // Stash the value into shared memory. Shared memory is sized and indexed
    // by the accumulator element size, not the input element size.
    writeln!(
        ptx,
        "\tmul.wide.s32 %rd3, %r2, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    // NOTE: keep `mov.u64` here (not `cvta.shared.u64`). The result %rd4 is
    // used downstream with `st.shared.*` / `ld.shared.*` instructions, which
    // expect a shared-state-space address. `cvta.shared.u64` would produce a
    // generic-space address and require a matching state-space change at every
    // shared load/store below — a wider refactor than this cleanup intends.
    writeln!(ptx, "\tmov.u64 %rd4, sdata;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd4, %rd3;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tst.shared.{} [%rd5], %{}0;",
        acc_store_suffix, acc_reg_class
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // Two-phase tree reduction: bar.sync for strides > 32, warp shuffle for
    // strides <= 32.
    //
    // Phase 1 (inter-warp, strides 128, 64, 32): standard shared-memory tree
    //   with `bar.sync` between halvings and a `tid < stride` short-circuit.
    //   These strides span multiple warps so a block-wide barrier is required.
    //
    // Phase 2 (intra-warp, strides 16, 8, 4, 2, 1): warp 0 only. Each lane
    //   loads its own value from shared memory into a register *once*, then we
    //   use `shfl.sync.down` (membermask 0xffffffff, all 32 lanes participate)
    //   to halve down to lane 0 with no further bar.sync or shared traffic.
    //
    //   For 32-bit accumulators (Int32/Float32) we shuffle directly with
    //   `shfl.sync.down.b32` (going via a `%r` scratch register and `mov.b32`
    //   for float dtypes since shfl operates on `.b32`).
    //
    //   For 64-bit accumulators (Int64/Float64) sm_70 has no native
    //   `shfl.sync.down.b64`, so we split the value into two `.b32` halves with
    //   `mov.b64 {lo,hi}, %val;`, shuffle each half, recombine with
    //   `mov.b64 %tmp, {lo,hi};`, and apply the combine op on the 64-bit
    //   recombined value.
    //
    //   Phase 2 saves 5 bar.syncs and 10 shared loads/stores per block versus
    //   the all-bar.sync tree.
    //
    // Strides 128, 64, 32: inter-warp tree reduction (phase 1).
    let phase1_strides: [i32; 3] = [128, 64, 32];
    let mut step = 0usize;
    for &stride in &phase1_strides {
        let neighbor_pred = format!("%p{}", 1 + (step % 4));
        let stride_bytes = stride as usize * elem_bytes;
        writeln!(
            ptx,
            "\tsetp.lt.s32 {pred}, %r2, {stride};",
            pred = neighbor_pred,
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\t@!{pred} bra SKIP_ADD_{step};",
            pred = neighbor_pred,
            step = step
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tadd.s64 %rd6, %rd5, {offset};",
            offset = stride_bytes
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tld.shared.{ld} %{rc}1, [%rd6];",
            ld = acc_load_suffix,
            rc = acc_reg_class
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tld.shared.{ld} %{rc}2, [%rd5];",
            ld = acc_load_suffix,
            rc = acc_reg_class
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\t{combine} %{rc}3, %{rc}2, %{rc}1;",
            combine = combine,
            rc = acc_reg_class
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tst.shared.{st} [%rd5], %{rc}3;",
            st = acc_store_suffix,
            rc = acc_reg_class
        )
        .map_err(write_err)?;
        writeln!(ptx, "SKIP_ADD_{step}:", step = step).map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        step += 1;
    }

    // Strides 16, 8, 4, 2, 1: intra-warp shuffle reduction (phase 2).
    //
    // Gate on tid < 32: only warp 0 participates, but all 32 lanes of warp 0
    // must stay live for `shfl.sync` (membermask 0xffffffff). Threads with
    // tid >= 32 jump straight to DONE — they have no value to contribute and
    // sdata[0..32] already holds warp 0's working set from phase 1.
    writeln!(ptx, "\tsetp.ge.s32 %p6, %r2, 32;").map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra DONE;").map_err(write_err)?;

    // Load this lane's value (sdata[tid]) into a register once. After this
    // point the warp-0 reduction is register-only.
    writeln!(
        ptx,
        "\tld.shared.{ld} %{rc}5, [%rd5];",
        ld = acc_load_suffix,
        rc = acc_reg_class
    )
    .map_err(write_err)?;

    for &stride in &[16i32, 8, 4, 2, 1] {
        match acc_dtype {
            DataType::Int32 => {
                // For Int32 the acc register class is "r" (b32), so %r5 is
                // already a valid `shfl.sync.down.b32` source — no mov needed.
                // Working value lives in %r5; shfl scratch is %r6.
                writeln!(
                    ptx,
                    "\tshfl.sync.down.b32 %r6, %r5, {stride}, 0x1f, 0xffffffff;",
                    stride = stride
                )
                .map_err(write_err)?;
                writeln!(ptx, "\t{combine} %r5, %r5, %r6;", combine = combine)
                    .map_err(write_err)?;
            }
            DataType::Float32 => {
                // Working value lives in %f5; bridge through %r6/%r7 for shfl.
                writeln!(ptx, "\tmov.b32 %r6, %f5;").map_err(write_err)?;
                writeln!(
                    ptx,
                    "\tshfl.sync.down.b32 %r7, %r6, {stride}, 0x1f, 0xffffffff;",
                    stride = stride
                )
                .map_err(write_err)?;
                writeln!(ptx, "\tmov.b32 %f6, %r7;").map_err(write_err)?;
                writeln!(ptx, "\t{combine} %f5, %f5, %f6;", combine = combine)
                    .map_err(write_err)?;
            }
            DataType::Int64 => {
                // Split %rl5 into two b32 halves, shuffle each, recombine.
                writeln!(ptx, "\tmov.b64 {{%r8, %r9}}, %rl5;").map_err(write_err)?;
                writeln!(
                    ptx,
                    "\tshfl.sync.down.b32 %r10, %r8, {stride}, 0x1f, 0xffffffff;",
                    stride = stride
                )
                .map_err(write_err)?;
                writeln!(
                    ptx,
                    "\tshfl.sync.down.b32 %r11, %r9, {stride}, 0x1f, 0xffffffff;",
                    stride = stride
                )
                .map_err(write_err)?;
                writeln!(ptx, "\tmov.b64 %rl6, {{%r10, %r11}};").map_err(write_err)?;
                writeln!(ptx, "\t{combine} %rl5, %rl5, %rl6;", combine = combine)
                    .map_err(write_err)?;
            }
            DataType::Float64 => {
                // Split %fd5 into two b32 halves, shuffle each, recombine.
                writeln!(ptx, "\tmov.b64 {{%r8, %r9}}, %fd5;").map_err(write_err)?;
                writeln!(
                    ptx,
                    "\tshfl.sync.down.b32 %r10, %r8, {stride}, 0x1f, 0xffffffff;",
                    stride = stride
                )
                .map_err(write_err)?;
                writeln!(
                    ptx,
                    "\tshfl.sync.down.b32 %r11, %r9, {stride}, 0x1f, 0xffffffff;",
                    stride = stride
                )
                .map_err(write_err)?;
                writeln!(ptx, "\tmov.b64 %fd6, {{%r10, %r11}};").map_err(write_err)?;
                writeln!(ptx, "\t{combine} %fd5, %fd5, %fd6;", combine = combine)
                    .map_err(write_err)?;
            }
            DataType::Bool | DataType::Utf8 => {
                // ptx_type_info already rejects these dtypes above, so this
                // arm is unreachable in practice. Keep the match exhaustive.
                return Err(JavelinError::Type(format!(
                    "agg_kernels: warp-shuffle reduction over dtype {:?} is not supported",
                    acc_dtype
                )));
            }
        }
    }

    // Only lane 0 of warp 0 proceeds to write the block's reduced value to
    // global output. (All other threads with tid != 0 either branched to DONE
    // at the start of phase 2 (tid >= 32) or are warp-0 lanes != 0 whose
    // shuffle results are discarded — only lane 0 holds the fully reduced
    // value.) The %r5/%f5/%rl5/%fd5 register *is* the final reduction; write
    // it straight to global memory without round-tripping through sdata[0].
    writeln!(ptx, "\tsetp.ne.s32 %p5, %r2, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra DONE;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tld.param.u64 %rd8, [{}_param_1];",
        REDUCTION_KERNEL_ENTRY
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.s32 %rd9, %r0, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd10, %rd8, %rd9;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tst.global.{st} [%rd10], %{rc}5;",
        st = acc_store_suffix,
        rc = acc_reg_class
    )
    .map_err(write_err)?;
    writeln!(ptx, "DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

/// Per-dtype PTX type strings:
/// `(ld_st_suffix, ld_st_suffix, value_reg_class, reg_decl_ty, mov_imm_ty)`.
///
/// We use the same suffix for `ld`/`st` (e.g. `s32` for Int32) — separate
/// returns keep the call sites symmetric and forward-compatible if we ever
/// need to differ.
fn ptx_type_info(
    dtype: DataType,
) -> JavelinResult<(&'static str, &'static str, &'static str, &'static str, &'static str)> {
    Ok(match dtype {
        DataType::Int32 => ("s32", "s32", "r", "b32", "s32"),
        DataType::Int64 => ("s64", "s64", "rl", "b64", "s64"),
        DataType::Float32 => ("f32", "f32", "f", "f32", "f32"),
        DataType::Float64 => ("f64", "f64", "fd", "f64", "f64"),
        DataType::Bool | DataType::Utf8 => {
            return Err(JavelinError::Type(format!(
                "agg_kernels: dtype {:?} not supported in reduction kernels",
                dtype
            )))
        }
    })
}

/// Adapt an `std::fmt::Error` into a `JavelinError`.
fn write_err(e: std::fmt::Error) -> JavelinError {
    JavelinError::Other(format!("agg_kernels: write failed: {}", e))
}
