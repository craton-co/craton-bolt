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
use crate::plan::logical_plan::{AggregateExpr, DataType};

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
pub fn compile_reduction_kernel(op: ReduceOp, dtype: DataType) -> JavelinResult<String> {
    let (load_suffix, store_suffix, reg_class, reg_ty, ptx_ty) = ptx_type_info(dtype)?;
    let identity = op.identity_ptx(dtype)?;
    let combine = op.combine_ptx(dtype)?;
    let elem_bytes = dtype.byte_width().ok_or_else(|| {
        JavelinError::Other(format!(
            "agg_kernels: variable-width dtype {:?} not supported",
            dtype
        ))
    })?;
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
    writeln!(ptx, "\t.reg .{}   %{}<8>;", reg_ty, reg_class).map_err(write_err)?;
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
    writeln!(
        ptx,
        "\tmul.wide.s32 %rd1, %r3, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.{} %{}0, [%rd2];", load_suffix, reg_class)
        .map_err(write_err)?;
    writeln!(ptx, "\tbra AFTER_LOAD;").map_err(write_err)?;
    writeln!(ptx, "LOAD_IDENTITY:").map_err(write_err)?;
    writeln!(ptx, "\tmov.{} %{}0, {};", ptx_ty, reg_class, identity).map_err(write_err)?;
    writeln!(ptx, "AFTER_LOAD:").map_err(write_err)?;

    // Stash the value into shared memory.
    writeln!(
        ptx,
        "\tmul.wide.s32 %rd3, %r2, {bytes};",
        bytes = elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd4, sdata;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd4, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.{} [%rd5], %{}0;", store_suffix, reg_class)
        .map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // Unrolled tree reduction in shared memory. For BLOCK_SIZE=256 this is
    // strides 128, 64, 32, 16, 8, 4, 2, 1.
    let mut stride = (BLOCK_SIZE / 2) as i32;
    let mut step = 0usize;
    while stride > 0 {
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
            ld = load_suffix,
            rc = reg_class
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tld.shared.{ld} %{rc}2, [%rd5];",
            ld = load_suffix,
            rc = reg_class
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\t{combine} %{rc}3, %{rc}2, %{rc}1;",
            combine = combine,
            rc = reg_class
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\tst.shared.{st} [%rd5], %{rc}3;",
            st = store_suffix,
            rc = reg_class
        )
        .map_err(write_err)?;
        writeln!(ptx, "SKIP_ADD_{step}:", step = step).map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        stride /= 2;
        step += 1;
    }

    // Thread 0 of each block writes sdata[0] to output[blockIdx.x].
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
        "\tld.shared.{ld} %{rc}4, [%rd4];",
        ld = load_suffix,
        rc = reg_class
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        "\tst.global.{st} [%rd10], %{rc}4;",
        st = store_suffix,
        rc = reg_class
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
