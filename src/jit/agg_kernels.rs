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
//! .visible .entry bolt_reduce(
//!     .param .u64 input_ptr,
//!     .param .u64 output_ptr,
//!     .param .u32 n_rows
//! )
//! ```
//! Block size is hard-coded to `BLOCK_SIZE` (256). The grid is sized by the
//! launcher to `ceil(n_rows / BLOCK_SIZE)`.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
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

/// F7: map an input column dtype to the integer dtype the reduction kernels
/// actually load/combine/store for `op`.
///
/// Temporal columns are stored as plain integers — `Date32` is an `i32` day
/// count and `Timestamp` is an `i64` tick count — and `MIN` / `MAX` / `COUNT`
/// over them are just the corresponding integer extrema / count. Rather than
/// teach every per-dtype PTX arm about temporal types, we collapse
/// `Date32 -> Int32` and `Timestamp -> Int64` here and let the existing i32 /
/// i64 codegen handle them unchanged.
///
/// `SUM` over a temporal dtype is not meaningful SQL (you cannot add two
/// dates), so it is rejected with the same message the per-dtype arms used to
/// emit. Every non-temporal dtype is returned unchanged so the caller's match
/// behaves exactly as before for the historically-supported types.
fn reduction_storage_dtype(op: ReduceOp, dtype: DataType) -> BoltResult<DataType> {
    use DataType::*;
    use ReduceOp::*;
    match (op, dtype) {
        // SUM(temporal) is undefined SQL — keep rejecting it.
        (Sum, Date32) | (Sum, Timestamp(_, _)) => Err(BoltError::Type(format!(
            "agg_kernels: SUM over temporal dtype {:?} is not supported \
             (only MIN/MAX/COUNT are defined for dates/timestamps)",
            dtype
        ))),
        // MIN/MAX/COUNT(temporal) reduce at the underlying integer width.
        (_, Date32) => Ok(Int32),
        (_, Timestamp(_, _)) => Ok(Int64),
        // Everything else is unchanged (incl. the still-unsupported Bool /
        // Utf8 / Decimal128, which the caller's match continues to reject).
        (_, other) => Ok(other),
    }
}

/// Threads per block for every reduction kernel.
///
/// The phase-2 warp-shuffle path emits `shfl.sync.down.b32` with membermask
/// `0xffffffff` and is gated on `tid < 32`. That requires at least one full
/// warp of live threads, i.e. `BLOCK_SIZE >= 32`. The phase-1 strided reduce
/// (`stride = BLOCK_SIZE / 2, /4, ...`) assumes `BLOCK_SIZE` is a power of
/// two. A future tuning change that violates either invariant would silently
/// deadlock at `shfl.sync` or produce a wrong result — the compile-time
/// asserts below catch that at the constant-edit site.
pub const BLOCK_SIZE: u32 = 256;

// Compile-time invariants for the reduction kernel codegen. See the doc
// comment on `BLOCK_SIZE` above.
const _: () = assert!(
    BLOCK_SIZE >= 32,
    "warp-shuffle phase requires BLOCK_SIZE >= 32 (one full warp)"
);
const _: () = assert!(
    BLOCK_SIZE.is_power_of_two(),
    "phase-1 strided reduce requires power-of-two block size"
);
const _: () = assert!(BLOCK_SIZE <= 1024, "CUDA hard limit");

/// PTX kernel entry-point name.
pub const REDUCTION_KERNEL_ENTRY: &str = "bolt_reduce";

/// PTX entry-point name of the validity-aware per-block reduction kernel
/// emitted by [`compile_reduction_kernel_with_validity`].
///
/// Distinct from [`REDUCTION_KERNEL_ENTRY`] so the host launcher can pick the
/// right symbol by name when it decides (at run time, from the input column's
/// Arrow null buffer) whether to route through the NULL-skipping variant.
pub const REDUCTION_KERNEL_WITH_VALIDITY_ENTRY: &str = "bolt_reduce_with_validity";

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
    pub fn from_agg(agg: &AggregateExpr) -> BoltResult<Self> {
        match agg {
            AggregateExpr::Sum(_) => Ok(ReduceOp::Sum),
            AggregateExpr::Min(_) => Ok(ReduceOp::Min),
            AggregateExpr::Max(_) => Ok(ReduceOp::Max),
            AggregateExpr::Count(_) => Ok(ReduceOp::Count),
            AggregateExpr::Avg(_) => Err(BoltError::Other(
                "agg_kernels: AVG must be decomposed into Sum + Count by the caller".into(),
            )),
            // v0.5 has no device-side Welford kernel yet; the scalar
            // aggregate path in `exec::aggregate` reduces these on the
            // host via `exec::welford`. Any caller that ends up here has
            // routed a VAR_POP/VAR_SAMP through the GPU reduction path
            // by mistake.
            AggregateExpr::VarPop(_) | AggregateExpr::VarSamp(_) => Err(BoltError::Other(
                "agg_kernels: VAR_POP / VAR_SAMP have no device-side reduction in v0.5; \
                 the scalar-aggregate executor must dispatch to crate::exec::welford"
                    .into(),
            )),
            AggregateExpr::StddevPop(_) | AggregateExpr::StddevSamp(_) => Err(BoltError::Other(
                "agg_kernels: STDDEV_POP / STDDEV_SAMP do not lower to a \
                 single ReduceOp; handled via the Welford state path".into(),
            )),
        }
    }

    /// PTX literal expression for the identity value of `self` at `dtype`.
    pub fn identity_ptx(self, dtype: DataType) -> BoltResult<String> {
        use DataType::*;
        use ReduceOp::*;
        // F7: temporal columns reduce at their underlying integer width —
        // Date32 is an i32 day count, Timestamp an i64 tick count. MIN/MAX/
        // COUNT are well-defined integer extrema/counts; SUM over a temporal
        // column is not meaningful SQL and is rejected below. Normalising
        // here lets the existing i32/i64 identity/combine arms serve them
        // without duplicating the literals.
        let dtype = reduction_storage_dtype(self, dtype)?;
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

            // `reduction_storage_dtype` has already collapsed Date32/Timestamp
            // to Int32/Int64 (or errored for SUM(temporal)); only the truly
            // unsupported dtypes remain here.
            (_, Bool) | (_, Utf8) | (_, Decimal128(_, _)) | (_, Date32) | (_, Timestamp(_, _)) => Err(BoltError::Type(format!(
                "agg_kernels: reduction over dtype {:?} is not supported",
                dtype
            ))),
        }
    }

    /// PTX combine instruction mnemonic for `self` at `dtype`.
    pub fn combine_ptx(self, dtype: DataType) -> BoltResult<String> {
        use DataType::*;
        use ReduceOp::*;
        // F7: collapse Date32 -> Int32, Timestamp -> Int64 for MIN/MAX/COUNT
        // (SUM over a temporal dtype is rejected). Identical normalisation to
        // `identity_ptx` so the two stay in lock-step.
        let dtype = reduction_storage_dtype(self, dtype)?;
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

            (_, Bool) | (_, Utf8) | (_, Decimal128(_, _)) | (_, Date32) | (_, Timestamp(_, _)) => {
                return Err(BoltError::Type(format!(
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
pub fn compile_reduction_kernel(op: ReduceOp, dtype: DataType) -> BoltResult<String> {
    emit_reduction_kernel(op, dtype, /* with_validity = */ false)
}

/// Generate PTX for a per-block reduction kernel that **honours an Arrow
/// validity bitmap**: NULL rows (validity bit 0) contribute the reduction's
/// identity instead of the garbage value sitting at the masked slot, so the
/// host no longer has to strip NULLs into a dense `Vec` before upload.
///
/// ## ABI
///
/// ```text
/// .visible .entry bolt_reduce_with_validity(
///     .param .u64 input_ptr,      // T, length n_rows
///     .param .u64 output_ptr,     // accumulator dtype, length grid_x
///     .param .u32 n_rows,
///     .param .u64 validity_ptr    // u8, length ceil(n_rows/8), Arrow LE bits
/// )
/// ```
///
/// The only delta versus [`compile_reduction_kernel`] is a packed-bit
/// validity gate inserted into the per-thread load: a thread whose row is in
/// range but whose validity bit is 0 takes the `LOAD_IDENTITY` path, exactly
/// as an out-of-range thread does. Because NULL rows fold to the identity,
/// SUM/MIN/MAX skip them and a COUNT(expr) (driven by an all-ones value
/// column) sees a 0 contribution for every NULL — matching SQL's
/// "COUNT of non-null" semantics without a host-side strip pass.
///
/// `dtype` is the *input* column dtype; the accumulator / output dtype is
/// [`reduction_output_dtype`] as in the no-validity variant.
pub fn compile_reduction_kernel_with_validity(
    op: ReduceOp,
    dtype: DataType,
) -> BoltResult<String> {
    emit_reduction_kernel(op, dtype, /* with_validity = */ true)
}

/// Shared emitter for the no-validity ([`compile_reduction_kernel`]) and
/// validity-aware ([`compile_reduction_kernel_with_validity`]) per-block
/// reduction kernels. `with_validity = true` appends a trailing
/// `.param .u64 validity_ptr` (Arrow LE packed-bit bitmap, one bit per row)
/// and emits a bit-test in the per-thread load that routes NULL rows to the
/// identity.
fn emit_reduction_kernel(
    op: ReduceOp,
    dtype: DataType,
    with_validity: bool,
) -> BoltResult<String> {
    // F7: a temporal input column reduces at its underlying integer width
    // (Date32 -> Int32 day count, Timestamp -> Int64 tick count). Normalise
    // the input dtype up front so every downstream helper here
    // (`ptx_type_info`, `reduction_output_dtype`, `byte_width`, the
    // warp-shuffle match on `acc_dtype`) operates on the integer dtype and
    // emits exactly the i32/i64 reduction code. MIN/MAX/COUNT are the only
    // ops that reach this for temporal inputs (SUM(temporal) is rejected by
    // `reduction_storage_dtype`), and for those the accumulator dtype equals
    // the input dtype, so collapsing the input is sound.
    let dtype = reduction_storage_dtype(op, dtype)?;
    let entry = if with_validity {
        REDUCTION_KERNEL_WITH_VALIDITY_ENTRY
    } else {
        REDUCTION_KERNEL_ENTRY
    };
    // Input-side PTX info governs the global load from the source column.
    let (input_load_suffix, _input_store_suffix, input_reg_class, input_reg_ty, _input_imm_ty) =
        ptx_type_info(dtype)?;
    // Accumulator-side PTX info governs identity, combine, shared-memory
    // storage, and the global output store. For SUM(Int32) these diverge:
    // input is `s32`, accumulator is `s64`.
    let acc_dtype = reduction_output_dtype(op, dtype);
    let (acc_load_suffix, acc_store_suffix, acc_reg_class, acc_reg_ty, acc_imm_ty) =
        ptx_type_info(acc_dtype)?;
    // Int32's natural register class "r" aliases the general-purpose b32 bank
    // that holds tid (%r2) and ctaid (%r0) — both live across the WHOLE
    // reduction: tid gates every stride step, and ctaid indexes the per-block
    // output store at the end. Putting the value/accumulator (and the phase-1
    // reduction scratch %{rc}1/2/3) in %r would clobber them, corrupting both
    // the reduction and the output address (the block's result lands in the
    // wrong slot, leaving the real slot zero-initialised). Route Int32 values
    // through a DISTINCT b32 bank %rv instead — Int64/Float32/Float64 already
    // use distinct rl/f/fd banks, which is why only Int32 reductions were
    // wrong. Applied to both the accumulator and (for SUM(Int32)->Int64) the
    // widening input load.
    let acc_reg_class = if acc_reg_class == "r" { "rv" } else { acc_reg_class };
    let input_reg_class = if input_reg_class == "r" { "rv" } else { input_reg_class };
    let widens = acc_dtype != dtype;

    // Identity and combine are computed against the ACCUMULATOR dtype: the
    // tree reduction in shared memory operates on widened values.
    let identity = op.identity_ptx(acc_dtype)?;
    let combine = op.combine_ptx(acc_dtype)?;

    let input_elem_bytes = dtype.byte_width().ok_or_else(|| {
        BoltError::Other(format!(
            "agg_kernels: variable-width dtype {:?} not supported",
            dtype
        ))
    })?;
    let acc_elem_bytes = acc_dtype.byte_width().ok_or_else(|| {
        BoltError::Other(format!(
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

    writeln!(ptx, ".visible .entry {}(", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", entry).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", entry).map_err(write_err)?;
    if with_validity {
        writeln!(ptx, "\t.param .u32 {}_param_2,", entry).map_err(write_err)?;
        // Trailing validity_ptr (Arrow LE packed bits, one bit per row).
        writeln!(ptx, "\t.param .u64 {}_param_3", entry).map_err(write_err)?;
    } else {
        writeln!(ptx, "\t.param .u32 {}_param_2", entry).map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Register declarations. We use generous counts because PTX `.reg`
    // declarations only allocate names, not real hardware registers.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<16>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<24>;").map_err(write_err)?;
    // The general-purpose banks above already cover the b32 (`%r`) and b64
    // (`%rd`) classes. An INTEGER accumulator/input shares those banks (the
    // body emits `%r`/`%rd` names for both index math and the accumulated
    // value), so a dedicated `.reg .b32 %r<..>;` / `.reg .b64 %rd<..>;` here
    // would be a duplicate-definition PTX error ("Duplicate definition of
    // variable '%r<'", aborting cuModuleLoadDataEx). Only FLOAT accumulators
    // (`%f`/`%fd`) and inputs need their own bank. `is_general_bank` gates
    // both the accumulator decl and the widening-input decl on that.
    let is_general_bank = |class: &str| matches!(class, "p" | "r" | "rd");
    if !is_general_bank(acc_reg_class) {
        writeln!(ptx, "\t.reg .{}   %{}<8>;", acc_reg_ty, acc_reg_class).map_err(write_err)?;
    }
    // When the kernel widens (e.g. SUM(Int32) -> Int64), the raw input load
    // uses a separate register class that is then sign-extended into the
    // accumulator. Emit it only when it's a NEW (float) bank that isn't already
    // declared above and doesn't coincide with the accumulator bank — otherwise
    // it's another duplicate `.reg` of the same class.
    if widens && !is_general_bank(input_reg_class) && input_reg_class != acc_reg_class {
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
        entry
    )
    .map_err(write_err)?;

    // Load this thread's value (or the identity if it's past the end).
    writeln!(
        ptx,
        "\tld.param.u64 %rd0, [{}_param_0];",
        entry
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra LOAD_IDENTITY;").map_err(write_err)?;

    // Validity gate (only in the `_with_validity` variant): an in-range row
    // whose Arrow validity bit is 0 is NULL — fold it to the identity rather
    // than reading garbage at the masked slot. The bitmap is Arrow LE packed
    // bits (bit `tid % 8` of byte `tid / 8`). Uses high registers / %p7 to
    // stay clear of the reduction body's namespace.
    if with_validity {
        writeln!(ptx, "\tld.param.u64 %rd15, [{}_param_3];", entry)
            .map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd15, %rd15;").map_err(write_err)?;
        // byte_idx = tid >> 3 ; bit_off = tid & 7
        writeln!(ptx, "\tshr.u32 %r12, %r3, 3;").map_err(write_err)?;
        writeln!(ptx, "\tand.b32 %r13, %r3, 7;").map_err(write_err)?;
        // addr = validity_ptr + byte_idx
        writeln!(ptx, "\tmul.wide.u32 %rd16, %r12, 1;").map_err(write_err)?;
        writeln!(ptx, "\tadd.s64 %rd15, %rd15, %rd16;").map_err(write_err)?;
        // byte = ld.global.u8 ; bit = bfe.u32(byte, bit_off, 1)
        writeln!(ptx, "\tld.global.u8 %r14, [%rd15];").map_err(write_err)?;
        writeln!(ptx, "\tbfe.u32 %r15, %r14, %r13, 1;").map_err(write_err)?;
        // NULL (bit == 0) -> identity.
        writeln!(ptx, "\tsetp.eq.s32 %p7, %r15, 0;").map_err(write_err)?;
        writeln!(ptx, "\t@%p7 bra LOAD_IDENTITY;").map_err(write_err)?;
    }
    // Address arithmetic for the source-column load uses the *input* element
    // size (the source array is laid out at input dtype, not accumulator).
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd1, %r3, {bytes};",
        bytes = input_elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    if widens {
        // Load the narrow value into an input-class register, then
        // sign-extend into the accumulator-class register. The acc class is
        // always wider (currently only s64 from s32), so `cvt.<acc>.<in>`
        // is the right widening cvt mnemonic.
        //
        // The source column is a read-only input — route through .nc.
        writeln!(
            ptx,
            "\tld.global.nc.{} %{}0, [%rd2];",
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
        // Same-width load from a read-only input — use .nc.
        writeln!(
            ptx,
            "\tld.global.nc.{} %{}0, [%rd2];",
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
        "\tmul.wide.u32 %rd3, %r2, {bytes};",
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
    writeln!(ptx, "\tsetp.ge.u32 %p6, %r2, 32;").map_err(write_err)?;
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
                // The Int32 acc bank %rv is b32, so %rv5 is already a valid
                // `shfl.sync.down.b32` source — no mov bridge needed. Working
                // value lives in %rv5; shfl scratch is %rv6 (same bank).
                writeln!(
                    ptx,
                    "\tshfl.sync.down.b32 %{rc}6, %{rc}5, {stride}, 0x1f, 0xffffffff;",
                    rc = acc_reg_class,
                    stride = stride
                )
                .map_err(write_err)?;
                writeln!(
                    ptx,
                    "\t{combine} %{rc}5, %{rc}5, %{rc}6;",
                    combine = combine,
                    rc = acc_reg_class
                )
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
            DataType::Bool
            | DataType::Utf8
            | DataType::Decimal128(_, _)
            | DataType::Date32
            | DataType::Timestamp(_, _) => {
                return Err(BoltError::Type(format!(
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
        entry
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd8, %rd8;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd9, %r0, {bytes};",
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
) -> BoltResult<(&'static str, &'static str, &'static str, &'static str, &'static str)> {
    Ok(match dtype {
        DataType::Int32 => ("s32", "s32", "r", "b32", "s32"),
        DataType::Int64 => ("s64", "s64", "rl", "b64", "s64"),
        DataType::Float32 => ("f32", "f32", "f", "f32", "f32"),
        DataType::Float64 => ("f64", "f64", "fd", "f64", "f64"),
        DataType::Bool
        | DataType::Utf8
        | DataType::Decimal128(_, _)
        | DataType::Date32
        | DataType::Timestamp(_, _) => {
            return Err(BoltError::Type(format!(
                "agg_kernels: dtype {:?} not supported in reduction kernels",
                dtype
            )))
        }
    })
}

/// Adapt an `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("agg_kernels: write failed: {}", e))
}

// ---------------------------------------------------------------------------
// Fused AVG kernel.
// ---------------------------------------------------------------------------

/// PTX entry-point name for the fused AVG reduction kernel.
pub const AVG_KERNEL_ENTRY: &str = "bolt_avg_reduce";

/// Per-block sum-of-squares output element type for the AVG kernel: always
/// `f64` regardless of input dtype, so the sum doesn't lose precision as soon
/// as the input is wider than the accumulator (e.g. SUM over a long Int64
/// column with values whose partial sums no longer fit in `f32`).
pub const AVG_SUM_ELEM_BYTES: usize = 8;

/// Per-block count output element type for the AVG kernel: always `u32`. We
/// can't overflow at `BLOCK_SIZE = 256` per block; the grand total fits in
/// u64 at the host-side finalize.
pub const AVG_COUNT_ELEM_BYTES: usize = 4;

/// Generate PTX for the **fused** per-block AVG kernel. In a single pass over
/// the input column each block computes:
///   - `block_sums[blockIdx.x]` (`f64`): partial sum of the in-range elements.
///   - `block_counts[blockIdx.x]` (`u32`): count of in-range elements (i.e.
///     `min(n_rows - blockIdx.x * BLOCK_SIZE, BLOCK_SIZE)` — equivalent to the
///     non-NULL count once the caller has stripped NULL rows on the host).
///
/// This replaces the previous "two separate kernel launches (SUM + COUNT) then
/// divide on the host" sequence, which paid two PTX compilations, two kernel
/// launches, and two D2H copies to compute a single float. The fused kernel
/// keeps the reduction shape identical to `compile_reduction_kernel` (phase 1
/// shared-memory tree + phase 2 warp shuffle) but tracks two accumulators in
/// parallel: an `f64` sum and a `b32` count.
///
/// ABI:
/// ```text
/// .visible .entry bolt_avg_reduce(
///     .param .u64 input_ptr,
///     .param .u64 block_sums_ptr,
///     .param .u64 block_counts_ptr,
///     .param .u32 n_rows
/// )
/// ```
///
/// `dtype` is the *input* column dtype. The sum accumulator is always `f64`;
/// the kernel cvts each loaded primitive into `f64` before the per-thread
/// contribution. NULL rows are not handled by this kernel — the caller (host)
/// is responsible for stripping them, exactly as it does for the SUM kernel.
pub fn compile_avg_reduction_kernel(dtype: DataType) -> BoltResult<String> {
    let (input_load_suffix, _, input_reg_class, input_reg_ty, _input_imm_ty) =
        ptx_type_info(dtype)?;
    // Int32's natural class "r" aliases the general bank holding ctaid (%r0,
    // used for the per-block output index) and tid (%r2); loading the input
    // into %r0 would clobber ctaid and write the block's sum/count to the wrong
    // slot. Route the (transient) Int32 input load through the distinct %rv
    // b32 bank, matching emit_reduction_kernel. Int64/Float inputs already use
    // distinct rl/f banks.
    let input_reg_class = if input_reg_class == "r" { "rv" } else { input_reg_class };
    let input_elem_bytes = dtype.byte_width().ok_or_else(|| {
        BoltError::Other(format!(
            "agg_kernels: variable-width dtype {:?} not supported in AVG kernel",
            dtype
        ))
    })?;

    // Two shared-memory scratch pads: one for the f64 sum, one for the u32 count.
    let sum_shared_bytes = BLOCK_SIZE as usize * AVG_SUM_ELEM_BYTES;
    let count_shared_bytes = BLOCK_SIZE as usize * AVG_COUNT_ELEM_BYTES;

    let mut ptx = String::new();
    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 8 .b8 sdata_sum[{n}];",
        n = sum_shared_bytes
    )
    .map_err(write_err)?;
    writeln!(
        ptx,
        ".shared .align 4 .b8 sdata_cnt[{n}];",
        n = count_shared_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, ".visible .entry {}(", AVG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_0,", AVG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_1,", AVG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u64 {}_param_2,", AVG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\t.param .u32 {}_param_3", AVG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    // Generous register names — PTX allocates names, not registers.
    writeln!(ptx, "\t.reg .pred  %p<8>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<32>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<16>;").map_err(write_err)?;
    // If the input dtype isn't already f64, allocate the input-class register
    // so we can `ld.global.<inp>` then `cvt.f64.<inp>` into the sum accumulator.
    // An INTEGER input (`%r`/`%rd`) shares the general-purpose banks declared
    // above, so emitting a dedicated decl would be a duplicate-definition PTX
    // error; only a distinct FLOAT input bank (`%f`) needs its own declaration.
    if dtype != DataType::Float64 && !matches!(input_reg_class, "r" | "rd") {
        writeln!(ptx, "\t.reg .{}   %{}<4>;", input_reg_ty, input_reg_class)
            .map_err(write_err)?;
    }
    writeln!(ptx).map_err(write_err)?;

    // gid = ctaid.x * ntid.x + tid.x
    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmad.lo.s32 %r3, %r0, %r1, %r2;").map_err(write_err)?;
    writeln!(ptx, "\tld.param.u32 %r4, [{}_param_3];", AVG_KERNEL_ENTRY).map_err(write_err)?;

    // Load input value as f64 (or +0.0 if past the end) and set the count
    // contribution (1 if in range, 0 otherwise).
    writeln!(ptx, "\tld.param.u64 %rd0, [{}_param_0];", AVG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p0, %r3, %r4;").map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra AVG_LOAD_IDENTITY;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd1, %r3, {bytes};",
        bytes = input_elem_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    if dtype == DataType::Float64 {
        writeln!(ptx, "\tld.global.f64 %fd0, [%rd2];").map_err(write_err)?;
    } else {
        writeln!(
            ptx,
            "\tld.global.{ld} %{rc}0, [%rd2];",
            ld = input_load_suffix,
            rc = input_reg_class
        )
        .map_err(write_err)?;
        // PTX requires a rounding modifier on integer -> floating-point cvt
        // (e.g. `cvt.rn.f64.s32`): the result of s32/s64 -> f64 must name a
        // rounding mode or ptxas aborts with "Rounding modifier required for
        // instruction 'cvt'". A widening f32 -> f64 conversion is exact and
        // takes (and allows) no rounding modifier.
        let cvt_round = if matches!(dtype, DataType::Int32 | DataType::Int64) {
            "rn."
        } else {
            ""
        };
        writeln!(
            ptx,
            "\tcvt.{round}f64.{ld} %fd0, %{rc}0;",
            round = cvt_round,
            ld = input_load_suffix,
            rc = input_reg_class
        )
        .map_err(write_err)?;
    }
    // Count contribution: 1 for in-range threads.
    writeln!(ptx, "\tmov.u32 %r5, 1;").map_err(write_err)?;
    writeln!(ptx, "\tbra AVG_AFTER_LOAD;").map_err(write_err)?;

    writeln!(ptx, "AVG_LOAD_IDENTITY:").map_err(write_err)?;
    // Sum identity is +0.0; count identity is 0.
    writeln!(
        ptx,
        "\tmov.f64 %fd0, 0d{:016X};",
        0f64.to_bits()
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r5, 0;").map_err(write_err)?;
    writeln!(ptx, "AVG_AFTER_LOAD:").map_err(write_err)?;

    // Stash sum and count into their shared-memory slots.
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd3, %r2, {bytes};",
        bytes = AVG_SUM_ELEM_BYTES
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd4, sdata_sum;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd5, %rd4, %rd3;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.f64 [%rd5], %fd0;").map_err(write_err)?;

    writeln!(
        ptx,
        "\tmul.wide.u32 %rd6, %r2, {bytes};",
        bytes = AVG_COUNT_ELEM_BYTES
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u64 %rd7, sdata_cnt;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd8, %rd7, %rd6;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.b32 [%rd8], %r5;").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;

    // Phase 1: inter-warp shared-memory tree at strides 128, 64, 32 — exactly
    // mirroring `compile_reduction_kernel`. We do both reductions in lockstep
    // under a single `tid < stride` predicate so we share the barrier and the
    // address arithmetic.
    let phase1_strides: [i32; 3] = [128, 64, 32];
    let mut step = 0usize;
    for &stride in &phase1_strides {
        let neighbor_pred = format!("%p{}", 1 + (step % 4));
        let sum_stride_bytes = stride as usize * AVG_SUM_ELEM_BYTES;
        let cnt_stride_bytes = stride as usize * AVG_COUNT_ELEM_BYTES;

        writeln!(
            ptx,
            "\tsetp.lt.s32 {pred}, %r2, {stride};",
            pred = neighbor_pred,
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(
            ptx,
            "\t@!{pred} bra AVG_SKIP_ADD_{step};",
            pred = neighbor_pred,
            step = step
        )
        .map_err(write_err)?;

        // Sum: load own + neighbour, add, store back.
        writeln!(
            ptx,
            "\tadd.s64 %rd10, %rd5, {offset};",
            offset = sum_stride_bytes
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tld.shared.f64 %fd1, [%rd10];").map_err(write_err)?;
        writeln!(ptx, "\tld.shared.f64 %fd2, [%rd5];").map_err(write_err)?;
        writeln!(ptx, "\tadd.f64 %fd3, %fd2, %fd1;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.f64 [%rd5], %fd3;").map_err(write_err)?;

        // Count.
        writeln!(
            ptx,
            "\tadd.s64 %rd11, %rd8, {offset};",
            offset = cnt_stride_bytes
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tld.shared.b32 %r6, [%rd11];").map_err(write_err)?;
        writeln!(ptx, "\tld.shared.b32 %r7, [%rd8];").map_err(write_err)?;
        writeln!(ptx, "\tadd.s32 %r8, %r7, %r6;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.b32 [%rd8], %r8;").map_err(write_err)?;

        writeln!(ptx, "AVG_SKIP_ADD_{step}:", step = step).map_err(write_err)?;
        writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
        step += 1;
    }

    // Phase 2: warp-shuffle intra-warp reduction. Gate on tid < 32; the
    // remaining strides (16, 8, 4, 2, 1) run register-only.
    writeln!(ptx, "\tsetp.ge.u32 %p6, %r2, 32;").map_err(write_err)?;
    writeln!(ptx, "\t@%p6 bra AVG_DONE;").map_err(write_err)?;

    // Lane 0..31 loads its own value into a register once.
    writeln!(ptx, "\tld.shared.f64 %fd5, [%rd5];").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.b32 %r9, [%rd8];").map_err(write_err)?;

    for &stride in &[16i32, 8, 4, 2, 1] {
        // Sum (f64): split into two b32 halves, shuffle each, recombine.
        writeln!(ptx, "\tmov.b64 {{%r10, %r11}}, %fd5;").map_err(write_err)?;
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
        writeln!(ptx, "\tmov.b64 %fd6, {{%r12, %r13}};").map_err(write_err)?;
        writeln!(ptx, "\tadd.f64 %fd5, %fd5, %fd6;").map_err(write_err)?;

        // Count (u32 / b32): single shfl.
        writeln!(
            ptx,
            "\tshfl.sync.down.b32 %r14, %r9, {stride}, 0x1f, 0xffffffff;",
            stride = stride
        )
        .map_err(write_err)?;
        writeln!(ptx, "\tadd.s32 %r9, %r9, %r14;").map_err(write_err)?;
    }

    // Only lane 0 of warp 0 writes the per-block partial.
    writeln!(ptx, "\tsetp.ne.s32 %p5, %r2, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra AVG_DONE;").map_err(write_err)?;

    // Write block_sums[blockIdx.x].
    writeln!(ptx, "\tld.param.u64 %rd20, [{}_param_1];", AVG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd20, %rd20;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd21, %r0, {bytes};",
        bytes = AVG_SUM_ELEM_BYTES
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd22, %rd20, %rd21;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.f64 [%rd22], %fd5;").map_err(write_err)?;

    // Write block_counts[blockIdx.x].
    writeln!(ptx, "\tld.param.u64 %rd23, [{}_param_2];", AVG_KERNEL_ENTRY).map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd23, %rd23;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tmul.wide.u32 %rd24, %r0, {bytes};",
        bytes = AVG_COUNT_ELEM_BYTES
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd25, %rd23, %rd24;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.b32 [%rd25], %r9;").map_err(write_err)?;

    writeln!(ptx, "AVG_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

// ---------------------------------------------------------------------------
// PTX-shape tests (host-only — no CUDA required).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod validity_reduction_tests {
    use super::*;

    /// The no-validity reduction kernel must NOT emit the validity gate:
    /// no trailing `validity_ptr` param, no `bfe.u32` bit-extract, and the
    /// classic 3-parameter ABI.
    #[test]
    fn no_validity_variant_has_no_bit_gate() {
        let ptx = compile_reduction_kernel(ReduceOp::Sum, DataType::Int64)
            .expect("kernel should compile");
        assert!(
            ptx.contains(REDUCTION_KERNEL_ENTRY),
            "expected the classic entry name:\n{ptx}"
        );
        assert!(
            !ptx.contains("bfe.u32"),
            "no-validity kernel must not emit a validity bit-extract:\n{ptx}"
        );
        // Classic ABI: 3 `.param ` declarations (input, output, n_rows).
        let param_count = ptx.matches(".param ").count();
        assert_eq!(
            param_count, 3,
            "expected 3 `.param ` decls in the no-validity variant, got {param_count}:\n{ptx}"
        );
    }

    /// The validity-aware reduction kernel must emit a 4-param ABI with a
    /// trailing `validity_ptr`, the `bfe.u32` bit-extract on the loaded
    /// validity byte, and a guarded branch to `LOAD_IDENTITY` so a NULL row
    /// folds to the reduction identity. The branch must precede the
    /// `LOAD_IDENTITY` label so it genuinely skips the value load.
    #[test]
    fn validity_variant_emits_null_guard_branch() {
        let ptx = compile_reduction_kernel_with_validity(ReduceOp::Sum, DataType::Int64)
            .expect("kernel should compile");
        assert!(
            ptx.contains(REDUCTION_KERNEL_WITH_VALIDITY_ENTRY),
            "expected the `_with_validity` entry name:\n{ptx}"
        );
        assert!(
            !ptx.contains(".visible .entry bolt_reduce("),
            "validity kernel must not re-emit the no-validity entry name:\n{ptx}"
        );
        // Validity-gate shape: byte load + bit-extract + branch-to-identity.
        assert!(
            ptx.contains("ld.global.u8 %r14, [%rd15];"),
            "expected validity byte load:\n{ptx}"
        );
        assert!(
            ptx.contains("bfe.u32 %r15, %r14, %r13, 1;"),
            "expected `bfe.u32` 1-bit extract of the validity bit:\n{ptx}"
        );
        assert!(
            ptx.contains("setp.eq.s32 %p7, %r15, 0;"),
            "expected `setp.eq` testing the extracted validity bit against 0:\n{ptx}"
        );
        assert!(
            ptx.contains("@%p7 bra LOAD_IDENTITY;"),
            "expected NULL rows to branch to LOAD_IDENTITY (fold to identity):\n{ptx}"
        );
        // The guard must come BEFORE the LOAD_IDENTITY label.
        let guard_pos = ptx.find("@%p7 bra LOAD_IDENTITY;").unwrap();
        let label_pos = ptx.find("LOAD_IDENTITY:").unwrap();
        assert!(
            guard_pos < label_pos,
            "validity guard must precede the LOAD_IDENTITY label:\n{ptx}"
        );
        // 4-param ABI: the trailing validity_ptr is param_3.
        let param_count = ptx.matches(".param ").count();
        assert_eq!(
            param_count, 4,
            "expected 4 `.param ` decls (added validity_ptr), got {param_count}:\n{ptx}"
        );
        assert!(
            ptx.contains(&format!("{}_param_3", REDUCTION_KERNEL_WITH_VALIDITY_ENTRY)),
            "expected param_3 (validity_ptr) in the emitted PTX:\n{ptx}"
        );
    }

    /// The widening SUM(Int32)->Int64 validity kernel must still carry the
    /// validity gate AND the `cvt.s64.s32` widening cvt: the two features are
    /// orthogonal and must compose.
    #[test]
    fn validity_variant_widening_sum_int32_keeps_gate_and_cvt() {
        let ptx = compile_reduction_kernel_with_validity(ReduceOp::Sum, DataType::Int32)
            .expect("kernel should compile");
        assert!(
            ptx.contains("bfe.u32 %r15, %r14, %r13, 1;"),
            "widening validity kernel must keep the bit-extract gate:\n{ptx}"
        );
        assert!(
            ptx.contains("cvt.s64.s32"),
            "SUM(Int32) must sign-extend into the s64 accumulator:\n{ptx}"
        );
    }

    /// MIN/Float32 with validity must fold NULL rows to the +inf identity
    /// (so they never win the MIN) — assert both the identity literal and the
    /// gate are present.
    #[test]
    fn validity_variant_min_f32_uses_inf_identity() {
        let ptx = compile_reduction_kernel_with_validity(ReduceOp::Min, DataType::Float32)
            .expect("kernel should compile");
        let inf = format!("0f{:08X}", f32::INFINITY.to_bits());
        assert!(
            ptx.contains(&inf),
            "MIN(Float32) identity must be +inf ({inf}):\n{ptx}"
        );
        assert!(
            ptx.contains("@%p7 bra LOAD_IDENTITY;"),
            "MIN(Float32) validity kernel must guard NULL rows:\n{ptx}"
        );
    }
}

// ---------------------------------------------------------------------------
// F7: temporal (Date32 / Timestamp) scalar reduction codegen. Host-only PTX
// shape + eligibility tests — no CUDA required. Date32 must reduce as Int32
// (4-byte loads), Timestamp as Int64 (8-byte loads); SUM over a temporal dtype
// stays rejected.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod temporal_reduction_tests {
    use super::*;
    use crate::plan::logical_plan::TimeUnit;

    const TS_US: DataType = DataType::Timestamp(TimeUnit::Microsecond, None);

    /// `reduction_output_dtype` already returns the input dtype for MIN/MAX;
    /// the *kernel* must collapse Date32->Int32 so it emits 4-byte i32 code,
    /// byte-identical to a real `MIN(Int32)` reduction.
    #[test]
    fn min_date32_emits_int32_kernel() {
        let date = compile_reduction_kernel(ReduceOp::Min, DataType::Date32)
            .expect("MIN(Date32) should compile via the Int32 path");
        let i32_ref = compile_reduction_kernel(ReduceOp::Min, DataType::Int32)
            .expect("MIN(Int32) reference");
        assert_eq!(
            date, i32_ref,
            "MIN(Date32) PTX must match MIN(Int32) PTX byte-for-byte"
        );
        // s32 MIN combine + i32::MAX identity must be present.
        assert!(date.contains("min.s32"), "expected min.s32 combine:\n{date}");
        assert!(
            date.contains(&i32::MAX.to_string()),
            "expected i32::MAX MIN identity:\n{date}"
        );
    }

    /// MAX(Timestamp) must collapse to the Int64 kernel (8-byte loads,
    /// `max.s64`, i64::MIN identity), matching `MAX(Int64)` exactly.
    #[test]
    fn max_timestamp_emits_int64_kernel() {
        let ts = compile_reduction_kernel(ReduceOp::Max, TS_US)
            .expect("MAX(Timestamp) should compile via the Int64 path");
        let i64_ref = compile_reduction_kernel(ReduceOp::Max, DataType::Int64)
            .expect("MAX(Int64) reference");
        assert_eq!(
            ts, i64_ref,
            "MAX(Timestamp) PTX must match MAX(Int64) PTX byte-for-byte"
        );
        assert!(ts.contains("max.s64"), "expected max.s64 combine:\n{ts}");
    }

    /// COUNT over temporal dtypes is just the integer count path.
    #[test]
    fn count_temporal_routes_to_integer_path() {
        assert_eq!(
            compile_reduction_kernel(ReduceOp::Count, DataType::Date32).unwrap(),
            compile_reduction_kernel(ReduceOp::Count, DataType::Int32).unwrap(),
        );
        assert_eq!(
            compile_reduction_kernel(ReduceOp::Count, TS_US).unwrap(),
            compile_reduction_kernel(ReduceOp::Count, DataType::Int64).unwrap(),
        );
    }

    /// The validity-aware variant must compose with the temporal normalisation:
    /// MIN(Date32) with validity == MIN(Int32) with validity.
    #[test]
    fn validity_min_date32_matches_int32() {
        let date = compile_reduction_kernel_with_validity(ReduceOp::Min, DataType::Date32)
            .expect("MIN(Date32) with validity should compile");
        let i32_ref = compile_reduction_kernel_with_validity(ReduceOp::Min, DataType::Int32)
            .expect("MIN(Int32) with validity reference");
        assert_eq!(date, i32_ref);
        assert!(date.contains("bfe.u32"), "validity gate must survive:\n{date}");
    }

    /// SUM over a temporal dtype is undefined SQL and must be rejected at both
    /// the codegen entry point and the `identity_ptx`/`combine_ptx` helpers.
    #[test]
    fn sum_temporal_is_rejected() {
        for dt in [DataType::Date32, TS_US] {
            let err = compile_reduction_kernel(ReduceOp::Sum, dt)
                .expect_err("SUM(temporal) must be rejected");
            assert!(
                err.to_string().contains("SUM over temporal"),
                "unexpected SUM(temporal) error for {dt:?}: {err}"
            );
            assert!(ReduceOp::Sum.identity_ptx(dt).is_err());
            assert!(ReduceOp::Sum.combine_ptx(dt).is_err());
        }
    }

    /// `identity_ptx` / `combine_ptx` accept MIN/MAX/COUNT over temporal and
    /// produce the integer-typed PTX literals/mnemonics.
    #[test]
    fn identity_and_combine_accept_temporal_minmax_count() {
        // Date32 -> Int32 mnemonics.
        assert_eq!(ReduceOp::Min.combine_ptx(DataType::Date32).unwrap(), "min.s32");
        assert_eq!(ReduceOp::Max.combine_ptx(DataType::Date32).unwrap(), "max.s32");
        assert_eq!(
            ReduceOp::Min.identity_ptx(DataType::Date32).unwrap(),
            i32::MAX.to_string()
        );
        // Timestamp -> Int64 mnemonics.
        assert_eq!(ReduceOp::Max.combine_ptx(TS_US).unwrap(), "max.s64");
        assert_eq!(
            ReduceOp::Max.identity_ptx(TS_US).unwrap(),
            i64::MIN.to_string()
        );
        // COUNT over temporal lowers to the additive (integer) identity 0.
        assert_eq!(ReduceOp::Count.identity_ptx(DataType::Date32).unwrap(), "0");
        assert_eq!(ReduceOp::Count.identity_ptx(TS_US).unwrap(), "0");
    }

    /// Bool / Utf8 / Decimal128 remain rejected (out of scope for F7) — the
    /// normalisation must not accidentally let them through.
    #[test]
    fn still_rejects_bool_utf8_decimal() {
        for dt in [
            DataType::Bool,
            DataType::Utf8,
            DataType::Decimal128(10, 2),
        ] {
            assert!(compile_reduction_kernel(ReduceOp::Min, dt).is_err());
            assert!(compile_reduction_kernel(ReduceOp::Max, dt).is_err());
            assert!(compile_reduction_kernel(ReduceOp::Sum, dt).is_err());
        }
    }
}
