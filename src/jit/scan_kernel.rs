// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for the predicate-only "scan" kernel used by filter compaction.
//!
//! Given a `KernelSpec` whose `predicate` field is `Some(reg)`, this module
//! emits a self-contained PTX module whose entry-point evaluates only the
//! predicate lineage from `spec.ops` and writes one `u8` per row to an extra
//! mask output column (`1` = keep, `0` = drop).
//!
//! ABI of the emitted kernel:
//!
//! ```text
//! .visible .entry bolt_predicate(
//!     .param .u64 bolt_predicate_param_0,        // input col 0
//!     ...
//!     .param .u64 bolt_predicate_param_{N-1},    // input col N-1
//!     .param .u64 bolt_predicate_param_{N},      // mask output (u8*)
//!     .param .u64 bolt_predicate_param_{N+1},    // validity ptr for input #i_a (u8*)
//!     ...                                        //   one per flagged input
//!     .param .u64 bolt_predicate_param_{N+K},    // validity ptr for input #i_K (u8*)
//!     .param .u32 bolt_predicate_param_{N+K+1}_n_rows
//! )
//! ```
//!
//! where `N == spec.inputs.len()` and `K` is the count of `true` entries in
//! `KernelSpec::input_has_validity` (zero when the field is empty — the
//! legacy "no validity" shape). The validity pointers appear AFTER the mask
//! output, in flagged-input order; `Op::IsNullCheck` references the
//! input-slot index, and `compile_predicate_kernel` resolves it through
//! `input_validity_ptrs[validity_input]` — same convention as
//! `ptx_gen::compile`. The grid is 1D, one thread per row, with block size
//! 256 (chosen by `crate::exec::compact::launch_predicate_kernel`).
//!
//! ## Why duplicate logic from `ptx_gen.rs`?
//!
//! The mainline projection codegen (`ptx_gen::compile`) walks the same op set
//! and shares the same register-allocation conventions (`%r`/`%rl`/`%f`/`%fd`
//! for value classes; `%p`/`%rd` for predicates and 64-bit address temps).
//! Rather than expose internal helpers from `ptx_gen.rs`, we inline the
//! pieces we need (compute-op lowering + register allocator) so the two
//! modules can evolve independently. The cost is ~250 lines of duplication
//! for full module independence.

use std::collections::HashMap;
use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{BinaryOp, DataType, Literal};
use crate::plan::physical_plan::{KernelSpec, Op, Reg};

/// PTX target metadata baked into every emitted module.
const PTX_VERSION: &str = ".version 7.5";
/// Target SM architecture string.
const PTX_TARGET: &str = ".target sm_70";
/// Address size directive (we always use 64-bit pointers).
const PTX_ADDRESS_SIZE: &str = ".address_size 64";

/// Register class tag used by the allocator.
type RegClass = &'static str;

/// Per-class register counter + logical-to-physical mapping table.
///
/// Mirrors `ptx_gen::RegAlloc` exactly; we duplicate to keep `scan_kernel`
/// self-contained.
struct RegAlloc {
    /// Next free index per class (e.g. `"r" -> 5` means `%r5` is next).
    next: HashMap<RegClass, u32>,
    /// Logical SSA register -> physical register name (e.g. `%r3`).
    mapping: HashMap<Reg, String>,
}

impl RegAlloc {
    /// New empty allocator.
    fn new() -> Self {
        Self {
            next: HashMap::new(),
            mapping: HashMap::new(),
        }
    }

    /// Allocate a fresh physical register of the given class and return its name.
    fn alloc(&mut self, class: RegClass) -> String {
        let n = self.next.entry(class).or_insert(0);
        let name = format!("%{}{}", class, *n);
        *n += 1;
        name
    }

    /// Assign a physical register to logical `reg` based on `dtype`; returns the name.
    fn assign(&mut self, reg: Reg, dtype: DataType) -> BoltResult<String> {
        let class = Self::class_for(dtype)?;
        let name = self.alloc(class);
        self.mapping.insert(reg, name.clone());
        Ok(name)
    }

    /// Look up the physical register name previously assigned to `reg`.
    fn get(&self, reg: Reg) -> BoltResult<&str> {
        self.mapping.get(&reg).map(|s| s.as_str()).ok_or_else(|| {
            BoltError::Other(format!("scan_kernel: undefined register {:?}", reg))
        })
    }

    /// Map a logical dtype to a PTX register class string.
    fn class_for(dtype: DataType) -> BoltResult<RegClass> {
        Ok(match dtype {
            DataType::Bool => "r",
            DataType::Int32 => "r",
            DataType::Int64 => "rl",
            DataType::Float32 => "f",
            DataType::Float64 => "fd",
            DataType::Utf8 => {
                return Err(BoltError::Other(
                    "scan_kernel: Utf8 not supported in PTX codegen".into(),
                ))
            }
            DataType::Decimal128(_, _) => {
                return Err(BoltError::Plan(
                    "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
                ))
            }
        })
    }

    /// Current count of registers issued for `class` (used to size `.reg` decls).
    fn count(&self, class: RegClass) -> u32 {
        *self.next.get(class).unwrap_or(&0)
    }
}

/// In-progress kernel body + register allocator.
struct PtxBuilder {
    alloc: RegAlloc,
    body: String,
    kernel_name: String,
}

impl PtxBuilder {
    fn new(kernel_name: &str) -> Self {
        Self {
            alloc: RegAlloc::new(),
            body: String::new(),
            kernel_name: kernel_name.to_string(),
        }
    }

    fn emit(&mut self, line: &str) -> BoltResult<()> {
        writeln!(self.body, "\t{}", line)
            .map_err(|e| BoltError::Other(format!("scan_kernel: write failed: {}", e)))
    }

    fn emit_label(&mut self, label: &str) -> BoltResult<()> {
        writeln!(self.body, "{}:", label)
            .map_err(|e| BoltError::Other(format!("scan_kernel: write failed: {}", e)))
    }

    /// Mangled `.param` identifier for the `i`th parameter.
    fn param_name(&self, i: usize) -> String {
        format!("{}_param_{}", self.kernel_name, i)
    }

    /// Mangled `.param` identifier for the row-count parameter.
    ///
    /// `n_input_params` is the count of input column pointers; the mask output
    /// pointer occupies index `n_input_params`, and `n_rows` sits at
    /// `n_input_params + 1 + n_validity_params` (one validity pointer per
    /// flagged input). `n_validity_params == 0` reproduces the historical
    /// param layout bit-for-bit.
    fn n_rows_param_name(&self, n_input_params: usize, n_validity_params: usize) -> String {
        format!(
            "{}_param_{}_n_rows",
            self.kernel_name,
            n_input_params + 1 + n_validity_params
        )
    }
}

/// Compile a predicate-only PTX module from `spec`.
///
/// Errors if `spec.predicate` is `None` (this kernel only makes sense for
/// filtered queries) or if the predicate references unsupported types
/// (`Utf8` in any form).
pub fn compile_predicate_kernel(spec: &KernelSpec, kernel_name: &str) -> BoltResult<String> {
    validate_kernel_name(kernel_name)?;

    let predicate_reg = spec.predicate.ok_or_else(|| {
        BoltError::Other(
            "scan_kernel: compile_predicate_kernel requires a predicate; spec.predicate is None"
                .into(),
        )
    })?;

    // -------- Validity wiring (mirror ptx_gen::compile). The
    //          `input_has_validity` field is opt-in: when empty we emit the
    //          historical PTX shape (no extra params, no validity loads) so
    //          every legacy caller continues to work bit-for-bit. When set,
    //          it MUST be parallel to `inputs`.
    let input_valid: Vec<bool> = if spec.input_has_validity.is_empty() {
        vec![false; spec.inputs.len()]
    } else {
        if spec.input_has_validity.len() != spec.inputs.len() {
            return Err(BoltError::Other(format!(
                "scan_kernel: input_has_validity len {} != inputs len {}",
                spec.input_has_validity.len(),
                spec.inputs.len()
            )));
        }
        spec.input_has_validity.clone()
    };
    let n_input_validity: usize = input_valid.iter().filter(|b| **b).count();

    let mut b = PtxBuilder::new(kernel_name);

    // -------- TID setup: tid = ctaid.x * ntid.x + tid.x ; bail if tid >= n_rows.
    let ctaid = b.alloc.alloc("r");
    let ntid = b.alloc.alloc("r");
    let tid_x = b.alloc.alloc("r");
    let tid = b.alloc.alloc("r");
    let n_rows = b.alloc.alloc("r");
    let pred_oob = b.alloc.alloc("p");

    b.emit(&format!("mov.u32 {}, %ctaid.x;", ctaid))?;
    b.emit(&format!("mov.u32 {}, %ntid.x;", ntid))?;
    b.emit(&format!("mov.u32 {}, %tid.x;", tid_x))?;
    b.emit(&format!("mad.lo.s32 {}, {}, {}, {};", tid, ctaid, ntid, tid_x))?;
    b.emit(&format!(
        "ld.param.u32 {}, [{}];",
        n_rows,
        b.n_rows_param_name(spec.inputs.len(), n_input_validity)
    ))?;
    b.emit(&format!("setp.ge.u32 {}, {}, {};", pred_oob, tid, n_rows))?;
    b.emit(&format!("@{} bra DONE;", pred_oob))?;

    // -------- Load and globalize each input column base pointer.
    let mut input_ptrs: Vec<String> = Vec::with_capacity(spec.inputs.len());
    for (i, col) in spec.inputs.iter().enumerate() {
        if matches!(col.dtype, DataType::Utf8) {
            return Err(BoltError::Other(
                "scan_kernel: Utf8 inputs not supported in PTX codegen".into(),
            ));
        }
        let rd = b.alloc.alloc("rd");
        b.emit(&format!("ld.param.u64 {}, [{}];", rd, b.param_name(i)))?;
        b.emit(&format!("cvta.to.global.u64 {}, {};", rd, rd))?;
        input_ptrs.push(rd);
    }

    // -------- Load + globalize the mask output pointer (at param index N).
    let mask_ptr = b.alloc.alloc("rd");
    b.emit(&format!(
        "ld.param.u64 {}, [{}];",
        mask_ptr,
        b.param_name(spec.inputs.len())
    ))?;
    b.emit(&format!("cvta.to.global.u64 {}, {};", mask_ptr, mask_ptr))?;

    // -------- Load + globalize each flagged-input validity pointer. The
    //          pointers occupy param indices `N+1 .. N+1+K` (mask is at
    //          `N`). `input_validity_ptrs[i]` is `Some(name)` iff the
    //          corresponding input was flagged; `Op::IsNullCheck` indexes
    //          into this table via its `validity_input` field.
    let mut input_validity_ptrs: Vec<Option<String>> = vec![None; spec.inputs.len()];
    let mut next_param = spec.inputs.len() + 1;
    for (i, has) in input_valid.iter().enumerate() {
        if *has {
            let rd = b.alloc.alloc("rd");
            b.emit(&format!(
                "ld.param.u64 {}, [{}];",
                rd,
                b.param_name(next_param)
            ))?;
            b.emit(&format!("cvta.to.global.u64 {}, {};", rd, rd))?;
            input_validity_ptrs[i] = Some(rd);
            next_param += 1;
        }
    }

    // -------- Lower compute ops (skip Store — projection's responsibility).
    //
    // The predicate's compute lineage is fully contained in the spec's
    // non-Store ops because `lower_projection` emits the predicate first and
    // every transitive operand precedes it. We simply emit every compute op
    // in order, which both produces the predicate register and naturally
    // dead-code-eliminates none of them — fine for correctness; the JIT
    // assembler can prune unused values when it lowers PTX -> SASS.
    for op in &spec.ops {
        match op {
            Op::Store { .. } => continue,
            other => emit_op(&mut b, other, &input_ptrs, &input_validity_ptrs, &tid)?,
        }
    }

    // -------- Write the mask byte for this row.
    //
    // The predicate register is a `b32` holding 0 or 1 (see `emit_binary` for
    // comparisons / `emit_const` for Bool literals). We narrow it to u8 with
    // `cvt.u8.s32` (using s32 view because the bool register class is "r" /
    // .b32, but `cvt` accepts both signed and unsigned source types
    // interchangeably for 0/1 values), then store one byte at
    // `mask_ptr + tid`.
    let pred_phys = b.alloc.get(predicate_reg)?.to_string();

    // Compute byte offset = tid (1-byte stride), zero-extend to 64 bits, add
    // to base. `cvt.u64.u32` is the canonical "widen unsigned tid to b64" —
    // simpler than `mul.wide.u32 ..., tid, 1` and avoids a multiplier slot.
    let off = b.alloc.alloc("rd");
    let addr = b.alloc.alloc("rd");
    b.emit(&format!("cvt.u64.u32 {}, {};", off, tid))?;
    b.emit(&format!("add.s64 {}, {}, {};", addr, mask_ptr, off))?;

    // Narrow the 0/1 b32 value into a b16 temp, since PTX `st.global.u8`
    // accepts a b16 source register. We use class "r" only for b32 values, so
    // we have to introduce a small b16 temp class. Simpler: use the existing
    // `selp.b16` form by materializing via a predicate.
    //
    // Concretely:
    //   setp.ne.s32  %pX, <pred_phys>, 0;
    //   selp.b16     %rsY, 1, 0, %pX;
    //   st.global.u8 [addr], %rsY;
    let p_mask = b.alloc.alloc("p");
    let rs_mask = b.alloc.alloc("rs");
    b.emit(&format!("setp.ne.s32 {}, {}, 0;", p_mask, pred_phys))?;
    b.emit(&format!("selp.b16 {}, 1, 0, {};", rs_mask, p_mask))?;
    b.emit(&format!("st.global.u8 [{}], {};", addr, rs_mask))?;

    // -------- DONE label + return.
    b.emit_label("DONE")?;
    b.emit("ret;")?;

    // -------- Assemble the final module.
    let mut out = String::new();
    writeln!(out, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(out, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(out, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(out).map_err(write_err)?;

    write_signature(&mut out, &b, spec.inputs.len(), n_input_validity)?;

    writeln!(out, "{{").map_err(write_err)?;
    write_reg_decls(&mut out, &b.alloc)?;
    out.push_str(&b.body);
    writeln!(out, "}}").map_err(write_err)?;

    Ok(out)
}

/// Lower a single non-Store op into PTX. Mirrors `ptx_gen::emit_op` for the
/// compute subset, including the `Op::IsNullCheck` validity load (Batch 7,
/// IS NULL e2e).
fn emit_op(
    b: &mut PtxBuilder,
    op: &Op,
    input_ptrs: &[String],
    input_validity_ptrs: &[Option<String>],
    tid: &str,
) -> BoltResult<()> {
    match op {
        Op::LoadColumn { dst, col_idx, dtype } => {
            emit_load(b, *dst, *col_idx, *dtype, input_ptrs, tid)
        }
        Op::Const { dst, lit } => emit_const(b, *dst, lit),
        Op::Cast { dst, src, from, to } => emit_cast(b, *dst, *src, *from, *to),
        Op::Binary {
            dst,
            op,
            lhs,
            rhs,
            dtype,
            result_dtype,
        } => emit_binary(b, *dst, *op, *lhs, *rhs, *dtype, *result_dtype),
        Op::Store { .. } => Err(BoltError::Other(
            "scan_kernel: emit_op should not be called with a Store op".into(),
        )),
        // Predicate-only kernel's IsNullCheck arm: mirror `ptx_gen::
        // emit_is_null_check`. The host side
        // (`crate::exec::compact::launch_predicate_kernel`) pushes the
        // flagged-input validity pointers AFTER the mask output, in
        // input-slot order; `compile_predicate_kernel` loaded them into
        // `input_validity_ptrs[i]` so this op just reads byte `tid` from
        // the right slot.
        Op::IsNullCheck {
            dst,
            validity_input,
            want_null,
        } => emit_is_null_check(
            b,
            *dst,
            *validity_input,
            *want_null,
            input_validity_ptrs,
            tid,
        ),
    }
}

/// Emit PTX for `Op::IsNullCheck`: load the validity byte at row `tid` from
/// `input_validity_ptrs[validity_input]` and produce a Bool (0/1) in `dst`.
///
/// Wire shape (matches `ptx_gen::emit_is_null_check`):
///
/// ```text
///   cvt.s64.s32 %off,  %tid                 // widen row index to b64
///   add.s64     %addr, %vptr, %off          // &validity[tid]
///   ld.global.nc.u8 %byte, [%addr]          // 0=null, 1=non-null
///   setp.eq.u32 %p,    %byte, 0             // (or setp.ne for IS NOT NULL)
///   selp.s32    %dst,  1, 0, %p             // 0/1 Bool result
/// ```
///
/// # Errors
///
/// Returns `BoltError::Other` if `validity_input` is out of range for
/// `input_validity_ptrs`, or the slot is `None` (the spec was built without
/// `KernelSpec::input_has_validity` set for this column — a planner bug;
/// `Codegen::emit_unary` flips the flag whenever it emits this op).
fn emit_is_null_check(
    b: &mut PtxBuilder,
    dst: Reg,
    validity_input: usize,
    want_null: bool,
    input_validity_ptrs: &[Option<String>],
    tid: &str,
) -> BoltResult<()> {
    if validity_input >= input_validity_ptrs.len() {
        return Err(BoltError::Other(format!(
            "scan_kernel: IsNullCheck validity_input {} out of range (have {} input slots)",
            validity_input,
            input_validity_ptrs.len()
        )));
    }
    let vptr = match &input_validity_ptrs[validity_input] {
        Some(p) => p.clone(),
        None => {
            return Err(BoltError::Other(format!(
                "scan_kernel: IsNullCheck on input {} but KernelSpec::input_has_validity \
                 has no validity pointer wired through — planner bug \
                 (Codegen::emit_unary must flip input_has_validity[{}] = true)",
                validity_input, validity_input
            )));
        }
    };

    let off = b.alloc.alloc("rd");
    let addr = b.alloc.alloc("rd");
    let byte_reg = b.alloc.alloc("r");
    b.emit(&format!("cvt.s64.s32 {}, {};", off, tid))?;
    b.emit(&format!("add.s64 {}, {}, {};", addr, vptr, off))?;
    b.emit(&format!("ld.global.nc.u8 {}, [{}];", byte_reg, addr))?;

    let dst_name = b.alloc.assign(dst, DataType::Bool)?;
    let pred = b.alloc.alloc("p");
    let cmp = if want_null { "setp.eq.u32" } else { "setp.ne.u32" };
    b.emit(&format!("{} {}, {}, 0;", cmp, pred, byte_reg))?;
    b.emit(&format!("selp.s32 {}, 1, 0, {};", dst_name, pred))?;
    Ok(())
}

/// Emit a typed `ld.global.<ty>` of input column `col_idx` at row `tid`.
fn emit_load(
    b: &mut PtxBuilder,
    dst: Reg,
    col_idx: usize,
    dtype: DataType,
    input_ptrs: &[String],
    tid: &str,
) -> BoltResult<()> {
    if col_idx >= input_ptrs.len() {
        return Err(BoltError::Other(format!(
            "scan_kernel: LoadColumn col_idx {} out of range (have {} inputs)",
            col_idx,
            input_ptrs.len()
        )));
    }
    let width = byte_width(dtype)?;
    let off = b.alloc.alloc("rd");
    let addr = b.alloc.alloc("rd");
    b.emit(&format!("mul.wide.u32 {}, {}, {};", off, tid, width))?;
    b.emit(&format!(
        "add.s64 {}, {}, {};",
        addr, input_ptrs[col_idx], off
    ))?;
    let dst_name = b.alloc.assign(dst, dtype)?;
    let suffix = ld_st_suffix(dtype)?;
    b.emit(&format!("ld.global.{} {}, [{}];", suffix, dst_name, addr))?;
    Ok(())
}

/// Emit a `mov` of an immediate into a fresh register typed by the literal.
fn emit_const(b: &mut PtxBuilder, dst: Reg, lit: &Literal) -> BoltResult<()> {
    match lit {
        Literal::Null => Err(BoltError::Other(
            "scan_kernel: NULL literal not supported".into(),
        )),
        Literal::Utf8(_) => Err(BoltError::Other(
            "scan_kernel: Utf8 literal not supported".into(),
        )),
        Literal::Decimal128(..) => Err(BoltError::Plan(
            "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
        )),
        Literal::Bool(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Bool)?;
            let n: u32 = if *v { 1 } else { 0 };
            b.emit(&format!("mov.b32 {}, {};", dst_name, n))
        }
        Literal::Int32(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Int32)?;
            b.emit(&format!("mov.s32 {}, {};", dst_name, *v as i64))
        }
        Literal::Int64(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Int64)?;
            b.emit(&format!("mov.s64 {}, {};", dst_name, v))
        }
        Literal::Float32(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Float32)?;
            b.emit(&format!("mov.f32 {}, 0f{:08X};", dst_name, v.to_bits()))
        }
        Literal::Float64(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Float64)?;
            b.emit(&format!("mov.f64 {}, 0d{:016X};", dst_name, v.to_bits()))
        }
    }
}

/// Emit a `cvt` (or trivial `mov`) realizing `from -> to` on `src`.
fn emit_cast(
    b: &mut PtxBuilder,
    dst: Reg,
    src: Reg,
    from: DataType,
    to: DataType,
) -> BoltResult<()> {
    let src_name = b.alloc.get(src)?.to_string();
    let dst_name = b.alloc.assign(dst, to)?;

    use DataType::*;
    let instr = match (from, to) {
        (a, c) if a == c => {
            let mov_ty = match to {
                Bool => "b32",
                Int32 => "s32",
                Int64 => "s64",
                Float32 => "f32",
                Float64 => "f64",
                Utf8 => {
                    return Err(BoltError::Other(
                        "scan_kernel: cannot cast Utf8".into(),
                    ))
                }
                Decimal128(_, _) => {
                    return Err(BoltError::Plan(
                        "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
                    ))
                }
            };
            format!("mov.{} {}, {};", mov_ty, dst_name, src_name)
        }

        (Int32, Int64) => format!("cvt.s64.s32 {}, {};", dst_name, src_name),
        (Int64, Int32) => format!("cvt.s32.s64 {}, {};", dst_name, src_name),

        (Bool, Int32) => format!("mov.b32 {}, {};", dst_name, src_name),
        (Bool, Int64) => format!("cvt.s64.s32 {}, {};", dst_name, src_name),
        (Bool, Float32) => format!("cvt.rn.f32.s32 {}, {};", dst_name, src_name),
        (Bool, Float64) => format!("cvt.rn.f64.s32 {}, {};", dst_name, src_name),

        (Int32, Bool) => {
            let p = b.alloc.alloc("p");
            b.emit(&format!("setp.ne.s32 {}, {}, 0;", p, src_name))?;
            format!("selp.s32 {}, 1, 0, {};", dst_name, p)
        }
        (Int64, Bool) => {
            let p = b.alloc.alloc("p");
            b.emit(&format!("setp.ne.s64 {}, {}, 0;", p, src_name))?;
            format!("selp.s32 {}, 1, 0, {};", dst_name, p)
        }
        (Float32, Bool) => {
            let p = b.alloc.alloc("p");
            b.emit(&format!("setp.ne.f32 {}, {}, 0f00000000;", p, src_name))?;
            format!("selp.s32 {}, 1, 0, {};", dst_name, p)
        }
        (Float64, Bool) => {
            let p = b.alloc.alloc("p");
            b.emit(&format!(
                "setp.ne.f64 {}, {}, 0d0000000000000000;",
                p, src_name
            ))?;
            format!("selp.s32 {}, 1, 0, {};", dst_name, p)
        }

        (Int32, Float32) => format!("cvt.rn.f32.s32 {}, {};", dst_name, src_name),
        (Int32, Float64) => format!("cvt.rn.f64.s32 {}, {};", dst_name, src_name),
        (Int64, Float32) => format!("cvt.rn.f32.s64 {}, {};", dst_name, src_name),
        (Int64, Float64) => format!("cvt.rn.f64.s64 {}, {};", dst_name, src_name),

        (Float32, Float64) => format!("cvt.f64.f32 {}, {};", dst_name, src_name),
        (Float64, Float32) => format!("cvt.rn.f32.f64 {}, {};", dst_name, src_name),

        (Float32, Int32) => format!("cvt.rzi.s32.f32 {}, {};", dst_name, src_name),
        (Float32, Int64) => format!("cvt.rzi.s64.f32 {}, {};", dst_name, src_name),
        (Float64, Int32) => format!("cvt.rzi.s32.f64 {}, {};", dst_name, src_name),
        (Float64, Int64) => format!("cvt.rzi.s64.f64 {}, {};", dst_name, src_name),

        (Utf8, _) | (_, Utf8) => {
            return Err(BoltError::Other(
                "scan_kernel: Utf8 casts not supported".into(),
            ))
        }

        (Decimal128(_, _), _) | (_, Decimal128(_, _)) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
            ))
        }

        _ => {
            return Err(BoltError::Other(format!(
                "scan_kernel: unhandled cast {:?} -> {:?}",
                from, to
            )))
        }
    };

    b.emit(&instr)
}

/// Emit a binary op (arithmetic, comparison, or logical).
fn emit_binary(
    b: &mut PtxBuilder,
    dst: Reg,
    op: BinaryOp,
    lhs: Reg,
    rhs: Reg,
    dtype: DataType,
    result_dtype: DataType,
) -> BoltResult<()> {
    let lhs_name = b.alloc.get(lhs)?.to_string();
    let rhs_name = b.alloc.get(rhs)?.to_string();

    use BinaryOp::*;
    match op {
        Add | Sub | Mul | Div => {
            if result_dtype != dtype {
                return Err(BoltError::Other(format!(
                    "scan_kernel: arithmetic op {:?} expected result == operand dtype, got {:?}/{:?}",
                    op, dtype, result_dtype
                )));
            }
            if !is_numeric(dtype) {
                return Err(BoltError::Other(format!(
                    "scan_kernel: arithmetic op {:?} requires numeric operands, got {:?}",
                    op, dtype
                )));
            }
            let dst_name = b.alloc.assign(dst, result_dtype)?;
            let mnemonic = arith_mnemonic(op, dtype)?;
            b.emit(&format!(
                "{} {}, {}, {};",
                mnemonic, dst_name, lhs_name, rhs_name
            ))
        }
        Eq | NotEq | Lt | LtEq | Gt | GtEq => {
            if result_dtype != DataType::Bool {
                return Err(BoltError::Other(format!(
                    "scan_kernel: comparison op {:?} must produce Bool, got {:?}",
                    op, result_dtype
                )));
            }
            let dst_name = b.alloc.assign(dst, DataType::Bool)?;
            let p = b.alloc.alloc("p");
            let cmp = cmp_mnemonic(op, dtype)?;
            b.emit(&format!("{} {}, {}, {};", cmp, p, lhs_name, rhs_name))?;
            b.emit(&format!("selp.s32 {}, 1, 0, {};", dst_name, p))
        }
        And | Or => {
            if dtype != DataType::Bool || result_dtype != DataType::Bool {
                return Err(BoltError::Other(format!(
                    "scan_kernel: logical op {:?} requires Bool operands, got {:?}",
                    op, dtype
                )));
            }
            let dst_name = b.alloc.assign(dst, DataType::Bool)?;
            let mnemonic = match op {
                And => "and.b32",
                Or => "or.b32",
                _ => unreachable!(),
            };
            b.emit(&format!(
                "{} {}, {}, {};",
                mnemonic, dst_name, lhs_name, rhs_name
            ))
        }
    }
}

/// Mnemonic string for an arithmetic op at a given dtype.
fn arith_mnemonic(op: BinaryOp, dtype: DataType) -> BoltResult<String> {
    use BinaryOp::*;
    use DataType::*;
    let s = match (op, dtype) {
        (Add, Int32) => "add.s32",
        (Add, Int64) => "add.s64",
        (Add, Float32) => "add.f32",
        (Add, Float64) => "add.f64",
        (Sub, Int32) => "sub.s32",
        (Sub, Int64) => "sub.s64",
        (Sub, Float32) => "sub.f32",
        (Sub, Float64) => "sub.f64",
        (Mul, Int32) => "mul.lo.s32",
        (Mul, Int64) => "mul.lo.s64",
        (Mul, Float32) => "mul.f32",
        (Mul, Float64) => "mul.f64",
        (Div, Int32) => "div.s32",
        (Div, Int64) => "div.s64",
        (Div, Float32) => "div.rn.f32",
        (Div, Float64) => "div.rn.f64",
        _ => {
            return Err(BoltError::Other(format!(
                "scan_kernel: unsupported arithmetic {:?} on {:?}",
                op, dtype
            )))
        }
    };
    Ok(s.to_string())
}

/// Mnemonic string for a comparison `setp` at a given operand dtype.
fn cmp_mnemonic(op: BinaryOp, dtype: DataType) -> BoltResult<String> {
    use BinaryOp::*;
    use DataType::*;
    let cond = match op {
        Eq => "eq",
        NotEq => "ne",
        Lt => "lt",
        LtEq => "le",
        Gt => "gt",
        GtEq => "ge",
        _ => {
            return Err(BoltError::Other(format!(
                "scan_kernel: not a comparison op: {:?}",
                op
            )))
        }
    };
    let ty = match dtype {
        Bool => "u32",
        Int32 => "s32",
        Int64 => "s64",
        Float32 => "f32",
        Float64 => "f64",
        Utf8 => {
            return Err(BoltError::Other(
                "scan_kernel: cannot compare Utf8".into(),
            ))
        }
        Decimal128(_, _) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
            ))
        }
    };
    Ok(format!("setp.{}.{}", cond, ty))
}

/// True for numeric (int or float) dtypes.
fn is_numeric(dtype: DataType) -> bool {
    matches!(
        dtype,
        DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64
    )
}

/// PTX type suffix used on `ld.global`/`st.global` for `dtype`.
fn ld_st_suffix(dtype: DataType) -> BoltResult<&'static str> {
    Ok(match dtype {
        DataType::Bool => "u8",
        DataType::Int32 => "s32",
        DataType::Int64 => "s64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Utf8 => {
            return Err(BoltError::Other(
                "scan_kernel: Utf8 not supported in PTX codegen".into(),
            ))
        }
        DataType::Decimal128(_, _) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
            ))
        }
    })
}

/// Byte width of `dtype`, or an error for variable-width types.
fn byte_width(dtype: DataType) -> BoltResult<usize> {
    dtype.byte_width().ok_or_else(|| {
        BoltError::Other(format!("scan_kernel: variable-width dtype {:?}", dtype))
    })
}

/// Reject empty / whitespace-bearing kernel names that would break the PTX grammar.
fn validate_kernel_name(name: &str) -> BoltResult<()> {
    if name.is_empty() {
        return Err(BoltError::Other(
            "scan_kernel: kernel name must not be empty".into(),
        ));
    }
    let first = name.chars().next().unwrap_or('\0');
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(BoltError::Other(format!(
            "scan_kernel: kernel name '{}' must start with a letter or underscore",
            name
        )));
    }
    for c in name.chars() {
        if !(c.is_ascii_alphanumeric() || c == '_') {
            return Err(BoltError::Other(format!(
                "scan_kernel: kernel name '{}' contains illegal character '{}'",
                name, c
            )));
        }
    }
    Ok(())
}

/// Write the `.visible .entry` signature: N input ptrs, mask output ptr, K
/// input-validity ptrs (one per flagged input), n_rows.
fn write_signature(
    out: &mut String,
    b: &PtxBuilder,
    n_inputs: usize,
    n_input_validity: usize,
) -> BoltResult<()> {
    writeln!(out, ".visible .entry {}(", b.kernel_name).map_err(write_err)?;

    // N input column pointers + 1 mask output pointer + K input-validity
    // pointers, all `.u64`. When `n_input_validity == 0` the param block
    // reduces to the historical (N + 1) `.u64` slots, preserving binary
    // compatibility for the legacy no-validity callers.
    let total_ptr_params = n_inputs + 1 + n_input_validity;
    for i in 0..total_ptr_params {
        writeln!(out, "\t.param .u64 {},", b.param_name(i)).map_err(write_err)?;
    }
    // n_rows is u32, no trailing comma.
    writeln!(
        out,
        "\t.param .u32 {}",
        b.n_rows_param_name(n_inputs, n_input_validity)
    )
    .map_err(write_err)?;
    writeln!(out, ")").map_err(write_err)?;
    Ok(())
}

/// Emit the `.reg` declaration block sized to each class's used count.
fn write_reg_decls(out: &mut String, alloc: &RegAlloc) -> BoltResult<()> {
    // (class, ptx_type) pairs in deterministic emission order. `rs` is the
    // 16-bit register class we use for the narrowed mask byte source.
    let decls: [(&str, &str); 7] = [
        ("p", "pred"),
        ("rs", "b16"),
        ("r", "b32"),
        ("rl", "b64"),
        ("f", "f32"),
        ("fd", "f64"),
        ("rd", "b64"),
    ];
    for (class, ty) in decls {
        let n = alloc.count(class);
        if n > 0 {
            writeln!(out, "\t.reg .{} %{}<{}>;", ty, class, n).map_err(write_err)?;
        }
    }
    Ok(())
}

/// Adapt a `std::fmt::Error` into a `BoltError`.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("scan_kernel: write failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::Literal;
    use crate::plan::physical_plan::ColumnIO;

    /// Build the spec for `WHERE region_id = 1` against a single Int32 column.
    /// Mirrors the IR that `lower_projection` would produce for the predicate.
    fn region_eq_1_spec() -> KernelSpec {
        // r0 = LoadColumn(region_id : Int32)
        // r1 = Const(Int32(1))
        // r2 = Binary(Eq, r0, r1) -> Bool
        let ops = vec![
            Op::LoadColumn {
                dst: Reg(0),
                col_idx: 0,
                dtype: DataType::Int32,
            },
            Op::Const {
                dst: Reg(1),
                lit: Literal::Int32(1),
            },
            Op::Binary {
                dst: Reg(2),
                op: BinaryOp::Eq,
                lhs: Reg(0),
                rhs: Reg(1),
                dtype: DataType::Int32,
                result_dtype: DataType::Bool,
            },
        ];
        KernelSpec {
            inputs: vec![ColumnIO {
                name: "region_id".to_string(),
                dtype: DataType::Int32,
            }],
            // The predicate-only kernel ignores outputs; supply none.
            outputs: vec![],
            ops,
            predicate: Some(Reg(2)),
            register_count: 3,
            // No validity bitmap in test fixtures.
            input_has_validity: vec![],
            output_has_validity: vec![],
        }
    }

    #[test]
    fn requires_predicate() {
        let mut spec = region_eq_1_spec();
        spec.predicate = None;
        let err = compile_predicate_kernel(&spec, "bolt_predicate")
            .expect_err("must reject spec without a predicate");
        let msg = format!("{}", err);
        assert!(msg.contains("requires a predicate"), "got: {msg}");
    }

    #[test]
    fn region_eq_1_smoke() {
        let spec = region_eq_1_spec();
        let ptx = compile_predicate_kernel(&spec, "bolt_predicate")
            .expect("PTX codegen should succeed for a trivial integer predicate");

        // Header sanity.
        assert!(ptx.contains(".version 7.5"));
        assert!(ptx.contains(".target sm_70"));
        assert!(ptx.contains(".address_size 64"));

        // Signature: one input pointer, one mask pointer, one u32 n_rows.
        assert!(ptx.contains(".visible .entry bolt_predicate("));
        assert!(ptx.contains(".param .u64 bolt_predicate_param_0,"));
        assert!(ptx.contains(".param .u64 bolt_predicate_param_1,"));
        assert!(ptx.contains(".param .u32 bolt_predicate_param_2_n_rows"));

        // Body: the predicate eq + the mask byte store.
        assert!(ptx.contains("setp.eq.s32"));
        assert!(ptx.contains("selp.s32"));
        assert!(ptx.contains("selp.b16"));
        assert!(ptx.contains("st.global.u8"));
        assert!(ptx.contains("DONE:"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn rejects_utf8_input() {
        let mut spec = region_eq_1_spec();
        spec.inputs[0].dtype = DataType::Utf8;
        let err = compile_predicate_kernel(&spec, "bolt_predicate")
            .expect_err("must reject Utf8 inputs");
        assert!(format!("{}", err).contains("Utf8"));
    }

    #[test]
    fn rejects_bad_kernel_name() {
        let spec = region_eq_1_spec();
        let err = compile_predicate_kernel(&spec, "1bad")
            .expect_err("must reject kernel names that don't start with letter/underscore");
        assert!(format!("{}", err).contains("must start with"));
    }
}
