// SPDX-License-Identifier: Apache-2.0

//! PTX codegen: lower a `KernelSpec` into a complete PTX module string.

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
struct RegAlloc {
    /// Next free index per class (e.g. "r" -> 5 means `%r5` is next).
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
        self.mapping
            .get(&reg)
            .map(|s| s.as_str())
            .ok_or_else(|| BoltError::Other(format!("ptx_gen: undefined register {:?}", reg)))
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
                    "Utf8 not supported in PTX codegen yet".into(),
                ))
            }
            DataType::Decimal128(_, _) => {
                return Err(BoltError::Plan(
                    "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
                ))
            }
            DataType::Date32 | DataType::Timestamp(_, _) => {
                return Err(BoltError::Other(
                    "Date/Timestamp not yet lowered to GPU".into(),
                ))
            }
        })
    }

    /// Current count of registers issued for `class` (used to size `.reg` decls).
    fn count(&self, class: RegClass) -> u32 {
        *self.next.get(class).unwrap_or(&0)
    }
}

/// Internal codegen state: the in-progress kernel body and its register allocator.
struct PtxBuilder {
    /// Allocator covering value classes (r/rl/f/fd) and helpers (p/rd).
    alloc: RegAlloc,
    /// Body instructions (everything between the `.reg` block and the trailing brace).
    body: String,
    /// Kernel entry-point name (used to mangle `.param` identifiers).
    kernel_name: String,
}

impl PtxBuilder {
    /// New builder for a kernel with the given entry name.
    fn new(kernel_name: &str) -> Self {
        Self {
            alloc: RegAlloc::new(),
            body: String::new(),
            kernel_name: kernel_name.to_string(),
        }
    }

    /// Append one already-formatted PTX line (with leading tab, no trailing newline).
    fn emit(&mut self, line: &str) -> BoltResult<()> {
        writeln!(self.body, "\t{}", line)
            .map_err(|e| BoltError::Other(format!("ptx_gen: write failed: {}", e)))
    }

    /// Append a label (no leading tab) at column zero.
    fn emit_label(&mut self, label: &str) -> BoltResult<()> {
        writeln!(self.body, "{}:", label)
            .map_err(|e| BoltError::Other(format!("ptx_gen: write failed: {}", e)))
    }

    /// Build the mangled `.param` identifier for the `i`th parameter.
    fn param_name(&self, i: usize) -> String {
        format!("{}_param_{}", self.kernel_name, i)
    }

    /// Build the mangled `.param` identifier for the row-count parameter.
    ///
    /// The row-count param sits AFTER the value pointers AND any optional
    /// validity pointers — callers must pass `n_extra_validity_params` so
    /// the index matches the host-side parameter list.
    fn n_rows_param_name(
        &self,
        n_inputs: usize,
        n_outputs: usize,
        n_extra_validity_params: usize,
    ) -> String {
        format!(
            "{}_param_{}_n_rows",
            self.kernel_name,
            n_inputs + n_outputs + n_extra_validity_params
        )
    }
}

/// Compile a `KernelSpec` to a complete PTX module.
#[tracing::instrument(name = "codegen", level = "info", skip(spec), fields(kernel = kernel_name))]
pub fn compile(spec: &KernelSpec, kernel_name: &str) -> BoltResult<String> {
    validate_kernel_name(kernel_name)?;

    // -------- Validity wiring (Option B). The `input_has_validity` /
    //          `output_has_validity` fields are opt-in: when both are empty
    //          we emit the historical PTX shape verbatim (no extra params,
    //          no validity loads / stores) and every existing caller
    //          continues to work bit-for-bit. When set, they MUST be
    //          parallel to `inputs` / `outputs`.
    let input_valid: Vec<bool> = if spec.input_has_validity.is_empty() {
        vec![false; spec.inputs.len()]
    } else {
        if spec.input_has_validity.len() != spec.inputs.len() {
            return Err(BoltError::Other(format!(
                "ptx_gen: input_has_validity len {} != inputs len {}",
                spec.input_has_validity.len(),
                spec.inputs.len()
            )));
        }
        spec.input_has_validity.clone()
    };
    let output_valid: Vec<bool> = if spec.output_has_validity.is_empty() {
        vec![false; spec.outputs.len()]
    } else {
        if spec.output_has_validity.len() != spec.outputs.len() {
            return Err(BoltError::Other(format!(
                "ptx_gen: output_has_validity len {} != outputs len {}",
                spec.output_has_validity.len(),
                spec.outputs.len()
            )));
        }
        spec.output_has_validity.clone()
    };
    let n_input_validity: usize = input_valid.iter().filter(|b| **b).count();
    let n_output_validity: usize = output_valid.iter().filter(|b| **b).count();
    let n_extra_validity_params: usize = n_input_validity + n_output_validity;

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
    b.emit(&format!(
        "mad.lo.s32 {}, {}, {}, {};",
        tid, ctaid, ntid, tid_x
    ))?;
    b.emit(&format!(
        "ld.param.u32 {}, [{}];",
        n_rows,
        b.n_rows_param_name(spec.inputs.len(), spec.outputs.len(), n_extra_validity_params)
    ))?;
    b.emit(&format!(
        "setp.ge.u32 {}, {}, {};",
        pred_oob, tid, n_rows
    ))?;
    b.emit(&format!("@{} bra DONE;", pred_oob))?;

    // -------- Load and globalize all column base pointers (inputs then outputs).
    let mut input_ptrs: Vec<String> = Vec::with_capacity(spec.inputs.len());
    for (i, col) in spec.inputs.iter().enumerate() {
        // Reject Utf8 inputs eagerly — even if no LoadColumn op references them, we cannot lower.
        if matches!(col.dtype, DataType::Utf8) {
            return Err(BoltError::Other(
                "Utf8 not supported in PTX codegen yet".into(),
            ));
        }
        let rd = b.alloc.alloc("rd");
        b.emit(&format!("ld.param.u64 {}, [{}];", rd, b.param_name(i)))?;
        b.emit(&format!("cvta.to.global.u64 {}, {};", rd, rd))?;
        input_ptrs.push(rd);
    }

    let mut output_ptrs: Vec<String> = Vec::with_capacity(spec.outputs.len());
    let base = spec.inputs.len();
    for (i, col) in spec.outputs.iter().enumerate() {
        if matches!(col.dtype, DataType::Utf8) {
            return Err(BoltError::Other(
                "Utf8 not supported in PTX codegen yet".into(),
            ));
        }
        let rd = b.alloc.alloc("rd");
        b.emit(&format!(
            "ld.param.u64 {}, [{}];",
            rd,
            b.param_name(base + i)
        ))?;
        b.emit(&format!("cvta.to.global.u64 {}, {};", rd, rd))?;
        output_ptrs.push(rd);
    }

    // -------- (Option B) Load validity pointers in the order they appear
    //          in the param list: all flagged-input validities first, then
    //          all flagged-output validities. The host side
    //          (`agg_with_pre.rs::run_pre_stage`) builds the param list in
    //          the same order.
    let mut input_validity_ptrs: Vec<Option<String>> = vec![None; spec.inputs.len()];
    let mut next_param = base + spec.outputs.len();
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
    let mut output_validity_ptrs: Vec<Option<String>> = vec![None; spec.outputs.len()];
    for (i, has) in output_valid.iter().enumerate() {
        if *has {
            let rd = b.alloc.alloc("rd");
            b.emit(&format!(
                "ld.param.u64 {}, [{}];",
                rd,
                b.param_name(next_param)
            ))?;
            b.emit(&format!("cvta.to.global.u64 {}, {};", rd, rd))?;
            output_validity_ptrs[i] = Some(rd);
            next_param += 1;
        }
    }

    // -------- Compute the combined input validity: AND of every flagged
    //          input's validity byte at row tid. This is a conservative
    //          per-output validity (every output is marked valid only if
    //          EVERY input row is valid). A finer per-output dataflow
    //          analysis is a Stage C follow-up; for the common case
    //          (`SUM(price * tax)` etc.) every input feeds every output,
    //          so AND-of-all is exact.
    //
    //          When no input carries validity we still need a register
    //          holding `1` to drive flagged output stores (e.g. a kernel
    //          whose inputs are all-valid but whose outputs nonetheless
    //          carry a validity column for downstream-shape reasons).
    let combined_valid: Option<String> = if n_input_validity == 0 && n_output_validity == 0 {
        None
    } else {
        let combined = b.alloc.alloc("r");
        b.emit(&format!("mov.b32 {}, 1;", combined))?;
        for (i, opt) in input_validity_ptrs.iter().enumerate() {
            let Some(vptr) = opt else { continue };
            let _ = i;
            // off = tid (u8 stride => offset = tid).
            let off = b.alloc.alloc("rd");
            let addr = b.alloc.alloc("rd");
            let byte_reg = b.alloc.alloc("r");
            b.emit(&format!("cvt.s64.s32 {}, {};", off, tid))?;
            b.emit(&format!(
                "add.s64 {}, {}, {};",
                addr, vptr, off
            ))?;
            // Input-validity bytes live in distinct param buffers (host side
            // allocates them as fresh `GpuVec<u8>`). They're read-only here, so
            // route the load through the read-only cache.
            b.emit(&format!(
                "ld.global.nc.u8 {}, [{}];",
                byte_reg, addr
            ))?;
            // AND combined with this validity byte. Both live in the b32
            // (r) register class with 0/1 values; `and.b32` matches the
            // pattern used by logical Bool ops (see emit_binary).
            b.emit(&format!(
                "and.b32 {}, {}, {};",
                combined, combined, byte_reg
            ))?;
        }
        Some(combined)
    };

    // -------- Split ops into "compute" and "store" so the predicate gate sits between them.
    let mut compute_ops: Vec<&Op> = Vec::with_capacity(spec.ops.len());
    let mut store_ops: Vec<&Op> = Vec::with_capacity(spec.outputs.len());
    for op in &spec.ops {
        match op {
            Op::Store { .. } => store_ops.push(op),
            _ => compute_ops.push(op),
        }
    }

    // Emit all compute ops (loads, consts, casts, binaries, null-checks).
    for op in compute_ops {
        emit_op(
            &mut b,
            op,
            &input_ptrs,
            &output_ptrs,
            &input_validity_ptrs,
            &tid,
        )?;
    }

    // Predicate gate (single branch skips every store) if requested.
    if let Some(pred_reg) = spec.predicate {
        let phys = b.alloc.get(pred_reg)?.to_string();
        let gate = b.alloc.alloc("p");
        b.emit(&format!("setp.eq.s32 {}, {}, 0;", gate, phys))?;
        b.emit(&format!("@{} bra DONE;", gate))?;
    }

    // Emit all stores.
    for op in store_ops {
        emit_op(
            &mut b,
            op,
            &input_ptrs,
            &output_ptrs,
            &input_validity_ptrs,
            &tid,
        )?;
    }

    // -------- (Option B) Per-output validity stores. Each flagged output
    //          receives the same combined input validity at row tid. This
    //          runs AFTER the value stores so a Stage C optimisation that
    //          skips the value math when validity is 0 has a single,
    //          obvious gate site.
    if let Some(combined) = &combined_valid {
        for (i, opt) in output_validity_ptrs.iter().enumerate() {
            let Some(vptr) = opt else { continue };
            let _ = i;
            let off = b.alloc.alloc("rd");
            let addr = b.alloc.alloc("rd");
            b.emit(&format!("cvt.s64.s32 {}, {};", off, tid))?;
            b.emit(&format!(
                "add.s64 {}, {}, {};",
                addr, vptr, off
            ))?;
            b.emit(&format!(
                "st.global.u8 [{}], {};",
                addr, combined
            ))?;
        }
    }

    // -------- Done label + return.
    b.emit_label("DONE")?;
    b.emit("ret;")?;

    // -------- Assemble the final module: header + signature + .reg decls + body + close.
    let mut out = String::new();
    writeln!(out, "{}", PTX_VERSION).map_err(write_err)?;
    writeln!(out, "{}", PTX_TARGET).map_err(write_err)?;
    writeln!(out, "{}", PTX_ADDRESS_SIZE).map_err(write_err)?;
    writeln!(out).map_err(write_err)?;

    write_signature(&mut out, &b, spec, n_extra_validity_params)?;

    writeln!(out, "{{").map_err(write_err)?;
    write_reg_decls(&mut out, &b.alloc)?;
    out.push_str(&b.body);
    writeln!(out, "}}").map_err(write_err)?;

    Ok(out)
}

/// Lower a single non-Store op (or a Store, addressing into the right output column).
fn emit_op(
    b: &mut PtxBuilder,
    op: &Op,
    input_ptrs: &[String],
    output_ptrs: &[String],
    input_validity_ptrs: &[Option<String>],
    tid: &str,
) -> BoltResult<()> {
    match op {
        Op::LoadColumn { dst, col_idx, dtype } => emit_load(b, *dst, *col_idx, *dtype, input_ptrs, tid),
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
        Op::Store { src, col_idx, dtype } => emit_store(b, *src, *col_idx, *dtype, output_ptrs, tid),
        Op::IsNullCheck {
            dst,
            validity_input,
            want_null,
        } => emit_is_null_check(b, *dst, *validity_input, *want_null, input_validity_ptrs, tid),
    }
}

/// Emit PTX for `Op::IsNullCheck`: load the validity byte for the current
/// row from `input_validity_ptrs[validity_input]` and produce a Bool (0/1)
/// in `dst` reflecting the IS [NOT] NULL outcome.
///
/// Wire shape:
///
/// ```text
///   cvt.s64.s32 %off,  %tid                 // widen row index to b64
///   add.s64     %addr, %vptr, %off          // &validity[tid]
///   ld.global.nc.u8 %byte, [%addr]          // 0=null, 1=non-null
///   setp.eq.u32 %p,    %byte, 0             // (or setp.ne for IS NOT NULL)
///   selp.s32    %dst,  1, 0, %p             // 0/1 Bool result
/// ```
///
/// For `want_null == true` (`IS NULL`) we emit `setp.eq.u32` so the
/// predicate fires when the byte is 0. For `want_null == false`
/// (`IS NOT NULL`) we emit `setp.ne.u32` so the predicate fires when
/// the byte is non-zero. The `ld.global.nc.u8` form matches the
/// read-only-cache hint used by the rest of the pre-kernel loads.
///
/// # Errors
///
/// Returns `BoltError::Other` if `validity_input` is out of range for
/// `input_validity_ptrs`, or if the slot is `None` (the kernel was built
/// without `KernelSpec::input_has_validity` set for this column — a
/// planner bug; the codegen in `physical_plan::Codegen::emit_unary`
/// flips the flag whenever it emits this op).
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
            "ptx_gen: IsNullCheck validity_input {} out of range (have {} input validity slots)",
            validity_input,
            input_validity_ptrs.len()
        )));
    }
    let vptr = match &input_validity_ptrs[validity_input] {
        Some(p) => p.clone(),
        None => {
            return Err(BoltError::Other(format!(
                "ptx_gen: IsNullCheck on input {} but KernelSpec::input_has_validity \
                 has no validity pointer wired through — planner bug \
                 (Codegen::emit_unary must flip input_has_validity[{}] = true)",
                validity_input, validity_input
            )));
        }
    };

    // Address arithmetic: validity bitmap is a parallel `*u8` of length
    // n_rows where byte `tid` carries 0 = NULL, 1 = non-null (matching
    // the Option-B convention used by the AND-of-inputs fold above).
    let off = b.alloc.alloc("rd");
    let addr = b.alloc.alloc("rd");
    let byte_reg = b.alloc.alloc("r");
    b.emit(&format!("cvt.s64.s32 {}, {};", off, tid))?;
    b.emit(&format!("add.s64 {}, {}, {};", addr, vptr, off))?;
    b.emit(&format!("ld.global.nc.u8 {}, [{}];", byte_reg, addr))?;

    // Predicate + Bool result. `setp.{eq,ne}.u32` is the right typed
    // comparator for the b32 byte_reg above (zero-extended from the u8
    // load). `selp.s32` materialises the 0/1 Bool in the b32 class to
    // match the existing Bool ABI (see `RegAlloc::class_for(Bool)`).
    let dst_name = b.alloc.assign(dst, DataType::Bool)?;
    let pred = b.alloc.alloc("p");
    let cmp = if want_null { "setp.eq.u32" } else { "setp.ne.u32" };
    b.emit(&format!("{} {}, {}, 0;", cmp, pred, byte_reg))?;
    b.emit(&format!("selp.s32 {}, 1, 0, {};", dst_name, pred))?;
    Ok(())
}

/// Emit a `ld.global.nc.<type>` of input column `col_idx` at row `tid` into a fresh register.
///
/// Uses `ld.global.nc` (non-coherent / read-only cache) because every kernel-param
/// pointer is declared `.ptr .global .restrict` (see `write_signature`): the planner
/// guarantees that input column buffers never alias any output buffer of the same
/// kernel. The read-only cache path takes L2 pressure off shared scalar loads on
/// sm_70+. `ld.global.nc` returns the same bytes as `ld.global`; only the cache
/// hierarchy differs.
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
            "ptx_gen: LoadColumn col_idx {} out of range (have {} inputs)",
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
    b.emit(&format!("ld.global.nc.{} {}, [{}];", suffix, dst_name, addr))?;
    Ok(())
}

/// Emit a `st.global.<type>` of register `src` to output column `col_idx` at row `tid`.
fn emit_store(
    b: &mut PtxBuilder,
    src: Reg,
    col_idx: usize,
    dtype: DataType,
    output_ptrs: &[String],
    tid: &str,
) -> BoltResult<()> {
    if col_idx >= output_ptrs.len() {
        return Err(BoltError::Other(format!(
            "ptx_gen: Store col_idx {} out of range (have {} outputs)",
            col_idx,
            output_ptrs.len()
        )));
    }
    let width = byte_width(dtype)?;
    let off = b.alloc.alloc("rd");
    let addr = b.alloc.alloc("rd");
    let src_name = b.alloc.get(src)?.to_string();
    b.emit(&format!("mul.wide.u32 {}, {}, {};", off, tid, width))?;
    b.emit(&format!(
        "add.s64 {}, {}, {};",
        addr, output_ptrs[col_idx], off
    ))?;
    let suffix = ld_st_suffix(dtype)?;
    b.emit(&format!("st.global.{} [{}], {};", suffix, addr, src_name))?;
    Ok(())
}

/// Emit a `mov` of an immediate into a fresh register typed by the literal.
///
/// SECURITY: literals are emitted as **hex bit-patterns** (e.g. `0x{:08X}` for
/// 32-bit, `0x{:016X}` for 64-bit) so no attacker-controlled value can produce
/// PTX with characters other than `[0-9A-F]`. PTX `mov.s32`/`mov.s64` is a
/// bitwise copy, so reading back the value as signed is sound — `0xFFFFFFFF`
/// loaded into an `.s32` register is `-1`, identical to writing `-1` directly.
/// Float literals are likewise hex-encoded (PTX `0f...` / `0d...` syntax).
/// This closes the codegen-injection class even if a future planner regression
/// lets attacker-controlled SQL values reach this function.
fn emit_const(b: &mut PtxBuilder, dst: Reg, lit: &Literal) -> BoltResult<()> {
    match lit {
        Literal::Null => Err(BoltError::Other(
            "ptx_gen: NULL literal not supported".into(),
        )),
        Literal::Utf8(_) => Err(BoltError::Other(
            "ptx_gen: Utf8 literal not supported".into(),
        )),
        Literal::Decimal128(..) => Err(BoltError::Plan(
            "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
        )),
        Literal::Date32(_) | Literal::Timestamp(_, _, _) => Err(BoltError::Other(
            "Date/Timestamp not yet lowered to GPU".into(),
        )),
        Literal::Bool(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Bool)?;
            // Value space is {0, 1}; not an injection surface, but keep the
            // emission consistent with the other integer paths for clarity.
            let n: u32 = if *v { 1 } else { 0 };
            b.emit(&format!("mov.b32 {}, {};", dst_name, n))
        }
        Literal::Int32(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Int32)?;
            // Emit the bit-pattern as hex: `mov.s32` is a bitwise copy, so
            // `0xFFFFFFFF` here is -1, identical to writing `-1`. This avoids
            // any sign / INT32_MIN parsing concerns AND removes the codegen-
            // injection surface (output is restricted to `[0-9A-F]`).
            b.emit(&format!("mov.s32 {}, 0x{:08X};", dst_name, *v as u32))
        }
        Literal::Int64(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Int64)?;
            b.emit(&format!("mov.s64 {}, 0x{:016X};", dst_name, *v as u64))
        }
        Literal::Float32(v) => {
            // Already hex-encoded via PTX `0f<8 hex>` syntax — no injection surface.
            let dst_name = b.alloc.assign(dst, DataType::Float32)?;
            b.emit(&format!("mov.f32 {}, 0f{:08X};", dst_name, v.to_bits()))
        }
        Literal::Float64(v) => {
            // Already hex-encoded via PTX `0d<16 hex>` syntax — no injection surface.
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
        // Same type -> typed mov of the appropriate width.
        (a, c) if a == c => {
            let mov_ty = match to {
                Bool => "b32",
                Int32 => "s32",
                Int64 => "s64",
                Float32 => "f32",
                Float64 => "f64",
                Utf8 => {
                    return Err(BoltError::Other(
                        "ptx_gen: cannot cast Utf8".into(),
                    ))
                }
                Decimal128(_, _) => {
                    return Err(BoltError::Plan(
                        "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
                    ))
                }
                Date32 | Timestamp(_, _) => {
                    return Err(BoltError::Other(
                        "Date/Timestamp not yet lowered to GPU".into(),
                    ))
                }
            };
            format!("mov.{} {}, {};", mov_ty, dst_name, src_name)
        }

        // Integer widening / narrowing.
        (Int32, Int64) => format!("cvt.s64.s32 {}, {};", dst_name, src_name),
        (Int64, Int32) => format!("cvt.s32.s64 {}, {};", dst_name, src_name),

        // Bool to numeric: zero-extend; bool register already holds 0/1 as a b32.
        (Bool, Int32) => format!("mov.b32 {}, {};", dst_name, src_name),
        (Bool, Int64) => format!("cvt.s64.s32 {}, {};", dst_name, src_name),
        (Bool, Float32) => format!("cvt.rn.f32.s32 {}, {};", dst_name, src_name),
        (Bool, Float64) => format!("cvt.rn.f64.s32 {}, {};", dst_name, src_name),

        // Numeric -> Bool: nonzero -> 1.
        (Int32, Bool) => {
            // src != 0 ? 1 : 0
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

        // Int -> Float.
        (Int32, Float32) => format!("cvt.rn.f32.s32 {}, {};", dst_name, src_name),
        (Int32, Float64) => format!("cvt.rn.f64.s32 {}, {};", dst_name, src_name),
        (Int64, Float32) => format!("cvt.rn.f32.s64 {}, {};", dst_name, src_name),
        (Int64, Float64) => format!("cvt.rn.f64.s64 {}, {};", dst_name, src_name),

        // Float widening / narrowing.
        (Float32, Float64) => format!("cvt.f64.f32 {}, {};", dst_name, src_name),
        (Float64, Float32) => format!("cvt.rn.f32.f64 {}, {};", dst_name, src_name),

        // Float -> Int (round toward zero, then convert).
        (Float32, Int32) => format!("cvt.rzi.s32.f32 {}, {};", dst_name, src_name),
        (Float32, Int64) => format!("cvt.rzi.s64.f32 {}, {};", dst_name, src_name),
        (Float64, Int32) => format!("cvt.rzi.s32.f64 {}, {};", dst_name, src_name),
        (Float64, Int64) => format!("cvt.rzi.s64.f64 {}, {};", dst_name, src_name),

        (Utf8, _) | (_, Utf8) => {
            return Err(BoltError::Other(
                "ptx_gen: Utf8 casts not supported".into(),
            ))
        }

        (Decimal128(_, _), _) | (_, Decimal128(_, _)) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
            ))
        }

        // Unreachable: the `a == c` guard above already covers every
        // same-dtype pair, but rustc can't prove guard exhaustiveness.
        _ => {
            return Err(BoltError::Other(format!(
                "ptx_gen: internal — unhandled cast {:?} -> {:?}",
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
            // Arithmetic preserves the operand dtype; the spec already unified.
            if result_dtype != dtype {
                return Err(BoltError::Other(format!(
                    "ptx_gen: arithmetic op {:?} expected result dtype == operand dtype, got {:?}/{:?}",
                    op, dtype, result_dtype
                )));
            }
            if !is_numeric(dtype) {
                return Err(BoltError::Other(format!(
                    "ptx_gen: arithmetic op {:?} requires numeric operands, got {:?}",
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
                    "ptx_gen: comparison op {:?} must produce Bool, got {:?}",
                    op, result_dtype
                )));
            }
            let dst_name = b.alloc.assign(dst, DataType::Bool)?;
            let p = b.alloc.alloc("p");
            let cmp = cmp_mnemonic(op, dtype)?;
            b.emit(&format!(
                "{} {}, {}, {};",
                cmp, p, lhs_name, rhs_name
            ))?;
            b.emit(&format!("selp.s32 {}, 1, 0, {};", dst_name, p))
        }
        And | Or => {
            if dtype != DataType::Bool || result_dtype != DataType::Bool {
                return Err(BoltError::Other(format!(
                    "ptx_gen: logical op {:?} requires Bool operands, got {:?}",
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
        Concat => {
            // String concat lives entirely on the host (see
            // `crate::exec::string_ops::host_concat_strings`); the
            // physical-plan lowerer routes any expression that contains
            // `BinaryOp::Concat` through `PhysicalPlan::Project` (host
            // executor) instead of the fused GPU kernel. Reaching this
            // arm therefore indicates a missing route; surface a clear
            // error rather than emitting nonsense PTX.
            Err(BoltError::Other(
                "ptx_gen: string concat (||) is not lowered to GPU; \
                 the planner should route this through the host-side \
                 PhysicalPlan::Project executor"
                    .into(),
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
                "ptx_gen: unsupported arithmetic {:?} on {:?}",
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
                "ptx_gen: not a comparison op: {:?}",
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
                "ptx_gen: cannot compare Utf8".into(),
            ))
        }
        Decimal128(_, _) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
            ))
        }
        Date32 | Timestamp(_, _) => {
            return Err(BoltError::Other(
                "Date/Timestamp not yet lowered to GPU".into(),
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
                "Utf8 not supported in PTX codegen yet".into(),
            ))
        }
        DataType::Decimal128(_, _) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
            ))
        }
        DataType::Date32 | DataType::Timestamp(_, _) => {
            return Err(BoltError::Other(
                "Date/Timestamp not yet lowered to GPU".into(),
            ))
        }
    })
}

/// Byte width of `dtype`, or an error for variable-width types.
fn byte_width(dtype: DataType) -> BoltResult<usize> {
    dtype.byte_width().ok_or_else(|| {
        BoltError::Other(format!("ptx_gen: variable-width dtype {:?}", dtype))
    })
}

/// Reject empty / whitespace-bearing kernel names that would break the PTX grammar.
///
/// SECURITY: also rejects PTX reserved identifiers (instruction mnemonics, type
/// suffixes, state-space keywords), names starting with `__` (compiler-reserved),
/// and any name containing `_param_` (would collide with synthesised parameter
/// names from `PtxBuilder::param_name`). PTX is case-sensitive, so the reject
/// list is matched case-sensitively.
fn validate_kernel_name(name: &str) -> BoltResult<()> {
    if name.is_empty() {
        return Err(BoltError::Other(
            "ptx_gen: kernel name must not be empty".into(),
        ));
    }
    let first = name.chars().next().unwrap_or('\0');
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(BoltError::Other(format!(
            "ptx_gen: kernel name '{}' must start with a letter or underscore",
            name
        )));
    }
    for c in name.chars() {
        if !(c.is_ascii_alphanumeric() || c == '_') {
            return Err(BoltError::Other(format!(
                "ptx_gen: kernel name '{}' contains illegal character '{}'",
                name, c
            )));
        }
    }

    // PTX reserved identifiers: instruction mnemonics, type suffixes, state-space
    // keywords, and control words. Using one of these as an `.entry` name would
    // either collide with the grammar outright or produce baffling assembler
    // errors downstream. Case-sensitive — PTX is case-sensitive.
    const RESERVED: &[&str] = &[
        "bra", "ret", "mov", "entry", "ld", "st", "add", "sub", "mul", "div",
        "mad", "cvt", "setp", "selp", "bar", "atom", "membar", "cvta", "shl",
        "shr", "and", "or", "xor", "not", "sin", "cos", "exp2", "lg2", "sqrt",
        "rsqrt", "rcp", "abs", "neg", "min", "max", "mma", "tex", "tld4",
        "wmma", "cp", "callp", "ret", "exit", "trap", "brkpt", "prefetch",
        "fma", "global", "shared", "local", "param", "const", "tex", "surf",
        "sm", "sreg", "reg", "b8", "b16", "b32", "b64", "u8", "u16", "u32",
        "u64", "s8", "s16", "s32", "s64", "f16", "f32", "f64", "pred",
    ];
    if RESERVED.iter().any(|r| *r == name) {
        return Err(BoltError::Other(format!(
            "ptx_gen: kernel name '{}' is a PTX reserved identifier",
            name
        )));
    }

    // Compiler-reserved: identifiers beginning with `__` are reserved for the
    // PTX toolchain (libdevice, NVVM intrinsics, etc.).
    if name.starts_with("__") {
        return Err(BoltError::Other(format!(
            "ptx_gen: kernel name '{}' starts with '__' (compiler-reserved)",
            name
        )));
    }

    // Would collide with synthesised parameter names like `_param_0`, `_param_1`.
    if name.contains("_param_") {
        return Err(BoltError::Other(format!(
            "ptx_gen: kernel name '{}' contains reserved substring '_param_'",
            name
        )));
    }

    Ok(())
}

/// Write the `.visible .entry` signature, one parameter per line.
fn write_signature(
    out: &mut String,
    b: &PtxBuilder,
    spec: &KernelSpec,
    n_extra_validity_params: usize,
) -> BoltResult<()> {
    writeln!(out, ".visible .entry {}(", b.kernel_name).map_err(write_err)?;

    let total_params = spec.inputs.len() + spec.outputs.len() + n_extra_validity_params;
    // NOTE: .ptr .global .restrict relies on the invariant that no two kernel-param
    // pointers alias. The PhysicalPlan lowering guarantees this — never reuse a
    // column buffer as both an input and a non-trivial output. Validity
    // pointers (when present) are fresh `GpuVec<u8>` allocations on the host
    // side, separate from any value buffer, so they also satisfy non-alias.
    // TODO(orchestrator): golden test update — tests/ptx_golden_tests.rs may need
    // its `.param .u64 ...` assertions widened to allow the new attribute string
    // (e.g. assert `contains(".restrict")`).
    for i in 0..total_params {
        let comma = ",";
        writeln!(
            out,
            "\t.param .u64 .ptr .global .align 16 {}{}",
            b.param_name(i),
            comma
        )
        .map_err(write_err)?;
    }
    // n_rows is u32, no trailing comma. Scalar param — no .ptr attributes.
    writeln!(
        out,
        "\t.param .u32 {}",
        b.n_rows_param_name(spec.inputs.len(), spec.outputs.len(), n_extra_validity_params)
    )
    .map_err(write_err)?;
    writeln!(out, ")").map_err(write_err)?;
    Ok(())
}

/// Emit the `.reg` declaration block sized to each class's used count.
fn write_reg_decls(out: &mut String, alloc: &RegAlloc) -> BoltResult<()> {
    // (class, ptx_type) pairs in deterministic emission order.
    let decls: [(&str, &str); 6] = [
        ("p", "pred"),
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
    BoltError::Other(format!("ptx_gen: write failed: {}", e))
}

#[cfg(test)]
mod validity_emission_tests {
    //! Golden tests for the Option B validity-propagation codegen. These
    //! live inline because they need to construct a `Reg` directly, and
    //! its tuple field is `pub(crate)`.
    use super::*;
    use crate::plan::logical_plan::BinaryOp;
    use crate::plan::physical_plan::{ColumnIO, KernelSpec, Op, Reg};

    /// Build a minimal hand-crafted `KernelSpec` for `out0 = in0 * in1`
    /// (Int64) with validity wired on both inputs and the single output.
    /// Mirrors what the planner would emit for `SUM(price * tax)`
    /// once the SQL frontend learns to set the validity flags from
    /// `arr.null_count() > 0` — for now the host side
    /// (`exec::agg_with_pre::run_pre_stage`) sets them per-call.
    fn mul_with_validity_spec() -> KernelSpec {
        let ops = vec![
            Op::LoadColumn {
                dst: Reg(0),
                col_idx: 0,
                dtype: DataType::Int64,
            },
            Op::LoadColumn {
                dst: Reg(1),
                col_idx: 1,
                dtype: DataType::Int64,
            },
            Op::Binary {
                dst: Reg(2),
                op: BinaryOp::Mul,
                lhs: Reg(0),
                rhs: Reg(1),
                dtype: DataType::Int64,
                result_dtype: DataType::Int64,
            },
            Op::Store {
                src: Reg(2),
                col_idx: 0,
                dtype: DataType::Int64,
            },
        ];
        KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "in0".into(),
                    dtype: DataType::Int64,
                },
                ColumnIO {
                    name: "in1".into(),
                    dtype: DataType::Int64,
                },
            ],
            outputs: vec![ColumnIO {
                name: "out0".into(),
                dtype: DataType::Int64,
            }],
            ops,
            predicate: None,
            register_count: 3,
            input_has_validity: vec![true, true],
            output_has_validity: vec![true],
        }
    }

    #[test]
    fn validity_emits_and_b32_and_u8_store() {
        // Contract:
        //   1. Each flagged input contributes an `ld.global.nc.u8` for its
        //      per-row validity byte (read-only-cache hint — input
        //      validity buffers are guaranteed non-aliasing).
        //   2. The bytes are AND-folded into a single combined register
        //      via `and.b32` (booleans live in the b32 register class).
        //   3. The combined byte is written via `st.global.u8` to every
        //      flagged output's validity buffer.
        //   4. Param signature carries `n_inputs + n_outputs + n_flagged
        //      input/output validity` pointer params plus one `.u32`
        //      n_rows.
        let spec = mul_with_validity_spec();
        let ptx = compile(&spec, "bolt_pre_kernel_validity").expect("compile");

        // 2 u8 loads (one per flagged input) — routed through the read-only
        // cache via `ld.global.nc.u8`.
        let n_u8_loads = ptx.matches("ld.global.nc.u8").count();
        assert!(
            n_u8_loads >= 2,
            "expected >=2 ld.global.nc.u8 for input validity, got {n_u8_loads}\n{ptx}"
        );

        // and.b32 for the combined-validity fold (Mul doesn't emit one).
        assert!(
            ptx.contains("and.b32"),
            "expected and.b32 for combined input validity\n{ptx}"
        );

        // st.global.u8 for the output validity write.
        let n_u8_stores = ptx.matches("st.global.u8").count();
        assert!(
            n_u8_stores >= 1,
            "expected >=1 st.global.u8 for output validity, got {n_u8_stores}\n{ptx}"
        );

        // 2 inputs + 1 output + 2 input-validity + 1 output-validity = 6
        // pointer params.
        let n_ptr_params = ptx.matches(".param .u64 .ptr").count();
        assert_eq!(
            n_ptr_params, 6,
            "expected 6 .ptr params (3 value + 3 validity), got {n_ptr_params}\n{ptx}"
        );
    }

    #[test]
    fn no_validity_emits_original_shape() {
        // Regression guard: when `*_has_validity` is empty the emitter
        // MUST produce the historical PTX shape (no extra params, no u8
        // loads/stores, original `n_rows` param index). The projection
        // path in `engine.rs` relies on this byte-for-byte
        // compatibility.
        let mut spec = mul_with_validity_spec();
        spec.input_has_validity = vec![];
        spec.output_has_validity = vec![];
        let ptx = compile(&spec, "bolt_pre_kernel_no_validity").expect("compile");

        let n_ptr_params = ptx.matches(".param .u64 .ptr").count();
        assert_eq!(
            n_ptr_params, 3,
            "expected 3 .ptr params (2 inputs + 1 output, no validity), got {n_ptr_params}\n{ptx}"
        );
        assert!(
            !ptx.contains("ld.global.nc.u8"),
            "expected NO ld.global.nc.u8 in the no-validity path\n{ptx}"
        );
        assert!(
            !ptx.contains("st.global.u8"),
            "expected NO st.global.u8 in the no-validity path\n{ptx}"
        );
    }

    #[test]
    fn validity_param_count_mismatch_is_error() {
        // Defensive: a non-empty `input_has_validity` of the wrong length
        // is a planning bug. The emitter must surface it rather than
        // silently produce a kernel with a desynchronised param list.
        let mut spec = mul_with_validity_spec();
        spec.input_has_validity = vec![true]; // should be len 2
        let err = compile(&spec, "bolt_pre_kernel_bad").expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("input_has_validity") || msg.contains("validity"),
            "error should mention validity, got: {msg}"
        );
    }

    /// PTX-shape coverage for the new `Op::IsNullCheck` op (Batch 5).
    ///
    /// Builds a hand-crafted spec that selects an Int32 column and writes
    /// the `IS NULL` result for a flagged-nullable input as a Bool output.
    /// The PTX must:
    ///
    ///   1. Carry one extra `.param .u64 .ptr` for the input's validity
    ///      buffer (no output validity, no AND-fold).
    ///   2. Issue an `ld.global.nc.u8` for the validity byte at row `tid`.
    ///   3. Compare the byte to zero with `setp.eq.u32` (IS NULL — fire
    ///      when validity = 0).
    ///   4. Materialise the 0/1 result with `selp.s32 ..., 1, 0, ...`.
    ///
    /// Mirrors the contract documented on `Op::IsNullCheck` in
    /// `physical_plan.rs`.
    #[test]
    fn is_null_check_emits_validity_load_and_setp() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".into(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "x_is_null".into(),
                dtype: DataType::Bool,
            }],
            ops: vec![
                // The codegen always emits a LoadColumn for the bare-column
                // operand (cache miss path); mirror that so the IR shape is
                // realistic. The loaded value register is unused — the
                // IsNullCheck reads validity, not value.
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
                Op::IsNullCheck {
                    dst: Reg(1),
                    validity_input: 0,
                    want_null: true,
                },
                Op::Store {
                    src: Reg(1),
                    col_idx: 0,
                    dtype: DataType::Bool,
                },
            ],
            predicate: None,
            register_count: 2,
            // The codegen sets this when emitting IsNullCheck; we mirror
            // it manually here so the kernel param list grows by one for
            // the validity pointer.
            input_has_validity: vec![true],
            output_has_validity: vec![],
        };

        let ptx = compile(&spec, "bolt_is_null_check").expect("compile");

        // One extra .ptr param for the input validity pointer:
        // 1 input value + 1 output value + 1 input validity = 3.
        let n_ptr_params = ptx.matches(".param .u64 .ptr").count();
        assert_eq!(
            n_ptr_params, 3,
            "expected 3 .ptr params (1 input + 1 output + 1 validity), got {n_ptr_params}\n{ptx}"
        );

        // The body must contain the read-only-cache validity load.
        assert!(
            ptx.contains("ld.global.nc.u8"),
            "expected ld.global.nc.u8 for validity byte load\n{ptx}"
        );

        // The IS NULL predicate is `byte == 0`.
        assert!(
            ptx.contains("setp.eq.u32"),
            "expected setp.eq.u32 for IS NULL (validity == 0)\n{ptx}"
        );

        // 0/1 materialisation.
        assert!(
            ptx.contains("selp.s32"),
            "expected selp.s32 to materialise the Bool 0/1 result\n{ptx}"
        );
    }

    /// `IS NOT NULL` should swap `setp.eq.u32` for `setp.ne.u32` —
    /// otherwise the PTX shape is identical to the IS NULL case.
    #[test]
    fn is_not_null_check_uses_setp_ne() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".into(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "x_is_not_null".into(),
                dtype: DataType::Bool,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
                Op::IsNullCheck {
                    dst: Reg(1),
                    validity_input: 0,
                    want_null: false,
                },
                Op::Store {
                    src: Reg(1),
                    col_idx: 0,
                    dtype: DataType::Bool,
                },
            ],
            predicate: None,
            register_count: 2,
            input_has_validity: vec![true],
            output_has_validity: vec![],
        };

        let ptx = compile(&spec, "bolt_is_not_null_check").expect("compile");

        assert!(
            ptx.contains("ld.global.nc.u8"),
            "expected ld.global.nc.u8 for validity byte load\n{ptx}"
        );
        assert!(
            ptx.contains("setp.ne.u32"),
            "IS NOT NULL must use setp.ne.u32 (fire when validity != 0)\n{ptx}"
        );
        // The IS NULL form must NOT appear — otherwise the want_null=false
        // branch silently degraded to want_null=true.
        assert!(
            !ptx.contains("setp.eq.u32"),
            "IS NOT NULL must NOT contain setp.eq.u32 (would invert semantics)\n{ptx}"
        );
    }

    /// Planner-bug guard: an `IsNullCheck` referring to a `validity_input`
    /// slot that wasn't flagged in `KernelSpec::input_has_validity` must
    /// surface as an error from `compile`, not silently produce bad PTX.
    #[test]
    fn is_null_check_without_validity_flag_is_error() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".into(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "x_is_null".into(),
                dtype: DataType::Bool,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
                Op::IsNullCheck {
                    dst: Reg(1),
                    validity_input: 0,
                    want_null: true,
                },
                Op::Store {
                    src: Reg(1),
                    col_idx: 0,
                    dtype: DataType::Bool,
                },
            ],
            predicate: None,
            register_count: 2,
            // Forget to flag input 0 — the kernel won't have a validity
            // pointer for it, so the IsNullCheck has nothing to read.
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let err = compile(&spec, "bolt_is_null_unflagged").expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("IsNullCheck") || msg.contains("validity"),
            "error should mention validity wiring, got: {msg}"
        );
    }
}


// ---------------------------------------------------------------------------
// PV-stage-d: per-output validity dataflow analysis.
//
// The pre-stage emitter previously ANDed the validity of every flagged input
// into every output column's validity bit. That over-approximated the true
// dataflow: a `Store(col=2)` that only reads inputs 0 and 1 should not need
// input 3's bitmap. This module exposes the corrected analysis so callers
// (the pre kernel codegen, the predicate kernel codegen) can emit a
// narrower AND-tree per output.
// ---------------------------------------------------------------------------

/// For each `Store` op in `spec.ops`, compute the set of `LoadColumn`
/// `col_idx` values it transitively depends on. Returns a `Vec<Vec<usize>>`
/// parallel to `spec.outputs` — `result[i]` is the (sorted, deduplicated)
/// list of input column ordinals whose validity feeds output `i`'s
/// validity bit.
///
/// The walk is a backward def-use sweep over `spec.ops` keyed by `Reg`:
///
/// 1. Build a `Reg -> Op` index from `spec.ops` (each register is
///    written exactly once — SSA).
/// 2. For each `Store { src, col_idx }`, do a DFS from `src` collecting
///    every `LoadColumn::col_idx` reached.
/// 3. The result is the input set the AND-tree should reference for
///    output `col_idx`.
///
/// The result is sorted + deduplicated so downstream PTX emission
/// produces deterministic output regardless of HashMap iteration order.
///
/// # Caller responsibilities
///
/// The caller must intersect this with `spec.input_has_validity` — an input
/// dependency only contributes to the AND-tree if that input actually
/// carries a NULL bitmap. Doing the intersection outside this function
/// keeps the analysis purely structural (testable without a provider).
///
/// # Output ordering
///
/// `result[i]` corresponds to `spec.outputs[i]` (i.e. `Store { col_idx: i }`).
/// If multiple `Store`s target the same `col_idx` (the IR doesn't currently
/// emit that, but defensively): the result merges all of their dependencies.
/// If no `Store` targets `col_idx`, the result is an empty Vec for that
/// position (output has no validity dependencies — vacuously valid).
pub fn output_input_dependencies(
    spec: &crate::plan::physical_plan::KernelSpec,
) -> Vec<Vec<usize>> {
    use crate::plan::physical_plan::Op;

    // (a) Map every produced Reg to the Op that produced it. Since the IR
    // is SSA each Reg appears as `dst` exactly once.
    let mut reg_to_op: HashMap<u32, &Op> = HashMap::with_capacity(spec.ops.len());
    for op in &spec.ops {
        match op {
            Op::LoadColumn { dst, .. }
            | Op::Const { dst, .. }
            | Op::Cast { dst, .. }
            | Op::Binary { dst, .. }
            | Op::IsNullCheck { dst, .. } => {
                reg_to_op.insert(dst.id(), op);
            }
            Op::Store { .. } => { /* no dst */ }
        }
    }

    // (b) Pre-allocate one Vec per output. `spec.outputs.len()` is the
    // declared output count; in practice every output has a matching
    // Store, but defaulting to empty preserves correctness if the IR
    // is ever incomplete.
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); spec.outputs.len()];

    for op in &spec.ops {
        if let Op::Store { src, col_idx, .. } = op {
            if *col_idx >= deps.len() {
                // Defensive: a Store referencing an unknown output index
                // is a planner bug — skip rather than panic so codegen
                // can surface the real diagnostic elsewhere.
                continue;
            }
            let mut found: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            let mut stack: Vec<u32> = vec![src.id()];
            let mut visited: std::collections::HashSet<u32> =
                std::collections::HashSet::new();
            while let Some(r) = stack.pop() {
                if !visited.insert(r) {
                    continue;
                }
                let producer = match reg_to_op.get(&r) {
                    Some(o) => *o,
                    None => continue, // dangling reg — IR bug, skip.
                };
                match producer {
                    Op::LoadColumn { col_idx, .. } => {
                        found.insert(*col_idx);
                    }
                    Op::Const { .. } => { /* leaf — no input dep */ }
                    Op::Cast { src, .. } => {
                        stack.push(src.id());
                    }
                    Op::Binary { lhs, rhs, .. } => {
                        stack.push(lhs.id());
                        stack.push(rhs.id());
                    }
                    Op::IsNullCheck { .. } => {
                        // IS NULL / IS NOT NULL is itself never-null: it
                        // turns a (value, validity) pair into a Bool
                        // {0,1} that already encodes the NULL outcome.
                        // From a per-output validity AND-tree standpoint
                        // it acts as a leaf with no upstream input-VALUE
                        // dependency — even though the op reads its
                        // input's validity bitmap, that read does NOT
                        // need to be folded into a downstream output's
                        // validity bit.
                    }
                    Op::Store { .. } => {
                        // Stores don't produce a Reg, so reg_to_op can't
                        // return one. Unreachable in practice.
                    }
                }
            }
            // Merge into the per-output set (sorted + dedup at the end).
            for c in found {
                if !deps[*col_idx].contains(&c) {
                    deps[*col_idx].push(c);
                }
            }
        }
    }

    for v in &mut deps {
        v.sort_unstable();
    }
    deps
}

#[cfg(test)]
mod dataflow_tests {
    use super::*;
    use crate::plan::logical_plan::{BinaryOp, Literal};
    use crate::plan::physical_plan::{ColumnIO, KernelSpec, Op, Reg};

    /// `KernelSpec::ops = [Load(0), Load(1), Add(0,1) -> r2, Store(r2 -> 0)]`:
    /// the single output's dependency set should be exactly `{0, 1}`,
    /// not "every flagged input".
    #[test]
    fn output_deps_add_two_loads() {
        let spec = KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "a".into(),
                    dtype: DataType::Int32,
                },
                ColumnIO {
                    name: "b".into(),
                    dtype: DataType::Int32,
                },
                ColumnIO {
                    name: "c".into(),
                    dtype: DataType::Int32,
                },
            ],
            outputs: vec![ColumnIO {
                name: "ab".into(),
                dtype: DataType::Int32,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
                Op::LoadColumn {
                    dst: Reg(1),
                    col_idx: 1,
                    dtype: DataType::Int32,
                },
                Op::Binary {
                    dst: Reg(2),
                    op: BinaryOp::Add,
                    lhs: Reg(0),
                    rhs: Reg(1),
                    dtype: DataType::Int32,
                    result_dtype: DataType::Int32,
                },
                Op::Store {
                    src: Reg(2),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
            ],
            predicate: None,
            register_count: 3,
            // Every input flagged — still, the analysis should only
            // include col 0 and col 1, NOT col 2.
            input_has_validity: vec![true, true, true],
            output_has_validity: vec![],
        };
        let deps = output_input_dependencies(&spec);
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0], vec![0, 1], "output 0 should depend on inputs 0 and 1 only");
    }

    /// Two outputs touching disjoint inputs must produce disjoint dep sets.
    #[test]
    fn output_deps_two_disjoint_stores() {
        let spec = KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "a".into(),
                    dtype: DataType::Int32,
                },
                ColumnIO {
                    name: "b".into(),
                    dtype: DataType::Int32,
                },
            ],
            outputs: vec![
                ColumnIO {
                    name: "a_out".into(),
                    dtype: DataType::Int32,
                },
                ColumnIO {
                    name: "b_out".into(),
                    dtype: DataType::Int32,
                },
            ],
            ops: vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
                Op::LoadColumn {
                    dst: Reg(1),
                    col_idx: 1,
                    dtype: DataType::Int32,
                },
                Op::Store {
                    src: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
                Op::Store {
                    src: Reg(1),
                    col_idx: 1,
                    dtype: DataType::Int32,
                },
            ],
            predicate: None,
            register_count: 2,
            input_has_validity: vec![true, true],
            output_has_validity: vec![],
        };
        let deps = output_input_dependencies(&spec);
        assert_eq!(deps.len(), 2);
        assert_eq!(deps[0], vec![0], "output 0 should depend only on input 0");
        assert_eq!(deps[1], vec![1], "output 1 should depend only on input 1");
    }

    /// A `Const` leaf has no input dependencies — output that only writes
    /// a constant should have an empty dep set.
    #[test]
    fn output_deps_const_only() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "a".into(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "k".into(),
                dtype: DataType::Int32,
            }],
            ops: vec![
                Op::Const {
                    dst: Reg(0),
                    lit: Literal::Int32(42),
                },
                Op::Store {
                    src: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
            ],
            predicate: None,
            register_count: 1,
            input_has_validity: vec![true],
            output_has_validity: vec![],
        };
        let deps = output_input_dependencies(&spec);
        assert_eq!(deps.len(), 1);
        assert!(
            deps[0].is_empty(),
            "constant-only output should have no input deps, got {:?}",
            deps[0]
        );
    }

    /// `Cast` is transparent for the analysis — depends on whatever its
    /// `src` depends on.
    #[test]
    fn output_deps_walk_through_cast() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".into(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "y".into(),
                dtype: DataType::Float64,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
                Op::Cast {
                    dst: Reg(1),
                    src: Reg(0),
                    from: DataType::Int32,
                    to: DataType::Float64,
                },
                Op::Store {
                    src: Reg(1),
                    col_idx: 0,
                    dtype: DataType::Float64,
                },
            ],
            predicate: None,
            register_count: 2,
            input_has_validity: vec![true],
            output_has_validity: vec![],
        };
        let deps = output_input_dependencies(&spec);
        assert_eq!(deps[0], vec![0]);
    }
}
