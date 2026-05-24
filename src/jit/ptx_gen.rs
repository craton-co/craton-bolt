// SPDX-License-Identifier: Apache-2.0

//! PTX codegen: lower a `KernelSpec` into a complete PTX module string.

use std::collections::HashMap;
use std::fmt::Write;

use crate::error::{JavelinError, JavelinResult};
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
    fn assign(&mut self, reg: Reg, dtype: DataType) -> JavelinResult<String> {
        let class = Self::class_for(dtype)?;
        let name = self.alloc(class);
        self.mapping.insert(reg, name.clone());
        Ok(name)
    }

    /// Look up the physical register name previously assigned to `reg`.
    fn get(&self, reg: Reg) -> JavelinResult<&str> {
        self.mapping
            .get(&reg)
            .map(|s| s.as_str())
            .ok_or_else(|| JavelinError::Other(format!("ptx_gen: undefined register {:?}", reg)))
    }

    /// Map a logical dtype to a PTX register class string.
    fn class_for(dtype: DataType) -> JavelinResult<RegClass> {
        Ok(match dtype {
            DataType::Bool => "r",
            DataType::Int32 => "r",
            DataType::Int64 => "rl",
            DataType::Float32 => "f",
            DataType::Float64 => "fd",
            DataType::Utf8 => {
                return Err(JavelinError::Other(
                    "Utf8 not supported in PTX codegen yet".into(),
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
    fn emit(&mut self, line: &str) -> JavelinResult<()> {
        writeln!(self.body, "\t{}", line)
            .map_err(|e| JavelinError::Other(format!("ptx_gen: write failed: {}", e)))
    }

    /// Append a label (no leading tab) at column zero.
    fn emit_label(&mut self, label: &str) -> JavelinResult<()> {
        writeln!(self.body, "{}:", label)
            .map_err(|e| JavelinError::Other(format!("ptx_gen: write failed: {}", e)))
    }

    /// Build the mangled `.param` identifier for the `i`th parameter.
    fn param_name(&self, i: usize) -> String {
        format!("{}_param_{}", self.kernel_name, i)
    }

    /// Build the mangled `.param` identifier for the row-count parameter.
    fn n_rows_param_name(&self, n_inputs: usize, n_outputs: usize) -> String {
        format!(
            "{}_param_{}_n_rows",
            self.kernel_name,
            n_inputs + n_outputs
        )
    }
}

/// Compile a `KernelSpec` to a complete PTX module.
pub fn compile(spec: &KernelSpec, kernel_name: &str) -> JavelinResult<String> {
    validate_kernel_name(kernel_name)?;

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
        b.n_rows_param_name(spec.inputs.len(), spec.outputs.len())
    ))?;
    b.emit(&format!(
        "setp.ge.s32 {}, {}, {};",
        pred_oob, tid, n_rows
    ))?;
    b.emit(&format!("@{} bra DONE;", pred_oob))?;

    // -------- Load and globalize all column base pointers (inputs then outputs).
    let mut input_ptrs: Vec<String> = Vec::with_capacity(spec.inputs.len());
    for (i, col) in spec.inputs.iter().enumerate() {
        // Reject Utf8 inputs eagerly — even if no LoadColumn op references them, we cannot lower.
        if matches!(col.dtype, DataType::Utf8) {
            return Err(JavelinError::Other(
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
            return Err(JavelinError::Other(
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

    // -------- Split ops into "compute" and "store" so the predicate gate sits between them.
    let mut compute_ops: Vec<&Op> = Vec::with_capacity(spec.ops.len());
    let mut store_ops: Vec<&Op> = Vec::with_capacity(spec.outputs.len());
    for op in &spec.ops {
        match op {
            Op::Store { .. } => store_ops.push(op),
            _ => compute_ops.push(op),
        }
    }

    // Emit all compute ops (loads, consts, casts, binaries).
    for op in compute_ops {
        emit_op(&mut b, op, &input_ptrs, &output_ptrs, &tid)?;
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
        emit_op(&mut b, op, &input_ptrs, &output_ptrs, &tid)?;
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

    write_signature(&mut out, &b, spec)?;

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
    tid: &str,
) -> JavelinResult<()> {
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
    }
}

/// Emit a `ld.global.<type>` of input column `col_idx` at row `tid` into a fresh register.
fn emit_load(
    b: &mut PtxBuilder,
    dst: Reg,
    col_idx: usize,
    dtype: DataType,
    input_ptrs: &[String],
    tid: &str,
) -> JavelinResult<()> {
    if col_idx >= input_ptrs.len() {
        return Err(JavelinError::Other(format!(
            "ptx_gen: LoadColumn col_idx {} out of range (have {} inputs)",
            col_idx,
            input_ptrs.len()
        )));
    }
    let width = byte_width(dtype)?;
    let off = b.alloc.alloc("rd");
    let addr = b.alloc.alloc("rd");
    b.emit(&format!("mul.wide.s32 {}, {}, {};", off, tid, width))?;
    b.emit(&format!(
        "add.s64 {}, {}, {};",
        addr, input_ptrs[col_idx], off
    ))?;
    let dst_name = b.alloc.assign(dst, dtype)?;
    let suffix = ld_st_suffix(dtype)?;
    b.emit(&format!("ld.global.{} {}, [{}];", suffix, dst_name, addr))?;
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
) -> JavelinResult<()> {
    if col_idx >= output_ptrs.len() {
        return Err(JavelinError::Other(format!(
            "ptx_gen: Store col_idx {} out of range (have {} outputs)",
            col_idx,
            output_ptrs.len()
        )));
    }
    let width = byte_width(dtype)?;
    let off = b.alloc.alloc("rd");
    let addr = b.alloc.alloc("rd");
    let src_name = b.alloc.get(src)?.to_string();
    b.emit(&format!("mul.wide.s32 {}, {}, {};", off, tid, width))?;
    b.emit(&format!(
        "add.s64 {}, {}, {};",
        addr, output_ptrs[col_idx], off
    ))?;
    let suffix = ld_st_suffix(dtype)?;
    b.emit(&format!("st.global.{} [{}], {};", suffix, addr, src_name))?;
    Ok(())
}

/// Emit a `mov` of an immediate into a fresh register typed by the literal.
fn emit_const(b: &mut PtxBuilder, dst: Reg, lit: &Literal) -> JavelinResult<()> {
    match lit {
        Literal::Null => Err(JavelinError::Other(
            "ptx_gen: NULL literal not supported".into(),
        )),
        Literal::Utf8(_) => Err(JavelinError::Other(
            "ptx_gen: Utf8 literal not supported".into(),
        )),
        Literal::Bool(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Bool)?;
            let n: u32 = if *v { 1 } else { 0 };
            b.emit(&format!("mov.b32 {}, {};", dst_name, n))
        }
        Literal::Int32(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Int32)?;
            // Format via i64 so INT32_MIN's `-` parses as a unary on a 32-bit literal.
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
) -> JavelinResult<()> {
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
                    return Err(JavelinError::Other(
                        "ptx_gen: cannot cast Utf8".into(),
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
            return Err(JavelinError::Other(
                "ptx_gen: Utf8 casts not supported".into(),
            ))
        }

        // Unreachable: the `a == c` guard above already covers every
        // same-dtype pair, but rustc can't prove guard exhaustiveness.
        _ => {
            return Err(JavelinError::Other(format!(
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
) -> JavelinResult<()> {
    let lhs_name = b.alloc.get(lhs)?.to_string();
    let rhs_name = b.alloc.get(rhs)?.to_string();

    use BinaryOp::*;
    match op {
        Add | Sub | Mul | Div => {
            // Arithmetic preserves the operand dtype; the spec already unified.
            if result_dtype != dtype {
                return Err(JavelinError::Other(format!(
                    "ptx_gen: arithmetic op {:?} expected result dtype == operand dtype, got {:?}/{:?}",
                    op, dtype, result_dtype
                )));
            }
            if !is_numeric(dtype) {
                return Err(JavelinError::Other(format!(
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
                return Err(JavelinError::Other(format!(
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
                return Err(JavelinError::Other(format!(
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
    }
}

/// Mnemonic string for an arithmetic op at a given dtype.
fn arith_mnemonic(op: BinaryOp, dtype: DataType) -> JavelinResult<String> {
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
            return Err(JavelinError::Other(format!(
                "ptx_gen: unsupported arithmetic {:?} on {:?}",
                op, dtype
            )))
        }
    };
    Ok(s.to_string())
}

/// Mnemonic string for a comparison `setp` at a given operand dtype.
fn cmp_mnemonic(op: BinaryOp, dtype: DataType) -> JavelinResult<String> {
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
            return Err(JavelinError::Other(format!(
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
            return Err(JavelinError::Other(
                "ptx_gen: cannot compare Utf8".into(),
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
fn ld_st_suffix(dtype: DataType) -> JavelinResult<&'static str> {
    Ok(match dtype {
        DataType::Bool => "u8",
        DataType::Int32 => "s32",
        DataType::Int64 => "s64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Utf8 => {
            return Err(JavelinError::Other(
                "Utf8 not supported in PTX codegen yet".into(),
            ))
        }
    })
}

/// Byte width of `dtype`, or an error for variable-width types.
fn byte_width(dtype: DataType) -> JavelinResult<usize> {
    dtype.byte_width().ok_or_else(|| {
        JavelinError::Other(format!("ptx_gen: variable-width dtype {:?}", dtype))
    })
}

/// Reject empty / whitespace-bearing kernel names that would break the PTX grammar.
fn validate_kernel_name(name: &str) -> JavelinResult<()> {
    if name.is_empty() {
        return Err(JavelinError::Other(
            "ptx_gen: kernel name must not be empty".into(),
        ));
    }
    let first = name.chars().next().unwrap_or('\0');
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(JavelinError::Other(format!(
            "ptx_gen: kernel name '{}' must start with a letter or underscore",
            name
        )));
    }
    for c in name.chars() {
        if !(c.is_ascii_alphanumeric() || c == '_') {
            return Err(JavelinError::Other(format!(
                "ptx_gen: kernel name '{}' contains illegal character '{}'",
                name, c
            )));
        }
    }
    Ok(())
}

/// Write the `.visible .entry` signature, one parameter per line.
fn write_signature(out: &mut String, b: &PtxBuilder, spec: &KernelSpec) -> JavelinResult<()> {
    writeln!(out, ".visible .entry {}(", b.kernel_name).map_err(write_err)?;

    let total_params = spec.inputs.len() + spec.outputs.len();
    for i in 0..total_params {
        let comma = ",";
        writeln!(out, "\t.param .u64 {}{}", b.param_name(i), comma).map_err(write_err)?;
    }
    // n_rows is u32, no trailing comma.
    writeln!(
        out,
        "\t.param .u32 {}",
        b.n_rows_param_name(spec.inputs.len(), spec.outputs.len())
    )
    .map_err(write_err)?;
    writeln!(out, ")").map_err(write_err)?;
    Ok(())
}

/// Emit the `.reg` declaration block sized to each class's used count.
fn write_reg_decls(out: &mut String, alloc: &RegAlloc) -> JavelinResult<()> {
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

/// Adapt a `std::fmt::Error` into a `JavelinError`.
fn write_err(e: std::fmt::Error) -> JavelinError {
    JavelinError::Other(format!("ptx_gen: write failed: {}", e))
}
