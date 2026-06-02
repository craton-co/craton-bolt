// SPDX-License-Identifier: Apache-2.0

//! PTX codegen: lower a `KernelSpec` into a complete PTX module string.

use std::collections::HashMap;
use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{BinaryOp, DataType, Literal};
use crate::plan::physical_plan::{KernelSpec, Op, Reg};

/// PTX target metadata baked into every emitted module.
///
/// Exposed `pub(crate)` so the on-disk / in-process cache salt
/// ([`crate::jit::disk_cache::codegen_salt`]) can fold the PTX ISA
/// `.version` and the `.target` arch string into the cache key. See the
/// `JIT-arch` note on `codegen_salt`: the target is a hardcoded `sm_70`
/// today, but folding it into the salt means that if the target ever
/// becomes device-derived, cached kernels can never be mis-routed across
/// GPU architectures (a key written under `sm_70` won't be served to an
/// `sm_90` process).
pub(crate) const PTX_VERSION: &str = ".version 7.5";
/// Target SM architecture string.
pub(crate) const PTX_TARGET: &str = ".target sm_70";
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

    /// Allocate a pair of adjacent `b64` registers for a 128-bit (Decimal128 /
    /// i128) value split into `lo` / `hi` halves. The PTX side has no native
    /// 128-bit register class, so v0.7 represents an i128 value as two SSA
    /// `Reg`s in the `rl` class; `assign_pair` issues both at once so the
    /// `(lo_index, hi_index)` pair stays contiguous in the emitted `.reg`
    /// block (helps SASS see the temporary as a single live range without
    /// affecting correctness).
    ///
    /// Returns `(lo_name, hi_name)`. Both registers are tracked in
    /// `RegAlloc::mapping` exactly as `assign` does so subsequent
    /// `RegAlloc::get` calls can resolve either half independently.
    fn assign_pair(&mut self, reg_lo: Reg, reg_hi: Reg) -> BoltResult<(String, String)> {
        // Use the existing `rl` (b64) class — every 128-bit op reads/writes
        // its halves with `ld.global.nc.u64`, `mov.u64`, `add.cc.u64`, etc.,
        // all of which take `rl` operands. Going through `alloc` keeps the
        // class-counter bookkeeping (used by `write_reg_decls`) consistent.
        let lo_name = self.alloc("rl");
        let hi_name = self.alloc("rl");
        self.mapping.insert(reg_lo, lo_name.clone());
        self.mapping.insert(reg_hi, hi_name.clone());
        Ok((lo_name, hi_name))
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
            // v0.7: Date32 / Timestamp lower to their underlying integer
            // register classes. Date32 is i32 days-since-epoch (`r` class);
            // Timestamp is i64 ticks-since-epoch in the source unit (`rl`
            // class). The logical dtype is preserved on the IR `Value` so
            // downstream type-checks still see the temporal type.
            DataType::Date32 => "r",
            DataType::Timestamp(_, _) => "rl",
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

/// Emit one PTX instruction line directly into the builder body.
///
/// PERF (codegen alloc): this is the allocation-free twin of
/// [`PtxBuilder::emit`]. `b.emit(&format!(...))` materialises a throwaway
/// `String` per instruction (one heap allocation each); for large specs that
/// is thousands of tiny allocations on the codegen hot path. This macro
/// `writeln!`s the formatted instruction *straight into* `b.body`, reusing the
/// existing buffer and never allocating an intermediate.
///
/// Byte-for-byte equivalence with `emit`: `emit` writes `"\t{}\n"` where `{}`
/// is the formatted instruction; here `concat!("\t", $fmt)` prepends the same
/// leading tab to the (always-literal) format string and `writeln!` appends the
/// same trailing newline. The emitted text is identical, so the PTX
/// golden/snapshot tests stay valid.
///
/// Because the format arguments are evaluated *inside* the `writeln!`, operand
/// names can be passed as `b.alloc.get(reg)?` (`&str`) borrows instead of
/// `.to_string()` clones: `$b.body` and `$b.alloc` are disjoint struct fields,
/// so the immutable `alloc` borrow coexists with the mutable `body` borrow for
/// the duration of the single write.
macro_rules! emit_fmt {
    ($b:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        writeln!($b.body, concat!("\t", $fmt) $(, $arg)*)
            .map_err(|e| BoltError::Other(format!("ptx_gen: write failed: {}", e)))
    };
}

/// Compile a `KernelSpec` to a complete PTX module.
///
/// # Row-count limit (C-3 / C-4)
///
/// The global thread id is computed in **signed 32-bit** arithmetic
/// (`mad.lo.s32 %tid, %ctaid, %ntid, %tid.x`, see TID setup below), so the
/// per-launch addressable row space is capped at `i32::MAX` (~2.1 billion
/// rows). All offset arithmetic — value loads/stores **and** validity-byte
/// loads/stores — now widens the row index **unsigned** (`mul.wide.u32`), so
/// addressing is internally consistent and correct for every `tid` the s32
/// `mad.lo` can produce. The host launch path MUST therefore ensure
/// `n_rows <= i32::MAX`; larger row counts require migrating the tid math to
/// 64-bit grid addressing (see review C-4).
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

    // PERF (codegen alloc): emit straight into `b.body` via `emit_fmt!`.
    emit_fmt!(b, "mov.u32 {}, %ctaid.x;", ctaid)?;
    emit_fmt!(b, "mov.u32 {}, %ntid.x;", ntid)?;
    emit_fmt!(b, "mov.u32 {}, %tid.x;", tid_x)?;
    emit_fmt!(b, "mad.lo.s32 {}, {}, {}, {};", tid, ctaid, ntid, tid_x)?;
    // `n_rows_param_name` borrows all of `&b`, which would overlap the
    // `&mut b.body` inside the macro — compute it into a local first.
    let n_rows_param =
        b.n_rows_param_name(spec.inputs.len(), spec.outputs.len(), n_extra_validity_params);
    emit_fmt!(b, "ld.param.u32 {}, [{}];", n_rows, n_rows_param)?;
    emit_fmt!(b, "setp.ge.u32 {}, {}, {};", pred_oob, tid, n_rows)?;
    emit_fmt!(b, "@{} bra DONE;", pred_oob)?;

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
        // PERF (codegen alloc): emit via `emit_fmt!`. `param_name` borrows all
        // of `&b`, which would overlap the macro's `&mut b.body`, so it is
        // computed into a local first.
        let pname = b.param_name(i);
        emit_fmt!(b, "ld.param.u64 {}, [{}];", rd, pname)?;
        emit_fmt!(b, "cvta.to.global.u64 {}, {};", rd, rd)?;
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
        // PERF (codegen alloc): `param_name` into a local (whole-`&b` borrow),
        // then emit via `emit_fmt!`.
        let pname = b.param_name(base + i);
        emit_fmt!(b, "ld.param.u64 {}, [{}];", rd, pname)?;
        emit_fmt!(b, "cvta.to.global.u64 {}, {};", rd, rd)?;
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
            // PERF (codegen alloc): `param_name` into a local, then `emit_fmt!`.
            let pname = b.param_name(next_param);
            emit_fmt!(b, "ld.param.u64 {}, [{}];", rd, pname)?;
            emit_fmt!(b, "cvta.to.global.u64 {}, {};", rd, rd)?;
            input_validity_ptrs[i] = Some(rd);
            next_param += 1;
        }
    }
    let mut output_validity_ptrs: Vec<Option<String>> = vec![None; spec.outputs.len()];
    for (i, has) in output_valid.iter().enumerate() {
        if *has {
            let rd = b.alloc.alloc("rd");
            // PERF (codegen alloc): `param_name` into a local, then `emit_fmt!`.
            let pname = b.param_name(next_param);
            emit_fmt!(b, "ld.param.u64 {}, [{}];", rd, pname)?;
            emit_fmt!(b, "cvta.to.global.u64 {}, {};", rd, rd)?;
            output_validity_ptrs[i] = Some(rd);
            next_param += 1;
        }
    }

    // -------- Per-output validity (issue B: precise NULL propagation).
    //
    //          Each flagged output's NULL mask is the AND of ONLY the inputs
    //          that output transitively depends on — not the AND of *every*
    //          kernel input. `output_input_dependencies` does the backward
    //          def-use walk from each `Store`'s source register down to the
    //          `LoadColumn`s it reaches, returning the (sorted) input column
    //          ordinals per output. We then keep only those that actually
    //          carry a validity bitmap (`input_validity_ptrs[c].is_some()`).
    //
    //          Correctness for the common single-output case is preserved
    //          byte-for-byte: when there is exactly one flagged output and it
    //          depends on every flagged input (e.g. `SUM(price * tax)`), the
    //          filtered dep-set equals "all flagged inputs in ascending column
    //          order", so the emitted AND-tree, register allocation order, and
    //          store are identical to the previous AND-of-all-inputs shape.
    //          Multi-output kernels and CASE branches now get a *tighter*
    //          (per-output) mask, which is a deliberate behavior change: an
    //          output is no longer NULLed by a NULL in an input it never reads.
    //
    //          When an output's filtered dep-set is empty (no flagged input
    //          feeds it) we still emit a register holding `1`, so a flagged
    //          output whose inputs are all-valid (or which carries a validity
    //          column purely for downstream-shape reasons) stores valid=1.
    //
    //          The AND-trees are emitted here (before the compute/store ops) so
    //          the store sites below stay a single, obvious gate point; the
    //          per-output combined register is recorded for the store loop.
    let output_deps = output_input_dependencies(spec);
    let mut output_combined: Vec<Option<String>> = vec![None; spec.outputs.len()];
    for (out_idx, opt_vptr) in output_validity_ptrs.iter().enumerate() {
        if opt_vptr.is_none() {
            // Output carries no validity bitmap -> nothing to compute/store.
            continue;
        }
        // Inputs this output depends on AND that carry a validity bitmap, in
        // ascending column order (the dep list is already sorted). Indexing
        // `output_deps` is bounds-safe: it is parallel to `spec.outputs`.
        let deps: &[usize] = output_deps.get(out_idx).map(Vec::as_slice).unwrap_or(&[]);
        let combined = b.alloc.alloc("r");
        // PERF (codegen alloc): emit straight into `b.body` via `emit_fmt!`.
        emit_fmt!(b, "mov.b32 {}, 1;", combined)?;
        for &c in deps {
            let Some(vptr) = input_validity_ptrs.get(c).and_then(|o| o.as_ref()) else {
                // Dependency on an input with no validity bitmap: it can never
                // be NULL, so it contributes nothing to the AND-tree.
                continue;
            };
            // off = tid (u8 stride => offset = tid).
            // C-3: widen the row index UNSIGNED (`mul.wide.u32 .. , 1`) so the
            // validity-byte offset matches the value-load path
            // (`emit_load`/`emit_load_128` use `mul.wide.u32`). A signed
            // `cvt.s64.s32` would sign-extend `tid` once it crosses 2^31 and
            // produce a huge negative offset → OOB validity load, while the
            // value load at the same row stays correct. `mul.wide.u32 _, 1`
            // zero-extends the 32-bit tid into the 64-bit offset register.
            let off = b.alloc.alloc("rd");
            let addr = b.alloc.alloc("rd");
            let byte_reg = b.alloc.alloc("r");
            emit_fmt!(b, "mul.wide.u32 {}, {}, 1;", off, tid)?;
            emit_fmt!(b, "add.s64 {}, {}, {};", addr, vptr, off)?;
            // Input-validity bytes live in distinct param buffers (host side
            // allocates them as fresh `GpuVec<u8>`). They're read-only here, so
            // route the load through the read-only cache.
            emit_fmt!(b, "ld.global.nc.u8 {}, [{}];", byte_reg, addr)?;
            // AND combined with this validity byte. Both live in the b32
            // (r) register class with 0/1 values; `and.b32` matches the
            // pattern used by logical Bool ops (see emit_binary).
            emit_fmt!(b, "and.b32 {}, {}, {};", combined, combined, byte_reg)?;
        }
        output_combined[out_idx] = Some(combined);
    }

    // -------- Split ops into "compute" and "store" so the predicate gate
    //          sits between them. `Store128` joins `Store` in the sink
    //          partition so the predicate gate also masks Decimal128 row
    //          writes (v0.7 Sub-task A).
    let mut compute_ops: Vec<&Op> = Vec::with_capacity(spec.ops.len());
    let mut store_ops: Vec<&Op> = Vec::with_capacity(spec.outputs.len());
    for op in &spec.ops {
        match op {
            Op::Store { .. } | Op::Store128 { .. } => store_ops.push(op),
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
        // PERF (codegen alloc): `pred_reg` name borrowed inline (no
        // `.to_string()`); the `gate` predicate is allocated first.
        let gate = b.alloc.alloc("p");
        emit_fmt!(b, "setp.eq.s32 {}, {}, 0;", gate, b.alloc.get(pred_reg)?)?;
        emit_fmt!(b, "@{} bra DONE;", gate)?;
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

    // -------- (Issue B) Per-output validity stores. Each flagged output
    //          stores ITS OWN combined mask (the AND of just the inputs it
    //          depends on, computed above) at row tid. This runs AFTER the
    //          value stores so a Stage C optimisation that skips the value
    //          math when validity is 0 has a single, obvious gate site.
    for (i, opt) in output_validity_ptrs.iter().enumerate() {
        let Some(vptr) = opt else { continue };
        // A flagged output always has a recorded combined register (the loop
        // above allocates one for every `Some` validity ptr); skip defensively
        // if somehow absent rather than emitting a store of an unset register.
        let Some(combined) = output_combined.get(i).and_then(|o| o.as_ref()) else {
            continue;
        };
        let off = b.alloc.alloc("rd");
        let addr = b.alloc.alloc("rd");
        // PERF (codegen alloc): emit straight into `b.body` via `emit_fmt!`;
        // `vptr`/`combined` are borrows of locals, not `b`.
        // C-3: UNSIGNED widen (`mul.wide.u32 _, 1`) to match the value path
        // and the AND-of-inputs fold above; see the input-validity comment.
        emit_fmt!(b, "mul.wide.u32 {}, {}, 1;", off, tid)?;
        emit_fmt!(b, "add.s64 {}, {}, {};", addr, vptr, off)?;
        emit_fmt!(b, "st.global.u8 [{}], {};", addr, combined)?;
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
        Op::Select {
            dst,
            cond,
            then_val,
            else_val,
            dtype,
        } => emit_select(b, *dst, *cond, *then_val, *else_val, *dtype),
        // Logical NOT over a Bool register — `xor.b32 dst, src, 1`.
        Op::Not { dst, src } => emit_not(b, *dst, *src),
        // ---- Decimal128 / i128 dual-register ops (v0.7 Sub-task A) ----
        // v0.7 Sub-task B wired these through `Codegen::emit_column` /
        // `emit_literal` / `emit_binary` (Add/Sub/Mul only) — see
        // `physical_plan.rs`. Div / comparisons / CAST involving
        // Decimal128 stay on the host fallback (rejected with a tighter
        // message at lower time).
        Op::LoadColumn128 {
            dst_lo,
            dst_hi,
            col_idx,
        } => emit_load_128(b, *dst_lo, *dst_hi, *col_idx, input_ptrs, tid),
        Op::Const128 {
            dst_lo,
            dst_hi,
            lo,
            hi,
        } => emit_const_128(b, *dst_lo, *dst_hi, *lo, *hi),
        Op::Store128 {
            src_lo,
            src_hi,
            col_idx,
        } => emit_store_128(b, *src_lo, *src_hi, *col_idx, output_ptrs, tid),
        Op::Add128 {
            dst_lo,
            dst_hi,
            a_lo,
            a_hi,
            b_lo,
            b_hi,
        } => emit_add_128(b, *dst_lo, *dst_hi, *a_lo, *a_hi, *b_lo, *b_hi),
        Op::Sub128 {
            dst_lo,
            dst_hi,
            a_lo,
            a_hi,
            b_lo,
            b_hi,
        } => emit_sub_128(b, *dst_lo, *dst_hi, *a_lo, *a_hi, *b_lo, *b_hi),
        Op::Mul128 {
            dst_lo,
            dst_hi,
            a_lo,
            a_hi,
            b_lo,
            b_hi,
        } => emit_mul_128(b, *dst_lo, *dst_hi, *a_lo, *a_hi, *b_lo, *b_hi),
        Op::Cmp128 {
            dst,
            op,
            a_lo,
            a_hi,
            b_lo,
            b_hi,
        } => emit_cmp_128(b, *dst, *op, *a_lo, *a_hi, *b_lo, *b_hi),
        // ---- Decimal128 / i128 ops added by F5 ----
        Op::WidenToI128 {
            dst_lo,
            dst_hi,
            src,
            from,
        } => emit_widen_to_i128(b, *dst_lo, *dst_hi, *src, *from),
        Op::NarrowI128ToInt {
            dst,
            src_lo,
            src_hi,
            to,
        } => emit_narrow_i128_to_int(b, *dst, *src_lo, *src_hi, *to),
        Op::Div128 {
            dst_lo,
            dst_hi,
            a_lo,
            a_hi,
            b_lo,
            b_hi,
        } => emit_div_128(b, *dst_lo, *dst_hi, *a_lo, *a_hi, *b_lo, *b_hi),
        Op::Select128 {
            dst_lo,
            dst_hi,
            cond,
            then_lo,
            then_hi,
            else_lo,
            else_hi,
        } => emit_select_128(b, *dst_lo, *dst_hi, *cond, *then_lo, *then_hi, *else_lo, *else_hi),
        Op::F64ToI128 { dst_lo, dst_hi, src } => {
            emit_f64_to_i128(b, *dst_lo, *dst_hi, *src)
        }
        Op::I128ToF64 { dst, src_lo, src_hi } => {
            emit_i128_to_f64(b, *dst, *src_lo, *src_hi)
        }
    }
}

/// Emit `Op::LoadColumn128` — two `ld.global.nc.u64` reads at byte offsets
/// `tid * 16` (lo) and `tid * 16 + 8` (hi) from input column `col_idx`'s
/// base pointer.
///
/// Address arithmetic mirrors `emit_load`'s pattern (`mul.wide.u32` widens
/// `tid` to 64 bits, then `add.s64` to the base) but uses a stride of 16
/// instead of the dtype's byte width — every Decimal128 row is exactly 16
/// bytes. The high half adds another `add.s64` of `+8` for the second load.
fn emit_load_128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    col_idx: usize,
    input_ptrs: &[String],
    tid: &str,
) -> BoltResult<()> {
    if col_idx >= input_ptrs.len() {
        return Err(BoltError::Other(format!(
            "ptx_gen: LoadColumn128 col_idx {} out of range (have {} inputs)",
            col_idx,
            input_ptrs.len()
        )));
    }
    // PERF (codegen alloc): emit straight into `b.body` via `emit_fmt!`.
    let off = b.alloc.alloc("rd");
    let addr_lo = b.alloc.alloc("rd");
    let addr_hi = b.alloc.alloc("rd");
    emit_fmt!(b, "mul.wide.u32 {}, {}, 16;", off, tid)?;
    emit_fmt!(b, "add.s64 {}, {}, {};", addr_lo, input_ptrs[col_idx], off)?;
    emit_fmt!(b, "add.s64 {}, {}, 8;", addr_hi, addr_lo)?;
    let (lo_name, hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    // Read-only-cache hint matches `emit_load`: input column buffers never
    // alias outputs of the same kernel (see `.ptr .global .restrict` in
    // `write_signature`), so `ld.global.nc` is sound.
    emit_fmt!(b, "ld.global.nc.u64 {}, [{}];", lo_name, addr_lo)?;
    emit_fmt!(b, "ld.global.nc.u64 {}, [{}];", hi_name, addr_hi)?;
    Ok(())
}

/// Emit `Op::Store128` — two `st.global.u64` writes at byte offsets
/// `tid * 16` (lo) and `tid * 16 + 8` (hi) to output column `col_idx`'s
/// base pointer.
fn emit_store_128(
    b: &mut PtxBuilder,
    src_lo: Reg,
    src_hi: Reg,
    col_idx: usize,
    output_ptrs: &[String],
    tid: &str,
) -> BoltResult<()> {
    if col_idx >= output_ptrs.len() {
        return Err(BoltError::Other(format!(
            "ptx_gen: Store128 col_idx {} out of range (have {} outputs)",
            col_idx,
            output_ptrs.len()
        )));
    }
    // PERF (codegen alloc): `src_lo`/`src_hi` names are borrowed inline in the
    // two store writes (no `.to_string()`); the `off`/`addr_*` allocations all
    // happen first so the immutable lookups never overlap a `&mut b.alloc`.
    let off = b.alloc.alloc("rd");
    let addr_lo = b.alloc.alloc("rd");
    let addr_hi = b.alloc.alloc("rd");
    emit_fmt!(b, "mul.wide.u32 {}, {}, 16;", off, tid)?;
    emit_fmt!(b, "add.s64 {}, {}, {};", addr_lo, output_ptrs[col_idx], off)?;
    emit_fmt!(b, "add.s64 {}, {}, 8;", addr_hi, addr_lo)?;
    emit_fmt!(b, "st.global.u64 [{}], {};", addr_lo, b.alloc.get(src_lo)?)?;
    emit_fmt!(b, "st.global.u64 [{}], {};", addr_hi, b.alloc.get(src_hi)?)?;
    Ok(())
}

/// Emit `Op::Const128` — two `mov.u64` instructions loading the hex
/// bit-patterns into the low / high halves.
///
/// SECURITY: the bit-patterns are emitted as `0x{:016X}` hex, restricting
/// the output to `[0-9A-F]`. Matches the same codegen-injection-hardening
/// convention used by `emit_const` for `Int64` / `Float64`.
fn emit_const_128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    lo: u64,
    hi: u64,
) -> BoltResult<()> {
    let (lo_name, hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    // PERF (codegen alloc): emit straight into `b.body` via `emit_fmt!`.
    emit_fmt!(b, "mov.u64 {}, 0x{:016X};", lo_name, lo)?;
    emit_fmt!(b, "mov.u64 {}, 0x{:016X};", hi_name, hi)?;
    Ok(())
}

/// Emit `Op::Add128` — `add.cc.u64` on the low half (sets the implicit
/// `%CC` carry flag), then `addc.u64` on the high half (adds the carry-in).
fn emit_add_128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    a_lo: Reg,
    a_hi: Reg,
    b_lo: Reg,
    b_hi: Reg,
) -> BoltResult<()> {
    // PERF (codegen alloc): destination names are owned Strings from
    // `assign_pair`; operand names are read as `&str` borrows inside each
    // `emit_fmt!` so no per-operand `.to_string()` clone is made.
    let (dst_lo_name, dst_hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    emit_fmt!(
        b,
        "add.cc.u64 {}, {}, {};",
        dst_lo_name,
        b.alloc.get(a_lo)?,
        b.alloc.get(b_lo)?
    )?;
    emit_fmt!(
        b,
        "addc.u64 {}, {}, {};",
        dst_hi_name,
        b.alloc.get(a_hi)?,
        b.alloc.get(b_hi)?
    )?;
    Ok(())
}

/// Emit `Op::Sub128` — `sub.cc.u64` on the low half (sets the implicit
/// `%CC` borrow flag), then `subc.u64` on the high half (subtracts the
/// borrow-in).
fn emit_sub_128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    a_lo: Reg,
    a_hi: Reg,
    b_lo: Reg,
    b_hi: Reg,
) -> BoltResult<()> {
    // PERF (codegen alloc): operand names borrowed inline (no `.to_string()`);
    // destination names are owned Strings from `assign_pair`.
    let (dst_lo_name, dst_hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    emit_fmt!(
        b,
        "sub.cc.u64 {}, {}, {};",
        dst_lo_name,
        b.alloc.get(a_lo)?,
        b.alloc.get(b_lo)?
    )?;
    emit_fmt!(
        b,
        "subc.u64 {}, {}, {};",
        dst_hi_name,
        b.alloc.get(a_hi)?,
        b.alloc.get(b_hi)?
    )?;
    Ok(())
}

/// Emit `Op::Mul128` — 128-bit truncating multiply via schoolbook
/// cross-multiply (4 partial products, summing into the high half).
///
/// Algebra:
///
/// ```text
///   a = (a_hi << 64) | a_lo
///   b = (b_hi << 64) | b_lo
///   a * b = a_lo*b_lo
///         + (a_lo*b_hi + a_hi*b_lo) << 64
///         + (a_hi*b_hi)             << 128   // discarded (wraps)
/// ```
///
/// We compute:
///
/// ```text
///   dst_lo            = mul.lo.u64 a_lo, b_lo
///   hi_acc            = mul.hi.u64 a_lo, b_lo
///   cross1            = mul.lo.u64 a_lo, b_hi
///   cross2            = mul.lo.u64 a_hi, b_lo
///   hi_acc            = add.u64 hi_acc, cross1
///   dst_hi            = add.u64 hi_acc, cross2
/// ```
///
/// We use plain `add.u64` (not `add.cc.u64`) for the two high-half sums:
/// any overflow there falls into bits 128+, which 128-bit wrapping
/// arithmetic discards. This matches `i128::wrapping_mul` semantics and
/// Arrow's Decimal128 arithmetic (which checks overflow at the validation
/// layer, not in the kernel).
fn emit_mul_128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    a_lo: Reg,
    a_hi: Reg,
    b_lo: Reg,
    b_hi: Reg,
) -> BoltResult<()> {
    // PERF (codegen alloc): operand names (`a_lo`/`a_hi`/`b_lo`/`b_hi`) are
    // borrowed inline in each `emit_fmt!` rather than cloned to owned Strings
    // up front. All allocator mutation (temps + destination pair) happens
    // first, so the fleeting `b.alloc.get(...)` borrows inside the writes never
    // overlap a `&mut b.alloc`.
    //
    // Temporaries for the three high-half partial products. Allocate
    // before the destination pair so the dst registers land at higher
    // indices in the `rl` class (purely cosmetic — SASS register
    // assignment is downstream of PTX).
    let hi_acc = b.alloc.alloc("rl");
    let cross1 = b.alloc.alloc("rl");
    let cross2 = b.alloc.alloc("rl");
    let (dst_lo_name, dst_hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    // Low half: a_lo * b_lo (truncated).
    emit_fmt!(
        b,
        "mul.lo.u64 {}, {}, {};",
        dst_lo_name,
        b.alloc.get(a_lo)?,
        b.alloc.get(b_lo)?
    )?;
    // High half: cross-multiply sum.
    emit_fmt!(
        b,
        "mul.hi.u64 {}, {}, {};",
        hi_acc,
        b.alloc.get(a_lo)?,
        b.alloc.get(b_lo)?
    )?;
    emit_fmt!(
        b,
        "mul.lo.u64 {}, {}, {};",
        cross1,
        b.alloc.get(a_lo)?,
        b.alloc.get(b_hi)?
    )?;
    emit_fmt!(
        b,
        "mul.lo.u64 {}, {}, {};",
        cross2,
        b.alloc.get(a_hi)?,
        b.alloc.get(b_lo)?
    )?;
    emit_fmt!(b, "add.u64 {}, {}, {};", hi_acc, hi_acc, cross1)?;
    emit_fmt!(b, "add.u64 {}, {}, {};", dst_hi_name, hi_acc, cross2)?;
    Ok(())
}

/// Emit `Op::Cmp128` — split-register 128-bit signed comparison producing
/// a Bool (0/1) in `dst`. The PTX side has no native 128-bit `setp`, so
/// we emit the standard high-half / low-half decomposition documented in
/// the PTX ISA reference under "Integer Compare Operations".
///
/// Per-op wire shape (signed-high, unsigned-low):
///
/// ```text
///   eq: setp.eq.u64 p_lo, a_lo, b_lo
///       setp.eq.s64 p_hi, a_hi, b_hi
///       and.pred    p,    p_lo, p_hi
///
///   ne: setp.ne.u64 p_lo, a_lo, b_lo
///       setp.ne.s64 p_hi, a_hi, b_hi
///       or.pred     p,    p_lo, p_hi
///
///   lt: setp.lt.s64 p_hi_lt, a_hi, b_hi
///       setp.eq.s64 p_hi_eq, a_hi, b_hi
///       setp.lt.u64 p_lo_lt, a_lo, b_lo
///       and.pred    p_eq_lt, p_hi_eq, p_lo_lt
///       or.pred     p,       p_hi_lt, p_eq_lt
///
///   gt: same as lt with .lt -> .gt on the high-half compares
///       and .lt -> .gt on the low-half compare.
///
///   le: same as lt with the low-half compare promoted to `<=`
///       (setp.le.u64) so the equal-low path also fires.
///
///   ge: same as gt with the low-half compare promoted to `>=`.
/// ```
///
/// Why signed on the high half and unsigned on the low half: the i128's
/// sign lives in bit 127, which is the top bit of the *high* half. The
/// low half always carries plain magnitude bits — its raw u64 ordering
/// IS the within-equal-high-half ordering of the full i128 (negatives
/// have all-set high halves but arbitrary low halves; once the high
/// halves are equal the low halves can be compared as unsigned).
///
/// Materialising the result: a single `selp.s32 dst, 1, 0, p` turns the
/// final predicate into the canonical Bool (0/1) `b32` representation
/// matching every other comparison emitter in this file.
fn emit_cmp_128(
    b: &mut PtxBuilder,
    dst: Reg,
    op: BinaryOp,
    a_lo: Reg,
    a_hi: Reg,
    b_lo: Reg,
    b_hi: Reg,
) -> BoltResult<()> {
    // PERF (codegen alloc): the four operand names are read once into owned
    // Strings here because each match arm interleaves `b.alloc.alloc("p")`
    // (a `&mut b.alloc`) between this point and the uses below; a held `&str`
    // borrow from `b.alloc.get(...)` would not survive those mutations. The
    // per-instruction `format!` allocations are still eliminated by emitting
    // through `emit_fmt!` straight into `b.body`.
    let a_lo_name = b.alloc.get(a_lo)?.to_string();
    let a_hi_name = b.alloc.get(a_hi)?.to_string();
    let b_lo_name = b.alloc.get(b_lo)?.to_string();
    let b_hi_name = b.alloc.get(b_hi)?.to_string();
    let dst_name = b.alloc.assign(dst, DataType::Bool)?;

    use BinaryOp::*;
    match op {
        Eq => {
            // Both halves must match.
            let p_lo = b.alloc.alloc("p");
            let p_hi = b.alloc.alloc("p");
            let p = b.alloc.alloc("p");
            emit_fmt!(b, "setp.eq.u64 {}, {}, {};", p_lo, a_lo_name, b_lo_name)?;
            emit_fmt!(b, "setp.eq.s64 {}, {}, {};", p_hi, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "and.pred {}, {}, {};", p, p_lo, p_hi)?;
            emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, p)?;
        }
        NotEq => {
            // Either half differs.
            let p_lo = b.alloc.alloc("p");
            let p_hi = b.alloc.alloc("p");
            let p = b.alloc.alloc("p");
            emit_fmt!(b, "setp.ne.u64 {}, {}, {};", p_lo, a_lo_name, b_lo_name)?;
            emit_fmt!(b, "setp.ne.s64 {}, {}, {};", p_hi, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "or.pred {}, {}, {};", p, p_lo, p_hi)?;
            emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, p)?;
        }
        Lt => {
            // a < b  <=>  (a_hi <s b_hi) || (a_hi == b_hi && a_lo <u b_lo)
            let p_hi_lt = b.alloc.alloc("p");
            let p_hi_eq = b.alloc.alloc("p");
            let p_lo_lt = b.alloc.alloc("p");
            let p_eq_lt = b.alloc.alloc("p");
            let p = b.alloc.alloc("p");
            emit_fmt!(b, "setp.lt.s64 {}, {}, {};", p_hi_lt, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "setp.eq.s64 {}, {}, {};", p_hi_eq, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "setp.lt.u64 {}, {}, {};", p_lo_lt, a_lo_name, b_lo_name)?;
            emit_fmt!(b, "and.pred {}, {}, {};", p_eq_lt, p_hi_eq, p_lo_lt)?;
            emit_fmt!(b, "or.pred {}, {}, {};", p, p_hi_lt, p_eq_lt)?;
            emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, p)?;
        }
        Gt => {
            // a > b  <=>  (a_hi >s b_hi) || (a_hi == b_hi && a_lo >u b_lo)
            let p_hi_gt = b.alloc.alloc("p");
            let p_hi_eq = b.alloc.alloc("p");
            let p_lo_gt = b.alloc.alloc("p");
            let p_eq_gt = b.alloc.alloc("p");
            let p = b.alloc.alloc("p");
            emit_fmt!(b, "setp.gt.s64 {}, {}, {};", p_hi_gt, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "setp.eq.s64 {}, {}, {};", p_hi_eq, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "setp.gt.u64 {}, {}, {};", p_lo_gt, a_lo_name, b_lo_name)?;
            emit_fmt!(b, "and.pred {}, {}, {};", p_eq_gt, p_hi_eq, p_lo_gt)?;
            emit_fmt!(b, "or.pred {}, {}, {};", p, p_hi_gt, p_eq_gt)?;
            emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, p)?;
        }
        LtEq => {
            // a <= b  <=>  (a_hi <s b_hi) || (a_hi == b_hi && a_lo <=u b_lo)
            let p_hi_lt = b.alloc.alloc("p");
            let p_hi_eq = b.alloc.alloc("p");
            let p_lo_le = b.alloc.alloc("p");
            let p_eq_le = b.alloc.alloc("p");
            let p = b.alloc.alloc("p");
            emit_fmt!(b, "setp.lt.s64 {}, {}, {};", p_hi_lt, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "setp.eq.s64 {}, {}, {};", p_hi_eq, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "setp.le.u64 {}, {}, {};", p_lo_le, a_lo_name, b_lo_name)?;
            emit_fmt!(b, "and.pred {}, {}, {};", p_eq_le, p_hi_eq, p_lo_le)?;
            emit_fmt!(b, "or.pred {}, {}, {};", p, p_hi_lt, p_eq_le)?;
            emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, p)?;
        }
        GtEq => {
            // a >= b  <=>  (a_hi >s b_hi) || (a_hi == b_hi && a_lo >=u b_lo)
            let p_hi_gt = b.alloc.alloc("p");
            let p_hi_eq = b.alloc.alloc("p");
            let p_lo_ge = b.alloc.alloc("p");
            let p_eq_ge = b.alloc.alloc("p");
            let p = b.alloc.alloc("p");
            emit_fmt!(b, "setp.gt.s64 {}, {}, {};", p_hi_gt, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "setp.eq.s64 {}, {}, {};", p_hi_eq, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "setp.ge.u64 {}, {}, {};", p_lo_ge, a_lo_name, b_lo_name)?;
            emit_fmt!(b, "and.pred {}, {}, {};", p_eq_ge, p_hi_eq, p_lo_ge)?;
            emit_fmt!(b, "or.pred {}, {}, {};", p, p_hi_gt, p_eq_ge)?;
            emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, p)?;
        }
        _ => {
            return Err(BoltError::Other(format!(
                "ptx_gen: Op::Cmp128 with non-comparison op {:?} — planner bug \
                 (Codegen::emit_binary_decimal128_cmp must reject non-comparison \
                 ops before emitting Op::Cmp128)",
                op
            )));
        }
    }
    Ok(())
}

/// Emit `Op::WidenToI128` — sign-extend a 32/64-bit signed integer into an
/// i128 `(lo, hi)` pair.
///
/// Wire shape:
///
/// ```text
///   // lo half: bring the value into a b64 register (sign-extending if i32)
///   cvt.s64.s32 lo, src        // (Int32 / Date32)
///   mov.u64     lo, src        // (Int64)
///   // hi half: arithmetic shift right by 63 splats the sign bit
///   shr.s64     hi, lo, 63
/// ```
///
/// The `shr.s64 ..., 63` produces all-zero for non-negative values and
/// all-ones (`0xFFFF...`) for negatives, which is exactly the i128 high
/// half of a sign-extended 64-bit value.
fn emit_widen_to_i128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    src: Reg,
    from: DataType,
) -> BoltResult<()> {
    let src_name = b.alloc.get(src)?.to_string();
    let (lo_name, hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    match from {
        // Int32 lives in the b32 (`r`) class; sign-extend into the b64 lo.
        // Date32 (i32 days) rides the same path.
        DataType::Int32 | DataType::Date32 | DataType::Bool => {
            emit_fmt!(b, "cvt.s64.s32 {}, {};", lo_name, src_name)?;
        }
        // Int64 (and Timestamp ticks) already occupy a b64 register.
        DataType::Int64 | DataType::Timestamp(_, _) => {
            emit_fmt!(b, "mov.u64 {}, {};", lo_name, src_name)?;
        }
        other => {
            return Err(BoltError::Other(format!(
                "ptx_gen: WidenToI128 source must be an integer dtype, got {other:?}"
            )));
        }
    }
    // Splat the sign bit of the (now 64-bit) lo half across the whole hi half.
    emit_fmt!(b, "shr.s64 {}, {}, 63;", hi_name, lo_name)?;
    Ok(())
}

/// Emit `Op::NarrowI128ToInt` — take the low 64 bits of an i128 pair as the
/// integer result (truncating, matching `as i64` / `as i32`).
///
/// For `Int64` the low half *is* the result. For `Int32` we additionally
/// truncate the low half to 32 bits via `cvt.s32.s64` (the `_hi` operand is
/// unused — the value bits already live in the low half once the caller has
/// divided the scale out).
fn emit_narrow_i128_to_int(
    b: &mut PtxBuilder,
    dst: Reg,
    src_lo: Reg,
    _src_hi: Reg,
    to: DataType,
) -> BoltResult<()> {
    let lo_name = b.alloc.get(src_lo)?.to_string();
    let dst_name = b.alloc.assign(dst, to)?;
    match to {
        DataType::Int64 => emit_fmt!(b, "mov.u64 {}, {};", dst_name, lo_name),
        DataType::Int32 => emit_fmt!(b, "cvt.s32.s64 {}, {};", dst_name, lo_name),
        other => Err(BoltError::Other(format!(
            "ptx_gen: NarrowI128ToInt target must be Int32/Int64, got {other:?}"
        ))),
    }
}

/// Emit `Op::Select128` — predicated 128-bit selection, the i128 twin of
/// `Op::Select`. Two `selp.b64` (one per half) gated on the same predicate.
///
/// ```text
///   setp.ne.s32 p, cond, 0
///   selp.b64    dst_lo, then_lo, else_lo, p
///   selp.b64    dst_hi, then_hi, else_hi, p
/// ```
fn emit_select_128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    cond: Reg,
    then_lo: Reg,
    then_hi: Reg,
    else_lo: Reg,
    else_hi: Reg,
) -> BoltResult<()> {
    let cond_name = b.alloc.get(cond)?.to_string();
    let then_lo_name = b.alloc.get(then_lo)?.to_string();
    let then_hi_name = b.alloc.get(then_hi)?.to_string();
    let else_lo_name = b.alloc.get(else_lo)?.to_string();
    let else_hi_name = b.alloc.get(else_hi)?.to_string();
    let pred = b.alloc.alloc("p");
    let (dst_lo_name, dst_hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    emit_fmt!(b, "setp.ne.s32 {}, {}, 0;", pred, cond_name)?;
    emit_fmt!(
        b,
        "selp.b64 {}, {}, {}, {};",
        dst_lo_name,
        then_lo_name,
        else_lo_name,
        pred
    )?;
    emit_fmt!(
        b,
        "selp.b64 {}, {}, {}, {};",
        dst_hi_name,
        then_hi_name,
        else_hi_name,
        pred
    )?;
    Ok(())
}

/// Emit `Op::Div128` — 128-bit signed truncating division.
///
/// sm_70 has no native 128-bit divide, so we emit a sign-fixup wrapper
/// around an unsigned restoring shift-subtract long division operating on
/// the `(lo, hi)` halves:
///
/// 1. Compute the signs of dividend (`a`) and divisor (`b`) from the top
///    bit of each high half; the quotient sign is their XOR.
/// 2. Take the absolute value of both operands (negate the i128 if the
///    sign bit is set — two's-complement negate is `~x + 1` over the pair).
/// 3. Guard a zero divisor: if `|b| == 0`, branch to a "store zero quotient"
///    tail (deterministic, non-trapping — see `Op::Div128` rustdoc).
/// 4. Restoring long division, 128 iterations, MSB-first: shift the running
///    remainder left by one and pull in the next dividend bit; if the
///    remainder >= divisor, subtract and set the quotient bit.
/// 5. Negate the unsigned quotient back if the quotient sign is negative.
///
/// The loop is emitted as straight PTX with a back-edge label and a 32-bit
/// counter; all arithmetic is on the two-`u64`-half representation already
/// used by `emit_add_128` / `emit_sub_128`.
fn emit_div_128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    a_lo: Reg,
    a_hi: Reg,
    b_lo: Reg,
    b_hi: Reg,
) -> BoltResult<()> {
    // Snapshot operand names (the loop allocates many temporaries, so a held
    // `&str` borrow would not survive the interleaved `&mut b.alloc` calls).
    let a_lo_n = b.alloc.get(a_lo)?.to_string();
    let a_hi_n = b.alloc.get(a_hi)?.to_string();
    let b_lo_n = b.alloc.get(b_lo)?.to_string();
    let b_hi_n = b.alloc.get(b_hi)?.to_string();

    // A unique label suffix per emission so multiple Div128 ops in one kernel
    // don't collide. `b.alloc` register indices are monotonic and unique, so
    // the first quotient register's index is a convenient discriminator.
    let (q_lo, q_hi) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    let tag = q_lo.trim_start_matches('%').to_string();

    // Working registers for |a| (running dividend / remainder feed) and |b|.
    let ua_lo = b.alloc.alloc("rl");
    let ua_hi = b.alloc.alloc("rl");
    let ub_lo = b.alloc.alloc("rl");
    let ub_hi = b.alloc.alloc("rl");
    // Quotient-sign flag (1 if result should be negated).
    let qsign = b.alloc.alloc("r");
    let sa = b.alloc.alloc("r");
    let sb = b.alloc.alloc("r");
    // Remainder pair, current quotient pair (built up), shift temporaries.
    let rem_lo = b.alloc.alloc("rl");
    let rem_hi = b.alloc.alloc("rl");
    let i = b.alloc.alloc("r");
    let bit = b.alloc.alloc("rl");
    let tmp_lo = b.alloc.alloc("rl");
    let tmp_hi = b.alloc.alloc("rl");
    let carry = b.alloc.alloc("rl");
    let p0 = b.alloc.alloc("p");
    let p1 = b.alloc.alloc("p");
    let p2 = b.alloc.alloc("p");

    // ---- sign extraction: sign = (hi >> 63) & 1, via shr.u64 ... 63 ----
    emit_fmt!(b, "shr.u64 {}, {}, 63;", tmp_lo, a_hi_n)?;
    emit_fmt!(b, "cvt.u32.u64 {}, {};", sa, tmp_lo)?;
    emit_fmt!(b, "shr.u64 {}, {}, 63;", tmp_hi, b_hi_n)?;
    emit_fmt!(b, "cvt.u32.u64 {}, {};", sb, tmp_hi)?;
    emit_fmt!(b, "xor.b32 {}, {}, {};", qsign, sa, sb)?;

    // ---- |a|: negate the i128 if sa==1 (two's complement over the pair) ----
    // ua = a; if (sa) ua = 0 - a
    emit_fmt!(b, "mov.u64 {}, {};", ua_lo, a_lo_n)?;
    emit_fmt!(b, "mov.u64 {}, {};", ua_hi, a_hi_n)?;
    emit_fmt!(b, "setp.ne.s32 {}, {}, 0;", p0, sa)?;
    emit_fmt!(b, "@!{} bra DIV_A_POS_{};", p0, tag)?;
    emit_fmt!(b, "sub.cc.u64 {}, 0, {};", ua_lo, a_lo_n)?;
    emit_fmt!(b, "subc.u64 {}, 0, {};", ua_hi, a_hi_n)?;
    b.emit_label(&format!("DIV_A_POS_{}", tag))?;

    // ---- |b|: same ----
    emit_fmt!(b, "mov.u64 {}, {};", ub_lo, b_lo_n)?;
    emit_fmt!(b, "mov.u64 {}, {};", ub_hi, b_hi_n)?;
    emit_fmt!(b, "setp.ne.s32 {}, {}, 0;", p0, sb)?;
    emit_fmt!(b, "@!{} bra DIV_B_POS_{};", p0, tag)?;
    emit_fmt!(b, "sub.cc.u64 {}, 0, {};", ub_lo, b_lo_n)?;
    emit_fmt!(b, "subc.u64 {}, 0, {};", ub_hi, b_hi_n)?;
    b.emit_label(&format!("DIV_B_POS_{}", tag))?;

    // ---- init quotient = 0, remainder = 0 ----
    emit_fmt!(b, "mov.u64 {}, 0;", q_lo)?;
    emit_fmt!(b, "mov.u64 {}, 0;", q_hi)?;
    emit_fmt!(b, "mov.u64 {}, 0;", rem_lo)?;
    emit_fmt!(b, "mov.u64 {}, 0;", rem_hi)?;

    // ---- div-by-zero guard: if |b| == 0, quotient stays 0 -> done ----
    emit_fmt!(b, "or.b64 {}, {}, {};", tmp_lo, ub_lo, ub_hi)?;
    emit_fmt!(b, "setp.eq.u64 {}, {}, 0;", p0, tmp_lo)?;
    emit_fmt!(b, "@{} bra DIV_DONE_{};", p0, tag)?;

    // ---- restoring long division, MSB first, 128 iterations ----
    emit_fmt!(b, "mov.u32 {}, 0;", i)?;
    b.emit_label(&format!("DIV_LOOP_{}", tag))?;
    emit_fmt!(b, "setp.ge.u32 {}, {}, 128;", p0, i)?;
    emit_fmt!(b, "@{} bra DIV_NEG_{};", p0, tag)?;

    // rem = (rem << 1) | next-MSB-of-ua
    //   shift rem left by 1 (carry low->high)
    emit_fmt!(b, "shr.u64 {}, {}, 63;", carry, rem_lo)?; // bit leaving lo
    emit_fmt!(b, "shl.b64 {}, {}, 1;", rem_hi, rem_hi)?;
    emit_fmt!(b, "or.b64 {}, {}, {};", rem_hi, rem_hi, carry)?;
    emit_fmt!(b, "shl.b64 {}, {}, 1;", rem_lo, rem_lo)?;
    //   pull current MSB of ua (bit 127-i) into rem_lo bit 0.
    //   We rotate ua left by 1 each iteration so its MSB is always bit 127.
    emit_fmt!(b, "shr.u64 {}, {}, 63;", bit, ua_hi)?; // ua MSB
    emit_fmt!(b, "or.b64 {}, {}, {};", rem_lo, rem_lo, bit)?;
    //   ua <<= 1
    emit_fmt!(b, "shr.u64 {}, {}, 63;", carry, ua_lo)?;
    emit_fmt!(b, "shl.b64 {}, {}, 1;", ua_hi, ua_hi)?;
    emit_fmt!(b, "or.b64 {}, {}, {};", ua_hi, ua_hi, carry)?;
    emit_fmt!(b, "shl.b64 {}, {}, 1;", ua_lo, ua_lo)?;

    // if rem >= |b| (unsigned 128-bit compare): rem -= |b|; set quotient bit.
    //   compute (rem >= ub): rem_hi > ub_hi || (rem_hi==ub_hi && rem_lo>=ub_lo)
    emit_fmt!(b, "setp.gt.u64 {}, {}, {};", p0, rem_hi, ub_hi)?;
    emit_fmt!(b, "setp.eq.u64 {}, {}, {};", p1, rem_hi, ub_hi)?;
    emit_fmt!(b, "setp.ge.u64 {}, {}, {};", p2, rem_lo, ub_lo)?;
    emit_fmt!(b, "and.pred {}, {}, {};", p1, p1, p2)?;
    emit_fmt!(b, "or.pred {}, {}, {};", p0, p0, p1)?;
    emit_fmt!(b, "@!{} bra DIV_NOSUB_{};", p0, tag)?;
    //   rem -= ub (128-bit borrow chain)
    emit_fmt!(b, "sub.cc.u64 {}, {}, {};", rem_lo, rem_lo, ub_lo)?;
    emit_fmt!(b, "subc.u64 {}, {}, {};", rem_hi, rem_hi, ub_hi)?;
    //   q = (q << 1) | 1   — but we build q MSB-first too: shift then set bit0.
    emit_fmt!(b, "shr.u64 {}, {}, 63;", carry, q_lo)?;
    emit_fmt!(b, "shl.b64 {}, {}, 1;", q_hi, q_hi)?;
    emit_fmt!(b, "or.b64 {}, {}, {};", q_hi, q_hi, carry)?;
    emit_fmt!(b, "shl.b64 {}, {}, 1;", q_lo, q_lo)?;
    emit_fmt!(b, "or.b64 {}, {}, 1;", q_lo, q_lo)?;
    emit_fmt!(b, "bra DIV_NEXT_{};", tag)?;
    b.emit_label(&format!("DIV_NOSUB_{}", tag))?;
    //   q = (q << 1) | 0
    emit_fmt!(b, "shr.u64 {}, {}, 63;", carry, q_lo)?;
    emit_fmt!(b, "shl.b64 {}, {}, 1;", q_hi, q_hi)?;
    emit_fmt!(b, "or.b64 {}, {}, {};", q_hi, q_hi, carry)?;
    emit_fmt!(b, "shl.b64 {}, {}, 1;", q_lo, q_lo)?;
    b.emit_label(&format!("DIV_NEXT_{}", tag))?;
    emit_fmt!(b, "add.u32 {}, {}, 1;", i, i)?;
    emit_fmt!(b, "bra DIV_LOOP_{};", tag)?;

    // ---- apply quotient sign: if qsign==1, negate q ----
    b.emit_label(&format!("DIV_NEG_{}", tag))?;
    emit_fmt!(b, "setp.ne.s32 {}, {}, 0;", p0, qsign)?;
    emit_fmt!(b, "@!{} bra DIV_DONE_{};", p0, tag)?;
    emit_fmt!(b, "sub.cc.u64 {}, 0, {};", q_lo, q_lo)?;
    emit_fmt!(b, "subc.u64 {}, 0, {};", q_hi, q_hi)?;

    b.emit_label(&format!("DIV_DONE_{}", tag))?;
    Ok(())
}

/// Emit `Op::I128ToF64` — convert a signed i128 `(lo, hi)` pair to `f64`,
/// computing `hi * 2^64 + lo` in floating point.
///
/// ```text
///   cvt.rn.f64.s64 %hf, %hi        // signed high half -> f64
///   cvt.rn.f64.u64 %lf, %lo        // UNSIGNED low half -> f64
///   mov.f64        %two64, 0d43F0000000000000   // 2^64
///   fma.rn.f64     %dst, %hf, %two64, %lf        // hi*2^64 + lo
/// ```
///
/// The low half is converted as UNSIGNED (`cvt.rn.f64.u64`) because in a
/// two's-complement i128 the value is exactly `hi*2^64 + lo_unsigned` — the
/// sign already lives entirely in `hi` (converted signed). `fma.rn`
/// round-to-nearest keeps the single unavoidable rounding step. Precision
/// loss beyond f64's 53 significant bits is expected for a decimal->float
/// conversion (documented on `Op::I128ToF64`).
fn emit_i128_to_f64(b: &mut PtxBuilder, dst: Reg, src_lo: Reg, src_hi: Reg) -> BoltResult<()> {
    let lo_n = b.alloc.get(src_lo)?.to_string();
    let hi_n = b.alloc.get(src_hi)?.to_string();
    let hf = b.alloc.alloc("fd");
    let lf = b.alloc.alloc("fd");
    let two64 = b.alloc.alloc("fd");
    let dst_name = b.alloc.assign(dst, DataType::Float64)?;
    // 0x43F0000000000000 is 2^64 as an IEEE-754 double.
    emit_fmt!(b, "cvt.rn.f64.s64 {}, {};", hf, hi_n)?;
    emit_fmt!(b, "cvt.rn.f64.u64 {}, {};", lf, lo_n)?;
    emit_fmt!(b, "mov.f64 {}, 0d43F0000000000000;", two64)?;
    emit_fmt!(b, "fma.rn.f64 {}, {}, {}, {};", dst_name, hf, two64, lf)?;
    Ok(())
}

/// Emit `Op::F64ToI128` — convert an `f64` to a signed i128 `(lo, hi)` pair,
/// rounding HALF AWAY FROM ZERO.
///
/// Strategy (all in f64 until the final integer extraction):
///
/// 1. `mag = |x|`, `sgn = sign(x)` (via `abs.f64` / `copysign`).
/// 2. Round half away from zero on the magnitude: `m = trunc(mag + 0.5)`
///    (`cvt.rzi.f64.f64` truncates toward zero; adding 0.5 to the
///    non-negative magnitude first gives round-half-up = half-away-from-zero
///    once the sign is reapplied).
/// 3. Split `m` into two unsigned 64-bit limbs:
///    `hi = trunc(m * 2^-64)`, `lo = m - hi*2^64`, each `cvt.rzi.u64.f64`.
///    For `m < 2^64` the high limb is 0 and the low limb is `m`; for
///    `m` in `[2^64, 2^128)` both limbs are in range.
/// 4. Reassemble the unsigned magnitude `(lo, hi)`, then negate the i128
///    (two's-complement over the pair) iff `x < 0`.
///
/// OVERFLOW / NaN (non-trapping, per the `Op::Div128` convention): the
/// conversion saturates to the i128 bounds. A magnitude `>= 2^127` (including
/// `+inf`) yields `i128::MAX` (hi=`0x7FFF…FFFF`, lo=`0xFFFF…FFFF`); a value
/// `<= -(2^127)` (including `-inf`) yields `i128::MIN` (hi=`0x8000…0000`,
/// lo=`0`); NaN converts to 0. The magnitude is compared against `2^127`
/// BEFORE the round-half-away `+0.5`, so the threshold tests the true
/// magnitude. There is no per-row validity signal for cast overflow on this
/// IR path (see the F5 CAST notes in `physical_plan`).
fn emit_f64_to_i128(b: &mut PtxBuilder, dst_lo: Reg, dst_hi: Reg, src: Reg) -> BoltResult<()> {
    let src_n = b.alloc.get(src)?.to_string();
    let mag = b.alloc.alloc("fd");
    let m = b.alloc.alloc("fd");
    let half = b.alloc.alloc("fd");
    let two127 = b.alloc.alloc("fd");
    let two64 = b.alloc.alloc("fd");
    let inv_two64 = b.alloc.alloc("fd");
    let hi_f = b.alloc.alloc("fd");
    let lo_f = b.alloc.alloc("fd");
    let prod = b.alloc.alloc("fd");
    let neg = b.alloc.alloc("r");
    let p0 = b.alloc.alloc("p");
    let psat = b.alloc.alloc("p");
    let (lo_n, hi_n) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    // Unique label suffix per emission (the lo destination index is unique).
    let tag = lo_n.trim_start_matches('%').to_string();

    // sgn: is x negative? (setp on the original value; NaN compares false.)
    emit_fmt!(b, "setp.lt.f64 {}, {}, 0d0000000000000000;", p0, src_n)?;
    emit_fmt!(b, "selp.b32 {}, 1, 0, {};", neg, p0)?;
    // mag = |x|  (true magnitude, BEFORE the +0.5 round addend).
    emit_fmt!(b, "abs.f64 {}, {};", mag, src_n)?;
    // i128 saturation gate: |x| >= 2^127 -> clamp to i128::MIN/MAX.
    // NaN compares false here, so NaN falls through to the normal path (-> 0).
    emit_fmt!(b, "mov.f64 {}, 0d47E0000000000000;", two127)?; // 2^127
    emit_fmt!(b, "setp.ge.f64 {}, {}, {};", psat, mag, two127)?;
    emit_fmt!(b, "@!{} bra F2I_NOSAT_{};", psat, tag)?;
    // Saturating branch: select i128::MAX (positive) or i128::MIN (negative)
    // from the already-computed sign flag, then jump to the shared tail.
    emit_fmt!(b, "setp.ne.s32 {}, {}, 0;", p0, neg)?;
    emit_fmt!(b, "@{} bra F2I_SATNEG_{};", p0, tag)?;
    // x >= 2^127 (or +inf): i128::MAX.
    emit_fmt!(b, "mov.u64 {}, 0xFFFFFFFFFFFFFFFF;", lo_n)?;
    emit_fmt!(b, "mov.u64 {}, 0x7FFFFFFFFFFFFFFF;", hi_n)?;
    emit_fmt!(b, "bra F2I_DONE_{};", tag)?;
    b.emit_label(&format!("F2I_SATNEG_{}", tag))?;
    // x <= -(2^127) (or -inf): i128::MIN.
    emit_fmt!(b, "mov.u64 {}, 0x0000000000000000;", lo_n)?;
    emit_fmt!(b, "mov.u64 {}, 0x8000000000000000;", hi_n)?;
    emit_fmt!(b, "bra F2I_DONE_{};", tag)?;
    b.emit_label(&format!("F2I_NOSAT_{}", tag))?;
    // Normal in-range path. m = trunc(mag + 0.5)  (round half away from zero).
    emit_fmt!(b, "mov.f64 {}, 0d3FE0000000000000;", half)?; // 0.5
    emit_fmt!(b, "add.f64 {}, {}, {};", mag, mag, half)?;
    emit_fmt!(b, "cvt.rzi.f64.f64 {}, {};", m, mag)?;
    // hi_f = trunc(m * 2^-64); lo_f = m - hi_f * 2^64.
    emit_fmt!(b, "mov.f64 {}, 0d3BF0000000000000;", inv_two64)?; // 2^-64
    emit_fmt!(b, "mov.f64 {}, 0d43F0000000000000;", two64)?; // 2^64
    emit_fmt!(b, "mul.f64 {}, {}, {};", hi_f, m, inv_two64)?;
    emit_fmt!(b, "cvt.rzi.f64.f64 {}, {};", hi_f, hi_f)?;
    emit_fmt!(b, "mul.f64 {}, {}, {};", prod, hi_f, two64)?;
    emit_fmt!(b, "sub.f64 {}, {}, {};", lo_f, m, prod)?;
    // Extract the two unsigned 64-bit limbs (saturating on overflow / NaN->0).
    emit_fmt!(b, "cvt.rzi.u64.f64 {}, {};", hi_n, hi_f)?;
    emit_fmt!(b, "cvt.rzi.u64.f64 {}, {};", lo_n, lo_f)?;
    // If x < 0, negate the i128 magnitude (two's complement over the pair).
    emit_fmt!(b, "setp.ne.s32 {}, {}, 0;", p0, neg)?;
    emit_fmt!(b, "@!{} bra F2I_DONE_{};", p0, tag)?;
    emit_fmt!(b, "sub.cc.u64 {}, 0, {};", lo_n, lo_n)?;
    emit_fmt!(b, "subc.u64 {}, 0, {};", hi_n, hi_n)?;
    b.emit_label(&format!("F2I_DONE_{}", tag))?;
    Ok(())
}

/// Host reference mirror of the kernel code emitted by [`emit_f64_to_i128`].
///
/// This is **not** used at codegen time — it is a pure host-side
/// reimplementation of the exact f64→i128 value computation that the emitted
/// PTX performs, so the conversion's rounding / saturation / NaN contract can
/// be unit-tested without a GPU. Every step below is a deliberate one-to-one
/// mirror of the PTX sequence in `emit_f64_to_i128`; if you change the
/// emitter you MUST change this in lockstep (and vice versa).
///
/// Step-for-step correspondence with the emitted PTX:
///
/// 1. `neg = x < 0`  ⟷  `setp.lt.f64` (NaN compares false, so `neg = false`).
/// 2. `mag = |x| + 0.5`  ⟷  `abs.f64` then `add.f64` with `0.5`.
/// 3. `m = trunc(mag)`  ⟷  `cvt.rzi.f64.f64` (round toward zero). Adding 0.5
///    to the non-negative magnitude before truncation is round-half-UP on the
///    magnitude, i.e. round-HALF-AWAY-FROM-ZERO once the sign is reapplied.
/// 4. `hi_f = trunc(m * 2^-64)`, `lo_f = m - hi_f * 2^64`  ⟷  the two
///    `mul.f64` / `cvt.rzi.f64.f64` / `sub.f64` limb-split steps.
/// 5. `hi_limb = sat_u64(hi_f)`, `lo_limb = sat_u64(lo_f)`  ⟷ the two
///    `cvt.rzi.u64.f64` extractions, which on the PTX side clamp NaN→0,
///    negatives→0, and values ≥ 2^64 → `u64::MAX` ([`f64_to_sat_u64`]).
/// 6. Reassemble the unsigned magnitude `(lo, hi)` as a 128-bit value and,
///    iff `neg`, take its two's complement  ⟷  the `sub.cc.u64` /
///    `subc.u64` negate-the-pair tail.
///
/// Contract (matches the `Op::F64ToI128` docs): rounds half away from zero;
/// NaN → 0; non-trapping; saturates to the i128 bounds. A magnitude `>= 2^127`
/// (including `+inf`) → `i128::MAX`; a value `<= -(2^127)` (including `-inf`) →
/// `i128::MIN`; in-range values take the limb-decomposition path below.
// `allow(dead_code)`: this is a host *reference* mirror of `emit_f64_to_i128`;
// its only caller today is the `#[cfg(test)]` conversion test, so a plain
// (non-test) build sees no use. Kept `pub(crate)` so non-test code can adopt
// it as the canonical conversion if/when the IR path materialises i128 casts.
#[allow(dead_code)]
pub(crate) fn f64_to_i128_saturating(x: f64) -> i128 {
    // i128 saturation gate (mirrors the `setp.ge.f64` against 2^127 in the
    // emitter, evaluated on the TRUE magnitude before the +0.5 round addend).
    // NaN is not >= / <= anything, so it falls through to the normal path and
    // clamps to 0 via the limb extractions below.
    const TWO127: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0; // 2^127
    if x.is_nan() {
        return 0;
    }
    if x >= TWO127 {
        return i128::MAX;
    }
    if x <= -TWO127 {
        return i128::MIN;
    }

    // Step 1: sign. (NaN already handled above.)
    let neg = x < 0.0;

    // Steps 2-3: round half away from zero on the magnitude.
    let mag = x.abs() + 0.5;
    let m = mag.trunc(); // cvt.rzi.f64.f64 == round-toward-zero == trunc.

    // Step 4: split the rounded magnitude into two unsigned 64-bit limbs in
    // f64 space, exactly as the PTX does (note `hi_f` is re-truncated).
    const TWO64: f64 = 18_446_744_073_709_551_616.0; // 2^64
    const INV_TWO64: f64 = 1.0 / TWO64; // 2^-64 (exact: both are powers of two)
    let hi_f = (m * INV_TWO64).trunc();
    let lo_f = m - hi_f * TWO64;

    // Step 5: saturating f64 → u64 per limb (NaN→0, <0→0, ≥2^64→u64::MAX).
    let hi_limb = f64_to_sat_u64(hi_f);
    let lo_limb = f64_to_sat_u64(lo_f);

    // Step 6: reassemble the unsigned magnitude, then two's-complement negate
    // the 128-bit pair iff the original value was negative.
    let mag_u128 = ((hi_limb as u128) << 64) | (lo_limb as u128);
    let bits = if neg { mag_u128.wrapping_neg() } else { mag_u128 };
    bits as i128
}

/// Saturating `f64` → `u64` mirroring PTX `cvt.rzi.u64.f64`:
/// round toward zero, clamp NaN → 0, negatives → 0, and values at or above
/// `2^64` → `u64::MAX`. Split out so [`f64_to_i128_saturating`] reads as a
/// direct transcription of the emitter's two limb extractions.
// `allow(dead_code)`: helper for the test-only `f64_to_i128_saturating` mirror
// above; unused in a plain (non-test) build for the same reason.
#[allow(dead_code)]
fn f64_to_sat_u64(x: f64) -> u64 {
    const TWO64: f64 = 18_446_744_073_709_551_616.0; // 2^64
    if x.is_nan() || x <= 0.0 {
        // NaN and ≤0 both clamp to 0 (the magnitude limbs are non-negative;
        // exact 0.0 truncates to 0 either way).
        0
    } else if x >= TWO64 {
        u64::MAX
    } else {
        // In [0, 2^64): truncation toward zero fits a u64 exactly.
        x.trunc() as u64
    }
}

/// Emit PTX for `Op::IsNullCheck`: load the validity byte for the current
/// row from `input_validity_ptrs[validity_input]` and produce a Bool (0/1)
/// in `dst` reflecting the IS [NOT] NULL outcome.
///
/// Wire shape:
///
/// ```text
///   mul.wide.u32 %off, %tid, 1              // UNSIGNED widen row index to b64
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
    // PERF (codegen alloc): emit straight into `b.body` via `emit_fmt!`.
    let off = b.alloc.alloc("rd");
    let addr = b.alloc.alloc("rd");
    let byte_reg = b.alloc.alloc("r");
    // C-3: UNSIGNED widen (`mul.wide.u32 _, 1`) so the validity-byte offset
    // matches the value-load path (`emit_load` uses `mul.wide.u32`). A signed
    // `cvt.s64.s32` would sign-extend `tid` above 2^31 rows → OOB validity load.
    emit_fmt!(b, "mul.wide.u32 {}, {}, 1;", off, tid)?;
    emit_fmt!(b, "add.s64 {}, {}, {};", addr, vptr, off)?;
    emit_fmt!(b, "ld.global.nc.u8 {}, [{}];", byte_reg, addr)?;

    // Predicate + Bool result. `setp.{eq,ne}.u32` is the right typed
    // comparator for the b32 byte_reg above (zero-extended from the u8
    // load). `selp.s32` materialises the 0/1 Bool in the b32 class to
    // match the existing Bool ABI (see `RegAlloc::class_for(Bool)`).
    let dst_name = b.alloc.assign(dst, DataType::Bool)?;
    let pred = b.alloc.alloc("p");
    let cmp = if want_null { "setp.eq.u32" } else { "setp.ne.u32" };
    emit_fmt!(b, "{} {}, {}, 0;", cmp, pred, byte_reg)?;
    emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, pred)?;
    Ok(())
}

/// Emit PTX for `Op::Select`: pick `then_val` when `cond` is true (Bool 1)
/// and `else_val` otherwise, materialising the chosen value in `dst`.
///
/// Wire shape (for a Float32 example):
///
/// ```text
///   setp.ne.s32 %p,    %cond, 0           // Bool 0/1 -> predicate
///   selp.f32    %dst,  %then, %else, %p   // dst = p ? then : else
/// ```
///
/// `cond` must live in the b32 (`r`) register class because the codegen
/// invariant says every Bool value sits there (see `RegAlloc::class_for`).
/// The materialisation uses `setp.ne.s32 cond, 0` so any nonzero Bool
/// register (defensively wider than {0, 1}) still picks the THEN branch.
///
/// Supported value dtypes (v0.7 envelope):
///
/// * `Bool` / `Int32` -> `selp.s32`
/// * `Int64`          -> `selp.s64`
/// * `Float32`        -> `selp.f32`
/// * `Float64`        -> `selp.f64`
/// * `Date32`         -> `selp.b32`  (i32 storage, bit-copy)
/// * `Timestamp`      -> `selp.b64`  (i64 storage, bit-copy)
///
/// `Codegen::emit_case` rejects Utf8 / Decimal128 at the plan layer with a
/// tighter message (Decimal128 is i128 — no `selp.b128`), so by the time we
/// get here the dtype envelope is guaranteed.
fn emit_select(
    b: &mut PtxBuilder,
    dst: Reg,
    cond: Reg,
    then_val: Reg,
    else_val: Reg,
    dtype: DataType,
) -> BoltResult<()> {
    // PERF (codegen alloc): operand names are borrowed inline at the write
    // sites below (no `.to_string()`); all allocator mutation (`assign` of the
    // destination + the `p` predicate) happens first.
    let selp_ty = match dtype {
        // Bool values live in the b32 (r) class same as Int32.
        DataType::Bool | DataType::Int32 => "s32",
        DataType::Int64 => "s64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        // v0.7: Date32 (i32 storage) and Timestamp (i64 storage) are plain
        // fixed-width integers. `selp` just copies the chosen operand's bits,
        // so the untyped bit-class suffixes `b32` / `b64` are the natural fit
        // — no arithmetic interpretation of the value is needed. They live in
        // the same `r` / `rl` register classes as Int32 / Int64 (see
        // `RegAlloc::class_for`), so the operand registers are already correct.
        DataType::Date32 => "b32",
        DataType::Timestamp(_, _) => "b64",
        DataType::Utf8 => {
            return Err(BoltError::Other(
                "ptx_gen: Select over Utf8 not supported \
                 (planner should have rejected CASE over string types)"
                    .into(),
            ))
        }
        DataType::Decimal128(_, _) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
            ))
        }
    };
    let dst_name = b.alloc.assign(dst, dtype)?;
    let pred = b.alloc.alloc("p");
    emit_fmt!(b, "setp.ne.s32 {}, {}, 0;", pred, b.alloc.get(cond)?)?;
    emit_fmt!(
        b,
        "selp.{} {}, {}, {}, {};",
        selp_ty,
        dst_name,
        b.alloc.get(then_val)?,
        b.alloc.get(else_val)?,
        pred
    )
}

/// Emit PTX for `Op::Not`: logical negation of a Bool register.
///
/// Every Bool value is a canonical {0, 1} in the b32 (`r`) register class
/// (see `RegAlloc::class_for` and the comparison / `Op::Select` emitters
/// that produce them), so negation is a single low-bit flip:
///
/// ```text
///   xor.b32 %dst, %src, 1;
/// ```
///
/// `1` toggles bit 0, mapping 0 -> 1 and 1 -> 0; the result stays a
/// canonical {0, 1} Bool. `Codegen::emit_unary` guarantees `src` is a Bool
/// register before pushing this op.
fn emit_not(b: &mut PtxBuilder, dst: Reg, src: Reg) -> BoltResult<()> {
    // PERF (codegen alloc): assign the destination, then reference `src`
    // inline at the write site (no `.to_string()` temporary).
    let dst_name = b.alloc.assign(dst, DataType::Bool)?;
    emit_fmt!(b, "xor.b32 {}, {}, 1;", dst_name, b.alloc.get(src)?)
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
    // PERF (codegen alloc): write instructions straight into `b.body` via
    // `emit_fmt!`, dropping the per-line `format!` temporaries. `input_ptrs`
    // and `tid` are caller-owned borrows, not `b.alloc` lookups.
    let width = byte_width(dtype)?;
    let off = b.alloc.alloc("rd");
    let addr = b.alloc.alloc("rd");
    emit_fmt!(b, "mul.wide.u32 {}, {}, {};", off, tid, width)?;
    emit_fmt!(b, "add.s64 {}, {}, {};", addr, input_ptrs[col_idx], off)?;
    let dst_name = b.alloc.assign(dst, dtype)?;
    let suffix = ld_st_suffix(dtype)?;
    emit_fmt!(b, "ld.global.nc.{} {}, [{}];", suffix, dst_name, addr)?;
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
    // PERF (codegen alloc): `src` operand name is borrowed inline in the
    // final store (no `.to_string()`); the `off`/`addr` allocations precede
    // it, so the immutable `b.alloc.get` borrow does not overlap a `&mut`.
    let width = byte_width(dtype)?;
    let off = b.alloc.alloc("rd");
    let addr = b.alloc.alloc("rd");
    emit_fmt!(b, "mul.wide.u32 {}, {}, {};", off, tid, width)?;
    emit_fmt!(b, "add.s64 {}, {}, {};", addr, output_ptrs[col_idx], off)?;
    let suffix = ld_st_suffix(dtype)?;
    emit_fmt!(b, "st.global.{} [{}], {};", suffix, addr, b.alloc.get(src)?)?;
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
        // v0.7: Date32 / Timestamp literals lower to integer constants.
        // Date32 is i32 days-since-epoch; Timestamp is i64 ticks-since-epoch
        // in the source unit. Same hex-bit-pattern emission convention as
        // the Int32 / Int64 paths (no codegen-injection surface).
        // PERF (codegen alloc): every arm emits a single `mov` straight into
        // `b.body` via `emit_fmt!`, dropping the per-line `format!` temporary.
        Literal::Date32(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Date32)?;
            emit_fmt!(b, "mov.s32 {}, 0x{:08X};", dst_name, *v as u32)
        }
        Literal::Timestamp(v, unit, tz) => {
            let dst_name = b.alloc.assign(dst, DataType::Timestamp(*unit, *tz))?;
            emit_fmt!(b, "mov.s64 {}, 0x{:016X};", dst_name, *v as u64)
        }
        Literal::Bool(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Bool)?;
            // Value space is {0, 1}; not an injection surface, but keep the
            // emission consistent with the other integer paths for clarity.
            let n: u32 = if *v { 1 } else { 0 };
            emit_fmt!(b, "mov.b32 {}, {};", dst_name, n)
        }
        Literal::Int32(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Int32)?;
            // Emit the bit-pattern as hex: `mov.s32` is a bitwise copy, so
            // `0xFFFFFFFF` here is -1, identical to writing `-1`. This avoids
            // any sign / INT32_MIN parsing concerns AND removes the codegen-
            // injection surface (output is restricted to `[0-9A-F]`).
            emit_fmt!(b, "mov.s32 {}, 0x{:08X};", dst_name, *v as u32)
        }
        Literal::Int64(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Int64)?;
            emit_fmt!(b, "mov.s64 {}, 0x{:016X};", dst_name, *v as u64)
        }
        Literal::Float32(v) => {
            // Already hex-encoded via PTX `0f<8 hex>` syntax — no injection surface.
            let dst_name = b.alloc.assign(dst, DataType::Float32)?;
            emit_fmt!(b, "mov.f32 {}, 0f{:08X};", dst_name, v.to_bits())
        }
        Literal::Float64(v) => {
            // Already hex-encoded via PTX `0d<16 hex>` syntax — no injection surface.
            let dst_name = b.alloc.assign(dst, DataType::Float64)?;
            emit_fmt!(b, "mov.f64 {}, 0d{:016X};", dst_name, v.to_bits())
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
    // PERF (codegen alloc): `src_name`/`dst_name` are kept as owned Strings
    // here for a borrow-checker reason — the Numeric->Bool arms below allocate
    // a `p` predicate (`&mut b.alloc`) between this read and the use, so a held
    // `&str` borrow would not survive. The remaining per-line `format!`
    // temporaries are removed by routing the predicate emits through
    // `emit_fmt!` and the final instruction through `emit_fmt!` as well.
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
                // v0.7: identity-cast on Date32 / Timestamp is a typed mov
                // on the underlying integer width. Same logical dtype on
                // both sides so the register class stays consistent.
                Date32 => "s32",
                Timestamp(_, _) => "s64",
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
            emit_fmt!(b, "setp.ne.s32 {}, {}, 0;", p, src_name)?;
            format!("selp.s32 {}, 1, 0, {};", dst_name, p)
        }
        (Int64, Bool) => {
            let p = b.alloc.alloc("p");
            emit_fmt!(b, "setp.ne.s64 {}, {}, 0;", p, src_name)?;
            format!("selp.s32 {}, 1, 0, {};", dst_name, p)
        }
        (Float32, Bool) => {
            let p = b.alloc.alloc("p");
            emit_fmt!(b, "setp.ne.f32 {}, {}, 0f00000000;", p, src_name)?;
            format!("selp.s32 {}, 1, 0, {};", dst_name, p)
        }
        (Float64, Bool) => {
            let p = b.alloc.alloc("p");
            emit_fmt!(b, "setp.ne.f64 {}, {}, 0d0000000000000000;", p, src_name)?;
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

    emit_fmt!(b, "{}", instr)
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
    // PERF (codegen alloc): `lhs`/`rhs` operand names are borrowed inline at
    // each `emit_fmt!` (no `.to_string()`). Within every arm all allocator
    // mutation (`assign` of the destination, any `p` predicate) happens before
    // the writes, so the transient `b.alloc.get(...)` borrows never overlap a
    // `&mut b.alloc`.
    use BinaryOp::*;
    match op {
        Add | Sub | Mul | Div => {
            // Arithmetic preserves the operand dtype for numerics. v0.7
            // adds two temporal-Sub shapes that differ from numeric arith:
            //   * Date32  - Date32          → Int32 (days count)
            //   * Timestamp - Timestamp     → Int64 (tick count in source unit)
            // The result_dtype is the underlying integer width because the
            // difference is a unit-less count, not a calendar value. The
            // PTX is otherwise identical to the corresponding integer
            // sub.s32 / sub.s64.
            let is_temporal_sub = matches!(op, Sub)
                && match (dtype, result_dtype) {
                    (DataType::Date32, DataType::Int32) => true,
                    (DataType::Timestamp(_, _), DataType::Int64) => true,
                    _ => false,
                };
            if !is_temporal_sub {
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
            }
            let dst_name = b.alloc.assign(dst, result_dtype)?;
            // JIT-DIV0/INT_MIN UB fix: integer `Div` must NOT lower to a bare
            // `div.s32`/`div.s64`. On the GPU both `b == 0` and the single
            // overflow case `INT_MIN / -1` are UNDEFINED (div-by-zero is UB;
            // `INT_MIN / -1` overflows the signed result). Route signed
            // integer division through a guarded sequence that produces a
            // fully-defined result in every case. Float division keeps the
            // historical single-instruction emission (IEEE `div.rn.*` is
            // already total — `x/0.0` yields ±inf/NaN, no UB).
            if matches!(op, Div) && matches!(dtype, DataType::Int32 | DataType::Int64) {
                return emit_int_div_guarded(b, &dst_name, lhs, rhs, dtype);
            }
            let mnemonic = arith_mnemonic(op, dtype)?;
            emit_fmt!(
                b,
                "{} {}, {}, {};",
                mnemonic,
                dst_name,
                b.alloc.get(lhs)?,
                b.alloc.get(rhs)?
            )
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
            emit_fmt!(
                b,
                "{} {}, {}, {};",
                cmp,
                p,
                b.alloc.get(lhs)?,
                b.alloc.get(rhs)?
            )?;
            emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, p)
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
            emit_fmt!(
                b,
                "{} {}, {}, {};",
                mnemonic,
                dst_name,
                b.alloc.get(lhs)?,
                b.alloc.get(rhs)?
            )
        }
        Concat => {
            // The `||` BINARY OPERATOR (this arm) is a fused-kernel scalar op
            // that has no variable-width output-buffer allocation, so it stays
            // host-routed: the physical-plan lowerer routes any expression that
            // contains `BinaryOp::Concat` through `PhysicalPlan::Project` (SELECT
            // list, v0.5) or `PhysicalPlan::Filter` (WHERE predicate, v0.7) —
            // both backed by `expr_agg::eval_expr`. Reaching this arm therefore
            // indicates a missing route; surface a clear error rather than
            // emitting nonsense PTX.
            //
            // NOTE: the CONCAT *function* (`ScalarFnKind::Concat`,
            // `CONCAT(a, b, ...)`) is a SEPARATE path — it now HAS a GPU
            // producer via the N-input two-pass kernels
            // (`jit::string_kernel::compile_concat_{len,write}_pass`) wired
            // through `PhysicalPlan::StringProject` (see
            // `physical_plan::try_lower_string_project`), with a host fallback
            // (`exec::string_project::host_concat_strings`). That path never
            // reaches `emit_binary`; only the `||` operator does.
            Err(BoltError::Other(
                "ptx_gen: string concat operator (||) is not lowered to GPU; \
                 the planner should route this through the host-side \
                 PhysicalPlan::Project (SELECT) or PhysicalPlan::Filter \
                 (WHERE) executor"
                    .into(),
            ))
        }
    }
}

/// Emit a guarded **signed integer** division `dst = lhs / rhs` that is
/// fully defined for every operand pair, removing two GPU undefined-behavior
/// hazards that a bare `div.s32`/`div.s64` leaves open:
///
///   1. **Divide-by-zero** (`rhs == 0`). On the GPU integer div-by-zero is
///      UB (the result is unspecified; on some architectures it can fault).
///      SQL/DuckDB semantics treat integer divide-by-zero as producing
///      **NULL**. The proper fix is to mark the output row NULL via the
///      kernel's per-row validity bitmap — but `emit_binary` runs in a
///      codegen context that has **no access** to the output-validity
///      pointers (those are wired up in `compile`, keyed by output column,
///      not by intermediate `Reg`). So at this layer we do the next-best,
///      UB-removing thing: divide by a safe non-zero stand-in and `selp` a
///      **defined `0`** into `dst` when the real divisor was zero. The result
///      is deterministic (no UB / no fault) but is `0`, not NULL.
///
///      LIMITATION (documented intentionally): the divide-by-zero row is NOT
///      flagged NULL here. The conservative AND-of-inputs validity fold in
///      `compile` still NULLs the row if any *input* was NULL, but a
///      genuine `x / 0` over non-NULL inputs currently yields `0` rather than
///      SQL-NULL. Closing that gap requires threading a per-op "computed
///      NULL" signal into the output-validity store, which is a larger IR
///      change tracked separately. The critical property delivered here is:
///      **no undefined behavior / no hardware trap**.
///
///   2. **Signed overflow `INT_MIN / -1`**. The mathematical result
///      (`+2^31` / `+2^63`) is unrepresentable, so the hardware result is
///      undefined and can trap. We detect this single case and produce the
///      wrapping result `INT_MIN` (matching two's-complement wraparound,
///      which is what callers/SQL engines expect for this corner).
///
/// PTX shape (Int32; Int64 is identical with `.s64` / the `rl` class and the
/// 64-bit `INT_MIN` literal):
/// ```text
///   setp.eq.s32     %p_dz,  %rhs, 0;            // divisor == 0 ?
///   selp.s32        %safe,  1, %rhs, %p_dz;     // avoid UB: divide by 1 when 0
///   setp.eq.s32     %p_min, %lhs, -2147483648;  // lhs == INT_MIN ?
///   setp.eq.s32     %p_m1,  %rhs, -1;           // rhs == -1 ?
///   and.pred        %p_ovf, %p_min, %p_m1;      // INT_MIN / -1 overflow ?
///   selp.s32        %safe2, 1, %safe, %p_ovf;   // also dodge the trap divisor
///   div.s32         %q,     %lhs, %safe2;       // defined division
///   selp.s32        %q,     %lhs, %q, %p_ovf;   // overflow -> INT_MIN (== lhs)
///   selp.s32        %dst,   0,    %q, %p_dz;    // div-by-zero -> defined 0
/// ```
fn emit_int_div_guarded(
    b: &mut PtxBuilder,
    dst_name: &str,
    lhs: Reg,
    rhs: Reg,
    dtype: DataType,
) -> BoltResult<()> {
    // PTX type suffix + value register class for this integer width, and the
    // two's-complement minimum literal used for the overflow corner.
    let (suf, class, int_min) = match dtype {
        DataType::Int32 => ("s32", "r", "-2147483648"),
        DataType::Int64 => ("s64", "rl", "-9223372036854775808"),
        // Caller (`emit_binary`) only routes Int32/Int64 here.
        _ => {
            return Err(BoltError::Other(format!(
                "ptx_gen: emit_int_div_guarded called on non-integer dtype {:?}",
                dtype
            )))
        }
    };

    // Resolve operand names up front (immutable `alloc` borrows) before we
    // start allocating temporaries; copy into owned Strings so the later
    // `&mut b.alloc.alloc(..)` calls don't overlap a live borrow.
    let lhs_name = b.alloc.get(lhs)?.to_string();
    let rhs_name = b.alloc.get(rhs)?.to_string();

    // Predicates live in the `p` class; the safe-divisor / quotient temporaries
    // share the value register class of the dtype.
    let p_dz = b.alloc.alloc("p");
    let p_min = b.alloc.alloc("p");
    let p_m1 = b.alloc.alloc("p");
    let p_ovf = b.alloc.alloc("p");
    let safe = b.alloc.alloc(class);
    let safe2 = b.alloc.alloc(class);
    let q = b.alloc.alloc(class);

    // Divide-by-zero detection + a non-zero stand-in divisor (1) to keep the
    // `div` instruction itself defined.
    emit_fmt!(b, "setp.eq.{} {}, {}, 0;", suf, p_dz, rhs_name)?;
    emit_fmt!(b, "selp.{} {}, 1, {}, {};", suf, safe, rhs_name, p_dz)?;
    // INT_MIN / -1 overflow detection.
    emit_fmt!(b, "setp.eq.{} {}, {}, {};", suf, p_min, lhs_name, int_min)?;
    emit_fmt!(b, "setp.eq.{} {}, {}, -1;", suf, p_m1, rhs_name)?;
    emit_fmt!(b, "and.pred {}, {}, {};", p_ovf, p_min, p_m1)?;
    // Also avoid feeding the trapping (INT_MIN, -1) pair to `div`.
    emit_fmt!(b, "selp.{} {}, 1, {}, {};", suf, safe2, safe, p_ovf)?;
    // Defined division by the sanitised divisor.
    emit_fmt!(b, "div.{} {}, {}, {};", suf, q, lhs_name, safe2)?;
    // Overflow corner: the wrapping result is INT_MIN, which equals lhs here.
    emit_fmt!(b, "selp.{} {}, {}, {}, {};", suf, q, lhs_name, q, p_ovf)?;
    // Divide-by-zero: produce a defined 0 (see LIMITATION in the doc comment).
    emit_fmt!(b, "selp.{} {}, 0, {}, {};", suf, dst_name, q, p_dz)
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
        // v0.7: Date32 / Timestamp arithmetic. The physical-plan lowerer
        // only emits `Sub` for these types (Date32 - Date32 → Int32 days;
        // Timestamp(u, tz) - Timestamp(u, tz) → Int64 ticks); any other
        // op surfaces here as an "unsupported" error below.
        (Sub, Date32) => "sub.s32",
        (Sub, Timestamp(_, _)) => "sub.s64",
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
        // v0.7: Date32 / Timestamp comparisons fall through to integer
        // setp on the underlying days / ticks. The logical type-checker
        // already enforces matching unit / tz on Timestamp operands; at
        // the PTX level it's identical to the corresponding integer cmp.
        Date32 => "s32",
        Timestamp(_, _) => "s64",
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
        // v0.7: Date32 is i32 days; Timestamp is i64 ticks. Storage layout
        // matches the underlying integer type bit-for-bit.
        DataType::Date32 => "s32",
        DataType::Timestamp(_, _) => "s64",
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
///
/// V-12: this is the single source of truth for kernel-name validation. The
/// scan-kernel codegen path (`scan_kernel::compile`) calls this same function
/// rather than maintaining a weaker duplicate, so both external-name consumers
/// share identical hardening (charset, leading char, PTX reserved words, `__`
/// prefix, and the `_param_` substring check).
pub(crate) fn validate_kernel_name(name: &str) -> BoltResult<()> {
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
///
/// dedup (ptx_common): intentionally NOT shared with
/// `scan_kernel::write_signature`. They emit different bytes — this one uses
/// the `inputs + outputs + extra` param formula and `.param .u64 .ptr .global
/// .align 16` pointer attributes; the scan variant uses `n_inputs + 1 + K`
/// (mask output) with plain `.param .u64`. Both shapes are pinned by goldens.
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
///
/// dedup (ptx_common): intentionally NOT shared with
/// `scan_kernel::write_reg_decls`, which declares an extra `("rs", "b16")`
/// class (7 vs 6) for its narrowed mask byte. Different `decls` => different
/// emitted block.
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
///
/// dedup (ptx_common): intentionally NOT shared. Every JIT codegen module
/// keeps its own one-liner so the `write failed` error is prefixed with the
/// owning module name (here `ptx_gen:`). Sharing would change that text.
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
    fn validity_offset_widens_unsigned() {
        // C-3 regression: validity-byte addressing MUST widen the row index
        // UNSIGNED (`mul.wide.u32 .., 1`) to match the value-load path
        // (`emit_load`/`emit_load_128` use `mul.wide.u32`). A signed
        // `cvt.s64.s32` would sign-extend `tid` once it crosses 2^31 rows and
        // compute a huge negative validity offset → OOB load/store, while the
        // value load at the same row stays correct. This test locks down the
        // signed→unsigned fix on both the input-AND fold and the output store.
        let spec = mul_with_validity_spec();
        let ptx = compile(&spec, "bolt_pre_kernel_validity_widen").expect("compile");

        // The validity path uses the unsigned-widen stride-1 form.
        assert!(
            ptx.contains("mul.wide.u32") && ptx.contains(", 1;"),
            "expected `mul.wide.u32 .., 1;` unsigned widen on the validity path\n{ptx}"
        );

        // And it must NOT use the signed widen for offset arithmetic. The only
        // legitimate `cvt.s64.s32` is an Int32->Int64 CAST, which this spec
        // (Mul of two value columns + validity) never emits.
        assert!(
            !ptx.contains("cvt.s64.s32"),
            "validity path must not sign-extend the row index via cvt.s64.s32\n{ptx}"
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

        // C-3: the IsNullCheck validity offset is widened UNSIGNED via
        // `mul.wide.u32 .., 1` (stride-1 byte addressing) — never the signed
        // `cvt.s64.s32`, which would mis-address above 2^31 rows. This spec
        // emits no Int32->Int64 CAST, so `cvt.s64.s32` must be entirely absent.
        assert!(
            ptx.contains("mul.wide.u32") && ptx.contains(", 1;"),
            "expected `mul.wide.u32 .., 1;` unsigned widen for the validity offset\n{ptx}"
        );
        assert!(
            !ptx.contains("cvt.s64.s32"),
            "IsNullCheck validity offset must not sign-extend via cvt.s64.s32\n{ptx}"
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

    // ---- v0.7: BinaryOp::Concat in a kernel spec must reject at PTX-gen ----
    //
    // The physical-plan lowerer routes every Concat-bearing expression
    // through a host-side executor (`PhysicalPlan::Project` for SELECT lists,
    // `PhysicalPlan::Filter` for WHERE predicates), so a `BinaryOp::Concat`
    // op should never reach this codegen. The arm in `emit_binary` is the
    // last-line guard for a planner regression; these tests pin both the
    // shapes the WHERE path now lowers cleanly and the error surface a
    // hand-built kernel would see if Concat ever leaked through.

    /// Hand-built kernel for `a || b` over two Utf8 columns. The PTX emitter
    /// rejects Utf8 inputs eagerly at the parameter walk (before any op is
    /// emitted), so we can't actually get into the `Concat` arm of
    /// `emit_binary` via the public `compile` entry point — Utf8 inputs
    /// fire the gate first. The right contract is therefore: a kernel spec
    /// that contains a `BinaryOp::Concat` op MUST surface a `BoltError`
    /// from `compile`. We assert the Utf8-input rejection here because the
    /// downstream Concat-arm rejection is unreachable for a well-formed
    /// spec (Concat's operands are necessarily Utf8).
    #[test]
    fn concat_a_b_eq_foo_compile_rejects_utf8_inputs() {
        // Hand-built `WHERE a || b = 'foo'` projection kernel. The physical
        // planner would NOT emit this — it routes the predicate to the host
        // filter — but a hand-build round-trips the rejection so a future
        // planner regression that misroutes the Concat surfaces here.
        let spec = KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "a".into(),
                    dtype: DataType::Utf8,
                },
                ColumnIO {
                    name: "b".into(),
                    dtype: DataType::Utf8,
                },
            ],
            outputs: vec![ColumnIO {
                name: "out".into(),
                dtype: DataType::Bool,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Utf8,
                },
                Op::LoadColumn {
                    dst: Reg(1),
                    col_idx: 1,
                    dtype: DataType::Utf8,
                },
                // a || b → Utf8 (the op the planner is forbidden to emit
                // into a GPU kernel; rejected at PTX-gen). Result register
                // dtype is Utf8 — the emitter never gets to allocate it
                // because the Utf8 input gate fires first.
                Op::Binary {
                    dst: Reg(2),
                    op: BinaryOp::Concat,
                    lhs: Reg(0),
                    rhs: Reg(1),
                    dtype: DataType::Utf8,
                    result_dtype: DataType::Utf8,
                },
                Op::Store {
                    src: Reg(2),
                    col_idx: 0,
                    dtype: DataType::Bool,
                },
            ],
            predicate: None,
            register_count: 3,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let err = compile(&spec, "bolt_concat_a_b_eq_foo").expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("Utf8"),
            "expected Utf8 rejection (since Concat operands are Utf8), got: {msg}"
        );
    }

    /// Companion: `'a' || b` — a Utf8-literal-on-left shape. Same outcome
    /// as the column||column case (Utf8 input rejection fires first), but
    /// pinning both shapes makes the regression message obvious whichever
    /// half a future planner bug hits first.
    #[test]
    fn concat_literal_b_eq_ab_compile_rejects_utf8_input() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "b".into(),
                dtype: DataType::Utf8,
            }],
            outputs: vec![ColumnIO {
                name: "out".into(),
                dtype: DataType::Bool,
            }],
            ops: vec![
                Op::Const {
                    dst: Reg(0),
                    lit: Literal::Utf8("a".into()),
                },
                Op::LoadColumn {
                    dst: Reg(1),
                    col_idx: 0,
                    dtype: DataType::Utf8,
                },
                Op::Binary {
                    dst: Reg(2),
                    op: BinaryOp::Concat,
                    lhs: Reg(0),
                    rhs: Reg(1),
                    dtype: DataType::Utf8,
                    result_dtype: DataType::Utf8,
                },
                Op::Store {
                    src: Reg(2),
                    col_idx: 0,
                    dtype: DataType::Bool,
                },
            ],
            predicate: None,
            register_count: 3,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let err = compile(&spec, "bolt_concat_lit_b_eq_ab").expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("Utf8"),
            "expected Utf8 rejection on hand-built `'a' || b` kernel, got: {msg}"
        );
    }

    /// The v0.7 lowering contract: `WHERE a || b = 'foo'` must NOT raise an
    /// error from the physical-plan lowerer. The route lives in
    /// `physical_plan::lower_depth`'s Filter arm, which detects
    /// `BinaryOp::Concat` in the predicate and routes the whole Filter
    /// through the host-side `PhysicalPlan::Filter` executor. This test
    /// double-checks the public `lower_physical` re-export, complementing
    /// the structural tests in `physical_plan.rs::tests`.
    #[test]
    fn where_concat_eq_lowers_via_public_api_without_error() {
        use crate::plan::logical_plan::{Expr, Field, LogicalPlan, Schema};
        use crate::plan::lower_physical;
        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("a", DataType::Utf8, false),
                Field::new("b", DataType::Utf8, false),
            ]),
        };
        let pred = Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(Expr::Binary {
                op: BinaryOp::Concat,
                left: Box::new(Expr::Column("a".into())),
                right: Box::new(Expr::Column("b".into())),
            }),
            right: Box::new(Expr::Literal(Literal::Utf8("foo".into()))),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate: pred,
        };
        let _phys = lower_physical(&plan)
            .expect("WHERE a || b = 'foo' must lower via the public API in v0.7");
    }

    /// `WHERE 'a' || b = 'ab'` — literal-on-left companion to the column-
    /// on-left case. Locks the routing for both binary shapes.
    #[test]
    fn where_literal_concat_b_eq_lowers_via_public_api_without_error() {
        use crate::plan::logical_plan::{Expr, Field, LogicalPlan, Schema};
        use crate::plan::lower_physical;
        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("b", DataType::Utf8, false)]),
        };
        let pred = Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(Expr::Binary {
                op: BinaryOp::Concat,
                left: Box::new(Expr::Literal(Literal::Utf8("a".into()))),
                right: Box::new(Expr::Column("b".into())),
            }),
            right: Box::new(Expr::Literal(Literal::Utf8("ab".into()))),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate: pred,
        };
        let _phys = lower_physical(&plan)
            .expect("WHERE 'a' || b = 'ab' must lower via the public API in v0.7");
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

    // ---- Op::Select (CASE WHEN) -------------------------------------------

    /// v0.7 CASE WHEN lowering: a single-WHEN `Op::Select` must materialise
    /// the chosen value via `setp.ne.s32` (Bool 0/1 -> predicate) followed
    /// by `selp.<ty>` on the value class. Mirrors the contract documented
    /// on `Op::Select` in physical_plan.rs and `emit_select` above.
    ///
    /// Spec shape (logically `CASE WHEN cond THEN then ELSE else END` where
    /// `cond` is a Bool column and the value branches are Int32 columns):
    ///
    /// ```text
    ///   r0 = LoadColumn(0)   ; cond  (Bool)
    ///   r1 = LoadColumn(1)   ; then  (Int32)
    ///   r2 = LoadColumn(2)   ; else  (Int32)
    ///   r3 = Select(r0, r1, r2, Int32)
    ///   Store(r3 -> out0, Int32)
    /// ```
    #[test]
    fn select_emits_setp_and_selp_s32() {
        let spec = KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "cond".into(),
                    dtype: DataType::Bool,
                },
                ColumnIO {
                    name: "t".into(),
                    dtype: DataType::Int32,
                },
                ColumnIO {
                    name: "e".into(),
                    dtype: DataType::Int32,
                },
            ],
            outputs: vec![ColumnIO {
                name: "out".into(),
                dtype: DataType::Int32,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Bool,
                },
                Op::LoadColumn {
                    dst: Reg(1),
                    col_idx: 1,
                    dtype: DataType::Int32,
                },
                Op::LoadColumn {
                    dst: Reg(2),
                    col_idx: 2,
                    dtype: DataType::Int32,
                },
                Op::Select {
                    dst: Reg(3),
                    cond: Reg(0),
                    then_val: Reg(1),
                    else_val: Reg(2),
                    dtype: DataType::Int32,
                },
                Op::Store {
                    src: Reg(3),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
            ],
            predicate: None,
            register_count: 4,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };

        let ptx = compile(&spec, "bolt_select_s32").expect("compile");

        // The Bool -> predicate materialisation is `setp.ne.s32 ..., cond, 0`.
        assert!(
            ptx.contains("setp.ne.s32"),
            "expected setp.ne.s32 to turn the Bool cond register into a %p\n{ptx}"
        );

        // The value selection uses `selp.s32` for the Int32 branch dtype.
        assert!(
            ptx.contains("selp.s32"),
            "expected selp.s32 for Int32 CASE branches\n{ptx}"
        );

        // 3 inputs + 1 output = 4 pointer params, no validity wiring.
        let n_ptr_params = ptx.matches(".param .u64 .ptr").count();
        assert_eq!(
            n_ptr_params, 4,
            "expected 4 .ptr params (3 in + 1 out, no validity), got {n_ptr_params}\n{ptx}"
        );
    }

    /// `selp.f64` is used for Float64-valued CASE branches; the predicate
    /// materialisation is the same `setp.ne.s32` regardless of value dtype.
    #[test]
    fn select_uses_selp_f64_for_float64_branches() {
        let spec = KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "cond".into(),
                    dtype: DataType::Bool,
                },
                ColumnIO {
                    name: "t".into(),
                    dtype: DataType::Float64,
                },
                ColumnIO {
                    name: "e".into(),
                    dtype: DataType::Float64,
                },
            ],
            outputs: vec![ColumnIO {
                name: "out".into(),
                dtype: DataType::Float64,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Bool,
                },
                Op::LoadColumn {
                    dst: Reg(1),
                    col_idx: 1,
                    dtype: DataType::Float64,
                },
                Op::LoadColumn {
                    dst: Reg(2),
                    col_idx: 2,
                    dtype: DataType::Float64,
                },
                Op::Select {
                    dst: Reg(3),
                    cond: Reg(0),
                    then_val: Reg(1),
                    else_val: Reg(2),
                    dtype: DataType::Float64,
                },
                Op::Store {
                    src: Reg(3),
                    col_idx: 0,
                    dtype: DataType::Float64,
                },
            ],
            predicate: None,
            register_count: 4,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };

        let ptx = compile(&spec, "bolt_select_f64").expect("compile");
        assert!(
            ptx.contains("setp.ne.s32"),
            "Bool cond -> predicate materialisation should still be setp.ne.s32\n{ptx}"
        );
        assert!(
            ptx.contains("selp.f64"),
            "Float64 branch dtype must use selp.f64\n{ptx}"
        );
        assert!(
            !ptx.contains("selp.s32"),
            "Float64-branch CASE must NOT emit selp.s32 (would truncate value)\n{ptx}"
        );
    }

    /// A two-WHEN CASE folds to a chain of two Selects. The PTX must
    /// contain at least two `selp.<ty>` instructions and the inner Select's
    /// dst must thread into the outer Select's `else_val`.
    #[test]
    fn nested_selects_emit_two_selp_instructions() {
        // Logical: CASE WHEN c0 THEN t0 WHEN c1 THEN t1 ELSE e END
        //   inner = Select(c1, t1, e)
        //   outer = Select(c0, t0, inner)
        let spec = KernelSpec {
            inputs: vec![
                // c0 (Bool), c1 (Bool), t0 (Int32), t1 (Int32), e (Int32)
                ColumnIO {
                    name: "c0".into(),
                    dtype: DataType::Bool,
                },
                ColumnIO {
                    name: "c1".into(),
                    dtype: DataType::Bool,
                },
                ColumnIO {
                    name: "t0".into(),
                    dtype: DataType::Int32,
                },
                ColumnIO {
                    name: "t1".into(),
                    dtype: DataType::Int32,
                },
                ColumnIO {
                    name: "e".into(),
                    dtype: DataType::Int32,
                },
            ],
            outputs: vec![ColumnIO {
                name: "out".into(),
                dtype: DataType::Int32,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Bool,
                },
                Op::LoadColumn {
                    dst: Reg(1),
                    col_idx: 1,
                    dtype: DataType::Bool,
                },
                Op::LoadColumn {
                    dst: Reg(2),
                    col_idx: 2,
                    dtype: DataType::Int32,
                },
                Op::LoadColumn {
                    dst: Reg(3),
                    col_idx: 3,
                    dtype: DataType::Int32,
                },
                Op::LoadColumn {
                    dst: Reg(4),
                    col_idx: 4,
                    dtype: DataType::Int32,
                },
                // inner = Select(c1, t1, e)
                Op::Select {
                    dst: Reg(5),
                    cond: Reg(1),
                    then_val: Reg(3),
                    else_val: Reg(4),
                    dtype: DataType::Int32,
                },
                // outer = Select(c0, t0, inner)
                Op::Select {
                    dst: Reg(6),
                    cond: Reg(0),
                    then_val: Reg(2),
                    else_val: Reg(5),
                    dtype: DataType::Int32,
                },
                Op::Store {
                    src: Reg(6),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
            ],
            predicate: None,
            register_count: 7,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };

        let ptx = compile(&spec, "bolt_select_nested").expect("compile");
        let n_selp = ptx.matches("selp.s32").count();
        assert!(
            n_selp >= 2,
            "two-WHEN CASE must emit at least two selp.s32 instructions, got {n_selp}\n{ptx}"
        );
        let n_setp = ptx.matches("setp.ne.s32").count();
        assert!(
            n_setp >= 2,
            "two-WHEN CASE must materialise each Bool cond via setp.ne.s32, got {n_setp}\n{ptx}"
        );
    }
}

// ---------------------------------------------------------------------------
// Scalar string functions: substrate + lowering-status coverage.
//
// Two distinct boundaries are exercised here; keep them separate:
//
//   1. The *fused PTX codegen* (`compile` in this file) still has no Utf8
//      register class: it rejects any `KernelSpec` carrying a Utf8 input or
//      output column eagerly at the parameter walk with "Utf8 not supported in
//      PTX codegen yet" (see the `inputs` / `outputs` loops near the top of
//      `compile`). This is unchanged and unrelated to whether a SQL string
//      function can run on the GPU — the GPU string path does NOT go through
//      this fused kernel.
//
//   2. The *physical-plan lowering* (`crate::plan::physical_plan::lower`) has
//      since grown dedicated GPU string producers, so the scalar string
//      functions are no longer uniformly rejected. Current status per
//      `ScalarFnKind` (verified against `lower`'s `LogicalPlan::Project` arm):
//
//        * UPPER / LOWER         → GPU `PhysicalPlan::StringProject`
//                                  (`try_lower_string_project`). lower() OK.
//        * LENGTH(<bare col>)    → GPU `PhysicalPlan::StringLength`
//                                  (`try_lower_string_length`). lower() OK.
//        * SUBSTRING / TRIM*     → host-side `PhysicalPlan::Project`
//                                  (`all_scalar_fns_host_evaluable`). lower() OK.
//        * CONCAT (as ScalarFn)  → STILL rejected at lowering with
//                                  "string scalar function CONCAT is not yet
//                                  lowered to GPU ... (coming in a follow-up)".
//
// The first two `compile`-preflight tests below pin boundary (1) (the PTX
// emitter refuses Utf8 IO). The integration test pins boundary (2): the
// supported kinds lower without error and CONCAT — the one kind still
// rejected — names the function in its actionable rejection message.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod scalar_string_fn_substrate_tests {
    use super::*;
    use crate::plan::physical_plan::{ColumnIO, KernelSpec, Op, Reg};

    /// The *fused PTX codegen* (`compile`) has no Utf8 register class: a
    /// `KernelSpec` declaring a Utf8 output is rejected at the
    /// parameter-emission preflight (the loop in `compile` over
    /// `spec.outputs`) with "Utf8 not supported in PTX codegen yet".
    ///
    /// NOTE: this is a property of the fused kernel only. The GPU string
    /// functions (UPPER / LOWER via `PhysicalPlan::StringProject`, LENGTH via
    /// `PhysicalPlan::StringLength`) do NOT route through this `compile` path —
    /// they have their own dedicated producers in `crate::exec::string_project`
    /// / `string_length`. So this test pins the fused-kernel Utf8-output gate,
    /// not the (now-supported) high-level lowering of those SQL functions.
    #[test]
    fn utf8_output_rejected_by_fused_ptx_codegen() {
        // Hand-build a `KernelSpec` whose single output is Utf8. The
        // body is intentionally trivial — we only care that the preflight
        // refuses to emit kernel params for a Utf8 output buffer. The
        // codegen check fires before any `Op` is lowered.
        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "in0".into(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "out0".into(),
                dtype: DataType::Utf8,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
                // Store is unreachable past the Utf8 output preflight; we
                // include it so the spec is internally well-formed.
                Op::Store {
                    src: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
            ],
            predicate: None,
            register_count: 1,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };

        let err = compile(&spec, "bolt_utf8_out_blocker").expect_err(
            "fused PTX codegen must reject a Utf8 output column",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("Utf8"),
            "rejection should mention Utf8 (got {msg})"
        );
    }

    /// The *fused PTX codegen* (`compile`) rejects any `KernelSpec` whose
    /// `inputs` carry a `Utf8` slot — even before any `Op` references it —
    /// with "Utf8 not supported in PTX codegen yet".
    ///
    /// NOTE: GPU `LENGTH` is wired end-to-end via `PhysicalPlan::StringLength`
    /// (`crate::exec::string_length` / `jit::string_kernel::
    /// compile_length_gather_kernel`), which does NOT use this fused `compile`
    /// path. This test therefore pins only the fused-kernel Utf8-input gate,
    /// which remains a real constraint for the fused arithmetic/predicate
    /// codegen.
    #[test]
    fn utf8_input_rejected_by_fused_ptx_codegen() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "s".into(),
                dtype: DataType::Utf8,
            }],
            outputs: vec![ColumnIO {
                name: "len".into(),
                dtype: DataType::Int32,
            }],
            // No ops — the Utf8 preflight rejects before lowering any IR.
            ops: vec![],
            predicate: None,
            register_count: 0,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };

        let err = compile(&spec, "bolt_utf8_in_blocker").expect_err(
            "fused PTX codegen must reject a Utf8 input column",
        );
        let msg = format!("{err}");
        assert!(
            msg.contains("Utf8"),
            "rejection should mention Utf8 (got {msg})"
        );
    }

    /// Current scalar-string-fn lowering contract (repurposed from a stale
    /// "every kind is rejected" test). The GPU string producers have landed,
    /// so `lower()` no longer rejects UPPER/LOWER/LENGTH/SUBSTRING/TRIM. We
    /// pin BOTH halves of the live contract end-to-end (logical plan →
    /// `lower`):
    ///
    ///   * the now-SUPPORTED kinds lower WITHOUT error to the expected
    ///     physical node (UPPER/LOWER → `StringProject`, LENGTH → `StringLength`,
    ///     SUBSTRING/TRIM → host-side `Project`, and — newly — `CONCAT(col, col)`
    ///     over a bare scan → GPU `StringProject` via the N-input two-pass
    ///     producer).
    ///
    /// This keeps the assertion meaningful (not a tautology) while matching the
    /// behavior the code actually has now.
    #[test]
    fn scalar_string_fn_lowering_matches_current_contract() {
        use crate::plan::logical_plan::{
            DataType as PlanDt, Expr, Field, Literal, LogicalPlan, ScalarFnKind, Schema,
        };
        use crate::plan::physical_plan::{lower, PhysicalPlan};

        // Minimal single-Utf8-column fixture over a *bare scan* — the shape the
        // GPU string producers (`try_lower_string_length` /
        // `try_lower_string_project`) accept.
        let schema = Schema::new(vec![Field {
            name: "s".into(),
            dtype: PlanDt::Utf8,
            nullable: false,
        }]);
        let scan = LogicalPlan::Scan {
            table: "txt".into(),
            projection: None,
            schema,
        };
        let s_col = Expr::Column("s".into());

        let project = |expr: Expr| LogicalPlan::Project {
            input: Box::new(scan.clone()),
            exprs: vec![expr],
        };

        // ---- Now-SUPPORTED: UPPER / LOWER lower to GPU `StringProject`. ----
        for kind in [ScalarFnKind::Upper, ScalarFnKind::Lower] {
            let plan = project(Expr::ScalarFn {
                kind,
                args: vec![s_col.clone()],
            });
            let phys = lower(&plan)
                .unwrap_or_else(|e| panic!("{} should lower to GPU now, got Err: {e}", kind.sql_name()));
            assert!(
                matches!(phys, PhysicalPlan::StringProject { .. }),
                "{} should lower to PhysicalPlan::StringProject; got: {phys:?}",
                kind.sql_name()
            );
        }

        // ---- Now-SUPPORTED: LENGTH(<bare col>) lowers to GPU `StringLength`. ----
        {
            let plan = project(Expr::ScalarFn {
                kind: ScalarFnKind::Length,
                args: vec![s_col.clone()],
            });
            let phys = lower(&plan).expect("LENGTH(<col>) should lower to GPU now");
            assert!(
                matches!(phys, PhysicalPlan::StringLength { .. }),
                "LENGTH should lower to PhysicalPlan::StringLength; got: {phys:?}"
            );
        }

        // ---- Now-SUPPORTED (F9): SUBSTRING(col, lit, lit) and single-arg
        // TRIM(col) over a bare Utf8 scan lower to `PhysicalPlan::StringProject`
        // (host-realized two-pass producer), no longer a plain host Project. ----
        {
            let plan = project(Expr::ScalarFn {
                kind: ScalarFnKind::Substring,
                args: vec![
                    s_col.clone(),
                    Expr::Literal(Literal::Int64(1)),
                    Expr::Literal(Literal::Int64(2)),
                ],
            });
            let phys = lower(&plan).expect("SUBSTRING(col, lit, lit) should lower (not an Err)");
            assert!(
                matches!(phys, PhysicalPlan::StringProject { .. }),
                "SUBSTRING should lower to PhysicalPlan::StringProject; got: {phys:?}"
            );
        }
        {
            let plan = project(Expr::ScalarFn {
                kind: ScalarFnKind::TrimBoth,
                args: vec![s_col.clone()],
            });
            let phys = lower(&plan).expect("TRIM(col) should lower (not an Err)");
            assert!(
                matches!(phys, PhysicalPlan::StringProject { .. }),
                "single-arg TRIM should lower to PhysicalPlan::StringProject; got: {phys:?}"
            );
        }

        // ---- Now-SUPPORTED: CONCAT(<col>, <col>) over a bare scan lowers to
        // the GPU `StringProject` via the N-input two-pass producer
        // (`jit::string_kernel::compile_concat_{len,write}_pass`). The executor
        // keeps a host fallback for unsupported arities / layouts. ----
        let concat_plan = project(Expr::ScalarFn {
            kind: ScalarFnKind::Concat,
            args: vec![s_col.clone(), s_col.clone()],
        });
        let phys = lower(&concat_plan)
            .expect("CONCAT(<col>, <col>) should lower to GPU StringProject now");
        assert!(
            matches!(phys, PhysicalPlan::StringProject { .. }),
            "CONCAT should lower to PhysicalPlan::StringProject; got: {phys:?}"
        );
    }
}

#[cfg(test)]
mod decimal128_ir_tests {
    //! PTX shape tests for the dual-register Decimal128 / i128 ops
    //! (v0.7 Sub-task A). These cover the *IR-emission* layer only —
    //! `Codegen::emit_column` / `emit_literal` / `emit_binary` still
    //! reject Decimal128 with a `Plan` error, so the end-to-end SQL
    //! reject tests in `tests/decimal_type_test.rs` continue to pass
    //! unchanged. The tests here build a `KernelSpec` by hand to drive
    //! `compile` directly.
    use super::*;
    use crate::plan::physical_plan::{ColumnIO, KernelSpec, Op, Reg};
    use crate::plan::logical_plan::DataType;
    fn dec(prec: u8, scale: i8) -> DataType {
        DataType::Decimal128(prec, scale)
    }
    /// `Op::LoadColumn128` should emit two `ld.global.nc.u64` loads — one
    /// at byte offset `tid * 16` (low half) and one at `+8` (high half).
    /// Mirrors the read-only-cache hint used by `Op::LoadColumn` so the
    /// input column buffer doesn't pollute the L1 of stores.
    #[test]
    fn load_column_128_emits_two_u64_loads() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "amt".into(),
                dtype: dec(18, 2),
            }],
            outputs: vec![ColumnIO {
                name: "amt_out".into(),
                dtype: dec(18, 2),
            }],
            ops: vec![
                Op::LoadColumn128 {
                    dst_lo: Reg(0),
                    dst_hi: Reg(1),
                    col_idx: 0,
                },
                Op::Store128 {
                    src_lo: Reg(0),
                    src_hi: Reg(1),
                    col_idx: 0,
                },
            ],
            predicate: None,
            register_count: 2,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let ptx = compile(&spec, "bolt_dec128_load").expect("compile");
        // Two read-only-cache 64-bit loads (one per half).
        let n_u64_loads = ptx.matches("ld.global.nc.u64").count();
        assert!(
            n_u64_loads >= 2,
            "expected >=2 ld.global.nc.u64 for Decimal128 row load, got {n_u64_loads}\n{ptx}"
        );
        // `mul.wide.u32 ..., 16;` is the stride-16 offset arithmetic.
        assert!(
            ptx.contains("mul.wide.u32") && ptx.contains(", 16;"),
            "expected `mul.wide.u32 ..., 16;` for tid*16 stride arithmetic\n{ptx}"
        );
        // Hi-half address is lo-address + 8 (single `add.s64 ..., 8;`).
        assert!(
            ptx.contains("add.s64") && ptx.contains(", 8;"),
            "expected `add.s64 ..., 8;` for hi-half address offset\n{ptx}"
        );
    }
    /// `Op::Store128` should emit two `st.global.u64` writes at the same
    /// offsets the load uses (lo + lo+8).
    #[test]
    fn store_128_emits_two_u64_stores() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "amt".into(),
                dtype: dec(18, 2),
            }],
            outputs: vec![ColumnIO {
                name: "amt_out".into(),
                dtype: dec(18, 2),
            }],
            ops: vec![
                Op::LoadColumn128 {
                    dst_lo: Reg(0),
                    dst_hi: Reg(1),
                    col_idx: 0,
                },
                Op::Store128 {
                    src_lo: Reg(0),
                    src_hi: Reg(1),
                    col_idx: 0,
                },
            ],
            predicate: None,
            register_count: 2,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let ptx = compile(&spec, "bolt_dec128_store").expect("compile");
        let n_u64_stores = ptx.matches("st.global.u64").count();
        assert!(
            n_u64_stores >= 2,
            "expected >=2 st.global.u64 for Decimal128 row store, got {n_u64_stores}\n{ptx}"
        );
    }
    /// `Op::Const128` should emit two `mov.u64` instructions of the hex
    /// bit-patterns. The values are chosen so the lo / hi hex strings are
    /// distinguishable on inspection.
    #[test]
    fn const_128_emits_two_movs_of_hex_constants() {
        let lo: u64 = 0x0123_4567_89AB_CDEF;
        let hi: u64 = 0xFEDC_BA98_7654_3210;
        let spec = KernelSpec {
            inputs: vec![],
            outputs: vec![ColumnIO {
                name: "k".into(),
                dtype: dec(38, 0),
            }],
            ops: vec![
                Op::Const128 {
                    dst_lo: Reg(0),
                    dst_hi: Reg(1),
                    lo,
                    hi,
                },
                Op::Store128 {
                    src_lo: Reg(0),
                    src_hi: Reg(1),
                    col_idx: 0,
                },
            ],
            predicate: None,
            register_count: 2,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let ptx = compile(&spec, "bolt_dec128_const").expect("compile");
        let expected_lo = format!("0x{:016X}", lo);
        let expected_hi = format!("0x{:016X}", hi);
        assert!(
            ptx.contains(&expected_lo),
            "expected lo half hex constant {} in PTX\n{ptx}",
            expected_lo
        );
        assert!(
            ptx.contains(&expected_hi),
            "expected hi half hex constant {} in PTX\n{ptx}",
            expected_hi
        );
        // Both `mov.u64`s emitted.
        let n_mov_u64 = ptx.matches("mov.u64").count();
        assert!(
            n_mov_u64 >= 2,
            "expected >=2 mov.u64 for Const128, got {n_mov_u64}\n{ptx}"
        );
    }
    /// `Op::Add128` lowers to `add.cc.u64` (low half, sets carry) then
    /// `addc.u64` (high half, adds carry-in). Order matters — `addc` must
    /// follow `add.cc` so PTX picks up the right `%CC` value.
    #[test]
    fn add_128_emits_add_cc_then_addc() {
        let spec = KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "a".into(),
                    dtype: dec(18, 2),
                },
                ColumnIO {
                    name: "b".into(),
                    dtype: dec(18, 2),
                },
            ],
            outputs: vec![ColumnIO {
                name: "sum".into(),
                dtype: dec(18, 2),
            }],
            ops: vec![
                Op::LoadColumn128 {
                    dst_lo: Reg(0),
                    dst_hi: Reg(1),
                    col_idx: 0,
                },
                Op::LoadColumn128 {
                    dst_lo: Reg(2),
                    dst_hi: Reg(3),
                    col_idx: 1,
                },
                Op::Add128 {
                    dst_lo: Reg(4),
                    dst_hi: Reg(5),
                    a_lo: Reg(0),
                    a_hi: Reg(1),
                    b_lo: Reg(2),
                    b_hi: Reg(3),
                },
                Op::Store128 {
                    src_lo: Reg(4),
                    src_hi: Reg(5),
                    col_idx: 0,
                },
            ],
            predicate: None,
            register_count: 6,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let ptx = compile(&spec, "bolt_dec128_add").expect("compile");
        assert!(
            ptx.contains("add.cc.u64"),
            "expected add.cc.u64 (low half + sets carry)\n{ptx}"
        );
        assert!(
            ptx.contains("addc.u64"),
            "expected addc.u64 (high half + carry-in)\n{ptx}"
        );
        // Order: add.cc.u64 must come before addc.u64.
        let pos_cc = ptx.find("add.cc.u64").expect("add.cc.u64 present");
        let pos_c = ptx.find("addc.u64").expect("addc.u64 present");
        assert!(
            pos_cc < pos_c,
            "add.cc.u64 must precede addc.u64 so the carry flag flows correctly\n{ptx}"
        );
    }
    /// `Op::Sub128` lowers to `sub.cc.u64` (low half, sets borrow) then
    /// `subc.u64` (high half, subtracts borrow-in).
    #[test]
    fn sub_128_emits_sub_cc_then_subc() {
        let spec = KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "a".into(),
                    dtype: dec(18, 2),
                },
                ColumnIO {
                    name: "b".into(),
                    dtype: dec(18, 2),
                },
            ],
            outputs: vec![ColumnIO {
                name: "diff".into(),
                dtype: dec(18, 2),
            }],
            ops: vec![
                Op::LoadColumn128 {
                    dst_lo: Reg(0),
                    dst_hi: Reg(1),
                    col_idx: 0,
                },
                Op::LoadColumn128 {
                    dst_lo: Reg(2),
                    dst_hi: Reg(3),
                    col_idx: 1,
                },
                Op::Sub128 {
                    dst_lo: Reg(4),
                    dst_hi: Reg(5),
                    a_lo: Reg(0),
                    a_hi: Reg(1),
                    b_lo: Reg(2),
                    b_hi: Reg(3),
                },
                Op::Store128 {
                    src_lo: Reg(4),
                    src_hi: Reg(5),
                    col_idx: 0,
                },
            ],
            predicate: None,
            register_count: 6,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let ptx = compile(&spec, "bolt_dec128_sub").expect("compile");
        assert!(
            ptx.contains("sub.cc.u64"),
            "expected sub.cc.u64 (low half + sets borrow)\n{ptx}"
        );
        assert!(
            ptx.contains("subc.u64"),
            "expected subc.u64 (high half + borrow-in)\n{ptx}"
        );
        let pos_cc = ptx.find("sub.cc.u64").expect("sub.cc.u64 present");
        let pos_c = ptx.find("subc.u64").expect("subc.u64 present");
        assert!(
            pos_cc < pos_c,
            "sub.cc.u64 must precede subc.u64 so the borrow flag flows correctly\n{ptx}"
        );
    }
    /// `Op::Mul128` lowers to a schoolbook cross-multiply: 1 `mul.lo.u64`
    /// for the low half, then 1 `mul.hi.u64` + 2 more `mul.lo.u64`
    /// instructions summed for the high half. No Karatsuba.
    #[test]
    fn mul_128_emits_cross_multiply_pattern() {
        let spec = KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "a".into(),
                    dtype: dec(18, 2),
                },
                ColumnIO {
                    name: "b".into(),
                    dtype: dec(18, 2),
                },
            ],
            outputs: vec![ColumnIO {
                name: "prod".into(),
                dtype: dec(18, 2),
            }],
            ops: vec![
                Op::LoadColumn128 {
                    dst_lo: Reg(0),
                    dst_hi: Reg(1),
                    col_idx: 0,
                },
                Op::LoadColumn128 {
                    dst_lo: Reg(2),
                    dst_hi: Reg(3),
                    col_idx: 1,
                },
                Op::Mul128 {
                    dst_lo: Reg(4),
                    dst_hi: Reg(5),
                    a_lo: Reg(0),
                    a_hi: Reg(1),
                    b_lo: Reg(2),
                    b_hi: Reg(3),
                },
                Op::Store128 {
                    src_lo: Reg(4),
                    src_hi: Reg(5),
                    col_idx: 0,
                },
            ],
            predicate: None,
            register_count: 6,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let ptx = compile(&spec, "bolt_dec128_mul").expect("compile");
        // The cross-multiply uses exactly one `mul.hi.u64` (for
        // a_lo*b_lo's high half) and three `mul.lo.u64` (a_lo*b_lo low,
        // a_lo*b_hi low, a_hi*b_lo low).
        assert!(
            ptx.contains("mul.hi.u64"),
            "expected `mul.hi.u64` for high half of a_lo*b_lo\n{ptx}"
        );
        let n_mul_lo = ptx.matches("mul.lo.u64").count();
        assert!(
            n_mul_lo >= 3,
            "expected >=3 mul.lo.u64 partial products, got {n_mul_lo}\n{ptx}"
        );
        // The high half accumulates via two `add.u64` (plain, not .cc) —
        // overflow above 128 bits is intentionally discarded.
        let n_add_u64 = ptx.matches("add.u64").count();
        assert!(
            n_add_u64 >= 2,
            "expected >=2 add.u64 for high-half partial-product sum, got {n_add_u64}\n{ptx}"
        );
    }
    /// `RegAlloc::assign_pair` must hand out two distinct `rl` registers
    /// and record both in the mapping table. Regression guard against an
    /// accidental shared-name mistake.
    #[test]
    fn assign_pair_gives_two_distinct_rl_registers() {
        let mut alloc = RegAlloc::new();
        let (lo, hi) = alloc
            .assign_pair(Reg(0), Reg(1))
            .expect("assign_pair must succeed");
        assert!(lo.starts_with("%rl"), "lo should be in the rl class, got {lo}");
        assert!(hi.starts_with("%rl"), "hi should be in the rl class, got {hi}");
        assert_ne!(lo, hi, "lo and hi must be distinct physical registers");
        // Both halves resolvable via `get`.
        assert_eq!(alloc.get(Reg(0)).unwrap(), lo);
        assert_eq!(alloc.get(Reg(1)).unwrap(), hi);
        // Class counter advanced by 2.
        assert_eq!(alloc.count("rl"), 2);
    }

    // -------- Op::Cmp128 PTX shape tests (v0.7 follow-up to Sub-task B) ----
    //
    // Each test below builds a 6-op kernel:
    //
    //   ld lhs.lo/hi  -> r0/r1
    //   ld rhs.lo/hi  -> r2/r3
    //   cmp128(op, r0,r1, r2,r3) -> r4  (Bool 0/1)
    //   store r4 -> output Bool column
    //
    // and inspects the emitted PTX for the expected `setp` / `and.pred` /
    // `or.pred` / `selp.s32` mnemonics matching the per-op wire shape
    // documented on `Op::Cmp128`. The kernels do NOT execute on a GPU
    // here — these are textual PTX-shape regression tests, mirroring the
    // pattern used by every other op in this module.

    /// Build a 6-op `KernelSpec` that compares two Decimal128 columns
    /// with `op` and stores the Bool result. Shared helper across the
    /// per-operator Cmp128 tests below.
    fn cmp128_spec(op: BinaryOp) -> KernelSpec {
        KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "a".into(),
                    dtype: dec(18, 2),
                },
                ColumnIO {
                    name: "b".into(),
                    dtype: dec(18, 2),
                },
            ],
            outputs: vec![ColumnIO {
                name: "r".into(),
                dtype: DataType::Bool,
            }],
            ops: vec![
                Op::LoadColumn128 {
                    dst_lo: Reg(0),
                    dst_hi: Reg(1),
                    col_idx: 0,
                },
                Op::LoadColumn128 {
                    dst_lo: Reg(2),
                    dst_hi: Reg(3),
                    col_idx: 1,
                },
                Op::Cmp128 {
                    dst: Reg(4),
                    op,
                    a_lo: Reg(0),
                    a_hi: Reg(1),
                    b_lo: Reg(2),
                    b_hi: Reg(3),
                },
                Op::Store {
                    src: Reg(4),
                    col_idx: 0,
                    dtype: DataType::Bool,
                },
            ],
            predicate: None,
            register_count: 5,
            input_has_validity: vec![],
            output_has_validity: vec![],
        }
    }

    /// `Op::Cmp128 { op: Eq }` lowers to:
    ///
    ///   `setp.eq.u64 p_lo, a_lo, b_lo` + `setp.eq.s64 p_hi, a_hi, b_hi`
    ///   + `and.pred p, p_lo, p_hi` + `selp.s32 dst, 1, 0, p`.
    #[test]
    fn cmp_128_eq_emits_setp_eq_with_and_pred() {
        let spec = cmp128_spec(BinaryOp::Eq);
        let ptx = compile(&spec, "bolt_dec128_cmp_eq").expect("compile");
        assert!(
            ptx.contains("setp.eq.u64"),
            "expected setp.eq.u64 for low-half equality\n{ptx}"
        );
        assert!(
            ptx.contains("setp.eq.s64"),
            "expected setp.eq.s64 for high-half equality (signed)\n{ptx}"
        );
        assert!(
            ptx.contains("and.pred"),
            "expected and.pred to combine low + high equality predicates\n{ptx}"
        );
        assert!(
            ptx.contains("selp.s32"),
            "expected selp.s32 to materialise the 0/1 Bool result\n{ptx}"
        );
        // No `or.pred` for Eq — that's the NotEq shape.
        assert!(
            !ptx.contains("or.pred"),
            "Eq must not use or.pred (that's NotEq's combiner)\n{ptx}"
        );
    }

    /// `Op::Cmp128 { op: NotEq }` lowers to setp.ne on both halves
    /// combined with `or.pred`.
    #[test]
    fn cmp_128_ne_emits_setp_ne_with_or_pred() {
        let spec = cmp128_spec(BinaryOp::NotEq);
        let ptx = compile(&spec, "bolt_dec128_cmp_ne").expect("compile");
        assert!(
            ptx.contains("setp.ne.u64"),
            "expected setp.ne.u64 for low-half inequality\n{ptx}"
        );
        assert!(
            ptx.contains("setp.ne.s64"),
            "expected setp.ne.s64 for high-half inequality (signed)\n{ptx}"
        );
        assert!(
            ptx.contains("or.pred"),
            "expected or.pred to combine low + high inequality predicates\n{ptx}"
        );
        assert!(
            ptx.contains("selp.s32"),
            "expected selp.s32 to materialise the 0/1 Bool result\n{ptx}"
        );
        // No `and.pred` for NotEq.
        assert!(
            !ptx.contains("and.pred"),
            "NotEq must not use and.pred (that's Eq's combiner)\n{ptx}"
        );
    }

    /// `Op::Cmp128 { op: Lt }` lowers to:
    ///
    ///   `setp.lt.s64` (hi_lt) + `setp.eq.s64` (hi_eq) + `setp.lt.u64` (lo_lt)
    ///   + `and.pred p_eq_lt, hi_eq, lo_lt` + `or.pred p, hi_lt, p_eq_lt`
    ///   + `selp.s32 dst, 1, 0, p`.
    ///
    /// Signed-high / unsigned-low because the i128's sign lives in the top
    /// bit of the high half; once the high halves are equal the low half's
    /// raw u64 ordering IS the within-equal-high-half ordering.
    #[test]
    fn cmp_128_lt_emits_split_signed_high_unsigned_low_pattern() {
        let spec = cmp128_spec(BinaryOp::Lt);
        let ptx = compile(&spec, "bolt_dec128_cmp_lt").expect("compile");
        assert!(
            ptx.contains("setp.lt.s64"),
            "expected setp.lt.s64 for high-half signed compare\n{ptx}"
        );
        assert!(
            ptx.contains("setp.eq.s64"),
            "expected setp.eq.s64 for high-half tie-break\n{ptx}"
        );
        assert!(
            ptx.contains("setp.lt.u64"),
            "expected setp.lt.u64 for low-half unsigned compare\n{ptx}"
        );
        assert!(
            ptx.contains("and.pred"),
            "expected and.pred to combine (hi_eq AND lo_lt)\n{ptx}"
        );
        assert!(
            ptx.contains("or.pred"),
            "expected or.pred to combine (hi_lt OR (hi_eq AND lo_lt))\n{ptx}"
        );
        assert!(
            ptx.contains("selp.s32"),
            "expected selp.s32 to materialise the 0/1 Bool result\n{ptx}"
        );
    }

    /// `Op::Cmp128 { op: Gt }` mirrors `Lt` with `.gt` on both halves.
    #[test]
    fn cmp_128_gt_emits_split_signed_high_unsigned_low_pattern() {
        let spec = cmp128_spec(BinaryOp::Gt);
        let ptx = compile(&spec, "bolt_dec128_cmp_gt").expect("compile");
        assert!(
            ptx.contains("setp.gt.s64"),
            "expected setp.gt.s64 for high-half signed compare\n{ptx}"
        );
        assert!(
            ptx.contains("setp.eq.s64"),
            "expected setp.eq.s64 for high-half tie-break\n{ptx}"
        );
        assert!(
            ptx.contains("setp.gt.u64"),
            "expected setp.gt.u64 for low-half unsigned compare\n{ptx}"
        );
        assert!(
            ptx.contains("and.pred") && ptx.contains("or.pred"),
            "expected and.pred + or.pred combining the three predicates\n{ptx}"
        );
    }

    /// `Op::Cmp128 { op: LtEq }` — high `lt` plus low `le` for the
    /// equal-high-half tie path.
    #[test]
    fn cmp_128_le_emits_setp_lt_high_setp_le_low() {
        let spec = cmp128_spec(BinaryOp::LtEq);
        let ptx = compile(&spec, "bolt_dec128_cmp_le").expect("compile");
        assert!(
            ptx.contains("setp.lt.s64"),
            "expected setp.lt.s64 for high-half signed compare\n{ptx}"
        );
        assert!(
            ptx.contains("setp.eq.s64"),
            "expected setp.eq.s64 for high-half tie-break\n{ptx}"
        );
        assert!(
            ptx.contains("setp.le.u64"),
            "expected setp.le.u64 for low-half unsigned <= so equal-low fires\n{ptx}"
        );
    }

    /// `Op::Cmp128 { op: GtEq }` — high `gt` plus low `ge` for the
    /// equal-high-half tie path.
    #[test]
    fn cmp_128_ge_emits_setp_gt_high_setp_ge_low() {
        let spec = cmp128_spec(BinaryOp::GtEq);
        let ptx = compile(&spec, "bolt_dec128_cmp_ge").expect("compile");
        assert!(
            ptx.contains("setp.gt.s64"),
            "expected setp.gt.s64 for high-half signed compare\n{ptx}"
        );
        assert!(
            ptx.contains("setp.eq.s64"),
            "expected setp.eq.s64 for high-half tie-break\n{ptx}"
        );
        assert!(
            ptx.contains("setp.ge.u64"),
            "expected setp.ge.u64 for low-half unsigned >= so equal-low fires\n{ptx}"
        );
    }

    /// F5: `Op::Select128` (the i128 CASE selector) lowers to a `setp.ne.s32`
    /// predicate plus TWO `selp.b64` (one per half) gated on it.
    #[test]
    fn select_128_emits_two_selp_b64() {
        // Load a Bool flag (col 0) and a decimal (col 1); Select128 picks
        // between the loaded decimal (THEN) and a constant (ELSE).
        let spec = KernelSpec {
            inputs: vec![
                ColumnIO { name: "flag".into(), dtype: DataType::Bool },
                ColumnIO { name: "d".into(), dtype: dec(18, 2) },
            ],
            outputs: vec![ColumnIO { name: "out".into(), dtype: dec(18, 2) }],
            ops: vec![
                Op::LoadColumn { dst: Reg(0), col_idx: 0, dtype: DataType::Bool },
                Op::LoadColumn128 { dst_lo: Reg(1), dst_hi: Reg(2), col_idx: 1 },
                Op::Const128 { dst_lo: Reg(3), dst_hi: Reg(4), lo: 0, hi: 0 },
                Op::Select128 {
                    dst_lo: Reg(5),
                    dst_hi: Reg(6),
                    cond: Reg(0),
                    then_lo: Reg(1),
                    then_hi: Reg(2),
                    else_lo: Reg(3),
                    else_hi: Reg(4),
                },
                Op::Store128 { src_lo: Reg(5), src_hi: Reg(6), col_idx: 0 },
            ],
            predicate: None,
            register_count: 7,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let ptx = compile(&spec, "bolt_dec128_select").expect("compile");
        assert!(ptx.contains("setp.ne.s32"), "expected predicate from cond\n{ptx}");
        assert!(
            ptx.matches("selp.b64").count() >= 2,
            "expected >=2 selp.b64 (one per i128 half)\n{ptx}"
        );
    }

    /// F5: `Op::F64ToI128` (Float -> Decimal128 conversion) decomposes the
    /// f64 into two unsigned 64-bit limbs and reassembles them, rounding half
    /// away from zero. Shape check: `abs.f64`, a `cvt.rzi.f64.f64` truncation,
    /// and two `cvt.rzi.u64.f64` limb extractions.
    #[test]
    fn f64_to_i128_emits_limb_decomposition() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO { name: "f".into(), dtype: DataType::Float64 }],
            outputs: vec![ColumnIO { name: "out".into(), dtype: dec(20, 0) }],
            ops: vec![
                Op::LoadColumn { dst: Reg(0), col_idx: 0, dtype: DataType::Float64 },
                Op::F64ToI128 { dst_lo: Reg(1), dst_hi: Reg(2), src: Reg(0) },
                Op::Store128 { src_lo: Reg(1), src_hi: Reg(2), col_idx: 0 },
            ],
            predicate: None,
            register_count: 3,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let ptx = compile(&spec, "bolt_f64_to_i128").expect("compile");
        assert!(ptx.contains("abs.f64"), "expected abs.f64 for |x|\n{ptx}");
        assert!(
            ptx.contains("cvt.rzi.f64.f64"),
            "expected cvt.rzi.f64.f64 for the round-half-away truncation\n{ptx}"
        );
        assert!(
            ptx.matches("cvt.rzi.u64.f64").count() >= 2,
            "expected >=2 cvt.rzi.u64.f64 limb extractions\n{ptx}"
        );
        // Round-half-away adds 0.5 (0d3FE0000000000000) to the magnitude.
        // The value-level semantics this shape encodes are exercised against
        // a pure host mirror in `f64_to_i128_saturating_matches_emitter`
        // (see `f64_to_i128_saturating`): the `0.5` constant here is the same
        // round-half-away addend that helper applies via `x.abs() + 0.5`.
        assert!(
            ptx.contains("0d3FE0000000000000"),
            "expected the 0.5 round constant\n{ptx}"
        );
        // The i128-bound saturation gate: compare |x| against 2^127 and emit
        // the clamp limbs (i128::MAX hi-limb 0x7FFF... / i128::MIN hi-limb
        // 0x8000...). These guard against the per-limb-only overflow bug.
        assert!(
            ptx.contains("0d47E0000000000000"),
            "expected the 2^127 saturation threshold constant\n{ptx}"
        );
        assert!(
            ptx.to_uppercase().contains("0X7FFFFFFFFFFFFFFF")
                && ptx.to_uppercase().contains("0X8000000000000000"),
            "expected i128::MAX / i128::MIN saturation limbs\n{ptx}"
        );
    }

    /// GPU-required smoke test: confirm the f64→i128 kernel PTX (including the
    /// new saturation gate) is accepted by ptxas / the CUDA driver. Skipped by
    /// default; run on a CUDA host with `BOLT_BENCH_GPU=1 <lib_exe> --ignored`.
    #[test]
    #[ignore = "gpu:f64i128 — requires a CUDA driver + BOLT_BENCH_GPU=1"]
    fn f64_to_i128_ptx_loads_into_cuda_driver() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO { name: "f".into(), dtype: DataType::Float64 }],
            outputs: vec![ColumnIO { name: "out".into(), dtype: dec(20, 0) }],
            ops: vec![
                Op::LoadColumn { dst: Reg(0), col_idx: 0, dtype: DataType::Float64 },
                Op::F64ToI128 { dst_lo: Reg(1), dst_hi: Reg(2), src: Reg(0) },
                Op::Store128 { src_lo: Reg(1), src_hi: Reg(2), col_idx: 0 },
            ],
            predicate: None,
            register_count: 3,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let ptx = compile(&spec, "bolt_f64_to_i128").expect("kernel compiles");
        let module = crate::jit::CudaModule::from_ptx(&ptx)
            .expect("f64→i128 PTX (with saturation gate) should load via cuModuleLoadDataEx");
        let _fn = module
            .function("bolt_f64_to_i128")
            .expect("kernel entry point should be reachable");
    }

    /// Value-level coverage for the f64→i128 conversion, exercised through the
    /// host reference mirror [`f64_to_i128_saturating`]. The PTX emitter
    /// `emit_f64_to_i128` is not host-runnable, so this asserts the mirror's
    /// arithmetic against the emitter's *actual* documented sequence (round
    /// half away from zero, NaN→0, non-trapping, i128-bound saturation).
    ///
    /// SATURATION CONTRACT (true i128 bounds): the emitter compares the true
    /// magnitude against `2^127` (`setp.ge.f64`) and clamps to `i128::MAX` /
    /// `i128::MIN` (including `±inf`) before the limb-decomposition path. NaN
    /// compares false and falls through to the normal path, clamping to 0.
    #[test]
    fn f64_to_i128_saturating_matches_emitter() {
        use super::f64_to_i128_saturating as f2i;

        // Threshold sanity: confirm the boundary values used below straddle
        // 2^127 ≈ 1.70e38 as claimed.
        const TWO127: f64 = 170_141_183_460_469_231_731_687_303_715_884_105_728.0;
        assert!(1.2e38 < TWO127); // in-range
        assert!(2.0e38 > TWO127); // saturates

        // Exact small integers.
        assert_eq!(f2i(0.0), 0);
        assert_eq!(f2i(-0.0), 0); // -0.0 is not < 0.0 → positive branch → 0.
        assert_eq!(f2i(1.0), 1);
        assert_eq!(f2i(-1.0), -1);

        // Round HALF AWAY FROM ZERO: 2.5 → 3, -2.5 → -3 (|x|+0.5 then trunc).
        assert_eq!(f2i(2.5), 3);
        assert_eq!(f2i(-2.5), -3);
        // 0.5 → 1, 1.5 → 2 (and the negatives) confirm the rounding direction.
        assert_eq!(f2i(0.5), 1);
        assert_eq!(f2i(-0.5), -1);
        assert_eq!(f2i(1.5), 2);
        assert_eq!(f2i(-1.5), -2);
        // Just-below-half rounds toward zero.
        assert_eq!(f2i(2.4), 2);
        assert_eq!(f2i(-2.4), -2);

        // Large in-range value that crosses the 2^64 limb boundary (1e18 is
        // exactly representable in f64).
        assert_eq!(f2i(1e18), 1_000_000_000_000_000_000);
        assert_eq!(f2i(-1e18), -1_000_000_000_000_000_000);

        // A large in-range magnitude below 2^127 (1.2e38 < 2^127 ≈ 1.7e38).
        // f64 cannot represent it exactly, so the result carries the nearest
        // double's value — this is the documented decimal↔float precision
        // loss, not a bug. Computed by replaying the emitter's f64 sequence.
        assert_eq!(f2i(1.2e38), 120_000_000_000_000_008_632_251_347_034_389_348_352);
        assert_eq!(f2i(-1.2e38), -120_000_000_000_000_008_632_251_347_034_389_348_352);

        // Magnitude at/above 2^127 saturates to the i128 bounds (NOT a wrapping
        // limb pattern). 2e38 > 2^127 → i128::MAX / i128::MIN.
        assert_eq!(f2i(2.0e38), i128::MAX);
        assert_eq!(f2i(-2.0e38), i128::MIN);

        // Magnitude ≥ 2^128 (3.3e38 < 2^128 ≈ 3.4e38, but well above 2^127):
        // still saturates, non-trapping, deterministic.
        assert_eq!(f2i(3.3e38), i128::MAX);
        assert_eq!(f2i(-3.3e38), i128::MIN);

        // ±inf saturate to the i128 bounds (inf >= 2^127, -inf <= -(2^127)).
        assert_eq!(f2i(f64::INFINITY), i128::MAX);
        assert_eq!(f2i(f64::NEG_INFINITY), i128::MIN);

        // NaN → 0 (non-trapping): NaN compares false at the saturation gate,
        // flows through trunc as NaN, and every limb extraction clamps to 0.
        assert_eq!(f2i(f64::NAN), 0);
    }

    /// F5: `Op::I128ToF64` (Decimal128 -> Float conversion) computes
    /// `hi*2^64 + lo` via `cvt.rn.f64.s64` (signed hi), `cvt.rn.f64.u64`
    /// (UNSIGNED lo), and an `fma.rn.f64` against the 2^64 constant.
    #[test]
    fn i128_to_f64_emits_hi_times_2pow64_plus_lo() {
        let spec = KernelSpec {
            inputs: vec![ColumnIO { name: "d".into(), dtype: dec(20, 0) }],
            outputs: vec![ColumnIO { name: "out".into(), dtype: DataType::Float64 }],
            ops: vec![
                Op::LoadColumn128 { dst_lo: Reg(0), dst_hi: Reg(1), col_idx: 0 },
                Op::I128ToF64 { dst: Reg(2), src_lo: Reg(0), src_hi: Reg(1) },
                Op::Store { src: Reg(2), col_idx: 0, dtype: DataType::Float64 },
            ],
            predicate: None,
            register_count: 3,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let ptx = compile(&spec, "bolt_i128_to_f64").expect("compile");
        assert!(ptx.contains("cvt.rn.f64.s64"), "expected signed hi conversion\n{ptx}");
        assert!(ptx.contains("cvt.rn.f64.u64"), "expected UNSIGNED lo conversion\n{ptx}");
        assert!(ptx.contains("fma.rn.f64"), "expected fma for hi*2^64+lo\n{ptx}");
        assert!(
            ptx.contains("0d43F0000000000000"),
            "expected the 2^64 constant\n{ptx}"
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

/// Backward DFS shared by every `Store*` arm in `output_input_dependencies`.
/// Seeded with the sink register IDs of one store, walks back through
/// `reg_to_op` collecting every `LoadColumn` / `LoadColumn128` column
/// ordinal that the sinks transitively depend on.
fn walk_store_deps(
    reg_to_op: &HashMap<u32, &crate::plan::physical_plan::Op>,
    sinks: &[u32],
) -> std::collections::HashSet<usize> {
    use crate::plan::physical_plan::Op;
    let mut found: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut visited: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut stack: Vec<u32> = sinks.to_vec();
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
            Op::Not { src, .. } => {
                // Logical NOT forwards its single Bool operand's validity:
                // `NOT x` is NULL iff `x` is NULL. Walk back through `src`.
                stack.push(src.id());
            }
            Op::Binary { lhs, rhs, .. } => {
                stack.push(lhs.id());
                stack.push(rhs.id());
            }
            Op::Select {
                cond,
                then_val,
                else_val,
                ..
            } => {
                // CASE's value-producing path: a downstream output's
                // value depends on every input feeding any of the three
                // operands. We don't try to model per-branch conditional
                // liveness here — the AND-fold caller is the conservative
                // validity tree, not a value dataflow optimiser.
                stack.push(cond.id());
                stack.push(then_val.id());
                stack.push(else_val.id());
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
            Op::Store { .. } | Op::Store128 { .. } => {
                // Stores don't produce a Reg, so reg_to_op can't
                // return one. Unreachable in practice.
            }
            // 128-bit producers: a `LoadColumn128` is a leaf that
            // contributes its input col ordinal exactly like
            // `LoadColumn`. A `Const128` is a leaf. Add128 / Sub128 /
            // Mul128 each read four operand halves; push every
            // operand half so the walker reaches the underlying
            // LoadColumn128s through either dst half.
            Op::LoadColumn128 { col_idx, .. } => {
                found.insert(*col_idx);
            }
            Op::Const128 { .. } => { /* leaf */ }
            Op::Add128 {
                a_lo,
                a_hi,
                b_lo,
                b_hi,
                ..
            }
            | Op::Sub128 {
                a_lo,
                a_hi,
                b_lo,
                b_hi,
                ..
            }
            | Op::Mul128 {
                a_lo,
                a_hi,
                b_lo,
                b_hi,
                ..
            }
            | Op::Cmp128 {
                a_lo,
                a_hi,
                b_lo,
                b_hi,
                ..
            } => {
                // Cmp128 produces a single Bool dst (not a 128-bit pair),
                // but the operand-half structure is the same as the other
                // 128-bit ops: four halves to walk back through. The Bool
                // result is whatever a downstream Store / Cmp / etc. consumes
                // via `dst`.
                stack.push(a_lo.id());
                stack.push(a_hi.id());
                stack.push(b_lo.id());
                stack.push(b_hi.id());
            }
            // F5 Decimal128 ops. WidenToI128 reads one integer source;
            // NarrowI128ToInt reads an i128 (lo, hi) pair; Div128 reads four
            // operand halves (like Add128); Select128 reads the cond Bool plus
            // both branch (lo, hi) pairs. Push every operand so the validity
            // walk reaches the underlying LoadColumn / LoadColumn128 leaves.
            Op::WidenToI128 { src, .. } => {
                stack.push(src.id());
            }
            // F5 Float<->Decimal: F64ToI128 reads one f64 source; I128ToF64
            // reads an i128 (lo, hi) pair.
            Op::F64ToI128 { src, .. } => {
                stack.push(src.id());
            }
            Op::NarrowI128ToInt { src_lo, src_hi, .. }
            | Op::I128ToF64 { src_lo, src_hi, .. } => {
                stack.push(src_lo.id());
                stack.push(src_hi.id());
            }
            Op::Div128 {
                a_lo,
                a_hi,
                b_lo,
                b_hi,
                ..
            } => {
                stack.push(a_lo.id());
                stack.push(a_hi.id());
                stack.push(b_lo.id());
                stack.push(b_hi.id());
            }
            Op::Select128 {
                cond,
                then_lo,
                then_hi,
                else_lo,
                else_hi,
                ..
            } => {
                stack.push(cond.id());
                stack.push(then_lo.id());
                stack.push(then_hi.id());
                stack.push(else_lo.id());
                stack.push(else_hi.id());
            }
        }
    }
    found
}

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
    //
    // The 128-bit ops produce *two* SSA destinations (a lo/hi pair); we
    // insert both halves so a `Store128 { src_lo, src_hi, .. }` walking
    // back from either half lands on the same producer op. The DFS below
    // then dispatches per-op-kind, walking through whichever operand
    // halves matter.
    let mut reg_to_op: HashMap<u32, &Op> = HashMap::with_capacity(spec.ops.len());
    for op in &spec.ops {
        match op {
            Op::LoadColumn { dst, .. }
            | Op::Const { dst, .. }
            | Op::Cast { dst, .. }
            | Op::Binary { dst, .. }
            | Op::IsNullCheck { dst, .. }
            | Op::Select { dst, .. }
            | Op::Not { dst, .. } => {
                reg_to_op.insert(dst.id(), op);
            }
            // Cmp128 collapses the (lo, hi, lo, hi) operand quartet to a
            // single Bool dst — register it as a single-register producer
            // exactly like Op::Binary's comparison shape.
            Op::Cmp128 { dst, .. } => {
                reg_to_op.insert(dst.id(), op);
            }
            Op::Store { .. } | Op::Store128 { .. } => { /* no dst */ }
            Op::LoadColumn128 { dst_lo, dst_hi, .. }
            | Op::Const128 { dst_lo, dst_hi, .. }
            | Op::Add128 { dst_lo, dst_hi, .. }
            | Op::Sub128 { dst_lo, dst_hi, .. }
            | Op::Mul128 { dst_lo, dst_hi, .. }
            // F5: WidenToI128 / Div128 / Select128 each produce an i128
            // (lo, hi) destination pair — register both halves so a Store128
            // (or downstream i128 consumer) walking back from either half
            // lands on this producer.
            | Op::WidenToI128 { dst_lo, dst_hi, .. }
            | Op::Div128 { dst_lo, dst_hi, .. }
            // F5 Float<->Decimal: F64ToI128 produces an i128 (lo, hi) pair.
            | Op::F64ToI128 { dst_lo, dst_hi, .. }
            | Op::Select128 { dst_lo, dst_hi, .. } => {
                reg_to_op.insert(dst_lo.id(), op);
                reg_to_op.insert(dst_hi.id(), op);
            }
            // NarrowI128ToInt collapses an i128 pair to a single 64-bit int
            // dst; I128ToF64 collapses it to a single f64 dst — register each
            // as a single-register producer.
            Op::NarrowI128ToInt { dst, .. } | Op::I128ToF64 { dst, .. } => {
                reg_to_op.insert(dst.id(), op);
            }
        }
    }

    // (b) Pre-allocate one Vec per output. `spec.outputs.len()` is the
    // declared output count; in practice every output has a matching
    // Store, but defaulting to empty preserves correctness if the IR
    // is ever incomplete.
    let mut deps: Vec<Vec<usize>> = vec![Vec::new(); spec.outputs.len()];

    for op in &spec.ops {
        // Identify each store and the register(s) it sinks. The 128-bit
        // `Store128` shape produces two sink registers (`src_lo`,
        // `src_hi`); both are seeded into the DFS so we collect every
        // input column reached through either half. Other ops contribute
        // nothing here (they're def-only — the sinks of the dataflow are
        // exclusively `Store` and `Store128`).
        let (col_idx, sinks): (usize, Vec<u32>) = match op {
            Op::Store { src, col_idx, .. } => (*col_idx, vec![src.id()]),
            Op::Store128 {
                src_lo,
                src_hi,
                col_idx,
            } => (*col_idx, vec![src_lo.id(), src_hi.id()]),
            _ => continue,
        };
        if col_idx >= deps.len() {
            // Defensive: a Store referencing an unknown output index
            // is a planner bug — skip rather than panic so codegen
            // can surface the real diagnostic elsewhere.
            continue;
        }
        let found = walk_store_deps(&reg_to_op, &sinks);
        // Merge into the per-output set (sorted + dedup at the end).
        for c in found {
            if !deps[col_idx].contains(&c) {
                deps[col_idx].push(c);
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

#[cfg(test)]
mod cast_emission_tests {
    //! Per-pair coverage for the `Op::Cast` PTX emission table in
    //! [`emit_cast`].
    //!
    //! v0.7 wires `CAST(<numeric> AS <numeric>)` (plus `Bool ↔ Int`)
    //! through the existing `emit_cast` helper, which the codegen has
    //! used since v0.5 for binary-op dtype unification. These tests pin
    //! the emitted `cvt.*` mnemonic for each accepted source/target
    //! pair so future relaxations (e.g. `Bool -> Float` going through a
    //! different code path) regress visibly rather than silently
    //! changing rounding / extension semantics.
    //!
    //! Approach: drive [`compile`] over a hand-crafted three-op spec
    //! `[LoadColumn -> Cast -> Store]` so the output PTX contains
    //! exactly one `cvt.*` instruction for the pair under test, and
    //! assert the substring is present. The full compile pipeline
    //! (rather than a direct `emit_cast` call) keeps the test honest
    //! about the register-class allocator wiring around `emit_cast`.
    use super::*;
    use crate::plan::physical_plan::{ColumnIO, KernelSpec, Op, Reg};

    /// Build a `[LoadColumn -> Cast -> Store]` spec converting input
    /// column 0 (`from`) into output column 0 (`to`) and compile it.
    fn compile_single_cast(from: DataType, to: DataType) -> String {
        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".into(),
                dtype: from,
            }],
            outputs: vec![ColumnIO {
                name: "y".into(),
                dtype: to,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: from,
                },
                Op::Cast {
                    dst: Reg(1),
                    src: Reg(0),
                    from,
                    to,
                },
                Op::Store {
                    src: Reg(1),
                    col_idx: 0,
                    dtype: to,
                },
            ],
            predicate: None,
            register_count: 2,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        compile(&spec, "bolt_cast_test").expect("compile cast spec")
    }

    /// `CAST(Int32 AS Int64)` lowers to `cvt.s64.s32` — sign-extending widen.
    #[test]
    fn cast_int32_to_int64_emits_cvt_s64_s32() {
        let ptx = compile_single_cast(DataType::Int32, DataType::Int64);
        assert!(
            ptx.contains("cvt.s64.s32"),
            "expected cvt.s64.s32 for Int32 -> Int64, got:\n{ptx}"
        );
    }

    /// `CAST(Int64 AS Int32)` lowers to `cvt.s32.s64` — truncating narrow.
    #[test]
    fn cast_int64_to_int32_emits_cvt_s32_s64() {
        let ptx = compile_single_cast(DataType::Int64, DataType::Int32);
        assert!(
            ptx.contains("cvt.s32.s64"),
            "expected cvt.s32.s64 for Int64 -> Int32, got:\n{ptx}"
        );
    }

    /// `CAST(Int32 AS Float32)` lowers to `cvt.rn.f32.s32` — round-to-nearest.
    #[test]
    fn cast_int32_to_float32_emits_cvt_rn_f32_s32() {
        let ptx = compile_single_cast(DataType::Int32, DataType::Float32);
        assert!(
            ptx.contains("cvt.rn.f32.s32"),
            "expected cvt.rn.f32.s32 for Int32 -> Float32, got:\n{ptx}"
        );
    }

    /// `CAST(Int32 AS Float64)` lowers to `cvt.rn.f64.s32` — round-to-nearest.
    #[test]
    fn cast_int32_to_float64_emits_cvt_rn_f64_s32() {
        let ptx = compile_single_cast(DataType::Int32, DataType::Float64);
        assert!(
            ptx.contains("cvt.rn.f64.s32"),
            "expected cvt.rn.f64.s32 for Int32 -> Float64, got:\n{ptx}"
        );
    }

    /// `CAST(Int64 AS Float64)` lowers to `cvt.rn.f64.s64` — round-to-nearest.
    #[test]
    fn cast_int64_to_float64_emits_cvt_rn_f64_s64() {
        let ptx = compile_single_cast(DataType::Int64, DataType::Float64);
        assert!(
            ptx.contains("cvt.rn.f64.s64"),
            "expected cvt.rn.f64.s64 for Int64 -> Float64, got:\n{ptx}"
        );
    }

    /// `CAST(Float32 AS Float64)` lowers to `cvt.f64.f32` — exact widen
    /// (no rounding mode needed; f64 covers f32 losslessly).
    #[test]
    fn cast_float32_to_float64_emits_cvt_f64_f32() {
        let ptx = compile_single_cast(DataType::Float32, DataType::Float64);
        assert!(
            ptx.contains("cvt.f64.f32"),
            "expected cvt.f64.f32 for Float32 -> Float64, got:\n{ptx}"
        );
    }

    /// `CAST(Float64 AS Float32)` lowers to `cvt.rn.f32.f64` — narrowing
    /// requires an explicit rounding mode.
    #[test]
    fn cast_float64_to_float32_emits_cvt_rn_f32_f64() {
        let ptx = compile_single_cast(DataType::Float64, DataType::Float32);
        assert!(
            ptx.contains("cvt.rn.f32.f64"),
            "expected cvt.rn.f32.f64 for Float64 -> Float32, got:\n{ptx}"
        );
    }

    /// `CAST(Float64 AS Int64)` lowers to `cvt.rzi.s64.f64` — round-to-zero
    /// integer (SQL "truncation toward zero" semantics).
    #[test]
    fn cast_float64_to_int64_emits_cvt_rzi_s64_f64() {
        let ptx = compile_single_cast(DataType::Float64, DataType::Int64);
        assert!(
            ptx.contains("cvt.rzi.s64.f64"),
            "expected cvt.rzi.s64.f64 for Float64 -> Int64, got:\n{ptx}"
        );
    }

    /// `CAST(Float32 AS Int32)` lowers to `cvt.rzi.s32.f32` — same
    /// round-toward-zero contract as Float64 -> Int64.
    #[test]
    fn cast_float32_to_int32_emits_cvt_rzi_s32_f32() {
        let ptx = compile_single_cast(DataType::Float32, DataType::Int32);
        assert!(
            ptx.contains("cvt.rzi.s32.f32"),
            "expected cvt.rzi.s32.f32 for Float32 -> Int32, got:\n{ptx}"
        );
    }

    /// `CAST(Float64 AS Int32)` lowers to `cvt.rzi.s32.f64` — round-to-zero
    /// AND narrow in a single instruction. Pinned separately from the
    /// matching-width pair so a future split into rzi + s32-narrow shows
    /// up as a regression.
    #[test]
    fn cast_float64_to_int32_emits_cvt_rzi_s32_f64() {
        let ptx = compile_single_cast(DataType::Float64, DataType::Int32);
        assert!(
            ptx.contains("cvt.rzi.s32.f64"),
            "expected cvt.rzi.s32.f64 for Float64 -> Int32, got:\n{ptx}"
        );
    }
}

#[cfg(test)]
mod temporal_arith_tests {
    //! v0.7: PTX-shape coverage for Date32 / Timestamp arithmetic.
    //!
    //! The supported v0.7 surface is:
    //!   * `Date32 - Date32` → `Int32` (number of days)
    //!   * `Timestamp(unit, tz) - Timestamp(unit, tz)` → `Int64` (ticks
    //!     in the source unit; matching unit + tz enforced upstream)
    //!
    //! Anything else (Add/Mul/Div on temporal operands, mixed-unit
    //! Timestamp subtraction, INTERVAL MONTH/YEAR — when an INTERVAL
    //! expr eventually exists) must surface a clear rejection.
    use super::*;
    use crate::plan::logical_plan::{BinaryOp, Literal, TimeUnit};
    use crate::plan::physical_plan::{ColumnIO, KernelSpec, Op, Reg};

    /// Hand-build a kernel: `out0 = in0 - in1` with both inputs Date32
    /// and the output typed Int32 — the IR shape the physical-plan
    /// lowerer produces for `SELECT a - b FROM t` when `a, b` are
    /// Date32 columns.
    fn date_minus_date_spec() -> KernelSpec {
        let ops = vec![
            Op::LoadColumn {
                dst: Reg(0),
                col_idx: 0,
                dtype: DataType::Date32,
            },
            Op::LoadColumn {
                dst: Reg(1),
                col_idx: 1,
                dtype: DataType::Date32,
            },
            // operand dtype is Date32, result dtype is Int32 — exactly
            // the shape the lowerer emits for `Date32 - Date32`.
            Op::Binary {
                dst: Reg(2),
                op: BinaryOp::Sub,
                lhs: Reg(0),
                rhs: Reg(1),
                dtype: DataType::Date32,
                result_dtype: DataType::Int32,
            },
            Op::Store {
                src: Reg(2),
                col_idx: 0,
                dtype: DataType::Int32,
            },
        ];
        KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "a".into(),
                    dtype: DataType::Date32,
                },
                ColumnIO {
                    name: "b".into(),
                    dtype: DataType::Date32,
                },
            ],
            outputs: vec![ColumnIO {
                name: "diff".into(),
                dtype: DataType::Int32,
            }],
            ops,
            predicate: None,
            register_count: 3,
            input_has_validity: vec![],
            output_has_validity: vec![],
        }
    }

    /// Hand-build a kernel: `out0 = in0 - in1` for two Timestamps with
    /// matching unit + tz, producing Int64.
    fn timestamp_minus_timestamp_spec() -> KernelSpec {
        let ts = DataType::Timestamp(TimeUnit::Microsecond, None);
        let ops = vec![
            Op::LoadColumn {
                dst: Reg(0),
                col_idx: 0,
                dtype: ts,
            },
            Op::LoadColumn {
                dst: Reg(1),
                col_idx: 1,
                dtype: ts,
            },
            Op::Binary {
                dst: Reg(2),
                op: BinaryOp::Sub,
                lhs: Reg(0),
                rhs: Reg(1),
                dtype: ts,
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
                    name: "a".into(),
                    dtype: ts,
                },
                ColumnIO {
                    name: "b".into(),
                    dtype: ts,
                },
            ],
            outputs: vec![ColumnIO {
                name: "diff".into(),
                dtype: DataType::Int64,
            }],
            ops,
            predicate: None,
            register_count: 3,
            input_has_validity: vec![],
            output_has_validity: vec![],
        }
    }

    /// `Date32 - Date32` → `Int32`. The PTX must:
    ///   1. Load each input column as a 32-bit signed integer
    ///      (`ld.global.nc.s32`) — Date32 storage is i32 days.
    ///   2. Emit `sub.s32` (NOT `sub.s64`) for the subtraction —
    ///      both operands are i32 days.
    ///   3. Store the result as `st.global.s32` — output dtype is Int32.
    ///   4. Use `b32` registers (the `r` allocator class) for the
    ///      whole chain.
    #[test]
    fn date32_minus_date32_emits_sub_s32() {
        let spec = date_minus_date_spec();
        let ptx = compile(&spec, "bolt_kernel_date_sub").expect("compile");

        let n_s32_loads = ptx.matches("ld.global.nc.s32").count();
        assert!(
            n_s32_loads >= 2,
            "expected >=2 ld.global.nc.s32 for Date32 inputs, got {n_s32_loads}\n{ptx}"
        );

        assert!(
            ptx.contains("sub.s32"),
            "expected sub.s32 for Date32 - Date32, got:\n{ptx}"
        );
        assert!(
            !ptx.contains("sub.s64"),
            "Date32 - Date32 should NOT lower to s64 arith:\n{ptx}"
        );

        let n_s32_stores = ptx.matches("st.global.s32").count();
        assert!(
            n_s32_stores >= 1,
            "expected >=1 st.global.s32 for Int32 output, got {n_s32_stores}\n{ptx}"
        );
    }

    /// `Timestamp - Timestamp` → `Int64`. The PTX must:
    ///   1. Load each input as a 64-bit signed integer (`ld.global.nc.s64`).
    ///   2. Emit `sub.s64` (NOT `sub.s32`).
    ///   3. Store the result as `st.global.s64`.
    #[test]
    fn timestamp_minus_timestamp_emits_sub_s64() {
        let spec = timestamp_minus_timestamp_spec();
        let ptx = compile(&spec, "bolt_kernel_ts_sub").expect("compile");

        let n_s64_loads = ptx.matches("ld.global.nc.s64").count();
        assert!(
            n_s64_loads >= 2,
            "expected >=2 ld.global.nc.s64 for Timestamp inputs, got {n_s64_loads}\n{ptx}"
        );

        assert!(
            ptx.contains("sub.s64"),
            "expected sub.s64 for Timestamp - Timestamp:\n{ptx}"
        );

        let n_s64_stores = ptx.matches("st.global.s64").count();
        assert!(
            n_s64_stores >= 1,
            "expected >=1 st.global.s64 for Int64 output, got {n_s64_stores}\n{ptx}"
        );
    }

    /// Date32 literals lower to a 32-bit `mov.s32` of the days-since-epoch
    /// value. Specifically, `Literal::Date32(2)` (2 days post-epoch) must
    /// emit the hex bit pattern `0x00000002`.
    #[test]
    fn date32_literal_emits_mov_s32() {
        let ops = vec![
            Op::Const {
                dst: Reg(0),
                lit: Literal::Date32(2),
            },
            Op::Store {
                src: Reg(0),
                col_idx: 0,
                dtype: DataType::Date32,
            },
        ];
        let spec = KernelSpec {
            inputs: vec![],
            outputs: vec![ColumnIO {
                name: "d".into(),
                dtype: DataType::Date32,
            }],
            ops,
            predicate: None,
            register_count: 1,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let ptx = compile(&spec, "bolt_kernel_date_lit").expect("compile");
        assert!(
            ptx.contains("mov.s32") && ptx.contains("0x00000002"),
            "expected mov.s32 with 0x00000002 for Date32(2), got:\n{ptx}"
        );
        assert!(
            ptx.contains("st.global.s32"),
            "expected st.global.s32 for Date32 output, got:\n{ptx}"
        );
    }

    /// Sanity: `Date32 + Date32` is not in scope and must surface a clear
    /// rejection from the codegen — `arith_mnemonic` only knows `Sub`
    /// for Date32. (The physical-plan lowerer would normally have
    /// rejected this earlier with a tighter "only Sub is wired"
    /// message; this guards the codegen layer in case a future planner
    /// regression lets the shape through.)
    #[test]
    fn date32_add_date32_codegen_rejected() {
        let ops = vec![
            Op::LoadColumn {
                dst: Reg(0),
                col_idx: 0,
                dtype: DataType::Date32,
            },
            Op::LoadColumn {
                dst: Reg(1),
                col_idx: 1,
                dtype: DataType::Date32,
            },
            Op::Binary {
                dst: Reg(2),
                op: BinaryOp::Add,
                lhs: Reg(0),
                rhs: Reg(1),
                dtype: DataType::Date32,
                result_dtype: DataType::Date32,
            },
            Op::Store {
                src: Reg(2),
                col_idx: 0,
                dtype: DataType::Date32,
            },
        ];
        let spec = KernelSpec {
            inputs: vec![
                ColumnIO {
                    name: "a".into(),
                    dtype: DataType::Date32,
                },
                ColumnIO {
                    name: "b".into(),
                    dtype: DataType::Date32,
                },
            ],
            outputs: vec![ColumnIO {
                name: "out".into(),
                dtype: DataType::Date32,
            }],
            ops,
            predicate: None,
            register_count: 3,
            input_has_validity: vec![],
            output_has_validity: vec![],
        };
        let err = compile(&spec, "bolt_kernel_date_add").expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("unsupported arithmetic") || msg.contains("Add"),
            "rejection should mention unsupported Add on Date32, got: {msg}"
        );
    }
}

#[cfg(test)]
mod temporal_plan_tests {
    //! v0.7: logical / physical plan type-check coverage for the Date32 /
    //! Timestamp subtraction surface.
    //!
    //! These tests construct the plan directly (no SQL frontend) so they
    //! cover the type-check independent of how the user might phrase the
    //! query. INTERVAL-based arithmetic is intentionally not tested
    //! because the SQL frontend has no INTERVAL expression literal yet
    //! (see `sql_frontend.rs`); the physical-plan helper rejects every
    //! op other than `Sub` on temporal operands, which is exercised here.
    use crate::plan::logical_plan::{
        AggregateExpr, BinaryOp, DataType, Expr, Field, Schema, TimeUnit,
    };

    fn date_schema() -> Schema {
        Schema::new(vec![
            Field::new("a", DataType::Date32, false),
            Field::new("b", DataType::Date32, false),
        ])
    }

    fn ts_schema(unit: TimeUnit, tz: Option<&'static str>) -> Schema {
        let ty = DataType::Timestamp(unit, tz);
        Schema::new(vec![
            Field::new("a", ty, false),
            Field::new("b", ty, false),
        ])
    }

    /// `a - b` over two Date32 columns must type as `Int32` (a count of
    /// days, NOT another Date).
    #[test]
    fn date_minus_date_types_as_int32() {
        let schema = date_schema();
        let e = Expr::Binary {
            op: BinaryOp::Sub,
            left: Box::new(Expr::Column("a".into())),
            right: Box::new(Expr::Column("b".into())),
        };
        let dt = e.dtype(&schema).expect("typecheck");
        assert_eq!(dt, DataType::Int32, "Date32 - Date32 must produce Int32");
    }

    /// `a - b` over two matching Timestamps types as `Int64` (a count of
    /// ticks in the source unit).
    #[test]
    fn timestamp_minus_timestamp_types_as_int64() {
        let schema = ts_schema(TimeUnit::Microsecond, None);
        let e = Expr::Binary {
            op: BinaryOp::Sub,
            left: Box::new(Expr::Column("a".into())),
            right: Box::new(Expr::Column("b".into())),
        };
        let dt = e.dtype(&schema).expect("typecheck");
        assert_eq!(
            dt,
            DataType::Int64,
            "Timestamp - Timestamp must produce Int64"
        );
    }

    /// Mixed Timestamp units must be rejected with a message naming
    /// "TimeUnit" — coercion is out of scope for v0.7.
    #[test]
    fn timestamp_minus_timestamp_mismatched_units_rejected() {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new(
                "b",
                DataType::Timestamp(TimeUnit::Nanosecond, None),
                false,
            ),
        ]);
        let e = Expr::Binary {
            op: BinaryOp::Sub,
            left: Box::new(Expr::Column("a".into())),
            right: Box::new(Expr::Column("b".into())),
        };
        let err = e.dtype(&schema).expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("TimeUnit") || msg.contains("matching"),
            "rejection should mention TimeUnit / matching, got: {msg}"
        );
    }

    /// `Date32 + Date32` is not a meaningful operation (adding two days-
    /// since-epoch values is nonsense). Must be rejected with the
    /// v0.7-tightened message rather than the generic "requires numeric
    /// operands".
    #[test]
    fn date_plus_date_rejected() {
        let schema = date_schema();
        let e = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::Column("a".into())),
            right: Box::new(Expr::Column("b".into())),
        };
        let err = e.dtype(&schema).expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("Date") || msg.contains("Timestamp") || msg.contains("not supported"),
            "rejection should mention Date/Timestamp, got: {msg}"
        );
    }

    /// `AVG(date_col)` is non-standard SQL — must be rejected at the
    /// logical-plane aggregate output-dtype check with a clear message.
    #[test]
    fn avg_over_date32_rejected() {
        let schema = date_schema();
        let agg = AggregateExpr::Avg(Expr::Column("a".into()));
        let err = avg_output(&agg, &schema).expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("AVG") && (msg.contains("Date") || msg.contains("non-standard")),
            "rejection should mention AVG/Date/non-standard, got: {msg}"
        );
    }

    /// AVG over a Timestamp is likewise non-standard and rejected.
    #[test]
    fn avg_over_timestamp_rejected() {
        let schema = ts_schema(TimeUnit::Microsecond, None);
        let agg = AggregateExpr::Avg(Expr::Column("a".into()));
        let err = avg_output(&agg, &schema).expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("AVG") && (msg.contains("Timestamp") || msg.contains("non-standard")),
            "rejection should mention AVG/Timestamp/non-standard, got: {msg}"
        );
    }

    /// Mixed-tz Timestamp subtraction: tz conversion is out of scope, so
    /// `Timestamp(_, Some("UTC")) - Timestamp(_, None)` (or any tz
    /// mismatch) must be rejected with a message naming "zone".
    #[test]
    fn timestamp_minus_timestamp_mismatched_tz_rejected() {
        let schema = Schema::new(vec![
            Field::new(
                "a",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC")),
                false,
            ),
            Field::new(
                "b",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
        ]);
        let e = Expr::Binary {
            op: BinaryOp::Sub,
            left: Box::new(Expr::Column("a".into())),
            right: Box::new(Expr::Column("b".into())),
        };
        let err = e.dtype(&schema).expect_err("should reject");
        let msg = format!("{err}");
        assert!(
            msg.contains("zone") || msg.contains("matching"),
            "rejection should mention time zones / matching, got: {msg}"
        );
    }

    /// Route an AVG aggregate through the public `LogicalPlan::schema()`
    /// surface so the rejection error message reflects exactly what the
    /// planner produces. `AggregateExpr::output_dtype` is crate-private,
    /// hence the indirection.
    fn avg_output(
        agg: &AggregateExpr,
        schema: &Schema,
    ) -> crate::error::BoltResult<DataType> {
        use crate::plan::logical_plan::LogicalPlan;
        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: schema.clone(),
        };
        let plan = LogicalPlan::Aggregate {
            input: Box::new(scan),
            group_by: vec![],
            aggregates: vec![agg.clone()],
        };
        plan.schema().map(|s| s.fields[0].dtype)
    }
}
