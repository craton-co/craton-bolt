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
//!
//! ## dedup (ptx_common) — audit outcome
//!
//! A dedup pass evaluated whether the emission scaffolding shared with
//! `ptx_gen.rs` (the validity-param wiring, [`write_signature`],
//! [`write_reg_decls`], [`write_err`]) could be hoisted into a shared
//! `ptx_common` module. Conclusion: **none of these are byte-for-byte
//! identical** — each was deliberately adapted for the predicate-kernel ABI,
//! so hoisting them would change emitted PTX (breaking the golden/snapshot
//! tests in `tests/ptx_golden_tests.rs`) or change error text. They are left
//! local on purpose; see the per-fn `dedup (ptx_common)` notes for the exact
//! divergence. The one helper that WAS genuinely identical —
//! `validate_kernel_name` — is already unified (V-12): this module imports
//! `ptx_gen::validate_kernel_name` rather than keeping a copy.

use std::collections::HashMap;
use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
// V-12: reuse the canonical, thorough kernel-name validator from `ptx_gen`
// instead of a weaker local duplicate, so the scan path gets the same
// hardening (reserved words, `__` prefix, `_param_` substring).
use crate::jit::ptx_gen::validate_kernel_name;
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

/// Allocation-free instruction emitter — mirrors `ptx_gen::emit_fmt!`.
///
/// `b.emit(&format!(...))` allocated a throwaway `String` per emitted
/// instruction (one heap allocation each); for large predicate specs that is
/// thousands of tiny allocations on the codegen hot path. This macro
/// `writeln!`s the formatted instruction *straight into* `b.body`, reusing the
/// existing buffer and never allocating an intermediate.
///
/// Byte-for-byte equivalence with [`PtxBuilder::emit`]: `emit` writes
/// `"\t{}\n"` where `{}` is the formatted instruction; here
/// `concat!("\t", $fmt)` prepends the same leading tab to the (always-literal)
/// format string and `writeln!` appends the same trailing newline. The emitted
/// text is identical, so the PTX golden/snapshot tests stay valid.
///
/// `emit_fmt!` from `ptx_gen` is module-private (a bare `macro_rules!`, neither
/// `#[macro_export]` nor `pub(crate) use`), so it cannot be imported here; this
/// is the same definition, kept local — same `self-contained module` rationale
/// documented at the top of this file.
///
/// Because the format arguments are evaluated *inside* the `writeln!`, operand
/// names can be passed as `b.alloc.get(reg)?` (`&str`) borrows instead of
/// `.to_string()` clones: `$b.body` and `$b.alloc` are disjoint struct fields,
/// so the immutable `alloc` borrow coexists with the mutable `body` borrow for
/// the duration of the single write.
macro_rules! emit_fmt {
    ($b:expr, $fmt:literal $(, $arg:expr)* $(,)?) => {
        writeln!($b.body, concat!("\t", $fmt) $(, $arg)*)
            .map_err(|e| BoltError::Other(format!("scan_kernel: write failed: {}", e)))
    };
}

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

    /// Allocate a pair of adjacent `rl` (b64) registers for a 128-bit
    /// (Decimal128 / i128) value split into `lo` / `hi` halves. Mirrors
    /// `ptx_gen::RegAlloc::assign_pair` — see that function for the
    /// rationale (no native 128-bit class; both halves tracked
    /// independently in `mapping`).
    ///
    /// Used by the predicate-kernel emitters for `Op::LoadColumn128`,
    /// `Op::Const128`, and the dual-register arithmetic / compare ops
    /// that can appear inside a `WHERE` predicate over Decimal128 columns
    /// (e.g. `WHERE d1 = d2`).
    fn assign_pair(&mut self, reg_lo: Reg, reg_hi: Reg) -> BoltResult<(String, String)> {
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
            .ok_or_else(|| BoltError::Other(format!("scan_kernel: undefined register {:?}", reg)))
    }

    /// Map a logical dtype to a PTX register class string.
    ///
    /// Note: `Decimal128` does NOT resolve here — its halves go through
    /// `assign_pair` directly because the IR carries an explicit `(lo,
    /// hi)` register pair rather than a single logical register with a
    /// 128-bit class. The arm below stays as an explicit reject so any
    /// stray single-Reg path tripping over a Decimal128 dtype gets a
    /// loud planner-bug error rather than silently mis-classifying.
    fn class_for(dtype: DataType) -> BoltResult<RegClass> {
        Ok(match dtype {
            DataType::Bool => "r",
            DataType::Int32 => "r",
            DataType::Int64 => "rl",
            DataType::Float32 => "f",
            DataType::Float64 => "fd",
            // v0.7: Date32 / Timestamp lower to their underlying integer
            // register classes (Date32 = i32, Timestamp = i64). Matches
            // `ptx_gen::RegAlloc::class_for`.
            DataType::Date32 => "r",
            DataType::Timestamp(_, _) => "rl",
            DataType::Utf8 => {
                return Err(BoltError::Other(
                    "scan_kernel: Utf8 not supported in PTX codegen".into(),
                ))
            }
            DataType::Decimal128(_, _) => {
                return Err(BoltError::Other(
                    "scan_kernel: Decimal128 uses split (lo, hi) register pair \
                     via assign_pair, not class_for — planner bug if this fires"
                        .into(),
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
    //
    // dedup (ptx_common): NOT hoisted. This is the INPUT-only half of the
    // validity wiring; `ptx_gen::compile` additionally computes `output_valid`
    // / `n_output_validity` / `n_extra_validity_params` (the predicate kernel
    // has no value outputs, only a mask), and its error string is prefixed
    // `ptx_gen:` per the per-module `write failed`/error convention. Extracting
    // just this block would yield a thin wrapper that still differs in error
    // text and would not cover ptx_gen's output path, increasing coupling for
    // no real reuse — so it stays local.
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

    // PERF (codegen alloc): emit straight into `b.body` via `emit_fmt!`.
    emit_fmt!(b, "mov.u32 {}, %ctaid.x;", ctaid)?;
    emit_fmt!(b, "mov.u32 {}, %ntid.x;", ntid)?;
    emit_fmt!(b, "mov.u32 {}, %tid.x;", tid_x)?;
    emit_fmt!(b, "mad.lo.s32 {}, {}, {}, {};", tid, ctaid, ntid, tid_x)?;
    // `n_rows_param_name` borrows all of `&b`, which would overlap the
    // `&mut b.body` inside the macro — compute it into a local first.
    let n_rows_param = b.n_rows_param_name(spec.inputs.len(), n_input_validity);
    emit_fmt!(b, "ld.param.u32 {}, [{}];", n_rows, n_rows_param)?;
    emit_fmt!(b, "setp.ge.u32 {}, {}, {};", pred_oob, tid, n_rows)?;
    emit_fmt!(b, "@{} bra DONE;", pred_oob)?;

    // -------- Load and globalize each input column base pointer.
    let mut input_ptrs: Vec<String> = Vec::with_capacity(spec.inputs.len());
    for (i, col) in spec.inputs.iter().enumerate() {
        if matches!(col.dtype, DataType::Utf8) {
            return Err(BoltError::Other(
                "scan_kernel: Utf8 inputs not supported in PTX codegen".into(),
            ));
        }
        let rd = b.alloc.alloc("rd");
        let param = b.param_name(i);
        emit_fmt!(b, "ld.param.u64 {}, [{}];", rd, param)?;
        emit_fmt!(b, "cvta.to.global.u64 {}, {};", rd, rd)?;
        input_ptrs.push(rd);
    }

    // -------- Load + globalize the mask output pointer (at param index N).
    let mask_ptr = b.alloc.alloc("rd");
    let mask_param = b.param_name(spec.inputs.len());
    emit_fmt!(b, "ld.param.u64 {}, [{}];", mask_ptr, mask_param)?;
    emit_fmt!(b, "cvta.to.global.u64 {}, {};", mask_ptr, mask_ptr)?;

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
            let param = b.param_name(next_param);
            emit_fmt!(b, "ld.param.u64 {}, [{}];", rd, param)?;
            emit_fmt!(b, "cvta.to.global.u64 {}, {};", rd, rd)?;
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
            // The predicate kernel ignores value Stores — projection's
            // responsibility. `Store128` is similarly skipped (it never
            // contributes to a Bool predicate's compute lineage; would
            // also blow up in `emit_op` via the i128 reject arm).
            Op::Store { .. } | Op::Store128 { .. } => continue,
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
    emit_fmt!(b, "cvt.u64.u32 {}, {};", off, tid)?;
    emit_fmt!(b, "add.s64 {}, {}, {};", addr, mask_ptr, off)?;

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
    emit_fmt!(b, "setp.ne.s32 {}, {}, 0;", p_mask, pred_phys)?;
    emit_fmt!(b, "selp.b16 {}, 1, 0, {};", rs_mask, p_mask)?;
    emit_fmt!(b, "st.global.u8 [{}], {};", addr, rs_mask)?;

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
        Op::LoadColumn {
            dst,
            col_idx,
            dtype,
        } => emit_load(b, *dst, *col_idx, *dtype, input_ptrs, tid),
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
        // CASE WHEN ... THEN ... [ELSE ...] END nested inside a WHERE
        // predicate lowers to one or more `Op::Select` ops (right-to-left
        // fold over the WHEN branches; see `physical_plan::Codegen::emit_case`).
        // Mirrors `ptx_gen::emit_select`.
        Op::Select {
            dst,
            cond,
            then_val,
            else_val,
            dtype,
        } => emit_select(b, *dst, *cond, *then_val, *else_val, *dtype),
        // Logical NOT over a Bool predicate operand — `xor.b32 dst, src, 1`.
        // A `WHERE NOT (a > b)` predicate lowers to one `Op::Not` over the
        // comparison's Bool result. Mirrors `ptx_gen::emit_not`.
        Op::Not { dst, src } => emit_not(b, *dst, *src),
        // Decimal128 / i128 dual-register ops (v0.7 Sub-task A). A WHERE
        // predicate over Decimal128 columns (e.g. `WHERE d1 = d2`,
        // `WHERE d1 + d2 > d3`) flows through this kernel via:
        //
        //   * `Op::LoadColumn128` — load the i128 row into a (lo, hi)
        //     register pair, mirroring `ptx_gen::emit_load_128`.
        //   * `Op::Const128`     — Decimal128 literal in the predicate.
        //   * `Op::Add128 / Sub128 / Mul128` — Decimal128 arithmetic
        //     nested inside the predicate.
        //   * `Op::Cmp128`       — Decimal128 comparison producing the
        //     Bool predicate value.
        //
        // `Op::Store128` remains invalid in a predicate kernel because
        // predicates never produce 128-bit values for downstream consumers
        // (the only sink is the 1-byte mask).
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
        // Store128 sinks an i128 into an output column — predicates only
        // emit a 1-byte mask, so a Store128 in a predicate kernel is a
        // planner bug.
        Op::Store128 { .. } => Err(BoltError::Other(
            "scan_kernel: Op::Store128 is not valid in a predicate kernel \
             (predicates write 1-byte mask, never i128); planner bug if this fires"
                .into(),
        )),
        // F5 Decimal128 projection-only ops. Scale-aligned Decimal *comparison*
        // in a WHERE predicate is lowered via the existing Mul128 (rescale) +
        // Cmp128 path above, so these value-producing ops (Div / CAST widen-
        // narrow / CASE-decimal select) are not expected in a predicate kernel.
        // Decline cleanly so the planner routes such predicates to the
        // projection path or host fallback rather than emitting wrong PTX.
        Op::WidenToI128 { .. }
        | Op::NarrowI128ToInt { .. }
        | Op::Div128 { .. }
        | Op::Select128 { .. }
        | Op::F64ToI128 { .. }
        | Op::I128ToF64 { .. } => Err(BoltError::Other(
            "scan_kernel: Decimal128 Div / CAST / CASE-select ops are not \
             lowered into a predicate kernel (use the projection path or host \
             fallback); planner bug if this fires in a WHERE predicate"
                .into(),
        )),
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
    emit_fmt!(b, "cvt.s64.s32 {}, {};", off, tid)?;
    emit_fmt!(b, "add.s64 {}, {}, {};", addr, vptr, off)?;
    emit_fmt!(b, "ld.global.nc.u8 {}, [{}];", byte_reg, addr)?;

    let dst_name = b.alloc.assign(dst, DataType::Bool)?;
    let pred = b.alloc.alloc("p");
    let cmp = if want_null {
        "setp.eq.u32"
    } else {
        "setp.ne.u32"
    };
    emit_fmt!(b, "{} {}, {}, 0;", cmp, pred, byte_reg)?;
    emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, pred)?;
    Ok(())
}

/// Emit PTX for `Op::Select`: `dst = cond ? then_val : else_val`. Mirrors
/// `ptx_gen::emit_select`; the predicate-only scan-kernel emitter needs
/// this op for predicates whose Bool result is itself the output of a
/// CASE WHEN expression (e.g. `WHERE CASE WHEN x > 0 THEN a > 0 ELSE FALSE END`).
fn emit_select(
    b: &mut PtxBuilder,
    dst: Reg,
    cond: Reg,
    then_val: Reg,
    else_val: Reg,
    dtype: DataType,
) -> BoltResult<()> {
    let cond_name = b.alloc.get(cond)?.to_string();
    let then_name = b.alloc.get(then_val)?.to_string();
    let else_name = b.alloc.get(else_val)?.to_string();
    let selp_ty = match dtype {
        DataType::Bool | DataType::Int32 => "s32",
        DataType::Int64 => "s64",
        DataType::Float32 => "f32",
        DataType::Float64 => "f64",
        DataType::Utf8 => {
            return Err(BoltError::Other(
                "scan_kernel: Select over Utf8 not supported \
                 (planner should have rejected CASE over string types)"
                    .into(),
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
    };
    let dst_name = b.alloc.assign(dst, dtype)?;
    let pred = b.alloc.alloc("p");
    emit_fmt!(b, "setp.ne.s32 {}, {}, 0;", pred, cond_name)?;
    emit_fmt!(
        b,
        "selp.{} {}, {}, {}, {};",
        selp_ty,
        dst_name,
        then_name,
        else_name,
        pred
    )
}

/// Emit PTX for `Op::Not`: logical negation of a Bool predicate register.
/// Mirrors `ptx_gen::emit_not` — every Bool is a canonical {0, 1} in the
/// b32 (`r`) register class, so the negation is a single low-bit flip:
///
/// ```text
///   xor.b32 %dst, %src, 1;
/// ```
///
/// Used for `WHERE NOT (<bool-expr>)` predicates whose Bool result feeds
/// the mask byte. `Codegen::emit_unary` guarantees `src` is a Bool.
fn emit_not(b: &mut PtxBuilder, dst: Reg, src: Reg) -> BoltResult<()> {
    let src_name = b.alloc.get(src)?.to_string();
    let dst_name = b.alloc.assign(dst, DataType::Bool)?;
    emit_fmt!(b, "xor.b32 {}, {}, 1;", dst_name, src_name)
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
    emit_fmt!(b, "mul.wide.u32 {}, {}, {};", off, tid, width)?;
    emit_fmt!(b, "add.s64 {}, {}, {};", addr, input_ptrs[col_idx], off)?;
    let dst_name = b.alloc.assign(dst, dtype)?;
    let suffix = ld_st_suffix(dtype)?;
    emit_fmt!(b, "ld.global.{} {}, [{}];", suffix, dst_name, addr)?;
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
        // v0.7: Date32 / Timestamp literals lower to integer movs on the
        // underlying days / ticks. Matches `ptx_gen::emit_const`.
        //
        // HARDENING (codegen-injection, mirror ptx_gen): emit the integer /
        // date / timestamp bit-pattern as HEX (`0x{:08X}` for 32-bit,
        // `0x{:016X}` for 64-bit) rather than a signed decimal literal. PTX
        // `mov.s32`/`mov.s64` is a bitwise copy, so reading the value back as
        // signed is sound — `0xFFFFFFFF` into an `.s32` register is `-1`,
        // identical to the previous decimal `-1`. This restricts the emitted
        // characters to `[0-9A-FxX]` even if a future planner regression lets
        // attacker-controlled SQL literals reach this path. Values are typed
        // integers today so no injection exists, but match the convention.
        // PERF (codegen alloc): each arm `mov`s straight into `b.body` via
        // `emit_fmt!`, dropping the per-line `format!` temporary.
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
            // emission consistent with ptx_gen's Bool arm (plain decimal).
            let n: u32 = if *v { 1 } else { 0 };
            emit_fmt!(b, "mov.b32 {}, {};", dst_name, n)
        }
        Literal::Int32(v) => {
            let dst_name = b.alloc.assign(dst, DataType::Int32)?;
            // Hex bit-pattern: `mov.s32` is a bitwise copy, so `0xFFFFFFFF`
            // here is -1, identical to writing `-1`. Removes the codegen-
            // injection surface (output restricted to `[0-9A-F]`).
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
                Utf8 => return Err(BoltError::Other("scan_kernel: cannot cast Utf8".into())),
                Decimal128(_, _) => {
                    return Err(BoltError::Plan(
                        "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
                    ))
                }
                // v0.7: identity-cast on Date32 / Timestamp = typed mov on
                // the underlying integer width.
                Date32 => "s32",
                Timestamp(_, _) => "s64",
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
            // v0.7: same temporal-Sub shape as `ptx_gen::emit_binary`.
            //   * Date32 - Date32 → Int32 (day count)
            //   * Timestamp - Timestamp → Int64 (tick count in source unit)
            // Other arithmetic on temporal operands surfaces as the
            // catch-all unsupported error below (no mnemonic).
            let is_temporal_sub = matches!(op, Sub)
                && match (dtype, result_dtype) {
                    (DataType::Date32, DataType::Int32) => true,
                    (DataType::Timestamp(_, _), DataType::Int64) => true,
                    _ => false,
                };
            if !is_temporal_sub {
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
            }
            let dst_name = b.alloc.assign(dst, result_dtype)?;
            let mnemonic = arith_mnemonic(op, dtype)?;
            emit_fmt!(b, "{} {}, {}, {};", mnemonic, dst_name, lhs_name, rhs_name)
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
            emit_fmt!(b, "{} {}, {}, {};", cmp, p, lhs_name, rhs_name)?;
            emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, p)
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
            emit_fmt!(b, "{} {}, {}, {};", mnemonic, dst_name, lhs_name, rhs_name)
        }
        Concat => {
            // String concat is host-only (see ptx_gen.rs for the same
            // arm). The physical-plan lowerer routes Concat through the
            // host-side PhysicalPlan::Project, so this kernel path
            // should never see one.
            Err(BoltError::Other(
                "scan_kernel: string concat (||) is not lowered to GPU; \
                 the planner should route through host-side execution"
                    .into(),
            ))
        }
        Mod | BitAnd | BitOr | BitXor | Shl | Shr => {
            // Modulo / bitwise / shift are integer ops kept OFF the eager-safe
            // allowlist, so a predicate using them routes to the host-side
            // PhysicalPlan::Filter (evaluated by exec::expr_agg), not this
            // WHERE-scan kernel. Projection codegen for these ops lives in
            // ptx_gen.rs; the scan-predicate path is host-only for now.
            Err(BoltError::Other(format!(
                "scan_kernel: integer op {op:?} is not lowered to the GPU scan \
                 (WHERE) kernel; the planner routes it through host-side execution"
            )))
        }
    }
}

// ---------------------------------------------------------------------------
// Decimal128 / i128 dual-register ops inside a WHERE predicate.
//
// These mirror `ptx_gen::emit_{load,const,add,sub,mul,cmp}_128` exactly —
// the only reason for the duplication is the same module-independence
// rationale documented at the top of this file (the scan kernel ABI
// evolves independently from the projection kernel ABI). Bug fixes to
// either should propagate to both.
// ---------------------------------------------------------------------------

/// Emit `Op::LoadColumn128` — two `ld.global.nc.u64` reads at byte offsets
/// `tid * 16` (lo) and `tid * 16 + 8` (hi) from input column `col_idx`'s
/// base pointer. Mirrors `ptx_gen::emit_load_128`.
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
            "scan_kernel: LoadColumn128 col_idx {} out of range (have {} inputs)",
            col_idx,
            input_ptrs.len()
        )));
    }
    let off = b.alloc.alloc("rd");
    let addr_lo = b.alloc.alloc("rd");
    let addr_hi = b.alloc.alloc("rd");
    emit_fmt!(b, "mul.wide.u32 {}, {}, 16;", off, tid)?;
    emit_fmt!(b, "add.s64 {}, {}, {};", addr_lo, input_ptrs[col_idx], off)?;
    emit_fmt!(b, "add.s64 {}, {}, 8;", addr_hi, addr_lo)?;
    let (lo_name, hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    emit_fmt!(b, "ld.global.nc.u64 {}, [{}];", lo_name, addr_lo)?;
    emit_fmt!(b, "ld.global.nc.u64 {}, [{}];", hi_name, addr_hi)?;
    Ok(())
}

/// Emit `Op::Const128` — two `mov.u64`s of the hex bit-patterns. Mirrors
/// `ptx_gen::emit_const_128`.
fn emit_const_128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    lo: u64,
    hi: u64,
) -> BoltResult<()> {
    let (lo_name, hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    emit_fmt!(b, "mov.u64 {}, 0x{:016X};", lo_name, lo)?;
    emit_fmt!(b, "mov.u64 {}, 0x{:016X};", hi_name, hi)?;
    Ok(())
}

/// Emit `Op::Add128` — `add.cc.u64` (low) then `addc.u64` (high). Mirrors
/// `ptx_gen::emit_add_128`.
fn emit_add_128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    a_lo: Reg,
    a_hi: Reg,
    b_lo: Reg,
    b_hi: Reg,
) -> BoltResult<()> {
    let a_lo_name = b.alloc.get(a_lo)?.to_string();
    let a_hi_name = b.alloc.get(a_hi)?.to_string();
    let b_lo_name = b.alloc.get(b_lo)?.to_string();
    let b_hi_name = b.alloc.get(b_hi)?.to_string();
    let (dst_lo_name, dst_hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    emit_fmt!(
        b,
        "add.cc.u64 {}, {}, {};",
        dst_lo_name,
        a_lo_name,
        b_lo_name
    )?;
    emit_fmt!(b, "addc.u64 {}, {}, {};", dst_hi_name, a_hi_name, b_hi_name)?;
    Ok(())
}

/// Emit `Op::Sub128` — `sub.cc.u64` (low) then `subc.u64` (high). Mirrors
/// `ptx_gen::emit_sub_128`.
fn emit_sub_128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    a_lo: Reg,
    a_hi: Reg,
    b_lo: Reg,
    b_hi: Reg,
) -> BoltResult<()> {
    let a_lo_name = b.alloc.get(a_lo)?.to_string();
    let a_hi_name = b.alloc.get(a_hi)?.to_string();
    let b_lo_name = b.alloc.get(b_lo)?.to_string();
    let b_hi_name = b.alloc.get(b_hi)?.to_string();
    let (dst_lo_name, dst_hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    emit_fmt!(
        b,
        "sub.cc.u64 {}, {}, {};",
        dst_lo_name,
        a_lo_name,
        b_lo_name
    )?;
    emit_fmt!(b, "subc.u64 {}, {}, {};", dst_hi_name, a_hi_name, b_hi_name)?;
    Ok(())
}

/// Emit `Op::Mul128` — schoolbook cross-multiply, low and high halves.
/// Mirrors `ptx_gen::emit_mul_128`.
fn emit_mul_128(
    b: &mut PtxBuilder,
    dst_lo: Reg,
    dst_hi: Reg,
    a_lo: Reg,
    a_hi: Reg,
    b_lo: Reg,
    b_hi: Reg,
) -> BoltResult<()> {
    let a_lo_name = b.alloc.get(a_lo)?.to_string();
    let a_hi_name = b.alloc.get(a_hi)?.to_string();
    let b_lo_name = b.alloc.get(b_lo)?.to_string();
    let b_hi_name = b.alloc.get(b_hi)?.to_string();
    let hi_acc = b.alloc.alloc("rl");
    let cross1 = b.alloc.alloc("rl");
    let cross2 = b.alloc.alloc("rl");
    let (dst_lo_name, dst_hi_name) = b.alloc.assign_pair(dst_lo, dst_hi)?;
    emit_fmt!(
        b,
        "mul.lo.u64 {}, {}, {};",
        dst_lo_name,
        a_lo_name,
        b_lo_name
    )?;
    emit_fmt!(b, "mul.hi.u64 {}, {}, {};", hi_acc, a_lo_name, b_lo_name)?;
    emit_fmt!(b, "mul.lo.u64 {}, {}, {};", cross1, a_lo_name, b_hi_name)?;
    emit_fmt!(b, "mul.lo.u64 {}, {}, {};", cross2, a_hi_name, b_lo_name)?;
    emit_fmt!(b, "add.u64 {}, {}, {};", hi_acc, hi_acc, cross1)?;
    emit_fmt!(b, "add.u64 {}, {}, {};", dst_hi_name, hi_acc, cross2)?;
    Ok(())
}

/// Emit `Op::Cmp128` — signed 128-bit comparison producing a single
/// `b32` Bool (0/1). Mirrors `ptx_gen::emit_cmp_128`; see that
/// function's rustdoc for the per-op PTX wire shape.
fn emit_cmp_128(
    b: &mut PtxBuilder,
    dst: Reg,
    op: BinaryOp,
    a_lo: Reg,
    a_hi: Reg,
    b_lo: Reg,
    b_hi: Reg,
) -> BoltResult<()> {
    let a_lo_name = b.alloc.get(a_lo)?.to_string();
    let a_hi_name = b.alloc.get(a_hi)?.to_string();
    let b_lo_name = b.alloc.get(b_lo)?.to_string();
    let b_hi_name = b.alloc.get(b_hi)?.to_string();
    let dst_name = b.alloc.assign(dst, DataType::Bool)?;

    use BinaryOp::*;
    match op {
        Eq => {
            let p_lo = b.alloc.alloc("p");
            let p_hi = b.alloc.alloc("p");
            let p = b.alloc.alloc("p");
            emit_fmt!(b, "setp.eq.u64 {}, {}, {};", p_lo, a_lo_name, b_lo_name)?;
            emit_fmt!(b, "setp.eq.s64 {}, {}, {};", p_hi, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "and.pred {}, {}, {};", p, p_lo, p_hi)?;
            emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, p)?;
        }
        NotEq => {
            let p_lo = b.alloc.alloc("p");
            let p_hi = b.alloc.alloc("p");
            let p = b.alloc.alloc("p");
            emit_fmt!(b, "setp.ne.u64 {}, {}, {};", p_lo, a_lo_name, b_lo_name)?;
            emit_fmt!(b, "setp.ne.s64 {}, {}, {};", p_hi, a_hi_name, b_hi_name)?;
            emit_fmt!(b, "or.pred {}, {}, {};", p, p_lo, p_hi)?;
            emit_fmt!(b, "selp.s32 {}, 1, 0, {};", dst_name, p)?;
        }
        Lt => {
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
                "scan_kernel: Op::Cmp128 with non-comparison op {:?} — planner bug \
                 (Codegen::emit_binary_decimal128_cmp must reject non-comparison \
                 ops before emitting Op::Cmp128)",
                op
            )));
        }
    }
    Ok(())
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
        // v0.7: Date32 / Timestamp Sub lowers to the underlying integer
        // sub. Add/Mul/Div on temporal operands has no meaningful PTX
        // form here — the catch-all below produces "unsupported".
        (Sub, Date32) => "sub.s32",
        (Sub, Timestamp(_, _)) => "sub.s64",
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
        Utf8 => return Err(BoltError::Other("scan_kernel: cannot compare Utf8".into())),
        Decimal128(_, _) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up".into(),
            ))
        }
        // v0.7: temporal compare lowers to integer setp on the underlying
        // days / ticks. Matching unit + tz is enforced upstream.
        Date32 => "s32",
        Timestamp(_, _) => "s64",
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
        // v0.7: Date32 / Timestamp ld/st widths follow the underlying
        // integer type. Matches `ptx_gen::ld_st_suffix`.
        DataType::Date32 => "s32",
        DataType::Timestamp(_, _) => "s64",
    })
}

/// Byte width of `dtype`, or an error for variable-width types.
fn byte_width(dtype: DataType) -> BoltResult<usize> {
    dtype
        .byte_width()
        .ok_or_else(|| BoltError::Other(format!("scan_kernel: variable-width dtype {:?}", dtype)))
}

// V-12: the previous weaker local `validate_kernel_name` (only empty /
// leading-char / charset checks) has been removed. `scan_kernel::compile`
// now calls the canonical `ptx_gen::validate_kernel_name` (imported above),
// which additionally rejects PTX reserved words, the `__` prefix, and the
// `_param_` substring — closing the validation gap on the scan path.

/// Write the `.visible .entry` signature: N input ptrs, mask output ptr, K
/// input-validity ptrs (one per flagged input), n_rows.
///
/// dedup (ptx_common): NOT hoisted. Emits a DIFFERENT byte sequence from
/// `ptx_gen::write_signature`: the param count is `n_inputs + 1 + K` (the `+1`
/// is the mask output, which has no analogue in ptx_gen's `inputs + outputs +
/// extra` formula), the pointer params are plain `.param .u64` (ptx_gen adds
/// `.ptr .global .align 16`), and `n_rows_param_name` here is the 2-arg scan
/// variant. The golden tests pin this exact text — merging would break them.
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
///
/// dedup (ptx_common): NOT hoisted. Diverges from `ptx_gen::write_reg_decls`,
/// which declares 6 classes; this path declares 7 — it adds `("rs", "b16")`
/// for the narrowed mask-byte source unique to the predicate kernel. Different
/// `decls` table => different emitted `.reg` block => stays local.
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
///
/// dedup (ptx_common): NOT hoisted. The `write failed` message is prefixed
/// with the owning module name (`scan_kernel:`) — a convention every JIT
/// codegen module follows (ptx_gen, sort_kernel, agg_kernels, ...). A shared
/// helper would have to pick one prefix and would silently change this path's
/// error text, so each module keeps its own one-liner.
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
        let err =
            compile_predicate_kernel(&spec, "bolt_predicate").expect_err("must reject Utf8 inputs");
        assert!(format!("{}", err).contains("Utf8"));
    }

    #[test]
    fn rejects_bad_kernel_name() {
        let spec = region_eq_1_spec();
        let err = compile_predicate_kernel(&spec, "1bad")
            .expect_err("must reject kernel names that don't start with letter/underscore");
        assert!(format!("{}", err).contains("must start with"));
    }

    /// V-12: the scan path now delegates to the canonical
    /// `ptx_gen::validate_kernel_name`, so the checks that the old weaker
    /// local copy lacked (PTX reserved word, `__` prefix, `_param_`
    /// substring) must be enforced here too.
    #[test]
    fn rejects_reserved_and_compiler_reserved_names_on_scan_path() {
        let spec = region_eq_1_spec();

        // PTX reserved identifier.
        let err = compile_predicate_kernel(&spec, "mov")
            .expect_err("must reject PTX reserved identifier as kernel name");
        assert!(
            format!("{}", err).contains("reserved identifier"),
            "got: {err}"
        );

        // Compiler-reserved `__` prefix.
        let err = compile_predicate_kernel(&spec, "__bolt_kernel")
            .expect_err("must reject `__`-prefixed (compiler-reserved) kernel name");
        assert!(
            format!("{}", err).contains("compiler-reserved"),
            "got: {err}"
        );

        // `_param_` substring would collide with synthesised parameter names.
        let err = compile_predicate_kernel(&spec, "bolt_param_evil")
            .expect_err("must reject kernel name containing `_param_`");
        assert!(format!("{}", err).contains("_param_"), "got: {err}");
    }
}
