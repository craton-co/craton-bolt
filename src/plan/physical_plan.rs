// SPDX-License-Identifier: Apache-2.0

//! Physical plan: column-ordinal-resolved, register-machine IR for GPU codegen.

use std::collections::HashMap;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{
    join_combined_schema, AggregateExpr, BinaryOp, DataType, Expr, Field, JoinType, Literal,
    LogicalPlan, Schema, SortExpr, UnaryOp,
};

/// SSA register handle. Just an index into the IR's value table.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub(crate) u32);

impl Reg {
    /// Read-only accessor for the underlying register index. Useful for
    /// external rustdoc consumers / debuggers; the field itself is
    /// `pub(crate)` so the wire representation isn't part of the public
    /// SemVer contract.
    pub fn id(self) -> u32 {
        self.0
    }
}

/// A typed value in the IR: a register plus its known dtype.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct Value {
    /// The register holding the value.
    pub reg: Reg,
    /// The runtime dtype of the value.
    pub dtype: DataType,
}

/// A single instruction in the IR.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum Op {
    /// Load row `tid` of input column `col_idx` into a register.
    LoadColumn {
        /// Destination register.
        dst: Reg,
        /// Ordinal of the column in `KernelSpec::inputs`.
        col_idx: usize,
        /// Dtype of the loaded value.
        dtype: DataType,
    },
    /// A literal constant value.
    Const {
        /// Destination register.
        dst: Reg,
        /// The constant.
        lit: Literal,
    },
    /// Cast `src` from its current dtype to `dtype`.
    Cast {
        /// Destination register.
        dst: Reg,
        /// Source register.
        src: Reg,
        /// Original dtype of `src`.
        from: DataType,
        /// Target dtype.
        to: DataType,
    },
    /// Binary op. Both operands must already have the same dtype (insert Cast first).
    Binary {
        /// Destination register.
        dst: Reg,
        /// The operator.
        op: BinaryOp,
        /// Left operand register.
        lhs: Reg,
        /// Right operand register.
        rhs: Reg,
        /// Common dtype of `lhs` and `rhs`.
        dtype: DataType,
        /// Dtype of the produced value.
        result_dtype: DataType,
    },
    /// Store value `src` to output column `col_idx` at row `tid` (mask permitting).
    Store {
        /// Source register.
        src: Reg,
        /// Ordinal of the column in `KernelSpec::outputs`.
        col_idx: usize,
        /// Dtype of the stored value.
        dtype: DataType,
    },
    /// Per-row null test against an input column's validity bitmap.
    ///
    /// Emits PTX that loads the validity byte for the current row from the
    /// per-input validity pointer (`input_validity_ptrs[validity_input]`,
    /// passed as a kernel parameter when
    /// [`KernelSpec::input_has_validity`]`[validity_input]` is `true`) and
    /// produces a Bool (0/1) result in `dst`:
    ///
    /// * `want_null == true`  → result is 1 iff the byte is 0 (the row IS NULL).
    /// * `want_null == false` → result is 1 iff the byte is non-zero (the row
    ///   IS NOT NULL).
    ///
    /// The codegen ([`Codegen::emit_unary`]) only emits this op when the
    /// operand is a bare column reference whose schema marks it nullable;
    /// non-nullable operands collapse to a constant Bool at plan time.
    /// Compound unary operands (e.g. `(x + y) IS NULL`) take the host
    /// fallback in [`predicate_contains_unary`].
    IsNullCheck {
        /// Destination register (Bool predicate).
        dst: Reg,
        /// Ordinal of the source column in `KernelSpec::inputs`. The
        /// emitter indexes the kernel's validity-pointer table by this slot;
        /// for it to resolve to a real pointer,
        /// `KernelSpec::input_has_validity[validity_input]` must be `true`.
        validity_input: usize,
        /// `true` if this is `IS NULL` (result 1 when validity byte is 0);
        /// `false` for `IS NOT NULL` (result 1 when validity byte is
        /// non-zero).
        want_null: bool,
    },
    /// Predicated value selection: `dst = cond ? then_val : else_val`.
    ///
    /// Used as the kernel-level building block for SQL `CASE WHEN cond THEN a
    /// ELSE b END`. The codegen (`Codegen::emit_case`) folds a multi-arm
    /// CASE right-to-left into a chain of these ops:
    ///
    /// ```text
    ///   CASE WHEN c1 THEN v1
    ///        WHEN c2 THEN v2
    ///        ELSE e
    ///   END
    /// ```
    ///
    /// lowers to (logically):
    ///
    /// ```text
    ///   r1 = Select(c2, v2, e)
    ///   r0 = Select(c1, v1, r1)
    /// ```
    ///
    /// `cond` must be a Bool register; `then_val` and `else_val` must already
    /// share the unified result dtype (insert `Op::Cast` first). PTX lowers
    /// this to a per-dtype `selp.<ty>` after materialising `cond` as a `%p`
    /// predicate via `setp.ne.s32 cond, 0`.
    ///
    /// v0.7 minimum: Bool / Int32 / Int64 / Float32 / Float64. Utf8 /
    /// Decimal128 / Date / Timestamp values are rejected at
    /// `Codegen::emit_case`.
    Select {
        /// Destination register holding the chosen value.
        dst: Reg,
        /// Bool register (0/1) driving the choice.
        cond: Reg,
        /// Register holding the value selected when `cond == 1`.
        then_val: Reg,
        /// Register holding the value selected when `cond == 0`.
        else_val: Reg,
        /// Common dtype of `then_val` / `else_val` and of `dst`.
        dtype: DataType,
    },
    // ---------------------------------------------------------------------
    // Decimal128 / i128 dual-register ops (v0.7 Sub-task A).
    //
    // The PTX side has no native 128-bit register class, so an i128 value
    // is represented in the IR as a *pair* of u64 registers (`_lo` / `_hi`,
    // little-endian: `value == ((hi as i128) << 64) | (lo as u128 as i128)`).
    // Every 128-bit op therefore carries a pair of destination registers
    // and (for binary ops) a pair of source-operand registers each. The
    // pair-of-`Reg` representation (rather than a `Reg128(Reg, Reg)`
    // wrapper) preserves the SSA invariant — each `Reg` is still written
    // exactly once — and lets the existing dataflow walker treat the
    // halves as independent values reachable through a single op.
    //
    // No end-to-end wiring lives in this slice: `Codegen::emit_column` /
    // `emit_literal` / `emit_binary` still reject Decimal128 at lowering
    // time, so these ops are unreachable from `lower()` today. They exist
    // so the PTX emitter has a structurally complete IR target before the
    // upload / Codegen-routing follow-up commits land.
    // ---------------------------------------------------------------------
    /// Load row `tid` of an input Decimal128 column into a pair of u64
    /// registers. Emits two `ld.global.nc.u64` reads at byte offsets
    /// `tid * 16` (lo) and `tid * 16 + 8` (hi) from the input buffer base.
    LoadColumn128 {
        /// Destination register for the low 64 bits.
        dst_lo: Reg,
        /// Destination register for the high 64 bits.
        dst_hi: Reg,
        /// Ordinal of the column in `KernelSpec::inputs`.
        col_idx: usize,
    },
    /// Materialise a 128-bit constant into a pair of u64 registers by
    /// emitting two `mov.u64` instructions of the hex bit-patterns.
    Const128 {
        /// Destination register for the low 64 bits.
        dst_lo: Reg,
        /// Destination register for the high 64 bits.
        dst_hi: Reg,
        /// Low 64 bits of the constant (little-endian half).
        lo: u64,
        /// High 64 bits of the constant (little-endian half).
        hi: u64,
    },
    /// Store value `src_lo`/`src_hi` to an output Decimal128 column at row
    /// `tid` (mask permitting). Emits two `st.global.u64` writes at byte
    /// offsets `tid * 16` (lo) and `tid * 16 + 8` (hi).
    Store128 {
        /// Source register for the low 64 bits.
        src_lo: Reg,
        /// Source register for the high 64 bits.
        src_hi: Reg,
        /// Ordinal of the column in `KernelSpec::outputs`.
        col_idx: usize,
    },
    /// 128-bit add: lowered as `add.cc.u64` on the low half followed by
    /// `addc.u64` on the high half (carry propagation via the implicit
    /// `%CC` carry flag — see PTX ISA §8.7.1.1).
    Add128 {
        /// Destination low half.
        dst_lo: Reg,
        /// Destination high half.
        dst_hi: Reg,
        /// Left operand low half.
        a_lo: Reg,
        /// Left operand high half.
        a_hi: Reg,
        /// Right operand low half.
        b_lo: Reg,
        /// Right operand high half.
        b_hi: Reg,
    },
    /// 128-bit subtract: lowered as `sub.cc.u64` then `subc.u64` (borrow
    /// propagation via `%CC`).
    Sub128 {
        /// Destination low half.
        dst_lo: Reg,
        /// Destination high half.
        dst_hi: Reg,
        /// Left operand low half.
        a_lo: Reg,
        /// Left operand high half.
        a_hi: Reg,
        /// Right operand low half.
        b_lo: Reg,
        /// Right operand high half.
        b_hi: Reg,
    },
    /// 128-bit multiply (truncated to 128 bits — matches `i128::wrapping_mul`
    /// / Arrow Decimal128 arithmetic semantics).
    ///
    /// Lowered as schoolbook cross-multiply:
    ///
    /// ```text
    ///   dst_lo = a_lo * b_lo                                  (mul.lo.u64)
    ///   dst_hi = mul.hi(a_lo, b_lo)                           (mul.hi.u64)
    ///          + a_lo * b_hi                                  (mul.lo.u64)
    ///          + a_hi * b_lo                                  (mul.lo.u64)
    /// ```
    ///
    /// The `a_hi * b_hi` partial product (and the high halves of the two
    /// cross terms) shifts into bits 128+, which we discard for wrapping
    /// semantics. No Karatsuba — clarity > shaving one multiply.
    Mul128 {
        /// Destination low half.
        dst_lo: Reg,
        /// Destination high half.
        dst_hi: Reg,
        /// Left operand low half.
        a_lo: Reg,
        /// Left operand high half.
        a_hi: Reg,
        /// Right operand low half.
        b_lo: Reg,
        /// Right operand high half.
        b_hi: Reg,
    },
}

/// Description of an input column the kernel consumes.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ColumnIO {
    /// Column name.
    pub name: String,
    /// Column dtype.
    pub dtype: DataType,
}

/// A single GPU kernel description, derived from a fused (Scan -> [Filter ->] Project) chain.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct KernelSpec {
    /// Columns the kernel reads, in load order.
    pub inputs: Vec<ColumnIO>,
    /// Columns the kernel writes, in store order.
    pub outputs: Vec<ColumnIO>,
    /// Linear IR.
    pub ops: Vec<Op>,
    /// Optional predicate register; if Some, only rows where this is true emit output.
    pub predicate: Option<Reg>,
    /// Number of registers used by this kernel.
    pub register_count: u32,
    /// Pre-stage NULL handling (Option B): one entry per input column. `true`
    /// means the caller will pass a parallel `*u8` validity pointer (1=valid,
    /// 0=null) AFTER the value+output pointer list and the codegen should
    /// load the validity byte at `tid` and AND it into the combined-validity
    /// register that drives every output's validity store.
    ///
    /// Default is `Vec::new()` which is treated as "no input carries
    /// validity" and the existing PTX layout is emitted verbatim — every
    /// existing caller (e.g. the projection path in `engine.rs`) continues
    /// to work bit-for-bit. When non-empty, must be parallel to `inputs`.
    ///
    /// # Planner visibility (PV-stage-d)
    ///
    /// Populated at lowering time by
    /// [`populate_input_validity`] via
    /// [`crate::plan::sql_frontend::TableProvider::has_nulls`]. The default —
    /// safe-`false` for every input — preserves the legacy "no-validity"
    /// path: an empty vector OR a vector of all-false means the kernel
    /// emits validity-free PTX exactly as before.
    ///
    /// PV-stage-d wires this through `ptx_gen.rs`'s pre-stage emitter so
    /// the per-output validity AND-tree references only the LoadColumn ops
    /// flagged here (see `ptx_gen::output_input_dependencies`). The GROUP
    /// BY value-column validity path in
    /// [`crate::jit::hash_kernels::compile_groupby_agg_kernel_with_validity`]
    /// also consults the executor-time signal as a fallback.
    #[doc(hidden)]
    pub input_has_validity: Vec<bool>,
    /// Pre-stage NULL handling (Option B): one entry per output column.
    /// `true` means the caller will pass a parallel `*u8` validity pointer
    /// where the kernel writes the per-row combined-validity result. The
    /// validity stores are appended after the regular value stores.
    ///
    /// Default `Vec::new()` => no output carries validity (no validity
    /// pointers added, no validity stores emitted). When non-empty, must
    /// be parallel to `outputs`.
    #[doc(hidden)]
    pub output_has_validity: Vec<bool>,
}

/// v0.7: domain-separated planner IR for a scalar (no-GROUP-BY) GPU
/// reduction kernel.
///
/// The scalar-aggregate executor in [`crate::exec::aggregate`] historically
/// fed `(op, dtype)` directly to the JIT layer, which means every warm call
/// still paid the per-PTX-text codegen cost (`compile_reduction_kernel` /
/// `compile_avg_reduction_kernel`). The projection path already routes through
/// a `KernelSpec`-keyed cache (see [`crate::exec::module_cache`]); this type
/// is the matching planner-IR handle for the scalar-aggregate family.
///
/// # Why a separate spec type?
///
/// `KernelSpec` describes the fused scan/filter/project IR — its `inputs`,
/// `outputs`, and `ops` list don't make sense for a pure reduction. A scalar
/// aggregate is fully described by `(op, input_dtype)`: every other knob
/// (block size, identity, combine instruction) is derived inside
/// [`crate::jit::agg_kernels`] from those two fields. Keeping the shape
/// minimal also keeps the `Debug` fingerprint short, which is what the
/// module cache hashes on the warm path.
///
/// The `Debug` shape is intentionally distinct from `KernelSpec`'s
/// (`KernelSpec { inputs: [...], ... }`) so the disk-cache key prefix wired
/// up in [`crate::exec::module_cache::get_or_build_module_for_scalar_agg`]
/// (`"scalar_agg::"`) further domain-separates the two PTX families on
/// disk — a hand inspection of a cache directory shows immediately which
/// family produced an entry.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScalarAggSpec {
    /// The reduction operator.
    pub op: ScalarAggOp,
    /// Element dtype of the input column (what the kernel loads). The
    /// accumulator dtype is derived from `(op, input_dtype)` inside the
    /// JIT layer per `crate::jit::agg_kernels::reduction_output_dtype` /
    /// `compile_avg_reduction_kernel`.
    pub input_dtype: DataType,
}

/// Reduction family for a [`ScalarAggSpec`]. Mirrors the variants the
/// scalar-aggregate executor actually dispatches into the JIT layer for —
/// the broader `AggregateExpr` enum has additional variants (VAR_*, STDDEV_*)
/// that the v0.5/v0.6 path handles host-side via
/// [`crate::exec::welford`], so they're absent here.
///
/// `Avg` is its own variant rather than a (`Sum`, `Count`) decomposition
/// because the scalar-aggregate executor emits a **fused** single-kernel
/// AVG (`bolt_avg_reduce`) that produces both partial buffers in one pass.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarAggOp {
    /// SUM — `bolt_reduce` with `ReduceOp::Sum`.
    Sum,
    /// MIN — `bolt_reduce` with `ReduceOp::Min`.
    Min,
    /// MAX — `bolt_reduce` with `ReduceOp::Max`.
    Max,
    /// COUNT — `bolt_reduce` with `ReduceOp::Count`. (Distinct from `Sum`
    /// so the disk-cache key fingerprint differs; the JIT layer treats them
    /// identically beyond the identity value.)
    Count,
    /// AVG — the fused `bolt_avg_reduce` kernel; emits `(sum, count)` per-block.
    Avg,
}

/// Description of an aggregation kernel.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct AggregateSpec {
    /// Columns read from the pre-aggregation kernel (or directly from the scan).
    pub inputs: Vec<ColumnIO>,
    /// Group-by key column ordinals (into `inputs`).
    pub group_by: Vec<usize>,
    /// Aggregates to compute; codegen will lower further.
    pub aggregates: Vec<AggregateExpr>,
    /// Output schema (group-by columns first, then aggregate result columns).
    pub output_schema: Schema,
    /// PV-stage-f: parallel to `inputs` — `true` means the underlying
    /// table column carries an Arrow validity bitmap (i.e. its
    /// `null_count() > 0` per the [`crate::plan::sql_frontend::TableProvider`]
    /// extension).
    ///
    /// Default is `Vec::new()` which is treated as "no input carries
    /// validity" — every existing call site that builds an `AggregateSpec`
    /// without consulting the provider sees the legacy host-strip
    /// fallback. When non-empty, must be parallel to `inputs`.
    ///
    /// Populated at lowering time by [`populate_input_validity`] (which
    /// also fills [`KernelSpec::input_has_validity`] for projection /
    /// pre-aggregation kernels). The single-key GROUP BY executors
    /// ([`crate::exec::groupby`], [`crate::exec::groupby_valid`]) consult
    /// this flag together with the runtime per-column null check to
    /// decide whether to dispatch through the native `_with_validity`
    /// kernel variants or fall back to the host-strip path.
    #[doc(hidden)]
    pub input_has_validity: Vec<bool>,
}

// ---------------------------------------------------------------------------
// v0.7: KernelSpec coverage for non-projection kernel kinds.
//
// Background. The original [`KernelSpec`] struct above models the
// fused-projection (+ optional filter) kernel ONLY — its fields (`inputs`,
// `outputs`, `ops`, `predicate`, …) are accessed from over a dozen call sites
// (the executor in `engine.rs`, `groupby_with_pre.rs`, the PTX emitter in
// `jit::ptx_gen`, scan kernels, etc.). Refactoring that struct into an enum
// would ripple through every field-access site AND change its `Debug` output
// — which is the disk-cache key shape via `hash_to_key(spec_hash_hi,
// spec_hash_lo)` in `exec::engine`. Both are out of scope for this task per
// the brief; see the explicit escape hatch documented there.
//
// Instead we introduce SIBLING spec types — one per non-projection kernel
// kind — each independently cacheable through a per-kind wrapper around the
// existing `crate::exec::module_cache::get_or_build_module` machinery
// (wired in a follow-up; see #14 / v0.7). Each sibling type:
//
//   * `#[derive(Debug, Clone, PartialEq, Eq, Hash)]` so the existing
//     `KernelSpecKey::new(spec, entry)` machinery in `exec::module_cache`
//     (which hashes the `Debug` output with a domain-separated
//     `DefaultHasher`) keeps working bit-for-bit, and so callers can use
//     these specs as `HashMap` keys directly if they prefer to bypass the
//     module cache.
//   * Carries every knob the codegen / launcher consult — dtype, op flavour,
//     pass index, key shape, etc. — so two specs that differ in any
//     observable way produce different `Debug` strings and therefore land in
//     distinct cache slots.
//   * Provides no field-access dependency on the existing `KernelSpec` —
//     each new type is self-contained, so wiring a single executor through
//     to a new spec is a localised change.
//
// The [`KernelSpecKind`] wrapper at the bottom is a single envelope every
// executor can use as a uniform cache-key carrier without having to spell
// out the variant at the call site; the projection variant wraps the
// existing struct unchanged so its hash shape is `Projection(KernelSpec
// { … })`. That is intentionally DIFFERENT from the bare `KernelSpec { … }`
// hash used by the wired projection cache today — wiring callers through
// `KernelSpecKind::Projection(spec)` would force a cache rebuild on first
// run. Callers that want the legacy projection-cache shape continue passing
// `&KernelSpec` directly; callers that want the new uniform envelope use
// `KernelSpecKind`.

/// Which entry point of the hash-join kernel set this spec selects. The
/// hash-join PTX is emitted by `compile_*_kernel` helpers in
/// `crate::jit::hash_join_kernel`; each helper takes no arguments and
/// returns a fixed PTX string for a fixed entry symbol. The codegen-time
/// knob is therefore which helper to call.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HashJoinKernelKind {
    /// `compile_build_kernel` — insert encoded keys into the open-addressed
    /// table at `keys_table_ptr` / `row_idx_table_ptr`.
    Build,
    /// `compile_probe_kernel` — emit `(probe_idx, build_idx)` pairs for every
    /// matched probe row. Linear-probe variant.
    Probe,
    /// `compile_probe_kernel_tiled` — tiled-shared-memory probe variant for
    /// small build sides.
    ProbeTiled,
    /// `compile_build_collision_kernel` — build variant that records
    /// per-key collision chains.
    BuildCollision,
    /// `compile_probe_collision_kernel` — probe variant matching
    /// `BuildCollision`.
    ProbeCollision,
    /// `compile_build_aos_kernel` — array-of-structs build kernel for the
    /// AOS code path.
    BuildAos,
    /// `compile_probe_aos_kernel` — array-of-structs probe kernel.
    ProbeAos,
    /// `compile_unmatched_build_kernel` — emit unmatched build-side row
    /// indices for `LEFT` / `FULL` outer joins.
    UnmatchedBuild,
    /// `compile_cross_kernel` — full Cartesian product for `CROSS JOIN`.
    Cross,
    /// `compile_string_hash_kernel` — Utf8 candidate filter for
    /// `KeyShape::SingleI32Candidate`. The `_i64` flavour is selected by
    /// the boolean field below.
    StringHash,
}

/// Spec for one entry point of the hash-join kernel set.
///
/// The hash-join kernels themselves don't take a `DataType` parameter —
/// every encoded key arrives as an `i64` at the kernel boundary (see
/// `encode_keys_for_shape` in `crate::exec::gpu_join`). The `key_dtype`
/// field is stored here purely so the cache key is unambiguous across
/// joins built on different source-column types (which DO produce
/// different host-side encoders and would otherwise share a slot here —
/// harmless for correctness, surprising for telemetry).
///
/// `string_hash_returns_i64` is the single bit that distinguishes the two
/// `StringHash` flavours (`bolt_string_hash` vs `bolt_string_hash_i64`);
/// it's ignored for every other variant.
///
/// # Cache key
///
/// The `Debug` impl emits all three fields, so distinct
/// `(kind, key_dtype, string_hash_returns_i64)` triples land in distinct
/// cache slots.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HashJoinKernelSpec {
    /// Which of the hash-join PTX entry points this spec selects.
    pub kind: HashJoinKernelKind,
    /// Dtype of the source key column BEFORE host-side encoding to i64.
    /// Kept here for cache-key disambiguation only; the kernel itself
    /// always operates on `i64`-encoded keys.
    pub key_dtype: DataType,
    /// For `kind == StringHash`: `true` selects the `_i64` flavour
    /// (`bolt_string_hash_i64`), `false` selects the regular flavour
    /// (`bolt_string_hash`). Ignored for every other `kind`.
    pub string_hash_returns_i64: bool,
}

/// Which pass of the radix-sort driver this spec compiles. The radix
/// driver in `crate::exec::gpu_sort` (and the per-pass kernels in
/// `crate::jit::sort_kernel_radix`) breaks the sort into a histogram
/// pass and a scatter pass per 4-bit digit; the same PTX is reused
/// across passes (the `shift` is a kernel parameter, not a codegen knob).
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RadixSortPass {
    /// `compile_radix_histogram` — count digit occurrences into a per-block
    /// histogram.
    Histogram,
    /// `compile_radix_scatter` — scatter keys to their final positions
    /// using the prefix-summed histogram.
    Scatter,
    /// `compile_radix_scatter_with_indices` — variant of `Scatter` that
    /// carries a parallel `u32` row-index payload through the scatter step.
    /// This is the standard path for multi-column ORDER BY (see
    /// `gpu_sort::run_radix_pipeline_*`). It is a distinct codegen knob from
    /// `Scatter` (different PTX entry point, different ABI) and therefore
    /// must occupy its own cache slot.
    ScatterWithIndices,
    /// `compile_radix_msb_flip` — one-shot in-place XOR over the keys buffer
    /// that flips the MSB so the per-pass histogram/scatter kernels can treat
    /// signed keys as plain unsigned bit-blobs.
    ///
    /// In the current `gpu_sort` driver this kernel is **not invoked** — the
    /// host-side pre-transform during gather subsumes both the signed-MSB
    /// XOR and the per-key DESC bit-not in one pass (see the long comment
    /// at `run_radix_pipeline_i32`). The variant is retained on the planner
    /// IR so the kernel-side helper can be re-introduced (e.g. for an
    /// in-place / no-gather code path) without churning the IR enum.
    MsbFlip,
}

/// Spec for one pass of the radix sort kernel pair.
///
/// The codegen knobs are exactly `(pass, dtype)`. The per-pass shift is a
/// kernel parameter passed at launch time — it does NOT participate in
/// codegen and therefore does NOT participate in the cache key. Two
/// `(Histogram, Int32)` calls at shift 0 and shift 4 hit the same cached
/// module.
///
/// # Cache key
///
/// The `Debug` impl emits both fields, so a `(Histogram, Int32)` spec and
/// a `(Histogram, Int64)` spec hash to distinct strings and never collide.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RadixSortKernelSpec {
    /// Which pass (Histogram or Scatter) this spec compiles.
    pub pass: RadixSortPass,
    /// Key dtype the kernel reads. Drives the PTX load-suffix / register
    /// class (`b32` vs `b64`).
    pub dtype: DataType,
}

/// Which prefix-scan algorithm a [`CompactionKernelSpec::PrefixScan`]
/// variant compiles. Mirror of `crate::exec::gpu_compact::PrefixScanAlgo`
/// (the original is module-private); the two enums are kept in sync by
/// hand. The unit test [`compaction_spec_prefix_scan_round_trips`] pins
/// that every variant produces a distinct cache key.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrefixScanAlgoTag {
    /// O(n log n) ping-pong scan — `SCAN_KERNEL_ENTRY`, the default.
    HillisSteele,
    /// O(n) upsweep/downsweep — `SCAN_KERNEL_ENTRY_BLELLOCH`.
    Blelloch,
    /// Single-pass decoupled-lookback — `SCAN_KERNEL_ENTRY_LOOKBACK`,
    /// runs with an extra `partial_status` buffer.
    Lookback,
}

/// Which compaction-pipeline kernel this spec compiles. The compaction
/// pipeline in `crate::exec::gpu_compact` has three distinct PTX shapes
/// — prefix scan over a `u8` mask, per-dtype gather, and a Bool-nullable
/// gather — each with its own knobs.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompactionKernelKind {
    /// Prefix-scan over the keep-mask, parameterised by which algorithm
    /// implementation is selected (see [`PrefixScanAlgoTag`]).
    PrefixScan(PrefixScanAlgoTag),
    /// Per-dtype gather kernel from `compile_gather_kernel(dtype)`.
    Gather(DataType),
    /// Bool-with-validity gather variant — used by
    /// `gather_bool_nullable`. The validity store path is distinct PTX
    /// from the plain `Gather(Bool)` variant.
    GatherBoolNullable,
}

/// Spec for one of the compaction-pipeline kernels.
///
/// The codegen-time knob is entirely captured by the `kind` variant —
/// `PrefixScan` selects between three algorithms, `Gather` is
/// parameterised by dtype, and `GatherBoolNullable` is a single fixed
/// shape. Wrapping all three in one spec type lets the executor pass a
/// single `&CompactionKernelSpec` to the cache layer regardless of which
/// pipeline stage is being looked up.
///
/// # Cache key
///
/// The `Debug` impl emits the wrapped variant in full, so
/// `PrefixScan(HillisSteele)` and `PrefixScan(Blelloch)` hash to distinct
/// strings, and `Gather(Int32)` vs `Gather(Int64)` likewise.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CompactionKernelSpec {
    /// Which compaction-pipeline kernel this spec compiles.
    pub kind: CompactionKernelKind,
}

/// Uniform-envelope wrapper over every kernel-spec kind the planner can
/// produce. Lets the cache layer accept a single type for every executor
/// without each call site spelling the variant.
///
/// # Hash-shape caveat
///
/// The `Debug` output of `KernelSpecKind::Projection(spec)` is
/// `Projection(KernelSpec { … })` — strictly different from the bare
/// `KernelSpec { … }` shape that the wired projection cache in
/// `engine.rs::get_or_build_module` produces today. **Callers that want
/// to hit the legacy projection-cache slot must continue passing
/// `&KernelSpec` directly**, not the envelope; the envelope is for the
/// new spec kinds (`ScalarAgg`, `HashJoin`, `RadixSort`, `Compaction`)
/// that have no legacy cache slot to collide with.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum KernelSpecKind {
    /// The fused-projection / filter kernel — wraps the existing
    /// [`KernelSpec`] struct unchanged. See caveat above on hash shape.
    Projection(KernelSpec),
    /// A scalar-reduction kernel; see [`ScalarAggSpec`].
    ScalarAgg(ScalarAggSpec),
    /// One entry point of the hash-join kernel set; see
    /// [`HashJoinKernelSpec`].
    HashJoin(HashJoinKernelSpec),
    /// One pass of the radix sort kernel pair; see
    /// [`RadixSortKernelSpec`].
    RadixSort(RadixSortKernelSpec),
    /// One of the compaction-pipeline kernels; see
    /// [`CompactionKernelSpec`].
    Compaction(CompactionKernelSpec),
}

/// The top-level physical plan: a small ordered pipeline of kernels.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// Project (optionally with filter): single fused kernel over a table scan.
    Projection {
        /// Source table name.
        table: String,
        /// The fused projection kernel.
        kernel: KernelSpec,
        /// Output schema of the kernel.
        output_schema: Schema,
    },
    /// Aggregate over a (possibly filtered) projection.
    Aggregate {
        /// Source table name.
        table: String,
        /// Optional pre-aggregation kernel resolving group keys + aggregate-input exprs.
        pre: Option<KernelSpec>,
        /// The aggregation specification.
        aggregate: AggregateSpec,
    },
    /// DISTINCT over `input`'s rows. Output schema = `input.output_schema()`.
    Distinct {
        /// Source plan.
        input: Box<PhysicalPlan>,
    },
    /// LIMIT [OFFSET]. Output schema = `input.output_schema()`.
    Limit {
        /// Source plan.
        input: Box<PhysicalPlan>,
        /// Maximum number of rows to emit.
        limit: usize,
        /// Number of leading rows to skip.
        offset: usize,
    },
    /// ORDER BY over `input`. Output schema = `input.output_schema()`.
    Sort {
        /// Source plan.
        input: Box<PhysicalPlan>,
        /// Sort keys, most-significant first.
        sort_exprs: Vec<SortExpr>,
    },
    /// UNION ALL: concatenate `inputs` in order. Schema is the first input's.
    /// (Dedup UNION is `Distinct(Union { ... })` in the logical plan.)
    Union {
        /// Branches to concatenate, in source order.
        inputs: Vec<PhysicalPlan>,
    },
    /// Pure column-rename / reorder layer over `input`. Used when the SQL
    /// frontend places a `Project` on top of an `Aggregate` (or other
    /// non-scan-chain operator) purely to surface SELECT-list order and
    /// aliases. Each `exprs` entry must be `Column(name)` or
    /// `Alias(Column(name), out_name)`; the executor just rearranges and
    /// renames the input batch's columns to match `output_schema`. No
    /// compute happens here — anything more elaborate (e.g. post-aggregate
    /// arithmetic) is rejected upstream.
    Project {
        /// Source plan.
        input: Box<PhysicalPlan>,
        /// One entry per output column; each references a column of `input`.
        exprs: Vec<Expr>,
        /// Output schema, in `exprs` order with aliases applied.
        output_schema: Schema,
    },
    /// Host-side post-aggregate (or other non-scan-chain) filter layer.
    ///
    /// Used when a `LogicalPlan::Filter` sits above an operator that
    /// `lower_projection` can't fold into a single fused kernel — most
    /// importantly `HAVING`, which produces `Filter { Project { Aggregate { .. } } }`.
    /// The aggregate's output is already a small `RecordBatch` (one row per
    /// group), so the executor evaluates `predicate` host-side via
    /// `expr_agg::eval_expr` and applies `arrow::compute::filter` to keep
    /// only the surviving rows. The output schema is `input.output_schema()`
    /// — Filter doesn't add or rename columns.
    Filter {
        /// Source plan whose output rows are filtered.
        input: Box<PhysicalPlan>,
        /// Boolean expression evaluated against `input`'s output schema.
        predicate: Expr,
    },
    /// JOIN (INNER, LEFT, RIGHT, FULL, CROSS). The `output_schema` is
    /// `left.output_schema() ++ right` with right-side collisions
    /// disambiguated by `join_combined_schema`, and with nullability of
    /// the non-preserved side widened for outer joins; it's stored on the
    /// variant so `output_schema()` can return a borrow-stable `&Schema`
    /// without allocating per call.
    Join {
        /// Left input.
        left: Box<PhysicalPlan>,
        /// Right input.
        right: Box<PhysicalPlan>,
        /// Join kind.
        join_type: JoinType,
        /// Equi-join predicate pairs `(left_expr, right_expr)`. Empty for
        /// `CROSS` joins (which have no ON clause) and for pure non-equi
        /// joins (whose residual predicate lives entirely in `filter`).
        on: Vec<(Expr, Expr)>,
        /// Optional residual non-equi predicate evaluated against the
        /// combined left ++ right schema. When `Some(_)`, the executor
        /// dispatches to the nested-loop fallback (see
        /// [`crate::exec::join`]). `None` is the equi-join / CROSS fast
        /// path.
        filter: Option<Expr>,
        /// Combined left ++ right schema with right-side collisions
        /// renamed and outer-side nullability widened; see
        /// [`join_combined_schema`].
        output_schema: Schema,
    },
}

impl PhysicalPlan {
    /// Output schema of the whole plan.
    ///
    /// For the row-shape-preserving wrappers (`Distinct`, `Limit`, `Sort`),
    /// the schema is recursively the input's. `Union` returns the first
    /// branch's schema (UNION ALL semantics; branch compatibility was
    /// verified at logical-plan time). `Join` returns its stored
    /// `output_schema` field — the concatenated left ++ right schema with
    /// right-side name collisions disambiguated; the same shape
    /// `LogicalPlan::Join::schema()` produces, computed once at lowering
    /// time so this accessor can keep the `&Schema` borrow-stable.
    pub fn output_schema(&self) -> &Schema {
        match self {
            PhysicalPlan::Projection { output_schema, .. } => output_schema,
            PhysicalPlan::Aggregate { aggregate, .. } => &aggregate.output_schema,
            PhysicalPlan::Distinct { input }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::Sort { input, .. }
            | PhysicalPlan::Filter { input, .. } => input.output_schema(),
            PhysicalPlan::Union { inputs } => {
                // Caller will have ensured at logical-plan time that all
                // branches share a schema; here we just return the first.
                //
                // The `Union { inputs: vec![] }` case is gated out at every
                // public construction site that feeds the physical planner:
                //   - `sql_frontend` never emits a zero-branch Union (the
                //     SQL grammar requires at least one `SELECT`).
                //   - `DataFrame::from_plan` runs `check_no_empty_union` and
                //     records a `BoltError::Plan` in `first_error`, which
                //     surfaces through `validation_error()` / `schema()`
                //     before any caller can hand the plan to the engine.
                //   - `lower()` itself re-rejects empty Union with
                //     `BoltError::Plan` (see the `LogicalPlan::Union` arm
                //     in this file), so a `PhysicalPlan::Union { inputs: [] }`
                //     cannot arise from lowering.
                //
                // The only remaining way to reach this branch with no inputs
                // is hand-constructing `PhysicalPlan::Union { inputs: vec![] }`
                // directly — a clearly malformed plan that this crate does
                // not produce. The `expect` documents that contract.
                inputs
                    .first()
                    .expect(
                        "PhysicalPlan::Union { inputs: vec![] } is malformed; \
                         construction sites (sql_frontend, DataFrame::from_plan, \
                         lower) all reject empty Union before this accessor is \
                         reached",
                    )
                    .output_schema()
            }
            PhysicalPlan::Project { output_schema, .. } => output_schema,
            PhysicalPlan::Join { output_schema, .. } => output_schema,
        }
    }
}

/// Promote two numeric types to the wider one (float beats int, 64 beats 32).
fn unify_numeric(a: DataType, b: DataType) -> BoltResult<DataType> {
    use DataType::*;
    match (a, b) {
        (x, y) if x == y => Ok(x),
        (Float64, _) | (_, Float64) => Ok(Float64),
        (Float32, Int64) | (Int64, Float32) => Ok(Float64),
        (Float32, _) | (_, Float32) => Ok(Float32),
        (Int64, _) | (_, Int64) => Ok(Int64),
        (Int32, _) | (_, Int32) => Ok(Int32),
        _ => Err(BoltError::Type(format!(
            "cannot unify {:?} and {:?}",
            a, b
        ))),
    }
}

/// Gate the source side of a CAST against the v0.7 GPU codegen surface.
///
/// Accepted source dtypes: `Bool`, `Int32`, `Int64`, `Float32`, `Float64`.
/// Everything else surfaces a `BoltError::Plan` whose message names the
/// specific offending category — "CAST to/from {Decimal128|Date32|
/// Timestamp|String} not yet lowered to GPU" — so callers see one
/// consistent message regardless of which type tripped the rejection.
/// The logical-plane `cast_is_supported` predicate accepts identity
/// casts for every primitive (including `Utf8 -> Utf8`); this guard
/// keeps such hand-built physical plans from sneaking into the kernel.
fn cast_source_is_supported(src: DataType) -> BoltResult<()> {
    match src {
        DataType::Bool
        | DataType::Int32
        | DataType::Int64
        | DataType::Float32
        | DataType::Float64 => Ok(()),
        DataType::Decimal128(_, _) => Err(BoltError::Plan(
            "CAST to/from Decimal128 not yet lowered to GPU".into(),
        )),
        DataType::Date32 => Err(BoltError::Plan(
            "CAST to/from Date32 not yet lowered to GPU".into(),
        )),
        DataType::Timestamp(_, _) => Err(BoltError::Plan(
            "CAST to/from Timestamp not yet lowered to GPU".into(),
        )),
        DataType::Utf8 => Err(BoltError::Plan(
            "CAST to/from String not yet lowered to GPU".into(),
        )),
    }
}

/// Mirror of [`cast_source_is_supported`] for the target dtype. Kept as
/// a separate predicate (rather than a single combined `(src, target)`
/// check) so the error message can name the offending side directly.
fn cast_target_is_supported(target: DataType) -> BoltResult<()> {
    match target {
        DataType::Bool
        | DataType::Int32
        | DataType::Int64
        | DataType::Float32
        | DataType::Float64 => Ok(()),
        DataType::Decimal128(_, _) => Err(BoltError::Plan(
            "CAST to/from Decimal128 not yet lowered to GPU".into(),
        )),
        DataType::Date32 => Err(BoltError::Plan(
            "CAST to/from Date32 not yet lowered to GPU".into(),
        )),
        DataType::Timestamp(_, _) => Err(BoltError::Plan(
            "CAST to/from Timestamp not yet lowered to GPU".into(),
        )),
        DataType::Utf8 => Err(BoltError::Plan(
            "CAST to/from String not yet lowered to GPU".into(),
        )),
    }
}

/// v0.7: result dtype for an arithmetic op on Date32 / Timestamp operands.
///
/// Mirrors the logical-plane helper of the same intent
/// (`date_or_timestamp_arith_result` in `logical_plan.rs`) but returns
/// errors and `Option<DataType>` instead of `Option<Result>` so the
/// codegen caller can fall through to `unify_numeric` for the
/// no-temporal case.
///
///   * `Sub, Date32, Date32`                          → `Some(Int32)`
///   * `Sub, Timestamp(u, tz), Timestamp(u, tz)`      → `Some(Int64)`
///   * `Add/Mul/Div on Date/Timestamp`                → `Err` with tight msg
///   * `Sub, Timestamp(u1, _), Timestamp(u2, _)` with `u1 != u2` → `Err`
///   * neither operand is Date/Timestamp                → `None` (fall through)
///
/// INTERVAL-day arithmetic on Date/Timestamp is intentionally not handled:
/// the SQL frontend has no INTERVAL expression literal yet, so there is no
/// path to produce a typed `Expr::Binary` with that shape.
fn temporal_arith_result_dtype(
    op: BinaryOp,
    l: DataType,
    r: DataType,
) -> BoltResult<Option<DataType>> {
    use DataType::*;
    let l_is_temporal = matches!(l, Date32 | Timestamp(_, _));
    let r_is_temporal = matches!(r, Date32 | Timestamp(_, _));
    if !l_is_temporal && !r_is_temporal {
        return Ok(None);
    }
    match (op, l, r) {
        (BinaryOp::Sub, Date32, Date32) => Ok(Some(Int32)),
        (BinaryOp::Sub, Timestamp(lu, ltz), Timestamp(ru, rtz)) => {
            if lu != ru {
                return Err(BoltError::Type(format!(
                    "Timestamp subtraction requires matching TimeUnit, \
                     got {lu:?} and {ru:?}"
                )));
            }
            if ltz != rtz {
                return Err(BoltError::Type(format!(
                    "Timestamp subtraction requires matching time zones, \
                     got {ltz:?} and {rtz:?}"
                )));
            }
            Ok(Some(Int64))
        }
        _ => Err(BoltError::Type(format!(
            "arithmetic {op:?} on Date/Timestamp operands ({l:?}, {r:?}) is not \
             supported; only Date32 - Date32 and Timestamp - Timestamp \
             (matching unit and tz) are wired in v0.7"
        ))),
    }
}

/// Returns the output column name for a projected expression at position `i`.
fn output_name_for(expr: &Expr, i: usize) -> String {
    match expr {
        Expr::Column(n) => n.clone(),
        Expr::Alias(_, n) => n.clone(),
        _ => format!("__expr_{i}"),
    }
}

/// True if every expression is a bare column reference (no alias, no compute).
fn all_bare_columns(exprs: &[Expr]) -> bool {
    exprs.iter().all(|e| matches!(e, Expr::Column(_)))
}

/// Emitter for a single kernel's IR.
struct Codegen<'a> {
    /// Schema of the underlying scan (column lookup).
    scan_schema: &'a Schema,
    /// Emitted ops.
    ops: Vec<Op>,
    /// Allocator for fresh registers.
    next_reg: u32,
    /// Input column metadata, in load order.
    inputs: Vec<ColumnIO>,
    /// Cache of already-loaded columns: name -> (input ordinal, Value).
    column_cache: HashMap<String, (usize, Value)>,
    /// Parallel to `inputs`: `true` if any emitted op (today only
    /// [`Op::IsNullCheck`]) needs the kernel to consume that column's
    /// validity bitmap at runtime. Propagated into
    /// [`KernelSpec::input_has_validity`] by [`Codegen::finish`].
    ///
    /// Plan-time validity tracking is OR-combined with the provider-side
    /// signal populated by [`populate_input_validity`]: a column that the
    /// codegen flags here stays flagged even if the provider returns
    /// `has_nulls == false`, since we still need the validity pointer
    /// wired through to satisfy the `IsNullCheck` op. See
    /// [`populate_one_kernel`] for the merge semantics.
    input_needs_validity: Vec<bool>,
}

impl<'a> Codegen<'a> {
    /// New empty emitter against `scan_schema`.
    fn new(scan_schema: &'a Schema) -> Self {
        Self {
            scan_schema,
            ops: Vec::new(),
            next_reg: 0,
            inputs: Vec::new(),
            column_cache: HashMap::new(),
            input_needs_validity: Vec::new(),
        }
    }

    /// Allocate a fresh register.
    fn fresh(&mut self) -> Reg {
        let r = Reg(self.next_reg);
        self.next_reg += 1;
        r
    }

    /// Emit ops for `e`, returning the produced value.
    fn emit_expr(&mut self, e: &Expr) -> BoltResult<Value> {
        match e {
            Expr::Column(name) => self.emit_column(name),
            Expr::Literal(lit) => self.emit_literal(lit),
            Expr::Binary { op, left, right } => self.emit_binary(*op, left, right),
            Expr::Unary { op, operand } => self.emit_unary(*op, operand),
            // GPU codegen does not yet lower LIKE — it requires Utf8 column
            // access which the kernel doesn't support. The lowering layer
            // (`lower_depth`) routes LIKE predicates through the host-side
            // `PhysicalPlan::Filter` path; this arm is unreachable for any
            // plan that came through `lower()`. Surface a clear error if it
            // ever gets here (e.g. via a hand-built physical plan).
            Expr::Like { .. } => Err(BoltError::Plan(
                "GPU codegen: LIKE requires host fallback".into(),
            )),
            // v0.7: numeric ↔ numeric (and Bool ↔ Int) CASTs lower to a
            // PTX `cvt.*` instruction via `emit_cast` (which already exists
            // for arithmetic dtype unification). Non-numeric source/target
            // (Decimal128 / Date32 / Timestamp / Utf8) are still rejected
            // here with a tightened message — see `cast_target_is_supported`.
            Expr::Cast { expr, target } => self.emit_cast_expr(expr, *target),
            // v0.5/v0.7 surface: parser + type-check land, GPU execution
            // wiring is blocked at the substrate level.
            //
            // The fused-projection GPU codegen consumes one device pointer
            // per `KernelSpec::inputs` slot (see `compile` in
            // `crate::jit::ptx_gen` and the `ColumnIO` doc — Utf8 inputs are
            // rejected eagerly). Strings on the device are dictionary-
            // encoded (`GpuColumnData::DictUtf8` in
            // `crate::exec::gpu_table`): only the integer index column ever
            // crosses the kernel boundary, and the dictionary itself lives
            // host-side. Every scalar string function in the v0.5 surface
            // needs at least one of:
            //
            //   * UPPER / LOWER / SUBSTRING — a new output-string-buffer
            //     allocation path (the kernel must produce a fresh
            //     `(values, offsets)` pair, or build a fresh dictionary).
            //     The IR has no Utf8 output op and the codegen has no way to
            //     size or allocate variable-width output buffers at launch
            //     time.
            //   * LENGTH — a way to read per-row Utf8 input from the kernel.
            //     With the dictionary-encoded layout this is a gather op
            //     ("look up byte length at dict index N"), which would need
            //     a sidecar lengths buffer threaded through `ColumnIO` plus
            //     a new `Op::GatherInt32` in the IR. Neither exists today.
            //
            // Until that substrate lands every variant is rejected here with
            // a per-kind message that names the concrete blocker, so users
            // get an actionable hint instead of a generic "follow-up" string.
            Expr::ScalarFn { kind, .. } => {
                let blocker = match kind {
                    crate::plan::logical_plan::ScalarFnKind::Upper
                    | crate::plan::logical_plan::ScalarFnKind::Lower
                    | crate::plan::logical_plan::ScalarFnKind::Substring => {
                        "no GPU output-string-buffer allocation in the scalar emitter"
                    }
                    crate::plan::logical_plan::ScalarFnKind::Length => {
                        "no GPU Utf8 input support — dictionary lengths sidecar / \
                         Op::GatherInt32 not yet wired"
                    }
                    crate::plan::logical_plan::ScalarFnKind::Concat => {
                        "GPU codegen has no Utf8 support (Concat routes through host fallback)"
                    }
                };
                Err(BoltError::Plan(format!(
                    "string scalar function {} is not yet lowered to GPU: {}; \
                     coming in a follow-up",
                    kind.sql_name(),
                    blocker
                )))
            }
            Expr::Alias(inner, _) => self.emit_expr(inner),
            // v0.7: CASE WHEN ... THEN ... [ELSE ...] END is lowered to a
            // right-to-left fold of `Op::Select` ops. See `Codegen::emit_case`
            // for the fold and the supported dtype envelope (numeric / Bool
            // only — Utf8 and the wider numeric variants are rejected with a
            // tighter message there).
            Expr::Case {
                branches,
                else_branch,
            } => self.emit_case(branches, else_branch.as_deref()),
        }
    }

    /// Emit a unary op.
    ///
    /// Today this only covers `IS NULL` / `IS NOT NULL`, and only when the
    /// operand is a bare column reference (optionally wrapped in `Alias`).
    /// Compound operands (e.g. `(x + y) IS NULL`) are surfaced as a
    /// `Plan` error so the caller routes the predicate through the host
    /// fallback — see [`predicate_contains_unary`] for the routing
    /// invariant.
    ///
    /// `UnaryOp::Not` is not yet lowered to the GPU at all; it is rejected
    /// here with a clear error so [`predicate_contains_unary`] can route
    /// every `NOT`-bearing predicate through the host-side filter path
    /// (which dispatches to `crate::exec::expr_agg::eval_unary`).
    ///
    /// The emitted IR shape:
    /// * Non-nullable input schema → `Op::Const { Bool(false_or_true) }`,
    ///   since the column can never be NULL at runtime. `IS NULL` collapses
    ///   to `false`, `IS NOT NULL` to `true`.
    /// * Nullable input schema → `Op::IsNullCheck` referencing the input
    ///   slot's validity bitmap. The codegen also flips
    ///   [`Codegen::input_needs_validity`] for that slot so the lowered
    ///   `KernelSpec::input_has_validity` will request the validity
    ///   pointer at kernel-launch time.
    fn emit_unary(&mut self, op: UnaryOp, operand: &Expr) -> BoltResult<Value> {
        // NOT is not yet lowered to GPU. Surface a Plan error so the planner
        // regression surfaces clearly if the route is ever miswired —
        // `predicate_contains_unary` routes every `NOT` to the host fallback.
        if matches!(op, UnaryOp::Not) {
            return Err(BoltError::Plan(
                "GPU codegen: NOT not yet lowered to GPU; requires host fallback".into(),
            ));
        }
        // Peel through any `Alias` wrappers so `x AS y IS NULL` lowers the
        // same as `x IS NULL`.
        let mut bare = operand;
        loop {
            match bare {
                Expr::Alias(inner, _) => bare = inner.as_ref(),
                _ => break,
            }
        }
        let col_name = match bare {
            Expr::Column(n) => n.as_str(),
            // Compound operand — caller must have routed through the host
            // fallback. Return a Plan error so the planner regression
            // surfaces clearly if the route is ever miswired.
            _ => {
                return Err(BoltError::Plan(format!(
                    "GPU codegen: {:?} on non-column operand requires host fallback",
                    op
                )));
            }
        };

        // Resolve the column in the scan schema. Unknown columns surface
        // as `BoltError::Schema` — same as `emit_column`.
        let field = self.scan_schema.field(col_name)?;
        let nullable = field.nullable;
        let want_null = matches!(op, UnaryOp::IsNull);

        if !nullable {
            // Non-nullable column: the answer is a constant. IS NULL → false,
            // IS NOT NULL → true. No validity pointer needed, no IsNullCheck.
            let lit = Literal::Bool(!want_null);
            let dst = self.fresh();
            self.ops.push(Op::Const { dst, lit });
            return Ok(Value {
                reg: dst,
                dtype: DataType::Bool,
            });
        }

        // Nullable column: route through the value-load path so the input
        // slot exists, then flip its validity flag. We reuse `emit_column`
        // (which is idempotent via `column_cache`) so a query that touches
        // the same column for both value and null-check ends up with a
        // single input slot.
        let value = self.emit_column(col_name)?;
        let col_idx = self
            .column_cache
            .get(col_name)
            .map(|(idx, _)| *idx)
            .ok_or_else(|| {
                BoltError::Other(format!(
                    "physical_plan: column '{}' missing from cache after emit_column",
                    col_name
                ))
            })?;
        let _ = value; // value reg unused — IsNullCheck only reads validity.

        // Flag this input as needing its validity pointer wired through.
        // `input_needs_validity` is parallel to `inputs`; `emit_column`
        // already extended both vectors (via the cache miss path).
        if col_idx < self.input_needs_validity.len() {
            self.input_needs_validity[col_idx] = true;
        }

        let dst = self.fresh();
        self.ops.push(Op::IsNullCheck {
            dst,
            validity_input: col_idx,
            want_null,
        });
        Ok(Value {
            reg: dst,
            dtype: DataType::Bool,
        })
    }

    /// Emit (or reuse) a column load.
    fn emit_column(&mut self, name: &str) -> BoltResult<Value> {
        if let Some((_, v)) = self.column_cache.get(name) {
            return Ok(*v);
        }
        let field = self.scan_schema.field(name)?;
        let dtype = field.dtype;
        // v0.6 / M4: Decimal128 is plan-level only. The GPU codegen has no
        // i128 register class or kernel atomics yet, so any attempt to lower
        // a Decimal column into a kernel is surfaced here as a clear
        // BoltError::Plan. The follow-up will fill in the actual codegen.
        if matches!(dtype, DataType::Decimal128(_, _)) {
            return Err(BoltError::Plan(format!(
                "Decimal128 not yet lowered to GPU; coming in a follow-up \
                 (column '{}' has dtype {:?})",
                name, dtype
            )));
        }
        // v0.7: Date32 and Timestamp lower to integer registers (Date32 as
        // i32 days-since-epoch, Timestamp as i64 ticks in the source unit).
        // The codegen treats them as their underlying integer type at the
        // PTX layer, but preserves the logical dtype on the `Value` so
        // downstream type-checks (e.g. binary subtraction yielding a plain
        // Int32/Int64) see the temporal type and reject mixed-type ops.
        let col_idx = self.inputs.len();
        self.inputs.push(ColumnIO {
            name: name.to_string(),
            dtype,
        });
        // Keep `input_needs_validity` parallel to `inputs`. Default is
        // `false` — `emit_unary` flips the slot to `true` if any
        // `Op::IsNullCheck` reads this column's validity bitmap.
        self.input_needs_validity.push(false);
        let dst = self.fresh();
        self.ops.push(Op::LoadColumn { dst, col_idx, dtype });
        let value = Value { reg: dst, dtype };
        self.column_cache
            .insert(name.to_string(), (col_idx, value));
        Ok(value)
    }

    /// Emit a constant literal load.
    fn emit_literal(&mut self, lit: &Literal) -> BoltResult<Value> {
        let dtype = lit
            .dtype()
            .ok_or_else(|| BoltError::Type("untyped NULL literal".into()))?;
        // v0.6 / M4: Decimal128 literals are plan-level only; reject before
        // the codegen would attempt to allocate an i128 register class it
        // doesn't have. Mirrors the column-load rejection in `emit_column`.
        if matches!(dtype, DataType::Decimal128(_, _)) {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up \
                 (Decimal128 literal in expression)"
                    .into(),
            ));
        }
        // v0.7: Date32 / Timestamp literals lower to integer constants on
        // the GPU side; ptx_gen emits the underlying i32/i64 bit pattern.
        let dst = self.fresh();
        self.ops.push(Op::Const {
            dst,
            lit: lit.clone(),
        });
        Ok(Value { reg: dst, dtype })
    }

    /// Emit a `Const { Literal::Null }` op whose register carries `dtype`.
    ///
    /// Used by the NULL-peer-typing rule in `emit_binary`: when one operand
    /// of a binary expression is an untyped `Literal::Null` and the other
    /// is a typed expression, we type the NULL as the peer's dtype so the
    /// downstream cast/comparison logic can treat the two operands
    /// uniformly. The `Op::Const { Literal::Null }` op still flows through
    /// codegen verbatim; only the surrounding `Value`'s dtype is borrowed
    /// from the peer.
    fn emit_null_as(&mut self, dtype: DataType) -> Value {
        let dst = self.fresh();
        self.ops.push(Op::Const {
            dst,
            lit: Literal::Null,
        });
        Value { reg: dst, dtype }
    }

    /// Insert a Cast from `value` to `to`, returning the cast value.
    fn emit_cast(&mut self, value: Value, to: DataType) -> Value {
        if value.dtype == to {
            return value;
        }
        let dst = self.fresh();
        self.ops.push(Op::Cast {
            dst,
            src: value.reg,
            from: value.dtype,
            to,
        });
        Value { reg: dst, dtype: to }
    }

    /// Lower a SQL `CAST(<inner> AS <target>)` expression for the GPU
    /// codegen path.
    ///
    /// Numeric ↔ numeric (Int32 / Int64 / Float32 / Float64) plus
    /// Bool ↔ Int conversions are lowered to a single PTX `cvt.*`
    /// instruction by `emit_cast` (which already exists for binary-op
    /// dtype unification). The accepted set is exactly what
    /// [`crate::plan::logical_plan::cast_is_supported`] admits MINUS the
    /// non-numeric / non-bool types this codegen still can't handle —
    /// any `Decimal128` / `Date32` / `Timestamp` / `Utf8` involvement is
    /// rejected here with a clear error so the planner regression is
    /// obvious. The non-numeric types are also rejected upstream by
    /// `emit_column` / `emit_literal`, but we keep this guard so a
    /// hand-built physical plan can't sneak past the policy.
    ///
    /// `Literal::Null` source is special-cased the same way it is for
    /// `IS [NOT] NULL` and binary ops — the result is a typed NULL at
    /// `target`. The runtime semantics of NULL through `cvt.*` are
    /// undefined; the executor's per-row validity bitmap masks the result
    /// out before any consumer can observe it.
    fn emit_cast_expr(&mut self, inner: &Expr, target: DataType) -> BoltResult<Value> {
        // CAST(NULL AS T) is admitted at the logical plane (see
        // `Expr::dtype_depth`); mirror that here by emitting a typed NULL
        // constant directly rather than recursing into the bare-null literal
        // and hitting "untyped NULL literal" in `emit_literal`.
        if matches!(inner, Expr::Literal(Literal::Null)) {
            cast_target_is_supported(target)?;
            return Ok(self.emit_null_as(target));
        }
        let value = self.emit_expr(inner)?;
        // Identity cast is always fine — emit_cast collapses it to no-op.
        if value.dtype == target {
            return Ok(self.emit_cast(value, target));
        }
        cast_source_is_supported(value.dtype)?;
        cast_target_is_supported(target)?;
        Ok(self.emit_cast(value, target))
    }

    /// Emit a binary op, inserting casts and computing the result dtype.
    ///
    /// NULL-peer-typing: if one operand is `Literal::Null` and the other is
    /// a typed expression, we pre-resolve the peer's dtype against the scan
    /// schema and emit the NULL operand's `Const` op carrying the peer's
    /// dtype (rather than failing in `emit_literal` with
    /// `untyped NULL literal`). The runtime semantics of NULL in
    /// comparisons / arithmetic are left to the kernel — see
    /// `crate::jit::ptx_gen` for current behaviour.
    fn emit_binary(&mut self, op: BinaryOp, left: &Expr, right: &Expr) -> BoltResult<Value> {
        let left_is_null = matches!(left, Expr::Literal(Literal::Null));
        let right_is_null = matches!(right, Expr::Literal(Literal::Null));
        // If exactly one side is an untyped NULL, type it as the peer's dtype
        // (resolved against the scan schema). Two NULLs fall through and will
        // surface the legacy "untyped NULL literal" error from `emit_literal`.
        let (l, r) = if left_is_null && !right_is_null {
            let r = self.emit_expr(right)?;
            let l = self.emit_null_as(r.dtype);
            (l, r)
        } else if right_is_null && !left_is_null {
            let l = self.emit_expr(left)?;
            let r = self.emit_null_as(l.dtype);
            (l, r)
        } else {
            let l = self.emit_expr(left)?;
            let r = self.emit_expr(right)?;
            (l, r)
        };

        let (lhs_v, rhs_v, operand_dtype, result_dtype) = if op_is_arithmetic(op) {
            // v0.7: Date32 / Timestamp arithmetic. Only `Date32 - Date32`
            // (→ Int32) and `Timestamp(u, tz) - Timestamp(u, tz)` (→ Int64
            // in the source unit) are wired. The operand_dtype handed to
            // ptx_gen is still the temporal type so the PTX layer knows
            // which integer width to use; the codegen path for Sub on
            // Date32 / Timestamp emits the same `sub.s32` / `sub.s64`
            // mnemonic it would for the underlying integer type. The
            // result_dtype is the plain integer because the difference is
            // a unit-less count of days / ticks, not a calendar value.
            if let Some(out) = temporal_arith_result_dtype(op, l.dtype, r.dtype)? {
                (l, r, l.dtype, out)
            } else {
                let unified = unify_numeric(l.dtype, r.dtype)?;
                let lv = self.emit_cast(l, unified);
                let rv = self.emit_cast(r, unified);
                (lv, rv, unified, unified)
            }
        } else if op_is_comparison(op) {
            if l.dtype == r.dtype {
                (l, r, l.dtype, DataType::Bool)
            } else {
                // Match logical_plan's behavior: numeric cross-compare unifies, result Bool.
                let unified = unify_numeric(l.dtype, r.dtype)?;
                let lv = self.emit_cast(l, unified);
                let rv = self.emit_cast(r, unified);
                (lv, rv, unified, DataType::Bool)
            }
        } else if op_is_logical(op) {
            // SHORT-CIRCUIT SEMANTICS — KNOWN DIVERGENCE FROM SQL STANDARD
            //
            // craton-bolt's GPU kernels evaluate both operands of AND/OR eagerly. SQL
            // semantics require short-circuit: `WHERE b<>0 AND a/b>5` must NOT evaluate
            // `a/b` when `b=0`. Predicates that rely on short-circuit evaluation for
            // safety (divide-by-zero, NULL-poisoning, etc.) can produce wrong results.
            //
            // TODO(short-circuit): emit masked second-operand evaluation in JIT.
            // Until that lands, the engine emits a warning when a query plan contains
            // a divide / modulo nested under AND/OR (see
            // `warn_if_eager_shortcircuit_unsafe`, invoked from `lower`).
            if l.dtype != DataType::Bool || r.dtype != DataType::Bool {
                return Err(BoltError::Type(format!(
                    "logical {op:?} requires Bool operands, got {:?} and {:?}",
                    l.dtype, r.dtype
                )));
            }
            (l, r, DataType::Bool, DataType::Bool)
        } else {
            return Err(BoltError::Type(format!("unsupported operator {op:?}")));
        };

        let dst = self.fresh();
        self.ops.push(Op::Binary {
            dst,
            op,
            lhs: lhs_v.reg,
            rhs: rhs_v.reg,
            dtype: operand_dtype,
            result_dtype,
        });
        Ok(Value {
            reg: dst,
            dtype: result_dtype,
        })
    }

    /// Lower a `CASE WHEN c1 THEN v1 [WHEN c2 THEN v2 ...] [ELSE e] END`
    /// expression to a right-to-left fold of `Op::Select` ops.
    ///
    /// Pipeline:
    ///
    /// 1. Resolve the unified result dtype from the logical-plane's CASE
    ///    type-check (it already enforced that every THEN/ELSE arm shares a
    ///    unifiable dtype). We re-derive it here by asking `Expr::dtype`
    ///    against the scan schema so the codegen knows what dtype to allocate
    ///    for the result register.
    /// 2. v0.7 envelope: reject Utf8 / Decimal128 / Date / Timestamp result
    ///    dtypes with a tighter, GPU-codegen-specific message. The PTX
    ///    `selp.*` mnemonic doesn't cover those classes, and string-typed
    ///    CASE in particular needs a heap-aware ABI we don't have yet.
    /// 3. Emit the ELSE arm first, cast to result dtype. If ELSE is omitted,
    ///    materialise a zero literal of the result dtype as the "no WHEN
    ///    fired" sentinel. SQL semantics call for SQL NULL in that case;
    ///    full per-row validity propagation through the Select op is a
    ///    v0.7 follow-up, so the v0.7 envelope settles for a deterministic
    ///    zero on the missed-every-WHEN path.
    /// 4. Fold backwards over the branches: `cur = Select(cond_i, then_i, cur)`.
    ///    Right-to-left iteration mirrors SQL's left-to-right WHEN priority:
    ///    earlier WHENs sit closer to the top of the chain and win when their
    ///    condition fires.
    ///
    /// NULL propagation: an `Op::Const { Literal::Null }` register is allowed
    /// through `Op::Select` just like any other typed value — the kernel
    /// stores the bit-pattern unchanged. Per-row NULL propagation through
    /// CASE branches is a follow-up; v0.7 emits the value with no
    /// per-output validity AND-fold beyond what the input loads already
    /// contribute.
    fn emit_case(
        &mut self,
        branches: &[(Expr, Expr)],
        else_branch: Option<&Expr>,
    ) -> BoltResult<Value> {
        if branches.is_empty() {
            // Type-checker already rejects this at the logical plane; mirror
            // the error here so a hand-built plan that bypasses `dtype()`
            // surfaces it consistently.
            return Err(BoltError::Plan(
                "GPU codegen: CASE requires at least one WHEN/THEN branch".into(),
            ));
        }

        // (1) Pin the unified result dtype against the scan schema. Any
        //     dtype error here (incompatible arms etc.) is the same error
        //     `lower_depth` would surface from a later `plan.schema()` call;
        //     we just trigger it earlier so codegen has a concrete dtype.
        let case_expr = Expr::Case {
            branches: branches.to_vec(),
            else_branch: else_branch.map(|e| Box::new(e.clone())),
        };
        let result_dtype = case_expr.dtype(self.scan_schema)?;

        // (2) v0.7 dtype envelope. PTX `selp` only supports the b32/b64
        //     register classes (Bool/Int32/Int64/Float32/Float64). Utf8
        //     CASE is a heap-aware ABI we don't have; Decimal128 and
        //     Date/Timestamp share the same "not yet lowered to GPU"
        //     story as the other scalar code paths.
        match result_dtype {
            DataType::Utf8 => {
                return Err(BoltError::Plan(
                    "CASE over string (Utf8) types not yet lowered to GPU; \
                     coming in a follow-up"
                        .into(),
                ))
            }
            DataType::Decimal128(_, _) => {
                return Err(BoltError::Plan(
                    "CASE over Decimal128 types not yet lowered to GPU; \
                     coming in a follow-up"
                        .into(),
                ))
            }
            DataType::Date32 | DataType::Timestamp(_, _) => {
                return Err(BoltError::Plan(
                    "CASE over Date/Timestamp types not yet lowered to GPU; \
                     coming in a follow-up"
                        .into(),
                ))
            }
            DataType::Bool
            | DataType::Int32
            | DataType::Int64
            | DataType::Float32
            | DataType::Float64 => {}
        }

        // (3) ELSE seed. If ELSE is omitted, materialise a zero literal of
        //     the result dtype as the "missed every WHEN" sentinel.
        //     Validity-aware SQL-NULL propagation through CASE is a v0.7
        //     follow-up; the SQL frontend's COALESCE / NULLIF desugar
        //     always supplies an explicit ELSE so this branch is dead for
        //     those common entry points.
        let mut cur = match else_branch {
            Some(e) => {
                // NULL-arm typing (mirrors `emit_binary`'s NULL-peer rule):
                // a bare `Literal::Null` arm carries no static type. Type it
                // as the unified result dtype so the downstream Select op
                // sees a register typed compatibly with the other arms. The
                // current PTX emitter still rejects a `Literal::Null` Const,
                // so an ELSE-only-NULL CASE remains best-effort — but
                // NULLIF's `CASE WHEN a = b THEN NULL ELSE a END` always
                // populates ELSE with `a`, so this arm is exercised in
                // practice for typed (non-NULL) ELSEs.
                let v = if matches!(e, Expr::Literal(Literal::Null)) {
                    self.emit_null_as(result_dtype)
                } else {
                    self.emit_expr(e)?
                };
                self.emit_cast(v, result_dtype)
            }
            None => {
                // Synthesise a zero literal of the result dtype as the
                // "missed every WHEN" sentinel. Validity-aware NULL
                // propagation through CASE is a v0.7 follow-up.
                let lit = match result_dtype {
                    DataType::Bool => Literal::Bool(false),
                    DataType::Int32 => Literal::Int32(0),
                    DataType::Int64 => Literal::Int64(0),
                    DataType::Float32 => Literal::Float32(0.0),
                    DataType::Float64 => Literal::Float64(0.0),
                    // The envelope check above already rejected every
                    // non-numeric/non-Bool dtype.
                    other => {
                        return Err(BoltError::Other(format!(
                            "physical_plan: CASE without ELSE: unexpected \
                             result dtype {other:?} survived the envelope check"
                        )))
                    }
                };
                self.emit_literal(&lit)?
            }
        };

        // (4) Right-to-left fold over the WHEN branches. For each (cond,
        //     then) pair we evaluate `cond`, cast `then` to the result dtype,
        //     and emit a Select that picks `then` when `cond` is true and
        //     `cur` otherwise. Iterating in reverse gives earlier WHENs
        //     higher priority — `Select(c1, v1, Select(c2, v2, else))`
        //     evaluates c1 first and short-circuits visually even though
        //     the kernel evaluates both arms eagerly (the same eager-arm
        //     caveat that applies to AND/OR — see the SHORT-CIRCUIT
        //     SEMANTICS note in `emit_binary`).
        for (cond_expr, then_expr) in branches.iter().rev() {
            let cond_v = self.emit_expr(cond_expr)?;
            if cond_v.dtype != DataType::Bool {
                return Err(BoltError::Type(format!(
                    "CASE WHEN condition must be Bool, got {:?}",
                    cond_v.dtype
                )));
            }
            // NULL-arm typing for THEN: same rule as the ELSE arm above. A
            // bare `Literal::Null` THEN inherits the result dtype so the
            // Op::Cast / Op::Select stack sees a typed register.
            let then_v = if matches!(then_expr, Expr::Literal(Literal::Null)) {
                self.emit_null_as(result_dtype)
            } else {
                self.emit_expr(then_expr)?
            };
            let then_cast = self.emit_cast(then_v, result_dtype);
            let dst = self.fresh();
            self.ops.push(Op::Select {
                dst,
                cond: cond_v.reg,
                then_val: then_cast.reg,
                else_val: cur.reg,
                dtype: result_dtype,
            });
            cur = Value {
                reg: dst,
                dtype: result_dtype,
            };
        }
        Ok(cur)
    }

    /// Append a Store op for column `col_idx`.
    fn emit_store(&mut self, value: Value, col_idx: usize) {
        self.ops.push(Op::Store {
            src: value.reg,
            col_idx,
            dtype: value.dtype,
        });
    }

    /// Finalize into a `KernelSpec`.
    ///
    /// `input_has_validity` is left empty by default; callers that have
    /// already consulted a [`crate::plan::sql_frontend::TableProvider`] for
    /// the underlying table can populate the per-input flags afterwards via
    /// [`KernelSpec::input_has_validity`]. An empty vector is treated as
    /// "no input carries validity", preserving the pre-stage-D codegen
    /// shape for callers that don't yet wire the provider through.
    fn finish(self, outputs: Vec<ColumnIO>, predicate: Option<Reg>) -> KernelSpec {
        // PV-stage-g: surface plan-time validity needs.
        //
        // If any `Op::IsNullCheck` was emitted, `input_needs_validity` has
        // at least one `true` entry parallel to `inputs`. Propagate the
        // whole vector into `KernelSpec::input_has_validity` so the
        // launch-time wiring knows to pass through the matching `*u8`
        // validity pointer. Otherwise (no IsNullCheck anywhere) keep the
        // legacy empty-vector shape — every existing caller / PTX golden
        // test continues to see the historical no-validity layout.
        //
        // The provider-side signal populated by
        // [`populate_input_validity`] is OR-merged into this vector
        // afterwards (see [`populate_one_kernel`]), so a column that the
        // codegen flags here stays flagged even if the provider returns
        // `has_nulls == false`.
        let input_has_validity = if self.input_needs_validity.iter().any(|b| *b) {
            self.input_needs_validity
        } else {
            Vec::new()
        };
        KernelSpec {
            inputs: self.inputs,
            outputs,
            ops: self.ops,
            predicate,
            register_count: self.next_reg,
            // Validity propagation (Option B) is opt-in: callers populate
            // these via `populate_input_validity` or per-stage upload
            // helpers. The default codegen path emits the historical PTX
            // shape unchanged.
            input_has_validity,
            output_has_validity: Vec::new(),
        }
    }
}

/// True for `+ - * /`.
fn op_is_arithmetic(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
    )
}

/// True for `= <> < <= > >=`.
fn op_is_comparison(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
    )
}

/// True for `AND OR`.
fn op_is_logical(op: BinaryOp) -> bool {
    matches!(op, BinaryOp::And | BinaryOp::Or)
}

/// Result of folding a chain of Scan/Filter/Project nodes down to a single
/// scan + predicate + (optional) explicit projection list.
struct ResolvedSource<'a> {
    /// Source table name.
    table: &'a str,
    /// Underlying scan schema (column lookup namespace for all exprs below).
    scan_schema: &'a Schema,
    /// Combined predicate, AND-folded, expressed in scan namespace.
    predicate: Option<Expr>,
    /// If the outermost layer was a `Project`, its output list as
    /// `(output_name, scan-namespace expr)`. `None` means no top projection
    /// was present, so callers should default to "all scan columns".
    projection: Option<Vec<(String, Expr)>>,
}

/// Approach (decision note): the `Codegen` emitter is hard-wired to a single
/// underlying scan schema (every `LoadColumn` references a scan column
/// ordinal). Composing it across multiple Project layers would require a
/// larger refactor. Instead we fold `Project`/`Filter` chains by *expression
/// substitution*: walk the chain top-down, then iteratively rewrite every
/// outer `Column(name)` reference into the inner `Project`'s expression for
/// that column. The substituted expression tree then resolves entirely
/// against the underlying scan schema, which is what `Codegen` already
/// supports. This collapses any `Project(Filter(Project(... Scan)))` shape
/// into a single equivalent (scan, AND-folded predicate, projection list).
/// Because the lowerer handles arbitrary chains now, no defensive guard in
/// `DataFrame::select`/`filter` is needed.
fn resolve_source<'a>(
    plan: &'a LogicalPlan,
) -> BoltResult<ResolvedSource<'a>> {
    // Walk down from the outermost node toward the underlying `Scan`,
    // collecting each layer. We delay all substitution to the end so each
    // node's expressions stay in their *own* input namespace until we know
    // the full chain. At resolution time, every layer's input namespace is
    // the output namespace of the next-deeper layer (or the scan), so we
    // can iteratively rewrite from innermost upward.
    enum Layer {
        // A `Filter` whose predicate lives in its input layer's namespace.
        Filter(Expr),
        // A `Project` whose `exprs` live in its input layer's namespace,
        // producing the named outputs (in the given order).
        Project(Vec<(String, Expr)>),
    }

    let mut cur = plan;
    let mut layers: Vec<Layer> = Vec::new();
    // Position of the *outermost* Project encountered (== index in `layers`).
    // The outermost Project defines the chain's effective output schema; any
    // Filters above it preserve the schema and just AND into the predicate.
    let mut outermost_project_idx: Option<usize> = None;
    // The walk is iterative, so we can't blow the stack here — but an
    // attacker-controlled deeply nested Filter/Project chain would still
    // force us to allocate one `Layer` per node before the eventual error,
    // and the substitution loop below would then take O(depth^2) time.
    // Cap the chain length at MAX_RECURSION_DEPTH so we surface a clear
    // error long before either pathology shows up. The cap is generous —
    // realistic plan chains land in single digits.
    let mut steps = 0usize;

    let (table, scan_schema) = loop {
        if steps > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
            return Err(BoltError::Plan(format!(
                "plan nesting exceeds depth limit ({})",
                crate::plan::sql_frontend::MAX_RECURSION_DEPTH
            )));
        }
        steps += 1;
        match cur {
            LogicalPlan::Scan { table, schema, .. } => {
                break (table.as_str(), schema);
            }
            LogicalPlan::Filter { input, predicate } => {
                layers.push(Layer::Filter(predicate.clone()));
                cur = input.as_ref();
            }
            LogicalPlan::Project { input, exprs } => {
                let mut named: Vec<(String, Expr)> = Vec::with_capacity(exprs.len());
                for (i, e) in exprs.iter().enumerate() {
                    let name = output_name_for(e, i);
                    // Strip the outer Alias — it only affects output naming,
                    // not substitution semantics. Inner Aliases are left alone.
                    let body = match e {
                        Expr::Alias(inner, _) => (**inner).clone(),
                        _ => e.clone(),
                    };
                    named.push((name, body));
                }
                if outermost_project_idx.is_none() {
                    outermost_project_idx = Some(layers.len());
                }
                layers.push(Layer::Project(named));
                cur = input.as_ref();
            }
            other => {
                return Err(BoltError::Plan(format!(
                    "unsupported plan shape: expected Scan/Filter/Project chain, got {}",
                    shape(other)
                )));
            }
        }
    };

    // `layers` is ordered outermost-first; the innermost (closest to the
    // scan) is at the end. Walk innermost-to-outermost, maintaining the
    // current "name -> expression-in-scan-namespace" map (which represents
    // the output of the layer we just processed) and the accumulated
    // predicates (already in scan namespace).
    //
    // Initial state: just-above-scan output namespace == scan schema, so
    // every column resolves to itself (no entry needed — `substitute_one`
    // leaves unknown columns alone, which means "look it up in the scan").
    let mut name_map: HashMap<String, Expr> = HashMap::new();
    let mut name_map_active = false;
    let mut predicates_scan_ns: Vec<Expr> = Vec::new();
    // The outermost Project's output list, captured (in original order, with
    // exprs lowered to scan namespace) when we process that layer.
    let mut top_projection: Option<Vec<(String, Expr)>> = None;

    // Indices iterate from innermost (largest) to outermost (smallest).
    for (rev_i, layer) in layers.into_iter().enumerate().rev() {
        match layer {
            Layer::Filter(pred) => {
                // The predicate is in the current layer's input namespace
                // (== output of whatever's directly below). Rewrite into
                // scan namespace using `name_map` if a Project sits below us.
                let lowered = if name_map_active {
                    substitute_one(&pred, &name_map)
                } else {
                    pred
                };
                predicates_scan_ns.push(lowered);
            }
            Layer::Project(named) => {
                // Each Project replaces the output namespace. Its `exprs`
                // are in the *current* (below-it) namespace, so rewrite
                // each through `name_map` first, then install the new map.
                let mut next: HashMap<String, Expr> = HashMap::new();
                let mut named_lowered: Vec<(String, Expr)> = Vec::with_capacity(named.len());
                for (name, body) in named {
                    let lowered = if name_map_active {
                        substitute_one(&body, &name_map)
                    } else {
                        body
                    };
                    next.insert(name.clone(), lowered.clone());
                    named_lowered.push((name, lowered));
                }
                name_map = next;
                name_map_active = true;
                // If this is the outermost Project, capture its output list.
                if Some(rev_i) == outermost_project_idx {
                    top_projection = Some(named_lowered);
                }
            }
        }
    }

    Ok(ResolvedSource {
        table,
        scan_schema,
        predicate: and_all(predicates_scan_ns),
        projection: top_projection,
    })
}

/// Substitute `Column(name)` references in `expr` using `map`; leave unknown
/// columns alone (they pass through to a deeper layer or to the scan).
///
/// Thin wrapper that starts the depth counter at zero; the real recursion
/// lives in [`substitute_one_depth`], which enforces
/// [`crate::plan::sql_frontend::MAX_RECURSION_DEPTH`] as a defense-in-depth
/// guard against deeply nested attacker-controlled expressions reaching the
/// substitution pass.
fn substitute_one(expr: &Expr, map: &HashMap<String, Expr>) -> Expr {
    substitute_one_depth(expr, map, 0)
}

/// Inner recursion for [`substitute_one`]. `depth` is the current recursion
/// depth; when it exceeds [`crate::plan::sql_frontend::MAX_RECURSION_DEPTH`]
/// we stop recursing and return the sub-expression unchanged. The public
/// `substitute_one` does not return a `Result`, so we cannot surface the
/// overflow — but in practice the input `Expr` is itself produced by the
/// depth-bounded `lower_expr`, so the ceiling is only hit for inputs
/// constructed programmatically (DataFrame builder, tests, malicious calls
/// through public APIs we don't yet bound). Leaving the sub-tree
/// unsubstituted is sound: any unmatched `Column(name)` simply resolves
/// against the scan namespace, which is the existing fallback for unknown
/// columns.
fn substitute_one_depth(
    expr: &Expr,
    map: &HashMap<String, Expr>,
    depth: usize,
) -> Expr {
    if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
        return expr.clone();
    }
    match expr {
        Expr::Column(name) => match map.get(name) {
            Some(replacement) => replacement.clone(),
            None => expr.clone(),
        },
        Expr::Literal(_) => expr.clone(),
        Expr::Binary { op, left, right } => Expr::Binary {
            op: *op,
            left: Box::new(substitute_one_depth(left, map, depth + 1)),
            right: Box::new(substitute_one_depth(right, map, depth + 1)),
        },
        Expr::Unary { op, operand } => Expr::Unary {
            op: *op,
            operand: Box::new(substitute_one(operand, map)),
        },
        Expr::Case {
            branches,
            else_branch,
        } => Expr::Case {
            branches: branches
                .iter()
                .map(|(w, t)| {
                    (
                        substitute_one_depth(w, map, depth + 1),
                        substitute_one_depth(t, map, depth + 1),
                    )
                })
                .collect(),
            else_branch: else_branch
                .as_deref()
                .map(|e| Box::new(substitute_one_depth(e, map, depth + 1))),
        },
        Expr::Like {
            expr,
            pattern,
            escape,
            negated,
        } => Expr::Like {
            expr: Box::new(substitute_one_depth(expr, map, depth + 1)),
            pattern: pattern.clone(),
            escape: *escape,
            negated: *negated,
        },
        Expr::Cast { expr, target } => Expr::Cast {
            expr: Box::new(substitute_one(expr, map)),
            target: *target,
        },
        Expr::ScalarFn { kind, args } => Expr::ScalarFn {
            kind: *kind,
            args: args
                .iter()
                .map(|a| substitute_one_depth(a, map, depth + 1))
                .collect(),
        },
        Expr::Alias(inner, name) => {
            Expr::Alias(Box::new(substitute_one(inner, map)), name.clone())
        }
    }
}

/// AND-fold a list of predicates into a single optional expression.
fn and_all(mut preds: Vec<Expr>) -> Option<Expr> {
    if preds.is_empty() {
        return None;
    }
    let mut acc = preds.remove(0);
    for p in preds {
        acc = Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(acc),
            right: Box::new(p),
        };
    }
    Some(acc)
}

/// Short tag describing a plan node's variant for error messages.
fn shape(plan: &LogicalPlan) -> &'static str {
    match plan {
        LogicalPlan::Scan { .. } => "Scan",
        LogicalPlan::Filter { .. } => "Filter",
        LogicalPlan::Project { .. } => "Project",
        LogicalPlan::Aggregate { .. } => "Aggregate",
        LogicalPlan::Distinct { .. } => "Distinct",
        LogicalPlan::Limit { .. } => "Limit",
        LogicalPlan::Sort { .. } => "Sort",
        LogicalPlan::Union { .. } => "Union",
        LogicalPlan::Join { .. } => "Join",
    }
}

/// Build a projection kernel over `input`, producing `exprs` (or all input columns).
/// Pure codegen helper: given a scan, optional predicate, and a list of
/// `(output_name, scan-namespace expr)`, build a `Projection` physical
/// plan. No chain resolution — all exprs must already reference scan
/// columns. Both `lower_projection` and `lower_aggregate` funnel through
/// this after they've folded any Project chain.
fn build_projection_kernel(
    table: &str,
    scan_schema: &Schema,
    predicate: Option<&Expr>,
    projected: &[(String, Expr)],
) -> BoltResult<PhysicalPlan> {
    let mut cg = Codegen::new(scan_schema);

    let predicate_reg = if let Some(pred) = predicate {
        let v = cg.emit_expr(pred)?;
        if v.dtype != DataType::Bool {
            return Err(BoltError::Type(format!(
                "filter predicate must be Bool, got {:?}",
                v.dtype
            )));
        }
        Some(v.reg)
    } else {
        None
    };

    let mut outputs = Vec::with_capacity(projected.len());
    let mut output_fields = Vec::with_capacity(projected.len());
    for (i, (name, expr)) in projected.iter().enumerate() {
        let value = cg.emit_expr(expr)?;
        cg.emit_store(value, i);
        outputs.push(ColumnIO {
            name: name.clone(),
            dtype: value.dtype,
        });
        output_fields.push(Field::new(name.clone(), value.dtype, true));
    }

    let kernel = cg.finish(outputs, predicate_reg);
    Ok(PhysicalPlan::Projection {
        table: table.to_string(),
        kernel,
        output_schema: Schema::new(output_fields),
    })
}

fn lower_projection(
    input: &LogicalPlan,
    exprs: Option<&[Expr]>,
    extra_predicate: Option<&Expr>,
) -> BoltResult<PhysicalPlan> {
    let resolved = resolve_source(input)?;
    let ResolvedSource {
        table,
        scan_schema,
        predicate: chain_predicate,
        projection: chain_projection,
    } = resolved;

    // Any `extra_predicate` lives in `input`'s output namespace; if a chain
    // Project sits at the top of `input`, rewrite the predicate through that
    // Project's output map so it ends up in scan namespace.
    let mut chain_proj_map: Option<HashMap<String, Expr>> = chain_projection
        .as_ref()
        .map(|named| named.iter().cloned().collect());

    let extra_pred_lowered: Option<Expr> = match extra_predicate {
        Some(p) => Some(match &chain_proj_map {
            Some(m) => substitute_one(p, m),
            None => p.clone(),
        }),
        None => None,
    };

    // Combine chain predicate AND extra predicate.
    let predicate = match (chain_predicate, extra_pred_lowered) {
        (None, None) => None,
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (Some(a), Some(b)) => Some(Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(a),
            right: Box::new(b),
        }),
    };

    // Determine the projection list, in scan namespace, with output names.
    // Priority order:
    //   1. Explicit `exprs` argument (from a top-level Project on `input`),
    //      substituted through any chain Project map.
    //   2. The chain's own top Project, if any.
    //   3. Default: all scan columns as bare references.
    let owned_default: Vec<(String, Expr)>;
    let projected: &[(String, Expr)] = if let Some(es) = exprs {
        let mut subbed: Vec<(String, Expr)> = Vec::with_capacity(es.len());
        for (i, e) in es.iter().enumerate() {
            let name = output_name_for(e, i);
            let body = match e {
                Expr::Alias(inner, _) => (**inner).clone(),
                _ => e.clone(),
            };
            let lowered = match chain_proj_map.as_mut() {
                Some(m) => substitute_one(&body, m),
                None => body,
            };
            subbed.push((name, lowered));
        }
        owned_default = subbed;
        &owned_default
    } else if let Some(named) = chain_projection.as_ref() {
        named.as_slice()
    } else {
        owned_default = scan_schema
            .fields
            .iter()
            .map(|f| (f.name.clone(), Expr::Column(f.name.clone())))
            .collect();
        &owned_default
    };

    build_projection_kernel(table, scan_schema, predicate.as_ref(), projected)
}

/// Build an aggregate plan over `input` with the given group keys and aggregates.
fn lower_aggregate(
    plan: &LogicalPlan,
    input: &LogicalPlan,
    group_by: &[Expr],
    aggregates: &[AggregateExpr],
) -> BoltResult<PhysicalPlan> {
    let resolved = resolve_source(input)?;
    let table = resolved.table;
    let scan_schema = resolved.scan_schema;
    let scan_predicate = resolved.predicate;
    let chain_proj_map: Option<HashMap<String, Expr>> = resolved
        .projection
        .as_ref()
        .map(|named| named.iter().cloned().collect());
    let output_schema = plan.schema()?;

    // Collect the expressions that feed the aggregation: group keys first, then per-aggregate input.
    let mut agg_input_exprs: Vec<Expr> = Vec::with_capacity(aggregates.len());
    for agg in aggregates {
        let e = match agg {
            AggregateExpr::Count(e)
            | AggregateExpr::Sum(e)
            | AggregateExpr::Min(e)
            | AggregateExpr::Max(e)
            | AggregateExpr::Avg(e) => e.clone(),
            // VAR_POP/VAR_SAMP store their inner expression inside a Box;
            // unbox a clone so it threads through the same `feed` list as
            // the other aggregates.
            AggregateExpr::VarPop(e) | AggregateExpr::VarSamp(e) => (**e).clone(),
            // STDDEV variants box their operand; clone the contents so the
            // collected feed list shape matches the other arms (a Vec<Expr>,
            // not Vec<Box<Expr>>). The physical-plan lowerer treats the
            // feed-collection step uniformly across aggregate variants.
            AggregateExpr::StddevPop(e) | AggregateExpr::StddevSamp(e) => (**e).clone(),
        };
        agg_input_exprs.push(e);
    }

    // The "feed" exprs are group_by then aggregate inputs, in that order.
    // These exprs live in `input`'s output namespace; if a chain Project
    // sits at the top of `input`, substitute them through it so the rest
    // of this function sees scan-namespace exprs.
    let mut feed: Vec<Expr> = Vec::with_capacity(group_by.len() + agg_input_exprs.len());
    for e in group_by.iter().chain(agg_input_exprs.iter()) {
        let lowered = match chain_proj_map.as_ref() {
            Some(m) => substitute_one(e, m),
            None => e.clone(),
        };
        feed.push(lowered);
    }

    // If there is no filter and every feed expression is a bare column ref, we can skip the
    // pre-aggregation kernel entirely; the aggregator can read those columns straight from the scan.
    let trivial = scan_predicate.is_none() && all_bare_columns(&feed);

    let (pre, agg_inputs, group_indices) = if trivial {
        // Build inputs / ordinals directly from the scan columns referenced by `feed`.
        let mut inputs: Vec<ColumnIO> = Vec::new();
        let mut name_to_ord: HashMap<String, usize> = HashMap::new();
        let mut ordinals: Vec<usize> = Vec::with_capacity(feed.len());
        for e in &feed {
            let name = match e {
                Expr::Column(n) => n.clone(),
                // `all_bare_columns` guarantees this branch is unreachable.
                _ => {
                    return Err(BoltError::Plan(
                        "internal: trivial aggregate feed contained non-column expression".into(),
                    ))
                }
            };
            let ord = if let Some(o) = name_to_ord.get(&name) {
                *o
            } else {
                let field = scan_schema.field(&name)?;
                let o = inputs.len();
                inputs.push(ColumnIO {
                    name: name.clone(),
                    dtype: field.dtype,
                });
                name_to_ord.insert(name, o);
                o
            };
            ordinals.push(ord);
        }
        let group_ords: Vec<usize> = ordinals[..group_by.len()].to_vec();
        (None, inputs, group_ords)
    } else {
        // Emit a pre-aggregation kernel whose outputs are `feed` (group keys then aggregate inputs).
        // `feed` exprs were already substituted into scan namespace above,
        // and `scan_predicate` is already in scan namespace, so go straight
        // to the pure-codegen builder (skipping another round of chain
        // resolution that would double-substitute).
        let named_feed: Vec<(String, Expr)> = feed
            .iter()
            .enumerate()
            .map(|(i, e)| (output_name_for(e, i), e.clone()))
            .collect();
        let pre_plan = build_projection_kernel(
            table,
            scan_schema,
            scan_predicate.as_ref(),
            &named_feed,
        )?;
        let (pre_kernel, _pre_schema) = match pre_plan {
            PhysicalPlan::Projection {
                kernel,
                output_schema,
                ..
            } => (kernel, output_schema),
            // `build_projection_kernel` always returns `Projection`.
            _ => {
                return Err(BoltError::Plan(
                    "internal: build_projection_kernel returned non-Projection".into(),
                ))
            }
        };
        let inputs: Vec<ColumnIO> = pre_kernel.outputs.clone();
        let group_ords: Vec<usize> = (0..group_by.len()).collect();
        (Some(pre_kernel), inputs, group_ords)
    };

    // Substitute aggregate exprs through any chain Project map so the
    // column names they reference match what the pre-aggregation kernel
    // actually exposes (i.e., scan namespace).
    let lowered_aggregates: Vec<AggregateExpr> = aggregates
        .iter()
        .map(|agg| match chain_proj_map.as_ref() {
            None => agg.clone(),
            Some(m) => match agg {
                AggregateExpr::Count(e) => AggregateExpr::Count(substitute_one(e, m)),
                AggregateExpr::Sum(e) => AggregateExpr::Sum(substitute_one(e, m)),
                AggregateExpr::Min(e) => AggregateExpr::Min(substitute_one(e, m)),
                AggregateExpr::Max(e) => AggregateExpr::Max(substitute_one(e, m)),
                AggregateExpr::Avg(e) => AggregateExpr::Avg(substitute_one(e, m)),
                AggregateExpr::VarPop(e) => {
                    AggregateExpr::VarPop(Box::new(substitute_one(e.as_ref(), m)))
                }
                AggregateExpr::VarSamp(e) => {
                    AggregateExpr::VarSamp(Box::new(substitute_one(e.as_ref(), m)))
                }
                AggregateExpr::StddevPop(e) => {
                    AggregateExpr::StddevPop(Box::new(substitute_one(e.as_ref(), m)))
                }
                AggregateExpr::StddevSamp(e) => {
                    AggregateExpr::StddevSamp(Box::new(substitute_one(e.as_ref(), m)))
                }
            },
        })
        .collect();

    let aggregate = AggregateSpec {
        inputs: agg_inputs,
        group_by: group_indices,
        aggregates: lowered_aggregates,
        output_schema,
        // PV-stage-f: filled in by `populate_input_validity` after lowering.
        // Empty (safe-default) for callers that build a plan directly without
        // consulting a `TableProvider`.
        input_has_validity: Vec::new(),
    };

    Ok(PhysicalPlan::Aggregate {
        table: table.to_string(),
        pre,
        aggregate,
    })
}

/// True if `plan` is a Scan/Filter/Project chain that bottoms out in a Scan —
/// Recursively test whether `expr` contains a node the GPU codegen cannot
/// lower — currently:
///   * an `Expr::Unary` whose operand is something other than a bare
///     column reference (with any number of transparent `Alias` wrappers
///     around it), or
///   * any `Expr::Like` (v0.5 has no GPU codegen for `LIKE` — every
///     LIKE forces the host fallback). The function name is kept
///     historic for grep-stability; callers care only about "does this
///     need the host path?".
///
/// Used by `lower()` to gate Filter predicates. Today the GPU codegen
/// emits `Op::IsNullCheck` for `column IS [NOT] NULL` and
/// `column-alias IS [NOT] NULL`; anything more elaborate (a binary
/// expression or literal underneath the unary, e.g. `(x + y) IS NULL`)
/// still has to fall back to the host-side `expr_agg::eval_unary` path
/// because the codegen has no register-level NULL propagation for
/// arbitrary subexpressions.
///
/// Aliases are transparent — we look through them and into their inner
/// expression.
///
/// # Naming
///
/// This used to return `true` for *any* `Expr::Unary`, since the IR had
/// no `IsNullCheck` op. Now that bare-column unary lowers cleanly to
/// the GPU, the function returns `true` ONLY for the cases that still
/// need the host fallback. The function name stays the same so the
/// existing call sites (Filter / Project gating in `lower()`) read
/// naturally — "does the predicate contain a Unary we can't handle?".
fn predicate_contains_unary(expr: &Expr) -> bool {
    match expr {
        Expr::Unary { op, operand } => {
            // `NOT` always routes to the host fallback — the GPU codegen
            // does not lower it yet (see `Codegen::emit_unary`).
            if matches!(op, UnaryOp::Not) {
                return true;
            }
            // Peel through any Alias wrappers — `x AS y IS NULL` is
            // still a bare-column unary that the codegen can lower.
            let mut bare = operand.as_ref();
            loop {
                match bare {
                    Expr::Alias(inner, _) => bare = inner.as_ref(),
                    _ => break,
                }
            }
            // Bare column → GPU path; anything else → host path.
            !matches!(bare, Expr::Column(_))
        }
        Expr::Binary { left, right, .. } => {
            predicate_contains_unary(left) || predicate_contains_unary(right)
        }
        Expr::Alias(inner, _) => predicate_contains_unary(inner),
        // CASE inside a predicate cannot be lowered to the fused GPU
        // kernel today; route to host path.
        Expr::Case { .. } => true,
        // LIKE has no GPU codegen yet — every LIKE forces the host fallback.
        Expr::Like { expr, .. } => {
            let _ = predicate_contains_unary(expr);
            true
        }
        // CAST is rejected wholesale at `lower()` (see the early-reject
        // walk in `lower_depth`), so we should never actually reach a
        // routing decision over a Cast-bearing predicate. Recurse for
        // safety — the answer is the same as for any transparent wrapper.
        Expr::Cast { expr, .. } => predicate_contains_unary(expr),
        // ScalarFn predicates are rejected at `lower()` outright (no
        // host-fallback path yet), so the unary-detection routing decision
        // is moot here. Recurse into the args for completeness — if a
        // future host-fallback wires ScalarFn through Filter, this keeps
        // the routing correct.
        Expr::ScalarFn { args, .. } => args.iter().any(predicate_contains_unary),
        Expr::Column(_) | Expr::Literal(_) => false,
    }
}

/// True if `expr` contains a `BinaryOp::Concat` subexpression anywhere.
///
/// String `||` is Utf8-valued and lives entirely on the host (the GPU
/// codegen has no Utf8 support). Used by `lower()` to detect SELECT-list /
/// Filter predicate trees that need the host-side projection path; see
/// the `LogicalPlan::Project` arm in `lower_depth`.
fn expr_contains_concat(expr: &Expr) -> bool {
    match expr {
        Expr::Binary { op, left, right } => {
            matches!(op, BinaryOp::Concat)
                || expr_contains_concat(left)
                || expr_contains_concat(right)
        }
        Expr::Unary { operand, .. } => expr_contains_concat(operand),
        Expr::Alias(inner, _) => expr_contains_concat(inner),
        Expr::Case { branches, else_branch } => {
            branches.iter().any(|(w, t)| expr_contains_concat(w) || expr_contains_concat(t))
                || else_branch.as_deref().map(expr_contains_concat).unwrap_or(false)
        }
        Expr::Like { expr, .. } => expr_contains_concat(expr),
        Expr::Cast { expr, .. } => expr_contains_concat(expr),
        Expr::ScalarFn { args, .. } => args.iter().any(expr_contains_concat),
        Expr::Column(_) | Expr::Literal(_) => false,
    }
}

/// Walk a `Scan` / `Filter` / `Project` chain and return true if any
/// `Filter` node carries a predicate that contains a `BinaryOp::Concat`.
///
/// The GPU codegen path's fused-projection kernel cannot lower `||`
/// (Utf8 has no device-side support), so the Project arm of `lower()`
/// routes the whole chain through the host-side executor when this
/// returns true.
fn scan_chain_has_concat_filter(plan: &LogicalPlan) -> bool {
    let mut cur = plan;
    loop {
        match cur {
            LogicalPlan::Filter { input, predicate } => {
                if expr_contains_concat(predicate) {
                    return true;
                }
                cur = input.as_ref();
            }
            LogicalPlan::Project { input, .. } => {
                cur = input.as_ref();
            }
            _ => return false,
        }
    }
}

/// Walk a `Scan` / `Filter` / `Project` chain and return true if any
/// `Filter` node carries a predicate that contains an `Expr::Unary`.
///
/// Used by the `Project` arm of `lower()` to detect when a SELECT-list
/// projection sits on top of a `WHERE … IS [NOT] NULL` chain. The GPU
/// codegen path (`lower_projection` → `build_projection_kernel`) hoists
/// every chain Filter into the fused projection kernel's predicate; the
/// kernel cannot lower an Expr::Unary, so we have to detect this here
/// and route the whole stack through the host-side executors.
fn scan_chain_has_unary_filter(plan: &LogicalPlan) -> bool {
    let mut cur = plan;
    loop {
        match cur {
            LogicalPlan::Filter { input, predicate } => {
                if predicate_contains_unary(predicate) {
                    return true;
                }
                cur = input.as_ref();
            }
            LogicalPlan::Project { input, .. } => {
                cur = input.as_ref();
            }
            // Reached the leaf or a non-scan-chain node; stop. (Callers
            // gate this with `is_scan_chain` so the fall-through case is
            // really just `Scan`.)
            _ => return false,
        }
    }
}

/// i.e. something `resolve_source` (and therefore `lower_projection`) can fold
/// down into a single-kernel `Projection`. Anything else (Aggregate, Distinct,
/// Limit, Sort, Union, Join) needs a recursive `lower` call instead, with the
/// outer Project/Filter applied — for wave 7 scaffolds, simply dropped — on
/// top of the result.
fn is_scan_chain(plan: &LogicalPlan) -> bool {
    let mut cur = plan;
    loop {
        match cur {
            LogicalPlan::Scan { .. } => return true,
            LogicalPlan::Filter { input, .. } | LogicalPlan::Project { input, .. } => {
                cur = input.as_ref();
            }
            _ => return false,
        }
    }
}

/// Lower a `LogicalPlan` to a `PhysicalPlan`.
/// True if a `Project { exprs, output_schema }` is a pure pass-through over
/// `input_schema`: every output field's name and source column line up with
/// the corresponding input field in the same position. When true, the
/// caller can drop the Project layer entirely (it would just clone the
/// input batch as-is). Aliased exprs only count as identity when the alias
/// happens to match the input column's name.
fn project_is_identity(
    exprs: &[Expr],
    input_schema: &Schema,
    output_schema: &Schema,
) -> bool {
    if exprs.len() != input_schema.fields.len() {
        return false;
    }
    if output_schema.fields.len() != input_schema.fields.len() {
        return false;
    }
    for (i, e) in exprs.iter().enumerate() {
        let in_name = input_schema.fields[i].name.as_str();
        let out_name = output_schema.fields[i].name.as_str();
        // Only bare `Column(name)` exprs can be no-ops: an `Alias(_, _)`
        // wrapper changes the output column's name even when the source
        // expression matches the input column, so it's never identity.
        let src_name = match e {
            Expr::Column(n) => n.as_str(),
            _ => return false,
        };
        if src_name != in_name || out_name != in_name {
            return false;
        }
    }
    true
}

/// Populate `KernelSpec::input_has_validity` for every kernel inside `plan`
/// by consulting `provider`'s `has_nulls(table, col_idx)` for each input
/// column. Walks `Projection`, `Aggregate.pre`, and recursively into
/// `Distinct` / `Limit` / `Sort` / `Project` / `Union` / `Join` wrappers.
///
/// This is the plan-time signal documented on
/// [`crate::plan::sql_frontend::TableProvider::has_nulls`]: by populating
/// the per-input flag here, downstream codegen ([`crate::jit::ptx_gen`])
/// can emit per-output validity AND-trees referencing only the LoadColumn
/// ops that feed each `Store` — without the executor having to inspect
/// `RecordBatch::null_count()` at run time.
///
/// Safe-`false` semantics: any provider that doesn't override `has_nulls`
/// leaves the per-input flag at `false`, preserving the legacy "no input
/// carries validity" codegen path. So this pass is always sound — at
/// worst it under-flags an input that actually has nulls, in which case
/// the executor's run-time host-strip fallback (see
/// `crate::exec::groupby_with_pre`, `groupby_valid`) handles the row
/// filtering before any kernel sees the data.
pub fn populate_input_validity(
    plan: &mut PhysicalPlan,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
) {
    match plan {
        PhysicalPlan::Projection { table, kernel, .. } => {
            populate_one_kernel(kernel, table, provider);
        }
        PhysicalPlan::Aggregate {
            table,
            pre,
            aggregate,
        } => {
            if let Some(k) = pre.as_mut() {
                populate_one_kernel(k, table, provider);
            }
            // PV-stage-f: mirror the same provider signal onto
            // `AggregateSpec::input_has_validity` so the no-pre executors
            // (`groupby.rs` / `groupby_valid.rs`) see the same plan-time
            // hint that `groupby_with_pre` already consumes via
            // `KernelSpec::input_has_validity`.
            populate_aggregate_spec(aggregate, table, provider);
        }
        PhysicalPlan::Distinct { input }
        | PhysicalPlan::Limit { input, .. }
        | PhysicalPlan::Sort { input, .. }
        | PhysicalPlan::Project { input, .. }
        | PhysicalPlan::Filter { input, .. } => {
            populate_input_validity(input.as_mut(), provider);
        }
        PhysicalPlan::Union { inputs } => {
            for branch in inputs {
                populate_input_validity(branch, provider);
            }
        }
        PhysicalPlan::Join { left, right, .. } => {
            populate_input_validity(left.as_mut(), provider);
            populate_input_validity(right.as_mut(), provider);
        }
    }
}

/// Populate one `KernelSpec`'s `input_has_validity` from the provider.
///
/// Scans `kernel.inputs` and asks the provider for each column's
/// null-bearing status by name. Columns not found in the provider's
/// schema (e.g. synthesised pre-aggregation columns whose names won't
/// resolve there) inherit safe-`false`.
///
/// OR-merge semantics: any `true` flag already present in
/// `kernel.input_has_validity` (e.g. set by [`Codegen::emit_unary`] for
/// an `IS NULL` check) is preserved. A provider that reports
/// `has_nulls == false` cannot *clear* a codegen-set flag, because the
/// `Op::IsNullCheck` instruction still needs the validity pointer wired
/// through even when no row will actually be null. The provider can
/// only ADD more `true` flags (e.g. for columns the codegen merely
/// loaded as values).
fn populate_one_kernel(
    kernel: &mut KernelSpec,
    table: &str,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
) {
    // Resolve column indices against the provider's schema (by name).
    let schema = match provider.schema(table) {
        Ok(s) => s,
        Err(_) => return, // table unknown to provider — leave safe-false.
    };
    let mut flags = Vec::with_capacity(kernel.inputs.len());
    for (i, io) in kernel.inputs.iter().enumerate() {
        let provider_says = schema
            .fields
            .iter()
            .position(|f| f.name == io.name)
            .map(|idx| provider.has_nulls(table, idx))
            .unwrap_or(false);
        // OR with any pre-existing codegen-set flag. If the kernel was
        // built with an empty `input_has_validity` (legacy path) this
        // simplifies to just `provider_says`.
        let existing = kernel
            .input_has_validity
            .get(i)
            .copied()
            .unwrap_or(false);
        flags.push(existing || provider_says);
    }
    kernel.input_has_validity = flags;
}

/// PV-stage-f: populate one `AggregateSpec`'s `input_has_validity` from
/// the provider. Mirror of [`populate_one_kernel`] for the no-pre GROUP
/// BY executors that consume an `AggregateSpec` directly (rather than a
/// `KernelSpec`).
///
/// Scans `aggregate.inputs` and asks the provider for each column's
/// null-bearing status by name. Columns not found in the provider's
/// schema (e.g. synthesised pre-aggregation outputs whose names won't
/// resolve there, which happens in the non-trivial path that emits a
/// `pre` kernel) inherit safe-`false` — under-flagging is sound because
/// the executor's run-time host-strip remains the correctness fallback.
fn populate_aggregate_spec(
    aggregate: &mut AggregateSpec,
    table: &str,
    provider: &dyn crate::plan::sql_frontend::TableProvider,
) {
    let schema = match provider.schema(table) {
        Ok(s) => s,
        Err(_) => return,
    };
    let mut flags = Vec::with_capacity(aggregate.inputs.len());
    for io in &aggregate.inputs {
        let has = schema
            .fields
            .iter()
            .position(|f| f.name == io.name)
            .map(|idx| provider.has_nulls(table, idx))
            .unwrap_or(false);
        flags.push(has);
    }
    aggregate.input_has_validity = flags;
}

/// True if `e` contains a divide (or modulo, once `BinaryOp::Mod` lands)
/// anywhere in its subtree. Used to detect predicates whose correctness
/// relies on SQL's standard short-circuit semantics — see
/// [`warn_if_eager_shortcircuit_unsafe`].
fn expr_contains_div_or_mod(e: &Expr) -> bool {
    match e {
        Expr::Column(_) | Expr::Literal(_) => false,
        Expr::Binary { op, left, right } => {
            // `BinaryOp::Mod` is not yet a variant; once it lands the matcher
            // below will pick it up automatically. Today only `Div` exists.
            if matches!(op, BinaryOp::Div) {
                return true;
            }
            expr_contains_div_or_mod(left) || expr_contains_div_or_mod(right)
        }
        Expr::Alias(inner, _) => expr_contains_div_or_mod(inner),
        Expr::Unary { operand, .. } => expr_contains_div_or_mod(operand),
        Expr::Case {
            branches,
            else_branch,
        } => {
            branches.iter().any(|(w, t)| {
                expr_contains_div_or_mod(w) || expr_contains_div_or_mod(t)
            }) || else_branch
                .as_deref()
                .is_some_and(expr_contains_div_or_mod)
        }
        Expr::Like { expr, .. } => expr_contains_div_or_mod(expr),
        Expr::Cast { expr, .. } => expr_contains_div_or_mod(expr),
        // String scalar functions can't contain Div/Mod themselves, but
        // their arguments are arbitrary scalar expressions, so recurse.
        Expr::ScalarFn { args, .. } => args.iter().any(expr_contains_div_or_mod),
    }
}

/// True if `e` contains a `BinaryOp::And` or `BinaryOp::Or` whose left or
/// right subtree contains a divide / modulo. This is the unsafe pattern
/// described on the AND/OR arm of `Codegen::emit_binary`: standard
/// SQL would short-circuit and skip the divide when the guard fails, but
/// craton-bolt's GPU codegen evaluates both operands eagerly.
fn expr_has_unsafe_eager_shortcircuit(e: &Expr) -> bool {
    match e {
        Expr::Column(_) | Expr::Literal(_) => false,
        Expr::Binary { op, left, right } => {
            if matches!(op, BinaryOp::And | BinaryOp::Or)
                && (expr_contains_div_or_mod(left) || expr_contains_div_or_mod(right))
            {
                return true;
            }
            expr_has_unsafe_eager_shortcircuit(left)
                || expr_has_unsafe_eager_shortcircuit(right)
        }
        Expr::Alias(inner, _) => expr_has_unsafe_eager_shortcircuit(inner),
        Expr::Unary { operand, .. } => expr_has_unsafe_eager_shortcircuit(operand),
        Expr::Case {
            branches,
            else_branch,
        } => {
            branches.iter().any(|(w, t)| {
                expr_has_unsafe_eager_shortcircuit(w)
                    || expr_has_unsafe_eager_shortcircuit(t)
            }) || else_branch
                .as_deref()
                .is_some_and(expr_has_unsafe_eager_shortcircuit)
        }
        Expr::Like { expr, .. } => expr_has_unsafe_eager_shortcircuit(expr),
        Expr::Cast { expr, .. } => expr_has_unsafe_eager_shortcircuit(expr),
        // No And/Or wrapper inside a ScalarFn can short-circuit a sibling
        // operand of the *function call*, but the arguments themselves
        // might contain the unsafe pattern, so recurse.
        Expr::ScalarFn { args, .. } => args.iter().any(expr_has_unsafe_eager_shortcircuit),
    }
}

/// True if any `Expr::Binary { op: Div, .. }` reachable from `op` lives
/// underneath an `Op::Binary { op: And/Or }` in the linear IR. The IR is a
/// register machine (operands are already evaluated before the binary op
/// fires), so the eager-evaluation hazard is intrinsic to the IR shape:
/// the presence of *any* Div op alongside *any* And/Or op in the same
/// kernel means the divide ran unconditionally.
fn kernel_has_unsafe_eager_shortcircuit(kernel: &KernelSpec) -> bool {
    let mut has_div = false;
    let mut has_logical = false;
    for op in &kernel.ops {
        if let Op::Binary { op, .. } = op {
            if matches!(op, BinaryOp::Div) {
                has_div = true;
            }
            if matches!(op, BinaryOp::And | BinaryOp::Or) {
                has_logical = true;
            }
        }
    }
    has_div && has_logical
}

/// Walk `plan` and emit a `log::warn!` if any predicate / projection
/// expression — or any compiled kernel's linear IR — contains `BinaryOp::And`
/// or `BinaryOp::Or` whose subtree includes `BinaryOp::Div` (or `Mod`, once
/// that variant exists).
///
/// This is a discoverability safety net for the documented divergence from
/// SQL short-circuit semantics; see the doc block on the AND/OR arm of
/// `Codegen::emit_binary`. The warning is non-fatal — the plan still
/// executes, just with eager evaluation of both operands.
fn warn_if_eager_shortcircuit_unsafe(plan: &PhysicalPlan) {
    fn check_kernel(kernel: &KernelSpec) -> bool {
        kernel_has_unsafe_eager_shortcircuit(kernel)
    }
    fn walk(plan: &PhysicalPlan) -> bool {
        match plan {
            PhysicalPlan::Projection { kernel, .. } => check_kernel(kernel),
            PhysicalPlan::Aggregate { pre, .. } => {
                pre.as_ref().map(check_kernel).unwrap_or(false)
            }
            PhysicalPlan::Filter { input, predicate } => {
                expr_has_unsafe_eager_shortcircuit(predicate) || walk(input)
            }
            PhysicalPlan::Project { input, exprs, .. } => {
                exprs.iter().any(expr_has_unsafe_eager_shortcircuit) || walk(input)
            }
            PhysicalPlan::Distinct { input }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::Sort { input, .. } => walk(input),
            PhysicalPlan::Union { inputs } => inputs.iter().any(walk),
            PhysicalPlan::Join {
                left, right, on, ..
            } => {
                on.iter()
                    .any(|(l, r)| {
                        expr_has_unsafe_eager_shortcircuit(l)
                            || expr_has_unsafe_eager_shortcircuit(r)
                    })
                    || walk(left)
                    || walk(right)
            }
        }
    }
    if walk(plan) {
        log::warn!(
            "query plan: AND/OR with divide/modulo child — short-circuit \
             semantics not yet implemented; ensure no divide-by-zero in your data"
        );
    }
}

#[tracing::instrument(name = "lower", level = "info", skip_all)]
pub fn lower(plan: &LogicalPlan) -> BoltResult<PhysicalPlan> {
    // v0.7: CASE is now lowered through the GPU codegen path via
    // `Op::Select` (see `Codegen::emit_case`). Scan-chain Project /
    // Filter / pre-aggregation kernels accept CASE; host-side positions
    // (HAVING, post-aggregate SELECT, sort keys) surface a clear
    // not-yet-supported error from `expr_agg::eval_inner` when reached.
    // The previous global pre-flight gate (`plan_contains_case`) has
    // therefore been retired — the codegen and host evaluator each carry
    // their own targeted message.
    //
    // v0.7: numeric ↔ numeric CASTs lower to a PTX `cvt.*` instruction via
    // `Codegen::emit_cast_expr` (which routes through the existing
    // `emit_cast` helper for binary-op dtype unification). What remains
    // rejected here is any CAST whose declared TARGET is non-numeric —
    // i.e. Decimal128 / Date32 / Timestamp / Utf8 — because those types
    // have no PTX register class. CAST with a non-numeric SOURCE is
    // caught downstream when the underlying column / literal is loaded
    // (see `emit_column` / `emit_literal`). Surfacing the target-side
    // rejection here keeps the error message one consistent line
    // regardless of where in the plan the CAST appears (Sort key
    // expressions, in particular, don't go through `Codegen::emit_expr`).
    if let Some(target) = logical_plan_contains_unsupported_cast_target(plan) {
        return Err(BoltError::Plan(format!(
            "CAST to/from {} not yet lowered to GPU",
            cast_unsupported_type_label(target)
        )));
    }
    let phys = lower_depth(plan, 0)?;
    // Static-analysis safety net for the documented short-circuit divergence
    // (see the AND/OR arm in `emit_binary`). Runs once at the lowering
    // boundary; non-fatal warning only.
    warn_if_eager_shortcircuit_unsafe(&phys);
    Ok(phys)
}

/// Human-readable label naming the unsupported category in a CAST
/// rejection — used by [`lower`] to format
/// `"CAST to/from {label} not yet lowered to GPU"` consistently
/// whether the trip-point fired on a target-type scan or on a runtime
/// source-type rejection from `cast_target_is_supported`. The label
/// elides the type parameters (precision/scale for Decimal128,
/// TimeUnit/tz for Timestamp) so the message stays stable as those
/// vary across schemas.
fn cast_unsupported_type_label(dt: DataType) -> &'static str {
    match dt {
        DataType::Decimal128(_, _) => "Decimal128",
        DataType::Date32 => "Date32",
        DataType::Timestamp(_, _) => "Timestamp",
        DataType::Utf8 => "String",
        // Numeric / Bool targets are supported — this label is for the
        // rejection path only. Fall through to a generic catch-all if
        // it's ever called on a supported type (a programmer bug — the
        // walker's job is to return Some only for unsupported targets).
        _ => "unsupported",
    }
}

/// Walk `plan` looking for any `Expr::Cast` node whose declared TARGET
/// dtype is non-numeric / non-Bool — i.e. `Decimal128`, `Date32`,
/// `Timestamp(_, _)`, or `Utf8`. Used by [`lower`] to reject such
/// CASTs at the physical-plan boundary with a tightened message before
/// any kernel codegen runs.
///
/// Numeric ↔ numeric (and Bool ↔ Int) targets are accepted here and
/// then lowered to a PTX `cvt.*` instruction by `Codegen::emit_cast_expr`
/// (see that method for the per-pair PTX mnemonic mapping).
///
/// Source-side rejection is the symmetric guard's job, but the SOURCE
/// dtype of a `CAST` node depends on its inner expression and therefore
/// the surrounding schema — which this walker does not have. The
/// source-side rejection is therefore deferred to runtime in
/// `Codegen::emit_cast_expr`, where the source dtype is known after
/// `emit_expr` has resolved the inner expression. The user observes a
/// single consistent error message regardless of which side tripped the
/// guard, via [`cast_unsupported_type_label`].
///
/// The traversal is recursion-bounded via
/// [`crate::plan::sql_frontend::MAX_RECURSION_DEPTH`] the same way
/// [`lower_depth`] guards itself; depth overflow here degrades safely
/// to "no offending cast found" (the subsequent `lower_depth` will
/// surface the same depth error with a more specific message).
fn logical_plan_contains_unsupported_cast_target(plan: &LogicalPlan) -> Option<DataType> {
    fn cast_target_unsupported(target: DataType) -> bool {
        matches!(
            target,
            DataType::Decimal128(_, _)
                | DataType::Date32
                | DataType::Timestamp(_, _)
                | DataType::Utf8
        )
    }
    fn expr_bad_cast(e: &Expr) -> Option<DataType> {
        match e {
            Expr::Cast { expr, target } => {
                if cast_target_unsupported(*target) {
                    return Some(*target);
                }
                expr_bad_cast(expr)
            }
            Expr::Column(_) | Expr::Literal(_) => None,
            Expr::Binary { left, right, .. } => expr_bad_cast(left).or_else(|| expr_bad_cast(right)),
            Expr::Unary { operand, .. } => expr_bad_cast(operand),
            Expr::Alias(inner, _) => expr_bad_cast(inner),
            Expr::Case { branches, else_branch } => {
                for (w, t) in branches {
                    if let Some(d) = expr_bad_cast(w).or_else(|| expr_bad_cast(t)) {
                        return Some(d);
                    }
                }
                else_branch.as_deref().and_then(expr_bad_cast)
            }
            Expr::Like { expr, .. } => expr_bad_cast(expr),
            Expr::ScalarFn { args, .. } => args.iter().find_map(expr_bad_cast),
        }
    }
    fn agg_bad_cast(a: &AggregateExpr) -> Option<DataType> {
        match a {
            AggregateExpr::Count(e)
            | AggregateExpr::Sum(e)
            | AggregateExpr::Min(e)
            | AggregateExpr::Max(e)
            | AggregateExpr::Avg(e) => expr_bad_cast(e),
            AggregateExpr::VarPop(e)
            | AggregateExpr::VarSamp(e)
            | AggregateExpr::StddevPop(e)
            | AggregateExpr::StddevSamp(e) => expr_bad_cast(e.as_ref()),
        }
    }
    fn walk(plan: &LogicalPlan, depth: usize) -> Option<DataType> {
        if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
            return None;
        }
        match plan {
            LogicalPlan::Scan { .. } => None,
            LogicalPlan::Filter { input, predicate } => {
                expr_bad_cast(predicate).or_else(|| walk(input, depth + 1))
            }
            LogicalPlan::Project { input, exprs } => exprs
                .iter()
                .find_map(expr_bad_cast)
                .or_else(|| walk(input, depth + 1)),
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregates,
            } => group_by
                .iter()
                .find_map(expr_bad_cast)
                .or_else(|| aggregates.iter().find_map(agg_bad_cast))
                .or_else(|| walk(input, depth + 1)),
            LogicalPlan::Distinct { input } => walk(input, depth + 1),
            LogicalPlan::Limit { input, .. } => walk(input, depth + 1),
            LogicalPlan::Sort { input, sort_exprs } => sort_exprs
                .iter()
                .find_map(|se| expr_bad_cast(&se.expr))
                .or_else(|| walk(input, depth + 1)),
            LogicalPlan::Union { inputs } => inputs.iter().find_map(|b| walk(b, depth + 1)),
            LogicalPlan::Join {
                left, right, on, ..
            } => on
                .iter()
                .find_map(|(l, r)| expr_bad_cast(l).or_else(|| expr_bad_cast(r)))
                .or_else(|| walk(left, depth + 1))
                .or_else(|| walk(right, depth + 1)),
        }
    }
    walk(plan, 0)
}

/// Inner recursion for [`lower`]. `depth` is the current recursion depth;
/// returns Err if [`crate::plan::sql_frontend::MAX_RECURSION_DEPTH`] is
/// exceeded — guarding against attacker-controlled deeply nested plans
/// that would otherwise overflow the host thread stack.
fn lower_depth(plan: &LogicalPlan, depth: usize) -> BoltResult<PhysicalPlan> {
    if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
        return Err(BoltError::Plan(format!(
            "plan nesting exceeds depth limit ({})",
            crate::plan::sql_frontend::MAX_RECURSION_DEPTH
        )));
    }
    match plan {
        LogicalPlan::Project { input, exprs } => {
            // Scan/Filter/Project chain → single fused kernel via `lower_projection`.
            // Otherwise (Project over Aggregate, Join, Distinct, etc.) we
            // can't fold into one kernel; emit a thin `Project` rename/reorder
            // layer over the lowered inner plan so SELECT-list order and
            // aliases survive (wave 1 fix #5: SELECT-order Project on top
            // of Aggregate must surface aliased / reordered output names).
            // If the Project is a structural no-op (same field names in the
            // same order as the lowered inner plan), drop it so downstream
            // pattern-matchers (and tests) see the bare inner plan.
            //
            // Pre-flight: if any Filter in the underlying scan chain
            // carries an `Expr::Unary` over a NON-bare-column operand
            // (e.g. `(x + y) IS NULL`), the predicate cannot survive the
            // GPU codegen path — `Op::IsNullCheck` only reads validity
            // bitmaps for input columns, not for arbitrary subexpressions.
            // Force the SQL's `WHERE … IS [NOT] NULL` shape onto the
            // host fallback by lowering the inner plan as-is and wrapping
            // the SELECT-list Project on top. Each layer keeps its
            // host-side semantics; the Project layer evaluates simple
            // bare-column renames against the host RecordBatch produced by
            // the Filter (see `engine.rs` PhysicalPlan::Project arm).
            //
            // Bare-column unary (`x IS NULL`, `x IS NOT NULL`) falls
            // through to the GPU codegen path below — `Codegen::emit_unary`
            // emits `Op::IsNullCheck` and `predicate_contains_unary`
            // returns `false` for that shape, so this branch does NOT
            // fire.
            if is_scan_chain(input) && scan_chain_has_unary_filter(input) {
                log::debug!(
                    "physical_plan: Expr::Unary in scan-chain Filter; \
                     lowering Project to host-side stack \
                     (GPU codegen for IS NULL is deferred)"
                );
                let inner = lower(input)?;
                let output_schema = plan.schema()?;
                if project_is_identity(exprs, inner.output_schema(), &output_schema) {
                    return Ok(inner);
                }
                return Ok(PhysicalPlan::Project {
                    input: Box::new(inner),
                    exprs: exprs.clone(),
                    output_schema,
                });
            }
            // v0.5: SQL `||` (BinaryOp::Concat) is Utf8-valued and lives
            // host-side. If any SELECT-list expression contains Concat —
            // or any chain Filter does — we cannot fold the Project into
            // the GPU codegen kernel. Instead, lower the inner plan (which
            // is still a Scan / Filter / non-Concat-Project chain that the
            // codegen can handle) so it surfaces every input column needed
            // by the Concat expressions, then wrap a host-side
            // `PhysicalPlan::Project` whose executor evaluates the Concat
            // via `expr_agg::eval_expr` over a `HostColumn` env (see
            // `engine.rs` PhysicalPlan::Project arm).
            let proj_has_concat = exprs.iter().any(expr_contains_concat);
            if proj_has_concat || scan_chain_has_concat_filter(input) {
                log::debug!(
                    "physical_plan: BinaryOp::Concat in Project / chain Filter; \
                     lowering to host-side PhysicalPlan::Project \
                     (GPU codegen has no Utf8 support)"
                );
                // Build a bare projection of every input column the
                // Concat expressions reference, plus any other plain
                // Column / Alias outputs in the SELECT list. We just
                // gather column names into a set; the simplest correct
                // thing is to lower the inner plan with NO projection
                // override (i.e. surface every scan column) and let the
                // host Project pull what it needs.
                let inner = lower(input)?;
                let output_schema = plan.schema()?;
                return Ok(PhysicalPlan::Project {
                    input: Box::new(inner),
                    exprs: exprs.clone(),
                    output_schema,
                });
            }
            if is_scan_chain(input) {
                lower_projection(input, Some(exprs), None)
            } else {
                let inner = lower_depth(input, depth + 1)?;
                let output_schema = plan.schema()?;
                if project_is_identity(exprs, inner.output_schema(), &output_schema) {
                    Ok(inner)
                } else {
                    Ok(PhysicalPlan::Project {
                        input: Box::new(inner),
                        exprs: exprs.clone(),
                        output_schema,
                    })
                }
            }
        }
        LogicalPlan::Filter { input, predicate } => {
            // v0.7: `||` in a WHERE predicate (e.g. `WHERE a || b = 'foo'`)
            // routes through the host-side `PhysicalPlan::Filter` executor,
            // mirroring how compound `IS NULL` and `LIKE` are handled. The
            // GPU codegen has no Utf8 register class or string-compare ops
            // and the SELECT-list `||` path is itself host-side
            // (`expr_agg::eval_expr` → `host_concat_strings`), so the
            // cleanest lift is to keep `||` host-side everywhere: lower the
            // inner plan so its output batch carries the Utf8 columns the
            // predicate references, then evaluate `predicate` row-by-row in
            // `crate::exec::filter::execute_filter`. This composes for free
            // with `LIKE` (the host evaluator already routes
            // `(a || b) LIKE 'pat'` through `eval_like` → `eval_inner` →
            // `eval_binary(Concat)`), so equality, inequality, and LIKE all
            // work without a separate code path.
            if expr_contains_concat(predicate) {
                log::debug!(
                    "physical_plan: BinaryOp::Concat in Filter predicate; \
                     lowering to host-side PhysicalPlan::Filter \
                     (GPU codegen has no Utf8 support)"
                );
                let inner = lower(input)?;
                return Ok(PhysicalPlan::Filter {
                    input: Box::new(inner),
                    predicate: predicate.clone(),
                });
            }
            // GPU codegen handles `column IS [NOT] NULL` natively via
            // `Op::IsNullCheck` — see `Codegen::emit_unary`. The host
            // fallback is only required for compound Unary operands
            // (e.g. `(x + y) IS NULL`), which `predicate_contains_unary`
            // still flags. The host-side `PhysicalPlan::Filter` executor
            // (`crate::exec::filter::execute_filter`) drives the full
            // `expr_agg::eval_unary` path for those cases.
            if predicate_contains_unary(predicate) {
                log::debug!(
                    "physical_plan: Expr::Unary in Filter predicate; \
                     lowering to host-side PhysicalPlan::Filter \
                     (GPU codegen for IS NULL is deferred)"
                );
                let inner = lower(input)?;
                return Ok(PhysicalPlan::Filter {
                    input: Box::new(inner),
                    predicate: predicate.clone(),
                });
            }
            if is_scan_chain(input) {
                lower_projection(input, None, Some(predicate))
            } else {
                // Non-scan-chain inputs (Aggregate, Project-over-Aggregate,
                // Join, etc.) can't be folded into the predicate kernel. The
                // classic case is HAVING, which the SQL frontend produces as
                // `Filter { Project { Aggregate { .. } } }`. Lower the inner
                // plan and wrap it in a host-side `PhysicalPlan::Filter`; the
                // executor evaluates `predicate` against the inner plan's
                // output RecordBatch via `expr_agg::eval_expr` and drops the
                // rows that don't satisfy it. The inner plan's output is
                // typically tiny (one row per group for HAVING), so a
                // host-side pass is fine for 0.3.
                let inner = lower_depth(input, depth + 1)?;
                Ok(PhysicalPlan::Filter {
                    input: Box::new(inner),
                    predicate: predicate.clone(),
                })
            }
        }
        LogicalPlan::Scan { .. } => lower_projection(plan, None, None),
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => lower_aggregate(plan, input, group_by, aggregates),
        LogicalPlan::Distinct { input } => {
            let inner = lower_depth(input, depth + 1)?;
            Ok(PhysicalPlan::Distinct {
                input: Box::new(inner),
            })
        }
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            let inner = lower_depth(input, depth + 1)?;
            Ok(PhysicalPlan::Limit {
                input: Box::new(inner),
                limit: *limit,
                offset: *offset,
            })
        }
        LogicalPlan::Sort { input, sort_exprs } => {
            let inner = lower_depth(input, depth + 1)?;
            Ok(PhysicalPlan::Sort {
                input: Box::new(inner),
                sort_exprs: sort_exprs.clone(),
            })
        }
        LogicalPlan::Union { inputs } => {
            if inputs.is_empty() {
                return Err(BoltError::Plan(
                    "UNION requires at least one input".into(),
                ));
            }
            let mut lowered: Vec<PhysicalPlan> = Vec::with_capacity(inputs.len());
            for branch in inputs {
                lowered.push(lower_depth(branch, depth + 1)?);
            }
            // Schema integrity (matching shapes across branches) was
            // already enforced by `LogicalPlan::schema()`; trust that.
            Ok(PhysicalPlan::Union { inputs: lowered })
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => {
            let l = lower_depth(left, depth + 1)?;
            let r = lower_depth(right, depth + 1)?;
            // Build the combined schema *from the physical inputs*: the
            // logical sides may have been folded / projected differently
            // than their physical counterparts, but for the operators
            // currently supported below a Join the two agree. Using the
            // physical sides keeps the stored schema in lock-step with
            // what the executor will actually see at run time.
            let output_schema = join_combined_schema(
                l.output_schema(),
                r.output_schema(),
                *join_type,
            );
            Ok(PhysicalPlan::Join {
                left: Box::new(l),
                right: Box::new(r),
                join_type: *join_type,
                on: on.clone(),
                filter: filter.clone(),
                output_schema,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{
        AggregateExpr, BinaryOp, DataType, Expr, Field, Literal, LogicalPlan, Schema,
    };

    /// Regression: HAVING produces `Filter { Project { Aggregate { .. } } }` in
    /// the logical plan. Before the fix the lowerer silently dropped the outer
    /// Filter (returning the unfiltered aggregate); now the predicate must
    /// survive lowering as a `PhysicalPlan::Filter` wrapper.
    #[test]
    fn having_filter_over_project_aggregate_retains_predicate() {
        // Build the same shape `plan_select` produces for
        // `SELECT k, SUM(v) FROM t GROUP BY k HAVING SUM(v) > 10`.
        let scan_schema = Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("v", DataType::Int64, false),
        ]);
        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: scan_schema,
        };
        let aggregate = LogicalPlan::Aggregate {
            input: Box::new(scan),
            group_by: vec![Expr::Column("k".into())],
            aggregates: vec![AggregateExpr::Sum(Expr::Column("v".into()))],
        };
        // SELECT-order Project on top of the aggregate, surfacing the
        // aggregate output name `sum_v`.
        let project = LogicalPlan::Project {
            input: Box::new(aggregate),
            exprs: vec![
                Expr::Column("k".into()),
                Expr::Column("sum_v".into()),
            ],
        };
        // HAVING SUM(v) > 10 — the SQL frontend rewrites the SUM(v) call into
        // a reference to the aggregate-output column `sum_v`.
        let having = Expr::Binary {
            op: BinaryOp::Gt,
            left: Box::new(Expr::Column("sum_v".into())),
            right: Box::new(Expr::Literal(Literal::Int64(10))),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(project),
            predicate: having.clone(),
        };

        let phys = lower(&plan).expect("lower must succeed");
        match phys {
            PhysicalPlan::Filter { input, predicate } => {
                // Predicate must be the same shape we built.
                match predicate {
                    Expr::Binary { op, left, right } => {
                        assert_eq!(op, BinaryOp::Gt);
                        assert!(matches!(*left, Expr::Column(ref n) if n == "sum_v"));
                        assert!(matches!(*right, Expr::Literal(Literal::Int64(10))));
                    }
                    other => panic!("predicate not preserved: {other:?}"),
                }
                // The inner plan is the lowered Project/Aggregate. The
                // SELECT-order Project here is *not* a structural no-op (its
                // output schema follows SELECT order, but the lowered
                // Aggregate already places group keys first / aggregates
                // second in the same order, so it happens to be identity for
                // this query and the Project layer collapses). Accept either
                // shape: a bare Aggregate, or a Project wrapping an Aggregate.
                match *input {
                    PhysicalPlan::Aggregate { .. } => {}
                    PhysicalPlan::Project { input: inner, .. } => {
                        assert!(
                            matches!(*inner, PhysicalPlan::Aggregate { .. }),
                            "expected Aggregate under Project, got something else"
                        );
                    }
                    other => panic!(
                        "expected Aggregate or Project(Aggregate) under Filter, got {other:?}"
                    ),
                }
            }
            other => panic!("expected PhysicalPlan::Filter at top, got {other:?}"),
        }
    }

    // ---- PV-stage-f: populate_aggregate_spec from TableProvider ----

    /// Tiny in-memory provider for the populate-validity tests below.
    /// Two columns: "k" with no nulls, "v" with nulls.
    struct FakeProvider;

    impl crate::plan::sql_frontend::TableProvider for FakeProvider {
        fn schema(&self, name: &str) -> BoltResult<Schema> {
            if name == "t" {
                Ok(Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("v", DataType::Int64, true),
                ]))
            } else {
                Err(BoltError::Plan(format!("unknown table {name}")))
            }
        }
        fn has_nulls(&self, table: &str, col_idx: usize) -> bool {
            // "v" (idx 1) has nulls; everything else doesn't.
            table == "t" && col_idx == 1
        }
        fn null_count(&self, table: &str, col_idx: usize) -> Option<usize> {
            if table == "t" && col_idx == 1 {
                Some(2)
            } else if table == "t" {
                Some(0)
            } else {
                None
            }
        }
    }

    /// `populate_input_validity` must fill `AggregateSpec::input_has_validity`
    /// parallel to `aggregate.inputs`, surfacing the provider's `has_nulls`
    /// signal column-by-column. Mirrors the existing `KernelSpec`
    /// population covered in PV-stage-d.
    #[test]
    fn pv_stage_f_populate_aggregate_spec_mirrors_provider_signal() {
        let scan_schema = Schema::new(vec![
            Field::new("k", DataType::Int32, false),
            Field::new("v", DataType::Int64, true),
        ]);
        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: scan_schema,
        };
        let agg = LogicalPlan::Aggregate {
            input: Box::new(scan),
            group_by: vec![Expr::Column("k".into())],
            aggregates: vec![AggregateExpr::Sum(Expr::Column("v".into()))],
        };
        let mut phys = lower(&agg).expect("lower must succeed");
        let provider = FakeProvider;
        populate_input_validity(&mut phys, &provider);
        match phys {
            PhysicalPlan::Aggregate { aggregate, .. } => {
                assert_eq!(
                    aggregate.input_has_validity.len(),
                    aggregate.inputs.len(),
                    "input_has_validity must be parallel to inputs"
                );
                // Inputs are (k, v) in feed order; only `v` has nulls per the
                // provider mock. The post-lowering input order matches the
                // group-by-keys-first / aggregate-inputs-second contract in
                // `lower_aggregate`, so input 0 = "k" (no nulls), input 1 = "v"
                // (has nulls).
                let by_name: std::collections::HashMap<&str, bool> = aggregate
                    .inputs
                    .iter()
                    .zip(aggregate.input_has_validity.iter())
                    .map(|(io, &v)| (io.name.as_str(), v))
                    .collect();
                assert_eq!(by_name.get("k"), Some(&false), "k has no nulls");
                assert_eq!(by_name.get("v"), Some(&true), "v has nulls");
            }
            other => panic!("expected Aggregate, got {other:?}"),
        }
    }

    /// Before `populate_input_validity` runs, `AggregateSpec::input_has_validity`
    /// is the empty-vector safe default — preserving every literal-constructor
    /// caller's bit-identical legacy behaviour.
    #[test]
    fn pv_stage_f_aggregate_spec_default_input_has_validity_is_empty() {
        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("k", DataType::Int32, false),
                Field::new("v", DataType::Int64, true),
            ]),
        };
        let agg = LogicalPlan::Aggregate {
            input: Box::new(scan),
            group_by: vec![Expr::Column("k".into())],
            aggregates: vec![AggregateExpr::Sum(Expr::Column("v".into()))],
        };
        let phys = lower(&agg).expect("lower must succeed");
        match phys {
            PhysicalPlan::Aggregate { aggregate, .. } => {
                assert!(
                    aggregate.input_has_validity.is_empty(),
                    "default (pre-populate_input_validity) must be empty Vec — \
                     legacy code path"
                );
            }
            other => panic!("expected Aggregate, got {other:?}"),
        }
    }

    /// Regression (review C9): a hand-constructed `LogicalPlan::Union` with
    /// zero inputs must surface as a `BoltError::Plan` through the public
    /// `DataFrame::from_plan` entry point — never a panic. Before this fix,
    /// the panic site was `PhysicalPlan::output_schema`'s
    /// `inputs.first().expect(..)`, reachable because `from_plan` accepted
    /// any `LogicalPlan` shape unconditionally.
    ///
    /// We assert three things, in order of how a real caller would hit them:
    ///   1. `lower()` rejects the empty Union with `BoltError::Plan` (the
    ///      pre-existing guard — kept as defence in depth).
    ///   2. `DataFrame::from_plan(...).validation_error()` returns Some,
    ///      naming the empty Union — the user-facing surface from review C9.
    ///   3. `DataFrame::from_plan(...).schema()` returns the same
    ///      `BoltError::Plan` rather than panicking.
    #[test]
    fn empty_union_surfaces_as_plan_error_not_panic() {
        use crate::error::BoltError;
        use crate::plan::dataframe::DataFrame;

        // Build a zero-branch UNION directly. This is the malformed shape a
        // user could hand `DataFrame::from_plan` to trip the old `expect()`.
        let empty_union = LogicalPlan::Union { inputs: vec![] };

        // (1) The lowerer's own guard still catches it.
        let lower_err = lower(&empty_union).expect_err("lower must reject empty Union");
        match lower_err {
            BoltError::Plan(msg) => assert!(
                msg.contains("UNION") || msg.contains("Union"),
                "lower() error should mention UNION; got: {msg}",
            ),
            other => panic!("expected BoltError::Plan, got {other:?}"),
        }

        // (2) `DataFrame::from_plan` records the error in `first_error`,
        //     surfaced via `validation_error()`. No panic, no `expect()`.
        let df = DataFrame::from_plan(empty_union.clone());
        let err_msg = df
            .validation_error()
            .expect("from_plan must record an error for empty Union");
        assert!(
            err_msg.contains("Union") || err_msg.contains("UNION"),
            "validation_error should mention Union; got: {err_msg}",
        );

        // (3) `schema()` mirrors the same error rather than calling through
        //     to the panicking accessor path.
        let schema_err = df.schema().expect_err("schema() must surface the error");
        match schema_err {
            BoltError::Plan(msg) => assert!(
                msg.contains("Union") || msg.contains("UNION"),
                "schema() error should mention Union; got: {msg}",
            ),
            other => panic!("expected BoltError::Plan, got {other:?}"),
        }
    }

    // ---- v0.7: WHERE `||` (BinaryOp::Concat) lowers to host-side filter ----

    /// Schema fixture: two Utf8 columns `a`, `b` plus an Int64 `v`. Mirrors
    /// the realistic shape of a `WHERE name || surname = 'JohnDoe'`
    /// predicate alongside a non-Utf8 projection column.
    fn ab_v_scan() -> LogicalPlan {
        LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("a", DataType::Utf8, false),
                Field::new("b", DataType::Utf8, false),
                Field::new("v", DataType::Int64, false),
            ]),
        }
    }

    /// `WHERE a || b = 'foo'` must lower to a `PhysicalPlan::Filter` that
    /// preserves the predicate verbatim, mirroring the routing for LIKE and
    /// compound `IS NULL`. The inner plan is whatever `lower()` produces for
    /// the underlying Scan; the executor (`exec::filter::execute_filter`)
    /// evaluates the concat row-by-row via `expr_agg::eval_expr`.
    #[test]
    fn where_concat_column_column_eq_literal_lowers_to_host_filter() {
        let scan = ab_v_scan();
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
        let phys = lower(&plan).expect("WHERE a || b = 'foo' must lower cleanly in v0.7");
        match phys {
            PhysicalPlan::Filter { predicate, .. } => {
                // Predicate is preserved verbatim — the host filter
                // executor sees the same tree the planner built.
                match &predicate {
                    Expr::Binary {
                        op: BinaryOp::Eq,
                        left,
                        right,
                    } => {
                        assert!(
                            matches!(
                                left.as_ref(),
                                Expr::Binary { op: BinaryOp::Concat, .. }
                            ),
                            "LHS must be a Concat, got: {left:?}",
                        );
                        match right.as_ref() {
                            Expr::Literal(Literal::Utf8(s)) => assert_eq!(s, "foo"),
                            other => panic!("RHS must be Utf8 literal 'foo', got: {other:?}"),
                        }
                    }
                    other => panic!("predicate not preserved: {other:?}"),
                }
            }
            other => panic!(
                "expected PhysicalPlan::Filter for WHERE-|| predicate, got {other:?}"
            ),
        }
    }

    /// `WHERE 'a' || b = 'ab'` — literal-on-left shape. Same routing as the
    /// column-on-left case; just confirms the `expr_contains_concat` walk
    /// catches the Concat regardless of which operand carries the literal.
    #[test]
    fn where_concat_literal_column_eq_literal_lowers_to_host_filter() {
        let scan = ab_v_scan();
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
        let phys = lower(&plan).expect("WHERE 'a' || b = 'ab' must lower cleanly in v0.7");
        assert!(
            matches!(phys, PhysicalPlan::Filter { .. }),
            "expected PhysicalPlan::Filter for literal||column predicate, got {phys:?}",
        );
    }

    /// `WHERE a || b <> 'foo'` — inequality composes the same way. The host
    /// filter handles `=`, `<>`, and `LIKE` over a Concat operand uniformly
    /// via `expr_agg::eval_expr`, so the routing test only needs to confirm
    /// the lower produces a Filter shape (no concat-rejection error).
    #[test]
    fn where_concat_neq_literal_lowers_to_host_filter() {
        let scan = ab_v_scan();
        let pred = Expr::Binary {
            op: BinaryOp::NotEq,
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
        let phys = lower(&plan).expect("WHERE a || b <> 'foo' must lower cleanly in v0.7");
        assert!(
            matches!(phys, PhysicalPlan::Filter { .. }),
            "expected PhysicalPlan::Filter for WHERE concat <> 'lit', got {phys:?}",
        );
    }

    /// A Concat nested under an AND (e.g.
    /// `WHERE v > 0 AND a || b = 'foo'`) must still route through the host
    /// filter — the walk is recursive. Guards against a future refactor
    /// that accidentally only inspects the top-level binary op.
    #[test]
    fn where_concat_under_and_lowers_to_host_filter() {
        let scan = ab_v_scan();
        let pred = Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(Expr::Column("v".into())),
                right: Box::new(Expr::Literal(Literal::Int64(0))),
            }),
            right: Box::new(Expr::Binary {
                op: BinaryOp::Eq,
                left: Box::new(Expr::Binary {
                    op: BinaryOp::Concat,
                    left: Box::new(Expr::Column("a".into())),
                    right: Box::new(Expr::Column("b".into())),
                }),
                right: Box::new(Expr::Literal(Literal::Utf8("foo".into()))),
            }),
        };
        let plan = LogicalPlan::Filter {
            input: Box::new(scan),
            predicate: pred,
        };
        let phys = lower(&plan)
            .expect("WHERE v > 0 AND a || b = 'foo' must lower cleanly in v0.7");
        assert!(
            matches!(phys, PhysicalPlan::Filter { .. }),
            "expected PhysicalPlan::Filter for AND-wrapped concat predicate, got {phys:?}",
        );
    }

    /// Companion to `empty_union_surfaces_as_plan_error_not_panic`: a
    /// *nested* empty Union (e.g. as one branch of a Filter or a non-empty
    /// outer Union) must also be caught at `from_plan` time. Ensures the
    /// recursive walk in `check_no_empty_union` covers the structural cases
    /// we care about.
    #[test]
    fn nested_empty_union_is_rejected_by_from_plan() {
        use crate::plan::dataframe::DataFrame;

        let scan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("k", DataType::Int32, false)]),
        };
        // Outer Union has one valid branch and one degenerate empty Union —
        // the recursive walk must still flag the inner empty.
        let nested = LogicalPlan::Union {
            inputs: vec![scan, LogicalPlan::Union { inputs: vec![] }],
        };
        let df = DataFrame::from_plan(nested);
        assert!(
            df.validation_error().is_some(),
            "nested empty Union must be flagged by from_plan",
        );
    }

    // ---------------------------------------------------------------------
    // v0.7: KernelSpec coverage for non-projection kernel kinds.
    //
    // These tests pin the cache-key roundtrip invariant for each new sibling
    // spec type: distinct knob values must produce distinct `Debug` outputs
    // (the basis for `exec::module_cache::KernelSpecKey::new`, which hashes
    // `format!("{:?}", spec)` with a domain-separated `DefaultHasher`). We
    // reproduce the same hashing shape locally rather than reach into the
    // private `KernelSpecKey` — the property under test is "different specs
    // hash differently", not the specific 128-bit fingerprint.
    // ---------------------------------------------------------------------

    /// Mirror of `KernelSpecKey::new` for tests: hash `format!("{:?}", spec)`
    /// with two domain-separated `DefaultHasher` instances and return the
    /// 128-bit fingerprint as a tuple. Two specs with the same `Debug`
    /// output produce the same fingerprint; distinct `Debug` outputs are
    /// overwhelmingly likely to produce distinct fingerprints.
    fn dbg_key<T: std::fmt::Debug>(spec: &T) -> (u64, u64) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;
        let s = format!("{:?}", spec);
        let mut hi = DefaultHasher::new();
        hi.write_u8(0x01);
        hi.write(s.as_bytes());
        let mut lo = DefaultHasher::new();
        lo.write_u8(0x02);
        lo.write(s.as_bytes());
        (hi.finish(), lo.finish())
    }

    /// Build a `ScalarAggSpec` for each `(op, dtype)` pair the codegen
    /// accepts and confirm:
    ///   1. The Debug output of each pair is unique (no string collisions).
    ///   2. The hashed cache-key roundtrip distinguishes every pair.
    ///   3. Cloning a spec produces the same cache key.
    #[test]
    fn scalar_agg_spec_key_roundtrip() {
        use std::collections::HashSet;

        let ops = [
            ScalarAggOp::Sum,
            ScalarAggOp::Min,
            ScalarAggOp::Max,
            ScalarAggOp::Count,
            ScalarAggOp::Avg,
        ];
        let dtypes = [
            DataType::Int32,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
        ];

        let mut keys: HashSet<(u64, u64)> = HashSet::new();
        let mut dbgs: HashSet<String> = HashSet::new();
        for op in ops {
            for dtype in dtypes {
                let spec = ScalarAggSpec {
                    op,
                    input_dtype: dtype,
                };
                let dbg = format!("{:?}", spec);
                assert!(
                    dbgs.insert(dbg.clone()),
                    "Debug output collision for ScalarAggSpec ({:?}, {:?}): {dbg}",
                    op,
                    dtype,
                );
                let key = dbg_key(&spec);
                assert!(
                    keys.insert(key),
                    "cache-key collision for ScalarAggSpec ({:?}, {:?})",
                    op,
                    dtype,
                );
                // Cloning preserves the cache key.
                let clone = spec.clone();
                assert_eq!(
                    dbg_key(&spec),
                    dbg_key(&clone),
                    "Clone changed the cache key for ScalarAggSpec ({:?}, {:?})",
                    op,
                    dtype,
                );
            }
        }
        assert_eq!(
            keys.len(),
            ops.len() * dtypes.len(),
            "every (op, dtype) pair must produce a distinct key",
        );
    }

    /// Round-trip pin: every `(kind, key_dtype, returns_i64)` triple
    /// produces a distinct cache key.
    #[test]
    fn hash_join_kernel_spec_key_roundtrip() {
        use std::collections::HashSet;

        // Cover at least one of each kind plus a couple of cross-cuts (the
        // StringHash flavour bit, the key_dtype field).
        let specs = [
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::Build,
                key_dtype: DataType::Int32,
                string_hash_returns_i64: false,
            },
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::Build,
                key_dtype: DataType::Int64,
                string_hash_returns_i64: false,
            },
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::Probe,
                key_dtype: DataType::Int32,
                string_hash_returns_i64: false,
            },
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::ProbeTiled,
                key_dtype: DataType::Int32,
                string_hash_returns_i64: false,
            },
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::BuildCollision,
                key_dtype: DataType::Int32,
                string_hash_returns_i64: false,
            },
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::ProbeCollision,
                key_dtype: DataType::Int32,
                string_hash_returns_i64: false,
            },
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::BuildAos,
                key_dtype: DataType::Int32,
                string_hash_returns_i64: false,
            },
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::ProbeAos,
                key_dtype: DataType::Int32,
                string_hash_returns_i64: false,
            },
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::UnmatchedBuild,
                key_dtype: DataType::Int32,
                string_hash_returns_i64: false,
            },
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::Cross,
                key_dtype: DataType::Int32,
                string_hash_returns_i64: false,
            },
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::StringHash,
                key_dtype: DataType::Utf8,
                string_hash_returns_i64: false,
            },
            // The `_i64` flavour of StringHash must NOT collide with the
            // regular one.
            HashJoinKernelSpec {
                kind: HashJoinKernelKind::StringHash,
                key_dtype: DataType::Utf8,
                string_hash_returns_i64: true,
            },
        ];

        let mut keys: HashSet<(u64, u64)> = HashSet::new();
        for s in &specs {
            assert!(
                keys.insert(dbg_key(s)),
                "cache-key collision for HashJoinKernelSpec {:?}",
                s,
            );
        }
        assert_eq!(
            keys.len(),
            specs.len(),
            "every distinct (kind, key_dtype, returns_i64) triple must produce a \
             distinct cache key",
        );

        // Clone roundtrip: cloning preserves the cache key.
        let s = specs[0];
        let clone = s.clone();
        assert_eq!(dbg_key(&s), dbg_key(&clone));
    }

    /// Round-trip pin: every `(pass, dtype)` pair produces a distinct
    /// cache key. The shift parameter is intentionally NOT part of the
    /// spec — it's a runtime kernel arg, so two `(Histogram, Int32)` specs
    /// at different shifts MUST land in the same cache slot.
    #[test]
    fn radix_sort_kernel_spec_key_roundtrip() {
        use std::collections::HashSet;

        let passes = [
            RadixSortPass::Histogram,
            RadixSortPass::Scatter,
            RadixSortPass::ScatterWithIndices,
            RadixSortPass::MsbFlip,
        ];
        // The radix kernels in `jit::sort_kernel_radix` support b32 and b64
        // integer keys today; the spec admits any dtype the codegen accepts.
        let dtypes = [DataType::Int32, DataType::Int64];

        let mut keys: HashSet<(u64, u64)> = HashSet::new();
        for pass in passes {
            for dtype in dtypes {
                let spec = RadixSortKernelSpec { pass, dtype };
                assert!(
                    keys.insert(dbg_key(&spec)),
                    "cache-key collision for RadixSortKernelSpec ({:?}, {:?})",
                    pass,
                    dtype,
                );
                let clone = spec.clone();
                assert_eq!(dbg_key(&spec), dbg_key(&clone));
            }
        }
        assert_eq!(
            keys.len(),
            passes.len() * dtypes.len(),
            "every (pass, dtype) pair must produce a distinct key",
        );
    }

    /// Round-trip pin for the compaction-pipeline specs. Three independent
    /// invariants:
    ///   1. The three `PrefixScan` algorithm variants must produce
    ///      distinct keys (so the env-driven algorithm switch does NOT
    ///      collide on cache slots).
    ///   2. Per-dtype `Gather` variants must produce distinct keys.
    ///   3. `GatherBoolNullable` must NOT collide with `Gather(Bool)` —
    ///      the validity store path is distinct PTX.
    #[test]
    fn compaction_kernel_spec_key_roundtrip() {
        use std::collections::HashSet;

        let specs = [
            CompactionKernelSpec {
                kind: CompactionKernelKind::PrefixScan(PrefixScanAlgoTag::HillisSteele),
            },
            CompactionKernelSpec {
                kind: CompactionKernelKind::PrefixScan(PrefixScanAlgoTag::Blelloch),
            },
            CompactionKernelSpec {
                kind: CompactionKernelKind::PrefixScan(PrefixScanAlgoTag::Lookback),
            },
            CompactionKernelSpec {
                kind: CompactionKernelKind::Gather(DataType::Bool),
            },
            CompactionKernelSpec {
                kind: CompactionKernelKind::Gather(DataType::Int32),
            },
            CompactionKernelSpec {
                kind: CompactionKernelKind::Gather(DataType::Int64),
            },
            CompactionKernelSpec {
                kind: CompactionKernelKind::Gather(DataType::Float32),
            },
            CompactionKernelSpec {
                kind: CompactionKernelKind::Gather(DataType::Float64),
            },
            CompactionKernelSpec {
                kind: CompactionKernelKind::GatherBoolNullable,
            },
        ];

        let mut keys: HashSet<(u64, u64)> = HashSet::new();
        for s in &specs {
            assert!(
                keys.insert(dbg_key(s)),
                "cache-key collision for CompactionKernelSpec {:?}",
                s,
            );
        }
        assert_eq!(keys.len(), specs.len(), "all variants must hash distinctly");

        // Specifically pin (3): Gather(Bool) vs GatherBoolNullable.
        let plain = CompactionKernelSpec {
            kind: CompactionKernelKind::Gather(DataType::Bool),
        };
        let nullable = CompactionKernelSpec {
            kind: CompactionKernelKind::GatherBoolNullable,
        };
        assert_ne!(
            dbg_key(&plain),
            dbg_key(&nullable),
            "Gather(Bool) and GatherBoolNullable must produce distinct keys — the \
             validity store path is distinct PTX",
        );
    }

    /// Sanity test for the `ScalarAggOp` <-> `crate::jit::agg_kernels::ReduceOp`
    /// mapping. We can't depend on `jit::` from a planner-side test without
    /// pulling in the whole codegen layer, but we CAN pin the local enum's
    /// variant count and ordering so a stale mirror is caught at test time.
    ///
    /// The pin: there are exactly five variants and each formats as its name
    /// in `Debug`. If a future change adds (say) `BitAnd` here, this test
    /// will fail with a count mismatch — prompting the author to update the
    /// jit-side mirror in lockstep.
    #[test]
    fn scalar_agg_spec_op_round_trips() {
        let all = [
            ScalarAggOp::Sum,
            ScalarAggOp::Min,
            ScalarAggOp::Max,
            ScalarAggOp::Count,
            ScalarAggOp::Avg,
        ];
        assert_eq!(
            all.len(),
            5,
            "ScalarAggOp has {} variants; if you added one, also update \
             the mirror in crate::jit::agg_kernels::ReduceOp / the AVG path",
            all.len()
        );
        // Per-variant Debug pin — guards against an unexpected rename, which
        // would silently reshape every existing cache slot for that op.
        let names = [
            format!("{:?}", ScalarAggOp::Sum),
            format!("{:?}", ScalarAggOp::Min),
            format!("{:?}", ScalarAggOp::Max),
            format!("{:?}", ScalarAggOp::Count),
            format!("{:?}", ScalarAggOp::Avg),
        ];
        assert_eq!(names, ["Sum", "Min", "Max", "Count", "Avg"]);
    }

    /// Round-trip pin for the local `PrefixScanAlgoTag` mirror of
    /// `gpu_compact::PrefixScanAlgo`. Same shape as the `ScalarAggOp`
    /// pin above — a count check so a stale mirror is caught at test time.
    #[test]
    fn compaction_spec_prefix_scan_round_trips() {
        let all = [
            PrefixScanAlgoTag::HillisSteele,
            PrefixScanAlgoTag::Blelloch,
            PrefixScanAlgoTag::Lookback,
        ];
        assert_eq!(
            all.len(),
            3,
            "PrefixScanAlgoTag has {} variants; if you added one, also \
             update the mirror in crate::exec::gpu_compact::PrefixScanAlgo",
            all.len()
        );
        let names = [
            format!("{:?}", PrefixScanAlgoTag::HillisSteele),
            format!("{:?}", PrefixScanAlgoTag::Blelloch),
            format!("{:?}", PrefixScanAlgoTag::Lookback),
        ];
        assert_eq!(names, ["HillisSteele", "Blelloch", "Lookback"]);
    }

    /// The envelope `KernelSpecKind::Projection(spec)` MUST hash
    /// differently from the bare `&spec` — this is the documented caveat
    /// on `KernelSpecKind` (the envelope is for the new spec kinds only;
    /// the wired projection cache continues passing `&KernelSpec` so its
    /// legacy disk-cache slots don't get evicted on first run).
    #[test]
    fn kernel_spec_kind_projection_envelope_differs_from_bare_spec() {
        let spec = KernelSpec {
            inputs: Vec::new(),
            outputs: Vec::new(),
            ops: Vec::new(),
            predicate: None,
            register_count: 0,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        let envelope = KernelSpecKind::Projection(spec.clone());
        // Sanity: distinct Debug outputs.
        let bare_dbg = format!("{:?}", spec);
        let env_dbg = format!("{:?}", envelope);
        assert_ne!(
            bare_dbg, env_dbg,
            "envelope must wrap-and-rename the Debug output (or the wired \
             projection cache key would collide with the new-envelope key)",
        );
        // The envelope `Debug` carries the inner Projection variant tag.
        assert!(
            env_dbg.starts_with("Projection("),
            "KernelSpecKind::Projection(_) Debug must start with `Projection(`; got: {env_dbg}",
        );
    }

    /// The envelope variants must each produce a distinct cache key.
    /// Pins the "uniform envelope" property — any executor can route through
    /// `KernelSpecKind` and get correct cache disambiguation for free.
    #[test]
    fn kernel_spec_kind_envelope_variants_hash_distinctly() {
        use std::collections::HashSet;

        let bare_spec = KernelSpec {
            inputs: Vec::new(),
            outputs: Vec::new(),
            ops: Vec::new(),
            predicate: None,
            register_count: 0,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        let variants = [
            KernelSpecKind::Projection(bare_spec),
            KernelSpecKind::ScalarAgg(ScalarAggSpec {
                op: ScalarAggOp::Sum,
                input_dtype: DataType::Int32,
            }),
            KernelSpecKind::HashJoin(HashJoinKernelSpec {
                kind: HashJoinKernelKind::Build,
                key_dtype: DataType::Int32,
                string_hash_returns_i64: false,
            }),
            KernelSpecKind::RadixSort(RadixSortKernelSpec {
                pass: RadixSortPass::Histogram,
                dtype: DataType::Int32,
            }),
            KernelSpecKind::Compaction(CompactionKernelSpec {
                kind: CompactionKernelKind::GatherBoolNullable,
            }),
        ];

        let mut keys: HashSet<(u64, u64)> = HashSet::new();
        for v in &variants {
            assert!(
                keys.insert(dbg_key(v)),
                "envelope variants collide: {:?}",
                v,
            );
        }
        assert_eq!(keys.len(), variants.len());
    }
}
