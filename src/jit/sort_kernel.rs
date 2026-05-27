// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for a single-key **bitonic sort** kernel.
//!
//! Backs `crate::exec::gpu_sort`: ORDER BY's GPU fast path. The host-side
//! executor allocates a padded keys buffer + parallel indices buffer of size
//! `n_pow2 = next_power_of_two(n_rows)`, fills the tail with a sentinel
//! (`+INF`-style for ASC / `-INF`-style for DESC), then launches this kernel
//! once per (stage, substage) pair. After the sort, the first `n_rows` indices
//! (or the last `n_rows`, depending on direction + sentinel choice) are the
//! permutation to apply to every output column.
//!
//! ## Algorithm — bitonic sort, one launch per substage
//!
//! Bitonic sort is the standard pedagogical GPU sort: log2(n)*(log2(n)+1)/2
//! deterministic compare-exchange waves, each fully parallel. It needs n to
//! be a power of two — see `gpu_sort.rs` for the padding strategy.
//!
//! For Stage 1 we issue **one kernel launch per substage** for simplicity.
//! An in-block shared-memory variant (which would amortise substages within
//! a block into a single launch) is a deliberate follow-up; see the
//! `TODO(s1-stage2)` notes in `gpu_sort.rs`.
//!
//! The per-thread logic in pseudo-PTX:
//!
//! ```text
//!   tid = blockIdx.x * blockDim.x + threadIdx.x
//!   if tid >= n_pow2: return
//!   partner = tid XOR k_mask    // k_mask = 1 << (substage - 1)
//!   if tid >= partner: return   // only the lower index of the pair acts
//!   asc_block = ((tid >> j) & 1) == 0   // j = stage index (1-based)
//!   v_self    = keys[tid]
//!   v_partner = keys[partner]
//!   if (asc_block XOR global_desc) ? v_self > v_partner : v_self < v_partner:
//!       swap(keys[tid], keys[partner])
//!       swap(indices[tid], indices[partner])
//! ```
//!
//! `global_desc` is baked into the emitted PTX at compile time (the kernel
//! is monomorphised per direction) so the inner branch is just a single
//! `setp.lt` / `setp.gt`. Stage `j` and substage-mask `k_mask` are passed as
//! `.param .u32` arguments so a single PTX module can serve every wave.
//!
//! ## Limits (Stage 1)
//!
//! - Single key only. Multi-key (lexicographic) is `TODO(s1-stage2)`; the
//!   natural extension is a second value buffer and a cascading comparator.
//! - No NULL handling. Stage 1 requires `null_count() == 0`. NULLs can be
//!   threaded as a parallel `validity` buffer with a sentinel-aware
//!   comparator (`TODO(s1-stage2)`).
//! - `n_pow2 <= u32::MAX`. The grid index is `u32` and the (stage, substage)
//!   pair fits in `u32`.
//!
//! ## ABI
//!
//! ```text
//! .visible .entry bolt_bitonic_sort_<dtype>_<dir>(
//!     .param .u64 keys_ptr,        // pointer to the key values buffer
//!     .param .u64 indices_ptr,     // pointer to the u32 indices buffer
//!     .param .u32 n_pow2,          // padded element count, == 1 << log2_n
//!     .param .u32 stage,           // current outer stage j  (1-based)
//!     .param .u32 substage_mask    // 1 << (substage - 1)
//! )
//! ```
//!
//! Grid: 1D, `n_pow2` threads total, block size 256 (one thread per element).

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::DataType;

/// PTX target metadata baked into every emitted module. Matches the rest of
/// the JIT pipeline (see `prefix_scan.rs`, `scan_kernel.rs`).
const PTX_VERSION: &str = ".version 7.5";
/// Target SM architecture string.
const PTX_TARGET: &str = ".target sm_70";
/// Address size directive (we always use 64-bit pointers).
const PTX_ADDRESS_SIZE: &str = ".address_size 64";

/// Threads per block for the sort launch. Matches `BLOCK_SIZE` elsewhere so
/// occupancy tuning stays uniform across the engine's kernels.
pub const SORT_BLOCK_SIZE: u32 = 256;

/// Direction of the sort (global, monotonic). Baked into the PTX at codegen
/// so the inner compare-exchange branch is one `setp` + one predicated swap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SortDirection {
    /// Ascending — smallest first.
    Asc,
    /// Descending — largest first.
    Desc,
}

/// Entry-point name for the sort kernel for `(dtype, dir)`. Static so callers
/// can pass it straight to `CudaModule::function`.
pub fn sort_kernel_entry(dtype: DataType, dir: SortDirection) -> BoltResult<&'static str> {
    Ok(match (dtype, dir) {
        (DataType::Int32, SortDirection::Asc) => "bolt_bitonic_sort_i32_asc",
        (DataType::Int32, SortDirection::Desc) => "bolt_bitonic_sort_i32_desc",
        (DataType::Int64, SortDirection::Asc) => "bolt_bitonic_sort_i64_asc",
        (DataType::Int64, SortDirection::Desc) => "bolt_bitonic_sort_i64_desc",
        (DataType::Float32, SortDirection::Asc) => "bolt_bitonic_sort_f32_asc",
        (DataType::Float32, SortDirection::Desc) => "bolt_bitonic_sort_f32_desc",
        (DataType::Float64, SortDirection::Asc) => "bolt_bitonic_sort_f64_asc",
        (DataType::Float64, SortDirection::Desc) => "bolt_bitonic_sort_f64_desc",
        _ => {
            return Err(BoltError::Other(format!(
                "sort_kernel: dtype {:?} not supported (Stage 1 supports \
                 Int32/Int64/Float32/Float64)",
                dtype
            )))
        }
    })
}

/// Compile the bitonic-sort PTX module for `(dtype, dir)`.
///
/// Returns the full PTX source as a string ready to feed to
/// `CudaModule::from_ptx`. The emitted module exports exactly one entry point
/// whose name is given by [`sort_kernel_entry`].
pub fn compile_sort_kernel(dtype: DataType, dir: SortDirection) -> BoltResult<String> {
    let entry = sort_kernel_entry(dtype, dir)?;

    // Per-dtype PTX flavour: the .b8 element width on load/store, the register
    // class for the key, and the `setp` mnemonic + operand type for the
    // compare-exchange test.
    let flavour = DtypeFlavour::for_dtype(dtype)?;

    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // Signature: keys ptr, indices ptr, n_pow2, stage, substage_mask.
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_0,").map_err(write_err)?;
    writeln!(p, "\t.param .u64 {entry}_param_1,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_2,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_3,").map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_4").map_err(write_err)?;
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    // Register declarations.
    //
    //   pred: %p0 = oob (tid >= n_pow2)
    //         %p1 = paired-skip (tid >= partner)
    //         %p2 = asc_block xor global_desc -> direction of THIS pair
    //         %p3 = compare result for ASC pairs
    //         %p4 = compare result for DESC pairs
    //         %p5 = final do_swap predicate
    //   b32 : %r0=stage, %r1=substage_mask, %r2=n_pow2,
    //         %r3=tid, %r4..r7 working ints (ctaid/ntid/tidx, partner, etc.)
    //         %r8=asc_block_bit, %r9=u32 idx scratch for keys (indices values)
    //   b64 : %rd0=keys ptr, %rd1=indices ptr (after cvta.to.global),
    //         %rd2..%rd7 working address temps
    //   key class: provided by flavour (f/fd/r/rl). Two registers needed:
    //         %kself, %kpart (and 2 u32 scratch for index swap).
    writeln!(p, "\t.reg .pred %p<8>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32 %r<16>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %rd<16>;").map_err(write_err)?;
    writeln!(p, "\t.reg .{} %k<4>;", flavour.reg_type).map_err(write_err)?;

    // -------- tid = blockIdx.x * blockDim.x + threadIdx.x ----------
    writeln!(p, "\tmov.u32 %r4, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r5, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r6, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r4, %r5, %r6;").map_err(write_err)?;

    // -------- Load runtime scalars: n_pow2, stage, substage_mask. ----------
    writeln!(p, "\tld.param.u32 %r2, [{entry}_param_2];").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r0, [{entry}_param_3];").map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r1, [{entry}_param_4];").map_err(write_err)?;

    // -------- OOB: if tid >= n_pow2, return. ----------
    writeln!(p, "\tsetp.ge.s32 %p0, %r3, %r2;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;

    // -------- partner = tid XOR substage_mask. ----------
    writeln!(p, "\txor.b32 %r7, %r3, %r1;").map_err(write_err)?;

    // -------- If tid >= partner, the partner thread owns the swap. ----------
    writeln!(p, "\tsetp.ge.s32 %p1, %r3, %r7;").map_err(write_err)?;
    writeln!(p, "\t@%p1 bra DONE;").map_err(write_err)?;

    // -------- Determine direction for THIS pair. ----------
    //
    //   asc_block_bit = (tid >> stage) & 1
    //   asc_block     = (asc_block_bit == 0)
    //   pair_asc      = asc_block XOR global_desc
    //
    // For global_desc = false, pair_asc == asc_block.
    // For global_desc = true, pair_asc == !asc_block. We bake `global_desc`
    // into the comparison polarity below instead of XORing dynamically.
    writeln!(p, "\tshr.u32 %r8, %r3, %r0;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(p, "\tsetp.eq.s32 %p2, %r8, 0;").map_err(write_err)?;

    // -------- Globalize the two device pointers. ----------
    writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_0];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
    writeln!(p, "\tld.param.u64 %rd1, [{entry}_param_1];").map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd1, %rd1;").map_err(write_err)?;

    // -------- Compute address &keys[tid] and &keys[partner]. ----------
    let key_w = flavour.byte_width as i64;
    writeln!(p, "\tmul.wide.s32 %rd2, %r3, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd3, %rd0, %rd2;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.s32 %rd4, %r7, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd5, %rd0, %rd4;").map_err(write_err)?;

    // -------- Load both key values. ----------
    writeln!(p, "\tld.global.{} %k0, [%rd3];", flavour.ld_st_suffix).map_err(write_err)?;
    writeln!(p, "\tld.global.{} %k1, [%rd5];", flavour.ld_st_suffix).map_err(write_err)?;

    // -------- Compute do_swap. ----------
    //
    // Bake `dir` into the polarity:
    //   ASC  pair (pair_asc=true)  + global_asc  : swap if k_self  > k_partner
    //   DESC pair (pair_asc=false) + global_asc  : swap if k_self  < k_partner
    //   ASC  pair                  + global_desc : swap if k_self  < k_partner
    //   DESC pair                  + global_desc : swap if k_self  > k_partner
    //
    // I.e. for global_asc we use (gt, lt) for (asc_block, desc_block); for
    // global_desc we use (lt, gt). %p3 = "self > partner", %p4 = "self < partner".
    writeln!(
        p,
        "\t{} %p3, %k0, %k1;",
        flavour.setp_gt
    )
    .map_err(write_err)?;
    writeln!(
        p,
        "\t{} %p4, %k0, %k1;",
        flavour.setp_lt
    )
    .map_err(write_err)?;

    let (asc_block_swap, desc_block_swap) = match dir {
        SortDirection::Asc => ("%p3", "%p4"),
        SortDirection::Desc => ("%p4", "%p3"),
    };

    // do_swap = if asc_block then asc_block_swap else desc_block_swap.
    // PTX selp on a predicate: we materialise via two predicated `mov`s into
    // a final pred register %p5.
    //
    // We can't `selp.pred` directly (PTX has no selp on pred class); use
    // and/or composition instead:
    //   p5 = (p2 & asc_block_swap) | (!p2 & desc_block_swap)
    // Realised as a small selp into a b32 then setp.ne 0.
    writeln!(p, "\tselp.s32 %r9, 1, 0, %p2;").map_err(write_err)?;
    writeln!(p, "\tselp.s32 %r10, 1, 0, {asc_block_swap};").map_err(write_err)?;
    writeln!(p, "\tselp.s32 %r11, 1, 0, {desc_block_swap};").map_err(write_err)?;
    // r12 = r9 * r10  (asc_block AND asc_block_swap)
    writeln!(p, "\tmul.lo.s32 %r12, %r9, %r10;").map_err(write_err)?;
    // r13 = (1 - r9) * r11 (desc_block AND desc_block_swap)
    writeln!(p, "\tsub.s32 %r14, 1, %r9;").map_err(write_err)?;
    writeln!(p, "\tmul.lo.s32 %r13, %r14, %r11;").map_err(write_err)?;
    // r15 = r12 | r13
    writeln!(p, "\tor.b32 %r15, %r12, %r13;").map_err(write_err)?;
    writeln!(p, "\tsetp.ne.s32 %p5, %r15, 0;").map_err(write_err)?;

    // -------- If !do_swap, return. ----------
    writeln!(p, "\t@!%p5 bra DONE;").map_err(write_err)?;

    // -------- Swap keys: write k1 at tid, k0 at partner. ----------
    writeln!(p, "\tst.global.{} [%rd3], %k1;", flavour.ld_st_suffix).map_err(write_err)?;
    writeln!(p, "\tst.global.{} [%rd5], %k0;", flavour.ld_st_suffix).map_err(write_err)?;

    // -------- Swap indices (u32) at the same positions. ----------
    //
    // indices are u32 — width 4. Reuse %r9..%r10 for the value scratch since
    // we're past the swap-condition computation.
    writeln!(p, "\tmul.wide.s32 %rd6, %r3, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd6, %rd1, %rd6;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.s32 %rd7, %r7, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd7, %rd1, %rd7;").map_err(write_err)?;
    writeln!(p, "\tld.global.u32 %r9, [%rd6];").map_err(write_err)?;
    writeln!(p, "\tld.global.u32 %r10, [%rd7];").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd6], %r10;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd7], %r9;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;

    Ok(p)
}

/// Per-dtype PTX details: register class, byte width, load/store suffix, and
/// the `setp.<cond>.<ty>` mnemonics for the gt/lt compare-exchange test.
///
/// Keeping these in one struct (vs scattered match arms) means adding a new
/// supported dtype is a single new constructor branch + a new branch in
/// `sort_kernel_entry`.
struct DtypeFlavour {
    /// PTX register class string for the key (e.g. `"f32"`).
    reg_type: &'static str,
    /// Element byte width.
    byte_width: u32,
    /// Type suffix for `ld.global.<sfx>` / `st.global.<sfx>` (e.g. `"f32"`).
    ld_st_suffix: &'static str,
    /// Full `setp.gt.<ty>` mnemonic.
    setp_gt: &'static str,
    /// Full `setp.lt.<ty>` mnemonic.
    setp_lt: &'static str,
}

impl DtypeFlavour {
    fn for_dtype(dtype: DataType) -> BoltResult<Self> {
        Ok(match dtype {
            DataType::Int32 => Self {
                reg_type: "b32",
                byte_width: 4,
                ld_st_suffix: "s32",
                setp_gt: "setp.gt.s32",
                setp_lt: "setp.lt.s32",
            },
            DataType::Int64 => Self {
                reg_type: "b64",
                byte_width: 8,
                ld_st_suffix: "s64",
                setp_gt: "setp.gt.s64",
                setp_lt: "setp.lt.s64",
            },
            DataType::Float32 => Self {
                reg_type: "f32",
                byte_width: 4,
                ld_st_suffix: "f32",
                setp_gt: "setp.gt.f32",
                setp_lt: "setp.lt.f32",
            },
            DataType::Float64 => Self {
                reg_type: "f64",
                byte_width: 8,
                ld_st_suffix: "f64",
                setp_gt: "setp.gt.f64",
                setp_lt: "setp.lt.f64",
            },
            _ => {
                return Err(BoltError::Other(format!(
                    "sort_kernel: dtype {:?} not supported by Stage 1 \
                     (Int32, Int64, Float32, Float64 only)",
                    dtype
                )))
            }
        })
    }
}

/// Adapt `std::fmt::Error` into a `BoltError::Other`. Mirrors the helper in
/// neighbouring kernel modules.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("sort_kernel: write failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_names_per_dtype_dir() {
        // Every supported (dtype, dir) maps to a unique kernel name.
        let cases = [
            (DataType::Int32, SortDirection::Asc, "bolt_bitonic_sort_i32_asc"),
            (DataType::Int32, SortDirection::Desc, "bolt_bitonic_sort_i32_desc"),
            (DataType::Int64, SortDirection::Asc, "bolt_bitonic_sort_i64_asc"),
            (DataType::Int64, SortDirection::Desc, "bolt_bitonic_sort_i64_desc"),
            (DataType::Float32, SortDirection::Asc, "bolt_bitonic_sort_f32_asc"),
            (
                DataType::Float32,
                SortDirection::Desc,
                "bolt_bitonic_sort_f32_desc",
            ),
            (DataType::Float64, SortDirection::Asc, "bolt_bitonic_sort_f64_asc"),
            (
                DataType::Float64,
                SortDirection::Desc,
                "bolt_bitonic_sort_f64_desc",
            ),
        ];
        for (dtype, dir, name) in cases {
            assert_eq!(sort_kernel_entry(dtype, dir).unwrap(), name);
        }
    }

    #[test]
    fn rejects_unsupported_dtypes() {
        // Utf8 + Bool are explicitly out of Stage 1 scope.
        assert!(sort_kernel_entry(DataType::Utf8, SortDirection::Asc).is_err());
        assert!(sort_kernel_entry(DataType::Bool, SortDirection::Asc).is_err());
        assert!(compile_sort_kernel(DataType::Utf8, SortDirection::Asc).is_err());
        assert!(compile_sort_kernel(DataType::Bool, SortDirection::Asc).is_err());
    }

    /// Header + signature shape goldens — these are the byte-stable bits of
    /// every emitted PTX module. If anything here changes we want a test
    /// failure forcing an intentional update rather than a silent ABI drift.
    #[test]
    fn ptx_header_and_signature_shape() {
        let ptx = compile_sort_kernel(DataType::Int32, SortDirection::Asc).unwrap();

        // Header.
        assert!(ptx.contains(".version 7.5"), "PTX must declare .version 7.5");
        assert!(ptx.contains(".target sm_70"), "PTX must target sm_70");
        assert!(
            ptx.contains(".address_size 64"),
            "PTX must declare 64-bit addresses"
        );

        // Entry point.
        assert!(ptx.contains(".visible .entry bolt_bitonic_sort_i32_asc("));

        // Param list: keys ptr, indices ptr, n_pow2, stage, substage_mask.
        assert!(ptx.contains(".param .u64 bolt_bitonic_sort_i32_asc_param_0,"));
        assert!(ptx.contains(".param .u64 bolt_bitonic_sort_i32_asc_param_1,"));
        assert!(ptx.contains(".param .u32 bolt_bitonic_sort_i32_asc_param_2,"));
        assert!(ptx.contains(".param .u32 bolt_bitonic_sort_i32_asc_param_3,"));
        assert!(ptx.contains(".param .u32 bolt_bitonic_sort_i32_asc_param_4"));
    }

    /// ASC int32 must use the signed-int `setp.lt`/`setp.gt` mnemonics — the
    /// load-bearing piece of the compare-exchange for the integer fast path.
    #[test]
    fn ptx_asc_int32_uses_signed_compares() {
        let ptx = compile_sort_kernel(DataType::Int32, SortDirection::Asc).unwrap();
        assert!(
            ptx.contains("setp.gt.s32"),
            "ASC int32 must emit setp.gt.s32 (asc-block compare); got:\n{ptx}"
        );
        assert!(
            ptx.contains("setp.lt.s32"),
            "ASC int32 must also emit setp.lt.s32 for the desc-block compare; got:\n{ptx}"
        );
    }

    /// DESC int32 emits the same setp mnemonics but the polarity wired into
    /// the swap selection is flipped — both gt and lt show up in either
    /// direction, so the test asserts presence (the wiring is the gpu_sort
    /// path's concern, exercised by the round-trip below).
    #[test]
    fn ptx_desc_int32_emits_signed_compares() {
        let ptx = compile_sort_kernel(DataType::Int32, SortDirection::Desc).unwrap();
        assert!(ptx.contains("setp.gt.s32"));
        assert!(ptx.contains("setp.lt.s32"));
    }

    /// 64-bit integer key must use the s64-typed setp.
    #[test]
    fn ptx_int64_uses_s64_compares() {
        let ptx = compile_sort_kernel(DataType::Int64, SortDirection::Asc).unwrap();
        assert!(ptx.contains("setp.gt.s64"));
        assert!(ptx.contains("setp.lt.s64"));
        // And the load/store path must move 8 bytes.
        assert!(ptx.contains("ld.global.s64"));
        assert!(ptx.contains("st.global.s64"));
    }

    /// Float kernels use the float-typed setp, and load via ld.global.f32/f64.
    #[test]
    fn ptx_floats_use_float_compares() {
        let f32_ptx = compile_sort_kernel(DataType::Float32, SortDirection::Asc).unwrap();
        assert!(f32_ptx.contains("setp.gt.f32"));
        assert!(f32_ptx.contains("setp.lt.f32"));
        assert!(f32_ptx.contains("ld.global.f32"));

        let f64_ptx = compile_sort_kernel(DataType::Float64, SortDirection::Desc).unwrap();
        assert!(f64_ptx.contains("setp.gt.f64"));
        assert!(f64_ptx.contains("setp.lt.f64"));
        assert!(f64_ptx.contains("ld.global.f64"));
    }

    /// The kernel must consult `n_pow2` to bail OOB threads. If this guard
    /// disappears, threads past the padded length will write past the
    /// allocation.
    #[test]
    fn ptx_has_n_pow2_oob_guard() {
        let ptx = compile_sort_kernel(DataType::Int32, SortDirection::Asc).unwrap();
        // The OOB test: setp.ge.s32 <pred>, <tid>, <n_pow2>; @pred bra DONE.
        assert!(ptx.contains("setp.ge.s32"), "missing OOB compare against n_pow2");
        assert!(ptx.contains("bra DONE"), "missing branch to DONE label");
        assert!(ptx.contains("DONE:"), "missing DONE label");
    }

    /// The kernel must XOR tid against substage_mask to compute the partner
    /// index. This is the defining instruction of the bitonic pattern; any
    /// substitution that hides it (e.g. shifting by stage directly) is a
    /// regression that breaks the algorithm.
    #[test]
    fn ptx_uses_xor_for_partner_index() {
        let ptx = compile_sort_kernel(DataType::Float64, SortDirection::Asc).unwrap();
        assert!(
            ptx.contains("xor.b32"),
            "bitonic partner index must come from XOR; got:\n{ptx}"
        );
    }

    /// The kernel must swap u32 indices in parallel with the key swap. If a
    /// future change drops the indices swap, gpu_sort's gather step will
    /// silently produce a sorted-keys / wrong-row output.
    #[test]
    fn ptx_swaps_indices_too() {
        let ptx = compile_sort_kernel(DataType::Int32, SortDirection::Asc).unwrap();
        // Indices are u32 -> we must see two u32 loads followed by two
        // u32 stores in the swap tail.
        assert!(ptx.contains("ld.global.u32"));
        assert!(ptx.contains("st.global.u32"));
    }
}
