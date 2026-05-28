// SPDX-License-Identifier: Apache-2.0

//! PTX codegen for a **bitonic sort** kernel.
//!
//! Backs `crate::exec::gpu_sort`: ORDER BY's GPU fast path. The host-side
//! executor allocates a padded keys buffer + parallel indices buffer of size
//! `n_pow2 = next_power_of_two(n_rows)`, fills the tail with a sentinel
//! (`+INF`-style for ASC / `-INF`-style for DESC), then launches this kernel
//! once per (stage, substage) pair. After the sort, the first `n_rows` indices
//! (or the last `n_rows`, depending on direction + sentinel choice) are the
//! permutation to apply to every output column.
//!
//! ## Stage 2 — multi-key, NULL-aware, in-block shmem variant
//!
//! The Stage-1 single-key kernel was retired in Stage 4 — `try_gpu_sort`
//! routes every supported case through the multi-key driver
//! ([`compile_sort_kernel_spec`]), and a single-key sort is just a
//! [`SortKernelSpec`] with one entry in `keys`. The PTX-shape golden tests
//! were migrated to drive `compile_sort_kernel_spec` directly. Stage 2
//! emits:
//!
//! - A **multi-key (lexicographic) comparator**. Up to four sort keys, each
//!   with its own dtype, direction (ASC/DESC), and NULL placement
//!   (NULLS FIRST/LAST). The comparator emits one `setp.eq` + one `setp.lt`
//!   per key and branches to swap-or-keep on the first non-equal column —
//!   i.e. no full materialization of the lexicographic rank, just early
//!   exit. See [`KeyDesc`], [`SortKernelSpec`], and [`compile_sort_kernel_spec`].
//!
//! - A **NULL-aware compare**. Each key may have an optional Arrow-style
//!   packed bit validity buffer; if provided, the comparator reads the bit
//!   for `tid` and `partner`, treats NULL==NULL as equal, and routes a NULL
//!   ahead of or behind a non-NULL value per the per-key `nulls_first`
//!   flag. The flag is baked into the PTX at codegen so the inner branch is
//!   a single predicated jump.
//!
//! - An **in-block shared-memory variant**. When `n_pow2 <= block_size` the
//!   whole sort fits in shared memory. The shmem variant loads keys + index
//!   into `__shared__`, walks all `log2(n) * (log2(n)+1) / 2` substages with
//!   `__syncthreads()` between them, then writes back — collapsing ~36
//!   launches (for n=256) into one. See `SortLayout::Shmem`.
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

/// Per-dtype PTX details: register class, byte width, load/store suffix, and
/// the `setp.<cond>.<ty>` mnemonics for the gt/lt compare-exchange test.
///
/// Keeping these in one struct (vs scattered match arms) means adding a new
/// supported dtype is a single new constructor branch.
///
/// Stage 4 dropped the `reg_type` field — its only consumer was the
/// retired Stage-1 single-key `compile_sort_kernel`, which declared a
/// shared register pool typed by the key's register class. The multi-key
/// emitter declares per-dtype register pools via `key_regs()` instead.
struct DtypeFlavour {
    /// Element byte width.
    byte_width: u32,
    /// Type suffix for `ld.global.<sfx>` / `st.global.<sfx>` (e.g. `"f32"`).
    ld_st_suffix: &'static str,
    /// Full `setp.gt.<ty>` mnemonic.
    setp_gt: &'static str,
    /// Full `setp.lt.<ty>` mnemonic.
    setp_lt: &'static str,
    /// Full `setp.eq.<ty>` mnemonic (used by the multi-key lex comparator —
    /// "if a == b, fall through to the next key").
    setp_eq: &'static str,
}

impl DtypeFlavour {
    fn for_dtype(dtype: DataType) -> BoltResult<Self> {
        Ok(match dtype {
            DataType::Int32 => Self {
                byte_width: 4,
                ld_st_suffix: "s32",
                setp_gt: "setp.gt.s32",
                setp_lt: "setp.lt.s32",
                setp_eq: "setp.eq.s32",
            },
            DataType::Int64 => Self {
                byte_width: 8,
                ld_st_suffix: "s64",
                setp_gt: "setp.gt.s64",
                setp_lt: "setp.lt.s64",
                setp_eq: "setp.eq.s64",
            },
            DataType::Float32 => Self {
                byte_width: 4,
                ld_st_suffix: "f32",
                setp_gt: "setp.gt.f32",
                setp_lt: "setp.lt.f32",
                setp_eq: "setp.eq.f32",
            },
            DataType::Float64 => Self {
                byte_width: 8,
                ld_st_suffix: "f64",
                setp_gt: "setp.gt.f64",
                setp_lt: "setp.lt.f64",
                setp_eq: "setp.eq.f64",
            },
            // Stage 3: Bool keys are widened to i32 on the device — the host
            // uploads `0u8`/`1u8` values and we treat them as i32 0/1 (the
            // top 24 bits are unused; load is .u8 but the register is b32).
            DataType::Bool => Self {
                byte_width: 1,
                ld_st_suffix: "u8",
                setp_gt: "setp.gt.s32",
                setp_lt: "setp.lt.s32",
                setp_eq: "setp.eq.s32",
            },
            _ => {
                return Err(BoltError::Other(format!(
                    "sort_kernel: dtype {:?} not supported \
                     (Int32/Int64/Float32/Float64/Bool — Utf8 only via the \
                     dictionary-index adapter in gpu_sort.rs)",
                    dtype
                )))
            }
        })
    }
}

// ============================================================================
// Stage 2: multi-key + NULL-aware + in-block shmem variant.
// ============================================================================

/// Hard upper bound on sort-key count for the Stage-2/3 emitter.
///
/// **Stage 3:** the historical Stage-2 cap of 4 is lifted — the real
/// constraint is per-thread register pressure, not the literal key count.
/// `MAX_SORT_KEYS` is kept as a safety ceiling at 12 (well below any plausible
/// SQL `ORDER BY` arity), and the per-spec validator additionally rejects
/// kernels whose tallied key-register budget exceeds [`SM70_KEY_REG_BUDGET`].
///
/// Practical width — sm_70 (Volta) gives every thread 64 registers before
/// spills bite occupancy. We reserve ~32 for the bitonic scaffold (tid /
/// partner / addresses / predicate scratch) and budget the remaining 32 for
/// keys. Per-dtype cost:
///
///   - `i32`/`f32`        : 2 regs/key (self + partner, b32)
///   - `i64`/`f64`        : 4 regs/key (self + partner, b64 = 2× b32)
///   - `bool`             : 2 regs/key (loaded into b32 like i32)
///
/// So the comparator handles up to **16 i32-style keys**, **8 i64-style
/// keys**, or any mix that tallies to <=32 b32-register equivalents. The
/// soft cap of 12 covers every observed SQL workload while keeping the
/// per-key-register pool [`MAX_SORT_KEYS * 2`] declarations below 32.
pub const MAX_SORT_KEYS: usize = 12;

/// Register budget (in b32-equivalent units) per thread for the key
/// comparator on sm_70. Anything beyond this risks register spills which
/// halve occupancy on the bitonic kernel.
///
/// `64 total regs - 32 reserved for the bitonic scaffold = 32 left for keys`.
pub const SM70_KEY_REG_BUDGET: u32 = 32;

/// Per-key b32-register cost. b64 dtypes count twice. Used by the validator
/// to enforce [`SM70_KEY_REG_BUDGET`] before the PTX is even compiled.
pub fn key_reg_cost(dtype: DataType) -> u32 {
    match dtype {
        DataType::Int32 | DataType::Float32 | DataType::Bool => 2,
        DataType::Int64 | DataType::Float64 => 4,
        // Utf8 is dictionary-decoded into i32/i64 indices in gpu_sort; the
        // dtype that reaches the kernel is always one of the numeric ones.
        DataType::Utf8 => 4,
    }
}

/// One sort key in a (possibly) multi-key bitonic sort.
///
/// `dtype` is the column type (only the Stage-1 fixed-width set is allowed).
/// `ascending` baked into the kernel polarity; `nulls_first` baked into the
/// NULL-vs-non-NULL branch direction. `nullable` says whether the host will
/// pass a validity bitmap pointer for this key. If `false`, the comparator
/// skips the validity-load fast-path entirely and the host may pass a null
/// pointer for the slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeyDesc {
    /// Key column dtype.
    pub dtype: DataType,
    /// Direction for this key alone.
    pub direction: SortDirection,
    /// Where NULLs sort relative to non-NULLs for this key. Ignored if
    /// `nullable == false`.
    pub nulls_first: bool,
    /// True if the host will provide a validity bitmap for this key. False
    /// for keys known not to contain NULLs (the comparator skips the branch).
    pub nullable: bool,
}

/// Which kernel layout to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SortLayout {
    /// One launch per (stage, substage) pair — works for any `n_pow2`.
    MultiLaunch,
    /// All log²n substages collapsed into a single kernel launch with the
    /// keys held in shared memory. Only valid when `n_pow2 <= block_size`.
    Shmem,
}

/// Full spec for emitting a Stage 2 bitonic sort kernel. Distilled from the
/// host gate so the PTX emitter has everything it needs without re-deriving
/// branches from runtime parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SortKernelSpec {
    /// Sort keys in lexicographic order — `keys[0]` is the major key,
    /// `keys[n-1]` is the minor tiebreaker. Length must be in `1..=MAX_SORT_KEYS`.
    pub keys: Vec<KeyDesc>,
    /// Kernel layout (multi-launch vs in-block shmem).
    pub layout: SortLayout,
    /// For `Shmem`: the upper bound on `n_pow2` baked into the shared-memory
    /// allocation. Ignored for `MultiLaunch`. Must be a power of two and equal
    /// the runtime n_pow2 at launch time.
    pub shmem_n_pow2: u32,
}

impl SortKernelSpec {
    /// Validate basic invariants. Returns `Err` if the spec is unbuildable.
    fn validate(&self) -> BoltResult<()> {
        if self.keys.is_empty() {
            return Err(BoltError::Other(
                "sort_kernel: SortKernelSpec.keys must have at least 1 entry".into(),
            ));
        }
        if self.keys.len() > MAX_SORT_KEYS {
            return Err(BoltError::Other(format!(
                "sort_kernel: SortKernelSpec.keys has {} entries; hard cap is {} \
                 (sm_70 register budget — practical SQL ORDER BY rarely exceeds 8)",
                self.keys.len(),
                MAX_SORT_KEYS
            )));
        }
        let mut reg_tally: u32 = 0;
        for (i, k) in self.keys.iter().enumerate() {
            // Reject unsupported dtypes early; reuses the flavour table as
            // the single source of truth for what's emittable.
            DtypeFlavour::for_dtype(k.dtype)
                .map_err(|e| BoltError::Other(format!("sort_kernel: key[{i}]: {e}")))?;
            reg_tally = reg_tally.saturating_add(key_reg_cost(k.dtype));
        }
        if reg_tally > SM70_KEY_REG_BUDGET {
            return Err(BoltError::Other(format!(
                "sort_kernel: keys tally {} b32-register equivalents; sm_70 budget \
                 is {} (drop a key or split into a multi-pass sort)",
                reg_tally, SM70_KEY_REG_BUDGET
            )));
        }
        if matches!(self.layout, SortLayout::Shmem) {
            if !self.shmem_n_pow2.is_power_of_two() {
                return Err(BoltError::Other(format!(
                    "sort_kernel: Shmem layout requires shmem_n_pow2 power-of-two; got {}",
                    self.shmem_n_pow2
                )));
            }
            if self.shmem_n_pow2 > SORT_BLOCK_SIZE {
                return Err(BoltError::Other(format!(
                    "sort_kernel: Shmem layout requires shmem_n_pow2 <= block_size ({}); got {}",
                    SORT_BLOCK_SIZE, self.shmem_n_pow2
                )));
            }
        }
        Ok(())
    }
}

/// Build a stable, content-addressed kernel name from a [`SortKernelSpec`].
///
/// The name encodes layout + per-key (dtype, dir, nullable, nulls_first), so
/// the PTX module cache can key on the name and two specs that differ in any
/// observable way produce different modules.
pub fn sort_kernel_entry_spec(spec: &SortKernelSpec) -> BoltResult<String> {
    spec.validate()?;
    let layout_tag = match spec.layout {
        SortLayout::MultiLaunch => "ml",
        SortLayout::Shmem => "sh",
    };
    let mut s = format!("bolt_bitonic_sort_{}", layout_tag);
    if matches!(spec.layout, SortLayout::Shmem) {
        // Bake n_pow2 into the entry name so a 128-element sort and a 256-
        // element sort don't collide in the module cache. Both legal under
        // Shmem layout; just need distinct PTX.
        let _ = write!(&mut s, "_n{}", spec.shmem_n_pow2);
    }
    for k in &spec.keys {
        let dty = match k.dtype {
            DataType::Int32 => "i32",
            DataType::Int64 => "i64",
            DataType::Float32 => "f32",
            DataType::Float64 => "f64",
            DataType::Bool => "b",
            _ => unreachable!("validate() rejects other dtypes"),
        };
        let dir = match k.direction {
            SortDirection::Asc => "a",
            SortDirection::Desc => "d",
        };
        let nulls = if k.nullable {
            if k.nulls_first {
                "nf"
            } else {
                "nl"
            }
        } else {
            "nn" // non-nullable
        };
        let _ = write!(&mut s, "_{}{}{}", dty, dir, nulls);
    }
    Ok(s)
}

/// Compile a Stage 2 PTX module from `spec`.
///
/// Layout-specific differences:
///
/// - `MultiLaunch`: one (stage, substage) pair per launch. ABI:
///   `(keys0_ptr, validity0_ptr, keys1_ptr, validity1_ptr, ..., indices_ptr,
///   n_pow2, stage, substage_mask)`. The number of `(keysK, validityK)`
///   slots equals `MAX_SORT_KEYS`; unused slots are nullable-skipped at
///   codegen and the host passes null pointers.
///
/// - `Shmem`: one launch total. The kernel loads everything into shared
///   memory, runs every substage in-kernel with `bar.sync 0`, and writes
///   back. ABI: same as MultiLaunch minus `stage` and `substage_mask`
///   (the kernel walks all stages internally).
pub fn compile_sort_kernel_spec(spec: &SortKernelSpec) -> BoltResult<String> {
    spec.validate()?;
    let entry = sort_kernel_entry_spec(spec)?;

    let mut p = String::new();
    writeln!(p, "{PTX_VERSION}").map_err(write_err)?;
    writeln!(p, "{PTX_TARGET}").map_err(write_err)?;
    writeln!(p, "{PTX_ADDRESS_SIZE}").map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    match spec.layout {
        SortLayout::MultiLaunch => emit_multikey_multilaunch(&mut p, &entry, spec)?,
        SortLayout::Shmem => emit_multikey_shmem(&mut p, &entry, spec)?,
    }

    Ok(p)
}

/// Emit the multi-key, multi-launch bitonic kernel. One launch per substage.
///
/// ABI laid out so every (key, validity) pair sits adjacent in the param
/// list — the host can pass `[k0, v0, k1, v1, k2, v2, k3, v3, indices,
/// n_pow2, stage, mask]` directly. Unused key slots are null pointers; the
/// kernel only loads slots `0..spec.keys.len()`.
fn emit_multikey_multilaunch(
    p: &mut String,
    entry: &str,
    spec: &SortKernelSpec,
) -> BoltResult<()> {
    // -- Signature ----------------------------------------------------
    //
    // Stage 3 ABI extension: a trailing pointer (`is_padded_ptr`) is added
    // after the indices pointer. It is a packed-bit array (1 bit per row,
    // LSB-first, Arrow-style packed u8). Bit=1 means "this row is part of
    // the padding to next-pow2; route it past every real row regardless of
    // its sentinel value". This eliminates the Stage-2 silent-row-drop bug
    // when real keys equal the sentinel (e.g. `i32::MAX` as a legitimate
    // value with the ASC `+INF`-style pad).
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    let n_ptr_params = MAX_SORT_KEYS * 2 + 2; // (k,v)*MAX + indices + is_padded
    let total_params = n_ptr_params + 3; // + n_pow2 + stage + mask
    for i in 0..n_ptr_params {
        // All pointers are .u64.
        writeln!(p, "\t.param .u64 {entry}_param_{i},").map_err(write_err)?;
    }
    // n_pow2, stage, substage_mask — three u32s.
    writeln!(p, "\t.param .u32 {entry}_param_{}, ", total_params - 3).map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_{}, ", total_params - 2).map_err(write_err)?;
    writeln!(p, "\t.param .u32 {entry}_param_{}", total_params - 1).map_err(write_err)?;
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    // -- Register declarations ---------------------------------------
    //
    // Predicates: %p0 oob, %p1 paired-skip, %p2 asc_block, %p3 do_swap.
    // Named preds for the lex/null branches are declared individually so
    // PTX can resolve them as identifiers (the `%p<N>` shorthand only
    // creates %p0..%p<N-1>).
    //
    // b32: %r0 stage, %r1 mask, %r2 n_pow2, %r3 tid, %r4-%r9 work,
    //      %r10 do_swap_flag (1 = swap, 0 = no, -1 = undecided continue).
    // b64: %rd0..%rd47 ptrs + addr scratch (need 2 ptrs/key + 2 validity).
    writeln!(p, "\t.reg .pred %p<8>;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_eq;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_gt;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_lt;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_both_null;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_sn;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_pn;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_sn2;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_pn2;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_self_null;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_partner_null;").map_err(write_err)?;
    // Stage 3 padded-routing predicates.
    writeln!(p, "\t.reg .pred %p_self_pad;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_partner_pad;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_both_pad;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_sp1;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_sp2;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_pp1;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_pp2;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32 %r<32>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %rd<48>;").map_err(write_err)?;
    // Key registers: 2 per key (self/partner), max across all dtype reg
    // classes — emit a small pool per width. Bool keys share the i32 pool.
    writeln!(p, "\t.reg .b32 %ki32<{}>;", MAX_SORT_KEYS * 2).map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %ki64<{}>;", MAX_SORT_KEYS * 2).map_err(write_err)?;
    writeln!(p, "\t.reg .f32 %kf32<{}>;", MAX_SORT_KEYS * 2).map_err(write_err)?;
    writeln!(p, "\t.reg .f64 %kf64<{}>;", MAX_SORT_KEYS * 2).map_err(write_err)?;

    // -- tid -----------------------------------------------------------
    writeln!(p, "\tmov.u32 %r4, %ctaid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r5, %ntid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.u32 %r6, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmad.lo.s32 %r3, %r4, %r5, %r6;").map_err(write_err)?;

    // -- n_pow2, stage, substage_mask ---------------------------------
    let p_n_pow2 = MAX_SORT_KEYS * 2 + 2;
    let p_stage = p_n_pow2 + 1;
    let p_mask = p_n_pow2 + 2;
    writeln!(p, "\tld.param.u32 %r2, [{entry}_param_{}];", p_n_pow2).map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r0, [{entry}_param_{}];", p_stage).map_err(write_err)?;
    writeln!(p, "\tld.param.u32 %r1, [{entry}_param_{}];", p_mask).map_err(write_err)?;

    // -- OOB + paired-skip ---------------------------------------------
    writeln!(p, "\tsetp.ge.u32 %p0, %r3, %r2;").map_err(write_err)?;
    writeln!(p, "\t@%p0 bra DONE;").map_err(write_err)?;
    writeln!(p, "\txor.b32 %r7, %r3, %r1;").map_err(write_err)?;
    writeln!(p, "\tsetp.ge.u32 %p1, %r3, %r7;").map_err(write_err)?;
    writeln!(p, "\t@%p1 bra DONE;").map_err(write_err)?;

    // asc_block_bit = (tid >> stage) & 1; %p2 = (asc_block_bit == 0)
    writeln!(p, "\tshr.u32 %r8, %r3, %r0;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r8, %r8, 1;").map_err(write_err)?;
    writeln!(p, "\tsetp.eq.s32 %p2, %r8, 0;").map_err(write_err)?;

    // -- Lex compare: walk keys; on first non-equal write %r10=1/0 and
    //    jump to DECIDED. If all keys equal we fall through to DECIDED
    //    with %r10=0 (no swap).
    writeln!(p, "\tmov.b32 %r10, 0;").map_err(write_err)?;

    // -- Stage 3: padded-row pre-routing -------------------------------
    //
    // Read `is_padded[self]` and `is_padded[partner]` from the trailing
    // packed-bit pointer (Arrow-style: byte i>>3, bit i&7, 1=padded).
    // Three outcomes:
    //   - both padded     -> tied, fall through to lex compare
    //   - self padded only -> route to global END (swap iff in ASC block)
    //   - partner padded only -> route partner to global END (swap iff DESC block)
    // This dodges the Stage-2 sentinel-tie corruption: real `i32::MAX`
    // values (the ASC sentinel) no longer fight with padded slots over the
    // last position; the explicit padded bit wins.
    let padded_param_idx = MAX_SORT_KEYS * 2 + 1;
    emit_padded_route_global(p, entry, padded_param_idx)?;

    let indices_param_idx = MAX_SORT_KEYS * 2;
    for (ki, k) in spec.keys.iter().enumerate() {
        emit_key_compare(p, entry, ki, k, /*shmem=*/ false)?;
    }
    // After last key: if we got here all keys equal, no swap. Fall through.
    writeln!(p, "\tbra DECIDED;").map_err(write_err)?;

    // SWAP_YES / SWAP_NO labels jumped to from emit_key_compare.
    // emit_key_compare emits per-key jumps to `KEY_<ki>_NEXT` on equality and
    // to `DECIDED` after setting %r10. So the per-key blocks end with bra
    // DECIDED. We materialise DECIDED here.
    writeln!(p, "DECIDED:").map_err(write_err)?;
    // do_swap predicate from %r10
    writeln!(p, "\tsetp.ne.s32 %p3, %r10, 0;").map_err(write_err)?;
    writeln!(p, "\t@!%p3 bra DONE;").map_err(write_err)?;

    // -- Perform the swap of every active key + indices ---------------
    for (ki, k) in spec.keys.iter().enumerate() {
        emit_key_swap(p, entry, ki, k)?;
    }
    // Indices swap (u32 at indices_ptr).
    writeln!(
        p,
        "\tld.param.u64 %rd40, [{entry}_param_{}];",
        indices_param_idx
    )
    .map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd40, %rd40;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd41, %r3, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd41, %rd40, %rd41;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd42, %r7, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd42, %rd40, %rd42;").map_err(write_err)?;
    writeln!(p, "\tld.global.u32 %r11, [%rd41];").map_err(write_err)?;
    writeln!(p, "\tld.global.u32 %r12, [%rd42];").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd41], %r12;").map_err(write_err)?;
    writeln!(p, "\tst.global.u32 [%rd42], %r11;").map_err(write_err)?;

    writeln!(p, "DONE:").map_err(write_err)?;
    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;
    Ok(())
}

/// Emit a single key's compare block (null-aware if `k.nullable`).
///
/// Strategy per key:
///
/// 1. If nullable: load validity bits for self & partner.
///    - both NULL -> equal, jump to `KEY_<ki>_NEXT`.
///    - self NULL, partner not -> route per `nulls_first` (ASC). Set %r10
///      to swap_yes if (asc_block XOR self_first_for_dir) else no, then bra
///      DECIDED.
///    - vice versa.
///    - both non-NULL -> fall through to value compare.
/// 2. Load key values for self & partner.
/// 3. `setp.eq` -> if equal jump to KEY_<ki>_NEXT (try next key).
/// 4. `setp.lt` -> compute do_swap based on:
///    `do_swap = (asc_block == k.direction==Asc) ? (self > partner) : (self < partner)`
///    Bake the per-key direction into the polarity so the inner branch is
///    one predicated path.
fn emit_key_compare(
    p: &mut String,
    entry: &str,
    ki: usize,
    k: &KeyDesc,
    shmem: bool,
) -> BoltResult<()> {
    let flavour = DtypeFlavour::for_dtype(k.dtype)?;
    let key_param_idx = ki * 2;
    let valid_param_idx = ki * 2 + 1;
    let key_w = flavour.byte_width as i64;

    // Per-key labels, unique by key index.
    let lbl_next = format!("KEY_{}_NEXT", ki);
    let lbl_swap_yes = format!("KEY_{}_SWAP_YES", ki);
    let lbl_swap_no = format!("KEY_{}_SWAP_NO", ki);

    writeln!(p, "// ---- compare key {} (dtype={:?}, dir={:?}, nullable={}, nulls_first={}) ----",
             ki, k.dtype, k.direction, k.nullable, k.nulls_first).map_err(write_err)?;

    if k.nullable {
        // Validity bitmap: Arrow-format, packed u8 with bit `i & 7` of byte
        // `i >> 3`. 1 = valid, 0 = null. Load both bits.
        writeln!(p, "\tld.param.u64 %rd20, [{entry}_param_{}];", valid_param_idx)
            .map_err(write_err)?;
        writeln!(p, "\tcvta.to.global.u64 %rd20, %rd20;").map_err(write_err)?;
        // self bit
        writeln!(p, "\tshr.u32 %r20, %r3, 3;").map_err(write_err)?; // byte idx
        writeln!(p, "\tand.b32 %r21, %r3, 7;").map_err(write_err)?; // bit within byte
        writeln!(p, "\tmul.wide.u32 %rd21, %r20, 1;").map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd21, %rd20, %rd21;").map_err(write_err)?;
        writeln!(p, "\tld.global.u8 %r22, [%rd21];").map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r22, %r22, %r21;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r22, %r22, 1;").map_err(write_err)?; // %r22 = self_valid
        // partner bit
        writeln!(p, "\tshr.u32 %r23, %r7, 3;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r24, %r7, 7;").map_err(write_err)?;
        writeln!(p, "\tmul.wide.u32 %rd22, %r23, 1;").map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd22, %rd20, %rd22;").map_err(write_err)?;
        writeln!(p, "\tld.global.u8 %r25, [%rd22];").map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r25, %r25, %r24;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r25, %r25, 1;").map_err(write_err)?; // %r25 = partner_valid

        // both null -> equal -> next key
        writeln!(p, "\tor.b32 %r26, %r22, %r25;").map_err(write_err)?;
        writeln!(p, "\tsetp.eq.s32 %p_both_null, %r26, 0;",).map_err(write_err)?;
        writeln!(p, "\t@%p_both_null bra {};", lbl_next).map_err(write_err)?;

        // self null, partner not -> route per (nulls_first XOR asc_block XOR dir)
        //
        // Sort order: NULLs land "first" if NULLS FIRST; otherwise last.
        // ASC + NULLS FIRST: a NULL self comes before non-NULL partner.
        //   We are in an ASC block iff %p2. If ASC and (self NULL & partner not):
        //   self should be left of partner — i.e. no swap. If DESC block,
        //   bitonic semantics flip: ascending position in a DESC block means
        //   self should be RIGHT of partner (swap = yes).
        // ASC + NULLS LAST: a NULL self goes AFTER partner -> swap (in ASC block).
        // DESC + NULLS FIRST: NULL self should come first in the DESC global
        //   order (i.e. left). DESC block (p2=false): self left -> no swap.
        // ...
        //
        // Net rule: in any block direction, "self should be left of partner"
        // means swap iff the block direction is DESC (i.e. !p2).
        //
        // Let `null_left` = "self (NULL) should sort left of partner (non-NULL)
        // in the global order". This is:
        //   null_left = nulls_first XOR (direction == Desc)
        // (NULLS FIRST + ASC: null on the left; NULLS LAST + DESC: also null
        // on the left.)
        //
        // Then swap iff: (block is ASC AND null_left is false) OR
        //                (block is DESC AND null_left is true)
        //              = (asc_block XOR null_left) flipped
        //              = !(asc_block XOR null_left)
        //              = asc_block == null_left
        let null_left = k.nulls_first ^ matches!(k.direction, SortDirection::Desc);

        // self_null_partner_not: %r22==0, %r25==1
        writeln!(p, "\tsetp.eq.s32 %p_sn, %r22, 0;").map_err(write_err)?;
        writeln!(p, "\tsetp.ne.s32 %p_pn, %r25, 0;").map_err(write_err)?;
        writeln!(p, "\tand.pred %p_self_null, %p_sn, %p_pn;").map_err(write_err)?;
        if null_left {
            // swap iff asc_block == true (p2)
            writeln!(p, "\t@%p_self_null selp.b32 %r27, 1, 0, %p2;").map_err(write_err)?;
        } else {
            // swap iff asc_block == false
            writeln!(p, "\t@%p_self_null selp.b32 %r27, 0, 1, %p2;").map_err(write_err)?;
        }
        writeln!(p, "\t@%p_self_null mov.b32 %r10, %r27;").map_err(write_err)?;
        writeln!(p, "\t@%p_self_null bra DECIDED;").map_err(write_err)?;

        // partner null, self not -> opposite of above:
        writeln!(p, "\tsetp.eq.s32 %p_pn2, %r25, 0;").map_err(write_err)?;
        writeln!(p, "\tsetp.ne.s32 %p_sn2, %r22, 0;").map_err(write_err)?;
        writeln!(p, "\tand.pred %p_partner_null, %p_pn2, %p_sn2;").map_err(write_err)?;
        if null_left {
            // partner should be left of self -> swap iff asc_block == false
            writeln!(p, "\t@%p_partner_null selp.b32 %r28, 0, 1, %p2;").map_err(write_err)?;
        } else {
            writeln!(p, "\t@%p_partner_null selp.b32 %r28, 1, 0, %p2;").map_err(write_err)?;
        }
        writeln!(p, "\t@%p_partner_null mov.b32 %r10, %r28;").map_err(write_err)?;
        writeln!(p, "\t@%p_partner_null bra DECIDED;").map_err(write_err)?;
        // both valid: fall through to value compare.
    }

    // -- Value compare (both non-NULL or non-nullable column). ---------
    if shmem {
        // shmem variant emits its own load via a helper; for now we share the
        // global-mem version. Shmem variant overrides this with shared offsets.
        emit_global_key_load(p, entry, ki, k, key_param_idx, key_w, &flavour)?;
    } else {
        emit_global_key_load(p, entry, ki, k, key_param_idx, key_w, &flavour)?;
    }

    // setp.eq -> next key
    let (self_reg, part_reg) = key_regs(ki, k.dtype);
    writeln!(p, "\t{} %p_eq, {}, {};", flavour.setp_eq, self_reg, part_reg).map_err(write_err)?;
    writeln!(p, "\t@%p_eq bra {};", lbl_next).map_err(write_err)?;

    // do_swap = (asc_block XOR dir_is_desc) ? (self > partner) : (self < partner)
    //
    // For dir=Asc: asc_block true -> swap if self>partner; asc_block false ->
    // swap if self<partner.
    // For dir=Desc: asc_block true -> swap if self<partner; asc_block false ->
    // swap if self>partner.
    writeln!(p, "\t{} %p_gt, {}, {};", flavour.setp_gt, self_reg, part_reg).map_err(write_err)?;
    writeln!(p, "\t{} %p_lt, {}, {};", flavour.setp_lt, self_reg, part_reg).map_err(write_err)?;

    let (asc_pred, desc_pred) = match k.direction {
        SortDirection::Asc => ("%p_gt", "%p_lt"),
        SortDirection::Desc => ("%p_lt", "%p_gt"),
    };
    // %r10 = p2 ? (asc_pred ? 1 : 0) : (desc_pred ? 1 : 0)
    writeln!(p, "\tselp.b32 %r29, 1, 0, {};", asc_pred).map_err(write_err)?;
    writeln!(p, "\tselp.b32 %r30, 1, 0, {};", desc_pred).map_err(write_err)?;
    writeln!(p, "\tselp.b32 %r10, %r29, %r30, %p2;").map_err(write_err)?;
    writeln!(p, "\tbra DECIDED;").map_err(write_err)?;

    // Tie-equal target: try next key (or fall through to "all equal -> no swap").
    writeln!(p, "{}:", lbl_next).map_err(write_err)?;
    // suppress unused-label warnings even if no shmem variant references it
    let _ = (lbl_swap_yes, lbl_swap_no);
    Ok(())
}

/// PTX register names for a key's (self, partner) registers.
///
/// Bool keys share the b32 (`ki32`) register pool: bytes are loaded with
/// `ld.global.u8 %ki32X, [...]`, which PTX zero-extends into the 32-bit
/// register. The compare then uses the same s32 mnemonic as Int32.
fn key_regs(ki: usize, dtype: DataType) -> (String, String) {
    let prefix = match dtype {
        DataType::Int32 | DataType::Bool => "ki32",
        DataType::Int64 => "ki64",
        DataType::Float32 => "kf32",
        DataType::Float64 => "kf64",
        _ => unreachable!("validated"),
    };
    (
        format!("%{}{}", prefix, ki * 2),
        format!("%{}{}", prefix, ki * 2 + 1),
    )
}

/// **Stage 3** — emit the padded-row pre-routing block for the multi-launch
/// kernel (global-memory `is_padded` bitmap).
///
/// Reads bit `tid` and bit `partner` from a packed-bit array (Arrow-style:
/// `byte = i >> 3`, `bit = i & 7`, 1 = padded). On a one-sided padded result
/// the routing decision is baked: padded rows always sort to the global end.
/// If both rows are padded we fall through to the lex comparator unchanged
/// (they're tied — doesn't matter which "wins"; the truncation step in
/// `gpu_sort` drops them all).
///
/// The decision matches the null-routing logic with `null_left = false`:
///   - self_padded && !partner_padded -> swap iff in ASC block (`%p2`)
///   - !self_padded && partner_padded -> swap iff in DESC block (`!%p2`)
///
/// Writing into `%r10` and branching to `DECIDED` short-circuits the entire
/// per-key compare loop.
fn emit_padded_route_global(p: &mut String, entry: &str, padded_param_idx: usize) -> BoltResult<()> {
    writeln!(p, "// ---- stage3: padded-row pre-routing (is_padded bitmap) ----")
        .map_err(write_err)?;
    // Load padded bit for self into %r16.
    writeln!(p, "\tld.param.u64 %rd44, [{entry}_param_{}];", padded_param_idx)
        .map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd44, %rd44;").map_err(write_err)?;
    writeln!(p, "\tshr.u32 %r16, %r3, 3;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r17, %r3, 7;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd45, %r16, 1;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd45, %rd44, %rd45;").map_err(write_err)?;
    writeln!(p, "\tld.global.u8 %r18, [%rd45];").map_err(write_err)?;
    writeln!(p, "\tshr.u32 %r18, %r18, %r17;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r18, %r18, 1;").map_err(write_err)?; // %r18 = self_padded
    // partner bit -> %r19.
    writeln!(p, "\tshr.u32 %r20, %r7, 3;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r21, %r7, 7;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd46, %r20, 1;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd46, %rd44, %rd46;").map_err(write_err)?;
    writeln!(p, "\tld.global.u8 %r19, [%rd46];").map_err(write_err)?;
    writeln!(p, "\tshr.u32 %r19, %r19, %r21;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r19, %r19, 1;").map_err(write_err)?; // %r19 = partner_padded
    // Both padded -> fall through.
    writeln!(p, "\tand.b32 %r22, %r18, %r19;").map_err(write_err)?;
    writeln!(p, "\tsetp.ne.s32 %p_both_pad, %r22, 0;").map_err(write_err)?;
    // self_pad && !partner_pad
    writeln!(p, "\tsetp.ne.s32 %p_sp1, %r18, 0;").map_err(write_err)?;
    writeln!(p, "\tsetp.eq.s32 %p_sp2, %r19, 0;").map_err(write_err)?;
    writeln!(p, "\tand.pred %p_self_pad, %p_sp1, %p_sp2;").map_err(write_err)?;
    // partner_pad && !self_pad
    writeln!(p, "\tsetp.ne.s32 %p_pp1, %r19, 0;").map_err(write_err)?;
    writeln!(p, "\tsetp.eq.s32 %p_pp2, %r18, 0;").map_err(write_err)?;
    writeln!(p, "\tand.pred %p_partner_pad, %p_pp1, %p_pp2;").map_err(write_err)?;
    // Apply decision: self padded -> swap iff p2 (ASC block).
    writeln!(p, "\t@%p_self_pad selp.b32 %r23, 1, 0, %p2;").map_err(write_err)?;
    writeln!(p, "\t@%p_self_pad mov.b32 %r10, %r23;").map_err(write_err)?;
    writeln!(p, "\t@%p_self_pad bra DECIDED;").map_err(write_err)?;
    // Partner padded -> swap iff !p2 (DESC block).
    writeln!(p, "\t@%p_partner_pad selp.b32 %r24, 0, 1, %p2;").map_err(write_err)?;
    writeln!(p, "\t@%p_partner_pad mov.b32 %r10, %r24;").map_err(write_err)?;
    writeln!(p, "\t@%p_partner_pad bra DECIDED;").map_err(write_err)?;
    // Otherwise (both real, or both padded -> tie): fall through to lex.
    Ok(())
}

/// Emit a load from global memory for key `ki` (self & partner).
fn emit_global_key_load(
    p: &mut String,
    entry: &str,
    ki: usize,
    k: &KeyDesc,
    key_param_idx: usize,
    key_w: i64,
    flavour: &DtypeFlavour,
) -> BoltResult<()> {
    let (self_reg, part_reg) = key_regs(ki, k.dtype);
    writeln!(p, "\tld.param.u64 %rd30, [{entry}_param_{}];", key_param_idx).map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd30, %rd30;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd31, %r3, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd32, %rd30, %rd31;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd33, %r7, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd34, %rd30, %rd33;").map_err(write_err)?;
    writeln!(p, "\tld.global.{} {}, [%rd32];", flavour.ld_st_suffix, self_reg).map_err(write_err)?;
    writeln!(p, "\tld.global.{} {}, [%rd34];", flavour.ld_st_suffix, part_reg).map_err(write_err)?;
    Ok(())
}

/// Emit the swap of key `ki`'s self & partner cells.
fn emit_key_swap(p: &mut String, entry: &str, ki: usize, k: &KeyDesc) -> BoltResult<()> {
    let flavour = DtypeFlavour::for_dtype(k.dtype)?;
    let key_w = flavour.byte_width as i64;
    let key_param_idx = ki * 2;
    let (self_reg, part_reg) = key_regs(ki, k.dtype);
    // Reload the addresses (we trampled %rd30..34 across multiple keys'
    // compares; recomputing keeps each key's swap block self-contained).
    writeln!(p, "\tld.param.u64 %rd35, [{entry}_param_{}];", key_param_idx).map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd35, %rd35;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd36, %r3, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd37, %rd35, %rd36;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd38, %r7, {key_w};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd39, %rd35, %rd38;").map_err(write_err)?;
    // Re-read; we lost the registers between compare blocks for later keys.
    writeln!(p, "\tld.global.{} {}, [%rd37];", flavour.ld_st_suffix, self_reg).map_err(write_err)?;
    writeln!(p, "\tld.global.{} {}, [%rd39];", flavour.ld_st_suffix, part_reg).map_err(write_err)?;
    writeln!(p, "\tst.global.{} [%rd37], {};", flavour.ld_st_suffix, part_reg).map_err(write_err)?;
    writeln!(p, "\tst.global.{} [%rd39], {};", flavour.ld_st_suffix, self_reg).map_err(write_err)?;
    // Note: we don't swap validity bits — the validity bitmap is not
    // permuted by the sort because the permutation is the indices buffer;
    // gpu_sort gathers each column (including validity) using the final
    // indices array, so swapping validity in-place here would be wrong.
    Ok(())
}

/// Emit the in-block shared-memory bitonic kernel.
///
/// All threads in the block hold the entire keys + indices array in shmem.
/// We run the full log²n stage/substage schedule inside one kernel, syncing
/// between substages via `bar.sync 0` (= `__syncthreads()`).
///
/// ABI: same as `MultiLaunch` minus `stage` and `substage_mask` (the kernel
/// walks every stage internally based on the compile-time `shmem_n_pow2`).
fn emit_multikey_shmem(p: &mut String, entry: &str, spec: &SortKernelSpec) -> BoltResult<()> {
    let n_pow2 = spec.shmem_n_pow2;
    if n_pow2 == 0 {
        return Err(BoltError::Other(
            "sort_kernel: Shmem layout requires shmem_n_pow2 >= 1".into(),
        ));
    }
    let log2_n = n_pow2.trailing_zeros();

    // Shared-memory arrays at module scope (matches the convention used by
    // shmem_count_kernel.rs / shmem_sum_kernel.rs). One key buffer per key,
    // optional validity buffer per nullable key, a u32 indices buffer, and
    // a packed-bit is_padded array.
    //
    // **Stage 3** — validity is now packed at 1 bit per row (u32 per 32
    // elements) instead of 1 byte per row. For n_pow2=256 this drops shmem
    // from `keys*256B (validity)` to `keys*32B`, an 8× reduction. The
    // is_padded array uses the same packed-bit layout.
    let pad_words = ((n_pow2 + 31) / 32) as u64; // ceil(n_pow2 / 32)
    for (ki, k) in spec.keys.iter().enumerate() {
        let flavour = DtypeFlavour::for_dtype(k.dtype)?;
        let bytes = (n_pow2 as u64) * (flavour.byte_width as u64);
        writeln!(p, ".shared .align 8 .b8 sh_k{}[{}];", ki, bytes).map_err(write_err)?;
        if k.nullable {
            // Packed bits, u32 words. Each u32 holds 32 validity bits.
            writeln!(p, ".shared .align 4 .b8 sh_v{}[{}];", ki, pad_words * 4)
                .map_err(write_err)?;
        }
    }
    writeln!(p, ".shared .align 4 .b8 sh_idx[{}];", (n_pow2 as u64) * 4).map_err(write_err)?;
    // is_padded as a packed-bit u32 array (Stage 3).
    writeln!(p, ".shared .align 4 .b8 sh_pad[{}];", pad_words * 4).map_err(write_err)?;
    writeln!(p).map_err(write_err)?;

    // Signature: (k0, v0, ..., kN-1, vN-1, indices, is_padded, n_pow2).
    // No stage/mask. Stage 3: trailing pointer is `is_padded` (Arrow-style
    // packed u8 bitmap).
    writeln!(p, ".visible .entry {entry}(").map_err(write_err)?;
    let total_ptr_params = MAX_SORT_KEYS * 2 + 2;
    for i in 0..total_ptr_params {
        writeln!(p, "\t.param .u64 {entry}_param_{i},").map_err(write_err)?;
    }
    writeln!(p, "\t.param .u32 {entry}_param_{}", total_ptr_params).map_err(write_err)?;
    writeln!(p, ")").map_err(write_err)?;
    writeln!(p, "{{").map_err(write_err)?;

    // Registers.
    writeln!(p, "\t.reg .pred %p<8>;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_in;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_skip;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_pgt;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_eq;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_gt;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_lt;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_bn;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_sn;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_pn;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_sn2;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_pn2;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_selfn;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_partn;").map_err(write_err)?;
    // Stage 3 padded-routing predicates (shmem variant).
    writeln!(p, "\t.reg .pred %p_self_pad;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_partner_pad;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_sp1;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_sp2;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_pp1;").map_err(write_err)?;
    writeln!(p, "\t.reg .pred %p_pp2;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32 %r<40>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %rd<40>;").map_err(write_err)?;
    writeln!(p, "\t.reg .b32 %ki32<{}>;", MAX_SORT_KEYS * 2).map_err(write_err)?;
    writeln!(p, "\t.reg .b64 %ki64<{}>;", MAX_SORT_KEYS * 2).map_err(write_err)?;
    writeln!(p, "\t.reg .f32 %kf32<{}>;", MAX_SORT_KEYS * 2).map_err(write_err)?;
    writeln!(p, "\t.reg .f64 %kf64<{}>;", MAX_SORT_KEYS * 2).map_err(write_err)?;

    // tid
    writeln!(p, "\tmov.u32 %r6, %tid.x;").map_err(write_err)?;
    writeln!(p, "\tmov.b32 %r3, %r6;").map_err(write_err)?; // alias

    // n_pow2 (runtime, must equal compile-time shmem_n_pow2)
    writeln!(p, "\tld.param.u32 %r2, [{entry}_param_{}];", total_ptr_params).map_err(write_err)?;

    // -- Load all keys + indices from global into shmem. --------------
    writeln!(p, "\tsetp.lt.s32 %p_in, %r3, %r2;").map_err(write_err)?;

    // Stage 3: initialise the packed-bit shmem buffers to zero before any
    // atomic-OR sets bits into them. One thread covers each u32 word.
    // For n_pow2<=block_size, every thread whose tid < pad_words zeroes
    // one word. Validity buffers are also packed-bit; same init.
    writeln!(p, "\tsetp.lt.s32 %p_skip, %r3, {};", pad_words).map_err(write_err)?;
    for (ki, k) in spec.keys.iter().enumerate() {
        if k.nullable {
            writeln!(p, "\tmov.u64 %rd0, sh_v{};", ki).map_err(write_err)?;
            writeln!(p, "\tmul.wide.u32 %rd1, %r3, 4;").map_err(write_err)?;
            writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
            writeln!(p, "\t@%p_skip st.shared.u32 [%rd2], 0;").map_err(write_err)?;
        }
    }
    writeln!(p, "\tmov.u64 %rd0, sh_pad;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd1, %r3, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
    writeln!(p, "\t@%p_skip st.shared.u32 [%rd2], 0;").map_err(write_err)?;
    writeln!(p, "\tbar.sync 0;").map_err(write_err)?;

    for (ki, k) in spec.keys.iter().enumerate() {
        let flavour = DtypeFlavour::for_dtype(k.dtype)?;
        let kw = flavour.byte_width as i64;
        // Load from global.
        writeln!(p, "\tld.param.u64 %rd0, [{entry}_param_{}];", ki * 2).map_err(write_err)?;
        writeln!(p, "\tcvta.to.global.u64 %rd0, %rd0;").map_err(write_err)?;
        writeln!(p, "\tmul.wide.u32 %rd1, %r3, {kw};").map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd2, %rd0, %rd1;").map_err(write_err)?;
        let (self_reg, _) = key_regs(ki, k.dtype);
        writeln!(p, "\t@%p_in ld.global.{} {}, [%rd2];", flavour.ld_st_suffix, self_reg)
            .map_err(write_err)?;
        // Store into shmem at offset tid * kw.
        writeln!(p, "\tmov.u64 %rd3, sh_k{};", ki).map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd4, %rd3, %rd1;").map_err(write_err)?;
        writeln!(p, "\t@%p_in st.shared.{} [%rd4], {};", flavour.ld_st_suffix, self_reg)
            .map_err(write_err)?;

        if k.nullable {
            // Stage 3: pack into bits. Load the validity bit for `tid` from
            // global (Arrow-packed u8 bitmap), then atom.or into the u32
            // word at `sh_v<ki>[tid>>5]` shifted to bit `tid & 31`.
            writeln!(p, "\tld.param.u64 %rd5, [{entry}_param_{}];", ki * 2 + 1).map_err(write_err)?;
            writeln!(p, "\tcvta.to.global.u64 %rd5, %rd5;").map_err(write_err)?;
            writeln!(p, "\tshr.u32 %r10, %r3, 3;").map_err(write_err)?;
            writeln!(p, "\tand.b32 %r11, %r3, 7;").map_err(write_err)?;
            writeln!(p, "\tmul.wide.u32 %rd6, %r10, 1;").map_err(write_err)?;
            writeln!(p, "\tadd.s64 %rd6, %rd5, %rd6;").map_err(write_err)?;
            writeln!(p, "\t@%p_in ld.global.u8 %r12, [%rd6];").map_err(write_err)?;
            writeln!(p, "\tshr.u32 %r12, %r12, %r11;").map_err(write_err)?;
            writeln!(p, "\tand.b32 %r12, %r12, 1;").map_err(write_err)?;
            // Compute shmem word address: sh_v<ki> + (tid>>5)*4
            writeln!(p, "\tshr.u32 %r13, %r3, 5;").map_err(write_err)?;
            writeln!(p, "\tand.b32 %r14, %r3, 31;").map_err(write_err)?;
            writeln!(p, "\tmov.u64 %rd7, sh_v{};", ki).map_err(write_err)?;
            writeln!(p, "\tmul.wide.u32 %rd8, %r13, 4;").map_err(write_err)?;
            writeln!(p, "\tadd.s64 %rd8, %rd7, %rd8;").map_err(write_err)?;
            // Shifted bit: %r12 << (tid & 31)
            writeln!(p, "\tshl.b32 %r12, %r12, %r14;").map_err(write_err)?;
            writeln!(p, "\t@%p_in atom.shared.or.b32 %r15, [%rd8], %r12;").map_err(write_err)?;
        }
    }

    // Stage 3: load is_padded bit for `tid` into sh_pad packed bits.
    {
        writeln!(p, "\tld.param.u64 %rd5, [{entry}_param_{}];", MAX_SORT_KEYS * 2 + 1)
            .map_err(write_err)?;
        writeln!(p, "\tcvta.to.global.u64 %rd5, %rd5;").map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r10, %r3, 3;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r11, %r3, 7;").map_err(write_err)?;
        writeln!(p, "\tmul.wide.u32 %rd6, %r10, 1;").map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd6, %rd5, %rd6;").map_err(write_err)?;
        writeln!(p, "\tld.global.u8 %r12, [%rd6];").map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r12, %r12, %r11;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r12, %r12, 1;").map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r13, %r3, 5;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r14, %r3, 31;").map_err(write_err)?;
        writeln!(p, "\tmov.u64 %rd7, sh_pad;").map_err(write_err)?;
        writeln!(p, "\tmul.wide.u32 %rd8, %r13, 4;").map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd8, %rd7, %rd8;").map_err(write_err)?;
        writeln!(p, "\tshl.b32 %r12, %r12, %r14;").map_err(write_err)?;
        writeln!(p, "\tatom.shared.or.b32 %r15, [%rd8], %r12;").map_err(write_err)?;
    }

    // Identity index in shmem
    writeln!(p, "\tmov.u64 %rd9, sh_idx;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd10, %r3, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd11, %rd9, %rd10;").map_err(write_err)?;
    writeln!(p, "\t@%p_in st.shared.u32 [%rd11], %r3;").map_err(write_err)?;

    writeln!(p, "\tbar.sync 0;").map_err(write_err)?;

    // -- Stage/substage loop, fully unrolled (compile-time log2_n). ----
    //
    // For each (stage j, substage s):
    //   partner = tid ^ (1 << (s-1))
    //   asc_block = ((tid >> j) & 1) == 0
    //   compare (lex) keys; swap-if-needed
    //   bar.sync 0
    for stage in 1..=log2_n {
        let mut substage = stage;
        loop {
            let mask = 1u32 << (substage - 1);
            // partner = tid XOR mask
            writeln!(p, "\txor.b32 %r7, %r3, {};", mask).map_err(write_err)?;
            // skip if tid >= partner (paired-skip)
            writeln!(p, "\tsetp.ge.u32 %p1, %r3, %r7;").map_err(write_err)?;
            // also skip if tid >= n_pow2 (oob)
            writeln!(p, "\tsetp.ge.u32 %p0, %r3, %r2;").map_err(write_err)?;
            writeln!(p, "\tor.pred %p_skip, %p0, %p1;").map_err(write_err)?;
            // also skip if partner >= n_pow2 (when shmem holds a padded slot)
            writeln!(p, "\tsetp.ge.u32 %p_pgt, %r7, %r2;").map_err(write_err)?;
            writeln!(p, "\tor.pred %p_skip, %p_skip, %p_pgt;").map_err(write_err)?;

            // asc_block_bit = (tid >> stage) & 1
            writeln!(p, "\tshr.u32 %r8, %r3, {};", stage).map_err(write_err)?;
            writeln!(p, "\tand.b32 %r8, %r8, 1;").map_err(write_err)?;
            writeln!(p, "\tsetp.eq.s32 %p2, %r8, 0;").map_err(write_err)?;

            writeln!(p, "\tmov.b32 %r10, 0;").map_err(write_err)?;
            writeln!(p, "\t@%p_skip bra SH_S{}_T{}_AFTER;", stage, substage).map_err(write_err)?;

            // Stage 3 padded-row pre-routing (shmem variant) — read
            // is_padded for self and partner via the indices indirection
            // (sh_idx[cell] gives the original row index), then test the
            // packed-bit sh_pad array.
            emit_padded_route_shmem(p, stage, substage)?;

            // Lex compare (read shmem instead of global).
            for (ki, k) in spec.keys.iter().enumerate() {
                emit_shmem_key_compare(p, ki, k, stage, substage)?;
            }
            writeln!(p, "\tbra SH_S{}_T{}_DECIDED;", stage, substage).map_err(write_err)?;
            writeln!(p, "SH_S{}_T{}_DECIDED:", stage, substage).map_err(write_err)?;

            writeln!(p, "\tsetp.ne.s32 %p3, %r10, 0;").map_err(write_err)?;
            writeln!(p, "\t@!%p3 bra SH_S{}_T{}_AFTER;", stage, substage).map_err(write_err)?;

            // Swap shmem cells for every key + indices.
            for (ki, k) in spec.keys.iter().enumerate() {
                emit_shmem_key_swap(p, ki, k)?;
            }
            // indices
            writeln!(p, "\tmov.u64 %rd9, sh_idx;").map_err(write_err)?;
            writeln!(p, "\tmul.wide.u32 %rd10, %r3, 4;").map_err(write_err)?;
            writeln!(p, "\tadd.s64 %rd11, %rd9, %rd10;").map_err(write_err)?;
            writeln!(p, "\tmul.wide.u32 %rd12, %r7, 4;").map_err(write_err)?;
            writeln!(p, "\tadd.s64 %rd13, %rd9, %rd12;").map_err(write_err)?;
            writeln!(p, "\tld.shared.u32 %r13, [%rd11];").map_err(write_err)?;
            writeln!(p, "\tld.shared.u32 %r14, [%rd13];").map_err(write_err)?;
            writeln!(p, "\tst.shared.u32 [%rd11], %r14;").map_err(write_err)?;
            writeln!(p, "\tst.shared.u32 [%rd13], %r13;").map_err(write_err)?;

            writeln!(p, "SH_S{}_T{}_AFTER:", stage, substage).map_err(write_err)?;
            writeln!(p, "\tbar.sync 0;").map_err(write_err)?;

            if substage == 1 {
                break;
            }
            substage -= 1;
        }
    }

    // -- Writeback indices (the only output the host reads back). ------
    // Keys themselves are discarded after the sort; gpu_sort already keeps
    // its own copy and uses the indices to gather.
    writeln!(p, "\tld.param.u64 %rd20, [{entry}_param_{}];", MAX_SORT_KEYS * 2).map_err(write_err)?;
    writeln!(p, "\tcvta.to.global.u64 %rd20, %rd20;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd21, %r3, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd22, %rd20, %rd21;").map_err(write_err)?;
    writeln!(p, "\tmov.u64 %rd23, sh_idx;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd24, %rd23, %rd21;").map_err(write_err)?;
    writeln!(p, "\t@%p_in ld.shared.u32 %r15, [%rd24];").map_err(write_err)?;
    writeln!(p, "\t@%p_in st.global.u32 [%rd22], %r15;").map_err(write_err)?;

    writeln!(p, "\tret;").map_err(write_err)?;
    writeln!(p, "}}").map_err(write_err)?;
    Ok(())
}

/// **Stage 3** — emit the shmem padded-row pre-routing block.
///
/// The shmem variant keeps `sh_pad` indexed by ORIGINAL row index (i.e. it
/// is not permuted by the in-block swaps; only `sh_k*` and `sh_idx` move).
/// To read the padded bit for the current pair we look up
/// `orig_self = sh_idx[self_cell]` and `orig_partner = sh_idx[partner_cell]`,
/// then bfe the corresponding bit out of `sh_pad[orig >> 5]`.
///
/// Routing matches `emit_padded_route_global`: padded rows go to the global
/// end (= ASC-direction "max").
fn emit_padded_route_shmem(p: &mut String, stage: u32, substage: u32) -> BoltResult<()> {
    writeln!(p, "// ---- stage3 shmem: padded-row pre-routing ----")
        .map_err(write_err)?;
    // Read orig indices for self and partner from sh_idx.
    writeln!(p, "\tmov.u64 %rd33, sh_idx;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd34, %r3, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd34, %rd33, %rd34;").map_err(write_err)?;
    writeln!(p, "\tld.shared.u32 %r25, [%rd34];").map_err(write_err)?; // orig_self
    writeln!(p, "\tmul.wide.u32 %rd35, %r7, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd35, %rd33, %rd35;").map_err(write_err)?;
    writeln!(p, "\tld.shared.u32 %r26, [%rd35];").map_err(write_err)?; // orig_partner

    // sh_pad lookup: word = sh_pad[orig >> 5]; bit = (word >> (orig & 31)) & 1.
    writeln!(p, "\tmov.u64 %rd36, sh_pad;").map_err(write_err)?;
    writeln!(p, "\tshr.u32 %r27, %r25, 5;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r28, %r25, 31;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd37, %r27, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd37, %rd36, %rd37;").map_err(write_err)?;
    writeln!(p, "\tld.shared.u32 %r29, [%rd37];").map_err(write_err)?;
    writeln!(p, "\tshr.u32 %r29, %r29, %r28;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r29, %r29, 1;").map_err(write_err)?; // self_padded
    writeln!(p, "\tshr.u32 %r30, %r26, 5;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r31, %r26, 31;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd38, %r30, 4;").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd38, %rd36, %rd38;").map_err(write_err)?;
    writeln!(p, "\tld.shared.u32 %r32, [%rd38];").map_err(write_err)?;
    writeln!(p, "\tshr.u32 %r32, %r32, %r31;").map_err(write_err)?;
    writeln!(p, "\tand.b32 %r32, %r32, 1;").map_err(write_err)?; // partner_padded

    // self_pad && !partner_pad -> swap iff p2.
    writeln!(p, "\tsetp.ne.s32 %p_sp1, %r29, 0;").map_err(write_err)?;
    writeln!(p, "\tsetp.eq.s32 %p_sp2, %r32, 0;").map_err(write_err)?;
    writeln!(p, "\tand.pred %p_self_pad, %p_sp1, %p_sp2;").map_err(write_err)?;
    writeln!(p, "\t@%p_self_pad selp.b32 %r33, 1, 0, %p2;").map_err(write_err)?;
    writeln!(p, "\t@%p_self_pad mov.b32 %r10, %r33;").map_err(write_err)?;
    writeln!(p, "\t@%p_self_pad bra SH_S{}_T{}_DECIDED;", stage, substage).map_err(write_err)?;
    writeln!(p, "\tsetp.ne.s32 %p_pp1, %r32, 0;").map_err(write_err)?;
    writeln!(p, "\tsetp.eq.s32 %p_pp2, %r29, 0;").map_err(write_err)?;
    writeln!(p, "\tand.pred %p_partner_pad, %p_pp1, %p_pp2;").map_err(write_err)?;
    writeln!(p, "\t@%p_partner_pad selp.b32 %r34, 0, 1, %p2;").map_err(write_err)?;
    writeln!(p, "\t@%p_partner_pad mov.b32 %r10, %r34;").map_err(write_err)?;
    writeln!(p, "\t@%p_partner_pad bra SH_S{}_T{}_DECIDED;", stage, substage).map_err(write_err)?;
    Ok(())
}

/// Per-key compare against shmem within the shmem kernel's stage loop.
///
/// Generates per-(key, stage, substage) labels so multiple compares can
/// coexist in the same kernel.
///
/// **Stage 3**: validity is now stored as a packed-bit u32 array indexed by
/// ORIGINAL row index (not by cell position). We look up
/// `orig = sh_idx[cell]` and extract bit `orig & 31` from word
/// `sh_v<ki>[orig >> 5]`. This cuts shmem usage 8× vs the Stage-2
/// 1-byte-per-element layout (n=256, 1 nullable key: 256B -> 32B).
fn emit_shmem_key_compare(
    p: &mut String,
    ki: usize,
    k: &KeyDesc,
    stage: u32,
    substage: u32,
) -> BoltResult<()> {
    let flavour = DtypeFlavour::for_dtype(k.dtype)?;
    let kw = flavour.byte_width as i64;
    let (self_reg, part_reg) = key_regs(ki, k.dtype);
    let lbl_next = format!("SH_S{}_T{}_K{}_NEXT", stage, substage, ki);

    if k.nullable {
        // Pull original indices for self / partner from sh_idx (these were
        // already read by `emit_padded_route_shmem`; reload defensively in
        // case a future refactor reorders the blocks).
        writeln!(p, "\tmov.u64 %rd15, sh_idx;").map_err(write_err)?;
        writeln!(p, "\tmul.wide.u32 %rd16, %r3, 4;").map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd16, %rd15, %rd16;").map_err(write_err)?;
        writeln!(p, "\tld.shared.u32 %r20, [%rd16];").map_err(write_err)?; // orig_self
        writeln!(p, "\tmul.wide.u32 %rd17, %r7, 4;").map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd17, %rd15, %rd17;").map_err(write_err)?;
        writeln!(p, "\tld.shared.u32 %r21, [%rd17];").map_err(write_err)?; // orig_partner

        // Packed-bit validity load. Stage 3.
        writeln!(p, "\tmov.u64 %rd15, sh_v{};", ki).map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r12, %r20, 5;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r13, %r20, 31;").map_err(write_err)?;
        writeln!(p, "\tmul.wide.u32 %rd16, %r12, 4;").map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd16, %rd15, %rd16;").map_err(write_err)?;
        writeln!(p, "\tld.shared.u32 %r22, [%rd16];").map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r22, %r22, %r13;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r22, %r22, 1;").map_err(write_err)?; // self_valid
        writeln!(p, "\tshr.u32 %r14, %r21, 5;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r16, %r21, 31;").map_err(write_err)?;
        writeln!(p, "\tmul.wide.u32 %rd17, %r14, 4;").map_err(write_err)?;
        writeln!(p, "\tadd.s64 %rd17, %rd15, %rd17;").map_err(write_err)?;
        writeln!(p, "\tld.shared.u32 %r25, [%rd17];").map_err(write_err)?;
        writeln!(p, "\tshr.u32 %r25, %r25, %r16;").map_err(write_err)?;
        writeln!(p, "\tand.b32 %r25, %r25, 1;").map_err(write_err)?; // partner_valid

        // both null
        writeln!(p, "\tor.b32 %r26, %r22, %r25;").map_err(write_err)?;
        writeln!(p, "\tsetp.eq.s32 %p_bn, %r26, 0;").map_err(write_err)?;
        writeln!(p, "\t@%p_bn bra {};", lbl_next).map_err(write_err)?;

        let null_left = k.nulls_first ^ matches!(k.direction, SortDirection::Desc);
        // self_null && partner_not_null
        writeln!(p, "\tsetp.eq.s32 %p_sn, %r22, 0;").map_err(write_err)?;
        writeln!(p, "\tsetp.ne.s32 %p_pn, %r25, 0;").map_err(write_err)?;
        writeln!(p, "\tand.pred %p_selfn, %p_sn, %p_pn;").map_err(write_err)?;
        if null_left {
            writeln!(p, "\t@%p_selfn selp.b32 %r27, 1, 0, %p2;").map_err(write_err)?;
        } else {
            writeln!(p, "\t@%p_selfn selp.b32 %r27, 0, 1, %p2;").map_err(write_err)?;
        }
        writeln!(p, "\t@%p_selfn mov.b32 %r10, %r27;").map_err(write_err)?;
        writeln!(p, "\t@%p_selfn bra SH_S{}_T{}_DECIDED;", stage, substage).map_err(write_err)?;

        writeln!(p, "\tsetp.eq.s32 %p_pn2, %r25, 0;").map_err(write_err)?;
        writeln!(p, "\tsetp.ne.s32 %p_sn2, %r22, 0;").map_err(write_err)?;
        writeln!(p, "\tand.pred %p_partn, %p_pn2, %p_sn2;").map_err(write_err)?;
        if null_left {
            writeln!(p, "\t@%p_partn selp.b32 %r28, 0, 1, %p2;").map_err(write_err)?;
        } else {
            writeln!(p, "\t@%p_partn selp.b32 %r28, 1, 0, %p2;").map_err(write_err)?;
        }
        writeln!(p, "\t@%p_partn mov.b32 %r10, %r28;").map_err(write_err)?;
        writeln!(p, "\t@%p_partn bra SH_S{}_T{}_DECIDED;", stage, substage).map_err(write_err)?;
    }

    // Value compare from shmem.
    writeln!(p, "\tmov.u64 %rd18, sh_k{};", ki).map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd19, %r3, {kw};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd19, %rd18, %rd19;").map_err(write_err)?;
    writeln!(p, "\tld.shared.{} {}, [%rd19];", flavour.ld_st_suffix, self_reg).map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd20, %r7, {kw};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd20, %rd18, %rd20;").map_err(write_err)?;
    writeln!(p, "\tld.shared.{} {}, [%rd20];", flavour.ld_st_suffix, part_reg).map_err(write_err)?;

    writeln!(p, "\t{} %p_eq, {}, {};", flavour.setp_eq, self_reg, part_reg).map_err(write_err)?;
    writeln!(p, "\t@%p_eq bra {};", lbl_next).map_err(write_err)?;
    writeln!(p, "\t{} %p_gt, {}, {};", flavour.setp_gt, self_reg, part_reg).map_err(write_err)?;
    writeln!(p, "\t{} %p_lt, {}, {};", flavour.setp_lt, self_reg, part_reg).map_err(write_err)?;
    let (asc_pred, desc_pred) = match k.direction {
        SortDirection::Asc => ("%p_gt", "%p_lt"),
        SortDirection::Desc => ("%p_lt", "%p_gt"),
    };
    writeln!(p, "\tselp.b32 %r29, 1, 0, {};", asc_pred).map_err(write_err)?;
    writeln!(p, "\tselp.b32 %r30, 1, 0, {};", desc_pred).map_err(write_err)?;
    writeln!(p, "\tselp.b32 %r10, %r29, %r30, %p2;").map_err(write_err)?;
    writeln!(p, "\tbra SH_S{}_T{}_DECIDED;", stage, substage).map_err(write_err)?;
    writeln!(p, "{}:", lbl_next).map_err(write_err)?;
    Ok(())
}

/// Swap shmem cells for key `ki`.
///
/// **Stage 3**: only the key values and `sh_idx` (handled separately) move.
/// The packed-bit validity (`sh_v*`) and is_padded (`sh_pad`) arrays stay
/// indexed by ORIGINAL row index — the comparator reads them via
/// `sh_idx[cell]` indirection. Saves swap traffic and avoids costly
/// atomic-bit twiddling for what is functionally a constant lookup.
fn emit_shmem_key_swap(p: &mut String, ki: usize, k: &KeyDesc) -> BoltResult<()> {
    let flavour = DtypeFlavour::for_dtype(k.dtype)?;
    let kw = flavour.byte_width as i64;
    let (self_reg, part_reg) = key_regs(ki, k.dtype);
    writeln!(p, "\tmov.u64 %rd25, sh_k{};", ki).map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd26, %r3, {kw};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd26, %rd25, %rd26;").map_err(write_err)?;
    writeln!(p, "\tmul.wide.u32 %rd27, %r7, {kw};").map_err(write_err)?;
    writeln!(p, "\tadd.s64 %rd27, %rd25, %rd27;").map_err(write_err)?;
    writeln!(p, "\tld.shared.{} {}, [%rd26];", flavour.ld_st_suffix, self_reg).map_err(write_err)?;
    writeln!(p, "\tld.shared.{} {}, [%rd27];", flavour.ld_st_suffix, part_reg).map_err(write_err)?;
    writeln!(p, "\tst.shared.{} [%rd26], {};", flavour.ld_st_suffix, part_reg).map_err(write_err)?;
    writeln!(p, "\tst.shared.{} [%rd27], {};", flavour.ld_st_suffix, self_reg).map_err(write_err)?;
    Ok(())
}

/// Adapt `std::fmt::Error` into a `BoltError::Other`. Mirrors the helper in
/// neighbouring kernel modules.
fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!("sort_kernel: write failed: {}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helpers used by the Stage-4-migrated single-key golden tests below.
    //
    // Stage 4: the original `compile_sort_kernel(dtype, dir)` Stage-1 entry
    // was retired (host driver routes everything through the multi-key
    // driver). The PTX-shape goldens were ported to drive the multi-key
    // `compile_sort_kernel_spec` with a single-key spec — the assertions
    // still check the same load/compare/swap mnemonics, which haven't moved.
    fn single_key_spec(dtype: DataType, dir: SortDirection) -> SortKernelSpec {
        SortKernelSpec {
            keys: vec![KeyDesc {
                dtype,
                direction: dir,
                nullable: false,
                nulls_first: false,
            }],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        }
    }

    /// Single-key entry names must be unique per (dtype, dir). The new entry
    /// names come from `sort_kernel_entry_spec`, which encodes layout + per-
    /// key fields. Stage 4 dropped the old "bolt_bitonic_sort_<dt>_<dir>"
    /// shape; check uniqueness rather than literal strings.
    #[test]
    fn entry_names_per_dtype_dir() {
        let mut names: Vec<String> = Vec::new();
        for dtype in [
            DataType::Int32,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
        ] {
            for dir in [SortDirection::Asc, SortDirection::Desc] {
                names.push(sort_kernel_entry_spec(&single_key_spec(dtype, dir)).unwrap());
            }
        }
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len(), "kernel names must be unique");
    }

    /// Plain Utf8 is still rejected by the kernel's flavour table — the
    /// Utf8-to-i32 inline dictionary lives in `gpu_sort.rs`, so by the time
    /// a spec reaches `compile_sort_kernel_spec` the dtype is Int32.
    #[test]
    fn rejects_unsupported_dtypes() {
        let spec_utf8 = single_key_spec(DataType::Utf8, SortDirection::Asc);
        assert!(compile_sort_kernel_spec(&spec_utf8).is_err());
    }

    /// Header + signature shape goldens — these are the byte-stable bits of
    /// every emitted PTX module. If anything here changes we want a test
    /// failure forcing an intentional update rather than a silent ABI drift.
    #[test]
    fn ptx_header_and_signature_shape() {
        let spec = single_key_spec(DataType::Int32, SortDirection::Asc);
        let entry = sort_kernel_entry_spec(&spec).unwrap();
        let ptx = compile_sort_kernel_spec(&spec).unwrap();

        // Header.
        assert!(ptx.contains(".version 7.5"), "PTX must declare .version 7.5");
        assert!(ptx.contains(".target sm_70"), "PTX must target sm_70");
        assert!(
            ptx.contains(".address_size 64"),
            "PTX must declare 64-bit addresses"
        );

        // Entry point.
        assert!(
            ptx.contains(&format!(".visible .entry {entry}(")),
            "PTX must declare the entry-point signature; got:\n{ptx}"
        );
    }

    /// ASC int32 must use the signed-int `setp.lt`/`setp.gt` mnemonics — the
    /// load-bearing piece of the compare-exchange for the integer fast path.
    #[test]
    fn ptx_asc_int32_uses_signed_compares() {
        let ptx = compile_sort_kernel_spec(&single_key_spec(DataType::Int32, SortDirection::Asc))
            .unwrap();
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
        let ptx = compile_sort_kernel_spec(&single_key_spec(DataType::Int32, SortDirection::Desc))
            .unwrap();
        assert!(ptx.contains("setp.gt.s32"));
        assert!(ptx.contains("setp.lt.s32"));
    }

    /// 64-bit integer key must use the s64-typed setp.
    #[test]
    fn ptx_int64_uses_s64_compares() {
        let ptx = compile_sort_kernel_spec(&single_key_spec(DataType::Int64, SortDirection::Asc))
            .unwrap();
        assert!(ptx.contains("setp.gt.s64"));
        assert!(ptx.contains("setp.lt.s64"));
        // And the load/store path must move 8 bytes.
        assert!(ptx.contains("ld.global.s64"));
        assert!(ptx.contains("st.global.s64"));
    }

    /// Float kernels use the float-typed setp, and load via ld.global.f32/f64.
    #[test]
    fn ptx_floats_use_float_compares() {
        let f32_ptx =
            compile_sort_kernel_spec(&single_key_spec(DataType::Float32, SortDirection::Asc))
                .unwrap();
        assert!(f32_ptx.contains("setp.gt.f32"));
        assert!(f32_ptx.contains("setp.lt.f32"));
        assert!(f32_ptx.contains("ld.global.f32"));

        let f64_ptx =
            compile_sort_kernel_spec(&single_key_spec(DataType::Float64, SortDirection::Desc))
                .unwrap();
        assert!(f64_ptx.contains("setp.gt.f64"));
        assert!(f64_ptx.contains("setp.lt.f64"));
        assert!(f64_ptx.contains("ld.global.f64"));
    }

    /// The kernel must consult `n_pow2` to bail OOB threads. If this guard
    /// disappears, threads past the padded length will write past the
    /// allocation.
    #[test]
    fn ptx_has_n_pow2_oob_guard() {
        let ptx = compile_sort_kernel_spec(&single_key_spec(DataType::Int32, SortDirection::Asc))
            .unwrap();
        // The OOB test: setp.ge.u32 <pred>, <tid>, <n_pow2>; some return path.
        assert!(ptx.contains("setp.ge.u32"), "missing OOB compare against n_pow2");
    }

    /// The kernel must XOR tid against substage_mask to compute the partner
    /// index. This is the defining instruction of the bitonic pattern; any
    /// substitution that hides it (e.g. shifting by stage directly) is a
    /// regression that breaks the algorithm.
    #[test]
    fn ptx_uses_xor_for_partner_index() {
        let ptx =
            compile_sort_kernel_spec(&single_key_spec(DataType::Float64, SortDirection::Asc))
                .unwrap();
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
        let ptx = compile_sort_kernel_spec(&single_key_spec(DataType::Int32, SortDirection::Asc))
            .unwrap();
        // Indices are u32 -> we must see two u32 loads followed by two
        // u32 stores in the swap tail.
        assert!(ptx.contains("ld.global.u32"));
        assert!(ptx.contains("st.global.u32"));
    }

    // ------------------------------------------------------------------
    // Stage 2: multi-key + NULL-aware + shmem variant.
    // ------------------------------------------------------------------

    fn key(dtype: DataType, dir: SortDirection, nullable: bool, nulls_first: bool) -> KeyDesc {
        KeyDesc {
            dtype,
            direction: dir,
            nullable,
            nulls_first,
        }
    }

    /// Multi-key compile: two Int32 keys ASC, DESC. Each key must produce its
    /// own `setp.eq.s32` (the "fall through to next key" branch) and its
    /// own `setp.lt.s32` / `setp.gt.s32` for the value compare. The early-
    /// exit pattern must show: setp.eq -> branch to KEY_<ki>_NEXT.
    #[test]
    fn ptx_multikey_emits_per_key_setp_eq_and_branch() {
        let spec = SortKernelSpec {
            keys: vec![
                key(DataType::Int32, SortDirection::Asc, false, false),
                key(DataType::Int32, SortDirection::Desc, false, false),
            ],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        };
        let ptx = compile_sort_kernel_spec(&spec).unwrap();

        // Per-key setp.eq for the early-equal-skip.
        let eq_count = ptx.matches("setp.eq.s32").count();
        assert!(
            eq_count >= 2,
            "expected per-key setp.eq.s32 for lex early-exit; got {} occurrences in:\n{ptx}",
            eq_count
        );
        // Per-key "next-key" labels.
        assert!(ptx.contains("KEY_0_NEXT:"), "missing KEY_0_NEXT label");
        assert!(ptx.contains("KEY_1_NEXT:"), "missing KEY_1_NEXT label");
        // setp.lt for the value compare (both keys are i32).
        assert!(ptx.contains("setp.lt.s32"));
        assert!(ptx.contains("setp.gt.s32"));
        // Lex falls through to a single DECIDED label.
        assert!(ptx.contains("DECIDED:"));
    }

    /// Multi-key with mixed dtypes — Int64 major, Float32 minor — emits
    /// the right typed mnemonics for each key.
    #[test]
    fn ptx_multikey_mixed_dtypes() {
        let spec = SortKernelSpec {
            keys: vec![
                key(DataType::Int64, SortDirection::Asc, false, false),
                key(DataType::Float32, SortDirection::Desc, false, false),
            ],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        };
        let ptx = compile_sort_kernel_spec(&spec).unwrap();
        assert!(ptx.contains("setp.eq.s64"));
        assert!(ptx.contains("setp.eq.f32"));
        assert!(ptx.contains("ld.global.s64"));
        assert!(ptx.contains("ld.global.f32"));
    }

    /// Null-aware compare must load the validity bitmap and emit a "both
    /// null -> next key" branch + a self-null vs partner-null routing.
    #[test]
    fn ptx_nullable_key_emits_validity_load_and_branch() {
        let spec = SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Asc, true, true)],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        };
        let ptx = compile_sort_kernel_spec(&spec).unwrap();
        // Validity bitmap byte load.
        assert!(
            ptx.contains("ld.global.u8"),
            "nullable key must load validity bits via ld.global.u8; got:\n{ptx}"
        );
        // Bit extraction: shr + and.b32 1
        assert!(ptx.contains("and.b32"));
        // both-null branch -> KEY_0_NEXT
        assert!(ptx.contains("KEY_0_NEXT"));
    }

    /// nulls_first flips the routing direction. We can't easily golden the
    /// exact selp polarity (that's tested by E2E), but we can confirm both
    /// flavours compile and *differ* in PTX content.
    #[test]
    fn ptx_nulls_first_vs_last_differ() {
        let first = compile_sort_kernel_spec(&SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Asc, true, true)],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        })
        .unwrap();
        let last = compile_sort_kernel_spec(&SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Asc, true, false)],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        })
        .unwrap();
        assert_ne!(
            first, last,
            "NULLS FIRST and NULLS LAST must emit different PTX for the routing branch"
        );
    }

    /// Shmem variant must use `ld.shared` / `st.shared` and `bar.sync 0`
    /// (the PTX form of __syncthreads).
    #[test]
    fn ptx_shmem_variant_uses_shared_and_syncthreads() {
        let spec = SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Asc, false, false)],
            layout: SortLayout::Shmem,
            shmem_n_pow2: 128,
        };
        let ptx = compile_sort_kernel_spec(&spec).unwrap();
        assert!(
            ptx.contains("ld.shared.s32") || ptx.contains("ld.shared.u32"),
            "shmem variant must load keys from shared memory; got:\n{ptx}"
        );
        assert!(
            ptx.contains("st.shared.s32") || ptx.contains("st.shared.u32"),
            "shmem variant must store keys to shared memory; got:\n{ptx}"
        );
        assert!(
            ptx.contains("bar.sync 0"),
            "shmem variant must use bar.sync 0 (=__syncthreads); got:\n{ptx}"
        );
        // The shared-memory allocation declaration.
        assert!(ptx.contains(".shared"));
    }

    /// Shmem variant size must be a power of two and <= block_size.
    #[test]
    fn shmem_variant_rejects_non_pow2_or_too_large() {
        // Not power of two.
        let bad_npow2 = SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Asc, false, false)],
            layout: SortLayout::Shmem,
            shmem_n_pow2: 100,
        };
        assert!(compile_sort_kernel_spec(&bad_npow2).is_err());

        // Bigger than block size.
        let too_big = SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Asc, false, false)],
            layout: SortLayout::Shmem,
            shmem_n_pow2: SORT_BLOCK_SIZE * 2,
        };
        assert!(compile_sort_kernel_spec(&too_big).is_err());
    }

    /// The MAX_SORT_KEYS cap is enforced.
    #[test]
    fn rejects_more_than_max_keys() {
        let too_many = SortKernelSpec {
            keys: vec![
                key(DataType::Int32, SortDirection::Asc, false, false);
                MAX_SORT_KEYS + 1
            ],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        };
        assert!(compile_sort_kernel_spec(&too_many).is_err());
    }

    /// Entry name encodes per-key direction + nullability so two specs
    /// don't collide in the module cache.
    #[test]
    fn entry_name_uniqueness_across_specs() {
        let a = sort_kernel_entry_spec(&SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Asc, false, false)],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        })
        .unwrap();
        let b = sort_kernel_entry_spec(&SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Desc, false, false)],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        })
        .unwrap();
        let c = sort_kernel_entry_spec(&SortKernelSpec {
            keys: vec![
                key(DataType::Int32, SortDirection::Asc, false, false),
                key(DataType::Int32, SortDirection::Desc, false, false),
            ],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        })
        .unwrap();
        let d = sort_kernel_entry_spec(&SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Asc, false, false)],
            layout: SortLayout::Shmem,
            shmem_n_pow2: 256,
        })
        .unwrap();
        let e = sort_kernel_entry_spec(&SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Asc, false, false)],
            layout: SortLayout::Shmem,
            shmem_n_pow2: 128,
        })
        .unwrap();
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
        assert_ne!(d, e); // different shmem_n_pow2 -> different module
    }

    // ------------------------------------------------------------------
    // Stage 3: lifted key cap, padded-row routing, Bool/Utf8-via-dict,
    // packed-bit shmem validity.
    // ------------------------------------------------------------------

    /// 8 i32 keys (= 16 b32 regs) is well within the sm_70 budget and must
    /// compile despite exceeding the old Stage-2 hard cap of 4. The compiled
    /// PTX must emit one `setp.eq.s32` block per key (early-equal-skip).
    #[test]
    fn ptx_eight_key_compiles_and_emits_per_key_setp() {
        let spec = SortKernelSpec {
            keys: (0..8)
                .map(|_| key(DataType::Int32, SortDirection::Asc, false, false))
                .collect(),
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        };
        let ptx = compile_sort_kernel_spec(&spec).unwrap();
        // 8 keys -> at least 8 setp.eq.s32 (one per key).
        assert!(
            ptx.matches("setp.eq.s32").count() >= 8,
            "8-key comparator must emit >=8 setp.eq.s32 blocks; PTX:\n{ptx}"
        );
        for ki in 0..8 {
            assert!(
                ptx.contains(&format!("KEY_{}_NEXT:", ki)),
                "missing KEY_{}_NEXT label for 8-key sort",
                ki
            );
        }
    }

    /// The register budget rejects specs that would spill on sm_70.
    /// 9 Int64 keys = 36 b32 regs, > 32 budget.
    #[test]
    fn rejects_over_register_budget() {
        let spec = SortKernelSpec {
            keys: (0..9)
                .map(|_| key(DataType::Int64, SortDirection::Asc, false, false))
                .collect(),
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        };
        assert!(
            compile_sort_kernel_spec(&spec).is_err(),
            "9 i64 keys (36 b32 regs) must be rejected by the sm_70 budget"
        );
    }

    /// Bool keys emit the s32 compare mnemonics (Bool is widened to i32 0/1
    /// after load via `ld.global.u8`).
    #[test]
    fn ptx_bool_key_emits_signed_compare_with_byte_load() {
        let spec = SortKernelSpec {
            keys: vec![key(DataType::Bool, SortDirection::Asc, false, false)],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        };
        let ptx = compile_sort_kernel_spec(&spec).unwrap();
        assert!(
            ptx.contains("ld.global.u8"),
            "Bool key must load via ld.global.u8; got:\n{ptx}"
        );
        assert!(
            ptx.contains("setp.eq.s32"),
            "Bool compare uses the s32 mnemonic; got:\n{ptx}"
        );
        assert!(
            ptx.contains("setp.lt.s32"),
            "Bool compare uses the s32 mnemonic; got:\n{ptx}"
        );
    }

    /// Every multi-key kernel must declare and consult the `is_padded`
    /// parameter so real values colliding with the sentinel are routed past
    /// padded rows (Stage 3 sentinel-tie fix).
    #[test]
    fn ptx_multikey_has_padded_routing_block() {
        let spec = SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Asc, false, false)],
            layout: SortLayout::MultiLaunch,
            shmem_n_pow2: 0,
        };
        let ptx = compile_sort_kernel_spec(&spec).unwrap();
        // The marker comment from emit_padded_route_global.
        assert!(
            ptx.contains("padded-row pre-routing"),
            "Stage 3 padded-routing block missing; PTX:\n{ptx}"
        );
        // Two u8 bitmap loads (self + partner pad bits).
        assert!(
            ptx.matches("ld.global.u8").count() >= 2,
            "padded routing must load self & partner pad bits; PTX:\n{ptx}"
        );
        // is_padded predicate names.
        assert!(ptx.contains("%p_self_pad"));
        assert!(ptx.contains("%p_partner_pad"));
    }

    /// The shmem variant must use a packed-bit u32 validity layout (one
    /// 32-bit word per 32 elements) and an `atom.shared.or.b32` to set the
    /// per-thread bit during the load. Stage-3 8× footprint reduction.
    #[test]
    fn ptx_shmem_uses_packed_bit_validity_and_atomic_or() {
        let spec = SortKernelSpec {
            keys: vec![key(DataType::Int32, SortDirection::Asc, true, true)],
            layout: SortLayout::Shmem,
            shmem_n_pow2: 128,
        };
        let ptx = compile_sort_kernel_spec(&spec).unwrap();
        // sh_v0 must be declared as (128/32)*4 = 16 bytes (packed bits).
        assert!(
            ptx.contains(".shared .align 4 .b8 sh_v0[16];"),
            "Stage 3 shmem must declare packed-bit sh_v0 (16B for n=128); PTX:\n{ptx}"
        );
        // Atomic OR for the per-thread bit set in shmem.
        assert!(
            ptx.contains("atom.shared.or.b32"),
            "packed-bit shmem validity must use atom.shared.or.b32; PTX:\n{ptx}"
        );
        // Validity read uses ld.shared.u32 + shr/and (bfe-style extraction).
        assert!(
            ptx.contains("ld.shared.u32"),
            "packed-bit shmem validity read must use ld.shared.u32; PTX:\n{ptx}"
        );
        // sh_pad packed-bit array also present.
        assert!(
            ptx.contains("sh_pad"),
            "Stage 3 shmem must declare sh_pad packed-bit array; PTX:\n{ptx}"
        );
    }

    /// `key_reg_cost` matches the per-dtype expectations the budget math
    /// in `MAX_SORT_KEYS` documentation builds on. Regression guard if a
    /// new dtype is added without bumping the budget.
    #[test]
    fn key_reg_costs_per_dtype() {
        assert_eq!(key_reg_cost(DataType::Int32), 2);
        assert_eq!(key_reg_cost(DataType::Float32), 2);
        assert_eq!(key_reg_cost(DataType::Bool), 2);
        assert_eq!(key_reg_cost(DataType::Int64), 4);
        assert_eq!(key_reg_cost(DataType::Float64), 4);
    }
}
