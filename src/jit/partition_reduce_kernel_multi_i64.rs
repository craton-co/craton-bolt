// SPDX-License-Identifier: Apache-2.0

//! Per-partition shared-memory reduce kernel — **i64-key, multi-value
//! SUM**. The intersection of `partition_reduce_kernel_multi` (i32-key,
//! N-value) and `partition_reduce_kernel_i64` (i64-key, single-value).
//!
//! Used by the two-key multi-aggregate Tier-2.1 path: pack two i32 keys
//! into an i64 (per `groupby.rs::pack_keys`), partition + scatter
//! through the i64 pipeline, then this kernel reduces each partition
//! into N parallel f64 SUMs keyed by i64.
//!
//! ## Shared-memory layout per block
//!
//! block_keys    : i64 × 1024 =  8 KiB
//! block_vals_0  : f64 × 1024 =  8 KiB
//! block_vals_1  : f64 × 1024 =  8 KiB   (only if n_vals ≥ 2)
//! block_vals_2  : f64 × 1024 =  8 KiB   (only if n_vals ≥ 3)
//! block_vals_3  : f64 × 1024 =  8 KiB   (only if n_vals ≥ 4)
//! block_set     : u32 × 1024 =  4 KiB
//!
//! Totals: N=1 → 20 KiB ; N=2 → 28 KiB ; N=3 → 36 KiB ; N=4 → 44 KiB.
//! All within sm_70's 48 KiB static-shared-mem budget.

use std::fmt::Write;

use crate::error::{BoltError, BoltResult};

pub const BLOCK_GROUPS: u32 = 1024;
pub const BLOCK_THREADS: u32 = 256;
pub const MAX_VALS: u32 = 4;
pub const NUM_PARTITIONS: u32 = 4096;
const MAX_PROBES: u32 = BLOCK_GROUPS;

/// Per-iteration `nanosleep.u32` operand for the collision-advance path
/// (sm_70+). See `partition_reduce_kernel::SPIN_BACKOFF_NS` for full
/// rationale. TODO(perf): exponential back-off.
const SPIN_BACKOFF_NS: u32 = 32;

pub fn kernel_entry(n_vals: u32) -> String {
    format!("bolt_partition_reduce_multi_sum_i64_{}", n_vals)
}

pub fn compile_partition_reduce_kernel_multi_i64(n_vals: u32) -> BoltResult<String> {
    if n_vals == 0 || n_vals > MAX_VALS {
        return Err(BoltError::Other(format!(
            "partition_reduce_kernel_multi_i64: n_vals must be 1..={MAX_VALS}, got {n_vals}"
        )));
    }
    let mut ptx = String::new();
    let entry = kernel_entry(n_vals);
    let entry = entry.as_str();
    let block_groups = BLOCK_GROUPS;
    let mask = BLOCK_GROUPS - 1;
    let block_threads = BLOCK_THREADS;
    let keys_bytes = BLOCK_GROUPS * 8; // i64 keys
    let vals_bytes = BLOCK_GROUPS * 8;
    let set_bytes = BLOCK_GROUPS * 4;
    let max_probes = MAX_PROBES;

    writeln!(ptx, ".version 7.5").map_err(write_err)?;
    writeln!(ptx, ".target sm_70").map_err(write_err)?;
    writeln!(ptx, ".address_size 64").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(
        ptx,
        ".shared .align 8 .b8 block_keys_buf[{bytes}];",
        bytes = keys_bytes
    )
    .map_err(write_err)?;
    for j in 0..n_vals {
        writeln!(
            ptx,
            ".shared .align 8 .b8 block_vals{j}_buf[{bytes}];",
            j = j,
            bytes = vals_bytes
        )
        .map_err(write_err)?;
    }
    writeln!(
        ptx,
        ".shared .align 4 .b8 block_set_buf[{bytes}];",
        bytes = set_bytes
    )
    .map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Kernel signature: keys + N val ptrs + offsets + out_keys + N out_val ptrs + out_set = 4 + 2N
    writeln!(ptx, ".visible .entry {entry}(").map_err(write_err)?;
    let total_params = 4 + 2 * n_vals;
    for p in 0..total_params {
        let trailing = if p == total_params - 1 { "" } else { "," };
        writeln!(ptx, "\t.param .u64 {entry}_param_{p}{trailing}").map_err(write_err)?;
    }
    writeln!(ptx, ")").map_err(write_err)?;
    writeln!(ptx, "{{").map_err(write_err)?;

    writeln!(ptx, "\t.reg .pred  %p<24>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b32   %r<96>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .b64   %rd<128>;").map_err(write_err)?;
    writeln!(ptx, "\t.reg .f64   %fd<32>;").map_err(write_err)?;
    // Operand register for the per-collision `nanosleep.u32` back-off.
    writeln!(ptx, "\t.reg .u32   %nstime;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    writeln!(ptx, "\tmov.u32 %r0, %ctaid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r1, %ntid.x;").map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r2, %tid.x;").map_err(write_err)?;

    // Shared bases: %rd0=keys, %rd1..%rd{n_vals}=vals_j, %rd{n_vals+1}=set
    writeln!(ptx, "\tmov.u64 %rd0, block_keys_buf;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd = 1 + j;
        writeln!(ptx, "\tmov.u64 %rd{rd}, block_vals{j}_buf;").map_err(write_err)?;
    }
    let rd_set = 1 + n_vals;
    writeln!(ptx, "\tmov.u64 %rd{rd_set}, block_set_buf;").map_err(write_err)?;

    // Global ptrs.
    let rd_pkeys = rd_set + 1;
    let rd_pvals_base = rd_pkeys + 1;
    let rd_poff = rd_pvals_base + n_vals;
    let rd_okeys = rd_poff + 1;
    let rd_ovals_base = rd_okeys + 1;
    let rd_oset = rd_ovals_base + n_vals;

    writeln!(ptx, "\tld.param.u64 %rd{rd_pkeys}, [{entry}_param_0];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd{rd_pkeys}, %rd{rd_pkeys};").map_err(write_err)?;
    for j in 0..n_vals {
        let rd = rd_pvals_base + j;
        let p = 1 + j;
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    let p_off = 1 + n_vals;
    writeln!(ptx, "\tld.param.u64 %rd{rd_poff}, [{entry}_param_{p_off}];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd{rd_poff}, %rd{rd_poff};").map_err(write_err)?;
    let p_ok = 2 + n_vals;
    writeln!(ptx, "\tld.param.u64 %rd{rd_okeys}, [{entry}_param_{p_ok}];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd{rd_okeys}, %rd{rd_okeys};").map_err(write_err)?;
    for j in 0..n_vals {
        let rd = rd_ovals_base + j;
        let p = 3 + n_vals + j;
        writeln!(ptx, "\tld.param.u64 %rd{rd}, [{entry}_param_{p}];").map_err(write_err)?;
        writeln!(ptx, "\tcvta.to.global.u64 %rd{rd}, %rd{rd};").map_err(write_err)?;
    }
    let p_os = 3 + 2 * n_vals;
    writeln!(ptx, "\tld.param.u64 %rd{rd_oset}, [{entry}_param_{p_os}];").map_err(write_err)?;
    writeln!(ptx, "\tcvta.to.global.u64 %rd{rd_oset}, %rd{rd_oset};").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Read partition slice [start, end).
    writeln!(ptx, "\tmul.wide.u32 %rd80, %r0, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd81, %rd{rd_poff}, %rd80;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r10, [%rd81];").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd82, %rd81, 4;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.u32 %r11, [%rd82];").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 1: zero shared.
    writeln!(ptx, "\tmov.u32 %r20, %r2;").map_err(write_err)?;
    writeln!(ptx, "ZERO_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p0, %r20, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p0 bra ZERO_DONE;").map_err(write_err)?;
    // block_keys[s] = 0 (i64)
    writeln!(ptx, "\tmul.wide.u32 %rd83, %r20, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd84, %rd0, %rd83;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd84], 0;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_v = 1 + j;
        writeln!(ptx, "\tadd.s64 %rd86, %rd{rd_v}, %rd83;").map_err(write_err)?;
        writeln!(ptx, "\tst.shared.u64 [%rd86], 0;").map_err(write_err)?;
    }
    writeln!(ptx, "\tmul.wide.u32 %rd85, %r20, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd87, %rd{rd_set}, %rd85;").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u32 [%rd87], 0;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tadd.u32 %r20, %r20, {bt};",
        bt = block_threads
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra ZERO_TOP;").map_err(write_err)?;
    writeln!(ptx, "ZERO_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 2: probe + N-way sum (i64 keys).
    writeln!(ptx, "\tadd.u32 %r30, %r10, %r2;").map_err(write_err)?;
    writeln!(ptx, "LOOP_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ge.u32 %p1, %r30, %r11;").map_err(write_err)?;
    writeln!(ptx, "\t@%p1 bra LOOP_DONE;").map_err(write_err)?;

    // key = partition_keys[i] (i64)
    writeln!(ptx, "\tmul.wide.u32 %rd88, %r30, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd89, %rd{rd_pkeys}, %rd88;").map_err(write_err)?;
    writeln!(ptx, "\tld.global.s64 %rd60, [%rd89];").map_err(write_err)?; // %rd60 = key

    // val_j = partition_vals_j[i]
    for j in 0..n_vals {
        let rd_v = rd_pvals_base + j;
        let fd_v = j;
        writeln!(ptx, "\tadd.s64 %rd91, %rd{rd_v}, %rd88;").map_err(write_err)?;
        writeln!(ptx, "\tld.global.f64 %fd{fd_v}, [%rd91];").map_err(write_err)?;
    }

    // slot = (low_32(key)) & mask
    writeln!(ptx, "\tcvt.u32.u64 %r31, %rd60;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r32, %r31, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r33, 0;").map_err(write_err)?;

    writeln!(ptx, "PROBE_TOP:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r33, %r33, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.gt.u32 %p2, %r33, {mp};",
        mp = max_probes
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p2 bra LOOP_NEXT;").map_err(write_err)?;

    // Slot addresses. Keys are i64 (×8); set is u32 (×4); vals are f64 (×8).
    writeln!(ptx, "\tmul.wide.u32 %rd92, %r32, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd93, %rd{rd_set}, %rd92;").map_err(write_err)?; // addr_set
    writeln!(ptx, "\tmul.wide.u32 %rd95, %r32, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd94, %rd0, %rd95;").map_err(write_err)?; // addr_key (i64)

    writeln!(ptx, "\tatom.shared.cas.b32 %r34, [%rd93], 0, 1;").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s32 %p3, %r34, 0;").map_err(write_err)?;
    writeln!(ptx, "\t@%p3 bra CLAIM;").map_err(write_err)?;

    // Slot occupied — membar.cta orders the set CAS against the i64
    // key load (different addresses). PTX sm_70 has no inter-address
    // ordering; without this fence a racing thread can see set==1 with
    // a zero key and false-match.
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd61, [%rd94];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.eq.s64 %p4, %rd61, %rd60;").map_err(write_err)?;
    writeln!(ptx, "\t@%p4 bra MATCH;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r32, %r32, 1;").map_err(write_err)?;
    writeln!(
        ptx,
        "\tand.b32 %r32, %r32, 0x{mask:X};",
        mask = mask
    )
    .map_err(write_err)?;
    // Occupancy-friendly back-off on the collision-advance path.
    writeln!(
        ptx,
        "\tmov.u32 %nstime, {ns};",
        ns = SPIN_BACKOFF_NS
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tnanosleep.u32 %nstime;").map_err(write_err)?;
    writeln!(ptx, "\tbra PROBE_TOP;").map_err(write_err)?;

    writeln!(ptx, "CLAIM:").map_err(write_err)?;
    writeln!(ptx, "\tst.shared.u64 [%rd94], %rd60;").map_err(write_err)?;
    writeln!(ptx, "\tmembar.cta;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_v = 1 + j;
        let fd_v = j;
        writeln!(ptx, "\tadd.s64 %rd96, %rd{rd_v}, %rd95;").map_err(write_err)?;
        let fd_scratch = 16 + j;
        writeln!(
            ptx,
            "\tatom.shared.add.f64 %fd{fd_scratch}, [%rd96], %fd{fd_v};"
        )
        .map_err(write_err)?;
    }
    writeln!(ptx, "\tbra LOOP_NEXT;").map_err(write_err)?;

    writeln!(ptx, "MATCH:").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_v = 1 + j;
        let fd_v = j;
        writeln!(ptx, "\tadd.s64 %rd96, %rd{rd_v}, %rd95;").map_err(write_err)?;
        let fd_scratch = 24 + j;
        writeln!(
            ptx,
            "\tatom.shared.add.f64 %fd{fd_scratch}, [%rd96], %fd{fd_v};"
        )
        .map_err(write_err)?;
    }

    writeln!(ptx, "LOOP_NEXT:").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r30, %r30, %r1;").map_err(write_err)?;
    writeln!(ptx, "\tbra LOOP_TOP;").map_err(write_err)?;
    writeln!(ptx, "LOOP_DONE:").map_err(write_err)?;
    writeln!(ptx, "\tbar.sync 0;").map_err(write_err)?;
    writeln!(ptx).map_err(write_err)?;

    // Phase 3: export populated slots.
    writeln!(
        ptx,
        "\tmul.lo.u32 %r40, %r0, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tmov.u32 %r41, %r2;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_TOP:").map_err(write_err)?;
    writeln!(
        ptx,
        "\tsetp.ge.u32 %p5, %r41, {bg};",
        bg = block_groups
    )
    .map_err(write_err)?;
    writeln!(ptx, "\t@%p5 bra EXPORT_DONE;").map_err(write_err)?;
    writeln!(ptx, "\tadd.u32 %r42, %r40, %r41;").map_err(write_err)?;

    // Load shared slot's i64 key + N vals + set.
    writeln!(ptx, "\tmul.wide.u32 %rd99, %r41, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd98, %rd0, %rd99;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.s64 %rd62, [%rd98];").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_v = 1 + j;
        let fd_v = j;
        writeln!(ptx, "\tadd.s64 %rd100, %rd{rd_v}, %rd99;").map_err(write_err)?;
        writeln!(ptx, "\tld.shared.f64 %fd{fd_v}, [%rd100];").map_err(write_err)?;
    }
    writeln!(ptx, "\tmul.wide.u32 %rd97, %r41, 4;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd101, %rd{rd_set}, %rd97;").map_err(write_err)?;
    writeln!(ptx, "\tld.shared.u32 %r44, [%rd101];").map_err(write_err)?;
    writeln!(ptx, "\tsetp.ne.s32 %p6, %r44, 0;").map_err(write_err)?;
    writeln!(ptx, "\tselp.u32 %r45, 1, 0, %p6;").map_err(write_err)?;

    // Store: i64 key, N f64 vals, u8 set.
    writeln!(ptx, "\tmul.wide.u32 %rd104, %r42, 8;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd103, %rd{rd_okeys}, %rd104;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.s64 [%rd103], %rd62;").map_err(write_err)?;
    for j in 0..n_vals {
        let rd_ov = rd_ovals_base + j;
        let fd_v = j;
        writeln!(ptx, "\tadd.s64 %rd105, %rd{rd_ov}, %rd104;").map_err(write_err)?;
        writeln!(ptx, "\tst.global.f64 [%rd105], %fd{fd_v};").map_err(write_err)?;
    }
    writeln!(ptx, "\tcvt.u64.u32 %rd106, %r42;").map_err(write_err)?;
    writeln!(ptx, "\tadd.s64 %rd107, %rd{rd_oset}, %rd106;").map_err(write_err)?;
    writeln!(ptx, "\tst.global.u8 [%rd107], %r45;").map_err(write_err)?;

    writeln!(
        ptx,
        "\tadd.u32 %r41, %r41, {bt};",
        bt = block_threads
    )
    .map_err(write_err)?;
    writeln!(ptx, "\tbra EXPORT_TOP;").map_err(write_err)?;
    writeln!(ptx, "EXPORT_DONE:").map_err(write_err)?;

    writeln!(ptx, "\tret;").map_err(write_err)?;
    writeln!(ptx, "}}").map_err(write_err)?;

    Ok(ptx)
}

fn write_err(e: std::fmt::Error) -> BoltError {
    BoltError::Other(format!(
        "partition_reduce_kernel_multi_i64: write failed: {}",
        e
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compiles_for_all_n_vals() {
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi_i64(n)
                .unwrap_or_else(|e| panic!("n_vals={n} should compile: {e}"));
            assert!(!ptx.is_empty());
        }
    }

    #[test]
    fn rejects_bad_n_vals() {
        assert!(compile_partition_reduce_kernel_multi_i64(0).is_err());
        assert!(compile_partition_reduce_kernel_multi_i64(MAX_VALS + 1).is_err());
    }

    #[test]
    fn uses_i64_key_loads_and_stores() {
        let ptx = compile_partition_reduce_kernel_multi_i64(2).unwrap();
        assert!(ptx.contains("ld.global.s64"));
        assert!(ptx.contains("st.global.s64"));
        assert!(ptx.contains("ld.shared.s64"));
        assert!(ptx.contains("st.shared.u64"));
    }

    #[test]
    fn emits_n_atomic_adds_per_path() {
        // CLAIM + MATCH each issue n_vals atom.shared.add.f64.
        for n in 1..=MAX_VALS {
            let ptx = compile_partition_reduce_kernel_multi_i64(n).unwrap();
            let c = ptx.matches("atom.shared.add.f64").count();
            assert_eq!(c, (2 * n) as usize, "n={n}: want {} atomics", 2 * n);
        }
    }
}
