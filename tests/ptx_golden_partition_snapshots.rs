// SPDX-License-Identifier: Apache-2.0
//
// FULL-PTX golden snapshot tests for the `partition_reduce_kernel_*` emitter
// family. These exist specifically to make the upcoming jit-dedup refactor
// (collapsing the 11 sibling `partition_reduce_kernel_*` files into one
// spec-parameterised emitter — see `reviews/jit.md`) provably byte-identical.
//
// Each `#[test]` calls one public `compile_*` entry point, runs the emitted
// PTX through `normalize_ptx` (a verbatim copy of the helper in
// `tests/ptx_golden_tests.rs`, duplicated here so this file is self-contained),
// and pins the result with an explicitly-NAMED `insta::assert_snapshot!`. The
// explicit names make the generated `tests/snapshots/*.snap` filenames
// deterministic and stable, so the dedup agent can diff against them.
//
// These tests are HOST-SIDE only — PTX codegen runs entirely on the CPU, needs
// no GPU, and is NOT `#[ignore]`d. They compile under
// `--no-default-features --features cuda-stub` (the same feature set the
// sibling `ptx_golden_tests.rs` already builds under).
//
// Accepting snapshots:
//   On first build run `cargo insta accept` (or `cargo insta review`). After
//   that, the dedup refactor MUST keep every snapshot byte-identical.
//
// Coverage (one snapshot per public `compile_*` entry point, both the base and
// the `_with_spill` twin of every file; arg-taking generators get 1-2
// representative inputs):
//
//   partition_reduce_kernel                 -> base + spill
//   partition_reduce_kernel_i64             -> base + spill
//   partition_reduce_kernel_count           -> base + spill
//   partition_reduce_kernel_count_i64       -> base + spill
//   partition_reduce_kernel_multi(n)        -> n=1,2 base + spill
//   partition_reduce_kernel_multi_i64(n)    -> n=1,2 base + spill
//   partition_reduce_kernel_minmax          -> {Min,Max}x{I32,I64} base + spill
//   partition_reduce_kernel_minmax_i64      -> {Min,Max}x{I32,I64} base + spill
//   partition_reduce_kernel_minmax_float    -> {Min,Max}x{F32,F64} base + spill
//   partition_reduce_kernel_minmax_float_i64-> {Min,Max}x{F32,F64} base + spill
//
// `partition_reduce_kernel_spill_common` is NOT directly snapshotted: it
// exposes only `pub(crate)` helper emitters (`emit_ptx_header`,
// `emit_thread_block_ids`, `emit_spin_backoff`, `emit_spill_bump_*`,
// `emit_loop_next_done`) — none are reachable from an external integration
// test, and none is a full-kernel generator. Its output is covered transitively
// because every `*_with_spill` snapshot below embeds it verbatim.

use craton_bolt::jit::partition_reduce_kernel::{
    compile_partition_reduce_kernel, compile_partition_reduce_kernel_with_spill,
};
use craton_bolt::jit::partition_reduce_kernel_count::{
    compile_partition_reduce_kernel_count, compile_partition_reduce_kernel_count_with_spill,
};
use craton_bolt::jit::partition_reduce_kernel_count_i64::{
    compile_partition_reduce_kernel_count_i64, compile_partition_reduce_kernel_count_i64_with_spill,
};
use craton_bolt::jit::partition_reduce_kernel_i64::{
    compile_partition_reduce_kernel_i64, compile_partition_reduce_kernel_i64_with_spill,
};
use craton_bolt::jit::partition_reduce_kernel_minmax::{
    compile_partition_reduce_kernel_minmax, compile_partition_reduce_kernel_minmax_with_spill,
    MinMaxDtype, MinMaxOp,
};
use craton_bolt::jit::partition_reduce_kernel_minmax_float::{
    compile_partition_reduce_kernel_minmax_float,
    compile_partition_reduce_kernel_minmax_float_with_spill, FloatDtype,
};
use craton_bolt::jit::partition_reduce_kernel_minmax_float_i64::{
    compile_partition_reduce_kernel_minmax_float_i64,
    compile_partition_reduce_kernel_minmax_float_i64_with_spill,
};
use craton_bolt::jit::partition_reduce_kernel_minmax_i64::{
    compile_partition_reduce_kernel_minmax_i64,
    compile_partition_reduce_kernel_minmax_i64_with_spill,
};
use craton_bolt::jit::partition_reduce_kernel_multi::{
    compile_partition_reduce_kernel_multi, compile_partition_reduce_kernel_multi_with_spill,
};
use craton_bolt::jit::partition_reduce_kernel_multi_i64::{
    compile_partition_reduce_kernel_multi_i64, compile_partition_reduce_kernel_multi_i64_with_spill,
};

// ---- PTX normalization (verbatim copy from tests/ptx_golden_tests.rs) --------
//
// Kept byte-identical to the original so both snapshot suites normalize the
// same way. See `tests/ptx_golden_tests.rs` for the full design rationale: the
// short version is that PTX register numbers (`%rdN`, `%rN`, `%fN`, `%fdN`,
// `%rlN`, `%pN`) are issued by a monotonic allocator counter, so any new
// upstream op shifts every later number; this pass rewrites each class's
// numbers into stable first-seen `%rd{N}` / ... placeholders that preserve the
// def/use *flow* (a register-feeds-instruction swap still diffs) while
// absorbing pure-numbering churn. Labels, mnemonics, dtypes, immediates, and
// whitespace are left untouched.

/// Normalize a PTX string for snapshot testing. See module-level note above.
fn normalize_ptx(ptx: &str) -> String {
    use std::collections::HashMap;
    let mut rd: HashMap<u32, u32> = HashMap::new();
    let mut rl: HashMap<u32, u32> = HashMap::new();
    let mut r: HashMap<u32, u32> = HashMap::new();
    let mut f: HashMap<u32, u32> = HashMap::new();
    let mut fd: HashMap<u32, u32> = HashMap::new();
    let mut p: HashMap<u32, u32> = HashMap::new();

    let mut out = String::with_capacity(ptx.len());

    // Skip `// inline asm 0x…` debug comments (per-build address noise).
    for line in ptx.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("// inline asm 0x") {
            continue;
        }

        let bytes = line.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            // Look for `%` followed by a register-class prefix.
            if bytes[i] == b'%' {
                // Try the longest prefixes first so `%rd` wins over `%r`,
                // `%fd` wins over `%f`.
                let rest = &bytes[i + 1..];
                let matched = if rest.starts_with(b"rd") {
                    parse_reg_suffix(&rest[2..]).map(|(num, len)| (2 + len, "rd", num))
                } else if rest.starts_with(b"rl") {
                    parse_reg_suffix(&rest[2..]).map(|(num, len)| (2 + len, "rl", num))
                } else if rest.starts_with(b"fd") {
                    parse_reg_suffix(&rest[2..]).map(|(num, len)| (2 + len, "fd", num))
                } else if rest.starts_with(b"r") {
                    parse_reg_suffix(&rest[1..]).map(|(num, len)| (1 + len, "r", num))
                } else if rest.starts_with(b"f") {
                    parse_reg_suffix(&rest[1..]).map(|(num, len)| (1 + len, "f", num))
                } else if rest.starts_with(b"p") {
                    parse_reg_suffix(&rest[1..]).map(|(num, len)| (1 + len, "p", num))
                } else {
                    None
                };

                if let Some((consumed, class, num)) = matched {
                    let table = match class {
                        "rd" => &mut rd,
                        "rl" => &mut rl,
                        "r" => &mut r,
                        "f" => &mut f,
                        "fd" => &mut fd,
                        "p" => &mut p,
                        _ => unreachable!(),
                    };
                    let next_idx = table.len() as u32;
                    let stable = *table.entry(num).or_insert(next_idx);
                    out.push('%');
                    out.push_str(class);
                    out.push('{');
                    out.push_str(&stable.to_string());
                    out.push('}');
                    i += 1 + consumed;
                    continue;
                }
            }
            out.push(bytes[i] as char);
            i += 1;
        }
        out.push('\n');
    }

    out
}

/// Helper for `normalize_ptx`: parse the digit suffix of a register name.
/// Returns `(parsed_number, digits_consumed)`, or `None` if no digits.
fn parse_reg_suffix(bytes: &[u8]) -> Option<(u32, usize)> {
    let mut end = 0;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == 0 {
        return None;
    }
    if let Some(&next) = bytes.get(end) {
        if next.is_ascii_alphabetic() || next == b'_' {
            return None;
        }
    }
    let num: u32 = std::str::from_utf8(&bytes[..end]).ok()?.parse().ok()?;
    Some((num, end))
}

// ---- base SUM (i32 key) -----------------------------------------------------

#[test]
fn golden_partition_reduce() {
    let ptx = compile_partition_reduce_kernel().expect("compile");
    insta::assert_snapshot!("partition_reduce", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_spill() {
    let ptx = compile_partition_reduce_kernel_with_spill().expect("compile");
    insta::assert_snapshot!("partition_reduce_spill", normalize_ptx(&ptx));
}

// ---- base SUM (i64 key) -----------------------------------------------------

#[test]
fn golden_partition_reduce_i64() {
    let ptx = compile_partition_reduce_kernel_i64().expect("compile");
    insta::assert_snapshot!("partition_reduce_i64", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_i64_spill() {
    let ptx = compile_partition_reduce_kernel_i64_with_spill().expect("compile");
    insta::assert_snapshot!("partition_reduce_i64_spill", normalize_ptx(&ptx));
}

// ---- COUNT (i32 key) --------------------------------------------------------

#[test]
fn golden_partition_reduce_count() {
    let ptx = compile_partition_reduce_kernel_count().expect("compile");
    insta::assert_snapshot!("partition_reduce_count", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_count_spill() {
    let ptx = compile_partition_reduce_kernel_count_with_spill().expect("compile");
    insta::assert_snapshot!("partition_reduce_count_spill", normalize_ptx(&ptx));
}

// ---- COUNT (i64 key) --------------------------------------------------------

#[test]
fn golden_partition_reduce_count_i64() {
    let ptx = compile_partition_reduce_kernel_count_i64().expect("compile");
    insta::assert_snapshot!("partition_reduce_count_i64", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_count_i64_spill() {
    let ptx = compile_partition_reduce_kernel_count_i64_with_spill().expect("compile");
    insta::assert_snapshot!("partition_reduce_count_i64_spill", normalize_ptx(&ptx));
}

// ---- MULTI SUM (i32 key) — parametric value count ---------------------------
// Two representative arities: n_vals=1 (degenerate scalar) and n_vals=2
// (exercises the per-value loop/unroll in the claim + match paths).

#[test]
fn golden_partition_reduce_multi_n1() {
    let ptx = compile_partition_reduce_kernel_multi(1).expect("compile");
    insta::assert_snapshot!("partition_reduce_multi_n1", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_multi_n2() {
    let ptx = compile_partition_reduce_kernel_multi(2).expect("compile");
    insta::assert_snapshot!("partition_reduce_multi_n2", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_multi_n1_spill() {
    let ptx = compile_partition_reduce_kernel_multi_with_spill(1).expect("compile");
    insta::assert_snapshot!("partition_reduce_multi_n1_spill", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_multi_n2_spill() {
    let ptx = compile_partition_reduce_kernel_multi_with_spill(2).expect("compile");
    insta::assert_snapshot!("partition_reduce_multi_n2_spill", normalize_ptx(&ptx));
}

// ---- MULTI SUM (i64 key) — parametric value count ---------------------------

#[test]
fn golden_partition_reduce_multi_i64_n1() {
    let ptx = compile_partition_reduce_kernel_multi_i64(1).expect("compile");
    insta::assert_snapshot!("partition_reduce_multi_i64_n1", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_multi_i64_n2() {
    let ptx = compile_partition_reduce_kernel_multi_i64(2).expect("compile");
    insta::assert_snapshot!("partition_reduce_multi_i64_n2", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_multi_i64_n1_spill() {
    let ptx = compile_partition_reduce_kernel_multi_i64_with_spill(1).expect("compile");
    insta::assert_snapshot!("partition_reduce_multi_i64_n1_spill", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_multi_i64_n2_spill() {
    let ptx = compile_partition_reduce_kernel_multi_i64_with_spill(2).expect("compile");
    insta::assert_snapshot!("partition_reduce_multi_i64_n2_spill", normalize_ptx(&ptx));
}

// ---- MIN/MAX int (i32 key) — {Min,Max} x {Int32,Int64 value} ----------------

#[test]
fn golden_partition_reduce_minmax_min_i32() {
    let ptx =
        compile_partition_reduce_kernel_minmax(MinMaxOp::Min, MinMaxDtype::Int32).expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_min_i32", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_max_i32() {
    let ptx =
        compile_partition_reduce_kernel_minmax(MinMaxOp::Max, MinMaxDtype::Int32).expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_max_i32", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_min_i64() {
    let ptx =
        compile_partition_reduce_kernel_minmax(MinMaxOp::Min, MinMaxDtype::Int64).expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_min_i64", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_max_i64() {
    let ptx =
        compile_partition_reduce_kernel_minmax(MinMaxOp::Max, MinMaxDtype::Int64).expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_max_i64", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_min_i32_spill() {
    let ptx = compile_partition_reduce_kernel_minmax_with_spill(MinMaxOp::Min, MinMaxDtype::Int32)
        .expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_min_i32_spill", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_max_i64_spill() {
    let ptx = compile_partition_reduce_kernel_minmax_with_spill(MinMaxOp::Max, MinMaxDtype::Int64)
        .expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_max_i64_spill", normalize_ptx(&ptx));
}

// ---- MIN/MAX int (i64 key) — {Min,Max} x {Int32,Int64 value} ----------------

#[test]
fn golden_partition_reduce_minmax_i64_min_i32() {
    let ptx = compile_partition_reduce_kernel_minmax_i64(MinMaxOp::Min, MinMaxDtype::Int32)
        .expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_i64_min_i32", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_i64_max_i32() {
    let ptx = compile_partition_reduce_kernel_minmax_i64(MinMaxOp::Max, MinMaxDtype::Int32)
        .expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_i64_max_i32", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_i64_min_i64() {
    let ptx = compile_partition_reduce_kernel_minmax_i64(MinMaxOp::Min, MinMaxDtype::Int64)
        .expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_i64_min_i64", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_i64_max_i64() {
    let ptx = compile_partition_reduce_kernel_minmax_i64(MinMaxOp::Max, MinMaxDtype::Int64)
        .expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_i64_max_i64", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_i64_min_i32_spill() {
    let ptx =
        compile_partition_reduce_kernel_minmax_i64_with_spill(MinMaxOp::Min, MinMaxDtype::Int32)
            .expect("compile");
    insta::assert_snapshot!(
        "partition_reduce_minmax_i64_min_i32_spill",
        normalize_ptx(&ptx)
    );
}

#[test]
fn golden_partition_reduce_minmax_i64_max_i64_spill() {
    let ptx =
        compile_partition_reduce_kernel_minmax_i64_with_spill(MinMaxOp::Max, MinMaxDtype::Int64)
            .expect("compile");
    insta::assert_snapshot!(
        "partition_reduce_minmax_i64_max_i64_spill",
        normalize_ptx(&ptx)
    );
}

// ---- MIN/MAX float (i32 key) — {Min,Max} x {Float32,Float64 value} ----------
// These take the CAS-loop path (no native atom.shared.min/max.f*) — the most
// refactor-sensitive variant, so all four op x dtype combos are snapshotted.

#[test]
fn golden_partition_reduce_minmax_float_min_f32() {
    let ptx = compile_partition_reduce_kernel_minmax_float(MinMaxOp::Min, FloatDtype::Float32)
        .expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_float_min_f32", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_float_max_f32() {
    let ptx = compile_partition_reduce_kernel_minmax_float(MinMaxOp::Max, FloatDtype::Float32)
        .expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_float_max_f32", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_float_min_f64() {
    let ptx = compile_partition_reduce_kernel_minmax_float(MinMaxOp::Min, FloatDtype::Float64)
        .expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_float_min_f64", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_float_max_f64() {
    let ptx = compile_partition_reduce_kernel_minmax_float(MinMaxOp::Max, FloatDtype::Float64)
        .expect("compile");
    insta::assert_snapshot!("partition_reduce_minmax_float_max_f64", normalize_ptx(&ptx));
}

#[test]
fn golden_partition_reduce_minmax_float_min_f32_spill() {
    let ptx =
        compile_partition_reduce_kernel_minmax_float_with_spill(MinMaxOp::Min, FloatDtype::Float32)
            .expect("compile");
    insta::assert_snapshot!(
        "partition_reduce_minmax_float_min_f32_spill",
        normalize_ptx(&ptx)
    );
}

#[test]
fn golden_partition_reduce_minmax_float_max_f64_spill() {
    let ptx =
        compile_partition_reduce_kernel_minmax_float_with_spill(MinMaxOp::Max, FloatDtype::Float64)
            .expect("compile");
    insta::assert_snapshot!(
        "partition_reduce_minmax_float_max_f64_spill",
        normalize_ptx(&ptx)
    );
}

// ---- MIN/MAX float (i64 key) — {Min,Max} x {Float32,Float64 value} ----------

#[test]
fn golden_partition_reduce_minmax_float_i64_min_f32() {
    let ptx = compile_partition_reduce_kernel_minmax_float_i64(MinMaxOp::Min, FloatDtype::Float32)
        .expect("compile");
    insta::assert_snapshot!(
        "partition_reduce_minmax_float_i64_min_f32",
        normalize_ptx(&ptx)
    );
}

#[test]
fn golden_partition_reduce_minmax_float_i64_max_f32() {
    let ptx = compile_partition_reduce_kernel_minmax_float_i64(MinMaxOp::Max, FloatDtype::Float32)
        .expect("compile");
    insta::assert_snapshot!(
        "partition_reduce_minmax_float_i64_max_f32",
        normalize_ptx(&ptx)
    );
}

#[test]
fn golden_partition_reduce_minmax_float_i64_min_f64() {
    let ptx = compile_partition_reduce_kernel_minmax_float_i64(MinMaxOp::Min, FloatDtype::Float64)
        .expect("compile");
    insta::assert_snapshot!(
        "partition_reduce_minmax_float_i64_min_f64",
        normalize_ptx(&ptx)
    );
}

#[test]
fn golden_partition_reduce_minmax_float_i64_max_f64() {
    let ptx = compile_partition_reduce_kernel_minmax_float_i64(MinMaxOp::Max, FloatDtype::Float64)
        .expect("compile");
    insta::assert_snapshot!(
        "partition_reduce_minmax_float_i64_max_f64",
        normalize_ptx(&ptx)
    );
}

#[test]
fn golden_partition_reduce_minmax_float_i64_min_f32_spill() {
    let ptx = compile_partition_reduce_kernel_minmax_float_i64_with_spill(
        MinMaxOp::Min,
        FloatDtype::Float32,
    )
    .expect("compile");
    insta::assert_snapshot!(
        "partition_reduce_minmax_float_i64_min_f32_spill",
        normalize_ptx(&ptx)
    );
}

#[test]
fn golden_partition_reduce_minmax_float_i64_max_f64_spill() {
    let ptx = compile_partition_reduce_kernel_minmax_float_i64_with_spill(
        MinMaxOp::Max,
        FloatDtype::Float64,
    )
    .expect("compile");
    insta::assert_snapshot!(
        "partition_reduce_minmax_float_i64_max_f64_spill",
        normalize_ptx(&ptx)
    );
}
