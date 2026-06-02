// SPDX-License-Identifier: Apache-2.0
//
// Golden snapshot tests for emitted PTX. Updates to these tests are intentional
// codegen contract changes — review the PTX diff carefully before accepting.
//
// Strategy (two-layer):
//
//   1. Substring assertions (`assert!(ptx.contains("…"))`) pin the *behavioral
//      contract* — which mnemonics, which dtypes, which structural markers
//      must be present. These catch "dropped mnemonic" regressions
//      (e.g. C6: missing s32→s64 widening; C7: missing probe-loop bound) and
//      are stable across cosmetic churn.
//
//   2. `insta` snapshot tests (`assert_snapshot!(normalize_ptx(&ptx))`) pin
//      the *register flow* — they catch regressions where a refactor changes
//      which register feeds which instruction, not just whether the mnemonic
//      is present. We pipe the PTX through `normalize_ptx` first to rewrite
//      `%rdN` / `%rN` / `%fN` / `%fdN` / `%pN` into stable `%rd{N}` /
//      `%r{N}` / `%f{N}` / `%fd{N}` / `%p{N}` placeholders so allocator
//      counter drift (every new upstream op shifts every later number) does
//      not flap the snapshots. Labels are NOT normalized — they're load-
//      bearing for jump correctness.
//
// Accepting snapshots:
//   On first build (and after intentional codegen changes), run
//   `cargo insta accept` (or `cargo insta review` for interactive review).
//   Snapshot files live under `tests/snapshots/`.

use craton_bolt::jit::agg_kernels::{compile_reduction_kernel, ReduceOp};
use craton_bolt::jit::compile_ptx;
use craton_bolt::jit::float_atomics::compile_groupby_float_atomic_kernel;
use craton_bolt::jit::hash_kernels::{
    compile_groupby_agg_kernel, compile_groupby_keys_kernel,
};
use craton_bolt::jit::prefix_scan::{
    compile_prefix_scan_kernel, compile_prefix_scan_kernel_blelloch,
    compile_prefix_scan_kernel_lookback,
};
use craton_bolt::jit::string_kernel::{
    compile_length_gather_kernel, compile_varwidth_len_pass, compile_varwidth_write_pass,
};
use craton_bolt::jit::window_kernel::{
    compile_boundary_flag_kernel, compile_segmented_scan_kernel, WINDOW_BLOCK_SIZE,
};
use craton_bolt::plan::ScalarFnKind;
use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, PhysicalPlan, Schema, TimeUnit,
};

// ---- PTX normalization (for `insta` snapshots) ------------------------------

/// Normalize a PTX string for snapshot testing.
///
/// PTX register names (`%rdN`, `%rN`, `%fN`, `%fdN`, `%pN`) are issued by a
/// monotonic counter inside `RegAlloc`. Any new upstream compute op shifts
/// every later register number, which would flap a byte-equality snapshot on
/// every codegen refactor. This pass rewrites those numbers per-class into
/// stable `%rd{N}` / `%r{N}` / `%f{N}` / `%fd{N}` / `%p{N}` placeholders
/// while preserving the *flow* (i.e. the relationship between defs and uses):
/// the Nth distinct `%rdK` encountered in the input becomes `%rd{N}`, so a
/// refactor that swaps which register feeds which instruction will still
/// produce a snapshot diff. Across PTXAS / NVRTC version drift, however,
/// pure numbering churn is absorbed.
///
/// We use `{N}` (curly braces) rather than the more natural-looking `<N>`
/// because real PTX *already* uses the angle-bracket form for register-vector
/// declarations like `.reg .b64 %rd<24>;` (the `<24>` is the vector size).
/// Using a syntax that PTX itself never emits keeps placeholders visually
/// unambiguous from declarations in the snapshot diff.
///
/// What we strip / normalize:
///   * `%rd\d+` → `%rd{N}` (64-bit integer / pointer registers)
///   * `%rl\d+` → `%rl{N}` (64-bit "long" integer regs, hash kernel use)
///   * `%r\d+`  → `%r{N}`  (32-bit integer registers)
///   * `%f\d+`  → `%f{N}`  (32-bit float registers)
///   * `%fd\d+` → `%fd{N}` (64-bit float registers)
///   * `%p\d+`  → `%p{N}`  (predicate registers)
///   * `// inline asm 0x…` debug comments dropped
///
/// What we DO NOT touch:
///   * `.reg` vector declarations like `%rd<24>;` (no digit suffix → never
///     matches; left as-is so the snapshot still records register-count
///     changes, which ARE a real codegen contract).
///   * Labels (`PROBE_LOOP:`, `bra DONE`, …) — load-bearing for jump
///     correctness. A label rename IS a real diff.
///   * Instruction mnemonics, dtypes, immediates.
///   * Whitespace (left as-is so snapshot diffs read naturally).
fn normalize_ptx(ptx: &str) -> String {
    // Per-class assignment table: first-seen original number → stable index.
    // We track them separately so `%rd1` and `%r1` don't collide.
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
                    // Use a 0-based stable index so the *first* register
                    // seen in each class is `{0}`. This makes diffs read
                    // top-to-bottom in declaration order.
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
/// Requires the digit run to be followed by a non-identifier byte so we
/// don't accidentally truncate `%foo` into a register match.
fn parse_reg_suffix(bytes: &[u8]) -> Option<(u32, usize)> {
    let mut end = 0;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if end == 0 {
        return None;
    }
    // Reject if the next byte is an identifier continuation (letter / digit /
    // underscore) — we already consumed all digits, so the only way this
    // fires is `%r1foo`, which isn't a real PTX register.
    if let Some(&next) = bytes.get(end) {
        if next.is_ascii_alphabetic() || next == b'_' {
            return None;
        }
    }
    // Safe: digits only.
    let num: u32 = std::str::from_utf8(&bytes[..end]).ok()?.parse().ok()?;
    Some((num, end))
}

// ---- Fixture ----------------------------------------------------------------

/// Schema covering every dtype the projection-path tests need:
/// * `int_col`  — Int32 (for s32 load/store, sum widening)
/// * `f64_col`  — Float64 (for f64 load/store, mul)
/// * `k`        — Int32 group key
/// * `v`        — Int32 aggregate input
/// * `a`, `b`, `c` — Int32 columns for compound predicates
fn fixture_schema() -> Schema {
    Schema::new(vec![
        Field {
            name: "int_col".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "f64_col".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
        Field {
            name: "k".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "v".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "a".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "b".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "c".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ])
}

fn fixture_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("t", fixture_schema())
}

/// Build PTX for a SQL query whose lowered plan is a `Projection`. Panics if
/// the plan isn't a single projection kernel (aggregation queries need to
/// call the per-kernel compile_* functions directly because the projection
/// path doesn't cover SUM / MIN / GROUP BY).
fn build_ptx_for(sql: &str) -> String {
    let provider = fixture_provider();
    let plan = parse_sql(sql, &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let kernel = match &phys {
        PhysicalPlan::Projection { kernel, .. } => kernel,
        other => panic!(
            "build_ptx_for: expected Projection plan for `{sql}`, got {other:?}; \
             aggregation queries must compile their kernels directly"
        ),
    };
    compile_ptx(kernel, "bolt_test_kernel").expect("compile_ptx")
}

// ---- Tests: scalar projection ----------------------------------------------

#[test]
fn golden_scalar_projection_int32_smoke() {
    let ptx = build_ptx_for("SELECT int_col + 1 FROM t");
    // Header.
    assert!(ptx.contains(".version 7.5"), "missing .version\n{ptx}");
    assert!(ptx.contains(".target sm_70"), "missing .target\n{ptx}");
    assert!(
        ptx.contains(".address_size 64"),
        "missing .address_size\n{ptx}"
    );
    // Entry name comes from the `kernel_name` arg.
    assert!(
        ptx.contains(".visible .entry bolt_test_kernel"),
        "wrong entry name\n{ptx}"
    );
    // `int_col + 1` widens to s64 (int literals are Int64 by parse default),
    // so the load is s32 but the arithmetic and store run at s64. The input
    // column is read-only, so the load goes through the read-only cache.
    assert!(ptx.contains("ld.global.nc.s32"), "missing s32 read-only-cache load\n{ptx}");
    assert!(
        ptx.contains("cvt.s64.s32"),
        "missing s32->s64 widening for literal-1 add\n{ptx}"
    );
    assert!(ptx.contains("add.s64"), "missing s64 add\n{ptx}");
    assert!(ptx.contains("st.global.s64"), "missing s64 store\n{ptx}");
}

#[test]
fn golden_scalar_projection_float64_smoke() {
    let ptx = build_ptx_for("SELECT f64_col * 2.0 FROM t");
    assert!(ptx.contains(".version 7.5"));
    assert!(ptx.contains(".target sm_70"));
    assert!(ptx.contains("ld.global.nc.f64"), "missing f64 read-only-cache load\n{ptx}");
    assert!(ptx.contains("mul.f64"), "missing f64 multiply\n{ptx}");
    assert!(ptx.contains("st.global.f64"), "missing f64 store\n{ptx}");
}

// ---- Tests: predicate (filter) ---------------------------------------------

#[test]
fn golden_predicate_filter_int32_smoke() {
    let ptx = build_ptx_for("SELECT int_col FROM t WHERE int_col = 5");
    // The literal `5` is parsed as Int64, so the operands unify to Int64
    // and the comparison emits `setp.eq.s64` (NOT s32). This matches the
    // existing assertion in `e2e_tests::ptx_with_predicate_contains_gate_before_store`.
    assert!(
        ptx.contains("setp.eq.s64"),
        "expected setp.eq.s64 (int literal widens to Int64)\n{ptx}"
    );
    // The codegen pattern for a predicated kernel: after the predicate
    // register is produced, we synthesise a gate `setp.eq.s32 %pX, <pred>, 0;
    // @%pX bra DONE;`. The `bra DONE` is the conditional branch we want
    // present and ordered before any store.
    let gate_pos = ptx.find("bra DONE").expect("missing conditional bra DONE");
    let store_pos = ptx.find("st.global").expect("missing store");
    assert!(
        gate_pos < store_pos,
        "predicate gate must precede the store\n{ptx}"
    );
}

#[test]
fn golden_predicate_filter_and_or() {
    let ptx = build_ptx_for("SELECT a FROM t WHERE a = 1 AND (b = 2 OR c = 3)");
    // Three comparisons total — one per `col = literal` (all widen to s64).
    let n_setp_eq = ptx.matches("setp.eq.s64").count();
    assert!(
        n_setp_eq >= 3,
        "expected >=3 setp.eq.s64 (one per comparison), got {n_setp_eq}\n{ptx}"
    );
    // Logical AND/OR over Bool operands are emitted as `and.b32` / `or.b32`
    // (boolean values live in the `%r` (32-bit integer) register class with
    // 0/1 values), NOT as `and.pred` / `or.pred`. This is the contract — if
    // the codegen switches to predicate-class booleans, update these.
    assert!(
        ptx.contains("and.b32"),
        "missing and.b32 (logical AND on Bool)\n{ptx}"
    );
    assert!(
        ptx.contains("or.b32"),
        "missing or.b32 (logical OR on Bool)\n{ptx}"
    );
}

#[test]
fn golden_predicate_filter_not_comparison() {
    // `WHERE NOT (int_col > 1)` lowers `NOT` to `Op::Not`, which emits a
    // single `xor.b32 dst, src, 1` over the comparison's Bool register.
    // The comparison itself widens to s64 (int literal `1` is Int64).
    let ptx = build_ptx_for("SELECT int_col FROM t WHERE NOT (int_col > 1)");
    // The inner comparison must still be present.
    assert!(
        ptx.contains("setp.gt.s64"),
        "expected the inner `int_col > 1` comparison `setp.gt.s64`\n{ptx}"
    );
    // The negation is the load-bearing assertion: `xor.b32 %r, %r, 1`.
    assert!(
        ptx.contains("xor.b32"),
        "expected `xor.b32` from the NOT negation\n{ptx}"
    );
    // The negated Bool is the predicate, so the kernel still gates the
    // store behind a `bra DONE` placed before any store.
    let gate_pos = ptx.find("bra DONE").expect("missing conditional bra DONE");
    let store_pos = ptx.find("st.global").expect("missing store");
    assert!(
        gate_pos < store_pos,
        "predicate gate must precede the store\n{ptx}"
    );
}

// ---- Tests: reduction-kernel widening (wave-3 regression) ------------------

#[test]
fn golden_sum_int32_widens_to_s64_in_kernel() {
    // SELECT SUM(int_col) FROM t goes through the scalar reduction kernel,
    // not the projection path. The kernel must:
    //   1. Sign-extend each loaded s32 value to s64 (`cvt.s64.s32`), and
    //   2. Combine in s64 (`add.s64`).
    // This is the wave-3 widening contract; regressing it produces silent
    // overflow for sums > i32::MAX.
    //
    // NOTE: This particular path uses `add.s64` for the per-block tree
    // reduction in shared memory, NOT `atom.global.add.s64`. The atomic-add
    // form only appears in the GROUP BY aggregate kernel
    // (`hash_kernels::compile_groupby_agg_kernel`), where the accumulator
    // table lives in global memory and one thread per row issues an atomic.
    // For SUM-without-GROUP-BY the per-block partials are combined on the
    // host. The widening contract — `cvt.s64.s32` then `add.s64` — is the
    // load-bearing assertion either way.
    let ptx = compile_reduction_kernel(ReduceOp::Sum, DataType::Int32).expect("compile");
    assert!(
        ptx.contains("cvt.s64.s32"),
        "SUM(Int32) must sign-extend to Int64 before accumulating\n{ptx}"
    );
    assert!(
        ptx.contains("add.s64"),
        "SUM(Int32) accumulator must combine at s64\n{ptx}"
    );
    // The widened input register declaration is the visible side-effect of
    // the `widens` branch in `compile_reduction_kernel`. Two .reg classes
    // (s32 input + s64 accumulator) must both be present.
    assert!(
        ptx.contains(".reg .s32") || ptx.contains(".reg .b32"),
        "expected an s32/b32 reg class for the narrow input load\n{ptx}"
    );
}

// ---- Tests: GROUP BY keys-kernel probe bound (wave-2 regression) -----------

#[test]
fn golden_groupby_keys_kernel_has_probe_bound() {
    // The keys kernel runs a linear-probe insertion loop; wave-2 added a
    // bounded-probe counter that gives up after `MAX_PROBE_FACTOR * k`
    // attempts so a misbehaving host (e.g. wrong load factor) can't pin a
    // streaming multiprocessor in an infinite loop. The bound surfaces as a
    // `setp.gt.u32` against the probe counter and a `@%pN bra DONE` early
    // exit inside the probe loop. Both must be present, and the give-up
    // bra must appear after `PROBE_LOOP:`.
    let ptx = compile_groupby_keys_kernel().expect("compile keys kernel");
    assert!(
        ptx.contains("PROBE_LOOP:"),
        "missing PROBE_LOOP label\n{ptx}"
    );
    assert!(
        ptx.contains("setp.gt.u32"),
        "missing setp.gt.u32 bound check (wave-2 probe limit)\n{ptx}"
    );
    let probe_start = ptx.find("PROBE_LOOP:").expect("PROBE_LOOP exists");
    let setp_pos = ptx[probe_start..]
        .find("setp.gt.u32")
        .expect("setp.gt.u32 should live inside the probe loop");
    // The probe bound now branches to the OVERFLOW block (which atomically
    // bumps the host-visible overflow counter before exiting), not directly
    // to DONE — overflow is surfaced to the host rather than silently dropped.
    let bra_pos = ptx[probe_start + setp_pos..]
        .find("bra OVERFLOW")
        .expect("expected @%pN bra OVERFLOW immediately after the bound check");
    // Sanity: the bra OVERFLOW must come within a few lines of the setp.
    assert!(
        bra_pos < 100,
        "bra OVERFLOW too far from setp.gt.u32 (probe bound check broken)\n{ptx}"
    );
    // And the kernel must actually emit the OVERFLOW block.
    assert!(
        ptx.contains("OVERFLOW:"),
        "missing OVERFLOW block (overflow must be host-visible, not silent)\n{ptx}"
    );
}

#[test]
fn golden_groupby_agg_kernel_has_probe_bound() {
    // The agg kernel runs a read-only linear-probe loop over an already-
    // populated keys table. Previously this loop had no bound, so a
    // partially-populated keys table (caused by a violated cross-stream
    // synchronisation contract — see the doc comment on
    // `compile_groupby_agg_kernel`) would spin forever and hang the SM.
    // The fix mirrors the keys kernel's `MAX_PROBE_FACTOR * k` cap:
    // increment a probe counter each iteration, `setp.gt.u32` against the
    // precomputed `max_probes` register, and branch to the `OVERFLOW` block
    // on over-probe. That block atomically bumps the host-visible overflow
    // counter (no aggregation atomic is issued for the over-probing row) and
    // then falls through to DONE — overflow is reported, not silently dropped.
    let ptx = compile_groupby_agg_kernel(ReduceOp::Sum, DataType::Int32)
        .expect("compile agg kernel");
    assert!(
        ptx.contains("PROBE_LOOP:"),
        "missing PROBE_LOOP label\n{ptx}"
    );
    assert!(
        ptx.contains("setp.gt.u32"),
        "missing setp.gt.u32 probe-bound check\n{ptx}"
    );
    // The bound check must live inside the probe loop and be followed
    // quickly by a `bra DONE` exit path — this is the give-up branch.
    let probe_start = ptx.find("PROBE_LOOP:").expect("PROBE_LOOP exists");
    let setp_pos = ptx[probe_start..]
        .find("setp.gt.u32")
        .expect("setp.gt.u32 should live inside the probe loop");
    let bra_pos = ptx[probe_start + setp_pos..]
        .find("bra OVERFLOW")
        .expect("expected @%pN bra OVERFLOW immediately after the probe bound");
    assert!(
        bra_pos < 100,
        "bra OVERFLOW too far from setp.gt.u32 (probe bound check broken)\n{ptx}"
    );
    // The give-up `bra OVERFLOW` must precede the `FOUND` label so a thread
    // that exceeds the bound exits (via the overflow counter) without issuing
    // the aggregation atomic update.
    let bra_overflow_abs = probe_start + setp_pos + bra_pos;
    let found_pos = ptx.find("FOUND:").expect("FOUND label exists");
    assert!(
        bra_overflow_abs < found_pos,
        "probe-bound `bra OVERFLOW` must precede the FOUND label (otherwise the \
         aggregation atomic still fires on over-probe)\n{ptx}"
    );
    // Overflow must be host-visible: the kernel emits an OVERFLOW block.
    assert!(
        ptx.contains("OVERFLOW:"),
        "missing OVERFLOW block (overflow must be host-visible, not silent)\n{ptx}"
    );
}

// ---- Tests: prefix-scan kernel block size -----------------------------------

#[test]
fn golden_prefix_scan_block_size_is_256() {
    // The prefix-scan kernel hard-codes BLOCK_SIZE = 256. The visible
    // marker is the shared-memory declaration: two ping-pong buffers of
    // BLOCK_SIZE * sizeof(u32) = 256 * 4 = 1024 bytes each, total 2048 B.
    // (No `.maxntid` directive is currently emitted — the block size is
    // enforced by the host launcher. The shared-memory size is the stable
    // proxy.)
    let ptx = compile_prefix_scan_kernel().expect("compile prefix scan");
    assert!(
        ptx.contains(".shared .align 4 .b8 sdata[2048]"),
        "expected sdata[2048] (BLOCK_SIZE=256, 2 ping-pong u32 buffers)\n{ptx}"
    );
}

/// Substring shape test for the Blelloch upsweep+downsweep variant.
///
/// Pins the load-bearing structural markers:
///   * single shared-memory buffer (no ping-pong) — Blelloch operates
///     in-place across barriers.
///   * matching entry-name and 4-arg ABI so host code can swap kernels
///     transparently.
///   * BOTH algorithm-section markers present (`UPSWEEP` + `DOWNSWEEP`)
///     plus the exclusive-scan `ZERO-INIT` pivot.
///   * exact bar.sync count: 2*log2(BLOCK_SIZE) + 2 = 18 for BLOCK_SIZE=256,
///     i.e. seed + K upsweep + pivot + K downsweep. If a future refactor
///     drops a barrier this asserts the regression up-front.
#[test]
fn golden_prefix_scan_blelloch_has_shape() {
    let ptx = compile_prefix_scan_kernel_blelloch().expect("compile blelloch");

    // Single buffer = BLOCK_SIZE * 4 = 1024 bytes (vs. 2048 for Hillis-Steele).
    assert!(
        ptx.contains(".shared .align 4 .b8 sdata[1024]"),
        "expected sdata[1024] (BLOCK_SIZE=256, single u32 buffer)\n{ptx}"
    );

    // Entry + ABI.
    assert!(
        ptx.contains(".visible .entry bolt_prefix_scan_blelloch("),
        "missing Blelloch entry name\n{ptx}"
    );
    assert!(ptx.contains(".param .u64 bolt_prefix_scan_blelloch_param_0,"));
    assert!(ptx.contains(".param .u64 bolt_prefix_scan_blelloch_param_1,"));
    assert!(ptx.contains(".param .u64 bolt_prefix_scan_blelloch_param_2,"));
    assert!(ptx.contains(".param .u32 bolt_prefix_scan_blelloch_param_3"));

    // Phase markers: dropping any of these means the algorithm is missing
    // a phase or has been silently renamed.
    assert!(
        ptx.contains("BLELLOCH UPSWEEP"),
        "missing upsweep marker:\n{ptx}"
    );
    assert!(
        ptx.contains("BLELLOCH DOWNSWEEP"),
        "missing downsweep marker:\n{ptx}"
    );
    assert!(
        ptx.contains("BLELLOCH ZERO-INIT"),
        "missing exclusive-scan zero-init pivot:\n{ptx}"
    );

    // bar.sync count: seed + K upsweep levels + pivot + K downsweep levels.
    // For BLOCK_SIZE = 256, K = log2(256) = 8, total = 2 * 8 + 2 = 18.
    let n_sync = ptx.matches("bar.sync 0;").count();
    assert_eq!(
        n_sync, 18,
        "expected exactly 18 bar.syncs (seed + 8 upsweep + pivot + 8 downsweep), got {n_sync}\n{ptx}"
    );
}

/// Golden lock for the decoupled-lookback variant. Pins the load-bearing
/// substrings of the publication protocol — drop any of these and the
/// cross-CTA scan loses correctness on sm_70+:
///
///   * `PUBLISH_AGGREGATE` / `LOOKBACK_SPIN` / `PUBLISH_INCLUSIVE` /
///     `BROADCAST` labels — the four phases of the algorithm.
///   * `membar.gl;` (>= 2) — the post-publish fences that make a peer's
///     `ld.acquire.gpu.u32` of the same address observe the new status.
///   * `ld.acquire.gpu.u32` — acquire-scope read of `partial_status[pred]`
///     during the spin loop.
///   * 5-arg ABI (4 u64 + 1 u32) with the lookback entry name.
#[test]
fn golden_prefix_scan_lookback_has_shape() {
    let ptx = compile_prefix_scan_kernel_lookback().expect("compile lookback");

    // Entry + 5-param ABI.
    assert!(
        ptx.contains(".visible .entry bolt_prefix_scan_lookback("),
        "missing lookback entry name:\n{ptx}"
    );
    assert!(ptx.contains(".param .u64 bolt_prefix_scan_lookback_param_0,"));
    assert!(ptx.contains(".param .u64 bolt_prefix_scan_lookback_param_1,"));
    assert!(ptx.contains(".param .u64 bolt_prefix_scan_lookback_param_2,"));
    assert!(ptx.contains(".param .u32 bolt_prefix_scan_lookback_param_3,"));
    assert!(ptx.contains(".param .u64 bolt_prefix_scan_lookback_param_4"));

    // Phase markers.
    for label in [
        "PUBLISH_AGGREGATE:",
        "LOOKBACK_SPIN:",
        "PUBLISH_INCLUSIVE:",
        "BROADCAST:",
    ] {
        assert!(
            ptx.contains(label),
            "missing label `{label}` in lookback PTX:\n{ptx}"
        );
    }

    // Memory-order primitives. Two membar.gl fences (one per publish) +
    // an acquire-scope load on the spin path are the load-bearing
    // synchronization for the decoupled-lookback protocol on sm_70+.
    let n_membar = ptx.matches("membar.gl;").count();
    assert!(
        n_membar >= 2,
        "expected >=2 membar.gl (one per publish), got {n_membar}:\n{ptx}"
    );
    assert!(
        ptx.contains("ld.acquire.gpu.u32"),
        "missing ld.acquire.gpu.u32 on partial_status spin read:\n{ptx}"
    );
}

// ---- Tests: partition-reduce kernel fences the set/key publish race --------

#[test]
fn golden_partition_reduce_fences_set_key_race() {
    // CRITICAL CORRECTNESS REGRESSION GUARD.
    //
    // The per-partition reduce kernel claims a slot with
    // `atom.shared.cas.b32` on `block_set[slot]`, then publishes the key
    // with `st.shared.u32` on `block_keys[slot]`. Those two operations
    // touch DIFFERENT shared addresses, so PTX on sm_70 gives no
    // inter-address ordering between them. Without an explicit
    // `membar.cta`:
    //
    //   * On the CLAIM path the key store can be reordered after the
    //     val atomic — racing readers see set==1 with a stale key.
    //   * On the MATCH path the key load can be hoisted before the
    //     racing winner's key store becomes visible — racing readers
    //     see a still-zeroed key and false-match key 0.
    //
    // Both lead to silent wrong-sum output when any user key happens to
    // be 0. The fix emits `membar.cta`:
    //   * Between the key store and the val atomic on CLAIM.
    //   * Between the set-CAS and the key load on MATCH.
    //
    // This test pins the contract: regressing it reopens the
    // correctness bug.
    //
    // NOTE — the VALIDATED fix (commits be85833 / e3db739, compute-sanitizer
    // clean) is a 3-state `set` flag (0=empty, 1=claiming, 2=published):
    //   * WINNER (CLAIM): `st key; membar.cta; st set:=2` — one release fence
    //     so any reader observing set==2 also sees the published key.
    //   * LOSER: spin in `PUBLISH_WAIT` re-reading `set` via
    //     `ld.volatile.shared.u32` until it equals 2, THEN read the key.
    // An earlier design used a SECOND membar + `ld.acquire.cta` on the MATCH
    // path, but `ld.acquire` defaults to GLOBAL space and faulted on the shared
    // address ("invalid __global__ read"); the volatile-spin protocol replaced
    // it. So the kernel now emits exactly ONE `membar.cta` (the CLAIM release),
    // and the loser's acquire is the volatile spin — this test guards that.
    use craton_bolt::jit::partition_reduce_kernel::compile_partition_reduce_kernel;
    let ptx = compile_partition_reduce_kernel().expect("kernel compiles");

    // (1) CLAIM release fence: key store -> membar.cta -> set:=2 -> val atomic.
    let claim_label = ptx.find("CLAIM:").expect("missing CLAIM: label");
    let claim_tail = &ptx[claim_label..];
    let key_store = claim_tail
        .find("st.shared.u32")
        .expect("missing CLAIM-path key store");
    let after_store = &claim_tail[key_store..];
    let membar = after_store
        .find("membar.cta")
        .expect("missing CLAIM-path membar.cta after the key store");
    let set_publish = after_store
        .find("], 2;")
        .expect("missing CLAIM-path set:=2 publish store");
    let val_atomic = after_store
        .find("atom.shared.add.f64")
        .expect("missing CLAIM-path val atomic");
    assert!(
        membar < set_publish,
        "CLAIM: membar.cta must precede the set:=2 publish (release order):\n{ptx}"
    );
    assert!(
        membar < val_atomic,
        "CLAIM: membar.cta must precede the f64 val atomic:\n{ptx}"
    );

    // (2) LOSER acquire: the PUBLISH_WAIT spin re-reads the set flag via
    //     ld.volatile.shared.u32 until it observes the published value 2 — this
    //     replaces the old MATCH-path membar/ld.acquire and is the load-bearing
    //     half of the publish protocol.
    let wait = ptx
        .find("PUBLISH_WAIT:")
        .expect("missing PUBLISH_WAIT spin (loser acquire path)");
    let wait_tail = &ptx[wait..];
    let vol = wait_tail
        .find("ld.volatile.shared.u32")
        .expect("PUBLISH_WAIT must re-read the set flag via ld.volatile.shared.u32");
    assert!(
        wait_tail[vol..].contains(", 2;"),
        "PUBLISH_WAIT must spin until the set flag equals the published value 2:\n{ptx}"
    );

    // And the kernel must still issue the slot-claim CAS on the set flag.
    assert!(
        ptx.contains("atom.shared.cas.b32"),
        "partition-reduce kernel must issue atom.shared.cas.b32 to claim a slot:\n{ptx}"
    );
}

// ---- Tests: float MIN uses CAS loop ----------------------------------------

#[test]
fn golden_float_atomic_min_uses_cas_loop() {
    // PTX has no native `atom.global.min.f64` on sm_70. The float-atomic
    // path implements MIN via a CAS loop: load current accumulator, take
    // the float-typed min, then `atom.global.cas.b64` until success. The
    // CAS instruction is the load-bearing marker — pair it with a
    // setp.lt.f64 to confirm the comparison is the float one (not an
    // unrelated atomic CAS borrowed from the keys-table probe).
    let ptx = compile_groupby_float_atomic_kernel(ReduceOp::Min, DataType::Float64)
        .expect("compile float atomic kernel");
    assert!(
        ptx.contains("atom.global.cas.b64"),
        "MIN(f64) must use the b64 CAS loop\n{ptx}"
    );
    assert!(
        ptx.contains("setp.lt.f64"),
        "MIN(f64) must use float < comparison\n{ptx}"
    );
    // ORDERING (not just presence): the float comparison that decides whether
    // the candidate improves the accumulator must be emitted BEFORE the CAS
    // that publishes it, and the `@%p4 bra DONE` skip-on-no-improvement gate
    // must also precede the CAS. Otherwise a thread would issue the atomic CAS
    // unconditionally (or before computing the new value), reopening the
    // hot-cacheline storm the gate exists to avoid and risking a stale store.
    // Presence checks pass either way, so pin the order.
    assert_emitted_before(&ptx, "setp.lt.f64", "atom.global.cas.b64");
    assert_emitted_before(&ptx, "@%p4 bra DONE;", "atom.global.cas.b64");
}

// NOTE: Option B pre-stage validity propagation golden tests live in
// `src/jit/ptx_gen.rs` (the unit-test module) because they need to
// construct a `physical_plan::Reg` directly and `Reg`'s tuple field is
// `pub(crate)`. Keeping the codegen smoke tests next to the codegen
// itself also matches the existing structure of `scan_kernel.rs::tests`.

// ---- Tests: partition_reduce_kernel variants (review L4) -------------------
//
// One canonical substring-shape test per emitter variant. Each pins the
// kernel's entry-point name plus a load-bearing mnemonic that's specific to
// that variant (e.g. atom.shared.add.u64 for COUNT, the CAS suffix for the
// float CAS-loop variants, ld.global.s64 for i64-key variants). Keeps the
// existence of each emitter in CI without going as deep as the wave-3
// correctness goldens above.

#[test]
fn golden_partition_reduce_kernel_count_smoke() {
    use craton_bolt::jit::partition_reduce_kernel_count::compile_partition_reduce_kernel_count;
    let ptx = compile_partition_reduce_kernel_count().expect("compile");
    assert!(ptx.contains(".visible .entry bolt_partition_reduce_count"), "{ptx}");
    // COUNT increments a per-slot u64 counter via shared-memory atomic add.
    assert!(ptx.contains("atom.shared.add.u64"), "missing u64 atomic add\n{ptx}");
    // Slot-ownership still goes through the b32 CAS like the base kernel.
    assert!(ptx.contains("atom.shared.cas.b32"), "{ptx}");
}

#[test]
fn golden_partition_reduce_kernel_count_i64_smoke() {
    use craton_bolt::jit::partition_reduce_kernel_count_i64::compile_partition_reduce_kernel_count_i64;
    let ptx = compile_partition_reduce_kernel_count_i64().expect("compile");
    assert!(ptx.contains(".visible .entry bolt_partition_reduce_count_i64"), "{ptx}");
    // i64-key variant must load the row key with s64.
    assert!(ptx.contains("ld.global.s64"), "missing s64 key load\n{ptx}");
    assert!(ptx.contains("atom.shared.add.u64"), "{ptx}");
}

#[test]
fn golden_partition_reduce_kernel_i64_smoke() {
    use craton_bolt::jit::partition_reduce_kernel_i64::compile_partition_reduce_kernel_i64;
    let ptx = compile_partition_reduce_kernel_i64().expect("compile");
    assert!(ptx.contains(".visible .entry bolt_partition_reduce_i64"), "{ptx}");
    // f64 SUM combines via the shared-memory float atomic add.
    assert!(ptx.contains("atom.shared.add.f64"), "missing f64 atomic\n{ptx}");
    // i64-key variant identifies the i64 key load.
    assert!(ptx.contains("ld.global.s64"), "{ptx}");
}

#[test]
fn golden_partition_reduce_kernel_minmax_i32_min_smoke() {
    use craton_bolt::jit::partition_reduce_kernel_minmax::{
        compile_partition_reduce_kernel_minmax, MinMaxDtype, MinMaxOp,
    };
    let ptx =
        compile_partition_reduce_kernel_minmax(MinMaxOp::Min, MinMaxDtype::Int32).expect("compile");
    assert!(ptx.contains(".visible .entry bolt_partition_reduce_min_i32"), "{ptx}");
    // The integer MIN path uses a native shared-memory min atomic.
    assert!(ptx.contains("atom.shared.min.s32"), "missing min.s32 atomic\n{ptx}");
}

#[test]
fn golden_partition_reduce_kernel_minmax_i64val_max_smoke() {
    use craton_bolt::jit::partition_reduce_kernel_minmax::{
        compile_partition_reduce_kernel_minmax, MinMaxDtype, MinMaxOp,
    };
    let ptx =
        compile_partition_reduce_kernel_minmax(MinMaxOp::Max, MinMaxDtype::Int64).expect("compile");
    assert!(ptx.contains(".visible .entry bolt_partition_reduce_max_i64"), "{ptx}");
    assert!(ptx.contains("atom.shared.max.s64"), "missing max.s64 atomic\n{ptx}");
}

#[test]
fn golden_partition_reduce_kernel_minmax_i64_key_min_smoke() {
    use craton_bolt::jit::partition_reduce_kernel_minmax::{MinMaxDtype, MinMaxOp};
    use craton_bolt::jit::partition_reduce_kernel_minmax_i64::compile_partition_reduce_kernel_minmax_i64;
    let ptx = compile_partition_reduce_kernel_minmax_i64(MinMaxOp::Min, MinMaxDtype::Int32)
        .expect("compile");
    // Entry name encodes the i64-key suffix.
    assert!(ptx.contains(".visible .entry bolt_partition_reduce_min_i32_keyi64"), "{ptx}");
    assert!(ptx.contains("atom.shared.min.s32"), "{ptx}");
    // The i64 key must be loaded with an s64 instruction.
    assert!(ptx.contains("ld.global.s64"), "{ptx}");
}

#[test]
fn golden_partition_reduce_kernel_minmax_float_f64_min_smoke() {
    use craton_bolt::jit::partition_reduce_kernel_minmax::MinMaxOp;
    use craton_bolt::jit::partition_reduce_kernel_minmax_float::{
        compile_partition_reduce_kernel_minmax_float, FloatDtype,
    };
    let ptx = compile_partition_reduce_kernel_minmax_float(MinMaxOp::Min, FloatDtype::Float64)
        .expect("compile");
    assert!(ptx.contains(".visible .entry bolt_partition_reduce_min_f64"), "{ptx}");
    // PTX has no atom.shared.min.f64 on sm_70 — CAS loop with float compare.
    assert!(ptx.contains("atom.shared.cas.b64"), "missing b64 CAS loop\n{ptx}");
    assert!(ptx.contains("setp.lt.f64"), "missing float < compare\n{ptx}");
}

#[test]
fn golden_partition_reduce_kernel_minmax_float_f32_max_smoke() {
    use craton_bolt::jit::partition_reduce_kernel_minmax::MinMaxOp;
    use craton_bolt::jit::partition_reduce_kernel_minmax_float::{
        compile_partition_reduce_kernel_minmax_float, FloatDtype,
    };
    let ptx = compile_partition_reduce_kernel_minmax_float(MinMaxOp::Max, FloatDtype::Float32)
        .expect("compile");
    assert!(ptx.contains(".visible .entry bolt_partition_reduce_max_f32"), "{ptx}");
    assert!(ptx.contains("atom.shared.cas.b32"), "missing b32 CAS loop\n{ptx}");
    assert!(ptx.contains("setp.gt.f32"), "missing float > compare\n{ptx}");
}

#[test]
fn golden_partition_reduce_kernel_minmax_float_i64_key_smoke() {
    use craton_bolt::jit::partition_reduce_kernel_minmax::MinMaxOp;
    use craton_bolt::jit::partition_reduce_kernel_minmax_float::FloatDtype;
    use craton_bolt::jit::partition_reduce_kernel_minmax_float_i64::compile_partition_reduce_kernel_minmax_float_i64;
    let ptx =
        compile_partition_reduce_kernel_minmax_float_i64(MinMaxOp::Min, FloatDtype::Float64)
            .expect("compile");
    assert!(ptx.contains(".visible .entry bolt_partition_reduce_min_f64_keyi64"), "{ptx}");
    assert!(ptx.contains("atom.shared.cas.b64"), "{ptx}");
    // i64-key signature: s64 row-key load.
    assert!(ptx.contains("ld.global.s64"), "{ptx}");
}

#[test]
fn golden_partition_reduce_kernel_multi_smoke() {
    use craton_bolt::jit::partition_reduce_kernel_multi::compile_partition_reduce_kernel_multi;
    // n_vals = 2 — exercises the multi-sum entry-name formatting and the
    // per-val atomic-add emission inside the slot-claim and slot-match paths.
    let ptx = compile_partition_reduce_kernel_multi(2).expect("compile");
    assert!(ptx.contains(".visible .entry bolt_partition_reduce_multi_sum_2"), "{ptx}");
    // CLAIM and MATCH each issue n_vals f64 atomic adds -> at least 2 total.
    let n = ptx.matches("atom.shared.add.f64").count();
    assert!(n >= 2, "expected >=2 atom.shared.add.f64 for n_vals=2, got {n}\n{ptx}");
}

#[test]
fn golden_partition_reduce_kernel_multi_i64_smoke() {
    use craton_bolt::jit::partition_reduce_kernel_multi_i64::compile_partition_reduce_kernel_multi_i64;
    let ptx = compile_partition_reduce_kernel_multi_i64(2).expect("compile");
    assert!(ptx.contains(".visible .entry bolt_partition_reduce_multi_sum_i64_2"), "{ptx}");
    assert!(ptx.contains("atom.shared.add.f64"), "{ptx}");
    assert!(ptx.contains("ld.global.s64"), "i64-key load missing\n{ptx}");
}

// ---- Tests: hash_join_kernel variants (review L4) --------------------------

#[test]
fn golden_hash_join_build_kernel_smoke() {
    use craton_bolt::jit::hash_join_kernel::compile_build_kernel;
    let ptx = compile_build_kernel().expect("compile");
    assert!(ptx.contains(".visible .entry bolt_hash_join_build"), "{ptx}");
    // Build claims a slot via global b64 CAS on the keys table.
    assert!(ptx.contains("atom.global.cas.b64"), "missing build-side CAS\n{ptx}");
    assert!(ptx.contains("PROBE_LOOP:"), "{ptx}");
}

/// Helper: assert `needle` appears strictly before the first occurrence of
/// `marker` in `haystack`. Used by the speculative-load goldens to lock the
/// pre-check → CAS ordering.
fn assert_appears_before_with_ctx(haystack: &str, needle: &str, marker: &str, ctx: &str) {
    let needle_pos = haystack.find(needle).unwrap_or_else(|| {
        panic!("{ctx}: expected '{needle}' to appear in PTX:\n{haystack}")
    });
    let marker_pos = haystack.find(marker).unwrap_or_else(|| {
        panic!("{ctx}: expected marker '{marker}' to appear in PTX:\n{haystack}")
    });
    assert!(
        needle_pos < marker_pos,
        "{ctx}: '{needle}' must appear before '{marker}' \
         (needle@{needle_pos}, marker@{marker_pos})\n{haystack}"
    );
}

/// Batch 6: SoA build kernel emits a speculative `ld.acquire.gpu.s64`
/// pre-check immediately before its slot-claim CAS. Skipping the CAS when
/// the slot is already occupied by a non-matching key removes a
/// guaranteed-miss atomic under hot-key skew.
#[test]
fn golden_hash_join_build_kernel_speculative_pre_check() {
    use craton_bolt::jit::hash_join_kernel::compile_build_kernel;
    let ptx = compile_build_kernel().expect("compile");
    assert!(
        ptx.contains("ld.acquire.gpu.s64"),
        "SoA build must emit speculative ld.acquire.gpu.s64 before CAS\n{ptx}"
    );
    assert!(ptx.contains("DO_CAS:"), "SoA build must emit DO_CAS: label\n{ptx}");
    assert_appears_before_with_ctx(
        &ptx,
        "ld.acquire.gpu.s64",
        "atom.global.cas.b64",
        "compile_build_kernel",
    );
    assert_appears_before_with_ctx(
        &ptx,
        "DO_CAS:",
        "atom.global.cas.b64",
        "compile_build_kernel",
    );
}

/// Batch 6: collision-list build kernel emits the speculative pre-check
/// before its slot-claim CAS. The chain-prepend `atom.global.exch.b32` on
/// the head pointer is a different atomic and is NOT preceded by the
/// pre-check.
#[test]
fn golden_hash_join_build_collision_kernel_speculative_pre_check() {
    use craton_bolt::jit::hash_join_kernel::compile_build_collision_kernel;
    let ptx = compile_build_collision_kernel().expect("compile");
    assert!(
        ptx.contains("ld.acquire.gpu.s64"),
        "collision build must emit speculative ld.acquire.gpu.s64 before CAS\n{ptx}"
    );
    assert!(ptx.contains("DO_CAS:"), "collision build must emit DO_CAS: label\n{ptx}");
    assert_appears_before_with_ctx(
        &ptx,
        "ld.acquire.gpu.s64",
        "atom.global.cas.b64",
        "compile_build_collision_kernel",
    );
    // Chain-head atomic stays untouched.
    assert!(ptx.contains("atom.global.exch.b32"), "{ptx}");
    // The speculative pre-check only guards the slot-claim CAS, not the
    // head-pointer exch: there must be exactly one `ld.acquire.gpu.s64`.
    let n_spec = ptx.matches("ld.acquire.gpu.s64").count();
    assert_eq!(
        n_spec, 1,
        "collision build must emit exactly one speculative pre-check (slot CAS only); saw {n_spec}\n{ptx}"
    );
}

/// Batch 6: AoS build kernel emits the speculative pre-check before its
/// slot-claim CAS. AoS slot layout (`[key:u64, head:u32, _pad:u32]`)
/// doesn't change the analysis — the i64 key word at slot offset 0 is the
/// CAS target.
#[test]
fn golden_hash_join_build_aos_kernel_speculative_pre_check() {
    use craton_bolt::jit::hash_join_kernel::compile_build_aos_kernel;
    let ptx = compile_build_aos_kernel().expect("compile");
    assert!(
        ptx.contains("ld.acquire.gpu.s64"),
        "AoS build must emit speculative ld.acquire.gpu.s64 before CAS\n{ptx}"
    );
    assert!(ptx.contains("DO_CAS:"), "AoS build must emit DO_CAS: label\n{ptx}");
    assert_appears_before_with_ctx(
        &ptx,
        "ld.acquire.gpu.s64",
        "atom.global.cas.b64",
        "compile_build_aos_kernel",
    );
}

#[test]
fn golden_hash_join_probe_kernel_smoke() {
    use craton_bolt::jit::hash_join_kernel::compile_probe_kernel;
    let ptx = compile_probe_kernel().expect("compile");
    assert!(ptx.contains(".visible .entry bolt_hash_join_probe"), "{ptx}");
    // Probe is non-mutating; it only does atomic-add on the output counter.
    assert!(ptx.contains("atom.global.add.u32"), "missing output-claim atomic\n{ptx}");
    assert!(ptx.contains("PROBE_LOOP:"), "{ptx}");
    assert!(ptx.contains("MATCH:"), "missing MATCH label\n{ptx}");
}

// ---- Batch 6: tile-aware 2-way unrolled SoA probe goldens -------------------
//
// The tiled probe is a drop-in for compile_probe_kernel with the same nine-
// parameter ABI but reads two adjacent slots per iteration via one
// ld.global.nc.v2.u64. These goldens pin the three load-bearing behaviours
// that distinguish it from a naive 2-way unroll: the fused 16-byte load
// (the entire point), the SCALAR_STEP wrap-edge handler (obstacle 1), and
// the empty-check-before-second-lane ordering (obstacle 3).

#[test]
fn golden_hash_join_probe_tiled_kernel_smoke() {
    use craton_bolt::jit::hash_join_kernel::compile_probe_kernel_tiled;
    let ptx = compile_probe_kernel_tiled().expect("compile");
    // Entry-point shape.
    assert!(
        ptx.contains(".visible .entry bolt_hash_join_probe_tiled"),
        "{ptx}"
    );
    // Same output-claim atomic as the single-load probe.
    assert!(
        ptx.contains("atom.global.add.u32"),
        "missing output-claim atomic\n{ptx}"
    );
    // The tile loop has TILE_TOP and the two match labels split out so
    // MATCH_S uses slot %r8 and MATCH_SP1 uses slot %r8 + 1.
    assert!(ptx.contains("TILE_TOP:"), "{ptx}");
    assert!(ptx.contains("MATCH_S:"), "{ptx}");
    assert!(ptx.contains("MATCH_SP1:"), "{ptx}");
}

/// The tiled probe MUST emit the fused 16-byte `ld.global.nc.v2.u64` —
/// that's the entire reason the kernel exists. Dropping it (e.g. an
/// accidental regression to two scalar loads inside the tile) defeats
/// the bandwidth win.
#[test]
fn golden_hash_join_probe_tiled_emits_v2_fused_load() {
    use craton_bolt::jit::hash_join_kernel::compile_probe_kernel_tiled;
    let ptx = compile_probe_kernel_tiled().expect("compile");
    assert!(
        ptx.contains("ld.global.nc.v2.u64"),
        "tiled probe must emit ld.global.nc.v2.u64 for the fused 16-byte load\n{ptx}"
    );
}

/// The tiled probe MUST contain a `SCALAR_STEP:` label that handles the
/// wraparound edge (obstacle 1). Without it, the v2 load at slot == cap-1
/// reads past the end of keys_table.
#[test]
fn golden_hash_join_probe_tiled_has_scalar_step_for_wraparound() {
    use craton_bolt::jit::hash_join_kernel::compile_probe_kernel_tiled;
    let ptx = compile_probe_kernel_tiled().expect("compile");
    assert!(
        ptx.contains("SCALAR_STEP:"),
        "tiled probe must have a SCALAR_STEP label for wrap-edge\n{ptx}"
    );
    assert!(
        ptx.contains("bra SCALAR_STEP"),
        "tiled probe must conditionally branch to SCALAR_STEP on wrap\n{ptx}"
    );
}

/// Obstacle 3 — the empty-slot check on `slot[s]` must precede the match
/// check on `slot[s + 1]`. Otherwise a stale chain-tail key in slot[s + 1]
/// can false-match against a probe key whose chain actually ended at
/// slot[s].
#[test]
fn golden_hash_join_probe_tiled_empty_check_precedes_second_lane_match() {
    use craton_bolt::jit::hash_join_kernel::compile_probe_kernel_tiled;
    let ptx = compile_probe_kernel_tiled().expect("compile");
    // slot[s] vs EMPTY (in %rl5 vs %rl4) must appear before slot[s+1] vs
    // probe-key (in %rl6 vs %rl0).
    let empty_slot_s = "setp.eq.s64 %p12, %rl5, %rl4;";
    let match_slot_sp1 = "setp.eq.s64 %p13, %rl6, %rl0;";
    assert_appears_before(&ptx, empty_slot_s, match_slot_sp1);
}

#[test]
fn golden_hash_join_build_collision_kernel_smoke() {
    use craton_bolt::jit::hash_join_kernel::compile_build_collision_kernel;
    let ptx = compile_build_collision_kernel().expect("compile");
    assert!(ptx.contains(".visible .entry bolt_hash_join_build_collision"), "{ptx}");
    // The collision-list build atomically swaps the head pointer when
    // prepending a new chain entry.
    assert!(ptx.contains("atom.global.exch.b32"), "missing chain exch\n{ptx}");
}

#[test]
fn golden_hash_join_probe_collision_kernel_smoke() {
    use craton_bolt::jit::hash_join_kernel::compile_probe_collision_kernel;
    let ptx = compile_probe_collision_kernel().expect("compile");
    assert!(ptx.contains(".visible .entry bolt_hash_join_probe_collision"), "{ptx}");
    // Collision probe walks the chain via WALK_CHAIN + CHAIN_LOOP labels.
    assert!(ptx.contains("WALK_CHAIN:"), "{ptx}");
    assert!(ptx.contains("CHAIN_LOOP:"), "{ptx}");
}

#[test]
fn golden_hash_join_unmatched_build_kernel_smoke() {
    use craton_bolt::jit::hash_join_kernel::compile_unmatched_build_kernel;
    let ptx = compile_unmatched_build_kernel().expect("compile");
    assert!(
        ptx.contains(".visible .entry bolt_hash_join_emit_unmatched_build"),
        "{ptx}"
    );
    // The outer-join unmatched scanner reads the matched-bitmap word and
    // claims output slots via an atomic add on a u32 counter.
    assert!(ptx.contains("atom.global.add.u32"), "{ptx}");
}

// ---- Tests: speculative ld.acquire pre-check before output-counter atom.add
//
// All probe (and unmatched-build) kernels that claim output slots via
// `atom.global.add.u32` on a shared counter must first emit a speculative
// `ld.acquire.gpu.u32` of the counter and a `setp.ge.u32` against the
// out_capacity register, branching to the bail label. Without this, under
// capacity overflow EVERY matching thread issues an atomic increment on the
// hot counter cacheline, serializing all warps even when no writes will
// happen. The pre-check is purely additive — the atom.add's post-increment
// bounds check still guards correctness, so a thread that races past the
// pre-check is still safe.
//
// Each test asserts the literal speculative-load shape and verifies it
// appears textually BEFORE the atom.global.add.u32 site in the emitted PTX.

/// Helper: assert that `needle_pre` appears at a lower byte offset than
/// `needle_post` in `ptx`. Reports both offsets on failure.
fn assert_appears_before(ptx: &str, needle_pre: &str, needle_post: &str) {
    let pre = ptx
        .find(needle_pre)
        .unwrap_or_else(|| panic!("missing pre-check `{needle_pre}`\n{ptx}"));
    let post = ptx
        .find(needle_post)
        .unwrap_or_else(|| panic!("missing post site `{needle_post}`\n{ptx}"));
    assert!(
        pre < post,
        "pre-check `{needle_pre}` must appear before `{needle_post}`; \
         pre@{pre} >= post@{post}\n{ptx}"
    );
}

/// Assert that a validity / predicate *gate* is emitted strictly before the
/// store / atomic it protects.
///
/// This is the load-bearing counterpart to the many substring-PRESENCE checks
/// in this file. A test that asserts `ptx.contains("@%pN bra SKIP")` AND
/// `ptx.contains("st.global…")` passes even if codegen emits the store BEFORE
/// the gate — i.e. the write happens unconditionally and the "skip" branch is
/// dead. Such a regression is silently wrong (a NULL row gets written, an
/// unimproved slot gets an atomic) yet invisible to a presence check. By
/// pinning the byte order of the two markers we turn an ordering bug into a
/// red test.
///
/// `earlier` is the gate marker (the `setp`/`@%p..` guard or the `bra` skip),
/// `later` is the guarded store/atomic. Both must be unique enough that
/// `str::find` (first occurrence) lands on the intended site; the callers
/// below pick markers — exact register-bearing lines or distinctive labels —
/// that satisfy this. On failure we dump the full PTX plus both byte offsets
/// so the contract violation is actionable from the test log alone.
fn assert_emitted_before(ptx: &str, earlier: &str, later: &str) {
    let gate = ptx.find(earlier).unwrap_or_else(|| {
        panic!("ordering check: gate marker `{earlier}` not found in PTX\n{ptx}")
    });
    let store = ptx.find(later).unwrap_or_else(|| {
        panic!("ordering check: guarded marker `{later}` not found in PTX\n{ptx}")
    });
    assert!(
        gate < store,
        "ordering violation: the gate `{earlier}` (byte {gate}) must be emitted \
         BEFORE the store/atomic `{later}` (byte {store}) it protects, but it is \
         emitted after. A substring-presence check would not catch this — the \
         store now fires before (or instead of being skipped by) its validity \
         gate, which is a silent-wrong-output regression.\n{ptx}"
    );
}

#[test]
fn probe_soa_ptx_speculative_load_before_atom_add() {
    use craton_bolt::jit::hash_join_kernel::compile_probe_kernel;
    let ptx = compile_probe_kernel().expect("compile");
    // The speculative load shape: ld.acquire.gpu.u32 + setp.ge.u32.
    assert!(
        ptx.contains("ld.acquire.gpu.u32"),
        "SoA probe must emit speculative ld.acquire.gpu.u32 before atom.add\n{ptx}"
    );
    assert_appears_before(&ptx, "ld.acquire.gpu.u32", "atom.global.add.u32");
    // The pre-check's OWN branch must bail to DONE: a `bra DONE` must appear
    // between the speculative load and the atom.add it guards. (Asserting the
    // load merely precedes *some* `bra DONE` would wrongly match the earlier
    // `tid >= n_probe` thread-bounds early-exit, which always sits before the
    // probe loop — making the check vacuous.)
    let load = ptx
        .find("ld.acquire.gpu.u32")
        .expect("missing speculative load");
    let atom = ptx
        .find("atom.global.add.u32")
        .expect("missing atom.global.add.u32");
    assert!(
        ptx[load..atom].contains("bra DONE"),
        "pre-check must branch to DONE between the speculative load and the \
         atom.global.add.u32 it guards\n{ptx}"
    );
}

#[test]
fn probe_collision_ptx_speculative_load_before_atom_add() {
    use craton_bolt::jit::hash_join_kernel::compile_probe_collision_kernel;
    let ptx = compile_probe_collision_kernel().expect("compile");
    assert!(
        ptx.contains("ld.acquire.gpu.u32"),
        "collision probe must emit speculative ld.acquire.gpu.u32 before atom.add\n{ptx}"
    );
    assert_appears_before(&ptx, "ld.acquire.gpu.u32", "atom.global.add.u32");
}

#[test]
fn probe_aos_ptx_speculative_load_before_atom_add() {
    use craton_bolt::jit::hash_join_kernel::compile_probe_aos_kernel;
    let ptx = compile_probe_aos_kernel().expect("compile");
    assert!(
        ptx.contains("ld.acquire.gpu.u32"),
        "AoS probe must emit speculative ld.acquire.gpu.u32 before atom.add\n{ptx}"
    );
    assert_appears_before(&ptx, "ld.acquire.gpu.u32", "atom.global.add.u32");
}

#[test]
fn unmatched_build_ptx_speculative_load_before_atom_add() {
    use craton_bolt::jit::hash_join_kernel::compile_unmatched_build_kernel;
    let ptx = compile_unmatched_build_kernel().expect("compile");
    assert!(
        ptx.contains("ld.acquire.gpu.u32"),
        "unmatched-build kernel must emit speculative ld.acquire.gpu.u32 before atom.add\n{ptx}"
    );
    assert_appears_before(&ptx, "ld.acquire.gpu.u32", "atom.global.add.u32");
}

// ---- Tests: sort_kernel layouts (review L4) --------------------------------

#[test]
fn golden_sort_kernel_multilaunch_smoke() {
    use craton_bolt::jit::sort_kernel::{
        compile_sort_kernel_spec, KeyDesc, SortDirection, SortKernelSpec, SortLayout,
    };
    // Two-key spec — exercises the multi-key comparator emission. Both keys
    // are non-nullable so the validity-load fast-path is skipped.
    let spec = SortKernelSpec {
        keys: vec![
            KeyDesc {
                dtype: DataType::Int32,
                direction: SortDirection::Asc,
                nullable: false,
                nulls_first: false,
            },
            KeyDesc {
                dtype: DataType::Float64,
                direction: SortDirection::Desc,
                nullable: false,
                nulls_first: false,
            },
        ],
        layout: SortLayout::MultiLaunch,
        shmem_n_pow2: 0,
    };
    let ptx = compile_sort_kernel_spec(&spec).expect("compile");
    // MultiLaunch entry prefix.
    assert!(ptx.contains(".visible .entry bolt_bitonic_sort_ml"), "{ptx}");
    // Bitonic partner index from XOR over tid + substage mask.
    assert!(ptx.contains("xor.b32"), "{ptx}");
    // Per-dtype compares for both keys.
    assert!(ptx.contains("setp.gt.s32"), "{ptx}");
    assert!(ptx.contains("setp.gt.f64"), "{ptx}");
}

#[test]
fn golden_sort_kernel_shmem_smoke() {
    use craton_bolt::jit::sort_kernel::{
        compile_sort_kernel_spec, KeyDesc, SortDirection, SortKernelSpec, SortLayout,
    };
    let spec = SortKernelSpec {
        keys: vec![KeyDesc {
            dtype: DataType::Int32,
            direction: SortDirection::Asc,
            nullable: false,
            nulls_first: false,
        }],
        layout: SortLayout::Shmem,
        shmem_n_pow2: 128,
    };
    let ptx = compile_sort_kernel_spec(&spec).expect("compile");
    // Shmem layout bakes n_pow2 into the entry name.
    assert!(ptx.contains(".visible .entry bolt_bitonic_sort_sh_n128"), "{ptx}");
    // Shared-memory key buffer declaration is exclusive to the shmem layout.
    assert!(ptx.contains(".shared .align"), "{ptx}");
    assert!(ptx.contains("sh_k0"), "missing shmem key buffer sh_k0\n{ptx}");
    // In-kernel barriers between substages.
    assert!(ptx.contains("bar.sync 0"), "{ptx}");
}

// ---- Tests: distinct_kernel (adjacent-distinct flag) -----------------------

#[test]
fn golden_distinct_flag_i32_nonnull_smoke() {
    use craton_bolt::jit::distinct_kernel::compile_distinct_flag_kernel;
    let ptx = compile_distinct_flag_kernel(DataType::Int32, false).expect("compile");
    // Non-nullable entry name (no `_v` suffix).
    assert!(
        ptx.contains(".visible .entry bolt_distinct_flag_i32("),
        "{ptx}"
    );
    // tid==0 always-keep + adjacent value compare.
    assert!(ptx.contains("setp.eq.s32 %p1, %r3, 0;"), "{ptx}");
    assert!(ptx.contains("setp.eq.s32 %p4"), "{ptx}");
    // u8 mask stores for keep(1)/drop(0).
    assert!(ptx.contains("st.global.u8 [%rd1], %r20;"), "{ptx}");
    assert!(ptx.contains("st.global.u8 [%rd1], %r21;"), "{ptx}");
    // Non-nullable variant has no 4th param and no validity byte load.
    assert!(!ptx.contains("bolt_distinct_flag_i32_param_3"), "{ptx}");
}

#[test]
fn golden_distinct_flag_i64_nullable_smoke() {
    use craton_bolt::jit::distinct_kernel::compile_distinct_flag_kernel;
    let ptx = compile_distinct_flag_kernel(DataType::Int64, true).expect("compile");
    // Nullable entry name carries the `_v` suffix + a 4th (n_rows) param.
    assert!(
        ptx.contains(".visible .entry bolt_distinct_flag_i64_v("),
        "{ptx}"
    );
    assert!(ptx.contains("bolt_distinct_flag_i64_v_param_3"), "{ptx}");
    // Packed-bit validity loads for self + prev, both-NULL collapse + one-NULL
    // boundary keep.
    assert!(ptx.contains("ld.global.u8 %r10"), "{ptx}");
    assert!(ptx.contains("ld.global.u8 %r11"), "{ptx}");
    assert!(ptx.contains("@%p2 bra DROP;"), "{ptx}");
    assert!(ptx.contains("@%p3 bra KEEP;"), "{ptx}");
    // i64 key compare.
    assert!(ptx.contains("setp.eq.s64"), "{ptx}");
    // ORDERING (not just presence): the validity-derived branches that route a
    // row to KEEP (store 1) vs DROP (store 0) must be emitted BEFORE the
    // keep-store. If a refactor moved the `st.global.u8 …, %r20` keep-store
    // above its `@%p2 bra DROP` / `@%p3 bra KEEP` gates, every row would take
    // the fall-through store regardless of validity — a presence check still
    // passes, so pin the order explicitly.
    assert_emitted_before(&ptx, "@%p2 bra DROP;", "st.global.u8 [%rd1], %r20;");
    assert_emitted_before(&ptx, "@%p3 bra KEEP;", "st.global.u8 [%rd1], %r20;");
}

#[test]
fn golden_distinct_flag_f64_uses_float_eq() {
    use craton_bolt::jit::distinct_kernel::compile_distinct_flag_kernel;
    let ptx = compile_distinct_flag_kernel(DataType::Float64, false).expect("compile");
    // After host -0.0/NaN canonicalisation the device compare is a plain f64
    // equality.
    assert!(ptx.contains("setp.eq.f64"), "{ptx}");
    assert!(ptx.contains("ld.global.f64"), "{ptx}");
}

// ---- Tests: scatter_kernel variants (review L4) ----------------------------

#[test]
fn golden_scatter_kernel_i64_smoke() {
    use craton_bolt::jit::scatter_kernel_i64::compile_scatter_kernel_i64;
    let ptx = compile_scatter_kernel_i64().expect("compile");
    assert!(ptx.contains(".visible .entry bolt_scatter_i64"), "{ptx}");
    // i64 key variant -- must load the key with s64.
    assert!(ptx.contains("ld.global.s64"), "{ptx}");
    // Cursor reservation uses atom.global.add.u32 (per-partition u32 counter).
    assert!(ptx.contains("atom.global.add.u32"), "{ptx}");
}

#[test]
fn golden_scatter_with_dest_idx_kernel_smoke() {
    use craton_bolt::jit::scatter_with_dest_idx_kernel::compile_scatter_with_dest_idx_kernel;
    let ptx = compile_scatter_with_dest_idx_kernel().expect("compile");
    assert!(ptx.contains(".visible .entry bolt_scatter_with_dest_idx"), "{ptx}");
    assert!(ptx.contains("atom.global.add.u32"), "missing cursor atomic\n{ptx}");
    // Two distinct u32 stores: out_keys[dest] and dest_idx[tid].
    assert!(ptx.contains("st.global.u32"), "{ptx}");
}

#[test]
fn golden_scatter_values_by_dest_idx_kernel_smoke() {
    use craton_bolt::jit::scatter_values_by_dest_idx_kernel::compile_scatter_values_by_dest_idx_kernel;
    let ptx = compile_scatter_values_by_dest_idx_kernel().expect("compile");
    assert!(
        ptx.contains(".visible .entry bolt_scatter_values_by_dest_idx"),
        "{ptx}"
    );
    // f64 value scatter -- no cursor atomic, just an indexed store.
    assert!(ptx.contains("st.global.f64"), "missing f64 value store\n{ptx}");
}

// ---- Tests: agg_kernels op x dtype matrix (review L4) ----------------------
//
// Sum/Int32 is already covered by `golden_sum_int32_widens_to_s64_in_kernel`
// above. The 11 tests below cover the remaining op x dtype combinations,
// each pinning the combine mnemonic from `ReduceOp::combine_ptx`.

#[test]
fn golden_agg_kernel_sum_int64_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Sum, DataType::Int64).expect("compile");
    assert!(ptx.contains(".visible .entry bolt_reduce"), "{ptx}");
    assert!(ptx.contains("add.s64"), "{ptx}");
    // Source-column load routed through the read-only cache.
    assert!(ptx.contains("ld.global.nc.s64"), "{ptx}");
}

#[test]
fn golden_agg_kernel_sum_float32_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Sum, DataType::Float32).expect("compile");
    assert!(ptx.contains("add.f32"), "{ptx}");
    assert!(ptx.contains("ld.global.nc.f32"), "{ptx}");
}

#[test]
fn golden_agg_kernel_sum_float64_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Sum, DataType::Float64).expect("compile");
    assert!(ptx.contains("add.f64"), "{ptx}");
    assert!(ptx.contains("ld.global.nc.f64"), "{ptx}");
}

#[test]
fn golden_agg_kernel_min_int32_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Min, DataType::Int32).expect("compile");
    assert!(ptx.contains("min.s32"), "{ptx}");
}

#[test]
fn golden_agg_kernel_min_int64_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Min, DataType::Int64).expect("compile");
    assert!(ptx.contains("min.s64"), "{ptx}");
}

#[test]
fn golden_agg_kernel_min_float32_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Min, DataType::Float32).expect("compile");
    assert!(ptx.contains("min.f32"), "{ptx}");
}

#[test]
fn golden_agg_kernel_min_float64_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Min, DataType::Float64).expect("compile");
    assert!(ptx.contains("min.f64"), "{ptx}");
}

#[test]
fn golden_agg_kernel_max_int32_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Max, DataType::Int32).expect("compile");
    assert!(ptx.contains("max.s32"), "{ptx}");
}

#[test]
fn golden_agg_kernel_max_int64_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Max, DataType::Int64).expect("compile");
    assert!(ptx.contains("max.s64"), "{ptx}");
}

#[test]
fn golden_agg_kernel_max_float32_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Max, DataType::Float32).expect("compile");
    assert!(ptx.contains("max.f32"), "{ptx}");
}

#[test]
fn golden_agg_kernel_max_float64_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Max, DataType::Float64).expect("compile");
    assert!(ptx.contains("max.f64"), "{ptx}");
}

// ---- Snapshot tests (insta) -------------------------------------------------
//
// Each test below mirrors a substring assertion test above, but pins the
// *full normalized PTX* via `insta::assert_snapshot!`. The substring tests
// catch dropped mnemonics; these snapshot tests catch register-flow
// regressions where a refactor changes which register feeds which
// instruction (the substring would still pass, but the wiring would be
// silently wrong).
//
// First-run bootstrap (`#[ignore]`):
//   These tests are `#[ignore]`'d so the default `cargo test` run on a
//   fresh checkout does NOT fail just because the snapshot files don't
//   exist yet. To populate / refresh the snapshots, run:
//
//     cargo insta test --accept -- --include-ignored
//
//   (or `cargo test --include-ignored` followed by `cargo insta accept`).
//   Once the snapshot files exist under `tests/snapshots/` they can be
//   checked into version control, and a future PR that changes the
//   normalized PTX will produce a reviewable diff via `cargo insta review`.
//
//   We use `omit_expression => true` so each snapshot file is just the
//   normalized PTX payload (no `expression: ...` header) — small, stable,
//   and trivially diffable.

/// Wrap `insta::assert_snapshot!` with our preferred settings. The
/// `omit_expression` setting keeps the snapshot file contents to just the
/// payload (no `expression: "normalize_ptx(&ptx)"` boilerplate) so the
/// diffs read as pure PTX.
macro_rules! assert_ptx_snapshot {
    ($name:expr, $ptx:expr) => {{
        ::insta::with_settings!({ omit_expression => true }, {
            ::insta::assert_snapshot!($name, normalize_ptx(&$ptx));
        });
    }};
}

// The bootstrap-gate reason — repeated in every `#[ignore = "..."]` below
// because Rust requires that attribute argument to be a string literal (it
// cannot reference a const). To avoid drift, edit all sites together.
//
// Bootstrap: `cargo insta test --accept -- --include-ignored`

#[test]
#[ignore = "bootstrap"]
fn snapshot_scalar_projection_int32() {
    let ptx = build_ptx_for("SELECT int_col + 1 FROM t");
    assert_ptx_snapshot!("scalar_projection_int32", ptx);
}

#[test]
#[ignore = "bootstrap"]
fn snapshot_scalar_projection_float64() {
    let ptx = build_ptx_for("SELECT f64_col * 2.0 FROM t");
    assert_ptx_snapshot!("scalar_projection_float64", ptx);
}

#[test]
#[ignore = "bootstrap"]
fn snapshot_predicate_filter_int32() {
    let ptx = build_ptx_for("SELECT int_col FROM t WHERE int_col = 5");
    assert_ptx_snapshot!("predicate_filter_int32", ptx);
}

#[test]
#[ignore = "bootstrap"]
fn snapshot_predicate_filter_and_or() {
    let ptx = build_ptx_for("SELECT a FROM t WHERE a = 1 AND (b = 2 OR c = 3)");
    assert_ptx_snapshot!("predicate_filter_and_or", ptx);
}

#[test]
#[ignore = "bootstrap"]
fn snapshot_sum_int32_reduction_kernel() {
    let ptx = compile_reduction_kernel(ReduceOp::Sum, DataType::Int32).expect("compile");
    assert_ptx_snapshot!("sum_int32_reduction_kernel", ptx);
}

#[test]
#[ignore = "bootstrap"]
fn snapshot_groupby_keys_kernel() {
    let ptx = compile_groupby_keys_kernel().expect("compile keys kernel");
    assert_ptx_snapshot!("groupby_keys_kernel", ptx);
}

#[test]
#[ignore = "bootstrap"]
fn snapshot_prefix_scan_kernel() {
    let ptx = compile_prefix_scan_kernel().expect("compile prefix scan");
    assert_ptx_snapshot!("prefix_scan_kernel", ptx);
}

/// Snapshot test for the Blelloch variant. Bootstrap with
/// `cargo insta test --accept -- --include-ignored` — the snapshot will
/// land alongside the Hillis-Steele one under `tests/snapshots/` and
/// later refactors that drift the normalized PTX will produce a
/// reviewable diff.
#[test]
#[ignore = "bootstrap"]
fn snapshot_prefix_scan_kernel_blelloch() {
    let ptx = compile_prefix_scan_kernel_blelloch().expect("compile blelloch");
    assert_ptx_snapshot!("prefix_scan_kernel_blelloch", ptx);
}

#[test]
#[ignore = "bootstrap"]
fn snapshot_float_atomic_min_kernel() {
    let ptx = compile_groupby_float_atomic_kernel(ReduceOp::Min, DataType::Float64)
        .expect("compile float atomic kernel");
    assert_ptx_snapshot!("float_atomic_min_kernel", ptx);
}

// ---- Unit tests for the normalizer itself ----------------------------------

#[test]
fn normalize_ptx_assigns_stable_indices_per_class() {
    let input = "
        ld.global.s32 %r5, [%rd2];
        ld.global.s32 %r7, [%rd2];
        add.s32       %r9, %r5, %r7;
        setp.eq.s32   %p3, %r9, 0;
        @%p3 bra DONE;
    ";
    let out = normalize_ptx(input);
    // First-seen `%rd2` → `%rd{0}`; first-seen `%r5` → `%r{0}`, then `%r7` →
    // `%r{1}`, etc. Per-class numbering: `%p3` is the first predicate so it
    // becomes `%p{0}`.
    assert!(out.contains("[%rd{0}]"), "rd numbering broken: {out}");
    assert!(out.contains("%r{0}"), "r numbering broken: {out}");
    assert!(out.contains("%r{1}"), "r numbering broken: {out}");
    assert!(out.contains("%r{2}"), "r numbering broken: {out}");
    assert!(out.contains("%p{0}"), "p numbering broken: {out}");
    // Original numbers must be gone.
    assert!(!out.contains("%r5"), "raw %r5 leaked: {out}");
    assert!(!out.contains("%rd2"), "raw %rd2 leaked: {out}");
    assert!(!out.contains("%p3"), "raw %p3 leaked: {out}");
    // Labels are NOT normalized.
    assert!(out.contains("bra DONE"), "label dropped: {out}");
}

#[test]
fn normalize_ptx_separates_classes() {
    // `%r1` and `%rd1` and `%p1` must each become `%X{0}` (per-class), not
    // collide on a single shared counter.
    let out = normalize_ptx("mov %r1, %rd1; setp %p1, %r1, 0;");
    assert!(out.contains("%r{0}"));
    assert!(out.contains("%rd{0}"));
    assert!(out.contains("%p{0}"));
}

#[test]
fn normalize_ptx_strips_inline_asm_address_comments() {
    let input = "
        add.s64 %rd1, %rd2, %rd3;
        // inline asm 0xabc123
        st.global.s64 [%rd4], %rd1;
    ";
    let out = normalize_ptx(input);
    assert!(
        !out.contains("inline asm 0x"),
        "address comment not stripped: {out}"
    );
    // Surrounding instructions survive.
    assert!(out.contains("add.s64"));
    assert!(out.contains("st.global.s64"));
}

#[test]
fn normalize_ptx_preserves_float_classes() {
    // Important: `%fd3` must be matched as the `%fd` (f64) class, NOT as
    // `%f` with an identifier-suffix `d3`. The `parse_reg_suffix` helper
    // rejects trailing letters precisely so the longest-prefix dispatch
    // (`%fd` before `%f`) is the only way `%fd3` can match.
    let input = "ld.global.f32 %f2, [%rd1]; mul.f64 %fd3, %fd4, %fd5;";
    let out = normalize_ptx(input);
    // `%f2` is the only 32-bit float register here → `%f{0}`.
    assert!(out.contains("%f{0}"), "%f class broken: {out}");
    // Three distinct `%fd` registers in declaration order.
    assert!(out.contains("%fd{0}"), "%fd class broken: {out}");
    assert!(out.contains("%fd{1}"));
    assert!(out.contains("%fd{2}"));
    // `%rd1` got its own class.
    assert!(out.contains("%rd{0}"));
}

#[test]
fn normalize_ptx_leaves_register_vector_declarations_alone() {
    // `.reg .b64 %rd<24>;` declares a 24-element register vector. The `<24>`
    // is a size, not a register reference — our normalizer must NOT touch it
    // (`parse_reg_suffix` rejects non-digit follow-ups, and `<` is one). The
    // declaration thus stays in the snapshot, where a register-count change
    // is itself a real codegen contract diff worth catching.
    let input = "\t.reg .pred  %p<8>;\n\t.reg .b64   %rd<24>;\n\tld.global.s64 %rd2, [%rd1];";
    let out = normalize_ptx(input);
    assert!(out.contains("%p<8>"), "vector decl mangled: {out}");
    assert!(out.contains("%rd<24>"), "vector decl mangled: {out}");
    // Usages still normalized.
    assert!(out.contains("%rd{0}"), "usage not normalized: {out}");
    assert!(out.contains("%rd{1}"), "usage not normalized: {out}");
}

// ---- Tests: GPU string kernels (variable-width Utf8 codegen) -----------------
//
// These pin the load-bearing PTX shape of `jit::string_kernel`: the fully-GPU
// fixed-width LENGTH dictionary-gather, and the two-pass (length + write)
// variable-width producers for UPPER / LOWER / SUBSTRING. The two-pass design
// reuses the existing prefix-scan kernels between the two passes (the scan is
// not re-emitted here).

#[test]
fn golden_string_length_gather_is_double_indirection() {
    // LENGTH on a dictionary column: out[tid] = length_table[indices[tid]].
    // The two read-only-cache loads (indices, then the table at that index)
    // plus a single Int32 store are the contract.
    let ptx = compile_length_gather_kernel().expect("compile length gather");
    assert!(
        ptx.contains(".visible .entry bolt_str_length_gather("),
        "missing entry name\n{ptx}"
    );
    // 4-arg ABI: indices, length_table, out, n_rows.
    assert!(ptx.contains(".param .u64 bolt_str_length_gather_param_0,"));
    assert!(ptx.contains(".param .u64 bolt_str_length_gather_param_1,"));
    assert!(ptx.contains(".param .u64 bolt_str_length_gather_param_2,"));
    assert!(ptx.contains(".param .u32 bolt_str_length_gather_param_3"));
    let n_nc = ptx.matches("ld.global.nc.u32").count();
    assert!(
        n_nc >= 2,
        "expected >=2 read-only-cache loads (indices + table), got {n_nc}\n{ptx}"
    );
    assert!(ptx.contains("st.global.u32"), "missing Int32 store\n{ptx}");
    // The n_rows guard must precede the store.
    assert_appears_before(&ptx, "bra DONE", "st.global.u32");
}

#[test]
fn golden_string_upper_len_pass_is_length_preserving() {
    // Pass 1 for UPPER: out_len == in_len = src_offsets[tid+1]-src_offsets[tid].
    let ptx = compile_varwidth_len_pass(ScalarFnKind::Upper).expect("compile");
    assert!(
        ptx.contains(".visible .entry bolt_str_len_pass_upper("),
        "missing entry name\n{ptx}"
    );
    // 4-arg ABI; UPPER has no start/len params.
    assert!(ptx.contains(".param .u32 bolt_str_len_pass_upper_param_3"));
    assert!(
        !ptx.contains("bolt_str_len_pass_upper_param_4"),
        "UPPER len pass must have exactly 4 params\n{ptx}"
    );
    // The input-length subtraction (end - begin) feeds the row_lens store.
    assert!(ptx.contains("sub.s32"), "missing end-begin length\n{ptx}");
    assert!(ptx.contains("st.global.u32"), "missing row_lens store\n{ptx}");
}

#[test]
fn golden_string_substring_len_pass_clamps_with_start_len() {
    // Pass 1 for SUBSTRING takes start + sub_len params and computes the output
    // byte length by walking WHOLE UTF-8 characters (1-based char start, char
    // length) — NOT byte offsets — so a multi-byte char is never split/leaked.
    let ptx = compile_varwidth_len_pass(ScalarFnKind::Substring).expect("compile");
    assert!(ptx.contains(".visible .entry bolt_str_len_pass_substring("));
    // 6-arg ABI: ..., n_rows, start, sub_len.
    assert!(ptx.contains(".param .u32 bolt_str_len_pass_substring_param_4,"));
    assert!(ptx.contains(".param .u32 bolt_str_len_pass_substring_param_5"));
    // Char-window walk: skip (start-1) whole chars to the byte start, then take
    // sub_len whole chars, using the UTF-8 lead-byte test `(b & 0xC0) != 0x80`
    // (mask 0xC0 == 192). This replaced the old byte-offset max/min clamp that
    // could splice adjacent bytes of a multi-byte character.
    assert!(ptx.contains("LEN_SKIP:"), "missing char-skip loop\n{ptx}");
    assert!(ptx.contains("LEN_TAKE:"), "missing char-take loop\n{ptx}");
    assert!(ptx.contains(", 192"), "missing UTF-8 continuation-byte mask (0xC0)\n{ptx}");
}

#[test]
fn golden_string_upper_write_pass_ascii_case_folds_in_loop() {
    // Pass 2 for UPPER: per-byte copy loop with an ASCII a-z → A-Z fold.
    let ptx = compile_varwidth_write_pass(ScalarFnKind::Upper).expect("compile");
    assert!(ptx.contains(".visible .entry bolt_str_write_pass_upper("));
    // The per-byte copy loop is the heart of the write pass.
    assert!(ptx.contains("WRITE_LOOP:"), "missing loop label\n{ptx}");
    assert!(ptx.contains("WRITE_DONE:"), "missing loop exit\n{ptx}");
    // ASCII fold: 'a'(97) / 'z'(122) range test, subtract 32.
    assert!(ptx.contains("97") && ptx.contains("122"), "missing a-z bounds\n{ptx}");
    assert!(ptx.contains("sub.s32 %r12, %r11, 32"), "missing -32 fold\n{ptx}");
    assert!(ptx.contains("st.global.u8"), "missing per-byte store\n{ptx}");
    // The loop bound check precedes the byte store.
    assert_appears_before(&ptx, "WRITE_LOOP:", "st.global.u8");
}

#[test]
fn golden_string_lower_write_pass_ascii_case_folds_up() {
    let ptx = compile_varwidth_write_pass(ScalarFnKind::Lower).expect("compile");
    assert!(ptx.contains(".visible .entry bolt_str_write_pass_lower("));
    // ASCII fold: 'A'(65) / 'Z'(90), add 32.
    assert!(ptx.contains("65") && ptx.contains("90"), "missing A-Z bounds\n{ptx}");
    assert!(ptx.contains("add.s32 %r12, %r11, 32"), "missing +32 fold\n{ptx}");
}

#[test]
fn golden_string_substring_write_pass_is_plain_copy() {
    let ptx = compile_varwidth_write_pass(ScalarFnKind::Substring).expect("compile");
    assert!(ptx.contains(".visible .entry bolt_str_write_pass_substring("));
    // 7-arg ABI (start + sub_len appended).
    assert!(ptx.contains(".param .u32 bolt_str_write_pass_substring_param_5,"));
    assert!(ptx.contains(".param .u32 bolt_str_write_pass_substring_param_6"));
    // Plain byte copy, no case fold.
    assert!(ptx.contains("mov.b32 %r13, %r11"), "substring must be a plain copy\n{ptx}");
    assert!(
        !ptx.contains("sub.s32 %r12, %r11, 32"),
        "substring must not case-fold\n{ptx}"
    );
}

#[test]
fn golden_string_concat_two_pass_is_deferred() {
    // CONCAT is the deferred multi-input two-pass producer; both passes reject
    // it with a clear message so callers fall back to the host path.
    let e = compile_varwidth_len_pass(ScalarFnKind::Concat).unwrap_err();
    assert!(format!("{e}").contains("CONCAT"), "{e}");
    let e = compile_varwidth_write_pass(ScalarFnKind::Concat).unwrap_err();
    assert!(format!("{e}").contains("CONCAT"), "{e}");
}

// ---- Tests: CASE over Date32 / Timestamp result types ----------------------
//
// v0.7: `Codegen::emit_case` accepts Date32 (i32 storage) and Timestamp (i64
// storage) as CASE result dtypes — they are fixed-width integers that fold
// cleanly through `selp` exactly like Int32 / Int64. The bit-copy nature of
// `selp` means we emit the untyped class suffixes `selp.b32` (Date32) and
// `selp.b64` (Timestamp); no arithmetic interpretation of the value is
// needed. Decimal128 (i128) stays rejected at the plan layer — there is no
// `selp.b128`. These goldens pin the suffix contract end-to-end from SQL.

/// Fixture provider carrying a Date32 column (`d`) and a Timestamp column
/// (`ts`) alongside an i32 key (`id`) so a SQL CASE over the temporal columns
/// types as Date32 / Timestamp and reaches the projection codegen path.
fn temporal_provider() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "d".into(),
            dtype: DataType::Date32,
            nullable: false,
        },
        Field {
            name: "ts".into(),
            dtype: DataType::Timestamp(TimeUnit::Nanosecond, None),
            nullable: false,
        },
    ]);
    MemTableProvider::new().with_table("events", schema)
}

/// Build PTX for a CASE-bearing SQL query over the `temporal_provider`
/// fixture. Mirrors `build_ptx_for` but with the temporal schema; panics if
/// the plan isn't a single projection kernel.
fn build_temporal_ptx_for(sql: &str) -> String {
    let provider = temporal_provider();
    let plan = parse_sql(sql, &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let kernel = match &phys {
        PhysicalPlan::Projection { kernel, .. } => kernel,
        other => panic!(
            "build_temporal_ptx_for: expected Projection plan for `{sql}`, got {other:?}"
        ),
    };
    compile_ptx(kernel, "bolt_test_kernel").expect("compile_ptx")
}

/// CASE producing a Date32 result must fold to `selp.b32` — Date32 is i32
/// days-since-epoch, a plain 32-bit bit-copy through the `selp` else-slot.
/// The Bool cond -> predicate materialisation is the same `setp.ne.s32`
/// used by every other value dtype.
#[test]
fn golden_case_date32_emits_selp_b32() {
    let ptx = build_temporal_ptx_for("SELECT CASE WHEN id > 0 THEN d ELSE d END AS r FROM events");
    assert!(
        ptx.contains("setp.ne.s32"),
        "Bool cond -> predicate materialisation should be setp.ne.s32\n{ptx}"
    );
    assert!(
        ptx.contains("selp.b32"),
        "Date32 CASE result must fold to selp.b32 (i32 bit-copy)\n{ptx}"
    );
    // Must NOT mis-emit a 64-bit select (would alias an adjacent register
    // word) for an i32-storage temporal type.
    assert!(
        !ptx.contains("selp.b64"),
        "Date32 CASE must not emit selp.b64\n{ptx}"
    );
}

/// CASE producing a Timestamp result must fold to `selp.b64` — Timestamp is
/// i64 ticks-since-epoch, a plain 64-bit bit-copy. The predicate setup is
/// unchanged from the Int32 / Date32 paths.
#[test]
fn golden_case_timestamp_emits_selp_b64() {
    let ptx =
        build_temporal_ptx_for("SELECT CASE WHEN id > 0 THEN ts ELSE ts END AS r FROM events");
    assert!(
        ptx.contains("setp.ne.s32"),
        "Bool cond -> predicate materialisation should be setp.ne.s32\n{ptx}"
    );
    assert!(
        ptx.contains("selp.b64"),
        "Timestamp CASE result must fold to selp.b64 (i64 bit-copy)\n{ptx}"
    );
    // Must NOT truncate an i64-storage temporal type to a 32-bit select.
    assert!(
        !ptx.contains("selp.b32"),
        "Timestamp CASE must not emit selp.b32 (would truncate i64 ticks)\n{ptx}"
    );
}


// ---- Window-function kernels (GPU framed window path) -----------------------

/// Boundary-flag kernel: pinned entry name + 5-arg ABI + the two i64 key
/// compares and two u8 flag stores that define partition/peer boundaries.
#[test]
fn golden_window_boundary_flags_abi() {
    let ptx = compile_boundary_flag_kernel().expect("compile boundary kernel");
    assert!(
        ptx.contains(".visible .entry bolt_window_boundary_flags("),
        "missing boundary entry\n{ptx}"
    );
    // 4 pointers + n_rows; no 6th param.
    assert!(ptx.contains("bolt_window_boundary_flags_param_4"));
    assert!(!ptx.contains("bolt_window_boundary_flags_param_5"));
    // Partition-key and order-key inequality compares (i64 lane).
    assert!(
        ptx.matches("setp.ne.s64").count() >= 2,
        "expected >=2 i64 ne compares\n{ptx}"
    );
    // peer_head = part_head | order_changed.
    assert!(ptx.contains("or.b32"), "missing peer-head OR\n{ptx}");
    // Two u8 flag stores (part_head, peer_head).
    assert_eq!(
        ptx.matches("st.global.u8").count(),
        2,
        "expected exactly 2 u8 flag stores\n{ptx}"
    );
    // ORDERING (not just presence): the `tid >= n_rows` bounds gate
    // (`@%p0 bra DONE`) must be emitted BEFORE the first flag store. If the
    // store were hoisted above the gate, an out-of-range thread would write
    // past the end of the part_head/peer_head buffers — a presence check on
    // the gate + the store cannot catch that, so pin the order.
    assert_emitted_before(&ptx, "@%p0 bra DONE;", "st.global.u8");
}

/// Segmented-scan kernel: pinned entry name + 4-arg ABI + segmented combine
/// (flag OR, value add, segment-reset select) + log2(BLOCK_SIZE) barriers.
#[test]
fn golden_window_segmented_scan_abi_and_combine() {
    let ptx = compile_segmented_scan_kernel().expect("compile segmented scan");
    assert!(
        ptx.contains(".visible .entry bolt_window_segmented_scan("),
        "missing segmented-scan entry\n{ptx}"
    );
    assert!(ptx.contains("bolt_window_segmented_scan_param_3"));
    assert!(!ptx.contains("bolt_window_segmented_scan_param_4"));
    // Segmented combine: OR the flags, add the values, select on the head flag.
    assert!(ptx.contains("or.b32"), "missing flag OR\n{ptx}");
    assert!(ptx.contains("add.s64"), "missing value add\n{ptx}");
    assert!(ptx.contains("selp.b64"), "missing segment-reset select\n{ptx}");
    // Shared buffer = 2 ping-pong * BLOCK_SIZE * 16-byte (i64+flag) records.
    let expect = WINDOW_BLOCK_SIZE * 16 * 2;
    assert!(
        ptx.contains(&format!(".shared .align 8 .b8 sdata[{expect}];")),
        "incorrect shared-buffer size (want {expect})\n{ptx}"
    );
    // Barriers: one seed + one per Hillis-Steele round.
    let expect_bars = 1 + WINDOW_BLOCK_SIZE.trailing_zeros() as usize;
    assert_eq!(
        ptx.matches("bar.sync 0;").count(),
        expect_bars,
        "expected {expect_bars} barriers\n{ptx}"
    );
}
