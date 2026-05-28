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
use craton_bolt::jit::prefix_scan::compile_prefix_scan_kernel;
use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, PhysicalPlan, Schema,
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
    // so the load is s32 but the arithmetic and store run at s64.
    assert!(ptx.contains("ld.global.s32"), "missing s32 load\n{ptx}");
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
    assert!(ptx.contains("ld.global.f64"), "missing f64 load\n{ptx}");
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
    let bra_pos = ptx[probe_start + setp_pos..]
        .find("bra DONE")
        .expect("expected @%pN bra DONE immediately after the bound check");
    // Sanity: the bra DONE must come within a few lines of the setp.
    assert!(
        bra_pos < 100,
        "bra DONE too far from setp.gt.u32 (probe bound check broken)\n{ptx}"
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
    // precomputed `max_probes` register, and branch to the `DONE` exit
    // label on overflow (silent-drop semantics — no atomic is issued for
    // the over-probing row, matching the keys kernel).
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
        .find("bra DONE")
        .expect("expected @%pN bra DONE immediately after the probe bound");
    assert!(
        bra_pos < 100,
        "bra DONE too far from setp.gt.u32 (probe bound check broken)\n{ptx}"
    );
    // The give-up `bra DONE` must precede the `FOUND` label so a thread
    // that exceeds the bound exits without issuing the atomic update.
    let bra_done_abs = probe_start + setp_pos + bra_pos;
    let found_pos = ptx.find("FOUND:").expect("FOUND label exists");
    assert!(
        bra_done_abs < found_pos,
        "probe-bound `bra DONE` must precede the FOUND label (otherwise the \
         atomic update still fires on over-probe)\n{ptx}"
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
    use craton_bolt::jit::partition_reduce_kernel::compile_partition_reduce_kernel;
    let ptx = compile_partition_reduce_kernel().expect("kernel compiles");
    let mb_count = ptx.matches("membar.cta").count();
    assert!(
        mb_count >= 2,
        "partition-reduce kernel must emit >=2 membar.cta (CLAIM + MATCH \
         paths); saw {mb_count}:\n{ptx}"
    );
    // Ordering: the MATCH-path membar must sit between the CAS and the
    // key load. Search anchored at the CAS to dodge false hits in
    // comments at the top of the file.
    let cas_pos = ptx
        .find("atom.shared.cas.b32")
        .expect("partition-reduce kernel must issue atom.shared.cas.b32");
    let tail = &ptx[cas_pos..];
    let mb_after_cas = tail
        .find("membar.cta")
        .expect("missing MATCH-path membar.cta after CAS");
    let key_load = tail
        .find("ld.shared.s32 %r35")
        .expect("missing MATCH-path key load");
    assert!(
        mb_after_cas < key_load,
        "membar.cta must precede the MATCH-path key load:\n{ptx}"
    );
    // Ordering: the CLAIM-path membar must sit between the key store
    // and the f64 val atomic. CLAIM-path key store is the
    // `st.shared.u32 [%rd36], %r31;` line.
    let claim_label = ptx.find("CLAIM:").expect("missing CLAIM: label");
    let claim_tail = &ptx[claim_label..];
    let key_store = claim_tail
        .find("st.shared.u32")
        .expect("missing CLAIM-path key store");
    let mb_after_store = claim_tail[key_store..]
        .find("membar.cta")
        .expect("missing CLAIM-path membar.cta after key store");
    let val_atomic = claim_tail[key_store..]
        .find("atom.shared.add.f64")
        .expect("missing CLAIM-path val atomic");
    assert!(
        mb_after_store < val_atomic,
        "membar.cta must precede the CLAIM-path val atomic:\n{ptx}"
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
    assert!(ptx.contains("ld.global.s64"), "{ptx}");
}

#[test]
fn golden_agg_kernel_sum_float32_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Sum, DataType::Float32).expect("compile");
    assert!(ptx.contains("add.f32"), "{ptx}");
    assert!(ptx.contains("ld.global.f32"), "{ptx}");
}

#[test]
fn golden_agg_kernel_sum_float64_smoke() {
    let ptx = compile_reduction_kernel(ReduceOp::Sum, DataType::Float64).expect("compile");
    assert!(ptx.contains("add.f64"), "{ptx}");
    assert!(ptx.contains("ld.global.f64"), "{ptx}");
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
#[ignore = "bootstrap-gated snapshot; populate via `cargo insta test --accept -- --include-ignored`"]
fn snapshot_scalar_projection_int32() {
    let ptx = build_ptx_for("SELECT int_col + 1 FROM t");
    assert_ptx_snapshot!("scalar_projection_int32", ptx);
}

#[test]
#[ignore = "bootstrap-gated snapshot; populate via `cargo insta test --accept -- --include-ignored`"]
fn snapshot_scalar_projection_float64() {
    let ptx = build_ptx_for("SELECT f64_col * 2.0 FROM t");
    assert_ptx_snapshot!("scalar_projection_float64", ptx);
}

#[test]
#[ignore = "bootstrap-gated snapshot; populate via `cargo insta test --accept -- --include-ignored`"]
fn snapshot_predicate_filter_int32() {
    let ptx = build_ptx_for("SELECT int_col FROM t WHERE int_col = 5");
    assert_ptx_snapshot!("predicate_filter_int32", ptx);
}

#[test]
#[ignore = "bootstrap-gated snapshot; populate via `cargo insta test --accept -- --include-ignored`"]
fn snapshot_predicate_filter_and_or() {
    let ptx = build_ptx_for("SELECT a FROM t WHERE a = 1 AND (b = 2 OR c = 3)");
    assert_ptx_snapshot!("predicate_filter_and_or", ptx);
}

#[test]
#[ignore = "bootstrap-gated snapshot; populate via `cargo insta test --accept -- --include-ignored`"]
fn snapshot_sum_int32_reduction_kernel() {
    let ptx = compile_reduction_kernel(ReduceOp::Sum, DataType::Int32).expect("compile");
    assert_ptx_snapshot!("sum_int32_reduction_kernel", ptx);
}

#[test]
#[ignore = "bootstrap-gated snapshot; populate via `cargo insta test --accept -- --include-ignored`"]
fn snapshot_groupby_keys_kernel() {
    let ptx = compile_groupby_keys_kernel().expect("compile keys kernel");
    assert_ptx_snapshot!("groupby_keys_kernel", ptx);
}

#[test]
#[ignore = "bootstrap-gated snapshot; populate via `cargo insta test --accept -- --include-ignored`"]
fn snapshot_prefix_scan_kernel() {
    let ptx = compile_prefix_scan_kernel().expect("compile prefix scan");
    assert_ptx_snapshot!("prefix_scan_kernel", ptx);
}

#[test]
#[ignore = "bootstrap-gated snapshot; populate via `cargo insta test --accept -- --include-ignored`"]
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

