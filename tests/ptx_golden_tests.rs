// SPDX-License-Identifier: Apache-2.0
//
// Golden snapshot tests for emitted PTX. Updates to these tests are intentional
// codegen contract changes — review the PTX diff carefully before accepting.
//
// Strategy:
//   We don't use `insta` (not a project dependency). Instead each test asserts
//   a small set of *stable substrings* that should appear in the emitted
//   module. Stable substrings catch real codegen regressions (instruction
//   mnemonic changes, dropped widening casts, lost loop bounds, wrong shared
//   memory size, etc.) while tolerating cosmetic churn (register numbering,
//   whitespace, label names that aren't externally meaningful).
//
// What's intentionally NOT byte-equality:
//   The `%rN` / `%rdN` / `%pN` register numbers are issued by a counter inside
//   `RegAlloc`, so any new compute op inserted upstream will shift every
//   later name. A full string snapshot would flap on every codegen
//   refactor. The substring assertions below pin the *behavioral contract*
//   (which mnemonics, which dtypes, which structural markers) without
//   pinning the allocator state.

use craton_patina::jit::agg_kernels::{compile_reduction_kernel, ReduceOp};
use craton_patina::jit::compile_ptx;
use craton_patina::jit::float_atomics::compile_groupby_float_atomic_kernel;
use craton_patina::jit::hash_kernels::compile_groupby_keys_kernel;
use craton_patina::jit::prefix_scan::compile_prefix_scan_kernel;
use craton_patina::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, PhysicalPlan, Schema,
};

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
    compile_ptx(kernel, "patina_test_kernel").expect("compile_ptx")
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
        ptx.contains(".visible .entry patina_test_kernel"),
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
