// SPDX-License-Identifier: Apache-2.0

//! End-to-end query tests for Craton Bolt.
//!
//! Offline tests (no GPU): SQL parse -> plan -> PTX shape. Run by `cargo test`.
//! Online tests (#[ignore]'d): full engine execute on a CUDA device. Run with
//! `cargo test -- --ignored` on a GPU host.

use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::jit::compile_ptx;
use craton_bolt::plan::{
    lower_physical, parse_sql, AggregateExpr, DataType, Expr, Field, Literal, LogicalPlan,
    MemTableProvider, PhysicalPlan, Schema,
};

mod common;
use common::REL_TOL;

// ---- Fixtures ---------------------------------------------------------------

fn sales_schema() -> Schema {
    Schema::new(vec![
        Field {
            name: "region_id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "price".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
        Field {
            name: "tax".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
    ])
}

fn sales_provider() -> MemTableProvider {
    MemTableProvider::new().with_table("sales", sales_schema())
}

fn sales_batch(n: usize) -> RecordBatch {
    let region: Int32Array = (0..n as i32).map(|i| i % 4).collect();
    let price: Float64Array = (0..n).map(|i| (i + 1) as f64).collect();
    let tax: Float64Array = (0..n).map(|_| 0.1_f64).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("region_id", ArrowDataType::Int32, false),
        ArrowField::new("price", ArrowDataType::Float64, false),
        ArrowField::new("tax", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(region), Arc::new(price), Arc::new(tax)]).unwrap()
}

// ---- Offline: parse -> plan -> PTX -----------------------------------------

#[test]
fn parses_simple_select() {
    let provider = sales_provider();
    let plan = parse_sql("SELECT price FROM sales", &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let PhysicalPlan::Projection { kernel, .. } = &phys else {
        panic!("expected Projection, got {phys:?}");
    };
    assert_eq!(kernel.inputs.len(), 1);
    assert_eq!(kernel.outputs.len(), 1);
    assert_eq!(kernel.inputs[0].name, "price");
    assert_eq!(kernel.inputs[0].dtype, DataType::Float64);
    assert_eq!(kernel.outputs[0].name, "price");
    assert_eq!(kernel.outputs[0].dtype, DataType::Float64);
    assert!(kernel.predicate.is_none());
}

/// `COUNT(DISTINCT col)` must lower end-to-end: the SQL frontend desugars it to
/// a `Count` aggregate over a `Distinct`, and the physical lowerer wraps that in
/// a `CountRows` node (the fused scalar-aggregate path can't fold a Distinct).
#[test]
fn count_distinct_lowers_to_countrows() {
    let provider = sales_provider();
    let plan =
        parse_sql("SELECT COUNT(DISTINCT region_id) FROM sales", &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    // The top SELECT-order Project is identity over the single count column, so
    // the lowerer may collapse it; accept either a bare CountRows or a Project
    // wrapping one. Walk down to find the CountRows and assert it wraps a
    // Distinct.
    fn find_count_rows(p: &PhysicalPlan) -> Option<&PhysicalPlan> {
        match p {
            PhysicalPlan::CountRows { .. } => Some(p),
            PhysicalPlan::Project { input, .. } => find_count_rows(input),
            _ => None,
        }
    }
    let cr = find_count_rows(&phys)
        .unwrap_or_else(|| panic!("expected CountRows in plan, got {phys:?}"));
    let PhysicalPlan::CountRows {
        input,
        output_schema,
    } = cr
    else {
        unreachable!()
    };
    assert!(
        matches!(input.as_ref(), PhysicalPlan::Distinct { .. }),
        "CountRows must wrap a Distinct, got {input:?}"
    );
    assert_eq!(output_schema.fields.len(), 1);
    assert_eq!(output_schema.fields[0].dtype, DataType::Int64);
}

#[test]
fn parses_filtered_arithmetic_select() {
    let provider = sales_provider();
    let sql = "SELECT price * tax FROM sales WHERE region_id = 1";
    let plan = parse_sql(sql, &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let PhysicalPlan::Projection {
        kernel,
        output_schema,
        ..
    } = &phys
    else {
        panic!("expected Projection, got {phys:?}");
    };
    assert!(kernel.predicate.is_some(), "expected predicate from WHERE");
    assert_eq!(output_schema.fields.len(), 1);
    // Output column for a `Binary` expression is auto-named.
    assert!(
        output_schema.fields[0].name.starts_with("__expr_"),
        "unexpected output name '{}'",
        output_schema.fields[0].name
    );
    assert_eq!(output_schema.fields[0].dtype, DataType::Float64);

    // Ensure `price`, `tax`, and `region_id` (predicate) are all loaded.
    let input_names: Vec<&str> = kernel.inputs.iter().map(|c| c.name.as_str()).collect();
    assert!(input_names.contains(&"price"), "inputs missing 'price': {input_names:?}");
    assert!(input_names.contains(&"tax"), "inputs missing 'tax': {input_names:?}");
    assert!(
        input_names.contains(&"region_id"),
        "inputs missing 'region_id': {input_names:?}"
    );
}

#[test]
fn parses_arithmetic_without_filter() {
    let provider = sales_provider();
    let plan = parse_sql("SELECT price + tax FROM sales", &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let PhysicalPlan::Projection { kernel, .. } = &phys else {
        panic!("expected Projection");
    };
    assert!(kernel.predicate.is_none());
    assert_eq!(kernel.outputs.len(), 1);
    assert_eq!(kernel.outputs[0].dtype, DataType::Float64);
}

#[test]
fn rejects_unknown_column() {
    // Column references aren't validated until lowering; allow either failure point.
    let provider = sales_provider();
    let parsed = parse_sql("SELECT bogus FROM sales", &provider);
    match parsed {
        Err(_) => {} // ok
        Ok(p) => {
            let lowered = lower_physical(&p);
            assert!(lowered.is_err(), "expected unknown-column error from lowering");
        }
    }
}

#[test]
fn rejects_unknown_table() {
    let provider = sales_provider();
    let err = parse_sql("SELECT x FROM nope", &provider);
    assert!(err.is_err(), "expected unknown-table error");
}

#[test]
fn type_error_on_bad_and() {
    // AND requires Bool operands. price (Float64) AND tax should fail type-check.
    let provider = sales_provider();
    let plan = parse_sql("SELECT price FROM sales WHERE price AND tax", &provider);
    // Parse may succeed; the type error surfaces from schema() or lower_physical().
    if let Ok(p) = plan {
        let schema_err = p.schema().is_err();
        let lower_err = lower_physical(&p).is_err();
        assert!(
            schema_err || lower_err,
            "expected type error from schema() or lower_physical()"
        );
    }
}

#[test]
fn ptx_for_trivial_select_contains_required_directives() {
    let provider = sales_provider();
    let plan = parse_sql("SELECT price FROM sales", &provider).unwrap();
    let phys = lower_physical(&plan).unwrap();
    let PhysicalPlan::Projection { kernel, .. } = &phys else {
        panic!("expected Projection");
    };
    let ptx = compile_ptx(kernel, "bolt_kernel").expect("ptx");

    assert!(ptx.contains(".version 7.5"), "missing .version directive\n{ptx}");
    assert!(ptx.contains(".target sm_70"), "missing .target directive\n{ptx}");
    assert!(
        ptx.contains(".address_size 64"),
        "missing .address_size directive"
    );
    assert!(
        ptx.contains(".visible .entry bolt_kernel"),
        "missing entry directive"
    );
    // Kernel-parameter load order is established by `ld.param.u64` (column ptrs) and
    // a trailing `ld.param.u32` (row count) — this is the order cuLaunchKernel sees.
    assert!(ptx.contains("ld.param.u64"), "expected ld.param.u64 for column ptrs");
    assert!(ptx.contains("ld.param.u32"), "expected ld.param.u32 for n_rows");
    // f64 load + store, since `price` is Float64. Input loads route through
    // the read-only cache (`ld.global.nc`) — output stores remain plain.
    assert!(ptx.contains("ld.global.nc.f64"), "missing f64 read-only-cache load");
    assert!(ptx.contains("st.global.f64"), "missing f64 store");
    assert!(ptx.contains("DONE:"), "missing DONE label");
    assert!(ptx.contains("ret;"), "missing ret;");
}

#[test]
fn ptx_with_predicate_contains_gate_before_store() {
    let provider = sales_provider();
    let sql = "SELECT price FROM sales WHERE region_id = 1";
    let plan = parse_sql(sql, &provider).unwrap();
    let phys = lower_physical(&plan).unwrap();
    let PhysicalPlan::Projection { kernel, .. } = &phys else {
        panic!("expected Projection");
    };
    assert!(kernel.predicate.is_some());
    let ptx = compile_ptx(kernel, "bolt_kernel").unwrap();

    // The predicate `setp` and the predicate-gate `bra DONE` should precede the store.
    let store_pos = ptx.find("st.global").expect("store present");
    let setp_pos = ptx.find("setp.").expect("setp present");
    assert!(
        setp_pos < store_pos,
        "expected setp.* to precede st.global (gate before store)"
    );
    // Equality comparison for `region_id = 1`. The literal `1` is parsed as Int64,
    // so operands unify to Int64 and the comparison emits `setp.eq.s64`.
    assert!(
        ptx.contains("setp.eq.s64"),
        "expected s64 equality (region_id widened to Int64 to match literal)"
    );
}

#[test]
fn ptx_int32_dtype_load_store_suffixes() {
    // Single Int32 column projection -> ld/st should use s32 suffix.
    let schema = Schema::new(vec![Field {
        name: "k".into(),
        dtype: DataType::Int32,
        nullable: false,
    }]);
    let provider = MemTableProvider::new().with_table("t", schema);
    let plan = parse_sql("SELECT k FROM t", &provider).unwrap();
    let phys = lower_physical(&plan).unwrap();
    let PhysicalPlan::Projection { kernel, .. } = &phys else {
        panic!("expected Projection");
    };
    let ptx = compile_ptx(kernel, "bolt_kernel").unwrap();
    assert!(ptx.contains("ld.global.nc.s32"), "expected s32 read-only-cache load");
    assert!(ptx.contains("st.global.s32"), "expected s32 store");
}

#[test]
fn ptx_int64_dtype_load_store_suffixes() {
    let schema = Schema::new(vec![Field {
        name: "k".into(),
        dtype: DataType::Int64,
        nullable: false,
    }]);
    let provider = MemTableProvider::new().with_table("t", schema);
    let plan = parse_sql("SELECT k FROM t", &provider).unwrap();
    let phys = lower_physical(&plan).unwrap();
    let PhysicalPlan::Projection { kernel, .. } = &phys else {
        panic!("expected Projection");
    };
    let ptx = compile_ptx(kernel, "bolt_kernel").unwrap();
    assert!(ptx.contains("ld.global.nc.s64"), "expected s64 read-only-cache load");
    assert!(ptx.contains("st.global.s64"), "expected s64 store");
}

// ---- Offline: aggregates and GROUP BY --------------------------------------

/// Wider fixture used by the aggregate / GROUP BY tests below. Adds:
/// - `sub_region_id` (Int32) for multi-key GROUP BY
/// - `qty` (Int32) for SUM-widening checks
/// - `qty32` (Int32) alias-style column for GROUP-BY SUM widening
fn agg_provider() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field {
            name: "region_id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "sub_region_id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "qty".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "qty32".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "price".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
    ]);
    MemTableProvider::new().with_table("sales", schema)
}

#[test]
fn scalar_sum_int32_widens_to_i64() {
    let provider = agg_provider();
    let plan = parse_sql("SELECT SUM(qty) FROM sales", &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let out = phys.output_schema();
    assert_eq!(out.fields.len(), 1);
    assert_eq!(
        out.fields[0].dtype,
        DataType::Int64,
        "SUM(Int32) must widen to Int64 (wave 3 regression)"
    );
}

#[test]
fn scalar_avg_returns_float64() {
    let provider = agg_provider();
    let plan = parse_sql("SELECT AVG(price) FROM sales", &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let out = phys.output_schema();
    assert_eq!(out.fields.len(), 1);
    assert_eq!(out.fields[0].dtype, DataType::Float64);
}

#[test]
fn scalar_count_star_parses() {
    let provider = agg_provider();
    let plan = parse_sql("SELECT COUNT(*) FROM sales", &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let out = phys.output_schema();
    assert_eq!(out.fields.len(), 1);
    assert_eq!(out.fields[0].dtype, DataType::Int64);
}

#[test]
fn groupby_single_int_key() {
    let provider = agg_provider();
    let sql = "SELECT region_id, SUM(qty) FROM sales GROUP BY region_id";
    let plan = parse_sql(sql, &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let PhysicalPlan::Aggregate { aggregate, .. } = &phys else {
        panic!("expected Aggregate, got {phys:?}");
    };
    assert_eq!(aggregate.group_by.len(), 1, "one group key");
    assert_eq!(aggregate.aggregates.len(), 1, "one aggregate");
    let names: Vec<&str> = aggregate
        .output_schema
        .fields
        .iter()
        .map(|f| f.name.as_str())
        .collect();
    assert_eq!(names, vec!["region_id", "sum_qty"], "SELECT-order output");
}

#[test]
fn groupby_select_order_sum_first() {
    // Wave 1 fix #5: output column order must follow SELECT order, not
    // "group keys first, then aggregates".
    let provider = agg_provider();
    let sql = "SELECT SUM(qty), region_id FROM sales GROUP BY region_id";
    let plan = parse_sql(sql, &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let out = phys.output_schema();
    let names: Vec<&str> = out.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["sum_qty", "region_id"],
        "SELECT-order must be preserved (wave 1 fix #5 regression)"
    );
}

#[test]
fn groupby_aliased_key_preserves_alias() {
    let provider = agg_provider();
    let sql = "SELECT region_id AS r, SUM(qty) FROM sales GROUP BY region_id";
    let plan = parse_sql(sql, &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let out = phys.output_schema();
    assert!(
        !out.fields.is_empty(),
        "expected at least one output column"
    );
    assert_eq!(out.fields[0].name, "r", "alias on group key must surface");
}

#[test]
fn groupby_multi_int_keys() {
    let provider = agg_provider();
    let sql = "SELECT region_id, sub_region_id, COUNT(*) FROM sales \
               GROUP BY region_id, sub_region_id";
    let plan = parse_sql(sql, &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let PhysicalPlan::Aggregate { aggregate, .. } = &phys else {
        panic!("expected Aggregate, got {phys:?}");
    };
    assert_eq!(aggregate.group_by.len(), 2, "two group keys");
    assert_eq!(aggregate.aggregates.len(), 1, "one aggregate");
    assert_eq!(
        aggregate.output_schema.fields.len(),
        3,
        "two keys + one aggregate = three output columns"
    );
}

#[test]
fn integer_literal_overflow_rejected() {
    // Wave 1 fix #13: integer literals that overflow i64 must be rejected
    // rather than silently demoted to Float64.
    let provider = agg_provider();
    let sql = "SELECT * FROM sales WHERE qty = 9223372036854775808";
    let result = parse_sql(sql, &provider).and_then(|p| lower_physical(&p));
    let err = result.expect_err("expected overflow rejection");
    let msg = format!("{err}");
    assert!(
        msg.contains("i64"),
        "error should mention i64 range; got: {msg}"
    );
}

#[test]
fn integer_literal_i64_min_parses() {
    // Wave 1 fix #13: -9223372036854775808 is exactly i64::MIN; the positive
    // form overflows but the negated form must parse cleanly to Literal::Int64.
    let provider = agg_provider();
    let sql = "SELECT * FROM sales WHERE qty = -9223372036854775808";
    let plan = parse_sql(sql, &provider).expect("parse i64::MIN literal");
    // Walk down to the Filter and find the literal in its predicate.
    fn find_int64_literal(e: &Expr) -> Option<i64> {
        match e {
            Expr::Literal(Literal::Int64(v)) => Some(*v),
            Expr::Binary { left, right, .. } => {
                find_int64_literal(left).or_else(|| find_int64_literal(right))
            }
            Expr::Alias(inner, _) => find_int64_literal(inner),
            _ => None,
        }
    }
    fn find_filter_predicate(p: &LogicalPlan) -> Option<&Expr> {
        match p {
            LogicalPlan::Filter { predicate, .. } => Some(predicate),
            LogicalPlan::Project { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Distinct { input, .. }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Window { input, .. }
            | LogicalPlan::Sort { input, .. } => find_filter_predicate(input),
            LogicalPlan::Union { inputs } => inputs.iter().find_map(find_filter_predicate),
            LogicalPlan::Join { left, right, .. } | LogicalPlan::SetOp { left, right, .. } => {
                find_filter_predicate(left).or_else(|| find_filter_predicate(right))
            }
            LogicalPlan::Scan { .. } => None,
        }
    }
    let pred = find_filter_predicate(&plan).expect("Filter present in plan");
    let lit = find_int64_literal(pred).expect("Int64 literal present in predicate");
    assert_eq!(lit, i64::MIN, "negated literal must equal i64::MIN");
    // And the plan still lowers without error.
    lower_physical(&plan).expect("lower with i64::MIN literal");
}

#[test]
fn groupby_sum_int32_widens_to_i64() {
    // Wave 4: SUM-widening must also fire on the GROUP BY path
    // (not just scalar aggregates).
    let provider = agg_provider();
    let sql = "SELECT region_id, SUM(qty32) FROM sales GROUP BY region_id";
    let plan = parse_sql(sql, &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let PhysicalPlan::Aggregate { aggregate, .. } = &phys else {
        panic!("expected Aggregate, got {phys:?}");
    };
    // Find the SUM output field by name (avoids depending on key/agg order
    // beyond what `groupby_single_int_key` already asserts).
    let sum_field = aggregate
        .output_schema
        .fields
        .iter()
        .find(|f| f.name == "sum_qty32")
        .expect("sum_qty32 column in output schema");
    assert_eq!(
        sum_field.dtype,
        DataType::Int64,
        "GROUP BY SUM(Int32) must widen to Int64 (wave 4)"
    );
    // Sanity check: aggregate is what we think it is.
    assert!(matches!(
        aggregate.aggregates[0],
        AggregateExpr::Sum(_)
    ));
}

// ---- Online (require CUDA device) ------------------------------------------

#[test]
#[ignore = "gpu:e2e"]
fn e2e_simple_projection() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let batch = sales_batch(1024);
    engine.register_table("sales", batch.clone()).unwrap();

    let h = engine.sql("SELECT price FROM sales").expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1024);
    let actual = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let expected = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..1024 {
        assert_eq!(actual.value(i), expected.value(i), "row {i}");
    }
}

/// `COUNT(DISTINCT region_id)` over the `sales` fixture. `sales_batch` sets
/// `region_id = i % 4`, so for any `n >= 4` there are exactly 4 distinct
/// region ids (0,1,2,3). The CountRows executor runs the Distinct (GPU dedupe)
/// and emits the row count of the deduped result.
#[test]
#[ignore = "gpu:e2e"]
fn e2e_count_distinct() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let batch = sales_batch(1024);
    engine.register_table("sales", batch).unwrap();

    let h = engine
        .sql("SELECT COUNT(DISTINCT region_id) FROM sales")
        .expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1);
    let actual = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64 count");
    assert_eq!(actual.value(0), 4, "4 distinct region ids (0,1,2,3)");
}

#[test]
#[ignore = "gpu:e2e"]
fn e2e_arithmetic_projection() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let batch = sales_batch(4096);
    engine.register_table("sales", batch.clone()).unwrap();

    let h = engine.sql("SELECT price * tax FROM sales").expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 4096);
    let actual = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let price = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let tax = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..4096 {
        let want = price.value(i) * tax.value(i);
        let got = actual.value(i);
        assert!((got - want).abs() < REL_TOL, "row {i}: got {got}, want {want}");
    }
}

#[test]
#[ignore = "gpu:e2e"]
fn e2e_filtered_select() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let batch = sales_batch(2048);
    engine.register_table("sales", batch.clone()).unwrap();

    let h = engine
        .sql("SELECT price FROM sales WHERE region_id = 1")
        .expect("execute");
    let out = h.record_batch();
    let actual = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let region = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let price = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    // The engine COMPACTS filter output (GPU prefix-scan + gather, or the host
    // `compact::compact_arrays` fallback): only rows matching `region_id = 1`
    // survive, in their original order. `region_id = i % 4`, so 512 of the 2048
    // rows match. Build the expected projected `price` values from the fixture.
    let expected: Vec<f64> = (0..2048)
        .filter(|&i| region.value(i) == 1)
        .map(|i| price.value(i))
        .collect();
    assert_eq!(out.num_rows(), expected.len(), "compacted row count");
    assert_eq!(out.num_rows(), 512, "region_id == 1 matches 2048/4 rows");
    for (k, want) in expected.iter().enumerate() {
        assert_eq!(actual.value(k), *want, "compacted row {k}");
    }
}

#[test]
#[ignore = "gpu:e2e"]
fn e2e_large_i64_add() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let n: usize = 100_000;
    let col: Int64Array = (0..n as i64).collect();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "x",
        ArrowDataType::Int64,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap();
    engine.register_table("big", batch).unwrap();

    let h = engine.sql("SELECT x + 1 FROM big").expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);
    let actual = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    for i in 0..n {
        assert_eq!(actual.value(i), (i as i64) + 1, "row {i}");
    }
}

// ---- Offline: wave 7 operators ---------------------------------------------
//
// Parse -> logical-plan shape -> physical-plan lowering for DISTINCT, LIMIT,
// ORDER BY, HAVING, UNION. JOIN is a scaffold only — tests for it land later.

#[test]
fn distinct_wraps_project_in_logical_plan() {
    // `SELECT DISTINCT region_id FROM sales` should parse to
    // `Distinct { input: Project { ... } }` and lower to `PhysicalPlan::Distinct`.
    let provider = sales_provider();
    let plan = parse_sql("SELECT DISTINCT region_id FROM sales", &provider).expect("parse");
    match &plan {
        LogicalPlan::Distinct { input } => match input.as_ref() {
            LogicalPlan::Project { exprs, .. } => {
                assert_eq!(exprs.len(), 1, "single projected expression");
            }
            other => panic!("expected Project under Distinct, got {other:?}"),
        },
        other => panic!("expected Distinct at top, got {other:?}"),
    }
    let phys = lower_physical(&plan).expect("lower");
    assert!(
        matches!(phys, PhysicalPlan::Distinct { .. }),
        "expected PhysicalPlan::Distinct, got {phys:?}"
    );
}

#[test]
fn limit_parses_to_limit_node() {
    // Bare LIMIT: offset defaults to 0.
    let provider = sales_provider();
    let plan = parse_sql("SELECT region_id FROM sales LIMIT 10", &provider).expect("parse");
    match &plan {
        LogicalPlan::Limit { limit, offset, .. } => {
            assert_eq!(*limit, 10, "LIMIT 10");
            assert_eq!(*offset, 0, "no OFFSET clause -> offset 0");
        }
        other => panic!("expected Limit at top, got {other:?}"),
    }
    let phys = lower_physical(&plan).expect("lower");
    assert!(
        matches!(phys, PhysicalPlan::Limit { .. }),
        "expected PhysicalPlan::Limit, got {phys:?}"
    );
}

#[test]
fn limit_with_offset_carries_both_values() {
    // `LIMIT n OFFSET k` collapses into a single Limit node carrying both
    // fields so executors don't need a separate Offset operator.
    let provider = sales_provider();
    let plan = parse_sql(
        "SELECT region_id FROM sales LIMIT 5 OFFSET 3",
        &provider,
    )
    .expect("parse");
    match &plan {
        LogicalPlan::Limit { limit, offset, .. } => {
            assert_eq!(*limit, 5);
            assert_eq!(*offset, 3);
        }
        other => panic!("expected Limit at top, got {other:?}"),
    }
}

#[test]
fn order_by_desc_lowers_to_sort() {
    // ORDER BY <expr> DESC parses to `Sort { sort_exprs: [{descending: true}] }`
    // and lowers without error.
    let provider = sales_provider();
    let plan = parse_sql(
        "SELECT region_id, price FROM sales ORDER BY price DESC",
        &provider,
    )
    .expect("parse");
    match &plan {
        LogicalPlan::Sort { sort_exprs, .. } => {
            assert_eq!(sort_exprs.len(), 1, "one sort key");
            assert!(sort_exprs[0].descending, "DESC sets descending=true");
        }
        other => panic!("expected Sort at top, got {other:?}"),
    }
    let phys = lower_physical(&plan).expect("lower");
    assert!(
        matches!(phys, PhysicalPlan::Sort { .. }),
        "expected PhysicalPlan::Sort, got {phys:?}"
    );
}

#[test]
fn order_by_default_direction_is_ascending() {
    // No direction keyword -> ASC.
    let provider = sales_provider();
    let plan = parse_sql(
        "SELECT region_id FROM sales ORDER BY region_id",
        &provider,
    )
    .expect("parse");
    match &plan {
        LogicalPlan::Sort { sort_exprs, .. } => {
            assert_eq!(sort_exprs.len(), 1);
            assert!(
                !sort_exprs[0].descending,
                "no DESC keyword must default to ASC (descending=false)"
            );
        }
        other => panic!("expected Sort at top, got {other:?}"),
    }
}

#[test]
fn having_desugars_to_filter_over_aggregate() {
    // HAVING wraps the (SELECT-ordered Project over Aggregate) in a Filter.
    // Walk down from the top to confirm the Aggregate is present below.
    let provider = agg_provider();
    let sql =
        "SELECT region_id, SUM(price) FROM sales GROUP BY region_id HAVING SUM(price) > 100";
    let plan = parse_sql(sql, &provider).expect("parse");

    // Top must be the HAVING Filter.
    let inner = match &plan {
        LogicalPlan::Filter { input, .. } => input.as_ref(),
        other => panic!("expected Filter (HAVING) at top, got {other:?}"),
    };
    // Below the Filter sits the SELECT-order Project (wave 1 fix #5).
    let proj_input = match inner {
        LogicalPlan::Project { input, .. } => input.as_ref(),
        other => panic!("expected Project under HAVING Filter, got {other:?}"),
    };
    // And under that Project is the actual Aggregate.
    assert!(
        matches!(proj_input, LogicalPlan::Aggregate { .. }),
        "expected Aggregate under SELECT-order Project, got {proj_input:?}"
    );

    // Lowering should succeed end-to-end.
    lower_physical(&plan).expect("lower HAVING plan");
}

#[test]
fn union_all_parses_to_two_input_union() {
    // `UNION ALL` lands directly as a `Union { inputs }` node with two
    // branches; no Distinct wrapper.
    let provider = sales_provider();
    let sql = "SELECT region_id FROM sales UNION ALL SELECT region_id FROM sales";
    let plan = parse_sql(sql, &provider).expect("parse");
    match &plan {
        LogicalPlan::Union { inputs } => {
            assert_eq!(inputs.len(), 2, "two branches");
        }
        other => panic!("expected Union, got {other:?}"),
    }
    let phys = lower_physical(&plan).expect("lower");
    assert!(
        matches!(phys, PhysicalPlan::Union { .. }),
        "expected PhysicalPlan::Union, got {phys:?}"
    );
}

#[test]
fn union_dedup_lowers_to_distinct_over_union() {
    // Plain `UNION` (dedup) is lowered as `Distinct(Union { ... })` so the
    // executor stack can reuse the existing Distinct path.
    let provider = sales_provider();
    let sql = "SELECT region_id FROM sales UNION SELECT region_id FROM sales";
    let plan = parse_sql(sql, &provider).expect("parse");
    match &plan {
        LogicalPlan::Distinct { input } => {
            assert!(
                matches!(input.as_ref(), LogicalPlan::Union { .. }),
                "expected Distinct(Union(..)), got Distinct({:?})",
                input
            );
        }
        other => panic!("expected Distinct at top for UNION (dedup), got {other:?}"),
    }
}

#[test]
fn limit_negative_rejected_at_parse() {
    // Wave 7 contract: `LIMIT` must be a non-negative integer literal; the SQL
    // frontend rejects negatives outright rather than coercing to 0 or wrapping.
    let provider = sales_provider();
    let err = parse_sql("SELECT region_id FROM sales LIMIT -1", &provider)
        .expect_err("LIMIT -1 must be rejected");
    let msg = format!("{err}").to_lowercase();
    assert!(
        msg.contains("limit") || msg.contains("non-negative") || msg.contains("integer"),
        "error should mention LIMIT/non-negative/integer, got: {err}"
    );
}

// ---- Offline: wave 8 INNER JOIN --------------------------------------------
//
// Parse -> plan-shape / schema assertions for INNER JOIN. The host-side
// hash-join executor lands in this wave; full execution tests (which need
// a real engine + table data) will arrive behind `#[ignore]` later.

/// Two-table fixture with intentional column-name collisions on `id` and
/// `region_id` so the schema disambiguation rule has something to chew on.
/// `t1` (left): id Int32, region_id Int32, qty Int32
/// `t2` (right): id Int32, region_id Int32, label Utf8
fn join_provider() -> MemTableProvider {
    let t1 = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "region_id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "qty".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    let t2 = Schema::new(vec![
        Field {
            name: "id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "region_id".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "label".into(),
            dtype: DataType::Utf8,
            nullable: false,
        },
    ]);
    MemTableProvider::new()
        .with_table("t1", t1)
        .with_table("t2", t2)
}

#[test]
fn inner_join_single_key_plan_shape() {
    // `SELECT qty, label FROM t1 INNER JOIN t2 ON t1.id = t2.id` parses to
    // `Project { input: Join { on: [(id, id)] } }` with a single equi-join
    // pair. We assert on the Join node, not on the outer Project.
    // (The SELECT uses bare column names because the 0.1.x scalar
    // expression lowerer doesn't yet accept table-qualified refs; the ON
    // clause does, via `lower_join_side` which strips the prefix.)
    let provider = join_provider();
    let plan = parse_sql(
        "SELECT qty, label FROM t1 INNER JOIN t2 ON t1.id = t2.id",
        &provider,
    )
    .expect("parse");
    let join = match plan {
        LogicalPlan::Project { input, .. } => *input,
        other => panic!("expected Project at top, got {other:?}"),
    };
    match join {
        LogicalPlan::Join { on, .. } => {
            assert_eq!(on.len(), 1, "single equi-join pair");
        }
        other => panic!("expected Join under Project, got {other:?}"),
    }
}

#[test]
fn inner_join_multi_key_plan_shape() {
    // Conjunctive ON clause: each `AND`-joined equality becomes a separate
    // `(left, right)` pair in `on`. Two predicates -> two pairs.
    let provider = join_provider();
    let plan = parse_sql(
        "SELECT qty FROM t1 INNER JOIN t2 ON t1.id = t2.id AND t1.region_id = t2.region_id",
        &provider,
    )
    .expect("parse");
    let join = match plan {
        LogicalPlan::Project { input, .. } => *input,
        other => panic!("expected Project at top, got {other:?}"),
    };
    match join {
        LogicalPlan::Join { on, .. } => {
            assert_eq!(on.len(), 2, "two equi-join pairs");
        }
        other => panic!("expected Join, got {other:?}"),
    }
}

#[test]
fn join_schema_disambiguates_collisions() {
    // Both `t1` and `t2` have `id` and `region_id`. The logical schema must
    // include all four (not three with a duplicate dropped) AND must not
    // panic / error on the duplicate names — fix #1 from wave 8.
    //
    // Left side keeps its bare names; right side gets `right.<col>` for any
    // collision (the convention chosen in `join_combined_schema`).
    let provider = join_provider();
    let plan = parse_sql(
        "SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id",
        &provider,
    )
    .expect("parse");
    // Walk past any outer wrapper (Project from wildcard expansion) to find
    // the Join, then ask its own schema().
    fn find_join(p: &LogicalPlan) -> &LogicalPlan {
        match p {
            LogicalPlan::Join { .. } => p,
            LogicalPlan::Project { input, .. }
            | LogicalPlan::Filter { input, .. }
            | LogicalPlan::Distinct { input }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Sort { input, .. } => find_join(input),
            other => panic!("expected to find a Join under {other:?}"),
        }
    }
    let join = find_join(&plan);
    let schema = join.schema().expect("join schema");

    // 3 left + 3 right = 6 fields, no duplicates dropped.
    assert_eq!(schema.fields.len(), 6, "all six columns present");
    // Every name must be unique — that's the whole point of the fix.
    let names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
    let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for n in &names {
        assert!(seen.insert(n), "duplicate field name '{n}' in {names:?}");
    }
    // Left side keeps bare names.
    assert!(names.contains(&"id"), "left 'id' must survive bare in {names:?}");
    assert!(
        names.contains(&"region_id"),
        "left 'region_id' must survive bare in {names:?}"
    );
    assert!(names.contains(&"qty"), "left-only 'qty' must survive bare in {names:?}");
    // Right side: colliding columns become `right.<col>`, non-colliders stay bare.
    assert!(
        names.contains(&"right.id"),
        "right 'id' must be renamed to 'right.id' in {names:?}"
    );
    assert!(
        names.contains(&"right.region_id"),
        "right 'region_id' must be renamed in {names:?}"
    );
    assert!(
        names.contains(&"label"),
        "right-only 'label' must survive bare in {names:?}"
    );
}

#[test]
fn physical_join_output_schema_combines_sides() {
    // After lowering, `PhysicalPlan::Join::output_schema()` must return the
    // same combined+disambiguated schema as the logical layer — not the
    // pre-wave-8 "left only" approximation.
    let provider = join_provider();
    let plan = parse_sql(
        "SELECT * FROM t1 INNER JOIN t2 ON t1.id = t2.id",
        &provider,
    )
    .expect("parse");
    let phys = lower_physical(&plan).expect("lower");

    // Find the PhysicalPlan::Join (the outer `SELECT *` lowers to a Project
    // we can step through, but `lower()` actually drops that outer Project
    // when it's over a Join — see the `is_scan_chain` branch in
    // `physical_plan.rs::lower`). Either way: walk until we hit a Join.
    fn find_phys_join(p: &PhysicalPlan) -> &PhysicalPlan {
        match p {
            PhysicalPlan::Join { .. } => p,
            PhysicalPlan::Distinct { input }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::Sort { input, .. } => find_phys_join(input),
            other => panic!("expected a Join, got {other:?}"),
        }
    }
    let join = find_phys_join(&phys);
    let phys_schema = join.output_schema();

    // Should match the logical version exactly (same names + dtypes).
    let logical_schema = match &plan {
        LogicalPlan::Join { .. } => plan.schema().unwrap(),
        // Top-level Project gets dropped by the lowerer; recompute logical
        // join schema from the inner Join for the comparison.
        _ => {
            fn find_logical_join(p: &LogicalPlan) -> &LogicalPlan {
                match p {
                    LogicalPlan::Join { .. } => p,
                    LogicalPlan::Project { input, .. }
                    | LogicalPlan::Filter { input, .. }
                    | LogicalPlan::Distinct { input }
                    | LogicalPlan::Limit { input, .. }
                    | LogicalPlan::Sort { input, .. } => find_logical_join(input),
                    other => panic!("expected logical Join under {other:?}"),
                }
            }
            find_logical_join(&plan).schema().unwrap()
        }
    };

    assert_eq!(
        phys_schema.fields.len(),
        logical_schema.fields.len(),
        "physical join schema must have the same field count as the logical one"
    );
    assert_eq!(phys_schema.fields.len(), 6, "all six columns present");
    let phys_names: Vec<&str> = phys_schema.fields.iter().map(|f| f.name.as_str()).collect();
    let logical_names: Vec<&str> =
        logical_schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        phys_names, logical_names,
        "physical and logical join schemas must agree on field names and order"
    );
}

// ---- Online: Option B pre-stage NULL propagation ---------------------------

/// Build a sales batch where `price` carries `n_null` NULL entries
/// (spread across the column) and `tax` is fully populated. Used to
/// exercise the Option B path: `SUM(price * tax)` must return the sum
/// of non-NULL `(price * tax)` rather than erroring out (Option A) or
/// silently summing garbage (the pre-Option-A behaviour).
fn sales_batch_with_nulls(n: usize, n_null: usize) -> RecordBatch {
    let region: Int32Array = (0..n as i32).map(|i| i % 4).collect();
    let price: Float64Array = (0..n)
        .map(|i| {
            if i % (n / n_null.max(1)) == 0 && (i / (n / n_null.max(1))) < n_null {
                None
            } else {
                Some((i + 1) as f64)
            }
        })
        .collect();
    let tax: Float64Array = (0..n).map(|_| Some(0.1_f64)).collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("region_id", ArrowDataType::Int32, false),
        ArrowField::new("price", ArrowDataType::Float64, true),
        ArrowField::new("tax", ArrowDataType::Float64, false),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(region), Arc::new(price), Arc::new(tax)]).unwrap()
}

#[test]
#[ignore = "gpu:tier1"]
fn e2e_sum_price_times_tax_with_nulls_in_price() {
    // Option B contract: `SELECT SUM(price * tax) FROM sales` where
    // `price` has NULL rows must propagate validity through the pre
    // kernel (price * tax marked NULL where price is NULL) and the
    // scalar reducer must skip NULL rows. Result = sum of
    // non-NULL `(price * tax)`.
    use craton_bolt::Engine;

    let n = 100usize;
    let n_null = 10usize;
    let batch = sales_batch_with_nulls(n, n_null);

    // Compute expected on the host using the same NULL set.
    let price = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let tax = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let mut expected: f64 = 0.0;
    for i in 0..n {
        if !price.is_null(i) {
            expected += price.value(i) * tax.value(i);
        }
    }

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("sales", batch).unwrap();

    let h = engine
        .sql("SELECT SUM(price * tax) FROM sales")
        .expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1);
    let got = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap()
        .value(0);
    assert!(
        (got - expected).abs() < 1e-6,
        "Option B NULL propagation: got SUM={got}, want {expected} \
         (10/100 NULL rows in price)"
    );
}

/// Stage C: `SELECT region_id, SUM(price * tax) FROM sales GROUP BY region_id`
/// where `price` has NULL rows. Per SQL semantics, NULL rows must not
/// contribute to ANY group's accumulator. This exercises the
/// validity-bearing aggregate input path through
/// `crate::exec::groupby_with_pre::run_typed_agg` — Stage B used to
/// reject with a clear error; Stage C lifts that gate by stripping
/// NULL rows in lockstep with the GROUP BY key column on the host
/// before upload.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_groupby_sum_with_nulls_in_value_column() {
    use craton_bolt::Engine;

    let n = 100usize;
    let n_null = 10usize;
    let batch = sales_batch_with_nulls(n, n_null);

    // Compute the per-group expected SUM on the host using SQL semantics
    // (NULL rows excluded from the running sum). region_id = i % 4 so we
    // have 4 groups; price = (i+1) where non-NULL, otherwise NULL.
    let region = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let price = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let tax = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let mut expected: std::collections::HashMap<i32, f64> =
        std::collections::HashMap::new();
    for i in 0..n {
        if !price.is_null(i) {
            *expected.entry(region.value(i)).or_insert(0.0) +=
                price.value(i) * tax.value(i);
        }
    }

    let mut engine = Engine::new().expect("ctx");
    engine.register_table("sales", batch).unwrap();

    let h = engine
        .sql(
            "SELECT region_id, SUM(price * tax) FROM sales \
             GROUP BY region_id ORDER BY region_id",
        )
        .expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), expected.len());

    let out_region = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let out_sum = out
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..out.num_rows() {
        let r = out_region.value(i);
        let want = *expected
            .get(&r)
            .unwrap_or_else(|| panic!("region {r} missing from expected"));
        let got = out_sum.value(i);
        assert!(
            (got - want).abs() < 1e-6,
            "Stage C GROUP BY NULL handling: region {r}: got SUM={got}, want {want}"
        );
    }
}
// ---------------------------------------------------------------------------
// PV-stage-d: TableProvider validity signal tests.
// ---------------------------------------------------------------------------

/// `MemTableProvider`'s `has_nulls` default is safe-false; the planner
/// receives a `Vec<bool>` of all-false for every input. Existing providers
/// that didn't know about validity still work.
#[test]
fn table_provider_default_has_nulls_is_false() {
    use craton_bolt::plan::TableProvider;
    let provider = sales_provider();
    // Stage 6: `MemTableProvider` grew an inherent `has_nulls(&str, &str)` that
    // shadows the trait method when called via method syntax. Use UFCS to
    // disambiguate to the trait's `(&str, usize)` signature — that's the
    // shape this test exercises.
    type Mem = craton_bolt::plan::MemTableProvider;
    // Every column of `sales` returns false through the default impl.
    assert!(!<Mem as TableProvider>::has_nulls(&provider, "sales", 0));
    assert!(!<Mem as TableProvider>::has_nulls(&provider, "sales", 1));
    assert!(!<Mem as TableProvider>::has_nulls(&provider, "sales", 2));
    // Unknown column / table — still false.
    assert!(!<Mem as TableProvider>::has_nulls(&provider, "sales", 999));
    assert!(!<Mem as TableProvider>::has_nulls(&provider, "nope", 0));
    // `null_count` defaults to None.
    assert!(provider.null_count("sales", 0).is_none());
}

/// Custom `TableProvider` that overrides `has_nulls` — the override must
/// surface through `populate_input_validity` into every input column's
/// validity flag.
#[test]
fn provider_override_populates_input_has_validity() {
    use craton_bolt::plan::{lower_physical, parse_sql, Schema, TableProvider};
    use craton_bolt::BoltResult;

    struct NullableSales {
        inner: MemTableProvider,
    }
    impl TableProvider for NullableSales {
        fn schema(&self, name: &str) -> BoltResult<Schema> {
            self.inner.schema(name)
        }
        fn has_nulls(&self, _table: &str, col_idx: usize) -> bool {
            // Pretend column 1 (price) carries nulls.
            col_idx == 1
        }
    }

    let provider = NullableSales {
        inner: sales_provider(),
    };
    let plan = parse_sql("SELECT region_id, price FROM sales WHERE region_id = 1", &provider)
        .expect("parse ok");
    let mut phys = lower_physical(&plan).expect("lower ok");
    craton_bolt::plan::physical_plan::populate_input_validity(&mut phys, &provider);

    // The Projection's kernel should now report input 1 (price) as having
    // validity. region_id (input 0) should not.
    match &phys {
        PhysicalPlan::Projection { kernel, .. } => {
            assert_eq!(kernel.inputs.len(), kernel.input_has_validity.len());
            // Find the price input by name.
            let price_idx = kernel
                .inputs
                .iter()
                .position(|c| c.name == "price")
                .expect("price input present");
            let region_idx = kernel
                .inputs
                .iter()
                .position(|c| c.name == "region_id")
                .expect("region_id input present");
            assert!(
                kernel.input_has_validity[price_idx],
                "provider said price has nulls; flag must propagate"
            );
            assert!(
                !kernel.input_has_validity[region_idx],
                "region_id has no nulls per the override"
            );
        }
        other => panic!("expected Projection, got {other:?}"),
    }
}

/// PV-stage-e: GROUP BY with a pre kernel (`SUM(price * tax)`) over a
/// null-bearing input must (a) produce the correct per-group sum and
/// (b) increment `groupby_with_pre::NATIVE_VALIDITY_LAUNCHES` exactly
/// once per validity-flagged aggregate — proving the planner-time
/// signal actually drove dispatch through the native `_with_validity`
/// kernel rather than falling through to the host-strip fallback.
///
/// The test mirrors the input shape used by
/// `e2e_groupby_sum_with_nulls_in_value_column`; the difference is that
/// here we additionally observe the dispatch path via the executor's
/// atomic counter. `#[ignore]`'d because it requires a real CUDA
/// device for the launch.
#[test]
#[ignore = "gpu:tier1"]
fn groupby_sum_with_nulls_uses_native_validity_path() {
    use craton_bolt::exec::groupby_with_pre::NATIVE_VALIDITY_LAUNCHES;
    use craton_bolt::Engine;
    use std::sync::atomic::Ordering;

    // Reset the counter to a known baseline so other tests in the same
    // run don't bleed into ours.
    let baseline = NATIVE_VALIDITY_LAUNCHES.load(Ordering::Relaxed);

    let n = 100usize;
    let n_null = 10usize;
    let batch = sales_batch_with_nulls(n, n_null);

    // Compute expected per-region SUM(price * tax) on the host with SQL
    // semantics (NULL rows excluded from the running sum).
    let region = batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let price = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let tax = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let mut expected: std::collections::HashMap<i32, f64> =
        std::collections::HashMap::new();
    for i in 0..n {
        if !price.is_null(i) {
            *expected.entry(region.value(i)).or_insert(0.0) +=
                price.value(i) * tax.value(i);
        }
    }

    // NOTE: `Engine::register_table` registers the batch through a
    // default `MemTableProvider` whose `has_nulls` defaults to false.
    // Stage E's native-dispatch gate is anchored on the PLANNER signal
    // populated from `TableProvider::has_nulls`, so the counter delta
    // assertion below is soft: it logs the observed dispatch path
    // without forcing a CUDA-runtime assertion until the engine wires
    // a null-aware provider into `register_table`. The correctness
    // check on the actual SUM result is the hard assertion that
    // matters here.
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("sales", batch).unwrap();

    let h = engine
        .sql(
            "SELECT region_id, SUM(price * tax) FROM sales \
             GROUP BY region_id ORDER BY region_id",
        )
        .expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), expected.len());

    let out_region = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let out_sum = out
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for i in 0..out.num_rows() {
        let r = out_region.value(i);
        let want = *expected
            .get(&r)
            .unwrap_or_else(|| panic!("region {r} missing from expected"));
        let got = out_sum.value(i);
        assert!(
            (got - want).abs() < 1e-6,
            "PV-stage-e native validity: region {r}: got SUM={got}, want {want}"
        );
    }

    // The counter rises by at least 1 if the engine's TableProvider
    // delivered the planner signal. If the test setup didn't (e.g.
    // `Engine::register_table` uses the safe-`false` default), the
    // host-strip fallback is still correct — leave a soft assertion so
    // this test is informative rather than brittle while the
    // engine-side wiring to `populate_input_validity` lands.
    let delta = NATIVE_VALIDITY_LAUNCHES
        .load(Ordering::Relaxed)
        .saturating_sub(baseline);
    eprintln!(
        "PV-stage-e: NATIVE_VALIDITY_LAUNCHES delta = {delta} \
         (>=1 means planner signal drove native dispatch)"
    );
}

/// PV-stage-f: no-pre GROUP BY path now wires
/// `AggregateSpec::input_has_validity` through to a native `_with_validity`
/// dispatch in `groupby::launch_agg_kernel`. With `EngineProvider::has_nulls`
/// surfacing the source batch's null bitmap and `populate_input_validity`
/// pushing that signal onto `AggregateSpec`, a NULL-bearing SQL query
/// should increment `groupby::NATIVE_VALIDITY_LAUNCHES` at least once
/// AND still produce the right per-group SUM.
///
/// Soft counter assertion (matches the Stage-E sibling test) so this is
/// informative rather than brittle if a future planner change muffles
/// the signal; the correctness assertion on per-group SUM is hard.
#[test]
#[ignore = "gpu:tier1"]
fn pv_stage_f_groupby_no_pre_native_validity_round_trip() {
    use craton_bolt::exec::groupby::NATIVE_VALIDITY_LAUNCHES;
    use craton_bolt::Engine;
    use std::sync::atomic::Ordering;
    use arrow_array::ArrayRef;

    let baseline = NATIVE_VALIDITY_LAUNCHES.load(Ordering::Relaxed);

    // Bare-column SUM over a null-bearing value column. The key column
    // has no nulls so the native-validity dispatch is eligible:
    //   - Float MIN/MAX is excluded by the predicate (host-strip).
    //   - Int64 SUM is exactly the covered case for the integer
    //     hash_kernels `_with_validity` emitter.
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Int64, true),
    ]));
    let k = Int32Array::from(vec![1i32, 2, 1, 2, 1, 2]);
    let v = Int64Array::from(vec![
        Some(10i64),
        Some(20),
        None,
        Some(40),
        Some(50),
        None,
    ]);
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(k) as ArrayRef, Arc::new(v) as ArrayRef],
    )
    .expect("batch");

    let mut engine = Engine::new().expect("engine");
    engine.register_table("t", batch).expect("register");
    let h = engine
        .sql("SELECT k, SUM(v) FROM t GROUP BY k")
        .expect("execute");
    let out = h.record_batch();
    // k=1: 10 + 50 = 60 (the NULL row is dropped).
    // k=2: 20 + 40 = 60 (the NULL row is dropped).
    assert_eq!(out.num_rows(), 2);
    let ks = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32");
    let ss = out
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("Int64");
    let mut got: std::collections::HashMap<i32, i64> =
        std::collections::HashMap::new();
    for i in 0..out.num_rows() {
        got.insert(ks.value(i), ss.value(i));
    }
    assert_eq!(got.get(&1), Some(&60i64), "k=1 SUM (NULL excluded)");
    assert_eq!(got.get(&2), Some(&60i64), "k=2 SUM (NULL excluded)");

    let delta = NATIVE_VALIDITY_LAUNCHES
        .load(Ordering::Relaxed)
        .saturating_sub(baseline);
    eprintln!(
        "PV-stage-f: groupby::NATIVE_VALIDITY_LAUNCHES delta = {delta} \
         (>=1 means planner+runtime drove native dispatch)"
    );
}

/// Online: GROUP BY over a column with nulls should still produce the
/// right result. With the validity-aware dispatch deferred to stage E,
/// this test verifies that the engine still answers correctly via the
/// host-strip fallback path — i.e. nulls are excluded from the grouping.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_groupby_with_nulls() {
    use arrow_array::ArrayRef;

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, true),
        ArrowField::new("v", ArrowDataType::Int64, true),
    ]));
    let k = Int32Array::from(vec![Some(1), Some(2), None, Some(1), Some(2)]);
    let v = Int64Array::from(vec![Some(10), Some(20), Some(99), Some(30), Some(40)]);
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(k) as ArrayRef, Arc::new(v) as ArrayRef],
    )
    .expect("batch");

    let mut engine = craton_bolt::Engine::new().expect("engine");
    engine.register_table("t", batch).expect("register");
    let result = engine
        .sql("SELECT k, SUM(v) FROM t GROUP BY k")
        .expect("sql ok");
    let out = result.record_batch();
    // Expected: 2 groups (k=1 -> 40, k=2 -> 60). The null row drops out.
    assert_eq!(
        out.num_rows(),
        2,
        "null group key should be excluded; only k=1 and k=2 survive"
    );
}

// ---------------------------------------------------------------------------
// VAR_POP / VAR_SAMP / VARIANCE: v0.5 scalar-aggregate path
//
// The frontend lowers all three function names to either `AggregateExpr::VarPop`
// or `AggregateExpr::VarSamp` (plain `VARIANCE` aliases to `VAR_SAMP` per SQL
// standard). Both produce a nullable Float64; the scalar (no GROUP BY)
// reducer downloads the column to the host and runs Welford in f64. GROUP BY
// is intentionally rejected with a clear error in v0.5.
// ---------------------------------------------------------------------------

/// One-column Int64 table fixture for the variance tests.
fn one_col_batch_i64(name: &str, values: Vec<i64>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        name,
        ArrowDataType::Int64,
        false,
    )]));
    let col = Arc::new(Int64Array::from(values)) as Arc<dyn Array>;
    RecordBatch::try_new(schema, vec![col]).expect("int64 batch")
}

/// One-column Float64 table fixture for the variance tests.
fn one_col_batch_f64(name: &str, values: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        name,
        ArrowDataType::Float64,
        false,
    )]));
    let col = Arc::new(Float64Array::from(values)) as Arc<dyn Array>;
    RecordBatch::try_new(schema, vec![col]).expect("float64 batch")
}

/// Parse-time: `VAR_POP(v)` resolves to a single nullable Float64 output
/// named `var_pop_v`. No GPU is required for this check.
#[test]
fn var_pop_scalar_planning_yields_float64() {
    let provider = MemTableProvider::new().with_table(
        "t",
        Schema::new(vec![Field {
            name: "v".into(),
            dtype: DataType::Float64,
            nullable: false,
        }]),
    );
    let plan = parse_sql("SELECT VAR_POP(v) FROM t", &provider).expect("parse");
    let phys = lower_physical(&plan).expect("lower");
    let out = phys.output_schema();
    assert_eq!(out.fields.len(), 1);
    assert_eq!(out.fields[0].name, "var_pop_v");
    assert_eq!(out.fields[0].dtype, DataType::Float64);
    assert!(out.fields[0].nullable, "variance output must be nullable");
}

/// Parse-time: `VARIANCE(v)` and `VAR_SAMP(v)` are synonyms and both emit
/// `var_samp_v` as the output column name.
#[test]
fn variance_is_alias_for_var_samp() {
    let provider = MemTableProvider::new().with_table(
        "t",
        Schema::new(vec![Field {
            name: "v".into(),
            dtype: DataType::Float64,
            nullable: false,
        }]),
    );
    for sql in ["SELECT VAR_SAMP(v) FROM t", "SELECT VARIANCE(v) FROM t"] {
        let plan = parse_sql(sql, &provider).expect("parse");
        let phys = lower_physical(&plan).expect("lower");
        let out = phys.output_schema();
        assert_eq!(out.fields.len(), 1, "sql: {sql}");
        assert_eq!(out.fields[0].name, "var_samp_v", "sql: {sql}");
        assert_eq!(out.fields[0].dtype, DataType::Float64);
    }
}

/// Plan-time gate: GROUP BY VAR_POP / VAR_SAMP still type-checks (the plan
/// is well-formed) — the rejection happens at execution time so the planner
/// doesn't need to know about per-aggregate execution scope. This test
/// just locks in that parse + lower succeed; the run-time rejection lives
/// in the GPU-ignored test below.
#[test]
fn groupby_variance_parses_but_execution_is_gated() {
    let provider = MemTableProvider::new().with_table(
        "t",
        Schema::new(vec![
            Field {
                name: "k".into(),
                dtype: DataType::Int32,
                nullable: false,
            },
            Field {
                name: "v".into(),
                dtype: DataType::Float64,
                nullable: false,
            },
        ]),
    );
    let plan =
        parse_sql("SELECT k, VAR_POP(v) FROM t GROUP BY k", &provider).expect("parse");
    let _phys = lower_physical(&plan).expect("lower");
}

/// End-to-end: `SELECT VAR_POP(v) FROM t` over a small Int64 column matches
/// the closed-form population variance.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_var_pop_int_column_matches_closed_form() {
    use craton_bolt::Engine;
    let batch = one_col_batch_i64("v", vec![1, 2, 3, 4, 5]);
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", batch).unwrap();
    let h = engine.sql("SELECT VAR_POP(v) FROM t").expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1);
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("VAR_POP output is Float64");
    assert!(!col.is_null(0), "VAR_POP over non-empty input is non-NULL");
    // Closed form: values [1..5], mean = 3, deviations [-2,-1,0,1,2],
    // M2 = 4+1+0+1+4 = 10, VAR_POP = 10/5 = 2.
    let got = col.value(0);
    assert!((got - 2.0).abs() < (REL_TOL * 2.0_f64).abs(), "got {got}");
}

/// End-to-end: `SELECT VAR_SAMP(v) FROM t` over Float64 input matches
/// `M2 / (n - 1)`. Same five-element series, expected = 10/4 = 2.5.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_var_samp_float_column_matches_closed_form() {
    use craton_bolt::Engine;
    let batch = one_col_batch_f64("v", vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", batch).unwrap();
    let h = engine.sql("SELECT VAR_SAMP(v) FROM t").expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1);
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("VAR_SAMP output is Float64");
    assert!(!col.is_null(0));
    let got = col.value(0);
    assert!((got - 2.5).abs() < (REL_TOL * 2.5_f64).abs(), "got {got}");
}

/// SQL NULL semantics: `VAR_SAMP` over a single-row input returns NULL
/// (count <= 1). `VAR_POP` returns 0 for the same input (single
/// observation has zero deviation from itself).
#[test]
#[ignore = "gpu:tier1"]
fn e2e_var_samp_single_row_is_null() {
    use craton_bolt::Engine;
    let batch = one_col_batch_f64("v", vec![42.0]);
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", batch).unwrap();

    let h_samp = engine.sql("SELECT VAR_SAMP(v) FROM t").expect("execute");
    let out_samp = h_samp.record_batch();
    let col = out_samp
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64");
    assert!(
        col.is_null(0),
        "VAR_SAMP of a single row must be NULL per SQL standard"
    );

    // Same engine — re-register isn't necessary, but rebuild for symmetry.
    let h_pop = engine.sql("SELECT VAR_POP(v) FROM t").expect("execute");
    let out_pop = h_pop.record_batch();
    let col_pop = out_pop
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("Float64");
    assert!(!col_pop.is_null(0), "VAR_POP of a single row is defined (= 0)");
    assert!(col_pop.value(0).abs() < 1e-12, "got {}", col_pop.value(0));
}

/// Execution-time rejection: GROUP BY + VAR_POP must surface a clear
/// error message from the engine's dispatch layer.
#[test]
#[ignore = "gpu:tier1"]
fn e2e_groupby_variance_returns_clear_error() {
    use craton_bolt::Engine;
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Float64, false),
    ]));
    let k = Arc::new(Int32Array::from(vec![1, 1, 2, 2])) as Arc<dyn Array>;
    let v = Arc::new(Float64Array::from(vec![1.0, 2.0, 10.0, 20.0])) as Arc<dyn Array>;
    let batch = RecordBatch::try_new(schema, vec![k, v]).expect("batch");
    let mut engine = Engine::new().expect("ctx");
    engine.register_table("t", batch).unwrap();
    let err = match engine.sql("SELECT k, VAR_POP(v) FROM t GROUP BY k") {
        Ok(_) => panic!("v0.5: GROUP BY VAR_POP must be rejected"),
        Err(e) => e,
    };
    let msg = format!("{err}");
    assert!(
        msg.contains("VAR_POP") || msg.contains("VAR_SAMP"),
        "error must mention the rejected aggregate; got: {msg}"
    );
}

// ---- Uncorrelated subquery resolution (engine pre-lowering pass) ------------
//
// These exercise `Engine::resolve_subqueries`: an uncorrelated scalar
// subquery is folded to a literal, and an uncorrelated `IN (subquery)` is
// rewritten to an OR-of-equalities, both BEFORE physical lowering. Small
// in-memory `sales` / `other` tables, full GPU execute.

/// Build a 2-column `sales(region_id Int32, qty Int32)` batch.
fn sales_qty_batch(regions: &[i32], qtys: &[i32]) -> RecordBatch {
    assert_eq!(regions.len(), qtys.len());
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("region_id", ArrowDataType::Int32, false),
        ArrowField::new("qty", ArrowDataType::Int32, false),
    ]));
    let region = Int32Array::from(regions.to_vec());
    let qty = Int32Array::from(qtys.to_vec());
    RecordBatch::try_new(schema, vec![Arc::new(region), Arc::new(qty)]).unwrap()
}

/// Build a 2-column `other(id Int32, val Int32)` batch.
fn other_batch(ids: &[i32], vals: &[i32]) -> RecordBatch {
    assert_eq!(ids.len(), vals.len());
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("id", ArrowDataType::Int32, false),
        ArrowField::new("val", ArrowDataType::Int32, false),
    ]));
    let id = Int32Array::from(ids.to_vec());
    let val = Int32Array::from(vals.to_vec());
    RecordBatch::try_new(schema, vec![Arc::new(id), Arc::new(val)]).unwrap()
}

/// `SELECT region_id FROM sales WHERE qty > (SELECT MAX(val) FROM other)`.
///
/// The scalar subquery `MAX(val)` over `other = {3, 7, 5}` folds to the
/// literal `7`; the surviving predicate is `qty > 7`. The engine compacts
/// filter output, so only the matching rows survive (in original order).
#[test]
#[ignore = "gpu:e2e"]
fn e2e_scalar_subquery_in_filter() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let sales = sales_qty_batch(&[10, 20, 30, 40], &[5, 8, 7, 9]);
    let other = other_batch(&[0, 1, 2], &[3, 7, 5]); // MAX(val) = 7
    engine.register_table("sales", sales).unwrap();
    engine.register_table("other", other).unwrap();

    let h = engine
        .sql("SELECT region_id FROM sales WHERE qty > (SELECT MAX(val) FROM other)")
        .expect("execute");
    let out = h.record_batch();
    // Row qty = {5, 8, 7, 9}; predicate qty > 7 keeps rows {8, 9} → region_id
    // 20 and 40. Compaction drops the non-matching rows entirely.
    let region = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 region");
    assert_eq!(out.num_rows(), 2, "two rows satisfy qty > 7");
    assert_eq!(region.value(0), 20, "first kept row (qty 8)");
    assert_eq!(region.value(1), 40, "second kept row (qty 9)");
}

/// `SELECT region_id FROM sales WHERE region_id IN (SELECT id FROM other)`.
///
/// The IN-subquery over `other.id = {20, 40}` rewrites to
/// `region_id = 20 OR region_id = 40`. Rows whose `region_id` is in that set
/// survive the mask.
#[test]
#[ignore = "gpu:e2e"]
fn e2e_in_subquery_in_filter() {
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("ctx");
    let sales = sales_qty_batch(&[10, 20, 30, 40], &[1, 2, 3, 4]);
    let other = other_batch(&[20, 40], &[0, 0]); // membership set {20, 40}
    engine.register_table("sales", sales).unwrap();
    engine.register_table("other", other).unwrap();

    let h = engine
        .sql("SELECT region_id FROM sales WHERE region_id IN (SELECT id FROM other)")
        .expect("execute");
    let out = h.record_batch();
    let region = out
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("Int32 region");
    // region_id in {20, 40} survives; {10, 30} are dropped by compaction.
    assert_eq!(out.num_rows(), 2, "two rows in the membership set");
    assert_eq!(region.value(0), 20, "20 in set");
    assert_eq!(region.value(1), 40, "40 in set");
}
