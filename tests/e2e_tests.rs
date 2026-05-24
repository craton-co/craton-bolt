// SPDX-License-Identifier: Apache-2.0

//! End-to-end query tests for Javelin.
//!
//! Offline tests (no GPU): SQL parse -> plan -> PTX shape. Run by `cargo test`.
//! Online tests (#[ignore]'d): full engine execute on a CUDA device. Run with
//! `cargo test -- --ignored` on a GPU host.

use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use javelin::jit::compile_ptx;
use javelin::plan::{
    lower_physical, parse_sql, DataType, Field, MemTableProvider, PhysicalPlan, Schema,
};

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
    let ptx = compile_ptx(kernel, "javelin_kernel").expect("ptx");

    assert!(ptx.contains(".version 7.5"), "missing .version directive\n{ptx}");
    assert!(ptx.contains(".target sm_70"), "missing .target directive\n{ptx}");
    assert!(
        ptx.contains(".address_size 64"),
        "missing .address_size directive"
    );
    assert!(
        ptx.contains(".visible .entry javelin_kernel"),
        "missing entry directive"
    );
    // Kernel-parameter load order is established by `ld.param.u64` (column ptrs) and
    // a trailing `ld.param.u32` (row count) — this is the order cuLaunchKernel sees.
    assert!(ptx.contains("ld.param.u64"), "expected ld.param.u64 for column ptrs");
    assert!(ptx.contains("ld.param.u32"), "expected ld.param.u32 for n_rows");
    // f64 load + store, since `price` is Float64.
    assert!(ptx.contains("ld.global.f64"), "missing f64 load");
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
    let ptx = compile_ptx(kernel, "javelin_kernel").unwrap();

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
    let ptx = compile_ptx(kernel, "javelin_kernel").unwrap();
    assert!(ptx.contains("ld.global.s32"), "expected s32 load");
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
    let ptx = compile_ptx(kernel, "javelin_kernel").unwrap();
    assert!(ptx.contains("ld.global.s64"), "expected s64 load");
    assert!(ptx.contains("st.global.s64"), "expected s64 store");
}

// ---- Online (require CUDA device) ------------------------------------------

#[test]
#[ignore = "requires CUDA device - run with `cargo test -- --ignored`"]
fn e2e_simple_projection() {
    use javelin::Engine;

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

#[test]
#[ignore = "requires CUDA device"]
fn e2e_arithmetic_projection() {
    use javelin::Engine;

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
        assert!((got - want).abs() < 1e-9, "row {i}: got {got}, want {want}");
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn e2e_filtered_select() {
    use javelin::Engine;

    let mut engine = Engine::new().expect("ctx");
    let batch = sales_batch(2048);
    engine.register_table("sales", batch.clone()).unwrap();

    let h = engine
        .sql("SELECT price FROM sales WHERE region_id = 1")
        .expect("execute");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 2048);
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
    // The engine doesn't compact: masked positions remain at the zero-init value (0.0);
    // unmasked positions hold the projected `price`.
    for i in 0..2048 {
        if region.value(i) == 1 {
            assert_eq!(actual.value(i), price.value(i), "unmasked row {i}");
        } else {
            assert_eq!(actual.value(i), 0.0, "masked row {i}");
        }
    }
}

#[test]
#[ignore = "requires CUDA device"]
fn e2e_large_i64_add() {
    use javelin::Engine;

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
