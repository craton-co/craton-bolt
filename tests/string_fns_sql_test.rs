// SPDX-License-Identifier: Apache-2.0

//! Parser + type-check tests for the v0.5 SQL string scalar functions:
//! `UPPER`, `LOWER`, `LENGTH`, `SUBSTRING`, `CONCAT`.
//!
//! Scope: the SQL frontend recognises these calls and lowers them into
//! `Expr::ScalarFn`; the logical plan's type-checker validates their argument
//! shapes. Lowering then routes each to its execution path: LENGTH →
//! `StringLength` (GPU), UPPER/LOWER → `StringProject` (GPU), and
//! SUBSTRING/TRIM → the host-side `PhysicalPlan::Project`
//! (`expr_agg::eval_expr`). CONCAT and any not-yet-wired shape are still
//! rejected at the physical-plan boundary with a `Plan` error.
//!
//! These tests pin:
//!   * Successful parse+lower of each function (logical-plan shape only —
//!     we do NOT round-trip through `lower_physical` for the positive
//!     paths because that boundary intentionally rejects the call).
//!   * Type errors for wrong argument dtypes / arity.
//!   * The `lower_physical` rejection error message (a single
//!     representative case is enough — the rejection lives in one place).

use craton_bolt::exec::string_project::StringTransform;
use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Expr, Field, LogicalPlan, MemTableProvider, PhysicalPlan,
    ScalarFnKind, Schema, StringLengthOutput, StringProjectOutput,
};

// ---- Fixture ----------------------------------------------------------------

/// A small `txt` table with one Utf8 column and one Int32 column. Lets us
/// test both the happy path (UPPER/LOWER/etc. on a Utf8 column) and the
/// type-error path (the same function on a non-Utf8 column).
fn fixture() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field {
            name: "s".into(),
            dtype: DataType::Utf8,
            nullable: false,
        },
        Field {
            name: "n".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
    ]);
    MemTableProvider::new().with_table("txt", schema)
}

// ---- Helpers ----------------------------------------------------------------

/// Parse `sql` against `fixture()` and return the resulting plan or a
/// stringified parse error.
fn parse(sql: &str) -> Result<LogicalPlan, String> {
    parse_sql(sql, &fixture()).map_err(|e| format!("{e}"))
}

/// Assert `result` is `Err` and its message contains `needle` (case-insensitive).
fn assert_err_contains<T: std::fmt::Debug>(result: Result<T, String>, needle: &str) {
    match result {
        Ok(v) => panic!("expected error containing {needle:?}, got Ok({v:?})"),
        Err(msg) => assert!(
            msg.to_ascii_lowercase().contains(&needle.to_ascii_lowercase()),
            "expected error to contain {needle:?}, got: {msg}"
        ),
    }
}

/// Walk the plan to the first projected expression and return it. The
/// fixture queries are always of the shape `SELECT <expr> FROM txt`, so
/// the top of the tree is a `Project { exprs: [<expr>] }`.
fn first_project_expr(plan: &LogicalPlan) -> &Expr {
    match plan {
        LogicalPlan::Project { exprs, .. } => {
            assert!(!exprs.is_empty(), "expected at least one SELECT expr");
            &exprs[0]
        }
        other => panic!("expected Project at top, got {other:?}"),
    }
}

/// Peel `Alias` wrappers off `e`, returning the innermost non-Alias expr.
fn strip_alias(e: &Expr) -> &Expr {
    let mut cur = e;
    while let Expr::Alias(inner, _) = cur {
        cur = inner;
    }
    cur
}

// ---- Positive: each function parses + lowers to the right ScalarFnKind ------

#[test]
fn upper_parses_to_scalarfn_upper() {
    let plan = parse("SELECT UPPER(s) FROM txt").expect("UPPER on Utf8 should parse");
    let top = first_project_expr(&plan);
    match strip_alias(top) {
        Expr::ScalarFn { kind, args } => {
            assert_eq!(*kind, ScalarFnKind::Upper);
            assert_eq!(args.len(), 1);
            assert!(matches!(args[0], Expr::Column(ref n) if n == "s"));
        }
        other => panic!("expected ScalarFn(Upper), got {other:?}"),
    }
    // The logical plan must type-check (output schema resolved without error).
    plan.schema().expect("UPPER(s) must type-check");
}

#[test]
fn lower_parses_to_scalarfn_lower() {
    let plan = parse("SELECT LOWER(s) FROM txt").expect("LOWER on Utf8 should parse");
    let top = first_project_expr(&plan);
    match strip_alias(top) {
        Expr::ScalarFn { kind, args } => {
            assert_eq!(*kind, ScalarFnKind::Lower);
            assert_eq!(args.len(), 1);
        }
        other => panic!("expected ScalarFn(Lower), got {other:?}"),
    }
    plan.schema().expect("LOWER(s) must type-check");
}

#[test]
fn length_parses_to_scalarfn_length_with_int64_output() {
    let plan = parse("SELECT LENGTH(s) FROM txt").expect("LENGTH on Utf8 should parse");
    let top = first_project_expr(&plan);
    match strip_alias(top) {
        Expr::ScalarFn { kind, args } => {
            assert_eq!(*kind, ScalarFnKind::Length);
            assert_eq!(args.len(), 1);
        }
        other => panic!("expected ScalarFn(Length), got {other:?}"),
    }
    // LENGTH(Utf8) -> Int64 (the rule that distinguishes it from UPPER/LOWER).
    let schema = plan.schema().expect("LENGTH(s) must type-check");
    assert_eq!(schema.fields.len(), 1);
    assert_eq!(schema.fields[0].dtype, DataType::Int64);
}

#[test]
fn substring_from_for_parses_to_scalarfn_substring_with_three_args() {
    let plan = parse("SELECT SUBSTRING(s FROM 1 FOR 3) FROM txt")
        .expect("SUBSTRING ... FROM ... FOR ... should parse");
    let top = first_project_expr(&plan);
    match strip_alias(top) {
        Expr::ScalarFn { kind, args } => {
            assert_eq!(*kind, ScalarFnKind::Substring);
            assert_eq!(args.len(), 3, "expected three args (s, start, length)");
        }
        other => panic!("expected ScalarFn(Substring), got {other:?}"),
    }
    plan.schema().expect("SUBSTRING(s FROM 1 FOR 3) must type-check");
}

#[test]
fn substring_comma_syntax_parses_to_scalarfn_substring() {
    let plan = parse("SELECT SUBSTRING(s, 1, 3) FROM txt")
        .expect("SUBSTRING(s, 1, 3) should parse");
    let top = first_project_expr(&plan);
    match strip_alias(top) {
        Expr::ScalarFn { kind, args } => {
            assert_eq!(*kind, ScalarFnKind::Substring);
            assert_eq!(args.len(), 3);
        }
        other => panic!("expected ScalarFn(Substring), got {other:?}"),
    }
    plan.schema().expect("SUBSTRING(s, 1, 3) must type-check");
}

#[test]
fn substring_two_args_parses_with_optional_length_omitted() {
    let plan = parse("SELECT SUBSTRING(s FROM 2) FROM txt")
        .expect("SUBSTRING(s FROM 2) should parse (length is optional)");
    let top = first_project_expr(&plan);
    match strip_alias(top) {
        Expr::ScalarFn { kind, args } => {
            assert_eq!(*kind, ScalarFnKind::Substring);
            assert_eq!(args.len(), 2);
        }
        other => panic!("expected ScalarFn(Substring), got {other:?}"),
    }
    plan.schema().expect("SUBSTRING(s, start) must type-check");
}

#[test]
fn concat_two_args_parses_to_scalarfn_concat() {
    let plan = parse("SELECT CONCAT(s, s) FROM txt").expect("CONCAT(s, s) should parse");
    let top = first_project_expr(&plan);
    match strip_alias(top) {
        Expr::ScalarFn { kind, args } => {
            assert_eq!(*kind, ScalarFnKind::Concat);
            assert_eq!(args.len(), 2);
        }
        other => panic!("expected ScalarFn(Concat), got {other:?}"),
    }
    plan.schema().expect("CONCAT(s, s) must type-check");
}

#[test]
fn concat_variadic_three_args_parses() {
    let plan = parse("SELECT CONCAT(s, s, s) FROM txt")
        .expect("CONCAT(s, s, s) (variadic) should parse");
    let top = first_project_expr(&plan);
    match strip_alias(top) {
        Expr::ScalarFn { kind, args } => {
            assert_eq!(*kind, ScalarFnKind::Concat);
            assert_eq!(args.len(), 3, "variadic CONCAT preserves arg count");
        }
        other => panic!("expected ScalarFn(Concat), got {other:?}"),
    }
}

// ---- Type-error / arity-error paths ----------------------------------------

#[test]
fn upper_rejects_non_utf8_argument() {
    let plan = parse("SELECT UPPER(n) FROM txt").expect("UPPER(n) should parse (lowers fine)");
    let res: Result<_, _> = plan.schema().map_err(|e| format!("{e}"));
    // Type error fires at schema-resolution time.
    assert_err_contains(res, "UPPER");
}

#[test]
fn lower_rejects_non_utf8_argument() {
    let plan = parse("SELECT LOWER(n) FROM txt").expect("LOWER(n) should parse (lowers fine)");
    let res: Result<_, _> = plan.schema().map_err(|e| format!("{e}"));
    assert_err_contains(res, "LOWER");
}

#[test]
fn length_rejects_non_utf8_argument() {
    let plan = parse("SELECT LENGTH(n) FROM txt").expect("LENGTH(n) should parse (lowers fine)");
    let res: Result<_, _> = plan.schema().map_err(|e| format!("{e}"));
    assert_err_contains(res, "LENGTH");
}

#[test]
fn substring_rejects_non_utf8_first_argument() {
    let plan = parse("SELECT SUBSTRING(n FROM 1 FOR 2) FROM txt")
        .expect("SUBSTRING(n,...) should parse (lowers fine)");
    let res: Result<_, _> = plan.schema().map_err(|e| format!("{e}"));
    assert_err_contains(res, "SUBSTRING");
}

#[test]
fn concat_rejects_non_utf8_argument() {
    let plan = parse("SELECT CONCAT(s, n) FROM txt")
        .expect("CONCAT(s, n) should parse (lowers fine)");
    let res: Result<_, _> = plan.schema().map_err(|e| format!("{e}"));
    assert_err_contains(res, "CONCAT");
}

#[test]
fn upper_rejects_wrong_arity() {
    // UPPER takes exactly one argument; UPPER(s, s) must be a type/arity error.
    let plan = parse("SELECT UPPER(s, s) FROM txt").expect("UPPER(s,s) should parse");
    let res: Result<_, _> = plan.schema().map_err(|e| format!("{e}"));
    assert_err_contains(res, "UPPER");
}

#[test]
fn concat_rejects_single_argument() {
    // CONCAT requires at least two arguments at this layer.
    let plan = parse("SELECT CONCAT(s) FROM txt").expect("CONCAT(s) should parse");
    let res: Result<_, _> = plan.schema().map_err(|e| format!("{e}"));
    assert_err_contains(res, "CONCAT");
}

// ---- GPU LENGTH: now lowered to the fully-GPU StringLength variant ---------

#[test]
fn length_lowers_to_gpu_string_length_variant() {
    // `SELECT LENGTH(s) FROM txt` over a bare scan now lowers to the fully-GPU
    // `PhysicalPlan::StringLength` gather (no host fallback at the lowering
    // boundary). The output column is the `Int64` LENGTH contract.
    let plan = parse("SELECT LENGTH(s) FROM txt").expect("LENGTH(s) parses");
    let phys = lower_physical(&plan).expect("LENGTH(s) must lower to a GPU StringLength");
    match phys {
        PhysicalPlan::StringLength {
            ref table,
            ref outputs,
            ref output_schema,
        } => {
            assert_eq!(table, "txt");
            assert_eq!(outputs.len(), 1);
            assert!(
                matches!(&outputs[0], StringLengthOutput::Length { source } if source == "s"),
                "expected a single LENGTH(s) output, got {:?}",
                outputs[0]
            );
            assert_eq!(output_schema.fields.len(), 1);
            assert_eq!(output_schema.fields[0].dtype, DataType::Int64);
        }
        other => panic!("expected PhysicalPlan::StringLength, got {other:?}"),
    }
}

#[test]
fn length_with_passthrough_column_lowers_to_string_length() {
    // A mix of `LENGTH(s)` and a passthrough column still routes to the GPU
    // gather (the passthrough is lifted from the host batch by the executor).
    let plan = parse("SELECT n, LENGTH(s) FROM txt").expect("n, LENGTH(s) parses");
    let phys = lower_physical(&plan).expect("must lower to StringLength");
    match phys {
        PhysicalPlan::StringLength { outputs, .. } => {
            assert_eq!(outputs.len(), 2);
            assert!(matches!(&outputs[0], StringLengthOutput::Passthrough { source } if source == "n"));
            assert!(matches!(&outputs[1], StringLengthOutput::Length { source } if source == "s"));
        }
        other => panic!("expected StringLength, got {other:?}"),
    }
}

// ---- GPU UPPER/LOWER: lowered to the fully-GPU StringProject variant -------

#[test]
fn upper_lowers_to_gpu_string_project_variant() {
    // `SELECT UPPER(s) FROM txt` over a bare scan now lowers to the fully-GPU
    // two-pass `PhysicalPlan::StringProject` producer (was rejected before the
    // variable-width path was wired). Output column is `Utf8`.
    let plan = parse("SELECT UPPER(s) FROM txt").expect("UPPER(s) parses");
    let phys = lower_physical(&plan).expect("UPPER(s) must lower to a GPU StringProject");
    match phys {
        PhysicalPlan::StringProject {
            ref table,
            ref outputs,
            ref output_schema,
        } => {
            assert_eq!(table, "txt");
            assert_eq!(outputs.len(), 1);
            assert!(
                matches!(
                    &outputs[0],
                    StringProjectOutput::Transform { source, transform }
                        if source == "s" && *transform == StringTransform::Upper
                ),
                "expected a single UPPER(s) transform output, got {:?}",
                outputs[0]
            );
            assert_eq!(output_schema.fields.len(), 1);
            assert_eq!(output_schema.fields[0].dtype, DataType::Utf8);
        }
        other => panic!("expected PhysicalPlan::StringProject, got {other:?}"),
    }
}

#[test]
fn lower_with_passthrough_column_lowers_to_string_project() {
    // A mix of `LOWER(s)` and a non-Utf8 passthrough column routes to the GPU
    // two-pass producer (the passthrough is lifted from the host batch).
    let plan = parse("SELECT n, LOWER(s) FROM txt").expect("n, LOWER(s) parses");
    let phys = lower_physical(&plan).expect("must lower to StringProject");
    match phys {
        PhysicalPlan::StringProject { outputs, .. } => {
            assert_eq!(outputs.len(), 2);
            assert!(matches!(
                &outputs[0],
                StringProjectOutput::Passthrough { source } if source == "n"
            ));
            assert!(matches!(
                &outputs[1],
                StringProjectOutput::Transform { source, transform }
                    if source == "s" && *transform == StringTransform::Lower
            ));
        }
        other => panic!("expected StringProject, got {other:?}"),
    }
}

#[test]
fn substring_lowers_to_string_project() {
    // F9: SUBSTRING(col, lit, lit) over a bare Utf8 scan now lowers to the
    // GPU StringProject (host-realized two-pass producer), like UPPER/LOWER.
    let plan = parse("SELECT SUBSTRING(s, 1, 2) FROM txt").expect("SUBSTRING parses");
    let phys = lower_physical(&plan).expect("SUBSTRING must lower");
    assert!(
        matches!(phys, PhysicalPlan::StringProject { .. }),
        "SUBSTRING must lower to PhysicalPlan::StringProject, got {phys:?}"
    );
}

#[test]
fn trim_lowers_to_string_project() {
    // F9: single-arg TRIM(col) over a bare Utf8 scan now lowers to StringProject.
    let plan = parse("SELECT TRIM(s) FROM txt").expect("TRIM parses");
    let phys = lower_physical(&plan).expect("TRIM must lower");
    assert!(
        matches!(phys, PhysicalPlan::StringProject { .. }),
        "TRIM must lower to PhysicalPlan::StringProject, got {phys:?}"
    );
}

// ---- Unknown function names still rejected ---------------------------------

#[test]
fn unknown_scalar_function_still_rejected() {
    let provider = fixture();
    // `SQRT` is not in the recognised UPPER/LOWER/LENGTH/CONCAT set; the
    // catch-all "scalar function calls are not supported" rejection must
    // still fire.
    let res: Result<LogicalPlan, String> =
        parse_sql("SELECT SQRT(n) FROM txt", &provider).map_err(|e| format!("{e}"));
    assert_err_contains(res, "scalar function");
}

// ---- GPU end-to-end: LENGTH runs on the device -----------------------------
//
// Gated on `#[ignore = "gpu:string"]` per project convention (GpuVec uploads
// and kernel launches require a CUDA device). Run with:
//     cargo test --test string_fns_sql_test -- --ignored
// on a GPU host.

#[test]
#[ignore = "gpu:string"]
fn length_runs_on_gpu_for_plain_utf8_column() {
    use std::sync::Arc;

    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("CUDA ctx");
    // Plain StringArray → the engine's `Utf8` GpuColumnData variant
    // (slot-0-NULL, 1-based keys). Byte lengths: 5,3,4,5.
    let s = StringArray::from(vec!["alpha", "bb!", "gamm", "alpha"]);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(s)]).expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT LENGTH(s) FROM t")
        .expect("execute SELECT LENGTH(s)");
    let out = h.record_batch();
    assert_eq!(out.num_columns(), 1);
    let lens = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("LENGTH output is Int64");
    let got: Vec<i64> = (0..lens.len()).map(|i| lens.value(i)).collect();
    assert_eq!(got, vec![5, 3, 4, 5]);
}

#[test]
#[ignore = "gpu:string"]
fn length_runs_on_gpu_for_dict_encoded_column() {
    use std::sync::Arc;

    use arrow_array::builder::StringDictionaryBuilder;
    use arrow_array::types::Int32Type;
    use arrow_array::{Int64Array, RecordBatch};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("CUDA ctx");
    // Dictionary-encoded Utf8 (no nulls) → the native `DictUtf8` variant
    // (0-based keys), exercising the GPU ZeroBased gather path.
    let mut b: StringDictionaryBuilder<Int32Type> = StringDictionaryBuilder::new();
    for v in ["us", "eu", "us", "canada"] {
        b.append_value(v);
    }
    let dict = b.finish();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "region",
        ArrowDataType::Dictionary(
            Box::new(ArrowDataType::Int32),
            Box::new(ArrowDataType::Utf8),
        ),
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(dict)]).expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT LENGTH(region) FROM t")
        .expect("execute SELECT LENGTH(region)");
    let out = h.record_batch();
    let lens = out
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("LENGTH output is Int64");
    let got: Vec<i64> = (0..lens.len()).map(|i| lens.value(i)).collect();
    // byte lengths: "us"=2, "eu"=2, "us"=2, "canada"=6
    assert_eq!(got, vec![2, 2, 2, 6]);
}

// ---- GPU end-to-end: UPPER / LOWER run on the device (two-pass) -------------

#[test]
#[ignore = "gpu:string"]
fn upper_runs_on_gpu_for_plain_utf8_column() {
    use std::sync::Arc;

    use arrow_array::{Array, RecordBatch, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("CUDA ctx");
    // Plain (ASCII) StringArray → engine `Utf8` variant (slot-0-NULL, 1-based).
    let s = StringArray::from(vec!["alpha", "Bb!", "gamMa", "alpha"]);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(s)]).expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT UPPER(s) FROM t")
        .expect("execute SELECT UPPER(s)");
    let out = h.record_batch();
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("UPPER output is Utf8");
    let got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
    // ASCII upper-case, length-preserving.
    assert_eq!(got, vec!["ALPHA", "BB!", "GAMMA", "ALPHA"]);
}

#[test]
#[ignore = "gpu:string"]
fn lower_runs_on_gpu_for_dict_encoded_column() {
    use std::sync::Arc;

    use arrow_array::builder::StringDictionaryBuilder;
    use arrow_array::types::Int32Type;
    use arrow_array::{Array, RecordBatch, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("CUDA ctx");
    // Dictionary-encoded (no nulls) → native `DictUtf8` (ZeroBased) layout.
    let mut b: StringDictionaryBuilder<Int32Type> = StringDictionaryBuilder::new();
    for v in ["US", "Eu", "US", "CaNaDa"] {
        b.append_value(v);
    }
    let dict = b.finish();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "region",
        ArrowDataType::Dictionary(
            Box::new(ArrowDataType::Int32),
            Box::new(ArrowDataType::Utf8),
        ),
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(dict)]).expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT LOWER(region) FROM t")
        .expect("execute SELECT LOWER(region)");
    let out = h.record_batch();
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("LOWER output is Utf8");
    let got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
    assert_eq!(got, vec!["us", "eu", "us", "canada"]);
}

#[test]
#[ignore = "gpu:string"]
fn upper_preserves_nulls_on_gpu() {
    use std::sync::Arc;

    use arrow_array::{Array, RecordBatch, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("CUDA ctx");
    // NULL row must surface as Arrow NULL (not "").
    let s = StringArray::from(vec![Some("ab"), None, Some("cD")]);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(s)]).expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT UPPER(s) FROM t")
        .expect("execute SELECT UPPER(s)");
    let out = h.record_batch();
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("UPPER output is Utf8");
    assert_eq!(col.len(), 3);
    assert_eq!(col.value(0), "AB");
    assert!(col.is_null(1), "NULL row must stay NULL");
    assert_eq!(col.value(2), "CD");
}

// ---- SUBSTRING / TRIM end-to-end (host-side projection) --------------------
//
// SUBSTRING and TRIM have no GPU producer wired into the executor yet (see the
// TODO(string-fn-gpu) markers); they lower to the host-side
// `PhysicalPlan::Project` whose executor evaluates them via
// `expr_agg::eval_expr`. These tests still need a registered table (and thus a
// CUDA context for the scan), so they are gated like the rest of the suite.

#[test]
#[ignore = "gpu:string"]
fn substring_runs_end_to_end() {
    use std::sync::Arc;

    use arrow_array::{Array, RecordBatch, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("CUDA ctx");
    let s = StringArray::from(vec!["hello", "world", "abcdef"]);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(s)]).expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT SUBSTRING(s, 2, 3) FROM t")
        .expect("execute SELECT SUBSTRING(s,2,3)");
    let out = h.record_batch();
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("SUBSTRING output is Utf8");
    let got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
    // 1-based start=2, length=3.
    assert_eq!(got, vec!["ell", "orl", "bcd"]);
}

#[test]
#[ignore = "gpu:string"]
fn trim_runs_end_to_end() {
    use std::sync::Arc;

    use arrow_array::{Array, RecordBatch, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use craton_bolt::Engine;

    let mut engine = Engine::new().expect("CUDA ctx");
    let s = StringArray::from(vec!["  hi  ", "nochange", "  pad"]);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(s)]).expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT TRIM(s) FROM t")
        .expect("execute SELECT TRIM(s)");
    let out = h.record_batch();
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("TRIM output is Utf8");
    let got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
    assert_eq!(got, vec!["hi", "nochange", "pad"]);
}

// ---- GPU end-to-end: LIKE / NOT LIKE over a non-dict Utf8 column ------------
//
// ⚠️ UNVALIDATED device path. The matcher kernel
// (`jit::string_kernel::compile_like_match_kernel`) has not run on GPU
// hardware; these `#[ignore = "gpu:string"]` tests are the bring-up harness.
// Until they pass on a real device, correctness is guaranteed by the host
// mirror (`exec::string_like::like_match_row` vs `exec::like::PatternMatcher`)
// and the PTX-shape tests in `jit::string_kernel`. Run with:
//     cargo test --test string_fns_sql_test -- --ignored
// on a GPU host.

/// Shared fixture: register a plain (non-dict) Utf8 column `s` and return the
/// engine + the values, so each LIKE test can assert against a known set.
fn like_e2e_engine() -> craton_bolt::Engine {
    use std::sync::Arc;
    use arrow_array::{RecordBatch, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    let mut engine = craton_bolt::Engine::new().expect("CUDA ctx");
    let s = StringArray::from(vec![
        Some("foobar"),  // prefix foo, contains foo
        Some("xfoo"),    // suffix foo, contains foo
        Some("foo"),     // exact foo
        Some("bar"),     // none
        None,            // NULL stays NULL under both LIKE and NOT LIKE
        Some("afoob"),   // contains foo only
    ]);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(s)]).expect("batch");
    engine.register_table("t", batch).expect("register");
    engine
}

fn collect_s(h: &craton_bolt::exec::QueryHandle) -> Vec<String> {
    use arrow_array::{Array, StringArray};
    let out = h.record_batch();
    let col = out
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("Utf8 output");
    (0..col.len())
        .filter(|&i| !col.is_null(i))
        .map(|i| col.value(i).to_string())
        .collect()
}

#[test]
#[ignore = "gpu:string"]
fn like_prefix_runs_on_gpu() {
    let engine = like_e2e_engine();
    let h = engine
        .sql("SELECT s FROM t WHERE s LIKE 'foo%'")
        .expect("LIKE 'foo%'");
    let mut got = collect_s(&h);
    got.sort();
    // starts_with("foo"): "foobar", "foo".
    assert_eq!(got, vec!["foo".to_string(), "foobar".to_string()]);
}

#[test]
#[ignore = "gpu:string"]
fn like_suffix_runs_on_gpu() {
    let engine = like_e2e_engine();
    let h = engine
        .sql("SELECT s FROM t WHERE s LIKE '%foo'")
        .expect("LIKE '%foo'");
    let mut got = collect_s(&h);
    got.sort();
    // ends_with("foo"): "xfoo", "foo".
    assert_eq!(got, vec!["foo".to_string(), "xfoo".to_string()]);
}

#[test]
#[ignore = "gpu:string"]
fn like_contains_runs_on_gpu() {
    let engine = like_e2e_engine();
    let h = engine
        .sql("SELECT s FROM t WHERE s LIKE '%foo%'")
        .expect("LIKE '%foo%'");
    let mut got = collect_s(&h);
    got.sort();
    // contains("foo"): foobar, xfoo, foo, afoob.
    assert_eq!(
        got,
        vec![
            "afoob".to_string(),
            "foo".to_string(),
            "foobar".to_string(),
            "xfoo".to_string(),
        ]
    );
}

#[test]
#[ignore = "gpu:string"]
fn like_exact_runs_on_gpu() {
    let engine = like_e2e_engine();
    let h = engine
        .sql("SELECT s FROM t WHERE s LIKE 'foo'")
        .expect("LIKE 'foo'");
    let got = collect_s(&h);
    // exact "foo" only.
    assert_eq!(got, vec!["foo".to_string()]);
}

#[test]
#[ignore = "gpu:string"]
fn not_like_runs_on_gpu_and_preserves_nulls() {
    let engine = like_e2e_engine();
    let h = engine
        .sql("SELECT s FROM t WHERE s NOT LIKE 'foo%'")
        .expect("NOT LIKE 'foo%'");
    let mut got = collect_s(&h);
    got.sort();
    // NOT starts_with("foo"), with NULL dropped (NULL NOT LIKE = NULL):
    // "xfoo", "bar", "afoob".
    assert_eq!(
        got,
        vec!["afoob".to_string(), "bar".to_string(), "xfoo".to_string()]
    );
}
