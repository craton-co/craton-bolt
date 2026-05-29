// SPDX-License-Identifier: Apache-2.0

//! Parser + type-check tests for the v0.5 SQL string scalar functions:
//! `UPPER`, `LOWER`, `LENGTH`, `SUBSTRING`, `CONCAT`.
//!
//! Scope (v0.5 MVP): the SQL frontend recognises these calls and lowers
//! them into `Expr::ScalarFn`; the logical plan's type-checker validates
//! their argument shapes; the physical-plan boundary rejects them cleanly
//! with a `Plan` error ("string scalar function ... is not yet lowered to
//! GPU"). Execution wiring is a follow-up.
//!
//! These tests pin:
//!   * Successful parse+lower of each function (logical-plan shape only —
//!     we do NOT round-trip through `lower_physical` for the positive
//!     paths because that boundary intentionally rejects the call).
//!   * Type errors for wrong argument dtypes / arity.
//!   * The `lower_physical` rejection error message (a single
//!     representative case is enough — the rejection lives in one place).

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Expr, Field, LogicalPlan, MemTableProvider, PhysicalPlan,
    ScalarFnKind, Schema, StringLengthOutput,
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

// ---- Physical-plan boundary: rejected with a clear error -------------------

#[test]
fn upper_rejected_at_lower_with_followup_marker() {
    let plan = parse("SELECT UPPER(s) FROM txt").expect("UPPER(s) parses");
    let err = lower_physical(&plan).expect_err("lower_physical must reject ScalarFn for v0.5");
    let msg = format!("{err}");
    let lower = msg.to_ascii_lowercase();
    assert!(
        lower.contains("upper") || lower.contains("scalar function"),
        "rejection message should mention the function or 'scalar function', got: {msg}"
    );
    assert!(
        lower.contains("follow-up") || lower.contains("not yet"),
        "rejection should flag this as a follow-up; got: {msg}"
    );
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

#[test]
fn upper_still_rejected_not_string_length() {
    // Only LENGTH is wired to the GPU; UPPER (a variable-width producer) must
    // still reject at the lowering boundary rather than route to StringLength.
    let plan = parse("SELECT UPPER(s) FROM txt").expect("UPPER(s) parses");
    let res = lower_physical(&plan);
    assert!(res.is_err(), "UPPER must still be rejected at lowering");
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
