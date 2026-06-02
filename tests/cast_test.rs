// SPDX-License-Identifier: Apache-2.0

//! Integration tests for SQL `CAST(<expr> AS <type>)` over primitive types.
//!
//! v0.7 scope (this PR): parser + type-check + numeric ↔ numeric (and
//! Bool ↔ Int) GPU lowering.
//!
//! * The SQL frontend parses `CAST(<expr> AS <type>)` and the Postgres
//!   `expr::type` shortcut, mapping the type name onto the engine's
//!   internal [`DataType`]. `TRY_CAST` is now supported and lowers to a
//!   safe (`safe: true`) `Expr::Cast`; the BigQuery `FORMAT` clause is
//!   still rejected.
//! * The logical plane type-checks the source-target pair against
//!   `cast_is_supported`; unsupported pairs (e.g. `Utf8 -> Int32`) error
//!   with a `BoltError::Type` carrying the source and target types.
//! * The physical plane lowers numeric ↔ numeric and Bool ↔ Int casts
//!   to a PTX `cvt.*` instruction (see `Codegen::emit_cast_expr` in
//!   `physical_plan.rs`). CASTs whose target is `Decimal128`, `Date32`,
//!   `Timestamp`, or `Utf8` are still rejected at `lower()` with a
//!   tightened "CAST to/from {Type} not yet lowered to GPU" message.
//!
//! These tests cover the three surfaces individually so a regression in
//! any one of them surfaces clearly.

use craton_bolt::plan::{
    lower_physical, parse_sql, DataType, Expr, Field, LogicalPlan, MemTableProvider, Schema,
};

// ---- Fixture ----------------------------------------------------------------

/// Single-table fixture with one column per primitive dtype we care about.
fn fixture() -> MemTableProvider {
    let schema = Schema::new(vec![
        Field {
            name: "i32".into(),
            dtype: DataType::Int32,
            nullable: false,
        },
        Field {
            name: "i64".into(),
            dtype: DataType::Int64,
            nullable: false,
        },
        Field {
            name: "f32".into(),
            dtype: DataType::Float32,
            nullable: false,
        },
        Field {
            name: "f64".into(),
            dtype: DataType::Float64,
            nullable: false,
        },
        Field {
            name: "b".into(),
            dtype: DataType::Bool,
            nullable: false,
        },
        Field {
            name: "s".into(),
            dtype: DataType::Utf8,
            nullable: false,
        },
    ]);
    MemTableProvider::new().with_table("t", schema)
}

// ---- Helpers ---------------------------------------------------------------

/// Walk the logical plan down to the SELECT projection list and return the
/// nth expression's `Expr::Cast { target, .. }` target dtype, or panic with
/// the actual expression shape if the nth slot isn't a Cast.
fn nth_project_cast_target(plan: &LogicalPlan, n: usize) -> DataType {
    let mut cur = plan;
    loop {
        match cur {
            LogicalPlan::Project { exprs, .. } => {
                let e = exprs.get(n).unwrap_or_else(|| {
                    panic!("expected at least {} projection exprs, got {}", n + 1, exprs.len())
                });
                // Aliases are transparent — peel them off so `CAST(x AS Int64)`
                // and `CAST(x AS Int64) AS y` both pin the same shape.
                let mut peel = e;
                while let Expr::Alias(inner, _) = peel {
                    peel = inner;
                }
                match peel {
                    Expr::Cast { target, .. } => return *target,
                    other => panic!("expected Expr::Cast at projection slot {n}, got {other:?}"),
                }
            }
            LogicalPlan::Filter { input, .. }
            | LogicalPlan::Aggregate { input, .. }
            | LogicalPlan::Distinct { input }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Sort { input, .. } => cur = input.as_ref(),
            other => panic!("expected to reach a Project node, stopped at {other:?}"),
        }
    }
}

// ---- 1. Parser: each accepted source/target pair parses to Expr::Cast -----

/// Same-type identity casts always parse and type-check. We pick one
/// representative for every primitive type so the no-op fold path is
/// covered end-to-end at the logical plane.
#[test]
fn cast_same_type_parses_and_typechecks() {
    let provider = fixture();
    for (sql, want) in [
        ("SELECT CAST(i32 AS INT) FROM t", DataType::Int32),
        ("SELECT CAST(i64 AS BIGINT) FROM t", DataType::Int64),
        ("SELECT CAST(f32 AS REAL) FROM t", DataType::Float32),
        ("SELECT CAST(f64 AS DOUBLE) FROM t", DataType::Float64),
        ("SELECT CAST(b AS BOOLEAN) FROM t", DataType::Bool),
    ] {
        let plan = parse_sql(sql, &provider).unwrap_or_else(|e| panic!("parse `{sql}`: {e:?}"));
        assert_eq!(nth_project_cast_target(&plan, 0), want, "for SQL `{sql}`");
        // Type-check must succeed.
        plan.schema().unwrap_or_else(|e| panic!("typecheck `{sql}`: {e:?}"));
    }
}

/// Int32 <-> Int64 widening/narrowing.
#[test]
fn cast_int32_int64_both_directions() {
    let provider = fixture();
    let plan = parse_sql("SELECT CAST(i32 AS BIGINT) FROM t", &provider).unwrap();
    assert_eq!(nth_project_cast_target(&plan, 0), DataType::Int64);
    plan.schema().expect("Int32 -> Int64 must typecheck");

    let plan = parse_sql("SELECT CAST(i64 AS INT) FROM t", &provider).unwrap();
    assert_eq!(nth_project_cast_target(&plan, 0), DataType::Int32);
    plan.schema().expect("Int64 -> Int32 must typecheck");
}

/// Integer -> Float (both widths).
#[test]
fn cast_int_to_float() {
    let provider = fixture();
    for (sql, want) in [
        ("SELECT CAST(i32 AS REAL) FROM t", DataType::Float32),
        ("SELECT CAST(i32 AS DOUBLE) FROM t", DataType::Float64),
        ("SELECT CAST(i64 AS REAL) FROM t", DataType::Float32),
        ("SELECT CAST(i64 AS DOUBLE) FROM t", DataType::Float64),
    ] {
        let plan = parse_sql(sql, &provider).unwrap_or_else(|e| panic!("parse `{sql}`: {e:?}"));
        assert_eq!(nth_project_cast_target(&plan, 0), want, "for SQL `{sql}`");
        plan.schema().unwrap_or_else(|e| panic!("typecheck `{sql}`: {e:?}"));
    }
}

/// Float32 <-> Float64 widening/narrowing.
#[test]
fn cast_float32_float64_both_directions() {
    let provider = fixture();
    let plan = parse_sql("SELECT CAST(f32 AS DOUBLE) FROM t", &provider).unwrap();
    assert_eq!(nth_project_cast_target(&plan, 0), DataType::Float64);
    plan.schema().expect("Float32 -> Float64 must typecheck");

    let plan = parse_sql("SELECT CAST(f64 AS REAL) FROM t", &provider).unwrap();
    assert_eq!(nth_project_cast_target(&plan, 0), DataType::Float32);
    plan.schema().expect("Float64 -> Float32 must typecheck");
}

/// Int <-> Bool (0/1 round-trip).
#[test]
fn cast_int_bool_both_directions() {
    let provider = fixture();
    for (sql, want) in [
        ("SELECT CAST(i32 AS BOOLEAN) FROM t", DataType::Bool),
        ("SELECT CAST(i64 AS BOOLEAN) FROM t", DataType::Bool),
        ("SELECT CAST(b AS INT) FROM t", DataType::Int32),
        ("SELECT CAST(b AS BIGINT) FROM t", DataType::Int64),
    ] {
        let plan = parse_sql(sql, &provider).unwrap_or_else(|e| panic!("parse `{sql}`: {e:?}"));
        assert_eq!(nth_project_cast_target(&plan, 0), want, "for SQL `{sql}`");
        plan.schema().unwrap_or_else(|e| panic!("typecheck `{sql}`: {e:?}"));
    }
}

/// Postgres `expr::type` shortcut must lower to the same `Expr::Cast` shape
/// as the standard `CAST(expr AS type)` form. This pins the `CastKind::DoubleColon`
/// arm of the SQL lowerer alongside the standard `CastKind::Cast` arm.
#[test]
fn cast_postgres_double_colon_shortcut() {
    let provider = fixture();
    let plan = parse_sql("SELECT i32::BIGINT FROM t", &provider).unwrap();
    assert_eq!(nth_project_cast_target(&plan, 0), DataType::Int64);
    plan.schema().expect("postgres :: shortcut must typecheck");
}

/// `CAST(NULL AS Int32)` is a legitimate SQL fragment — the result type
/// is purely declared by the target, and the inner is an untyped NULL.
/// Pin the dtype rule on `Expr::Cast` for the NULL-operand fast path.
#[test]
fn cast_null_takes_target_dtype() {
    let provider = fixture();
    let plan = parse_sql("SELECT CAST(NULL AS BIGINT) FROM t", &provider).unwrap();
    assert_eq!(nth_project_cast_target(&plan, 0), DataType::Int64);
    let schema = plan.schema().expect("CAST(NULL AS BIGINT) must typecheck");
    // The projection slot must carry the declared target dtype, regardless
    // of the inner being an untyped NULL.
    assert_eq!(schema.fields[0].dtype, DataType::Int64);
}

// ---- 2. Type-check: unsupported pairs error cleanly -----------------------

/// String -> int is rejected at the logical plane (not at the physical
/// reject); the error message names both the source and target dtypes so
/// the user sees what they wrote vs what's missing.
#[test]
fn cast_utf8_to_int_rejected_at_typecheck() {
    let provider = fixture();
    let plan = parse_sql("SELECT CAST(s AS INT) FROM t", &provider)
        .expect("parser must accept Utf8 -> Int32 syntactically");
    let err = plan
        .schema()
        .expect_err("Utf8 -> Int32 must error at typecheck");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Utf8") && msg.contains("Int32"),
        "expected error to name both Utf8 and Int32, got: {msg}"
    );
    assert!(
        msg.to_lowercase().contains("unsupported")
            || msg.to_lowercase().contains("cast"),
        "expected error to mention CAST/unsupported, got: {msg}"
    );
}

/// Numeric -> string is also rejected — symmetric to the Utf8 -> Int32 case.
#[test]
fn cast_int_to_utf8_rejected_at_typecheck() {
    let provider = fixture();
    // sqlparser accepts VARCHAR as a target type name, so we use that.
    // If the type name itself is rejected by `lower_cast_data_type` (it
    // isn't in our accepted list), the parser surfaces the error rather
    // than the typechecker — either path is a hard error, which is what
    // we care about here.
    let res = parse_sql("SELECT CAST(i32 AS VARCHAR) FROM t", &provider)
        .and_then(|p| p.schema().map(|_| p));
    assert!(
        res.is_err(),
        "Int32 -> Utf8 (VARCHAR) must error somewhere along parse+typecheck, got Ok"
    );
}

/// Bool -> Float is rejected (we only support Bool <-> integer, not Bool -> float).
/// Pins the boundary on `cast_is_supported`: a future relaxation would also
/// need to extend the accepted pairs and update this test.
#[test]
fn cast_bool_to_float_rejected_at_typecheck() {
    let provider = fixture();
    let plan = parse_sql("SELECT CAST(b AS REAL) FROM t", &provider)
        .expect("parser must accept Bool -> Float32 syntactically");
    let err = plan
        .schema()
        .expect_err("Bool -> Float32 must error at typecheck");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Bool") && msg.contains("Float32"),
        "expected error to name Bool and Float32, got: {msg}"
    );
}

/// Float -> Bool is also rejected (the integer <-> Bool round-trip is the
/// only Bool conversion in the v0.5 surface).
#[test]
fn cast_float_to_bool_rejected_at_typecheck() {
    let provider = fixture();
    let plan = parse_sql("SELECT CAST(f64 AS BOOLEAN) FROM t", &provider)
        .expect("parser must accept Float64 -> Bool syntactically");
    let err = plan
        .schema()
        .expect_err("Float64 -> Bool must error at typecheck");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("Float64") && msg.contains("Bool"),
        "expected error to name Float64 and Bool, got: {msg}"
    );
}

/// `TRY_CAST` and `SAFE_CAST` carry NULL-on-failure semantics the planner
/// can't honour yet; the SQL frontend rejects them with a clear message
/// pointing the user at the plain `CAST` form.
#[test]
fn try_cast_lowers_as_safe_cast() {
    // TRY_CAST is now supported: the frontend lowers it to the same
    // `Expr::Cast` IR as plain CAST but with `safe: true` (non-trapping /
    // NULL-on-failure semantics). It plans and lowers cleanly.
    let provider = fixture();
    let plan = parse_sql("SELECT TRY_CAST(i32 AS BIGINT) FROM t", &provider)
        .expect("TRY_CAST must parse and plan");
    lower_physical(&plan).expect("TRY_CAST must lower cleanly");
    // The projected expression must be a safe Int32 -> Int64 cast.
    if let LogicalPlan::Project { exprs, .. } = &plan {
        assert!(
            matches!(
                &exprs[0],
                Expr::Cast { target: DataType::Int64, safe: true, .. }
            ),
            "TRY_CAST should lower to a safe Int64 cast, got: {:?}",
            exprs[0]
        );
    } else {
        panic!("expected a Project at the plan root, got {plan:?}");
    }
}

// ---- 3. Physical-plan lowering --------------------------------------------

/// Numeric ↔ numeric CASTs lower cleanly through `lower_physical` — no
/// rejection, the physical plan carries an `Op::Cast` IR node that the
/// PTX emitter turns into a `cvt.*` instruction (golden coverage for
/// the per-pair PTX opcodes lives in `src/jit/ptx_gen.rs`'s unit tests).
#[test]
fn cast_int32_to_int64_lowers_cleanly() {
    let provider = fixture();
    let plan = parse_sql("SELECT CAST(i32 AS BIGINT) FROM t", &provider).unwrap();
    lower_physical(&plan).expect("Int32 -> Int64 CAST must lower cleanly");
}

/// CAST inside a Filter predicate also lowers — `Codegen::emit_expr`
/// handles `Expr::Cast` uniformly across projection-list and predicate
/// positions.
#[test]
fn cast_inside_filter_lowers_cleanly() {
    let provider = fixture();
    let plan =
        parse_sql("SELECT i32 FROM t WHERE CAST(i32 AS BIGINT) > 0", &provider).unwrap();
    lower_physical(&plan).expect("CAST in WHERE clause must lower cleanly");
}
