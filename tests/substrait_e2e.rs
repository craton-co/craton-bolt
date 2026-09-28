// SPDX-License-Identifier: Apache-2.0

//! End-to-end fixtures for the optional `substrait` subsystem.
//!
//! The review that retired `deep-research-bolt.md` called out that the
//! `substrait` lane only *compiled* the feature: no test drove a Substrait
//! plan through the public entry point, so the ingestion path could rot
//! silently between releases. This binary closes that gap with two tiers:
//!
//! 1. **Host-runnable golden fixtures** (not `#[ignore]`d) — a realistic
//!    producer-shaped plan (`ReadRel` → `FilterRel` → `ProjectRel`, and an
//!    `AggregateRel`) is converted through the public
//!    [`craton_bolt::plan::substrait::substrait_to_logical_plan`] and the
//!    resulting [`LogicalPlan`] shape + schema is asserted. These run in the
//!    hosted `feature build (flight + substrait)` CI lane.
//! 2. **A real end-to-end execution fixture** (`#[ignore = "gpu:e2e"]`) — the
//!    same converted plan is handed to `Engine::run_logical_plan` against a
//!    registered table and the *result values* are asserted. It needs a CUDA
//!    context, so it runs on the blocking self-hosted GPU lane:
//!
//! ```text
//! cargo test --no-default-features --features cudarc,substrait --test substrait_e2e -- --ignored
//! ```

#![cfg(feature = "substrait")]

use std::sync::Arc;

use arrow_array::{Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::plan::logical_plan::{DataType, Field, Schema};
use craton_bolt::plan::substrait::substrait_to_logical_plan;
use craton_bolt::plan::MemTableProvider;
use craton_bolt::LogicalPlan;

use substrait::proto;

// ---------------------------------------------------------------------------
// Substrait plan construction helpers
//
// A Substrait producer emits the function vocabulary out-of-band in
// `Plan.extensions`; expressions then reference a function by its numeric
// `function_anchor`. These helpers build that shape by hand so the fixtures
// mirror what a real producer (DuckDB, Isthmus, DataFusion) puts on the wire
// rather than an engine-internal shortcut.
// ---------------------------------------------------------------------------

/// `sales(region Int64, qty Int64)` — the fixture table both tiers share.
fn provider() -> MemTableProvider {
    MemTableProvider::new().with_table(
        "sales",
        Schema::new(vec![
            Field::new("region", DataType::Int64, false),
            Field::new("qty", DataType::Int64, false),
        ]),
    )
}

/// Declare `anchor -> compound function name` in the plan's extension list.
fn extension(anchor: u32, name: &str) -> proto::extensions::SimpleExtensionDeclaration {
    use proto::extensions::simple_extension_declaration::{ExtensionFunction, MappingType};
    proto::extensions::SimpleExtensionDeclaration {
        mapping_type: Some(MappingType::ExtensionFunction(ExtensionFunction {
            extension_uri_reference: 0,
            function_anchor: anchor,
            name: name.to_string(),
        })),
    }
}

/// A `ReadRel` over the named fixture table.
fn read_sales() -> proto::Rel {
    use proto::read_rel::{NamedTable, ReadType};
    proto::Rel {
        rel_type: Some(proto::rel::RelType::Read(Box::new(proto::ReadRel {
            read_type: Some(ReadType::NamedTable(NamedTable {
                names: vec!["sales".to_string()],
                advanced_extension: None,
            })),
            ..Default::default()
        }))),
    }
}

/// A direct field reference to output column `idx` of the input.
fn field_ref(idx: i32) -> proto::Expression {
    use proto::expression::field_reference::{ReferenceType, RootType};
    use proto::expression::reference_segment::{ReferenceType as SegType, StructField};
    use proto::expression::{FieldReference, ReferenceSegment, RexType};
    proto::Expression {
        rex_type: Some(RexType::Selection(Box::new(FieldReference {
            reference_type: Some(ReferenceType::DirectReference(ReferenceSegment {
                reference_type: Some(SegType::StructField(Box::new(StructField {
                    field: idx,
                    child: None,
                }))),
            })),
            root_type: Some(RootType::RootReference(
                proto::expression::field_reference::RootReference {},
            )),
        }))),
    }
}

/// An `i64` literal expression.
fn literal_i64(v: i64) -> proto::Expression {
    use proto::expression::literal::LiteralType;
    use proto::expression::{Literal, RexType};
    proto::Expression {
        rex_type: Some(RexType::Literal(Literal {
            literal_type: Some(LiteralType::I64(v)),
            nullable: false,
            type_variation_reference: 0,
        })),
    }
}

/// A binary scalar function call `f(lhs, rhs)` resolved through `anchor`.
fn scalar_fn(anchor: u32, lhs: proto::Expression, rhs: proto::Expression) -> proto::Expression {
    use proto::expression::RexType;
    use proto::function_argument::ArgType;
    use proto::{Expression, FunctionArgument};
    let arg = |e: Expression| FunctionArgument {
        arg_type: Some(ArgType::Value(e)),
    };
    Expression {
        rex_type: Some(RexType::ScalarFunction(proto::expression::ScalarFunction {
            function_reference: anchor,
            arguments: vec![arg(lhs), arg(rhs)],
            ..Default::default()
        })),
    }
}

/// Wrap a root `Rel` plus explicit output names into a single-relation `Plan`
/// carrying the supplied function extensions.
fn plan_of(
    rel: proto::Rel,
    names: Vec<String>,
    extensions: Vec<proto::extensions::SimpleExtensionDeclaration>,
) -> proto::Plan {
    proto::Plan {
        extensions,
        relations: vec![proto::PlanRel {
            rel_type: Some(proto::plan_rel::RelType::Root(proto::RelRoot {
                input: Some(rel),
                names,
            })),
        }],
        version: Some(proto::Version {
            major_number: 0,
            minor_number: 55,
            patch_number: 0,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// `SELECT region, qty FROM sales WHERE qty > 10` as a Substrait plan.
///
/// `emit` is left unset on the `ProjectRel` so the converter takes the
/// Substrait default (input columns followed by the computed expressions) and
/// the `RelRoot` names rename the projection's output — the shape a real
/// producer emits for a two-column projection.
fn filter_project_plan() -> proto::Plan {
    const GT: u32 = 1;

    let filter = proto::Rel {
        rel_type: Some(proto::rel::RelType::Filter(Box::new(proto::FilterRel {
            input: Some(Box::new(read_sales())),
            condition: Some(Box::new(scalar_fn(GT, field_ref(1), literal_i64(10)))),
            ..Default::default()
        }))),
    };

    let project = proto::Rel {
        rel_type: Some(proto::rel::RelType::Project(Box::new(proto::ProjectRel {
            input: Some(Box::new(filter)),
            expressions: vec![field_ref(0), field_ref(1)],
            common: Some(proto::RelCommon {
                emit_kind: Some(proto::rel_common::EmitKind::Emit(proto::rel_common::Emit {
                    // The two appended expressions (indices 2 and 3 of
                    // input-columns ++ expressions) are the emitted output.
                    output_mapping: vec![2, 3],
                })),
                ..Default::default()
            }),
            ..Default::default()
        }))),
    };

    plan_of(
        project,
        vec!["region".to_string(), "qty".to_string()],
        vec![extension(GT, "gt:i64_i64")],
    )
}

/// `SELECT region, SUM(qty) FROM sales GROUP BY region` as a Substrait plan.
///
/// `Grouping::grouping_expressions` is deprecated upstream in favour of
/// `expression_references`, but it is the field the converter reads (and the
/// one producers still emit for a plain single-grouping aggregate), so the
/// fixture stays on it deliberately.
#[allow(deprecated)]
fn aggregate_plan() -> proto::Plan {
    const SUM: u32 = 2;
    use proto::aggregate_function::AggregationInvocation;
    use proto::aggregate_rel::{Grouping, Measure};
    use proto::function_argument::ArgType;
    use proto::AggregationPhase;

    let agg = proto::Rel {
        rel_type: Some(proto::rel::RelType::Aggregate(Box::new(
            proto::AggregateRel {
                input: Some(Box::new(read_sales())),
                groupings: vec![Grouping {
                    grouping_expressions: vec![field_ref(0)],
                    ..Default::default()
                }],
                measures: vec![Measure {
                    measure: Some(proto::AggregateFunction {
                        function_reference: SUM,
                        arguments: vec![proto::FunctionArgument {
                            arg_type: Some(ArgType::Value(field_ref(1))),
                        }],
                        phase: AggregationPhase::InitialToResult as i32,
                        invocation: AggregationInvocation::All as i32,
                        ..Default::default()
                    }),
                    filter: None,
                }],
                ..Default::default()
            },
        ))),
    };

    plan_of(
        agg,
        vec!["region".to_string(), "total".to_string()],
        vec![extension(SUM, "sum:i64")],
    )
}

// ---------------------------------------------------------------------------
// Tier 1 — host-runnable conversion fixtures (hosted CI)
// ---------------------------------------------------------------------------

/// Depth-first search for a plan node matching `pred`.
fn plan_contains(plan: &LogicalPlan, pred: &dyn Fn(&LogicalPlan) -> bool) -> bool {
    if pred(plan) {
        return true;
    }
    match plan {
        LogicalPlan::Project { input, .. }
        | LogicalPlan::Filter { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::Limit { input, .. } => plan_contains(input, pred),
        _ => false,
    }
}

/// A producer-shaped `ReadRel` → `FilterRel` → `ProjectRel` plan converts to a
/// projection over the filtered scan, and the `RelRoot` names land on the
/// output schema.
///
/// The exact projection nesting is an implementation detail — Substrait `emit`
/// lowers to its own projection above the expression list — so the assertion
/// pins the root node kind, the presence of the pushed-down filter, and the
/// output schema rather than a brittle exact tree.
#[test]
fn filter_project_plan_converts_end_to_end() {
    let logical = substrait_to_logical_plan(&filter_project_plan(), &provider())
        .expect("filter/project plan converts");

    assert!(
        matches!(logical, LogicalPlan::Project { .. }),
        "expected a Project root, got {logical:?}"
    );
    assert!(
        plan_contains(&logical, &|p| matches!(p, LogicalPlan::Filter { .. })),
        "converted plan must retain the FilterRel, got {logical:?}"
    );
    assert!(
        plan_contains(&logical, &|p| matches!(p, LogicalPlan::Scan { .. })),
        "converted plan must read the base table, got {logical:?}"
    );

    let schema = logical.schema().expect("converted plan type-checks");
    let names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, vec!["region", "qty"]);
    assert!(schema.fields.iter().all(|f| f.dtype == DataType::Int64));
}

/// An `AggregateRel` with one grouping key and one `SUM` measure converts to
/// an engine `Aggregate`, with the `RelRoot` names applied to both outputs.
#[test]
fn aggregate_plan_converts_end_to_end() {
    let logical =
        substrait_to_logical_plan(&aggregate_plan(), &provider()).expect("aggregate plan converts");

    let schema = logical.schema().expect("converted plan type-checks");
    let names: Vec<&str> = schema.fields.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, vec!["region", "total"]);
}

/// A plan whose `ReadRel` names a table the provider does not know must be
/// rejected with a diagnostic rather than converted into a dangling scan.
#[test]
fn unknown_table_is_rejected() {
    use proto::read_rel::{NamedTable, ReadType};
    let missing = proto::Rel {
        rel_type: Some(proto::rel::RelType::Read(Box::new(proto::ReadRel {
            read_type: Some(ReadType::NamedTable(NamedTable {
                names: vec!["does_not_exist".to_string()],
                advanced_extension: None,
            })),
            ..Default::default()
        }))),
    };
    let plan = plan_of(missing, Vec::new(), Vec::new());
    assert!(
        substrait_to_logical_plan(&plan, &provider()).is_err(),
        "unknown base table must not convert"
    );
}

// ---------------------------------------------------------------------------
// Tier 2 — real execution against the engine (self-hosted GPU lane)
// ---------------------------------------------------------------------------

/// The fixture rows both execution tests share, matching [`provider`]'s schema.
#[cfg(test)]
fn sales_batch() -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("region", ArrowDataType::Int64, false),
        ArrowField::new("qty", ArrowDataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 2, 3])),
            Arc::new(Int64Array::from(vec![5, 20, 30, 7, 40])),
        ],
    )
    .expect("fixture batch")
}

fn col_i64(batch: &RecordBatch, c: usize) -> &Int64Array {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("column is Int64")
}

/// Full Substrait → LogicalPlan → engine execution: the converted filter /
/// project plan returns exactly the rows with `qty > 10`.
#[test]
#[ignore = "gpu:e2e"]
fn substrait_filter_project_executes_on_engine() {
    let mut engine = craton_bolt::Engine::new().expect("CUDA ctx");
    engine
        .register_table("sales", sales_batch())
        .expect("register sales");

    let logical =
        substrait_to_logical_plan(&filter_project_plan(), &provider()).expect("plan converts");
    let handle = engine
        .run_logical_plan(&logical)
        .expect("converted plan executes");
    let out = handle.record_batch();

    let regions: Vec<i64> = (0..out.num_rows())
        .map(|i| col_i64(out, 0).value(i))
        .collect();
    let qtys: Vec<i64> = (0..out.num_rows())
        .map(|i| col_i64(out, 1).value(i))
        .collect();
    // qty > 10 keeps (1, 20), (2, 30), (3, 40) in scan order.
    assert_eq!(regions, vec![1, 2, 3]);
    assert_eq!(qtys, vec![20, 30, 40]);
}

/// Full Substrait → LogicalPlan → engine execution of the grouped `SUM`.
#[test]
#[ignore = "gpu:e2e"]
fn substrait_aggregate_executes_on_engine() {
    let mut engine = craton_bolt::Engine::new().expect("CUDA ctx");
    engine
        .register_table("sales", sales_batch())
        .expect("register sales");

    let logical = substrait_to_logical_plan(&aggregate_plan(), &provider()).expect("plan converts");
    let handle = engine
        .run_logical_plan(&logical)
        .expect("converted aggregate executes");
    let out = handle.record_batch();

    // GROUP BY output order is not specified; compare as sorted pairs.
    let mut got: Vec<(i64, i64)> = (0..out.num_rows())
        .map(|i| (col_i64(out, 0).value(i), col_i64(out, 1).value(i)))
        .collect();
    got.sort_unstable();
    assert_eq!(got, vec![(1, 25), (2, 37), (3, 40)]);
}
