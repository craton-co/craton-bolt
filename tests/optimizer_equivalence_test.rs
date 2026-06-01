// SPDX-License-Identifier: Apache-2.0

//! Optimizer-equivalence guard: does running the built-in logical optimizer
//! over a query preserve the query's meaning?
//!
//! Until now there was NO test that compared a query "with the optimizer" vs
//! "without it". That is the one regression that the unit tests per pass cannot
//! catch on their own: each pass can be individually correct, yet a future edit
//! to predicate-pushdown / filter-into-join / join-reorder / constant-fold
//! could *silently* produce wrong results once the passes compose. A wrong
//! rewrite does not crash — it returns a plausible-but-wrong answer — so a
//! dedicated equivalence guard is the right shape of test.
//!
//! ## What this file pins
//!
//! ### Group (a) — HOST-ONLY structural soundness (runs in CI, no GPU)
//!
//! The optimizer operates purely on the logical IR
//! ([`craton_bolt::plan::LogicalPlan`]); none of the passes are GPU-aware. So
//! we can run the real pipeline — [`default_passes`] driven by the real
//! [`run_to_fixpoint`], exactly as the engine does internally — over plans
//! built from SQL via the public [`parse_sql`], and assert two properties that
//! together approximate "the result did not change" without touching a device:
//!
//!   1. **Schema is preserved.** The optimized plan must type-check to the same
//!      output schema (field *names*, *dtypes*, and *nullability*) as the
//!      unoptimized plan. A rewrite that dropped/renamed/retyped a column, or
//!      flipped a nullability flag, would change the query's contract — this
//!      catches that for a representative spread of plan shapes (filters over
//!      INNER / LEFT / RIGHT joins, predicates above aggregates,
//!      const-foldable expressions, projection-prunable plans).
//!
//!   2. **Known-unsound rewrites did NOT happen.** The headline soundness trap
//!      for predicate pushdown is sinking a filter that references the
//!      *nullable* (non-preserved) side of an OUTER join *below* that join —
//!      which silently turns a LEFT join into an INNER-ish join and drops
//!      NULL-padded rows. We assert structurally that a right-side predicate
//!      over a LEFT join stays ABOVE the join after optimization, and a
//!      left-side predicate over a RIGHT join likewise stays above.
//!
//! These run on any machine: `cargo test --test optimizer_equivalence_test`
//! (and under CI's `--no-default-features --features cuda-stub`, since nothing
//! here links a kernel).
//!
//! ### Group (b) — GPU execution-equivalence (compiles, `#[ignore]`'d in CI)
//!
//! The ideal test executes the *same* query twice — once with the optimizer,
//! once without — and asserts identical result rows. **That is not reachable
//! through the public API today:** both [`Engine::sql`] and
//! [`Engine::run_logical_plan`] *always* run `run_to_fixpoint` internally
//! (verified in `src/exec/engine.rs`), and the only optimizer-free execution
//! path (`lower_physical` + the engine's private `execute`) is not public. An
//! "optimizer truly disabled" run would need an internal hook (e.g. an
//! `EngineBuilder::without_optimizer()` toggle, or a `pub` test-only
//! `execute_physical`).
//!
//! The strongest variant we CAN build without that hook is below
//! ([`gpu_optimized_plan_executes_like_raw_plan`]): we hand the engine BOTH the
//! raw SQL-built plan AND a copy we pre-optimized ourselves to fixpoint, run
//! both through [`Engine::run_logical_plan`], and assert the result rows match.
//! Because the engine re-optimizes idempotently, this proves that
//! hand-applying the optimizer before execution does not change results vs.
//! letting the engine do it — i.e. the optimizer is execution-idempotent on a
//! real device. It is the best reachable execution-level guard; upgrade it to a
//! true opt-vs-no-opt comparison the day an optimizer-bypass hook lands.

use craton_bolt::plan::{
    default_passes, parse_sql, DataType, Field, LogicalPlan, MemTableProvider, Schema,
};
use craton_bolt::plan::logical_plan::JoinType;
use craton_bolt::plan::optimizer::run_to_fixpoint;

// ---------------------------------------------------------------------------
// Shared fixtures + helpers
// ---------------------------------------------------------------------------

/// A provider with three small tables whose columns let us write joins,
/// aggregates, const-folds, and projection-prunable selects. Nullability is
/// set deliberately (some columns nullable, some not) so the schema-preservation
/// check actually exercises the nullable flag, not just names/dtypes.
fn provider() -> MemTableProvider {
    let l = Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, true),
        Field::new("c", DataType::Float64, true),
    ]);
    let r = Schema::new(vec![
        Field::new("d", DataType::Int64, false),
        Field::new("e", DataType::Int64, true),
        Field::new("f", DataType::Float64, true),
    ]);
    let s = Schema::new(vec![
        Field::new("g", DataType::Int64, false),
        Field::new("h", DataType::Int64, true),
    ]);
    MemTableProvider::new()
        .with_table("l", l)
        .with_table("r", r)
        .with_table("s", s)
}

/// Build a logical plan from SQL (no optimization applied).
fn raw_plan(sql: &str) -> LogicalPlan {
    parse_sql(sql, &provider()).unwrap_or_else(|e| panic!("parse `{sql}` failed: {e:?}"))
}

/// Run the REAL default optimizer pipeline to a bounded fixpoint — byte-for-byte
/// the pass list and driver the engine uses internally (`default_passes()` +
/// `run_to_fixpoint`, see `src/exec/engine.rs`).
fn optimize(plan: LogicalPlan) -> LogicalPlan {
    run_to_fixpoint(&default_passes(), plan).expect("optimizer pipeline must succeed")
}

/// Type-check a plan to its output schema, attributing failures to the label.
fn schema_of(label: &str, plan: &LogicalPlan) -> Schema {
    plan.schema()
        .unwrap_or_else(|e| panic!("`{label}` failed to type-check: {e:?}"))
}

/// Assert two schemas are equivalent in every externally-visible respect:
/// field count, per-field name, dtype, and nullability — in order.
fn assert_schema_equiv(sql: &str, before: &Schema, after: &Schema) {
    assert_eq!(
        before.fields.len(),
        after.fields.len(),
        "`{sql}`: optimizer changed the output column COUNT\n  before: {before:?}\n  after:  {after:?}"
    );
    for (i, (b, a)) in before.fields.iter().zip(after.fields.iter()).enumerate() {
        assert_eq!(
            b.name, a.name,
            "`{sql}`: column {i} NAME changed: {:?} -> {:?}",
            b.name, a.name
        );
        assert_eq!(
            b.dtype, a.dtype,
            "`{sql}`: column {i} ({}) DTYPE changed: {:?} -> {:?}",
            b.name, b.dtype, a.dtype
        );
        assert_eq!(
            b.nullable, a.nullable,
            "`{sql}`: column {i} ({}) NULLABILITY changed: {} -> {}",
            b.name, b.nullable, a.nullable
        );
    }
}

/// Representative queries spanning the rewrite-sensitive plan shapes.
/// Each must survive optimization with its output schema unchanged.
fn representative_queries() -> Vec<&'static str> {
    vec![
        // const-foldable predicate (1=1 folds to true and drops)
        "SELECT a FROM l WHERE 1 = 1 AND b > 0",
        // const-foldable arithmetic in the predicate
        "SELECT a, c FROM l WHERE a + 0 > 10 AND (2 * 3) = 6",
        // projection-prunable: only a couple of columns survive
        "SELECT a FROM l WHERE b > 5",
        // predicate above an INNER join, references both sides
        "SELECT a, d FROM l JOIN r ON a = d WHERE a > 0 AND b > e",
        // INNER join with a const-foldable conjunct mixed in
        "SELECT a, e FROM l JOIN r ON a = d WHERE (10 - 10) = 0 AND a < 100",
        // predicate above a LEFT join (left-side only -> pushable; soundness ok)
        "SELECT a, e FROM l LEFT JOIN r ON a = d WHERE a > 0",
        // predicate above a RIGHT join (right-side only -> pushable)
        "SELECT a, d FROM l RIGHT JOIN r ON a = d WHERE d > 0",
        // predicate above an aggregate (HAVING-style filter over GROUP BY)
        "SELECT a, SUM(b) FROM l GROUP BY a HAVING SUM(b) > 0",
        // aggregate with a const-foldable group filter
        "SELECT a, COUNT(*) FROM l WHERE (5 > 4) GROUP BY a",
        // three-way INNER join chain (join-reorder candidate; NoStats => no-op)
        "SELECT a, d, g FROM l JOIN r ON a = d JOIN s ON d = g WHERE a > 0",
    ]
}

// ---------------------------------------------------------------------------
// Group (a): host-only structural soundness — RUNS IN CI
// ---------------------------------------------------------------------------

/// The optimized plan must type-check to the SAME output schema as the
/// unoptimized plan, for every representative query. Names, dtypes, and
/// nullability are all compared in order.
#[test]
fn optimizer_preserves_output_schema_for_representative_queries() {
    for sql in representative_queries() {
        let raw = raw_plan(sql);
        let before = schema_of("unoptimized", &raw);
        let opt = optimize(raw);
        let after = schema_of("optimized", &opt);
        assert_schema_equiv(sql, &before, &after);
    }
}

/// Running the optimizer a second time over an already-optimized plan must not
/// change the schema again — a weak but cheap idempotence check on top of the
/// fixpoint driver's own guarantee.
#[test]
fn optimizer_is_schema_idempotent() {
    for sql in representative_queries() {
        let once = optimize(raw_plan(sql));
        let s1 = schema_of("optimized-once", &once);
        let twice = optimize(once);
        let s2 = schema_of("optimized-twice", &twice);
        assert_schema_equiv(sql, &s1, &s2);
    }
}

/// SOUNDNESS TRAP: a predicate that references ONLY the nullable (non-preserved)
/// side of a LEFT join must NOT be pushed below the join. Doing so would drop
/// rows that should have been NULL-padded — silently turning the LEFT join into
/// an INNER-style join. We assert structurally that after optimization the
/// right-side predicate is still applied ABOVE the join (there is a Filter over
/// the LEFT join whose predicate touches a right-side column), and that neither
/// the join's input subtree nor its residual `filter` absorbed it.
#[test]
fn left_join_does_not_push_right_side_predicate_below_join() {
    // `e` belongs to `r`, the NULL-padded side of a LEFT join. `e > 0` must
    // stay above the join.
    let sql = "SELECT a, e FROM l LEFT JOIN r ON a = d WHERE e > 0";
    let opt = optimize(raw_plan(sql));

    let join = find_left_join(&opt)
        .unwrap_or_else(|| panic!("expected a LEFT join to survive in:\n{opt:#?}"));
    // The join must still be a LEFT join (not silently rewritten to INNER).
    if let LogicalPlan::Join {
        join_type, filter, left, right, ..
    } = join
    {
        assert_eq!(
            *join_type,
            JoinType::LeftOuter,
            "LEFT join must not be rewritten to a different join type"
        );
        // The right-side predicate must NOT have been folded into the join
        // residual (that path is only valid for INNER/CROSS).
        if let Some(f) = filter {
            assert!(
                !expr_mentions(f, "e"),
                "right-side predicate `e > 0` was wrongly folded into the LEFT join residual: {f:?}"
            );
        }
        // ...nor pushed into the (non-preserved) right input.
        assert!(
            !subtree_filters_on(right, "e"),
            "right-side predicate `e > 0` was wrongly pushed into the LEFT join's right input"
        );
        // (Left input never legitimately filters on `e`; assert for completeness.)
        assert!(
            !subtree_filters_on(left, "e"),
            "right-side predicate `e > 0` leaked into the LEFT join's left input"
        );
    }

    // And it must still be applied SOMEWHERE above the join.
    assert!(
        filter_on_above_join(&opt, "e"),
        "the `e > 0` predicate vanished entirely — it must remain applied above the LEFT join:\n{opt:#?}"
    );
}

/// Mirror of the LEFT-join trap for RIGHT joins: a predicate over ONLY the
/// (non-preserved) LEFT side must stay ABOVE the join.
#[test]
fn right_join_does_not_push_left_side_predicate_below_join() {
    // `a`/`b` belong to `l`, the NULL-padded side of a RIGHT join. `b > 0`
    // must stay above the join.
    let sql = "SELECT a, d FROM l RIGHT JOIN r ON a = d WHERE b > 0";
    let opt = optimize(raw_plan(sql));

    let join = find_right_join(&opt)
        .unwrap_or_else(|| panic!("expected a RIGHT join to survive in:\n{opt:#?}"));
    if let LogicalPlan::Join {
        join_type, filter, left, ..
    } = join
    {
        assert_eq!(
            *join_type,
            JoinType::RightOuter,
            "RIGHT join must not be rewritten to a different join type"
        );
        if let Some(f) = filter {
            assert!(
                !expr_mentions(f, "b"),
                "left-side predicate `b > 0` was wrongly folded into the RIGHT join residual: {f:?}"
            );
        }
        assert!(
            !subtree_filters_on(left, "b"),
            "left-side predicate `b > 0` was wrongly pushed into the RIGHT join's left input"
        );
    }

    assert!(
        filter_on_above_join(&opt, "b"),
        "the `b > 0` predicate vanished — it must remain applied above the RIGHT join:\n{opt:#?}"
    );
}

/// Conversely, a predicate over the PRESERVED side of an outer join SHOULD be
/// pushable — this is the legitimate optimization the trap above must not block.
/// We only assert it stays a valid, schema-preserving plan (we do not *require*
/// the push, to avoid over-fitting the test to the current pass internals), and
/// that the join type is preserved.
#[test]
fn left_join_preserved_side_predicate_keeps_join_type_and_schema() {
    let sql = "SELECT a, e FROM l LEFT JOIN r ON a = d WHERE a > 0";
    let raw = raw_plan(sql);
    let before = schema_of("unoptimized", &raw);
    let opt = optimize(raw);
    let after = schema_of("optimized", &opt);
    assert_schema_equiv(sql, &before, &after);

    let join = find_left_join(&opt).expect("LEFT join must survive");
    if let LogicalPlan::Join { join_type, .. } = join {
        assert_eq!(*join_type, JoinType::LeftOuter);
    }
}

// ---------------------------------------------------------------------------
// Plan-tree inspection helpers (group a)
// ---------------------------------------------------------------------------

/// Find the first LEFT-outer `Join` node anywhere in the plan.
fn find_left_join(plan: &LogicalPlan) -> Option<&LogicalPlan> {
    find_join_of_type(plan, JoinType::LeftOuter)
}

/// Find the first RIGHT-outer `Join` node anywhere in the plan.
fn find_right_join(plan: &LogicalPlan) -> Option<&LogicalPlan> {
    find_join_of_type(plan, JoinType::RightOuter)
}

fn find_join_of_type(plan: &LogicalPlan, want: JoinType) -> Option<&LogicalPlan> {
    if let LogicalPlan::Join { join_type, .. } = plan {
        if *join_type == want {
            return Some(plan);
        }
    }
    for child in children(plan) {
        if let Some(found) = find_join_of_type(child, want) {
            return Some(found);
        }
    }
    None
}

/// True if there is a `Filter` node *above* some `Join` whose predicate mentions
/// `col` — i.e. the predicate is still applied at or above the join, not sunk
/// below it.
fn filter_on_above_join(plan: &LogicalPlan, col: &str) -> bool {
    if let LogicalPlan::Filter { input, predicate } = plan {
        if expr_mentions(predicate, col) && contains_join(input) {
            return true;
        }
    }
    children(plan).iter().any(|c| filter_on_above_join(c, col))
}

/// True if any `Filter` node *within* `plan` (the subtree, e.g. a join input)
/// has a predicate that mentions `col`. Used to assert a predicate did NOT get
/// pushed into a non-preserved join input.
fn subtree_filters_on(plan: &LogicalPlan, col: &str) -> bool {
    if let LogicalPlan::Filter { predicate, .. } = plan {
        if expr_mentions(predicate, col) {
            return true;
        }
    }
    children(plan).iter().any(|c| subtree_filters_on(c, col))
}

fn contains_join(plan: &LogicalPlan) -> bool {
    if matches!(plan, LogicalPlan::Join { .. }) {
        return true;
    }
    children(plan).iter().any(|c| contains_join(c))
}

/// Immediate logical children of a plan node, for generic recursion.
fn children(plan: &LogicalPlan) -> Vec<&LogicalPlan> {
    match plan {
        LogicalPlan::Scan { .. } => vec![],
        LogicalPlan::Filter { input, .. }
        | LogicalPlan::Project { input, .. }
        | LogicalPlan::Aggregate { input, .. }
        | LogicalPlan::Distinct { input }
        | LogicalPlan::Limit { input, .. }
        | LogicalPlan::Sort { input, .. }
        | LogicalPlan::Window { input, .. } => vec![input.as_ref()],
        LogicalPlan::Join { left, right, .. } => vec![left.as_ref(), right.as_ref()],
        LogicalPlan::SetOp { left, right, .. } => vec![left.as_ref(), right.as_ref()],
        LogicalPlan::Union { inputs } => inputs.iter().collect(),
    }
}

/// Does an expression tree mention a column named `name`? Conservative textual
/// match on the `Debug` rendering — sufficient for our single-letter fixture
/// columns and robust to the exact `Expr` variant shape.
fn expr_mentions(expr: &craton_bolt::plan::Expr, name: &str) -> bool {
    // The Debug form spells columns as `Column("e")`; match that precise token
    // so we don't accidentally hit a substring of some other identifier.
    let needle = format!("Column(\"{name}\")");
    format!("{expr:?}").contains(&needle)
}

// ---------------------------------------------------------------------------
// Group (b): GPU execution-equivalence — COMPILES, IGNORED IN CI
// ---------------------------------------------------------------------------
//
// See the module header for why a true opt-vs-no-opt execution comparison is
// not reachable through the public API today (both `Engine::sql` and
// `Engine::run_logical_plan` always run the optimizer). This is the strongest
// reachable variant: run the raw plan AND a hand-pre-optimized copy through the
// engine and assert identical result rows.

/// Normalised result cell, mirroring `tests/diff_duckdb.rs::Cell` so a null on
/// one side never compares equal to `0`/`""` on the other.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
enum Cell {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

#[allow(dead_code)]
fn close_enough(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    let denom = a.abs().max(b.abs()).max(f64::MIN_POSITIVE);
    (a - b).abs() / denom <= craton_bolt::REL_TOL_TEST
}

/// Execute `plan` on the engine and decode every cell into the normalised
/// [`Cell`] form, sorted into a canonical row order so two result sets that
/// differ only by row order still compare equal.
#[allow(dead_code)]
fn run_and_normalize(
    engine: &mut craton_bolt::Engine,
    plan: &LogicalPlan,
) -> Vec<Vec<Cell>> {
    use arrow_array::{
        Array, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, StringArray,
    };

    let handle = engine
        .run_logical_plan(plan)
        .expect("run_logical_plan must succeed on a GPU host");
    let batch = handle.record_batch();

    let n = batch.num_rows();
    let mut rows: Vec<Vec<Cell>> = Vec::with_capacity(n);
    for row in 0..n {
        let mut cells = Vec::with_capacity(batch.num_columns());
        for col in 0..batch.num_columns() {
            let arr = batch.column(col);
            let cell = if arr.is_null(row) {
                Cell::Null
            } else if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
                Cell::Int(a.value(row))
            } else if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
                Cell::Int(a.value(row) as i64)
            } else if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
                Cell::Float(a.value(row))
            } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
                Cell::Float(a.value(row) as f64)
            } else if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
                Cell::Bool(a.value(row))
            } else if let Some(a) = arr.as_any().downcast_ref::<StringArray>() {
                Cell::Str(a.value(row).to_string())
            } else {
                panic!("unhandled result column dtype at col {col}");
            };
            cells.push(cell);
        }
        rows.push(cells);
    }
    // Canonicalise row order so order-insensitive results still compare equal.
    rows.sort_by(|x, y| format!("{x:?}").cmp(&format!("{y:?}")));
    rows
}

/// Two [`Cell`] grids compare equal cell-by-cell, with float tolerance.
#[allow(dead_code)]
fn grids_equal(a: &[Vec<Cell>], b: &[Vec<Cell>]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (ra, rb) in a.iter().zip(b.iter()) {
        if ra.len() != rb.len() {
            return false;
        }
        for (ca, cb) in ra.iter().zip(rb.iter()) {
            let eq = match (ca, cb) {
                (Cell::Float(x), Cell::Float(y)) => close_enough(*x, *y),
                (Cell::Int(x), Cell::Float(y)) | (Cell::Float(y), Cell::Int(x)) => {
                    close_enough(*x as f64, *y)
                }
                _ => ca == cb,
            };
            if !eq {
                return false;
            }
        }
    }
    true
}

/// GPU execution guard. For each representative query, run the engine on (1) the
/// raw SQL-built plan and (2) a copy we pre-optimized to fixpoint ourselves, and
/// assert the result rows are identical.
///
/// NOTE: the engine re-runs `run_to_fixpoint` on BOTH inputs, so this asserts
/// optimizer *idempotence under execution*, not a true optimizer-on vs
/// optimizer-off comparison — which is unreachable until an optimizer-bypass
/// hook exists (see the module header). It is the strongest end-to-end guard the
/// current public API allows, and it WILL trip if a future rewrite makes the
/// hand-optimized plan execute differently from the engine's own optimization of
/// the raw plan. Promote this to opt-vs-no-opt the day a bypass hook lands.
#[test]
#[ignore = "gpu:opt-equiv — needs a CUDA device; also see header re: true opt/no-opt hook"]
fn gpu_optimized_plan_executes_like_raw_plan() {
    let mut engine = craton_bolt::Engine::new().expect("open CUDA device");

    // Register the same three tables the host fixtures describe, with concrete
    // data, so the SQL above actually executes.
    register_fixture_tables(&mut engine);

    for sql in representative_queries() {
        let raw = parse_sql(sql, engine_provider_schema())
            .unwrap_or_else(|e| panic!("parse `{sql}`: {e:?}"));
        let pre_optimized = optimize(raw.clone());

        let from_raw = run_and_normalize(&mut engine, &raw);
        let from_opt = run_and_normalize(&mut engine, &pre_optimized);

        assert!(
            grids_equal(&from_raw, &from_opt),
            "optimizer changed results for `{sql}`\n  raw:       {from_raw:?}\n  optimized: {from_opt:?}"
        );
    }
}

/// Schema-only provider used for parsing inside the GPU test (the engine owns
/// the real data; parsing only needs column types).
#[allow(dead_code)]
fn engine_provider_schema() -> &'static MemTableProvider {
    // A leaked provider keeps a `'static` borrow for `parse_sql`; this is a
    // test-only one-time allocation.
    use std::sync::OnceLock;
    static P: OnceLock<MemTableProvider> = OnceLock::new();
    P.get_or_init(provider)
}

/// Populate the engine with concrete Arrow batches matching [`provider`]'s
/// schemas so the representative queries execute end-to-end on the device.
#[allow(dead_code)]
fn register_fixture_tables(engine: &mut craton_bolt::Engine) {
    use std::sync::Arc;

    use arrow_array::{Float64Array, Int64Array, RecordBatch};
    use arrow_schema::{DataType as ADt, Field as AField, Schema as ASchema};

    // l(a:i64 not null, b:i64 null, c:f64 null)
    let l_schema = Arc::new(ASchema::new(vec![
        AField::new("a", ADt::Int64, false),
        AField::new("b", ADt::Int64, true),
        AField::new("c", ADt::Float64, true),
    ]));
    let l = RecordBatch::try_new(
        l_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            Arc::new(Int64Array::from(vec![Some(10), None, Some(30), Some(40)])),
            Arc::new(Float64Array::from(vec![
                Some(1.5),
                Some(2.5),
                None,
                Some(4.5),
            ])),
        ],
    )
    .expect("build l batch");

    // r(d:i64 not null, e:i64 null, f:f64 null)
    let r_schema = Arc::new(ASchema::new(vec![
        AField::new("d", ADt::Int64, false),
        AField::new("e", ADt::Int64, true),
        AField::new("f", ADt::Float64, true),
    ]));
    let r = RecordBatch::try_new(
        r_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 5])),
            Arc::new(Int64Array::from(vec![Some(100), None, Some(500)])),
            Arc::new(Float64Array::from(vec![Some(1.0), Some(2.0), Some(5.0)])),
        ],
    )
    .expect("build r batch");

    // s(g:i64 not null, h:i64 null)
    let s_schema = Arc::new(ASchema::new(vec![
        AField::new("g", ADt::Int64, false),
        AField::new("h", ADt::Int64, true),
    ]));
    let s = RecordBatch::try_new(
        s_schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![Some(7), Some(8), None])),
        ],
    )
    .expect("build s batch");

    engine.register_table("l", l).expect("register l");
    engine.register_table("r", r).expect("register r");
    engine.register_table("s", s).expect("register s");
}
