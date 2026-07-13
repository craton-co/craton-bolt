// SPDX-License-Identifier: Apache-2.0

//! Pre-lowering resolution of uncorrelated subqueries.
//!
//! `Expr::ScalarSubquery` and `Expr::InSubquery` parse + type-check in the SQL
//! frontend, and correlated subqueries are rejected there — so every subquery
//! that survives to the engine is *uncorrelated*, meaning its boxed
//! [`LogicalPlan`] is a self-contained, independently-executable query that
//! references no columns from the enclosing query.
//!
//! This module turns those subqueries into plain constants *before* physical
//! lowering. It walks the plan's expressions, executes each subplan via a
//! caller-supplied executor closure, and rewrites:
//!
//! * `ScalarSubquery(subplan)` → the single produced value as an
//!   `Expr::Literal` (0 rows → SQL `NULL`; >1 row → a clean error).
//! * `InSubquery { expr, subquery, negated }` → a boolean fold of equalities
//!   over `expr` (`expr = v1 OR expr = v2 …`, or the negated `<>`/`AND` form).
//!
//! Resolution is *inner-first*: subqueries nested inside another subquery's
//! subplan are resolved when that subplan is executed (the executor closure
//! runs the full engine pipeline, which itself re-enters this pass), and
//! subqueries appearing as siblings recurse normally.
//!
//! # Why a closure rather than a direct `&Engine` dependency?
//!
//! The value-extraction and IN-list-build helpers are pure functions over an
//! Arrow [`RecordBatch`] / `&[Literal]`, with no GPU or engine state, so they
//! are unit-tested on the host. The plan walker is generic over a
//! `FnMut(LogicalPlan) -> BoltResult<RecordBatch>` executor so the engine can
//! inject its `&self` execution path without this module taking an `Engine`
//! dependency.

use std::collections::HashSet;
use std::sync::OnceLock;

use arrow_array::{
    Array, BooleanArray, Date32Array, Decimal128Array, Float32Array, Float64Array, Int32Array,
    Int64Array, RecordBatch, StringArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_schema::{DataType as ArrowDataType, TimeUnit as ArrowTimeUnit};

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{
    AggregateExpr, BinaryOp, Expr, Literal, LogicalPlan, SortExpr, TimeUnit, UnaryOp,
};

/// Upper bound on the number of **distinct** values an `IN`/`NOT IN` subquery
/// may materialise into a host-side membership set.
///
/// Without a cap, `x IN (SELECT high_cardinality_col …)` is a memory-DoS /
/// stack-overflow surface on user-controlled input: [`in_set_from_batch`]
/// would buffer one [`Literal`] per distinct row and [`build_in_predicate`]
/// would fold all of them into a single boolean expression tree that every
/// later recursive pass (resolve / lower / JIT codegen) walks. The cap turns
/// unbounded growth into a clean [`BoltError::Other`] long before the OOM
/// killer (or a stack overflow during a deep recursive walk) gets involved.
///
/// This mirrors `setops::SETOP_HOST_MAX_ROWS` (and `DISTINCT_HOST_MAX_ROWS`):
/// the default (10M) matches those so the host set-building paths share a
/// single resource budget. Overridable at runtime via [`IN_SET_MAX_ROWS_ENV`]
/// (parsed once on first call; see [`in_set_max_rows`]).
const IN_SET_MAX_ROWS: usize = 10_000_000;

/// Environment variable that overrides [`IN_SET_MAX_ROWS`] at runtime. Parsed
/// as a base-10 `usize`; `0` is rejected (it would disable the cap and
/// reintroduce the unbounded-growth bug). On any parse failure a `log::warn!`
/// is emitted and the default is used. Mirrors `CRATON_SETOP_HOST_MAX_ROWS`.
const IN_SET_MAX_ROWS_ENV: &str = "CRATON_IN_SET_MAX_ROWS";

/// Latch for the per-process IN-set host-row cap. First call resolves the env
/// var; subsequent calls hit the cached `usize`. Mirrors the `OnceLock` latch
/// in `setops.rs` / `distinct.rs`.
static IN_SET_MAX_ROWS_CACHE: OnceLock<usize> = OnceLock::new();

/// Resolve the per-process IN-set host-row cap. First call performs the
/// env-var lookup; subsequent calls hit the latch. On any parse failure a
/// one-time `log::warn!` is emitted and the compile-time default
/// [`IN_SET_MAX_ROWS`] is used.
fn in_set_max_rows() -> usize {
    *IN_SET_MAX_ROWS_CACHE.get_or_init(parse_in_set_max_rows_env)
}

/// Pure parser for [`IN_SET_MAX_ROWS_ENV`]. Extracted from the `OnceLock` so
/// tests can exercise the parsing rules without touching the latch. Returns
/// the compile-time default on unset / empty / unparseable / zero values,
/// logging a warning in the unparseable / zero cases. Mirrors
/// `setops::parse_setop_host_max_rows_env`.
fn parse_in_set_max_rows_env() -> usize {
    let raw = match std::env::var(IN_SET_MAX_ROWS_ENV) {
        Ok(v) => v,
        Err(_) => return IN_SET_MAX_ROWS,
    };
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return IN_SET_MAX_ROWS;
    }
    match trimmed.parse::<usize>() {
        Ok(0) => {
            log::warn!(
                "subquery_resolve: {IN_SET_MAX_ROWS_ENV}='0' would disable the host-side cap; \
                 using default of {IN_SET_MAX_ROWS}"
            );
            IN_SET_MAX_ROWS
        }
        Ok(v) => v,
        Err(e) => {
            log::warn!(
                "subquery_resolve: {IN_SET_MAX_ROWS_ENV}='{trimmed}' is not a valid usize ({e}); \
                 using default of {IN_SET_MAX_ROWS}"
            );
            IN_SET_MAX_ROWS
        }
    }
}

/// Extract the value at row `row` of the (single) first column of `batch` as a
/// [`Literal`]. A null at that position yields [`Literal::Null`]. Unsupported
/// Arrow dtypes are rejected with a clean [`BoltError`].
///
/// Supports the dtype set the engine can produce as a subquery output:
/// Int32 / Int64 / Float32 / Float64 / Bool / Utf8 / Date32 / Timestamp
/// (all four resolutions) / Decimal128.
fn literal_from_column(batch: &RecordBatch, row: usize) -> BoltResult<Literal> {
    let col = batch.column(0);
    if col.is_null(row) {
        return Ok(Literal::Null);
    }
    macro_rules! downcast {
        ($ty:ty, $what:literal) => {
            col.as_any().downcast_ref::<$ty>().ok_or_else(|| {
                BoltError::Other(format!(
                    "subquery result column claimed dtype {:?} but did not downcast to {}",
                    col.data_type(),
                    $what
                ))
            })?
        };
    }
    let lit = match col.data_type() {
        ArrowDataType::Int32 => Literal::Int32(downcast!(Int32Array, "Int32Array").value(row)),
        ArrowDataType::Int64 => Literal::Int64(downcast!(Int64Array, "Int64Array").value(row)),
        ArrowDataType::Float32 => {
            Literal::Float32(downcast!(Float32Array, "Float32Array").value(row))
        }
        ArrowDataType::Float64 => {
            Literal::Float64(downcast!(Float64Array, "Float64Array").value(row))
        }
        ArrowDataType::Boolean => Literal::Bool(downcast!(BooleanArray, "BooleanArray").value(row)),
        ArrowDataType::Utf8 => {
            Literal::Utf8(downcast!(StringArray, "StringArray").value(row).to_string())
        }
        ArrowDataType::Date32 => Literal::Date32(downcast!(Date32Array, "Date32Array").value(row)),
        ArrowDataType::Decimal128(p, s) => {
            let v = downcast!(Decimal128Array, "Decimal128Array").value(row);
            Literal::Decimal128(v, *p, *s)
        }
        ArrowDataType::Timestamp(unit, tz) => {
            let ticks = match unit {
                ArrowTimeUnit::Second => {
                    downcast!(TimestampSecondArray, "TimestampSecondArray").value(row)
                }
                ArrowTimeUnit::Millisecond => {
                    downcast!(TimestampMillisecondArray, "TimestampMillisecondArray").value(row)
                }
                ArrowTimeUnit::Microsecond => {
                    downcast!(TimestampMicrosecondArray, "TimestampMicrosecondArray").value(row)
                }
                ArrowTimeUnit::Nanosecond => {
                    downcast!(TimestampNanosecondArray, "TimestampNanosecondArray").value(row)
                }
            };
            let plan_unit = crate::exec::schema_convert::arrow_time_unit_to_plan(unit);
            Literal::timestamp_with_tz(ticks, plan_unit, tz.as_deref().map(|s| s.to_string()))
        }
        other => {
            return Err(BoltError::Plan(format!(
                "subquery result dtype {other:?} is not supported for constant folding"
            )))
        }
    };
    Ok(lit)
}

/// Reduce a scalar-subquery result `batch` to a single [`Literal`].
///
/// Contract (SQL scalar subquery):
/// * the batch must have **exactly one column** (the frontend already
///   type-checks this, but we re-verify defensively);
/// * **0 rows** → SQL `NULL` ([`Literal::Null`]);
/// * **1 row** → that value;
/// * **>1 row** → a clean [`BoltError`] (scalar subquery returned more than
///   one row).
pub fn scalar_value_from_batch(batch: &RecordBatch) -> BoltResult<Literal> {
    if batch.num_columns() != 1 {
        return Err(BoltError::Plan(format!(
            "scalar subquery must return exactly one column, got {}",
            batch.num_columns()
        )));
    }
    match batch.num_rows() {
        0 => Ok(Literal::Null),
        1 => literal_from_column(batch, 0),
        n => Err(BoltError::Plan(format!(
            "scalar subquery returned {n} rows; expected at most one"
        ))),
    }
}

/// Collect the **distinct** values of the (single) first column of `batch` as
/// [`Literal`]s, preserving first-seen order.
///
/// The batch must have exactly one column. `NULL`s are collected as
/// [`Literal::Null`] (at most one, deduped like any other value) so the
/// IN-list builder can reason about their presence; see
/// [`build_in_predicate`] for how SQL three-valued `NULL` membership is
/// handled.
pub fn in_set_from_batch(batch: &RecordBatch) -> BoltResult<Vec<Literal>> {
    in_set_from_batch_capped(batch, in_set_max_rows())
}

/// Cap-parameterised core of [`in_set_from_batch`]. Split out so the cap can be
/// exercised in unit tests without going through the process-wide `OnceLock`
/// latch in [`in_set_max_rows`].
fn in_set_from_batch_capped(batch: &RecordBatch, max_rows: usize) -> BoltResult<Vec<Literal>> {
    if batch.num_columns() != 1 {
        return Err(BoltError::Plan(format!(
            "IN subquery must return exactly one column, got {}",
            batch.num_columns()
        )));
    }
    // Up-front capacity is clamped to `min(n_rows, max_rows)` so a giant
    // `num_rows` whose distinct cardinality is tiny can't drive a multi-GiB
    // reservation (mirrors `setops`).
    let cap_hint = batch.num_rows().min(max_rows);
    let mut out: Vec<Literal> = Vec::with_capacity(cap_hint);
    // O(1)-amortised dedup keyed on a hashable form of the literal, replacing
    // the previous O(N^2) `out.iter().any(..)` linear scan per row that made a
    // high-cardinality subquery quadratic to materialise.
    let mut seen: HashSet<LiteralKey> = HashSet::with_capacity(cap_hint);
    for row in 0..batch.num_rows() {
        let lit = literal_from_column(batch, row)?;
        if seen.insert(LiteralKey::from(&lit)) {
            out.push(lit);
            // Bound the *distinct* set size: a source full of duplicates still
            // completes; only the distinct cardinality is capped. Fires when
            // the distinct count crosses the cap, converting an unbounded
            // membership set into a clean error (memory-DoS / deep-recursion
            // stack-overflow guard — see [`IN_SET_MAX_ROWS`]).
            if out.len() > max_rows {
                return Err(BoltError::Other(format!(
                    "IN/NOT IN subquery produced more than {max_rows} distinct values; \
                     LIMIT the subquery or rewrite as a join (override via {IN_SET_MAX_ROWS_ENV})"
                )));
            }
        }
    }
    Ok(out)
}

/// A hashable, `Eq`-able projection of a [`Literal`] used purely to dedup the
/// IN-subquery value set in [`in_set_from_batch`].
///
/// [`Literal`] is not `Eq`/`Hash` (it carries `f32`/`f64`), so we key on the
/// raw bit pattern of the floats. Two `Null`s map to the same key (deduped
/// like any other value, matching the previous `literal_eq` behaviour). NaN
/// floats hash by their bit pattern: identical NaN encodings collapse to one
/// entry, which is harmless for the `=` fold we build (a NaN probe never
/// matches a NaN literal under SQL `=` anyway). A `Decimal128` is keyed on its
/// raw `i128` value *and* its precision/scale so two numerically-equal but
/// differently-scaled decimals are not wrongly merged. Timestamps key on the
/// tick / unit / (interned) tz pointer triple.
#[derive(PartialEq, Eq, Hash)]
enum LiteralKey {
    Null,
    Bool(bool),
    Int32(i32),
    Int64(i64),
    /// `f32` keyed by raw bits (so `Hash`/`Eq` are well-defined).
    Float32(u32),
    /// `f64` keyed by raw bits.
    Float64(u64),
    Utf8(String),
    Decimal128(i128, u8, i8),
    Date32(i32),
    Timestamp(i64, TimeUnit, Option<&'static str>),
}

impl From<&Literal> for LiteralKey {
    fn from(lit: &Literal) -> Self {
        match lit {
            Literal::Null => LiteralKey::Null,
            Literal::Bool(b) => LiteralKey::Bool(*b),
            Literal::Int32(v) => LiteralKey::Int32(*v),
            Literal::Int64(v) => LiteralKey::Int64(*v),
            Literal::Float32(v) => LiteralKey::Float32(v.to_bits()),
            Literal::Float64(v) => LiteralKey::Float64(v.to_bits()),
            Literal::Utf8(s) => LiteralKey::Utf8(s.clone()),
            Literal::Decimal128(v, p, s) => LiteralKey::Decimal128(*v, *p, *s),
            Literal::Date32(v) => LiteralKey::Date32(*v),
            Literal::Timestamp(ticks, unit, tz) => LiteralKey::Timestamp(*ticks, *unit, *tz),
        }
    }
}

/// Fold `leaves` (each a comparison over the probe) into a single boolean
/// expression with `op` (`Or` for `IN`, `And` for `NOT IN`), as a **balanced**
/// binary tree.
///
/// The previous implementation built a *left-deep* chain (`((a op b) op c)…`)
/// whose nesting depth equals `leaves.len()`. A high-cardinality subquery
/// (millions of distinct values) therefore produced a tree millions of nodes
/// deep, and every later *recursive* pass over the expression — `resolve_expr`,
/// physical lowering, JIT codegen — would blow the host stack walking it. A
/// balanced tree bounds the depth to `O(log N)`, so the same N leaves are safe
/// to recurse over. (The total node count is unchanged; only the depth shrinks.)
///
/// `leaves` must be non-empty. With a single leaf the result is that leaf
/// (no wrapper node).
fn fold_balanced(mut leaves: Vec<Expr>, op: BinaryOp) -> Expr {
    debug_assert!(!leaves.is_empty(), "fold_balanced requires >= 1 leaf");
    // Pairwise reduction: combine adjacent nodes level by level until one
    // remains. Each pass halves the count, giving a tree of depth ceil(log2 N).
    while leaves.len() > 1 {
        let mut next = Vec::with_capacity(leaves.len().div_ceil(2));
        let mut it = leaves.into_iter();
        while let Some(left) = it.next() {
            match it.next() {
                Some(right) => next.push(Expr::Binary {
                    op,
                    left: Box::new(left),
                    right: Box::new(right),
                }),
                // Odd one out: carry it up to the next level unchanged.
                None => next.push(left),
            }
        }
        leaves = next;
    }
    leaves.pop().expect("non-empty leaves")
}

/// Build the boolean expression that replaces an `expr [NOT] IN (subquery)`
/// node once the subquery's value set is known.
///
/// `values` is the distinct set produced by [`in_set_from_batch`]. The shape
/// of the membership test:
///
/// * **`IN` (not negated):** `expr = v1 OR expr = v2 OR …` (balanced tree).
/// * **`NOT IN` (negated):** `expr <> v1 AND expr <> v2 AND …` (balanced tree).
///
/// `filter_context` selects between two *correct* lowerings that differ only
/// in how a SQL `UNKNOWN` (NULL) result is represented:
///
/// * `filter_context == true` — the predicate sits at (or on the boolean
///   conjunction/disjunction spine of) a `WHERE` clause, where a row is kept
///   iff the predicate is `TRUE`; `FALSE` and `UNKNOWN` are both dropped. Here
///   we may legitimately collapse `UNKNOWN` to `FALSE` (cheaper folds, and the
///   `NOT IN` path can use the GPU-friendly `IS NOT NULL` guard described
///   below).
/// * `filter_context == false` — the predicate's *value* is observed (e.g.
///   `SELECT x IN (sub)`, `CASE WHEN x IN (sub) …`, or under an explicit
///   `NOT (x IN (sub))`), so `UNKNOWN` must be emitted as a genuine NULL, never
///   silently turned into `FALSE`. Collapsing to `FALSE` here would be a
///   correctness bug (e.g. `NOT (x IN (NULL-set))` would yield `TRUE` instead
///   of `UNKNOWN`).
///
/// # NULL handling (strict SQL three-valued logic)
///
/// Let `S+` be the non-NULL elements of `values` and `has_null` whether
/// `values` contains a NULL.
///
/// * **`IN`**: `x IN S` is `TRUE` if `x` matches some `v in S+`, else
///   `UNKNOWN` if `has_null` (the NULL element makes the membership unknown),
///   else `FALSE`. We build `OR(x = v for v in S+)`; this already gives
///   `TRUE`/`FALSE`/`UNKNOWN(NULL probe)` correctly. When `has_null` and we are
///   **not** in filter context we OR in a bare `NULL` literal so a non-match
///   becomes `UNKNOWN` rather than `FALSE` (`TRUE OR NULL = TRUE`,
///   `FALSE OR NULL = NULL`). In filter context the `NULL` is omitted (FALSE
///   and UNKNOWN are filtered identically).
/// * **`NOT IN`**: `x NOT IN S` is `FALSE` if `x` matches some `v in S+`, else
///   `UNKNOWN` if `has_null`, else `TRUE`/`UNKNOWN(NULL probe)`. We build
///   `AND(x <> v for v in S+)`; when `has_null` and not in filter context we
///   AND in a bare `NULL` so a non-match becomes `UNKNOWN`
///   (`FALSE AND NULL = FALSE`, `TRUE AND NULL = NULL`). In filter context a
///   set containing any NULL can never make the predicate `TRUE`, so we fold
///   straight to `Bool(false)` (no row passes).
///
/// Empty `S+`: with no non-NULL elements there is nothing to compare against.
/// `x IN ()` is `FALSE` and `x NOT IN ()` is `TRUE` for a *truly* empty set.
/// If instead the set was non-empty but all-NULL, strict SQL says `UNKNOWN`;
/// outside filter context we emit a bare `NULL` for that case, while in filter
/// context `FALSE`/`TRUE` (IN/NOT IN) is sound because UNKNOWN filters like
/// FALSE and the already-handled negated-with-NULL branch folded NOT IN to
/// `Bool(false)`.
pub fn build_in_predicate(
    expr: &Expr,
    values: &[Literal],
    negated: bool,
    filter_context: bool,
) -> Expr {
    // Does the value set contain a NULL? Under SQL 3VL, equality / inequality
    // against a NULL literal yields UNKNOWN, never TRUE.
    let set_has_null = values.iter().any(|l| matches!(l, Literal::Null));

    // Strict `NOT IN` under a WHERE filter: a NULL anywhere in the set makes
    // the predicate UNKNOWN for every row (a match → FALSE, a non-match →
    // NULL), so no row passes. Collapsing to `Bool(false)` is sound ONLY when
    // UNKNOWN filters like FALSE — i.e. in filter context. Outside it we must
    // preserve the UNKNOWN (handled by the general path below).
    if negated && set_has_null && filter_context {
        return Expr::Literal(Literal::Bool(false));
    }

    // Non-NULL elements: equality / inequality against a NULL literal is never
    // TRUE in SQL (it is UNKNOWN), so a NULL element never contributes a
    // comparison leaf — its only effect is to poison non-matches to UNKNOWN,
    // captured by the `set_has_null` handling below.
    let non_null: Vec<&Literal> = values
        .iter()
        .filter(|l| !matches!(l, Literal::Null))
        .collect();

    if non_null.is_empty() {
        // No comparison leaves to build.
        if set_has_null && !filter_context {
            // Non-empty all-NULL set, value observed: `x IN (NULL)` /
            // `x NOT IN (NULL)` are both UNKNOWN for every row → emit NULL.
            return Expr::Literal(Literal::Null);
        }
        // Truly empty set (subquery returned 0 rows): `IN` → FALSE,
        // `NOT IN` → TRUE — correct for every probe, NULL included. Also the
        // filter-context all-NULL case: UNKNOWN filters like the constant we
        // emit here (`IN`→FALSE drops the row; `NOT IN`→TRUE was already
        // intercepted above for the has_null branch, so here `set_has_null`
        // is false and TRUE is exactly right).
        return Expr::Literal(Literal::Bool(negated));
    }

    let (cmp_op, fold_op) = if negated {
        (BinaryOp::NotEq, BinaryOp::And)
    } else {
        (BinaryOp::Eq, BinaryOp::Or)
    };

    // One comparison leaf per non-NULL element, folded into a BALANCED tree so
    // the nesting depth is O(log N) rather than O(N) (see `fold_balanced`).
    let leaves: Vec<Expr> = non_null
        .into_iter()
        .map(|v| Expr::Binary {
            op: cmp_op,
            left: Box::new(expr.clone()),
            right: Box::new(Expr::Literal(v.clone())),
        })
        .collect();
    let folded = fold_balanced(leaves, fold_op);

    if negated {
        if set_has_null {
            // Reachable only outside filter context (the filter-context
            // has_null case folded to `Bool(false)` above). Strict 3VL: with a
            // NULL in the set the predicate is FALSE on a match and UNKNOWN on
            // a non-match — never TRUE. `AND NULL` over the inequality fold
            // does exactly this: `FALSE AND NULL = FALSE` (match),
            // `TRUE AND NULL = NULL` (non-match), `NULL AND NULL = NULL` (NULL
            // probe). No `IS NOT NULL` guard: that guard forces NULL→FALSE,
            // which is wrong when the value is observed.
            Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(folded),
                right: Box::new(Expr::Literal(Literal::Null)),
            }
        } else if filter_context {
            // NULL-free `NOT IN` under WHERE. SQL 3VL: `x NOT IN (set)` is
            // UNKNOWN (→ row excluded) when `x` itself is NULL. The lowered
            // `expr <> v AND …` does NOT capture this on the GPU: the `<>`
            // comparator reads a NULL probe as its raw stored value (e.g. 0)
            // and would wrongly include the row. AND in an explicit
            // `expr IS NOT NULL` guard so NULL probe rows are dropped. This is
            // only valid in filter context, where forcing the UNKNOWN result
            // to FALSE is indistinguishable from the correct UNKNOWN (both
            // exclude the row).
            Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(folded),
                right: Box::new(Expr::Unary {
                    op: UnaryOp::IsNotNull,
                    operand: Box::new(expr.clone()),
                }),
            }
        } else {
            // NULL-free `NOT IN`, value observed. The bare `expr <> v AND …`
            // fold is already correct 3VL: a NULL probe yields
            // `NULL <> v` = UNKNOWN, AND-folded to UNKNOWN. No guard (it would
            // corrupt UNKNOWN into FALSE).
            folded
        }
    } else if set_has_null && !filter_context {
        // Non-negated `IN` with a NULL in the set, value observed. Strict 3VL:
        // TRUE on a match, UNKNOWN otherwise (never FALSE). `OR NULL` over the
        // equality fold: `TRUE OR NULL = TRUE` (match), `FALSE OR NULL = NULL`
        // (non-match), `NULL OR NULL = NULL` (NULL probe). In filter context
        // this `OR NULL` is omitted: UNKNOWN and FALSE filter identically.
        Expr::Binary {
            op: BinaryOp::Or,
            left: Box::new(folded),
            right: Box::new(Expr::Literal(Literal::Null)),
        }
    } else {
        // Non-negated `IN`: the equality fold is correct 3VL on its own
        // (TRUE on match, FALSE on non-match, UNKNOWN on NULL probe). In filter
        // context any set-NULL is irrelevant (UNKNOWN filters like FALSE).
        folded
    }
}

/// Recursively resolve every subquery in `plan`, executing subplans via
/// `exec`.
///
/// `exec` runs a self-contained [`LogicalPlan`] end-to-end and returns its
/// result [`RecordBatch`]. The executor is expected to itself route through
/// the engine pipeline (including *this* pass), which is what makes nested
/// subqueries resolve inner-first.
pub fn resolve_plan<F>(plan: LogicalPlan, exec: &mut F) -> BoltResult<LogicalPlan>
where
    F: FnMut(LogicalPlan) -> BoltResult<RecordBatch>,
{
    Ok(match plan {
        LogicalPlan::Scan { .. } => plan,
        LogicalPlan::Filter { input, predicate } => LogicalPlan::Filter {
            input: Box::new(resolve_plan(*input, exec)?),
            // The predicate is consumed by a WHERE-style filter: a row is kept
            // iff the predicate is TRUE, so SQL UNKNOWN filters identically to
            // FALSE. This is the one position where the IN/NOT-IN fold may use
            // its WHERE-only shortcuts — see `build_in_predicate`.
            predicate: resolve_expr_ctx(predicate, exec, true)?,
        },
        LogicalPlan::Project { input, exprs } => LogicalPlan::Project {
            input: Box::new(resolve_plan(*input, exec)?),
            exprs: resolve_exprs(exprs, exec)?,
        },
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => LogicalPlan::Aggregate {
            input: Box::new(resolve_plan(*input, exec)?),
            group_by: resolve_exprs(group_by, exec)?,
            aggregates: aggregates
                .into_iter()
                .map(|a| resolve_aggregate(a, exec))
                .collect::<BoltResult<Vec<_>>>()?,
        },
        LogicalPlan::Distinct { input } => LogicalPlan::Distinct {
            input: Box::new(resolve_plan(*input, exec)?),
        },
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => LogicalPlan::Limit {
            input: Box::new(resolve_plan(*input, exec)?),
            limit,
            offset,
        },
        LogicalPlan::Sort { input, sort_exprs } => LogicalPlan::Sort {
            input: Box::new(resolve_plan(*input, exec)?),
            sort_exprs: sort_exprs
                .into_iter()
                .map(|s| {
                    Ok::<SortExpr, BoltError>(SortExpr {
                        expr: resolve_expr(s.expr, exec)?,
                        descending: s.descending,
                        nulls_first: s.nulls_first,
                    })
                })
                .collect::<BoltResult<Vec<_>>>()?,
        },
        LogicalPlan::Window {
            input,
            window_exprs,
            partition_by,
            order_by,
        } => LogicalPlan::Window {
            input: Box::new(resolve_plan(*input, exec)?),
            // WindowExpr's inner argument is a column/expr that the SQL
            // frontend does not currently allow a subquery inside; the
            // partition/order keys are plain exprs we still walk for safety.
            window_exprs,
            partition_by: resolve_exprs(partition_by, exec)?,
            order_by: order_by
                .into_iter()
                .map(|s| {
                    Ok::<SortExpr, BoltError>(SortExpr {
                        expr: resolve_expr(s.expr, exec)?,
                        descending: s.descending,
                        nulls_first: s.nulls_first,
                    })
                })
                .collect::<BoltResult<Vec<_>>>()?,
        },
        LogicalPlan::Union { inputs } => LogicalPlan::Union {
            inputs: inputs
                .into_iter()
                .map(|p| resolve_plan(p, exec))
                .collect::<BoltResult<Vec<_>>>()?,
        },
        LogicalPlan::SetOp {
            left,
            right,
            op,
            all,
        } => LogicalPlan::SetOp {
            left: Box::new(resolve_plan(*left, exec)?),
            right: Box::new(resolve_plan(*right, exec)?),
            op,
            all,
        },
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
            filter,
        } => LogicalPlan::Join {
            left: Box::new(resolve_plan(*left, exec)?),
            right: Box::new(resolve_plan(*right, exec)?),
            join_type,
            on: on
                .into_iter()
                .map(|(l, r)| Ok::<_, BoltError>((resolve_expr(l, exec)?, resolve_expr(r, exec)?)))
                .collect::<BoltResult<Vec<_>>>()?,
            // The residual join `filter` is WHERE-style: the joined row is kept
            // iff it is TRUE, so UNKNOWN filters like FALSE here too.
            filter: filter
                .map(|f| resolve_expr_ctx(f, exec, true))
                .transpose()?,
        },
    })
}

/// Resolve every expression in a `Vec`.
fn resolve_exprs<F>(exprs: Vec<Expr>, exec: &mut F) -> BoltResult<Vec<Expr>>
where
    F: FnMut(LogicalPlan) -> BoltResult<RecordBatch>,
{
    exprs.into_iter().map(|e| resolve_expr(e, exec)).collect()
}

/// Resolve the inner expression(s) of an [`AggregateExpr`].
fn resolve_aggregate<F>(agg: AggregateExpr, exec: &mut F) -> BoltResult<AggregateExpr>
where
    F: FnMut(LogicalPlan) -> BoltResult<RecordBatch>,
{
    Ok(match agg {
        AggregateExpr::Count(e) => AggregateExpr::Count(resolve_expr(e, exec)?),
        AggregateExpr::Sum(e) => AggregateExpr::Sum(resolve_expr(e, exec)?),
        AggregateExpr::Min(e) => AggregateExpr::Min(resolve_expr(e, exec)?),
        AggregateExpr::Max(e) => AggregateExpr::Max(resolve_expr(e, exec)?),
        AggregateExpr::Avg(e) => AggregateExpr::Avg(resolve_expr(e, exec)?),
        AggregateExpr::VarPop(e) => AggregateExpr::VarPop(Box::new(resolve_expr(*e, exec)?)),
        AggregateExpr::VarSamp(e) => AggregateExpr::VarSamp(Box::new(resolve_expr(*e, exec)?)),
        AggregateExpr::StddevPop(e) => AggregateExpr::StddevPop(Box::new(resolve_expr(*e, exec)?)),
        AggregateExpr::StddevSamp(e) => {
            AggregateExpr::StddevSamp(Box::new(resolve_expr(*e, exec)?))
        }
    })
}

/// Recursively resolve subqueries in a single [`Expr`] that is **not** in a
/// WHERE-style filter context (its boolean value is observed). Thin wrapper
/// over [`resolve_expr_ctx`] with `filter_context = false` for the many plan
/// positions where that is the right default (projections, sort/group keys,
/// join equi-keys, …).
fn resolve_expr<F>(expr: Expr, exec: &mut F) -> BoltResult<Expr>
where
    F: FnMut(LogicalPlan) -> BoltResult<RecordBatch>,
{
    resolve_expr_ctx(expr, exec, false)
}

/// Recursively resolve subqueries in a single [`Expr`], tracking whether the
/// expression sits in a WHERE-style **filter context** (a position where a row
/// is kept iff the value is `TRUE`, so SQL `UNKNOWN` is indistinguishable from
/// `FALSE`).
///
/// For the two subquery variants the subplan is itself run through
/// `resolve_plan` first (inner subqueries resolve before the outer one
/// executes), then executed via `exec`, then folded to a constant.
///
/// `filter_context` is threaded down the **boolean connective spine** only:
///
/// * It is preserved across `AND` / `OR` (at a WHERE root, a sub-result being
///   `FALSE` vs `UNKNOWN` never changes whether the row is ultimately kept —
///   `FALSE AND x` and `NULL AND x` both drop, `_ OR _` likewise) and across
///   an `Alias` wrapper.
/// * It is **reset to `false`** the moment the value is otherwise observed:
///   under a `NOT` (negation turns "drop on UNKNOWN" into "keep on UNKNOWN" —
///   the classic `NOT (x IN sub)` footgun), inside a `CASE` (its WHEN/THEN
///   values are read), under a comparison / arithmetic operator, a cast, a
///   scalar function, etc.
///
/// The flag only affects how `Expr::InSubquery` folds (see
/// [`build_in_predicate`]); every other variant just propagates it so a nested
/// `InSubquery` deeper in the spine sees the right context.
fn resolve_expr_ctx<F>(expr: Expr, exec: &mut F, filter_context: bool) -> BoltResult<Expr>
where
    F: FnMut(LogicalPlan) -> BoltResult<RecordBatch>,
{
    Ok(match expr {
        Expr::Column(_) | Expr::Literal(_) => expr,
        Expr::Binary { op, left, right } => {
            // `AND` / `OR` keep the row-keep semantics of the WHERE root, so a
            // nested IN/NOT-IN on this spine may still use the filter-context
            // shortcut. Any other operator (comparison, arithmetic, …) observes
            // the boolean value, so the operands are NOT in filter context.
            let child_ctx = filter_context && matches!(op, BinaryOp::And | BinaryOp::Or);
            Expr::Binary {
                op,
                left: Box::new(resolve_expr_ctx(*left, exec, child_ctx)?),
                right: Box::new(resolve_expr_ctx(*right, exec, child_ctx)?),
            }
        }
        Expr::Unary { op, operand } => {
            // `NOT` flips row-keep semantics (UNKNOWN would now be *kept*), so
            // its operand is no longer in filter context. `IS [NOT] NULL`
            // observe the value too. Conservatively drop the flag for any unary.
            Expr::Unary {
                op,
                operand: Box::new(resolve_expr_ctx(*operand, exec, false)?),
            }
        }
        Expr::Case {
            branches,
            else_branch,
        } => Expr::Case {
            // CASE reads its WHEN conditions and THEN/ELSE values, so none of
            // them are in filter context.
            branches: branches
                .into_iter()
                .map(|(w, t)| Ok::<_, BoltError>((resolve_expr(w, exec)?, resolve_expr(t, exec)?)))
                .collect::<BoltResult<Vec<_>>>()?,
            else_branch: else_branch
                .map(|e| Ok::<_, BoltError>(Box::new(resolve_expr(*e, exec)?)))
                .transpose()?,
        },
        Expr::Like {
            expr,
            pattern,
            escape,
            negated,
            case_insensitive,
        } => Expr::Like {
            expr: Box::new(resolve_expr(*expr, exec)?),
            pattern,
            escape,
            negated,
            case_insensitive,
        },
        Expr::Cast { expr, target, safe } => Expr::Cast {
            expr: Box::new(resolve_expr(*expr, exec)?),
            target,
            safe,
        },
        Expr::CastFormat {
            expr,
            target,
            pattern,
            to_text,
        } => Expr::CastFormat {
            expr: Box::new(resolve_expr(*expr, exec)?),
            target,
            pattern,
            to_text,
        },
        Expr::ScalarFn { kind, args } => Expr::ScalarFn {
            kind,
            args: resolve_exprs(args, exec)?,
        },
        Expr::Extract { field, expr } => Expr::Extract {
            field,
            expr: Box::new(resolve_expr(*expr, exec)?),
        },
        Expr::DateTrunc { unit, expr } => Expr::DateTrunc {
            unit,
            expr: Box::new(resolve_expr(*expr, exec)?),
        },
        // An alias is transparent: it neither observes nor negates the value,
        // so the filter context flows straight through.
        Expr::Alias(inner, name) => Expr::Alias(
            Box::new(resolve_expr_ctx(*inner, exec, filter_context)?),
            name,
        ),
        Expr::ScalarSubquery(subplan) => {
            // Resolve inner subqueries first, then execute, then fold.
            let resolved = resolve_plan(*subplan, exec)?;
            let batch = exec(resolved)?;
            let lit = scalar_value_from_batch(&batch)?;
            Expr::Literal(lit)
        }
        Expr::InSubquery {
            expr,
            subquery,
            negated,
        } => {
            // The probe `expr` lives in the *outer* query's schema and may
            // itself contain a subquery — resolve it too. The probe's value is
            // observed by the `=`/`<>` comparison, so it is never itself in a
            // filter context regardless of where this InSubquery sits.
            let probe = resolve_expr(*expr, exec)?;
            let resolved_sub = resolve_plan(*subquery, exec)?;
            let batch = exec(resolved_sub)?;
            let values = in_set_from_batch(&batch)?;
            build_in_predicate(&probe, &values, negated, filter_context)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use arrow_array::{Int32Array, Int64Array, StringArray};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    fn single_col_batch(arr: arrow_array::ArrayRef) -> RecordBatch {
        let field = ArrowField::new("c", arr.data_type().clone(), true);
        let schema = Arc::new(ArrowSchema::new(vec![field]));
        RecordBatch::try_new(schema, vec![arr]).unwrap()
    }

    #[test]
    fn scalar_zero_rows_is_null() {
        let arr = Arc::new(Int64Array::from(Vec::<i64>::new())) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        assert_eq!(scalar_value_from_batch(&b).unwrap(), Literal::Null);
    }

    #[test]
    fn scalar_one_row_int64() {
        let arr = Arc::new(Int64Array::from(vec![42_i64])) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        assert_eq!(scalar_value_from_batch(&b).unwrap(), Literal::Int64(42));
    }

    #[test]
    fn scalar_one_row_null_value() {
        let arr = Arc::new(Int32Array::from(vec![None::<i32>])) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        assert_eq!(scalar_value_from_batch(&b).unwrap(), Literal::Null);
    }

    #[test]
    fn scalar_many_rows_errors() {
        let arr = Arc::new(Int64Array::from(vec![1_i64, 2])) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        let err = scalar_value_from_batch(&b).unwrap_err();
        assert!(format!("{err}").contains("returned 2 rows"), "{err}");
    }

    #[test]
    fn scalar_rejects_multi_column() {
        let a = Arc::new(Int64Array::from(vec![1_i64])) as arrow_array::ArrayRef;
        let b = Arc::new(Int64Array::from(vec![2_i64])) as arrow_array::ArrayRef;
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int64, true),
            ArrowField::new("b", ArrowDataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(schema, vec![a, b]).unwrap();
        assert!(scalar_value_from_batch(&batch).is_err());
    }

    #[test]
    fn in_set_dedups_preserving_order() {
        let arr = Arc::new(Int32Array::from(vec![3, 1, 3, 2, 1])) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        let set = in_set_from_batch(&b).unwrap();
        assert_eq!(
            set,
            vec![Literal::Int32(3), Literal::Int32(1), Literal::Int32(2)]
        );
    }

    #[test]
    fn in_set_utf8() {
        let arr = Arc::new(StringArray::from(vec!["x", "y", "x"])) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        let set = in_set_from_batch(&b).unwrap();
        assert_eq!(
            set,
            vec![Literal::Utf8("x".into()), Literal::Utf8("y".into())]
        );
    }

    #[test]
    fn build_in_empty_set() {
        let probe = Expr::Column("x".into());
        // Truly empty set (0-row subquery): `IN` → FALSE, `NOT IN` → TRUE in
        // BOTH contexts (these constants are exact, not WHERE-only shortcuts).
        for fc in [false, true] {
            assert!(matches!(
                build_in_predicate(&probe, &[], false, fc),
                Expr::Literal(Literal::Bool(false))
            ));
            assert!(matches!(
                build_in_predicate(&probe, &[], true, fc),
                Expr::Literal(Literal::Bool(true))
            ));
        }
    }

    #[test]
    fn build_in_only_nulls_set() {
        let probe = Expr::Column("x".into());
        // Under WHERE (filter context): a set of only NULLs collapses to the
        // empty non-null case → `Bool(false)` (UNKNOWN filters like FALSE).
        assert!(matches!(
            build_in_predicate(&probe, &[Literal::Null], false, true),
            Expr::Literal(Literal::Bool(false))
        ));
    }

    #[test]
    fn build_in_or_of_equalities() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(&probe, &[Literal::Int32(1), Literal::Int32(2)], false, true);
        // `Expr` doesn't implement `PartialEq`, so destructure and compare the
        // structure / scalar leaves (which do) instead of `assert_eq!`.
        match got {
            Expr::Binary {
                op: BinaryOp::Or,
                left,
                right,
            } => {
                check_cmp(&left, "x", BinaryOp::Eq, Literal::Int32(1));
                check_cmp(&right, "x", BinaryOp::Eq, Literal::Int32(2));
            }
            other => panic!("expected OR of equalities, got {other:?}"),
        }
    }

    /// Asserts `e` is `Binary { op, Column(col), Literal(lit) }`.
    fn check_cmp(e: &Expr, col: &str, op: BinaryOp, lit: Literal) {
        match e {
            Expr::Binary {
                op: got_op,
                left,
                right,
            } => {
                assert_eq!(*got_op, op, "binary op");
                match (&**left, &**right) {
                    (Expr::Column(name), Expr::Literal(got_lit)) => {
                        assert_eq!(name.as_str(), col, "column name");
                        assert_eq!(*got_lit, lit, "literal");
                    }
                    other => panic!("expected Column op Literal, got {other:?}"),
                }
            }
            other => panic!("expected Binary, got {other:?}"),
        }
    }

    #[test]
    fn build_not_in_and_of_inequalities() {
        let probe = Expr::Column("x".into());
        // Filter context: the WHERE-only `IS NOT NULL` guard is added.
        let got = build_in_predicate(&probe, &[Literal::Int32(1), Literal::Int32(2)], true, true);
        // NOT IN lowers to `(x <> 1 AND x <> 2) AND x IS NOT NULL` — the trailing
        // IS NOT NULL guard drops NULL probe rows (SQL 3VL: NULL NOT IN ... is
        // UNKNOWN → excluded under WHERE).
        match got {
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                // right-hand operand is the IS NOT NULL guard over the probe.
                match &*right {
                    Expr::Unary {
                        op: UnaryOp::IsNotNull,
                        operand,
                    } => match &**operand {
                        Expr::Column(name) => assert_eq!(name.as_str(), "x"),
                        other => panic!("expected Column in IS NOT NULL, got {other:?}"),
                    },
                    other => panic!("expected IS NOT NULL guard, got {other:?}"),
                }
                // left-hand operand is the AND-of-inequalities.
                match &*left {
                    Expr::Binary {
                        op: BinaryOp::And,
                        left: l2,
                        right: r2,
                    } => {
                        check_cmp(l2, "x", BinaryOp::NotEq, Literal::Int32(1));
                        check_cmp(r2, "x", BinaryOp::NotEq, Literal::Int32(2));
                    }
                    other => panic!("expected AND of inequalities, got {other:?}"),
                }
            }
            other => panic!("expected AND with IS NOT NULL guard, got {other:?}"),
        }
    }

    #[test]
    fn in_predicate_drops_nulls_keeps_non_null() {
        let probe = Expr::Column("x".into());
        // Filter context: a set-NULL is irrelevant under WHERE (UNKNOWN filters
        // like FALSE), so it is dropped and a single non-null element → bare
        // equality, no OR fold.
        let got = build_in_predicate(&probe, &[Literal::Int32(7), Literal::Null], false, true);
        check_cmp(&got, "x", BinaryOp::Eq, Literal::Int32(7));
    }

    // ---- F-6: strict SQL 3VL for `NOT IN (subquery)` ----------------------

    /// `x NOT IN (… , NULL , …)`: with a NULL anywhere in the set, the strict
    /// SQL semantics make the predicate UNKNOWN for every row, so NO row
    /// passes. We must fold to `Bool(false)` — never build an `AND` of `<>`
    /// that would let rows through.
    #[test]
    fn not_in_with_null_in_set_excludes_all_rows() {
        let probe = Expr::Column("x".into());
        // Filter context (WHERE): UNKNOWN filters like FALSE, so the whole
        // predicate folds to `Bool(false)` (no rows).
        let got = build_in_predicate(
            &probe,
            &[Literal::Int32(1), Literal::Int32(2), Literal::Null],
            true,
            true,
        );
        assert!(
            matches!(got, Expr::Literal(Literal::Bool(false))),
            "NOT IN with a NULL in the set must yield Bool(false) (no rows), got {got:?}"
        );
    }

    /// A set of *only* NULLs under `NOT IN` is still UNKNOWN for every row →
    /// `Bool(false)` (this is the same SQL footgun as a set containing one
    /// non-NULL plus a NULL).
    #[test]
    fn not_in_with_only_null_set_excludes_all_rows() {
        let probe = Expr::Column("x".into());
        // Filter context (WHERE).
        let got = build_in_predicate(&probe, &[Literal::Null], true, true);
        assert!(
            matches!(got, Expr::Literal(Literal::Bool(false))),
            "NOT IN over an all-NULL set must yield Bool(false), got {got:?}"
        );
    }

    /// `x NOT IN (1, 2)` with NO NULL in the set keeps the normal strict
    /// `<>`/`AND` fold over the non-NULL elements.
    #[test]
    fn not_in_without_null_builds_and_of_inequalities() {
        let probe = Expr::Column("x".into());
        // Filter context keeps the WHERE-only `IS NOT NULL` guard.
        let got = build_in_predicate(&probe, &[Literal::Int32(1), Literal::Int32(2)], true, true);
        // `(x <> 1 AND x <> 2) AND x IS NOT NULL` — the IS NOT NULL guard drops
        // NULL probe rows (SQL 3VL); the inequality fold is over the non-NULL set.
        match got {
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                assert!(
                    matches!(
                        &*right,
                        Expr::Unary {
                            op: UnaryOp::IsNotNull,
                            ..
                        }
                    ),
                    "expected trailing IS NOT NULL guard, got {right:?}"
                );
                match &*left {
                    Expr::Binary {
                        op: BinaryOp::And,
                        left: l2,
                        right: r2,
                    } => {
                        check_cmp(l2, "x", BinaryOp::NotEq, Literal::Int32(1));
                        check_cmp(r2, "x", BinaryOp::NotEq, Literal::Int32(2));
                    }
                    other => panic!("expected AND of inequalities, got {other:?}"),
                }
            }
            other => panic!("expected AND with IS NOT NULL guard, got {other:?}"),
        }
    }

    /// `x IN (… , NULL , …)` (NON-negated) is unaffected by F-6: the NULL is
    /// dropped and the row matches iff it equals a non-NULL element. A NULL in
    /// the set must NOT collapse the IN form to a constant.
    #[test]
    fn in_with_null_in_set_keeps_non_null_membership() {
        let probe = Expr::Column("x".into());
        // Filter context: the set-NULL is dropped under WHERE.
        let got = build_in_predicate(&probe, &[Literal::Int32(7), Literal::Null], false, true);
        // Single non-null element → bare equality (the NULL is dropped).
        check_cmp(&got, "x", BinaryOp::Eq, Literal::Int32(7));
    }

    /// Probe value being NULL is orthogonal to the *set's* NULLs: the predicate
    /// structure is built over the probe expression as-is. A probe `Column`
    /// that resolves to NULL at runtime is handled by the downstream `=`/`<>`
    /// 3VL evaluation, not by `build_in_predicate`. Here we assert the builder
    /// faithfully embeds the (possibly-NULL-valued) probe expression and does
    /// not special-case it, for a NULL-free set.
    #[test]
    fn probe_expr_preserved_for_null_free_set() {
        // A probe that is itself a literal NULL — the builder must still emit
        // the equality fold; runtime 3VL (NULL = v → UNKNOWN) handles exclusion.
        let probe = Expr::Literal(Literal::Null);
        // NULL-free set → context-independent; assert in filter context.
        let got = build_in_predicate(&probe, &[Literal::Int32(3)], false, true);
        match got {
            Expr::Binary {
                op: BinaryOp::Eq,
                left,
                right,
            } => match (&*left, &*right) {
                (Expr::Literal(Literal::Null), Expr::Literal(Literal::Int32(3))) => {}
                other => panic!("expected (NULL = 3), got {other:?}"),
            },
            other => panic!("expected Eq fold over probe, got {other:?}"),
        }
    }

    #[test]
    fn resolve_plan_replaces_scalar_subquery() {
        // Outer plan: Filter(Scan, x = ScalarSubquery(inner)). The executor
        // closure returns a one-row Int32 batch holding 99.
        let inner = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: crate::plan::Schema::new(vec![crate::plan::Field::new(
                "v",
                crate::plan::DataType::Int32,
                false,
            )]),
        };
        let outer = LogicalPlan::Filter {
            input: Box::new(LogicalPlan::Scan {
                table: "s".into(),
                projection: None,
                schema: crate::plan::Schema::new(vec![crate::plan::Field::new(
                    "x",
                    crate::plan::DataType::Int32,
                    false,
                )]),
            }),
            predicate: Expr::Binary {
                op: BinaryOp::Eq,
                left: Box::new(Expr::Column("x".into())),
                right: Box::new(Expr::ScalarSubquery(Box::new(inner))),
            },
        };
        let mut exec = |_p: LogicalPlan| -> BoltResult<RecordBatch> {
            let arr = Arc::new(Int32Array::from(vec![99])) as arrow_array::ArrayRef;
            Ok(single_col_batch(arr))
        };
        let resolved = resolve_plan(outer, &mut exec).unwrap();
        match resolved {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::Binary { right, .. } => match *right {
                    Expr::Literal(lit) => assert_eq!(lit, Literal::Int32(99)),
                    other => panic!("expected folded literal, got {other:?}"),
                },
                other => panic!("unexpected predicate {other:?}"),
            },
            other => panic!("unexpected plan {other:?}"),
        }
    }

    // ---- IN-set size cap (memory-DoS / deep-recursion guard) ---------------

    /// Override the IN-set cap via env around `f`, restoring the prior value.
    /// Because [`in_set_max_rows`] latches in an `OnceLock`, these tests use the
    /// pure parser [`parse_in_set_max_rows_env`] (which re-reads the env every
    /// call) rather than the latched accessor, mirroring `setops`'s approach.
    fn with_in_set_env<R>(val: Option<&str>, f: impl FnOnce() -> R) -> R {
        use std::sync::Mutex;
        static ENV_LOCK: Mutex<()> = Mutex::new(());
        let _guard = ENV_LOCK.lock().unwrap();
        let prev = std::env::var(IN_SET_MAX_ROWS_ENV).ok();
        match val {
            Some(v) => std::env::set_var(IN_SET_MAX_ROWS_ENV, v),
            None => std::env::remove_var(IN_SET_MAX_ROWS_ENV),
        }
        let out = f();
        match prev {
            Some(v) => std::env::set_var(IN_SET_MAX_ROWS_ENV, v),
            None => std::env::remove_var(IN_SET_MAX_ROWS_ENV),
        }
        out
    }

    #[test]
    fn in_set_env_parser_rules() {
        with_in_set_env(None, || {
            assert_eq!(parse_in_set_max_rows_env(), IN_SET_MAX_ROWS)
        });
        with_in_set_env(Some(""), || {
            assert_eq!(parse_in_set_max_rows_env(), IN_SET_MAX_ROWS)
        });
        // `0` is rejected (it would disable the cap) → default.
        with_in_set_env(Some("0"), || {
            assert_eq!(parse_in_set_max_rows_env(), IN_SET_MAX_ROWS)
        });
        with_in_set_env(Some("not-a-number"), || {
            assert_eq!(parse_in_set_max_rows_env(), IN_SET_MAX_ROWS)
        });
        with_in_set_env(Some("7"), || assert_eq!(parse_in_set_max_rows_env(), 7));
    }

    /// The HashSet-based dedup collapses a large duplicate-heavy run into its
    /// distinct values without the old O(N^2) linear-scan blowup. Feeds 50k
    /// rows of only 100 distinct values and asserts the deduped set is exactly
    /// 100 (a quadratic dedup would be ~2.5e9 comparisons).
    #[test]
    fn in_set_dedups_large_duplicate_run_without_quadratic_blowup() {
        // 50k rows but only 100 distinct values: the HashSet dedup collapses
        // them. (A quadratic linear-scan dedup would be ~2.5e9 comparisons.)
        let vals: Vec<i32> = (0..50_000).map(|i| i % 100).collect();
        let arr = Arc::new(Int32Array::from(vals)) as arrow_array::ArrayRef;
        let b = single_col_batch(arr);
        let set = in_set_from_batch(&b).unwrap();
        assert_eq!(set.len(), 100, "distinct set must be exactly {{0..100}}");
    }

    #[test]
    fn in_set_cap_rejects_oversized_distinct_set() {
        // 25 distinct values, cap of 10 → clean error (not a panic / OOM).
        let vals: Vec<i32> = (0..25).collect();
        let b = set_batch(vals.into_iter().map(Some).collect());
        let err = in_set_from_batch_capped(&b, 10).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("more than 10 distinct values"),
            "cap error should name the bound, got: {msg}"
        );
    }

    #[test]
    fn in_set_cap_allows_duplicates_under_distinct_bound() {
        // 50 rows but only 5 DISTINCT values, cap of 10 → completes (only the
        // distinct cardinality is bounded, a duplicate-heavy source is fine).
        let vals: Vec<Option<i32>> = (0..50).map(|i| Some(i % 5)).collect();
        let b = set_batch(vals);
        let set = in_set_from_batch_capped(&b, 10).unwrap();
        assert_eq!(set.len(), 5, "5 distinct values are under the cap of 10");
    }

    #[test]
    fn in_set_cap_boundary_exactly_at_limit_ok() {
        // Exactly `max` distinct values is allowed; `max + 1` errors (the guard
        // is `> max`).
        let exactly: Vec<Option<i32>> = (0..10).map(Some).collect();
        let b = set_batch(exactly);
        assert!(in_set_from_batch_capped(&b, 10).is_ok());
        let over: Vec<Option<i32>> = (0..11).map(Some).collect();
        let b2 = set_batch(over);
        assert!(in_set_from_batch_capped(&b2, 10).is_err());
    }

    // ---- balanced-tree fold (bounded recursion depth) ----------------------

    /// Build the OR-fold over N distinct values and assert the resulting tree
    /// depth is O(log N), not O(N) — the property that keeps the later
    /// recursive walks (resolve / lower / JIT) off a deep-recursion stack
    /// overflow. A left-deep chain over N leaves has depth N.
    #[test]
    fn build_in_fold_is_balanced_logarithmic_depth() {
        fn depth(e: &Expr) -> usize {
            match e {
                Expr::Binary { left, right, .. } => 1 + depth(left).max(depth(right)),
                _ => 0,
            }
        }
        let probe = Expr::Column("x".into());
        let n = 1024usize;
        let values: Vec<Literal> = (0..n as i32).map(Literal::Int32).collect();
        let folded = build_in_predicate(&probe, &values, false, true);
        let d = depth(&folded);
        // ceil(log2(1024)) = 10 for the OR spine, +1 for each Eq leaf = 11.
        // A left-deep chain would be ~1024. Assert it is far below linear.
        assert!(
            d <= 12,
            "balanced fold over {n} values should have depth ~log2(n)+1, got {d}"
        );
    }

    /// `fold_balanced` over a single leaf returns that leaf unwrapped.
    #[test]
    fn fold_balanced_single_leaf_is_unwrapped() {
        let leaf = Expr::Column("x".into());
        let got = fold_balanced(vec![leaf], BinaryOp::Or);
        assert!(matches!(got, Expr::Column(ref n) if n == "x"));
    }

    // ---- SQL-3VL-correct folds OUTSIDE filter context ----------------------

    /// `SELECT x IN (sub)` with a NULL in the set (value observed, NOT a WHERE
    /// filter): strict 3VL is TRUE-on-match, UNKNOWN otherwise — never FALSE.
    /// The builder must OR a bare NULL onto the equality fold so a non-match
    /// becomes NULL (`FALSE OR NULL = NULL`), not FALSE.
    #[test]
    fn in_with_null_set_non_filter_or_nulls() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(&probe, &[Literal::Int32(7), Literal::Null], false, false);
        match got {
            Expr::Binary {
                op: BinaryOp::Or,
                left,
                right,
            } => {
                check_cmp(&left, "x", BinaryOp::Eq, Literal::Int32(7));
                assert!(
                    matches!(&*right, Expr::Literal(Literal::Null)),
                    "expected trailing NULL to poison non-matches, got {right:?}"
                );
            }
            other => panic!("expected (x = 7) OR NULL, got {other:?}"),
        }
    }

    /// `SELECT x IN (NULL)` (all-NULL non-empty set, value observed): every row
    /// is UNKNOWN → a bare NULL literal (NOT Bool(false)).
    #[test]
    fn in_with_only_null_set_non_filter_is_null() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(&probe, &[Literal::Null], false, false);
        assert!(
            matches!(got, Expr::Literal(Literal::Null)),
            "all-NULL IN set with value observed must be NULL, got {got:?}"
        );
    }

    /// Truly empty set (0-row subquery): `x IN ()` is FALSE even with the value
    /// observed — `Bool(false)` is exact here, not a WHERE shortcut.
    #[test]
    fn in_with_empty_set_non_filter_is_false() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(&probe, &[], false, false);
        assert!(matches!(got, Expr::Literal(Literal::Bool(false))));
    }

    /// `SELECT x NOT IN (sub)` with a NULL in the set (value observed): strict
    /// 3VL is FALSE-on-match, UNKNOWN otherwise — never the WHERE shortcut
    /// `Bool(false)`. The builder ANDs a bare NULL onto the inequality fold so
    /// a non-match becomes NULL (`TRUE AND NULL = NULL`), and there is NO
    /// `IS NOT NULL` guard (which would corrupt UNKNOWN into FALSE).
    #[test]
    fn not_in_with_null_set_non_filter_ands_null() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(
            &probe,
            &[Literal::Int32(1), Literal::Int32(2), Literal::Null],
            true,
            false,
        );
        match got {
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                // The poisoning NULL, NOT an IS NOT NULL guard.
                assert!(
                    matches!(&*right, Expr::Literal(Literal::Null)),
                    "expected trailing NULL (not an IS NOT NULL guard), got {right:?}"
                );
                // The left subtree is the AND of the two inequalities.
                match &*left {
                    Expr::Binary {
                        op: BinaryOp::And,
                        left: l2,
                        right: r2,
                    } => {
                        check_cmp(l2, "x", BinaryOp::NotEq, Literal::Int32(1));
                        check_cmp(r2, "x", BinaryOp::NotEq, Literal::Int32(2));
                    }
                    other => panic!("expected AND of inequalities, got {other:?}"),
                }
            }
            other => panic!("expected (x<>1 AND x<>2) AND NULL, got {other:?}"),
        }
    }

    /// `SELECT x NOT IN (1, 2)` (NULL-free, value observed): the bare
    /// `x <> 1 AND x <> 2` fold is already correct 3VL — NO `IS NOT NULL`
    /// guard (that guard is a WHERE-only GPU workaround that would force a NULL
    /// probe's UNKNOWN result to FALSE).
    #[test]
    fn not_in_null_free_non_filter_has_no_guard() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(&probe, &[Literal::Int32(1), Literal::Int32(2)], true, false);
        match got {
            Expr::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                // Both sides are inequalities — there is no IS NOT NULL guard.
                check_cmp(&left, "x", BinaryOp::NotEq, Literal::Int32(1));
                check_cmp(&right, "x", BinaryOp::NotEq, Literal::Int32(2));
            }
            other => panic!("expected bare AND of inequalities, got {other:?}"),
        }
    }

    /// `SELECT x NOT IN (NULL)` (all-NULL, value observed): every row is
    /// UNKNOWN → a bare NULL (NOT Bool(false)).
    #[test]
    fn not_in_only_null_set_non_filter_is_null() {
        let probe = Expr::Column("x".into());
        let got = build_in_predicate(&probe, &[Literal::Null], true, false);
        assert!(
            matches!(got, Expr::Literal(Literal::Null)),
            "all-NULL NOT IN set with value observed must be NULL, got {got:?}"
        );
    }

    // ---- filter-context threading through resolve_plan / resolve_expr ------

    /// Helper: a 1-column Int32 batch with an optional trailing NULL, used as
    /// the IN-subquery result.
    fn set_batch(vals: Vec<Option<i32>>) -> RecordBatch {
        let arr = Arc::new(Int32Array::from(vals)) as arrow_array::ArrayRef;
        single_col_batch(arr)
    }

    fn scan(table: &str, col: &str) -> LogicalPlan {
        LogicalPlan::Scan {
            table: table.into(),
            projection: None,
            schema: crate::plan::Schema::new(vec![crate::plan::Field::new(
                col,
                crate::plan::DataType::Int32,
                true,
            )]),
        }
    }

    fn in_sub(col: &str, negated: bool) -> Expr {
        Expr::InSubquery {
            expr: Box::new(Expr::Column(col.into())),
            subquery: Box::new(scan("other", "id")),
            negated,
        }
    }

    /// In a `Filter` predicate (WHERE), a `NOT IN` with a NULL in the set folds
    /// to the WHERE shortcut `Bool(false)`.
    #[test]
    fn resolve_filter_predicate_uses_filter_context() {
        let plan = LogicalPlan::Filter {
            input: Box::new(scan("t", "k")),
            predicate: in_sub("k", true),
        };
        let mut exec =
            |_p: LogicalPlan| -> BoltResult<RecordBatch> { Ok(set_batch(vec![Some(1), None])) };
        let resolved = resolve_plan(plan, &mut exec).unwrap();
        match resolved {
            LogicalPlan::Filter { predicate, .. } => assert!(
                matches!(predicate, Expr::Literal(Literal::Bool(false))),
                "WHERE NOT IN with set-NULL must fold to Bool(false), got {predicate:?}"
            ),
            other => panic!("unexpected plan {other:?}"),
        }
    }

    /// In a `Project` expression (value observed), the SAME `NOT IN` with a
    /// NULL in the set must NOT use the WHERE shortcut — it folds to the strict
    /// 3VL `(… ) AND NULL` form so the projected value is genuinely NULL.
    #[test]
    fn resolve_projection_does_not_use_filter_context() {
        let plan = LogicalPlan::Project {
            input: Box::new(scan("t", "k")),
            exprs: vec![in_sub("k", true)],
        };
        let mut exec =
            |_p: LogicalPlan| -> BoltResult<RecordBatch> { Ok(set_batch(vec![Some(1), None])) };
        let resolved = resolve_plan(plan, &mut exec).unwrap();
        match resolved {
            LogicalPlan::Project { exprs, .. } => match &exprs[0] {
                Expr::Binary {
                    op: BinaryOp::And,
                    right,
                    ..
                } => assert!(
                    matches!(&**right, Expr::Literal(Literal::Null)),
                    "projected NOT IN with set-NULL must AND in NULL (3VL), got {right:?}"
                ),
                other => panic!("expected (…) AND NULL, got {other:?}"),
            },
            other => panic!("unexpected plan {other:?}"),
        }
    }

    /// An explicit `WHERE NOT (k IN (sub))` reaches `resolve_expr` as a Unary
    /// NOT wrapping a NON-negated `InSubquery` (the sqlparser `negated` flag is
    /// only set for the `NOT IN` spelling). Crossing the `NOT` must RESET the
    /// filter context: with a NULL in the set the inner non-negated IN must
    /// fold to the strict `(x = v) OR NULL` 3VL form, so `NOT (...)` evaluates
    /// to UNKNOWN (excluded) rather than the wrong `NOT FALSE = TRUE`.
    #[test]
    fn resolve_not_wrapped_in_subquery_resets_filter_context() {
        let plan = LogicalPlan::Filter {
            input: Box::new(scan("t", "k")),
            predicate: Expr::Unary {
                op: UnaryOp::Not,
                operand: Box::new(in_sub("k", false)),
            },
        };
        let mut exec =
            |_p: LogicalPlan| -> BoltResult<RecordBatch> { Ok(set_batch(vec![Some(1), None])) };
        let resolved = resolve_plan(plan, &mut exec).unwrap();
        match resolved {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::Unary {
                    op: UnaryOp::Not,
                    operand,
                } => match *operand {
                    // Inner IN with set-NULL, value observed → (k = 1) OR NULL.
                    Expr::Binary {
                        op: BinaryOp::Or,
                        right,
                        ..
                    } => assert!(
                        matches!(*right, Expr::Literal(Literal::Null)),
                        "NOT-wrapped IN must keep 3VL OR-NULL form, got {right:?}"
                    ),
                    other => panic!("expected (k = 1) OR NULL under NOT, got {other:?}"),
                },
                other => panic!("expected Unary NOT, got {other:?}"),
            },
            other => panic!("unexpected plan {other:?}"),
        }
    }

    /// Filter context is preserved across an `AND` spine: `WHERE (k NOT IN sub)
    /// AND (k > 0)` still lets the NOT-IN use the WHERE shortcut.
    #[test]
    fn resolve_filter_context_flows_through_and() {
        let plan = LogicalPlan::Filter {
            input: Box::new(scan("t", "k")),
            predicate: Expr::Binary {
                op: BinaryOp::And,
                left: Box::new(in_sub("k", true)),
                right: Box::new(Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Column("k".into())),
                    right: Box::new(Expr::Literal(Literal::Int32(0))),
                }),
            },
        };
        let mut exec =
            |_p: LogicalPlan| -> BoltResult<RecordBatch> { Ok(set_batch(vec![Some(1), None])) };
        let resolved = resolve_plan(plan, &mut exec).unwrap();
        match resolved {
            LogicalPlan::Filter { predicate, .. } => match predicate {
                Expr::Binary {
                    op: BinaryOp::And,
                    left,
                    ..
                } => assert!(
                    matches!(*left, Expr::Literal(Literal::Bool(false))),
                    "NOT IN on the AND spine should keep filter context, got {left:?}"
                ),
                other => panic!("expected AND, got {other:?}"),
            },
            other => panic!("unexpected plan {other:?}"),
        }
    }
}
