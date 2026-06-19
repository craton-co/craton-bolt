// SPDX-License-Identifier: Apache-2.0

//! Host-side expression evaluator for aggregate inputs.
//!
//! `agg_with_pre.rs` and `groupby_with_pre.rs` historically required every
//! `AggregateExpr` inner to be a bare `Expr::Column(_)` (after alias
//! unwrapping). The physical-plan lowering arranges that to be true in
//! practice — non-trivial aggregate inputs get materialised by the pre
//! kernel, leaving the aggregator with a bare reference to a pre-output
//! column. But the planner can change, and users may construct
//! `PhysicalPlan`s by hand, so the executors should not panic or reject
//! a perfectly legal `Sum(price * tax)` shape.
//!
//! This module exposes a small *host-side* expression evaluator that takes
//! one [`HostColumn`] per source column plus an [`Expr`], and produces a
//! materialised [`HostColumn`] of the same length carrying the computed
//! values. The aggregate executor uses this evaluator when (and only when)
//! the aggregate's inner expression is not a bare column ref — in that case
//! the materialised column is fed row-by-row to the existing reduction
//! kernels exactly as if it had been produced by `pre`.
//!
//! ## Scope
//!
//! Supported expression shapes:
//!   - `Expr::Column(name)` — looked up in the env, cast to `out_dtype`.
//!   - `Expr::Literal(lit)` — broadcast to `n_rows`.
//!   - `Expr::Alias(inner, _)` — transparent recursion.
//!   - `Expr::Binary { op, left, right }`:
//!     * Arithmetic Add/Sub/Mul/Div on Int32/Int64/Float32/Float64.
//!     * Comparison Eq/NotEq/Lt/LtEq/Gt/GtEq → `Bool` output.
//!     * Logical And/Or on `Bool` operands → `Bool` output.
//!   - `Expr::Unary { op, operand }`:
//!     * `IsNull` / `IsNotNull` — accepts any operand dtype; returns
//!       a non-null `Bool` column whose value is `Some(true)` /
//!       `Some(false)` per row depending on the operand's validity.
//!
//! Anything else returns `BoltError::Other` with a `{:?}` of the
//! offending expression. CAST, CASE, NULLIF, scalar functions and so on
//! are explicitly out of scope: the lowering does not produce them
//! today, and adding them belongs in a separate change so that the GPU
//! codegen path can keep up.
//!
//! ## Numeric type promotion
//!
//! The promotion rules mirror `crate::plan::physical_plan::unify_numeric`
//! byte-for-byte:
//!   - same → same,
//!   - either `Float64` → `Float64`,
//!   - `Float32 + Int64` (either order) → `Float64`,
//!   - either `Float32` → `Float32`,
//!   - either `Int64` → `Int64`,
//!   - else `Int32`.
//!
//! The first-cut planner already inserts an explicit cast for every
//! binary op, so this evaluator never has to invent a wider type that the
//! GPU would not also pick.
//!
//! ## NULL propagation (precise contract)
//!
//! Every operand cell carries `Option<T>`. `None` means SQL `NULL`. This
//! module implements SQL three-valued logic (3VL) at the binary-op level:
//!
//!   1. **Arithmetic** (`+ - * /`): If *either* operand is `None`, the
//!      result is `None`. Two `Some` values produce `Some(op(x, y))`.
//!      Integer overflow wraps (matches the device codegen). Integer
//!      division by zero produces `None` per SQL. Float division by zero
//!      follows IEEE-754 (`+inf` / `-inf` / `NaN`); we deliberately do
//!      not promote those to `None` because the GPU path keeps the IEEE
//!      bit pattern.
//!
//!   2. **Comparison** (`= <> < <= > >=`): If *either* operand is `None`,
//!      the result is `None` (NOT `Some(false)`). In particular
//!      `NULL = NULL` is `None`, not `Some(true)`. Two `Some` values
//!      produce `Some(cmp)`.
//!
//!   3. **Logical** (`AND`, `OR`): SQL 3VL with short-circuit on the
//!      "absorbing" value, so `None` does *not* always propagate:
//!      - `AND`: `Some(false) AND _ = Some(false)` (either side);
//!        `Some(true) AND Some(true) = Some(true)`; anything else
//!        involving `None` is `None`. In particular,
//!        `None AND Some(true) = None` and `None AND None = None`.
//!      - `OR`: `Some(true) OR _ = Some(true)` (either side);
//!        `Some(false) OR Some(false) = Some(false)`; anything else
//!        involving `None` is `None`. In particular,
//!        `None OR Some(false) = None` and `None OR None = None`.
//!
//!   4. **Casts**: `cast_column` is `None`-preserving: a `None` input cell
//!      stays `None` in the output column regardless of target dtype.
//!      `Literal::Null` lowers to an `I64` column of `None`s, which then
//!      casts to a `None`-only column of the caller-chosen dtype.
//!
//! ### Consumer contract (important)
//!
//! `HostColumn` exposes its `None` cells to callers verbatim. **It is the
//! caller's responsibility to filter `None`s before passing the column to
//! any downstream reduction that does not itself accept `Option<T>`**.
//! In particular, `agg_with_pre::from_expr_host` collapses `None → 0` so
//! that the primitive reduction path can consume a flat `Vec<T>`; that is
//! safe for `SUM` (identity 0) and the count-of-non-NULL semantics, but
//! callers using this module's output for `MIN`/`MAX` over a column
//! containing `None`s must filter those rows out first — leaving them in
//! would make zero a candidate minimum/maximum, which is wrong. This is
//! an out-of-scope concern for this module; we only guarantee the
//! `Option<T>` contract above.
//!
//! ### Out of scope
//!
//! The AST has no `NOT`, no unary minus, no `IS NULL`, no `COALESCE`, no
//! `CASE`, no `NULLIF`. If those are added to `Expr`, the corresponding
//! 3VL rules belong here — see the test block at the bottom for the
//! shape they should take.
//!
//! TODO(h1): this module correctly propagates NULL as `Option<T>` end-to-end,
//! but its callers (`agg_with_pre::from_expr_host`, `groupby_with_pre::
//! from_expr_host`) collapse `None` into the dtype zero before feeding the
//! GPU reduction. Those collapse points were the original H1 bug source for
//! the pre-projection paths. The fix landed there as part of the C2/C2b
//! work: the NULL rows are now filtered upstream of `eval_expr` via the
//! predicate mask, so by the time `from_expr_host` runs no `None` should
//! ever remain. Callers that bypass the pre-stage (e.g. a future inline
//! evaluator on top of the classic groupby path) must replicate the same
//! filter-then-evaluate ordering or they will reintroduce the bug.
//!
//! That invariant is now CHECKABLE rather than merely documented:
//! [`HostColumn::ensure_no_surviving_nulls`] enforces it at the collapse
//! point — a surviving `None` trips a `debug_assert!` in debug builds and
//! returns a `BoltError::Other` internal-error in release builds, so a
//! contract-violating caller fails loudly instead of silently corrupting the
//! aggregate via the `None → 0` collapse. Collapse callers should invoke it
//! immediately before flattening `Option<T>` into a dense `Vec<T>`.
//!
//! ## Tests
//!
//! Self-contained unit tests live in the `#[cfg(test)] mod tests` at the
//! bottom of the file. They exercise the public API only; no GPU calls.
//! The `null_3vl_*` tests pin the 3VL invariants above.

use std::collections::HashMap;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{
    AggregateExpr, BinaryOp, DataType, Expr, FormatToken, Literal, ScalarFnKind, TimeUnit, UnaryOp,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single tagged host-side value. Mirrors `Literal` but in the value
/// space instead of the AST space, and admits `None` for SQL `NULL`.
///
/// Currently unused by the in-tree executors; exposed for downstream
/// callers that want to surface a single computed scalar (e.g. tests that
/// peek at one element of a [`HostColumn`]).
#[derive(Debug, Clone)]
pub enum HostScalar {
    /// Boolean cell (`None` is SQL NULL).
    Bool(Option<bool>),
    /// Int32 cell.
    I32(Option<i32>),
    /// Int64 cell.
    I64(Option<i64>),
    /// Float32 cell.
    F32(Option<f32>),
    /// Float64 cell.
    F64(Option<f64>),
    /// Utf8 cell.
    Utf8(Option<String>),
}

/// A whole materialised host-side column. Variant order matches
/// `HostScalar`. Within each variant, `None` means the SQL `NULL`.
#[derive(Debug, Clone)]
pub enum HostColumn {
    /// Boolean column.
    Bool(Vec<Option<bool>>),
    /// Int32 column.
    I32(Vec<Option<i32>>),
    /// Int64 column.
    I64(Vec<Option<i64>>),
    /// Float32 column.
    F32(Vec<Option<f32>>),
    /// Float64 column.
    F64(Vec<Option<f64>>),
    /// Utf8 column. Strings are owned — this is a host-side construction
    /// helper, not a zero-copy view.
    Utf8(Vec<Option<String>>),
}

impl HostColumn {
    /// Row count of the column.
    pub fn len(&self) -> usize {
        match self {
            HostColumn::Bool(v) => v.len(),
            HostColumn::I32(v) => v.len(),
            HostColumn::I64(v) => v.len(),
            HostColumn::F32(v) => v.len(),
            HostColumn::F64(v) => v.len(),
            HostColumn::Utf8(v) => v.len(),
        }
    }

    /// True iff the column has zero rows.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Plan-level dtype of the column.
    pub fn dtype(&self) -> DataType {
        match self {
            HostColumn::Bool(_) => DataType::Bool,
            HostColumn::I32(_) => DataType::Int32,
            HostColumn::I64(_) => DataType::Int64,
            HostColumn::F32(_) => DataType::Float32,
            HostColumn::F64(_) => DataType::Float64,
            HostColumn::Utf8(_) => DataType::Utf8,
        }
    }

    /// True iff any cell of this column is SQL `NULL` (`None`).
    pub fn has_nulls(&self) -> bool {
        match self {
            HostColumn::Bool(v) => v.iter().any(Option::is_none),
            HostColumn::I32(v) => v.iter().any(Option::is_none),
            HostColumn::I64(v) => v.iter().any(Option::is_none),
            HostColumn::F32(v) => v.iter().any(Option::is_none),
            HostColumn::F64(v) => v.iter().any(Option::is_none),
            HostColumn::Utf8(v) => v.iter().any(Option::is_none),
        }
    }

    /// H1 / V-10 invariant guard for the GPU-reduction collapse points.
    ///
    /// The collapse callers (`agg_with_pre::from_expr_host`,
    /// `groupby_with_pre::from_expr_host`) flatten this column's `Option<T>`
    /// cells into a dense `Vec<T>` before the primitive GPU reduction, mapping
    /// `None → dtype zero`. As documented in the module-level `TODO(h1)`, that
    /// is only sound because the pre-stage predicate mask is supposed to have
    /// already removed every NULL row upstream of `eval_expr`. A caller that
    /// bypasses the pre-stage (e.g. a future inline evaluator on the classic
    /// groupby path) would reintroduce the H1 zero-injection bug, where a
    /// surviving `None` becomes a real `0` candidate and corrupts
    /// `MIN`/`MAX`/`SUM`.
    ///
    /// That invariant was previously documented but UNENFORCED. This guard
    /// makes it checkable: callers invoke it immediately before collapsing,
    /// naming the call site via `context`. In debug builds a surviving NULL
    /// trips a `debug_assert!` so the contract violation is loud during
    /// development; in release builds it returns a `BoltError::Other`
    /// internal-error (rather than silently producing a wrong answer) so the
    /// engine's "never silently wrong" invariant holds even if the assertion
    /// is compiled out.
    ///
    /// For a correct caller (no surviving NULLs) this is a cheap scan that
    /// returns `Ok(())` and changes no observable behavior.
    pub fn ensure_no_surviving_nulls(&self, context: &str) -> BoltResult<()> {
        debug_assert!(
            !self.has_nulls(),
            "expr_agg: surviving NULL reached the GPU-reduction collapse in {context}; \
             the pre-stage predicate mask must filter NULL rows before from_expr_host \
             (see TODO(h1)) — collapsing None→0 here would corrupt MIN/MAX/SUM",
        );
        if self.has_nulls() {
            return Err(BoltError::Other(format!(
                "expr_agg: internal invariant violated — surviving NULL reached the \
                 GPU-reduction collapse in {context}; None→0 collapse would corrupt \
                 the aggregate (see TODO(h1))"
            )));
        }
        Ok(())
    }
}

/// Source columns indexed by name. The evaluator does not own the
/// columns: callers build the env from their own storage and pass
/// references in. Borrowing keeps `eval_expr` cheap to set up for many
/// expressions over the same env.
pub type ColumnEnv<'a> = HashMap<String, &'a HostColumn>;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Evaluate `expr` against `env`, producing a column of dtype `out_dtype`
/// and length `n_rows`.
///
/// `n_rows` is the authoritative row count: the result is always
/// `n_rows` rows long, even when `expr` is a pure literal or when `env`
/// is empty. When `expr` references a column, that column's length must
/// equal `n_rows` (mismatch is an error). When `expr` references
/// multiple columns, all referenced columns must have the same length
/// (the caller is responsible for that — we still cross-check against
/// `n_rows`).
///
/// The output is always cast to `out_dtype`. For numeric → numeric the
/// cast is the standard Rust `as` conversion (saturating for narrowing,
/// because that's what the device-side codegen does). For Bool ↔ numeric
/// see [`cast_column`]. Utf8 ↔ non-Utf8 is an error.
pub fn eval_expr(
    expr: &Expr,
    env: &ColumnEnv<'_>,
    out_dtype: DataType,
    n_rows: usize,
) -> BoltResult<HostColumn> {
    let raw = eval_inner(expr, env, n_rows)?;
    if raw.len() != n_rows {
        return Err(BoltError::Other(format!(
            "expr_agg: eval_expr produced {} rows, expected {}",
            raw.len(),
            n_rows
        )));
    }
    if raw.dtype() == out_dtype {
        Ok(raw)
    } else {
        cast_column(raw, out_dtype)
    }
}

/// Convenience: if `expr` (after stripping aliases) is a bare
/// `Expr::Column(name)`, return that name. Otherwise return `None`.
///
/// Aggregate executors call this to detect the fast path where the
/// aggregate's inner is just a column ref and no host-side evaluation is
/// needed. The non-erroring shape (versus `bare_column_name` in
/// `agg_with_pre.rs`) lets the caller cleanly choose between the two
/// branches without `try_…`-style control flow.
pub fn try_bare_column(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Column(name) => Some(name.as_str()),
        Expr::Alias(inner, _) => try_bare_column(inner),
        _ => None,
    }
}

/// Unwrap the inner expression of an `AggregateExpr` and evaluate it.
///
/// This is the supplement consumed by `agg_with_pre.rs` and
/// `groupby_with_pre.rs` on the slow path: when `try_bare_column(inner)`
/// returned `None`, the aggregate executor builds a `ColumnEnv` over the
/// pre stage's host columns and calls this to materialise the
/// per-aggregate input.
pub fn materialize_agg_input(
    agg: &AggregateExpr,
    env: &ColumnEnv<'_>,
    expected_dtype: DataType,
    n_rows: usize,
) -> BoltResult<HostColumn> {
    let inner = match agg {
        AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e)
        | AggregateExpr::Count(e) => e,
        AggregateExpr::VarPop(e) | AggregateExpr::VarSamp(e) => e.as_ref(),
        // STDDEV variants store their operand boxed; deref to the inner
        // expression so the evaluator's `eval_expr` borrow shape matches.
        AggregateExpr::StddevPop(e) | AggregateExpr::StddevSamp(e) => e.as_ref(),
    };
    eval_expr(inner, env, expected_dtype, n_rows)
}

// ---------------------------------------------------------------------------
// Evaluator core
// ---------------------------------------------------------------------------

/// Like `eval_expr` but does *not* coerce the result to a caller-chosen
/// dtype. Returns the natural dtype produced by the expression.
fn eval_inner(expr: &Expr, env: &ColumnEnv<'_>, n_rows: usize) -> BoltResult<HostColumn> {
    match expr {
        Expr::Column(name) => eval_column(name, env, n_rows),
        Expr::Literal(lit) => eval_literal(lit, n_rows),
        Expr::Alias(inner, _) => eval_inner(inner, env, n_rows),
        Expr::Binary { op, left, right } => eval_binary(*op, left, right, env, n_rows),
        Expr::Unary { op, operand } => eval_unary(*op, operand, env, n_rows),
        // v0.7: CASE is lowered to GPU `Op::Select` for scan-chain
        // Project / Filter positions (and the pre-aggregation kernel
        // feeding GROUP BY / aggregates). It still has no host-side
        // evaluator, so any CASE that survives to a host-side
        // `PhysicalPlan::Project` / `PhysicalPlan::Filter` (HAVING,
        // post-aggregate SELECT, etc.) lands here with a clear
        // not-yet-supported message.
        Expr::Case { .. } => Err(BoltError::Plan(
            "CASE in host-side expressions (HAVING / post-aggregate \
             projection / sort) is not yet supported; coming in a follow-up"
                .into(),
        )),
        Expr::Like {
            expr,
            pattern,
            escape,
            negated,
            case_insensitive,
        } => eval_like(
            expr,
            pattern,
            *escape,
            *negated,
            *case_insensitive,
            env,
            n_rows,
        ),
        // F4: TRY_CAST / SAFE_CAST (`safe = true`) is evaluated host-side —
        // the physical-plan lowering routes any projection carrying one here
        // (see `physical_plan::expr_contains_safe_cast`). A conversion failure
        // on a non-null input yields SQL NULL instead of erroring. Plain CAST
        // (`safe = false`) keeps its error-on-failure semantics; it normally
        // lowers to the GPU `cvt.*` path, but evaluating it host-side here (if
        // a future caller routes it) must behave identically to that path, so
        // we defer to `cast_column` (saturating numeric `as`, Utf8 errors).
        Expr::Cast { expr, target, safe } => {
            let inner = eval_inner(expr, env, n_rows)?;
            if *safe {
                safe_cast_column(inner, *target)
            } else {
                cast_column(inner, *target)
            }
        }
        // CAST FORMAT: host-only temporal ⇄ string conversion (feature
        // CAST FORMAT). The physical-plan boundary routes any projection
        // carrying a `CastFormat` here (see
        // `physical_plan::expr_contains_cast_format`). NULL propagates as NULL.
        //
        // IMPORTANT: temporals live in `HostColumn` as their fixed-width
        // storage — `Date32` as `I32` (days since epoch), `Timestamp` as `I64`
        // (ticks). So the *string→temporal* arm returns `I32`/`I64` and the
        // *temporal→string* arm returns `Utf8`; the caller's `eval_expr`
        // out-dtype coercion is a no-op in both cases (Date32/Timestamp out
        // fields surface as I32/I64; Utf8 stays Utf8).
        Expr::CastFormat {
            expr,
            target,
            pattern,
            to_text,
        } => eval_cast_format(expr, *target, pattern, *to_text, env, n_rows),
        // String scalar functions evaluated host-side. SUBSTRING and TRIM are
        // wired here (the physical-plan boundary routes Projects carrying them
        // to `PhysicalPlan::Project`, whose executor calls `eval_expr`).
        // UPPER/LOWER/LENGTH have dedicated GPU producers and CONCAT is handled
        // as `BinaryOp::Concat`; if one of those reaches here it means a caller
        // bypassed lowering, so we surface a clear `Plan` error.
        Expr::ScalarFn { kind, args } => eval_scalar_fn(*kind, args, env, n_rows),
        // v0.7 date-scalar-fns: EXTRACT / DATE_TRUNC have a standalone GPU
        // codegen module but no host evaluator yet. The physical-plan boundary
        // rejects them before execution; reaching here means a future caller
        // built a Filter / Project around one without going through `lower()`.
        Expr::Extract { field, .. } => Err(BoltError::Plan(format!(
            "expr_agg: EXTRACT({} FROM ...) is not yet evaluated host-side; \
             coming in a follow-up",
            field.sql_name()
        ))),
        Expr::DateTrunc { unit, .. } => Err(BoltError::Plan(format!(
            "expr_agg: DATE_TRUNC('{}', ...) is not yet evaluated host-side; \
             coming in a follow-up",
            unit.sql_name()
        ))),
        // Subqueries have no host evaluator yet; the physical-plan boundary
        // rejects any plan carrying one (see `physical_plan::Codegen::emit_expr`),
        // so reaching here means a future caller bypassed `lower()`. Surface a
        // clear `Plan` error rather than silently producing wrong output.
        Expr::ScalarSubquery(_) | Expr::InSubquery { .. } => Err(BoltError::Plan(
            "expr_agg: subqueries are not yet evaluated host-side; \
             coming in a follow-up"
                .into(),
        )),
    }
}

/// Evaluate `expr LIKE 'pattern'` / `expr NOT LIKE 'pattern'` on the host.
///
/// `expr` must produce a `Utf8` (or `Utf8`-castable) column. `pattern`'s
/// `%` matches zero-or-more characters and `_` matches exactly one. Rows
/// where the operand is `NULL` produce `None` in the output (SQL 3VL —
/// `NULL LIKE 'x'` is NULL, not false). For non-NULL rows the result is
/// `Some(true)` / `Some(false)`. `negated` inverts the per-row Bool but
/// preserves the `None` cells.
///
/// `escape` carries the optional `ESCAPE '<char>'` clause.
///
/// `case_insensitive` is `true` for `ILIKE` — the pattern and input are
/// Unicode case-folded before matching (see
/// [`crate::exec::like::PatternMatcher::compile_ci`]); `false` for the
/// case-sensitive plain `LIKE` path (unchanged behaviour).
fn eval_like(
    expr: &Expr,
    pattern: &str,
    escape: Option<char>,
    negated: bool,
    case_insensitive: bool,
    env: &ColumnEnv<'_>,
    n_rows: usize,
) -> BoltResult<HostColumn> {
    let raw = eval_inner(expr, env, n_rows)?;
    let utf8 = match raw {
        HostColumn::Utf8(v) => v,
        other => {
            return Err(BoltError::Type(format!(
                "expr_agg: LIKE requires a Utf8 operand, got {:?}",
                other.dtype()
            )));
        }
    };
    if utf8.len() != n_rows {
        return Err(BoltError::Other(format!(
            "expr_agg: LIKE operand produced {} rows, expected {}",
            utf8.len(),
            n_rows
        )));
    }
    let matcher = crate::exec::like::PatternMatcher::compile_ci(pattern, escape, case_insensitive)?;
    let out: Vec<Option<bool>> = utf8
        .iter()
        .map(|cell| match cell {
            None => None,
            Some(s) => {
                let m = matcher.matches(s);
                Some(if negated { !m } else { m })
            }
        })
        .collect();
    Ok(HostColumn::Bool(out))
}

// ---------------------------------------------------------------------------
// CAST FORMAT (feature CAST FORMAT) — host-only temporal ⇄ string conversion
// ---------------------------------------------------------------------------

/// Days since the Unix epoch (1970-01-01) for `(year, month, day)`, proleptic
/// Gregorian (Hinnant's `days_from_civil`). Returns `None` if outside the
/// `i32` Date32 range. Mirrors `sql_frontend::days_since_epoch` but is local so
/// this module has no dependency on the SQL frontend.
fn days_from_civil(y: i64, m: i64, d: i64) -> Option<i32> {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as i64; // [0, 399]
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    i32::try_from(days).ok()
}

/// Inverse: `(year, month, day)` for a Date32 day count (Hinnant's
/// `civil_from_days`). Exact across the whole `i32` day range.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d)
}

/// A decomposed civil date-time used by the format / parse routines.
#[derive(Debug, Clone, Copy, Default)]
struct CivilDateTime {
    year: i64,
    month: i64,
    day: i64,
    hour: i64,
    minute: i64,
    second: i64,
}

/// Format one [`CivilDateTime`] using the validated token sequence.
fn format_civil(parts: &CivilDateTime, pattern: &[FormatToken]) -> String {
    let mut s = String::with_capacity(pattern.len() * 2);
    for tok in pattern {
        match tok {
            FormatToken::Year4 => {
                // 4-digit zero-padded year. Negative / >9999 years still emit
                // their full digits (the leading sign / extra digits are
                // tolerated rather than truncated).
                if parts.year < 0 {
                    s.push('-');
                    s.push_str(&format!("{:04}", -parts.year));
                } else {
                    s.push_str(&format!("{:04}", parts.year));
                }
            }
            FormatToken::Month => s.push_str(&format!("{:02}", parts.month)),
            FormatToken::Day => s.push_str(&format!("{:02}", parts.day)),
            FormatToken::Hour24 => s.push_str(&format!("{:02}", parts.hour)),
            FormatToken::Minute => s.push_str(&format!("{:02}", parts.minute)),
            FormatToken::Second => s.push_str(&format!("{:02}", parts.second)),
            FormatToken::Literal(c) => s.push(*c),
        }
    }
    s
}

/// Parse a string against the validated token sequence into a
/// [`CivilDateTime`]. Field tokens consume their fixed digit width; literal
/// tokens must match verbatim. Returns `None` on any structural mismatch,
/// non-digit field, or out-of-range field value (months 1..=12, days against
/// the real month length, hour 0..=23, minute/second 0..=59).
fn parse_civil(input: &str, pattern: &[FormatToken]) -> Option<CivilDateTime> {
    let b = input.as_bytes();
    let mut i = 0usize;
    let mut p = CivilDateTime {
        // Defaults for fields a pattern may omit (e.g. a date-only pattern
        // leaves the clock at midnight).
        year: 1970,
        month: 1,
        day: 1,
        ..Default::default()
    };
    // Read `width` ASCII digits starting at `i`, advancing it. ASCII digits
    // are single-byte so byte indexing is UTF-8-safe.
    fn take_digits(b: &[u8], i: &mut usize, width: usize) -> Option<i64> {
        if *i + width > b.len() {
            return None;
        }
        let mut v: i64 = 0;
        for _ in 0..width {
            let c = b[*i];
            if !c.is_ascii_digit() {
                return None;
            }
            v = v * 10 + (c - b'0') as i64;
            *i += 1;
        }
        Some(v)
    }
    for tok in pattern {
        match tok {
            FormatToken::Year4 => p.year = take_digits(b, &mut i, 4)?,
            FormatToken::Month => p.month = take_digits(b, &mut i, 2)?,
            FormatToken::Day => p.day = take_digits(b, &mut i, 2)?,
            FormatToken::Hour24 => p.hour = take_digits(b, &mut i, 2)?,
            FormatToken::Minute => p.minute = take_digits(b, &mut i, 2)?,
            FormatToken::Second => p.second = take_digits(b, &mut i, 2)?,
            FormatToken::Literal(c) => {
                // The literal must match exactly one byte (all literals are
                // ASCII).
                if i >= b.len() || b[i] != (*c as u8) {
                    return None;
                }
                i += 1;
            }
        }
    }
    // The whole input must be consumed (no trailing garbage).
    if i != b.len() {
        return None;
    }
    // Range-validate the decomposed fields.
    if !(1..=12).contains(&p.month) {
        return None;
    }
    let dim = days_in_month_local(p.year, p.month as u32);
    if p.day < 1 || p.day > dim as i64 {
        return None;
    }
    if p.hour > 23 || p.minute > 59 || p.second > 59 || p.hour < 0 || p.minute < 0 || p.second < 0 {
        return None;
    }
    Some(p)
}

/// Local copy of the proleptic-Gregorian month length used by `parse_civil`.
fn days_in_month_local(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            let leap = (y % 4 == 0) && (y % 100 != 0 || y % 400 == 0);
            if leap {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Evaluate a `CAST(... FORMAT ...)` expression host-side.
///
/// `to_text = true`  → format `Date32`(I32) / `Timestamp`(I64) → `Utf8`.
/// `to_text = false` → parse `Utf8` → `Date32`(I32) / `Timestamp`(I64).
///
/// NULL cells propagate as NULL. A parse failure on a non-null cell is a hard
/// error (plain CAST semantics — FORMAT is only carried by plain CAST; see
/// `sql_frontend::lower_cast_format`).
fn eval_cast_format(
    expr: &Expr,
    target: DataType,
    pattern: &[FormatToken],
    to_text: bool,
    env: &ColumnEnv<'_>,
    n_rows: usize,
) -> BoltResult<HostColumn> {
    let inner = eval_inner(expr, env, n_rows)?;
    if to_text {
        // Temporal → string. Source is I32 (Date32 days) or I64 (Timestamp
        // ticks, nanoseconds).
        let out: Vec<Option<String>> = match inner {
            HostColumn::I32(v) => v
                .into_iter()
                .map(|cell| {
                    cell.map(|days| {
                        let (y, m, d) = civil_from_days(days as i64);
                        format_civil(
                            &CivilDateTime {
                                year: y,
                                month: m,
                                day: d,
                                ..Default::default()
                            },
                            pattern,
                        )
                    })
                })
                .collect(),
            HostColumn::I64(v) => v
                .into_iter()
                .map(|cell| cell.map(|ticks| format_civil(&civil_from_ticks_ns(ticks), pattern)))
                .collect(),
            other => {
                return Err(BoltError::Type(format!(
                    "CAST(... FORMAT ...) temporal→string requires a Date32/Timestamp \
                     operand (stored as I32/I64), got {:?}",
                    other.dtype()
                )));
            }
        };
        Ok(HostColumn::Utf8(out))
    } else {
        // String → temporal. Source must be Utf8.
        let utf8 = match inner {
            HostColumn::Utf8(v) => v,
            other => {
                return Err(BoltError::Type(format!(
                    "CAST(... FORMAT ...) string→temporal requires a Utf8 operand, \
                     got {:?}",
                    other.dtype()
                )));
            }
        };
        match target {
            DataType::Date32 => {
                let mut out: Vec<Option<i32>> = Vec::with_capacity(utf8.len());
                for cell in utf8 {
                    match cell {
                        None => out.push(None),
                        Some(s) => {
                            let parts = parse_civil(s.trim(), pattern).ok_or_else(|| {
                                BoltError::Other(format!(
                                    "CAST(... AS DATE FORMAT ...): value '{s}' does not \
                                     match the format pattern"
                                ))
                            })?;
                            let days = days_from_civil(parts.year, parts.month, parts.day)
                                .ok_or_else(|| {
                                    BoltError::Other(format!(
                                        "CAST(... AS DATE FORMAT ...): value '{s}' is out \
                                         of the supported Date32 range"
                                    ))
                                })?;
                            out.push(Some(days));
                        }
                    }
                }
                Ok(HostColumn::I32(out))
            }
            DataType::Timestamp(unit, _tz) => {
                let mut out: Vec<Option<i64>> = Vec::with_capacity(utf8.len());
                for cell in utf8 {
                    match cell {
                        None => out.push(None),
                        Some(s) => {
                            let parts = parse_civil(s.trim(), pattern).ok_or_else(|| {
                                BoltError::Other(format!(
                                    "CAST(... AS TIMESTAMP FORMAT ...): value '{s}' does \
                                     not match the format pattern"
                                ))
                            })?;
                            let ticks = civil_to_ticks(&parts, unit).ok_or_else(|| {
                                BoltError::Other(format!(
                                    "CAST(... AS TIMESTAMP FORMAT ...): value '{s}' is out \
                                     of the supported range"
                                ))
                            })?;
                            out.push(Some(ticks));
                        }
                    }
                }
                Ok(HostColumn::I64(out))
            }
            other => Err(BoltError::Type(format!(
                "CAST(... FORMAT ...) string→temporal target must be Date32 or \
                 Timestamp, got {other:?}"
            ))),
        }
    }
}

/// Decompose a nanosecond Timestamp tick count into civil fields. Uses
/// floor-division so negative (pre-epoch) timestamps decompose correctly.
fn civil_from_ticks_ns(ticks: i64) -> CivilDateTime {
    let secs = ticks.div_euclid(1_000_000_000);
    let days = secs.div_euclid(86_400);
    let sod = secs.rem_euclid(86_400); // seconds-of-day, 0..86399
    let (y, m, d) = civil_from_days(days);
    CivilDateTime {
        year: y,
        month: m,
        day: d,
        hour: sod / 3600,
        minute: (sod % 3600) / 60,
        second: sod % 60,
    }
}

/// Recompose civil fields into a Timestamp tick count at `unit` resolution.
/// Returns `None` on i64 overflow. Only whole-second precision is produced
/// (the supported pattern vocabulary has no sub-second token).
fn civil_to_ticks(p: &CivilDateTime, unit: TimeUnit) -> Option<i64> {
    let days = days_from_civil(p.year, p.month, p.day)? as i64;
    let secs = days
        .checked_mul(86_400)?
        .checked_add(p.hour * 3600 + p.minute * 60 + p.second)?;
    let mul = match unit {
        TimeUnit::Second => 1i64,
        TimeUnit::Millisecond => 1_000,
        TimeUnit::Microsecond => 1_000_000,
        TimeUnit::Nanosecond => 1_000_000_000,
    };
    secs.checked_mul(mul)
}

/// Evaluate a string scalar function host-side.
///
/// Currently wired: `SUBSTRING` (`ScalarFnKind::Substring`) and `TRIM`
/// (`ScalarFnKind::Trim{Both,Leading,Trailing}`). UPPER/LOWER/LENGTH have
/// dedicated GPU producers and CONCAT is lowered to `BinaryOp::Concat`; if one
/// of those reaches here, lowering was bypassed and we surface a `Plan` error.
///
/// All operands are evaluated to per-row columns so column-valued arguments
/// (e.g. `SUBSTRING(s, start_col, len_col)`) work uniformly with the common
/// literal case. NULL in the source string (and, for SUBSTRING, NULL in a
/// position argument) propagates as NULL.
fn eval_scalar_fn(
    kind: ScalarFnKind,
    args: &[Expr],
    env: &ColumnEnv<'_>,
    n_rows: usize,
) -> BoltResult<HostColumn> {
    use crate::exec::string_ops_extended::{
        initcap_str, left_str, octet_length_str, pad_str, position_str, replace_str, reverse_str,
        right_str, substring_str, trim_str, PadSide, TrimSide,
    };
    match kind {
        ScalarFnKind::Substring => {
            if args.len() != 2 && args.len() != 3 {
                return Err(BoltError::Plan(format!(
                    "expr_agg: SUBSTRING expects 2 or 3 args, got {}",
                    args.len()
                )));
            }
            let src = eval_utf8_arg(&args[0], env, n_rows, "SUBSTRING")?;
            let start = eval_i64_arg(&args[1], env, n_rows, "SUBSTRING")?;
            // Length defaults to "to the end" (i32::MAX) when the 2-arg form
            // is used. Each row's length is the per-row value, clamped to i32.
            let length: Vec<Option<i64>> = if args.len() == 3 {
                eval_i64_arg(&args[2], env, n_rows, "SUBSTRING")?
            } else {
                vec![Some(i32::MAX as i64); n_rows]
            };
            let mut out: Vec<Option<String>> = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                // NULL source OR NULL position -> NULL.
                match (&src[i], start[i], length[i]) {
                    (Some(s), Some(st), Some(ln)) => {
                        let st_i32 = clamp_i64_to_i32(st);
                        let ln_i32 = clamp_i64_to_i32(ln);
                        out.push(Some(substring_str(s, st_i32, ln_i32)));
                    }
                    _ => out.push(None),
                }
            }
            Ok(HostColumn::Utf8(out))
        }
        ScalarFnKind::TrimBoth | ScalarFnKind::TrimLeading | ScalarFnKind::TrimTrailing => {
            if args.is_empty() || args.len() > 2 {
                return Err(BoltError::Plan(format!(
                    "expr_agg: TRIM expects 1 or 2 args, got {}",
                    args.len()
                )));
            }
            let side = match kind {
                ScalarFnKind::TrimLeading => TrimSide::Leading,
                ScalarFnKind::TrimTrailing => TrimSide::Trailing,
                _ => TrimSide::Both,
            };
            let src = eval_utf8_arg(&args[0], env, n_rows, "TRIM")?;
            let chars: Option<Vec<Option<String>>> = if args.len() == 2 {
                Some(eval_utf8_arg(&args[1], env, n_rows, "TRIM")?)
            } else {
                None
            };
            let mut out: Vec<Option<String>> = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                let s = match &src[i] {
                    Some(s) => s,
                    None => {
                        out.push(None);
                        continue;
                    }
                };
                match &chars {
                    // A NULL trim-character set yields NULL (standard SQL: any
                    // NULL operand to the scalar function propagates).
                    Some(c) => match &c[i] {
                        Some(cset) => out.push(Some(trim_str(s, side, Some(cset)))),
                        None => out.push(None),
                    },
                    None => out.push(Some(trim_str(s, side, None))),
                }
            }
            Ok(HostColumn::Utf8(out))
        }
        ScalarFnKind::OctetLength => {
            let src = eval_utf8_arg(&args[0], env, n_rows, "OCTET_LENGTH")?;
            let out: Vec<Option<i64>> = src
                .iter()
                .map(|c| c.as_deref().map(octet_length_str))
                .collect();
            Ok(HostColumn::I64(out))
        }
        ScalarFnKind::Position => {
            let s = eval_utf8_arg(&args[0], env, n_rows, "POSITION")?;
            let sub = eval_utf8_arg(&args[1], env, n_rows, "POSITION")?;
            let mut out = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                out.push(match (&s[i], &sub[i]) {
                    (Some(s), Some(sub)) => Some(position_str(s, sub)),
                    _ => None,
                });
            }
            Ok(HostColumn::I64(out))
        }
        ScalarFnKind::Replace => {
            let s = eval_utf8_arg(&args[0], env, n_rows, "REPLACE")?;
            let from = eval_utf8_arg(&args[1], env, n_rows, "REPLACE")?;
            let to = eval_utf8_arg(&args[2], env, n_rows, "REPLACE")?;
            let mut out = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                out.push(match (&s[i], &from[i], &to[i]) {
                    (Some(s), Some(f), Some(t)) => Some(replace_str(s, f, t)),
                    _ => None,
                });
            }
            Ok(HostColumn::Utf8(out))
        }
        ScalarFnKind::Left | ScalarFnKind::Right => {
            let s = eval_utf8_arg(&args[0], env, n_rows, kind.sql_name())?;
            let n = eval_i64_arg(&args[1], env, n_rows, kind.sql_name())?;
            let is_left = matches!(kind, ScalarFnKind::Left);
            let mut out = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                out.push(match (&s[i], n[i]) {
                    (Some(s), Some(n)) => Some(if is_left {
                        left_str(s, n)
                    } else {
                        right_str(s, n)
                    }),
                    _ => None,
                });
            }
            Ok(HostColumn::Utf8(out))
        }
        ScalarFnKind::Lpad | ScalarFnKind::Rpad => {
            let s = eval_utf8_arg(&args[0], env, n_rows, kind.sql_name())?;
            let len = eval_i64_arg(&args[1], env, n_rows, kind.sql_name())?;
            let pad = eval_utf8_arg(&args[2], env, n_rows, kind.sql_name())?;
            let side = if matches!(kind, ScalarFnKind::Lpad) {
                PadSide::Left
            } else {
                PadSide::Right
            };
            let mut out = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                out.push(match (&s[i], len[i], &pad[i]) {
                    (Some(s), Some(l), Some(p)) => Some(pad_str(s, l, p, side)),
                    _ => None,
                });
            }
            Ok(HostColumn::Utf8(out))
        }
        ScalarFnKind::Reverse => {
            let s = eval_utf8_arg(&args[0], env, n_rows, "REVERSE")?;
            let out = s.iter().map(|c| c.as_deref().map(reverse_str)).collect();
            Ok(HostColumn::Utf8(out))
        }
        ScalarFnKind::Initcap => {
            let s = eval_utf8_arg(&args[0], env, n_rows, "INITCAP")?;
            let out = s.iter().map(|c| c.as_deref().map(initcap_str)).collect();
            Ok(HostColumn::Utf8(out))
        }
        // These never legitimately reach the host evaluator (see fn docs).
        ScalarFnKind::Upper | ScalarFnKind::Lower | ScalarFnKind::Length | ScalarFnKind::Concat => {
            Err(BoltError::Plan(format!(
                "expr_agg: string scalar function {} is not evaluated host-side; \
             it has a dedicated lowering path",
                kind.sql_name()
            )))
        }
    }
}

/// Evaluate `arg` to a `Utf8` column of length `n_rows`. Errors if the result
/// is not Utf8. `fname` is the SQL function name used in error messages.
fn eval_utf8_arg(
    arg: &Expr,
    env: &ColumnEnv<'_>,
    n_rows: usize,
    fname: &str,
) -> BoltResult<Vec<Option<String>>> {
    match eval_inner(arg, env, n_rows)? {
        HostColumn::Utf8(v) => Ok(v),
        other => Err(BoltError::Type(format!(
            "expr_agg: {fname} requires a Utf8 argument, got {:?}",
            other.dtype()
        ))),
    }
}

/// Evaluate `arg` to an `i64`-valued column of length `n_rows`, widening
/// narrower integer columns. Errors on non-integer dtypes.
fn eval_i64_arg(
    arg: &Expr,
    env: &ColumnEnv<'_>,
    n_rows: usize,
    fname: &str,
) -> BoltResult<Vec<Option<i64>>> {
    match eval_inner(arg, env, n_rows)? {
        HostColumn::I64(v) => Ok(v),
        HostColumn::I32(v) => Ok(v.into_iter().map(|c| c.map(|x| x as i64)).collect()),
        other => Err(BoltError::Type(format!(
            "expr_agg: {fname} position argument must be an integer, got {:?}",
            other.dtype()
        ))),
    }
}

/// Saturating cast of an `i64` position/length to `i32` (the domain the
/// byte-substring helper works in). Values outside the `i32` range clamp to
/// the nearest bound, which is benign: a huge positive length means "to the
/// end" and a huge negative one means "empty".
fn clamp_i64_to_i32(v: i64) -> i32 {
    if v > i32::MAX as i64 {
        i32::MAX
    } else if v < i32::MIN as i64 {
        i32::MIN
    } else {
        v as i32
    }
}

/// Look up `name` in `env` and clone the referenced column. Validates the
/// length against `n_rows`. The dtype is left alone — the outer
/// `eval_expr` is responsible for the final cast.
fn eval_column(name: &str, env: &ColumnEnv<'_>, n_rows: usize) -> BoltResult<HostColumn> {
    let col = env.get(name).ok_or_else(|| {
        BoltError::Plan(format!(
            "expr_agg: column '{name}' not found in evaluator env"
        ))
    })?;
    if col.len() != n_rows {
        return Err(BoltError::Other(format!(
            "expr_agg: column '{}' has {} rows, expected {}",
            name,
            col.len(),
            n_rows
        )));
    }
    Ok((*col).clone())
}

/// Broadcast a literal to a column of length `n_rows`. `Literal::Null`
/// has no static dtype, so we produce an `I64` column of `None`s — the
/// outer cast lifts it to the caller-chosen `out_dtype`. (NULL of any
/// dtype is still NULL.)
fn eval_literal(lit: &Literal, n_rows: usize) -> BoltResult<HostColumn> {
    Ok(match lit {
        Literal::Null => HostColumn::I64(vec![None; n_rows]),
        Literal::Bool(b) => HostColumn::Bool(vec![Some(*b); n_rows]),
        Literal::Int32(v) => HostColumn::I32(vec![Some(*v); n_rows]),
        Literal::Int64(v) => HostColumn::I64(vec![Some(*v); n_rows]),
        Literal::Float32(v) => HostColumn::F32(vec![Some(*v); n_rows]),
        Literal::Float64(v) => HostColumn::F64(vec![Some(*v); n_rows]),
        Literal::Utf8(s) => HostColumn::Utf8(vec![Some(s.clone()); n_rows]),
        Literal::Decimal128(..) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up \
                 (host-side literal broadcast)"
                    .into(),
            ))
        }
        // v0.6 / M4: Date32 stores as i32 days; Timestamp stores as i64
        // ticks. Broadcast as the underlying integer; the expression
        // evaluator does not yet apply temporal semantics.
        Literal::Date32(v) => HostColumn::I32(vec![Some(*v); n_rows]),
        Literal::Timestamp(v, _unit, _tz) => HostColumn::I64(vec![Some(*v); n_rows]),
    })
}

/// Evaluate a binary op. Both operands are evaluated, then the op is
/// dispatched by category. Per-category unification + op application
/// lives in the helper functions below.
fn eval_binary(
    op: BinaryOp,
    left: &Expr,
    right: &Expr,
    env: &ColumnEnv<'_>,
    n_rows: usize,
) -> BoltResult<HostColumn> {
    let l = eval_inner(left, env, n_rows)?;
    let r = eval_inner(right, env, n_rows)?;
    if l.len() != n_rows || r.len() != n_rows {
        return Err(BoltError::Other(format!(
            "expr_agg: binary operand length mismatch: lhs={}, rhs={}, expected={}",
            l.len(),
            r.len(),
            n_rows
        )));
    }

    if is_arithmetic(op) {
        eval_arithmetic(op, l, r)
    } else if matches!(
        op,
        BinaryOp::Mod
            | BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr
    ) {
        eval_integer_op(op, l, r)
    } else if is_comparison(op) {
        eval_comparison(op, l, r)
    } else if is_logical(op) {
        eval_logical(op, l, r)
    } else if is_string(op) {
        eval_string(op, l, r)
    } else {
        Err(BoltError::Other(format!(
            "expr_agg: unsupported operator {:?}",
            op
        )))
    }
}

/// Evaluate string-valued binary ops — today only `BinaryOp::Concat` (SQL
/// `||`). Both operands must be Utf8; result is Utf8. NULL on either side
/// propagates as NULL (standard SQL).
fn eval_string(op: BinaryOp, lhs: HostColumn, rhs: HostColumn) -> BoltResult<HostColumn> {
    if !matches!(op, BinaryOp::Concat) {
        return Err(BoltError::Other(format!(
            "expr_agg: unsupported string operator {:?}",
            op
        )));
    }
    let (a, b) = match (lhs, rhs) {
        (HostColumn::Utf8(a), HostColumn::Utf8(b)) => (a, b),
        (l, r) => {
            return Err(BoltError::Type(format!(
                "expr_agg: string {:?} requires Utf8 operands, got {:?} and {:?}",
                op,
                l.dtype(),
                r.dtype()
            )));
        }
    };
    let out = crate::exec::string_ops::host_concat_option_strings(&a, &b)?;
    Ok(HostColumn::Utf8(out))
}

/// Evaluate a unary op — today: `IS NULL` / `IS NOT NULL` / `NOT`.
///
/// For `IS [NOT] NULL` the result is always a non-nullable `Bool` column:
/// the test inspects the operand's validity bit, never its value, so the
/// answer is always defined even when the operand row is NULL.
///
/// For `NOT` the operand must be `Bool`-typed; the result preserves
/// nullability (SQL `NOT NULL` is NULL).
///
/// Implementation: evaluate the operand into a `HostColumn`, then walk
/// each cell's `Option<_>` to derive the per-row boolean. Aliases inside
/// the operand are handled transparently by the recursive call.
///
/// TODO(perf): this scans the whole operand row-by-row on the host. For
/// large columns this is significantly slower than a kernel that just
/// reads the validity bitmap. The plan-time `lower()` already routes any
/// Filter whose predicate contains `Expr::Unary` through this host path
/// (see `physical_plan::predicate_contains_unary`), which is fine for
/// small inputs — push to GPU once the IR/codegen learn to read validity.
fn eval_unary(
    op: UnaryOp,
    operand: &Expr,
    env: &ColumnEnv<'_>,
    n_rows: usize,
) -> BoltResult<HostColumn> {
    let col = eval_inner(operand, env, n_rows)?;
    if col.len() != n_rows {
        return Err(BoltError::Other(format!(
            "expr_agg: unary operand produced {} rows, expected {}",
            col.len(),
            n_rows
        )));
    }
    // `NOT` requires a Bool operand and propagates NULL per SQL three-valued
    // logic: `NOT NULL = NULL`. Type-checking has already accepted the
    // operand as Bool at the logical layer, so a non-Bool here is an
    // internal invariant violation.
    if matches!(op, UnaryOp::Not) {
        return match col {
            HostColumn::Bool(v) => Ok(HostColumn::Bool(
                v.into_iter().map(|c| c.map(|b| !b)).collect(),
            )),
            other => Err(BoltError::Other(format!(
                "expr_agg: NOT requires Bool operand, got {:?}",
                other.dtype()
            ))),
        };
    }
    let out: Vec<Option<bool>> = match &col {
        HostColumn::Bool(v) => v
            .iter()
            .map(|c| Some(is_null_to_bool(op, c.is_none())))
            .collect(),
        HostColumn::I32(v) => v
            .iter()
            .map(|c| Some(is_null_to_bool(op, c.is_none())))
            .collect(),
        HostColumn::I64(v) => v
            .iter()
            .map(|c| Some(is_null_to_bool(op, c.is_none())))
            .collect(),
        HostColumn::F32(v) => v
            .iter()
            .map(|c| Some(is_null_to_bool(op, c.is_none())))
            .collect(),
        HostColumn::F64(v) => v
            .iter()
            .map(|c| Some(is_null_to_bool(op, c.is_none())))
            .collect(),
        HostColumn::Utf8(v) => v
            .iter()
            .map(|c| Some(is_null_to_bool(op, c.is_none())))
            .collect(),
    };
    Ok(HostColumn::Bool(out))
}

/// Map a per-row "operand was null" flag to the boolean result that
/// `op` defines. Centralised so the per-variant arms of `eval_unary`
/// don't each have to repeat the `match op { .. }` dispatch.
///
/// `UnaryOp::Not` is dispatched separately in `eval_unary` (it operates on
/// the operand value, not the validity bit) and never reaches this helper.
#[inline]
fn is_null_to_bool(op: UnaryOp, was_null: bool) -> bool {
    match op {
        UnaryOp::IsNull => was_null,
        UnaryOp::IsNotNull => !was_null,
        // Unreachable: NOT is short-circuited above in `eval_unary`. Treat
        // any leak as a programming bug — return `false` rather than
        // panicking so a regression manifests as a wrong answer that
        // tests can pin, not a runtime abort.
        UnaryOp::Not => false,
    }
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

/// Apply `+ - * /` after unifying numeric dtypes. The unified dtype is the
/// output dtype; both operands are cast to it row-by-row first.
fn eval_arithmetic(op: BinaryOp, lhs: HostColumn, rhs: HostColumn) -> BoltResult<HostColumn> {
    let l_dt = lhs.dtype();
    let r_dt = rhs.dtype();
    if !is_numeric(l_dt) || !is_numeric(r_dt) {
        return Err(BoltError::Other(format!(
            "expr_agg: arithmetic {:?} requires numeric operands, got {:?} and {:?}",
            op, l_dt, r_dt
        )));
    }
    let unified = unify_numeric(l_dt, r_dt)?;
    let lhs_u = cast_column(lhs, unified)?;
    let rhs_u = cast_column(rhs, unified)?;
    match (lhs_u, rhs_u) {
        (HostColumn::I32(a), HostColumn::I32(b)) => Ok(HostColumn::I32(zip_arith_int(op, a, b))),
        (HostColumn::I64(a), HostColumn::I64(b)) => Ok(HostColumn::I64(zip_arith_int(op, a, b))),
        (HostColumn::F32(a), HostColumn::F32(b)) => Ok(HostColumn::F32(zip_arith_float(op, a, b))),
        (HostColumn::F64(a), HostColumn::F64(b)) => Ok(HostColumn::F64(zip_arith_float(op, a, b))),
        // Unification only produces the four numeric variants above; any
        // other shape here means `cast_column` and `unify_numeric` got out
        // of sync, which would be an internal bug.
        (other_l, other_r) => Err(BoltError::Other(format!(
            "expr_agg: internal: arithmetic operand shape ({:?}, {:?})",
            other_l.dtype(),
            other_r.dtype()
        ))),
    }
}

/// Host evaluation of the integer-only operators: modulo (`%`), bitwise
/// (`& | ^`), and shifts (`<< >>`). The GPU projection path lowers these in
/// `ptx_gen`; this host path serves forced fallbacks (e.g. a `WHERE` Filter
/// using `a % b`). Operands must be `Int32`/`Int64` (float/decimal rejected,
/// matching the planner's type-check). Result width: `%`/`&`/`|`/`^` use the
/// wider of the two operands; shifts follow the LEFT operand (the right is a
/// count). Semantics match the kernel: `%` returns NULL on a zero divisor (and
/// `INT_MIN % -1`); shift amounts outside `0..bits` saturate as PTX does
/// (`shl`→0, arithmetic `shr`→sign fill). Either-side NULL propagates.
fn eval_integer_op(op: BinaryOp, lhs: HostColumn, rhs: HostColumn) -> BoltResult<HostColumn> {
    let (l_dt, r_dt) = (lhs.dtype(), rhs.dtype());
    let is_int = |dt| matches!(dt, DataType::Int32 | DataType::Int64);
    if !is_int(l_dt) || !is_int(r_dt) {
        return Err(BoltError::Type(format!(
            "expr_agg: operator {:?} requires integer (Int32/Int64) operands, got {:?} and {:?}",
            op, l_dt, r_dt
        )));
    }
    let is_shift = matches!(op, BinaryOp::Shl | BinaryOp::Shr);
    // Shifts keep the left width; the others widen to the wider operand.
    let target = if is_shift {
        l_dt
    } else if l_dt == DataType::Int64 || r_dt == DataType::Int64 {
        DataType::Int64
    } else {
        DataType::Int32
    };
    let lhs_u = cast_column(lhs, target)?;
    // The shift count is read as the same width as the value; for the other
    // ops both sides share `target`.
    let rhs_u = cast_column(rhs, target)?;
    match (lhs_u, rhs_u) {
        (HostColumn::I32(a), HostColumn::I32(b)) => Ok(HostColumn::I32(zip_integer_i32(op, a, b))),
        (HostColumn::I64(a), HostColumn::I64(b)) => Ok(HostColumn::I64(zip_integer_i64(op, a, b))),
        (other_l, other_r) => Err(BoltError::Other(format!(
            "expr_agg: internal: integer-op operand shape ({:?}, {:?})",
            other_l.dtype(),
            other_r.dtype()
        ))),
    }
}

fn zip_integer_i32(op: BinaryOp, a: Vec<Option<i32>>, b: Vec<Option<i32>>) -> Vec<Option<i32>> {
    debug_assert_eq!(a.len(), b.len());
    a.into_iter()
        .zip(b.into_iter())
        .map(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => match op {
                BinaryOp::Mod => x.checked_rem(y), // None on y==0 or i32::MIN % -1
                BinaryOp::BitAnd => Some(x & y),
                BinaryOp::BitOr => Some(x | y),
                BinaryOp::BitXor => Some(x ^ y),
                BinaryOp::Shl => Some(if (0..32).contains(&y) {
                    x.wrapping_shl(y as u32)
                } else {
                    0
                }),
                BinaryOp::Shr => Some(if (0..32).contains(&y) {
                    x >> y
                } else {
                    x >> 31
                }),
                _ => unreachable!("non-integer op routed to zip_integer_i32"),
            },
            _ => None,
        })
        .collect()
}

fn zip_integer_i64(op: BinaryOp, a: Vec<Option<i64>>, b: Vec<Option<i64>>) -> Vec<Option<i64>> {
    debug_assert_eq!(a.len(), b.len());
    a.into_iter()
        .zip(b.into_iter())
        .map(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => match op {
                BinaryOp::Mod => x.checked_rem(y),
                BinaryOp::BitAnd => Some(x & y),
                BinaryOp::BitOr => Some(x | y),
                BinaryOp::BitXor => Some(x ^ y),
                BinaryOp::Shl => Some(if (0..64).contains(&y) {
                    x.wrapping_shl(y as u32)
                } else {
                    0
                }),
                BinaryOp::Shr => Some(if (0..64).contains(&y) {
                    x >> y
                } else {
                    x >> 63
                }),
                _ => unreachable!("non-integer op routed to zip_integer_i64"),
            },
            _ => None,
        })
        .collect()
}

/// Pointwise arithmetic on an integer type. Wrapping for +/-/*; integer
/// division by zero returns `None` (SQL semantics). Either-side `None`
/// propagates.
fn zip_arith_int<T>(op: BinaryOp, a: Vec<Option<T>>, b: Vec<Option<T>>) -> Vec<Option<T>>
where
    T: IntArith,
{
    debug_assert_eq!(a.len(), b.len());
    a.into_iter()
        .zip(b.into_iter())
        .map(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => match op {
                BinaryOp::Add => Some(T::wrapping_add(x, y)),
                BinaryOp::Sub => Some(T::wrapping_sub(x, y)),
                BinaryOp::Mul => Some(T::wrapping_mul(x, y)),
                BinaryOp::Div => T::checked_div(x, y),
                _ => unreachable!("non-arithmetic op routed to zip_arith_int"),
            },
            _ => None,
        })
        .collect()
}

/// Pointwise arithmetic on a float type. NaN/inf follow IEEE-754 — in
/// particular, `1.0 / 0.0 = +inf` and `0.0 / 0.0 = NaN`. Either-side
/// `None` still propagates as `None`.
fn zip_arith_float<T>(op: BinaryOp, a: Vec<Option<T>>, b: Vec<Option<T>>) -> Vec<Option<T>>
where
    T: FloatArith,
{
    debug_assert_eq!(a.len(), b.len());
    a.into_iter()
        .zip(b.into_iter())
        .map(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => Some(match op {
                BinaryOp::Add => T::add(x, y),
                BinaryOp::Sub => T::sub(x, y),
                BinaryOp::Mul => T::mul(x, y),
                BinaryOp::Div => T::div(x, y),
                _ => unreachable!("non-arithmetic op routed to zip_arith_float"),
            }),
            _ => None,
        })
        .collect()
}

/// Integer arithmetic abstraction: wrapping ops plus checked division
/// (returns `None` on a divide-by-zero).
trait IntArith: Copy {
    fn wrapping_add(a: Self, b: Self) -> Self;
    fn wrapping_sub(a: Self, b: Self) -> Self;
    fn wrapping_mul(a: Self, b: Self) -> Self;
    fn checked_div(a: Self, b: Self) -> Option<Self>;
}

impl IntArith for i32 {
    fn wrapping_add(a: Self, b: Self) -> Self {
        a.wrapping_add(b)
    }
    fn wrapping_sub(a: Self, b: Self) -> Self {
        a.wrapping_sub(b)
    }
    fn wrapping_mul(a: Self, b: Self) -> Self {
        a.wrapping_mul(b)
    }
    fn checked_div(a: Self, b: Self) -> Option<Self> {
        if b == 0 {
            None
        } else {
            // `wrapping_div` handles `i32::MIN / -1`; the SQL spec is silent
            // on overflow here, but wrapping matches the device-side codegen.
            Some(a.wrapping_div(b))
        }
    }
}

impl IntArith for i64 {
    fn wrapping_add(a: Self, b: Self) -> Self {
        a.wrapping_add(b)
    }
    fn wrapping_sub(a: Self, b: Self) -> Self {
        a.wrapping_sub(b)
    }
    fn wrapping_mul(a: Self, b: Self) -> Self {
        a.wrapping_mul(b)
    }
    fn checked_div(a: Self, b: Self) -> Option<Self> {
        if b == 0 {
            None
        } else {
            Some(a.wrapping_div(b))
        }
    }
}

/// Float arithmetic abstraction: standard IEEE ops, no special handling.
trait FloatArith: Copy {
    fn add(a: Self, b: Self) -> Self;
    fn sub(a: Self, b: Self) -> Self;
    fn mul(a: Self, b: Self) -> Self;
    fn div(a: Self, b: Self) -> Self;
}

impl FloatArith for f32 {
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    fn sub(a: Self, b: Self) -> Self {
        a - b
    }
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
    fn div(a: Self, b: Self) -> Self {
        a / b
    }
}

impl FloatArith for f64 {
    fn add(a: Self, b: Self) -> Self {
        a + b
    }
    fn sub(a: Self, b: Self) -> Self {
        a - b
    }
    fn mul(a: Self, b: Self) -> Self {
        a * b
    }
    fn div(a: Self, b: Self) -> Self {
        a / b
    }
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

/// Apply `= <> < <= > >=`. Numeric operands are unified to the wider type
/// first (matching `Codegen::emit_binary`). Bool and Utf8 compare against
/// the same dtype only.
fn eval_comparison(op: BinaryOp, lhs: HostColumn, rhs: HostColumn) -> BoltResult<HostColumn> {
    let l_dt = lhs.dtype();
    let r_dt = rhs.dtype();

    // Numeric cross-comparisons unify first.
    if is_numeric(l_dt) && is_numeric(r_dt) {
        let unified = unify_numeric(l_dt, r_dt)?;
        let lhs_u = cast_column(lhs, unified)?;
        let rhs_u = cast_column(rhs, unified)?;
        return match (lhs_u, rhs_u) {
            (HostColumn::I32(a), HostColumn::I32(b)) => Ok(HostColumn::Bool(zip_cmp(op, &a, &b))),
            (HostColumn::I64(a), HostColumn::I64(b)) => Ok(HostColumn::Bool(zip_cmp(op, &a, &b))),
            (HostColumn::F32(a), HostColumn::F32(b)) => Ok(HostColumn::Bool(zip_cmp(op, &a, &b))),
            (HostColumn::F64(a), HostColumn::F64(b)) => Ok(HostColumn::Bool(zip_cmp(op, &a, &b))),
            (other_l, other_r) => Err(BoltError::Other(format!(
                "expr_agg: internal: comparison operand shape ({:?}, {:?})",
                other_l.dtype(),
                other_r.dtype()
            ))),
        };
    }

    if l_dt != r_dt {
        return Err(BoltError::Other(format!(
            "expr_agg: cannot compare {:?} with {:?}",
            l_dt, r_dt
        )));
    }

    match (lhs, rhs) {
        (HostColumn::Bool(a), HostColumn::Bool(b)) => Ok(HostColumn::Bool(zip_cmp(op, &a, &b))),
        (HostColumn::Utf8(a), HostColumn::Utf8(b)) => Ok(HostColumn::Bool(zip_cmp_str(op, &a, &b))),
        (l, r) => Err(BoltError::Other(format!(
            "expr_agg: internal: comparison fell through with shapes ({:?}, {:?})",
            l.dtype(),
            r.dtype()
        ))),
    }
}

/// Pointwise comparison on a partially-ordered type. NULL on either side
/// produces NULL (SQL three-valued logic).
fn zip_cmp<T>(op: BinaryOp, a: &[Option<T>], b: &[Option<T>]) -> Vec<Option<bool>>
where
    T: PartialOrd + PartialEq + Copy,
{
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => Some(apply_cmp(op, x, y)),
            _ => None,
        })
        .collect()
}

/// String version of `zip_cmp` — strings don't `Copy`, so we work with
/// references throughout.
fn zip_cmp_str(op: BinaryOp, a: &[Option<String>], b: &[Option<String>]) -> Vec<Option<bool>> {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| match (x, y) {
            (Some(x), Some(y)) => Some(apply_cmp(op, x.as_str(), y.as_str())),
            _ => None,
        })
        .collect()
}

/// Dispatch one comparison op given concrete references.
fn apply_cmp<T>(op: BinaryOp, x: T, y: T) -> bool
where
    T: PartialOrd + PartialEq,
{
    match op {
        BinaryOp::Eq => x == y,
        BinaryOp::NotEq => x != y,
        BinaryOp::Lt => x < y,
        BinaryOp::LtEq => x <= y,
        BinaryOp::Gt => x > y,
        BinaryOp::GtEq => x >= y,
        _ => unreachable!("non-comparison op routed to apply_cmp"),
    }
}

// ---------------------------------------------------------------------------
// Logical
// ---------------------------------------------------------------------------

/// Apply `AND OR`. Both operands must already be `Bool`. NULL behaves
/// per SQL three-valued logic: `NULL AND false = false`, `NULL OR true =
/// true`, otherwise NULL.
fn eval_logical(op: BinaryOp, lhs: HostColumn, rhs: HostColumn) -> BoltResult<HostColumn> {
    let l_dt = lhs.dtype();
    let r_dt = rhs.dtype();
    if l_dt != DataType::Bool || r_dt != DataType::Bool {
        return Err(BoltError::Other(format!(
            "expr_agg: logical {:?} requires Bool operands, got {:?} and {:?}",
            op, l_dt, r_dt
        )));
    }
    let (a, b) = match (lhs, rhs) {
        (HostColumn::Bool(a), HostColumn::Bool(b)) => (a, b),
        _ => unreachable!("dtype check above guarantees Bool variant"),
    };
    debug_assert_eq!(a.len(), b.len());
    let out: Vec<Option<bool>> = a
        .into_iter()
        .zip(b.into_iter())
        .map(|(x, y)| match op {
            BinaryOp::And => match (x, y) {
                (Some(false), _) | (_, Some(false)) => Some(false),
                (Some(true), Some(true)) => Some(true),
                _ => None,
            },
            BinaryOp::Or => match (x, y) {
                (Some(true), _) | (_, Some(true)) => Some(true),
                (Some(false), Some(false)) => Some(false),
                _ => None,
            },
            _ => unreachable!("non-logical op routed to eval_logical"),
        })
        .collect();
    Ok(HostColumn::Bool(out))
}

// ---------------------------------------------------------------------------
// Casts
// ---------------------------------------------------------------------------

/// Convert `col` into a column of `to`. Numeric ↔ numeric uses Rust `as`
/// (saturating for narrowing). Bool ↔ numeric: `true=1`, `false=0`, and
/// nonzero numeric → `true`. Utf8 only converts to Utf8 — anything else
/// is a type error.
fn cast_column(col: HostColumn, to: DataType) -> BoltResult<HostColumn> {
    if col.dtype() == to {
        return Ok(col);
    }
    Ok(match to {
        DataType::Bool => HostColumn::Bool(cast_to_bool(col)?),
        DataType::Int32 => HostColumn::I32(cast_to_i32(col)?),
        DataType::Int64 => HostColumn::I64(cast_to_i64(col)?),
        DataType::Float32 => HostColumn::F32(cast_to_f32(col)?),
        DataType::Float64 => HostColumn::F64(cast_to_f64(col)?),
        DataType::Utf8 => HostColumn::Utf8(cast_to_utf8(col)?),
        DataType::Decimal128(_, _) => {
            return Err(BoltError::Plan(
                "Decimal128 not yet lowered to GPU; coming in a follow-up \
                 (host-side cast)"
                    .into(),
            ))
        }
        DataType::Date32 | DataType::Timestamp(_, _) => {
            return Err(BoltError::Type(format!(
                "cast to {to:?} is not supported in the expression evaluator"
            )));
        }
    })
}

/// `TRY_CAST` / `SAFE_CAST`: like [`cast_column`] but a conversion failure on
/// a *non-null* cell yields `None` (SQL NULL) instead of an error, and the
/// conversion envelope is wider (string parses are accepted). A `None` input
/// cell stays `None` regardless. The per-conversion-class behaviour:
///
///   * Identity (`src == target`): returned unchanged (never fails).
///   * `Utf8 -> {Int32,Int64,Float32,Float64,Bool}`: parse the trimmed
///     string; unparseable / out-of-range → `None`.
///   * Float -> Int (`Float{32,64} -> Int{32,64}`): non-finite (NaN/±inf) or
///     out of the target integer's range → `None`; otherwise truncate toward
///     zero (matches Rust `as`, which the GPU `cvt.rzi` path also uses for
///     in-range values).
///   * Int64 -> Int32: out of `i32` range → `None`; otherwise the value.
///   * Every other supported pair (int widening, int->float, float widening,
///     bool<->int): cannot fail, so identical to the plain cast.
fn safe_cast_column(col: HostColumn, to: DataType) -> BoltResult<HostColumn> {
    use DataType::*;
    if col.dtype() == to {
        return Ok(col);
    }
    Ok(match to {
        Int32 => HostColumn::I32(safe_cast_to_i32(col)?),
        Int64 => HostColumn::I64(safe_cast_to_i64(col)?),
        Float32 => HostColumn::F32(safe_cast_to_f32(col)?),
        Float64 => HostColumn::F64(safe_cast_to_f64(col)?),
        Bool => HostColumn::Bool(safe_cast_to_bool(col)?),
        Utf8 => HostColumn::Utf8(cast_to_utf8(col)?),
        Decimal128(_, _) => {
            return Err(BoltError::Plan(
                "TRY_CAST to Decimal128 not yet supported (host-side); \
                 coming in a follow-up"
                    .into(),
            ))
        }
        Date32 | Timestamp(_, _) => {
            return Err(BoltError::Type(format!(
                "TRY_CAST to {to:?} is not supported in the expression evaluator"
            )));
        }
    })
}

/// Safe cast to `Vec<Option<i32>>`. Utf8 parses (failure → None); Int64 and
/// floats range-check (out-of-range / non-finite → None); other numerics and
/// Bool cannot fail.
fn safe_cast_to_i32(col: HostColumn) -> BoltResult<Vec<Option<i32>>> {
    Ok(match col {
        HostColumn::Bool(v) => v.into_iter().map(|o| o.map(|b| b as i32)).collect(),
        HostColumn::I32(v) => v,
        HostColumn::I64(v) => v
            .into_iter()
            .map(|o| o.and_then(|x| i32::try_from(x).ok()))
            .collect(),
        HostColumn::F32(v) => v
            .into_iter()
            .map(|o| {
                o.and_then(|x| {
                    float_to_int_checked(x as f64, i32::MIN as f64, i32::MAX as f64)
                        .map(|y| y as i32)
                })
            })
            .collect(),
        HostColumn::F64(v) => v
            .into_iter()
            .map(|o| {
                o.and_then(|x| {
                    float_to_int_checked(x, i32::MIN as f64, i32::MAX as f64).map(|y| y as i32)
                })
            })
            .collect(),
        HostColumn::Utf8(v) => v
            .into_iter()
            .map(|o| o.and_then(|s| s.trim().parse::<i32>().ok()))
            .collect(),
    })
}

/// Safe cast to `Vec<Option<i64>>`.
fn safe_cast_to_i64(col: HostColumn) -> BoltResult<Vec<Option<i64>>> {
    Ok(match col {
        HostColumn::Bool(v) => v.into_iter().map(|o| o.map(|b| b as i64)).collect(),
        HostColumn::I32(v) => v.into_iter().map(|o| o.map(|x| x as i64)).collect(),
        HostColumn::I64(v) => v,
        HostColumn::F32(v) => v
            .into_iter()
            .map(|o| {
                o.and_then(|x| {
                    float_to_int_checked(x as f64, i64::MIN as f64, i64::MAX as f64)
                        .map(|y| y as i64)
                })
            })
            .collect(),
        HostColumn::F64(v) => v
            .into_iter()
            .map(|o| {
                o.and_then(|x| {
                    float_to_int_checked(x, i64::MIN as f64, i64::MAX as f64).map(|y| y as i64)
                })
            })
            .collect(),
        HostColumn::Utf8(v) => v
            .into_iter()
            .map(|o| o.and_then(|s| s.trim().parse::<i64>().ok()))
            .collect(),
    })
}

/// Safe cast to `Vec<Option<f32>>`. Only Utf8 parses can fail; numeric / Bool
/// conversions are total (matching the plain-cast `as` behaviour).
fn safe_cast_to_f32(col: HostColumn) -> BoltResult<Vec<Option<f32>>> {
    Ok(match col {
        HostColumn::Utf8(v) => v
            .into_iter()
            .map(|o| o.and_then(|s| s.trim().parse::<f32>().ok()))
            .collect(),
        other => cast_to_f32(other)?,
    })
}

/// Safe cast to `Vec<Option<f64>>`.
fn safe_cast_to_f64(col: HostColumn) -> BoltResult<Vec<Option<f64>>> {
    Ok(match col {
        HostColumn::Utf8(v) => v
            .into_iter()
            .map(|o| o.and_then(|s| s.trim().parse::<f64>().ok()))
            .collect(),
        other => cast_to_f64(other)?,
    })
}

/// Safe cast to `Vec<Option<bool>>`. Utf8 accepts the SQL/`bool` spellings
/// `true/false`, `t/f`, `1/0`, `yes/no` (case-insensitive, trimmed); anything
/// else → None. Numeric → `x != 0` (total).
fn safe_cast_to_bool(col: HostColumn) -> BoltResult<Vec<Option<bool>>> {
    Ok(match col {
        HostColumn::Utf8(v) => v
            .into_iter()
            .map(|o| o.and_then(|s| parse_bool_loose(&s)))
            .collect(),
        other => cast_to_bool(other)?,
    })
}

/// Parse a string to bool for `TRY_CAST(... AS BOOL)`. Accepts the common SQL
/// truthy/falsy spellings, case-insensitively, after trimming whitespace.
/// Returns `None` for anything unrecognised.
fn parse_bool_loose(s: &str) -> Option<bool> {
    match s.trim().to_ascii_lowercase().as_str() {
        "true" | "t" | "1" | "yes" | "y" => Some(true),
        "false" | "f" | "0" | "no" | "n" => Some(false),
        _ => None,
    }
}

/// Truncate-toward-zero a finite float into an inclusive `[lo, hi]` range,
/// returning `None` if the value is non-finite (NaN/±inf) or rounds outside
/// the range. `lo`/`hi` are the target integer type's bounds expressed as
/// `f64`. The caller applies the final `as` narrowing once the value is known
/// to be in range, so no saturation/UB occurs.
fn float_to_int_checked(x: f64, lo: f64, hi: f64) -> Option<f64> {
    if !x.is_finite() {
        return None;
    }
    let t = x.trunc();
    if t < lo || t > hi {
        None
    } else {
        Some(t)
    }
}

/// Cast to `Vec<Option<i32>>`. Bool → 0/1; numerics use `as i32`
/// (saturating); Utf8 errors.
fn cast_to_i32(col: HostColumn) -> BoltResult<Vec<Option<i32>>> {
    Ok(match col {
        HostColumn::Bool(v) => v.into_iter().map(|o| o.map(|b| b as i32)).collect(),
        HostColumn::I32(v) => v,
        HostColumn::I64(v) => v.into_iter().map(|o| o.map(|x| x as i32)).collect(),
        HostColumn::F32(v) => v.into_iter().map(|o| o.map(|x| x as i32)).collect(),
        HostColumn::F64(v) => v.into_iter().map(|o| o.map(|x| x as i32)).collect(),
        HostColumn::Utf8(_) => {
            return Err(BoltError::Other(
                "expr_agg: cannot cast Utf8 to Int32".into(),
            ))
        }
    })
}

/// Cast to `Vec<Option<i64>>`.
fn cast_to_i64(col: HostColumn) -> BoltResult<Vec<Option<i64>>> {
    Ok(match col {
        HostColumn::Bool(v) => v.into_iter().map(|o| o.map(|b| b as i64)).collect(),
        HostColumn::I32(v) => v.into_iter().map(|o| o.map(|x| x as i64)).collect(),
        HostColumn::I64(v) => v,
        HostColumn::F32(v) => v.into_iter().map(|o| o.map(|x| x as i64)).collect(),
        HostColumn::F64(v) => v.into_iter().map(|o| o.map(|x| x as i64)).collect(),
        HostColumn::Utf8(_) => {
            return Err(BoltError::Other(
                "expr_agg: cannot cast Utf8 to Int64".into(),
            ))
        }
    })
}

/// Cast to `Vec<Option<f32>>`.
fn cast_to_f32(col: HostColumn) -> BoltResult<Vec<Option<f32>>> {
    Ok(match col {
        HostColumn::Bool(v) => v
            .into_iter()
            .map(|o| o.map(|b| if b { 1.0 } else { 0.0 }))
            .collect(),
        HostColumn::I32(v) => v.into_iter().map(|o| o.map(|x| x as f32)).collect(),
        HostColumn::I64(v) => v.into_iter().map(|o| o.map(|x| x as f32)).collect(),
        HostColumn::F32(v) => v,
        HostColumn::F64(v) => v.into_iter().map(|o| o.map(|x| x as f32)).collect(),
        HostColumn::Utf8(_) => {
            return Err(BoltError::Other(
                "expr_agg: cannot cast Utf8 to Float32".into(),
            ))
        }
    })
}

/// Cast to `Vec<Option<f64>>`.
fn cast_to_f64(col: HostColumn) -> BoltResult<Vec<Option<f64>>> {
    Ok(match col {
        HostColumn::Bool(v) => v
            .into_iter()
            .map(|o| o.map(|b| if b { 1.0 } else { 0.0 }))
            .collect(),
        HostColumn::I32(v) => v.into_iter().map(|o| o.map(|x| x as f64)).collect(),
        HostColumn::I64(v) => v.into_iter().map(|o| o.map(|x| x as f64)).collect(),
        HostColumn::F32(v) => v.into_iter().map(|o| o.map(|x| x as f64)).collect(),
        HostColumn::F64(v) => v,
        HostColumn::Utf8(_) => {
            return Err(BoltError::Other(
                "expr_agg: cannot cast Utf8 to Float64".into(),
            ))
        }
    })
}

/// Cast to `Vec<Option<bool>>`. Numeric → `x != 0`; Utf8 errors.
fn cast_to_bool(col: HostColumn) -> BoltResult<Vec<Option<bool>>> {
    Ok(match col {
        HostColumn::Bool(v) => v,
        HostColumn::I32(v) => v.into_iter().map(|o| o.map(|x| x != 0)).collect(),
        HostColumn::I64(v) => v.into_iter().map(|o| o.map(|x| x != 0)).collect(),
        HostColumn::F32(v) => v.into_iter().map(|o| o.map(|x| x != 0.0)).collect(),
        HostColumn::F64(v) => v.into_iter().map(|o| o.map(|x| x != 0.0)).collect(),
        HostColumn::Utf8(_) => {
            return Err(BoltError::Other(
                "expr_agg: cannot cast Utf8 to Bool".into(),
            ))
        }
    })
}

/// Cast to `Vec<Option<String>>`. Only legal from Utf8 itself.
fn cast_to_utf8(col: HostColumn) -> BoltResult<Vec<Option<String>>> {
    match col {
        HostColumn::Utf8(v) => Ok(v),
        other => Err(BoltError::Other(format!(
            "expr_agg: cannot cast {:?} to Utf8",
            other.dtype()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Numeric unification — byte-for-byte clone of
// `crate::plan::physical_plan::unify_numeric`, intentionally private here.
// ---------------------------------------------------------------------------

fn unify_numeric(a: DataType, b: DataType) -> BoltResult<DataType> {
    use DataType::*;
    match (a, b) {
        (x, y) if x == y => Ok(x),
        (Float64, _) | (_, Float64) => Ok(Float64),
        (Float32, Int64) | (Int64, Float32) => Ok(Float64),
        (Float32, _) | (_, Float32) => Ok(Float32),
        (Int64, _) | (_, Int64) => Ok(Int64),
        (Int32, _) | (_, Int32) => Ok(Int32),
        _ => Err(BoltError::Other(format!(
            "expr_agg: cannot unify {:?} and {:?}",
            a, b
        ))),
    }
}

/// True for any numeric (int or float) dtype.
fn is_numeric(d: DataType) -> bool {
    matches!(
        d,
        DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64
    )
}

/// Local mirror of the operator-category predicates from `physical_plan.rs`
/// — duplicated rather than re-exported to keep this module self-contained.
fn is_arithmetic(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
    )
}

/// True for `= <> < <= > >=`.
fn is_comparison(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::LtEq
            | BinaryOp::Gt
            | BinaryOp::GtEq
    )
}

/// True for `AND OR`.
fn is_logical(op: BinaryOp) -> bool {
    matches!(op, BinaryOp::And | BinaryOp::Or)
}

/// True for string-valued binary ops — today only `||` (Concat).
fn is_string(op: BinaryOp) -> bool {
    matches!(op, BinaryOp::Concat)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{col, lit, AggregateExpr, Expr, Literal};

    /// Build an env from a list of `(name, &HostColumn)` pairs. Tests
    /// keep their column storage as local stack/heap values and just
    /// hand references to the env.
    fn env_of<'a>(cols: &[(&str, &'a HostColumn)]) -> ColumnEnv<'a> {
        let mut m: ColumnEnv<'a> = HashMap::new();
        for (n, c) in cols {
            m.insert((*n).to_string(), *c);
        }
        m
    }

    // -----------------------------------------------------------------------
    // H1 / V-10: ensure_no_surviving_nulls collapse-point guard.
    // -----------------------------------------------------------------------

    #[test]
    fn ensure_no_surviving_nulls_ok_when_all_some() {
        // A correct caller (NULLs already filtered upstream) passes the guard
        // and observes no behavior change.
        let col = HostColumn::I32(vec![Some(1), Some(2), Some(3)]);
        assert!(!col.has_nulls());
        col.ensure_no_surviving_nulls("test_all_some")
            .expect("dense column must pass the guard");
    }

    #[test]
    fn ensure_no_surviving_nulls_ok_when_empty() {
        // An empty column trivially has no surviving NULLs.
        let col = HostColumn::F64(Vec::new());
        col.ensure_no_surviving_nulls("test_empty")
            .expect("empty column must pass the guard");
    }

    // In release builds (debug_assertions off) the guard returns an error;
    // in debug builds it panics via debug_assert! before reaching the error.
    // Gate the error-path assertion on the build profile so the test reflects
    // the active behavior either way.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "surviving NULL"))]
    fn ensure_no_surviving_nulls_catches_violation_i64() {
        let col = HostColumn::I64(vec![Some(1), None, Some(3)]);
        assert!(col.has_nulls());
        let res = col.ensure_no_surviving_nulls("test_violation");
        // Only reached in release builds (debug_assert compiled out); in debug
        // the debug_assert! above panics first and `should_panic` validates it.
        match res {
            Err(BoltError::Other(msg)) => {
                assert!(
                    msg.contains("surviving NULL") && msg.contains("test_violation"),
                    "unexpected guard error message: {msg}"
                );
            }
            other => panic!("expected BoltError::Other on contract violation, got {other:?}"),
        }
    }

    #[test]
    fn has_nulls_detects_none_in_each_variant() {
        assert!(HostColumn::Bool(vec![Some(true), None]).has_nulls());
        assert!(HostColumn::I32(vec![None]).has_nulls());
        assert!(HostColumn::F32(vec![Some(1.0), None]).has_nulls());
        assert!(HostColumn::Utf8(vec![Some("a".to_string()), None]).has_nulls());
        assert!(!HostColumn::I64(vec![Some(1), Some(2)]).has_nulls());
        assert!(!HostColumn::Utf8(vec![Some("x".to_string())]).has_nulls());
    }

    #[test]
    fn eval_bare_column() {
        let x = HostColumn::I64(vec![Some(1), Some(2), Some(3)]);
        let env = env_of(&[("x", &x)]);
        let out = eval_expr(&col("x"), &env, DataType::Int64, 3).unwrap();
        match out {
            HostColumn::I64(v) => assert_eq!(v, vec![Some(1), Some(2), Some(3)]),
            other => panic!("expected I64, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn eval_literal() {
        let env: ColumnEnv = HashMap::new();
        let out = eval_expr(&Expr::Literal(Literal::Int64(7)), &env, DataType::Int64, 3).unwrap();
        match out {
            HostColumn::I64(v) => assert_eq!(v, vec![Some(7), Some(7), Some(7)]),
            other => panic!("expected I64, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn eval_alias_passes_through() {
        let x = HostColumn::I32(vec![Some(5), Some(6)]);
        let env = env_of(&[("x", &x)]);
        let expr = col("x").alias("renamed");
        let out = eval_expr(&expr, &env, DataType::Int32, 2).unwrap();
        match out {
            HostColumn::I32(v) => assert_eq!(v, vec![Some(5), Some(6)]),
            other => panic!("expected I32, got {:?}", other.dtype()),
        }
    }

    // -------------------------------------------------------------------
    // F4: TRY_CAST / SAFE_CAST host evaluator (NULL-on-failure)
    // -------------------------------------------------------------------

    /// `TRY_CAST('abc' AS INT)` → NULL (unparseable string).
    #[test]
    fn try_cast_bad_string_to_int_is_null() {
        let s = HostColumn::Utf8(vec![Some("abc".into())]);
        let env = env_of(&[("s", &s)]);
        let expr = col("s").try_cast(DataType::Int32);
        let out = eval_expr(&expr, &env, DataType::Int32, 1).unwrap();
        match out {
            HostColumn::I32(v) => assert_eq!(v, vec![None]),
            other => panic!("expected I32, got {:?}", other.dtype()),
        }
    }

    /// `TRY_CAST('123' AS INT)` → 123 (good parse).
    #[test]
    fn try_cast_good_string_to_int() {
        let s = HostColumn::Utf8(vec![Some("123".into()), Some("  -7 ".into())]);
        let env = env_of(&[("s", &s)]);
        let expr = col("s").try_cast(DataType::Int32);
        let out = eval_expr(&expr, &env, DataType::Int32, 2).unwrap();
        match out {
            HostColumn::I32(v) => assert_eq!(v, vec![Some(123), Some(-7)]),
            other => panic!("expected I32, got {:?}", other.dtype()),
        }
    }

    /// NULL input → NULL output, independent of `safe`.
    #[test]
    fn try_cast_of_null_is_null() {
        let s = HostColumn::Utf8(vec![None]);
        let env = env_of(&[("s", &s)]);
        let expr = col("s").try_cast(DataType::Int64);
        let out = eval_expr(&expr, &env, DataType::Int64, 1).unwrap();
        match out {
            HostColumn::I64(v) => assert_eq!(v, vec![None]),
            other => panic!("expected I64, got {:?}", other.dtype()),
        }
    }

    /// Out-of-range narrowing under TRY_CAST → NULL; in-range value survives.
    #[test]
    fn try_cast_out_of_range_narrowing_is_null() {
        let x = HostColumn::I64(vec![Some(i64::from(i32::MAX) + 1), Some(42)]);
        let env = env_of(&[("x", &x)]);
        let expr = col("x").try_cast(DataType::Int32);
        let out = eval_expr(&expr, &env, DataType::Int32, 2).unwrap();
        match out {
            HostColumn::I32(v) => assert_eq!(v, vec![None, Some(42)]),
            other => panic!("expected I32, got {:?}", other.dtype()),
        }
    }

    /// Out-of-range / non-finite float→int under TRY_CAST → NULL.
    #[test]
    fn try_cast_float_to_int_overflow_and_nan_are_null() {
        let x = HostColumn::F64(vec![Some(1e30), Some(f64::NAN), Some(3.9)]);
        let env = env_of(&[("x", &x)]);
        let expr = col("x").try_cast(DataType::Int32);
        let out = eval_expr(&expr, &env, DataType::Int32, 3).unwrap();
        match out {
            // 1e30 overflows i32 → None; NaN → None; 3.9 truncates to 3.
            HostColumn::I32(v) => assert_eq!(v, vec![None, None, Some(3)]),
            other => panic!("expected I32, got {:?}", other.dtype()),
        }
    }

    /// A *plain* CAST of the same unparseable string is NOT silently NULL —
    /// the host evaluator errors (preserving plain-CAST semantics). The GPU
    /// path likewise rejects Utf8 source; here we assert the host contract.
    #[test]
    fn plain_cast_bad_string_to_int_errors() {
        let s = HostColumn::Utf8(vec![Some("abc".into())]);
        let env = env_of(&[("s", &s)]);
        let expr = col("s").cast(DataType::Int32);
        let err = eval_expr(&expr, &env, DataType::Int32, 1);
        assert!(
            err.is_err(),
            "plain CAST of bad string must error, not NULL"
        );
    }

    /// A widening safe cast (Int32→Int64) cannot fail: identical to plain CAST.
    #[test]
    fn try_cast_widening_matches_plain() {
        let x = HostColumn::I32(vec![Some(1), Some(-2), None]);
        let env = env_of(&[("x", &x)]);
        let out = eval_expr(
            &col("x").try_cast(DataType::Int64),
            &env,
            DataType::Int64,
            3,
        )
        .unwrap();
        match out {
            HostColumn::I64(v) => assert_eq!(v, vec![Some(1), Some(-2), None]),
            other => panic!("expected I64, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn eval_binary_add_int_int() {
        let a = HostColumn::I32(vec![Some(1), Some(2), Some(3)]);
        let b = HostColumn::I32(vec![Some(10), Some(20), Some(30)]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let expr = col("a").add(col("b"));
        let out = eval_expr(&expr, &env, DataType::Int32, 3).unwrap();
        match out {
            HostColumn::I32(v) => assert_eq!(v, vec![Some(11), Some(22), Some(33)]),
            other => panic!("expected I32, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn eval_binary_add_int_float() {
        // i32 + f64 unifies to f64.
        let a = HostColumn::I32(vec![Some(1), Some(2)]);
        let b = HostColumn::F64(vec![Some(0.5), Some(0.25)]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let expr = col("a").add(col("b"));
        let out = eval_expr(&expr, &env, DataType::Float64, 2).unwrap();
        match out {
            HostColumn::F64(v) => {
                assert_eq!(v.len(), 2);
                assert!((v[0].unwrap() - 1.5).abs() < 1e-12);
                assert!((v[1].unwrap() - 2.25).abs() < 1e-12);
            }
            other => panic!("expected F64, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn eval_binary_lt_returns_bool() {
        let a = HostColumn::I32(vec![Some(1), Some(2), Some(3)]);
        let b = HostColumn::I32(vec![Some(2), Some(2), Some(2)]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let expr = col("a").lt(col("b"));
        let out = eval_expr(&expr, &env, DataType::Bool, 3).unwrap();
        match out {
            HostColumn::Bool(v) => {
                assert_eq!(v, vec![Some(true), Some(false), Some(false)]);
            }
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn eval_null_propagates() {
        // 5 + NULL → NULL across an addition.
        let a = HostColumn::I32(vec![Some(5), None, Some(3)]);
        let b = HostColumn::I32(vec![None, Some(7), Some(4)]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let expr = col("a").add(col("b"));
        let out = eval_expr(&expr, &env, DataType::Int32, 3).unwrap();
        match out {
            HostColumn::I32(v) => assert_eq!(v, vec![None, None, Some(7)]),
            other => panic!("expected I32, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn eval_div_by_zero_int_returns_null() {
        let a = HostColumn::I32(vec![Some(10), Some(20), Some(30)]);
        let b = HostColumn::I32(vec![Some(2), Some(0), Some(5)]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let expr = col("a").div(col("b"));
        let out = eval_expr(&expr, &env, DataType::Int32, 3).unwrap();
        match out {
            HostColumn::I32(v) => assert_eq!(v, vec![Some(5), None, Some(6)]),
            other => panic!("expected I32, got {:?}", other.dtype()),
        }
    }

    /// Float division by zero follows IEEE-754: positive/0.0 → +inf,
    /// 0.0/0.0 → NaN. We test both shapes here.
    #[test]
    fn eval_div_by_zero_float_returns_nan() {
        let a = HostColumn::F64(vec![Some(1.0), Some(0.0)]);
        let b = HostColumn::F64(vec![Some(0.0), Some(0.0)]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let expr = col("a").div(col("b"));
        let out = eval_expr(&expr, &env, DataType::Float64, 2).unwrap();
        match out {
            HostColumn::F64(v) => {
                let r0 = v[0].unwrap();
                let r1 = v[1].unwrap();
                assert!(
                    r0.is_infinite() && r0 > 0.0,
                    "1.0/0.0 should be +inf, got {}",
                    r0
                );
                assert!(r1.is_nan(), "0.0/0.0 should be NaN, got {}", r1);
            }
            other => panic!("expected F64, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn eval_logical_and_or() {
        // AND / OR over Bool columns with a NULL mixed in.
        let a = HostColumn::Bool(vec![Some(true), Some(true), Some(false), None]);
        let b = HostColumn::Bool(vec![Some(true), Some(false), Some(false), Some(true)]);
        let env = env_of(&[("a", &a), ("b", &b)]);

        let and_out = eval_expr(&col("a").and(col("b")), &env, DataType::Bool, 4).unwrap();
        match and_out {
            HostColumn::Bool(v) => {
                // T&T=T, T&F=F, F&F=F, NULL&T=NULL (per 3-valued logic).
                assert_eq!(v, vec![Some(true), Some(false), Some(false), None]);
            }
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }

        let or_out = eval_expr(&col("a").or(col("b")), &env, DataType::Bool, 4).unwrap();
        match or_out {
            HostColumn::Bool(v) => {
                // T|T=T, T|F=T, F|F=F, NULL|T=T.
                assert_eq!(v, vec![Some(true), Some(true), Some(false), Some(true)]);
            }
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn materialize_agg_input_simple() {
        // SUM(price * tax) where price is f64 and tax is f64.
        let price = HostColumn::F64(vec![Some(10.0), Some(20.0), Some(30.0)]);
        let tax = HostColumn::F64(vec![Some(0.1), Some(0.2), Some(0.5)]);
        let env = env_of(&[("price", &price), ("tax", &tax)]);

        let agg = AggregateExpr::Sum(col("price").mul(col("tax")));
        let out = materialize_agg_input(&agg, &env, DataType::Float64, 3).unwrap();
        match out {
            HostColumn::F64(v) => {
                assert_eq!(v.len(), 3);
                assert!((v[0].unwrap() - 1.0).abs() < 1e-12);
                assert!((v[1].unwrap() - 4.0).abs() < 1e-12);
                assert!((v[2].unwrap() - 15.0).abs() < 1e-12);
            }
            other => panic!("expected F64, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn try_bare_column_recognises_bare_and_alias() {
        assert_eq!(try_bare_column(&col("a")), Some("a"));
        assert_eq!(try_bare_column(&col("a").alias("b")), Some("a"));
        assert_eq!(try_bare_column(&lit(1i64)), None);
        assert_eq!(try_bare_column(&col("a").add(col("b"))), None);
    }

    // -----------------------------------------------------------------
    // SQL three-valued-logic (3VL) invariant tests.
    //
    // Each test below pins one row of the precise NULL-propagation
    // contract documented at the top of the file. If any of these
    // tests change behaviour, the doc contract MUST be updated to
    // match — these are the load-bearing invariants other modules
    // (especially the consumers in agg_with_pre / groupby_with_pre)
    // rely on.
    // -----------------------------------------------------------------

    /// `eval(col + NULL_lit) = NULL` for every row. Exercises the
    /// "right operand is None" arm of arithmetic.
    #[test]
    fn null_3vl_arith_a_plus_null_literal() {
        let a = HostColumn::I32(vec![Some(1), Some(2), Some(3)]);
        let env = env_of(&[("a", &a)]);
        let expr = col("a").add(Expr::Literal(Literal::Null));
        let out = eval_expr(&expr, &env, DataType::Int32, 3).unwrap();
        match out {
            HostColumn::I32(v) => assert_eq!(v, vec![None, None, None]),
            other => panic!("expected I32, got {:?}", other.dtype()),
        }
    }

    /// `eval(NULL_lit + col) = NULL` for every row. Exercises the
    /// "left operand is None" arm of arithmetic.
    #[test]
    fn null_3vl_arith_null_literal_plus_b() {
        let b = HostColumn::I32(vec![Some(10), Some(20), Some(30)]);
        let env = env_of(&[("b", &b)]);
        let expr = Expr::Literal(Literal::Null).add(col("b"));
        let out = eval_expr(&expr, &env, DataType::Int32, 3).unwrap();
        match out {
            HostColumn::I32(v) => assert_eq!(v, vec![None, None, None]),
            other => panic!("expected I32, got {:?}", other.dtype()),
        }
    }

    /// Every arithmetic op propagates None symmetrically. We sweep
    /// `+ - * /` against a single (Some, None) and (None, Some) pair.
    #[test]
    fn null_3vl_arith_all_ops_propagate_both_sides() {
        for use_left_null in [false, true] {
            let a = if use_left_null {
                HostColumn::I64(vec![None])
            } else {
                HostColumn::I64(vec![Some(7)])
            };
            let b = if use_left_null {
                HostColumn::I64(vec![Some(3)])
            } else {
                HostColumn::I64(vec![None])
            };
            let env = env_of(&[("a", &a), ("b", &b)]);
            for expr in [
                col("a").add(col("b")),
                col("a").sub(col("b")),
                col("a").mul(col("b")),
                col("a").div(col("b")),
            ] {
                let out = eval_expr(&expr, &env, DataType::Int64, 1).unwrap();
                match out {
                    HostColumn::I64(v) => assert_eq!(
                        v,
                        vec![None],
                        "expected None for op with use_left_null={use_left_null}"
                    ),
                    other => panic!("expected I64, got {:?}", other.dtype()),
                }
            }
        }
    }

    /// `NULL = NULL` is `None`, NOT `Some(true)`. This is the most
    /// commonly mis-implemented 3VL rule.
    #[test]
    fn null_3vl_cmp_null_eq_null_is_null() {
        let a = HostColumn::I32(vec![None]);
        let b = HostColumn::I32(vec![None]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let out = eval_expr(&col("a").eq(col("b")), &env, DataType::Bool, 1).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(v, vec![None]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// All six comparison ops produce `None` whenever either side is `None`.
    #[test]
    fn null_3vl_cmp_all_ops_propagate() {
        // Row 0: (NULL, 1)  Row 1: (1, NULL)  Row 2: (NULL, NULL)
        let a = HostColumn::I32(vec![None, Some(1), None]);
        let b = HostColumn::I32(vec![Some(1), None, None]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        for expr in [
            col("a").eq(col("b")),
            col("a").neq(col("b")),
            col("a").lt(col("b")),
            col("a").lt_eq(col("b")),
            col("a").gt(col("b")),
            col("a").gt_eq(col("b")),
        ] {
            let out = eval_expr(&expr, &env, DataType::Bool, 3).unwrap();
            match out {
                HostColumn::Bool(v) => assert_eq!(v, vec![None, None, None]),
                other => panic!("expected Bool, got {:?}", other.dtype()),
            }
        }
    }

    /// `NULL AND false = false`. The absorbing element wins, NULL does
    /// not infect. This and the `NULL AND true = NULL` row below are
    /// the two rules that distinguish 3VL from naive `Option<bool>`
    /// propagation.
    #[test]
    fn null_3vl_logical_null_and_false_is_false() {
        let a = HostColumn::Bool(vec![None]);
        let b = HostColumn::Bool(vec![Some(false)]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        // Both orderings: NULL AND false, false AND NULL.
        let out_lr = eval_expr(&col("a").and(col("b")), &env, DataType::Bool, 1).unwrap();
        let out_rl = eval_expr(&col("b").and(col("a")), &env, DataType::Bool, 1).unwrap();
        match (out_lr, out_rl) {
            (HostColumn::Bool(lr), HostColumn::Bool(rl)) => {
                assert_eq!(lr, vec![Some(false)], "NULL AND false");
                assert_eq!(rl, vec![Some(false)], "false AND NULL");
            }
            _ => panic!("expected Bool"),
        }
    }

    /// `NULL AND true = NULL` (no absorbing element on the true side
    /// for AND), and `NULL AND NULL = NULL`.
    #[test]
    fn null_3vl_logical_null_and_true_is_null() {
        let a = HostColumn::Bool(vec![None, None]);
        let b = HostColumn::Bool(vec![Some(true), None]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let out = eval_expr(&col("a").and(col("b")), &env, DataType::Bool, 2).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(v, vec![None, None]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
        // Symmetric: true AND NULL is also NULL.
        let out_rev = eval_expr(&col("b").and(col("a")), &env, DataType::Bool, 2).unwrap();
        match out_rev {
            HostColumn::Bool(v) => assert_eq!(v, vec![None, None]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// `NULL OR true = true`. The absorbing element wins.
    #[test]
    fn null_3vl_logical_null_or_true_is_true() {
        let a = HostColumn::Bool(vec![None]);
        let b = HostColumn::Bool(vec![Some(true)]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let out_lr = eval_expr(&col("a").or(col("b")), &env, DataType::Bool, 1).unwrap();
        let out_rl = eval_expr(&col("b").or(col("a")), &env, DataType::Bool, 1).unwrap();
        match (out_lr, out_rl) {
            (HostColumn::Bool(lr), HostColumn::Bool(rl)) => {
                assert_eq!(lr, vec![Some(true)], "NULL OR true");
                assert_eq!(rl, vec![Some(true)], "true OR NULL");
            }
            _ => panic!("expected Bool"),
        }
    }

    /// `NULL OR false = NULL` (no absorbing element on the false side
    /// for OR), and `NULL OR NULL = NULL`.
    #[test]
    fn null_3vl_logical_null_or_false_is_null() {
        let a = HostColumn::Bool(vec![None, None]);
        let b = HostColumn::Bool(vec![Some(false), None]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let out = eval_expr(&col("a").or(col("b")), &env, DataType::Bool, 2).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(v, vec![None, None]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
        let out_rev = eval_expr(&col("b").or(col("a")), &env, DataType::Bool, 2).unwrap();
        match out_rev {
            HostColumn::Bool(v) => assert_eq!(v, vec![None, None]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// `Literal::Null` cast to any numeric dtype is still all-None, and
    /// participates correctly in arithmetic. This pins the
    /// "casts preserve None" contract.
    #[test]
    fn null_3vl_cast_null_literal_preserves_none() {
        // NULL_lit cast to Float64, then added to a Float64 column,
        // should give all-None.
        let a = HostColumn::F64(vec![Some(1.0), Some(2.0)]);
        let env = env_of(&[("a", &a)]);
        // Build NULL literal then add to a — natural type for NULL_lit
        // is I64; unify_numeric(F64, I64) = F64, so the add operates in F64.
        let expr = col("a").add(Expr::Literal(Literal::Null));
        let out = eval_expr(&expr, &env, DataType::Float64, 2).unwrap();
        match out {
            HostColumn::F64(v) => assert_eq!(v, vec![None, None]),
            other => panic!("expected F64, got {:?}", other.dtype()),
        }
    }

    /// Comparing a non-NULL Bool with a NULL Bool yields NULL, not
    /// Some(false). Bool-vs-Bool comparison is a separate code path
    /// from numeric-vs-numeric so we pin it explicitly.
    #[test]
    fn null_3vl_cmp_bool_with_null_is_null() {
        let a = HostColumn::Bool(vec![Some(true), Some(false), None]);
        let b = HostColumn::Bool(vec![None, None, Some(true)]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let out = eval_expr(&col("a").eq(col("b")), &env, DataType::Bool, 3).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(v, vec![None, None, None]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// Comparing two Utf8 strings where either side is NULL yields NULL.
    /// The string compare path is a separate function from the numeric
    /// one, so it gets its own test.
    #[test]
    fn null_3vl_cmp_utf8_with_null_is_null() {
        let a = HostColumn::Utf8(vec![Some("x".into()), None]);
        let b = HostColumn::Utf8(vec![None, Some("y".into())]);
        let env = env_of(&[("a", &a), ("b", &b)]);
        let out = eval_expr(&col("a").eq(col("b")), &env, DataType::Bool, 2).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(v, vec![None, None]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    // -- IS NULL / IS NOT NULL ------------------------------------------------

    /// `col IS NULL` on a mixed nullable column should flag exactly the
    /// rows whose Option is None, with no NULL bleed-through (the result
    /// of IS [NOT] NULL is itself always defined).
    #[test]
    fn eval_unary_is_null_on_nullable_int() {
        let x = HostColumn::I32(vec![Some(1), None, Some(3), None]);
        let env = env_of(&[("x", &x)]);
        let expr = col("x").is_null();
        let out = eval_expr(&expr, &env, DataType::Bool, 4).unwrap();
        match out {
            HostColumn::Bool(v) => {
                assert_eq!(v, vec![Some(false), Some(true), Some(false), Some(true)])
            }
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// `col IS NOT NULL` must be the pointwise inverse of `IS NULL` over
    /// the same operand.
    #[test]
    fn eval_unary_is_not_null_is_inverse() {
        let x = HostColumn::I32(vec![Some(1), None, Some(3), None]);
        let env = env_of(&[("x", &x)]);
        let expr = col("x").is_not_null();
        let out = eval_expr(&expr, &env, DataType::Bool, 4).unwrap();
        match out {
            HostColumn::Bool(v) => {
                assert_eq!(v, vec![Some(true), Some(false), Some(true), Some(false)])
            }
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// `col IS NULL` on an all-Some column produces all-false (and
    /// crucially never None — even though the input dtype is nullable).
    #[test]
    fn eval_unary_is_null_on_all_non_null_is_all_false() {
        let x = HostColumn::I64(vec![Some(1), Some(2), Some(3)]);
        let env = env_of(&[("x", &x)]);
        let expr = col("x").is_null();
        let out = eval_expr(&expr, &env, DataType::Bool, 3).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(v, vec![Some(false), Some(false), Some(false)]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// Works uniformly over every HostColumn variant — pin the Utf8
    /// path separately since it has a distinct branch in `eval_unary`.
    #[test]
    fn eval_unary_is_null_works_for_utf8() {
        let s = HostColumn::Utf8(vec![Some("a".into()), None, Some("c".into())]);
        let env = env_of(&[("s", &s)]);
        let out = eval_expr(&col("s").is_null(), &env, DataType::Bool, 3).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(v, vec![Some(false), Some(true), Some(false)]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// Aliases inside the operand should be transparent (mirrors how
    /// `eval_binary` and `Expr::Alias` handle the same).
    #[test]
    fn eval_unary_is_null_sees_through_alias() {
        let x = HostColumn::I32(vec![Some(1), None]);
        let env = env_of(&[("x", &x)]);
        // (x AS renamed) IS NULL
        let inner = col("x").alias("renamed");
        let expr = Expr::Unary {
            op: UnaryOp::IsNull,
            operand: Box::new(inner),
        };
        let out = eval_expr(&expr, &env, DataType::Bool, 2).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(v, vec![Some(false), Some(true)]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// `IS NULL` over a `Literal::Null` broadcasts a True column —
    /// because the literal evaluates to an all-`None` HostColumn, every
    /// row is "operand was null" → Some(true).
    #[test]
    fn eval_unary_is_null_on_literal_null_is_all_true() {
        let env: ColumnEnv = HashMap::new();
        let expr = Expr::Unary {
            op: UnaryOp::IsNull,
            operand: Box::new(Expr::Literal(Literal::Null)),
        };
        let out = eval_expr(&expr, &env, DataType::Bool, 3).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(v, vec![Some(true), Some(true), Some(true)]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// `IS NULL` over a non-null literal should be all-false.
    #[test]
    fn eval_unary_is_null_on_literal_int_is_all_false() {
        let env: ColumnEnv = HashMap::new();
        let expr = Expr::Unary {
            op: UnaryOp::IsNull,
            operand: Box::new(Expr::Literal(Literal::Int64(42))),
        };
        let out = eval_expr(&expr, &env, DataType::Bool, 2).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(v, vec![Some(false), Some(false)]),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    // -----------------------------------------------------------------
    // SUBSTRING / TRIM host evaluation (Expr::ScalarFn).
    // -----------------------------------------------------------------

    use crate::plan::logical_plan::ScalarFnKind;

    fn scalar_fn(kind: ScalarFnKind, args: Vec<Expr>) -> Expr {
        Expr::ScalarFn { kind, args }
    }

    fn utf8(strs: &[Option<&str>]) -> HostColumn {
        HostColumn::Utf8(strs.iter().map(|o| o.map(|s| s.to_string())).collect())
    }

    #[test]
    fn substring_three_arg_over_column() {
        let s = utf8(&[Some("hello"), Some("world"), None]);
        let env = env_of(&[("s", &s)]);
        let expr = scalar_fn(
            ScalarFnKind::Substring,
            vec![col("s"), lit(2i64), lit(3i64)],
        );
        let out = eval_expr(&expr, &env, DataType::Utf8, 3).unwrap();
        match out {
            HostColumn::Utf8(v) => assert_eq!(
                v,
                vec![Some("ell".to_string()), Some("orl".to_string()), None]
            ),
            other => panic!("expected Utf8, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn substring_two_arg_goes_to_end() {
        let s = utf8(&[Some("hello")]);
        let env = env_of(&[("s", &s)]);
        let expr = scalar_fn(ScalarFnKind::Substring, vec![col("s"), lit(3i64)]);
        let out = eval_expr(&expr, &env, DataType::Utf8, 1).unwrap();
        match out {
            HostColumn::Utf8(v) => assert_eq!(v, vec![Some("llo".to_string())]),
            other => panic!("expected Utf8, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn substring_null_position_is_null() {
        let s = utf8(&[Some("hello")]);
        let env = env_of(&[("s", &s)]);
        let expr = scalar_fn(
            ScalarFnKind::Substring,
            vec![col("s"), Expr::Literal(Literal::Null), lit(2i64)],
        );
        let out = eval_expr(&expr, &env, DataType::Utf8, 1).unwrap();
        match out {
            HostColumn::Utf8(v) => assert_eq!(v, vec![None]),
            other => panic!("expected Utf8, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn trim_both_default_whitespace() {
        let s = utf8(&[Some("  hi  "), Some("nope"), None]);
        let env = env_of(&[("s", &s)]);
        let expr = scalar_fn(ScalarFnKind::TrimBoth, vec![col("s")]);
        let out = eval_expr(&expr, &env, DataType::Utf8, 3).unwrap();
        match out {
            HostColumn::Utf8(v) => assert_eq!(
                v,
                vec![Some("hi".to_string()), Some("nope".to_string()), None]
            ),
            other => panic!("expected Utf8, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn trim_leading_and_trailing_custom_chars() {
        let s = utf8(&[Some("xxhixx")]);
        let env = env_of(&[("s", &s)]);

        let lead = eval_expr(
            &scalar_fn(ScalarFnKind::TrimLeading, vec![col("s"), lit_str("x")]),
            &env,
            DataType::Utf8,
            1,
        )
        .unwrap();
        match lead {
            HostColumn::Utf8(v) => assert_eq!(v, vec![Some("hixx".to_string())]),
            other => panic!("expected Utf8, got {:?}", other.dtype()),
        }

        let trail = eval_expr(
            &scalar_fn(ScalarFnKind::TrimTrailing, vec![col("s"), lit_str("x")]),
            &env,
            DataType::Utf8,
            1,
        )
        .unwrap();
        match trail {
            HostColumn::Utf8(v) => assert_eq!(v, vec![Some("xxhi".to_string())]),
            other => panic!("expected Utf8, got {:?}", other.dtype()),
        }
    }

    #[test]
    fn trim_null_source_is_null() {
        let s = utf8(&[None]);
        let env = env_of(&[("s", &s)]);
        let expr = scalar_fn(ScalarFnKind::TrimBoth, vec![col("s")]);
        let out = eval_expr(&expr, &env, DataType::Utf8, 1).unwrap();
        match out {
            HostColumn::Utf8(v) => assert_eq!(v, vec![None]),
            other => panic!("expected Utf8, got {:?}", other.dtype()),
        }
    }

    /// Helper: build a Utf8 literal expression.
    fn lit_str(s: &str) -> Expr {
        Expr::Literal(Literal::Utf8(s.to_string()))
    }

    // -----------------------------------------------------------------
    // ILIKE host evaluation (Expr::Like { case_insensitive }).
    // -----------------------------------------------------------------

    /// Build an `Expr::Like` over column `s`.
    fn like_expr(pattern: &str, negated: bool, case_insensitive: bool) -> Expr {
        Expr::Like {
            expr: Box::new(col("s")),
            pattern: pattern.to_string(),
            escape: None,
            negated,
            case_insensitive,
        }
    }

    /// `ILIKE` matches across case AND propagates NULL (3VL): the NULL row
    /// stays NULL, never false.
    #[test]
    fn ilike_matches_across_case_and_propagates_null() {
        let s = utf8(&[Some("FOO"), Some("food"), None, Some("bar")]);
        let env = env_of(&[("s", &s)]);
        let expr = like_expr("foo%", false, true);
        let out = eval_expr(&expr, &env, DataType::Bool, 4).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(
                v,
                vec![Some(true), Some(true), None, Some(false)],
                "FOO/food match 'foo%' case-insensitively; NULL stays NULL"
            ),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// `NOT ILIKE` inverts the per-row boolean but keeps NULL as NULL.
    #[test]
    fn not_ilike_inverts_and_preserves_null() {
        let s = utf8(&[Some("FOO"), None, Some("bar")]);
        let env = env_of(&[("s", &s)]);
        let expr = like_expr("foo", true, true);
        let out = eval_expr(&expr, &env, DataType::Bool, 3).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(
                v,
                vec![Some(false), None, Some(true)],
                "FOO ILIKE 'foo' is true → NOT ILIKE false; NULL stays NULL"
            ),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    /// Plain `LIKE` host eval is UNCHANGED: a case difference does NOT match,
    /// and NULL still propagates.
    #[test]
    fn plain_like_host_eval_is_case_sensitive() {
        let s = utf8(&[Some("FOO"), Some("foo"), None]);
        let env = env_of(&[("s", &s)]);
        let expr = like_expr("foo%", false, false);
        let out = eval_expr(&expr, &env, DataType::Bool, 3).unwrap();
        match out {
            HostColumn::Bool(v) => assert_eq!(
                v,
                vec![Some(false), Some(true), None],
                "case-sensitive LIKE: FOO does not match 'foo%'; NULL stays NULL"
            ),
            other => panic!("expected Bool, got {:?}", other.dtype()),
        }
    }

    // -----------------------------------------------------------------
    // CAST FORMAT host evaluator (Expr::CastFormat). Feature CAST FORMAT.
    // These exercise `eval_cast_format` directly via `eval_expr` over a
    // HostColumn env (no GPU, no Arrow round-trip). Temporals live as their
    // storage in HostColumn: Date32 as I32 (days), Timestamp as I64 (ticks).
    // -----------------------------------------------------------------

    use crate::plan::logical_plan::FormatToken;

    /// `parse_civil` / `format_civil` round-trip helper: build a date pattern.
    fn date_pattern() -> Vec<FormatToken> {
        vec![
            FormatToken::Year4,
            FormatToken::Literal('-'),
            FormatToken::Month,
            FormatToken::Literal('-'),
            FormatToken::Day,
        ]
    }

    fn cast_format(expr: Expr, target: DataType, pattern: Vec<FormatToken>, to_text: bool) -> Expr {
        Expr::CastFormat {
            expr: Box::new(expr),
            target,
            pattern,
            to_text,
        }
    }

    /// Date32 → string: format the day count `0` (1970-01-01) and a known
    /// reference (`10957` = 2000-01-01) with `YYYY-MM-DD`.
    #[test]
    fn cast_format_date32_to_string() {
        // Day 0 = 1970-01-01; day 10957 = 2000-01-01; day -1 = 1969-12-31.
        let d = HostColumn::I32(vec![Some(0), Some(10_957), Some(-1), None]);
        let env = env_of(&[("d", &d)]);
        let expr = cast_format(col("d"), DataType::Utf8, date_pattern(), true);
        let out = eval_expr(&expr, &env, DataType::Utf8, 4).unwrap();
        match out {
            HostColumn::Utf8(v) => assert_eq!(
                v,
                vec![
                    Some("1970-01-01".to_string()),
                    Some("2000-01-01".to_string()),
                    Some("1969-12-31".to_string()),
                    None,
                ]
            ),
            other => panic!("expected Utf8, got {:?}", other.dtype()),
        }
    }

    /// String → Date32 → string round-trips losslessly for the supported
    /// `YYYY-MM-DD` pattern.
    #[test]
    fn cast_format_string_to_date32_round_trip() {
        let s = utf8(&[
            Some("1970-01-01"),
            Some("2000-01-01"),
            Some("2024-02-29"),
            None,
        ]);
        let env = env_of(&[("s", &s)]);
        // string → Date32 (I32 storage)
        let to_date = cast_format(col("s"), DataType::Date32, date_pattern(), false);
        let days = eval_expr(&to_date, &env, DataType::Int32, 4).unwrap();
        let day_vals = match &days {
            HostColumn::I32(v) => v.clone(),
            other => panic!("expected I32, got {:?}", other.dtype()),
        };
        assert_eq!(
            day_vals,
            vec![Some(0), Some(10_957), Some(19_782), None],
            "parsed day counts (2024-02-29 = day 19782)"
        );
        // Date32 → string brings us back to the original text.
        let env2 = env_of(&[("d", &days)]);
        let back = cast_format(col("d"), DataType::Utf8, date_pattern(), true);
        let out = eval_expr(&back, &env2, DataType::Utf8, 4).unwrap();
        match out {
            HostColumn::Utf8(v) => assert_eq!(
                v,
                vec![
                    Some("1970-01-01".to_string()),
                    Some("2000-01-01".to_string()),
                    Some("2024-02-29".to_string()),
                    None,
                ]
            ),
            other => panic!("expected Utf8, got {:?}", other.dtype()),
        }
    }

    /// Timestamp (nanosecond ticks) → string with a full date-time pattern.
    #[test]
    fn cast_format_timestamp_to_string() {
        // 2000-01-01 00:00:00 UTC = 946684800 s = 946684800_000_000_000 ns.
        // Add 13h 37m 09s within the day.
        let base_secs: i64 = 946_684_800 + 13 * 3600 + 37 * 60 + 9;
        let ts = HostColumn::I64(vec![Some(base_secs * 1_000_000_000), None]);
        let env = env_of(&[("ts", &ts)]);
        let pattern = vec![
            FormatToken::Year4,
            FormatToken::Literal('-'),
            FormatToken::Month,
            FormatToken::Literal('-'),
            FormatToken::Day,
            FormatToken::Literal(' '),
            FormatToken::Hour24,
            FormatToken::Literal(':'),
            FormatToken::Minute,
            FormatToken::Literal(':'),
            FormatToken::Second,
        ];
        let expr = cast_format(col("ts"), DataType::Utf8, pattern, true);
        let out = eval_expr(&expr, &env, DataType::Utf8, 2).unwrap();
        match out {
            HostColumn::Utf8(v) => {
                assert_eq!(v, vec![Some("2000-01-01 13:37:09".to_string()), None])
            }
            other => panic!("expected Utf8, got {:?}", other.dtype()),
        }
    }

    /// String → Timestamp (nanosecond) parses the full date-time pattern and
    /// round-trips back to the same string.
    #[test]
    fn cast_format_string_to_timestamp_round_trip() {
        let s = utf8(&[Some("2000-01-01 13:37:09")]);
        let env = env_of(&[("s", &s)]);
        let pattern = vec![
            FormatToken::Year4,
            FormatToken::Literal('-'),
            FormatToken::Month,
            FormatToken::Literal('-'),
            FormatToken::Day,
            FormatToken::Literal(' '),
            FormatToken::Hour24,
            FormatToken::Literal(':'),
            FormatToken::Minute,
            FormatToken::Literal(':'),
            FormatToken::Second,
        ];
        let to_ts = cast_format(
            col("s"),
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            pattern.clone(),
            false,
        );
        let ticks = eval_expr(&to_ts, &env, DataType::Int64, 1).unwrap();
        let base_secs: i64 = 946_684_800 + 13 * 3600 + 37 * 60 + 9;
        match &ticks {
            HostColumn::I64(v) => {
                assert_eq!(v, &vec![Some(base_secs * 1_000_000_000)])
            }
            other => panic!("expected I64, got {:?}", other.dtype()),
        }
        let env2 = env_of(&[("ts", &ticks)]);
        let back = cast_format(col("ts"), DataType::Utf8, pattern, true);
        let out = eval_expr(&back, &env2, DataType::Utf8, 1).unwrap();
        match out {
            HostColumn::Utf8(v) => {
                assert_eq!(v, vec![Some("2000-01-01 13:37:09".to_string())])
            }
            other => panic!("expected Utf8, got {:?}", other.dtype()),
        }
    }

    /// A string that does not match the pattern is a hard error (plain-CAST
    /// semantics — FORMAT is only carried by plain CAST).
    #[test]
    fn cast_format_string_to_date32_bad_input_errors() {
        let s = utf8(&[Some("not-a-date")]);
        let env = env_of(&[("s", &s)]);
        let expr = cast_format(col("s"), DataType::Date32, date_pattern(), false);
        let err = eval_expr(&expr, &env, DataType::Int32, 1).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("does not match the format pattern"),
            "expected a parse-failure error, got: {msg}"
        );
    }

    /// An impossible calendar date (Feb 30) is rejected by the range check.
    #[test]
    fn cast_format_string_to_date32_rejects_impossible_date() {
        let s = utf8(&[Some("2023-02-30")]);
        let env = env_of(&[("s", &s)]);
        let expr = cast_format(col("s"), DataType::Date32, date_pattern(), false);
        assert!(
            eval_expr(&expr, &env, DataType::Int32, 1).is_err(),
            "Feb 30 must not parse"
        );
    }
}
