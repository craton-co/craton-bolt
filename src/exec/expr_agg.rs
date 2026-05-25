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
//!
//! Anything else returns `PatinaError::Other` with a `{:?}` of the
//! offending expression. CAST, CASE, NULLIF, unary ops, scalar functions
//! and so on are explicitly out of scope: the lowering does not produce
//! them today, and adding them belongs in a separate change so that the
//! GPU codegen path can keep up.
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
//! ## NULL propagation
//!
//! Every operand position carries `Option<T>`. A `None` on *either* side of
//! a binary op produces a `None` result, regardless of operator. This is
//! the SQL semantics that the device-side codegen targets (NULL is a
//! "third value" that infects compute). The one wrinkle is integer
//! division by zero: per SQL, that is also `None`. Float division by zero
//! follows IEEE-754 — positive / 0.0 is `+inf`, 0.0 / 0.0 is `NaN`, and so
//! on. (We chose IEEE for floats because the device path also relies on
//! IEEE behaviour; tagging those as `None` would diverge from the GPU.)
//!
//! ## Tests
//!
//! Self-contained unit tests live in the `#[cfg(test)] mod tests` at the
//! bottom of the file. They exercise the public API only; no GPU calls.

use std::collections::HashMap;

use crate::error::{PatinaError, PatinaResult};
use crate::plan::logical_plan::{AggregateExpr, BinaryOp, DataType, Expr, Literal};

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
) -> PatinaResult<HostColumn> {
    let raw = eval_inner(expr, env, n_rows)?;
    if raw.len() != n_rows {
        return Err(PatinaError::Other(format!(
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
) -> PatinaResult<HostColumn> {
    let inner = match agg {
        AggregateExpr::Sum(e)
        | AggregateExpr::Min(e)
        | AggregateExpr::Max(e)
        | AggregateExpr::Avg(e)
        | AggregateExpr::Count(e) => e,
    };
    eval_expr(inner, env, expected_dtype, n_rows)
}

// ---------------------------------------------------------------------------
// Evaluator core
// ---------------------------------------------------------------------------

/// Like `eval_expr` but does *not* coerce the result to a caller-chosen
/// dtype. Returns the natural dtype produced by the expression.
fn eval_inner(
    expr: &Expr,
    env: &ColumnEnv<'_>,
    n_rows: usize,
) -> PatinaResult<HostColumn> {
    match expr {
        Expr::Column(name) => eval_column(name, env, n_rows),
        Expr::Literal(lit) => eval_literal(lit, n_rows),
        Expr::Alias(inner, _) => eval_inner(inner, env, n_rows),
        Expr::Binary { op, left, right } => eval_binary(*op, left, right, env, n_rows),
    }
}

/// Look up `name` in `env` and clone the referenced column. Validates the
/// length against `n_rows`. The dtype is left alone — the outer
/// `eval_expr` is responsible for the final cast.
fn eval_column(
    name: &str,
    env: &ColumnEnv<'_>,
    n_rows: usize,
) -> PatinaResult<HostColumn> {
    let col = env.get(name).ok_or_else(|| {
        PatinaError::Plan(format!(
            "expr_agg: column '{name}' not found in evaluator env"
        ))
    })?;
    if col.len() != n_rows {
        return Err(PatinaError::Other(format!(
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
fn eval_literal(lit: &Literal, n_rows: usize) -> PatinaResult<HostColumn> {
    Ok(match lit {
        Literal::Null => HostColumn::I64(vec![None; n_rows]),
        Literal::Bool(b) => HostColumn::Bool(vec![Some(*b); n_rows]),
        Literal::Int32(v) => HostColumn::I32(vec![Some(*v); n_rows]),
        Literal::Int64(v) => HostColumn::I64(vec![Some(*v); n_rows]),
        Literal::Float32(v) => HostColumn::F32(vec![Some(*v); n_rows]),
        Literal::Float64(v) => HostColumn::F64(vec![Some(*v); n_rows]),
        Literal::Utf8(s) => HostColumn::Utf8(vec![Some(s.clone()); n_rows]),
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
) -> PatinaResult<HostColumn> {
    let l = eval_inner(left, env, n_rows)?;
    let r = eval_inner(right, env, n_rows)?;
    if l.len() != n_rows || r.len() != n_rows {
        return Err(PatinaError::Other(format!(
            "expr_agg: binary operand length mismatch: lhs={}, rhs={}, expected={}",
            l.len(),
            r.len(),
            n_rows
        )));
    }

    if is_arithmetic(op) {
        eval_arithmetic(op, l, r)
    } else if is_comparison(op) {
        eval_comparison(op, l, r)
    } else if is_logical(op) {
        eval_logical(op, l, r)
    } else {
        Err(PatinaError::Other(format!(
            "expr_agg: unsupported operator {:?}",
            op
        )))
    }
}

// ---------------------------------------------------------------------------
// Arithmetic
// ---------------------------------------------------------------------------

/// Apply `+ - * /` after unifying numeric dtypes. The unified dtype is the
/// output dtype; both operands are cast to it row-by-row first.
fn eval_arithmetic(
    op: BinaryOp,
    lhs: HostColumn,
    rhs: HostColumn,
) -> PatinaResult<HostColumn> {
    let l_dt = lhs.dtype();
    let r_dt = rhs.dtype();
    if !is_numeric(l_dt) || !is_numeric(r_dt) {
        return Err(PatinaError::Other(format!(
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
        (other_l, other_r) => Err(PatinaError::Other(format!(
            "expr_agg: internal: arithmetic operand shape ({:?}, {:?})",
            other_l.dtype(),
            other_r.dtype()
        ))),
    }
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
fn eval_comparison(
    op: BinaryOp,
    lhs: HostColumn,
    rhs: HostColumn,
) -> PatinaResult<HostColumn> {
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
            (other_l, other_r) => Err(PatinaError::Other(format!(
                "expr_agg: internal: comparison operand shape ({:?}, {:?})",
                other_l.dtype(),
                other_r.dtype()
            ))),
        };
    }

    if l_dt != r_dt {
        return Err(PatinaError::Other(format!(
            "expr_agg: cannot compare {:?} with {:?}",
            l_dt, r_dt
        )));
    }

    match (lhs, rhs) {
        (HostColumn::Bool(a), HostColumn::Bool(b)) => Ok(HostColumn::Bool(zip_cmp(op, &a, &b))),
        (HostColumn::Utf8(a), HostColumn::Utf8(b)) => Ok(HostColumn::Bool(zip_cmp_str(op, &a, &b))),
        (l, r) => Err(PatinaError::Other(format!(
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
fn zip_cmp_str(
    op: BinaryOp,
    a: &[Option<String>],
    b: &[Option<String>],
) -> Vec<Option<bool>> {
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
fn eval_logical(
    op: BinaryOp,
    lhs: HostColumn,
    rhs: HostColumn,
) -> PatinaResult<HostColumn> {
    let l_dt = lhs.dtype();
    let r_dt = rhs.dtype();
    if l_dt != DataType::Bool || r_dt != DataType::Bool {
        return Err(PatinaError::Other(format!(
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
fn cast_column(col: HostColumn, to: DataType) -> PatinaResult<HostColumn> {
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
    })
}

/// Cast to `Vec<Option<i32>>`. Bool → 0/1; numerics use `as i32`
/// (saturating); Utf8 errors.
fn cast_to_i32(col: HostColumn) -> PatinaResult<Vec<Option<i32>>> {
    Ok(match col {
        HostColumn::Bool(v) => v.into_iter().map(|o| o.map(|b| b as i32)).collect(),
        HostColumn::I32(v) => v,
        HostColumn::I64(v) => v.into_iter().map(|o| o.map(|x| x as i32)).collect(),
        HostColumn::F32(v) => v.into_iter().map(|o| o.map(|x| x as i32)).collect(),
        HostColumn::F64(v) => v.into_iter().map(|o| o.map(|x| x as i32)).collect(),
        HostColumn::Utf8(_) => {
            return Err(PatinaError::Other(
                "expr_agg: cannot cast Utf8 to Int32".into(),
            ))
        }
    })
}

/// Cast to `Vec<Option<i64>>`.
fn cast_to_i64(col: HostColumn) -> PatinaResult<Vec<Option<i64>>> {
    Ok(match col {
        HostColumn::Bool(v) => v.into_iter().map(|o| o.map(|b| b as i64)).collect(),
        HostColumn::I32(v) => v.into_iter().map(|o| o.map(|x| x as i64)).collect(),
        HostColumn::I64(v) => v,
        HostColumn::F32(v) => v.into_iter().map(|o| o.map(|x| x as i64)).collect(),
        HostColumn::F64(v) => v.into_iter().map(|o| o.map(|x| x as i64)).collect(),
        HostColumn::Utf8(_) => {
            return Err(PatinaError::Other(
                "expr_agg: cannot cast Utf8 to Int64".into(),
            ))
        }
    })
}

/// Cast to `Vec<Option<f32>>`.
fn cast_to_f32(col: HostColumn) -> PatinaResult<Vec<Option<f32>>> {
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
            return Err(PatinaError::Other(
                "expr_agg: cannot cast Utf8 to Float32".into(),
            ))
        }
    })
}

/// Cast to `Vec<Option<f64>>`.
fn cast_to_f64(col: HostColumn) -> PatinaResult<Vec<Option<f64>>> {
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
            return Err(PatinaError::Other(
                "expr_agg: cannot cast Utf8 to Float64".into(),
            ))
        }
    })
}

/// Cast to `Vec<Option<bool>>`. Numeric → `x != 0`; Utf8 errors.
fn cast_to_bool(col: HostColumn) -> PatinaResult<Vec<Option<bool>>> {
    Ok(match col {
        HostColumn::Bool(v) => v,
        HostColumn::I32(v) => v.into_iter().map(|o| o.map(|x| x != 0)).collect(),
        HostColumn::I64(v) => v.into_iter().map(|o| o.map(|x| x != 0)).collect(),
        HostColumn::F32(v) => v.into_iter().map(|o| o.map(|x| x != 0.0)).collect(),
        HostColumn::F64(v) => v.into_iter().map(|o| o.map(|x| x != 0.0)).collect(),
        HostColumn::Utf8(_) => {
            return Err(PatinaError::Other(
                "expr_agg: cannot cast Utf8 to Bool".into(),
            ))
        }
    })
}

/// Cast to `Vec<Option<String>>`. Only legal from Utf8 itself.
fn cast_to_utf8(col: HostColumn) -> PatinaResult<Vec<Option<String>>> {
    match col {
        HostColumn::Utf8(v) => Ok(v),
        other => Err(PatinaError::Other(format!(
            "expr_agg: cannot cast {:?} to Utf8",
            other.dtype()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Numeric unification — byte-for-byte clone of
// `crate::plan::physical_plan::unify_numeric`, intentionally private here.
// ---------------------------------------------------------------------------

fn unify_numeric(a: DataType, b: DataType) -> PatinaResult<DataType> {
    use DataType::*;
    match (a, b) {
        (x, y) if x == y => Ok(x),
        (Float64, _) | (_, Float64) => Ok(Float64),
        (Float32, Int64) | (Int64, Float32) => Ok(Float64),
        (Float32, _) | (_, Float32) => Ok(Float32),
        (Int64, _) | (_, Int64) => Ok(Int64),
        (Int32, _) | (_, Int32) => Ok(Int32),
        _ => Err(PatinaError::Other(format!(
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
        let out = eval_expr(
            &Expr::Literal(Literal::Int64(7)),
            &env,
            DataType::Int64,
            3,
        )
        .unwrap();
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
                assert!(r0.is_infinite() && r0 > 0.0, "1.0/0.0 should be +inf, got {}", r0);
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
}
