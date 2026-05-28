// SPDX-License-Identifier: Apache-2.0

//! Logical plan AST: schemas, expressions, and relational nodes.

use crate::error::{BoltError, BoltResult};

/// Minimal set of column data types the GPU engine handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// Boolean (one byte on device).
    Bool,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit IEEE-754 float.
    Float32,
    /// 64-bit IEEE-754 float.
    Float64,
    /// UTF-8 string; variable width, only legal in filter/group-by columns.
    Utf8,
}

impl DataType {
    /// Byte width for fixed-width types; `None` for variable-width.
    pub fn byte_width(self) -> Option<usize> {
        match self {
            DataType::Bool => Some(1),
            DataType::Int32 => Some(4),
            DataType::Int64 => Some(8),
            DataType::Float32 => Some(4),
            DataType::Float64 => Some(8),
            DataType::Utf8 => None,
        }
    }

    /// True for the floating-point types.
    fn is_float(self) -> bool {
        matches!(self, DataType::Float32 | DataType::Float64)
    }

    /// True for the integer types.
    fn is_int(self) -> bool {
        matches!(self, DataType::Int32 | DataType::Int64)
    }

    /// True for any numeric (int or float) type.
    fn is_numeric(self) -> bool {
        self.is_int() || self.is_float()
    }
}

/// A named, typed column slot in a schema.
#[derive(Debug, Clone)]
pub struct Field {
    /// Column name.
    pub name: String,
    /// Column data type.
    pub dtype: DataType,
    /// Whether the column admits nulls.
    pub nullable: bool,
}

impl Field {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, dtype: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            dtype,
            nullable,
        }
    }
}

/// Ordered list of fields describing a relation.
#[derive(Debug, Clone, Default)]
pub struct Schema {
    /// Fields in projection order.
    pub fields: Vec<Field>,
}

impl Schema {
    /// Build a schema from a vector of fields.
    pub fn new(fields: Vec<Field>) -> Self {
        Self { fields }
    }

    /// Index of `name` in this schema, or a `Plan` error if absent.
    pub fn index_of(&self, name: &str) -> BoltResult<usize> {
        self.fields
            .iter()
            .position(|f| f.name == name)
            .ok_or_else(|| BoltError::Plan(format!("column '{name}' not found in schema")))
    }

    /// Lookup a field by name, or a `Plan` error if absent.
    pub fn field(&self, name: &str) -> BoltResult<&Field> {
        let i = self.index_of(name)?;
        Ok(&self.fields[i])
    }
}

/// A scalar constant.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// SQL NULL — no static type.
    Null,
    /// Boolean constant.
    Bool(bool),
    /// 32-bit integer constant.
    Int32(i32),
    /// 64-bit integer constant.
    Int64(i64),
    /// 32-bit float constant.
    Float32(f32),
    /// 64-bit float constant.
    Float64(f64),
    /// UTF-8 string constant.
    Utf8(String),
}

impl Literal {
    /// Static type of this literal; `None` for `Null`.
    pub fn dtype(&self) -> Option<DataType> {
        match self {
            Literal::Null => None,
            Literal::Bool(_) => Some(DataType::Bool),
            Literal::Int32(_) => Some(DataType::Int32),
            Literal::Int64(_) => Some(DataType::Int64),
            Literal::Float32(_) => Some(DataType::Float32),
            Literal::Float64(_) => Some(DataType::Float64),
            Literal::Utf8(_) => Some(DataType::Utf8),
        }
    }
}

impl From<bool> for Literal {
    fn from(v: bool) -> Self {
        Literal::Bool(v)
    }
}

impl From<i32> for Literal {
    fn from(v: i32) -> Self {
        Literal::Int32(v)
    }
}

impl From<i64> for Literal {
    fn from(v: i64) -> Self {
        Literal::Int64(v)
    }
}

impl From<f32> for Literal {
    fn from(v: f32) -> Self {
        Literal::Float32(v)
    }
}

impl From<f64> for Literal {
    fn from(v: f64) -> Self {
        Literal::Float64(v)
    }
}

impl From<&str> for Literal {
    fn from(v: &str) -> Self {
        Literal::Utf8(v.to_string())
    }
}

impl From<String> for Literal {
    fn from(v: String) -> Self {
        Literal::Utf8(v)
    }
}

/// Binary operators codegen handles directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// `a / b`.
    Div,
    /// `a = b`.
    Eq,
    /// `a <> b`.
    NotEq,
    /// `a < b`.
    Lt,
    /// `a <= b`.
    LtEq,
    /// `a > b`.
    Gt,
    /// `a >= b`.
    GtEq,
    /// `a AND b`.
    And,
    /// `a OR b`.
    Or,
}

impl BinaryOp {
    /// True for `+ - * /`.
    fn is_arithmetic(self) -> bool {
        matches!(self, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div)
    }

    /// True for `= <> < <= > >=`.
    fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq
        )
    }

    /// True for `AND OR`.
    fn is_logical(self) -> bool {
        matches!(self, BinaryOp::And | BinaryOp::Or)
    }
}

/// Unary operators surfaced by the planner.
///
/// Covers SQL `IS NULL` / `IS NOT NULL` and logical `NOT`. These are
/// type-checked at the logical-plan level (`IS [NOT] NULL` always produces
/// `Bool` regardless of operand dtype; `NOT` requires a `Bool` operand and
/// produces `Bool`) and surfaced through the SQL frontend. The GPU executor
/// lowers bare-column `IS [NOT] NULL` natively; `NOT` is currently rejected
/// at the physical-plan boundary so the host-side filter path can handle it
/// without misleading the user about kernel support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// SQL `<expr> IS NULL`.
    IsNull,
    /// SQL `<expr> IS NOT NULL`.
    IsNotNull,
    /// SQL `NOT <bool-expr>`. Operand must be `Bool`; result is `Bool`.
    Not,
}

/// Scalar expression tree.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Reference to an input column by name.
    Column(String),
    /// Scalar constant.
    Literal(Literal),
    /// Two-operand expression.
    Binary {
        /// Operator.
        op: BinaryOp,
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    /// One-operand expression (currently only `IS NULL` / `IS NOT NULL`).
    ///
    /// Always type-checks to `Bool` at the logical plane regardless of the
    /// operand's dtype, including the untyped `Literal::Null` operand. The
    /// physical planner does not yet lower this to a GPU kernel — see
    /// [`crate::plan::physical_plan`].
    Unary {
        /// Unary operator.
        op: UnaryOp,
        /// The single operand.
        operand: Box<Expr>,
    },
    /// Rename an expression in the output schema.
    Alias(Box<Expr>, String),
}

/// Build a column reference expression.
pub fn col(name: impl Into<String>) -> Expr {
    Expr::Column(name.into())
}

/// Build a literal expression from anything that converts into `Literal`.
pub fn lit<T: Into<Literal>>(v: T) -> Expr {
    Expr::Literal(v.into())
}

fn binary(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(l),
        right: Box::new(r),
    }
}

impl Expr {
    /// Wrap `self` in an `Alias`.
    pub fn alias(self, name: impl Into<String>) -> Expr {
        Expr::Alias(Box::new(self), name.into())
    }

    /// `self + rhs`.
    pub fn add(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Add, self, rhs)
    }

    /// `self - rhs`.
    pub fn sub(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Sub, self, rhs)
    }

    /// `self * rhs`.
    pub fn mul(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Mul, self, rhs)
    }

    /// `self / rhs`.
    pub fn div(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Div, self, rhs)
    }

    /// `self = rhs`.
    pub fn eq(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Eq, self, rhs)
    }

    /// `self <> rhs`.
    pub fn neq(self, rhs: Expr) -> Expr {
        binary(BinaryOp::NotEq, self, rhs)
    }

    /// `self < rhs`.
    pub fn lt(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Lt, self, rhs)
    }

    /// `self <= rhs`.
    pub fn lt_eq(self, rhs: Expr) -> Expr {
        binary(BinaryOp::LtEq, self, rhs)
    }

    /// `self > rhs`.
    pub fn gt(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Gt, self, rhs)
    }

    /// `self >= rhs`.
    pub fn gt_eq(self, rhs: Expr) -> Expr {
        binary(BinaryOp::GtEq, self, rhs)
    }

    /// `self AND rhs`.
    pub fn and(self, rhs: Expr) -> Expr {
        binary(BinaryOp::And, self, rhs)
    }

    /// `self OR rhs`.
    pub fn or(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Or, self, rhs)
    }

    /// `self IS NULL`. Returns a Bool expression, never null.
    pub fn is_null(self) -> Expr {
        Expr::Unary {
            op: UnaryOp::IsNull,
            operand: Box::new(self),
        }
    }

    /// `self IS NOT NULL`. Returns a Bool expression, never null.
    pub fn is_not_null(self) -> Expr {
        Expr::Unary {
            op: UnaryOp::IsNotNull,
            operand: Box::new(self),
        }
    }

    /// `NOT self`. Operand must type-check to `Bool`; the result is `Bool`.
    pub fn not(self) -> Expr {
        Expr::Unary {
            op: UnaryOp::Not,
            operand: Box::new(self),
        }
    }

    /// Resolve the static type of this expression against `schema`.
    // TODO(nullable): add CASE/COALESCE/IS NULL/IS NOT NULL variants
    pub fn dtype(&self, schema: &Schema) -> BoltResult<DataType> {
        self.dtype_depth(schema, 0)
    }

    /// Inner recursion for [`Expr::dtype`]. `depth` is the current recursion
    /// depth; returns Err if [`crate::plan::sql_frontend::MAX_RECURSION_DEPTH`]
    /// is exceeded — guards against attacker-controlled deeply nested
    /// expressions reaching type-checking after construction.
    fn dtype_depth(&self, schema: &Schema, depth: usize) -> BoltResult<DataType> {
        if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
            return Err(BoltError::Type(format!(
                "expression nesting exceeds depth limit ({})",
                crate::plan::sql_frontend::MAX_RECURSION_DEPTH
            )));
        }
        match self {
            Expr::Column(name) => Ok(schema.field(name)?.dtype),
            Expr::Literal(lit) => lit
                .dtype()
                .ok_or_else(|| BoltError::Type("untyped NULL literal".into())),
            Expr::Binary { op, left, right } => {
                // NULL-peer typing: an untyped `Literal::Null` opposite a
                // typed peer takes that peer's dtype for the purposes of
                // type-checking the binary expression. The peer-typed
                // helper calls back into `dtype` (which starts a fresh
                // depth budget); that's fine since the parent depth check
                // has already bounded the enclosing recursion.
                let l = peer_typed_dtype(left, right, schema, *op)?;
                let r = peer_typed_dtype(right, left, schema, *op)?;
                let _ = depth; // depth threading enforced at function entry
                if op.is_arithmetic() {
                    if !l.is_numeric() || !r.is_numeric() {
                        return Err(BoltError::Type(format!(
                            "arithmetic {op:?} requires numeric operands, got {l:?} and {r:?}"
                        )));
                    }
                    unify_numeric(l, r)
                } else if op.is_comparison() {
                    if l == r {
                        Ok(DataType::Bool)
                    } else if l.is_numeric() && r.is_numeric() {
                        // Allow numeric cross-comparisons; result is still Bool.
                        let _ = unify_numeric(l, r)?;
                        Ok(DataType::Bool)
                    } else {
                        Err(BoltError::Type(format!(
                            "cannot compare {l:?} with {r:?}"
                        )))
                    }
                } else if op.is_logical() {
                    if l == DataType::Bool && r == DataType::Bool {
                        Ok(DataType::Bool)
                    } else {
                        Err(BoltError::Type(format!(
                            "logical {op:?} requires Bool operands, got {l:?} and {r:?}"
                        )))
                    }
                } else {
                    Err(BoltError::Type(format!("unsupported operator {op:?}")))
                }
            }
            Expr::Unary { op, operand } => match op {
                // IS NULL / IS NOT NULL always produce Bool, regardless of
                // operand dtype. We still resolve the operand's dtype when
                // it's resolvable (catches typos like `nonexistent IS NULL`),
                // but tolerate an untyped `Literal::Null` operand — that's
                // exactly the case this surface exists to support.
                UnaryOp::IsNull | UnaryOp::IsNotNull => {
                    if !matches!(operand.as_ref(), Expr::Literal(Literal::Null)) {
                        let _ = operand.dtype_depth(schema, depth + 1)?;
                    }
                    Ok(DataType::Bool)
                }
                // NOT requires a Bool operand; the result is Bool. An untyped
                // `Literal::Null` is accepted under the same NULL-peer-typing
                // spirit as `Binary` ops — `NOT NULL` is a Bool-typed NULL
                // value, not a type error.
                UnaryOp::Not => {
                    if matches!(operand.as_ref(), Expr::Literal(Literal::Null)) {
                        return Ok(DataType::Bool);
                    }
                    let t = operand.dtype_depth(schema, depth + 1)?;
                    if t != DataType::Bool {
                        return Err(BoltError::Type(format!(
                            "logical NOT requires a Bool operand, got {t:?}"
                        )));
                    }
                    Ok(DataType::Bool)
                }
            },
            Expr::Alias(inner, _) => inner.dtype_depth(schema, depth + 1),
        }
    }
}

/// Resolve `e`'s dtype against `schema`, but if `e` is `Literal::Null` and
/// `peer` resolves to a typed expression, return the peer's dtype instead.
///
/// This is the NULL-peer-typing rule used by `Expr::Binary` dtype resolution
/// so that SQL fragments like `WHERE x = NULL` or `SELECT x + NULL` don't
/// hard-error at type-check time. The rule applies to every BinaryOp:
/// arithmetic, comparison, and logical (where the typed peer is necessarily
/// Bool, so NULL becomes Bool). Two NULLs on both sides still surface the
/// original `Type("untyped NULL literal")` error — there is no peer to
/// borrow a type from.
fn peer_typed_dtype(
    e: &Expr,
    peer: &Expr,
    schema: &Schema,
    _op: BinaryOp,
) -> BoltResult<DataType> {
    if matches!(e, Expr::Literal(Literal::Null)) {
        // Try to borrow the peer's dtype. If the peer itself is also a
        // bare untyped NULL the recursive call will fail with the original
        // "untyped NULL literal" error, which is what we want.
        if let Ok(t) = peer.dtype(schema) {
            return Ok(t);
        }
    }
    e.dtype(schema)
}

/// Promote two numeric types to the wider one (float beats int, 64 beats 32).
fn unify_numeric(a: DataType, b: DataType) -> BoltResult<DataType> {
    use DataType::*;
    if !a.is_numeric() || !b.is_numeric() {
        return Err(BoltError::Type(format!(
            "cannot unify non-numeric types {a:?} and {b:?}"
        )));
    }
    let either_float = a.is_float() || b.is_float();
    let either_64 = matches!(a, Int64 | Float64) || matches!(b, Int64 | Float64);
    Ok(match (either_float, either_64) {
        (true, true) => Float64,
        (true, false) => Float32,
        (false, true) => Int64,
        (false, false) => Int32,
    })
}

/// Aggregate function applied over an expression.
#[derive(Debug, Clone)]
pub enum AggregateExpr {
    /// `COUNT(expr)` — output `Int64`.
    Count(Expr),
    /// `SUM(expr)` — output preserves input dtype.
    Sum(Expr),
    /// `MIN(expr)` — output preserves input dtype.
    Min(Expr),
    /// `MAX(expr)` — output preserves input dtype.
    Max(Expr),
    /// `AVG(expr)` — output `Float64`.
    Avg(Expr),
}

impl AggregateExpr {
    /// Default output column name.
    ///
    /// Authoritative naming rule for aggregate output columns. Called from
    /// `LogicalPlan::schema()` (this file) and re-exported via the
    /// free function [`aggregate_output_name`] which is consumed by
    /// `sql_frontend.rs::plan_select` (SELECT-list re-projection) and
    /// `sql_frontend.rs::lower_expr_in_having` (HAVING rewriter). Do not
    /// duplicate the rule at the call sites; route through this method
    /// (or the free function) instead.
    pub(crate) fn output_name(&self) -> String {
        match self {
            AggregateExpr::Count(e) => format!("count{}", suffix(e)),
            AggregateExpr::Sum(e) => format!("sum{}", suffix(e)),
            AggregateExpr::Min(e) => format!("min{}", suffix(e)),
            AggregateExpr::Max(e) => format!("max{}", suffix(e)),
            AggregateExpr::Avg(e) => format!("avg{}", suffix(e)),
        }
    }

    /// Output dtype of the aggregate against the input schema.
    ///
    /// `SUM` widens narrow integer inputs to the corresponding 64-bit type
    /// to prevent silent overflow under typical workloads (`SUM(Int32)` over
    /// more than ~2^31 small values would otherwise wrap). Float inputs and
    /// `Int64`/`UInt64` inputs are not widened (no wider primitive type is
    /// available); callers must be aware of overflow risk on extreme inputs.
    ///
    /// This widening contract is mirrored by the GPU-side accumulator in
    /// `crate::jit::agg_kernels` and the host-side scalar-aggregate path in
    /// `crate::exec::aggregate`; keep all three in sync.
    fn output_dtype(&self, input: &Schema) -> BoltResult<DataType> {
        match self {
            AggregateExpr::Count(_) => Ok(DataType::Int64),
            AggregateExpr::Sum(e) => Ok(sum_output_dtype(e.dtype(input)?)),
            AggregateExpr::Min(e) | AggregateExpr::Max(e) => e.dtype(input),
            AggregateExpr::Avg(_) => Ok(DataType::Float64),
        }
    }
}

/// Widen the input dtype of a `SUM` aggregate to its accumulator dtype.
///
/// Mirrors the widening contract documented on `AggregateExpr::output_dtype`:
/// narrow signed integers (currently only `Int32` in the supported `DataType`
/// set) widen to `Int64`; `Int64` and the float types are unchanged. This
/// helper is the single source of truth for the SUM widening rule and is also
/// consumed by `crate::jit::agg_kernels` (kernel emission must agree with the
/// plan's declared output type) and `crate::exec::aggregate` (accumulator
/// allocation and Arrow array packing).
pub fn sum_output_dtype(input: DataType) -> DataType {
    match input {
        // Narrow signed integer → widen to Int64.
        DataType::Int32 => DataType::Int64,
        // Already 64-bit-wide or float: unchanged (no wider primitive in this
        // engine's `DataType`). Overflow risk on Int64 is acknowledged at the
        // API boundary.
        DataType::Int64 | DataType::Float32 | DataType::Float64 => input,
        // Non-numeric types fall through unchanged; the downstream typecheck
        // (e.g. `ReduceOp::identity_ptx`) will reject the aggregate.
        DataType::Bool | DataType::Utf8 => input,
    }
}

/// `_colname` for a bare column ref, empty otherwise.
fn suffix(e: &Expr) -> String {
    match e {
        Expr::Column(n) => format!("_{n}"),
        Expr::Alias(_, n) => format!("_{n}"),
        _ => String::new(),
    }
}

/// A single ORDER BY entry: an expression plus direction / null placement.
#[derive(Debug, Clone)]
pub struct SortExpr {
    /// The sort key expression.
    pub expr: Expr,
    /// True for DESC, false for ASC.
    pub descending: bool,
    /// True if NULLs sort before non-NULLs (NULLS FIRST), false if after.
    pub nulls_first: bool,
}

/// Join kind. INNER, LEFT, RIGHT, FULL, CROSS supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// SQL INNER JOIN.
    Inner,
    /// SQL LEFT [OUTER] JOIN: every left row appears, NULL-padded on the
    /// right when no match is found.
    LeftOuter,
    /// SQL RIGHT [OUTER] JOIN: every right row appears, NULL-padded on the
    /// left when no match is found.
    RightOuter,
    /// SQL FULL [OUTER] JOIN: union of LEFT + RIGHT semantics — every
    /// unmatched row from either side emits with the opposite side NULL.
    FullOuter,
    /// SQL CROSS JOIN: cartesian product, no ON predicate.
    Cross,
}

impl JoinType {
    /// True if the left side is preserved (every left row emits at least
    /// once). Holds for INNER (matched only), LEFT, FULL, and CROSS;
    /// false for RIGHT (left rows may be dropped if unmatched on the
    /// right).
    pub fn left_preserved(self) -> bool {
        matches!(
            self,
            JoinType::LeftOuter | JoinType::FullOuter | JoinType::Cross
        )
    }

    /// True if the right side is preserved (every right row emits at least
    /// once). Holds for RIGHT, FULL, and CROSS; false for INNER and LEFT.
    pub fn right_preserved(self) -> bool {
        matches!(
            self,
            JoinType::RightOuter | JoinType::FullOuter | JoinType::Cross
        )
    }
}

/// Relational logical plan node.
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Read a registered table.
    Scan {
        /// Table name.
        table: String,
        /// Optional projected column subset.
        projection: Option<Vec<String>>,
        /// Schema of the (un-projected) table.
        schema: Schema,
    },
    /// Apply a boolean predicate.
    Filter {
        /// Source.
        input: Box<LogicalPlan>,
        /// Boolean expression.
        predicate: Expr,
    },
    /// SELECT list; output schema follows `exprs` in order.
    Project {
        /// Source.
        input: Box<LogicalPlan>,
        /// Output expressions.
        exprs: Vec<Expr>,
    },
    /// GROUP BY + aggregates; empty `group_by` yields a single output row.
    Aggregate {
        /// Source.
        input: Box<LogicalPlan>,
        /// Grouping expressions.
        group_by: Vec<Expr>,
        /// Aggregate expressions.
        aggregates: Vec<AggregateExpr>,
    },
    /// SQL DISTINCT: deduplicate rows from `input`. Schema = input.schema().
    Distinct {
        /// Source.
        input: Box<LogicalPlan>,
    },
    /// SQL LIMIT [OFFSET]: keep at most `limit` rows after skipping `offset`.
    /// Schema = input.schema().
    Limit {
        /// Source.
        input: Box<LogicalPlan>,
        /// Maximum number of rows to emit.
        limit: usize,
        /// Number of leading rows to skip (0 if no OFFSET clause).
        offset: usize,
    },
    /// SQL ORDER BY: sort `input` by `sort_exprs`. Schema = input.schema().
    Sort {
        /// Source.
        input: Box<LogicalPlan>,
        /// Sort keys, evaluated in order (first is most significant).
        sort_exprs: Vec<SortExpr>,
    },
    /// SQL UNION ALL — concatenation without dedup. UNION (with dedup) is
    /// parsed and lowered to `Distinct(Union { ... })`. All inputs must share
    /// the same schema; the result schema is the first input's schema.
    Union {
        /// Branches to concatenate, in source order.
        inputs: Vec<LogicalPlan>,
    },
    /// SQL JOIN: combine `left` and `right` rows that satisfy `on`.
    /// Supports `JoinType::Inner`, `LeftOuter`, `RightOuter`, `FullOuter`,
    /// and `Cross`. INNER and the OUTER variants require at least one
    /// equi-join predicate (`on` non-empty); CROSS requires `on` to be
    /// empty (it has no ON clause).
    Join {
        /// Left input.
        left: Box<LogicalPlan>,
        /// Right input.
        right: Box<LogicalPlan>,
        /// Join kind.
        join_type: JoinType,
        /// Equi-join predicate pairs `(left_expr, right_expr)`;
        /// conjunctive. Empty for `Cross`.
        on: Vec<(Expr, Expr)>,
    },
}

impl LogicalPlan {
    /// Type-check the plan and return its output schema.
    pub fn schema(&self) -> BoltResult<Schema> {
        self.schema_depth(0)
    }

    /// Inner recursion for [`LogicalPlan::schema`]. `depth` is the current
    /// recursion depth; returns Err if
    /// [`crate::plan::sql_frontend::MAX_RECURSION_DEPTH`] is exceeded —
    /// guards against attacker-controlled deeply nested plans reaching
    /// type-checking after construction (which would otherwise overflow
    /// the host thread stack).
    fn schema_depth(&self, depth: usize) -> BoltResult<Schema> {
        if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
            return Err(BoltError::Plan(format!(
                "plan nesting exceeds depth limit ({})",
                crate::plan::sql_frontend::MAX_RECURSION_DEPTH
            )));
        }
        match self {
            LogicalPlan::Scan {
                projection, schema, ..
            } => match projection {
                None => Ok(schema.clone()),
                Some(cols) => {
                    let mut fields = Vec::with_capacity(cols.len());
                    for c in cols {
                        fields.push(schema.field(c)?.clone());
                    }
                    Ok(Schema::new(fields))
                }
            },
            LogicalPlan::Filter { input, predicate } => {
                let s = input.schema_depth(depth + 1)?;
                let pt = predicate.dtype(&s)?;
                if pt != DataType::Bool {
                    return Err(BoltError::Type(format!(
                        "filter predicate must be Bool, got {pt:?}"
                    )));
                }
                Ok(s)
            }
            LogicalPlan::Project { input, exprs } => {
                let s = input.schema_depth(depth + 1)?;
                let mut fields = Vec::with_capacity(exprs.len());
                for (i, e) in exprs.iter().enumerate() {
                    let dtype = e.dtype(&s)?;
                    let name = match e {
                        Expr::Column(n) => n.clone(),
                        Expr::Alias(_, n) => n.clone(),
                        _ => format!("__expr_{i}"),
                    };
                    fields.push(Field::new(name, dtype, true));
                }
                Ok(Schema::new(fields))
            }
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                let s = input.schema_depth(depth + 1)?;
                let mut fields = Vec::with_capacity(group_by.len() + aggregates.len());
                for (i, g) in group_by.iter().enumerate() {
                    let dtype = g.dtype(&s)?;
                    // Route through the authoritative helper so the rule
                    // (Column/Alias keep their name, anything else gets a
                    // positional `__group_{i}` placeholder) lives in one
                    // place. `sql_frontend.rs` calls the same helper to
                    // recover these names when re-projecting the
                    // Aggregate's output into SELECT-list order.
                    let name = group_key_output_name(g, i);
                    fields.push(Field::new(name, dtype, false));
                }
                for agg in aggregates {
                    let dtype = agg.output_dtype(&s)?;
                    fields.push(Field::new(agg.output_name(), dtype, true));
                }
                Ok(Schema::new(fields))
            }
            // Row-shape preserving wrappers: schema is the input's schema.
            LogicalPlan::Distinct { input }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Sort { input, .. } => {
                // For Sort we additionally type-check the sort keys against
                // the input schema so misnamed columns surface here rather
                // than at execution time.
                let s = input.schema_depth(depth + 1)?;
                if let LogicalPlan::Sort { sort_exprs, .. } = self {
                    for se in sort_exprs {
                        // We don't constrain the key dtype (any orderable
                        // scalar is fine); just resolve it so unknown columns
                        // produce a Plan error.
                        let _ = se.expr.dtype(&s)?;
                    }
                }
                Ok(s)
            }
            LogicalPlan::Union { inputs } => {
                if inputs.is_empty() {
                    return Err(BoltError::Plan(
                        "UNION requires at least one input".into(),
                    ));
                }
                let first = inputs[0].schema_depth(depth + 1)?;
                for (i, branch) in inputs.iter().enumerate().skip(1) {
                    let other = branch.schema_depth(depth + 1)?;
                    if !schemas_compatible(&first, &other) {
                        return Err(BoltError::Plan(format!(
                            "UNION branch {i} schema does not match branch 0: \
                             expected {} fields ({}), got {} fields ({})",
                            first.fields.len(),
                            schema_summary(&first),
                            other.fields.len(),
                            schema_summary(&other),
                        )));
                    }
                }
                Ok(first)
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                ..
            } => {
                // Concatenate left and right schemas, disambiguating right-
                // side columns whose names collide with anything on the left.
                // For OUTER joins, the columns coming from the side that
                // may be NULL-padded are marked `nullable = true` (a row
                // from the preserved side may have no match on the other).
                // See `join_combined_schema` for the canonical rule (also
                // used by `PhysicalPlan::Join::output_schema()`);
                // duplicating it here would risk drift if either copy is
                // edited.
                let l = left.schema_depth(depth + 1)?;
                let r = right.schema_depth(depth + 1)?;
                Ok(join_combined_schema(&l, &r, *join_type))
            }
        }
    }
}

/// Build the output schema of a JOIN over `left` and `right`.
///
/// Concatenates the two schemas in order, but disambiguates any right-side
/// field whose name already appears on the left by prefixing it with
/// `"right."`. Left-side fields keep their bare names so existing
/// downstream references continue to resolve unchanged. The rule:
///
/// * For each right-side field `f`:
///   * if `f.name` does not collide with any left-side name, keep it as-is;
///   * otherwise rename it to `"right.{f.name}"`. If `"right.{f.name}"`
///     itself collides (rare — only if the left side has a literal
///     `"right.<name>"` column), append `__2`, `__3`, ... until unique.
///
/// Nullability of output fields is widened for OUTER joins. For a
/// `LEFT [OUTER]` join, every right-side column becomes nullable
/// (preserved-left rows with no match emit NULL-padded right columns).
/// `RIGHT [OUTER]` is symmetric; `FULL [OUTER]` widens both sides;
/// `CROSS` and `INNER` leave nullability untouched.
///
/// This is the single source of truth for join output schemas, called by
/// both [`LogicalPlan::Join::schema`](LogicalPlan#method.schema)
/// and [`PhysicalPlan::Join::output_schema`](crate::plan::physical_plan::PhysicalPlan::output_schema)
/// so the logical and physical layers can never disagree on what a join
/// produces.
pub fn join_combined_schema(left: &Schema, right: &Schema, join_type: JoinType) -> Schema {
    // Outer joins NULL-pad the *non-preserved* side: LEFT preserves the
    // left side and may NULL-pad the right; RIGHT is symmetric; FULL may
    // NULL-pad either side. CROSS and INNER never NULL-pad here.
    let left_may_null = matches!(join_type, JoinType::RightOuter | JoinType::FullOuter);
    let right_may_null = matches!(join_type, JoinType::LeftOuter | JoinType::FullOuter);

    let mut fields: Vec<Field> = Vec::with_capacity(left.fields.len() + right.fields.len());
    for lf in &left.fields {
        fields.push(Field {
            name: lf.name.clone(),
            dtype: lf.dtype,
            nullable: lf.nullable || left_may_null,
        });
    }
    // Snapshot the names already taken by the left side so collision lookup
    // doesn't depend on later right-side insertions. `join_rename` mutates
    // this set so each rename also sees the names produced for prior
    // right-side columns.
    let mut taken: std::collections::HashSet<String> =
        left.fields.iter().map(|f| f.name.clone()).collect();
    for rf in &right.fields {
        let name = join_rename(&rf.name, &mut taken);
        fields.push(Field {
            name,
            dtype: rf.dtype,
            nullable: rf.nullable || right_may_null,
        });
    }
    Schema { fields }
}

/// True if `a` and `b` have the same shape (same number of fields, same dtype
/// per position). Field names need not match: SQL UNION ALL takes the names
/// from the leftmost branch.
fn schemas_compatible(a: &Schema, b: &Schema) -> bool {
    if a.fields.len() != b.fields.len() {
        return false;
    }
    a.fields
        .iter()
        .zip(b.fields.iter())
        .all(|(x, y)| x.dtype == y.dtype)
}

/// One-line summary of a schema for error messages (`name: Type, ...`).
fn schema_summary(s: &Schema) -> String {
    s.fields
        .iter()
        .map(|f| format!("{}: {:?}", f.name, f.dtype))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Authoritative naming rule for aggregate output columns.
///
/// Thin free-function wrapper around [`AggregateExpr::output_name`] for use
/// from outside this module (the method itself is `pub(crate)`, but a free
/// function keeps the call sites in `sql_frontend.rs` clear of method-syntax
/// borrows). Called from `sql_frontend.rs::plan_select` (SELECT-list
/// re-projection over an `Aggregate` plan) and
/// `sql_frontend.rs::lower_expr_in_having` (HAVING rewriter); do not
/// duplicate the rule at the call sites.
pub(crate) fn aggregate_output_name(agg: &AggregateExpr) -> String {
    agg.output_name()
}

/// Authoritative naming rule for GROUP BY output columns inside an
/// `Aggregate` plan's output schema: a bare `Column` or top-level `Alias`
/// keeps its name; anything else gets a positional `__group_{idx}`
/// placeholder.
///
/// Called from [`LogicalPlan::schema`] (this file, in the `Aggregate` arm)
/// and from `sql_frontend.rs::plan_select` (the SELECT-list re-projection,
/// which needs to recover these names to wire group keys through to the
/// user-visible projection). Do not duplicate the rule at either call site.
pub(crate) fn group_key_output_name(key: &Expr, idx: usize) -> String {
    match key {
        Expr::Column(n) => n.clone(),
        Expr::Alias(_, n) => n.clone(),
        _ => format!("__group_{idx}"),
    }
}

/// Authoritative naming rule for a single right-side JOIN column when
/// disambiguating against an accumulated set of already-taken names.
pub(crate) fn join_rename(name: &str, taken: &mut std::collections::HashSet<String>) -> String {
    let mut out_name = if taken.contains(name) {
        format!("right.{name}")
    } else {
        name.to_string()
    };
    if taken.contains(&out_name) {
        let base = out_name.clone();
        let mut i = 2usize;
        loop {
            let candidate = format!("{base}__{i}");
            if !taken.contains(&candidate) {
                out_name = candidate;
                break;
            }
            i += 1;
        }
    }
    taken.insert(out_name.clone());
    out_name
}

#[cfg(test)]
mod null_handling_tests {
    use super::*;

    /// Baseline contract: a bare `Literal::Null` still has no static type.
    /// The new NULL-peer-typing surface kicks in at the `Expr::Binary` /
    /// `Expr::Unary` layer, not at the literal layer itself.
    #[test]
    fn literal_null_dtype_is_none() {
        assert_eq!(Literal::Null.dtype(), None);
    }

    /// `WHERE x = NULL` with `x: Int32` must type-check (NULL borrows the
    /// peer's dtype) and resolve the binary expression to `Bool`. The
    /// runtime semantics of `= NULL` are a separate concern handled by the
    /// executor; the planner just needs not to hard-error.
    #[test]
    fn null_peer_typing_in_binary_eq() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, true)]);
        let expr = Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(Expr::Column("x".into())),
            right: Box::new(Expr::Literal(Literal::Null)),
        };
        let t = expr.dtype(&schema).expect("NULL peer-typing must succeed");
        assert_eq!(t, DataType::Bool);
        // Symmetric — NULL on the left side also works.
        let expr_rev = Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(Expr::Literal(Literal::Null)),
            right: Box::new(Expr::Column("x".into())),
        };
        let t_rev = expr_rev
            .dtype(&schema)
            .expect("NULL peer-typing must be symmetric");
        assert_eq!(t_rev, DataType::Bool);
    }

    /// Two NULLs on both sides still surface the legacy
    /// "untyped NULL literal" error — there is no peer to borrow a dtype from.
    #[test]
    fn binary_with_two_nulls_still_errors() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, true)]);
        let expr = Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(Expr::Literal(Literal::Null)),
            right: Box::new(Expr::Literal(Literal::Null)),
        };
        assert!(expr.dtype(&schema).is_err());
    }

    /// `x IS NULL` and `x IS NOT NULL` always type-check to Bool — even
    /// when the operand is itself an untyped `Literal::Null`.
    #[test]
    fn unary_is_null_typechecks_to_bool() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, true)]);
        for op in [UnaryOp::IsNull, UnaryOp::IsNotNull] {
            let on_col = Expr::Unary {
                op,
                operand: Box::new(Expr::Column("x".into())),
            };
            assert_eq!(on_col.dtype(&schema).unwrap(), DataType::Bool);
            let on_null = Expr::Unary {
                op,
                operand: Box::new(Expr::Literal(Literal::Null)),
            };
            assert_eq!(on_null.dtype(&schema).unwrap(), DataType::Bool);
        }
    }

    /// `NOT <bool>` type-checks to Bool when the operand is Bool, and
    /// errors when it's a non-Bool numeric / string / etc. The convenience
    /// `Expr::not()` constructor produces the same `Expr::Unary` shape.
    #[test]
    fn unary_not_typechecks_against_bool_operand() {
        let schema = Schema::new(vec![
            Field::new("b", DataType::Bool, true),
            Field::new("x", DataType::Int32, true),
        ]);
        // Bool column under NOT → Bool result.
        let on_bool = Expr::Column("b".into()).not();
        assert_eq!(on_bool.dtype(&schema).unwrap(), DataType::Bool);
        // Int column under NOT → Type error.
        let on_int = Expr::Column("x".into()).not();
        assert!(on_int.dtype(&schema).is_err());
        // NULL under NOT → Bool (NULL-peer-typing surface).
        let on_null = Expr::Literal(Literal::Null).not();
        assert_eq!(on_null.dtype(&schema).unwrap(), DataType::Bool);
    }

    /// Arithmetic peer-typing: `x + NULL` with `x: Int64` resolves to
    /// `Int64` (the arithmetic unification rule applied with NULL borrowing
    /// its peer's dtype).
    #[test]
    fn null_peer_typing_in_binary_add() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, true)]);
        let expr = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::Column("x".into())),
            right: Box::new(Expr::Literal(Literal::Null)),
        };
        assert_eq!(expr.dtype(&schema).unwrap(), DataType::Int64);
    }
}

#[cfg(test)]
mod naming_consistency_tests {
    //! Lock the authoritative naming rules in place. These tests guard the
    //! consolidation in this module against regressions: if anyone changes
    //! the rule here, downstream `sql_frontend.rs` must observe the same
    //! change because both sites route through these helpers.
    use super::*;

    #[test]
    fn aggregate_output_name_is_stable_for_representative_exprs() {
        // Bare Column: `_colname` suffix.
        let agg = AggregateExpr::Sum(Expr::Column("price".to_string()));
        assert_eq!(aggregate_output_name(&agg), "sum_price");
        assert_eq!(agg.output_name(), "sum_price");

        let agg = AggregateExpr::Avg(Expr::Column("qty".to_string()));
        assert_eq!(aggregate_output_name(&agg), "avg_qty");

        let agg = AggregateExpr::Min(Expr::Column("ts".to_string()));
        assert_eq!(aggregate_output_name(&agg), "min_ts");

        let agg = AggregateExpr::Max(Expr::Column("ts".to_string()));
        assert_eq!(aggregate_output_name(&agg), "max_ts");

        // Alias: take the alias name as the suffix.
        let aliased = Expr::Alias(Box::new(Expr::Column("c".to_string())), "renamed".to_string());
        let agg = AggregateExpr::Count(aliased);
        assert_eq!(aggregate_output_name(&agg), "count_renamed");

        // Non-column / non-alias inner expr: no suffix.
        let lit = Expr::Literal(Literal::Int64(1));
        let agg = AggregateExpr::Count(lit);
        assert_eq!(aggregate_output_name(&agg), "count");
    }

    #[test]
    fn group_key_output_name_is_stable_for_representative_exprs() {
        // Bare column keeps its name.
        assert_eq!(
            group_key_output_name(&Expr::Column("region".to_string()), 0),
            "region"
        );
        // Alias keeps its alias name regardless of index.
        let aliased = Expr::Alias(Box::new(Expr::Column("c".to_string())), "r".to_string());
        assert_eq!(group_key_output_name(&aliased, 3), "r");
        // Anything else falls back to a positional placeholder.
        let lit = Expr::Literal(Literal::Int64(7));
        assert_eq!(group_key_output_name(&lit, 0), "__group_0");
        assert_eq!(group_key_output_name(&lit, 2), "__group_2");
    }

    #[test]
    fn join_combined_schema_renames_colliding_right_side_to_right_dot_prefix() {
        // Both sides have a column named `a`; the right one should be
        // renamed to `right.a`. The left one stays as `a`.
        let left = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let right = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let out = join_combined_schema(&left, &right, JoinType::Inner);
        let names: Vec<&str> = out.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["a", "right.a"]);
    }

    #[test]
    fn join_combined_schema_passes_through_non_colliding_names() {
        // No collision — both sides keep their original names.
        let left = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let right = Schema::new(vec![Field::new("b", DataType::Int32, false)]);
        let out = join_combined_schema(&left, &right, JoinType::Inner);
        let names: Vec<&str> = out.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn join_combined_schema_falls_back_to_numeric_suffix_on_qualified_collision() {
        // Pathological case: the left side already has a column literally
        // named `right.a`, so the right-side `a` cannot be renamed to
        // `right.a` and must fall through to the `__2` suffix.
        let left = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("right.a", DataType::Int32, false),
        ]);
        let right = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let out = join_combined_schema(&left, &right, JoinType::Inner);
        let names: Vec<&str> = out.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["a", "right.a", "right.a__2"]);
    }

    #[test]
    fn join_rename_matches_join_combined_schema_for_simple_collision() {
        // The standalone helper must produce the same rename sequence the
        // full schema-building function produces; this is the contract
        // `sql_frontend.rs::NameResolver::push_join` relies on.
        let mut taken: std::collections::HashSet<String> = ["a".to_string()].into_iter().collect();
        let renamed = join_rename("a", &mut taken);
        assert_eq!(renamed, "right.a");
        assert!(taken.contains("right.a"));
        assert!(taken.contains("a"));
    }

    #[test]
    fn join_combined_schema_widens_nullability_for_outer_joins() {
        // LEFT OUTER: right-side columns become nullable.
        let left = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let right = Schema::new(vec![Field::new("b", DataType::Int32, false)]);
        let out = join_combined_schema(&left, &right, JoinType::LeftOuter);
        assert!(!out.fields[0].nullable, "left side stays non-null on LEFT");
        assert!(out.fields[1].nullable, "right side widens on LEFT");

        // RIGHT OUTER: left-side columns become nullable.
        let out = join_combined_schema(&left, &right, JoinType::RightOuter);
        assert!(out.fields[0].nullable, "left side widens on RIGHT");
        assert!(!out.fields[1].nullable, "right side stays non-null on RIGHT");

        // FULL OUTER: both sides become nullable.
        let out = join_combined_schema(&left, &right, JoinType::FullOuter);
        assert!(out.fields[0].nullable);
        assert!(out.fields[1].nullable);

        // INNER / CROSS: nullability untouched.
        let out = join_combined_schema(&left, &right, JoinType::Inner);
        assert!(!out.fields[0].nullable);
        assert!(!out.fields[1].nullable);
    }
}
