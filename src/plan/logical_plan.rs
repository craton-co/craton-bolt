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
                let l = left.dtype_depth(schema, depth + 1)?;
                let r = right.dtype_depth(schema, depth + 1)?;
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
            Expr::Alias(inner, _) => inner.dtype_depth(schema, depth + 1),
        }
    }
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
    fn output_name(&self) -> String {
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
                    let name = match g {
                        Expr::Column(n) => n.clone(),
                        Expr::Alias(_, n) => n.clone(),
                        _ => format!("__group_{i}"),
                    };
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
    // doesn't depend on later right-side insertions.
    let mut taken: std::collections::HashSet<String> =
        left.fields.iter().map(|f| f.name.clone()).collect();
    for rf in &right.fields {
        let mut name = if taken.contains(&rf.name) {
            format!("right.{}", rf.name)
        } else {
            rf.name.clone()
        };
        // Final-resort uniqueness suffix; only triggers if the qualified
        // name itself collides with an existing left-side column.
        if taken.contains(&name) {
            let base = name.clone();
            let mut i = 2usize;
            loop {
                let candidate = format!("{base}__{i}");
                if !taken.contains(&candidate) {
                    name = candidate;
                    break;
                }
                i += 1;
            }
        }
        taken.insert(name.clone());
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
