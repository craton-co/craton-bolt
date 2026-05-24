// SPDX-License-Identifier: Apache-2.0

//! Logical plan AST: schemas, expressions, and relational nodes.

use crate::error::{JavelinError, JavelinResult};

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
    pub fn index_of(&self, name: &str) -> JavelinResult<usize> {
        self.fields
            .iter()
            .position(|f| f.name == name)
            .ok_or_else(|| JavelinError::Plan(format!("column '{name}' not found in schema")))
    }

    /// Lookup a field by name, or a `Plan` error if absent.
    pub fn field(&self, name: &str) -> JavelinResult<&Field> {
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
    pub fn dtype(&self, schema: &Schema) -> JavelinResult<DataType> {
        match self {
            Expr::Column(name) => Ok(schema.field(name)?.dtype),
            Expr::Literal(lit) => lit
                .dtype()
                .ok_or_else(|| JavelinError::Type("untyped NULL literal".into())),
            Expr::Binary { op, left, right } => {
                let l = left.dtype(schema)?;
                let r = right.dtype(schema)?;
                if op.is_arithmetic() {
                    if !l.is_numeric() || !r.is_numeric() {
                        return Err(JavelinError::Type(format!(
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
                        Err(JavelinError::Type(format!(
                            "cannot compare {l:?} with {r:?}"
                        )))
                    }
                } else if op.is_logical() {
                    if l == DataType::Bool && r == DataType::Bool {
                        Ok(DataType::Bool)
                    } else {
                        Err(JavelinError::Type(format!(
                            "logical {op:?} requires Bool operands, got {l:?} and {r:?}"
                        )))
                    }
                } else {
                    Err(JavelinError::Type(format!("unsupported operator {op:?}")))
                }
            }
            Expr::Alias(inner, _) => inner.dtype(schema),
        }
    }
}

/// Promote two numeric types to the wider one (float beats int, 64 beats 32).
fn unify_numeric(a: DataType, b: DataType) -> JavelinResult<DataType> {
    use DataType::*;
    if !a.is_numeric() || !b.is_numeric() {
        return Err(JavelinError::Type(format!(
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
    fn output_dtype(&self, input: &Schema) -> JavelinResult<DataType> {
        match self {
            AggregateExpr::Count(_) => Ok(DataType::Int64),
            AggregateExpr::Sum(e) | AggregateExpr::Min(e) | AggregateExpr::Max(e) => e.dtype(input),
            AggregateExpr::Avg(_) => Ok(DataType::Float64),
        }
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
}

impl LogicalPlan {
    /// Type-check the plan and return its output schema.
    pub fn schema(&self) -> JavelinResult<Schema> {
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
                let s = input.schema()?;
                let pt = predicate.dtype(&s)?;
                if pt != DataType::Bool {
                    return Err(JavelinError::Type(format!(
                        "filter predicate must be Bool, got {pt:?}"
                    )));
                }
                Ok(s)
            }
            LogicalPlan::Project { input, exprs } => {
                let s = input.schema()?;
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
                let s = input.schema()?;
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
        }
    }
}
