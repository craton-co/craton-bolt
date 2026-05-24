// SPDX-License-Identifier: Apache-2.0

//! Physical plan: column-ordinal-resolved, register-machine IR for GPU codegen.

use std::collections::HashMap;

use crate::error::{JavelinError, JavelinResult};
use crate::plan::logical_plan::{
    AggregateExpr, BinaryOp, DataType, Expr, Field, Literal, LogicalPlan, Schema,
};

/// SSA register handle. Just an index into the IR's value table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub u32);

/// A typed value in the IR: a register plus its known dtype.
#[derive(Debug, Clone, Copy)]
pub struct Value {
    /// The register holding the value.
    pub reg: Reg,
    /// The runtime dtype of the value.
    pub dtype: DataType,
}

/// A single instruction in the IR.
#[derive(Debug, Clone)]
pub enum Op {
    /// Load row `tid` of input column `col_idx` into a register.
    LoadColumn {
        /// Destination register.
        dst: Reg,
        /// Ordinal of the column in `KernelSpec::inputs`.
        col_idx: usize,
        /// Dtype of the loaded value.
        dtype: DataType,
    },
    /// A literal constant value.
    Const {
        /// Destination register.
        dst: Reg,
        /// The constant.
        lit: Literal,
    },
    /// Cast `src` from its current dtype to `dtype`.
    Cast {
        /// Destination register.
        dst: Reg,
        /// Source register.
        src: Reg,
        /// Original dtype of `src`.
        from: DataType,
        /// Target dtype.
        to: DataType,
    },
    /// Binary op. Both operands must already have the same dtype (insert Cast first).
    Binary {
        /// Destination register.
        dst: Reg,
        /// The operator.
        op: BinaryOp,
        /// Left operand register.
        lhs: Reg,
        /// Right operand register.
        rhs: Reg,
        /// Common dtype of `lhs` and `rhs`.
        dtype: DataType,
        /// Dtype of the produced value.
        result_dtype: DataType,
    },
    /// Store value `src` to output column `col_idx` at row `tid` (mask permitting).
    Store {
        /// Source register.
        src: Reg,
        /// Ordinal of the column in `KernelSpec::outputs`.
        col_idx: usize,
        /// Dtype of the stored value.
        dtype: DataType,
    },
}

/// Description of an input column the kernel consumes.
#[derive(Debug, Clone)]
pub struct ColumnIO {
    /// Column name.
    pub name: String,
    /// Column dtype.
    pub dtype: DataType,
}

/// A single GPU kernel description, derived from a fused (Scan -> [Filter ->] Project) chain.
#[derive(Debug, Clone)]
pub struct KernelSpec {
    /// Columns the kernel reads, in load order.
    pub inputs: Vec<ColumnIO>,
    /// Columns the kernel writes, in store order.
    pub outputs: Vec<ColumnIO>,
    /// Linear IR.
    pub ops: Vec<Op>,
    /// Optional predicate register; if Some, only rows where this is true emit output.
    pub predicate: Option<Reg>,
    /// Number of registers used by this kernel.
    pub register_count: u32,
}

/// Description of an aggregation kernel.
#[derive(Debug, Clone)]
pub struct AggregateSpec {
    /// Columns read from the pre-aggregation kernel (or directly from the scan).
    pub inputs: Vec<ColumnIO>,
    /// Group-by key column ordinals (into `inputs`).
    pub group_by: Vec<usize>,
    /// Aggregates to compute; codegen will lower further.
    pub aggregates: Vec<AggregateExpr>,
    /// Output schema (group-by columns first, then aggregate result columns).
    pub output_schema: Schema,
}

/// The top-level physical plan: a small ordered pipeline of kernels.
#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// Project (optionally with filter): single fused kernel over a table scan.
    Projection {
        /// Source table name.
        table: String,
        /// The fused projection kernel.
        kernel: KernelSpec,
        /// Output schema of the kernel.
        output_schema: Schema,
    },
    /// Aggregate over a (possibly filtered) projection.
    Aggregate {
        /// Source table name.
        table: String,
        /// Optional pre-aggregation kernel resolving group keys + aggregate-input exprs.
        pre: Option<KernelSpec>,
        /// The aggregation specification.
        aggregate: AggregateSpec,
    },
}

impl PhysicalPlan {
    /// Output schema of the whole plan.
    pub fn output_schema(&self) -> &Schema {
        match self {
            PhysicalPlan::Projection { output_schema, .. } => output_schema,
            PhysicalPlan::Aggregate { aggregate, .. } => &aggregate.output_schema,
        }
    }
}

/// Promote two numeric types to the wider one (float beats int, 64 beats 32).
fn unify_numeric(a: DataType, b: DataType) -> JavelinResult<DataType> {
    use DataType::*;
    match (a, b) {
        (x, y) if x == y => Ok(x),
        (Float64, _) | (_, Float64) => Ok(Float64),
        (Float32, Int64) | (Int64, Float32) => Ok(Float64),
        (Float32, _) | (_, Float32) => Ok(Float32),
        (Int64, _) | (_, Int64) => Ok(Int64),
        (Int32, _) | (_, Int32) => Ok(Int32),
        _ => Err(JavelinError::Type(format!(
            "cannot unify {:?} and {:?}",
            a, b
        ))),
    }
}

/// Returns the output column name for a projected expression at position `i`.
fn output_name_for(expr: &Expr, i: usize) -> String {
    match expr {
        Expr::Column(n) => n.clone(),
        Expr::Alias(_, n) => n.clone(),
        _ => format!("__expr_{i}"),
    }
}

/// True if every expression is a bare column reference (no alias, no compute).
fn all_bare_columns(exprs: &[Expr]) -> bool {
    exprs.iter().all(|e| matches!(e, Expr::Column(_)))
}

/// Emitter for a single kernel's IR.
struct Codegen<'a> {
    /// Schema of the underlying scan (column lookup).
    scan_schema: &'a Schema,
    /// Emitted ops.
    ops: Vec<Op>,
    /// Allocator for fresh registers.
    next_reg: u32,
    /// Input column metadata, in load order.
    inputs: Vec<ColumnIO>,
    /// Cache of already-loaded columns: name -> (input ordinal, Value).
    column_cache: HashMap<String, (usize, Value)>,
}

impl<'a> Codegen<'a> {
    /// New empty emitter against `scan_schema`.
    fn new(scan_schema: &'a Schema) -> Self {
        Self {
            scan_schema,
            ops: Vec::new(),
            next_reg: 0,
            inputs: Vec::new(),
            column_cache: HashMap::new(),
        }
    }

    /// Allocate a fresh register.
    fn fresh(&mut self) -> Reg {
        let r = Reg(self.next_reg);
        self.next_reg += 1;
        r
    }

    /// Emit ops for `e`, returning the produced value.
    fn emit_expr(&mut self, e: &Expr) -> JavelinResult<Value> {
        match e {
            Expr::Column(name) => self.emit_column(name),
            Expr::Literal(lit) => self.emit_literal(lit),
            Expr::Binary { op, left, right } => self.emit_binary(*op, left, right),
            Expr::Alias(inner, _) => self.emit_expr(inner),
        }
    }

    /// Emit (or reuse) a column load.
    fn emit_column(&mut self, name: &str) -> JavelinResult<Value> {
        if let Some((_, v)) = self.column_cache.get(name) {
            return Ok(*v);
        }
        let field = self.scan_schema.field(name)?;
        let dtype = field.dtype;
        let col_idx = self.inputs.len();
        self.inputs.push(ColumnIO {
            name: name.to_string(),
            dtype,
        });
        let dst = self.fresh();
        self.ops.push(Op::LoadColumn { dst, col_idx, dtype });
        let value = Value { reg: dst, dtype };
        self.column_cache
            .insert(name.to_string(), (col_idx, value));
        Ok(value)
    }

    /// Emit a constant literal load.
    fn emit_literal(&mut self, lit: &Literal) -> JavelinResult<Value> {
        let dtype = lit
            .dtype()
            .ok_or_else(|| JavelinError::Type("untyped NULL literal".into()))?;
        let dst = self.fresh();
        self.ops.push(Op::Const {
            dst,
            lit: lit.clone(),
        });
        Ok(Value { reg: dst, dtype })
    }

    /// Insert a Cast from `value` to `to`, returning the cast value.
    fn emit_cast(&mut self, value: Value, to: DataType) -> Value {
        if value.dtype == to {
            return value;
        }
        let dst = self.fresh();
        self.ops.push(Op::Cast {
            dst,
            src: value.reg,
            from: value.dtype,
            to,
        });
        Value { reg: dst, dtype: to }
    }

    /// Emit a binary op, inserting casts and computing the result dtype.
    fn emit_binary(&mut self, op: BinaryOp, left: &Expr, right: &Expr) -> JavelinResult<Value> {
        let l = self.emit_expr(left)?;
        let r = self.emit_expr(right)?;

        let (lhs_v, rhs_v, operand_dtype, result_dtype) = if op_is_arithmetic(op) {
            let unified = unify_numeric(l.dtype, r.dtype)?;
            let lv = self.emit_cast(l, unified);
            let rv = self.emit_cast(r, unified);
            (lv, rv, unified, unified)
        } else if op_is_comparison(op) {
            if l.dtype == r.dtype {
                (l, r, l.dtype, DataType::Bool)
            } else {
                // Match logical_plan's behavior: numeric cross-compare unifies, result Bool.
                let unified = unify_numeric(l.dtype, r.dtype)?;
                let lv = self.emit_cast(l, unified);
                let rv = self.emit_cast(r, unified);
                (lv, rv, unified, DataType::Bool)
            }
        } else if op_is_logical(op) {
            if l.dtype != DataType::Bool || r.dtype != DataType::Bool {
                return Err(JavelinError::Type(format!(
                    "logical {op:?} requires Bool operands, got {:?} and {:?}",
                    l.dtype, r.dtype
                )));
            }
            (l, r, DataType::Bool, DataType::Bool)
        } else {
            return Err(JavelinError::Type(format!("unsupported operator {op:?}")));
        };

        let dst = self.fresh();
        self.ops.push(Op::Binary {
            dst,
            op,
            lhs: lhs_v.reg,
            rhs: rhs_v.reg,
            dtype: operand_dtype,
            result_dtype,
        });
        Ok(Value {
            reg: dst,
            dtype: result_dtype,
        })
    }

    /// Append a Store op for column `col_idx`.
    fn emit_store(&mut self, value: Value, col_idx: usize) {
        self.ops.push(Op::Store {
            src: value.reg,
            col_idx,
            dtype: value.dtype,
        });
    }

    /// Finalize into a `KernelSpec`.
    fn finish(self, outputs: Vec<ColumnIO>, predicate: Option<Reg>) -> KernelSpec {
        KernelSpec {
            inputs: self.inputs,
            outputs,
            ops: self.ops,
            predicate,
            register_count: self.next_reg,
        }
    }
}

/// True for `+ - * /`.
fn op_is_arithmetic(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
    )
}

/// True for `= <> < <= > >=`.
fn op_is_comparison(op: BinaryOp) -> bool {
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
fn op_is_logical(op: BinaryOp) -> bool {
    matches!(op, BinaryOp::And | BinaryOp::Or)
}

/// Resolve a (Scan | Filter{Scan}) source into (table, scan_schema, optional filter).
fn resolve_source<'a>(
    plan: &'a LogicalPlan,
) -> JavelinResult<(&'a str, &'a Schema, Option<&'a Expr>)> {
    match plan {
        LogicalPlan::Scan { table, schema, .. } => Ok((table.as_str(), schema, None)),
        LogicalPlan::Filter { input, predicate } => match input.as_ref() {
            LogicalPlan::Scan { table, schema, .. } => {
                Ok((table.as_str(), schema, Some(predicate)))
            }
            other => Err(JavelinError::Plan(format!(
                "unsupported plan shape: Filter over {:?}",
                shape(other)
            ))),
        },
        other => Err(JavelinError::Plan(format!(
            "unsupported plan shape: expected Scan or Filter(Scan), got {}",
            shape(other)
        ))),
    }
}

/// Short tag describing a plan node's variant for error messages.
fn shape(plan: &LogicalPlan) -> &'static str {
    match plan {
        LogicalPlan::Scan { .. } => "Scan",
        LogicalPlan::Filter { .. } => "Filter",
        LogicalPlan::Project { .. } => "Project",
        LogicalPlan::Aggregate { .. } => "Aggregate",
    }
}

/// Build a projection kernel over `input`, producing `exprs` (or all input columns).
fn lower_projection(
    input: &LogicalPlan,
    exprs: Option<&[Expr]>,
    extra_predicate: Option<&Expr>,
) -> JavelinResult<PhysicalPlan> {
    let (table, scan_schema, scan_predicate) = resolve_source(input)?;
    let predicate = extra_predicate.or(scan_predicate);

    let mut cg = Codegen::new(scan_schema);

    // Emit the predicate first if any, so its register is stable.
    let predicate_reg = if let Some(pred) = predicate {
        let v = cg.emit_expr(pred)?;
        if v.dtype != DataType::Bool {
            return Err(JavelinError::Type(format!(
                "filter predicate must be Bool, got {:?}",
                v.dtype
            )));
        }
        Some(v.reg)
    } else {
        None
    };

    // Build the list of output expressions: either the explicit list or all scan columns.
    let owned_default: Vec<Expr>;
    let projected: &[Expr] = match exprs {
        Some(es) => es,
        None => {
            owned_default = scan_schema
                .fields
                .iter()
                .map(|f| Expr::Column(f.name.clone()))
                .collect();
            &owned_default
        }
    };

    let mut outputs = Vec::with_capacity(projected.len());
    let mut output_fields = Vec::with_capacity(projected.len());
    for (i, expr) in projected.iter().enumerate() {
        let value = cg.emit_expr(expr)?;
        let name = output_name_for(expr, i);
        cg.emit_store(value, i);
        outputs.push(ColumnIO {
            name: name.clone(),
            dtype: value.dtype,
        });
        output_fields.push(Field::new(name, value.dtype, true));
    }

    let kernel = cg.finish(outputs, predicate_reg);
    Ok(PhysicalPlan::Projection {
        table: table.to_string(),
        kernel,
        output_schema: Schema::new(output_fields),
    })
}

/// Build an aggregate plan over `input` with the given group keys and aggregates.
fn lower_aggregate(
    plan: &LogicalPlan,
    input: &LogicalPlan,
    group_by: &[Expr],
    aggregates: &[AggregateExpr],
) -> JavelinResult<PhysicalPlan> {
    let (table, scan_schema, scan_predicate) = resolve_source(input)?;
    let output_schema = plan.schema()?;

    // Collect the expressions that feed the aggregation: group keys first, then per-aggregate input.
    let mut agg_input_exprs: Vec<Expr> = Vec::with_capacity(aggregates.len());
    for agg in aggregates {
        let e = match agg {
            AggregateExpr::Count(e)
            | AggregateExpr::Sum(e)
            | AggregateExpr::Min(e)
            | AggregateExpr::Max(e)
            | AggregateExpr::Avg(e) => e.clone(),
        };
        agg_input_exprs.push(e);
    }

    // The "feed" exprs are group_by then aggregate inputs, in that order.
    let mut feed: Vec<Expr> = Vec::with_capacity(group_by.len() + agg_input_exprs.len());
    feed.extend(group_by.iter().cloned());
    feed.extend(agg_input_exprs.iter().cloned());

    // If there is no filter and every feed expression is a bare column ref, we can skip the
    // pre-aggregation kernel entirely; the aggregator can read those columns straight from the scan.
    let trivial = scan_predicate.is_none() && all_bare_columns(&feed);

    let (pre, agg_inputs, group_indices) = if trivial {
        // Build inputs / ordinals directly from the scan columns referenced by `feed`.
        let mut inputs: Vec<ColumnIO> = Vec::new();
        let mut name_to_ord: HashMap<String, usize> = HashMap::new();
        let mut ordinals: Vec<usize> = Vec::with_capacity(feed.len());
        for e in &feed {
            let name = match e {
                Expr::Column(n) => n.clone(),
                // `all_bare_columns` guarantees this branch is unreachable.
                _ => {
                    return Err(JavelinError::Plan(
                        "internal: trivial aggregate feed contained non-column expression".into(),
                    ))
                }
            };
            let ord = if let Some(o) = name_to_ord.get(&name) {
                *o
            } else {
                let field = scan_schema.field(&name)?;
                let o = inputs.len();
                inputs.push(ColumnIO {
                    name: name.clone(),
                    dtype: field.dtype,
                });
                name_to_ord.insert(name, o);
                o
            };
            ordinals.push(ord);
        }
        let group_ords: Vec<usize> = ordinals[..group_by.len()].to_vec();
        (None, inputs, group_ords)
    } else {
        // Emit a pre-aggregation kernel whose outputs are `feed` (group keys then aggregate inputs).
        let pre_plan = lower_projection(input, Some(&feed), None)?;
        let (pre_kernel, _pre_schema) = match pre_plan {
            PhysicalPlan::Projection {
                kernel,
                output_schema,
                ..
            } => (kernel, output_schema),
            // `lower_projection` always returns `Projection`.
            _ => {
                return Err(JavelinError::Plan(
                    "internal: lower_projection returned non-Projection".into(),
                ))
            }
        };
        let inputs: Vec<ColumnIO> = pre_kernel.outputs.clone();
        let group_ords: Vec<usize> = (0..group_by.len()).collect();
        (Some(pre_kernel), inputs, group_ords)
    };

    let aggregate = AggregateSpec {
        inputs: agg_inputs,
        group_by: group_indices,
        aggregates: aggregates.to_vec(),
        output_schema,
    };

    Ok(PhysicalPlan::Aggregate {
        table: table.to_string(),
        pre,
        aggregate,
    })
}

/// Lower a `LogicalPlan` to a `PhysicalPlan`.
pub fn lower(plan: &LogicalPlan) -> JavelinResult<PhysicalPlan> {
    match plan {
        LogicalPlan::Project { input, exprs } => lower_projection(input, Some(exprs), None),
        LogicalPlan::Filter { input, predicate } => {
            lower_projection(input, None, Some(predicate))
        }
        LogicalPlan::Scan { .. } => lower_projection(plan, None, None),
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => lower_aggregate(plan, input, group_by, aggregates),
    }
}
