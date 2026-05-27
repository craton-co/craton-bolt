// SPDX-License-Identifier: Apache-2.0

//! Physical plan: column-ordinal-resolved, register-machine IR for GPU codegen.

use std::collections::HashMap;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{
    join_combined_schema, AggregateExpr, BinaryOp, DataType, Expr, Field, JoinType, Literal,
    LogicalPlan, Schema, SortExpr,
};

/// SSA register handle. Just an index into the IR's value table.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Reg(pub(crate) u32);

impl Reg {
    /// Read-only accessor for the underlying register index. Useful for
    /// external rustdoc consumers / debuggers; the field itself is
    /// `pub(crate)` so the wire representation isn't part of the public
    /// SemVer contract.
    pub fn id(self) -> u32 {
        self.0
    }
}

/// A typed value in the IR: a register plus its known dtype.
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct Value {
    /// The register holding the value.
    pub reg: Reg,
    /// The runtime dtype of the value.
    pub dtype: DataType,
}

/// A single instruction in the IR.
#[doc(hidden)]
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
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct ColumnIO {
    /// Column name.
    pub name: String,
    /// Column dtype.
    pub dtype: DataType,
}

/// A single GPU kernel description, derived from a fused (Scan -> [Filter ->] Project) chain.
#[doc(hidden)]
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
    /// Pre-stage NULL handling (Option B): one entry per input column. `true`
    /// means the caller will pass a parallel `*u8` validity pointer (1=valid,
    /// 0=null) AFTER the value+output pointer list and the codegen should
    /// load the validity byte at `tid` and AND it into the combined-validity
    /// register that drives every output's validity store.
    ///
    /// Default is `Vec::new()` which is treated as "no input carries
    /// validity" and the existing PTX layout is emitted verbatim — every
    /// existing caller (e.g. the projection path in `engine.rs`) continues
    /// to work bit-for-bit. When non-empty, must be parallel to `inputs`.
    #[doc(hidden)]
    pub input_has_validity: Vec<bool>,
    /// Pre-stage NULL handling (Option B): one entry per output column.
    /// `true` means the caller will pass a parallel `*u8` validity pointer
    /// where the kernel writes the per-row combined-validity result. The
    /// validity stores are appended after the regular value stores.
    ///
    /// Default `Vec::new()` => no output carries validity (no validity
    /// pointers added, no validity stores emitted). When non-empty, must
    /// be parallel to `outputs`.
    #[doc(hidden)]
    pub output_has_validity: Vec<bool>,
}

/// Description of an aggregation kernel.
#[doc(hidden)]
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
#[doc(hidden)]
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
    /// DISTINCT over `input`'s rows. Output schema = `input.output_schema()`.
    Distinct {
        /// Source plan.
        input: Box<PhysicalPlan>,
    },
    /// LIMIT [OFFSET]. Output schema = `input.output_schema()`.
    Limit {
        /// Source plan.
        input: Box<PhysicalPlan>,
        /// Maximum number of rows to emit.
        limit: usize,
        /// Number of leading rows to skip.
        offset: usize,
    },
    /// ORDER BY over `input`. Output schema = `input.output_schema()`.
    Sort {
        /// Source plan.
        input: Box<PhysicalPlan>,
        /// Sort keys, most-significant first.
        sort_exprs: Vec<SortExpr>,
    },
    /// UNION ALL: concatenate `inputs` in order. Schema is the first input's.
    /// (Dedup UNION is `Distinct(Union { ... })` in the logical plan.)
    Union {
        /// Branches to concatenate, in source order.
        inputs: Vec<PhysicalPlan>,
    },
    /// Pure column-rename / reorder layer over `input`. Used when the SQL
    /// frontend places a `Project` on top of an `Aggregate` (or other
    /// non-scan-chain operator) purely to surface SELECT-list order and
    /// aliases. Each `exprs` entry must be `Column(name)` or
    /// `Alias(Column(name), out_name)`; the executor just rearranges and
    /// renames the input batch's columns to match `output_schema`. No
    /// compute happens here — anything more elaborate (e.g. post-aggregate
    /// arithmetic) is rejected upstream.
    Project {
        /// Source plan.
        input: Box<PhysicalPlan>,
        /// One entry per output column; each references a column of `input`.
        exprs: Vec<Expr>,
        /// Output schema, in `exprs` order with aliases applied.
        output_schema: Schema,
    },
    /// INNER JOIN. The `output_schema` is `left.output_schema() ++ right`
    /// with right-side collisions disambiguated by `join_combined_schema`;
    /// it's stored on the variant so `output_schema()` can return a
    /// borrow-stable `&Schema` without allocating per call.
    Join {
        /// Left input.
        left: Box<PhysicalPlan>,
        /// Right input.
        right: Box<PhysicalPlan>,
        /// Join kind (INNER only in this version).
        join_type: JoinType,
        /// Equi-join predicate pairs `(left_expr, right_expr)`.
        on: Vec<(Expr, Expr)>,
        /// Combined left ++ right schema with right-side collisions
        /// renamed; see [`join_combined_schema`].
        output_schema: Schema,
    },
}

impl PhysicalPlan {
    /// Output schema of the whole plan.
    ///
    /// For the row-shape-preserving wrappers (`Distinct`, `Limit`, `Sort`),
    /// the schema is recursively the input's. `Union` returns the first
    /// branch's schema (UNION ALL semantics; branch compatibility was
    /// verified at logical-plan time). `Join` returns its stored
    /// `output_schema` field — the concatenated left ++ right schema with
    /// right-side name collisions disambiguated; the same shape
    /// `LogicalPlan::Join::schema()` produces, computed once at lowering
    /// time so this accessor can keep the `&Schema` borrow-stable.
    pub fn output_schema(&self) -> &Schema {
        match self {
            PhysicalPlan::Projection { output_schema, .. } => output_schema,
            PhysicalPlan::Aggregate { aggregate, .. } => &aggregate.output_schema,
            PhysicalPlan::Distinct { input }
            | PhysicalPlan::Limit { input, .. }
            | PhysicalPlan::Sort { input, .. } => input.output_schema(),
            PhysicalPlan::Union { inputs } => {
                // Caller will have ensured at logical-plan time that all
                // branches share a schema; here we just return the first.
                // The Union { inputs: vec![] } case can only arise from a
                // mis-constructed plan (parser rejects it) — fall back to
                // the first plan's schema if present, else panic-free path
                // by returning a degenerate empty schema would require an
                // allocation we can't make through a `&Schema` API. The
                // executor (engine.rs) errors on empty Union before this
                // accessor is ever called, so we let `inputs[0]` panic in
                // the degenerate case (consistent with `Vec::first` not
                // having a graceful sentinel).
                inputs
                    .first()
                    .expect("Union with zero inputs is invalid; rejected upstream")
                    .output_schema()
            }
            PhysicalPlan::Project { output_schema, .. } => output_schema,
            PhysicalPlan::Join { output_schema, .. } => output_schema,
        }
    }
}

/// Promote two numeric types to the wider one (float beats int, 64 beats 32).
fn unify_numeric(a: DataType, b: DataType) -> BoltResult<DataType> {
    use DataType::*;
    match (a, b) {
        (x, y) if x == y => Ok(x),
        (Float64, _) | (_, Float64) => Ok(Float64),
        (Float32, Int64) | (Int64, Float32) => Ok(Float64),
        (Float32, _) | (_, Float32) => Ok(Float32),
        (Int64, _) | (_, Int64) => Ok(Int64),
        (Int32, _) | (_, Int32) => Ok(Int32),
        _ => Err(BoltError::Type(format!(
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
    fn emit_expr(&mut self, e: &Expr) -> BoltResult<Value> {
        match e {
            Expr::Column(name) => self.emit_column(name),
            Expr::Literal(lit) => self.emit_literal(lit),
            Expr::Binary { op, left, right } => self.emit_binary(*op, left, right),
            Expr::Alias(inner, _) => self.emit_expr(inner),
        }
    }

    /// Emit (or reuse) a column load.
    fn emit_column(&mut self, name: &str) -> BoltResult<Value> {
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
    fn emit_literal(&mut self, lit: &Literal) -> BoltResult<Value> {
        let dtype = lit
            .dtype()
            .ok_or_else(|| BoltError::Type("untyped NULL literal".into()))?;
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
    fn emit_binary(&mut self, op: BinaryOp, left: &Expr, right: &Expr) -> BoltResult<Value> {
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
                return Err(BoltError::Type(format!(
                    "logical {op:?} requires Bool operands, got {:?} and {:?}",
                    l.dtype, r.dtype
                )));
            }
            (l, r, DataType::Bool, DataType::Bool)
        } else {
            return Err(BoltError::Type(format!("unsupported operator {op:?}")));
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
            // Validity propagation (Option B) is opt-in: callers populate
            // these via `KernelSpec::with_input_validity` etc. The default
            // codegen path emits the historical PTX shape unchanged.
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
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

/// Result of folding a chain of Scan/Filter/Project nodes down to a single
/// scan + predicate + (optional) explicit projection list.
struct ResolvedSource<'a> {
    /// Source table name.
    table: &'a str,
    /// Underlying scan schema (column lookup namespace for all exprs below).
    scan_schema: &'a Schema,
    /// Combined predicate, AND-folded, expressed in scan namespace.
    predicate: Option<Expr>,
    /// If the outermost layer was a `Project`, its output list as
    /// `(output_name, scan-namespace expr)`. `None` means no top projection
    /// was present, so callers should default to "all scan columns".
    projection: Option<Vec<(String, Expr)>>,
}

/// Approach (decision note): the `Codegen` emitter is hard-wired to a single
/// underlying scan schema (every `LoadColumn` references a scan column
/// ordinal). Composing it across multiple Project layers would require a
/// larger refactor. Instead we fold `Project`/`Filter` chains by *expression
/// substitution*: walk the chain top-down, then iteratively rewrite every
/// outer `Column(name)` reference into the inner `Project`'s expression for
/// that column. The substituted expression tree then resolves entirely
/// against the underlying scan schema, which is what `Codegen` already
/// supports. This collapses any `Project(Filter(Project(... Scan)))` shape
/// into a single equivalent (scan, AND-folded predicate, projection list).
/// Because the lowerer handles arbitrary chains now, no defensive guard in
/// `DataFrame::select`/`filter` is needed.
fn resolve_source<'a>(
    plan: &'a LogicalPlan,
) -> BoltResult<ResolvedSource<'a>> {
    // Walk down from the outermost node toward the underlying `Scan`,
    // collecting each layer. We delay all substitution to the end so each
    // node's expressions stay in their *own* input namespace until we know
    // the full chain. At resolution time, every layer's input namespace is
    // the output namespace of the next-deeper layer (or the scan), so we
    // can iteratively rewrite from innermost upward.
    enum Layer {
        // A `Filter` whose predicate lives in its input layer's namespace.
        Filter(Expr),
        // A `Project` whose `exprs` live in its input layer's namespace,
        // producing the named outputs (in the given order).
        Project(Vec<(String, Expr)>),
    }

    let mut cur = plan;
    let mut layers: Vec<Layer> = Vec::new();
    // Position of the *outermost* Project encountered (== index in `layers`).
    // The outermost Project defines the chain's effective output schema; any
    // Filters above it preserve the schema and just AND into the predicate.
    let mut outermost_project_idx: Option<usize> = None;

    let (table, scan_schema) = loop {
        match cur {
            LogicalPlan::Scan { table, schema, .. } => {
                break (table.as_str(), schema);
            }
            LogicalPlan::Filter { input, predicate } => {
                layers.push(Layer::Filter(predicate.clone()));
                cur = input.as_ref();
            }
            LogicalPlan::Project { input, exprs } => {
                let mut named: Vec<(String, Expr)> = Vec::with_capacity(exprs.len());
                for (i, e) in exprs.iter().enumerate() {
                    let name = output_name_for(e, i);
                    // Strip the outer Alias — it only affects output naming,
                    // not substitution semantics. Inner Aliases are left alone.
                    let body = match e {
                        Expr::Alias(inner, _) => (**inner).clone(),
                        _ => e.clone(),
                    };
                    named.push((name, body));
                }
                if outermost_project_idx.is_none() {
                    outermost_project_idx = Some(layers.len());
                }
                layers.push(Layer::Project(named));
                cur = input.as_ref();
            }
            other => {
                return Err(BoltError::Plan(format!(
                    "unsupported plan shape: expected Scan/Filter/Project chain, got {}",
                    shape(other)
                )));
            }
        }
    };

    // `layers` is ordered outermost-first; the innermost (closest to the
    // scan) is at the end. Walk innermost-to-outermost, maintaining the
    // current "name -> expression-in-scan-namespace" map (which represents
    // the output of the layer we just processed) and the accumulated
    // predicates (already in scan namespace).
    //
    // Initial state: just-above-scan output namespace == scan schema, so
    // every column resolves to itself (no entry needed — `substitute_one`
    // leaves unknown columns alone, which means "look it up in the scan").
    let mut name_map: HashMap<String, Expr> = HashMap::new();
    let mut name_map_active = false;
    let mut predicates_scan_ns: Vec<Expr> = Vec::new();
    // The outermost Project's output list, captured (in original order, with
    // exprs lowered to scan namespace) when we process that layer.
    let mut top_projection: Option<Vec<(String, Expr)>> = None;

    // Indices iterate from innermost (largest) to outermost (smallest).
    for (rev_i, layer) in layers.into_iter().enumerate().rev() {
        match layer {
            Layer::Filter(pred) => {
                // The predicate is in the current layer's input namespace
                // (== output of whatever's directly below). Rewrite into
                // scan namespace using `name_map` if a Project sits below us.
                let lowered = if name_map_active {
                    substitute_one(&pred, &name_map)
                } else {
                    pred
                };
                predicates_scan_ns.push(lowered);
            }
            Layer::Project(named) => {
                // Each Project replaces the output namespace. Its `exprs`
                // are in the *current* (below-it) namespace, so rewrite
                // each through `name_map` first, then install the new map.
                let mut next: HashMap<String, Expr> = HashMap::new();
                let mut named_lowered: Vec<(String, Expr)> = Vec::with_capacity(named.len());
                for (name, body) in named {
                    let lowered = if name_map_active {
                        substitute_one(&body, &name_map)
                    } else {
                        body
                    };
                    next.insert(name.clone(), lowered.clone());
                    named_lowered.push((name, lowered));
                }
                name_map = next;
                name_map_active = true;
                // If this is the outermost Project, capture its output list.
                if Some(rev_i) == outermost_project_idx {
                    top_projection = Some(named_lowered);
                }
            }
        }
    }

    Ok(ResolvedSource {
        table,
        scan_schema,
        predicate: and_all(predicates_scan_ns),
        projection: top_projection,
    })
}

/// Substitute `Column(name)` references in `expr` using `map`; leave unknown
/// columns alone (they pass through to a deeper layer or to the scan).
fn substitute_one(expr: &Expr, map: &HashMap<String, Expr>) -> Expr {
    match expr {
        Expr::Column(name) => match map.get(name) {
            Some(replacement) => replacement.clone(),
            None => expr.clone(),
        },
        Expr::Literal(_) => expr.clone(),
        Expr::Binary { op, left, right } => Expr::Binary {
            op: *op,
            left: Box::new(substitute_one(left, map)),
            right: Box::new(substitute_one(right, map)),
        },
        Expr::Alias(inner, name) => {
            Expr::Alias(Box::new(substitute_one(inner, map)), name.clone())
        }
    }
}

/// AND-fold a list of predicates into a single optional expression.
fn and_all(mut preds: Vec<Expr>) -> Option<Expr> {
    if preds.is_empty() {
        return None;
    }
    let mut acc = preds.remove(0);
    for p in preds {
        acc = Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(acc),
            right: Box::new(p),
        };
    }
    Some(acc)
}

/// Short tag describing a plan node's variant for error messages.
fn shape(plan: &LogicalPlan) -> &'static str {
    match plan {
        LogicalPlan::Scan { .. } => "Scan",
        LogicalPlan::Filter { .. } => "Filter",
        LogicalPlan::Project { .. } => "Project",
        LogicalPlan::Aggregate { .. } => "Aggregate",
        LogicalPlan::Distinct { .. } => "Distinct",
        LogicalPlan::Limit { .. } => "Limit",
        LogicalPlan::Sort { .. } => "Sort",
        LogicalPlan::Union { .. } => "Union",
        LogicalPlan::Join { .. } => "Join",
    }
}

/// Build a projection kernel over `input`, producing `exprs` (or all input columns).
/// Pure codegen helper: given a scan, optional predicate, and a list of
/// `(output_name, scan-namespace expr)`, build a `Projection` physical
/// plan. No chain resolution — all exprs must already reference scan
/// columns. Both `lower_projection` and `lower_aggregate` funnel through
/// this after they've folded any Project chain.
fn build_projection_kernel(
    table: &str,
    scan_schema: &Schema,
    predicate: Option<&Expr>,
    projected: &[(String, Expr)],
) -> BoltResult<PhysicalPlan> {
    let mut cg = Codegen::new(scan_schema);

    let predicate_reg = if let Some(pred) = predicate {
        let v = cg.emit_expr(pred)?;
        if v.dtype != DataType::Bool {
            return Err(BoltError::Type(format!(
                "filter predicate must be Bool, got {:?}",
                v.dtype
            )));
        }
        Some(v.reg)
    } else {
        None
    };

    let mut outputs = Vec::with_capacity(projected.len());
    let mut output_fields = Vec::with_capacity(projected.len());
    for (i, (name, expr)) in projected.iter().enumerate() {
        let value = cg.emit_expr(expr)?;
        cg.emit_store(value, i);
        outputs.push(ColumnIO {
            name: name.clone(),
            dtype: value.dtype,
        });
        output_fields.push(Field::new(name.clone(), value.dtype, true));
    }

    let kernel = cg.finish(outputs, predicate_reg);
    Ok(PhysicalPlan::Projection {
        table: table.to_string(),
        kernel,
        output_schema: Schema::new(output_fields),
    })
}

fn lower_projection(
    input: &LogicalPlan,
    exprs: Option<&[Expr]>,
    extra_predicate: Option<&Expr>,
) -> BoltResult<PhysicalPlan> {
    let resolved = resolve_source(input)?;
    let ResolvedSource {
        table,
        scan_schema,
        predicate: chain_predicate,
        projection: chain_projection,
    } = resolved;

    // Any `extra_predicate` lives in `input`'s output namespace; if a chain
    // Project sits at the top of `input`, rewrite the predicate through that
    // Project's output map so it ends up in scan namespace.
    let mut chain_proj_map: Option<HashMap<String, Expr>> = chain_projection
        .as_ref()
        .map(|named| named.iter().cloned().collect());

    let extra_pred_lowered: Option<Expr> = match extra_predicate {
        Some(p) => Some(match &chain_proj_map {
            Some(m) => substitute_one(p, m),
            None => p.clone(),
        }),
        None => None,
    };

    // Combine chain predicate AND extra predicate.
    let predicate = match (chain_predicate, extra_pred_lowered) {
        (None, None) => None,
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (Some(a), Some(b)) => Some(Expr::Binary {
            op: BinaryOp::And,
            left: Box::new(a),
            right: Box::new(b),
        }),
    };

    // Determine the projection list, in scan namespace, with output names.
    // Priority order:
    //   1. Explicit `exprs` argument (from a top-level Project on `input`),
    //      substituted through any chain Project map.
    //   2. The chain's own top Project, if any.
    //   3. Default: all scan columns as bare references.
    let owned_default: Vec<(String, Expr)>;
    let projected: &[(String, Expr)] = if let Some(es) = exprs {
        let mut subbed: Vec<(String, Expr)> = Vec::with_capacity(es.len());
        for (i, e) in es.iter().enumerate() {
            let name = output_name_for(e, i);
            let body = match e {
                Expr::Alias(inner, _) => (**inner).clone(),
                _ => e.clone(),
            };
            let lowered = match chain_proj_map.as_mut() {
                Some(m) => substitute_one(&body, m),
                None => body,
            };
            subbed.push((name, lowered));
        }
        owned_default = subbed;
        &owned_default
    } else if let Some(named) = chain_projection.as_ref() {
        named.as_slice()
    } else {
        owned_default = scan_schema
            .fields
            .iter()
            .map(|f| (f.name.clone(), Expr::Column(f.name.clone())))
            .collect();
        &owned_default
    };

    build_projection_kernel(table, scan_schema, predicate.as_ref(), projected)
}

/// Build an aggregate plan over `input` with the given group keys and aggregates.
fn lower_aggregate(
    plan: &LogicalPlan,
    input: &LogicalPlan,
    group_by: &[Expr],
    aggregates: &[AggregateExpr],
) -> BoltResult<PhysicalPlan> {
    let resolved = resolve_source(input)?;
    let table = resolved.table;
    let scan_schema = resolved.scan_schema;
    let scan_predicate = resolved.predicate;
    let chain_proj_map: Option<HashMap<String, Expr>> = resolved
        .projection
        .as_ref()
        .map(|named| named.iter().cloned().collect());
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
    // These exprs live in `input`'s output namespace; if a chain Project
    // sits at the top of `input`, substitute them through it so the rest
    // of this function sees scan-namespace exprs.
    let mut feed: Vec<Expr> = Vec::with_capacity(group_by.len() + agg_input_exprs.len());
    for e in group_by.iter().chain(agg_input_exprs.iter()) {
        let lowered = match chain_proj_map.as_ref() {
            Some(m) => substitute_one(e, m),
            None => e.clone(),
        };
        feed.push(lowered);
    }

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
                    return Err(BoltError::Plan(
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
        // `feed` exprs were already substituted into scan namespace above,
        // and `scan_predicate` is already in scan namespace, so go straight
        // to the pure-codegen builder (skipping another round of chain
        // resolution that would double-substitute).
        let named_feed: Vec<(String, Expr)> = feed
            .iter()
            .enumerate()
            .map(|(i, e)| (output_name_for(e, i), e.clone()))
            .collect();
        let pre_plan = build_projection_kernel(
            table,
            scan_schema,
            scan_predicate.as_ref(),
            &named_feed,
        )?;
        let (pre_kernel, _pre_schema) = match pre_plan {
            PhysicalPlan::Projection {
                kernel,
                output_schema,
                ..
            } => (kernel, output_schema),
            // `build_projection_kernel` always returns `Projection`.
            _ => {
                return Err(BoltError::Plan(
                    "internal: build_projection_kernel returned non-Projection".into(),
                ))
            }
        };
        let inputs: Vec<ColumnIO> = pre_kernel.outputs.clone();
        let group_ords: Vec<usize> = (0..group_by.len()).collect();
        (Some(pre_kernel), inputs, group_ords)
    };

    // Substitute aggregate exprs through any chain Project map so the
    // column names they reference match what the pre-aggregation kernel
    // actually exposes (i.e., scan namespace).
    let lowered_aggregates: Vec<AggregateExpr> = aggregates
        .iter()
        .map(|agg| match chain_proj_map.as_ref() {
            None => agg.clone(),
            Some(m) => match agg {
                AggregateExpr::Count(e) => AggregateExpr::Count(substitute_one(e, m)),
                AggregateExpr::Sum(e) => AggregateExpr::Sum(substitute_one(e, m)),
                AggregateExpr::Min(e) => AggregateExpr::Min(substitute_one(e, m)),
                AggregateExpr::Max(e) => AggregateExpr::Max(substitute_one(e, m)),
                AggregateExpr::Avg(e) => AggregateExpr::Avg(substitute_one(e, m)),
            },
        })
        .collect();

    let aggregate = AggregateSpec {
        inputs: agg_inputs,
        group_by: group_indices,
        aggregates: lowered_aggregates,
        output_schema,
    };

    Ok(PhysicalPlan::Aggregate {
        table: table.to_string(),
        pre,
        aggregate,
    })
}

/// True if `plan` is a Scan/Filter/Project chain that bottoms out in a Scan —
/// i.e. something `resolve_source` (and therefore `lower_projection`) can fold
/// down into a single-kernel `Projection`. Anything else (Aggregate, Distinct,
/// Limit, Sort, Union, Join) needs a recursive `lower` call instead, with the
/// outer Project/Filter applied — for wave 7 scaffolds, simply dropped — on
/// top of the result.
fn is_scan_chain(plan: &LogicalPlan) -> bool {
    let mut cur = plan;
    loop {
        match cur {
            LogicalPlan::Scan { .. } => return true,
            LogicalPlan::Filter { input, .. } | LogicalPlan::Project { input, .. } => {
                cur = input.as_ref();
            }
            _ => return false,
        }
    }
}

/// Lower a `LogicalPlan` to a `PhysicalPlan`.
/// True if a `Project { exprs, output_schema }` is a pure pass-through over
/// `input_schema`: every output field's name and source column line up with
/// the corresponding input field in the same position. When true, the
/// caller can drop the Project layer entirely (it would just clone the
/// input batch as-is). Aliased exprs only count as identity when the alias
/// happens to match the input column's name.
fn project_is_identity(
    exprs: &[Expr],
    input_schema: &Schema,
    output_schema: &Schema,
) -> bool {
    if exprs.len() != input_schema.fields.len() {
        return false;
    }
    if output_schema.fields.len() != input_schema.fields.len() {
        return false;
    }
    for (i, e) in exprs.iter().enumerate() {
        let in_name = input_schema.fields[i].name.as_str();
        let out_name = output_schema.fields[i].name.as_str();
        // Only bare `Column(name)` exprs can be no-ops: an `Alias(_, _)`
        // wrapper changes the output column's name even when the source
        // expression matches the input column, so it's never identity.
        let src_name = match e {
            Expr::Column(n) => n.as_str(),
            _ => return false,
        };
        if src_name != in_name || out_name != in_name {
            return false;
        }
    }
    true
}

pub fn lower(plan: &LogicalPlan) -> BoltResult<PhysicalPlan> {
    match plan {
        LogicalPlan::Project { input, exprs } => {
            // Scan/Filter/Project chain → single fused kernel via `lower_projection`.
            // Otherwise (Project over Aggregate, Join, Distinct, etc.) we
            // can't fold into one kernel; emit a thin `Project` rename/reorder
            // layer over the lowered inner plan so SELECT-list order and
            // aliases survive (wave 1 fix #5: SELECT-order Project on top
            // of Aggregate must surface aliased / reordered output names).
            // If the Project is a structural no-op (same field names in the
            // same order as the lowered inner plan), drop it so downstream
            // pattern-matchers (and tests) see the bare inner plan.
            if is_scan_chain(input) {
                lower_projection(input, Some(exprs), None)
            } else {
                let inner = lower(input)?;
                let output_schema = plan.schema()?;
                if project_is_identity(exprs, inner.output_schema(), &output_schema) {
                    Ok(inner)
                } else {
                    Ok(PhysicalPlan::Project {
                        input: Box::new(inner),
                        exprs: exprs.clone(),
                        output_schema,
                    })
                }
            }
        }
        LogicalPlan::Filter { input, predicate } => {
            if is_scan_chain(input) {
                lower_projection(input, None, Some(predicate))
            } else {
                // Same rationale as Project: forwarding the inner plan keeps
                // the lowerer total. Execution on the inner scaffold variant
                // will error before the filter ever applies.
                lower(input)
            }
        }
        LogicalPlan::Scan { .. } => lower_projection(plan, None, None),
        LogicalPlan::Aggregate {
            input,
            group_by,
            aggregates,
        } => lower_aggregate(plan, input, group_by, aggregates),
        LogicalPlan::Distinct { input } => {
            let inner = lower(input)?;
            Ok(PhysicalPlan::Distinct {
                input: Box::new(inner),
            })
        }
        LogicalPlan::Limit {
            input,
            limit,
            offset,
        } => {
            let inner = lower(input)?;
            Ok(PhysicalPlan::Limit {
                input: Box::new(inner),
                limit: *limit,
                offset: *offset,
            })
        }
        LogicalPlan::Sort { input, sort_exprs } => {
            let inner = lower(input)?;
            Ok(PhysicalPlan::Sort {
                input: Box::new(inner),
                sort_exprs: sort_exprs.clone(),
            })
        }
        LogicalPlan::Union { inputs } => {
            if inputs.is_empty() {
                return Err(BoltError::Plan(
                    "UNION requires at least one input".into(),
                ));
            }
            let mut lowered: Vec<PhysicalPlan> = Vec::with_capacity(inputs.len());
            for branch in inputs {
                lowered.push(lower(branch)?);
            }
            // Schema integrity (matching shapes across branches) was
            // already enforced by `LogicalPlan::schema()`; trust that.
            Ok(PhysicalPlan::Union { inputs: lowered })
        }
        LogicalPlan::Join {
            left,
            right,
            join_type,
            on,
        } => {
            let l = lower(left)?;
            let r = lower(right)?;
            // Build the combined schema *from the physical inputs*: the
            // logical sides may have been folded / projected differently
            // than their physical counterparts, but for the operators
            // currently supported below a Join the two agree. Using the
            // physical sides keeps the stored schema in lock-step with
            // what the executor will actually see at run time.
            let output_schema =
                join_combined_schema(l.output_schema(), r.output_schema());
            Ok(PhysicalPlan::Join {
                left: Box::new(l),
                right: Box::new(r),
                join_type: *join_type,
                on: on.clone(),
                output_schema,
            })
        }
    }
}
