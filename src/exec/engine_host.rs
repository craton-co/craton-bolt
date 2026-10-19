// SPDX-License-Identifier: Apache-2.0
//! The host execution tier chosen by the cost model (see
//! [`crate::exec::cost`]): row-wise statements and scalar aggregates over one
//! table, evaluated with the host expression evaluator.

use std::sync::Arc;
use std::time::Instant;

use arrow::compute::{lexsort_to_indices, take, SortColumn, SortOptions};
use arrow_array::{ArrayRef, RecordBatch};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

use crate::error::{BoltError, BoltResult};
use crate::exec::cost::{CostInputs, DefaultCostModel, OperatorCost, OperatorKind, TierPolicy};
use crate::exec::engine::{Engine, QueryHandle};
use crate::exec::expr_agg::{eval_expr, ColumnEnv, HostColumn};
use crate::plan::logical_plan::{
    AggregateExpr, BinaryOp, DataType, Expr, Literal, LogicalPlan, Schema,
};
use crate::plan::ExecutionTier;

/// Error the host evaluator returns for an edge case it leaves to the GPU
/// path (whose semantics are authoritative).
fn decline(what: &str) -> BoltError {
    BoltError::GpuCapacity(format!("host tier declines: {what}"))
}

/// `true` when the host evaluator handles `e` with the GPU path's semantics.
fn host_expr_ok(e: &Expr) -> bool {
    match e {
        Expr::Column(_) => true,
        Expr::Literal(l) => matches!(
            l,
            Literal::Null
                | Literal::Bool(_)
                | Literal::Int32(_)
                | Literal::Int64(_)
                | Literal::Float32(_)
                | Literal::Float64(_)
                | Literal::Utf8(_)
        ),
        Expr::Alias(inner, _) => host_expr_ok(inner),
        Expr::Binary { op, left, right } => {
            matches!(
                op,
                BinaryOp::Add
                    | BinaryOp::Sub
                    | BinaryOp::Mul
                    | BinaryOp::Eq
                    | BinaryOp::NotEq
                    | BinaryOp::Lt
                    | BinaryOp::LtEq
                    | BinaryOp::Gt
                    | BinaryOp::GtEq
                    | BinaryOp::And
                    | BinaryOp::Or
            ) && host_expr_ok(left)
                && host_expr_ok(right)
        }
        Expr::Unary { operand, .. } => host_expr_ok(operand),
        Expr::Case {
            branches,
            else_branch,
        } => {
            branches
                .iter()
                .all(|(w, t)| host_expr_ok(w) && host_expr_ok(t))
                && else_branch.as_deref().is_none_or(host_expr_ok)
        }
        Expr::Cast {
            expr,
            target,
            safe: false,
        } => {
            matches!(
                target,
                DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64
            ) && host_expr_ok(expr)
        }
        _ => false,
    }
}

/// A statement the host tier can run: its one table, shape, and the
/// expressions it evaluates per row.
struct HostShape<'p> {
    table: &'p str,
    kind: OperatorKind,
    expressions: usize,
    limit: Option<usize>,
}

fn host_shape(plan: &LogicalPlan) -> Option<HostShape<'_>> {
    fn walk<'p>(
        plan: &'p LogicalPlan,
        exprs: &mut usize,
        agg: &mut bool,
        limit: &mut Option<usize>,
    ) -> Option<&'p str> {
        match plan {
            LogicalPlan::Scan { table, .. } => Some(table),
            LogicalPlan::Filter { input, predicate } => {
                if !host_expr_ok(predicate) {
                    return None;
                }
                *exprs += 1;
                walk(input, exprs, agg, limit)
            }
            LogicalPlan::Project { input, exprs: es } => {
                if !es.iter().all(host_expr_ok) {
                    return None;
                }
                *exprs += es.len();
                walk(input, exprs, agg, limit)
            }
            LogicalPlan::Limit {
                input,
                limit: l,
                offset,
            } => {
                limit.get_or_insert(l.saturating_add(*offset));
                walk(input, exprs, agg, limit)
            }
            LogicalPlan::Sort { input, sort_exprs } => {
                if !sort_exprs.iter().all(|s| matches!(s.expr, Expr::Column(_))) {
                    return None;
                }
                *exprs += sort_exprs.len();
                walk(input, exprs, agg, limit)
            }
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                if *agg || !group_by.is_empty() {
                    return None;
                }
                let ok = aggregates.iter().all(|a| match a {
                    AggregateExpr::Count(e)
                    | AggregateExpr::Sum(e)
                    | AggregateExpr::Min(e)
                    | AggregateExpr::Max(e)
                    | AggregateExpr::Avg(e) => host_expr_ok(e),
                    _ => false,
                });
                if !ok {
                    return None;
                }
                *agg = true;
                *exprs += aggregates.len();
                walk(input, exprs, agg, limit)
            }
            _ => None,
        }
    }
    let mut expressions = 0;
    let mut agg = false;
    let mut limit = None;
    let table = walk(plan, &mut expressions, &mut agg, &mut limit)?;
    Some(HostShape {
        table,
        kind: if agg {
            OperatorKind::ScalarAggregate
        } else {
            OperatorKind::RowWise
        },
        expressions,
        limit,
    })
}

fn host_env(batch: &RecordBatch) -> BoltResult<Vec<(String, HostColumn)>> {
    let n = batch.num_rows();
    batch
        .schema()
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(f, c)| {
            Ok((
                f.name().clone(),
                crate::exec::filter::arrow_array_to_host_column(c.as_ref(), n)?,
            ))
        })
        .collect()
}

fn to_arrow(c: HostColumn) -> BoltResult<ArrayRef> {
    crate::exec::engine_support::host_column_to_arrow_array(c)
}

fn batch_with_schema(schema: &Schema, columns: Vec<ArrayRef>) -> BoltResult<RecordBatch> {
    // Temporal results arrive as their i32 / i64 storage.
    let declared = crate::exec::schema_convert::plan_schema_to_arrow_schema(schema)?;
    let columns = crate::exec::engine_support::restore_temporal_types(columns, &declared)?;
    let fields: Vec<ArrowField> = schema
        .fields
        .iter()
        .zip(&columns)
        .map(|(f, c)| ArrowField::new(f.name.clone(), c.data_type().clone(), true))
        .collect();
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
        .map_err(|e| BoltError::Other(format!("host tier: {e}")))
}

/// One scalar aggregate over an evaluated input column.
fn host_aggregate(a: &AggregateExpr, col: &HostColumn, out: DataType) -> BoltResult<ArrayRef> {
    use arrow_array::{Float32Array, Float64Array, Int32Array, Int64Array};
    if let AggregateExpr::Count(_) = a {
        let n = match col {
            HostColumn::Bool(v) => v.iter().flatten().count(),
            HostColumn::I32(v) => v.iter().flatten().count(),
            HostColumn::I64(v) => v.iter().flatten().count(),
            HostColumn::F32(v) => v.iter().flatten().count(),
            HostColumn::F64(v) => v.iter().flatten().count(),
            HostColumn::Utf8(v) => v.iter().flatten().count(),
        };
        return Ok(Arc::new(Int64Array::from(vec![n as i64])));
    }
    // Empty / all-NULL inputs are left to the GPU path.
    let ints = |v: &[Option<i64>]| -> Vec<i64> { v.iter().flatten().copied().collect() };
    let floats = |v: &[Option<f64>]| -> Vec<f64> { v.iter().flatten().copied().collect() };
    let (iv, fv): (Option<Vec<i64>>, Option<Vec<f64>>) = match col {
        HostColumn::I32(v) => (
            Some(v.iter().map(|x| x.map(i64::from)).collect::<Vec<_>>()).map(|v| ints(&v)),
            None,
        ),
        HostColumn::I64(v) => (Some(ints(v)), None),
        HostColumn::F32(v) => (
            None,
            Some(floats(
                &v.iter().map(|x| x.map(f64::from)).collect::<Vec<_>>(),
            )),
        ),
        HostColumn::F64(v) => (None, Some(floats(v))),
        _ => return Err(decline("aggregate over a non-numeric column")),
    };
    if iv.as_ref().is_some_and(Vec::is_empty) || fv.as_ref().is_some_and(Vec::is_empty) {
        return Err(decline("aggregate over no non-NULL values"));
    }
    let value: f64OrI64 = match (a, &iv, &fv) {
        (AggregateExpr::Sum(_), Some(v), _) => f64OrI64::I(
            v.iter()
                .try_fold(0i64, |s, x| s.checked_add(*x))
                .ok_or_else(|| decline("integer SUM overflow"))?,
        ),
        (AggregateExpr::Sum(_), _, Some(v)) => f64OrI64::F(v.iter().sum()),
        (AggregateExpr::Min(_), Some(v), _) => f64OrI64::I(*v.iter().min().expect("non-empty")),
        (AggregateExpr::Max(_), Some(v), _) => f64OrI64::I(*v.iter().max().expect("non-empty")),
        (AggregateExpr::Min(_), _, Some(v)) => {
            if v.iter().any(|x| x.is_nan()) {
                return Err(decline("MIN over NaN"));
            }
            f64OrI64::F(v.iter().copied().fold(f64::INFINITY, f64::min))
        }
        (AggregateExpr::Max(_), _, Some(v)) => {
            if v.iter().any(|x| x.is_nan()) {
                return Err(decline("MAX over NaN"));
            }
            f64OrI64::F(v.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        }
        (AggregateExpr::Avg(_), Some(v), _) => {
            f64OrI64::F(v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64)
        }
        (AggregateExpr::Avg(_), _, Some(v)) => f64OrI64::F(v.iter().sum::<f64>() / v.len() as f64),
        _ => return Err(decline("aggregate")),
    };
    Ok(match (out, value) {
        (DataType::Int64, f64OrI64::I(x)) => Arc::new(Int64Array::from(vec![x])),
        (DataType::Int32, f64OrI64::I(x)) => Arc::new(Int32Array::from(vec![
            i32::try_from(x).map_err(|_| decline("Int32 SUM overflow"))?
        ])),
        (DataType::Float64, f64OrI64::F(x)) => Arc::new(Float64Array::from(vec![x])),
        (DataType::Float64, f64OrI64::I(x)) => Arc::new(Float64Array::from(vec![x as f64])),
        (DataType::Float32, f64OrI64::F(x)) => Arc::new(Float32Array::from(vec![x as f32])),
        _ => return Err(decline("aggregate output type")),
    })
}

#[allow(non_camel_case_types)]
enum f64OrI64 {
    F(f64),
    I(i64),
}

impl Engine {
    /// Cost estimate of `plan` on each tier, if the host tier supports its
    /// shape. `compiled`: its GPU kernels are known to be cached.
    pub(crate) fn tier_estimate(
        &self,
        plan: &LogicalPlan,
        compiled: bool,
    ) -> Option<(CostInputs, OperatorCost)> {
        let shape = host_shape(plan)?;
        let table = shape.table;
        let rows = if let Some(t) = self.storage_table(table) {
            let tables = vec![table.to_string()];
            let req = self
                .storage_scan_requests(plan, &tables, &[])
                .remove(table)
                .unwrap_or_default();
            t.plan_scan(&req).rows
        } else if let Some(b) = self.tables.get(table) {
            b.iter().map(|b| b.num_rows() as u64).sum()
        } else {
            match self.streaming_sources.borrow().get(table) {
                Some(crate::exec::streaming::TableSource::Materialized(b)) => {
                    b.iter().map(|b| b.num_rows() as u64).sum()
                }
                // An undrained lazy stream: its size is unknown.
                _ => return None,
            }
        };
        let total_rows = self
            .table_statistics(table)
            .map(|s| s.row_count.max(1))
            .unwrap_or(1);
        let table_bytes = self.estimated_table_bytes(table).unwrap_or(0) as u64;
        let row_bytes = (table_bytes / total_rows).max(1);
        let input_bytes = rows * row_bytes;
        let out_rows = shape.limit.map_or(rows, |l| (l as u64).min(rows));
        let inputs = CostInputs {
            kind: shape.kind,
            rows,
            input_bytes,
            output_bytes: out_rows * row_bytes,
            expressions: shape.expressions,
            resident: self
                .gpu_tables
                .borrow()
                .get(table)
                .is_some_and(Option::is_some),
            compiled,
        };
        let cost = self.cost_model.borrow().estimate(&inputs);
        Some((inputs, cost))
    }

    /// Run `plan` on the host when the cost-based policy prefers it. `None`
    /// leaves it to the GPU path (including when the host evaluator
    /// declines the statement at run time).
    pub(crate) fn try_host_tier(
        &self,
        plan: &LogicalPlan,
        compiled: bool,
    ) -> BoltResult<Option<QueryHandle>> {
        if self.tier_policy != TierPolicy::CostBased {
            return Ok(None);
        }
        let Some((_, cost)) = self.tier_estimate(plan, compiled) else {
            return Ok(None);
        };
        if cost.preferred_tier() != ExecutionTier::Host {
            return Ok(None);
        }
        match self.run_on_host(plan) {
            Ok(b) => {
                crate::metrics::metrics().inc(crate::metrics::Counter::HostTierQueries);
                Ok(Some(QueryHandle::from_record_batch(b)))
            }
            Err(e) => {
                log::debug!("host tier: falling back to the GPU path ({e})");
                Ok(None)
            }
        }
    }

    /// Execute a [`host_shape`] plan with the host evaluator.
    fn run_on_host(&self, plan: &LogicalPlan) -> BoltResult<RecordBatch> {
        let table = host_shape(plan)
            .ok_or_else(|| decline("statement shape"))?
            .table
            .to_string();
        let tables = vec![table.clone()];
        self.scan_storage_tables(plan, &tables, &[], true)?;
        self.materialize_streaming_source(&table)?;
        self.host_eval(plan)
    }

    fn host_eval(&self, plan: &LogicalPlan) -> BoltResult<RecordBatch> {
        let out_schema = plan.schema()?;
        match plan {
            LogicalPlan::Scan { table, .. } => {
                let b = self.materialize_table(table)?;
                let mut fields = Vec::new();
                let mut cols = Vec::new();
                for f in &out_schema.fields {
                    match b.schema().index_of(&f.name) {
                        Ok(i) => {
                            fields.push(f.clone());
                            cols.push(Arc::clone(b.column(i)));
                        }
                        // Dictionary columns the GPU path synthesises; the
                        // statement itself never reads them.
                        Err(_) if f.name.starts_with("__idx_") || f.name.starts_with("__rank_") => {
                        }
                        Err(_) => return Err(decline("column missing from the host batch")),
                    }
                }
                batch_with_schema(&Schema::new(fields), cols)
            }
            LogicalPlan::Filter { input, predicate } => {
                let b = self.host_eval(input)?;
                Ok(crate::exec::filter::execute_filter(
                    QueryHandle::from_record_batch(b),
                    predicate,
                )?
                .into_record_batch())
            }
            LogicalPlan::Project { input, exprs } => {
                let b = self.host_eval(input)?;
                let n = b.num_rows();
                let owned = host_env(&b)?;
                let env: ColumnEnv<'_> = owned.iter().map(|(k, c)| (k.clone(), c)).collect();
                let mut cols = Vec::with_capacity(exprs.len());
                for (e, f) in exprs.iter().zip(&out_schema.fields) {
                    let mut inner = e;
                    while let Expr::Alias(x, _) = inner {
                        inner = x;
                    }
                    let col = match inner {
                        Expr::Column(c) => match b.schema().index_of(c) {
                            Ok(i) => Arc::clone(b.column(i)),
                            Err(_) => return Err(decline("unknown column")),
                        },
                        other => to_arrow(eval_expr(other, &env, f.dtype, n)?)?,
                    };
                    cols.push(col);
                }
                batch_with_schema(&out_schema, cols)
            }
            LogicalPlan::Limit {
                input,
                limit,
                offset,
            } => {
                let b = self.host_eval(input)?;
                let start = (*offset).min(b.num_rows());
                let len = (*limit).min(b.num_rows() - start);
                Ok(b.slice(start, len))
            }
            LogicalPlan::Sort { input, sort_exprs } => {
                let b = self.host_eval(input)?;
                if b.num_rows() == 0 {
                    return Ok(b);
                }
                let keys = sort_exprs
                    .iter()
                    .map(|s| {
                        let Expr::Column(c) = &s.expr else {
                            return Err(decline("sort key"));
                        };
                        let i = b.schema().index_of(c).map_err(|_| decline("sort key"))?;
                        Ok(SortColumn {
                            values: Arc::clone(b.column(i)),
                            options: Some(SortOptions {
                                descending: s.descending,
                                nulls_first: s.nulls_first,
                            }),
                        })
                    })
                    .collect::<BoltResult<Vec<_>>>()?;
                let idx = lexsort_to_indices(&keys, None)
                    .map_err(|e| BoltError::Other(format!("host tier: sort: {e}")))?;
                let cols = b
                    .columns()
                    .iter()
                    .map(|c| take(c.as_ref(), &idx, None))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| BoltError::Other(format!("host tier: sort: {e}")))?;
                RecordBatch::try_new(b.schema(), cols)
                    .map_err(|e| BoltError::Other(format!("host tier: sort: {e}")))
            }
            LogicalPlan::Aggregate {
                input, aggregates, ..
            } => {
                let b = self.host_eval(input)?;
                let n = b.num_rows();
                if n == 0 {
                    return Err(decline("aggregate over an empty input"));
                }
                let owned = host_env(&b)?;
                let env: ColumnEnv<'_> = owned.iter().map(|(k, c)| (k.clone(), c)).collect();
                let mut cols = Vec::with_capacity(aggregates.len());
                for (a, f) in aggregates.iter().zip(&out_schema.fields) {
                    let (e, dtype) = match a {
                        AggregateExpr::Count(e) => (e, None),
                        AggregateExpr::Avg(e) => (e, Some(DataType::Float64)),
                        AggregateExpr::Sum(e) | AggregateExpr::Min(e) | AggregateExpr::Max(e) => {
                            (e, Some(f.dtype))
                        }
                        _ => return Err(decline("aggregate")),
                    };
                    let col = match (e, dtype) {
                        // COUNT(*) / COUNT(<literal>): every row.
                        (Expr::Literal(l), None) if !matches!(l, Literal::Null) => {
                            Arc::new(arrow_array::Int64Array::from(vec![n as i64])) as ArrayRef
                        }
                        (e, None) => {
                            // COUNT(col) evaluates the column at its own type.
                            let Expr::Column(c) = e else {
                                return Err(decline("COUNT over an expression"));
                            };
                            let i = b
                                .schema()
                                .index_of(c)
                                .map_err(|_| decline("unknown column"))?;
                            let col = crate::exec::filter::arrow_array_to_host_column(
                                b.column(i).as_ref(),
                                n,
                            )?;
                            host_aggregate(a, &col, f.dtype)?
                        }
                        (e, Some(dt)) => host_aggregate(a, &eval_expr(e, &env, dt, n)?, f.dtype)?,
                    };
                    cols.push(col);
                }
                batch_with_schema(&out_schema, cols)
            }
            _ => Err(decline("statement shape")),
        }
    }

    /// Measure the [`DefaultCostModel`] constants on this device — copy
    /// bandwidth, per-statement launch overhead, per-row device and host
    /// throughput, compile time — and install the result. Takes about a
    /// second; registered tables are not touched (a temporary table is used
    /// and removed).
    pub fn calibrate_cost_model(&mut self) -> BoltResult<DefaultCostModel> {
        const ROWS: i64 = 1_000_000;
        const CAL: &str = "__bolt_calibrate";
        const CAL_SMALL: &str = "__bolt_calibrate_small";
        let secs = |t: Instant| t.elapsed().as_secs_f64().max(1e-9);
        // Copy bandwidth.
        let bytes = vec![0u8; 32 << 20];
        let t = Instant::now();
        let dev = crate::cuda::GpuVec::<u8>::from_slice(&bytes)?;
        let h2d = bytes.len() as f64 / secs(t);
        let t = Instant::now();
        let back = dev.to_vec()?;
        let d2h = back.len() as f64 / secs(t);
        drop((dev, back));

        let make = |n: i64| -> BoltResult<RecordBatch> {
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "x",
                arrow_schema::DataType::Int64,
                false,
            )]));
            RecordBatch::try_new(
                schema,
                vec![Arc::new(arrow_array::Int64Array::from(
                    (0..n).collect::<Vec<_>>(),
                ))],
            )
            .map_err(|e| BoltError::Other(e.to_string()))
        };
        for name in [CAL, CAL_SMALL] {
            if self.has_table(name) {
                self.deregister_table(name)?;
            }
        }
        self.register_table(CAL, make(ROWS)?)?;
        self.register_table(CAL_SMALL, make(16)?)?;
        let policy = std::mem::replace(&mut self.tier_policy, TierPolicy::GpuFirst);
        let result = (|| -> BoltResult<DefaultCostModel> {
            let q = |t: &str| format!("SELECT x + 1 AS y FROM {t} WHERE x > 10");
            let time = |e: &Engine, sql: &str| -> BoltResult<f64> {
                let t = Instant::now();
                e.sql(sql)?;
                Ok(t.elapsed().as_secs_f64() * 1e9)
            };
            // Cold (compile + upload), then warm.
            let cold_small = time(self, &q(CAL_SMALL))?;
            let warm_small = time(self, &q(CAL_SMALL))?.min(time(self, &q(CAL_SMALL))?);
            time(self, &q(CAL))?;
            let warm_big = time(self, &q(CAL))?.min(time(self, &q(CAL))?);
            let out_bytes = ROWS as f64 * 8.0;
            let per_row =
                ((warm_big - warm_small - out_bytes / d2h * 1e9) / (ROWS as f64 * 2.0)).max(0.001);
            let plan = crate::plan::sql_frontend::parse(&q(CAL), &self.provider)?;
            let t = Instant::now();
            self.host_eval(&plan)?;
            let cpu = (t.elapsed().as_secs_f64() * 1e9 / (ROWS as f64 * 2.0)).max(0.01);
            Ok(DefaultCostModel {
                h2d_bytes_per_sec: h2d,
                d2h_bytes_per_sec: d2h,
                launch_ns: warm_small,
                gpu_ns_per_row: per_row,
                cpu_ns_per_row: cpu,
                compile_ns: (cold_small - warm_small).max(0.0),
            })
        })();
        self.tier_policy = policy;
        self.deregister_table(CAL)?;
        self.deregister_table(CAL_SMALL)?;
        let model = result?;
        *self.cost_model.borrow_mut() = Arc::new(model);
        self.compiled_plans.borrow_mut().clear();
        Ok(model)
    }

    /// Change the [`TierPolicy`].
    pub fn set_tier_policy(&mut self, policy: TierPolicy) {
        self.tier_policy = policy;
        // Cached GPU plans encode the previous policy's decisions.
        self.compiled_plans.borrow_mut().clear();
    }

    /// The current [`TierPolicy`].
    pub fn tier_policy(&self) -> TierPolicy {
        self.tier_policy
    }

    /// Render the tier decision of `plan` for `EXPLAIN` (`None` when the
    /// host tier does not support its shape).
    pub(crate) fn explain_tier(&self, plan: &LogicalPlan) -> Option<String> {
        let (inputs, cost) = self.tier_estimate(plan, false)?;
        let chosen = match (self.tier_policy, cost.preferred_tier()) {
            (TierPolicy::CostBased, ExecutionTier::Host) => "host",
            (TierPolicy::CostBased, _) => "GPU",
            (_, _) => "GPU (policy: gpu-first)",
        };
        Some(format!(
            "Tier: {chosen} — estimate host ~{:.0} µs, GPU ~{:.0} µs (+{:.0} ms compile if cold), \
             {} rows, H2D {} bytes{}",
            cost.cpu_ns / 1e3,
            cost.gpu_ns / 1e3,
            cost.compilation_ns / 1e6,
            inputs.rows,
            cost.h2d_bytes,
            if inputs.resident { ", resident" } else { "" }
        ))
    }
}
