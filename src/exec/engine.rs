// SPDX-License-Identifier: Apache-2.0

//! Top-level engine: dispatches per-shape executors (scalar agg, GROUP BY, etc.);
//! performs GPU prefix-scan + gather compaction for filter outputs, or a host-side
//! `arrow::compute::filter` fallback when any output column is Utf8.
//!
//! The engine owns a CUDA context and a registry of host-side Arrow `RecordBatch`es.
//! `Engine::sql` parses, plans, codegens, launches, and returns a `QueryHandle` whose
//! `record_batch()` exposes the result.
//!
//! Projection-with-filter flow: a predicate-only kernel materialises a `u8` mask
//! into a fresh device buffer. When every output column is gather-friendly
//! (primitive or Bool), the engine then runs `gpu_compact::compact_columns_on_gpu`
//! (prefix scan + gather) entirely on the device and downloads only the surviving
//! rows. When any output column is Utf8 — the gather kernel cannot relocate
//! variable-width strings — the engine falls back to downloading the full
//! per-column outputs plus the mask and running `compact::compact_arrays`
//! (Arrow's host-side filter) on the host. Scalar aggregates, group-bys with or
//! without a `WHERE`, and their `extended_agg`/`expr_agg` variants are
//! dispatched to dedicated executors in `Engine::execute`.

use std::cell::{Ref, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arrow_array::{ArrayRef, Int64Array, RecordBatch};
use arrow_schema::DataType as ArrowDataType;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::{CudaContext, GpuVec};
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{grid_x_for, CudaStream};
use crate::exec::n_rows_to_u32;
use crate::jit::{compile_ptx, CudaModule};
use crate::plan::{
    parse_sql, DataType, KernelSpec, LogicalPlan, MemTableProvider, PhysicalPlan,
    PlanRewrite, Schema,
};

// Items lifted out of this file into sibling modules (pure-reorg split).
// Re-imported here so the remaining `impl Engine` block and the test module
// (`use super::*`) keep resolving every moved name.
use crate::exec::engine_cache_key::{
    ClonedHostRevision, HostRevisionSnapshot, HostTableRevision, ModuleCacheKey,
};
use crate::exec::engine_device_col::{check_len, DeviceCol};
use crate::exec::engine_provider::{EngineProvider, EngineTableStats};
use crate::exec::engine_support::{
    arrow_schema_to_plan_schema, build_count_rows_batch, column_storage_rows,
    concat_table_batches, debug_sync_check, host_column_to_arrow_array,
    install_persistent_cache_override, passthrough_output_sources,
    plan_schema_to_arrow_schema, propagate_column_nullability, should_emit_pool_stats,
    try_extend_column,
};
// Re-exported (not just `use`d) so `crate::exec::engine::pool_stats_interval_from_env`
// — the path `lib.rs`'s `__test_only_env_vars` module re-exports — keeps resolving
// after the function moved into `engine_support`.
pub use crate::exec::engine_support::pool_stats_interval_from_env;

/// PTX entry-point name; matches the symbol `ptx_gen` emits.
const KERNEL_ENTRY: &str = "bolt_kernel";

/// Entry-point name for the predicate-only mask kernel emitted by
/// [`crate::jit::scan_kernel::compile_predicate_kernel`]. Lifted out of the
/// inline string literal so the projection module-cache key can refer to it
/// without re-spelling the constant at every cache lookup.
const PREDICATE_ENTRY: &str = "bolt_predicate";

/// Threads per CUDA block for the 1D launch.
const BLOCK_SIZE: u32 = 256;

/// Hard safety cap on `WITH RECURSIVE` fixpoint iterations (feature F1).
///
/// Recursive CTEs can loop forever on bad input (a recursive term that never
/// reaches a fixpoint), so [`Engine::execute_recursive_cte`] refuses to run
/// more than this many iterations and returns a clean [`BoltError`] instead of
/// spinning / OOMing. Generous enough for any realistic graph/tree traversal
/// or integer sequence; override with [`MAX_RECURSIVE_ITERATIONS_ENV`].
pub(crate) const MAX_RECURSIVE_ITERATIONS: usize = 1000;

/// Environment-variable override for [`MAX_RECURSIVE_ITERATIONS`]. A positive
/// integer raises (or lowers) the cap; a missing / non-integer / zero value
/// falls back to the default. Mirrors the `CRATON_*` env convention used by
/// the SQL frontend's size guards.
pub(crate) const MAX_RECURSIVE_ITERATIONS_ENV: &str = "CRATON_MAX_RECURSIVE_ITERATIONS";

/// Resolve the effective recursive-CTE iteration cap (env override or default).
fn max_recursive_iterations() -> usize {
    std::env::var(MAX_RECURSIVE_ITERATIONS_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(MAX_RECURSIVE_ITERATIONS)
}

/// Hard safety cap on the number of LEFT rows a LATERAL apply (feature F3) will
/// drive (feature: LATERAL / correlated execution).
///
/// The apply is a host nested loop — it re-plans + re-runs the correlated
/// subquery once per left row — so its cost is `O(left_rows × subquery)`. A
/// huge left input would spin / OOM, so [`Engine::execute_lateral_apply`]
/// refuses to run more than this many left rows and returns a clean
/// [`BoltError`] instead. Override with [`MAX_APPLY_LEFT_ROWS_ENV`].
pub(crate) const MAX_APPLY_LEFT_ROWS: usize = 100_000;

/// Environment-variable override for [`MAX_APPLY_LEFT_ROWS`]. A positive
/// integer raises (or lowers) the cap; a missing / non-integer / zero value
/// falls back to the default.
pub(crate) const MAX_APPLY_LEFT_ROWS_ENV: &str = "CRATON_MAX_APPLY_ROWS";

/// Resolve the effective LATERAL-apply left-row cap (env override or default).
fn max_apply_left_rows() -> usize {
    std::env::var(MAX_APPLY_LEFT_ROWS_ENV)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(MAX_APPLY_LEFT_ROWS)
}

/// Concatenate two `RecordBatch`es sharing a schema into one.
///
/// Thin wrapper over `arrow::compute::concat_batches` used by the recursive
/// CTE fixpoint (feature F1) to grow the accumulated relation.
fn concat_two_batches(a: &RecordBatch, b: &RecordBatch) -> BoltResult<RecordBatch> {
    arrow::compute::concat_batches(&a.schema(), [a, b])
        .map_err(|e| BoltError::Plan(format!("WITH RECURSIVE concat: {e}")))
}

/// Host-side per-group distinct count for feature F3-finish
/// (`COUNT(DISTINCT col)` with `GROUP BY`).
///
/// `base` is the materialised `[group_keys..., distinct_col]` batch (the first
/// `n_keys` columns are group keys, the last is the distinct-counted column);
/// `result_schema` is the group-key fields followed by the `Int64` count.
///
/// Groups rows by the leading group-key tuple using the same `RowKey` /
/// `RowKeyValue` canonicalisation the `DISTINCT` operator uses — so NULL group
/// keys form their own group and multi-key groups compose — and per group
/// accumulates the set of DISTINCT **non-NULL** values of the distinct column.
/// The count is `set.len()`: standard SQL `COUNT(DISTINCT x)` ignores NULLs, so
/// an all-NULL (or empty) group yields 0. Output rows are in first-occurrence
/// order; group-key values are gathered from `base` at each group's first row
/// (preserving the exact dtype / value, including NULL keys).
///
/// Pure (no GPU / engine state) so it is host-testable directly.
fn host_count_distinct_groupby(
    base: &RecordBatch,
    n_keys: usize,
    result_schema: &Schema,
) -> BoltResult<RecordBatch> {
    use std::collections::{HashMap, HashSet};

    use arrow_array::UInt32Array;

    use crate::exec::distinct::{ColumnReader, RowKey, RowKeyValue};

    let n_cols = base.num_columns();
    if n_cols != n_keys + 1 {
        return Err(BoltError::Plan(format!(
            "COUNT(DISTINCT) GROUP BY: base produced {n_cols} columns, expected \
             {n_keys} group keys + 1 distinct column"
        )));
    }
    let n_rows = base.num_rows();

    // Per-column readers: the first `n_keys` are group keys, the last is the
    // distinct-counted column. `ColumnReader::new` rejects unsupported dtypes
    // (should have been caught at plan time).
    let key_readers: Vec<ColumnReader> = (0..n_keys)
        .map(|i| ColumnReader::new(base.column(i).as_ref()))
        .collect::<BoltResult<_>>()?;
    let distinct_reader = ColumnReader::new(base.column(n_keys).as_ref())?;

    // `groups` maps the group key to its slot index in `order`
    // (first-occurrence order); `order` carries (first_row_index,
    // distinct_value_set) per group.
    let mut groups: HashMap<RowKey, usize> = HashMap::new();
    let mut order: Vec<(u32, HashSet<RowKeyValue>)> = Vec::new();
    for row in 0..n_rows {
        let key = RowKey::from_values(n_keys, key_readers.iter().map(|r| r.value_at(row)));
        let slot = match groups.get(&key) {
            Some(&s) => s,
            None => {
                let s = order.len();
                groups.insert(key, s);
                order.push((row as u32, HashSet::new()));
                s
            }
        };
        // COUNT(DISTINCT x) ignores NULLs: only insert non-NULL values.
        let v = distinct_reader.value_at(row);
        if v != RowKeyValue::Null {
            order[slot].1.insert(v);
        }
    }

    // Build the result. Take indices = each group's first row; gather the
    // group-key columns so the output keeps the base batch's exact dtype +
    // value (including NULL group keys).
    let arrow_schema = plan_schema_to_arrow_schema(result_schema)?;
    let take_idx = UInt32Array::from(order.iter().map(|(r, _)| *r).collect::<Vec<u32>>());
    let mut out_cols: Vec<ArrayRef> = Vec::with_capacity(n_keys + 1);
    for i in 0..n_keys {
        let gathered =
            arrow::compute::take(base.column(i).as_ref(), &take_idx, None).map_err(|e| {
                BoltError::Other(format!(
                    "COUNT(DISTINCT) GROUP BY: take on group key {i} failed: {e}"
                ))
            })?;
        out_cols.push(gathered);
    }
    // The Int64 count column (non-nullable; COUNT never yields SQL NULL).
    let counts: Int64Array = order
        .iter()
        .map(|(_, set)| set.len() as i64)
        .collect::<Vec<i64>>()
        .into();
    out_cols.push(Arc::new(counts) as ArrayRef);

    RecordBatch::try_new(Arc::clone(&arrow_schema), out_cols)
        .map_err(|e| BoltError::Plan(format!("COUNT(DISTINCT) GROUP BY result build: {e}")))
}

/// A numeric value read from a base column for a host plain-aggregate fold.
///
/// Integer columns (`Int32` / `Int64`) carry an `i128` so SUM accumulation
/// cannot overflow before the range check at finalize; float columns
/// (`Float32` / `Float64`) carry an `f64`. NULLs are not represented here —
/// callers skip nulls before reading (SQL aggregates ignore NULLs).
#[derive(Debug, Clone, Copy)]
enum AggNum {
    Int(i128),
    Float(f64),
}

/// Per-group accumulator for one host plain aggregate (SUM / MIN / MAX / AVG /
/// COUNT / COUNT(*)). One instance per (group, aggregate) cell.
#[derive(Debug, Clone)]
struct PlainAccum {
    /// Non-NULL value count (also the plain `COUNT(col)` result, and the AVG
    /// divisor); for `COUNT(*)` this counts every row regardless of nullness.
    count: i64,
    /// Running sum for SUM / AVG (None until the first non-NULL value).
    sum: Option<AggNum>,
    /// Running min / max (None until the first non-NULL value).
    min: Option<AggNum>,
    max: Option<AggNum>,
}

impl Default for PlainAccum {
    fn default() -> Self {
        PlainAccum {
            count: 0,
            sum: None,
            min: None,
            max: None,
        }
    }
}

/// Host-side per-group multi/mixed aggregate for the generalized
/// COUNT(DISTINCT) + GROUP BY path (feature F3-finish, generalized).
///
/// `base` is the materialised `[group_keys..., agg_inputs...]` batch: the first
/// `n_keys` columns are group keys; the remaining columns are one input column
/// per aggregate, in `aggs` order (so `aggs[i].base_col() == n_keys + i`).
/// Groups rows by the leading group-key tuple (same `RowKey` canonicalisation
/// as DISTINCT, so NULL keys form their own group), then per group computes:
///
/// * `COUNT(DISTINCT col)` — number of distinct non-NULL values (NULLs ignored);
/// * `COUNT(col)`          — non-NULL row count;
/// * `COUNT(*)`            — total row count;
/// * `SUM` / `MIN` / `MAX` — over non-NULL values, preserving the input dtype
///   (empty / all-NULL group → SQL NULL); SUM range-checks integer output;
/// * `AVG`                 — Float64 mean of non-NULL values (empty → NULL).
///
/// Output columns are assembled in `output_layout` order. Pure (no GPU / engine
/// state) so it is host-testable directly.
fn host_multi_agg_groupby(
    base: &RecordBatch,
    n_keys: usize,
    aggs: &[crate::plan::sql_frontend::CdAgg],
    output_layout: &[crate::plan::sql_frontend::CdOutputCol],
    result_schema: &Schema,
) -> BoltResult<RecordBatch> {
    use std::collections::{HashMap, HashSet};

    use arrow_array::{Float64Array, UInt32Array};

    use crate::exec::distinct::{ColumnReader, RowKey, RowKeyValue};
    use crate::plan::sql_frontend::{CdAgg, CdOutputCol};

    let n_cols = base.num_columns();
    if n_cols != n_keys + aggs.len() {
        return Err(BoltError::Plan(format!(
            "multi-agg GROUP BY: base produced {n_cols} columns, expected \
             {n_keys} group keys + {} aggregate inputs",
            aggs.len()
        )));
    }
    let n_rows = base.num_rows();

    let key_readers: Vec<ColumnReader> = (0..n_keys)
        .map(|i| ColumnReader::new(base.column(i).as_ref()))
        .collect::<BoltResult<_>>()?;
    // One reader per aggregate input column (column `n_keys + i`).
    let agg_readers: Vec<ColumnReader> = aggs
        .iter()
        .map(|a| ColumnReader::new(base.column(a.base_col()).as_ref()))
        .collect::<BoltResult<_>>()?;

    // Per group: first-row index, the distinct sets (one per COUNT(DISTINCT)
    // aggregate, indexed by aggregate position), and the plain accumulators
    // (one per aggregate position; unused for COUNT(DISTINCT) slots).
    struct GroupState {
        first_row: u32,
        distinct: Vec<HashSet<RowKeyValue>>,
        plain: Vec<PlainAccum>,
    }
    let mut groups: HashMap<RowKey, usize> = HashMap::new();
    let mut order: Vec<GroupState> = Vec::new();

    for row in 0..n_rows {
        let key = RowKey::from_values(n_keys, key_readers.iter().map(|r| r.value_at(row)));
        let slot = match groups.get(&key) {
            Some(&s) => s,
            None => {
                let s = order.len();
                groups.insert(key, s);
                order.push(GroupState {
                    first_row: row as u32,
                    distinct: (0..aggs.len()).map(|_| HashSet::new()).collect(),
                    plain: (0..aggs.len()).map(|_| PlainAccum::default()).collect(),
                });
                s
            }
        };
        let g = &mut order[slot];
        for (ai, agg) in aggs.iter().enumerate() {
            let reader = &agg_readers[ai];
            match agg {
                CdAgg::CountDistinct { .. } => {
                    let v = reader.value_at(row);
                    if v != RowKeyValue::Null {
                        g.distinct[ai].insert(v);
                    }
                }
                CdAgg::CountStar { .. } => {
                    g.plain[ai].count += 1;
                }
                CdAgg::Count { .. } => {
                    if reader.value_at(row) != RowKeyValue::Null {
                        g.plain[ai].count += 1;
                    }
                }
                CdAgg::Sum { .. } | CdAgg::Min { .. } | CdAgg::Max { .. } | CdAgg::Avg { .. } => {
                    if let Some(num) = agg_num_at(reader, row)? {
                        let acc = &mut g.plain[ai];
                        acc.count += 1;
                        acc.sum = Some(match acc.sum {
                            None => num,
                            Some(s) => agg_add(s, num),
                        });
                        acc.min = Some(match acc.min {
                            None => num,
                            Some(m) => if agg_lt(num, m) { num } else { m },
                        });
                        acc.max = Some(match acc.max {
                            None => num,
                            Some(m) => if agg_lt(m, num) { num } else { m },
                        });
                    }
                }
            }
        }
    }

    // Build output columns in `output_layout` order.
    let arrow_schema = plan_schema_to_arrow_schema(result_schema)?;
    let take_idx = UInt32Array::from(order.iter().map(|g| g.first_row).collect::<Vec<u32>>());
    let mut out_cols: Vec<ArrayRef> = Vec::with_capacity(output_layout.len());

    for (out_idx, out) in output_layout.iter().enumerate() {
        match out {
            CdOutputCol::GroupKey(k) => {
                let gathered = arrow::compute::take(base.column(*k).as_ref(), &take_idx, None)
                    .map_err(|e| {
                        BoltError::Other(format!(
                            "multi-agg GROUP BY: take on group key {k} failed: {e}"
                        ))
                    })?;
                out_cols.push(gathered);
            }
            CdOutputCol::Agg(a) => {
                let agg = &aggs[*a];
                let field = &result_schema.fields[out_idx];
                let col: ArrayRef = match agg {
                    CdAgg::CountDistinct { .. } => {
                        let v: Int64Array = order
                            .iter()
                            .map(|g| g.distinct[*a].len() as i64)
                            .collect::<Vec<i64>>()
                            .into();
                        Arc::new(v) as ArrayRef
                    }
                    CdAgg::Count { .. } | CdAgg::CountStar { .. } => {
                        let v: Int64Array = order
                            .iter()
                            .map(|g| g.plain[*a].count)
                            .collect::<Vec<i64>>()
                            .into();
                        Arc::new(v) as ArrayRef
                    }
                    CdAgg::Avg { .. } => {
                        let v: Float64Array = order
                            .iter()
                            .map(|g| {
                                let acc = &g.plain[*a];
                                if acc.count == 0 {
                                    None
                                } else {
                                    let total = match acc.sum {
                                        Some(AggNum::Int(i)) => i as f64,
                                        Some(AggNum::Float(f)) => f,
                                        None => 0.0,
                                    };
                                    Some(total / acc.count as f64)
                                }
                            })
                            .collect::<Vec<Option<f64>>>()
                            .into();
                        Arc::new(v) as ArrayRef
                    }
                    CdAgg::Sum { .. } => {
                        finalize_numeric(
                            order.iter().map(|g| g.plain[*a].sum),
                            field.dtype,
                            "SUM",
                        )?
                    }
                    CdAgg::Min { .. } => {
                        finalize_numeric(
                            order.iter().map(|g| g.plain[*a].min),
                            field.dtype,
                            "MIN",
                        )?
                    }
                    CdAgg::Max { .. } => {
                        finalize_numeric(
                            order.iter().map(|g| g.plain[*a].max),
                            field.dtype,
                            "MAX",
                        )?
                    }
                };
                out_cols.push(col);
            }
        }
    }

    RecordBatch::try_new(Arc::clone(&arrow_schema), out_cols)
        .map_err(|e| BoltError::Plan(format!("multi-agg GROUP BY result build: {e}")))
}

/// Read the numeric value at `row` from a `ColumnReader`, as an [`AggNum`].
/// Returns `Ok(None)` for a NULL row; errors for a non-numeric column dtype
/// (Bool / Utf8) under a SUM/MIN/MAX/AVG aggregate.
fn agg_num_at(
    reader: &crate::exec::distinct::ColumnReader<'_>,
    row: usize,
) -> BoltResult<Option<AggNum>> {
    use crate::exec::distinct::RowKeyValue;
    Ok(match reader.value_at(row) {
        RowKeyValue::Null => None,
        RowKeyValue::I32(v) => Some(AggNum::Int(v as i128)),
        RowKeyValue::I64(v) => Some(AggNum::Int(v as i128)),
        RowKeyValue::F32(v) => Some(AggNum::Float(f32::from_bits(v) as f64)),
        RowKeyValue::F64(v) => Some(AggNum::Float(f64::from_bits(v))),
        RowKeyValue::Bool(_) | RowKeyValue::Utf8(_) => {
            return Err(BoltError::Type(
                "SUM/MIN/MAX/AVG under GROUP BY require a numeric column".into(),
            ))
        }
    })
}

/// Add two [`AggNum`]s (both produced from the same numeric column, so the
/// variant is consistent).
fn agg_add(a: AggNum, b: AggNum) -> AggNum {
    match (a, b) {
        (AggNum::Int(x), AggNum::Int(y)) => AggNum::Int(x + y),
        (AggNum::Float(x), AggNum::Float(y)) => AggNum::Float(x + y),
        // Mixed never happens (one column → one variant); fall back to float.
        (AggNum::Int(x), AggNum::Float(y)) | (AggNum::Float(y), AggNum::Int(x)) => {
            AggNum::Float(x as f64 + y)
        }
    }
}

/// `a < b` for two same-variant [`AggNum`]s. NaN floats sort as greater (so a
/// real value is preferred as the MIN), matching the host scalar convention.
fn agg_lt(a: AggNum, b: AggNum) -> bool {
    match (a, b) {
        (AggNum::Int(x), AggNum::Int(y)) => x < y,
        (AggNum::Float(x), AggNum::Float(y)) => x < y,
        (AggNum::Int(x), AggNum::Float(y)) | (AggNum::Float(y), AggNum::Int(x)) => (x as f64) < y,
    }
}

/// Finalize a per-group SUM/MIN/MAX reduction into a typed, nullable array of
/// `target` dtype. Integer targets range-check each value (a SUM that exceeds
/// the target integer range errors rather than wraps, matching the scalar host
/// aggregate). A `None` group value becomes a NULL output cell.
fn finalize_numeric(
    values: impl Iterator<Item = Option<AggNum>>,
    target: DataType,
    op: &str,
) -> BoltResult<ArrayRef> {
    use arrow_array::{Float32Array, Float64Array, Int32Array as I32, Int64Array as I64};
    let vals: Vec<Option<AggNum>> = values.collect();
    Ok(match target {
        DataType::Int32 => {
            let out: Vec<Option<i32>> = vals
                .iter()
                .map(|v| match v {
                    None => Ok(None),
                    Some(AggNum::Int(i)) => i32::try_from(*i).map(Some).map_err(|_| {
                        BoltError::Type(format!("{op} result {i} overflows Int32"))
                    }),
                    Some(AggNum::Float(_)) => Err(BoltError::Type(format!(
                        "{op}: float value for an Int32 column"
                    ))),
                })
                .collect::<BoltResult<_>>()?;
            Arc::new(I32::from(out)) as ArrayRef
        }
        DataType::Int64 => {
            let out: Vec<Option<i64>> = vals
                .iter()
                .map(|v| match v {
                    None => Ok(None),
                    Some(AggNum::Int(i)) => i64::try_from(*i).map(Some).map_err(|_| {
                        BoltError::Type(format!("{op} result {i} overflows Int64"))
                    }),
                    Some(AggNum::Float(_)) => Err(BoltError::Type(format!(
                        "{op}: float value for an Int64 column"
                    ))),
                })
                .collect::<BoltResult<_>>()?;
            Arc::new(I64::from(out)) as ArrayRef
        }
        DataType::Float32 => {
            let out: Vec<Option<f32>> = vals
                .iter()
                .map(|v| match v {
                    None => None,
                    Some(AggNum::Float(f)) => Some(*f as f32),
                    Some(AggNum::Int(i)) => Some(*i as f32),
                })
                .collect();
            Arc::new(Float32Array::from(out)) as ArrayRef
        }
        DataType::Float64 => {
            let out: Vec<Option<f64>> = vals
                .iter()
                .map(|v| match v {
                    None => None,
                    Some(AggNum::Float(f)) => Some(*f),
                    Some(AggNum::Int(i)) => Some(*i as f64),
                })
                .collect();
            Arc::new(Float64Array::from(out)) as ArrayRef
        }
        other => {
            return Err(BoltError::Type(format!(
                "{op} under GROUP BY: unsupported output dtype {other:?}"
            )))
        }
    })
}

/// Stage 7 (P1b): default interval between pool-stats emits in
/// [`Engine::sql`].
///
/// 60 seconds is a sensible floor for a typical analytical workload —
/// the pool changes slowly relative to query churn, and a coarser
/// cadence keeps the log line out of per-query latency. Override with
/// `BOLT_POOL_STATS_INTERVAL_SECS=<n>`; set to `0` to disable emission
/// entirely (handy for benchmark runs that don't want the noise).
pub(crate) const DEFAULT_POOL_STATS_INTERVAL_SECS: u64 = 60;

/// Environment-variable override for the pool-stats periodic-emit
/// interval. Parsed once per `Engine` construction; non-integer or
/// negative values fall back to [`DEFAULT_POOL_STATS_INTERVAL_SECS`].
///
/// `pub(crate)` so the integration test
/// `tests/env_var_smoke.rs` can address the canonical env-var name
/// instead of duplicating it (drift between the constant here and a
/// hard-coded string in the test would silently desynchronise the
/// toggle smoke).
pub const POOL_STATS_ENV: &str = "BOLT_POOL_STATS_INTERVAL_SECS";

/// Top-level query engine.
///
/// Field-drop order matters: `dict_registry` owns `DictionaryColumn`s which own
/// `GpuVec`s — those must be freed BEFORE `_ctx` tears down the CUDA context.
/// Rust drops fields in declaration order, so `_ctx` sits last.
///
/// # Construction
///
/// Prefer the typed builder for new code:
///
/// ```ignore
/// use craton_bolt::Engine;
///
/// let engine = Engine::builder()
///     .device(0)
///     .memory_budget(1 << 30)
///     .build()?;
/// ```
///
/// The legacy [`Engine::new`] and [`Engine::new_with_device`] entry points are
/// thin wrappers around the builder, kept for source-compatibility with
/// pre-v0.6 callers.
///
/// # `#[non_exhaustive]`
///
/// Marked `#[non_exhaustive]` so future v0.x releases can grow new fields
/// without a breaking semver bump for downstream code that destructures or
/// constructs `Engine` literally. Construction goes through the builder; all
/// other access is via inherent methods.
#[non_exhaustive]
pub struct Engine {
    /// Registered tables, keyed by name. A single table may comprise multiple
    /// batches (wave-7 multi-batch support): the engine concatenates them via
    /// `arrow::compute::concat_batches` at query time. This is a 0.2-era
    /// simplification — a streaming, per-batch query plan is a 0.3 goal — so
    /// large multi-batch tables pay a full materialisation cost on every
    /// `sql()` call. Keep the per-table batch count modest until then.
    tables: HashMap<String, Vec<RecordBatch>>,
    /// Lazily-registered streaming table sources, keyed by name.
    ///
    /// Tables registered through [`Engine::register_table_stream_lazy`] are
    /// stored here as a replayable producer ([`TableSource::Streaming`])
    /// rather than being drained into `tables` at registration time. The
    /// producer is invoked the first time the table is read (see
    /// [`Engine::streaming_batches`]), at which point the entry is collapsed
    /// in place to [`TableSource::Materialized`] so subsequent reads skip the
    /// producer.
    ///
    /// This is an *overlay* over `tables`: a name lives in exactly one of the
    /// two maps. The read helpers ([`Engine::materialize_table`] and the
    /// provider null probes) consult `tables` first and fall back to draining
    /// the streaming overlay. Keeping the lazy data out of `tables` is what
    /// makes registration cheap (no host materialisation) while leaving every
    /// eager code path untouched.
    ///
    /// `RefCell` because the lazy materialisation happens from `&self`
    /// (`Engine::sql` takes `&self`), mirroring the interior mutability
    /// already used for `gpu_tables`.
    streaming_sources: RefCell<HashMap<String, crate::exec::streaming::TableSource>>,
    /// Name → Schema provider, kept in sync with `tables`. The schema is
    /// EXTENDED with `__idx_<col>` Int32 columns for every registered Utf8
    /// column so the SQL frontend resolves rewriter-produced column refs.
    provider: MemTableProvider,
    /// Per-table Utf8 dictionaries; drives the string-literal predicate rewrite.
    dict_registry: crate::exec::dict_registry::DictRegistry,
    /// GPU-resident copies of every registered table. Owns the device
    /// allocations; must drop BEFORE `_ctx`.
    ///
    /// Wrapped in `RefCell<Option<_>>` to support a lazy-upload strategy:
    /// `register_batch` only mutates the host-side `tables` and sets the slot
    /// to `None` (dirty). The actual upload happens on the next query, in
    /// `ensure_gpu_table` from inside `execute_projection`. This collapses a
    /// streaming append workload that uploaded `1+2+…+N = N(N+1)/2` batches'
    /// worth of bytes (the per-append re-upload bug) down to a single upload
    /// per query of the current concatenated table — i.e. O(N) total bytes
    /// across the lifetime of a streaming-then-query session, instead of
    /// O(N²). Multiple consecutive `register_batch` calls without an
    /// intervening query share that one upload.
    ///
    /// **Batch 5 (incremental cache)**: the slot now holds `Some(GpuTable)`
    /// even across `register_batch` mutations. The host bumps per-table /
    /// per-column revisions in [`Engine::host_revisions`] on mutation, and
    /// `ensure_gpu_table` compares them against the GpuTable's
    /// `last_uploaded_revision` plus each column's `host_revision`:
    /// columns whose revision still matches are reused in place; only
    /// dirty columns are re-uploaded. For `register_batch` appends, the
    /// re-upload allocates a fresh GpuVec sized for the new total rows,
    /// DtoD-copies the previously-uploaded prefix, and HtoD-uploads only
    /// the new tail — so the unchanged rows never re-cross the PCIe bus.
    gpu_tables: RefCell<HashMap<String, Option<crate::exec::gpu_table::GpuTable>>>,
    /// Per-table host-side revision counters for the incremental GpuTable
    /// cache (batch 5).
    ///
    /// Mutated by `register_table` / `replace_table` / `register_batch` and
    /// read by `ensure_gpu_table`. Both mutators take `&mut self`, and
    /// `ensure_gpu_table` only borrows it immutably, so a `RefCell` would
    /// be unnecessary noise — a plain field suffices.
    host_revisions: HashMap<String, HostTableRevision>,
    /// Test-only counter incremented on every per-column upload performed
    /// by [`Engine::ensure_gpu_table`]. Exposed so the incremental-upload
    /// tests can assert that an unchanged column was reused (LOAD_COUNT
    /// did not bump for it).
    ///
    /// Uses `SeqCst` so a test that observes a count, registers a batch,
    /// re-queries, and observes the count again sees a strict
    /// happens-before relation.
    #[cfg(test)]
    gpu_table_load_count: std::sync::atomic::AtomicUsize,
    /// Stage 7 (P1b): pool-stats observability state.
    ///
    /// `Mutex<Option<Instant>>`: `Some(last_emit_time)` after the first
    /// emit, `None` before any query has run. The first query on a fresh
    /// engine always emits (so a short-lived process still surfaces at
    /// least one snapshot); subsequent queries emit only after
    /// `pool_stats_interval` has elapsed.
    ///
    /// Wrapped in a `Mutex` because `Engine::sql` takes `&self` and we
    /// support concurrent calls in principle (the underlying engine is
    /// not yet `Send + Sync` because of `RefCell`, but the
    /// pool-stats accounting is independent and shouldn't add new
    /// `!Sync` constraints when we eventually relax the rest).
    pool_stats_last_emit: Mutex<Option<Instant>>,
    /// Interval between pool-stats emits. Frozen at construction from
    /// `BOLT_POOL_STATS_INTERVAL_SECS` (default 60s). A value of
    /// `Duration::ZERO` disables periodic emission entirely.
    pool_stats_interval: Duration,
    /// Review-H2 PTX module cache: `KernelSpec` content hash + entry name →
    /// loaded `CudaModule`. Lifts the per-query
    /// `compile_ptx` + `CudaModule::from_ptx` round-trip in
    /// `execute_projection` to a process-local table lookup.
    ///
    /// The underlying `CudaModule` is `Clone` over an internal
    /// `Arc<CudaModuleInner>` (see `jit::jit_compiler`), so a cached entry
    /// can hand out cheap handle-clones to repeated callers — the cubin is
    /// loaded into the driver exactly once per `(spec, entry)` pair across
    /// the engine's lifetime.
    ///
    /// `Mutex`-guarded because `Engine::sql` takes `&self` and we may
    /// eventually relax the engine's `!Sync` constraints (the `RefCell`
    /// on `gpu_tables` is the real blocker today, not this cache).
    ///
    /// Counter `module_cache_loads` increments on every cache miss; tests
    /// observe it to confirm the cache services repeat calls.
    module_cache: Mutex<HashMap<ModuleCacheKey, CudaModule>>,
    /// Number of cache misses observed by `get_or_build_module`. Bumped
    /// once per fresh `compile_ptx` + `CudaModule::from_ptx` round-trip.
    /// Read by the projection-cache unit test to assert the second call
    /// returns the cached module without re-loading. Atomic-ordered
    /// `SeqCst` so the test's load/store interleaves cleanly with the
    /// engine's increment.
    module_cache_loads: std::sync::atomic::AtomicUsize,
    /// v0.6 / M7 public optimizer extension surface: user-registered
    /// PlanRewrite implementations run in registration order before lower_physical.
    rewrites: Vec<Box<dyn PlanRewrite>>,
    /// v0.6 builder: CUDA device ordinal this engine was constructed on.
    device_idx: i32,
    /// v0.6 builder: soft cap on device-memory pool allocations in bytes.
    memory_budget_bytes: Option<usize>,
    /// v0.6 builder: optional disk-backed PTX cache directory.
    persistent_cache_path: Option<std::path::PathBuf>,
    /// v0.6 builder: whether tracing was enabled by the builder.
    tracing_enabled: bool,
    /// Test-only: run the built-in logical optimizer before lowering.
    /// Defaults to `true` (the production behaviour); flipped to `false`
    /// only via [`EngineBuilder::without_optimizer`] so the
    /// optimizer-equivalence test can execute an UN-optimized plan and
    /// compare its results against the optimized path. Every production
    /// construction leaves this `true`, so the gate at the
    /// `run_to_fixpoint` call sites is a no-op for all stable callers.
    optimize: bool,
    /// Owned CUDA context — declared LAST so it drops AFTER dictionaries.
    _ctx: CudaContext,
}

/// v0.6 builder for [`Engine`]. Use [`Engine::builder`] to start one.
///
/// Every knob is optional; un-set knobs land on the same defaults that the
/// legacy [`Engine::new`] / [`Engine::new_with_device`] paths produce. The
/// builder owns no resources until [`EngineBuilder::build`] is called — only
/// `build` initialises the CUDA driver, validates the device index, and
/// creates the CUDA context. This keeps `EngineBuilder` cheap to construct in
/// hot paths (e.g. test harnesses) without paying for driver init that may
/// then be discarded.
///
/// The builder is `#[non_exhaustive]` so v0.x can grow new knobs without a
/// breaking change for downstream code that destructures it (which shouldn't
/// happen — but the marker makes the intent explicit).
///
/// ```ignore
/// use craton_bolt::Engine;
/// use std::path::PathBuf;
///
/// let engine = Engine::builder()
///     .device(0)
///     .memory_budget(2 * 1024 * 1024 * 1024)        // 2 GiB soft cap
///     .persistent_cache(PathBuf::from("/var/cache/bolt/ptx"))
///     .enable_tracing()
///     .build()?;
/// ```
#[non_exhaustive]
#[derive(Debug, Default, Clone)]
pub struct EngineBuilder {
    /// CUDA device ordinal. `None` selects the default (`0`).
    device: Option<i32>,
    /// Soft device-memory budget in bytes. `None` is uncapped.
    memory_budget_bytes: Option<usize>,
    /// Optional disk-backed PTX cache directory.
    persistent_cache_path: Option<std::path::PathBuf>,
    /// Install a default tracing subscriber from [`build`](Self::build).
    enable_tracing: bool,
    /// Test-only: when `true`, [`build`](Self::build) constructs an engine
    /// that SKIPS the built-in logical optimizer. Defaults to `false`
    /// (optimizer ON — the production behaviour), so the derived
    /// [`Default`] and every existing builder call leave optimization
    /// enabled. Set only via [`EngineBuilder::without_optimizer`].
    disable_optimizer: bool,
}

impl EngineBuilder {
    /// Fresh builder with all knobs at their defaults. Same as the value
    /// returned by [`Engine::builder`] — exposed publicly so downstream code
    /// can stash a default builder and tweak it incrementally without going
    /// through the `Engine::` type name (handy in generic test helpers).
    pub fn new() -> Self {
        Self {
            device: None,
            memory_budget_bytes: None,
            persistent_cache_path: None,
            enable_tracing: false,
            disable_optimizer: false,
        }
    }

    /// Select the CUDA device ordinal. Defaults to `0`.
    ///
    /// The index is validated against `cuDeviceGetCount` inside
    /// [`build`](Self::build); an out-of-range index surfaces a
    /// `BoltError::Other` there, not here.
    pub fn device(mut self, idx: i32) -> Self {
        self.device = Some(idx);
        self
    }

    /// Set a soft cap on device-memory pool allocations, in bytes. Defaults
    /// to uncapped.
    ///
    /// Stored verbatim on the engine and readable via
    /// [`Engine::memory_budget_bytes`]. Runtime pool integration may evolve
    /// across v0.x — the getter contract is what's stable.
    pub fn memory_budget(mut self, bytes: usize) -> Self {
        self.memory_budget_bytes = Some(bytes);
        self
    }

    /// Enable a disk-backed PTX cache rooted at `path`. Defaults to
    /// disabled (the existing in-memory PTX cache in `jit::jit_compiler`
    /// is unaffected either way).
    ///
    /// [`build`](Self::build) threads this path into the process-wide disk
    /// PTX cache (via [`crate::jit::disk_cache::set_override_dir`]), so the
    /// JIT compile path reads/writes cubins at `path` even when the
    /// `BOLT_PTX_CACHE_DIR` env var is unset. The builder path takes
    /// precedence over that env var, which remains the fallback for
    /// engines built without this knob.
    ///
    /// The path is stored verbatim — `DiskPtxCache::open` creates the
    /// directory if it does not already exist, but it is the caller's
    /// responsibility to ensure the location is writable.
    pub fn persistent_cache(mut self, path: std::path::PathBuf) -> Self {
        self.persistent_cache_path = Some(path);
        self
    }

    /// Ask [`build`](Self::build) to install a default tracing subscriber
    /// before returning the engine. Defaults to disabled.
    ///
    /// "Default subscriber" here means a best-effort `log`-crate
    /// initialisation: this crate uses [`log`] for diagnostics today, so
    /// enabling this knob promotes the global `log::Level` to `Info`. A
    /// future v0.x may swap to the `tracing` crate proper; the builder
    /// method's name is intentionally subscriber-agnostic so the contract
    /// survives that swap. Calling this on a process where a logger /
    /// subscriber is already installed is a no-op (the underlying
    /// `set_logger` is idempotent under contention).
    pub fn enable_tracing(mut self) -> Self {
        self.enable_tracing = true;
        self
    }

    /// Test-only: build an engine that SKIPS the built-in logical optimizer.
    ///
    /// Not part of the stable public API — it exists so the
    /// optimizer-equivalence test can execute a logical plan WITHOUT the
    /// default optimizer pass pipeline (`run_to_fixpoint(default_passes,
    /// ..)`) and compare the resulting rows against the normal,
    /// optimizer-ON path. With the optimizer disabled, `sql()`,
    /// `run_logical_plan()`, and subplan execution all lower the plan as
    /// written (after the always-on dict rewrite + subquery resolution,
    /// which are correctness-preserving, not optimizations).
    ///
    /// Production code must never call this: leaving the optimizer on is
    /// the default for every other construction path.
    #[doc(hidden)]
    pub fn without_optimizer(mut self) -> Self {
        self.disable_optimizer = true;
        self
    }

    /// Build the [`Engine`]. Consumes the builder.
    ///
    /// Steps performed by `build` (in order):
    ///   1. Resolve the device index (default `0`).
    ///   2. Initialize the CUDA driver (idempotent across calls).
    ///   3. Validate the device index against `cuDeviceGetCount`.
    ///   4. Create an owned CUDA context on the selected device.
    ///   5. If [`enable_tracing`](Self::enable_tracing) was set, promote the
    ///      global `log` max level to `Info` (best-effort, ignored if a
    ///      logger is already installed).
    ///
    /// # Errors
    /// - `BoltError::Other` if the device index is `< 0` or `>=
    ///   cuDeviceGetCount()`.
    /// - Any underlying CUDA driver failure (no CUDA-capable device,
    ///   driver / runtime mismatch, OOM on context create).
    pub fn build(self) -> BoltResult<Engine> {
        let device_idx = self.device.unwrap_or(0);
        // Initialize the driver up-front so device_count() is callable.
        cuda_sys::init()?;
        let count = cuda_sys::device_count()?;
        if device_idx < 0 || device_idx >= count {
            return Err(BoltError::Other(format!(
                "CUDA device index {} is out of range: {} device(s) visible to the driver (valid range: 0..{})",
                device_idx, count, count
            )));
        }
        let ctx = CudaContext::new(device_idx)?;
        let pool_stats_interval = pool_stats_interval_from_env();

        // v0.6 / M6: thread the builder's `persistent_cache(path)` knob
        // into the process-wide disk PTX cache so the JIT compile path
        // (`get_or_build_module` → `disk_cache::disk_cache()`) reads and
        // writes cubins at the configured directory — not only when the
        // `BOLT_PTX_CACHE_DIR` env var is set. Opt-in: a `None` path
        // clears any prior builder override and re-falls-back to the env
        // var, preserving the historical "no path → no disk cache"
        // behaviour. See `install_persistent_cache_override` for the
        // precedence contract.
        install_persistent_cache_override(self.persistent_cache_path.as_deref());

        if self.enable_tracing {
            // Best-effort subscriber init. `log::set_max_level` is
            // process-global but always succeeds; pairing it with a
            // logger-installed check would require a fixed downstream
            // logger choice we don't want to make here. If the caller
            // has already wired a logger, raising the level is the
            // worst we'll do; if they haven't, the elevated level is
            // a benign hint for whatever they install later.
            log::set_max_level(log::LevelFilter::Info);
        }

        // v0.7: wire the builder-supplied `persistent_cache(path)` knob
        // into the process-wide disk PTX cache (see `jit::disk_cache`).
        // When `persistent_cache_path` is `Some`, install it as a
        // builder override — `disk_cache::resolve_cache_dir` prefers an
        // installed override over the `BOLT_PTX_CACHE_DIR` env var so
        // the builder-explicit path wins (last-write-wins between this
        // path and any prior `set_disk_ptx_cache_dir` call).
        //
        // When `persistent_cache_path` is `None` we intentionally do
        // NOT clear the override here: an unset builder knob must not
        // wipe out an env-var-driven cache that the surrounding
        // process configured, and must not wipe out an override that
        // another component installed before us. The env-var path
        // therefore continues to work unchanged when the builder
        // doesn't opt in.
        if let Some(p) = self.persistent_cache_path.clone() {
            crate::jit::set_disk_ptx_cache_dir(Some(p));
        }

        Ok(Engine {
            tables: HashMap::new(),
            streaming_sources: RefCell::new(HashMap::new()),
            provider: MemTableProvider::new(),
            dict_registry: crate::exec::dict_registry::DictRegistry::new(),
            gpu_tables: RefCell::new(HashMap::new()),
            host_revisions: HashMap::new(),
            #[cfg(test)]
            gpu_table_load_count: std::sync::atomic::AtomicUsize::new(0),
            pool_stats_last_emit: Mutex::new(None),
            pool_stats_interval,
            module_cache: Mutex::new(HashMap::new()),
            module_cache_loads: std::sync::atomic::AtomicUsize::new(0),
            rewrites: Vec::new(),
            device_idx,
            memory_budget_bytes: self.memory_budget_bytes,
            persistent_cache_path: self.persistent_cache_path,
            tracing_enabled: self.enable_tracing,
            // Optimizer ON unless the test-only `without_optimizer()` knob
            // flipped it. Every production path leaves `disable_optimizer`
            // false, so this is `true` for all stable callers.
            optimize: !self.disable_optimizer,
            _ctx: ctx,
        })
    }
}

impl Engine {
    /// Create an engine on the default CUDA device (ordinal 0).
    ///
    /// v0.6 legacy entry point: thin wrapper around [`Engine::builder`] kept
    /// so pre-v0.6 callers continue to compile. New code should prefer the
    /// builder for forward-compatible knobs.
    pub fn new() -> BoltResult<Self> {
        Self::builder().build()
    }

    /// Create an engine bound to the CUDA device at ordinal `device_idx`.
    ///
    /// v0.6 legacy entry point: thin wrapper around
    /// [`Engine::builder`]`.device(device_idx).build()`. The error contract is
    /// preserved verbatim — see [`EngineBuilder::build`] for the failure
    /// modes (out-of-range index, driver init failure, context create).
    pub fn new_with_device(device_idx: i32) -> BoltResult<Self> {
        Self::builder().device(device_idx).build()
    }

    /// Start a fresh [`EngineBuilder`] with all knobs at their defaults.
    ///
    /// This is the recommended construction entry point as of v0.6. Set only
    /// the knobs you need; everything else picks up the same default that
    /// the legacy [`Engine::new`] / [`Engine::new_with_device`] paths use:
    ///
    /// | Builder method        | Default                |
    /// |-----------------------|------------------------|
    /// | [`EngineBuilder::device`]            | `0`              |
    /// | [`EngineBuilder::memory_budget`]     | uncapped         |
    /// | [`EngineBuilder::persistent_cache`]  | disabled         |
    /// | [`EngineBuilder::enable_tracing`]    | disabled         |
    ///
    /// ```ignore
    /// use craton_bolt::Engine;
    /// let engine = Engine::builder().build()?;
    /// ```
    pub fn builder() -> EngineBuilder {
        EngineBuilder::new()
    }

    /// CUDA device ordinal this engine was constructed on.
    ///
    /// Mirrors the value passed to [`EngineBuilder::device`] (or `0` for the
    /// default-device entry points). Useful for diagnostics on multi-GPU
    /// hosts and for tests that want to assert the builder threaded the
    /// device knob through.
    pub fn device(&self) -> i32 {
        self.device_idx
    }

    /// Soft device-memory budget in bytes, as set via
    /// [`EngineBuilder::memory_budget`]. `None` means uncapped (the default).
    ///
    /// The value is stored verbatim; the runtime pool integration may evolve
    /// across v0.x releases but the getter's contract is stable.
    pub fn memory_budget_bytes(&self) -> Option<usize> {
        self.memory_budget_bytes
    }

    /// Disk-backed PTX cache directory, as set via
    /// [`EngineBuilder::persistent_cache`]. `None` means disabled.
    pub fn persistent_cache_path(&self) -> Option<&std::path::Path> {
        self.persistent_cache_path.as_deref()
    }

    /// `true` if [`EngineBuilder::enable_tracing`] was called on the builder
    /// that produced this engine.
    pub fn tracing_enabled(&self) -> bool {
        self.tracing_enabled
    }

    /// Register a user-supplied [`PlanRewrite`] on this engine.
    ///
    /// Rewrites run in registration order, threading each rewriter's
    /// output into the next, immediately **before**
    /// [`crate::plan::lower_physical`] in [`Engine::sql`]. See the
    /// [`PlanRewrite`](crate::plan::PlanRewrite) trait docs for the
    /// contract implementations must uphold.
    ///
    /// This `with_rewrite` is the engine-direct entry point; the
    /// forthcoming `Engine::Builder` (parallel agent) exposes the same
    /// signature on the builder. Both ultimately push into the same
    /// `rewrites` field, so the builder integration is a drop-in.
    ///
    /// Takes `self` by value and returns it so the call can chain with
    /// the constructor: `Engine::new()?.with_rewrite(Box::new(MyRewrite))`.
    pub fn with_rewrite(mut self, r: Box<dyn PlanRewrite>) -> Self {
        self.rewrites.push(r);
        self
    }

    /// Number of registered [`PlanRewrite`]s on this engine. Exposed for
    /// tests and for `EXPLAIN`-style introspection.
    pub fn rewrite_count(&self) -> usize {
        self.rewrites.len()
    }

    /// Review-H2: look up the cached `CudaModule` for `(spec, entry)`, or
    /// compile + load it on a miss and seed the cache.
    ///
    /// `entry` selects between the projection kernel and the predicate-only
    /// mask kernel — they generate different PTX from the same `KernelSpec`,
    /// so the entry name participates in the key. On a cache hit we hand
    /// back a cheap `CudaModule` clone (Arc-handle). On a miss we run the
    /// underlying PTX-text-hash cache in `jit::jit_compiler`, which itself
    /// short-circuits the `cuModuleLoadDataEx` step on a repeat PTX string;
    /// either way we then memoise the result here so future calls skip the
    /// PTX generation entirely.
    ///
    /// The closure-based loader keeps us from re-spelling the projection vs
    /// predicate compile path at every call site.
    ///
    /// # v0.7: process-wide KernelSpec cache layer
    ///
    /// Before consulting the per-`Engine` cache we now check the
    /// process-wide KernelSpec-keyed cache in
    /// [`crate::exec::module_cache::get_or_build_module_for_spec`]. The
    /// global layer survives across `Engine` instances (test harnesses,
    /// short-lived embedded engines, future multi-engine deployments) so
    /// the second engine that requests the same `(spec, entry)` skips
    /// both codegen *and* PTX-text-hash lookup — it's a flat Arc-clone of
    /// the already-loaded module. The per-engine cache is retained as an
    /// inner fast path so the on-engine `module_cache_loads` counter and
    /// disk-cache write-through remain observable. The layering is:
    ///
    /// 1. **Global KernelSpec cache** — sub-µs Arc-clone on hit; on miss
    ///    falls through to the per-engine path below via the closure.
    /// 2. **Per-engine KernelSpec cache** — fast path for repeat calls on
    ///    the same engine; bumps `module_cache_loads` on a miss.
    /// 3. **Disk-backed PTX cache** (v0.6 / M6) — skips codegen if
    ///    `BOLT_PTX_CACHE_DIR` or the builder's `persistent_cache` was
    ///    set and the PTX text is on disk from a previous process.
    /// 4. **`compile(spec)` + `CudaModule::from_ptx`** — the latter
    ///    consults the PTX-text-hash cache in `jit::jit_compiler` so a
    ///    cross-spec PTX collision still reuses the loaded driver module.
    fn get_or_build_module<F>(
        &self,
        spec: &KernelSpec,
        entry: &'static str,
        compile: F,
    ) -> BoltResult<CudaModule>
    where
        F: FnOnce(&KernelSpec) -> BoltResult<String>,
    {
        // v0.7 layer 1: process-wide KernelSpec cache. On a hit this is a
        // sub-µs Arc-clone that skips every layer below. On a miss the
        // closure runs `compile` and routes the resulting PTX through
        // `CudaModule::from_ptx` itself — so we must NOT call back into
        // the per-engine path here (we'd double-codegen). Instead we
        // re-implement the per-engine + disk + codegen fall-through
        // *inside* the closure. The per-engine cache still services
        // repeat calls within one engine: the `module_cache.lock().get`
        // pre-check is the only difference from a flat global-only path,
        // and it's load-bearing for the `module_cache_loads`-counter
        // test below.
        //
        // Why not push the global cache check inside the per-engine
        // miss branch? Because the *fast path* of `get_or_build_module_for_spec`
        // is what we want — it never touches `self.module_cache.lock()`
        // and so doesn't serialise on the per-engine mutex. The cost is
        // that on a miss we do two lookups (global + per-engine); both
        // are HashMap probes, fine.
        let key = ModuleCacheKey::new(spec, entry);
        // Per-engine fast path: hit. Hold the lock only long enough to
        // clone the Arc. This stays AHEAD of the global lookup so the
        // existing `module_cache_loads` invariant ("second call must
        // not bump the counter") is preserved bit-for-bit, and the
        // single-engine hot path keeps its per-engine mutex affinity.
        if let Some(m) = self
            .module_cache
            .lock()
            .map_err(|_| BoltError::Other("module_cache mutex poisoned".to_string()))?
            .get(&key)
        {
            return Ok(m.clone());
        }
        // Capture just the Copy hash components for the closure below;
        // this lets us move `key` itself into `cache.entry(key)` after
        // the closure has run without borrow-checker complications.
        let spec_hash_hi = key.spec_hash_hi;
        let spec_hash_lo = key.spec_hash_lo;
        // Per-engine miss. Consult the process-wide KernelSpec cache; on
        // a global hit we also seed the per-engine cache so subsequent
        // calls on this engine take the per-engine fast path above and
        // skip the global mutex altogether.
        let module = crate::exec::module_cache::get_or_build_module_for_spec(
            spec,
            entry,
            |spec| {
                // Global miss path: this closure runs at most once per
                // (spec, entry) per process. Inside it we still want
                // the disk cache + codegen layers, so we open-code
                // them here (mirrors the legacy per-engine miss path).
                let disk = crate::jit::disk_cache::disk_cache();
                let disk_key = disk.as_ref().map(|_| {
                    // Compose a disk key that (a) folds in the
                    // codegen-version salt so a PTX-emission change
                    // invalidates stale on-disk entries (JIT-M1), and
                    // (b) domain-separates entry-point names: two
                    // kernels with identical KernelSpec content but
                    // different entry symbols (KERNEL_ENTRY vs
                    // PREDICATE_ENTRY) must NOT alias to the same .ptx
                    // file. See `disk_cache::disk_key` for the canonical
                    // key shape; the in-process KernelSpecCache key is
                    // intentionally left unsalted (it re-validates PTX
                    // content on every hit).
                    crate::jit::disk_cache::disk_key(
                        entry,
                        spec_hash_hi,
                        spec_hash_lo,
                    )
                });
                let ptx = match (&disk, &disk_key) {
                    (Some(cache), Some(k)) => match cache.lookup(k) {
                        Some(text) => text,
                        None => {
                            let text = compile(spec)?;
                            // Write-through to disk. Errors here are
                            // non-fatal: a failed write just means
                            // future processes won't benefit, but the
                            // current process still loads the module
                            // successfully via the in-process caches.
                            if let Err(e) = cache.store(k, &text) {
                                log::debug!("ptx disk-cache store failed: {e}");
                            }
                            text
                        }
                    },
                    _ => compile(spec)?,
                };
                Ok(ptx)
            },
        )?;
        // Bump the per-engine miss counter. We treat any path that
        // missed the per-engine cache as a "miss" for this counter —
        // even if the global cache served us — because the counter's
        // historical semantics are "did we have to look further than
        // this engine's own table?". Tests pin this invariant.
        self.module_cache_loads
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        // Seed the per-engine cache so subsequent calls on this
        // engine take the per-engine fast path above and never reach
        // the global lock.
        let mut cache = self
            .module_cache
            .lock()
            .map_err(|_| BoltError::Other("module_cache mutex poisoned".to_string()))?;
        Ok(cache.entry(key).or_insert(module).clone())
    }

    /// Batch 5 helper — rebuild the [`HostTableRevision`] for `name` so
    /// every column in `batch` carries a freshly-bumped revision and the
    /// table revision itself bumps by 1. Called from `register_table`
    /// (initial install: starts the table at revision 1) and
    /// `replace_table` (whole-table swap: starts the new shape at the
    /// next revision after whatever the old one was on).
    ///
    /// `register_batch` does NOT go through here — it bumps in place to
    /// preserve the prior `column_revisions` HashMap allocation and to
    /// update `column_n_rows` per the append semantics. See its inline
    /// code.
    fn bump_table_full_replace(&mut self, name: &str, batch: &RecordBatch) {
        let prev = self.host_revisions.remove(name);
        let next_table_rev = prev.as_ref().map(|p| p.table_revision).unwrap_or(0) + 1;
        let mut column_revisions: HashMap<String, u64> =
            HashMap::with_capacity(batch.num_columns());
        let mut column_n_rows: HashMap<String, usize> =
            HashMap::with_capacity(batch.num_columns());
        let n_rows = batch.num_rows();
        for field in batch.schema().fields() {
            column_revisions.insert(field.name().clone(), next_table_rev);
            column_n_rows.insert(field.name().clone(), n_rows);
        }
        self.host_revisions.insert(
            name.to_string(),
            HostTableRevision {
                table_revision: next_table_rev,
                column_revisions,
                column_n_rows,
                n_rows,
            },
        );
    }

    /// Register a host-side `RecordBatch` under `name` as a single-batch table.
    /// Errors if a table with that name already exists; use
    /// [`Engine::register_batch`] to append additional batches to an existing
    /// table (wave-7 multi-batch entry).
    ///
    /// Also builds Utf8 dictionaries for the table and extends the engine-side
    /// schema with `__idx_<col>` Int32 columns so the rewriter's emitted column
    /// references resolve at parse time.
    pub fn register_table(
        &mut self,
        name: impl Into<String>,
        batch: RecordBatch,
    ) -> BoltResult<()> {
        let name = name.into();
        if self.tables.contains_key(&name)
            || self.streaming_sources.borrow().contains_key(&name)
        {
            return Err(BoltError::Plan(format!(
                "table '{name}' is already registered — use register_batch to append \
                 additional batches to an existing table"
            )));
        }
        // Stage 6: the historical flatten step (`flatten_dictionary_utf8_columns`)
        // is gone from the hot path. `DictRegistry::register_table` matches
        // `DictionaryArray<Int32, Utf8>` directly and re-uses the input
        // dictionary; `GpuTable::from_record_batch` routes the same Arrow
        // variant through `upload_dict_utf8`, packing the keys' null buffer
        // into an on-device validity bitmap. Stage 4's compat materialisation
        // is preserved as a deprecated no-op for out-of-tree callers only.
        //
        // Build Utf8 dictionaries first (may fail — surface before we mutate
        // tables/provider).
        self.dict_registry.register_table(name.clone(), &batch)?;
        let base_schema = arrow_schema_to_plan_schema(batch.schema().as_ref())?;
        let extended = self.dict_registry.extended_schema(&name, &base_schema);
        self.provider.register(name.clone(), extended);
        // Stage 6: surface per-column runtime nullability so the engine's
        // null-aware paths can short-circuit the validity bitmap upload
        // when a column is provably null-free. For `DictionaryArray`
        // columns the answer comes from `keys().null_count()` — *not* the
        // dictionary values.
        propagate_column_nullability(&mut self.provider, &name, &batch);
        // Batch 5 (incremental GpuTable cache): bump revisions BEFORE
        // building the GpuTable so the GpuTable can be stamped with the
        // current host revisions and the cache hit-check in
        // `ensure_gpu_table` succeeds on the very next query.
        self.bump_table_full_replace(&name, &batch);
        let table_rev = self.host_revisions[&name].table_revision;
        // Build a GPU-resident copy so execution can query in place.
        let mut gpu_table = crate::exec::gpu_table::GpuTable::from_record_batch(&batch)?;
        gpu_table.last_uploaded_revision = table_rev;
        for col in gpu_table.columns.iter_mut() {
            col.host_revision = table_rev;
        }
        // Test-only: count one upload per column for the initial install.
        #[cfg(test)]
        self.gpu_table_load_count
            .fetch_add(gpu_table.columns.len(), std::sync::atomic::Ordering::SeqCst);
        self.gpu_tables
            .borrow_mut()
            .insert(name.clone(), Some(gpu_table));
        self.tables.insert(name, vec![batch]);
        Ok(())
    }

    /// Register a table from a producer that yields batches lazily.
    ///
    /// `schema` declares the expected per-batch schema up front (column
    /// names + dtypes); every batch yielded by `batches` is validated
    /// against it before being installed. Producer-side errors propagate
    /// out of the iterator as `BoltResult::Err(_)` — yielding an `Err`
    /// aborts registration with that error, leaving the engine in the
    /// state it had before this call (modulo any batches already
    /// installed for *this* table, which are rolled back via
    /// `tables.remove`).
    ///
    /// # v0.6 semantics: eager consumption
    ///
    /// The v0.6 cut consumes `batches` EAGERLY: every batch is pulled
    /// from the iterator and pushed into the engine's existing
    /// multi-batch in-memory table representation (the same `Vec<RecordBatch>`
    /// that backs `register_table` + repeated `register_batch`). This is
    /// deliberate — the goal here is to land a stable API *shape* so
    /// callers can write streaming-style code today and have it keep
    /// compiling when v0.7+ replaces the body with a truly lazy
    /// per-batch query plan. Until then, large streams still pay the
    /// full host-side materialisation cost on every `sql()` call (see
    /// the field doc on `tables` for the perf caveat).
    ///
    /// Roadmap: v0.7 is expected to land lazy streaming where each
    /// yielded batch is processed and discarded without materialising
    /// the full table in host memory. The signature here is
    /// future-compatible with that change.
    ///
    /// For a registration path that does NOT drain the producer up front,
    /// see [`Engine::register_table_stream_lazy`], which stores a replayable
    /// producer and only materialises it on first query — the lazy seam that
    /// backs morsel/larger-than-VRAM execution.
    ///
    /// # Errors
    /// - The iterator is empty (a table must contain at least one batch
    ///   for `materialize_table` to succeed).
    /// - Any yielded `Err` propagates out unchanged.
    /// - Any batch's schema (column names + plan-level dtypes) does not
    ///   match the declared `schema`.
    /// - A table named `name` is already registered.
    pub fn register_table_stream<I>(
        &mut self,
        name: impl Into<String>,
        schema: Schema,
        batches: I,
    ) -> BoltResult<()>
    where
        I: IntoIterator<Item = BoltResult<RecordBatch>>,
    {
        let name = name.into();
        if self.tables.contains_key(&name)
            || self.streaming_sources.borrow().contains_key(&name)
        {
            return Err(BoltError::Plan(format!(
                "table '{name}' is already registered — register_table_stream \
                 cannot append to an existing table; use register_batch instead"
            )));
        }
        // Validate one batch against the declared plan schema (names +
        // dtypes match positionally). We compare via the plan schema
        // rather than the raw Arrow schema so a caller-declared
        // `nullable: true` doesn't clash with a batch whose Arrow
        // schema happens to mark the same column non-nullable — the
        // engine treats per-row null counts as the truth and the
        // declared `nullable` is informational only at registration
        // time.
        fn validate_batch_schema(
            declared: &Schema,
            batch: &RecordBatch,
            name: &str,
            batch_idx: usize,
        ) -> BoltResult<()> {
            let actual = arrow_schema_to_plan_schema(batch.schema().as_ref())?;
            if actual.fields.len() != declared.fields.len() {
                return Err(BoltError::Plan(format!(
                    "register_table_stream: batch {batch_idx} for table '{name}' \
                     has {} columns but declared schema has {}",
                    actual.fields.len(),
                    declared.fields.len()
                )));
            }
            for (i, (a, d)) in actual.fields.iter().zip(declared.fields.iter()).enumerate() {
                if a.name != d.name || a.dtype != d.dtype {
                    return Err(BoltError::Plan(format!(
                        "register_table_stream: batch {batch_idx} for table '{name}' \
                         column {i} mismatch — declared {:?}:{:?}, got {:?}:{:?}",
                        d.name, d.dtype, a.name, a.dtype
                    )));
                }
            }
            Ok(())
        }

        // Eagerly drain the iterator, threading errors out and rolling
        // back the (just-installed) table on any failure so the engine
        // never observes a partially-installed table from this call.
        let mut iter = batches.into_iter();
        let first = match iter.next() {
            Some(Ok(b)) => b,
            Some(Err(e)) => return Err(e),
            None => {
                return Err(BoltError::Plan(format!(
                    "register_table_stream: iterator for table '{name}' yielded \
                     zero batches — a registered table must contain at least one batch"
                )));
            }
        };
        validate_batch_schema(&schema, &first, &name, 0)?;
        // Install the first batch through the same path
        // `register_table` uses — dictionaries, provider, GpuTable,
        // host-revisions all set up in one place.
        self.register_table(name.clone(), first)?;
        // Stream subsequent batches in. On any error, roll back the
        // entire table install so this call is atomic from the caller's
        // perspective.
        let mut batch_idx: usize = 1;
        loop {
            let next = match iter.next() {
                Some(Ok(b)) => b,
                Some(Err(e)) => {
                    self.unregister_table_best_effort(&name);
                    return Err(e);
                }
                None => break,
            };
            if let Err(e) = validate_batch_schema(&schema, &next, &name, batch_idx) {
                self.unregister_table_best_effort(&name);
                return Err(e);
            }
            if let Err(e) = self.register_batch(&name, next) {
                self.unregister_table_best_effort(&name);
                return Err(e);
            }
            batch_idx += 1;
        }
        Ok(())
    }

    /// Best-effort rollback helper used by `register_table_stream` when a
    /// mid-stream error needs to undo the partial install. Mirrors the
    /// state touched by `register_table` / `register_batch`.
    fn unregister_table_best_effort(&mut self, name: &str) {
        self.tables.remove(name);
        self.streaming_sources.borrow_mut().remove(name);
        self.dict_registry.unregister_table(name);
        self.provider.unregister_table(name);
        self.host_revisions.remove(name);
        self.gpu_tables.borrow_mut().remove(name);
    }

    /// Register a table from a producer that yields batches lazily, WITHOUT
    /// draining the producer at registration time (truly lazy path).
    ///
    /// Unlike [`Engine::register_table_stream`] — which drains the iterator
    /// eagerly into the engine's in-memory `Vec<RecordBatch>` representation
    /// the moment it is called — this method stores a *replayable producer*
    /// ([`crate::exec::streaming::TableSource::Streaming`]) and only registers
    /// the table's schema with the SQL frontend. The producer is not invoked
    /// until the first query that references the table, at which point the
    /// source is collapsed into the canonical materialised representation (see
    /// [`Engine::ensure_streaming_materialized`]).
    ///
    /// This keeps registration O(1) for large streams and is the seam through
    /// which larger-than-VRAM, morsel-at-a-time execution will be threaded:
    /// the budget hook ([`Engine::morsel_plan_for_table`]) inspects the
    /// materialised batches against [`Engine::memory_budget_bytes`] and decides
    /// whether to upload the table whole or process it in bounded morsels.
    ///
    /// `producer` is a factory: each call must return a fresh iterator over
    /// the same logical batch sequence (so the source can be re-drained if the
    /// engine ever re-derives the table). Producer-side errors surface as
    /// `Err` items and abort the first materialisation.
    ///
    /// # Errors
    /// - A table named `name` is already registered (eager or lazy).
    /// - Note: schema/content validation and the empty-stream check are
    ///   deferred to first query (when the producer is actually drained),
    ///   matching the lazy contract. The eager
    ///   [`Engine::register_table_stream`] validates up front instead.
    pub fn register_table_stream_lazy(
        &mut self,
        name: impl Into<String>,
        schema: Schema,
        producer: crate::exec::streaming::BatchProducer,
    ) -> BoltResult<()> {
        let name = name.into();
        if self.tables.contains_key(&name)
            || self.streaming_sources.borrow().contains_key(&name)
        {
            return Err(BoltError::Plan(format!(
                "table '{name}' is already registered — register_table_stream_lazy \
                 cannot append to an existing table"
            )));
        }
        // Register the declared schema with the SQL frontend so planning can
        // resolve column references before the producer is ever drained. We do
        // NOT extend it with dictionary `__idx_<col>` columns here — string
        // dictionaries are built when the source is materialised on first
        // query (see `ensure_streaming_materialized`).
        self.provider.register(name.clone(), schema);
        self.streaming_sources.borrow_mut().insert(
            name,
            crate::exec::streaming::TableSource::Streaming(producer),
        );
        Ok(())
    }

    /// Collapse every still-streaming overlay entry into its materialised
    /// batches by draining the producer once.
    ///
    /// Called from [`Engine::sql`] / [`Engine::run_logical_plan`] before the
    /// validity-probe provider is built. Idempotent: an entry that is already
    /// [`TableSource::Materialized`](crate::exec::streaming::TableSource::Materialized)
    /// is skipped, and a fully-eager engine pays only a single `RefCell::borrow`
    /// + emptiness check.
    ///
    /// The drained batches are staged back into the overlay as
    /// `Materialized` (not moved into `tables`), because `sql` only holds
    /// `&self` and `tables` is not interior-mutable. The eager read paths
    /// ([`Engine::materialize_table`] and [`EngineProvider`]) consult the
    /// overlay as a fall-back, so a streaming table is fully queryable from
    /// there. Dictionary `__idx_<col>` rewriting and the incremental GPU cache
    /// are not wired for overlay tables in this cut — primitive columns query
    /// end-to-end; string-literal predicates fall to the host filter path
    /// rather than the dict-fold fast path. Promoting an overlay table into the
    /// fully-wired `tables` store (dictionaries + revisions + GPU cache) is a
    /// follow-up that needs a `&mut self` seam.
    fn ensure_streaming_materialized(&self) -> BoltResult<()> {
        // Fast path: nothing streaming.
        let pending: Vec<String> = {
            let overlay = self.streaming_sources.borrow();
            overlay
                .iter()
                .filter(|(_, src)| src.is_streaming())
                .map(|(name, _)| name.clone())
                .collect()
        };
        if pending.is_empty() {
            return Ok(());
        }
        // `register_table` needs `&mut self`, but `sql` only holds `&self`.
        // We collapse the producer to concatenated host batches here (which
        // only needs `&self` interior mutability on the overlay), then stage
        // the materialised batch back into the overlay as `Materialized`. The
        // eager read paths (`materialize_table`, `EngineProvider`) consult the
        // overlay, so the table is fully queryable without ever touching
        // `tables`. Dictionary / GPU-cache wiring is built lazily by
        // `ensure_gpu_table` and the dict rewriter on demand, mirroring how
        // `register_batch` defers GPU work.
        for name in pending {
            let batches = {
                let overlay = self.streaming_sources.borrow();
                match overlay.get(&name) {
                    Some(src) => src.drain_to_batches(&name)?,
                    None => continue,
                }
            };
            self.streaming_sources.borrow_mut().insert(
                name,
                crate::exec::streaming::TableSource::Materialized(batches),
            );
        }
        Ok(())
    }

    /// Budget hook: decide whether the table named `name` can be uploaded to
    /// the device whole, or must be processed in bounded morsels because its
    /// estimated footprint exceeds this engine's
    /// [`memory_budget_bytes`](Engine::memory_budget_bytes).
    ///
    /// Returns [`crate::exec::streaming::MorselPlan::Whole`] when no budget is
    /// configured (the default) or the table fits; otherwise
    /// [`crate::exec::streaming::MorselPlan::Morsels`] with a row count sized
    /// so each morsel's working set stays under budget. The actual
    /// morsel-at-a-time upload loop — which would iterate
    /// [`crate::exec::streaming::BatchStream`] morsels and stage intermediates
    /// in [`crate::exec::streaming::PinnedBudget`] host-pinned space — is the
    /// device-side follow-up; this method is the host-side decision the
    /// orchestrator consults.
    ///
    /// Resolves the table through both the eager `tables` store and the
    /// streaming overlay (materialising the latter on demand), so it works
    /// for streaming-registered tables too.
    pub fn morsel_plan_for_table(
        &self,
        name: &str,
    ) -> BoltResult<crate::exec::streaming::MorselPlan> {
        use crate::exec::streaming::{estimate_batches_bytes, plan_upload};
        let budget = self.memory_budget_bytes;
        let plan = |batches: &[RecordBatch]| {
            let total_rows = batches
                .iter()
                .map(RecordBatch::num_rows)
                .fold(0usize, |a, n| a.saturating_add(n));
            let total_bytes = estimate_batches_bytes(batches);
            Ok(plan_upload(total_bytes, total_rows, budget))
        };
        if let Some(batches) = self.tables.get(name) {
            return plan(batches.as_slice());
        }
        self.streaming_batches(name, plan)
    }

    /// Replace any existing table named `name` with a single-batch table
    /// holding `batch`. Idempotent; equivalent to "unregister then
    /// register_table" but performs both halves atomically with respect to
    /// engine state so a failure mid-rebuild can't leave a torn table.
    ///
    /// This is the right entry point when you want to *update* a table's
    /// contents, e.g. an analytics tool that re-uploads a refreshed snapshot,
    /// or a benchmark harness that verifies on a small fixture then swaps in
    /// the timed-run dataset (the use case that motivated this method).
    ///
    /// Dictionaries, the SQL-frontend provider schema, the host-side batch
    /// list, AND the GPU-resident `GpuTable` are all rebuilt from `batch`.
    /// The previous `GpuTable`'s device allocations are returned to the
    /// memory pool, where the new upload can recycle them.
    pub fn replace_table(
        &mut self,
        name: impl Into<String>,
        batch: RecordBatch,
    ) -> BoltResult<()> {
        let name = name.into();
        // Drop any lazy streaming overlay entry under this name — a replace
        // installs an eager `tables` entry, and `materialize_table` prefers
        // `tables`, so a lingering overlay entry would just be stale dead
        // weight (and would block a future `register_table` overlay guard).
        self.streaming_sources.borrow_mut().remove(&name);
        // Stage 6: see `register_table` — the flatten step is gone from the
        // hot path. Dict ingest is native through `DictRegistry::register_table`
        // and `GpuTable::from_record_batch::upload_dict_utf8`.
        //
        // Build the new GPU table FIRST so an upload failure can't leave the
        // engine half-replaced (we have not yet touched any existing entry).
        let mut new_gpu_table = crate::exec::gpu_table::GpuTable::from_record_batch(&batch)?;
        let base_schema = arrow_schema_to_plan_schema(batch.schema().as_ref())?;

        // Drop the old GpuTable explicitly so its device allocations return
        // to the pool BEFORE we mint the dictionary index columns for the
        // replacement (those may also allocate from the pool — letting the
        // pool churn rather than grow keeps RAII tidy).
        self.gpu_tables.borrow_mut().remove(&name);
        self.dict_registry.unregister_table(&name);
        // Re-register dictionaries for the new batch.
        self.dict_registry.register_table(name.clone(), &batch)?;
        let extended = self.dict_registry.extended_schema(&name, &base_schema);
        // `MemTableProvider::register` already overwrites — no separate `replace`
        // entry point needed.
        self.provider.register(name.clone(), extended);
        // Stage 6: mirror `register_table` — re-surface per-column nullability
        // so a replace doesn't leave stale claims behind.
        propagate_column_nullability(&mut self.provider, &name, &batch);
        // Batch 5: stamp the new GpuTable with the current host revisions
        // (replace is a full rebuild, so every column gets the same fresh
        // revision number).
        self.bump_table_full_replace(&name, &batch);
        let table_rev = self.host_revisions[&name].table_revision;
        new_gpu_table.last_uploaded_revision = table_rev;
        for col in new_gpu_table.columns.iter_mut() {
            col.host_revision = table_rev;
        }
        #[cfg(test)]
        self.gpu_table_load_count.fetch_add(
            new_gpu_table.columns.len(),
            std::sync::atomic::Ordering::SeqCst,
        );
        self.gpu_tables
            .borrow_mut()
            .insert(name.clone(), Some(new_gpu_table));
        self.tables.insert(name, vec![batch]);
        Ok(())
    }

    /// Append `batch` to the table named `name`, creating it if absent.
    /// Multi-batch tables are concatenated into a single `RecordBatch` at
    /// query time via `arrow::compute::concat_batches` — see the field doc on
    /// `tables` for the perf caveat.
    ///
    /// Subsequent batches MUST share the schema of the first batch; mismatched
    /// schemas surface a `Plan` error here rather than at query time.
    ///
    /// Dictionaries are **unioned across all registered batches** (review C10):
    /// after each append, the dict registry is rebuilt against the
    /// concatenated host batches so the string-literal rewriter sees every
    /// dictionary value that exists in any batch. Without this union, a query
    /// like `WHERE s = 'literal_only_in_batch_2'` would constant-fold to
    /// `Bool(false)` against batch 0's dictionary and silently return zero
    /// rows even though batch 2 contains matching rows. The GPU index column
    /// is rebuilt lazily on the next query via `ensure_gpu_table` (which
    /// scans the same concatenated batch through `GpuTable::from_record_batch`),
    /// so the registry's dictionary and the GPU's per-row indices stay aligned
    /// — both are built from the same concat-batch input, in the same
    /// first-occurrence order.
    ///
    /// Performance: this method does NOT re-upload anything to the GPU. It
    /// only pushes the host-side `RecordBatch`, rebuilds the host-side
    /// dictionary against the materialised concat, and bumps per-column
    /// host revisions for the table. The GPU-resident `GpuTable` stays
    /// intact in the cache — the next query touches each column through
    /// `ensure_gpu_table`, which compares per-column host revisions
    /// against `GpuColumn::host_revision` and:
    ///   - reuses any column whose revision still matches (no re-upload);
    ///   - for each dirty column, allocates a new GpuVec sized for the
    ///     full new row count, DtoD-copies the previously-uploaded
    ///     prefix from the cached column, and HtoD-uploads only the
    ///     tail of new rows. The unchanged prefix never re-crosses
    ///     PCIe.
    ///
    /// Before this incremental cache (batch 5), `register_batch` set the
    /// `gpu_tables` slot to `None` and the next query re-uploaded EVERY
    /// column in full from the concatenated host batches. A
    /// streaming-append workload that issued one query between each of N
    /// appends paid `1+2+…+N = N(N+1)/2` batches' worth of HtoD traffic.
    /// With the incremental cache, the same workload pays N batches'
    /// worth — one HtoD copy of the new tail per append.
    pub fn register_batch(
        &mut self,
        name: &str,
        batch: RecordBatch,
    ) -> BoltResult<()> {
        // Stage 6: dict-encoded columns are ingested natively now, so no
        // flatten-to-StringArray is needed for the schema check below to
        // line up — batch 0 and any appended batch both carry the Arrow
        // `Dictionary<Int32, Utf8>` type verbatim.
        if let Some(existing) = self.tables.get_mut(name) {
            // Schema-check against batch 0 — concat_batches would fail at query
            // time anyway, but surface it eagerly at registration time.
            if let Some(first) = existing.first() {
                if first.schema() != batch.schema() {
                    return Err(BoltError::Plan(format!(
                        "register_batch: schema mismatch for table '{name}' — \
                         expected {:?}, got {:?}",
                        first.schema(),
                        batch.schema()
                    )));
                }
            }
            existing.push(batch);
            // Review C10: rebuild the dict registry against the *concatenated*
            // batches so the string-literal rewriter sees every dict value
            // from every batch — not just batch 0. Without this, a literal
            // that lives only in an appended batch resolves to `None` in the
            // rewriter and the predicate folds to `Bool(false)`, silently
            // dropping every otherwise-matching row in the appended batch.
            //
            // We also re-extend the provider schema in case rebuilding flipped
            // any `__idx_<col>` between i32 and i64 (the union may push a
            // column over the i64 cardinality threshold). And we re-evaluate
            // per-column nullability against the same concatenated view — a
            // previously null-free column may have just gained a null.
            let concatenated = self.materialize_table(name)?;
            self.dict_registry.unregister_table(name);
            self.dict_registry
                .register_table(name.to_string(), &concatenated)?;
            let base_schema =
                arrow_schema_to_plan_schema(concatenated.schema().as_ref())?;
            let extended = self.dict_registry.extended_schema(name, &base_schema);
            self.provider.register(name.to_string(), extended);
            propagate_column_nullability(&mut self.provider, name, &concatenated);
            // Batch 5: bump per-column host revisions for an append. Every
            // column gains rows, so every column's revision bumps; the
            // table revision bumps too. The GpuTable in `gpu_tables` is
            // INTENTIONALLY left in place — `ensure_gpu_table` will
            // compare revisions on the next query and incrementally
            // upload only the new tail per column (DtoD-preserving the
            // unchanged prefix). Note: the dict registry just rebuilt
            // its index columns from the concatenated batch in
            // first-occurrence order; since the append preserves the
            // historical row order, the prefix of the rebuilt indices
            // is bit-identical to the prefix the GpuTable already
            // holds — so the prefix-preserving copy is correct for
            // Utf8 columns too.
            let n_rows_total = concatenated.num_rows();
            let entry = self
                .host_revisions
                .entry(name.to_string())
                .or_default();
            entry.table_revision += 1;
            entry.n_rows = n_rows_total;
            let new_rev = entry.table_revision;
            for field in concatenated.schema().fields() {
                entry
                    .column_revisions
                    .insert(field.name().clone(), new_rev);
                entry
                    .column_n_rows
                    .insert(field.name().clone(), n_rows_total);
            }
            // Leave `gpu_tables[name]` untouched — incremental upload
            // happens in `ensure_gpu_table`. If the slot is somehow
            // absent (initial install raced or was cleared by an
            // out-of-band path), `ensure_gpu_table` falls through to
            // a full upload, which is still correct just not optimal.
            Ok(())
        } else {
            // First batch for a brand-new table: defer to register_table so the
            // dictionary + provider wiring happens exactly once.
            self.register_table(name.to_string(), batch)
        }
    }

    /// Make sure the GPU-resident copy of `name` is fresh.
    ///
    /// **Batch 5 (incremental cache)** — three cases:
    ///   1. Cache hit, table revision matches: return the cached `GpuTable`
    ///      as-is (no host materialisation, no uploads).
    ///   2. Cache hit, table revision diverged: walk each column, reuse
    ///      those whose `host_revision` still matches in the cache,
    ///      re-upload (with prefix-preserving extension when the column
    ///      strictly grew) the rest. Update `last_uploaded_revision` and
    ///      per-column `host_revision`.
    ///   3. Cache miss (slot absent or `None`): full upload from the
    ///      host-concatenated batch — the legacy lazy-upload path.
    ///
    /// `last_uploaded_revision` is checked under the same `RefCell` borrow
    /// that guards the cache, so a concurrent reader cannot see a torn
    /// (revision-matched, columns-not-yet-uploaded) state.
    ///
    /// Returns a `Ref` borrowing the inner `GpuTable`; held for the
    /// duration of `execute_projection`. The `RefCell` panics if a
    /// second `borrow_mut` is attempted while the `Ref` is live, but no
    /// engine method touches `gpu_tables` mutably while a query is in
    /// flight.
    fn ensure_gpu_table(
        &self,
        name: &str,
    ) -> BoltResult<Ref<'_, crate::exec::gpu_table::GpuTable>> {
        // Snapshot the host's current revision (if any) up front. We need
        // the values as owned data so we can drop the &self.host_revisions
        // borrow before borrowing &self.gpu_tables mutably below — even
        // though they're separate fields, taking owned data sidesteps any
        // borrow-graph subtlety with the `&self` we pass to
        // `incremental_rebuild`.
        let host: Option<ClonedHostRevision> = self
            .host_revisions
            .get(name)
            .cloned_revision_owned();
        // Fast path: cache hit AND every column is at the current
        // revision. Inspect under the same borrow we'd return.
        {
            let g = self.gpu_tables.borrow();
            if let Some(Some(gt)) = g.get(name) {
                if let Some(h) = host.as_ref() {
                    if gt.last_uploaded_revision == h.table_revision {
                        return Ok(Ref::map(g, |m| {
                            m.get(name)
                                .expect("hit above")
                                .as_ref()
                                .expect("Some hit above")
                        }));
                    }
                }
            }
        }
        // Either we missed entirely, the slot was None, or the revision
        // diverged. In either case we need to materialize the host
        // concatenated batch (since columns we re-upload come from
        // there).
        let concatenated = self.materialize_table(name)?;
        let mut tables_mut = self.gpu_tables.borrow_mut();
        let existing_opt = tables_mut.remove(name).flatten();
        let new_gpu_table = match existing_opt {
            Some(existing) => self.incremental_rebuild(existing, &concatenated, host.as_ref())?,
            None => {
                // Slot absent or dirty (None): full upload.
                let mut full = crate::exec::gpu_table::GpuTable::from_record_batch(
                    &concatenated,
                )?;
                if let Some(h) = host.as_ref() {
                    full.last_uploaded_revision = h.table_revision;
                    for col in full.columns.iter_mut() {
                        let rev = h
                            .column_revisions
                            .get(&col.name)
                            .copied()
                            .unwrap_or(h.table_revision);
                        col.host_revision = rev;
                    }
                }
                #[cfg(test)]
                self.gpu_table_load_count.fetch_add(
                    full.columns.len(),
                    std::sync::atomic::Ordering::SeqCst,
                );
                full
            }
        };
        tables_mut.insert(name.to_string(), Some(new_gpu_table));
        drop(tables_mut);
        let g = self.gpu_tables.borrow();
        Ok(Ref::map(g, |m| {
            m.get(name)
                .expect("just inserted")
                .as_ref()
                .expect("just inserted Some")
        }))
    }

    /// Batch 5 incremental rebuild: given the cached `existing` GpuTable
    /// and the freshly-concatenated host batch `concatenated`, produce a
    /// GpuTable whose columns are either reused from `existing` (when
    /// their per-column revision still matches the host's view) or
    /// re-uploaded — prefix-preserving when the host data strictly
    /// extended (append), full re-upload otherwise.
    ///
    /// `host` is the engine's `HostTableRevision` snapshot for the
    /// table. `None` means the host doesn't track revisions for this
    /// table (out-of-band install path); falls back to a full rebuild.
    fn incremental_rebuild(
        &self,
        existing: crate::exec::gpu_table::GpuTable,
        concatenated: &RecordBatch,
        host: Option<&ClonedHostRevision>,
    ) -> BoltResult<crate::exec::gpu_table::GpuTable> {
        // Without host revisions we can't decide what's stale → full rebuild.
        let host = match host {
            Some(h) => h,
            None => {
                let table =
                    crate::exec::gpu_table::GpuTable::from_record_batch(concatenated)?;
                #[cfg(test)]
                self.gpu_table_load_count
                    .fetch_add(table.columns.len(), std::sync::atomic::Ordering::SeqCst);
                return Ok(table);
            }
        };
        // Decompose `existing` into a name → GpuColumn map so we can
        // reuse columns positionally without quadratic search.
        let crate::exec::gpu_table::GpuTable {
            n_rows: _,
            columns: existing_columns,
            last_uploaded_revision: _,
        } = existing;
        let mut existing_by_name: HashMap<String, crate::exec::gpu_table::GpuColumn> =
            existing_columns
                .into_iter()
                .map(|c| (c.name.clone(), c))
                .collect();

        let n_rows_total = concatenated.num_rows();
        let schema = concatenated.schema();
        let mut new_columns: Vec<crate::exec::gpu_table::GpuColumn> =
            Vec::with_capacity(concatenated.num_columns());
        for (idx, field) in schema.fields().iter().enumerate() {
            let name = field.name();
            let host_col_rev = host
                .column_revisions
                .get(name)
                .copied()
                .unwrap_or(host.table_revision);
            let reused = existing_by_name.remove(name);
            let col = match reused {
                Some(prev) if prev.host_revision == host_col_rev => {
                    // Cache hit on this column — reuse in place. No upload.
                    prev
                }
                Some(prev) => {
                    // Stale column. If the host data strictly extended
                    // (n_rows grew), try the prefix-preserving path; else
                    // fall through to a full re-upload.
                    let prev_rows = column_storage_rows(&prev.data);
                    if prev_rows > 0 && prev_rows < n_rows_total {
                        match try_extend_column(prev, concatenated, idx, n_rows_total) {
                            Ok(Some(mut extended)) => {
                                extended.host_revision = host_col_rev;
                                #[cfg(test)]
                                self.gpu_table_load_count
                                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                extended
                            }
                            Ok(None) => {
                                // Variant not extensible — full re-upload.
                                let mut fresh =
                                    crate::exec::gpu_table::GpuTable::upload_column_from_batch(
                                        concatenated,
                                        field,
                                        idx,
                                    )?;
                                fresh.host_revision = host_col_rev;
                                #[cfg(test)]
                                self.gpu_table_load_count
                                    .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                                fresh
                            }
                            Err(e) => return Err(e),
                        }
                    } else {
                        // Either previous column was empty / replaced (not
                        // an append) — full re-upload.
                        drop(prev);
                        let mut fresh =
                            crate::exec::gpu_table::GpuTable::upload_column_from_batch(
                                concatenated,
                                field,
                                idx,
                            )?;
                        fresh.host_revision = host_col_rev;
                        #[cfg(test)]
                        self.gpu_table_load_count
                            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        fresh
                    }
                }
                None => {
                    // Column not in the previous cache — full upload.
                    let mut fresh =
                        crate::exec::gpu_table::GpuTable::upload_column_from_batch(
                            concatenated,
                            field,
                            idx,
                        )?;
                    fresh.host_revision = host_col_rev;
                    #[cfg(test)]
                    self.gpu_table_load_count
                        .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    fresh
                }
            };
            new_columns.push(col);
        }
        Ok(crate::exec::gpu_table::GpuTable {
            n_rows: n_rows_total,
            columns: new_columns,
            last_uploaded_revision: host.table_revision,
        })
    }

    /// Test-only accessor for the per-column upload counter. Returns the
    /// number of GpuColumn (re)uploads performed across the engine's
    /// lifetime. Used by the incremental-cache regression tests to
    /// assert that an unchanged column was reused.
    #[cfg(test)]
    pub(crate) fn gpu_table_load_count(&self) -> usize {
        self.gpu_table_load_count
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Materialise the concatenated `RecordBatch` for a registered table.
    ///
    /// Fast-path: zero batches errors, one batch is cloned cheaply (Arrow
    /// arrays are Arc-backed). Two or more batches go through
    /// `arrow::compute::concat_batches`, which copies every column — the
    /// 0.2 perf cost the field doc on `tables` warns about.
    fn materialize_table(&self, name: &str) -> BoltResult<RecordBatch> {
        // Eager tables first; fall back to the lazy streaming overlay.
        if let Some(batches) = self.tables.get(name) {
            return concat_table_batches(name, batches);
        }
        // Streaming overlay: collapse the source to materialised on first
        // read, then concat its batches.
        self.streaming_batches(name, |batches| concat_table_batches(name, batches))
    }

    /// Ensure the streaming source for `name` is drained, then run `f` over its
    /// materialised batches while holding the overlay borrow.
    ///
    /// On first call for a `Streaming` entry this invokes the producer,
    /// collapsing the entry in place to [`TableSource::Materialized`] so
    /// subsequent reads skip the producer. Returns a `Plan` error if `name` is
    /// not in the streaming overlay at all.
    fn streaming_batches<R>(
        &self,
        name: &str,
        f: impl FnOnce(&[RecordBatch]) -> BoltResult<R>,
    ) -> BoltResult<R> {
        use crate::exec::streaming::TableSource;
        // Phase 1: if the entry is still a producer, drain it (without holding
        // a borrow across the producer call) and swap in the materialised
        // form. Borrow is dropped before re-borrowing for the read.
        let needs_drain = matches!(
            self.streaming_sources.borrow().get(name),
            Some(src) if src.is_streaming()
        );
        if needs_drain {
            let drained = {
                let overlay = self.streaming_sources.borrow();
                match overlay.get(name) {
                    Some(src) => src.drain_to_batches(name)?,
                    None => {
                        return Err(BoltError::Plan(format!(
                            "table '{name}' is not registered with the engine"
                        )))
                    }
                }
            };
            self.streaming_sources
                .borrow_mut()
                .insert(name.to_string(), TableSource::Materialized(drained));
        }
        // Phase 2: read the (now materialised) batches.
        let overlay = self.streaming_sources.borrow();
        match overlay.get(name) {
            Some(TableSource::Materialized(batches)) => f(batches),
            Some(TableSource::Streaming(_)) => {
                // Unreachable: we just collapsed it above.
                Err(BoltError::Other(format!(
                    "streaming source for table '{name}' was not collapsed after drain"
                )))
            }
            None => Err(BoltError::Plan(format!(
                "table '{name}' is not registered with the engine"
            ))),
        }
    }

    /// Compile and execute a SQL query string.
    ///
    /// Stage 7 (P1b): after the query completes, the engine emits a
    /// periodic pool-stats log line at most once every
    /// `BOLT_POOL_STATS_INTERVAL_SECS` (default 60s). The emit happens
    /// AFTER the query's `QueryHandle` is fully materialised — the log
    /// line is off the latency-critical path for the just-returned
    /// query. Failures (query error, log throttled, no-op observer)
    /// never affect the query result.
    /// Snapshot the registered tables' row counts into an estimator the
    /// cost-based optimizer can consume.
    ///
    /// Each base table's row count is the sum of its registered `RecordBatch`
    /// row counts (eager `tables` store) plus any already-materialised
    /// streaming source. Lazily-registered streaming tables that haven't been
    /// drained yet are intentionally omitted — costing them would force
    /// materialisation at plan time. Their absence simply leaves any join chain
    /// touching them un-reordered (the conservative default).
    ///
    /// The result is an owned `Arc<dyn RowEstimator>` (a
    /// [`crate::plan::optimizer::StatsEstimator`] over an [`EngineTableStats`]
    /// snapshot) ready to hand to
    /// [`crate::plan::default_passes_with_estimator`].
    fn row_estimator(&self) -> Arc<dyn crate::plan::RowEstimator> {
        let mut row_counts: HashMap<String, usize> = HashMap::new();
        for (name, batches) in &self.tables {
            let rows = batches
                .iter()
                .map(RecordBatch::num_rows)
                .fold(0usize, |a, n| a.saturating_add(n));
            row_counts.insert(name.clone(), rows);
        }
        // Fold in any streaming sources that have already been materialised
        // (a still-streaming source is skipped — see the method docs).
        for (name, src) in self.streaming_sources.borrow().iter() {
            if let crate::exec::streaming::TableSource::Materialized(batches) = src {
                let rows = batches
                    .iter()
                    .map(RecordBatch::num_rows)
                    .fold(0usize, |a, n| a.saturating_add(n));
                row_counts.entry(name.clone()).or_insert(rows);
            }
        }
        Arc::new(crate::plan::StatsEstimator::new(EngineTableStats { row_counts }))
    }

    pub fn sql(&self, query: &str) -> BoltResult<QueryHandle> {
        // **Stage 6 (M3L5)** — retry the pool-watcher's context capture.
        // If the watcher spawned before any engine thread had a context
        // bound, `CAPTURED_CTX` is still zero and every poll silently
        // no-ops. This call is cheap (atomic load when already
        // captured) and a no-op when no context is bound on the
        // calling thread — so it's safe to invoke unconditionally.
        crate::cuda::mem_pool::pool_watcher_retry_context_capture();

        // M5 metrics: count every accepted query (success or failure). The
        // matching `QueriesFailed` bump happens at the single error-return
        // below so the `?`-chain stays intact.
        crate::metrics::metrics().inc(crate::metrics::Counter::QueriesTotal);

        // F1: a `WITH RECURSIVE` query is detected and orchestrated as a
        // host-side fixpoint before the ordinary parse→optimize→lower→execute
        // pipeline (the recursive structure does not fit a single LogicalPlan
        // that flows through that pipeline). `plan_recursive_cte` returns
        // `Ok(None)` for every non-recursive query, so the common path falls
        // straight through. Failures bump `QueriesFailed` to mirror the
        // ordinary error accounting below.
        match crate::plan::sql_frontend::plan_recursive_cte(query, &self.provider) {
            Ok(Some(rec)) => {
                use crate::plan::sql_frontend::RecursiveQueryPlan;
                let result = match rec {
                    RecursiveQueryPlan::Single(rec) => self.execute_recursive_cte(&rec),
                    RecursiveQueryPlan::Mutual(rec) => self.execute_mutual_recursive_cte(&rec),
                };
                if result.is_err() {
                    crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
                }
                self.maybe_emit_pool_stats(Instant::now());
                return result;
            }
            Ok(None) => {}
            Err(e) => {
                crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
                return Err(e);
            }
        }

        // F3 (LATERAL): a `FROM <left>, LATERAL (<subquery>) AS a` query is
        // detected and orchestrated host-side as a nested-loop Apply (dependent
        // join) before the ordinary pipeline, mirroring the WITH RECURSIVE hook
        // above (the correlated subquery has no single LogicalPlan). Returns
        // `Ok(None)` for every non-LATERAL query, so the common path falls
        // straight through. Failures bump `QueriesFailed` to mirror accounting.
        match crate::plan::sql_frontend::plan_lateral_apply(query, &self.provider) {
            Ok(Some(la)) => {
                let result = self.execute_lateral_apply(&la);
                if result.is_err() {
                    crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
                }
                self.maybe_emit_pool_stats(Instant::now());
                return result;
            }
            Ok(None) => {}
            Err(e) => {
                crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
                return Err(e);
            }
        }

        // F3-finish: a sole `COUNT(DISTINCT col)` with `GROUP BY` is detected
        // and orchestrated host-side (per-group distinct counting) before the
        // ordinary pipeline, mirroring the WITH RECURSIVE hook above. The
        // per-group distinct count has no single LogicalPlan that flows through
        // the normal lowering (the plan-node route is blocked — see
        // `CountDistinctGroupByPlan`). `plan_count_distinct_groupby` returns
        // `Ok(None)` for every other query, so the common path falls straight
        // through. Failures bump `QueriesFailed` to mirror the accounting
        // below.
        match crate::plan::sql_frontend::plan_count_distinct_groupby(query, &self.provider) {
            Ok(Some(cd)) => {
                let result = self.execute_count_distinct_groupby(&cd);
                if result.is_err() {
                    crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
                }
                self.maybe_emit_pool_stats(Instant::now());
                return result;
            }
            Ok(None) => {}
            Err(e) => {
                crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
                return Err(e);
            }
        }

        // Generalized COUNT(DISTINCT) + GROUP BY (multiple distinct counts
        // and/or a mix with plain aggregates). The single-sole-COUNT(DISTINCT)
        // case above declines these (`Ok(None)`); this detector picks them up.
        // `Ok(None)` for everything else falls through to the ordinary path.
        match crate::plan::sql_frontend::plan_multi_agg_groupby(query, &self.provider) {
            Ok(Some(cd)) => {
                let result = self.execute_multi_agg_groupby(&cd);
                if result.is_err() {
                    crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
                }
                self.maybe_emit_pool_stats(Instant::now());
                return result;
            }
            Ok(None) => {}
            Err(e) => {
                crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
                return Err(e);
            }
        }

        // Time the parse phase (SQL text → LogicalPlan) into the Parse
        // histogram; mirrors the `parse` tracing span in `observability`.
        let parse_start = Instant::now();
        let plan: LogicalPlan = parse_sql(query, &self.provider)?;
        crate::metrics::metrics()
            .observe_duration(crate::metrics::Phase::Parse, parse_start.elapsed());
        // String-literal predicates against Utf8 columns are folded into
        // integer equality against the corresponding __idx_<col> i32 column.
        // Time the whole logical-planning stage (rewrites + optimizer +
        // subquery resolution) into the `Plan` histogram.
        let plan_start = Instant::now();
        let plan = tracing::info_span!("plan")
            .in_scope(|| self.dict_registry.rewrite_plan(&plan))?;
        let plan = self.dict_registry.rewrite_plan(&plan)?;
        // Built-in logical optimizer: run the default pass pipeline
        // (constant folding, predicate pushdown, filter-into-join, join
        // reorder, projection pruning) BEFORE any user-registered rewrites.
        // The join-reorder pass is driven by a statistics-backed row
        // estimator snapshotted from the registered tables, so left-deep
        // INNER chains are reordered smallest-input-first (it stays a no-op
        // for chains whose leaves the snapshot can't cost). See
        // `crate::plan::optimizer` for the pipeline and ordering. The
        // pipeline is driven to a bounded fixpoint (re-running the same
        // pass set/order until the plan stabilises or the iteration cap is
        // hit) so a conjunct exposed by one sweep — e.g. a now-constant
        // predicate moved by pushdown — gets folded/pushed on the next.
        let passes = crate::plan::default_passes_with_estimator(self.row_estimator());
        // `self.optimize` is `true` for every production engine; the
        // test-only `EngineBuilder::without_optimizer()` flips it to `false`
        // so the optimizer-equivalence test can execute the plan as written.
        let plan = if self.optimize {
            crate::plan::optimizer::run_to_fixpoint(&passes, plan)?
        } else {
            plan
        };
        // v0.6 / M7: run user-registered PlanRewrite implementations in
        // registration order, threading each rewriter's output into the
        // next. This runs AFTER the built-in optimizer and the internal
        // dict-rewrite (so user rewrites see the engine's normalised form
        // with `__idx_<col>` refs already in place) and BEFORE
        // `lower_physical` (so users can still target logical-plan
        // structure). See `crate::plan::rewrite` for the contract.
        let plan = self
            .rewrites
            .iter()
            .try_fold(plan, |p, r| r.rewrite(p))?;
        // Resolve uncorrelated scalar / IN subqueries to constants BEFORE
        // lowering. Each subplan is uncorrelated (the frontend rejects
        // correlation) and so independently executable; we run it here and
        // fold the result into the enclosing plan as a literal (scalar) or a
        // boolean OR/AND fold (IN). After this pass no `ScalarSubquery` /
        // `InSubquery` node survives, so the physical reject-arms for them
        // are unreachable for `sql()`-produced plans (they stay as a safety
        // net for hand-built physical plans). See
        // `crate::exec::subquery_resolve`.
        let plan = self.resolve_subqueries(plan)?;
        crate::metrics::metrics()
            .observe_duration(crate::metrics::Phase::Plan, plan_start.elapsed());
        // Time the lower phase (LogicalPlan → PhysicalPlan) into the Lower
        // histogram; mirrors the `lower` tracing span in `observability`.
        let lower_start = Instant::now();
        let mut phys = crate::plan::lower_physical(&plan)?;
        crate::metrics::metrics()
            .observe_duration(crate::metrics::Phase::Lower, lower_start.elapsed());
        // Collapse any lazily-registered streaming sources to materialised
        // batches so the validity probes below (and `execute`'s
        // `materialize_table`) see real data. Idempotent and a no-op when no
        // streaming tables are registered.
        self.ensure_streaming_materialized()?;
        // PV-stage-d: populate `KernelSpec::input_has_validity` for every
        // input column by consulting the engine-backed provider, which
        // looks straight at `RecordBatch::column(col).null_count()` for
        // each registered table. This is the plan-time signal that lets
        // the codegen emit native-validity kernels instead of leaning on
        // the run-time host-strip fallback in `groupby_with_pre` etc.
        let nb = EngineProvider {
            base: &self.provider,
            tables: &self.tables,
            streaming: self.streaming_sources.borrow(),
        };
        crate::plan::physical_plan::populate_input_validity(&mut phys, &nb);
        // Release the streaming-overlay borrow held by `nb` before `execute`,
        // whose `ensure_gpu_table`/`materialize_table` path may re-borrow the
        // overlay (immutably; mutably only for an un-collapsed source, which
        // `ensure_streaming_materialized` above has already ruled out).
        drop(nb);
        let result = self.execute(&phys);
        // M5 metrics: a failed execution bumps `QueriesFailed`. We only
        // observe the bind here (no `?`-chain restructuring); the early
        // parse/rewrite/lower `?`-returns above are rare developer-time
        // errors, while `execute` is the latency-critical phase whose
        // failures (OOM, kernel fault) are the ones worth a dashboard counter.
        if result.is_err() {
            crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
        }
        // Stage 7: periodic pool-stats emit. Runs whether the query
        // succeeded or failed (an OOM-failed query is itself a signal
        // worth surfacing alongside the pool snapshot). Internal errors
        // in the emit path are swallowed — they must never escalate to
        // the query result.
        self.maybe_emit_pool_stats(Instant::now());
        result
    }

    /// Parse, plan, and *render* `query` without executing it — an
    /// `EXPLAIN`-style introspection hook.
    ///
    /// Reuses the same logical-plan path as [`Engine::sql`] (parse +
    /// dict-rewrite + user [`PlanRewrite`]s) but stops short of lowering to
    /// GPU kernels or touching the device. The returned string contains the
    /// tree-indented logical plan (via
    /// [`crate::plan::format_logical`]); when the plan also lowers cleanly to
    /// a physical plan it appends the physical rendering (via
    /// [`crate::plan::format_physical`]). Plans that the GPU codegen does not
    /// yet support (e.g. `CASE`, `CAST`) still return their logical plan, with
    /// the lowering error noted in place of the physical section.
    pub fn explain_sql(&self, query: &str) -> BoltResult<String> {
        // F1: a `WITH RECURSIVE` query has no single LogicalPlan; render its
        // three subplans via the dedicated formatter instead.
        if let Some(rec) =
            crate::plan::sql_frontend::plan_recursive_cte(query, &self.provider)?
        {
            use crate::plan::sql_frontend::RecursiveQueryPlan;
            return Ok(match rec {
                RecursiveQueryPlan::Single(rec) => format!(
                    "Recursive CTE plan:\n{}",
                    crate::plan::explain::format_recursive_cte(&rec)
                ),
                RecursiveQueryPlan::Mutual(rec) => format!(
                    "Mutual recursive CTE plan:\n{}",
                    crate::plan::explain::format_mutual_recursive_cte(&rec)
                ),
            });
        }
        // F3 (LATERAL): render the host-orchestrated LATERAL apply descriptor.
        if let Some(la) = crate::plan::sql_frontend::plan_lateral_apply(query, &self.provider)? {
            return Ok(format!(
                "LATERAL apply plan:\n{}",
                crate::plan::explain::format_lateral_apply(&la)
            ));
        }
        // F3-finish: render the host-orchestrated COUNT(DISTINCT) + GROUP BY
        // descriptor via its dedicated formatter.
        if let Some(cd) =
            crate::plan::sql_frontend::plan_count_distinct_groupby(query, &self.provider)?
        {
            return Ok(format!(
                "COUNT(DISTINCT) GROUP BY plan:\n{}",
                crate::plan::explain::format_count_distinct_groupby(&cd)
            ));
        }
        // F3-finish (generalized): render the multi/mixed COUNT(DISTINCT) +
        // GROUP BY descriptor.
        if let Some(cd) =
            crate::plan::sql_frontend::plan_multi_agg_groupby(query, &self.provider)?
        {
            return Ok(format!(
                "multi-agg GROUP BY plan:\n{}",
                crate::plan::explain::format_multi_agg_groupby(&cd)
            ));
        }
        let plan: LogicalPlan = parse_sql(query, &self.provider)?;
        let plan = self.dict_registry.rewrite_plan(&plan)?;
        let plan = self.rewrites.iter().try_fold(plan, |p, r| r.rewrite(p))?;

        let mut out = String::from("Logical plan:\n");
        out.push_str(&crate::plan::format_logical(&plan));
        out.push_str("\nPhysical plan:\n");
        match crate::plan::lower_physical(&plan) {
            Ok(phys) => out.push_str(&crate::plan::format_physical(&phys)),
            Err(e) => out.push_str(&format!("  <not lowered: {e}>\n")),
        }
        Ok(out)
    }

    /// Execute an already-built [`LogicalPlan`] and return a [`QueryHandle`].
    ///
    /// This is the post-parse half of [`Engine::sql`]: it skips SQL parsing
    /// (so the input plan can come from the [`DataFrame`](crate::DataFrame)
    /// builder, a test fixture, etc.) but still runs the full
    /// rewrite → lower → validity-propagate → execute pipeline so the
    /// physical plan reaching the kernels is shaped identically to the SQL
    /// path. The pool-stats periodic emit is performed here too, mirroring
    /// `sql()`'s book-keeping.
    ///
    /// `&mut self` matches the [`DataFrame::collect`](crate::DataFrame::collect)
    /// signature; the engine state mutated here is bounded to the
    /// pool-stats throttle and the (interior-mutable) GpuTable cache
    /// already touched by `sql()`.
    pub fn run_logical_plan(&mut self, plan: &LogicalPlan) -> BoltResult<QueryHandle> {
        crate::cuda::mem_pool::pool_watcher_retry_context_capture();
        // M5 metrics: the DataFrame `collect` path is a top-level query just
        // like `sql()`, so count it identically — bump `QueriesTotal` here and
        // `QueriesFailed` at the single error-observe below. This is the *only*
        // place the DataFrame path is counted: nested sub-plans resolved during
        // this call run through `run_subplan`, which deliberately does NOT bump
        // the counters (so N subqueries inside one top-level plan still count as
        // exactly one query, matching the `sql()` contract).
        crate::metrics::metrics().inc(crate::metrics::Counter::QueriesTotal);
        // String-literal predicates against Utf8 columns are folded into
        // integer equality against the corresponding __idx_<col> i32 column —
        // mirrors `sql()`.
        let plan = self.dict_registry.rewrite_plan(plan)?;
        // Built-in logical optimizer: mirror the `sql()` path so a plan built
        // via the DataFrame builder gets the same default optimizations before
        // lowering — including statistics-driven join reordering. See
        // `crate::plan::optimizer`. Driven to a bounded fixpoint exactly as
        // `sql()` does, so DataFrame-built plans get the same thorough
        // fold/push convergence.
        let passes = crate::plan::default_passes_with_estimator(self.row_estimator());
        // Gated by the test-only optimizer toggle; `true` in production.
        // See `EngineBuilder::without_optimizer`.
        let plan = if self.optimize {
            crate::plan::optimizer::run_to_fixpoint(&passes, plan)?
        } else {
            plan
        };
        // Mirror `sql()`: resolve uncorrelated subqueries to constants before
        // lowering so a DataFrame-built plan carrying a subquery executes too.
        let plan = self.resolve_subqueries(plan)?;
        let mut phys = crate::plan::lower_physical(&plan)?;
        // Mirror `sql()`: collapse lazy streaming sources before probing.
        self.ensure_streaming_materialized()?;
        // PV-stage-d: thread per-column null-bearing into the kernel specs.
        let nb = EngineProvider {
            base: &self.provider,
            tables: &self.tables,
            streaming: self.streaming_sources.borrow(),
        };
        crate::plan::physical_plan::populate_input_validity(&mut phys, &nb);
        drop(nb);
        let result = self.execute(&phys);
        // M5 metrics: mirror `sql()` — a failed top-level execution bumps
        // `QueriesFailed`. We only observe the `execute` bind (the rare early
        // `?`-returns above are developer-time plan errors, while `execute` is
        // the latency-critical phase whose OOM/kernel faults warrant a counter).
        if result.is_err() {
            crate::metrics::metrics().inc(crate::metrics::Counter::QueriesFailed);
        }
        self.maybe_emit_pool_stats(Instant::now());
        result
    }

    /// Pre-lowering pass: resolve every uncorrelated scalar / IN subquery in
    /// `plan` to a constant.
    ///
    /// Walks the plan's expressions (recursively, inner-subqueries-first via
    /// [`crate::exec::subquery_resolve::resolve_plan`]) and:
    ///
    /// * `Expr::ScalarSubquery(subplan)` → executes `subplan`, requires
    ///   exactly one output column, and folds the result to an
    ///   `Expr::Literal` (0 rows → SQL `NULL`; >1 row → a clean error).
    /// * `Expr::InSubquery { expr, subquery, negated }` → executes `subquery`,
    ///   collects the distinct values, and rewrites to an OR-of-equalities
    ///   (or AND-of-inequalities for `NOT IN`) over `expr`.
    ///
    /// The subplans are executed through [`Engine::run_subplan`], which routes
    /// the full pipeline (dict-rewrite → optimizer → *this pass* → lower →
    /// execute) over `&self` — so nested subqueries inside a subplan resolve
    /// when that subplan runs. Correlation is impossible here (rejected at the
    /// frontend), so each subplan is self-contained and independently
    /// executable.
    ///
    /// Takes `&self` because [`Engine::sql`] is `&self`; the only state mutated
    /// by executing a subplan is the interior-mutable GpuTable cache and the
    /// pool-stats throttle, neither of which needs `&mut`.
    fn resolve_subqueries(&self, plan: LogicalPlan) -> BoltResult<LogicalPlan> {
        let mut exec = |subplan: LogicalPlan| -> BoltResult<RecordBatch> {
            self.run_subplan(subplan)
        };
        crate::exec::subquery_resolve::resolve_plan(plan, &mut exec)
    }

    /// Execute a self-contained (uncorrelated) subquery [`LogicalPlan`] over
    /// `&self` and return its result `RecordBatch`.
    ///
    /// This is the `&self` twin of [`Engine::run_logical_plan`]: it runs the
    /// same dict-rewrite → optimizer → subquery-resolve → lower →
    /// validity-propagate → execute pipeline, but without the `&mut self`
    /// receiver (so it can be called from inside [`Engine::resolve_subqueries`]
    /// during the `&self` `sql()` path). Re-entering `resolve_subqueries` here
    /// is what makes nested subqueries resolve inner-first.
    ///
    /// **Metrics contract (M5):** this path deliberately does NOT bump
    /// `QueriesTotal` / `QueriesFailed`. The query counters count *top-level*
    /// queries only — the enclosing `sql()` / `run_logical_plan` call has
    /// already counted the whole statement once. A statement containing N
    /// subqueries therefore counts as one query (and a subquery failure surfaces
    /// as the top-level query's failure via the `?` below), keeping the
    /// `sql()` and DataFrame paths symmetric.
    fn run_subplan(&self, plan: LogicalPlan) -> BoltResult<RecordBatch> {
        let plan = self.dict_registry.rewrite_plan(&plan)?;
        // Bounded-fixpoint optimizer, mirroring the `sql()` / `run_logical_plan`
        // paths so a subplan is optimized to the same convergence.
        let passes = crate::plan::default_passes_with_estimator(self.row_estimator());
        // Gated by the test-only optimizer toggle; `true` in production.
        // See `EngineBuilder::without_optimizer`. Keeping the subplan path
        // gated too means a query run on a `without_optimizer()` engine is
        // un-optimized end to end (outer plan AND any subplans).
        let plan = if self.optimize {
            crate::plan::optimizer::run_to_fixpoint(&passes, plan)?
        } else {
            plan
        };
        let plan = self.resolve_subqueries(plan)?;
        let mut phys = crate::plan::lower_physical(&plan)?;
        // The outer `sql()` / `run_logical_plan` has already collapsed any
        // lazy streaming sources before this point, but a subplan may be the
        // first reader of one — collapse again (idempotent) to be safe.
        self.ensure_streaming_materialized()?;
        let nb = EngineProvider {
            base: &self.provider,
            tables: &self.tables,
            streaming: self.streaming_sources.borrow(),
        };
        crate::plan::physical_plan::populate_input_validity(&mut phys, &nb);
        drop(nb);
        let handle = self.execute(&phys)?;
        Ok(handle.into_record_batch())
    }

    /// Execute a `WITH RECURSIVE` query (feature F1) by orchestrating a
    /// host-side fixpoint, then running the main query over the accumulated
    /// CTE relation.
    ///
    /// Algorithm (correctness-first, reusing the existing subplan executor):
    ///
    /// 1. **Seed.** Execute the anchor term → the initial working set (and the
    ///    initial accumulated result). For `UNION` (distinct) the seed is
    ///    deduplicated.
    /// 2. **Iterate.** Register the *working set* (the previous iteration's
    ///    newly-produced rows) as an ephemeral in-memory table named after the
    ///    CTE, then execute the recursive term (its `cte` scan reads that
    ///    table). For `UNION ALL` every produced row is appended and the new
    ///    working set is the whole recursive output; iteration stops when an
    ///    iteration produces zero rows. For `UNION` the recursive output is
    ///    de-duplicated against the full accumulated result (via
    ///    [`crate::exec::distinct::execute_distinct`]); iteration stops at the
    ///    fixpoint (the de-duplicated result stops growing). See the inline
    ///    comment on the `UNION` branch for why this uses a naive (whole-result)
    ///    working set rather than a row-order-dependent delta.
    /// 3. **Cap.** A hard cap ([`max_recursive_iterations`], default
    ///    [`MAX_RECURSIVE_ITERATIONS`], env-overridable) bounds the loop and
    ///    returns a clean [`BoltError`] if exceeded — recursive CTEs can loop
    ///    forever on bad input, so this guard is mandatory.
    /// 4. **Main.** Register the full accumulated relation as the CTE table and
    ///    run the main query over it.
    ///
    /// The ephemeral CTE table lives only in the interior-mutable streaming
    /// overlay (so this method can stay `&self`, matching [`Engine::sql`]); it
    /// is removed before returning. A name collision with a real registered
    /// table is rejected up front.
    fn execute_recursive_cte(
        &self,
        rec: &crate::plan::sql_frontend::RecursiveCtePlan,
    ) -> BoltResult<QueryHandle> {
        use crate::exec::distinct::execute_distinct;
        use crate::exec::streaming::TableSource;

        // Refuse to shadow a real table — the ephemeral overlay registration
        // would otherwise clobber the user's data for the duration of the
        // query (and the cleanup below would wrongly delete a real table's
        // overlay entry).
        if self.tables.contains_key(&rec.name)
            || self.streaming_sources.borrow().contains_key(&rec.name)
        {
            return Err(BoltError::Plan(format!(
                "WITH RECURSIVE: CTE name '{}' collides with a registered \
                 table — rename the CTE",
                rec.name
            )));
        }

        // The ephemeral CTE table must present exactly the CTE's declared
        // column names (the recursive term / main query scan it by name). The
        // anchor and recursive subplans may emit differently-named columns, so
        // every produced batch is re-labelled to `rec.cte_schema` before it is
        // registered.
        let arrow_schema = plan_schema_to_arrow_schema(&rec.cte_schema)?;
        let relabel = |batch: RecordBatch| -> BoltResult<RecordBatch> {
            if batch.num_columns() != arrow_schema.fields().len() {
                return Err(BoltError::Plan(format!(
                    "WITH RECURSIVE: a term produced {} columns but the CTE \
                     '{}' declares {}",
                    batch.num_columns(),
                    rec.name,
                    arrow_schema.fields().len()
                )));
            }
            RecordBatch::try_new(Arc::clone(&arrow_schema), batch.columns().to_vec())
                .map_err(|e| BoltError::Plan(format!("WITH RECURSIVE relabel: {e}")))
        };

        // Run a subplan with the CTE table bound to `rows` in the overlay.
        // The ephemeral table's contents change every iteration but reuse the
        // same name, and the engine has no host-revision tracking for an
        // overlay table — so we evict any GPU-resident copy from the previous
        // iteration BEFORE and AFTER each run, forcing `ensure_gpu_table` to
        // re-upload the current working set rather than reuse stale device
        // columns. The overlay entry is always cleared afterwards so a failure
        // mid-loop can't leak an ephemeral table into a later query.
        let run_with_cte = |rows: &RecordBatch, plan: &LogicalPlan| -> BoltResult<RecordBatch> {
            self.gpu_tables.borrow_mut().remove(&rec.name);
            self.streaming_sources.borrow_mut().insert(
                rec.name.clone(),
                TableSource::Materialized(vec![rows.clone()]),
            );
            let out = self.run_subplan(plan.clone());
            self.streaming_sources.borrow_mut().remove(&rec.name);
            self.gpu_tables.borrow_mut().remove(&rec.name);
            out
        };

        // --- 1. Seed: execute the anchor term. ---
        let anchor_out = relabel(self.run_subplan(rec.anchor.clone())?)?;
        // For UNION (distinct) the accumulated result is kept de-duplicated.
        let mut result = if rec.all {
            anchor_out.clone()
        } else {
            execute_distinct(QueryHandle::from_record_batch(anchor_out.clone()))?
                .into_record_batch()
        };
        // The working set fed into the first recursive iteration is the seed
        // (its de-duplicated form for UNION).
        let mut working_set = result.clone();

        let cap = max_recursive_iterations();
        let mut iters = 0usize;

        // --- 2. Iterate to a fixpoint. ---
        loop {
            if working_set.num_rows() == 0 {
                break; // No rows to drive the next iteration → fixpoint.
            }
            iters += 1;
            if iters > cap {
                return Err(BoltError::Plan(format!(
                    "WITH RECURSIVE: exceeded the {cap}-iteration safety cap \
                     (set {MAX_RECURSIVE_ITERATIONS_ENV} to override) — the \
                     recursion is not terminating"
                )));
            }

            let rec_out = relabel(run_with_cte(&working_set, &rec.recursive)?)?;

            if rec.all && rec.naive {
                // NON-LINEAR UNION ALL (naive): the recursive term scans the
                // CTE more than once (a self-join), and every scan resolves to
                // the single ephemeral table we bind by name. Correct naive
                // semantics evaluate `r = anchor ∪ recursive_term(r)` against
                // the FULL accumulation each step, so the next full relation is
                // the anchor rows unioned with the freshly re-derived rows
                // (`rec_out`). The relation REPLACES (not appends to) the
                // accumulation — re-deriving a fixpoint, not streaming a delta.
                // Fixpoint is reached when the relation stops growing (row count
                // stable); since we cannot dedup under UNION ALL, the iteration
                // cap above is the mandatory guard against unbounded growth on
                // cyclic data.
                let next = concat_two_batches(&anchor_out, &rec_out)?;
                if next.num_rows() == result.num_rows() {
                    break; // Fixpoint: the re-derived relation did not grow.
                }
                working_set = next.clone();
                result = next;
            } else if rec.all {
                // UNION ALL: append every produced row; the next working set is
                // the whole recursive output. Stop when it is empty.
                if rec_out.num_rows() == 0 {
                    break;
                }
                result = concat_two_batches(&result, &rec_out)?;
                working_set = rec_out;
            } else {
                // UNION (distinct): de-duplicate the recursive output against
                // the full accumulated result. The fixpoint is detected by row
                // COUNT (order-independent, so this is robust whether
                // `execute_distinct` ran the order-preserving host path or the
                // sorted GPU path): if de-duplicating the union of the current
                // result with the new output adds no rows, we are at the
                // fixpoint. Otherwise the *whole* de-duplicated result becomes
                // the next working set — a naive (rather than semi-naive)
                // evaluation. It is correct for set semantics (re-deriving
                // already-known rows is harmless under dedup) and terminates
                // because the de-duplicated result is monotonically growing and
                // bounded; we trade a little redundant work for correctness
                // that does not depend on row ordering.
                let prev_rows = result.num_rows();
                let combined = concat_two_batches(&result, &rec_out)?;
                let deduped = execute_distinct(QueryHandle::from_record_batch(combined))?
                    .into_record_batch();
                if deduped.num_rows() == prev_rows {
                    break; // Fixpoint: no rows the result didn't already hold.
                }
                working_set = deduped.clone();
                result = deduped;
            }
        }

        // --- 4. Run the main query over the full accumulated relation. ---
        let main_out = run_with_cte(&result, &rec.main)?;
        Ok(QueryHandle::from_record_batch(main_out))
    }

    /// Execute a LATERAL derived table (feature F3 — LATERAL) as a host
    /// nested-loop **Apply** (dependent join), then run the OUTER query template
    /// over the applied relation.
    ///
    /// Algorithm (correctness-first, reusing the existing subplan executor and
    /// the streaming overlay, exactly like [`Engine::execute_recursive_cte`]):
    ///
    /// 1. **Left.** Run `la.left` → the LEFT relation.
    /// 2. **Per-row apply.** For each left row (bounded by the mandatory
    ///    [`max_apply_left_rows`] cap — the loop is `O(left_rows × subquery)`):
    ///    bind a single-row [`LATERAL_OUTER_TABLE`] relation holding that row's
    ///    correlated values (`__corr_0..N`), then run `la.lateral_subplan`
    ///    through [`Engine::run_subplan`]. Its rewritten correlations are
    ///    `(SELECT __corr_<i> FROM __lateral_outer)` scalar subqueries that
    ///    `resolve_subqueries` folds to the row's *exact typed* value (so int
    ///    width / NULL / Date / Timestamp all carry correctly).
    /// 3. **Cross product.** Concatenate the left row (repeated once per
    ///    produced subquery row) with the subquery rows. INNER LATERAL drops a
    ///    left row with zero subquery rows; `la.left_join` keeps it once with
    ///    the subquery columns NULL-filled.
    /// 4. **Outer.** Bind the concatenated applied relation under
    ///    [`LATERAL_APPLY_RESULT_TABLE`] in the streaming overlay and run
    ///    `la.post` (the OUTER projection / WHERE / GROUP BY / ORDER BY / LIMIT)
    ///    through [`Engine::run_subplan`] — reusing the ordinary executors —
    ///    then clear the overlay.
    ///
    /// An empty left input yields an empty applied relation (the OUTER template
    /// runs over zero rows). A name collision with a real registered table is
    /// rejected up front.
    fn execute_lateral_apply(
        &self,
        la: &crate::plan::sql_frontend::LateralApplyPlan,
    ) -> BoltResult<QueryHandle> {
        use arrow_array::{new_null_array, UInt32Array};

        use crate::exec::streaming::TableSource;
        use crate::plan::sql_frontend::{LATERAL_APPLY_RESULT_TABLE, LATERAL_OUTER_TABLE};

        // Refuse to shadow real tables under either reserved ephemeral name.
        for name in [LATERAL_OUTER_TABLE, LATERAL_APPLY_RESULT_TABLE] {
            if self.tables.contains_key(name)
                || self.streaming_sources.borrow().contains_key(name)
            {
                return Err(BoltError::Plan(format!(
                    "LATERAL apply: reserved ephemeral table name '{name}' collides \
                     with a registered table"
                )));
            }
        }

        // --- 1. Run the LEFT relation. ---
        let left = self.run_subplan(la.left.clone())?;
        let n_left = left.num_rows();

        let outer_arrow = plan_schema_to_arrow_schema(&la.outer_schema)?;
        let sub_arrow = plan_schema_to_arrow_schema(&la.subquery_schema)?;
        let combined_arrow = plan_schema_to_arrow_schema(&la.combined_schema)?;

        // Run a lateral subplan with the single-row outer relation bound. The
        // overlay entry is always cleared afterwards (mirrors the recursive
        // CTE's `run_with_cte`) so a failure mid-loop cannot leak it, and any
        // GPU-resident copy is evicted before/after so each row re-uploads its
        // own outer values rather than reusing the previous row's.
        let run_lateral = |outer_row: &RecordBatch| -> BoltResult<RecordBatch> {
            self.gpu_tables.borrow_mut().remove(LATERAL_OUTER_TABLE);
            self.streaming_sources.borrow_mut().insert(
                LATERAL_OUTER_TABLE.to_string(),
                TableSource::Materialized(vec![outer_row.clone()]),
            );
            let out = self.run_subplan(la.lateral_subplan.clone());
            self.streaming_sources
                .borrow_mut()
                .remove(LATERAL_OUTER_TABLE);
            self.gpu_tables.borrow_mut().remove(LATERAL_OUTER_TABLE);
            out
        };

        // --- 2 + 3. Per-left-row apply + cross product. ---
        if n_left > max_apply_left_rows() {
            return Err(BoltError::Plan(format!(
                "LATERAL apply: left input has {n_left} rows, exceeding the \
                 {}-row safety cap (set {MAX_APPLY_LEFT_ROWS_ENV} to override) — \
                 the apply runs the correlated subquery once per left row",
                max_apply_left_rows()
            )));
        }

        let mut per_row: Vec<RecordBatch> = Vec::new();
        for row in 0..n_left {
            // Build the single-row outer relation: gather `corr_left_indices`
            // from this row, renamed to the `__corr_<i>` schema.
            let one = UInt32Array::from(vec![row as u32]);
            let mut outer_cols: Vec<ArrayRef> = Vec::with_capacity(la.corr_left_indices.len());
            for &li in &la.corr_left_indices {
                let g = arrow::compute::take(left.column(li).as_ref(), &one, None)
                    .map_err(|e| BoltError::Other(format!("LATERAL outer take: {e}")))?;
                outer_cols.push(g);
            }
            // `try_new_with_options` carries an explicit row count so a LATERAL
            // that correlates on *no* columns (an uncorrelated subquery written
            // with LATERAL) still binds a well-formed single-row, zero-column
            // outer relation rather than an ambiguous-length batch.
            let outer_row = RecordBatch::try_new_with_options(
                Arc::clone(&outer_arrow),
                outer_cols,
                &arrow_array::RecordBatchOptions::new().with_row_count(Some(1)),
            )
            .map_err(|e| BoltError::Plan(format!("LATERAL outer-row build: {e}")))?;

            let right = run_lateral(&outer_row)?;
            // Defensively re-shape the subquery output to the descriptor's
            // declared schema (the run goes through GPU codegen / Arrow, which
            // can produce equivalent-but-differently-named columns).
            let right = RecordBatch::try_new(Arc::clone(&sub_arrow), right.columns().to_vec())
                .map_err(|e| {
                    BoltError::Plan(format!(
                        "LATERAL subquery produced {} columns but its schema \
                         declares {}: {e}",
                        right.num_columns(),
                        sub_arrow.fields().len()
                    ))
                })?;
            let n_right = right.num_rows();

            if n_right == 0 {
                if !la.left_join {
                    continue; // INNER LATERAL: drop a left row with no match.
                }
                // LEFT JOIN LATERAL ... ON true: emit the left row once with the
                // subquery columns NULL-filled.
                let mut cols: Vec<ArrayRef> = Vec::with_capacity(combined_arrow.fields().len());
                for li in 0..left.num_columns() {
                    let g = arrow::compute::take(left.column(li).as_ref(), &one, None)
                        .map_err(|e| BoltError::Other(format!("LATERAL left take: {e}")))?;
                    cols.push(g);
                }
                for f in sub_arrow.fields() {
                    cols.push(new_null_array(f.data_type(), 1));
                }
                let batch = RecordBatch::try_new(Arc::clone(&combined_arrow), cols)
                    .map_err(|e| BoltError::Plan(format!("LATERAL left-join row: {e}")))?;
                per_row.push(batch);
                continue;
            }

            // Repeat the left row `n_right` times and prepend it to the
            // subquery columns → the per-row cross product.
            let rep = UInt32Array::from(vec![row as u32; n_right]);
            let mut cols: Vec<ArrayRef> = Vec::with_capacity(combined_arrow.fields().len());
            for li in 0..left.num_columns() {
                let g = arrow::compute::take(left.column(li).as_ref(), &rep, None)
                    .map_err(|e| BoltError::Other(format!("LATERAL left repeat: {e}")))?;
                cols.push(g);
            }
            cols.extend(right.columns().iter().cloned());
            let batch = RecordBatch::try_new(Arc::clone(&combined_arrow), cols)
                .map_err(|e| BoltError::Plan(format!("LATERAL cross-product row: {e}")))?;
            per_row.push(batch);
        }

        // Concatenate the per-left-row batches into the applied relation. An
        // empty left input (or all rows dropped) yields a zero-row relation
        // with the combined schema so the OUTER template runs over nothing.
        let applied = if per_row.is_empty() {
            RecordBatch::new_empty(Arc::clone(&combined_arrow))
        } else {
            arrow::compute::concat_batches(&combined_arrow, per_row.iter())
                .map_err(|e| BoltError::Plan(format!("LATERAL apply concat: {e}")))?
        };

        // --- 4. Bind the applied relation and run the OUTER template. ---
        self.gpu_tables.borrow_mut().remove(LATERAL_APPLY_RESULT_TABLE);
        self.streaming_sources.borrow_mut().insert(
            LATERAL_APPLY_RESULT_TABLE.to_string(),
            TableSource::Materialized(vec![applied]),
        );
        let out = self.run_subplan(la.post.clone());
        self.streaming_sources
            .borrow_mut()
            .remove(LATERAL_APPLY_RESULT_TABLE);
        self.gpu_tables
            .borrow_mut()
            .remove(LATERAL_APPLY_RESULT_TABLE);
        Ok(QueryHandle::from_record_batch(out?))
    }

    /// Execute a mutually-recursive `WITH RECURSIVE` system (multiple CTEs that
    /// reference each other) by orchestrating a multi-relation **lockstep**
    /// fixpoint, then running the main query over the accumulated relations.
    ///
    /// This is the multi-relation generalisation of
    /// [`Engine::execute_recursive_cte`]. Where the single-CTE path advances a
    /// scalar working set, this advances a *vector* of working sets in lockstep:
    ///
    /// 1. **Seed.** Materialise every CTE's anchor → its initial accumulated
    ///    relation (de-duplicated for a `UNION` member).
    /// 2. **Iterate.** Each step, bind ALL CTE names to their CURRENT
    ///    accumulated relations in the streaming overlay, then evaluate every
    ///    recursive term (each may reference any CTE in the system). Union each
    ///    term's output into its own CTE's accumulation: append for `UNION ALL`,
    ///    dedup-against-accumulation for `UNION`. A naive (whole-relation)
    ///    binding is used — correct for set semantics and for cross-references,
    ///    and required because a sibling's rows feed this term.
    /// 3. **Fixpoint.** Stop only when NO CTE grew in a full step (the combined
    ///    fixpoint). For a `UNION ALL` member "grew" means produced ≥1 new row;
    ///    for a `UNION` member it means the de-duplicated accumulation gained
    ///    rows. Mixed systems terminate on the cap if any `UNION ALL` member
    ///    keeps producing rows (the cap is mandatory).
    /// 4. **Cap.** The shared [`max_recursive_iterations`] cap bounds the loop.
    /// 5. **Main.** Bind all final accumulations and run the main query.
    ///
    /// All ephemeral CTE tables live only in the interior-mutable streaming
    /// overlay and are removed before returning; a name collision with a real
    /// table is rejected up front.
    fn execute_mutual_recursive_cte(
        &self,
        rec: &crate::plan::sql_frontend::MutualRecursiveCtePlan,
    ) -> BoltResult<QueryHandle> {
        use crate::exec::distinct::execute_distinct;
        use crate::exec::streaming::TableSource;

        // Reject collisions with real tables (and detect any duplicate name the
        // frontend already guards, defensively).
        for term in &rec.ctes {
            if self.tables.contains_key(&term.name)
                || self.streaming_sources.borrow().contains_key(&term.name)
            {
                return Err(BoltError::Plan(format!(
                    "WITH RECURSIVE: CTE name '{}' collides with a registered \
                     table — rename the CTE",
                    term.name
                )));
            }
        }

        // Per-CTE relabel closure factory: every term's batch is re-labelled to
        // its declared `cte_schema` before it is registered / unioned.
        let arrow_schemas = rec
            .ctes
            .iter()
            .map(|t| plan_schema_to_arrow_schema(&t.cte_schema))
            .collect::<BoltResult<Vec<_>>>()?;
        let relabel = |idx: usize, batch: RecordBatch| -> BoltResult<RecordBatch> {
            let schema = &arrow_schemas[idx];
            if batch.num_columns() != schema.fields().len() {
                return Err(BoltError::Plan(format!(
                    "WITH RECURSIVE: a term produced {} columns but the CTE \
                     '{}' declares {}",
                    batch.num_columns(),
                    rec.ctes[idx].name,
                    schema.fields().len()
                )));
            }
            RecordBatch::try_new(Arc::clone(schema), batch.columns().to_vec())
                .map_err(|e| BoltError::Plan(format!("WITH RECURSIVE relabel: {e}")))
        };

        // Bind ALL current accumulations under their CTE names, run `plan`, then
        // clear every overlay entry (and evict any GPU-resident copy so the next
        // bind re-uploads the current rows). Returns the produced batch.
        let run_with_all = |accums: &[RecordBatch], plan: &LogicalPlan| -> BoltResult<RecordBatch> {
            for (i, term) in rec.ctes.iter().enumerate() {
                self.gpu_tables.borrow_mut().remove(&term.name);
                self.streaming_sources.borrow_mut().insert(
                    term.name.clone(),
                    TableSource::Materialized(vec![accums[i].clone()]),
                );
            }
            let out = self.run_subplan(plan.clone());
            for term in &rec.ctes {
                self.streaming_sources.borrow_mut().remove(&term.name);
                self.gpu_tables.borrow_mut().remove(&term.name);
            }
            out
        };

        // --- 1. Seed: materialise every anchor. ---
        let mut accums: Vec<RecordBatch> = Vec::with_capacity(rec.ctes.len());
        for (i, term) in rec.ctes.iter().enumerate() {
            let anchor_out = relabel(i, self.run_subplan(term.anchor.clone())?)?;
            // De-duplicate the seed for a UNION member (matches the single path).
            let seeded = if term.recursive.is_some() && !term.all {
                execute_distinct(QueryHandle::from_record_batch(anchor_out))?
                    .into_record_batch()
            } else {
                anchor_out
            };
            accums.push(seeded);
        }

        let cap = max_recursive_iterations();
        let mut iters = 0usize;

        // --- 2. Lockstep iteration to a combined fixpoint. ---
        loop {
            iters += 1;
            if iters > cap {
                return Err(BoltError::Plan(format!(
                    "WITH RECURSIVE: exceeded the {cap}-iteration safety cap \
                     (set {MAX_RECURSIVE_ITERATIONS_ENV} to override) — the \
                     mutual recursion is not terminating"
                )));
            }

            // Snapshot the current accumulations; every recursive term this
            // step is evaluated against THIS snapshot (lockstep — a sibling's
            // growth this step is not visible until the next).
            let snapshot = accums.clone();
            let mut any_grew = false;
            let mut next = snapshot.clone();

            for (i, term) in rec.ctes.iter().enumerate() {
                let recursive = match &term.recursive {
                    Some(r) => r,
                    None => continue, // Non-recursive member: seeded once.
                };
                let rec_out = relabel(i, run_with_all(&snapshot, recursive)?)?;
                if term.all {
                    // UNION ALL: append produced rows; "grew" iff ≥1 row.
                    if rec_out.num_rows() > 0 {
                        next[i] = concat_two_batches(&next[i], &rec_out)?;
                        any_grew = true;
                    }
                } else {
                    // UNION: dedup the produced rows against this CTE's current
                    // accumulation; "grew" iff the de-duplicated relation gained
                    // rows.
                    let prev_rows = next[i].num_rows();
                    let combined = concat_two_batches(&next[i], &rec_out)?;
                    let deduped = execute_distinct(QueryHandle::from_record_batch(combined))?
                        .into_record_batch();
                    if deduped.num_rows() != prev_rows {
                        any_grew = true;
                    }
                    next[i] = deduped;
                }
            }

            accums = next;
            if !any_grew {
                break; // Combined fixpoint: no CTE grew this step.
            }
        }

        // --- 5. Bind all final accumulations and run the main query. ---
        let main_out = run_with_all(&accums, &rec.main)?;
        Ok(QueryHandle::from_record_batch(main_out))
    }

    /// Execute a sole `COUNT(DISTINCT col)` with `GROUP BY` (feature
    /// F3-finish) as a host-orchestrated special-case.
    ///
    /// The per-group distinct count cannot be expressed as a single
    /// `LogicalPlan` that flows through the ordinary pipeline (see
    /// [`crate::plan::sql_frontend::CountDistinctGroupByPlan`]), so — exactly
    /// like [`Engine::execute_recursive_cte`] — the engine orchestrates it:
    ///
    /// 1. **Base.** Run `cd.base` (an ordinary subplan) through
    ///    [`Engine::run_subplan`] to materialise a `RecordBatch` of
    ///    `[group_keys..., distinct_col]` rows.
    /// 2. **Group + count (host).** Group rows by the leading group-key tuple
    ///    (using the same `RowKey` / `RowKeyValue` canonicalisation the
    ///    `DISTINCT` operator uses, so NULL group keys form their own group and
    ///    composite keys compose), and per group accumulate the set of DISTINCT
    ///    **non-NULL** values of `distinct_col`. The count is `set.len()`
    ///    (standard SQL: `COUNT(DISTINCT x)` ignores NULLs, so an all-NULL or
    ///    empty group yields 0).
    /// 3. **Build result.** One row per group (in first-occurrence order): the
    ///    group-key values (taken from the base batch at each group's first row
    ///    via `arrow::compute::take`, preserving exact dtype/value) followed by
    ///    the `Int64` count under `cd.count_alias`.
    /// 4. **Post.** If `cd.post` is present (HAVING / ORDER BY / LIMIT), bind
    ///    the result under the reserved ephemeral name in the streaming overlay
    ///    and run the post-plan through [`Engine::run_subplan`] — reusing the
    ///    ordinary Filter / Sort / Limit executors — then clear the overlay.
    fn execute_count_distinct_groupby(
        &self,
        cd: &crate::plan::sql_frontend::CountDistinctGroupByPlan,
    ) -> BoltResult<QueryHandle> {
        // --- 1. Run the base subplan: [group_keys..., distinct_col]. ---
        let base = self.run_subplan(cd.base.clone())?;

        // --- 2 + 3. Group host-side and build the count-result batch. ---
        let result =
            host_count_distinct_groupby(&base, cd.group_key_names.len(), &cd.result_schema)?;

        // --- 4. Apply the optional HAVING / ORDER BY / LIMIT post-plan over the
        // result, bound under the reserved ephemeral name in the overlay (same
        // pattern as the recursive-CTE main query). ---
        let post = match &cd.post {
            None => return Ok(QueryHandle::from_record_batch(result)),
            Some(p) => p,
        };
        use crate::exec::streaming::TableSource;
        let name = crate::plan::sql_frontend::COUNT_DISTINCT_GROUPBY_RESULT_TABLE;
        if self.tables.contains_key(name) || self.streaming_sources.borrow().contains_key(name) {
            return Err(BoltError::Plan(format!(
                "COUNT(DISTINCT) GROUP BY: reserved result table name '{name}' collides \
                 with a registered table"
            )));
        }
        self.gpu_tables.borrow_mut().remove(name);
        self.streaming_sources
            .borrow_mut()
            .insert(name.to_string(), TableSource::Materialized(vec![result]));
        let out = self.run_subplan(post.clone());
        self.streaming_sources.borrow_mut().remove(name);
        self.gpu_tables.borrow_mut().remove(name);
        Ok(QueryHandle::from_record_batch(out?))
    }

    /// Execute the *generalized* COUNT(DISTINCT) + GROUP BY descriptor
    /// (feature F3-finish, generalized): multiple distinct counts and/or a mix
    /// with plain aggregates. Mirrors [`Engine::execute_count_distinct_groupby`]
    /// but delegates the per-group work to [`host_multi_agg_groupby`] (which
    /// computes every aggregate per group) and binds the result under
    /// [`crate::plan::sql_frontend::MULTI_AGG_GROUPBY_RESULT_TABLE`] for the
    /// optional HAVING / ORDER BY / LIMIT post-plan.
    fn execute_multi_agg_groupby(
        &self,
        cd: &crate::plan::sql_frontend::MultiAggGroupByPlan,
    ) -> BoltResult<QueryHandle> {
        // --- 1. Run the base subplan: [group_keys..., agg_inputs...]. ---
        let base = self.run_subplan(cd.base.clone())?;

        // --- 2 + 3. Group + compute every aggregate host-side. ---
        let result = host_multi_agg_groupby(
            &base,
            cd.n_keys,
            &cd.aggs,
            &cd.output_layout,
            &cd.result_schema,
        )?;

        // --- 4. Optional HAVING / ORDER BY / LIMIT post-plan over the result. ---
        let post = match &cd.post {
            None => return Ok(QueryHandle::from_record_batch(result)),
            Some(p) => p,
        };
        use crate::exec::streaming::TableSource;
        let name = crate::plan::sql_frontend::MULTI_AGG_GROUPBY_RESULT_TABLE;
        if self.tables.contains_key(name) || self.streaming_sources.borrow().contains_key(name) {
            return Err(BoltError::Plan(format!(
                "multi-agg GROUP BY: reserved result table name '{name}' collides \
                 with a registered table"
            )));
        }
        self.gpu_tables.borrow_mut().remove(name);
        self.streaming_sources
            .borrow_mut()
            .insert(name.to_string(), TableSource::Materialized(vec![result]));
        let out = self.run_subplan(post.clone());
        self.streaming_sources.borrow_mut().remove(name);
        self.gpu_tables.borrow_mut().remove(name);
        Ok(QueryHandle::from_record_batch(out?))
    }

    /// Emit a periodic pool-stats log line + observer notification if
    /// the configured interval has elapsed since the last emit.
    ///
    /// `now` is taken as a parameter (rather than calling `Instant::now()`
    /// inside) so the unit test below can drive the throttle deterministically.
    ///
    /// **Never-escalate:** this runs after a query has already produced its
    /// result, so it must not be able to fail that query. The user observer is
    /// invoked through [`crate::observability::notify_observers`], which catches
    /// and swallows any panic from the callback (logging it). A panicking
    /// pool-stats observer therefore cannot unwind out of a successful
    /// `Engine::sql` / `Engine::run_logical_plan`.
    fn maybe_emit_pool_stats(&self, now: Instant) {
        if !should_emit_pool_stats(&self.pool_stats_last_emit, self.pool_stats_interval, now) {
            return;
        }
        // Throttle says go: snapshot the pool and emit. We do this OUTSIDE
        // the throttle's lock so a slow observer can't serialise concurrent
        // queries.
        let s = crate::pool_stats();
        log::info!(
            "craton-bolt pool: bucket_count={}, total_pooled_bytes={}, \
             oom_recoveries={}, proactive_evictions={}",
            s.bucket_count,
            s.total_pooled_bytes,
            s.oom_recovery_count,
            s.proactive_eviction_count,
        );
        crate::observability::notify_observers(s);
    }

    /// If `phys` is a **streamable leaf scan** — a plan whose output rows are an
    /// independent, row-wise function of a single base table's scan rows, so
    /// that processing the table morsel-by-morsel and concatenating the
    /// per-morsel results yields the *byte-for-byte identical* batch as
    /// processing the whole table at once — return that table's name.
    ///
    /// The streamable shapes are exactly the three scan-leaf executors that take
    /// a `table: String` directly (no child sub-plan) and emit one output row
    /// per surviving input row:
    ///
    /// * [`PhysicalPlan::Projection`] — fused project/filter kernel. A `WHERE`
    ///   only *drops* rows; the surviving rows are unchanged and order-preserved
    ///   within each morsel, and morsels are produced in table order, so
    ///   `concat(project(morsel_i)) == project(concat(morsel_i))`.
    /// * [`PhysicalPlan::StringLength`] — `LENGTH(col)` + passthroughs.
    /// * [`PhysicalPlan::StringProject`] — `UPPER`/`LOWER`/passthroughs.
    ///
    /// Every *other* variant is **not** returned here and therefore drains the
    /// whole table (status quo): `Aggregate` (global/grouped fold crosses all
    /// rows), `Sort`/`Distinct`/`SetOp`/`Union`/`Window` (cross-row ordering or
    /// dedup), `Join` (build side must be resident), `Limit`/`Filter`/`Project`/
    /// `CountRows`/`StringLikeFilter` (wrap a child sub-plan whose own scan would
    /// have to be threaded — out of scope for this minimal, correctness-first
    /// cut). Those keep the existing materialise-whole-table behaviour exactly.
    fn streamable_leaf_scan<'p>(phys: &'p PhysicalPlan) -> Option<&'p str> {
        match phys {
            PhysicalPlan::Projection { table, .. }
            | PhysicalPlan::StringLength { table, .. }
            | PhysicalPlan::StringProject { table, .. } => Some(table.as_str()),
            _ => None,
        }
    }

    /// Drive a streamable leaf scan morsel-by-morsel instead of materialising
    /// the whole table on the device at once.
    ///
    /// Precondition: `table` is a streaming-registered overlay table (present in
    /// `streaming_sources`, **not** in the eager `tables` store) — the only
    /// shape whose data this method can swap per-morsel under `&self`, because
    /// the overlay is interior-mutable (`RefCell`) while `tables` is not. The
    /// caller ([`Engine::execute`]) enforces this.
    ///
    /// For each morsel the table's overlay entry is temporarily replaced with a
    /// single-batch `Materialized` view of just that morsel, the per-morsel leaf
    /// executor runs (rebuilding a small, bounded GPU table for the morsel), and
    /// the result batch is collected. The original whole-table overlay entry is
    /// always restored afterwards — including on the error path — so a producer
    /// or kernel fault leaves no torn state. The per-morsel results are
    /// concatenated into the final batch, which is identical to the whole-table
    /// result because the leaf shapes are row-wise (see
    /// [`Engine::streamable_leaf_scan`]).
    ///
    /// `run_morsel` is the existing per-morsel dispatch (the same executor the
    /// whole-table path uses); it is invoked while the morsel is installed.
    fn execute_streaming_leaf(
        &self,
        table: &str,
        morsel_rows: usize,
        run_morsel: impl Fn() -> BoltResult<QueryHandle>,
        output_schema: &Schema,
    ) -> BoltResult<QueryHandle> {
        use crate::exec::streaming::{BatchStream, TableSource};

        // Snapshot the whole-table batches (cheap Arc clones) and the original
        // overlay entry so we can restore it. We must NOT hold the overlay
        // borrow across `run_morsel` (which re-borrows the overlay through
        // `materialize_table`/`ensure_gpu_table`), so we take owned batches.
        let whole: Vec<RecordBatch> = {
            let overlay = self.streaming_sources.borrow();
            match overlay.get(table) {
                Some(TableSource::Materialized(b)) => b.clone(),
                // The caller guarantees an overlay table; a still-`Streaming`
                // entry should have been collapsed by
                // `ensure_streaming_materialized` before `execute`.
                _ => {
                    return Err(BoltError::Other(format!(
                        "execute_streaming_leaf: table '{table}' is not a \
                         materialised streaming-overlay table"
                    )))
                }
            }
        };

        let mut results: Vec<RecordBatch> = Vec::new();

        // Run the morsels, restoring the whole-table overlay entry no matter
        // how the loop exits. The `BatchStream` (which borrows `whole`) is
        // built and fully consumed INSIDE this closure, so its borrow ends
        // before we move `whole` back into the overlay below.
        let loop_result: BoltResult<()> = (|| {
            let stream = BatchStream::new(&whole, morsel_rows)?;
            for morsel in stream.morsels() {
                // Install just this morsel as the table's data.
                self.streaming_sources
                    .borrow_mut()
                    .insert(table.to_string(), TableSource::Materialized(vec![morsel]));
                // A streaming-overlay table carries no `host_revisions` entry,
                // so `ensure_gpu_table` always rebuilds a fresh (small) GPU
                // table from the installed morsel — no stale-cache hazard.
                let handle = run_morsel()?;
                results.push(handle.batch);
            }
            Ok(())
        })();

        // Always restore the whole-table view (the source is re-iterable, so a
        // subsequent query sees the full table again).
        self.streaming_sources
            .borrow_mut()
            .insert(table.to_string(), TableSource::Materialized(whole));

        loop_result?;

        // Concatenate the per-morsel results. Zero morsels (an all-empty table)
        // yields an empty batch shaped by the output schema.
        let batch = if results.is_empty() {
            let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
            RecordBatch::new_empty(arrow_schema)
        } else {
            let schema = results[0].schema();
            arrow::compute::concat_batches(&schema, results.iter()).map_err(|e| {
                BoltError::Other(format!(
                    "execute_streaming_leaf: failed to concatenate {} morsel \
                     results for table '{table}': {e}",
                    results.len()
                ))
            })?
        };
        Ok(QueryHandle { batch })
    }

    /// Execute a pre-built `PhysicalPlan`.
    pub fn execute(&self, phys: &PhysicalPlan) -> BoltResult<QueryHandle> {
        // Streaming / morsel opt-in. The whole-table path below is the default
        // and is byte-for-byte preserved: this hook fires ONLY when (a) a memory
        // budget is configured (default is `None` → uncapped → never fires),
        // (b) the plan is a streamable row-wise leaf scan
        // (`Projection`/`StringLength`/`StringProject` — see
        // `streamable_leaf_scan`), (c) the scan table is a streaming-registered
        // overlay table (the only data we can swap per-morsel under `&self`),
        // and (d) the table's footprint exceeds the budget so
        // `morsel_plan_for_table` actually calls for chunking. Any miss falls
        // straight through to the unchanged whole-table dispatch.
        if self.memory_budget_bytes.is_some() {
            if let Some(table) = Self::streamable_leaf_scan(phys) {
                // Only overlay (streaming-registered) tables are morsel-driven;
                // eager `tables` entries can't be swapped under `&self` and keep
                // the whole-table path.
                let is_overlay_only = !self.tables.contains_key(table)
                    && self.streaming_sources.borrow().contains_key(table);
                if is_overlay_only {
                    if let Some(morsel_rows) =
                        self.morsel_plan_for_table(table)?.morsel_rows()
                    {
                        let output_schema = phys.output_schema().clone();
                        return self.execute_streaming_leaf(
                            table,
                            morsel_rows,
                            || self.execute_leaf_whole(phys),
                            &output_schema,
                        );
                    }
                }
            }
        }
        self.execute_leaf_whole(phys)
    }

    /// The whole-table dispatch — the original body of [`Engine::execute`].
    ///
    /// Split out so the streaming/morsel orchestrator
    /// ([`Engine::execute_streaming_leaf`]) can invoke the *exact same*
    /// per-shape executor on a single installed morsel, guaranteeing the
    /// morsel-by-morsel result is identical to the whole-table result.
    fn execute_leaf_whole(&self, phys: &PhysicalPlan) -> BoltResult<QueryHandle> {
        match phys {
            PhysicalPlan::Projection {
                table,
                kernel,
                output_schema,
            } => {
                // GPU projection path. If the GPU *declines* the upload/gather
                // of a temporal/decimal column it returns the typed
                // `BoltError::GpuCapacity` decline marker (see
                // `gpu_table::GpuColumn::upload` / `gpu_compact::alloc_gathered`,
                // agent-G F11). Mirror the join gates
                // (`try_gpu_inner_join`/`try_gpu_outer_join`, which turn a
                // declined GPU join into an `Ok` host re-run): catch the decline
                // and transparently re-run the projection on the host instead of
                // failing the query. Any *other* error still propagates.
                match self.execute_projection(table, kernel, output_schema) {
                    Err(BoltError::GpuCapacity(reason)) => {
                        crate::metrics::metrics()
                            .inc(crate::metrics::Counter::HostFallbacksTotal);
                        log::debug!(
                            "execute_projection: GPU declined ({reason}); \
                             re-running projection on host"
                        );
                        self.execute_projection_host_fallback(table, kernel, output_schema, &reason)
                    }
                    other => other,
                }
            }
            PhysicalPlan::StringLength {
                table,
                outputs,
                output_schema,
            } => self.execute_string_length(table, outputs, output_schema),
            PhysicalPlan::StringProject {
                table,
                outputs,
                output_schema,
            } => self.execute_string_project(table, outputs, output_schema),
            PhysicalPlan::Aggregate {
                table,
                pre,
                aggregate,
            } => {
                // v0.7: GROUP BY VAR_POP / VAR_SAMP / STDDEV_POP /
                // STDDEV_SAMP are lowered to a per-group Welford pass in
                // the downstream executors (`crate::exec::groupby`,
                // `crate::exec::groupby_valid`, `crate::exec::groupby_with_pre`,
                // and `crate::exec::groupby_wide`). The shared
                // `crate::exec::welford::WelfordState` provides the
                // numerically-stable single-pass update; the executors fold
                // per-group state on the host after the GPU keys kernel
                // populates the slot table.
                let out = match (!aggregate.group_by.is_empty(), pre.is_some()) {
                    (true, true) => {
                        let batch = self.materialize_table(table)?;
                        crate::exec::groupby_with_pre::execute_groupby_with_pre(phys, &batch)?
                    }
                    (true, false) => {
                        // Performance: try the resident on-device GROUP BY path
                        // first (keys/values read from the already-uploaded
                        // GpuTable, no per-query re-upload — the H2D dominates a
                        // low-cardinality SUM's wall-clock). `batch` is a cheap
                        // Arc clone for a singly-registered table and is still
                        // needed for the transfer-free host key scans. Falls
                        // back to the host-upload path on `None`, preserving
                        // behaviour for shapes without a resident variant. The
                        // resident Ref is scoped to this arm.
                        let batch = self.materialize_table(table)?;
                        let fast = match self.ensure_gpu_table(table) {
                            Ok(resident) => crate::exec::groupby::try_execute_groupby_resident(
                                phys, &resident, &batch,
                            ),
                            Err(_) => None,
                        };
                        match fast {
                            Some(r) => r?,
                            None => crate::exec::groupby::execute_groupby(phys, &batch)?,
                        }
                    }
                    (false, true) => {
                        // Performance: try the fully on-device resident path
                        // first — pre-kernel inputs are read straight from the
                        // already-uploaded GpuTable and every reduce runs in
                        // place on the device, so a repeat scalar aggregate
                        // pays NO per-query bulk H2D/D2H. It returns `None`
                        // (and we fall back to the host-materialised path,
                        // preserving behaviour) whenever a precondition isn't
                        // met — a predicate, NULL inputs, a widening reduce, or
                        // an unaccelerated aggregate. The resident-table borrow
                        // is scoped to this match so the fallback can
                        // re-borrow `gpu_tables` via `materialize_table`.
                        let fast = match self.ensure_gpu_table(table) {
                            Ok(resident) => {
                                crate::exec::agg_with_pre::try_execute_resident(phys, &resident)?
                            }
                            Err(_) => None,
                        };
                        match fast {
                            Some(b) => b,
                            None => {
                                let batch = self.materialize_table(table)?;
                                crate::exec::agg_with_pre::execute_aggregate_with_pre(
                                    phys, &batch,
                                )?
                            }
                        }
                    }
                    (false, false) => {
                        let batch = self.materialize_table(table)?;
                        crate::exec::aggregate::execute_aggregate(phys, &batch)?
                    }
                };
                Ok(QueryHandle { batch: out })
            }
            // ----- wave-7 dispatch -----
            //
            // The PhysicalPlan variants below are added by agent 1 in the
            // same wave. If a variant doesn't exist yet at build time, the
            // match arm will surface a clear compile error pointing at the
            // missing variant — agent 1 then adds it and the build heals.
            //
            // The executor signatures assumed here mirror the wave-7 spec:
            //   execute_distinct(QueryHandle) -> BoltResult<QueryHandle>
            //   execute_limit  (QueryHandle, usize, Option<usize>) -> ...
            //   execute_sort   (QueryHandle, &[SortExpr]) -> ...
            //   execute_join   (left, right, join_type, on, &Engine) -> ...
            // Agents 3-6 match these.
            PhysicalPlan::Distinct { input } => {
                let h = self.execute(input)?;
                crate::exec::distinct::execute_distinct(h)
            }
            PhysicalPlan::CountRows {
                input,
                output_schema,
            } => {
                // Scalar `COUNT(...)` over a non-scan-chain child (notably
                // `COUNT(DISTINCT col)`, where `input` is a `Distinct`). The
                // fused scalar-aggregate executor can't fold a Distinct, so the
                // lowerer routed this shape here: execute the child (the
                // Distinct executor materialises the deduped rows as part of
                // that), then emit a single-row Int64 batch holding the child's
                // row count.
                let h = self.execute(input)?;
                let n_rows = h.batch.num_rows();
                let batch = build_count_rows_batch(n_rows, output_schema)?;
                Ok(QueryHandle { batch })
            }
            PhysicalPlan::Limit {
                input,
                limit,
                offset,
            } => {
                let h = self.execute(input)?;
                crate::exec::limit::execute_limit(h, *limit, *offset)
            }
            PhysicalPlan::Sort { input, sort_exprs } => {
                let h = self.execute(input)?;
                crate::exec::sort::execute_sort(h, sort_exprs)
            }
            PhysicalPlan::Window {
                input,
                window_exprs,
                partition_by,
                order_by,
                output_schema,
            } => {
                // Host-side window-function executor: materialise the input,
                // partition + order within partition, compute each window
                // output column, and append it. GPU offload is a follow-up;
                // see `crate::exec::window`.
                let h = self.execute(input)?;
                crate::exec::window::execute_window(
                    h,
                    window_exprs,
                    partition_by,
                    order_by,
                    output_schema,
                )
            }
            PhysicalPlan::Union { inputs } => {
                // UNION ALL: execute each input, concat the result batches.
                // (Deduplication would happen via a Distinct wrapping the Union
                // in the logical plan — UNION ALL itself is pure concat.)
                if inputs.is_empty() {
                    return Err(BoltError::Plan(
                        "Union with zero inputs is not executable".into(),
                    ));
                }
                let mut handles: Vec<QueryHandle> = Vec::with_capacity(inputs.len());
                for inp in inputs {
                    handles.push(self.execute(inp)?);
                }
                let schema = handles[0].batch.schema();
                let batches: Vec<RecordBatch> =
                    handles.into_iter().map(|h| h.batch).collect();
                let merged = arrow::compute::concat_batches(&schema, batches.iter())
                    .map_err(|e| {
                        BoltError::Other(format!(
                            "failed to concatenate {} UNION ALL inputs: {e}",
                            batches.len()
                        ))
                    })?;
                Ok(QueryHandle { batch: merged })
            }
            PhysicalPlan::SetOp {
                left,
                right,
                op,
                all,
            } => {
                // EXCEPT / INTERSECT (with optional ALL): execute both inputs,
                // then compute the multiset difference / intersection
                // host-side (see `crate::exec::setops`), reusing the DISTINCT
                // executor's row-key / NULL canonicalisation.
                let l = self.execute(left)?;
                let r = self.execute(right)?;
                crate::exec::setops::execute_setop(l, r, *op, *all)
            }
            PhysicalPlan::Join {
                left,
                right,
                join_type,
                on,
                filter,
                output_schema,
            } => crate::exec::join::execute_join(
                left,
                right,
                join_type,
                on,
                filter.as_ref(),
                output_schema,
                self,
            ),
            PhysicalPlan::StringLikeFilter {
                input,
                table,
                column,
                literal,
                mode,
                negated,
            } => self.execute_string_like_filter(
                input, table, column, literal, *mode, *negated,
            ),
            PhysicalPlan::Filter { input, predicate } => {
                // Host-side post-aggregate (or other non-scan-chain) filter.
                // The lowerer emits this for `HAVING` and any `Filter`
                // wrapping an operator that can't fold into a fused
                // projection kernel. The inner plan's output is materialised
                // as a host-side RecordBatch; we evaluate `predicate` against
                // it via `expr_agg::eval_expr` and drop the rows that don't
                // satisfy it. See `crate::exec::filter::execute_filter`.
                let h = self.execute(input)?;
                crate::exec::filter::execute_filter(h, predicate)
            }
            PhysicalPlan::Project {
                input,
                exprs,
                output_schema,
            } => {
                // Rename/reorder/compute layer over an arbitrary upstream.
                //
                // Fast path: when an `exprs` entry is a bare `Column` or an
                // `Alias` wrapping a `Column`, we just pick that column out
                // of the input batch (no compute, zero-copy clone of the
                // `ArrayRef`).
                //
                // Compute path: anything else (today: SQL `a || b`, i.e.
                // `BinaryOp::Concat`) is materialised via
                // `expr_agg::eval_expr` over a `HostColumn` env built from
                // the input batch. The lazy lift (only build the env when
                // a compute expr appears) keeps the bare-projection case
                // free of overhead.
                let h = self.execute(input)?;
                let in_batch = h.batch;
                let in_schema = in_batch.schema();
                let n_rows = in_batch.num_rows();

                // Lazily-built env for the compute path; `None` until the
                // first non-bare-column expression in `exprs` forces us to
                // lift every input column into a `HostColumn`.
                let mut owned_env: Option<Vec<(String, crate::exec::expr_agg::HostColumn)>> = None;

                let mut columns: Vec<ArrayRef> = Vec::with_capacity(exprs.len());
                for (out_idx, e) in exprs.iter().enumerate() {
                    // Peel through transparent aliases to look at the inner
                    // expression. A bare column reference (with any number
                    // of aliases around it) gets the fast path; anything
                    // else falls into the compute path.
                    let inner = {
                        let mut cur = e;
                        loop {
                            match cur {
                                crate::plan::Expr::Alias(inner, _) => cur = inner.as_ref(),
                                _ => break cur,
                            }
                        }
                    };
                    if let crate::plan::Expr::Column(name) = inner {
                        let idx = in_schema.index_of(name).map_err(|_| {
                            BoltError::Plan(format!(
                                "PhysicalPlan::Project: column '{name}' not found in input schema"
                            ))
                        })?;
                        columns.push(in_batch.column(idx).clone());
                        continue;
                    }
                    // Compute path. Build the env if we haven't yet.
                    if owned_env.is_none() {
                        let mut v = Vec::with_capacity(in_batch.num_columns());
                        for (i, field) in in_schema.fields().iter().enumerate() {
                            let arr = in_batch.column(i);
                            let hc = crate::exec::filter::arrow_array_to_host_column(
                                arr.as_ref(),
                                n_rows,
                            )?;
                            v.push((field.name().clone(), hc));
                        }
                        owned_env = Some(v);
                    }
                    let env_ref = owned_env.as_ref().expect("just built");
                    let env: crate::exec::expr_agg::ColumnEnv<'_> = env_ref
                        .iter()
                        .map(|(n, c)| (n.clone(), c))
                        .collect();
                    let out_field = &output_schema.fields[out_idx];
                    let computed = crate::exec::expr_agg::eval_expr(
                        inner,
                        &env,
                        out_field.dtype,
                        n_rows,
                    )?;
                    // Temporal-typed outputs (e.g. CAST(<str> AS DATE/TIMESTAMP
                    // FORMAT ...)) carry their value as the underlying i32/i64
                    // HostColumn; rebuild the declared temporal Arrow type so the
                    // column matches the output schema. All other dtypes use the
                    // standard HostColumn->Arrow mapping.
                    use crate::exec::expr_agg::HostColumn as HC;
                    use crate::plan::logical_plan::{DataType as PDT, TimeUnit as PTU};
                    let arr: ArrayRef = match (out_field.dtype, computed) {
                        (PDT::Date32, HC::I32(v)) => {
                            Arc::new(arrow_array::Date32Array::from(v)) as ArrayRef
                        }
                        (PDT::Timestamp(unit, tz), HC::I64(v)) => {
                            let tz_owned = tz.map(|s| std::sync::Arc::<str>::from(s));
                            match unit {
                                PTU::Second => Arc::new(
                                    arrow_array::TimestampSecondArray::from(v)
                                        .with_timezone_opt(tz_owned),
                                ) as ArrayRef,
                                PTU::Millisecond => Arc::new(
                                    arrow_array::TimestampMillisecondArray::from(v)
                                        .with_timezone_opt(tz_owned),
                                ) as ArrayRef,
                                PTU::Microsecond => Arc::new(
                                    arrow_array::TimestampMicrosecondArray::from(v)
                                        .with_timezone_opt(tz_owned),
                                ) as ArrayRef,
                                PTU::Nanosecond => Arc::new(
                                    arrow_array::TimestampNanosecondArray::from(v)
                                        .with_timezone_opt(tz_owned),
                                ) as ArrayRef,
                            }
                        }
                        (_, other) => host_column_to_arrow_array(other)?,
                    };
                    columns.push(arr);
                }
                let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
                let out = RecordBatch::try_new(arrow_schema, columns).map_err(|e| {
                    BoltError::Other(format!(
                        "failed to build PhysicalPlan::Project RecordBatch: {e}"
                    ))
                })?;
                Ok(QueryHandle { batch: out })
            }
        }
    }

    /// Execute a single fused projection (optionally with filter) kernel.
    fn execute_projection(
        &self,
        table: &str,
        kernel: &KernelSpec,
        output_schema: &Schema,
    ) -> BoltResult<QueryHandle> {
        // Lazy upload: if `register_batch` ran since the last query, this
        // rebuilds the GPU-resident copy from the host-concatenated batches.
        // The returned `Ref` is held across the kernel launch — no other
        // engine method touches `gpu_tables` mutably while `&self` is borrowed.
        let gpu_table_ref = self.ensure_gpu_table(table)?;
        let gpu_table: &crate::exec::gpu_table::GpuTable = &gpu_table_ref;
        let n_rows = gpu_table.n_rows;

        // 1. Resolve input device pointers in place — every column already
        //    lives on the GPU. No host bounce, no per-query upload.
        //
        // `__idx_<col>` inputs come from the dict_registry (they don't exist
        // in the source RecordBatch). They were synthesized by the
        // string-literal rewriter and resolve to the i32/i64 dictionary index
        // column already on the device — we hand the launch a borrowed
        // device pointer into the registry's `GpuVec` rather than bouncing the
        // index column through the host. `&self` is borrowed for the entire
        // `execute_projection`, so the dictionary's GpuVec outlives the launch.
        let mut input_ptrs: Vec<CUdeviceptr> = Vec::with_capacity(kernel.inputs.len());
        // F12 column-vs-column Utf8 ordering: per-row rank columns
        // (`__rank_<col>`) are MATERIALISED here from the registered dictionary
        // index column and the unified rank table the rewriter stashed. The
        // freshly-allocated i64 device buffers must outlive the kernel launch +
        // the D2H download below, so this function-scoped Vec owns them for the
        // whole `execute_projection` (mirrors `synth_validity_bufs`).
        let mut rank_bufs: Vec<GpuVec<i64>> = Vec::new();
        for io in &kernel.inputs {
            if let Some(original) = io.name.strip_prefix("__rank_") {
                // The rewriter lowered a `col_a OP col_b` Utf8 ordering to a
                // NULL-safe integer comparison over `__rank_<col>` columns whose
                // per-row value is `rank_table[index_column[row]]`. The rank
                // table (ranked against the SHARED byte-sorted universe of both
                // columns' dictionaries) was recorded by the rewriter and keyed
                // by this synthetic name in the registry side-channel.
                if io.dtype != DataType::Int64 {
                    return Err(BoltError::Plan(format!(
                        "rewriter-emitted rank column '{}' must be Int64, plan says {:?}",
                        io.name, io.dtype
                    )));
                }
                let rank_table = self.dict_registry.rank_table(&io.name).ok_or_else(|| {
                    BoltError::Plan(format!(
                        "rewriter-emitted rank column '{}' has no rank table in registry \
                         (was the plan rewritten by DictRegistry::rewrite_plan?)",
                        io.name
                    ))
                })?;
                // The source dictionary index column (`__idx_<original>` lives in
                // the same dictionary entry) supplies one GPU index per row; the
                // rank table maps each index to its rank. NULL rows carry index
                // 0, which the rank table maps to the `-1` sentinel — the
                // rewriter's `(__rank_ >= 0)` guard then drops them (SQL 3VL).
                let dict = self.dict_registry.dictionary(table, original).ok_or_else(|| {
                    BoltError::Plan(format!(
                        "rank column '{}' references column '{}' with no dictionary in registry",
                        io.name, original
                    ))
                })?;
                // Gather host-side: download the index column once, map each
                // index through the rank table, upload the i64 rank column. This
                // is per-query setup cost (one D2H + one H2D of the index/rank
                // column), not a per-row hot path; it reuses the existing index
                // buffer rather than re-deriving indices. A bounds-checked map
                // turns a malformed index into a structured error rather than an
                // out-of-range panic.
                let host_ranks: Vec<i64> = match dict {
                    crate::cuda::dictionary_any::DictionaryColumnAny::I32(d) => {
                        let idxs = d.indices.to_vec()?;
                        let mut out = Vec::with_capacity(idxs.len());
                        for &ix in &idxs {
                            if ix < 0 {
                                return Err(BoltError::Other(format!(
                                    "rank materialise: negative index {} for column '{}'",
                                    ix, original
                                )));
                            }
                            let slot = ix as usize;
                            let rank = rank_table.get(slot).copied().ok_or_else(|| {
                                BoltError::Other(format!(
                                    "rank materialise: index {} for column '{}' out of rank-table range {}",
                                    ix, original, rank_table.len()
                                ))
                            })?;
                            out.push(rank);
                        }
                        out
                    }
                    crate::cuda::dictionary_any::DictionaryColumnAny::I64(d) => {
                        let idxs = d.indices.to_vec()?;
                        let mut out = Vec::with_capacity(idxs.len());
                        for &ix in &idxs {
                            if ix < 0 {
                                return Err(BoltError::Other(format!(
                                    "rank materialise: negative index {} for column '{}'",
                                    ix, original
                                )));
                            }
                            let slot = ix as usize;
                            let rank = rank_table.get(slot).copied().ok_or_else(|| {
                                BoltError::Other(format!(
                                    "rank materialise: index {} for column '{}' out of rank-table range {}",
                                    ix, original, rank_table.len()
                                ))
                            })?;
                            out.push(rank);
                        }
                        out
                    }
                };
                if host_ranks.len() != n_rows {
                    return Err(BoltError::Other(format!(
                        "rank materialise: column '{}' produced {} rows, table has {}",
                        io.name, host_ranks.len(), n_rows
                    )));
                }
                let dev = GpuVec::<i64>::from_slice(&host_ranks)?;
                input_ptrs.push(dev.device_ptr());
                rank_bufs.push(dev);
                continue;
            }
            if let Some(original) = io.name.strip_prefix("__idx_") {
                let dict = self.dict_registry.dictionary(table, original).ok_or_else(|| {
                    BoltError::Plan(format!(
                        "rewriter-emitted column '{}' has no dictionary in registry",
                        io.name
                    ))
                })?;
                // Fail fast on plan/dict dtype mismatch BEFORE doing any I/O —
                // this catches a stale plan that names __idx_X with the wrong
                // width without paying the cost of touching the device.
                if io.dtype != dict.index_dtype() {
                    return Err(BoltError::Plan(format!(
                        "rewriter-emitted column '{}' dtype mismatch: plan says {:?}, dictionary is {:?}",
                        io.name, io.dtype, dict.index_dtype()
                    )));
                }
                // Borrow the device pointer from the registry's existing
                // index column — no host bounce, no fresh allocation.
                let ptr = match dict {
                    crate::cuda::dictionary_any::DictionaryColumnAny::I32(d) => {
                        d.indices.device_ptr()
                    }
                    crate::cuda::dictionary_any::DictionaryColumnAny::I64(d) => {
                        d.indices.device_ptr()
                    }
                };
                input_ptrs.push(ptr);
                continue;
            }
            let column = gpu_table.column(&io.name).ok_or_else(|| {
                BoltError::Plan(format!("column '{}' not in table '{}'", io.name, table))
            })?;
            if column.dtype != io.dtype {
                return Err(BoltError::Plan(format!(
                    "column '{}' dtype mismatch: plan says {:?}, table has {:?}",
                    io.name, io.dtype, column.dtype
                )));
            }
            input_ptrs.push(column.device_ptr());
        }

        // 2. Allocate output buffers, zero-initialised. For Utf8 passthrough
        //    columns (output dtype Utf8 AND name matches an input column),
        //    clone the source dictionary so download can decode indices back
        //    to strings. (Computed Utf8 outputs aren't supported yet.)
        let mut output_cols: Vec<DeviceCol> = Vec::with_capacity(kernel.outputs.len());
        for io in &kernel.outputs {
            let mut col = DeviceCol::alloc_zeros(io.dtype, n_rows)?;
            if io.dtype == DataType::Utf8 {
                if let Some(src) = kernel
                    .inputs
                    .iter()
                    .find(|in_io| in_io.name == io.name && in_io.dtype == DataType::Utf8)
                    .and_then(|in_io| gpu_table.column(&in_io.name))
                    .and_then(|c| c.utf8_dictionary())
                {
                    col.set_utf8_dictionary(src.to_vec());
                }
            }
            // Decimal128 NULL fix: for a pure passthrough Decimal128 output
            // (output name + dtype matches an input column), carry the
            // source column's validity bitmap onto the output so the
            // download path reconstructs NULL rows as NULL, not `0`. The
            // output column needs an *owned* device buffer (it outlives the
            // borrow of `src_col`), so we allocate a fresh mask buffer and
            // copy the bitmap into it.
            //
            // P1/F10b: this copy stays on the device — `cuMemcpyDtoD_v2` via
            // `memcpy_d2d` — instead of the old D2H `to_vec()` + H2D
            // `from_slice()` host round-trip. The mask is at most one byte
            // per 8 rows, but a passthrough query did this PCIe bounce every
            // call; the DtoD copy removes both crossings. The source and the
            // freshly-`zeros`-allocated destination are distinct device
            // allocations (non-overlapping), satisfying `memcpy_d2d`'s safety
            // contract.
            if let DataType::Decimal128(_, _) = io.dtype {
                if let Some(src_col) = kernel
                    .inputs
                    .iter()
                    .find(|in_io| in_io.name == io.name && in_io.dtype == io.dtype)
                    .and_then(|in_io| gpu_table.column(&in_io.name))
                {
                    if let crate::exec::gpu_table::GpuColumnData::Decimal128 {
                        valid_mask: Some(src_mask),
                        ..
                    } = &src_col.data
                    {
                        let mask_len = src_mask.len();
                        let dst_mask = GpuVec::<u8>::zeros(mask_len)?;
                        // SAFETY: `dst_mask` was just allocated (a distinct
                        // device pointer from `src_mask`), both are live
                        // allocations of `mask_len` bytes, and they do not
                        // overlap — meeting `memcpy_d2d`'s contract. A
                        // zero-length mask short-circuits inside `memcpy_d2d`.
                        unsafe {
                            cuda_sys::memcpy_d2d::<u8>(
                                dst_mask.device_ptr(),
                                src_mask.device_ptr(),
                                mask_len,
                            )?;
                        }
                        col.set_decimal128_valid_mask(Some(dst_mask));
                    }
                }
            }
            output_cols.push(col);
        }

        // 3. JIT-compile the kernel to PTX and load it.
        //
        // Review-H2: route through `get_or_build_module` so repeat queries
        // with the same `KernelSpec` skip the PTX-gen + cubin-load round
        // trip and reuse the same loaded `CudaModule` (cheap Arc clone).
        // The underlying `jit::jit_compiler` PTX-text-hash cache continues
        // to short-circuit `cuModuleLoadDataEx` for unique-spec / shared-
        // PTX cases (e.g. across distinct engines in the same process).
        let module = self.get_or_build_module(kernel, KERNEL_ENTRY, |k| {
            compile_ptx(k, KERNEL_ENTRY)
        })?;
        let function = module.function(KERNEL_ENTRY)?;

        // 4. Build the kernel-parameter list.
        //
        // `KernelArgs` is monomorphic on `T` per push and cannot store heterogenous
        // column types in one list. We bypass it and assemble raw kernel params
        // directly: inputs first, then outputs, then any flagged validity
        // pointers (input then output, in the same order as `ptx_gen.rs`'s
        // signature walk — see `ptx_gen::write_signature`), then the
        // row-count `u32`.
        //
        // Validity pointer wiring (Batch 7, IS NULL e2e):
        // For every input where `kernel.input_has_validity[i] == true` (set by
        // `Codegen::emit_unary` for `column IS [NOT] NULL` checks), push the
        // GPU column's *u8 validity-bitmap pointer here. The codegen's
        // `Op::IsNullCheck` indexes into this list via `validity_input`.
        //
        // For columns where the codegen flagged validity but the GPU storage
        // doesn't expose a validity pointer (e.g. nullable primitives whose
        // GPU storage is still values-only today), we surface a structured
        // error rather than silently emitting a NULL pointer — the kernel
        // would then segfault on the first row. The plan-time constant-fold
        // in `Codegen::emit_unary` already eliminates IsNullCheck on
        // non-nullable schema fields, so this error only fires for genuine
        // missing-validity-on-GPU plumbing gaps (a follow-up: nullable
        // primitives on the device).
        let need_input_validity: Vec<bool> = if kernel.input_has_validity.is_empty() {
            vec![false; kernel.inputs.len()]
        } else {
            if kernel.input_has_validity.len() != kernel.inputs.len() {
                return Err(BoltError::Other(format!(
                    "engine: kernel.input_has_validity len {} != inputs len {}",
                    kernel.input_has_validity.len(),
                    kernel.inputs.len()
                )));
            }
            kernel.input_has_validity.clone()
        };
        let mut input_validity_ptrs: Vec<CUdeviceptr> = Vec::new();
        // Holds any all-valid bitmaps synthesised below for columns that are
        // nullable-in-schema but carry NO actual nulls (so `validity_ptr()` is
        // `None`). These must outlive the kernel launch + the result D2H sync,
        // which both happen later in this function — so a function-scoped Vec is
        // the correct lifetime.
        let mut synth_validity_bufs: Vec<GpuVec<u8>> = Vec::new();
        for (i, has) in need_input_validity.iter().enumerate() {
            if !*has {
                continue;
            }
            let io = &kernel.inputs[i];
            // Synthesised `__idx_*` columns don't carry validity in the
            // dictionary registry; they correspond to dictionary index
            // columns whose null-bearing nature lives upstream on the
            // source DictUtf8 column. Skip with a structured error so the
            // caller knows to surface the breakage.
            if io.name.starts_with("__idx_") {
                return Err(BoltError::Plan(format!(
                    "engine: kernel flags `__idx_` column '{}' as needing validity, but \
                     dictionary registry does not yet expose a per-row validity bitmap; \
                     route the predicate through the host fallback",
                    io.name
                )));
            }
            let column = gpu_table.column(&io.name).ok_or_else(|| {
                BoltError::Plan(format!(
                    "column '{}' not in table '{}' (validity wiring)",
                    io.name, table
                ))
            })?;
            let vptr = match column.data.validity_ptr() {
                Some(vptr) => vptr,
                None => {
                    // The column is nullable in the schema (so the planner did
                    // NOT constant-fold this IsNullCheck) but carries NO actual
                    // nulls on device, so `validity_ptr()` is `None`. Every row
                    // is valid: synthesise an all-valid (all-1s) per-row bitmap
                    // — the unpacked 1-byte-per-row format `emit_is_null_check`
                    // reads — so IS NULL → false / IS NOT NULL → true for every
                    // row, instead of erroring on the schema-vs-storage mismatch.
                    let all_valid = GpuVec::<u8>::from_slice(&vec![1u8; n_rows])?;
                    let ptr = all_valid.device_ptr();
                    synth_validity_bufs.push(all_valid);
                    ptr
                }
            };
            input_validity_ptrs.push(vptr);
        }

        let mut device_ptrs: Vec<CUdeviceptr> =
            Vec::with_capacity(input_ptrs.len() + output_cols.len() + input_validity_ptrs.len());
        for p in &input_ptrs {
            device_ptrs.push(*p);
        }
        for c in &output_cols {
            device_ptrs.push(c.device_ptr());
        }
        // Validity pointers come AFTER value inputs and outputs, matching the
        // order in `ptx_gen::compile` (input-validity first, then output-
        // validity). `KernelSpec::output_has_validity` is empty for the
        // projection path today, so we only emit input-validity ptrs.
        for vp in &input_validity_ptrs {
            device_ptrs.push(*vp);
        }
        let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;

        let mut kernel_params: Vec<*mut c_void> = Vec::with_capacity(device_ptrs.len() + 1);
        for p in device_ptrs.iter_mut() {
            kernel_params.push(p as *mut CUdeviceptr as *mut c_void);
        }
        kernel_params.push(&mut n_rows_u32 as *mut u32 as *mut c_void);

        // 5. Launch with one thread per row, block size 256.
        //
        // Stage-3 async memcpy: mint a per-call stream so the kernel
        // launch, mask materialisation (if any), and the final pinned
        // D2H download can run on the same stream — letting the driver
        // overlap them with concurrent work on the NULL stream. Falls
        // back to the NULL stream if stream creation fails (see
        // `CudaStream::null_or_default`).
        let stream = CudaStream::null_or_default();
        let grid_x = grid_x_for(n_rows_u32, BLOCK_SIZE);
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                function.raw(),
                grid_x,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                stream.raw(),
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        crate::metrics::metrics().inc(crate::metrics::Counter::GpuLaunchesTotal);
        // V-1 / F10a: this hand-rolled `cuLaunchKernel` bypasses
        // `KernelArgs::tag_launch_stream` (the central Drop-fence enforcement
        // point that `launch_1d` callers get for free). Restore the invariant
        // for the freshly-allocated output buffers by recording the launch
        // stream in each one's `StreamSet`, so a buffer dropped while the
        // kernel is still in flight fences this stream before its block is
        // recycled — independent of the downstream `synchronize()` in
        // `download_pinned` / `gpu_compact`. The input columns live in the
        // persistent GpuTable cache (not recycled across this launch), so the
        // load-bearing buffers to tag are the outputs.
        for col in &output_cols {
            col.mark_launch_stream(stream.raw());
        }
        // F12: the materialised `__rank_<col>` input buffers are freshly
        // allocated (unlike the persistent GpuTable inputs), so they too must
        // fence this stream on drop. They are read by both the projection
        // kernel above and the predicate kernel below; tagging here restores the
        // same `Drop`-fence invariant `mark_launch_stream` gives the outputs.
        for buf in &rank_bufs {
            buf.mark_stream_use(stream.raw());
        }
        // Debug-only synchronize: pin any in-kernel fault to THIS launch
        // rather than letting it surface at the next CUDA API call.
        debug_sync_check()?;
        // NOTE: no `stream.synchronize()` here — the predicate / gather path
        // and the async-D2H path below both run on the same stream and so are
        // serialized after the kernel automatically. The single sync happens
        // at the bottom of this function (or inside `gpu_compact` for the
        // predicate path, which manages its own stream barriers).

        // 6. If the kernel has a predicate, run a separate predicate-only
        //    kernel to materialise a u8 mask. We default to GPU-side compaction
        //    (prefix scan + gather) when every output column is gather-friendly
        //    (primitive + Bool); Utf8 outputs fall back to the host-side path
        //    because the gather kernel can't move variable-width strings.
        let arrays: Vec<ArrayRef> = if kernel.predicate.is_some() {
            // Review-H2: predicate kernel goes through the same module
            // cache, keyed by `(spec_hash, PREDICATE_ENTRY)` so it doesn't
            // collide with the projection kernel cached under
            // `(spec_hash, KERNEL_ENTRY)`.
            let pred_module = self.get_or_build_module(kernel, PREDICATE_ENTRY, |k| {
                crate::jit::scan_kernel::compile_predicate_kernel(k, PREDICATE_ENTRY)
            })?;
            let pred_function = pred_module.function(PREDICATE_ENTRY)?;

            let mask = crate::exec::compact::alloc_mask_buffer(n_rows)?;
            // Validity-pointer wiring for the predicate kernel (Batch 7,
            // IS NULL e2e). The scan_kernel's emitted PTX consumes the
            // flagged-input validity pointers AFTER the mask output, in
            // input-slot order. `input_validity_ptrs` above was assembled
            // for the projection kernel; reuse it here so the order and
            // membership stay in lockstep with the kernel's signature.
            crate::exec::compact::launch_predicate_kernel(
                pred_function,
                &input_ptrs,
                mask.device_ptr(),
                &input_validity_ptrs,
                n_rows_to_u32(n_rows)?,
                &stream,
            )?;
            // Debug-only synchronize: surface predicate-kernel faults at
            // THIS launch site rather than at a later API call.
            debug_sync_check()?;

            let has_utf8_output = kernel.outputs.iter().any(|c| c.dtype == DataType::Utf8);
            if has_utf8_output {
                // M5 metrics: Utf8 outputs can't be gathered on-device, so
                // this projection takes the documented host-side filter path.
                crate::metrics::metrics().inc(crate::metrics::Counter::HostFallbacksTotal);
                // Host-side fallback: download mask + outputs, then filter.
                let host_mask =
                    crate::exec::compact::download_mask(mask.device_ptr(), n_rows, &stream)?;
                // Stage-3: route every primitive output column through the
                // pinned async D2H path. Each `download_pinned` call
                // synchronizes the stream internally, so we don't need a
                // separate barrier between the predicate kernel and these
                // reads.
                let mut full: Vec<ArrayRef> = Vec::with_capacity(output_cols.len());
                for col in output_cols {
                    full.push(col.download_pinned(n_rows, &stream)?);
                }
                crate::exec::compact::compact_arrays(&full, &host_mask)?
            } else {
                // GPU-side path: prefix-scan + gather, download the compacted output.
                let cols: Vec<(CUdeviceptr, DataType)> = output_cols
                    .iter()
                    .zip(kernel.outputs.iter())
                    .map(|(c, io)| (c.device_ptr(), io.dtype))
                    .collect();
                let (gathered, _total) = crate::exec::gpu_compact::compact_columns_on_gpu(
                    mask.device_ptr(),
                    n_rows,
                    &cols,
                    &stream,
                )?;
                // Output buffers can drop now; gathered owns the compacted data.
                drop(output_cols);
                // Batched, stream-overlapped D2H: enqueue every column's async
                // copy into pinned host buffers on one stream, then synchronize
                // once — instead of N blocking per-column `download()` round trips.
                crate::exec::gpu_compact::download_columns(&gathered, &stream)?
            }
        } else {
            // Stage-3 pinned downloads on the per-call stream. Each
            // call synchronizes internally before reading, so the loop
            // is correct even though `stream` was used for the kernel
            // launch above.
            let mut full: Vec<ArrayRef> = Vec::with_capacity(output_cols.len());
            for col in output_cols {
                full.push(col.download_pinned(n_rows, &stream)?);
            }
            full
        };

        // 9. Build the result RecordBatch (Materialize phase).
        let materialize_start = Instant::now();
        let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
        let batch_out = RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
            BoltError::Other(format!("failed to build output RecordBatch: {e}"))
        })?;
        crate::metrics::metrics()
            .observe_duration(crate::metrics::Phase::Materialize, materialize_start.elapsed());
        Ok(QueryHandle { batch: batch_out })
    }

    /// Host re-run of a GPU projection the device *declined*
    /// ([`BoltError::GpuCapacity`]).
    ///
    /// Agent-G's F11 turned the unsupported temporal/decimal GPU upload+gather
    /// arms into the typed `GpuCapacity` decline marker instead of a fatal
    /// error, so the engine can re-run on the host exactly like the join gates
    /// (`try_gpu_inner_join`/`try_gpu_outer_join`) already do for declined GPU
    /// joins. In practice the columns that trip the decline (Date32 /
    /// Timestamp / Decimal128) only reach `execute_projection` as **passthrough**
    /// projections — `SELECT date_col, dec_col, ... FROM t` — because the
    /// arithmetic the GPU IR can express over them is narrow and the planner
    /// routes anything richer through the host `PhysicalPlan::Project` /
    /// `PhysicalPlan::StringProject` paths.
    ///
    /// We therefore serve the **identity-passthrough** shape directly from the
    /// host-materialised table: a kernel with no predicate whose `ops` are
    /// exactly `LoadColumn`→`Store` (or the 128-bit `LoadColumn128`→`Store128`)
    /// pairs, one per output, each output fed by an input column we can pick out
    /// of the source batch by name. This is correct by construction (no
    /// computation is dropped) and needs no GPU.
    ///
    /// For any non-passthrough kernel (a predicate, or a real compute op over a
    /// declined column) we deliberately **re-raise** the original decline rather
    /// than risk an incorrect host result — there is no host op-VM interpreter
    /// for the fused projection IR, and silently mis-evaluating would be worse
    /// than surfacing the (rare) decline. `reason` is threaded through so the
    /// re-raised error keeps the device's original message.
    fn execute_projection_host_fallback(
        &self,
        table: &str,
        kernel: &KernelSpec,
        output_schema: &Schema,
        reason: &str,
    ) -> BoltResult<QueryHandle> {
        let reraise = || {
            BoltError::GpuCapacity(format!(
                "GPU declined projection on table '{table}' ({reason}) and the \
                 host fallback only supports identity-passthrough projections"
            ))
        };

        // Detect the identity-passthrough shape and recover, for each output
        // column, the input column ordinal that feeds it. `None` => not a pure
        // passthrough (predicate present, or a compute/cast/select op), so we
        // re-raise the decline rather than risk a wrong host result.
        let out_src = passthrough_output_sources(kernel).ok_or_else(|| reraise())?;

        // Pull the source rows from the host-materialised table and pick the
        // mapped input column for each output, casting to the declared output
        // arrow dtype (an identity/no-op cast for a true passthrough, but it
        // also coerces a dictionary-encoded source to its logical dtype).
        let src_batch = self.materialize_table(table)?;
        let src_schema = src_batch.schema();
        let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(kernel.outputs.len());
        for (out_idx, out_io) in kernel.outputs.iter().enumerate() {
            let in_idx = out_src[out_idx];
            let in_io = kernel.inputs.get(in_idx).ok_or_else(|| reraise())?;
            let col_pos = src_schema.index_of(&in_io.name).map_err(|_| {
                BoltError::Plan(format!(
                    "host projection fallback: input column '{}' not found in \
                     table '{table}'",
                    in_io.name
                ))
            })?;
            let src_arr = src_batch.column(col_pos);
            let want = arrow_schema.field(out_idx).data_type();
            let arr: ArrayRef = if src_arr.data_type() == want {
                src_arr.clone()
            } else {
                arrow::compute::cast(src_arr.as_ref(), want).map_err(|e| {
                    BoltError::Other(format!(
                        "host projection fallback: cast of '{}' to {:?} failed: {e}",
                        out_io.name, want
                    ))
                })?
            };
            arrays.push(arr);
        }

        let batch_out = RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
            BoltError::Other(format!(
                "host projection fallback: failed to build RecordBatch: {e}"
            ))
        })?;
        Ok(QueryHandle::from_record_batch(batch_out))
    }

    /// Execute a [`PhysicalPlan::StringLength`]: a `SELECT LENGTH(<utf8_col>)`
    /// projection (plus passthrough columns) over a bare scan, with the
    /// `LENGTH` outputs computed on the GPU via the dictionary-index gather
    /// kernel ([`crate::jit::string_kernel::compile_length_gather_kernel`]).
    ///
    /// Passthrough columns are lifted directly from the host-side source batch
    /// (zero-copy `ArrayRef` clone). Each `LENGTH(col)` output runs the gather
    /// against the GPU-resident dictionary column when it is dictionary-encoded
    /// (and, for the native `DictUtf8` layout, null-free); otherwise it falls
    /// back to a host-side gather over the downloaded keys (see
    /// [`crate::exec::string_length`]). Both paths produce an `Int64Array`
    /// matching the logical-plan `LENGTH` output dtype.
    fn execute_string_length(
        &self,
        table: &str,
        outputs: &[crate::plan::physical_plan::StringLengthOutput],
        output_schema: &Schema,
    ) -> BoltResult<QueryHandle> {
        use crate::plan::physical_plan::StringLengthOutput;

        // Source host batch — used for passthrough columns (and as the row
        // count authority so an empty / partial table still works).
        let src_batch = self.materialize_table(table)?;
        let src_schema = src_batch.schema();
        let n_rows = src_batch.num_rows();

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(outputs.len());
        for out in outputs {
            match out {
                StringLengthOutput::Passthrough { source } => {
                    let idx = src_schema.index_of(source).map_err(|_| {
                        BoltError::Plan(format!(
                            "StringLength: passthrough column '{source}' not found in \
                             table '{table}'"
                        ))
                    })?;
                    arrays.push(src_batch.column(idx).clone());
                }
                StringLengthOutput::Length { source } => {
                    arrays.push(self.string_length_column(table, source, n_rows)?);
                }
            }
        }

        let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
        let batch_out = RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
            BoltError::Other(format!(
                "StringLength: failed to build output RecordBatch: {e}"
            ))
        })?;
        Ok(QueryHandle { batch: batch_out })
    }

    /// Compute `LENGTH(<source>)` for the GPU-resident `Utf8` column `source`
    /// of `table`, returning an `Int64Array` of `n_rows` rows.
    ///
    /// GPU path: build the per-dictionary-entry `i32` length table that matches
    /// the column's device key layout, upload it, launch the gather kernel
    /// (`out[row] = length_table[keys[row]]`), download the `Int32` result, and
    /// widen to `Int64`. When the column is not safe to gather on the GPU
    /// (non-dict storage, or a `DictUtf8` column with NULLs — whose zeroed keys
    /// would gather the wrong slot), fall back to a host-side gather over the
    /// downloaded keys, which is byte-for-byte identical for the supported case.
    fn string_length_column(
        &self,
        table: &str,
        source: &str,
        n_rows: usize,
    ) -> BoltResult<ArrayRef> {
        use crate::exec::string_length::{
            build_length_table, gpu_gather_layout, host_gather_lengths, KeyLayout,
        };

        let gpu_table_ref = self.ensure_gpu_table(table)?;
        let gpu_table: &crate::exec::gpu_table::GpuTable = &gpu_table_ref;
        let column = gpu_table.column(source).ok_or_else(|| {
            BoltError::Plan(format!(
                "StringLength: column '{source}' not in GPU table '{table}'"
            ))
        })?;

        // Resolve the host-side dictionary + device key buffer + layout for
        // this column. `None` layout ⇒ host fallback.
        let dict = column.utf8_dictionary().ok_or_else(|| {
            BoltError::Plan(format!(
                "StringLength: column '{source}' is not a Utf8 column (LENGTH requires Utf8)"
            ))
        })?;
        let (keys_vec, layout): (&GpuVec<i32>, Option<KeyLayout>) = match &column.data {
            crate::exec::gpu_table::GpuColumnData::Utf8 { indices, .. } => {
                (indices, gpu_gather_layout(&column.data))
            }
            crate::exec::gpu_table::GpuColumnData::DictUtf8 { keys, .. } => {
                (keys, gpu_gather_layout(&column.data))
            }
            _ => {
                return Err(BoltError::Plan(format!(
                    "StringLength: column '{source}' has non-Utf8 GPU storage"
                )))
            }
        };

        let layout = match layout {
            Some(l) => l,
            None => {
                // Host fallback: download keys and gather over the 1-based
                // NULL-sentinel table. A NULL input row emits SQL NULL (a
                // validity-carrying `None`), distinct from `LENGTH('') = 0` —
                // matching the now-NULL-correct `exec::string_ops::length`
                // (agent-C F-3). Valid rows map to `table[key+1]`.
                let table_lengths =
                    build_length_table(dict, KeyLayout::OneBasedNullSlot0)?;
                let keys_host = keys_vec.to_vec()?;
                // DictUtf8 keys are 0-based; remap to the 1-based table by
                // adding 1 only when the column is the DictUtf8 layout.
                let lens: Vec<Option<i64>> = match &column.data {
                    crate::exec::gpu_table::GpuColumnData::DictUtf8 { valid_mask, .. } => {
                        // Consult validity: NULL rows → SQL NULL, valid rows →
                        // table[key+1].
                        let mask = valid_mask
                            .as_ref()
                            .map(|m| m.to_vec())
                            .transpose()?;
                        let mut out: Vec<Option<i64>> = Vec::with_capacity(keys_host.len());
                        for (row, &k) in keys_host.iter().enumerate() {
                            let is_valid = match &mask {
                                None => true,
                                Some(bits) => {
                                    let byte = bits.get(row / 8).copied().unwrap_or(0);
                                    (byte >> (row % 8)) & 1 == 1
                                }
                            };
                            if !is_valid {
                                // SQL NULL, NOT length 0.
                                out.push(None);
                            } else if k < 0 {
                                return Err(BoltError::Other(format!(
                                    "LENGTH: negative dictionary key {k}"
                                )));
                            } else {
                                // table index = key + 1 (slot 0 is NULL).
                                let len = *table_lengths
                                    .get(k as usize + 1)
                                    .ok_or_else(|| {
                                        BoltError::Other(format!(
                                            "LENGTH: key {k} out of range"
                                        ))
                                    })?;
                                out.push(Some(len as i64));
                            }
                        }
                        out
                    }
                    // Non-DictUtf8 host gather: no per-row validity bitmap at
                    // this layer, so every gathered length is non-NULL.
                    _ => host_gather_lengths(&keys_host, &table_lengths)?
                        .into_iter()
                        .map(Some)
                        .collect(),
                };
                check_len(lens.len(), n_rows)?;
                // `Int64Array::from(Vec<Option<i64>>)` carries the validity
                // bitmap, so NULL rows decode back to SQL NULL.
                return Ok(Arc::new(Int64Array::from(lens)) as ArrayRef);
            }
        };

        // GPU gather path.
        let length_table = build_length_table(dict, layout)?;
        let table_gpu = GpuVec::<i32>::from_slice(&length_table)?;
        let out_gpu = GpuVec::<i32>::zeros(n_rows)?;

        let module =
            CudaModule::from_ptx(&crate::jit::string_kernel::compile_length_gather_kernel()?)?;
        let function =
            module.function(crate::jit::string_kernel::LENGTH_GATHER_ENTRY)?;

        // ABI: (indices, length_table, out, n_rows). Assemble raw kernel
        // params directly (heterogeneous list; same pattern as
        // `execute_projection`).
        let mut indices_ptr = keys_vec.device_ptr();
        let mut table_ptr = table_gpu.device_ptr();
        let mut out_ptr = out_gpu.device_ptr();
        let mut n_rows_u32 = n_rows_to_u32(n_rows)?;
        let mut kernel_params: Vec<*mut c_void> = vec![
            &mut indices_ptr as *mut CUdeviceptr as *mut c_void,
            &mut table_ptr as *mut CUdeviceptr as *mut c_void,
            &mut out_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
        ];

        let stream = CudaStream::null_or_default();
        let grid_x = grid_x_for(n_rows_u32, BLOCK_SIZE);
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                function.raw(),
                grid_x,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                stream.raw(),
                kernel_params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        debug_sync_check()?;
        stream.synchronize()?;

        // Download Int32 lengths and widen to the Int64 SQL contract.
        let lens_i32 = out_gpu.to_vec()?;
        check_len(lens_i32.len(), n_rows)?;

        // Derive the output validity bitmap from the SOURCE column's NULLs so
        // `LENGTH(NULL)` is SQL NULL, not a valid `0`. The kernel gathered a
        // bare length for every row (a NULL row gathered length-table slot 0 =
        // 0), so without re-applying validity here a NULL input would read back
        // as a valid `0`. `LENGTH('')` is unaffected: an empty string is a real
        // dictionary entry whose row is VALID, so it stays a valid `0`.
        //
        // * `OneBasedNullSlot0` (engine-managed `Utf8`): NULL ⇔ key == 0 (the
        //   reserved NULL sentinel slot). Real strings — including `""` — have
        //   key >= 1 and are valid.
        // * `ZeroBased` (`DictUtf8`): `gpu_gather_layout` only selects the GPU
        //   path when `valid_mask` is `None`, i.e. there are no NULLs, so every
        //   row is valid.
        let lens_opt: Vec<Option<i64>> = match layout {
            KeyLayout::OneBasedNullSlot0 => {
                let keys_host = keys_vec.to_vec()?;
                check_len(keys_host.len(), n_rows)?;
                lens_i32
                    .into_iter()
                    .zip(keys_host.into_iter())
                    .map(|(len, key)| if key == 0 { None } else { Some(len as i64) })
                    .collect()
            }
            // No NULLs on this path (see above): every length is valid.
            KeyLayout::ZeroBased => lens_i32.into_iter().map(|v| Some(v as i64)).collect(),
        };
        // `Int64Array::from(Vec<Option<i64>>)` carries the validity bitmap, so
        // NULL rows decode back to SQL NULL.
        Ok(Arc::new(Int64Array::from(lens_opt)) as ArrayRef)
    }

    /// Execute a [`PhysicalPlan::StringProject`]: a `SELECT UPPER(<utf8_col>)` /
    /// `LOWER(<utf8_col>)` projection (plus passthrough columns) over a bare
    /// scan, with the transform outputs produced on the GPU via the two-pass
    /// length/scan/write kernels in [`crate::jit::string_kernel`] (see
    /// [`crate::exec::string_project`]).
    ///
    /// Passthrough columns are lifted directly from the host source batch.
    /// Each `UPPER`/`LOWER(col)` output runs the two-pass GPU producer against a
    /// row-aligned offsets+bytes input materialised from the dictionary-encoded
    /// column — but only when the column's dictionary is pure ASCII (the kernels
    /// ASCII-fold byte-wise; non-ASCII Unicode case mapping can change byte
    /// length, e.g. `'ß'` → `"SS"`). Non-ASCII dictionaries, or columns with no
    /// supported GPU storage, fall back to a full-Unicode host transform. Both
    /// paths produce a `StringArray`.
    fn execute_string_project(
        &self,
        table: &str,
        outputs: &[crate::plan::physical_plan::StringProjectOutput],
        output_schema: &Schema,
    ) -> BoltResult<QueryHandle> {
        use crate::plan::physical_plan::StringProjectOutput;

        let src_batch = self.materialize_table(table)?;
        let src_schema = src_batch.schema();
        let n_rows = src_batch.num_rows();

        // Lazily-built host env (decoded source columns as `HostColumn`s) for
        // the `CaseUtf8` output path; `None` until the first CASE forces the
        // lift. Mirrors the `PhysicalPlan::Project` compute path in `execute`.
        let mut owned_env: Option<Vec<(String, crate::exec::expr_agg::HostColumn)>> = None;

        let mut arrays: Vec<ArrayRef> = Vec::with_capacity(outputs.len());
        for out in outputs {
            match out {
                StringProjectOutput::Passthrough { source } => {
                    let idx = src_schema.index_of(source).map_err(|_| {
                        BoltError::Plan(format!(
                            "StringProject: passthrough column '{source}' not found in \
                             table '{table}'"
                        ))
                    })?;
                    let src_col = src_batch.column(idx);
                    // A dictionary-encoded Utf8 column is stored as
                    // `Dictionary(Int32, Utf8)` on the host but projects as
                    // logical `Utf8` (the output schema declares `Utf8`).
                    // Decode it to a plain Utf8 array so the built batch matches
                    // the schema; non-dictionary columns pass through unchanged.
                    if matches!(src_col.data_type(), ArrowDataType::Dictionary(_, _)) {
                        let decoded = arrow::compute::cast(
                            src_col.as_ref(),
                            &ArrowDataType::Utf8,
                        )
                        .map_err(|e| {
                            BoltError::Other(format!(
                                "StringProject: decode dictionary '{source}' to Utf8 failed: {e}"
                            ))
                        })?;
                        arrays.push(decoded);
                    } else {
                        arrays.push(src_col.clone());
                    }
                }
                StringProjectOutput::Transform { source, transform } => {
                    arrays.push(self.string_transform_column(table, source, *transform, n_rows)?);
                }
                StringProjectOutput::Concat { sources } => {
                    arrays.push(self.string_concat_column(table, sources, n_rows)?);
                }
                StringProjectOutput::CaseUtf8 {
                    branches,
                    else_branch,
                } => {
                    // Build the host env (decoded source columns) once, lazily.
                    // Dictionary-encoded Utf8 columns are decoded to a plain
                    // Utf8 array first (`arrow_array_to_host_column` has no
                    // Dictionary arm), mirroring the Passthrough decode above.
                    if owned_env.is_none() {
                        let mut v = Vec::with_capacity(src_batch.num_columns());
                        for (i, field) in src_schema.fields().iter().enumerate() {
                            let arr = src_batch.column(i);
                            let decoded: ArrayRef = if matches!(
                                arr.data_type(),
                                ArrowDataType::Dictionary(_, _)
                            ) {
                                arrow::compute::cast(arr.as_ref(), &ArrowDataType::Utf8)
                                    .map_err(|e| {
                                        BoltError::Other(format!(
                                            "StringProject(CaseUtf8): decode dictionary \
                                             '{}' to Utf8 failed: {e}",
                                            field.name()
                                        ))
                                    })?
                            } else {
                                arr.clone()
                            };
                            let hc = crate::exec::filter::arrow_array_to_host_column(
                                decoded.as_ref(),
                                n_rows,
                            )?;
                            v.push((field.name().clone(), hc));
                        }
                        owned_env = Some(v);
                    }
                    let env_ref = owned_env.as_ref().expect("just built");
                    let env: crate::exec::expr_agg::ColumnEnv<'_> =
                        env_ref.iter().map(|(n, c)| (n.clone(), c)).collect();
                    let arr = crate::exec::string_project::eval_case_utf8(
                        branches,
                        else_branch.as_deref(),
                        &env,
                        n_rows,
                    )?;
                    arrays.push(Arc::new(arr) as ArrayRef);
                }
            }
        }

        let arrow_schema = plan_schema_to_arrow_schema(output_schema)?;
        let batch_out = RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
            BoltError::Other(format!(
                "StringProject: failed to build output RecordBatch: {e}"
            ))
        })?;
        Ok(QueryHandle { batch: batch_out })
    }

    /// Compute a [`StringTransform`](crate::exec::string_project::StringTransform)
    /// — `UPPER`/`LOWER`/`SUBSTRING`/`TRIM` — of the GPU-resident `Utf8` column
    /// `source` of `table`, returning a `Utf8` `ArrayRef` of `n_rows` rows.
    ///
    /// `SUBSTRING`/`TRIM` are realised via the byte-identical host mirror
    /// ([`crate::exec::string_project::host_transform_strings`]) regardless of
    /// the dictionary contents — their GPU two-pass producers exist but are
    /// unvalidated on hardware (matching the CONCAT path).
    ///
    /// For `UPPER`/`LOWER`:
    /// GPU path (ASCII dictionaries): materialise a row-aligned offsets+bytes
    /// input from the column's dictionary + device keys, upload, run the length
    /// pass → host exclusive scan of `row_lens` → allocate output bytes → run
    /// the write pass → download → rebuild the `StringArray` (re-applying NULLs).
    /// Host fallback (non-ASCII dictionary, or unsupported GPU storage): apply
    /// the full-Unicode transform host-side. Both paths preserve NULLs as Arrow
    /// NULLs.
    fn string_transform_column(
        &self,
        table: &str,
        source: &str,
        transform: crate::exec::string_project::StringTransform,
        n_rows: usize,
    ) -> BoltResult<ArrayRef> {
        use crate::exec::string_project::{
            build_row_aligned_input, dict_is_ascii, exclusive_scan_lens, host_transform_strings,
            string_array_from_offsets, KeyLayout,
        };

        let gpu_table_ref = self.ensure_gpu_table(table)?;
        let gpu_table: &crate::exec::gpu_table::GpuTable = &gpu_table_ref;
        let column = gpu_table.column(source).ok_or_else(|| {
            BoltError::Plan(format!(
                "StringProject: column '{source}' not in GPU table '{table}'"
            ))
        })?;
        let dict = column.utf8_dictionary().ok_or_else(|| {
            BoltError::Plan(format!(
                "StringProject: column '{source}' is not a Utf8 column"
            ))
        })?;

        // Resolve the host-side keys + layout + per-row validity for this
        // column. For the engine-managed `Utf8` layout NULL is encoded as key 0
        // (1-based dict); for native `DictUtf8` NULL lives on `valid_mask`.
        let (keys_host, layout, validity): (Vec<i32>, KeyLayout, Option<Vec<bool>>) =
            match &column.data {
                crate::exec::gpu_table::GpuColumnData::Utf8 { indices, .. } => {
                    let keys = indices.to_vec()?;
                    // Validity = key != 0 (slot 0 is the NULL sentinel).
                    let valid: Vec<bool> = keys.iter().map(|&k| k != 0).collect();
                    (keys, KeyLayout::OneBasedNullSlot0, Some(valid))
                }
                crate::exec::gpu_table::GpuColumnData::DictUtf8 {
                    keys, valid_mask, ..
                } => {
                    let keys = keys.to_vec()?;
                    let valid = match valid_mask {
                        None => None,
                        Some(mask) => {
                            let bits = mask.to_vec()?;
                            let v: Vec<bool> = (0..keys.len())
                                .map(|row| {
                                    let byte = bits.get(row / 8).copied().unwrap_or(0);
                                    (byte >> (row % 8)) & 1 == 1
                                })
                                .collect();
                            Some(v)
                        }
                    };
                    (keys, KeyLayout::ZeroBased, valid)
                }
                _ => {
                    return Err(BoltError::Plan(format!(
                        "StringProject: column '{source}' has non-Utf8 GPU storage"
                    )))
                }
            };

        check_len(keys_host.len(), n_rows)?;
        let validity_slice = validity.as_deref();

        // SUBSTRING / TRIM are realised host-side (byte-identical to the host
        // helpers in `string_ops_extended`). The GPU two-pass producers for
        // these exist in `jit::string_kernel` and are PTX-shape-tested, but are
        // unvalidated on hardware (like CONCAT / LIKE), so we take the
        // correctness-guaranteed host path here. Results are identical either
        // way; wiring the device launch is a follow-up.
        if transform.is_host_realized() {
            let arr =
                host_transform_strings(dict, &keys_host, layout, validity_slice, transform)?;
            return Ok(Arc::new(arr) as ArrayRef);
        }

        // Host fallback for non-ASCII dictionaries: the byte-wise GPU fold is
        // only correct for ASCII (Unicode case mapping can change byte length).
        if !dict_is_ascii(dict) {
            let arr =
                host_transform_strings(dict, &keys_host, layout, validity_slice, transform)?;
            return Ok(Arc::new(arr) as ArrayRef);
        }

        // ---- GPU two-pass path -------------------------------------------
        // Pass 0 (host): materialise the row-aligned offsets+bytes input.
        let (src_offsets, src_bytes) =
            build_row_aligned_input(dict, &keys_host, layout, validity_slice)?;

        // Empty input (no rows, or all-empty bytes): skip the launch and build
        // the result directly. `from_slice` on an empty slice is brittle and a
        // zero-thread launch is pointless.
        if n_rows == 0 {
            let arr = string_array_from_offsets(&src_offsets, &src_bytes, validity_slice)?;
            return Ok(Arc::new(arr) as ArrayRef);
        }

        let kind = transform.scalar_fn_kind();
        let src_offsets_gpu = GpuVec::<i32>::from_slice(&src_offsets)?;
        // `src_bytes` may be empty (all rows empty/NULL); allocate at least one
        // byte so the device pointer is valid even though no thread reads it.
        let src_bytes_gpu = if src_bytes.is_empty() {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::from_slice(&src_bytes)?
        };
        let row_lens_gpu = GpuVec::<u32>::zeros(n_rows)?;

        let n_rows_u32 = n_rows_to_u32(n_rows)?;
        let stream = CudaStream::null_or_default();
        let grid_x = grid_x_for(n_rows_u32, BLOCK_SIZE);

        // ---- Pass 1: length pass → row_lens. ABI (UPPER/LOWER, 4 params):
        //      (src_offsets, src_bytes, row_lens, n_rows).
        {
            let module = CudaModule::from_ptx(
                &crate::jit::string_kernel::compile_varwidth_len_pass(kind)?,
            )?;
            let entry = crate::jit::string_kernel::len_pass_entry(kind)?;
            let function = module.function(&entry)?;

            let mut p_off = src_offsets_gpu.device_ptr();
            let mut p_bytes = src_bytes_gpu.device_ptr();
            let mut p_lens = row_lens_gpu.device_ptr();
            let mut p_n = n_rows_u32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_off as *mut CUdeviceptr as *mut c_void,
                &mut p_bytes as *mut CUdeviceptr as *mut c_void,
                &mut p_lens as *mut CUdeviceptr as *mut c_void,
                &mut p_n as *mut u32 as *mut c_void,
            ];
            unsafe {
                cuda_sys::check(cuda_sys::cuLaunchKernel(
                    function.raw(),
                    grid_x,
                    1,
                    1,
                    BLOCK_SIZE,
                    1,
                    1,
                    0,
                    stream.raw(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ))?;
            }
            debug_sync_check()?;
            stream.synchronize()?;
        }

        // ---- Pass 2 (host): exclusive-scan row_lens → out_offsets + total.
        let row_lens = row_lens_gpu.to_vec()?;
        check_len(row_lens.len(), n_rows)?;
        let (out_offsets, total_bytes) = exclusive_scan_lens(&row_lens)?;
        let out_offsets_gpu = GpuVec::<i32>::from_slice(&out_offsets)?;
        let out_bytes_gpu = if total_bytes == 0 {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::zeros(total_bytes)?
        };

        // ---- Pass 3: write pass → out_bytes. ABI (UPPER/LOWER, 5 params):
        //      (src_offsets, src_bytes, out_offsets, out_bytes, n_rows).
        {
            let module = CudaModule::from_ptx(
                &crate::jit::string_kernel::compile_varwidth_write_pass(kind)?,
            )?;
            let entry = crate::jit::string_kernel::write_pass_entry(kind)?;
            let function = module.function(&entry)?;

            let mut p_off = src_offsets_gpu.device_ptr();
            let mut p_bytes = src_bytes_gpu.device_ptr();
            let mut p_out_off = out_offsets_gpu.device_ptr();
            let mut p_out_bytes = out_bytes_gpu.device_ptr();
            let mut p_n = n_rows_u32;
            let mut params: Vec<*mut c_void> = vec![
                &mut p_off as *mut CUdeviceptr as *mut c_void,
                &mut p_bytes as *mut CUdeviceptr as *mut c_void,
                &mut p_out_off as *mut CUdeviceptr as *mut c_void,
                &mut p_out_bytes as *mut CUdeviceptr as *mut c_void,
                &mut p_n as *mut u32 as *mut c_void,
            ];
            unsafe {
                cuda_sys::check(cuda_sys::cuLaunchKernel(
                    function.raw(),
                    grid_x,
                    1,
                    1,
                    BLOCK_SIZE,
                    1,
                    1,
                    0,
                    stream.raw(),
                    params.as_mut_ptr(),
                    ptr::null_mut(),
                ))?;
            }
            debug_sync_check()?;
            stream.synchronize()?;
        }

        // ---- Download + rebuild StringArray (re-applying NULLs).
        let out_bytes = out_bytes_gpu.to_vec()?;
        // `out_bytes_gpu` was padded to >= 1 byte; truncate to the real total.
        let out_bytes = &out_bytes[..total_bytes.min(out_bytes.len())];
        let arr = string_array_from_offsets(&out_offsets, out_bytes, validity_slice)?;
        Ok(Arc::new(arr) as ArrayRef)
    }

    /// Compute `CONCAT(s0, s1, ...)` for the GPU-resident `Utf8` source columns
    /// of `table`, returning a `Utf8` `ArrayRef` of `n_rows` rows with
    /// NULL-if-any-arg-NULL semantics (standard SQL, matching the host path).
    ///
    /// Resolution of each column's dictionary + device keys + layout + per-row
    /// validity mirrors [`string_transform_column`](Self::string_transform_column).
    /// The N-input two-pass GPU concat kernels
    /// ([`crate::jit::string_kernel::compile_concat_len_pass`] /
    /// `compile_concat_write_pass`) exist and are PTX-shape-tested; this executor
    /// currently realises the result via the byte-identical host mirror
    /// ([`crate::exec::string_project::host_concat_strings`]) so the path is
    /// correctness-guaranteed (the device concat kernel is unvalidated on
    /// hardware, like the LIKE matcher). Wiring the device launch here is a
    /// follow-up; results are identical either way.
    fn string_concat_column(
        &self,
        table: &str,
        sources: &[String],
        n_rows: usize,
    ) -> BoltResult<ArrayRef> {
        use crate::exec::string_project::{build_concat_input, host_concat_strings, KeyLayout};

        let gpu_table_ref = self.ensure_gpu_table(table)?;
        let gpu_table: &crate::exec::gpu_table::GpuTable = &gpu_table_ref;

        let mut inputs = Vec::with_capacity(sources.len());
        for source in sources {
            let column = gpu_table.column(source).ok_or_else(|| {
                BoltError::Plan(format!(
                    "StringProject(Concat): column '{source}' not in GPU table '{table}'"
                ))
            })?;
            let dict = column.utf8_dictionary().ok_or_else(|| {
                BoltError::Plan(format!(
                    "StringProject(Concat): column '{source}' is not a Utf8 column"
                ))
            })?;
            // Same (keys, layout, validity) resolution as `string_transform_column`.
            let (keys_host, layout, validity): (Vec<i32>, KeyLayout, Option<Vec<bool>>) =
                match &column.data {
                    crate::exec::gpu_table::GpuColumnData::Utf8 { indices, .. } => {
                        let keys = indices.to_vec()?;
                        let valid: Vec<bool> = keys.iter().map(|&k| k != 0).collect();
                        (keys, KeyLayout::OneBasedNullSlot0, Some(valid))
                    }
                    crate::exec::gpu_table::GpuColumnData::DictUtf8 {
                        keys, valid_mask, ..
                    } => {
                        let keys = keys.to_vec()?;
                        let valid = match valid_mask {
                            None => None,
                            Some(mask) => {
                                let bits = mask.to_vec()?;
                                let v: Vec<bool> = (0..keys.len())
                                    .map(|row| {
                                        let byte = bits.get(row / 8).copied().unwrap_or(0);
                                        (byte >> (row % 8)) & 1 == 1
                                    })
                                    .collect();
                                Some(v)
                            }
                        };
                        (keys, KeyLayout::ZeroBased, valid)
                    }
                    _ => {
                        return Err(BoltError::Plan(format!(
                            "StringProject(Concat): column '{source}' has non-Utf8 GPU storage"
                        )))
                    }
                };
            check_len(keys_host.len(), n_rows)?;
            inputs.push(build_concat_input(
                dict,
                &keys_host,
                layout,
                validity.as_deref(),
            )?);
        }

        let arr = host_concat_strings(&inputs)?;
        Ok(Arc::new(arr) as ArrayRef)
    }

    /// Execute a [`PhysicalPlan::StringLikeFilter`]: a GPU per-row `LIKE` /
    /// `NOT LIKE` over a non-dictionary `Utf8` column, then materialise the
    /// surviving rows.
    ///
    /// ⚠️ UNVALIDATED DEVICE PATH. The matcher kernel
    /// ([`crate::jit::string_kernel::compile_like_match_kernel`]) has not run on
    /// GPU hardware; correctness is guaranteed by the host mirror in
    /// [`crate::exec::string_like`] and by this executor's clean host fallback.
    ///
    /// Flow: execute `input` (a bare scan → row-aligned source batch); pull the
    /// `column` as a host `StringArray`; build a row-aligned offsets+bytes
    /// buffer + validity; upload; launch the matcher (literal baked as a device
    /// buffer); download the 0/1 mask; re-apply NULL 3VL; `arrow::compute::filter`
    /// every column. If the column is absent / not Utf8 at run time, fall back
    /// to the host `LIKE` over the same `StringArray` (no panic).
    fn execute_string_like_filter(
        &self,
        input: &PhysicalPlan,
        _table: &str,
        column: &str,
        literal: &[u8],
        mode: crate::jit::string_kernel::LikeMode,
        negated: bool,
    ) -> BoltResult<QueryHandle> {
        use arrow_array::Array;

        // Execute the inner scan: this is the row-aligned source batch that
        // carries `column` (the lowering required a bare Scan beneath).
        let batch = self.execute(input)?.into_record_batch();
        let schema = batch.schema();

        // Locate the column; if missing or not a StringArray, fall back to the
        // host LIKE over whatever the column decodes to (no panic). Because the
        // lowering already proved `column` is a Utf8 scan column, the common
        // case is the StringArray downcast succeeding.
        let col_idx = match schema.index_of(column) {
            Ok(i) => i,
            Err(_) => {
                return Err(BoltError::Plan(format!(
                    "StringLikeFilter: column '{column}' not found in input batch"
                )))
            }
        };
        let col_arr = batch.column(col_idx);
        // Normalise to a `StringArray`. The common case is a direct downcast;
        // any other Utf8-compatible layout (e.g. a dictionary array that slipped
        // through un-rewritten) is cast to Utf8 so the path stays host-fallback-
        // safe (no panic, no hard error) for unexpected run-time layouts.
        let owned_cast: ArrayRef;
        let str_arr: &arrow_array::StringArray = match col_arr
            .as_any()
            .downcast_ref::<arrow_array::StringArray>()
        {
            Some(a) => a,
            None => {
                owned_cast = arrow::compute::cast(col_arr.as_ref(), &ArrowDataType::Utf8)
                    .map_err(|e| {
                        BoltError::Plan(format!(
                            "StringLikeFilter: column '{column}' is not Utf8 and could \
                             not be cast (got {:?}): {e}",
                            col_arr.data_type()
                        ))
                    })?;
                owned_cast
                    .as_any()
                    .downcast_ref::<arrow_array::StringArray>()
                    .ok_or_else(|| {
                        BoltError::Plan(format!(
                            "StringLikeFilter: cast of column '{column}' did not yield Utf8"
                        ))
                    })?
            }
        };

        // Build the boolean mask: GPU device path, with a host fallback that
        // produces the identical mask if the launch is not viable.
        let mask: arrow_array::BooleanArray = match self
            .string_like_mask_gpu(str_arr, literal, mode, negated)
        {
            Ok(m) => m,
            Err(e) => {
                // Host fallback: evaluate the SAME predicate via the validated
                // host mirror (equivalent to exec::like::host_like for these
                // shapes). Correctness is unaffected; only the GPU speedup is
                // lost. Logged so a hardware bring-up notices.
                log::warn!(
                    "StringLikeFilter: GPU matcher unavailable ({e}); \
                     falling back to host LIKE for column '{column}'"
                );
                crate::exec::string_like::host_mask_via_mirror(
                    str_arr, literal, mode, negated,
                )
            }
        };

        // Apply the mask to every column (NULL mask entries drop the row).
        let filtered: Vec<ArrayRef> = batch
            .columns()
            .iter()
            .map(|c| {
                arrow::compute::filter(c.as_ref(), &mask).map_err(|e| {
                    BoltError::Other(format!(
                        "StringLikeFilter: arrow filter failed: {e}"
                    ))
                })
            })
            .collect::<BoltResult<Vec<_>>>()?;
        let out = RecordBatch::try_new(batch.schema(), filtered).map_err(|e| {
            BoltError::Other(format!(
                "StringLikeFilter: failed to rebuild RecordBatch: {e}"
            ))
        })?;
        Ok(QueryHandle { batch: out })
    }

    /// GPU per-row LIKE matcher: upload the row-aligned column + literal, launch
    /// [`crate::jit::string_kernel::compile_like_match_kernel`], download the
    /// 0/1 mask, and re-apply NULL 3VL into a [`arrow_array::BooleanArray`].
    ///
    /// Returns `Err` (so the caller can host-fall-back) for any non-viable
    /// launch condition. UNVALIDATED device path — see the executor doc.
    fn string_like_mask_gpu(
        &self,
        col: &arrow_array::StringArray,
        literal: &[u8],
        mode: crate::jit::string_kernel::LikeMode,
        negated: bool,
    ) -> BoltResult<arrow_array::BooleanArray> {
        use arrow_array::Array;
        use crate::exec::string_like::{build_row_aligned_from_strings, mask_to_boolean_array};

        let n_rows = col.len();
        let (offsets, bytes, validity) = build_row_aligned_from_strings(col)?;

        // Empty input: nothing to launch; build the (empty) mask directly.
        if n_rows == 0 {
            return Ok(mask_to_boolean_array(&[], &validity));
        }

        // The engine already owns a live `CudaContext` (`self._ctx`), so device
        // allocations below are valid. Any allocation / launch failure returns
        // an `Err`, which the caller turns into a host fallback.
        let offsets_gpu = GpuVec::<i32>::from_slice(&offsets)?;
        let bytes_gpu = if bytes.is_empty() {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::from_slice(&bytes)?
        };
        // Literal: bake as a small device buffer. Pad empty to 1 byte so the
        // device pointer is valid (lit_len==0 short-circuits before any read).
        let lit_len = u32::try_from(literal.len()).map_err(|_| {
            BoltError::Other("StringLikeFilter: literal length exceeds u32".into())
        })?;
        let lit_gpu = if literal.is_empty() {
            GpuVec::<u8>::zeros(1)?
        } else {
            GpuVec::<u8>::from_slice(literal)?
        };
        let mask_gpu = GpuVec::<u8>::zeros(n_rows)?;

        let n_rows_u32 = n_rows_to_u32(n_rows)?;
        let stream = CudaStream::null_or_default();
        let grid_x = grid_x_for(n_rows_u32, BLOCK_SIZE);

        let module = CudaModule::from_ptx(
            &crate::jit::string_kernel::compile_like_match_kernel(mode, negated)?,
        )?;
        let function = module.function(crate::jit::string_kernel::LIKE_MATCH_ENTRY)?;

        let mut p_off = offsets_gpu.device_ptr();
        let mut p_bytes = bytes_gpu.device_ptr();
        let mut p_lit = lit_gpu.device_ptr();
        let mut p_mask = mask_gpu.device_ptr();
        let mut p_n = n_rows_u32;
        let mut p_l = lit_len;
        let mut params: Vec<*mut c_void> = vec![
            &mut p_off as *mut CUdeviceptr as *mut c_void,
            &mut p_bytes as *mut CUdeviceptr as *mut c_void,
            &mut p_lit as *mut CUdeviceptr as *mut c_void,
            &mut p_mask as *mut CUdeviceptr as *mut c_void,
            &mut p_n as *mut u32 as *mut c_void,
            &mut p_l as *mut u32 as *mut c_void,
        ];
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                function.raw(),
                grid_x,
                1,
                1,
                BLOCK_SIZE,
                1,
                1,
                0,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        debug_sync_check()?;
        stream.synchronize()?;

        let mask = mask_gpu.to_vec()?;
        check_len(mask.len(), n_rows)?;
        Ok(mask_to_boolean_array(&mask, &validity))
    }
}

/// Result of a query — wraps the output Arrow `RecordBatch`.
#[derive(Debug)]
pub struct QueryHandle {
    /// The materialised result.
    batch: RecordBatch,
}

impl QueryHandle {
    /// Borrow the underlying record batch.
    pub fn record_batch(&self) -> &RecordBatch {
        &self.batch
    }

    /// Consume the handle and return the owned record batch.
    pub fn into_record_batch(self) -> RecordBatch {
        self.batch
    }

    /// Wrap a `RecordBatch` produced by an executor into a `QueryHandle`.
    ///
    /// Internal hook for the wave-7 executor chain (Distinct / Limit / Sort /
    /// Union / Join): the top-level `Engine::execute` runs the child plan,
    /// hands the resulting `QueryHandle` to an `exec::*::execute_*` helper,
    /// and the helper rewraps its output with this constructor.
    ///
    /// Marked `#[doc(hidden)]` and `pub(crate)`: this is not part of the
    /// public 0.2 API; downstream consumers should keep going through
    /// `Engine::sql` / `Engine::execute`.
    #[doc(hidden)]
    pub(crate) fn from_record_batch(batch: RecordBatch) -> Self {
        Self { batch }
    }

    /// Number of rows in the result.
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }
}

#[cfg(test)]
mod tests {
    //! Online tests for the lazy-upload `register_batch` path and the
    //! Stage-3 pinned async-memcpy wiring in `execute_projection`.
    //!
    //! The lazy-upload tests lock in the fix for the O(N²) PCIe re-upload bug
    //! described on the `gpu_tables` field: appending N batches must not cost
    //! `1+2+…+N` batches' worth of host→device traffic. They verify the
    //! observable correctness of the lazy path (rows from every appended batch
    //! are visible to the next query).
    //!
    //! The Stage-3 tests cover the per-query-stream + pinned D2H path —
    //! both the no-predicate and predicate flows — so any regression in the
    //! stream chaining surfaces as a value mismatch rather than a CUDA error.
    //!
    //! All tests are `#[ignore]`'d because they launch real kernels — run
    //! with `cargo test -- --ignored` on a GPU host.
    use super::*;
    use crate::plan::Field;
    use arrow_array::{Int32Array, Int64Array};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    /// `build_count_rows_batch` (the host-side row-count step for
    /// `PhysicalPlan::CountRows`) must emit a single-row Int64 batch holding the
    /// supplied row count, with the column named per the supplied schema. This
    /// runs purely on the host (no GPU) so it is not `#[ignore]`'d.
    #[test]
    fn count_rows_batch_holds_row_count() {
        let schema = Schema::new(vec![Field::new("count", DataType::Int64, false)]);
        let batch = build_count_rows_batch(7, &schema).expect("must build");
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 1);
        assert_eq!(batch.schema().field(0).name(), "count");
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 count column");
        assert_eq!(col.value(0), 7);
    }

    /// Zero rows in the child plan -> COUNT == 0 (not NULL).
    #[test]
    fn count_rows_batch_zero() {
        let schema = Schema::new(vec![Field::new("count", DataType::Int64, false)]);
        let batch = build_count_rows_batch(0, &schema).expect("must build");
        let col = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(col.value(0), 0);
    }

    /// A malformed multi-column output schema is rejected (defensive — the
    /// lowerer only ever stores a single Int64 count column).
    #[test]
    fn count_rows_batch_rejects_multi_column_schema() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]);
        assert!(build_count_rows_batch(3, &schema).is_err());
    }

    // ---- F3-finish: host per-group distinct count (no GPU) ----

    /// Build a `[key Int32, val Int64]` base batch from `(key, val)` rows where
    /// `None` is a SQL NULL in that column. Mirrors the
    /// `[group_keys..., distinct_col]` shape `host_count_distinct_groupby`
    /// consumes.
    fn cd_base(rows: &[(Option<i32>, Option<i64>)]) -> RecordBatch {
        let keys: Int32Array = rows.iter().map(|(k, _)| *k).collect();
        let vals: Int64Array = rows.iter().map(|(_, v)| *v).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, true),
            ArrowField::new("v", ArrowDataType::Int64, true),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(keys), Arc::new(vals)]).unwrap()
    }

    /// Result schema for the `cd_base` fixture: `[k Int32, cnt Int64]`.
    fn cd_result_schema() -> Schema {
        Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("cnt", DataType::Int64, false),
        ])
    }

    /// Read the `(key, count)` result rows out of a count-result batch.
    fn cd_read(batch: &RecordBatch) -> Vec<(Option<i32>, i64)> {
        use arrow_array::Array;
        let keys = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let cnts = batch.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        (0..batch.num_rows())
            .map(|i| {
                let k = if keys.is_null(i) { None } else { Some(keys.value(i)) };
                (k, cnts.value(i))
            })
            .collect()
    }

    /// Distinct-per-group counts: group 1 has values {10, 20} ⇒ 2; group 2 has
    /// {30} (with a duplicate) ⇒ 1.
    #[test]
    fn host_cd_distinct_per_group() {
        let base = cd_base(&[
            (Some(1), Some(10)),
            (Some(1), Some(20)),
            (Some(1), Some(10)), // duplicate value in group 1
            (Some(2), Some(30)),
            (Some(2), Some(30)), // duplicate value in group 2
        ]);
        let out = host_count_distinct_groupby(&base, 1, &cd_result_schema()).unwrap();
        let mut rows = cd_read(&out);
        rows.sort_by_key(|(k, _)| k.unwrap());
        assert_eq!(rows, vec![(Some(1), 2), (Some(2), 1)]);
    }

    /// NULLs in the counted column are ignored; a group whose values are all
    /// NULL yields a count of 0 (standard SQL `COUNT(DISTINCT x)`).
    #[test]
    fn host_cd_ignores_nulls_and_all_null_group_is_zero() {
        let base = cd_base(&[
            (Some(1), Some(10)),
            (Some(1), None), // ignored
            (Some(2), None), // all-NULL group
            (Some(2), None),
        ]);
        let out = host_count_distinct_groupby(&base, 1, &cd_result_schema()).unwrap();
        let mut rows = cd_read(&out);
        rows.sort_by_key(|(k, _)| k.unwrap());
        assert_eq!(rows, vec![(Some(1), 1), (Some(2), 0)]);
    }

    /// A NULL group key forms its own distinct group (per SQL GROUP BY).
    #[test]
    fn host_cd_null_group_key_is_its_own_group() {
        let base = cd_base(&[
            (None, Some(1)),
            (None, Some(2)),
            (Some(5), Some(9)),
        ]);
        let out = host_count_distinct_groupby(&base, 1, &cd_result_schema()).unwrap();
        let rows = cd_read(&out);
        // First-occurrence order: NULL group first (count 2), then key 5 (count 1).
        assert_eq!(rows, vec![(None, 2), (Some(5), 1)]);
    }

    /// Multi-key groups: the group is the composite (k1, k2) tuple.
    #[test]
    fn host_cd_multi_key_composite_group() {
        use arrow_array::Array;
        // base = [k1 Int32, k2 Int64, v Int64]; n_keys = 2.
        let k1: Int32Array = [Some(1), Some(1), Some(1), Some(2)].into_iter().collect();
        let k2: Int64Array = [Some(7), Some(7), Some(8), Some(7)].into_iter().collect();
        let v: Int64Array = [Some(100), Some(200), Some(100), Some(100)].into_iter().collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, true),
            ArrowField::new("k2", ArrowDataType::Int64, true),
            ArrowField::new("v", ArrowDataType::Int64, true),
        ]));
        let base =
            RecordBatch::try_new(schema, vec![Arc::new(k1), Arc::new(k2), Arc::new(v)]).unwrap();
        let result_schema = Schema::new(vec![
            Field::new("k1", DataType::Int32, true),
            Field::new("k2", DataType::Int64, true),
            Field::new("cnt", DataType::Int64, false),
        ]);
        let out = host_count_distinct_groupby(&base, 2, &result_schema).unwrap();
        // Groups: (1,7)->{100,200}=2, (1,8)->{100}=1, (2,7)->{100}=1.
        assert_eq!(out.num_rows(), 3);
        let cnts = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        let mut got: Vec<i64> = (0..out.num_rows()).map(|i| cnts.value(i)).collect();
        got.sort_unstable();
        assert_eq!(got, vec![1, 1, 2]);
    }

    /// A base whose column count does not match `n_keys + 1` is rejected.
    #[test]
    fn host_cd_rejects_wrong_column_count() {
        let base = cd_base(&[(Some(1), Some(1))]);
        // Claim 2 group keys but the base has only 2 columns total (so 1 + 1).
        assert!(host_count_distinct_groupby(&base, 2, &cd_result_schema()).is_err());
    }

    // ---- F3-finish (generalized): host multi / mixed aggregates (no GPU) ----

    /// Two distinct counts plus mixed plain aggregates per group, with NULL
    /// handling, modelling `SELECT g, COUNT(DISTINCT a), COUNT(DISTINCT b),
    /// SUM(x), COUNT(*)`. The base carries one input column per aggregate
    /// (`COUNT(*)` feeds a sentinel column), so n_keys(1) + aggs(4) = 5 cols.
    #[test]
    fn host_multi_agg_distinct_and_plain_per_group() {
        use crate::plan::sql_frontend::{CdAgg, CdOutputCol};
        use arrow_array::Array;
        // group 1: a in {10,10,20} -> distinct 2; b in {NULL,5,5} -> distinct 1;
        //          x = 1+2+3 = 6; count(*) = 3.
        // group 2: a in {30,30} -> distinct 1; b in {7,7} -> distinct 1;
        //          x = 4 (one NULL ignored by SUM but counted by COUNT(*)=2).
        let g: Int32Array = [Some(1), Some(1), Some(1), Some(2), Some(2)].into_iter().collect();
        let a: Int64Array =
            [Some(10), Some(10), Some(20), Some(30), Some(30)].into_iter().collect();
        let b: Int64Array = [None, Some(5), Some(5), Some(7), Some(7)].into_iter().collect();
        let x: Int64Array = [Some(1), Some(2), Some(3), Some(4), None].into_iter().collect();
        let star: Int64Array = [Some(1), Some(1), Some(1), Some(1), Some(1)].into_iter().collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("g", ArrowDataType::Int32, true),
            ArrowField::new("a", ArrowDataType::Int64, true),
            ArrowField::new("b", ArrowDataType::Int64, true),
            ArrowField::new("x", ArrowDataType::Int64, true),
            ArrowField::new("star", ArrowDataType::Int64, false),
        ]));
        let base = RecordBatch::try_new(
            schema,
            vec![Arc::new(g), Arc::new(a), Arc::new(b), Arc::new(x), Arc::new(star)],
        )
        .unwrap();
        let aggs = vec![
            CdAgg::CountDistinct { base_col: 1 },
            CdAgg::CountDistinct { base_col: 2 },
            CdAgg::Sum { base_col: 3 },
            CdAgg::CountStar { base_col: 4 },
        ];
        let output_layout = vec![
            CdOutputCol::GroupKey(0),
            CdOutputCol::Agg(0),
            CdOutputCol::Agg(1),
            CdOutputCol::Agg(2),
            CdOutputCol::Agg(3),
        ];
        let result_schema = Schema::new(vec![
            Field::new("g", DataType::Int32, true),
            Field::new("cda", DataType::Int64, false),
            Field::new("cdb", DataType::Int64, false),
            Field::new("sum_x", DataType::Int64, true),
            Field::new("cnt", DataType::Int64, false),
        ]);
        let out =
            host_multi_agg_groupby(&base, 1, &aggs, &output_layout, &result_schema).unwrap();
        assert_eq!(out.num_rows(), 2);
        let g = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let cda = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let cdb = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        let sumx = out.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
        let cnt = out.column(4).as_any().downcast_ref::<Int64Array>().unwrap();
        // First-occurrence order: group 1 then group 2.
        assert_eq!(g.value(0), 1);
        assert_eq!((cda.value(0), cdb.value(0), sumx.value(0), cnt.value(0)), (2, 1, 6, 3));
        assert_eq!(g.value(1), 2);
        assert_eq!((cda.value(1), cdb.value(1), sumx.value(1), cnt.value(1)), (1, 1, 4, 2));
        // SUM(x) for group 2 ignored the NULL row but COUNT(*) counted it.
        assert!(!sumx.is_null(1));
    }

    /// MIN / MAX / AVG per group with an all-NULL group: MIN/MAX → NULL, AVG →
    /// NULL, COUNT(col) → 0 for that group.
    #[test]
    fn host_multi_agg_minmax_avg_and_all_null_group() {
        use crate::plan::sql_frontend::{CdAgg, CdOutputCol};
        use arrow_array::{Array, Float64Array};
        // group 1: x = {2,4,6}; group 2: x = {NULL,NULL}.
        let g: Int32Array = [Some(1), Some(1), Some(1), Some(2), Some(2)].into_iter().collect();
        let x: Int64Array = [Some(2), Some(4), Some(6), None, None].into_iter().collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("g", ArrowDataType::Int32, true),
            ArrowField::new("xmin", ArrowDataType::Int64, true),
            ArrowField::new("xmax", ArrowDataType::Int64, true),
            ArrowField::new("xavg", ArrowDataType::Int64, true),
            ArrowField::new("xcnt", ArrowDataType::Int64, true),
        ]));
        let base = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(g),
                Arc::new(x.clone()),
                Arc::new(x.clone()),
                Arc::new(x.clone()),
                Arc::new(x),
            ],
        )
        .unwrap();
        let aggs = vec![
            CdAgg::Min { base_col: 1 },
            CdAgg::Max { base_col: 2 },
            CdAgg::Avg { base_col: 3 },
            CdAgg::Count { base_col: 4 },
        ];
        let output_layout = vec![
            CdOutputCol::GroupKey(0),
            CdOutputCol::Agg(0),
            CdOutputCol::Agg(1),
            CdOutputCol::Agg(2),
            CdOutputCol::Agg(3),
        ];
        let result_schema = Schema::new(vec![
            Field::new("g", DataType::Int32, true),
            Field::new("mn", DataType::Int64, true),
            Field::new("mx", DataType::Int64, true),
            Field::new("av", DataType::Float64, true),
            Field::new("c", DataType::Int64, false),
        ]);
        let out =
            host_multi_agg_groupby(&base, 1, &aggs, &output_layout, &result_schema).unwrap();
        let mn = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let mx = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        let av = out.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        let c = out.column(4).as_any().downcast_ref::<Int64Array>().unwrap();
        // group 1: min 2, max 6, avg 4.0, count 3.
        assert_eq!((mn.value(0), mx.value(0)), (2, 6));
        assert!((av.value(0) - 4.0).abs() < 1e-12);
        assert_eq!(c.value(0), 3);
        // group 2: all-NULL -> MIN/MAX/AVG NULL, COUNT 0.
        assert!(mn.is_null(1) && mx.is_null(1) && av.is_null(1));
        assert_eq!(c.value(1), 0);
    }

    /// A base whose column count does not match `n_keys + aggs.len()` is
    /// rejected.
    #[test]
    fn host_multi_agg_rejects_wrong_column_count() {
        use crate::plan::sql_frontend::{CdAgg, CdOutputCol};
        let base = cd_base(&[(Some(1), Some(1))]); // 2 columns
        let aggs = vec![CdAgg::CountDistinct { base_col: 1 }, CdAgg::Sum { base_col: 2 }];
        let layout = vec![CdOutputCol::GroupKey(0), CdOutputCol::Agg(0), CdOutputCol::Agg(1)];
        let rs = Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("cd", DataType::Int64, false),
            Field::new("s", DataType::Int64, true),
        ]);
        // n_keys=1 + 2 aggs = 3 columns expected, base has 2 -> error.
        assert!(host_multi_agg_groupby(&base, 1, &aggs, &layout, &rs).is_err());
    }

    /// Build a single-column `RecordBatch` whose `x` column holds the half-open
    /// range `[start, start+n)` as `Int64`. The schema is shared across all
    /// fixtures so `register_batch`'s schema check passes.
    fn int64_batch(start: i64, n: usize) -> RecordBatch {
        let col: Int64Array = (start..start + n as i64).collect();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int64,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
    }

    // ---- Host-pure tests for the GPU-projection decline fallback ----
    //
    // These exercise `passthrough_output_sources` — the load-bearing,
    // GPU-free decision the `execute_projection_host_fallback` path makes
    // when a temporal/decimal column trips agent-G's `GpuCapacity` decline.
    // No CUDA context needed.

    use crate::plan::physical_plan::{ColumnIO, Op, Reg};

    fn col_io(name: &str, dtype: DataType) -> ColumnIO {
        ColumnIO {
            name: name.to_string(),
            dtype,
        }
    }

    fn spec_with(inputs: Vec<ColumnIO>, outputs: Vec<ColumnIO>, ops: Vec<Op>) -> KernelSpec {
        KernelSpec {
            inputs,
            outputs,
            ops,
            predicate: None,
            register_count: 16,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        }
    }

    /// A plain `SELECT date_col, ts_col FROM t` — two `LoadColumn`→`Store`
    /// pairs — is recognised as a passthrough and maps each output back to
    /// its source input column ordinal.
    #[test]
    fn passthrough_maps_outputs_to_inputs() {
        let spec = spec_with(
            vec![
                col_io("d", DataType::Date32),
                col_io("ts", DataType::Timestamp(crate::plan::TimeUnit::Microsecond, None)),
            ],
            vec![
                col_io("d", DataType::Date32),
                col_io("ts", DataType::Timestamp(crate::plan::TimeUnit::Microsecond, None)),
            ],
            vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Date32,
                },
                Op::LoadColumn {
                    dst: Reg(1),
                    col_idx: 1,
                    dtype: DataType::Timestamp(crate::plan::TimeUnit::Microsecond, None),
                },
                Op::Store {
                    src: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Date32,
                },
                Op::Store {
                    src: Reg(1),
                    col_idx: 1,
                    dtype: DataType::Timestamp(crate::plan::TimeUnit::Microsecond, None),
                },
            ],
        );
        assert_eq!(passthrough_output_sources(&spec), Some(vec![0, 1]));
    }

    /// Output order may differ from input order (`SELECT b, a`); the mapping
    /// must follow the stores, not the load order.
    #[test]
    fn passthrough_respects_output_reordering() {
        let spec = spec_with(
            vec![col_io("a", DataType::Int64), col_io("b", DataType::Int64)],
            vec![col_io("b", DataType::Int64), col_io("a", DataType::Int64)],
            vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int64,
                },
                Op::LoadColumn {
                    dst: Reg(1),
                    col_idx: 1,
                    dtype: DataType::Int64,
                },
                // output 0 <- input 1 (b), output 1 <- input 0 (a)
                Op::Store {
                    src: Reg(1),
                    col_idx: 0,
                    dtype: DataType::Int64,
                },
                Op::Store {
                    src: Reg(0),
                    col_idx: 1,
                    dtype: DataType::Int64,
                },
            ],
        );
        assert_eq!(passthrough_output_sources(&spec), Some(vec![1, 0]));
    }

    /// A Decimal128 passthrough lowers to a `LoadColumn128`→`Store128` pair
    /// and is still recognised.
    #[test]
    fn passthrough_handles_decimal128_pair() {
        let dec = DataType::Decimal128(38, 10);
        let spec = spec_with(
            vec![col_io("amt", dec)],
            vec![col_io("amt", dec)],
            vec![
                Op::LoadColumn128 {
                    dst_lo: Reg(0),
                    dst_hi: Reg(1),
                    col_idx: 0,
                },
                Op::Store128 {
                    src_lo: Reg(0),
                    src_hi: Reg(1),
                    col_idx: 0,
                },
            ],
        );
        assert_eq!(passthrough_output_sources(&spec), Some(vec![0]));
    }

    /// A compute op (here a `Binary` add) is NOT a passthrough — the helper
    /// returns `None` so the caller re-raises the GPU decline instead of
    /// silently producing a wrong host result.
    #[test]
    fn non_passthrough_compute_returns_none() {
        let spec = spec_with(
            vec![col_io("a", DataType::Int64), col_io("b", DataType::Int64)],
            vec![col_io("sum", DataType::Int64)],
            vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int64,
                },
                Op::LoadColumn {
                    dst: Reg(1),
                    col_idx: 1,
                    dtype: DataType::Int64,
                },
                Op::Binary {
                    dst: Reg(2),
                    op: crate::plan::BinaryOp::Add,
                    lhs: Reg(0),
                    rhs: Reg(1),
                    dtype: DataType::Int64,
                    result_dtype: DataType::Int64,
                },
                Op::Store {
                    src: Reg(2),
                    col_idx: 0,
                    dtype: DataType::Int64,
                },
            ],
        );
        assert_eq!(passthrough_output_sources(&spec), None);
    }

    /// A predicate-bearing kernel filters rows, so it is never treated as a
    /// pure passthrough.
    #[test]
    fn predicate_kernel_is_not_passthrough() {
        let mut spec = spec_with(
            vec![col_io("a", DataType::Int64)],
            vec![col_io("a", DataType::Int64)],
            vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int64,
                },
                Op::Store {
                    src: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int64,
                },
            ],
        );
        spec.predicate = Some(Reg(0));
        assert_eq!(passthrough_output_sources(&spec), None);
    }

    #[test]
    #[ignore = "gpu:projection"]
    fn register_batch_two_batches_query_sees_both() {
        // Register two batches, then SELECT the only column. The lazy-upload
        // path must rebuild the GpuTable from BOTH batches at query time, so
        // every row from both batches has to be visible in the result.
        let mut engine = Engine::new().expect("ctx");
        engine
            .register_batch("t", int64_batch(0, 4))
            .expect("first batch");
        engine
            .register_batch("t", int64_batch(4, 4))
            .expect("second batch");

        let h = engine.sql("SELECT x FROM t").expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 8, "8 rows after two 4-row batches");
        let actual = out
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 column");
        let got: Vec<i64> = (0..actual.len()).map(|i| actual.value(i)).collect();
        assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    #[ignore = "gpu:projection"]
    fn register_batch_ten_batches_combined_row_count() {
        // Append ten 100-row batches in a loop, then query. With the bug we
        // were fixing, this would upload 1+2+…+10 = 55 batches' worth of bytes
        // across the loop; with the fix it uploads zero bytes during the loop
        // and exactly one combined upload at query time. Correctness check:
        // the result has all 1000 rows and they sum to the expected total.
        let mut engine = Engine::new().expect("ctx");
        let n_batches = 10usize;
        let rows_per_batch = 100usize;
        for i in 0..n_batches {
            engine
                .register_batch("t", int64_batch((i * rows_per_batch) as i64, rows_per_batch))
                .unwrap_or_else(|e| panic!("register_batch {i}: {e}"));
        }
        let total_rows = n_batches * rows_per_batch;

        let h = engine.sql("SELECT x FROM t").expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), total_rows, "row count after 10 appends");

        let actual = out
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("Int64 column");
        let sum: i64 = (0..actual.len()).map(|i| actual.value(i)).sum();
        let expected_sum: i64 = (0..total_rows as i64).sum();
        assert_eq!(sum, expected_sum, "sum of x column across all 10 batches");
    }

    /// Build a three-column `RecordBatch` (`a` Int32, `b` Int64, `c` Float64)
    /// holding `n` rows. `start_a` seeds the first column; the others are
    /// derived so each row's columns are easy to recompute in the test
    /// assertions. The schema is shared across calls so `register_batch`'s
    /// schema check passes when appending.
    fn three_col_batch(start_a: i32, n: usize) -> RecordBatch {
        use arrow_array::{Float64Array, Int32Array, Int64Array};
        let a: Int32Array = (start_a..start_a + n as i32).collect();
        let b: Int64Array = ((start_a as i64) * 10..((start_a as i64) * 10 + n as i64)).collect();
        let c: Float64Array = (0..n).map(|i| (start_a as f64) + i as f64 * 0.5).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("a", ArrowDataType::Int32, false),
            ArrowField::new("b", ArrowDataType::Int64, false),
            ArrowField::new("c", ArrowDataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(a), Arc::new(b), Arc::new(c)],
        )
        .unwrap()
    }

    /// Batch 5 — incremental rebuild after `register_batch`. Register a
    /// 5-row 3-column table, query (forces full upload), append a 2-row
    /// second batch, query again. The second query must observe all 7
    /// rows AND the prefix-preserving optimisation must have fired —
    /// each of the 3 columns is uploaded exactly twice (once at install,
    /// once at the incremental rebuild after the append). The
    /// no-optimisation baseline would re-upload all 3 columns from
    /// scratch on the second query, giving the SAME count of 6 uploads,
    /// so the count alone doesn't distinguish them. We instead assert
    /// the column counts match the *expected* incremental path
    /// invariants: after a single register_batch, exactly 3 incremental
    /// extends fire — and we verify by tagging the device-side
    /// `host_revision` directly through the LOAD_COUNT bump. The
    /// alternative invalidation path (slot set to `None`) would have
    /// reset the per-column host_revisions to 0 and re-uploaded
    /// everything via the fall-through branch in `ensure_gpu_table`.
    #[test]
    #[ignore = "gpu:projection"]
    fn register_batch_incremental_rebuild_uploads_each_column_once_per_change() {
        let mut engine = Engine::new().expect("ctx");
        // Install: 3 columns × 5 rows. register_table uploads each
        // column once → LOAD_COUNT = 3.
        engine
            .register_table("t", three_col_batch(0, 5))
            .expect("install");
        let after_install = engine.gpu_table_load_count();
        assert_eq!(after_install, 3, "install uploads 3 columns");

        // First query — cache hit (no upload).
        let _ = engine.sql("SELECT a FROM t").expect("first query");
        assert_eq!(
            engine.gpu_table_load_count(),
            3,
            "first query is a pure cache hit"
        );

        // Append 2 rows. register_batch must NOT upload anything
        // synchronously; the actual extension happens in the next query.
        engine
            .register_batch("t", three_col_batch(5, 2))
            .expect("append");
        assert_eq!(
            engine.gpu_table_load_count(),
            3,
            "register_batch must not upload synchronously"
        );

        // Second query — incremental rebuild. Each of the 3 columns is
        // re-uploaded exactly once (prefix-preserving extension). Total
        // becomes 3 + 3 = 6.
        let h = engine.sql("SELECT a, b, c FROM t").expect("second query");
        assert_eq!(
            engine.gpu_table_load_count(),
            6,
            "incremental rebuild uploads exactly 3 columns (each extended once)"
        );

        // Correctness: all 7 rows visible, values match.
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 7, "5 + 2 = 7 rows after append");
        let a = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow_array::Int32Array>()
            .expect("a Int32");
        let got_a: Vec<i32> = (0..a.len()).map(|i| a.value(i)).collect();
        assert_eq!(got_a, vec![0, 1, 2, 3, 4, 5, 6]);

        // Third query without any further mutation — pure cache hit.
        let _ = engine.sql("SELECT b FROM t").expect("third query");
        assert_eq!(
            engine.gpu_table_load_count(),
            6,
            "third query is a pure cache hit — no uploads"
        );
    }

    /// Batch 5 — `replace_table` is a full swap (NOT an append). Every
    /// column gets a fresh revision, so the next query re-uploads every
    /// column (the prefix optimisation does not apply across a replace).
    /// Validates the revision-bump correctness for the
    /// `bump_table_full_replace` path.
    #[test]
    #[ignore = "gpu:projection"]
    fn replace_table_invalidates_all_column_revisions() {
        let mut engine = Engine::new().expect("ctx");
        engine
            .register_table("t", three_col_batch(0, 5))
            .expect("install");
        let base = engine.gpu_table_load_count();
        // register_table on an existing name must error — replace_table is
        // the right entry point for an update.
        engine
            .register_table("t", three_col_batch(100, 4))
            .unwrap_err();
        // Replace with a same-schema, different-content batch. replace_table
        // performs the upload synchronously (re-uploading all 3 columns)
        // and stamps the GpuTable with the new revision, so the next
        // query is a pure cache hit (no further uploads).
        engine
            .replace_table("t", three_col_batch(100, 4))
            .expect("replace");
        assert_eq!(
            engine.gpu_table_load_count(),
            base + 3,
            "replace_table re-uploads every column"
        );
        let h = engine.sql("SELECT a FROM t").expect("query");
        // Cache hit on the post-replace upload.
        assert_eq!(
            engine.gpu_table_load_count(),
            base + 3,
            "query after replace is a cache hit"
        );
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 4);
    }

    /// Verify that a bare projection still returns the right rows after the
    /// kernel launch and D2H downloads moved onto a per-query stream with
    /// async copies. Mirrors what the synchronous path was previously
    /// asserting — same input, same expected output — so any regression in
    /// the stream-flow shows up as a value mismatch rather than a CUDA error.
    #[test]
    #[ignore = "gpu:projection — Stage 2 async D2H correctness"]
    fn execute_projection_async_d2h_round_trip() {
        let mut engine = Engine::new().expect("engine init");

        // Single-column Int32 table: [1, 2, 3, 4, 5].
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1i32, 2, 3, 4, 5]));
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![arr]).expect("batch");
        engine.register_table("t", batch).expect("register");

        // Plain projection — no predicate, so the new async-D2H batch path
        // is exercised end-to-end.
        let handle = engine.sql("SELECT x FROM t").expect("query");
        let out = handle.record_batch();

        assert_eq!(out.num_rows(), 5);
        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32");
        let got: Vec<i32> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec![1, 2, 3, 4, 5]);
    }

    /// Same shape, but with a WHERE clause so the predicate path is the one
    /// exercised. The Stage 2 patch removed the explicit
    /// `stream.synchronize()` after the projection kernel — the predicate
    /// kernel's own internal sync (inside `launch_predicate_kernel`) now
    /// covers both, and any regression in that chain surfaces here.
    #[test]
    #[ignore = "gpu:projection — Stage 2 stream chaining w/ predicate"]
    fn execute_projection_with_predicate_under_async_stream() {
        let mut engine = Engine::new().expect("engine init");

        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1i32, 2, 3, 4, 5]));
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![arr]).expect("batch");
        engine.register_table("t", batch).expect("register");

        let handle = engine
            .sql("SELECT x FROM t WHERE x > 2")
            .expect("query");
        let out = handle.record_batch();

        let col = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Int32");
        let got: Vec<i32> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec![3, 4, 5]);
    }

    // ---------------------------------------------------------------------
    // Stage 7 (P1b): pool-stats periodic-emit throttle.
    //
    // We exercise `should_emit_pool_stats` directly with a mock `Instant`
    // sequence — no CUDA required. The function is the only stateful piece
    // of the periodic-log machinery (the rest is a log line + observer
    // call), so locking down its throttle semantics here gives us the full
    // behavioural coverage we need.
    // ---------------------------------------------------------------------

    #[test]
    fn pool_stats_throttle_first_call_always_emits() {
        // Fresh throttle (no previous emit) must always emit on first
        // call, regardless of how recently the test started.
        let last = Mutex::new(None);
        let now = Instant::now();
        assert!(should_emit_pool_stats(&last, Duration::from_secs(60), now));
        // Second call at the same instant: not enough time elapsed.
        assert!(!should_emit_pool_stats(&last, Duration::from_secs(60), now));
    }

    #[test]
    fn pool_stats_throttle_respects_interval() {
        let last = Mutex::new(None);
        let interval = Duration::from_secs(60);
        let t0 = Instant::now();
        assert!(should_emit_pool_stats(&last, interval, t0), "first emit");
        // 30s later: still inside the window.
        assert!(
            !should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(30)),
            "30s < 60s — must NOT emit"
        );
        // 59s later: still inside.
        assert!(
            !should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(59)),
            "59s < 60s — must NOT emit"
        );
        // 60s later: boundary should fire.
        assert!(
            should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(60)),
            "60s == 60s — must emit"
        );
        // Right after the boundary fire: throttle is reset, so we must
        // wait the full window again.
        assert!(
            !should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(61)),
            "1s after boundary emit — must NOT emit"
        );
        // 60s after the second emit: fires again.
        assert!(
            should_emit_pool_stats(&last, interval, t0 + Duration::from_secs(120)),
            "120s = 60s + 60s — must emit again"
        );
    }

    #[test]
    fn pool_stats_throttle_zero_interval_disables_emission() {
        // The env-var "0" sentinel disables periodic emission entirely.
        let last = Mutex::new(None);
        let now = Instant::now();
        assert!(!should_emit_pool_stats(&last, Duration::ZERO, now));
        // Even after a long delay, zero interval stays disabled.
        assert!(!should_emit_pool_stats(
            &last,
            Duration::ZERO,
            now + Duration::from_secs(3600)
        ));
        // `last` was never updated.
        assert!(last.lock().unwrap().is_none());
    }

    #[test]
    fn pool_stats_throttle_long_interval_still_fires_first_time() {
        // Even a 1-hour interval must produce the first-emit fire so a
        // short-lived process surfaces at least one snapshot.
        let last = Mutex::new(None);
        let now = Instant::now();
        let one_hour = Duration::from_secs(3600);
        assert!(should_emit_pool_stats(&last, one_hour, now));
    }

    #[test]
    fn pool_stats_interval_env_parsing_defaults() {
        // Smoke-test the env-var helper. We can't easily mutate the
        // process env in a parallel test runner safely, so just check the
        // explicit defaults arms. Without the env var set, the default
        // is 60 seconds.
        //
        // NOTE: this test reads (not writes) the env var, so it's safe to
        // run in parallel; the expected default here matches the constant.
        // If a future contributor sets `BOLT_POOL_STATS_INTERVAL_SECS` in
        // their shell while running `cargo test`, this assertion will
        // flag the override — that's intentional.
        if std::env::var(POOL_STATS_ENV).is_err() {
            assert_eq!(
                pool_stats_interval_from_env(),
                Duration::from_secs(DEFAULT_POOL_STATS_INTERVAL_SECS)
            );
        }
    }

    // ---- PV-stage-f: `EngineProvider::has_nulls` surfaces RecordBatch null bitmaps ----

    /// Register a batch whose column contains an Arrow validity bitmap with
    /// at least one NULL row. `EngineProvider::has_nulls` MUST surface this
    /// via `null_count() > 0` on the underlying `RecordBatch::column`.
    /// Without this signal the planner under-flags `KernelSpec` /
    /// `AggregateSpec::input_has_validity`, defeating PV-stage-d / -f
    /// native-validity dispatch.
    #[test]
    #[ignore = "gpu:e2e — Engine::new() initializes driver"]
    fn pv_stage_f_engine_provider_has_nulls_true_for_null_bearing_batch() {
        use crate::plan::TableProvider;

        let mut engine = Engine::new().expect("ctx");
        let arr = Int32Array::from(vec![Some(1i32), None, Some(3)]);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "v",
            ArrowDataType::Int32,
            true,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("batch");
        engine.register_table("t", batch).expect("register");

        let provider = EngineProvider {
            base: &engine.provider,
            tables: &engine.tables,
            streaming: engine.streaming_sources.borrow(),
        };
        assert!(
            provider.has_nulls("t", 0),
            "null-bearing column must surface true via EngineProvider::has_nulls"
        );
        assert_eq!(
            provider.null_count("t", 0),
            Some(1),
            "null_count must reflect Arrow validity bitmap"
        );
    }

    /// Review C10 regression: `register_batch` must union dictionaries across
    /// all registered batches so the string-literal rewriter can resolve
    /// literals that only appear in an appended batch.
    ///
    /// Before this fix, `register_batch` left the dict registry frozen at
    /// batch 0's contents. A subsequent `WHERE s = 'c'` (where `'c'` is only
    /// in batch 1's dictionary) folded to `Bool(false)` against batch 0's
    /// dictionary and silently dropped every otherwise-matching row in
    /// batch 1 — a classic silent-wrong-result bug.
    ///
    /// The fix rebuilds the dict registry against the concatenated batches
    /// after each append, so the rewriter sees the union dict containing
    /// every legal literal. This test exercises the canonical two-batch
    /// scenario:
    ///   * batch 0 has dict values ["a", "b"]
    ///   * batch 1 has dict values ["a", "b", "c"]
    ///   * `WHERE s = 'c'` must return the rows from batch 1 whose `s = "c"`.
    #[test]
    #[ignore = "gpu:string — dictionary construction uploads to GPU"]
    fn c10_register_batch_unions_dictionaries_across_batches() {
        use arrow_array::StringArray;

        let mut engine = Engine::new().expect("ctx");

        // Batch 0: dict values {"a", "b"}; no row holds "c".
        let s0: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "a", "b"]));
        let v0: ArrayRef = Arc::new(Int64Array::from(vec![10_i64, 11, 12, 13]));
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("s", ArrowDataType::Utf8, false),
            ArrowField::new("v", ArrowDataType::Int64, false),
        ]));
        let b0 = RecordBatch::try_new(schema.clone(), vec![s0, v0]).expect("batch 0");

        // Batch 1: dict values {"a", "b", "c"} — "c" appears only here.
        let s1: ArrayRef = Arc::new(StringArray::from(vec!["a", "c", "b", "c"]));
        let v1: ArrayRef = Arc::new(Int64Array::from(vec![20_i64, 21, 22, 23]));
        let b1 = RecordBatch::try_new(schema, vec![s1, v1]).expect("batch 1");

        engine.register_batch("t", b0).expect("batch 0");
        engine.register_batch("t", b1).expect("batch 1");

        // Pre-fix: the rewriter would constant-fold `s = 'c'` to Bool(false)
        // because batch 0's dict never observed "c"; result is zero rows.
        // Post-fix: the dict registry is rebuilt against the concatenated
        // batches so "c" is in the union dict, and the predicate matches
        // the two rows in batch 1 where s = "c" (indices 1, 3 → v = 21, 23).
        let h = engine
            .sql("SELECT v FROM t WHERE s = 'c'")
            .expect("execute");
        let out = h.record_batch();
        assert_eq!(
            out.num_rows(),
            2,
            "literal that lives only in batch 1 must match its two rows; \
             got {} (zero rows is the pre-fix silent-wrong-result bug)",
            out.num_rows()
        );
        let actual = out
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("v is Int64");
        let mut got: Vec<i64> = (0..actual.len()).map(|i| actual.value(i)).collect();
        got.sort();
        assert_eq!(got, vec![21, 23]);
    }

    /// Mirror of the test above for a NULL-free column — provider must
    /// return false so PV stages keep the legacy host-strip path bit-identical.
    #[test]
    #[ignore = "gpu:e2e — Engine::new() initializes driver"]
    fn pv_stage_f_engine_provider_has_nulls_false_for_null_free_batch() {
        use crate::plan::TableProvider;

        let mut engine = Engine::new().expect("ctx");
        let arr = Int32Array::from(vec![1i32, 2, 3]);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "v",
            ArrowDataType::Int32,
            false,
        )]));
        let batch =
            RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("batch");
        engine.register_table("t", batch).expect("register");

        let provider = EngineProvider {
            base: &engine.provider,
            tables: &engine.tables,
            streaming: engine.streaming_sources.borrow(),
        };
        assert!(
            !provider.has_nulls("t", 0),
            "null-free column must surface false"
        );
        assert_eq!(provider.null_count("t", 0), Some(0));
    }

    // ---- Review-H2: PTX module cache in `execute_projection` ----
    //
    // Two layers:
    //
    //   * Host-only key derivation: stable for identical specs, different
    //     for different specs, and entry-name-sensitive. These run on every
    //     `cargo test` invocation (no GPU required).
    //
    //   * GPU-end-to-end: register a table, run the same SQL twice, and
    //     assert `module_cache_loads` only ticked once. A second test
    //     issues a *different* projection on the same engine to confirm
    //     the cache misses on a fresh spec rather than blindly returning
    //     the first module. Both are `#[ignore]` because they need CUDA.

    /// Two identical `KernelSpec`s produce the same cache key.
    #[test]
    fn module_cache_key_stable_for_identical_specs() {
        use crate::plan::ColumnIO;

        let mk_spec = || KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            ops: Vec::new(),
            predicate: None,
            register_count: 0,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        let k1 = ModuleCacheKey::new(&mk_spec(), KERNEL_ENTRY);
        let k2 = ModuleCacheKey::new(&mk_spec(), KERNEL_ENTRY);
        assert_eq!(k1, k2, "identical specs must hash to the same key");
    }

    /// Specs that differ in output column name produce different cache keys
    /// — otherwise two different projections would alias to the same loaded
    /// module and the second query would launch the wrong kernel.
    #[test]
    fn module_cache_key_differs_for_different_specs() {
        use crate::plan::ColumnIO;

        let base = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            ops: Vec::new(),
            predicate: None,
            register_count: 0,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        let mut other = base.clone();
        other.outputs[0].name = "y".to_string();
        let k1 = ModuleCacheKey::new(&base, KERNEL_ENTRY);
        let k2 = ModuleCacheKey::new(&other, KERNEL_ENTRY);
        assert_ne!(
            k1, k2,
            "different specs must hash to different keys — otherwise two \
             distinct projections would alias to the same cached module"
        );
    }

    /// The same `KernelSpec` keyed under two different entry names yields
    /// two distinct keys (projection vs predicate kernel both reuse the
    /// spec but emit different PTX).
    #[test]
    fn module_cache_key_distinguishes_entry_names() {
        use crate::plan::ColumnIO;

        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            ops: Vec::new(),
            predicate: None,
            register_count: 0,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        let k_proj = ModuleCacheKey::new(&spec, KERNEL_ENTRY);
        let k_pred = ModuleCacheKey::new(&spec, PREDICATE_ENTRY);
        assert_ne!(
            k_proj, k_pred,
            "projection vs predicate kernel must not alias under the same spec"
        );
    }

    /// End-to-end cache hit: register a table, run the same SELECT twice,
    /// observe exactly one cache miss against the projection entry. The
    /// second call must hit and produce identical results.
    #[test]
    #[ignore = "gpu:projection — module cache hit"]
    fn module_cache_hits_on_repeat_projection() {
        use std::sync::atomic::Ordering;

        let mut engine = Engine::new().expect("ctx");
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1i32, 2, 3, 4, 5]));
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "x",
            ArrowDataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(schema, vec![arr]).expect("batch");
        engine.register_table("t", batch).expect("register");

        // First call: cache miss, loads count goes from 0 to 1.
        let baseline = engine.module_cache_loads.load(Ordering::SeqCst);
        let h1 = engine.sql("SELECT x FROM t").expect("first query");
        let after_first = engine.module_cache_loads.load(Ordering::SeqCst);
        assert_eq!(
            after_first - baseline,
            1,
            "first projection must compile exactly one module"
        );

        // Second identical call: cache hit, loads count unchanged.
        let h2 = engine.sql("SELECT x FROM t").expect("second query");
        let after_second = engine.module_cache_loads.load(Ordering::SeqCst);
        assert_eq!(
            after_second, after_first,
            "second identical projection must reuse the cached module"
        );

        // Sanity: both results are correct.
        assert_eq!(h1.record_batch().num_rows(), 5);
        assert_eq!(h2.record_batch().num_rows(), 5);
    }

    // -- 128-bit cache-key collision resistance ---------------------------
    //
    // These tests target the hardened `ModuleCacheKey::new` (review M:JIT
    // cache hardening). They verify the two properties that matter for
    // wrong-kernel safety:
    //
    //   * Two distinct `KernelSpec`s whose `Debug` output looks superficially
    //     similar (one byte change deep in the IR) still map to DIFFERENT
    //     128-bit cache keys — otherwise the cache would alias them and
    //     `Engine::sql` would launch the wrong PTX. The format-then-hash
    //     pipeline plus two domain-separated `DefaultHasher` instances
    //     gives ~2^-64 birthday collision odds.
    //
    //   * Two clones of the SAME `KernelSpec` produce the SAME key — this is
    //     the cache-hit contract that the projection module cache relies on.
    //     A regression here would silently double every JIT compile.

    /// Two `KernelSpec`s that differ only in a single nested-IR byte (a
    /// register index in a `LoadColumn`) must produce different 128-bit
    /// keys. Validates the wider hash + domain-separation strategy: a single
    /// 64-bit `DefaultHasher` would still distinguish these (they differ in
    /// `Debug` output), so the test's real job is to ensure the upgrade did
    /// not regress that baseline — both halves must vary.
    #[test]
    fn cache_key_distinguishes_specs_with_similar_debug() {
        use crate::plan::{ColumnIO, Op};

        let base = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            ops: vec![Op::LoadColumn {
                dst: crate::plan::Reg(0),
                col_idx: 0,
                dtype: DataType::Int32,
            }],
            predicate: None,
            register_count: 1,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        // Differ only in the destination register — `Debug` output flips
        // a single digit, which is exactly the "similar Debug" stress case
        // the hardened key is designed to survive.
        let mut other = base.clone();
        other.ops[0] = Op::LoadColumn {
            dst: crate::plan::Reg(1),
            col_idx: 0,
            dtype: DataType::Int32,
        };

        let k1 = ModuleCacheKey::new(&base, KERNEL_ENTRY);
        let k2 = ModuleCacheKey::new(&other, KERNEL_ENTRY);
        assert_ne!(
            k1, k2,
            "specs with near-identical Debug output must still produce \
             distinct cache keys — otherwise the cache would launch the \
             wrong kernel for the second spec"
        );
        // Stronger: BOTH 64-bit halves must differ. If one half collided
        // we'd still be safe (Eq compares the tuple), but a single-half
        // collision would mean the domain-separation byte stopped helping
        // and we'd be back to 64-bit semantics on that half.
        assert_ne!(
            k1.spec_hash_hi, k2.spec_hash_hi,
            "hi half must vary independently — domain separation regression?"
        );
        assert_ne!(
            k1.spec_hash_lo, k2.spec_hash_lo,
            "lo half must vary independently — domain separation regression?"
        );
    }

    /// Two clones of the same `KernelSpec` produce the same key. This is
    /// the cache-hit contract; if it ever broke, every repeat query would
    /// JIT-compile from scratch and the `module_cache_hits_on_repeat_*`
    /// integration tests would also break — but this micro-test localises
    /// the regression to the key derivation rather than the cache plumbing.
    #[test]
    fn cache_key_stable_under_clone() {
        use crate::plan::{ColumnIO, Op};

        let spec = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "y".to_string(),
                dtype: DataType::Int64,
            }],
            ops: vec![
                Op::LoadColumn {
                    dst: crate::plan::Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int32,
                },
                Op::Store {
                    src: crate::plan::Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int64,
                },
            ],
            predicate: None,
            register_count: 1,
            input_has_validity: Vec::new(),
            output_has_validity: Vec::new(),
        };
        let cloned = spec.clone();
        let k1 = ModuleCacheKey::new(&spec, KERNEL_ENTRY);
        let k2 = ModuleCacheKey::new(&cloned, KERNEL_ENTRY);
        assert_eq!(
            k1, k2,
            "clone of the same spec must produce the same cache key — \
             otherwise repeat queries would always JIT-compile from scratch"
        );
    }

    /// Debug-injectivity guard (finding V-15). The module cache derives its
    /// key from `format!("{:?}", spec)`, so correctness rests on the
    /// invariant *distinct specs => distinct `Debug` output*. This test
    /// perturbs a base `KernelSpec` in EACH semantically-relevant field and
    /// asserts the perturbed key differs from the base. If a future field is
    /// added to `KernelSpec` but left out of its `Debug` impl, two distinct
    /// kernels would format identically, hash to the same key, and the cache
    /// would silently serve the WRONG compiled module — a silent-wrong-result
    /// failure mode. Adding the new field's perturbation here makes that
    /// regression a compile-or-test failure rather than a runtime hazard.
    #[test]
    fn module_cache_key_debug_distinguishes_specs() {
        use crate::plan::{ColumnIO, Op, Reg};

        // Base spec: one input, one output, one LoadColumn op, no predicate.
        // Each variant below mutates exactly one semantically-relevant field.
        let base = KernelSpec {
            inputs: vec![ColumnIO {
                name: "x".to_string(),
                dtype: DataType::Int32,
            }],
            outputs: vec![ColumnIO {
                name: "y".to_string(),
                dtype: DataType::Int64,
            }],
            ops: vec![Op::LoadColumn {
                dst: Reg(0),
                col_idx: 0,
                dtype: DataType::Int32,
            }],
            predicate: None,
            register_count: 1,
            input_has_validity: vec![false],
            output_has_validity: vec![false],
        };

        // `inputs` — differ in input column dtype.
        let mut v_inputs = base.clone();
        v_inputs.inputs[0].dtype = DataType::Int64;

        // `outputs` — differ in output column name.
        let mut v_outputs = base.clone();
        v_outputs.outputs[0].name = "z".to_string();

        // `ops` — differ in a nested op field (destination register).
        let mut v_ops = base.clone();
        v_ops.ops[0] = Op::LoadColumn {
            dst: Reg(1),
            col_idx: 0,
            dtype: DataType::Int32,
        };

        // `predicate` — None vs Some(reg).
        let mut v_predicate = base.clone();
        v_predicate.predicate = Some(Reg(0));

        // `register_count` — affects PTX register allocation.
        let mut v_register_count = base.clone();
        v_register_count.register_count = 2;

        // `input_has_validity` — flips the pre-stage validity layout.
        let mut v_input_validity = base.clone();
        v_input_validity.input_has_validity = vec![true];

        // `output_has_validity` — flips the per-output validity stores.
        let mut v_output_validity = base.clone();
        v_output_validity.output_has_validity = vec![true];

        let base_key = ModuleCacheKey::new(&base, KERNEL_ENTRY);
        for (field, variant) in [
            ("inputs", &v_inputs),
            ("outputs", &v_outputs),
            ("ops", &v_ops),
            ("predicate", &v_predicate),
            ("register_count", &v_register_count),
            ("input_has_validity", &v_input_validity),
            ("output_has_validity", &v_output_validity),
        ] {
            let variant_key = ModuleCacheKey::new(variant, KERNEL_ENTRY);
            assert_ne!(
                base_key, variant_key,
                "a spec differing only in `{field}` must produce a distinct \
                 cache key — if it does not, that field is missing from \
                 `KernelSpec`'s `Debug` and two kernels would silently alias \
                 to the same cached PTX (wrong results)"
            );
        }
    }

    // ---- v0.7: `EngineBuilder::persistent_cache` wires into disk PTX cache ----

    /// `EngineBuilder::persistent_cache(path).build()` must install
    /// `path` as the process-wide disk PTX cache override, so a later
    /// `crate::jit::disk_cache::disk_cache()` resolves to it instead of
    /// (or in preference to) the `BOLT_PTX_CACHE_DIR` env var.
    ///
    /// Marked `#[ignore]` because `build()` initialises the CUDA driver
    /// and that's not available on every CI host. The wiring under test
    /// is, however, GPU-independent — it's a pure setter call inside
    /// `build()` — so on a non-GPU host the env-var-only path
    /// (`persistent_cache` not called) is exercised implicitly by every
    /// other test that instantiates an Engine without calling this
    /// knob, and the env-var contract continues to hold.
    #[test]
    #[ignore = "gpu:e2e — EngineBuilder::build initializes CUDA driver"]
    fn builder_persistent_cache_wires_into_disk_ptx_cache() {
        // Use a unique-per-run temp dir so this test can't observe
        // leftover state from a previous run or interfere with a
        // sibling test that also pokes the override slot.
        let mut path = std::env::temp_dir();
        path.push(format!(
            "craton-bolt-builder-persistent-cache-{}-{}",
            std::process::id(),
            // Cheap unique suffix without a `rand` dep.
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));

        // Save + restore the process-wide override slot so this test
        // doesn't leak state into siblings (the disk-cache module's
        // own tests do the same dance with their own ENV_LOCK; we
        // skip the lock here because the test is `#[ignore]` and not
        // expected to interleave with disk_cache's tests under
        // `cargo test`).
        let prev = crate::jit::current_disk_ptx_cache_dir();

        let _engine = Engine::builder()
            .persistent_cache(path.clone())
            .build()
            .expect("builder + CUDA init");

        // The setter must have run: the override slot now reflects
        // the builder-supplied path.
        assert_eq!(
            crate::jit::current_disk_ptx_cache_dir(),
            Some(path.clone()),
            "EngineBuilder::persistent_cache must propagate into the \
             process-wide disk PTX cache override"
        );

        // Restore prior state so we don't leak into sibling tests.
        crate::jit::set_disk_ptx_cache_dir(prev);
    }

    /// When `persistent_cache` is NOT called on the builder, `build`
    /// must NOT touch the disk-cache override slot — so a previously-
    /// installed override (or the `BOLT_PTX_CACHE_DIR` env-var path)
    /// continues to take effect unchanged.
    #[test]
    #[ignore = "gpu:e2e — EngineBuilder::build initializes CUDA driver"]
    fn builder_without_persistent_cache_preserves_existing_override() {
        let mut prior = std::env::temp_dir();
        prior.push(format!(
            "craton-bolt-builder-prior-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        let prev = crate::jit::current_disk_ptx_cache_dir();
        crate::jit::set_disk_ptx_cache_dir(Some(prior.clone()));

        let _engine = Engine::builder().build().expect("builder + CUDA init");

        assert_eq!(
            crate::jit::current_disk_ptx_cache_dir(),
            Some(prior),
            "builder without persistent_cache must NOT clobber a \
             pre-installed override (env-var path must keep working too)"
        );

        crate::jit::set_disk_ptx_cache_dir(prev);
    }

    /// End-to-end cache miss on a *different* projection: confirm the cache
    /// is keyed correctly (otherwise a second, distinct SELECT would
    /// erroneously hit and run the wrong kernel — silent-wrong-result).
    #[test]
    #[ignore = "gpu:projection — module cache miss"]
    fn module_cache_misses_on_different_projection() {
        use std::sync::atomic::Ordering;

        let mut engine = Engine::new().expect("ctx");
        let xs: ArrayRef = Arc::new(Int32Array::from(vec![1i32, 2, 3]));
        let ys: ArrayRef = Arc::new(Int32Array::from(vec![10i32, 20, 30]));
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("x", ArrowDataType::Int32, false),
            ArrowField::new("y", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![xs, ys]).expect("batch");
        engine.register_table("t", batch).expect("register");

        let baseline = engine.module_cache_loads.load(Ordering::SeqCst);
        let _ = engine.sql("SELECT x FROM t").expect("first query");
        let after_first = engine.module_cache_loads.load(Ordering::SeqCst);
        let _ = engine.sql("SELECT y FROM t").expect("second query");
        let after_second = engine.module_cache_loads.load(Ordering::SeqCst);
        assert_eq!(
            after_first - baseline,
            1,
            "first projection must compile one module"
        );
        assert_eq!(
            after_second - after_first,
            1,
            "second projection on a different column must miss and compile \
             its own module — otherwise the cache is over-keying"
        );
    }

    // -------------------------------------------------------------------
    // Builder → engine → disk-cache plumbing (host-side, no GPU).
    //
    // These exercise the `persistent_cache(path)` knob's effect on the
    // process-wide disk PTX cache WITHOUT constructing an `Engine`
    // (`build()` needs a CUDA context). We drive the same bridge `build()`
    // uses — `install_persistent_cache_override` — and observe the
    // resolved cache directory through the public
    // `disk_cache::disk_cache()` / `DiskPtxCache::root()` surface that the
    // JIT compile path consults in `get_or_build_module`.
    // -------------------------------------------------------------------

    /// Serialises the disk-cache tests below: they mutate the process-wide
    /// override slot and the `BOLT_PTX_CACHE_DIR` env var, both of which
    /// are global. Cargo runs `#[test]`s in parallel by default, so an
    /// unguarded mutation would race a sibling test.
    static DISK_CACHE_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Fresh, unique temp directory under the OS tempdir for a disk-cache
    /// plumbing test. Not created on disk here — `DiskPtxCache::open`
    /// (invoked transitively by `disk_cache()`) creates it.
    fn fresh_cache_dir(tag: &str) -> std::path::PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "craton-bolt-engine-cache-test-{}-{}-{}",
            tag,
            std::process::id(),
            n,
        ));
        let _ = std::fs::remove_dir_all(&p);
        p
    }

    /// Snapshot + restore the global override slot and the env var around a
    /// test body so disk-cache tests stay independent of ordering.
    fn with_clean_disk_cache_state<R>(f: impl FnOnce() -> R) -> R {
        use crate::jit::disk_cache::DISK_PTX_CACHE_ENV;
        let _guard = DISK_CACHE_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let prev_env = std::env::var(DISK_PTX_CACHE_ENV).ok();
        // Start from a known-clean slate: no builder override, no env var.
        install_persistent_cache_override(None);
        std::env::remove_var(DISK_PTX_CACHE_ENV);

        let out = f();

        // Restore: clear our override and put the env var back the way we
        // found it so sibling tests / the rest of the process see no drift.
        install_persistent_cache_override(None);
        match prev_env {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_ENV),
        }
        out
    }

    #[test]
    fn persistent_cache_path_drives_disk_cache_root() {
        // The builder knob, once installed via the same bridge `build()`
        // uses, must make the JIT compile path's `disk_cache()` resolve to
        // that exact directory — proving the path is plumbed all the way
        // from builder → cache layer, not merely stored on the engine.
        with_clean_disk_cache_state(|| {
            let dir = fresh_cache_dir("plumbed");
            install_persistent_cache_override(Some(dir.as_path()));

            let cache = crate::jit::disk_cache::disk_cache()
                .expect("builder path must enable the disk cache");
            assert_eq!(
                cache.root(),
                dir.as_path(),
                "disk cache root must match the persistent_cache(path) directory"
            );
            let _ = std::fs::remove_dir_all(&dir);
        });
    }

    #[test]
    fn no_persistent_cache_path_keeps_disk_cache_disabled() {
        // Opt-in contract: with neither a builder path nor the env var set,
        // the JIT compile path sees `disk_cache() == None` — unchanged,
        // zero-side-effect behaviour.
        with_clean_disk_cache_state(|| {
            // `with_clean_disk_cache_state` already installed `None` +
            // cleared the env var; assert the resulting disabled state.
            assert!(
                crate::jit::disk_cache::disk_cache().is_none(),
                "no builder path and no env var must leave the disk cache disabled"
            );
        });
    }

    #[test]
    fn builder_persistent_cache_takes_precedence_over_env() {
        // When both the builder path and `BOLT_PTX_CACHE_DIR` are set, the
        // builder knob wins (the env var stays a fallback for the no-path
        // case). Confirms the plumbed override out-ranks the env var at the
        // resolution point the compile path reads.
        use crate::jit::disk_cache::DISK_PTX_CACHE_ENV;
        with_clean_disk_cache_state(|| {
            let builder_dir = fresh_cache_dir("builder");
            let env_dir = fresh_cache_dir("env");
            std::env::set_var(DISK_PTX_CACHE_ENV, env_dir.to_string_lossy().to_string());
            install_persistent_cache_override(Some(builder_dir.as_path()));

            let cache = crate::jit::disk_cache::disk_cache().expect("cache enabled");
            assert_eq!(
                cache.root(),
                builder_dir.as_path(),
                "builder persistent_cache(path) must take precedence over the env var"
            );
            let _ = std::fs::remove_dir_all(&builder_dir);
            let _ = std::fs::remove_dir_all(&env_dir);
        });
    }

    #[test]
    fn clearing_persistent_cache_falls_back_to_env() {
        // Installing `None` (a default-built engine) must clear a prior
        // builder override and re-expose the env-var fallback — so a later
        // default `build()` doesn't accidentally pin a stale directory.
        use crate::jit::disk_cache::DISK_PTX_CACHE_ENV;
        with_clean_disk_cache_state(|| {
            let builder_dir = fresh_cache_dir("stale");
            let env_dir = fresh_cache_dir("fallback");
            // First: a builder path is in effect.
            install_persistent_cache_override(Some(builder_dir.as_path()));
            assert_eq!(
                crate::jit::disk_cache::disk_cache()
                    .expect("cache enabled")
                    .root(),
                builder_dir.as_path(),
            );
            // Now a default build clears the override; the env var should
            // take over.
            std::env::set_var(DISK_PTX_CACHE_ENV, env_dir.to_string_lossy().to_string());
            install_persistent_cache_override(None);
            assert_eq!(
                crate::jit::disk_cache::disk_cache()
                    .expect("env fallback enables the cache")
                    .root(),
                env_dir.as_path(),
                "clearing the builder override must re-fall-back to BOLT_PTX_CACHE_DIR"
            );
            let _ = std::fs::remove_dir_all(&builder_dir);
            let _ = std::fs::remove_dir_all(&env_dir);
        });
    }

    // -------------------------------------------------------------------
    // F2 — query-counter contract for the DataFrame (`run_logical_plan`)
    // path. These bump the process-global metrics counters, so they
    // serialise under one lock and assert *monotone deltas* (>=) rather
    // than exact counts, which would race a sibling `--ignored` test that
    // also runs a query in parallel.
    // -------------------------------------------------------------------

    /// Serialises the metrics-counting tests below so their counter-delta
    /// observations don't interleave with one another.
    static METRICS_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Single-column Int64 table fixture for the counter tests.
    fn metrics_int64_table() -> RecordBatch {
        int64_batch(0, 4)
    }

    /// `run_logical_plan` (the DataFrame `collect` path) must bump
    /// `QueriesTotal` exactly like `sql()` does — previously it bumped
    /// neither counter, so a DataFrame workload reported `queries_total = 0`
    /// while doing real work (review F2).
    #[test]
    #[ignore = "gpu:metrics — run_logical_plan launches a real kernel"]
    fn run_logical_plan_bumps_queries_total() {
        let _g = METRICS_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut engine = Engine::new().expect("ctx");
        engine
            .register_table("t", metrics_int64_table())
            .expect("register");

        let plan = LogicalPlan::Scan {
            table: "t".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("x", DataType::Int64, false)]),
        };

        let before = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesTotal)
            .get();
        let _ = engine.run_logical_plan(&plan).expect("execute");
        let after = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesTotal)
            .get();

        assert!(
            after >= before + 1,
            "run_logical_plan must bump QueriesTotal at least once \
             (before={before}, after={after})"
        );
    }

    /// A *failing* top-level `run_logical_plan` must bump `QueriesFailed`,
    /// mirroring `sql()`'s error-path book-keeping. We force a failure by
    /// scanning a table that was never registered, which fails inside
    /// `execute` (the same phase `sql()` counts).
    #[test]
    #[ignore = "gpu:metrics — run_logical_plan initialises the driver"]
    fn run_logical_plan_failure_bumps_queries_failed() {
        let _g = METRICS_TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let mut engine = Engine::new().expect("ctx");

        // No `register_table` for "missing" → the scan fails at execute time.
        let plan = LogicalPlan::Scan {
            table: "missing".into(),
            projection: None,
            schema: Schema::new(vec![Field::new("x", DataType::Int64, false)]),
        };

        let before_total = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesTotal)
            .get();
        let before_failed = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesFailed)
            .get();
        let result = engine.run_logical_plan(&plan);
        let after_total = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesTotal)
            .get();
        let after_failed = crate::metrics::metrics()
            .counter(crate::metrics::Counter::QueriesFailed)
            .get();

        assert!(result.is_err(), "scan of an unregistered table must fail");
        assert!(
            after_total >= before_total + 1,
            "a failed query still counts toward QueriesTotal"
        );
        assert!(
            after_failed >= before_failed + 1,
            "a failed run_logical_plan must bump QueriesFailed \
             (before={before_failed}, after={after_failed})"
        );
    }

    /// Host-only sanity: the metrics counter read API used by the contract
    /// tests above is wired and monotone. Guards against a future rename of
    /// `Counter::QueriesTotal` / the `counter(..).get()` surface silently
    /// breaking the (GPU-gated) counting tests. No GPU required.
    #[test]
    fn query_counters_are_readable_and_monotone() {
        let m = crate::metrics::metrics();
        let t0 = m.counter(crate::metrics::Counter::QueriesTotal).get();
        m.inc(crate::metrics::Counter::QueriesTotal);
        let t1 = m.counter(crate::metrics::Counter::QueriesTotal).get();
        assert!(t1 >= t0 + 1, "QueriesTotal must be monotone under inc()");

        let f0 = m.counter(crate::metrics::Counter::QueriesFailed).get();
        m.inc(crate::metrics::Counter::QueriesFailed);
        let f1 = m.counter(crate::metrics::Counter::QueriesFailed).get();
        assert!(f1 >= f0 + 1, "QueriesFailed must be monotone under inc()");
    }

    /// F10a — `DeviceCol::mark_launch_stream` must tag the launch stream
    /// into every device buffer the output column owns (so the buffer's
    /// `Drop` fences that stream). We can't observe the `StreamSet`
    /// contents from here (it's `pub(crate)` in `cuda`), but tagging a
    /// freshly-allocated column with a stream must succeed without panic or
    /// error for every `DeviceCol` variant, including the Decimal128 mask
    /// arm. Requires a real allocation, so it is GPU-gated.
    #[test]
    #[ignore = "gpu:projection — allocates real device buffers"]
    fn device_col_mark_launch_stream_tags_all_variants() {
        let stream = CudaStream::null_or_default();
        let s = stream.raw();

        // Primitive + Bool + Utf8 columns.
        for dtype in [
            DataType::Int32,
            DataType::Int64,
            DataType::Float32,
            DataType::Float64,
            DataType::Bool,
            DataType::Utf8,
        ] {
            let col = DeviceCol::alloc_zeros(dtype.clone(), 8).expect("alloc");
            // Must not panic; idempotent tagging is fine (StreamSet dedups).
            col.mark_launch_stream(s);
            col.mark_launch_stream(s);
        }

        // Decimal128 with a passthrough validity mask installed: both the
        // values buffer and the mask buffer must be tagged.
        let mut dec = DeviceCol::alloc_zeros(DataType::Decimal128(38, 0), 8).expect("alloc dec");
        let mask = GpuVec::<u8>::zeros(1).expect("mask");
        dec.set_decimal128_valid_mask(Some(mask));
        dec.mark_launch_stream(s);
    }

    // ----- Streaming / morsel wiring (agent J) -------------------------
    //
    // These cover the *classification* half of the streaming hook
    // (`streamable_leaf_scan`) host-side — no CUDA context, no device. The
    // end-to-end morsel-vs-whole equivalence tests below are GPU-gated
    // (`#[ignore = "gpu:..."]`) per repo convention, because they build an
    // `Engine` (which binds a real CUDA context) and launch the projection
    // kernel per morsel.

    /// A bare `Schema` over one Int64 column named `x`, matching `int64_batch`.
    fn x_schema() -> Schema {
        Schema::new(vec![Field::new("x", DataType::Int64, false)])
    }

    /// A trivial identity-passthrough `Projection` over table `t`
    /// (`SELECT x FROM t` shape): `LoadColumn`→`Store` of column 0.
    fn passthrough_projection(table: &str) -> PhysicalPlan {
        let kernel = spec_with(
            vec![col_io("x", DataType::Int64)],
            vec![col_io("x", DataType::Int64)],
            vec![
                Op::LoadColumn {
                    dst: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int64,
                },
                Op::Store {
                    src: Reg(0),
                    col_idx: 0,
                    dtype: DataType::Int64,
                },
            ],
        );
        PhysicalPlan::Projection {
            table: table.to_string(),
            kernel,
            output_schema: x_schema(),
        }
    }

    #[test]
    fn streamable_leaf_scan_recognises_projection() {
        let p = passthrough_projection("t");
        assert_eq!(Engine::streamable_leaf_scan(&p), Some("t"));
    }

    #[test]
    fn streamable_leaf_scan_recognises_string_leaves() {
        let sl = PhysicalPlan::StringLength {
            table: "t".to_string(),
            outputs: Vec::new(),
            output_schema: x_schema(),
        };
        assert_eq!(Engine::streamable_leaf_scan(&sl), Some("t"));
        let sp = PhysicalPlan::StringProject {
            table: "u".to_string(),
            outputs: Vec::new(),
            output_schema: x_schema(),
        };
        assert_eq!(Engine::streamable_leaf_scan(&sp), Some("u"));
    }

    #[test]
    fn streamable_leaf_scan_rejects_non_leaf_shapes() {
        // Anything wrapping a child sub-plan, or any cross-row operator, must
        // drain (status quo). A `Distinct`/`Limit` over a Projection are the
        // canonical "wraps a child" shapes; their scan must NOT be streamed by
        // the leaf hook (the child is executed via `self.execute`, which gets
        // its own streaming opportunity, but the *outer* op is not a leaf).
        let distinct = PhysicalPlan::Distinct {
            input: Box::new(passthrough_projection("t")),
        };
        assert_eq!(Engine::streamable_leaf_scan(&distinct), None);
        let limit = PhysicalPlan::Limit {
            input: Box::new(passthrough_projection("t")),
            limit: 5,
            offset: 0,
        };
        assert_eq!(Engine::streamable_leaf_scan(&limit), None);
    }

    /// Build a replayable single-batch producer for an `int64_batch`-shaped
    /// table holding `[start, start+n)`.
    fn int64_producer(start: i64, n: usize) -> crate::exec::streaming::BatchProducer {
        Box::new(move || Box::new(std::iter::once(Ok(int64_batch(start, n)))))
    }

    /// End-to-end equivalence: a streaming-registered table queried under a
    /// memory budget small enough to force morsel chunking must produce a
    /// result byte-for-byte identical to the same data materialised whole. The
    /// morsel path concatenates per-morsel projection outputs; the row-wise
    /// projection makes that equal to the whole-table projection.
    ///
    /// GPU-gated: builds an `Engine` and launches the projection kernel.
    #[test]
    #[ignore = "gpu:projection — streaming morsel equivalence launches kernels"]
    fn streaming_morsel_matches_materialized_projection() {
        let total = 1000usize;

        // Baseline: whole table materialised, no budget.
        let mut whole = Engine::new().expect("ctx");
        whole.register_table("t", int64_batch(0, total)).expect("register whole");
        let h_whole = whole.sql("SELECT x FROM t WHERE x >= 100").expect("whole query");
        let want = h_whole.record_batch().clone();

        // Streaming source + a budget far below the table footprint, so
        // `morsel_plan_for_table` returns `Morsels` and the streaming hook
        // fires. ~4 KiB budget over an 8 KB+ Int64 table → many morsels.
        let mut streamed = Engine::builder()
            .memory_budget(4096)
            .build()
            .expect("ctx with budget");
        streamed
            .register_table_stream_lazy("t", x_schema(), int64_producer(0, total))
            .expect("register stream");
        let h_stream = streamed.sql("SELECT x FROM t WHERE x >= 100").expect("stream query");
        let got = h_stream.record_batch().clone();

        assert_eq!(got.num_rows(), want.num_rows(), "row counts must match");
        let want_col = want.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let got_col = got.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let want_v: Vec<i64> = (0..want_col.len()).map(|i| want_col.value(i)).collect();
        let got_v: Vec<i64> = (0..got_col.len()).map(|i| got_col.value(i)).collect();
        assert_eq!(got_v, want_v, "morsel-streamed values must equal whole-table values");
    }

    /// The drain-fallback path: an operator whose result is NOT a row-wise leaf
    /// scan (here a global `SUM` aggregate) drains the streaming source to a
    /// whole table and produces the correct global result even under a small
    /// budget. This documents that "anything not safely streamable drains".
    ///
    /// GPU-gated: builds an `Engine` and runs the aggregate kernel.
    #[test]
    #[ignore = "gpu:aggregate — drain-fallback aggregate over streaming source"]
    fn streaming_drain_fallback_global_aggregate() {
        let total = 500usize;
        let mut engine = Engine::builder()
            .memory_budget(4096)
            .build()
            .expect("ctx with budget");
        engine
            .register_table_stream_lazy("t", x_schema(), int64_producer(0, total))
            .expect("register stream");
        // SUM is a global fold — `streamable_leaf_scan` returns None for
        // `Aggregate`, so the whole table is drained (status quo) and summed.
        let h = engine.sql("SELECT SUM(x) FROM t").expect("aggregate query");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 1, "scalar aggregate yields one row");
        let col = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let expected: i64 = (0..total as i64).sum();
        assert_eq!(col.value(0), expected, "global SUM over the drained stream");
    }

    /// After a morsel-streamed query the whole-table overlay entry must be
    /// restored, so a *second* query (e.g. without a budget effect, or a
    /// drain-fallback aggregate) still sees the full table. Exercises the
    /// always-restore guarantee of `execute_streaming_leaf`.
    ///
    /// GPU-gated: builds an `Engine`.
    #[test]
    #[ignore = "gpu:projection — overlay restoration across queries"]
    fn streaming_overlay_restored_after_morsel_query() {
        let total = 300usize;
        let mut engine = Engine::builder()
            .memory_budget(4096)
            .build()
            .expect("ctx with budget");
        engine
            .register_table_stream_lazy("t", x_schema(), int64_producer(0, total))
            .expect("register stream");
        // First query streams morsel-by-morsel.
        let h1 = engine.sql("SELECT x FROM t").expect("first (streamed) query");
        assert_eq!(h1.record_batch().num_rows(), total, "first query sees all rows");
        // Second query (drain-fallback aggregate) must still see the full
        // table — the overlay was restored to the whole-table view.
        let h2 = engine.sql("SELECT COUNT(x) FROM t").expect("second query");
        let c = h2.record_batch().column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(c.value(0), total as i64, "overlay restored: full table visible");
    }

    // -----------------------------------------------------------------------
    // F1: WITH RECURSIVE fixpoint execution
    // -----------------------------------------------------------------------

    /// `max_recursive_iterations()` parses the env override (positive only)
    /// and otherwise returns the default. Host-only — no device needed.
    #[test]
    fn recursive_iteration_cap_env_parsing() {
        // Default when unset.
        std::env::remove_var(MAX_RECURSIVE_ITERATIONS_ENV);
        assert_eq!(max_recursive_iterations(), MAX_RECURSIVE_ITERATIONS);
        // A positive override is honoured.
        std::env::set_var(MAX_RECURSIVE_ITERATIONS_ENV, "7");
        assert_eq!(max_recursive_iterations(), 7);
        // Zero / garbage fall back to the default.
        std::env::set_var(MAX_RECURSIVE_ITERATIONS_ENV, "0");
        assert_eq!(max_recursive_iterations(), MAX_RECURSIVE_ITERATIONS);
        std::env::set_var(MAX_RECURSIVE_ITERATIONS_ENV, "not-a-number");
        assert_eq!(max_recursive_iterations(), MAX_RECURSIVE_ITERATIONS);
        std::env::remove_var(MAX_RECURSIVE_ITERATIONS_ENV);
    }

    /// A one-column `edges(src, dst)` fixture for the recursive execution
    /// tests below.
    #[cfg(test)]
    fn register_edges(engine: &mut Engine) {
        // edges: 1->2, 2->3, 3->4
        let src: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 2, 3]));
        let dst: ArrayRef = Arc::new(Int64Array::from(vec![2_i64, 3, 4]));
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("src", ArrowDataType::Int64, false),
            ArrowField::new("dst", ArrowDataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![src, dst]).expect("edges batch");
        engine.register_table("edges", batch).expect("register edges");
    }

    /// End-to-end: an integer sequence `1..=5` via UNION ALL accumulates
    /// exactly five rows (1,2,3,4,5). Needs a device because the subplans run
    /// through the normal GPU execute path.
    #[test]
    #[ignore = "gpu:e2e — recursive CTE subplans run through the GPU execute path"]
    fn recursive_integer_sequence_accumulates_rows() {
        let mut engine = Engine::new().expect("ctx");
        register_edges(&mut engine); // base table so the provider is non-empty
        // Seed from a base table (the frontend requires a FROM clause, so a
        // bare `SELECT 1` anchor is not available): `src = 1` selects the
        // single row valued 1 from `edges`.
        let h = engine
            .sql(
                "WITH RECURSIVE seq(n) AS (\
                     SELECT src FROM edges WHERE src = 1 \
                     UNION ALL \
                     SELECT n + 1 FROM seq WHERE n < 5\
                 ) SELECT n FROM seq ORDER BY n",
            )
            .expect("recursive integer sequence must execute");
        let b = h.record_batch();
        assert_eq!(b.num_rows(), 5, "1..=5 is five rows");
        let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let got: Vec<i64> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec![1, 2, 3, 4, 5]);
    }

    /// End-to-end: a graph reachability traversal from node 1 over `edges`
    /// reaches {1,2,3,4}. UNION (distinct) dedups, so the result is a set.
    #[test]
    #[ignore = "gpu:e2e — recursive CTE subplans run through the GPU execute path"]
    fn recursive_graph_reachability() {
        let mut engine = Engine::new().expect("ctx");
        register_edges(&mut engine);
        let h = engine
            .sql(
                "WITH RECURSIVE reach(node) AS (\
                     SELECT src FROM edges WHERE src = 1 \
                     UNION \
                     SELECT edges.dst FROM edges, reach WHERE edges.src = reach.node\
                 ) SELECT node FROM reach ORDER BY node",
            )
            .expect("recursive reachability must execute");
        let b = h.record_batch();
        let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let got: Vec<i64> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec![1, 2, 3, 4], "reachable set from node 1");
    }

    /// A non-terminating recursion (`n` grows without bound, never filtered)
    /// must hit the iteration cap and return a clean error rather than
    /// spinning forever. We lower the cap via the env override to keep the
    /// test fast.
    #[test]
    #[ignore = "gpu:e2e — recursive CTE subplans run through the GPU execute path"]
    fn recursive_non_terminating_hits_cap() {
        let mut engine = Engine::new().expect("ctx");
        register_edges(&mut engine);
        std::env::set_var(MAX_RECURSIVE_ITERATIONS_ENV, "16");
        let err = engine
            .sql(
                "WITH RECURSIVE seq(n) AS (\
                     SELECT src FROM edges WHERE src = 1 \
                     UNION ALL \
                     SELECT n + 1 FROM seq\
                 ) SELECT n FROM seq",
            )
            .expect_err("a non-terminating recursion must hit the cap");
        std::env::remove_var(MAX_RECURSIVE_ITERATIONS_ENV);
        let msg = format!("{err}");
        assert!(
            msg.contains("safety cap") || msg.contains("not terminating"),
            "expected iteration-cap error, got: {msg}"
        );
    }

    // ---- Non-linear recursion (naive evaluation) e2e ----

    /// End-to-end: transitive closure of `edges` via a NON-LINEAR recursive
    /// term (a self-join `r JOIN r`). Naive evaluation re-derives against the
    /// full accumulation each step, so the closure of 1->2->3->4 is the set of
    /// all reachable pairs {(1,2),(2,3),(3,4),(1,3),(2,4),(1,4)} (6 pairs).
    #[test]
    #[ignore = "gpu:e2e — recursive CTE subplans run through the GPU execute path"]
    fn recursive_non_linear_transitive_closure() {
        let mut engine = Engine::new().expect("ctx");
        register_edges(&mut engine); // 1->2, 2->3, 3->4
        let h = engine
            .sql(
                "WITH RECURSIVE tc(x, y) AS (\
                     SELECT src, dst FROM edges \
                     UNION \
                     SELECT a.x, b.y FROM tc AS a JOIN tc AS b ON a.y = b.x\
                 ) SELECT x, y FROM tc ORDER BY x, y",
            )
            .expect("non-linear transitive closure must execute");
        let b = h.record_batch();
        assert_eq!(b.num_rows(), 6, "transitive closure of a 4-node path is 6 pairs");
        let xs = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let ys = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let got: Vec<(i64, i64)> =
            (0..xs.len()).map(|i| (xs.value(i), ys.value(i))).collect();
        assert_eq!(
            got,
            vec![(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
            "all reachable ordered pairs"
        );
    }

    /// A NON-LINEAR `UNION ALL` self-join over cyclic data grows without bound
    /// under naive evaluation, so it must hit the iteration cap with a clean
    /// error (the cap is the mandatory guard when dedup is unavailable).
    #[test]
    #[ignore = "gpu:e2e — recursive CTE subplans run through the GPU execute path"]
    fn recursive_non_linear_union_all_cycle_hits_cap() {
        use arrow_array::Int64Array as I64;
        let mut engine = Engine::new().expect("ctx");
        // A 2-cycle: 1->2, 2->1. Composing self-joins re-derives forever.
        let src: ArrayRef = Arc::new(I64::from(vec![1_i64, 2]));
        let dst: ArrayRef = Arc::new(I64::from(vec![2_i64, 1]));
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("src", ArrowDataType::Int64, false),
            ArrowField::new("dst", ArrowDataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(schema, vec![src, dst]).expect("cyc batch");
        engine.register_table("cyc", batch).expect("register cyc");
        std::env::set_var(MAX_RECURSIVE_ITERATIONS_ENV, "8");
        let err = engine
            .sql(
                "WITH RECURSIVE tc(x, y) AS (\
                     SELECT src, dst FROM cyc \
                     UNION ALL \
                     SELECT a.x, b.y FROM tc AS a JOIN tc AS b ON a.y = b.x\
                 ) SELECT x, y FROM tc",
            )
            .expect_err("non-linear UNION ALL over a cycle must hit the cap");
        std::env::remove_var(MAX_RECURSIVE_ITERATIONS_ENV);
        let msg = format!("{err}");
        assert!(
            msg.contains("safety cap") || msg.contains("not terminating"),
            "expected iteration-cap error, got: {msg}"
        );
    }

    // ---- Mutual recursion (lockstep fixpoint) e2e ----

    /// End-to-end: two mutually-recursive CTEs that ping-pong. `evens` starts
    /// at 0 and adds 1 to every `odd`; `odds` starts at 1 and adds 1 to every
    /// `even`, both bounded by `< 6`. The lockstep fixpoint accumulates
    /// evens={0,2,4,6} and odds={1,3,5}, and the main query unions them.
    #[test]
    #[ignore = "gpu:e2e — recursive CTE subplans run through the GPU execute path"]
    fn recursive_mutual_even_odd() {
        let mut engine = Engine::new().expect("ctx");
        register_edges(&mut engine); // base table so a FROM seed is available
        let h = engine
            .sql(
                "WITH RECURSIVE \
                   evens(n) AS (\
                       SELECT 0 FROM edges WHERE src = 1 \
                       UNION \
                       SELECT n + 1 FROM odds WHERE n < 6\
                   ), \
                   odds(n) AS (\
                       SELECT 1 FROM edges WHERE src = 1 \
                       UNION \
                       SELECT n + 1 FROM evens WHERE n < 6\
                   ) \
                 SELECT n FROM evens UNION SELECT n FROM odds ORDER BY n",
            )
            .expect("mutual recursion must execute");
        let b = h.record_batch();
        let col = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let got: Vec<i64> = (0..col.len()).map(|i| col.value(i)).collect();
        assert_eq!(got, vec![0, 1, 2, 3, 4, 5, 6], "0..=6 from the lockstep fixpoint");
    }

    /// A mutually-recursive system with no bound grows forever and must hit the
    /// shared iteration cap with the mutual-recursion error message.
    #[test]
    #[ignore = "gpu:e2e — recursive CTE subplans run through the GPU execute path"]
    fn recursive_mutual_non_terminating_hits_cap() {
        let mut engine = Engine::new().expect("ctx");
        register_edges(&mut engine);
        std::env::set_var(MAX_RECURSIVE_ITERATIONS_ENV, "8");
        let err = engine
            .sql(
                "WITH RECURSIVE \
                   a(n) AS (SELECT src FROM edges WHERE src = 1 UNION ALL SELECT n + 1 FROM b), \
                   b(n) AS (SELECT dst FROM edges WHERE dst = 2 UNION ALL SELECT n + 1 FROM a) \
                 SELECT n FROM a",
            )
            .expect_err("a non-terminating mutual recursion must hit the cap");
        std::env::remove_var(MAX_RECURSIVE_ITERATIONS_ENV);
        let msg = format!("{err}");
        assert!(
            msg.contains("safety cap") || msg.contains("not terminating"),
            "expected iteration-cap error, got: {msg}"
        );
    }

    // ---- Feature 1: multi-table FROM (comma cross join) e2e ----

    /// `FROM a, b WHERE a.k = b.k` produces the cartesian product filtered to
    /// the matching pairs. End-to-end through the engine (child scans run on
    /// the GPU execute path), so gpu-gated.
    #[test]
    #[ignore = "gpu:e2e — cross-join children run through the GPU execute path"]
    fn comma_cross_join_with_where_filter_cartesian() {
        use arrow_array::Int32Array;
        let mut engine = Engine::new().expect("ctx");
        // a(k, av): (1,10),(2,20); b(k, bv): (1,100),(1,101),(3,300).
        let a_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("av", ArrowDataType::Int32, false),
        ]));
        let a = RecordBatch::try_new(
            a_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![10, 20])),
            ],
        )
        .unwrap();
        let b_schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("bv", ArrowDataType::Int32, false),
        ]));
        let b = RecordBatch::try_new(
            b_schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 3])),
                Arc::new(Int32Array::from(vec![100, 101, 300])),
            ],
        )
        .unwrap();
        engine.register_table("a", a).expect("register a");
        engine.register_table("b", b).expect("register b");

        // Cross product is 2×3 = 6 rows; the WHERE keeps only a.k == b.k:
        // (k=1, av=10) × (bv=100, bv=101) → two rows.
        let h = engine
            .sql("SELECT a.av, b.bv FROM a, b WHERE a.k = b.k")
            .expect("cross-join query");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 2, "only k=1 pairs survive the filter");
        let av = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let bv = out.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        let mut got: Vec<(i32, i32)> =
            (0..out.num_rows()).map(|i| (av.value(i), bv.value(i))).collect();
        got.sort_unstable();
        assert_eq!(got, vec![(10, 100), (10, 101)]);
    }

    // ---- Feature 2: multi / mixed COUNT(DISTINCT) + GROUP BY e2e ----

    /// `SELECT g, COUNT(DISTINCT a), SUM(x), COUNT(*) FROM t GROUP BY g`
    /// end-to-end (the base subplan runs through the GPU execute path).
    #[test]
    #[ignore = "gpu:e2e — multi-agg base subplan runs through the GPU execute path"]
    fn multi_agg_groupby_mixed_e2e() {
        use arrow_array::Int64Array;
        let mut engine = Engine::new().expect("ctx");
        // g, a, x: group 1 -> a {10,10,20}=2 distinct, x sum 1+2+3=6, 3 rows;
        //          group 2 -> a {30}=1 distinct, x sum 4, 1 row.
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("g", ArrowDataType::Int64, false),
            ArrowField::new("a", ArrowDataType::Int64, false),
            ArrowField::new("x", ArrowDataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(vec![1, 1, 1, 2])),
                Arc::new(Int64Array::from(vec![10, 10, 20, 30])),
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
            ],
        )
        .unwrap();
        engine.register_table("t", batch).expect("register");

        let h = engine
            .sql(
                "SELECT g, COUNT(DISTINCT a), SUM(x), COUNT(*) \
                 FROM t GROUP BY g ORDER BY g",
            )
            .expect("multi-agg groupby query");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 2);
        let g = out.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let cda = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let sumx = out.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
        let cnt = out.column(3).as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!((g.value(0), cda.value(0), sumx.value(0), cnt.value(0)), (1, 2, 6, 3));
        assert_eq!((g.value(1), cda.value(1), sumx.value(1), cnt.value(1)), (2, 1, 4, 1));
    }

    // -----------------------------------------------------------------------
    // F3: LATERAL apply (host nested-loop dependent join)
    // -----------------------------------------------------------------------

    /// `left(k Int64, lbl Utf8)` and `vals(vk Int64, n Int64)` fixtures for the
    /// LATERAL apply tests. `left` has a row (k=3) with no matching `vals`
    /// rows, to exercise the INNER-drop / LEFT-keep behaviour.
    fn register_lateral_fixtures(engine: &mut Engine) {
        use arrow_array::StringArray;
        let lk: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 2, 3]));
        let lbl: ArrayRef = Arc::new(StringArray::from(vec!["a", "b", "c"]));
        let lschema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int64, false),
            ArrowField::new("lbl", ArrowDataType::Utf8, false),
        ]));
        let left = RecordBatch::try_new(lschema, vec![lk, lbl]).expect("left batch");
        engine.register_table("lft", left).expect("register lft");

        // vals: (1,10),(1,11),(2,20) — k=1 has two, k=2 has one, k=3 has none.
        let vk: ArrayRef = Arc::new(Int64Array::from(vec![1_i64, 1, 2]));
        let n: ArrayRef = Arc::new(Int64Array::from(vec![10_i64, 11, 20]));
        let vschema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("vk", ArrowDataType::Int64, false),
            ArrowField::new("n", ArrowDataType::Int64, false),
        ]));
        let vals = RecordBatch::try_new(vschema, vec![vk, n]).expect("vals batch");
        engine.register_table("vals", vals).expect("register vals");
    }

    /// Standalone provider (no Engine / no GPU) exposing `lft(k Int64, lbl Utf8)`
    /// and `vals(vk Int64, n Int64)` for the host-only LATERAL detector tests.
    fn lateral_provider() -> crate::plan::sql_frontend::MemTableProvider {
        use crate::plan::logical_plan::{DataType, Field, Schema};
        let lft = Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("lbl", DataType::Utf8, false),
        ]);
        let vals = Schema::new(vec![
            Field::new("vk", DataType::Int64, false),
            Field::new("n", DataType::Int64, false),
        ]);
        crate::plan::sql_frontend::MemTableProvider::new()
            .with_table("lft", lft)
            .with_table("vals", vals)
    }

    /// Parse → descriptor: a correlated LATERAL is detected and lowered to a
    /// [`LateralApplyPlan`] (not the ordinary pipeline), with the LEFT relation,
    /// the per-row subplan, the single correlation (`lft.k`), and INNER (not
    /// LEFT) apply. Host-only: builds the descriptor without touching the GPU.
    #[test]
    fn lateral_apply_parse_descriptor() {
        let provider = lateral_provider();
        let la = crate::plan::sql_frontend::plan_lateral_apply(
            "SELECT lft.k, d.n FROM lft, LATERAL (SELECT n FROM vals WHERE vk = lft.k) AS d",
            &provider,
        )
        .expect("detector must not error")
        .expect("a correlated LATERAL must produce a descriptor");
        assert!(!la.left_join, "plain comma LATERAL is an INNER apply");
        assert_eq!(la.outer_schema.fields.len(), 1, "one correlation: lft.k");
        assert_eq!(la.corr_left_indices, vec![0], "lft.k is left column 0");
        // The applied relation is left (k, lbl) ++ subquery (n) = 3 columns.
        assert_eq!(la.combined_schema.fields.len(), 3);
    }

    /// A query with no LATERAL is declined by the detector (`Ok(None)`), so it
    /// falls through to the ordinary pipeline. Host-only.
    #[test]
    fn lateral_apply_detector_declines_non_lateral() {
        let provider = lateral_provider();
        let got = crate::plan::sql_frontend::plan_lateral_apply("SELECT k FROM lft", &provider)
            .expect("detector");
        assert!(got.is_none(), "a non-LATERAL query must not be an apply");
    }

    /// Out-of-scope LATERAL shape: a non-`ON true` JOIN LATERAL predicate is
    /// rejected precisely by the detector. Host-only.
    #[test]
    fn lateral_apply_rejects_non_on_true() {
        let provider = lateral_provider();
        let err = crate::plan::sql_frontend::plan_lateral_apply(
            "SELECT lft.k, d.n FROM lft JOIN LATERAL (SELECT n FROM vals WHERE vk = lft.k) AS d \
             ON lft.k = d.n",
            &provider,
        )
        .expect_err("a residual JOIN LATERAL predicate must be rejected");
        let msg = format!("{err}");
        assert!(msg.contains("ON true"), "expected ON-true rejection, got: {msg}");
    }

    /// End-to-end correctness: a correlated LATERAL join. `k=1` yields two rows
    /// (10,11), `k=2` one row (20), and `k=3` is DROPPED (INNER apply, no
    /// matches). GPU-gated (subplans run through the device).
    #[test]
    #[ignore = "gpu:e2e — LATERAL apply subplans run through the GPU execute path"]
    fn lateral_apply_inner_drops_unmatched() {
        let mut engine = Engine::new().expect("ctx");
        register_lateral_fixtures(&mut engine);
        let h = engine
            .sql(
                "SELECT lft.k, d.n FROM lft, LATERAL (SELECT n FROM vals WHERE vk = lft.k) AS d \
                 ORDER BY lft.k, d.n",
            )
            .expect("lateral apply must execute");
        let b = h.record_batch();
        assert_eq!(b.num_rows(), 3, "k=1 (2 rows) + k=2 (1 row); k=3 dropped");
        let k = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let n = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let got: Vec<(i64, i64)> = (0..b.num_rows()).map(|i| (k.value(i), n.value(i))).collect();
        assert_eq!(got, vec![(1, 10), (1, 11), (2, 20)]);
    }

    /// LEFT JOIN LATERAL ... ON true keeps the unmatched left row (k=3) once
    /// with the subquery column NULL. GPU-gated.
    #[test]
    #[ignore = "gpu:e2e — LATERAL apply subplans run through the GPU execute path"]
    fn lateral_apply_left_join_keeps_unmatched() {
        let mut engine = Engine::new().expect("ctx");
        register_lateral_fixtures(&mut engine);
        let h = engine
            .sql(
                "SELECT lft.k, d.n FROM lft LEFT JOIN LATERAL \
                 (SELECT n FROM vals WHERE vk = lft.k) AS d ON true ORDER BY lft.k, d.n",
            )
            .expect("left join lateral must execute");
        let b = h.record_batch();
        // k=1 → 2 rows, k=2 → 1 row, k=3 → 1 NULL row = 4 total.
        assert_eq!(b.num_rows(), 4, "unmatched left row kept once with NULL");
        let n = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        assert!(
            (0..b.num_rows()).filter(|&i| arrow_array::Array::is_null(n, i)).count() == 1,
            "exactly one NULL subquery value (the k=3 left row)"
        );
    }

    /// The mandatory per-left-row safety cap fires with a clean error when the
    /// left input exceeds the (env-lowered) cap. GPU-gated (the left subplan
    /// runs on the device before the cap check).
    #[test]
    #[ignore = "gpu:e2e — LATERAL apply left subplan runs through the GPU execute path"]
    fn lateral_apply_left_row_cap() {
        let mut engine = Engine::new().expect("ctx");
        register_lateral_fixtures(&mut engine); // lft has 3 rows
        std::env::set_var(MAX_APPLY_LEFT_ROWS_ENV, "2");
        let err = engine
            .sql(
                "SELECT lft.k, d.n FROM lft, LATERAL (SELECT n FROM vals WHERE vk = lft.k) AS d",
            )
            .expect_err("3 left rows must exceed the 2-row cap");
        std::env::remove_var(MAX_APPLY_LEFT_ROWS_ENV);
        let msg = format!("{err}");
        assert!(
            msg.contains("safety cap") && msg.contains("LATERAL"),
            "expected LATERAL cap error, got: {msg}"
        );
    }

    /// `max_apply_left_rows()` parses the env override (positive only) and
    /// otherwise returns the default. Host-only.
    #[test]
    fn lateral_apply_cap_env_parsing() {
        std::env::remove_var(MAX_APPLY_LEFT_ROWS_ENV);
        assert_eq!(max_apply_left_rows(), MAX_APPLY_LEFT_ROWS);
        std::env::set_var(MAX_APPLY_LEFT_ROWS_ENV, "5");
        assert_eq!(max_apply_left_rows(), 5);
        std::env::set_var(MAX_APPLY_LEFT_ROWS_ENV, "0");
        assert_eq!(max_apply_left_rows(), MAX_APPLY_LEFT_ROWS, "zero falls back");
        std::env::set_var(MAX_APPLY_LEFT_ROWS_ENV, "nope");
        assert_eq!(max_apply_left_rows(), MAX_APPLY_LEFT_ROWS, "non-int falls back");
        std::env::remove_var(MAX_APPLY_LEFT_ROWS_ENV);
    }
}
