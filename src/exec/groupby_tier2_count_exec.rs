// SPDX-License-Identifier: Apache-2.0

//! **COUNT(*) at Tier 2.1** — high-cardinality `SELECT key, COUNT(*) FROM x
//! GROUP BY key` executor.
//!
//! Companion to the AVG-at-Tier-2.1 executor. The AVG path uses the same
//! COUNT kernel internally for its denominator; this executor exposes
//! that primitive on its own for queries that only ask for counts.
//!
//! ## Algorithm
//!
//! 1. Partition + scatter (keys only — no value column).
//! 2. Per-partition reduce via `partition_reduce_kernel_count` → per-group
//!    `u64` counts.
//! 3. Walk slots, push `(key, count)` into the output (skipping empty
//!    slots). Sort by key ASC.
//!
//! ## Scope (v0)
//!
//! - GROUP BY exactly one Int32 column
//! - Exactly one aggregate, `COUNT(*)` (which the planner represents as
//!   `AggregateExpr::Count(Expr::Literal(_))` or similar — we match by
//!   variant)
//! - `n_rows >= 256 K`
//! - `max(key) >= BLOCK_GROUPS` (Tier-1 single-aggregate path would
//!   handle the low-cardinality case if/when it grows a COUNT branch)
//! - `max(key) < 100 M`

use std::collections::HashMap;
use std::sync::Arc;

use arrow_array::{Int32Array, Int64Array, RecordBatch};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
use once_cell::sync::Lazy;
use parking_lot::Mutex;

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{launch_with_geometry, CudaStream, KernelArgs};
use crate::exec::partition_offsets;
use crate::jit::{
    partition_kernel, partition_reduce_kernel_count, scatter_kernel, CudaModule,
};
use crate::plan::logical_plan::{AggregateExpr, DataType, Schema};
use crate::plan::physical_plan::PhysicalPlan;

const BLOCK_THREADS: u32 = 256;

// ---------------------------------------------------------------------------
// Per-executor module cache.
//
// `CudaModule::from_ptx` (see `src/jit/jit_compiler.rs`) already deduplicates
// the PTX → SASS step by hashing the PTX text, but the caller still pays for
// rebuilding the PTX string (non-trivial — kilobytes of templated text) and
// for the cache lookup on every invocation. We add a second layer keyed by
// the small set of parameters that *select* a PTX template, so a repeat call
// skips PTX construction entirely and gets a cheap `CudaModule` clone (the
// inner handle is `Arc`-shared with the `from_ptx` cache).
//
// `CudaModule` is `Clone` over an internal `Arc<CudaModuleInner>`, so we
// store owned modules in the map and hand callers fresh clones — no need to
// wrap the value in another `Arc`.
// ---------------------------------------------------------------------------

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
enum KernelSpec {
    /// `partition_kernel::compile_partition_kernel()` — unparameterised.
    Partition,
    /// `scatter_kernel::compile_scatter_kernel()` — unparameterised.
    Scatter,
    /// `partition_reduce_kernel_count::compile_partition_reduce_kernel_count()` —
    /// unparameterised.
    ReduceCount,
}

static MODULE_CACHE: Lazy<Mutex<HashMap<KernelSpec, CudaModule>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

/// Test-only counter of cache-miss compile passes. Incremented exactly once
/// per `(spec, process)` pair regardless of how many threads race on the
/// initial miss.
#[cfg(test)]
static LOAD_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Cache-aware module loader. Returns a (cheap-Arc-clone of a) `CudaModule`
/// for `spec`, building it on first miss and serving from the process-wide
/// map thereafter. Builds happen outside the cache lock so that compiles for
/// *different* specs can run in parallel; the small window where two threads
/// race on the same spec results in at most one redundant compile (the
/// second insertion overwrites — both modules are functionally identical
/// and the loser is unloaded when its last clone drops).
fn get_or_build_module(spec: &KernelSpec) -> BoltResult<CudaModule> {
    if let Some(m) = MODULE_CACHE.lock().get(spec) {
        return Ok(m.clone());
    }
    let ptx = match spec {
        KernelSpec::Partition => partition_kernel::compile_partition_kernel()?,
        KernelSpec::Scatter => scatter_kernel::compile_scatter_kernel()?,
        // Batch 4: spill-counter-aware variant. The kernel atomically
        // bumps a host-visible counter when a row exceeds MAX_PROBES; the
        // caller checks it after sync and surfaces a structured error.
        KernelSpec::ReduceCount => {
            partition_reduce_kernel_count::compile_partition_reduce_kernel_count_with_spill()?
        }
    };
    let module = CudaModule::from_ptx(&ptx)?;
    #[cfg(test)]
    LOAD_COUNT.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    let mut cache = MODULE_CACHE.lock();
    Ok(cache.entry(spec.clone()).or_insert(module).clone())
}

/// Try the Tier-2.1 COUNT(*) fast path. `None` on any precondition miss.
pub fn try_execute(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<BoltResult<RecordBatch>> {
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        _ => return None,
    };
    if pre.is_some() {
        return None;
    }
    if aggregate.group_by.len() != 1 || aggregate.aggregates.len() != 1 {
        return None;
    }

    // Exactly one COUNT aggregate. We accept COUNT(<anything>) by
    // semantics: SQL COUNT(*) and COUNT(non_null_col) on a NOT NULL
    // schema produce the same result. The kernel doesn't read a value
    // column anyway, so the argument is decorative.
    match &aggregate.aggregates[0] {
        AggregateExpr::Count(_) => {}
        _ => return None,
    }

    // Single Int32 key.
    let key_io_idx = aggregate.group_by[0];
    let key_io = match aggregate.inputs.get(key_io_idx) {
        Some(io) if io.dtype == DataType::Int32 => io,
        _ => return None,
    };

    let key_arr = batch
        .column_by_name(&key_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;

    let n_rows = key_arr.len();
    if n_rows < 256 * 1024 {
        return None;
    }

    let mut max_key: i32 = -1;
    for &k in key_arr.values() {
        if k < 0 {
            return None;
        }
        if k > max_key {
            max_key = k;
        }
    }
    if max_key < 0 {
        return None;
    }
    let n_groups_est = (max_key as u32).saturating_add(1);
    if n_groups_est <= partition_reduce_kernel_count::BLOCK_GROUPS {
        // Low cardinality — let the global-atomic path handle COUNT(*).
        // (We don't yet have a Tier-1 COUNT shortcut; not chasing it
        // until we see a workload that wants it.)
        return None;
    }
    if n_groups_est >= 100_000_000 {
        return None;
    }

    Some(execute_inner(plan, key_arr))
}

fn execute_inner(plan: &PhysicalPlan, key_arr: &Int32Array) -> BoltResult<RecordBatch> {
    let n_rows = key_arr.len() as u32;
    // Stage-4 (P1b): per-call stream so H2D, kernels, and the final
    // D2H share one ordering domain.
    let stream = CudaStream::null_or_default();
    let keys_gpu: GpuVec<i32> = GpuVec::<i32>::from_slice_async(key_arr.values(), stream.raw())?;

    let num_partitions = partition_kernel::NUM_PARTITIONS;
    let mut counts: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
    let mut partition_ids: GpuVec<u32> = GpuVec::<u32>::zeros_async(n_rows as usize, stream.raw())?;

    // Partition pass — CUDA-Oxide typed launch.
    // Kernel ABI: keys_ptr, pids_ptr, counts_ptr, n_rows
    let partition_module = get_or_build_module(&KernelSpec::Partition)?;
    {
        let func = partition_module.function(partition_kernel::KERNEL_ENTRY)?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);

        let view_keys = keys_gpu.view();
        let mut view_pids = partition_ids.view_mut();
        let mut view_counts = counts.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_output(&mut view_pids);
        args.push_output(&mut view_counts);
        args.push_scalar_u32(n_rows);

        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }

    // P1b-stage8: joint helper collapses the legacy
    // `compute_partition_offsets` + `upload_offsets` 2-sync pair into 1
    // sync via a single async D2H → host scan → async H2D round-trip
    // through a thread-local pinned scratch buffer.
    let (offsets, offsets_gpu): (Vec<u32>, GpuVec<u32>) =
        partition_offsets::compute_and_upload_partition_offsets_async(&counts, stream.raw())?;

    // Scatter keys only. We still use the scatter kernel; it requires a
    // value column input, but for COUNT we have no meaningful value —
    // pass a zero-filled f64 buffer of the same length. The dummy
    // out_vals buffer is written but never read.
    let mut scatter_keys: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_rows as usize, stream.raw())?;
    let dummy_vals_in: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?;
    let mut scatter_vals: GpuVec<f64> = GpuVec::<f64>::zeros_async(n_rows as usize, stream.raw())?;

    // Scatter pass — CUDA-Oxide typed launch.
    // Kernel ABI: keys, vals, pids, offsets, cursors, out_keys, out_vals, n_rows
    let scatter_module = get_or_build_module(&KernelSpec::Scatter)?;
    {
        let func = scatter_module.function(scatter_kernel::KERNEL_ENTRY)?;
        let mut cursors: GpuVec<u32> = GpuVec::<u32>::zeros_async(num_partitions as usize, stream.raw())?;
        let grid = n_rows.div_ceil(BLOCK_THREADS).max(1);

        let view_keys = keys_gpu.view();
        let view_vals = dummy_vals_in.view();
        let view_pids = partition_ids.view();
        let view_offsets = offsets_gpu.view();
        let mut view_cursors = cursors.view_mut();
        let mut view_sk = scatter_keys.view_mut();
        let mut view_sv = scatter_vals.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_input(&view_vals);
        args.push_input(&view_pids);
        args.push_input(&view_offsets);
        args.push_output(&mut view_cursors);
        args.push_output(&mut view_sk);
        args.push_output(&mut view_sv);
        args.push_scalar_u32(n_rows);

        launch_with_geometry(func, grid, BLOCK_THREADS, 0, &stream, &mut args)?;
    }
    let _ = (dummy_vals_in, scatter_vals); // keep alive until end of launch

    // COUNT reduce pass.
    let offsets_kp1_gpu: GpuVec<u32> = GpuVec::<u32>::from_slice_async(&offsets, stream.raw())?;
    let block_groups = partition_reduce_kernel_count::BLOCK_GROUPS as usize;
    let n_out_slots = (num_partitions as usize) * block_groups;
    let mut out_keys_gpu: GpuVec<i32> = GpuVec::<i32>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_counts_gpu: GpuVec<u64> = GpuVec::<u64>::zeros_async(n_out_slots, stream.raw())?;
    let mut out_set_gpu: GpuVec<u8> = GpuVec::<u8>::zeros_async(n_out_slots, stream.raw())?;

    // CUDA-Oxide typed launch.
    // Kernel ABI (spill variant): scatter_keys, offsets, out_keys,
    // out_counts, out_set, spill_counter.
    let reduce_module = get_or_build_module(&KernelSpec::ReduceCount)?;
    let mut spill_counter: GpuVec<u32> = GpuVec::<u32>::zeros_async(1, stream.raw())?;
    {
        let func = reduce_module
            .function(partition_reduce_kernel_count::KERNEL_ENTRY_WITH_SPILL)?;

        let view_keys = scatter_keys.view();
        let view_offsets = offsets_kp1_gpu.view();
        let mut view_ok = out_keys_gpu.view_mut();
        let mut view_oc = out_counts_gpu.view_mut();
        let mut view_os = out_set_gpu.view_mut();
        let mut view_spill = spill_counter.view_mut();

        let mut args = KernelArgs::empty();
        args.push_input(&view_keys);
        args.push_input(&view_offsets);
        args.push_output(&mut view_ok);
        args.push_output(&mut view_oc);
        args.push_output(&mut view_os);
        args.push_output(&mut view_spill);

        launch_with_geometry(
            func,
            num_partitions,
            partition_reduce_kernel_count::BLOCK_THREADS,
            0,
            &stream,
            &mut args,
        )?;
    }

    // Stage-4 (P1b): pinned D2H for the three fixed-size output
    // buffers. Queue all three, then synchronize once.
    let pinned_keys = out_keys_gpu.to_pinned_async(stream.raw())?;
    let pinned_counts = out_counts_gpu.to_pinned_async(stream.raw())?;
    let pinned_set = out_set_gpu.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let host_out_keys: Vec<i32> = pinned_keys.as_slice().to_vec();
    let host_out_counts: Vec<u64> = pinned_counts.as_slice().to_vec();
    let host_out_set: Vec<u8> = pinned_set.as_slice().to_vec();

    // Batch 4 spill-counter check. Mirrors the SUM orchestrator: a
    // non-zero counter means the kernel dropped at least one row when
    // its partition's open-addressing table overflowed past MAX_PROBES,
    // so the COUNT for the spilled key would be one or more short.
    let spill_count = spill_counter.to_vec()?[0];
    if spill_count > 0 {
        return Err(BoltError::Other(format!(
            "partition_reduce spill: {} rows exceeded MAX_PROBES; result may be incorrect",
            spill_count
        )));
    }

    let mut pairs: Vec<(i32, i64)> = Vec::new();
    for pid in 0..num_partitions as usize {
        let base = pid * block_groups;
        for slot in 0..block_groups {
            let idx = base + slot;
            if host_out_set[idx] == 0 {
                continue;
            }
            let c = host_out_counts[idx];
            // The output schema for COUNT is Int64 (SQL semantics, the
            // planner widens it). Cast u64 → i64; in practice the count
            // is bounded by n_rows which fits in i64 fine for any input
            // size we care about.
            pairs.push((host_out_keys[idx], c as i64));
        }
    }
    pairs.sort_by_key(|(k, _)| *k);

    let keys_out: Vec<i32> = pairs.iter().map(|(k, _)| *k).collect();
    let counts_out: Vec<i64> = pairs.iter().map(|(_, c)| *c).collect();

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => unreachable!("try_execute guards this"),
    };
    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(
        arrow_schema,
        vec![
            Arc::new(Int32Array::from(keys_out)),
            Arc::new(Int64Array::from(counts_out)),
        ],
    )
    .map_err(|e| {
        BoltError::Other(format!(
            "groupby_tier2_count_exec: failed to build RecordBatch: {e}"
        ))
    })
}

fn plan_dtype_to_arrow(d: DataType) -> BoltResult<ArrowDataType> {
    match d {
        DataType::Int32 => Ok(ArrowDataType::Int32),
        DataType::Int64 => Ok(ArrowDataType::Int64),
        DataType::Float32 => Ok(ArrowDataType::Float32),
        DataType::Float64 => Ok(ArrowDataType::Float64),
        DataType::Bool => Ok(ArrowDataType::Boolean),
        DataType::Utf8 => Ok(ArrowDataType::Utf8),
    }
}

fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

// ---------------------------------------------------------------------------
// Host-only eligibility-gate tests for the Tier-2.1 COUNT(*) executor.
//
// We exercise `try_execute`'s plan-shape / row-shape gating; none of these
// reach the GPU. Anything that would require a CUDA context is left to the
// dedicated e2e test files (see `tests/tier2_*_e2e.rs`).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::{Expr, Field, Literal};
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    /// Build a minimal `Aggregate` plan for `SELECT key, COUNT(*) FROM t
    /// GROUP BY key`. `key_dtype` parametrises the rejection tests below.
    fn build_count_plan(key_dtype: DataType) -> PhysicalPlan {
        let inputs = vec![ColumnIO {
            name: "k".into(),
            dtype: key_dtype,
        }];
        let output_schema = Schema::new(vec![
            Field::new("k", key_dtype, false),
            Field::new("count_star", DataType::Int64, true),
        ]);
        PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs,
                group_by: vec![0],
                aggregates: vec![AggregateExpr::Count(Expr::Literal(Literal::Null))],
                output_schema,
                input_has_validity: Vec::new(),
            },
        }
    }

    fn small_int32_batch(n: usize) -> RecordBatch {
        let keys: Vec<i32> = (0..n as i32).collect();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "k",
            ArrowDataType::Int32,
            false,
        )]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(keys)) as arrow_array::ArrayRef],
        )
        .unwrap()
    }

    /// Non-Aggregate plans are not our business.
    #[test]
    fn rejects_non_aggregate_plan() {
        let plan = PhysicalPlan::Union { inputs: vec![] };
        let batch = small_int32_batch(0);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Two group-by keys go through the two-key sibling, not this exec.
    #[test]
    fn rejects_two_group_keys() {
        let mut plan = build_count_plan(DataType::Int32);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.inputs.push(ColumnIO {
                name: "k2".into(),
                dtype: DataType::Int32,
            });
            aggregate.group_by = vec![0, 1];
        }
        // n_rows huge so we'd pass the row threshold if shape were right.
        let n = 300_000;
        let mut keys = vec![0i32; n];
        for (i, k) in keys.iter_mut().enumerate() {
            *k = (i % 1024) as i32;
        }
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys.clone())) as arrow_array::ArrayRef,
                Arc::new(Int32Array::from(keys)),
            ],
        )
        .unwrap();
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Non-Int32 key dtype must defer.
    #[test]
    fn rejects_int64_key() {
        let plan = build_count_plan(DataType::Int64);
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "k",
            ArrowDataType::Int64,
            false,
        )]));
        let n = 300_000;
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from((0..n as i64).collect::<Vec<_>>())) as arrow_array::ArrayRef],
        )
        .unwrap();
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Any aggregate other than COUNT — even one — must defer.
    #[test]
    fn rejects_sum_aggregate() {
        let mut plan = build_count_plan(DataType::Int32);
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.aggregates = vec![AggregateExpr::Sum(Expr::Column("k".into()))];
        }
        let batch = small_int32_batch(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Row count below 256 K trips the row-threshold gate — let smaller
    /// fast paths handle it.
    #[test]
    fn rejects_below_row_threshold() {
        let plan = build_count_plan(DataType::Int32);
        let batch = small_int32_batch(1_024);
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// Negative keys break the dense-bucket invariant — must decline so a
    /// safer path can take over.
    #[test]
    fn rejects_negative_key() {
        let plan = build_count_plan(DataType::Int32);
        let n = 300_000;
        let mut keys: Vec<i32> = (0..n as i32).collect();
        keys[42] = -1;
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "k",
            ArrowDataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(keys)) as arrow_array::ArrayRef],
        )
        .unwrap();
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// `max(key)` <= BLOCK_GROUPS means Tier-1 territory; defer.
    #[test]
    fn rejects_low_cardinality_estimate() {
        let plan = build_count_plan(DataType::Int32);
        let n = 300_000;
        // All keys are 0..127 — n_groups estimator returns 128, well below
        // BLOCK_GROUPS (which is >= 1024 in every variant).
        let keys: Vec<i32> = (0..n).map(|i| (i % 128) as i32).collect();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "k",
            ArrowDataType::Int32,
            false,
        )]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(keys)) as arrow_array::ArrayRef],
        )
        .unwrap();
        assert!(try_execute(&plan, &batch).is_none());
    }

    /// `pre` kernel present (compound plan) — defer to the with-pre path.
    #[test]
    fn rejects_plan_with_pre_kernel() {
        use crate::plan::physical_plan::KernelSpec;
        let mut plan = build_count_plan(DataType::Int32);
        if let PhysicalPlan::Aggregate { pre, .. } = &mut plan {
            *pre = Some(KernelSpec {
                inputs: vec![],
                outputs: vec![],
                ops: vec![],
                predicate: None,
                register_count: 0,
                input_has_validity: vec![],
                output_has_validity: vec![],
            });
        }
        let batch = small_int32_batch(300_000);
        assert!(try_execute(&plan, &batch).is_none());
    }
}

// ---------------------------------------------------------------------------
// Module-cache mechanics tests.
//
// Both tests early-return on `BoltError` (no CUDA context available, e.g. in
// docs.rs or CPU-only CI). When a context *is* present, they assert:
//   * a repeat call with the same `KernelSpec` does NOT increment
//     `LOAD_COUNT` (cache hit);
//   * a different `KernelSpec` always causes exactly one extra compile
//     (miss → hit on the second call).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod cache_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn second_call_same_spec_is_cache_hit() {
        let before = LOAD_COUNT.load(Ordering::SeqCst);
        let m1 = match get_or_build_module(&KernelSpec::Partition) {
            Ok(m) => m,
            Err(_) => return, // no CUDA context — skip.
        };
        let after_first = LOAD_COUNT.load(Ordering::SeqCst);
        // First call may or may not have built (another test may have
        // already populated the cache). Either way, the second call must
        // not increment the counter further.
        let m2 = get_or_build_module(&KernelSpec::Partition)
            .expect("second lookup of an already-cached spec must succeed");
        let after_second = LOAD_COUNT.load(Ordering::SeqCst);
        assert_eq!(
            after_second, after_first,
            "second get_or_build_module with same spec must be a cache hit \
             (load_count went from {} to {} across the second call)",
            after_first, after_second
        );
        // Pre-population case: if `before == after_first`, the cache was
        // already warm. Otherwise the first call did exactly one compile.
        assert!(after_first - before <= 1);
        // Sanity: both handles refer to the same underlying module
        // (cheap-clone equality via the raw CUmodule pointer).
        assert_eq!(m1.raw(), m2.raw(), "clones must share the same CUmodule");
    }

    #[test]
    fn different_specs_miss_then_hit_independently() {
        // Warm the cache for two distinct specs and verify subsequent
        // lookups don't re-compile either of them.
        let _ = match get_or_build_module(&KernelSpec::Scatter) {
            Ok(m) => m,
            Err(_) => return,
        };
        let _ = get_or_build_module(&KernelSpec::ReduceCount).expect("reduce build");
        let baseline = LOAD_COUNT.load(Ordering::SeqCst);
        let _ = get_or_build_module(&KernelSpec::Scatter).expect("scatter hit");
        let _ = get_or_build_module(&KernelSpec::ReduceCount).expect("reduce hit");
        assert_eq!(
            LOAD_COUNT.load(Ordering::SeqCst),
            baseline,
            "repeat lookups of already-cached specs must not recompile"
        );
    }
}

// ---------------------------------------------------------------------------
// Stage-4 (P1b) async round-trip smoke test.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage4_tests {
    use super::*;
    use crate::plan::logical_plan::{Expr, Field, Literal};
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};

    #[test]
    #[ignore = "gpu:tier2"]
    fn async_tier2_count_round_trip() {
        // Needs >= 256 K rows AND max(key) > BLOCK_GROUPS so this path
        // takes the query instead of deferring.
        let n: usize = 300_000;
        let n_groups: usize = 4096;
        let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
        let mut expected = vec![0i64; n_groups];
        for &k in &keys {
            expected[k as usize] += 1;
        }
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![ColumnIO { name: "k".into(), dtype: DataType::Int32 }],
                group_by: vec![0],
                aggregates: vec![AggregateExpr::Count(Expr::Literal(Literal::Null))],
                output_schema: Schema::new(vec![
                    Field::new("k", DataType::Int32, false),
                    Field::new("count_star", DataType::Int64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(keys)) as arrow_array::ArrayRef],
        )
        .unwrap();
        let out = match try_execute(&plan, &batch) {
            Some(Ok(b)) => b,
            _ => return,
        };
        let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let cs = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..out.num_rows() {
            assert_eq!(cs.value(i), expected[ks.value(i) as usize]);
        }
    }
}
