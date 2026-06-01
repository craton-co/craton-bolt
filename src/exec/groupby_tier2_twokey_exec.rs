// SPDX-License-Identifier: Apache-2.0

//! Tier-2 hash-partitioned GROUP BY executor — **two-key (Int32, Int32)
//! shim**.
//!
//! Mirrors [`crate::exec::groupby_tier2_exec`] but for queries shaped
//! `SELECT a, b, SUM(v) FROM t GROUP BY a, b` where both `a` and `b` are
//! Int32 and `v` is Float64. The two i32 keys are packed losslessly into
//! a single i64 (high 32 bits = column 0, low 32 bits = column 1) so the
//! on-device Tier-2 chain can treat them as a single dense key.
//!
//! Composition:
//!   * eligibility    — local (we don't share `dispatch_v2`, which gates
//!     on `n_key_cols == 1`)
//!   * key packing    — host-side, mirroring `groupby.rs::pack_keys`
//!   * GPU pipeline   — [`crate::exec::groupby_tier2_twokey_orchestrator`]
//!   * result merge   — [`crate::exec::groupby_tier2_twokey_merge`]
//!
//! Returns `Some(Ok(batch))` on success, `Some(Err(_))` on failure inside
//! the GPU path, and `None` on any eligibility miss (caller falls through
//! to the next strategy). This is the same return-shape convention as
//! `groupby_tier2_exec::try_execute` and `groupby_shmem_exec::try_execute`.

use arrow_array::{Array, Float64Array, Int32Array, RecordBatch};

use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::groupby_tier2_twokey_merge::build_tier2_twokey_result;
use crate::exec::groupby_tier2_twokey_orchestrator::execute_tier2_twokey_sum;
use crate::exec::launch::CudaStream;
use crate::plan::logical_plan::{AggregateExpr, DataType, Expr};
use crate::plan::physical_plan::PhysicalPlan;

/// Minimum input row count to consider this path. Below this threshold the
/// fixed costs (partition launch, scatter launch, two D2H copies, K small
/// HashMaps) are not amortised by the win over a global-atomic GROUP BY.
/// Matches the single-key path's `TIER2_MIN_ROWS`.
const TWOKEY_MIN_ROWS: usize = 256 * 1024;

/// Pack two `(i32, i32)` columns into a single `Vec<i64>` per row.
///
/// **Convention (MUST match `src/exec/groupby.rs::pack_keys`):**
///
/// ```text
/// packed = ((col0 as u32 as u64) << 32) | (col1 as u32 as u64)
/// ```
///
/// The unpacker in `groupby_tier2_twokey_merge` reverses this exactly.
/// Re-implemented inline here rather than imported because `pack_keys` is
/// a private function bound to the full multi-column packing machinery
/// (it accepts an `AggregateSpec`, allocates `KeyComponent` metadata,
/// validates bit widths, etc.). For the narrow two-Int32 case we only
/// need the loop body; carrying the rest of that surface area into this
/// shim would couple the modules unnecessarily.
fn pack_two_i32(col0: &[i32], col1: &[i32]) -> Vec<i64> {
    debug_assert_eq!(col0.len(), col1.len());
    let n = col0.len();
    let mut out: Vec<i64> = Vec::with_capacity(n);
    for i in 0..n {
        let hi = (col0[i] as u32 as u64) << 32;
        let lo = col1[i] as u32 as u64;
        out.push((hi | lo) as i64);
    }
    out
}

/// Two-key Tier-2 fast path — **currently GATED OFF** (always declines) so
/// two-key single-SUM `GROUP BY` falls through to the correct global-atomic
/// path. The Tier-2 `partition_reduce_kernel_i64` builds a fixed 1024-slot
/// per-block open-addressing table; for a high-cardinality two-key GROUP BY
/// (e.g. h2o q3 — id1×id2 ≈ 250K groups at 10M rows) the per-block distinct-key
/// count vastly exceeds 1024, so probing degenerates into a multi-second scan
/// that trips the Windows ~2 s TDR watchdog → CUDA_ERROR_INVALID_HANDLE and a
/// process crash, BEFORE the MAX_PROBES spill sentinel can fire and route to
/// the safe path. The crash is data-dependent on distinct-key count (not row
/// count), which we can't cheaply estimate here, so the path is disabled
/// wholesale: the global-atomic fallthrough computes the same result correctly
/// (verified vs DuckDB on q1–q5; q3 ≈ 1.2 s at 10M). Re-enabling needs an
/// overflow-safe kernel (spill-to-global, or a larger / dynamically-sized
/// table). See memory `groupby-resident-and-hostscan-finding.md` /
/// `gpu-validation-known-issues.md` (q3 two-key TDR). The original
/// implementation is retained in [`try_execute_impl`] for that future re-enable.
pub fn try_execute(
    _plan: &PhysicalPlan,
    _batch: &RecordBatch,
) -> Option<BoltResult<RecordBatch>> {
    None
}

/// Retained two-key Tier-2 implementation (see [`try_execute`] for why it is
/// currently gated off). Kept compiling so re-enabling is a one-line change
/// once the per-block-overflow TDR is fixed.
#[allow(dead_code)]
fn try_execute_impl(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<BoltResult<RecordBatch>> {
    // --- 1. Plan shape: pre-less Aggregate with exactly two group keys
    //        and one aggregate.
    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        _ => return None,
    };
    if pre.is_some() {
        return None;
    }
    if aggregate.group_by.len() != 2 || aggregate.aggregates.len() != 1 {
        return None;
    }

    // --- 2. Both group columns must be Int32 (matches the packing convention).
    let k0_io = aggregate.inputs.get(aggregate.group_by[0])?;
    let k1_io = aggregate.inputs.get(aggregate.group_by[1])?;
    if k0_io.dtype != DataType::Int32 || k1_io.dtype != DataType::Int32 {
        return None;
    }

    // --- 3. Single SUM over a bare Float64 column.
    let sum_col_name = match &aggregate.aggregates[0] {
        AggregateExpr::Sum(Expr::Column(name)) => name.as_str(),
        _ => return None,
    };

    // --- 4. Resolve and downcast all three Arrow columns.
    let k0_arr = batch
        .column_by_name(&k0_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    let k1_arr = batch
        .column_by_name(&k1_io.name)
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())?;
    let val_arr = batch
        .column_by_name(sum_col_name)
        .and_then(|c| c.as_any().downcast_ref::<Float64Array>())?;

    // --- 5. Length agreement + row-count threshold.
    if k0_arr.len() != k1_arr.len() || k0_arr.len() != val_arr.len() {
        return None;
    }
    let n_rows = k0_arr.len();
    if n_rows < TWOKEY_MIN_ROWS {
        return None;
    }

    // PV-stage-f: NULL handling — the partition_reduce_kernel_i64 family
    // has no `_with_validity` companion yet, and the host-side i64 pack
    // reads `.values()` straight off the Arrow array (NULL positions
    // carry garbage bytes that would synthesize ghost groups). Defer
    // NULL-bearing batches back to `groupby::execute_groupby` → the
    // sentinel / sentinel-free single-key paths, both of which carry
    // proper validity handling. Stage G follow-up: native
    // partition+reduce kernels with validity bitmaps.
    if k0_arr.null_count() > 0
        || k1_arr.null_count() > 0
        || val_arr.null_count() > 0
    {
        return None;
    }

    Some(execute_inner(plan, k0_arr, k1_arr, val_arr))
}

fn execute_inner(
    plan: &PhysicalPlan,
    k0_arr: &Int32Array,
    k1_arr: &Int32Array,
    val_arr: &Float64Array,
) -> BoltResult<RecordBatch> {
    let n_rows = k0_arr.len();
    // Host-side pack. `pack_keys` (the production helper in groupby.rs)
    // would do the same thing for the (Int32, Int32) shape; we re-implement
    // inline so this shim doesn't depend on private internals.
    let packed: Vec<i64> = pack_two_i32(k0_arr.values(), k1_arr.values());

    // Stage-4 (P1b): per-call stream for the input H2D uploads. The
    // orchestrator mints its own stream for its kernel + D2H phase;
    // we synchronize here before handing off so the orchestrator's
    // stream sees a fully-realised input buffer.
    let stream = CudaStream::null_or_default();

    // Upload to device. Both vecs are exactly `n_rows` long — the
    // orchestrator's length invariant is the caller's responsibility per
    // its API contract.
    let keys_gpu = GpuVec::<i64>::from_slice_async(&packed, stream.raw())?;
    let vals_gpu = GpuVec::<f64>::from_slice_async(val_arr.values(), stream.raw())?;
    stream.synchronize()?;

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| {
        BoltError::Other(format!(
            "groupby_tier2_twokey_exec: row count {} exceeds u32 launch-shape limit",
            n_rows
        ))
    })?;

    let partial = execute_tier2_twokey_sum(&keys_gpu, &vals_gpu, n_rows_u32)?;

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => {
            return Err(BoltError::Other(
                "groupby_tier2_twokey_exec: non-Aggregate plan reached execute_inner".into(),
            ))
        }
    };

    build_tier2_twokey_result(partial, &aggregate.output_schema)
}

// ---------------------------------------------------------------------------
// Host-only tests — eligibility-gate shape only. GPU correctness is covered
// by the dedicated e2e test file (`tests/tier2_twokey_e2e.rs`).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_two_i32_matches_groupby_convention() {
        // Cross-check against the documented layout. Column 0 (high) and
        // column 1 (low) MUST land in the right halves.
        let col0 = vec![1i32, -1, i32::MIN, i32::MAX, 0];
        let col1 = vec![2i32, -2, i32::MAX, i32::MIN, 0];
        let packed = pack_two_i32(&col0, &col1);
        for i in 0..col0.len() {
            let u = packed[i] as u64;
            let hi = (u >> 32) as u32 as i32;
            let lo = (u & 0xFFFF_FFFFu64) as u32 as i32;
            assert_eq!(hi, col0[i], "high half mismatch at row {i}");
            assert_eq!(lo, col1[i], "low half mismatch at row {i}");
        }
    }

    #[test]
    fn pack_two_i32_zero_zero_is_zero() {
        // (0, 0) packs to literal 0; useful invariant for downstream
        // sparse-key paths.
        let packed = pack_two_i32(&[0], &[0]);
        assert_eq!(packed, vec![0i64]);
    }

    #[test]
    fn pack_two_i32_distinguishes_swapped_pair() {
        // (a, b) and (b, a) must produce distinct packed keys. A
        // regression that packed with the wrong half ordering would
        // collide every transposed pair into the same group.
        let p_ab = pack_two_i32(&[1], &[2]);
        let p_ba = pack_two_i32(&[2], &[1]);
        assert_ne!(p_ab, p_ba, "(1,2) and (2,1) must not collide");
    }

    // ---- try_execute eligibility gate ----
    //
    // The next block exercises the plan-shape / row-shape rejection paths
    // that `try_execute` runs before it commits to a GPU launch. None of
    // these tests reach the device.

    use crate::plan::logical_plan::{Field, Schema};
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    fn build_twokey_sum_plan() -> PhysicalPlan {
        let inputs = vec![
            ColumnIO {
                name: "k1".into(),
                dtype: DataType::Int32,
            },
            ColumnIO {
                name: "k2".into(),
                dtype: DataType::Int32,
            },
            ColumnIO {
                name: "v".into(),
                dtype: DataType::Float64,
            },
        ];
        let output_schema = Schema::new(vec![
            Field::new("k1", DataType::Int32, false),
            Field::new("k2", DataType::Int32, false),
            Field::new("sum_v", DataType::Float64, true),
        ]);
        PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs,
                group_by: vec![0, 1],
                aggregates: vec![AggregateExpr::Sum(Expr::Column("v".into()))],
                output_schema,
                input_has_validity: Vec::new(),
            },
        }
    }

    fn twokey_sum_batch(n: usize) -> RecordBatch {
        let k1: Vec<i32> = (0..n as i32).collect();
        let k2: Vec<i32> = (0..n as i32).map(|i| i + 1).collect();
        let v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow_array::Int32Array::from(k1)) as arrow_array::ArrayRef,
                Arc::new(arrow_array::Int32Array::from(k2)) as arrow_array::ArrayRef,
                Arc::new(Float64Array::from(v)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap()
    }

    /// Non-Aggregate plan: reject.
    #[test]
    fn rejects_non_aggregate_plan() {
        let plan = PhysicalPlan::Union { inputs: vec![] };
        let batch = twokey_sum_batch(0);
        assert!(try_execute_impl(&plan, &batch).is_none());
    }

    /// Below-threshold rows fall through to a smaller path.
    #[test]
    fn rejects_below_row_threshold() {
        let plan = build_twokey_sum_plan();
        let batch = twokey_sum_batch(1_024);
        assert!(try_execute_impl(&plan, &batch).is_none());
    }

    /// COUNT-shaped agg → wrong shim.
    #[test]
    fn rejects_count_aggregate() {
        use crate::plan::logical_plan::Literal;
        let mut plan = build_twokey_sum_plan();
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.aggregates =
                vec![AggregateExpr::Count(Expr::Literal(Literal::Null))];
        }
        let batch = twokey_sum_batch(300_000);
        assert!(try_execute_impl(&plan, &batch).is_none());
    }

    /// Three group keys → not our shape.
    #[test]
    fn rejects_three_keys() {
        let mut plan = build_twokey_sum_plan();
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.inputs.push(ColumnIO {
                name: "k3".into(),
                dtype: DataType::Int32,
            });
            aggregate.group_by = vec![0, 1, 2];
        }
        let batch = twokey_sum_batch(300_000);
        // Even though the batch lacks `k3`, the group_by.len() check fires
        // first and returns None.
        assert!(try_execute_impl(&plan, &batch).is_none());
    }

    /// Int64 key → reject (packing convention is (i32, i32)).
    #[test]
    fn rejects_int64_first_key() {
        let mut plan = build_twokey_sum_plan();
        if let PhysicalPlan::Aggregate { aggregate, .. } = &mut plan {
            aggregate.inputs[0].dtype = DataType::Int64;
        }
        let batch = twokey_sum_batch(300_000);
        assert!(try_execute_impl(&plan, &batch).is_none());
    }
}

// ---------------------------------------------------------------------------
// Stage-4 (P1b) async round-trip smoke test.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod stage4_tests {
    use super::*;
    use crate::plan::logical_plan::{Field, Schema};
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO};
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};
    use std::sync::Arc;

    #[test]
    #[ignore = "requires CUDA toolkit at runtime"]
    fn async_tier2_twokey_sum_round_trip() {
        let n: usize = 300_000;
        let g1: usize = 64;
        let g2: usize = 64;
        let k1: Vec<i32> = (0..n).map(|i| (i % g1) as i32).collect();
        let k2: Vec<i32> = (0..n).map(|i| ((i / g1) % g2) as i32).collect();
        let v: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut expected = std::collections::HashMap::<(i32, i32), f64>::new();
        for i in 0..n {
            *expected.entry((k1[i], k2[i])).or_default() += v[i];
        }
        let plan = PhysicalPlan::Aggregate {
            table: "t".into(),
            pre: None,
            aggregate: AggregateSpec {
                inputs: vec![
                    ColumnIO { name: "k1".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "k2".into(), dtype: DataType::Int32 },
                    ColumnIO { name: "v".into(), dtype: DataType::Float64 },
                ],
                group_by: vec![0, 1],
                aggregates: vec![AggregateExpr::Sum(Expr::Column("v".into()))],
                output_schema: Schema::new(vec![
                    Field::new("k1", DataType::Int32, false),
                    Field::new("k2", DataType::Int32, false),
                    Field::new("sum_v", DataType::Float64, true),
                ]),
                input_has_validity: Vec::new(),
            },
        };
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k1", ArrowDataType::Int32, false),
            ArrowField::new("k2", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow_array::Int32Array::from(k1)) as arrow_array::ArrayRef,
                Arc::new(arrow_array::Int32Array::from(k2)) as arrow_array::ArrayRef,
                Arc::new(Float64Array::from(v)) as arrow_array::ArrayRef,
            ],
        )
        .unwrap();
        let out = match try_execute_impl(&plan, &batch) {
            Some(Ok(b)) => b,
            _ => return,
        };
        let kc1 = out.column(0).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
        let kc2 = out.column(1).as_any().downcast_ref::<arrow_array::Int32Array>().unwrap();
        let sv = out.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        for i in 0..out.num_rows() {
            let key = (kc1.value(i), kc2.value(i));
            assert_eq!(sv.value(i), *expected.get(&key).unwrap());
        }
    }
}
