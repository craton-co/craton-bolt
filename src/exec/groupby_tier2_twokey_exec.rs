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

use arrow_array::{Float64Array, Int32Array, RecordBatch};

use crate::cuda::GpuVec;
use crate::error::{PatinaError, PatinaResult};
use crate::exec::groupby_tier2_twokey_merge::build_tier2_twokey_result;
use crate::exec::groupby_tier2_twokey_orchestrator::execute_tier2_twokey_sum;
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

/// Try the two-key Tier-2 fast path. Returns `None` on any precondition
/// miss so the caller falls through to the next strategy.
pub fn try_execute(
    plan: &PhysicalPlan,
    batch: &RecordBatch,
) -> Option<PatinaResult<RecordBatch>> {
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

    Some(execute_inner(plan, k0_arr, k1_arr, val_arr))
}

fn execute_inner(
    plan: &PhysicalPlan,
    k0_arr: &Int32Array,
    k1_arr: &Int32Array,
    val_arr: &Float64Array,
) -> PatinaResult<RecordBatch> {
    let n_rows = k0_arr.len();
    // Host-side pack. `pack_keys` (the production helper in groupby.rs)
    // would do the same thing for the (Int32, Int32) shape; we re-implement
    // inline so this shim doesn't depend on private internals.
    let packed: Vec<i64> = pack_two_i32(k0_arr.values(), k1_arr.values());

    // Upload to device. Both vecs are exactly `n_rows` long — the
    // orchestrator's length invariant is the caller's responsibility per
    // its API contract.
    let keys_gpu = GpuVec::<i64>::from_slice(&packed)?;
    let vals_gpu = GpuVec::<f64>::from_slice(val_arr.values())?;

    let n_rows_u32 = u32::try_from(n_rows).map_err(|_| {
        PatinaError::Other(format!(
            "groupby_tier2_twokey_exec: row count {} exceeds u32 launch-shape limit",
            n_rows
        ))
    })?;

    let partial = execute_tier2_twokey_sum(&keys_gpu, &vals_gpu, n_rows_u32)?;

    let aggregate = match plan {
        PhysicalPlan::Aggregate { aggregate, .. } => aggregate,
        _ => {
            return Err(PatinaError::Other(
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
}
