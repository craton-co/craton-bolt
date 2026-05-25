// SPDX-License-Identifier: Apache-2.0

//! Tier-1 GROUP BY dispatch: pick between the existing global-atomic kernel
//! and the new per-block shared-memory pre-aggregation kernel.
//!
//! Background (see `docs/GROUPBY_PERF.md`, "Tier 1"): the new shared-mem
//! kernel only beats the global-atomic path when the per-block hash table
//! fits in shared memory, i.e. `n_groups <= BLOCK_GROUPS` (1024 in the
//! first cut). For larger cardinalities the old kernel still wins; the
//! hash-partitioned two-pass design is Tier-2 work.
//!
//! This module is **pure selection logic** — no GPU calls, no I/O. The
//! decision is unit-testable in isolation. Threshold values are exposed as
//! `pub const` so a future auto-tuner (or just `grep`) can find and adjust
//! them without spelunking through the function body.
//!
//! # Policy (v0)
//!
//! Pick `SharedMemPreAgg` iff **all** of:
//!
//! 1. `n_groups <= SHARED_MEM_MAX_GROUPS` — table fits in per-block smem.
//! 2. `n_rows  >= SHARED_MEM_MIN_ROWS`    — amortise launch overhead.
//! 3. `n_key_cols == 1`                   — multi-key is Tier-2 work.
//! 4. `op == Sum && value_dtype == Float64` — first cut targets `SUM`/F64.
//! 5. `key_dtype == Int32`                — first cut targets Int32 keys.
//!
//! Otherwise fall back to `GlobalAtomic`, which is always correct.

use crate::plan::DataType;

/// Aggregate op the dispatcher cares about.
///
/// Defined locally rather than reusing [`crate::plan::AggregateExpr`] because
/// `AggregateExpr` wraps an `Expr` (not `Copy`) and we want
/// [`DispatchInputs`] to be `Copy` for cheap fan-out from the planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    /// `SUM(expr)`.
    Sum,
    /// `COUNT(expr)` / `COUNT(*)`.
    Count,
    /// `MIN(expr)`.
    Min,
    /// `MAX(expr)`.
    Max,
    /// `AVG(expr)`.
    Avg,
}

/// Result of the dispatch decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupByStrategy {
    /// Use the existing global-atomic kernel (the
    /// [`crate::exec::groupby_valid`] path). Always correct; the safe
    /// choice when the new fast path's preconditions aren't met.
    GlobalAtomic,
    /// Use the new per-block shared-memory pre-aggregation kernel.
    ///
    /// Preconditions enforced by [`dispatch`]: keys bounded by
    /// [`SHARED_MEM_MAX_GROUPS`], op + dtype in the supported set
    /// (`SUM`/`Float64` in this first cut), single-key GROUP BY, input
    /// rows above [`SHARED_MEM_MIN_ROWS`].
    SharedMemPreAgg,
}

/// Inputs to the dispatcher's decision.
///
/// All fields are cheap (POD): `Copy`, so the planner can pass this by
/// value without lifetime gymnastics.
#[derive(Debug, Clone, Copy)]
pub struct DispatchInputs {
    /// Number of distinct group keys the planner expects.
    ///
    /// May be an upper bound (e.g. plumbed from `keys_table.len()`); the
    /// dispatcher treats it as the worst case when checking against
    /// [`SHARED_MEM_MAX_GROUPS`], so an over-estimate is conservative
    /// (will more often fall back to the safe path).
    pub n_groups: u32,
    /// Number of input rows. Used to gate-out shared-mem on tiny inputs
    /// where launch overhead dominates.
    pub n_rows: u32,
    /// Number of GROUP BY key columns. Multi-key is not yet supported on
    /// the fast path.
    pub n_key_cols: usize,
    /// Aggregate op (`SUM` / `COUNT` / `MIN` / `MAX` / `AVG`).
    pub op: AggOp,
    /// Aggregate input dtype.
    pub value_dtype: DataType,
    /// First (and only, for now) group-key dtype.
    pub key_dtype: DataType,
}

/// Maximum distinct group count the shared-mem kernel can handle in one
/// pass — equal to `BLOCK_GROUPS` in the sibling kernel emitter
/// (`src/jit/shmem_sum_kernel.rs::BLOCK_GROUPS`). Defined here as well to
/// avoid a cross-agent build dependency; the merger reconciles the two.
pub const SHARED_MEM_MAX_GROUPS: u32 = 1024;

/// Minimum input-row count to consider the shared-mem path. Below this,
/// the extra kernel launch + per-block reduction overhead is not
/// amortised by the reduced atomic contention.
pub const SHARED_MEM_MIN_ROWS: u32 = 64 * 1024;

/// Decide which GROUP BY path to take for a single-aggregate query.
///
/// Pure function: no I/O, no GPU calls. See the module docs for the full
/// policy.
pub fn dispatch(inputs: DispatchInputs) -> GroupByStrategy {
    let DispatchInputs {
        n_groups,
        n_rows,
        n_key_cols,
        op,
        value_dtype,
        key_dtype,
    } = inputs;

    // 1. Cardinality must fit in a single block's shared-mem table.
    if n_groups > SHARED_MEM_MAX_GROUPS {
        return GroupByStrategy::GlobalAtomic;
    }
    // 2. Input must be large enough to amortise launch + reduction overhead.
    if n_rows < SHARED_MEM_MIN_ROWS {
        return GroupByStrategy::GlobalAtomic;
    }
    // 3. Multi-key GROUP BY is Tier-2.
    if n_key_cols != 1 {
        return GroupByStrategy::GlobalAtomic;
    }
    // 4. First cut supports only SUM(Float64).
    if !matches!((op, value_dtype), (AggOp::Sum, DataType::Float64)) {
        return GroupByStrategy::GlobalAtomic;
    }
    // 5. First cut supports only Int32 keys.
    if key_dtype != DataType::Int32 {
        return GroupByStrategy::GlobalAtomic;
    }

    GroupByStrategy::SharedMemPreAgg
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shape that satisfies every precondition; tests below mutate one
    /// field at a time to verify the policy rejects the corresponding
    /// dimension.
    fn eligible_baseline() -> DispatchInputs {
        DispatchInputs {
            n_groups: 100,
            n_rows: 10_000_000,
            n_key_cols: 1,
            op: AggOp::Sum,
            value_dtype: DataType::Float64,
            key_dtype: DataType::Int32,
        }
    }

    #[test]
    fn chooses_shmem_for_small_card_sum_f64() {
        let inputs = eligible_baseline();
        assert_eq!(dispatch(inputs), GroupByStrategy::SharedMemPreAgg);
    }

    #[test]
    fn falls_back_for_large_card() {
        let inputs = DispatchInputs {
            n_groups: 10_000,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::GlobalAtomic);
    }

    #[test]
    fn falls_back_for_tiny_input() {
        let inputs = DispatchInputs {
            n_rows: 1_000,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::GlobalAtomic);
    }

    #[test]
    fn falls_back_for_avg() {
        let inputs = DispatchInputs {
            op: AggOp::Avg,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::GlobalAtomic);
    }

    #[test]
    fn falls_back_for_int_value() {
        let inputs = DispatchInputs {
            value_dtype: DataType::Int32,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::GlobalAtomic);
    }

    #[test]
    fn falls_back_for_int64_key() {
        let inputs = DispatchInputs {
            key_dtype: DataType::Int64,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::GlobalAtomic);
    }

    #[test]
    fn falls_back_for_two_key_groupby() {
        let inputs = DispatchInputs {
            n_key_cols: 2,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::GlobalAtomic);
    }

    // -- Boundary sanity checks -------------------------------------------

    #[test]
    fn boundary_n_groups_at_limit_is_eligible() {
        // The condition is `n_groups > SHARED_MEM_MAX_GROUPS`, so the
        // exact threshold value is still accepted.
        let inputs = DispatchInputs {
            n_groups: SHARED_MEM_MAX_GROUPS,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::SharedMemPreAgg);
    }

    #[test]
    fn boundary_n_groups_one_over_limit_falls_back() {
        let inputs = DispatchInputs {
            n_groups: SHARED_MEM_MAX_GROUPS + 1,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::GlobalAtomic);
    }

    #[test]
    fn boundary_n_rows_at_min_is_eligible() {
        let inputs = DispatchInputs {
            n_rows: SHARED_MEM_MIN_ROWS,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::SharedMemPreAgg);
    }

    #[test]
    fn boundary_n_rows_one_under_min_falls_back() {
        let inputs = DispatchInputs {
            n_rows: SHARED_MEM_MIN_ROWS - 1,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::GlobalAtomic);
    }

    #[test]
    fn falls_back_for_zero_key_cols() {
        // Defensive: scalar aggregate (no GROUP BY) should never hit this
        // dispatcher, but if it does, route it through the safe path.
        let inputs = DispatchInputs {
            n_key_cols: 0,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::GlobalAtomic);
    }

    #[test]
    fn falls_back_for_count_op() {
        let inputs = DispatchInputs {
            op: AggOp::Count,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::GlobalAtomic);
    }

    #[test]
    fn falls_back_for_min_max_ops() {
        let min_in = DispatchInputs {
            op: AggOp::Min,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(min_in), GroupByStrategy::GlobalAtomic);
        let max_in = DispatchInputs {
            op: AggOp::Max,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(max_in), GroupByStrategy::GlobalAtomic);
    }

    #[test]
    fn falls_back_for_float32_value() {
        let inputs = DispatchInputs {
            value_dtype: DataType::Float32,
            ..eligible_baseline()
        };
        assert_eq!(dispatch(inputs), GroupByStrategy::GlobalAtomic);
    }
}
