// SPDX-License-Identifier: Apache-2.0

//! Tier-2 GROUP BY dispatch (v2): pick between the existing global-atomic
//! kernel, the Tier-1 per-block shared-memory pre-aggregation kernel, and
//! the new Tier-2 hash-partitioned two-pass kernel.
//!
//! Background (see `docs/GROUPBY_PERF.md`, "Tier 2"): when `n_groups`
//! exceeds what a single block-shared hash table can hold (the Tier-1
//! upper bound, `TIER1_MAX_GROUPS`), but is still in a range where a
//! hash-partition + per-partition Tier-1 reduction is profitable, route
//! the query through the Tier-2 kernel.
//!
//! Above the Tier-2 cardinality cap, both fast paths' assumptions break
//! down (per-partition tables would themselves overflow shared memory)
//! and we fall back to the always-correct global-atomic path.
//!
//! This module is **additive**: it does NOT modify the existing Tier-1
//! dispatcher in [`crate::exec::groupby_shmem_dispatch`]. The two
//! dispatchers will be merged by a follow-up consolidation; for now,
//! call-sites that want the three-way decision should call
//! [`dispatch_v2`].
//!
//! Like the Tier-1 dispatcher, this is **pure selection logic** — no GPU
//! calls, no I/O. Threshold values are exposed as `pub const` so an
//! auto-tuner (or `grep`) can find and adjust them without spelunking
//! through the function body.
//!
//! # Policy (v0)
//!
//! All three paths require the common precondition set that Tier-1
//! enforces: single-key `SUM(Float64)` with `Int32` keys.  Under that
//! umbrella:
//!
//! 1. Pick `SharedMemPreAgg` iff
//!    `n_groups <= TIER1_MAX_GROUPS` and `n_rows >= TIER1_MIN_ROWS`.
//! 2. Otherwise pick `Tier2Partitioned` iff
//!    `n_groups <= TIER2_MAX_GROUPS` and `n_rows >= TIER2_MIN_ROWS`
//!    (the higher row floor amortises the extra partition pass).
//! 3. Otherwise fall back to `GlobalAtomic`, which is always correct.
//!
//! Queries that fail the common precondition set (multi-key, non-SUM
//! ops, non-`f64` values, non-`i32` keys) always go to `GlobalAtomic`.
//!
//! # dedup (tier2/shmem): what is and isn't shared across the variants
//!
//! The ~20 `groupby_tier2_*` / `groupby_shmem_*` `try_execute` variants look
//! superficially duplicative, but only one block is genuinely identical and
//! safe to share: the host-side max-nonneg-key scan, now in
//! [`crate::exec::groupby_tier2_common::scan_max_nonneg_key`]. Every
//! single-key executor calls it; the per-variant empty-input handling
//! (`None` to decline vs an empty-schema result batch) and `n_groups`
//! arithmetic stay local.
//!
//! The rest is *intentionally specialized* and a blind consolidation would
//! be unsafe (and unverifiable without GPU hardware):
//!
//! * **Eligibility tails diverge** — single-key SUM gates on
//!   `group_by.len() == 1 && aggregates.len() == 1`; AVG/multi gate on an
//!   `n_vals` range; the two-key shim gates on `group_by.len() == 2` and
//!   does NOT use `dispatch_v2` (which rejects `n_key_cols != 1`); MIN/MAX
//!   branch on the value dtype.
//! * **Upload/scatter/reduce ABIs diverge** — SUM defers to an orchestrator;
//!   COUNT inlines a keys-only partition→scatter→reduce; AVG runs a
//!   deterministic `dest_idx` scatter plus *two* reduces (multi-SUM + COUNT)
//!   then divides host-side; MIN/MAX specializes the scatter and reduce on
//!   Int32 vs Int64 atomics. The kernel parameter lists are not
//!   interchangeable.
//! * **The spill-counter error string is a cross-module contract.** Its
//!   prefix is matched by `groupby.rs`'s GB-S2 soft-fallback path and
//!   exported as
//!   [`crate::exec::groupby_tier2_orchestrator::PARTITION_REDUCE_SPILL_PREFIX`];
//!   the single-counter and multi-counter messages differ by design
//!   (`"… {n} rows …"` vs `"… multi={a} count={b} …"`). Folding these behind
//!   one helper risks the sentinel and was deliberately left local.

use crate::plan::logical_plan::DataType;

/// Result of the v2 dispatch decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupByStrategyV2 {
    /// Use the existing global-atomic kernel. Always correct; chosen
    /// when neither fast path's preconditions hold.
    GlobalAtomic,
    /// Use the Tier-1 per-block shared-memory kernel. Best for low
    /// cardinality (`n_groups <= TIER1_MAX_GROUPS`).
    SharedMemPreAgg,
    /// Use the Tier-2 hash-partitioned two-pass kernel. Best for
    /// medium-to-high cardinality
    /// (`TIER1_MAX_GROUPS < n_groups <= TIER2_MAX_GROUPS`).
    Tier2Partitioned,
}

/// Aggregate op the dispatcher cares about.
///
/// Mirrors [`crate::exec::groupby_shmem_dispatch::AggOp`] verbatim
/// — defined locally so callers can use `dispatch_v2` without pulling
/// in the Tier-1 module.  Kept `Copy` so [`DispatchInputsV2`] is `Copy`.
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

/// Inputs to the v2 dispatcher's decision.
///
/// All fields are POD: `Copy`, so the planner can pass this by value
/// without lifetime gymnastics.
#[derive(Debug, Clone, Copy)]
pub struct DispatchInputsV2 {
    /// Number of distinct group keys the planner expects.  May be an
    /// upper bound; the dispatcher treats it as the worst case when
    /// checking against the tier thresholds, so an over-estimate is
    /// conservative (will more often fall back to the safe path).
    pub n_groups: u32,
    /// Number of input rows.  Used to gate-out fast paths on small
    /// inputs where launch / partition overhead dominates.
    pub n_rows: u32,
    /// Number of GROUP BY key columns.  Multi-key is not yet supported
    /// on either fast path.
    pub n_key_cols: usize,
    /// Aggregate op (`SUM` / `COUNT` / `MIN` / `MAX` / `AVG`).
    pub op: AggOp,
    /// Aggregate input dtype.
    pub value_dtype: DataType,
    /// First (and only, for now) group-key dtype.
    pub key_dtype: DataType,
}

/// Maximum distinct group count the Tier-1 (per-block shared-mem)
/// kernel can handle in one pass.  Equal to `BLOCK_GROUPS` in the
/// sibling kernel emitter (`src/jit/shmem_sum_kernel.rs`).
pub const TIER1_MAX_GROUPS: u32 = 1024;

/// Maximum distinct group count the Tier-2 (hash-partitioned two-pass)
/// kernel will accept.  Above this, the per-partition hashtables
/// themselves would exceed shared memory even after partitioning, and
/// we route through the always-correct global-atomic path.
pub const TIER2_MAX_GROUPS: u32 = 100_000_000;

/// Minimum input-row count to consider the Tier-1 path.  Below this,
/// the extra kernel launch + per-block reduction overhead is not
/// amortised by the reduced atomic contention.
pub const TIER1_MIN_ROWS: u32 = 64 * 1024;

/// Minimum input-row count to consider the Tier-2 path.  Higher than
/// [`TIER1_MIN_ROWS`] because the partition pass itself reads + writes
/// every input row; the two-pass design only amortises on larger
/// inputs.
pub const TIER2_MIN_ROWS: u32 = 256 * 1024;

/// Decide which GROUP BY path to take for a single-aggregate query.
///
/// Pure function: no I/O, no GPU calls.  See the module docs for the
/// full policy.
pub fn dispatch_v2(inputs: DispatchInputsV2) -> GroupByStrategyV2 {
    let DispatchInputsV2 {
        n_groups,
        n_rows,
        n_key_cols,
        op,
        value_dtype,
        key_dtype,
    } = inputs;

    // Common preconditions — failing any of these means neither fast
    // path can handle the query; route to the safe path.
    if n_key_cols != 1 {
        return GroupByStrategyV2::GlobalAtomic;
    }
    if !matches!((op, value_dtype), (AggOp::Sum, DataType::Float64)) {
        return GroupByStrategyV2::GlobalAtomic;
    }
    if key_dtype != DataType::Int32 {
        return GroupByStrategyV2::GlobalAtomic;
    }

    // Tier-1: cardinality fits in a single block-shared hash table
    // AND the input is large enough to amortise launch overhead.
    if n_groups <= TIER1_MAX_GROUPS && n_rows >= TIER1_MIN_ROWS {
        return GroupByStrategyV2::SharedMemPreAgg;
    }

    // Tier-2: cardinality fits in (TIER1_MAX_GROUPS, TIER2_MAX_GROUPS]
    // AND the input is large enough to amortise the partition pass.
    if n_groups > TIER1_MAX_GROUPS
        && n_groups <= TIER2_MAX_GROUPS
        && n_rows >= TIER2_MIN_ROWS
    {
        return GroupByStrategyV2::Tier2Partitioned;
    }

    GroupByStrategyV2::GlobalAtomic
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Shape that satisfies the common precondition set and the Tier-1
    /// thresholds.  Tests below mutate one field at a time to verify
    /// the policy routes each case correctly.
    fn eligible_baseline() -> DispatchInputsV2 {
        DispatchInputsV2 {
            n_groups: 500,
            n_rows: 10_000_000,
            n_key_cols: 1,
            op: AggOp::Sum,
            value_dtype: DataType::Float64,
            key_dtype: DataType::Int32,
        }
    }

    #[test]
    fn tier1_for_low_card() {
        let inputs = DispatchInputsV2 {
            n_groups: 500,
            ..eligible_baseline()
        };
        assert_eq!(dispatch_v2(inputs), GroupByStrategyV2::SharedMemPreAgg);
    }

    #[test]
    fn tier2_for_medium_card() {
        let inputs = DispatchInputsV2 {
            n_groups: 10_000,
            ..eligible_baseline()
        };
        assert_eq!(dispatch_v2(inputs), GroupByStrategyV2::Tier2Partitioned);
    }

    #[test]
    fn tier2_for_high_card() {
        let inputs = DispatchInputsV2 {
            n_groups: 1_000_000,
            ..eligible_baseline()
        };
        assert_eq!(dispatch_v2(inputs), GroupByStrategyV2::Tier2Partitioned);
    }

    #[test]
    fn tier2_boundary_low() {
        // One above the Tier-1 cap → must route to Tier-2.
        let inputs = DispatchInputsV2 {
            n_groups: TIER1_MAX_GROUPS + 1,
            ..eligible_baseline()
        };
        assert_eq!(dispatch_v2(inputs), GroupByStrategyV2::Tier2Partitioned);
    }

    #[test]
    fn global_for_extreme_card() {
        // Above the Tier-2 cap → neither fast path is safe.
        let inputs = DispatchInputsV2 {
            n_groups: 200_000_000,
            ..eligible_baseline()
        };
        assert_eq!(dispatch_v2(inputs), GroupByStrategyV2::GlobalAtomic);
    }

    #[test]
    fn global_for_tiny_input() {
        let inputs = DispatchInputsV2 {
            n_rows: 10_000,
            ..eligible_baseline()
        };
        assert_eq!(dispatch_v2(inputs), GroupByStrategyV2::GlobalAtomic);
    }

    #[test]
    fn global_for_two_key() {
        let inputs = DispatchInputsV2 {
            n_key_cols: 2,
            ..eligible_baseline()
        };
        assert_eq!(dispatch_v2(inputs), GroupByStrategyV2::GlobalAtomic);
    }

    #[test]
    fn global_for_avg() {
        let inputs = DispatchInputsV2 {
            op: AggOp::Avg,
            ..eligible_baseline()
        };
        assert_eq!(dispatch_v2(inputs), GroupByStrategyV2::GlobalAtomic);
    }

    #[test]
    fn global_for_int_value() {
        let inputs = DispatchInputsV2 {
            value_dtype: DataType::Int64,
            ..eligible_baseline()
        };
        assert_eq!(dispatch_v2(inputs), GroupByStrategyV2::GlobalAtomic);
    }

    #[test]
    fn tier1_boundary_high() {
        // Exactly at the Tier-1 cap → still Tier-1 (inclusive bound).
        let inputs = DispatchInputsV2 {
            n_groups: TIER1_MAX_GROUPS,
            ..eligible_baseline()
        };
        assert_eq!(dispatch_v2(inputs), GroupByStrategyV2::SharedMemPreAgg);
    }
}
