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
//! Above the Tier-2 cardinality cap ([`TIER2_MAX_GROUPS`], the physical
//! `NUM_PARTITIONS * BLOCK_GROUPS` output-slot capacity) spill is
//! guaranteed by pigeonhole, so we route straight to the always-correct
//! global-atomic path instead of running the full pipeline to a
//! guaranteed spill + soft-fallback.
//!
//! Like the Tier-1 dispatcher, this is **pure selection logic** — no GPU
//! calls, no I/O. Threshold values are exposed as `pub const` so an
//! auto-tuner (or `grep`) can find and adjust them without spelunking
//! through the function body.
//!
//! # Source-of-truth map for GROUP BY dispatch (read before editing)
//!
//! There is no single dispatcher; selection logic lives in three layers,
//! each with a distinct, current responsibility. None is dead code (every
//! function below has a live production caller — verified by `git grep`):
//!
//! 1. **Macro fall-through order** in
//!    [`crate::exec::groupby::execute_groupby`] (`try_fast_path!`) is the
//!    OUTERMOST source of truth: it fixes the *order* in which the ~20
//!    `try_execute` fast-path executors are offered the query. The first
//!    one whose preconditions match wins; the rest never see the query.
//! 2. **`dispatch_v2`** (this module) is the SUM/Float64/Int32 single-key
//!    cardinality dispatcher. Live callers: the single-SUM Tier-2 executor
//!    [`crate::exec::groupby_tier2_exec`] and the multi-SUM Tier-2 executor
//!    [`crate::exec::groupby_tier2_multi_exec`]. It models only SUM/F64
//!    because those are the only ops whose Tier-1-vs-Tier-2-vs-global
//!    decision is shared; it is NOT a stale stub.
//! 3. **`dispatch`** (the Tier-1 dispatcher in
//!    [`crate::exec::groupby_shmem_dispatch`]) is the analogous Tier-1
//!    SUM/F64 cardinality gate. Live callers: the single-SUM
//!    [`crate::exec::groupby_shmem_exec`] and multi-SUM
//!    [`crate::exec::groupby_shmem_multi_exec`] shmem executors.
//!
//! The non-SUM shmem/Tier-2 executors (COUNT / MIN/MAX / AVG, single- and
//! two-key) deliberately do NOT route through `dispatch` / `dispatch_v2`:
//! their eligibility tails diverge (an `n_vals` range, a value-dtype branch,
//! `n_key_cols == 2`, etc. — see the dedup note below) and each gates its
//! own cardinality floor inline against the same `pub const` thresholds
//! ([`TIER1_MAX_GROUPS`] / [`TIER1_MIN_ROWS`] / [`TIER2_MIN_ROWS`] and the
//! Tier-1 module's `SHARED_MEM_*`). Those constants are the shared knobs;
//! the per-op gates are intentionally local. An earlier note here claimed
//! the two dispatchers "will be merged by a follow-up" — that merge never
//! landed and would change executor selection (and touch files outside this
//! module), so it is NOT pursued here; this map is the reconciliation.
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
/// kernel can ever resolve correctly — the *physical* slot capacity of
/// the reduce output, `NUM_PARTITIONS * BLOCK_GROUPS`.
///
/// The Tier-2 pipeline scatters rows into `NUM_PARTITIONS` partitions and
/// then reduces each partition into a fixed `BLOCK_GROUPS`-slot shared-mem
/// hash table (see [`crate::exec::groupby_tier2_orchestrator`] and
/// [`crate::jit::partition_reduce_kernel`]). Distinct keys therefore land
/// in exactly `NUM_PARTITIONS * BLOCK_GROUPS = 4096 * 1024 = 4_194_304`
/// output slots in total. Those two constants live in
/// [`crate::jit::partition_kernel::NUM_PARTITIONS`] /
/// [`crate::jit::partition_reduce_kernel::BLOCK_GROUPS`] (NOT in this
/// file's editable set), so the value is spelled out here as `4096 *
/// 1024` with this reference comment.
///
/// Above this many distinct keys, spill is *guaranteed* by the pigeonhole
/// principle: there are strictly more distinct keys than total slots, so
/// at least one per-partition table overflows `MAX_PROBES`, the reduce
/// kernel raises the `partition_reduce spill` sentinel, and
/// `groupby::execute_groupby`'s GB-S2 soft-fallback recomputes on the
/// always-correct global-atomic path. Routing those queries straight to
/// `GlobalAtomic` here (instead of running the full partition + scatter +
/// reduce pipeline only to spill and fall back) is pure wasted-work
/// elimination — the final executor and result are identical either way.
///
/// The previous value (`100_000_000`) vastly overshot this physical
/// capacity: every query with `4_194_304 < n_groups <= 100_000_000`
/// distinct keys ran the entire pipeline to a guaranteed spill before
/// falling back, and the doc comment ("per-partition hashtables would
/// exceed shared memory") was wrong — the limit is total output slots,
/// not per-block smem.
///
/// NOTE on load factor: this is the *hard* ceiling. Real workloads can
/// spill below it when hash skew packs more than `BLOCK_GROUPS` distinct
/// keys into one partition (the orchestrator docs note correctness holds
/// "precisely when total distinct keys are <= K * BLOCK_GROUPS"). We do
/// NOT bake a load-factor margin into the cap: below the physical ceiling
/// many queries still succeed on Tier-2 today, and lowering the gate
/// would change which executor produces their result. A margin belongs in
/// an auto-tuner with hardware coverage, not in this behaviour-preserving
/// constant.
pub const TIER2_MAX_GROUPS: u32 = 4096 * 1024;

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
    if n_groups > TIER1_MAX_GROUPS && n_groups <= TIER2_MAX_GROUPS && n_rows >= TIER2_MIN_ROWS {
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

    /// The Tier-2 cap must equal the physical reduce-output slot capacity
    /// `NUM_PARTITIONS * BLOCK_GROUPS = 4096 * 1024`. Those constants live
    /// outside this file's editable set
    /// (`partition_kernel::NUM_PARTITIONS` /
    /// `partition_reduce_kernel::BLOCK_GROUPS`), so this test pins the
    /// derivation locally; if either kernel constant ever changes, this
    /// assertion documents the value that must move in lock-step.
    #[test]
    fn tier2_cap_equals_physical_slot_capacity() {
        assert_eq!(TIER2_MAX_GROUPS, 4096 * 1024);
        assert_eq!(TIER2_MAX_GROUPS, 4_194_304);
        // Cross-check against the kernel constants this value is derived
        // from (read-only here — they are authored in `src/jit/`).
        assert_eq!(
            TIER2_MAX_GROUPS,
            crate::jit::partition_kernel::NUM_PARTITIONS
                * crate::jit::partition_reduce_kernel::BLOCK_GROUPS
        );
    }

    /// Exactly at the new cap → still Tier-2 (the bound is inclusive:
    /// `n_groups <= TIER2_MAX_GROUPS`), so a query that *just* fits the
    /// physical slot capacity is not forced off the fast path.
    #[test]
    fn tier2_boundary_at_cap_is_eligible() {
        let inputs = DispatchInputsV2 {
            n_groups: TIER2_MAX_GROUPS,
            ..eligible_baseline()
        };
        assert_eq!(dispatch_v2(inputs), GroupByStrategyV2::Tier2Partitioned);
    }

    /// One distinct key over the physical capacity → spill is guaranteed
    /// by pigeonhole, so dispatch must route straight to GlobalAtomic
    /// rather than running the pipeline to a guaranteed spill + fallback.
    #[test]
    fn tier2_boundary_one_over_cap_falls_back() {
        let inputs = DispatchInputsV2 {
            n_groups: TIER2_MAX_GROUPS + 1,
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
