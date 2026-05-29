// SPDX-License-Identifier: Apache-2.0

//! GROUP BY aggregate execution.
//!
//! Single-pass open-addressing GPU hash table:
//!
//!   1. Host inspects the group-by key column to estimate `K` (the hash-table
//!      size, rounded up to a power of two), validates the key dtype, and
//!      pre-flight scans for collisions with the `EMPTY_KEY = i64::MIN`
//!      sentinel; if any key collides the executor routes to the
//!      sentinel-free [`crate::exec::groupby_valid`] (review C7).
//!   2. Allocate keys table (`length K`, initialised to `EMPTY_KEY`) and one
//!      accumulator table per aggregate (`length K`, initialised to that
//!      aggregate's identity).
//!   3. Upload the key column to the GPU as `i64` (Int32 columns are upcast
//!      host-side).
//!   4. Launch the keys kernel (`bolt_groupby_keys`) — one thread per row.
//!      Each row hashes its key and inserts it into the open-addressing table
//!      via an `atom.cas` linear probe.
//!   5. For each aggregate, upload its input column, JIT + launch
//!      `bolt_groupby_agg` against the already-populated keys table and
//!      that aggregate's accumulator table.
//!   6. Download the keys table and each accumulator table; walk slots,
//!      filtering out empty ones, sort by key for deterministic ordering, and
//!      build the output `RecordBatch` matching `aggregate.output_schema`.
//!
//! Scope (v1):
//!   - GROUP BY keys are encoded host-side into i64 before upload. Supported
//!     packings (all LOSSLESS — distinct tuples yield distinct i64 keys, so
//!     the existing single-i64-key kernel needs no changes):
//!       * 1 col Int32   → upcast to i64 (sign-extended).
//!       * 1 col Int64   → as i64.
//!       * 1 col Float32 → `f32::to_bits() as u32 as i64`.
//!       * 1 col Float64 → `f64::to_bits() as i64`.
//!       * 2 cols (Int32, Int32)     → `(a as u64 << 32) | (b as u32 as u64)`.
//!       * 2 cols (Int32, Float32)   → same packing on the bit patterns.
//!       * 2 cols (Float32, Float32) → same packing on the bit patterns.
//!     Anything wider than 64 bits of key material (e.g. 2× Int64, 2× Float64,
//!     3+ columns) returns a "not yet supported" error. The general fallback
//!     (composite hash + host-side per-slot tuple verification) is deferred.
//!   - Float keys: bitwise-equal floats group together AFTER signed-zero
//!     canonicalisation. `+0.0` and `-0.0` are mapped to the same key
//!     (review C12) via a host-side pre-pass in `load_key_column_bits`,
//!     matching SQL/IEEE semantics and the DISTINCT / JOIN executors.
//!     The kernel itself is not changed — only the i64-encoded buffer
//!     uploaded to the GPU. NaN bit patterns are LEFT AS-IS (NaN != NaN
//!     per IEEE / SQL standard; DuckDB does the same).
//!   - Aggregates: `SUM`, `MIN`, `MAX`, `COUNT(*)`, `AVG`. `MIN`/`MAX` over
//!     float inputs are rejected (would need float-CAS loops on sm_70).
//!   - Aggregate inputs must be bare column references (mirrors
//!     `aggregate.rs`'s scalar path); the host fetches them straight from the
//!     input `RecordBatch`.
//!   - The `pre` kernel (filter / projection feeding the aggregate) is not
//!     yet supported here; the caller should run the scalar path or extend
//!     this module to materialise its outputs first.
//!   - EMPTY_KEY sentinel: a single-Int64 key column whose value equals
//!     `i64::MIN` no longer aborts the query (review C7). Instead, both the
//!     pre-encoding Arrow-array scan and the post-encoding `host_keys` scan
//!     detect the collision and dispatch to
//!     [`crate::exec::groupby_valid::execute_groupby_valid`], which uses
//!     the slot-valid-flag protocol and so reserves no `i64` value as a
//!     sentinel. For multi-column packed keys, a packed value of
//!     `0x8000_0000_0000_0000` (a real composite tuple whose high bit is
//!     set, e.g. `(i32::MIN, 0)`) is caught by the post-encoding scan and
//!     routed the same way.
//!
//! ## PV-stage-d: validity handling
//!
//! Today this executor only enters when all aggregate inputs (and the
//! group-by key) come from a `RecordBatch` with `null_count() == 0` — the
//! orchestrator's dispatcher routes null-bearing batches to
//! [`crate::exec::groupby_valid`] (sentinel-free) or, when the plan
//! includes a pre-projection, to [`crate::exec::groupby_with_pre`] (host
//! strip). The classic kernels in [`crate::jit::hash_kernels`] therefore
//! never need to consume a validity bitmap — every row they see is valid
//! by construction.
//!
//! Stage E may unify this with the native-validity kernels by letting
//! `KernelSpec::input_has_validity` drive dispatch right here; for now
//! the host-side null check at the top of this function (mirrored by
//! `groupby_valid::execute_groupby_valid`) is the safety net.

// dedup (groupby_common): `HashSet` was used only by the now-relocated
// `unique_count`; the shared copy lives in `crate::exec::groupby_common`.
use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{
    DataType as ArrowDataType, Schema as ArrowSchema,
};
use bytemuck::Pod;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::{grid_x_for, CudaStream};
use crate::exec::module_cache;
use crate::exec::n_rows_to_u32;
use crate::jit::agg_kernels::ReduceOp;
// Stage C: the `_with_validity` variants in `crate::jit::hash_kernels`
// (`compile_groupby_agg_kernel_with_validity`,
// `compile_groupby_keys_kernel_with_validity`) are wired up and available;
// today this executor keeps the H1 host-strip-at-call-site pattern, which
// is correct for NULL keys (via `key_valid` from `groupby_valid::pack_keys`)
// and for NULL values (the source path here reads a `RecordBatch` and the
// classic groupby rejects EMPTY_KEY collisions early). Switching to the
// native GPU validity path is a performance follow-up — the host-strip
// remains correct.
use crate::jit::hash_kernels::{
    compile_groupby_agg_kernel, compile_groupby_agg_kernel_with_validity,
    compile_groupby_keys_kernel_dispatched, groupby_block_size,
    AGG_KERNEL_ENTRY, I64_EMPTY_SENTINEL, KEYS_KERNEL_ENTRY,
};
use crate::plan::logical_plan::{
    sum_output_dtype, AggregateExpr, DataType, Expr, Field, Schema,
};
use crate::plan::physical_plan::{AggregateSpec, ColumnIO, PhysicalPlan};

/// Empty-slot sentinel; mirrors the literal baked into the keys kernel.
/// Re-export of [`I64_EMPTY_SENTINEL`] under the legacy local name to keep
/// existing call sites in this module unchanged.
const EMPTY_KEY: i64 = I64_EMPTY_SENTINEL;

/// Review C7: host-side pre-flight scan of an Int64 key column for the
/// classic-kernel empty-slot sentinel ([`I64_EMPTY_SENTINEL`] = `i64::MIN`).
///
/// The non-validity keys kernel ([`compile_groupby_keys_kernel`]) reserves
/// `i64::MIN` as the marker for "this slot is empty" in the open-addressing
/// table. An Int64 input that legitimately contains that value would collide
/// with the marker and the kernel would silently produce wrong results —
/// either dropping the affected row's group entirely or merging it with
/// adjacent groups depending on probe order. This helper lets the Tier-1
/// dispatcher detect that condition and route to the sentinel-free
/// valid-flag executor in [`crate::exec::groupby_valid`].
///
/// Returns `Ok(false)` for any dtype that cannot widen to `i64::MIN` in the
/// kernel: Int32 keys are upcast via sign-extension and `i32::MIN` widens
/// to `-2147483648i64`, which is far inside the safe range. Float keys,
/// Utf8, Bool, and unsupported dtypes also return `Ok(false)` here — the
/// Float path's `-0.0` collision is caught downstream by the
/// post-encoding `host_keys` scan in `execute_groupby`.
///
/// Skips NULL rows (the kernel won't see them anyway: NULL-keyed rows are
/// either pre-filtered upstream or routed through a separate path).
fn key_array_contains_sentinel(arr: &dyn Array) -> BoltResult<bool> {
    if let Some(int64) = arr.as_any().downcast_ref::<Int64Array>() {
        // `iter()` yields `Option<i64>` and naturally skips NULLs only via
        // the `flatten` step below; we want to inspect every non-NULL value.
        for opt in int64.iter() {
            if let Some(v) = opt {
                if v == I64_EMPTY_SENTINEL {
                    return Ok(true);
                }
            }
        }
        return Ok(false);
    }
    // Int32 → i64 widens via sign-extension. i32::MIN as i64 is
    // -2147483648, nowhere near i64::MIN, so no value in an Int32Array can
    // collide with the sentinel after widening. Confirm the cast is the
    // expected one (defensive guard) and return false.
    if arr.as_any().downcast_ref::<Int32Array>().is_some() {
        debug_assert!(
            (i32::MIN as i64) != I64_EMPTY_SENTINEL,
            "Int32 widening invariant violated: i32::MIN as i64 must not equal I64_EMPTY_SENTINEL"
        );
        return Ok(false);
    }
    // Other dtypes (Float32/Float64/Bool/Utf8): the executor's encoding
    // step either handles the collision itself (Float64 -0.0) or rejects
    // the dtype upstream. Pre-flight scan is a no-op here.
    Ok(false)
}

/// PV-stage-e: observability counter — increments once per agg launch this
/// executor routes through a native `_with_validity` kernel path instead
/// of falling through to the legacy host-strip-before-upload pattern.
///
/// `execute_groupby` does not consume a `KernelSpec` (its plan shape has
/// `pre = None`), so the planner-time
/// [`crate::plan::physical_plan::KernelSpec::input_has_validity`] signal
/// is unavailable here. The runtime gate is therefore the source
/// `RecordBatch`'s per-column `null_count()` — see
/// [`column_should_use_native_validity`]. Stage F will plumb the planner
/// signal through `AggregateSpec` so this executor can match
/// `groupby_with_pre`'s plan-time dispatch exactly.
///
/// Used by inline `#[cfg(test)]` tests; production code does not read it.
#[doc(hidden)]
pub static NATIVE_VALIDITY_LAUNCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// PV-stage-f: runtime predicate — should this column's agg launch route
/// through the native `_with_validity` kernel?
///
/// Conditions checked here:
/// 1. The Arrow array's `null_count() > 0` (no NULLs => classic kernel
///    is already correct without any extra bitmap traffic).
/// 2. The `(op, dtype)` combination has a `_with_validity` emitter in
///    [`crate::jit::hash_kernels`]. Integer SUM/MIN/MAX and float SUM
///    are covered; float MIN/MAX goes through `float_atomics` and has
///    no companion yet — host-strip remains the fallback there.
///
/// The caller in `run_typed_agg` additionally gates on the planner's
/// `AggregateSpec::input_has_validity` signal so the runtime cost of
/// inspecting `null_count()` is only paid when the planner believes the
/// underlying table actually carries validity.
fn column_should_use_native_validity(
    arr: &dyn arrow_array::Array,
    op: ReduceOp,
    dtype: DataType,
) -> bool {
    if arr.null_count() == 0 {
        return false;
    }
    // Same coverage as `groupby_with_pre::dispatch_native_validity`: integer
    // SUM/MIN/MAX and float SUM. Float MIN/MAX routes through
    // `float_atomics` which has no `_with_validity` companion yet.
    // Bool/Utf8 are rejected by the kernel; don't dispatch.
    matches!(
        (op, dtype),
        (ReduceOp::Sum, DataType::Int32)
            | (ReduceOp::Sum, DataType::Int64)
            | (ReduceOp::Sum, DataType::Float32)
            | (ReduceOp::Sum, DataType::Float64)
            | (ReduceOp::Min, DataType::Int32)
            | (ReduceOp::Max, DataType::Int32)
            | (ReduceOp::Min, DataType::Int64)
            | (ReduceOp::Max, DataType::Int64)
    )
}

// dedup (groupby_common): the Stage-3 pinned D2H helpers
// (`download_pinned_{i32,i64,f32,f64}`) plus the key-packing types/functions
// (`KeyComponent`, `KeyValue`, `PackedKeys`, `pack_keys`, `decode_key`,
// `load_key_column_bits`, `and_masks`, `key_bit_width`, `canonicalise_f32`,
// `canonicalise_f64`) and the `unique_count` / `next_pow2` scan helpers used
// to be defined locally here and copy-pasted into `groupby_valid.rs` /
// `groupby_with_pre.rs`. They now live in `crate::exec::groupby_common` (see
// that module's header for the drift-bug rationale, incl. V-17 pack_keys).
use crate::exec::groupby_common::{
    decode_key, download_pinned_f32, download_pinned_f64, download_pinned_i32,
    download_pinned_i64, next_pow2, pack_keys, unique_count, KeyComponent, KeyValue,
};

/// Execute a GROUP BY aggregate plan against a host-side `RecordBatch`.
///
/// `plan` must be `PhysicalPlan::Aggregate` with non-empty `group_by`.
/// Supports single-column (Int32/Int64/Float32/Float64) and a limited set of
/// 2-column packings whose combined width fits in 64 bits; see module docs.
pub fn execute_groupby(
    plan: &PhysicalPlan,
    table_batch: &RecordBatch,
) -> BoltResult<RecordBatch> {
    // Layered fast-paths. Each `try_execute` returns `Some(_)` only if the
    // query's shape matches that path's preconditions; misses fall through
    // to the next. See docs/GROUPBY_PERF.md for the policy + cardinality
    // breakdown.
    //
    //   Tier-1 single-SUM      — single Int32 key, ONE SUM(Float64), small n_groups
    //   Tier-1 multi-SUM       — single Int32 key, 1..=4 SUMs(Float64), small n_groups
    //   Tier-1 AVG             — single Int32 key, 1..=4 AVGs(Float64), small n_groups
    //   Tier-2 hash-partitioned — single Int32 key, ONE SUM(Float64), large n_groups
    //   GlobalAtomic (below)   — everything else
    //
    // GB-S2: a fast path may return `Some(Err(_))` where the error is the
    // Tier-2 reduce `partition_reduce spill` sentinel — the open-addressing
    // hash table overflowed MAX_PROBES and dropped rows, so its result is
    // incorrect. That is a *soft miss*, not a hard failure: the
    // global-atomic fallthrough below recomputes correctly. The
    // `try_fast_path!` macro recognises the sentinel (via the orchestrator's
    // `PARTITION_REDUCE_SPILL_PREFIX` const), logs a warning, and continues
    // to the next strategy. Every other `Err` propagates unchanged, and
    // `Ok` returns immediately.
    macro_rules! try_fast_path {
        ($call:expr) => {
            match $call {
                Some(Ok(batch)) => return Ok(batch),
                Some(Err(e)) => {
                    let is_spill = matches!(
                        &e,
                        BoltError::Other(msg)
                            if msg.starts_with(
                                crate::exec::groupby_tier2_orchestrator
                                    ::PARTITION_REDUCE_SPILL_PREFIX
                            )
                    );
                    if is_spill {
                        log::warn!(
                            "execute_groupby: fast path hit MAX_PROBES spill \
                             ({e}); falling through to global-atomic path"
                        );
                    } else {
                        return Err(e);
                    }
                }
                None => {}
            }
        };
    }
    try_fast_path!(crate::exec::groupby_shmem_exec::try_execute(plan, table_batch));
    try_fast_path!(crate::exec::groupby_shmem_multi_exec::try_execute(plan, table_batch));
    try_fast_path!(crate::exec::groupby_shmem_avg_exec::try_execute(plan, table_batch));
    try_fast_path!(crate::exec::groupby_tier2_exec::try_execute(plan, table_batch));
    // Multi-SUM Tier-2: enabled with `MULTI_SUM_MIN_GROUPS = 100_000` floor
    // in the executor itself. Below 100K groups the global-atomic baseline
    // wins (q2 / 10K groups regressed 444 ms → 1.05 s when this path was
    // unconditional); the gate now lets q2 fall through cleanly while
    // capturing future workloads with more groups.
    try_fast_path!(crate::exec::groupby_tier2_multi_exec::try_execute(plan, table_batch));
    // Two-key Tier-2: enabled now that `partition_reduce_kernel_i64`
    // replaces the host-HashMap pass-2 (Tier 2.1 for two-key).
    try_fast_path!(crate::exec::groupby_tier2_twokey_exec::try_execute(plan, table_batch));
    // Two-key MULTI-aggregate Tier-2.1: `SELECT a, b, SUM(v1), SUM(v2)
    // FROM x GROUP BY a, b` — combines i64 partitioning with
    // multi-value reduce. Two-key single-SUM falls through to the line
    // above first.
    try_fast_path!(crate::exec::groupby_tier2_twokey_multi_exec::try_execute(plan, table_batch));
    // AVG-at-Tier-2.1: SUM (via multi-SUM reduce) + COUNT (via count
    // reduce) → divide host-side. High-cardinality AVG over Float64.
    try_fast_path!(crate::exec::groupby_tier2_avg_exec::try_execute(plan, table_batch));
    // Two-key multi-AVG Tier-2.1: `SELECT a, b, AVG(v1), AVG(v2), ...
    // FROM x GROUP BY a, b`. Same shape as the single-key AVG path but
    // with i64-packed (Int32, Int32) keys.
    try_fast_path!(crate::exec::groupby_tier2_twokey_avg_exec::try_execute(plan, table_batch));
    // COUNT(*) at Tier-2.1: high-cardinality `SELECT k, COUNT(*) FROM x
    // GROUP BY k`. Reuses partition + scatter; one COUNT reduce launch.
    try_fast_path!(crate::exec::groupby_tier2_count_exec::try_execute(plan, table_batch));
    // Two-key COUNT(*) Tier-2.1: `SELECT a, b, COUNT(*) FROM x GROUP BY
    // a, b`. Same shape as the single-key COUNT path but with i64-packed
    // (Int32, Int32) keys.
    try_fast_path!(crate::exec::groupby_tier2_twokey_count_exec::try_execute(plan, table_batch));
    // Two-key integer MIN/MAX at Tier-2.1: `SELECT a, b, {MIN,MAX}(v)
    // FROM x GROUP BY a, b` with Int32 / Int64 value column. Routes
    // through partition_reduce_kernel_minmax_i64. Must come before the
    // single-key minmax path so the two-key shape isn't mishandled.
    try_fast_path!(crate::exec::groupby_tier2_twokey_minmax_exec::try_execute(plan, table_batch));
    // Two-key float MIN/MAX at Tier-2.1: same shape, Float64 value
    // column, CAS-loop kernel (partition_reduce_kernel_minmax_float_i64).
    try_fast_path!(crate::exec::groupby_tier2_twokey_minmax_float_exec::try_execute(plan, table_batch));
    // MIN/MAX at Tier-2.1: high-cardinality integer MIN/MAX. Float
    // MIN/MAX is deferred — needs a CAS-loop kernel and no workload
    // demands it yet.
    try_fast_path!(crate::exec::groupby_tier2_minmax_exec::try_execute(plan, table_batch));
    // Float MIN/MAX: routes through partition_reduce_kernel_minmax_float
    // (CAS-loop kernel). Integer MIN/MAX above catches first; this
    // handles the float-value-column path.
    try_fast_path!(crate::exec::groupby_tier2_minmax_float_exec::try_execute(plan, table_batch));
    // Tier-1 COUNT(*): low-cardinality COUNT GROUP BY.
    try_fast_path!(crate::exec::groupby_shmem_count_exec::try_execute(plan, table_batch));
    // Tier-1 MIN/MAX: low-cardinality integer MIN/MAX. Float MIN/MAX
    // is deferred — needs a CAS-loop kernel.
    try_fast_path!(crate::exec::groupby_shmem_minmax_exec::try_execute(plan, table_batch));

    let (pre, aggregate) = match plan {
        PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
        other => {
            return Err(BoltError::Other(format!(
                "execute_groupby: expected Aggregate plan, got {:?}",
                std::mem::discriminant(other)
            )))
        }
    };

    if aggregate.group_by.is_empty() {
        return Err(BoltError::Other(
            "execute_groupby: aggregate has no GROUP BY columns; use execute_aggregate".into(),
        ));
    }
    if pre.is_some() {
        return Err(BoltError::Other(
            "GROUP BY with projection/filter pre-kernel not yet implemented".into(),
        ));
    }

    // Review C7: pre-flight scan each GROUP BY key column for the classic
    // kernel's empty-slot sentinel (`i64::MIN`). If any Int64 input
    // legitimately contains that value the row would silently collide with
    // the marker; route to the sentinel-free valid-flag executor before
    // we waste time encoding + uploading. The post-encoding scan below
    // remains as a safety net for the Float64 `-0.0` case (its bit pattern
    // also encodes to `i64::MIN`).
    for &ord in &aggregate.group_by {
        let io = aggregate.inputs.get(ord).ok_or_else(|| {
            BoltError::Plan(format!(
                "execute_groupby: group_by ordinal {} out of range (only {} inputs)",
                ord,
                aggregate.inputs.len()
            ))
        })?;
        let col_idx = table_batch
            .schema()
            .index_of(&io.name)
            .map_err(|_| {
                BoltError::Plan(format!(
                    "execute_groupby: GROUP BY key '{}' not present in input batch schema",
                    io.name
                ))
            })?;
        let key_array: &dyn Array = table_batch.column(col_idx).as_ref();
        if key_array_contains_sentinel(key_array)? {
            log::warn!(
                "execute_groupby: GROUP BY key '{}' contains i64::MIN \
                 (classic-kernel empty-slot sentinel); routing to \
                 sentinel-free valid-flag executor to preserve correctness",
                io.name
            );
            return crate::exec::groupby_valid::execute_groupby_valid(plan, table_batch);
        }
    }

    // Encode all group-by columns into i64 keys (host-side packing). If the
    // key is too wide to pack losslessly into i64, delegate to the wide-key
    // host-side fallback in `crate::exec::groupby_wide`.
    let packed = match pack_keys(aggregate, table_batch) {
        Ok(p) => p,
        Err(BoltError::Other(msg))
            if msg.contains("> 64 bits") || msg.contains("not yet supported") =>
        {
            return crate::exec::groupby_wide::execute_groupby_wide(plan, table_batch);
        }
        Err(e) => return Err(e),
    };
    let key_components = packed.components;
    let key_valid = packed.key_valid;

    // NULL keys: SQL standard semantics are implementation-defined for whether
    // a NULL key forms its own group. For v1 we drop rows whose key is NULL
    // (matching the simplest behaviour: NULL keys are not a group). The
    // alternative — synthesise an explicit "NULL group" — would require
    // either a reserved sentinel collision check (we already use i64::MIN
    // for that purpose) or a separate code path; the dropped-rows approach
    // is the conservative-and-correct first cut.
    let host_keys: Vec<i64> = match &key_valid {
        Some(mask) => packed
            .keys_i64
            .iter()
            .zip(mask.iter())
            .filter_map(|(k, &keep)| if keep { Some(*k) } else { None })
            .collect(),
        None => packed.keys_i64,
    };

    // Check for EMPTY_KEY sentinel collision. If any encoded key equals
    // i64::MIN (most commonly: a Float64 column containing -0.0, since
    // `(-0.0f64).to_bits() as i64 == i64::MIN`), the sentinel-based classic
    // kernel can't tell that row from an empty slot. Fall back to the
    // sentinel-free valid-flag variant. Review C7 also adds a pre-encoding
    // scan above for the Int64 case; this remains the safety net for
    // encoded packings whose bit pattern only collides post-encoding.
    if host_keys.iter().any(|&k| k == EMPTY_KEY) {
        log::warn!(
            "execute_groupby: encoded GROUP BY key collides with i64::MIN \
             sentinel after packing (likely Float64 -0.0 or a 2-col \
             packing whose high bit is set); routing to valid-flag executor"
        );
        return crate::exec::groupby_valid::execute_groupby_valid(plan, table_batch);
    }

    let n_rows = host_keys.len();

    // Estimate K from the unique-key count (host scan).
    let n_unique = unique_count(&host_keys);
    let k = next_pow2((n_unique.saturating_mul(2)).saturating_add(16)).max(64);
    let k_u32 = u32::try_from(k).map_err(|_| {
        BoltError::Other(format!(
            "GROUP BY hash table size {} exceeds u32::MAX",
            k
        ))
    })?;

    // Stage-3: mint a per-call stream up front so the host->device
    // uploads, kernel launches, and the final D2Hs share a single
    // ordering domain — the driver can then overlap kernel work with
    // any unrelated activity on the NULL stream. Falls back to NULL if
    // stream creation fails (functionally identical, just no overlap).
    let stream = CudaStream::null_or_default();

    // Build the keys table on the host (filled with EMPTY_KEY) and upload it.
    //
    // Stage-3: H2D upload is async on `stream`. The keys-kernel launch
    // (queued on the same stream below) depends on this copy, so the
    // kernel is automatically ordered after the upload. We do NOT
    // pinned-source these uploads — they're one-shot at executor entry
    // and the pinned pool would only see ~k * 8 bytes of churn, which
    // isn't worth the host-side allocator pressure for a single query.
    let host_keys_init: Vec<i64> = vec![EMPTY_KEY; k];
    let mut keys_table = GpuVec::<i64>::from_slice_async(&host_keys_init, stream.raw())?;
    let key_col_gpu = GpuVec::<i64>::from_slice_async(&host_keys, stream.raw())?;

    // Launch the keys-only kernel.
    launch_keys_kernel(&key_col_gpu, &mut keys_table, n_rows, k_u32, &stream)?;

    // For each aggregate, prepare its accumulator and launch the agg kernel.
    // We collect (input_dtype_for_acc, downloaded acc vector as a typed enum)
    // per aggregate so that the host-side post-processing knows what to read.
    //
    // PV-stage-f: read the planner-time validity signal off the
    // `AggregateSpec`. If ANY agg-input column was flagged by
    // `populate_aggregate_spec`, the per-aggregate dispatch below uses this
    // together with the runtime per-column null check to decide between the
    // native `_with_validity` kernel and the host-strip fallback. An empty
    // flag vector (legacy default) collapses to `false` so existing
    // construction sites remain bit-identical.
    let any_input_has_validity: bool =
        aggregate.input_has_validity.iter().any(|&v| v);
    // PERF L3 (fused multi-agg): when `aggregate.aggregates.len() > 1`, the
    // keys are shared across every aggregate, so the N per-agg launches below
    // re-hash + re-probe the key column N times. The fused kernel hashes the
    // keys ONCE and issues N atomic updates back-to-back, which would replace
    // the loop. The eligible shape is: N>1, no float MIN/MAX (those still need
    // the `float_atomics` CAS path), no `_with_validity` plumbing (the fused
    // emitter explicitly does NOT emit the Stage-C validity gate — see its doc
    // comment), and only fixed-width atomic-compatible dtypes.
    //
    // STATUS — dispatch deliberately NOT flipped. Audit notwithstanding, only
    // the *PTX emitter* is shipped, not a callable executor. Verified by grep:
    // `crate::jit::hash_kernels::compile_groupby_agg_kernel_multi` /
    // `AGG_KERNEL_MULTI_ENTRY` ("bolt_groupby_agg_multi") have NO production
    // caller — they are referenced only by their own definition, doc comments,
    // and three PTX-string unit tests in `hash_kernels.rs`. The host-side
    // launch/download driver does not exist anywhere in the tree.
    //
    // What is MISSING to flip the dispatch (cannot be done in `launch.rs` /
    // `module_cache` shims — needs a NEW fused launcher, which would have to be
    // authored here in groupby.rs since this is the only file we may touch):
    //   (a) build `&[AggSpec]` in canonical agg order, mapping each
    //       AggregateExpr to its (op, input_dtype) — including SUM(i32)->i64
    //       widening and COUNT(*)->i64, exactly as `run_typed_agg` does today;
    //   (b) allocate N accumulators of HETEROGENEOUS widths (i32/i64/f32/f64)
    //       initialised to each op's identity, and upload N input columns of
    //       mixed dtype — the current `launch_agg_kernel<T: Pod>` is monomorphic
    //       in a single element type `T` and cannot express this;
    //   (c) marshal a RUNTIME-LENGTH param array of `2 + 2*N + 2` entries
    //       (group_ptr, keys_ptr, N input_ptrs, N acc_ptrs, n_rows, k) into
    //       `cuLaunchKernel` — the existing launcher uses a fixed `[_; 6]`/`[_; 7]`,
    //       and the N owning typed `GpuVec`s must be kept alive across the call;
    //   (d) download N heterogeneous accumulators back into `AccDownload`.
    // All of (a)-(d) is new `unsafe` FFI glue with a hand-marshalled variadic
    // param list that CANNOT be runtime-verified on this host. Routing live
    // multi-agg queries onto unverified launch glue is a correctness hazard, so
    // the always-correct N-launch path below is retained until a fused launcher
    // lands with hardware coverage. The per-agg loop already preserves the
    // `try_fast_path!` soft-miss / global-atomic semantics established above.
    let mut acc_results: Vec<AccDownload> = Vec::with_capacity(aggregate.aggregates.len());
    for agg in &aggregate.aggregates {
        let acc = run_one_aggregate(
            agg,
            &aggregate.inputs,
            &key_col_gpu,
            &keys_table,
            table_batch,
            n_rows,
            k,
            k_u32,
            &stream,
            key_valid.as_deref(),
            any_input_has_validity,
        )?;
        acc_results.push(acc);
    }

    // Stage-3: download the keys table through a pinned host buffer
    // so the driver can DMA directly. Sync once on this stream, then
    // hand the data off to host-side group assembly.
    let host_keys_table: Vec<i64> = {
        let pinned = keys_table.to_pinned_async(stream.raw())?;
        stream.synchronize()?;
        pinned.as_slice().to_vec()
    };
    drop(keys_table);
    drop(key_col_gpu);

    // Walk the keys table: every non-empty slot is a group. Build a list of
    // `(key, slot)` and sort by key for deterministic output ordering.
    let mut groups: Vec<(i64, usize)> = host_keys_table
        .iter()
        .enumerate()
        .filter_map(|(slot, &k)| if k == EMPTY_KEY { None } else { Some((k, slot)) })
        .collect();
    groups.sort_unstable_by_key(|(k, _)| *k);

    // Assemble the output RecordBatch.
    let n_groups = groups.len();
    let m_keys = key_components.len();
    let mut arrays: Vec<ArrayRef> =
        Vec::with_capacity(m_keys + aggregate.aggregates.len());

    // Columns 0..M: one per group-by column, decoded from the packed i64 key.
    let key_arrays = build_key_arrays(&groups, &key_components)?;
    for arr in key_arrays {
        arrays.push(arr);
    }

    // Columns M..M+N: one per aggregate, taken from the corresponding
    // accumulator.
    for (i, agg) in aggregate.aggregates.iter().enumerate() {
        let out_field =
            aggregate.output_schema.fields.get(m_keys + i).ok_or_else(|| {
                BoltError::Other(format!(
                    "execute_groupby: output_schema missing field for aggregate index {}",
                    i
                ))
            })?;
        let arr = build_agg_array(agg, out_field, &acc_results[i], &groups, n_groups)?;
        arrays.push(arr);
    }

    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(arrow_schema, arrays).map_err(|e| {
        BoltError::Other(format!("failed to build GROUP BY RecordBatch: {e}"))
    })
}

// ---------------------------------------------------------------------------
// Key column extraction, packing, and decoding.
//
// dedup (groupby_common): `key_bit_width`, `KeyComponent`, `PackedKeys`,
// `load_key_column_bits`, `canonicalise_f32`, `canonicalise_f64`, `pack_keys`,
// `and_masks`, `decode_key`, `KeyValue`, `unique_count`, and `next_pow2` were
// formerly defined here and duplicated into `groupby_valid.rs` /
// `groupby_with_pre.rs`. They are now the single canonical copies in
// `crate::exec::groupby_common` (imported at the top of this module). The
// canonical `pack_keys` keeps the V-17 `wrapping_shl` hardening — see that
// module for the full drift-bug history. Call sites below are unchanged.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Kernel launches.
// ---------------------------------------------------------------------------

/// Launch the keys-only kernel.
fn launch_keys_kernel(
    group_col: &GpuVec<i64>,
    keys_table: &mut GpuVec<i64>,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_rows == 0 {
        // Nothing to insert; the empty keys table is already correct.
        return Ok(());
    }

    // Dispatch to classic linear-probe (default) or Robin Hood (when
    // BOLT_HASH_ALGO=robin_hood). Both kernels share the same 4-param ABI;
    // only the entry-point name varies. Route through the consolidated
    // module cache so the two variants are cached separately by spec id.
    let want_rh = std::env::var("BOLT_HASH_ALGO")
        .map(|s| {
            let l = s.to_ascii_lowercase();
            l == "robin_hood" || l == "rh"
        })
        .unwrap_or(false);
    let (spec_id, kernel_entry) = if want_rh {
        ("groupby_keys_rh", crate::jit::hash_kernels::KEYS_KERNEL_RH_ENTRY)
    } else {
        ("groupby_keys", KEYS_KERNEL_ENTRY)
    };
    let module = module_cache::get_or_build_module(
        module_path!(),
        spec_id.to_string(),
        None,
        || {
            let (ptx, _e) = compile_groupby_keys_kernel_dispatched()?;
            Ok(ptx)
        },
    )?;
    let function = module.function(kernel_entry)?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;

    let mut params: [*mut c_void; 4] = [
        &mut group_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut k_param as *mut u32 as *mut c_void,
    ];

    let block = groupby_block_size();
    let grid_x = grid_x_for(n_rows_u32, block);

    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x,
            1,
            1,
            block,
            1,
            1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    stream.synchronize()?;
    let _ = (group_ptr, keys_ptr);
    Ok(())
}

/// Launch one aggregate-update kernel for a (typed) input column and a
/// (typed) accumulator buffer.
///
/// PV-stage-f: when `validity_ptr` is `Some(_)`, route through the native
/// `_with_validity` PTX emitter
/// ([`compile_groupby_agg_kernel_with_validity`]) — the kernel reads the
/// packed-bit validity bitmap and skips NULL rows on the device, leaving
/// the classic 6-param ABI alone for the no-validity case. The pointer
/// must reference a `Vec<u32>` packed via
/// [`crate::exec::groupby_with_pre::pack_validity_bits`] (little-endian
/// bit `tid % 32` of word `tid / 32` per the
/// [`crate::jit::hash_kernels`] module-level docs).
///
/// Float MIN/MAX still routes to `float_atomics` even when validity is
/// requested — that emitter has no `_with_validity` companion yet
/// (Stage G follow-up); the caller's
/// [`column_should_use_native_validity`] gate already drops that combo.
#[allow(clippy::too_many_arguments)]
fn launch_agg_kernel<T: Pod>(
    op: ReduceOp,
    input_dtype: DataType,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    input_col: &GpuVec<T>,
    acc_table: &mut GpuVec<T>,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
    validity_ptr: Option<CUdeviceptr>,
) -> BoltResult<()> {
    if n_rows == 0 {
        return Ok(());
    }

    // MIN/MAX over floats need a CAS loop (no native atom.min.fXX on sm_70);
    // route those to the float_atomics codegen. Everything else uses the
    // standard integer-atomic agg kernel. The native-validity variant is
    // only available for the integer-atomic kernel today — float MIN/MAX
    // through `float_atomics` has no `_with_validity` companion, so a
    // caller passing `validity_ptr = Some(_)` for that combo is a bug
    // (the dispatch predicate filters it out).
    let use_validity = validity_ptr.is_some();
    let is_float_min_max = matches!(
        (op, input_dtype),
        (ReduceOp::Min, DataType::Float32)
            | (ReduceOp::Max, DataType::Float32)
            | (ReduceOp::Min, DataType::Float64)
            | (ReduceOp::Max, DataType::Float64)
    );
    debug_assert!(
        !(is_float_min_max && use_validity),
        "float MIN/MAX has no _with_validity emitter — \
         dispatch predicate should have rejected this combo"
    );
    let module = module_cache::get_or_build_module(
        module_path!(),
        format!(
            "groupby_agg:{:?}:{:?}:float_min_max={}:validity={}",
            op, input_dtype, is_float_min_max, use_validity
        ),
        None,
        || {
            if is_float_min_max {
                crate::jit::float_atomics::compile_groupby_float_atomic_kernel(op, input_dtype)
            } else if use_validity {
                compile_groupby_agg_kernel_with_validity(op, input_dtype)
            } else {
                compile_groupby_agg_kernel(op, input_dtype)
            }
        },
    )?;
    let function = module.function(AGG_KERNEL_ENTRY)?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut input_ptr: CUdeviceptr = input_col.device_ptr();
    let mut acc_ptr: CUdeviceptr = acc_table.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;

    let block = groupby_block_size();
    let grid_x = grid_x_for(n_rows_u32, block);

    if let Some(vp) = validity_ptr {
        // Account the native-dispatch launch for inline-test observability.
        NATIVE_VALIDITY_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut vptr: CUdeviceptr = vp;
        let mut params: [*mut c_void; 7] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut input_ptr as *mut CUdeviceptr as *mut c_void,
            &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
            &mut k_param as *mut u32 as *mut c_void,
            &mut vptr as *mut CUdeviceptr as *mut c_void,
        ];
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                function.raw(),
                grid_x,
                1,
                1,
                block,
                1,
                1,
                0,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
        let _ = vptr;
    } else {
        let mut params: [*mut c_void; 6] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut input_ptr as *mut CUdeviceptr as *mut c_void,
            &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
            &mut k_param as *mut u32 as *mut c_void,
        ];
        unsafe {
            cuda_sys::check(cuda_sys::cuLaunchKernel(
                function.raw(),
                grid_x,
                1,
                1,
                block,
                1,
                1,
                0,
                stream.raw(),
                params.as_mut_ptr(),
                ptr::null_mut(),
            ))?;
        }
    }
    stream.synchronize()?;
    let _ = (group_ptr, keys_ptr, input_ptr, acc_ptr);
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-aggregate plumbing: prepare buffer, launch kernel(s), download result.
// ---------------------------------------------------------------------------

/// Downloaded accumulator table for a single aggregate. The dtype variant
/// tells the host-side assembler how to read each slot.
enum AccDownload {
    /// Int32 SUM/MIN/MAX result column (length `k`).
    I32(Vec<i32>),
    /// Int64 SUM/MIN/MAX/COUNT result column (length `k`).
    I64(Vec<i64>),
    /// Float32 SUM result column (length `k`).
    F32(Vec<f32>),
    /// Float64 SUM result column (length `k`).
    F64(Vec<f64>),
    /// AVG: a SUM accumulator (downloaded as f64) and a COUNT accumulator
    /// (downloaded as i64), both length `k`.
    Avg { sum: Vec<f64>, count: Vec<i64> },
    /// VAR_POP / VAR_SAMP / STDDEV_POP / STDDEV_SAMP per-group Welford
    /// `(count, mean, M2)` states. `states[slot]` is the running state for
    /// the group occupying that slot of the hash table; empty slots have
    /// `WelfordState::empty()`. Length `k`.
    ///
    /// Per-group accumulation is computed on the host: the GPU keys kernel
    /// is responsible for hash-table layout, but the (count, mean, M2)
    /// triple does not fit a single atomic; emitting a CAS-loop kernel over
    /// a packed 24-byte slot is a v0.7+ follow-up. The host pass walks one
    /// value column with one slot lookup per row and folds via
    /// [`crate::exec::welford::WelfordState::push`]. Numerically stable by
    /// construction (single-pass Welford, no `sum_sq - n*mean^2`
    /// cancellation).
    Welford { states: Vec<crate::exec::welford::WelfordState> },
}

/// Compile + launch one aggregate kernel (or, for `Avg`, two), download its
/// accumulator table(s), and return the result.
///
/// `key_valid` is the per-row keep mask produced by `pack_keys` over the
/// ORIGINAL (pre-filter) row indices — `None` means no key column has nulls.
/// When the value column itself has NULLs we logically AND the masks and
/// upload a fresh, per-aggregate filtered key column so that the GPU sees
/// only (non-NULL key, non-NULL value) pairs. `n_rows` is the post-key-filter
/// row count: it equals `group_col`'s length when we end up reusing the
/// shared key column.
#[allow(clippy::too_many_arguments)]
fn run_one_aggregate(
    agg: &AggregateExpr,
    inputs: &[ColumnIO],
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    batch: &RecordBatch,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    stream: &CudaStream,
    key_valid: Option<&[bool]>,
    any_input_has_validity: bool,
) -> BoltResult<AccDownload> {
    match agg {
        AggregateExpr::Sum(expr)
        | AggregateExpr::Min(expr)
        | AggregateExpr::Max(expr) => {
            let op = ReduceOp::from_agg(agg)?;
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;
            run_typed_agg(
                op, col_io, group_col, keys_table, batch, n_rows, k, k_u32, stream,
                key_valid, any_input_has_validity,
            )
        }

        AggregateExpr::Count(expr) => {
            // COUNT(col) excludes NULL inputs; COUNT(*) (an expression that
            // doesn't resolve to a column) counts surviving (post-key-filter)
            // rows. Either way we synthesise an all-ones column over the
            // post-filter rows; the only difference is whether the value-NULL
            // mask is applied. Stage-3: async H2D + pinned D2H.
            let value_valid: Option<Vec<bool>> = match bare_column_name(expr)
                .ok()
                .and_then(|name| resolve_input(inputs, name).ok())
            {
                Some(col_io) => column_null_mask(col_io, batch)?,
                None => None,
            };

            let filtered = prepare_filtered_keys(group_col, n_rows, key_valid, value_valid.as_deref(), stream)?;
            let count_n_rows = filtered.n_rows();

            let ones: Vec<i64> = vec![1i64; count_n_rows];
            let input_gpu = GpuVec::<i64>::from_slice_async(&ones, stream.raw())?;
            let identity_init: Vec<i64> = vec![0i64; k];
            let mut acc_table = GpuVec::<i64>::from_slice_async(&identity_init, stream.raw())?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                filtered.col(),
                keys_table,
                &input_gpu,
                &mut acc_table,
                count_n_rows,
                k_u32,
                stream,
                None,
            )?;
            Ok(AccDownload::I64(download_pinned_i64(&acc_table, stream)?))
        }

        AggregateExpr::VarPop(expr)
        | AggregateExpr::VarSamp(expr)
        | AggregateExpr::StddevPop(expr)
        | AggregateExpr::StddevSamp(expr) => {
            // v0.7: per-group Welford. The GPU hash-aggregate kernel can
            // only atomically update a single scalar slot per group, but
            // numerically-stable Welford requires the coupled
            // `(count, mean, M2)` triple to evolve together. We sidestep
            // a packed-24-byte CAS-loop PTX kernel (correct but ~order of
            // magnitude slower than the scalar atomics, and tricky to get
            // right under contention) and fold the per-group state on the
            // host. The host loop runs after the GPU keys kernel has
            // populated the slot table, so the slot lookup is a single
            // hash + linear-probe per row.
            let col_name = bare_column_name(expr.as_ref())?;
            let col_io = resolve_input(inputs, col_name)?;
            run_welford_aggregate(
                col_io,
                group_col,
                keys_table,
                batch,
                n_rows,
                k,
                k_u32,
                stream,
                key_valid,
            )
        }

        AggregateExpr::Avg(expr) => {
            // AVG = SUM(expr) / COUNT(expr), where COUNT is the non-NULL row
            // count of the value column within each group. SUM in f64 (so we
            // don't worry about int-overflow during accumulation), COUNT in
            // i64. Both kernels share the (key_valid ∧ value_valid) filter so
            // every contribution to the SUM increments the matching COUNT.
            //
            // Stage-3: async H2D for the input + identity tables; pinned
            // D2H for both final accumulators.
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;

            let value_valid = column_null_mask(col_io, batch)?;
            let filtered = prepare_filtered_keys(group_col, n_rows, key_valid, value_valid.as_deref(), stream)?;
            let avg_n_rows = filtered.n_rows();

            // --- SUM(expr) cast to f64. We upcast the input host-side and
            //     drop NULL positions in the same step. ---
            let sum_input: Vec<f64> = load_input_column_as_f64_filtered(
                col_io,
                batch,
                key_valid,
                value_valid.as_deref(),
            )?;
            debug_assert_eq!(sum_input.len(), avg_n_rows);
            let input_gpu = GpuVec::<f64>::from_slice_async(&sum_input, stream.raw())?;
            let sum_init: Vec<f64> = vec![0.0f64; k];
            let mut sum_acc = GpuVec::<f64>::from_slice_async(&sum_init, stream.raw())?;
            launch_agg_kernel::<f64>(
                ReduceOp::Sum,
                DataType::Float64,
                filtered.col(),
                keys_table,
                &input_gpu,
                &mut sum_acc,
                avg_n_rows,
                k_u32,
                stream,
                None,
            )?;
            let sum_host = download_pinned_f64(&sum_acc, stream)?;

            // --- COUNT(non-null) per group. ---
            let ones: Vec<i64> = vec![1i64; avg_n_rows];
            let count_input = GpuVec::<i64>::from_slice_async(&ones, stream.raw())?;
            let count_init: Vec<i64> = vec![0i64; k];
            let mut count_acc = GpuVec::<i64>::from_slice_async(&count_init, stream.raw())?;
            launch_agg_kernel::<i64>(
                ReduceOp::Sum,
                DataType::Int64,
                filtered.col(),
                keys_table,
                &count_input,
                &mut count_acc,
                avg_n_rows,
                k_u32,
                stream,
                None,
            )?;
            let count_host = download_pinned_i64(&count_acc, stream)?;

            Ok(AccDownload::Avg {
                sum: sum_host,
                count: count_host,
            })
        }
    }
}

/// Per-group Welford (`count`, `mean`, `M2`) accumulation for VAR_POP /
/// VAR_SAMP / STDDEV_POP / STDDEV_SAMP under GROUP BY.
///
/// The GPU side has already populated `keys_table` (one i64 per slot,
/// `EMPTY_KEY` for unused slots) via the keys kernel. This pass:
///   1. downloads `keys_table` to the host,
///   2. builds a `HashMap<i64, usize>` mapping key -> slot,
///   3. downloads the i64-encoded per-row keys (`group_col`) to the host
///      (it was uploaded from the same `host_keys` the executor built),
///   4. reads the value column from the input batch, dropping NULL rows,
///   5. folds each (key, value) pair into `states[slot]` via the canonical
///      [`crate::exec::welford::WelfordState::push`] update.
///
/// The result is a per-slot `Vec<WelfordState>` (length `k`); the assembler
/// indexes by slot just like the SUM/MIN/MAX/AVG paths.
///
/// Why host-side: the Welford update requires the coupled
/// `(count, mean, M2)` triple to advance together (otherwise `mean` and
/// `M2` desynchronise and the variance is wrong). The existing
/// hash-aggregate kernel emits one `atom.global.<op>.<dtype>` per row
/// against a single scalar accumulator slot — fine for SUM, AVG (a SUM
/// + a COUNT in separate buffers), MIN, MAX, COUNT, but not for the
/// 24-byte Welford state. A correct CAS-loop kernel over a packed
/// `(u64, f64, f64)` slot is the natural follow-up; until then the host
/// fold is correct and numerically stable (single-pass Welford, no
/// `sum_sq - n*mean^2` cancellation).
#[allow(clippy::too_many_arguments)]
fn run_welford_aggregate(
    col_io: &ColumnIO,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    batch: &RecordBatch,
    n_rows: usize,
    k: usize,
    _k_u32: u32,
    stream: &CudaStream,
    key_valid: Option<&[bool]>,
) -> BoltResult<AccDownload> {
    // Pull the keys table host-side. The keys kernel has already filled it
    // (the per-call stream synchronises on every launch — see
    // `launch_keys_kernel` / `launch_agg_kernel` — so a plain `to_vec` is
    // safe here without an extra explicit sync).
    let host_keys_table: Vec<i64> = keys_table.to_vec()?;
    debug_assert_eq!(host_keys_table.len(), k);

    // Build slot -> key mapping by walking the table; convert to key -> slot
    // for the per-row lookup. EMPTY_KEY slots are unused and excluded.
    let mut key_to_slot: std::collections::HashMap<i64, usize> =
        std::collections::HashMap::with_capacity(host_keys_table.len().min(1 << 20));
    for (slot, &k_val) in host_keys_table.iter().enumerate() {
        if k_val != EMPTY_KEY {
            key_to_slot.insert(k_val, slot);
        }
    }

    // Per-slot accumulator, initialised to the Welford identity (empty
    // state). `states[slot]` is `WelfordState::empty()` for every empty
    // slot and stays so — `finalize_welford_array` reads the count and
    // emits SQL NULL when count == 0 (or count <= 1 for VAR_SAMP /
    // STDDEV_SAMP), so empty slots produce nullable output naturally.
    let mut states: Vec<crate::exec::welford::WelfordState> =
        vec![crate::exec::welford::WelfordState::empty(); k];

    // Pull the per-row i64-encoded keys back to the host so we can map each
    // row to its slot. These are the same keys the keys kernel saw.
    let host_keys: Vec<i64> = group_col.to_vec()?;
    debug_assert_eq!(host_keys.len(), n_rows);
    // Belt-and-braces stream sync in case the upload of `group_col` is
    // still in-flight (`from_slice_async` queues a copy on the stream;
    // `to_vec` reads via cuMemcpyDtoH which is synchronous on the NULL
    // stream but not necessarily ordered against the per-call stream
    // we minted in `execute_groupby`).
    stream.synchronize()?;

    // Build per-row value pulls + NULL filter. We always promote to f64
    // because the Welford state is f64; matches the scalar (no GROUP BY)
    // path in `aggregate.rs`. The `key_valid` filter has already been
    // applied to `host_keys` (the executor builds it from
    // `packed.keys_i64` AND'd with `key_valid` upstream), so we only
    // need to drop value-NULL rows here.
    //
    // NOTE: `host_keys.len() == n_rows` is the POST-key-filter row count.
    // To align the value column's rows with `host_keys`, we have to apply
    // the same `key_valid` mask to the raw value column BEFORE indexing.
    // The shared helper `load_input_column_as_f64_filtered` does exactly
    // that (drops rows whose key is NULL AND/OR whose value is NULL).
    let value_valid = column_null_mask(col_io, batch)?;
    let values_f64 = load_input_column_as_f64_filtered(
        col_io,
        batch,
        key_valid,
        value_valid.as_deref(),
    )?;

    // After load_input_column_as_f64_filtered, `values_f64.len()` is the
    // count of rows that survive BOTH key_valid AND value_valid. To match
    // each surviving value to its host_keys entry we re-project the
    // value-valid mask through key_valid (the same shape used by
    // prepare_filtered_keys above). When `value_valid` is None every key
    // row survives and we can zip directly.
    let value_valid_filtered: Option<Vec<bool>> = match value_valid.as_deref() {
        None => None,
        Some(v) => Some(match key_valid {
            Some(kv) => kv
                .iter()
                .zip(v.iter())
                .filter_map(|(&kk, &vv)| if kk { Some(vv) } else { None })
                .collect(),
            None => v.to_vec(),
        }),
    };

    // Fold rows into per-slot Welford states. The two row pointers are
    // `idx_keys` (walks `host_keys`, always one per surviving key row) and
    // `idx_vals` (walks `values_f64`, only the value-NULL-free subset).
    let mut idx_vals: usize = 0;
    match value_valid_filtered {
        None => {
            // Common fast path: no value-NULLs, lengths match.
            debug_assert_eq!(values_f64.len(), host_keys.len());
            for (i, &key) in host_keys.iter().enumerate() {
                if let Some(&slot) = key_to_slot.get(&key) {
                    states[slot].push(values_f64[i]);
                }
            }
        }
        Some(vv) => {
            debug_assert_eq!(vv.len(), host_keys.len());
            for (i, &key) in host_keys.iter().enumerate() {
                if !vv[i] {
                    continue;
                }
                if let Some(&slot) = key_to_slot.get(&key) {
                    states[slot].push(values_f64[idx_vals]);
                    idx_vals += 1;
                } else {
                    // Key didn't make it into the table (shouldn't happen at
                    // load factor < 0.5 — but be defensive). Still consume
                    // the value to keep idx_vals aligned.
                    idx_vals += 1;
                }
            }
        }
    }

    Ok(AccDownload::Welford { states })
}

/// Return the per-row validity mask for `col_io` in `batch`, or `None` if
/// the column has no nulls (saves the per-row allocation in the hot path).
fn column_null_mask(
    col_io: &ColumnIO,
    batch: &RecordBatch,
) -> BoltResult<Option<Vec<bool>>> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);
    if arr.null_count() == 0 {
        Ok(None)
    } else {
        Ok(Some((0..arr.len()).map(|i| !arr.is_null(i)).collect()))
    }
}

/// A filtered key column for a single aggregate launch: either a borrowed
/// view of the shared `group_col` (the common fast path: no value-NULLs
/// shrunk the row set further) or a freshly-uploaded smaller column for the
/// (key_valid AND value_valid) joint mask. The variants paper over Rust's
/// borrow-checker constraints around returning `&GpuVec<i64>` whose lifetime
/// might come from local storage.
enum FilteredKeys<'a> {
    /// Reuse the shared post-key-filter key column.
    Borrowed { group_col: &'a GpuVec<i64>, n_rows: usize },
    /// Freshly-uploaded smaller column applying a value-NULL filter on top
    /// of `key_valid`. The owned vec must live across the kernel launch.
    Owned { group_col: GpuVec<i64>, n_rows: usize },
}

impl<'a> FilteredKeys<'a> {
    fn col(&self) -> &GpuVec<i64> {
        match self {
            FilteredKeys::Borrowed { group_col, .. } => *group_col,
            FilteredKeys::Owned { group_col, .. } => group_col,
        }
    }
    fn n_rows(&self) -> usize {
        match self {
            FilteredKeys::Borrowed { n_rows, .. } | FilteredKeys::Owned { n_rows, .. } => *n_rows,
        }
    }
}

/// Decide whether to reuse the shared `group_col` (when no value-NULL filter
/// shrinks the row set further) or upload a freshly-filtered key column for
/// this aggregate. The shared `group_col` was built from a `host_keys` that
/// already had the `key_valid` rows kept; if `value_valid` is `None` we can
/// reuse it directly. Otherwise we re-download once and refilter against
/// the joint mask, then upload a fresh i64 column.
///
/// v0.7 async-memcpy: the DtoH of the shared key column and the HtoD of the
/// freshly-filtered keys both ride `stream` via the pinned-D2H +
/// `from_slice_async` pair, mirroring the per-aggregate buffer plumbing in
/// `run_typed_agg`. This keeps the entire per-aggregate launch on a single
/// ordering domain so the driver can chain the upload behind the previous
/// kernel's D2H and overlap with unrelated stream activity.
fn prepare_filtered_keys<'a>(
    group_col: &'a GpuVec<i64>,
    n_rows: usize,
    key_valid: Option<&[bool]>,
    value_valid: Option<&[bool]>,
    stream: &CudaStream,
) -> BoltResult<FilteredKeys<'a>> {
    if value_valid.is_none() {
        return Ok(FilteredKeys::Borrowed { group_col, n_rows });
    }

    // Project `value_valid` (indexed by ORIGINAL row position) through the
    // key filter to align with the post-key-filter `host_keys` list.
    let value_valid_unwrapped = value_valid.expect("checked above");
    let value_valid_filtered: Vec<bool> = match key_valid {
        Some(kv) => kv
            .iter()
            .zip(value_valid_unwrapped.iter())
            .filter_map(|(&k, &v)| if k { Some(v) } else { None })
            .collect(),
        None => value_valid_unwrapped.to_vec(),
    };
    debug_assert_eq!(value_valid_filtered.len(), n_rows);

    // v0.7 async DtoH of the shared key column via the pinned-buffer
    // helper. The pinned hop lets the driver DMA into host memory without
    // staging through a bounce buffer; the trailing `to_vec()` is the one
    // unavoidable host-host copy. Routes through `download_pinned_i64` so
    // the per-aggregate plumbing shares one D2H code path.
    let host_keys: Vec<i64> = download_pinned_i64(group_col, stream)?;
    debug_assert_eq!(host_keys.len(), n_rows);
    let filtered: Vec<i64> = host_keys
        .iter()
        .zip(value_valid_filtered.iter())
        .filter_map(|(&k, &v)| if v { Some(k) } else { None })
        .collect();
    let filtered_n = filtered.len();
    // v0.7 async HtoD of the freshly-filtered key column on the caller's
    // stream so the subsequent agg-kernel launch is automatically ordered
    // after this upload without a separate barrier. Replaces the prior
    // synchronous `GpuVec::from_slice` here.
    let owned = GpuVec::<i64>::from_slice_async(&filtered, stream.raw())?;
    Ok(FilteredKeys::Owned { group_col: owned, n_rows: filtered_n })
}

/// Common path for SUM/MIN/MAX. Uploads the typed input column, allocates a
/// typed accumulator initialised to the op's identity, launches the agg
/// kernel, and downloads the accumulator.
///
/// `key_valid` (from `pack_keys`) and the value column's own validity mask
/// are AND'd together: rows where EITHER is NULL are dropped before upload.
/// This matches the standard SQL semantics — NULL inputs are skipped by
/// SUM/MIN/MAX rather than coerced to 0 / dtype-min / dtype-max (which is
/// what reading the raw `.values()` buffer at NULL positions would do).
#[allow(clippy::too_many_arguments)]
fn run_typed_agg(
    op: ReduceOp,
    col_io: &ColumnIO,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    batch: &RecordBatch,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    stream: &CudaStream,
    key_valid: Option<&[bool]>,
    any_input_has_validity: bool,
) -> BoltResult<AccDownload> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "aggregate input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);
    let arr_dtype = arrow_dtype_to_plan(arr.data_type())?;
    if arr_dtype != col_io.dtype {
        return Err(BoltError::Type(format!(
            "aggregate input '{}' dtype mismatch: plan says {:?}, batch has {:?}",
            col_io.name, col_io.dtype, arr_dtype
        )));
    }

    let value_valid = column_null_mask(col_io, batch)?;

    // PV-stage-f: native-validity dispatch. When the planner flagged this
    // input AND the column actually carries nulls AND the (op, dtype)
    // combination has a `_with_validity` emitter, route through the GPU
    // bitmap path. Falls through to the legacy host-strip otherwise.
    // Float MIN/MAX is explicitly excluded — `float_atomics` has no
    // `_with_validity` companion yet.
    if any_input_has_validity
        && column_should_use_native_validity(arr.as_ref(), op, col_io.dtype)
        && key_valid.is_none()
    {
        // `key_valid.is_none()`: the native kernel reads rows by index, so
        // its parallel keys array must be the same length as the value
        // column. When `pack_keys` already dropped NULL-key rows the
        // `group_col` we receive is shorter than the source batch, breaking
        // the index alignment. Fall back to the host-strip path in that
        // case — Stage G can lift this once the kernel accepts a
        // key-filtered companion.
        let vv = value_valid.as_deref().expect(
            "column_should_use_native_validity guarantees arr.null_count() > 0, \
             so column_null_mask returned Some(_)",
        );
        return run_typed_agg_native_validity(
            op,
            col_io,
            arr.as_ref(),
            vv,
            group_col,
            keys_table,
            n_rows,
            k,
            k_u32,
            stream,
        );
    }

    let filtered = prepare_filtered_keys(group_col, n_rows, key_valid, value_valid.as_deref(), stream)?;
    let n = filtered.n_rows();

    match col_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int32"))?;

            // SUM(Int32) widens to Int64 per the single-source-of-truth
            // `crate::plan::logical_plan::sum_output_dtype`: silent i32
            // overflow inside the GPU atomic was the prior bug. The
            // groupby agg kernel (`compile_groupby_agg_kernel`) does not
            // itself sign-extend at load time, so we widen host-side by
            // upcasting each i32 to i64 before upload, allocate an i64
            // accumulator, and request the i64-typed kernel — which then
            // emits `atom.global.add.u64` (PTX has no `.s64` variant of
            // atom.add — `.u64` is bit-identical for two's-complement signed
            // addition). MIN/MAX preserve the input
            // dtype and stay on the i32 path.
            let widened_dtype = sum_output_dtype(DataType::Int32);
            let widen_to_i64 = matches!(op, ReduceOp::Sum) && widened_dtype == DataType::Int64;
            if widen_to_i64 {
                let host: Vec<i64> = collect_filtered_primitive(pa, key_valid, value_valid.as_deref())
                    .into_iter()
                    .map(|v| v as i64)
                    .collect();
                debug_assert_eq!(host.len(), n);
                let input_gpu = GpuVec::<i64>::from_slice_async(&host, stream.raw())?;
                let init: Vec<i64> = vec![identity_i64(op); k];
                let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
                launch_agg_kernel::<i64>(
                    op,
                    DataType::Int64,
                    filtered.col(),
                    keys_table,
                    &input_gpu,
                    &mut acc,
                    n,
                    k_u32,
                    stream,
                    None,
                )?;
                Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?))
            } else {
                let host: Vec<i32> = collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
                debug_assert_eq!(host.len(), n);
                let input_gpu = GpuVec::<i32>::from_slice_async(&host, stream.raw())?;
                let init: Vec<i32> = vec![identity_i32(op); k];
                let mut acc = GpuVec::<i32>::from_slice_async(&init, stream.raw())?;
                launch_agg_kernel::<i32>(
                    op,
                    DataType::Int32,
                    filtered.col(),
                    keys_table,
                    &input_gpu,
                    &mut acc,
                    n,
                    k_u32,
                    stream,
                    None,
                )?;
                Ok(AccDownload::I32(download_pinned_i32(&acc, stream)?))
            }
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            let host: Vec<i64> = collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
            debug_assert_eq!(host.len(), n);
            let input_gpu = GpuVec::<i64>::from_slice_async(&host, stream.raw())?;
            let init: Vec<i64> = vec![identity_i64(op); k];
            let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel::<i64>(
                op,
                DataType::Int64,
                filtered.col(),
                keys_table,
                &input_gpu,
                &mut acc,
                n,
                k_u32,
                stream,
                None,
            )?;
            Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?))
        }
        DataType::Float32 => {
            // MIN/MAX over floats are routed to the float-atomic CAS kernel
            // by launch_agg_kernel; no early rejection needed.
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            let host: Vec<f32> = collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
            debug_assert_eq!(host.len(), n);
            let input_gpu = GpuVec::<f32>::from_slice_async(&host, stream.raw())?;
            let init: Vec<f32> = vec![identity_f32(op); k];
            let mut acc = GpuVec::<f32>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel::<f32>(
                op,
                DataType::Float32,
                filtered.col(),
                keys_table,
                &input_gpu,
                &mut acc,
                n,
                k_u32,
                stream,
                None,
            )?;
            Ok(AccDownload::F32(download_pinned_f32(&acc, stream)?))
        }
        DataType::Float64 => {
            // MIN/MAX over floats are routed to the float-atomic CAS kernel
            // by launch_agg_kernel; no early rejection needed.
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            let host: Vec<f64> = collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
            debug_assert_eq!(host.len(), n);
            let input_gpu = GpuVec::<f64>::from_slice_async(&host, stream.raw())?;
            let init: Vec<f64> = vec![identity_f64(op); k];
            let mut acc = GpuVec::<f64>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel::<f64>(
                op,
                DataType::Float64,
                filtered.col(),
                keys_table,
                &input_gpu,
                &mut acc,
                n,
                k_u32,
                stream,
                None,
            )?;
            Ok(AccDownload::F64(download_pinned_f64(&acc, stream)?))
        }
        DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
            "aggregate input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// PV-stage-f: native-validity dispatch — upload the FULL value column +
/// the packed validity bitmap and let the GPU skip NULL rows on the
/// device, instead of host-stripping them before upload.
///
/// Preconditions (enforced at the call site in `run_typed_agg`):
/// 1. `arr.null_count() > 0` — there's actually a null to skip.
/// 2. `(op, col_io.dtype)` is one of the integer SUM/MIN/MAX or float SUM
///    combinations handled by [`compile_groupby_agg_kernel_with_validity`].
/// 3. The key column (`group_col`) is parallel to the source batch by row
///    index (i.e. `pack_keys` did not drop NULL-key rows).
///
/// The launcher increments [`NATIVE_VALIDITY_LAUNCHES`] each time it
/// actually consumes a validity pointer, which inline tests assert on.
#[allow(clippy::too_many_arguments)]
fn run_typed_agg_native_validity(
    op: ReduceOp,
    col_io: &ColumnIO,
    arr: &dyn Array,
    value_valid: &[bool],
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    stream: &CudaStream,
) -> BoltResult<AccDownload> {
    debug_assert_eq!(value_valid.len(), n_rows);
    // Pack the validity bitmap into the kernel's u32-word layout. Reuse
    // the helper exposed by `groupby_with_pre` so all native-validity
    // executors agree on the LE bit ordering documented in
    // [`crate::jit::hash_kernels`].
    let validity_bytes: Vec<u8> = value_valid
        .iter()
        .map(|&v| if v { 1u8 } else { 0u8 })
        .collect();
    let packed = crate::exec::groupby_with_pre::pack_validity_bits(&validity_bytes);
    let validity_gpu = GpuVec::<u32>::from_slice_async(&packed, stream.raw())?;
    let validity_ptr = Some(validity_gpu.device_ptr());

    match col_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int32"))?;
            // SUM(Int32) widens to i64 (same rule as the host-strip path).
            let widened_dtype = sum_output_dtype(DataType::Int32);
            if matches!(op, ReduceOp::Sum) && widened_dtype == DataType::Int64 {
                let widened: Vec<i64> = pa.values().iter().map(|&v| v as i64).collect();
                let input_gpu = GpuVec::<i64>::from_slice_async(&widened, stream.raw())?;
                let init: Vec<i64> = vec![identity_i64(op); k];
                let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
                launch_agg_kernel::<i64>(
                    op,
                    DataType::Int64,
                    group_col,
                    keys_table,
                    &input_gpu,
                    &mut acc,
                    n_rows,
                    k_u32,
                    stream,
                    validity_ptr,
                )?;
                let _ = validity_gpu; // keep validity buffer alive across launch
                Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?))
            } else {
                let host: Vec<i32> = pa.values().to_vec();
                let input_gpu = GpuVec::<i32>::from_slice_async(&host, stream.raw())?;
                let init: Vec<i32> = vec![identity_i32(op); k];
                let mut acc = GpuVec::<i32>::from_slice_async(&init, stream.raw())?;
                launch_agg_kernel::<i32>(
                    op,
                    DataType::Int32,
                    group_col,
                    keys_table,
                    &input_gpu,
                    &mut acc,
                    n_rows,
                    k_u32,
                    stream,
                    validity_ptr,
                )?;
                let _ = validity_gpu;
                Ok(AccDownload::I32(download_pinned_i32(&acc, stream)?))
            }
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            let host: Vec<i64> = pa.values().to_vec();
            let input_gpu = GpuVec::<i64>::from_slice_async(&host, stream.raw())?;
            let init: Vec<i64> = vec![identity_i64(op); k];
            let mut acc = GpuVec::<i64>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel::<i64>(
                op,
                DataType::Int64,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_rows,
                k_u32,
                stream,
                validity_ptr,
            )?;
            let _ = validity_gpu;
            Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?))
        }
        DataType::Float32 => {
            // Predicate gates float MIN/MAX out; only SUM reaches here.
            debug_assert!(matches!(op, ReduceOp::Sum));
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            let host: Vec<f32> = pa.values().to_vec();
            let input_gpu = GpuVec::<f32>::from_slice_async(&host, stream.raw())?;
            let init: Vec<f32> = vec![identity_f32(op); k];
            let mut acc = GpuVec::<f32>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel::<f32>(
                op,
                DataType::Float32,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_rows,
                k_u32,
                stream,
                validity_ptr,
            )?;
            let _ = validity_gpu;
            Ok(AccDownload::F32(download_pinned_f32(&acc, stream)?))
        }
        DataType::Float64 => {
            debug_assert!(matches!(op, ReduceOp::Sum));
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            let host: Vec<f64> = pa.values().to_vec();
            let input_gpu = GpuVec::<f64>::from_slice_async(&host, stream.raw())?;
            let init: Vec<f64> = vec![identity_f64(op); k];
            let mut acc = GpuVec::<f64>::from_slice_async(&init, stream.raw())?;
            launch_agg_kernel::<f64>(
                op,
                DataType::Float64,
                group_col,
                keys_table,
                &input_gpu,
                &mut acc,
                n_rows,
                k_u32,
                stream,
                validity_ptr,
            )?;
            let _ = validity_gpu;
            Ok(AccDownload::F64(download_pinned_f64(&acc, stream)?))
        }
        DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
            "native-validity dispatch reached unsupported dtype {:?} (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// Collect a primitive Arrow array's values into a fresh `Vec`, filtering
/// out positions where (key_valid AND value_valid) is false. Either mask
/// being `None` means "all true" for that side. The output length equals the
/// post-filter row count.
fn collect_filtered_primitive<P>(
    pa: &arrow_array::PrimitiveArray<P>,
    key_valid: Option<&[bool]>,
    value_valid: Option<&[bool]>,
) -> Vec<P::Native>
where
    P: arrow_array::types::ArrowPrimitiveType,
    P::Native: Copy,
{
    let n = pa.len();
    let vals = pa.values();
    let mut out: Vec<P::Native> = Vec::with_capacity(n);
    for i in 0..n {
        let kv = key_valid.map(|m| m[i]).unwrap_or(true);
        let vv = value_valid.map(|m| m[i]).unwrap_or(true);
        if kv && vv {
            out.push(vals[i]);
        }
    }
    out
}

/// Pull a numeric input column out of `batch`, upcast each element to f64,
/// and drop positions where (key_valid AND value_valid) is false. Either
/// mask being `None` means "all true" for that side. Used by AVG so the
/// numerator and denominator stay aligned with the (key-NULL, value-NULL)
/// filter applied to the rest of the launch.
fn load_input_column_as_f64_filtered(
    col_io: &ColumnIO,
    batch: &RecordBatch,
    key_valid: Option<&[bool]>,
    value_valid: Option<&[bool]>,
) -> BoltResult<Vec<f64>> {
    let idx = batch.schema().index_of(&col_io.name).map_err(|e| {
        BoltError::Plan(format!(
            "AVG input '{}' not present in table batch: {}",
            col_io.name, e
        ))
    })?;
    let arr = batch.column(idx);
    match col_io.dtype {
        DataType::Int32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int32"))?;
            Ok(filter_iter_to_f64(pa, key_valid, value_valid, |v| v as f64))
        }
        DataType::Int64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Int64"))?;
            Ok(filter_iter_to_f64(pa, key_valid, value_valid, |v| v as f64))
        }
        DataType::Float32 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float32"))?;
            Ok(filter_iter_to_f64(pa, key_valid, value_valid, |v| v as f64))
        }
        DataType::Float64 => {
            let pa = arr
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Float64"))?;
            Ok(filter_iter_to_f64(pa, key_valid, value_valid, |v| v))
        }
        DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
            "AVG input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// Helper for `load_input_column_as_f64_filtered`: walks a primitive Arrow
/// array, applies the joint key/value validity mask, and casts each surviving
/// element via `f`.
fn filter_iter_to_f64<P, F>(
    pa: &arrow_array::PrimitiveArray<P>,
    key_valid: Option<&[bool]>,
    value_valid: Option<&[bool]>,
    f: F,
) -> Vec<f64>
where
    P: arrow_array::types::ArrowPrimitiveType,
    P::Native: Copy,
    F: Fn(P::Native) -> f64,
{
    let n = pa.len();
    let vals = pa.values();
    let mut out: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let kv = key_valid.map(|m| m[i]).unwrap_or(true);
        let vv = value_valid.map(|m| m[i]).unwrap_or(true);
        if kv && vv {
            out.push(f(vals[i]));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Output assembly.
// ---------------------------------------------------------------------------

/// Build one Arrow array per group-by column by decoding each packed i64 key
/// back through `decode_key`. Returns arrays in the order the columns appear
/// in `aggregate.group_by` (which matches `components`).
fn build_key_arrays(
    groups: &[(i64, usize)],
    components: &[KeyComponent],
) -> BoltResult<Vec<ArrayRef>> {
    let m = components.len();
    let n = groups.len();

    // Per-column typed buffers; we allocate the right one based on the
    // component dtype and push exactly one value per group.
    enum ColBuf {
        I32(Vec<i32>),
        I64(Vec<i64>),
        F32(Vec<f32>),
        F64(Vec<f64>),
    }

    let mut buffers: Vec<ColBuf> = Vec::with_capacity(m);
    for comp in components {
        match comp.original_dtype {
            DataType::Int32 => buffers.push(ColBuf::I32(Vec::with_capacity(n))),
            DataType::Int64 => buffers.push(ColBuf::I64(Vec::with_capacity(n))),
            DataType::Float32 => buffers.push(ColBuf::F32(Vec::with_capacity(n))),
            DataType::Float64 => buffers.push(ColBuf::F64(Vec::with_capacity(n))),
            DataType::Bool | DataType::Utf8 | DataType::Decimal128(_, _) | DataType::Date32 | DataType::Timestamp(_, _) => {
                return Err(BoltError::Type(format!(
                    "GROUP BY key dtype {:?} not supported on output",
                    comp.original_dtype
                )))
            }
        }
    }

    for (k, _) in groups {
        let decoded = decode_key(*k, components);
        for (buf, val) in buffers.iter_mut().zip(decoded.iter()) {
            match (buf, val) {
                (ColBuf::I32(v), KeyValue::I32(x)) => v.push(*x),
                (ColBuf::I64(v), KeyValue::I64(x)) => v.push(*x),
                (ColBuf::F32(v), KeyValue::F32(x)) => v.push(*x),
                (ColBuf::F64(v), KeyValue::F64(x)) => v.push(*x),
                _ => {
                    return Err(BoltError::Other(
                        "internal: decode_key produced a KeyValue variant \
                         that disagrees with its KeyComponent dtype"
                            .into(),
                    ))
                }
            }
        }
    }

    let mut out: Vec<ArrayRef> = Vec::with_capacity(m);
    for buf in buffers {
        match buf {
            ColBuf::I32(v) => out.push(Arc::new(Int32Array::from(v)) as ArrayRef),
            ColBuf::I64(v) => out.push(Arc::new(Int64Array::from(v)) as ArrayRef),
            ColBuf::F32(v) => out.push(Arc::new(Float32Array::from(v)) as ArrayRef),
            ColBuf::F64(v) => out.push(Arc::new(Float64Array::from(v)) as ArrayRef),
        }
    }
    Ok(out)
}

/// Build the output Arrow array for one aggregate, indexing the downloaded
/// accumulator by each group's slot.
///
/// For SUM(Int32) the accumulator was widened to i64 host-side per
/// `crate::plan::logical_plan::sum_output_dtype`, so the `AccDownload` arrives
/// as `I64` and `out_field.dtype` is `Int64` — `pack_array` consumes those
/// directly. SUM(Int64), SUM(Float32), SUM(Float64), and all MIN/MAX paths
/// preserve their input dtype unchanged.
fn build_agg_array(
    agg: &AggregateExpr,
    out_field: &Field,
    acc: &AccDownload,
    groups: &[(i64, usize)],
    n_groups: usize,
) -> BoltResult<ArrayRef> {
    match (agg, acc) {
        (AggregateExpr::Count(_), AccDownload::I64(host)) => {
            let mut out: Vec<i64> = Vec::with_capacity(n_groups);
            for (_, slot) in groups {
                out.push(host[*slot]);
            }
            pack_array(out_field.dtype, Scalars::I64(out))
        }
        (AggregateExpr::Avg(_), AccDownload::Avg { sum, count }) => {
            let mut out: Vec<f64> = Vec::with_capacity(n_groups);
            for (_, slot) in groups {
                let s = sum[*slot];
                let c = count[*slot];
                let v = if c == 0 { 0.0 } else { s / (c as f64) };
                out.push(v);
            }
            pack_array(out_field.dtype, Scalars::F64(out))
        }
        (AggregateExpr::Sum(_) | AggregateExpr::Min(_) | AggregateExpr::Max(_), other) => {
            let scalars = match other {
                AccDownload::I32(host) => Scalars::I32(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::I64(host) => Scalars::I64(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::F32(host) => Scalars::F32(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::F64(host) => Scalars::F64(
                    groups.iter().map(|(_, slot)| host[*slot]).collect(),
                ),
                AccDownload::Avg { .. } => {
                    return Err(BoltError::Other(
                        "internal: AVG accumulator passed to non-AVG aggregate"
                            .into(),
                    ))
                }
                AccDownload::Welford { .. } => {
                    return Err(BoltError::Other(
                        "internal: Welford accumulator passed to SUM/MIN/MAX aggregate"
                            .into(),
                    ))
                }
            };
            pack_array(out_field.dtype, scalars)
        }
        // v0.7: VAR_POP / VAR_SAMP / STDDEV_POP / STDDEV_SAMP per-group
        // finalisation from the host-side Welford state. Output is always
        // a nullable Float64 array:
        //   - VAR_POP / STDDEV_POP:  NULL when count == 0, M2/count
        //     (sqrt for STDDEV) otherwise. Single observation gives 0.
        //   - VAR_SAMP / STDDEV_SAMP: NULL when count <= 1 (divisor
        //     `count - 1` is undefined), M2/(count-1) (sqrt for STDDEV)
        //     otherwise. See `crate::exec::welford` for the canonical
        //     numerics — both arms route through `WelfordState::var_pop`
        //     / `var_samp` / `stddev_pop` / `stddev_samp`.
        (AggregateExpr::VarPop(_), AccDownload::Welford { states }) => {
            finalize_welford_array(states, groups, WelfordOutKind::VarPop, out_field)
        }
        (AggregateExpr::VarSamp(_), AccDownload::Welford { states }) => {
            finalize_welford_array(states, groups, WelfordOutKind::VarSamp, out_field)
        }
        (AggregateExpr::StddevPop(_), AccDownload::Welford { states }) => {
            finalize_welford_array(states, groups, WelfordOutKind::StddevPop, out_field)
        }
        (AggregateExpr::StddevSamp(_), AccDownload::Welford { states }) => {
            finalize_welford_array(states, groups, WelfordOutKind::StddevSamp, out_field)
        }
        // Defensive arms — Welford aggregates always pair with the
        // `Welford` accumulator variant. A mismatch here means a
        // dispatch bug upstream.
        (
            AggregateExpr::VarPop(_)
            | AggregateExpr::VarSamp(_)
            | AggregateExpr::StddevPop(_)
            | AggregateExpr::StddevSamp(_),
            _,
        ) => Err(BoltError::Other(
            "internal: VAR/STDDEV aggregate received a non-Welford accumulator"
                .into(),
        )),
        (_, _) => Err(BoltError::Other(
            "internal: aggregate / accumulator-variant mismatch".into(),
        )),
    }
}

/// Typed batch of per-group scalar values, prior to dtype-casting into Arrow.
enum Scalars {
    /// Int32 column.
    I32(Vec<i32>),
    /// Int64 column.
    I64(Vec<i64>),
    /// Float32 column.
    F32(Vec<f32>),
    /// Float64 column.
    F64(Vec<f64>),
}

/// Which finaliser to apply on top of a per-group [`crate::exec::welford::WelfordState`].
///
/// Tags the `(count, mean, M2)` triple with the SQL aggregate it backs:
/// `VAR_POP` and `VAR_SAMP` return raw variances, `STDDEV_POP` and
/// `STDDEV_SAMP` return their square roots. NULL semantics for each are
/// owned by [`crate::exec::welford::WelfordState`] (see `var_pop`,
/// `var_samp`, `stddev_pop`, `stddev_samp` — the public API documents the
/// `count == 0` and `count <= 1` NULL cases).
#[derive(Clone, Copy, Debug)]
enum WelfordOutKind {
    VarPop,
    VarSamp,
    StddevPop,
    StddevSamp,
}

/// Walk `groups` and emit one Float64 cell per group by reading the
/// per-slot `WelfordState` and finalising it through the matching
/// [`crate::exec::welford`] helper. Empty slots / single-observation
/// `VAR_SAMP` etc. yield SQL NULL via `Float64Array::from(Vec<Option<f64>>)`.
fn finalize_welford_array(
    states: &[crate::exec::welford::WelfordState],
    groups: &[(i64, usize)],
    kind: WelfordOutKind,
    out_field: &Field,
) -> BoltResult<ArrayRef> {
    if out_field.dtype != DataType::Float64 {
        return Err(BoltError::Type(format!(
            "GROUP BY VAR/STDDEV output dtype must be Float64, got {:?}",
            out_field.dtype
        )));
    }
    let mut out: Vec<Option<f64>> = Vec::with_capacity(groups.len());
    for (_, slot) in groups {
        let st = states.get(*slot).ok_or_else(|| {
            BoltError::Other(format!(
                "internal: groupby Welford slot {} out of range (len {})",
                slot,
                states.len()
            ))
        })?;
        let v = match kind {
            WelfordOutKind::VarPop => st.var_pop(),
            WelfordOutKind::VarSamp => st.var_samp(),
            WelfordOutKind::StddevPop => st.stddev_pop(),
            WelfordOutKind::StddevSamp => st.stddev_samp(),
        };
        out.push(v);
    }
    Ok(Arc::new(Float64Array::from(out)) as ArrayRef)
}

/// Cast a `Scalars` batch into an Arrow array of `out_dtype`. Mirrors the
/// cross-dtype matrix in `aggregate.rs::scalar_to_array`.
fn pack_array(out_dtype: DataType, scalars: Scalars) -> BoltResult<ArrayRef> {
    match (scalars, out_dtype) {
        (Scalars::I32(v), DataType::Int32) => Ok(Arc::new(Int32Array::from(v)) as ArrayRef),
        (Scalars::I64(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(v)) as ArrayRef),
        (Scalars::F32(v), DataType::Float32) => {
            Ok(Arc::new(Float32Array::from(v)) as ArrayRef)
        }
        (Scalars::F64(v), DataType::Float64) => {
            Ok(Arc::new(Float64Array::from(v)) as ArrayRef)
        }

        // Cross-dtype paths the scalar reducer also accepts.
        (Scalars::I32(v), DataType::Int64) => Ok(Arc::new(Int64Array::from(
            v.into_iter().map(|x| x as i64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::I32(v), DataType::Float32) => Ok(Arc::new(Float32Array::from(
            v.into_iter().map(|x| x as f32).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::I32(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::I64(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),
        (Scalars::F32(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(
            v.into_iter().map(|x| x as f64).collect::<Vec<_>>(),
        )) as ArrayRef),

        (_, dt) => Err(BoltError::Type(format!(
            "GROUP BY: cannot pack scalars into output dtype {:?}",
            dt
        ))),
    }
}

// ---------------------------------------------------------------------------
// Identities for the accumulator initialiser.
// ---------------------------------------------------------------------------

fn identity_i32(op: ReduceOp) -> i32 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0,
        ReduceOp::Min => i32::MAX,
        ReduceOp::Max => i32::MIN,
    }
}

fn identity_i64(op: ReduceOp) -> i64 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0,
        ReduceOp::Min => i64::MAX,
        ReduceOp::Max => i64::MIN,
    }
}

fn identity_f32(op: ReduceOp) -> f32 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0.0,
        ReduceOp::Min => f32::INFINITY,
        ReduceOp::Max => f32::NEG_INFINITY,
    }
}

fn identity_f64(op: ReduceOp) -> f64 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0.0,
        ReduceOp::Min => f64::INFINITY,
        ReduceOp::Max => f64::NEG_INFINITY,
    }
}

// ---------------------------------------------------------------------------
// Misc helpers (mirror of the private helpers in aggregate.rs).
// ---------------------------------------------------------------------------

/// Resolve `name` to its `ColumnIO` within `inputs`.
fn resolve_input<'a>(inputs: &'a [ColumnIO], name: &str) -> BoltResult<&'a ColumnIO> {
    inputs.iter().find(|c| c.name == name).ok_or_else(|| {
        BoltError::Plan(format!(
            "aggregate input column '{}' not found in plan inputs",
            name
        ))
    })
}

/// Extract the column name from a bare-column-ref expression. The v1 path
/// requires every aggregate input to be a bare column ref.
fn bare_column_name(expr: &Expr) -> BoltResult<&str> {
    match expr {
        Expr::Column(name) => Ok(name.as_str()),
        Expr::Alias(inner, _) => bare_column_name(inner),
        _ => Err(BoltError::Other(
            "GROUP BY: aggregate input must be a bare column reference in v1".into(),
        )),
    }
}

/// `Type` error for a failed Arrow downcast on column `name`.
fn downcast_err(name: &str, expected: &str) -> BoltError {
    BoltError::Type(format!(
        "GROUP BY input column '{}' could not be downcast to {}",
        name, expected
    ))
}

/// Map Arrow `DataType` to our plan `DataType` (mirrors `aggregate.rs`).
fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    crate::exec::schema_convert::arrow_dtype_to_plan_basic(d, "")
}

/// Build an Arrow `Schema` from our plan `Schema` for the output `RecordBatch`.
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    crate::exec::schema_convert::plan_schema_to_arrow_schema_no_temporal(s, "this aggregate output path")
}

// ---------------------------------------------------------------------------
// Host-only tests for `pack_keys` / `decode_key` (no GPU required).
// ---------------------------------------------------------------------------
// v0.7: these arrow/plan aliases are used only by the #[cfg(test)] modules
// below; the non-test schema conversion now lives in exec::schema_convert.
// cfg(test)-gated so normal builds don't see an unused import.
#[cfg(test)]
use arrow_schema::{Field as ArrowField};

#[cfg(test)]
mod tests {
    use super::*;

    // dedup (groupby_common): the `spec` and `two_col_batch` test fixtures
    // moved to `groupby_common`'s test module alongside the `pack_keys` tests
    // that used them. This module keeps `one_col_batch` for its own
    // `column_null_mask` / `collect_filtered_primitive` coverage.

    /// Build a single-column RecordBatch from a typed Arrow array. The field
    /// is marked nullable so callers can pass arrays with NULL validity
    /// bitmaps (used by the H1 NULL tests below).
    fn one_col_batch(name: &str, arr: ArrayRef) -> RecordBatch {
        let dt = arr.data_type().clone();
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            name, dt, true,
        )]));
        RecordBatch::try_new(schema, vec![arr]).expect("one-col batch")
    }

    // dedup (groupby_common): the `pack_keys` / `decode_key` / `and_masks`
    // unit tests (single/two-column Int32, Float32, the >64-bit reject paths,
    // the V-17 round-trips, and the NULL-mask surfacing tests) were
    // centralised into `crate::exec::groupby_common`'s `#[cfg(test)] mod
    // tests` when those helpers were consolidated there — coverage is the
    // union of this module's and `groupby_valid.rs`'s former copies. The
    // tests below cover this executor's OWN (non-shared) host helpers.

    /// `column_null_mask` returns `None` when the column has no nulls and
    /// a per-row mask when it does. Drives the SUM/MIN/MAX value-NULL
    /// filtering in `run_typed_agg`.
    #[test]
    fn column_null_mask_basic() {
        let arr: ArrayRef = Arc::new(Int64Array::from(vec![1i64, 2, 3]));
        let batch = one_col_batch("v", arr);
        let io = ColumnIO {
            name: "v".to_string(),
            dtype: DataType::Int64,
        };
        assert!(column_null_mask(&io, &batch).unwrap().is_none());

        let arr2: ArrayRef =
            Arc::new(Int64Array::from(vec![Some(1i64), None, Some(3)]));
        let batch2 = one_col_batch("v", arr2);
        let mask = column_null_mask(&io, &batch2).unwrap().expect("mask");
        assert_eq!(mask, vec![true, false, true]);
    }

    /// `collect_filtered_primitive` drops positions where EITHER mask is
    /// false; this is exactly the SUM/MIN/MAX value-upload path. The
    /// garbage-at-NULL bytes that would otherwise corrupt the reduction
    /// stay in the source buffer and never reach the kernel.
    #[test]
    fn collect_filtered_primitive_drops_null_rows() {
        // Underlying values buffer at NULL positions could be anything;
        // here we use a large value (1000) that would visibly skew SUM.
        let arr = Int32Array::from(vec![Some(1i32), None, Some(3), Some(4)]);
        // value_valid derived from arrow: [T, F, T, T]
        let vv: Vec<bool> = (0..arr.len()).map(|i| !arr.is_null(i)).collect();
        let out = collect_filtered_primitive::<arrow_array::types::Int32Type>(
            &arr,
            None,
            Some(&vv),
        );
        assert_eq!(out, vec![1, 3, 4]);

        // With a key_valid that ALSO drops row 0, only rows 2 and 3 survive.
        let kv = vec![false, true, true, true];
        let out2 = collect_filtered_primitive::<arrow_array::types::Int32Type>(
            &arr,
            Some(&kv),
            Some(&vv),
        );
        assert_eq!(out2, vec![3, 4]);
    }

    /// `filter_iter_to_f64` (the AVG-input helper) drops NULL positions
    /// from both the key and the value side and upcasts in one pass.
    #[test]
    fn filter_iter_to_f64_drops_and_casts() {
        let arr = Int32Array::from(vec![Some(2i32), None, Some(4), Some(6)]);
        let vv: Vec<bool> = (0..arr.len()).map(|i| !arr.is_null(i)).collect();
        let out = filter_iter_to_f64::<arrow_array::types::Int32Type, _>(
            &arr,
            None,
            Some(&vv),
            |v| v as f64,
        );
        assert_eq!(out, vec![2.0f64, 4.0, 6.0]);
    }

    // -------- Stage-3 async round-trip (requires GPU) ----------------

    /// Single-key GROUP BY through the engine: confirms that the
    /// Stage-3 async memcpy + pinned D2H plumbing produces the same
    /// per-group sums as a host-side check.
    #[test]
    #[ignore = "gpu:tier1"]
    fn async_groupby_int32_sum_round_trip() {
        use crate::Engine;
        use arrow_array::Int64Array;

        let mut engine = Engine::new().expect("ctx");
        // 12 rows, key in {0, 1, 2}; expected SUMs derived from the
        // closed form 0..12 grouped by key % 3.
        let keys: Vec<i32> = (0..12i32).map(|i| i % 3).collect();
        let vals: Vec<i32> = (0..12i32).collect();
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys)) as ArrayRef,
                Arc::new(Int32Array::from(vals)) as ArrayRef,
            ],
        )
        .unwrap();
        engine.register_table("t", batch).unwrap();

        let h = engine
            .sql("SELECT k, SUM(v) FROM t GROUP BY k")
            .expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 3);
        let ks = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let ss = out
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        // Build a host-side expected map and compare.
        let mut expected = std::collections::HashMap::<i32, i64>::new();
        for v in 0..12i64 {
            *expected.entry((v as i32) % 3).or_default() += v;
        }
        for i in 0..3 {
            let k = ks.value(i);
            let s = ss.value(i);
            assert_eq!(
                Some(&s),
                expected.get(&k).map(|x| x),
                "key={} sum={}",
                k,
                s
            );
        }
    }

    // ---- PV-stage-e: runtime native-validity dispatch decision ----

    /// Columns with no NULLs must keep the classic kernel — the
    /// `_with_validity` path costs an extra bitmap upload for no gain.
    #[test]
    fn pv_stage_e_no_nulls_skips_native_validity() {
        let arr = Int64Array::from(vec![1i64, 2, 3, 4, 5]);
        assert_eq!(arr.null_count(), 0);
        assert!(
            !column_should_use_native_validity(&arr, ReduceOp::Sum, DataType::Int64),
            "no NULLs in column -> classic kernel"
        );
    }

    /// SUM(Int64) over a NULL-bearing column is the canonical Stage E
    /// dispatch: planner+runtime both indicate validity and the kernel
    /// has a native `_with_validity` variant.
    #[test]
    fn pv_stage_e_sum_int64_with_nulls_uses_native_validity() {
        let arr = Int64Array::from(vec![Some(1i64), None, Some(3), None, Some(5)]);
        assert_eq!(arr.null_count(), 2);
        assert!(
            column_should_use_native_validity(&arr, ReduceOp::Sum, DataType::Int64),
            "NULL-bearing Int64 SUM should dispatch native validity"
        );
    }

    /// MIN(Float64) over NULL-bearing data must NOT dispatch native — the
    /// CAS-loop `float_atomics` kernel has no `_with_validity` companion
    /// at this stage. The legacy host-strip path is the correct fallback.
    #[test]
    fn pv_stage_e_float_minmax_with_nulls_falls_back_to_host_strip() {
        let arr = Float64Array::from(vec![Some(1.0f64), None, Some(3.0)]);
        assert!(arr.null_count() > 0);
        assert!(
            !column_should_use_native_validity(&arr, ReduceOp::Min, DataType::Float64),
            "Float MIN has no _with_validity emitter; expected host-strip"
        );
        assert!(
            !column_should_use_native_validity(&arr, ReduceOp::Max, DataType::Float64),
            "Float MAX has no _with_validity emitter; expected host-strip"
        );
    }

    // -------- v0.7 async-memcpy: prepare_filtered_keys round-trips --------

    /// GROUP BY AVG over a Float64 column with NULLs: exercises the
    /// `prepare_filtered_keys` value-validity branch end-to-end. The
    /// downloaded key column is rebuilt for the filtered row set via the
    /// async DtoH+HtoD pair added in v0.7. NULL value rows are dropped
    /// before they reach the GPU (matching the SUM/COUNT pair feeding
    /// AVG).
    #[test]
    #[ignore = "gpu:tier1"]
    fn async_groupby_avg_float64_with_value_nulls_round_trip() {
        use crate::Engine;

        let mut engine = Engine::new().expect("ctx");
        // Keys in {1, 2}, values with NULLs scattered through both groups.
        // Expected per-group means computed below from the non-NULL pairs.
        let keys: Vec<i32> = vec![1, 2, 1, 2, 1, 2, 1, 2];
        let vals: Vec<Option<f64>> = vec![
            Some(1.0),
            Some(10.0),
            None,
            Some(20.0),
            Some(3.0),
            None,
            Some(5.0),
            Some(40.0),
        ];
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys.clone())) as ArrayRef,
                Arc::new(Float64Array::from(vals.clone())) as ArrayRef,
            ],
        )
        .unwrap();
        engine.register_table("t", batch).unwrap();

        let h = engine
            .sql("SELECT k, AVG(v) FROM t GROUP BY k")
            .expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 2);

        let ks = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let avgs = out
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        // Build host-side expected (sum, count) pairs from the non-NULL
        // values only — same filter the GPU side has to materialise.
        let mut sums = std::collections::HashMap::<i32, f64>::new();
        let mut counts = std::collections::HashMap::<i32, i64>::new();
        for (k, v) in keys.iter().zip(vals.iter()) {
            if let Some(v) = v {
                *sums.entry(*k).or_default() += *v;
                *counts.entry(*k).or_default() += 1;
            }
        }
        for i in 0..out.num_rows() {
            let k = ks.value(i);
            let got = avgs.value(i);
            let expected = sums[&k] / counts[&k] as f64;
            assert!(
                (got - expected).abs() < 1e-12,
                "k={k} got={got} expected={expected}"
            );
        }
    }

    /// GROUP BY COUNT(col) with NULL values: same `prepare_filtered_keys`
    /// path as AVG but only the COUNT half. Pins that the v0.7 async
    /// keys upload still produces the right per-group non-null counts.
    #[test]
    #[ignore = "gpu:tier1"]
    fn async_groupby_count_with_value_nulls_round_trip() {
        use crate::Engine;
        use arrow_array::Int64Array;

        let mut engine = Engine::new().expect("ctx");
        let keys: Vec<i32> = vec![0, 1, 0, 1, 0, 1];
        let vals: Vec<Option<i32>> = vec![Some(7), None, Some(9), Some(11), None, Some(13)];
        let schema = Arc::new(ArrowSchema::new(vec![
            ArrowField::new("k", ArrowDataType::Int32, false),
            ArrowField::new("v", ArrowDataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(keys.clone())) as ArrayRef,
                Arc::new(Int32Array::from(vals.clone())) as ArrayRef,
            ],
        )
        .unwrap();
        engine.register_table("t", batch).unwrap();

        let h = engine
            .sql("SELECT k, COUNT(v) FROM t GROUP BY k")
            .expect("execute");
        let out = h.record_batch();
        assert_eq!(out.num_rows(), 2);

        let ks = out
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let cs = out
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();

        let mut expected_counts = std::collections::HashMap::<i32, i64>::new();
        for (k, v) in keys.iter().zip(vals.iter()) {
            if v.is_some() {
                *expected_counts.entry(*k).or_default() += 1;
            }
        }
        for i in 0..out.num_rows() {
            let k = ks.value(i);
            let c = cs.value(i);
            assert_eq!(c, expected_counts[&k], "k={k}");
        }
    }


    // ----- v0.7: GROUP BY VAR/STDDEV finaliser (host-only) ---------------
    //
    // `finalize_welford_array` is the host-side join between the per-group
    // `WelfordState` table and the post-sort group order. These tests cover
    // its slot indexing + NULL-emission semantics without touching the GPU.

    /// VAR_POP over a single-observation group is exactly 0.0, and the
    /// finaliser picks the right state via the (key, slot) tuple.
    #[test]
    fn finalize_welford_var_pop_basic() {
        use crate::exec::welford::WelfordState;
        // Slot layout: slot 0 holds the only group; slot 1 is empty.
        let mut states = vec![WelfordState::empty(); 2];
        states[0].push(1.0);
        states[0].push(2.0);
        states[0].push(3.0);
        // groups: one entry, key = 7, lives at slot 0.
        let groups = vec![(7i64, 0usize)];
        let out_field = Field::new("var_pop", DataType::Float64, true);
        let arr = finalize_welford_array(
            &states,
            &groups,
            WelfordOutKind::VarPop,
            &out_field,
        )
        .expect("finalize ok");
        let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(a.len(), 1);
        // var_pop([1,2,3]) = 2/3
        let v = a.value(0);
        assert!((v - (2.0 / 3.0)).abs() < 1e-12);
        assert!(!a.is_null(0));
    }

    /// VAR_SAMP over count == 1 must emit SQL NULL: the divisor (n-1)
    /// is 0. The Float64 output array is nullable, so this is observable
    /// via `is_null`.
    #[test]
    fn finalize_welford_var_samp_single_obs_is_null() {
        use crate::exec::welford::WelfordState;
        let mut states = vec![WelfordState::empty(); 1];
        states[0].push(42.0);
        let groups = vec![(0i64, 0usize)];
        let out_field = Field::new("var_samp", DataType::Float64, true);
        let arr = finalize_welford_array(
            &states,
            &groups,
            WelfordOutKind::VarSamp,
            &out_field,
        )
        .expect("finalize ok");
        let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(a.is_null(0), "VAR_SAMP(single) must be SQL NULL");
    }

    /// VAR_POP over count == 0 (empty state, e.g. all-NULL inputs in a
    /// group) is SQL NULL, mirroring the scalar-aggregate path.
    #[test]
    fn finalize_welford_var_pop_empty_is_null() {
        use crate::exec::welford::WelfordState;
        let states = vec![WelfordState::empty(); 1];
        let groups = vec![(0i64, 0usize)];
        let out_field = Field::new("var_pop", DataType::Float64, true);
        let arr = finalize_welford_array(
            &states,
            &groups,
            WelfordOutKind::VarPop,
            &out_field,
        )
        .expect("finalize ok");
        let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(a.is_null(0), "VAR_POP(empty) must be SQL NULL");
    }

    /// STDDEV_POP is the sqrt of VAR_POP, and the finaliser routes
    /// `STDDEV_POP` through `WelfordState::stddev_pop`.
    #[test]
    fn finalize_welford_stddev_pop_basic() {
        use crate::exec::welford::WelfordState;
        let mut states = vec![WelfordState::empty(); 1];
        states[0].push(1.0);
        states[0].push(2.0);
        states[0].push(3.0);
        let groups = vec![(0i64, 0usize)];
        let out_field = Field::new("stddev_pop", DataType::Float64, true);
        let arr = finalize_welford_array(
            &states,
            &groups,
            WelfordOutKind::StddevPop,
            &out_field,
        )
        .expect("finalize ok");
        let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        let expected = (2.0_f64 / 3.0).sqrt();
        assert!((a.value(0) - expected).abs() < 1e-12);
    }

    /// STDDEV_SAMP over count > 1 returns sqrt(var_samp). For [1,2,3]
    /// the sample variance is 1.0 and the stddev is 1.0.
    #[test]
    fn finalize_welford_stddev_samp_basic() {
        use crate::exec::welford::WelfordState;
        let mut states = vec![WelfordState::empty(); 1];
        states[0].push(1.0);
        states[0].push(2.0);
        states[0].push(3.0);
        let groups = vec![(0i64, 0usize)];
        let out_field = Field::new("stddev_samp", DataType::Float64, true);
        let arr = finalize_welford_array(
            &states,
            &groups,
            WelfordOutKind::StddevSamp,
            &out_field,
        )
        .expect("finalize ok");
        let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!((a.value(0) - 1.0).abs() < 1e-12);
        assert!(!a.is_null(0));
    }

    /// Multi-group: walk groups in slot-order and confirm each slot's
    /// state lands on the right output row.
    #[test]
    fn finalize_welford_multi_group_var_pop() {
        use crate::exec::welford::WelfordState;
        // 4-slot table; groups land at slots 0, 2, 3.
        let mut states = vec![WelfordState::empty(); 4];
        // slot 0: [1, 2, 3]  -> var_pop = 2/3
        for &x in &[1.0, 2.0, 3.0] {
            states[0].push(x);
        }
        // slot 2: constant value -> var_pop == 0.0
        for _ in 0..10 {
            states[2].push(7.5);
        }
        // slot 3: empty -> NULL
        let groups = vec![(10i64, 0usize), (20i64, 2usize), (30i64, 3usize)];
        let out_field = Field::new("var_pop", DataType::Float64, true);
        let arr = finalize_welford_array(
            &states,
            &groups,
            WelfordOutKind::VarPop,
            &out_field,
        )
        .expect("finalize ok");
        let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(a.len(), 3);
        assert!((a.value(0) - (2.0 / 3.0)).abs() < 1e-12);
        assert!(!a.is_null(0));
        assert_eq!(a.value(1), 0.0);
        assert!(!a.is_null(1));
        assert!(a.is_null(2), "empty group must be NULL");
    }

    /// The output dtype guard rejects non-Float64 fields up front so a
    /// future refactor can't silently emit the wrong column type.
    #[test]
    fn finalize_welford_rejects_non_float64() {
        use crate::exec::welford::WelfordState;
        let states = vec![WelfordState::empty(); 1];
        let groups = vec![(0i64, 0usize)];
        let out_field = Field::new("bad", DataType::Int64, true);
        let r = finalize_welford_array(
            &states,
            &groups,
            WelfordOutKind::VarPop,
            &out_field,
        );
        assert!(r.is_err(), "non-Float64 output dtype must be rejected");
    }

    /// GB-S2: the `partition_reduce spill` sentinel is recognised as a soft
    /// miss. `execute_groupby` matches a fast path's `Some(Err(spill))`
    /// against `PARTITION_REDUCE_SPILL_PREFIX` and falls through instead of
    /// aborting dispatch. This test exercises the exact recognition
    /// predicate the `try_fast_path!` macro uses, and confirms an unrelated
    /// `Other` error is NOT treated as a spill (so it would still
    /// propagate).
    #[test]
    fn spill_sentinel_is_recognized_as_soft_miss() {
        use crate::exec::groupby_tier2_orchestrator::PARTITION_REDUCE_SPILL_PREFIX;

        // Mirror the literal each orchestrator formats on MAX_PROBES overflow.
        let spill = BoltError::Other(format!(
            "partition_reduce spill: {} rows exceeded MAX_PROBES; result may be incorrect",
            42
        ));
        let is_spill = matches!(
            &spill,
            BoltError::Other(msg) if msg.starts_with(PARTITION_REDUCE_SPILL_PREFIX)
        );
        assert!(is_spill, "spill error must match the soft-miss sentinel prefix");

        // The two-key AVG variant uses a different trailing detail but the
        // same prefix — still recognised.
        let spill_avg = BoltError::Other(format!(
            "partition_reduce spill: multi-sum={} count={} rows exceeded MAX_PROBES; result may be incorrect",
            1, 2
        ));
        assert!(matches!(
            &spill_avg,
            BoltError::Other(msg) if msg.starts_with(PARTITION_REDUCE_SPILL_PREFIX)
        ));

        // An unrelated error must NOT be swallowed (it would propagate).
        let other = BoltError::Other("tier2: reduce-kernel output buffers".to_string());
        assert!(!matches!(
            &other,
            BoltError::Other(msg) if msg.starts_with(PARTITION_REDUCE_SPILL_PREFIX)
        ));
    }
}
