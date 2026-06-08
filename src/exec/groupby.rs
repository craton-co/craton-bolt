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
    Array, ArrayRef, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch,
};
use arrow_schema::{DataType as ArrowDataType, Schema as ArrowSchema};
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
    compile_groupby_decimal_kernel, compile_groupby_keys_kernel_dispatched, groupby_block_size,
    GroupedDecimalOp, AGG_DECIMAL_KERNEL_ENTRY, AGG_KERNEL_ENTRY, I64_EMPTY_SENTINEL,
    KEYS_KERNEL_ENTRY,
};
use crate::plan::logical_plan::{
    sum_output_dtype, AggregateExpr, DataType, Expr, Field, Schema, TimeUnit,
};
use crate::plan::physical_plan::{ColumnIO, PhysicalPlan};

/// Empty-slot sentinel; mirrors the literal baked into the keys kernel.
/// Re-export of [`I64_EMPTY_SENTINEL`] under the legacy local name to keep
/// existing call sites in this module unchanged.
const EMPTY_KEY: i64 = I64_EMPTY_SENTINEL;

/// Reserved encoded-key value used to represent the single SQL "NULL group"
/// on the classic kernel path (NULL-key fix). SQL standard / DuckDB /
/// Postgres group all NULL-keyed rows into ONE group whose key is NULL; the
/// previous code silently DROPPED those rows. We re-introduce them under
/// this sentinel so the existing keys + agg kernels compute the NULL group's
/// aggregates exactly like any other group, then [`build_key_arrays`] emits
/// a NULL (not a decoded value) for the slot that carries this key.
///
/// `i64::MAX` is chosen as the mirror of the `i64::MIN` empty-slot sentinel.
/// It can only collide with a real key if a single Int64 GROUP BY column
/// literally contains `i64::MAX`; that case is detected in `execute_groupby`
/// and routed to the sentinel-free valid-flag executor (which preserves the
/// pre-existing drop-NULLs behaviour — see the call site note). For every
/// other key dtype (Int32 widening, Float32/Float64 bit patterns, and
/// 2-column packs) the encoded value range never reaches `i64::MAX` for a
/// genuine key, so the sentinel is safe.
const NULL_GROUP_KEY: i64 = i64::MAX;

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
    decode_key, download_pinned_f32, download_pinned_f64, download_pinned_i32, download_pinned_i64,
    next_pow2, pack_keys, unique_count, KeyComponent, KeyValue,
};

/// Resident on-device GROUP BY fast path — the perf companion to
/// [`execute_groupby`]. Tries each tier executor's *resident* variant, which
/// reads keys/values straight from the already-uploaded `GpuTable` instead of
/// re-uploading them from the host batch every query (the H2D upload is ~78%
/// of a low-cardinality SUM's wall-clock at 10M rows — see
/// `examples/profile_groupby.rs`). Returns `None` to fall back to
/// `execute_groupby` (the host-upload path), preserving behaviour for every
/// shape without a resident variant yet.
///
/// `batch` is the host-materialised table (an Arc clone for a singly-registered
/// table) consulted only for transfer-free host scans (key range / presence);
/// `resident` carries the device buffers. Currently only the Tier-1 shared-mem
/// single-`SUM(Float64)`-by-`Int32` path is implemented on-device; everything
/// else falls through.
pub fn try_execute_groupby_resident(
    plan: &PhysicalPlan,
    resident: &crate::exec::gpu_table::GpuTable,
    batch: &RecordBatch,
) -> Option<BoltResult<RecordBatch>> {
    crate::exec::groupby_shmem_exec::try_execute_resident(plan, resident, batch)
}

/// Execute a GROUP BY aggregate plan against a host-side `RecordBatch`.
///
/// `plan` must be `PhysicalPlan::Aggregate` with non-empty `group_by`.
/// Supports single-column (Int32/Int64/Float32/Float64) and a limited set of
/// 2-column packings whose combined width fits in 64 bits; see module docs.
pub fn execute_groupby(plan: &PhysicalPlan, table_batch: &RecordBatch) -> BoltResult<RecordBatch> {
    // R1-utf8-groupby: Utf8 (string) GROUP BY keys. The classic + tier
    // integer kernels only key on i32/i64/float, so a string key column
    // previously fell through to `groupby_wide`, which REJECTS Utf8. We now
    // dictionary-encode each Utf8 key column into a LEX-RANKED i32 code
    // column, run the ordinary integer GROUP BY on the codes, then
    // reconstruct the Utf8 key column(s) from the dictionary on output. This
    // keeps the integer kernels (and `dict_registry`) untouched. The detector
    // returns `None` for plans with no Utf8 key, so the all-integer path is
    // bit-for-bit unchanged. See `utf8_groupby` below.
    if let Some(res) = utf8_groupby::try_execute_groupby_utf8(plan, table_batch) {
        return res;
    }

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
    try_fast_path!(crate::exec::groupby_shmem_exec::try_execute(
        plan,
        table_batch
    ));
    try_fast_path!(crate::exec::groupby_shmem_multi_exec::try_execute(
        plan,
        table_batch
    ));
    try_fast_path!(crate::exec::groupby_shmem_avg_exec::try_execute(
        plan,
        table_batch
    ));
    try_fast_path!(crate::exec::groupby_tier2_exec::try_execute(
        plan,
        table_batch
    ));
    // Multi-SUM Tier-2: enabled with `MULTI_SUM_MIN_GROUPS = 100_000` floor
    // in the executor itself. Below 100K groups the global-atomic baseline
    // wins (q2 / 10K groups regressed 444 ms → 1.05 s when this path was
    // unconditional); the gate now lets q2 fall through cleanly while
    // capturing future workloads with more groups.
    try_fast_path!(crate::exec::groupby_tier2_multi_exec::try_execute(
        plan,
        table_batch
    ));
    // Two-key Tier-2: enabled now that `partition_reduce_kernel_i64`
    // replaces the host-HashMap pass-2 (Tier 2.1 for two-key).
    try_fast_path!(crate::exec::groupby_tier2_twokey_exec::try_execute(
        plan,
        table_batch
    ));
    // Two-key MULTI-aggregate Tier-2.1: `SELECT a, b, SUM(v1), SUM(v2)
    // FROM x GROUP BY a, b` — combines i64 partitioning with
    // multi-value reduce. Two-key single-SUM falls through to the line
    // above first.
    try_fast_path!(crate::exec::groupby_tier2_twokey_multi_exec::try_execute(
        plan,
        table_batch
    ));
    // AVG-at-Tier-2.1: SUM (via multi-SUM reduce) + COUNT (via count
    // reduce) → divide host-side. High-cardinality AVG over Float64.
    try_fast_path!(crate::exec::groupby_tier2_avg_exec::try_execute(
        plan,
        table_batch
    ));
    // Two-key multi-AVG Tier-2.1: `SELECT a, b, AVG(v1), AVG(v2), ...
    // FROM x GROUP BY a, b`. Same shape as the single-key AVG path but
    // with i64-packed (Int32, Int32) keys.
    try_fast_path!(crate::exec::groupby_tier2_twokey_avg_exec::try_execute(
        plan,
        table_batch
    ));
    // COUNT(*) at Tier-2.1: high-cardinality `SELECT k, COUNT(*) FROM x
    // GROUP BY k`. Reuses partition + scatter; one COUNT reduce launch.
    try_fast_path!(crate::exec::groupby_tier2_count_exec::try_execute(
        plan,
        table_batch
    ));
    // Two-key COUNT(*) Tier-2.1: `SELECT a, b, COUNT(*) FROM x GROUP BY
    // a, b`. Same shape as the single-key COUNT path but with i64-packed
    // (Int32, Int32) keys.
    try_fast_path!(crate::exec::groupby_tier2_twokey_count_exec::try_execute(
        plan,
        table_batch
    ));
    // Two-key integer MIN/MAX at Tier-2.1: `SELECT a, b, {MIN,MAX}(v)
    // FROM x GROUP BY a, b` with Int32 / Int64 value column. Routes
    // through partition_reduce_kernel_minmax_i64. Must come before the
    // single-key minmax path so the two-key shape isn't mishandled.
    try_fast_path!(crate::exec::groupby_tier2_twokey_minmax_exec::try_execute(
        plan,
        table_batch
    ));
    // Two-key float MIN/MAX at Tier-2.1: same shape, Float64 value
    // column, CAS-loop kernel (partition_reduce_kernel_minmax_float_i64).
    try_fast_path!(
        crate::exec::groupby_tier2_twokey_minmax_float_exec::try_execute(plan, table_batch)
    );
    // MIN/MAX at Tier-2.1: high-cardinality integer MIN/MAX. Float
    // MIN/MAX is deferred — needs a CAS-loop kernel and no workload
    // demands it yet.
    try_fast_path!(crate::exec::groupby_tier2_minmax_exec::try_execute(
        plan,
        table_batch
    ));
    // Float MIN/MAX: routes through partition_reduce_kernel_minmax_float
    // (CAS-loop kernel). Integer MIN/MAX above catches first; this
    // handles the float-value-column path.
    try_fast_path!(crate::exec::groupby_tier2_minmax_float_exec::try_execute(
        plan,
        table_batch
    ));
    // Tier-1 COUNT(*): low-cardinality COUNT GROUP BY.
    try_fast_path!(crate::exec::groupby_shmem_count_exec::try_execute(
        plan,
        table_batch
    ));
    // Tier-1 MIN/MAX: low-cardinality integer MIN/MAX. Float MIN/MAX
    // is deferred — needs a CAS-loop kernel.
    try_fast_path!(crate::exec::groupby_shmem_minmax_exec::try_execute(
        plan,
        table_batch
    ));

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
        let col_idx = table_batch.schema().index_of(&io.name).map_err(|_| {
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
    let mut key_valid = packed.key_valid;

    // NULL keys (NULL-group fix): SQL standard / DuckDB / Postgres group ALL
    // NULL-keyed rows into a SINGLE group whose key is NULL, with the
    // aggregates computed over those rows. The previous behaviour silently
    // DROPPED NULL-key rows entirely, which diverged from that semantics.
    //
    // We re-introduce them by REMAPPING each NULL-key row's encoded key to
    // the reserved `NULL_GROUP_KEY` sentinel (a real, non-empty hash-table
    // value) instead of filtering it out. Because the agg kernels read every
    // value column BY ROW INDEX and gate only on `key_valid` (key nulls) and
    // each column's own value-null mask, remapping the key and then clearing
    // `key_valid` to `None` makes the NULL-key rows flow through exactly like
    // any other group: they hash to one slot (`NULL_GROUP_KEY`), and the agg
    // kernels accumulate their values there. `build_key_arrays` later emits a
    // NULL (not a decoded value) for that slot's key column.
    //
    // SCOPE: this NULL-group synthesis is applied to SINGLE-column GROUP BY
    // only. For a multi-column key, the standard rule treats a NULL in each
    // column independently — `(NULL, 5)` and `(NULL, 6)` are DISTINCT groups —
    // but `pack_keys` collapses any per-column NULL into one combined keep
    // bit, so the per-column NULL structure is unrecoverable here. We keep the
    // previous drop-NULL-rows behaviour for the multi-column case and leave a
    // distinct-NULL-tuple grouping to the wide-key path / a follow-up. (The
    // common GROUP BY shape in practice is a single column.)
    let single_col = key_components.len() == 1;
    let synthesise_null_group = single_col && key_valid.is_some();

    // NULL-group sentinel collision (only relevant when we are about to
    // synthesise a NULL group): if a GENUINE (non-NULL) key already encodes
    // to `NULL_GROUP_KEY` (only possible for a single Int64 column literally
    // holding `i64::MAX`), we cannot tell that real key apart from the
    // synthesised NULL group. Route to the sentinel-free valid-flag executor.
    // NOTE: that executor currently DROPS NULL keys (it shares the legacy
    // behaviour), so this rare edge regresses to "no NULL group" rather than
    // producing a WRONG result — a correctness-preserving degradation,
    // flagged for the follow-up that teaches `groupby_valid` the same
    // NULL-group synthesis. We check only genuine keys (via the keep mask),
    // since NULL positions in `packed.keys_i64` carry undefined bits.
    if synthesise_null_group {
        let mask = key_valid
            .as_ref()
            .expect("synthesise_null_group implies Some");
        let genuine_hits_sentinel = packed
            .keys_i64
            .iter()
            .zip(mask.iter())
            .any(|(&k, &keep)| keep && k == NULL_GROUP_KEY);
        if genuine_hits_sentinel {
            log::warn!(
                "execute_groupby: a genuine GROUP BY key encodes to the reserved \
                 NULL-group sentinel (i64::MAX); routing to valid-flag executor \
                 (NULL group not synthesised on that path)"
            );
            return crate::exec::groupby_valid::execute_groupby_valid(plan, table_batch);
        }
    }

    let host_keys: Vec<i64> = if synthesise_null_group {
        // Single-column + nulls present: remap NULL-key rows to the NULL
        // sentinel, keep every row (preserve row alignment with the agg
        // value columns), and drop the now-redundant key-null mask.
        let mask = key_valid
            .as_ref()
            .expect("synthesise_null_group implies Some");
        let remapped: Vec<i64> = packed
            .keys_i64
            .iter()
            .zip(mask.iter())
            .map(|(&k, &keep)| if keep { k } else { NULL_GROUP_KEY })
            .collect();
        // After remapping, no row carries a NULL key, so the agg path must
        // NOT filter on key validity any more (the NULL group is now a real
        // key like any other). Value-NULL masks are handled independently
        // per aggregate input inside `run_one_aggregate`.
        key_valid = None;
        remapped
    } else {
        // Multi-column (NULL semantics scoped out — see above) or no nulls:
        // unchanged path. Drop NULL-key rows when a mask is present.
        match &key_valid {
            Some(mask) => packed
                .keys_i64
                .iter()
                .zip(mask.iter())
                .filter_map(|(k, &keep)| if keep { Some(*k) } else { None })
                .collect(),
            None => packed.keys_i64,
        }
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
        BoltError::Other(format!("GROUP BY hash table size {} exceeds u32::MAX", k))
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

    // Round 1 ABI: a single zero-initialised `u32` device counter shared by the
    // keys launch, every agg launch, and the decimal launch. Each classic
    // kernel atomically increments it when a probe-bound (or decimal lock-bound)
    // overflow would otherwise SILENTLY DROP a row. We read it back after the
    // final stream sync and error if non-zero, rather than returning a wrong
    // GROUP BY result. (The Robin Hood keys kernel and the float-atomics
    // MIN/MAX kernel were NOT changed and do not touch this counter.)
    let overflow_counter = GpuVec::<u32>::zeros_async(1, stream.raw())?;
    let overflow_ptr: CUdeviceptr = overflow_counter.device_ptr();

    // Launch the keys-only kernel.
    launch_keys_kernel(
        &key_col_gpu,
        &mut keys_table,
        n_rows,
        k_u32,
        &stream,
        overflow_ptr,
    )?;

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
    let any_input_has_validity: bool = aggregate.input_has_validity.iter().any(|&v| v);
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
            // PERF (filtered-keys round-trip): hoist the invariant key-column
            // host image OUT of the per-aggregate loop. `host_keys` is the
            // exact vec we uploaded into `key_col_gpu` above; every aggregate
            // shares that same device column, so its host image is identical
            // across the whole query. Passing it by reference lets
            // `prepare_filtered_keys` host-filter value-NULLs without a
            // per-aggregate D2H re-download of `key_col_gpu`.
            &host_keys,
            any_input_has_validity,
            overflow_ptr,
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

    // Round 1 ABI: after the final sync, read the shared overflow counter back.
    // A non-zero value means at least one row was dropped on a hash-probe
    // (or decimal lock) overflow, so the GROUP BY result is incomplete — error
    // rather than return a wrong aggregate.
    let overflow_count: u32 = {
        let pinned = overflow_counter.to_pinned_async(stream.raw())?;
        stream.synchronize()?;
        pinned.as_slice().first().copied().unwrap_or(0)
    };
    drop(overflow_counter);
    if overflow_count != 0 {
        return Err(BoltError::Other(format!(
            "GROUP BY result incomplete: {} rows dropped on hash-table overflow \
             (increase hash table size)",
            overflow_count
        )));
    }

    // Walk the keys table: every non-empty slot is a group. Build a list of
    // `(key, slot)` and sort by key for deterministic output ordering.
    let mut groups: Vec<(i64, usize)> = host_keys_table
        .iter()
        .enumerate()
        .filter_map(|(slot, &k)| {
            if k == EMPTY_KEY {
                None
            } else {
                Some((k, slot))
            }
        })
        .collect();
    groups.sort_unstable_by_key(|(k, _)| *k);

    // Assemble the output RecordBatch.
    let n_groups = groups.len();
    let m_keys = key_components.len();
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(m_keys + aggregate.aggregates.len());

    // Columns 0..M: one per group-by column, decoded from the packed i64 key.
    let key_arrays = build_key_arrays(&groups, &key_components)?;
    for arr in key_arrays {
        arrays.push(arr);
    }

    // Columns M..M+N: one per aggregate, taken from the corresponding
    // accumulator.
    for (i, agg) in aggregate.aggregates.iter().enumerate() {
        let out_field = aggregate
            .output_schema
            .fields
            .get(m_keys + i)
            .ok_or_else(|| {
                BoltError::Other(format!(
                    "execute_groupby: output_schema missing field for aggregate index {}",
                    i
                ))
            })?;
        let arr = build_agg_array(agg, out_field, &acc_results[i], &groups, n_groups)?;
        arrays.push(arr);
    }

    let arrow_schema = plan_schema_to_arrow_schema(&aggregate.output_schema)?;
    RecordBatch::try_new(arrow_schema, arrays)
        .map_err(|e| BoltError::Other(format!("failed to build GROUP BY RecordBatch: {e}")))
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
///
/// `overflow_ptr` is the device pointer of the shared single-`u32` overflow
/// counter (host-init `0`). The CLASSIC linear-probe kernel takes it as its
/// new trailing argument (Round 1 ABI: idx 4) and atomically increments it
/// when a probe-bound overflow would otherwise silently drop a row. The Robin
/// Hood variant was NOT changed and keeps its old 4-param ABI, so the counter
/// is only appended on the linear branch.
fn launch_keys_kernel(
    group_col: &GpuVec<i64>,
    keys_table: &mut GpuVec<i64>,
    n_rows: usize,
    k_u32: u32,
    stream: &CudaStream,
    overflow_ptr: CUdeviceptr,
) -> BoltResult<()> {
    if n_rows == 0 {
        // Nothing to insert; the empty keys table is already correct.
        return Ok(());
    }

    // Dispatch to classic linear-probe (default) or Robin Hood (when
    // BOLT_HASH_ALGO=robin_hood). The classic kernel now takes a trailing
    // `overflow_counter_ptr` (5 params); the Robin Hood kernel was NOT
    // changed and keeps the old 4-param ABI. Route through the consolidated
    // module cache so the two variants are cached separately by spec id.
    let want_rh = std::env::var("BOLT_HASH_ALGO")
        .map(|s| {
            let l = s.to_ascii_lowercase();
            l == "robin_hood" || l == "rh"
        })
        .unwrap_or(false);
    let (spec_id, kernel_entry) = if want_rh {
        (
            "groupby_keys_rh",
            crate::jit::hash_kernels::KEYS_KERNEL_RH_ENTRY,
        )
    } else {
        ("groupby_keys", KEYS_KERNEL_ENTRY)
    };
    let module =
        module_cache::get_or_build_module(module_path!(), spec_id.to_string(), None, || {
            let (ptx, _e) = compile_groupby_keys_kernel_dispatched()?;
            Ok(ptx)
        })?;
    let function = module.function(kernel_entry)?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;
    let mut overflow: CUdeviceptr = overflow_ptr;

    let block = groupby_block_size();
    let grid_x = grid_x_for(n_rows_u32, block);

    // The classic linear-probe kernel now takes the trailing overflow counter
    // (5 params); the Robin Hood kernel was NOT changed and keeps the 4-param
    // ABI. Attach the extra arg only on the linear branch.
    if want_rh {
        let mut params: [*mut c_void; 4] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
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
    } else {
        let mut params: [*mut c_void; 5] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
            &mut k_param as *mut u32 as *mut c_void,
            &mut overflow as *mut CUdeviceptr as *mut c_void,
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
    let _ = (group_ptr, keys_ptr, overflow);
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
///
/// `overflow_ptr` is the device pointer of the shared single-`u32` overflow
/// counter (host-init `0`). The classic integer agg kernel takes it as its
/// new trailing argument (idx 6 classic / idx 7 with-validity) and bumps it on
/// a probe-bound overflow that would otherwise drop a row silently. The float
/// MIN/MAX path routes through `float_atomics`, whose kernel was NOT changed
/// and keeps its old 6-param ABI — the counter is NOT attached there.
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
    overflow_ptr: CUdeviceptr,
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

    let mut overflow: CUdeviceptr = overflow_ptr;
    if let Some(vp) = validity_ptr {
        // Account the native-dispatch launch for inline-test observability.
        NATIVE_VALIDITY_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // With-validity ABI: 7 classic params + trailing overflow counter at
        // idx 7 = 8 params. (The native-validity kernel is the integer-atomic
        // emitter; float MIN/MAX never reaches this branch — see the dispatch
        // predicate.)
        let mut vptr: CUdeviceptr = vp;
        let mut params: [*mut c_void; 8] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut input_ptr as *mut CUdeviceptr as *mut c_void,
            &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
            &mut k_param as *mut u32 as *mut c_void,
            &mut vptr as *mut CUdeviceptr as *mut c_void,
            &mut overflow as *mut CUdeviceptr as *mut c_void,
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
    } else if is_float_min_max {
        // Float MIN/MAX uses the UNCHANGED `float_atomics` kernel (6 params,
        // no overflow counter). Do NOT attach the trailing arg here.
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
    } else {
        // Classic integer-atomic ABI: 6 classic params + trailing overflow
        // counter at idx 6 = 7 params.
        let mut params: [*mut c_void; 7] = [
            &mut group_ptr as *mut CUdeviceptr as *mut c_void,
            &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
            &mut input_ptr as *mut CUdeviceptr as *mut c_void,
            &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
            &mut n_rows_u32 as *mut u32 as *mut c_void,
            &mut k_param as *mut u32 as *mut c_void,
            &mut overflow as *mut CUdeviceptr as *mut c_void,
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
    let _ = (group_ptr, keys_ptr, input_ptr, acc_ptr, overflow);
    Ok(())
}

/// Launch the grouped Decimal128 SUM/MIN/MAX kernel
/// ([`compile_groupby_decimal_kernel`], entry [`AGG_DECIMAL_KERNEL_ENTRY`]).
///
/// Unlike [`launch_agg_kernel`] this carries a 16-byte (`i128`) per-slot
/// accumulator updated under a per-slot spin lock (sm_70 has no 128-bit
/// atomic), so it needs two extra device buffers: a `u32` lock table (one word
/// per slot, init 0) and a single `u32` overflow flag (init 0; SUM-only). The
/// 128-bit per-slot combine is bit-for-bit identical to the scalar decimal
/// path: carry-chain add for SUM (with signed-overflow detection that raises
/// the flag) and signed-hi/unsigned-lo compare-select for MIN/MAX.
///
/// Returns `Ok(true)` if a SUM overflow was detected on-device (the caller
/// raises the same error as the scalar/host path); `Ok(false)` otherwise.
/// MIN/MAX never overflow and always return `Ok(false)`.
///
/// `overflow_ptr` is the device pointer of the shared single-`u32` probe/lock
/// OVERFLOW COUNTER (host-init `0`), distinct from the per-launch SUM
/// signed-overflow FLAG allocated below. The decimal kernel takes it as its
/// new trailing argument (idx 8 classic) and bumps it on a probe-bound or
/// lock-bound bailout that would otherwise drop a row silently.
#[allow(clippy::too_many_arguments)]
fn launch_decimal_agg_kernel(
    op: GroupedDecimalOp,
    group_col: &GpuVec<i64>,
    keys_table: &GpuVec<i64>,
    input_col: &GpuVec<i128>,
    acc_table: &mut GpuVec<i128>,
    n_rows: usize,
    k: usize,
    k_u32: u32,
    stream: &CudaStream,
    overflow_ptr: CUdeviceptr,
) -> BoltResult<bool> {
    if n_rows == 0 {
        return Ok(false);
    }

    let module = module_cache::get_or_build_module(
        module_path!(),
        format!("groupby_agg_decimal:{:?}", op),
        None,
        || compile_groupby_decimal_kernel(op, false),
    )?;
    let function = module.function(AGG_DECIMAL_KERNEL_ENTRY)?;

    // Per-slot lock table (init 0) + single-word overflow flag (init 0).
    let lock_table = GpuVec::<u32>::zeros_async(k, stream.raw())?;
    let overflow_flag = GpuVec::<u32>::zeros_async(1, stream.raw())?;

    let mut group_ptr: CUdeviceptr = group_col.device_ptr();
    let mut keys_ptr: CUdeviceptr = keys_table.device_ptr();
    let mut input_ptr: CUdeviceptr = input_col.device_ptr();
    let mut acc_ptr: CUdeviceptr = acc_table.device_ptr();
    let mut n_rows_u32: u32 = n_rows_to_u32(n_rows)?;
    let mut k_param: u32 = k_u32;
    let mut lock_ptr: CUdeviceptr = lock_table.device_ptr();
    // param_7: the SUM signed-overflow FLAG (per-launch, distinct from the
    // shared probe/lock overflow counter passed in as `overflow_ptr`).
    let mut sum_overflow_flag_ptr: CUdeviceptr = overflow_flag.device_ptr();
    // param_8: the shared probe/lock OVERFLOW COUNTER (new Round 1 trailing arg).
    let mut overflow_counter: CUdeviceptr = overflow_ptr;

    let block = groupby_block_size();
    let grid_x = grid_x_for(n_rows_u32, block);

    let mut params: [*mut c_void; 9] = [
        &mut group_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut input_ptr as *mut CUdeviceptr as *mut c_void,
        &mut acc_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut k_param as *mut u32 as *mut c_void,
        &mut lock_ptr as *mut CUdeviceptr as *mut c_void,
        &mut sum_overflow_flag_ptr as *mut CUdeviceptr as *mut c_void,
        &mut overflow_counter as *mut CUdeviceptr as *mut c_void,
    ];
    // SAFETY: `function` is borrowed from a live module; every param points at
    // a stack local that outlives the synchronize below, and the lock/overflow
    // GpuVecs stay alive until after `stream.synchronize()`.
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

    // Read back the overflow flag (SUM only ever sets it).
    let flag_pinned = overflow_flag.to_pinned_async(stream.raw())?;
    stream.synchronize()?;
    let overflowed = flag_pinned.as_slice().first().copied().unwrap_or(0) != 0;
    drop(flag_pinned);
    let _ = (
        group_ptr,
        keys_ptr,
        input_ptr,
        acc_ptr,
        lock_ptr,
        sum_overflow_flag_ptr,
        overflow_counter,
    );
    drop(lock_table);
    drop(overflow_flag);
    Ok(overflowed)
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
    /// Decimal128 SUM/MIN/MAX result column (length `k`), carried as raw
    /// `i128`. `(precision, scale)` are the planner's declared OUTPUT dtype
    /// (SUM widens precision to 38 keeping scale; MIN/MAX preserve `(p, s)`);
    /// `build_agg_array` rebuilds a `Decimal128Array` with that dtype. Slots
    /// for groups with no contributing rows hold the op's identity
    /// (`0` / `i128::MAX` / `i128::MIN`), but every emitted group has at least
    /// one row so the identity never surfaces.
    Decimal128 {
        /// Per-slot raw `i128` accumulator table (length `k`).
        acc: Vec<i128>,
        /// Declared output precision (SUM => 38; MIN/MAX => input precision).
        precision: u8,
        /// Declared output scale (== input scale for all three ops).
        scale: i8,
    },
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
    Welford {
        states: Vec<crate::exec::welford::WelfordState>,
    },
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
    // PERF (filtered-keys round-trip): per-query host image of the shared key
    // column (`execute_groupby`'s `host_keys`). Invariant across every
    // aggregate, so it is hoisted out of the per-aggregate loop and threaded
    // here to feed `prepare_filtered_keys` without a per-aggregate D2H.
    cached_host_keys: &[i64],
    any_input_has_validity: bool,
    // Round 1 ABI: device pointer of the shared overflow counter, forwarded to
    // every classic kernel launch this aggregate issues.
    overflow_ptr: CUdeviceptr,
) -> BoltResult<AccDownload> {
    match agg {
        AggregateExpr::Sum(expr) | AggregateExpr::Min(expr) | AggregateExpr::Max(expr) => {
            let op = ReduceOp::from_agg(agg)?;
            let col_name = bare_column_name(expr)?;
            let col_io = resolve_input(inputs, col_name)?;
            run_typed_agg(
                op,
                col_io,
                group_col,
                keys_table,
                batch,
                n_rows,
                k,
                k_u32,
                stream,
                key_valid,
                cached_host_keys,
                any_input_has_validity,
                overflow_ptr,
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

            let filtered = prepare_filtered_keys(
                group_col,
                n_rows,
                key_valid,
                value_valid.as_deref(),
                cached_host_keys,
                stream,
            )?;
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
                overflow_ptr,
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
                col_io, group_col, keys_table, batch, n_rows, k, k_u32, stream, key_valid,
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
            let filtered = prepare_filtered_keys(
                group_col,
                n_rows,
                key_valid,
                value_valid.as_deref(),
                cached_host_keys,
                stream,
            )?;
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
                overflow_ptr,
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
                overflow_ptr,
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
    let values_f64 =
        load_input_column_as_f64_filtered(col_io, batch, key_valid, value_valid.as_deref())?;

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
fn column_null_mask(col_io: &ColumnIO, batch: &RecordBatch) -> BoltResult<Option<Vec<bool>>> {
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
    /// Reuse the shared post-key-filter key column. `host_keys` borrows the
    /// per-query cached host image of that same column (bit-identical to a
    /// D2H of `group_col`) so the host-side overflow check can read the
    /// per-row group assignment without a device round-trip.
    Borrowed {
        group_col: &'a GpuVec<i64>,
        host_keys: &'a [i64],
        n_rows: usize,
    },
    /// Freshly-uploaded smaller column applying a value-NULL filter on top
    /// of `key_valid`. The owned vec must live across the kernel launch.
    /// `host_keys` owns the matching host image used to build it.
    Owned {
        group_col: GpuVec<i64>,
        host_keys: Vec<i64>,
        n_rows: usize,
    },
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
    /// Host image of the per-row group-key column, aligned 1:1 with the
    /// filtered value vec produced by `collect_filtered_primitive`. Used by
    /// the host-side SUM overflow check to replicate the kernel's grouping.
    fn host_keys(&self) -> &[i64] {
        match self {
            FilteredKeys::Borrowed { host_keys, .. } => host_keys,
            FilteredKeys::Owned { host_keys, .. } => host_keys,
        }
    }
}

/// V-10 (grouped): host-side integer SUM overflow guard for GROUP BY.
///
/// The GPU group-by SUM accumulates each group with `atom.global.add.u64`
/// (PTX has no signed `atom.add`; `.u64` is bit-identical for two's-complement
/// addition). That atomic WRAPS modulo 2^64 on overflow, so a per-group sum
/// exceeding `i64::MAX` would silently come back as a wrapped (often negative)
/// value. The SCALAR SUM path in `aggregate.rs` instead ERRORS on i64 overflow
/// via `i64::checked_add` (see `ReduceScalar::finalize`, the
/// `"SUM(integer) overflow"` arm) to honour the engine's "never silently
/// wrong" invariant. Without this guard, `SELECT SUM(x) FROM t` would error on
/// overflow while `SELECT k, SUM(x) FROM t GROUP BY k` over the SAME data
/// silently wrapped — an inconsistency this restores.
///
/// We cannot change the kernel PTX from this file, so we replicate the
/// grouping on the host: `host_keys[i]` is the group key of `values[i]`
/// (the two vecs are aligned 1:1 — both are produced from the same post-filter
/// row set). We fold each group with `i64::checked_add`; the first group whose
/// true sum leaves i64 range yields a `BoltError::Type` whose message mirrors
/// the scalar path verbatim. Groups that do not overflow are untouched, so a
/// correct (non-overflowing) query is never regressed — the kernel result is
/// returned unchanged.
///
/// This pass is the exact arithmetic the kernel performs (integer addition is
/// associative and commutative, so any accumulation order reaches the same i64
/// result and overflows iff the device sum overflowed), making the detection
/// faithful rather than a conservative magnitude bound. It costs one O(n) host
/// pass plus an O(#groups) map; acceptable next to the H2D/D2H already on this
/// path.
///
/// NOTE (kernel follow-up): the device atomic is still wrapping; this host
/// guard is the build-safe restoration of the invariant. The proper long-term
/// fix is an overflow flag raised inside the atomic kernel.
// COVERAGE (integer SUM overflow, as of V-10): every reachable integer-SUM
// finalization in the engine now raises the canonical "SUM(integer) overflow"
// error rather than silently wrapping:
//   - grouped SUM(Int64) / SUM(Int32->i64), host-strip path: guarded here via
//     `checked_group_sum` in `run_typed_agg` (the values are host-materialised
//     by `collect_filtered_primitive`, so the re-fold is free of an extra D2H);
//   - grouped SUM, native-validity path: guarded via
//     `checked_group_sum_native_validity` in `run_typed_agg_native_validity`;
//   - the grouped shared-mem fast path is gated to (Sum, Float64) at
//     `groupby_shmem_dispatch.rs:~149`, so integer grouped SUM never reaches an
//     on-device-only finalize and always falls through to the guarded paths
//     above;
//   - scalar (non-grouped) integer SUM: `aggregate.rs`'s and `agg_with_pre.rs`'s
//     `ReduceScalar for {i32,i64}` finalize the device partials with
//     `checked_add` (agg_with_pre.rs was fixed in V-10 — it previously merged
//     `Sum | Count` into `wrapping_add` and wrapped silently).
// RESIDUAL/FUTURE GAP: the device group atomic (`atom.global.add.u64`) is still
// itself wrapping; the host re-fold here is the build-safe restoration of the
// invariant and depends on the grouped values being host-materialised. A
// future never-materialised streaming grouped aggregate would bypass it.
// TODO(overflow-kernel): add a saturating/overflow-detecting variant of the
// u64 group SUM atomic (e.g. a per-launch device "overflow" flag set on signed
// carry) so overflow is caught on-device for that future streaming case and the
// host re-fold can be dropped.
fn checked_group_sum(values: &[i64], host_keys: &[i64]) -> BoltResult<()> {
    debug_assert_eq!(
        values.len(),
        host_keys.len(),
        "checked_group_sum: value/key columns must be aligned 1:1",
    );
    let mut sums: std::collections::HashMap<i64, i64> =
        std::collections::HashMap::with_capacity(host_keys.len().min(1 << 20));
    for (&v, &key) in values.iter().zip(host_keys.iter()) {
        let slot = sums.entry(key).or_insert(0i64);
        *slot = match slot.checked_add(v) {
            Some(s) => s,
            None => {
                // Mirror the scalar contract's message verbatim
                // (`aggregate.rs` ReduceScalar::finalize SUM arm) so callers
                // and tests see one canonical integer-overflow error string.
                return Err(BoltError::Type(
                    "SUM(integer) overflow: accumulator exceeds i64 range".to_string(),
                ));
            }
        };
    }
    Ok(())
}

/// V-10 (grouped, native-validity path): overflow guard for the
/// `run_typed_agg_native_validity` SUM branches.
///
/// That path uploads the FULL value column (`values`, length `n_rows`) plus a
/// device validity bitmap and lets the kernel skip NULL rows on-device, so it
/// never builds a host-compacted value vec. To replicate the kernel's grouping
/// for the overflow check we download the per-row key column host image
/// (`group_col`, parallel to the batch by row index — guaranteed by the
/// `key_valid.is_none()` precondition at the dispatch site) and fold per group
/// with `i64::checked_add`, skipping rows where `value_valid[i]` is false
/// exactly as the kernel does. Returns the same `BoltError::Type` as the
/// host-strip path on overflow.
///
/// The single D2H of the key column here is the cost of restoring the
/// invariant on this path; see the `// TODO(overflow-kernel)` on
/// `checked_group_sum` for the on-device follow-up that would remove it.
fn checked_group_sum_native_validity(
    values: &[i64],
    value_valid: &[bool],
    group_col: &GpuVec<i64>,
) -> BoltResult<()> {
    debug_assert_eq!(values.len(), value_valid.len());
    let host_keys: Vec<i64> = group_col.to_vec()?;
    debug_assert_eq!(host_keys.len(), values.len());
    // Compact out NULL value rows so the host fold sees exactly the rows the
    // kernel accumulates.
    let mut vals: Vec<i64> = Vec::with_capacity(values.len());
    let mut keys: Vec<i64> = Vec::with_capacity(values.len());
    for ((&v, &valid), &key) in values.iter().zip(value_valid.iter()).zip(host_keys.iter()) {
        if valid {
            vals.push(v);
            keys.push(key);
        }
    }
    checked_group_sum(&vals, &keys)
}

/// Decide whether to reuse the shared `group_col` (when no value-NULL filter
/// shrinks the row set further) or upload a freshly-filtered key column for
/// this aggregate. The shared `group_col` was built from a `host_keys` that
/// already had the `key_valid` rows kept; if `value_valid` is `None` we can
/// reuse it directly. Otherwise we refilter against the joint mask on the host
/// and upload a fresh i64 column.
///
/// PERF (filtered-keys round-trip): the host image of the shared key column is
/// INVARIANT across every aggregate in a query — they all reuse the same
/// `key_col_gpu`, which `execute_groupby` uploaded from the per-query
/// `host_keys` vec. Previously this function re-downloaded that identical
/// column (one D2H per NULL-bearing aggregate) only to host-filter it; that
/// extra D2H+H2D round-trip was redundant work re-done for the SAME keys. We
/// now thread the already-resident `host_keys` slice in via `cached_host_keys`
/// and host-filter directly off it, so the per-query D2H of the key column
/// happens ZERO times here (it never needed to leave the host in the first
/// place). The genuinely per-aggregate part — projecting the value-column NULL
/// mask through `key_valid` and compacting the keys against it — is unchanged
/// and still produces a per-aggregate `Owned` column with exactly the same
/// rows as before.
///
/// `cached_host_keys` MUST be the same host vec that produced `group_col`
/// (i.e. `execute_groupby`'s post-key-filter `host_keys`); it is therefore
/// length `n_rows` and bit-identical to what a D2H of `group_col` would yield.
///
/// v0.7 async-memcpy: the HtoD of the freshly-filtered keys still rides
/// `stream` via `from_slice_async`, mirroring the per-aggregate buffer
/// plumbing in `run_typed_agg`, so the upload chains behind the previous
/// kernel's D2H and overlaps with unrelated stream activity.
fn prepare_filtered_keys<'a>(
    group_col: &'a GpuVec<i64>,
    n_rows: usize,
    key_valid: Option<&[bool]>,
    value_valid: Option<&[bool]>,
    // Tied to `'a` so the `Borrowed` variant can hold this slice as the
    // per-row group host image for the V-10 overflow check (it lives as long
    // as the borrowed `group_col`, which is what the returned value borrows).
    cached_host_keys: &'a [i64],
    stream: &CudaStream,
) -> BoltResult<FilteredKeys<'a>> {
    if value_valid.is_none() {
        return Ok(FilteredKeys::Borrowed {
            group_col,
            host_keys: cached_host_keys,
            n_rows,
        });
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

    // PERF (filtered-keys round-trip): host-filter directly off the cached,
    // already-resident `host_keys` slice instead of re-downloading the shared
    // key column from the device. `cached_host_keys` is the very vec
    // `execute_groupby` uploaded into `group_col`, so it is bit-identical to a
    // D2H of `group_col` would-be result — we just skip the trip. This elides
    // the per-aggregate D2H (and the prior pinned-buffer hop) entirely; only
    // the genuinely per-aggregate H2D of the freshly-compacted column below
    // remains. The debug assert pins the invariant that the cache matches the
    // device column's length (== post-key-filter `n_rows`).
    debug_assert_eq!(cached_host_keys.len(), n_rows);
    let filtered: Vec<i64> = cached_host_keys
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
    Ok(FilteredKeys::Owned {
        group_col: owned,
        host_keys: filtered,
        n_rows: filtered_n,
    })
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
    // PERF (filtered-keys round-trip): per-query host image of the shared key
    // column, forwarded straight to `prepare_filtered_keys` so the value-NULL
    // host-filter reuses it instead of re-downloading `group_col`.
    cached_host_keys: &[i64],
    any_input_has_validity: bool,
    // Round 1 ABI: device pointer of the shared overflow counter.
    overflow_ptr: CUdeviceptr,
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
            overflow_ptr,
        );
    }

    let filtered = prepare_filtered_keys(
        group_col,
        n_rows,
        key_valid,
        value_valid.as_deref(),
        cached_host_keys,
        stream,
    )?;
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
                let host: Vec<i64> =
                    collect_filtered_primitive(pa, key_valid, value_valid.as_deref())
                        .into_iter()
                        .map(|v| v as i64)
                        .collect();
                debug_assert_eq!(host.len(), n);
                // V-10 (grouped): error on per-group i64 overflow before the
                // wrapping u64 atomic can silently produce a wrong answer,
                // matching the scalar SUM contract. SUM(Int32->i64) cannot
                // actually overflow i64 for realistic row counts, but we run
                // the same guard for contract uniformity — it never rejects a
                // correct result.
                checked_group_sum(&host, filtered.host_keys())?;
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
                    overflow_ptr,
                )?;
                Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?))
            } else {
                let host: Vec<i32> =
                    collect_filtered_primitive(pa, key_valid, value_valid.as_deref());
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
                    overflow_ptr,
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
            // V-10 (grouped): for SUM, error on any per-group i64 overflow
            // BEFORE the wrapping `atom.global.add.u64` group atomic can
            // silently return a wrapped result — matching the scalar SUM
            // path in aggregate.rs. MIN/MAX are non-arithmetic and skipped.
            if matches!(op, ReduceOp::Sum) {
                checked_group_sum(&host, filtered.host_keys())?;
            }
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
                overflow_ptr,
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
                overflow_ptr,
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
                overflow_ptr,
            )?;
            Ok(AccDownload::F64(download_pinned_f64(&acc, stream)?))
        }
        // F7-finish: Date32 normalises to an i32 grouped reduction. MIN/MAX
        // preserve the date type — the result is rebuilt as a `Date32Array` by
        // `pack_array` keyed on `out_field.dtype == Date32`. SUM over a date is
        // meaningless and stays rejected. (COUNT(temporal) routes through the
        // dedicated count path, not here.)
        DataType::Date32 => {
            if matches!(op, ReduceOp::Sum) {
                return Err(BoltError::Type(format!(
                    "SUM over Date32 is not supported (column '{}')",
                    col_io.name
                )));
            }
            let pa = arr
                .as_any()
                .downcast_ref::<arrow_array::Date32Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Date32"))?;
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
                overflow_ptr,
            )?;
            Ok(AccDownload::I32(download_pinned_i32(&acc, stream)?))
        }
        // F7-finish: Timestamp normalises to an i64 grouped reduction. MIN/MAX
        // preserve the unit + timezone — rebuilt as the concrete
        // `Timestamp*Array` by `pack_array` keyed on `out_field.dtype`. SUM
        // over a timestamp is meaningless and stays rejected.
        DataType::Timestamp(_, _) => {
            if matches!(op, ReduceOp::Sum) {
                return Err(BoltError::Type(format!(
                    "SUM over Timestamp is not supported (column '{}')",
                    col_io.name
                )));
            }
            let host: Vec<i64> = collect_filtered_timestamp(
                arr.as_ref(),
                &col_io.name,
                key_valid,
                value_valid.as_deref(),
            )?;
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
                overflow_ptr,
            )?;
            Ok(AccDownload::I64(download_pinned_i64(&acc, stream)?))
        }
        // Grouped Decimal128 SUM/MIN/MAX run ON THE GPU via the per-slot-lock
        // 128-bit accumulator kernel (`compile_groupby_decimal_kernel`,
        // `launch_decimal_agg_kernel`). sm_70 has no 128-bit atomic, so a
        // per-slot spin lock guards an exclusive read-modify-write of the
        // 16-byte slot; the combine (carry-chain add for SUM, signed-hi /
        // unsigned-lo compare-select for MIN/MAX) is bit-for-bit identical to
        // the scalar decimal path in `crate::jit::decimal_agg`, so GPU and host
        // agree on value, ordering, and overflow. Result dtype mirrors the
        // planner / host grouped Decimal rule (see
        // `crate::plan::logical_plan::sum_output_dtype` and
        // `aggregate.rs::minmax_decimal128_from_batch`):
        //   * SUM     -> Decimal128(38, s)  (precision widened to 38, scale kept)
        //   * MIN/MAX -> Decimal128(p,  s)  (preserved)
        // NULLs are excluded by the host-strip in `collect_filtered_primitive`
        // (key-NULL ∧ value-NULL). SUM overflow of the i128 accumulator is
        // detected on-device and raised here as the same error the scalar/host
        // SUM path raises. COUNT(Decimal128) never reaches here (it is a plain
        // i64 count-of-ones). Bool / Utf8 have no kernel and stay rejected.
        DataType::Decimal128(p, s) => {
            let dec_op = GroupedDecimalOp::from_reduce_op(op)?;
            let pa = arr
                .as_any()
                .downcast_ref::<Decimal128Array>()
                .ok_or_else(|| downcast_err(&col_io.name, "Decimal128"))?;
            // Guard: the Arrow array's own (p, s) must match the plan dtype,
            // mirroring the scalar `decimal_sum_from_batch` /
            // `minmax_decimal128_from_batch` guards (a scale mismatch would
            // silently misinterpret the raw i128s).
            if let ArrowDataType::Decimal128(ap, as_) = pa.data_type() {
                if *ap != p || *as_ != s {
                    return Err(BoltError::Type(format!(
                        "GROUP BY {:?}(Decimal128) column '{}': plan dtype Decimal128({p}, {s}) \
                         disagrees with Arrow dtype Decimal128({ap}, {as_})",
                        op, col_io.name
                    )));
                }
            }
            // Dense NULL-stripped host i128 column (key-NULL ∧ value-NULL drop),
            // exactly the rows the kernel will accumulate.
            let host: Vec<i128> = collect_filtered_primitive::<arrow_array::types::Decimal128Type>(
                pa,
                key_valid,
                value_valid.as_deref(),
            );
            debug_assert_eq!(host.len(), n);

            // Output dtype per the engine's grouped Decimal rule.
            let (out_p, out_s) = match op {
                ReduceOp::Sum => match sum_output_dtype(DataType::Decimal128(p, s)) {
                    DataType::Decimal128(wp, ws) => (wp, ws),
                    other => {
                        return Err(BoltError::Type(format!(
                            "SUM(Decimal128) output dtype expected Decimal128, got {:?}",
                            other
                        )))
                    }
                },
                // MIN/MAX preserve (p, s).
                ReduceOp::Min | ReduceOp::Max => (p, s),
                ReduceOp::Count => unreachable!("Count never routes through run_typed_agg"),
            };

            // Per-slot accumulator identity:
            //   SUM -> 0 ; MIN -> i128::MAX ; MAX -> i128::MIN.
            let identity: i128 = match dec_op {
                GroupedDecimalOp::Sum => 0,
                GroupedDecimalOp::Min => i128::MAX,
                GroupedDecimalOp::Max => i128::MIN,
            };
            let input_gpu = GpuVec::<i128>::from_slice_async(&host, stream.raw())?;
            let init: Vec<i128> = vec![identity; k];
            let mut acc = GpuVec::<i128>::from_slice_async(&init, stream.raw())?;
            let overflowed = launch_decimal_agg_kernel(
                dec_op,
                filtered.col(),
                keys_table,
                &input_gpu,
                &mut acc,
                n,
                k,
                k_u32,
                stream,
                overflow_ptr,
            )?;
            if overflowed {
                // Same canonical message as the scalar/host SUM(Decimal128)
                // overflow path (`aggregate.rs::decimal_sum_host`).
                return Err(BoltError::Type(
                    "SUM(Decimal128) precision overflow: accumulator exceeds i128 range"
                        .to_string(),
                ));
            }
            let acc_host: Vec<i128> = {
                let pinned = acc.to_pinned_async(stream.raw())?;
                stream.synchronize()?;
                pinned.as_slice().to_vec()
            };
            Ok(AccDownload::Decimal128 {
                acc: acc_host,
                precision: out_p,
                scale: out_s,
            })
        }
        DataType::Bool | DataType::Utf8 => Err(BoltError::Type(format!(
            "aggregate input dtype {:?} not supported (column '{}')",
            col_io.dtype, col_io.name
        ))),
    }
}

/// F7-finish: NULL-filtered i64 tick extraction for a grouped MIN/MAX over a
/// `Timestamp*Array`. Dispatches on the concrete Arrow timestamp unit and
/// reuses [`collect_filtered_primitive`] (which drops rows where either the
/// key or the value is NULL) per unit. Returns an owned `Vec<i64>` because the
/// concrete array type differs per unit.
fn collect_filtered_timestamp(
    arr: &dyn arrow_array::Array,
    name: &str,
    key_valid: Option<&[bool]>,
    value_valid: Option<&[bool]>,
) -> BoltResult<Vec<i64>> {
    use arrow_array::types::{
        TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType,
    };
    use arrow_schema::{DataType as ArrowDataType, TimeUnit as ArrowTimeUnit};
    macro_rules! collect_ts {
        ($p:ty, $arr_ty:ty, $label:literal) => {{
            let pa = arr
                .as_any()
                .downcast_ref::<$arr_ty>()
                .ok_or_else(|| downcast_err(name, $label))?;
            Ok(collect_filtered_primitive::<$p>(pa, key_valid, value_valid))
        }};
    }
    match arr.data_type() {
        ArrowDataType::Timestamp(ArrowTimeUnit::Second, _) => {
            collect_ts!(
                TimestampSecondType,
                arrow_array::TimestampSecondArray,
                "TimestampSecond"
            )
        }
        ArrowDataType::Timestamp(ArrowTimeUnit::Millisecond, _) => collect_ts!(
            TimestampMillisecondType,
            arrow_array::TimestampMillisecondArray,
            "TimestampMillisecond"
        ),
        ArrowDataType::Timestamp(ArrowTimeUnit::Microsecond, _) => collect_ts!(
            TimestampMicrosecondType,
            arrow_array::TimestampMicrosecondArray,
            "TimestampMicrosecond"
        ),
        ArrowDataType::Timestamp(ArrowTimeUnit::Nanosecond, _) => collect_ts!(
            TimestampNanosecondType,
            arrow_array::TimestampNanosecondArray,
            "TimestampNanosecond"
        ),
        other => Err(BoltError::Type(format!(
            "GROUP BY aggregate input '{}' is not a Timestamp array: {:?}",
            name, other
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
    // Round 1 ABI: device pointer of the shared overflow counter.
    overflow_ptr: CUdeviceptr,
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
                // V-10 (grouped, native-validity path): same overflow contract
                // as the host-strip SUM path — the device atomic wraps, so
                // re-fold per group on the host skipping NULL rows (the kernel
                // skips them via the validity bitmap) and error on i64
                // overflow. SUM(Int32->i64) cannot realistically overflow, but
                // we run the guard for contract uniformity.
                checked_group_sum_native_validity(&widened, value_valid, group_col)?;
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
                    overflow_ptr,
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
                    overflow_ptr,
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
            // V-10 (grouped, native-validity path): for SUM, error on per-group
            // i64 overflow before the wrapping device atomic can silently wrap.
            // MIN/MAX are non-arithmetic and skip the guard.
            if matches!(op, ReduceOp::Sum) {
                checked_group_sum_native_validity(&host, value_valid, group_col)?;
            }
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
                overflow_ptr,
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
                overflow_ptr,
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
                overflow_ptr,
            )?;
            let _ = validity_gpu;
            Ok(AccDownload::F64(download_pinned_f64(&acc, stream)?))
        }
        DataType::Bool
        | DataType::Utf8
        | DataType::Decimal128(_, _)
        | DataType::Date32
        | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
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
        DataType::Bool
        | DataType::Utf8
        | DataType::Decimal128(_, _)
        | DataType::Date32
        | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
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
            DataType::Bool
            | DataType::Utf8
            | DataType::Decimal128(_, _)
            | DataType::Date32
            | DataType::Timestamp(_, _) => {
                return Err(BoltError::Type(format!(
                    "GROUP BY key dtype {:?} not supported on output",
                    comp.original_dtype
                )))
            }
        }
    }

    // NULL-group fix: per-group validity for the key column(s). A group whose
    // encoded key is the reserved `NULL_GROUP_KEY` sentinel is the synthesised
    // SQL NULL group — its key value must be emitted as NULL, not decoded.
    // (Only the single-column path synthesises this sentinel; see
    // `execute_groupby`.) For genuine groups every key column is valid. We
    // still push a placeholder value into the typed buffer at the NULL slot
    // so the buffer length matches `n`; the validity mask masks it out.
    let mut row_valid: Vec<bool> = Vec::with_capacity(n);
    let mut any_null_group = false;

    for (k, _) in groups {
        let is_null_group = *k == NULL_GROUP_KEY;
        row_valid.push(!is_null_group);
        if is_null_group {
            any_null_group = true;
            // Push a placeholder (zero / 0.0) into each typed buffer; it is
            // never observed because the validity mask marks this row NULL.
            for buf in buffers.iter_mut() {
                match buf {
                    ColBuf::I32(v) => v.push(0),
                    ColBuf::I64(v) => v.push(0),
                    ColBuf::F32(v) => v.push(0.0),
                    ColBuf::F64(v) => v.push(0.0),
                }
            }
            continue;
        }
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

    // Build each output array. When a NULL group is present we attach the
    // per-row validity mask so its key cell is NULL; otherwise we build the
    // dense (all-valid) array exactly as before (no behaviour change for the
    // common no-NULL-key case). `*Array::from(Vec<Option<_>>)` threads the
    // validity buffer through Arrow's builders.
    let mut out: Vec<ArrayRef> = Vec::with_capacity(m);
    for buf in buffers {
        let arr: ArrayRef = if any_null_group {
            match buf {
                ColBuf::I32(v) => Arc::new(Int32Array::from(
                    v.into_iter()
                        .zip(row_valid.iter())
                        .map(|(x, &ok)| if ok { Some(x) } else { None })
                        .collect::<Vec<Option<i32>>>(),
                )) as ArrayRef,
                ColBuf::I64(v) => Arc::new(Int64Array::from(
                    v.into_iter()
                        .zip(row_valid.iter())
                        .map(|(x, &ok)| if ok { Some(x) } else { None })
                        .collect::<Vec<Option<i64>>>(),
                )) as ArrayRef,
                ColBuf::F32(v) => Arc::new(Float32Array::from(
                    v.into_iter()
                        .zip(row_valid.iter())
                        .map(|(x, &ok)| if ok { Some(x) } else { None })
                        .collect::<Vec<Option<f32>>>(),
                )) as ArrayRef,
                ColBuf::F64(v) => Arc::new(Float64Array::from(
                    v.into_iter()
                        .zip(row_valid.iter())
                        .map(|(x, &ok)| if ok { Some(x) } else { None })
                        .collect::<Vec<Option<f64>>>(),
                )) as ArrayRef,
            }
        } else {
            match buf {
                ColBuf::I32(v) => Arc::new(Int32Array::from(v)) as ArrayRef,
                ColBuf::I64(v) => Arc::new(Int64Array::from(v)) as ArrayRef,
                ColBuf::F32(v) => Arc::new(Float32Array::from(v)) as ArrayRef,
                ColBuf::F64(v) => Arc::new(Float64Array::from(v)) as ArrayRef,
            }
        };
        out.push(arr);
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
        // Grouped Decimal128 SUM/MIN/MAX: build a `Decimal128Array` directly
        // from the per-group raw i128 accumulator slots, tagged with the
        // planner's declared output `(precision, scale)`. Validate that the
        // output field agrees, mirroring the scalar
        // `minmax_decimal128_from_batch` / `decimal_sum_from_batch` guards.
        (
            AggregateExpr::Sum(_) | AggregateExpr::Min(_) | AggregateExpr::Max(_),
            AccDownload::Decimal128 {
                acc,
                precision,
                scale,
            },
        ) => {
            let (out_p, out_s) = match out_field.dtype {
                DataType::Decimal128(p, s) => (p, s),
                ref other => {
                    return Err(BoltError::Type(format!(
                        "GROUP BY Decimal128 aggregate output dtype must be Decimal128, got {:?}",
                        other
                    )))
                }
            };
            if out_p != *precision || out_s != *scale {
                return Err(BoltError::Type(format!(
                    "GROUP BY Decimal128 output dtype Decimal128({out_p}, {out_s}) disagrees \
                     with computed Decimal128({precision}, {scale})"
                )));
            }
            let vals: Vec<i128> = groups.iter().map(|(_, slot)| acc[*slot]).collect();
            let arr = Decimal128Array::from(vals)
                .with_precision_and_scale(out_p, out_s)
                .map_err(|e| {
                    BoltError::Type(format!(
                        "GROUP BY Decimal128 result: precision/scale ({out_p}, {out_s}) \
                         invalid: {e}"
                    ))
                })?;
            Ok(Arc::new(arr) as ArrayRef)
        }
        (AggregateExpr::Sum(_) | AggregateExpr::Min(_) | AggregateExpr::Max(_), other) => {
            let scalars = match other {
                AccDownload::I32(host) => {
                    Scalars::I32(groups.iter().map(|(_, slot)| host[*slot]).collect())
                }
                AccDownload::I64(host) => {
                    Scalars::I64(groups.iter().map(|(_, slot)| host[*slot]).collect())
                }
                AccDownload::F32(host) => {
                    Scalars::F32(groups.iter().map(|(_, slot)| host[*slot]).collect())
                }
                AccDownload::F64(host) => {
                    Scalars::F64(groups.iter().map(|(_, slot)| host[*slot]).collect())
                }
                AccDownload::Avg { .. } => {
                    return Err(BoltError::Other(
                        "internal: AVG accumulator passed to non-AVG aggregate".into(),
                    ))
                }
                AccDownload::Welford { .. } => {
                    return Err(BoltError::Other(
                        "internal: Welford accumulator passed to SUM/MIN/MAX aggregate".into(),
                    ))
                }
                // Decimal128 is handled by the dedicated arm above; reaching
                // here means the pattern order was broken.
                AccDownload::Decimal128 { .. } => {
                    return Err(BoltError::Other(
                        "internal: Decimal128 accumulator should be handled by the \
                         dedicated build_agg_array arm"
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
            "internal: VAR/STDDEV aggregate received a non-Welford accumulator".into(),
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
        (Scalars::F32(v), DataType::Float32) => Ok(Arc::new(Float32Array::from(v)) as ArrayRef),
        (Scalars::F64(v), DataType::Float64) => Ok(Arc::new(Float64Array::from(v)) as ArrayRef),

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

        // F7-finish: MIN/MAX(Date32) reduced on the i32-normalized storage;
        // rebuild a `Date32Array` so the grouped result preserves the date
        // type (a plain `Int32Array` would silently downgrade it).
        (Scalars::I32(v), DataType::Date32) => {
            Ok(Arc::new(arrow_array::Date32Array::from(v)) as ArrayRef)
        }
        // F7-finish: MIN/MAX(Timestamp) reduced on the i64-normalized storage;
        // rebuild the concrete `Timestamp*Array` matching the output unit and
        // reattach its timezone so the unit + tz survive end-to-end.
        (Scalars::I64(v), DataType::Timestamp(unit, tz)) => {
            Ok(timestamp_array_from_i64(v, unit, tz))
        }

        (_, dt) => Err(BoltError::Type(format!(
            "GROUP BY: cannot pack scalars into output dtype {:?}",
            dt
        ))),
    }
}

/// F7-finish: build the concrete Arrow `Timestamp*Array` for a per-group i64
/// tick buffer, dispatching on `TimeUnit` and reattaching the optional
/// timezone. Mirror of `gpu_compact::timestamp_array_from_i64`, kept local so
/// `pack_array` can rebuild the grouped MIN/MAX result with the correct unit +
/// tz.
fn timestamp_array_from_i64(host: Vec<i64>, unit: TimeUnit, tz: Option<&'static str>) -> ArrayRef {
    let tz_owned: Option<Arc<str>> = tz.map(Arc::from);
    match unit {
        TimeUnit::Second => {
            Arc::new(arrow_array::TimestampSecondArray::from(host).with_timezone_opt(tz_owned))
                as ArrayRef
        }
        TimeUnit::Millisecond => {
            Arc::new(arrow_array::TimestampMillisecondArray::from(host).with_timezone_opt(tz_owned))
                as ArrayRef
        }
        TimeUnit::Microsecond => {
            Arc::new(arrow_array::TimestampMicrosecondArray::from(host).with_timezone_opt(tz_owned))
                as ArrayRef
        }
        TimeUnit::Nanosecond => {
            Arc::new(arrow_array::TimestampNanosecondArray::from(host).with_timezone_opt(tz_owned))
                as ArrayRef
        }
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

// MIN float identity is NaN (not +inf): the no-validity float MIN/MAX path
// (`run_min_max_aggregate`) seeds these accumulators and reduces them on the
// device via the NaN-as-largest CAS kernel
// (`jit::float_atomics::compile_groupby_float_atomic_kernel`). Under the DuckDB
// total order that kernel and the scalar path (`aggregate.rs::float_total_cmp`)
// implement, NaN is the *largest* value, so the correct MIN identity (the
// maximum of the order, such that `min(identity, x) == x`) is NaN, not +inf.
//
// With a +inf seed the kernel's MIN rule never lets a NaN candidate replace the
// seed (replace = `(!cand_nan & slot_nan) | ordered`), so an all-NaN group would
// wrongly surface +inf. Seeding NaN makes the first finite candidate replace it
// (finite beats a NaN slot) while an all-NaN group keeps NaN — matching the
// scalar reduction, which seeds MIN from the first element rather than ±inf.
//
// MAX keeps the -inf identity (the minimum of the order); the MAX rule already
// lets a NaN candidate beat -inf, so all-NaN MAX correctly yields NaN.
//
// NOTE: this only feeds the NaN-aware `float_atomics` kernel. The validity path
// (`groupby_valid.rs`) has its own `identity_f64`/`identity_f32` seeding the
// bare-`setp` `valid_flag_float` kernel, which still relies on a ±inf seed and
// is intentionally left unchanged here.
fn identity_f32(op: ReduceOp) -> f32 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0.0,
        ReduceOp::Min => f32::NAN,
        ReduceOp::Max => f32::NEG_INFINITY,
    }
}

fn identity_f64(op: ReduceOp) -> f64 {
    match op {
        ReduceOp::Sum | ReduceOp::Count => 0.0,
        ReduceOp::Min => f64::NAN,
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
///
/// F7-finish: uses the MIN/MAX-temporal variant so a grouped
/// `MIN`/`MAX`/`COUNT` over a `Date32`/`Timestamp` value column can build its
/// temporal-typed output field (unit + tz carried through). Temporal GROUP BY
/// *keys* are still rejected earlier in `build_key_arrays` (which runs before
/// this schema build), and `SUM(temporal)` is rejected in `run_typed_agg`'s
/// dtype dispatch — so the only temporal output fields that reach here are the
/// supported aggregate reductions.
fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    crate::exec::schema_convert::plan_schema_to_arrow_schema_minmax_temporal(s)
}

// ---------------------------------------------------------------------------
// R1-utf8-groupby: Utf8 (string) GROUP BY keys via lex-ranked dictionary
// codes.
// ---------------------------------------------------------------------------
//
// The classic / tier integer kernels key on i32/i64/float only. A Utf8 key
// column previously hit `pack_keys` → `key_bit_width(Utf8)` (a hard error)
// and routed to `groupby_wide`, which itself REJECTS Utf8 keys — so a query
// like `SELECT s, SUM(x) FROM t GROUP BY s` over a string column had no GPU
// path at all.
//
// Strategy (single catalog of lex-ranked dictionary codes):
//   1. Detect each GROUP BY key column whose Arrow array is Utf8
//      (`StringArray`) or `Dictionary<Int32, Utf8>` (flattened to Utf8).
//   2. Build a LEX-RANKED dictionary per such column: the distinct non-null
//      strings sorted byte-lexicographically, code `i` = rank `i` (0-based).
//      Encode the column into an `Int32` code array (NULLs preserved as Arrow
//      nulls so the integer path's NULL-group synthesis handles them).
//   3. Rewrite the `AggregateSpec` (key `ColumnIO` dtype Utf8 → Int32, the
//      matching `output_schema` field Utf8 → Int32) and the input
//      `RecordBatch` (Utf8 key column → Int32 code column), then RECURSE into
//      `execute_groupby` — the ordinary integer GROUP BY runs on the codes.
//   4. Reconstruct: map each output Int32 code column back to a `StringArray`
//      through the dictionary (NULL code slot → NULL string), and rebuild the
//      output batch with the ORIGINAL (Utf8) output schema.
//
// Because the codes are lex-ranked, an `ORDER BY s` over the grouped result
// (or any range predicate on `s` folded elsewhere) is monotone in the code —
// matching the sibling agent's lex-ranked ingest dictionaries. Group identity
// only needs the codes to be DISTINCT per string, which any bijection gives;
// the lex ranking is the property that makes downstream ordering sound.
//
// Huge cardinality: if the distinct-string count exceeds
// [`utf8_groupby::MAX_DICT_CARDINALITY`] the i32 code space / hash-table sizing
// is impractical, so we fall back to a correct host GROUP BY
// ([`utf8_groupby::execute_groupby_utf8_host`]) rather than the GPU code path.
pub(crate) mod utf8_groupby {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow_array::{
        Array, ArrayRef, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray,
    };
    use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

    use crate::error::{BoltError, BoltResult};
    use crate::plan::logical_plan::{AggregateExpr, DataType, Expr, Field, Schema};
    use crate::plan::physical_plan::{AggregateSpec, ColumnIO, PhysicalPlan};

    /// Above this distinct-string count we abandon the i32-dict-code GPU path
    /// (the open-addressing hash table sizes to `next_pow2(2*cardinality)`,
    /// and an i32 code space is the hard ceiling) and fall back to the host
    /// GROUP BY. ~16M distinct keys is already far past any sane GPU
    /// group-by working set; the host path stays correct beyond it.
    pub(crate) const MAX_DICT_CARDINALITY: usize = 1 << 24;

    /// A lex-ranked dictionary for one Utf8 GROUP BY key column.
    pub(crate) struct LexDict {
        /// `code -> string`, indexed by the i32 code (0-based, lex order).
        pub(crate) decode: Vec<String>,
    }

    impl LexDict {
        /// Build a lex-ranked dictionary from the distinct non-null values of
        /// `arr`, returning the dictionary plus the per-row i32 code column
        /// (`None` at NULL rows). `arr` must be a Utf8 `StringArray`.
        fn build(arr: &StringArray) -> (LexDict, Vec<Option<i32>>) {
            // Collect distinct non-null strings.
            let mut distinct: Vec<&str> = Vec::new();
            {
                let mut seen: std::collections::HashSet<&str> = std::collections::HashSet::new();
                for i in 0..arr.len() {
                    if arr.is_null(i) {
                        continue;
                    }
                    let s = arr.value(i);
                    if seen.insert(s) {
                        distinct.push(s);
                    }
                }
            }
            // Lex rank: byte-lexicographic sort over the distinct set. Arrow
            // `StringArray` is UTF-8, and Rust `str` `Ord` is byte-lexicographic
            // — the same total order the sibling lex-ranked ingest uses, so
            // grouped output codes are monotone in the original string.
            distinct.sort_unstable();
            let mut code_of: HashMap<&str, i32> = HashMap::with_capacity(distinct.len());
            for (rank, s) in distinct.iter().enumerate() {
                code_of.insert(*s, rank as i32);
            }
            let codes: Vec<Option<i32>> = (0..arr.len())
                .map(|i| {
                    if arr.is_null(i) {
                        None
                    } else {
                        Some(code_of[arr.value(i)])
                    }
                })
                .collect();
            let decode: Vec<String> = distinct.into_iter().map(|s| s.to_string()).collect();
            (LexDict { decode }, codes)
        }

        /// Map an i32 code back to its string, or `None` if out of range.
        fn string_of(&self, code: i32) -> Option<&str> {
            usize::try_from(code)
                .ok()
                .and_then(|i| self.decode.get(i))
                .map(|s| s.as_str())
        }
    }

    /// Flatten an Arrow array that is either a Utf8 `StringArray` or a
    /// `Dictionary<Int32, Utf8>` into an owned `StringArray`. Returns `None`
    /// for any other Arrow type (i.e. this column is not a string key).
    fn as_string_array(arr: &dyn Array) -> Option<StringArray> {
        match arr.data_type() {
            ArrowDataType::Utf8 => arr.as_any().downcast_ref::<StringArray>().cloned(),
            ArrowDataType::Dictionary(_, val_t) if val_t.as_ref() == &ArrowDataType::Utf8 => {
                // Materialise the dictionary into a flat StringArray.
                let casted = arrow::compute::cast(arr, &ArrowDataType::Utf8).ok()?;
                casted.as_any().downcast_ref::<StringArray>().cloned()
            }
            _ => None,
        }
    }

    /// Entry point invoked from [`super::execute_groupby`]. Returns `None`
    /// when no GROUP BY key column is a Utf8 / dictionary string (the all-
    /// integer/float path then runs unchanged), and `Some(result)` when at
    /// least one key is a string and this module owns the query.
    pub(crate) fn try_execute_groupby_utf8(
        plan: &PhysicalPlan,
        table_batch: &RecordBatch,
    ) -> Option<BoltResult<RecordBatch>> {
        let (pre, aggregate) = match plan {
            PhysicalPlan::Aggregate { pre, aggregate, .. } => (pre, aggregate),
            _ => return None,
        };
        if aggregate.group_by.is_empty() {
            return None;
        }

        // Locate every group-by key column whose batch array is a string.
        // `string_keys[j] = Some(string_array)` for group_by position `j`.
        let mut any_string = false;
        let mut string_keys: Vec<Option<StringArray>> =
            Vec::with_capacity(aggregate.group_by.len());
        for &ord in &aggregate.group_by {
            let io = match aggregate.inputs.get(ord) {
                Some(io) => io,
                None => return None, // malformed; let the main path raise it
            };
            let col_idx = match table_batch.schema().index_of(&io.name) {
                Ok(i) => i,
                Err(_) => return None,
            };
            let arr = table_batch.column(col_idx);
            // Treat as a string key when the plan dtype is Utf8 OR the batch
            // array is a string/dict-of-string (covers both the explicit Utf8
            // ColumnIO and a dictionary-typed source column).
            let is_string_plan = matches!(io.dtype, DataType::Utf8);
            match as_string_array(arr.as_ref()) {
                Some(sa) => {
                    any_string = true;
                    string_keys.push(Some(sa));
                }
                None => {
                    if is_string_plan {
                        // Plan says Utf8 but the array isn't a string — let the
                        // main path surface the dtype-mismatch error.
                        return None;
                    }
                    string_keys.push(None);
                }
            }
        }
        if !any_string {
            return None;
        }

        Some(run(pre, aggregate, plan, table_batch, string_keys))
    }

    /// Core driver: build dictionaries, rewrite the plan + batch to integer
    /// codes, recurse into the integer GROUP BY, and reconstruct the Utf8 key
    /// columns on output.
    fn run(
        pre: &Option<crate::plan::physical_plan::KernelSpec>,
        aggregate: &AggregateSpec,
        orig_plan: &PhysicalPlan,
        table_batch: &RecordBatch,
        string_keys: Vec<Option<StringArray>>,
    ) -> BoltResult<RecordBatch> {
        // The pre-projection GROUP BY path (`groupby_with_pre`) rejects Utf8
        // keys with a clear error and has no dictionary plumbing; a Utf8 key
        // there is rare (it would require a string key threaded through a
        // pre-kernel). Defer to the host fallback in that case.
        if pre.is_some() {
            return execute_groupby_utf8_host(aggregate, table_batch, &string_keys);
        }

        // Build a lex-ranked dictionary + i32 code column for each string key.
        // `dicts[j]` is `Some(dict)` exactly when `string_keys[j].is_some()`.
        let mut dicts: Vec<Option<LexDict>> = Vec::with_capacity(string_keys.len());
        // `code_columns[j]` is the Int32 code ArrayRef for the rewritten batch.
        let mut code_columns: Vec<Option<ArrayRef>> = Vec::with_capacity(string_keys.len());
        for sk in &string_keys {
            match sk {
                Some(sa) => {
                    let (dict, codes) = LexDict::build(sa);
                    if dict.decode.len() > MAX_DICT_CARDINALITY {
                        log::warn!(
                            "execute_groupby (utf8): GROUP BY string key has {} \
                             distinct values (> {}); falling back to host GROUP BY",
                            dict.decode.len(),
                            MAX_DICT_CARDINALITY
                        );
                        return execute_groupby_utf8_host(aggregate, table_batch, &string_keys);
                    }
                    let arr: ArrayRef = Arc::new(Int32Array::from(codes));
                    dicts.push(Some(dict));
                    code_columns.push(Some(arr));
                }
                None => {
                    dicts.push(None);
                    code_columns.push(None);
                }
            }
        }

        // --- Rewrite the AggregateSpec: each Utf8 key ColumnIO + its
        //     output_schema field flips Utf8 -> Int32. ---
        let mut new_inputs = aggregate.inputs.clone();
        // Map group_by position -> whether this key is a string (for output
        // reconstruction) and the original Utf8 field.
        for (j, &ord) in aggregate.group_by.iter().enumerate() {
            if dicts[j].is_some() {
                new_inputs[ord] = ColumnIO {
                    name: aggregate.inputs[ord].name.clone(),
                    dtype: DataType::Int32,
                };
            }
        }

        // The output schema lists the group-by key columns first (in
        // `group_by` order), then the aggregate result columns. Flip each
        // string key field to Int32 in a cloned schema, remembering the
        // ORIGINAL field so we can restore it (and its name) on output.
        let m_keys = aggregate.group_by.len();
        if aggregate.output_schema.fields.len() < m_keys {
            return Err(BoltError::Other(
                "execute_groupby (utf8): output_schema has fewer fields than GROUP BY keys".into(),
            ));
        }
        let mut new_fields: Vec<Field> = aggregate.output_schema.fields.clone();
        let mut orig_key_fields: Vec<Field> = Vec::with_capacity(m_keys);
        for j in 0..m_keys {
            orig_key_fields.push(aggregate.output_schema.fields[j].clone());
            if dicts[j].is_some() {
                new_fields[j] = Field::new(
                    aggregate.output_schema.fields[j].name.clone(),
                    DataType::Int32,
                    aggregate.output_schema.fields[j].nullable,
                );
            }
        }

        let new_aggregate = AggregateSpec {
            inputs: new_inputs,
            group_by: aggregate.group_by.clone(),
            aggregates: aggregate.aggregates.clone(),
            output_schema: Schema::new(new_fields),
            input_has_validity: aggregate.input_has_validity.clone(),
        };

        // Sanity: the rewritten spec must no longer claim any Utf8 key.
        debug_assert!(
            new_aggregate
                .group_by
                .iter()
                .all(|&o| new_aggregate.inputs[o].dtype != DataType::Utf8),
            "utf8 keys must be rewritten to Int32 before recursing"
        );
        let _ = orig_plan;

        // --- Rewrite the input batch: replace each string key column with its
        //     Int32 code column, leaving every other column untouched. ---
        let new_batch = rewrite_batch(table_batch, aggregate, &code_columns)?;

        // --- Recurse into the integer GROUP BY. ---
        let rewritten_plan = PhysicalPlan::Aggregate {
            table: String::new(),
            pre: None,
            aggregate: new_aggregate,
        };
        let int_result = super::execute_groupby(&rewritten_plan, &new_batch)?;

        // --- Reconstruct: map the Int32 key code columns back to Utf8. ---
        reconstruct(
            &int_result,
            &dicts,
            &orig_key_fields,
            &aggregate.output_schema,
        )
    }

    /// Build a new `RecordBatch` identical to `table_batch` except that each
    /// Utf8 GROUP BY key column (located by its `ColumnIO` name) is replaced
    /// by the matching Int32 code column. The schema field for that column is
    /// changed to `Int32` with the same nullability.
    fn rewrite_batch(
        table_batch: &RecordBatch,
        aggregate: &AggregateSpec,
        code_columns: &[Option<ArrayRef>],
    ) -> BoltResult<RecordBatch> {
        // name -> code column for the string keys.
        let mut replace: HashMap<&str, &ArrayRef> = HashMap::new();
        for (j, &ord) in aggregate.group_by.iter().enumerate() {
            if let Some(codes) = &code_columns[j] {
                replace.insert(aggregate.inputs[ord].name.as_str(), codes);
            }
        }

        let schema = table_batch.schema();
        let mut fields: Vec<ArrowField> = Vec::with_capacity(schema.fields().len());
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(schema.fields().len());
        for (i, f) in schema.fields().iter().enumerate() {
            match replace.get(f.name().as_str()) {
                Some(codes) => {
                    fields.push(ArrowField::new(
                        f.name(),
                        ArrowDataType::Int32,
                        f.is_nullable(),
                    ));
                    columns.push((*codes).clone());
                }
                None => {
                    fields.push(f.as_ref().clone());
                    columns.push(table_batch.column(i).clone());
                }
            }
        }
        let new_schema = Arc::new(ArrowSchema::new(fields));
        RecordBatch::try_new(new_schema, columns).map_err(|e| {
            BoltError::Other(format!(
                "execute_groupby (utf8): failed to build dict-coded batch: {e}"
            ))
        })
    }

    /// Replace the Int32 key code columns in `int_result` with reconstructed
    /// Utf8 string columns, returning a batch with the ORIGINAL output schema.
    fn reconstruct(
        int_result: &RecordBatch,
        dicts: &[Option<LexDict>],
        orig_key_fields: &[Field],
        orig_output_schema: &Schema,
    ) -> BoltResult<RecordBatch> {
        let m_keys = dicts.len();
        if int_result.num_columns() < m_keys {
            return Err(BoltError::Other(
                "execute_groupby (utf8): integer result has fewer columns than GROUP BY keys"
                    .into(),
            ));
        }

        let mut out_cols: Vec<ArrayRef> = Vec::with_capacity(int_result.num_columns());
        for col in 0..int_result.num_columns() {
            if col < m_keys {
                if let Some(dict) = &dicts[col] {
                    // This key was a string: decode the Int32 codes back to
                    // strings, preserving NULL slots (the synthesised NULL
                    // group surfaces as a NULL code -> NULL string).
                    let codes = int_result
                        .column(col)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .ok_or_else(|| {
                            BoltError::Other(format!(
                                "execute_groupby (utf8): key column {col} of the integer \
                                 result was not Int32"
                            ))
                        })?;
                    let mut strings: Vec<Option<String>> = Vec::with_capacity(codes.len());
                    for i in 0..codes.len() {
                        if codes.is_null(i) {
                            strings.push(None);
                        } else {
                            let c = codes.value(i);
                            match dict.string_of(c) {
                                Some(s) => strings.push(Some(s.to_string())),
                                None => {
                                    return Err(BoltError::Other(format!(
                                        "execute_groupby (utf8): group key code {c} out of \
                                         dictionary range ({} entries)",
                                        dict.decode.len()
                                    )))
                                }
                            }
                        }
                    }
                    out_cols.push(Arc::new(StringArray::from(strings)) as ArrayRef);
                    continue;
                }
                // Non-string key: pass the column through unchanged.
                out_cols.push(int_result.column(col).clone());
            } else {
                // Aggregate result column: unchanged.
                out_cols.push(int_result.column(col).clone());
            }
        }

        // Build the output schema: restore the ORIGINAL key fields (Utf8),
        // keep the aggregate fields from the original output schema.
        let mut fields: Vec<Field> = Vec::with_capacity(orig_output_schema.fields.len());
        for f in orig_key_fields.iter() {
            fields.push(f.clone());
        }
        for f in orig_output_schema.fields.iter().skip(m_keys) {
            fields.push(f.clone());
        }
        let arrow_schema = super::plan_schema_to_arrow_schema(&Schema::new(fields))?;
        RecordBatch::try_new(arrow_schema, out_cols).map_err(|e| {
            BoltError::Other(format!(
                "execute_groupby (utf8): failed to rebuild Utf8-keyed result: {e}"
            ))
        })
    }

    // -----------------------------------------------------------------------
    // Host fallback for Utf8 GROUP BY keys.
    //
    // Used when the GPU dict-code path can't be taken: a pre-projection
    // kernel is present, or the distinct-string cardinality exceeds
    // `MAX_DICT_CARDINALITY`. Computes the same SUM / MIN / MAX / COUNT / AVG
    // result on the host, keyed by the (possibly multi-column) tuple of
    // string + non-string key values. Output ordering matches the GPU path:
    // groups sorted by their key tuple (string keys compared lexicographically,
    // which equals the lex-rank order the GPU path emits).
    // -----------------------------------------------------------------------

    /// One key component value for the host fallback's group tuple.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
    enum HostKey {
        /// A string key value (`None` => SQL NULL).
        Str(Option<String>),
        /// A non-string key, normalised to its i64 bit/widened representation
        /// (`None` => SQL NULL). Float bit patterns are canonicalised the same
        /// way the GPU path packs them is NOT required here because the host
        /// fallback only groups — equality on the raw i64 suffices.
        Int(Option<i64>),
    }

    /// Host GROUP BY over (possibly mixed) string + integer keys. Supports the
    /// common single-string-key shape `SELECT s, SUM(x) FROM t GROUP BY s`
    /// plus multi-key tuples. Rejects shapes outside the host fallback's
    /// coverage with a clear error (the caller has already established that at
    /// least one key is a string).
    pub(crate) fn execute_groupby_utf8_host(
        aggregate: &AggregateSpec,
        table_batch: &RecordBatch,
        string_keys: &[Option<StringArray>],
    ) -> BoltResult<RecordBatch> {
        let n_rows = table_batch.num_rows();
        let m_keys = aggregate.group_by.len();

        // Resolve each key column to a per-row HostKey extractor.
        // `key_arrays[j]` is the source array for group_by position `j`.
        let mut key_is_string: Vec<bool> = Vec::with_capacity(m_keys);
        let mut int_key_arrays: Vec<Option<Int64KeyView>> = Vec::with_capacity(m_keys);
        for (j, &ord) in aggregate.group_by.iter().enumerate() {
            if string_keys[j].is_some() {
                key_is_string.push(true);
                int_key_arrays.push(None);
            } else {
                let io = &aggregate.inputs[ord];
                let idx = table_batch.schema().index_of(&io.name).map_err(|e| {
                    BoltError::Plan(format!(
                        "execute_groupby (utf8 host): key '{}' not in batch: {e}",
                        io.name
                    ))
                })?;
                key_is_string.push(false);
                int_key_arrays.push(Some(Int64KeyView::new(
                    table_batch.column(idx).as_ref(),
                    &io.name,
                )?));
            }
        }

        // Build the per-row key tuple and a parallel group index.
        let mut group_of_row: Vec<usize> = Vec::with_capacity(n_rows);
        let mut keys_in_order: Vec<Vec<HostKey>> = Vec::new();
        let mut index: HashMap<Vec<HostKey>, usize> = HashMap::new();
        for row in 0..n_rows {
            let mut tuple: Vec<HostKey> = Vec::with_capacity(m_keys);
            for j in 0..m_keys {
                if key_is_string[j] {
                    let sa = string_keys[j].as_ref().expect("string key present");
                    let v = if sa.is_null(row) {
                        None
                    } else {
                        Some(sa.value(row).to_string())
                    };
                    tuple.push(HostKey::Str(v));
                } else {
                    let kv = int_key_arrays[j].as_ref().expect("int key present");
                    tuple.push(HostKey::Int(kv.value(row)));
                }
            }
            let g = match index.get(&tuple) {
                Some(&g) => g,
                None => {
                    let g = keys_in_order.len();
                    index.insert(tuple.clone(), g);
                    keys_in_order.push(tuple);
                    g
                }
            };
            group_of_row.push(g);
        }

        let n_groups = keys_in_order.len();

        // Deterministic ordering: sort groups by their key tuple. `HostKey`'s
        // derived `Ord` compares strings lexicographically and NULLs sort
        // before non-NULLs (matching the GPU path's i64::MAX NULL sentinel
        // would sort NULLs LAST; we instead keep SQL's common "NULLs implied"
        // ordering — the engine applies any explicit ORDER BY downstream, so
        // only determinism matters here).
        let mut order: Vec<usize> = (0..n_groups).collect();
        order.sort_by(|&a, &b| keys_in_order[a].cmp(&keys_in_order[b]));
        // remap[g] = output row position of group g.
        let mut remap: Vec<usize> = vec![0; n_groups];
        for (out_pos, &g) in order.iter().enumerate() {
            remap[g] = out_pos;
        }

        // Compute each aggregate host-side into per-output-row buffers.
        let mut agg_arrays: Vec<ArrayRef> = Vec::with_capacity(aggregate.aggregates.len());
        for (ai, agg) in aggregate.aggregates.iter().enumerate() {
            let out_field = aggregate
                .output_schema
                .fields
                .get(m_keys + ai)
                .ok_or_else(|| {
                    BoltError::Other(format!(
                    "execute_groupby (utf8 host): output_schema missing field for aggregate {ai}"
                ))
                })?;
            let arr = compute_host_aggregate(
                agg,
                aggregate,
                table_batch,
                &group_of_row,
                &remap,
                n_groups,
                out_field,
            )?;
            agg_arrays.push(arr);
        }

        // Build the key output columns in sorted order.
        let mut out_cols: Vec<ArrayRef> = Vec::with_capacity(m_keys + agg_arrays.len());
        for j in 0..m_keys {
            out_cols.push(build_host_key_column(j, &keys_in_order, &order)?);
        }
        out_cols.extend(agg_arrays);

        let arrow_schema = super::plan_schema_to_arrow_schema(&aggregate.output_schema)?;
        RecordBatch::try_new(arrow_schema, out_cols).map_err(|e| {
            BoltError::Other(format!(
                "execute_groupby (utf8 host): failed to build result batch: {e}"
            ))
        })
    }

    /// Build one key output column (position `j`) from the sorted group order.
    fn build_host_key_column(
        j: usize,
        keys_in_order: &[Vec<HostKey>],
        order: &[usize],
    ) -> BoltResult<ArrayRef> {
        // Inspect the first group's value to choose the column kind.
        match keys_in_order.first().and_then(|t| t.get(j)) {
            Some(HostKey::Str(_)) => {
                let vals: Vec<Option<String>> = order
                    .iter()
                    .map(|&g| match &keys_in_order[g][j] {
                        HostKey::Str(s) => s.clone(),
                        HostKey::Int(_) => None,
                    })
                    .collect();
                Ok(Arc::new(StringArray::from(vals)) as ArrayRef)
            }
            Some(HostKey::Int(_)) => {
                // Non-string key columns in a mixed tuple are emitted as Int64
                // here; the only Utf8-keyed host-fallback shapes the engine
                // produces today are single-string keys, so a mixed tuple is
                // rare. (Decoding the exact original dtype would need the
                // per-column dtype threaded in; Int64 is a safe superset for
                // the integer key types this fallback accepts.)
                let vals: Vec<Option<i64>> = order
                    .iter()
                    .map(|&g| match &keys_in_order[g][j] {
                        HostKey::Int(v) => *v,
                        HostKey::Str(_) => None,
                    })
                    .collect();
                Ok(Arc::new(Int64Array::from(vals)) as ArrayRef)
            }
            None => Ok(Arc::new(StringArray::from(Vec::<Option<String>>::new())) as ArrayRef),
        }
    }

    /// A read-only view over a non-string key column, exposing each row as an
    /// `Option<i64>` (NULL-aware) for the host fallback's group tuple. Only the
    /// integer key dtypes the GPU path accepts are supported.
    struct Int64KeyView {
        vals: Vec<Option<i64>>,
    }

    impl Int64KeyView {
        fn new(arr: &dyn Array, name: &str) -> BoltResult<Self> {
            let vals: Vec<Option<i64>> = match arr.data_type() {
                ArrowDataType::Int32 => {
                    let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
                    (0..a.len())
                        .map(|i| {
                            if a.is_null(i) {
                                None
                            } else {
                                Some(a.value(i) as i64)
                            }
                        })
                        .collect()
                }
                ArrowDataType::Int64 => {
                    let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
                    (0..a.len())
                        .map(|i| if a.is_null(i) { None } else { Some(a.value(i)) })
                        .collect()
                }
                other => {
                    return Err(BoltError::Type(format!(
                        "execute_groupby (utf8 host): non-string GROUP BY key '{name}' has \
                         dtype {other:?}, which the host fallback does not support alongside \
                         a string key"
                    )))
                }
            };
            Ok(Int64KeyView { vals })
        }

        fn value(&self, row: usize) -> Option<i64> {
            self.vals[row]
        }
    }

    /// Compute one aggregate host-side, scattering per-group results into an
    /// Arrow array in the SORTED output-row order given by `remap`.
    fn compute_host_aggregate(
        agg: &AggregateExpr,
        _aggregate: &AggregateSpec,
        table_batch: &RecordBatch,
        group_of_row: &[usize],
        remap: &[usize],
        n_groups: usize,
        out_field: &Field,
    ) -> BoltResult<ArrayRef> {
        // Resolve the aggregate's input column to host f64 values + validity.
        // COUNT(*) has no column; everything else resolves a bare column ref.
        let col_name: Option<String> = match agg {
            AggregateExpr::Sum(e)
            | AggregateExpr::Min(e)
            | AggregateExpr::Max(e)
            | AggregateExpr::Avg(e)
            | AggregateExpr::Count(e) => bare_col(e),
            AggregateExpr::VarPop(e)
            | AggregateExpr::VarSamp(e)
            | AggregateExpr::StddevPop(e)
            | AggregateExpr::StddevSamp(e) => bare_col(e),
        };

        // Per-row (value, valid). For COUNT(*) (no resolvable column) value is
        // unused and valid is always true.
        let (vals, valid): (Vec<f64>, Vec<bool>) = match &col_name {
            Some(name) => load_f64_column(table_batch, name)?,
            None => (
                vec![0.0; group_of_row.len()],
                vec![true; group_of_row.len()],
            ),
        };

        match agg {
            AggregateExpr::Count(e) => {
                // COUNT(col) counts non-NULL rows; COUNT(*) counts every row.
                let is_count_col = bare_col(e).is_some();
                let mut counts: Vec<i64> = vec![0; n_groups];
                for (row, &g) in group_of_row.iter().enumerate() {
                    if !is_count_col || valid[row] {
                        counts[g] += 1;
                    }
                }
                let mut out: Vec<i64> = vec![0; n_groups];
                for (g, &c) in counts.iter().enumerate() {
                    out[remap[g]] = c;
                }
                Ok(Arc::new(Int64Array::from(out)) as ArrayRef)
            }
            AggregateExpr::Sum(_) => {
                let mut sums: Vec<f64> = vec![0.0; n_groups];
                let mut seen: Vec<bool> = vec![false; n_groups];
                for (row, &g) in group_of_row.iter().enumerate() {
                    if valid[row] {
                        sums[g] += vals[row];
                        seen[g] = true;
                    }
                }
                scatter_f64_to_field(&sums, Some(&seen), remap, n_groups, out_field)
            }
            AggregateExpr::Min(_) | AggregateExpr::Max(_) => {
                let is_min = matches!(agg, AggregateExpr::Min(_));
                let mut acc: Vec<f64> = vec![0.0; n_groups];
                let mut seen: Vec<bool> = vec![false; n_groups];
                for (row, &g) in group_of_row.iter().enumerate() {
                    if !valid[row] {
                        continue;
                    }
                    if !seen[g] {
                        acc[g] = vals[row];
                        seen[g] = true;
                    } else if is_min {
                        if vals[row] < acc[g] {
                            acc[g] = vals[row];
                        }
                    } else if vals[row] > acc[g] {
                        acc[g] = vals[row];
                    }
                }
                scatter_f64_to_field(&acc, Some(&seen), remap, n_groups, out_field)
            }
            AggregateExpr::Avg(_) => {
                let mut sums: Vec<f64> = vec![0.0; n_groups];
                let mut counts: Vec<i64> = vec![0; n_groups];
                for (row, &g) in group_of_row.iter().enumerate() {
                    if valid[row] {
                        sums[g] += vals[row];
                        counts[g] += 1;
                    }
                }
                let mut out: Vec<Option<f64>> = vec![None; n_groups];
                for g in 0..n_groups {
                    out[remap[g]] = if counts[g] == 0 {
                        None
                    } else {
                        Some(sums[g] / counts[g] as f64)
                    };
                }
                Ok(Arc::new(Float64Array::from(out)) as ArrayRef)
            }
            AggregateExpr::VarPop(_)
            | AggregateExpr::VarSamp(_)
            | AggregateExpr::StddevPop(_)
            | AggregateExpr::StddevSamp(_) => {
                // Welford per group on the host.
                let mut states: Vec<crate::exec::welford::WelfordState> =
                    vec![crate::exec::welford::WelfordState::empty(); n_groups];
                for (row, &g) in group_of_row.iter().enumerate() {
                    if valid[row] {
                        states[g].push(vals[row]);
                    }
                }
                let mut out: Vec<Option<f64>> = vec![None; n_groups];
                for g in 0..n_groups {
                    let st = &states[g];
                    out[remap[g]] = match agg {
                        AggregateExpr::VarPop(_) => st.var_pop(),
                        AggregateExpr::VarSamp(_) => st.var_samp(),
                        AggregateExpr::StddevPop(_) => st.stddev_pop(),
                        AggregateExpr::StddevSamp(_) => st.stddev_samp(),
                        _ => unreachable!(),
                    };
                }
                Ok(Arc::new(Float64Array::from(out)) as ArrayRef)
            }
        }
    }

    /// Scatter a per-group f64 accumulator into the output array in `remap`
    /// order, casting to `out_field.dtype`. `seen` (when present) marks groups
    /// with no contributing rows as SQL NULL.
    fn scatter_f64_to_field(
        acc: &[f64],
        seen: Option<&[bool]>,
        remap: &[usize],
        n_groups: usize,
        out_field: &Field,
    ) -> BoltResult<ArrayRef> {
        // Build per-output-row Option<f64> first, then cast to the field dtype.
        let mut ordered: Vec<Option<f64>> = vec![None; n_groups];
        for g in 0..n_groups {
            let present = seen.map(|s| s[g]).unwrap_or(true);
            ordered[remap[g]] = if present { Some(acc[g]) } else { None };
        }
        match out_field.dtype {
            DataType::Int32 => Ok(Arc::new(Int32Array::from(
                ordered
                    .into_iter()
                    .map(|o| o.map(|v| v as i32))
                    .collect::<Vec<Option<i32>>>(),
            )) as ArrayRef),
            DataType::Int64 => Ok(Arc::new(Int64Array::from(
                ordered
                    .into_iter()
                    .map(|o| o.map(|v| v as i64))
                    .collect::<Vec<Option<i64>>>(),
            )) as ArrayRef),
            DataType::Float32 => Ok(Arc::new(arrow_array::Float32Array::from(
                ordered
                    .into_iter()
                    .map(|o| o.map(|v| v as f32))
                    .collect::<Vec<Option<f32>>>(),
            )) as ArrayRef),
            DataType::Float64 => Ok(Arc::new(Float64Array::from(ordered)) as ArrayRef),
            ref other => Err(BoltError::Type(format!(
                "execute_groupby (utf8 host): aggregate output dtype {other:?} not supported \
                 in the host fallback"
            ))),
        }
    }

    /// Load a numeric column as host `(values_f64, validity)`. Supports the
    /// integer / float value dtypes the GPU group-by aggregates accept.
    fn load_f64_column(batch: &RecordBatch, name: &str) -> BoltResult<(Vec<f64>, Vec<bool>)> {
        let idx = batch.schema().index_of(name).map_err(|e| {
            BoltError::Plan(format!(
                "execute_groupby (utf8 host): aggregate input '{name}' not in batch: {e}"
            ))
        })?;
        let arr = batch.column(idx);
        let n = arr.len();
        let valid: Vec<bool> = (0..n).map(|i| !arr.is_null(i)).collect();
        let vals: Vec<f64> = match arr.data_type() {
            ArrowDataType::Int32 => {
                let a = arr.as_any().downcast_ref::<Int32Array>().unwrap();
                (0..n).map(|i| a.value(i) as f64).collect()
            }
            ArrowDataType::Int64 => {
                let a = arr.as_any().downcast_ref::<Int64Array>().unwrap();
                (0..n).map(|i| a.value(i) as f64).collect()
            }
            ArrowDataType::Float32 => {
                let a = arr
                    .as_any()
                    .downcast_ref::<arrow_array::Float32Array>()
                    .unwrap();
                (0..n).map(|i| a.value(i) as f64).collect()
            }
            ArrowDataType::Float64 => {
                let a = arr.as_any().downcast_ref::<Float64Array>().unwrap();
                (0..n).map(|i| a.value(i)).collect()
            }
            other => {
                return Err(BoltError::Type(format!(
                    "execute_groupby (utf8 host): aggregate input '{name}' has dtype {other:?}, \
                     which the host fallback does not support"
                )))
            }
        };
        Ok((vals, valid))
    }

    /// Extract a bare column name from an aggregate-input expression, or
    /// `None` when the expression is not a bare column ref (e.g. COUNT(*) over
    /// a literal).
    fn bare_col(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Column(name) => Some(name.clone()),
            Expr::Alias(inner, _) => bare_col(inner),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // Host-only tests for the Utf8 GROUP BY plumbing. All pure host logic —
    // they build dictionaries, encode/decode codes, and run the host
    // fallback group-by without touching CUDA, so they run under
    // `--no-default-features --features cuda-stub`.
    // -----------------------------------------------------------------------
    #[cfg(test)]
    mod tests {
        use super::*;

        fn agg_spec(
            inputs: Vec<(&str, DataType)>,
            group_by: Vec<usize>,
            aggregates: Vec<AggregateExpr>,
            out_fields: Vec<(&str, DataType)>,
        ) -> AggregateSpec {
            AggregateSpec {
                inputs: inputs
                    .into_iter()
                    .map(|(n, d)| ColumnIO {
                        name: n.to_string(),
                        dtype: d,
                    })
                    .collect(),
                group_by,
                aggregates,
                output_schema: Schema::new(
                    out_fields
                        .into_iter()
                        .map(|(n, d)| Field::new(n, d, true))
                        .collect(),
                ),
                input_has_validity: Vec::new(),
            }
        }

        /// LexDict assigns codes in byte-lexicographic order of the distinct
        /// non-null strings, and encodes each row to its code (NULL preserved).
        #[test]
        fn lex_dict_build_is_lex_ranked() {
            let arr = StringArray::from(vec![Some("US"), Some("EU"), Some("US"), None, Some("AU")]);
            let (dict, codes) = LexDict::build(&arr);
            // Distinct sorted: AU, EU, US => codes 0, 1, 2.
            assert_eq!(
                dict.decode,
                vec!["AU".to_string(), "EU".into(), "US".into()]
            );
            assert_eq!(
                codes,
                vec![Some(2), Some(1), Some(2), None, Some(0)],
                "rows encode to their lex rank; NULL stays None"
            );
            // Round-trip decode.
            assert_eq!(dict.string_of(0), Some("AU"));
            assert_eq!(dict.string_of(2), Some("US"));
            assert_eq!(dict.string_of(3), None, "out-of-range code -> None");
        }

        /// `as_string_array` flattens a plain Utf8 StringArray.
        #[test]
        fn as_string_array_passes_utf8_through() {
            let arr = StringArray::from(vec!["a", "b"]);
            let got = as_string_array(&arr).expect("utf8 should flatten");
            assert_eq!(got.value(0), "a");
            assert_eq!(got.value(1), "b");
        }

        /// `as_string_array` returns None for a non-string array.
        #[test]
        fn as_string_array_rejects_non_string() {
            let arr = Int64Array::from(vec![1i64, 2]);
            assert!(as_string_array(&arr).is_none());
        }

        /// Host fallback: `SELECT s, SUM(x) FROM t GROUP BY s` over a string
        /// key. Groups emit in lex order (EU, US) with the correct sums.
        #[test]
        fn host_fallback_sum_by_string_key() {
            let s = Arc::new(StringArray::from(vec!["US", "EU", "US", "EU", "US"])) as ArrayRef;
            let x = Arc::new(Int64Array::from(vec![1i64, 10, 2, 20, 3])) as ArrayRef;
            let schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("s", ArrowDataType::Utf8, true),
                ArrowField::new("x", ArrowDataType::Int64, false),
            ]));
            let batch = RecordBatch::try_new(schema, vec![s.clone(), x]).unwrap();

            let agg = agg_spec(
                vec![("s", DataType::Utf8), ("x", DataType::Int64)],
                vec![0],
                vec![AggregateExpr::Sum(Expr::Column("x".into()))],
                vec![("s", DataType::Utf8), ("sum_x", DataType::Int64)],
            );
            let string_keys = vec![Some(
                s.as_any().downcast_ref::<StringArray>().unwrap().clone(),
            )];

            let out = execute_groupby_utf8_host(&agg, &batch, &string_keys).unwrap();
            assert_eq!(out.num_rows(), 2);
            let keys = out
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let sums = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            // Lex order: EU first (10+20=30), then US (1+2+3=6).
            assert_eq!(keys.value(0), "EU");
            assert_eq!(sums.value(0), 30);
            assert_eq!(keys.value(1), "US");
            assert_eq!(sums.value(1), 6);
        }

        /// Host fallback: COUNT(*) per string key and a NULL key form their
        /// own group (NULL sorts first under the derived ordering).
        #[test]
        fn host_fallback_count_with_null_key() {
            let s = Arc::new(StringArray::from(vec![
                Some("b"),
                None,
                Some("a"),
                Some("b"),
                None,
            ])) as ArrayRef;
            let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
                "s",
                ArrowDataType::Utf8,
                true,
            )]));
            let batch = RecordBatch::try_new(schema, vec![s.clone()]).unwrap();

            let agg = agg_spec(
                vec![("s", DataType::Utf8)],
                vec![0],
                // COUNT(*) lowers to Count(Literal) — a non-column expr.
                vec![AggregateExpr::Count(crate::plan::logical_plan::lit(1_i64))],
                vec![("s", DataType::Utf8), ("cnt", DataType::Int64)],
            );
            let string_keys = vec![Some(
                s.as_any().downcast_ref::<StringArray>().unwrap().clone(),
            )];

            let out = execute_groupby_utf8_host(&agg, &batch, &string_keys).unwrap();
            assert_eq!(out.num_rows(), 3, "groups: NULL, a, b");
            let keys = out
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let cnt = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            // NULL sorts first, then 'a', then 'b'.
            assert!(keys.is_null(0), "NULL group sorts first");
            assert_eq!(cnt.value(0), 2, "two NULL-key rows");
            assert_eq!(keys.value(1), "a");
            assert_eq!(cnt.value(1), 1);
            assert_eq!(keys.value(2), "b");
            assert_eq!(cnt.value(2), 2);
        }

        /// `reconstruct` maps an Int32 key code column back to its strings,
        /// preserving a NULL slot, and restores the original Utf8 schema.
        #[test]
        fn reconstruct_decodes_codes_to_strings() {
            // Integer-keyed result: codes [0, 1, NULL] + a SUM column.
            let codes = Arc::new(Int32Array::from(vec![Some(0), Some(1), None])) as ArrayRef;
            let sums = Arc::new(Int64Array::from(vec![5i64, 7, 9])) as ArrayRef;
            let int_schema = Arc::new(ArrowSchema::new(vec![
                ArrowField::new("s", ArrowDataType::Int32, true),
                ArrowField::new("sum_x", ArrowDataType::Int64, true),
            ]));
            let int_result = RecordBatch::try_new(int_schema, vec![codes, sums]).unwrap();

            let dict = LexDict {
                decode: vec!["AU".to_string(), "EU".to_string()],
            };
            let dicts = vec![Some(dict)];
            let orig_key_fields = vec![Field::new("s", DataType::Utf8, true)];
            let orig_out = Schema::new(vec![
                Field::new("s", DataType::Utf8, true),
                Field::new("sum_x", DataType::Int64, true),
            ]);

            let out = reconstruct(&int_result, &dicts, &orig_key_fields, &orig_out)
                .expect("reconstruct ok");
            let keys = out
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            assert_eq!(keys.value(0), "AU");
            assert_eq!(keys.value(1), "EU");
            assert!(keys.is_null(2), "NULL code -> NULL string");
            // Aggregate column carried through unchanged.
            let sums = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            assert_eq!(sums.value(2), 9);
            // Output schema's key field is restored to Utf8.
            assert_eq!(out.schema().field(0).data_type(), &ArrowDataType::Utf8);
        }
    }
}

// ---------------------------------------------------------------------------
// Host-only tests for `pack_keys` / `decode_key` (no GPU required).
// ---------------------------------------------------------------------------
// v0.7: these arrow/plan aliases are used only by the #[cfg(test)] modules
// below; the non-test schema conversion now lives in exec::schema_convert.
// cfg(test)-gated so normal builds don't see an unused import.
#[cfg(test)]
use arrow_schema::Field as ArrowField;

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
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(name, dt, true)]));
        RecordBatch::try_new(schema, vec![arr]).expect("one-col batch")
    }

    // V-10 (grouped): `checked_group_sum` is the host-side overflow guard
    // that restores the scalar SUM "never silently wrong" contract for the
    // GROUP BY SUM(i64) / SUM(i32->i64) paths. These tests are pure host
    // logic and run under `--no-default-features --features cuda-stub`.

    /// A grouped SUM whose per-group accumulator stays inside i64 range
    /// passes the guard untouched (no false positive — correct queries are
    /// never regressed). Two distinct groups, each summing within range.
    #[test]
    fn checked_group_sum_within_range_ok() {
        // group 7 -> 1+2+3 = 6 ; group 9 -> 10+20 = 30
        let values = [1i64, 2, 3, 10, 20];
        let keys = [7i64, 7, 7, 9, 9];
        assert!(checked_group_sum(&values, &keys).is_ok());
    }

    /// Even when the GLOBAL sum would overflow, the guard operates PER GROUP:
    /// here every individual group stays in range, so no error is raised
    /// (mirrors the kernel, which accumulates independently per group).
    #[test]
    fn checked_group_sum_per_group_not_global() {
        // Two groups each at i64::MAX individually; global sum would overflow
        // but neither group does, so this must succeed.
        let values = [i64::MAX, i64::MAX];
        let keys = [1i64, 2];
        assert!(checked_group_sum(&values, &keys).is_ok());
    }

    /// A single group whose true sum exceeds `i64::MAX` must ERROR with a
    /// `BoltError::Type` whose message matches the scalar SUM path verbatim
    /// — not silently wrap the way the device `atom.add.u64` would.
    #[test]
    fn checked_group_sum_overflow_errors_not_wraps() {
        // group 5: i64::MAX + 1 overflows.
        let values = [i64::MAX, 1i64];
        let keys = [5i64, 5];
        match checked_group_sum(&values, &keys) {
            Err(BoltError::Type(msg)) => {
                assert!(
                    msg.contains("SUM(integer) overflow"),
                    "message should mirror the scalar SUM contract, got: {msg}",
                );
            }
            other => panic!("expected BoltError::Type on grouped SUM overflow, got {other:?}"),
        }
    }

    /// Negative-direction overflow (below `i64::MIN`) is equally detected.
    #[test]
    fn checked_group_sum_underflow_errors() {
        let values = [i64::MIN, -1i64];
        let keys = [0i64, 0];
        assert!(matches!(
            checked_group_sum(&values, &keys),
            Err(BoltError::Type(_)),
        ));
    }

    /// Empty input is trivially in-range.
    #[test]
    fn checked_group_sum_empty_ok() {
        assert!(checked_group_sum(&[], &[]).is_ok());
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

        let arr2: ArrayRef = Arc::new(Int64Array::from(vec![Some(1i64), None, Some(3)]));
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
        let out =
            collect_filtered_primitive::<arrow_array::types::Int32Type>(&arr, None, Some(&vv));
        assert_eq!(out, vec![1, 3, 4]);

        // With a key_valid that ALSO drops row 0, only rows 2 and 3 survive.
        let kv = vec![false, true, true, true];
        let out2 =
            collect_filtered_primitive::<arrow_array::types::Int32Type>(&arr, Some(&kv), Some(&vv));
        assert_eq!(out2, vec![3, 4]);
    }

    /// `filter_iter_to_f64` (the AVG-input helper) drops NULL positions
    /// from both the key and the value side and upcasts in one pass.
    #[test]
    fn filter_iter_to_f64_drops_and_casts() {
        let arr = Int32Array::from(vec![Some(2i32), None, Some(4), Some(6)]);
        let vv: Vec<bool> = (0..arr.len()).map(|i| !arr.is_null(i)).collect();
        let out =
            filter_iter_to_f64::<arrow_array::types::Int32Type, _>(&arr, None, Some(&vv), |v| {
                v as f64
            });
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
        let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let ss = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        // Build a host-side expected map and compare.
        let mut expected = std::collections::HashMap::<i32, i64>::new();
        for v in 0..12i64 {
            *expected.entry((v as i32) % 3).or_default() += v;
        }
        for i in 0..3 {
            let k = ks.value(i);
            let s = ss.value(i);
            assert_eq!(Some(&s), expected.get(&k).map(|x| x), "key={} sum={}", k, s);
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

        let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
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

        let ks = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let cs = out.column(1).as_any().downcast_ref::<Int64Array>().unwrap();

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
        let arr = finalize_welford_array(&states, &groups, WelfordOutKind::VarPop, &out_field)
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
        let arr = finalize_welford_array(&states, &groups, WelfordOutKind::VarSamp, &out_field)
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
        let arr = finalize_welford_array(&states, &groups, WelfordOutKind::VarPop, &out_field)
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
        let arr = finalize_welford_array(&states, &groups, WelfordOutKind::StddevPop, &out_field)
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
        let arr = finalize_welford_array(&states, &groups, WelfordOutKind::StddevSamp, &out_field)
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
        let arr = finalize_welford_array(&states, &groups, WelfordOutKind::VarPop, &out_field)
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
        let r = finalize_welford_array(&states, &groups, WelfordOutKind::VarPop, &out_field);
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
        assert!(
            is_spill,
            "spill error must match the soft-miss sentinel prefix"
        );

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

    // -------- F7-finish: grouped MIN/MAX(Date32 / Timestamp) rebuild --------

    /// `pack_array` rebuilds a grouped `Date32` MIN/MAX result (reduced on the
    /// i32-normalized accumulator) as a `Date32Array`.
    #[test]
    fn pack_array_rebuilds_date32() {
        let arr =
            pack_array(DataType::Date32, Scalars::I32(vec![19_000, 19_010])).expect("date32 pack");
        let d = arr
            .as_any()
            .downcast_ref::<arrow_array::Date32Array>()
            .expect("Date32Array");
        assert_eq!(d.values(), &[19_000, 19_010]);
    }

    /// `pack_array` rebuilds a grouped `Timestamp(Microsecond, "UTC")` MIN/MAX
    /// result preserving the unit + timezone.
    #[test]
    fn pack_array_rebuilds_timestamp_with_unit_and_tz() {
        let tz = crate::plan::logical_plan::intern_timezone("UTC");
        let arr = pack_array(
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz)),
            Scalars::I64(vec![100, 200]),
        )
        .expect("timestamp pack");
        assert_eq!(
            arr.data_type(),
            &arrow_schema::DataType::Timestamp(
                arrow_schema::TimeUnit::Microsecond,
                Some(Arc::from("UTC"))
            )
        );
        let t = arr
            .as_any()
            .downcast_ref::<arrow_array::TimestampMicrosecondArray>()
            .expect("TimestampMicrosecondArray");
        assert_eq!(t.values(), &[100, 200]);
    }

    /// `timestamp_array_from_i64` round-trips ticks for each unit and reattaches
    /// the timezone.
    #[test]
    fn timestamp_array_from_i64_dispatches_units() {
        let tz = crate::plan::logical_plan::intern_timezone("UTC");
        let sec = timestamp_array_from_i64(vec![1, 2], TimeUnit::Second, Some(tz));
        assert_eq!(
            sec.data_type(),
            &arrow_schema::DataType::Timestamp(
                arrow_schema::TimeUnit::Second,
                Some(Arc::from("UTC"))
            )
        );
        let nano = timestamp_array_from_i64(vec![9], TimeUnit::Nanosecond, None);
        assert_eq!(
            nano.data_type(),
            &arrow_schema::DataType::Timestamp(arrow_schema::TimeUnit::Nanosecond, None)
        );
    }

    /// `collect_filtered_timestamp` extracts the i64 tick buffer, dropping rows
    /// where the value is NULL (host-strip semantics shared with the numeric
    /// grouped paths).
    #[test]
    fn collect_filtered_timestamp_drops_value_nulls() {
        let arr: ArrayRef = Arc::new(arrow_array::TimestampMicrosecondArray::from(vec![
            Some(10i64),
            None,
            Some(30),
        ]));
        let vv: Vec<bool> = (0..arr.len()).map(|i| !arr.is_null(i)).collect();
        let out = collect_filtered_timestamp(arr.as_ref(), "ts", None, Some(&vv)).expect("extract");
        assert_eq!(out, vec![10, 30]);
    }

    // -------- Grouped Decimal128 SUM/MIN/MAX result-assembly + device --------

    /// Host-value parity: `build_agg_array` reads the per-group raw `i128`
    /// accumulator slots into a `Decimal128Array` carrying the declared output
    /// `(precision, scale)`. This is the pure-host result-assembly step shared
    /// by the SUM (precision widened to 38) and MIN/MAX (preserved) paths.
    #[test]
    fn build_agg_array_decimal128_reads_slots_with_dtype() {
        // Two groups occupying slots 3 and 7; per-group accumulator values.
        let mut acc = vec![0i128; 8];
        acc[3] = 12_345i128; // group A
        acc[7] = -6_789i128; // group B (negative)
        let groups = vec![(100i64, 3usize), (200i64, 7usize)];

        let agg = AggregateExpr::Sum(Expr::Column("v".into()));
        // SUM(Decimal128(10, 2)) -> Decimal128(38, 2).
        let out_field = Field::new("s", DataType::Decimal128(38, 2), true);
        let dl = AccDownload::Decimal128 {
            acc: acc.clone(),
            precision: 38,
            scale: 2,
        };
        let arr = build_agg_array(&agg, &out_field, &dl, &groups, groups.len()).expect("build");
        assert_eq!(
            arr.data_type(),
            &ArrowDataType::Decimal128(38, 2),
            "output dtype must carry the declared (p, s)"
        );
        let da = arr
            .as_any()
            .downcast_ref::<Decimal128Array>()
            .expect("Decimal128Array");
        assert_eq!(da.value(0), 12_345);
        assert_eq!(da.value(1), -6_789);
    }

    /// `build_agg_array` rejects a Decimal accumulator whose computed `(p, s)`
    /// disagrees with the output field — a guard that mirrors the scalar
    /// decimal path and prevents silently mislabelling the result scale.
    #[test]
    fn build_agg_array_decimal128_rejects_pscale_mismatch() {
        let acc = vec![1i128; 1];
        let groups = vec![(0i64, 0usize)];
        let agg = AggregateExpr::Min(Expr::Column("v".into()));
        // Field says (10,2) but the accumulator claims (10,3) -> error.
        let out_field = Field::new("m", DataType::Decimal128(10, 2), true);
        let dl = AccDownload::Decimal128 {
            acc,
            precision: 10,
            scale: 3,
        };
        assert!(
            build_agg_array(&agg, &out_field, &dl, &groups, 1).is_err(),
            "scale mismatch between field and accumulator must be rejected"
        );
    }

    /// SUM(Decimal128(p,s)) output dtype follows the engine rule: precision
    /// widened to 38, scale preserved. MIN/MAX preserve `(p,s)`. This pins the
    /// rule the GPU `run_typed_agg` arm mirrors against the planner.
    #[test]
    fn decimal_output_dtype_rule_matches_planner() {
        assert_eq!(
            sum_output_dtype(DataType::Decimal128(10, 2)),
            DataType::Decimal128(38, 2)
        );
        assert_eq!(
            sum_output_dtype(DataType::Decimal128(38, 6)),
            DataType::Decimal128(38, 6)
        );
    }

    // ---- Device (gpu:) execution tests for the grouped Decimal kernel. ----
    //
    // These drive the real keys kernel + `launch_decimal_agg_kernel` over a
    // small batch and verify per-group SUM/MIN/MAX, NULL exclusion, and SUM
    // overflow — the bit-for-bit-with-host contract. Gated `#[ignore] gpu:`
    // because they allocate on the device.

    /// Run the full GPU grouped-decimal pipeline for `op` over `(keys, vals)`
    /// pairs (already NULL-stripped) and return a host `slot -> i128` map keyed
    /// by the decoded group key, plus the overflow flag.
    #[cfg(test)]
    fn gpu_grouped_decimal(
        op: GroupedDecimalOp,
        keys: &[i64],
        vals: &[i128],
        identity: i128,
    ) -> BoltResult<(std::collections::HashMap<i64, i128>, bool)> {
        let n_rows = keys.len();
        let n_unique = unique_count(keys);
        let k = next_pow2((n_unique.saturating_mul(2)).saturating_add(16)).max(64);
        let k_u32 = u32::try_from(k).unwrap();

        let stream = CudaStream::null_or_default();
        let host_keys_init: Vec<i64> = vec![EMPTY_KEY; k];
        let mut keys_table = GpuVec::<i64>::from_slice_async(&host_keys_init, stream.raw())?;
        let key_col_gpu = GpuVec::<i64>::from_slice_async(keys, stream.raw())?;
        let overflow_counter = GpuVec::<u32>::zeros_async(1, stream.raw())?;
        let overflow_ptr: CUdeviceptr = overflow_counter.device_ptr();
        launch_keys_kernel(
            &key_col_gpu,
            &mut keys_table,
            n_rows,
            k_u32,
            &stream,
            overflow_ptr,
        )?;

        let input_gpu = GpuVec::<i128>::from_slice_async(vals, stream.raw())?;
        let init: Vec<i128> = vec![identity; k];
        let mut acc = GpuVec::<i128>::from_slice_async(&init, stream.raw())?;
        let overflowed = launch_decimal_agg_kernel(
            op,
            &key_col_gpu,
            &keys_table,
            &input_gpu,
            &mut acc,
            n_rows,
            k,
            k_u32,
            &stream,
            overflow_ptr,
        )?;

        let acc_host: Vec<i128> = {
            let pinned = acc.to_pinned_async(stream.raw())?;
            stream.synchronize()?;
            pinned.as_slice().to_vec()
        };
        let keys_host: Vec<i64> = {
            let pinned = keys_table.to_pinned_async(stream.raw())?;
            stream.synchronize()?;
            pinned.as_slice().to_vec()
        };
        let mut out = std::collections::HashMap::new();
        for (slot, &kv) in keys_host.iter().enumerate() {
            if kv != EMPTY_KEY {
                out.insert(kv, acc_host[slot]);
            }
        }
        Ok((out, overflowed))
    }

    /// gpu: grouped SUM(Decimal128) accumulates the correct per-group i128 sum,
    /// including values whose group sum crosses the 2^64 boundary (exercises
    /// the carry-chain add under the spin lock).
    #[test]
    #[ignore = "gpu: grouped Decimal128 SUM allocates on device"]
    fn gpu_grouped_decimal_sum_per_group() {
        // group 1: 3 + 4 = 7 ; group 2: (2^64) + 1 (crosses the word boundary)
        let big = 1i128 << 64;
        let keys = [1i64, 1, 2, 2];
        let vals = [3i128, 4, big, 1];
        let (m, of) = gpu_grouped_decimal(GroupedDecimalOp::Sum, &keys, &vals, 0).unwrap();
        assert!(!of, "no overflow expected");
        assert_eq!(m[&1], 7);
        assert_eq!(m[&2], big + 1);
    }

    /// gpu: grouped MIN/MAX(Decimal128) pick the correct extremum per group
    /// across the sign boundary (raw i128 ordering == decimal ordering).
    #[test]
    #[ignore = "gpu: grouped Decimal128 MIN/MAX allocates on device"]
    fn gpu_grouped_decimal_minmax_per_group() {
        let keys = [1i64, 1, 1, 2, 2];
        let vals = [-300i128, 50, -1000, 7, -7];

        let (mn, _) = gpu_grouped_decimal(GroupedDecimalOp::Min, &keys, &vals, i128::MAX).unwrap();
        assert_eq!(mn[&1], -1000);
        assert_eq!(mn[&2], -7);

        let (mx, _) = gpu_grouped_decimal(GroupedDecimalOp::Max, &keys, &vals, i128::MIN).unwrap();
        assert_eq!(mx[&1], 50);
        assert_eq!(mx[&2], 7);
    }

    /// gpu: NULLs are excluded by host-strip before upload, so a group whose
    /// only non-NULL contribution is a single value reduces to exactly that
    /// value (here the strip is done by the caller; this asserts the kernel
    /// only folds the rows it is handed).
    #[test]
    #[ignore = "gpu: grouped Decimal128 NULL handling allocates on device"]
    fn gpu_grouped_decimal_null_excluded() {
        // Caller already stripped NULLs: group 5 contributes only [100, 200].
        let keys = [5i64, 5];
        let vals = [100i128, 200];
        let (m, _) = gpu_grouped_decimal(GroupedDecimalOp::Sum, &keys, &vals, 0).unwrap();
        assert_eq!(m[&5], 300, "only the surviving (non-NULL) rows are summed");
    }

    /// gpu: a grouped SUM(Decimal128) whose per-group accumulator overflows
    /// i128 sets the device overflow flag (the executor turns this into the
    /// same error the scalar/host SUM path raises) rather than wrapping.
    #[test]
    #[ignore = "gpu: grouped Decimal128 SUM overflow allocates on device"]
    fn gpu_grouped_decimal_sum_overflow_flags() {
        let keys = [9i64, 9];
        let vals = [i128::MAX, 1];
        let (_m, of) = gpu_grouped_decimal(GroupedDecimalOp::Sum, &keys, &vals, 0).unwrap();
        assert!(
            of,
            "i128 overflow must raise the device overflow flag, not wrap"
        );
    }
}
