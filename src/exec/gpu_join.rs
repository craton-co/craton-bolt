// SPDX-License-Identifier: Apache-2.0

//! GPU-side INNER JOIN: build a hash table on the GPU from the smaller
//! (build) side, probe it with the larger (probe) side, materialise the
//! result host-side via `arrow::compute::take`.
//!
//! Pairs with [`crate::jit::hash_join_kernel`], which emits the PTX. Flow:
//!
//! ```text
//!  build keys (host, Int32/Int64, n_build)
//!     │
//!     ├─ encode -> i64           (sign-extend Int32; bitcast Int64)
//!     │
//!     ▼ h2d
//!  build_keys_dev (GpuVec<i64>)
//!     │
//!     ├─ keys_table_dev      (GpuVec<i64>, cap, init=i64::MIN)
//!     └─ row_idx_table_dev   (GpuVec<u32>, cap, init=u32::MAX)
//!     │
//!     ▼ launch BUILD kernel (1 thread / build row)
//!  fully-populated (keys_table_dev, row_idx_table_dev)
//!
//!  probe keys (host, same dtype, n_probe)
//!     │
//!     ├─ encode -> i64
//!     │
//!     ▼ h2d
//!  probe_keys_dev (GpuVec<i64>)
//!     │
//!     ├─ out_probe_idx_dev   (GpuVec<u32>, out_capacity)
//!     ├─ out_build_idx_dev   (GpuVec<u32>, out_capacity)
//!     └─ out_counter_dev     (GpuVec<u32>, length 1, init=0)
//!     │
//!     ▼ launch PROBE kernel (1 thread / probe row)
//!  out buffers populated up to *counter[0]* entries (arbitrary order)
//!     │
//!     ▼ d2h
//!  (probe_indices: Vec<u32>, build_indices: Vec<u32>)
//!     │
//!     ▼ arrow::compute::take per column on both sides
//!  joined RecordBatch
//! ```
//!
//! ## Stage 1 scope
//!
//! * **INNER only.** LEFT/RIGHT/FULL/CROSS stay host-side (J1's path).
//! * **Single equi-key.** Multi-key joins are Stage 2.
//! * **Int32 or Int64 key dtype.** Float / Bool / Utf8 fall through to host.
//! * **No NULLs in keys.** SQL NULL-keys-never-match drops the row from the
//!   inner-join output anyway; the host path enforces that, so we gate on
//!   `null_count() == 0` and re-use that contract here.
//! * **Both sides ≥ 1024 rows.** Below this, host wins (upload + JIT load).
//! * **Build side ≤ ~2.8M rows** (`2 * build_n_rows * 12 ≤ 64 MiB`).
//! * **Build keys are unique on the join column.** A collision in the build
//!   side's keys would lose all but one match in the row_idx_table slot. The
//!   host check is conservative (set-cardinality probe before upload), but
//!   even without it the probe-kernel correctness for the unique case is the
//!   bigger ROI for Stage 1.
//! * **No build key equals `i64::MIN`.** That value is the empty-slot
//!   sentinel; conflict would corrupt the build kernel's CAS step.
//!
//! On any gate miss we surface `Ok(None)`; the caller (the host path in
//! `crate::exec::join`) handles the input correctly.
//!
//! ## Stage-2 follow-ups
//!
//! * Multi-key joins (composite key packing, à la `groupby::pack_keys`).
//! * OUTER variants by adding a "matched" bitmap to the hash table and a
//!   second pass.
//! * Build-side duplicates via a per-slot collision list (or a "spill"
//!   buffer).
//! * Lift the 64 MiB cap on newer cards.
//! * Float / Bool / Utf8 keys.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Int32Array, Int64Array, RecordBatch, UInt32Array,
};
use arrow_schema::Schema as ArrowSchema;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;
use crate::exec::n_rows_to_u32;
use crate::jit::hash_join_kernel::{
    compile_build_kernel, compile_probe_kernel, BUILD_KERNEL_ENTRY,
    HASH_JOIN_BLOCK_SIZE, PROBE_KERNEL_ENTRY,
};
use crate::jit::jit_compiler::CudaModule;
use crate::plan::logical_plan::DataType;

/// Minimum size threshold (per side) below which the host hash join wins. The
/// GPU path eats a JIT-compile + h2d round trip; empirically 1024 rows on
/// either side is the break-even point on a discrete card.
pub const GPU_JOIN_MIN_ROWS: usize = 1024;

/// Hash-table memory cap (bytes). Capacity ≈ `cap_bytes / (8 + 4)` slots
/// because each slot needs an i64 key and a u32 row index. We over-cap
/// conservatively to leave room for the probe-side output buffer too.
const HASH_TABLE_BYTE_CAP: usize = 64 * 1024 * 1024; // 64 MiB
/// Maximum number of slots given the byte cap. `12 = sizeof(i64) + sizeof(u32)`.
const HASH_TABLE_SLOT_CAP: usize = HASH_TABLE_BYTE_CAP / 12;

/// 50% peak load factor: capacity = next_pow2(2 * n_build_rows). Higher load
/// factors blow up probe lengths quickly; lower wastes memory. 0.5 is the
/// engine-wide convention (matches `groupby::pack_keys`).
const LOAD_FACTOR_DENOM: usize = 2;

/// Compute the hash-table capacity for `n_build_rows`: smallest power of two
/// ≥ `LOAD_FACTOR_DENOM * n_build_rows`. Returns `Err` if the result exceeds
/// the table-size cap.
fn compute_capacity(n_build_rows: usize) -> BoltResult<usize> {
    let target = n_build_rows
        .checked_mul(LOAD_FACTOR_DENOM)
        .ok_or_else(|| {
            BoltError::Other(format!(
                "gpu_join: capacity calc overflowed (n_build_rows={n_build_rows})"
            ))
        })?;
    // next_power_of_two on 1 returns 1; we want a minimum of 2 so the mask
    // (cap - 1) has at least one valid bit. Probe loop assumes cap >= 2.
    let target = target.max(2);
    if target > HASH_TABLE_SLOT_CAP {
        return Err(BoltError::Other(format!(
            "gpu_join: required capacity {target} exceeds hash-table slot cap {HASH_TABLE_SLOT_CAP} \
             (64 MiB total). Fall back to host path."
        )));
    }
    let cap = target.next_power_of_two();
    if cap > HASH_TABLE_SLOT_CAP {
        return Err(BoltError::Other(format!(
            "gpu_join: rounded capacity {cap} exceeds hash-table slot cap {HASH_TABLE_SLOT_CAP}"
        )));
    }
    Ok(cap)
}

/// Encode an Arrow key column to a `Vec<i64>` for upload. The build and probe
/// sides share this encoding so their hashes agree byte-for-byte.
///
/// Returns `Err` if any encoded value collides with the `i64::MIN` empty-slot
/// sentinel — that would corrupt the kernel's CAS step.
fn encode_keys_i64(column: &dyn Array, dtype: DataType) -> BoltResult<Vec<i64>> {
    let n = column.len();
    let mut out: Vec<i64> = Vec::with_capacity(n);
    match dtype {
        DataType::Int32 => {
            let arr = column
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: column dtype said Int32 but downcast failed".into())
                })?;
            // Int32 can't equal i64::MIN once sign-extended (range is
            // [i32::MIN..=i32::MAX] which doesn't include i64::MIN). No
            // sentinel-collision check needed for Int32.
            for v in arr.values().iter() {
                out.push(*v as i64);
            }
        }
        DataType::Int64 => {
            let arr = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: column dtype said Int64 but downcast failed".into())
                })?;
            for v in arr.values().iter() {
                if *v == i64::MIN {
                    return Err(BoltError::Other(
                        "gpu_join: build/probe key equals the i64::MIN empty-slot sentinel; \
                         falling back to host path"
                            .into(),
                    ));
                }
                out.push(*v);
            }
        }
        other => {
            return Err(BoltError::Other(format!(
                "gpu_join: unsupported key dtype {other:?} (Stage 1: Int32 / Int64 only)"
            )));
        }
    }
    Ok(out)
}

/// Run the build kernel: insert (key, row_idx) for every build row into the
/// device hash table.
fn launch_build_kernel(
    build_keys_dev: &GpuVec<i64>,
    keys_table_dev: &mut GpuVec<i64>,
    row_idx_table_dev: &mut GpuVec<u32>,
    n_build_rows: u32,
    cap: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_build_rows == 0 {
        // Empty build side: keys_table_dev is already initialised to all
        // i64::MIN, so the probe will see every slot empty and emit no
        // matches. This path shouldn't be reached anyway (the gate rejects
        // empty sides) but defensive bailout matches the sort kernel's
        // n_pow2 <= 1 short-circuit.
        return Ok(());
    }

    let ptx = compile_build_kernel()?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(BUILD_KERNEL_ENTRY)?;

    let mut build_keys_ptr: CUdeviceptr = build_keys_dev.device_ptr();
    let mut keys_table_ptr: CUdeviceptr = keys_table_dev.device_ptr();
    let mut row_idx_table_ptr: CUdeviceptr = row_idx_table_dev.device_ptr();
    let mut n_rows_u32: u32 = n_build_rows;
    let mut cap_u32: u32 = cap;

    let mut params: [*mut c_void; 5] = [
        &mut build_keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut row_idx_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut cap_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_build_rows.div_ceil(block).max(1);

    // SAFETY: every entry of `params` points at a stack local that outlives
    // the launch+sync below; the device buffers are owned by the caller and
    // outlive the launch.
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
    Ok(())
}

/// Run the probe kernel: for each probe row, walk the hash table and emit
/// `(probe_idx, build_idx)` into the output buffers via an atomic counter.
///
/// Returns the number of matches actually claimed (the post-launch value of
/// the GPU-side counter), capped at `out_capacity`. If the kernel claimed
/// more than `out_capacity` slots the counter will still hold the true count
/// (the kernel only skips the *writes* on overflow), so callers can detect
/// the overflow and re-run with a bigger output buffer.
fn launch_probe_kernel(
    probe_keys_dev: &GpuVec<i64>,
    keys_table_dev: &GpuVec<i64>,
    row_idx_table_dev: &GpuVec<u32>,
    out_probe_idx_dev: &mut GpuVec<u32>,
    out_build_idx_dev: &mut GpuVec<u32>,
    out_counter_dev: &mut GpuVec<u32>,
    n_probe_rows: u32,
    cap: u32,
    out_capacity: u32,
    stream: &CudaStream,
) -> BoltResult<u32> {
    if n_probe_rows == 0 {
        // No probe rows -> no matches; counter stays at 0.
        return Ok(0);
    }

    let ptx = compile_probe_kernel()?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(PROBE_KERNEL_ENTRY)?;

    let mut probe_keys_ptr: CUdeviceptr = probe_keys_dev.device_ptr();
    let mut keys_table_ptr: CUdeviceptr = keys_table_dev.device_ptr();
    let mut row_idx_table_ptr: CUdeviceptr = row_idx_table_dev.device_ptr();
    let mut out_probe_idx_ptr: CUdeviceptr = out_probe_idx_dev.device_ptr();
    let mut out_build_idx_ptr: CUdeviceptr = out_build_idx_dev.device_ptr();
    let mut out_counter_ptr: CUdeviceptr = out_counter_dev.device_ptr();
    let mut n_probe_u32: u32 = n_probe_rows;
    let mut cap_u32: u32 = cap;
    let mut out_capacity_u32: u32 = out_capacity;

    let mut params: [*mut c_void; 9] = [
        &mut probe_keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut row_idx_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_probe_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_build_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_counter_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_probe_u32 as *mut u32 as *mut c_void,
        &mut cap_u32 as *mut u32 as *mut c_void,
        &mut out_capacity_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_probe_rows.div_ceil(block).max(1);

    // SAFETY: same rationale as launch_build_kernel.
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

    // Read back the actual number of matches.
    let counter_host: Vec<u32> = out_counter_dev.to_vec()?;
    let n_matches_raw = counter_host[0];
    Ok(n_matches_raw)
}

/// Execute a single-key INNER equi-join on the GPU.
///
/// Returns the two index arrays `(build_indices, probe_indices)` in
/// *arbitrary order* — the host caller is expected to either accept that
/// ordering (INNER doesn't promise one) or sort post-hoc.
///
/// `build_keys_col` and `probe_keys_col` must have the same `dtype` (the
/// caller validates this); the executor only checks at the entry into
/// `encode_keys_i64`.
pub fn hash_join_indices_on_gpu(
    build_keys_col: &dyn Array,
    probe_keys_col: &dyn Array,
    dtype: DataType,
) -> BoltResult<(UInt32Array, UInt32Array)> {
    let n_build = build_keys_col.len();
    let n_probe = probe_keys_col.len();

    // Trivial empty-side short-circuit: no matches possible.
    if n_build == 0 || n_probe == 0 {
        return Ok((
            UInt32Array::from(Vec::<u32>::new()),
            UInt32Array::from(Vec::<u32>::new()),
        ));
    }

    // n_build and n_probe must fit in u32 for the kernel launch shape.
    let n_build_u32 = n_rows_to_u32(n_build)?;
    let n_probe_u32 = n_rows_to_u32(n_probe)?;

    let cap = compute_capacity(n_build)?;
    let cap_u32 = u32::try_from(cap).map_err(|_| {
        BoltError::Other(format!("gpu_join: cap {cap} doesn't fit in u32"))
    })?;

    // Encode + upload both key columns.
    let build_keys_host = encode_keys_i64(build_keys_col, dtype)?;
    let probe_keys_host = encode_keys_i64(probe_keys_col, dtype)?;

    let build_keys_dev = GpuVec::<i64>::from_slice(&build_keys_host)?;
    let probe_keys_dev = GpuVec::<i64>::from_slice(&probe_keys_host)?;

    // Hash table buffers: keys init to i64::MIN, row_idx init to u32::MAX.
    let keys_init: Vec<i64> = vec![i64::MIN; cap];
    let row_idx_init: Vec<u32> = vec![u32::MAX; cap];
    let mut keys_table_dev = GpuVec::<i64>::from_slice(&keys_init)?;
    let mut row_idx_table_dev = GpuVec::<u32>::from_slice(&row_idx_init)?;

    // Output buffers. We pre-size for the worst INNER-equi case under the
    // unique-build-key invariant: every probe row matches at most one build
    // row, so n_probe is a safe upper bound. We add n_build as a safety
    // pad just in case (Stage 2: tight sizing for non-unique builds).
    let out_capacity_usize = n_probe
        .checked_add(n_build)
        .ok_or_else(|| BoltError::Other("gpu_join: output buffer size overflow".into()))?;
    let out_capacity_u32 = u32::try_from(out_capacity_usize).map_err(|_| {
        BoltError::Other(format!(
            "gpu_join: out_capacity {out_capacity_usize} doesn't fit in u32"
        ))
    })?;

    let mut out_probe_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_build_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_counter_dev = GpuVec::<u32>::zeros(1)?;

    let stream = CudaStream::null();

    // Build phase.
    launch_build_kernel(
        &build_keys_dev,
        &mut keys_table_dev,
        &mut row_idx_table_dev,
        n_build_u32,
        cap_u32,
        &stream,
    )?;

    // Probe phase.
    let n_matches_raw = launch_probe_kernel(
        &probe_keys_dev,
        &keys_table_dev,
        &row_idx_table_dev,
        &mut out_probe_idx_dev,
        &mut out_build_idx_dev,
        &mut out_counter_dev,
        n_probe_u32,
        cap_u32,
        out_capacity_u32,
        &stream,
    )?;

    if n_matches_raw > out_capacity_u32 {
        // Overflow: the kernel saw more matches than we sized for. This
        // shouldn't happen under the INNER + unique-build invariant
        // enforced by the gate, but if it does we surface a clear error
        // rather than silently truncating.
        return Err(BoltError::Other(format!(
            "gpu_join: probe kernel claimed {n_matches_raw} matches but \
             output buffer was sized for {out_capacity_u32}; \
             likely a build-side duplicate-key violation. Fall back to host path."
        )));
    }

    let n_matches = n_matches_raw as usize;

    // Download the index pairs.
    let probe_idx_full = out_probe_idx_dev.to_vec()?;
    let build_idx_full = out_build_idx_dev.to_vec()?;

    // Drop trailing buffers; we want the first n_matches entries.
    let probe_idx: Vec<u32> = probe_idx_full.into_iter().take(n_matches).collect();
    let build_idx: Vec<u32> = build_idx_full.into_iter().take(n_matches).collect();

    Ok((
        UInt32Array::from(build_idx),
        UInt32Array::from(probe_idx),
    ))
}

/// End-to-end GPU INNER join over two `RecordBatch`es.
///
/// `build_key_idx` and `probe_key_idx` point at the join key columns within
/// the build / probe batches (the caller picks which side builds — typically
/// the smaller one). `dtype` is the (validated-equal) key dtype.
///
/// `lhs` and `rhs` are passed through as the *physical* left and right side
/// of the join, used purely for the final `take` on every column. The
/// (build_indices, probe_indices) pair from the GPU is re-oriented into
/// (left_indices, right_indices) according to `build_is_left`.
///
/// Returns a new `RecordBatch` with the joined rows. The output schema is
/// `output_schema` — the disambiguated combined schema computed by the
/// planner (left ++ right). Output row ordering is *unspecified* (atomic-
/// counter race in the probe kernel).
pub fn execute_inner_join_on_gpu(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    build_is_left: bool,
    build_key_idx: usize,
    probe_key_idx: usize,
    dtype: DataType,
    output_schema: Arc<ArrowSchema>,
) -> BoltResult<RecordBatch> {
    let (build_batch, probe_batch) = if build_is_left {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };

    let build_keys_col = build_batch.column(build_key_idx);
    let probe_keys_col = probe_batch.column(probe_key_idx);

    let (build_indices, probe_indices) =
        hash_join_indices_on_gpu(build_keys_col.as_ref(), probe_keys_col.as_ref(), dtype)?;

    // Re-orient (build, probe) -> (left, right).
    let (left_idx, right_idx) = if build_is_left {
        (&build_indices, &probe_indices)
    } else {
        (&probe_indices, &build_indices)
    };

    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), left_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: arrow take (left): {e}")))?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), right_idx, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: arrow take (right): {e}")))?,
        );
    }

    RecordBatch::try_new(output_schema, output_cols)
        .map_err(|e| BoltError::Other(format!("gpu_join: building RecordBatch failed: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_schema::{DataType as ArrowDataType, Field, Schema as ArrowSchema};

    // -- Pure-host helpers --

    #[test]
    fn compute_capacity_powers_of_two() {
        // 50% load factor: capacity = next_pow2(2 * n).
        assert_eq!(compute_capacity(1).unwrap(), 2);
        assert_eq!(compute_capacity(2).unwrap(), 4);
        assert_eq!(compute_capacity(3).unwrap(), 8); // 2*3 = 6 -> next_pow2 = 8
        assert_eq!(compute_capacity(4).unwrap(), 8);
        assert_eq!(compute_capacity(1024).unwrap(), 2048);
        assert_eq!(compute_capacity(1025).unwrap(), 4096);
        // Just under the cap.
        assert!(compute_capacity(1_000_000).is_ok());
    }

    #[test]
    fn compute_capacity_rejects_oversized() {
        // HASH_TABLE_SLOT_CAP = 64 MiB / 12 ≈ 5_592_405 slots; 2 * n must
        // fit, so any n above ~2.7M should be rejected.
        let oversized = HASH_TABLE_SLOT_CAP; // 2 * this is > cap.
        assert!(compute_capacity(oversized).is_err());
    }

    #[test]
    fn encode_int32_sign_extends() {
        let arr = Int32Array::from(vec![1i32, -1, i32::MIN, i32::MAX]);
        let enc = encode_keys_i64(&arr, DataType::Int32).unwrap();
        assert_eq!(enc, vec![1i64, -1, i32::MIN as i64, i32::MAX as i64]);
        // None of these can equal i64::MIN, so no sentinel collision.
        assert!(enc.iter().all(|v| *v != i64::MIN));
    }

    #[test]
    fn encode_int64_identity() {
        let arr = Int64Array::from(vec![0i64, 1, -1, i64::MAX, i64::MIN + 1]);
        let enc = encode_keys_i64(&arr, DataType::Int64).unwrap();
        assert_eq!(enc, vec![0i64, 1, -1, i64::MAX, i64::MIN + 1]);
    }

    #[test]
    fn encode_int64_rejects_sentinel() {
        let arr = Int64Array::from(vec![0i64, i64::MIN, 1]);
        let err = encode_keys_i64(&arr, DataType::Int64);
        assert!(
            err.is_err(),
            "i64::MIN in input must be rejected as a sentinel collision"
        );
    }

    #[test]
    fn encode_rejects_unsupported_dtype() {
        let arr = arrow_array::Float64Array::from(vec![1.0, 2.0]);
        assert!(encode_keys_i64(&arr, DataType::Float64).is_err());
    }

    // -- GPU round-trip --

    /// Build two batches with a known overlap, run the GPU join, and verify
    /// the recovered match set matches the host-computed answer. The
    /// arbitrary-order output is reconciled by sorting both sides on
    /// (probe_idx, build_idx) before comparison.
    #[test]
    #[ignore = "requires CUDA toolkit + driver at runtime"]
    fn gpu_hash_join_int32_round_trip() {
        // Build side: 2000 unique keys 0..2000, payload = key + 1000.
        // Probe side: 4000 keys, every other one matching a build key.
        let n_build = 2000usize;
        let n_probe = 4000usize;

        let build_keys: Vec<i32> = (0..n_build as i32).collect();
        let build_payload: Vec<i32> = build_keys.iter().map(|k| k + 1000).collect();
        let probe_keys: Vec<i32> = (0..n_probe as i32).map(|i| i % 3000).collect();
        let probe_payload: Vec<i32> = (0..n_probe as i32).map(|i| 10_000 + i).collect();

        let build_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("bp", ArrowDataType::Int32, false),
        ]));
        let build_batch = RecordBatch::try_new(
            build_schema,
            vec![
                Arc::new(Int32Array::from(build_keys.clone())),
                Arc::new(Int32Array::from(build_payload.clone())),
            ],
        )
        .unwrap();

        let probe_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("pp", ArrowDataType::Int32, false),
        ]));
        let probe_batch = RecordBatch::try_new(
            probe_schema,
            vec![
                Arc::new(Int32Array::from(probe_keys.clone())),
                Arc::new(Int32Array::from(probe_payload.clone())),
            ],
        )
        .unwrap();

        let (build_idx, probe_idx) = hash_join_indices_on_gpu(
            build_batch.column(0).as_ref(),
            probe_batch.column(0).as_ref(),
            DataType::Int32,
        )
        .expect("gpu join");

        assert_eq!(build_idx.len(), probe_idx.len(), "matched pair count must agree");

        // Reconstruct the host-side expected set.
        let mut expected: Vec<(u32, u32)> = Vec::new();
        for (pi, pk) in probe_keys.iter().enumerate() {
            if (*pk as usize) < n_build {
                expected.push((*pk as u32, pi as u32));
            }
        }
        expected.sort_unstable();

        let mut got: Vec<(u32, u32)> = (0..build_idx.len())
            .map(|i| (build_idx.value(i), probe_idx.value(i)))
            .collect();
        got.sort_unstable();

        assert_eq!(got, expected, "GPU join match set must equal host expected");
    }

    /// End-to-end test through `execute_inner_join_on_gpu`: same fixture as
    /// above but exercises the full take + concat path.
    #[test]
    #[ignore = "requires CUDA toolkit + driver at runtime"]
    fn gpu_inner_join_full_batch_round_trip() {
        let n_build = 2000usize;
        let n_probe = 4000usize;
        let build_keys: Vec<i32> = (0..n_build as i32).collect();
        let build_payload: Vec<i32> = build_keys.iter().map(|k| k + 1000).collect();
        let probe_keys: Vec<i32> = (0..n_probe as i32).map(|i| i % 3000).collect();
        let probe_payload: Vec<i32> = (0..n_probe as i32).map(|i| 10_000 + i).collect();

        let build_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("bp", ArrowDataType::Int32, false),
        ]));
        let build_batch = RecordBatch::try_new(
            build_schema,
            vec![
                Arc::new(Int32Array::from(build_keys.clone())),
                Arc::new(Int32Array::from(build_payload.clone())),
            ],
        )
        .unwrap();

        let probe_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("pp", ArrowDataType::Int32, false),
        ]));
        let probe_batch = RecordBatch::try_new(
            probe_schema,
            vec![
                Arc::new(Int32Array::from(probe_keys.clone())),
                Arc::new(Int32Array::from(probe_payload.clone())),
            ],
        )
        .unwrap();

        // Output schema = left (build) ++ right (probe).
        let out_schema = Arc::new(ArrowSchema::new(vec![
            Field::new("k", ArrowDataType::Int32, false),
            Field::new("bp", ArrowDataType::Int32, false),
            Field::new("k_2", ArrowDataType::Int32, false),
            Field::new("pp", ArrowDataType::Int32, false),
        ]));

        let out = execute_inner_join_on_gpu(
            &build_batch,
            &probe_batch,
            /* build_is_left */ true,
            0,
            0,
            DataType::Int32,
            out_schema,
        )
        .expect("gpu inner join");

        // Expected match count: probe rows whose key < n_build = 2000.
        // probe_keys = 0..4000 % 3000 -> keys 0..2999 each appear at least
        // once; specifically the matches are those probe_keys < 2000.
        let expected: usize = probe_keys.iter().filter(|k| (**k as usize) < n_build).count();
        assert_eq!(out.num_rows(), expected, "match count must match host estimate");

        // Spot-check that every output row satisfies the equi-join:
        // build_payload column (col 1) = build_key + 1000 = probe_key + 1000.
        let bp_col = out
            .column(1)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        let pk_col = out
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        for i in 0..out.num_rows() {
            assert_eq!(
                bp_col.value(i),
                pk_col.value(i) + 1000,
                "row {i}: bp must equal probe_key + 1000 (left.k == right.k invariant)"
            );
        }
    }

    /// 0-row probe side: must produce an empty result without panicking.
    #[test]
    #[ignore = "requires CUDA toolkit + driver at runtime"]
    fn gpu_hash_join_empty_probe() {
        let build_keys: Vec<i32> = (0..1024).collect();
        let probe_keys: Vec<i32> = Vec::new();

        let (build_idx, probe_idx) = hash_join_indices_on_gpu(
            &Int32Array::from(build_keys),
            &Int32Array::from(probe_keys),
            DataType::Int32,
        )
        .expect("gpu join");

        assert_eq!(build_idx.len(), 0);
        assert_eq!(probe_idx.len(), 0);
    }

    /// Int64 keys, same shape as the Int32 round-trip.
    #[test]
    #[ignore = "requires CUDA toolkit + driver at runtime"]
    fn gpu_hash_join_int64_round_trip() {
        let n_build = 1500usize;
        let n_probe = 3000usize;
        let build_keys: Vec<i64> = (0..n_build as i64).collect();
        let probe_keys: Vec<i64> = (0..n_probe as i64).map(|i| i % 2000).collect();

        let (build_idx, probe_idx) = hash_join_indices_on_gpu(
            &Int64Array::from(build_keys),
            &Int64Array::from(probe_keys.clone()),
            DataType::Int64,
        )
        .expect("gpu join");

        let mut expected: Vec<(u32, u32)> = Vec::new();
        for (pi, pk) in probe_keys.iter().enumerate() {
            if (*pk as usize) < n_build {
                expected.push((*pk as u32, pi as u32));
            }
        }
        expected.sort_unstable();

        let mut got: Vec<(u32, u32)> = (0..build_idx.len())
            .map(|i| (build_idx.value(i), probe_idx.value(i)))
            .collect();
        got.sort_unstable();

        assert_eq!(got, expected);
    }
}
