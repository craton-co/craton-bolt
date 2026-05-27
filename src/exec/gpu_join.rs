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
//! ## Stage-2 additions
//!
//! Stage 2 layers four host-visible capabilities on top of the Stage-1
//! kernels without changing the Stage-1 fast path:
//!
//! * **[`KeyShape`]-aware encoding** ([`encode_keys_for_shape`]) — Int32,
//!   Int64, Bool, Float32, Float64 single keys plus `TwoI32` composite
//!   pack `(hi as u32 << 32) | (lo as u32 as u64)`. Three-or-more keys
//!   fold to a single i64 via splitmix (`MultiI32(n)` and `TwoI64`).
//!   Lossy folds (`is_exact_in_i64() == false`) are gated off the GPU
//!   path; that lets us reuse the Stage-1 atomic-CAS lookup without
//!   per-pair host-side verification.
//! * **Outer joins** (`LEFT` / `RIGHT` / `FULL`) — orchestrated via
//!   [`execute_outer_join_on_gpu`]. The probe kernel atomically OR-sets
//!   bits in a `matched: u32[ceil(build_n_rows/32)]` bitmap; a second-
//!   pass [`compile_unmatched_build_kernel`] kernel emits the build-row
//!   index of every still-zero bit. The host pairs those indices with
//!   `None` in the take-indices array so `arrow::compute::take` NULL-pads
//!   the probe side.
//! * **Collision-list build/probe** — drops the Stage-1 "unique build
//!   keys" gate. The build kernel atomically prepends each row to a
//!   per-slot linked list (`head: u32[cap]` + `next_idx: u32[n_build]`),
//!   and the probe kernel walks that list emitting one output pair per
//!   visited build row. Used unconditionally for outer joins (so the
//!   matched-bitmap path stays linear in the number of build rows) and
//!   for inner joins with non-unique build keys.
//! * **VRAM-driven cap** — at first use, the executor queries the device's
//!   total memory via [`cuda_sys::device_total_mem`]. Cards with ≥ 8 GiB
//!   total VRAM bump the hash-table byte budget from 64 MiB to 512 MiB
//!   (=~ 44.7 M slots, ~22.4 M build rows under the 50% load factor).
//!   The cap stays at 64 MiB on smaller cards.
//!
//! ## Stage-3 follow-ups
//!
//! * Per-pair host-side verification so lossy-fold shapes (`MultiI32(n)`,
//!   `TwoI64`) can take the GPU path as a candidate filter.
//! * Utf8 keys (string interning + offsets array).
//! * Plumb runtime-tunable cap (`BOLT_GPU_JOIN_TABLE_CAP_MB`).

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;
use std::sync::OnceLock;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array,
    RecordBatch, UInt32Array,
};
use arrow_schema::Schema as ArrowSchema;

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;
use crate::exec::n_rows_to_u32;
use crate::jit::hash_join_kernel::{
    compile_build_collision_kernel, compile_build_kernel, compile_probe_collision_kernel,
    compile_probe_kernel, compile_unmatched_build_kernel, KeyShape, BUILD_COLLISION_KERNEL_ENTRY,
    BUILD_KERNEL_ENTRY, HASH_JOIN_BLOCK_SIZE, PROBE_COLLISION_KERNEL_ENTRY, PROBE_KERNEL_ENTRY,
    UNMATCHED_BUILD_KERNEL_ENTRY,
};
use crate::jit::jit_compiler::CudaModule;
use crate::plan::logical_plan::DataType;

/// Minimum size threshold (per side) below which the host hash join wins. The
/// GPU path eats a JIT-compile + h2d round trip; empirically 1024 rows on
/// either side is the break-even point on a discrete card.
pub const GPU_JOIN_MIN_ROWS: usize = 1024;

/// Conservative hash-table byte cap for cards with limited VRAM
/// (default fallback, used when the driver query fails or reports < 8 GiB).
/// Capacity ≈ `cap_bytes / 12` slots — each slot is one i64 key + one u32
/// row index. The probe-side output and the collision-list `head` + `next_idx`
/// buffers consume independent VRAM and are sized per-query.
const HASH_TABLE_BYTE_CAP_DEFAULT: usize = 64 * 1024 * 1024; // 64 MiB

/// Lifted byte cap for cards with ≥ 8 GiB of total VRAM. Driving this from
/// the device's reported total memory keeps the cap measure-driven without
/// needing user configuration. 512 MiB ≈ 44.7 M slots ⇒ ~22.3 M build rows
/// at the engine's 50% load factor.
const HASH_TABLE_BYTE_CAP_LARGE: usize = 512 * 1024 * 1024; // 512 MiB

/// VRAM threshold (bytes) above which the executor uses
/// [`HASH_TABLE_BYTE_CAP_LARGE`] instead of [`HASH_TABLE_BYTE_CAP_DEFAULT`].
const LARGE_VRAM_THRESHOLD: usize = 8 * 1024 * 1024 * 1024; // 8 GiB

/// `CU_DEVICE_ATTRIBUTE_TOTAL_MEMORY` isn't an attribute — the driver
/// surfaces total memory through `cuDeviceTotalMem_v2` directly. We keep
/// the name for documentation symmetry with the task spec.
///
/// Latched once on first use so the FFI cost is paid exactly once per
/// process. Stores `Some(usize)` on success, `Some(default)` on FFI error
/// (logged at debug level), so callers can always go through `unwrap_or`.
static HASH_TABLE_BYTE_CAP_CACHE: OnceLock<usize> = OnceLock::new();

/// Resolve the per-process hash-table byte cap. First call performs the
/// driver query; subsequent calls hit the latch.
///
/// Selection rule: total VRAM ≥ 8 GiB ⇒ 512 MiB cap; else 64 MiB cap.
/// On any FFI error we fall back to the 64 MiB default and emit a
/// debug-level log line — the GPU path stays correct, just smaller.
fn hash_table_byte_cap() -> usize {
    *HASH_TABLE_BYTE_CAP_CACHE.get_or_init(|| {
        // We deliberately swallow any error from the query: this is a
        // tuning knob, not a correctness gate. On any failure (cuda-stub
        // mode, driver glitch, …) fall back to the conservative default.
        let cap = match resolve_byte_cap_from_driver() {
            Ok(cap) => cap,
            Err(e) => {
                log::debug!(
                    "gpu_join: device total-memory query failed ({e}); \
                     falling back to {HASH_TABLE_BYTE_CAP_DEFAULT} bytes"
                );
                HASH_TABLE_BYTE_CAP_DEFAULT
            }
        };
        log::debug!("gpu_join: hash-table byte cap resolved to {cap} bytes");
        cap
    })
}

/// Pure-driver path for [`hash_table_byte_cap`]. Extracted so unit tests can
/// reason about it independent of the OnceLock latch.
fn resolve_byte_cap_from_driver() -> BoltResult<usize> {
    cuda_sys::init()?;
    // Single-GPU only. Multi-GPU plumbing is a separate workstream — when
    // it lands, this should query the engine's bound device.
    let dev = cuda_sys::device_get(0)?;
    let total = cuda_sys::device_total_mem(dev)?;
    Ok(if total >= LARGE_VRAM_THRESHOLD {
        HASH_TABLE_BYTE_CAP_LARGE
    } else {
        HASH_TABLE_BYTE_CAP_DEFAULT
    })
}

/// Hash-table slot cap given the active byte cap. `12 = sizeof(i64) + sizeof(u32)`.
fn hash_table_slot_cap() -> usize {
    hash_table_byte_cap() / 12
}

/// 50% peak load factor: capacity = next_pow2(2 * n_build_rows). Higher load
/// factors blow up probe lengths quickly; lower wastes memory. 0.5 is the
/// engine-wide convention (matches `groupby::pack_keys`).
const LOAD_FACTOR_DENOM: usize = 2;

/// Compute the hash-table capacity for `n_build_rows`: smallest power of two
/// ≥ `LOAD_FACTOR_DENOM * n_build_rows`. Returns `Err` if the result exceeds
/// the table-size cap.
fn compute_capacity(n_build_rows: usize) -> BoltResult<usize> {
    compute_capacity_with_slot_cap(n_build_rows, hash_table_slot_cap())
}

/// Inner helper exposed for unit tests so we can plug an explicit slot cap
/// without depending on a CUDA device being present (the runtime cap query
/// is OnceLock'd, which makes test ordering brittle).
fn compute_capacity_with_slot_cap(
    n_build_rows: usize,
    slot_cap: usize,
) -> BoltResult<usize> {
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
    if target > slot_cap {
        return Err(BoltError::Other(format!(
            "gpu_join: required capacity {target} exceeds hash-table slot cap {slot_cap}. \
             Fall back to host path."
        )));
    }
    let cap = target.next_power_of_two();
    if cap > slot_cap {
        return Err(BoltError::Other(format!(
            "gpu_join: rounded capacity {cap} exceeds hash-table slot cap {slot_cap}"
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

/// Splitmix-style 64-bit finaliser, identical to the kernel-side `FX_MUL`
/// shift-right-32 reduction. Used to fold multi-column tuples into a single
/// i64 on the host. Matches `crate::jit::hash_kernels::FX_MUL` so that
/// the host-side splatter agrees with kernel-side replays of the same input.
const HOST_FX_MUL: u64 = 0x9E37_79B9_7F4A_7C15;

#[inline]
fn host_splitmix(mut h: u64) -> u64 {
    // We do NOT do (h * FX_MUL) >> 32 here because that's the *slot* reduction;
    // we want the full 64-bit folded value so the *kernel-side* `(key * FX_MUL) >> 32`
    // still spreads it across the table. So this is the cheap fold: multiply
    // and xor-shift, matching the SplitMix64 finalizer family.
    h ^= h >> 33;
    h = h.wrapping_mul(HOST_FX_MUL);
    h ^= h >> 29;
    h = h.wrapping_mul(HOST_FX_MUL);
    h ^= h >> 32;
    h
}

/// Canonicalise an f64's bit pattern so every NaN hashes the same. This is
/// the engine-wide "NaNs are equal" convention shared with `distinct.rs` and
/// the rest of the float-key pipeline.
#[inline]
fn canonical_f64_bits(v: f64) -> u64 {
    if v.is_nan() {
        f64::NAN.to_bits()
    } else if v == 0.0 {
        // Collapse -0.0 and +0.0 to the same key: SQL treats them equal.
        0u64
    } else {
        v.to_bits()
    }
}

/// Same as [`canonical_f64_bits`] but for f32, widened to u32 then to u64
/// (high bits zeroed) so the i64 cast preserves the bit pattern exactly.
#[inline]
fn canonical_f32_bits(v: f32) -> u64 {
    let bits = if v.is_nan() {
        f32::NAN.to_bits()
    } else if v == 0.0 {
        0u32
    } else {
        v.to_bits()
    };
    bits as u64
}

/// Encode an Arrow key column (or set of key columns) to a `Vec<i64>` for
/// upload, using the host-side strategy implied by `shape`. The build and
/// probe sides MUST call this with the same `shape` so their kernel-side
/// hashes line up byte-for-byte.
///
/// `columns` carries one slice per key column for the row being encoded. For
/// single-key shapes the slice length is 1; for `TwoI32` / `TwoI64` it's 2;
/// for `MultiI32(n)` it's n.
///
/// Returns `Err` if any encoded value collides with the `i64::MIN`
/// empty-slot sentinel — the host then surfaces that to the caller, which
/// falls back to the host path for the whole query.
pub fn encode_keys_for_shape(
    columns: &[&dyn Array],
    shape: KeyShape,
) -> BoltResult<Vec<i64>> {
    if columns.is_empty() {
        return Err(BoltError::Other(
            "gpu_join: encode_keys_for_shape called with zero columns".into(),
        ));
    }
    let n = columns[0].len();
    for c in &columns[1..] {
        if c.len() != n {
            return Err(BoltError::Other(format!(
                "gpu_join: encode_keys_for_shape: row-count mismatch ({} vs {})",
                c.len(),
                n
            )));
        }
    }

    let mut out: Vec<i64> = Vec::with_capacity(n);
    match shape {
        // Single-column shapes route through the Stage-1 encoder for Int32 /
        // Int64 to keep PTX-level behaviour byte-stable. Bool / Float are
        // handled inline below.
        KeyShape::SingleI32 => return encode_keys_i64(columns[0], DataType::Int32),
        KeyShape::SingleI64 => return encode_keys_i64(columns[0], DataType::Int64),
        KeyShape::SingleBool => {
            let arr = columns[0]
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: SingleBool key column is not BooleanArray".into())
                })?;
            // BooleanArray::value follows the bit layout directly.
            for i in 0..n {
                out.push(arr.value(i) as i64);
            }
        }
        KeyShape::SingleF32 => {
            let arr = columns[0]
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: SingleF32 key column is not Float32Array".into())
                })?;
            for v in arr.values().iter() {
                let bits = canonical_f32_bits(*v) as i64;
                if bits == i64::MIN {
                    return Err(BoltError::Other(
                        "gpu_join: Float32 key collided with i64::MIN sentinel after canonicalisation".into(),
                    ));
                }
                out.push(bits);
            }
        }
        KeyShape::SingleF64 => {
            let arr = columns[0]
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: SingleF64 key column is not Float64Array".into())
                })?;
            for v in arr.values().iter() {
                let bits = canonical_f64_bits(*v) as i64;
                if bits == i64::MIN {
                    return Err(BoltError::Other(
                        "gpu_join: Float64 key collided with i64::MIN sentinel after canonicalisation".into(),
                    ));
                }
                out.push(bits);
            }
        }
        KeyShape::TwoI32 => {
            if columns.len() != 2 {
                return Err(BoltError::Other(format!(
                    "gpu_join: TwoI32 expects 2 columns, got {}",
                    columns.len()
                )));
            }
            let hi = columns[0]
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: TwoI32 high column is not Int32".into())
                })?;
            let lo = columns[1]
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| {
                    BoltError::Other("gpu_join: TwoI32 low column is not Int32".into())
                })?;
            for i in 0..n {
                // Same convention as groupby::pack_keys: first column in the
                // high half, second in the low half.
                let h = (hi.value(i) as u32 as u64) << 32;
                let l = lo.value(i) as u32 as u64;
                let packed = (h | l) as i64;
                if packed == i64::MIN {
                    return Err(BoltError::Other(
                        "gpu_join: TwoI32 composite key collided with i64::MIN sentinel".into(),
                    ));
                }
                out.push(packed);
            }
        }
        KeyShape::TwoI64 | KeyShape::MultiI32(_) => {
            // Lossy fold path. The host-side splitmix can collide; the
            // executor gates this off the GPU path until per-pair host-side
            // verification lands. Encoding is implemented because the
            // hash_join_indices_on_gpu_with_shape entry point still takes
            // a KeyShape and we want a single encoder rather than a forked
            // path. Callers must NOT take the GPU fast path with these
            // shapes today (see KeyShape::is_exact_in_i64).
            for i in 0..n {
                let mut h: u64 = 0;
                for col in columns {
                    let bits = match col.data_type() {
                        arrow_schema::DataType::Int32 => {
                            let a = col
                                .as_any()
                                .downcast_ref::<Int32Array>()
                                .ok_or_else(|| {
                                    BoltError::Other("gpu_join: multi-key column dtype said Int32 but downcast failed".into())
                                })?;
                            a.value(i) as u32 as u64
                        }
                        arrow_schema::DataType::Int64 => {
                            let a = col
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .ok_or_else(|| {
                                    BoltError::Other("gpu_join: multi-key column dtype said Int64 but downcast failed".into())
                                })?;
                            a.value(i) as u64
                        }
                        other => {
                            return Err(BoltError::Other(format!(
                                "gpu_join: multi-key column has unsupported dtype {other:?}"
                            )))
                        }
                    };
                    h = host_splitmix(h ^ bits);
                }
                let packed = h as i64;
                if packed == i64::MIN {
                    return Err(BoltError::Other(
                        "gpu_join: multi-key folded value collided with i64::MIN sentinel".into(),
                    ));
                }
                out.push(packed);
            }
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

// =========================================================================
// Stage 2: collision-list build/probe, outer-join orchestration,
// multi-key + bool/float keys.
// =========================================================================

/// Sentinel value used in the collision-list `head[]` and `next_idx[]` arrays
/// to denote "no more entries". Picked to match the kernel side
/// (`u32::MAX`).
const COLLISION_LIST_SENTINEL: u32 = u32::MAX;

/// Result of a Stage-2 GPU INNER join: index pairs in arbitrary order, ready
/// for `arrow::compute::take`.
pub struct GpuJoinIndices {
    /// Build-side row index for each emitted pair.
    pub build: UInt32Array,
    /// Probe-side row index for each emitted pair.
    pub probe: UInt32Array,
}

/// Result of a Stage-2 GPU OUTER join: index pairs that may include NULLs.
pub struct GpuOuterJoinIndices {
    /// Build-side row index per emitted output row. `None` for rows where the
    /// build side is NULL-padded (LEFT OUTER probe-only matches).
    pub build: Vec<Option<u32>>,
    /// Probe-side row index per emitted output row. `None` for rows where the
    /// probe side is NULL-padded (RIGHT/FULL OUTER unmatched-build pass).
    pub probe: Vec<Option<u32>>,
}

/// Stage-2 generalisation of [`hash_join_indices_on_gpu`]: runs the
/// collision-list build + probe kernels so duplicate build keys are handled
/// correctly. Used by the multi-key and outer-join paths; the Stage-1
/// unique-key fast path still routes through [`hash_join_indices_on_gpu`]
/// for the byte-stable single-key tests.
///
/// `shape` controls only the host-side encoding — the kernels see only i64
/// keys. The caller is responsible for verifying `shape.is_exact_in_i64()`.
pub fn hash_join_indices_on_gpu_with_shape(
    build_key_columns: &[&dyn Array],
    probe_key_columns: &[&dyn Array],
    shape: KeyShape,
) -> BoltResult<GpuJoinIndices> {
    if !shape.is_exact_in_i64() {
        return Err(BoltError::Other(format!(
            "gpu_join: lossy fold for shape {shape:?} would risk false matches; \
             fall back to host path"
        )));
    }
    let n_build = build_key_columns[0].len();
    let n_probe = probe_key_columns[0].len();

    if n_build == 0 || n_probe == 0 {
        return Ok(GpuJoinIndices {
            build: UInt32Array::from(Vec::<u32>::new()),
            probe: UInt32Array::from(Vec::<u32>::new()),
        });
    }

    let n_build_u32 = n_rows_to_u32(n_build)?;
    let n_probe_u32 = n_rows_to_u32(n_probe)?;

    let cap = compute_capacity(n_build)?;
    let cap_u32 = u32::try_from(cap)
        .map_err(|_| BoltError::Other(format!("gpu_join: cap {cap} doesn't fit in u32")))?;

    let build_keys_host = encode_keys_for_shape(build_key_columns, shape)?;
    let probe_keys_host = encode_keys_for_shape(probe_key_columns, shape)?;

    let build_keys_dev = GpuVec::<i64>::from_slice(&build_keys_host)?;
    let probe_keys_dev = GpuVec::<i64>::from_slice(&probe_keys_host)?;

    let keys_init: Vec<i64> = vec![i64::MIN; cap];
    let row_idx_init: Vec<u32> = vec![u32::MAX; cap];
    let head_init: Vec<u32> = vec![COLLISION_LIST_SENTINEL; cap];
    let next_idx_init: Vec<u32> = vec![COLLISION_LIST_SENTINEL; n_build];

    let mut keys_table_dev = GpuVec::<i64>::from_slice(&keys_init)?;
    let mut row_idx_table_dev = GpuVec::<u32>::from_slice(&row_idx_init)?;
    let mut head_dev = GpuVec::<u32>::from_slice(&head_init)?;
    let mut next_idx_dev = GpuVec::<u32>::from_slice(&next_idx_init)?;

    // Output buffer size: 2x (n_build + n_probe) — a generous estimate for
    // typical workloads where duplicate-key fan-out is small. Pathological
    // cartesian-explosion cases (e.g. every key duplicated thousands of
    // times) overflow the counter and the kernel keeps it climbing; we
    // detect on counter readback and fall through to host.
    let out_capacity_usize = n_build
        .checked_add(n_probe)
        .and_then(|x| x.checked_mul(2))
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

    launch_build_collision_kernel(
        &build_keys_dev,
        &mut keys_table_dev,
        &mut row_idx_table_dev,
        &mut head_dev,
        &mut next_idx_dev,
        n_build_u32,
        cap_u32,
        &stream,
    )?;

    let n_matches_raw = launch_probe_collision_kernel(
        &probe_keys_dev,
        &keys_table_dev,
        &head_dev,
        &next_idx_dev,
        &mut out_probe_idx_dev,
        &mut out_build_idx_dev,
        &mut out_counter_dev,
        None,
        n_probe_u32,
        n_build_u32,
        cap_u32,
        out_capacity_u32,
        &stream,
    )?;

    if n_matches_raw > out_capacity_u32 {
        return Err(BoltError::Other(format!(
            "gpu_join: probe kernel claimed {n_matches_raw} matches but \
             output buffer was sized for {out_capacity_u32}"
        )));
    }
    let n_matches = n_matches_raw as usize;
    let probe_idx_full = out_probe_idx_dev.to_vec()?;
    let build_idx_full = out_build_idx_dev.to_vec()?;
    let probe_idx: Vec<u32> = probe_idx_full.into_iter().take(n_matches).collect();
    let build_idx: Vec<u32> = build_idx_full.into_iter().take(n_matches).collect();

    Ok(GpuJoinIndices {
        build: UInt32Array::from(build_idx),
        probe: UInt32Array::from(probe_idx),
    })
}

/// Stage-2 OUTER orchestration. Returns `(build, probe)` index streams that
/// may contain `None` slots where one side is NULL-padded. The caller passes
/// `emit_unmatched_probe` for LEFT/FULL semantics and `emit_unmatched_build`
/// for RIGHT/FULL semantics.
pub fn execute_outer_join_indices_on_gpu(
    build_key_columns: &[&dyn Array],
    probe_key_columns: &[&dyn Array],
    shape: KeyShape,
    emit_unmatched_probe: bool,
    emit_unmatched_build: bool,
) -> BoltResult<GpuOuterJoinIndices> {
    if !shape.is_exact_in_i64() {
        return Err(BoltError::Other(format!(
            "gpu_join: outer-join lossy fold for shape {shape:?} would risk false matches"
        )));
    }
    let n_build = build_key_columns[0].len();
    let n_probe = probe_key_columns[0].len();

    // Empty-build / empty-probe degenerate cases — handle in the host
    // orchestrator (join.rs) since this layer is purely GPU.
    if n_build == 0 || n_probe == 0 {
        return Ok(GpuOuterJoinIndices {
            build: Vec::new(),
            probe: Vec::new(),
        });
    }

    let n_build_u32 = n_rows_to_u32(n_build)?;
    let n_probe_u32 = n_rows_to_u32(n_probe)?;

    let cap = compute_capacity(n_build)?;
    let cap_u32 = u32::try_from(cap)
        .map_err(|_| BoltError::Other(format!("gpu_join: cap {cap} doesn't fit in u32")))?;

    let build_keys_host = encode_keys_for_shape(build_key_columns, shape)?;
    let probe_keys_host = encode_keys_for_shape(probe_key_columns, shape)?;

    let build_keys_dev = GpuVec::<i64>::from_slice(&build_keys_host)?;
    let probe_keys_dev = GpuVec::<i64>::from_slice(&probe_keys_host)?;

    let keys_init: Vec<i64> = vec![i64::MIN; cap];
    let row_idx_init: Vec<u32> = vec![u32::MAX; cap];
    let head_init: Vec<u32> = vec![COLLISION_LIST_SENTINEL; cap];
    let next_idx_init: Vec<u32> = vec![COLLISION_LIST_SENTINEL; n_build];

    let mut keys_table_dev = GpuVec::<i64>::from_slice(&keys_init)?;
    let mut row_idx_table_dev = GpuVec::<u32>::from_slice(&row_idx_init)?;
    let mut head_dev = GpuVec::<u32>::from_slice(&head_init)?;
    let mut next_idx_dev = GpuVec::<u32>::from_slice(&next_idx_init)?;

    // matched: u32[ceil(build_n_rows / 32)], zero-initialised. We always
    // allocate it for OUTER even when only the LEFT case is requested,
    // because the collision probe kernel needs a real pointer (we pass
    // null only for INNER).
    let matched_words = n_build.div_ceil(32);
    let mut matched_dev = GpuVec::<u32>::zeros(matched_words)?;

    // Output buffer sizing: 2x (n_build + n_probe). Generous for typical
    // outer joins; cartesian-explosion cases overflow the counter (kernel
    // keeps incrementing past the writes) and fall through to host.
    let out_capacity_usize = n_build
        .checked_add(n_probe)
        .and_then(|x| x.checked_mul(2))
        .ok_or_else(|| BoltError::Other("gpu_join: outer output sizing overflowed".into()))?;
    let out_capacity_u32 = u32::try_from(out_capacity_usize).map_err(|_| {
        BoltError::Other(format!(
            "gpu_join: out_capacity {out_capacity_usize} doesn't fit in u32"
        ))
    })?;

    let mut out_probe_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_build_idx_dev = GpuVec::<u32>::zeros(out_capacity_usize)?;
    let mut out_counter_dev = GpuVec::<u32>::zeros(1)?;

    let stream = CudaStream::null();

    launch_build_collision_kernel(
        &build_keys_dev,
        &mut keys_table_dev,
        &mut row_idx_table_dev,
        &mut head_dev,
        &mut next_idx_dev,
        n_build_u32,
        cap_u32,
        &stream,
    )?;

    let n_matches_raw = launch_probe_collision_kernel(
        &probe_keys_dev,
        &keys_table_dev,
        &head_dev,
        &next_idx_dev,
        &mut out_probe_idx_dev,
        &mut out_build_idx_dev,
        &mut out_counter_dev,
        Some(&matched_dev),
        n_probe_u32,
        n_build_u32,
        cap_u32,
        out_capacity_u32,
        &stream,
    )?;

    if n_matches_raw > out_capacity_u32 {
        return Err(BoltError::Other(format!(
            "gpu_join: outer-join probe claimed {n_matches_raw} matches > capacity {out_capacity_u32}"
        )));
    }
    let n_matches = n_matches_raw as usize;
    let probe_idx_full = out_probe_idx_dev.to_vec()?;
    let build_idx_full = out_build_idx_dev.to_vec()?;

    // Host-side post-pass for LEFT/FULL: we need to know which probe rows
    // were NEVER matched. The kernel doesn't track that directly (it would
    // require a second u32 bitmap on the probe side); cheaper to derive it
    // from the matched-pair set on the host.
    let mut probe_was_matched: Vec<bool> = vec![false; n_probe];
    let mut build: Vec<Option<u32>> = Vec::with_capacity(n_matches + n_probe + n_build);
    let mut probe: Vec<Option<u32>> = Vec::with_capacity(n_matches + n_probe + n_build);

    for i in 0..n_matches {
        let p = probe_idx_full[i];
        let b = build_idx_full[i];
        build.push(Some(b));
        probe.push(Some(p));
        if (p as usize) < n_probe {
            probe_was_matched[p as usize] = true;
        }
    }

    // LEFT/FULL: emit (None, probe_row) for unmatched probe rows.
    if emit_unmatched_probe {
        for (p, matched) in probe_was_matched.iter().enumerate() {
            if !matched {
                build.push(None);
                probe.push(Some(p as u32));
            }
        }
    }

    // RIGHT/FULL: second-pass kernel emits build-row indices for unmatched
    // build rows; the host pairs each with a NULL probe index.
    if emit_unmatched_build {
        let mut out_unmatched_dev = GpuVec::<u32>::zeros(n_build)?;
        let mut out_unmatched_counter_dev = GpuVec::<u32>::zeros(1)?;
        let n_unmatched = launch_unmatched_build_kernel(
            &matched_dev,
            &mut out_unmatched_dev,
            &mut out_unmatched_counter_dev,
            n_build_u32,
            n_build_u32,
            &stream,
        )?;
        if n_unmatched > n_build_u32 {
            return Err(BoltError::Other(format!(
                "gpu_join: unmatched-build kernel claimed {n_unmatched} > n_build {n_build_u32}"
            )));
        }
        let n_unmatched = n_unmatched as usize;
        let unmatched_full = out_unmatched_dev.to_vec()?;
        for &b in unmatched_full.iter().take(n_unmatched) {
            build.push(Some(b));
            probe.push(None);
        }
    }

    Ok(GpuOuterJoinIndices { build, probe })
}

/// Launch the Stage-2 collision-list build kernel.
#[allow(clippy::too_many_arguments)]
fn launch_build_collision_kernel(
    build_keys_dev: &GpuVec<i64>,
    keys_table_dev: &mut GpuVec<i64>,
    row_idx_table_dev: &mut GpuVec<u32>,
    head_dev: &mut GpuVec<u32>,
    next_idx_dev: &mut GpuVec<u32>,
    n_build_rows: u32,
    cap: u32,
    stream: &CudaStream,
) -> BoltResult<()> {
    if n_build_rows == 0 {
        return Ok(());
    }
    let ptx = compile_build_collision_kernel()?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(BUILD_COLLISION_KERNEL_ENTRY)?;

    let mut build_keys_ptr: CUdeviceptr = build_keys_dev.device_ptr();
    let mut keys_table_ptr: CUdeviceptr = keys_table_dev.device_ptr();
    let mut row_idx_table_ptr: CUdeviceptr = row_idx_table_dev.device_ptr();
    let mut head_ptr: CUdeviceptr = head_dev.device_ptr();
    let mut next_idx_ptr: CUdeviceptr = next_idx_dev.device_ptr();
    let mut n_rows_u32: u32 = n_build_rows;
    let mut cap_u32: u32 = cap;

    let mut params: [*mut c_void; 7] = [
        &mut build_keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut row_idx_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut head_ptr as *mut CUdeviceptr as *mut c_void,
        &mut next_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_rows_u32 as *mut u32 as *mut c_void,
        &mut cap_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_build_rows.div_ceil(block).max(1);
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x, 1, 1,
            block, 1, 1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    stream.synchronize()?;
    Ok(())
}

/// Launch the Stage-2 collision-list probe kernel. `matched_dev` is `Some`
/// for outer joins (the kernel sets bits in it) and `None` for inner joins
/// (the kernel skips the OR via a null-pointer check).
#[allow(clippy::too_many_arguments)]
fn launch_probe_collision_kernel(
    probe_keys_dev: &GpuVec<i64>,
    keys_table_dev: &GpuVec<i64>,
    head_dev: &GpuVec<u32>,
    next_idx_dev: &GpuVec<u32>,
    out_probe_idx_dev: &mut GpuVec<u32>,
    out_build_idx_dev: &mut GpuVec<u32>,
    out_counter_dev: &mut GpuVec<u32>,
    matched_dev: Option<&GpuVec<u32>>,
    n_probe_rows: u32,
    n_build_rows: u32,
    cap: u32,
    out_capacity: u32,
    stream: &CudaStream,
) -> BoltResult<u32> {
    if n_probe_rows == 0 {
        return Ok(0);
    }
    let ptx = compile_probe_collision_kernel()?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(PROBE_COLLISION_KERNEL_ENTRY)?;

    let mut probe_keys_ptr: CUdeviceptr = probe_keys_dev.device_ptr();
    let mut keys_table_ptr: CUdeviceptr = keys_table_dev.device_ptr();
    let mut head_ptr: CUdeviceptr = head_dev.device_ptr();
    let mut next_idx_ptr: CUdeviceptr = next_idx_dev.device_ptr();
    let mut out_probe_idx_ptr: CUdeviceptr = out_probe_idx_dev.device_ptr();
    let mut out_build_idx_ptr: CUdeviceptr = out_build_idx_dev.device_ptr();
    let mut out_counter_ptr: CUdeviceptr = out_counter_dev.device_ptr();
    let mut matched_ptr: CUdeviceptr = matched_dev.map(|m| m.device_ptr()).unwrap_or(0);
    let mut n_probe_u32: u32 = n_probe_rows;
    let mut cap_u32: u32 = cap;
    let mut out_capacity_u32: u32 = out_capacity;
    let mut n_build_u32: u32 = n_build_rows;

    let mut params: [*mut c_void; 12] = [
        &mut probe_keys_ptr as *mut CUdeviceptr as *mut c_void,
        &mut keys_table_ptr as *mut CUdeviceptr as *mut c_void,
        &mut head_ptr as *mut CUdeviceptr as *mut c_void,
        &mut next_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_probe_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_build_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_counter_ptr as *mut CUdeviceptr as *mut c_void,
        &mut matched_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_probe_u32 as *mut u32 as *mut c_void,
        &mut cap_u32 as *mut u32 as *mut c_void,
        &mut out_capacity_u32 as *mut u32 as *mut c_void,
        &mut n_build_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_probe_rows.div_ceil(block).max(1);
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x, 1, 1,
            block, 1, 1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    stream.synchronize()?;
    let counter_host: Vec<u32> = out_counter_dev.to_vec()?;
    Ok(counter_host[0])
}

/// Launch the Stage-2 outer-join second-pass kernel.
fn launch_unmatched_build_kernel(
    matched_dev: &GpuVec<u32>,
    out_build_idx_dev: &mut GpuVec<u32>,
    out_counter_dev: &mut GpuVec<u32>,
    n_build_rows: u32,
    out_capacity: u32,
    stream: &CudaStream,
) -> BoltResult<u32> {
    if n_build_rows == 0 {
        return Ok(0);
    }
    let ptx = compile_unmatched_build_kernel()?;
    let module = CudaModule::from_ptx(&ptx)?;
    let function = module.function(UNMATCHED_BUILD_KERNEL_ENTRY)?;

    let mut matched_ptr: CUdeviceptr = matched_dev.device_ptr();
    let mut out_build_idx_ptr: CUdeviceptr = out_build_idx_dev.device_ptr();
    let mut out_counter_ptr: CUdeviceptr = out_counter_dev.device_ptr();
    let mut n_build_u32: u32 = n_build_rows;
    let mut out_capacity_u32: u32 = out_capacity;

    let mut params: [*mut c_void; 5] = [
        &mut matched_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_build_idx_ptr as *mut CUdeviceptr as *mut c_void,
        &mut out_counter_ptr as *mut CUdeviceptr as *mut c_void,
        &mut n_build_u32 as *mut u32 as *mut c_void,
        &mut out_capacity_u32 as *mut u32 as *mut c_void,
    ];

    let block: u32 = HASH_JOIN_BLOCK_SIZE;
    let grid_x: u32 = n_build_rows.div_ceil(block).max(1);
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            grid_x, 1, 1,
            block, 1, 1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    stream.synchronize()?;
    let counter_host: Vec<u32> = out_counter_dev.to_vec()?;
    Ok(counter_host[0])
}

/// End-to-end GPU INNER join over two `RecordBatch`es with a [`KeyShape`]
/// (multi-key + bool/float support). Differs from
/// [`execute_inner_join_on_gpu`] only in that the slot-finding step uses
/// the collision-list kernels rather than the Stage-1 unique-key fast path,
/// so duplicate build keys produce correct (multiple) matches.
#[allow(clippy::too_many_arguments)]
pub fn execute_inner_join_on_gpu_with_shape(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    build_is_left: bool,
    build_key_idxs: &[usize],
    probe_key_idxs: &[usize],
    shape: KeyShape,
    output_schema: Arc<ArrowSchema>,
) -> BoltResult<RecordBatch> {
    let (build_batch, probe_batch) = if build_is_left {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };

    let build_cols: Vec<&dyn Array> = build_key_idxs
        .iter()
        .map(|&i| build_batch.column(i).as_ref())
        .collect();
    let probe_cols: Vec<&dyn Array> = probe_key_idxs
        .iter()
        .map(|&i| probe_batch.column(i).as_ref())
        .collect();

    let idx = hash_join_indices_on_gpu_with_shape(&build_cols, &probe_cols, shape)?;

    let (left_idx, right_idx) = if build_is_left {
        (&idx.build, &idx.probe)
    } else {
        (&idx.probe, &idx.build)
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

/// End-to-end GPU OUTER join over two `RecordBatch`es.
///
/// `emit_unmatched_probe` is true for LEFT/FULL; `emit_unmatched_build` is
/// true for RIGHT/FULL. INNER doesn't reach this function (use
/// [`execute_inner_join_on_gpu_with_shape`] for the multi-key path).
///
/// `lhs` and `rhs` are the physical left and right sides of the join (in
/// `output_schema` order); `build_is_left` says which the build side is.
/// For LEFT outer, the build side is the right table (so unmatched probe
/// rows can be detected); for RIGHT outer the build is the left.
#[allow(clippy::too_many_arguments)]
pub fn execute_outer_join_on_gpu(
    lhs: &RecordBatch,
    rhs: &RecordBatch,
    build_is_left: bool,
    build_key_idxs: &[usize],
    probe_key_idxs: &[usize],
    shape: KeyShape,
    emit_unmatched_probe: bool,
    emit_unmatched_build: bool,
    output_schema: Arc<ArrowSchema>,
) -> BoltResult<RecordBatch> {
    let (build_batch, probe_batch) = if build_is_left {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };

    let build_cols: Vec<&dyn Array> = build_key_idxs
        .iter()
        .map(|&i| build_batch.column(i).as_ref())
        .collect();
    let probe_cols: Vec<&dyn Array> = probe_key_idxs
        .iter()
        .map(|&i| probe_batch.column(i).as_ref())
        .collect();

    let outer = execute_outer_join_indices_on_gpu(
        &build_cols,
        &probe_cols,
        shape,
        emit_unmatched_probe,
        emit_unmatched_build,
    )?;

    let (left_idx_vec, right_idx_vec) = if build_is_left {
        (outer.build, outer.probe)
    } else {
        (outer.probe, outer.build)
    };
    let left_idx_arr = UInt32Array::from(left_idx_vec);
    let right_idx_arr = UInt32Array::from(right_idx_vec);

    let mut output_cols: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    for col in lhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &left_idx_arr, None)
                .map_err(|e| BoltError::Other(format!("gpu_join: arrow take (left): {e}")))?,
        );
    }
    for col in rhs.columns() {
        output_cols.push(
            arrow::compute::take(col.as_ref(), &right_idx_arr, None)
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
        // Pin a deterministic slot cap so the test is independent of which
        // VRAM regime the host happens to be running on.
        let cap = HASH_TABLE_BYTE_CAP_DEFAULT / 12;
        // 50% load factor: capacity = next_pow2(2 * n).
        assert_eq!(compute_capacity_with_slot_cap(1, cap).unwrap(), 2);
        assert_eq!(compute_capacity_with_slot_cap(2, cap).unwrap(), 4);
        assert_eq!(compute_capacity_with_slot_cap(3, cap).unwrap(), 8); // 2*3 = 6 -> next_pow2 = 8
        assert_eq!(compute_capacity_with_slot_cap(4, cap).unwrap(), 8);
        assert_eq!(compute_capacity_with_slot_cap(1024, cap).unwrap(), 2048);
        assert_eq!(compute_capacity_with_slot_cap(1025, cap).unwrap(), 4096);
        // Just under the cap.
        assert!(compute_capacity_with_slot_cap(1_000_000, cap).is_ok());
    }

    #[test]
    fn compute_capacity_rejects_oversized() {
        let cap = HASH_TABLE_BYTE_CAP_DEFAULT / 12;
        // cap ≈ 5_592_405 slots; 2 * n must fit, so the cap itself is
        // already over the limit.
        assert!(compute_capacity_with_slot_cap(cap, cap).is_err());
    }

    #[test]
    fn large_vram_cap_is_512_mib() {
        // The lifted cap matches the documented Stage-2 number.
        assert_eq!(HASH_TABLE_BYTE_CAP_LARGE, 512 * 1024 * 1024);
        // The default cap stays at the Stage-1 64 MiB.
        assert_eq!(HASH_TABLE_BYTE_CAP_DEFAULT, 64 * 1024 * 1024);
        // 8 GiB threshold per the task spec.
        assert_eq!(LARGE_VRAM_THRESHOLD, 8 * 1024 * 1024 * 1024);
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

    // ===== Stage 2 host-side helpers =====

    #[test]
    fn two_i32_packing_matches_pack_keys_convention() {
        // Pack convention: first column in high 32 bits, second in low 32.
        // Identical to `groupby::pack_keys` so kernel/host hashes line up
        // across the join/groupby boundary.
        let a = Int32Array::from(vec![1i32, -1, 0, i32::MAX]);
        let b = Int32Array::from(vec![2i32, -2, 5, i32::MIN]);
        let enc =
            encode_keys_for_shape(&[&a as &dyn Array, &b as &dyn Array], KeyShape::TwoI32).unwrap();
        assert_eq!(enc[0], ((1u32 as u64) << 32 | (2u32 as u64)) as i64);
        // -1 high, -2 low → both as u32, OR'd at the right offset.
        assert_eq!(enc[1], (((-1i32) as u32 as u64) << 32 | ((-2i32) as u32 as u64)) as i64);
        assert_eq!(enc[2], (0u64 << 32 | 5u64) as i64);
        assert_eq!(
            enc[3],
            (((i32::MAX as u32 as u64) << 32) | (i32::MIN as u32 as u64)) as i64
        );
    }

    #[test]
    fn bool_keys_encode_as_zero_one() {
        let a = arrow_array::BooleanArray::from(vec![true, false, true, false]);
        let enc = encode_keys_for_shape(&[&a as &dyn Array], KeyShape::SingleBool).unwrap();
        assert_eq!(enc, vec![1, 0, 1, 0]);
    }

    #[test]
    fn float_nan_canonicalisation_collapses_distinct_nan_patterns() {
        // f64::NAN.to_bits() is the canonical pattern; any other NaN bit
        // pattern should reduce to it before hashing.
        let weird_nan = f64::from_bits(f64::NAN.to_bits() ^ 0x1);
        assert!(weird_nan.is_nan());
        let a = arrow_array::Float64Array::from(vec![f64::NAN, weird_nan, 1.0]);
        let enc = encode_keys_for_shape(&[&a as &dyn Array], KeyShape::SingleF64).unwrap();
        assert_eq!(enc[0], enc[1], "all NaNs must canonicalise to the same key");
        assert_ne!(enc[0], enc[2]);
    }

    #[test]
    fn float_negative_zero_collapses_to_positive_zero() {
        // SQL treats -0.0 == +0.0; our encoder must too.
        let a = arrow_array::Float64Array::from(vec![-0.0, 0.0]);
        let enc = encode_keys_for_shape(&[&a as &dyn Array], KeyShape::SingleF64).unwrap();
        assert_eq!(enc[0], enc[1]);
        assert_eq!(enc[0], 0);
    }

    #[test]
    fn lossy_fold_shapes_decline_gpu_path() {
        let a = Int64Array::from(vec![1i64, 2, 3]);
        let b = Int64Array::from(vec![10i64, 20, 30]);
        let cols: Vec<&dyn Array> = vec![&a, &b];
        // hash_join_indices_on_gpu_with_shape must surface a clear error
        // for lossy shapes — host orchestrator translates to fall-through.
        let err = hash_join_indices_on_gpu_with_shape(&cols, &cols, KeyShape::TwoI64);
        assert!(err.is_err(), "TwoI64 must decline (lossy fold)");
        let err = hash_join_indices_on_gpu_with_shape(&cols, &cols, KeyShape::MultiI32(3));
        assert!(err.is_err(), "MultiI32 must decline (lossy fold)");
    }

    #[test]
    fn host_splitmix_is_deterministic_and_avalanches() {
        // Same input -> same output (purity).
        assert_eq!(host_splitmix(0), host_splitmix(0));
        assert_eq!(host_splitmix(1), host_splitmix(1));
        // Single-bit input flip changes ~half the output bits — loose
        // avalanche check (≥ 16 of 64 bits differ for nearby inputs).
        let h0 = host_splitmix(1);
        let h1 = host_splitmix(2);
        assert_ne!(h0, h1);
        let differing = (h0 ^ h1).count_ones();
        assert!(
            differing >= 16,
            "splitmix should differ in at least 16 bits across nearby inputs; got {differing}"
        );
    }
}
