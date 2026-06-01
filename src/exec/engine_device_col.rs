// SPDX-License-Identifier: Apache-2.0

//! Heterogenous owned device output column ([`DeviceCol`]) and its
//! download / staging helpers, lifted out of `exec::engine` (pure
//! reorganization; no behavior change).

use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Decimal128Array, Float32Array, Float64Array, Int32Array, Int64Array,
};

use crate::cuda::cuda_sys::CUdeviceptr;
use crate::cuda::dictionary::DictionaryColumn;
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::launch::CudaStream;
use crate::plan::DataType;

/// Heterogenous owned device column. Keeps each `GpuVec<T>` alive past the kernel launch.
///
/// Used only for OUTPUT buffers in `execute_projection`. Input columns are
/// resolved through `GpuTable` (uploaded once at table-registration time) and
/// fed to kernels as raw `CUdeviceptr`s; the upload-from-Arrow path that used
/// to live here as `DeviceCol::upload` is gone — `GpuColumn::upload` in
/// `gpu_table.rs` is the single source of truth for host→device column
/// uploads. The historical `BoolNullable` and `Borrowed` variants and the
/// `utf8_dictionary` accessor went with it; both were only reachable through
/// `upload`.
pub(crate) enum DeviceCol {
    /// 32-bit signed integer column.
    I32(GpuVec<i32>),
    /// 64-bit signed integer column.
    I64(GpuVec<i64>),
    /// 32-bit float column.
    F32(GpuVec<f32>),
    /// 64-bit float column.
    F64(GpuVec<f64>),
    /// Bool stored as one byte per row (0 / 1). Used when the source Arrow
    /// array has no nulls.
    Bool(GpuVec<u8>),
    /// Utf8 stored as i32 dictionary indices; host dictionary lives alongside.
    Utf8(DictionaryColumn),
    /// v0.7 sub-task B: 128-bit fixed-point output column. Stored as the
    /// same interleaved `[lo0, hi0, lo1, hi1, ...]` u64 buffer the input
    /// `GpuColumnData::Decimal128` uses, so the PTX `Op::Store128` can
    /// write 16 bytes per row at offset `tid * 16` with no per-row
    /// indirection. The plan-level `(precision, scale)` rides along so
    /// the download path can reattach them to the resulting
    /// `Decimal128Array`.
    Decimal128 {
        /// Interleaved 16-bytes-per-row output buffer (length `2 * n_rows`).
        values: GpuVec<u64>,
        /// Plan-level precision (digits of significance).
        precision: u8,
        /// Plan-level scale.
        scale: i8,
        /// Optional Arrow-LE packed validity bitmap on the device, one byte
        /// per 8 rows (lsb-first) — mirrors
        /// [`GpuColumnData::Decimal128`](crate::exec::gpu_table::GpuColumnData::Decimal128)'s
        /// `valid_mask`. For pure passthrough columns we copy the source
        /// column's mask so the download path can reconstruct NULL rows as
        /// NULL rather than `0`. `None` ⇒ all rows valid (no nulls on the
        /// source, or a freshly-allocated output buffer).
        valid_mask: Option<GpuVec<u8>>,
    },
}

impl DeviceCol {
    /// Allocate a zero-initialised device column of `n` rows.
    ///
    /// Utf8 outputs allocate an empty dictionary; the engine must replace it
    /// with the source column's dictionary before download (today this only
    /// works for pure column-passthrough projections — `output_schema` field
    /// name matching an input column name).
    pub(crate) fn alloc_zeros(dtype: DataType, n: usize) -> BoltResult<Self> {
        match dtype {
            DataType::Int32 => Ok(DeviceCol::I32(GpuVec::<i32>::zeros(n)?)),
            DataType::Int64 => Ok(DeviceCol::I64(GpuVec::<i64>::zeros(n)?)),
            DataType::Float32 => Ok(DeviceCol::F32(GpuVec::<f32>::zeros(n)?)),
            DataType::Float64 => Ok(DeviceCol::F64(GpuVec::<f64>::zeros(n)?)),
            DataType::Bool => Ok(DeviceCol::Bool(GpuVec::<u8>::zeros(n)?)),
            DataType::Utf8 => Ok(DeviceCol::Utf8(DictionaryColumn {
                dictionary: Vec::new(),
                indices: GpuVec::<i32>::zeros(n)?,
                n_rows: n,
            })),
            // v0.7 sub-task B: allocate the interleaved [lo, hi] u64 buffer
            // (length `2 * n`) that `Op::Store128` writes into. Plan-level
            // `(precision, scale)` rides on the variant so the download path
            // can rebuild a `Decimal128Array` with the correct dtype.
            DataType::Decimal128(precision, scale) => Ok(DeviceCol::Decimal128 {
                values: GpuVec::<u64>::zeros(2 * n)?,
                precision,
                scale,
                // Freshly-allocated output buffer: no validity yet. A
                // passthrough column copies the source mask in after alloc
                // (see the output-allocation loop in `run_kernel`).
                valid_mask: None,
            }),
            // v0.7: PTX codegen for Date32 / Timestamp arithmetic is wired
            // (see `crate::jit::ptx_gen`), but the device-side download
            // path is dtype-blind — `DeviceCol::I32::download` always
            // emits an `Int32Array`, which would silently downgrade a
            // Date32 output to plain Int32. Keep the engine boundary
            // rejecting these types until a follow-up wires the
            // Date32Array / TimestampArray reconstruction. The
            // physical-plan codegen still produces correct PTX for
            // `Date32 - Date32` and `Timestamp - Timestamp`; the
            // top-level engine routes any temporal column through the
            // host path until then.
            DataType::Date32 | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
                "Date/Timestamp output column lowering pending download-path \
                 wiring (PTX codegen is done; got {:?})",
                dtype
            ))),
        }
    }

    /// Raw device pointer for kernel-parameter assembly.
    pub(crate) fn device_ptr(&self) -> CUdeviceptr {
        match self {
            DeviceCol::I32(v) => v.device_ptr(),
            DeviceCol::I64(v) => v.device_ptr(),
            DeviceCol::F32(v) => v.device_ptr(),
            DeviceCol::F64(v) => v.device_ptr(),
            DeviceCol::Bool(v) => v.device_ptr(),
            DeviceCol::Utf8(d) => d.indices.device_ptr(),
            // v0.7 sub-task B: the interleaved [lo, hi] u64 buffer is
            // the column's single base pointer — PTX `Op::Store128`
            // computes per-row offsets as `tid * 16`.
            DeviceCol::Decimal128 { values, .. } => values.device_ptr(),
        }
    }

    /// Record `stream` as having launched a kernel against every device
    /// buffer this output column owns, so each buffer's `Drop` fences `stream`
    /// before its block is recycled to the pool.
    ///
    /// `execute_projection` assembles its kernel parameters by hand and drives
    /// a raw `cuLaunchKernel` off `device_ptr()` rather than through
    /// [`KernelArgs`](crate::exec::launch::KernelArgs)/`launch_1d`, so it does
    /// not get the central `tag_launch_stream` enforcement that the
    /// `launch_1d` / `launch_with_geometry` callers rely on (review finding
    /// V-1 / F10a). Calling this immediately after the launch restores the
    /// same `Drop`-fence invariant for the freshly-allocated output buffers:
    /// the launch stream is recorded in each buffer's `StreamSet` exactly as
    /// `KernelArgs::tag_launch_stream` would, so a buffer dropped while the
    /// kernel is still in flight fences the stream before recycling — even if
    /// a future edit removes a downstream `synchronize()`. Delegates to the
    /// public [`GpuVec::mark_stream_use`], the documented entry point for
    /// callers that bypass `KernelArgs`.
    pub(crate) fn mark_launch_stream(&self, stream: crate::cuda::CUstream) {
        match self {
            DeviceCol::I32(v) => v.mark_stream_use(stream),
            DeviceCol::I64(v) => v.mark_stream_use(stream),
            DeviceCol::F32(v) => v.mark_stream_use(stream),
            DeviceCol::F64(v) => v.mark_stream_use(stream),
            DeviceCol::Bool(v) => v.mark_stream_use(stream),
            DeviceCol::Utf8(d) => d.indices.mark_stream_use(stream),
            DeviceCol::Decimal128 {
                values, valid_mask, ..
            } => {
                values.mark_stream_use(stream);
                if let Some(mask) = valid_mask {
                    mask.mark_stream_use(stream);
                }
            }
        }
    }

    /// Install a dictionary on a Utf8 column (for output columns whose source dictionary
    /// the engine knows). No-op for non-Utf8 columns.
    pub(crate) fn set_utf8_dictionary(&mut self, dict: Vec<String>) {
        if let DeviceCol::Utf8(d) = self {
            d.dictionary = dict;
        }
    }

    /// Install a device-side validity bitmap on a Decimal128 output column
    /// (for pure passthrough projections whose source column carries one).
    /// No-op for non-Decimal128 columns or a `None` mask. Mirrors
    /// [`Self::set_utf8_dictionary`]'s passthrough plumbing.
    pub(crate) fn set_decimal128_valid_mask(&mut self, mask: Option<GpuVec<u8>>) {
        if let DeviceCol::Decimal128 { valid_mask, .. } = self {
            if mask.is_some() {
                *valid_mask = mask;
            }
        }
    }

    /// Copy the device column back to a host Arrow array of length `n_rows`.
    pub(crate) fn download(self, n_rows: usize) -> BoltResult<ArrayRef> {
        match self {
            DeviceCol::I32(v) => {
                let host = copy_back::<i32>(&v, n_rows)?;
                Ok(Arc::new(Int32Array::from(host)) as ArrayRef)
            }
            DeviceCol::I64(v) => {
                let host = copy_back::<i64>(&v, n_rows)?;
                Ok(Arc::new(Int64Array::from(host)) as ArrayRef)
            }
            DeviceCol::F32(v) => {
                let host = copy_back::<f32>(&v, n_rows)?;
                Ok(Arc::new(Float32Array::from(host)) as ArrayRef)
            }
            DeviceCol::F64(v) => {
                let host = copy_back::<f64>(&v, n_rows)?;
                Ok(Arc::new(Float64Array::from(host)) as ArrayRef)
            }
            DeviceCol::Bool(v) => {
                let host = copy_back::<u8>(&v, n_rows)?;
                let bools: Vec<bool> = host.into_iter().map(|b| b != 0).collect();
                Ok(Arc::new(BooleanArray::from(bools)) as ArrayRef)
            }
            DeviceCol::Utf8(d) => {
                let arr = d.to_string_array()?;
                Ok(Arc::new(arr) as ArrayRef)
            }
            // v0.7 sub-task B: reassemble the interleaved [lo, hi] u64
            // buffer back into a `Decimal128Array`. Each pair of u64s
            // reconstitutes one i128 via
            //   `lo | ((hi as u128) << 64)` then `as i128`
            // which preserves the sign because the high half carries
            // the sign bits unchanged through the unsigned/signed cast.
            DeviceCol::Decimal128 {
                values,
                precision,
                scale,
                valid_mask,
            } => {
                let host = copy_back::<u64>(&values, 2 * n_rows)?;
                // Decimal128 NULL fix: download the validity bitmap (if any)
                // so NULL rows reconstruct as Arrow NULL, not `0`.
                let mask_bits = valid_mask.as_ref().map(|m| m.to_vec()).transpose()?;
                let arr = decimal128_from_interleaved(
                    &host,
                    n_rows,
                    mask_bits.as_deref(),
                    precision,
                    scale,
                    "Decimal128 download",
                )?;
                Ok(Arc::new(arr) as ArrayRef)
            }
        }
    }

    /// Stage-3 async download: enqueue D2H from every primitive variant
    /// into pinned host buffers on `stream`, then synchronize ONCE and
    /// build the Arrow arrays from the resulting `Vec`s. Behaves
    /// identically to [`download`] for the Utf8 / Borrowed variants —
    /// those don't currently have a pinned fast path.
    ///
    /// The caller is responsible for ensuring `stream` is the same one
    /// the producing kernel was launched on (so the D2H sees committed
    /// results), and the function performs the synchronize internally
    /// before reading the pinned buffer.
    pub(crate) fn download_pinned(
        self,
        n_rows: usize,
        stream: &CudaStream,
    ) -> BoltResult<ArrayRef> {
        match self {
            DeviceCol::I32(v) => {
                let staged = StagedDownload::<i32>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Int32Array::from(host)) as ArrayRef)
            }
            DeviceCol::I64(v) => {
                let staged = StagedDownload::<i64>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Int64Array::from(host)) as ArrayRef)
            }
            DeviceCol::F32(v) => {
                let staged = StagedDownload::<f32>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Float32Array::from(host)) as ArrayRef)
            }
            DeviceCol::F64(v) => {
                let staged = StagedDownload::<f64>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                Ok(Arc::new(Float64Array::from(host)) as ArrayRef)
            }
            DeviceCol::Bool(v) => {
                let staged = StagedDownload::<u8>::from_gpu(&v, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), n_rows)?;
                let bools: Vec<bool> = host.into_iter().map(|b| b != 0).collect();
                Ok(Arc::new(BooleanArray::from(bools)) as ArrayRef)
            }
            DeviceCol::Utf8(_) => {
                // Utf8 doesn't (yet) have a pinned fast path — fall back
                // to the sync download. The stream has already been
                // synchronized above for the primitive siblings, so this
                // is safe to invoke regardless.
                self.download(n_rows)
            }
            // v0.7 sub-task B: Decimal128's pinned path mirrors the
            // primitive pattern (u64 element type, length `2 * n_rows`).
            // The check_len guard catches a buffer that didn't get sized
            // correctly at alloc time.
            DeviceCol::Decimal128 {
                values,
                precision,
                scale,
                valid_mask,
            } => {
                let staged = StagedDownload::<u64>::from_gpu(&values, stream.raw())?;
                stream.synchronize()?;
                let host = staged.into_vec();
                check_len(host.len(), 2 * n_rows)?;
                // Decimal128 NULL fix: same validity-aware reassembly as the
                // sync `download` site (shared helper keeps them consistent).
                let mask_bits = valid_mask.as_ref().map(|m| m.to_vec()).transpose()?;
                let arr = decimal128_from_interleaved(
                    &host,
                    n_rows,
                    mask_bits.as_deref(),
                    precision,
                    scale,
                    "Decimal128 pinned download",
                )?;
                Ok(Arc::new(arr) as ArrayRef)
            }
        }
    }
}

/// Tiny invariant check used by the pinned-download path: every
/// `DeviceCol` output buffer is sized at allocation time to `n_rows`, so
/// a length mismatch on download is a bug, not a runtime condition.
pub(crate) fn check_len(have: usize, want: usize) -> BoltResult<()> {
    if have != want {
        return Err(BoltError::Other(format!(
            "internal: device buffer length {} did not match expected {}",
            have, want
        )));
    }
    Ok(())
}

/// Decimal128 NULL fix — shared reassembly used by BOTH download sites
/// (`DeviceCol::download` and `DeviceCol::download_pinned`) so they cannot
/// drift. Reconstruct each row's `i128` from the interleaved `[lo, hi]` u64
/// pair, then attach Arrow validity from the (optional, lsb-first packed)
/// `mask_bits`: a row whose validity bit is 0 becomes an Arrow NULL rather
/// than the zeroed bit-pattern it was stored as. `mask_bits == None` ⇒ every
/// row is valid (non-null source), preserving the original non-null
/// behaviour byte-for-byte.
///
/// `host` must be `2 * n_rows` u64s (already length-checked by callers).
fn decimal128_from_interleaved(
    host: &[u64],
    n_rows: usize,
    mask_bits: Option<&[u8]>,
    precision: u8,
    scale: i8,
    ctx: &str,
) -> BoltResult<Decimal128Array> {
    let mut out: Vec<Option<i128>> = Vec::with_capacity(n_rows);
    for row in 0..n_rows {
        let lo = host[2 * row];
        let hi = host[2 * row + 1];
        let bits = (lo as u128) | ((hi as u128) << 64);
        // lsb-first packed bitmap: bit `row % 8` of byte `row / 8`. Absent
        // mask ⇒ all rows valid.
        let is_valid = match mask_bits {
            None => true,
            Some(b) => {
                let byte = b.get(row / 8).copied().unwrap_or(0);
                (byte >> (row % 8)) & 1 == 1
            }
        };
        out.push(if is_valid { Some(bits as i128) } else { None });
    }
    // `FromIterator<Option<i128>>` builds the array with the correct null
    // bitmap; `with_precision_and_scale` reattaches the plan dtype.
    out.into_iter()
        .collect::<Decimal128Array>()
        .with_precision_and_scale(precision, scale)
        .map_err(|e| {
            BoltError::Type(format!(
                "{ctx}: precision/scale ({precision}, {scale}) rejected by Arrow: {e}"
            ))
        })
}

/// Copy back a `GpuVec<T>` into a host `Vec<T>` of length `n_rows`.
///
/// Output buffers are allocated via `GpuVec::zeros(n_rows)`, whose `len()` is `n_rows`,
/// so `to_vec()` returns exactly that many elements.
pub(crate) fn copy_back<T>(v: &GpuVec<T>, n_rows: usize) -> BoltResult<Vec<T>>
where
    T: bytemuck::Pod,
{
    let host = v.to_vec()?;
    if host.len() != n_rows {
        return Err(BoltError::Other(format!(
            "internal: device buffer length {} did not match expected {}",
            host.len(),
            n_rows
        )));
    }
    Ok(host)
}

/// Stage-3 D2H staging buffer: async-copies a `GpuVec<T>` into a
/// page-locked host buffer on a caller-supplied stream, synchronises
/// once, and produces a regular `Vec<T>` for Arrow consumption.
///
/// Why a separate type vs. an inline call? Arrow array constructors
/// (`Int32Array::from(Vec<i32>)`) want owned `Vec`s with the standard
/// allocator — they will NOT accept a `PinnedHostBuffer` as a
/// zero-copy backing buffer (the lifecycle is incompatible: pinned
/// memory must be released via `cuMemFreeHost`, while Arrow buffers
/// release through the global allocator). So the pinned hop is purely
/// to get a true DMA without staging through a kernel-managed bounce
/// buffer; the final `.to_vec()` is the one host-host copy we keep.
///
/// Usage:
///
/// ```ignore
/// let staged = StagedDownload::from_gpu(&gpu_vec, stream.raw())?;
/// stream.synchronize()?;
/// let arrow_vec: Vec<i32> = staged.into_vec();
/// ```
struct StagedDownload<T: bytemuck::Pod> {
    pinned: crate::cuda::PinnedHostBuffer<T>,
}

impl<T: bytemuck::Pod> StagedDownload<T> {
    /// Enqueue an async D2H from `v` into a fresh pinned host buffer on
    /// `stream`. The caller MUST synchronize `stream` before calling
    /// [`into_vec`] / borrowing the pinned slice.
    fn from_gpu(v: &GpuVec<T>, stream: crate::cuda::CUstream) -> BoltResult<Self> {
        let pinned = v.to_pinned_async(stream)?;
        Ok(Self { pinned })
    }

    /// Consume the staged download and produce a regular host `Vec<T>`.
    ///
    /// Assumes the caller has synchronized the stream — there is no way
    /// to detect "not yet synchronized" without an event, which we skip
    /// in Stage 3. Calling this before sync produces uninitialised
    /// bytes (defined behaviour for `T: Pod` but functionally
    /// incorrect).
    fn into_vec(self) -> Vec<T> {
        self.pinned.as_slice().to_vec()
    }
}
