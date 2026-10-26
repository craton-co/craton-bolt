// SPDX-License-Identifier: Apache-2.0
//! Device-resident operator pipelines: operators that hand their results to
//! the next GPU operator as device columns instead of a host batch.
//!
//! [`Engine::execute`] returns host batches, so a GPU operator feeding
//! another one used to download its result and the consumer upload it again.
//! [`Engine::execute_device`] instead returns a [`GpuTable`] for the subplans
//! it can keep on the device:
//!
//! * a fused projection kernel (with or without a predicate): its output
//!   buffers, compacted on the device, with SQL NULLs as device validity;
//! * an inner equi-join on one integer key over two such inputs: the hash
//!   build / probe runs on the device keys, and every output column is
//!   gathered on the device by the matched row indices;
//! * a `Project` of bare columns over one of those (a column selection).
//!
//! Consumers: an aggregate over a joined input (`AggregateOverInput`) binds
//! the device table as its input, so its kernels read the joined columns in
//! place (a host copy is downloaded only if a host fallback asks for one); a
//! row count reads the row count; a join returned to the caller downloads
//! only its final columns. Anything else — Utf8 / decimal columns, NULL join
//! keys, other operators — returns `None` and runs on the host path as
//! before.

use std::ffi::c_void;
use std::ptr;
use std::sync::Arc;

use arrow_array::{
    ArrayRef, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use arrow_schema::{Field as ArrowField, Schema as ArrowSchema};

use crate::cuda::cuda_sys::{self, CUdeviceptr};
use crate::cuda::GpuVec;
use crate::error::{BoltError, BoltResult};
use crate::exec::engine::Engine;
use crate::exec::gpu_table::{GpuColumn, GpuColumnData, GpuTable};
use crate::exec::launch::CudaStream;
use crate::plan::logical_plan::{DataType, Expr, JoinType, Schema};
use crate::plan::PhysicalPlan;

const BLOCK: u32 = 256;
const TAKE_B8: &str = "bolt_take_b8";
const TAKE_B32: &str = "bolt_take_b32";
const TAKE_B64: &str = "bolt_take_b64";
const WIDEN: &str = "bolt_widen_i32_i64";
const ANY_EQ: &str = "bolt_any_eq_i64";

/// `dst[i] = src[idx[i]]` for 1-, 4- and 8-byte elements; `i32 → i64`
/// widening; "does any `src[i]` equal `v`".
fn pipeline_ptx() -> String {
    let take = |name: &str, bytes: u32, ty: &str, reg: &str| {
        format!(
            r#"
.visible .entry {name}(.param .u64 p_src, .param .u64 p_idx, .param .u64 p_dst, .param .u32 p_n)
{{
	.reg .pred %p<2>;
	.reg .b32 %r<8>;
	.reg .b64 %rd<10>;
	mov.u32 %r0, %ctaid.x;
	mov.u32 %r1, %ntid.x;
	mov.u32 %r2, %tid.x;
	mad.lo.u32 %r3, %r0, %r1, %r2;
	ld.param.u32 %r4, [p_n];
	setp.ge.u32 %p0, %r3, %r4;
	@%p0 ret;
	ld.param.u64 %rd0, [p_src];
	ld.param.u64 %rd1, [p_idx];
	ld.param.u64 %rd2, [p_dst];
	cvta.to.global.u64 %rd0, %rd0;
	cvta.to.global.u64 %rd1, %rd1;
	cvta.to.global.u64 %rd2, %rd2;
	mul.wide.u32 %rd3, %r3, 4;
	add.u64 %rd4, %rd1, %rd3;
	ld.global.u32 %r5, [%rd4];
	mul.wide.u32 %rd5, %r5, {bytes};
	add.u64 %rd6, %rd0, %rd5;
	ld.global.{ty} {reg}, [%rd6];
	mul.wide.u32 %rd7, %r3, {bytes};
	add.u64 %rd8, %rd2, %rd7;
	st.global.{ty} [%rd8], {reg};
	ret;
}}
"#
        )
    };
    let mut p = String::from(".version 7.5\n.target sm_70\n.address_size 64\n");
    p.push_str(&take(TAKE_B8, 1, "u8", "%r6"));
    p.push_str(&take(TAKE_B32, 4, "b32", "%r6"));
    p.push_str(&take(TAKE_B64, 8, "b64", "%rd9"));
    p.push_str(&format!(
        r#"
.visible .entry {WIDEN}(.param .u64 p_src, .param .u64 p_dst, .param .u32 p_n)
{{
	.reg .pred %p<2>;
	.reg .b32 %r<8>;
	.reg .b64 %rd<8>;
	mov.u32 %r0, %ctaid.x;
	mov.u32 %r1, %ntid.x;
	mov.u32 %r2, %tid.x;
	mad.lo.u32 %r3, %r0, %r1, %r2;
	ld.param.u32 %r4, [p_n];
	setp.ge.u32 %p0, %r3, %r4;
	@%p0 ret;
	ld.param.u64 %rd0, [p_src];
	ld.param.u64 %rd1, [p_dst];
	cvta.to.global.u64 %rd0, %rd0;
	cvta.to.global.u64 %rd1, %rd1;
	mul.wide.u32 %rd2, %r3, 4;
	add.u64 %rd3, %rd0, %rd2;
	ld.global.s32 %r5, [%rd3];
	cvt.s64.s32 %rd4, %r5;
	mul.wide.u32 %rd5, %r3, 8;
	add.u64 %rd6, %rd1, %rd5;
	st.global.s64 [%rd6], %rd4;
	ret;
}}

.visible .entry {ANY_EQ}(.param .u64 p_src, .param .u32 p_n, .param .s64 p_v, .param .u64 p_flag)
{{
	.reg .pred %p<3>;
	.reg .b32 %r<8>;
	.reg .b64 %rd<8>;
	mov.u32 %r0, %ctaid.x;
	mov.u32 %r1, %ntid.x;
	mov.u32 %r2, %tid.x;
	mad.lo.u32 %r3, %r0, %r1, %r2;
	ld.param.u32 %r4, [p_n];
	setp.ge.u32 %p0, %r3, %r4;
	@%p0 ret;
	ld.param.u64 %rd0, [p_src];
	cvta.to.global.u64 %rd0, %rd0;
	mul.wide.u32 %rd1, %r3, 8;
	add.u64 %rd2, %rd0, %rd1;
	ld.global.s64 %rd3, [%rd2];
	ld.param.s64 %rd4, [p_v];
	setp.ne.s64 %p1, %rd3, %rd4;
	@%p1 ret;
	ld.param.u64 %rd5, [p_flag];
	cvta.to.global.u64 %rd5, %rd5;
	mov.u32 %r5, 1;
	st.global.u32 [%rd5], %r5;
	ret;
}}
"#
    ));
    p
}

/// Launch `entry` over `n` threads with raw kernel parameters.
fn launch(
    entry: &str,
    n: usize,
    params: &mut [*mut c_void],
    stream: &CudaStream,
) -> BoltResult<()> {
    if n == 0 {
        return Ok(());
    }
    let module = crate::exec::module_cache::get_or_build_module(
        "device_pipeline",
        "v1".to_string(),
        None,
        || Ok(pipeline_ptx()),
    )?;
    let function = module.function(entry)?;
    let n_u32 = u32::try_from(n)
        .map_err(|_| BoltError::GpuCapacity(format!("device pipeline: {n} rows exceed u32")))?;
    cuda_sys::ensure_ctx_current()?;
    // SAFETY: every pointer parameter names a live device allocation the
    // caller keeps alive until `stream` is synchronised, sized for `n`
    // elements (sources: for every index the index buffer can hold).
    unsafe {
        cuda_sys::check(cuda_sys::cuLaunchKernel(
            function.raw(),
            n_u32.div_ceil(BLOCK).max(1),
            1,
            1,
            BLOCK,
            1,
            1,
            0,
            stream.raw(),
            params.as_mut_ptr(),
            ptr::null_mut(),
        ))?;
    }
    crate::metrics::metrics().inc(crate::metrics::Counter::GpuLaunchesTotal);
    Ok(())
}

/// `out[i] = src[idx[i]]` for `i < n` (`T` of 1, 4 or 8 bytes).
fn take<T: bytemuck::Pod>(
    src: &GpuVec<T>,
    idx: &GpuVec<u32>,
    n: usize,
    stream: &CudaStream,
) -> BoltResult<GpuVec<T>> {
    let out = GpuVec::<T>::zeros(n)?;
    let entry = match std::mem::size_of::<T>() {
        1 => TAKE_B8,
        4 => TAKE_B32,
        8 => TAKE_B64,
        w => return Err(BoltError::Other(format!("device take: {w}-byte elements"))),
    };
    let (mut s, mut i, mut d, mut n32): (CUdeviceptr, CUdeviceptr, CUdeviceptr, u32) = (
        src.device_ptr(),
        idx.device_ptr(),
        out.device_ptr(),
        n as u32,
    );
    let mut params: [*mut c_void; 4] = [
        &mut s as *mut CUdeviceptr as *mut c_void,
        &mut i as *mut CUdeviceptr as *mut c_void,
        &mut d as *mut CUdeviceptr as *mut c_void,
        &mut n32 as *mut u32 as *mut c_void,
    ];
    launch(entry, n, &mut params, stream)?;
    src.mark_stream_use(stream.raw());
    idx.mark_stream_use(stream.raw());
    out.mark_stream_use(stream.raw());
    Ok(out)
}

/// A column gathered by `idx` (the first `n` entries), or `None` for
/// layouts the gather does not move (Utf8, decimals).
fn take_column(
    data: &GpuColumnData,
    idx: &GpuVec<u32>,
    n: usize,
    stream: &CudaStream,
) -> BoltResult<Option<GpuColumnData>> {
    let validity = |v: &Option<GpuVec<u8>>| -> BoltResult<Option<GpuVec<u8>>> {
        v.as_ref().map(|v| take(v, idx, n, stream)).transpose()
    };
    Ok(Some(match data {
        GpuColumnData::I32 {
            values,
            validity: v,
        } => GpuColumnData::I32 {
            values: take(values, idx, n, stream)?,
            validity: validity(v)?,
        },
        GpuColumnData::I64 {
            values,
            validity: v,
        } => GpuColumnData::I64 {
            values: take(values, idx, n, stream)?,
            validity: validity(v)?,
        },
        GpuColumnData::F32 {
            values,
            validity: v,
        } => GpuColumnData::F32 {
            values: take(values, idx, n, stream)?,
            validity: validity(v)?,
        },
        GpuColumnData::F64 {
            values,
            validity: v,
        } => GpuColumnData::F64 {
            values: take(values, idx, n, stream)?,
            validity: validity(v)?,
        },
        GpuColumnData::Bool(values) => GpuColumnData::Bool(take(values, idx, n, stream)?),
        GpuColumnData::BoolNullable { values, validity } => GpuColumnData::BoolNullable {
            values: take(values, idx, n, stream)?,
            validity: take(validity, idx, n, stream)?,
        },
        _ => return Ok(None),
    }))
}

/// Join keys as one `i64` per row: the column itself, or a widened copy.
enum Keys<'a> {
    Borrowed(&'a GpuVec<i64>),
    Owned(GpuVec<i64>),
}

impl Keys<'_> {
    fn get(&self) -> &GpuVec<i64> {
        match self {
            Keys::Borrowed(k) => k,
            Keys::Owned(k) => k,
        }
    }
}

/// A device copy of a column (a scan's input stays in the resident table),
/// or `None` for layouts device tables do not carry.
fn copy_column(data: &GpuColumnData, n: usize) -> BoltResult<Option<GpuColumnData>> {
    fn copy<T: bytemuck::Pod>(v: &GpuVec<T>, n: usize) -> BoltResult<GpuVec<T>> {
        let out = GpuVec::<T>::zeros(n)?;
        // SAFETY: both are live device allocations of at least `n` elements
        // (`v` holds the table's `n` rows) and do not overlap.
        unsafe { cuda_sys::memcpy_d2d::<T>(out.device_ptr(), v.device_ptr(), n)? };
        Ok(out)
    }
    let validity = |v: &Option<GpuVec<u8>>| v.as_ref().map(|v| copy(v, n)).transpose();
    Ok(Some(match data {
        GpuColumnData::I32 {
            values,
            validity: v,
        } => GpuColumnData::I32 {
            values: copy(values, n)?,
            validity: validity(v)?,
        },
        GpuColumnData::I64 {
            values,
            validity: v,
        } => GpuColumnData::I64 {
            values: copy(values, n)?,
            validity: validity(v)?,
        },
        GpuColumnData::F32 {
            values,
            validity: v,
        } => GpuColumnData::F32 {
            values: copy(values, n)?,
            validity: validity(v)?,
        },
        GpuColumnData::F64 {
            values,
            validity: v,
        } => GpuColumnData::F64 {
            values: copy(values, n)?,
            validity: validity(v)?,
        },
        GpuColumnData::Bool(values) => GpuColumnData::Bool(copy(values, n)?),
        GpuColumnData::BoolNullable { values, validity } => GpuColumnData::BoolNullable {
            values: copy(values, n)?,
            validity: copy(validity, n)?,
        },
        _ => return Ok(None),
    }))
}

/// The join key of a device column as one `i64` per row, or `None` when it
/// cannot be one (not an integer, or nullable).
fn key_i64<'a>(
    data: &'a GpuColumnData,
    n: usize,
    stream: &CudaStream,
) -> BoltResult<Option<Keys<'a>>> {
    match data {
        GpuColumnData::I64 {
            values,
            validity: None,
        } => Ok(Some(Keys::Borrowed(values))),
        GpuColumnData::I32 {
            values,
            validity: None,
        } => {
            let out = GpuVec::<i64>::zeros(n)?;
            let (mut s, mut d, mut n32) = (values.device_ptr(), out.device_ptr(), n as u32);
            let mut params: [*mut c_void; 3] = [
                &mut s as *mut CUdeviceptr as *mut c_void,
                &mut d as *mut CUdeviceptr as *mut c_void,
                &mut n32 as *mut u32 as *mut c_void,
            ];
            launch(WIDEN, n, &mut params, stream)?;
            values.mark_stream_use(stream.raw());
            out.mark_stream_use(stream.raw());
            Ok(Some(Keys::Owned(out)))
        }
        _ => Ok(None),
    }
}

/// Whether any of the first `n` keys equals `v`.
fn any_equal(keys: &GpuVec<i64>, n: usize, v: i64, stream: &CudaStream) -> BoltResult<bool> {
    let flag = GpuVec::<u32>::zeros(1)?;
    let (mut s, mut n32, mut val, mut f) = (keys.device_ptr(), n as u32, v, flag.device_ptr());
    let mut params: [*mut c_void; 4] = [
        &mut s as *mut CUdeviceptr as *mut c_void,
        &mut n32 as *mut u32 as *mut c_void,
        &mut val as *mut i64 as *mut c_void,
        &mut f as *mut CUdeviceptr as *mut c_void,
    ];
    launch(ANY_EQ, n, &mut params, stream)?;
    stream.synchronize()?;
    Ok(flag.to_vec()?.first().copied().unwrap_or(0) != 0)
}

/// Host copy of a device table, with the plan types of `schema` (the
/// table's columns in order).
pub(crate) fn download_table(gt: &GpuTable, schema: &Schema) -> BoltResult<RecordBatch> {
    fn nulls<T: Copy>(values: Vec<T>, validity: Option<Vec<u8>>) -> Vec<Option<T>> {
        match validity {
            None => values.into_iter().map(Some).collect(),
            Some(v) => values
                .into_iter()
                .zip(v)
                .map(|(x, ok)| (ok != 0).then_some(x))
                .collect(),
        }
    }
    let valid = |v: &Option<GpuVec<u8>>| v.as_ref().map(GpuVec::to_vec).transpose();
    let mut fields = Vec::with_capacity(gt.columns.len());
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(gt.columns.len());
    for (c, f) in gt.columns.iter().zip(&schema.fields) {
        let n = gt.n_rows;
        let trim = |mut v: Vec<u8>| {
            v.truncate(n);
            v
        };
        let array: ArrayRef = match &c.data {
            GpuColumnData::I32 { values, validity } => {
                let mut v = values.to_vec()?;
                v.truncate(n);
                Arc::new(Int32Array::from(nulls(v, valid(validity)?.map(trim))))
            }
            GpuColumnData::I64 { values, validity } => {
                let mut v = values.to_vec()?;
                v.truncate(n);
                Arc::new(Int64Array::from(nulls(v, valid(validity)?.map(trim))))
            }
            GpuColumnData::F32 { values, validity } => {
                let mut v = values.to_vec()?;
                v.truncate(n);
                Arc::new(Float32Array::from(nulls(v, valid(validity)?.map(trim))))
            }
            GpuColumnData::F64 { values, validity } => {
                let mut v = values.to_vec()?;
                v.truncate(n);
                Arc::new(Float64Array::from(nulls(v, valid(validity)?.map(trim))))
            }
            GpuColumnData::Bool(values) => {
                let v = trim(values.to_vec()?);
                Arc::new(BooleanArray::from(
                    v.into_iter().map(|b| b != 0).collect::<Vec<_>>(),
                ))
            }
            GpuColumnData::BoolNullable { values, validity } => {
                let v = trim(values.to_vec()?);
                let ok = trim(validity.to_vec()?);
                Arc::new(BooleanArray::from(
                    v.into_iter()
                        .zip(ok)
                        .map(|(b, ok)| (ok != 0).then_some(b != 0))
                        .collect::<Vec<_>>(),
                ))
            }
            _ => {
                return Err(BoltError::Other(format!(
                    "device pipeline: cannot download column '{}'",
                    c.name
                )))
            }
        };
        let target = crate::exec::schema_convert::plan_dtype_to_arrow(f.dtype)?;
        let array = if array.data_type() == &target {
            array
        } else {
            arrow::compute::cast(&array, &target)
                .map_err(|e| BoltError::Other(format!("device pipeline: {e}")))?
        };
        fields.push(ArrowField::new(
            f.name.clone(),
            array.data_type().clone(),
            true,
        ));
        columns.push(array);
    }
    RecordBatch::try_new(Arc::new(ArrowSchema::new(fields)), columns)
        .map_err(|e| BoltError::Other(format!("device pipeline: {e}")))
}

/// Whether `data` has (or may have) NULLs.
pub(crate) fn column_has_validity(data: &GpuColumnData) -> bool {
    matches!(
        data,
        GpuColumnData::I32 {
            validity: Some(_),
            ..
        } | GpuColumnData::I64 {
            validity: Some(_),
            ..
        } | GpuColumnData::F32 {
            validity: Some(_),
            ..
        } | GpuColumnData::F64 {
            validity: Some(_),
            ..
        } | GpuColumnData::BoolNullable { .. }
    )
}

/// A compacted projection output as device column storage.
pub(crate) fn gathered_to_column(
    c: crate::exec::gpu_compact::GatheredCol,
) -> BoltResult<GpuColumnData> {
    use crate::exec::gpu_compact::GatheredCol as G;
    Ok(match c {
        G::I32(values) | G::Date32(values) => GpuColumnData::I32 {
            values,
            validity: None,
        },
        G::I64(values) | G::Timestamp { values, .. } => GpuColumnData::I64 {
            values,
            validity: None,
        },
        G::F32(values) => GpuColumnData::F32 {
            values,
            validity: None,
        },
        G::F64(values) => GpuColumnData::F64 {
            values,
            validity: None,
        },
        G::Bool(values) => GpuColumnData::Bool(values),
        G::BoolNullable { values, validity } => GpuColumnData::BoolNullable { values, validity },
        G::Decimal128 { .. } => {
            return Err(BoltError::Other(
                "device pipeline: decimal outputs stay on the host path".into(),
            ))
        }
    })
}

/// An unfiltered projection output as device column storage.
pub(crate) fn device_col_to_column(
    c: crate::exec::engine_device_col::DeviceCol,
) -> BoltResult<GpuColumnData> {
    use crate::exec::engine_device_col::DeviceCol as D;
    Ok(match c {
        D::I32(values) => GpuColumnData::I32 {
            values,
            validity: None,
        },
        D::I64(values) => GpuColumnData::I64 {
            values,
            validity: None,
        },
        D::F32(values) => GpuColumnData::F32 {
            values,
            validity: None,
        },
        D::F64(values) => GpuColumnData::F64 {
            values,
            validity: None,
        },
        D::Bool(values) => GpuColumnData::Bool(values),
        _ => {
            return Err(BoltError::Other(
                "device pipeline: string / decimal outputs stay on the host path".into(),
            ))
        }
    })
}

/// A projection kernel's outputs as a device table named after
/// `output_schema`: hidden `__bolt_valid_<j>` outputs (see
/// `plan::null_semantics`) become output `j`'s validity.
pub(crate) fn device_table_from_outputs(
    mut columns: Vec<GpuColumnData>,
    outputs: &[crate::plan::ColumnIO],
    output_schema: &Schema,
    n_rows: usize,
) -> BoltResult<GpuTable> {
    let prefix = crate::plan::null_semantics::HIDDEN_VALIDITY_PREFIX;
    let hidden: Vec<(usize, usize)> = outputs
        .iter()
        .enumerate()
        .filter_map(|(k, io)| {
            io.name
                .strip_prefix(prefix)
                .and_then(|j| j.parse().ok())
                .map(|j| (k, j))
        })
        .collect();
    if let Some(&(first, _)) = hidden.first() {
        let mut valid: Vec<(usize, GpuVec<u8>)> = Vec::new();
        for &(k, j) in hidden.iter().rev() {
            let v = match columns.remove(k) {
                GpuColumnData::Bool(v) => v,
                _ => {
                    return Err(BoltError::Other(
                        "hidden validity output is not Bool".into(),
                    ))
                }
            };
            valid.push((j, v));
        }
        columns.truncate(first);
        for (j, v) in valid {
            let target = columns
                .get_mut(j)
                .ok_or_else(|| BoltError::Other("hidden validity output without target".into()))?;
            let placeholder = GpuColumnData::Bool(GpuVec::<u8>::zeros(0)?);
            *target = match std::mem::replace(target, placeholder) {
                GpuColumnData::I32 { values, .. } => GpuColumnData::I32 {
                    values,
                    validity: Some(v),
                },
                GpuColumnData::I64 { values, .. } => GpuColumnData::I64 {
                    values,
                    validity: Some(v),
                },
                GpuColumnData::F32 { values, .. } => GpuColumnData::F32 {
                    values,
                    validity: Some(v),
                },
                GpuColumnData::F64 { values, .. } => GpuColumnData::F64 {
                    values,
                    validity: Some(v),
                },
                GpuColumnData::Bool(values) => GpuColumnData::BoolNullable {
                    values,
                    validity: v,
                },
                other => other,
            };
        }
    }
    if columns.len() != output_schema.fields.len() {
        return Err(BoltError::Other(format!(
            "device pipeline: {} outputs for {} columns",
            columns.len(),
            output_schema.fields.len()
        )));
    }
    Ok(GpuTable {
        n_rows,
        columns: columns
            .into_iter()
            .zip(&output_schema.fields)
            .map(|(data, f)| column(&f.name, f.dtype, data))
            .collect(),
        last_uploaded_revision: 0,
    })
}

fn column(name: &str, dtype: DataType, data: GpuColumnData) -> GpuColumn {
    GpuColumn {
        name: name.to_string(),
        dtype,
        data,
        host_revision: 0,
    }
}

/// Answers for a device-bound overlay table from its device columns, and
/// for everything else from `inner`.
struct DeviceInputProvider<'a> {
    inner: &'a dyn crate::plan::TableProvider,
    table: &'a str,
    schema: &'a Schema,
    nulls: Vec<bool>,
}

impl crate::plan::TableProvider for DeviceInputProvider<'_> {
    fn schema(&self, name: &str) -> BoltResult<Schema> {
        if name == self.table {
            return Ok(self.schema.clone());
        }
        self.inner.schema(name)
    }

    fn has_nulls(&self, table_name: &str, col_idx: usize) -> bool {
        if table_name == self.table {
            return self.nulls.get(col_idx).copied().unwrap_or(false);
        }
        self.inner.has_nulls(table_name, col_idx)
    }
}

/// `BOLT_DEVICE_PIPELINE=0` turns device-resident pipelines off (every
/// operator then returns a host batch, as before).
fn enabled() -> bool {
    std::env::var("BOLT_DEVICE_PIPELINE").map_or(true, |v| v != "0")
}

/// Whether `plan` is made only of operators [`Engine::execute_device`] can
/// keep on the device (checked before running anything).
pub(crate) fn device_eligible(plan: &PhysicalPlan) -> bool {
    enabled() && eligible(plan)
}

fn eligible(plan: &PhysicalPlan) -> bool {
    match plan {
        PhysicalPlan::StringProject {
            outputs,
            output_schema,
            ..
        } => {
            outputs.iter().all(|o| {
                matches!(
                    o,
                    crate::plan::physical_plan::StringProjectOutput::Passthrough { .. }
                )
            }) && output_schema
                .fields
                .iter()
                .all(|f| !matches!(f.dtype, DataType::Utf8 | DataType::Decimal128(_, _)))
        }
        PhysicalPlan::Projection { kernel, .. } => !kernel
            .outputs
            .iter()
            .any(|io| matches!(io.dtype, DataType::Utf8 | DataType::Decimal128(_, _))),
        PhysicalPlan::Join {
            left,
            right,
            join_type: JoinType::Inner,
            on,
            filter: None,
            ..
        } => on.len() == 1 && eligible(left) && eligible(right),
        PhysicalPlan::Project { input, exprs, .. } => {
            exprs.iter().all(|e| {
                let mut e = e;
                while let Expr::Alias(inner, _) = e {
                    e = inner;
                }
                matches!(e, Expr::Column(_))
            }) && eligible(input)
        }
        _ => false,
    }
}

/// Whether `plan` contains a join [`Engine::execute_device`] may run.
pub(crate) fn has_device_join(plan: &PhysicalPlan) -> bool {
    match plan {
        PhysicalPlan::Join { .. } => true,
        PhysicalPlan::Project { input, .. } => has_device_join(input),
        _ => false,
    }
}

impl Engine {
    /// Run `aggregate` over the device table `input`, bound as the overlay
    /// table `table` (columns named after `input_schema`): its kernels read
    /// the device columns in place. A host fallback inside the aggregate
    /// downloads a copy on demand.
    pub(crate) fn aggregate_over_device(
        &self,
        mut input: GpuTable,
        table: &str,
        input_schema: &Schema,
        aggregate: &PhysicalPlan,
    ) -> BoltResult<crate::exec::engine::QueryHandle> {
        if input.columns.len() != input_schema.fields.len() {
            return Err(BoltError::Other(format!(
                "aggregate input produced {} columns, expected {}",
                input.columns.len(),
                input_schema.fields.len()
            )));
        }
        for (c, f) in input.columns.iter_mut().zip(&input_schema.fields) {
            c.name = f.name.clone();
        }
        if self.has_table(table) {
            return Err(BoltError::Plan(format!(
                "reserved aggregate-input table name '{table}' collides with a registered table"
            )));
        }
        let nulls: Vec<bool> = input
            .columns
            .iter()
            .map(|c| column_has_validity(&c.data))
            .collect();
        self.gpu_tables
            .borrow_mut()
            .insert(table.to_string(), Some(input));
        self.device_overlays
            .borrow_mut()
            .insert(table.to_string(), input_schema.clone());
        // Registered for name resolution; its rows come from the device copy
        // (`materialize_streaming_source` downloads them before any drain).
        let never: crate::exec::streaming::BatchProducer = Box::new(|| {
            Box::new(std::iter::once(Err(BoltError::Other(
                "device-resident aggregate input has no host producer".into(),
            ))))
        });
        self.streaming_sources.borrow_mut().insert(
            table.to_string(),
            crate::exec::streaming::TableSource::Streaming(never),
        );
        let mut aggregate = aggregate.clone();
        let base = crate::exec::engine_provider::EngineProvider {
            base: &self.provider,
            tables: &self.tables,
            streaming: self.streaming_sources.borrow(),
        };
        let provider = DeviceInputProvider {
            inner: &base,
            table,
            schema: input_schema,
            nulls,
        };
        crate::plan::physical_plan::populate_input_validity(&mut aggregate, &provider);
        drop(provider);
        drop(base);
        let result = self.execute(&aggregate);
        self.streaming_sources.borrow_mut().remove(table);
        self.gpu_tables.borrow_mut().remove(table);
        self.device_overlays.borrow_mut().remove(table);
        self.overlay_uploads.borrow_mut().remove(table);
        result
    }

    /// The result of `plan` as a device table, when the whole subplan can
    /// stay on the device (see the [module docs](self)); `None` otherwise.
    pub(crate) fn execute_device(&self, plan: &PhysicalPlan) -> BoltResult<Option<GpuTable>> {
        match plan {
            PhysicalPlan::Projection {
                table,
                kernel,
                output_schema,
            } => self.execute_projection_device(table, kernel, output_schema),
            PhysicalPlan::StringProject {
                table,
                outputs,
                output_schema,
            } => {
                // A passthrough scan: columns of the resident table.
                let gt = match self.ensure_gpu_table(table) {
                    Ok(gt) => gt,
                    Err(BoltError::GpuCapacity(_)) => return Ok(None),
                    Err(e) => return Err(e),
                };
                let mut columns = Vec::with_capacity(outputs.len());
                for (o, f) in outputs.iter().zip(&output_schema.fields) {
                    let crate::plan::physical_plan::StringProjectOutput::Passthrough { source } = o
                    else {
                        return Ok(None);
                    };
                    let Some(src) = gt.column(source) else {
                        return Ok(None);
                    };
                    let Some(data) = copy_column(&src.data, gt.n_rows)? else {
                        return Ok(None);
                    };
                    columns.push(column(&f.name, f.dtype, data));
                }
                Ok(Some(GpuTable {
                    n_rows: gt.n_rows,
                    columns,
                    last_uploaded_revision: 0,
                }))
            }
            PhysicalPlan::Join {
                left,
                right,
                join_type: JoinType::Inner,
                on,
                filter: None,
                output_schema,
            } if on.len() == 1 => {
                let Some(l) = self.execute_device(left)? else {
                    return Ok(None);
                };
                let Some(r) = self.execute_device(right)? else {
                    return Ok(None);
                };
                self.device_inner_join(l, r, &on[0], output_schema)
            }
            PhysicalPlan::Project {
                input,
                exprs,
                output_schema,
            } => {
                let mut names = Vec::with_capacity(exprs.len());
                for e in exprs {
                    let mut e = e;
                    while let Expr::Alias(inner, _) = e {
                        e = inner;
                    }
                    match e {
                        Expr::Column(c) => names.push(c.clone()),
                        _ => return Ok(None),
                    }
                }
                let mut seen = std::collections::HashSet::new();
                if !names.iter().all(|n| seen.insert(n.clone())) {
                    return Ok(None);
                }
                let Some(mut t) = self.execute_device(input)? else {
                    return Ok(None);
                };
                let mut columns = Vec::with_capacity(names.len());
                for (n, f) in names.iter().zip(&output_schema.fields) {
                    let Some(i) = t.columns.iter().position(|c| &c.name == n) else {
                        return Ok(None);
                    };
                    let mut c = t.columns.swap_remove(i);
                    c.name = f.name.clone();
                    columns.push(c);
                }
                Ok(Some(GpuTable {
                    n_rows: t.n_rows,
                    columns,
                    last_uploaded_revision: 0,
                }))
            }
            _ => Ok(None),
        }
    }

    /// Inner equi-join of two device tables on one integer key: device
    /// build / probe, device gather of every output column. `None` when a
    /// key is nullable or not an integer, an input is too small to be worth
    /// a GPU join, a key holds the hash table's sentinel, or a column cannot
    /// be gathered.
    fn device_inner_join(
        &self,
        l: GpuTable,
        r: GpuTable,
        on: &(Expr, Expr),
        output_schema: &Schema,
    ) -> BoltResult<Option<GpuTable>> {
        let key = |t: &GpuTable, e: &Expr| -> Option<usize> {
            match e {
                Expr::Column(c) => t.columns.iter().position(|col| &col.name == c),
                _ => None,
            }
        };
        let (Some(lk), Some(rk)) = (key(&l, &on.0), key(&r, &on.1)) else {
            return Ok(None);
        };
        let min = crate::exec::gpu_join::GPU_JOIN_MIN_ROWS;
        if l.n_rows < min || r.n_rows < min {
            return Ok(None);
        }
        if output_schema.fields.len() != l.columns.len() + r.columns.len() {
            return Ok(None);
        }
        let stream = CudaStream::null_or_default();
        let (Some(lkeys), Some(rkeys)) = (
            key_i64(&l.columns[lk].data, l.n_rows, &stream)?,
            key_i64(&r.columns[rk].data, r.n_rows, &stream)?,
        ) else {
            return Ok(None);
        };
        let (lkeys, rkeys) = (lkeys.get(), rkeys.get());
        if any_equal(lkeys, l.n_rows, i64::MIN, &stream)?
            || any_equal(rkeys, r.n_rows, i64::MIN, &stream)?
        {
            return Ok(None);
        }
        // Build on the smaller side.
        let build_is_left = l.n_rows <= r.n_rows;
        let ((bk, bn), (pk, pn)) = if build_is_left {
            ((lkeys, l.n_rows), (rkeys, r.n_rows))
        } else {
            ((rkeys, r.n_rows), (lkeys, l.n_rows))
        };
        let pairs =
            match crate::exec::gpu_join::hash_join_indices_on_gpu_resident_keys(bk, bn, pk, pn) {
                Ok(p) => p,
                Err(BoltError::GpuCapacity(_)) => return Ok(None),
                Err(e) => return Err(e),
            };
        let n = pairs.n_matches;
        let (lidx, ridx) = if build_is_left {
            (&pairs.build_idx, &pairs.probe_idx)
        } else {
            (&pairs.probe_idx, &pairs.build_idx)
        };
        let mut columns = Vec::with_capacity(output_schema.fields.len());
        let sides = l
            .columns
            .iter()
            .map(|c| (c, lidx))
            .chain(r.columns.iter().map(|c| (c, ridx)));
        for ((c, idx), f) in sides.zip(&output_schema.fields) {
            let Some(data) = take_column(&c.data, idx, n, &stream)? else {
                return Ok(None);
            };
            columns.push(column(&f.name, c.dtype, data));
        }
        stream.synchronize()?;
        crate::metrics::metrics().inc(crate::metrics::Counter::DevicePipelines);
        Ok(Some(GpuTable {
            n_rows: n,
            columns,
            last_uploaded_revision: 0,
        }))
    }
}
