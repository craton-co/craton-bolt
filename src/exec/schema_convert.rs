// SPDX-License-Identifier: Apache-2.0
//! Single-source-of-truth conversions between the plan-layer `DataType` /
//! `Schema` and Arrow's `DataType` / `Schema`.
//!
//! ## Why this module exists
//!
//! Prior to the v0.7 consolidation these converters (`plan_dtype_to_arrow`,
//! `arrow_dtype_to_plan`, `plan_schema_to_arrow_schema`, and the `TimeUnit`
//! mappers) were copy-pasted into ~25 executor files. The copies had already
//! drifted: the engine path mapped `Date32`/`Timestamp` through to Arrow,
//! while the GROUP BY and join output paths *rejected* them on purpose (a
//! "loud regression" guard, since temporal types are not yet wired through
//! those kernels). A fix to the match logic therefore had to be hand-applied
//! to every copy, and the GB-S1-style "guard present in some files, missing
//! in others" divergence was a constant risk.
//!
//! This module keeps the *logic* in one place. The two behaviours are exposed
//! as separate functions:
//!
//! * The plain converters (`plan_dtype_to_arrow`, `arrow_dtype_to_plan`,
//!   `plan_schema_to_arrow_schema`) are the **full** mappers — they map every
//!   supported plan dtype, including `Date32`/`Timestamp`, through to Arrow.
//!   These match the historical engine-path behaviour.
//! * The `*_no_temporal` / `*_basic` variants **reject** temporal (and, for
//!   `arrow_dtype_to_plan_basic`, dictionary) types with a caller-supplied
//!   context string, reproducing the historical GROUP BY / join guards
//!   verbatim.
//!
//! Each executor keeps a one-line wrapper around the variant it needs so the
//! call sites and error messages are byte-for-byte unchanged.

use std::sync::Arc;

use arrow_schema::{
    DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema,
    TimeUnit as ArrowTimeUnit,
};

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{DataType, Schema, TimeUnit};

/// Map our plan `TimeUnit` to Arrow's `TimeUnit`.
pub(crate) fn plan_time_unit_to_arrow(u: TimeUnit) -> ArrowTimeUnit {
    match u {
        TimeUnit::Second => ArrowTimeUnit::Second,
        TimeUnit::Millisecond => ArrowTimeUnit::Millisecond,
        TimeUnit::Microsecond => ArrowTimeUnit::Microsecond,
        TimeUnit::Nanosecond => ArrowTimeUnit::Nanosecond,
    }
}

/// Map Arrow's `TimeUnit` to our plan `TimeUnit`.
pub(crate) fn arrow_time_unit_to_plan(u: &ArrowTimeUnit) -> TimeUnit {
    match u {
        ArrowTimeUnit::Second => TimeUnit::Second,
        ArrowTimeUnit::Millisecond => TimeUnit::Millisecond,
        ArrowTimeUnit::Microsecond => TimeUnit::Microsecond,
        ArrowTimeUnit::Nanosecond => TimeUnit::Nanosecond,
    }
}

/// Full plan→Arrow dtype mapping, including `Date32`/`Timestamp`.
///
/// This is the historical engine-path behaviour. Paths that intentionally
/// reject temporal types should call [`plan_dtype_to_arrow_no_temporal`].
pub(crate) fn plan_dtype_to_arrow(d: DataType) -> BoltResult<ArrowDataType> {
    match d {
        DataType::Int32 => Ok(ArrowDataType::Int32),
        DataType::Int64 => Ok(ArrowDataType::Int64),
        DataType::Float32 => Ok(ArrowDataType::Float32),
        DataType::Float64 => Ok(ArrowDataType::Float64),
        DataType::Bool => Ok(ArrowDataType::Boolean),
        DataType::Utf8 => Ok(ArrowDataType::Utf8),
        DataType::Decimal128(p, s) => Ok(ArrowDataType::Decimal128(p, s)),
        // v0.6 / M4: Date32 maps to Arrow `Date32`; Timestamp carries unit + tz.
        DataType::Date32 => Ok(ArrowDataType::Date32),
        DataType::Timestamp(unit, tz) => Ok(ArrowDataType::Timestamp(
            plan_time_unit_to_arrow(unit),
            tz.map(Arc::from),
        )),
    }
}

/// Plan→Arrow dtype mapping that rejects `Date32`/`Timestamp`.
///
/// Used by the GROUP BY and join output-schema builders, which do not yet
/// wire temporal types through their kernels and want a loud failure if one
/// slips through. `ctx` is interpolated into the error message so each call
/// site reports the same wording it did before consolidation (e.g.
/// `"this aggregate output path"`, `"join output path"`).
pub(crate) fn plan_dtype_to_arrow_no_temporal(d: DataType, ctx: &str) -> BoltResult<ArrowDataType> {
    match d {
        DataType::Date32 | DataType::Timestamp(_, _) => Err(BoltError::Type(format!(
            "Date/Timestamp not yet supported in {}: {:?}",
            ctx, d
        ))),
        other => plan_dtype_to_arrow(other),
    }
}

/// Full Arrow→plan dtype mapping, including `Date32`/`Timestamp` and
/// `Dictionary(_, Utf8)` (mapped to `Utf8`). Errors on unsupported types.
pub(crate) fn arrow_dtype_to_plan(d: &ArrowDataType) -> BoltResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        ArrowDataType::Decimal128(precision, scale) => Ok(DataType::Decimal128(*precision, *scale)),
        ArrowDataType::Date32 => Ok(DataType::Date32),
        ArrowDataType::Timestamp(unit, tz) => {
            let interned: Option<&'static str> = tz
                .as_deref()
                .map(crate::plan::logical_plan::intern_timezone);
            Ok(DataType::Timestamp(arrow_time_unit_to_plan(unit), interned))
        }
        ArrowDataType::Dictionary(_key, value) if matches!(value.as_ref(), ArrowDataType::Utf8) => {
            Ok(DataType::Utf8)
        }
        other => Err(BoltError::Type(format!(
            "unsupported Arrow dtype {:?}",
            other
        ))),
    }
}

/// Arrow→plan dtype mapping for paths that support the primitive +
/// `Decimal128` + `Utf8` set plus temporal (`Date32`/`Timestamp`) **input**
/// columns. No dictionary. `prefix` is prepended to the error message so each
/// call site keeps its original wording (e.g. `""` for most GROUP BY paths,
/// `"wide GROUP BY: "` for the wide path).
///
/// F7-finish: `Date32`/`Timestamp` are now accepted here because temporal
/// columns are valid on-device (F6 added GPU gather/upload for them) and the
/// aggregate / GROUP BY executors that call this use it only to validate that
/// the *input* column's Arrow dtype matches the plan dtype. The op-level
/// dispatch downstream (which still rejects e.g. `SUM(Timestamp)` and the
/// temporal arms of paths that have not been wired) is what actually gates
/// support — this converter must not reject a temporal input before that
/// dispatch is reached, or MIN/MAX/COUNT over temporal columns can never run.
pub(crate) fn arrow_dtype_to_plan_basic(d: &ArrowDataType, prefix: &str) -> BoltResult<DataType> {
    match d {
        ArrowDataType::Int32 => Ok(DataType::Int32),
        ArrowDataType::Int64 => Ok(DataType::Int64),
        ArrowDataType::Float32 => Ok(DataType::Float32),
        ArrowDataType::Float64 => Ok(DataType::Float64),
        ArrowDataType::Boolean => Ok(DataType::Bool),
        ArrowDataType::Utf8 => Ok(DataType::Utf8),
        ArrowDataType::Decimal128(precision, scale) => Ok(DataType::Decimal128(*precision, *scale)),
        // F7-finish: temporal inputs. `Date32` normalises to an i32 storage
        // dtype, `Timestamp(unit, tz)` to i64 — preserved verbatim so the
        // op dispatch can route MIN/MAX to the normalized reduction and
        // rebuild the correct temporal Arrow array.
        ArrowDataType::Date32 => Ok(DataType::Date32),
        ArrowDataType::Timestamp(unit, tz) => {
            let interned: Option<&'static str> = tz
                .as_deref()
                .map(crate::plan::logical_plan::intern_timezone);
            Ok(DataType::Timestamp(arrow_time_unit_to_plan(unit), interned))
        }
        other => Err(BoltError::Type(format!(
            "{}unsupported Arrow dtype {:?}",
            prefix, other
        ))),
    }
}

/// Build an Arrow schema from a plan `Schema`, mapping every field dtype with
/// the full [`plan_dtype_to_arrow`].
pub(crate) fn plan_schema_to_arrow_schema(s: &Schema) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

/// Build an Arrow schema from a plan `Schema`, rejecting temporal field dtypes
/// via [`plan_dtype_to_arrow_no_temporal`]. `ctx` is forwarded to the error
/// message.
pub(crate) fn plan_schema_to_arrow_schema_no_temporal(
    s: &Schema,
    ctx: &str,
) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow_no_temporal(f.dtype, ctx)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

/// Build an Arrow schema from an aggregate-output plan `Schema`, allowing
/// `Date32`/`Timestamp` output fields to round-trip through to their real
/// Arrow type (the full [`plan_dtype_to_arrow`], which rebuilds
/// `Date32` / `Timestamp(unit, tz)` with the carried unit + timezone).
///
/// F7-finish: this is the **MIN/MAX temporal** variant of
/// [`plan_schema_to_arrow_schema_no_temporal`]. The no-temporal variant is
/// shared by GROUP BY / join / shmem output builders that still reject
/// temporal results (those kernels are not wired for temporal output), so it
/// must keep its loud guard. The scalar-aggregate and global-atomic GROUP BY
/// paths, by contrast, now produce correct `MIN`/`MAX`/`COUNT` results over
/// temporal columns — they call this variant instead so a
/// `MIN(Timestamp(Microsecond, "UTC"))` output field builds a
/// `Timestamp(Microsecond, "UTC")` Arrow field rather than erroring.
///
/// Non-temporal fields map identically to the no-temporal path; only the
/// `Date32`/`Timestamp` rejection is lifted. `SUM(Timestamp)` etc. is still
/// rejected upstream at the executor's op dispatch, so a temporal field only
/// reaches here for the supported MIN/MAX/COUNT reductions.
pub(crate) fn plan_schema_to_arrow_schema_minmax_temporal(
    s: &Schema,
) -> BoltResult<Arc<ArrowSchema>> {
    let mut fields = Vec::with_capacity(s.fields.len());
    for f in &s.fields {
        let dt = plan_dtype_to_arrow(f.dtype)?;
        fields.push(ArrowField::new(&f.name, dt, f.nullable));
    }
    Ok(Arc::new(ArrowSchema::new(fields)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::logical_plan::Field;

    /// F7-finish: `arrow_dtype_to_plan_basic` now accepts a `Date32` input and
    /// maps it to the plan `Date32` (no longer the loud rejection).
    #[test]
    fn basic_accepts_date32_input() {
        let plan = arrow_dtype_to_plan_basic(&ArrowDataType::Date32, "").expect("date32 ok");
        assert_eq!(plan, DataType::Date32);
    }

    /// F7-finish: `arrow_dtype_to_plan_basic` accepts `Timestamp(unit, tz)` and
    /// preserves both the unit and the (interned) timezone on the round-trip
    /// back to Arrow.
    #[test]
    fn basic_accepts_timestamp_input_preserving_unit_and_tz() {
        let arrow_in = ArrowDataType::Timestamp(ArrowTimeUnit::Microsecond, Some(Arc::from("UTC")));
        let plan = arrow_dtype_to_plan_basic(&arrow_in, "").expect("timestamp ok");
        match plan {
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz)) => assert_eq!(tz, "UTC"),
            other => panic!("expected Timestamp(Microsecond, UTC), got {:?}", other),
        }
        // Round-trip back to Arrow must reproduce the exact dtype.
        let arrow_out = plan_dtype_to_arrow(plan).expect("plan->arrow ok");
        assert_eq!(arrow_out, arrow_in);
    }

    /// Non-temporal dtypes are unaffected by the relaxation.
    #[test]
    fn basic_still_rejects_dictionary() {
        let dict = ArrowDataType::Dictionary(
            Box::new(ArrowDataType::Int32),
            Box::new(ArrowDataType::Utf8),
        );
        assert!(arrow_dtype_to_plan_basic(&dict, "").is_err());
    }

    /// F7-finish: the MIN/MAX-temporal output schema builder rebuilds a
    /// `Date32` output field as Arrow `Date32` (where the no-temporal builder
    /// would reject it).
    #[test]
    fn minmax_temporal_builds_date32_output() {
        let schema = Schema::new(vec![Field::new("d", DataType::Date32, true)]);
        // The no-temporal builder must still reject (guard preserved).
        assert!(plan_schema_to_arrow_schema_no_temporal(&schema, "ctx").is_err());
        // The MIN/MAX-temporal builder accepts and produces Arrow Date32.
        let arrow = plan_schema_to_arrow_schema_minmax_temporal(&schema).expect("date32 schema");
        assert_eq!(arrow.field(0).data_type(), &ArrowDataType::Date32);
    }

    /// F7-finish: the MIN/MAX-temporal output schema builder preserves the
    /// Timestamp unit + timezone on the output field.
    #[test]
    fn minmax_temporal_builds_timestamp_output_with_unit_and_tz() {
        let tz = crate::plan::logical_plan::intern_timezone("UTC");
        let schema = Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz)),
            true,
        )]);
        assert!(plan_schema_to_arrow_schema_no_temporal(&schema, "ctx").is_err());
        let arrow = plan_schema_to_arrow_schema_minmax_temporal(&schema).expect("timestamp schema");
        assert_eq!(
            arrow.field(0).data_type(),
            &ArrowDataType::Timestamp(ArrowTimeUnit::Microsecond, Some(Arc::from("UTC"))),
        );
    }
}
