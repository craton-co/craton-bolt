// SPDX-License-Identifier: Apache-2.0

//! Substrait plan ingestion (feature `substrait`).
//!
//! Converts a [Substrait](https://substrait.io) plan (the cross-engine
//! protobuf relational-algebra IR) into the engine's own
//! [`LogicalPlan`](crate::plan::logical_plan::LogicalPlan). The whole module
//! is gated behind the `substrait` cargo feature so the default build never
//! pulls in the (large) generated prost types nor the `substrait` crate.
//!
//! # Layout
//!
//! * `mod.rs` (this file) — the public entry point
//!   ([`substrait_to_logical_plan`]), the scalar **type** and **literal**
//!   mapping ([`substrait_type_to_dtype`] / [`substrait_literal_to_literal`]),
//!   and the shared [`ConvertCtx`] threaded through the relation / expression
//!   converters.
//! * `rel.rs` — relation conversion: `Rel` → `LogicalPlan`
//!   (Read/Filter/Project/Aggregate/Sort/Fetch/Set/Join). Entry point
//!   `rel::convert_rel`.
//! * `expr.rs` — expression conversion: Substrait `Expression` →
//!   [`Expr`](crate::plan::logical_plan::Expr). Entry point
//!   `expr::convert_expr` (re-exported here as [`convert_expr`]).
//!
//! # Scope (v0 core)
//!
//! This is a *correct narrow slice*: the type / literal mapping below is the
//! complete, host-testable foundation; the relation / expression converters
//! (siblings) implement the common nodes and reject anything else with a
//! clear `BoltError::Plan("substrait: <node> not yet supported")` message.
//! Nothing here touches the GPU; the result is a plain logical plan that the
//! existing optimizer / physical planner consume unchanged.
#![cfg(feature = "substrait")]

mod expr;
mod rel;

use std::collections::HashMap;

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::LogicalPlan;
use crate::plan::logical_plan::{DataType, Literal, Schema, TimeUnit};
use crate::plan::sql_frontend::{MemTableProvider, TableProvider};

// Re-export the sibling converter entry points so callers (and tests) can
// reach the lower-level conversions without going through the full plan
// entry point. `pub(crate)` keeps them crate-internal — the only *public*
// surface of this module is [`substrait_to_logical_plan`].
pub(crate) use expr::convert_expr;
pub(crate) use rel::convert_rel;

use substrait::proto;

/// Convert a Substrait [`Plan`](proto::Plan) into the engine's
/// [`LogicalPlan`].
///
/// A Substrait `Plan` carries one or more `relations`; each top-level
/// relation is a [`PlanRel`](proto::PlanRel) that is either a bare `Rel` or a
/// [`RelRoot`](proto::RelRoot) (a `Rel` plus the output column *names*). This
/// entry point converts the **first** relation in the plan — the common case
/// for a single-statement query — and dispatches its root `Rel` to the
/// relation converter ([`rel::convert_rel`]).
///
/// `provider` resolves base-table names (from Substrait `ReadRel` named
/// tables) to their engine [`Schema`]. The same provider the SQL frontend
/// uses ([`MemTableProvider`]) is accepted here so a Substrait-sourced query
/// and a SQL-sourced query share one table catalog.
///
/// # Errors
///
/// * `BoltError::Plan("substrait: plan has no relations")` — empty plan.
/// * `BoltError::Plan("substrait: <node> not yet supported")` — a relation /
///   expression node outside the implemented core (propagated from the
///   sibling converters).
pub fn substrait_to_logical_plan(
    plan: &proto::Plan,
    provider: &MemTableProvider,
) -> BoltResult<LogicalPlan> {
    let plan_rel = plan
        .relations
        .first()
        .ok_or_else(|| BoltError::Plan("substrait: plan has no relations".into()))?;

    // A `PlanRel` is a oneof of `Rel` (bare) or `Root` (RelRoot = Rel +
    // output names). We accept either; the output-name list on a RelRoot is
    // advisory for the engine's purposes (our `Schema` already carries the
    // column names produced by the converted plan), so we currently convert
    // the inner `Rel` and ignore the explicit names. A follow-up can apply
    // them as a final rename projection.
    let rel = match &plan_rel.rel_type {
        Some(proto::plan_rel::RelType::Rel(rel)) => rel,
        Some(proto::plan_rel::RelType::Root(root)) => root
            .input
            .as_ref()
            .ok_or_else(|| BoltError::Plan("substrait: RelRoot has no input Rel".into()))?,
        None => {
            return Err(BoltError::Plan(
                "substrait: PlanRel carries neither a Rel nor a RelRoot".into(),
            ))
        }
    };

    // Build the function-extension registry: a Substrait `Plan` carries its
    // function vocabulary out-of-band in `plan.extensions`, each a
    // `SimpleExtensionDeclaration` whose `mapping_type` oneof may be an
    // `ExtensionFunction { function_anchor, name, .. }`. Scalar / aggregate
    // expressions then reference a function by its numeric `function_anchor`.
    // We index anchor -> compound name here once so the converters can resolve
    // anchors without re-walking the declaration list.
    let functions = build_function_registry(plan);

    let ctx = ConvertCtx::new(provider, &functions);
    convert_rel(rel, &ctx)
}

/// Build the `anchor -> function name` map from a plan's extension
/// declarations. Only `ExtensionFunction` mappings contribute; type / type-
/// variation mappings are irrelevant to expression conversion and skipped.
fn build_function_registry(plan: &proto::Plan) -> HashMap<u32, String> {
    use proto::extensions::simple_extension_declaration::MappingType;

    let mut map = HashMap::new();
    for decl in &plan.extensions {
        if let Some(MappingType::ExtensionFunction(f)) = &decl.mapping_type {
            map.insert(f.function_anchor, f.name.clone());
        }
    }
    map
}

/// Shared conversion context threaded through the relation and expression
/// converters.
///
/// # Why a context object
///
/// Substrait expressions reference input columns **positionally** — a
/// `FieldReference` is an *index* into the flattened input schema, not a
/// name. The engine's [`Expr::Column`](crate::plan::logical_plan::Expr) is
/// **name-based**. The expression converter therefore needs the input
/// relation's schema to map index → column name. That schema is produced by
/// the relation converter as it walks the tree, so it lives here and is
/// rebuilt (cheaply, via [`ConvertCtx::with_input_schema`]) at each relation
/// boundary.
///
/// # API contract for the sibling converters (`rel.rs` / `expr.rs`)
///
/// * `ctx.provider` — the [`MemTableProvider`]; `rel.rs` calls
///   `ctx.provider.schema(name)` to resolve a `ReadRel` named table to its
///   [`Schema`].
/// * `ctx.input_schema` — `Option<&Schema>` of the *current input relation*.
///   `None` at a leaf (`ReadRel`), `Some(_)` for any relation that has
///   already converted its child(ren). The expression converter reads this
///   to resolve a positional `FieldReference` to a column name via
///   [`ConvertCtx::field_name`].
/// * `ctx.with_input_schema(&schema)` — returns a *new* `ConvertCtx` borrowing
///   `schema` as the input schema, leaving `provider` intact. `rel.rs` uses
///   this after converting a child relation, before converting the
///   expressions (predicate / projections / group keys) that reference that
///   child's output columns.
/// * `ctx.field_name(idx)` — resolve a 0-based field index against
///   `input_schema`, returning the column *name* (`BoltResult`). The
///   expression converter calls this for every `FieldReference`.
///
/// The struct is `Copy`-cheap to clone (it holds only references), so
/// `with_input_schema` is allocation-free.
#[derive(Clone, Copy)]
pub(crate) struct ConvertCtx<'a> {
    /// Table catalog used to resolve `ReadRel` named tables to a [`Schema`].
    pub(crate) provider: &'a MemTableProvider,
    /// `anchor -> function name` registry built from the plan's extension
    /// declarations. Resolves a Substrait `function_reference` /
    /// `function_anchor` to its compound name (e.g. `"add:i32_i32"`).
    pub(crate) functions: &'a HashMap<u32, String>,
    /// Schema of the current input relation, used to resolve positional
    /// `FieldReference`s to column names. `None` at leaf relations that have
    /// no input (e.g. `ReadRel`).
    pub(crate) input_schema: Option<&'a Schema>,
}

impl<'a> ConvertCtx<'a> {
    /// Root context with no input schema yet (used at the plan entry point;
    /// `rel.rs` narrows it via [`Self::with_input_schema`] as it descends).
    pub(crate) fn new(provider: &'a MemTableProvider, functions: &'a HashMap<u32, String>) -> Self {
        Self {
            provider,
            functions,
            input_schema: None,
        }
    }

    /// Return a copy of this context with `schema` installed as the input
    /// schema (provider / function registry unchanged). Used by `rel.rs` to
    /// give the expression converter the column-name table for the relation's
    /// child output.
    ///
    /// The returned context borrows `schema` for a (possibly) shorter lifetime
    /// `'b` than `'a`; the provider / function references covariantly reborrow
    /// to `'b`. This lets `rel.rs` pass a *locally-owned* combined schema (e.g.
    /// the join output schema) without having to leak it for `'a`.
    pub(crate) fn with_input_schema<'b>(&self, schema: &'b Schema) -> ConvertCtx<'b>
    where
        'a: 'b,
    {
        ConvertCtx {
            provider: self.provider,
            functions: self.functions,
            input_schema: Some(schema),
        }
    }

    /// Resolve a `NamedTable`'s multi-part name to a base `(table, Schema)`.
    ///
    /// Substrait identifies a base table by a list of name parts (e.g.
    /// `["my_db", "t"]`). The engine catalog is flat and keyed by a single
    /// table name, so we resolve against the *last* (most-specific) segment —
    /// the bare table name — and return it alongside the registered schema.
    pub(crate) fn resolve_table(&self, names: &[String]) -> BoltResult<(String, Schema)> {
        let table = names
            .last()
            .ok_or_else(|| BoltError::Plan("substrait: NamedTable has no name parts".into()))?
            .clone();
        let schema = self.provider.schema(&table)?;
        Ok((table, schema))
    }

    /// Resolve a Substrait function anchor to its registered extension name.
    pub(crate) fn function_name(&self, anchor: u32) -> BoltResult<String> {
        self.functions.get(&anchor).cloned().ok_or_else(|| {
            BoltError::Plan(format!(
                "substrait: unknown function anchor {anchor} \
                 (not declared in plan extensions)"
            ))
        })
    }

    /// Resolve a 0-based field index against the current input schema,
    /// returning the column name. Errors if there is no input schema in
    /// scope or the index is out of range.
    pub(crate) fn field_name(&self, idx: usize) -> BoltResult<String> {
        let schema = self.input_schema.ok_or_else(|| {
            BoltError::Plan("substrait: field reference with no input schema in scope".into())
        })?;
        schema
            .fields
            .get(idx)
            .map(|f| f.name.clone())
            .ok_or_else(|| {
                BoltError::Plan(format!(
                    "substrait: field reference index {idx} out of range \
                     (input has {} columns)",
                    schema.fields.len()
                ))
            })
    }
}

/// `ConvertCtx` is THE concrete context the expression converter
/// ([`expr::convert_expr`]) runs against: it implements [`expr::SubstraitCtx`]
/// by delegating to its own inherent methods plus the module-level literal
/// mapping.
impl<'a> expr::SubstraitCtx for ConvertCtx<'a> {
    fn field_name(&self, index: usize) -> BoltResult<String> {
        ConvertCtx::field_name(self, index)
    }

    fn function_name(&self, anchor: u32) -> BoltResult<String> {
        ConvertCtx::function_name(self, anchor)
    }

    fn literal_to_literal(&self, lit: &proto::expression::Literal) -> BoltResult<Literal> {
        substrait_literal_to_literal(lit)
    }
}

/// Map a Substrait [`Type`](proto::Type) to the engine [`DataType`].
///
/// Covers the scalar types the engine supports: `bool`, `i32`, `i64`,
/// `fp32`, `fp64`, `string` (also `varchar` / `fixed_char` → `Utf8`),
/// `decimal` (→ `Decimal128(precision, scale)`), `date` (→ `Date32`), and
/// `timestamp` / `timestamp_tz` (→ `Timestamp`).
///
/// Substrait timestamps are microsecond-resolution `i64` ticks since the Unix
/// epoch (`timestamp` is naive; `timestamp_tz` is UTC-anchored), so both map
/// to [`DataType::Timestamp`] with [`TimeUnit::Microsecond`]. A `timestamp_tz`
/// records the timezone as `"UTC"` (the Substrait wire convention stores the
/// instant in UTC and carries no IANA zone name); `timestamp` maps to a naive
/// `None` zone.
///
/// Any other Substrait type (list, struct, map, interval, uuid, binary, the
/// `i8`/`i16` narrow integers, the precision-timestamp family, …) is rejected
/// with `BoltError::Plan("substrait: type <X> not yet supported")`.
pub(crate) fn substrait_type_to_dtype(t: &proto::Type) -> BoltResult<DataType> {
    use proto::r#type::Kind;
    let kind = t
        .kind
        .as_ref()
        .ok_or_else(|| BoltError::Plan("substrait: empty Type (no kind)".into()))?;
    let dtype = match kind {
        Kind::Bool(_) => DataType::Bool,
        Kind::I32(_) => DataType::Int32,
        Kind::I64(_) => DataType::Int64,
        Kind::Fp32(_) => DataType::Float32,
        Kind::Fp64(_) => DataType::Float64,
        Kind::String(_) | Kind::Varchar(_) | Kind::FixedChar(_) => DataType::Utf8,
        Kind::Decimal(d) => {
            // Substrait `precision` / `scale` are i32 on the wire; the engine
            // `Decimal128(u8 precision, i8 scale)` is the narrower domain.
            // Validate the SQL-standard envelope (1..=38 precision, scale in
            // 0..=precision) so a malformed plan surfaces a clear error here
            // rather than a panic on the `as` narrowing later.
            let precision = d.precision;
            let scale = d.scale;
            if !(1..=38).contains(&precision) {
                return Err(BoltError::Plan(format!(
                    "substrait: decimal precision {precision} out of range (1..=38)"
                )));
            }
            if scale < 0 || scale > precision {
                return Err(BoltError::Plan(format!(
                    "substrait: decimal scale {scale} out of range (0..={precision})"
                )));
            }
            DataType::Decimal128(precision as u8, scale as i8)
        }
        Kind::Date(_) => DataType::Date32,
        // Substrait `timestamp` is microsecond ticks, naive (no zone).
        Kind::Timestamp(_) => DataType::Timestamp(TimeUnit::Microsecond, None),
        // `timestamp_tz` is the same i64-micros but anchored to UTC.
        Kind::TimestampTz(_) => DataType::Timestamp(
            TimeUnit::Microsecond,
            Some(crate::plan::logical_plan::intern_timezone("UTC")),
        ),
        other => {
            return Err(BoltError::Plan(format!(
                "substrait: type {} not yet supported",
                type_kind_name(other)
            )))
        }
    };
    Ok(dtype)
}

/// Map a Substrait scalar [`Literal`](proto::expression::Literal) to the
/// engine [`Literal`].
///
/// Covers the same scalar domain as [`substrait_type_to_dtype`]: booleans,
/// the two integer widths, the two float widths, strings (`string` /
/// `var_char` / `fixed_char`), decimals, dates and timestamps. A Substrait
/// literal with the `null` field set (a typed NULL) maps to [`Literal::Null`].
///
/// Substrait encodes a decimal literal as a little-endian 16-byte two's-
/// complement value plus `precision` / `scale`; we decode it to the `i128`
/// the engine's [`Literal::Decimal128`] carries.
///
/// Anything else is rejected with
/// `BoltError::Plan("substrait: literal <X> not yet supported")`.
pub(crate) fn substrait_literal_to_literal(l: &proto::expression::Literal) -> BoltResult<Literal> {
    use proto::expression::literal::LiteralType;

    // A literal whose `null` type field is set is a typed SQL NULL.
    if matches!(&l.literal_type, Some(LiteralType::Null(_))) {
        return Ok(Literal::Null);
    }

    let lt = l
        .literal_type
        .as_ref()
        .ok_or_else(|| BoltError::Plan("substrait: empty Literal (no literal_type)".into()))?;

    let out = match lt {
        LiteralType::Boolean(b) => Literal::Bool(*b),
        LiteralType::I32(v) => Literal::Int32(*v),
        LiteralType::I64(v) => Literal::Int64(*v),
        LiteralType::Fp32(v) => Literal::Float32(*v),
        LiteralType::Fp64(v) => Literal::Float64(*v),
        LiteralType::String(s) => Literal::Utf8(s.clone()),
        LiteralType::VarChar(vc) => Literal::Utf8(vc.value.clone()),
        LiteralType::FixedChar(s) => Literal::Utf8(s.clone()),
        LiteralType::Decimal(d) => {
            // Substrait stores the unscaled value as a 16-byte little-endian
            // two's-complement integer. Decode to i128.
            let raw = decode_decimal_le(&d.value)?;
            let precision = d.precision;
            let scale = d.scale;
            if !(1..=38).contains(&precision) {
                return Err(BoltError::Plan(format!(
                    "substrait: decimal literal precision {precision} out of range (1..=38)"
                )));
            }
            if scale < 0 || scale > precision {
                return Err(BoltError::Plan(format!(
                    "substrait: decimal literal scale {scale} out of range (0..={precision})"
                )));
            }
            Literal::Decimal128(raw, precision as u8, scale as i8)
        }
        // Substrait `date` is an i32 day count since the Unix epoch — exactly
        // the engine's `Date32` storage.
        LiteralType::Date(days) => Literal::Date32(*days),
        // `timestamp` is i64 microseconds since the epoch, naive.
        LiteralType::Timestamp(micros) => Literal::Timestamp(*micros, TimeUnit::Microsecond, None),
        // `timestamp_tz` is i64 microseconds since the epoch, UTC-anchored.
        LiteralType::TimestampTz(micros) => {
            Literal::timestamp_with_tz(*micros, TimeUnit::Microsecond, Some("UTC".to_string()))
        }
        other => {
            return Err(BoltError::Plan(format!(
                "substrait: literal {} not yet supported",
                literal_type_name(other)
            )))
        }
    };
    Ok(out)
}

/// Decode a little-endian two's-complement byte slice (Substrait decimal
/// `value`, always 16 bytes) into an `i128`. A length other than 16 is a
/// malformed literal.
fn decode_decimal_le(bytes: &[u8]) -> BoltResult<i128> {
    let arr: [u8; 16] = bytes.try_into().map_err(|_| {
        BoltError::Plan(format!(
            "substrait: decimal literal value must be 16 bytes, got {}",
            bytes.len()
        ))
    })?;
    Ok(i128::from_le_bytes(arr))
}

/// Human-readable name for a Substrait type kind, for error messages on the
/// unsupported arm.
fn type_kind_name(kind: &proto::r#type::Kind) -> &'static str {
    use proto::r#type::Kind;
    match kind {
        Kind::Bool(_) => "bool",
        Kind::I8(_) => "i8",
        Kind::I16(_) => "i16",
        Kind::I32(_) => "i32",
        Kind::I64(_) => "i64",
        Kind::Fp32(_) => "fp32",
        Kind::Fp64(_) => "fp64",
        Kind::String(_) => "string",
        Kind::Binary(_) => "binary",
        Kind::Timestamp(_) => "timestamp",
        Kind::Date(_) => "date",
        Kind::Time(_) => "time",
        Kind::IntervalYear(_) => "interval_year",
        Kind::IntervalDay(_) => "interval_day",
        Kind::TimestampTz(_) => "timestamp_tz",
        Kind::Uuid(_) => "uuid",
        Kind::FixedChar(_) => "fixed_char",
        Kind::Varchar(_) => "varchar",
        Kind::FixedBinary(_) => "fixed_binary",
        Kind::Decimal(_) => "decimal",
        Kind::Struct(_) => "struct",
        Kind::List(_) => "list",
        Kind::Map(_) => "map",
        Kind::UserDefined(_) => "user_defined",
        Kind::UserDefinedTypeReference(_) => "user_defined_type_reference",
        _ => "<unknown>",
    }
}

/// Human-readable name for a Substrait literal kind, for error messages on
/// the unsupported arm.
fn literal_type_name(lt: &proto::expression::literal::LiteralType) -> &'static str {
    use proto::expression::literal::LiteralType;
    match lt {
        LiteralType::Boolean(_) => "boolean",
        LiteralType::I8(_) => "i8",
        LiteralType::I16(_) => "i16",
        LiteralType::I32(_) => "i32",
        LiteralType::I64(_) => "i64",
        LiteralType::Fp32(_) => "fp32",
        LiteralType::Fp64(_) => "fp64",
        LiteralType::String(_) => "string",
        LiteralType::Binary(_) => "binary",
        LiteralType::Timestamp(_) => "timestamp",
        LiteralType::Date(_) => "date",
        LiteralType::Time(_) => "time",
        LiteralType::IntervalYearToMonth(_) => "interval_year_to_month",
        LiteralType::IntervalDayToSecond(_) => "interval_day_to_second",
        LiteralType::FixedChar(_) => "fixed_char",
        LiteralType::VarChar(_) => "var_char",
        LiteralType::FixedBinary(_) => "fixed_binary",
        LiteralType::Decimal(_) => "decimal",
        LiteralType::Struct(_) => "struct",
        LiteralType::Map(_) => "map",
        LiteralType::TimestampTz(_) => "timestamp_tz",
        LiteralType::Uuid(_) => "uuid",
        LiteralType::Null(_) => "null",
        LiteralType::List(_) => "list",
        LiteralType::EmptyList(_) => "empty_list",
        LiteralType::EmptyMap(_) => "empty_map",
        _ => "<unknown>",
    }
}

#[cfg(test)]
mod type_mapping_tests {
    //! Host-side coverage for the scalar type / literal mapping. These need
    //! no GPU and no provider — they exercise the leaf conversions in
    //! isolation by hand-building the relevant prost messages.
    use super::*;
    use substrait::proto;

    /// Wrap a `type::Kind` into a `Type` for the mapper.
    fn ty(kind: proto::r#type::Kind) -> proto::Type {
        proto::Type { kind: Some(kind) }
    }

    /// Default nullable type-parameter struct for a given kind variant; the
    /// inner nullability / variation fields are irrelevant to the dtype
    /// mapping so `Default` is fine.
    #[test]
    fn primitive_types_map() {
        use proto::r#type::Kind;
        assert_eq!(
            substrait_type_to_dtype(&ty(Kind::Bool(Default::default()))).unwrap(),
            DataType::Bool
        );
        assert_eq!(
            substrait_type_to_dtype(&ty(Kind::I32(Default::default()))).unwrap(),
            DataType::Int32
        );
        assert_eq!(
            substrait_type_to_dtype(&ty(Kind::I64(Default::default()))).unwrap(),
            DataType::Int64
        );
        assert_eq!(
            substrait_type_to_dtype(&ty(Kind::Fp32(Default::default()))).unwrap(),
            DataType::Float32
        );
        assert_eq!(
            substrait_type_to_dtype(&ty(Kind::Fp64(Default::default()))).unwrap(),
            DataType::Float64
        );
        assert_eq!(
            substrait_type_to_dtype(&ty(Kind::String(Default::default()))).unwrap(),
            DataType::Utf8
        );
        assert_eq!(
            substrait_type_to_dtype(&ty(Kind::Date(Default::default()))).unwrap(),
            DataType::Date32
        );
    }

    #[test]
    fn varchar_and_fixedchar_map_to_utf8() {
        use proto::r#type::Kind;
        assert_eq!(
            substrait_type_to_dtype(&ty(Kind::Varchar(Default::default()))).unwrap(),
            DataType::Utf8
        );
        assert_eq!(
            substrait_type_to_dtype(&ty(Kind::FixedChar(Default::default()))).unwrap(),
            DataType::Utf8
        );
    }

    #[test]
    fn timestamp_maps_to_micros() {
        use proto::r#type::Kind;
        assert_eq!(
            substrait_type_to_dtype(&ty(Kind::Timestamp(Default::default()))).unwrap(),
            DataType::Timestamp(TimeUnit::Microsecond, None)
        );
        // timestamp_tz carries a UTC zone.
        match substrait_type_to_dtype(&ty(Kind::TimestampTz(Default::default()))).unwrap() {
            DataType::Timestamp(TimeUnit::Microsecond, Some(tz)) => assert_eq!(tz, "UTC"),
            other => panic!("expected UTC timestamp, got {other:?}"),
        }
    }

    #[test]
    fn decimal_maps_with_precision_scale() {
        use proto::r#type::Kind;
        let dec = proto::r#type::Decimal {
            scale: 2,
            precision: 10,
            ..Default::default()
        };
        assert_eq!(
            substrait_type_to_dtype(&ty(Kind::Decimal(dec))).unwrap(),
            DataType::Decimal128(10, 2)
        );
    }

    #[test]
    fn decimal_out_of_range_precision_rejected() {
        use proto::r#type::Kind;
        let dec = proto::r#type::Decimal {
            scale: 0,
            precision: 99,
            ..Default::default()
        };
        assert!(substrait_type_to_dtype(&ty(Kind::Decimal(dec))).is_err());
    }

    #[test]
    fn empty_type_rejected() {
        assert!(substrait_type_to_dtype(&proto::Type { kind: None }).is_err());
    }

    #[test]
    fn unsupported_type_has_clear_message() {
        use proto::r#type::Kind;
        let err = substrait_type_to_dtype(&ty(Kind::I8(Default::default()))).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("substrait"), "msg = {msg}");
        assert!(msg.contains("i8"), "msg = {msg}");
        assert!(msg.contains("not yet supported"), "msg = {msg}");
    }
}

#[cfg(test)]
mod literal_mapping_tests {
    use super::*;
    use proto::expression::literal::LiteralType;
    use substrait::proto;

    /// Wrap a `LiteralType` into a `Literal` message.
    fn lit(lt: LiteralType) -> proto::expression::Literal {
        proto::expression::Literal {
            literal_type: Some(lt),
            ..Default::default()
        }
    }

    #[test]
    fn scalar_literals_map() {
        assert_eq!(
            substrait_literal_to_literal(&lit(LiteralType::Boolean(true))).unwrap(),
            Literal::Bool(true)
        );
        assert_eq!(
            substrait_literal_to_literal(&lit(LiteralType::I32(7))).unwrap(),
            Literal::Int32(7)
        );
        assert_eq!(
            substrait_literal_to_literal(&lit(LiteralType::I64(7))).unwrap(),
            Literal::Int64(7)
        );
        assert_eq!(
            substrait_literal_to_literal(&lit(LiteralType::Fp32(1.5))).unwrap(),
            Literal::Float32(1.5)
        );
        assert_eq!(
            substrait_literal_to_literal(&lit(LiteralType::Fp64(1.5))).unwrap(),
            Literal::Float64(1.5)
        );
        assert_eq!(
            substrait_literal_to_literal(&lit(LiteralType::String("hi".into()))).unwrap(),
            Literal::Utf8("hi".into())
        );
        assert_eq!(
            substrait_literal_to_literal(&lit(LiteralType::Date(123))).unwrap(),
            Literal::Date32(123)
        );
    }

    #[test]
    fn timestamp_literals_map() {
        assert_eq!(
            substrait_literal_to_literal(&lit(LiteralType::Timestamp(42))).unwrap(),
            Literal::Timestamp(42, TimeUnit::Microsecond, None)
        );
        match substrait_literal_to_literal(&lit(LiteralType::TimestampTz(42))).unwrap() {
            Literal::Timestamp(ticks, TimeUnit::Microsecond, Some(tz)) => {
                assert_eq!(ticks, 42);
                assert_eq!(tz, "UTC");
            }
            other => panic!("expected UTC timestamp literal, got {other:?}"),
        }
    }

    #[test]
    fn decimal_literal_decodes_le_value() {
        // value = 12345 (unscaled), precision 10, scale 2 → 123.45
        let raw: i128 = 12345;
        let d = proto::expression::literal::Decimal {
            value: raw.to_le_bytes().to_vec(),
            precision: 10,
            scale: 2,
        };
        assert_eq!(
            substrait_literal_to_literal(&lit(LiteralType::Decimal(d))).unwrap(),
            Literal::Decimal128(12345, 10, 2)
        );
    }

    #[test]
    fn decimal_literal_negative_value() {
        let raw: i128 = -98765;
        let d = proto::expression::literal::Decimal {
            value: raw.to_le_bytes().to_vec(),
            precision: 12,
            scale: 3,
        };
        assert_eq!(
            substrait_literal_to_literal(&lit(LiteralType::Decimal(d))).unwrap(),
            Literal::Decimal128(-98765, 12, 3)
        );
    }

    #[test]
    fn decimal_literal_bad_length_rejected() {
        let d = proto::expression::literal::Decimal {
            value: vec![1, 2, 3], // not 16 bytes
            precision: 10,
            scale: 2,
        };
        assert!(substrait_literal_to_literal(&lit(LiteralType::Decimal(d))).is_err());
    }

    #[test]
    fn null_literal_maps_to_null() {
        let null_ty = proto::Type {
            kind: Some(proto::r#type::Kind::I64(Default::default())),
        };
        let l = lit(LiteralType::Null(null_ty));
        assert_eq!(substrait_literal_to_literal(&l).unwrap(), Literal::Null);
    }

    #[test]
    fn empty_literal_rejected() {
        let l = proto::expression::Literal {
            literal_type: None,
            ..Default::default()
        };
        assert!(substrait_literal_to_literal(&l).is_err());
    }

    #[test]
    fn unsupported_literal_has_clear_message() {
        let err = substrait_literal_to_literal(&lit(LiteralType::I8(3))).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("substrait"), "msg = {msg}");
        assert!(msg.contains("not yet supported"), "msg = {msg}");
    }
}
