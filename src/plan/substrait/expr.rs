// SPDX-License-Identifier: Apache-2.0

//! Substrait `Expression` → engine [`Expr`] conversion (feature `substrait`).
//!
//! This module is the expression half of the Substrait *ingestion* path: it
//! lowers a [`substrait::proto::Expression`] tree into the engine's
//! [`crate::plan::logical_plan::Expr`]. It is deliberately host-only and
//! carries no GPU concerns — the produced `Expr` re-enters the normal logical
//! → physical lowering pipeline exactly like a SQL-frontend-built expression.
//!
//! # Conversion coverage (core slice)
//!
//! * [`Literal`](substrait::proto::expression::Literal) → [`Expr::Literal`]
//!   (delegated to [`SubstraitCtx::literal_to_literal`], owned by `mod.rs` /
//!   task B2a so the literal vocabulary stays in one place).
//! * [`FieldReference`](substrait::proto::expression::FieldReference) — a
//!   *direct* struct-field selection — → [`Expr::Column`], resolving the
//!   0-based field index against the input schema via
//!   [`SubstraitCtx::field_name`].
//! * [`ScalarFunction`](substrait::proto::expression::ScalarFunction) → an
//!   [`Expr::Binary`] (arithmetic / comparison / logical / integer ops), or
//!   one of [`Expr::Like`], [`Expr::Cast`] (via `cast` function), or a
//!   COALESCE-as-`CASE` rewrite, keyed by the function's *anchor name* looked
//!   up through [`SubstraitCtx::function_name`].
//! * [`Cast`](substrait::proto::expression::Cast) → [`Expr::Cast`].
//! * [`IfThen`](substrait::proto::expression::IfThen) /
//!   [`SwitchExpression`](substrait::proto::expression::SwitchExpression) →
//!   [`Expr::Case`].
//!
//! Anything outside this envelope maps to [`BoltError::Plan`] with a message
//! that names the offending node, so a partially-supported plan fails loudly
//! rather than silently mis-lowering.
//!
//! # Context contract (shared with B2a / B2b — see integration notes)
//!
//! The conversion needs two pieces of plan-global state that live outside a
//! single `Expression`: the **input schema** (to turn a field *index* into a
//! column *name*) and the **function-extension registry** (to turn a function
//! *anchor* into a function *name*). Both are threaded through the
//! [`SubstraitCtx`] trait, which `mod.rs` (B2a) implements on its concrete
//! conversion context. Defining it as a trait here keeps this module
//! compilable and unit-testable in isolation (the tests below use a tiny
//! in-module fake) while letting B2a own the real registry plumbing.

#![cfg(feature = "substrait")]

use crate::error::{BoltError, BoltResult};
use crate::plan::logical_plan::{BinaryOp, DataType, Expr, Literal};

use substrait::proto::expression::{
    field_reference::ReferenceType as FieldReferenceType,
    reference_segment::ReferenceType as SegmentReferenceType,
    Cast as SubstraitCast, FieldReference, IfThen, Literal as SubstraitLiteral,
    ReferenceSegment, ScalarFunction, SwitchExpression,
};
use substrait::proto::function_argument::ArgType;
use substrait::proto::r#type::Kind as TypeKind;
use substrait::proto::Expression;

/// Plan-global resolution context the expression converter needs.
///
/// Implemented by the concrete conversion context in `mod.rs` (task B2a).
/// Kept as a trait so this module compiles and unit-tests stand-alone; the two
/// methods are the *only* coupling between expression conversion and the
/// rest of the ingestion machinery.
///
/// ## For B2a / B2b implementors
///
/// * [`field_name`](Self::field_name) — map a 0-based input field index to its
///   column name. The input schema is whatever relation feeds the expression
///   (a `Read`'s base schema, or the output schema of the child rel for a
///   `Project`/`Filter`). Return [`BoltError::Plan`] for an out-of-range index.
/// * [`function_name`](Self::function_name) — map a Substrait *function anchor*
///   (`ScalarFunction::function_reference`) to its registered extension name.
///   The name SHOULD be the Substrait simple/compound name (e.g. `"add"`,
///   `"equal"`, `"and"`); this converter strips any `:<type-suffix>` itself, so
///   passing either `"add"` or `"add:i32_i32"` works. Return
///   [`BoltError::Plan`] for an unknown anchor.
/// * [`literal_to_literal`](Self::literal_to_literal) — convert a Substrait
///   literal into the engine [`Literal`]. Owned by B2a (`substrait_literal_to_literal`
///   in `mod.rs`) so the literal vocabulary lives in one place; the default
///   impl here errors so a partial B2a build still links.
pub(crate) trait SubstraitCtx {
    /// Resolve a 0-based input field index to its column name.
    fn field_name(&self, index: usize) -> BoltResult<String>;

    /// Resolve a Substrait function anchor to its registered extension name.
    fn function_name(&self, anchor: u32) -> BoltResult<String>;

    /// Convert a Substrait literal into the engine [`Literal`].
    ///
    /// B2a overrides this to call its `substrait_literal_to_literal`. The
    /// default errors so the trait is object-safe and a not-yet-wired B2a
    /// still compiles.
    fn literal_to_literal(&self, _lit: &SubstraitLiteral) -> BoltResult<Literal> {
        Err(BoltError::Plan(
            "substrait: literal conversion not wired (SubstraitCtx::literal_to_literal)".into(),
        ))
    }
}

/// Convert a Substrait [`Expression`] into the engine [`Expr`].
///
/// Entry point for the expression half of Substrait ingestion. See the
/// module docs for the supported-node envelope; unsupported nodes surface a
/// [`BoltError::Plan`].
pub(crate) fn convert_expr<C: SubstraitCtx + ?Sized>(
    e: &Expression,
    ctx: &C,
) -> BoltResult<Expr> {
    convert_expr_depth(e, ctx, 0)
}

/// Bound on Substrait expression nesting, mirroring the SQL frontend's guard
/// against adversarially deep trees reaching the converter.
const MAX_SUBSTRAIT_DEPTH: usize = 128;

fn convert_expr_depth<C: SubstraitCtx + ?Sized>(
    e: &Expression,
    ctx: &C,
    depth: usize,
) -> BoltResult<Expr> {
    if depth > MAX_SUBSTRAIT_DEPTH {
        return Err(BoltError::Plan(format!(
            "substrait: expression nesting exceeds depth limit ({MAX_SUBSTRAIT_DEPTH})"
        )));
    }
    use substrait::proto::expression::RexType;
    let rex = e
        .rex_type
        .as_ref()
        .ok_or_else(|| BoltError::Plan("substrait: Expression has no rex_type".into()))?;
    match rex {
        RexType::Literal(lit) => Ok(Expr::Literal(ctx.literal_to_literal(lit)?)),
        RexType::Selection(field_ref) => convert_field_reference(field_ref, ctx),
        RexType::ScalarFunction(sf) => convert_scalar_function(sf, ctx, depth),
        RexType::Cast(cast) => convert_cast(cast, ctx, depth),
        RexType::IfThen(if_then) => convert_if_then(if_then, ctx, depth),
        RexType::SwitchExpression(switch) => convert_switch(switch, ctx, depth),
        other => Err(BoltError::Plan(format!(
            "substrait: unsupported expression node {}",
            rex_type_name(other)
        ))),
    }
}

/// Human-readable tag for an unsupported `RexType` (for error messages).
fn rex_type_name(rex: &substrait::proto::expression::RexType) -> &'static str {
    use substrait::proto::expression::RexType;
    match rex {
        RexType::Literal(_) => "Literal",
        RexType::Selection(_) => "FieldReference",
        RexType::ScalarFunction(_) => "ScalarFunction",
        RexType::WindowFunction(_) => "WindowFunction",
        RexType::IfThen(_) => "IfThen",
        RexType::SwitchExpression(_) => "SwitchExpression",
        RexType::SingularOrList(_) => "SingularOrList",
        RexType::MultiOrList(_) => "MultiOrList",
        RexType::Cast(_) => "Cast",
        RexType::Subquery(_) => "Subquery",
        RexType::Nested(_) => "Nested",
        RexType::Enum(_) => "Enum",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------
// FieldReference -> Expr::Column
// ---------------------------------------------------------------------------

/// Convert a *direct struct-field* reference to [`Expr::Column`].
///
/// Only the common case is supported: a [`FieldReference`] whose
/// `reference_type` is `DirectReference` wrapping a single
/// `StructField { field, child: None }`. Masked references, indirect
/// references, nested struct paths, list/map element access, and references
/// rooted in an outer query / enum are rejected with a clear message.
fn convert_field_reference<C: SubstraitCtx + ?Sized>(
    field_ref: &FieldReference,
    ctx: &C,
) -> BoltResult<Expr> {
    let rt = field_ref.reference_type.as_ref().ok_or_else(|| {
        BoltError::Plan("substrait: FieldReference has no reference_type".into())
    })?;
    let seg = match rt {
        FieldReferenceType::DirectReference(seg) => seg,
        FieldReferenceType::MaskedReference(_) => {
            return Err(BoltError::Plan(
                "substrait: masked FieldReference is not supported".into(),
            ));
        }
    };
    let index = direct_struct_field_index(seg)?;
    let name = ctx.field_name(index)?;
    Ok(Expr::Column(name))
}

/// Extract the 0-based struct-field index from a *flat* direct reference
/// segment (`StructField { field, child: None }`). Nested paths
/// (`child: Some(..)`) and non-struct segments (list/map element) are rejected.
fn direct_struct_field_index(seg: &ReferenceSegment) -> BoltResult<usize> {
    let inner = seg.reference_type.as_ref().ok_or_else(|| {
        BoltError::Plan("substrait: ReferenceSegment has no reference_type".into())
    })?;
    match inner {
        SegmentReferenceType::StructField(sf) => {
            if sf.child.is_some() {
                return Err(BoltError::Plan(
                    "substrait: nested struct-field reference is not supported \
                     (only flat top-level columns)"
                        .into(),
                ));
            }
            if sf.field < 0 {
                return Err(BoltError::Plan(format!(
                    "substrait: negative struct-field index {}",
                    sf.field
                )));
            }
            Ok(sf.field as usize)
        }
        SegmentReferenceType::ListElement(_) => Err(BoltError::Plan(
            "substrait: list-element reference is not supported".into(),
        )),
        SegmentReferenceType::MapKey(_) => Err(BoltError::Plan(
            "substrait: map-key reference is not supported".into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// ScalarFunction -> Expr::Binary / Like / Cast / Coalesce-as-Case
// ---------------------------------------------------------------------------

/// Convert a Substrait [`ScalarFunction`] into an engine [`Expr`].
///
/// The function anchor is resolved to its extension name, the `:<types>`
/// suffix is stripped, and the bare name is matched against the engine's
/// operator vocabulary. Binary operators take exactly two value arguments;
/// `not`, `coalesce`, `like`/`cast` have their own arities.
fn convert_scalar_function<C: SubstraitCtx + ?Sized>(
    sf: &ScalarFunction,
    ctx: &C,
    depth: usize,
) -> BoltResult<Expr> {
    let raw = ctx.function_name(sf.function_reference)?;
    let name = strip_type_suffix(&raw).to_ascii_lowercase();

    // Materialise the value arguments up front (most ops are over values).
    // Non-value arguments (enum / type args) are rejected lazily by the
    // helpers that need them.
    let args = scalar_fn_value_args(sf, ctx, depth)?;

    // Binary operators: arithmetic, comparison, logical, integer/bitwise.
    if let Some(op) = binary_op_from_name(&name) {
        let (l, r) = expect_binary(&name, args)?;
        return Ok(Expr::Binary {
            op,
            left: Box::new(l),
            right: Box::new(r),
        });
    }

    match name.as_str() {
        // n-ary AND / OR fold left-associatively into a chain of Binary nodes.
        "and" => fold_binary(BinaryOp::And, &name, args),
        "or" => fold_binary(BinaryOp::Or, &name, args),
        // Logical NOT -> engine UnaryOp::Not.
        "not" => {
            let operand = expect_unary(&name, args)?;
            Ok(Expr::Unary {
                op: crate::plan::logical_plan::UnaryOp::Not,
                operand: Box::new(operand),
            })
        }
        // COALESCE(a, b, c) is rewritten as nested CASE: the engine has no
        // dedicated Coalesce node, but `CASE WHEN a IS NOT NULL THEN a ...`
        // is semantically identical and already lowerable.
        "coalesce" => coalesce_to_case(&name, args),
        // LIKE: `like(expr, pattern)`; pattern must be a string literal.
        "like" => like_from_args(args, /*negated=*/ false, /*ci=*/ false),
        "not_like" => like_from_args(args, /*negated=*/ true, /*ci=*/ false),
        // ILIKE is a common extension spelling for case-insensitive LIKE.
        "ilike" => like_from_args(args, /*negated=*/ false, /*ci=*/ true),
        other => Err(BoltError::Plan(format!(
            "substrait: unsupported scalar function '{other}'"
        ))),
    }
}

/// Collect the *value* arguments of a scalar function as converted [`Expr`]s.
///
/// Substrait function arguments can be values, types, or enums. This engine's
/// scalar/binary ops take only value arguments; a type/enum argument in a
/// position we expect a value is a hard error.
fn scalar_fn_value_args<C: SubstraitCtx + ?Sized>(
    sf: &ScalarFunction,
    ctx: &C,
    depth: usize,
) -> BoltResult<Vec<Expr>> {
    let mut out = Vec::with_capacity(sf.arguments.len());
    for (i, arg) in sf.arguments.iter().enumerate() {
        let at = arg.arg_type.as_ref().ok_or_else(|| {
            BoltError::Plan(format!("substrait: function argument {i} has no arg_type"))
        })?;
        match at {
            ArgType::Value(v) => out.push(convert_expr_depth(v, ctx, depth + 1)?),
            ArgType::Type(_) => {
                return Err(BoltError::Plan(format!(
                    "substrait: type argument at position {i} is not supported in this function"
                )))
            }
            ArgType::Enum(_) => {
                return Err(BoltError::Plan(format!(
                    "substrait: enum argument at position {i} is not supported in this function"
                )))
            }
        }
    }
    Ok(out)
}

/// Map a bare (suffix-stripped, lowercase) Substrait function name to an engine
/// [`BinaryOp`]. Covers arithmetic, comparison, logical-binary, and the
/// integer / bitwise / shift family. Returns `None` for non-binary names.
fn binary_op_from_name(name: &str) -> Option<BinaryOp> {
    Some(match name {
        // Arithmetic.
        "add" => BinaryOp::Add,
        "subtract" => BinaryOp::Sub,
        "multiply" => BinaryOp::Mul,
        "divide" => BinaryOp::Div,
        // Comparison.
        "equal" => BinaryOp::Eq,
        "not_equal" => BinaryOp::NotEq,
        "lt" | "less_than" => BinaryOp::Lt,
        "lte" | "less_than_equal" => BinaryOp::LtEq,
        "gt" | "greater_than" => BinaryOp::Gt,
        "gte" | "greater_than_equal" => BinaryOp::GtEq,
        // String concatenation (binary form). Substrait's `concat` is n-ary
        // over strings; the engine's `||` is binary Concat, so we only accept
        // the two-arg form here and fold longer ones below via fold_binary if
        // needed — but Concat is intentionally NOT folded n-ary because the
        // engine treats it as a strict binary op.
        "concat" => BinaryOp::Concat,
        // Integer modulo + bitwise + shift family (new integer ops).
        "modulus" | "modulo" | "mod" => BinaryOp::Mod,
        "bitwise_and" => BinaryOp::BitAnd,
        "bitwise_or" => BinaryOp::BitOr,
        "bitwise_xor" => BinaryOp::BitXor,
        "shift_left" => BinaryOp::Shl,
        "shift_right" => BinaryOp::Shr,
        _ => return None,
    })
}

/// Strip a Substrait compound-name `:<type-suffix>` (e.g. `add:i32_i32` →
/// `add`). The simple-name form (no colon) passes through unchanged.
fn strip_type_suffix(name: &str) -> &str {
    match name.split_once(':') {
        Some((base, _)) => base,
        None => name,
    }
}

/// Require exactly two arguments for a binary op, returning `(left, right)`.
fn expect_binary(name: &str, mut args: Vec<Expr>) -> BoltResult<(Expr, Expr)> {
    if args.len() != 2 {
        return Err(BoltError::Plan(format!(
            "substrait: '{name}' expects 2 arguments, got {}",
            args.len()
        )));
    }
    let right = args.pop().expect("len checked == 2");
    let left = args.pop().expect("len checked == 2");
    Ok((left, right))
}

/// Require exactly one argument (for unary ops like `not`).
fn expect_unary(name: &str, mut args: Vec<Expr>) -> BoltResult<Expr> {
    if args.len() != 1 {
        return Err(BoltError::Plan(format!(
            "substrait: '{name}' expects 1 argument, got {}",
            args.len()
        )));
    }
    Ok(args.pop().expect("len checked == 1"))
}

/// Fold `>= 2` arguments left-associatively into a chain of [`Expr::Binary`]
/// with `op`. Used for the n-ary `and` / `or` spellings Substrait emits.
fn fold_binary(op: BinaryOp, name: &str, args: Vec<Expr>) -> BoltResult<Expr> {
    if args.len() < 2 {
        return Err(BoltError::Plan(format!(
            "substrait: '{name}' expects at least 2 arguments, got {}",
            args.len()
        )));
    }
    let mut it = args.into_iter();
    let mut acc = it.next().expect("len checked >= 2");
    for next in it {
        acc = Expr::Binary {
            op,
            left: Box::new(acc),
            right: Box::new(next),
        };
    }
    Ok(acc)
}

/// Rewrite `COALESCE(a, b, c, ...)` into the equivalent nested
/// `CASE WHEN a IS NOT NULL THEN a WHEN b IS NOT NULL THEN b ... ELSE <last> END`.
///
/// The last argument becomes the ELSE so an all-null input falls through to it
/// (matching SQL COALESCE, which returns NULL when every argument is NULL only
/// if the last argument is NULL too).
fn coalesce_to_case(name: &str, args: Vec<Expr>) -> BoltResult<Expr> {
    if args.is_empty() {
        return Err(BoltError::Plan(format!(
            "substrait: '{name}' (coalesce) expects at least 1 argument"
        )));
    }
    if args.len() == 1 {
        // COALESCE(x) === x.
        return Ok(args.into_iter().next().expect("len checked == 1"));
    }
    let mut branches: Vec<(Expr, Expr)> = Vec::with_capacity(args.len() - 1);
    let n = args.len();
    let mut iter = args.into_iter().enumerate().peekable();
    let mut else_branch: Option<Box<Expr>> = None;
    while let Some((i, arg)) = iter.next() {
        if i + 1 == n {
            // Last argument: the ELSE.
            else_branch = Some(Box::new(arg));
        } else {
            let cond = Expr::Unary {
                op: crate::plan::logical_plan::UnaryOp::IsNotNull,
                operand: Box::new(arg.clone()),
            };
            branches.push((cond, arg));
        }
    }
    Ok(Expr::Case {
        branches,
        else_branch,
    })
}

/// Build an [`Expr::Like`] from `[expr, pattern]`, requiring `pattern` to be a
/// string literal.
fn like_from_args(args: Vec<Expr>, negated: bool, case_insensitive: bool) -> BoltResult<Expr> {
    if args.len() != 2 {
        return Err(BoltError::Plan(format!(
            "substrait: 'like' expects 2 arguments (expr, pattern), got {}",
            args.len()
        )));
    }
    let mut it = args.into_iter();
    let expr = it.next().expect("len checked == 2");
    let pattern_expr = it.next().expect("len checked == 2");
    let pattern = match pattern_expr {
        Expr::Literal(Literal::Utf8(s)) => s,
        other => {
            return Err(BoltError::Plan(format!(
                "substrait: LIKE pattern must be a string literal, got {other:?}"
            )))
        }
    };
    Ok(Expr::Like {
        expr: Box::new(expr),
        pattern,
        escape: None,
        negated,
        case_insensitive,
    })
}

// ---------------------------------------------------------------------------
// Cast -> Expr::Cast
// ---------------------------------------------------------------------------

/// Convert a Substrait [`Cast`](SubstraitCast) into [`Expr::Cast`].
///
/// `failure_behavior == RETURN_NULL` (1) maps to a *safe* cast
/// (`TRY_CAST` semantics); `THROW_EXCEPTION` (2) and `UNSPECIFIED` (0) map to a
/// strict cast.
fn convert_cast<C: SubstraitCtx + ?Sized>(
    cast: &SubstraitCast,
    ctx: &C,
    depth: usize,
) -> BoltResult<Expr> {
    let input = cast
        .input
        .as_ref()
        .ok_or_else(|| BoltError::Plan("substrait: Cast has no input expression".into()))?;
    let inner = convert_expr_depth(input, ctx, depth + 1)?;
    let ty = cast
        .r#type
        .as_ref()
        .ok_or_else(|| BoltError::Plan("substrait: Cast has no target type".into()))?;
    let target = substrait_type_to_datatype(ty)?;
    // failure_behavior: 0 = UNSPECIFIED, 1 = RETURN_NULL, 2 = THROW_EXCEPTION.
    let safe = cast.failure_behavior == 1;
    Ok(Expr::Cast {
        expr: Box::new(inner),
        target,
        safe,
    })
}

/// Convert a Substrait [`Type`](substrait::proto::Type) into an engine
/// [`DataType`]. Only the primitive types the engine supports are accepted;
/// everything else (struct, list, map, user-defined, etc.) is rejected.
fn substrait_type_to_datatype(ty: &substrait::proto::Type) -> BoltResult<DataType> {
    use substrait::proto::r#type::{
        Decimal as DecimalType, PrecisionTimestamp, PrecisionTimestampTz, Timestamp,
        TimestampTz,
    };
    let kind = ty
        .kind
        .as_ref()
        .ok_or_else(|| BoltError::Plan("substrait: Type has no kind".into()))?;
    Ok(match kind {
        TypeKind::Bool(_) => DataType::Bool,
        TypeKind::I32(_) => DataType::Int32,
        TypeKind::I64(_) => DataType::Int64,
        TypeKind::Fp32(_) => DataType::Float32,
        TypeKind::Fp64(_) => DataType::Float64,
        TypeKind::String(_) => DataType::Utf8,
        TypeKind::Varchar(_) | TypeKind::FixedChar(_) => DataType::Utf8,
        TypeKind::Date(_) => DataType::Date32,
        TypeKind::Decimal(DecimalType {
            precision, scale, ..
        }) => decimal_datatype(*precision, *scale)?,
        // Microsecond-resolution timestamp variants map to the engine's
        // microsecond TimeUnit (the historical Substrait default).
        TypeKind::Timestamp(Timestamp { .. }) => {
            DataType::Timestamp(crate::plan::logical_plan::TimeUnit::Microsecond, None)
        }
        TypeKind::TimestampTz(TimestampTz { .. }) => {
            // Tz timestamps carry UTC semantics in Substrait but no named zone;
            // intern "UTC" so the engine retains the tz-awareness flag.
            DataType::Timestamp(
                crate::plan::logical_plan::TimeUnit::Microsecond,
                Some(crate::plan::logical_plan::intern_timezone("UTC")),
            )
        }
        TypeKind::PrecisionTimestamp(PrecisionTimestamp { precision, .. }) => {
            DataType::Timestamp(precision_to_timeunit(*precision)?, None)
        }
        TypeKind::PrecisionTimestampTz(PrecisionTimestampTz { precision, .. }) => DataType::Timestamp(
            precision_to_timeunit(*precision)?,
            Some(crate::plan::logical_plan::intern_timezone("UTC")),
        ),
        other => {
            return Err(BoltError::Plan(format!(
                "substrait: unsupported cast target type {}",
                type_kind_name(other)
            )))
        }
    })
}

/// Validate Substrait decimal precision/scale and build [`DataType::Decimal128`].
fn decimal_datatype(precision: i32, scale: i32) -> BoltResult<DataType> {
    if !(0..=38).contains(&precision) {
        return Err(BoltError::Plan(format!(
            "substrait: decimal precision {precision} out of range (0..=38)"
        )));
    }
    if !(-128..=127).contains(&scale) {
        return Err(BoltError::Plan(format!(
            "substrait: decimal scale {scale} out of i8 range"
        )));
    }
    Ok(DataType::Decimal128(precision as u8, scale as i8))
}

/// Map a Substrait timestamp `precision` (decimal digits of sub-second
/// resolution) to the engine [`TimeUnit`](crate::plan::logical_plan::TimeUnit).
fn precision_to_timeunit(
    precision: i32,
) -> BoltResult<crate::plan::logical_plan::TimeUnit> {
    use crate::plan::logical_plan::TimeUnit;
    Ok(match precision {
        0 => TimeUnit::Second,
        3 => TimeUnit::Millisecond,
        6 => TimeUnit::Microsecond,
        9 => TimeUnit::Nanosecond,
        other => {
            return Err(BoltError::Plan(format!(
                "substrait: unsupported timestamp precision {other} (expected 0/3/6/9)"
            )))
        }
    })
}

/// Human-readable tag for an unsupported `Type::Kind` (for error messages).
fn type_kind_name(kind: &TypeKind) -> &'static str {
    match kind {
        TypeKind::Bool(_) => "bool",
        TypeKind::I8(_) => "i8",
        TypeKind::I16(_) => "i16",
        TypeKind::I32(_) => "i32",
        TypeKind::I64(_) => "i64",
        TypeKind::Fp32(_) => "fp32",
        TypeKind::Fp64(_) => "fp64",
        TypeKind::String(_) => "string",
        TypeKind::Binary(_) => "binary",
        TypeKind::Timestamp(_) => "timestamp",
        TypeKind::Date(_) => "date",
        TypeKind::Time(_) => "time",
        TypeKind::IntervalYear(_) => "interval_year",
        TypeKind::IntervalDay(_) => "interval_day",
        TypeKind::TimestampTz(_) => "timestamp_tz",
        TypeKind::Uuid(_) => "uuid",
        TypeKind::FixedChar(_) => "fixedchar",
        TypeKind::Varchar(_) => "varchar",
        TypeKind::FixedBinary(_) => "fixedbinary",
        TypeKind::Decimal(_) => "decimal",
        TypeKind::Struct(_) => "struct",
        TypeKind::List(_) => "list",
        TypeKind::Map(_) => "map",
        TypeKind::UserDefined(_) => "user_defined",
        _ => "unknown",
    }
}

// ---------------------------------------------------------------------------
// IfThen / SwitchExpression -> Expr::Case
// ---------------------------------------------------------------------------

/// Convert a Substrait [`IfThen`] into [`Expr::Case`].
///
/// `IfThen.ifs` is a list of `{ if: <bool expr>, then: <value> }` clauses that
/// maps 1:1 onto the engine's CASE WHEN/THEN branches; `IfThen.else` becomes
/// the ELSE branch.
fn convert_if_then<C: SubstraitCtx + ?Sized>(
    if_then: &IfThen,
    ctx: &C,
    depth: usize,
) -> BoltResult<Expr> {
    if if_then.ifs.is_empty() {
        return Err(BoltError::Plan(
            "substrait: IfThen has no clauses".into(),
        ));
    }
    let mut branches: Vec<(Expr, Expr)> = Vec::with_capacity(if_then.ifs.len());
    for clause in &if_then.ifs {
        let cond_e = clause
            .r#if
            .as_ref()
            .ok_or_else(|| BoltError::Plan("substrait: IfThen clause has no condition".into()))?;
        let then_e = clause
            .then
            .as_ref()
            .ok_or_else(|| BoltError::Plan("substrait: IfThen clause has no `then` value".into()))?;
        let cond = convert_expr_depth(cond_e, ctx, depth + 1)?;
        let then = convert_expr_depth(then_e, ctx, depth + 1)?;
        branches.push((cond, then));
    }
    let else_branch = match &if_then.r#else {
        Some(e) => Some(Box::new(convert_expr_depth(e, ctx, depth + 1)?)),
        None => None,
    };
    Ok(Expr::Case {
        branches,
        else_branch,
    })
}

/// Convert a Substrait [`SwitchExpression`] into [`Expr::Case`].
///
/// A switch matches a `match` value against per-clause literal keys. The
/// engine has no value-form CASE, so each clause is rewritten as a boolean
/// `WHEN <match> = <key> THEN <value>` branch (the searched-CASE form), which
/// is semantically equivalent.
fn convert_switch<C: SubstraitCtx + ?Sized>(
    switch: &SwitchExpression,
    ctx: &C,
    depth: usize,
) -> BoltResult<Expr> {
    let match_e = switch
        .r#match
        .as_ref()
        .ok_or_else(|| BoltError::Plan("substrait: SwitchExpression has no `match` value".into()))?;
    let match_expr = convert_expr_depth(match_e, ctx, depth + 1)?;
    if switch.ifs.is_empty() {
        return Err(BoltError::Plan(
            "substrait: SwitchExpression has no clauses".into(),
        ));
    }
    let mut branches: Vec<(Expr, Expr)> = Vec::with_capacity(switch.ifs.len());
    for clause in &switch.ifs {
        let key_lit = clause
            .r#if
            .as_ref()
            .ok_or_else(|| BoltError::Plan("substrait: switch clause has no key literal".into()))?;
        let then_e = clause
            .then
            .as_ref()
            .ok_or_else(|| BoltError::Plan("substrait: switch clause has no `then` value".into()))?;
        let key = ctx.literal_to_literal(key_lit)?;
        let cond = Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(match_expr.clone()),
            right: Box::new(Expr::Literal(key)),
        };
        let then = convert_expr_depth(then_e, ctx, depth + 1)?;
        branches.push((cond, then));
    }
    let else_branch = match &switch.r#else {
        Some(e) => Some(Box::new(convert_expr_depth(e, ctx, depth + 1)?)),
        None => None,
    };
    Ok(Expr::Case {
        branches,
        else_branch,
    })
}

// ---------------------------------------------------------------------------
// Tests (host-only; tiny in-module fake ctx so we don't depend on B2a).
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use substrait::proto::expression::literal::LiteralType;
    use substrait::proto::expression::reference_segment::StructField;
    use substrait::proto::expression::{
        FieldReference, Literal as SubstraitLiteral, ReferenceSegment, RexType, ScalarFunction,
    };
    use substrait::proto::function_argument::ArgType;
    use substrait::proto::FunctionArgument;

    /// Tiny fake context: a fixed column list + a fixed anchor→name map. Stands
    /// in for B2a's real registry-backed context for unit tests.
    struct FakeCtx {
        cols: Vec<String>,
        fns: Vec<(u32, String)>,
    }

    impl SubstraitCtx for FakeCtx {
        fn field_name(&self, index: usize) -> BoltResult<String> {
            self.cols
                .get(index)
                .cloned()
                .ok_or_else(|| BoltError::Plan(format!("field index {index} out of range")))
        }
        fn function_name(&self, anchor: u32) -> BoltResult<String> {
            self.fns
                .iter()
                .find(|(a, _)| *a == anchor)
                .map(|(_, n)| n.clone())
                .ok_or_else(|| BoltError::Plan(format!("unknown function anchor {anchor}")))
        }
        fn literal_to_literal(&self, lit: &SubstraitLiteral) -> BoltResult<Literal> {
            // Minimal literal vocab for tests: i32, i64, fp64, string, bool.
            match lit.literal_type.as_ref() {
                Some(LiteralType::I32(v)) => Ok(Literal::Int32(*v)),
                Some(LiteralType::I64(v)) => Ok(Literal::Int64(*v)),
                Some(LiteralType::Fp64(v)) => Ok(Literal::Float64(*v)),
                Some(LiteralType::String(s)) => Ok(Literal::Utf8(s.clone())),
                Some(LiteralType::Boolean(b)) => Ok(Literal::Bool(*b)),
                _ => Err(BoltError::Plan("test ctx: unsupported literal".into())),
            }
        }
    }

    fn ctx() -> FakeCtx {
        FakeCtx {
            cols: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            fns: vec![
                (1, "add:i32_i32".to_string()),
                (2, "equal".to_string()),
                (3, "and:bool_bool".to_string()),
            ],
        }
    }

    fn lit_i32(v: i32) -> Expression {
        Expression {
            rex_type: Some(RexType::Literal(SubstraitLiteral {
                literal_type: Some(LiteralType::I32(v)),
                ..Default::default()
            })),
        }
    }

    fn field(index: i32) -> Expression {
        Expression {
            rex_type: Some(RexType::Selection(Box::new(FieldReference {
                reference_type: Some(FieldReferenceType::DirectReference(ReferenceSegment {
                    reference_type: Some(SegmentReferenceType::StructField(Box::new(
                        StructField {
                            field: index,
                            child: None,
                        },
                    ))),
                })),
                ..Default::default()
            }))),
        }
    }

    fn value_arg(e: Expression) -> FunctionArgument {
        FunctionArgument {
            arg_type: Some(ArgType::Value(e)),
        }
    }

    #[test]
    fn literal_converts() {
        let c = ctx();
        let got = convert_expr(&lit_i32(42), &c).unwrap();
        assert!(matches!(got, Expr::Literal(Literal::Int32(42))));
    }

    #[test]
    fn direct_field_reference_resolves_to_column() {
        let c = ctx();
        let got = convert_expr(&field(1), &c).unwrap();
        match got {
            Expr::Column(name) => assert_eq!(name, "b"),
            other => panic!("expected Column, got {other:?}"),
        }
    }

    #[test]
    fn field_reference_out_of_range_errors() {
        let c = ctx();
        let err = convert_expr(&field(9), &c).unwrap_err();
        assert!(matches!(err, BoltError::Plan(_)));
    }

    #[test]
    fn binary_scalar_function_add() {
        let c = ctx();
        let e = Expression {
            rex_type: Some(RexType::ScalarFunction(ScalarFunction {
                function_reference: 1, // "add:i32_i32" -> stripped to "add"
                arguments: vec![value_arg(field(0)), value_arg(lit_i32(7))],
                ..Default::default()
            })),
        };
        let got = convert_expr(&e, &c).unwrap();
        match got {
            Expr::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(*left, Expr::Column(ref n) if n == "a"));
                assert!(matches!(*right, Expr::Literal(Literal::Int32(7))));
            }
            other => panic!("expected Binary Add, got {other:?}"),
        }
    }

    #[test]
    fn comparison_scalar_function_equal() {
        let c = ctx();
        let e = Expression {
            rex_type: Some(RexType::ScalarFunction(ScalarFunction {
                function_reference: 2, // "equal"
                arguments: vec![value_arg(field(0)), value_arg(field(1))],
                ..Default::default()
            })),
        };
        let got = convert_expr(&e, &c).unwrap();
        assert!(matches!(
            got,
            Expr::Binary {
                op: BinaryOp::Eq,
                ..
            }
        ));
    }

    #[test]
    fn nary_and_folds_left() {
        let c = ctx();
        // and(a, b, c-as-lit) -> ((a AND b) AND lit)
        let e = Expression {
            rex_type: Some(RexType::ScalarFunction(ScalarFunction {
                function_reference: 3, // "and:bool_bool" -> "and"
                arguments: vec![
                    value_arg(field(0)),
                    value_arg(field(1)),
                    value_arg(lit_i32(1)),
                ],
                ..Default::default()
            })),
        };
        let got = convert_expr(&e, &c).unwrap();
        match got {
            Expr::Binary {
                op: BinaryOp::And,
                left,
                ..
            } => {
                // The left of the outer AND is itself an AND.
                assert!(matches!(
                    *left,
                    Expr::Binary {
                        op: BinaryOp::And,
                        ..
                    }
                ));
            }
            other => panic!("expected nested And, got {other:?}"),
        }
    }

    #[test]
    fn unknown_function_errors() {
        let c = FakeCtx {
            cols: vec!["a".to_string()],
            fns: vec![(5, "no_such_fn".to_string())],
        };
        let e = Expression {
            rex_type: Some(RexType::ScalarFunction(ScalarFunction {
                function_reference: 5,
                arguments: vec![value_arg(field(0))],
                ..Default::default()
            })),
        };
        let err = convert_expr(&e, &c).unwrap_err();
        assert!(matches!(err, BoltError::Plan(_)));
    }

    #[test]
    fn missing_rex_type_errors() {
        let c = ctx();
        let e = Expression { rex_type: None };
        let err = convert_expr(&e, &c).unwrap_err();
        assert!(matches!(err, BoltError::Plan(_)));
    }
}
