// SPDX-License-Identifier: Apache-2.0

//! Logical plan AST: schemas, expressions, and relational nodes.

use crate::error::{BoltError, BoltResult};

/// Time-unit for `DataType::Timestamp` values.
///
/// Mirrors `arrow::datatypes::TimeUnit` and indicates the resolution of the
/// underlying `i64` count of ticks since the Unix epoch (1970-01-01T00:00:00Z).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    /// One tick = 1 second.
    Second,
    /// One tick = 1 millisecond (10^-3 s).
    Millisecond,
    /// One tick = 1 microsecond (10^-6 s).
    Microsecond,
    /// One tick = 1 nanosecond (10^-9 s). Matches Arrow's default
    /// `TimestampNanosecondArray` storage.
    Nanosecond,
}

/// Intern a timezone name string into the process-static interner so a
/// [`DataType::Timestamp`] value can be `Copy` while still carrying an
/// optional IANA timezone name.
///
/// The set of IANA timezone names is bounded (~600 strings) and the planner
/// only materialises a handful per query, so the small one-time leak is
/// acceptable. Subsequent calls with the same string return the same
/// `&'static str` so `Eq` / `Hash` on `DataType` are well-defined.
pub fn intern_timezone(name: &str) -> &'static str {
    use std::collections::HashSet;
    use std::sync::Mutex;
    static INTERN: Mutex<Option<HashSet<&'static str>>> = Mutex::new(None);
    let mut guard = INTERN.lock().expect("timezone interner mutex poisoned");
    let set = guard.get_or_insert_with(HashSet::new);
    if let Some(existing) = set.get(name) {
        return *existing;
    }
    let leaked: &'static str = Box::leak(name.to_string().into_boxed_str());
    set.insert(leaked);
    leaked
}

/// Minimal set of column data types the GPU engine handles.
///
/// **v0.6 / M4**: `Date32` and `Timestamp` were added. The variant signature
/// for `Timestamp` is documented in the project spec as
/// `Timestamp(TimeUnit, Option<String>)`. To keep `DataType: Copy` (a deep
/// invariant the executor depends on across hundreds of by-value call sites)
/// without churning the entire codebase, the timezone is stored as
/// `Option<&'static str>` interned via [`intern_timezone`] — semantically
/// equivalent to `Option<String>`. The companion [`Literal::Timestamp`]
/// constructor accepts a `String` and routes it through the interner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// Boolean (one byte on device).
    Bool,
    /// 32-bit signed integer.
    Int32,
    /// 64-bit signed integer.
    Int64,
    /// 32-bit IEEE-754 float.
    Float32,
    /// 64-bit IEEE-754 float.
    Float64,
    /// UTF-8 string; variable width, only legal in filter/group-by columns.
    Utf8,
    /// Fixed-point decimal with `precision` total digits and `scale` digits
    /// after the decimal point (v0.6 / M4 plan-level only — GPU codegen
    /// rejects this with "Decimal128 not yet lowered to GPU" until the
    /// follow-up lands). Round-trips losslessly to/from Arrow's
    /// `Decimal128(precision, scale)` array type.
    Decimal128(u8, i8),
    /// Date as a 32-bit count of days since the Unix epoch. Matches Arrow's
    /// `Date32Array` storage layout.
    Date32,
    /// Timestamp stored as `i64` ticks since the Unix epoch in `TimeUnit`
    /// resolution, with an optional process-interned IANA time-zone name
    /// (see [`intern_timezone`]). A `None` timezone means "local / naive".
    Timestamp(TimeUnit, Option<&'static str>),
}

impl DataType {
    /// Byte width for fixed-width types; `None` for variable-width.
    pub fn byte_width(self) -> Option<usize> {
        match self {
            DataType::Bool => Some(1),
            DataType::Int32 => Some(4),
            DataType::Int64 => Some(8),
            DataType::Float32 => Some(4),
            DataType::Float64 => Some(8),
            DataType::Utf8 => None,
            // Decimal128 is a 16-byte (i128) fixed-width value, regardless of
            // precision / scale (those affect interpretation, not storage).
            DataType::Decimal128(_, _) => Some(16),
            // `i32` days since epoch (mirrors Arrow `Date32Array`).
            DataType::Date32 => Some(4),
            // `i64` ticks since epoch regardless of TimeUnit / tz
            // (mirrors Arrow `Timestamp*Array`).
            DataType::Timestamp(_, _) => Some(8),
        }
    }

    /// True for the floating-point types.
    fn is_float(self) -> bool {
        matches!(self, DataType::Float32 | DataType::Float64)
    }

    /// True for the integer types.
    fn is_int(self) -> bool {
        matches!(self, DataType::Int32 | DataType::Int64)
    }

    /// True for any numeric (int or float) type.
    fn is_numeric(self) -> bool {
        self.is_int() || self.is_float()
    }

    /// True if this is a `Decimal128(precision, scale)` type. Plan-level
    /// helper used by callers that want to special-case decimal handling
    /// before GPU codegen rejects the type. Not part of `is_numeric()` yet:
    /// arithmetic / comparison between decimals and the existing primitive
    /// numerics is intentionally NOT defined in v0.6, so leaving
    /// `is_numeric()` unchanged means a decimal column will surface a clear
    /// type error if anyone tries to add it to (e.g.) an Int64.
    pub fn is_decimal(self) -> bool {
        matches!(self, DataType::Decimal128(_, _))
    }
}

/// A named, typed column slot in a schema.
#[derive(Debug, Clone)]
pub struct Field {
    /// Column name.
    pub name: String,
    /// Column data type.
    pub dtype: DataType,
    /// Whether the column admits nulls.
    pub nullable: bool,
}

impl Field {
    /// Convenience constructor.
    pub fn new(name: impl Into<String>, dtype: DataType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            dtype,
            nullable,
        }
    }
}

/// Ordered list of fields describing a relation.
#[derive(Debug, Clone, Default)]
pub struct Schema {
    /// Fields in projection order.
    pub fields: Vec<Field>,
}

impl Schema {
    /// Build a schema from a vector of fields.
    pub fn new(fields: Vec<Field>) -> Self {
        Self { fields }
    }

    /// Index of `name` in this schema, or a `Plan` error if absent.
    ///
    /// Lookup is case-sensitive first. If the exact match misses and the
    /// requested `name` is all-ASCII-lowercase, falls back to an ASCII
    /// case-insensitive scan; otherwise the miss is final. On final miss
    /// the error message includes a "did you mean '<X>'?" suggestion when
    /// any field name is within edit distance 2.
    pub fn index_of(&self, name: &str) -> BoltResult<usize> {
        if let Some(i) = self.fields.iter().position(|f| f.name == name) {
            return Ok(i);
        }
        // Case-insensitive fallback only when key is already lowercase
        // (case-folded by the SQL frontend) — quoted SQL identifiers
        // / verbatim programmatic callers take the strict path.
        if !name.chars().any(|c| c.is_ascii_uppercase()) {
            if let Some(i) = self
                .fields
                .iter()
                .position(|f| f.name.eq_ignore_ascii_case(name))
            {
                return Ok(i);
            }
        }
        let suffix = crate::plan::suggest::did_you_mean_suffix(
            name,
            self.fields.iter().map(|f| f.name.as_str()),
        );
        Err(BoltError::Plan(format!(
            "column '{name}' not found in schema{suffix}"
        )))
    }

    /// Lookup a field by name, or a `Plan` error if absent.
    ///
    /// Honours the same case-sensitive-then-case-insensitive fallback as
    /// [`Self::index_of`]; see that method for the rationale.
    pub fn field(&self, name: &str) -> BoltResult<&Field> {
        let i = self.index_of(name)?;
        Ok(&self.fields[i])
    }
}

/// A scalar constant.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// SQL NULL — no static type.
    Null,
    /// Boolean constant.
    Bool(bool),
    /// 32-bit integer constant.
    Int32(i32),
    /// 64-bit integer constant.
    Int64(i64),
    /// 32-bit float constant.
    Float32(f32),
    /// 64-bit float constant.
    Float64(f64),
    /// UTF-8 string constant.
    Utf8(String),
    /// Fixed-point decimal constant: raw i128 value scaled by `10^scale`,
    /// with `precision` total digits. v0.6 / M4 plan-level only; the GPU
    /// codegen rejects this with "Decimal128 not yet lowered to GPU" until
    /// the follow-up lands.
    Decimal128(i128, u8, i8),
    /// Date constant — days since the Unix epoch (matches Arrow `Date32Array`).
    Date32(i32),
    /// Timestamp constant — `i64` ticks since the Unix epoch in the given
    /// resolution, with an optional process-interned IANA time-zone name
    /// (see [`intern_timezone`]). The variant stores the interned `&'static
    /// str` so the literal stays cheap to clone; use
    /// [`Literal::timestamp_with_tz`] to construct one from an owned `String`.
    Timestamp(i64, TimeUnit, Option<&'static str>),
}

impl Literal {
    /// Static type of this literal; `None` for `Null`.
    pub fn dtype(&self) -> Option<DataType> {
        match self {
            Literal::Null => None,
            Literal::Bool(_) => Some(DataType::Bool),
            Literal::Int32(_) => Some(DataType::Int32),
            Literal::Int64(_) => Some(DataType::Int64),
            Literal::Float32(_) => Some(DataType::Float32),
            Literal::Float64(_) => Some(DataType::Float64),
            Literal::Utf8(_) => Some(DataType::Utf8),
            Literal::Decimal128(_, p, s) => Some(DataType::Decimal128(*p, *s)),
            Literal::Date32(_) => Some(DataType::Date32),
            Literal::Timestamp(_, unit, tz) => Some(DataType::Timestamp(*unit, *tz)),
        }
    }

    /// Construct a `Literal::Timestamp` from an owned timezone `String`,
    /// routing the tz through the process-static interner so the resulting
    /// literal can be `Copy`-style cheap. Pass `tz = None` for a naive
    /// timestamp.
    pub fn timestamp_with_tz(ticks: i64, unit: TimeUnit, tz: Option<String>) -> Self {
        let interned = tz.map(|s| intern_timezone(&s));
        Literal::Timestamp(ticks, unit, interned)
    }
}

impl From<bool> for Literal {
    fn from(v: bool) -> Self {
        Literal::Bool(v)
    }
}

impl From<i32> for Literal {
    fn from(v: i32) -> Self {
        Literal::Int32(v)
    }
}

impl From<i64> for Literal {
    fn from(v: i64) -> Self {
        Literal::Int64(v)
    }
}

impl From<f32> for Literal {
    fn from(v: f32) -> Self {
        Literal::Float32(v)
    }
}

impl From<f64> for Literal {
    fn from(v: f64) -> Self {
        Literal::Float64(v)
    }
}

impl From<&str> for Literal {
    fn from(v: &str) -> Self {
        Literal::Utf8(v.to_string())
    }
}

impl From<String> for Literal {
    fn from(v: String) -> Self {
        Literal::Utf8(v)
    }
}

impl From<i128> for Literal {
    /// Lift a raw `i128` into a `Decimal128(v, 38, 0)` — i.e. max-precision
    /// integer-valued decimal. Callers that need a non-zero scale should
    /// construct the variant directly (`Literal::Decimal128(value, p, s)`).
    fn from(v: i128) -> Self {
        Literal::Decimal128(v, 38, 0)
    }
}

/// Binary operators codegen handles directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    /// `a + b`.
    Add,
    /// `a - b`.
    Sub,
    /// `a * b`.
    Mul,
    /// `a / b`.
    Div,
    /// `a = b`.
    Eq,
    /// `a <> b`.
    NotEq,
    /// `a < b`.
    Lt,
    /// `a <= b`.
    LtEq,
    /// `a > b`.
    Gt,
    /// `a >= b`.
    GtEq,
    /// `a AND b`.
    And,
    /// `a OR b`.
    Or,
    /// SQL `a || b` — string concatenation. Both operands must be Utf8;
    /// result is Utf8. Lowered host-side (the GPU codegen path has no
    /// Utf8 support); see `crate::exec::string_ops::host_concat_strings`.
    Concat,
}

impl BinaryOp {
    /// True for `+ - * /`.
    fn is_arithmetic(self) -> bool {
        matches!(self, BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div)
    }

    /// True for `= <> < <= > >=`.
    fn is_comparison(self) -> bool {
        matches!(
            self,
            BinaryOp::Eq | BinaryOp::NotEq | BinaryOp::Lt | BinaryOp::LtEq | BinaryOp::Gt | BinaryOp::GtEq
        )
    }

    /// True for `AND OR`.
    fn is_logical(self) -> bool {
        matches!(self, BinaryOp::And | BinaryOp::Or)
    }

    /// True for string ops (currently only `||`). Result is Utf8; both
    /// operands must be Utf8 (the type-checker enforces this).
    fn is_string(self) -> bool {
        matches!(self, BinaryOp::Concat)
    }
}

/// Unary operators surfaced by the planner.
///
/// Covers SQL `IS NULL` / `IS NOT NULL` and logical `NOT`. These are
/// type-checked at the logical-plan level (`IS [NOT] NULL` always produces
/// `Bool` regardless of operand dtype; `NOT` requires a `Bool` operand and
/// produces `Bool`) and surfaced through the SQL frontend. The GPU executor
/// lowers bare-column `IS [NOT] NULL` natively; `NOT` is currently rejected
/// at the physical-plan boundary so the host-side filter path can handle it
/// without misleading the user about kernel support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    /// SQL `<expr> IS NULL`.
    IsNull,
    /// SQL `<expr> IS NOT NULL`.
    IsNotNull,
    /// SQL `NOT <bool-expr>`. Operand must be `Bool`; result is `Bool`.
    Not,
}

/// Scalar string functions surfaced through the SQL frontend.
///
/// v0.5 MVP scope: parser + type-check only. The physical-plan boundary
/// rejects every variant cleanly with a `BoltError::Plan` so the planner
/// can accept the syntax without misleading the user about kernel support.
/// Execution wiring (host-side fallback or GPU codegen) is a follow-up.
///
/// Type-check rules (enforced in [`Expr::dtype_depth`]):
///
/// * `Upper(s)` / `Lower(s)`: requires a single `Utf8` argument; returns
///   `Utf8`.
/// * `Length(s)`: requires a single `Utf8` argument; returns `Int64`.
/// * `Substring(s, start)` or `Substring(s, start, length)`:
///   first argument `Utf8`, remaining arguments `Int64`; returns `Utf8`.
///   `length` is optional (2 or 3 args).
/// * `Concat(s1, s2, ...)`: at least two `Utf8` arguments; returns `Utf8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarFnKind {
    /// `UPPER(s)` — uppercase the input string.
    Upper,
    /// `LOWER(s)` — lowercase the input string.
    Lower,
    /// `LENGTH(s)` — character length of the input string, as `Int64`.
    Length,
    /// `SUBSTRING(s, start [, length])` — extract a substring.
    Substring,
    /// `CONCAT(s1, s2, ...)` — concatenate two or more strings.
    Concat,
    /// `TRIM([BOTH] [chars] FROM s)` — strip `chars` (default: whitespace)
    /// from BOTH ends of `s`. Args: `[s]` or `[s, chars]`.
    TrimBoth,
    /// `TRIM(LEADING [chars] FROM s)` — strip `chars` (default: whitespace)
    /// from the START of `s`. Args: `[s]` or `[s, chars]`.
    TrimLeading,
    /// `TRIM(TRAILING [chars] FROM s)` — strip `chars` (default: whitespace)
    /// from the END of `s`. Args: `[s]` or `[s, chars]`.
    TrimTrailing,
    /// `OCTET_LENGTH(s)` — UTF-8 byte length of `s`, as `Int64`.
    OctetLength,
    /// `POSITION(substr IN s)` / `STRPOS(s, substr)` — 1-based character index
    /// of the first occurrence of `substr` in `s` (0 if absent), as `Int64`.
    /// Args are always `[s, substr]` (the frontend normalises POSITION's
    /// `substr IN s` spelling into this order).
    Position,
    /// `REPLACE(s, from, to)` — replace every occurrence of `from` in `s` with
    /// `to`. Args: `[s, from, to]`; returns `Utf8`.
    Replace,
    /// `LEFT(s, n)` — first `n` characters of `s` (negative `n` drops from the
    /// end). Args: `[s, n]`; returns `Utf8`.
    Left,
    /// `RIGHT(s, n)` — last `n` characters of `s` (negative `n` drops from the
    /// front). Args: `[s, n]`; returns `Utf8`.
    Right,
    /// `LPAD(s, len, pad)` — left-pad/truncate `s` to `len` characters using
    /// `pad`. Args: `[s, len, pad]`; returns `Utf8`.
    Lpad,
    /// `RPAD(s, len, pad)` — right-pad/truncate `s` to `len` characters using
    /// `pad`. Args: `[s, len, pad]`; returns `Utf8`.
    Rpad,
    /// `REVERSE(s)` — reverse the characters of `s`. Args: `[s]`; returns
    /// `Utf8`.
    Reverse,
    /// `INITCAP(s)` — capitalise the first letter of each word. Args: `[s]`;
    /// returns `Utf8`.
    Initcap,
}

impl ScalarFnKind {
    /// Canonical uppercase function name as it appears in SQL (for error
    /// messages).
    pub fn sql_name(self) -> &'static str {
        match self {
            ScalarFnKind::Upper => "UPPER",
            ScalarFnKind::Lower => "LOWER",
            ScalarFnKind::Length => "LENGTH",
            ScalarFnKind::Substring => "SUBSTRING",
            ScalarFnKind::Concat => "CONCAT",
            ScalarFnKind::TrimBoth
            | ScalarFnKind::TrimLeading
            | ScalarFnKind::TrimTrailing => "TRIM",
            ScalarFnKind::OctetLength => "OCTET_LENGTH",
            ScalarFnKind::Position => "POSITION",
            ScalarFnKind::Replace => "REPLACE",
            ScalarFnKind::Left => "LEFT",
            ScalarFnKind::Right => "RIGHT",
            ScalarFnKind::Lpad => "LPAD",
            ScalarFnKind::Rpad => "RPAD",
            ScalarFnKind::Reverse => "REVERSE",
            ScalarFnKind::Initcap => "INITCAP",
        }
    }
}

/// Calendar field selectable by SQL `EXTRACT(field FROM ts)`.
///
/// **v0.7 / date-scalar-fns**: a focused subset that lowers to pure integer
/// arithmetic on the underlying fixed-width storage (`Date32` = days since the
/// Unix epoch as `i32`; `Timestamp` = `i64` ticks since the epoch). The fields
/// that need a full proleptic-Gregorian civil-date decomposition (`YEAR`,
/// `MONTH`, `DAY`, `DAYOFWEEK`) share the day-count → civil-date algorithm
/// (Howard Hinnant's `civil_from_days`), while the intra-day fields (`HOUR`,
/// `MINUTE`, `SECOND`) are plain modular arithmetic on the tick count and are
/// only defined for `Timestamp` inputs.
///
/// The numeric result is always `Int64` (matching the SQL standard, which
/// returns an exact numeric for `EXTRACT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DateField {
    /// Calendar year (e.g. 2026). Defined for `Date32` and `Timestamp`.
    Year,
    /// Month of year, 1..=12. Defined for `Date32` and `Timestamp`.
    Month,
    /// Day of month, 1..=31. Defined for `Date32` and `Timestamp`.
    Day,
    /// Hour of day, 0..=23. `Timestamp` only.
    Hour,
    /// Minute of hour, 0..=59. `Timestamp` only.
    Minute,
    /// Second of minute, 0..=59. `Timestamp` only.
    Second,
}

impl DateField {
    /// Canonical SQL spelling for error messages.
    pub fn sql_name(self) -> &'static str {
        match self {
            DateField::Year => "YEAR",
            DateField::Month => "MONTH",
            DateField::Day => "DAY",
            DateField::Hour => "HOUR",
            DateField::Minute => "MINUTE",
            DateField::Second => "SECOND",
        }
    }

    /// True for the intra-day fields that are only meaningful on a `Timestamp`
    /// (a bare `Date32` has no time-of-day component).
    pub fn is_intraday(self) -> bool {
        matches!(self, DateField::Hour | DateField::Minute | DateField::Second)
    }
}

/// Granularity selectable by SQL `DATE_TRUNC(unit, ts)`.
///
/// **v0.7 / date-scalar-fns**: `DATE_TRUNC` rounds a temporal value *down* to
/// the start of the given unit, preserving the input dtype (`Date32` →
/// `Date32`, `Timestamp` → `Timestamp` at the same `TimeUnit`/timezone). The
/// sub-day units (`Hour`, `Minute`, `Second`) are only defined on `Timestamp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DateTruncUnit {
    /// Truncate to Jan 1 of the year.
    Year,
    /// Truncate to the first day of the month.
    Month,
    /// Truncate to midnight of the day.
    Day,
    /// Truncate to the top of the hour. `Timestamp` only.
    Hour,
    /// Truncate to the top of the minute. `Timestamp` only.
    Minute,
    /// Truncate to the whole second. `Timestamp` only.
    Second,
}

impl DateTruncUnit {
    /// Canonical SQL spelling for error messages.
    pub fn sql_name(self) -> &'static str {
        match self {
            DateTruncUnit::Year => "year",
            DateTruncUnit::Month => "month",
            DateTruncUnit::Day => "day",
            DateTruncUnit::Hour => "hour",
            DateTruncUnit::Minute => "minute",
            DateTruncUnit::Second => "second",
        }
    }

    /// True for the sub-day units only valid on a `Timestamp` input.
    pub fn is_intraday(self) -> bool {
        matches!(
            self,
            DateTruncUnit::Hour | DateTruncUnit::Minute | DateTruncUnit::Second
        )
    }
}

/// Scalar expression tree.
#[derive(Debug, Clone)]
pub enum Expr {
    /// Reference to an input column by name.
    Column(String),
    /// Scalar constant.
    Literal(Literal),
    /// Two-operand expression.
    Binary {
        /// Operator.
        op: BinaryOp,
        /// Left operand.
        left: Box<Expr>,
        /// Right operand.
        right: Box<Expr>,
    },
    /// One-operand expression (currently only `IS NULL` / `IS NOT NULL`).
    ///
    /// Always type-checks to `Bool` at the logical plane regardless of the
    /// operand's dtype, including the untyped `Literal::Null` operand. The
    /// physical planner does not yet lower this to a GPU kernel — see
    /// [`crate::plan::physical_plan`].
    Unary {
        /// Unary operator.
        op: UnaryOp,
        /// The single operand.
        operand: Box<Expr>,
    },
    /// SQL `CASE WHEN cond1 THEN val1 [WHEN cond2 THEN val2 ...] [ELSE
    /// val_else] END`.
    Case {
        /// One or more WHEN/THEN branches in source order.
        branches: Vec<(Expr, Expr)>,
        /// Optional ELSE value; SQL NULL when omitted.
        else_branch: Option<Box<Expr>>,
    },
    /// SQL `expr LIKE 'pattern'` / `expr NOT LIKE 'pattern'`,
    /// optionally with `ESCAPE '<char>'`.
    ///
    /// The pattern must be a string literal constant.
    /// Wildcards: `%` matches zero-or-more characters; `_` matches exactly
    /// one character. When `escape` is `Some(c)`, a `c` in the pattern
    /// causes the next character to be interpreted literally — see
    /// [`crate::exec::like::PatternMatcher::compile`] for the exact rules.
    Like {
        /// Operand: must be a `Utf8`-typed expression.
        expr: Box<Expr>,
        /// Literal pattern (`%` = wildcard any, `_` = wildcard one).
        pattern: String,
        /// Optional ESCAPE character (single Unicode scalar). When set,
        /// any occurrence of this character in `pattern` marks the next
        /// character as a literal.
        escape: Option<char>,
        /// `true` for `NOT LIKE`, `false` for `LIKE`.
        negated: bool,
        /// `true` for `ILIKE` (case-insensitive), `false` for plain `LIKE`.
        /// When set, both the pattern and the input are Unicode
        /// case-folded (`to_lowercase`) before matching — see
        /// [`crate::exec::like::PatternMatcher::compile`].
        case_insensitive: bool,
    },
    /// SQL `CAST(expr AS type)` over primitive (non-Utf8) types.
    ///
    /// Type-checks to `target` regardless of the source dtype, but only the
    /// pairs documented on [`cast_is_supported`] are accepted at the logical
    /// plane — anything else surfaces a `BoltError::Type` from
    /// [`Expr::dtype`]. The physical planner currently rejects every `Cast`
    /// at the lowering boundary (see [`crate::plan::physical_plan::lower`]);
    /// GPU codegen for the runtime conversion is a v0.6 follow-up.
    Cast {
        /// Inner expression whose value is converted.
        expr: Box<Expr>,
        /// Target dtype the conversion produces.
        target: DataType,
    },
    /// Scalar string function call (UPPER / LOWER / LENGTH / SUBSTRING /
    /// CONCAT). v0.5 MVP scope: parser + type-check only — the physical
    /// planner rejects every variant at `lower()` with a clear "string
    /// scalar functions are not yet lowered to GPU" error. See
    /// [`ScalarFnKind`] for the per-kind type-check contract.
    ScalarFn {
        /// Which scalar function this is.
        kind: ScalarFnKind,
        /// Arguments, evaluated left-to-right.
        args: Vec<Expr>,
    },
    /// SQL `EXTRACT(field FROM ts)` — pull a calendar/clock field out of a
    /// `Date32` or `Timestamp` value as an `Int64`.
    ///
    /// **v0.7 / date-scalar-fns**: lowered to integer arithmetic on the
    /// underlying fixed-width storage. See [`DateField`] for the per-field
    /// dtype rules (intra-day fields require a `Timestamp` operand). The GPU
    /// codegen lives in [`crate::jit::date_scalar`].
    Extract {
        /// Which calendar/clock field to extract.
        field: DateField,
        /// Operand expression; must type-check to `Date32` or `Timestamp`.
        expr: Box<Expr>,
    },
    /// SQL `DATE_TRUNC(unit, ts)` — round a temporal value down to the start
    /// of `unit`, preserving the operand's dtype.
    ///
    /// **v0.7 / date-scalar-fns**: lowered to integer arithmetic on the
    /// underlying fixed-width storage. See [`DateTruncUnit`] for the per-unit
    /// dtype rules (sub-day units require a `Timestamp` operand). The GPU
    /// codegen lives in [`crate::jit::date_scalar`].
    DateTrunc {
        /// Truncation granularity.
        unit: DateTruncUnit,
        /// Operand expression; must type-check to `Date32` or `Timestamp`.
        expr: Box<Expr>,
    },
    /// Rename an expression in the output schema.
    Alias(Box<Expr>, String),
    /// Uncorrelated scalar subquery: `(SELECT max(y) FROM t2)`.
    ///
    /// The boxed [`LogicalPlan`] is a self-contained query that references no
    /// columns from the enclosing query (correlation is rejected at the SQL
    /// frontend — see [`crate::plan::subquery::reject_if_correlated`]). The
    /// subplan must produce **exactly one output column**; the scalar
    /// subquery's static type is that column's dtype.
    ///
    /// Runtime semantics (single-row contract) are deferred to the physical
    /// layer, which currently rejects any plan carrying a subquery node with
    /// a clear "subqueries are not yet lowered" message. The logical plane
    /// stops at producing a correct, type-checked tree.
    ScalarSubquery(Box<LogicalPlan>),
    /// Uncorrelated `IN` / `NOT IN` subquery: `x IN (SELECT id FROM t2)`.
    ///
    /// `expr` is the probe value evaluated against the enclosing query's
    /// schema. `subquery` is a self-contained, uncorrelated [`LogicalPlan`]
    /// producing **exactly one output column** whose dtype must be
    /// comparable with `expr`'s dtype. `negated` distinguishes `NOT IN` from
    /// `IN`. The expression type-checks to `Bool`.
    ///
    /// As with [`Expr::ScalarSubquery`], execution is deferred — the physical
    /// layer rejects the node for now.
    InSubquery {
        /// Probe value evaluated against the outer query's schema.
        expr: Box<Expr>,
        /// Single-column, uncorrelated subquery supplying the membership set.
        subquery: Box<LogicalPlan>,
        /// `true` for `NOT IN`, `false` for `IN`.
        negated: bool,
    },
}

/// Build a column reference expression.
pub fn col(name: impl Into<String>) -> Expr {
    Expr::Column(name.into())
}

/// Build a literal expression from anything that converts into `Literal`.
pub fn lit<T: Into<Literal>>(v: T) -> Expr {
    Expr::Literal(v.into())
}

fn binary(op: BinaryOp, l: Expr, r: Expr) -> Expr {
    Expr::Binary {
        op,
        left: Box::new(l),
        right: Box::new(r),
    }
}

impl Expr {
    /// Wrap `self` in an `Alias`.
    pub fn alias(self, name: impl Into<String>) -> Expr {
        Expr::Alias(Box::new(self), name.into())
    }

    /// `self + rhs`.
    pub fn add(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Add, self, rhs)
    }

    /// `self - rhs`.
    pub fn sub(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Sub, self, rhs)
    }

    /// `self * rhs`.
    pub fn mul(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Mul, self, rhs)
    }

    /// `self / rhs`.
    pub fn div(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Div, self, rhs)
    }

    /// `self = rhs`.
    pub fn eq(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Eq, self, rhs)
    }

    /// `self <> rhs`.
    pub fn neq(self, rhs: Expr) -> Expr {
        binary(BinaryOp::NotEq, self, rhs)
    }

    /// `self < rhs`.
    pub fn lt(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Lt, self, rhs)
    }

    /// `self <= rhs`.
    pub fn lt_eq(self, rhs: Expr) -> Expr {
        binary(BinaryOp::LtEq, self, rhs)
    }

    /// `self > rhs`.
    pub fn gt(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Gt, self, rhs)
    }

    /// `self >= rhs`.
    pub fn gt_eq(self, rhs: Expr) -> Expr {
        binary(BinaryOp::GtEq, self, rhs)
    }

    /// `self AND rhs`.
    pub fn and(self, rhs: Expr) -> Expr {
        binary(BinaryOp::And, self, rhs)
    }

    /// `self OR rhs`.
    pub fn or(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Or, self, rhs)
    }

    /// `self || rhs` — SQL string concatenation. Both sides must type-check
    /// to `Utf8`; result is `Utf8`. See [`BinaryOp::Concat`].
    pub fn concat(self, rhs: Expr) -> Expr {
        binary(BinaryOp::Concat, self, rhs)
    }

    /// `self IS NULL`. Returns a Bool expression, never null.
    pub fn is_null(self) -> Expr {
        Expr::Unary {
            op: UnaryOp::IsNull,
            operand: Box::new(self),
        }
    }

    /// `self IS NOT NULL`. Returns a Bool expression, never null.
    pub fn is_not_null(self) -> Expr {
        Expr::Unary {
            op: UnaryOp::IsNotNull,
            operand: Box::new(self),
        }
    }

    /// `NOT self`. Operand must type-check to `Bool`; the result is `Bool`.
    pub fn not(self) -> Expr {
        Expr::Unary {
            op: UnaryOp::Not,
            operand: Box::new(self),
        }
    }

    /// `CAST(self AS target)`. Type-checking against the source dtype happens
    /// later in [`Expr::dtype`]; see [`cast_is_supported`] for the accepted
    /// source/target pairs.
    pub fn cast(self, target: DataType) -> Expr {
        Expr::Cast {
            expr: Box::new(self),
            target,
        }
    }

    /// Resolve the static type of this expression against `schema`.
    // TODO(nullable): add COALESCE / NULLIF variants (CASE, IS NULL,
    // IS NOT NULL all landed).
    pub fn dtype(&self, schema: &Schema) -> BoltResult<DataType> {
        self.dtype_depth(schema, 0)
    }

    /// Inner recursion for [`Expr::dtype`]. `depth` is the current recursion
    /// depth; returns Err if [`crate::plan::sql_frontend::MAX_RECURSION_DEPTH`]
    /// is exceeded — guards against attacker-controlled deeply nested
    /// expressions reaching type-checking after construction.
    fn dtype_depth(&self, schema: &Schema, depth: usize) -> BoltResult<DataType> {
        if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
            return Err(BoltError::Type(format!(
                "expression nesting exceeds depth limit ({})",
                crate::plan::sql_frontend::MAX_RECURSION_DEPTH
            )));
        }
        match self {
            Expr::Column(name) => Ok(schema.field(name)?.dtype),
            Expr::Literal(lit) => lit
                .dtype()
                .ok_or_else(|| BoltError::Type("untyped NULL literal".into())),
            Expr::Binary { op, left, right } => {
                // NULL-peer typing: an untyped `Literal::Null` opposite a
                // typed peer takes that peer's dtype for the purposes of
                // type-checking the binary expression. The peer-typed
                // helper calls back into `dtype` (which starts a fresh
                // depth budget); that's fine since the parent depth check
                // has already bounded the enclosing recursion.
                let l = peer_typed_dtype(left, right, schema, *op)?;
                let r = peer_typed_dtype(right, left, schema, *op)?;
                let _ = depth; // depth threading enforced at function entry
                if op.is_arithmetic() {
                    // v0.7: Date32/Timestamp subtraction lowers to integer sub
                    // on the underlying days/ticks. Only Sub is wired (Date - Date
                    // yields a day count; Timestamp - Timestamp yields a tick
                    // count in the source unit). Mixing two Timestamps with
                    // different TimeUnits or non-matching tz literals is
                    // intentionally out of scope; rejected with a tighter
                    // message so the user knows what to do.
                    if let Some(dt) = date_or_timestamp_arith_result(*op, l, r) {
                        return dt;
                    }
                    // v0.7 sub-task B: Decimal128 arithmetic at the logical
                    // plane returns the SQL-convention result dtype directly;
                    // the physical-plan codegen mirrors the rule via
                    // `decimal128_arith_result_dtype`. Mixed Decimal / non-
                    // Decimal arithmetic isn't wired yet (it would require an
                    // implicit promotion path); reject with a tight message
                    // so the user sees the v0.7 envelope.
                    if let Some(dt) = decimal128_arith_result(*op, l, r) {
                        return dt;
                    }
                    if !l.is_numeric() || !r.is_numeric() {
                        return Err(BoltError::Type(format!(
                            "arithmetic {op:?} requires numeric operands, got {l:?} and {r:?}"
                        )));
                    }
                    unify_numeric(l, r)
                } else if op.is_comparison() {
                    if l == r {
                        Ok(DataType::Bool)
                    } else if l.is_numeric() && r.is_numeric() {
                        // Allow numeric cross-comparisons; result is still Bool.
                        let _ = unify_numeric(l, r)?;
                        Ok(DataType::Bool)
                    } else {
                        Err(BoltError::Type(format!(
                            "cannot compare {l:?} with {r:?}"
                        )))
                    }
                } else if op.is_logical() {
                    if l == DataType::Bool && r == DataType::Bool {
                        Ok(DataType::Bool)
                    } else {
                        Err(BoltError::Type(format!(
                            "logical {op:?} requires Bool operands, got {l:?} and {r:?}"
                        )))
                    }
                } else if op.is_string() {
                    // String concat (`||`): both operands must be Utf8.
                    // Result is always Utf8. NULL-peer-typing has already
                    // run above, so a bare NULL opposite a Utf8 peer is
                    // typed as Utf8 here and accepted.
                    if l == DataType::Utf8 && r == DataType::Utf8 {
                        Ok(DataType::Utf8)
                    } else {
                        Err(BoltError::Type(format!(
                            "string {op:?} requires Utf8 operands, got {l:?} and {r:?}"
                        )))
                    }
                } else {
                    Err(BoltError::Type(format!("unsupported operator {op:?}")))
                }
            }
            Expr::Unary { op, operand } => match op {
                // IS NULL / IS NOT NULL always produce Bool, regardless of
                // operand dtype. We still resolve the operand's dtype when
                // it's resolvable (catches typos like `nonexistent IS NULL`),
                // but tolerate an untyped `Literal::Null` operand — that's
                // exactly the case this surface exists to support.
                UnaryOp::IsNull | UnaryOp::IsNotNull => {
                    if !matches!(operand.as_ref(), Expr::Literal(Literal::Null)) {
                        let _ = operand.dtype_depth(schema, depth + 1)?;
                    }
                    Ok(DataType::Bool)
                }
                // NOT requires a Bool operand; the result is Bool. An untyped
                // `Literal::Null` is accepted under the same NULL-peer-typing
                // spirit as `Binary` ops — `NOT NULL` is a Bool-typed NULL
                // value, not a type error.
                UnaryOp::Not => {
                    if matches!(operand.as_ref(), Expr::Literal(Literal::Null)) {
                        return Ok(DataType::Bool);
                    }
                    let t = operand.dtype_depth(schema, depth + 1)?;
                    if t != DataType::Bool {
                        return Err(BoltError::Type(format!(
                            "logical NOT requires a Bool operand, got {t:?}"
                        )));
                    }
                    Ok(DataType::Bool)
                }
            },
            Expr::Case {
                branches,
                else_branch,
            } => {
                if branches.is_empty() {
                    return Err(BoltError::Type(
                        "CASE expression requires at least one WHEN/THEN branch".into(),
                    ));
                }
                for (i, (when, _)) in branches.iter().enumerate() {
                    let wt = when.dtype_depth(schema, depth + 1)?;
                    if wt != DataType::Bool {
                        return Err(BoltError::Type(format!(
                            "CASE WHEN condition {i} must be Bool, got {wt:?}"
                        )));
                    }
                }
                let mut arms: Vec<(String, Option<DataType>)> =
                    Vec::with_capacity(branches.len() + 1);
                for (i, (_, then)) in branches.iter().enumerate() {
                    let t = case_arm_dtype(then, schema, depth + 1)?;
                    arms.push((format!("THEN {i}"), t));
                }
                if let Some(e) = else_branch {
                    let t = case_arm_dtype(e, schema, depth + 1)?;
                    arms.push(("ELSE".to_string(), t));
                }
                let mut acc: Option<DataType> = None;
                for (label, t) in &arms {
                    match t {
                        Some(t) => match acc {
                            None => acc = Some(*t),
                            Some(prev) => {
                                acc = Some(unify_case_dtypes(prev, *t).ok_or_else(|| {
                                    BoltError::Type(format!(
                                        "CASE {label} has incompatible dtype {t:?} \
                                         with previous arms ({prev:?})"
                                    ))
                                })?);
                            }
                        },
                        None => {}
                    }
                }
                acc.ok_or_else(|| {
                    BoltError::Type(
                        "CASE expression: every THEN/ELSE arm is an untyped NULL — \
                         cannot infer a result dtype"
                            .into(),
                    )
                })
            }
            Expr::Like { expr, .. } => {
                let t = expr.dtype_depth(schema, depth + 1)?;
                if t != DataType::Utf8 {
                    return Err(BoltError::Type(format!(
                        "LIKE requires a Utf8 operand, got {t:?}"
                    )));
                }
                Ok(DataType::Bool)
            }
            Expr::Cast { expr, target } => {
                // Tolerate an untyped `Literal::Null` operand the same way
                // `IS NULL` does — `CAST(NULL AS Int32)` is a legitimate
                // SQL fragment and the result type is purely declared by
                // `target` anyway.
                if matches!(expr.as_ref(), Expr::Literal(Literal::Null)) {
                    return Ok(*target);
                }
                let src = expr.dtype_depth(schema, depth + 1)?;
                if !cast_is_supported(src, *target) {
                    return Err(BoltError::Type(format!(
                        "unsupported CAST from {src:?} to {target:?}"
                    )));
                }
                Ok(*target)
            }
            Expr::ScalarFn { kind, args } => scalar_fn_dtype(*kind, args, schema, depth + 1),
            Expr::Extract { field, expr } => {
                let src = expr.dtype_depth(schema, depth + 1)?;
                extract_output_dtype(*field, src)
            }
            Expr::DateTrunc { unit, expr } => {
                let src = expr.dtype_depth(schema, depth + 1)?;
                date_trunc_output_dtype(*unit, src)
            }
            Expr::Alias(inner, _) => inner.dtype_depth(schema, depth + 1),
            // A scalar subquery's type is the (single) output column dtype of
            // its self-contained plan. The plan is type-checked here so a
            // malformed subquery surfaces at the enclosing query's type-check
            // rather than being silently deferred. The enclosing `schema` is
            // intentionally NOT consulted: the subquery is uncorrelated, so
            // its schema derives solely from its own FROM tree.
            Expr::ScalarSubquery(plan) => {
                let s = plan.schema_depth(depth + 1)?;
                if s.fields.len() != 1 {
                    return Err(BoltError::Type(format!(
                        "scalar subquery must return exactly one column, got {}",
                        s.fields.len()
                    )));
                }
                Ok(s.fields[0].dtype)
            }
            // `x [NOT] IN (subquery)` is always Bool. We resolve the probe
            // expression against the enclosing schema and the subquery's
            // single output column against its own schema, then require the
            // two dtypes to be comparable (same type, or both numeric — the
            // same rule `Expr::Binary` comparisons use). An untyped
            // `Literal::Null` probe is tolerated (its membership is NULL at
            // runtime, but that is an execution concern).
            Expr::InSubquery {
                expr,
                subquery,
                negated: _,
            } => {
                let sub_schema = subquery.schema_depth(depth + 1)?;
                if sub_schema.fields.len() != 1 {
                    return Err(BoltError::Type(format!(
                        "IN subquery must return exactly one column, got {}",
                        sub_schema.fields.len()
                    )));
                }
                let sub_dtype = sub_schema.fields[0].dtype;
                // Tolerate a bare NULL probe (no static type to compare).
                if !matches!(expr.as_ref(), Expr::Literal(Literal::Null)) {
                    let probe = expr.dtype_depth(schema, depth + 1)?;
                    let comparable = probe == sub_dtype
                        || (probe.is_numeric() && sub_dtype.is_numeric());
                    if !comparable {
                        return Err(BoltError::Type(format!(
                            "IN subquery: cannot compare {probe:?} with subquery \
                             column type {sub_dtype:?}"
                        )));
                    }
                }
                Ok(DataType::Bool)
            }
        }
    }
}

/// Resolve a single CASE arm's dtype against `schema`, returning `None` if
/// the arm is an untyped `Literal::Null` (which carries no static type).
/// Any other resolvable arm returns `Some(dtype)`; resolution errors bubble
/// through verbatim (e.g. unknown column references).
fn case_arm_dtype(
    arm: &Expr,
    schema: &Schema,
    depth: usize,
) -> BoltResult<Option<DataType>> {
    if matches!(arm, Expr::Literal(Literal::Null)) {
        return Ok(None);
    }
    Ok(Some(arm.dtype_depth(schema, depth)?))
}

/// Unify two CASE arm dtypes. Numeric pairs go through [`unify_numeric`];
/// non-numeric arms must match exactly. Returns `None` on incompatibility
/// so the caller can build a friendly per-arm error message.
fn unify_case_dtypes(a: DataType, b: DataType) -> Option<DataType> {
    if a == b {
        return Some(a);
    }
    if a.is_numeric() && b.is_numeric() {
        unify_numeric(a, b).ok()
    } else {
        None
    }
}

/// Type-check `EXTRACT(field FROM src)` and return its output dtype.
///
/// Every `EXTRACT` produces `Int64` (an exact numeric per the SQL standard).
/// The operand must be `Date32` or `Timestamp`; intra-day fields (`HOUR` /
/// `MINUTE` / `SECOND`) additionally require a `Timestamp` because a bare
/// `Date32` has no time-of-day component.
pub fn extract_output_dtype(field: DateField, src: DataType) -> BoltResult<DataType> {
    match src {
        DataType::Date32 => {
            if field.is_intraday() {
                return Err(BoltError::Type(format!(
                    "EXTRACT({} FROM <Date32>) is undefined — a Date32 has no \
                     time-of-day component; cast to Timestamp first",
                    field.sql_name()
                )));
            }
            Ok(DataType::Int64)
        }
        DataType::Timestamp(_, _) => Ok(DataType::Int64),
        other => Err(BoltError::Type(format!(
            "EXTRACT({} FROM ...) requires a Date32 or Timestamp operand, got {:?}",
            field.sql_name(),
            other
        ))),
    }
}

/// Type-check `DATE_TRUNC(unit, src)` and return its output dtype.
///
/// `DATE_TRUNC` preserves the operand dtype (`Date32` → `Date32`, `Timestamp`
/// → the same `Timestamp` type). Sub-day units (`hour` / `minute` / `second`)
/// require a `Timestamp` operand.
pub fn date_trunc_output_dtype(unit: DateTruncUnit, src: DataType) -> BoltResult<DataType> {
    match src {
        DataType::Date32 => {
            if unit.is_intraday() {
                return Err(BoltError::Type(format!(
                    "DATE_TRUNC('{}', <Date32>) is undefined — a Date32 has no \
                     sub-day component; cast to Timestamp first",
                    unit.sql_name()
                )));
            }
            Ok(DataType::Date32)
        }
        DataType::Timestamp(tu, tz) => Ok(DataType::Timestamp(tu, tz)),
        other => Err(BoltError::Type(format!(
            "DATE_TRUNC('{}', ...) requires a Date32 or Timestamp operand, got {:?}",
            unit.sql_name(),
            other
        ))),
    }
}

/// Type-check a scalar string function call against `schema` and return
/// its output dtype. Centralised here so the rule lives in one place; the
/// SQL frontend builds `Expr::ScalarFn` without inspecting argument types
/// itself and lets this helper surface the type error.
///
/// `depth` is the current recursion depth at which the arguments will be
/// resolved (the caller has already incremented for entering the ScalarFn
/// node itself). Errors surface as `BoltError::Type`.
fn scalar_fn_dtype(
    kind: ScalarFnKind,
    args: &[Expr],
    schema: &Schema,
    depth: usize,
) -> BoltResult<DataType> {
    // Resolve every argument's dtype up front so column-typo errors fire
    // before we apply per-kind shape checks.
    let mut arg_types: Vec<DataType> = Vec::with_capacity(args.len());
    for a in args {
        arg_types.push(a.dtype_depth(schema, depth)?);
    }
    let name = kind.sql_name();
    match kind {
        ScalarFnKind::Upper | ScalarFnKind::Lower => {
            if arg_types.len() != 1 {
                return Err(BoltError::Type(format!(
                    "{name} expects exactly 1 argument, got {}",
                    arg_types.len()
                )));
            }
            if arg_types[0] != DataType::Utf8 {
                return Err(BoltError::Type(format!(
                    "{name} requires a Utf8 argument, got {:?}",
                    arg_types[0]
                )));
            }
            Ok(DataType::Utf8)
        }
        ScalarFnKind::Length => {
            if arg_types.len() != 1 {
                return Err(BoltError::Type(format!(
                    "{name} expects exactly 1 argument, got {}",
                    arg_types.len()
                )));
            }
            if arg_types[0] != DataType::Utf8 {
                return Err(BoltError::Type(format!(
                    "{name} requires a Utf8 argument, got {:?}",
                    arg_types[0]
                )));
            }
            Ok(DataType::Int64)
        }
        ScalarFnKind::Substring => {
            if arg_types.len() != 2 && arg_types.len() != 3 {
                return Err(BoltError::Type(format!(
                    "{name} expects 2 or 3 arguments (string, start [, length]), got {}",
                    arg_types.len()
                )));
            }
            if arg_types[0] != DataType::Utf8 {
                return Err(BoltError::Type(format!(
                    "{name} first argument must be Utf8, got {:?}",
                    arg_types[0]
                )));
            }
            for (i, t) in arg_types.iter().enumerate().skip(1) {
                if *t != DataType::Int64 {
                    return Err(BoltError::Type(format!(
                        "{name} argument {} must be Int64, got {:?}",
                        i + 1,
                        t
                    )));
                }
            }
            Ok(DataType::Utf8)
        }
        ScalarFnKind::Concat => {
            if arg_types.len() < 2 {
                return Err(BoltError::Type(format!(
                    "{name} expects at least 2 arguments, got {}",
                    arg_types.len()
                )));
            }
            for (i, t) in arg_types.iter().enumerate() {
                if *t != DataType::Utf8 {
                    return Err(BoltError::Type(format!(
                        "{name} argument {} must be Utf8, got {:?}",
                        i + 1,
                        t
                    )));
                }
            }
            Ok(DataType::Utf8)
        }
        ScalarFnKind::TrimBoth | ScalarFnKind::TrimLeading | ScalarFnKind::TrimTrailing => {
            // TRIM(s) or TRIM([side] chars FROM s): 1 or 2 Utf8 args -> Utf8.
            if arg_types.len() != 1 && arg_types.len() != 2 {
                return Err(BoltError::Type(format!(
                    "{name} expects 1 or 2 Utf8 arguments (string [, trim chars]), got {}",
                    arg_types.len()
                )));
            }
            for (i, t) in arg_types.iter().enumerate() {
                if *t != DataType::Utf8 {
                    return Err(BoltError::Type(format!(
                        "{name} argument {} must be Utf8, got {:?}",
                        i + 1,
                        t
                    )));
                }
            }
            Ok(DataType::Utf8)
        }
        ScalarFnKind::OctetLength => {
            // OCTET_LENGTH(s): 1 Utf8 arg -> Int64 (byte length).
            if arg_types.len() != 1 {
                return Err(BoltError::Type(format!(
                    "{name} expects exactly 1 argument, got {}",
                    arg_types.len()
                )));
            }
            if arg_types[0] != DataType::Utf8 {
                return Err(BoltError::Type(format!(
                    "{name} requires a Utf8 argument, got {:?}",
                    arg_types[0]
                )));
            }
            Ok(DataType::Int64)
        }
        ScalarFnKind::Position => {
            // POSITION(substr IN s) / STRPOS(s, substr): 2 Utf8 args -> Int64.
            if arg_types.len() != 2 {
                return Err(BoltError::Type(format!(
                    "{name} expects exactly 2 arguments (string, substring), got {}",
                    arg_types.len()
                )));
            }
            for (i, t) in arg_types.iter().enumerate() {
                if *t != DataType::Utf8 {
                    return Err(BoltError::Type(format!(
                        "{name} argument {} must be Utf8, got {:?}",
                        i + 1,
                        t
                    )));
                }
            }
            Ok(DataType::Int64)
        }
        ScalarFnKind::Replace => {
            // REPLACE(s, from, to): 3 Utf8 args -> Utf8.
            if arg_types.len() != 3 {
                return Err(BoltError::Type(format!(
                    "{name} expects exactly 3 arguments (string, from, to), got {}",
                    arg_types.len()
                )));
            }
            for (i, t) in arg_types.iter().enumerate() {
                if *t != DataType::Utf8 {
                    return Err(BoltError::Type(format!(
                        "{name} argument {} must be Utf8, got {:?}",
                        i + 1,
                        t
                    )));
                }
            }
            Ok(DataType::Utf8)
        }
        ScalarFnKind::Left | ScalarFnKind::Right => {
            // LEFT/RIGHT(s, n): first arg Utf8, second Int64 -> Utf8.
            if arg_types.len() != 2 {
                return Err(BoltError::Type(format!(
                    "{name} expects exactly 2 arguments (string, count), got {}",
                    arg_types.len()
                )));
            }
            if arg_types[0] != DataType::Utf8 {
                return Err(BoltError::Type(format!(
                    "{name} first argument must be Utf8, got {:?}",
                    arg_types[0]
                )));
            }
            if arg_types[1] != DataType::Int64 {
                return Err(BoltError::Type(format!(
                    "{name} second argument must be Int64, got {:?}",
                    arg_types[1]
                )));
            }
            Ok(DataType::Utf8)
        }
        ScalarFnKind::Lpad | ScalarFnKind::Rpad => {
            // LPAD/RPAD(s, len, pad): Utf8, Int64, Utf8 -> Utf8.
            if arg_types.len() != 3 {
                return Err(BoltError::Type(format!(
                    "{name} expects exactly 3 arguments (string, length, pad), got {}",
                    arg_types.len()
                )));
            }
            if arg_types[0] != DataType::Utf8 {
                return Err(BoltError::Type(format!(
                    "{name} first argument must be Utf8, got {:?}",
                    arg_types[0]
                )));
            }
            if arg_types[1] != DataType::Int64 {
                return Err(BoltError::Type(format!(
                    "{name} second argument (length) must be Int64, got {:?}",
                    arg_types[1]
                )));
            }
            if arg_types[2] != DataType::Utf8 {
                return Err(BoltError::Type(format!(
                    "{name} third argument (pad) must be Utf8, got {:?}",
                    arg_types[2]
                )));
            }
            Ok(DataType::Utf8)
        }
        ScalarFnKind::Reverse | ScalarFnKind::Initcap => {
            // REVERSE(s) / INITCAP(s): 1 Utf8 arg -> Utf8.
            if arg_types.len() != 1 {
                return Err(BoltError::Type(format!(
                    "{name} expects exactly 1 argument, got {}",
                    arg_types.len()
                )));
            }
            if arg_types[0] != DataType::Utf8 {
                return Err(BoltError::Type(format!(
                    "{name} requires a Utf8 argument, got {:?}",
                    arg_types[0]
                )));
            }
            Ok(DataType::Utf8)
        }
    }
}

/// Resolve `e`'s dtype against `schema`, but if `e` is `Literal::Null` and
/// `peer` resolves to a typed expression, return the peer's dtype instead.
///
/// This is the NULL-peer-typing rule used by `Expr::Binary` dtype resolution
/// so that SQL fragments like `WHERE x = NULL` or `SELECT x + NULL` don't
/// hard-error at type-check time. The rule applies to every BinaryOp:
/// arithmetic, comparison, and logical (where the typed peer is necessarily
/// Bool, so NULL becomes Bool). Two NULLs on both sides still surface the
/// original `Type("untyped NULL literal")` error — there is no peer to
/// borrow a type from.
fn peer_typed_dtype(
    e: &Expr,
    peer: &Expr,
    schema: &Schema,
    _op: BinaryOp,
) -> BoltResult<DataType> {
    if matches!(e, Expr::Literal(Literal::Null)) {
        // Try to borrow the peer's dtype. If the peer itself is also a
        // bare untyped NULL the recursive call will fail with the original
        // "untyped NULL literal" error, which is what we want.
        if let Ok(t) = peer.dtype(schema) {
            return Ok(t);
        }
    }
    e.dtype(schema)
}

/// True if the engine's logical plane accepts `CAST(<src> AS <target>)`.
///
/// The v0.5 surface is intentionally small — only the primitive conversions
/// the GPU executor will plausibly lower in v0.6 are admitted. Anything
/// involving `Utf8` is rejected at this layer (string -> numeric parsing
/// would need a runtime fallible path we don't have yet).
///
/// Accepted pairs:
///   * `T -> T` (identity, no-op) for any primitive `T`
///   * `Int32 <-> Int64`
///   * `Int32` / `Int64` -> `Float32` / `Float64`
///   * `Float32 <-> Float64`
///   * `Bool <-> Int32` / `Bool <-> Int64`
///
/// Everything else returns `false` and surfaces a `BoltError::Type` at the
/// caller in [`Expr::dtype_depth`].
pub(crate) fn cast_is_supported(src: DataType, target: DataType) -> bool {
    use DataType::*;
    if src == target {
        // Identity cast is always allowed (covers Utf8 -> Utf8 too, which
        // is harmless — the executor would just return the column as-is).
        return true;
    }
    match (src, target) {
        // Integer widening / narrowing.
        (Int32, Int64) | (Int64, Int32) => true,
        // Integer -> float.
        (Int32, Float32) | (Int32, Float64) | (Int64, Float32) | (Int64, Float64) => true,
        // Float widening / narrowing.
        (Float32, Float64) | (Float64, Float32) => true,
        // Bool <-> integer (0/1 round-trip).
        (Bool, Int32) | (Bool, Int64) | (Int32, Bool) | (Int64, Bool) => true,
        _ => false,
    }
}

/// v0.7: result dtype for an arithmetic op on Date32 / Timestamp operands.
///
/// Returns `Some(Ok(dtype))` if the op is in scope:
///   * `Date32 - Date32`              → `Int32` (number of days)
///   * `Timestamp(u, tz) - Timestamp(u, tz)` → `Int64` (ticks in the source unit)
///
/// Returns `Some(Err(_))` with a clear message for out-of-scope cases that
/// touch Date32 / Timestamp (e.g. `Date32 + Date32`, mixed `TimeUnit`s, or
/// non-matching tz literals on a subtraction).
///
/// Returns `None` if neither operand is Date32 / Timestamp, letting the
/// caller fall through to the standard numeric arithmetic path.
///
/// INTERVAL-based arithmetic (`Date + INTERVAL n DAY`) is intentionally not
/// recognised here: the SQL frontend does not yet parse INTERVAL into an
/// `Expr::Literal`, so there is no in-tree producer for the typed expression.
///
/// SINGLE SOURCE OF TRUTH for temporal arithmetic typing. The physical
/// plane's `temporal_arith_result_dtype` is a thin wrapper that delegates
/// here and re-shapes `Option<Result<_>>` into `Result<Option<_>>`; do not
/// re-derive the rule there. `pub(crate)` so physical_plan can call it.
pub(crate) fn date_or_timestamp_arith_result(
    op: BinaryOp,
    l: DataType,
    r: DataType,
) -> Option<BoltResult<DataType>> {
    use DataType::*;
    // Fast path: neither operand involves a temporal type.
    let l_is_temporal = matches!(l, Date32 | Timestamp(_, _));
    let r_is_temporal = matches!(r, Date32 | Timestamp(_, _));
    if !l_is_temporal && !r_is_temporal {
        return None;
    }
    match (op, l, r) {
        (BinaryOp::Sub, Date32, Date32) => Some(Ok(Int32)),
        (BinaryOp::Sub, Timestamp(lu, ltz), Timestamp(ru, rtz)) => {
            if lu != ru {
                return Some(Err(BoltError::Type(format!(
                    "Timestamp subtraction requires matching TimeUnit, \
                     got {lu:?} and {ru:?}"
                ))));
            }
            if ltz != rtz {
                return Some(Err(BoltError::Type(format!(
                    "Timestamp subtraction requires matching time zones, \
                     got {ltz:?} and {rtz:?}"
                ))));
            }
            Some(Ok(Int64))
        }
        // Catch-all: any other Date/Timestamp arithmetic shape is rejected
        // here with a tight message rather than falling through to the
        // generic "requires numeric operands" rejection.
        _ => Some(Err(BoltError::Type(format!(
            "arithmetic {op:?} on Date/Timestamp operands ({l:?}, {r:?}) is not \
             supported; only Date32 - Date32 and Timestamp - Timestamp \
             (matching unit and tz) are wired in v0.7"
        )))),
    }
}

/// v0.7 sub-task B: result dtype for `Decimal128(p1, s1) op Decimal128(p2, s2)`
/// arithmetic at the logical plane. Mirrors the physical-plane helper
/// `physical_plan::decimal128_arith_result_dtype` so a SELECT that
/// references a Decimal arithmetic expression resolves to the same dtype
/// whether the planner asks the logical or the physical layer for it.
///
/// Returns `None` if neither operand is Decimal128 (let the caller fall
/// through to the regular numeric path). Returns `Some(Err)` for the
/// "either side is Decimal128 but the shape isn't supported" cases
/// (Decimal vs non-Decimal, scale mismatch on Add/Sub, precision
/// overflow on the result, etc.).
///
/// SINGLE SOURCE OF TRUTH for Decimal128 arithmetic typing. The physical
/// plane's `decimal128_arith_result_dtype` is a thin wrapper that delegates
/// here (passing pre-extracted `(p, s)` pairs as `Decimal128` dtypes); do
/// not re-derive the promotion/overflow rule there. `pub(crate)` so
/// physical_plan can call it.
pub(crate) fn decimal128_arith_result(
    op: BinaryOp,
    l: DataType,
    r: DataType,
) -> Option<BoltResult<DataType>> {
    use DataType::*;
    let l_dec = matches!(l, Decimal128(_, _));
    let r_dec = matches!(r, Decimal128(_, _));
    if !l_dec && !r_dec {
        return None;
    }
    let (p1, s1) = match l {
        Decimal128(p, s) => (p, s),
        other => {
            return Some(Err(BoltError::Type(format!(
                "arithmetic {op:?} on mixed Decimal128 / {other:?} is not yet \
                 supported; CAST to a common Decimal128(p, s) first"
            ))));
        }
    };
    let (p2, s2) = match r {
        Decimal128(p, s) => (p, s),
        other => {
            return Some(Err(BoltError::Type(format!(
                "arithmetic {op:?} on mixed Decimal128 / {other:?} is not yet \
                 supported; CAST to a common Decimal128(p, s) first"
            ))));
        }
    };
    // 38 = Arrow Decimal128 max precision.
    const MAX_P: u8 = 38;
    match op {
        BinaryOp::Add | BinaryOp::Sub => {
            if s1 != s2 {
                return Some(Err(BoltError::Type(format!(
                    "Decimal128 {op:?} requires matching scale, \
                     got Decimal128({p1}, {s1}) and Decimal128({p2}, {s2})"
                ))));
            }
            let p = p1.max(p2);
            let new_p = match p.checked_add(1) {
                Some(v) if v <= MAX_P => v,
                _ => {
                    return Some(Err(BoltError::Type(format!(
                        "Decimal128 {op:?} result precision exceeds {MAX_P} \
                         (max({p1}, {p2}) + 1)"
                    ))));
                }
            };
            Some(Ok(Decimal128(new_p, s1)))
        }
        BinaryOp::Mul => {
            let new_p = match p1.checked_add(p2) {
                Some(v) if v <= MAX_P => v,
                _ => {
                    return Some(Err(BoltError::Type(format!(
                        "Decimal128 Mul result precision {p1} + {p2} exceeds {MAX_P}"
                    ))));
                }
            };
            let new_s = match s1.checked_add(s2) {
                Some(v) => v,
                None => {
                    return Some(Err(BoltError::Type(format!(
                        "Decimal128 Mul scale overflow: {s1} + {s2} does not fit in i8"
                    ))));
                }
            };
            Some(Ok(Decimal128(new_p, new_s)))
        }
        BinaryOp::Div => Some(Err(BoltError::Type(
            "Decimal128 Div not yet lowered to GPU; only Add/Sub/Mul are wired in v0.7"
                .into(),
        ))),
        other => Some(Err(BoltError::Type(format!(
            "arithmetic {other:?} on Decimal128 operands is not supported in v0.7"
        )))),
    }
}

/// Promote two numeric types to the wider one (float beats int, 64 beats 32).
///
/// SINGLE SOURCE OF TRUTH for numeric type promotion. The physical plane's
/// `unify_numeric` is a thin wrapper that delegates here for every numeric
/// pair (and only keeps a local `a == b` short-circuit so that already-equal
/// non-numeric dtypes — e.g. `Utf8`/`Bool` arms reachable from its codegen
/// call sites — round-trip unchanged). `pub(crate)` so physical_plan can
/// call it.
pub(crate) fn unify_numeric(a: DataType, b: DataType) -> BoltResult<DataType> {
    use DataType::*;
    if !a.is_numeric() || !b.is_numeric() {
        return Err(BoltError::Type(format!(
            "cannot unify non-numeric types {a:?} and {b:?}"
        )));
    }
    let either_float = a.is_float() || b.is_float();
    let either_64 = matches!(a, Int64 | Float64) || matches!(b, Int64 | Float64);
    Ok(match (either_float, either_64) {
        (true, true) => Float64,
        (true, false) => Float32,
        (false, true) => Int64,
        (false, false) => Int32,
    })
}

/// Aggregate function applied over an expression.
#[derive(Debug, Clone)]
pub enum AggregateExpr {
    /// `COUNT(expr)` — output `Int64`.
    Count(Expr),
    /// `SUM(expr)` — output preserves input dtype.
    Sum(Expr),
    /// `MIN(expr)` — output preserves input dtype.
    Min(Expr),
    /// `MAX(expr)` — output preserves input dtype.
    Max(Expr),
    /// `AVG(expr)` — output `Float64`.
    Avg(Expr),
    /// `VAR_POP(expr)` — population variance, output `Float64`.
    ///
    /// Computed via Welford's online algorithm at scalar-aggregate level
    /// (M2 / count, returning NULL for empty input). The scalar path lives
    /// in `crate::exec::aggregate`; the GROUP BY path is intentionally
    /// rejected with a clear error in v0.5.
    VarPop(Box<Expr>),
    /// `VAR_SAMP(expr)` — sample variance (`VARIANCE` / `VAR_SAMP` per SQL
    /// standard), output `Float64`. Returns NULL when count <= 1.
    VarSamp(Box<Expr>),
    /// `STDDEV_POP(expr)` — population standard deviation; output `Float64`.
    /// Computed via Welford's one-pass algorithm on the host (see
    /// [`crate::exec::welford`]); GPU offload is a v0.6 stretch goal.
    /// Returns `0.0` (not NULL) when no rows are aggregated, mirroring the
    /// existing AVG convention for an empty/all-NULL input.
    StddevPop(Box<Expr>),
    /// `STDDEV_SAMP(expr)` — sample standard deviation; output `Float64`.
    /// Computed via Welford's one-pass algorithm (shared state with
    /// `STDDEV_POP`). Returns SQL NULL when `count <= 1` (the divisor
    /// `count - 1` is zero or negative — undefined per SQL standard); also
    /// returns NULL when the input is empty / all-NULL.
    StddevSamp(Box<Expr>),
}

impl AggregateExpr {
    /// Default output column name.
    ///
    /// Authoritative naming rule for aggregate output columns. Called from
    /// `LogicalPlan::schema()` (this file) and re-exported via the
    /// free function [`aggregate_output_name`] which is consumed by
    /// `sql_frontend.rs::plan_select` (SELECT-list re-projection) and
    /// `sql_frontend.rs::lower_expr_in_having` (HAVING rewriter). Do not
    /// duplicate the rule at the call sites; route through this method
    /// (or the free function) instead.
    pub(crate) fn output_name(&self) -> String {
        match self {
            AggregateExpr::Count(e) => format!("count{}", suffix(e)),
            AggregateExpr::Sum(e) => format!("sum{}", suffix(e)),
            AggregateExpr::Min(e) => format!("min{}", suffix(e)),
            AggregateExpr::Max(e) => format!("max{}", suffix(e)),
            AggregateExpr::Avg(e) => format!("avg{}", suffix(e)),
            AggregateExpr::VarPop(e) => format!("var_pop{}", suffix(e)),
            AggregateExpr::VarSamp(e) => format!("var_samp{}", suffix(e)),
            AggregateExpr::StddevPop(e) => format!("stddev_pop{}", suffix(e)),
            AggregateExpr::StddevSamp(e) => format!("stddev_samp{}", suffix(e)),
        }
    }

    /// Output dtype of the aggregate against the input schema.
    ///
    /// `SUM` widens narrow integer inputs to the corresponding 64-bit type
    /// to prevent silent overflow under typical workloads (`SUM(Int32)` over
    /// more than ~2^31 small values would otherwise wrap). Float inputs and
    /// `Int64`/`UInt64` inputs are not widened (no wider primitive type is
    /// available); callers must be aware of overflow risk on extreme inputs.
    ///
    /// This widening contract is mirrored by the GPU-side accumulator in
    /// `crate::jit::agg_kernels` and the host-side scalar-aggregate path in
    /// `crate::exec::aggregate`; keep all three in sync.
    fn output_dtype(&self, input: &Schema) -> BoltResult<DataType> {
        match self {
            AggregateExpr::Count(_) => Ok(DataType::Int64),
            AggregateExpr::Sum(e) => Ok(sum_output_dtype(e.dtype(input)?)),
            AggregateExpr::Min(e) | AggregateExpr::Max(e) => e.dtype(input),
            AggregateExpr::Avg(e) => {
                // v0.7: AVG over Date / Timestamp is non-standard SQL — the
                // "mean of two dates" is not well-defined in the SQL spec
                // (you can't average two calendar instants). Reject at the
                // logical layer with a clear message rather than silently
                // producing a Float64 from the underlying days/ticks.
                let dt = e.dtype(input)?;
                if matches!(dt, DataType::Date32 | DataType::Timestamp(_, _)) {
                    return Err(BoltError::Type(format!(
                        "AVG over Date/Timestamp is non-standard SQL (got {dt:?})"
                    )));
                }
                Ok(DataType::Float64)
            }
            AggregateExpr::VarPop(e) | AggregateExpr::VarSamp(e) => {
                let _ = e.dtype(input)?;
                Ok(DataType::Float64)
            }
            AggregateExpr::StddevPop(e) | AggregateExpr::StddevSamp(e) => {
                let dt = e.dtype(input)?;
                if !matches!(
                    dt,
                    DataType::Int32 | DataType::Int64 | DataType::Float32 | DataType::Float64
                ) {
                    return Err(BoltError::Type(format!(
                        "STDDEV requires a numeric operand, got {dt:?}"
                    )));
                }
                Ok(DataType::Float64)
            }
        }
    }
}

/// Widen the input dtype of a `SUM` aggregate to its accumulator dtype.
///
/// Mirrors the widening contract documented on `AggregateExpr::output_dtype`:
/// narrow signed integers (currently only `Int32` in the supported `DataType`
/// set) widen to `Int64`; `Int64` and the float types are unchanged. This
/// helper is the single source of truth for the SUM widening rule and is also
/// consumed by `crate::jit::agg_kernels` (kernel emission must agree with the
/// plan's declared output type) and `crate::exec::aggregate` (accumulator
/// allocation and Arrow array packing).
pub fn sum_output_dtype(input: DataType) -> DataType {
    match input {
        // Narrow signed integer → widen to Int64.
        DataType::Int32 => DataType::Int64,
        // Already 64-bit-wide or float: unchanged (no wider primitive in this
        // engine's `DataType`). Overflow risk on Int64 is acknowledged at the
        // API boundary.
        DataType::Int64 | DataType::Float32 | DataType::Float64 => input,
        // v0.7: `SUM(Decimal128(p, s))` widens precision to the Arrow
        // Decimal128 maximum (38 digits) and keeps the same scale. The
        // host-side accumulator in `crate::exec::aggregate` (see
        // `sum_decimal128_from_batch`) folds rows into an `i128` with a
        // checked add — overflow on the i128 representation surfaces as
        // a clear `BoltError::Type` rather than wrapping silently, so
        // packing the result at the widest declared precision (38) is
        // sound: the bit pattern is guaranteed to fit. Scale is preserved
        // because SUM over a fixed-point input does not change the
        // location of the decimal point.
        DataType::Decimal128(_, s) => DataType::Decimal128(38, s),
        // Other non-numeric types fall through unchanged; the downstream
        // typecheck (e.g. `ReduceOp::identity_ptx`) will reject the
        // aggregate.
        DataType::Bool
        | DataType::Utf8
        | DataType::Date32
        | DataType::Timestamp(_, _) => input,
    }
}

/// `_colname` for a bare column ref, empty otherwise.
fn suffix(e: &Expr) -> String {
    match e {
        Expr::Column(n) => format!("_{n}"),
        Expr::Alias(_, n) => format!("_{n}"),
        _ => String::new(),
    }
}

/// A window function applied over a partition / ordering.
///
/// Two families are supported (host-side only — see
/// [`crate::exec::window`]):
///
/// * **Ranking functions** — [`WindowFunc::RowNumber`],
///   [`WindowFunc::Rank`], [`WindowFunc::DenseRank`]. These take no
///   argument and depend only on the row's position within its partition
///   under the window's ORDER BY. All three output `Int64`.
/// * **Aggregate windows** — [`WindowFunc::Sum`], [`WindowFunc::Avg`],
///   [`WindowFunc::Min`], [`WindowFunc::Max`], [`WindowFunc::Count`]. These
///   carry an inner [`Expr`] and compute a running (cumulative) aggregate
///   over the default frame `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT
///   ROW`. With an ORDER BY the value at each row is the aggregate of all
///   peer-or-earlier rows; without an ORDER BY every row in the partition
///   sees the full-partition aggregate (the standard SQL behaviour).
///
/// Output dtypes mirror the scalar-aggregate contract in
/// [`AggregateExpr::output_dtype`]: `COUNT` → `Int64`, `AVG` → `Float64`,
/// `SUM` widens narrow integers via [`sum_output_dtype`], `MIN`/`MAX`
/// preserve the input dtype.
#[derive(Debug, Clone)]
pub enum WindowFunc {
    /// `ROW_NUMBER()` — 1-based sequential index within the partition,
    /// ties broken by row order. Output `Int64`.
    RowNumber,
    /// `RANK()` — 1-based rank with gaps: tied rows (equal ORDER BY keys)
    /// share the lowest rank, and the next distinct key skips the tied
    /// count. Output `Int64`.
    Rank,
    /// `DENSE_RANK()` — 1-based rank without gaps: tied rows share a rank
    /// and the next distinct key is exactly one greater. Output `Int64`.
    DenseRank,
    /// `SUM(expr) OVER (...)` — running sum. Output follows
    /// [`sum_output_dtype`].
    Sum(Expr),
    /// `AVG(expr) OVER (...)` — running average. Output `Float64`.
    Avg(Expr),
    /// `MIN(expr) OVER (...)` — running minimum. Output preserves the
    /// input dtype.
    Min(Expr),
    /// `MAX(expr) OVER (...)` — running maximum. Output preserves the
    /// input dtype.
    Max(Expr),
    /// `COUNT(expr) OVER (...)` — running count of non-NULL inputs.
    /// Output `Int64`.
    Count(Expr),
}

impl WindowFunc {
    /// Canonical SQL name of the function (for error messages / default
    /// output naming).
    pub fn sql_name(&self) -> &'static str {
        match self {
            WindowFunc::RowNumber => "ROW_NUMBER",
            WindowFunc::Rank => "RANK",
            WindowFunc::DenseRank => "DENSE_RANK",
            WindowFunc::Sum(_) => "SUM",
            WindowFunc::Avg(_) => "AVG",
            WindowFunc::Min(_) => "MIN",
            WindowFunc::Max(_) => "MAX",
            WindowFunc::Count(_) => "COUNT",
        }
    }

    /// The inner argument expression, if this is an aggregate window
    /// (`None` for the argument-less ranking functions).
    pub fn arg(&self) -> Option<&Expr> {
        match self {
            WindowFunc::RowNumber | WindowFunc::Rank | WindowFunc::DenseRank => None,
            WindowFunc::Sum(e)
            | WindowFunc::Avg(e)
            | WindowFunc::Min(e)
            | WindowFunc::Max(e)
            | WindowFunc::Count(e) => Some(e),
        }
    }

    /// Output dtype of this window function against the input schema.
    ///
    /// Mirrors [`AggregateExpr::output_dtype`] for the aggregate family;
    /// the ranking functions are always `Int64`.
    pub fn output_dtype(&self, input: &Schema) -> BoltResult<DataType> {
        match self {
            WindowFunc::RowNumber | WindowFunc::Rank | WindowFunc::DenseRank => {
                Ok(DataType::Int64)
            }
            WindowFunc::Count(_) => Ok(DataType::Int64),
            WindowFunc::Avg(e) => {
                let _ = e.dtype(input)?;
                Ok(DataType::Float64)
            }
            WindowFunc::Sum(e) => Ok(sum_output_dtype(e.dtype(input)?)),
            WindowFunc::Min(e) | WindowFunc::Max(e) => e.dtype(input),
        }
    }
}

/// A single window-function output column: the function plus the name the
/// computed column receives in the [`LogicalPlan::Window`] output schema.
///
/// The partition / ordering is shared across every `WindowExpr` in a
/// [`LogicalPlan::Window`] node (one node per distinct window spec), so it
/// lives on the node rather than here.
#[derive(Debug, Clone)]
pub struct WindowExpr {
    /// The window function to compute.
    pub func: WindowFunc,
    /// Output column name appended to the input schema.
    pub output_name: String,
}

/// A single ORDER BY entry: an expression plus direction / null placement.
#[derive(Debug, Clone)]
pub struct SortExpr {
    /// The sort key expression.
    pub expr: Expr,
    /// True for DESC, false for ASC.
    pub descending: bool,
    /// True if NULLs sort before non-NULLs (NULLS FIRST), false if after.
    pub nulls_first: bool,
}

/// Join kind. INNER, LEFT, RIGHT, FULL, CROSS supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// SQL INNER JOIN.
    Inner,
    /// SQL LEFT [OUTER] JOIN: every left row appears, NULL-padded on the
    /// right when no match is found.
    LeftOuter,
    /// SQL RIGHT [OUTER] JOIN: every right row appears, NULL-padded on the
    /// left when no match is found.
    RightOuter,
    /// SQL FULL [OUTER] JOIN: union of LEFT + RIGHT semantics — every
    /// unmatched row from either side emits with the opposite side NULL.
    FullOuter,
    /// SQL CROSS JOIN: cartesian product, no ON predicate.
    Cross,
}

impl JoinType {
    /// True if the left side is preserved (every left row emits at least
    /// once). Holds for INNER (matched only), LEFT, FULL, and CROSS;
    /// false for RIGHT (left rows may be dropped if unmatched on the
    /// right).
    pub fn left_preserved(self) -> bool {
        matches!(
            self,
            JoinType::LeftOuter | JoinType::FullOuter | JoinType::Cross
        )
    }

    /// True if the right side is preserved (every right row emits at least
    /// once). Holds for RIGHT, FULL, and CROSS; false for INNER and LEFT.
    pub fn right_preserved(self) -> bool {
        matches!(
            self,
            JoinType::RightOuter | JoinType::FullOuter | JoinType::Cross
        )
    }
}

/// Set-operation kind for [`LogicalPlan::SetOp`] / `PhysicalPlan::SetOp`.
///
/// `UNION` is *not* represented here — it lowers to
/// [`LogicalPlan::Union`] (concatenation, plus a wrapping
/// [`LogicalPlan::Distinct`] for the dedup variant). This enum carries only
/// the two operators that need a dedicated host-side multiset executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOpKind {
    /// `EXCEPT` — rows present in the left input but not the right
    /// (multiset difference; see [`crate::exec::setops`] for the ALL vs
    /// distinct multiset semantics).
    Except,
    /// `INTERSECT` — rows present in both inputs (multiset intersection).
    Intersect,
}

impl SetOpKind {
    /// SQL keyword for this operator (`"EXCEPT"` / `"INTERSECT"`), used in
    /// EXPLAIN output and error messages.
    pub fn keyword(self) -> &'static str {
        match self {
            SetOpKind::Except => "EXCEPT",
            SetOpKind::Intersect => "INTERSECT",
        }
    }
}

/// Relational logical plan node.
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Read a registered table.
    Scan {
        /// Table name.
        table: String,
        /// Optional projected column subset.
        projection: Option<Vec<String>>,
        /// Schema of the (un-projected) table.
        schema: Schema,
    },
    /// Apply a boolean predicate.
    Filter {
        /// Source.
        input: Box<LogicalPlan>,
        /// Boolean expression.
        predicate: Expr,
    },
    /// SELECT list; output schema follows `exprs` in order.
    Project {
        /// Source.
        input: Box<LogicalPlan>,
        /// Output expressions.
        exprs: Vec<Expr>,
    },
    /// GROUP BY + aggregates; empty `group_by` yields a single output row.
    Aggregate {
        /// Source.
        input: Box<LogicalPlan>,
        /// Grouping expressions.
        group_by: Vec<Expr>,
        /// Aggregate expressions.
        aggregates: Vec<AggregateExpr>,
    },
    /// SQL DISTINCT: deduplicate rows from `input`. Schema = input.schema().
    Distinct {
        /// Source.
        input: Box<LogicalPlan>,
    },
    /// SQL LIMIT [OFFSET]: keep at most `limit` rows after skipping `offset`.
    /// Schema = input.schema().
    Limit {
        /// Source.
        input: Box<LogicalPlan>,
        /// Maximum number of rows to emit.
        limit: usize,
        /// Number of leading rows to skip (0 if no OFFSET clause).
        offset: usize,
    },
    /// SQL ORDER BY: sort `input` by `sort_exprs`. Schema = input.schema().
    Sort {
        /// Source.
        input: Box<LogicalPlan>,
        /// Sort keys, evaluated in order (first is most significant).
        sort_exprs: Vec<SortExpr>,
    },
    /// SQL window functions: `func(...) OVER (PARTITION BY ... ORDER BY
    /// ...)`. Each [`WindowExpr`] appends one computed column to the input's
    /// schema; the partition / ordering is shared across every expr in the
    /// node (one node per distinct window spec). Schema =
    /// `input.schema() ++ [one field per window_expr]`.
    ///
    /// **Host-only (v0.x):** the default frame (`RANGE UNBOUNDED PRECEDING`)
    /// is the only frame supported; the SQL frontend rejects explicit exotic
    /// frames. Execution is host-side — see [`crate::exec::window`].
    Window {
        /// Source.
        input: Box<LogicalPlan>,
        /// One output column per window function sharing this spec.
        window_exprs: Vec<WindowExpr>,
        /// `PARTITION BY` keys (empty = single partition over all rows).
        partition_by: Vec<Expr>,
        /// `ORDER BY` keys within each partition (empty = no ordering, so
        /// aggregate windows see the whole partition and ranking functions
        /// fall back to physical row order).
        order_by: Vec<SortExpr>,
    },
    /// SQL UNION ALL — concatenation without dedup. UNION (with dedup) is
    /// parsed and lowered to `Distinct(Union { ... })`. All inputs must share
    /// the same schema; the result schema is the first input's schema.
    Union {
        /// Branches to concatenate, in source order.
        inputs: Vec<LogicalPlan>,
    },
    /// SQL `EXCEPT` / `INTERSECT` (with optional `ALL`). `left` and `right`
    /// must share a compatible schema (same field count + per-field dtypes,
    /// the same rule [`LogicalPlan::Union`] enforces); the result schema is
    /// the left input's. The dedup-vs-multiset semantics are chosen by
    /// `all`:
    ///
    /// * `all == false` (plain `EXCEPT` / `INTERSECT`) — the result is a
    ///   *set*: each surviving row appears at most once.
    /// * `all == true` (`EXCEPT ALL` / `INTERSECT ALL`) — the result is a
    ///   *multiset*: row multiplicities follow the SQL standard
    ///   (`EXCEPT ALL`: `max(0, lc - rc)` copies; `INTERSECT ALL`:
    ///   `min(lc, rc)` copies, where `lc`/`rc` are the per-row counts in the
    ///   left / right inputs).
    ///
    /// Executed host-side by [`crate::exec::setops`], which reuses the
    /// `DISTINCT` executor's row-key / NULL canonicalisation so two NULLs in
    /// the same column compare equal (matching the engine-wide convention).
    SetOp {
        /// Left input.
        left: Box<LogicalPlan>,
        /// Right input.
        right: Box<LogicalPlan>,
        /// `EXCEPT` or `INTERSECT`.
        op: SetOpKind,
        /// `true` for the `ALL` (multiset) variant; `false` for the
        /// set-returning default.
        all: bool,
    },
    /// SQL JOIN: combine `left` and `right` rows that satisfy `on`.
    /// Supports `JoinType::Inner`, `LeftOuter`, `RightOuter`, `FullOuter`,
    /// and `Cross`. INNER and the OUTER variants require at least one
    /// equi-join predicate (`on` non-empty) OR a non-equi `filter`;
    /// CROSS requires both `on` and `filter` to be empty (no ON clause).
    ///
    /// # Non-equi join contract (v0.6)
    ///
    /// `on` carries pure equi pairs (`left.col = right.col`) for the
    /// hash-join fast path. `filter` carries the residual non-equi predicate
    /// — `<`, `>`, `BETWEEN`, etc. — that cannot be expressed as a hash
    /// lookup, evaluated against the join's *combined* schema. When `filter`
    /// is `Some`, the executor switches to a nested-loop fallback (see
    /// [`crate::exec::join`]); the cap on the inner side is enforced at
    /// runtime.
    Join {
        /// Left input.
        left: Box<LogicalPlan>,
        /// Right input.
        right: Box<LogicalPlan>,
        /// Join kind.
        join_type: JoinType,
        /// Equi-join predicate pairs `(left_expr, right_expr)`;
        /// conjunctive. Empty for `Cross` and for pure non-equi joins.
        on: Vec<(Expr, Expr)>,
        /// Optional residual non-equi predicate evaluated against the
        /// combined left ++ right schema (with right-side rename rules of
        /// [`join_combined_schema`] applied). `None` means a pure
        /// equi/cross join; `Some(_)` routes through the nested-loop
        /// executor.
        filter: Option<Expr>,
    },
}

impl LogicalPlan {
    /// Type-check the plan and return its output schema.
    pub fn schema(&self) -> BoltResult<Schema> {
        self.schema_depth(0)
    }

    /// Inner recursion for [`LogicalPlan::schema`]. `depth` is the current
    /// recursion depth; returns Err if
    /// [`crate::plan::sql_frontend::MAX_RECURSION_DEPTH`] is exceeded —
    /// guards against attacker-controlled deeply nested plans reaching
    /// type-checking after construction (which would otherwise overflow
    /// the host thread stack).
    fn schema_depth(&self, depth: usize) -> BoltResult<Schema> {
        if depth > crate::plan::sql_frontend::MAX_RECURSION_DEPTH {
            return Err(BoltError::Plan(format!(
                "plan nesting exceeds depth limit ({})",
                crate::plan::sql_frontend::MAX_RECURSION_DEPTH
            )));
        }
        match self {
            LogicalPlan::Scan {
                projection, schema, ..
            } => match projection {
                None => Ok(schema.clone()),
                Some(cols) => {
                    let mut fields = Vec::with_capacity(cols.len());
                    for c in cols {
                        fields.push(schema.field(c)?.clone());
                    }
                    Ok(Schema::new(fields))
                }
            },
            LogicalPlan::Filter { input, predicate } => {
                let s = input.schema_depth(depth + 1)?;
                let pt = predicate.dtype(&s)?;
                if pt != DataType::Bool {
                    return Err(BoltError::Type(format!(
                        "filter predicate must be Bool, got {pt:?}"
                    )));
                }
                Ok(s)
            }
            LogicalPlan::Project { input, exprs } => {
                let s = input.schema_depth(depth + 1)?;
                let mut fields = Vec::with_capacity(exprs.len());
                for (i, e) in exprs.iter().enumerate() {
                    let dtype = e.dtype(&s)?;
                    let name = match e {
                        Expr::Column(n) => n.clone(),
                        Expr::Alias(_, n) => n.clone(),
                        _ => format!("__expr_{i}"),
                    };
                    fields.push(Field::new(name, dtype, true));
                }
                Ok(Schema::new(fields))
            }
            LogicalPlan::Aggregate {
                input,
                group_by,
                aggregates,
            } => {
                let s = input.schema_depth(depth + 1)?;
                let mut fields = Vec::with_capacity(group_by.len() + aggregates.len());
                for (i, g) in group_by.iter().enumerate() {
                    let dtype = g.dtype(&s)?;
                    // Route through the authoritative helper so the rule
                    // (Column/Alias keep their name, anything else gets a
                    // positional `__group_{i}` placeholder) lives in one
                    // place. `sql_frontend.rs` calls the same helper to
                    // recover these names when re-projecting the
                    // Aggregate's output into SELECT-list order.
                    let name = group_key_output_name(g, i);
                    // GROUP BY key fields are nullable: SQL groups NULL keys into
                    // a single NULL group (see `execute_groupby`'s NULL-group
                    // handling), so the key column of the output can carry a NULL
                    // cell. A nullable field still accepts a fully-non-null key
                    // array, so the common no-NULL path is unaffected.
                    fields.push(Field::new(name, dtype, true));
                }
                for agg in aggregates {
                    let dtype = agg.output_dtype(&s)?;
                    fields.push(Field::new(agg.output_name(), dtype, true));
                }
                Ok(Schema::new(fields))
            }
            // Row-shape preserving wrappers: schema is the input's schema.
            LogicalPlan::Distinct { input }
            | LogicalPlan::Limit { input, .. }
            | LogicalPlan::Sort { input, .. } => {
                // For Sort we additionally type-check the sort keys against
                // the input schema so misnamed columns surface here rather
                // than at execution time.
                let s = input.schema_depth(depth + 1)?;
                if let LogicalPlan::Sort { sort_exprs, .. } = self {
                    for se in sort_exprs {
                        // We don't constrain the key dtype (any orderable
                        // scalar is fine); just resolve it so unknown columns
                        // produce a Plan error.
                        let _ = se.expr.dtype(&s)?;
                    }
                }
                Ok(s)
            }
            LogicalPlan::Window {
                input,
                window_exprs,
                partition_by,
                order_by,
            } => {
                let s = input.schema_depth(depth + 1)?;
                // Type-check the partition / ordering keys against the input
                // schema so misnamed columns surface here rather than at
                // execution time. We don't constrain their dtype (any
                // orderable scalar is fine).
                for p in partition_by {
                    let _ = p.dtype(&s)?;
                }
                for se in order_by {
                    let _ = se.expr.dtype(&s)?;
                }
                // Output schema = input fields, then one appended field per
                // window expression. Each window output is nullable (running
                // aggregates over an all-NULL prefix yield NULL; ranking
                // functions never emit NULL but a uniform nullable=true keeps
                // the appended-column contract simple).
                let mut fields = s.fields.clone();
                fields.reserve(window_exprs.len());
                for we in window_exprs {
                    let dtype = we.func.output_dtype(&s)?;
                    fields.push(Field::new(we.output_name.clone(), dtype, true));
                }
                Ok(Schema::new(fields))
            }
            LogicalPlan::Union { inputs } => {
                if inputs.is_empty() {
                    return Err(BoltError::Plan(
                        "UNION requires at least one input".into(),
                    ));
                }
                let first = inputs[0].schema_depth(depth + 1)?;
                for (i, branch) in inputs.iter().enumerate().skip(1) {
                    let other = branch.schema_depth(depth + 1)?;
                    if !schemas_compatible(&first, &other) {
                        return Err(BoltError::Plan(format!(
                            "UNION branch {i} schema does not match branch 0: \
                             expected {} fields ({}), got {} fields ({})",
                            first.fields.len(),
                            schema_summary(&first),
                            other.fields.len(),
                            schema_summary(&other),
                        )));
                    }
                }
                Ok(first)
            }
            LogicalPlan::SetOp {
                left, right, op, ..
            } => {
                // Both inputs must share a compatible schema (same field
                // count + per-field dtypes), the same rule UNION enforces.
                // The result schema is the left input's.
                let l = left.schema_depth(depth + 1)?;
                let r = right.schema_depth(depth + 1)?;
                if !schemas_compatible(&l, &r) {
                    return Err(BoltError::Plan(format!(
                        "{} inputs have incompatible schemas: \
                         left has {} fields ({}), right has {} fields ({})",
                        op.keyword(),
                        l.fields.len(),
                        schema_summary(&l),
                        r.fields.len(),
                        schema_summary(&r),
                    )));
                }
                Ok(l)
            }
            LogicalPlan::Join {
                left,
                right,
                join_type,
                ..
            } => {
                // Concatenate left and right schemas, disambiguating right-
                // side columns whose names collide with anything on the left.
                // For OUTER joins, the columns coming from the side that
                // may be NULL-padded are marked `nullable = true` (a row
                // from the preserved side may have no match on the other).
                // See `join_combined_schema` for the canonical rule (also
                // used by `PhysicalPlan::Join::output_schema()`);
                // duplicating it here would risk drift if either copy is
                // edited.
                let l = left.schema_depth(depth + 1)?;
                let r = right.schema_depth(depth + 1)?;
                Ok(join_combined_schema(&l, &r, *join_type))
            }
        }
    }
}

/// Build the output schema of a JOIN over `left` and `right`.
///
/// Concatenates the two schemas in order, but disambiguates any right-side
/// field whose name already appears on the left by prefixing it with
/// `"right."`. Left-side fields keep their bare names so existing
/// downstream references continue to resolve unchanged. The rule:
///
/// * For each right-side field `f`:
///   * if `f.name` does not collide with any left-side name, keep it as-is;
///   * otherwise rename it to `"right.{f.name}"`. If `"right.{f.name}"`
///     itself collides (rare — only if the left side has a literal
///     `"right.<name>"` column), append `__2`, `__3`, ... until unique.
///
/// Nullability of output fields is widened for OUTER joins. For a
/// `LEFT [OUTER]` join, every right-side column becomes nullable
/// (preserved-left rows with no match emit NULL-padded right columns).
/// `RIGHT [OUTER]` is symmetric; `FULL [OUTER]` widens both sides;
/// `CROSS` and `INNER` leave nullability untouched.
///
/// This is the single source of truth for join output schemas, called by
/// both [`LogicalPlan::Join::schema`](LogicalPlan#method.schema)
/// and [`PhysicalPlan::Join::output_schema`](crate::plan::physical_plan::PhysicalPlan::output_schema)
/// so the logical and physical layers can never disagree on what a join
/// produces.
pub fn join_combined_schema(left: &Schema, right: &Schema, join_type: JoinType) -> Schema {
    // Outer joins NULL-pad the *non-preserved* side: LEFT preserves the
    // left side and may NULL-pad the right; RIGHT is symmetric; FULL may
    // NULL-pad either side. CROSS and INNER never NULL-pad here.
    let left_may_null = matches!(join_type, JoinType::RightOuter | JoinType::FullOuter);
    let right_may_null = matches!(join_type, JoinType::LeftOuter | JoinType::FullOuter);

    let mut fields: Vec<Field> = Vec::with_capacity(left.fields.len() + right.fields.len());
    for lf in &left.fields {
        fields.push(Field {
            name: lf.name.clone(),
            dtype: lf.dtype,
            nullable: lf.nullable || left_may_null,
        });
    }
    // Snapshot the names already taken by the left side so collision lookup
    // doesn't depend on later right-side insertions. `join_rename` mutates
    // this set so each rename also sees the names produced for prior
    // right-side columns.
    let mut taken: std::collections::HashSet<String> =
        left.fields.iter().map(|f| f.name.clone()).collect();
    for rf in &right.fields {
        let name = join_rename(&rf.name, &mut taken);
        fields.push(Field {
            name,
            dtype: rf.dtype,
            nullable: rf.nullable || right_may_null,
        });
    }
    Schema { fields }
}

/// True if `a` and `b` have the same shape (same number of fields, same dtype
/// per position). Field names need not match: SQL UNION ALL takes the names
/// from the leftmost branch.
fn schemas_compatible(a: &Schema, b: &Schema) -> bool {
    if a.fields.len() != b.fields.len() {
        return false;
    }
    a.fields
        .iter()
        .zip(b.fields.iter())
        .all(|(x, y)| x.dtype == y.dtype)
}

/// One-line summary of a schema for error messages (`name: Type, ...`).
fn schema_summary(s: &Schema) -> String {
    s.fields
        .iter()
        .map(|f| format!("{}: {:?}", f.name, f.dtype))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Authoritative naming rule for aggregate output columns.
///
/// Thin free-function wrapper around [`AggregateExpr::output_name`] for use
/// from outside this module (the method itself is `pub(crate)`, but a free
/// function keeps the call sites in `sql_frontend.rs` clear of method-syntax
/// borrows). Called from `sql_frontend.rs::plan_select` (SELECT-list
/// re-projection over an `Aggregate` plan) and
/// `sql_frontend.rs::lower_expr_in_having` (HAVING rewriter); do not
/// duplicate the rule at the call sites.
pub(crate) fn aggregate_output_name(agg: &AggregateExpr) -> String {
    agg.output_name()
}

/// Authoritative naming rule for GROUP BY output columns inside an
/// `Aggregate` plan's output schema: a bare `Column` or top-level `Alias`
/// keeps its name; anything else gets a positional `__group_{idx}`
/// placeholder.
///
/// Called from [`LogicalPlan::schema`] (this file, in the `Aggregate` arm)
/// and from `sql_frontend.rs::plan_select` (the SELECT-list re-projection,
/// which needs to recover these names to wire group keys through to the
/// user-visible projection). Do not duplicate the rule at either call site.
pub(crate) fn group_key_output_name(key: &Expr, idx: usize) -> String {
    match key {
        Expr::Column(n) => n.clone(),
        Expr::Alias(_, n) => n.clone(),
        _ => format!("__group_{idx}"),
    }
}

/// Authoritative naming rule for a single right-side JOIN column when
/// disambiguating against an accumulated set of already-taken names.
pub(crate) fn join_rename(name: &str, taken: &mut std::collections::HashSet<String>) -> String {
    let mut out_name = if taken.contains(name) {
        format!("right.{name}")
    } else {
        name.to_string()
    };
    if taken.contains(&out_name) {
        let base = out_name.clone();
        let mut i = 2usize;
        loop {
            let candidate = format!("{base}__{i}");
            if !taken.contains(&candidate) {
                out_name = candidate;
                break;
            }
            i += 1;
        }
    }
    taken.insert(out_name.clone());
    out_name
}

#[cfg(test)]
mod scalar_fn_typecheck_tests {
    //! Direct type-check coverage for `Expr::ScalarFn` at the logical-plan
    //! layer. The SQL frontend has its own integration coverage in
    //! `tests/string_fns_sql_test.rs`; the tests here pin the per-kind
    //! contract documented on [`ScalarFnKind`].
    use super::*;

    fn s_schema() -> Schema {
        Schema::new(vec![
            Field::new("s", DataType::Utf8, false),
            Field::new("n", DataType::Int32, false),
        ])
    }

    #[test]
    fn upper_lower_utf8_to_utf8() {
        let schema = s_schema();
        for kind in [ScalarFnKind::Upper, ScalarFnKind::Lower] {
            let e = Expr::ScalarFn {
                kind,
                args: vec![Expr::Column("s".into())],
            };
            assert_eq!(e.dtype(&schema).unwrap(), DataType::Utf8);
        }
    }

    #[test]
    fn length_utf8_to_int64() {
        let schema = s_schema();
        let e = Expr::ScalarFn {
            kind: ScalarFnKind::Length,
            args: vec![Expr::Column("s".into())],
        };
        assert_eq!(e.dtype(&schema).unwrap(), DataType::Int64);
    }

    #[test]
    fn substring_two_or_three_int64_args_returns_utf8() {
        let schema = s_schema();
        let two = Expr::ScalarFn {
            kind: ScalarFnKind::Substring,
            args: vec![
                Expr::Column("s".into()),
                Expr::Literal(Literal::Int64(1)),
            ],
        };
        assert_eq!(two.dtype(&schema).unwrap(), DataType::Utf8);

        let three = Expr::ScalarFn {
            kind: ScalarFnKind::Substring,
            args: vec![
                Expr::Column("s".into()),
                Expr::Literal(Literal::Int64(1)),
                Expr::Literal(Literal::Int64(3)),
            ],
        };
        assert_eq!(three.dtype(&schema).unwrap(), DataType::Utf8);
    }

    #[test]
    fn concat_requires_two_or_more_utf8() {
        let schema = s_schema();
        // 2 args OK.
        let e2 = Expr::ScalarFn {
            kind: ScalarFnKind::Concat,
            args: vec![
                Expr::Column("s".into()),
                Expr::Literal(Literal::Utf8("x".into())),
            ],
        };
        assert_eq!(e2.dtype(&schema).unwrap(), DataType::Utf8);
        // 3 args OK (variadic).
        let e3 = Expr::ScalarFn {
            kind: ScalarFnKind::Concat,
            args: vec![
                Expr::Column("s".into()),
                Expr::Literal(Literal::Utf8("x".into())),
                Expr::Column("s".into()),
            ],
        };
        assert_eq!(e3.dtype(&schema).unwrap(), DataType::Utf8);
        // 1 arg: error.
        let e1 = Expr::ScalarFn {
            kind: ScalarFnKind::Concat,
            args: vec![Expr::Column("s".into())],
        };
        assert!(e1.dtype(&schema).is_err());
    }

    #[test]
    fn upper_rejects_non_utf8() {
        let schema = s_schema();
        let e = Expr::ScalarFn {
            kind: ScalarFnKind::Upper,
            args: vec![Expr::Column("n".into())],
        };
        assert!(e.dtype(&schema).is_err());
    }

    #[test]
    fn substring_rejects_non_int64_start() {
        let schema = s_schema();
        let e = Expr::ScalarFn {
            kind: ScalarFnKind::Substring,
            args: vec![
                Expr::Column("s".into()),
                Expr::Column("n".into()), // Int32 not Int64
            ],
        };
        assert!(e.dtype(&schema).is_err());
    }

    #[test]
    fn substring_rejects_zero_args() {
        let schema = s_schema();
        let e = Expr::ScalarFn {
            kind: ScalarFnKind::Substring,
            args: vec![],
        };
        assert!(e.dtype(&schema).is_err());
    }

    #[test]
    fn length_rejects_non_utf8() {
        let schema = s_schema();
        let e = Expr::ScalarFn {
            kind: ScalarFnKind::Length,
            args: vec![Expr::Column("n".into())],
        };
        assert!(e.dtype(&schema).is_err());
    }
}

#[cfg(test)]
mod date_scalar_typecheck_tests {
    //! Type-check coverage for `Expr::Extract` / `Expr::DateTrunc` at the
    //! logical-plan layer. The GPU codegen lives in
    //! `crate::jit::date_scalar` and has its own PTX-assertion tests; here we
    //! pin the per-(field/unit, operand-dtype) dtype rules.
    use super::*;

    fn dt_schema() -> Schema {
        Schema::new(vec![
            Field::new("d", DataType::Date32, false),
            Field::new("ts", DataType::Timestamp(TimeUnit::Second, None), false),
            Field::new("n", DataType::Int32, false),
        ])
    }

    #[test]
    fn extract_calendar_field_is_int64() {
        let schema = dt_schema();
        for (col, field) in [
            ("d", DateField::Year),
            ("d", DateField::Month),
            ("d", DateField::Day),
            ("ts", DateField::Year),
            ("ts", DateField::Hour),
            ("ts", DateField::Second),
        ] {
            let e = Expr::Extract {
                field,
                expr: Box::new(Expr::Column(col.into())),
            };
            assert_eq!(e.dtype(&schema).unwrap(), DataType::Int64);
        }
    }

    #[test]
    fn extract_intraday_from_date32_rejected() {
        let schema = dt_schema();
        for field in [DateField::Hour, DateField::Minute, DateField::Second] {
            let e = Expr::Extract {
                field,
                expr: Box::new(Expr::Column("d".into())),
            };
            assert!(
                e.dtype(&schema).is_err(),
                "EXTRACT({:?} FROM Date32) must be a type error",
                field
            );
        }
    }

    #[test]
    fn extract_from_non_temporal_rejected() {
        let schema = dt_schema();
        let e = Expr::Extract {
            field: DateField::Year,
            expr: Box::new(Expr::Column("n".into())),
        };
        assert!(e.dtype(&schema).is_err());
    }

    #[test]
    fn date_trunc_preserves_operand_dtype() {
        let schema = dt_schema();
        // Date32 → Date32 for calendar units.
        let e = Expr::DateTrunc {
            unit: DateTruncUnit::Month,
            expr: Box::new(Expr::Column("d".into())),
        };
        assert_eq!(e.dtype(&schema).unwrap(), DataType::Date32);
        // Timestamp → same Timestamp type.
        let e = Expr::DateTrunc {
            unit: DateTruncUnit::Hour,
            expr: Box::new(Expr::Column("ts".into())),
        };
        assert_eq!(
            e.dtype(&schema).unwrap(),
            DataType::Timestamp(TimeUnit::Second, None)
        );
    }

    #[test]
    fn date_trunc_subday_on_date32_rejected() {
        let schema = dt_schema();
        for unit in [
            DateTruncUnit::Hour,
            DateTruncUnit::Minute,
            DateTruncUnit::Second,
        ] {
            let e = Expr::DateTrunc {
                unit,
                expr: Box::new(Expr::Column("d".into())),
            };
            assert!(
                e.dtype(&schema).is_err(),
                "DATE_TRUNC({:?}, Date32) must be a type error",
                unit
            );
        }
    }
}

#[cfg(test)]
mod null_handling_tests {
    use super::*;

    /// Baseline contract: a bare `Literal::Null` still has no static type.
    /// The new NULL-peer-typing surface kicks in at the `Expr::Binary` /
    /// `Expr::Unary` layer, not at the literal layer itself.
    #[test]
    fn literal_null_dtype_is_none() {
        assert_eq!(Literal::Null.dtype(), None);
    }

    /// `WHERE x = NULL` with `x: Int32` must type-check (NULL borrows the
    /// peer's dtype) and resolve the binary expression to `Bool`. The
    /// runtime semantics of `= NULL` are a separate concern handled by the
    /// executor; the planner just needs not to hard-error.
    #[test]
    fn null_peer_typing_in_binary_eq() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, true)]);
        let expr = Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(Expr::Column("x".into())),
            right: Box::new(Expr::Literal(Literal::Null)),
        };
        let t = expr.dtype(&schema).expect("NULL peer-typing must succeed");
        assert_eq!(t, DataType::Bool);
        // Symmetric — NULL on the left side also works.
        let expr_rev = Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(Expr::Literal(Literal::Null)),
            right: Box::new(Expr::Column("x".into())),
        };
        let t_rev = expr_rev
            .dtype(&schema)
            .expect("NULL peer-typing must be symmetric");
        assert_eq!(t_rev, DataType::Bool);
    }

    /// Two NULLs on both sides still surface the legacy
    /// "untyped NULL literal" error — there is no peer to borrow a dtype from.
    #[test]
    fn binary_with_two_nulls_still_errors() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, true)]);
        let expr = Expr::Binary {
            op: BinaryOp::Eq,
            left: Box::new(Expr::Literal(Literal::Null)),
            right: Box::new(Expr::Literal(Literal::Null)),
        };
        assert!(expr.dtype(&schema).is_err());
    }

    /// `x IS NULL` and `x IS NOT NULL` always type-check to Bool — even
    /// when the operand is itself an untyped `Literal::Null`.
    #[test]
    fn unary_is_null_typechecks_to_bool() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, true)]);
        for op in [UnaryOp::IsNull, UnaryOp::IsNotNull] {
            let on_col = Expr::Unary {
                op,
                operand: Box::new(Expr::Column("x".into())),
            };
            assert_eq!(on_col.dtype(&schema).unwrap(), DataType::Bool);
            let on_null = Expr::Unary {
                op,
                operand: Box::new(Expr::Literal(Literal::Null)),
            };
            assert_eq!(on_null.dtype(&schema).unwrap(), DataType::Bool);
        }
    }

    /// `NOT <bool>` type-checks to Bool when the operand is Bool.
    #[test]
    fn unary_not_typechecks_against_bool_operand() {
        let schema = Schema::new(vec![
            Field::new("b", DataType::Bool, true),
            Field::new("x", DataType::Int32, true),
        ]);
        let on_bool = Expr::Column("b".into()).not();
        assert_eq!(on_bool.dtype(&schema).unwrap(), DataType::Bool);
        let on_int = Expr::Column("x".into()).not();
        assert!(on_int.dtype(&schema).is_err());
        let on_null = Expr::Literal(Literal::Null).not();
        assert_eq!(on_null.dtype(&schema).unwrap(), DataType::Bool);
    }

    /// `expr LIKE 'pat'` resolves to Bool when the operand is Utf8.
    #[test]
    fn like_typechecks_to_bool_on_utf8() {
        let schema = Schema::new(vec![
            Field::new("s", DataType::Utf8, false),
            Field::new("v", DataType::Int32, false),
        ]);
        let ok = Expr::Like {
            expr: Box::new(Expr::Column("s".into())),
            pattern: "foo%".into(),
            escape: None,
            negated: false,
            case_insensitive: false,
        };
        assert_eq!(ok.dtype(&schema).unwrap(), DataType::Bool);

        let bad = Expr::Like {
            expr: Box::new(Expr::Column("v".into())),
            pattern: "foo%".into(),
            escape: None,
            negated: false,
            case_insensitive: false,
        };
        let err = bad.dtype(&schema).expect_err("LIKE on Int32 must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("LIKE requires a Utf8 operand"),
            "expected Utf8 type error, got: {msg}"
        );
    }

    /// Arithmetic peer-typing: `x + NULL` with `x: Int64` resolves to
    /// `Int64` (the arithmetic unification rule applied with NULL borrowing
    /// its peer's dtype).
    #[test]
    fn null_peer_typing_in_binary_add() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, true)]);
        let expr = Expr::Binary {
            op: BinaryOp::Add,
            left: Box::new(Expr::Column("x".into())),
            right: Box::new(Expr::Literal(Literal::Null)),
        };
        assert_eq!(expr.dtype(&schema).unwrap(), DataType::Int64);
    }

    /// CASE with a single Bool WHEN and two Int64 arms (THEN + ELSE)
    /// resolves to Int64 without widening.
    #[test]
    fn case_with_else_uniform_int_arms_typechecks_to_int64() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, false)]);
        let case = Expr::Case {
            branches: vec![(
                Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Column("x".into())),
                    right: Box::new(Expr::Literal(Literal::Int64(0))),
                },
                Expr::Literal(Literal::Int64(1)),
            )],
            else_branch: Some(Box::new(Expr::Literal(Literal::Int64(0)))),
        };
        assert_eq!(case.dtype(&schema).unwrap(), DataType::Int64);
    }

    /// CASE without ELSE: dtype is taken from the THEN arms alone.
    #[test]
    fn case_without_else_takes_then_dtype() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, false)]);
        let case = Expr::Case {
            branches: vec![(
                Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Column("x".into())),
                    right: Box::new(Expr::Literal(Literal::Int64(0))),
                },
                Expr::Literal(Literal::Float64(1.0)),
            )],
            else_branch: None,
        };
        assert_eq!(case.dtype(&schema).unwrap(), DataType::Float64);
    }

    /// Mixed numeric arms widen pairwise: Int64 THEN + Float64 ELSE
    /// resolves to Float64 via the same `unify_numeric` rule used for
    /// arithmetic.
    #[test]
    fn case_with_mixed_numeric_arms_widens_to_float64() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int64, false)]);
        let case = Expr::Case {
            branches: vec![(
                Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Column("x".into())),
                    right: Box::new(Expr::Literal(Literal::Int64(0))),
                },
                Expr::Literal(Literal::Int64(1)),
            )],
            else_branch: Some(Box::new(Expr::Literal(Literal::Float64(0.5)))),
        };
        assert_eq!(case.dtype(&schema).unwrap(), DataType::Float64);
    }

    /// Non-Bool WHEN condition surfaces a Type error naming the offending
    /// branch index and the dtype that broke it.
    #[test]
    fn case_rejects_non_bool_when_condition() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
        let case = Expr::Case {
            branches: vec![(
                Expr::Column("x".into()),
                Expr::Literal(Literal::Int64(1)),
            )],
            else_branch: Some(Box::new(Expr::Literal(Literal::Int64(0)))),
        };
        let err = case.dtype(&schema).expect_err("non-Bool WHEN must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("CASE WHEN condition") && msg.contains("Bool"),
            "error should mention the failure, got: {msg}"
        );
    }

    /// Incompatible non-numeric arms (Utf8 vs Bool) cannot unify; the
    /// error message must mention the offending arm label.
    #[test]
    fn case_rejects_incompatible_non_numeric_arms() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
        let case = Expr::Case {
            branches: vec![(
                Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Column("x".into())),
                    right: Box::new(Expr::Literal(Literal::Int64(0))),
                },
                Expr::Literal(Literal::Utf8("yes".into())),
            )],
            else_branch: Some(Box::new(Expr::Literal(Literal::Bool(true)))),
        };
        let err = case
            .dtype(&schema)
            .expect_err("Utf8 + Bool arms must error");
        let msg = format!("{err}");
        assert!(
            msg.contains("CASE") && msg.contains("incompatible"),
            "expected incompatibility message, got: {msg}"
        );
    }

    /// Untyped-NULL THEN borrows the ELSE arm's dtype just like the
    /// NULL-peer-typing rule for binary ops. THEN = NULL, ELSE = Int64
    /// resolves to Int64.
    #[test]
    fn case_with_null_then_borrows_else_dtype() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
        let case = Expr::Case {
            branches: vec![(
                Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Column("x".into())),
                    right: Box::new(Expr::Literal(Literal::Int64(0))),
                },
                Expr::Literal(Literal::Null),
            )],
            else_branch: Some(Box::new(Expr::Literal(Literal::Int64(0)))),
        };
        assert_eq!(case.dtype(&schema).unwrap(), DataType::Int64);
    }

    /// Every arm being an untyped NULL leaves the planner with nothing
    /// to anchor on; surface a clear Type error rather than guessing a
    /// default dtype.
    #[test]
    fn case_with_only_null_arms_errors() {
        let schema = Schema::new(vec![Field::new("x", DataType::Int32, false)]);
        let case = Expr::Case {
            branches: vec![(
                Expr::Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Column("x".into())),
                    right: Box::new(Expr::Literal(Literal::Int64(0))),
                },
                Expr::Literal(Literal::Null),
            )],
            else_branch: Some(Box::new(Expr::Literal(Literal::Null))),
        };
        let err = case
            .dtype(&schema)
            .expect_err("all-NULL arms must error at type-check");
        let msg = format!("{err}");
        assert!(
            msg.contains("CASE") && msg.contains("untyped NULL"),
            "expected all-NULL error, got: {msg}"
        );
    }
}

#[cfg(test)]
mod naming_consistency_tests {
    //! Lock the authoritative naming rules in place. These tests guard the
    //! consolidation in this module against regressions: if anyone changes
    //! the rule here, downstream `sql_frontend.rs` must observe the same
    //! change because both sites route through these helpers.
    use super::*;

    #[test]
    fn aggregate_output_name_is_stable_for_representative_exprs() {
        // Bare Column: `_colname` suffix.
        let agg = AggregateExpr::Sum(Expr::Column("price".to_string()));
        assert_eq!(aggregate_output_name(&agg), "sum_price");
        assert_eq!(agg.output_name(), "sum_price");

        let agg = AggregateExpr::Avg(Expr::Column("qty".to_string()));
        assert_eq!(aggregate_output_name(&agg), "avg_qty");

        let agg = AggregateExpr::Min(Expr::Column("ts".to_string()));
        assert_eq!(aggregate_output_name(&agg), "min_ts");

        let agg = AggregateExpr::Max(Expr::Column("ts".to_string()));
        assert_eq!(aggregate_output_name(&agg), "max_ts");

        // VAR_POP / VAR_SAMP follow the same suffix rule.
        let agg = AggregateExpr::VarPop(Box::new(Expr::Column("v".to_string())));
        assert_eq!(aggregate_output_name(&agg), "var_pop_v");
        let agg = AggregateExpr::VarSamp(Box::new(Expr::Column("v".to_string())));
        assert_eq!(aggregate_output_name(&agg), "var_samp_v");

        // Alias: take the alias name as the suffix.
        let aliased = Expr::Alias(Box::new(Expr::Column("c".to_string())), "renamed".to_string());
        let agg = AggregateExpr::Count(aliased);
        assert_eq!(aggregate_output_name(&agg), "count_renamed");

        // Non-column / non-alias inner expr: no suffix.
        let lit = Expr::Literal(Literal::Int64(1));
        let agg = AggregateExpr::Count(lit);
        assert_eq!(aggregate_output_name(&agg), "count");
    }

    #[test]
    fn group_key_output_name_is_stable_for_representative_exprs() {
        // Bare column keeps its name.
        assert_eq!(
            group_key_output_name(&Expr::Column("region".to_string()), 0),
            "region"
        );
        // Alias keeps its alias name regardless of index.
        let aliased = Expr::Alias(Box::new(Expr::Column("c".to_string())), "r".to_string());
        assert_eq!(group_key_output_name(&aliased, 3), "r");
        // Anything else falls back to a positional placeholder.
        let lit = Expr::Literal(Literal::Int64(7));
        assert_eq!(group_key_output_name(&lit, 0), "__group_0");
        assert_eq!(group_key_output_name(&lit, 2), "__group_2");
    }

    #[test]
    fn join_combined_schema_renames_colliding_right_side_to_right_dot_prefix() {
        // Both sides have a column named `a`; the right one should be
        // renamed to `right.a`. The left one stays as `a`.
        let left = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let right = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let out = join_combined_schema(&left, &right, JoinType::Inner);
        let names: Vec<&str> = out.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["a", "right.a"]);
    }

    #[test]
    fn join_combined_schema_passes_through_non_colliding_names() {
        // No collision — both sides keep their original names.
        let left = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let right = Schema::new(vec![Field::new("b", DataType::Int32, false)]);
        let out = join_combined_schema(&left, &right, JoinType::Inner);
        let names: Vec<&str> = out.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn join_combined_schema_falls_back_to_numeric_suffix_on_qualified_collision() {
        // Pathological case: the left side already has a column literally
        // named `right.a`, so the right-side `a` cannot be renamed to
        // `right.a` and must fall through to the `__2` suffix.
        let left = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("right.a", DataType::Int32, false),
        ]);
        let right = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let out = join_combined_schema(&left, &right, JoinType::Inner);
        let names: Vec<&str> = out.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, vec!["a", "right.a", "right.a__2"]);
    }

    #[test]
    fn join_rename_matches_join_combined_schema_for_simple_collision() {
        // The standalone helper must produce the same rename sequence the
        // full schema-building function produces; this is the contract
        // `sql_frontend.rs::NameResolver::push_join` relies on.
        let mut taken: std::collections::HashSet<String> = ["a".to_string()].into_iter().collect();
        let renamed = join_rename("a", &mut taken);
        assert_eq!(renamed, "right.a");
        assert!(taken.contains("right.a"));
        assert!(taken.contains("a"));
    }

    #[test]
    fn join_combined_schema_widens_nullability_for_outer_joins() {
        // LEFT OUTER: right-side columns become nullable.
        let left = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let right = Schema::new(vec![Field::new("b", DataType::Int32, false)]);
        let out = join_combined_schema(&left, &right, JoinType::LeftOuter);
        assert!(!out.fields[0].nullable, "left side stays non-null on LEFT");
        assert!(out.fields[1].nullable, "right side widens on LEFT");

        // RIGHT OUTER: left-side columns become nullable.
        let out = join_combined_schema(&left, &right, JoinType::RightOuter);
        assert!(out.fields[0].nullable, "left side widens on RIGHT");
        assert!(!out.fields[1].nullable, "right side stays non-null on RIGHT");

        // FULL OUTER: both sides become nullable.
        let out = join_combined_schema(&left, &right, JoinType::FullOuter);
        assert!(out.fields[0].nullable);
        assert!(out.fields[1].nullable);

        // INNER / CROSS: nullability untouched.
        let out = join_combined_schema(&left, &right, JoinType::Inner);
        assert!(!out.fields[0].nullable);
        assert!(!out.fields[1].nullable);
    }
}

#[cfg(test)]
mod subquery_node_tests {
    //! Type-check + Debug coverage for the [`Expr::ScalarSubquery`] and
    //! [`Expr::InSubquery`] logical nodes. These pin the single-output-column
    //! contract and the comparability rule independent of the SQL frontend
    //! (which has its own integration coverage in `tests/parser_tests.rs`).
    use super::*;

    /// Outer query schema (the enclosing query the subquery sits inside).
    fn outer_schema() -> Schema {
        Schema::new(vec![
            Field::new("region_id", DataType::Int32, false),
            Field::new("qty", DataType::Int32, false),
        ])
    }

    /// A single-column subquery plan: `Scan(other) -> Project(id)`.
    fn one_col_subquery() -> LogicalPlan {
        let scan = LogicalPlan::Scan {
            table: "other".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("val", DataType::Int32, false),
            ]),
        };
        LogicalPlan::Project {
            input: Box::new(scan),
            exprs: vec![Expr::Column("id".into())],
        }
    }

    /// A two-column subquery plan (illegal for scalar / IN positions).
    fn two_col_subquery() -> LogicalPlan {
        LogicalPlan::Scan {
            table: "other".into(),
            projection: None,
            schema: Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("val", DataType::Int32, false),
            ]),
        }
    }

    #[test]
    fn scalar_subquery_takes_single_output_column_dtype() {
        let e = Expr::ScalarSubquery(Box::new(one_col_subquery()));
        assert_eq!(e.dtype(&outer_schema()).unwrap(), DataType::Int32);
    }

    #[test]
    fn scalar_subquery_multi_column_errors() {
        let e = Expr::ScalarSubquery(Box::new(two_col_subquery()));
        let err = e.dtype(&outer_schema()).expect_err("multi-column scalar");
        assert!(
            format!("{err}").contains("exactly one column"),
            "got: {err}"
        );
    }

    #[test]
    fn in_subquery_typechecks_to_bool() {
        let e = Expr::InSubquery {
            expr: Box::new(Expr::Column("region_id".into())),
            subquery: Box::new(one_col_subquery()),
            negated: false,
        };
        assert_eq!(e.dtype(&outer_schema()).unwrap(), DataType::Bool);
    }

    #[test]
    fn in_subquery_incomparable_probe_errors() {
        // Probe is Bool-typed via a comparison; the subquery column is Int32 —
        // Bool vs Int32 is not comparable.
        let e = Expr::InSubquery {
            expr: Box::new(Expr::Binary {
                op: BinaryOp::Gt,
                left: Box::new(Expr::Column("qty".into())),
                right: Box::new(Expr::Literal(Literal::Int64(0))),
            }),
            subquery: Box::new(one_col_subquery()),
            negated: false,
        };
        let err = e.dtype(&outer_schema()).expect_err("Bool vs Int32 probe");
        assert!(format!("{err}").contains("cannot compare"), "got: {err}");
    }

    #[test]
    fn in_subquery_multi_column_errors() {
        let e = Expr::InSubquery {
            expr: Box::new(Expr::Column("region_id".into())),
            subquery: Box::new(two_col_subquery()),
            negated: true,
        };
        let err = e.dtype(&outer_schema()).expect_err("multi-column IN");
        assert!(
            format!("{err}").contains("exactly one column"),
            "got: {err}"
        );
    }

    #[test]
    fn subquery_nodes_are_debug() {
        // `Debug` is required for the `{plan:?}` diagnostics used across the
        // planner; exercise it so a missing derive can't slip through.
        let scalar = Expr::ScalarSubquery(Box::new(one_col_subquery()));
        let in_sq = Expr::InSubquery {
            expr: Box::new(Expr::Column("region_id".into())),
            subquery: Box::new(one_col_subquery()),
            negated: false,
        };
        assert!(!format!("{scalar:?}").is_empty());
        assert!(!format!("{in_sq:?}").is_empty());
    }
}
