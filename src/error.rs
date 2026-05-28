// SPDX-License-Identifier: Apache-2.0

use std::ops::Range;

use thiserror::Error;

/// Engine-wide error enum.
///
/// # v0.6 / M5: `#[non_exhaustive]`
///
/// Marked `#[non_exhaustive]` so future minor releases can add new variants
/// (e.g. a richer `Plan` shape carrying its own span) without forcing every
/// downstream crate's exhaustive `match` to break. The trade-off is that any
/// match on `BoltError` *outside* this crate now requires a wildcard arm; we
/// already use `other => …` / `_ => …` everywhere inside the crate, so the
/// addition was a no-op for existing code at the time `#[non_exhaustive]`
/// was introduced (see the v0.6 commit "v0.6: structured errors with optional
/// source spans" for the audit).
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum BoltError {
    /// Free-form CUDA-related error without a driver `CUresult` code.
    ///
    /// Reserved for failures whose origin genuinely has no associated
    /// `CUresult` integer — e.g. `CString::new` rejecting an interior
    /// NUL byte in PTX source, or NVRTC compile diagnostics surfaced
    /// as a free-form string. Stage 5 (M3L5) migrated every site that
    /// did have a `CUresult` to use [`BoltError::CudaWithCode`] so
    /// downstream pattern-matching code (e.g. `mem_pool::is_oom_error`)
    /// can recognise the code directly.
    ///
    /// **New code SHOULD prefer [`BoltError::CudaWithCode`] whenever a
    /// `CUresult` integer is available.** This variant is intentionally
    /// not `#[deprecated]` — a handful of legitimate "no code"
    /// callsites remain — but extending the legacy variant to a new
    /// site that DOES have a code is a regression.
    #[error("CUDA driver error: {0}")]
    Cuda(String),

    /// Driver-API error carrying the raw `CUresult` integer alongside
    /// the human-readable message. Emitted by [`crate::cuda::cuda_sys::check`]
    /// for every non-success `CUresult`. The Display impl is wire-compatible
    /// with the old `Cuda(format!("CUDA driver error {code}: {message}"))`
    /// shape so any callers that pattern-match on `other => other.to_string()`
    /// (e.g. `jit_compiler::inner_msg`) keep working unchanged.
    ///
    /// Pattern-match on `{ code, .. }` to recognise specific driver errors
    /// without parsing a formatted string — `mem_pool` uses this for the
    /// `CUDA_ERROR_OUT_OF_MEMORY = 2` recovery hook (Stage 4).
    #[error("CUDA driver error {code}: {message}")]
    CudaWithCode {
        /// Raw `CUresult` integer as returned by the CUDA driver.
        /// `CUDA_ERROR_OUT_OF_MEMORY` is `2`. See the CUDA Driver API
        /// reference for the full enum.
        code: i32,
        /// Human-readable description, typically the output of
        /// `cuGetErrorString`. May be `"unknown CUDA error <code>"`
        /// if the driver did not provide a string.
        message: String,
    },

    #[error("SQL parse error: {0}")]
    Sql(String),

    /// SQL parse / planning error carrying an optional source span
    /// (byte offsets into the original SQL string).
    ///
    /// Introduced in v0.6 (M5) so editor / IDE consumers can underline the
    /// offending token. The legacy [`BoltError::Sql`] variant is preserved
    /// because most internal call sites have only a human-readable string
    /// and no positional information to attach. Prefer this variant whenever
    /// a span IS available (for example, the SQL frontend's
    /// `parse_error_to_bolt_error` mapper extracts the `at Line: N, Column: M`
    /// suffix that sqlparser appends to its `Display` output and converts it
    /// into a byte-offset range).
    ///
    /// The `Display` impl renders the span as a bracketed `[start..end]`
    /// suffix after the message so callers that already render the string
    /// (logs, tests) still surface the location without having to teach
    /// every renderer about [`Self::span`].
    #[error("SQL parse error: {msg} [{}..{}]", .span.start, .span.end)]
    SqlWithSpan {
        /// Human-readable error message (without the inline location).
        msg: String,
        /// Half-open byte range into the original SQL string that points at
        /// the offending token. Empty ranges (`start == end`) are legal and
        /// mean "the error is at this position but has no extent" — typical
        /// for end-of-input errors.
        span: Range<usize>,
    },

    #[error("plan error: {0}")]
    Plan(String),

    #[error("type error: {0}")]
    Type(String),

    #[error("memory error: {0}")]
    Memory(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// Typed marker for "GPU path declined due to a capacity / sizing
    /// mismatch — please retry on the host". Emitted by the GPU hash-join
    /// when the kernel's match counter overshoots the pre-sized output
    /// buffer (cartesian explosion, lossy-fold false-positive blow-up,
    /// duplicate-build-key invariant violation, etc.). The host hash-join
    /// handles the same input fine, so callers that have a host fallback
    /// (`try_gpu_inner_join`, `try_gpu_outer_join`) MAP this variant to
    /// their "fall back to host" signal (`Ok(None)`) — they do this for
    /// any `Err(_)` today, the variant is here so the pattern-match is
    /// type-safe rather than string-parsed.
    #[error("GPU capacity exceeded: {0}")]
    GpuCapacity(String),

    #[error("{0}")]
    Other(String),
}

impl From<sqlparser::parser::ParserError> for BoltError {
    fn from(e: sqlparser::parser::ParserError) -> Self {
        // The legacy `From` impl predates the span-aware variant and is
        // kept on the unspanned [`BoltError::Sql`] shape so existing
        // callers that use `?` against a `ParserError` without holding
        // the original SQL string still compile and behave the same. The
        // SQL frontend's `parse_error_to_bolt_error` helper produces
        // [`BoltError::SqlWithSpan`] when the original SQL *is* in scope.
        BoltError::Sql(format!("{}", e))
    }
}

impl BoltError {
    /// Returns the source span attached to this error, if any.
    ///
    /// Currently only [`BoltError::SqlWithSpan`] carries a span; every
    /// other variant returns `None`. The accessor is a stable API surface
    /// independent of the variant set — adding a new span-bearing variant
    /// in a future release just means extending the `match` arm here, not
    /// breaking downstream callers that already pattern-match on the
    /// resulting `Option<Range<usize>>`.
    pub fn span(&self) -> Option<Range<usize>> {
        match self {
            BoltError::SqlWithSpan { span, .. } => Some(span.clone()),
            // `_` here is correct for two reasons:
            //  1. `BoltError` is `#[non_exhaustive]`, so even within this
            //     crate we want a future-proof default for any new
            //     non-spanning variant.
            //  2. None of the other current variants ([`Cuda`],
            //     [`CudaWithCode`], [`Sql`], [`Plan`], [`Type`],
            //     [`Memory`], [`Io`], [`GpuCapacity`], [`Other`]) carries
            //     positional information.
            _ => None,
        }
    }
}

pub type BoltResult<T> = Result<T, BoltError>;

#[cfg(test)]
mod tests {
    //! Stage 4 — verify `CudaWithCode`'s pattern-match shape and that its
    //! Display rendering stays wire-compatible with the legacy formatted
    //! `Cuda(String)` shape that earlier consumers (e.g. `mem_pool`'s
    //! pre-Stage-4 prefix matcher, `jit_compiler::inner_msg`) relied on.
    use super::*;

    #[test]
    fn cuda_with_code_matches_by_code() {
        let e = BoltError::CudaWithCode {
            code: 2,
            message: "out of memory".to_string(),
        };
        // Direct, type-safe pattern match — no string parsing.
        let is_oom = matches!(&e, BoltError::CudaWithCode { code: 2, .. });
        assert!(is_oom, "should match code 2 directly");

        // And the Display form keeps the historical "CUDA driver error
        // <code>: <message>" shape so any caller that still walks the
        // formatted output stays compatible.
        let rendered = e.to_string();
        assert_eq!(rendered, "CUDA driver error 2: out of memory");
    }

    #[test]
    fn legacy_cuda_string_variant_still_present() {
        // Backwards-compat: the freeform Cuda(String) variant remains so
        // cudarc-backend errors and PTX compilation errors continue to
        // build and behave as before.
        let e = BoltError::Cuda("freeform message".into());
        assert_eq!(e.to_string(), "CUDA driver error: freeform message");
        assert!(matches!(e, BoltError::Cuda(_)));
    }

    // ----- v0.6 / M5: SqlWithSpan variant + span() accessor ---------------

    /// The new span-bearing variant pattern-matches on `{ msg, span }` and
    /// renders both the message and a bracketed `[start..end]` location in
    /// its Display form. Editor / IDE consumers consume `span` via the
    /// structured accessor below; log lines see the rendered location for
    /// free.
    #[test]
    fn sql_with_span_renders_message_and_range() {
        let e = BoltError::SqlWithSpan {
            msg: "expected expression, found EOF".to_string(),
            span: 12..15,
        };
        let rendered = e.to_string();
        assert_eq!(
            rendered,
            "SQL parse error: expected expression, found EOF [12..15]"
        );
        // Pattern-match works as the docs promise.
        match &e {
            BoltError::SqlWithSpan { msg, span } => {
                assert_eq!(msg, "expected expression, found EOF");
                assert_eq!(span.start, 12);
                assert_eq!(span.end, 15);
            }
            other => panic!("expected SqlWithSpan, got {other:?}"),
        }
    }

    /// `span()` returns the inner range for `SqlWithSpan` and `None` for
    /// every other variant. This is the type-safe public hook consumers
    /// use to drive a squiggly underline without parsing the rendered
    /// string.
    #[test]
    fn span_accessor_returns_range_for_sql_with_span_only() {
        let with_span = BoltError::SqlWithSpan {
            msg: "bad token".into(),
            span: 4..9,
        };
        assert_eq!(with_span.span(), Some(4..9));

        // Every other variant returns None. Cover representative cases to
        // make sure the wildcard arm in `span()` doesn't silently swallow
        // a future variant we forget to wire up — adding such a variant
        // means re-running this test and either (a) adjusting the
        // expectation to `Some` or (b) confirming `None` is correct.
        assert_eq!(BoltError::Sql("legacy".into()).span(), None);
        assert_eq!(BoltError::Plan("plan".into()).span(), None);
        assert_eq!(BoltError::Type("type".into()).span(), None);
        assert_eq!(BoltError::Memory("mem".into()).span(), None);
        assert_eq!(BoltError::GpuCapacity("gc".into()).span(), None);
        assert_eq!(BoltError::Other("other".into()).span(), None);
        assert_eq!(BoltError::Cuda("c".into()).span(), None);
        assert_eq!(
            BoltError::CudaWithCode {
                code: 2,
                message: "oom".into(),
            }
            .span(),
            None
        );
    }

    /// Empty spans (`start == end`) are legal — they encode "the error is
    /// at this position but has no extent", e.g. an end-of-input error.
    /// Verify they round-trip cleanly through `span()` and Display.
    #[test]
    fn sql_with_span_empty_range_is_legal() {
        let e = BoltError::SqlWithSpan {
            msg: "unexpected end of input".into(),
            span: 7..7,
        };
        assert_eq!(e.span(), Some(7..7));
        assert_eq!(
            e.to_string(),
            "SQL parse error: unexpected end of input [7..7]"
        );
    }

    /// The legacy `From<ParserError> for BoltError` impl keeps producing
    /// the unspanned `Sql` variant — back-compat for any `?` site that has
    /// only the parser error in scope (no SQL string to compute byte
    /// offsets against). Span-aware mapping lives in the SQL frontend's
    /// `parse_error_to_bolt_error` helper, not in this `From` impl.
    #[test]
    fn from_parser_error_keeps_unspanned_shape() {
        let pe = sqlparser::parser::ParserError::ParserError(
            "test failure at Line: 1, Column: 3".into(),
        );
        let be: BoltError = pe.into();
        assert!(matches!(be, BoltError::Sql(_)));
        assert_eq!(be.span(), None);
    }
}
