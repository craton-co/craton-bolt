// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BoltError {
    /// Free-form CUDA-related error without a driver `CUresult` code.
    /// Used for cudarc backend errors, PTX compilation issues, and other
    /// CUDA-adjacent failures whose origin is not a `cuGetErrorString`-
    /// translatable driver call. The legacy variant retained for
    /// backwards compatibility with consumers that still match on the
    /// formatted-string shape.
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

    #[error("plan error: {0}")]
    Plan(String),

    #[error("type error: {0}")]
    Type(String),

    #[error("memory error: {0}")]
    Memory(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    Other(String),
}

impl From<sqlparser::parser::ParserError> for BoltError {
    fn from(e: sqlparser::parser::ParserError) -> Self {
        BoltError::Sql(format!("{}", e))
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
}
