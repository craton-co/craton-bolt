// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

#[derive(Debug, Error)]
pub enum JavelinError {
    #[error("CUDA driver error ({code}): {msg}")]
    Cuda { code: i32, msg: String },

    // FIXME(orchestrator): variant slated for removal (Javelin uses cuModuleLoadData,
    // not NVRTC), but still referenced by src/jit/jit_compiler.rs (4 call sites).
    // Migrate those to a more appropriate variant (e.g. `Cuda` or `Other`) before
    // dropping this variant.
    #[error("NVRTC compile error: {0}")]
    Nvrtc(String),

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

impl From<sqlparser::parser::ParserError> for JavelinError {
    fn from(e: sqlparser::parser::ParserError) -> Self {
        JavelinError::Sql(format!("{}", e))
    }
}

pub type JavelinResult<T> = Result<T, JavelinError>;
