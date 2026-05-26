// SPDX-License-Identifier: Apache-2.0

use thiserror::Error;

#[derive(Debug, Error)]
pub enum BoltError {
    #[error("CUDA driver error: {0}")]
    Cuda(String),

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
