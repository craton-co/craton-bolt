// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "flight")]

//! Arrow Flight SQL — command layer (feature `flight`).
//!
//! This module is the **command layer** that the Flight server (sibling
//! [`crate::flight::server`]) calls into. It owns the translation between
//! Flight SQL protobuf commands / tickets and the craton-bolt [`Engine`]
//! public query API, plus the [`BoltError`] → [`tonic::Status`] mapping.
//!
//! ## Unified server ↔ sql ABI
//!
//! The server calls exactly two entry points, both taking the shared,
//! mutex-guarded engine handle ([`Arc<Mutex<Engine>>`]) and the raw command
//! bytes (a prost-encoded `google.protobuf.Any`):
//!
//! * [`execute_flight_command`] — decode + execute a command, returning a
//!   materialised [`SqlCommandResult`] (schema + batches + echoed command
//!   bytes for the ticket round-trip). Used by both `get_flight_info` (to
//!   learn the schema / row count) and `do_get` (to fetch the data).
//! * [`schema_for_command`] — decode the command and recover just its result
//!   [`SchemaRef`] (for `get_schema`).
//!
//! Everything else — the typed command enum, the `Any` matching, the ticket
//! codec, the [`BoltError`] mapper — is internal to this module.
//!
//! ## Supported commands
//!
//! Only ad-hoc SQL (`CommandStatementQuery`) is executed today. The ticket
//! produced by `get_flight_info` carries the original command bytes verbatim
//! (see [`crate::flight::SqlCommandResult::ticket_cmd`]), so `do_get` re-runs
//! the same command — keeping the server stateless. Other command types
//! (prepared statements, catalog metadata) decode to
//! [`FlightSqlCommand::Unsupported`] / their typed variants and currently
//! return `UNIMPLEMENTED`.

use std::sync::{Arc, Mutex};

use arrow_schema::SchemaRef;

use arrow_flight::sql::{
    Any, CommandGetCatalogs, CommandGetDbSchemas, CommandGetTableTypes, CommandGetTables,
    CommandPreparedStatementQuery, CommandStatementQuery, ProstMessageExt,
};

use prost::Message;
use tonic::Status;

use crate::error::BoltError;
use crate::exec::Engine;
use crate::flight::SqlCommandResult;

/// A decoded Flight SQL command extracted from a [`FlightDescriptor`]'s `cmd`
/// (a `prost` [`Any`]).
///
/// [`FlightDescriptor`]: arrow_flight::FlightDescriptor
///
/// Only the variants the command layer recognises are enumerated; anything
/// else surfaces as [`FlightSqlCommand::Unsupported`] carrying the protobuf
/// `type_url` so the caller can return a precise `unimplemented` [`Status`].
#[derive(Debug, Clone)]
pub(crate) enum FlightSqlCommand {
    /// An ad-hoc SQL query (`CommandStatementQuery`). The core supported path.
    StatementQuery(CommandStatementQuery),
    /// A prepared-statement execution (`CommandPreparedStatementQuery`).
    PreparedStatementQuery(CommandPreparedStatementQuery),
    /// Metadata: list catalogs.
    GetCatalogs(CommandGetCatalogs),
    /// Metadata: list DB schemas.
    GetDbSchemas(CommandGetDbSchemas),
    /// Metadata: list tables.
    GetTables(CommandGetTables),
    /// Metadata: list table types.
    GetTableTypes(CommandGetTableTypes),
    /// A command whose protobuf `type_url` we do not (yet) handle.
    Unsupported {
        /// The protobuf `type_url` of the `Any` we failed to match.
        type_url: String,
    },
}

/// Map a [`BoltError`] onto a [`tonic::Status`] with a sensible gRPC code.
///
/// * Parse / plan / type errors → `INVALID_ARGUMENT` (the client's SQL).
/// * `Unsupported` → `UNIMPLEMENTED`.
/// * Everything else (CUDA, memory, IO, capacity, other) → `INTERNAL`.
fn bolt_err_to_status(err: BoltError) -> Status {
    match err {
        BoltError::Sql(msg) | BoltError::Plan(msg) | BoltError::Type(msg) => {
            Status::invalid_argument(msg)
        }
        BoltError::SqlWithSpan { msg, span } => {
            Status::invalid_argument(format!("{msg} [{}..{}]", span.start, span.end))
        }
        BoltError::Unsupported(msg) => Status::unimplemented(msg),
        // CUDA / memory / IO / GPU-capacity / freeform are server-side
        // failures from the client's point of view.
        other => Status::internal(other.to_string()),
    }
}

/// Decode the `cmd` bytes of a [`FlightDescriptor`] into a typed
/// [`FlightSqlCommand`].
///
/// [`FlightDescriptor`]: arrow_flight::FlightDescriptor
///
/// The bytes are a serialized `google.protobuf.Any` (Flight SQL's wrapper);
/// we decode it then match on the inner message type. A malformed `Any`
/// returns an `INVALID_ARGUMENT` [`Status`]; an unrecognised inner type yields
/// [`FlightSqlCommand::Unsupported`] (not an error — the caller decides the
/// response code).
fn decode_command(cmd: &[u8]) -> Result<FlightSqlCommand, Status> {
    let any = Any::decode(cmd)
        .map_err(|e| Status::invalid_argument(format!("invalid Flight SQL command Any: {e}")))?;
    decode_any(&any)
}

/// Decode an already-unpacked [`Any`] into a [`FlightSqlCommand`].
fn decode_any(any: &Any) -> Result<FlightSqlCommand, Status> {
    // `Any::unpack::<T>()` returns `Ok(None)` when the type_url does not match
    // `T`, `Ok(Some(msg))` on a match, and `Err` only on a corrupt payload of
    // the matching type. We try each known type in turn.
    macro_rules! try_unpack {
        ($ty:ty, $variant:expr) => {
            match any.unpack::<$ty>() {
                Ok(Some(msg)) => return Ok($variant(msg)),
                Ok(None) => {}
                Err(e) => {
                    return Err(Status::invalid_argument(format!(
                        "corrupt Flight SQL command payload: {e}"
                    )))
                }
            }
        };
    }

    try_unpack!(CommandStatementQuery, FlightSqlCommand::StatementQuery);
    try_unpack!(
        CommandPreparedStatementQuery,
        FlightSqlCommand::PreparedStatementQuery
    );
    try_unpack!(CommandGetCatalogs, FlightSqlCommand::GetCatalogs);
    try_unpack!(CommandGetDbSchemas, FlightSqlCommand::GetDbSchemas);
    try_unpack!(CommandGetTables, FlightSqlCommand::GetTables);
    try_unpack!(CommandGetTableTypes, FlightSqlCommand::GetTableTypes);

    Ok(FlightSqlCommand::Unsupported {
        type_url: any.type_url.clone(),
    })
}

/// Extract the ad-hoc SQL string from a decoded command, or return a precise
/// `UNIMPLEMENTED` [`Status`] for command types we do not execute yet.
fn statement_sql(command: FlightSqlCommand) -> Result<String, Status> {
    match command {
        FlightSqlCommand::StatementQuery(c) => Ok(c.query),
        FlightSqlCommand::PreparedStatementQuery(_) => Err(Status::unimplemented(
            "prepared statements are not yet supported; use CommandStatementQuery (ad-hoc SQL)",
        )),
        FlightSqlCommand::GetCatalogs(_)
        | FlightSqlCommand::GetDbSchemas(_)
        | FlightSqlCommand::GetTables(_)
        | FlightSqlCommand::GetTableTypes(_) => Err(Status::unimplemented(
            "Flight SQL catalog-metadata commands are not yet supported",
        )),
        FlightSqlCommand::Unsupported { type_url } => Err(Status::unimplemented(format!(
            "unsupported Flight SQL command type_url: {type_url}"
        ))),
    }
}

/// Lock the shared engine, mapping a poisoned mutex to an `INTERNAL`
/// [`Status`] instead of panicking inside a gRPC handler.
fn lock_engine(
    engine: &Arc<Mutex<Engine>>,
) -> Result<std::sync::MutexGuard<'_, Engine>, Status> {
    engine
        .lock()
        .map_err(|_| Status::internal("query engine mutex poisoned"))
}

/// Decode + execute a Flight SQL command against the shared engine and return
/// the materialised result.
///
/// The result carries the original `cmd` bytes verbatim
/// ([`SqlCommandResult::ticket_cmd`]) so `get_flight_info` can echo them into
/// the ticket and the follow-up `do_get` re-runs the same command. The engine
/// materialises eagerly into a single [`RecordBatch`], so `batches` has length
/// 0 or 1 today; the `Vec` keeps the ABI stable if the engine grows
/// multi-batch streaming.
///
/// [`RecordBatch`]: arrow_array::RecordBatch
///
/// Returns `INVALID_ARGUMENT` for client SQL errors, `UNIMPLEMENTED` for
/// command types we do not execute, and `INTERNAL` for engine failures (via
/// the [`BoltError`] mapper).
pub fn execute_flight_command(
    engine: &Arc<Mutex<Engine>>,
    cmd: &[u8],
) -> Result<SqlCommandResult, Status> {
    let command = decode_command(cmd)?;
    let sql = statement_sql(command)?;

    let guard = lock_engine(engine)?;
    let handle = guard.sql(&sql).map_err(bolt_err_to_status)?;
    let batch = handle.into_record_batch();
    drop(guard);

    let schema = batch.schema();
    Ok(SqlCommandResult {
        schema,
        batches: vec![batch],
        ticket_cmd: bytes::Bytes::copy_from_slice(cmd),
    })
}

/// Decode a Flight SQL command and recover just its result [`SchemaRef`].
///
/// There is no cheap "plan-only, get schema" entry point on the engine today,
/// so we execute the query once to recover the result Arrow schema. (TODO: a
/// plan-only schema path to avoid executing twice across
/// `get_schema` + `do_get`.)
pub fn schema_for_command(
    engine: &Arc<Mutex<Engine>>,
    cmd: &[u8],
) -> Result<SchemaRef, Status> {
    let command = decode_command(cmd)?;
    let sql = statement_sql(command)?;

    let guard = lock_engine(engine)?;
    let handle = guard.sql(&sql).map_err(bolt_err_to_status)?;
    Ok(handle.record_batch().schema())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_command_recognises_statement_query() {
        let cmd = CommandStatementQuery {
            query: "SELECT 1".to_string(),
            transaction_id: None,
        };
        let bytes = cmd.as_any().encode_to_vec();
        match decode_command(&bytes).expect("decode") {
            FlightSqlCommand::StatementQuery(c) => assert_eq!(c.query, "SELECT 1"),
            other => panic!("expected StatementQuery, got {other:?}"),
        }
    }

    #[test]
    fn decode_command_reports_unsupported_type_url() {
        // A GetSqlInfo command is a valid Flight SQL Any we deliberately do
        // not enumerate, so it must surface as Unsupported with its type_url.
        let cmd = arrow_flight::sql::CommandGetSqlInfo { info: vec![] };
        let bytes = cmd.as_any().encode_to_vec();
        match decode_command(&bytes).expect("decode") {
            FlightSqlCommand::Unsupported { type_url } => {
                assert!(type_url.contains("CommandGetSqlInfo"), "got {type_url}");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn decode_command_rejects_garbage() {
        // Not a valid protobuf Any.
        let err = decode_command(&[0xFF, 0x00, 0xAB, 0xCD]).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

    #[test]
    fn statement_sql_unwraps_query() {
        let command = FlightSqlCommand::StatementQuery(CommandStatementQuery {
            query: "SELECT 42".to_string(),
            transaction_id: None,
        });
        assert_eq!(statement_sql(command).unwrap(), "SELECT 42");
    }

    #[test]
    fn statement_sql_rejects_prepared() {
        let command =
            FlightSqlCommand::PreparedStatementQuery(CommandPreparedStatementQuery {
                prepared_statement_handle: prost::bytes::Bytes::new(),
            });
        assert_eq!(
            statement_sql(command).unwrap_err().code(),
            tonic::Code::Unimplemented
        );
    }

    #[test]
    fn bolt_err_mapping_codes() {
        assert_eq!(
            bolt_err_to_status(BoltError::Sql("bad".into())).code(),
            tonic::Code::InvalidArgument
        );
        assert_eq!(
            bolt_err_to_status(BoltError::Plan("bad".into())).code(),
            tonic::Code::InvalidArgument
        );
        assert_eq!(
            bolt_err_to_status(BoltError::Unsupported("nope".into())).code(),
            tonic::Code::Unimplemented
        );
        assert_eq!(
            bolt_err_to_status(BoltError::Memory("oom".into())).code(),
            tonic::Code::Internal
        );
        assert_eq!(
            bolt_err_to_status(BoltError::SqlWithSpan {
                msg: "x".into(),
                span: 1..2
            })
            .code(),
            tonic::Code::InvalidArgument
        );
    }
}
