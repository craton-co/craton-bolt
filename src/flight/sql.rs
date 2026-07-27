// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "flight")]

//! Arrow Flight SQL — command layer (feature `flight`).
//!
//! This module is the **command layer** that the Flight server (sibling
//! [`crate::flight::server`]) calls into. It owns the translation between
//! Flight SQL protobuf commands / tickets and the craton-bolt
//! [`Engine`](crate::exec::Engine) public query API, plus the [`BoltError`] →
//! [`tonic::Status`] mapping.
//!
//! ## Unified server ↔ sql ABI
//!
//! The server calls exactly two entry points, both taking the shared engine
//! handle ([`SharedEngine`](crate::flight::SharedEngine), i.e.
//! `Arc<parking_lot::Mutex<Engine>>`) and the raw command bytes (a
//! prost-encoded `google.protobuf.Any`):
//!
//! * [`execute_flight_command`] — decode + execute a command, returning a
//!   materialised [`SqlCommandResult`] (schema + batches + echoed command
//!   bytes for the ticket round-trip). Used by `do_get` to fetch the data;
//!   bounded by the [`FlightSqlConfig`](crate::flight::FlightSqlConfig) caps.
//! * [`schema_for_command`] — decode the command and recover just its result
//!   [`SchemaRef`] (for `get_flight_info` and `get_schema`).
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

use arrow_schema::SchemaRef;

use arrow_flight::sql::{
    Any, CommandGetCatalogs, CommandGetDbSchemas, CommandGetTableTypes, CommandGetTables,
    CommandPreparedStatementQuery, CommandStatementQuery, ProstMessageExt,
};

use prost::Message;
use tonic::Status;

use crate::error::BoltError;
use crate::flight::{FlightSqlConfig, SharedEngine, SqlCommandResult};

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
///
/// # Security
///
/// Parse/plan/type errors describe the *client's own SQL*, so their text is
/// safe (and useful) to return. But CUDA driver codes, IO paths, memory
/// figures, and other internal failure text can leak server-side
/// implementation detail (filesystem layout, driver versions, capacity), so
/// for the `INTERNAL` bucket we return a **generic** client-facing message and
/// log the real error server-side (at `warn`) for operators.
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
        // failures from the client's point of view. Do NOT forward the verbose
        // internal text to the (possibly untrusted) client — log it here and
        // return a generic message.
        other => {
            log::warn!("craton-bolt flight: query failed (internal): {other}");
            Status::internal("internal error executing query")
        }
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

/// Enforce the configured result-size caps on a materialised batch, returning
/// `RESOURCE_EXHAUSTED` (a clean, retry-discouraging gRPC status) when a cap is
/// exceeded.
///
/// # Security
///
/// This is the availability guard: it stops a single (possibly hostile) query
/// from forcing the server to hand back an unbounded result over the wire. The
/// check is *post-materialisation* — the engine has already built the batch in
/// host memory — so it bounds egress / IPC-encode work, not the query's own
/// peak memory. Pair with a sane `max_result_*` and a trusted network.
fn enforce_result_caps(
    batch: &arrow_array::RecordBatch,
    config: &FlightSqlConfig,
) -> Result<(), Status> {
    if let Some(max_rows) = config.max_result_rows {
        let rows = batch.num_rows();
        if rows > max_rows {
            return Err(Status::resource_exhausted(format!(
                "result of {rows} rows exceeds the configured limit of {max_rows} rows"
            )));
        }
    }
    if let Some(max_bytes) = config.max_result_bytes {
        // `get_array_memory_size` lives on the `Array` trait.
        use arrow_array::Array as _;
        // Sum the per-column Arrow in-memory footprint (matches the engine's
        // own `estimate_batch_bytes`). Saturating so the sum cannot wrap.
        let bytes = batch
            .columns()
            .iter()
            .map(|c| c.get_array_memory_size())
            .fold(0usize, |acc, n| acc.saturating_add(n));
        if bytes > max_bytes {
            return Err(Status::resource_exhausted(format!(
                "result of {bytes} bytes exceeds the configured limit of {max_bytes} bytes"
            )));
        }
    }
    Ok(())
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
/// The result is bounded by `config`'s caps (see [`enforce_result_caps`]).
///
/// Returns `INVALID_ARGUMENT` for client SQL errors, `UNIMPLEMENTED` for
/// command types we do not execute, `RESOURCE_EXHAUSTED` for over-cap results,
/// and `INTERNAL` for engine failures (via the [`BoltError`] mapper).
///
/// # Concurrency
///
/// The shared engine is not `Sync`, so execution is serialised behind a
/// [`parking_lot::Mutex`]. The lock is held only for the duration of
/// [`Engine::sql`](crate::exec::Engine::sql) and released immediately after the
/// batch is materialised;
/// encoding happens lock-free in the caller. `parking_lot` does not poison, so
/// a panicking query frees the lock cleanly for the next client.
pub fn execute_flight_command(
    engine: &SharedEngine,
    cmd: &[u8],
    config: &FlightSqlConfig,
) -> Result<SqlCommandResult, Status> {
    let command = decode_command(cmd)?;
    let sql = statement_sql(command)?;

    let batch = {
        // Scope the guard tightly: hold the engine lock only across execution,
        // not across cap-checking or result construction.
        let guard = engine.lock();
        let handle = guard.sql(&sql).map_err(bolt_err_to_status)?;
        handle.into_record_batch()
    };

    enforce_result_caps(&batch, config)?;

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
/// plan-only schema path to avoid executing the query merely to learn its
/// schema.)
///
/// Note: this path does **not** apply the result-size caps — it discards the
/// data and keeps only the schema. Callers that go on to fetch the data
/// (`do_get`) re-run via [`execute_flight_command`], which *does* enforce the
/// caps.
pub fn schema_for_command(engine: &SharedEngine, cmd: &[u8]) -> Result<SchemaRef, Status> {
    let command = decode_command(cmd)?;
    let sql = statement_sql(command)?;

    let guard = engine.lock();
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
        let command = FlightSqlCommand::PreparedStatementQuery(CommandPreparedStatementQuery {
            prepared_statement_handle: prost::bytes::Bytes::new(),
        });
        assert_eq!(
            statement_sql(command).unwrap_err().code(),
            tonic::Code::Unimplemented
        );
    }

    fn sample_batch(rows: usize) -> arrow_array::RecordBatch {
        use arrow_array::Int32Array;
        use arrow_schema::{DataType, Field, Schema};
        use std::sync::Arc;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let data: Vec<i32> = (0..rows as i32).collect();
        arrow_array::RecordBatch::try_new(schema, vec![Arc::new(Int32Array::from(data))]).unwrap()
    }

    #[test]
    fn caps_allow_within_limit() {
        let batch = sample_batch(10);
        let config = FlightSqlConfig {
            max_result_rows: Some(100),
            max_result_bytes: Some(1 << 30),
        };
        assert!(enforce_result_caps(&batch, &config).is_ok());
    }

    #[test]
    fn caps_reject_too_many_rows() {
        let batch = sample_batch(10);
        let config = FlightSqlConfig {
            max_result_rows: Some(5),
            max_result_bytes: None,
        };
        let err = enforce_result_caps(&batch, &config).unwrap_err();
        assert_eq!(err.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn caps_reject_too_many_bytes() {
        let batch = sample_batch(100);
        let config = FlightSqlConfig {
            max_result_rows: None,
            max_result_bytes: Some(1), // 100 i32s far exceed 1 byte
        };
        let err = enforce_result_caps(&batch, &config).unwrap_err();
        assert_eq!(err.code(), tonic::Code::ResourceExhausted);
    }

    #[test]
    fn caps_disabled_when_none() {
        let batch = sample_batch(1_000);
        let config = FlightSqlConfig {
            max_result_rows: None,
            max_result_bytes: None,
        };
        assert!(enforce_result_caps(&batch, &config).is_ok());
    }

    #[test]
    fn internal_errors_are_redacted() {
        // The verbose internal text must NOT reach the client.
        let secret = "C:/secret/path/driver.so CUDA_ERROR_OUT_OF_MEMORY (700)";
        let status = bolt_err_to_status(BoltError::Memory(secret.into()));
        assert_eq!(status.code(), tonic::Code::Internal);
        assert!(
            !status.message().contains("secret"),
            "internal error text leaked to client: {}",
            status.message()
        );
        assert_eq!(status.message(), "internal error executing query");
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
