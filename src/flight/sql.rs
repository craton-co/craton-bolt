// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "flight")]

//! Arrow Flight SQL — command handlers (feature `flight`).
//!
//! This module is the **command layer** that the Flight server (sibling
//! [`crate::flight::server`], task B1a) calls into. It owns the translation
//! between Flight SQL protobuf commands / tickets and the craton-bolt
//! [`Engine`] public query API, plus the [`BoltError`] → [`tonic::Status`]
//! error mapping.
//!
//! ## What lives here
//!
//! * [`get_flight_info_for_statement`] — handles `GetFlightInfo` for a
//!   `CommandStatementQuery` (an ad-hoc SQL string). It plans/executes the
//!   query just enough to recover the result Arrow [`Schema`], then returns a
//!   [`FlightInfo`] whose single endpoint carries a [`Ticket`] that round-trips
//!   the SQL back to `DoGet`.
//! * [`execute_ticket`] — handles `DoGet`: decodes the ticket back to a SQL
//!   string and runs it through [`Engine::sql`], returning the result
//!   [`RecordBatch`]es for the server to stream via the sibling `encode`
//!   helpers (task B1c).
//! * [`decode_command`] / [`FlightSqlCommand`] — unpack a `prost` [`Any`] from
//!   a [`FlightDescriptor`]'s `cmd` bytes into a typed command enum, so the
//!   server can dispatch without re-implementing the protobuf matching.
//! * [`bolt_err_to_status`] — the single [`BoltError`] → [`tonic::Status`]
//!   mapper used across the layer.
//!
//! ## Ticket format
//!
//! We use Flight SQL's [`TicketStatementQuery`] as the ticket payload: its
//! `statement_handle` bytes carry the raw UTF-8 SQL. This keeps the server
//! stateless — no prepared-statement registry is required for ad-hoc queries —
//! while still using the canonical Flight SQL ticket message so generic Flight
//! SQL clients (JDBC/ADBC) interoperate. [`encode_statement_ticket`] /
//! [`decode_ticket`] are the inverse pair.
//!
//! ## ABI coordination (see `integration_notes`)
//!
//! The server (B1a) is expected to call, per RPC:
//! * `get_flight_info` → [`decode_command`] then, for the
//!   [`FlightSqlCommand::StatementQuery`] arm, [`get_flight_info_for_statement`].
//! * `do_get` → [`execute_ticket`] with the raw `Ticket::ticket` bytes, then
//!   feed the returned `Vec<RecordBatch>` to B1c's encoder.
//!
//! All of these take `&mut Engine` even though [`Engine::sql`] only needs
//! `&self`, so the server can hold the engine behind a `Mutex`/owned handle
//! without an API churn if a future statement type needs `&mut`.

use arrow_array::RecordBatch;
use arrow_schema::{Schema, SchemaRef};

use arrow_flight::sql::{
    Any, CommandGetCatalogs, CommandGetDbSchemas, CommandGetTableTypes, CommandGetTables,
    CommandPreparedStatementQuery, CommandStatementQuery, ProstMessageExt, TicketStatementQuery,
};
use arrow_flight::{FlightDescriptor, FlightEndpoint, FlightInfo, Ticket};

use prost::bytes::Bytes;
use prost::Message;
use tonic::Status;

use crate::error::BoltError;
use crate::exec::Engine;

/// A decoded Flight SQL command extracted from a [`FlightDescriptor`]'s `cmd`
/// (a `prost` [`Any`]).
///
/// Only the variants the command layer actually understands are enumerated;
/// anything else surfaces as [`FlightSqlCommand::Unsupported`] carrying the
/// protobuf `type_url` so the caller can return a precise `unimplemented`
/// [`Status`].
#[derive(Debug, Clone)]
pub enum FlightSqlCommand {
    /// An ad-hoc SQL query (`CommandStatementQuery`). The core supported path.
    StatementQuery(CommandStatementQuery),
    /// A prepared-statement execution (`CommandPreparedStatementQuery`).
    /// Minimally recognised; see [`get_flight_info_for_prepared_statement`].
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
/// * Parse / plan / type errors → `INVALID_ARGUMENT` (the client's SQL is at
///   fault).
/// * `Unsupported` → `UNIMPLEMENTED`.
/// * Everything else (CUDA, memory, IO, capacity, other) → `INTERNAL`.
pub fn bolt_err_to_status(err: BoltError) -> Status {
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
/// The bytes are a serialized `google.protobuf.Any` (Flight SQL's wrapper);
/// we decode it then match on the inner message type. A malformed `Any`
/// returns an `INVALID_ARGUMENT` [`Status`]; an unrecognised inner type yields
/// [`FlightSqlCommand::Unsupported`] (not an error — the caller decides the
/// response code).
pub fn decode_command(cmd: &[u8]) -> Result<FlightSqlCommand, Status> {
    let any = Any::decode(cmd)
        .map_err(|e| Status::invalid_argument(format!("invalid Flight SQL command Any: {e}")))?;
    decode_any(&any)
}

/// Decode an already-unpacked [`Any`] into a [`FlightSqlCommand`].
///
/// Split out from [`decode_command`] so callers that already hold an `Any`
/// (e.g. from a `DoAction` body) can reuse the matching.
pub fn decode_any(any: &Any) -> Result<FlightSqlCommand, Status> {
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

/// Build the [`Ticket`] payload for an ad-hoc SQL statement.
///
/// Wraps the raw SQL bytes in a [`TicketStatementQuery`] (`statement_handle`)
/// and serialises it through the Flight SQL `Any` envelope so a generic Flight
/// SQL client treats it as an opaque ticket. [`decode_ticket`] is the inverse.
pub fn encode_statement_ticket(sql: &str) -> Ticket {
    let tsq = TicketStatementQuery {
        statement_handle: Bytes::copy_from_slice(sql.as_bytes()),
    };
    // `as_any()` wraps it in a protobuf `Any` with the canonical type_url;
    // `.encode_to_vec()` serialises that envelope.
    let any = tsq.as_any();
    Ticket {
        ticket: Bytes::from(any.encode_to_vec()),
    }
}

/// Decode a [`Ticket`]'s raw bytes back to the SQL string they carry.
///
/// Accepts two shapes for robustness:
/// 1. The canonical [`TicketStatementQuery`] wrapped in an `Any` (what
///    [`encode_statement_ticket`] produces), whose `statement_handle` is the
///    UTF-8 SQL.
/// 2. A bare UTF-8 SQL string (a convenience fallback for hand-rolled clients
///    that put the SQL directly in the ticket).
pub fn decode_ticket(ticket: &[u8]) -> Result<String, Status> {
    // First, try the canonical Any(TicketStatementQuery) envelope.
    if let Ok(any) = Any::decode(ticket) {
        if let Ok(Some(tsq)) = any.unpack::<TicketStatementQuery>() {
            return String::from_utf8(tsq.statement_handle.to_vec()).map_err(|e| {
                Status::invalid_argument(format!("ticket SQL is not valid UTF-8: {e}"))
            });
        }
    }
    // Fallback: treat the raw bytes as UTF-8 SQL.
    String::from_utf8(ticket.to_vec())
        .map_err(|e| Status::invalid_argument(format!("ticket is neither a valid Flight SQL TicketStatementQuery nor UTF-8 SQL: {e}")))
}

/// Handle `GetFlightInfo` for a [`CommandStatementQuery`].
///
/// Strategy: there is no cheap "plan-only, get schema" entry point on the
/// engine today, so we execute the query once to recover the result Arrow
/// schema (the [`Engine`] materialises eagerly into a single [`RecordBatch`]).
/// We then build a [`FlightInfo`] whose lone endpoint carries a ticket that
/// round-trips the SQL to `DoGet`. The query is executed again in
/// [`execute_ticket`] during `DoGet` — acceptable for a correct first cut;
/// see the TODO about a plan-only schema path to avoid double execution.
///
/// Returns `INVALID_ARGUMENT` for client SQL errors and `INTERNAL` for engine
/// failures (via [`bolt_err_to_status`]).
pub fn get_flight_info_for_statement(
    cmd: CommandStatementQuery,
    engine: &mut Engine,
) -> Result<FlightInfo, Status> {
    let sql = cmd.query;

    // Recover the result schema by executing the query. The handle wraps a
    // single materialised RecordBatch whose schema is what we advertise.
    let handle = engine.sql(&sql).map_err(bolt_err_to_status)?;
    let schema: SchemaRef = handle.record_batch().schema();

    let ticket = encode_statement_ticket(&sql);
    let endpoint = FlightEndpoint::new().with_ticket(ticket);

    // Re-wrap the original command as the descriptor `cmd` so the FlightInfo
    // echoes what the client asked for (per the Flight SQL convention).
    let descriptor = FlightDescriptor::new_cmd(
        CommandStatementQuery {
            query: sql,
            transaction_id: cmd.transaction_id,
        }
        .as_any()
        .encode_to_vec(),
    );

    build_flight_info(schema.as_ref(), descriptor, endpoint)
}

/// Minimal `GetFlightInfo` for a [`CommandPreparedStatementQuery`].
///
/// Prepared statements require a server-side handle registry (created via the
/// `ActionCreatePreparedStatementRequest` DoAction) that this stateless first
/// cut does not implement. We return `UNIMPLEMENTED` with a clear message.
///
/// TODO(B1b): wire a prepared-statement registry (handle → SQL + bound
/// params) once the server (B1a) grows a `DoAction` dispatch for
/// `ActionCreatePreparedStatementRequest` / `...Close`.
pub fn get_flight_info_for_prepared_statement(
    _cmd: CommandPreparedStatementQuery,
    _engine: &mut Engine,
) -> Result<FlightInfo, Status> {
    Err(Status::unimplemented(
        "prepared statements are not yet supported; use CommandStatementQuery (ad-hoc SQL)",
    ))
}

/// Handle `DoGet`: decode the ticket to SQL, run it, return the result
/// batches.
///
/// The returned `Vec<RecordBatch>` is what the server feeds to the B1c
/// encoder for streaming. The engine materialises into a single batch today,
/// so the vector has length 0 or 1; returning a `Vec` keeps the ABI stable if
/// the engine grows multi-batch streaming.
pub fn execute_ticket(ticket: &[u8], engine: &mut Engine) -> Result<Vec<RecordBatch>, Status> {
    let sql = decode_ticket(ticket)?;
    let handle = engine.sql(&sql).map_err(bolt_err_to_status)?;
    let batch = handle.into_record_batch();
    // Empty result still carries a schema via the FlightInfo; an empty batch
    // is fine to return (the encoder emits the schema message regardless).
    Ok(vec![batch])
}

/// Assemble a [`FlightInfo`] from a result schema, a descriptor, and a single
/// endpoint.
///
/// Centralises the `try_with_schema` / `with_endpoint` / `with_descriptor`
/// builder chain and maps the IPC schema-encoding error (which can fail if the
/// schema carries un-encodable metadata) to an `INTERNAL` [`Status`].
fn build_flight_info(
    schema: &Schema,
    descriptor: FlightDescriptor,
    endpoint: FlightEndpoint,
) -> Result<FlightInfo, Status> {
    FlightInfo::new()
        .try_with_schema(schema)
        .map_err(|e| Status::internal(format!("failed to encode result schema: {e}")))
        .map(|info| {
            info.with_descriptor(descriptor)
                .with_endpoint(endpoint)
                // Row/byte counts are unknown until execution; -1 means
                // "unknown" per the Flight protocol.
                .with_total_records(-1)
                .with_total_bytes(-1)
                .with_ordered(false)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statement_ticket_round_trips_sql() {
        let sql = "SELECT 1 AS a, 'hi' AS b";
        let ticket = encode_statement_ticket(sql);
        let decoded = decode_ticket(&ticket.ticket).expect("decode");
        assert_eq!(decoded, sql);
    }

    #[test]
    fn decode_ticket_accepts_bare_utf8_sql() {
        let sql = "SELECT 42";
        let decoded = decode_ticket(sql.as_bytes()).expect("decode bare");
        assert_eq!(decoded, sql);
    }

    #[test]
    fn decode_ticket_rejects_invalid_utf8() {
        // 0xFF is never valid UTF-8 and is not a valid Any prefix either.
        let err = decode_ticket(&[0xFF, 0xFE, 0xFD]).unwrap_err();
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }

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

    #[test]
    #[ignore = "constructs a real CUDA-backed Engine; run with --ignored on a GPU host"]
    fn prepared_statement_get_info_is_unimplemented() {
        let mut engine = Engine::new().expect("engine");
        let cmd = CommandPreparedStatementQuery {
            prepared_statement_handle: Bytes::new(),
        };
        let err = get_flight_info_for_prepared_statement(cmd, &mut engine).unwrap_err();
        assert_eq!(err.code(), tonic::Code::Unimplemented);
    }
}
