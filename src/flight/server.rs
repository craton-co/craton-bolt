// SPDX-License-Identifier: Apache-2.0

//! `FlightService` trait implementation for [`FlightSqlServer`].
//!
//! This is the gRPC surface of the Flight SQL endpoint. The implementation is
//! intentionally thin:
//!
//! - `handshake` trivially accepts (no auth in the skeleton).
//! - `get_flight_info` / `get_schema` / `do_get` decode the command via the
//!   sibling [`crate::flight::sql`] module and stream the result via the
//!   sibling [`crate::flight::encode`] module.
//! - every other method returns [`tonic::Status::unimplemented`] for now
//!   (TODO: prepared statements via `do_put` / `do_action`, bidirectional
//!   `do_exchange`, catalog enumeration via `list_flights` / `list_actions`).

#![cfg(feature = "flight")]

use std::pin::Pin;

use arrow_flight::flight_service_server::FlightService;
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PutResult, SchemaResult, Ticket,
};
use futures::stream::{self, BoxStream};
use futures::StreamExt;
use tonic::{Request, Response, Status, Streaming};

use crate::flight::{encode, sql, FlightSqlServer};

/// Boxed, `Send`-able server stream alias — every Flight RPC returns one of
/// these (over a different item type).
type RpcStream<T> = Pin<Box<dyn futures::Stream<Item = Result<T, Status>> + Send + 'static>>;

#[tonic::async_trait]
impl FlightService for FlightSqlServer {
    type HandshakeStream = RpcStream<HandshakeResponse>;
    type ListFlightsStream = RpcStream<FlightInfo>;
    type DoGetStream = RpcStream<FlightData>;
    type DoPutStream = RpcStream<PutResult>;
    type DoActionStream = RpcStream<arrow_flight::Result>;
    type ListActionsStream = RpcStream<ActionType>;
    type DoExchangeStream = RpcStream<FlightData>;

    /// Trivial handshake: echo back an empty response and accept the
    /// connection. No authentication is performed in the skeleton.
    ///
    /// TODO: support bearer-token / basic auth handshake payloads and emit a
    /// session token in `HandshakeResponse::payload`.
    async fn handshake(
        &self,
        _request: Request<Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        let resp = HandshakeResponse {
            protocol_version: 0,
            payload: Default::default(),
        };
        let output = stream::once(async move { Ok(resp) });
        Ok(Response::new(Box::pin(output) as Self::HandshakeStream))
    }

    /// Enumerate available flights. Not implemented in the skeleton — a Flight
    /// SQL client discovers result sets via `get_flight_info` on a command
    /// descriptor, not via `list_flights`.
    ///
    /// TODO: optionally surface registered tables here.
    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Err(Status::unimplemented("list_flights is not yet supported"))
    }

    /// Resolve a [`FlightDescriptor`] into a [`FlightInfo`] — the schema plus
    /// the endpoint/ticket the client uses to fetch the data via `do_get`.
    ///
    /// Only `DescriptorType::Cmd` descriptors are supported: the `cmd` bytes
    /// carry a prost-encoded Flight SQL command which [`sql`] decodes and
    /// executes. The resulting schema is IPC-encoded into the `FlightInfo`, and
    /// the original command bytes are echoed back into the [`Ticket`] so the
    /// follow-up `do_get` can re-run / fetch the same result.
    async fn get_flight_info(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let descriptor = request.into_inner();
        let cmd = command_bytes(&descriptor)?;

        // Decode + execute via the SQL command layer to learn the schema.
        let result = sql::execute_flight_command(self.engine(), &cmd)?;
        let schema = result.schema.clone();

        // The ticket carries the command bytes verbatim so `do_get` can map it
        // back to the same result set.
        let ticket = Ticket {
            ticket: cmd.clone().into(),
        };
        let endpoint = arrow_flight::FlightEndpoint {
            ticket: Some(ticket),
            location: Vec::new(),
            ..Default::default()
        };

        // IPC-encode the schema into the FlightInfo. `try_with_schema` does the
        // IPC serialisation for us (arrow-flight helper).
        let info = FlightInfo::new()
            .try_with_schema(schema.as_ref())
            .map_err(|e| Status::internal(format!("failed to encode schema: {e}")))?
            .with_descriptor(descriptor)
            .with_endpoint(endpoint)
            .with_total_records(
                result.batches.iter().map(|b| b.num_rows() as i64).sum(),
            )
            .with_total_bytes(-1);

        Ok(Response::new(info))
    }

    /// Return the IPC-encoded schema for a command descriptor without
    /// materialising the data.
    async fn get_schema(
        &self,
        request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        let descriptor = request.into_inner();
        let cmd = command_bytes(&descriptor)?;

        let schema = sql::schema_for_command(self.engine(), &cmd)?;
        let ipc = encode::schema_to_ipc_bytes(schema.as_ref())?;
        Ok(Response::new(SchemaResult { schema: ipc }))
    }

    /// Fetch a result set identified by the ticket produced in
    /// `get_flight_info`. The ticket bytes are the original Flight SQL command;
    /// we decode + execute it again and stream the resulting batches as
    /// `FlightData` via the encode layer.
    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let ticket = request.into_inner();
        let cmd = ticket.ticket;

        let result = sql::execute_flight_command(self.engine(), &cmd)?;
        let flight_stream: BoxStream<'static, Result<FlightData, Status>> =
            encode::batches_to_flight_stream(result);

        Ok(Response::new(flight_stream.boxed() as Self::DoGetStream))
    }

    /// Upload / prepared-statement bind path. Not implemented in the skeleton.
    ///
    /// TODO: handle `ActionCreatePreparedStatementRequest` bind parameters and
    /// `CommandStatementUpdate` (INSERT/UPDATE/DELETE) DML.
    async fn do_put(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented("do_put is not yet supported"))
    }

    /// Flight SQL action dispatch (prepared statements, transactions, …). Not
    /// implemented in the skeleton.
    ///
    /// TODO: route `CreatePreparedStatement` / `ClosePreparedStatement` and the
    /// transaction actions here.
    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("do_action is not yet supported"))
    }

    /// Advertise the supported actions. Empty in the skeleton (no actions yet).
    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        let output = stream::empty::<Result<ActionType, Status>>();
        Ok(Response::new(Box::pin(output) as Self::ListActionsStream))
    }

    /// Bidirectional data exchange. Not implemented in the skeleton.
    async fn do_exchange(
        &self,
        _request: Request<Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("do_exchange is not yet supported"))
    }
}

/// Extract the command bytes from a `Cmd`-type [`FlightDescriptor`], rejecting
/// `Path`-type descriptors (which the Flight SQL command protocol does not
/// use).
fn command_bytes(descriptor: &FlightDescriptor) -> Result<bytes::Bytes, Status> {
    use arrow_flight::flight_descriptor::DescriptorType;
    match DescriptorType::try_from(descriptor.r#type) {
        Ok(DescriptorType::Cmd) => Ok(descriptor.cmd.clone()),
        Ok(DescriptorType::Path) => Err(Status::unimplemented(
            "PATH descriptors are not supported; use a Flight SQL command descriptor",
        )),
        _ => Err(Status::invalid_argument(
            "unknown FlightDescriptor type",
        )),
    }
}

#[cfg(test)]
mod tests {
    //! Host-testable bits of the server layer. The full RPC paths require a
    //! running tonic server + a CUDA context (via the sibling `sql` module),
    //! so these tests cover only the pure descriptor-decode helper.
    use super::*;
    use arrow_flight::flight_descriptor::DescriptorType;

    #[test]
    fn command_bytes_extracts_cmd_payload() {
        let desc = FlightDescriptor {
            r#type: DescriptorType::Cmd as i32,
            cmd: bytes::Bytes::from_static(b"SELECT 1"),
            path: Vec::new(),
        };
        let got = command_bytes(&desc).expect("cmd descriptor should decode");
        assert_eq!(&got[..], b"SELECT 1");
    }

    #[test]
    fn command_bytes_rejects_path_descriptor() {
        let desc = FlightDescriptor {
            r#type: DescriptorType::Path as i32,
            cmd: bytes::Bytes::new(),
            path: vec!["t".to_string()],
        };
        let err = command_bytes(&desc).expect_err("path descriptor must be rejected");
        assert_eq!(err.code(), tonic::Code::Unimplemented);
    }

    #[test]
    fn command_bytes_rejects_unknown_type() {
        let desc = FlightDescriptor {
            r#type: 99,
            cmd: bytes::Bytes::new(),
            path: Vec::new(),
        };
        let err = command_bytes(&desc).expect_err("unknown type must be rejected");
        assert_eq!(err.code(), tonic::Code::InvalidArgument);
    }
}
