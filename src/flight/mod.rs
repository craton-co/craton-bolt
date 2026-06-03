// SPDX-License-Identifier: Apache-2.0

//! Arrow Flight SQL service skeleton (feature `flight`).
//!
//! This module exposes the [`crate::Engine`] over the Arrow Flight SQL
//! gRPC protocol. It is split across three sibling modules:
//!
//! - [`server`] (this agent): the tonic `FlightService` trait implementation —
//!   the gRPC entry points (handshake, get_flight_info, do_get, …) and the
//!   routing into the command + encode layers.
//! - `sql` (sibling agent): decode of Flight SQL command messages
//!   (`CommandStatementQuery`, `CommandGetTables`, …) and execution against the
//!   engine, returning a host-side Arrow result.
//! - `encode` (sibling agent): conversion of an Arrow [`RecordBatch`] /
//!   [`SchemaRef`] result into the `FlightData` stream the client consumes.
//!
//! # Internal ABI (server ↔ sql ↔ encode)
//!
//! The server layer is deliberately thin: it owns no Flight SQL command
//! semantics of its own. It calls into `sql` to turn the bytes on the wire
//! into a plan/result, and into `encode` to turn a result back into the wire
//! format. The exact function signatures the server expects are documented on
//! [`SqlCommandResult`] and in the integration notes; the sibling agents
//! implement those entry points.
//!
//! # Feature gating
//!
//! The whole subsystem is gated behind the `flight` cargo feature. The default
//! build does not pull in `tonic`, `arrow-flight`, `futures`, or `prost`, and
//! does not compile any of this code.
//!
//! # Status
//!
//! CORE skeleton: handshake (trivial accept), and the `get_flight_info` /
//! `get_schema` / `do_get` happy path for a statement query are routed to the
//! command layer. `do_put`, `do_action`, `do_exchange`, `list_flights`, and
//! `list_actions` are stubbed with `Status::unimplemented` and marked TODO.

#![cfg(feature = "flight")]

use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use arrow_flight::flight_service_server::FlightServiceServer;
use tonic::transport::Server;

use crate::exec::Engine;

pub mod server;

// Sibling-agent-owned modules. Declared here so the server can route into
// them; their public items follow the ABI in the module docs / integration
// notes. Until those modules land, the orchestrator wires these `mod`
// declarations (and this comment serves as the contract).
pub mod sql;
pub mod encode;

/// A Flight SQL service backed by a single shared [`Engine`].
///
/// The engine is wrapped in `Arc<Mutex<…>>` because [`Engine::sql`] takes
/// `&self` but the engine carries interior `RefCell`/CUDA-context state that is
/// not `Sync`; serialising query execution through a `Mutex` is the simplest
/// correct shape for the skeleton. A future revision can swap this for a pool
/// of engines or a `tokio::task::spawn_blocking` worker once the per-query
/// concurrency story is settled (TODO).
#[derive(Clone)]
pub struct FlightSqlServer {
    /// Shared, mutex-guarded query engine. `Arc` so the handle is cheap to
    /// clone into each tonic request task.
    pub(crate) engine: Arc<Mutex<Engine>>,
}

impl FlightSqlServer {
    /// Wrap an already-constructed engine in a Flight SQL service.
    pub fn new(engine: Engine) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
        }
    }

    /// Wrap a pre-shared engine handle (e.g. one shared with other subsystems).
    pub fn from_shared(engine: Arc<Mutex<Engine>>) -> Self {
        Self { engine }
    }

    /// Borrow the shared engine handle.
    pub(crate) fn engine(&self) -> &Arc<Mutex<Engine>> {
        &self.engine
    }
}

/// Result of decoding + executing a Flight SQL command (the shape `sql`
/// hands back to the server / encode layer).
///
/// This is the central ABI type between the server and the `sql` module. The
/// sibling `sql` agent is expected to expose roughly:
///
/// ```ignore
/// // sql.rs
/// pub fn execute_flight_command(
///     engine: &std::sync::Arc<std::sync::Mutex<crate::exec::Engine>>,
///     cmd: &[u8],            // the FlightDescriptor `cmd` bytes (prost-encoded Any)
/// ) -> Result<crate::flight::SqlCommandResult, tonic::Status>;
///
/// pub fn schema_for_command(
///     engine: &std::sync::Arc<std::sync::Mutex<crate::exec::Engine>>,
///     cmd: &[u8],
/// ) -> Result<arrow_schema::SchemaRef, tonic::Status>;
/// ```
///
/// and `encode` is expected to expose:
///
/// ```ignore
/// // encode.rs
/// pub fn batches_to_flight_stream(
///     result: crate::flight::SqlCommandResult,
/// ) -> futures::stream::BoxStream<'static, Result<arrow_flight::FlightData, tonic::Status>>;
///
/// pub fn schema_to_ipc_bytes(
///     schema: &arrow_schema::Schema,
/// ) -> Result<bytes::Bytes, tonic::Status>;
/// ```
///
/// The fields below are the minimum the encode layer needs.
pub struct SqlCommandResult {
    /// Output schema of the result set.
    pub schema: arrow_schema::SchemaRef,
    /// Materialised result batches (host-side Arrow).
    pub batches: Vec<arrow_array::RecordBatch>,
    /// The original command bytes, echoed so `get_flight_info` can stuff them
    /// back into the returned ticket for the matching `do_get` round-trip.
    pub ticket_cmd: bytes::Bytes,
}

/// Build a tonic [`Server`] hosting the Flight SQL service and serve it on
/// `addr` until the process is shut down.
///
/// This is the convenience entry point for embedding a Flight SQL endpoint:
///
/// ```ignore
/// use craton_bolt::flight::{serve, FlightSqlServer};
/// use craton_bolt::Engine;
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let engine = Engine::new()?;
/// let svc = FlightSqlServer::new(engine);
/// serve("127.0.0.1:50051".parse()?, svc).await?;
/// # Ok(())
/// # }
/// ```
pub async fn serve(
    addr: SocketAddr,
    server: FlightSqlServer,
) -> Result<(), tonic::transport::Error> {
    Server::builder()
        .add_service(FlightServiceServer::new(server))
        .serve(addr)
        .await
}

/// Build the configured tonic [`Server`] router without binding/serving it.
///
/// Useful when the caller wants to add additional services, install
/// interceptors, or drive a graceful-shutdown future of their own. The
/// returned [`tonic::transport::server::Router`] is ready for `.serve(addr)`
/// or `.serve_with_shutdown(addr, signal)`.
pub fn router(server: FlightSqlServer) -> tonic::transport::server::Router {
    Server::builder().add_service(FlightServiceServer::new(server))
}
