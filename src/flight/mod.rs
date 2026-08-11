// SPDX-License-Identifier: Apache-2.0

//! Arrow Flight SQL service skeleton (feature `flight`).
//!
//! # ⚠️ EXPERIMENTAL — no authentication or TLS by default
//!
//! This subsystem is **experimental** and ships *without transport security
//! and without authentication by default*. Treat it as a trusted-network /
//! localhost-only convenience, not an internet-facing endpoint:
//!
//! - **No TLS.** [`serve`] / [`router`] build a *plaintext* tonic server. Any
//!   query text and result data travels in the clear. Put it behind a TLS
//!   terminator (reverse proxy / service mesh) before exposing it off-box.
//! - **Authentication is opt-in and minimal.** By default *any* client may
//!   connect and run *arbitrary SQL*. A single shared bearer token can be
//!   required via [`FlightSqlServer::with_bearer_token`]; this is a coarse
//!   gate, not a substitute for real authn/z. There is no per-user identity,
//!   no authorization, and no auditing.
//! - **Bind to loopback.** The convenience [`serve`] entry point and the
//!   [`FlightSqlConfig`] default to a loopback bind address. Do **not** bind
//!   `0.0.0.0` unless the port is firewalled to a trusted network.
//! - **Remote code-shaped risk.** A connected client can run any SQL the
//!   engine accepts. Result size is capped (see [`FlightSqlConfig`]) but query
//!   *execution* is not time-bounded here — a hostile or pathological query
//!   can still consume GPU/host resources for its duration.
//!
//! See the individual `# Security` notes on [`serve`], [`FlightSqlServer`], and
//! [`FlightSqlConfig`].
//!
//! ---
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
//! CORE skeleton: handshake (accept, with optional shared-bearer-token check),
//! and the `get_flight_info` / `get_schema` / `do_get` happy path for a
//! statement query are routed to the command layer. `get_flight_info` resolves
//! only the schema (it no longer fully executes the query — `do_get`
//! materialises once). `do_put`, `do_action`, `do_exchange`, `list_flights`,
//! and `list_actions` are stubbed with `Status::unimplemented` and marked TODO.

#![cfg(feature = "flight")]

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use arrow_flight::flight_service_server::FlightServiceServer;
use parking_lot::Mutex;
use tonic::transport::Server;

use crate::exec::Engine;

/// Shared engine handle used throughout the Flight subsystem.
///
/// A [`parking_lot::Mutex`] (not [`std::sync::Mutex`]) is used deliberately:
/// it does **not** poison on panic. A query that panics while holding the lock
/// therefore does not permanently brick every subsequent request — the lock is
/// simply released and the next client can proceed. (`std::sync::Mutex` would
/// stay poisoned for the life of the process.)
pub(crate) type SharedEngine = Arc<Mutex<Engine>>;

pub mod server;

// Sibling-agent-owned modules. Declared here so the server can route into
// them; their public items follow the ABI in the module docs / integration
// notes. Until those modules land, the orchestrator wires these `mod`
// declarations (and this comment serves as the contract).
pub mod encode;
pub mod sql;

/// Resource / safety limits for a [`FlightSqlServer`].
///
/// # Security
///
/// These caps bound how much work / data a single (already-connected,
/// possibly hostile) client can extract per query. They are the *only*
/// availability guard rails the skeleton offers — there is no per-query
/// execution timeout here (query duration is bounded only by the engine), so
/// keep the endpoint on a trusted network regardless.
#[derive(Debug, Clone)]
pub struct FlightSqlConfig {
    /// Maximum number of result rows a single query may return before it is
    /// rejected with `RESOURCE_EXHAUSTED`. `None` disables the row cap.
    pub max_result_rows: Option<usize>,
    /// Maximum total result size, in bytes (Arrow in-memory array size), a
    /// single query may return before it is rejected with
    /// `RESOURCE_EXHAUSTED`. `None` disables the byte cap.
    pub max_result_bytes: Option<usize>,
}

impl Default for FlightSqlConfig {
    /// Conservative defaults: cap a single result at ~16M rows / ~1 GiB so a
    /// runaway or hostile `SELECT` cannot trivially exhaust host memory while
    /// being materialised + IPC-encoded. Tune via the public fields.
    fn default() -> Self {
        Self {
            max_result_rows: Some(16 * 1024 * 1024),
            max_result_bytes: Some(1024 * 1024 * 1024),
        }
    }
}

/// A Flight SQL service backed by a single shared [`Engine`].
///
/// The engine is wrapped in [`SharedEngine`] (`Arc<parking_lot::Mutex<…>>`)
/// because [`Engine::sql`] takes `&self` but the engine carries interior
/// `RefCell`/CUDA-context state that is not `Sync`; serialising query execution
/// through a `Mutex` is the simplest correct shape for the skeleton. A future
/// revision can swap this for a pool of engines or a
/// `tokio::task::spawn_blocking` worker once the per-query concurrency story is
/// settled (TODO). `parking_lot` is used so a panicking query does not poison
/// the lock and brick the server.
///
/// # Security
///
/// Construct with [`FlightSqlServer::new`] for the no-auth default, or
/// [`FlightSqlServer::with_bearer_token`] to require a shared bearer token on
/// every RPC. See the [module docs](crate::flight) for the full caveats (no
/// TLS, no per-user identity, no execution timeout).
#[derive(Clone)]
pub struct FlightSqlServer {
    /// Shared, mutex-guarded query engine. `Arc` so the handle is cheap to
    /// clone into each tonic request task.
    pub(crate) engine: SharedEngine,
    /// Resource limits applied to every query result.
    pub(crate) config: FlightSqlConfig,
    /// Optional shared bearer token. When `Some`, every RPC must present a
    /// matching `authorization: Bearer <token>` metadata header or it is
    /// rejected with `UNAUTHENTICATED`. When `None` (the default), the server
    /// accepts all clients (see module-level security warning).
    pub(crate) bearer_token: Option<Arc<str>>,
}

impl FlightSqlServer {
    /// Wrap an already-constructed engine in a Flight SQL service with default
    /// resource limits and **no authentication** (see module security note).
    pub fn new(engine: Engine) -> Self {
        Self::from_shared(Arc::new(Mutex::new(engine)))
    }

    /// Wrap a pre-shared engine handle (e.g. one shared with other subsystems).
    pub fn from_shared(engine: SharedEngine) -> Self {
        Self {
            engine,
            config: FlightSqlConfig::default(),
            bearer_token: None,
        }
    }

    /// Require a shared bearer token on every RPC.
    ///
    /// Clients must send `authorization: Bearer <token>` gRPC metadata; any
    /// request without a matching token is rejected with `UNAUTHENTICATED`.
    ///
    /// # Security
    ///
    /// This is a *single shared secret*, not per-user identity, and it is only
    /// as confidential as the transport (there is no TLS by default — see the
    /// module docs). The comparison is a simple equality check (not
    /// constant-time); pair with TLS and treat the token as a coarse gate.
    pub fn with_bearer_token(mut self, token: impl Into<String>) -> Self {
        self.bearer_token = Some(Arc::from(token.into().into_boxed_str()));
        self
    }

    /// Override the per-query resource limits.
    pub fn with_config(mut self, config: FlightSqlConfig) -> Self {
        self.config = config;
        self
    }

    /// Borrow the shared engine handle.
    pub(crate) fn engine(&self) -> &SharedEngine {
        &self.engine
    }

    /// Borrow the resource limits.
    pub(crate) fn config(&self) -> &FlightSqlConfig {
        &self.config
    }

    /// Validate the `authorization` metadata of an incoming request against the
    /// configured bearer token. A no-op (always `Ok`) when no token is
    /// configured.
    ///
    /// Returns `UNAUTHENTICATED` when a token is required but missing or wrong.
    pub(crate) fn check_auth<T>(&self, request: &tonic::Request<T>) -> Result<(), tonic::Status> {
        let Some(expected) = self.bearer_token.as_deref() else {
            return Ok(());
        };
        let presented = request
            .metadata()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .and_then(|s| {
                s.strip_prefix("Bearer ")
                    .or_else(|| s.strip_prefix("bearer "))
            });
        match presented {
            Some(tok) if tok == expected => Ok(()),
            _ => Err(tonic::Status::unauthenticated(
                "missing or invalid bearer token",
            )),
        }
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
///     engine: &crate::flight::SharedEngine,   // Arc<parking_lot::Mutex<Engine>>
///     cmd: &[u8],            // the FlightDescriptor `cmd` bytes (prost-encoded Any)
///     config: &crate::flight::FlightSqlConfig, // result-size caps
/// ) -> Result<crate::flight::SqlCommandResult, tonic::Status>;
///
/// pub fn schema_for_command(
///     engine: &crate::flight::SharedEngine,
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

/// The default loopback bind address (`127.0.0.1:50051`).
///
/// # Security
///
/// Loopback is the intended default exposure for this experimental, unauth'd,
/// plaintext endpoint. Prefer [`serve_local`] over hand-rolling a bind address
/// so an off-box bind (`0.0.0.0`) is always a deliberate, explicit choice.
pub const DEFAULT_LOOPBACK_ADDR: SocketAddr =
    SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 50051);

/// Build a tonic [`Server`] hosting the Flight SQL service and serve it on
/// `addr` until the process is shut down.
///
/// This is the convenience entry point for embedding a Flight SQL endpoint:
///
/// ```ignore
/// use craton_bolt::flight::{serve, FlightSqlServer, DEFAULT_LOOPBACK_ADDR};
/// use craton_bolt::Engine;
///
/// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
/// let engine = Engine::new()?;
/// let svc = FlightSqlServer::new(engine);
/// serve(DEFAULT_LOOPBACK_ADDR, svc).await?;
/// # Ok(())
/// # }
/// ```
///
/// # Security
///
/// The server is **plaintext** (no TLS) and, unless the service was built with
/// [`FlightSqlServer::with_bearer_token`], performs **no authentication**.
/// `serve` logs a warning when asked to bind a non-loopback address as a
/// guard against accidentally exposing it. Prefer [`serve_local`] for the
/// common case. For an off-box deployment, front it with a TLS terminator and
/// require a bearer token (and ideally additional network ACLs).
pub async fn serve(
    addr: SocketAddr,
    server: FlightSqlServer,
) -> Result<(), tonic::transport::Error> {
    if !addr.ip().is_loopback() {
        log::warn!(
            "craton-bolt flight: binding Flight SQL server to NON-loopback address {addr}; \
             this endpoint has no TLS and {auth} — expose only on a trusted network",
            auth = if server.bearer_token.is_some() {
                "uses a single shared bearer token"
            } else {
                "NO authentication"
            }
        );
    }
    Server::builder()
        .add_service(FlightServiceServer::new(server))
        .serve(addr)
        .await
}

/// Serve the Flight SQL service on the default loopback address
/// ([`DEFAULT_LOOPBACK_ADDR`]).
///
/// This is the recommended entry point: it cannot accidentally expose the
/// endpoint off-box. Use [`serve`] with an explicit address only when a
/// non-loopback bind is genuinely intended (and read its `# Security` note).
pub async fn serve_local(server: FlightSqlServer) -> Result<(), tonic::transport::Error> {
    serve(DEFAULT_LOOPBACK_ADDR, server).await
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
