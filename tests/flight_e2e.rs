// SPDX-License-Identifier: Apache-2.0

//! End-to-end fixtures for the optional `flight` subsystem.
//!
//! The review that retired `deep-research-bolt.md` called out that the
//! `flight` lane only *compiled* the feature: nothing drove bytes through the
//! Flight SQL wire path, so encode/decode drift would land unnoticed. This
//! binary closes that gap with two tiers:
//!
//! 1. **A host-runnable wire round-trip** (not `#[ignore]`d) — a
//!    [`RecordBatch`] is encoded through the server's own
//!    [`craton_bolt::flight::encode::batches_to_flight_stream`] and decoded
//!    back with the stock arrow-flight client decoder, asserting the values
//!    survive the IPC round-trip. This needs no CUDA context and runs in the
//!    hosted `feature build (flight + substrait)` CI lane.
//! 2. **A real gRPC round-trip against a live server** (`#[ignore = "gpu:e2e"]`)
//!    — a tonic server is bound on loopback, a `FlightServiceClient` issues a
//!    Flight SQL `CommandStatementQuery` through `get_flight_info` + `do_get`,
//!    and the decoded result values are asserted. Building the service needs
//!    an [`craton_bolt::Engine`] (and therefore a CUDA context), so it runs on
//!    the blocking self-hosted GPU lane:
//!
//! ```text
//! cargo test --no-default-features --features cudarc,flight --test flight_e2e -- --ignored
//! ```

#![cfg(feature = "flight")]

use std::sync::Arc;

use arrow_array::{Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};

use arrow_flight::decode::FlightRecordBatchStream;
use arrow_flight::error::FlightError;
use arrow_flight::FlightData;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt};

use craton_bolt::flight::{encode, SqlCommandResult};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// `t(id Int64, name Utf8)` with one NULL, exercising both a fixed-width and a
/// variable-width column plus a validity bitmap across the IPC boundary.
fn sample() -> (Arc<Schema>, RecordBatch) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])),
            Arc::new(StringArray::from(vec![Some("alpha"), None, Some("gamma")])),
        ],
    )
    .expect("fixture batch");
    (schema, batch)
}

/// Drive a `FlightData` stream through the stock arrow-flight decoder — the
/// same code path a third-party Flight SQL client uses — and collect the
/// decoded batches.
async fn decode_flight_stream(
    stream: BoxStream<'static, Result<FlightData, tonic::Status>>,
) -> Vec<RecordBatch> {
    let mapped = stream.map(|item| item.map_err(FlightError::Tonic));
    FlightRecordBatchStream::new_from_flight_data(mapped)
        .try_collect::<Vec<_>>()
        .await
        .expect("client decodes the server's FlightData stream")
}

fn col_i64(batch: &RecordBatch, c: usize) -> &Int64Array {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("column is Int64")
}

fn col_str(batch: &RecordBatch, c: usize) -> &StringArray {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("column is Utf8")
}

// ---------------------------------------------------------------------------
// Tier 1 — host-runnable wire round-trip (hosted CI)
// ---------------------------------------------------------------------------

/// The server's encode path and a stock arrow-flight client decoder agree:
/// schema, row count, and every value (including the NULL) survive.
#[tokio::test]
async fn flight_wire_roundtrip_preserves_batch() {
    let (schema, batch) = sample();
    let result = SqlCommandResult {
        schema: schema.clone(),
        batches: vec![batch.clone()],
        ticket_cmd: bytes::Bytes::new(),
    };

    let decoded = decode_flight_stream(encode::batches_to_flight_stream(result)).await;
    assert_eq!(decoded.len(), 1, "one batch in, one batch out");
    let got = &decoded[0];

    assert_eq!(got.schema().fields(), schema.fields());
    assert_eq!(got.num_rows(), 3);
    let ids = col_i64(got, 0);
    assert_eq!(
        (0..3).map(|i| ids.value(i)).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    let names = col_str(got, 1);
    assert_eq!(names.value(0), "alpha");
    assert!(names.is_null(1), "NULL must survive the IPC round-trip");
    assert_eq!(names.value(2), "gamma");
}

/// Multiple batches encode into one stream that decodes back to the same
/// per-batch row counts, so a multi-morsel result is not silently coalesced or
/// truncated.
#[tokio::test]
async fn flight_wire_roundtrip_preserves_batch_boundaries() {
    let (schema, batch) = sample();
    let result = SqlCommandResult {
        schema: schema.clone(),
        batches: vec![batch.clone(), batch],
        ticket_cmd: bytes::Bytes::new(),
    };

    let decoded = decode_flight_stream(encode::batches_to_flight_stream(result)).await;
    assert_eq!(decoded.len(), 2);
    assert!(decoded.iter().all(|b| b.num_rows() == 3));
}

/// `get_schema`'s IPC payload decodes back to the schema the server holds, so
/// a client that reads the schema without fetching data sees the same fields.
#[test]
fn schema_ipc_payload_decodes_to_source_schema() {
    let (schema, _) = sample();
    let bytes = encode::schema_to_ipc_bytes(schema.as_ref()).expect("encode schema");
    let decoded: Schema = arrow_flight::SchemaResult { schema: bytes }
        .try_into()
        .expect("decode schema");
    assert_eq!(decoded.fields(), schema.fields());
}

// ---------------------------------------------------------------------------
// Tier 2 — live gRPC round-trip against a real engine (self-hosted GPU lane)
// ---------------------------------------------------------------------------

/// Reserve a free loopback port by binding and immediately releasing it.
///
/// tonic's `Router::serve` binds the address itself, so the test cannot hand
/// it an already-bound listener without pulling in `tokio-stream`. Probing for
/// a free ephemeral port keeps the fixture dependency-free; the window between
/// release and re-bind is a test-only concern.
fn free_loopback_addr() -> std::net::SocketAddr {
    let probe = std::net::TcpListener::bind("127.0.0.1:0").expect("bind probe socket");
    probe.local_addr().expect("probe addr")
}

/// Build an engine holding `t(id, name)` for the live-server fixtures.
fn engine_with_fixture() -> craton_bolt::Engine {
    let (_, batch) = sample();
    let mut engine = craton_bolt::Engine::new().expect("CUDA ctx");
    engine.register_table("t", batch).expect("register t");
    engine
}

/// Encode an ad-hoc SQL string as the Flight SQL `cmd` bytes a client puts in
/// a `Cmd` descriptor (a prost-encoded `google.protobuf.Any`).
fn statement_cmd(sql: &str) -> bytes::Bytes {
    use arrow_flight::sql::{CommandStatementQuery, ProstMessageExt};
    use prost::Message;
    CommandStatementQuery {
        query: sql.to_string(),
        transaction_id: None,
    }
    .as_any()
    .encode_to_vec()
    .into()
}

/// Serve `server` on `addr` until `shutdown` resolves, on a background task.
async fn spawn_server(
    server: craton_bolt::flight::FlightSqlServer,
    addr: std::net::SocketAddr,
    shutdown: tokio::sync::oneshot::Receiver<()>,
) -> tokio::task::JoinHandle<()> {
    let handle = tokio::spawn(async move {
        craton_bolt::flight::router(server)
            .serve_with_shutdown(addr, async {
                let _ = shutdown.await;
            })
            .await
            .expect("flight server serves");
    });
    // Give the listener a moment to bind before the client dials.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    handle
}

/// Full Flight SQL round-trip over real gRPC: `get_flight_info` resolves the
/// schema and a ticket, `do_get` streams the result, and the decoded values
/// match what `Engine::sql` would have produced directly.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "gpu:e2e"]
async fn flight_sql_query_round_trips_over_grpc() {
    use arrow_flight::flight_descriptor::DescriptorType;
    use arrow_flight::flight_service_client::FlightServiceClient;
    use arrow_flight::FlightDescriptor;

    let addr = free_loopback_addr();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let svc = craton_bolt::flight::FlightSqlServer::new(engine_with_fixture());
    let served = spawn_server(svc, addr, rx).await;

    let mut client = FlightServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("client connects");

    let descriptor = FlightDescriptor {
        r#type: DescriptorType::Cmd as i32,
        cmd: statement_cmd("SELECT id FROM t WHERE id > 1"),
        path: Vec::new(),
    };

    let info = client
        .get_flight_info(descriptor)
        .await
        .expect("get_flight_info succeeds")
        .into_inner();
    let ticket = info
        .endpoint
        .first()
        .and_then(|e| e.ticket.clone())
        .expect("FlightInfo carries a ticket");

    let stream = client
        .do_get(ticket)
        .await
        .expect("do_get succeeds")
        .into_inner()
        .map_err(FlightError::Tonic);
    let batches = FlightRecordBatchStream::new_from_flight_data(stream)
        .try_collect::<Vec<_>>()
        .await
        .expect("decode do_get stream");

    let rows: Vec<i64> = batches
        .iter()
        .flat_map(|b| {
            let ids = col_i64(b, 0);
            (0..b.num_rows()).map(|i| ids.value(i)).collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(rows, vec![2, 3]);

    let _ = tx.send(());
    let _ = served.await;
}

/// A server built with `with_bearer_token` rejects an unauthenticated RPC with
/// `UNAUTHENTICATED` and accepts the same call once the token is presented.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "gpu:e2e"]
async fn flight_bearer_token_gates_rpcs() {
    use arrow_flight::flight_descriptor::DescriptorType;
    use arrow_flight::flight_service_client::FlightServiceClient;
    use arrow_flight::FlightDescriptor;

    let addr = free_loopback_addr();
    let (tx, rx) = tokio::sync::oneshot::channel();
    let svc = craton_bolt::flight::FlightSqlServer::new(engine_with_fixture())
        .with_bearer_token("s3cret");
    let served = spawn_server(svc, addr, rx).await;

    let mut client = FlightServiceClient::connect(format!("http://{addr}"))
        .await
        .expect("client connects");

    let descriptor = || FlightDescriptor {
        r#type: DescriptorType::Cmd as i32,
        cmd: statement_cmd("SELECT id FROM t"),
        path: Vec::new(),
    };

    let err = client
        .get_flight_info(descriptor())
        .await
        .expect_err("unauthenticated RPC must be rejected");
    assert_eq!(err.code(), tonic::Code::Unauthenticated);

    let mut authed = tonic::Request::new(descriptor());
    authed.metadata_mut().insert(
        "authorization",
        "Bearer s3cret".parse().expect("valid metadata value"),
    );
    client
        .get_flight_info(authed)
        .await
        .expect("authenticated RPC succeeds");

    let _ = tx.send(());
    let _ = served.await;
}
