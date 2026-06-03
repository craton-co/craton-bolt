// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "flight")]

//! Arrow ↔ Arrow Flight result-encoding helpers (feature `flight`).
//!
//! These are the shared encode primitives the Flight SQL server's
//! `do_get` / `get_schema` handlers call. Keeping them in one module means the
//! server (`src/flight/server.rs`) shares exactly one encoding path — there is
//! no second, drifting copy of "how we turn batches into `FlightData`".
//!
//! ## Unified server ↔ encode ABI
//!
//! * [`batches_to_flight_stream`]: `SqlCommandResult -> BoxStream<Result<
//!   FlightData, Status>>` — encodes the result's schema + batches into the
//!   Flight wire messages (a leading schema message followed by one or more
//!   data messages per batch) and hands back the boxed stream `do_get`
//!   returns.
//! * [`schema_to_ipc_bytes`]: `&Schema -> Bytes` — the IPC-serialised schema
//!   payload for a `SchemaResult` (`get_schema`).
//!
//! ## Encoder choice
//!
//! We use [`arrow_flight::utils::batches_to_flight_data`] (arrow-flight 53)
//! for the eager `Vec<FlightData>` path: it is the canonical helper and it
//! prepends the schema message for us. Encoding eagerly and then streaming
//! keeps the hot path simple (Bolt collects the full result into a single
//! `RecordBatch` today); a future truly-streaming encoder
//! (`FlightDataEncoderBuilder`) can replace this without changing the server
//! signature.

use arrow_flight::FlightData;
use arrow_schema::Schema;
use bytes::Bytes;
use futures::stream::{self, BoxStream};
// `.boxed()` on stream adapters comes from the StreamExt extension trait.
use futures::stream::StreamExt as _;
use tonic::Status;

use crate::flight::SqlCommandResult;

/// Encode a [`SqlCommandResult`]'s schema + batches into the boxed
/// `FlightData` stream `do_get` returns.
///
/// The stream starts with a single schema `FlightData` message and is
/// followed by one data message per batch (arrow-flight may emit an extra
/// dictionary message ahead of a batch that introduces new dictionary values;
/// the sequence is opaque and must be streamed verbatim).
///
/// Encoding is eager: any IPC error is surfaced as a single
/// `Err(Status::internal(..))` item at the head of the stream rather than a
/// panic, because this drives a live gRPC response.
pub fn batches_to_flight_stream(
    result: SqlCommandResult,
) -> BoxStream<'static, Result<FlightData, Status>> {
    let SqlCommandResult {
        schema, batches, ..
    } = result;

    match arrow_flight::utils::batches_to_flight_data(schema.as_ref(), batches) {
        Ok(msgs) => stream::iter(msgs.into_iter().map(Ok)).boxed(),
        Err(e) => {
            let err = Status::internal(format!("flight IPC encode failed: {e}"));
            stream::once(async move { Err(err) }).boxed()
        }
    }
}

/// IPC-serialise a [`Schema`] into the bytes a Flight `SchemaResult` carries.
///
/// Used by `get_schema`. Mirrors the schema message arrow-flight emits at the
/// head of a `do_get` stream, so a client that decodes either path sees the
/// same schema.
pub fn schema_to_ipc_bytes(schema: &Schema) -> Result<Bytes, Status> {
    // `SchemaAsIpc` is arrow-flight's canonical schema-IPC encoder; converting
    // it into an `arrow_flight::SchemaResult` produces exactly the bytes a
    // Flight `SchemaResult.schema` field expects.
    let options = arrow::ipc::writer::IpcWriteOptions::default();
    let schema_as_ipc = arrow_flight::SchemaAsIpc::new(schema, &options);
    let schema_result: arrow_flight::SchemaResult = schema_as_ipc
        .try_into()
        .map_err(|e| Status::internal(format!("flight schema IPC encode failed: {e}")))?;
    Ok(schema_result.schema)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field};
    use std::sync::Arc;

    fn sample() -> (Arc<Schema>, RecordBatch) {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec![Some("a"), None, Some("c")])),
            ],
        )
        .unwrap();
        (schema, batch)
    }

    #[tokio::test]
    async fn stream_yields_schema_plus_data() {
        use futures::StreamExt;
        let (schema, batch) = sample();
        let result = SqlCommandResult {
            schema: schema.clone(),
            batches: vec![batch],
            ticket_cmd: Bytes::new(),
        };

        let mut s = batches_to_flight_stream(result);
        let mut n = 0;
        while let Some(item) = s.next().await {
            item.expect("stream item ok");
            n += 1;
        }
        // schema message + at least one data message.
        assert!(n >= 2, "expected schema + data, got {n}");
    }

    #[test]
    fn schema_ipc_bytes_are_non_empty() {
        let (schema, _) = sample();
        let bytes = schema_to_ipc_bytes(schema.as_ref()).expect("encode");
        assert!(!bytes.is_empty());
    }
}
