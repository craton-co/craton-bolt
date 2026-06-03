// SPDX-License-Identifier: Apache-2.0
#![cfg(feature = "flight")]

//! Arrow ↔ Arrow Flight result-encoding helpers (feature `flight`).
//!
//! These are the shared encode primitives the Flight SQL server's
//! `do_get` / `get_flight_info` handlers call. Keeping them in one module
//! means the server module (`src/flight/server.rs`, owned by a sibling) and
//! any tests share exactly one encoding path — there is no second, drifting
//! copy of "how we turn batches into `FlightData`".
//!
//! ## API surface (siblings call these)
//!
//! - [`batches_to_flight_data`]: `(&Schema, &[RecordBatch]) -> Vec<FlightData>`
//!   — a schema message followed by one (or more, on dictionary replacement)
//!   `FlightData` message per record batch. Suitable to feed straight into a
//!   `futures::stream::iter` for `do_get`.
//! - [`flight_data_stream`]: same inputs, but already wrapped as the boxed
//!   `Stream<Item = Result<FlightData, Status>>` the `do_get` response type
//!   wants (`Self::DoGetStream`).
//! - [`schema_to_flight_info`]: `(&Schema, FlightDescriptor, Ticket) ->
//!   FlightInfo` — one endpoint carrying the ticket, schema attached via
//!   IPC.
//!
//! ## Encoder choice
//!
//! We use [`arrow_flight::utils::batches_to_flight_data`] (arrow-flight 53)
//! for the eager `Vec<FlightData>` path: it is the canonical helper, it
//! prepends the schema message for us, and it round-trips cleanly through
//! `arrow_flight::utils::flight_data_to_arrow_batch` (exercised in the unit
//! tests below). The streaming `do_get` path reuses the same `Vec` and turns
//! it into a stream, so there is a single encode code path.

use std::pin::Pin;

use arrow_array::RecordBatch;
use arrow_flight::{
    FlightData, FlightDescriptor, FlightEndpoint, FlightInfo, Ticket,
};
use arrow_schema::Schema;
use futures::stream::{self, Stream};
use tonic::Status;

/// Boxed stream type the Flight server's `do_get` returns.
///
/// Matches the `tonic` `Streaming`-compatible shape: a pinned, boxed,
/// `Send` stream of `Result<FlightData, Status>`. The server's
/// `type DoGetStream = FlightDataStream;` should alias this so the encode
/// helper and the trait impl agree.
pub type FlightDataStream =
    Pin<Box<dyn Stream<Item = Result<FlightData, Status>> + Send + 'static>>;

/// Encode a schema + its record batches into the Flight wire messages.
///
/// The returned vector starts with a single schema `FlightData` message and
/// is followed by one data message per batch (arrow-flight may emit an extra
/// dictionary message ahead of a batch that introduces new dictionary
/// values; callers must treat the result as an opaque ordered sequence and
/// stream it verbatim).
///
/// All batches must share `schema`; a mismatch surfaces as
/// `Status::internal` rather than a panic, because this runs inside a gRPC
/// handler where a panic would tear down the connection.
pub fn batches_to_flight_data(
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<Vec<FlightData>, Status> {
    // Defensive: arrow-flight's helper assumes every batch matches `schema`.
    // Validate up front so the error is attributable instead of a deep IPC
    // panic / corrupt stream on the wire.
    for (i, b) in batches.iter().enumerate() {
        if b.schema().as_ref() != schema {
            return Err(Status::internal(format!(
                "batch {i} schema does not match the provided result schema"
            )));
        }
    }

    arrow_flight::utils::batches_to_flight_data(schema, batches.to_vec())
        .map_err(|e| Status::internal(format!("flight IPC encode failed: {e}")))
}

/// Encode schema + batches and wrap them as the boxed `do_get` response
/// stream.
///
/// Convenience over [`batches_to_flight_data`] for the common case where the
/// server already has all batches in memory (Bolt collects the full result
/// into a single `RecordBatch` / small `Vec` today). Encoding eagerly and
/// then streaming keeps the hot path simple; a future truly-streaming
/// encoder (`FlightDataEncoderBuilder` over a `BatchStream`) can replace this
/// without changing the server signature.
pub fn flight_data_stream(
    schema: &Schema,
    batches: &[RecordBatch],
) -> Result<FlightDataStream, Status> {
    let msgs = batches_to_flight_data(schema, batches)?;
    Ok(into_stream(msgs))
}

/// Wrap an already-encoded `Vec<FlightData>` as the boxed `do_get` stream.
///
/// Split out so the server can encode once (e.g. to also inspect / log the
/// message count) and still hand back the right stream type.
pub fn into_stream(msgs: Vec<FlightData>) -> FlightDataStream {
    Box::pin(stream::iter(msgs.into_iter().map(Ok)))
}

/// Build a [`FlightInfo`] describing a result set: its schema, the
/// originating descriptor, and a single endpoint carrying `ticket`.
///
/// `do_get` later receives that same ticket to stream the data. We attach the
/// schema via `try_with_schema` (IPC-encodes the schema into the message) and
/// leave `total_records` / `total_bytes` unset (-1) since Bolt does not know
/// the cardinality until execution.
///
/// The endpoint has an empty `location` list, which per the Flight spec means
/// "fetch from the same server that served `get_flight_info`" — correct for
/// our single-node embedded server.
pub fn schema_to_flight_info(
    schema: &Schema,
    descriptor: FlightDescriptor,
    ticket: Ticket,
) -> Result<FlightInfo, Status> {
    let endpoint = FlightEndpoint::new().with_ticket(ticket);

    // `try_with_schema` is the only fallible step (it IPC-encodes the schema).
    // `with_descriptor` / `with_endpoint` are infallible builder methods that
    // consume and return `FlightInfo` by value.
    let info = FlightInfo::new()
        .try_with_schema(schema)
        .map_err(|e| Status::internal(format!("flight schema encode failed: {e}")))?
        .with_descriptor(descriptor)
        .with_endpoint(endpoint);

    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{Int32Array, StringArray};
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

    #[test]
    fn encode_prepends_schema_message() {
        let (schema, batch) = sample();
        let msgs = batches_to_flight_data(&schema, &[batch]).unwrap();
        // schema message + one data message.
        assert!(msgs.len() >= 2, "expected schema + data, got {}", msgs.len());
    }

    #[test]
    fn encode_then_decode_round_trips() {
        let (schema, batch) = sample();
        let msgs = batches_to_flight_data(&schema, std::slice::from_ref(&batch)).unwrap();

        // First message is the schema; the rest decode back into batches.
        let decoded_schema: Schema = (&msgs[0]).try_into().expect("schema decode");
        assert_eq!(&decoded_schema, schema.as_ref());

        let dictionaries_by_id = std::collections::HashMap::new();
        // Find the data message (skip the leading schema message; there may be
        // dictionary messages in between for dict-encoded columns, but this
        // sample has none).
        let decoded = arrow_flight::utils::flight_data_to_arrow_batch(
            &msgs[1],
            schema.clone(),
            &dictionaries_by_id,
        )
        .expect("data decode");

        assert_eq!(decoded.num_rows(), batch.num_rows());
        assert_eq!(decoded.num_columns(), batch.num_columns());
        assert_eq!(decoded.schema().as_ref(), schema.as_ref());

        let ids = decoded
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(ids.values(), &[1, 2, 3]);

        let names = decoded
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(names.value(0), "a");
        assert!(names.is_null(1));
        assert_eq!(names.value(2), "c");
    }

    #[test]
    fn schema_mismatch_is_an_error_not_a_panic() {
        let (schema, _batch) = sample();
        let other_schema = Arc::new(Schema::new(vec![Field::new(
            "x",
            DataType::Int64,
            false,
        )]));
        let other = RecordBatch::try_new(
            other_schema,
            vec![Arc::new(arrow_array::Int64Array::from(vec![9_i64]))],
        )
        .unwrap();

        let err = batches_to_flight_data(&schema, &[other]).unwrap_err();
        assert_eq!(err.code(), tonic::Code::Internal);
    }

    #[test]
    fn flight_info_carries_schema_and_ticket() {
        let (schema, _) = sample();
        // `new_cmd` / `Ticket::new` take `impl Into<Bytes>`; a `Vec<u8>`
        // converts without naming the `bytes` crate directly.
        let descriptor = FlightDescriptor::new_cmd(b"SELECT 1".to_vec());
        let ticket = Ticket::new(b"tkt-123".to_vec());

        let info = schema_to_flight_info(&schema, descriptor, ticket.clone()).unwrap();

        assert_eq!(info.endpoint.len(), 1);
        let ep_ticket = info.endpoint[0].ticket.as_ref().expect("endpoint ticket");
        assert_eq!(ep_ticket.ticket.as_ref(), b"tkt-123");

        // Schema round-trips out of the FlightInfo IPC payload.
        let decoded: Schema = info
            .clone()
            .try_decode_schema()
            .expect("decode schema from flight info");
        assert_eq!(&decoded, schema.as_ref());
    }

    #[tokio::test]
    async fn stream_yields_all_messages() {
        use futures::StreamExt;
        let (schema, batch) = sample();
        let expected = batches_to_flight_data(&schema, &[batch.clone()]).unwrap().len();

        let mut s = flight_data_stream(&schema, &[batch]).unwrap();
        let mut n = 0;
        while let Some(item) = s.next().await {
            item.expect("stream item ok");
            n += 1;
        }
        assert_eq!(n, expected);
    }
}
