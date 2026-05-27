// SPDX-License-Identifier: Apache-2.0

//! End-to-end ORDER BY tests for the GPU sort path.
//!
//! All tests in this file are `#[ignore]`d: they exercise the full engine
//! (CUDA context + upload + sort kernel + download) and only make sense on
//! a host with a working GPU. Run with `cargo test --test sort_e2e -- --ignored`.

use std::sync::Arc;

use arrow_array::builder::StringDictionaryBuilder;
use arrow_array::types::Int32Type as ArrowInt32Type;
use arrow_array::{Array, DictionaryArray, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::exec::Engine;

/// Build a single-column `DictionaryArray<Int32, Utf8>` with a known mix of
/// values and NULLs. Layout:
///   row 0: "b"
///   row 1: NULL
///   row 2: "a"
///   row 3: "c"
///   row 4: NULL
///   row 5: "a"
///
/// Three distinct non-null values plus two NULLs is small enough to verify
/// the sorted output by inspection.
fn dict_utf8_with_nulls() -> DictionaryArray<ArrowInt32Type> {
    let mut b: StringDictionaryBuilder<ArrowInt32Type> = StringDictionaryBuilder::new();
    b.append_value("b");
    b.append_null();
    b.append_value("a");
    b.append_value("c");
    b.append_null();
    b.append_value("a");
    b.finish()
}

/// Build a `RecordBatch` whose single column `col` is the dict-encoded
/// fixture above.
fn dict_batch() -> RecordBatch {
    let arr = dict_utf8_with_nulls();
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "col",
        ArrowDataType::Dictionary(
            Box::new(ArrowDataType::Int32),
            Box::new(ArrowDataType::Utf8),
        ),
        true,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(arr)]).expect("build batch")
}

/// Decode the output `RecordBatch`'s first column into a `Vec<Option<String>>`
/// for comparison. Handles both `StringArray` and `DictionaryArray<Int32, Utf8>`
/// output shapes — the engine may downconvert on read, depending on the sort
/// path's output mode.
fn decode_utf8_column(batch: &RecordBatch) -> Vec<Option<String>> {
    let arr = batch.column(0);
    if let Some(sa) = arr.as_any().downcast_ref::<StringArray>() {
        return (0..sa.len())
            .map(|i| {
                if sa.is_null(i) {
                    None
                } else {
                    Some(sa.value(i).to_string())
                }
            })
            .collect();
    }
    if let Some(da) = arr
        .as_any()
        .downcast_ref::<DictionaryArray<ArrowInt32Type>>()
    {
        let values = da
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("dict values must be Utf8");
        let keys = da.keys();
        return (0..da.len())
            .map(|i| {
                if keys.is_null(i) {
                    None
                } else {
                    let k = keys.value(i) as usize;
                    if values.is_null(k) {
                        None
                    } else {
                        Some(values.value(k).to_string())
                    }
                }
            })
            .collect();
    }
    panic!(
        "unexpected output column dtype {:?} — expected Utf8 or Dictionary<Int32, Utf8>",
        arr.data_type()
    );
}

/// `ORDER BY col ASC NULLS LAST` on a Dict-encoded column with NULLs:
/// NULLs sort to the end, real values ascend lex.
#[test]
#[ignore = "requires CUDA device"]
fn order_by_dict_utf8_nulls_last() {
    let mut engine = Engine::new().expect("engine");
    engine.register_table("t", dict_batch()).expect("register");

    let h = engine
        .sql("SELECT col FROM t ORDER BY col NULLS LAST")
        .expect("sql");
    let batch = h.record_batch();
    let got = decode_utf8_column(batch);

    let want = vec![
        Some("a".to_string()),
        Some("a".to_string()),
        Some("b".to_string()),
        Some("c".to_string()),
        None,
        None,
    ];
    assert_eq!(got, want);
}

/// `ORDER BY col ASC NULLS FIRST` on a Dict-encoded column with NULLs:
/// NULLs sort to the front, real values ascend lex.
#[test]
#[ignore = "requires CUDA device"]
fn order_by_dict_utf8_nulls_first() {
    let mut engine = Engine::new().expect("engine");
    engine.register_table("t", dict_batch()).expect("register");

    let h = engine
        .sql("SELECT col FROM t ORDER BY col NULLS FIRST")
        .expect("sql");
    let batch = h.record_batch();
    let got = decode_utf8_column(batch);

    let want = vec![
        None,
        None,
        Some("a".to_string()),
        Some("a".to_string()),
        Some("b".to_string()),
        Some("c".to_string()),
    ];
    assert_eq!(got, want);
}
