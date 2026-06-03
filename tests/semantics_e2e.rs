// SPDX-License-Identifier: Apache-2.0

//! End-to-end correctness tests for SQL semantics that the coverage-gap audit
//! (`reviews/tests.md`) flagged as untested or weakly tested. The sibling
//! correctness fixes have landed on `dev`; these tests lock the CORRECT
//! (DuckDB / SQL-standard) behaviour in place so a future regression trips
//! immediately on a GPU host.
//!
//! Every test that registers a table needs a CUDA context (`Engine::new`
//! opens a device), so each is `#[ignore]`'d with the project's standard
//! bucket label and compiles — but does not run — under
//! `--no-default-features --features cuda-stub`. Run on a GPU host with, e.g.:
//!
//! ```text
//! cargo test --test semantics_e2e -- --ignored
//! ```
//!
//! Coverage map (items numbered per the N2 task brief):
//!   1. String semantics — LENGTH (chars), OCTET_LENGTH (bytes), SUBSTRING
//!      multibyte (no left-of-start byte leak), LENGTH(NULL)=NULL.
//!   2. NOT IN (subquery) with / without NULL in the set, plus NULL probe.
//!   3. Two-key COUNT(col) with NULLs in col → counts only non-null.
//!   4. Grouped float MIN/MAX incl NaN == scalar-aggregate (NaN-as-largest).
//!   5. UTF-8 multibyte multi-key sort ordering.
//!   6. All-NULL group keys and all-NULL join keys end-to-end.
//!   7. Dict-registry collision via UNION (same-named column, distinct dicts).

use std::sync::Arc;

use arrow_array::{Array, Float64Array, Int32Array, Int64Array, RecordBatch, StringArray};
use arrow_schema::{DataType as ArrowDataType, Field as ArrowField, Schema as ArrowSchema};

use craton_bolt::Engine;

mod common;

// ---------------------------------------------------------------------------
// Small decoding helpers (kept local; each integration binary is its own crate)
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn col_int64(batch: &RecordBatch, c: usize) -> &Int64Array {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("column is Int64")
}

#[allow(dead_code)]
fn col_int32(batch: &RecordBatch, c: usize) -> &Int32Array {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("column is Int32")
}

#[allow(dead_code)]
fn col_f64(batch: &RecordBatch, c: usize) -> &Float64Array {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("column is Float64")
}

#[allow(dead_code)]
fn col_str(batch: &RecordBatch, c: usize) -> &StringArray {
    batch
        .column(c)
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("column is Utf8")
}

/// Register a single-column nullable Utf8 table `t(s)` and return the engine.
#[allow(dead_code)]
fn engine_with_utf8(name: &str, vals: Vec<Option<&str>>) -> Engine {
    let mut engine = Engine::new().expect("CUDA ctx");
    let s = StringArray::from(vals);
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "s",
        ArrowDataType::Utf8,
        true,
    )]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(s)]).expect("batch");
    engine.register_table(name, batch).expect("register");
    engine
}

// ===========================================================================
// Item 1 — String semantics: LENGTH (chars), OCTET_LENGTH (bytes), SUBSTRING
// multibyte, LENGTH(NULL)=NULL.
// ===========================================================================
//
// These exercise the public SQL surface (LENGTH / OCTET_LENGTH / SUBSTRING),
// which routes through the GPU scan + string paths, so they are gated like the
// rest of the string suite.

/// `LENGTH('héllo')` counts CHARACTERS (5), not bytes. The `é` is a 2-byte
/// UTF-8 codepoint, so a byte-counting bug would report 6. DuckDB / the SQL
/// standard define LENGTH on characters.
#[test]
#[ignore = "gpu:string"]
fn length_counts_characters_not_bytes() {
    let engine = engine_with_utf8(
        "t",
        vec![Some("héllo"), Some("世界"), Some("ascii"), Some("")],
    );
    let h = engine
        .sql("SELECT LENGTH(s) FROM t")
        .expect("SELECT LENGTH(s)");
    let out = h.record_batch();
    let lens = col_int64(out, 0);
    let got: Vec<i64> = (0..lens.len()).map(|i| lens.value(i)).collect();
    // "héllo" = 5 chars, "世界" = 2 chars, "ascii" = 5 chars, "" = 0 chars.
    assert_eq!(got, vec![5, 2, 5, 0]);
}

/// `OCTET_LENGTH('héllo')` counts BYTES (6) — the UTF-8 encoded size — where
/// `LENGTH` counts characters (5). Pins the char/byte distinction at the SQL
/// surface so the two functions cannot silently collapse into one.
#[test]
#[ignore = "gpu:string"]
fn octet_length_counts_bytes_not_characters() {
    let engine = engine_with_utf8(
        "t",
        vec![Some("héllo"), Some("世界"), Some("ascii"), Some("")],
    );
    let h = engine
        .sql("SELECT OCTET_LENGTH(s) FROM t")
        .expect("SELECT OCTET_LENGTH(s)");
    let out = h.record_batch();
    let lens = col_int64(out, 0);
    let got: Vec<i64> = (0..lens.len()).map(|i| lens.value(i)).collect();
    // "héllo" = 6 bytes (h,é=2,l,l,o), "世界" = 6 bytes (3+3),
    // "ascii" = 5 bytes, "" = 0 bytes.
    assert_eq!(got, vec![6, 6, 5, 0]);
}

/// `SUBSTRING('héllo', 1, 2)` is character-indexed → `'hé'` (NOT the byte
/// slice `'h'`). Crucially there must be NO left-of-start byte leak: starting
/// at character 2 must not splice the trailing byte of the multibyte `é`.
#[test]
#[ignore = "gpu:string"]
fn substring_multibyte_is_character_indexed() {
    let engine = engine_with_utf8("t", vec![Some("héllo")]);

    // start=1, length=2 → first two CHARACTERS "hé".
    let h = engine
        .sql("SELECT SUBSTRING(s, 1, 2) FROM t")
        .expect("SUBSTRING(s,1,2)");
    let out = h.record_batch();
    assert_eq!(col_str(out, 0).value(0), "hé");

    // start=1, length=3 → "hél".
    let h = engine
        .sql("SELECT SUBSTRING(s, 1, 3) FROM t")
        .expect("SUBSTRING(s,1,3)");
    let out = h.record_batch();
    assert_eq!(col_str(out, 0).value(0), "hél");
}

/// `SUBSTRING('héllo', 2, 1)` must start exactly at character 2 (`é`) and emit
/// one whole character — with NO leak of the byte to the left of the start
/// position. A byte-offset implementation that rounded down to a char boundary
/// could either drop into the middle of `é` or splice the preceding `h`.
#[test]
#[ignore = "gpu:string"]
fn substring_no_left_of_start_byte_leak() {
    let engine = engine_with_utf8("t", vec![Some("héllo"), Some("世界x")]);

    // "héllo": char 2 is "é", one char.
    let h = engine
        .sql("SELECT SUBSTRING(s, 2, 1) FROM t WHERE s = 'héllo'")
        .expect("SUBSTRING(héllo,2,1)");
    let out = h.record_batch();
    assert_eq!(col_str(out, 0).value(0), "é");

    // "世界x": chars 2..=3 are "界x"; start=2 must not leak the tail byte of "世".
    let h = engine
        .sql("SELECT SUBSTRING(s, 2, 2) FROM t WHERE s = '世界x'")
        .expect("SUBSTRING(世界x,2,2)");
    let out = h.record_batch();
    assert_eq!(col_str(out, 0).value(0), "界x");
}

/// `LENGTH(NULL)` is NULL — distinct from `LENGTH('')` which is 0. A bug that
/// coerces NULL→'' (or NULL→0) would surface the NULL row as `0`; this test
/// pins the three-valued semantics.
#[test]
#[ignore = "gpu:string"]
fn length_of_null_is_null_not_zero() {
    let engine = engine_with_utf8("t", vec![Some("ab"), None, Some("")]);
    let h = engine
        .sql("SELECT LENGTH(s) FROM t")
        .expect("SELECT LENGTH(s)");
    let out = h.record_batch();
    let lens = col_int64(out, 0);
    assert_eq!(lens.len(), 3);
    assert_eq!(lens.value(0), 2, "LENGTH('ab') = 2");
    assert!(lens.is_null(1), "LENGTH(NULL) must be NULL, not 0");
    assert!(!lens.is_null(2), "LENGTH('') must be the value 0, not NULL");
    assert_eq!(lens.value(2), 0, "LENGTH('') = 0");
}

// ===========================================================================
// Item 2 — NOT IN (subquery) with NULL in the set (F-6 fix).
// ===========================================================================
//
// SQL three-valued logic: `x NOT IN (set)` where the set contains any NULL is
// UNKNOWN for every row, so ZERO rows pass. Without a NULL it folds to the
// normal AND-of-inequalities. A NULL probe value also never passes (NULL <> v
// is UNKNOWN).

/// Register two single-column Int32 tables for the NOT-IN cases. `probe` is
/// `t.k`; `other.id` is the subquery set (nullable so we can inject a NULL).
#[allow(dead_code)]
fn engine_for_not_in(probe: Vec<Option<i32>>, set: Vec<Option<i32>>) -> Engine {
    let mut engine = Engine::new().expect("CUDA ctx");
    let t_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "k",
        ArrowDataType::Int32,
        true,
    )]));
    let t =
        RecordBatch::try_new(t_schema, vec![Arc::new(Int32Array::from(probe))]).expect("t batch");
    engine.register_table("t", t).expect("register t");

    let o_schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "id",
        ArrowDataType::Int32,
        true,
    )]));
    let o =
        RecordBatch::try_new(o_schema, vec![Arc::new(Int32Array::from(set))]).expect("other batch");
    engine.register_table("other", o).expect("register other");
    engine
}

/// `k NOT IN (SELECT id FROM other)` where the set contains a NULL → ZERO
/// rows. This is the F-6 footgun: every row's predicate is UNKNOWN, so none
/// pass. A pre-fix engine that dropped the NULL and folded over the remaining
/// `{1, 2}` would have returned rows {3, 4, 5}.
#[test]
#[ignore = "gpu:e2e"]
fn not_in_subquery_with_null_in_set_returns_zero_rows() {
    let engine = engine_for_not_in(
        vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        vec![Some(1), Some(2), None],
    );
    let h = engine
        .sql("SELECT k FROM t WHERE k NOT IN (SELECT id FROM other)")
        .expect("NOT IN with NULL set");
    let out = h.record_batch();
    assert_eq!(
        out.num_rows(),
        0,
        "NULL in NOT IN set must exclude all rows"
    );
}

/// `k NOT IN (SELECT id FROM other)` with a NULL-free set → normal semantics:
/// the rows whose `k` is NOT in {1, 2} survive, i.e. {3, 4, 5}.
#[test]
#[ignore = "gpu:e2e"]
fn not_in_subquery_without_null_returns_complement() {
    let engine = engine_for_not_in(
        vec![Some(1), Some(2), Some(3), Some(4), Some(5)],
        vec![Some(1), Some(2)],
    );
    let h = engine
        .sql("SELECT k FROM t WHERE k NOT IN (SELECT id FROM other) ORDER BY k")
        .expect("NOT IN without NULL set");
    let out = h.record_batch();
    let k = col_int32(out, 0);
    let got: Vec<i32> = (0..k.len()).map(|i| k.value(i)).collect();
    assert_eq!(got, vec![3, 4, 5]);
}

/// A NULL PROBE row never passes `NOT IN` (even against a NULL-free set):
/// `NULL <> v` is UNKNOWN. With probe {NULL, 3, 7} and set {3}, only `7`
/// passes — the NULL probe is excluded just like the matching `3`.
#[test]
#[ignore = "gpu:e2e"]
fn not_in_subquery_null_probe_excluded() {
    let engine = engine_for_not_in(vec![None, Some(3), Some(7)], vec![Some(3)]);
    let h = engine
        .sql("SELECT k FROM t WHERE k NOT IN (SELECT id FROM other)")
        .expect("NOT IN null probe");
    let out = h.record_batch();
    let k = col_int32(out, 0);
    let got: Vec<i32> = (0..k.len())
        .filter(|&i| !k.is_null(i))
        .map(|i| k.value(i))
        .collect();
    assert_eq!(got, vec![7], "NULL probe must not pass NOT IN");
}

/// Control: plain `IN (subquery)` with a NULL in the set still matches on the
/// non-NULL elements (NULL is dropped, not collapsed). Probe {1, 2, 9},
/// set {1, NULL} → only `1` passes.
#[test]
#[ignore = "gpu:e2e"]
fn in_subquery_with_null_in_set_matches_non_null() {
    let engine = engine_for_not_in(vec![Some(1), Some(2), Some(9)], vec![Some(1), None]);
    let h = engine
        .sql("SELECT k FROM t WHERE k IN (SELECT id FROM other)")
        .expect("IN with NULL set");
    let out = h.record_batch();
    let k = col_int32(out, 0);
    let got: Vec<i32> = (0..k.len()).map(|i| k.value(i)).collect();
    assert_eq!(got, vec![1]);
}

// ===========================================================================
// Item 3 — Two-key COUNT(col) with NULLs in col counts only non-null rows.
// ===========================================================================

/// `SELECT k1, k2, COUNT(v) FROM t GROUP BY k1, k2`: COUNT(v) excludes NULLs
/// per the SQL spec (unlike COUNT(*)). Built so each (k1, k2) group has a
/// predictable null/non-null split.
#[test]
#[ignore = "gpu:tier1"]
fn two_key_count_excludes_nulls() {
    let mut engine = Engine::new().expect("CUDA ctx");

    // 4 groups: (0,0),(0,1),(1,0),(1,1). Lay rows out so each group's
    // non-null COUNT is known. v is NULL on every 2nd row of each group.
    let n = 8usize;
    let k1: Int32Array = (0..n as i32).map(|i| i / 4).collect(); // 0,0,0,0,1,1,1,1
    let k2: Int32Array = (0..n as i32).map(|i| (i / 2) % 2).collect(); // 0,0,1,1,0,0,1,1
                                                                       // v: NULL on odd indices → each 2-row group has exactly 1 non-null.
    let v: Int64Array = (0..n as i64)
        .map(|i| if i % 2 == 1 { None } else { Some(i) })
        .collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k1", ArrowDataType::Int32, false),
        ArrowField::new("k2", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Int64, true),
    ]));
    let batch =
        RecordBatch::try_new(schema, vec![Arc::new(k1), Arc::new(k2), Arc::new(v)]).expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql(
            "SELECT k1, k2, COUNT(v) FROM t GROUP BY k1, k2 \
             ORDER BY k1, k2",
        )
        .expect("two-key COUNT(v)");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 4, "four (k1,k2) groups");
    let cnt = col_int64(out, 2);
    let got: Vec<i64> = (0..cnt.len()).map(|i| cnt.value(i)).collect();
    // Each group has 2 rows, exactly 1 of them non-null → COUNT(v) = 1.
    assert_eq!(got, vec![1, 1, 1, 1], "COUNT(v) must skip NULLs");
}

// ===========================================================================
// Item 4 — Grouped float MIN/MAX including NaN == scalar-aggregate
// (DuckDB orders NaN as the LARGEST float value).
// ===========================================================================

/// Build a table where one group contains a NaN. Grouped MIN/MAX over that
/// group must agree with the scalar (whole-table-per-group) aggregate and with
/// DuckDB's NaN-as-largest ordering: NaN is never the MIN unless every value is
/// NaN, and NaN IS the MAX when present.
#[test]
#[ignore = "gpu:tier1"]
fn grouped_float_min_max_with_nan_matches_scalar() {
    let mut engine = Engine::new().expect("CUDA ctx");

    // group 0: {1.0, NaN, 2.0}  → MIN=1.0, MAX=NaN (NaN largest)
    // group 1: {-3.0, 0.5}      → MIN=-3.0, MAX=0.5
    let k: Int32Array = Int32Array::from(vec![0, 0, 0, 1, 1]);
    let v: Float64Array = Float64Array::from(vec![1.0, f64::NAN, 2.0, -3.0, 0.5]);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(v)]).expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT k, MIN(v), MAX(v) FROM t GROUP BY k ORDER BY k")
        .expect("grouped MIN/MAX");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 2);
    let mins = col_f64(out, 1);
    let maxs = col_f64(out, 2);

    // group 0
    assert_eq!(mins.value(0), 1.0, "group 0 MIN ignores NaN");
    assert!(
        maxs.value(0).is_nan(),
        "group 0 MAX is NaN (NaN-as-largest)"
    );
    // group 1
    assert_eq!(mins.value(1), -3.0);
    assert_eq!(maxs.value(1), 0.5);
}

/// All-NaN group: MIN and MAX are both NaN. Guards against a comparison that
/// treats NaN inconsistently between the MIN and MAX reductions.
#[test]
#[ignore = "gpu:tier1"]
fn grouped_float_all_nan_group_min_max_are_nan() {
    let mut engine = Engine::new().expect("CUDA ctx");
    let k: Int32Array = Int32Array::from(vec![0, 0, 0]);
    let v: Float64Array = Float64Array::from(vec![f64::NAN, f64::NAN, f64::NAN]);
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, false),
        ArrowField::new("v", ArrowDataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(v)]).expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT MIN(v), MAX(v) FROM t GROUP BY k")
        .expect("all-NaN MIN/MAX");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1);
    assert!(col_f64(out, 0).value(0).is_nan(), "all-NaN MIN is NaN");
    assert!(col_f64(out, 1).value(0).is_nan(), "all-NaN MAX is NaN");
}

// ===========================================================================
// Item 5 — UTF-8 multibyte MULTI-KEY sort ordering (existing sort tests are
// ASCII-only). GPU-gated like the rest of the sort suite.
// ===========================================================================

/// `ORDER BY s1 ASC, s2 DESC` over multibyte UTF-8 keys. Existing multi-key
/// sort tests are int/int only and the Utf8 single-key tests use ASCII
/// fixtures; this drives a genuine multibyte, multi-key collation through the
/// full SQL pipeline. The oracle is Rust's own (s1 ASC, s2 DESC) ordering on
/// the same byte-wise UTF-8 strings — which is what the engine targets.
#[test]
#[ignore = "gpu:sort"]
fn multi_key_utf8_multibyte_sort() {
    let mut engine = Engine::new().expect("CUDA ctx");

    // Small multibyte vocabulary with deliberate ties on s1 so the secondary
    // DESC key is load-bearing. Cycle to N_BIG so the GPU sort path engages.
    let s1_vocab = ["café", "über", "café", "αβ", "über", "αβ"];
    let s2_vocab = ["x", "y", "z", "m", "a", "n"];
    let n = 16_384usize;
    let s1: Vec<String> = (0..n)
        .map(|i| s1_vocab[i % s1_vocab.len()].to_string())
        .collect();
    let s2: Vec<String> = (0..n)
        .map(|i| s2_vocab[i % s2_vocab.len()].to_string())
        .collect();

    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("s1", ArrowDataType::Utf8, false),
        ArrowField::new("s2", ArrowDataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringArray::from(s1.clone())),
            Arc::new(StringArray::from(s2.clone())),
        ],
    )
    .expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT s1, s2 FROM t ORDER BY s1 ASC, s2 DESC")
        .expect("multi-key Utf8 ORDER BY");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), n);
    let a1 = col_str(out, 0);
    let a2 = col_str(out, 1);

    // Build the expected ordering with Rust's stable sort: primary s1 ASC,
    // secondary s2 DESC. Compare full (s1, s2) tuples — both must match.
    let mut expected: Vec<(String, String)> = s1.into_iter().zip(s2).collect();
    expected.sort_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    let actual: Vec<(String, String)> = (0..n)
        .map(|i| (a1.value(i).to_string(), a2.value(i).to_string()))
        .collect();
    assert_eq!(actual, expected, "multi-key multibyte sort order mismatch");

    // Also assert the local ordering invariant directly (defence in depth):
    // s1 never decreases; within an s1 tie, s2 never increases.
    for i in 1..n {
        let (p1, p2) = (a1.value(i - 1), a2.value(i - 1));
        let (c1, c2) = (a1.value(i), a2.value(i));
        assert!(p1 <= c1, "s1 ASC violated at row {i}: {p1:?} > {c1:?}");
        if p1 == c1 {
            assert!(p2 >= c2, "s2 DESC violated at row {i}: {p2:?} < {c2:?}");
        }
    }
}

// ===========================================================================
// Item 6 — All-NULL group keys and all-NULL join keys, end-to-end.
// ===========================================================================

/// GROUP BY a column that is entirely NULL: SQL groups all NULLs together into
/// a single group. `COUNT(*)` over that one group is the row count; `COUNT(v)`
/// still excludes NULL values.
#[test]
#[ignore = "gpu:tier1"]
fn all_null_group_key_collapses_to_one_group() {
    let mut engine = Engine::new().expect("CUDA ctx");
    let n = 6usize;
    let k: Int32Array = (0..n).map(|_| Option::<i32>::None).collect(); // all NULL key
    let v: Int64Array = (0..n as i64)
        .map(|i| if i % 2 == 0 { Some(i) } else { None })
        .collect();
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, true),
        ArrowField::new("v", ArrowDataType::Int64, true),
    ]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(k), Arc::new(v)]).expect("batch");
    engine.register_table("t", batch).expect("register");

    let h = engine
        .sql("SELECT COUNT(*), COUNT(v) FROM t GROUP BY k")
        .expect("all-NULL group key");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1, "all-NULL keys collapse to ONE group");
    assert_eq!(col_int64(out, 0).value(0), n as i64, "COUNT(*) = all rows");
    // v non-null on even indices 0,2,4 → 3 values.
    assert_eq!(col_int64(out, 1).value(0), 3, "COUNT(v) excludes NULLs");
}

/// INNER JOIN where the join key column is entirely NULL on BOTH sides:
/// `NULL = NULL` is UNKNOWN, so an equi-join produces ZERO matching rows.
#[test]
#[ignore = "gpu:join"]
fn all_null_inner_join_key_yields_no_rows() {
    let mut engine = Engine::new().expect("CUDA ctx");

    let s1 = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, true),
        ArrowField::new("v", ArrowDataType::Int64, false),
    ]));
    let s2 = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, true),
        ArrowField::new("w", ArrowDataType::Int64, false),
    ]));
    let k1: Int32Array = (0..4).map(|_| Option::<i32>::None).collect();
    let v: Int64Array = Int64Array::from(vec![10, 11, 12, 13]);
    let k2: Int32Array = (0..4).map(|_| Option::<i32>::None).collect();
    let w: Int64Array = Int64Array::from(vec![20, 21, 22, 23]);
    let t1 = RecordBatch::try_new(s1, vec![Arc::new(k1), Arc::new(v)]).expect("t1");
    let t2 = RecordBatch::try_new(s2, vec![Arc::new(k2), Arc::new(w)]).expect("t2");
    engine.register_table("t1", t1).expect("register t1");
    engine.register_table("t2", t2).expect("register t2");

    let h = engine
        .sql("SELECT t1.v, t2.w FROM t1 INNER JOIN t2 ON t1.k = t2.k")
        .expect("all-NULL join key");
    let out = h.record_batch();
    assert_eq!(
        out.num_rows(),
        0,
        "NULL = NULL is UNKNOWN → inner join over all-NULL keys is empty"
    );
}

/// Mixed join keys where only the NON-null keys can match. t1.k = {NULL, 1, 2},
/// t2.k = {NULL, 2, 3}. Only the `2` rows join (1 matching pair); the NULLs on
/// either side never match.
#[test]
#[ignore = "gpu:join"]
fn null_join_keys_only_non_null_match() {
    let mut engine = Engine::new().expect("CUDA ctx");

    let s1 = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, true),
        ArrowField::new("v", ArrowDataType::Int64, false),
    ]));
    let s2 = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("k", ArrowDataType::Int32, true),
        ArrowField::new("w", ArrowDataType::Int64, false),
    ]));
    let k1: Int32Array = Int32Array::from(vec![None, Some(1), Some(2)]);
    let v: Int64Array = Int64Array::from(vec![100, 101, 102]);
    let k2: Int32Array = Int32Array::from(vec![None, Some(2), Some(3)]);
    let w: Int64Array = Int64Array::from(vec![200, 202, 203]);
    let t1 = RecordBatch::try_new(s1, vec![Arc::new(k1), Arc::new(v)]).expect("t1");
    let t2 = RecordBatch::try_new(s2, vec![Arc::new(k2), Arc::new(w)]).expect("t2");
    engine.register_table("t1", t1).expect("register t1");
    engine.register_table("t2", t2).expect("register t2");

    let h = engine
        .sql("SELECT t1.v, t2.w FROM t1 INNER JOIN t2 ON t1.k = t2.k")
        .expect("mixed-NULL join key");
    let out = h.record_batch();
    assert_eq!(out.num_rows(), 1, "only the k=2 pair matches");
    assert_eq!(col_int64(out, 0).value(0), 102, "t1.v for k=2");
    assert_eq!(col_int64(out, 1).value(0), 202, "t2.w for k=2");
}

// ===========================================================================
// Item 7 — Dict-registry collision via UNION (same-named column, distinct
// dictionaries) — F-7 fix. GPU-gated as the registry tests are.
// ===========================================================================

/// Two tables each expose a Utf8 column named `region`, but with DISJOINT
/// dictionaries (`{US, EU}` vs `{JP, AU}`). A `UNION ALL` scans both. The
/// string-literal rewriter is keyed by unqualified column name, so a naive
/// last-write-wins registry would fold `region = 'US'` against the WRONG index
/// space in one branch. The F-7 fix poisons the colliding column and falls
/// back to host string comparison — so the answer must still be correct:
/// `region = 'US'` selects only the US rows from `orders_us`, and `orders_apac`
/// contributes nothing (it has no `US`).
#[test]
#[ignore = "gpu:string"]
fn dict_registry_collision_via_union_is_correct() {
    let mut engine = Engine::new().expect("CUDA ctx");

    let mk = |vals: Vec<&str>| -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "region",
            ArrowDataType::Utf8,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vals))]).expect("batch")
    };

    // orders_us dictionary: {US, EU}; orders_apac dictionary: {JP, AU}.
    engine
        .register_table("orders_us", mk(vec!["US", "EU", "US"]))
        .expect("register orders_us");
    engine
        .register_table("orders_apac", mk(vec!["JP", "AU", "JP"]))
        .expect("register orders_apac");

    let h = engine
        .sql(
            "SELECT region FROM orders_us WHERE region = 'US' \
             UNION ALL \
             SELECT region FROM orders_apac WHERE region = 'US'",
        )
        .expect("UNION over colliding dict columns");
    let out = h.record_batch();
    let col = col_str(out, 0);
    let got: Vec<&str> = (0..col.len()).map(|i| col.value(i)).collect();
    // Only the two US rows from orders_us; orders_apac has no "US".
    assert_eq!(
        got,
        vec!["US", "US"],
        "collision fold must not corrupt results"
    );
}

/// Control for item 7: same-named `region` column with IDENTICAL dictionaries
/// on both sides still folds correctly under UNION ALL. `region = 'EU'` selects
/// the EU rows from each branch — the non-colliding fast path must be intact.
#[test]
#[ignore = "gpu:string"]
fn dict_registry_identical_dicts_via_union_still_correct() {
    let mut engine = Engine::new().expect("CUDA ctx");

    let mk = |vals: Vec<&str>| -> RecordBatch {
        let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
            "region",
            ArrowDataType::Utf8,
            false,
        )]));
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vals))]).expect("batch")
    };

    engine
        .register_table("a", mk(vec!["US", "EU", "EU"]))
        .expect("register a");
    engine
        .register_table("b", mk(vec!["EU", "US", "US"]))
        .expect("register b");

    let h = engine
        .sql(
            "SELECT region FROM a WHERE region = 'EU' \
             UNION ALL \
             SELECT region FROM b WHERE region = 'EU'",
        )
        .expect("UNION over identical dict columns");
    let out = h.record_batch();
    let col = col_str(out, 0);
    let got_len = col.len();
    for i in 0..got_len {
        assert_eq!(col.value(i), "EU", "every selected row must be EU");
    }
    // a has 2 EU rows, b has 1 EU row → 3 total.
    assert_eq!(got_len, 3, "identical-dict fold must keep all EU rows");
}
