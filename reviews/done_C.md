# Agent C — string-ops remediation (done)

Branch `dev`. Scope honored: only edited
`src/exec/string_ops.rs`, `string_ops_extended.rs`, `string_length.rs`.
(`string_col.rs`, `string_project.rs`, `string_like.rs`, `like.rs`,
`validity_audit.rs` were reviewed but needed no change for the assigned
findings — see notes.) No `cargo` run per instructions. GPU paths left
host-fallback-safe; no unvalidated GPU path enabled.

## Findings implemented

### F-1 + F-2 — SUBSTRING is now CHARACTER-indexed (`string_ops_extended.rs`)
- Rewrote `sql_substring` from a byte-slice-with-round-down model to a
  codepoint-window model (`chars().skip(skip).take(take)`).
- Window = 1-based char positions `[start, start+len)`; positions `< 1` are
  honored as real out-of-range positions (they consume window length but emit
  nothing), matching DuckDB/Postgres. Examples now correct:
  - `SUBSTRING("héllo",3,2)` = `"ll"` (was the wrong `"él"`).
  - `SUBSTRING("héllo",2,4)` = `"éllo"`; `SUBSTRING("日本語",1,2)` = `"日本"`.
  - `SUBSTRING("abc",0,2)` = `"a"`; `SUBSTRING("abc",-5,3)` = `""`.
- No multibyte codepoint is ever split; **no byte left of `start` can leak**.
- Window math done in `i64` so `length = i32::MAX` ("to end") and `i32::MIN`
  start cannot overflow/panic.
- Removed now-unused `round_down_to_char_boundary` (avoids dead-code warning).
- Updated module caveats doc (byte → character semantics).
- **Tests:** rewrote `substring_unicode_boundary`,
  `sql_substring_unicode_round_down_at_start` →
  `sql_substring_unicode_character_indexed` (asserts `"ll"`, not `"él"`),
  `substring_start_clamps_to_one` → `substring_start_below_one_window_semantics`,
  `substring_str_public_wrapper_matches_internal`. Added
  `substring_multibyte_three_byte_chars`, `substring_emoji_four_byte_chars`,
  `sql_substring_no_left_of_start_leak` (exhaustive proof that no char left of
  `start` appears), `sql_substring_start_below_one_excludes_phantom_positions`,
  and a multibyte `i32::MAX` case.

### F-2 — LENGTH/CHAR_LENGTH count CHARACTERS (`string_ops.rs`, `string_length.rs`)
- `string_ops.rs::lengths_table_pure` now uses `s.chars().count()`.
  `LENGTH('héllo') = 5`.
- Added `string_ops.rs::octet_lengths_table_pure` (byte length, kept for any
  `OCTET_LENGTH` path).
- `string_length.rs::build_length_table` now emits character counts (kernel
  unchanged — it just gathers whatever per-entry value the table holds, so the
  GPU LENGTH path is now char-correct). Added
  `string_length.rs::build_octet_length_table` for byte length, factored both
  through a shared `build_length_table_with`.
- **Tests:** `lengths_table_byte_not_char` → `lengths_table_counts_characters_not_bytes`
  (+ `octet_lengths_table_counts_bytes`); `byte_length_not_char_length` →
  `length_table_counts_characters_not_bytes` (+ `octet_length_table_counts_bytes`).

### F-3 — LENGTH(NULL) = SQL NULL (`string_ops.rs`)
- `length()` now returns a validity-carrying array: dictionary index `0` (NULL)
  → SQL `NULL` (not `0`), so it is distinct from `LENGTH('') = 0`.
- Extracted pure core `length_from_indices(&[i32], &table, dict_len)` so the
  NULL / Int64 / 3VL behaviour is unit-testable without a CUDA device.
- **Tests:** `length_of_null_is_sql_null_not_zero`, `length_counts_characters`,
  `length_rejects_negative_index`.
- Note: I did NOT need to touch `validity_audit.rs` — the NULL is expressed
  directly via the `DictionaryColumn` index-0 sentinel on the result array's
  bitmap, which is cleaner than packing a separate bitmap. `packed_validity_for`
  remains available if a future caller wants the packed form.

### F-8 — LENGTH returns Int64Array (`string_ops.rs`)
- `length()` signature changed `Int32Array` → `Int64Array` (the SQL
  `Length → Int64` contract; matches the `string_length.rs` GPU path which
  already returned `Int64Array`). Import switched `Int32Array` → `Int64Array`.
- **Test:** `length_returns_int64_array` asserts `DataType::Int64`.

### F-5 — input_eq_literal honors 3VL (`string_ops.rs`)
- `input_eq_literal()` now emits `Option<bool>`: index `0` (NULL row) → SQL
  `NULL`, never `false`. Non-NULL row → `Some(idx == target)`. Literal absent
  from dict (`target = None`) still NULLs the NULL rows instead of collapsing to
  all-false. Mirrors `exec::like::host_like`. LIKE fast paths untouched
  (`string_like.rs` / `like.rs` not modified).
- Extracted pure core `eq_literal_from_indices(&[i32], Option<i32>)`.
- **Tests:** `eq_literal_null_row_is_sql_null`,
  `eq_literal_absent_literal_still_nulls_null_rows`.

## Signature / dtype changes downstream code may notice
- `string_ops::length(&DictionaryColumn) -> BoltResult<Int64Array>`
  (was `Int32Array`). **No in-tree caller** of the public fn exists today (it is
  staged behind the dictionary-rewrite path), so nothing breaks at compile time.
  Any future consumer must expect `Int64` and must handle the validity bitmap
  (NULL rows are now genuinely null, not `0`).
- `string_ops::input_eq_literal` return type is unchanged (`BooleanArray`) but
  the array now carries a **validity bitmap** (NULL rows). Callers composing it
  under `NOT`/`OR` or projecting it now get 3VL-correct results. The previous
  all-`false`-when-literal-absent shortcut is gone (NULL rows stay NULL).
- `string_length::build_length_table` signature unchanged
  (`(&[String], KeyLayout) -> BoltResult<Vec<i32>>`); values are now char
  counts. New `build_octet_length_table` available. The `engine.rs` GPU LENGTH
  executor (not in my scope) consumes `build_length_table` unchanged and now
  returns char counts automatically — no wiring change required there.

## Wiring for the orchestrator / follow-ups (NOT done — out of my file scope)
1. **`engine.rs::string_length_column` host fallback still pushes `0` for NULL
   rows** (the GPU LENGTH executor path), so the GPU/engine LENGTH path does not
   yet emit SQL NULL like `string_ops::length` now does. The F-3 task was scoped
   to `string_ops.rs::length`; making the engine path NULL-correct requires
   editing `engine.rs` (owned elsewhere). The pieces are ready: the 1-based
   `OneBasedNullSlot0` table already reserves slot 0, and `DictUtf8` validity is
   already consulted — the fix is to push `None`/null instead of `0` for the
   NULL branch and build an `Int64Array` from `Vec<Option<i64>>`.
2. **Stale comment** in `tests/string_ops_e2e.rs:181` and TODO at `:457` still
   say "Int32 byte counts" / "byte semantics" — they are comment-only (no typed
   call, compilation unaffected) but should be updated to "Int64 character
   counts" by whoever owns that test file.
3. If an `OCTET_LENGTH` SQL function is added later, route it to the new
   `octet_lengths_table_pure` / `build_octet_length_table`.

## Verification posture
- All edits are pure host logic; no GPU path was enabled. The GPU LENGTH gather
  reads the host-built table, so character-count correctness flows through with
  no kernel change and the host fallback remains the safety net.
- New/updated unit tests are pure (`#[cfg(test)]`) and need no CUDA device —
  the index→result cores were extracted specifically so the NULL/Int64/3VL/char
  behaviour is testable without constructing a `DictionaryColumn` (which would
  upload to the driver).
