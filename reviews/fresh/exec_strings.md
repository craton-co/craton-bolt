# Code Review — String-Operations Execution Subsystem

Scope: `src/exec/string_col.rs`, `string_length.rs`, `string_like.rs`,
`string_ops.rs`, `string_ops_extended.rs`, `string_project.rs`, `src/exec/like.rs`.
Tests: `tests/like_test.rs`, `string_fns_sql_test.rs`, `string_ops_e2e.rs`,
`case_folding_test.rs`.

Reviewer focus: UTF-8 correctness, LIKE/ILIKE semantics, case folding,
buffer overruns, offset off-by-one, NULL handling, empty strings.

Overall: this is well-engineered, defensive code. The dictionary-remap design
sidesteps variable-width device writes; every device path has a host mirror and
strict bounds checks; NULL/3VL handling is consistently correct; char-vs-byte is
treated carefully and tested. No crashes, no out-of-bounds, no UB found. Findings
are mostly semantic edge cases and missing test coverage, not memory-safety bugs.

---

## 1. CODE REVIEW (ranked by severity)

### CRITICAL
None.

### HIGH

**H1 — ILIKE case folding can desynchronise `_` wildcard with multi-char
lowercase expansions.** `like.rs:177-183` folds the *pattern* with
`to_lowercase()` and `matches()` (`like.rs:202-206`) folds the *input* the same
way, then matches in `char` units (`generic_match`, `like.rs:399`). For most text
this is correct, but `to_lowercase` is not length-preserving per codepoint: e.g.
`'İ'` (U+0130) lowercases to two scalars `i` + U+0307. A pattern `'_'` (one `_`)
is meant to match exactly one *input character*; after folding the input, one
source char can become two folded chars, so `'_'` against `'İ'` will fail to
match (two folded chars vs one `_`), and conversely literal lengths shift. This
diverges from engines that case-fold without re-segmenting. The fast-path shapes
(Exact/Prefix/Suffix/Contains) are unaffected (they compare folded substrings),
so impact is limited to `_`-bearing ILIKE patterns over the handful of
expanding-lowercase codepoints. Worth a documented limitation + test. (`like.rs:165-225`)

**H2 — `UPPER`/`LOWER` dictionary remap is correct, but ASCII-vs-Unicode routing
relies on a whole-dictionary scan that can silently change results across the
GPU/host boundary for mixed dictionaries.** `string_project.rs:346` (`dict_is_ascii`)
routes the *entire* column to the host fallback if *any* entry has a non-ASCII
byte, which is correct. But note the GPU byte-fold (`apply_ascii_byte`,
`string_project.rs:97-114`) and the host Unicode fold (`apply_host`) genuinely
disagree for non-ASCII (e.g. `ß`→`SS`), so correctness hinges entirely on that
guard never being bypassed. This is sound today; flag it as a fragile invariant —
any future caller that forgets to consult `dict_is_ascii` before launching the
GPU path will produce wrong results for Unicode. (`string_project.rs:340-348`)

### MEDIUM

**M1 — Greek final sigma / Turkish-I not handled in `UPPER`/`LOWER`; documented
but lossy round-trips.** `string_ops.rs:111-120` uses `to_uppercase`/`to_lowercase`
(Unicode default, locale-invariant). `upper_unicode_lowercase_collapses`
(`string_ops.rs:498-506`) shows `σ`/`ς` collapse under UPPER — correct for the
default mapping, but `LOWER(UPPER(x))` is not identity, and Turkish locale `İ/ı`
is wrong under default mapping. This is standard SQL behavior (no collation), so
Low-ish, but should be called out as an explicit non-goal. (`string_ops.rs:111-120`)

**M2 — `INITCAP` lower-cases the non-initial characters even when they are
already-correct multi-scalar expansions, and uses `is_alphanumeric` for word
boundaries.** `string_ops_extended.rs:387-404`. PostgreSQL's `initcap` treats only
ASCII alphanumerics specially in some builds; the Unicode `is_alphanumeric`
choice here is defensible but divergent from Postgres for e.g. superscript
digits and some symbols. Behavioral, not a bug. Add a doc note + Unicode test.

**M3 — `position_str` empty-substring returns 1 unconditionally, including for an
empty haystack.** `string_ops_extended.rs:263-273`: `POSITION('' IN '')` → 1.
ANSI/Postgres agree (`position('' in '')` = 1), so this is correct — but there is
no test for the empty-haystack/empty-needle interaction beyond
`position_empty_substring_is_one` (`position_str("", "")`), which does cover it.
OK; flagging only because it's a classic off-by-one trap. (`string_ops_extended.rs:263-273`)

**M4 — `LIKE` decomposition slices the pattern by bytes; safe only because `%` is
ASCII.** `string_like.rs:101-113` does `&pattern[1..]` / `&pattern[..len-1]`.
`%` is U+0025 (1 byte) so the slice boundaries are always on char boundaries —
correct. But if the decomposer is ever extended to strip a multibyte sentinel the
byte-index arithmetic would panic. Add an assertion or a comment pinning the
ASCII-`%` assumption (the doc at `string_like.rs:97-99` partially covers it). No
bug today.

**M5 — Stubs / unimplemented GPU producers.** Multiple `TODO(string-fn-gpu)` and
"not wired into the executor" markers: SUBSTRING GPU two-pass exists in
`jit::string_kernel` but is unused (`string_ops_extended.rs:168-170`); TRIM GPU
kernel exists but scalar TRIM routes to host (`string_ops_extended.rs:199-209`);
`octet_lengths_table_pure` is `#[allow(dead_code)]` with no caller
(`string_ops.rs:164-179`); the entire `string_like.rs` device path is
"⚠️ UNVALIDATED" — never run on hardware (`string_like.rs:7-24`). These are
honest, fenced, and host-fallback-safe, but they are dead/unvalidated weight.

### LOW

**L1 — Performance: host round-trips on every UPPER/LOWER/LENGTH.**
`string_ops.rs:194-219`, `remap_and_upload` (`string_ops.rs:322-356`) does
device→host copy of all indices + host→device re-upload per call. Documented as
intentional (no kernel launch). For LENGTH the GPU gather path exists
(`string_length.rs`), but UPPER/LOWER always pay the round trip unless routed to
the GPU two-pass producer (`string_project.rs`). Acceptable for v1; the prefix-scan
host step in `string_project.rs:225-241` is also a host stand-in for a device scan.

**L2 — `dedup_transformed` is duplicated** between `string_ops.rs:72-107` and
`string_ops_extended.rs:93-106` (acknowledged at `string_ops_extended.rs:88-92`).
Intentional decoupling; minor maintenance risk if NULL-slot semantics ever drift.

**L3 — `CONCAT`/`CONCAT_WS` output-dictionary size is uncapped** beyond `i32::MAX`
(`string_ops_extended.rs:45-49`, `concat_pure:421-491`). A pathological
cross-product could balloon host memory before hitting the i32 guard. Documented
as caller responsibility; consider a soft cap or metric.

**L4 — `LARGE`Utf8 unsupported.** `upload_utf8` only accepts `StringArray` (i32
offsets), not `LargeStringArray` (`string_col.rs:104-108`, test
`large_utf8_input_has_the_same_arrow_contract:341`). Total-bytes > i32::MAX
errors cleanly (`string_like.rs:198-204`, `string_project.rs:205-210`). Fine, but
a hard ceiling on column size.

**L5 — `mask_to_boolean_array` / `string_array_from_offsets` tolerate short mask /
validity via `unwrap_or(0)` / `unwrap_or(false)`** (`string_like.rs:222`,
`string_project.rs:202,262`). Defensive, but silently treats a truncated device
mask as "no match"/"null" rather than erroring — a kernel that wrote too few rows
would be masked. Low risk given host mirror, but an explicit length check would
surface kernel bugs.

---

## 2. TESTS

Coverage is genuinely strong on the **pure host helpers** — I'd estimate ~85% of
*host-side* logic, lower for the device paths.

What's well covered:
- LIKE: all four shapes, generic backtracking, ESCAPE (literal `%`/`_`/`\`,
  dangling-escape rejection, escape==wildcard rejection), ILIKE across shapes +
  Unicode (`école`), the V-6 ReDoS regression (`like.rs:771`), 3VL NULL.
- char-vs-byte: LENGTH/OCTET_LENGTH, SUBSTRING multibyte (2/3/4-byte + emoji),
  POSITION/LEFT/RIGHT/PAD/REVERSE character-indexed, no-left-of-start-leak
  property test (`string_ops_extended.rs:1087`).
- NULL/empty-string distinctness throughout (index-0 sentinel).
- GPU/host TRIM byte-rule mirror incl. NBSP non-stripping (`string_ops_extended.rs:1464`).

Gaps / enhancements:
- **ILIKE `_` + expanding-lowercase codepoint** (H1) — no test. Add `İ`, `ﬀ`
  ligature, German `ß` under ILIKE with `_`.
- **No execution-level LIKE/ILIKE on GPU validated** — all device tests are
  `#[ignore = "gpu:string"]` and the kernel is explicitly unvalidated
  (`string_like.rs:7`). The host mirror is tested vs `PatternMatcher` only over a
  fixed sample (`string_like.rs:328-352`); a proptest (random patterns × inputs,
  mirror == PatternMatcher == host_like) would harden this materially.
- **`string_col.rs` round-trips all `#[ignore]`** — host-only tests just assert
  the Arrow input contract, not the GPU encode/decode. No coverage of the
  validity-bitmap zip on a real download except behind `#[ignore]`.
- **CONCAT 3VL via SQL** — `host_concat_strings` NULL propagation is unit-tested,
  but no end-to-end NULL CONCAT test (e2e tests use non-null fixtures).
- **No `LARGE`Utf8 / >i32::MAX overflow path test** for the error branches in
  `build_row_aligned_*` (they are unreachable in practice but untested).
- **INITCAP / REVERSE / LPAD with combining marks / grapheme clusters** — all
  operate on scalars, so `é` as `e`+combining-accent reverses wrongly; no test
  pins this (expected, but should be documented).
- **Property test for `sql_substring`** exists for one string; broaden to random
  strings/positions.

Recommend adding a `proptest`-based equivalence harness for LIKE (host mirror vs
PatternMatcher vs host_like) and for substring/pad against a reference
implementation.

---

## 3. NEW FEATURES / DIRECTIONS

- **Collations** (highest value): current UPPER/LOWER/ILIKE are locale-invariant
  Unicode-default. Add a collation parameter (at minimum `und-ci`, Turkish-I,
  and a binary collation) to make case folding and equality locale-correct.
- **Regex** (`~`, `REGEXP_MATCHES`, `REGEXP_REPLACE`, `SIMILAR TO`): explicitly
  out of scope today (`string_ops.rs:36`). A host `regex`-crate path would be a
  natural sibling to `host_like`, with the same 3VL masking.
- **Validate & wire the GPU device paths**: LIKE matcher (`string_like.rs`),
  SUBSTRING/TRIM two-pass producers exist but are unvalidated/unwired. A GPU
  hardware test pass would let these stop being "performance can only" caveats.
- **More scalar fns**: `TRANSLATE`, `SPLIT_PART`, `ASCII`/`CHR`, `REGEXP_*`,
  `FORMAT`, `STARTS_WITH`/`ENDS_WITH`, `CONTAINS`. The host-helper pattern in
  `string_ops_extended.rs` makes these cheap to add.
- **Device prefix-scan** for the two-pass offsets (`string_project.rs:225` notes
  the host stand-in) — swap in `jit::prefix_scan` to remove a host round trip.
- **Grapheme-aware variants** (optional) for REVERSE/SUBSTRING if user-perceived
  characters matter.
- **NFC/NFKC normalization** function, since the engine deliberately does no
  normalization on upload (`string_col.rs:286`).

---

### Summary table

| ID | Sev | File:line | Issue |
|----|-----|-----------|-------|
| H1 | High | like.rs:165-225 | ILIKE `_` desync with expanding lowercase |
| H2 | High | string_project.rs:340-348 | GPU ASCII-fold correctness hinges on dict scan guard |
| M1 | Med | string_ops.rs:111-120 | No collation; Turkish-I / round-trip lossy |
| M2 | Med | string_ops_extended.rs:387-404 | INITCAP word-boundary Unicode divergence |
| M3 | Med | string_ops_extended.rs:263-273 | POSITION empty-needle (correct, trap-prone) |
| M4 | Med | string_like.rs:101-113 | Byte-slice on pattern (safe only for ASCII `%`) |
| M5 | Med | multiple | Unwired/unvalidated GPU producers, dead code |
| L1 | Low | string_ops.rs:194-356 | Host round-trips for UPPER/LOWER |
| L2 | Low | string_ops_extended.rs:93 | Duplicated `dedup_transformed` |
| L3 | Low | string_ops_extended.rs:421 | Uncapped CONCAT output dictionary |
| L4 | Low | string_col.rs:104 | No LargeUtf8 support |
| L5 | Low | string_like.rs:222, string_project.rs:202 | Silent short-mask tolerance |
