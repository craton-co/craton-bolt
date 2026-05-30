# Code Review — `craton-bolt` STRING-OPS + MISC (`src/exec/`)

Reviewer: senior GPU-database reviewer
Scope: `like.rs`, `string_col.rs`, `string_length.rs`, `string_like.rs`, `string_ops.rs`,
`string_ops_extended.rs`, `string_project.rs`, `schema_convert.rs`, `subquery_resolve.rs`,
`validity_audit.rs`, `dict_registry.rs`.
Apache-2.0, owner: Craton Software Company.

Overall: the host-side string logic is unusually careful and well unit-tested for the *pure* helpers.
The LIKE matcher, dictionary remap, TRIM, and CONCAT_WS paths are correct and well-covered. The
material problems cluster around three themes: (1) **SQL semantics that silently diverge from the
standard** — most importantly `LENGTH`/`SUBSTRING` being byte-based while docs invoke "UTF-8", and
`LENGTH(NULL)=0`; (2) **an unvalidated GPU device path** (`string_like.rs`) gated behind host
mirrors but never run on hardware; (3) **dictionary registry threading / collision** caveats that
become real bugs the moment JOINs land. No memory-safety/UB issues were found in the host paths;
all index math is bounds-checked.

Severity legend: CRITICAL = wrong results / crash in supported paths; HIGH = wrong results in a
realistic path or a real safety gap; MEDIUM = correctness gap in an edge case or notable perf;
LOW = polish / doc / minor.

---

## 1. CODE REVIEW — Findings

### F-1 — `SUBSTRING` byte rounding can return bytes *before* the requested start (wrong result)
**File:** `string_ops_extended.rs:126-158` (`sql_substring`), test at `:882-895`
**Severity:** HIGH
**Description:** When `byte_start_raw` lands inside a multi-byte codepoint, `round_down_to_char_boundary`
rounds the *start* down to the previous boundary. This pulls bytes that precede the requested start
position into the result. The module's own test documents this:
`sql_substring("héllo", 3, 2)` returns `"él"` (the `é` begins at byte 1, before the requested
byte-2 start). No SQL engine returns a character the user's `start` index excluded. Combined with the
end also rounding down, the returned substring can be both shifted and shortened relative to intent.
ANSI `SUBSTRING` is defined over **characters**, not bytes, so the entire byte model is the root cause
(see F-2); but even within the byte model, rounding the *start* DOWN (rather than UP) is the wrong
direction and produces surprising data.
**Suggested fix:** Either (a) move to codepoint-indexed SUBSTRING (correct ANSI semantics — see F-2),
or (b) at minimum round the **start up** to the next char boundary and the **end down**, so the result
is always a subset of `[requested_start, requested_end)` and never leaks earlier bytes. Add tests that
assert the result never contains characters left of `start`.

### F-2 — `LENGTH` / `SUBSTRING` are byte-based but docs/users expect characters; claims "UTF-8"
**Files:** `string_ops.rs:38-39, 120-141, 174-201`; `string_length.rs:75-92`;
`string_ops_extended.rs:36-37, 126-158`
**Severity:** HIGH (correctness vs SQL standard) / documentation
**Description:** `LENGTH` returns **byte** length (`s.len()`), and `SUBSTRING` slices **bytes**. ANSI
SQL `CHAR_LENGTH`/`LENGTH` and `SUBSTRING` operate on **characters**. For any non-ASCII data this
yields wrong answers: `LENGTH('héllo')` returns 6, not 5; `SUBSTRING('日本語', 1, 2)` returns ""
(byte 2 is mid-codepoint → rounds to "" ... actually "日" needs 3 bytes so length-2 rounds down to
empty). Tests at `string_ops.rs:428-434` and `string_length.rs:170-181` explicitly lock in the byte
behaviour. The module headers describe "UTF-8 boundary semantics" which makes the byte-only behaviour
read as if it were Unicode-correct; it is not. Note: many engines (PostgreSQL with bytea, MySQL
`LENGTH`) do offer byte length — but they expose it under `OCTET_LENGTH`, and `CHAR_LENGTH` is the
default `LENGTH`. Here the only `LENGTH` is byte length.
**Suggested fix:** Decide the contract explicitly. Recommended: make `LENGTH`/`SUBSTRING` character-based
(`s.chars().count()`, `s.chars().skip(start-1).take(len)`), and add `OCTET_LENGTH` for byte length. If
byte semantics are intentional for v1, rename docs to say "OCTET/byte length, NOT character count" and
stop referencing "UTF-8" as if it implied codepoint awareness. The GPU length-gather table
(`string_length.rs`) would need a codepoint-count table instead of `s.len()`.

### F-3 — `LENGTH(NULL)` returns `0` instead of SQL `NULL`
**File:** `string_ops.rs:34-37 (caveat), 168-201`
**Severity:** HIGH
**Description:** Documented but still a correctness bug versus SQL: `LENGTH(NULL)` must be `NULL`, not
`0`. The returned `Int32Array` carries no validity bitmap, so a downstream `WHERE LENGTH(x) > 0` or
`LENGTH(x) IS NULL` gives wrong answers, and `0` is indistinguishable from `LENGTH('')`. The newer
`string_length.rs` GPU path has the same gap by construction (length table slot 0 = 0). The codebase
now *has* a validity infrastructure (`validity_audit.rs`, `packed_validity_for`), so the original
"pipeline does not plumb validity" excuse is weaker than when written.
**Suggested fix:** Produce an `Int64Array` (note: F-9 — it currently returns `Int32Array`, also wrong
output width) with a null bitmap copied from the input column's validity (index 0 → null). Mirror in
`string_length.rs`.

### F-4 — `string_like.rs` drives an **unvalidated GPU kernel**; whole device path untested on hardware
**File:** `string_like.rs:7-24` (module header), `:122-163` (`like_match_row`)
**Severity:** MEDIUM (mitigated by host fallback) — but flag prominently
**Description:** The header states the device kernel "has **not** been executed on GPU hardware".
Correctness rests entirely on the host mirror `like_match_row` matching `PatternMatcher`, plus
"PTX-shape" tests. The executor is documented as host-fallback-safe, so a latent device bug only costs
performance — *if* the fallback is actually wired for every shape. This is a stub/unverified path: it
must not be enabled as the default execution path without a hardware test pass. The `Contains` mirror
(`:147-154`) is an O(n·l) naive scan, matching the kernel's double loop — fine as a mirror, but note it
will be slow for long rows.
**Suggested fix:** Keep the device path strictly opt-in until a GPU CI lane validates
`compile_like_match_kernel` against `like_match_row` over a fuzzed corpus. Add a runtime assertion (in
debug) that the device mask equals the host mirror on the first batch.

### F-5 — `input_eq_literal` ignores NULL 3VL: `NULL = 'lit'` yields `false`, not `NULL`
**File:** `string_ops.rs:212-225`
**Severity:** MEDIUM
**Description:** The returned `BooleanArray` is built from a plain `Vec<bool>` with no validity. For a
row whose dictionary index is `0` (NULL), `i == target` is `false`. Under a `WHERE` clause `NULL = x`
filtering to false is the same observable result as NULL, so this is usually harmless — but if this
predicate is ever composed under `NOT (...)`, `OR`, or surfaced as a projected boolean, the 3VL
divergence becomes visible (`NOT (NULL = 'x')` should be NULL, here becomes `true`). The `host_like`
path (`like.rs:404-424`) gets this right with `Option<bool>`; this path is inconsistent.
**Suggested fix:** Emit `Option<bool>` with `None` for index-0 rows, mirroring `host_like`.

### F-6 — `NOT IN (subquery)` with NULLs in the set returns wrong rows (documented divergence)
**File:** `subquery_resolve.rs:184-238` (`build_in_predicate`)
**Severity:** MEDIUM
**Description:** Standard SQL: if a `NOT IN` value set contains any NULL, the predicate is `NULL` for
every row (no rows pass). This code drops NULLs and evaluates `<>` over the non-null elements, so rows
*do* pass. The `IN` (non-negated) case is fine. This is explicitly documented as a "correct-enough"
tradeoff, but it is a real wrong-results bug for `x NOT IN (SELECT nullable_col ...)`, a common
pattern, and the classic SQL footgun.
**Suggested fix:** As the doc itself suggests: when `negated` and the set contains any NULL, emit
`Expr::Literal(Bool(false))` (no rows pass) to restore strict semantics. Cheap and removes the
divergence.

### F-7 — Dictionary registry: cross-table column-name collision → wrong dictionary ("last wins")
**File:** `dict_registry.rs:20-27, 194-209` (`rewrite_plan`)
**Severity:** MEDIUM (latent; becomes HIGH once JOINs land)
**Description:** `StringPredicateRewriter` is keyed by column name only. When a plan scans two tables
that both expose a Utf8 column of the same name with different dictionaries, the last-registered
dictionary wins and the other table's `col = 'lit'` predicate is folded against the wrong index space —
silently wrong results. Documented as moot "until JOINs land", but `collect_scan_tables`
(`:313-339`) already recurses into `Join`/`SetOp`/`Union` with multiple scans, so a `UNION`/`SetOp`
over two same-column tables can hit this **today**.
**Suggested fix:** Key the rewriter by `(table, column)` (or qualify columns per-relation) before
enabling any multi-scan plan. At minimum, detect a collision in `rewrite_plan` and bail to the
unrewritten (host) plan rather than folding against the wrong dictionary.

### F-8 — `length()` returns `Int32Array` but SQL `LENGTH` is `Int64`
**File:** `string_ops.rs:168-201` (returns `Int32Array`)
**Severity:** MEDIUM
**Description:** The `string_length.rs` path correctly widens to `Int64` (`host_gather_lengths` returns
`Vec<i64>`, doc at `:19-21` says the SQL contract is Int64). The older `string_ops::length` returns
`Int32Array`, inconsistent with the declared `Length → Int64` contract (`logical_plan.rs:409`). A
result schema built from one path vs the other will disagree on dtype.
**Suggested fix:** Return `Int64Array` (and fix F-3 validity at the same time). Verify which path the
executor actually calls and retire the divergent one.

### F-9 — `string_col.rs` `ExtendedDeviceCol` is a parallel, unmerged copy of `DeviceCol` (dead-ish / stub)
**File:** `string_col.rs:5-8, 32-61`
**Severity:** LOW (architecture / dead code risk)
**Description:** The module is explicitly a standalone half ("The orchestrator merges
`ExtendedDeviceCol` into `DeviceCol` once both halves are wired"). All GPU round-trip tests are
`#[ignore]`. This is fine as a staging module but is a stub awaiting integration; verify it is actually
referenced by the orchestrator and not orphaned. `upload_utf8` accepts only `StringArray` (i32 offsets),
not `LargeStringArray` — documented (`:340-363`) but a silent capability gap.
**Suggested fix:** Track the merge as a real TODO with an owner; add a `LargeUtf8` rejection error at
the entry point rather than relying on the type system to reject it downstream.

### F-10 — `to_uppercase`/`to_lowercase` are locale-independent; SQL `UPPER`/`LOWER` collations differ
**File:** `string_ops.rs:109-118`; `string_project.rs:81-86`
**Severity:** LOW
**Description:** Rust's `to_uppercase`/`to_lowercase` do full Unicode default case folding
(`'ß'→"SS"`, Greek final sigma handling per `upper_unicode_lowercase_collapses` test). This is
reasonable, but Turkish dotted/dotless I and other locale rules are not honoured. Acceptable for v1;
worth a documented note that case folding is locale-invariant (Unicode default).
**Suggested fix:** Document the locale-invariance; no code change required for v1.

### F-11 — GPU ASCII case-fold gate is per-dictionary all-or-nothing (perf, not correctness)
**File:** `string_project.rs:340-348` (`dict_is_ascii`) + `:44-49`
**Severity:** LOW (perf)
**Description:** A single non-ASCII byte anywhere in the dictionary forces the **entire** column to the
host path. For mostly-ASCII dictionaries with one stray Unicode entry this loses the GPU path entirely.
Correctness is fine (the gate is conservative). Note the ASCII byte-fold is correctly length-preserving
and the comment about `'ß'→"SS"` is the right reason for the gate.
**Suggested fix:** Optional: fall back per-entry rather than per-dictionary, or precompute the ASCII
flag at dictionary-build time so the scan isn't repeated on every projection.

### F-12 — `schema_convert.rs`: `arrow_dtype_to_plan` accepts only `Dictionary(_, Utf8)`; other dict value types silently error; `LargeUtf8`/`Date64` unsupported
**File:** `schema_convert.rs:106-131, 138-157`
**Severity:** LOW
**Description:** The mapper rejects `LargeUtf8`, `Date64`, `Dictionary(_, non-Utf8)`, `Int8/16`,
`UInt*`, `Float16` etc. with a generic "unsupported Arrow dtype". That is the intended loud-failure
guard and is correct, but the asymmetry is worth noting: `plan_dtype_to_arrow` maps `Utf8 → Utf8`
(never produces a Dictionary), while ingest accepts `Dictionary(_, Utf8)`. Round-tripping an Arrow
dictionary column through plan→arrow loses the dictionary encoding (becomes plain Utf8). Functionally
OK; a perf/representation note.
**Suggested fix:** Document the intentional dictionary→Utf8 collapse; consider mapping the key width
explicitly if dictionary output is ever needed.

### F-13 — `validity_audit.rs` is documentation-as-code; the matrix can silently drift from reality
**File:** `validity_audit.rs:44-80` (the propagation matrix)
**Severity:** LOW
**Description:** The propagation matrix is a hand-maintained comment table describing a dozen other
executors. There is no test asserting the described behaviour matches the actual executors, so it can
rot. `packed_validity_for` itself is correct and well-tested (Arrow LE bit order verified). The risk is
purely the doc table drifting.
**Suggested fix:** Where feasible, add a small integration test per row of the matrix (e.g. assert
`COUNT(col)` over a null-containing column equals the non-null count) so the documented invariant is
enforced, not just asserted in prose.

### F-14 — `like.rs` classifier: only `Shape::Exact` could pick up the dictionary-eq fast path but doesn't
**File:** `like.rs:33-38, 178-182`
**Severity:** LOW (perf)
**Description:** `LIKE 'foo'` with no wildcards becomes `Shape::Exact` and runs a per-row string
compare via `host_like`, even when the column is dictionary-encoded and a single `index_of` +
i32-compare (`input_eq_literal`) would be O(dict)+O(n) instead of O(n·|foo|). The doc acknowledges the
planner *could* rewrite this but this module doesn't. Correctness is fine.
**Suggested fix:** In the planner, rewrite wildcard-free `LIKE` to `=` so it hits the dictionary fast
path; or have the executor detect a dictionary column + `Shape::Exact` and delegate to
`input_eq_literal`.

### Verified-correct (no action)
- `like.rs` generic matcher is the non-recursive two-pointer LIKE algorithm; the V-6 ReDoS regression
  test (`:686`) confirms linear behaviour. ESCAPE handling, `%`/`_` semantics, `%%` collapse, and 3VL
  NULL propagation in `host_like` are all correct and well-covered.
- `string_ops_extended.rs` TRIM (custom char-set vs substring distinction, Unicode whitespace),
  CONCAT, CONCAT_WS (NULL-skip vs NULL-propagate, empty-string-vs-NULL), and the GPU ASCII-whitespace
  trim byte mirror are correct and thoroughly tested.
- `dedup_transformed`/`remap_and_upload` index math and i32::MAX overflow guards are correct; all
  dictionary index accesses are bounds-checked and reject negative/out-of-range keys rather than
  reading OOB.
- `subquery_resolve.rs` scalar 0/1/>1-row handling, distinct IN-set dedup, and the recursive plan
  walker (covers all `LogicalPlan`/`Expr` arms, including nested `InSubquery` probe) are correct.

---

## 2. TEST ADEQUACY

**Estimated coverage of pure host logic: ~85%.** The pure helpers are exhaustively tested; the gaps are
(a) the entire GPU half (all `#[ignore]`), and (b) the SQL-semantic edge cases the code gets *wrong*
(those are "tested" only in the sense of locking in the current—incorrect—behaviour).

**Well-covered:** LIKE shapes + ESCAPE + ReDoS + Unicode `_`; TRIM all variants; CONCAT/CONCAT_WS NULL
matrix; SUBSTRING byte-boundary cases; dictionary dedup/remap; subquery scalar/IN folding; schema
round-trips (implicit); `packed_validity_for` bit order.

**Untested / missing cases to add:**
1. **`LENGTH` of multibyte string** asserting the *intended* (character) result — currently only byte
   length is asserted. Add once F-2/F-3 are resolved.
2. **`LENGTH(NULL)`** asserting NULL output (currently no test; behaviour is wrong — F-3).
3. **`SUBSTRING` multibyte where start is mid-codepoint**, asserting no characters left of `start`
   leak in (F-1). Add `日本語` cases (3-byte chars) and emoji (4-byte).
4. **`SUBSTRING` very long string** (>i32::MAX guard is only hit theoretically; add a moderately long
   string round-trip).
5. **`input_eq_literal` with a NULL row** asserting 3VL (currently no NULL-row test — F-5).
6. **`NOT IN (subquery)` with a NULL in the set** asserting no rows pass (F-6) — only the predicate
   *structure* is tested, never the NULL-set negated case end-to-end.
7. **`build_in_predicate` with floats / NaN** (the code comments about NaN dedup but no test exercises
   a `Float64`/NaN set).
8. **Dictionary registry collision**: two scans (UNION/SetOp) with same-named Utf8 columns and
   different dictionaries, asserting either correct per-table folding or a safe bail-out (F-7).
9. **`schema_convert` unsupported dtypes**: `LargeUtf8`, `Date64`, `Dictionary(_, Int32)`, `UInt32`
   asserting the loud error message (no negative-path tests today).
10. **`string_like.rs` decompose with multibyte leading/trailing `%`** literal (e.g. `%café%`) —
    confirm byte-slice safety; current tests are ASCII-only for the decomposer.
11. **`host_like` with empty pattern vs empty string vs NULL** triad (partially covered; make explicit).
12. **Very long string in LIKE Contains** (perf + correctness on the naive mirror scan).

---

## 3. MISSING FEATURES / DIRECTIONS

**String functions not implemented (vs typical SQL surface):**
- `CHAR_LENGTH` / `OCTET_LENGTH` distinction (currently only byte `LENGTH`).
- `POSITION` / `STRPOS` / `INSTR`, `REPLACE`, `OVERLAY`, `LEFT`/`RIGHT`, `LPAD`/`RPAD`, `REPEAT`,
  `REVERSE`, `SPLIT_PART`, `INITCAP`, `ASCII`/`CHR`.
- `ILIKE` (case-insensitive LIKE) — referenced in `like.rs:290-293` as "a separate code path" but not
  present in scope.
- Regex (`~`, `SIMILAR TO`, `REGEXP_*`) — explicitly out of scope.
- Variadic `CONCAT` (currently binary; doc suggests routing through `CONCAT_WS('')`).
- `SUBSTRING(col, start)` two-arg form not a real entry point (must pass `i32::MAX`).
- Collation / `LOWER`/`UPPER` locale variants.

**Architecture directions:**
- **Wire validity through `LENGTH`** now that `validity_audit::packed_validity_for` exists — removes
  the F-3 caveat for the whole string-op family.
- **Validate the `string_like.rs` device kernel on real hardware** and add a GPU CI lane; until then
  keep it opt-in (F-4).
- **Merge `ExtendedDeviceCol` into `DeviceCol`** (F-9) so the Bool/Utf8 device paths aren't a parallel
  island.
- **Qualify dictionary lookups by `(table, column)`** before enabling multi-table plans (F-7).
- Consider a **device prefix-scan** for `string_project.rs` (currently a host exclusive-scan stand-in,
  `:216-241`, already flagged in-code) once the two-pass path is hardware-validated.

---

## Priority summary

| ID  | Sev   | File | One-line |
|-----|-------|------|----------|
| F-1 | HIGH  | string_ops_extended.rs:126 | SUBSTRING leaks bytes before `start` on multibyte input |
| F-2 | HIGH  | string_ops*.rs, string_length.rs | LENGTH/SUBSTRING byte-based, not character-based; "UTF-8" doc is misleading |
| F-3 | HIGH  | string_ops.rs:174 | LENGTH(NULL)=0, must be NULL; no validity bitmap |
| F-8 | MED   | string_ops.rs:174 | LENGTH returns Int32Array, contract is Int64 |
| F-4 | MED   | string_like.rs:7 | GPU LIKE kernel never run on hardware; keep opt-in |
| F-5 | MED   | string_ops.rs:212 | input_eq_literal: NULL=lit yields false, not NULL (3VL) |
| F-6 | MED   | subquery_resolve.rs:205 | NOT IN with NULL set passes rows it shouldn't |
| F-7 | MED   | dict_registry.rs:194 | cross-table same-name column collision folds wrong dictionary (hits UNION today) |
| F-9 | LOW   | string_col.rs | ExtendedDeviceCol unmerged stub; no LargeUtf8 |
| F-10| LOW   | string_ops.rs:109 | case fold is locale-invariant (doc only) |
| F-11| LOW   | string_project.rs:340 | ASCII gate is whole-dictionary (perf) |
| F-12| LOW   | schema_convert.rs:106 | dictionary→Utf8 collapse; many dtypes unsupported (intended) |
| F-13| LOW   | validity_audit.rs:44 | propagation matrix is unenforced prose |
| F-14| LOW   | like.rs:178 | wildcard-free LIKE doesn't use dictionary-eq fast path (perf) |
