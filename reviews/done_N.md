# Agent N2 — test-coverage remediation report

Branch `dev`, root `C:\Projects\bolt`. A prior tests agent was lost to a socket
error and left no files; this run reconstructs and lands the high-value tests
called out in `reviews/tests.md`.

All work is confined to `C:\Projects\bolt\tests\`. No `src/`, `.github/`,
`Cargo.toml`, or `reviews/` (other than this report) was edited. `cargo` was
NOT run, per instructions — the sources are written to compile under
`--no-default-features --features cuda-stub` (every GPU-requiring test is
`#[ignore]`'d, which still compiles).

## Files created

1. **`tests/semantics_e2e.rs`** — value-asserting end-to-end tests
   (items 1–7 of the brief). Drives the full `Engine::sql` pipeline and checks
   concrete expected values.
2. **`tests/diff_duckdb_semantics.rs`** — a `diff_duckdb`-style DuckDB oracle
   harness for items 1–4 (item 8 of the brief). A self-contained sibling of
   `tests/diff_duckdb.rs` (re-declares the `Cell` / decoder / driver machinery
   locally, since each `tests/*.rs` is its own crate), with new fixtures for the
   semantics gaps.

Existing files were NOT modified (no WRONG-behavior lock-in test was found in
`tests/` to invert — the known-wrong SUBSTRING assertion the audit flags lives
in `src/exec/string_ops_extended.rs`, outside the allowed edit set; the sibling
fix has landed there and the new tests pin the corrected char-indexed
semantics from the public surface).

## Coverage delivered (by brief item)

| # | Topic | Where | Gating |
|---|-------|-------|--------|
| 1 | `LENGTH`=chars, `OCTET_LENGTH`=bytes, multibyte `SUBSTRING` no left-leak, `LENGTH(NULL)`=NULL | `semantics_e2e.rs` (5 tests) + `diff_duckdb_semantics.rs` (2 tests) | `gpu:string` |
| 2 | `NOT IN (subquery)` with NULL → 0 rows; without NULL → complement; NULL probe; `IN` control | `semantics_e2e.rs` (4 tests) + `diff_duckdb_semantics.rs` (2 tests) | `gpu:e2e` |
| 3 | Two-key `COUNT(col)` with NULLs counts only non-null | `semantics_e2e.rs` (1) + `diff_duckdb_semantics.rs` (1) | `gpu:tier1` |
| 4 | Grouped float `MIN`/`MAX` incl NaN == scalar (NaN-as-largest); all-NaN group | `semantics_e2e.rs` (2) + `diff_duckdb_semantics.rs` (1) | `gpu:tier1` |
| 5 | UTF-8 multibyte **multi-key** sort (`ORDER BY s1 ASC, s2 DESC`) — existing sort tests are ASCII/int-only | `semantics_e2e.rs` (1) | `gpu:sort` |
| 6 | All-NULL group key → one group; all-NULL inner-join key → 0 rows; mixed-NULL join keys match only non-null | `semantics_e2e.rs` (3) | `gpu:tier1`, `gpu:join` |
| 7 | Dict-registry collision via `UNION` (same-named `region`, disjoint dicts) stays correct (F-7 poison fix) + identical-dict control | `semantics_e2e.rs` (2) | `gpu:string` |
| 8 | DuckDB oracle file for items 1–4 | `diff_duckdb_semantics.rs` (6 tests) | per-case, matches the `diff_duckdb.rs` harness |

## Host vs GPU-gated breakdown

- **GPU-gated (`#[ignore]`):** ALL 24 new test functions.
  - `tests/semantics_e2e.rs`: 18 tests — `gpu:string` (7), `gpu:e2e` (4),
    `gpu:tier1` (4), `gpu:sort` (1), `gpu:join` (2).
  - `tests/diff_duckdb_semantics.rs`: 6 tests — `gpu:string` (2), `gpu:e2e` (2),
    `gpu:tier1` (2).
- **Host-only (ungated):** 0.

  Rationale: every scenario in this brief needs a registered table, and
  `Engine::register_table` / `Engine::new` open a CUDA context — there is no
  host-foldable path for these queries (the string scalar fns route through the
  GPU scan + string paths; LENGTH/OCTET_LENGTH/SUBSTRING are not constant-
  foldable without a device). This matches the existing suite, where every
  table-driving test (`diff_duckdb.rs`, `e2e_tests.rs`, `string_fns_sql_test.rs`,
  `sort_e2e.rs`) is GPU-gated for the same reason. The pure host-side coverage
  (parser/plan/lower) is already dense in `tests/*_test.rs` and `src/` unit
  tests, so no additional ungated tests were warranted here.

## Conventions matched

- `#[ignore = "<bucket>"]` labels drawn from the standard set documented in
  `tests/common/mod.rs` (`gpu:string`, `gpu:e2e`, `gpu:tier1`, `gpu:sort`,
  `gpu:join`).
- `mod common;` include + `common::REL_TOL` reuse in the oracle file.
- The oracle harness mirrors `tests/diff_duckdb.rs` exactly: null-aware
  `Cell::approx_eq`, `close_enough` with `REL_TOL` and NaN==NaN, row-order
  canonicalisation, both-sides dump on mismatch, DuckDB `appender` load + Bolt
  `register_table` from the same Arrow batch.
- Public API only: `Engine::new`, `register_table`, `sql`, `QueryHandle`
  `record_batch` / `num_rows`. No internal `__test_only_*` surface needed.

## Notes / assumptions

- `OCTET_LENGTH(s)` is exercised at the SQL surface; the engine exposes the
  byte-length table builder (`src/exec/string_length.rs::build_octet_length_table`,
  `src/exec/string_ops.rs::octet_lengths_table_pure`). If `OCTET_LENGTH` is not
  yet wired into the SQL frontend on a given checkout, the two
  `octet_length`/`diff_length_and_octet_length` tests will surface a clean
  `Plan` error at `engine.sql(...)` on the GPU host rather than a false pass —
  the `.expect(...)` makes that loud. The compile is unaffected (it is a SQL
  string).
- The multibyte multi-key sort (item 5) self-oracles against Rust's stable sort
  on the same byte-wise UTF-8 strings (`s1.cmp` ASC, `s2.cmp` DESC) — that is
  the byte-lexicographic ordering the engine's Utf8 sort path targets, and the
  test additionally asserts the local monotonic invariant as defence in depth.
- NOT-IN-with-NULL (item 2) targets the F-6 fix
  (`src/exec/subquery_resolve.rs`); dict-collision-via-UNION (item 7) targets
  the F-7 fix (`src/exec/dict_registry.rs`). Both sibling fixes are on `dev`.
