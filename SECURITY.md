# Security Policy

Thank you for helping keep Craton Bolt and its users safe.

## Supported versions

Craton Bolt is pre-1.0. While the API is unstable, only the **latest minor
release line** receives security fixes — older minor lines are not
backported. This is standard practice for pre-1.0 crates and keeps the
maintenance surface narrow while the IR and public API are still moving.

The current supported line is `0.7.x`. Older minor lines (`0.6.x` and
earlier) are no longer supported; users should upgrade to `0.7.x` (note
that `0.2.0` and `0.4.0` were skipped — see `CHANGELOG.md`).

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| < 0.7   | :x:                |

## Reporting a vulnerability

**Please do not file public GitHub issues for security vulnerabilities.**
Private disclosure is strongly preferred so that we can ship a fix before
the issue is widely known.

Report vulnerabilities by email to:

> **security@craton.com.ar**

If you would prefer encrypted email, request our PGP key in your first
message and we will provide it before you send details.

Please include, where possible:

- A description of the issue and its impact.
- Steps to reproduce (a minimal Rust snippet or SQL query is ideal).
- Affected version(s) / commit SHA.
- Your assessment of severity, and whether the issue is already public.

You may also use [GitHub's private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
if you prefer.

## Our process

- We will acknowledge your report within **5 business days**.
- We aim to provide an initial assessment within **10 business days**.
- We follow a **90-day coordinated disclosure** timeline by default: the
  reporter and maintainers agree on a public disclosure date no later than
  90 days after the initial report. Extensions are possible for complex
  fixes by mutual agreement.
- Once a fix ships, we will publish a GitHub Security Advisory and credit
  the reporter (unless anonymity is requested).

## Scope

In scope: anything shipped from this repository — the Craton Bolt Rust crate,
its build scripts, CI workflows, and documentation.

Out of scope: vulnerabilities in upstream dependencies (please report
those to the relevant project), and issues that require an attacker with
existing root or physical access to the host.

## Threat model

Craton Bolt is a pre-1.0 GPU SQL engine that **JIT-compiles each query into
a fresh NVIDIA PTX kernel at runtime** and can optionally load PTX from an
on-disk cache. That runtime-codegen design is the single most security-
relevant property of the engine, so this section is explicit about what is
defended, what is best-effort, and what is out of scope. It is written to be
honest rather than reassuring: where a control is best-effort or a pre-1.0
caveat, it says so. See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for the
broader (non-security) list of gaps and required preconditions.

### Trust boundaries and assets

- **The SQL query string is untrusted input.** The engine treats the SQL
  text handed to `Engine::sql` as adversary-controlled and parses/lowers it
  defensively (see "Runtime JIT" and "Denial of service" below). It does
  **not** assume the SQL came from a trusted author.
- **The host process and the embedding application are trusted.** Craton
  Bolt is a library embedded in a host process; the operator who builds and
  runs that process, and any code in it, is trusted. There is no privilege
  boundary *inside* the process.
- **Assets to protect:** the correctness and confidentiality of query
  results and of the in-memory tables registered with the engine (the Arrow
  `RecordBatch`es and the device-side columns derived from them).
- **Trust boundaries crossed at runtime:** (1) the host → GPU boundary —
  device memory and the kernels launched on it; and (2) the PTX disk cache
  directory, which is *persistent, off-process state* that is read back and
  **launched as code**. Both are treated as boundaries below.

### Attack surface: runtime JIT (SQL → PTX)

The engine lowers SQL to a physical-plan IR and then emits PTX text from that
IR (`src/plan/sql_frontend.rs` → `src/plan/physical_plan.rs` →
`src/jit/ptx_gen.rs`). The key property is that **codegen is opcode/IR-based,
not string-templated from SQL source**, so there is no PTX source-injection
surface analogous to SQL injection:

- **Numeric and temporal literals are emitted as hex bit-patterns.** Integer,
  float, `Date32`, `Timestamp`, and 128-bit constants are written into the
  kernel as fixed-width hex immediates (`mov.s32 r, 0x{:08X}` /
  `0d{:016X}`, etc. — see the `SECURITY:` comments in `emit_const` /
  `Const128` in `src/jit/ptx_gen.rs`). An attacker-controlled literal value
  therefore cannot break out of the immediate and inject instructions — it
  can only choose the *value* of a constant, restricted to the
  charset `[0-9A-F]`.
- **String literals are never interpolated into kernel source.** `emit_const`
  explicitly rejects `Literal::Utf8` (and `Literal::Null`). String predicates
  are handled out-of-band: the dictionary registry rewrites `col = 'X'` into
  integer-index equality (`__idx_col = <idx>`) before lowering, and string
  data is uploaded as device buffers / dictionary indices rather than baked
  into PTX text (see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
  "Dictionary encoding for Utf8"). So even string content reaches the device
  as data, not code.
- **Parse/lowering is depth-bounded.** AST and `LogicalPlan` walking is capped
  at `MAX_RECURSION_DEPTH = 256` (`src/plan/sql_frontend.rs`); a pathologically
  nested expression surfaces a clean `BoltError::Sql(...)` instead of
  overflowing the host thread stack. Unsupported / malformed constructs are
  rejected as typed `BoltError::Sql` / `BoltError::Plan` errors rather than
  crashing (see "Rejected SQL constructs" in
  [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)).
- **Parse-time input-size caps.** Before any AST is built, `guard_sql_size`
  (`src/plan/sql_frontend.rs`, the single chokepoint at the top of
  `parse_uncached`) rejects over-cap input with a typed `BoltError::Sql` —
  no panic. Two caps apply: a byte-length cap (`MAX_SQL_BYTES_DEFAULT`,
  **1 MiB**, overridable via `CRATON_MAX_SQL_BYTES`) checked with no
  allocation, and a token-count cap (`MAX_SQL_TOKENS_DEFAULT`, **100k**,
  overridable via `CRATON_MAX_SQL_TOKENS`) checked via a cheap linear
  tokenizer scan that allocates only a flat token vector — never the
  recursive AST. The token cap matters because byte length alone does not
  bound AST node count (a short-per-token `a+a+a+…` blob is dense in nodes).
  Together these bound the parser/AST DoS surface **before** the parser runs,
  including the flat-operator-chain and deeply-nested-`IN`-subquery inputs
  whose over-deep AST previously crashed the process during recursive `Drop`
  — the depth guard (`MAX_RECURSION_DEPTH = 256`) fires only during lowering,
  too late to prevent that `Drop`. Residual, bounded surface: the caps are
  defaults and tunable, and an extremely large but still-under-cap query can
  still consume planning time (see "Denial of service").

### PTX disk cache

The disk-backed PTX cache (`src/jit/disk_cache.rs`) is **opt-in and disabled
by default**. It is enabled only by setting `BOLT_PTX_CACHE_DIR` (or the
builder's `persistent_cache(path)` knob). When disabled, all lookups/stores
are no-ops and PTX is always re-derived from source.

Because a cache hit returns bytes that are then handed to
`CudaModule::from_ptx` and **launched as a kernel**, the cache directory is a
genuine trust boundary. The enforced controls are:

- **The cache directory's ownership/permissions are the real security
  boundary.** On Unix the root is tightened to `0o700` (owner-only) on
  `open`; on Windows the per-user `%LOCALAPPDATA%` default already lives under
  the user's profile ACL, with a best-effort `icacls` tightening for explicit
  shared-dir overrides. **A locally-writable cache directory is the risk:** a
  different local user who can write the directory can plant or tamper with
  `<key>.ptx` files that this process would load and launch.
- **Fail-closed directory trust.** A handle that fails its directory trust
  check becomes a disk no-op (lookups miss, stores write nothing); the engine
  keeps running on in-process codegen, so a disabled disk layer is always
  safe.
- **Path-traversal hardening.** Cache keys are validated against a strict
  filename-safe charset (`^[0-9A-Za-z._-]+$`, no separators, no `..`, no NUL,
  no drive/ADS `:`) at the moment they become a filename (`valid_key` /
  `entry_path`); an unsafe key degrades to a cache miss rather than escaping
  the root.
- **Codegen-salted keys (freshness, not security).** The on-disk key folds in
  a codegen-version salt (`CODEGEN_VERSION` + crate version + arch/ISA token
  + optional build-time fingerprint). This guards against serving *stale*
  PTX written by a structurally different binary; it is a correctness guard,
  not an authenticity guard.
- **Integrity header is anti-corruption, not anti-tamper.** Each file carries
  a `#bolt-ptx-cache v1 <digest>` header and the body is verified on read.
  This is a non-cryptographic `DefaultHasher` digest: it catches accidental
  corruption / partial writes and trips naive tampering, but it is **not a
  MAC**. An attacker who can write the cache directory can recompute the
  digest, so the header does **not** substitute for the directory permissions
  above — those (the `0o700` / per-user ACL) are the load-bearing control.

If you cannot guarantee a single-owner cache directory, leave the disk cache
disabled (the default).

### Denial of service

- **Adversarial-shape SQL** is bounded at the parse entry by `guard_sql_size`
  (byte cap `CRATON_MAX_SQL_BYTES`, default 1 MiB; token cap
  `CRATON_MAX_SQL_TOKENS`, default 100k — see "runtime JIT" above), which
  rejects over-cap input with a typed `BoltError::Sql` before any AST is
  built, and by the `MAX_RECURSION_DEPTH = 256` depth cap during lowering,
  which converts surviving stack-exhausting inputs into a typed parse error.
  The pre-parse caps are what bound the recursive-`Drop` crash from
  flat-operator-chain / deeply-nested-subquery inputs.
- **GPU capacity overflows fall back to the host.** When a GPU path cannot
  size a query (e.g. a hash-join match-count overshoot), the engine returns
  the typed `BoltError::GpuCapacity` marker and callers route to a host-side
  fallback rather than faulting (see `src/error.rs` and the join/sort
  fallbacks in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)). This bounds
  device-memory blow-ups but shifts the cost to the host.
- **Residual DoS surface (documented honestly).** The parse-time input-size
  caps (above) bound the parser/AST surface, but there is still no per-query
  wall-clock/row-count budget and no multi-query fair scheduling — a single
  expensive but well-formed query (including a large but under-cap one) can
  monopolise the
  one CUDA context per process (the engine is single-context and serialises
  queries; see [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), "Concurrency").
  Host-fallback paths (sort, some joins, set ops, `DISTINCT`) can also be
  memory-intensive. Rate-limiting, query timeouts, and input-size limits are
  the **caller's** responsibility.

### Memory safety

- **Rust + RAII GPU buffer ownership is the memory-safety control.** GPU
  allocations are owned `GpuVec<T>` handles freed on `Drop`, borrowed as
  `GpuView` / `GpuViewMut`; use-after-free, double-free, and shared/mutable
  aliasing across kernel boundaries are rejected at compile time (the
  "CUDA-Oxide" model — see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md),
  "Memory safety", and the `compile_fail` doctests in
  `tests/memory_tests.rs`).
- **`unsafe` is confined to the CUDA FFI / codegen layer** (`src/cuda/`,
  kernel-launch parameter assembly), where the raw driver ABI forces it;
  those blocks carry `// SAFETY:` justifications and the launch synchronises
  before borrowed device memory can be freed.
- **GPU correctness is validated out-of-band, not in CI.** Per project
  convention, CI exercises **0 GPU code paths** (`cuda-stub` only); end-to-end
  GPU execution is validated on maintainer hardware via `#[ignore]`-gated
  tests (see the "CI runs no GPU code" callout in
  [`README.md`](README.md) and [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)).
  A green CI run attests host logic, planning, and codegen *shape* — not
  device behavior. Treat device-side memory-safety claims as best-effort and
  pre-1.0.

### Security properties / boundaries (summary)

| Property | Status |
|---|---|
| No PTX source injection from SQL literals | Enforced — hex-immediate / device-buffer emission, no string templating |
| Parser stack-exhaustion DoS | Mitigated — `MAX_RECURSION_DEPTH = 256` depth cap (lowering) + parse-time caps below |
| Input byte/token-size DoS | Mitigated — parse-time caps enforced before the parser runs (`CRATON_MAX_SQL_BYTES` default 1 MiB, `CRATON_MAX_SQL_TOKENS` default 100k); tunable defaults, residual under-cap planning cost |
| Query-time / row-count budget | **Not** enforced — caller's responsibility |
| PTX cache integrity vs. accidental corruption | Enforced — integrity header verified on read |
| PTX cache integrity vs. a writer of the cache dir | **Boundary is the dir permissions** (`0o700` / per-user ACL), not the header (no MAC) |
| Cache path traversal | Enforced — strict key charset, fail-to-miss |
| Host memory safety | Enforced at compile time (Rust + borrow-checked GPU handles) |
| Device-side execution correctness | Best-effort, validated out-of-band (not in CI) |

### Out of scope / non-goals

- **No multi-tenant isolation.** The engine executes whatever SQL it is
  given against whatever tables are registered; there is no per-tenant or
  per-query sandbox, and one expensive query can monopolise the process's
  single CUDA context.
- **No row-level / column-level access control.** There is no notion of
  users, roles, grants, or row filtering. Authorization of *which* data a
  caller may query must happen **before** SQL reaches the engine.
- **No protection against a local actor who already controls the cache dir
  or the host process.** A user who can write `BOLT_PTX_CACHE_DIR` (when the
  permission boundary is broken) or who controls the embedding process is
  inside the trust boundary.
- **No side-channel / timing guarantees.** Query latency and GPU memory
  behavior are data-dependent; the engine makes no constant-time or
  cache-/timing-side-channel claims.
- **No protection against a malicious GPU driver, firmware, or physical
  access**, consistent with the "Scope" section above.

### Assumptions

- A **trusted operator** builds and runs the embedding process and chooses
  the cache directory (and keeps it single-owner if the disk cache is
  enabled).
- The engine is used in a **single-tenant embedding** (one logical owner of
  the process and its registered data), with queries serialised through one
  long-lived `Engine` per process (see
  [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md), "Concurrency").
- The **caller is responsible for authentication and authorization** of the
  data and the SQL: deciding who may run queries and what data they may see
  happens upstream of `Engine::sql`. The engine's defensive posture toward
  the SQL *string* (no source injection, depth-bounded parsing) does **not**
  imply it is safe to expose raw SQL to untrusted end users without your own
  authz, resource limits, and input-size limits in front of it.
