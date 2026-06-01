# JIT Compiler Core Review — Bolt

Scope: `src/jit/jit_compiler.rs`, `src/jit/ptx_gen.rs`, `src/jit/disk_cache.rs`,
`src/jit/mod.rs`. Pipeline: PTX text → `cuModuleLoadDataEx` (driver PTXAS), the
expression→PTX codegen, the process-wide module cache, and the opt-in on-disk
PTX cache.

Verdict: the code is unusually well-documented and the obvious hazards
(path traversal, hash collision, concurrent compile, codegen-injection,
integrity header) have already been hardened. The remaining findings are
mostly *correctness gaps* (arch in cache key, integer DIV/0, NULL-propagation
modelling) and *test gaps* (zero disk-cache concurrency tests, no integer DIV/0
test, no GPU-level codegen execution test).

---

## 1. CODE REVIEW

### CRITICAL

**C-A. Disk cache key does not hash the PTX *target architecture* (`sm_70`).**
`ptx_gen.rs:15` hardcodes `.target sm_70`; the disk key
(`disk_cache.rs:236 disk_key` + `engine.rs:151 ModuleCacheKey::new`) is composed
from `codegen_salt()` (`CODEGEN_VERSION` + crate version) and
`format!("{:?}", spec)` — **neither carries the SM target, PTX ISA version, nor
the CUDA driver version**. Today this is latent because the target is a fixed
constant, but the moment `PTX_TARGET`/`PTX_VERSION` becomes runtime/GPU-derived
(an obvious near-term feature — see §3), a disk cache populated on one GPU arch
will be served verbatim to a process on a different arch. The module docs at
`jit_compiler.rs:18-23` lean entirely on "codegen is deterministic for a given
(PhysicalPlan, kernel_name)" — that invariant silently *excludes* the GPU/driver
identity. The salt comment (`disk_cache.rs:104-109`) even lists `PTX_TARGET` as
a thing requiring a manual `CODEGEN_VERSION` bump, which is the same
single-point-of-failure the file elsewhere calls out as the JIT-M1 hazard.
Recommendation: fold `cuDeviceGetAttribute(COMPUTE_CAPABILITY)` + driver version
into `codegen_salt()` now, before the target string is ever parameterized.
Severity Critical because mis-routed PTX is launched as a GPU kernel
(silent wrong results / illegal-instruction), not a soft failure.

**C-B. Integer division by zero / `INT_MIN / -1` emits trapping PTX with no
guard.** `arith_mnemonic` (`ptx_gen.rs:1534-1535`) lowers integer `Div` to
`div.s32` / `div.s64` directly. PTX `div.s32` by zero is undefined behaviour at
the SASS level (typically returns garbage; `INT_MIN / -1` overflows). There is
no zero-divisor predicate, no NULL-on-divide-by-zero, and no host-side
pre-check visible in this layer. SQL semantics for `x / 0` is normally an error
or NULL; here it is silent UB on the GPU. Float div uses `div.rn.f32/f64`
(IEEE, fine — yields inf/nan). Recommendation: either reject integer `Div` from
GPU lowering (host fallback) or emit a `setp.eq` zero-guard that produces NULL.

### HIGH

**H-A. NULL propagation is a coarse AND-of-all-inputs, documented as
approximate, and the precise per-output dataflow path appears unused.**
`compile` builds `combined_valid` as the AND of *every* flagged input's validity
byte (`ptx_gen.rs:356-389`) and writes that same combined bit to *every* flagged
output (`ptx_gen.rs:442-456`). The code comment (`ptx_gen.rs:344-350`) admits
this is conservative: "every output is marked valid only if EVERY input row is
valid." A precise per-output backward sweep exists
(`output_input_dependencies` / `walk_store_deps`, `ptx_gen.rs:3520,3668`) but
`compile` does not call it — it AND-folds all inputs unconditionally. For a
multi-output kernel where output0 depends only on input A and output1 only on
input B, output0 is incorrectly marked NULL whenever B is NULL. This is a
correctness bug for any multi-output validity-carrying kernel; it is masked
today only because, per the comment, "every input feeds every output" in the
current single-output `SUM(price*tax)` shape. Also note `Op::Select`
NULL-semantics: `walk_store_deps` folds `cond`+`then`+`else` validity together
(`ptx_gen.rs:3553-3567`), so `CASE WHEN c THEN a ELSE b END` is marked NULL if
*either* branch is NULL even when the taken branch is non-NULL — not SQL CASE
semantics. (Currently only reachable via the unused precise path.)

**H-B. `cuModuleLoadDataEx` error/info log buffers can be silently truncated and
the size-slot cast is fragile.** `load_uncached` (`jit_compiler.rs:557-581`)
uses a fixed 4096-byte log buffer and passes the size as a pointer-bit-pattern
(`JIT_LOG_BUF_SIZE as usize as *mut c_void`, line 574-575). A long PTXAS error
(multi-error module) is truncated; `decode_log` (`jit_compiler.rs:752`) stops at
the first NUL, so a buffer the driver fully fills *without* a NUL terminator is
read to `buf.len()` — acceptable, but a >4 KB diagnostic loses its tail. The
`*_SIZE_BYTES` option officially takes the value in the low 32 bits of the slot;
on a 64-bit host the cast is correct but the comment (`jit_compiler.rs:570-573`)
itself flags the contract as "require the value to fit in u32" — fine for 4096,
but brittle if anyone enlarges the buffer past `u32::MAX` (won't happen) or ports
to a 32-bit target. Low real-world risk; the issue is diagnostic loss on large
PTXAS failures, which undercuts the whole "surface PTXAS line numbers" goal.

### MEDIUM

**M-A. In-process module cache cap is frozen for process lifetime; a single
oversized query permanently can't be hot-fixed, and there is a stale TODO.**
`ptx_cache_cap()` (`jit_compiler.rs:141-149`) memoizes via `OnceLock` and the
TODO at line 143-144 ("re-read on each insert") is unimplemented. Default 256
modules. For a workload with >256 distinct hot kernels this thrashes (LRU
evict → recompile → ~10 ms PTXAS each). Not a correctness bug, but a cache-hit
cliff. Document or make the cap dynamic.

**M-B. Disk cache has no eviction / size bound / TTL — unbounded growth.**
`DiskPtxCache::store` (`disk_cache.rs:589`) writes `<key>.ptx` forever. Every
`CODEGEN_VERSION` bump, crate-version bump, and every distinct query plan
creates a new permanent file; stale-salt files are never reaped (they just stop
being looked up). On a long-lived serverless image or dev box the cache dir
grows without limit. No LRU, no max-bytes, no mtime sweep. Recommendation: a
size/age-bounded sweep on `open`, or a documented external cleanup contract.

**M-C. Disk store is not atomic against a concurrent reader on Windows in the
"replace existing" case across volumes, and TOCTOU on `lookup`→launch.**
`store` (`disk_cache.rs:612`) relies on `fs::rename` atomicity — correct on a
single volume (POSIX `rename`, Windows `MoveFileEx REPLACE_EXISTING`). The temp
file is created in the same dir (`disk_cache.rs:605-607`) so same-volume holds.
However, `lookup` reads + verifies the integrity digest
(`disk_cache.rs:544-557`) and then the caller hands the bytes to
`CudaModule::from_ptx` — between the digest check and the launch there is no
re-validation, so a local attacker who can write the dir (the exact threat the
`0o700`/`icacls` hardening targets, `disk_cache.rs:462-482,658-683`) on a
*shared* `BOLT_PTX_CACHE_DIR` can race a swap. The code is honest that
`body_digest` is "NOT a cryptographic MAC" and the dir perms are the real
boundary (`disk_cache.rs:785-792`) — so this is acceptable *as designed*, but
the residual TOCTOU between verify and launch should be noted, and the Windows
`icacls` hardening is best-effort/silent (`disk_cache.rs:675-682`) so on a host
without `icacls` a shared dir is unprotected.

**M-D. `set_override_dir` / `disk_cache()` memoisation race.** `set_override_dir`
(`disk_cache.rs:279-284`) clears the memo under one lock; `disk_cache()`
(`disk_cache.rs:327-334`) re-resolves under another (`HANDLE`). A builder call
concurrent with an in-flight `Engine::sql` lookup can transiently resolve the
old dir — the doc (`disk_cache.rs:275-278`) acknowledges this as "races
benignly." Benign for the intended single-threaded-build usage; flag as a
documented limitation, not a bug.

### LOW

**L-A. `DefaultHasher` (SipHash) byte output is not stable across Rust std
versions** — used for both the in-process key (`jit_compiler.rs:439 hash_ptx`),
the engine spec hash (`engine.rs:152`), and the disk body digest
(`disk_cache.rs:794`). The disk-cache file already notes this (`C-1`,
`disk_cache.rs:117-130`): a local toolchain bump with the same crate version
changes keys → benign miss + rewrite, and changes the digest → recomputed
consistently. Net effect is safe-degrade (miss), so Low, but the in-process
collision-probability claims ("2^-64") are toolchain-dependent.

**L-B. In-process cache key hashes PTX *text*, disk cache key hashes the
*spec* — two different identities.** `from_ptx` keys on `hash_ptx(ptx)`
(`jit_compiler.rs:491`) and re-verifies the full string on hit (collision-safe,
good). The disk layer keys on the spec Debug hash with no text re-verification —
its only guard is the salt + the body integrity digest. This is internally
consistent but worth noting: the disk layer's correctness rests entirely on the
salt being bumped (the documented JIT-M1 footgun), whereas the in-process layer
is self-validating.

**L-C. `Const` rejects `Literal::Null` (`ptx_gen.rs:1223`).** Any plan that
constant-folds a literal NULL into a kernel value reaches an error rather than
producing a NULL-validity value. Presumably the planner never emits it, but it
is an unhandled-by-design path with no test asserting the planner won't.

**L-D. `Mul128` is a truncating (wrapping) multiply with overflow discarded
silently** (`ptx_gen.rs:751-755,809-810`). Documented to match
`i128::wrapping_mul`/Arrow "checks at validation layer." If the validation layer
isn't actually wired for the GPU path, decimal overflow is silent. Out of
direct scope but worth a cross-check.

### Things done correctly (notable)
- 128-bit collision-safe in-process cache with full-string re-verification and
  `Slot::Collision` fallback (`jit_compiler.rs:340-364,517-531`).
- Race-free concurrent compile via per-key `Arc<OnceCell>` releasing the lock
  before PTXAS; failed compile does not poison the slot
  (`jit_compiler.rs:534-543`, test at `jit_compiler.rs:1210`).
- Path-traversal hardening is thorough: `valid_key` charset gate at the
  filename trust boundary, rejects `..`, separators, `:`, NUL, all-dots
  (`disk_cache.rs:761-773`); tested (`disk_cache.rs:1159,1194`).
- Codegen-injection hardening: all literals emitted as hex bit-patterns
  restricting output to `[0-9A-F]` (`ptx_gen.rs:1213-1276,648`); kernel-name
  validation rejects reserved words / `__` / `_param_` (`ptx_gen.rs:1651`).
- Atomic tempfile-then-rename writes + integrity header
  (`disk_cache.rs:589-621`); corrupt/headerless/tampered → miss
  (`disk_cache.rs:544-557`).
- Unsigned row-index widening (`mul.wide.u32`) consistently on value and
  validity paths — the C-3 fix — preventing sign-extension OOB above 2^31 rows
  (`ptx_gen.rs:377,452,1030`). Note the s32 `mad.lo` tid still caps a launch at
  `i32::MAX` rows (`ptx_gen.rs:205-215`); host must enforce.

---

## 2. TESTS

**Coverage by file:**
- `jit_compiler.rs` (lines 770-1236): strong on cache *state machine* — LRU
  eviction/reordering, hit/miss/collision counting exactness, concurrent
  compile-once-under-contention (16 threads), failed-compile-no-poison,
  env-var cap parsing, error-shape migration. **No test invokes the real CUDA
  driver** (by design — uses null-handle stub loader), so `load_uncached`,
  `decode_log`, the option-array marshalling, and PTXAS-error formatting are
  **untested**.
- `ptx_gen.rs` (lines 1793-2638 + cast/temporal/decimal/dataflow modules to
  4602): broad codegen golden/shape tests — validity emission, IsNull/IsNotNull,
  Select/nested-Select, Concat rejection, all cast pairs, temporal arith,
  Decimal128 add/sub/mul/cmp split-register patterns, dataflow dependency walk.
  Good breadth. These assert PTX *text shape*, never PTX that is assembled or
  executed.
- `disk_cache.rs` (lines 851-1326): round-trip, miss, atomic-no-tmp-leak,
  overwrite, nested-dir creation, corrupt→miss, traversal→miss/no-op, V-7
  header/tamper/legacy, Unix `0o700` perms, Windows round-trip-after-icacls,
  salt/version key rotation.

**Is the cache tested for eviction / corruption / concurrent?**
- Eviction: **yes** (in-process LRU, `jit_compiler.rs:843,885`). Disk eviction:
  **N/A — no eviction exists** (see M-B).
- Corruption: **yes** for disk (`corrupt_entry_falls_through_to_miss`
  `disk_cache.rs:964`, `tampered_body_is_a_miss` `:1254`).
- Concurrent: **in-process yes** (`from_ptx_compiles_once_under_contention`
  `:986`). **Disk: NO** — `rename_is_atomic_no_partial_file` (`disk_cache.rs:904`)
  explicitly does *not* race threads ("would be flaky on CI") and only asserts
  no leftover `.tmp`. There is **zero** multi-thread / multi-process disk-cache
  test. Given the whole atomicity argument rests on concurrent rename, this is
  the biggest test gap.

**~85%?** Cache *state-machine* and *codegen-shape* coverage is plausibly ~85%.
But three whole behavioural classes are at ~0%: (a) real PTX assembly/launch
(no GPU test asserts an emitted kernel actually computes the right answer);
(b) concurrent/multi-process disk cache; (c) integer DIV/0 and `Literal::Null`.

**Enhancements needed:**
1. Concurrent disk store/lookup test (spawn N threads racing the same key; assert
   final file integrity-verifies and lookup returns valid body). A
   multi-*process* variant (fork/exec self) would cover the real serverless race.
2. An end-to-end "emit PTX → assemble → launch → compare to host reference" test
   behind a `#[cfg(feature = "cuda")]` gate for at least one arithmetic, one
   validity, and one Decimal128 kernel — currently nothing proves the emitted
   bytes are *correct*, only that they *match a golden string*.
3. Integer `Div` by zero / `INT_MIN/-1` behaviour test (asserting whichever
   policy is chosen for C-B).
4. A multi-output validity kernel test that would expose H-A's coarse AND-fold.
5. A test that the disk key changes when the SM target changes (will fail today
   → drives C-A).
6. `decode_log` truncation/no-NUL-terminator unit test.

---

## 3. NEW FEATURES / DIRECTIONS

1. **Cubin/SASS caching, not just PTX.** The disk cache stores PTX text and
   still pays the ~10 ms PTXAS assembly on every cold process
   (`jit_compiler.rs:13-22` acknowledges this). Caching the assembled cubin
   (via `cuModuleLoadData` of a cubin, or `nvJitLink`/offline `ptxas`) would
   eliminate the cold-start cost entirely. This *requires* C-A first (cubin is
   arch-specific, so the key MUST include compute capability + driver/PTXAS
   version).
2. **Fatbin / multi-arch cache entries.** Store one entry per `(spec, arch)` or a
   fatbin spanning several SM versions so a heterogeneous fleet shares a cache.
   Naturally falls out of C-A's arch-in-key change.
3. **Persistent cache versioned across releases via a manifest** rather than
   the salt-rotation-and-leak model (M-B). A small `index.json` mapping key →
   {arch, driver, codegen_fp, size, last_used} enables LRU/size eviction and a
   clean cross-version story, replacing the "forget to bump `CODEGEN_VERSION`"
   footgun with an automatic fingerprint (the `BOLT_CODEGEN_FINGERPRINT`
   build.rs hook at `disk_cache.rs:148-175` is the intended seam — wire it up).
4. **Runtime-selected `PTX_TARGET`/`PTX_VERSION`** from the live device instead
   of the `sm_70` constant (`ptx_gen.rs:13-17`) — unlocks newer-arch
   instructions and is the precondition that makes C-A urgent.
5. **NVRTC integration for non-trivial kernels** (string ops, complex CASE)
   where hand-written PTX templating is brittle; the module name already
   anticipates NVRTC (`jit_compiler.rs:3-5`).
6. **Precise per-output NULL propagation**: actually invoke
   `output_input_dependencies` (`ptx_gen.rs:3668`) in `compile` and emit a
   per-output AND-tree, fixing H-A and enabling correct multi-output validity.

---

## Severity summary
| ID | Sev | One-liner | Cite |
|----|-----|-----------|------|
| C-A | Critical | Disk/module cache key omits GPU arch / PTX target / driver version | `ptx_gen.rs:15`, `disk_cache.rs:211,236`, `engine.rs:151` |
| C-B | Critical | Integer Div lowered to `div.s32/s64` with no zero/overflow guard (UB) | `ptx_gen.rs:1534-1535` |
| H-A | High | Coarse AND-of-all-inputs NULL prop; precise path unused; CASE branch NULL semantics wrong | `ptx_gen.rs:356-389,442-456,3553-3567` |
| H-B | High | PTXAS log truncated at 4 KB; size-slot cast brittle | `jit_compiler.rs:557-581,752` |
| M-A | Med | Module-cache cap frozen for process; stale TODO | `jit_compiler.rs:141-149` |
| M-B | Med | Disk cache unbounded — no eviction/TTL/size cap | `disk_cache.rs:589` |
| M-C | Med | TOCTOU between disk verify and launch; Windows ACL best-effort | `disk_cache.rs:544-557,658-683` |
| M-D | Med | Override-dir / handle memo race (documented benign) | `disk_cache.rs:279-284,327-334` |
| L-A | Low | `DefaultHasher` not stable across std versions | `jit_compiler.rs:439`, `disk_cache.rs:794` |
| L-B | Low | In-proc keys text (self-verifying); disk keys spec (salt-dependent) | `jit_compiler.rs:491`, `disk_cache.rs:236` |
| L-C | Low | `Literal::Null` const unhandled-by-design, no planner-guard test | `ptx_gen.rs:1223` |
| L-D | Low | `Mul128` silently wrapping; overflow check assumed elsewhere | `ptx_gen.rs:751-755` |
