// SPDX-License-Identifier: Apache-2.0

//! Optional disk-backed PTX cache (v0.6 / M6).
//!
//! # Why this exists
//!
//! The in-process module cache (`Engine::module_cache` + the
//! PTX-text-hash cache in [`super::jit_compiler`]) eliminates the
//! `cuModuleLoadDataEx` cost for repeat queries *within a single
//! process*. A fresh process — say, a CLI invocation, a benchmark
//! harness restart, or a serverless function cold start — has to
//! re-run the codegen pipeline (PhysicalPlan → PTX text) from scratch
//! every time, even though that step is byte-for-byte deterministic.
//!
//! This module provides an opt-in persistent layer: a directory of
//! `<hash>.ptx` files, one per cached spec, indexed by the same 128-bit
//! content hash the in-process module cache uses. On a cold-process
//! cache miss the caller looks up the disk cache first; on a hit it
//! gets the PTX text without re-running codegen, then hands the PTX to
//! `CudaModule::from_ptx` (which still pays the PTXAS assembly cost
//! once, since it's a fresh process). On a disk miss the caller runs
//! codegen and writes the result back through to both layers.
//!
//! # Opt-in
//!
//! The cache is **disabled by default** to keep the historical
//! zero-side-effect contract of `Engine::sql` intact. Two mechanisms
//! enable it:
//!
//! 1. **Environment variable `BOLT_PTX_CACHE_DIR`** — if set to any
//!    non-empty path, that directory is used as the cache root. This
//!    is the simplest way to flip the cache on for a benchmark run or
//!    a serverless deployment.
//!
//! 2. **Engine::Builder::persistent_cache(path)** — overrides the env
//!    var. `EngineBuilder::build()` calls [`set_override_dir`] with the
//!    configured path, so a builder-configured engine reads/writes
//!    cubins at that directory through the JIT compile path
//!    (`Engine::get_or_build_module` → [`disk_cache`]) without the env
//!    var being set. A default-built engine installs `None`, which
//!    clears any prior override and re-falls-back to the env var.
//!
//! If neither mechanism is active, [`disk_cache`] returns `None` and
//! all lookups / stores are zero-cost no-ops.
//!
//! # Path resolution
//!
//! When the configured path is *relative* or absent (env var unset),
//! the resolver falls back to the platform-conventional user cache
//! directory:
//!
//! - Linux:   `$XDG_CACHE_HOME/craton-bolt/ptx/` or `~/.cache/craton-bolt/ptx/`
//! - macOS:   `~/Library/Caches/craton-bolt/ptx/`
//! - Windows: `%LOCALAPPDATA%\craton-bolt\ptx\`
//!
//! No `dirs` crate dependency — we read the standard env vars directly
//! (`XDG_CACHE_HOME`, `HOME`, `LOCALAPPDATA`) and concatenate.
//!
//! # Atomicity
//!
//! Writes go to a temp file in the same directory and are then renamed
//! into place. `std::fs::rename` is atomic on a single filesystem on
//! all supported platforms (POSIX `rename(2)`; Windows `MoveFileEx`
//! with `MOVEFILE_REPLACE_EXISTING`). Concurrent readers therefore
//! never observe a partial file. Two writers racing on the same key
//! both produce identical bytes (the codegen pipeline is
//! deterministic), so the last-writer-wins outcome is harmless.
//!
//! # Security boundary (V-7) — fail-closed directory trust
//!
//! The bytes in this cache are read back and **launched** as PTX, so a
//! cache directory another local user can write is a code-execution
//! surface. The integrity boundary is therefore **directory ownership +
//! permissions plus key-bound integrity**, NOT the strength of the
//! integrity hash. The hash is a non-cryptographic `DefaultHasher` digest
//! and lives in the same directory as the body, so an attacker who can
//! write the directory could also recompute it — it only guards against
//! accidental corruption, naive tampering, and (via key binding) cross-key
//! content replay.
//!
//! On [`DiskPtxCache::open`] the cache directory is verified **fail-closed**:
//!
//! - Unix: the dir must be owned by the current effective uid and must not
//!   be group- or world-writable (`mode & 0o022 == 0`).
//! - Windows: the dir's ACL is locked to the current user via `icacls`, and
//!   a failure of that pass disables the disk cache.
//!
//! If the directory is not trusted, the disk layer is **disabled** (lookups
//! miss, stores are no-ops) and a warning is logged — the engine keeps
//! running on the in-process module cache + codegen. Because PTX is
//! deterministically re-derivable from source, a disabled disk cache is
//! always safe.
//!
//! Each on-disk entry additionally carries a `#bolt-ptx-cache v1 <digest>`
//! header whose digest is **bound to the cache key**, so a file's bytes
//! cannot be replayed under a different `<key>.ptx` filename. Lookups read
//! the bytes once and verify before returning them (TOCTOU-safe — no
//! re-open between check and use).

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use parking_lot::Mutex;

/// Environment variable that enables the disk-backed PTX cache and
/// names its root directory. Unset, empty, or unreadable → cache
/// disabled.
pub const DISK_PTX_CACHE_ENV: &str = "BOLT_PTX_CACHE_DIR";

/// Environment variable that caps the **total bytes** of cached `*.ptx`
/// entries (LRU-by-mtime eviction). Naming mirrors the GPU memory pool's
/// `CRATON_BOLT_POOL_MAX_BYTES` knob. Unset / unparseable → [`DEFAULT_MAX_CACHE_BYTES`].
/// A value of `0` disables the byte cap (entry-count cap still applies).
pub const DISK_PTX_CACHE_MAX_BYTES_ENV: &str = "CRATON_BOLT_PTX_CACHE_MAX_BYTES";

/// Environment variable that caps the **number** of cached `*.ptx` entries.
/// Unset / unparseable → [`DEFAULT_MAX_CACHE_ENTRIES`]. A value of `0`
/// disables the entry-count cap (byte cap still applies).
pub const DISK_PTX_CACHE_MAX_ENTRIES_ENV: &str = "CRATON_BOLT_PTX_CACHE_MAX_ENTRIES";

/// Default total-bytes cap for the on-disk PTX cache (64 MiB). PTX modules
/// are small (a few KiB each), so this comfortably holds thousands of
/// kernels while still bounding unbounded growth on a long-lived cache dir.
pub const DEFAULT_MAX_CACHE_BYTES: u64 = 64 * 1024 * 1024;

/// Default entry-count cap for the on-disk PTX cache. A second, independent
/// bound so a pathological flood of tiny entries can't blow up the directory
/// inode count even while staying under the byte cap.
pub const DEFAULT_MAX_CACHE_ENTRIES: u64 = 4096;

/// Codegen-version salt for the **on-disk** PTX cache key (fixes
/// JIT-M1).
///
/// # Why this exists
///
/// The disk key is derived from the [`KernelSpec`] content hash, which
/// captures *what query plan* a kernel implements but NOT *how the PTX
/// was emitted*. The in-process PTX-text-hash cache re-validates the
/// full PTX string on every hit, so it can never serve stale text; the
/// disk cache, by contrast, returns the on-disk bytes verbatim. That
/// means a populated cache directory written by an OLD binary — one
/// with different PTX emission but an unchanged `KernelSpec` hash —
/// would be loaded as-is by a NEW binary, yielding wrong kernels.
///
/// Folding this constant into the disk key (see the key composition in
/// `engine.rs`) guarantees that any change to PTX emission rotates the
/// on-disk filename, so the stale entry simply misses and the new
/// binary re-runs codegen.
///
/// # MAINTAINERS: bump this on ANY change to PTX emission
///
/// Increment `CODEGEN_VERSION` whenever you change anything that alters
/// the emitted PTX *text* for an otherwise-identical `KernelSpec`,
/// including but not limited to:
///   - `PTX_VERSION` / `PTX_TARGET` in `ptx_gen.rs`,
///   - any instruction mnemonic, modifier, or rounding mode,
///   - register naming / layout / allocation strategy,
///   - kernel signature, parameter order, or overall kernel structure,
///   - constant-folding or lowering changes that reshape the output.
///
/// Forgetting to bump it re-introduces JIT-M1. When in doubt, bump.
/// The crate version is also folded into the salt (see
/// [`codegen_salt`]) as a cheap cross-release guard, but that only
/// protects across published releases — `CODEGEN_VERSION` is what
/// protects within a release / between local dev builds.
///
/// # C-1: a Rust-std (toolchain) upgrade is ALSO a key-rotation event
///
/// The on-disk key and the V-7 integrity digest are derived from
/// `std::collections::hash_map::DefaultHasher` (SipHash-1-3), whose byte
/// output is **not contractually stable across Rust std versions**. The
/// disk cache is safe today only because the crate version is folded into
/// the salt and a published release bumps that version — but a *local*
/// toolchain bump (same crate version) could silently change every key
/// while still re-deriving the digest with the new hasher, so reads simply
/// miss rather than mis-route (the body is re-hashed on read with the same
/// toolchain that wrote nothing yet). Treat a `rustc`/std upgrade as a
/// codegen-freshness event: bump this constant (or wire the `build.rs`
/// `BOLT_CODEGEN_FINGERPRINT` below) so the key rotates deterministically
/// instead of relying on `DefaultHasher`'s undocumented stability.
///
/// # History
///   - v1: initial.
///   - v2: C-3 — validity-byte addressing switched from signed
///     `cvt.s64.s32` to unsigned `mul.wide.u32` in `ptx_gen.rs`; this
///     changes emitted PTX text for validity-carrying kernels, so the
///     disk key MUST rotate to avoid serving the old (mis-addressing) PTX.
///
/// Defense-in-depth: [`codegen_salt`] *additionally* folds in an
/// optional compile-time codegen fingerprint
/// ([`CODEGEN_FINGERPRINT`], from `BOLT_CODEGEN_FINGERPRINT`) when a
/// `build.rs` provides one. Where present, that fingerprint rotates the
/// salt automatically on any codegen change, so a forgotten bump of this
/// constant no longer silently serves stale PTX. Until that env var is
/// wired up, this constant remains the load-bearing in-release guard.
pub(crate) const CODEGEN_VERSION: u32 = 2;

/// Optional build-time codegen fingerprint env var.
///
/// # Why this exists (defense-in-depth against a forgotten `CODEGEN_VERSION` bump)
///
/// [`CODEGEN_VERSION`] is the *manual* freshness guard: a human has to
/// remember to bump it whenever PTX emission changes. That single point
/// of failure is exactly the JIT-M1 hazard — forget the bump and a NEW
/// binary happily serves PTX a structurally-different OLD binary wrote.
///
/// To reduce reliance on that manual constant we *also* fold an
/// automatically-derived fingerprint into the salt when one is available
/// at compile time. `build.rs` emits
/// `cargo:rustc-env=BOLT_CODEGEN_FINGERPRINT=<hash-of-codegen-surface>`
/// (a stable 128-bit FNV-1a digest over the `src/jit/*.rs` codegen tree
/// plus `src/plan/physical_plan.rs`, the plan IR codegen lowers into PTX),
/// and this helper consumes it via [`option_env!`] with **no** new
/// dependency. Because the digest rotates on *any* change to the codegen
/// source, a forgotten [`CODEGEN_VERSION`] bump no longer silently serves
/// stale PTX — the key rotates automatically instead.
///
/// We consume it via [`option_env!`] (rather than `env!`) so the file
/// still compiles even if the env var is absent — e.g. if `build.rs`
/// couldn't read `src/jit/` and fell back to emitting nothing. In that
/// case `option_env!` resolves to `None` at compile time and the salt
/// falls back to `CODEGEN_VERSION` + crate version alone — i.e. the
/// previous behaviour, never weaker. This also keeps the
/// `--no-default-features --features cuda-stub` build working unchanged.
const CODEGEN_FINGERPRINT: Option<&str> = option_env!("BOLT_CODEGEN_FINGERPRINT");

/// Compose the codegen-version salt component for the disk cache key.
///
/// The salt is **defense-in-depth**: it combines three independent
/// freshness signals so that a forgotten manual bump cannot, on its own,
/// re-introduce JIT-M1 (stale PTX served as correct):
///
///   1. [`CODEGEN_VERSION`] — the manual, in-release guard (`cgN`).
///   2. The crate version (`CARGO_PKG_VERSION`, `vX.Y.Z`) — a cheap
///      cross-release guard so two *published* releases that happen to
///      share a `CODEGEN_VERSION` value still land in distinct on-disk
///      keys. Across releases the crate version always changes, so even a
///      forgotten `CODEGEN_VERSION` bump can't serve another release's
///      stale PTX.
///   3. The compile-time codegen fingerprint
///      ([`CODEGEN_FINGERPRINT`], `fp<hash>`) — supplied by `build.rs`,
///      which hashes the `src/jit/*.rs` codegen tree plus
///      `src/plan/physical_plan.rs` into a stable digest and exports
///      `BOLT_CODEGEN_FINGERPRINT`. It makes the salt rotate
///      *automatically* on any change to the codegen surface, so a
///      forgotten manual bump is caught even between local dev builds of
///      the same crate version. It is absent only in the degenerate case
///      where `build.rs` could not read `src/jit/` (it then emits nothing
///      and the salt falls back to signals 1+2 — never weaker).
///
/// NOTE / MAINTAINERS: this salt MUST change whenever the emitted PTX
/// *text* changes for an otherwise-identical `KernelSpec`. The `build.rs`
/// fingerprint (signal 3) now does this automatically for any change to
/// the `src/jit/` codegen source (and `src/plan/physical_plan.rs`);
/// bumping [`CODEGEN_VERSION`] remains the
/// belt-and-braces in-release guard — see the maintainer note on
/// [`CODEGEN_VERSION`].
///
/// Returned as a short, filename-safe string (no path separators, no
/// shell metacharacters — the crate version is `MAJOR.MINOR.PATCH[-pre]`
/// which contains only `[0-9A-Za-z.\-+]`, and any fingerprint we emit is
/// expected to be hex). Callers prepend this to the spec-hash portion of
/// the disk key. Even if a future fingerprint contained a stray
/// separator, [`valid_key`] is the trust boundary that rejects an unsafe
/// final key, so the cache degrades to a miss rather than a path escape.
#[must_use]
pub(crate) fn codegen_salt() -> String {
    // JIT-arch: fold the PTX target arch + ISA `.version` into the salt.
    //
    // The cache key historically carried only a codegen version + crate
    // version. The emitted module, however, is pinned to a specific GPU
    // architecture via `.target sm_70` and a specific PTX ISA via
    // `.version 7.5` (both in `ptx_gen.rs`). Those are constants today, so
    // they can't drift within a build — but if the target ever becomes
    // device-derived (so an `sm_70` host and an `sm_90` host run the same
    // binary), a cached kernel compiled for one arch could be mis-routed to
    // a process targeting another. Folding the arch token into the salt makes
    // the key rotate per-arch, so that mis-route is impossible by
    // construction. Because the salt is the single source feeding BOTH the
    // in-process module key (`exec::module_cache::compose_disk_key`) and the
    // on-disk key ([`disk_key`]), both inherit this automatically.
    let arch = arch_salt_token();
    match CODEGEN_FINGERPRINT {
        Some(fp) => format!(
            "cg{}-v{}-{}-fp{}",
            CODEGEN_VERSION,
            env!("CARGO_PKG_VERSION"),
            arch,
            fp
        ),
        None => format!(
            "cg{}-v{}-{}",
            CODEGEN_VERSION,
            env!("CARGO_PKG_VERSION"),
            arch
        ),
    }
}

/// Derive a compact, filename-safe token from the PTX target arch + ISA
/// version strings (`crate::jit::ptx_gen::PTX_TARGET` /
/// [`PTX_VERSION`](crate::jit::ptx_gen::PTX_VERSION)).
///
/// The raw directives (`.target sm_70`, `.version 7.5`) contain spaces and a
/// leading `.`, neither of which is in the [`valid_key`] charset, so we
/// reduce them to a single hyphen-free token: the last whitespace-separated
/// word of each directive (`sm_70`, `7.5`), then map any remaining
/// non-`[0-9A-Za-z._]` byte to `_`. The result for today's constants is
/// `arch_sm_70_isa_7.5`. Keeping it derived (rather than hardcoding the
/// token) means the salt tracks the directives automatically if they change.
fn arch_salt_token() -> String {
    use crate::jit::ptx_gen::{PTX_TARGET, PTX_VERSION};
    // `.target sm_70` -> "sm_70"; `.version 7.5` -> "7.5". Fall back to the
    // whole trimmed string if there's no whitespace.
    let arch = PTX_TARGET
        .rsplit(char::is_whitespace)
        .next()
        .unwrap_or(PTX_TARGET);
    let isa = PTX_VERSION
        .rsplit(char::is_whitespace)
        .next()
        .unwrap_or(PTX_VERSION);
    let mut s = format!("arch_{arch}_isa_{isa}");
    // Sanitise to the `valid_key` charset so the salt can never introduce a
    // path separator / unsafe byte into the final cache key.
    s = s
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b == b'.' || b == b'_' {
                b as char
            } else {
                '_'
            }
        })
        .collect();
    s
}

/// Compose the full on-disk cache key for a kernel.
///
/// The key has three domain-separated components joined by `-`:
///   1. [`codegen_salt`] — the codegen-version + crate-version (+ optional
///      build-time codegen-fingerprint) salt that fixes JIT-M1 (a codegen
///      change rotates the key, so stale entries miss and codegen
///      re-runs).
///   2. `entry` — the kernel entry-point symbol, so two kernels with
///      identical `KernelSpec` content but different entry symbols
///      (e.g. `KERNEL_ENTRY` vs `PREDICATE_ENTRY`) never alias.
///   3. The 128-bit `KernelSpec` content hash, hex-encoded.
///
/// This is the single source of truth for disk-key composition; callers
/// (currently `engine.rs`) should route through here so the salt is
/// applied consistently. It deliberately does NOT touch the in-process
/// `KernelSpecCache` key — that cache re-validates PTX content on every
/// hit, so it needs no salt and its domain bytes must stay unchanged.
#[must_use]
pub(crate) fn disk_key(entry: &str, spec_hash_hi: u64, spec_hash_lo: u64) -> String {
    format!(
        "{}-{}-{}",
        codegen_salt(),
        entry,
        hash_to_key(spec_hash_hi, spec_hash_lo),
    )
}

/// Subdirectory under the platform cache root used when the env var
/// is unset but a builder override sets a non-absolute path, or when
/// future revisions opt into auto-resolution.
///
/// Only referenced by the Linux/BSD branch of [`platform_default_dir`]
/// (macOS and Windows compose the path inline); cfg-gated to avoid a
/// dead-code warning on those targets.
#[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
const CACHE_SUBDIR: &str = "craton-bolt/ptx";

/// Optional builder-supplied override path. Installed via
/// [`set_override_dir`] before any cache lookup; an installed override
/// takes precedence over the env var.
///
/// The `Mutex<Option<PathBuf>>` lets the builder swap the path during
/// engine construction without needing exclusive process state — the
/// global is intentionally process-wide because the PTX-text-hash key
/// is also process-wide.
static OVERRIDE_DIR: OnceLock<Mutex<Option<PathBuf>>> = OnceLock::new();

fn override_slot() -> &'static Mutex<Option<PathBuf>> {
    OVERRIDE_DIR.get_or_init(|| Mutex::new(None))
}

/// Install a builder-supplied cache directory. Subsequent calls to
/// [`disk_cache`] resolve to this directory regardless of the
/// `BOLT_PTX_CACHE_DIR` env var. Passing `None` clears the override
/// and re-falls-back to the env var.
///
/// Idempotent — replacing the same path is a cheap no-op-equivalent.
/// Safe to call from `Engine::Builder::build()` on the main thread
/// before any `Engine::sql` invocation; calling it concurrently with
/// in-flight cache lookups races benignly (the worst outcome is one
/// or two lookups hitting the previous directory).
pub fn set_override_dir(dir: Option<PathBuf>) {
    *override_slot().lock() = dir;
    // Invalidate the memoised handle so the next `disk_cache()` call
    // re-resolves against the new override.
    *HANDLE.get_or_init(|| Mutex::new(None)).lock() = None;
}

/// Snapshot the current builder-supplied override directory (if any).
///
/// Returns `Some(path)` when [`set_override_dir`] has been called with a
/// non-`None` argument and that path is still installed; returns `None`
/// when no override is active (in which case the env var
/// [`DISK_PTX_CACHE_ENV`] is the only path that would enable the
/// cache).
///
/// This is the inverse of [`set_override_dir`] and exists primarily so
/// the `EngineBuilder` integration tests can assert that the builder
/// successfully propagated its `persistent_cache(path)` knob into the
/// process-wide JIT state. Production callers typically don't need this
/// — use [`disk_cache`] to obtain a usable handle instead.
#[must_use]
pub fn current_override_dir() -> Option<PathBuf> {
    override_slot().lock().clone()
}

/// Process-wide cache handle, memoised after first resolution.
///
/// `None` means the cache is disabled (no env var, no override, or
/// the directory could not be created). `Some` wraps the resolved
/// absolute path. Wrapped in `Mutex<Option<...>>` (rather than a
/// `OnceLock<Option<...>>`) so [`set_override_dir`] can invalidate
/// the memoisation when the builder rebinds the path.
static HANDLE: OnceLock<Mutex<Option<DiskPtxCache>>> = OnceLock::new();

/// Resolve (and memoise) the process-wide disk cache handle.
///
/// Returns `None` if neither the env var nor a builder override is
/// set, or if the resolved directory cannot be created.
///
/// Memoisation rules:
/// - The first call that *successfully* resolves a directory caches
///   the resulting `DiskPtxCache` handle; subsequent calls are a
///   single `Mutex` lock + clone of the memoised handle.
/// - A call that resolves to `None` (cache disabled) is NOT memoised:
///   re-checking on the next call lets a late env-var / override
///   install kick in without needing a process restart.
/// - [`set_override_dir`] explicitly invalidates the memo so the next
///   call re-resolves against the freshly-installed override.
pub fn disk_cache() -> Option<DiskPtxCache> {
    let slot = HANDLE.get_or_init(|| Mutex::new(None));
    let mut guard = slot.lock();
    if guard.is_none() {
        *guard = resolve_cache_dir().and_then(|p| DiskPtxCache::open(p).ok());
    }
    guard.clone()
}

/// Resolve the cache root path from (in priority order):
///   1. Builder override installed via [`set_override_dir`].
///   2. `BOLT_PTX_CACHE_DIR` environment variable.
///
/// If neither is set, returns `None` — the cache stays disabled.
/// A platform-default location (e.g. `~/.cache/craton-bolt/ptx`) is
/// computed for convenience and exposed via [`platform_default_dir`],
/// but is *not* implicitly used as a fallback: the cache is opt-in.
fn resolve_cache_dir() -> Option<PathBuf> {
    if let Some(p) = override_slot().lock().clone() {
        return Some(p);
    }
    let raw = std::env::var(DISK_PTX_CACHE_ENV).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(PathBuf::from(trimmed))
}

/// Compute the platform-default cache directory. Exposed for the
/// builder / CLI to suggest a sensible path; not auto-selected by
/// [`disk_cache`].
///
/// Resolution rules:
/// - Linux/BSD: `$XDG_CACHE_HOME/craton-bolt/ptx/` or
///   `$HOME/.cache/craton-bolt/ptx/`.
/// - macOS:     `$HOME/Library/Caches/craton-bolt/ptx/`.
/// - Windows:   `%LOCALAPPDATA%\craton-bolt\ptx\` (falls back to
///   `%USERPROFILE%\AppData\Local\craton-bolt\ptx\` if LOCALAPPDATA
///   is unset).
///
/// Returns `None` if no relevant env var is set (rare; e.g.
/// stripped-down container with no HOME).
pub fn platform_default_dir() -> Option<PathBuf> {
    #[cfg(target_os = "windows")]
    {
        if let Ok(p) = std::env::var("LOCALAPPDATA") {
            if !p.is_empty() {
                return Some(PathBuf::from(p).join("craton-bolt").join("ptx"));
            }
        }
        if let Ok(p) = std::env::var("USERPROFILE") {
            if !p.is_empty() {
                return Some(
                    PathBuf::from(p)
                        .join("AppData")
                        .join("Local")
                        .join("craton-bolt")
                        .join("ptx"),
                );
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let home = std::env::var("HOME").ok()?;
        if home.is_empty() {
            return None;
        }
        Some(
            PathBuf::from(home)
                .join("Library")
                .join("Caches")
                .join("craton-bolt")
                .join("ptx"),
        )
    }
    #[cfg(all(not(target_os = "windows"), not(target_os = "macos")))]
    {
        if let Ok(xdg) = std::env::var("XDG_CACHE_HOME") {
            if !xdg.is_empty() {
                return Some(PathBuf::from(xdg).join(CACHE_SUBDIR));
            }
        }
        let home = std::env::var("HOME").ok()?;
        if home.is_empty() {
            return None;
        }
        Some(PathBuf::from(home).join(".cache").join(CACHE_SUBDIR))
    }
}

/// Handle to the on-disk PTX cache rooted at a particular directory.
///
/// Cheap to clone — wraps a single `Arc<PathBuf>` so concurrent callers
/// can each carry their own handle without lock contention. Per-entry
/// I/O takes no shared lock; the only synchronisation is the inherent
/// atomicity of `rename`.
///
/// # Disabled (fail-closed) state — V-7
///
/// A handle whose `disk_enabled` flag is `false` is a **disk no-op**:
/// [`Self::lookup`] always misses and [`Self::store`] silently writes
/// nothing. [`Self::open`] returns such a handle when the cache directory
/// fails the ownership / permission trust check (see the `open` doc). The
/// engine keeps running and falls back to its in-process module cache +
/// codegen, so a disabled disk layer is always safe: PTX is deterministically
/// re-derivable from source.
#[derive(Clone)]
pub struct DiskPtxCache {
    root: std::sync::Arc<PathBuf>,
    /// Whether disk I/O is permitted. `false` after a failed directory
    /// trust check (V-7) turns this handle into a disk no-op.
    disk_enabled: bool,
}

impl DiskPtxCache {
    /// Open (creating if needed) a cache rooted at `dir`. Returns an
    /// error only if the directory cannot be *created*; a directory that
    /// exists but fails the trust check yields a **disabled** (disk-no-op)
    /// handle rather than an error (see the fail-closed contract below).
    ///
    /// # Directory trust check — fail-closed (V-7)
    ///
    /// The on-disk cache stores PTX that is later read back and **launched**
    /// via `cuModuleLoadDataEx`. A cache directory that another local user
    /// can write — or that we don't even own — is therefore a code-execution
    /// surface: an attacker who can plant `<key>.ptx` files (and recompute
    /// the non-cryptographic integrity digest, which lives in the same
    /// directory) could get arbitrary PTX launched in our process. The hash
    /// strength is *not* the security boundary; **directory ownership +
    /// permissions** is. We therefore verify trust before using the dir and
    /// **fail closed** if it is not met.
    ///
    /// After `create_dir_all`, we tighten then *verify* the root:
    ///
    /// * **Unix** — we first best-effort `chmod 0o700` (owner-only rwx), then
    ///   `stat` the directory via [`dir_is_trusted_unix`] and require BOTH:
    ///     * the directory is owned by the current effective uid
    ///       (`st_uid == geteuid()`), and
    ///     * it is **not** group- or world-writable (`mode & 0o022 == 0`).
    ///   A pre-existing directory owned by another user, or one left
    ///   group/world-writable, fails the check even though the `chmod`
    ///   "succeeded" or was refused — we do not trust a dir we can't prove
    ///   is owner-private.
    ///
    /// * **Windows** — there is no portable `0o700` analogue in std. We run
    ///   the `icacls` ACL-tightening pass ([`harden_windows_dir`]) which
    ///   removes inherited ACEs and grants the current user sole Full-control.
    ///   Unlike the previous best-effort posture, a **failure of that pass
    ///   now disables the disk cache** (we can no longer prove the directory
    ///   is restricted to us), mirroring the Unix fail-closed branch. The
    ///   per-user `%LOCALAPPDATA%` default root still passes trivially; an
    ///   operator pointing `BOLT_PTX_CACHE_DIR` at a shared location we
    ///   cannot lock down falls back to in-memory only.
    ///
    /// # Fail-closed behavior
    ///
    /// If the trust check fails we do **not** hard-error the engine — we log
    /// a clear warning and return a handle with `disk_enabled == false`, so
    /// every subsequent [`Self::lookup`] misses and [`Self::store`] is a
    /// disk no-op. The in-process module cache and codegen path are
    /// untouched, and because PTX is deterministically re-derivable a
    /// disabled disk layer is always safe.
    pub fn open(dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;

        // V-7: tighten the cache root, then VERIFY ownership + permissions.
        // A directory we cannot prove is owner-private is not trusted, and
        // an untrusted dir disables the disk layer (fail-closed) rather than
        // being used with best-effort perms as before.
        #[cfg(unix)]
        let disk_enabled = {
            use std::os::unix::fs::PermissionsExt;
            // Best-effort first: try to bring a freshly-created (or fixable)
            // dir down to owner-only 0o700. The *verification* below — not
            // this call — is the trust decision, so we ignore failures here.
            let perms = std::fs::Permissions::from_mode(0o700);
            let _ = fs::set_permissions(&dir, perms);
            dir_is_trusted_unix(&dir)
        };
        // V-7 (Windows): tighten the ACL to the current user; a failure of
        // that pass now DISABLES the disk cache (fail-closed) instead of
        // proceeding with whatever ACL the directory happened to carry.
        #[cfg(windows)]
        let disk_enabled = harden_windows_dir(&dir);
        // On any other platform we cannot verify ownership/permissions
        // portably, so fail closed: the disk cache stays disabled.
        #[cfg(not(any(unix, windows)))]
        let disk_enabled = false;

        if !disk_enabled {
            log::warn!(
                "PTX disk cache at {} is not owner-private (untrusted \
                 ownership/permissions); disabling the on-disk cache and \
                 falling back to in-memory PTX only. PTX is re-derived from \
                 source, so correctness is unaffected.",
                dir.display()
            );
        }

        let cache = Self {
            root: std::sync::Arc::new(dir),
            disk_enabled,
        };
        // JIT-cache-bound: enforce the size / entry-count caps on open so a
        // directory left oversized by a previous process (or a lowered cap)
        // is trimmed before we start serving from it. Best-effort, and only
        // meaningful when the disk layer is enabled.
        if cache.disk_enabled {
            cache.enforce_bounds();
        }
        Ok(cache)
    }

    /// Path to this cache's root directory. Useful for tests, logging,
    /// and the builder's "where did the cache land?" diagnostic.
    pub fn root(&self) -> &Path {
        self.root.as_path()
    }

    /// Whether this handle's disk layer is enabled. `false` after the V-7
    /// directory trust check in [`Self::open`] failed — every `lookup`
    /// misses and every `store` is a disk no-op. Exposed for tests and for
    /// callers that want to surface "disk cache disabled" diagnostics.
    #[must_use]
    pub fn is_disk_enabled(&self) -> bool {
        self.disk_enabled
    }

    /// Construct a disk-disabled handle directly (test-only) so the
    /// fail-closed no-op contract can be asserted without having to
    /// fabricate an untrusted directory (impossible to do portably).
    #[cfg(test)]
    fn disabled_for_test(dir: PathBuf) -> Self {
        Self {
            root: std::sync::Arc::new(dir),
            disk_enabled: false,
        }
    }

    /// Compose the on-disk path for a given content-hash key, validating
    /// the key first (fixes V-3, path traversal).
    ///
    /// Returns `Some(path)` only when `key` passes [`valid_key`] — a
    /// strict filename-safe charset with no path separators, no `..`, and
    /// no NUL. Returns `None` for any key that could escape the cache
    /// root; callers turn that into a cache miss / store no-op.
    ///
    /// We validate here (rather than trusting the upstream key composer)
    /// because the key-composition helpers are `pub(crate)` and
    /// contractually accept arbitrary `entry` strings — the trust boundary
    /// is the moment the string becomes a filename, which is right here.
    fn entry_path(&self, key: &str) -> Option<PathBuf> {
        if !valid_key(key) {
            return None;
        }
        let mut p = (*self.root).clone();
        p.push(format!("{key}.ptx"));
        Some(p)
    }

    /// Look up `key` on disk. Returns `Some(ptx_text)` on a hit,
    /// `None` on a miss or any I/O failure. We intentionally do *not*
    /// surface read errors as `Err`: a corrupt or unreadable cache
    /// entry should silently fall through to the codegen path so the
    /// caller still gets a correct result.
    ///
    /// # Freshness (JIT-M1)
    ///
    /// Freshness across binary/codegen changes is guaranteed upstream by
    /// the codegen-version salt folded into the key (see [`codegen_salt`]
    /// / [`disk_key`]): when PTX emission changes, [`CODEGEN_VERSION`] (or
    /// the crate version) is bumped, the `key` rotates, and a stale entry
    /// written by the old binary simply misses here rather than being
    /// served as wrong PTX.
    ///
    /// # Path-traversal hardening (V-3)
    ///
    /// An invalid `key` (one that could escape the cache root — see
    /// [`valid_key`]) is treated as an immediate miss:
    /// [`Self::entry_path`] returns `None` and we never touch the
    /// filesystem with an unsanitised path.
    ///
    /// # Content-integrity check (V-7)
    ///
    /// The cache file carries a `#bolt-ptx-cache v1 <digest>` header line
    /// (see [`CACHE_HEADER_MAGIC`]). We parse that header, recompute
    /// [`body_digest`] over the body **bound to the lookup `key`**, and
    /// return the body **only** if the digests match. Binding the digest to
    /// the key means a file's contents cannot be replayed under a different
    /// `<key>.ptx` filename — a moved/renamed entry mismatches and misses.
    /// This catches accidental corruption / partial writes and trips on
    /// naive tampering. A file with no/old/garbled header (e.g. a raw-PTX
    /// entry from an older binary, or a digest mismatch) is treated as a miss
    /// so the caller recompiles and rewrites it in the current format. See
    /// [`Self::store`] for the write side.
    ///
    /// # Fail-closed disk layer (V-7)
    ///
    /// If this handle's disk layer is disabled (the cache directory failed
    /// the ownership/permission trust check in [`Self::open`]), every lookup
    /// is an immediate miss — we never read from an untrusted directory.
    ///
    /// # TOCTOU
    ///
    /// We read the file's bytes exactly once (`read_to_string`), verify the
    /// header against those same bytes, and return them — we never re-open
    /// the path after verification, so there is no time-of-check/time-of-use
    /// window between the integrity check and the bytes the caller launches.
    #[must_use]
    pub fn lookup(&self, key: &str) -> Option<String> {
        // V-7: a disabled (untrusted-dir) handle never touches disk.
        if !self.disk_enabled {
            return None;
        }
        // V-3: refuse to read through an unsanitised key.
        let path = self.entry_path(key)?;
        // TOCTOU-safe: read the bytes ONCE, then verify those same bytes.
        let raw = fs::read_to_string(path).ok()?;
        // V-7: require a well-formed integrity header, else miss.
        let (header_line, body) = split_header(&raw)?;
        let stored_digest = header_line.strip_prefix(CACHE_HEADER_MAGIC)?.trim();
        // V-7: the digest is bound to `key`, so contents planted under a
        // different filename (or a renamed entry) mismatch and miss.
        if stored_digest != body_digest(key, body) {
            // Corrupt, tampered, or replayed-under-another-key body -> treat
            // as a miss so the caller recompiles rather than launching
            // untrusted PTX.
            return None;
        }
        Some(body.to_string())
    }

    /// Persist `ptx` under `key` using a tempfile-then-rename to keep
    /// concurrent readers from ever observing a partial file.
    ///
    /// On a successful rename returns `Ok(())`. On a rename failure the
    /// stray tempfile is cleaned up best-effort and the underlying I/O
    /// error is propagated to the caller (`Err`). Callers treat a store
    /// error as non-fatal — the codegen pipeline is deterministic, so a
    /// concurrent writer racing on the same key produces identical
    /// bytes, and a failed write only means future processes re-run
    /// codegen rather than getting wrong results.
    ///
    /// (JIT-M2: the previous doc claimed a stat-the-target fallback that
    /// returned `Ok(())` when the content already landed at the target;
    /// no such fallback was implemented, so the doc is corrected here to
    /// match the actual propagate-the-error behavior.)
    ///
    /// # Path-traversal hardening (V-3)
    ///
    /// An invalid `key` (see [`valid_key`]) is a silent no-op: we return
    /// `Ok(())` without writing anything, so a traversal key can never
    /// clobber a file outside the cache root. `Ok(())` (rather than `Err`)
    /// keeps the best-effort contract — a refused store is indistinguishable
    /// to the caller from a successful one that a future process simply
    /// won't find.
    ///
    /// # Content-integrity header (V-7)
    ///
    /// The file is written as a `#bolt-ptx-cache v1 <digest>\n` header line
    /// (the [`body_digest`] of `ptx` **bound to `key`**) followed by the
    /// verbatim PTX body, so [`Self::lookup`] can verify the body — and that
    /// it was stored under this exact filename — on read.
    ///
    /// # Fail-closed disk layer (V-7)
    ///
    /// If this handle's disk layer is disabled (untrusted cache directory —
    /// see [`Self::open`]), the store is a silent no-op: we return `Ok(())`
    /// without writing anything, so a disabled disk layer never plants files
    /// in a directory we don't trust.
    pub fn store(&self, key: &str, ptx: &str) -> io::Result<()> {
        // V-7: a disabled (untrusted-dir) handle never writes to disk.
        if !self.disk_enabled {
            return Ok(());
        }
        // V-3: refuse to write through an unsanitised key. Best-effort
        // contract: a refused store is a silent no-op, not an error.
        let Some(target) = self.entry_path(key) else {
            return Ok(());
        };
        // V-7: prepend the integrity header. `body_digest` binds the body to
        // `key`, so the header line is self-describing and `lookup`
        // re-derives the same key-bound digest from the bytes after the
        // first `\n`.
        let contents = format!("{}{}\n{}", CACHE_HEADER_MAGIC, body_digest(key, ptx), ptx);
        // Tempfile name: same directory, suffixed with the OS PID and a
        // process-monotonic counter so concurrent writers in the same
        // process don't collide. We use a counter (not a random number)
        // to keep the test path deterministic and avoid pulling in a
        // `rand` dep just for this.
        let suffix = next_temp_suffix();
        let mut tmp = (*self.root).clone();
        let pid = std::process::id();
        tmp.push(format!("{key}.ptx.tmp.{pid}.{suffix}"));
        // Write the full header+body to the tempfile first.
        fs::write(&tmp, &contents)?;
        // Atomic rename. On Windows `fs::rename` already implements
        // `MOVEFILE_REPLACE_EXISTING` semantics in stable std.
        match fs::rename(&tmp, &target) {
            Ok(()) => {
                // JIT-cache-bound: after a successful insert, enforce the
                // size / entry-count caps so the directory can't grow without
                // limit over a long-lived process. Best-effort and never
                // fatal — the freshly-written entry stays, and a failed sweep
                // only means we retry the bound on the next store/open.
                self.enforce_bounds();
                Ok(())
            }
            Err(e) => {
                // Best-effort cleanup of the stray tempfile so we don't
                // leak inodes under repeated failures.
                let _ = fs::remove_file(&tmp);
                Err(e)
            }
        }
    }

    /// Enforce the cache's size / entry-count bounds via LRU-by-mtime
    /// eviction (fixes the unbounded-growth gap).
    ///
    /// # Policy
    ///
    /// The cache is capped by two independent limits, each overridable via a
    /// `CRATON_*` env var consistent with the crate's other knobs (e.g. the
    /// GPU pool's `CRATON_BOLT_POOL_MAX_BYTES`):
    ///
    ///   * total bytes — [`DISK_PTX_CACHE_MAX_BYTES_ENV`]
    ///     (default [`DEFAULT_MAX_CACHE_BYTES`]); and
    ///   * entry count — [`DISK_PTX_CACHE_MAX_ENTRIES_ENV`]
    ///     (default [`DEFAULT_MAX_CACHE_ENTRIES`]).
    ///
    /// A cap of `0` disables that particular limit. When either limit is
    /// exceeded we delete `*.ptx` entries oldest-first by last-modified time
    /// (an LRU approximation that needs no separate index) until BOTH limits
    /// are satisfied. PTX entries are deterministic and re-derivable, so an
    /// evicted entry simply costs a recompile on its next lookup.
    ///
    /// # Safety / robustness
    ///
    /// Entirely best-effort: every filesystem call is failure-tolerant and
    /// the method never panics and never returns an error. It only ever
    /// removes regular `*.ptx` cache files — never directories, never the
    /// in-flight `*.ptx.tmp.*` tempfiles `store` is racing on (so it cannot
    /// disturb the atomic-rename / integrity-header write path), and never
    /// anything outside the cache root.
    fn enforce_bounds(&self) {
        let max_bytes = env_u64(DISK_PTX_CACHE_MAX_BYTES_ENV, DEFAULT_MAX_CACHE_BYTES);
        let max_entries = env_u64(DISK_PTX_CACHE_MAX_ENTRIES_ENV, DEFAULT_MAX_CACHE_ENTRIES);
        // Both caps disabled -> nothing to do.
        if max_bytes == 0 && max_entries == 0 {
            return;
        }

        // Collect the committed cache entries: regular files whose name ends
        // in `.ptx` (NOT the `.ptx.tmp.<pid>.<n>` tempfiles a concurrent
        // `store` may be writing). Tolerate any per-entry I/O error by
        // skipping that entry.
        let read_dir = match fs::read_dir(self.root.as_path()) {
            Ok(rd) => rd,
            Err(_) => return, // can't scan -> give up this round, no panic.
        };
        struct Entry {
            path: PathBuf,
            mtime: std::time::SystemTime,
            size: u64,
        }
        let mut entries: Vec<Entry> = Vec::new();
        let mut total_bytes: u64 = 0;
        for dirent in read_dir.flatten() {
            let path = dirent.path();
            // Only committed `.ptx` files; explicitly skip tempfiles whose
            // name contains `.tmp.` so we never race the rename path.
            let name = dirent.file_name();
            let name = name.to_string_lossy();
            if !name.ends_with(".ptx") || name.contains(".tmp.") {
                continue;
            }
            let meta = match dirent.metadata() {
                Ok(m) if m.is_file() => m,
                _ => continue,
            };
            let size = meta.len();
            // Fall back to UNIX_EPOCH (oldest possible) when mtime is
            // unavailable, so an entry with no timestamp is evicted first.
            let mtime = meta.modified().unwrap_or(std::time::UNIX_EPOCH);
            total_bytes = total_bytes.saturating_add(size);
            entries.push(Entry { path, mtime, size });
        }

        let over_bytes = max_bytes != 0 && total_bytes > max_bytes;
        let over_entries = max_entries != 0 && entries.len() as u64 > max_entries;
        if !over_bytes && !over_entries {
            return;
        }

        // Oldest first (ascending mtime) so we evict least-recently-modified
        // entries until both caps are satisfied.
        entries.sort_by_key(|e| e.mtime);
        let mut count = entries.len() as u64;
        for e in &entries {
            let need_bytes = max_bytes != 0 && total_bytes > max_bytes;
            let need_entries = max_entries != 0 && count > max_entries;
            if !need_bytes && !need_entries {
                break;
            }
            // Best-effort delete; only adjust the running totals if it
            // actually succeeded so a stuck (e.g. permission-denied) file
            // doesn't make us under-count and stop early.
            if fs::remove_file(&e.path).is_ok() {
                total_bytes = total_bytes.saturating_sub(e.size);
                count = count.saturating_sub(1);
            }
        }
    }
}

/// Verify the cache root is owner-private (V-7, fail-closed trust check).
///
/// Returns `true` **only** when BOTH hold:
///   * the directory is owned by the current *effective* uid
///     (`st_uid == geteuid()`), and
///   * the directory is not group- or world-writable (`mode & 0o022 == 0`).
///
/// This is the Unix half of the fail-closed directory trust boundary. The
/// cache stores PTX that is later read back and launched, so a directory we
/// don't own, or one any other local user can write, is treated as
/// untrusted: [`DiskPtxCache::open`] disables the disk layer rather than
/// using it. We deliberately check ownership *and* the write bits (not just
/// the `0o700` we tried to set) because the `chmod` in `open` is best-effort
/// and may have been refused (e.g. a pre-existing dir owned by another user),
/// so the only sound signal is the *observed* post-stat state.
///
/// Uses `std::os::unix::fs::MetadataExt` for `st_uid`/`st_mode` and
/// `libc::geteuid()` for the current uid — both already in the dependency
/// set, so no new crate is pulled in. Any `stat` failure is treated as
/// untrusted (fail-closed).
#[cfg(unix)]
fn dir_is_trusted_unix(dir: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    let meta = match fs::metadata(dir) {
        Ok(m) => m,
        // Can't stat it -> can't prove it's safe -> fail closed.
        Err(_) => return false,
    };
    // SAFETY: `geteuid` is a pure getter with no preconditions and is
    // always-safe to call; it returns the effective uid of the process.
    let current_uid = unsafe { libc::geteuid() };
    let owned_by_us = meta.uid() == current_uid;
    // No group- or world-write bits (mode & 0o022 == 0).
    let not_group_world_writable = meta.mode() & 0o022 == 0;
    owned_by_us && not_group_world_writable
}

/// Tighten the ACL on the cache root to the current user and report whether
/// the directory can now be trusted (V-7, Windows analogue of the Unix
/// `0o700` + ownership check).
///
/// # Threat boundary
///
/// The on-disk cache stores PTX that is later read back and *launched*.
/// On a shared Windows host, if `BOLT_PTX_CACHE_DIR` points at a
/// world-writable directory, a different local user could plant or tamper
/// with `<key>.ptx` files for us to load. The per-user `%LOCALAPPDATA%`
/// default already lives under the user's profile ACL and is the
/// *primary* protection; this routine is defense-in-depth for the
/// explicit-shared-dir case.
///
/// # Implementation
///
/// We shell out to the built-in `icacls` tool rather than pull in the
/// `windows`/`winapi` crates (no new dependency, matching the task
/// budget). The invocation:
///
///   * `/inheritance:r` — remove inherited ACEs (a world-writable parent
///     would otherwise keep granting access), and
///   * `/grant:r "<user>":(OI)(CI)F` — replace the DACL with a single
///     Full-control grant to the current user, inherited by child files
///     `(OI)` and subdirectories `(CI)`.
///
/// The user principal is taken from `USERDOMAIN\USERNAME` when both are
/// present (the fully-qualified form `icacls` prefers), falling back to
/// bare `USERNAME`. If neither is set we cannot name a grantee to lock the
/// directory down, so we cannot trust it: return `false` (fail-closed).
///
/// # Fail-closed (V-7)
///
/// Unlike the previous best-effort posture, the return value is now the
/// Windows half of the directory trust decision. We suppress stdout/stderr
/// but DO inspect the exit status: `icacls` returning success means the DACL
/// was replaced with a sole current-user Full-control grant (inherited ACEs
/// removed), so the directory is trusted (`true`). A missing username, a
/// spawn/IO error (e.g. `icacls` absent), or a non-success exit status all
/// mean we could not prove the directory is restricted to us, so we return
/// `false` and [`DiskPtxCache::open`] disables the disk cache (in-memory
/// fallback). A host without `icacls` therefore loses the disk cache rather
/// than silently launching PTX out of a directory we couldn't lock down.
#[cfg(windows)]
fn harden_windows_dir(dir: &Path) -> bool {
    use std::process::{Command, Stdio};

    // Prefer the domain-qualified principal `DOMAIN\USER`; fall back to a
    // bare username. Without a username we can't name a grantee, so we can't
    // lock the dir down -> untrusted.
    let user = match std::env::var("USERNAME") {
        Ok(u) if !u.is_empty() => match std::env::var("USERDOMAIN") {
            Ok(d) if !d.is_empty() => format!("{d}\\{u}"),
            _ => u,
        },
        _ => return false,
    };

    // Spawn icacls and require a success exit status. We pass the grant spec
    // as a single argument (no shell involved, so the `(OI)(CI)F` parens and
    // the `:` are passed literally to icacls).
    match Command::new("icacls")
        .arg(dir)
        .arg("/inheritance:r")
        .arg("/grant:r")
        .arg(format!("{user}:(OI)(CI)F"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
    {
        Ok(status) => status.success(),
        // icacls missing / spawn failure -> can't prove the dir is locked
        // down -> fail closed.
        Err(_) => false,
    }
}

/// Read a `u64` tuning knob from the environment, falling back to `default`
/// when the var is unset, empty, or unparseable.
///
/// Mirrors the lenient parsing the GPU memory pool uses for its
/// `CRATON_BOLT_POOL_*` knobs: a malformed value is ignored (default wins)
/// rather than being treated as an error, so a typo in an env var can never
/// crash a process or silently disable a cap in a surprising way.
fn env_u64(var: &str, default: u64) -> u64 {
    match std::env::var(var) {
        Ok(v) => v.trim().parse::<u64>().unwrap_or(default),
        Err(_) => default,
    }
}

/// Monotonically-increasing counter for tempfile-suffix
/// disambiguation. Wraps on `u64` overflow (astronomically unreachable
/// — at one rename per nanosecond that's ~584 years).
fn next_temp_suffix() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Format a 128-bit spec hash as a fixed-width 32-character lowercase
/// hex string. Suitable as a filename component on every supported
/// platform (no path separators, no shell metacharacters).
///
/// This is the canonical key shape callers should use when bridging
/// between the engine's `ModuleCacheKey` (a `(u64, u64)` pair) and the
/// disk cache's string key.
#[must_use]
pub fn hash_to_key(hi: u64, lo: u64) -> String {
    format!("{:016x}{:016x}", hi, lo)
}

/// Validate that `key` is a safe single path component (fixes V-3,
/// path traversal).
///
/// # Why this exists (V-3, HIGH — path traversal)
///
/// [`DiskPtxCache::entry_path`] historically interpolated the caller's
/// `key` straight into `root.join(format!("{key}.ptx"))` with **zero
/// sanitisation** (the old doc-comment even said so). But the key is
/// composed upstream by `pub(crate)` helpers
/// ([`disk_key`] here, `compose_disk_key` in
/// `exec::module_cache`) that contractually accept *arbitrary* `entry`
/// strings — and the entry symbol ultimately traces back to planner IR.
/// A `key` containing a path separator (`/` or `\`), a parent-dir
/// component (`..`), a NUL byte, or an absolute-looking prefix (a leading
/// `/`, or a Windows drive like `C:`) would let the composed path escape
/// the cache root. The blast radius is severe in both directions:
///
///   * **Arbitrary read** — `lookup` does `fs::read_to_string(path)` and
///     the returned text is assembled by `CudaModule::from_ptx` and
///     *launched as PTX*. A traversal key could exfiltrate an arbitrary
///     file's contents into a kernel.
///   * **Arbitrary write** — `store` does `fs::write` + `fs::rename`, so a
///     traversal key could clobber an arbitrary file the process can
///     write.
///
/// # Contract
///
/// We refuse to trust the caller and validate the key *independent of
/// caller trust* right where it becomes a filename. A key is accepted
/// **iff** it is non-empty and every byte is in the strict filename-safe
/// charset `^[0-9A-Za-z._-]+$`. That charset:
///
///   * contains no path separators (`/`, `\`), so the key can never name a
///     subdirectory or escape via a separator;
///   * contains no `:` (NTFS alternate-data-stream / drive-letter syntax
///     on Windows) — this is *why* the `exec::module_cache` family
///     prefixes use `__` rather than `::` as their separator (see V-3
///     note on `SCALAR_AGG_DISK_PREFIX`);
///   * contains no NUL or other control bytes;
///   * cannot equal `.` or `..` *as a whole component*, because while `.`
///     and `-` are allowed bytes, the produced on-disk name is always
///     `"{key}.ptx"` — never a bare `.`/`..` — AND we additionally reject
///     a key that is exactly `.` or `..` (or any all-dots run) below as
///     defence in depth.
///
/// All legitimate keys pass: the codegen salt (`cg1-v0.7.0` →
/// `[0-9A-Za-z.-]`), the family prefixes (`scalar_agg__` etc. →
/// `[a-z_]`), the entry symbols (`bolt_*` → `[a-z0-9_]`), and the 32-char
/// lowercase hex content hash.
///
/// On an invalid key the cache treats the operation as a **miss**
/// (`lookup` → `None`) or a **no-op** (`store` → silently does nothing):
/// the cache is best-effort, so we never panic and never fall back to an
/// unsanitised path.
#[must_use]
pub(crate) fn valid_key(key: &str) -> bool {
    if key.is_empty() {
        return false;
    }
    // Defence in depth: reject a key that is entirely dots (`.`, `..`,
    // `...`). None of these is a legitimate cache key, and `..` is the
    // canonical traversal token.
    if key.bytes().all(|b| b == b'.') {
        return false;
    }
    key.bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'.' || b == b'_' || b == b'-')
}

/// Compute the **key-bound** content-integrity digest of a PTX body
/// (V-7).
///
/// Returns a fixed-width 32-char lowercase hex string — the same shape as
/// [`hash_to_key`] — derived from `(key, body)` via two domain-separated
/// `DefaultHasher` instances packed into 128 bits. This deliberately
/// reuses the crate's existing `DefaultHasher`-based 128-bit hashing
/// convention (the same shape `exec::module_cache::hash128` and the
/// `ModuleCacheKey` use for content keys) so we pull in **no new
/// dependency** for a hash.
///
/// # Key binding
///
/// The cache `key` (the `<key>.ptx` filename stem) is mixed into the digest
/// — with a length prefix so `key`/`body` can't be ambiguously concatenated
/// — *before* the body bytes. This binds a stored entry to the exact
/// filename it was written under: a file whose contents are copied or
/// renamed to a different `<key>.ptx` will fail [`DiskPtxCache::lookup`]'s
/// comparison (which re-derives the digest from the *lookup* key) and miss,
/// so contents can't be replayed under another key.
///
/// # This is NOT the security boundary
///
/// This is **not** a cryptographic MAC and the hash strength is **not** the
/// integrity boundary. The header and body live in the same directory, so an
/// attacker who can write that directory can recompute and rewrite the
/// digest. The actual confidentiality/integrity boundary for V-7 is the
/// **directory ownership + permission trust check** enforced fail-closed in
/// [`DiskPtxCache::open`] (owner-only on Unix; ACL-locked-to-current-user on
/// Windows — an untrusted dir disables the disk layer entirely). This digest
/// only provides integrity against accidental corruption / partial writes,
/// is a tripwire against naive tampering, and — via the key binding above —
/// prevents cross-key content replay within an otherwise-trusted directory.
#[must_use]
fn body_digest(key: &str, body: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    // Domain bytes (0xD1 / 0xD2) distinct from the key-hash domains used in
    // `exec::module_cache` so a digest can never be confused with a key. The
    // `key` is fed in with a u64 length prefix so `(key, body)` can never be
    // re-parsed as a different `(key', body')` split (length-prefix framing).
    let mut hi = DefaultHasher::new();
    hi.write_u8(0xD1);
    hi.write_u64(key.len() as u64);
    hi.write(key.as_bytes());
    hi.write(body.as_bytes());

    let mut lo = DefaultHasher::new();
    lo.write_u8(0xD2);
    lo.write_u64(key.len() as u64);
    lo.write(key.as_bytes());
    lo.write(body.as_bytes());

    hash_to_key(hi.finish(), lo.finish())
}

/// Header-line prefix written ahead of the PTX body in every cache file
/// (V-7 content-integrity check).
///
/// Format of a v0.7-and-later cache file:
/// ```text
/// #bolt-ptx-cache v1 <32-char-hex-digest>\n
/// <ptx body bytes...>
/// ```
/// The header is a single `\n`-terminated line. On lookup we parse the
/// first line, recompute [`body_digest`] over everything after it, and
/// serve the body only if the digests match. The format is
/// **backward-tolerant**: a file whose first line is not a well-formed
/// header (e.g. an entry written by an older binary, which is just raw
/// PTX) is treated as a **miss** rather than served unchecked — the old
/// raw entry simply gets recompiled and rewritten in the new format.
const CACHE_HEADER_MAGIC: &str = "#bolt-ptx-cache v1 ";

/// Split a cache-file's contents into `(header_line, body)` (V-7).
///
/// The header is the first `\n`-terminated line; the body is everything
/// after that newline. Returns `None` when there is no newline at all
/// (a degenerate/old file with no header line) so the caller treats it as
/// a miss. The header line is returned WITHOUT its trailing `\n`; the body
/// is returned verbatim (so a round-trip of `store` → `lookup` reproduces
/// the original PTX byte-for-byte).
fn split_header(raw: &str) -> Option<(&str, &str)> {
    let nl = raw.find('\n')?;
    let header = &raw[..nl];
    let body = &raw[nl + 1..];
    Some((header, body))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
//
// The tests use ad-hoc temp directories under the OS tempdir. They do
// NOT touch the global `HANDLE` or `OVERRIDE_DIR` slots — each test
// constructs its own `DiskPtxCache` so the process-wide memoisation
// stays unset and other tests are not affected by ordering.
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Tests that mutate the global `OVERRIDE_DIR` or the process
    /// `BOLT_PTX_CACHE_DIR` env var must serialise — cargo runs tests
    /// in parallel by default and the global slot is a race.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Unique per-test subdirectory under `std::env::temp_dir()`. We
    /// roll our own counter rather than pulling in a `tempfile` crate
    /// dep purely for the test harness.
    fn fresh_tempdir(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let mut p = std::env::temp_dir();
        p.push(format!(
            "craton-bolt-disk-cache-test-{}-{}-{}",
            tag,
            std::process::id(),
            n,
        ));
        // Best-effort cleanup of any leftover directory from a
        // previous run with the same suffix (PID reuse is rare but
        // possible on long-running CI hosts).
        let _ = fs::remove_dir_all(&p);
        p
    }

    #[test]
    fn store_then_lookup_round_trips() {
        let dir = fresh_tempdir("rt");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        let key = hash_to_key(0xdead_beef_dead_beef, 0xcafe_babe_cafe_babe);
        let ptx = "// test ptx\n.version 7.0\n.target sm_70\n";
        cache.store(&key, ptx).expect("store");
        let got = cache.lookup(&key).expect("hit");
        assert_eq!(got, ptx);
        // Clean up.
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn lookup_on_empty_dir_is_none() {
        let dir = fresh_tempdir("miss");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        let key = hash_to_key(1, 2);
        assert!(cache.lookup(&key).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rename_is_atomic_no_partial_file() {
        // We don't actually race threads here (would be flaky on CI);
        // instead we assert the invariant that store() never leaves a
        // ".tmp." file behind on success.
        let dir = fresh_tempdir("atomic");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        let key = hash_to_key(0x1234, 0x5678);
        cache.store(&key, "x").expect("store");
        let mut found_tmp = false;
        for entry in fs::read_dir(&dir).expect("readdir") {
            let entry = entry.expect("dirent");
            let name = entry.file_name().into_string().unwrap_or_default();
            if name.contains(".tmp.") {
                found_tmp = true;
            }
        }
        assert!(!found_tmp, "tempfile leaked into cache directory");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn store_overwrites_existing_entry() {
        let dir = fresh_tempdir("overwrite");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        let key = hash_to_key(7, 8);
        cache.store(&key, "v1").expect("store v1");
        cache.store(&key, "v2").expect("store v2");
        assert_eq!(cache.lookup(&key).as_deref(), Some("v2"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_creates_nested_directories() {
        let mut dir = fresh_tempdir("nested");
        dir.push("deeply");
        dir.push("nested");
        dir.push("ptx");
        let cache = DiskPtxCache::open(dir.clone()).expect("open creates dirs");
        assert!(dir.is_dir(), "open() should create the target directory");
        // Round-trip works on the freshly created path.
        cache.store("abc", "ptx").expect("store");
        assert_eq!(cache.lookup("abc").as_deref(), Some("ptx"));
        // Clean up the whole tree.
        let mut top = dir.clone();
        top.pop();
        top.pop();
        top.pop();
        let _ = fs::remove_dir_all(&top);
    }

    #[test]
    fn hash_to_key_is_fixed_width_lowercase_hex() {
        let k = hash_to_key(0, 0);
        assert_eq!(k.len(), 32);
        assert!(k
            .chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
        let k2 = hash_to_key(u64::MAX, u64::MAX);
        assert_eq!(k2, "ffffffffffffffffffffffffffffffff");
    }

    #[test]
    fn corrupt_entry_falls_through_to_miss() {
        // Manually drop a non-UTF-8 byte sequence under a key and
        // assert lookup() returns None rather than panicking.
        let dir = fresh_tempdir("corrupt");
        fs::create_dir_all(&dir).expect("mkdir");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        let key = "bogus";
        let p = dir.join(format!("{key}.ptx"));
        fs::write(&p, [0xff, 0xfe, 0xfd]).expect("write garbage");
        assert!(cache.lookup(key).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn override_dir_takes_precedence_over_env() {
        // We use a *local* DiskPtxCache rather than poking the global
        // HANDLE, so this test stays independent of other tests'
        // ordering. The asserted contract: when both an env var path
        // and an override path are configured, the override wins.
        // (Verifying via `resolve_cache_dir` directly.)
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let env_dir = fresh_tempdir("env");
        let override_dir = fresh_tempdir("override");
        fs::create_dir_all(&env_dir).expect("mk env");
        fs::create_dir_all(&override_dir).expect("mk override");

        // Save + restore the override slot so we don't leak state into
        // sibling tests.
        let prev = override_slot().lock().clone();
        *override_slot().lock() = Some(override_dir.clone());
        // SAFETY: set_var is documented as unsound across threads on
        // Unix, but cargo test runs each #[test] on its own thread and
        // no other thread reads BOLT_PTX_CACHE_DIR in this binary.
        std::env::set_var(DISK_PTX_CACHE_ENV, env_dir.to_string_lossy().to_string());

        let resolved = resolve_cache_dir().expect("override path");
        assert_eq!(resolved, override_dir);

        // Restore env + override.
        std::env::remove_var(DISK_PTX_CACHE_ENV);
        *override_slot().lock() = prev;

        let _ = fs::remove_dir_all(&env_dir);
        let _ = fs::remove_dir_all(&override_dir);
    }

    #[test]
    fn env_var_unset_resolves_to_none() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // Save + restore.
        let prev_override = override_slot().lock().clone();
        let prev_env = std::env::var(DISK_PTX_CACHE_ENV).ok();
        *override_slot().lock() = None;
        std::env::remove_var(DISK_PTX_CACHE_ENV);

        assert!(resolve_cache_dir().is_none());

        *override_slot().lock() = prev_override;
        if let Some(v) = prev_env {
            std::env::set_var(DISK_PTX_CACHE_ENV, v);
        }
    }

    #[test]
    fn codegen_version_change_changes_disk_key() {
        // JIT-M1: for an identical spec + entry, a different
        // CODEGEN_VERSION must produce a different on-disk key so a
        // codegen change can never serve stale PTX. We can't mutate the
        // const, so we reconstruct the salt shape with a bumped version
        // and assert the keys differ in exactly the salt component.
        let entry = "kernel_main";
        let (hi, lo) = (0xabcd_ef01_2345_6789u64, 0x0011_2233_4455_6677u64);
        let k_now = disk_key(entry, hi, lo);
        let salt_now = codegen_salt();
        let salt_bumped = format!("cg{}-v{}", CODEGEN_VERSION + 1, env!("CARGO_PKG_VERSION"));
        let k_bumped = format!("{}-{}-{}", salt_bumped, entry, hash_to_key(hi, lo));
        assert_ne!(
            k_now, k_bumped,
            "bumping CODEGEN_VERSION must rotate the disk key"
        );
        assert!(
            k_now.starts_with(&salt_now),
            "disk key must begin with the codegen salt"
        );
    }

    #[test]
    fn disk_key_is_stable_for_same_inputs() {
        // Same spec + entry ⇒ byte-identical key (deterministic), so a
        // hit lands on the same .ptx file across processes.
        let a = disk_key("kernel_main", 1, 2);
        let b = disk_key("kernel_main", 1, 2);
        assert_eq!(a, b);
        // The spec-hash tail is the canonical 32-char hex digest.
        assert!(a.ends_with(&hash_to_key(1, 2)));
    }

    #[test]
    fn disk_key_domain_separates_entry_and_spec() {
        // Different entry symbols must not alias under the same key.
        assert_ne!(disk_key("entry_a", 1, 2), disk_key("entry_b", 1, 2));
        // Different spec hashes must not alias either.
        assert_ne!(disk_key("entry_a", 1, 2), disk_key("entry_a", 9, 9));
    }

    #[test]
    fn codegen_salt_includes_crate_version() {
        // The crate version is folded in as a cross-release guard.
        let salt = codegen_salt();
        assert!(
            salt.contains(env!("CARGO_PKG_VERSION")),
            "codegen salt must embed the crate version"
        );
        assert!(
            salt.contains(&format!("cg{}", CODEGEN_VERSION)),
            "codegen salt must embed CODEGEN_VERSION"
        );
        // And the salt is the leading component of every disk key.
        assert!(disk_key("e", 0, 0).starts_with(&salt));
    }

    /// Changing the crate version must rotate the disk key independently of
    /// `CODEGEN_VERSION` — the cross-release defense-in-depth guard. We
    /// can't mutate `CARGO_PKG_VERSION` at runtime, so we reconstruct the
    /// salt shape with a different version and assert it diverges from the
    /// live key for an otherwise-identical spec + entry.
    #[test]
    fn crate_version_change_changes_disk_key() {
        let entry = "kernel_main";
        let (hi, lo) = (0x0123_4567_89ab_cdefu64, 0xfedc_ba98_7654_3210u64);
        let k_now = disk_key(entry, hi, lo);
        // A hypothetical different release with the same CODEGEN_VERSION.
        let salt_other = format!("cg{}-v{}", CODEGEN_VERSION, "0.0.0-other");
        let k_other = format!("{}-{}-{}", salt_other, entry, hash_to_key(hi, lo));
        assert_ne!(
            k_now, k_other,
            "a different crate version must rotate the disk key"
        );
    }

    /// The optional build-time codegen fingerprint, when present, must
    /// further partition the salt: a salt built with a fingerprint differs
    /// from the same salt without one (and from one with a different
    /// fingerprint). This guards the defense-in-depth path that catches a
    /// forgotten `CODEGEN_VERSION` bump. We exercise the salt *shape*
    /// directly because `CODEGEN_FINGERPRINT` is fixed at compile time.
    #[test]
    fn codegen_fingerprint_partitions_the_salt() {
        // JIT-arch: the salt shape is `cgN-vX.Y.Z-<arch>[-fp<fp>]`. The
        // arch token (`arch_sm_70_isa_7.5` today) is folded in between the
        // crate version and the optional fingerprint.
        let arch = arch_salt_token();
        let base = format!(
            "cg{}-v{}-{}",
            CODEGEN_VERSION,
            env!("CARGO_PKG_VERSION"),
            arch
        );
        let with_fp_a = format!("{base}-fp{}", "aaaa1111");
        let with_fp_b = format!("{base}-fp{}", "bbbb2222");
        // A fingerprint changes the salt vs. no fingerprint...
        assert_ne!(base, with_fp_a);
        // ...and two different fingerprints don't collide.
        assert_ne!(with_fp_a, with_fp_b);

        // Whatever the compile-time fingerprint is (Some or None), the live
        // salt must be consistent with codegen_salt()'s documented shape.
        let live = codegen_salt();
        let cg_v = format!("cg{}-v{}", CODEGEN_VERSION, env!("CARGO_PKG_VERSION"));
        assert!(
            live.starts_with(&cg_v),
            "live salt must start with cgN-vX.Y.Z"
        );
        assert!(
            live.contains(&arch),
            "live salt must fold in the PTX target arch token"
        );
        match CODEGEN_FINGERPRINT {
            Some(fp) => assert_eq!(live, format!("{base}-fp{fp}")),
            None => assert_eq!(live, base),
        }
        // The salt remains a filename-safe key component regardless.
        assert!(
            valid_key(&disk_key("e", 0, 0)),
            "disk key (incl. salt) must stay filename-safe"
        );
    }

    /// JIT-arch: the PTX target arch token must be present in the salt (and
    /// therefore in every disk key) so cached kernels are partitioned by GPU
    /// architecture and can never be mis-routed across `.target`s.
    #[test]
    fn codegen_salt_folds_in_target_arch() {
        let salt = codegen_salt();
        assert!(
            salt.contains("arch_sm_70_isa_7.5"),
            "salt must embed the PTX target arch + ISA token, got: {salt}"
        );
        // The arch token keeps the key filename-safe.
        assert!(valid_key(&disk_key("e", 0, 0)));
    }

    #[test]
    fn empty_env_var_resolves_to_none() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev_override = override_slot().lock().clone();
        let prev_env = std::env::var(DISK_PTX_CACHE_ENV).ok();
        *override_slot().lock() = None;
        std::env::set_var(DISK_PTX_CACHE_ENV, "");

        assert!(resolve_cache_dir().is_none());

        std::env::remove_var(DISK_PTX_CACHE_ENV);
        *override_slot().lock() = prev_override;
        if let Some(v) = prev_env {
            std::env::set_var(DISK_PTX_CACHE_ENV, v);
        }
    }

    // -----------------------------------------------------------------
    // JIT-cache-bound: size / entry-count eviction.
    // -----------------------------------------------------------------

    /// Counts committed `*.ptx` entries (ignores tempfiles) in `dir`.
    fn count_ptx_entries(dir: &Path) -> usize {
        fs::read_dir(dir)
            .map(|rd| {
                rd.flatten()
                    .filter(|e| {
                        let n = e.file_name();
                        let n = n.to_string_lossy();
                        n.ends_with(".ptx") && !n.contains(".tmp.")
                    })
                    .count()
            })
            .unwrap_or(0)
    }

    /// With the entry-count cap set to 1, a second insert must evict the
    /// older entry so the directory never exceeds the bound. Serialised via
    /// `ENV_LOCK` because it mutates a process-wide env var.
    #[test]
    fn entry_count_cap_evicts_oldest() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev = std::env::var(DISK_PTX_CACHE_MAX_ENTRIES_ENV).ok();
        std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, "1");

        let dir = fresh_tempdir("evict_entries");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        cache.store(&hash_to_key(1, 1), "first").expect("store 1");
        cache.store(&hash_to_key(2, 2), "second").expect("store 2");

        assert_eq!(
            count_ptx_entries(&dir),
            1,
            "entry-count cap of 1 must keep exactly one entry after two inserts"
        );

        // Restore env + clean up.
        match prev {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// A cap of `0` disables eviction for that dimension; with BOTH caps at
    /// `0` the cache grows unbounded (no entry is ever evicted).
    #[test]
    fn zero_caps_disable_eviction() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev_b = std::env::var(DISK_PTX_CACHE_MAX_BYTES_ENV).ok();
        let prev_e = std::env::var(DISK_PTX_CACHE_MAX_ENTRIES_ENV).ok();
        std::env::set_var(DISK_PTX_CACHE_MAX_BYTES_ENV, "0");
        std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, "0");

        let dir = fresh_tempdir("evict_off");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        for i in 0..5u64 {
            cache.store(&hash_to_key(i, i), "ptx").expect("store");
        }
        assert_eq!(count_ptx_entries(&dir), 5, "zero caps must not evict");

        match prev_b {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_MAX_BYTES_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_MAX_BYTES_ENV),
        }
        match prev_e {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// Total bytes of committed `*.ptx` entries (ignoring tempfiles) in `dir`.
    fn total_ptx_bytes(dir: &Path) -> u64 {
        fs::read_dir(dir)
            .map(|rd| {
                rd.flatten()
                    .filter(|e| {
                        let n = e.file_name();
                        let n = n.to_string_lossy();
                        n.ends_with(".ptx") && !n.contains(".tmp.")
                    })
                    .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Over the BYTE cap, `enforce_bounds` (run automatically after each
    /// `store`) evicts oldest-first until the directory is back under the cap.
    /// We assert the post-condition the policy guarantees portably: total
    /// bytes land at or below the cap and at least one entry was evicted. We do
    /// NOT assert *which* file survived here — mtime resolution is
    /// filesystem-dependent — that ordering claim is pinned by the
    /// `#[cfg(unix)]` mtime-controlled test below.
    #[test]
    fn byte_cap_evicts_until_under_cap() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev_b = std::env::var(DISK_PTX_CACHE_MAX_BYTES_ENV).ok();
        let prev_e = std::env::var(DISK_PTX_CACHE_MAX_ENTRIES_ENV).ok();
        // A tiny byte cap; entry-count cap disabled so only bytes drive it.
        std::env::set_var(DISK_PTX_CACHE_MAX_BYTES_ENV, "200");
        std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, "0");

        let dir = fresh_tempdir("evict_bytes");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        // Each stored file is header (~`#bolt-ptx-cache v1 ` + 32 hex + `\n`,
        // ~52 bytes) plus a 100-byte body => well over 100 bytes apiece, so a
        // handful of them blows past the 200-byte cap and forces eviction.
        let body: String = std::iter::repeat('x').take(100).collect();
        for i in 0..6u64 {
            cache.store(&hash_to_key(i, i), &body).expect("store");
        }

        let bytes = total_ptx_bytes(&dir);
        assert!(
            bytes <= 200,
            "byte cap of 200 must hold: {bytes} bytes remain after eviction"
        );
        assert!(
            count_ptx_entries(&dir) < 6,
            "at least one entry must have been evicted under the byte cap"
        );
        // Each entry (~152 B: a ~52 B header + 100 B body) fits under the
        // 200 B cap on its own, so the policy converges to a single most-
        // recently-written entry rather than emptying the directory.
        assert_eq!(
            count_ptx_entries(&dir),
            1,
            "a single ~152 B entry fits under the 200 B cap, so exactly one survives"
        );

        match prev_b {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_MAX_BYTES_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_MAX_BYTES_ENV),
        }
        match prev_e {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// Under BOTH caps (defaults are large), `enforce_bounds` is a no-op:
    /// every stored entry survives. We drive it through the public store path
    /// (which calls `enforce_bounds` after each insert) with generous caps.
    #[test]
    fn enforce_bounds_is_noop_under_cap() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev_b = std::env::var(DISK_PTX_CACHE_MAX_BYTES_ENV).ok();
        let prev_e = std::env::var(DISK_PTX_CACHE_MAX_ENTRIES_ENV).ok();
        std::env::set_var(DISK_PTX_CACHE_MAX_BYTES_ENV, "1000000");
        std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, "1000");

        let dir = fresh_tempdir("evict_noop");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        for i in 0..5u64 {
            cache.store(&hash_to_key(i, i), "small").expect("store");
        }
        assert_eq!(
            count_ptx_entries(&dir),
            5,
            "no entry may be evicted while comfortably under both caps"
        );
        // And every entry still round-trips (eviction did not touch them).
        for i in 0..5u64 {
            assert_eq!(
                cache.lookup(&hash_to_key(i, i)).as_deref(),
                Some("small"),
                "entry {i} must survive the no-op sweep"
            );
        }

        match prev_b {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_MAX_BYTES_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_MAX_BYTES_ENV),
        }
        match prev_e {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// A directory left OVER the entry-count cap by a previous process is
    /// trimmed back to exactly the cap on `open` (which calls
    /// `enforce_bounds` before serving). We pre-plant the files directly so
    /// the over-cap state exists before the handle is constructed.
    #[test]
    fn open_trims_oversized_dir_to_entry_cap() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev_b = std::env::var(DISK_PTX_CACHE_MAX_BYTES_ENV).ok();
        let prev_e = std::env::var(DISK_PTX_CACHE_MAX_ENTRIES_ENV).ok();
        std::env::set_var(DISK_PTX_CACHE_MAX_BYTES_ENV, "0"); // bytes disabled
        std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, "2");

        let dir = fresh_tempdir("evict_open");
        fs::create_dir_all(&dir).expect("mkdir");
        // Pre-plant 5 committed entries before any handle exists.
        for i in 0..5u64 {
            let key = hash_to_key(i, i);
            fs::write(dir.join(format!("{key}.ptx")), "x").expect("plant");
        }
        assert_eq!(count_ptx_entries(&dir), 5, "precondition: over-cap dir");

        // open() runs enforce_bounds() on an enabled handle, trimming to cap.
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        if cache.is_disk_enabled() {
            assert_eq!(
                count_ptx_entries(&dir),
                2,
                "open must trim an oversized dir down to the entry cap"
            );
        }
        // (On a platform where the trust check disables the disk layer,
        // enforce_bounds is intentionally skipped — see DiskPtxCache::open.)

        match prev_b {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_MAX_BYTES_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_MAX_BYTES_ENV),
        }
        match prev_e {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// LRU-by-mtime ordering (Unix only): with the oldest entries' mtimes
    /// pushed into the past via `libc::utimes`, `enforce_bounds` must evict
    /// the OLDEST-mtime entries first, keeping the newest under the cap. We
    /// gate to `cfg(unix)` because std offers no portable set-mtime API and
    /// `libc` (already a crate dep) gives us `utimes` only on Unix; the
    /// portable byte/entry-cap tests above cover the rest.
    #[cfg(unix)]
    #[test]
    fn lru_by_mtime_evicts_oldest_first_unix() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev_b = std::env::var(DISK_PTX_CACHE_MAX_BYTES_ENV).ok();
        let prev_e = std::env::var(DISK_PTX_CACHE_MAX_ENTRIES_ENV).ok();
        std::env::set_var(DISK_PTX_CACHE_MAX_BYTES_ENV, "0"); // bytes disabled
        std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, "2");

        let dir = fresh_tempdir("evict_lru");
        fs::create_dir_all(&dir).expect("mkdir");

        // Plant 4 entries and stamp each with a strictly increasing mtime so
        // their LRU order is unambiguous regardless of filesystem timestamp
        // resolution. keys[0] is oldest, keys[3] is newest.
        let keys: Vec<String> = (0..4u64).map(|i| hash_to_key(i, 100 + i)).collect();
        for (rank, key) in keys.iter().enumerate() {
            let p = dir.join(format!("{key}.ptx"));
            fs::write(&p, "x").expect("plant");
            set_mtime_secs_unix(&p, 1_000_000 + rank as i64 * 100);
        }
        assert_eq!(count_ptx_entries(&dir), 4, "precondition: 4 entries");

        // open -> enforce_bounds trims to the newest 2 (highest mtime).
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        if cache.is_disk_enabled() {
            assert_eq!(count_ptx_entries(&dir), 2, "entry cap of 2 after trim");
            // The two OLDEST (keys[0], keys[1]) must be gone; the two NEWEST
            // (keys[2], keys[3]) must remain.
            assert!(
                !dir.join(format!("{}.ptx", keys[0])).exists(),
                "oldest-mtime entry must be evicted first"
            );
            assert!(
                !dir.join(format!("{}.ptx", keys[1])).exists(),
                "second-oldest-mtime entry must be evicted"
            );
            assert!(
                dir.join(format!("{}.ptx", keys[2])).exists(),
                "newer entry must survive"
            );
            assert!(
                dir.join(format!("{}.ptx", keys[3])).exists(),
                "newest entry must survive"
            );
        }

        match prev_b {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_MAX_BYTES_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_MAX_BYTES_ENV),
        }
        match prev_e {
            Some(v) => std::env::set_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV, v),
            None => std::env::remove_var(DISK_PTX_CACHE_MAX_ENTRIES_ENV),
        }
        let _ = fs::remove_dir_all(&dir);
    }

    /// Set a file's atime+mtime to `secs` since the Unix epoch via
    /// `libc::utimes`. Test-only helper (Unix) so the LRU-by-mtime test can
    /// pin a deterministic ordering without depending on wall-clock spacing
    /// or filesystem timestamp granularity. `libc` is already a crate dep.
    #[cfg(unix)]
    fn set_mtime_secs_unix(path: &Path, secs: i64) {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;
        let c = CString::new(path.as_os_str().as_bytes()).expect("path has no NUL");
        let tv = libc::timeval {
            tv_sec: secs as libc::time_t,
            tv_usec: 0,
        };
        let times = [tv, tv];
        // SAFETY: `c` is a valid NUL-terminated path and `times` is a valid
        // 2-element array of `timeval`, matching the `utimes(2)` contract.
        let rc = unsafe { libc::utimes(c.as_ptr(), times.as_ptr()) };
        assert_eq!(rc, 0, "utimes must succeed on a just-written temp file");
    }

    // -----------------------------------------------------------------
    // V-3: path-traversal hardening.
    // -----------------------------------------------------------------

    /// `valid_key` must reject every shape that could escape the cache
    /// root and accept the legitimate filename-safe shapes.
    #[test]
    fn valid_key_rejects_traversal_and_accepts_safe() {
        // Traversal / separator / absolute / NUL / empty — all rejected.
        for bad in [
            "",
            ".",
            "..",
            "...",
            "../../evil",
            "a/b",
            "a\\b",
            "/etc/passwd",
            "C:\\windows\\system32",
            "a:b",           // Windows ADS / drive separator
            "a\0b",          // embedded NUL
            "foo bar",       // space
            "scalar_agg::x", // the old `::` separator must NOT pass
        ] {
            assert!(!valid_key(bad), "key must be rejected: {bad:?}");
        }
        // Legitimate shapes — accepted.
        for good in [
            "abc",
            "deadbeef",
            "cg1-v0.7.0-scalar_agg__bolt_reduce-00112233445566778899aabbccddeeff",
            hash_to_key(0xdead_beef, 0xcafe_babe).as_str(),
            "a.b_c-d",
        ] {
            assert!(valid_key(good), "key must be accepted: {good:?}");
        }
    }

    /// Exhaustive path-traversal / injection table for `valid_key`. This is
    /// the single gate in front of *every* filesystem touch the cache makes
    /// (`entry_path` → `lookup`/`store`), so we are deliberately broad about
    /// the shapes an attacker-influenced `entry` symbol could take. Each
    /// rejected key is one that — if interpolated into `root.join("{key}.ptx")`
    /// — could escape the cache root or name something other than a plain file
    /// directly under it.
    #[test]
    fn valid_key_traversal_table_is_exhaustive() {
        // (key, human-readable reason) so a failure points at the class.
        let rejected: &[(&str, &str)] = &[
            ("", "empty"),
            (".", "current-dir component"),
            ("..", "parent-dir traversal token"),
            ("...", "all-dots run"),
            ("....", "longer all-dots run"),
            // Forward-slash separators (POSIX traversal).
            ("a/b", "forward slash"),
            ("/", "bare root slash"),
            ("/abs", "absolute path marker"),
            ("/etc/passwd", "absolute POSIX path"),
            ("../../evil", "relative parent escape"),
            ("a/../b", "interior parent escape"),
            ("foo/", "trailing slash"),
            ("./foo", "leading dot-slash"),
            // Back-slash separators (Windows traversal).
            ("a\\b", "back slash"),
            ("\\", "bare backslash"),
            ("..\\..\\evil", "windows parent escape"),
            ("C:\\windows\\system32", "windows absolute drive path"),
            ("\\\\server\\share", "UNC path"),
            // Drive-letter / NTFS alternate-data-stream colon.
            ("a:b", "colon (drive / ADS separator)"),
            ("C:", "bare drive letter"),
            ("key:$DATA", "NTFS ADS stream syntax"),
            // NUL and other control bytes (filesystem-injection).
            ("a\0b", "embedded NUL"),
            ("\0", "bare NUL"),
            ("a\nb", "embedded newline"),
            ("a\tb", "embedded tab"),
            ("a\rb", "embedded carriage return"),
            // Whitespace and shell/glob metacharacters outside the charset.
            ("foo bar", "space"),
            (" leading", "leading space"),
            ("trailing ", "trailing space"),
            ("a*b", "glob star"),
            ("a?b", "glob question mark"),
            ("a|b", "pipe"),
            ("a;b", "semicolon"),
            ("$VAR", "dollar / shell var"),
            ("a%b", "percent"),
            ("a&b", "ampersand"),
            ("a'b", "single quote"),
            ("a\"b", "double quote"),
            ("a`b", "backtick"),
            ("~", "tilde / home"),
            ("héllo", "non-ASCII byte"),
            // The crate's own former separator must never slip through.
            ("scalar_agg::x", "old `::` separator"),
        ];
        for (bad, why) in rejected {
            assert!(!valid_key(bad), "key {bad:?} must be rejected ({why})");
            // The gate must also deny it a filesystem path everywhere it is
            // consulted: a fresh cache yields no entry_path / lookup / store.
            let dir = fresh_tempdir("vk_table");
            let cache = DiskPtxCache::open(dir.clone()).expect("open");
            assert!(
                cache.entry_path(bad).is_none(),
                "entry_path({bad:?}) must be None ({why})"
            );
            assert!(
                cache.lookup(bad).is_none(),
                "lookup({bad:?}) must miss ({why})"
            );
            assert!(
                cache.store(bad, "payload").is_ok(),
                "store({bad:?}) must be an Ok no-op ({why})"
            );
            let _ = fs::remove_dir_all(&dir);
        }

        // Accepted: the full filename-safe charset `[0-9A-Za-z._-]+`, plus the
        // real key shapes the engine composes.
        let accepted: &[&str] = &[
            "a",
            "0",
            "A",
            "_",
            "-",
            "a.b",      // interior dot is fine — it is not a whole `.`/`..`
            "..a",      // dots are allowed bytes; only an ALL-dots run is denied
            "a..",
            "a..b",
            "abc",
            "deadbeef",
            "DEADBEEF",
            "a.b_c-d",
            "cg2-v0.7.0-arch_sm_70_isa_7.5-scalar_agg__bolt_reduce-00112233445566778899aabbccddeeff",
        ];
        for good in accepted {
            assert!(valid_key(good), "key {good:?} must be accepted");
        }

        // Hex content keys and full composed disk keys always pass the gate.
        assert!(valid_key(hash_to_key(0, 0).as_str()));
        assert!(valid_key(hash_to_key(u64::MAX, u64::MAX).as_str()));
        assert!(valid_key(&disk_key(
            "bolt_reduce",
            0xdead_beef,
            0xcafe_babe
        )));
    }

    /// `valid_key` accepts a long-but-legitimate key (a real composed disk
    /// key is ~80+ chars) and the charset gate itself imposes no length
    /// ceiling — the contract is charset-based, not length-based, so we pin
    /// that behaviour explicitly rather than asserting a cap the code does
    /// not enforce.
    #[test]
    fn valid_key_length_behavior() {
        // An empty key is the one length that is rejected.
        assert!(!valid_key(""), "empty key must be rejected");
        // A single safe char is the minimum accepted key.
        assert!(valid_key("a"));
        // A very long all-safe-charset key is still accepted (no length cap).
        let long: String = std::iter::repeat('a').take(4096).collect();
        assert!(valid_key(&long), "a long all-safe key must be accepted");
        // A long key that is ALL dots is still rejected (traversal guard wins
        // regardless of length).
        let long_dots: String = std::iter::repeat('.').take(64).collect();
        assert!(
            !valid_key(&long_dots),
            "an all-dots run must be rejected at any length"
        );
    }

    /// A traversal key must produce no on-disk path: `lookup` misses and
    /// `store` is a no-op that writes nothing anywhere (not under the root,
    /// and crucially not outside it).
    #[test]
    fn traversal_key_is_lookup_miss_and_store_noop() {
        let dir = fresh_tempdir("traversal");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");

        for bad in ["../../evil", "a/b", "a\\b", "..", "/abs"] {
            // Lookup must be a miss, never an out-of-root read.
            assert!(cache.lookup(bad).is_none(), "lookup must miss for {bad:?}");
            // Store must be a silent no-op (Ok, but nothing written).
            assert!(
                cache.store(bad, "payload").is_ok(),
                "store must be Ok no-op for {bad:?}"
            );
        }
        // Nothing — no .ptx, no .tmp — landed in the cache root.
        let mut any = false;
        for entry in fs::read_dir(&dir).expect("readdir") {
            let _ = entry.expect("dirent");
            any = true;
        }
        assert!(
            !any,
            "traversal store must not create any file in the cache root"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// A normal hex key still round-trips after the validation gate.
    #[test]
    fn normal_hex_key_still_round_trips_after_validation() {
        let dir = fresh_tempdir("validrt");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        let key = hash_to_key(0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210);
        let ptx = ".version 7.0\n.target sm_70\n";
        cache.store(&key, ptx).expect("store");
        assert_eq!(cache.lookup(&key).as_deref(), Some(ptx));
        let _ = fs::remove_dir_all(&dir);
    }

    // -----------------------------------------------------------------
    // V-7: content-integrity check + header round-trip.
    // -----------------------------------------------------------------

    /// Round-trip: a stored entry carries the integrity header on disk and
    /// `lookup` returns the exact body (header stripped).
    #[test]
    fn store_writes_header_and_lookup_strips_it() {
        let dir = fresh_tempdir("v7header");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        let key = hash_to_key(1, 1);
        let ptx = "// body\n.version 7.0\n";
        cache.store(&key, ptx).expect("store");

        // On disk the file begins with the magic header line.
        let on_disk = fs::read_to_string(dir.join(format!("{key}.ptx"))).expect("read raw");
        assert!(
            on_disk.starts_with(CACHE_HEADER_MAGIC),
            "cache file must begin with the integrity header"
        );
        // lookup hands back the body verbatim, header stripped.
        assert_eq!(cache.lookup(&key).as_deref(), Some(ptx));
        let _ = fs::remove_dir_all(&dir);
    }

    /// A tampered body (digest no longer matches the header) must be a
    /// miss so the caller recompiles rather than launching altered PTX.
    #[test]
    fn tampered_body_is_a_miss() {
        let dir = fresh_tempdir("v7tamper");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        let key = hash_to_key(2, 2);
        cache.store(&key, "original").expect("store");

        // Rewrite the file keeping the (now-stale) header line but
        // swapping the body. The recomputed digest won't match.
        let p = dir.join(format!("{key}.ptx"));
        let raw = fs::read_to_string(&p).expect("read");
        let header_line = raw.split('\n').next().expect("header");
        fs::write(&p, format!("{header_line}\nEVIL PAYLOAD")).expect("tamper");

        assert!(cache.lookup(&key).is_none(), "tampered body must miss");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A legacy/headerless raw-PTX file (as an older binary would have
    /// written) is treated as a miss rather than served unchecked.
    #[test]
    fn headerless_legacy_entry_is_a_miss() {
        let dir = fresh_tempdir("v7legacy");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        let key = hash_to_key(3, 3);
        // No header, no newline at all -> split_header returns None -> miss.
        fs::write(
            dir.join(format!("{key}.ptx")),
            "raw legacy ptx with no header",
        )
        .expect("write");
        assert!(cache.lookup(&key).is_none(), "headerless entry must miss");
        let _ = fs::remove_dir_all(&dir);
    }

    /// `body_digest` is deterministic and content-sensitive: identical
    /// `(key, body)` pairs hash equal, different bodies (almost surely)
    /// differ, and — crucially for V-7 — the SAME body under a DIFFERENT key
    /// produces a different digest (key binding prevents cross-key replay).
    #[test]
    fn body_digest_is_deterministic_and_content_sensitive() {
        let k = "key0";
        assert_eq!(body_digest(k, "abc"), body_digest(k, "abc"));
        assert_ne!(body_digest(k, "abc"), body_digest(k, "abd"));
        assert_eq!(body_digest(k, "abc").len(), 32);
        // Key binding: identical body, different key -> different digest.
        assert_ne!(body_digest("key0", "abc"), body_digest("key1", "abc"));
    }

    /// V-7 key binding, end-to-end: an entry's bytes copied verbatim to a
    /// DIFFERENT `<key>.ptx` filename must miss, because `lookup` re-derives
    /// the digest from the *lookup* key and it won't match the header the
    /// bytes were written with. This is the cross-key content-replay guard.
    #[test]
    fn replayed_content_under_different_key_is_a_miss() {
        let dir = fresh_tempdir("v7replay");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        let key_a = hash_to_key(0x1111, 0x2222);
        let key_b = hash_to_key(0x3333, 0x4444);
        let ptx = "// body\n.version 7.0\n";
        cache.store(&key_a, ptx).expect("store under key_a");

        // Copy key_a's file bytes verbatim to key_b's filename.
        let raw = fs::read_to_string(dir.join(format!("{key_a}.ptx"))).expect("read a");
        fs::write(dir.join(format!("{key_b}.ptx")), &raw).expect("plant under b");

        // key_a still hits; the replayed key_b file is rejected (digest was
        // bound to key_a, not key_b).
        assert_eq!(cache.lookup(&key_a).as_deref(), Some(ptx));
        assert!(
            cache.lookup(&key_b).is_none(),
            "content replayed under a different key must miss"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// V-7 fail-closed: a disk-disabled handle is a total disk no-op —
    /// `store` writes nothing and `lookup` always misses — yet returns the
    /// best-effort `Ok`/`None` the caller expects, so the engine keeps
    /// running on the in-memory + codegen path.
    #[test]
    fn disabled_handle_is_a_disk_noop() {
        let dir = fresh_tempdir("v7disabled");
        fs::create_dir_all(&dir).expect("mkdir");
        let cache = DiskPtxCache::disabled_for_test(dir.clone());
        assert!(!cache.is_disk_enabled());
        let key = hash_to_key(9, 9);
        // store is a silent Ok no-op...
        cache
            .store(&key, "payload")
            .expect("store must be Ok no-op");
        // ...nothing landed on disk...
        assert!(
            !dir.join(format!("{key}.ptx")).exists(),
            "disabled store must not write any file"
        );
        // ...and lookup always misses, even if a file were present.
        assert!(cache.lookup(&key).is_none(), "disabled lookup must miss");
        let _ = fs::remove_dir_all(&dir);
    }

    /// V-7 (Unix only): a freshly-opened cache root is owner-only `0o700`
    /// AND passes the fail-closed trust check (owned by us, no group/world
    /// write), so the disk layer is enabled. Gated to `cfg(unix)` because
    /// Windows has no portable std equivalent (see [`DiskPtxCache::open`]).
    #[cfg(unix)]
    #[test]
    fn open_sets_owner_only_perms_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let dir = fresh_tempdir("v7perms");
        let cache = DiskPtxCache::open(dir.clone()).expect("open");
        let mode = fs::metadata(&dir).expect("metadata").permissions().mode();
        // Compare the low 9 permission bits; the file-type bits above are
        // irrelevant here.
        assert_eq!(mode & 0o777, 0o700, "cache root must be owner-only (0o700)");
        // The owner-private dir must be trusted, so the disk layer is live.
        assert!(
            cache.is_disk_enabled(),
            "an owner-only 0o700 dir we own must pass the V-7 trust check"
        );
        assert!(
            dir_is_trusted_unix(&dir),
            "trust check must accept 0o700 owner dir"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    /// V-7 (Unix only): a group/world-writable cache root FAILS the trust
    /// check, so `open` returns a disk-disabled handle (fail-closed) instead
    /// of using the untrusted directory.
    #[cfg(unix)]
    #[test]
    fn group_world_writable_dir_disables_disk_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let dir = fresh_tempdir("v7ww");
        let cache = DiskPtxCache::open(dir.clone()).expect("open must not hard-error");
        // Loosen the perms AFTER open's chmod to simulate a pre-existing
        // world-writable dir, then re-run the trust check directly.
        fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).expect("chmod 0777");
        assert!(
            !dir_is_trusted_unix(&dir),
            "a world-writable dir must fail the V-7 trust check"
        );
        // (The handle from the earlier open is still enabled because it was
        // verified at 0o700; the contract under test is the trust predicate
        // that `open` consults.)
        let _ = cache;
        let _ = fs::remove_dir_all(&dir);
    }

    /// V-7 (Windows only): `open` must still create the directory and a
    /// store→lookup round-trip must work after the best-effort `icacls`
    /// ACL-hardening pass. The hardening itself is best-effort and hard to
    /// assert on portably (it depends on host ACLs / `icacls` presence), so
    /// the contract under test is "hardening never breaks the cache".
    #[cfg(windows)]
    #[test]
    fn open_hardens_and_round_trips_on_windows() {
        let dir = fresh_tempdir("winacl");
        let cache = DiskPtxCache::open(dir.clone()).expect("open must succeed after icacls pass");
        assert!(dir.is_dir(), "open() must create the cache root");
        let key = hash_to_key(0xfeed_face_dead_beef, 0x0bad_cafe_f00d_d00d);
        let ptx = "// win ptx\n.version 7.0\n.target sm_70\n";
        cache.store(&key, ptx).expect("store");
        assert_eq!(
            cache.lookup(&key).as_deref(),
            Some(ptx),
            "round-trip after hardening"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
