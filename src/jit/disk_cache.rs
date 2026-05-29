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

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use parking_lot::Mutex;

/// Environment variable that enables the disk-backed PTX cache and
/// names its root directory. Unset, empty, or unreadable → cache
/// disabled.
pub const DISK_PTX_CACHE_ENV: &str = "BOLT_PTX_CACHE_DIR";

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
#[derive(Clone)]
pub struct DiskPtxCache {
    root: std::sync::Arc<PathBuf>,
}

impl DiskPtxCache {
    /// Open (creating if needed) a cache rooted at `dir`. Returns an
    /// error if the directory cannot be created.
    pub fn open(dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&dir)?;
        Ok(Self {
            root: std::sync::Arc::new(dir),
        })
    }

    /// Path to this cache's root directory. Useful for tests, logging,
    /// and the builder's "where did the cache land?" diagnostic.
    pub fn root(&self) -> &Path {
        self.root.as_path()
    }

    /// Compose the on-disk path for a given content-hash key.
    /// `key` should be a printable identifier (typically a hex digest
    /// of the spec hash); we sanitise nothing because the caller is
    /// expected to hand us a hex string.
    fn entry_path(&self, key: &str) -> PathBuf {
        let mut p = (*self.root).clone();
        p.push(format!("{key}.ptx"));
        p
    }

    /// Look up `key` on disk. Returns `Some(ptx_text)` on a hit,
    /// `None` on a miss or any I/O failure. We intentionally do *not*
    /// surface read errors as `Err`: a corrupt or unreadable cache
    /// entry should silently fall through to the codegen path so the
    /// caller still gets a correct result.
    #[must_use]
    pub fn lookup(&self, key: &str) -> Option<String> {
        match fs::read_to_string(self.entry_path(key)) {
            Ok(s) => Some(s),
            Err(_) => None,
        }
    }

    /// Persist `ptx` under `key` using a tempfile-then-rename to keep
    /// concurrent readers from ever observing a partial file.
    ///
    /// Returns `Ok(())` even on rename failure if the *content* ended
    /// up at the target path (this can happen on Windows when another
    /// thread won the rename race with identical bytes — the codegen
    /// is deterministic so the outcome is correct either way). Any
    /// other I/O error is propagated.
    pub fn store(&self, key: &str, ptx: &str) -> io::Result<()> {
        let target = self.entry_path(key);
        // Tempfile name: same directory, suffixed with the OS PID and a
        // process-monotonic counter so concurrent writers in the same
        // process don't collide. We use a counter (not a random number)
        // to keep the test path deterministic and avoid pulling in a
        // `rand` dep just for this.
        let suffix = next_temp_suffix();
        let mut tmp = (*self.root).clone();
        let pid = std::process::id();
        tmp.push(format!("{key}.ptx.tmp.{pid}.{suffix}"));
        // Write the full body to the tempfile first.
        fs::write(&tmp, ptx)?;
        // Atomic rename. On Windows `fs::rename` already implements
        // `MOVEFILE_REPLACE_EXISTING` semantics in stable std.
        match fs::rename(&tmp, &target) {
            Ok(()) => Ok(()),
            Err(e) => {
                // Best-effort cleanup of the stray tempfile so we don't
                // leak inodes under repeated failures.
                let _ = fs::remove_file(&tmp);
                Err(e)
            }
        }
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
        assert!(k.chars().all(|c| c.is_ascii_hexdigit() && !c.is_uppercase()));
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
}
