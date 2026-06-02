//! Host-only doc/code consistency guards.
//!
//! These tests lock down the documentation surfaces that code review flagged
//! as drift-prone: the crate version (`Cargo.toml` ↔ `CHANGELOG.md`) and the
//! set of runtime/build environment variables (`src/` ↔ `docs/ENV_VARS.md`).
//!
//! They are pure host-side file scans (no GPU, no CUDA), so they run and pass
//! under `cargo test --features cuda-stub --no-default-features`. Everything is
//! read with `std::fs` relative to `CARGO_MANIFEST_DIR` so the tests are
//! location-independent.
//!
//! The env-var scan deliberately *over-matches* `CRATON_*` / `BOLT_*` quoted
//! string literals in `src/`; the goal is documentation parity, not a precise
//! parse of every `std::env::var` call. A small explicit allowlist covers
//! identifiers that are deliberately not configuration knobs (platform path /
//! identity lookups, the compile-time codegen salt, and a test-only synthetic
//! var).

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// Repository root (the crate manifest directory).
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read_to_string(path: &Path) -> String {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
}

/// Extract the `[package]` `version = "X"` value from `Cargo.toml`.
///
/// Tolerant of whitespace around `=` and of other `version = ...` keys
/// elsewhere in the manifest (e.g. `rust-version`): we only accept a line
/// whose trimmed form *starts with* the bare key `version` after the
/// `[package]` header and before the next `[section]` header.
fn cargo_package_version(manifest: &str) -> String {
    let mut in_package = false;
    for raw in manifest.lines() {
        let line = raw.trim();
        if line.starts_with('[') {
            in_package = line == "[package]";
            continue;
        }
        if !in_package {
            continue;
        }
        // Match a key named exactly `version` (not `rust-version`, etc.).
        if let Some(rest) = line.strip_prefix("version") {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                let val = rest.trim().trim_matches('"');
                if !val.is_empty() {
                    return val.to_string();
                }
            }
        }
    }
    panic!("could not find `version = \"...\"` in the [package] section of Cargo.toml");
}

/// (1) The `[package]` version must appear verbatim in `CHANGELOG.md`.
#[test]
fn version_consistency() {
    let root = repo_root();
    let manifest = read_to_string(&root.join("Cargo.toml"));
    let version = cargo_package_version(&manifest);

    let changelog = read_to_string(&root.join("CHANGELOG.md"));
    assert!(
        changelog.contains(&version),
        "Cargo.toml [package] version `{version}` does not appear anywhere in \
         CHANGELOG.md. Add a changelog entry/section for this version (or fix \
         the version string) so the release notes and the crate version agree."
    );

    // Optional: if the README pins a version, it must match. We only assert
    // when the README plausibly references *some* `x.y.z`-shaped version near
    // the crate name, to avoid being brittle on prose that never pins one.
    let readme_path = root.join("README.md");
    if readme_path.exists() {
        let readme = read_to_string(&readme_path);
        // A README that mentions `craton-bolt = "x.y.z"` (a Cargo dependency
        // snippet) must pin the current version.
        if let Some(pinned) = readme_dependency_pin(&readme) {
            assert_eq!(
                pinned, version,
                "README.md pins `craton-bolt = \"{pinned}\"` but Cargo.toml is \
                 at `{version}`. Update the README dependency snippet."
            );
        }
    }
}

/// If the README contains a Cargo-dependency snippet of the form
/// `craton-bolt = "x.y.z"`, return the pinned version string. Returns `None`
/// when the README never pins the crate (the common case), so the optional
/// check stays non-brittle.
fn readme_dependency_pin(readme: &str) -> Option<String> {
    for line in readme.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("craton-bolt") {
            let rest = rest.trim_start();
            if let Some(rest) = rest.strip_prefix('=') {
                let val = rest.trim().trim_matches('"').trim();
                // Only accept a plain dotted version, not a richer table
                // dependency (`{ version = ..., features = ... }`).
                if !val.is_empty()
                    && val.chars().all(|c| c.is_ascii_digit() || c == '.')
                    && val.contains('.')
                {
                    return Some(val.to_string());
                }
            }
        }
    }
    None
}

/// Env-var names that are intentionally *not* documented in
/// `docs/ENV_VARS.md`, with a one-line justification each.
const ENV_VAR_ALLOWLIST: &[&str] = &[
    // Platform-default PTX-cache directory resolution — pure path lookups,
    // not configuration knobs (documented as omitted in ENV_VARS.md).
    "HOME",
    "LOCALAPPDATA",
    "USERPROFILE",
    "XDG_CACHE_HOME",
    // Windows user/domain identity, read only to compute a per-user cache
    // sub-path; not a configuration knob.
    "USERNAME",
    "USERDOMAIN",
    // Compile-time codegen salt consumed via `option_env!` in build.rs /
    // disk_cache.rs — a build fingerprint, never read at runtime.
    "BOLT_CODEGEN_FINGERPRINT",
    // Test-only synthetic var used inside a #[test] in jit_compiler.rs to
    // exercise the parse path; never read in production code.
    "CRATON_BOLT_PTX_CACHE_CAP_TEST_ENV",
];

/// Recursively collect `.rs` files under `dir`.
fn collect_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

/// Scan `text` for quoted string literals whose contents look like a
/// `CRATON_*` or `BOLT_*` environment-variable name, returning each distinct
/// name found. Deliberately tolerant: it matches any double-quoted run that
/// starts with one of those prefixes and is composed of uppercase ASCII,
/// digits, and underscores. Over-matching is fine — the test only asserts
/// documentation parity.
fn scan_env_var_literals(text: &str) -> BTreeSet<String> {
    let bytes = text.as_bytes();
    let mut found = BTreeSet::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'"' {
            // Read the literal contents up to the next unescaped quote.
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != b'"' && bytes[j] != b'\n' {
                // Stop on a backslash escape boundary; env-var names never
                // contain escapes, so a literal with one isn't a candidate.
                if bytes[j] == b'\\' {
                    break;
                }
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'"' {
                let content = &text[start..j];
                if (content.starts_with("CRATON_") || content.starts_with("BOLT_"))
                    && content
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
                {
                    found.insert(content.to_string());
                }
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    found
}

/// (2) Every `CRATON_*` / `BOLT_*` env-var name literal in `src/` must be
/// documented verbatim in `docs/ENV_VARS.md` (or be on the allowlist).
#[test]
fn env_var_doc_parity() {
    let root = repo_root();
    let src_dir = root.join("src");
    assert!(
        src_dir.is_dir(),
        "expected a src/ directory at {}",
        src_dir.display()
    );

    let mut rs_files = Vec::new();
    collect_rs_files(&src_dir, &mut rs_files);
    assert!(
        !rs_files.is_empty(),
        "found no .rs files under {}",
        src_dir.display()
    );

    let mut discovered: BTreeSet<String> = BTreeSet::new();
    for file in &rs_files {
        let text = read_to_string(file);
        discovered.extend(scan_env_var_literals(&text));
    }

    let docs = read_to_string(&root.join("docs").join("ENV_VARS.md"));

    let allow: BTreeSet<&str> = ENV_VAR_ALLOWLIST.iter().copied().collect();

    let mut missing: Vec<String> = Vec::new();
    for var in &discovered {
        if allow.contains(var.as_str()) {
            continue;
        }
        if !docs.contains(var) {
            missing.push(var.clone());
        }
    }

    assert!(
        missing.is_empty(),
        "the following env var name(s) appear as string literals in src/ but \
         are NOT documented in docs/ENV_VARS.md: {missing:?}. Either document \
         each in docs/ENV_VARS.md or, if it is deliberately not a configuration \
         knob, add it to ENV_VAR_ALLOWLIST in tests/doc_consistency_test.rs \
         with a one-line justification."
    );
}

/// (3) Best-effort: `CHANGELOG.md` must contain a section header referencing
/// the current `[package]` version, e.g. `## [0.7.0]` or `## 0.7.0`. This is
/// stricter than [`version_consistency`] (which only requires the string to
/// appear *somewhere*); it asserts a heading-shaped occurrence so the version
/// is a real release/section, not just an incidental mention.
#[test]
fn changelog_has_section_for_current_version() {
    let root = repo_root();
    let manifest = read_to_string(&root.join("Cargo.toml"));
    let version = cargo_package_version(&manifest);
    let changelog = read_to_string(&root.join("CHANGELOG.md"));

    let has_section = changelog.lines().any(|raw| {
        let line = raw.trim();
        if !line.starts_with('#') {
            return false;
        }
        // Drop the leading '#' run and surrounding whitespace, then look for
        // the version embedded in the heading (tolerant of `[x.y.z]` brackets
        // and a trailing ` - date`).
        let heading = line.trim_start_matches('#').trim();
        heading.contains(&version)
    });

    assert!(
        has_section,
        "CHANGELOG.md has no heading (`## ...`) referencing the current \
         Cargo.toml version `{version}`. Add a `## [{version}]` section so the \
         release has a changelog entry."
    );
}
