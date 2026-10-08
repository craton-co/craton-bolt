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
    fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()))
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

/// Collect production/runtime Rust inputs that can read environment variables.
fn collect_env_source_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_rs_files(&root.join("src"), &mut files);
    collect_rs_files(&root.join("benches"), &mut files);
    let build = root.join("build.rs");
    if build.is_file() {
        files.push(build);
    }
    files
}

/// (2) Every `CRATON_*` / `BOLT_*` env-var name literal in production,
/// build-time, or benchmark Rust must be documented verbatim in
/// `docs/ENV_VARS.md` (or be on the allowlist).
#[test]
fn env_var_doc_parity() {
    let root = repo_root();
    let rs_files = collect_env_source_files(&root);
    assert!(
        !rs_files.is_empty(),
        "found no production/build/benchmark Rust files under {}",
        root.display()
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
        "the following env var name(s) appear as string literals in production, \
         build-time, or benchmark Rust but \
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

/// Scan `text` (Markdown) for backtick-delimited `CRATON_*` / `BOLT_*` tokens
/// that look like environment-variable names, returning each distinct name.
///
/// `docs/ENV_VARS.md` always wraps env-var names in backticks — both in the
/// quick-start matrix (`| \`CRATON_FOO\` | ... |`) and in the per-var section
/// headings (`### \`CRATON_FOO\``). We deliberately only consider backtick runs
/// (not bare prose) and apply the same uppercase/digit/underscore shape filter
/// as [`scan_env_var_literals`], so incidental prose can't produce a false
/// "documented" hit.
fn scan_env_var_doc_names(text: &str) -> BTreeSet<String> {
    let bytes = text.as_bytes();
    let mut found = BTreeSet::new();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'`' {
            let start = i + 1;
            let mut j = start;
            while j < bytes.len() && bytes[j] != b'`' && bytes[j] != b'\n' {
                j += 1;
            }
            if j < bytes.len() && bytes[j] == b'`' {
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

/// (4) Reverse-direction parity: every documented variable must have a real
/// production, build-time, or benchmark source location. This makes
/// `ENV_VARS.md` a partially generated inventory: code discovers the canonical
/// set and CI rejects both undocumented code knobs and stale documented knobs.
#[test]
fn env_var_reverse_parity() {
    let root = repo_root();
    let rs_files = collect_env_source_files(&root);

    let mut in_code: BTreeSet<String> = BTreeSet::new();
    for file in &rs_files {
        let text = read_to_string(file);
        in_code.extend(scan_env_var_literals(&text));
    }

    let docs = read_to_string(&root.join("docs").join("ENV_VARS.md"));
    let documented = scan_env_var_doc_names(&docs);

    let mut documented_only: Vec<String> = Vec::new();
    for var in &documented {
        if in_code.contains(var) {
            continue;
        }
        documented_only.push(var.clone());
    }

    assert!(
        documented_only.is_empty(),
        "docs/ENV_VARS.md contains stale env-var entries with no literal in \
         production, build-time, or benchmark Rust: {documented_only:?}"
    );
}

/// (5) The user-facing documentation index must include every first-class
/// engineering reference and the archived kernel-research directory.
#[test]
fn readme_documentation_index_covers_first_class_docs() {
    let root = repo_root();
    let readme = read_to_string(&root.join("README.md"));
    for required in [
        "docs/CUDARC_ADOPTION.md",
        "docs/CONTRIBUTING_KERNEL.md",
        "docs/SQL_REFERENCE.md",
        "docs/API_SURFACE.md",
        "docs/ENV_VARS.md",
        "kernels/",
    ] {
        assert!(
            readme.contains(required),
            "README.md project/documentation index is missing `{required}`"
        );
    }
}

/// (6) Root API additions that define execution placement must be reflected in
/// the checked-in API inventory in the same change.
#[test]
fn execution_tier_public_api_is_documented() {
    let root = repo_root();
    let api = read_to_string(&root.join("docs").join("API_SURFACE.md"));
    for required in [
        "`QueryHandle`",
        "`ExecutionTier`",
        "planned_execution_tier",
        "pub use exec::{Engine, EngineBuilder, QueryHandle}",
    ] {
        assert!(
            api.contains(required),
            "docs/API_SURFACE.md is missing public API token `{required}`"
        );
    }
}

/// (7) SQL support claims must publish the same three-tier vocabulary exposed
/// by the planner API, making placement drift reviewable and machine-checked.
#[test]
fn sql_reference_has_execution_tier_contract() {
    let root = repo_root();
    let sql = read_to_string(&root.join("docs").join("SQL_REFERENCE.md"));
    for required in [
        "Execution tier",
        "`Gpu`",
        "`Host`",
        "`Hybrid`",
        "planned_execution_tier",
    ] {
        assert!(
            sql.contains(required),
            "docs/SQL_REFERENCE.md is missing execution-tier marker `{required}`"
        );
    }
}

/// (8) Local CI must rehearse the high-value hosted commands exactly. Keep the
/// strings intentionally literal so a workflow edit requires the matching
/// local-script edit in the same review.
#[test]
fn local_and_hosted_ci_commands_stay_in_sync() {
    let root = repo_root();
    let workflow = read_to_string(&root.join(".github").join("workflows").join("ci.yml"));
    let local = read_to_string(&root.join("ci_local.sh"));
    for command in [
        "cargo clippy --lib --tests --features cuda-stub,storage --no-default-features -- -D warnings",
        "cargo clippy --lib --features cuda-stub --no-default-features -- -D warnings",
        "cargo test --lib --tests --features cuda-stub,storage --no-default-features",
        "cargo test --lib --tests --no-default-features --features cuda-stub,flight",
        "cargo build --examples --features cuda-stub --no-default-features",
        "cargo run --example quickstart --features cuda-stub --no-default-features",
        "cargo test --lib --tests --no-default-features --features cuda-stub,substrait",
        "--ignore-filename-regex 'src/cuda/'",
        "--fail-under-lines 50",
        "cargo deny --all-features check advisories licenses bans",
        "bash scripts/check_public_api.sh",
        "--features cudarc,storage -- --ignored --test-threads=1",
    ] {
        assert!(
            workflow.contains(command),
            ".github/workflows/ci.yml is missing canonical command `{command}`"
        );
        assert!(
            local.contains(command),
            "ci_local.sh is missing hosted-CI command `{command}`"
        );
    }
    assert!(
        !local.contains("curl | bash") && !local.contains("| bash"),
        "ci_local.sh must not execute network-fetched shell installers"
    );
}

/// (9) Every GitHub Action is immutable-SHA pinned, and the retired rust-cuda
/// downloader cannot re-enter the root build graph unnoticed.
#[test]
fn release_hardening_guards() {
    let root = repo_root();
    let workflows = root.join(".github").join("workflows");
    for entry in fs::read_dir(&workflows).expect("read workflows directory") {
        let path = entry.expect("workflow directory entry").path();
        if !matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("yml" | "yaml")
        ) {
            continue;
        }
        let workflow = read_to_string(&path);
        for line in workflow.lines().map(str::trim) {
            let Some(action) = line.strip_prefix("uses: ") else {
                continue;
            };
            let Some((_, revision)) = action.split_once('@') else {
                panic!("action has no revision in {}: `{line}`", path.display());
            };
            let sha = revision.split_whitespace().next().unwrap_or("");
            assert!(
                sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit()),
                "GitHub Action must be pinned to a 40-hex commit SHA in {}: `{line}`",
                path.display()
            );
        }
    }

    let manifest = read_to_string(&root.join("Cargo.toml"));
    let build = read_to_string(&root.join("build.rs"));
    assert!(!manifest.contains("cuda_builder"));
    assert!(!manifest.contains("rust-cuda ="));
    assert!(!build.contains("compile_rust_cuda_kernels"));
}

/// (10) The declared MSRV is a real, uniformly-claimed gate.
///
/// `rust-version` in `Cargo.toml` is the single source of truth. This test
/// pins every place that repeats it — the hosted CI matrix leg, the local CI
/// rehearsal, and the three user-facing docs that quote a minimum Rust
/// version — so a bump (or a stale claim) cannot survive review. The 1.74
/// value that shipped before this guard existed could not build the committed
/// lockfile at all, which is exactly the failure mode being locked out.
#[test]
fn msrv_is_consistent_across_repo() {
    let root = repo_root();
    let manifest = read_to_string(&root.join("Cargo.toml"));

    let msrv = manifest
        .lines()
        .map(str::trim)
        .find_map(|line| line.strip_prefix("rust-version"))
        .and_then(|rest| rest.trim_start().strip_prefix('='))
        .map(|rest| rest.trim().trim_matches('"').to_string())
        .expect("Cargo.toml declares `rust-version`");

    // The hosted matrix must build the declared MSRV, not some other version.
    let workflow = read_to_string(&root.join(".github").join("workflows").join("ci.yml"));
    assert!(
        workflow.contains(&format!("\"stable\", \"{msrv}\"")),
        ".github/workflows/ci.yml matrix must include the declared MSRV `{msrv}`"
    );

    // The local rehearsal must exercise the same toolchain. `rustup` needs a
    // fully-qualified patch version, so accept `X.Y` extended with `.0`.
    let local = read_to_string(&root.join("ci_local.sh"));
    let pinned = format!("{msrv}.0");
    assert!(
        local.contains(&format!("cargo +{pinned} check"))
            || local.contains(&format!("cargo +{msrv} check")),
        "ci_local.sh must run an MSRV check on `{msrv}`"
    );

    // Every user-facing claim must quote the same number.
    for doc in ["README.md", "docs/INSTALL.md", "docs/DEVELOPMENT.md"] {
        let text = read_to_string(&root.join(doc));
        assert!(
            text.contains(&format!("Rust {msrv}")) || text.contains(&format!("MSRV-{msrv}")),
            "{doc} must state the declared MSRV `{msrv}`"
        );
        // Guard against a stale second claim left behind by a partial bump.
        assert!(
            !text.contains("Rust 1.74"),
            "{doc} still claims the retired 1.74 MSRV"
        );
    }
}

// ---- generated documentation blocks ------------------------------------------
//
// Some documentation facts are pure functions of the repository (the module
// tree, the CI job list). Instead of hand-maintaining them — and letting them
// drift — the docs carry a marker-delimited block that these tests regenerate
// and compare. On a mismatch the test fails with the expected block; running
// the doc-consistency tests with `BOLT_REGENERATE_DOCS=1` rewrites the block
// in place instead.

/// Replace (or verify) the block between `<!-- {marker}:begin` and
/// `<!-- {marker}:end -->` in `doc` with `body`.
fn check_generated_block(doc: &Path, marker: &str, body: &str) {
    let text = read_to_string(doc);
    let begin_tag = format!("<!-- {marker}:begin");
    let end_tag = format!("<!-- {marker}:end -->");
    let begin = text
        .find(&begin_tag)
        .unwrap_or_else(|| panic!("{} has no `{begin_tag} ... -->` marker", doc.display()));
    let after_begin = begin
        + text[begin..]
            .find("-->")
            .unwrap_or_else(|| panic!("unterminated `{begin_tag}` marker"))
        + 3;
    let end = text
        .find(&end_tag)
        .unwrap_or_else(|| panic!("{} has no `{end_tag}` marker", doc.display()));
    assert!(
        end > after_begin,
        "{marker}: end marker precedes begin marker"
    );
    let current = text[after_begin..end].replace("\r\n", "\n");
    let expected = format!("\n{}\n", body.trim_end());
    if current == expected {
        return;
    }
    if std::env::var("BOLT_REGENERATE_DOCS").as_deref() == Ok("1") {
        let updated = format!("{}{}{}", &text[..after_begin], expected, &text[end..]);
        fs::write(doc, updated).expect("rewrite generated doc block");
        return;
    }
    panic!(
        "{} block `{marker}` is stale. Re-run with BOLT_REGENERATE_DOCS=1 to refresh it, \
         or apply:\n{expected}",
        doc.display()
    );
}

/// A Rust source file's module summary: the first sentence of the first
/// paragraph of its `//!` docs.
fn module_summary(path: &Path) -> Option<String> {
    let text = read_to_string(path);
    let mut paragraph: Vec<&str> = Vec::new();
    for line in text
        .lines()
        .map(|l| l.trim_start_matches('\u{feff}').trim())
    {
        match line.strip_prefix("//!").map(str::trim) {
            Some("") if paragraph.is_empty() => continue,
            Some("") => break,
            Some(t) => paragraph.push(t),
            None if paragraph.is_empty() => continue,
            None => break,
        }
    }
    if paragraph.is_empty() {
        return None;
    }
    let joined = paragraph.join(" ");
    let sentence = match joined.find(". ") {
        Some(i) => &joined[..=i],
        None => joined.as_str(),
    };
    Some(sentence.to_string())
}

/// (11) `docs/ARCHITECTURE.md` carries a generated inventory of every module
/// under `src/`, one row per file with its `//!` summary line. A new module
/// without a summary, or a stale inventory, fails here.
#[test]
fn architecture_module_inventory_is_current() {
    let root = repo_root();
    let mut files = Vec::new();
    collect_rs_files(&root.join("src"), &mut files);
    let mut rows: Vec<(String, String)> = files
        .iter()
        .map(|p| {
            let rel = p
                .strip_prefix(&root)
                .unwrap()
                .to_string_lossy()
                .replace('\\', "/");
            let summary = module_summary(p).unwrap_or_else(|| {
                panic!("{rel} has no `//!` module summary line; add one (it feeds docs/ARCHITECTURE.md)")
            });
            (rel, summary.replace('|', "\\|"))
        })
        .collect();
    rows.sort();
    let mut body = String::from("| Module | Summary |\n|---|---|\n");
    for (rel, summary) in rows {
        body.push_str(&format!("| `{rel}` | {summary} |\n"));
    }
    check_generated_block(
        &root.join("docs").join("ARCHITECTURE.md"),
        "module-inventory",
        &body,
    );
}

/// Parse `(trigger list, [(job id, name, runs-on)])` out of a GitHub
/// workflow with the line-oriented shape this repository uses (two-space
/// indentation; `name:` / `runs-on:` directly under each job).
fn workflow_jobs(text: &str) -> (Vec<String>, Vec<(String, String, String)>) {
    #[derive(PartialEq)]
    enum Section {
        Other,
        On,
        Jobs,
    }
    let mut section = Section::Other;
    let mut triggers = Vec::new();
    let mut jobs: Vec<(String, String, String)> = Vec::new();
    for line in text.lines() {
        if !line.starts_with(' ') && !line.trim().is_empty() && !line.starts_with('#') {
            section = match line.trim_end() {
                "on:" => Section::On,
                "jobs:" => Section::Jobs,
                _ => Section::Other,
            };
            continue;
        }
        let two = line.strip_prefix("  ").filter(|r| !r.starts_with(' '));
        let four = line.strip_prefix("    ").filter(|r| !r.starts_with(' '));
        match section {
            Section::On => {
                if let Some(key) = two.and_then(|r| r.split(':').next()) {
                    if !key.starts_with('#') && !key.is_empty() {
                        triggers.push(key.trim().to_string());
                    }
                }
            }
            Section::Jobs => {
                if let Some(r) = two {
                    if let Some(id) = r.trim_end().strip_suffix(':') {
                        if !id.starts_with('#') {
                            jobs.push((id.to_string(), String::new(), String::new()));
                        }
                    }
                } else if let (Some(r), Some(job)) = (four, jobs.last_mut()) {
                    if let Some(v) = r.strip_prefix("name:") {
                        job.1 = v.trim().to_string();
                    } else if let Some(v) = r.strip_prefix("runs-on:") {
                        job.2 = v.trim().to_string();
                    }
                }
            }
            Section::Other => {}
        }
    }
    (triggers, jobs)
}

/// (12) `docs/DEVELOPMENT.md` carries a generated CI matrix — every workflow,
/// its triggers, and every job with its runner — so the documented CI never
/// drifts from `.github/workflows/`.
#[test]
fn ci_matrix_doc_is_current() {
    let root = repo_root();
    let dir = root.join(".github").join("workflows");
    let mut workflows: Vec<PathBuf> = fs::read_dir(&dir)
        .expect("read workflows")
        .map(|e| e.expect("entry").path())
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()), Some("yml" | "yaml")))
        .collect();
    workflows.sort();
    let mut body =
        String::from("| Workflow | Triggers | Job | Name | Runs on |\n|---|---|---|---|---|\n");
    for wf in workflows {
        let file = wf.file_name().unwrap().to_string_lossy().to_string();
        let (triggers, jobs) = workflow_jobs(&read_to_string(&wf));
        assert!(!jobs.is_empty(), "{file}: no jobs parsed");
        let triggers = triggers.join(", ");
        for (id, name, runs_on) in jobs {
            assert!(!runs_on.is_empty(), "{file}: job `{id}` has no runs-on");
            body.push_str(&format!(
                "| `{file}` | {triggers} | `{id}` | {} | `{}` |\n",
                name.replace('|', "\\|"),
                runs_on.replace('|', "\\|")
            ));
        }
    }
    check_generated_block(
        &root.join("docs").join("DEVELOPMENT.md"),
        "ci-matrix",
        &body,
    );
}

/// (13) Living documentation must not carry obsolete applicability markers.
///
/// The crate version and MSRV live in `Cargo.toml`; documents that restate
/// them drift (a page that says "current as of v0.7" while describing 0.9
/// behaviour cannot be trusted line by line). Historical documents —
/// the changelog, migration guide, roadmap, release process, the path-to-1.0
/// plan, and benchmark history (which must cite the build it measured) — are
/// exempt.
#[test]
fn living_docs_have_no_version_applicability_markers() {
    let root = repo_root();
    let manifest = read_to_string(&root.join("Cargo.toml"));
    let version = cargo_package_version(&manifest);
    let exempt = [
        "CHANGELOG.md",
        "ROADMAP.md",
        "RELEASING.md",
        "docs/MIGRATION_GUIDE.md",
        "docs/PATH_TO_1.0.md",
        "docs/BENCHMARKS.md",
        "docs/GROUPBY_PERF.md",
    ];
    let mut docs: Vec<PathBuf> = vec![root.join("README.md"), root.join("CONTRIBUTING.md")];
    for entry in fs::read_dir(root.join("docs")).expect("read docs") {
        let p = entry.expect("entry").path();
        if p.extension().and_then(|e| e.to_str()) == Some("md") {
            docs.push(p);
        }
    }
    let mut problems = Vec::new();
    for doc in docs {
        let rel = doc
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/");
        if exempt.contains(&rel.as_str()) {
            continue;
        }
        for (n, line) in read_to_string(&doc).lines().enumerate() {
            let lower = line.to_ascii_lowercase();
            let mut flag =
                |why: &str| problems.push(format!("{rel}:{}: {why}: {}", n + 1, line.trim()));
            if lower.contains("rust 1.74") || lower.contains("msrv 1.74") {
                flag("retired MSRV");
            }
            if lower.contains("as of v0.") || lower.contains("as of 0.") {
                flag("version applicability marker");
            }
            if lower.contains("current as of") {
                flag("version applicability marker");
            }
            if lower.contains(" wave") && lower.contains("0.") {
                flag("milestone-wave marker");
            }
            // `craton-bolt = "x.y"` dependency snippets must name the current
            // major.minor.
            if let Some(rest) = line.trim_start().strip_prefix("craton-bolt = ") {
                let mm: String = version.split('.').take(2).collect::<Vec<_>>().join(".");
                if !rest.contains(&format!("\"{mm}")) {
                    flag("stale dependency snippet");
                }
            }
            // A `v`-prefixed crate version (`v0.6`, `v0.9.0`) restates the
            // version a living doc should not pin.
            let bytes = line.as_bytes();
            for i in 0..bytes.len().saturating_sub(3) {
                let boundary = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
                if boundary
                    && bytes[i] == b'v'
                    && bytes[i + 1] == b'0'
                    && bytes[i + 2] == b'.'
                    && bytes[i + 3].is_ascii_digit()
                {
                    flag("pinned crate version");
                    break;
                }
            }
        }
    }
    assert!(
        problems.is_empty(),
        "living docs restate versions/milestones (keep them in Cargo.toml / CHANGELOG.md):\n{}",
        problems.join("\n")
    );
}

/// (14) Every long-lived document carries the metadata header
/// (`applies-to:` / `owner:`) so a reader can tell what a page describes and
/// who keeps it current.
#[test]
fn docs_carry_metadata_headers() {
    let root = repo_root();
    for entry in fs::read_dir(root.join("docs")).expect("read docs") {
        let p = entry.expect("entry").path();
        if p.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let text = read_to_string(&p);
        let head: String = text.chars().take(800).collect();
        assert!(
            head.contains("applies-to:") && head.contains("owner:"),
            "{} is missing the `<!-- applies-to: ... owner: ... -->` metadata header",
            p.display()
        );
    }
}
