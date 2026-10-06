// SPDX-License-Identifier: Apache-2.0

//! Reproducibility manifest for benchmark runs.
//!
//! Every published benchmark number must be traceable to the exact build and
//! hardware that produced it (see `docs/BENCHMARKS.md`, "Reproducibility
//! manifest"). [`write`] captures that context once per bench process and
//! writes it as JSON next to Criterion's output, under
//! `target/bench-manifests/<bench>-<unix-seconds>.json`, and echoes the path
//! to stdout so a CI job can upload it with the Criterion report.
//!
//! Collection is best-effort and never fails the run: a field whose source is
//! unavailable (no `git`, no `nvidia-smi`, a host-only `cuda-stub` build)
//! is recorded as `null` rather than guessed.

use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// Run `cmd args...` and return trimmed stdout, or `None` on any failure.
fn capture(cmd: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(cmd).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!s.is_empty()).then_some(s)
}

fn json_str(v: &Option<String>) -> String {
    match v {
        Some(s) => format!(
            "\"{}\"",
            s.replace('\\', "\\\\")
                .replace('"', "\\\"")
                .replace('\n', " ")
        ),
        None => "null".to_string(),
    }
}

/// The active Cargo features this bench binary was compiled with.
fn features() -> Vec<&'static str> {
    let mut f = Vec::new();
    if cfg!(feature = "cudarc") {
        f.push("cudarc");
    }
    if cfg!(feature = "cuda-stub") {
        f.push("cuda-stub");
    }
    if cfg!(feature = "reference-benches") {
        f.push("reference-benches");
    }
    if cfg!(feature = "flight") {
        f.push("flight");
    }
    if cfg!(feature = "substrait") {
        f.push("substrait");
    }
    f
}

/// Describe one benchmark configuration axis for the manifest's `axes` map.
pub struct Axis {
    pub name: &'static str,
    pub value: String,
}

/// Write the manifest for bench `bench` and return its path (if written).
///
/// `axes` records the experimental axes this run sweeps (rows, batches,
/// cold/warm, transfer-included, correctness oracle, ...), so the
/// manifest alone says what was measured.
pub fn write(bench: &str, axes: &[Axis]) -> Option<PathBuf> {
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let commit = std::env::var("GITHUB_SHA")
        .ok()
        .or_else(|| capture("git", &["rev-parse", "HEAD"]));
    let dirty = capture("git", &["status", "--porcelain", "--untracked-files=no"])
        .map(|s| (!s.is_empty()).to_string());
    let rustc = capture(
        &std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into()),
        &["--version"],
    );
    // name, compute capability, total memory, driver, power limit.
    let gpu = capture(
        "nvidia-smi",
        &[
            "--query-gpu=name,compute_cap,memory.total,driver_version,power.limit",
            "--format=csv,noheader",
        ],
    )
    .and_then(|s| s.lines().next().map(str::to_string));
    let gpu_fields: Vec<Option<String>> = match &gpu {
        Some(line) => line
            .split(',')
            .map(|f| Some(f.trim().to_string()))
            .collect(),
        None => vec![None; 5],
    };
    let field = |i: usize| gpu_fields.get(i).cloned().flatten();
    let cuda_toolkit = capture("nvcc", &["--version"]).and_then(|s| {
        s.lines()
            .find(|l| l.contains("release"))
            .map(|l| l.trim().to_string())
    });
    let profile = if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    };
    let features = features()
        .iter()
        .map(|f| format!("\"{f}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let axes_json = axes
        .iter()
        .map(|a| format!("    \"{}\": {}", a.name, json_str(&Some(a.value.clone()))))
        .collect::<Vec<_>>()
        .join(",\n");

    let json = format!(
        "{{\n  \"bench\": \"{bench}\",\n  \"unix_time\": {unix},\n  \"commit\": {},\n  \
         \"worktree_dirty\": {},\n  \"crate_version\": \"{}\",\n  \"rustc\": {},\n  \
         \"profile\": \"{profile}\",\n  \"features\": [{features}],\n  \"gpu\": {{\n    \
         \"model\": {},\n    \"compute_capability\": {},\n    \"vram\": {}\n  }},\n  \
         \"driver\": {},\n  \"power_limit\": {},\n  \"cuda_toolkit\": {},\n  \
         \"os\": \"{}\",\n  \"axes\": {{\n{axes_json}\n  }}\n}}\n",
        json_str(&commit),
        dirty.unwrap_or_else(|| "null".into()),
        env!("CARGO_PKG_VERSION"),
        json_str(&rustc),
        json_str(&field(0)),
        json_str(&field(1)),
        json_str(&field(2)),
        json_str(&field(3)),
        json_str(&field(4)),
        json_str(&cuda_toolkit),
        std::env::consts::OS,
    );

    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("bench-manifests");
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("{bench}-{unix}.json"));
    std::fs::write(&path, json).ok()?;
    println!("bench manifest: {}", path.display());
    Some(path)
}
