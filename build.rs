// SPDX-License-Identifier: Apache-2.0
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_STUB");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_RUST_CUDA");

    // --- Auto-rotating codegen fingerprint ------------------------------
    //
    // Hash the codegen-relevant source — the whole `src/jit/*.rs` tree
    // PLUS the `src/plan/physical_plan.rs` plan IR that codegen lowers
    // into PTX — into a stable hex digest and export it as
    // `BOLT_CODEGEN_FINGERPRINT`. The disk PTX cache
    // (src/jit/disk_cache.rs) reads it via `option_env!` and folds it
    // into the codegen salt, so ANY change to PTX-emitting source (or to
    // the plan shape it lowers) automatically rotates the on-disk cache
    // key — removing the manual `CODEGEN_VERSION` bump as a single point
    // of failure (it remains as a complementary in-release guard; the
    // fingerprint is additive).
    //
    // The digest is deterministic: files are hashed in sorted *relative
    // path* order and only file *bytes* (plus the bare relative path)
    // feed the hash — no timestamps, no absolute paths. If a directory or
    // file can't be read we emit nothing rather than panicking, so the
    // consumer falls back to `CODEGEN_VERSION` + crate version alone
    // (never weaker).
    emit_codegen_fingerprint();

    // --- Wave A: rust-cuda PTX generation -------------------------------
    //
    // When `--features rust-cuda` is on, compile the kernels/ crate to
    // PTX via cuda_builder (the rustc_codegen_nvvm front-end). The
    // resulting PTX is dropped at $OUT_DIR/partition.ptx and consumed
    // by src/jit/partition_kernel.rs via `include_str!`.
    //
    // When the feature is off, write an empty stub so the
    // `include_str!` site in partition_kernel.rs (also feature-gated)
    // doesn't fail to find the file. The host code under
    // `#[cfg(not(feature = "rust-cuda"))]` never reads it.
    //
    // See docs/JIT_PIPELINE.md for the rust-cuda build hook and stub pattern.
    compile_rust_cuda_kernels();

    // --- GRACEFUL DEGRADATION CONTRACT (GPU-less hosts) -----------------
    //
    // This build script NEVER hard-fails (panics / nonzero exit) because a
    // CUDA toolkit is absent. Two distinct cases are both made to succeed on
    // a GPU-less host so `cargo package`, docs.rs, and cuda-stub/type-check
    // CI all work:
    //
    //   1. cuda-stub feature ON  -> skip CUDA discovery entirely and return
    //      below. No `-L` search paths are needed because the stub does not
    //      link the real driver. (Engine::new returns Err at runtime.)
    //   2. real build, no toolkit -> CUDA discovery below finds no lib dir
    //      and emits actionable `cargo:warning=` lines, then RETURNS normally
    //      (see the `!cuda_lib_found` block). The build proceeds; only a real
    //      link step (which a GPU-less `cargo package`/`cargo check` does not
    //      reach) would later fail, with the warning already pointing at the
    //      cuda-stub escape hatch.
    //
    // The only `expect()`/panic paths in this file are gated behind
    // `--features rust-cuda` (cuda_builder) or read OUT_DIR (always set by
    // Cargo); none are reachable on a default/cuda-stub GPU-less build.
    //
    // Skip CUDA discovery when building with the `cuda-stub` feature
    // (e.g. on docs.rs or CUDA-less hosts).
    if std::env::var_os("CARGO_FEATURE_CUDA_STUB").is_some() {
        return;
    }

    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // Track whether any CUDA library search path was emitted below. A real
    // (non-stub) build links against `cuda.lib` / `libcuda.so`; if no lib
    // directory is discovered, the build will fail at LINK time with an
    // obscure "cannot find -lcuda" / "unresolved external symbol" error. We
    // surface a clear, actionable `cargo:warning=` up front so the failure is
    // self-explanatory and points at the CUDA-less escape hatch.
    let mut cuda_lib_found = false;

    // Check CUDA_PATH environment variable first.
    //
    // V-16: CUDA_PATH is a TRUSTED build-environment input. We use it verbatim
    // to add a linker search path (`-L`), which is the standard contract for
    // `*-sys` crates. Pointing CUDA_PATH at an attacker-controlled directory
    // is equivalent to build-host compromise (it can make the linker pick up
    // malicious import libraries), so we do not attempt to sandbox or validate
    // it beyond the exists() gating below — securing the build environment is a
    // prerequisite, not something build.rs can enforce.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let path = std::path::PathBuf::from(cuda_path);
        if cfg!(target_os = "windows") {
            let lib_path = path.join("lib").join("x64");
            if lib_path.exists() {
                println!("cargo:rustc-link-search=native={}", lib_path.display());
                cuda_lib_found = true;
            }
        } else {
            let lib64_path = path.join("lib64");
            let lib_path = path.join("lib");
            if lib64_path.exists() {
                println!("cargo:rustc-link-search=native={}", lib64_path.display());
                cuda_lib_found = true;
                let stubs_path = lib64_path.join("stubs");
                if stubs_path.exists() {
                    println!("cargo:rustc-link-search=native={}", stubs_path.display());
                }
            } else if lib_path.exists() {
                println!("cargo:rustc-link-search=native={}", lib_path.display());
                cuda_lib_found = true;
            }
        }
    } else {
        // Fallbacks if CUDA_PATH is not set.
        if cfg!(target_os = "windows") {
            // Check default installation directory structure.
            // On Windows, the standard path is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\lib\x64.
            // Collect all matching entries and sort by version descending so the
            // highest-version install (e.g. v12.6 beats v12.4 beats v11.8) wins.
            //
            // V-16: parse the "vMAJOR.MINOR" directory name into numeric
            // (major, minor) and sort by that, NOT lexicographically. A plain
            // string sort puts "v9.0" above "v12.0" (because '9' > '1'),
            // selecting an older CUDA. Directories that don't parse sort lowest
            // ((0, 0)) rather than panicking, preserving the exists()-gating
            // behavior below.
            fn parse_cuda_version(name: &std::ffi::OsStr) -> (u32, u32) {
                let s = name.to_string_lossy();
                let digits = s.strip_prefix('v').unwrap_or(&s);
                let mut parts = digits.split('.');
                let major = parts.next().and_then(|p| p.parse::<u32>().ok());
                let minor = parts.next().and_then(|p| p.parse::<u32>().ok());
                match major {
                    Some(major) => (major, minor.unwrap_or(0)),
                    None => (0, 0),
                }
            }
            let base_dir = std::path::Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
            if base_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(base_dir) {
                    let mut entries: Vec<_> = entries.flatten().collect();
                    entries.sort_by_key(|b| std::cmp::Reverse(parse_cuda_version(&b.file_name())));
                    for entry in entries {
                        let path = entry.path();
                        let lib_path = path.join("lib").join("x64");
                        if lib_path.exists() {
                            println!("cargo:rustc-link-search=native={}", lib_path.display());
                            cuda_lib_found = true;
                            break;
                        }
                    }
                }
            }
        } else {
            // Standard Linux/macOS paths. Also search `stubs/` subdirectories
            // because NVIDIA's toolkit ships libcuda.so there on installs
            // without a real driver (common in CI / docs.rs builders).
            for path in &[
                "/usr/local/cuda/lib64",
                "/usr/local/cuda/lib64/stubs",
                "/usr/local/cuda/lib",
                "/usr/lib/x86_64-linux-gnu",
                "/usr/lib/x86_64-linux-gnu/stubs",
                "/opt/cuda/lib64",
                "/opt/cuda/lib64/stubs",
            ] {
                if std::path::Path::new(path).exists() {
                    println!("cargo:rustc-link-search=native={}", path);
                    cuda_lib_found = true;
                }
            }
        }
    }

    // No CUDA library directory was discovered. This is NOT a stub build
    // (CARGO_FEATURE_CUDA_STUB was handled and returned above), so the crate
    // will try to link against the real CUDA driver and fail at the link
    // stage. Emit a clear, actionable warning rather than letting the build
    // die later with a cryptic linker error.
    if !cuda_lib_found {
        println!(
            "cargo:warning=craton-bolt: no CUDA library directory found \
             (checked CUDA_PATH and the standard install locations). This is a \
             real (non-stub) build, so it WILL FAIL at link time with an \
             unresolved CUDA driver symbol / \"cannot find -lcuda\" error."
        );
        println!(
            "cargo:warning=craton-bolt: to build WITHOUT a CUDA toolkit \
             (type-check / docs.rs / CUDA-less CI), use the cuda-stub feature: \
             `cargo build --no-default-features --features cuda-stub`. To build \
             the real GPU path, install a CUDA Toolkit >= 12 and set CUDA_PATH \
             (or place it under the standard install root)."
        );
    }
}

// ---------------------------------------------------------------------------
// Codegen fingerprint: auto-rotate the disk PTX cache key on codegen changes.
// ---------------------------------------------------------------------------
//
// Computes a stable hex digest over the codegen-relevant source — every
// `src/jit/*.rs` file PLUS `src/plan/physical_plan.rs` (the plan IR that
// codegen lowers into PTX) — and emits it as
// `cargo:rustc-env=BOLT_CODEGEN_FINGERPRINT=<digest>`, plus a
// `rerun-if-changed` on each hashed directory/file so the digest recomputes
// whenever the codegen surface changes.
//
// The hash is a self-contained 128-bit FNV-1a fold (no external crate): for
// each file, in sorted *relative-path* order, we mix in the relative path,
// the byte length, and the raw file bytes. Sorted order + relative-path-only
// (no absolute paths) + no timestamps make the digest reproducible across
// hosts and checkouts for identical source. Using the full relative path
// (e.g. `src/plan/physical_plan.rs`, not just `physical_plan.rs`) as the
// domain separator keeps files from distinct directories from aliasing.
fn emit_codegen_fingerprint() {
    use std::path::{Path, PathBuf};

    // Always re-run when the codegen dir changes (file added/removed/edited)
    // or the explicitly-named plan file changes.
    println!("cargo:rerun-if-changed=src/jit");
    println!("cargo:rerun-if-changed=src/plan/physical_plan.rs");

    // Collect `(relpath, abspath)` for every readable codegen-relevant file,
    // sorted by relative path so the hash is order-independent of the OS
    // directory listing.
    let mut files: Vec<(String, PathBuf)> = Vec::new();

    // 1. Every `*.rs` file in the `src/jit` codegen tree.
    let jit_dir = Path::new("src/jit");
    let Ok(entries) = std::fs::read_dir(jit_dir) else {
        // Codegen dir unreadable: fall back gracefully — emit no env var so
        // `option_env!` resolves to `None` and the salt uses CODEGEN_VERSION
        // + crate version alone (i.e. prior behaviour, never weaker).
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".rs") {
            files.push((format!("src/jit/{name}"), path));
        }
    }

    // 2. The plan IR codegen lowers from. A change to PhysicalPlan can alter
    //    emitted PTX for an otherwise-identical query, so it is part of the
    //    codegen surface even though it lives outside `src/jit`. If it is
    //    unreadable we fall back gracefully (emit nothing) like the dir case.
    let plan_path = PathBuf::from("src/plan/physical_plan.rs");
    if !plan_path.is_file() {
        return;
    }
    files.push(("src/plan/physical_plan.rs".to_string(), plan_path));

    files.sort_by(|a, b| a.0.cmp(&b.0));

    // 128-bit FNV-1a, split into two 64-bit lanes seeded with distinct
    // offset bases so the digest is wider than a single u64 (cheap collision
    // headroom) without pulling in a hashing crate.
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hi: u64 = FNV_OFFSET;
    let mut lo: u64 = FNV_OFFSET ^ 0x5555_5555_5555_5555;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            hi ^= u64::from(b);
            hi = hi.wrapping_mul(FNV_PRIME);
            lo ^= u64::from(b).rotate_left(1);
            lo = lo.wrapping_mul(FNV_PRIME);
        }
    };

    for (relpath, path) in &files {
        // Re-run codegen fingerprinting when this individual file changes
        // (the `src/jit` and plan-file rerun lines above already cover
        // additions/removals; this pins per-file content edits too).
        println!("cargo:rerun-if-changed={}", path.display());
        let Ok(bytes) = std::fs::read(path) else {
            // Skip an unreadable file rather than panicking; the directory
            // rerun-if-changed still covers it on a later successful build.
            continue;
        };
        // Domain-separate relpath | length | content so two files can't
        // alias by, e.g., shifting a byte across a file boundary or moving
        // identical bytes between directories.
        mix(relpath.as_bytes());
        mix(&[0x1f]); // unit separator between path and content
        mix(&(bytes.len() as u64).to_le_bytes());
        mix(&bytes);
        mix(&[0x1e]); // record separator between files
    }

    // Emit a fixed-width 32-char lowercase hex digest (same shape as the
    // disk cache's other keys). The consumer (codegen_salt) renders it as
    // `fp<digest>`.
    println!(
        "cargo:rustc-env=BOLT_CODEGEN_FINGERPRINT={:016x}{:016x}",
        hi, lo
    );
}

// ---------------------------------------------------------------------------
// rust-cuda (Wave A) PTX build hook.
// ---------------------------------------------------------------------------
//
// Gated on `cfg(feature = "rust-cuda")`. When ON, invokes cuda_builder
// against the sibling `kernels/` crate and writes the PTX to
// $OUT_DIR/partition.ptx. When OFF, writes an empty file at the same path
// so the `include_str!` in the feature-gated host code still resolves
// (the host code under `#[cfg(not(feature = "rust-cuda"))]` never reads it
// — see src/jit/partition_kernel.rs).

// ===========================================================================
// V-4 (HIGH) — SECURITY NOTE: build-time network fetch + external toolchain.
// ===========================================================================
//
// Enabling `--features rust-cuda` makes cuda_builder / rustc_codegen_nvvm
// DOWNLOAD and UNPACK an LLVM/libNVVM toolchain at build time and then RUN it
// as a codegen plugin during this build. There is currently NO in-repo
// integrity verification (no pinned checksum / signature) of the fetched
// toolchain — trusting it is equivalent to trusting whatever the upstream
// crate's downloader pulls onto the build host.
//
// This feature is intentionally OFF BY DEFAULT. Build it only on:
//   * network-isolated / egress-controlled CI runners, AND
//   * runners where the NVVM/LLVM toolchain artifacts are pre-pinned (vendored
//     or fetched from a checksum-verified internal mirror).
//
// Do NOT enable rust-cuda on developer laptops or shared CI without the above.
// Re-architecting the download to add integrity checks is tracked separately;
// this comment is the documented hardening guidance (see ci.yml V-4 note).
#[cfg(feature = "rust-cuda")]
fn compile_rust_cuda_kernels() {
    use cuda_builder::{CudaBuilder, NvvmArch};
    use std::path::PathBuf;

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let kernels_dir = manifest.join("kernels");
    let ptx_out = out_dir.join("partition.ptx");

    println!("cargo:rerun-if-changed=kernels/src");
    println!("cargo:rerun-if-changed=kernels/Cargo.toml");
    println!("cargo:rerun-if-changed=kernels/rust-toolchain.toml");

    // sm_70 matches Craton Bolt's hand-emit `.target sm_70` line so the PTX is
    // co-loadable with the other kernels (see docs/JIT_PIPELINE.md).
    CudaBuilder::new(&kernels_dir)
        .copy_to(&ptx_out)
        .arch(NvvmArch::Compute70)
        .build()
        .expect("cuda_builder failed to compile kernels/ to PTX");
}

#[cfg(not(feature = "rust-cuda"))]
fn compile_rust_cuda_kernels() {
    use std::path::PathBuf;

    // Write an empty PTX placeholder so the `include_str!` site in
    // src/jit/partition_kernel.rs has a file to point at when the host
    // crate is compiled. The macro must resolve at parse time even though
    // the body of the cfg-gated function never runs.
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR not set"));
    let ptx_out = out_dir.join("partition.ptx");
    if !ptx_out.exists() {
        std::fs::write(&ptx_out, "").expect("failed to write empty partition.ptx stub");
    }
}
