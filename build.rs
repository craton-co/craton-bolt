fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_CUDA_STUB");

    // Skip CUDA discovery when building with the `cuda-stub` feature
    // (e.g. on docs.rs or CUDA-less hosts).
    if std::env::var_os("CARGO_FEATURE_CUDA_STUB").is_some() {
        return;
    }

    println!("cargo:rerun-if-env-changed=CUDA_PATH");

    // Check CUDA_PATH environment variable first.
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let path = std::path::PathBuf::from(cuda_path);
        if cfg!(target_os = "windows") {
            let lib_path = path.join("lib").join("x64");
            if lib_path.exists() {
                println!("cargo:rustc-link-search=native={}", lib_path.display());
            }
        } else {
            let lib64_path = path.join("lib64");
            let lib_path = path.join("lib");
            if lib64_path.exists() {
                println!("cargo:rustc-link-search=native={}", lib64_path.display());
                let stubs_path = lib64_path.join("stubs");
                if stubs_path.exists() {
                    println!("cargo:rustc-link-search=native={}", stubs_path.display());
                }
            } else if lib_path.exists() {
                println!("cargo:rustc-link-search=native={}", lib_path.display());
            }
        }
    } else {
        // Fallbacks if CUDA_PATH is not set.
        if cfg!(target_os = "windows") {
            // Check default installation directory structure.
            // On Windows, the standard path is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\lib\x64.
            // Collect all matching entries and sort by name descending so the
            // highest-version install (e.g. v12.6 beats v12.4 beats v11.8) wins.
            let base_dir = std::path::Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
            if base_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(base_dir) {
                    let mut entries: Vec<_> = entries.flatten().collect();
                    entries.sort_by(|a, b| b.file_name().cmp(&a.file_name()));
                    for entry in entries {
                        let path = entry.path();
                        let lib_path = path.join("lib").join("x64");
                        if lib_path.exists() {
                            println!("cargo:rustc-link-search=native={}", lib_path.display());
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
                }
            }
        }
    }
}
