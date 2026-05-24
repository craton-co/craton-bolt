fn main() {
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
            } else if lib_path.exists() {
                println!("cargo:rustc-link-search=native={}", lib_path.display());
            }
        }
    } else {
        // Fallbacks if CUDA_PATH is not set.
        if cfg!(target_os = "windows") {
            // Check default installation directory structure.
            // On Windows, the standard path is C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\lib\x64.
            // We can search for directories matching this pattern.
            let base_dir = std::path::Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA");
            if base_dir.exists() {
                if let Ok(entries) = std::fs::read_dir(base_dir) {
                    for entry in entries.flatten() {
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
            // Standard Linux/macOS paths
            for path in &["/usr/local/cuda/lib64", "/usr/local/cuda/lib", "/usr/lib/x86_64-linux-gnu", "/opt/cuda/lib64"] {
                if std::path::Path::new(path).exists() {
                    println!("cargo:rustc-link-search=native={}", path);
                }
            }
        }
    }
}
