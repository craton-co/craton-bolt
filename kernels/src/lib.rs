// SPDX-License-Identifier: Apache-2.0
//
// Craton Bolt GPU kernels (rust-cuda Wave A spike).
//
// This crate is compiled to PTX by `rustc_codegen_nvvm` via `cuda_builder`.
// Wave A scope: ONE kernel — `bolt_partition` — the simplest non-trivial
// PTX emitter in `src/jit/`. See:
//   - docs/rust_cuda/03_partition_kernel_spike.md  (design + symbol cites)
//   - docs/rust_cuda/05_ptx_loader_compat.md       (proves the host loader
//     can consume `rustc_codegen_nvvm` PTX without changes)
//   - src/jit/partition_kernel.rs                  (the hand-emit equivalent
//     this is a 1-for-1 rewrite of)
//
// The `no_std` gate is conditional on the nvptx64 target so that host-side
// `cargo check` / IDE tooling still sees a normal `std`-enabled crate. The
// `target_arch = "nvptx64"` cfg is set by the nvptx64-nvidia-cuda target
// triple `rustc_codegen_nvvm` produces; native (host) compilation does not
// trigger it, so the lib remains importable from host code (for sharing
// const definitions in a future wave).

#![cfg_attr(target_arch = "nvptx64", no_std)]
// `cuda_std`'s `#[kernel]` attribute requires the `register_attr` feature
// (per the rust-cuda guide). Only enabled on the GPU build.
#![cfg_attr(target_arch = "nvptx64", feature(register_attr))]

// --- Shared constants -------------------------------------------------------
//
// These mirror `src/jit/partition_kernel.rs` byte-for-byte. The host crate
// continues to consume the originals from that module; the duplicates here
// are device-side only and exist because `no_std` on nvptx64 cannot pull in
// the host's `crate::jit::partition_kernel::*` symbols. If we ever flip the
// build so the host imports `craton-bolt-kernels` as an `rlib`, these become the
// single source of truth and the host re-exports them.

/// Number of hash partitions. Must be a power of two so that `% N` lowers to
/// `and.b32`. Mirrors `crate::jit::partition_kernel::NUM_PARTITIONS`.
pub const NUM_PARTITIONS: u32 = 4096;

/// Knuth multiplicative hash constant — `floor(2^32 / phi)` rounded to the
/// nearest odd integer. Mirrors `crate::jit::partition_kernel::HASH_MULTIPLIER`.
pub const HASH_MULTIPLIER: u32 = 0x9E37_79B1;

// --- GPU kernel -------------------------------------------------------------
//
// Everything below is `target_arch = "nvptx64"`-only. On the host build the
// crate degrades to a constants-only library, which is what makes the
// `rlib` half of `crate-type = ["cdylib", "rlib"]` usable from host code.

#[cfg(target_arch = "nvptx64")]
mod gpu {
    use cuda_std::atomic::intrinsics::atomic_fetch_add_relaxed_u32_device;
    use cuda_std::kernel;
    use cuda_std::thread;

    // Re-export at the parent scope's expectations: the symbols we use are
    // imported above; the `#[kernel]` attribute on the function below is what
    // makes `rustc_codegen_nvvm` emit `.visible .entry`.

    const HASH_MULT: u32 = super::HASH_MULTIPLIER;
    const NUM_PARTS: u32 = super::NUM_PARTITIONS;
    const MASK: u32 = NUM_PARTS - 1;

    /// Tier-2 hash-partition pass-1 kernel.
    ///
    /// Equivalent C-with-CUDA:
    /// ```c
    /// __global__ void bolt_partition(
    ///     const int32_t* keys,
    ///     uint32_t*      partition_ids,
    ///     uint32_t*      counts,
    ///     uint32_t       n_rows)
    /// {
    ///     uint32_t stride = gridDim.x * blockDim.x;
    ///     for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    ///          i < n_rows; i += stride) {
    ///         uint32_t key  = (uint32_t)keys[i];
    ///         uint32_t hash = key * 0x9E3779B1u;
    ///         uint32_t pid  = hash & (NUM_PARTITIONS - 1);
    ///         atomicAdd(&counts[pid], 1u);
    ///         partition_ids[i] = pid;
    ///     }
    /// }
    /// ```
    ///
    /// `counts` MUST be zero-initialised by the caller before launch
    /// (matches `src/jit/partition_kernel.rs` contract).
    ///
    /// Pointer parameters are raw `*mut` rather than `&[T]` slices because
    /// `cuda_std` 0.2.2's kernel-ABI doesn't yet carry slice metadata
    /// through the nvptx parameter convention reliably (see the audit doc
    /// section 1 — "high-level `AtomicU32` wrapper does not exist yet as
    /// of 0.2.2"). Raw pointers match the hand-emit's PTX param shape
    /// exactly: four `.u64 / .u32` slots in declaration order.
    #[kernel]
    #[allow(improper_ctypes_definitions)]
    #[export_name = "bolt_partition"]
    pub unsafe fn bolt_partition(
        keys: *const i32,
        partition_ids: *mut u32,
        counts: *mut u32,
        n_rows: u32,
    ) {
        // stride = gridDim.x * blockDim.x   (both u32)
        let stride: u32 = thread::grid_dim_x() * thread::block_dim_x();

        // i = blockIdx.x * blockDim.x + threadIdx.x
        let mut i: u32 = thread::index_1d();

        while i < n_rows {
            // key  = keys[i]                    // ld.global.s32
            let key: u32 = (*keys.add(i as usize)) as u32;

            // hash = key * HASH_MULT (wrap)     // mul.lo.u32
            let hash: u32 = key.wrapping_mul(HASH_MULT);

            // pid  = hash & MASK                // and.b32
            let pid: u32 = hash & MASK;

            // atomicAdd(&counts[pid], 1)        // atom.global.add.u32
            //
            // Return value of the intrinsic is the OLD counter value; we
            // discard it. The hand-emit uses `atom.global.add.u32 %r11, ...`
            // with %r11 unread for the same reason — PTX requires a
            // destination register but the SASS form is the same.
            let _ = atomic_fetch_add_relaxed_u32_device(counts.add(pid as usize), 1u32);

            // partition_ids[i] = pid            // st.global.u32
            *partition_ids.add(i as usize) = pid;

            i += stride;
        }
    }
}

// Re-export so the symbol name `bolt_partition` is reachable. With
// `#[export_name = "bolt_partition"]` and the `#[kernel]` attribute the
// PTX itself uses the unmangled name; this re-export keeps `cargo check`
// happy on the host where the gpu module is gated out.
#[cfg(target_arch = "nvptx64")]
pub use gpu::bolt_partition;
