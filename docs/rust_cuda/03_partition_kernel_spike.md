# Spike 03 — Rewriting `partition_kernel` in Rust-CUDA

**Target.** `src/jit/partition_kernel.rs` — the simplest non-trivial Craton Bolt
kernel: a grid-stride loop that hashes one `i32` key per row, masks to a
partition id, increments a per-partition counter via one global atomic, and
writes the pid back. No shared memory, no reduction, no warp intrinsics.

**Question.** Can we hand-roll the same kernel in Rust on the
`rustc_codegen_nvvm` toolchain, produce equivalent PTX, and integrate it into
Craton Bolt's existing JIT pipeline without giving up runtime parameterisation?

---

## 1. What we emit today

`compile_partition_kernel()` (150 lines of Rust `writeln!`s) produces about
60 lines of PTX. The hot inner loop is the only part that matters:

```ptx
.version 7.5
.target sm_70
.address_size 64

.visible .entry bolt_partition(
    .param .u64 bolt_partition_param_0,   // keys     (const i32*)
    .param .u64 bolt_partition_param_1,   // pids     (u32*)
    .param .u64 bolt_partition_param_2,   // counts   (u32*)
    .param .u32 bolt_partition_param_3    // n_rows
)
{
    .reg .pred  %p<4>;
    .reg .b32   %r<32>;
    .reg .b64   %rd<32>;

    mov.u32     %r0, %ctaid.x;
    mov.u32     %r1, %ntid.x;
    mov.u32     %r2, %tid.x;
    mad.lo.s32  %r3, %r0, %r1, %r2;        // gtid
    mov.u32     %r4, %nctaid.x;
    mul.lo.s32  %r5, %r4, %r1;             // stride
    ld.param.u32 %r6, [.._param_3];        // n_rows

    ld.param.u64 %rd0, [.._param_0];
    cvta.to.global.u64 %rd0, %rd0;
    ld.param.u64 %rd1, [.._param_1];
    cvta.to.global.u64 %rd1, %rd1;
    ld.param.u64 %rd2, [.._param_2];
    cvta.to.global.u64 %rd2, %rd2;

    mov.u32 %r7, 0x9E3779B1;                // Knuth multiplier

LOOP_TOP:
    setp.ge.u32 %p0, %r3, %r6;
    @%p0 bra LOOP_DONE;

    mul.wide.u32 %rd10, %r3, 4;
    add.s64      %rd11, %rd0, %rd10;
    ld.global.s32 %r8, [%rd11];             // key
    mul.lo.u32   %r9,  %r8, %r7;            // hash = key * mult (wrap)
    and.b32      %r10, %r9, 0xFFF;          // pid = hash & (4096-1)

    mul.wide.u32 %rd12, %r10, 4;
    add.s64      %rd13, %rd2, %rd12;
    mov.u32      %r12, 1;
    atom.global.add.u32 %r11, [%rd13], %r12; // counts[pid]++

    mul.wide.u32 %rd14, %r3, 4;
    add.s64      %rd15, %rd1, %rd14;
    st.global.u32 [%rd15], %r10;             // pids[i] = pid

    add.u32 %r3, %r3, %r5;
    bra LOOP_TOP;
LOOP_DONE:
    ret;
}
```

Five real instructions of arithmetic, one load, one store, one atomic. That's
the bar Rust-CUDA has to clear.

---

## 2. Rust kernel rewrite

Hypothetical placement: `kernels/src/partition.rs` in a new workspace member
`craton-bolt-kernels`. The kernel itself, with every `cuda_std` symbol verified
against [`cuda_std` 0.2.2 docs](https://docs.rs/cuda_std/0.2.2/cuda_std/)
and the
[`cuda_std::atomic::intrinsics`](https://docs.rs/cuda_std/latest/cuda_std/atomic/intrinsics/index.html)
listing:

```rust
#![no_std]
#![cfg_attr(target_os = "cuda", feature(register_attr))]

use cuda_std::kernel;
use cuda_std::thread;
use cuda_std::atomic::intrinsics::atomic_fetch_add_relaxed_u32_device;

const HASH_MULT: u32  = 0x9E37_79B1;
const NUM_PARTS: u32  = 4096;
const MASK: u32       = NUM_PARTS - 1;

/// Pass-1 of the Tier-2 GROUP BY: count rows per partition and
/// record each row's pid for the scatter pass.
#[kernel]
pub unsafe fn bolt_partition(
    keys: &[i32],              // length-checked slice (length passed via launch)
    pids: *mut u32,            // n_rows
    counts: *mut u32,          // NUM_PARTS
    n_rows: u32,
) {
    let stride = thread::grid_dim_x() * thread::block_dim_x();
    let mut i  = thread::index_1d();          // u32

    while i < n_rows {
        let key  = *keys.get_unchecked(i as usize) as u32;
        let hash = key.wrapping_mul(HASH_MULT);
        let pid  = hash & MASK;

        // Lowers to `atom.global.add.u32` — the relaxed/device variant is the
        // direct PTX equivalent of what we hand-emit today. Return value is
        // discarded.
        let _ = atomic_fetch_add_relaxed_u32_device(counts.add(pid as usize), 1);

        *pids.add(i as usize) = pid;
        i += stride;
    }
}
```

Symbol cites (all from `cuda_std` 0.2.2):

| What we want | `cuda_std` symbol |
|---|---|
| `mad.lo.s32` of ctaid/ntid/tid | `thread::index_1d() -> u32` ([docs](https://docs.rs/cuda_std/0.2.2/cuda_std/thread/index.html)) |
| `mul.lo.s32` of nctaid·ntid | `thread::grid_dim_x() * thread::block_dim_x()` |
| `atom.global.add.u32` | `atomic::intrinsics::atomic_fetch_add_relaxed_u32_device(*mut u32, u32) -> u32` (verified signature, fetched 2026-05-25) |

The high-level `AtomicU32` wrapper in `cuda_std::atomic` does not exist yet
as of 0.2.2; only `AtomicF32`/`AtomicF64` are shipped (issue
[Rust-GPU/Rust-CUDA#8](https://github.com/Rust-GPU/Rust-CUDA/issues/8)
tracks integer atomic wrappers). The intrinsic is the supported path today.

---

## 3. Build pipeline

Workspace layout:

```
craton-bolt/
├── Cargo.toml                    # workspace
├── src/...                       # host crate (unchanged)
├── build.rs                      # extended: invoke CudaBuilder
└── kernels/
    ├── Cargo.toml
    └── src/
        ├── lib.rs
        └── partition.rs
```

`kernels/Cargo.toml`:

```toml
[package]
name    = "craton-bolt-kernels"
edition = "2021"

[lib]
# cdylib: nvptx64-nvidia-cuda has no bin/rlib output for the codegen backend.
# rlib  : so host code can share const definitions (NUM_PARTS, etc.).
crate-type = ["cdylib", "rlib"]

[dependencies]
cuda_std = "0.2"
```

(The dual `cdylib + rlib` is the documented pattern; see the
[Rust-CUDA getting-started guide](https://rust-gpu.github.io/rust-cuda/guide/getting_started.html).)

Host `build.rs` addition:

```rust
use cuda_builder::{CudaBuilder, NvvmArch};

fn main() {
    let out = std::env::var("OUT_DIR").unwrap();
    CudaBuilder::new("kernels")
        .arch(NvvmArch::Compute70)               // match current sm_70 target
        .copy_to(format!("{out}/partition.ptx"))
        .build()
        .unwrap();
    println!("cargo:rerun-if-changed=kernels");
}
```

Host consumption (in place of `compile_partition_kernel()`):

```rust
pub fn compile_partition_kernel() -> BoltResult<String> {
    Ok(include_str!(concat!(env!("OUT_DIR"), "/partition.ptx")).to_owned())
}
```

The dispatcher's `cuModuleLoadDataEx` + `cuModuleGetFunction("bolt_partition")`
path stays identical — the `#[kernel]` attribute produces `.visible .entry`
with the function's name.

---

## 4. What Craton Bolt loses and gains

### Loss — runtime kernel specialisation

The big one. `compile_partition_reduce_kernel_multi(n_vals: usize)` and its
relatives generate **different PTX per call**: the number of aggregate
columns, value types, and inner unrolling all bake into the emitted text at
runtime. Rust kernels are AOT-monomorphised against a fixed const-generic
shape; you can't construct a new `n_vals` variant after the binary is
built. Options to recover this:

1. Pre-monomorphise `partition::<N>` for `N ∈ {1, 2, 4, 8, 16}`, dispatch via
   match. Cheap for the *partition* kernel (`N` is just 1), painful for the
   multi-agg reduce kernels.
2. Keep the existing string-templating path for the parameterised kernels;
   only rewrite the fixed-shape kernels (`partition`, `scatter`,
   `shmem_count`) in Rust. Mixed pipeline.

Partition is squarely in the "fixed-shape" bucket, which is why it's the
right spike target.

### Loss — toolchain weight

`rustc_codegen_nvvm` needs the libNVVM static libs from the CUDA toolkit, a
compatible LLVM, and a pinned nightly. Today Craton Bolt's only build dep is a
working `nvcc`/driver. Adding a codegen backend is a real CI surface.

### Gain — borrow-checked GPU code

`keys: &[i32]` carries length; the kernel can `keys.get_unchecked(i)` rather
than synthesise the `mul.wide.u32 / add.s64 / ld.global.s32` triple by hand.
Off-by-ones inside the JIT emitter (we've fixed two so far) become type
errors. Shared constants (`NUM_PARTITIONS`, `HASH_MULTIPLIER`,
`BLOCK_THREADS`) live in one place and host code can `use` them.

### Gain — debuggability

PTX-shape tests in `partition_kernel.rs` exist precisely because `writeln!`
emitters silently produce wrong code. A Rust kernel compiled by the actual
compiler can't emit malformed PTX; the failure modes shrink to "this lowers
to the wrong instruction", which `nvdisasm` on the OUT_DIR PTX answers
deterministically.

---

## 5. Concrete questions before committing

1. **sm_70 target reachable?** Yes. `CudaBuilder::arch(NvvmArch::Compute70)`
   is the documented knob; libNVVM has supported `compute_70` since CUDA 9.
   (Note: from the Rust 1.97 release on 2026-07-09, sm_70 becomes the
   default baseline for the `nvptx64-nvidia-cuda` target — see the
   [Rust blog post, 2026-05-01](https://blog.rust-lang.org/2026/05/01/nvptx-baseline-update/) —
   so we're aligned with the wind direction.)

2. **Incremental rebuilds?** Coarse. `CudaBuilder::build()` runs a full
   `cargo build` of the kernel crate against the nvptx target on every
   trigger of `cargo:rerun-if-changed=kernels`. Touching one kernel file
   re-codegens every kernel in that crate. Mitigation: split kernels into
   separate workspace crates, each with its own `CudaBuilder` invocation, so
   editing `partition.rs` only rebuilds `craton-bolt-kernel-partition`. Adds
   workspace bookkeeping but matches the granularity we already have in
   `src/jit/`.

3. **Does `atomic_fetch_add_relaxed_u32_device` lower to
   `atom.global.add.u32`?** That is its documented purpose; the
   intrinsic name is the PTX-canonical mapping for the relaxed/device-scope
   variant. The relaxed ordering is what `atom.global.add` is on sm_70
   without explicit `.acq_rel.gpu` semantics — equivalent to what we emit.
   **Verification step before committing:** build the kernel, run
   `nvdisasm`/grep on the produced PTX, confirm one
   `atom.global.add.u32` per loop iteration. This is a 30-minute experiment
   and gates the entire spike.

4. **Can we ship pre-compiled PTX in the cargo package?** Yes — the PTX is a
   plain UTF-8 string captured at build time, and `include_str!` embeds it
   in the host binary. End users of Craton Bolt do **not** need the Rust-CUDA
   toolchain; only Craton Bolt's CI does, and only when a kernel source
   changes. The PTX itself can be checked in alongside the source and
   refreshed by `cargo xtask rebuild-kernels`, mirroring how some
   `wgpu`/SPIR-V projects ship pre-compiled shaders.

---

## Recommendation

**Spike viable — yes, narrowly.** The partition kernel is a near-textbook
case for Rust-CUDA: no shared memory, no warp intrinsics, no runtime
shape parameterisation, and the one tricky operation (global atomic add on
u32) has a documented intrinsic with a clean PTX mapping. The rewrite is
plausibly 30 lines of Rust replacing 150 lines of string formatting, with
every operation cited against a stable cuda_std symbol.

The risk is not the kernel — it's the **toolchain**. Pinning a nightly +
libNVVM in CI, doubling the build matrix (cdylib for kernels, normal build
for host), and giving up runtime PTX templating for the multi-agg kernels is
a non-trivial commitment. Recommend keeping the JIT string emitter for
parameterised kernels and only migrating fixed-shape kernels (partition,
scatter, shmem_count, shmem_sum) to Rust.

**Next concrete step.** Build the kernel out-of-tree (separate scratch repo,
no Craton Bolt integration) using `cuda_builder` against `NvvmArch::Compute70`,
dump the emitted PTX, and diff it against the hand-emitted version above.
If the inner loop comes out within a couple of instructions of identical
and the atomic lowers to `atom.global.add.u32`, open a tracking issue for
the in-tree migration with the toolchain-pinning work broken out.

---

*Sources verified 2026-05-25:*
*[Rust-CUDA getting started](https://rust-gpu.github.io/rust-cuda/guide/getting_started.html)*,
*[cuda_std 0.2.2 docs](https://docs.rs/cuda_std/0.2.2/)*,
*[cuda_std::atomic::intrinsics listing](https://docs.rs/cuda_std/latest/cuda_std/atomic/intrinsics/index.html)*,
*[Rust-CUDA issue #8 — integer atomics design](https://github.com/Rust-GPU/Rust-CUDA/issues/8)*,
*[Rust 1.97 nvptx baseline update, blog 2026-05-01](https://blog.rust-lang.org/2026/05/01/nvptx-baseline-update/)*.
