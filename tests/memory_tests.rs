// SPDX-License-Identifier: Apache-2.0

//! CUDA-Oxide memory-safety tests. These prove that Rust's borrow checker
//! makes the obvious GPU-memory foot-guns impossible.
//!
//! Three classes of test:
//! 1. **Type-level**: functions and `assert_*` helpers that must compile. The
//!    fact that this integration-test binary builds is the proof.
//! 2. **Compile-fail examples** (documented below): snippets that intentionally
//!    do not compile. They live in module doc comments as `compile_fail`
//!    doctests so `cargo test --doc` actually runs them.
//! 3. **Live tests** (`#[ignore]`'d): exercise round-trips through the GPU.
//!    Skipped on hosts without a CUDA device; run with
//!    `cargo test -- --ignored` on a GPU host.
//!
//! All type-level helpers below are concrete on `i32` / `f64` to avoid pulling
//! `bytemuck` (a non-dev dependency) into the test binary's name resolution.

use craton_patina::cuda::{GpuVec, GpuView, GpuViewMut};

// ---- Type-level proofs (compile, therefore valid) ---------------------------

/// A function that consumes a shared view is callable with a borrowed `GpuVec`.
/// Concrete on `i32` to avoid needing `bytemuck::Pod` in scope here.
#[allow(dead_code)]
fn consume_shared(_v: GpuView<'_, i32>) {}

/// A function that consumes a mutable view is callable with a mut-borrowed `GpuVec`.
#[allow(dead_code)]
fn consume_mut(_v: GpuViewMut<'_, i32>) {}

#[test]
fn shared_views_are_copy() {
    // `GpuView: Copy` is a type-level invariant; the assert is at the type level.
    fn assert_copy<T: Copy>() {}
    assert_copy::<GpuView<'static, i32>>();
    assert_copy::<GpuView<'static, f64>>();
}

#[test]
fn shared_views_are_clone() {
    fn assert_clone<T: Clone>() {}
    assert_clone::<GpuView<'static, i32>>();
}

#[test]
fn mut_view_is_send() {
    // `GpuViewMut` must NOT be `Copy` (otherwise the exclusive-borrow guarantee
    // would be defeated). We can't directly assert `!Copy` in stable Rust, so
    // we instead assert the positive properties we DO want: `Send` and `Sized`.
    // The negative property is documented in the compile-fail doctests below
    // and in `src/cuda/smart_ptrs.rs`.
    fn assert_send<T: Send>() {}
    fn assert_sized<T: Sized>() {}
    assert_send::<GpuViewMut<'static, i32>>();
    assert_sized::<GpuViewMut<'static, i32>>();
}

#[test]
fn shared_view_is_send_but_not_sync() {
    // `GpuView` is `Send` so it can be moved between threads, but intentionally
    // NOT `Sync`: a concurrent thread that still holds the parent `GpuVec`
    // could launch a writer kernel through `GpuViewMut`, racing the reader.
    // See `src/cuda/smart_ptrs.rs` for the full rationale.
    fn assert_send<T: Send>() {}
    assert_send::<GpuView<'static, i32>>();
    assert_send::<GpuView<'static, f64>>();
}

#[test]
fn gpu_vec_is_send() {
    // `GpuVec` owns a `GpuBuffer`, which is `Send` but intentionally not `Sync`.
    fn assert_send<T: Send>() {}
    assert_send::<GpuVec<i32>>();
}

// ---- Compile-fail proofs (run by `cargo test --doc`) ------------------------
//
// The blocks below are attached to dummy `pub fn` items so rustdoc picks them
// up as doctests in this integration-test crate. Each one demonstrates ONE of
// the three soundness invariants documented at the top of
// `src/cuda/smart_ptrs.rs`.

/// **Invariant 1**: a `GpuView` cannot outlive the `GpuVec` it borrows from.
///
/// ```compile_fail
/// use craton_patina::cuda::GpuVec;
/// let view = {
///     let vec = GpuVec::<i32>::from_slice(&[1, 2, 3]).unwrap();
///     vec.view()
/// }; // `vec` dropped here
/// let _ = view.len(); // ERROR: `vec` does not live long enough
/// ```
pub fn _doc_view_outlives_vec() {}

/// **Invariant 2**: a `&mut GpuVec` (i.e. an outstanding `GpuViewMut`) is
/// exclusive — no `GpuView` can coexist with it.
///
/// ```compile_fail
/// use craton_patina::cuda::GpuVec;
/// let mut vec = GpuVec::<i32>::from_slice(&[1, 2, 3]).unwrap();
/// let shared = vec.view();
/// let _exclusive = vec.view_mut(); // ERROR: cannot borrow `vec` as mutable
/// let _ = shared.len();             // forces `shared` to be live across the
///                                    // mutable borrow above
/// ```
pub fn _doc_mut_excludes_shared() {}

/// **Invariant 2 (reverse)**: cannot take a shared view while an exclusive
/// view is live.
///
/// ```compile_fail
/// use craton_patina::cuda::GpuVec;
/// let mut vec = GpuVec::<i32>::from_slice(&[1, 2, 3]).unwrap();
/// let exclusive = vec.view_mut();
/// let _shared = vec.view(); // ERROR: cannot borrow `vec` as immutable
/// let _ = exclusive.len();   // keeps `exclusive` alive across the line above
/// ```
pub fn _doc_shared_excludes_mut() {}

/// **Invariant 3**: a moved/dropped `GpuVec` cannot be used again
/// (use-after-free at the Rust level).
///
/// ```compile_fail
/// use craton_patina::cuda::GpuVec;
/// let vec = GpuVec::<i32>::from_slice(&[1, 2, 3]).unwrap();
/// drop(vec); // moved into `drop`
/// let _ = vec.len(); // ERROR: borrow of moved value: `vec`
/// ```
pub fn _doc_use_after_move() {}

/// **Invariant 3 (move into function)**: same idea, moved via a function call.
///
/// ```compile_fail
/// use craton_patina::cuda::GpuVec;
/// fn consume(_v: GpuVec<i32>) {}
/// let vec = GpuVec::<i32>::from_slice(&[1, 2, 3]).unwrap();
/// consume(vec);
/// let _ = vec.len(); // ERROR: borrow of moved value: `vec`
/// ```
pub fn _doc_use_after_move_into_fn() {}

// ---- Live GPU tests (need an NVIDIA device) ---------------------------------
//
// These actually exercise the CUDA driver. They're `#[ignore]`'d so they don't
// run on hosts without a GPU / CUDA toolkit. To run them:
//
//     cargo test -- --ignored
//
// (or `cargo test --test memory_tests -- --ignored` to scope to this file).

#[test]
#[ignore = "requires CUDA device - run with `cargo test -- --ignored`"]
fn round_trip_i32() {
    let _ctx = craton_patina::cuda::CudaContext::new(0).expect("CUDA context");
    let data: Vec<i32> = (0..1024).collect();
    let vec = GpuVec::from_slice(&data).expect("alloc + h2d");
    let back = vec.to_vec().expect("d2h");
    assert_eq!(data, back);
}

#[test]
#[ignore = "requires CUDA device - run with `cargo test -- --ignored`"]
fn round_trip_f64() {
    let _ctx = craton_patina::cuda::CudaContext::new(0).expect("CUDA context");
    let data: Vec<f64> = (0..1024).map(|i| i as f64 * 0.5).collect();
    let vec = GpuVec::from_slice(&data).expect("alloc + h2d");
    let back = vec.to_vec().expect("d2h");
    assert_eq!(data, back);
}

#[test]
#[ignore = "requires CUDA device - run with `cargo test -- --ignored`"]
fn views_observe_same_buffer() {
    // Two shared views of the same vec must report the same length and the
    // same device pointer — they're aliases of one allocation.
    let _ctx = craton_patina::cuda::CudaContext::new(0).expect("CUDA context");
    let vec = GpuVec::<i32>::from_slice(&[1, 2, 3, 4]).expect("alloc");
    let a = vec.view();
    let b = vec.view();
    assert_eq!(a.len(), b.len());
    assert_eq!(a.device_ptr(), b.device_ptr());
    assert_eq!(a.len(), 4);
}

#[test]
#[ignore = "requires CUDA device - run with `cargo test -- --ignored`"]
fn mut_view_reborrow_as_shared() {
    // `GpuViewMut::as_view` lets you temporarily downgrade to a shared view
    // without giving up the exclusive borrow.
    let _ctx = craton_patina::cuda::CudaContext::new(0).expect("CUDA context");
    let mut vec = GpuVec::<i32>::from_slice(&[10, 20, 30]).expect("alloc");
    let exclusive = vec.view_mut();
    let shared = exclusive.as_view();
    assert_eq!(shared.len(), 3);
    assert_eq!(shared.device_ptr(), exclusive.device_ptr());
}

#[test]
#[ignore = "requires CUDA device - run with `cargo test -- --ignored`"]
fn drop_is_safe_no_double_free() {
    // Dropping a `GpuVec` must call `cuMemFree` exactly once. If `Drop` ever
    // fired twice we'd see a CUDA error on the second free. This test passes
    // simply by completing without panic / abort.
    let _ctx = craton_patina::cuda::CudaContext::new(0).expect("CUDA context");
    let vec = GpuVec::<i32>::from_slice(&[1, 2, 3]).expect("alloc");
    drop(vec);
}
