//! Minimal FFI surface for the Swift package.
//!
//! All real logic lives in the `mni_framework_core` crate. This file is
//! thin glue that converts C-ABI types into Rust types and back.
//!
//! # Memory model
//!
//! Rust allocates tensors on the heap and hands Swift an opaque pointer
//! (`*mut Tensor`). Swift must call `ml_tensor_free` exactly once per
//! pointer it received from a constructor. The Swift wrapper class uses
//! `deinit` to enforce this automatically.
//!
//! # Safety
//!
//! Every function taking a pointer assumes:
//! - The pointer was returned by one of this module's constructors.
//! - It has not been freed.
//! - No other thread is concurrently freeing it.
//!
//! Null pointers are checked explicitly and return a sentinel value
//! instead of dereferencing.

use mni_framework_core::Tensor;

// ---------------------------------------------------------------------------
// Value-returning helpers (simple, no lifetime management)
// ---------------------------------------------------------------------------

/// Returns a + b. Sanity-check that the FFI pipeline works at all.
#[no_mangle]
pub extern "C" fn ml_add(a: i32, b: i32) -> i32 {
    a + b
}

/// Allocate a `rows x cols` zero tensor, sum it, and drop it — all in one
/// call. Kept alongside the handle-based API as a simpler reference point.
#[no_mangle]
pub extern "C" fn ml_tensor_zeros_sum(rows: u32, cols: u32) -> f32 {
    let t = Tensor::zeros(&[rows as usize, cols as usize]);
    t.sum()
}

/// Allocate an `n`-element iota tensor, sum it, drop it.
#[no_mangle]
pub extern "C" fn ml_tensor_iota_sum(n: u32) -> f32 {
    let t = Tensor::iota(&[n as usize]);
    t.sum()
}

// ---------------------------------------------------------------------------
// Handle-based API: Rust owns memory, Swift holds opaque pointers
// ---------------------------------------------------------------------------

/// Constructor: allocate a `rows x cols` zero tensor on the heap and
/// return an owning pointer. Caller must eventually call `ml_tensor_free`.
#[no_mangle]
pub extern "C" fn ml_tensor_zeros(rows: u32, cols: u32) -> *mut Tensor {
    let t = Box::new(Tensor::zeros(&[rows as usize, cols as usize]));
    Box::into_raw(t)
}

/// Constructor: allocate an `n`-element iota tensor on the heap.
#[no_mangle]
pub extern "C" fn ml_tensor_iota(n: u32) -> *mut Tensor {
    let t = Box::new(Tensor::iota(&[n as usize]));
    Box::into_raw(t)
}

/// Borrow: read `sum()` from a tensor without taking ownership.
/// Returns 0.0 on null pointer so a misbehaving caller can't segfault us.
///
/// # Safety
/// See module docs. `handle` must be a live pointer from a constructor.
#[no_mangle]
pub unsafe extern "C" fn ml_tensor_sum(handle: *const Tensor) -> f32 {
    if handle.is_null() {
        return 0.0;
    }
    (*handle).sum()
}

/// Borrow: read `numel()` from a tensor. Returns 0 on null.
///
/// # Safety
/// See module docs.
#[no_mangle]
pub unsafe extern "C" fn ml_tensor_numel(handle: *const Tensor) -> u64 {
    if handle.is_null() {
        return 0;
    }
    (*handle).numel() as u64
}

/// Destructor: reclaim the heap allocation. Safe to call with null.
///
/// After this call the caller must not use `handle` again — doing so is
/// undefined behavior.
///
/// # Safety
/// `handle` must be a pointer returned by one of this module's
/// constructors, and must not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn ml_tensor_free(handle: *mut Tensor) {
    if handle.is_null() {
        return;
    }
    // Reconstruct the Box so Rust drops it properly.
    drop(Box::from_raw(handle));
}

// ---------------------------------------------------------------------------
// Data movement: Swift -> Rust and Rust -> Swift
//
// Two patterns are exposed:
//
//   1. COPY: `ml_tensor_from_data` / `ml_tensor_copy_into`.
//      Caller owns the memory on their side; we copy in/out. Safe and
//      simple; appropriate when the caller will retain the data anyway
//      or when the tensor is small.
//
//   2. BORROW: `ml_tensor_data_ptr`.
//      Returns a pointer into Rust's Vec. Zero-copy, but the pointer is
//      only valid until the tensor is freed or mutated. Swift must scope
//      its use tightly (`withUnsafeData { ... }`).
// ---------------------------------------------------------------------------

/// Allocate a tensor from a caller-provided row-major buffer.
///
/// Returns a null pointer if `rows * cols != len`, or if either input
/// pointer is null. Caller is responsible for calling `ml_tensor_free`.
///
/// # Safety
/// `data` must point to at least `len` valid `f32` values. The data is
/// copied; the caller's memory is not retained by Rust.
#[no_mangle]
pub unsafe extern "C" fn ml_tensor_from_data(
    data: *const f32,
    len: u64,
    rows: u32,
    cols: u32,
) -> *mut Tensor {
    if data.is_null() {
        return std::ptr::null_mut();
    }
    let expected = (rows as u64) * (cols as u64);
    if expected != len {
        return std::ptr::null_mut();
    }
    // Copy caller's buffer into a Rust-owned Vec.
    let slice = std::slice::from_raw_parts(data, len as usize);
    match Tensor::from_data(&[rows as usize, cols as usize], slice.to_vec()) {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Copy a tensor's row-major data into a caller-provided buffer.
/// Returns the number of f32 elements actually written (= min(numel,
/// out_len)). Returns 0 on null input.
///
/// # Safety
/// `out` must point to writable memory for at least `out_len` f32 values.
#[no_mangle]
pub unsafe extern "C" fn ml_tensor_copy_into(
    handle: *const Tensor,
    out: *mut f32,
    out_len: u64,
) -> u64 {
    if handle.is_null() || out.is_null() {
        return 0;
    }
    let t = &*handle;
    let n = std::cmp::min(t.numel() as u64, out_len);
    std::ptr::copy_nonoverlapping(t.as_slice().as_ptr(), out, n as usize);
    n
}

/// Return a read-only pointer to the tensor's internal buffer. Pointer
/// is only valid while the tensor is alive and unmodified. Swift must
/// NOT keep it past the tensor's lifetime.
///
/// Returns null on null input.
///
/// # Safety
/// The returned pointer must not be written through. It is invalidated
/// by any mutation of the tensor or by `ml_tensor_free`.
#[no_mangle]
pub unsafe extern "C" fn ml_tensor_data_ptr(handle: *const Tensor) -> *const f32 {
    if handle.is_null() {
        return std::ptr::null();
    }
    (*handle).as_slice().as_ptr()
}
