//! C-ABI FFI surface for the Swift package.
//!
//! # Architecture
//!
//! A single global [`Engine`] owns the real `TensorStore` + `Tape` from
//! `mni_framework_core`. All tensors created from Swift live inside this
//! store, keyed by a `u32` id (the `TensorId` type). Swift wraps these
//! ids in a `Tensor` class whose `deinit` calls `ml_engine_release`.
//!
//! This mirrors the pattern used by the Node N-API layer (see
//! `src/native/src/lib.rs`) — both FFI surfaces compose the same core
//! primitives.
//!
//! # Memory model
//!
//! - Constructors return a `u32` tensor id. `u32::MAX` is the error
//!   sentinel (invalid input, null pointer, shape mismatch).
//! - Borrowing functions (`*_sum`, `*_numel`, `*_shape_*`, `*_copy_into`,
//!   `*_data_ptr`) read from the store without mutating ids.
//! - `ml_engine_release` frees the underlying tensor data and returns
//!   the id slot to the store's free list. Calling it with an unknown id
//!   is a no-op (guarded by `TensorStore::free`).
//!
//! # Safety
//!
//! FFI safety is the caller's responsibility:
//! - Pointer arguments must point to at least `len` valid elements.
//! - Ids must come from this library's constructors and not have been
//!   released.
//! - Ops like `ml_engine_matmul` expect shape-compatible operands; the
//!   underlying Rust ops will `panic!` on mismatched shapes.

use mni_framework_core::autograd::Tape;
use mni_framework_core::ops;
use mni_framework_core::tensor::{TensorId, TensorStore};
use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Global engine
// ---------------------------------------------------------------------------

/// Swift-side counterpart to the Node `Engine` — just the pieces this
/// FFI needs. Add fields here (e.g. `int_store`, `kv_caches`) as more of
/// the real API is exposed.
struct Engine {
    store: TensorStore,
    tape: Tape,
}

static ENGINE: OnceLock<Mutex<Engine>> = OnceLock::new();

fn engine() -> &'static Mutex<Engine> {
    ENGINE.get_or_init(|| {
        Mutex::new(Engine {
            store: TensorStore::new(),
            tape: Tape::new(),
        })
    })
}

/// Sentinel id returned when a constructor fails (null pointer, shape
/// mismatch, etc). Swift treats this as "no tensor".
const INVALID_ID: u32 = u32::MAX;

/// Helper: build a shape Vec<usize> from a C pointer + length pair.
/// Returns None if the pointer is null.
unsafe fn shape_from_ptr(shape_ptr: *const u64, shape_len: u64) -> Option<Vec<usize>> {
    if shape_ptr.is_null() {
        return None;
    }
    let raw = std::slice::from_raw_parts(shape_ptr, shape_len as usize);
    Some(raw.iter().map(|&d| d as usize).collect())
}

// ===========================================================================
// FFI: sanity check
// ===========================================================================

/// `a + b` computed in Rust. Pure FFI smoke test, no engine involved.
#[no_mangle]
pub extern "C" fn ml_add(a: i32, b: i32) -> i32 {
    a + b
}

// ===========================================================================
// FFI: engine-backed tensor API
// ===========================================================================

/// Construct a zero tensor of the given shape. Returns `INVALID_ID` on
/// null shape pointer.
///
/// # Safety
/// `shape_ptr` must point to at least `shape_len` valid `u64` values.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_zeros(shape_ptr: *const u64, shape_len: u64) -> u32 {
    let Some(shape) = shape_from_ptr(shape_ptr, shape_len) else {
        return INVALID_ID;
    };
    let mut eng = engine().lock().unwrap();
    eng.store.zeros(&shape) as u32
}

/// Construct a 1-D iota tensor `[0, 1, ..., n-1]`. Convenience helper
/// kept for parity with earlier Swift tests; implemented on top of
/// `TensorStore::from_slice` since the real store has no `iota`.
#[no_mangle]
pub extern "C" fn ml_engine_iota(n: u32) -> u32 {
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let mut eng = engine().lock().unwrap();
    eng.store.from_slice(&data, &[n as usize]) as u32
}

/// Copy caller-owned data into a new tensor of the given shape.
/// Returns `INVALID_ID` on null pointers or size mismatch.
///
/// # Safety
/// `data_ptr` must point to `data_len` valid `f32` values; `shape_ptr`
/// to `shape_len` valid `u64` values.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_from_data(
    data_ptr: *const f32,
    data_len: u64,
    shape_ptr: *const u64,
    shape_len: u64,
) -> u32 {
    if data_ptr.is_null() {
        return INVALID_ID;
    }
    let Some(shape) = shape_from_ptr(shape_ptr, shape_len) else {
        return INVALID_ID;
    };
    let expected: usize = shape.iter().product();
    if expected != data_len as usize {
        return INVALID_ID;
    }
    let data = std::slice::from_raw_parts(data_ptr, data_len as usize);
    let mut eng = engine().lock().unwrap();
    eng.store.from_slice(data, &shape) as u32
}

/// Number of elements in the tensor (product of shape dims).
#[no_mangle]
pub extern "C" fn ml_engine_numel(id: u32) -> u64 {
    let eng = engine().lock().unwrap();
    eng.store.size(id as TensorId) as u64
}

/// Number of dims in the tensor's shape.
#[no_mangle]
pub extern "C" fn ml_engine_shape_len(id: u32) -> u64 {
    let eng = engine().lock().unwrap();
    eng.store.shape(id as TensorId).len() as u64
}

/// Copy the tensor's shape into a caller-provided `u64` buffer. Returns
/// the number of dims actually written (= `min(ndim, out_len)`).
///
/// # Safety
/// `out` must point to writable memory for `out_len` `u64` values.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_shape_copy(id: u32, out: *mut u64, out_len: u64) -> u64 {
    if out.is_null() {
        return 0;
    }
    let eng = engine().lock().unwrap();
    let shape = eng.store.shape(id as TensorId);
    let n = std::cmp::min(shape.len() as u64, out_len);
    for (i, &d) in shape.iter().take(n as usize).enumerate() {
        *out.add(i) = d as u64;
    }
    n
}

/// Sum of all elements. Reduces the raw data buffer — does not touch
/// the autograd tape.
#[no_mangle]
pub extern "C" fn ml_engine_sum(id: u32) -> f32 {
    let eng = engine().lock().unwrap();
    eng.store.data(id as TensorId).iter().sum()
}

/// Copy the tensor's row-major data into a caller-provided `f32` buffer.
/// Returns the element count actually written.
///
/// # Safety
/// `out` must point to writable memory for `out_len` `f32` values.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_copy_into(id: u32, out: *mut f32, out_len: u64) -> u64 {
    if out.is_null() {
        return 0;
    }
    let eng = engine().lock().unwrap();
    let data = eng.store.data(id as TensorId);
    let n = std::cmp::min(data.len() as u64, out_len);
    std::ptr::copy_nonoverlapping(data.as_ptr(), out, n as usize);
    n
}

/// Release the tensor: frees its data, returns the id slot to the store.
/// No-op if the id was already freed.
#[no_mangle]
pub extern "C" fn ml_engine_release(id: u32) {
    let mut eng = engine().lock().unwrap();
    eng.store.free(id as TensorId);
}

// ===========================================================================
// FFI: ops
// ===========================================================================

/// Matrix multiply: `a @ b`. Both operands must be 2-D with compatible
/// inner dims. Returns a new tensor id whose shape is `[a.rows, b.cols]`.
#[no_mangle]
pub extern "C" fn ml_engine_matmul(a: u32, b: u32) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::matmul::matmul(a as TensorId, b as TensorId, store, tape) as u32
}
