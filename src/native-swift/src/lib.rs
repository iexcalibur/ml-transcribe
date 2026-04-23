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
use mni_framework_core::io::safetensors;
use mni_framework_core::ops;
use mni_framework_core::tensor::{TensorId, TensorStore};
use std::path::PathBuf;
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

/// Elementwise add: `a + b`. Shapes must match (or be broadcast-compatible
/// per the elementwise op's rules).
#[no_mangle]
pub extern "C" fn ml_engine_add(a: u32, b: u32) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::elementwise::add(a as TensorId, b as TensorId, store, tape) as u32
}

/// Elementwise multiply: `a * b`.
#[no_mangle]
pub extern "C" fn ml_engine_mul(a: u32, b: u32) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::elementwise::mul(a as TensorId, b as TensorId, store, tape) as u32
}

/// ReLU activation: `max(0, x)` elementwise.
#[no_mangle]
pub extern "C" fn ml_engine_relu(a: u32) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::activation::relu_forward(a as TensorId, store, tape) as u32
}

/// Softmax along the given dim. Pass a negative dim to count from the
/// end (e.g. `-1` for the last dim, matching PyTorch convention).
#[no_mangle]
pub extern "C" fn ml_engine_softmax(a: u32, dim: i32) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::norm::softmax(a as TensorId, dim, store, tape) as u32
}

// ===========================================================================
// FFI: safetensors load + save
// ===========================================================================
//
// Load flow:
//   1. Swift calls `ml_engine_safetensors_load(path)` -> *mut LoadedWeights.
//   2. Swift iterates names via `safetensors_count` + `safetensors_name_at`.
//   3. Swift looks tensors up via `safetensors_get(handle, name)` -> TensorId.
//      The id lives in the global store; Swift wraps it as a borrowed
//      Tensor that does NOT release on deinit.
//   4. Swift drops its SafetensorsWeights wrapper; Rust's
//      `safetensors_free` releases every loaded id back to the store.
//
// Save flow (used primarily for test fixtures):
//   1. `ml_engine_safetensors_save_open()` -> *mut SavePlan
//   2. `ml_engine_safetensors_save_add(plan, name, id)` per tensor
//   3. `ml_engine_safetensors_save_finish(plan, path)` writes + frees plan

/// Parse UTF-8 path bytes into a PathBuf. Returns None on invalid UTF-8.
unsafe fn path_from_ptr(path_ptr: *const u8, path_len: u64) -> Option<PathBuf> {
    if path_ptr.is_null() {
        return None;
    }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len as usize);
    std::str::from_utf8(bytes).ok().map(PathBuf::from)
}

// --- Load side ---

/// Opaque handle returned by `ml_engine_safetensors_load`. Just boxes
/// the native-core `LoadedWeights`.
pub struct LoadedWeightsHandle(safetensors::LoadedWeights);

/// Load a `.safetensors` file into the engine's store. Returns a boxed
/// handle on success or null on any error (I/O, parse, unsupported dtype).
///
/// # Safety
/// `path_ptr` must be a valid UTF-8 byte buffer of length `path_len`.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_safetensors_load(
    path_ptr: *const u8,
    path_len: u64,
) -> *mut LoadedWeightsHandle {
    let Some(path) = path_from_ptr(path_ptr, path_len) else {
        return std::ptr::null_mut();
    };
    let mut eng = engine().lock().unwrap();
    match safetensors::load_into(&path, &mut eng.store) {
        Ok(loaded) => Box::into_raw(Box::new(LoadedWeightsHandle(loaded))),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Number of tensors in the loaded file.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_safetensors_count(
    handle: *const LoadedWeightsHandle,
) -> u64 {
    if handle.is_null() {
        return 0;
    }
    (*handle).0.len() as u64
}

/// Copy the UTF-8 bytes of the i-th tensor's name into `out`. Returns
/// the number of bytes written, or 0 on invalid input.
///
/// Caller should first call with `out = null`, `out_len = 0` — no,
/// actually the simpler convention: we write up to `out_len` bytes and
/// truncate if the name is longer, so callers must pre-allocate a
/// "reasonable" buffer (256 bytes covers every realistic tensor name).
///
/// # Safety
/// `out` must point to at least `out_len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_safetensors_name_at(
    handle: *const LoadedWeightsHandle,
    idx: u64,
    out: *mut u8,
    out_len: u64,
) -> u64 {
    if handle.is_null() || out.is_null() {
        return 0;
    }
    let Some(name) = (*handle).0.name_at(idx as usize) else {
        return 0;
    };
    let bytes = name.as_bytes();
    let n = std::cmp::min(bytes.len() as u64, out_len);
    std::ptr::copy_nonoverlapping(bytes.as_ptr(), out, n as usize);
    n
}

/// Exact length (in UTF-8 bytes) of the i-th tensor's name. Useful to
/// size the buffer for `safetensors_name_at`.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_safetensors_name_len(
    handle: *const LoadedWeightsHandle,
    idx: u64,
) -> u64 {
    if handle.is_null() {
        return 0;
    }
    (*handle)
        .0
        .name_at(idx as usize)
        .map(|n| n.len() as u64)
        .unwrap_or(0)
}

/// Look up a tensor by name. Returns `INVALID_ID` if the name is not
/// present or the input is invalid.
///
/// # Safety
/// `name_ptr` must be a valid UTF-8 byte buffer of length `name_len`.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_safetensors_get(
    handle: *const LoadedWeightsHandle,
    name_ptr: *const u8,
    name_len: u64,
) -> u32 {
    if handle.is_null() || name_ptr.is_null() {
        return INVALID_ID;
    }
    let bytes = std::slice::from_raw_parts(name_ptr, name_len as usize);
    let Ok(name) = std::str::from_utf8(bytes) else {
        return INVALID_ID;
    };
    (*handle)
        .0
        .get(name)
        .map(|id| id as u32)
        .unwrap_or(INVALID_ID)
}

/// Release every tensor referenced by this handle back to the store,
/// then free the handle itself. Safe to call with null.
///
/// # Safety
/// `handle` must have come from `ml_engine_safetensors_load` and must
/// not have been freed already.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_safetensors_free(
    handle: *mut LoadedWeightsHandle,
) {
    if handle.is_null() {
        return;
    }
    let weights = Box::from_raw(handle);
    let mut eng = engine().lock().unwrap();
    for &id in weights.0.map.values() {
        eng.store.free(id);
    }
}

// --- Save side (for test fixtures) ---

/// Accumulates `(name, id)` pairs for `safetensors::save()`.
pub struct SavePlan {
    entries: Vec<(String, TensorId)>,
}

#[no_mangle]
pub extern "C" fn ml_engine_safetensors_save_open() -> *mut SavePlan {
    Box::into_raw(Box::new(SavePlan {
        entries: Vec::new(),
    }))
}

/// Add one `(name, tensor_id)` pair to a save plan. Returns 0 on
/// success, -1 on invalid input.
///
/// # Safety
/// `plan` must be a live pointer from `save_open`. `name_ptr` must be
/// a valid UTF-8 buffer of length `name_len`.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_safetensors_save_add(
    plan: *mut SavePlan,
    name_ptr: *const u8,
    name_len: u64,
    id: u32,
) -> i32 {
    if plan.is_null() || name_ptr.is_null() {
        return -1;
    }
    let bytes = std::slice::from_raw_parts(name_ptr, name_len as usize);
    let Ok(name) = std::str::from_utf8(bytes) else {
        return -1;
    };
    (*plan).entries.push((name.to_string(), id as TensorId));
    0
}

/// Write the plan to disk as F32 and free it. Returns 0 on success,
/// -1 on any error (invalid plan, bad path, I/O). The plan is freed
/// regardless.
///
/// # Safety
/// `plan` must be a live pointer from `save_open`. `path_ptr` must be
/// a valid UTF-8 buffer of length `path_len`.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_safetensors_save_finish(
    plan: *mut SavePlan,
    path_ptr: *const u8,
    path_len: u64,
) -> i32 {
    finish_plan(plan, path_ptr, path_len, /*as_f16=*/ false)
}

/// Write the plan to disk as F16 (converted from the F32 tensors in
/// the store) and free the plan. Returns 0 on success, -1 on error.
///
/// Useful for producing realistic fp16 model files from tests without
/// needing a Python round-trip.
///
/// # Safety
/// Same as `ml_engine_safetensors_save_finish`.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_safetensors_save_finish_f16(
    plan: *mut SavePlan,
    path_ptr: *const u8,
    path_len: u64,
) -> i32 {
    finish_plan(plan, path_ptr, path_len, /*as_f16=*/ true)
}

unsafe fn finish_plan(
    plan: *mut SavePlan,
    path_ptr: *const u8,
    path_len: u64,
    as_f16: bool,
) -> i32 {
    if plan.is_null() {
        return -1;
    }
    let plan = Box::from_raw(plan);
    let Some(path) = path_from_ptr(path_ptr, path_len) else {
        return -1;
    };
    let eng = engine().lock().unwrap();
    let result = if as_f16 {
        safetensors::save_as_f16(&path, &plan.entries, &eng.store)
    } else {
        safetensors::save(&path, &plan.entries, &eng.store)
    };
    match result {
        Ok(()) => 0,
        Err(_) => -1,
    }
}
