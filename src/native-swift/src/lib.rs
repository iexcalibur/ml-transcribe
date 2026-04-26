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

mod audio;

use mni_framework_core::autograd::Tape;
use mni_framework_core::io::safetensors;
use mni_framework_core::ops;
use mni_framework_core::ops::kv_cache::{KvCache, KvCacheConfig};
use mni_framework_core::tensor::{Dtype, TensorId, TensorStore};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use tokenizers::tokenizer::Tokenizer;

// ---------------------------------------------------------------------------
// Global engine
// ---------------------------------------------------------------------------

/// Swift-side counterpart to the Node `Engine` — just the pieces this
/// FFI needs. Add fields here as more of the real API is exposed.
struct Engine {
    store: TensorStore,
    tape: Tape,
    kv_caches: HashMap<u32, KvCache>,
    next_kv_cache_id: u32,
}

static ENGINE: OnceLock<Mutex<Engine>> = OnceLock::new();

fn engine() -> &'static Mutex<Engine> {
    ENGINE.get_or_init(|| {
        Mutex::new(Engine {
            store: TensorStore::new(),
            tape: Tape::new(),
            kv_caches: HashMap::new(),
            next_kv_cache_id: 1,
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

/// Sum of all elements. Works on both F32 and F16 storage (F16 is
/// up-converted to F32 transiently inside `to_host`).
#[no_mangle]
pub extern "C" fn ml_engine_sum(id: u32) -> f32 {
    let eng = engine().lock().unwrap();
    eng.store.to_host(id as TensorId).iter().sum()
}

/// Copy the tensor's row-major data into a caller-provided `f32`
/// buffer. F16 storage is up-converted on the way out. Returns the
/// element count actually written.
///
/// # Safety
/// `out` must point to writable memory for `out_len` `f32` values.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_copy_into(id: u32, out: *mut f32, out_len: u64) -> u64 {
    if out.is_null() {
        return 0;
    }
    let eng = engine().lock().unwrap();
    let data = eng.store.to_host(id as TensorId);
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

// ---------------------------------------------------------------------------
// Dtype: F16 storage + cast.
//
// Encoded as a `u32` over the FFI: 0 = F32, 1 = F16. (Adding BF16
// later only needs a new code; the wire format stays stable.)
// ---------------------------------------------------------------------------

const DTYPE_F32: u32 = 0;
const DTYPE_F16: u32 = 1;

fn encode_dtype(d: Dtype) -> u32 {
    match d { Dtype::F32 => DTYPE_F32, Dtype::F16 => DTYPE_F16 }
}

fn decode_dtype(d: u32) -> Option<Dtype> {
    match d {
        DTYPE_F32 => Some(Dtype::F32),
        DTYPE_F16 => Some(Dtype::F16),
        _ => None,
    }
}

/// Storage dtype of the tensor at `id`. Returns `u32::MAX` on unknown id.
#[no_mangle]
pub extern "C" fn ml_engine_dtype(id: u32) -> u32 {
    let eng = engine().lock().unwrap();
    encode_dtype(eng.store.dtype(id as TensorId))
}

/// Cast a tensor to a different storage dtype. Returns the new id, or
/// the same id when the source is already in the requested dtype, or
/// `INVALID_ID` if the dtype code is unrecognized.
#[no_mangle]
pub extern "C" fn ml_engine_cast_dtype(id: u32, target_dtype: u32) -> u32 {
    let Some(target) = decode_dtype(target_dtype) else {
        return INVALID_ID;
    };
    let mut eng = engine().lock().unwrap();
    eng.store.cast(id as TensorId, target) as u32
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

/// GELU activation (Gaussian Error Linear Unit) — the smooth ReLU
/// variant used by most transformer FFN layers.
#[no_mangle]
pub extern "C" fn ml_engine_gelu(a: u32) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::activation::gelu_forward(a as TensorId, store, tape) as u32
}

/// Sigmoid: `1 / (1 + exp(-x))` elementwise. Also the building block
/// for SiLU / Swish (`x * sigmoid(x)`), which Swift composes directly.
#[no_mangle]
pub extern "C" fn ml_engine_sigmoid(a: u32) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::activation::sigmoid_forward(a as TensorId, store, tape) as u32
}

/// LayerNorm over the last dim: `y = gamma * (x - mean) / sqrt(var + eps) + beta`.
/// `gamma` and `beta` have shape `[C]` where `C = x.shape.last`.
#[no_mangle]
pub extern "C" fn ml_engine_layernorm(x: u32, gamma: u32, beta: u32, eps: f32) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::norm::layernorm(
        x as TensorId,
        gamma as TensorId,
        beta as TensorId,
        eps,
        store,
        tape,
    ) as u32
}

/// Scaled dot-product attention with optional causal mask. Inputs are
/// `[BH, S, D]` (batch * heads, seq_len, head_dim); output is the
/// same shape. `scale` is typically `1.0 / sqrt(D)`. Set `causal = 1`
/// to mask out positions `j > i` (decoder self-attention); `0` for
/// full attention (encoder / cross-attention).
///
/// Named "flash" in the core because it's computed in a streaming
/// online-softmax manner (no materialized NxN attention matrix), but
/// for Swift callers it behaves as standard SDPA.
///
/// Self-attention only — Q, K, V must all have the same seq_len.
/// For cross-attention with different seq_lens, see
/// `ml_engine_cross_attention`.
#[no_mangle]
pub extern "C" fn ml_engine_attention(
    q: u32,
    k: u32,
    v: u32,
    scale: f32,
    causal: i32,
) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::attention::flash_attention(
        q as TensorId,
        k as TensorId,
        v as TensorId,
        scale,
        causal != 0,
        store,
        tape,
    ) as u32
}

/// Cross-attention: standard scaled dot-product attention where Q
/// has its own seq_len (the decoder's current step or window) and
/// K, V have a different, typically larger seq_len (the encoder's
/// output). Never causal — every Q position can see every K position.
///
/// Shapes:
///   Q: [BH, S_q, D]
///   K: [BH, S_kv, D]    (K and V must agree on S_kv)
///   V: [BH, S_kv, D]
///   out: [BH, S_q, D]
///
/// `scale` is typically `1.0 / sqrt(Float(D))`. Returns `INVALID_ID`
/// on shape mismatch.
///
/// Why this is separate from `ml_engine_attention`: that op's flash
/// kernel assumes Q.seq_len == K.seq_len for the streaming-softmax
/// math. Cross-attention with S_q ≠ S_kv needs a different inner
/// loop, and we don't need its KV-cache machinery (encoder K, V are
/// computed once per inference, never appended).
#[no_mangle]
pub extern "C" fn ml_engine_cross_attention(
    q: u32,
    k: u32,
    v: u32,
    scale: f32,
) -> u32 {
    let mut eng = engine().lock().unwrap();
    let store = &mut eng.store;

    let q_shape = store.shape(q as TensorId).to_vec();
    let k_shape = store.shape(k as TensorId).to_vec();
    let v_shape = store.shape(v as TensorId).to_vec();

    if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
        return INVALID_ID;
    }
    let bh = q_shape[0];
    let s_q = q_shape[1];
    let d = q_shape[2];
    let s_kv = k_shape[1];
    if k_shape[0] != bh || v_shape[0] != bh
       || k_shape[2] != d || v_shape[2] != d
       || v_shape[1] != s_kv
    {
        return INVALID_ID;
    }

    let q_data = store.to_host(q as TensorId);
    let k_data = store.to_host(k as TensorId);
    let v_data = store.to_host(v as TensorId);
    let mut out = vec![0.0f32; bh * s_q * d];

    for b in 0..bh {
        for q_row in 0..s_q {
            // 1. scores[k_row] = (Q · K) * scale, with running max for
            //    numerical stability.
            let mut scores = vec![0.0f32; s_kv];
            let mut max_score = f32::NEG_INFINITY;
            for k_row in 0..s_kv {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q_data[b * s_q * d + q_row * d + dd]
                         * k_data[b * s_kv * d + k_row * d + dd];
                }
                let s = dot * scale;
                scores[k_row] = s;
                if s > max_score {
                    max_score = s;
                }
            }
            // 2. Softmax over scores.
            let mut sum_exp = 0.0f32;
            for k_row in 0..s_kv {
                let e = (scores[k_row] - max_score).exp();
                scores[k_row] = e;
                sum_exp += e;
            }
            // 3. Weighted sum: out[q_row] = Σ softmax[k_row] · V[k_row].
            for dd in 0..d {
                let mut acc = 0.0f32;
                for k_row in 0..s_kv {
                    acc += scores[k_row]
                         * v_data[b * s_kv * d + k_row * d + dd];
                }
                out[b * s_q * d + q_row * d + dd] = acc / sum_exp;
            }
        }
    }

    store.from_slice(&out, &[bh, s_q, d]) as u32
}

// ===========================================================================
// FFI: layout (reshape / permute / contiguous)
// ===========================================================================

/// Reshape: return a view of the tensor with a new shape. Total
/// element count must match the original (`product(new_shape) ==
/// numel(a)`); enforced by the underlying op.
///
/// NOTE: for a permuted (non-contiguous) tensor, the view may not
/// produce what the caller expects. Follow `permute` with
/// `contiguous` before reshaping.
///
/// # Safety
/// `shape_ptr` must point to `shape_len` valid `u64` values.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_reshape(
    a: u32,
    shape_ptr: *const u64,
    shape_len: u64,
) -> u32 {
    let Some(shape) = shape_from_ptr(shape_ptr, shape_len) else {
        return INVALID_ID;
    };
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::layout::view(a as TensorId, &shape, store, tape) as u32
}

/// Permute dimensions: `dims` is a permutation of `0..ndim`. Example:
/// to swap the last two dims of a 4-D tensor, pass `[0, 1, 3, 2]`.
///
/// The result is typically non-contiguous; call `contiguous` before
/// feeding it into ops that require row-major layout (e.g. matmul).
///
/// # Safety
/// `dims_ptr` must point to `dims_len` valid `u64` values forming a
/// valid permutation (caller's responsibility; core will panic on
/// invalid input).
#[no_mangle]
pub unsafe extern "C" fn ml_engine_permute(
    a: u32,
    dims_ptr: *const u64,
    dims_len: u64,
) -> u32 {
    if dims_ptr.is_null() {
        return INVALID_ID;
    }
    let raw = std::slice::from_raw_parts(dims_ptr, dims_len as usize);
    let dims: Vec<usize> = raw.iter().map(|&d| d as usize).collect();
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::layout::permute(a as TensorId, &dims, store, tape) as u32
}

/// Force the tensor into contiguous (row-major) memory layout. No-op
/// if it's already contiguous. Needed after `permute` before `reshape`
/// or any kernel that assumes contiguous data.
#[no_mangle]
pub extern "C" fn ml_engine_contiguous(a: u32) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::layout::contiguous(a as TensorId, store, tape) as u32
}

/// 1-D convolution. PyTorch convention:
/// - `input`  shape `[N, C_in,  L]`
/// - `weight` shape `[C_out, C_in, K]`
/// - output   shape `[N, C_out, L_out]`
///   where `L_out = (L + 2*padding - K) / stride + 1`.
///
/// No bias and no groups (single-group / "dense" conv only). Apply
/// bias separately via `.add(bias)`. Depthwise conv (groups == C_in)
/// will be a follow-up FFI.
#[no_mangle]
pub extern "C" fn ml_engine_conv1d(
    input: u32,
    weight: u32,
    stride: u64,
    padding: u64,
) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::conv::conv1d_forward(
        input as TensorId,
        weight as TensorId,
        stride as usize,
        padding as usize,
        store,
        tape,
    ) as u32
}

/// Repeat each slice along `dim` `repeats` times, consecutively
/// (PyTorch's `torch.repeat_interleave`).
///
/// For input shape `[d0, d1, ..., dn]` and `dim = k`, output shape is
/// `[d0, ..., d_{k-1}, d_k * repeats, d_{k+1}, ..., dn]`.
///
/// Layout: index `i` along `dim` in the input produces indices
/// `[i*repeats, i*repeats + 1, ..., (i+1)*repeats - 1]` in the output.
///
/// Primary use case: GQA expansion. A K or V tensor of shape
/// `[B, H_kv, S, D_h]` becomes `[B, H_kv*G, S, D_h] = [B, H_q, S, D_h]`
/// via `repeat_interleave(dim=1, repeats=G)`, which broadcasts each
/// KV head to the `G = H_q / H_kv` consecutive Q heads it serves.
///
/// Negative `dim` counts from the end, matching PyTorch.
/// Returns `INVALID_ID` on out-of-range `dim` or `repeats == 0`.
#[no_mangle]
pub extern "C" fn ml_engine_repeat_interleave(
    x: u32,
    dim: i32,
    repeats: u32,
) -> u32 {
    if repeats == 0 {
        return INVALID_ID;
    }
    let mut eng = engine().lock().unwrap();
    let x_shape = eng.store.shape(x as TensorId).to_vec();
    let ndim = x_shape.len() as i32;
    let actual_dim = if dim < 0 { ndim + dim } else { dim };
    if actual_dim < 0 || actual_dim as usize >= x_shape.len() {
        return INVALID_ID;
    }
    let dim_i = actual_dim as usize;
    let repeats = repeats as usize;
    let dim_size = x_shape[dim_i];

    let mut new_shape = x_shape.clone();
    new_shape[dim_i] = dim_size * repeats;

    let outer: usize = x_shape[..dim_i].iter().product();
    let inner: usize = x_shape[dim_i + 1..].iter().product();
    let src_slab = dim_size * inner;
    let dst_slab = dim_size * repeats * inner;

    let x_data = eng.store.to_host(x as TensorId);
    let mut out = vec![0.0f32; outer * dst_slab];

    for o in 0..outer {
        for d in 0..dim_size {
            let src_off = o * src_slab + d * inner;
            for r in 0..repeats {
                let dst_off = o * dst_slab + (d * repeats + r) * inner;
                out[dst_off..dst_off + inner]
                    .copy_from_slice(&x_data[src_off..src_off + inner]);
            }
        }
    }

    eng.store.from_slice(&out, &new_shape) as u32
}

// ===========================================================================
// FFI: embedding lookup
// ===========================================================================

/// Gather rows from an embedding table by integer token ids.
///
/// - `weight`: tensor of shape `[vocab_size, embed_dim]`.
/// - `indices_ptr`, `indices_len`: flat `u64` buffer of token ids.
///   Length must equal `batch * seq_len`. Each id must be in
///   `0..vocab_size`; the underlying op panics on out-of-bounds.
/// - Returns a new tensor with shape `[batch, seq_len, embed_dim]`.
///
/// Returns `INVALID_ID` on null pointer or length mismatch (other
/// validation is the caller's responsibility).
///
/// # Safety
/// `indices_ptr` must point to at least `indices_len` valid `u64`s.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_embedding(
    weight: u32,
    indices_ptr: *const u64,
    indices_len: u64,
    batch: u64,
    seq_len: u64,
) -> u32 {
    if indices_ptr.is_null() {
        return INVALID_ID;
    }
    if indices_len != batch * seq_len {
        return INVALID_ID;
    }
    let raw = std::slice::from_raw_parts(indices_ptr, indices_len as usize);
    let indices: Vec<usize> = raw.iter().map(|&i| i as usize).collect();
    let mut eng = engine().lock().unwrap();
    let Engine { store, tape, .. } = &mut *eng;
    ops::embedding::embedding_forward(
        weight as TensorId,
        &indices,
        batch as usize,
        seq_len as usize,
        store,
        tape,
    ) as u32
}

// ===========================================================================
// FFI: RoPE (rotary position embedding)
// ===========================================================================
//
// Half-split convention (LLaMA / HuggingFace style):
//
//   Given x of shape [..., S, D] (D must be even), let half = D/2.
//   For each position p and pair index i in 0..half:
//       freq  = base^(-2i / D)
//       theta = p * freq
//       x1 = x[..., p, i]
//       x2 = x[..., p, i + half]
//       y[..., p, i]        = x1 * cos(theta) - x2 * sin(theta)
//       y[..., p, i + half] = x1 * sin(theta) + x2 * cos(theta)
//
// `start_pos` shifts the effective position of the first token in the
// sequence — needed when RoPE is applied incrementally during KV-cached
// decoding (only the new token is processed; its absolute position is
// `start_pos + 0`).
//
// Shape `[..., S, D]`: any number of leading batch dims is supported.
// In typical use the caller has already split heads, so the shape is
// `[B*H, S, Dh]` where Dh is the head dim.

// ===========================================================================
// FFI: RMSNorm
// ===========================================================================
//
// RMSNorm(x) = gamma * x / sqrt(mean(x²) + eps)
//
// Differs from LayerNorm in two ways:
//   - No mean subtraction (only divides by RMS, keeps the mean intact).
//   - No learnable bias (beta); just a per-feature scale gamma.
//
// Normalizes along the last dim. x: [..., D], gamma: [D], output: [..., D].
//
// Used by LLaMA, Mistral, and most modern decoder-only LMs. Cheaper to
// compute than LayerNorm and empirically just as effective.

/// Apply RMSNorm along the last dim of `x`. `gamma` must have shape
/// `[D]` where `D = x.shape.last`.
/// Returns `INVALID_ID` on shape mismatch or null input.
#[no_mangle]
pub extern "C" fn ml_engine_rmsnorm(x: u32, gamma: u32, eps: f32) -> u32 {
    let mut eng = engine().lock().unwrap();

    let x_shape = eng.store.shape(x as TensorId).to_vec();
    let gamma_shape = eng.store.shape(gamma as TensorId).to_vec();
    if x_shape.is_empty() {
        return INVALID_ID;
    }
    let c = x_shape[x_shape.len() - 1];
    if gamma_shape.len() != 1 || gamma_shape[0] != c {
        return INVALID_ID;
    }

    let x_data = eng.store.to_host(x as TensorId);
    let gamma_data = eng.store.to_host(gamma as TensorId);

    let numel = x_data.len();
    let n = numel / c; // number of rows along the last dim
    let mut out = vec![0.0f32; numel];

    for row in 0..n {
        let off = row * c;
        // Compute mean of squares for this row.
        let mut sum_sq = 0.0f32;
        for j in 0..c {
            sum_sq += x_data[off + j] * x_data[off + j];
        }
        let mean_sq = sum_sq / c as f32;
        let rstd = 1.0 / (mean_sq + eps).sqrt();
        for j in 0..c {
            out[off + j] = x_data[off + j] * rstd * gamma_data[j];
        }
    }

    eng.store.from_slice(&out, &x_shape) as u32
}

/// Apply half-split RoPE to the last two dims of `x`.
/// Returns `INVALID_ID` if `D` is odd or the tensor is fewer than 2-D.
#[no_mangle]
pub extern "C" fn ml_engine_rope(x: u32, start_pos: u64, base: f32) -> u32 {
    let mut eng = engine().lock().unwrap();
    let x_shape = eng.store.shape(x as TensorId).to_vec();
    let ndim = x_shape.len();
    if ndim < 2 {
        return INVALID_ID;
    }
    let s = x_shape[ndim - 2];
    let d = x_shape[ndim - 1];
    if d % 2 != 0 {
        return INVALID_ID;
    }
    let half = d / 2;

    let x_data = eng.store.to_host(x as TensorId);
    let numel = x_data.len();
    let batch_total = numel / (s * d);
    let mut out = vec![0.0f32; numel];

    for b in 0..batch_total {
        for pos in 0..s {
            let abs_pos = (pos + start_pos as usize) as f32;
            for i in 0..half {
                let freq = base.powf(-2.0 * (i as f32) / (d as f32));
                let theta = abs_pos * freq;
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                let base_off = (b * s + pos) * d;
                let x1 = x_data[base_off + i];
                let x2 = x_data[base_off + half + i];
                out[base_off + i] = x1 * cos_t - x2 * sin_t;
                out[base_off + half + i] = x1 * sin_t + x2 * cos_t;
            }
        }
    }

    eng.store.from_slice(&out, &x_shape) as u32
}

// ===========================================================================
// FFI: audio preprocessing (log-mel spectrogram)
// ===========================================================================
//
// Produces a Whisper-compatible log-mel spectrogram from raw audio
// samples. Output is registered as a normal `TensorId` in the engine
// store, so downstream ops (matmul, conv, etc.) consume it like any
// other tensor.

/// Compute a Whisper-style log-mel spectrogram and return the result
/// as a TensorId of shape `[n_mels, n_frames]`.
///
/// Defaults match Whisper exactly: pass `sample_rate=16000`, `n_fft=400`,
/// `hop_length=160`, `n_mels=80`.
///
/// Returns `INVALID_ID` on null `samples_ptr` or zero-length input.
///
/// # Safety
/// `samples_ptr` must point to at least `samples_len` valid `f32`
/// values.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_log_mel_spectrogram(
    samples_ptr: *const f32,
    samples_len: u64,
    sample_rate: u32,
    n_fft: u64,
    hop_length: u64,
    n_mels: u64,
) -> u32 {
    if samples_ptr.is_null() || samples_len == 0 {
        return INVALID_ID;
    }
    let samples = std::slice::from_raw_parts(samples_ptr, samples_len as usize);
    let (data, shape) = audio::log_mel_spectrogram(
        samples,
        sample_rate,
        n_fft as usize,
        hop_length as usize,
        n_mels as usize,
    );
    let mut eng = engine().lock().unwrap();
    eng.store.from_slice(&data, &shape) as u32
}

// ===========================================================================
// FFI: tokenizer (HuggingFace tokenizer.json format)
// ===========================================================================
//
// Backed by the `tokenizers` crate. Supports any tokenizer that can be
// serialized to the standard `tokenizer.json` format (BPE, WordPiece,
// SentencePiece via Unigram, WordLevel, etc).
//
// Memory model: `load` returns an opaque `*mut Tokenizer`. Encode and
// decode allocate a fresh buffer on the Rust heap, return a raw pointer
// and length, and leave it to the caller to call the matching free fn
// after copying the data out. Swift's `Tokenizer` class does this
// pairing automatically.

/// Load a tokenizer from a JSON file. Returns null on any error.
///
/// # Safety
/// `path_ptr` must be a valid UTF-8 byte buffer of length `path_len`.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_tokenizer_load(
    path_ptr: *const u8,
    path_len: u64,
) -> *mut Tokenizer {
    let Some(path) = path_from_ptr(path_ptr, path_len) else {
        return std::ptr::null_mut();
    };
    match Tokenizer::from_file(&path) {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Encode `text` to a fresh `Vec<u32>` of token ids. On success, writes
/// the length into `*out_len` and returns a leaked pointer the caller
/// must free with `ml_engine_tokenizer_free_u32_buffer`. On error
/// returns null and sets `*out_len = 0`.
///
/// # Safety
/// `handle` must be a live pointer from `ml_engine_tokenizer_load`.
/// `text_ptr` must be valid UTF-8 of length `text_len`. `out_len` must
/// point to a writable `u64`.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_tokenizer_encode(
    handle: *const Tokenizer,
    text_ptr: *const u8,
    text_len: u64,
    add_special: i32,
    out_len: *mut u64,
) -> *mut u32 {
    if handle.is_null() || text_ptr.is_null() || out_len.is_null() {
        if !out_len.is_null() { *out_len = 0; }
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(text_ptr, text_len as usize);
    let Ok(text) = std::str::from_utf8(bytes) else {
        *out_len = 0;
        return std::ptr::null_mut();
    };
    let tokenizer = &*handle;
    let encoding = match tokenizer.encode(text, add_special != 0) {
        Ok(e) => e,
        Err(_) => {
            *out_len = 0;
            return std::ptr::null_mut();
        }
    };
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    let len = ids.len();
    *out_len = len as u64;
    let boxed: Box<[u32]> = ids.into_boxed_slice();
    Box::leak(boxed).as_mut_ptr()
}

/// Decode token ids back to a UTF-8 string. Returns a leaked `*mut u8`
/// buffer that the caller must free with
/// `ml_engine_tokenizer_free_u8_buffer`.
///
/// # Safety
/// See `encode`. `ids_ptr` must point to `ids_len` valid `u32`s.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_tokenizer_decode(
    handle: *const Tokenizer,
    ids_ptr: *const u32,
    ids_len: u64,
    skip_special: i32,
    out_len: *mut u64,
) -> *mut u8 {
    if handle.is_null() || ids_ptr.is_null() || out_len.is_null() {
        if !out_len.is_null() { *out_len = 0; }
        return std::ptr::null_mut();
    }
    let ids = std::slice::from_raw_parts(ids_ptr, ids_len as usize);
    let tokenizer = &*handle;
    let text = match tokenizer.decode(ids, skip_special != 0) {
        Ok(s) => s,
        Err(_) => {
            *out_len = 0;
            return std::ptr::null_mut();
        }
    };
    let bytes: Box<[u8]> = text.into_bytes().into_boxed_slice();
    *out_len = bytes.len() as u64;
    Box::leak(bytes).as_mut_ptr()
}

/// Free a `*mut u32` buffer previously returned by `tokenizer_encode`.
///
/// # Safety
/// `ptr` must have come from `ml_engine_tokenizer_encode` and the
/// `len` must match the value returned in `*out_len` at that call.
/// Calling with a null pointer is safe (no-op).
#[no_mangle]
pub unsafe extern "C" fn ml_engine_tokenizer_free_u32_buffer(ptr: *mut u32, len: u64) {
    if ptr.is_null() {
        return;
    }
    let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len as usize));
}

/// Free a `*mut u8` buffer previously returned by `tokenizer_decode`.
///
/// # Safety
/// See `tokenizer_free_u32_buffer`; same contract but for byte buffers.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_tokenizer_free_u8_buffer(ptr: *mut u8, len: u64) {
    if ptr.is_null() {
        return;
    }
    let _ = Box::from_raw(std::slice::from_raw_parts_mut(ptr, len as usize));
}

/// Free the tokenizer. Safe to call with null.
///
/// # Safety
/// `handle` must be a live pointer from `tokenizer_load`, not already
/// freed.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_tokenizer_free(handle: *mut Tokenizer) {
    if handle.is_null() {
        return;
    }
    drop(Box::from_raw(handle));
}

// ===========================================================================
// FFI: KV cache
// ===========================================================================
//
// Used for efficient autoregressive decoding. Each attention layer
// creates one cache; each decode step calls `append_and_decode` which
// appends the new K/V and returns attention output against all past
// keys/values.
//
// Expected tensor shapes: `[B, H, S=1, D]` (single-token step). The
// underlying op is `ops::kv_cache::KvCache`; we store instances in the
// Engine's HashMap keyed by a u32 handle, the same pattern the Node
// side uses.
//
// Sentinel values:
//   - handle == 0        → invalid handle (0 is reserved; ids start at 1)
//   - return i32 < 0     → error (shape mismatch, overflow, etc.)
//   - return u32::MAX    → error on a function returning TensorId

const INVALID_KV_HANDLE: u32 = 0;

/// Allocate a new KV cache. Returns an opaque u32 handle (not a
/// TensorId — do not confuse the two). Returns 0 on allocation error
/// (should not happen in practice).
#[no_mangle]
pub extern "C" fn ml_engine_kvcache_new(
    batch_size: u64,
    num_heads: u64,
    head_dim: u64,
    max_seq_len: u64,
    quantized: i32,
) -> u32 {
    let cfg = KvCacheConfig {
        batch_size: batch_size as usize,
        num_heads: num_heads as usize,
        head_dim: head_dim as usize,
        max_seq_len: max_seq_len as usize,
        quantized: quantized != 0,
    };
    let mut eng = engine().lock().unwrap();
    let handle = eng.next_kv_cache_id;
    eng.next_kv_cache_id = eng.next_kv_cache_id.wrapping_add(1);
    if eng.next_kv_cache_id == INVALID_KV_HANDLE {
        eng.next_kv_cache_id = 1;
    }
    eng.kv_caches.insert(handle, KvCache::new(cfg));
    handle
}

/// Current fill level of the cache. Returns 0 on unknown handle.
#[no_mangle]
pub extern "C" fn ml_engine_kvcache_len(handle: u32) -> u64 {
    let eng = engine().lock().unwrap();
    eng.kv_caches
        .get(&handle)
        .map(|c| c.len() as u64)
        .unwrap_or(0)
}

/// Zero out the cache, setting length back to 0. No-op on unknown handle.
#[no_mangle]
pub extern "C" fn ml_engine_kvcache_reset(handle: u32) {
    let mut eng = engine().lock().unwrap();
    if let Some(c) = eng.kv_caches.get_mut(&handle) {
        c.reset();
    }
}

/// Append a single step of key and value to the cache.
/// Returns 0 on success, -1 on unknown handle, -2 on shape error.
/// `k` and `v` must each have shape `[B, H, 1, D]` matching the
/// cache's config.
#[no_mangle]
pub extern "C" fn ml_engine_kvcache_append(handle: u32, k: u32, v: u32) -> i32 {
    let mut eng = engine().lock().unwrap();
    // Borrow the cache and store simultaneously via split borrows.
    let Engine { store, kv_caches, .. } = &mut *eng;
    let Some(cache) = kv_caches.get_mut(&handle) else {
        return -1;
    };
    match cache.append(k as TensorId, v as TensorId, store) {
        Ok(()) => 0,
        Err(_) => -2,
    }
}

/// Append K/V and compute attention in one call. Returns a new
/// TensorId with the attention output for the single query step, or
/// `u32::MAX` on any error (unknown handle, shape mismatch, overflow).
#[no_mangle]
pub extern "C" fn ml_engine_kvcache_append_and_decode(
    handle: u32,
    q: u32,
    k: u32,
    v: u32,
    scale: f32,
) -> u32 {
    let mut eng = engine().lock().unwrap();
    let Engine { store, kv_caches, .. } = &mut *eng;
    let Some(cache) = kv_caches.get_mut(&handle) else {
        return INVALID_ID;
    };
    match cache.append_and_decode(
        q as TensorId,
        k as TensorId,
        v as TensorId,
        scale,
        store,
    ) {
        Ok(out) => out as u32,
        Err(_) => INVALID_ID,
    }
}

/// Free the cache and release its storage. Safe to call with unknown
/// handle (no-op).
#[no_mangle]
pub extern "C" fn ml_engine_kvcache_free(handle: u32) {
    let mut eng = engine().lock().unwrap();
    eng.kv_caches.remove(&handle);
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
    safetensors_load_inner(path_ptr, path_len, /*keep_f16=*/ false)
}

/// Like `ml_engine_safetensors_load`, but F16 source tensors are kept
/// as F16 in storage (half the resident memory) instead of being
/// up-converted to F32. F32 and BF16 source dtypes are still
/// up-converted as before.
///
/// # Safety
/// Same as `ml_engine_safetensors_load`.
#[no_mangle]
pub unsafe extern "C" fn ml_engine_safetensors_load_keep_f16(
    path_ptr: *const u8,
    path_len: u64,
) -> *mut LoadedWeightsHandle {
    safetensors_load_inner(path_ptr, path_len, /*keep_f16=*/ true)
}

unsafe fn safetensors_load_inner(
    path_ptr: *const u8,
    path_len: u64,
    keep_f16: bool,
) -> *mut LoadedWeightsHandle {
    let Some(path) = path_from_ptr(path_ptr, path_len) else {
        return std::ptr::null_mut();
    };
    let mut eng = engine().lock().unwrap();
    let result = safetensors::load_into_with_dtype(&path, &mut eng.store, keep_f16);
    match result {
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
