#ifndef ML_TRANSCRIBE_FFI_H
#define ML_TRANSCRIBE_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Sanity check
// ---------------------------------------------------------------------------

int32_t ml_add(int32_t a, int32_t b);

// ---------------------------------------------------------------------------
// Engine-backed tensor API
//
// All tensors live inside a process-global TensorStore in Rust, keyed by
// a `uint32_t` id. The sentinel `0xFFFFFFFF` (UINT32_MAX) is returned
// from constructors on error (null pointer, shape mismatch).
//
// Lifetime: each id returned from a constructor must be passed back to
// `ml_engine_release` exactly once. Calling release with an unknown id
// is a no-op. The Swift `Tensor` class enforces single-release via
// `deinit`.
// ---------------------------------------------------------------------------

uint32_t ml_engine_zeros(const uint64_t* shape, uint64_t shape_len);
uint32_t ml_engine_iota(uint32_t n);
uint32_t ml_engine_from_data(const float* data, uint64_t data_len,
                             const uint64_t* shape, uint64_t shape_len);

uint64_t ml_engine_numel(uint32_t id);
uint64_t ml_engine_shape_len(uint32_t id);
uint64_t ml_engine_shape_copy(uint32_t id, uint64_t* out, uint64_t out_len);
float    ml_engine_sum(uint32_t id);
uint64_t ml_engine_copy_into(uint32_t id, float* out, uint64_t out_len);

void     ml_engine_release(uint32_t id);

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------

uint32_t ml_engine_matmul(uint32_t a, uint32_t b);

// Elementwise. Shapes must be broadcast-compatible.
uint32_t ml_engine_add(uint32_t a, uint32_t b);
uint32_t ml_engine_mul(uint32_t a, uint32_t b);

// Activations / reductions.
uint32_t ml_engine_relu(uint32_t a);
uint32_t ml_engine_gelu(uint32_t a);
uint32_t ml_engine_sigmoid(uint32_t a);
uint32_t ml_engine_softmax(uint32_t a, int32_t dim);

// Normalization. gamma / beta have shape [C] where C == x.shape.last.
uint32_t ml_engine_layernorm(uint32_t x, uint32_t gamma, uint32_t beta, float eps);
// RMSNorm: y = gamma * x / sqrt(mean(x²) + eps). No mean subtraction,
// no beta. gamma shape [C].
uint32_t ml_engine_rmsnorm(uint32_t x, uint32_t gamma, float eps);

// Scaled dot-product attention. q/k/v are [BH, S, D]; output is same.
// Pass causal=1 for decoder-style masking, 0 for full attention.
uint32_t ml_engine_attention(uint32_t q, uint32_t k, uint32_t v,
                             float scale, int32_t causal);

// Layout: reshape / permute / contiguous.
// - reshape: product(new_shape) must equal numel(a).
// - permute: dims is a permutation of 0..ndim. Result is non-contiguous.
// - contiguous: force row-major memory; needed after permute+reshape.
uint32_t ml_engine_reshape(uint32_t a, const uint64_t* shape, uint64_t shape_len);
uint32_t ml_engine_permute(uint32_t a, const uint64_t* dims, uint64_t dims_len);
uint32_t ml_engine_contiguous(uint32_t a);

// Embedding lookup. weight is [vocab_size, embed_dim]; indices is a
// flat buffer of length (batch * seq_len). Output shape is
// [batch, seq_len, embed_dim]. Each index must be in 0..vocab_size.
uint32_t ml_engine_embedding(uint32_t weight,
                             const uint64_t* indices, uint64_t indices_len,
                             uint64_t batch, uint64_t seq_len);

// Half-split RoPE applied to the last two dims of x (shape [..., S, D]).
// D must be even. start_pos shifts the position of the first token —
// useful when decoding incrementally with a KV cache. base is typically
// 10000.0.
uint32_t ml_engine_rope(uint32_t x, uint64_t start_pos, float base);

// ---------------------------------------------------------------------------
// Tokenizer (HuggingFace tokenizer.json format).
//
// load returns an opaque handle; encode/decode allocate fresh buffers
// on the Rust heap whose ownership is handed to the caller — they must
// be released with the matching free fn.
// ---------------------------------------------------------------------------

typedef struct Tokenizer Tokenizer;

Tokenizer* ml_engine_tokenizer_load(const uint8_t* path, uint64_t path_len);

// Encode: returns a *mut u32 buffer, length written to *out_len.
// Caller must release with tokenizer_free_u32_buffer.
// Returns NULL on error.
uint32_t* ml_engine_tokenizer_encode(const Tokenizer* handle,
                                     const uint8_t* text, uint64_t text_len,
                                     int32_t add_special,
                                     uint64_t* out_len);

// Decode: returns a *mut u8 (UTF-8 bytes) buffer, length in *out_len.
// Caller must release with tokenizer_free_u8_buffer.
uint8_t* ml_engine_tokenizer_decode(const Tokenizer* handle,
                                    const uint32_t* ids, uint64_t ids_len,
                                    int32_t skip_special,
                                    uint64_t* out_len);

void ml_engine_tokenizer_free_u32_buffer(uint32_t* ptr, uint64_t len);
void ml_engine_tokenizer_free_u8_buffer(uint8_t* ptr, uint64_t len);
void ml_engine_tokenizer_free(Tokenizer* handle);

// ---------------------------------------------------------------------------
// KV cache — for incremental decoding.
//
// Handle IDs are NOT TensorIds. Each cache lives in the engine's
// HashMap keyed by this u32. Handle 0 is reserved as "invalid".
// All tensor arguments use shape [B, H, 1, D].
// ---------------------------------------------------------------------------

uint32_t ml_engine_kvcache_new(uint64_t batch_size, uint64_t num_heads,
                               uint64_t head_dim, uint64_t max_seq_len,
                               int32_t quantized);
uint64_t ml_engine_kvcache_len(uint32_t handle);
void     ml_engine_kvcache_reset(uint32_t handle);
int32_t  ml_engine_kvcache_append(uint32_t handle, uint32_t k, uint32_t v);
uint32_t ml_engine_kvcache_append_and_decode(uint32_t handle, uint32_t q,
                                             uint32_t k, uint32_t v,
                                             float scale);
void     ml_engine_kvcache_free(uint32_t handle);

// ---------------------------------------------------------------------------
// Safetensors (F32 only for now)
//
// Load: returns an opaque handle. Names are retrieved by index; ids are
// retrieved by name. All loaded tensors are freed when the handle is.
//
// Save: builder pattern — open, add pairs, finish. Returns 0 on success.
// ---------------------------------------------------------------------------

typedef struct LoadedWeightsHandle LoadedWeightsHandle;
typedef struct SavePlan SavePlan;

LoadedWeightsHandle* ml_engine_safetensors_load(const uint8_t* path, uint64_t path_len);
uint64_t ml_engine_safetensors_count(const LoadedWeightsHandle* h);
uint64_t ml_engine_safetensors_name_len(const LoadedWeightsHandle* h, uint64_t idx);
uint64_t ml_engine_safetensors_name_at(const LoadedWeightsHandle* h, uint64_t idx,
                                       uint8_t* out, uint64_t out_len);
uint32_t ml_engine_safetensors_get(const LoadedWeightsHandle* h,
                                   const uint8_t* name, uint64_t name_len);
void     ml_engine_safetensors_free(LoadedWeightsHandle* h);

SavePlan* ml_engine_safetensors_save_open(void);
int32_t   ml_engine_safetensors_save_add(SavePlan* p, const uint8_t* name, uint64_t name_len,
                                         uint32_t id);
int32_t   ml_engine_safetensors_save_finish(SavePlan* p,
                                            const uint8_t* path, uint64_t path_len);
int32_t   ml_engine_safetensors_save_finish_f16(SavePlan* p,
                                                const uint8_t* path, uint64_t path_len);

#ifdef __cplusplus
}
#endif

#endif /* ML_TRANSCRIBE_FFI_H */
