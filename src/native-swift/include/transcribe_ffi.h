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

// Dtype encoding: 0 = F32, 1 = F16. Returns u32::MAX on unknown id.
uint32_t ml_engine_dtype(uint32_t id);
uint32_t ml_engine_cast_dtype(uint32_t id, uint32_t target_dtype);

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
// Self-attention only — Q, K, V must share the same seq_len.
uint32_t ml_engine_attention(uint32_t q, uint32_t k, uint32_t v,
                             float scale, int32_t causal);

// Cross-attention: Q at its own seq_len, K and V at a (typically
// larger) different seq_len. Never causal. Shapes:
//   Q: [BH, S_q, D], K: [BH, S_kv, D], V: [BH, S_kv, D] → [BH, S_q, D].
uint32_t ml_engine_cross_attention(uint32_t q, uint32_t k, uint32_t v,
                                   float scale);

// Layout: reshape / permute / contiguous.
// - reshape: product(new_shape) must equal numel(a).
// - permute: dims is a permutation of 0..ndim. Result is non-contiguous.
// - contiguous: force row-major memory; needed after permute+reshape.
uint32_t ml_engine_reshape(uint32_t a, const uint64_t* shape, uint64_t shape_len);
uint32_t ml_engine_permute(uint32_t a, const uint64_t* dims, uint64_t dims_len);
uint32_t ml_engine_contiguous(uint32_t a);

// PyTorch's repeat_interleave: each slice along `dim` is duplicated
// `repeats` times consecutively. Negative `dim` counts from the end.
// Primary use: GQA broadcasting of K/V heads across Q groups.
uint32_t ml_engine_repeat_interleave(uint32_t a, int32_t dim, uint32_t repeats);

// 1-D convolution, PyTorch convention:
//   input  [N, C_in,  L]
//   weight [C_out, C_in/groups, K]
//   output [N, C_out, L_out] where L_out = (L + 2*padding - K) / stride + 1
// No bias — apply bias via add() separately. groups=1 is a dense
// conv; groups=C_in (with C_out=C_in) is depthwise (used inside
// Conformer's ConvModule).
uint32_t ml_engine_conv1d(uint32_t input, uint32_t weight,
                          uint64_t stride, uint64_t padding, uint64_t groups);

// 2-D convolution, PyTorch convention.
//   input  [N, C_in,  H, W]
//   weight [C_out, C_in/groups, kH, kW]
//   output [N, C_out, H_out, W_out]
// Single (stride, padding) used for both spatial dims. No bias.
uint32_t ml_engine_conv2d(uint32_t input, uint32_t weight,
                          uint64_t stride, uint64_t padding, uint64_t groups);

// 1-D batch normalization, inference mode:
//   y = (x - running_mean) / sqrt(running_var + eps) * gamma + beta
// x is [N, C, L]; running_mean / running_var / gamma / beta are [C].
uint32_t ml_engine_batchnorm1d(uint32_t x, uint32_t running_mean,
                               uint32_t running_var, uint32_t gamma,
                               uint32_t beta, float eps);

// Gated Linear Unit. Splits along `dim` (must have even size); returns
// `first * sigmoid(second)`. `dim` may be negative (counts from end).
// Output shape matches input with the `dim` size halved.
uint32_t ml_engine_glu(uint32_t x, int32_t dim);

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

// Transformer-XL relative-position shift, used inside Conformer's
// `RelPositionMultiHeadAttention`.
//   input  shape: [B, H, T, 2T-1]
//   output shape: [B, H, T, T]
// Output is the gather  out[b,h,i,k] = x[b,h,i, (T-1) - k + i],
// realigning each row's relative-offset columns onto absolute key
// positions. Returns INVALID_ID if the input is not 4-D or the last
// dim is not 2*T-1.
uint32_t ml_engine_rel_shift(uint32_t x);

// ---------------------------------------------------------------------------
// Audio preprocessing.
//
// Configurable log-mel spectrogram. Whisper defaults: sr=16000,
// n_fft=400, hop=160, n_mels=80, normalize_mode=0. NeMo / Cohere
// Transcribe: sr=16000, n_fft=512, hop=160, n_mels=128,
// normalize_mode=1.
//
// normalize_mode:
//   0 = Whisper: clamp [max-8, max], (x+4)/4. Bit-compatible with
//       openai/whisper.
//   1 = per-feature: subtract per-bin mean, divide by per-bin std.
//       Matches NeMo's `normalize=per_feature`.
//   2 = none: raw log10 with 1e-10 floor only.
//
// Output: TensorId of shape [n_mels, n_frames].
// ---------------------------------------------------------------------------

uint32_t ml_engine_log_mel_spectrogram(const float* samples, uint64_t samples_len,
                                       uint32_t sample_rate, uint64_t n_fft,
                                       uint64_t hop_length, uint64_t n_mels,
                                       uint32_t normalize_mode);

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
// Like load(), but F16 source tensors are kept as F16 in storage
// (half memory). F32 and BF16 source dtypes still up-convert to F32.
LoadedWeightsHandle* ml_engine_safetensors_load_keep_f16(const uint8_t* path, uint64_t path_len);
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
