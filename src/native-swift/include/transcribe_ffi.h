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
uint32_t ml_engine_softmax(uint32_t a, int32_t dim);

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
