#ifndef ML_TRANSCRIBE_FFI_H
#define ML_TRANSCRIBE_FFI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Simple value-returning helpers
// ---------------------------------------------------------------------------

int32_t ml_add(int32_t a, int32_t b);

float ml_tensor_zeros_sum(uint32_t rows, uint32_t cols);
float ml_tensor_iota_sum(uint32_t n);

// ---------------------------------------------------------------------------
// Opaque tensor handles
//
// `MLTensor` is forward-declared and intentionally incomplete: callers may
// only use `MLTensor*` and must never dereference or dig into the struct.
// All operations go through the functions below.
//
// Lifetime: the pointer returned by a constructor must be released exactly
// once via `ml_tensor_free`. Passing a freed pointer to any function is
// undefined behavior.
// ---------------------------------------------------------------------------

typedef struct MLTensor MLTensor;

MLTensor* ml_tensor_zeros(uint32_t rows, uint32_t cols);
MLTensor* ml_tensor_iota(uint32_t n);

// Copy caller-owned data into a new tensor. Returns NULL on size
// mismatch (rows*cols != len) or null input.
MLTensor* ml_tensor_from_data(const float* data, uint64_t len,
                              uint32_t rows, uint32_t cols);

float ml_tensor_sum(const MLTensor* handle);
uint64_t ml_tensor_numel(const MLTensor* handle);

// Copy out: writes up to `out_len` elements into `out`, returns count.
uint64_t ml_tensor_copy_into(const MLTensor* handle,
                             float* out, uint64_t out_len);

// Borrow: returns a read-only pointer to the tensor's internal buffer.
// Valid only until the tensor is freed. DO NOT retain across calls that
// could mutate or free the tensor.
const float* ml_tensor_data_ptr(const MLTensor* handle);

void ml_tensor_free(MLTensor* handle);

#ifdef __cplusplus
}
#endif

#endif /* ML_TRANSCRIBE_FFI_H */
