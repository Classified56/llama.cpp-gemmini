#pragma once

#include <cstddef>
#include <cstdint>

// Internal Gemmini runtime abstraction.
//
// This header intentionally does not include ggml headers or Gemmini software
// headers. The GGML backend translates tensors into this simple interface;
// the hardware implementation translates this interface into Gemmini calls.

enum ggml_gemmini_runtime_status {
    GGML_GEMMINI_RUNTIME_SUCCESS = 0,
    GGML_GEMMINI_RUNTIME_UNAVAILABLE,
    GGML_GEMMINI_RUNTIME_UNSUPPORTED,
    GGML_GEMMINI_RUNTIME_INVALID_ARGUMENT,
    GGML_GEMMINI_RUNTIME_EXECUTION_FAILED,
};

struct ggml_gemmini_runtime_info {
    const char * name;
    bool available;

    // Initial hardware assumptions. These can later be populated from
    // gemmini_params.h or compile-time definitions.
    std::size_t dim;
    std::size_t input_element_size;
    std::size_t accumulator_element_size;
};

struct ggml_gemmini_matmul_i8_params {
    // Dense row-major matrices:
    //
    //   C[M, N] = A[M, K] * B[K, N]
    //
    // The first wrapper deliberately requires packed buffers. GGML tensor
    // strides, views, transposes, and quantization metadata stay outside this
    // layer.
    const std::int8_t * a;
    const std::int8_t * b;
    std::int32_t * c;

    std::size_t m;
    std::size_t n;
    std::size_t k;

    // Row strides measured in elements, not bytes.
    std::size_t stride_a;
    std::size_t stride_b;
    std::size_t stride_c;

    // Reserved for later Gemmini options.
    bool transpose_a;
    bool transpose_b;
};

const ggml_gemmini_runtime_info & ggml_gemmini_runtime_get_info();

bool ggml_gemmini_runtime_is_available();

bool ggml_gemmini_runtime_supports_i8_matmul(
    const ggml_gemmini_matmul_i8_params & params);

ggml_gemmini_runtime_status ggml_gemmini_runtime_matmul_i8(
    const ggml_gemmini_matmul_i8_params & params);

const char * ggml_gemmini_runtime_status_string(
    ggml_gemmini_runtime_status status);
