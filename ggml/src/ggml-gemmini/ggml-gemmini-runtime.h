#pragma once

#include <cstddef>
#include <cstdint>

enum ggml_gemmini_runtime_status {
    GGML_GEMMINI_RUNTIME_SUCCESS = 0,
    GGML_GEMMINI_RUNTIME_UNAVAILABLE,
    GGML_GEMMINI_RUNTIME_UNSUPPORTED,
    GGML_GEMMINI_RUNTIME_INVALID_ARGUMENT,
    GGML_GEMMINI_RUNTIME_EXECUTION_FAILED,
};

struct ggml_gemmini_matmul_i8_params {
    // Dense row-major matrices:
    //     C[M,N] = A[M,K] * B[K,N]
    const std::int8_t * a;
    const std::int8_t * b;
    std::int32_t * c;

    std::size_t m;
    std::size_t n;
    std::size_t k;

    // Row strides in elements, not bytes.
    std::size_t stride_a;
    std::size_t stride_b;
    std::size_t stride_c;

    bool transpose_a;
    bool transpose_b;
};

bool ggml_gemmini_runtime_is_available();

bool ggml_gemmini_runtime_supports_i8_matmul(
    const ggml_gemmini_matmul_i8_params & params);

ggml_gemmini_runtime_status ggml_gemmini_runtime_matmul_i8(
    const ggml_gemmini_matmul_i8_params & params);

const char * ggml_gemmini_runtime_status_string(
    ggml_gemmini_runtime_status status);
