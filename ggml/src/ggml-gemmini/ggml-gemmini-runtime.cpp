#include "ggml-gemmini-runtime.h"

#include "ggml-gemmini-kernel.h"

namespace {

bool params_valid(const ggml_gemmini_matmul_i8_params & params) {
    if (params.a == nullptr || params.b == nullptr || params.c == nullptr) {
        return false;
    }

    if (params.m == 0 || params.n == 0 || params.k == 0) {
        return false;
    }

    if (params.stride_a < params.k ||
        params.stride_b < params.n ||
        params.stride_c < params.n) {
        return false;
    }

    return true;
}

} // namespace

bool ggml_gemmini_runtime_is_available() {
    return ggml_gemmini_kernel_is_available();
}

bool ggml_gemmini_runtime_supports_i8_matmul(
        const ggml_gemmini_matmul_i8_params & params) {
    if (!ggml_gemmini_runtime_is_available() || !params_valid(params)) {
        return false;
    }

    // The first wrapper accepts packed, non-transposed row-major inputs.
    // The GGML-facing adapter is responsible for packing views/strides.
    return !params.transpose_a && !params.transpose_b;
}

ggml_gemmini_runtime_status ggml_gemmini_runtime_matmul_i8(
        const ggml_gemmini_matmul_i8_params & params) {
    if (!ggml_gemmini_runtime_is_available()) {
        return GGML_GEMMINI_RUNTIME_UNAVAILABLE;
    }

    if (!params_valid(params)) {
        return GGML_GEMMINI_RUNTIME_INVALID_ARGUMENT;
    }

    if (!ggml_gemmini_runtime_supports_i8_matmul(params)) {
        return GGML_GEMMINI_RUNTIME_UNSUPPORTED;
    }

    return ggml_gemmini_kernel_matmul_i8(params);
}

const char * ggml_gemmini_runtime_status_string(
        ggml_gemmini_runtime_status status) {
    switch (status) {
        case GGML_GEMMINI_RUNTIME_SUCCESS:
            return "success";
        case GGML_GEMMINI_RUNTIME_UNAVAILABLE:
            return "runtime unavailable";
        case GGML_GEMMINI_RUNTIME_UNSUPPORTED:
            return "operation unsupported";
        case GGML_GEMMINI_RUNTIME_INVALID_ARGUMENT:
            return "invalid argument";
        case GGML_GEMMINI_RUNTIME_EXECUTION_FAILED:
            return "execution failed";
    }

    return "unknown status";
}
