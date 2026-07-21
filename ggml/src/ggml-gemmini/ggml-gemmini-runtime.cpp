#include "ggml-gemmini-runtime.h"

#include <cstdlib>
#include <cstring>

namespace {

bool runtime_disabled_by_environment() {
    const char * value = std::getenv("GGML_DISABLE_GEMMINI");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

bool dimensions_are_valid(const ggml_gemmini_matmul_i8_params & params) {
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

const ggml_gemmini_runtime_info & ggml_gemmini_runtime_get_info() {
    static const ggml_gemmini_runtime_info info = {
        /* .name                     = */ "Gemmini runtime wrapper",
#if defined(GGML_GEMMINI_RUNTIME_FORCE_AVAILABLE)
        /* .available                = */ true,
#elif defined(GGML_GEMMINI_RUNTIME_HAS_HARDWARE)
        /* .available                = */ true,
#else
        /* .available                = */ false,
#endif
        /* .dim                      = */ 0,
        /* .input_element_size       = */ sizeof(std::int8_t),
        /* .accumulator_element_size = */ sizeof(std::int32_t),
    };

    return info;
}

bool ggml_gemmini_runtime_is_available() {
    return ggml_gemmini_runtime_get_info().available &&
           !runtime_disabled_by_environment();
}

bool ggml_gemmini_runtime_supports_i8_matmul(
        const ggml_gemmini_matmul_i8_params & params) {
    if (!ggml_gemmini_runtime_is_available()) {
        return false;
    }

    if (!dimensions_are_valid(params)) {
        return false;
    }

    // Keep the initial hardware contract simple and explicit. Packing arbitrary
    // GGML views into these row-major layouts is the backend adapter's job.
    if (params.transpose_a || params.transpose_b) {
        return false;
    }

    return true;
}

ggml_gemmini_runtime_status ggml_gemmini_runtime_matmul_i8(
        const ggml_gemmini_matmul_i8_params & params) {
    if (!ggml_gemmini_runtime_is_available()) {
        return GGML_GEMMINI_RUNTIME_UNAVAILABLE;
    }

    if (!dimensions_are_valid(params)) {
        return GGML_GEMMINI_RUNTIME_INVALID_ARGUMENT;
    }

    if (!ggml_gemmini_runtime_supports_i8_matmul(params)) {
        return GGML_GEMMINI_RUNTIME_UNSUPPORTED;
    }

#if defined(GGML_GEMMINI_RUNTIME_HAS_HARDWARE)
    // The real implementation belongs in ggml-gemmini-kernel.cpp.
    //
    // Keeping this return here prevents the backend from falsely reporting
    // successful execution before C has actually been written.
    return GGML_GEMMINI_RUNTIME_EXECUTION_FAILED;
#else
    return GGML_GEMMINI_RUNTIME_UNAVAILABLE;
#endif
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
