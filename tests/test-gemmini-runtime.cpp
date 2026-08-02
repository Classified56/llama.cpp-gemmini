#include "ggml-gemmini-runtime.h"

#include <cstdint>
#include <cstdio>

int main() {
    const bool available = ggml_gemmini_runtime_is_available();

    std::printf(
        "Gemmini hardware runtime available: %s\n",
        available ? "yes" : "no");

#if !defined(__riscv)
    if (available) {
        std::fprintf(
            stderr,
            "ERROR: Gemmini hardware runtime reported available on x86\n");
        return 1;
    }
#endif

    const std::int8_t a[4] = {
        1, 2,
        3, 4,
    };

    const std::int8_t b[4] = {
        5, 6,
        7, 8,
    };

    std::int32_t c[4] = {};

    const ggml_gemmini_matmul_i8_params params = {
        a,
        b,
        c,

        2, // m
        2, // n
        2, // k

        2, // stride_a
        2, // stride_b
        2, // stride_c

        false, // transpose_a
        false, // transpose_b
    };

    const bool supported =
        ggml_gemmini_runtime_supports_i8_matmul(params);

    const ggml_gemmini_runtime_status status =
        ggml_gemmini_runtime_matmul_i8(params);

    std::printf(
        "I8 matmul supported: %s\n"
        "I8 matmul status: %s\n",
        supported ? "yes" : "no",
        ggml_gemmini_runtime_status_string(status));

#if !defined(__riscv)
    if (supported) {
        std::fprintf(
            stderr,
            "ERROR: Gemmini hardware matmul reported supported on x86\n");
        return 1;
    }

    if (status != GGML_GEMMINI_RUNTIME_UNAVAILABLE) {
        std::fprintf(
            stderr,
            "ERROR: expected runtime unavailable, got: %s\n",
            ggml_gemmini_runtime_status_string(status));
        return 1;
    }
#endif

    std::puts("Gemmini runtime smoke test passed");
    return 0;
}
