#include "ggml-gemmini-kernel.h"

#include <cstdlib>
#include <cstring>
#include <type_traits>

#if defined(GGML_GEMMINI_RUNTIME_HAS_HARDWARE)
extern "C" {
#include "include/gemmini.h"
}

// This wrapper is intentionally I8 x I8 -> I32. Fail at compile time if the
// bundled/overridden generated Gemmini configuration does not match it.
static_assert(std::is_same<elem_t, std::int8_t>::value,
              "Gemmini elem_t must be int8_t for the initial GGML wrapper");
static_assert(std::is_same<acc_t, std::int32_t>::value,
              "Gemmini acc_t must be int32_t for the initial GGML wrapper");
#endif

#if defined(__GNUC__) || defined(__clang__)
#define GGML_GEMMINI_NOINLINE __attribute__((noinline))
#define GGML_GEMMINI_USED     __attribute__((used))
#else
#define GGML_GEMMINI_NOINLINE
#define GGML_GEMMINI_USED
#endif

bool ggml_gemmini_kernel_is_available() {
#if defined(GGML_GEMMINI_RUNTIME_HAS_HARDWARE) && defined(__riscv)
    const char * disabled = std::getenv("GGML_DISABLE_GEMMINI");
    return disabled == nullptr || std::strcmp(disabled, "1") != 0;
#else
    return false;
#endif
}

GGML_GEMMINI_NOINLINE
GGML_GEMMINI_USED
ggml_gemmini_runtime_status ggml_gemmini_kernel_matmul_i8(const ggml_gemmini_matmul_i8_params & params) {
#if !defined(GGML_GEMMINI_RUNTIME_HAS_HARDWARE)
    (void) params;
    return GGML_GEMMINI_RUNTIME_UNAVAILABLE;
#else
    // This is now the only function that needs the Gemmini API.
    //
    // Future implementation:
    //
    // tiled_matmul_auto(
    //     params.m, params.n, params.k,
    //     reinterpret_cast<const elem_t *>(params.a),
    //     reinterpret_cast<const elem_t *>(params.b),
    //     nullptr,
    //     reinterpret_cast<void *>(params.c),
    //     params.stride_a,
    //     params.stride_b,
    //     params.stride_c,   // ignored with D == nullptr, but keep valid
    //     params.stride_c,
    //     MVIN_SCALE_IDENTITY,
    //     MVIN_SCALE_IDENTITY,
    //     MVIN_SCALE_IDENTITY,
    //     NO_ACTIVATION,
    //     ACC_SCALE_IDENTITY,
    //     0,
    //     false,
    //     false,
    //     false,
    //     true,             // full_C: write acc_t / int32 output
    //     false,            // low_D
    //     0,
    //     WS);
    //
    // gemmini_fence();
    // return GGML_GEMMINI_RUNTIME_SUCCESS;

    (void) params;
    return GGML_GEMMINI_RUNTIME_EXECUTION_FAILED;
#endif
}
