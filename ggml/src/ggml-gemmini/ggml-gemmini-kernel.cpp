#include "ggml-gemmini-kernel.h"

#if defined(GGML_GEMMINI_RUNTIME_HAS_HARDWARE)
extern "C" {
#include "gemmini.h"
}
#endif

#if defined(__GNUC__) || defined(__clang__)
#define GGML_GEMMINI_NOINLINE __attribute__((noinline))
#define GGML_GEMMINI_USED     __attribute__((used))
#else
#define GGML_GEMMINI_NOINLINE
#define GGML_GEMMINI_USED
#endif

GGML_GEMMINI_NOINLINE
GGML_GEMMINI_USED
ggml_gemmini_runtime_status ggml_gemmini_kernel_matmul_i8(
        const ggml_gemmini_matmul_i8_params & params) {
#if !defined(GGML_GEMMINI_RUNTIME_HAS_HARDWARE)
    (void) params;
    return GGML_GEMMINI_RUNTIME_UNAVAILABLE;
#else
    // Future implementation:
    //
    // tiled_matmul_auto(
    //     params.m,
    //     params.n,
    //     params.k,
    //     reinterpret_cast<const elem_t *>(params.a),
    //     reinterpret_cast<const elem_t *>(params.b),
    //     nullptr,
    //     reinterpret_cast<acc_t *>(params.c),
    //     params.stride_a,
    //     params.stride_b,
    //     params.stride_c,
    //     ... exact arguments from this repository's gemmini.h ...
    // );
    //
    // gemmini_fence();
    //
    // return GGML_GEMMINI_RUNTIME_SUCCESS;

    (void) params;
    return GGML_GEMMINI_RUNTIME_EXECUTION_FAILED;
#endif
}
