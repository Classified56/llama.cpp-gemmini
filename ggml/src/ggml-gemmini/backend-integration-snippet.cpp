// Add near the top of ggml-gemmini.cpp:
#include "ggml-gemmini-runtime.h"

#include <vector>

// The backend-facing adapter owns GGML-specific shape/layout translation.
// The runtime wrapper never sees ggml_tensor.

static bool ggml_gemmini_can_use_runtime_i8(
        const struct ggml_tensor * dst) {
    if (dst == nullptr || dst->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }

    // Synthetic first milestone. Real GGUF inference will need a separate
    // Q8_0 x F32 packing/quantization adapter.
    if (src0->type != GGML_TYPE_I8 ||
        src1->type != GGML_TYPE_I8 ||
        dst->type  != GGML_TYPE_I32) {
        return false;
    }

    if (src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src1->ne[2] != 1 || src1->ne[3] != 1 ||
        dst->ne[2]  != 1 || dst->ne[3]  != 1) {
        return false;
    }

    const std::size_t k = static_cast<std::size_t>(src0->ne[0]);
    const std::size_t n = static_cast<std::size_t>(src0->ne[1]);
    const std::size_t m = static_cast<std::size_t>(src1->ne[1]);

    return src1->ne[0] == src0->ne[0] &&
           dst->ne[0] == src0->ne[1] &&
           dst->ne[1] == src1->ne[1] &&
           k > 0 && n > 0 && m > 0;
}

static bool ggml_gemmini_execute_runtime_i8(
        struct ggml_tensor * dst) {
    if (!ggml_gemmini_can_use_runtime_i8(dst)) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0]; // GGML [K, N]
    const struct ggml_tensor * src1 = dst->src[1]; // GGML [K, M]

    const std::size_t k = static_cast<std::size_t>(src0->ne[0]);
    const std::size_t n = static_cast<std::size_t>(src0->ne[1]);
    const std::size_t m = static_cast<std::size_t>(src1->ne[1]);

    // Initial correctness-first adapter:
    //
    // A[M,K] = transpose of GGML src1[K,M]
    // B[K,N] = logical matrix represented by GGML src0[K,N]
    // C[M,N] = temporary Gemmini output
    //
    // This deliberately handles arbitrary GGML strides by packing.
    std::vector<std::int8_t> a(m * k);
    std::vector<std::int8_t> b(k * n);
    std::vector<std::int32_t> c(m * n);

    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < k; ++col) {
            const auto * ptr = reinterpret_cast<const std::int8_t *>(
                reinterpret_cast<const char *>(src1->data) +
                col * src1->nb[0] +
                row * src1->nb[1]);
            a[row * k + col] = *ptr;
        }
    }

    for (std::size_t row = 0; row < k; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            const auto * ptr = reinterpret_cast<const std::int8_t *>(
                reinterpret_cast<const char *>(src0->data) +
                row * src0->nb[0] +
                col * src0->nb[1]);
            b[row * n + col] = *ptr;
        }
    }

    const ggml_gemmini_matmul_i8_params params = {
        /* .a           = */ a.data(),
        /* .b           = */ b.data(),
        /* .c           = */ c.data(),
        /* .m           = */ m,
        /* .n           = */ n,
        /* .k           = */ k,
        /* .stride_a    = */ k,
        /* .stride_b    = */ n,
        /* .stride_c    = */ n,
        /* .transpose_a = */ false,
        /* .transpose_b = */ false,
    };

    const ggml_gemmini_runtime_status status =
        ggml_gemmini_runtime_matmul_i8(params);

    if (status != GGML_GEMMINI_RUNTIME_SUCCESS) {
        GGML_LOG_DEBUG(
            "ggml-gemmini: runtime matmul declined: %s\n",
            ggml_gemmini_runtime_status_string(status));
        return false;
    }

    for (std::size_t row = 0; row < m; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            auto * ptr = reinterpret_cast<std::int32_t *>(
                reinterpret_cast<char *>(dst->data) +
                col * dst->nb[0] +
                row * dst->nb[1]);
            *ptr = c[row * n + col];
        }
    }

    return true;
}

// In your existing ggml_gemmini_mul_mat(), call this before the reference path:
//
// if (ggml_gemmini_execute_runtime_i8(dst)) {
//     return;
// }
//
// Do not advertise this hardware case in supports_op() until
// ggml_gemmini_runtime_is_available() is true and the kernel writes dst.
