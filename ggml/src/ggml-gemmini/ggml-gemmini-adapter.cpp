#include "ggml-gemmini-adapter.h"

#include "ggml-gemmini-kernel.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

static bool is_2d_matrix(const struct ggml_tensor * t) {
    return t != nullptr &&
           ggml_is_matrix(t) &&
           t->ne[2] == 1 &&
           t->ne[3] == 1;
}

static bool mul_mat_shape_ok(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (!is_2d_matrix(src0) ||
        !is_2d_matrix(src1) ||
        !is_2d_matrix(op)) {
        return false;
    }

    // GGML MUL_MAT:
    //   src0: [K, N]
    //   src1: [K, M]
    //   dst:  [N, M]
    const int64_t K0 = src0->ne[0];
    const int64_t N  = src0->ne[1];
    const int64_t K1 = src1->ne[0];
    const int64_t M  = src1->ne[1];

    return K0 == K1 &&
           op->ne[0] == N &&
           op->ne[1] == M;
}

static bool adapter_types_ok(const struct ggml_tensor * op) {
    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (op->type != GGML_TYPE_F32 ||
        src1->type != GGML_TYPE_F32) {
        return false;
    }

    if (src0->type != GGML_TYPE_F32 &&
        src0->type != GGML_TYPE_Q8_0) {
        return false;
    }

    // Q8_0 rows must consist of complete quantization blocks.
    if (src0->type == GGML_TYPE_Q8_0) {
        const int64_t block = ggml_blck_size(GGML_TYPE_Q8_0);
        if (block <= 0 || (src0->ne[0] % block) != 0) {
            return false;
        }
    }

    return true;
}

static float quantize_row_symmetric_i8(
        const float * src,
        int8_t * dst,
        size_t K) {
    float max_abs = 0.0f;

    for (size_t k = 0; k < K; ++k) {
        max_abs = std::max(max_abs, std::fabs(src[k]));
    }

    if (max_abs == 0.0f) {
        std::memset(dst, 0, K * sizeof(*dst));
        return 0.0f;
    }

    // Symmetric signed quantization. Deliberately use [-127, 127], not -128,
    // so one row scale applies symmetrically around zero.
    const float scale = max_abs / 127.0f;
    const float inv_scale = 1.0f / scale;

    for (size_t k = 0; k < K; ++k) {
        long q = std::lround(src[k] * inv_scale);
        q = std::max<long>(-127, std::min<long>(127, q));
        dst[k] = static_cast<int8_t>(q);
    }

    return scale;
}

static const uint8_t * row_bytes(
        const struct ggml_tensor * t,
        int64_t row) {
    return reinterpret_cast<const uint8_t *>(t->data) +
           static_cast<size_t>(row) * t->nb[1];
}

static uint8_t * row_bytes(
        struct ggml_tensor * t,
        int64_t row) {
    return reinterpret_cast<uint8_t *>(t->data) +
           static_cast<size_t>(row) * t->nb[1];
}

static bool pack_f32_weights(
        const struct ggml_tensor * src0,
        std::vector<int8_t> & q_weights,
        std::vector<float> & weight_scales) {
    const size_t K = static_cast<size_t>(src0->ne[0]);
    const size_t N = static_cast<size_t>(src0->ne[1]);

    for (size_t n = 0; n < N; ++n) {
        const float * row =
            reinterpret_cast<const float *>(row_bytes(src0, n));

        weight_scales[n] = quantize_row_symmetric_i8(
            row,
            q_weights.data() + n * K,
            K);
    }

    return true;
}

static bool pack_q8_0_weights(
        const struct ggml_tensor * src0,
        std::vector<int8_t> & q_weights,
        std::vector<float> & weight_scales) {
    const size_t K = static_cast<size_t>(src0->ne[0]);
    const size_t N = static_cast<size_t>(src0->ne[1]);

    std::vector<float> dequant(K);

    for (size_t n = 0; n < N; ++n) {
        const block_q8_0 * row =
            reinterpret_cast<const block_q8_0 *>(row_bytes(src0, n));

        // Use GGML's own Q8_0 decoder so this stays tied to the GGUF/GGML
        // format rather than duplicating the block representation here.
        dequantize_row_q8_0(row, dequant.data(), static_cast<int64_t>(K));

        weight_scales[n] = quantize_row_symmetric_i8(
            dequant.data(),
            q_weights.data() + n * K,
            K);
    }

    return true;
}

static bool pack_f32_activations(
        const struct ggml_tensor * src1,
        std::vector<int8_t> & q_activations,
        std::vector<float> & activation_scales) {
    const size_t K = static_cast<size_t>(src1->ne[0]);
    const size_t M = static_cast<size_t>(src1->ne[1]);

    for (size_t m = 0; m < M; ++m) {
        const float * row =
            reinterpret_cast<const float *>(row_bytes(src1, m));

        activation_scales[m] = quantize_row_symmetric_i8(
            row,
            q_activations.data() + m * K,
            K);
    }

    return true;
}

} // namespace

bool ggml_gemmini_adapter_supports(const struct ggml_tensor * op) {
#if !defined(GGML_GEMMINI_ENABLE_TILED_MATMUL)
    (void) op;
    return false;
#else
    if (!mul_mat_shape_ok(op) || !adapter_types_ok(op)) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    // First implementation stays deliberately narrow. Views, permutations,
    // broadcasting and batched dimensions should fall back to CPU.
    if (!ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(op)) {
        return false;
    }

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];

    // Keep decode/small GEMVs on CPU initially. This also gives offload_op()
    // a clean prefill-oriented threshold.
    if (K < 16 || N < 16 || M < 16) {
        return false;
    }

    return true;
#endif
}

bool ggml_gemmini_adapter_compute(struct ggml_tensor * dst) {
#if !defined(GGML_GEMMINI_ENABLE_TILED_MATMUL)
    (void) dst;
    return false;
#else
    if (!ggml_gemmini_adapter_supports(dst)) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0]; // weights [K,N]
    const struct ggml_tensor * src1 = dst->src[1]; // activations [K,M]

    const size_t K = static_cast<size_t>(src0->ne[0]);
    const size_t N = static_cast<size_t>(src0->ne[1]);
    const size_t M = static_cast<size_t>(src1->ne[1]);

    // Correctness-first implementation.
    //
    // NOTE: model weights are repacked every invocation here. Once the path is
    // validated on a real model, cache q_weights + weight_scales per weight
    // tensor so only activations are dynamically quantized.
    std::vector<int8_t>  q_weights(N * K);
    std::vector<int8_t>  q_activations(M * K);
    std::vector<int32_t> acc(M * N);
    std::vector<float>   weight_scales(N);
    std::vector<float>   activation_scales(M);

    bool packed = false;

    printf("ggml-gemmini: ADAPTER HW EXEC "
                    "src0=%s K=%zu N=%zu M=%zu\n",
                    ggml_type_name(src0->type),
                    K,
                    N,
                    M);

    GGML_LOG_DEBUG(
        "GEMMINI HW: %s x F32 -> F32 "
        "M=%zu N=%zu K=%zu\n",
        ggml_type_name(src0->type),
        M, N, K);

    switch (src0->type) {
        case GGML_TYPE_F32:
            packed = pack_f32_weights(src0, q_weights, weight_scales);
            break;

        case GGML_TYPE_Q8_0:
            packed = pack_q8_0_weights(src0, q_weights, weight_scales);
            break;

        default:
            return false;
    }

    if (!packed ||
        !pack_f32_activations(src1, q_activations, activation_scales)) {
        return false;
    }

    // Verified hardware primitive:
    //
    //   q_weights     = N rows x K   (GGML src0 storage)
    //   q_activations = M rows x K   (GGML src1 storage)
    //   acc           = M rows x N
    //
    // The Gemmini wrapper uses transpose_B internally to compute:
    //   C[M,N] = activations[M,K] * weights[N,K]^T
    if (!ggml_gemmini_mul_mat_i8_i32(
            q_weights.data(),
            q_activations.data(),
            acc.data(),
            K,
            N,
            M)) {
        return false;
    }

    // Convert the accumulator back to GGML's public F32 MUL_MAT result.
    for (size_t m = 0; m < M; ++m) {
        float * out =
            reinterpret_cast<float *>(row_bytes(dst, m));

        const float s_act = activation_scales[m];

        for (size_t n = 0; n < N; ++n) {
            out[n] =
                static_cast<float>(acc[m * N + n]) *
                s_act *
                weight_scales[n];
        }
    }

    return true;
#endif
}
