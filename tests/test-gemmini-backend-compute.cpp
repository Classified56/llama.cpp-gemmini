#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-gemmini.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

struct test_graph {
    ggml_context * ctx;
    ggml_cgraph  * graph;
    ggml_tensor  * src0;
    ggml_tensor  * src1;
    ggml_tensor  * dst;
};

static test_graph build_graph(int64_t K, int64_t N, int64_t M) {
    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        fail("ggml_init() failed");
    }

    // GGML MUL_MAT convention:
    //
    //   src0: [K, N] -> memory is N rows of K int8 values
    //   src1: [K, M] -> memory is M rows of K int8 values
    //   dst:  [N, M] -> memory is M rows of N int32 values
    //
    // Each dst[m,n] is the dot product:
    //
    //   sum_k src1[m,k] * src0[n,k]
    //
    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, GGML_TYPE_I8,  K, N);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, GGML_TYPE_I8,  K, M);
    ggml_tensor * dst  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N, M);

    if (!src0 || !src1 || !dst) {
        ggml_free(ctx);
        fail("failed to create test tensors");
    }

    ggml_set_name(src0, "gemmini-src0-nk");
    ggml_set_name(src1, "gemmini-src1-mk");
    ggml_set_name(dst,  "gemmini-dst-mn");

    ggml_set_input(src0);
    ggml_set_input(src1);
    ggml_set_output(dst);

    // Synthetic first-stage backend node. ggml_mul_mat() itself normally
    // creates F32 output, so until the F32 adapter is implemented we construct
    // the backend's current I8 x I8 -> I32 contract explicitly.
    dst->op     = GGML_OP_MUL_MAT;
    dst->src[0] = src0;
    dst->src[1] = src1;

    ggml_cgraph * graph = ggml_new_graph(ctx);
    if (!graph) {
        ggml_free(ctx);
        fail("ggml_new_graph() failed");
    }

    ggml_build_forward_expand(graph, dst);

    return {ctx, graph, src0, src1, dst};
}

static void fill_inputs(
        std::vector<int8_t> & src0,
        std::vector<int8_t> & src1,
        int64_t K,
        int64_t N,
        int64_t M) {
    // Use deterministic signed values with enough variety to expose transpose
    // and stride errors while remaining very far from int32 overflow.
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            src0[n * K + k] =
                static_cast<int8_t>(((n * 7 + k * 3 + 5) % 15) - 7);
        }
    }

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            src1[m * K + k] =
                static_cast<int8_t>(((m * 5 + k * 11 + 1) % 17) - 8);
        }
    }
}

static std::vector<int32_t> cpu_reference(
        const std::vector<int8_t> & src0,
        const std::vector<int8_t> & src1,
        int64_t K,
        int64_t N,
        int64_t M) {
    std::vector<int32_t> ref(static_cast<size_t>(M * N), 0);

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            int32_t sum = 0;

            for (int64_t k = 0; k < K; ++k) {
                sum +=
                    static_cast<int32_t>(src1[m * K + k]) *
                    static_cast<int32_t>(src0[n * K + k]);
            }

            ref[m * N + n] = sum;
        }
    }

    return ref;
}

static bool run_case(
        ggml_backend_t backend,
        int64_t K,
        int64_t N,
        int64_t M,
        const char * name) {
    std::printf(
        "\n=== %s: K=%lld N=%lld M=%lld ===\n",
        name,
        static_cast<long long>(K),
        static_cast<long long>(N),
        static_cast<long long>(M));

    test_graph tg = build_graph(K, N, M);

    if (!ggml_backend_supports_op(backend, tg.dst)) {
        std::fprintf(
            stderr,
            "FAIL: Gemmini does not advertise this test MUL_MAT\n");
        ggml_free(tg.ctx);
        return false;
    }

    // Gemmini currently uses a host-visible backend buffer. Allocate every
    // tensor in the GGML context through the Gemmini backend so graph_compute()
    // sees normal backend-owned GGML tensors.
    ggml_backend_buffer_t buffer =
        ggml_backend_alloc_ctx_tensors(tg.ctx, backend);

    if (!buffer) {
        std::fprintf(stderr, "FAIL: ggml_backend_alloc_ctx_tensors() failed\n");
        ggml_free(tg.ctx);
        return false;
    }

    std::vector<int8_t> src0(static_cast<size_t>(N * K));
    std::vector<int8_t> src1(static_cast<size_t>(M * K));

    fill_inputs(src0, src1, K, N, M);
    const std::vector<int32_t> ref =
        cpu_reference(src0, src1, K, N, M);

    ggml_backend_tensor_set(
        tg.src0, src0.data(), 0, src0.size() * sizeof(src0[0]));

    ggml_backend_tensor_set(
        tg.src1, src1.data(), 0, src1.size() * sizeof(src1[0]));

    // Initialize output to a nonzero pattern so a backend which incorrectly
    // reports success without writing dst is much easier to spot.
    ggml_backend_tensor_memset(
        tg.dst, 0xA5, 0, static_cast<size_t>(M * N) * sizeof(int32_t));

    const ggml_status status =
        ggml_backend_graph_compute(backend, tg.graph);

    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(
            stderr,
            "FAIL: graph compute returned %s\n",
            ggml_status_to_string(status));
        ggml_backend_buffer_free(buffer);
        ggml_free(tg.ctx);
        return false;
    }

    ggml_backend_synchronize(backend);

    std::vector<int32_t> got(static_cast<size_t>(M * N));
    ggml_backend_tensor_get(
        tg.dst,
        got.data(),
        0,
        got.size() * sizeof(got[0]));

    int errors = 0;
    int32_t max_abs_error = 0;

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const size_t i = static_cast<size_t>(m * N + n);
            const int32_t diff = got[i] - ref[i];
            const int32_t abs_diff = diff < 0 ? -diff : diff;
            max_abs_error = std::max(max_abs_error, abs_diff);

            if (got[i] != ref[i]) {
                if (errors < 16) {
                    std::fprintf(
                        stderr,
                        "  mismatch dst[m=%lld,n=%lld]: "
                        "expected=%d got=%d\n",
                        static_cast<long long>(m),
                        static_cast<long long>(n),
                        ref[i],
                        got[i]);
                }
                ++errors;
            }
        }
    }

    std::printf(
        "result: elements=%lld mismatches=%d max_abs_error=%d\n",
        static_cast<long long>(M * N),
        errors,
        max_abs_error);

    ggml_backend_buffer_free(buffer);
    ggml_free(tg.ctx);

    if (errors != 0) {
        std::fprintf(stderr, "FAIL: %s\n", name);
        return false;
    }

    std::printf("PASS: %s\n", name);
    return true;
}

int main() {
    std::printf("Gemmini GGML backend end-to-end compute test\n");

    if (!ggml_backend_gemmini_is_available()) {
        fail("Gemmini backend reports unavailable");
    }

    ggml_backend_t backend = ggml_backend_gemmini_init();
    if (!backend) {
        fail("failed to initialize Gemmini backend");
    }

    std::printf("backend: %s\n", ggml_backend_name(backend));

    bool ok = true;

    // Exactly one Gemmini DIM in each axis.
    ok &= run_case(backend, 16, 16, 16, "16x16x16 aligned");

    // All dimensions nonmultiples of DIM; exercises the same padding/tiling
    // behavior already validated by the direct kernel test, but now through
    // the GGML backend.
    ok &= run_case(backend, 23, 19, 17, "17x19x23 odd dimensions");

    // Larger multi-tile case to exercise more DMA/tile traffic.
    ok &= run_case(backend, 64, 48, 32, "32x48x64 multi-tile");

    ggml_backend_free(backend);

    if (!ok) {
        std::fprintf(stderr, "\nGemmini GGML backend test FAILED\n");
        return 1;
    }

    std::printf("\nGemmini GGML backend test PASSED\n");
    return 0;
}
