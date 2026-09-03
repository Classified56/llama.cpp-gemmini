#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void die(const char * msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    std::exit(1);
}

struct test_case {
    ggml_context * ctx = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * weights = nullptr;
    ggml_tensor * activations = nullptr;
    ggml_tensor * output = nullptr;
};

static test_case make_graph(
        ggml_type weight_type,
        int64_t K,
        int64_t N,
        int64_t M) {
    ggml_init_params ip = {
        /* mem_size   */ 1024 * 1024,
        /* mem_buffer */ nullptr,
        /* no_alloc   */ true,
    };

    test_case tc;
    tc.ctx = ggml_init(ip);
    if (!tc.ctx) {
        die("ggml_init failed");
    }

    tc.weights     = ggml_new_tensor_2d(tc.ctx, weight_type, K, N);
    tc.activations = ggml_new_tensor_2d(tc.ctx, GGML_TYPE_F32, K, M);
    tc.output      = ggml_mul_mat(tc.ctx, tc.weights, tc.activations);

    ggml_set_name(tc.weights, "gemmini-adapter-weights");
    ggml_set_name(tc.activations, "gemmini-adapter-activations");
    ggml_set_name(tc.output, "gemmini-adapter-output");

    ggml_set_input(tc.weights);
    ggml_set_input(tc.activations);
    ggml_set_output(tc.output);

    tc.graph = ggml_new_graph(tc.ctx);
    ggml_build_forward_expand(tc.graph, tc.output);

    return tc;
}

static std::vector<float> make_weights(int64_t K, int64_t N) {
    std::vector<float> w(static_cast<size_t>(K * N));

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            w[static_cast<size_t>(n * K + k)] =
                static_cast<float>(((n * 7 + k * 3 + 5) % 31) - 15) * 0.125f;
        }
    }

    return w;
}

static std::vector<float> make_activations(int64_t K, int64_t M) {
    std::vector<float> x(static_cast<size_t>(K * M));

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t k = 0; k < K; ++k) {
            x[static_cast<size_t>(m * K + k)] =
                static_cast<float>(((m * 5 + k * 11 + 1) % 29) - 14) * 0.10f;
        }
    }

    return x;
}

static std::vector<float> cpu_reference(
        const std::vector<float> & w,
        const std::vector<float> & x,
        int64_t K,
        int64_t N,
        int64_t M) {
    std::vector<float> y(static_cast<size_t>(M * N), 0.0f);

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;

            for (int64_t k = 0; k < K; ++k) {
                sum +=
                    x[static_cast<size_t>(m * K + k)] *
                    w[static_cast<size_t>(n * K + k)];
            }

            y[static_cast<size_t>(m * N + n)] = sum;
        }
    }

    return y;
}

static bool check_result(
        const char * name,
        const std::vector<float> & got,
        const std::vector<float> & ref) {
    double sum_abs = 0.0;
    double dot = 0.0;
    double ng = 0.0;
    double nr = 0.0;

    float max_abs_ref = 0.0f;
    float max_abs_err = 0.0f;

    for (size_t i = 0; i < got.size(); ++i) {
        const float e = std::fabs(got[i] - ref[i]);
        max_abs_err = std::max(max_abs_err, e);
        max_abs_ref = std::max(max_abs_ref, std::fabs(ref[i]));
        sum_abs += e;

        dot += static_cast<double>(got[i]) * ref[i];
        ng  += static_cast<double>(got[i]) * got[i];
        nr  += static_cast<double>(ref[i]) * ref[i];
    }

    const double mae = sum_abs / static_cast<double>(got.size());
    const double cosine =
        (ng > 0.0 && nr > 0.0) ? dot / std::sqrt(ng * nr) : 1.0;

    std::printf(
        "%s: max_ref=%g max_abs_err=%g mae=%g cosine=%0.8f\n",
        name,
        max_abs_ref,
        max_abs_err,
        mae,
        cosine);

    // Correctness-oriented threshold, not a final model-quality criterion.
    const float max_allowed = 0.05f * max_abs_ref + 0.10f;

    if (max_abs_err > max_allowed || cosine < 0.99) {
        std::fprintf(
            stderr,
            "FAIL: %s exceeded numerical tolerance "
            "(allowed max abs=%g)\n",
            name,
            max_allowed);
        return false;
    }

    std::printf("PASS: %s\n", name);
    return true;
}

static bool run_f32(
        ggml_backend_t backend,
        int64_t K,
        int64_t N,
        int64_t M) {
    test_case tc = make_graph(GGML_TYPE_F32, K, N, M);

    if (!ggml_backend_supports_op(backend, tc.output)) {
        std::fprintf(stderr, "FAIL: backend does not support F32 MUL_MAT\n");
        ggml_free(tc.ctx);
        return false;
    }

    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors(tc.ctx, backend);

    if (!buf) {
        ggml_free(tc.ctx);
        die("buffer allocation failed");
    }

    const auto w = make_weights(K, N);
    const auto x = make_activations(K, M);
    const auto ref = cpu_reference(w, x, K, N, M);

    ggml_backend_tensor_set(
        tc.weights, w.data(), 0, w.size() * sizeof(float));
    ggml_backend_tensor_set(
        tc.activations, x.data(), 0, x.size() * sizeof(float));

    const ggml_status st = ggml_backend_graph_compute(backend, tc.graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: F32 graph compute returned %s\n",
                     ggml_status_to_string(st));
        ggml_backend_buffer_free(buf);
        ggml_free(tc.ctx);
        return false;
    }

    ggml_backend_synchronize(backend);

    std::vector<float> got(static_cast<size_t>(M * N));
    ggml_backend_tensor_get(
        tc.output, got.data(), 0, got.size() * sizeof(float));

    const bool ok = check_result("F32 x F32 -> F32", got, ref);

    ggml_backend_buffer_free(buf);
    ggml_free(tc.ctx);
    return ok;
}

static bool run_q8(
        ggml_backend_t backend,
        int64_t K,
        int64_t N,
        int64_t M) {
    if ((K % ggml_blck_size(GGML_TYPE_Q8_0)) != 0) {
        die("Q8_0 test K is not block aligned");
    }

    test_case tc = make_graph(GGML_TYPE_Q8_0, K, N, M);

    if (!ggml_backend_supports_op(backend, tc.output)) {
        std::fprintf(stderr, "FAIL: backend does not support Q8_0/F32 MUL_MAT\n");
        ggml_free(tc.ctx);
        return false;
    }

    ggml_backend_buffer_t buf =
        ggml_backend_alloc_ctx_tensors(tc.ctx, backend);

    if (!buf) {
        ggml_free(tc.ctx);
        die("buffer allocation failed");
    }

    const auto w_original = make_weights(K, N);
    const auto x = make_activations(K, M);

    const size_t q8_row_bytes = ggml_row_size(GGML_TYPE_Q8_0, K);
    std::vector<uint8_t> q8(static_cast<size_t>(N) * q8_row_bytes);
    std::vector<float> w_dequant(static_cast<size_t>(N * K));

    for (int64_t n = 0; n < N; ++n) {
        const float * src =
            w_original.data() + static_cast<size_t>(n * K);

        block_q8_0 * qrow =
            reinterpret_cast<block_q8_0 *>(
                q8.data() + static_cast<size_t>(n) * q8_row_bytes);

        quantize_row_q8_0_ref(src, qrow, K);

        dequantize_row_q8_0(
            qrow,
            w_dequant.data() + static_cast<size_t>(n * K),
            K);
    }

    // Compare the Gemmini adapter against the values actually represented by
    // Q8_0, not against the pre-Q8 original F32 values.
    const auto ref = cpu_reference(w_dequant, x, K, N, M);

    ggml_backend_tensor_set(
        tc.weights, q8.data(), 0, q8.size());
    ggml_backend_tensor_set(
        tc.activations, x.data(), 0, x.size() * sizeof(float));

    const ggml_status st = ggml_backend_graph_compute(backend, tc.graph);
    if (st != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "FAIL: Q8 graph compute returned %s\n",
                     ggml_status_to_string(st));
        ggml_backend_buffer_free(buf);
        ggml_free(tc.ctx);
        return false;
    }

    ggml_backend_synchronize(backend);

    std::vector<float> got(static_cast<size_t>(M * N));
    ggml_backend_tensor_get(
        tc.output, got.data(), 0, got.size() * sizeof(float));

    const bool ok = check_result("Q8_0 x F32 -> F32", got, ref);

    ggml_backend_buffer_free(buf);
    ggml_free(tc.ctx);
    return ok;
}

int main() {
    std::printf("Gemmini registry + F32/Q8_0 adapter test\n");

    // Deliberately initialize through the generic registry, not the direct
    // ggml_backend_gemmini_init() symbol. This verifies the registration fix.
    ggml_backend_t backend =
        ggml_backend_init_by_name("Gemmini", nullptr);

    if (!backend) {
        die("Gemmini was not found through the central backend registry");
    }

    std::printf("registry backend: %s\n", ggml_backend_name(backend));

    bool ok = true;

    ok &= run_f32(backend, 23, 19, 17);

    // Q8_0 K must be a multiple of its block size (32).
    ok &= run_q8(backend, 64, 48, 32);

    ggml_backend_free(backend);

    if (!ok) {
        std::fprintf(stderr, "\nGemmini adapter test FAILED\n");
        return 1;
    }

    std::printf("\nGemmini adapter test PASSED\n");
    return 0;
}
