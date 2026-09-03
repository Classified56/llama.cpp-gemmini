#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-gemmini.h"

#include <cstdio>
#include <cstdlib>

static void fail(const char * message) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    std::exit(1);
}

static ggml_tensor * make_manual_mul_mat(
        ggml_context * ctx,
        ggml_type src0_type,
        ggml_type src1_type,
        ggml_type dst_type,
        int64_t K,
        int64_t N,
        int64_t M) {
    // GGML MUL_MAT layout:
    //   src0: [K, N]
    //   src1: [K, M]
    //   dst:  [N, M]
    ggml_tensor * src0 = ggml_new_tensor_2d(ctx, src0_type, K, N);
    ggml_tensor * src1 = ggml_new_tensor_2d(ctx, src1_type, K, M);
    ggml_tensor * dst  = ggml_new_tensor_2d(ctx, dst_type,  N, M);

    if (!src0 || !src1 || !dst) {
        fail("failed to create GGML tensors");
    }

    ggml_set_name(src0, "gemmini-test-src0");
    ggml_set_name(src1, "gemmini-test-src1");
    ggml_set_name(dst,  "gemmini-test-dst");

    // We intentionally construct the synthetic I8 x I8 -> I32 MUL_MAT node
    // manually. ggml_mul_mat() normally creates an F32 destination, while
    // the current first-stage Gemmini backend contract is I8 x I8 -> I32.
    dst->op     = GGML_OP_MUL_MAT;
    dst->src[0] = src0;
    dst->src[1] = src1;

    return dst;
}

int main() {
    std::printf("Gemmini GGML backend contract test\n\n");

    if (!ggml_backend_gemmini_is_available()) {
        fail("ggml_backend_gemmini_is_available() returned false");
    }

    ggml_backend_t backend = ggml_backend_gemmini_init();
    if (!backend) {
        fail("ggml_backend_gemmini_init() returned null");
    }

    std::printf("backend name: %s\n", ggml_backend_name(backend));

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (!dev) {
        ggml_backend_free(backend);
        fail("Gemmini backend has no device");
    }

    std::printf("device name:   %s\n", ggml_backend_dev_name(dev));
    std::printf("description:   %s\n", ggml_backend_dev_description(dev));
    std::printf("device type:   %d\n", static_cast<int>(ggml_backend_dev_type(dev)));

    if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_ACCEL) {
        ggml_backend_free(backend);
        fail("Gemmini device is not classified as ACCEL");
    }

    ggml_init_params params = {
        /*.mem_size   =*/ 1024 * 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        ggml_backend_free(backend);
        fail("ggml_init() failed");
    }

    // This is the exact operation the current hardware backend is intended
    // to support.
    ggml_tensor * supported =
        make_manual_mul_mat(ctx, GGML_TYPE_I8, GGML_TYPE_I8, GGML_TYPE_I32,
                            23, 19, 17);

    const bool backend_supports = ggml_backend_supports_op(backend, supported);
    const bool device_supports  = ggml_backend_dev_supports_op(dev, supported);

    std::printf(
        "I8 x I8 -> I32, K=23 N=19 M=17:\n"
        "  backend supports_op = %s\n"
        "  device  supports_op = %s\n",
        backend_supports ? "yes" : "no",
        device_supports  ? "yes" : "no");

    if (!backend_supports || !device_supports) {
        ggml_free(ctx);
        ggml_backend_free(backend);
        fail("Gemmini does not advertise the intended I8 x I8 -> I32 MUL_MAT");
    }

    // Informational probes. Do not assert these because they are expected to
    // change as the backend grows.
    ggml_tensor * f32 =
        make_manual_mul_mat(ctx, GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32,
                            23, 19, 17);

    ggml_tensor * tiny =
        make_manual_mul_mat(ctx, GGML_TYPE_I8, GGML_TYPE_I8, GGML_TYPE_I32,
                            8, 8, 8);

    std::printf(
        "\nInformational support probes:\n"
        "  F32 x F32 -> F32 = %s\n"
        "  I8 8x8x8         = %s\n",
        ggml_backend_supports_op(backend, f32)  ? "supported" : "not supported",
        ggml_backend_supports_op(backend, tiny) ? "supported" : "not supported");

    ggml_free(ctx);
    ggml_backend_free(backend);

    std::printf("\nPASS: Gemmini GGML backend contract is healthy\n");
    return 0;
}
