#include "ggml-gemmini.h"
#include "ggml-gemmini-kernel.h"
#include "ggml-gemmini-adapter.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

struct ggml_backend_gemmini_context {
    int n_threads = 1;
};

// -----------------------------------------------------------------------------
// Availability / identity
// -----------------------------------------------------------------------------

static ggml_guid_t ggml_backend_gemmini_guid(void) {
    // "gemmini-riscv-01" as bytes. The only strict requirement is uniqueness
    // within the process.
    static ggml_guid guid = {
        0x67, 0x65, 0x6d, 0x6d,
        0x69, 0x6e, 0x69, 0x2d,
        0x72, 0x69, 0x73, 0x63,
        0x76, 0x2d, 0x30, 0x31
    };

    return &guid;
}

bool ggml_backend_gemmini_is_available(void) {
#if defined(GGML_GEMMINI_FORCE_AVAILABLE)
    return true;
#elif defined(__riscv)
    const char * disable = std::getenv("GGML_DISABLE_GEMMINI");
    return disable == nullptr || std::strcmp(disable, "1") != 0;
#else
    // Do not claim a Gemmini device on native x86/ARM workstation builds unless
    // the user explicitly forces it for registry/debug testing.
    return false;
#endif
}

const char * ggml_backend_gemmini_get_device_name(void) {
    return "Gemmini";
}

// -----------------------------------------------------------------------------
// Tensor helpers
// -----------------------------------------------------------------------------

static inline const char * ggml_gemmini_op_name(const struct ggml_tensor * op) {
    return op && op->name[0] ? op->name : ggml_op_desc(op);
}

static inline float ggml_gemmini_get_f32(const struct ggml_tensor * t, int64_t i0, int64_t i1) {
    return *reinterpret_cast<const float *>(reinterpret_cast<const char *>(t->data) + i0*t->nb[0] + i1*t->nb[1]);
}

static inline void ggml_gemmini_set_f32(struct ggml_tensor * t, int64_t i0, int64_t i1, float v) {
    *reinterpret_cast<float *>(reinterpret_cast<char *>(t->data) + i0*t->nb[0] + i1*t->nb[1]) = v;
}

static inline int8_t ggml_gemmini_get_i8(const struct ggml_tensor * t, int64_t i0, int64_t i1) {
    return *reinterpret_cast<const int8_t *>(reinterpret_cast<const char *>(t->data) + i0*t->nb[0] + i1*t->nb[1]);
}

static inline void ggml_gemmini_set_i32(struct ggml_tensor * t, int64_t i0, int64_t i1, int32_t v) {
    *reinterpret_cast<int32_t *>(reinterpret_cast<char *>(t->data) + i0*t->nb[0] + i1*t->nb[1]) = v;
}

static bool ggml_gemmini_is_2d_matrix(const struct ggml_tensor * t) {
    return t != nullptr && ggml_is_matrix(t) && t->ne[2] == 1 && t->ne[3] == 1;
}

static bool ggml_gemmini_mul_mat_shape_ok(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (!ggml_gemmini_is_2d_matrix(src0) ||
        !ggml_gemmini_is_2d_matrix(src1) ||
        !ggml_gemmini_is_2d_matrix(op)) {
        return false;
    }

    // GGML MUL_MAT shape convention:
    //   src0: [K, N]
    //   src1: [K, M]
    //   dst:  [N, M]
    //   dst = src0^T * src1
    const int64_t k0 = src0->ne[0];
    const int64_t n  = src0->ne[1];
    const int64_t k1 = src1->ne[0];
    const int64_t m  = src1->ne[1];

    return k0 == k1 && op->ne[0] == n && op->ne[1] == m;
}

static bool ggml_gemmini_can_mul_mat_f32(const struct ggml_tensor * op) {
#if !defined(GGML_GEMMINI_ENABLE_NATIVE_TEST_MATMUL)
    GGML_UNUSED(op);
    return false;
#else
    if (!ggml_gemmini_mul_mat_shape_ok(op)) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    return src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type   == GGML_TYPE_F32;
#endif
}

static bool ggml_gemmini_can_mul_mat_i8_i32(const struct ggml_tensor * op) {
#if !defined(GGML_GEMMINI_ENABLE_NATIVE_TEST_MATMUL) && !defined(GGML_GEMMINI_ENABLE_TILED_MATMUL)
    GGML_UNUSED(op);
    return false;
#else
    if (!ggml_gemmini_mul_mat_shape_ok(op)) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (src0->type != GGML_TYPE_I8 ||
        src1->type != GGML_TYPE_I8 ||
        op->type   != GGML_TYPE_I32) {
        return false;
    }

    // Start with contiguous tensors only for the hardware path. Native test
    // fallback can handle strides, but keeping this narrow avoids surprising
    // scheduler placement before the real Gemmini kernel is mature.
    return ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
#endif
}

static bool ggml_gemmini_supports_mul_mat(const struct ggml_tensor * op) {
    GGML_LOG_DEBUG(
        "Gemmini supports? op=%s src0=%s src1=%s \n",
        ggml_op_desc(op),
        ggml_type_name(op->src[0]->type),
        ggml_type_name(op->src[1]->type)
    );
    return ggml_gemmini_can_mul_mat_f32(op) || ggml_gemmini_can_mul_mat_i8_i32(op) ||
           ggml_gemmini_adapter_supports(op);
}

// -----------------------------------------------------------------------------
// Reference matmul implementations
// -----------------------------------------------------------------------------

static void ggml_gemmini_mul_mat_f32_ref(struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_gemmini_can_mul_mat_f32(dst));

    const struct ggml_tensor * src0 = dst->src[0]; // [K, N]
    const struct ggml_tensor * src1 = dst->src[1]; // [K, M]

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                sum += ggml_gemmini_get_f32(src0, k, n) * ggml_gemmini_get_f32(src1, k, m);
            }
            ggml_gemmini_set_f32(dst, n, m, sum);
        }
    }
}

static void ggml_gemmini_mul_mat_i8_i32_ref(struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_gemmini_can_mul_mat_i8_i32(dst));

    const struct ggml_tensor * src0 = dst->src[0]; // [K, N]
    const struct ggml_tensor * src1 = dst->src[1]; // [K, M]

    const int64_t K = src0->ne[0];
    const int64_t N = src0->ne[1];
    const int64_t M = src1->ne[1];

    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            int32_t sum = 0;
            for (int64_t k = 0; k < K; ++k) {
                sum += (int32_t) ggml_gemmini_get_i8(src0, k, n) *
                       (int32_t) ggml_gemmini_get_i8(src1, k, m);
            }
            ggml_gemmini_set_i32(dst, n, m, sum);
        }
    }
}

// -----------------------------------------------------------------------------
// Gemmini hardware hook
// -----------------------------------------------------------------------------

static bool ggml_gemmini_try_mul_mat_i8_i32_hw(
        struct ggml_tensor * dst) {

#if !defined(GGML_GEMMINI_ENABLE_TILED_MATMUL)
    (void) dst;
    return false;
#else

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }

    if (src0->type != GGML_TYPE_I8 ||
        src1->type != GGML_TYPE_I8 ||
        dst->type  != GGML_TYPE_I32) {
        return false;
    }

    if (!ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(dst)) {
        return false;
    }

    const size_t K = src0->ne[0];
    const size_t N = src0->ne[1];
    const size_t M = src1->ne[1];

    if (src1->ne[0] != (int64_t) K) {
        return false;
    }

    if (dst->ne[0] != (int64_t) N ||
        dst->ne[1] != (int64_t) M) {
        return false;
    }

    return ggml_gemmini_mul_mat_i8_i32(
        static_cast<const int8_t *>(src0->data),
        static_cast<const int8_t *>(src1->data),
        static_cast<int32_t *>(dst->data),
        K,
        N,
        M);
#endif
}


// static void ggml_gemmini_mul_mat(ggml_backend_gemmini_context * ctx, struct ggml_tensor * dst) {
//     GGML_UNUSED(ctx);

//     if (ggml_gemmini_can_mul_mat_i8_i32(dst)) {
//         if (ggml_gemmini_try_mul_mat_i8_i32_hw(dst)) {
//             return;
//         }

// #if defined(GGML_GEMMINI_ENABLE_NATIVE_TEST_MATMUL)
//         ggml_gemmini_mul_mat_i8_i32_ref(dst);
//         return;
// #else
//         // GGML_ABORT("%s: I8xI8->I32 MUL_MAT reached Gemmini but hardware path is not implemented: %s",
//         //            __func__, ggml_gemmini_op_name(dst));
//         GGML_ABORT("%s: Gemmini F32/Q8 adapter failed for %s", __func__, ggml_gemmini_op_name(dst));
// #endif
//     }

//     if (ggml_gemmini_can_mul_mat_f32(dst)) {
//         ggml_gemmini_mul_mat_f32_ref(dst);
//         return;
//     }

//     GGML_ABORT("%s: unsupported MUL_MAT assigned to Gemmini backend: %s",
//                __func__, ggml_gemmini_op_name(dst));
// }

static void ggml_gemmini_mul_mat(
        ggml_backend_gemmini_context * ctx,
        struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);

    // ggml_gemmini_log_mul_mat(dst);

#if !defined(GGML_GEMMINI_ENABLE_TILED_MATMUL)

    GGML_ABORT(
        "%s: Gemmini tiled MUL_MAT path is not enabled",
        __func__);

#else

    // ---------------------------------------------------------
    // 1. Diagnostic / native Gemmini I8 x I8 -> I32 path
    // ---------------------------------------------------------

    if (ggml_gemmini_can_mul_mat_i8_i32(dst)) {
        GGML_LOG_INFO(
            "ggml-gemmini: dispatch I8 x I8 -> I32 hardware path\n");

        if (ggml_gemmini_try_mul_mat_i8_i32_hw(dst)) {
            return;
        }

        GGML_ABORT(
            "%s: Gemmini I8 x I8 -> I32 hardware execution failed",
            __func__);
    }

    // ---------------------------------------------------------
    // 2. Model-facing adapter
    //
    // F32  x F32 -> F32
    // Q8_0 x F32 -> F32
    // ---------------------------------------------------------

    if (ggml_gemmini_adapter_supports(dst)) {
        GGML_LOG_INFO(
            "ggml-gemmini: dispatch adapter "
            "src0=%s src1=%s dst=%s\n",
            ggml_type_name(dst->src[0]->type),
            ggml_type_name(dst->src[1]->type),
            ggml_type_name(dst->type));

        if (ggml_gemmini_adapter_compute(dst)) {
            return;
        }

        GGML_ABORT(
            "%s: Gemmini F32/Q8_0 adapter execution failed",
            __func__);
    }

#if defined(GGML_GEMMINI_ENABLE_NATIVE_TEST_MATMUL)

    // ---------------------------------------------------------
    // 3. Optional CPU/reference F32 path used during host testing
    // ---------------------------------------------------------

    if (ggml_gemmini_can_mul_mat_f32(dst)) {
        ggml_gemmini_mul_mat_f32_ref(dst);
        return;
    }

#endif

    GGML_ABORT(
        "%s: unsupported MUL_MAT assigned to Gemmini: "
        "src0=%s src1=%s dst=%s",
        __func__,
        ggml_type_name(dst->src[0]->type),
        ggml_type_name(dst->src[1]->type),
        ggml_type_name(dst->type));

#endif
}

// -----------------------------------------------------------------------------
// Backend interface
// -----------------------------------------------------------------------------

static const char * ggml_backend_gemmini_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "Gemmini";
}

static void ggml_backend_gemmini_free(ggml_backend_t backend) {
    ggml_backend_gemmini_context * ctx =
        static_cast<ggml_backend_gemmini_context *>(backend->context);

    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_gemmini_graph_compute(
        ggml_backend_t backend,
        struct ggml_cgraph * cgraph) {
    ggml_backend_gemmini_context * ctx =
        static_cast<ggml_backend_gemmini_context *>(backend->context);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            case GGML_OP_MUL_MAT:
                ggml_gemmini_mul_mat(ctx, node);
                break;

            default:
                GGML_ABORT("%s: unsupported op assigned to Gemmini backend: %s",
                           __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i ggml_backend_gemmini_i = {
    /* .get_name           = */ ggml_backend_gemmini_get_name,
    /* .free               = */ ggml_backend_gemmini_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .set_tensor_2d_async= */ nullptr,
    /* .get_tensor_2d_async= */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ nullptr,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_gemmini_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

bool ggml_backend_is_gemmini(ggml_backend_t backend) {
    return backend != nullptr &&
           ggml_guid_matches(backend->guid, ggml_backend_gemmini_guid());
}

// -----------------------------------------------------------------------------
// Device interface
// -----------------------------------------------------------------------------

static const char * ggml_backend_gemmini_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "Gemmini";
}

static const char * ggml_backend_gemmini_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "RISC-V Gemmini RoCC accelerator";
}

static void ggml_backend_gemmini_device_get_memory(
        ggml_backend_dev_t dev,
        size_t * free,
        size_t * total) {
    GGML_UNUSED(dev);

    // Gemmini is modeled as a CPU-attached accelerator using host-visible
    // buffers. It does not expose a separate device memory pool here.
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_gemmini_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_gemmini_device_get_props(
        ggml_backend_dev_t dev,
        struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_gemmini_device_get_name(dev);
    props->description = ggml_backend_gemmini_device_get_description(dev);
    props->type        = ggml_backend_gemmini_device_get_type(dev);
    props->device_id   = nullptr;

    ggml_backend_gemmini_device_get_memory(
        dev,
        &props->memory_free,
        &props->memory_total);

    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ true,
        /* .events               = */ false,
    };
}

static ggml_backend_t ggml_backend_gemmini_device_init_backend(
        ggml_backend_dev_t dev,
        const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);

    return ggml_backend_gemmini_init();
}

static ggml_backend_buffer_type_t ggml_backend_gemmini_device_get_buffer_type(
        ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_gemmini_device_buffer_from_host_ptr(
        ggml_backend_dev_t dev,
        void * ptr,
        size_t size,
        size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);

    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_gemmini_device_supports_op(
        ggml_backend_dev_t dev,
        const struct ggml_tensor * op) {
    GGML_UNUSED(dev);

    if (op == nullptr) {
        return false;
    }

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
            return ggml_gemmini_supports_mul_mat(op);

        default:
            return false;
    }
}

static bool ggml_backend_gemmini_device_offload_op(
        ggml_backend_dev_t dev,
        const struct ggml_tensor * op) {
    GGML_UNUSED(dev);

    if (op == nullptr) {
        return false;
    }

    GGML_LOG_DEBUG(
        "Gemmini offload? op=%s src0=%s src1=%s \n",
        ggml_op_desc(op),
        ggml_type_name(op->src[0]->type),
        ggml_type_name(op->src[1]->type)
    );

    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return ggml_gemmini_adapter_supports(op);

        default:
            return false;
    }
}

static bool ggml_backend_gemmini_device_supports_buft(
        ggml_backend_dev_t dev,
        ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_gemmini_device_i = {
    /* .get_name             = */ ggml_backend_gemmini_device_get_name,
    /* .get_description      = */ ggml_backend_gemmini_device_get_description,
    /* .get_memory           = */ ggml_backend_gemmini_device_get_memory,
    /* .get_type             = */ ggml_backend_gemmini_device_get_type,
    /* .get_props            = */ ggml_backend_gemmini_device_get_props,
    /* .init_backend         = */ ggml_backend_gemmini_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_gemmini_device_get_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ ggml_backend_gemmini_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_gemmini_device_supports_op,
    /* .supports_buft        = */ ggml_backend_gemmini_device_supports_buft,
    /* .offload_op           = */ ggml_backend_gemmini_device_offload_op,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

// -----------------------------------------------------------------------------
// Backend registry interface
// -----------------------------------------------------------------------------

static const char * ggml_backend_gemmini_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "Gemmini";
}

static size_t ggml_backend_gemmini_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return ggml_backend_gemmini_is_available() ? 1 : 0;
}

static ggml_backend_dev_t ggml_backend_gemmini_reg_get_device(
        ggml_backend_reg_t reg,
        size_t index) {
    GGML_ASSERT(index == 0);
    GGML_ASSERT(ggml_backend_gemmini_is_available());

    static ggml_backend_device ggml_backend_gemmini_device = {
        /* .iface   = */ ggml_backend_gemmini_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_gemmini_device;
}

static void * ggml_backend_gemmini_get_proc_address(
        ggml_backend_reg_t reg,
        const char * name) {
    GGML_UNUSED(reg);

    if (std::strcmp(name, "ggml_backend_gemmini_is_available") == 0) {
        return reinterpret_cast<void *>(ggml_backend_gemmini_is_available);
    }

    return nullptr;
}

static const struct ggml_backend_reg_i ggml_backend_gemmini_reg_i = {
    /* .get_name         = */ ggml_backend_gemmini_reg_get_name,
    /* .get_device_count = */ ggml_backend_gemmini_reg_get_device_count,
    /* .get_device       = */ ggml_backend_gemmini_reg_get_device,
    /* .get_proc_address = */ ggml_backend_gemmini_get_proc_address,
};

ggml_backend_reg_t ggml_backend_gemmini_reg(void) {
    static struct ggml_backend_reg ggml_backend_gemmini_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_gemmini_reg_i,
        /* .context     = */ nullptr,
    };

    return &ggml_backend_gemmini_reg;
}

ggml_backend_t ggml_backend_gemmini_init(void) {
    if (!ggml_backend_gemmini_is_available()) {
        return nullptr;
    }

    ggml_backend_gemmini_context * ctx = new ggml_backend_gemmini_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_gemmini_guid(),
        /* .iface   = */ ggml_backend_gemmini_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_gemmini_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_gemmini_reg)
