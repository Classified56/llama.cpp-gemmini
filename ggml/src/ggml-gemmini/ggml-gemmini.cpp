#include "ggml-gemmini.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

struct ggml_backend_gemmini_context {
    int n_threads = 1;
};

// -----------------------------------------------------------------------------
// Gemmini matmul eligibility
// -----------------------------------------------------------------------------

static bool ggml_gemmini_can_mul_mat_i8_i32(const struct ggml_tensor * op) {
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }

    // First milestone: synthetic I8 x I8 -> I32 only.
    if (src0->type != GGML_TYPE_I8 ||
        src1->type != GGML_TYPE_I8 ||
        op->type   != GGML_TYPE_I32) {
        return false;
    }

    // Keep the first implementation narrow.
    if (!ggml_is_matrix(src0) ||
        !ggml_is_matrix(src1) ||
        !ggml_is_matrix(op)) {
        return false;
    }

    if (!ggml_is_contiguous(src0) ||
        !ggml_is_contiguous(src1) ||
        !ggml_is_contiguous(op)) {
        return false;
    }

    // GGML MUL_MAT shape convention:
    //   src0: [K, N]
    //   src1: [K, M]
    //   dst:  [N, M]
    //
    // Mathematically:
    //   dst = src0^T * src1
    const int64_t K0 = src0->ne[0];
    const int64_t N  = src0->ne[1];
    const int64_t K1 = src1->ne[0];
    const int64_t M  = src1->ne[1];

    if (K0 != K1) {
        return false;
    }

    if (op->ne[0] != N || op->ne[1] != M) {
        return false;
    }

    // Avoid tiny GEMV/decode cases at first.
    if (M < 16 || N < 16 || K0 < 16) {
        return false;
    }

    return true;
}

static void ggml_gemmini_log_mul_mat(const struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_LOG_INFO(
        "ggml-gemmini: MUL_MAT\n"
        "  src0 name=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n"
        "  src1 name=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n"
        "  dst  name=%s type=%s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
        src0->name,
        ggml_type_name(src0->type),
        (long long) src0->ne[0], (long long) src0->ne[1],
        (long long) src0->ne[2], (long long) src0->ne[3],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],

        src1->name,
        ggml_type_name(src1->type),
        (long long) src1->ne[0], (long long) src1->ne[1],
        (long long) src1->ne[2], (long long) src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],

        dst->name,
        ggml_type_name(dst->type),
        (long long) dst->ne[0], (long long) dst->ne[1],
        (long long) dst->ne[2], (long long) dst->ne[3],
        dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]
    );
}

static void ggml_gemmini_mul_mat(
        ggml_backend_gemmini_context * ctx,
        struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);

    ggml_gemmini_log_mul_mat(dst);

#if !defined(GGML_GEMMINI_ENABLE_MATMUL)
    GGML_ABORT("%s: Gemmini MUL_MAT path is not enabled", __func__);
#else
    // TODO:
    //   1. Decode GGML MUL_MAT shape into Gemmini M/N/K.
    //   2. Pack or transpose if Gemmini requires a different layout.
    //   3. Call Gemmini tiled matmul.
    //   4. Write directly into dst->data.
    //
    // Do not return success until dst is fully computed.
    GGML_ABORT("%s: Gemmini MUL_MAT path is enabled but not implemented", __func__);
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
                GGML_ABORT(
                    "%s: unsupported op assigned to Gemmini backend: %s",
                    __func__,
                    ggml_op_desc(node)
                );
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i ggml_backend_gemmini_i = {
    /* .get_name                = */ ggml_backend_gemmini_get_name,
    /* .free                    = */ ggml_backend_gemmini_free,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .set_tensor_2d_async     = */ nullptr,
    /* .get_tensor_2d_async     = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ ggml_backend_gemmini_graph_compute,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .graph_optimize          = */ nullptr,
};

static ggml_guid_t ggml_backend_gemmini_guid(void) {
    static ggml_guid guid = {
        0x67, 0x65, 0x6d, 0x6d,
        0x69, 0x6e, 0x69, 0x2d,
        0x72, 0x69, 0x73, 0x63,
        0x76, 0x2d, 0x30, 0x31
    };

    return &guid;
}

ggml_backend_t ggml_backend_gemmini_init(void) {
    ggml_backend_gemmini_context * ctx = new ggml_backend_gemmini_context;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_gemmini_guid(),
        /* .iface   = */ ggml_backend_gemmini_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_gemmini_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_gemmini(ggml_backend_t backend) {
    return backend != nullptr &&
           ggml_guid_matches(backend->guid, ggml_backend_gemmini_guid());
}

bool ggml_backend_gemmini_is_available(void) {
    // TODO:
    // Replace with a real hardware/runtime probe if needed.
    //
    // For RoCC/custom-instruction Gemmini, this might be compile-time only.
    // For Linux driver/proxy-kernel Gemmini, this should check that the runtime
    // can actually submit work.
    return true;
}

const char * ggml_backend_gemmini_get_device_name(void) {
    return "Gemmini";
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
    return "RISC-V Gemmini accelerator";
}

static void ggml_backend_gemmini_device_get_memory(
        ggml_backend_dev_t dev,
        size_t * free,
        size_t * total) {
    GGML_UNUSED(dev);

    // Host-attached accelerator: no separate device memory to report yet.
    *free  = 0;
    *total = 0;
}

static enum ggml_backend_dev_type ggml_backend_gemmini_device_get_type(
        ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_gemmini_device_get_props(
        ggml_backend_dev_t dev,
        struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_gemmini_device_get_name(dev);
    props->description = ggml_backend_gemmini_device_get_description(dev);
    props->type        = ggml_backend_gemmini_device_get_type(dev);

    ggml_backend_gemmini_device_get_memory(
        dev,
        &props->memory_free,
        &props->memory_total
    );

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

    if (!ggml_backend_gemmini_is_available()) {
        return nullptr;
    }

    return ggml_backend_gemmini_init();
}

static ggml_backend_buffer_type_t ggml_backend_gemmini_device_get_buffer_type(
        ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);

    // Use CPU buffers until Gemmini has a real separate memory allocator.
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

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
            return ggml_gemmini_can_mul_mat_i8_i32(op);

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
    /* .offload_op           = */ nullptr,
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
    return 1;
}

static ggml_backend_dev_t ggml_backend_gemmini_reg_get_device(
        ggml_backend_reg_t reg,
        size_t index) {
    GGML_ASSERT(index == 0);

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

    // Optional extension point. Keep this minimal for now.
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

GGML_BACKEND_DL_IMPL(ggml_backend_gemmini_reg)