#include "ggml-gemmini.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"

#include <cstdlib>
#include <cstdint>
#include <cstring>

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
// Optional matmul-dispatch stub
// -----------------------------------------------------------------------------

static bool ggml_gemmini_can_mul_mat_i8_i32(const struct ggml_tensor * op) {
#if !defined(GGML_GEMMINI_ENABLE_MATMUL_STUB)
    GGML_UNUSED(op);
    return false;
#else
    if (op == nullptr || op->op != GGML_OP_MUL_MAT) {
        return false;
    }

    const struct ggml_tensor * src0 = op->src[0];
    const struct ggml_tensor * src1 = op->src[1];

    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }

    // First synthetic milestone only:
    //   src0: I8 [K, N]
    //   src1: I8 [K, M]
    //   dst:  I32[N, M]
    if (src0->type != GGML_TYPE_I8 ||
        src1->type != GGML_TYPE_I8 ||
        op->type   != GGML_TYPE_I32) {
        return false;
    }

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

    const int64_t k0 = src0->ne[0];
    const int64_t n  = src0->ne[1];
    const int64_t k1 = src1->ne[0];
    const int64_t m  = src1->ne[1];

    if (k0 != k1) {
        return false;
    }

    if (op->ne[0] != n || op->ne[1] != m) {
        return false;
    }

    // Keep tiny GEMV/decode cases on CPU for now.
    return m >= 16 && n >= 16 && k0 >= 16;
#endif
}

static void ggml_gemmini_mul_mat_stub(
        ggml_backend_gemmini_context * ctx,
        struct ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);

    // This backend is currently intended to build and register only. Once you
    // wire tiled_matmul_auto, replace this abort with real output writes.
    GGML_ABORT("%s: Gemmini MUL_MAT dispatch reached, but compute is not implemented yet", __func__);
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
                ggml_gemmini_mul_mat_stub(ctx, node);
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
