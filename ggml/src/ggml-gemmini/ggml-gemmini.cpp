#include "ggml-gemmini.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"  // For fallback CPU execution

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <stdint.h>

// ============================================================================
// Gemmini Backend Context
// ============================================================================

struct ggml_backend_gemmini_context {
    ggml_backend_t cpu_backend;       // CPU backend for actual computation
    
    // Performance metrics (for side-channel analysis)
    uint64_t total_matmuls;
    uint64_t total_flops;
    uint64_t total_matmul_bytes;      // Total data moved in matmuls
    
    // TODO: Gemmini hardware state
    // - Scratchpad memory pointer
    // - DMA controller state
    // - Systolic array configuration
};

// ============================================================================
// Operation Computing
// ============================================================================

// Detect if this is a matmul operation
static bool is_matmul_op(ggml_tensor * tensor) {
    return tensor->op == GGML_OP_MUL_MAT || 
           tensor->op == GGML_OP_MUL_MAT_ID;
}

// Log matmul operation details for side-channel analysis
static void log_matmul_op(int node_idx, ggml_tensor * tensor) {
    ggml_tensor * a = tensor->src[0];
    ggml_tensor * b = tensor->src[1];
    
    fprintf(stderr, "[GEMMINI] Node %d: matmul\n", node_idx);
    fprintf(stderr, "  A: [%lld, %lld] (%s)\n", 
            a->ne[0], a->ne[1], ggml_type_name(a->type));
    fprintf(stderr, "  B: [%lld, %lld] (%s)\n", 
            b->ne[0], b->ne[1], ggml_type_name(b->type));
    fprintf(stderr, "  C: [%lld, %lld]\n", 
            tensor->ne[0], tensor->ne[1]);
    
    // Calculate theoretical FLOPs
    uint64_t flops = 2ULL * a->ne[0] * a->ne[1] * b->ne[1];
    fprintf(stderr, "  FLOPs: %llu\n", flops);
    
    // Calculate data movement
    uint64_t a_bytes = ggml_nbytes(a);
    uint64_t b_bytes = ggml_nbytes(b);
    uint64_t c_bytes = ggml_nbytes(tensor);
    fprintf(stderr, "  Data: A=%llu B=%llu C=%llu total=%llu\n",
            a_bytes, b_bytes, c_bytes, a_bytes + b_bytes + c_bytes);
}

// Execute matmul on Gemmini (currently delegates to CPU, will implement Phase 2)
static void gemmini_execute_matmul(
    ggml_backend_gemmini_context * ctx,
    ggml_tensor * node
) {
    ggml_tensor * a = node->src[0];
    ggml_tensor * b = node->src[1];
    
    fprintf(stderr, "[GEMMINI] Matmul: [%lld,%lld] x [%lld,%lld]\n",
            a->ne[0], a->ne[1], b->ne[0], b->ne[1]);
    
    // TODO: Phase 2 - Implement actual Gemmini execution:
    // 1. Allocate/manage scratchpad memory
    // 2. Issue DMA transfers for A and B to scratchpad
    // 3. Configure systolic array (16x16 default Gemmini config)
    // 4. Issue matmul command to systolic array
    // 5. Collect performance counter data (cycles, bandwidth, L2 misses)
    // 6. DMA result back to main memory
    // 7. Update metrics for side-channel analysis
    
    ctx->total_matmuls++;
    uint64_t flops = 2ULL * a->ne[0] * a->ne[1] * b->ne[1];
    ctx->total_flops += flops;
    ctx->total_matmul_bytes += ggml_nbytes(a) + ggml_nbytes(b) + ggml_nbytes(node);
    
    // For now: CPU backend handles actual computation
}

// ============================================================================
// Backend Interface Implementation
// ============================================================================

static void ggml_backend_gemmini_free(ggml_backend_t backend) {
    struct ggml_backend_gemmini_context * ctx = 
        (struct ggml_backend_gemmini_context *)backend->context;
    
    fprintf(stderr, "[GEMMINI] Backend shutdown\n");
    fprintf(stderr, "  Total matmuls: %llu\n", ctx->total_matmuls);
    fprintf(stderr, "  Total FLOPs: %llu\n", ctx->total_flops);
    fprintf(stderr, "  Total data: %llu bytes\n", ctx->total_matmul_bytes);
    
    if (ctx->cpu_backend) {
        ggml_backend_free(ctx->cpu_backend);
    }
    
    delete ctx;
    delete backend;
}

static const char * ggml_backend_gemmini_get_name(ggml_backend_t backend) {
    return "Gemmini";
    GGML_UNUSED(backend);
}

// Graph compute: identify matmuls, log them, delegate execution
static enum ggml_status ggml_backend_gemmini_graph_compute(
    ggml_backend_t backend, 
    struct ggml_cgraph * cgraph
) {
    struct ggml_backend_gemmini_context * ctx = 
        (struct ggml_backend_gemmini_context *)backend->context;
    
    fprintf(stderr, "[GEMMINI] Computing graph with %d nodes\n", cgraph->n_nodes);
    
    // First pass: identify and log all matmul operations
    int matmul_count = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];
        
        if (is_matmul_op(node)) {
            matmul_count++;
            log_matmul_op(i, node);
            gemmini_execute_matmul(ctx, node);
        } else {
            fprintf(stderr, "[GEMMINI] Node %d: op=%d (delegating to CPU)\n", 
                    i, node->op);
        }
    }
    
    fprintf(stderr, "[GEMMINI] Found %d matmul operations\n\n", matmul_count);
    
    // For now: use CPU backend to actually compute the graph
    // In Phase 2, Gemmini will compute matmuls, CPU handles everything else
    if (ctx->cpu_backend) {
        return ggml_backend_graph_compute(ctx->cpu_backend, cgraph);
    }
    
    return GGML_STATUS_SUCCESS;
}

// ============================================================================
// Backend Interface
// ============================================================================

static const struct ggml_backend_i ggml_backend_gemmini_i = {
    /* .get_name            = */ ggml_backend_gemmini_get_name,
    /* .free                = */ ggml_backend_gemmini_free,
    /* .set_tensor_async    = */ NULL,
    /* .get_tensor_async    = */ NULL,
    /* .set_tensor_2d_async = */ NULL,
    /* .get_tensor_2d_async = */ NULL,
    /* .cpy_tensor_async    = */ NULL,
    /* .cpy_tensor_2d_async = */ NULL,
    /* .graph_plan_create   = */ NULL,
    /* .graph_plan_free     = */ NULL,
    /* .graph_plan_compute  = */ NULL,
    /* .graph_compute       = */ ggml_backend_gemmini_graph_compute,
    /* .supports_op         = */ NULL,
    /* .supports_buft       = */ NULL,
    /* .offload_op          = */ NULL,
    /* .event_new           = */ NULL,
    /* .event_free          = */ NULL,
    /* .event_record        = */ NULL,
    /* .event_wait          = */ NULL,
    /* .event_synchronize   = */ NULL,
};

// ============================================================================
// Public API
// ============================================================================

ggml_backend_t ggml_backend_gemmini_init(void) {
    fprintf(stderr, "[GEMMINI] Initializing Gemmini backend\n");
    
    // Create context
    struct ggml_backend_gemmini_context * ctx = 
        new ggml_backend_gemmini_context();
    
    if (ctx == NULL) {
        return NULL;
    }
    
    ctx->cpu_backend      = ggml_backend_cpu_init();
    ctx->total_matmuls    = 0;
    ctx->total_flops      = 0;
    ctx->total_matmul_bytes = 0;
    
    if (ctx->cpu_backend == NULL) {
        delete ctx;
        return NULL;
    }
    
    // Create backend
    ggml_backend_t gemmini_backend = new ggml_backend {
        /* .guid    = */ ggml_guid_t{0, 0},  // TODO: proper GUID
        /* .iface   = */ ggml_backend_gemmini_i,
        /* .device  = */ NULL,  // TODO: proper device
        /* .context = */ ctx,
    };
    
    if (gemmini_backend == NULL) {
        ggml_backend_free(ctx->cpu_backend);
        delete ctx;
        return NULL;
    }
    
    fprintf(stderr, "[GEMMINI] Backend initialized successfully\n");
    return gemmini_backend;
}

bool ggml_backend_gemmini_is_available(void) {
    // TODO: Check if we're running in FireSim with Gemmini
    // For now, assume available
    return true;
}

const char * ggml_backend_gemmini_get_device_name(void) {
    return "Gemmini RISC-V NPU (FireSim)";
}
