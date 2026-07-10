#include "ggml-backend.h"

#include <cstdio>

static const char * dev_type_name(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "CPU";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "GPU";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "IGPU";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "ACCEL";
        case GGML_BACKEND_DEVICE_TYPE_META:  return "META";
        default:                             return "UNKNOWN";
    }
}

int main() {
    ggml_backend_load_all();

    std::printf("Backend registries: %zu\n", ggml_backend_reg_count());

    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);

        std::printf("reg[%zu]: %s, devices=%zu\n",
            i,
            ggml_backend_reg_name(reg),
            ggml_backend_reg_dev_count(reg));
    }

    std::printf("\nBackend devices: %zu\n", ggml_backend_dev_count());

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);

        std::printf("dev[%zu]: name=%s, desc=%s, type=%s, host_buffer=%d, from_host_ptr=%d\n",
            i,
            props.name ? props.name : "(null)",
            props.description ? props.description : "(null)",
            dev_type_name(props.type),
            props.caps.host_buffer,
            props.caps.buffer_from_host_ptr);
    }

    ggml_backend_dev_t gemmini = ggml_backend_dev_by_name("Gemmini");
    std::printf("\nGemmini lookup: %s\n", gemmini ? "FOUND" : "NOT FOUND");

    if (gemmini) {
        ggml_backend_t backend = ggml_backend_dev_init(gemmini, nullptr);
        std::printf("Gemmini init: %s\n", backend ? "OK" : "FAILED");

        if (backend) {
            std::printf("Gemmini backend name: %s\n", ggml_backend_name(backend));
            ggml_backend_free(backend);
        }
    }

    return gemmini ? 0 : 1;
}