#ifndef GGML_GEMMINI_H
#define GGML_GEMMINI_H

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Initialize Gemmini backend
ggml_backend_t ggml_backend_gemmini_init(void);

// Check if Gemmini is available
bool ggml_backend_gemmini_is_available(void);

// Optional: Get device info
const char * ggml_backend_gemmini_get_device_name(void);

#ifdef __cplusplus
}
#endif

#endif // GGML_GEMMINI_H
