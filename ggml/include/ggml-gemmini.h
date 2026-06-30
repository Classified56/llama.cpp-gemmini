#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t     ggml_backend_gemmini_init(void);

GGML_BACKEND_API bool               ggml_backend_is_gemmini(ggml_backend_t backend);

GGML_BACKEND_API bool               ggml_backend_gemmini_is_available(void);

GGML_BACKEND_API const char *       ggml_backend_gemmini_get_device_name(void);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_gemmini_reg(void);

#ifdef __cplusplus
}
#endif