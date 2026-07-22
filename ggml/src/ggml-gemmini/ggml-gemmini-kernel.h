#pragma once

#include "ggml-gemmini-runtime.h"

bool ggml_gemmini_kernel_is_available();

ggml_gemmini_runtime_status ggml_gemmini_kernel_matmul_i8(const ggml_gemmini_matmul_i8_params & params);
