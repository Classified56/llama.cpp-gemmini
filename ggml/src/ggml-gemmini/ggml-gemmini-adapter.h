#pragma once

#include "ggml.h"

// Model-facing Gemmini MUL_MAT adapters.
//
// Supported first-stage contracts:
//   F32  [K,N] x F32 [K,M] -> F32 [N,M]
//   Q8_0 [K,N] x F32 [K,M] -> F32 [N,M]
//
// Internally both are converted to:
//   I8 [K,N] x I8 [K,M] -> I32 [N,M]
// through the already-validated Gemmini kernel.

bool ggml_gemmini_adapter_supports(const struct ggml_tensor * op);
bool ggml_gemmini_adapter_compute(struct ggml_tensor * dst);
