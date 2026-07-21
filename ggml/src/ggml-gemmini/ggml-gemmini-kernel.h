#pragma once

#include "ggml-gemmini-runtime.h"

// Hardware-only entry point.
//
// This function is deliberately separate from the GGML backend and generic
// runtime validation. It is the function to mark noinline and inspect using
// riscv64-unknown-linux-gnu-objdump once tiled_matmul_auto is added.

ggml_gemmini_runtime_status ggml_gemmini_kernel_matmul_i8(
    const ggml_gemmini_matmul_i8_params & params);
