#pragma once

#include <cstddef>
#include <cstdint>

bool ggml_gemmini_hw_init();

bool ggml_gemmini_mul_mat_i8_i32(
    const int8_t * src0_nk,
    const int8_t * src1_mk,
    int32_t * dst_mn,
    size_t k,
    size_t n,
    size_t m);