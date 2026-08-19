// ggml-gemmini-kernel.cpp

#include "ggml-gemmini-kernel.h"

#if defined(__riscv)

extern "C" {
#include "include/gemmini.h"
}

#include <cstdint>

static_assert(sizeof(elem_t) == sizeof(int8_t));
static_assert(sizeof(acc_t)  == sizeof(int32_t));

#ifndef ACC_READ_FULL_WIDTH
#error "Gemmini backend requires full-width accumulator reads"
#endif

bool ggml_gemmini_hw_init() {
    gemmini_flush(0);
    gemmini_fence();

    return true;
}

bool ggml_gemmini_mul_mat_i8_i32(
        const int8_t * src0_nk,
        const int8_t * src1_mk,
        int32_t * dst_mn,
        size_t k,
        size_t n,
        size_t m) {

    tiled_matmul_auto(
        m,                          // I
        n,                          // J
        k,                          // K

        reinterpret_cast<const elem_t *>(src1_mk),
        reinterpret_cast<const elem_t *>(src0_nk),

        nullptr,                    // no bias
        reinterpret_cast<void *>(dst_mn),

        k,                          // A row stride
        k,                          // B row stride
        0,                          // D row stride
        n,                          // C row stride

        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        static_cast<scale_acc_t>(1),

        NO_ACTIVATION,
        ACC_SCALE_IDENTITY,
        0,                          // bert scale

        false,                      // repeating bias

        false,                      // transpose A
        true,                       // transpose B

        true,                       // full C => acc_t / int32
        false,                      // low D

        0,                          // weightA
        WS                          // weight-stationary
    );

    return true;
}

#else

bool ggml_gemmini_hw_init() {
    return false;
}

bool ggml_gemmini_mul_mat_i8_i32(
        const int8_t *,
        const int8_t *,
        int32_t *,
        size_t,
        size_t,
        size_t) {

    return false;
}

#endif