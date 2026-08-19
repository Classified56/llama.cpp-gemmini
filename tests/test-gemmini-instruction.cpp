#include <cstdio>
#include <cstdint>

#if !defined(__riscv)
#error "test-gemmini-instruction must be compiled for RISC-V"
#endif

#include "include/gemmini.h"

static_assert(XCUSTOM_ACC == 3, "Expected Gemmini on CUSTOM_3");
static_assert(DIM == 16, "Expected a 16x16 Gemmini array");
static_assert(sizeof(elem_t) == 1, "Expected int8 Gemmini elem_t");
static_assert(sizeof(acc_t) == 4, "Expected int32 Gemmini acc_t");

int main() {
    std::printf("Gemmini instruction smoke test\n");
    std::printf("  XCUSTOM_ACC    = %d\n", XCUSTOM_ACC);
    std::printf("  DIM            = %d\n", DIM);
    std::printf("  BANK_NUM       = %d\n", BANK_NUM);
    std::printf("  BANK_ROWS      = %d\n", BANK_ROWS);
    std::printf("  ACC_ROWS       = %d\n", ACC_ROWS);
    std::printf("  sizeof(elem_t) = %zu\n", sizeof(elem_t));
    std::printf("  sizeof(acc_t)  = %zu\n", sizeof(acc_t));

#ifdef ACC_READ_SMALL_WIDTH
    std::printf("  ACC_READ_SMALL_WIDTH = yes\n");
#else
    std::printf("  ACC_READ_SMALL_WIDTH = no\n");
#endif

#ifdef ACC_READ_FULL_WIDTH
    std::printf("  ACC_READ_FULL_WIDTH  = yes\n");
#else
    std::printf("  ACC_READ_FULL_WIDTH  = no\n");
#endif

    std::printf("\nIssuing gemmini_flush(0)...\n");
    std::fflush(stdout);

    // Real Gemmini RoCC instruction.
    gemmini_flush(0);

    // In the current Gemmini software header this is a RISC-V fence.
    gemmini_fence();

    std::printf("PASS: Gemmini custom instruction returned successfully\n");
    return 0;
}
