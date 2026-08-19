#include <cstdio>
#include <cstdint>
#include <cstring>

#if !defined(__riscv)
#error "test-gemmini-kernel must be compiled for RISC-V"
#endif

#include "include/gemmini.h"

static_assert(XCUSTOM_ACC == 3, "Expected Gemmini on CUSTOM_3");
static_assert(DIM == 16, "Expected a 16x16 Gemmini array");
static_assert(sizeof(elem_t) == 1, "Expected int8 Gemmini elem_t");
static_assert(sizeof(acc_t) == 4, "Expected int32 Gemmini acc_t");

// -----------------------------------------------------------------------------
// Small-width output path: always valid for the current generated configuration.
// Values are intentionally kept small so the correct answer fits in int8_t and
// output scaling/saturation cannot hide a layout error.
// -----------------------------------------------------------------------------

template <size_t M, size_t N, size_t K>
static bool run_small_output_case(const char * name, bool identity_a) {
    alignas(64) static elem_t A[M * K];
    alignas(64) static elem_t B[N * K];
    alignas(64) static elem_t C[M * N];
    alignas(64) static elem_t reference[M * N];

    std::memset(A, 0, sizeof(A));
    std::memset(B, 0, sizeof(B));
    std::memset(C, 0, sizeof(C));
    std::memset(reference, 0, sizeof(reference));

    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            int value = identity_a
                ? ((m == k) ? 1 : 0)
                : static_cast<int>((m + 2 * k + 1) % 3) - 1; // -1..1
            A[m * K + k] = static_cast<elem_t>(value);
        }
    }

    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
            const int value =
                static_cast<int>((2 * n + k + 2) % 3) - 1; // -1..1
            B[n * K + k] = static_cast<elem_t>(value);
        }
    }

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum +=
                    static_cast<int32_t>(A[m * K + k]) *
                    static_cast<int32_t>(B[n * K + k]);
            }

            if (sum < -128 || sum > 127) {
                std::printf(
                    "INTERNAL TEST ERROR: reference does not fit int8: %d\n",
                    static_cast<int>(sum));
                return false;
            }

            reference[m * N + n] = static_cast<elem_t>(sum);
        }
    }

    std::printf(
        "Running %-28s M=%zu N=%zu K=%zu, output=int8\n",
        name, M, N, K);
    std::fflush(stdout);

    gemmini_flush(0);

    tiled_matmul_auto(
        M, N, K,
        A,
        B,
        nullptr,
        C,
        K,                          // A row stride
        K,                          // B row stride; B stored N x K
        0,
        N,                          // C row stride
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        static_cast<scale_acc_t>(1),
        NO_ACTIVATION,
        ACC_SCALE_IDENTITY,
        0,
        false,                      // repeating_bias
        false,                      // transpose_A
        true,                       // transpose_B
        false,                      // full_C: small-width elem_t output
        false,                      // low_D
        0,
        WS
    );

    gemmini_fence();

    int errors = 0;
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            const int got = static_cast<int>(C[m * N + n]);
            const int expected = static_cast<int>(reference[m * N + n]);

            if (got != expected) {
                if (errors < 16) {
                    std::printf(
                        "  mismatch C[%zu,%zu]: expected=%d got=%d\n",
                        m, n, expected, got);
                }
                ++errors;
            }
        }
    }

    if (errors) {
        std::printf("FAIL: %s had %d mismatches\n", name, errors);
        return false;
    }

    std::printf("PASS: %s\n", name);
    return true;
}

// -----------------------------------------------------------------------------
// Full accumulator-width output path. This is the path the planned llama.cpp
// I8 x I8 -> I32 backend needs. Only compile/run it if the generated hardware
// header says the deployed Gemmini supports full-width accumulator reads.
// -----------------------------------------------------------------------------

#ifdef ACC_READ_FULL_WIDTH

template <size_t M, size_t N, size_t K>
static bool run_full_output_case(const char * name) {
    alignas(64) static elem_t A[M * K];
    alignas(64) static elem_t B[N * K];
    alignas(64) static acc_t  C[M * N];
    alignas(64) static acc_t  reference[M * N];

    std::memset(A, 0, sizeof(A));
    std::memset(B, 0, sizeof(B));
    std::memset(C, 0, sizeof(C));
    std::memset(reference, 0, sizeof(reference));

    for (size_t m = 0; m < M; ++m) {
        for (size_t k = 0; k < K; ++k) {
            A[m * K + k] =
                static_cast<elem_t>(
                    static_cast<int>((m * 3 + k * 5 + 1) % 15) - 7);
        }
    }

    for (size_t n = 0; n < N; ++n) {
        for (size_t k = 0; k < K; ++k) {
            B[n * K + k] =
                static_cast<elem_t>(
                    static_cast<int>((n * 5 + k * 2 + 3) % 17) - 8);
        }
    }

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; ++k) {
                sum +=
                    static_cast<int32_t>(A[m * K + k]) *
                    static_cast<int32_t>(B[n * K + k]);
            }
            reference[m * N + n] = static_cast<acc_t>(sum);
        }
    }

    std::printf(
        "Running %-28s M=%zu N=%zu K=%zu, output=int32\n",
        name, M, N, K);
    std::fflush(stdout);

    gemmini_flush(0);

    tiled_matmul_auto(
        M, N, K,
        A,
        B,
        nullptr,
        C,
        K,
        K,
        0,
        N,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        static_cast<scale_acc_t>(1),
        NO_ACTIVATION,
        ACC_SCALE_IDENTITY,
        0,
        false,
        false,
        true,
        true,                       // full_C: acc_t / int32 output
        false,
        0,
        WS
    );

    gemmini_fence();

    int errors = 0;
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            const int32_t got = static_cast<int32_t>(C[m * N + n]);
            const int32_t expected =
                static_cast<int32_t>(reference[m * N + n]);

            if (got != expected) {
                if (errors < 16) {
                    std::printf(
                        "  mismatch C[%zu,%zu]: expected=%d got=%d\n",
                        m, n,
                        static_cast<int>(expected),
                        static_cast<int>(got));
                }
                ++errors;
            }
        }
    }

    if (errors) {
        std::printf("FAIL: %s had %d mismatches\n", name, errors);
        return false;
    }

    std::printf("PASS: %s\n", name);
    return true;
}

#endif

int main() {
    std::printf("Gemmini direct kernel test\n");
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

    std::printf("\n");

    bool ok = true;

    // Exact array-size baseline.
    ok &= run_small_output_case<16, 16, 16>(
        "16x16x16 baseline", true);

    // Exercises auto-tiler padding on all dimensions.
    ok &= run_small_output_case<17, 19, 23>(
        "17x19x23 odd shape", false);

#ifdef ACC_READ_FULL_WIDTH
    // This is the eventual llama.cpp hardware output format.
    ok &= run_full_output_case<17, 19, 23>(
        "17x19x23 full-width");
#else
    std::printf(
        "SKIP: I32/full-accumulator output test because "
        "ACC_READ_FULL_WIDTH is not enabled in gemmini_params.h\n");
#endif

    if (!ok) {
        std::printf("\nGemmini kernel test FAILED\n");
        return 1;
    }

    std::printf("\nGemmini kernel test PASSED\n");
    return 0;
}
