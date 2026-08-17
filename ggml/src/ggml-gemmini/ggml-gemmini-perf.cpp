#include "ggml-gemmini-perf.h"

#include <cstddef>
#include <cstdint>

#if defined(__riscv) && defined(GGML_GEMMINI_PERF_ENABLED)

extern "C" {
#include "include/gemmini.h"
}

namespace {

struct counter_definition {
    const char * name;
    std::size_t code;
};

// First profiling set: broad pipeline activity.
//
// There are eight physical counter slots, indexed 0 through 7.
constexpr counter_definition k_counter_set[] = {
    { "main_ld_cycles",      MAIN_LD_CYCLES       },
    { "main_st_cycles",      MAIN_ST_CYCLES       },
    { "main_ex_cycles",      MAIN_EX_CYCLES       },
    { "load_dma_wait",       LOAD_DMA_WAIT_CYCLE  },
    { "store_dma_wait",      STORE_DMA_WAIT_CYCLE },
    { "rdma_active",         RDMA_ACTIVE_CYCLE    },
    { "wdma_active",         WDMA_ACTIVE_CYCLE    },
    { "execute_active",      EXE_ACTIVE_CYCLE     },
};

constexpr counter_definition k_stall[] = {
    { "rdma_tlb_wait", RDMA_TLB_WAIT_CYCLES },
    { "rdma_tl_wait", RDMA_TL_WAIT_CYCLES },
    { "wdma_tlb_wait", WDMA_TLB_WAIT_CYCLES },
    { "wdma_tl_wait", WDMA_TL_WAIT_CYCLES },
    { "scratchpad_a_wait", SCRATCHPAD_A_WAIT_CYCLE },
    { "scratchpad_b_wait", SCRATCHPAD_B_WAIT_CYCLE },
    { "scratchpad_d_wait", SCRATCHPAD_D_WAIT_CYCLE },
    { "reservation_full", RESERVATION_STATION_FULL_CYCLES },
};

constexpr counter_definition k_matmul[] = {
    { "execute_active", EXE_ACTIVE_CYCLE },
    { "loop_matmul_active", LOOP_MATMUL_ACTIVE_CYCLES },
    { "preload_hazard", EXE_PRELOAD_HAZ_CYCLE },
    { "overlap_hazard", EXE_OVERLAP_HAZ_CYCLE },
    { "control_queue_block", EXE_CONTROL_Q_BLOCK_CYCLE },
    { "reservation_active", RESERVATION_STATION_ACTIVE_CYCLES },
    { "acc_a_wait", ACC_A_WAIT_CYCLE },
    { "acc_b_wait", ACC_B_WAIT_CYCLE },
};

static_assert(
    sizeof(k_counter_set) / sizeof(k_counter_set[0]) ==
        GGML_GEMMINI_COUNTER_COUNT);

bool g_initialized = false;

static inline std::uint64_t read_target_cycle() {
    std::uint64_t value;

    asm volatile(
        "rdcycle %0"
        : "=r"(value)
        :
        : "memory");

    return value;
}

void read_counter_snapshot(
    std::uint32_t values[GGML_GEMMINI_COUNTER_COUNT]) {

    // Freeze a coherent copy of all configured counters.
    counter_snapshot_take();

    for (std::size_t index = 0;
         index < GGML_GEMMINI_COUNTER_COUNT;
         ++index) {

        values[index] = counter_read(index);
    }

    // Return the counter block to normal accumulation.
    counter_snapshot_reset();
}

} // namespace

bool ggml_gemmini_perf_available() {
    return true;
}

bool ggml_gemmini_perf_initialize() {
    // Reset first because reset may also clear prior configuration.
    counter_reset();

    for (std::size_t index = 0;
         index < GGML_GEMMINI_COUNTER_COUNT;
         ++index) {

        counter_configure(index, k_counter_set[index].code);
    }

    counter_snapshot_reset();
    gemmini_fence();

    g_initialized = true;
    return true;
}

void ggml_gemmini_perf_begin(
    ggml_gemmini_perf_snapshot * snapshot) {

    if (snapshot == nullptr) {
        return;
    }

    if (!g_initialized) {
        ggml_gemmini_perf_initialize();
    }

    // Finish any previous accelerator work so it is not attributed
    // to the next expert.
    gemmini_fence();

    // Counter reads are intentionally outside the CPU cycle interval.
    read_counter_snapshot(snapshot->counters);

    snapshot->target_cycles = read_target_cycle();
}

void ggml_gemmini_perf_end(
    ggml_gemmini_perf_snapshot * snapshot) {

    if (snapshot == nullptr) {
        return;
    }

    if (!g_initialized) {
        ggml_gemmini_perf_initialize();
    }

    // Wait for all expert accelerator work to complete.
    gemmini_fence();

    // Read cycles before issuing the counter-read commands so their
    // software overhead is not charged to the expert.
    snapshot->target_cycles = read_target_cycle();

    read_counter_snapshot(snapshot->counters);
}

ggml_gemmini_perf_delta ggml_gemmini_perf_difference(
    const ggml_gemmini_perf_snapshot & begin,
    const ggml_gemmini_perf_snapshot & end) {

    ggml_gemmini_perf_delta result = {};

    result.target_cycles =
        end.target_cycles - begin.target_cycles;

    for (std::size_t index = 0;
         index < GGML_GEMMINI_COUNTER_COUNT;
         ++index) {

        // Unsigned subtraction naturally handles one 32-bit wrap.
        result.counters[index] =
            end.counters[index] - begin.counters[index];
    }

    return result;
}

const char * ggml_gemmini_perf_counter_name(
    std::size_t index) {

    if (index >= GGML_GEMMINI_COUNTER_COUNT) {
        return "invalid";
    }

    return k_counter_set[index].name;
}

#else

bool ggml_gemmini_perf_available() {
    return false;
}

bool ggml_gemmini_perf_initialize() {
    return false;
}

void ggml_gemmini_perf_begin(
    ggml_gemmini_perf_snapshot * snapshot) {

    if (snapshot != nullptr) {
        *snapshot = {};
    }
}

void ggml_gemmini_perf_end(
    ggml_gemmini_perf_snapshot * snapshot) {

    if (snapshot != nullptr) {
        *snapshot = {};
    }
}

ggml_gemmini_perf_delta ggml_gemmini_perf_difference(
    const ggml_gemmini_perf_snapshot &,
    const ggml_gemmini_perf_snapshot &) {

    return {};
}

const char * ggml_gemmini_perf_counter_name(
    std::size_t) {

    return "unavailable";
}

#endif