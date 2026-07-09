ggml_add_backend_library(ggml-gemmini
    ggml-gemmini.cpp
)

# Use CPU host buffers. Gemmini is treated as a CPU-attached accelerator.
if (TARGET ggml-cpu)
    target_link_libraries(ggml-gemmini PRIVATE ggml-cpu)
endif()

# Path to Chipyard/Gemmini software headers.
#
# Expected headers later:
#   gemmini.h
#   gemmini_params.h
#   optionally gemmini_nn.h, depending on your Gemmini version
#
# Example:
#   -DGEMMINI_INCLUDE_DIR=$CHIPYARD/generators/gemmini/software/gemmini-rocc-tests/include
set(GEMMINI_INCLUDE_DIR "" CACHE PATH "Path to Gemmini software include directory")

# Optional library path if you build/use libgemmini.
set(GEMMINI_LIBRARY "" CACHE FILEPATH "Path to libgemmini.a or equivalent")

option(GGML_GEMMINI_FORCE_AVAILABLE
    "Assume Gemmini hardware is available at runtime"
    OFF
)

# Compile the matmul eligibility stub. This does not implement Gemmini compute;
# it only allows the scheduler to hand synthetic I8xI8->I32 MUL_MAT nodes to
# the backend, where graph_compute will intentionally abort until implemented.
option(GGML_GEMMINI_ENABLE_MATMUL_STUB
    "Advertise a narrow synthetic I8xI8->I32 MUL_MAT stub for dispatch debugging"
    OFF
)

option(GGML_GEMMINI_ENABLE_TILED_MATMUL
    "Enable calls to Gemmini tiled_matmul_auto"
    OFF
)

if (GEMMINI_INCLUDE_DIR)
    target_include_directories(ggml-gemmini PRIVATE ${GEMMINI_INCLUDE_DIR})
    target_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_HAS_HEADERS)
endif()

if (GEMMINI_LIBRARY)
    target_link_libraries(ggml-gemmini PRIVATE ${GEMMINI_LIBRARY})
endif()

if (GGML_GEMMINI_FORCE_AVAILABLE)
    target_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_FORCE_AVAILABLE)
endif()

if (GGML_GEMMINI_ENABLE_MATMUL_STUB)
    target_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_ENABLE_MATMUL_STUB)
endif()

if (GGML_GEMMINI_ENABLE_TILED_MATMUL)
    if (NOT GEMMINI_INCLUDE_DIR)
        message(FATAL_ERROR "GGML_GEMMINI_ENABLE_TILED_MATMUL requires GEMMINI_INCLUDE_DIR")
    endif()

    target_compile_definitions(ggml-gemmini PRIVATE GGML_GEMMINI_ENABLE_TILED_MATMUL)
endif()
