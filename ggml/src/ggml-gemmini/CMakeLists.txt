ggml_add_backend_library(ggml-gemmini
    ggml-gemmini.cpp
)

# Gemmini currently behaves like a CPU-attached accelerator using host-visible buffers.
# This lets us reuse CPU buffer types in the backend device interface.
if (TARGET ggml-cpu)
    target_link_libraries(ggml-gemmini PRIVATE ggml-cpu)
endif()

target_compile_definitions(ggml-gemmini PRIVATE GGML_USE_GEMMINI)

# Optional hooks for later when you wire in a real Gemmini runtime.
#
# Example:
#   cmake -B build -DGGML_GEMMINI=ON \
#       -DGEMMINI_INCLUDE_DIR=/path/to/gemmini/software/libgemmini \
#       -DGEMMINI_LIBRARY=/path/to/libgemmini.a
#
if (DEFINED GEMMINI_INCLUDE_DIR)
    target_include_directories(ggml-gemmini PRIVATE ${GEMMINI_INCLUDE_DIR})
endif()

if (DEFINED GEMMINI_LIBRARY)
    target_link_libraries(ggml-gemmini PRIVATE ${GEMMINI_LIBRARY})
endif()