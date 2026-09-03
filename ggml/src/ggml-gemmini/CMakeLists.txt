ggml_add_backend_library(
    ggml-gemmini
    ggml-gemmini.cpp
    ggml-gemmini-kernel.cpp
    ggml-gemmini-adapter.cpp
)

if (TARGET ggml-cpu)
    target_link_libraries(
        ggml-gemmini
        PRIVATE ggml-cpu
    )
endif()

target_include_directories(
    ggml-gemmini
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/vendor/gemmini
)

option(
    GGML_GEMMINI_FORCE_AVAILABLE
    "Force Gemmini backend availability for host testing"
    OFF
)

option(
    GGML_GEMMINI_ENABLE_NATIVE_TEST_MATMUL
    "Enable CPU reference Gemmini matmul for host testing"
    OFF
)

option(
    GGML_GEMMINI_ENABLE_TILED_MATMUL
    "Enable real Gemmini tiled matmul"
    OFF
)

if (GGML_GEMMINI_FORCE_AVAILABLE)
    target_compile_definitions(
        ggml-gemmini
        PRIVATE GGML_GEMMINI_FORCE_AVAILABLE
    )
endif()

if (GGML_GEMMINI_ENABLE_NATIVE_TEST_MATMUL)
    target_compile_definitions(
        ggml-gemmini
        PRIVATE GGML_GEMMINI_ENABLE_NATIVE_TEST_MATMUL
    )
endif()

if (GGML_GEMMINI_ENABLE_TILED_MATMUL)
    target_compile_definitions(
        ggml-gemmini
        PRIVATE GGML_GEMMINI_ENABLE_TILED_MATMUL
    )
endif()

if (UNIX AND NOT APPLE)
    target_link_libraries(
        ggml-gemmini
        PRIVATE m
    )
endif()