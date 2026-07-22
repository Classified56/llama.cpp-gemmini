ggml_add_backend_library(ggml-gemmini
    ggml-gemmini.cpp
    ggml-gemmini-runtime.cpp
    ggml-gemmini-kernel.cpp
)

target_include_directories(ggml-gemmini PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Gemmini is a CPU-attached accelerator using host-visible GGML buffers.
if (TARGET ggml-cpu)
    target_link_libraries(ggml-gemmini PRIVATE ggml-cpu)
endif()

option(GGML_GEMMINI_FORCE_AVAILABLE
    "Expose the Gemmini backend for native registration/reference tests"
    OFF
)

option(GGML_GEMMINI_ENABLE_NATIVE_TEST_MATMUL
    "Enable native reference MUL_MAT execution in the Gemmini backend"
    OFF
)

option(GGML_GEMMINI_ENABLE_TILED_MATMUL
    "Compile the bundled Gemmini tiled_matmul_auto runtime"
    OFF
)

set(GGML_GEMMINI_VENDOR_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/vendor/gemmini"
    CACHE PATH
    "Root of the bundled Gemmini software interface"
)

set(GGML_GEMMINI_PARAMS_FILE
    ""
    CACHE FILEPATH
    "Optional generated gemmini_params.h matching another hardware configuration"
)

if (GGML_GEMMINI_FORCE_AVAILABLE)
    target_compile_definitions(ggml-gemmini PRIVATE
        GGML_GEMMINI_FORCE_AVAILABLE)
endif()

if (GGML_GEMMINI_ENABLE_NATIVE_TEST_MATMUL)
    target_compile_definitions(ggml-gemmini PRIVATE
        GGML_GEMMINI_ENABLE_NATIVE_TEST_MATMUL)
endif()

if (GGML_GEMMINI_ENABLE_TILED_MATMUL)
    if (NOT EXISTS "${GGML_GEMMINI_VENDOR_ROOT}/include/gemmini.h")
        message(FATAL_ERROR
            "Bundled Gemmini header missing: "
            "${GGML_GEMMINI_VENDOR_ROOT}/include/gemmini.h")
    endif()

    if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
        message(FATAL_ERROR
            "GGML_GEMMINI_ENABLE_TILED_MATMUL requires a RISC-V target. "
            "Use GGML_GEMMINI_FORCE_AVAILABLE for native registry tests.")
    endif()

    # Optional build-tree overlay for a generated parameter header that matches
    # a different Gemmini bitstream. It overrides only include/gemmini_params.h;
    # the bundled API and RoCC macros remain in use.
    if (GGML_GEMMINI_PARAMS_FILE)
        if (NOT EXISTS "${GGML_GEMMINI_PARAMS_FILE}")
            message(FATAL_ERROR
                "GGML_GEMMINI_PARAMS_FILE does not exist: "
                "${GGML_GEMMINI_PARAMS_FILE}")
        endif()

        set(GGML_GEMMINI_CONFIG_OVERLAY
            "${CMAKE_CURRENT_BINARY_DIR}/gemmini-config-overlay")
        file(MAKE_DIRECTORY
            "${GGML_GEMMINI_CONFIG_OVERLAY}/include")
        configure_file(
            "${GGML_GEMMINI_PARAMS_FILE}"
            "${GGML_GEMMINI_CONFIG_OVERLAY}/include/gemmini_params.h"
            COPYONLY
        )

        target_include_directories(ggml-gemmini SYSTEM BEFORE PRIVATE
            "${GGML_GEMMINI_CONFIG_OVERLAY}")
        message(STATUS
            "Gemmini parameters: ${GGML_GEMMINI_PARAMS_FILE}")
    else()
        message(STATUS
            "Gemmini parameters: bundled vendor/gemmini/include/gemmini_params.h")
    endif()

    # Preserve Gemmini's original relative includes:
    #   include/gemmini_params.h
    #   include/gemmini_counter.h
    #   rocc-software/src/xcustom.h
    target_include_directories(ggml-gemmini SYSTEM PRIVATE
        "${GGML_GEMMINI_VENDOR_ROOT}")

    target_compile_definitions(ggml-gemmini PRIVATE
        GGML_GEMMINI_RUNTIME_HAS_HARDWARE)
endif()
