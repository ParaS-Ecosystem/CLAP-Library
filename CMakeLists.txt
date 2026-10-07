# Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
# SPDX-License-Identifier: LGPL-3.0-or-later

cmake_minimum_required(VERSION 3.15)
project(CLAP VERSION 1.0.0 LANGUAGES C CXX)

include(GNUInstallDirs)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_BLAS_EXAMPLES "Build BLAS examples" ON)
option(BUILD_LAPACK_EXAMPLES "Build LAPACK examples" ON)
option(BUILD_DNN_EXAMPLES "Build DNN examples" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "[CLAP] C compiler:   ${CMAKE_C_COMPILER}")
message(STATUS "[CLAP] C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "[CLAP] Components: BLAS + LAPACK + DNN")
message(STATUS "[CLAP] Runtime backend selection: -cpu / -cuda / -rocm or CLAP_BACKEND=CPU|CUDA|ROCM")

# -----------------------------------------------------------------------------
# ONE CLAP LIBRARY
# BLAS + LAPACK + DNN are all part of this target.
# Produces libclap.so.
# -----------------------------------------------------------------------------

set(CLAP_COMMON_SOURCES
    src/common/dyn_backends.cpp
    src/common/blas_factory.cpp

    src/common/lapack_dyn.cpp
    src/common/lapack_factory.cpp
)

file(GLOB_RECURSE CLAP_BLAS_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/blas/*.cpp"
)

file(GLOB_RECURSE CLAP_LAPACK_SOURCES
    CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/lapack/*.cpp"
)

# DNN: oneDNN (CPU), cuDNN (NVIDIA), MIOpen (AMD); vendor libraries are
# loaded at runtime by src/common/dnn_dyn_backends.cpp.
set(CLAP_DNN_SOURCES
    src/common/dnn_dyn_backends.cpp
    src/common/dnn_factory.cpp

    src/onednn/onednn_backend.cpp
    src/cudnn/cudnn_backend.cpp
    src/miopen/miopen_backend.cpp
)

add_library(clap SHARED
    ${CLAP_COMMON_SOURCES}
    ${CLAP_BLAS_SOURCES}
    ${CLAP_LAPACK_SOURCES}
    ${CLAP_DNN_SOURCES}
)

target_include_directories(clap
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/BLAS>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/LAPACK>

        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/BLAS>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/LAPACK>
)

target_link_libraries(clap PUBLIC ${CMAKE_DL_LIBS} pthread m)

# -----------------------------------------------------------------------------
# Examples. They are built from source together with the libraries so stale
# prebuilt binaries in examples/ can never be mistaken for current results.
# -----------------------------------------------------------------------------
if(BUILD_BLAS_EXAMPLES)
    set(CLAP_BLAS_EXAMPLES
        level1/axpy_d level1/axpy_f
        level1/copy_d level1/copy_f
        level1/dot_d  level1/dot_f
        level1/scal_d level1/scal_f
        level1/swap_d level1/swap_f
        level2/gemv_d level2/gemv_f
        level2/ger_f
        level2/sbmv_d level2/sbmv_f
        level2/symv_d level2/symv_f
        level2/trmv_d level2/trmv_f
        level2/trsv_d level2/trsv_f
        level3/gemm_d level3/gemm_f
        level3/symm_d level3/symm_f
        level3/syrk_d level3/syrk_f
        level3/trmm_d level3/trmm_f
        level3/trsm_d level3/trsm_f
    )

    foreach(rel IN LISTS CLAP_BLAS_EXAMPLES)
        string(REPLACE "/" "_" target "blas_${rel}")
        get_filename_component(exe "${rel}" NAME)
        get_filename_component(group "${rel}" DIRECTORY)
        add_executable(${target} "examples/${rel}.cpp")
        set_target_properties(${target} PROPERTIES
            OUTPUT_NAME "${exe}"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/${group}"
        )
        target_link_libraries(${target} PRIVATE clap)
    endforeach()
endif()

if(BUILD_LAPACK_EXAMPLES)
    foreach(ex gesv_d getrf_getrs_f potrf_d syev_d gesvd_d)
        add_executable(lapack_${ex} "examples/lapack/${ex}.cpp")
        set_target_properties(lapack_${ex} PROPERTIES
            OUTPUT_NAME "${ex}"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/lapack"
        )
        target_link_libraries(lapack_${ex} PRIVATE clap)
    endforeach()

endif()

if(BUILD_DNN_EXAMPLES)
    set(CLAP_DNN_EXAMPLES
        activation/relu

        convolution/convolution
        convolution/convolution_backward_bias
        convolution/convolution_gpu_test
        convolution/convolution_transpose
        convolution/fused_conv_bias_activation
        convolution/one

        dropout/dropout
        normalization/batchnorm
        normalization/layernorm
        normalization/lrn
        elementwise/add
        elementwise/multiply
        elementwise/min
        elementwise/max
        reduction/sum
        reduction/product
        pooling/maxpool
        softmax/softmax

        # Framework execution contract / Param2 primitives
        activation/silu
        normalization/rmsnorm
        softmax/softmax_axis
        softmax/log_softmax
        attention/sdpa
        validation/param2_dnn_validation
    )

    foreach(rel IN LISTS CLAP_DNN_EXAMPLES)
        string(REPLACE "/" "_" target "dnn_${rel}")
        get_filename_component(exe "${rel}" NAME)
        get_filename_component(group "${rel}" DIRECTORY)
        add_executable(${target} "examples/dnn/${rel}.cpp")
        set_target_properties(${target} PROPERTIES
            OUTPUT_NAME "${exe}"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples/dnn/${group}"
        )
        target_link_libraries(${target} PRIVATE clap)
    endforeach()
endif()

# -----------------------------------------------------------------------------
# Install the library and all public headers with one `cmake --install`.
# -----------------------------------------------------------------------------
install(TARGETS clap 
    EXPORT clapTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT clapTargets
    NAMESPACE clap::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clap
)

