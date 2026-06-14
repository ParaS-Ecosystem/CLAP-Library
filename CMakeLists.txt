# -----------------------------------------------------------------------------
# This file is part of the CLAP library — an implementation of
# high-performance linear algebra routines based on the BLAS standard.
#
# CLAP provides a unified math interface across architectures,
# enabling portable and optimized performance on CPUs and GPUs.
#
# Copyright (C) The CLAP Contributors
#
# CLAP is released under the GNU Lesser General Public License (LGPL) v3.
# See the file LICENSE in the project root for full license details.
# -----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.15)
project(clap VERSION 1.0.0 LANGUAGES C CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "[CLAP] Using C compiler:   ${CMAKE_C_COMPILER}")
message(STATUS "[CLAP] Using C++ compiler: ${CMAKE_CXX_COMPILER}")

message(STATUS "[CLAP] Using Dynamic BLAS Factory via dlopen")

add_library(clap SHARED
    src/common/dyn_backends.cpp
    src/common/blas_factory.cpp

    src/openblas/openblas_backend.cpp
    src/cublas/cublas_backend.cpp
    src/rocblas/rocblas_backend.cpp

    src/openblas/level1.cpp
    src/openblas/level2.cpp
    src/openblas/level3.cpp

    src/cublas/level1.cpp
    src/cublas/level2.cpp
    src/cublas/level3.cpp

    src/rocblas/level1.cpp
    src/rocblas/level2.cpp
    src/rocblas/level3.cpp
)

target_include_directories(clap PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(clap PUBLIC ${CMAKE_DL_LIBS} pthread m)

include(GNUInstallDirs)

install(TARGETS clap
    EXPORT clapTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT clapTargets
    NAMESPACE clap::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clap
)
