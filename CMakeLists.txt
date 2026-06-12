cmake_minimum_required(VERSION 3.15)
project(clap_cpu VERSION 1.0.0 LANGUAGES C CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(CLAP_BUILD_FUNCTIONAL_TESTS "Build functional tests and run on 'make'" OFF)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "[CLAP] Using C compiler:   ${CMAKE_C_COMPILER}")
message(STATUS "[CLAP] Using C++ compiler: ${CMAKE_CXX_COMPILER}")

message(STATUS "[CLAP] Using Dynamic BLAS Factory via dlopen")

add_library(clapblas SHARED
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

target_include_directories(clapblas PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(clapblas PUBLIC ${CMAKE_DL_LIBS} pthread m)

if(CLAP_BUILD_FUNCTIONAL_TESTS)
    add_subdirectory(tests)

    add_custom_target(run_all_OK ALL
        COMMAND $<TARGET_FILE:run_all_tests>
        DEPENDS run_all_tests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running CLAP functional tests"
    )

    set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)
endif()

include(GNUInstallDirs)

install(TARGETS clapblas
    EXPORT clapblasTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT clapblasTargets
    NAMESPACE clap::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/clap
)