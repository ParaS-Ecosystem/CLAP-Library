// -----------------------------------------------------------------------------
// This file is part of the CLAP library — an implementation of
// high-performance linear algebra routines based on the BLAS standard.
//
// CLAP provides a unified math interface across architectures,
// enabling portable and optimized performance on CPUs and GPUs.
//
// Copyright (C) The CLAP Contributors
//
// CLAP is released under the GNU Lesser General Public License (LGPL) v3.
// See the file LICENSE in the project root for full license details.
// -----------------------------------------------------------------------------

#pragma once
#include "macros.hpp"
#include "types.hpp"
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>

namespace clap {

namespace dyn {

// CUDA functions
extern cudaError_t (*p_cudaMalloc)(void **ptr, size_t size);
extern cudaError_t (*p_cudaFree)(void *ptr);
extern cudaError_t (*p_cudaMemcpy)(void *dst, const void *src, size_t sizeBytes,
                                   cudaMemcpyKind kind);
extern cudaError_t (*p_cudaDeviceSynchronize)(void);
extern cublasStatus_t (*p_cublasSetPointerMode_v2)(cublasHandle_t handle,
                                                   cublasPointerMode_t mode);
extern cudaError_t (*p_cudaMemset)(void *, int64_t, size_t);

// cuBLAS functions

extern cublasStatus_t (*p_cublasCreate_v2)(cublasHandle_t *handle);
extern cublasStatus_t (*p_cublasDestroy_v2)(cublasHandle_t handle);
extern cublasStatus_t (*p_cublasSgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const float *alpha, const float *A,
    int64_t lda, const float *B, int64_t ldb, const float *beta, float *C,
    int64_t ldc);
extern cublasStatus_t (*p_cublasDgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const double *alpha, const double *A,
    int64_t lda, const double *B, int64_t ldb, const double *beta, double *C,
    int64_t ldc);

// level 1

extern cublasStatus_t (*p_cublasScopy_v2)(cublasHandle_t, int64_t,
                                          const float *, int64_t, float *,
                                          int64_t);
extern cublasStatus_t (*p_cublasDcopy_v2)(cublasHandle_t, int64_t,
                                          const double *, int64_t, double *,
                                          int64_t);

extern cublasStatus_t (*p_cublasSscal_v2)(cublasHandle_t, int64_t,
                                          const float *, float *, int64_t);
extern cublasStatus_t (*p_cublasDscal_v2)(cublasHandle_t, int64_t,
                                          const double *, double *, int64_t);

extern cublasStatus_t (*p_cublasSaxpy_v2)(cublasHandle_t, int64_t,
                                          const float *, const float *, int64_t,
                                          float *, int64_t);
extern cublasStatus_t (*p_cublasDaxpy_v2)(cublasHandle_t, int64_t,
                                          const double *, const double *,
                                          int64_t, double *, int64_t);

extern cublasStatus_t (*p_cublasSswap_v2)(cublasHandle_t, int64_t, float *,
                                          int64_t, float *, int64_t);
extern cublasStatus_t (*p_cublasDswap_v2)(cublasHandle_t, int64_t, double *,
                                          int64_t, double *, int64_t);

extern cublasStatus_t (*p_cublasSdot_v2)(cublasHandle_t, int64_t, const float *,
                                         int64_t, const float *, int64_t,
                                         float *);
extern cublasStatus_t (*p_cublasDdot_v2)(cublasHandle_t, int64_t,
                                         const double *, int64_t,
                                         const double *, int64_t, double *);

// level 2

extern cublasStatus_t (*p_cublasSgemv_v2)(cublasHandle_t handle,
                                          cublasOperation_t trans,
                                          std::int64_t m, std::int64_t n,
                                          const float *alpha, const float *A,
                                          std::int64_t lda, const float *x,
                                          std::int64_t incx, const float *beta,
                                          float *y, std::int64_t incy);
extern cublasStatus_t (*p_cublasDgemv_v2)(cublasHandle_t handle,
                                          cublasOperation_t trans,
                                          std::int64_t m, std::int64_t n,
                                          const double *alpha, const double *A,
                                          std::int64_t lda, const double *x,
                                          std::int64_t incx, const double *beta,
                                          double *y, std::int64_t incy);

extern cublasStatus_t (*p_cublasSsymv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, int64_t n,
                                          const float *alpha, const float *A,
                                          int64_t lda, const float *x,
                                          int64_t incx, const float *beta,
                                          float *y, int64_t incy);
extern cublasStatus_t (*p_cublasDsymv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, int64_t n,
                                          const double *alpha, const double *A,
                                          int64_t lda, const double *x,
                                          int64_t incx, const double *beta,
                                          double *y, int64_t incy);

extern cublasStatus_t (*p_cublasStrmv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag, std::int64_t n,
                                          const float *A, std::int64_t lda,
                                          float *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasDtrmv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag, std::int64_t n,
                                          const double *A, std::int64_t lda,
                                          double *X, std::int64_t incx);

extern cublasStatus_t (*p_cublasStrsv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag, std::int64_t n,
                                          const float *A, std::int64_t lda,
                                          float *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasDtrsv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag, std::int64_t n,
                                          const double *A, std::int64_t lda,
                                          double *X, std::int64_t incx);

extern cublasStatus_t (*p_cublasSsbmv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, std::int64_t n,
                                          std::int64_t k, const float *alpha,
                                          const float *A, std::int64_t lda,
                                          const float *X, std::int64_t incx,
                                          const float *beta, float *Y,
                                          std::int64_t incy);
extern cublasStatus_t (*p_cublasDsbmv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, std::int64_t n,
                                          std::int64_t k, const double *alpha,
                                          const double *A, std::int64_t lda,
                                          const double *X, std::int64_t incx,
                                          const double *beta, double *Y,
                                          std::int64_t incy);

// Level 3

/// GEMM
extern cublasStatus_t (*p_cublasSgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const float *alpha, const float *A,
    int64_t lda, const float *B, int64_t ldb, const float *beta, float *C,
    int64_t ldc);
extern cublasStatus_t (*p_cublasDgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const double *alpha, const double *A,
    int64_t lda, const double *B, int64_t ldb, const double *beta, double *C,
    int64_t ldc);

// SYMM
extern cublasStatus_t (*p_cublasSsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, int64_t, int64_t,
                                          const float *, const float *, int64_t,
                                          const float *, int64_t, const float *,
                                          float *, int64_t);
extern cublasStatus_t (*p_cublasDsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, int64_t, int64_t,
                                          const double *, const double *,
                                          int64_t, const double *, int64_t,
                                          const double *, double *, int64_t);

// SYR
extern cublasStatus_t (*p_cublasSsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                          cublasOperation_t, int64_t, int64_t,
                                          const float *, const float *, int64_t,
                                          const float *, float *, int64_t);
extern cublasStatus_t (*p_cublasDsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                          cublasOperation_t, int64_t, int64_t,
                                          const double *, const double *,
                                          int64_t, const double *, double *,
                                          int64_t);

// TRMM
extern cublasStatus_t (*p_cublasStrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int64_t, int64_t,
                                          const float *, const float *, int64_t,
                                          float *, int64_t, float *, int64_t);
extern cublasStatus_t (*p_cublasDtrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int64_t, int64_t,
                                          const double *, const double *,
                                          int64_t, double *, int64_t, double *,
                                          int64_t);

// TRSM
extern cublasStatus_t (*p_cublasStrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int64_t, int64_t,
                                          const float *, const float *, int64_t,
                                          float *, int64_t);
extern cublasStatus_t (*p_cublasDtrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int64_t, int64_t,
                                          const double *, const double *,
                                          int64_t, double *, int64_t);

// OpenBLAS functions

// Level 1

extern void (*p_cblas_scopy)(int64_t, const float *, int64_t, float *, int64_t);
extern void (*p_cblas_dcopy)(int64_t, const double *, int64_t, double *,
                             int64_t);

extern void (*p_cblas_sscal)(int64_t, float, float *, int64_t);
extern void (*p_cblas_dscal)(int64_t, double, double *, int64_t);

extern void (*p_cblas_saxpy)(int64_t, float, const float *, int64_t, float *,
                             int64_t);
extern void (*p_cblas_daxpy)(int64_t, double, const double *, int64_t, double *,
                             int64_t);

extern void (*p_cblas_sswap)(int64_t, float *, int64_t, float *, int64_t);
extern void (*p_cblas_dswap)(int64_t, double *, int64_t, double *, int64_t);

extern float (*p_cblas_sdot)(int64_t, const float *, int64_t, const float *,
                             int64_t);
extern double (*p_cblas_ddot)(int64_t, const double *, int64_t, const double *,
                              int64_t);

// LEVEL 2

extern void (*p_cblas_sgemv)(const CBLAS_ORDER Order,
                             const CBLAS_TRANSPOSE Trans, std::int64_t m,
                             std::int64_t n, const float alpha, const float *A,
                             std::int64_t lda, const float *x,
                             std::int64_t incx, const float beta, float *y,
                             std::int64_t incy);
extern void (*p_cblas_dgemv)(const CBLAS_ORDER Order,
                             const CBLAS_TRANSPOSE Trans, std::int64_t m,
                             std::int64_t n, const double alpha,
                             const double *A, std::int64_t lda, const double *x,
                             std::int64_t incx, const double beta, double *y,
                             std::int64_t incy);

extern void (*p_cblas_ssymv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             const int64_t N, const float alpha, const float *A,
                             const int64_t lda, const float *X,
                             const int64_t incX, const float beta, float *Y,
                             const int64_t incY);
extern void (*p_cblas_dsymv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             const int64_t N, const double alpha,
                             const double *A, const int64_t lda,
                             const double *X, const int64_t incX,
                             const double beta, double *Y, const int64_t incY);

extern void (*p_cblas_strmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const float *A, std::int64_t lda,
                             float *X, std::int64_t incx);
extern void (*p_cblas_dtrmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const double *A, std::int64_t lda,
                             double *X, std::int64_t incx);

extern void (*p_cblas_strsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const float *A, std::int64_t lda,
                             float *X, std::int64_t incX);
extern void (*p_cblas_dtrsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const double *A, std::int64_t lda,
                             double *X, std::int64_t incx);

extern void (*p_cblas_ssbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, std::int64_t k, const float alpha,
                             const float *A, std::int64_t lda, const float *X,
                             std::int64_t incx, const float beta, float *Y,
                             std::int64_t incy);
extern void (*p_cblas_dsbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, std::int64_t k, const double alpha,
                             const double *A, std::int64_t lda, const double *X,
                             std::int64_t incx, const double beta, double *Y,
                             std::int64_t incy);

// Level 3

extern void (*p_cblas_sgemm)(const CBLAS_ORDER Order,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_TRANSPOSE TransB, std::int64_t M,
                             std::int64_t N, std::int64_t K, const float alpha,
                             const float *A, std::int64_t lda, const float *B,
                             std::int64_t ldb, const float beta, float *C,
                             std::int64_t ldc);
extern void (*p_cblas_dgemm)(const CBLAS_ORDER Order,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_TRANSPOSE TransB, std::int64_t M,
                             std::int64_t N, std::int64_t K, const double alpha,
                             const double *A, std::int64_t lda, const double *B,
                             std::int64_t ldb, const double beta, double *C,
                             std::int64_t ldc);

extern void (*p_cblas_ssymm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo, std::int64_t M,
                             std::int64_t N, const float alpha, const float *A,
                             std::int64_t lda, const float *B, std::int64_t ldb,
                             const float beta, float *C, std::int64_t ldc);
extern void (*p_cblas_dsymm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo, std::int64_t M,
                             std::int64_t N, const double alpha,
                             const double *A, std::int64_t lda, const double *B,
                             std::int64_t ldb, const double beta, double *C,
                             std::int64_t ldc);

extern void (*p_cblas_ssyrk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE Trans, std::int64_t N,
                             std::int64_t K, const float alpha, const float *A,
                             std::int64_t lda, const float beta, float *C,
                             std::int64_t ldc);
extern void (*p_cblas_dsyrk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE Trans, std::int64_t N,
                             std::int64_t K, const double alpha,
                             const double *A, std::int64_t lda,
                             const double beta, double *C, std::int64_t ldc);

extern void (*p_cblas_strmm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_DIAG Diag, std::int64_t M,
                             std::int64_t N, const float alpha, const float *A,
                             std::int64_t lda, float *B, std::int64_t ldb);
extern void (*p_cblas_dtrmm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_DIAG Diag, std::int64_t M,
                             std::int64_t N, const double alpha,
                             const double *A, std::int64_t lda, double *B,
                             std::int64_t ldb);

extern void (*p_cblas_strsm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_DIAG Diag, std::int64_t M,
                             std::int64_t N, const float alpha, const float *A,
                             std::int64_t lda, float *B, std::int64_t ldb);
extern void (*p_cblas_dtrsm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_DIAG Diag, std::int64_t M,
                             std::int64_t N, const double alpha,
                             const double *A, std::int64_t lda, double *B,
                             std::int64_t ldb);

// ROCBLAS Functions

extern hipError_t (*p_hipMalloc)(void **, size_t);
extern hipError_t (*p_hipFree)(void *);
extern hipError_t (*p_hipMemcpy)(void *, const void *, size_t, hipMemcpyKind);

extern rocblas_status (*p_rocblas_create_handle)(rocblas_handle *handle);
extern rocblas_status (*p_rocblas_destroy_handle)(rocblas_handle handle);

// Level 1

extern rocblas_status (*p_rocblas_saxpy)(rocblas_handle, rocblas_int,
                                         const float *, const float *,
                                         rocblas_int, float *, rocblas_int);
extern rocblas_status (*p_rocblas_daxpy)(rocblas_handle, rocblas_int,
                                         const double *, const double *,
                                         rocblas_int, double *, rocblas_int);

extern rocblas_status (*p_rocblas_scopy)(rocblas_handle, rocblas_int,
                                         const float *, rocblas_int, float *,
                                         rocblas_int);
extern rocblas_status (*p_rocblas_dcopy)(rocblas_handle, rocblas_int,
                                         const double *, rocblas_int, double *,
                                         rocblas_int);

extern rocblas_status (*p_rocblas_sdot)(rocblas_handle, rocblas_int,
                                        const float *, rocblas_int,
                                        const float *, rocblas_int, float *);
extern rocblas_status (*p_rocblas_ddot)(rocblas_handle, rocblas_int,
                                        const double *, rocblas_int,
                                        const double *, rocblas_int, double *);

extern rocblas_status (*p_rocblas_sscal)(rocblas_handle, rocblas_int,
                                         const float *, float *, rocblas_int);
extern rocblas_status (*p_rocblas_dscal)(rocblas_handle, rocblas_int,
                                         const double *, double *, rocblas_int);

extern rocblas_status (*p_rocblas_sswap)(rocblas_handle, rocblas_int, float *,
                                         rocblas_int, float *, rocblas_int);
extern rocblas_status (*p_rocblas_dswap)(rocblas_handle, rocblas_int, double *,
                                         rocblas_int, double *, rocblas_int);

// LEVEL 2

extern rocblas_status (*p_rocblas_sgemv)(rocblas_handle handle,
                                         rocblas_operation trans,
                                         std::int64_t m, std::int64_t n,
                                         const float *alpha, const float *A,
                                         std::int64_t lda, const float *x,
                                         std::int64_t incx, const float *beta,
                                         float *y, std::int64_t incy);
extern rocblas_status (*p_rocblas_dgemv)(rocblas_handle handle,
                                         rocblas_operation trans,
                                         std::int64_t m, std::int64_t n,
                                         const double *alpha, const double *A,
                                         std::int64_t lda, const double *x,
                                         std::int64_t incx, const double *beta,
                                         double *y, std::int64_t incy);

extern rocblas_status (*p_rocblas_ssymv)(rocblas_handle handle,
                                         rocblas_fill uplo, int64_t n,
                                         const float *alpha, const float *A,
                                         int64_t lda, const float *X,
                                         int64_t incx, const float *beta,
                                         float *Y, int64_t incy);
extern rocblas_status (*p_rocblas_dsymv)(rocblas_handle handle,
                                         rocblas_fill uplo, int64_t n,
                                         const double *alpha, const double *A,
                                         int64_t lda, const double *X,
                                         int64_t incx, const double *beta,
                                         double *Y, int64_t incy);

extern rocblas_status (*p_rocblas_strmv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation trans,
                                         rocblas_diagonal diag, int64_t n,
                                         const float *A, int64_t lda, float *x,
                                         int64_t incx);
extern rocblas_status (*p_rocblas_dtrmv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation trans,
                                         rocblas_diagonal diag, int64_t n,
                                         const double *A, int64_t lda,
                                         double *x, int64_t incx);

extern rocblas_status (*p_rocblas_strsv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation trans,
                                         rocblas_diagonal diag, int64_t n,
                                         const float *A, int64_t lda, float *x,
                                         int64_t incx);
extern rocblas_status (*p_rocblas_dtrsv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation trans,
                                         rocblas_diagonal diag, int64_t n,
                                         const double *A, int64_t lda,
                                         double *x, int64_t incx);

extern rocblas_status (*p_rocblas_ssbmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n, std::int64_t k,
    const float *alpha, const float *A, std::int64_t lda, const float *X,
    std::int64_t incx, const float *beta, float *Y, std::int64_t incy);
extern rocblas_status (*p_rocblas_dsbmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n, std::int64_t k,
    const double *alpha, const double *A, std::int64_t lda, const double *X,
    std::int64_t incx, const double *beta, double *Y, std::int64_t incy);

// Level 3

extern rocblas_status (*p_rocblas_sgemm)(
    rocblas_handle, rocblas_operation, rocblas_operation, rocblas_int,
    rocblas_int, rocblas_int, const float *, const float *, rocblas_int,
    const float *, rocblas_int, const float *, float *, rocblas_int);
extern rocblas_status (*p_rocblas_dgemm)(
    rocblas_handle, rocblas_operation, rocblas_operation, rocblas_int,
    rocblas_int, rocblas_int, const double *, const double *, rocblas_int,
    const double *, rocblas_int, const double *, double *, rocblas_int);

extern rocblas_status (*p_rocblas_ssymm)(rocblas_handle, rocblas_side,
                                         rocblas_fill, rocblas_int, rocblas_int,
                                         const float *, const float *,
                                         rocblas_int, const float *,
                                         rocblas_int, const float *, float *,
                                         rocblas_int);
extern rocblas_status (*p_rocblas_dsymm)(rocblas_handle, rocblas_side,
                                         rocblas_fill, rocblas_int, rocblas_int,
                                         const double *, const double *,
                                         rocblas_int, const double *,
                                         rocblas_int, const double *, double *,
                                         rocblas_int);

extern rocblas_status (*p_rocblas_ssyrk)(rocblas_handle, rocblas_fill,
                                         rocblas_operation, rocblas_int,
                                         rocblas_int, const float *,
                                         const float *, rocblas_int,
                                         const float *, float *, rocblas_int);
extern rocblas_status (*p_rocblas_dsyrk)(rocblas_handle, rocblas_fill,
                                         rocblas_operation, rocblas_int,
                                         rocblas_int, const double *,
                                         const double *, rocblas_int,
                                         const double *, double *, rocblas_int);

extern rocblas_status (*p_rocblas_strmm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_operation,
    rocblas_diagonal, rocblas_int, rocblas_int, const float *, const float *,
    rocblas_int, const float *, rocblas_int, float *, rocblas_int);
extern rocblas_status (*p_rocblas_dtrmm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_operation,
    rocblas_diagonal, rocblas_int, rocblas_int, const double *, const double *,
    rocblas_int, const double *, rocblas_int, double *, rocblas_int);

extern rocblas_status (*p_rocblas_strsm)(rocblas_handle, rocblas_side,
                                         rocblas_fill, rocblas_operation,
                                         rocblas_diagonal, rocblas_int,
                                         rocblas_int, const float *,
                                         const float *, rocblas_int, float *,
                                         rocblas_int);
extern rocblas_status (*p_rocblas_dtrsm)(rocblas_handle, rocblas_side,
                                         rocblas_fill, rocblas_operation,
                                         rocblas_diagonal, rocblas_int,
                                         rocblas_int, const double *,
                                         const double *, rocblas_int, double *,
                                         rocblas_int);

bool loadCudaAndCublas();
bool loadOpenBlas();
bool loadHipAndRocblas();

} // namespace dyn
} // namespace clap
