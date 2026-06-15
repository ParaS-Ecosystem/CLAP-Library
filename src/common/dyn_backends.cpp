// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of the CLAP library, a component of the ParaS Ecosystem.
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License (LGPL) version 3
// as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <https://www.gnu.org/licenses/>.
// -----------------------------------------------------------------------------

#include "../include/clap/dyn_backends.hpp"
#include <mutex>

namespace clap {
namespace dyn {

//=======CUBLAS==========

cudaError_t (*p_cudaMalloc)(void **ptr, size_t size) = nullptr;
cudaError_t (*p_cudaFree)(void *ptr) = nullptr;
cudaError_t (*p_cudaMemcpy)(void *dst, const void *src, size_t sizeBytes,
                            cudaMemcpyKind kind) = nullptr;
cudaError_t (*p_cudaDeviceSynchronize)(void) = nullptr;

cublasStatus_t (*p_cublasCreate_v2)(cublasHandle_t *handle) = nullptr;
cublasStatus_t (*p_cublasDestroy_v2)(cublasHandle_t handle) = nullptr;
cublasStatus_t (*p_cublasSetPointerMode_v2)(cublasHandle_t handle,
                                            cublasPointerMode_t mode) = nullptr;
cudaError_t (*p_cudaMemset)(void *, int64_t, size_t) = nullptr;

// level 1

cublasStatus_t (*p_cublasScopy_v2)(cublasHandle_t, int64_t, const float *,
                                   int64_t, float *, int64_t) = nullptr;
cublasStatus_t (*p_cublasDcopy_v2)(cublasHandle_t, int64_t, const double *,
                                   int64_t, double *, int64_t) = nullptr;

cublasStatus_t (*p_cublasSscal_v2)(cublasHandle_t, int64_t, const float *,
                                   float *, int64_t) = nullptr;
cublasStatus_t (*p_cublasDscal_v2)(cublasHandle_t, int64_t, const double *,
                                   double *, int64_t) = nullptr;

cublasStatus_t (*p_cublasSaxpy_v2)(cublasHandle_t, int64_t, const float *,
                                   const float *, int64_t, float *,
                                   int64_t) = nullptr;
cublasStatus_t (*p_cublasDaxpy_v2)(cublasHandle_t, int64_t, const double *,
                                   const double *, int64_t, double *,
                                   int64_t) = nullptr;

cublasStatus_t (*p_cublasSswap_v2)(cublasHandle_t, int64_t, float *, int64_t,
                                   float *, int64_t) = nullptr;
cublasStatus_t (*p_cublasDswap_v2)(cublasHandle_t, int64_t, double *, int64_t,
                                   double *, int64_t) = nullptr;

cublasStatus_t (*p_cublasSdot_v2)(cublasHandle_t, int64_t, const float *,
                                  int64_t, const float *, int64_t,
                                  float *) = nullptr;
cublasStatus_t (*p_cublasDdot_v2)(cublasHandle_t, int64_t, const double *,
                                  int64_t, const double *, int64_t,
                                  double *) = nullptr;

// Level 2

cublasStatus_t (*p_cublasSgemv_v2)(cublasHandle_t handle,
                                   cublasOperation_t trans, std::int64_t m,
                                   std::int64_t n, const float *alpha,
                                   const float *A, std::int64_t lda,
                                   const float *x, std::int64_t incx,
                                   const float *beta, float *y,
                                   std::int64_t incy) = nullptr;
cublasStatus_t (*p_cublasDgemv_v2)(cublasHandle_t handle,
                                   cublasOperation_t trans, std::int64_t m,
                                   std::int64_t n, const double *alpha,
                                   const double *A, std::int64_t lda,
                                   const double *x, std::int64_t incx,
                                   const double *beta, double *y,
                                   std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasSsymv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   int64_t n, const float *alpha,
                                   const float *A, int64_t lda, const float *x,
                                   int64_t incx, const float *beta, float *y,
                                   int64_t incy) = nullptr;
cublasStatus_t (*p_cublasDsymv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   int64_t n, const double *alpha,
                                   const double *A, int64_t lda,
                                   const double *x, int64_t incx,
                                   const double *beta, double *y,
                                   int64_t incy) = nullptr;

cublasStatus_t (*p_cublasStrmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const float *A, std::int64_t lda, float *X,
                                   std::int64_t incx) = nullptr;
cublasStatus_t (*p_cublasDtrmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const double *A, std::int64_t lda, double *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasStrsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const float *A, std::int64_t lda, float *X,
                                   std::int64_t incx) = nullptr;
cublasStatus_t (*p_cublasDtrsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const double *A, std::int64_t lda, double *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasSsbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   std::int64_t n, std::int64_t k,
                                   const float *alpha, const float *A,
                                   std::int64_t lda, const float *X,
                                   std::int64_t incx, const float *beta,
                                   float *Y, std::int64_t incy) = nullptr;
cublasStatus_t (*p_cublasDsbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   std::int64_t n, std::int64_t k,
                                   const double *alpha, const double *A,
                                   std::int64_t lda, const double *X,
                                   std::int64_t incx, const double *beta,
                                   double *Y, std::int64_t incy) = nullptr;

// Level 3

cublasStatus_t (*p_cublasSgemm_v2)(cublasHandle_t handle,
                                   cublasOperation_t transa,
                                   cublasOperation_t transb, int64_t m,
                                   int64_t n, int64_t k, const float *alpha,
                                   const float *A, int64_t lda, const float *B,
                                   int64_t ldb, const float *beta, float *C,
                                   int64_t ldc) = nullptr;
cublasStatus_t (*p_cublasDgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int64_t m, int64_t n, int64_t k, const double *alpha, const double *A,
    int64_t lda, const double *B, int64_t ldb, const double *beta, double *C,
    int64_t ldc) = nullptr;

cublasStatus_t (*p_cublasSsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, int64_t, int64_t,
                                   const float *, const float *, int64_t,
                                   const float *, int64_t, const float *,
                                   float *, int64_t) = nullptr;
cublasStatus_t (*p_cublasDsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, int64_t, int64_t,
                                   const double *, const double *, int64_t,
                                   const double *, int64_t, const double *,
                                   double *, int64_t) = nullptr;

cublasStatus_t (*p_cublasSsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                   cublasOperation_t, int64_t, int64_t,
                                   const float *, const float *, int64_t,
                                   const float *, float *, int64_t) = nullptr;
cublasStatus_t (*p_cublasDsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                   cublasOperation_t, int64_t, int64_t,
                                   const double *, const double *, int64_t,
                                   const double *, double *, int64_t) = nullptr;

cublasStatus_t (*p_cublasStrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int64_t, int64_t,
                                   const float *, const float *, int64_t,
                                   float *, int64_t, float *,
                                   int64_t) = nullptr;
cublasStatus_t (*p_cublasDtrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int64_t, int64_t,
                                   const double *, const double *, int64_t,
                                   double *, int64_t, double *,
                                   int64_t) = nullptr;

cublasStatus_t (*p_cublasStrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int64_t, int64_t,
                                   const float *, const float *, int64_t,
                                   float *, int64_t) = nullptr;
cublasStatus_t (*p_cublasDtrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int64_t, int64_t,
                                   const double *, const double *, int64_t,
                                   double *, int64_t) = nullptr;

//=======OpenBLAS==========

// Level 1

void (*p_cblas_scopy)(int64_t, const float *, int64_t, float *,
                      int64_t) = nullptr;
void (*p_cblas_dcopy)(int64_t, const double *, int64_t, double *,
                      int64_t) = nullptr;

void (*p_cblas_sscal)(int64_t, float, float *, int64_t) = nullptr;
void (*p_cblas_dscal)(int64_t, double, double *, int64_t) = nullptr;

void (*p_cblas_saxpy)(int64_t, float, const float *, int64_t, float *,
                      int64_t) = nullptr;
void (*p_cblas_daxpy)(int64_t, double, const double *, int64_t, double *,
                      int64_t) = nullptr;

void (*p_cblas_sswap)(int64_t, float *, int64_t, float *, int64_t) = nullptr;
void (*p_cblas_dswap)(int64_t, double *, int64_t, double *, int64_t) = nullptr;

float (*p_cblas_sdot)(int64_t, const float *, int64_t, const float *,
                      int64_t) = nullptr;
double (*p_cblas_ddot)(int64_t, const double *, int64_t, const double *,
                       int64_t) = nullptr;

// Level 2

void (*p_cblas_sgemv)(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE Trans,
                      std::int64_t m, std::int64_t n, const float alpha,
                      const float *A, std::int64_t lda, const float *x,
                      std::int64_t incx, const float beta, float *y,
                      std::int64_t incy) = nullptr;
void (*p_cblas_dgemv)(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE Trans,
                      std::int64_t m, std::int64_t n, const double alpha,
                      const double *A, std::int64_t lda, const double *x,
                      std::int64_t incx, const double beta, double *y,
                      std::int64_t incy) = nullptr;

void (*p_cblas_ssymv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      const int64_t N, const float alpha, const float *A,
                      const int64_t lda, const float *X, const int64_t incX,
                      const float beta, float *Y, const int64_t incY) = nullptr;
void (*p_cblas_dsymv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      const int64_t N, const double alpha, const double *A,
                      const int64_t lda, const double *X, const int64_t incX,
                      const double beta, double *Y,
                      const int64_t incY) = nullptr;

void (*p_cblas_strmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const float *A, std::int64_t lda, float *X,
                      std::int64_t incx) = nullptr;
void (*p_cblas_dtrmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const double *A, std::int64_t lda, double *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_strsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const float *A, std::int64_t lda, float *X,
                      std::int64_t incX) = nullptr;
void (*p_cblas_dtrsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const double *A, std::int64_t lda, double *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_ssbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, std::int64_t k, const float alpha,
                      const float *A, std::int64_t lda, const float *X,
                      std::int64_t incx, const float beta, float *Y,
                      std::int64_t incy) = nullptr;
void (*p_cblas_dsbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, std::int64_t k, const double alpha,
                      const double *A, std::int64_t lda, const double *X,
                      std::int64_t incx, const double beta, double *Y,
                      std::int64_t incy) = nullptr;

// level 3

void (*p_cblas_sgemm)(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, std::int64_t M,
                      std::int64_t N, std::int64_t K, const float alpha,
                      const float *A, std::int64_t lda, const float *B,
                      std::int64_t ldb, const float beta, float *C,
                      std::int64_t ldc) = nullptr;
void (*p_cblas_dgemm)(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, std::int64_t M,
                      std::int64_t N, std::int64_t K, const double alpha,
                      const double *A, std::int64_t lda, const double *B,
                      std::int64_t ldb, const double beta, double *C,
                      std::int64_t ldc) = nullptr;

void (*p_cblas_ssymm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, std::int64_t M, std::int64_t N,
                      const float alpha, const float *A, std::int64_t lda,
                      const float *B, std::int64_t ldb, const float beta,
                      float *C, std::int64_t ldc) = nullptr;
void (*p_cblas_dsymm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, std::int64_t M, std::int64_t N,
                      const double alpha, const double *A, std::int64_t lda,
                      const double *B, std::int64_t ldb, const double beta,
                      double *C, std::int64_t ldc) = nullptr;

void (*p_cblas_ssyrk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      const CBLAS_TRANSPOSE Trans, std::int64_t N,
                      std::int64_t K, const float alpha, const float *A,
                      std::int64_t lda, const float beta, float *C,
                      std::int64_t ldc) = nullptr;
void (*p_cblas_dsyrk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      const CBLAS_TRANSPOSE Trans, std::int64_t N,
                      std::int64_t K, const double alpha, const double *A,
                      std::int64_t lda, const double beta, double *C,
                      std::int64_t ldc) = nullptr;

void (*p_cblas_strmm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_DIAG Diag, std::int64_t M, std::int64_t N,
                      const float alpha, const float *A, std::int64_t lda,
                      float *B, std::int64_t ldb) = nullptr;
void (*p_cblas_dtrmm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_DIAG Diag, std::int64_t M, std::int64_t N,
                      const double alpha, const double *A, std::int64_t lda,
                      double *B, std::int64_t ldb) = nullptr;

void (*p_cblas_strsm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_DIAG Diag, std::int64_t M, std::int64_t N,
                      const float alpha, const float *A, std::int64_t lda,
                      float *B, std::int64_t ldb) = nullptr;
void (*p_cblas_dtrsm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_DIAG Diag, std::int64_t M, std::int64_t N,
                      const double alpha, const double *A, std::int64_t lda,
                      double *B, std::int64_t ldb) = nullptr;

//=======ROCBLAS==========

hipError_t (*p_hipMalloc)(void **, size_t) = nullptr;
hipError_t (*p_hipFree)(void *) = nullptr;
hipError_t (*p_hipMemcpy)(void *, const void *, size_t,
                          hipMemcpyKind) = nullptr;

hipError_t (*p_hipDeviceSynchronize)(void) = nullptr;

// RocBlas

rocblas_status (*p_rocblas_create_handle)(rocblas_handle *handle) = nullptr;
rocblas_status (*p_rocblas_destroy_handle)(rocblas_handle handle) = nullptr;

// Level 1

rocblas_status (*p_rocblas_saxpy)(rocblas_handle, rocblas_int, const float *,
                                  const float *, rocblas_int, float *,
                                  rocblas_int) = nullptr;
rocblas_status (*p_rocblas_daxpy)(rocblas_handle, rocblas_int, const double *,
                                  const double *, rocblas_int, double *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_scopy)(rocblas_handle, rocblas_int, const float *,
                                  rocblas_int, float *, rocblas_int) = nullptr;
rocblas_status (*p_rocblas_dcopy)(rocblas_handle, rocblas_int, const double *,
                                  rocblas_int, double *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_sdot)(rocblas_handle, rocblas_int, const float *,
                                 rocblas_int, const float *, rocblas_int,
                                 float *) = nullptr;
rocblas_status (*p_rocblas_ddot)(rocblas_handle, rocblas_int, const double *,
                                 rocblas_int, const double *, rocblas_int,
                                 double *) = nullptr;

rocblas_status (*p_rocblas_sscal)(rocblas_handle, rocblas_int, const float *,
                                  float *, rocblas_int) = nullptr;
rocblas_status (*p_rocblas_dscal)(rocblas_handle, rocblas_int, const double *,
                                  double *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_sswap)(rocblas_handle, rocblas_int, float *,
                                  rocblas_int, float *, rocblas_int) = nullptr;
rocblas_status (*p_rocblas_dswap)(rocblas_handle, rocblas_int, double *,
                                  rocblas_int, double *, rocblas_int) = nullptr;

// Level 2

rocblas_status (*p_rocblas_sgemv)(rocblas_handle handle,
                                  rocblas_operation trans, std::int64_t m,
                                  std::int64_t n, const float *alpha,
                                  const float *A, std::int64_t lda,
                                  const float *x, std::int64_t incx,
                                  const float *beta, float *y,
                                  std::int64_t incy) = nullptr;
rocblas_status (*p_rocblas_dgemv)(rocblas_handle handle,
                                  rocblas_operation trans, std::int64_t m,
                                  std::int64_t n, const double *alpha,
                                  const double *A, std::int64_t lda,
                                  const double *x, std::int64_t incx,
                                  const double *beta, double *y,
                                  std::int64_t incy) = nullptr;

rocblas_status (*p_rocblas_ssymv)(rocblas_handle handle, rocblas_fill uplo,
                                  int64_t n, const float *alpha, const float *A,
                                  int64_t lda, const float *X, int64_t incx,
                                  const float *beta, float *Y,
                                  int64_t incy) = nullptr;
rocblas_status (*p_rocblas_dsymv)(rocblas_handle handle, rocblas_fill uplo,
                                  int64_t n, const double *alpha,
                                  const double *A, int64_t lda, const double *X,
                                  int64_t incx, const double *beta, double *Y,
                                  int64_t incy) = nullptr;

rocblas_status (*p_rocblas_strmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, int64_t n,
                                  const float *A, int64_t lda, float *x,
                                  int64_t incx) = nullptr;
rocblas_status (*p_rocblas_dtrmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, int64_t n,
                                  const double *A, int64_t lda, double *x,
                                  int64_t incx) = nullptr;

rocblas_status (*p_rocblas_strsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, int64_t n,
                                  const float *A, int64_t lda, float *x,
                                  int64_t incx) = nullptr;
rocblas_status (*p_rocblas_dtrsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, int64_t n,
                                  const double *A, int64_t lda, double *x,
                                  int64_t incx) = nullptr;

rocblas_status (*p_rocblas_ssbmv)(rocblas_handle handle, rocblas_fill uplo,
                                  std::int64_t n, std::int64_t k,
                                  const float *alpha, const float *A,
                                  std::int64_t lda, const float *X,
                                  std::int64_t incx, const float *beta,
                                  float *Y, std::int64_t incy) = nullptr;
rocblas_status (*p_rocblas_dsbmv)(rocblas_handle handle, rocblas_fill uplo,
                                  std::int64_t n, std::int64_t k,
                                  const double *alpha, const double *A,
                                  std::int64_t lda, const double *X,
                                  std::int64_t incx, const double *beta,
                                  double *Y, std::int64_t incy) = nullptr;

// Level 3

rocblas_status (*p_rocblas_sgemm)(rocblas_handle, rocblas_operation,
                                  rocblas_operation, rocblas_int, rocblas_int,
                                  rocblas_int, const float *, const float *,
                                  rocblas_int, const float *, rocblas_int,
                                  const float *, float *,
                                  rocblas_int) = nullptr;
rocblas_status (*p_rocblas_dgemm)(rocblas_handle, rocblas_operation,
                                  rocblas_operation, rocblas_int, rocblas_int,
                                  rocblas_int, const double *, const double *,
                                  rocblas_int, const double *, rocblas_int,
                                  const double *, double *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_ssymm)(rocblas_handle, rocblas_side, rocblas_fill,
                                  rocblas_int, rocblas_int, const float *,
                                  const float *, rocblas_int, const float *,
                                  rocblas_int, const float *, float *,
                                  rocblas_int) = nullptr;
rocblas_status (*p_rocblas_dsymm)(rocblas_handle, rocblas_side, rocblas_fill,
                                  rocblas_int, rocblas_int, const double *,
                                  const double *, rocblas_int, const double *,
                                  rocblas_int, const double *, double *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_ssyrk)(rocblas_handle, rocblas_fill,
                                  rocblas_operation, rocblas_int, rocblas_int,
                                  const float *, const float *, rocblas_int,
                                  const float *, float *,
                                  rocblas_int) = nullptr;
rocblas_status (*p_rocblas_dsyrk)(rocblas_handle, rocblas_fill,
                                  rocblas_operation, rocblas_int, rocblas_int,
                                  const double *, const double *, rocblas_int,
                                  const double *, double *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_strmm)(rocblas_handle, rocblas_side, rocblas_fill,
                                  rocblas_operation, rocblas_diagonal,
                                  rocblas_int, rocblas_int, const float *,
                                  const float *, rocblas_int, const float *,
                                  rocblas_int, float *, rocblas_int) = nullptr;
rocblas_status (*p_rocblas_dtrmm)(rocblas_handle, rocblas_side, rocblas_fill,
                                  rocblas_operation, rocblas_diagonal,
                                  rocblas_int, rocblas_int, const double *,
                                  const double *, rocblas_int, const double *,
                                  rocblas_int, double *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_strsm)(rocblas_handle, rocblas_side, rocblas_fill,
                                  rocblas_operation, rocblas_diagonal,
                                  rocblas_int, rocblas_int, const float *,
                                  const float *, rocblas_int, float *,
                                  rocblas_int) = nullptr;
rocblas_status (*p_rocblas_dtrsm)(rocblas_handle, rocblas_side, rocblas_fill,
                                  rocblas_operation, rocblas_diagonal,
                                  rocblas_int, rocblas_int, const double *,
                                  const double *, rocblas_int, double *,
                                  rocblas_int) = nullptr;

static void *cuda_lib = nullptr;
static void *cublas_lib = nullptr;
static void *openblas_lib = nullptr;
static void *rocblas_lib = nullptr;
static void *hip_lib = nullptr;

static bool cuda_loaded = false;
static bool openblas_loaded = false;
static bool hip_loaded = false;

static std::mutex cuda_mutex;
static std::mutex openblas_mutex;
static std::mutex hip_mutex;

bool loadCudaAndCublas() {
  std::lock_guard<std::mutex> lock(cuda_mutex);
  if (cuda_loaded)
    return true;

  cuda_lib = dlopen("libcudart.so", RTLD_LAZY | RTLD_LOCAL);
  if (!cuda_lib)
    cuda_lib = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_LOCAL);
  if (!cuda_lib)
    cuda_lib = dlopen("libcudart.so.11.0", RTLD_LAZY | RTLD_LOCAL);
  if (!cuda_lib) {
    const char *err = dlerror();
    std::cerr << "CLAP_BLAS: Failed to load libcudart.so. Error: "
              << (err ? err : "Unknown") << "\n";
    return false;
  }

  cublas_lib = dlopen("libcublas.so", RTLD_LAZY | RTLD_LOCAL);
  if (!cublas_lib)
    cublas_lib = dlopen("libcublas.so.12", RTLD_LAZY | RTLD_LOCAL);
  if (!cublas_lib)
    cublas_lib = dlopen("libcublas.so.11", RTLD_LAZY | RTLD_LOCAL);
  if (!cublas_lib) {
    const char *err = dlerror();
    std::cerr << "CLAP_BLAS: Failed to load libcublas.so. Error: "
              << (err ? err : "Unknown") << "\n";
    dlclose(cuda_lib);
    return false;
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

#define LOAD_CUDA_SYM(lib, name)                                               \
  p_##name = reinterpret_cast<decltype(p_##name)>(dlsym(lib, #name));          \
  if (!p_##name) {                                                             \
    std::cerr << "CLAP_BLAS: Symbol " << #name << " not found.\n";             \
    dlclose(cuda_lib);                                                         \
    dlclose(cublas_lib);                                                       \
    return false;                                                              \
  }

  LOAD_CUDA_SYM(cuda_lib, cudaMalloc);
  LOAD_CUDA_SYM(cuda_lib, cudaFree);
  LOAD_CUDA_SYM(cuda_lib, cudaMemcpy);
  LOAD_CUDA_SYM(cuda_lib, cudaDeviceSynchronize);
  LOAD_CUDA_SYM(cuda_lib, cudaMemset);

  LOAD_CUDA_SYM(cublas_lib, cublasSetPointerMode_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCreate_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDestroy_v2);

  // ===== cuBLAS =====

  // Level 1

  // COPY

  LOAD_CUDA_SYM(cublas_lib, cublasScopy_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDcopy_v2);

  // SCAL

  LOAD_CUDA_SYM(cublas_lib, cublasSscal_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDscal_v2);

  // AXPY

  LOAD_CUDA_SYM(cublas_lib, cublasSaxpy_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDaxpy_v2);

  // SWAP

  LOAD_CUDA_SYM(cublas_lib, cublasSswap_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDswap_v2);

  // DOT

  LOAD_CUDA_SYM(cublas_lib, cublasSdot_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDdot_v2);

  // Level 2

  // GEMV

  LOAD_CUDA_SYM(cublas_lib, cublasSgemv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDgemv_v2);

  // SYMV

  LOAD_CUDA_SYM(cublas_lib, cublasSsymv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDsymv_v2);

  // TRMV

  LOAD_CUDA_SYM(cublas_lib, cublasStrmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtrmv_v2);

  // TRSV

  LOAD_CUDA_SYM(cublas_lib, cublasStrsv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtrsv_v2);

  // SBMV

  LOAD_CUDA_SYM(cublas_lib, cublasSsbmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDsbmv_v2);

  // level 3

  // GEMM

  LOAD_CUDA_SYM(cublas_lib, cublasSgemm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDgemm_v2);

  // SYMM

  LOAD_CUDA_SYM(cublas_lib, cublasSsymm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDsymm_v2);

  // SYRK

  LOAD_CUDA_SYM(cublas_lib, cublasSsyrk_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDsyrk_v2);

  // TRMM

  LOAD_CUDA_SYM(cublas_lib, cublasStrmm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtrmm_v2);

  // TRSM

  LOAD_CUDA_SYM(cublas_lib, cublasStrsm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtrsm_v2);

#pragma GCC diagnostic pop

  cuda_loaded = true;
  return true;
}

bool loadOpenBlas() {
  std::lock_guard<std::mutex> lock(openblas_mutex);
  if (openblas_loaded)
    return true;

  openblas_lib = dlopen("libopenblas.so", RTLD_LAZY | RTLD_LOCAL);
  if (!openblas_lib)
    openblas_lib = dlopen("libopenblas.so.0", RTLD_LAZY | RTLD_LOCAL);
  if (!openblas_lib) {
    const char *err = dlerror();
    std::cerr << "CLAP_BLAS: Failed to load libopenblas.so. Error: "
              << (err ? err : "Unknown") << "\n";
    return false;
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

#define LOAD_OBLAS_SYM(lib, name)                                              \
  p_##name = reinterpret_cast<decltype(p_##name)>(dlsym(lib, #name));          \
  if (!p_##name) {                                                             \
    std::cerr << "CLAP_BLAS: Symbol " << #name << " not found.\n";             \
    dlclose(openblas_lib);                                                     \
    return false;                                                              \
  }

  // ===== OpenBLAS =====

  // Level 1

  // COPY

  LOAD_OBLAS_SYM(openblas_lib, cblas_scopy);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dcopy);

  // SCAL

  LOAD_OBLAS_SYM(openblas_lib, cblas_sscal);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dscal);

  // AXPY

  LOAD_OBLAS_SYM(openblas_lib, cblas_saxpy);
  LOAD_OBLAS_SYM(openblas_lib, cblas_daxpy);

  // SWAP

  LOAD_OBLAS_SYM(openblas_lib, cblas_sswap);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dswap);

  // DOT

  LOAD_OBLAS_SYM(openblas_lib, cblas_sdot);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ddot);

  // Level 2

  // GEMV

  LOAD_OBLAS_SYM(openblas_lib, cblas_sgemv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dgemv);

  // SYMV

  LOAD_OBLAS_SYM(openblas_lib, cblas_ssymv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dsymv);

  // TRMV

  LOAD_OBLAS_SYM(openblas_lib, cblas_strmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtrmv);

  // TRSV

  LOAD_OBLAS_SYM(openblas_lib, cblas_strsv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtrsv);

  // SBMV

  LOAD_OBLAS_SYM(openblas_lib, cblas_ssbmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dsbmv);

  // Level 3

  // GEMM

  LOAD_OBLAS_SYM(openblas_lib, cblas_sgemm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dgemm);

  // SYMM

  LOAD_OBLAS_SYM(openblas_lib, cblas_ssymm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dsymm);

  // SYRK

  LOAD_OBLAS_SYM(openblas_lib, cblas_ssyrk);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dsyrk);

  // TRMM

  LOAD_OBLAS_SYM(openblas_lib, cblas_strmm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtrmm);

  // TRSM

  LOAD_OBLAS_SYM(openblas_lib, cblas_strsm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtrsm);

#pragma GCC diagnostic pop

  openblas_loaded = true;
  return true;
}
bool loadHipAndRocblas() {
  std::lock_guard<std::mutex> lock(hip_mutex);

  if (hip_loaded)
    return true;

  hip_lib = dlopen("libamdhip64.so", RTLD_LAZY | RTLD_LOCAL);
  if (!hip_lib)
    hip_lib = dlopen("libamdhip64.so.6", RTLD_LAZY | RTLD_LOCAL);

  if (!hip_lib) {
    const char *err = dlerror();

    std::cerr << "CLAP_BLAS: Failed to load libamdhip64.so. Error: "
              << (err ? err : "Unknown") << "\n";

    return false;
  }

  rocblas_lib = dlopen("librocblas.so", RTLD_LAZY | RTLD_LOCAL);

  if (!rocblas_lib)
    rocblas_lib = dlopen("librocblas.so.4", RTLD_LAZY | RTLD_LOCAL);

  if (!rocblas_lib) {
    const char *err = dlerror();

    std::cerr << "CLAP_BLAS: Failed to load librocblas.so. Error: "
              << (err ? err : "Unknown") << "\n";

    dlclose(hip_lib);

    return false;
  }

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"

#define LOAD_HIP_SYM(lib, name)                                                \
  p_##name = reinterpret_cast<decltype(p_##name)>(dlsym(lib, #name));          \
  if (!p_##name) {                                                             \
    std::cerr << "CLAP_BLAS: Symbol " << #name << " not found.\n";             \
    dlclose(hip_lib);                                                          \
    dlclose(rocblas_lib);                                                      \
    return false;                                                              \
  }

  // HIP Runtime

  LOAD_HIP_SYM(hip_lib, hipMalloc);
  LOAD_HIP_SYM(hip_lib, hipFree);
  LOAD_HIP_SYM(hip_lib, hipMemcpy);

  // rocBLAS Handle

  LOAD_HIP_SYM(rocblas_lib, rocblas_create_handle);
  LOAD_HIP_SYM(rocblas_lib, rocblas_destroy_handle);

  // ====ROCBLAS====

  // Level 1

  // AXPY

  LOAD_HIP_SYM(rocblas_lib, rocblas_saxpy);
  LOAD_HIP_SYM(rocblas_lib, rocblas_daxpy);

  // COPY

  LOAD_HIP_SYM(rocblas_lib, rocblas_scopy);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dcopy);

  // DOT

  LOAD_HIP_SYM(rocblas_lib, rocblas_sdot);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ddot);

  // SCAL

  LOAD_HIP_SYM(rocblas_lib, rocblas_sscal);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dscal);

  // SWAP

  LOAD_HIP_SYM(rocblas_lib, rocblas_sswap);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dswap);

  // level 2

  // GEMV

  LOAD_HIP_SYM(rocblas_lib, rocblas_sgemv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dgemv);

  // SYMV

  LOAD_HIP_SYM(rocblas_lib, rocblas_ssymv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsymv);

  // TRMV

  LOAD_HIP_SYM(rocblas_lib, rocblas_strmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtrmv);

  // TRSV

  LOAD_HIP_SYM(rocblas_lib, rocblas_strsv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtrsv);

  // SBMV

  LOAD_HIP_SYM(rocblas_lib, rocblas_ssbmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsbmv);

  // level 3

  // GEMM

  LOAD_HIP_SYM(rocblas_lib, rocblas_sgemm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dgemm);

  // SYMM

  LOAD_HIP_SYM(rocblas_lib, rocblas_ssymm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsymm);

  // SYRK

  LOAD_HIP_SYM(rocblas_lib, rocblas_ssyrk);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsyrk);

  // TRMM

  LOAD_HIP_SYM(rocblas_lib, rocblas_strmm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtrmm);

  // TRSM

  LOAD_HIP_SYM(rocblas_lib, rocblas_strsm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtrsm);

#pragma GCC diagnostic pop

  hip_loaded = true;

  return true;
}

} // namespace dyn
} // namespace clap
