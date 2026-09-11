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
cudaError_t (*p_cudaMemset)(void *, int, size_t) = nullptr;

//LEVEL 1 ROUTINES

cublasStatus_t (*p_cublasScopy_v2)(cublasHandle_t, int, const float *, int,
                                   float *, int) = nullptr;

cublasStatus_t (*p_cublasDcopy_v2)(cublasHandle_t, int, const double *, int,
                                   double *, int) = nullptr;

cublasStatus_t (*p_cublasSscal_v2)(cublasHandle_t, int, const float *, float *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasDscal_v2)(cublasHandle_t, int, const double *,
                                   double *, int) = nullptr;

cublasStatus_t (*p_cublasSaxpy_v2)(cublasHandle_t, int, const float *,
                                   const float *, int, float *, int) = nullptr;

cublasStatus_t (*p_cublasDaxpy_v2)(cublasHandle_t, int, const double *,
                                   const double *, int, double *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasIsamax_v2)(cublasHandle_t, int, const float *, int,
                                    int *) = nullptr;

cublasStatus_t (*p_cublasIdamax_v2)(cublasHandle_t, int, const double *, int,
                                    int *) = nullptr;

cublasStatus_t (*p_cublasSswap_v2)(cublasHandle_t, int, float *, int, float *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasDswap_v2)(cublasHandle_t, int, double *, int, double *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasSdot_v2)(cublasHandle_t, int, const float *, int,
                                  const float *, int, float *) = nullptr;

cublasStatus_t (*p_cublasDdot_v2)(cublasHandle_t, int, const double *, int,
                                  const double *, int, double *) = nullptr;

cublasStatus_t (*p_cublasSnrm2_v2)(cublasHandle_t, int, const float *, int,
                                   float *) = nullptr;

cublasStatus_t (*p_cublasDnrm2_v2)(cublasHandle_t, int, const double *, int,
                                   double *) = nullptr;

cublasStatus_t (*p_cublasSasum_v2)(cublasHandle_t, int, const float *, int,
                                   float *) = nullptr;

cublasStatus_t (*p_cublasDasum_v2)(cublasHandle_t, int, const double *, int,
                                   double *) = nullptr;

cublasStatus_t (*p_cublasSrot_v2)(cublasHandle_t, int, float *, int, float *,
                                  int, const float *, const float *) = nullptr;

cublasStatus_t (*p_cublasDrot_v2)(cublasHandle_t, int, double *, int, double *,
                                  int, const double *,
                                  const double *) = nullptr;

cublasStatus_t (*p_cublasSrotg_v2)(cublasHandle_t, float *, float *, float *,
                                   float *) = nullptr;

cublasStatus_t (*p_cublasDrotg_v2)(cublasHandle_t, double *, double *, double *,
                                   double *) = nullptr;

cublasStatus_t (*p_cublasSrotm_v2)(cublasHandle_t, int, float *, int, float *,
                                   int, const float *) = nullptr;

cublasStatus_t (*p_cublasDrotm_v2)(cublasHandle_t, int, double *, int, double *,
                                   int, const double *) = nullptr;

cublasStatus_t (*p_cublasSrotmg_v2)(cublasHandle_t, float *, float *, float *,
                                    const float *, float *) = nullptr;

cublasStatus_t (*p_cublasDrotmg_v2)(cublasHandle_t, double *, double *,
                                    double *, const double *,
                                    double *) = nullptr;

cublasStatus_t (*p_cublasIcamax_v2)(cublasHandle_t, int, const cuComplex *, int,
                                    int *) = nullptr;

cublasStatus_t (*p_cublasIzamax_v2)(cublasHandle_t, int,
                                    const cuDoubleComplex *, int,
                                    int *) = nullptr;

cublasStatus_t (*p_cublasScasum_v2)(cublasHandle_t, int, const cuComplex *, int,
                                    float *) = nullptr;

cublasStatus_t (*p_cublasDzasum_v2)(cublasHandle_t, int,
                                    const cuDoubleComplex *, int,
                                    double *) = nullptr;

cublasStatus_t (*p_cublasCaxpy_v2)(cublasHandle_t, int, const cuComplex *,
                                   const cuComplex *, int, cuComplex *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasZaxpy_v2)(cublasHandle_t, int, const cuDoubleComplex *,
                                   const cuDoubleComplex *, int,
                                   cuDoubleComplex *, int) = nullptr;

cublasStatus_t (*p_cublasCcopy_v2)(cublasHandle_t, int, const cuComplex *, int,
                                   cuComplex *, int) = nullptr;

cublasStatus_t (*p_cublasZcopy_v2)(cublasHandle_t, int, const cuDoubleComplex *,
                                   int, cuDoubleComplex *, int) = nullptr;

cublasStatus_t (*p_cublasCdotu_v2)(cublasHandle_t, int, const cuComplex *, int,
                                   const cuComplex *, int,
                                   cuComplex *) = nullptr;

cublasStatus_t (*p_cublasCdotc_v2)(cublasHandle_t, int, const cuComplex *, int,
                                   const cuComplex *, int,
                                   cuComplex *) = nullptr;

cublasStatus_t (*p_cublasZdotu_v2)(cublasHandle_t, int, const cuDoubleComplex *,
                                   int, const cuDoubleComplex *, int,
                                   cuDoubleComplex *) = nullptr;

cublasStatus_t (*p_cublasZdotc_v2)(cublasHandle_t, int, const cuDoubleComplex *,
                                   int, const cuDoubleComplex *, int,
                                   cuDoubleComplex *) = nullptr;

cublasStatus_t (*p_cublasScnrm2_v2)(cublasHandle_t, int, const cuComplex *, int,
                                    float *) = nullptr;

cublasStatus_t (*p_cublasDznrm2_v2)(cublasHandle_t, int,
                                    const cuDoubleComplex *, int,
                                    double *) = nullptr;

cublasStatus_t (*p_cublasCscal_v2)(cublasHandle_t, int, const cuComplex *,
                                   cuComplex *, int) = nullptr;

cublasStatus_t (*p_cublasZscal_v2)(cublasHandle_t, int, const cuDoubleComplex *,
                                   cuDoubleComplex *, int) = nullptr;

cublasStatus_t (*p_cublasCsscal_v2)(cublasHandle_t, int, const float *,
                                    cuComplex *, int) = nullptr;

cublasStatus_t (*p_cublasZdscal_v2)(cublasHandle_t, int, const double *,
                                    cuDoubleComplex *, int) = nullptr;

cublasStatus_t (*p_cublasCswap_v2)(cublasHandle_t, int, cuComplex *, int,
                                   cuComplex *, int) = nullptr;

cublasStatus_t (*p_cublasZswap_v2)(cublasHandle_t, int, cuDoubleComplex *, int,
                                   cuDoubleComplex *, int) = nullptr;

//LEVEL 2 ROUTINES

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

cublasStatus_t (*p_cublasCgemv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, const std::complex<float> *alpha,
    const std::complex<float> *A, std::int64_t lda,
    const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y,
    std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasZgemv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, const std::complex<double> *alpha,
    const std::complex<double> *A, std::int64_t lda,
    const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasSsymv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   int n, const float *alpha, const float *A,
                                   int lda, const float *x, int incx,
                                   const float *beta, float *y,
                                   int incy) = nullptr;
cublasStatus_t (*p_cublasDsymv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   int n, const double *alpha, const double *A,
                                   int lda, const double *x, int incx,
                                   const double *beta, double *y,
                                   int incy) = nullptr;

cublasStatus_t (*p_cublasChemv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *A,
    std::int64_t lda, const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y,
    std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasZhemv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy) = nullptr;

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

cublasStatus_t (*p_cublasCtrmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const std::complex<float> *A,
                                   std::int64_t lda, std::complex<float> *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasZtrmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const std::complex<double> *A,
                                   std::int64_t lda, std::complex<double> *X,
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

cublasStatus_t (*p_cublasCtrsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const std::complex<float> *A,
                                   std::int64_t lda, std::complex<float> *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasZtrsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const std::complex<double> *A,
                                   std::int64_t lda, std::complex<double> *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasCgerc_v2)(
    cublasHandle_t handle, std::int64_t m, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasZgerc_v2)(
    cublasHandle_t handle, std::int64_t m, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasSger_v2)(cublasHandle_t handle, std::int64_t m,
                                  std::int64_t n, const float *alpha,
                                  const float *X, std::int64_t incx,
                                  const float *Y, std::int64_t incy, float *A,
                                  std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasDger_v2)(cublasHandle_t handle, std::int64_t m,
                                  std::int64_t n, const double *alpha,
                                  const double *X, std::int64_t incx,
                                  const double *Y, std::int64_t incy, double *A,
                                  std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasCgeru_v2)(
    cublasHandle_t handle, std::int64_t m, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasZgeru_v2)(
    cublasHandle_t handle, std::int64_t m, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasSsyr_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                  std::int64_t n, const float *alpha,
                                  const float *X, std::int64_t incx, float *A,
                                  std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasDsyr_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                  std::int64_t n, const double *alpha,
                                  const double *X, std::int64_t incx, double *A,
                                  std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasSsyr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   std::int64_t n, const float *alpha,
                                   const float *X, std::int64_t incx,
                                   const float *Y, std::int64_t incy, float *A,
                                   std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasDsyr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   std::int64_t n, const double *alpha,
                                   const double *X, std::int64_t incx,
                                   const double *Y, std::int64_t incy,
                                   double *A, std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasCher_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                  std::int64_t n, const float *alpha,
                                  const std::complex<float> *X,
                                  std::int64_t incx, std::complex<float> *A,
                                  std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasZher_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                  std::int64_t n, const double *alpha,
                                  const std::complex<double> *X,
                                  std::int64_t incx, std::complex<double> *A,
                                  std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasCher2_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasZher2_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda) = nullptr;

cublasStatus_t (*p_cublasChpr_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                  std::int64_t n, const float *alpha,
                                  const std::complex<float> *X,
                                  std::int64_t incx,
                                  std::complex<float> *AP) = nullptr;

cublasStatus_t (*p_cublasZhpr_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                  std::int64_t n, const double *alpha,
                                  const std::complex<double> *X,
                                  std::int64_t incx,
                                  std::complex<double> *AP) = nullptr;

cublasStatus_t (*p_cublasChpr2_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *AP) = nullptr;

cublasStatus_t (*p_cublasZhpr2_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *AP) = nullptr;

cublasStatus_t (*p_cublasSgbmv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU, const float *alpha,
    const float *A, std::int64_t lda, const float *X, std::int64_t incx,
    const float *beta, float *Y, std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasDgbmv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU, const double *alpha,
    const double *A, std::int64_t lda, const double *X, std::int64_t incx,
    const double *beta, double *Y, std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasCgbmv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, std::int64_t Kl, std::int64_t Ku,
    const std::complex<float> *alpha, const std::complex<float> *A,
    std::int64_t lda, const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y,
    std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasZgbmv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, std::int64_t Kl, std::int64_t Ku,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasChbmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    std::int64_t k, const std::complex<float> *alpha,
    const std::complex<float> *A, std::int64_t lda,
    const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y,
    std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasZhbmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n, int64_t k,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy) = nullptr;

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

cublasStatus_t (*p_cublasStbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   std::int64_t k, const float *A,
                                   std::int64_t lda, float *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasDtbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   std::int64_t k, const double *A,
                                   std::int64_t lda, double *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasCtbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   std::int64_t k, const std::complex<float> *A,
                                   std::int64_t lda, std::complex<float> *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasZtbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   std::int64_t k,
                                   const std::complex<double> *A,
                                   std::int64_t lda, std::complex<double> *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasStbsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   std::int64_t k, const float *A,
                                   std::int64_t lda, float *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasDtbsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   std::int64_t k, const double *A,
                                   std::int64_t lda, double *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasCtbsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   std::int64_t k, const std::complex<float> *A,
                                   std::int64_t lda, std::complex<float> *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasZtbsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   std::int64_t k,
                                   const std::complex<double> *A,
                                   std::int64_t lda, std::complex<double> *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasChpmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *Ap,
    const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y,
    std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasZhpmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *Ap,
    const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasSspmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   std::int64_t n, const float *alpha,
                                   const float *Ap, const float *x,
                                   std::int64_t incX, const float *beta,
                                   float *Y, std::int64_t incY) = nullptr;

cublasStatus_t (*p_cublasDspmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   std::int64_t n, const double *alpha,
                                   const double *Ap, const double *x,
                                   std::int64_t incX, const double *beta,
                                   double *y, std::int64_t incY) = nullptr;

cublasStatus_t (*p_cublasStpmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const float *Ap, float *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasDtpmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const double *Ap, double *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasCtpmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const std::complex<float> *Ap,
                                   std::complex<float> *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasZtpmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const std::complex<double> *Ap,
                                   std::complex<double> *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasStpsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const float *Ap, float *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasDtpsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const double *Ap, double *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasCtpsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const std::complex<float> *Ap,
                                   std::complex<float> *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasZtpsv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans,
                                   cublasDiagType_t diag, std::int64_t n,
                                   const std::complex<double> *Ap,
                                   std::complex<double> *X,
                                   std::int64_t incx) = nullptr;

cublasStatus_t (*p_cublasSspr_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                  std::int64_t n, const float *alpha,
                                  const float *X, std::int64_t incx,
                                  float *Ap) = nullptr;

cublasStatus_t (*p_cublasDspr_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                  std::int64_t n, const double *alpha,
                                  const double *X, std::int64_t incx,
                                  double *Ap) = nullptr;

cublasStatus_t (*p_cublasSspr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   std::int64_t n, const float *alpha,
                                   const float *X, std::int64_t incx,
                                   const float *Y, std::int64_t incy,
                                   float *Ap) = nullptr;

cublasStatus_t (*p_cublasDspr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   std::int64_t n, const double *alpha,
                                   const double *X, std::int64_t incx,
                                   const double *Y, std::int64_t incy,
                                   double *Ap) = nullptr;

//LEVEL 3 ROUTINES

cublasStatus_t (*p_cublasSgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    const float *B, int ldb, const float *beta, float *C, int ldc) = nullptr;

cublasStatus_t (*p_cublasDgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    const double *B, int ldb, const double *beta, double *C, int ldc) = nullptr;

cublasStatus_t (*p_cublasCgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda,
    const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C,
    int ldc) = nullptr;

cublasStatus_t (*p_cublasZgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A,
    int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc) = nullptr;

cublasStatus_t (*p_cublasSsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, int, int, const float *,
                                   const float *, int, const float *, int,
                                   const float *, float *, int) = nullptr;

cublasStatus_t (*p_cublasDsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, int, int, const double *,
                                   const double *, int, const double *, int,
                                   const double *, double *, int) = nullptr;

cublasStatus_t (*p_cublasCsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, int, int,
                                   const cuComplex *, const cuComplex *, int,
                                   const cuComplex *, int, const cuComplex *,
                                   cuComplex *, int) = nullptr;

cublasStatus_t (*p_cublasZsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, int, int,
                                   const cuDoubleComplex *,
                                   const cuDoubleComplex *, int,
                                   const cuDoubleComplex *, int,
                                   const cuDoubleComplex *, cuDoubleComplex *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasSsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                   cublasOperation_t, int, int, const float *,
                                   const float *, int, const float *, float *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasDsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                   cublasOperation_t, int, int, const double *,
                                   const double *, int, const double *,
                                   double *, int) = nullptr;

cublasStatus_t (*p_cublasCsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                   cublasOperation_t, int, int,
                                   const cuComplex *, const cuComplex *, int,
                                   const cuComplex *, cuComplex *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasZsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                   cublasOperation_t, int, int,
                                   const cuDoubleComplex *,
                                   const cuDoubleComplex *, int,
                                   const cuDoubleComplex *, cuDoubleComplex *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasSsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                    cublasOperation_t, int, int, const float *,
                                    const float *, int, const float *, int,
                                    const float *, float *, int) = nullptr;

cublasStatus_t (*p_cublasDsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                    cublasOperation_t, int, int, const double *,
                                    const double *, int, const double *, int,
                                    const double *, double *, int) = nullptr;

cublasStatus_t (*p_cublasCsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                    cublasOperation_t, int, int,
                                    const cuComplex *, const cuComplex *, int,
                                    const cuComplex *, int, const cuComplex *,
                                    cuComplex *, int) = nullptr;

cublasStatus_t (*p_cublasZsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                    cublasOperation_t, int, int,
                                    const cuDoubleComplex *,
                                    const cuDoubleComplex *, int,
                                    const cuDoubleComplex *, int,
                                    const cuDoubleComplex *, cuDoubleComplex *,
                                    int) = nullptr;

cublasStatus_t (*p_cublasSsyr2k_v2)(
    cublasHandle_t handle,
    cublasFillMode_t uplo,
    cublasOperation_t trans,
    int n,
    int k,
    const float *alpha,
    const float *A,
    int lda,
    const float *B,
    int ldb,
    const float *beta,
    float *C,
    int ldc) = nullptr;

cublasStatus_t (*p_cublasSsbmv_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   std::int64_t n, std::int64_t k,
                                   const float *alpha, const float *A,
                                   std::int64_t lda, const float *X,
                                   std::int64_t incx, const float *beta,
                                   float *Y, std::int64_t incy) = nullptr;

cublasStatus_t (*p_cublasStrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int, const float *,
                                   const float *, int, float *, int, float *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasDtrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int, const double *,
                                   const double *, int, double *, int, double *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasCtrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int,
                                   const cuComplex *, const cuComplex *, int,
                                   cuComplex *, int, cuComplex *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasZtrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int,
                                   const cuDoubleComplex *,
                                   const cuDoubleComplex *, int,
                                   cuDoubleComplex *, int, cuDoubleComplex *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasStrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int, const float *,
                                   const float *, int, float *, int) = nullptr;

cublasStatus_t (*p_cublasDtrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int, const double *,
                                   const double *, int, double *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasCtrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int,
                                   const cuComplex *, const cuComplex *, int,
                                   cuComplex *, int) = nullptr;

cublasStatus_t (*p_cublasZtrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int,
                                   const cuDoubleComplex *,
                                   const cuDoubleComplex *, int,
                                   cuDoubleComplex *, int) = nullptr;

cublasStatus_t (*p_cublasChemm_v2)(cublasHandle_t handle, cublasSideMode_t side,
                                   cublasFillMode_t uplo, int m, int n,
                                   const cuComplex *alpha, const cuComplex *A,
                                   int lda, const cuComplex *B, int ldb,
                                   const cuComplex *beta, cuComplex *C,
                                   int ldc) = nullptr;

cublasStatus_t (*p_cublasZhemm_v2)(cublasHandle_t handle, cublasSideMode_t side,
                                   cublasFillMode_t uplo, int m, int n,
                                   const cuDoubleComplex *alpha,
                                   const cuDoubleComplex *A, int lda,
                                   const cuDoubleComplex *B, int ldb,
                                   const cuDoubleComplex *beta,
                                   cuDoubleComplex *C, int ldc) = nullptr;

cublasStatus_t (*p_cublasCherk_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans, int n, int k,
                                   const float *alpha, const cuComplex *A,
                                   int lda, const float *beta, cuComplex *C,
                                   int ldc) = nullptr;

cublasStatus_t (*p_cublasZherk_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   cublasOperation_t trans, int n, int k,
                                   const double *alpha,
                                   const cuDoubleComplex *A, int lda,
                                   const double *beta, cuDoubleComplex *C,
                                   int ldc) = nullptr;

cublasStatus_t (*p_cublasCher2k_v2)(cublasHandle_t handle,
                                    cublasFillMode_t uplo,
                                    cublasOperation_t trans, int n, int k,
                                    const cuComplex *alpha, const cuComplex *A,
                                    int lda, const cuComplex *B, int ldb,
                                    const float *beta, cuComplex *C,
                                    int ldc) = nullptr;

cublasStatus_t (*p_cublasZher2k_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A,
    int lda, const cuDoubleComplex *B, int ldb, const double *beta,
    cuDoubleComplex *C, int ldc) = nullptr;


cublasStatus_t (*p_cublasSspr_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                  std::int64_t n, const float *alpha,
                                  const float *X, std::int64_t incx,
                                  float *Ap) = nullptr;

cublasStatus_t (*p_cublasDspr_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                  std::int64_t n, const double *alpha,
                                  const double *X, std::int64_t incx,
                                  double *Ap) = nullptr;

cublasStatus_t (*p_cublasSspr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   std::int64_t n, const float *alpha,
                                   const float *X, std::int64_t incx,
                                   const float *Y, std::int64_t incy,
                                   float *Ap) = nullptr;

cublasStatus_t (*p_cublasDspr2_v2)(cublasHandle_t handle, cublasFillMode_t uplo,
                                   std::int64_t n, const double *alpha,
                                   const double *X, std::int64_t incx,
                                   const double *Y, std::int64_t incy,
                                   double *Ap) = nullptr;

//LEVEL 3 ROUTINES

cublasStatus_t (*p_cublasSgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    const float *B, int ldb, const float *beta, float *C, int ldc) = nullptr;

cublasStatus_t (*p_cublasDgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    const double *B, int ldb, const double *beta, double *C, int ldc) = nullptr;

cublasStatus_t (*p_cublasCgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda,
    const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C,
    int ldc) = nullptr;

cublasStatus_t (*p_cublasZgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A,
    int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc) = nullptr;

cublasStatus_t (*p_cublasSsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, int, int, const float *,
                                   const float *, int, const float *, int,
                                   const float *, float *, int) = nullptr;

cublasStatus_t (*p_cublasDsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, int, int, const double *,
                                   const double *, int, const double *, int,
                                   const double *, double *, int) = nullptr;

cublasStatus_t (*p_cublasCsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, int, int,
                                   const cuComplex *, const cuComplex *, int,
                                   const cuComplex *, int, const cuComplex *,
                                   cuComplex *, int) = nullptr;

cublasStatus_t (*p_cublasZsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, int, int,
                                   const cuDoubleComplex *,
                                   const cuDoubleComplex *, int,
                                   const cuDoubleComplex *, int,
                                   const cuDoubleComplex *, cuDoubleComplex *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasSsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                   cublasOperation_t, int, int, const float *,
                                   const float *, int, const float *, float *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasDsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                   cublasOperation_t, int, int, const double *,
                                   const double *, int, const double *,
                                   double *, int) = nullptr;

cublasStatus_t (*p_cublasCsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                   cublasOperation_t, int, int,
                                   const cuComplex *, const cuComplex *, int,
                                   const cuComplex *, cuComplex *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasZsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                   cublasOperation_t, int, int,
                                   const cuDoubleComplex *,
                                   const cuDoubleComplex *, int,
                                   const cuDoubleComplex *, cuDoubleComplex *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasSsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                    cublasOperation_t, int, int, const float *,
                                    const float *, int, const float *, int,
                                    const float *, float *, int) = nullptr;

cublasStatus_t (*p_cublasDsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                    cublasOperation_t, int, int, const double *,
                                    const double *, int, const double *, int,
                                    const double *, double *, int) = nullptr;

cublasStatus_t (*p_cublasCsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                    cublasOperation_t, int, int,
                                    const cuComplex *, const cuComplex *, int,
                                    const cuComplex *, int, const cuComplex *,
                                    cuComplex *, int) = nullptr;

cublasStatus_t (*p_cublasZsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                    cublasOperation_t, int, int,
                                    const cuDoubleComplex *,
                                    const cuDoubleComplex *, int,
                                    const cuDoubleComplex *, int,
                                    const cuDoubleComplex *, cuDoubleComplex *,
                                    int) = nullptr;

cublasStatus_t (*p_cublasStrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int, const float *,
                                   const float *, int, float *, int, float *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasDtrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int, const double *,
                                   const double *, int, double *, int, double *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasCtrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int,
                                   const cuComplex *, const cuComplex *, int,
                                   cuComplex *, int, cuComplex *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasZtrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int,
                                   const cuDoubleComplex *,
                                   const cuDoubleComplex *, int,
                                   cuDoubleComplex *, int, cuDoubleComplex *,
                                   int) = nullptr;

cublasStatus_t (*p_cublasStrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                   cublasFillMode_t, cublasOperation_t,
                                   cublasDiagType_t, int, int, const float *,
                                   const float *, int, float *, int) = nullptr;

//LEVEL 1 ROUTINES

void (*p_cblas_scopy)(int, const float *, int, float *, int) = nullptr;
void (*p_cblas_dcopy)(int, const double *, int, double *, int) = nullptr;
void (*p_cblas_sscal)(int, float, float *, int) = nullptr;
void (*p_cblas_dscal)(int, double, double *, int) = nullptr;
void (*p_cblas_saxpy)(int, float, const float *, int, float *, int) = nullptr;
void (*p_cblas_daxpy)(int, double, const double *, int, double *,
                      int) = nullptr;
int (*p_cblas_isamax)(int, const float *, int) = nullptr;
int (*p_cblas_idamax)(int, const double *, int) = nullptr;
void (*p_cblas_sswap)(int, float *, int, float *, int) = nullptr;
void (*p_cblas_dswap)(int, double *, int, double *, int) = nullptr;
float (*p_cblas_sdot)(int, const float *, int, const float *, int) = nullptr;
double (*p_cblas_ddot)(int, const double *, int, const double *, int) = nullptr;
float (*p_cblas_snrm2)(int, const float *, int) = nullptr;
double (*p_cblas_dnrm2)(int, const double *, int) = nullptr;
float (*p_cblas_sasum)(int, const float *, int) = nullptr;
double (*p_cblas_dasum)(int, const double *, int) = nullptr;
void (*p_cblas_srot)(int, float *, int, float *, int, float, float) = nullptr;
void (*p_cblas_drot)(int, double *, int, double *, int, double,
                     double) = nullptr;
void (*p_cblas_srotg)(float *, float *, float *, float *) = nullptr;
void (*p_cblas_drotg)(double *, double *, double *, double *) = nullptr;
void (*p_cblas_srotm)(int, float *, int, float *, int, const float *) = nullptr;
void (*p_cblas_drotm)(int, double *, int, double *, int,
                      const double *) = nullptr;
void (*p_cblas_srotmg)(float *, float *, float *, float,
                       const float *) = nullptr;
void (*p_cblas_drotmg)(double *, double *, double *, double,
                       const double *) = nullptr;

int (*p_cblas_icamax)(int, const std::complex<float> *, int) = nullptr;

int (*p_cblas_izamax)(int, const std::complex<double> *, int) = nullptr;

float (*p_cblas_scasum)(int, const std::complex<float> *, int) = nullptr;

double (*p_cblas_dzasum)(int, const std::complex<double> *, int) = nullptr;

void (*p_cblas_caxpy)(int, const std::complex<float> *,
                      const std::complex<float> *, int, std::complex<float> *,
                      int) = nullptr;

void (*p_cblas_zaxpy)(int, const std::complex<double> *,
                      const std::complex<double> *, int, std::complex<double> *,
                      int) = nullptr;

void (*p_cblas_ccopy)(int, const std::complex<float> *, int,
                      std::complex<float> *, int) = nullptr;

void (*p_cblas_zcopy)(int, const std::complex<double> *, int,
                      std::complex<double> *, int) = nullptr;

void (*p_cblas_cdotu_sub)(int, const std::complex<float> *, int,
                          const std::complex<float> *, int,
                          std::complex<float> *) = nullptr;

void (*p_cblas_cdotc_sub)(int, const std::complex<float> *, int,
                          const std::complex<float> *, int,
                          std::complex<float> *) = nullptr;

void (*p_cblas_zdotu_sub)(int, const std::complex<double> *, int,
                          const std::complex<double> *, int,
                          std::complex<double> *) = nullptr;

void (*p_cblas_zdotc_sub)(int, const std::complex<double> *, int,
                          const std::complex<double> *, int,
                          std::complex<double> *) = nullptr;

float (*p_cblas_scnrm2)(int, const std::complex<float> *, int) = nullptr;

double (*p_cblas_dznrm2)(int, const std::complex<double> *, int) = nullptr;

void (*p_cblas_cscal)(int, const std::complex<float> *, std::complex<float> *,
                      int) = nullptr;

void (*p_cblas_zscal)(int, const std::complex<double> *, std::complex<double> *,
                      int) = nullptr;

void (*p_cblas_csscal)(int, float, std::complex<float> *, int) = nullptr;

void (*p_cblas_zdscal)(int, double, std::complex<double> *, int) = nullptr;

void (*p_cblas_cswap)(int, std::complex<float> *, int, std::complex<float> *,
                      int) = nullptr;

void (*p_cblas_zswap)(int, std::complex<double> *, int, std::complex<double> *,
                      int) = nullptr;

//LEVEL 2 ROUTINES

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

void (*p_cblas_cgemv)(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE Trans,
                      std::int64_t m, std::int64_t n,
                      const std::complex<float> *alpha,
                      const std::complex<float> *A, std::int64_t lda,
                      const std::complex<float> *x, std::int64_t incx,
                      const std::complex<float> *beta, std::complex<float> *y,
                      std::int64_t incy) = nullptr;

void (*p_cblas_zgemv)(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE Trans,
                      std::int64_t m, std::int64_t n,
                      const std::complex<double> *alpha,
                      const std::complex<double> *A, std::int64_t lda,
                      const std::complex<double> *x, std::int64_t incx,
                      const std::complex<double> *beta, std::complex<double> *y,
                      std::int64_t incy) = nullptr;

void (*p_cblas_ssymv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      const int N, const float alpha, const float *A,
                      const int lda, const float *X, const int incX,
                      const float beta, float *Y, const int incY) = nullptr;

void (*p_cblas_dsymv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      const int N, const double alpha, const double *A,
                      const int lda, const double *X, const int incX,
                      const double beta, double *Y, const int incY) = nullptr;

void (*p_cblas_chemv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const std::complex<float> *alpha,
                      const std::complex<float> *A, std::int64_t lda,
                      const std::complex<float> *x, std::int64_t incx,
                      const std::complex<float> *beta, std::complex<float> *y,
                      std::int64_t incy) = nullptr;

void (*p_cblas_zhemv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const std::complex<double> *alpha,
                      const std::complex<double> *A, std::int64_t lda,
                      const std::complex<double> *x, std::int64_t incx,
                      const std::complex<double> *beta, std::complex<double> *y,
                      std::int64_t incy) = nullptr;

void (*p_cblas_strmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const float *A, std::int64_t lda, float *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_dtrmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const double *A, std::int64_t lda, double *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_ctrmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const std::complex<float> *A, std::int64_t lda,
                      std::complex<float> *X, std::int64_t incx) = nullptr;

void (*p_cblas_ztrmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const std::complex<double> *A, std::int64_t lda,
                      std::complex<double> *X, std::int64_t incx) = nullptr;

void (*p_cblas_strsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const float *A, std::int64_t lda, float *X,
                      std::int64_t incX) = nullptr;

void (*p_cblas_dtrsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const double *A, std::int64_t lda, double *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_ctrsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const std::complex<float> *A, std::int64_t lda,
                      std::complex<float> *X, std::int64_t incX) = nullptr;

void (*p_cblas_ztrsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const std::complex<double> *A, std::int64_t lda,
                      std::complex<double> *X, std::int64_t incX) = nullptr;

void (*p_cblas_cgerc)(const CBLAS_ORDER Order, std::int64_t m, std::int64_t n,
                      const std::complex<float> *alpha,
                      const std::complex<float> *X, std::int64_t incx,
                      const std::complex<float> *Y, std::int64_t incy,
                      std::complex<float> *A, std::int64_t lda) = nullptr;

void (*p_cblas_zgerc)(const CBLAS_ORDER Order, std::int64_t m, std::int64_t n,
                      const std::complex<double> *alpha,
                      const std::complex<double> *X, std::int64_t incx,
                      const std::complex<double> *Y, std::int64_t incy,
                      std::complex<double> *A, std::int64_t lda) = nullptr;

void (*p_cblas_stbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const float *A, std::int64_t lda,
                      float *X, std::int64_t incX) = nullptr;

void (*p_cblas_dtbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const double *A, std::int64_t lda,
                      double *X, std::int64_t incX) = nullptr;

void (*p_cblas_ctbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t k, const std::complex<float> *A,
                      std::int64_t lda, std::complex<float> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_ztbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t k, const std::complex<double> *A,
                      std::int64_t lda, std::complex<double> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_stbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const float *A, std::int64_t lda,
                      float *X, std::int64_t incX) = nullptr;

void (*p_cblas_dtbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const double *A, std::int64_t lda,
                      double *X, std::int64_t incX) = nullptr;

void (*p_cblas_ctbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const std::complex<float> *A,
                      std::int64_t lda, std::complex<float> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_ztbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const std::complex<double> *A,
                      std::int64_t lda, std::complex<double> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_chpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const std::complex<float> *alpha,
                      const std::complex<float> *Ap,
                      const std::complex<float> *X, std::int64_t incx,
                      const std::complex<float> *beta, std::complex<float> *Y,
                      std::int64_t incy) = nullptr;

void (*p_cblas_zhpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const std::complex<double> *alpha,
                      const std::complex<double> *Ap,
                      const std::complex<double> *X, std::int64_t incx,
                      const std::complex<double> *beta, std::complex<double> *Y,
                      std::int64_t incy) = nullptr;

void (*p_cblas_sspmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const float alpha, const float *Ap,
                      const float *x, std::int64_t incX, const float beta,
                      float *y, std::int64_t incY) = nullptr;

void (*p_cblas_dspmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const double alpha, const double *Ap,
                      const double *x, std::int64_t incX, const double beta,
                      double *y, std::int64_t incY) = nullptr;

void (*p_cblas_stpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const float *Ap, float *X, std::int64_t incX) = nullptr;

void (*p_cblas_dtpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const double *Ap, double *X, std::int64_t incX) = nullptr;

void (*p_cblas_ctpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const std::complex<float> *Ap, std::complex<float> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_ztpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const std::complex<double> *Ap, std::complex<double> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_cgeru)(const CBLAS_ORDER Order, std::int64_t m, std::int64_t n,
                      const std::complex<float> *alpha,
                      const std::complex<float> *X, std::int64_t incx,
                      const std::complex<float> *Y, std::int64_t incy,
                      std::complex<float> *A, std::int64_t lda) = nullptr;

void (*p_cblas_zgeru)(const CBLAS_ORDER Order, std::int64_t m, std::int64_t n,
                      const std::complex<double> *alpha,
                      const std::complex<double> *X, std::int64_t incx,
                      const std::complex<double> *Y, std::int64_t incy,
                      std::complex<double> *A, std::int64_t lda) = nullptr;

void (*p_cblas_ssyr)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                     std::int64_t n, const float alpha, const float *X,
                     std::int64_t incX, float *A, std::int64_t lda) = nullptr;

void (*p_cblas_dtpsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const double *Ap, double *X, std::int64_t incX) = nullptr;

void (*p_cblas_ssyr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, int n,
                      const float alpha, const float *X, int incX,
                      const float *Y, int incY, float *A, int lda) = nullptr;

void (*p_cblas_dsyr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo, int n,
                      const double alpha, const double *X, int incX,
                      const double *Y, int incY, double *A, int lda) = nullptr;

void (*p_cblas_cher)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                     std::int64_t n, const float alpha,
                     const std::complex<float> *x, std::int64_t incx,
                     std::complex<float> *A, std::int64_t lda) = nullptr;

void (*p_cblas_zher)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                     std::int64_t n, const double alpha,
                     const std::complex<double> *x, std::int64_t incx,
                     std::complex<double> *A, std::int64_t lda) = nullptr;

void (*p_cblas_cher2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const std::complex<float> *alpha,
                      const std::complex<float> *x, std::int64_t incx,
                      const std::complex<float> *y, std::int64_t incy,
                      std::complex<float> *A, std::int64_t lda) = nullptr;

void (*p_cblas_zher2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const std::complex<double> *alpha,
                      const std::complex<double> *x, std::int64_t incx,
                      const std::complex<double> *y, std::int64_t incy,
                      std::complex<double> *A, std::int64_t lda) = nullptr;

void (*p_cblas_chpr)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                     std::int64_t n, const float alpha,
                     const std::complex<float> *X, std::int64_t incx,
                     std::complex<float> *ap) = nullptr;

void (*p_cblas_zhpr)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                     std::int64_t n, const double alpha,
                     const std::complex<double> *X, std::int64_t incx,
                     std::complex<double> *ap) = nullptr;

void (*p_cblas_chpr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const std::complex<float> *alpha,
                      const std::complex<float> *X, std::int64_t incx,
                      const std::complex<float> *Y, std::int64_t incy,
                      std::complex<float> *ap) = nullptr;

void (*p_cblas_zhpr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const std::complex<double> *alpha,
                      const std::complex<double> *X, std::int64_t incx,
                      const std::complex<double> *Y, std::int64_t incy,
                      std::complex<double> *ap) = nullptr;

void (*p_cblas_sgbmv)(const CBLAS_ORDER Order, CBLAS_TRANSPOSE trans,
                      std::int64_t m, std::int64_t n, std::int64_t KL,
                      std::int64_t KU, const float alpha, const float *A,
                      std::int64_t lda, const float *X, std::int64_t incX,
                      const float beta, float *Y, std::int64_t incY) = nullptr;

void (*p_cblas_ztpsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const std::complex<double> *Ap, std::complex<double> *X,
                      std::int64_t incX) = nullptr;

void (*p_cblas_cgbmv)(const CBLAS_ORDER Order, CBLAS_TRANSPOSE trans,
                      std::int64_t m, std::int64_t n, std::int64_t KL,
                      std::int64_t KU, const std::complex<float> *alpha,
                      const std::complex<float> *A, std::int64_t lda,
                      const std::complex<float> *X, std::int64_t incX,
                      const std::complex<float> *beta, std::complex<float> *Y,
                      std::int64_t incY) = nullptr;

void (*p_cblas_zgbmv)(const CBLAS_ORDER Order, CBLAS_TRANSPOSE trans,
                      std::int64_t m, std::int64_t n, std::int64_t KL,
                      std::int64_t KU, const std::complex<double> *alpha,
                      const std::complex<double> *A, std::int64_t lda,
                      const std::complex<double> *X, std::int64_t incX,
                      const std::complex<double> *beta, std::complex<double> *Y,
                      std::int64_t incY) = nullptr;

void (*p_cblas_chbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, std::int64_t k,
                      const std::complex<float> *alpha,
                      const std::complex<float> *A, std::int64_t lda,
                      const std::complex<float> *X, std::int64_t incx,
                      const std::complex<float> *beta, std::complex<float> *Y,
                      std::int64_t incy) = nullptr;

void (*p_cblas_zhbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, std::int64_t k,
                      const std::complex<double> *alpha,
                      const std::complex<double> *A, std::int64_t lda,
                      const std::complex<double> *X, std::int64_t incx,
                      const std::complex<double> *beta, std::complex<double> *Y,
                      std::int64_t incy) = nullptr;

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

void (*p_cblas_stbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const float *A, std::int64_t lda,
                      float *X, std::int64_t incX) = nullptr;

void (*p_cblas_dtbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const double *A, std::int64_t lda,
                      double *X, std::int64_t incX) = nullptr;

void (*p_cblas_ctbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t k, const std::complex<float> *A,
                      std::int64_t lda, std::complex<float> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_ztbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t k, const std::complex<double> *A,
                      std::int64_t lda, std::complex<double> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_stbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const float *A, std::int64_t lda,
                      float *X, std::int64_t incX) = nullptr;

void (*p_cblas_dtbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const double *A, std::int64_t lda,
                      double *X, std::int64_t incX) = nullptr;

void (*p_cblas_ctbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const std::complex<float> *A,
                      std::int64_t lda, std::complex<float> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_ztbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      std::int64_t K, const std::complex<double> *A,
                      std::int64_t lda, std::complex<double> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_chpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const std::complex<float> *alpha,
                      const std::complex<float> *Ap,
                      const std::complex<float> *X, std::int64_t incx,
                      const std::complex<float> *beta, std::complex<float> *Y,
                      std::int64_t incy) = nullptr;

void (*p_cblas_zhpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const std::complex<double> *alpha,
                      const std::complex<double> *Ap,
                      const std::complex<double> *X, std::int64_t incx,
                      const std::complex<double> *beta, std::complex<double> *Y,
                      std::int64_t incy) = nullptr;

void (*p_cblas_sspmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const float alpha, const float *Ap,
                      const float *x, std::int64_t incX, const float beta,
                      float *y, std::int64_t incY) = nullptr;

void (*p_cblas_dspmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const double alpha, const double *Ap,
                      const double *x, std::int64_t incX, const double beta,
                      double *y, std::int64_t incY) = nullptr;

void (*p_cblas_stpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const float *Ap, float *X, std::int64_t incX) = nullptr;

void (*p_cblas_dtpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const double *Ap, double *X, std::int64_t incX) = nullptr;

void (*p_cblas_ctpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const std::complex<float> *Ap, std::complex<float> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_ztpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const std::complex<double> *Ap, std::complex<double> *X,
                      std::int64_t incx) = nullptr;

void (*p_cblas_stpsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const float *Ap, float *X, std::int64_t incX) = nullptr;

void (*p_cblas_dtpsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const double *Ap, double *X, std::int64_t incX) = nullptr;

void (*p_cblas_ctpsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const std::complex<float> *Ap, std::complex<float> *X,
                      std::int64_t incX) = nullptr;

void (*p_cblas_ztpsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, std::int64_t n,
                      const std::complex<double> *Ap, std::complex<double> *X,
                      std::int64_t incX) = nullptr;

void (*p_cblas_sspr)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                     std::int64_t n, const float alpha, const float *X,
                     std::int64_t incX, float *Ap) = nullptr;

void (*p_cblas_dspr)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                     std::int64_t n, const double alpha, const double *X,
                     std::int64_t incX, double *Ap) = nullptr;

void (*p_cblas_sspr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const float alpha, const float *X,
                      std::int64_t incX, const float *Y, std::int64_t incY,
                      float *A) = nullptr;

void (*p_cblas_dspr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      std::int64_t n, const double alpha, const double *X,
                      std::int64_t incX, const double *Y, std::int64_t incY,
                      double *A) = nullptr;

//LEVEL 3 ROUTINES

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

void (*p_cblas_ssyr2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                       const CBLAS_TRANSPOSE Trans, std::int64_t N,
                       std::int64_t K, const float alpha, const float *A,
                       std::int64_t lda, const float *B, std::int64_t ldb,
                       const float beta, float *C, std::int64_t ldc) = nullptr;

void (*p_cblas_dsyr2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                       const CBLAS_TRANSPOSE Trans, std::int64_t N,
                       std::int64_t K, const double alpha, const double *A,
                       std::int64_t lda, const double *B, std::int64_t ldb,
                       const double beta, double *C,
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

void (*p_cblas_cgemm)(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, std::int64_t M,
                      std::int64_t N, std::int64_t K, const void *alpha,
                      const void *A, std::int64_t lda, const void *B,
                      std::int64_t ldb, const void *beta, void *C,
                      std::int64_t ldc) = nullptr;

void (*p_cblas_zgemm)(const CBLAS_ORDER Order, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_TRANSPOSE TransB, std::int64_t M,
                      std::int64_t N, std::int64_t K, const void *alpha,
                      const void *A, std::int64_t lda, const void *B,
                      std::int64_t ldb, const void *beta, void *C,
                      std::int64_t ldc) = nullptr;

void (*p_cblas_csymm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, std::int64_t M, std::int64_t N,
                      const void *alpha, const void *A, std::int64_t lda,
                      const void *B, std::int64_t ldb, const void *beta,
                      void *C, std::int64_t ldc) = nullptr;

void (*p_cblas_zsymm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, std::int64_t M, std::int64_t N,
                      const void *alpha, const void *A, std::int64_t lda,
                      const void *B, std::int64_t ldb, const void *beta,
                      void *C, std::int64_t ldc) = nullptr;

void (*p_cblas_csyrk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      const CBLAS_TRANSPOSE Trans, std::int64_t N,
                      std::int64_t K, const void *alpha, const void *A,
                      std::int64_t lda, const void *beta, void *C,
                      std::int64_t ldc) = nullptr;

void (*p_cblas_zsyrk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      const CBLAS_TRANSPOSE Trans, std::int64_t N,
                      std::int64_t K, const void *alpha, const void *A,
                      std::int64_t lda, const void *beta, void *C,
                      std::int64_t ldc) = nullptr;

void (*p_cblas_csyr2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                       const CBLAS_TRANSPOSE Trans, std::int64_t N,
                       std::int64_t K, const void *alpha, const void *A,
                       std::int64_t lda, const void *B, std::int64_t ldb,
                       const void *beta, void *C, std::int64_t ldc) = nullptr;

void (*p_cblas_zsyr2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                       const CBLAS_TRANSPOSE Trans, std::int64_t N,
                       std::int64_t K, const void *alpha, const void *A,
                       std::int64_t lda, const void *B, std::int64_t ldb,
                       const void *beta, void *C, std::int64_t ldc) = nullptr;

void (*p_cblas_ctrmm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_DIAG Diag, std::int64_t M, std::int64_t N,
                      const void *alpha, const void *A, std::int64_t lda,
                      void *B, std::int64_t ldb) = nullptr;

void (*p_cblas_ztrmm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_DIAG Diag, std::int64_t M, std::int64_t N,
                      const void *alpha, const void *A, std::int64_t lda,
                      void *B, std::int64_t ldb) = nullptr;

void (*p_cblas_ctrsm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_DIAG Diag, std::int64_t M, std::int64_t N,
                      const void *alpha, const void *A, std::int64_t lda,
                      void *B, std::int64_t ldb) = nullptr;

void (*p_cblas_ztrsm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE TransA,
                      const CBLAS_DIAG Diag, std::int64_t M, std::int64_t N,
                      const void *alpha, const void *A, std::int64_t lda,
                      void *B, std::int64_t ldb) = nullptr;

void (*p_cblas_chemm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, std::int64_t M, std::int64_t N,
                      const void *alpha, const void *A, std::int64_t lda,
                      const void *B, std::int64_t ldb, const void *beta,
                      void *C, std::int64_t ldc) = nullptr;

void (*p_cblas_zhemm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                      const CBLAS_UPLO Uplo, std::int64_t M, std::int64_t N,
                      const void *alpha, const void *A, std::int64_t lda,
                      const void *B, std::int64_t ldb, const void *beta,
                      void *C, std::int64_t ldc) = nullptr;

void (*p_cblas_cherk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      const CBLAS_TRANSPOSE Trans, std::int64_t N,
                      std::int64_t K, const float alpha, const void *A,
                      std::int64_t lda, const float beta, void *C,
                      std::int64_t ldc) = nullptr;

void (*p_cblas_zherk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                      const CBLAS_TRANSPOSE Trans, std::int64_t N,
                      std::int64_t K, const double alpha, const void *A,
                      std::int64_t lda, const double beta, void *C,
                      std::int64_t ldc) = nullptr;

void (*p_cblas_cher2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                       const CBLAS_TRANSPOSE Trans, std::int64_t N,
                       std::int64_t K, const void *alpha, const void *A,
                       std::int64_t lda, const void *B, std::int64_t ldb,
                       const float beta, void *C, std::int64_t ldc) = nullptr;

void (*p_cblas_zher2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                       const CBLAS_TRANSPOSE Trans, std::int64_t N,
                       std::int64_t K, const void *alpha, const void *A,
                       std::int64_t lda, const void *B, std::int64_t ldb,
                       const double beta, void *C, std::int64_t ldc) = nullptr;


//=======ROCBLAS==========

hipError_t (*p_hipMalloc)(void **, size_t) = nullptr;
hipError_t (*p_hipFree)(void *) = nullptr;
hipError_t (*p_hipMemcpy)(void *, const void *, size_t,
                          hipMemcpyKind) = nullptr;

hipError_t (*p_hipDeviceSynchronize)(void) = nullptr;

rocblas_status (*p_rocblas_create_handle)(rocblas_handle *handle) = nullptr;

rocblas_status (*p_rocblas_destroy_handle)(rocblas_handle handle) = nullptr;

//LEVEL 1 ROUTINES

rocblas_status (*p_rocblas_isamax)(rocblas_handle, rocblas_int, const float *,
                                   rocblas_int, rocblas_int *) = nullptr;

rocblas_status (*p_rocblas_idamax)(rocblas_handle, rocblas_int, const double *,
                                   rocblas_int, rocblas_int *) = nullptr;

rocblas_status (*p_rocblas_sasum)(rocblas_handle, rocblas_int, const float *,
                                  rocblas_int, float *) = nullptr;

rocblas_status (*p_rocblas_dasum)(rocblas_handle, rocblas_int, const double *,
                                  rocblas_int, double *) = nullptr;

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

rocblas_status (*p_rocblas_snrm2)(rocblas_handle, rocblas_int, const float *,
                                  rocblas_int, float *) = nullptr;

rocblas_status (*p_rocblas_dnrm2)(rocblas_handle, rocblas_int, const double *,
                                  rocblas_int, double *) = nullptr;

rocblas_status (*p_rocblas_srot)(rocblas_handle, rocblas_int, float *,
                                 rocblas_int, float *, rocblas_int,
                                 const float *, const float *) = nullptr;

rocblas_status (*p_rocblas_drot)(rocblas_handle, rocblas_int, double *,
                                 rocblas_int, double *, rocblas_int,
                                 const double *, const double *) = nullptr;

rocblas_status (*p_rocblas_srotg)(rocblas_handle, float *, float *, float *,
                                  float *) = nullptr;

rocblas_status (*p_rocblas_drotg)(rocblas_handle, double *, double *, double *,
                                  double *) = nullptr;

rocblas_status (*p_rocblas_srotm)(rocblas_handle, rocblas_int, float *,
                                  rocblas_int, float *, rocblas_int,
                                  const float *) = nullptr;

rocblas_status (*p_rocblas_drotm)(rocblas_handle, rocblas_int, double *,
                                  rocblas_int, double *, rocblas_int,
                                  const double *) = nullptr;

rocblas_status (*p_rocblas_srotmg)(rocblas_handle, float *, float *, float *,
                                   const float *, float *) = nullptr;

rocblas_status (*p_rocblas_drotmg)(rocblas_handle, double *, double *, double *,
                                   const double *, double *) = nullptr;

rocblas_status (*p_rocblas_sscal)(rocblas_handle, rocblas_int, const float *,
                                  float *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_dscal)(rocblas_handle, rocblas_int, const double *,
                                  double *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_sswap)(rocblas_handle, rocblas_int, float *,
                                  rocblas_int, float *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_dswap)(rocblas_handle, rocblas_int, double *,
                                  rocblas_int, double *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_icamax)(rocblas_handle, rocblas_int,
                                   const rocblas_float_complex *, rocblas_int,
                                   rocblas_int *) = nullptr;

rocblas_status (*p_rocblas_izamax)(rocblas_handle, rocblas_int,
                                   const rocblas_double_complex *, rocblas_int,
                                   rocblas_int *) = nullptr;

rocblas_status (*p_rocblas_scasum)(rocblas_handle, rocblas_int,
                                   const rocblas_float_complex *, rocblas_int,
                                   float *) = nullptr;

rocblas_status (*p_rocblas_dzasum)(rocblas_handle, rocblas_int,
                                   const rocblas_double_complex *, rocblas_int,
                                   double *) = nullptr;

rocblas_status (*p_rocblas_caxpy)(rocblas_handle, rocblas_int,
                                  const rocblas_float_complex *,
                                  const rocblas_float_complex *, rocblas_int,
                                  rocblas_float_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zaxpy)(rocblas_handle, rocblas_int,
                                  const rocblas_double_complex *,
                                  const rocblas_double_complex *, rocblas_int,
                                  rocblas_double_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_ccopy)(rocblas_handle, rocblas_int,
                                  const rocblas_float_complex *, rocblas_int,
                                  rocblas_float_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zcopy)(rocblas_handle, rocblas_int,
                                  const rocblas_double_complex *, rocblas_int,
                                  rocblas_double_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_cdotu)(rocblas_handle, rocblas_int,
                                  const rocblas_float_complex *, rocblas_int,
                                  const rocblas_float_complex *, rocblas_int,
                                  rocblas_float_complex *) = nullptr;

rocblas_status (*p_rocblas_cdotc)(rocblas_handle, rocblas_int,
                                  const rocblas_float_complex *, rocblas_int,
                                  const rocblas_float_complex *, rocblas_int,
                                  rocblas_float_complex *) = nullptr;

rocblas_status (*p_rocblas_zdotu)(rocblas_handle, rocblas_int,
                                  const rocblas_double_complex *, rocblas_int,
                                  const rocblas_double_complex *, rocblas_int,
                                  rocblas_double_complex *) = nullptr;

rocblas_status (*p_rocblas_zdotc)(rocblas_handle, rocblas_int,
                                  const rocblas_double_complex *, rocblas_int,
                                  const rocblas_double_complex *, rocblas_int,
                                  rocblas_double_complex *) = nullptr;

rocblas_status (*p_rocblas_scnrm2)(rocblas_handle, rocblas_int,
                                   const rocblas_float_complex *, rocblas_int,
                                   float *) = nullptr;

rocblas_status (*p_rocblas_dznrm2)(rocblas_handle, rocblas_int,
                                   const rocblas_double_complex *, rocblas_int,
                                   double *) = nullptr;

rocblas_status (*p_rocblas_cscal)(rocblas_handle, rocblas_int,
                                  const rocblas_float_complex *,
                                  rocblas_float_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zscal)(rocblas_handle, rocblas_int,
                                  const rocblas_double_complex *,
                                  rocblas_double_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_csscal)(rocblas_handle, rocblas_int, const float *,
                                   rocblas_float_complex *,
                                   rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zdscal)(rocblas_handle, rocblas_int, const double *,
                                   rocblas_double_complex *,
                                   rocblas_int) = nullptr;

rocblas_status (*p_rocblas_cswap)(rocblas_handle, rocblas_int,
                                  rocblas_float_complex *, rocblas_int,
                                  rocblas_float_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zswap)(rocblas_handle, rocblas_int,
                                  rocblas_double_complex *, rocblas_int,
                                  rocblas_double_complex *,
                                  rocblas_int) = nullptr;

//LEVEL 2 ROUTINES

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

rocblas_status (*p_rocblas_cgemv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, const std::complex<float> *alpha,
    const std::complex<float> *A, std::int64_t lda,
    const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y,
    std::int64_t incy) = nullptr;

rocblas_status (*p_rocblas_zgemv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, const std::complex<double> *alpha,
    const std::complex<double> *A, std::int64_t lda,
    const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy) = nullptr;

rocblas_status (*p_rocblas_ssymv)(rocblas_handle handle, rocblas_fill uplo,
                                  int n, const float *alpha, const float *A,
                                  int lda, const float *X, int incx,
                                  const float *beta, float *Y,
                                  int incy) = nullptr;

rocblas_status (*p_rocblas_dsymv)(rocblas_handle handle, rocblas_fill uplo,
                                  int n, const double *alpha, const double *A,
                                  int lda, const double *X, int incx,
                                  const double *beta, double *Y,
                                  int incy) = nullptr;

rocblas_status (*p_rocblas_chemv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *A,
    std::int64_t lda, const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y,
    std::int64_t incy) = nullptr;

rocblas_status (*p_rocblas_zhemv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy) = nullptr;

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

rocblas_status (*p_rocblas_ctrmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const std::complex<float> *A,
                                  std::int64_t lda, std::complex<float> *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_ztrmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const std::complex<double> *A,
                                  std::int64_t lda, std::complex<double> *X,
                                  std::int64_t incx) = nullptr;

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

rocblas_status (*p_rocblas_ctrsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const std::complex<float> *A,
                                  std::int64_t lda, std::complex<float> *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_ztrsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const std::complex<double> *A,
                                  std::int64_t lda, std::complex<double> *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_chpmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *Ap,
    const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y,
    std::int64_t incy) = nullptr;

rocblas_status (*p_rocblas_zhpmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *Ap,
    const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy) = nullptr;

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

rocblas_status (*p_rocblas_stbmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  std::int64_t k, const float *A,
                                  std::int64_t lda, float *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_dtbmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  std::int64_t k, const double *A,
                                  std::int64_t lda, double *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_ctbmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  std::int64_t k, const std::complex<float> *A,
                                  std::int64_t lda, std::complex<float> *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_ztbmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  std::int64_t k, const std::complex<double> *A,
                                  std::int64_t lda, std::complex<double> *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_stbsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  std::int64_t k, const float *A,
                                  std::int64_t lda, float *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_dtbsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  std::int64_t k, const double *A,
                                  std::int64_t lda, double *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_ctbsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  std::int64_t k, const std::complex<float> *A,
                                  std::int64_t lda, std::complex<float> *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_ztbsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  std::int64_t k, const std::complex<double> *A,
                                  std::int64_t lda, std::complex<double> *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_sger)(rocblas_handle handle, std::int64_t m,
                                 std::int64_t n, const float *alpha,
                                 const float *X, std::int64_t incx,
                                 const float *Y, std::int64_t incy, float *A,
                                 std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_dger)(rocblas_handle handle, std::int64_t m,
                                 std::int64_t n, const double *alpha,
                                 const double *X, std::int64_t incx,
                                 const double *Y, std::int64_t incy, double *A,
                                 std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_cgerc)(
    rocblas_handle handle, std::int64_t m, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_zgerc)(
    rocblas_handle handle, std::int64_t m, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_cgeru)(
    rocblas_handle handle, std::int64_t m, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_zgeru)(
    rocblas_handle handle, std::int64_t m, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_ssyr)(rocblas_handle handle, rocblas_fill uplo,
                                 std::int64_t n, const float *alpha,
                                 const float *X, std::int64_t incx, float *A,
                                 std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_dsyr)(rocblas_handle handle, rocblas_fill uplo,
                                 std::int64_t n, const double *alpha,
                                 const double *X, std::int64_t incx, double *A,
                                 std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_ssyr2)(rocblas_handle handle, rocblas_fill uplo,
                                  std::int64_t n, const float *alpha,
                                  const float *X, std::int64_t incx,
                                  const float *Y, std::int64_t incy, float *A,
                                  std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_dsyr2)(rocblas_handle handle, rocblas_fill uplo,
                                  std::int64_t n, const double *alpha,
                                  const double *X, std::int64_t incx,
                                  const double *Y, std::int64_t incy, double *A,
                                  std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_cher)(rocblas_handle handle, rocblas_fill uplo,
                                 std::int64_t n, const float *alpha,
                                 const std::complex<float> *X,
                                 std::int64_t incx, std::complex<float> *A,
                                 std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_zher)(rocblas_handle handle, rocblas_fill uplo,
                                 std::int64_t n, const double *alpha,
                                 const std::complex<double> *X,
                                 std::int64_t incx, std::complex<double> *A,
                                 std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_cher2)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_zher2)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda) = nullptr;

rocblas_status (*p_rocblas_chpr)(rocblas_handle handle, rocblas_fill uplo,
                                 std::int64_t n, const float *alpha,
                                 const std::complex<float> *X,
                                 std::int64_t incx,
                                 std::complex<float> *AP) = nullptr;

rocblas_status (*p_rocblas_zhpr)(rocblas_handle handle, rocblas_fill uplo,
                                 std::int64_t n, const double *alpha,
                                 const std::complex<double> *X,
                                 std::int64_t incx,
                                 std::complex<double> *AP) = nullptr;

rocblas_status (*p_rocblas_chpr2)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *AP) = nullptr;

rocblas_status (*p_rocblas_zhpr2)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *AP) = nullptr;

rocblas_status (*p_rocblas_sgbmv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU, const float *alpha,
    const float *A, std::int64_t lda, const float *X, std::int64_t incx,
    const float *beta, float *Y, std::int64_t incy) = nullptr;

rocblas_status (*p_rocblas_dgbmv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU, const double *alpha,
    const double *A, std::int64_t lda, const double *X, std::int64_t incx,
    const double *beta, double *Y, std::int64_t incy) = nullptr;

rocblas_status (*p_rocblas_cgbmv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU,
    const std::complex<float> *alpha, const std::complex<float> *A,
    std::int64_t lda, const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y,
    std::int64_t incy) = nullptr;

rocblas_status (*p_rocblas_zgbmv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy) = nullptr;

rocblas_status (*p_rocblas_chbmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n, std::int64_t k,
    const std::complex<float> *alpha, const std::complex<float> *A,
    std::int64_t lda, const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y,
    std::int64_t incy) = nullptr;

rocblas_status (*p_rocblas_zhbmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n, int64_t k,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy) = nullptr;

rocblas_status (*p_rocblas_sspmv)(rocblas_handle handle, rocblas_fill uplo,
                                  std::int64_t n, const float *alpha,
                                  const float *Ap, const float *x,
                                  std::int64_t incX, const float *beta,
                                  float *y, std::int64_t incY) = nullptr;

rocblas_status (*p_rocblas_dspmv)(rocblas_handle handle, rocblas_fill uplo,
                                  std::int64_t n, const double *alpha,
                                  const double *Ap, const double *x,
                                  std::int64_t incX, const double *beta,
                                  double *y, std::int64_t incY) = nullptr;

rocblas_status (*p_rocblas_stpmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const float *Ap, float *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_dtpmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const double *Ap, double *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_ctpmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const std::complex<float> *Ap,
                                  std::complex<float> *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_ztpmv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const std::complex<double> *Ap,
                                  std::complex<double> *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_stpsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const float *Ap, float *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_dtpsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const double *Ap, double *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_ctpsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const std::complex<float> *Ap,
                                  std::complex<float> *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_ztpsv)(rocblas_handle handle, rocblas_fill uplo,
                                  rocblas_operation trans,
                                  rocblas_diagonal diag, std::int64_t n,
                                  const std::complex<double> *Ap,
                                  std::complex<double> *X,
                                  std::int64_t incx) = nullptr;

rocblas_status (*p_rocblas_sspr)(rocblas_handle handle, rocblas_fill uplo,
                                 std::int64_t n, const float *alpha,
                                 const float *X, std::int64_t incx,
                                 float *Ap) = nullptr;

rocblas_status (*p_rocblas_dspr)(rocblas_handle handle, rocblas_fill uplo,
                                 std::int64_t n, const double *alpha,
                                 const double *X, std::int64_t incx,
                                 double *Ap) = nullptr;

rocblas_status (*p_rocblas_sspr2)(rocblas_handle handle, rocblas_fill uplo,
                                  std::int64_t n, const float *alpha,
                                  const float *X, std::int64_t incx,
                                  const float *Y, std::int64_t incy,
                                  float *Ap) = nullptr;

rocblas_status (*p_rocblas_dspr2)(rocblas_handle handle, rocblas_fill uplo,
                                  std::int64_t n, const double *alpha,
                                  const double *X, std::int64_t incx,
                                  const double *Y, std::int64_t incy,
                                  double *Ap) = nullptr;

//LEVEL 3 ROUTINES

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

rocblas_status (*p_rocblas_ssyr2k)(rocblas_handle, rocblas_fill,
                                   rocblas_operation, rocblas_int, rocblas_int,
                                   const float *, const float *, rocblas_int,
                                   const float *, rocblas_int, const float *,
                                   float *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_dsyr2k)(rocblas_handle, rocblas_fill,
                                   rocblas_operation, rocblas_int, rocblas_int,
                                   const double *, const double *, rocblas_int,
                                   const double *, rocblas_int, const double *,
                                   double *, rocblas_int) = nullptr;

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

rocblas_status (*p_rocblas_cgemm)(rocblas_handle, rocblas_operation,
                                  rocblas_operation, rocblas_int, rocblas_int,
                                  rocblas_int, const rocblas_float_complex *,
                                  const rocblas_float_complex *, rocblas_int,
                                  const rocblas_float_complex *, rocblas_int,
                                  const rocblas_float_complex *,
                                  rocblas_float_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zgemm)(rocblas_handle, rocblas_operation,
                                  rocblas_operation, rocblas_int, rocblas_int,
                                  rocblas_int, const rocblas_double_complex *,
                                  const rocblas_double_complex *, rocblas_int,
                                  const rocblas_double_complex *, rocblas_int,
                                  const rocblas_double_complex *,
                                  rocblas_double_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_csymm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_int, rocblas_int,
    const rocblas_float_complex *, const rocblas_float_complex *, rocblas_int,
    const rocblas_float_complex *, rocblas_int, const rocblas_float_complex *,
    rocblas_float_complex *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zsymm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_int, rocblas_int,
    const rocblas_double_complex *, const rocblas_double_complex *, rocblas_int,
    const rocblas_double_complex *, rocblas_int, const rocblas_double_complex *,
    rocblas_double_complex *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_csyrk)(rocblas_handle, rocblas_fill,
                                  rocblas_operation, rocblas_int, rocblas_int,
                                  const rocblas_float_complex *,
                                  const rocblas_float_complex *, rocblas_int,
                                  const rocblas_float_complex *,
                                  rocblas_float_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zsyrk)(rocblas_handle, rocblas_fill,
                                  rocblas_operation, rocblas_int, rocblas_int,
                                  const rocblas_double_complex *,
                                  const rocblas_double_complex *, rocblas_int,
                                  const rocblas_double_complex *,
                                  rocblas_double_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_csyr2k)(
    rocblas_handle, rocblas_fill, rocblas_operation, rocblas_int, rocblas_int,
    const rocblas_float_complex *, const rocblas_float_complex *, rocblas_int,
    const rocblas_float_complex *, rocblas_int, const rocblas_float_complex *,
    rocblas_float_complex *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zsyr2k)(
    rocblas_handle, rocblas_fill, rocblas_operation, rocblas_int, rocblas_int,
    const rocblas_double_complex *, const rocblas_double_complex *, rocblas_int,
    const rocblas_double_complex *, rocblas_int, const rocblas_double_complex *,
    rocblas_double_complex *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_ctrmm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_operation,
    rocblas_diagonal, rocblas_int, rocblas_int, const rocblas_float_complex *,
    const rocblas_float_complex *, rocblas_int, const rocblas_float_complex *,
    rocblas_int, rocblas_float_complex *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_ztrmm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_operation,
    rocblas_diagonal, rocblas_int, rocblas_int, const rocblas_double_complex *,
    const rocblas_double_complex *, rocblas_int, const rocblas_double_complex *,
    rocblas_int, rocblas_double_complex *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_ctrsm)(rocblas_handle, rocblas_side, rocblas_fill,
                                  rocblas_operation, rocblas_diagonal,
                                  rocblas_int, rocblas_int,
                                  const rocblas_float_complex *,
                                  const rocblas_float_complex *, rocblas_int,
                                  rocblas_float_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_ztrsm)(rocblas_handle, rocblas_side, rocblas_fill,
                                  rocblas_operation, rocblas_diagonal,
                                  rocblas_int, rocblas_int,
                                  const rocblas_double_complex *,
                                  const rocblas_double_complex *, rocblas_int,
                                  rocblas_double_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_chemm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_int, rocblas_int,
    const rocblas_float_complex *, const rocblas_float_complex *, rocblas_int,
    const rocblas_float_complex *, rocblas_int, const rocblas_float_complex *,
    rocblas_float_complex *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zhemm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_int, rocblas_int,
    const rocblas_double_complex *, const rocblas_double_complex *, rocblas_int,
    const rocblas_double_complex *, rocblas_int, const rocblas_double_complex *,
    rocblas_double_complex *, rocblas_int) = nullptr;

rocblas_status (*p_rocblas_cherk)(rocblas_handle, rocblas_fill,
                                  rocblas_operation, rocblas_int, rocblas_int,
                                  const float *, const rocblas_float_complex *,
                                  rocblas_int, const float *,
                                  rocblas_float_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zherk)(rocblas_handle, rocblas_fill,
                                  rocblas_operation, rocblas_int, rocblas_int,
                                  const double *,
                                  const rocblas_double_complex *, rocblas_int,
                                  const double *, rocblas_double_complex *,
                                  rocblas_int) = nullptr;

rocblas_status (*p_rocblas_cher2k)(rocblas_handle, rocblas_fill,
                                   rocblas_operation, rocblas_int, rocblas_int,
                                   const rocblas_float_complex *,
                                   const rocblas_float_complex *, rocblas_int,
                                   const rocblas_float_complex *, rocblas_int,
                                   const float *, rocblas_float_complex *,
                                   rocblas_int) = nullptr;

rocblas_status (*p_rocblas_zher2k)(rocblas_handle, rocblas_fill,
                                   rocblas_operation, rocblas_int, rocblas_int,
                                   const rocblas_double_complex *,
                                   const rocblas_double_complex *, rocblas_int,
                                   const rocblas_double_complex *, rocblas_int,
                                   const double *, rocblas_double_complex *,
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

  //SCAL
  LOAD_CUDA_SYM(cublas_lib, cublasSscal_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDscal_v2);

  //AXPY
  LOAD_CUDA_SYM(cublas_lib, cublasSaxpy_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDaxpy_v2);

  //IAMAX
  LOAD_CUDA_SYM(cublas_lib, cublasIsamax_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasIdamax_v2);

  //SWAP
  LOAD_CUDA_SYM(cublas_lib, cublasSswap_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDswap_v2);

  //DOT
  LOAD_CUDA_SYM(cublas_lib, cublasSdot_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDdot_v2);

  //NRM2
  LOAD_CUDA_SYM(cublas_lib, cublasSnrm2_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDnrm2_v2);

  //ASUM
  LOAD_CUDA_SYM(cublas_lib, cublasSasum_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDasum_v2);

  //ROT
  LOAD_CUDA_SYM(cublas_lib, cublasSrot_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDrot_v2);

  //ROTG
  LOAD_CUDA_SYM(cublas_lib, cublasSrotg_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDrotg_v2);

  //ROTM
  LOAD_CUDA_SYM(cublas_lib, cublasSrotm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDrotm_v2);

  //ROTMG
  LOAD_CUDA_SYM(cublas_lib, cublasSrotmg_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDrotmg_v2);


  //Complex

  //IAMAX
  LOAD_CUDA_SYM(cublas_lib, cublasIcamax_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasIzamax_v2);

  //ASUM
  LOAD_CUDA_SYM(cublas_lib, cublasScasum_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDzasum_v2);

  //AXPY
  LOAD_CUDA_SYM(cublas_lib, cublasCaxpy_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZaxpy_v2);

  //COPY
  LOAD_CUDA_SYM(cublas_lib, cublasCcopy_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZcopy_v2);

  //DOT
  LOAD_CUDA_SYM(cublas_lib, cublasCdotu_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCdotc_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZdotu_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZdotc_v2);

  //NRM2
  LOAD_CUDA_SYM(cublas_lib, cublasScnrm2_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDznrm2_v2);

  //SCAL
  LOAD_CUDA_SYM(cublas_lib, cublasCscal_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZscal_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCsscal_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZdscal_v2);

  //SWAP
  LOAD_CUDA_SYM(cublas_lib, cublasCswap_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZswap_v2);

  //GEMV
  LOAD_CUDA_SYM(cublas_lib, cublasSgemv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDgemv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCgemv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZgemv_v2);

  //SYMV
  LOAD_CUDA_SYM(cublas_lib, cublasSsymv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDsymv_v2);
  
  //HEMV
  LOAD_CUDA_SYM(cublas_lib, cublasChemv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZhemv_v2);
 
  //TRMV
  LOAD_CUDA_SYM(cublas_lib, cublasStrmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtrmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCtrmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZtrmv_v2);
  
  //TRSV
  LOAD_CUDA_SYM(cublas_lib, cublasStrsv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtrsv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCtrsv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZtrsv_v2);
  
  //GERC
  LOAD_CUDA_SYM(cublas_lib, cublasCgerc_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZgerc_v2);
  
  //GER
  LOAD_CUDA_SYM(cublas_lib, cublasSger_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDger_v2);
  
  //GERU
  LOAD_CUDA_SYM(cublas_lib, cublasCgeru_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZgeru_v2);
  
  //SYR
  LOAD_CUDA_SYM(cublas_lib, cublasSsyr_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDsyr_v2);
  
  //SYR2
  LOAD_CUDA_SYM(cublas_lib, cublasSsyr2_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDsyr2_v2);
  
  //HER
  LOAD_CUDA_SYM(cublas_lib, cublasCher_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZher_v2);
  
  //HER2
  LOAD_CUDA_SYM(cublas_lib, cublasCher2_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZher2_v2);
  
  //HPR
  LOAD_CUDA_SYM(cublas_lib, cublasChpr_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZhpr_v2);
  
  //HPR2
  LOAD_CUDA_SYM(cublas_lib, cublasChpr2_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZhpr2_v2);
  
  //GBMV
  LOAD_CUDA_SYM(cublas_lib, cublasSgbmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDgbmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCgbmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZgbmv_v2);
 
  //HBMV
  LOAD_CUDA_SYM(cublas_lib, cublasChbmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZhbmv_v2);
 
  //SBMV
  LOAD_CUDA_SYM(cublas_lib, cublasSsbmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDsbmv_v2);
 
  //TBMV
  LOAD_CUDA_SYM(cublas_lib, cublasStbmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtbmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCtbmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZtbmv_v2);
 
  //TBSV
  LOAD_CUDA_SYM(cublas_lib, cublasStbsv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtbsv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCtbsv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZtbsv_v2);
 
  //HPMV
  LOAD_CUDA_SYM(cublas_lib, cublasChpmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZhpmv_v2);
 
  //SPMV
  LOAD_CUDA_SYM(cublas_lib, cublasSspmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDspmv_v2);
 
  //TPMV
  LOAD_CUDA_SYM(cublas_lib, cublasStpmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtpmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCtpmv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZtpmv_v2);
 
  //TPSV
  LOAD_CUDA_SYM(cublas_lib, cublasStpsv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtpsv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCtpsv_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZtpsv_v2);
 
  //SPR
  LOAD_CUDA_SYM(cublas_lib, cublasSspr_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDspr_v2);
 
  //SPR2
  LOAD_CUDA_SYM(cublas_lib, cublasSspr2_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDspr2_v2);

  //Level 3

  //GEMM
  LOAD_CUDA_SYM(cublas_lib, cublasSgemm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDgemm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCgemm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZgemm_v2);

  //SYMM
  LOAD_CUDA_SYM(cublas_lib, cublasSsymm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDsymm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCsymm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZsymm_v2);

  //SYRK
  LOAD_CUDA_SYM(cublas_lib, cublasSsyrk_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDsyrk_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCsyrk_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZsyrk_v2);

  //SYRK2
  LOAD_CUDA_SYM(cublas_lib, cublasSsyr2k_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDsyr2k_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCsyr2k_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZsyr2k_v2);

  //TRMM
  LOAD_CUDA_SYM(cublas_lib, cublasStrmm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtrmm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCtrmm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZtrmm_v2);

  //TRSM
  LOAD_CUDA_SYM(cublas_lib, cublasStrsm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasDtrsm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasCtrsm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZtrsm_v2);

  //HEMM
  LOAD_CUDA_SYM(cublas_lib, cublasChemm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZhemm_v2);

  //HERK
  LOAD_CUDA_SYM(cublas_lib, cublasCherk_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZherk_v2);

  //HER2K
  LOAD_CUDA_SYM(cublas_lib, cublasCher2k_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZher2k_v2);

  //HEMM
  LOAD_CUDA_SYM(cublas_lib, cublasChemm_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZhemm_v2);

  //HERK
  LOAD_CUDA_SYM(cublas_lib, cublasCherk_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZherk_v2);

  //HER2K
  LOAD_CUDA_SYM(cublas_lib, cublasCher2k_v2);
  LOAD_CUDA_SYM(cublas_lib, cublasZher2k_v2);

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

  //SCAL
  LOAD_OBLAS_SYM(openblas_lib, cblas_sscal);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dscal);

  //AXPY
  LOAD_OBLAS_SYM(openblas_lib, cblas_saxpy);
  LOAD_OBLAS_SYM(openblas_lib, cblas_daxpy);

  //IAMAX
  LOAD_OBLAS_SYM(openblas_lib, cblas_isamax);
  LOAD_OBLAS_SYM(openblas_lib, cblas_idamax);

  //SWAP
  LOAD_OBLAS_SYM(openblas_lib, cblas_sswap);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dswap);

  //DOT
  LOAD_OBLAS_SYM(openblas_lib, cblas_sdot);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ddot);

  //NRM2
  LOAD_OBLAS_SYM(openblas_lib, cblas_snrm2);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dnrm2);

  //ASUM
  LOAD_OBLAS_SYM(openblas_lib, cblas_sasum);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dasum);

  //ROT
  LOAD_OBLAS_SYM(openblas_lib, cblas_srot);
  LOAD_OBLAS_SYM(openblas_lib, cblas_drot);

  //ROTG
  LOAD_OBLAS_SYM(openblas_lib, cblas_srotg);
  LOAD_OBLAS_SYM(openblas_lib, cblas_drotg);

  //ROTM
  LOAD_OBLAS_SYM(openblas_lib, cblas_srotm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_drotm);

  //ROTMG
  LOAD_OBLAS_SYM(openblas_lib, cblas_srotmg);
  LOAD_OBLAS_SYM(openblas_lib, cblas_drotmg);

  //Complex

  //IAMAX
  LOAD_OBLAS_SYM(openblas_lib, cblas_icamax);
  LOAD_OBLAS_SYM(openblas_lib, cblas_izamax);

  //ASUM
  LOAD_OBLAS_SYM(openblas_lib, cblas_scasum);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dzasum);

  //AXPY
  LOAD_OBLAS_SYM(openblas_lib, cblas_caxpy);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zaxpy);

  //COPY
  LOAD_OBLAS_SYM(openblas_lib, cblas_ccopy);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zcopy);

  //DOT
  LOAD_OBLAS_SYM(openblas_lib, cblas_cdotu_sub);
  LOAD_OBLAS_SYM(openblas_lib, cblas_cdotc_sub);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zdotu_sub);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zdotc_sub);

  //NRM2
  LOAD_OBLAS_SYM(openblas_lib, cblas_scnrm2);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dznrm2);

  //SCAL
  LOAD_OBLAS_SYM(openblas_lib, cblas_cscal);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zscal);
  LOAD_OBLAS_SYM(openblas_lib, cblas_csscal);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zdscal);

  //SWAP
  LOAD_OBLAS_SYM(openblas_lib, cblas_cswap);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zswap);

  //Level 2

  //GEMV
  LOAD_OBLAS_SYM(openblas_lib, cblas_sgemv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dgemv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_cgemv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zgemv);
  
  //SYMV
  LOAD_OBLAS_SYM(openblas_lib, cblas_ssymv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dsymv);

  //HEMV
  LOAD_OBLAS_SYM(openblas_lib, cblas_chemv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zhemv);
  
  //TRMV
  LOAD_OBLAS_SYM(openblas_lib, cblas_strmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtrmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ctrmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ztrmv);
 
  //TRSV
  LOAD_OBLAS_SYM(openblas_lib, cblas_strsv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtrsv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ztrsv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ztrsv);
 
  //GERC
  LOAD_OBLAS_SYM(openblas_lib, cblas_cgerc);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zgerc);
 
  //GER
  LOAD_OBLAS_SYM(openblas_lib, cblas_sger);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dger);
 
  //GERU
  LOAD_OBLAS_SYM(openblas_lib, cblas_cgeru);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zgeru);
 
  //SYR
  LOAD_OBLAS_SYM(openblas_lib, cblas_ssyr);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dsyr);
 
  //SYR2
  LOAD_OBLAS_SYM(openblas_lib, cblas_ssyr2);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dsyr2);
 
  //HER
  LOAD_OBLAS_SYM(openblas_lib, cblas_cher);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zher);
 
  //HER2
  LOAD_OBLAS_SYM(openblas_lib, cblas_cher2);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zher2);
 
  //HPR
  LOAD_OBLAS_SYM(openblas_lib, cblas_chpr);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zhpr);
 
  //HPR2
  LOAD_OBLAS_SYM(openblas_lib, cblas_chpr2);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zhpr2);
 
  //GBMV
  LOAD_OBLAS_SYM(openblas_lib, cblas_sgbmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dgbmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_cgbmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zgbmv);
 
  //HBMV
  LOAD_OBLAS_SYM(openblas_lib, cblas_chbmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zhbmv);
 
  //SBMV
  LOAD_OBLAS_SYM(openblas_lib, cblas_ssbmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dsbmv);
 
  //TBMV
  LOAD_OBLAS_SYM(openblas_lib, cblas_stbmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtbmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ctbmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ztbmv);
 
  //TBSV
  LOAD_OBLAS_SYM(openblas_lib, cblas_stbsv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtbsv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ctbsv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ztbsv);
 
  //HPMV
  LOAD_OBLAS_SYM(openblas_lib, cblas_chpmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zhpmv);
 
  //SPMV
  LOAD_OBLAS_SYM(openblas_lib, cblas_sspmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dspmv);
 
  //TPMV
  LOAD_OBLAS_SYM(openblas_lib, cblas_stpmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtpmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ctpmv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ztpmv);
 
  //TPSV
  LOAD_OBLAS_SYM(openblas_lib, cblas_stpsv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtpsv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ctpsv);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ztpsv);
 
  //SPR
  LOAD_OBLAS_SYM(openblas_lib, cblas_sspr);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dspr);
 
  //SPR2
  LOAD_OBLAS_SYM(openblas_lib, cblas_sspr2);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dspr2);

  //Level 3

  //GEMM
  LOAD_OBLAS_SYM(openblas_lib, cblas_sgemm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dgemm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_cgemm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zgemm);

  //SYMM
  LOAD_OBLAS_SYM(openblas_lib, cblas_ssymm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dsymm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_csymm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zsymm);

  //SYRK
  LOAD_OBLAS_SYM(openblas_lib, cblas_ssyrk);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dsyrk);
  LOAD_OBLAS_SYM(openblas_lib, cblas_csyrk);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zsyrk);

  //SYR2K
  LOAD_OBLAS_SYM(openblas_lib, cblas_ssyr2k);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dsyr2k);
  LOAD_OBLAS_SYM(openblas_lib, cblas_csyr2k);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zsyr2k);

  //TRMM
  LOAD_OBLAS_SYM(openblas_lib, cblas_strmm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtrmm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ctrmm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ztrmm);

  //TRSM
  LOAD_OBLAS_SYM(openblas_lib, cblas_strsm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_dtrsm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ctrsm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_ztrsm);

  //HEMM
  LOAD_OBLAS_SYM(openblas_lib, cblas_chemm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zhemm);

  //HERK
  LOAD_OBLAS_SYM(openblas_lib, cblas_cherk);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zherk);

  //HER2K
  LOAD_OBLAS_SYM(openblas_lib, cblas_cher2k);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zher2k);

  //HEMM
  LOAD_OBLAS_SYM(openblas_lib, cblas_chemm);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zhemm);

  //HERK
  LOAD_OBLAS_SYM(openblas_lib, cblas_cherk);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zherk);

  //HER2K
  LOAD_OBLAS_SYM(openblas_lib, cblas_cher2k);
  LOAD_OBLAS_SYM(openblas_lib, cblas_zher2k);

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

  LOAD_HIP_SYM(hip_lib, hipMalloc);
  LOAD_HIP_SYM(hip_lib, hipFree);
  LOAD_HIP_SYM(hip_lib, hipMemcpy);

  LOAD_HIP_SYM(rocblas_lib, rocblas_create_handle);
  LOAD_HIP_SYM(rocblas_lib, rocblas_destroy_handle);

// ====ROCBLAS====

  // Level 1

  //IAMAX
  LOAD_HIP_SYM(rocblas_lib, rocblas_isamax);
  LOAD_HIP_SYM(rocblas_lib, rocblas_idamax);
  LOAD_HIP_SYM(rocblas_lib, rocblas_icamax);
  LOAD_HIP_SYM(rocblas_lib, rocblas_izamax);

  //ASUM
  LOAD_HIP_SYM(rocblas_lib, rocblas_sasum);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dasum);
  LOAD_HIP_SYM(rocblas_lib, rocblas_scasum);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dzasum);

  //AXPY
  LOAD_HIP_SYM(rocblas_lib, rocblas_saxpy);
  LOAD_HIP_SYM(rocblas_lib, rocblas_daxpy);
  LOAD_HIP_SYM(rocblas_lib, rocblas_caxpy);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zaxpy);

  //COPY
  LOAD_HIP_SYM(rocblas_lib, rocblas_scopy);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dcopy);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ccopy);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zcopy);

  //DOT
  LOAD_HIP_SYM(rocblas_lib, rocblas_sdot);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ddot);
  LOAD_HIP_SYM(rocblas_lib, rocblas_cdotu);
  LOAD_HIP_SYM(rocblas_lib, rocblas_cdotc);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zdotu);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zdotc);

  //NRM2
  LOAD_HIP_SYM(rocblas_lib, rocblas_snrm2);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dnrm2);
  LOAD_HIP_SYM(rocblas_lib, rocblas_scnrm2);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dznrm2);

  //ROT
  LOAD_HIP_SYM(rocblas_lib, rocblas_srot);
  LOAD_HIP_SYM(rocblas_lib, rocblas_drot);
 
  //ROTG
  LOAD_HIP_SYM(rocblas_lib, rocblas_srotg);
  LOAD_HIP_SYM(rocblas_lib, rocblas_drotg);
 
  //ROTM
  LOAD_HIP_SYM(rocblas_lib, rocblas_srotm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_drotm);
 
  //ROTMG
  LOAD_HIP_SYM(rocblas_lib, rocblas_srotmg);
  LOAD_HIP_SYM(rocblas_lib, rocblas_drotmg);

  //SCAL
  LOAD_HIP_SYM(rocblas_lib, rocblas_sscal);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dscal);
  LOAD_HIP_SYM(rocblas_lib, rocblas_cscal);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zscal);
  LOAD_HIP_SYM(rocblas_lib, rocblas_csscal);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zdscal);

  //SWAP
  LOAD_HIP_SYM(rocblas_lib, rocblas_sswap);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dswap);
  LOAD_HIP_SYM(rocblas_lib, rocblas_cswap);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zswap);

//Level 2

  //GEMV
  LOAD_HIP_SYM(rocblas_lib, rocblas_sgemv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dgemv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_cgemv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zgemv);
 
  //SYMV
  LOAD_HIP_SYM(rocblas_lib, rocblas_ssymv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsymv);
 
  //HEMV
  LOAD_HIP_SYM(rocblas_lib, rocblas_chemv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zhemv);
 
  //TRMV
  LOAD_HIP_SYM(rocblas_lib, rocblas_strmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtrmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ctrmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ztrmv);
 
  //TRSV
  LOAD_HIP_SYM(rocblas_lib, rocblas_strsv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtrsv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ctrsv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ztrsv);
 
  //HPMV
  LOAD_HIP_SYM(rocblas_lib, rocblas_chpmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zhpmv);
 
  //SBMV
  LOAD_HIP_SYM(rocblas_lib, rocblas_ssbmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsbmv);
 
  //TBMV
  LOAD_HIP_SYM(rocblas_lib, rocblas_stbmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtbmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ctbmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ztbmv);
 
  //TBSV
  LOAD_HIP_SYM(rocblas_lib, rocblas_stbsv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtbsv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ctbsv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ztbsv);
 
  //GER
  LOAD_HIP_SYM(rocblas_lib, rocblas_sger);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dger);
 
  //GERC
  LOAD_HIP_SYM(rocblas_lib, rocblas_cgerc);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zgerc);
 
  //GERU
  LOAD_HIP_SYM(rocblas_lib, rocblas_cgeru);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zgeru);
 
  //SYR
  LOAD_HIP_SYM(rocblas_lib, rocblas_ssyr);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsyr);
 
  //SYR2
  LOAD_HIP_SYM(rocblas_lib, rocblas_ssyr2);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsyr2);
 
  //HER
  LOAD_HIP_SYM(rocblas_lib, rocblas_cher);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zher);
 
  //HER2
  LOAD_HIP_SYM(rocblas_lib, rocblas_cher2);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zher2);
 
  //HPR
  LOAD_HIP_SYM(rocblas_lib, rocblas_chpr);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zhpr);
 
  //HPR2
  LOAD_HIP_SYM(rocblas_lib, rocblas_chpr2);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zhpr2);
 
  //GBMV
  LOAD_HIP_SYM(rocblas_lib, rocblas_sgbmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dgbmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_cgbmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zgbmv);
 
  //HBMV
  LOAD_HIP_SYM(rocblas_lib, rocblas_chbmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zhbmv);
 
  //SPMV
  LOAD_HIP_SYM(rocblas_lib, rocblas_sspmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dspmv);
 
  //TPMV
  LOAD_HIP_SYM(rocblas_lib, rocblas_stpmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtpmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ctpmv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ztpmv);
 
  //TPSV
  LOAD_HIP_SYM(rocblas_lib, rocblas_stpsv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtpsv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ctpsv);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ztpsv);
 
  //SPR
  LOAD_HIP_SYM(rocblas_lib, rocblas_sspr);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dspr);
 
  //SPR2
  LOAD_HIP_SYM(rocblas_lib, rocblas_sspr2);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dspr2);

  //Level 3

  //GEMM
  LOAD_HIP_SYM(rocblas_lib, rocblas_sgemm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dgemm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_cgemm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zgemm);
 
  //SYMM
  LOAD_HIP_SYM(rocblas_lib, rocblas_ssymm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsymm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_csymm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zsymm);
 
  //SYRK
  LOAD_HIP_SYM(rocblas_lib, rocblas_ssyrk);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsyrk);
  LOAD_HIP_SYM(rocblas_lib, rocblas_csyrk);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zsyrk);
 
  //SYR2K
  LOAD_HIP_SYM(rocblas_lib, rocblas_ssyr2k);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsyr2k);
  LOAD_HIP_SYM(rocblas_lib, rocblas_csyr2k);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zsyr2k);
 
  //TRMM
  LOAD_HIP_SYM(rocblas_lib, rocblas_strmm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtrmm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ctrmm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ztrmm);
 
  //TRSM
  LOAD_HIP_SYM(rocblas_lib, rocblas_strsm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dtrsm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ctrsm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_ztrsm);
 
  //HEMM
  LOAD_HIP_SYM(rocblas_lib, rocblas_chemm);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zhemm);
 
  //HERK
  LOAD_HIP_SYM(rocblas_lib, rocblas_cherk);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zherk);
 
  //HER2K
  LOAD_HIP_SYM(rocblas_lib, rocblas_cher2k);
  LOAD_HIP_SYM(rocblas_lib, rocblas_zher2k);

  // SYR2K

  LOAD_HIP_SYM(rocblas_lib, rocblas_ssyr2k);
  LOAD_HIP_SYM(rocblas_lib, rocblas_dsyr2k);


#pragma GCC diagnostic pop

  hip_loaded = true;

  return true;
}
}
}
