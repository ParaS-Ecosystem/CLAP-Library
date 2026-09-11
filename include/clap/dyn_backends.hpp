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

#pragma once

#include "macros.hpp"
#include "types.hpp"
#include <complex>
#include <cstddef>
#include <cstdint>
#include <dlfcn.h>
#include <iostream>
namespace clap {
namespace dyn {

// =====Cublas=====

extern cudaError_t (*p_cudaMalloc)(void **ptr, size_t size);
extern cudaError_t (*p_cudaFree)(void *ptr);
extern cudaError_t (*p_cudaMemcpy)(void *dst, const void *src, size_t sizeBytes,
                                   cudaMemcpyKind kind);
extern cudaError_t (*p_cudaDeviceSynchronize)(void);
extern cublasStatus_t (*p_cublasSetPointerMode_v2)(cublasHandle_t handle,
                                                   cublasPointerMode_t mode);
extern cudaError_t (*p_cudaMemset)(void *, int, size_t);

extern cublasStatus_t (*p_cublasCreate_v2)(cublasHandle_t *handle);
extern cublasStatus_t (*p_cublasDestroy_v2)(cublasHandle_t handle);
extern cublasStatus_t (*p_cublasSgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    const float *B, int ldb, const float *beta, float *C, int ldc);
extern cublasStatus_t (*p_cublasDgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    const double *B, int ldb, const double *beta, double *C, int ldc);

//Level 1

extern cublasStatus_t (*p_cublasScopy_v2)(cublasHandle_t, int, const float *,
                                          int, float *, int);
extern cublasStatus_t (*p_cublasDcopy_v2)(cublasHandle_t, int, const double *,
                                          int, double *, int);
extern cublasStatus_t (*p_cublasSscal_v2)(cublasHandle_t, int, const float *,
                                          float *, int);
extern cublasStatus_t (*p_cublasDscal_v2)(cublasHandle_t, int, const double *,
                                          double *, int);
extern cublasStatus_t (*p_cublasSaxpy_v2)(cublasHandle_t, int, const float *,
                                          const float *, int, float *, int);
extern cublasStatus_t (*p_cublasDaxpy_v2)(cublasHandle_t, int, const double *,
                                          const double *, int, double *, int);
extern cublasStatus_t (*p_cublasIsamax_v2)(cublasHandle_t, int, const float *,
                                           int, int *);
extern cublasStatus_t (*p_cublasIdamax_v2)(cublasHandle_t, int, const double *,
                                           int, int *);
extern cublasStatus_t (*p_cublasSswap_v2)(cublasHandle_t, int, float *, int,
                                          float *, int);
extern cublasStatus_t (*p_cublasDswap_v2)(cublasHandle_t, int, double *, int,
                                          double *, int);
extern cublasStatus_t (*p_cublasSdot_v2)(cublasHandle_t, int, const float *,
                                         int, const float *, int, float *);
extern cublasStatus_t (*p_cublasDdot_v2)(cublasHandle_t, int, const double *,
                                         int, const double *, int, double *);
extern cublasStatus_t (*p_cublasSnrm2_v2)(cublasHandle_t, int, const float *,
                                          int, float *);
extern cublasStatus_t (*p_cublasDnrm2_v2)(cublasHandle_t, int, const double *,
                                          int, double *);
extern cublasStatus_t (*p_cublasSasum_v2)(cublasHandle_t, int, const float *,
                                          int, float *);
extern cublasStatus_t (*p_cublasDasum_v2)(cublasHandle_t, int, const double *,
                                          int, double *);
extern cublasStatus_t (*p_cublasSrot_v2)(cublasHandle_t, int, float *, int,
                                         float *, int, const float *,
                                         const float *);
extern cublasStatus_t (*p_cublasDrot_v2)(cublasHandle_t, int, double *, int,
                                         double *, int, const double *,
                                         const double *);
extern cublasStatus_t (*p_cublasSrotg_v2)(cublasHandle_t, float *, float *,
                                          float *, float *);
extern cublasStatus_t (*p_cublasDrotg_v2)(cublasHandle_t, double *, double *,
                                          double *, double *);
extern cublasStatus_t (*p_cublasSrotm_v2)(cublasHandle_t, int, float *, int,
                                          float *, int, const float *);
extern cublasStatus_t (*p_cublasDrotm_v2)(cublasHandle_t, int, double *, int,
                                          double *, int, const double *);
extern cublasStatus_t (*p_cublasSrotmg_v2)(cublasHandle_t, float *, float *,
                                           float *, const float *, float *);
extern cublasStatus_t (*p_cublasDrotmg_v2)(cublasHandle_t, double *, double *,
                                           double *, const double *, double *);

extern cublasStatus_t (*p_cublasIcamax_v2)(cublasHandle_t, int,
                                           const cuComplex *, int, int *);
extern cublasStatus_t (*p_cublasIzamax_v2)(cublasHandle_t, int,
                                           const cuDoubleComplex *, int, int *);
extern cublasStatus_t (*p_cublasScasum_v2)(cublasHandle_t, int,
                                           const cuComplex *, int, float *);
extern cublasStatus_t (*p_cublasDzasum_v2)(cublasHandle_t, int,
                                           const cuDoubleComplex *, int,
                                           double *);
extern cublasStatus_t (*p_cublasCaxpy_v2)(cublasHandle_t, int,
                                          const cuComplex *, const cuComplex *,
                                          int, cuComplex *, int);
extern cublasStatus_t (*p_cublasZaxpy_v2)(cublasHandle_t, int,
                                          const cuDoubleComplex *,
                                          const cuDoubleComplex *, int,
                                          cuDoubleComplex *, int);
extern cublasStatus_t (*p_cublasCcopy_v2)(cublasHandle_t, int,
                                          const cuComplex *, int, cuComplex *,
                                          int);
extern cublasStatus_t (*p_cublasZcopy_v2)(cublasHandle_t, int,
                                          const cuDoubleComplex *, int,
                                          cuDoubleComplex *, int);
extern cublasStatus_t (*p_cublasCdotu_v2)(cublasHandle_t, int,
                                          const cuComplex *, int,
                                          const cuComplex *, int, cuComplex *);
extern cublasStatus_t (*p_cublasCdotc_v2)(cublasHandle_t, int,
                                          const cuComplex *, int,
                                          const cuComplex *, int, cuComplex *);
extern cublasStatus_t (*p_cublasZdotu_v2)(cublasHandle_t, int,
                                          const cuDoubleComplex *, int,
                                          const cuDoubleComplex *, int,
                                          cuDoubleComplex *);
extern cublasStatus_t (*p_cublasZdotc_v2)(cublasHandle_t, int,
                                          const cuDoubleComplex *, int,
                                          const cuDoubleComplex *, int,
                                          cuDoubleComplex *);
extern cublasStatus_t (*p_cublasScnrm2_v2)(cublasHandle_t, int,
                                           const cuComplex *, int, float *);
extern cublasStatus_t (*p_cublasDznrm2_v2)(cublasHandle_t, int,
                                           const cuDoubleComplex *, int,
                                           double *);
extern cublasStatus_t (*p_cublasCscal_v2)(cublasHandle_t, int,
                                          const cuComplex *, cuComplex *, int);
extern cublasStatus_t (*p_cublasZscal_v2)(cublasHandle_t, int,
                                          const cuDoubleComplex *,
                                          cuDoubleComplex *, int);
extern cublasStatus_t (*p_cublasCsscal_v2)(cublasHandle_t, int, const float *,
                                           cuComplex *, int);
extern cublasStatus_t (*p_cublasZdscal_v2)(cublasHandle_t, int, const double *,
                                           cuDoubleComplex *, int);
extern cublasStatus_t (*p_cublasCswap_v2)(cublasHandle_t, int, cuComplex *, int,
                                          cuComplex *, int);
extern cublasStatus_t (*p_cublasZswap_v2)(cublasHandle_t, int,
                                          cuDoubleComplex *, int,
                                          cuDoubleComplex *, int);

//Level 2

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
extern cublasStatus_t (*p_cublasCgemv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, const std::complex<float> *alpha,
    const std::complex<float> *A, std::int64_t lda,
    const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y, std::int64_t incy);
extern cublasStatus_t (*p_cublasZgemv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, const std::complex<double> *alpha,
    const std::complex<double> *A, std::int64_t lda,
    const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy);
extern cublasStatus_t (*p_cublasSsymv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, int n,
                                          const float *alpha, const float *A,
                                          int lda, const float *x, int incx,
                                          const float *beta, float *y,
                                          int incy);
extern cublasStatus_t (*p_cublasDsymv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, int n,
                                          const double *alpha, const double *A,
                                          int lda, const double *x, int incx,
                                          const double *beta, double *y,
                                          int incy);
extern cublasStatus_t (*p_cublasChemv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *A,
    std::int64_t lda, const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y, std::int64_t incy);
extern cublasStatus_t (*p_cublasZhemv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy);
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
extern cublasStatus_t (*p_cublasCtrmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, const std::complex<float> *A,
    std::int64_t lda, std::complex<float> *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasZtrmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, const std::complex<double> *A,
    std::int64_t lda, std::complex<double> *X, std::int64_t incx);
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
extern cublasStatus_t (*p_cublasCtrsv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, const std::complex<float> *A,
    std::int64_t lda, std::complex<float> *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasZtrsv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, const std::complex<double> *A,
    std::int64_t lda, std::complex<double> *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasCgerc_v2)(
    cublasHandle_t handle, std::int64_t m, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda);
extern cublasStatus_t (*p_cublasZgerc_v2)(
    cublasHandle_t handle, std::int64_t m, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda);
extern cublasStatus_t (*p_cublasSger_v2)(cublasHandle_t handle, std::int64_t m,
                                         std::int64_t n, const float *alpha,
                                         const float *X, std::int64_t incx,
                                         const float *Y, std::int64_t incy,
                                         float *A, std::int64_t lda);
extern cublasStatus_t (*p_cublasDger_v2)(cublasHandle_t handle, std::int64_t m,
                                         std::int64_t n, const double *alpha,
                                         const double *X, std::int64_t incx,
                                         const double *Y, std::int64_t incy,
                                         double *A, std::int64_t lda);
extern cublasStatus_t (*p_cublasCgeru_v2)(
    cublasHandle_t handle, std::int64_t m, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda);
extern cublasStatus_t (*p_cublasZgeru_v2)(
    cublasHandle_t handle, std::int64_t m, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda);
extern cublasStatus_t (*p_cublasSsyr_v2)(cublasHandle_t handle,
                                         cublasFillMode_t uplo, std::int64_t n,
                                         const float *alpha, const float *X,
                                         std::int64_t incx, float *A,
                                         std::int64_t lda);
extern cublasStatus_t (*p_cublasDsyr_v2)(cublasHandle_t handle,
                                         cublasFillMode_t uplo, std::int64_t n,
                                         const double *alpha, const double *X,
                                         std::int64_t incx, double *A,
                                         std::int64_t lda);
extern cublasStatus_t (*p_cublasSsyr2_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, std::int64_t n,
                                          const float *alpha, const float *X,
                                          std::int64_t incx, const float *Y,
                                          std::int64_t incy, float *A,
                                          std::int64_t lda);
extern cublasStatus_t (*p_cublasDsyr2_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, std::int64_t n,
                                          const double *alpha, const double *X,
                                          std::int64_t incx, const double *Y,
                                          std::int64_t incy, double *A,
                                          std::int64_t lda);
extern cublasStatus_t (*p_cublasCher_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const float *alpha, const std::complex<float> *X, std::int64_t incx,
    std::complex<float> *A, std::int64_t lda);
extern cublasStatus_t (*p_cublasZher_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const double *alpha, const std::complex<double> *X, std::int64_t incx,
    std::complex<double> *A, std::int64_t lda);
extern cublasStatus_t (*p_cublasCher2_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda);
extern cublasStatus_t (*p_cublasZher2_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda);
extern cublasStatus_t (*p_cublasChpr_v2)(cublasHandle_t handle,
                                         cublasFillMode_t uplo, std::int64_t n,
                                         const float *alpha,
                                         const std::complex<float> *X,
                                         std::int64_t incx,
                                         std::complex<float> *AP);
extern cublasStatus_t (*p_cublasZhpr_v2)(cublasHandle_t handle,
                                         cublasFillMode_t uplo, std::int64_t n,
                                         const double *alpha,
                                         const std::complex<double> *X,
                                         std::int64_t incx,
                                         std::complex<double> *AP);
extern cublasStatus_t (*p_cublasChpr2_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *AP);
extern cublasStatus_t (*p_cublasZhpr2_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *AP);
extern cublasStatus_t (*p_cublasSgbmv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU, const float *alpha,
    const float *A, std::int64_t lda, const float *X, std::int64_t incx,
    const float *beta, float *Y, std::int64_t incy);
extern cublasStatus_t (*p_cublasDgbmv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU, const double *alpha,
    const double *A, std::int64_t lda, const double *X, std::int64_t incx,
    const double *beta, double *Y, std::int64_t incy);
extern cublasStatus_t (*p_cublasCgbmv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, std::int64_t Kl, std::int64_t Ku,
    const std::complex<float> *alpha, const std::complex<float> *A,
    std::int64_t lda, const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y, std::int64_t incy);
extern cublasStatus_t (*p_cublasZgbmv_v2)(
    cublasHandle_t handle, cublasOperation_t trans, std::int64_t m,
    std::int64_t n, std::int64_t Kl, std::int64_t Ku,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy);
extern cublasStatus_t (*p_cublasChbmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    std::int64_t k, const std::complex<float> *alpha,
    const std::complex<float> *A, std::int64_t lda,
    const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y, std::int64_t incy);
extern cublasStatus_t (*p_cublasZhbmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n, int64_t k,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy);
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
extern cublasStatus_t (*p_cublasStbmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, std::int64_t k, const float *A,
    std::int64_t lda, float *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasDtbmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, std::int64_t k, const double *A,
    std::int64_t lda, double *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasCtbmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, std::int64_t k,
    const std::complex<float> *A, std::int64_t lda, std::complex<float> *X,
    std::int64_t incx);
extern cublasStatus_t (*p_cublasZtbmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, std::int64_t k,
    const std::complex<double> *A, std::int64_t lda, std::complex<double> *X,
    std::int64_t incx);
extern cublasStatus_t (*p_cublasStbsv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, std::int64_t k, const float *A,
    std::int64_t lda, float *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasDtbsv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, std::int64_t k, const double *A,
    std::int64_t lda, double *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasCtbsv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, std::int64_t k,
    const std::complex<float> *A, std::int64_t lda, std::complex<float> *X,
    std::int64_t incx);
extern cublasStatus_t (*p_cublasZtbsv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, std::int64_t k,
    const std::complex<double> *A, std::int64_t lda, std::complex<double> *X,
    std::int64_t incx);
extern cublasStatus_t (*p_cublasChpmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *Ap,
    const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y, std::int64_t incy);
extern cublasStatus_t (*p_cublasZhpmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *Ap,
    const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy);
extern cublasStatus_t (*p_cublasSspmv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, std::int64_t n,
                                          const float *alpha, const float *Ap,
                                          const float *x, std::int64_t incX,
                                          const float *beta, float *y,
                                          std::int64_t incY);
extern cublasStatus_t (*p_cublasDspmv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, std::int64_t n,
                                          const double *alpha, const double *Ap,
                                          const double *x, std::int64_t incX,
                                          const double *beta, double *y,
                                          std::int64_t incY);
extern cublasStatus_t (*p_cublasStpmv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag, std::int64_t n,
                                          const float *Ap, float *X,
                                          std::int64_t incx);
extern cublasStatus_t (*p_cublasDtpmv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag, std::int64_t n,
                                          const double *Ap, double *X,
                                          std::int64_t incx);
extern cublasStatus_t (*p_cublasCtpmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, const std::complex<float> *Ap,
    std::complex<float> *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasZtpmv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, const std::complex<double> *Ap,
    std::complex<double> *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasStpsv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag, std::int64_t n,
                                          const float *Ap, float *X,
                                          std::int64_t incx);
extern cublasStatus_t (*p_cublasDtpsv_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo,
                                          cublasOperation_t trans,
                                          cublasDiagType_t diag, std::int64_t n,
                                          const double *Ap, double *X,
                                          std::int64_t incx);
extern cublasStatus_t (*p_cublasCtpsv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, const std::complex<float> *Ap,
    std::complex<float> *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasZtpsv_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    cublasDiagType_t diag, std::int64_t n, const std::complex<double> *Ap,
    std::complex<double> *X, std::int64_t incx);
extern cublasStatus_t (*p_cublasSspr_v2)(cublasHandle_t handle,
                                         cublasFillMode_t uplo, std::int64_t n,
                                         const float *alpha, const float *X,
                                         std::int64_t incx, float *Ap);
extern cublasStatus_t (*p_cublasDspr_v2)(cublasHandle_t handle,
                                         cublasFillMode_t uplo, std::int64_t n,
                                         const double *alpha, const double *X,
                                         std::int64_t incx, double *Ap);
extern cublasStatus_t (*p_cublasSspr2_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, std::int64_t n,
                                          const float *alpha, const float *X,
                                          std::int64_t incx, const float *Y,
                                          std::int64_t incy, float *Ap);
extern cublasStatus_t (*p_cublasDspr2_v2)(cublasHandle_t handle,
                                          cublasFillMode_t uplo, std::int64_t n,
                                          const double *alpha, const double *X,
                                          std::int64_t incx, const double *Y,
                                          std::int64_t incy, double *Ap);

//Level 3

extern cublasStatus_t (*p_cublasSgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float *alpha, const float *A, int lda,
    const float *B, int ldb, const float *beta, float *C, int ldc);
extern cublasStatus_t (*p_cublasDgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double *alpha, const double *A, int lda,
    const double *B, int ldb, const double *beta, double *C, int ldc);
extern cublasStatus_t (*p_cublasCgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda,
    const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc);
extern cublasStatus_t (*p_cublasZgemm_v2)(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A,
    int lda, const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc);

extern cublasStatus_t (*p_cublasSsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, int, int,
                                          const float *, const float *, int,
                                          const float *, int, const float *,
                                          float *, int);
extern cublasStatus_t (*p_cublasDsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, int, int,
                                          const double *, const double *, int,
                                          const double *, int, const double *,
                                          double *, int);
extern cublasStatus_t (*p_cublasCsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, int, int,
                                          const cuComplex *, const cuComplex *,
                                          int, const cuComplex *, int,
                                          const cuComplex *, cuComplex *, int);
extern cublasStatus_t (*p_cublasZsymm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, int, int,
                                          const cuDoubleComplex *,
                                          const cuDoubleComplex *, int,
                                          const cuDoubleComplex *, int,
                                          const cuDoubleComplex *,
                                          cuDoubleComplex *, int);

extern cublasStatus_t (*p_cublasSsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                          cublasOperation_t, int, int,
                                          const float *, const float *, int,
                                          const float *, float *, int);
extern cublasStatus_t (*p_cublasDsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                          cublasOperation_t, int, int,
                                          const double *, const double *, int,
                                          const double *, double *, int);
extern cublasStatus_t (*p_cublasCsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                          cublasOperation_t, int, int,
                                          const cuComplex *, const cuComplex *,
                                          int, const cuComplex *, cuComplex *,
                                          int);
extern cublasStatus_t (*p_cublasZsyrk_v2)(cublasHandle_t, cublasFillMode_t,
                                          cublasOperation_t, int, int,
                                          const cuDoubleComplex *,
                                          const cuDoubleComplex *, int,
                                          const cuDoubleComplex *,
                                          cuDoubleComplex *, int);

extern cublasStatus_t (*p_cublasSsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                           cublasOperation_t, int, int,
                                           const float *, const float *, int,
                                           const float *, int, const float *,
                                           float *, int);
extern cublasStatus_t (*p_cublasDsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                           cublasOperation_t, int, int,
                                           const double *, const double *, int,
                                           const double *, int, const double *,
                                           double *, int);
extern cublasStatus_t (*p_cublasCsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                           cublasOperation_t, int, int,
                                           const cuComplex *, const cuComplex *,
                                           int, const cuComplex *, int,
                                           const cuComplex *, cuComplex *, int);
extern cublasStatus_t (*p_cublasZsyr2k_v2)(cublasHandle_t, cublasFillMode_t,
                                           cublasOperation_t, int, int,
                                           const cuDoubleComplex *,
                                           const cuDoubleComplex *, int,
                                           const cuDoubleComplex *, int,
                                           const cuDoubleComplex *,
                                           cuDoubleComplex *, int);

extern cublasStatus_t (*p_cublasStrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int, int,
                                          const float *, const float *, int,
                                          float *, int, float *, int);
extern cublasStatus_t (*p_cublasDtrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int, int,
                                          const double *, const double *, int,
                                          double *, int, double *, int);
extern cublasStatus_t (*p_cublasCtrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int, int,
                                          const cuComplex *, const cuComplex *,
                                          int, cuComplex *, int, cuComplex *,
                                          int);
extern cublasStatus_t (*p_cublasZtrmm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int, int,
                                          const cuDoubleComplex *,
                                          const cuDoubleComplex *, int,
                                          cuDoubleComplex *, int,
                                          cuDoubleComplex *, int);
extern cublasStatus_t (*p_cublasStrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int, int,
                                          const float *, const float *, int,
                                          float *, int);
extern cublasStatus_t (*p_cublasDtrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int, int,
                                          const double *, const double *, int,
                                          double *, int);
extern cublasStatus_t (*p_cublasCtrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int, int,
                                          const cuComplex *, const cuComplex *,
                                          int, cuComplex *, int);
extern cublasStatus_t (*p_cublasZtrsm_v2)(cublasHandle_t, cublasSideMode_t,
                                          cublasFillMode_t, cublasOperation_t,
                                          cublasDiagType_t, int, int,
                                          const cuDoubleComplex *,
                                          const cuDoubleComplex *, int,
                                          cuDoubleComplex *, int);
extern cublasStatus_t (*p_cublasChemm_v2)(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m,
    int n, const cuComplex *alpha, const cuComplex *A, int lda,
    const cuComplex *B, int ldb, const cuComplex *beta, cuComplex *C, int ldc);
extern cublasStatus_t (*p_cublasZhemm_v2)(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, int m,
    int n, const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb, const cuDoubleComplex *beta,
    cuDoubleComplex *C, int ldc);
extern cublasStatus_t (*p_cublasCherk_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k, const float *alpha, const cuComplex *A, int lda,
    const float *beta, cuComplex *C, int ldc);
extern cublasStatus_t (*p_cublasZherk_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k, const double *alpha, const cuDoubleComplex *A, int lda,
    const double *beta, cuDoubleComplex *C, int ldc);
extern cublasStatus_t (*p_cublasCher2k_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k, const cuComplex *alpha, const cuComplex *A, int lda,
    const cuComplex *B, int ldb, const float *beta, cuComplex *C, int ldc);
extern cublasStatus_t (*p_cublasZher2k_v2)(
    cublasHandle_t handle, cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A,
    int lda, const cuDoubleComplex *B, int ldb, const double *beta,
    cuDoubleComplex *C, int ldc);

//==========Openblas=======

//Level 1

extern void (*p_cblas_scopy)(int, const float *, int, float *, int);
extern void (*p_cblas_dcopy)(int, const double *, int, double *, int);
extern void (*p_cblas_sscal)(int, float, float *, int);
extern void (*p_cblas_dscal)(int, double, double *, int);
extern void (*p_cblas_saxpy)(int, float, const float *, int, float *, int);
extern void (*p_cblas_daxpy)(int, double, const double *, int, double *, int);
extern int (*p_cblas_isamax)(int, const float *, int);
extern int (*p_cblas_idamax)(int, const double *, int);
extern void (*p_cblas_sswap)(int, float *, int, float *, int);
extern void (*p_cblas_dswap)(int, double *, int, double *, int);
extern float (*p_cblas_sdot)(int, const float *, int, const float *, int);
extern double (*p_cblas_ddot)(int, const double *, int, const double *, int);
extern float (*p_cblas_snrm2)(int, const float *, int);
extern double (*p_cblas_dnrm2)(int, const double *, int);
extern float (*p_cblas_sasum)(int, const float *, int);
extern double (*p_cblas_dasum)(int, const double *, int);
extern void (*p_cblas_srot)(int, float *, int, float *, int, float, float);
extern void (*p_cblas_drot)(int, double *, int, double *, int, double, double);
extern void (*p_cblas_srotg)(float *, float *, float *, float *);
extern void (*p_cblas_drotg)(double *, double *, double *, double *);
extern void (*p_cblas_srotm)(int, float *, int, float *, int, const float *);
extern void (*p_cblas_drotm)(int, double *, int, double *, int, const double *);
extern void (*p_cblas_srotmg)(float *, float *, float *, float, const float *);
extern void (*p_cblas_drotmg)(double *, double *, double *, double,
                              const double *);

extern void (*p_cblas_cdotu_sub)(int, const std::complex<float> *, int,
                                 const std::complex<float> *, int,
                                 std::complex<float> *);
extern void (*p_cblas_cdotc_sub)(int, const std::complex<float> *, int,
                                 const std::complex<float> *, int,
                                 std::complex<float> *);
extern void (*p_cblas_zdotu_sub)(int, const std::complex<double> *, int,
                                 const std::complex<double> *, int,
                                 std::complex<double> *);
extern void (*p_cblas_zdotc_sub)(int, const std::complex<double> *, int,
                                 const std::complex<double> *, int,
                                 std::complex<double> *);
extern void (*p_cblas_cswap)(int, std::complex<float> *, int,
                             std::complex<float> *, int);
extern void (*p_cblas_zswap)(int, std::complex<double> *, int,
                             std::complex<double> *, int);
extern void (*p_cblas_ccopy)(int, const std::complex<float> *, int,
                             std::complex<float> *, int);
extern void (*p_cblas_zcopy)(int, const std::complex<double> *, int,
                             std::complex<double> *, int);
extern void (*p_cblas_cscal)(int, const std::complex<float> *,
                             std::complex<float> *, int);
extern void (*p_cblas_zscal)(int, const std::complex<double> *,
                             std::complex<double> *, int);
extern void (*p_cblas_csscal)(int, float, std::complex<float> *, int);
extern void (*p_cblas_zdscal)(int, double, std::complex<double> *, int);
extern void (*p_cblas_caxpy)(int, const std::complex<float> *,
                             const std::complex<float> *, int,
                             std::complex<float> *, int);
extern void (*p_cblas_zaxpy)(int, const std::complex<double> *,
                             const std::complex<double> *, int,
                             std::complex<double> *, int);
extern float (*p_cblas_scnrm2)(int, const std::complex<float> *, int);
extern float (*p_cblas_scasum)(int, const std::complex<float> *, int);
extern double (*p_cblas_dznrm2)(int, const std::complex<double> *, int);
extern double (*p_cblas_dzasum)(int, const std::complex<double> *, int);
extern int (*p_cblas_icamax)(int, const std::complex<float> *, int);
extern int (*p_cblas_izamax)(int, const std::complex<double> *, int);

//Level 2

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
extern void (*p_cblas_cgemv)(const CBLAS_ORDER Order,
                             const CBLAS_TRANSPOSE Trans, std::int64_t m,
                             std::int64_t n, const std::complex<float> *alpha,
                             const std::complex<float> *A, std::int64_t lda,
                             const std::complex<float> *x, std::int64_t incx,
                             const std::complex<float> *beta,
                             std::complex<float> *y, std::int64_t incy);
extern void (*p_cblas_zgemv)(const CBLAS_ORDER Order,
                             const CBLAS_TRANSPOSE Trans, std::int64_t m,
                             std::int64_t n, const std::complex<double> *alpha,
                             const std::complex<double> *A, std::int64_t lda,
                             const std::complex<double> *x, std::int64_t incx,
                             const std::complex<double> *beta,
                             std::complex<double> *y, std::int64_t incy);
extern void (*p_cblas_ssymv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             const int N, const float alpha, const float *A,
                             const int lda, const float *X, const int incX,
                             const float beta, float *Y, const int incY);
extern void (*p_cblas_dsymv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             const int N, const double alpha, const double *A,
                             const int lda, const double *X, const int incX,
                             const double beta, double *Y, const int incY);
extern void (*p_cblas_chemv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const std::complex<float> *alpha,
                             const std::complex<float> *A, std::int64_t lda,
                             const std::complex<float> *x, std::int64_t incx,
                             const std::complex<float> *beta,
                             std::complex<float> *y, std::int64_t incy);
extern void (*p_cblas_zhemv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const std::complex<double> *alpha,
                             const std::complex<double> *A, std::int64_t lda,
                             const std::complex<double> *x, std::int64_t incx,
                             const std::complex<double> *beta,
                             std::complex<double> *y, std::int64_t incy);
extern void (*p_cblas_strmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const float *A, std::int64_t lda,
                             float *X, std::int64_t incx);
extern void (*p_cblas_dtrmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const double *A, std::int64_t lda,
                             double *X, std::int64_t incx);
extern void (*p_cblas_ctrmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const std::complex<float> *A,
                             std::int64_t lda, std::complex<float> *X,
                             std::int64_t incx);
extern void (*p_cblas_ztrmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const std::complex<double> *A,
                             std::int64_t lda, std::complex<double> *X,
                             std::int64_t incx);
extern void (*p_cblas_strsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const float *A, std::int64_t lda,
                             float *X, std::int64_t incX);
extern void (*p_cblas_dtrsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const double *A, std::int64_t lda,
                             double *X, std::int64_t incx);
extern void (*p_cblas_ctrsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const std::complex<float> *A,
                             std::int64_t lda, std::complex<float> *X,
                             std::int64_t incX);
extern void (*p_cblas_ztrsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const std::complex<double> *A,
                             std::int64_t lda, std::complex<double> *X,
                             std::int64_t incX);
extern void (*p_cblas_cgerc)(const CBLAS_ORDER Order, std::int64_t m,
                             std::int64_t n, const std::complex<float> *alpha,
                             const std::complex<float> *X, std::int64_t incx,
                             const std::complex<float> *Y, std::int64_t incy,
                             std::complex<float> *A, std::int64_t lda);
extern void (*p_cblas_zgerc)(const CBLAS_ORDER Order, std::int64_t m,
                             std::int64_t n, const std::complex<double> *alpha,
                             const std::complex<double> *X, std::int64_t incx,
                             const std::complex<double> *Y, std::int64_t incy,
                             std::complex<double> *A, std::int64_t lda);
extern void (*p_cblas_sger)(const CBLAS_ORDER Order, int m, int n,
                            const float alpha, const float *X, int incx,
                            const float *Y, int incy, float *A, int lda);
extern void (*p_cblas_dger)(const CBLAS_ORDER Order, std::int64_t m,
                            std::int64_t n, const double alpha, const double *X,
                            std::int64_t incx, const double *Y,
                            std::int64_t incy, double *A, std::int64_t lda);
extern void (*p_cblas_cgeru)(const CBLAS_ORDER Order, std::int64_t m,
                             std::int64_t n, const std::complex<float> *alpha,
                             const std::complex<float> *X, std::int64_t incx,
                             const std::complex<float> *Y, std::int64_t incy,
                             std::complex<float> *A, std::int64_t lda);
extern void (*p_cblas_zgeru)(const CBLAS_ORDER Order, std::int64_t m,
                             std::int64_t n, const std::complex<double> *alpha,
                             const std::complex<double> *X, std::int64_t incx,
                             const std::complex<double> *Y, std::int64_t incy,
                             std::complex<double> *A, std::int64_t lda);
extern void (*p_cblas_ssyr)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                            std::int64_t n, const float alpha, const float *X,
                            std::int64_t incX, float *A, std::int64_t lda);
extern void (*p_cblas_dsyr)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                            std::int64_t n, const double alpha, const double *X,
                            std::int64_t incX, double *A, std::int64_t lda);
extern void (*p_cblas_ssyr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             int n, const float alpha, const float *X, int incX,
                             const float *Y, int incY, float *A, int lda);
extern void (*p_cblas_dsyr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             int n, const double alpha, const double *X,
                             int incX, const double *Y, int incY, double *A,
                             int lda);
extern void (*p_cblas_cher)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                            std::int64_t n, const float alpha,
                            const std::complex<float> *x, std::int64_t incx,
                            std::complex<float> *A, std::int64_t lda);
extern void (*p_cblas_zher)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                            std::int64_t n, const double alpha,
                            const std::complex<double> *x, std::int64_t incx,
                            std::complex<double> *A, std::int64_t lda);
extern void (*p_cblas_cher2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const std::complex<float> *alpha,
                             const std::complex<float> *x, std::int64_t incx,
                             const std::complex<float> *y, std::int64_t incy,
                             std::complex<float> *A, std::int64_t lda);
extern void (*p_cblas_zher2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const std::complex<double> *alpha,
                             const std::complex<double> *x, std::int64_t incx,
                             const std::complex<double> *y, std::int64_t incy,
                             std::complex<double> *A, std::int64_t lda);
extern void (*p_cblas_chpr)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                            std::int64_t n, const float alpha,
                            const std::complex<float> *X, std::int64_t incx,
                            std::complex<float> *ap);
extern void (*p_cblas_zhpr)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                            std::int64_t n, const double alpha,
                            const std::complex<double> *X, std::int64_t incx,
                            std::complex<double> *ap);
extern void (*p_cblas_chpr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const std::complex<float> *alpha,
                             const std::complex<float> *X, std::int64_t incx,
                             const std::complex<float> *Y, std::int64_t incy,
                             std::complex<float> *ap);
extern void (*p_cblas_zhpr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const std::complex<double> *alpha,
                             const std::complex<double> *X, std::int64_t incx,
                             const std::complex<double> *Y, std::int64_t incy,
                             std::complex<double> *ap);
extern void (*p_cblas_sgbmv)(const CBLAS_ORDER Order, CBLAS_TRANSPOSE trans,
                             std::int64_t m, std::int64_t n, std::int64_t KL,
                             std::int64_t KU, const float alpha, const float *A,
                             std::int64_t lda, const float *X,
                             std::int64_t incX, const float beta, float *Y,
                             std::int64_t incY);
extern void (*p_cblas_dgbmv)(const CBLAS_ORDER Order, CBLAS_TRANSPOSE trans,
                             std::int64_t m, std::int64_t n, std::int64_t KL,
                             std::int64_t KU, const double alpha,
                             const double *A, std::int64_t lda, const double *X,
                             std::int64_t incX, const double beta, double *Y,
                             std::int64_t incY);
extern void (*p_cblas_cgbmv)(const CBLAS_ORDER Order, CBLAS_TRANSPOSE trans,
                             std::int64_t m, std::int64_t n, std::int64_t KL,
                             std::int64_t KU, const std::complex<float> *alpha,
                             const std::complex<float> *A, std::int64_t lda,
                             const std::complex<float> *X, std::int64_t incX,
                             const std::complex<float> *beta,
                             std::complex<float> *Y, std::int64_t incY);
extern void (*p_cblas_zgbmv)(const CBLAS_ORDER Order, CBLAS_TRANSPOSE trans,
                             std::int64_t m, std::int64_t n, std::int64_t KL,
                             std::int64_t KU, const std::complex<double> *alpha,
                             const std::complex<double> *A, std::int64_t lda,
                             const std::complex<double> *X, std::int64_t incX,
                             const std::complex<double> *beta,
                             std::complex<double> *Y, std::int64_t incY);
extern void (*p_cblas_chbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, std::int64_t k,
                             const std::complex<float> *alpha,
                             const std::complex<float> *A, std::int64_t lda,
                             const std::complex<float> *X, std::int64_t incx,
                             const std::complex<float> *beta,
                             std::complex<float> *Y, std::int64_t incy);
extern void (*p_cblas_zhbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, std::int64_t k,
                             const std::complex<double> *alpha,
                             const std::complex<double> *A, std::int64_t lda,
                             const std::complex<double> *X, std::int64_t incx,
                             const std::complex<double> *beta,
                             std::complex<double> *Y, std::int64_t incy);

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

extern void (*p_cblas_stbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, std::int64_t K, const float *A,
                             std::int64_t lda, float *X, std::int64_t incX);

extern void (*p_cblas_dtbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, std::int64_t K, const double *A,
                             std::int64_t lda, double *X, std::int64_t incX);

extern void (*p_cblas_ctbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, std::int64_t k,
                             const std::complex<float> *A, std::int64_t lda,
                             std::complex<float> *X, std::int64_t incx);

extern void (*p_cblas_ztbmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, std::int64_t k,
                             const std::complex<double> *A, std::int64_t lda,
                             std::complex<double> *X, std::int64_t incx);

extern void (*p_cblas_stbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, std::int64_t K, const float *A,
                             std::int64_t lda, float *X, std::int64_t incX);

extern void (*p_cblas_dtbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, std::int64_t K, const double *A,
                             std::int64_t lda, double *X, std::int64_t incX);

extern void (*p_cblas_ctbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, std::int64_t K,
                             const std::complex<float> *A, std::int64_t lda,
                             std::complex<float> *X, std::int64_t incx);

extern void (*p_cblas_ztbsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, std::int64_t K,
                             const std::complex<double> *A, std::int64_t lda,
                             std::complex<double> *X, std::int64_t incx);

extern void (*p_cblas_chpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const std::complex<float> *alpha,
                             const std::complex<float> *Ap,
                             const std::complex<float> *X, std::int64_t incx,
                             const std::complex<float> *beta,
                             std::complex<float> *Y, std::int64_t incy);

extern void (*p_cblas_zhpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const std::complex<double> *alpha,
                             const std::complex<double> *Ap,
                             const std::complex<double> *X, std::int64_t incx,
                             const std::complex<double> *beta,
                             std::complex<double> *Y, std::int64_t incy);

extern void (*p_cblas_sspmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const float alpha, const float *Ap,
                             const float *x, std::int64_t incX,
                             const float beta, float *y, std::int64_t incY);

extern void (*p_cblas_dspmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const double alpha,
                             const double *Ap, const double *x,
                             std::int64_t incX, const double beta, double *y,
                             std::int64_t incY);

extern void (*p_cblas_stpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const float *Ap, float *X,
                             std::int64_t incX);

extern void (*p_cblas_dtpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const double *Ap, double *X,
                             std::int64_t incX);

extern void (*p_cblas_ctpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const std::complex<float> *Ap,
                             std::complex<float> *X, std::int64_t incx);

extern void (*p_cblas_ztpmv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const std::complex<double> *Ap,
                             std::complex<double> *X, std::int64_t incx);

extern void (*p_cblas_stpsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const float *Ap, float *X,
                             std::int64_t incX);

extern void (*p_cblas_dtpsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const double *Ap, double *X,
                             std::int64_t incX);

extern void (*p_cblas_ctpsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const std::complex<float> *Ap,
                             std::complex<float> *X, std::int64_t incX);

extern void (*p_cblas_ztpsv)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
                             std::int64_t n, const std::complex<double> *Ap,
                             std::complex<double> *X, std::int64_t incX);

extern void (*p_cblas_sspr)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                            std::int64_t n, const float alpha, const float *X,
                            std::int64_t incX, float *Ap);

extern void (*p_cblas_dspr)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                            std::int64_t n, const double alpha, const double *X,
                            std::int64_t incX, double *Ap);

extern void (*p_cblas_sspr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const float alpha, const float *X,
                             std::int64_t incX, const float *Y,
                             std::int64_t incY, float *A);

extern void (*p_cblas_dspr2)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             std::int64_t n, const double alpha,
                             const double *X, std::int64_t incX,
                             const double *Y, std::int64_t incY, double *A);

//Level 3

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
extern void (*p_cblas_ssyr2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                              const CBLAS_TRANSPOSE Trans, std::int64_t N,
                              std::int64_t K, const float alpha, const float *A,
                              std::int64_t lda, const float *B,
                              std::int64_t ldb, const float beta, float *C,
                              std::int64_t ldc);
extern void (*p_cblas_dsyr2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                              const CBLAS_TRANSPOSE Trans, std::int64_t N,
                              std::int64_t K, const double alpha,
                              const double *A, std::int64_t lda,
                              const double *B, std::int64_t ldb,
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

extern void (*p_cblas_cgemm)(const CBLAS_ORDER Order,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_TRANSPOSE TransB, std::int64_t M,
                             std::int64_t N, std::int64_t K, const void *alpha,
                             const void *A, std::int64_t lda, const void *B,
                             std::int64_t ldb, const void *beta, void *C,
                             std::int64_t ldc);
extern void (*p_cblas_zgemm)(const CBLAS_ORDER Order,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_TRANSPOSE TransB, std::int64_t M,
                             std::int64_t N, std::int64_t K, const void *alpha,
                             const void *A, std::int64_t lda, const void *B,
                             std::int64_t ldb, const void *beta, void *C,
                             std::int64_t ldc);
extern void (*p_cblas_csymm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo, std::int64_t M,
                             std::int64_t N, const void *alpha, const void *A,
                             std::int64_t lda, const void *B, std::int64_t ldb,
                             const void *beta, void *C, std::int64_t ldc);
extern void (*p_cblas_zsymm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo, std::int64_t M,
                             std::int64_t N, const void *alpha, const void *A,
                             std::int64_t lda, const void *B, std::int64_t ldb,
                             const void *beta, void *C, std::int64_t ldc);
extern void (*p_cblas_csyrk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE Trans, std::int64_t N,
                             std::int64_t K, const void *alpha, const void *A,
                             std::int64_t lda, const void *beta, void *C,
                             std::int64_t ldc);
extern void (*p_cblas_zsyrk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE Trans, std::int64_t N,
                             std::int64_t K, const void *alpha, const void *A,
                             std::int64_t lda, const void *beta, void *C,
                             std::int64_t ldc);
extern void (*p_cblas_csyr2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                              const CBLAS_TRANSPOSE Trans, std::int64_t N,
                              std::int64_t K, const void *alpha, const void *A,
                              std::int64_t lda, const void *B, std::int64_t ldb,
                              const void *beta, void *C, std::int64_t ldc);
extern void (*p_cblas_zsyr2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                              const CBLAS_TRANSPOSE Trans, std::int64_t N,
                              std::int64_t K, const void *alpha, const void *A,
                              std::int64_t lda, const void *B, std::int64_t ldb,
                              const void *beta, void *C, std::int64_t ldc);
extern void (*p_cblas_ctrmm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_DIAG Diag, std::int64_t M,
                             std::int64_t N, const void *alpha, const void *A,
                             std::int64_t lda, void *B, std::int64_t ldb);
extern void (*p_cblas_ztrmm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_DIAG Diag, std::int64_t M,
                             std::int64_t N, const void *alpha, const void *A,
                             std::int64_t lda, void *B, std::int64_t ldb);
extern void (*p_cblas_ctrsm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_DIAG Diag, std::int64_t M,
                             std::int64_t N, const void *alpha, const void *A,
                             std::int64_t lda, void *B, std::int64_t ldb);
extern void (*p_cblas_ztrsm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE TransA,
                             const CBLAS_DIAG Diag, std::int64_t M,
                             std::int64_t N, const void *alpha, const void *A,
                             std::int64_t lda, void *B, std::int64_t ldb);
extern void (*p_cblas_chemm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo, std::int64_t M,
                             std::int64_t N, const void *alpha, const void *A,
                             std::int64_t lda, const void *B, std::int64_t ldb,
                             const void *beta, void *C, std::int64_t ldc);
extern void (*p_cblas_zhemm)(const CBLAS_ORDER Order, const CBLAS_SIDE Side,
                             const CBLAS_UPLO Uplo, std::int64_t M,
                             std::int64_t N, const void *alpha, const void *A,
                             std::int64_t lda, const void *B, std::int64_t ldb,
                             const void *beta, void *C, std::int64_t ldc);
extern void (*p_cblas_cherk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE Trans, std::int64_t N,
                             std::int64_t K, const float alpha, const void *A,
                             std::int64_t lda, const float beta, void *C,
                             std::int64_t ldc);
extern void (*p_cblas_zherk)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                             const CBLAS_TRANSPOSE Trans, std::int64_t N,
                             std::int64_t K, const double alpha, const void *A,
                             std::int64_t lda, const double beta, void *C,
                             std::int64_t ldc);
extern void (*p_cblas_cher2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                              const CBLAS_TRANSPOSE Trans, std::int64_t N,
                              std::int64_t K, const void *alpha, const void *A,
                              std::int64_t lda, const void *B, std::int64_t ldb,
                              const float beta, void *C, std::int64_t ldc);
extern void (*p_cblas_zher2k)(const CBLAS_ORDER Order, const CBLAS_UPLO Uplo,
                              const CBLAS_TRANSPOSE Trans, std::int64_t N,
                              std::int64_t K, const void *alpha, const void *A,
                              std::int64_t lda, const void *B, std::int64_t ldb,
                              const double beta, void *C, std::int64_t ldc);

//=============Rocblas=========

extern hipError_t (*p_hipMalloc)(void **, size_t);
extern hipError_t (*p_hipFree)(void *);
extern hipError_t (*p_hipMemcpy)(void *, const void *, size_t, hipMemcpyKind);

extern rocblas_status (*p_rocblas_create_handle)(rocblas_handle *handle);
extern rocblas_status (*p_rocblas_destroy_handle)(rocblas_handle handle);

//Level 1

extern rocblas_status (*p_rocblas_isamax)(rocblas_handle, rocblas_int,
                                          const float *, rocblas_int,
                                          rocblas_int *);
extern rocblas_status (*p_rocblas_idamax)(rocblas_handle, rocblas_int,
                                          const double *, rocblas_int,
                                          rocblas_int *);
extern rocblas_status (*p_rocblas_sasum)(rocblas_handle, rocblas_int,
                                         const float *, rocblas_int, float *);
extern rocblas_status (*p_rocblas_dasum)(rocblas_handle, rocblas_int,
                                         const double *, rocblas_int, double *);
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
extern rocblas_status (*p_rocblas_snrm2)(rocblas_handle, rocblas_int,
                                         const float *, rocblas_int, float *);
extern rocblas_status (*p_rocblas_dnrm2)(rocblas_handle, rocblas_int,
                                         const double *, rocblas_int, double *);
extern rocblas_status (*p_rocblas_srot)(rocblas_handle, rocblas_int, float *,
                                        rocblas_int, float *, rocblas_int,
                                        const float *, const float *);
extern rocblas_status (*p_rocblas_drot)(rocblas_handle, rocblas_int, double *,
                                        rocblas_int, double *, rocblas_int,
                                        const double *, const double *);
extern rocblas_status (*p_rocblas_srotg)(rocblas_handle, float *, float *,
                                         float *, float *);
extern rocblas_status (*p_rocblas_drotg)(rocblas_handle, double *, double *,
                                         double *, double *);
extern rocblas_status (*p_rocblas_srotm)(rocblas_handle, rocblas_int, float *,
                                         rocblas_int, float *, rocblas_int,
                                         const float *);
extern rocblas_status (*p_rocblas_drotm)(rocblas_handle, rocblas_int, double *,
                                         rocblas_int, double *, rocblas_int,
                                         const double *);
extern rocblas_status (*p_rocblas_srotmg)(rocblas_handle, float *, float *,
                                          float *, const float *, float *);
extern rocblas_status (*p_rocblas_drotmg)(rocblas_handle, double *, double *,
                                          double *, const double *, double *);
extern rocblas_status (*p_rocblas_sscal)(rocblas_handle, rocblas_int,
                                         const float *, float *, rocblas_int);
extern rocblas_status (*p_rocblas_dscal)(rocblas_handle, rocblas_int,
                                         const double *, double *, rocblas_int);
extern rocblas_status (*p_rocblas_sswap)(rocblas_handle, rocblas_int, float *,
                                         rocblas_int, float *, rocblas_int);
extern rocblas_status (*p_rocblas_dswap)(rocblas_handle, rocblas_int, double *,
                                         rocblas_int, double *, rocblas_int);

extern rocblas_status (*p_rocblas_icamax)(rocblas_handle, rocblas_int,
                                          const rocblas_float_complex *,
                                          rocblas_int, rocblas_int *);
extern rocblas_status (*p_rocblas_izamax)(rocblas_handle, rocblas_int,
                                          const rocblas_double_complex *,
                                          rocblas_int, rocblas_int *);
extern rocblas_status (*p_rocblas_scasum)(rocblas_handle, rocblas_int,
                                          const rocblas_float_complex *,
                                          rocblas_int, float *);
extern rocblas_status (*p_rocblas_dzasum)(rocblas_handle, rocblas_int,
                                          const rocblas_double_complex *,
                                          rocblas_int, double *);
extern rocblas_status (*p_rocblas_caxpy)(rocblas_handle, rocblas_int,
                                         const rocblas_float_complex *,
                                         const rocblas_float_complex *,
                                         rocblas_int, rocblas_float_complex *,
                                         rocblas_int);
extern rocblas_status (*p_rocblas_zaxpy)(rocblas_handle, rocblas_int,
                                         const rocblas_double_complex *,
                                         const rocblas_double_complex *,
                                         rocblas_int, rocblas_double_complex *,
                                         rocblas_int);
extern rocblas_status (*p_rocblas_ccopy)(rocblas_handle, rocblas_int,
                                         const rocblas_float_complex *,
                                         rocblas_int, rocblas_float_complex *,
                                         rocblas_int);
extern rocblas_status (*p_rocblas_zcopy)(rocblas_handle, rocblas_int,
                                         const rocblas_double_complex *,
                                         rocblas_int, rocblas_double_complex *,
                                         rocblas_int);
extern rocblas_status (*p_rocblas_cdotu)(rocblas_handle, rocblas_int,
                                         const rocblas_float_complex *,
                                         rocblas_int,
                                         const rocblas_float_complex *,
                                         rocblas_int, rocblas_float_complex *);
extern rocblas_status (*p_rocblas_cdotc)(rocblas_handle, rocblas_int,
                                         const rocblas_float_complex *,
                                         rocblas_int,
                                         const rocblas_float_complex *,
                                         rocblas_int, rocblas_float_complex *);
extern rocblas_status (*p_rocblas_zdotu)(rocblas_handle, rocblas_int,
                                         const rocblas_double_complex *,
                                         rocblas_int,
                                         const rocblas_double_complex *,
                                         rocblas_int, rocblas_double_complex *);
extern rocblas_status (*p_rocblas_zdotc)(rocblas_handle, rocblas_int,
                                         const rocblas_double_complex *,
                                         rocblas_int,
                                         const rocblas_double_complex *,
                                         rocblas_int, rocblas_double_complex *);
extern rocblas_status (*p_rocblas_scnrm2)(rocblas_handle, rocblas_int,
                                          const rocblas_float_complex *,
                                          rocblas_int, float *);
extern rocblas_status (*p_rocblas_dznrm2)(rocblas_handle, rocblas_int,
                                          const rocblas_double_complex *,
                                          rocblas_int, double *);
extern rocblas_status (*p_rocblas_cscal)(rocblas_handle, rocblas_int,
                                         const rocblas_float_complex *,
                                         rocblas_float_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_zscal)(rocblas_handle, rocblas_int,
                                         const rocblas_double_complex *,
                                         rocblas_double_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_csscal)(rocblas_handle, rocblas_int,
                                          const float *,
                                          rocblas_float_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_zdscal)(rocblas_handle, rocblas_int,
                                          const double *,
                                          rocblas_double_complex *,
                                          rocblas_int);
extern rocblas_status (*p_rocblas_cswap)(rocblas_handle, rocblas_int,
                                         rocblas_float_complex *, rocblas_int,
                                         rocblas_float_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_zswap)(rocblas_handle, rocblas_int,
                                         rocblas_double_complex *, rocblas_int,
                                         rocblas_double_complex *, rocblas_int);

//Level 2

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

extern rocblas_status (*p_rocblas_cgemv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, const std::complex<float> *alpha,
    const std::complex<float> *A, std::int64_t lda,
    const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y, std::int64_t incy);

extern rocblas_status (*p_rocblas_zgemv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, const std::complex<double> *alpha,
    const std::complex<double> *A, std::int64_t lda,
    const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy);

extern rocblas_status (*p_rocblas_ssymv)(rocblas_handle handle,
                                         rocblas_fill uplo, int n,
                                         const float *alpha, const float *A,
                                         int lda, const float *X, int incx,
                                         const float *beta, float *Y, int incy);

extern rocblas_status (*p_rocblas_dsymv)(rocblas_handle handle,
                                         rocblas_fill uplo, int n,
                                         const double *alpha, const double *A,
                                         int lda, const double *X, int incx,
                                         const double *beta, double *Y,
                                         int incy);

extern rocblas_status (*p_rocblas_chemv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *A,
    std::int64_t lda, const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y, std::int64_t incy);

extern rocblas_status (*p_rocblas_zhemv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy);

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

extern rocblas_status (*p_rocblas_ctrmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, const std::complex<float> *A,
    std::int64_t lda, std::complex<float> *X, std::int64_t incx);

extern rocblas_status (*p_rocblas_ztrmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, const std::complex<double> *A,
    std::int64_t lda, std::complex<double> *X, std::int64_t incx);

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

extern rocblas_status (*p_rocblas_ctrsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, const std::complex<float> *A,
    std::int64_t lda, std::complex<float> *X, std::int64_t incx);

extern rocblas_status (*p_rocblas_ztrsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, const std::complex<double> *A,
    std::int64_t lda, std::complex<double> *X, std::int64_t incx);

extern rocblas_status (*p_rocblas_chpmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *Ap,
    const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y, std::int64_t incy);

extern rocblas_status (*p_rocblas_zhpmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *Ap,
    const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy);

extern rocblas_status (*p_rocblas_ssbmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n, std::int64_t k,
    const float *alpha, const float *A, std::int64_t lda, const float *X,
    std::int64_t incx, const float *beta, float *Y, std::int64_t incy);

extern rocblas_status (*p_rocblas_dsbmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n, std::int64_t k,
    const double *alpha, const double *A, std::int64_t lda, const double *X,
    std::int64_t incx, const double *beta, double *Y, std::int64_t incy);

extern rocblas_status (*p_rocblas_stbmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, std::int64_t k, const float *A,
    std::int64_t lda, float *X, std::int64_t incx);

extern rocblas_status (*p_rocblas_dtbmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, std::int64_t k, const double *A,
    std::int64_t lda, double *X, std::int64_t incx);

extern rocblas_status (*p_rocblas_ctbmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, std::int64_t k,
    const std::complex<float> *A, std::int64_t lda, std::complex<float> *X,
    std::int64_t incx);

extern rocblas_status (*p_rocblas_ztbmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, std::int64_t k,
    const std::complex<double> *A, std::int64_t lda, std::complex<double> *X,
    std::int64_t incx);

extern rocblas_status (*p_rocblas_stbsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, std::int64_t k, const float *A,
    std::int64_t lda, float *X, std::int64_t incx);

extern rocblas_status (*p_rocblas_dtbsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, std::int64_t k, const double *A,
    std::int64_t lda, double *X, std::int64_t incx);

extern rocblas_status (*p_rocblas_ctbsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, std::int64_t k,
    const std::complex<float> *A, std::int64_t lda, std::complex<float> *X,
    std::int64_t incx);

extern rocblas_status (*p_rocblas_ztbsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, std::int64_t k,
    const std::complex<double> *A, std::int64_t lda, std::complex<double> *X,
    std::int64_t incx);

extern rocblas_status (*p_rocblas_sger)(rocblas_handle handle, std::int64_t m,
                                        std::int64_t n, const float *alpha,
                                        const float *X, std::int64_t incx,
                                        const float *Y, std::int64_t incy,
                                        float *A, std::int64_t lda);

extern rocblas_status (*p_rocblas_dger)(rocblas_handle handle, std::int64_t m,
                                        std::int64_t n, const double *alpha,
                                        const double *X, std::int64_t incx,
                                        const double *Y, std::int64_t incy,
                                        double *A, std::int64_t lda);

extern rocblas_status (*p_rocblas_cgerc)(
    rocblas_handle handle, std::int64_t m, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda);

extern rocblas_status (*p_rocblas_zgerc)(
    rocblas_handle handle, std::int64_t m, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda);

extern rocblas_status (*p_rocblas_cgeru)(
    rocblas_handle handle, std::int64_t m, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda);

extern rocblas_status (*p_rocblas_zgeru)(
    rocblas_handle handle, std::int64_t m, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda);

extern rocblas_status (*p_rocblas_ssyr)(rocblas_handle handle,
                                        rocblas_fill uplo, std::int64_t n,
                                        const float *alpha, const float *X,
                                        std::int64_t incx, float *A,
                                        std::int64_t lda);

extern rocblas_status (*p_rocblas_dsyr)(rocblas_handle handle,
                                        rocblas_fill uplo, std::int64_t n,
                                        const double *alpha, const double *X,
                                        std::int64_t incx, double *A,
                                        std::int64_t lda);

extern rocblas_status (*p_rocblas_ssyr2)(rocblas_handle handle,
                                         rocblas_fill uplo, std::int64_t n,
                                         const float *alpha, const float *X,
                                         std::int64_t incx, const float *Y,
                                         std::int64_t incy, float *A,
                                         std::int64_t lda);

extern rocblas_status (*p_rocblas_dsyr2)(rocblas_handle handle,
                                         rocblas_fill uplo, std::int64_t n,
                                         const double *alpha, const double *X,
                                         std::int64_t incx, const double *Y,
                                         std::int64_t incy, double *A,
                                         std::int64_t lda);

extern rocblas_status (*p_rocblas_cher)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const float *alpha, const std::complex<float> *X, std::int64_t incx,
    std::complex<float> *A, std::int64_t lda);

extern rocblas_status (*p_rocblas_zher)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const double *alpha, const std::complex<double> *X, std::int64_t incx,
    std::complex<double> *A, std::int64_t lda);

extern rocblas_status (*p_rocblas_cher2)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *A, std::int64_t lda);

extern rocblas_status (*p_rocblas_zher2)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *A, std::int64_t lda);

extern rocblas_status (*p_rocblas_chpr)(rocblas_handle handle,
                                        rocblas_fill uplo, std::int64_t n,
                                        const float *alpha,
                                        const std::complex<float> *X,
                                        std::int64_t incx,
                                        std::complex<float> *AP);

extern rocblas_status (*p_rocblas_zhpr)(rocblas_handle handle,
                                        rocblas_fill uplo, std::int64_t n,
                                        const double *alpha,
                                        const std::complex<double> *X,
                                        std::int64_t incx,
                                        std::complex<double> *AP);

extern rocblas_status (*p_rocblas_chpr2)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<float> *alpha, const std::complex<float> *X,
    std::int64_t incx, const std::complex<float> *Y, std::int64_t incy,
    std::complex<float> *AP);

extern rocblas_status (*p_rocblas_zhpr2)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n,
    const std::complex<double> *alpha, const std::complex<double> *X,
    std::int64_t incx, const std::complex<double> *Y, std::int64_t incy,
    std::complex<double> *AP);

extern rocblas_status (*p_rocblas_sgbmv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU, const float *alpha,
    const float *A, std::int64_t lda, const float *X, std::int64_t incx,
    const float *beta, float *Y, std::int64_t incy);

extern rocblas_status (*p_rocblas_dgbmv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU, const double *alpha,
    const double *A, std::int64_t lda, const double *X, std::int64_t incx,
    const double *beta, double *Y, std::int64_t incy);

extern rocblas_status (*p_rocblas_cgbmv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU,
    const std::complex<float> *alpha, const std::complex<float> *A,
    std::int64_t lda, const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y, std::int64_t incy);

extern rocblas_status (*p_rocblas_zgbmv)(
    rocblas_handle handle, rocblas_operation trans, std::int64_t m,
    std::int64_t n, std::int64_t KL, std::int64_t KU,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy);

extern rocblas_status (*p_rocblas_chbmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n, std::int64_t k,
    const std::complex<float> *alpha, const std::complex<float> *A,
    std::int64_t lda, const std::complex<float> *X, std::int64_t incx,
    const std::complex<float> *beta, std::complex<float> *Y, std::int64_t incy);

extern rocblas_status (*p_rocblas_zhbmv)(
    rocblas_handle handle, rocblas_fill uplo, std::int64_t n, int64_t k,
    const std::complex<double> *alpha, const std::complex<double> *A,
    std::int64_t lda, const std::complex<double> *X, std::int64_t incx,
    const std::complex<double> *beta, std::complex<double> *Y,
    std::int64_t incy);

extern rocblas_status (*p_rocblas_sspmv)(rocblas_handle handle,
                                         rocblas_fill uplo, std::int64_t n,
                                         const float *alpha, const float *Ap,
                                         const float *x, std::int64_t incX,
                                         const float *beta, float *y,
                                         std::int64_t incY);

extern rocblas_status (*p_rocblas_dspmv)(rocblas_handle handle,
                                         rocblas_fill uplo, std::int64_t n,
                                         const double *alpha, const double *Ap,
                                         const double *x, std::int64_t incX,
                                         const double *beta, double *y,
                                         std::int64_t incY);

extern rocblas_status (*p_rocblas_stpmv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation trans,
                                         rocblas_diagonal diag, std::int64_t n,
                                         const float *Ap, float *X,
                                         std::int64_t incx);

extern rocblas_status (*p_rocblas_dtpmv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation trans,
                                         rocblas_diagonal diag, std::int64_t n,
                                         const double *Ap, double *X,
                                         std::int64_t incx);

extern rocblas_status (*p_rocblas_ctpmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, const std::complex<float> *Ap,
    std::complex<float> *X, std::int64_t incx);

extern rocblas_status (*p_rocblas_ztpmv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, const std::complex<double> *Ap,
    std::complex<double> *X, std::int64_t incx);

extern rocblas_status (*p_rocblas_stpsv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation trans,
                                         rocblas_diagonal diag, std::int64_t n,
                                         const float *Ap, float *X,
                                         std::int64_t incx);

extern rocblas_status (*p_rocblas_dtpsv)(rocblas_handle handle,
                                         rocblas_fill uplo,
                                         rocblas_operation trans,
                                         rocblas_diagonal diag, std::int64_t n,
                                         const double *Ap, double *X,
                                         std::int64_t incx);

extern rocblas_status (*p_rocblas_ctpsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, const std::complex<float> *Ap,
    std::complex<float> *X, std::int64_t incx);

extern rocblas_status (*p_rocblas_ztpsv)(
    rocblas_handle handle, rocblas_fill uplo, rocblas_operation trans,
    rocblas_diagonal diag, std::int64_t n, const std::complex<double> *Ap,
    std::complex<double> *X, std::int64_t incx);

extern rocblas_status (*p_rocblas_sspr)(rocblas_handle handle,
                                        rocblas_fill uplo, std::int64_t n,
                                        const float *alpha, const float *X,
                                        std::int64_t incx, float *Ap);

extern rocblas_status (*p_rocblas_dspr)(rocblas_handle handle,
                                        rocblas_fill uplo, std::int64_t n,
                                        const double *alpha, const double *X,
                                        std::int64_t incx, double *Ap);

extern rocblas_status (*p_rocblas_sspr2)(rocblas_handle handle,
                                         rocblas_fill uplo, std::int64_t n,
                                         const float *alpha, const float *X,
                                         std::int64_t incx, const float *Y,
                                         std::int64_t incy, float *Ap);

extern rocblas_status (*p_rocblas_dspr2)(rocblas_handle handle,
                                         rocblas_fill uplo, std::int64_t n,
                                         const double *alpha, const double *X,
                                         std::int64_t incx, const double *Y,
                                         std::int64_t incy, double *Ap);

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
extern rocblas_status (*p_rocblas_ssyr2k)(rocblas_handle, rocblas_fill,
                                          rocblas_operation, rocblas_int,
                                          rocblas_int, const float *,
                                          const float *, rocblas_int,
                                          const float *, rocblas_int,
                                          const float *, float *, rocblas_int);

extern rocblas_status (*p_rocblas_dsyr2k)(
    rocblas_handle, rocblas_fill, rocblas_operation, rocblas_int, rocblas_int,
    const double *, const double *, rocblas_int, const double *, rocblas_int,
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

//Level 3

extern rocblas_status (*p_rocblas_cgemm)(
    rocblas_handle, rocblas_operation, rocblas_operation, rocblas_int,
    rocblas_int, rocblas_int, const rocblas_float_complex *,
    const rocblas_float_complex *, rocblas_int, const rocblas_float_complex *,
    rocblas_int, const rocblas_float_complex *, rocblas_float_complex *,
    rocblas_int);
extern rocblas_status (*p_rocblas_zgemm)(
    rocblas_handle, rocblas_operation, rocblas_operation, rocblas_int,
    rocblas_int, rocblas_int, const rocblas_double_complex *,
    const rocblas_double_complex *, rocblas_int, const rocblas_double_complex *,
    rocblas_int, const rocblas_double_complex *, rocblas_double_complex *,
    rocblas_int);
extern rocblas_status (*p_rocblas_csymm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_int, rocblas_int,
    const rocblas_float_complex *, const rocblas_float_complex *, rocblas_int,
    const rocblas_float_complex *, rocblas_int, const rocblas_float_complex *,
    rocblas_float_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_zsymm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_int, rocblas_int,
    const rocblas_double_complex *, const rocblas_double_complex *, rocblas_int,
    const rocblas_double_complex *, rocblas_int, const rocblas_double_complex *,
    rocblas_double_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_csyrk)(
    rocblas_handle, rocblas_fill, rocblas_operation, rocblas_int, rocblas_int,
    const rocblas_float_complex *, const rocblas_float_complex *, rocblas_int,
    const rocblas_float_complex *, rocblas_float_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_zsyrk)(
    rocblas_handle, rocblas_fill, rocblas_operation, rocblas_int, rocblas_int,
    const rocblas_double_complex *, const rocblas_double_complex *, rocblas_int,
    const rocblas_double_complex *, rocblas_double_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_csyr2k)(
    rocblas_handle, rocblas_fill, rocblas_operation, rocblas_int, rocblas_int,
    const rocblas_float_complex *, const rocblas_float_complex *, rocblas_int,
    const rocblas_float_complex *, rocblas_int, const rocblas_float_complex *,
    rocblas_float_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_zsyr2k)(
    rocblas_handle, rocblas_fill, rocblas_operation, rocblas_int, rocblas_int,
    const rocblas_double_complex *, const rocblas_double_complex *, rocblas_int,
    const rocblas_double_complex *, rocblas_int, const rocblas_double_complex *,
    rocblas_double_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_ctrmm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_operation,
    rocblas_diagonal, rocblas_int, rocblas_int, const rocblas_float_complex *,
    const rocblas_float_complex *, rocblas_int, const rocblas_float_complex *,
    rocblas_int, rocblas_float_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_ztrmm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_operation,
    rocblas_diagonal, rocblas_int, rocblas_int, const rocblas_double_complex *,
    const rocblas_double_complex *, rocblas_int, const rocblas_double_complex *,
    rocblas_int, rocblas_double_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_ctrsm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_operation,
    rocblas_diagonal, rocblas_int, rocblas_int, const rocblas_float_complex *,
    const rocblas_float_complex *, rocblas_int, rocblas_float_complex *,
    rocblas_int);
extern rocblas_status (*p_rocblas_ztrsm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_operation,
    rocblas_diagonal, rocblas_int, rocblas_int, const rocblas_double_complex *,
    const rocblas_double_complex *, rocblas_int, rocblas_double_complex *,
    rocblas_int);
extern rocblas_status (*p_rocblas_chemm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_int, rocblas_int,
    const rocblas_float_complex *, const rocblas_float_complex *, rocblas_int,
    const rocblas_float_complex *, rocblas_int, const rocblas_float_complex *,
    rocblas_float_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_zhemm)(
    rocblas_handle, rocblas_side, rocblas_fill, rocblas_int, rocblas_int,
    const rocblas_double_complex *, const rocblas_double_complex *, rocblas_int,
    const rocblas_double_complex *, rocblas_int, const rocblas_double_complex *,
    rocblas_double_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_cherk)(rocblas_handle, rocblas_fill,
                                         rocblas_operation, rocblas_int,
                                         rocblas_int, const float *,
                                         const rocblas_float_complex *,
                                         rocblas_int, const float *,
                                         rocblas_float_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_zherk)(rocblas_handle, rocblas_fill,
                                         rocblas_operation, rocblas_int,
                                         rocblas_int, const double *,
                                         const rocblas_double_complex *,
                                         rocblas_int, const double *,
                                         rocblas_double_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_cher2k)(
    rocblas_handle, rocblas_fill, rocblas_operation, rocblas_int, rocblas_int,
    const rocblas_float_complex *, const rocblas_float_complex *, rocblas_int,
    const rocblas_float_complex *, rocblas_int, const float *,
    rocblas_float_complex *, rocblas_int);
extern rocblas_status (*p_rocblas_zher2k)(
    rocblas_handle, rocblas_fill, rocblas_operation, rocblas_int, rocblas_int,
    const rocblas_double_complex *, const rocblas_double_complex *, rocblas_int,
    const rocblas_double_complex *, rocblas_int, const double *,
    rocblas_double_complex *, rocblas_int);

bool loadCudaAndCublas();
bool loadOpenBlas();
bool loadHipAndRocblas();

}
}
