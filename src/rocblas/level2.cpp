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


// #include "clap/openblas_utils.hpp"

#include "clap/rocblas_backend.hpp"
#include "clap/rocblas_utils.hpp"
namespace clap{
    
//Float GEMV

void RocBlasBackend::gemv(Layout layout, Transpose trans, int64_t m, 
          int64_t n, const float alpha, const float *A, int64_t lda,
          const float *x, int64_t incx, const float beta,
          float *y, int64_t incy)
{
    float *d_A = nullptr;
    float *d_x = nullptr;
    float *d_y = nullptr;

    rocblas_operation roc_trans = clap::to_rocblas_trans(trans);

    std::size_t sizeA = lda * n;

    std::size_t sizex =
        (roc_trans == rocblas_operation_none)
            ? (1 + (n - 1) * std::abs(incx))
            : (1 + (m - 1) * std::abs(incx));

    std::size_t sizey =
        (roc_trans == rocblas_operation_none)
            ? (1 + (m - 1) * std::abs(incy))
            : (1 + (n - 1) * std::abs(incy));

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(float));
    clap_hipMalloc((void**)&d_x, sizex * sizeof(float));
    clap_hipMalloc((void**)&d_y, sizey * sizeof(float));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_x, x, sizex * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_y, y, sizey * sizeof(float), hipMemcpyHostToDevice);


    clap_rocblasSgemv(handle, roc_trans, (int)m, (int)n, &alpha, d_A, (int)lda, d_x, 
                     (int)incx, &beta, d_y, (int)incy);

    clap_hipMemcpy(y, d_y, sizey * sizeof(float), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_x);
    clap_hipFree(d_y);
}

//Double GEMV

void RocBlasBackend::gemv(Layout layout, Transpose trans, int64_t m,
          int64_t n, const double alpha, const double *A, int64_t lda,
          const double *x, int64_t incx, const double beta,
          double *y, int64_t incy)
{
    double *d_A = nullptr, *d_x = nullptr, *d_y = nullptr;

    rocblas_operation roc_trans =
        clap::to_rocblas_trans(trans);

    std::size_t sizeA = lda * n;

    std::size_t sizex =
        (roc_trans == rocblas_operation_none)
            ? (1 + (n - 1) * std::abs(incx))
            : (1 + (m - 1) * std::abs(incx));

    std::size_t sizey =
        (roc_trans == rocblas_operation_none)
            ? (1 + (m - 1) * std::abs(incy))
            : (1 + (n - 1) * std::abs(incy));

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(double));
    clap_hipMalloc((void**)&d_x, sizex * sizeof(double));
    clap_hipMalloc((void**)&d_y, sizey * sizeof(double));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_x, x, sizex * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_y, y, sizey * sizeof(double), hipMemcpyHostToDevice);


    clap_rocblasDgemv(handle, roc_trans, (int)m, (int)n, &alpha, d_A, (int)lda, d_x,
                     (int)incx, &beta, d_y, (int)incy);

    clap_hipMemcpy(y, d_y, sizey * sizeof(double), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_x);
    clap_hipFree(d_y);
}

//Float SYMV

void RocBlasBackend::symv(Layout layout, Uplo uplo, int64_t n, const float alpha, const float *A, int64_t lda,
          const float *X, int64_t incx,  const float beta, float *Y, int64_t incy)
{
    float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

    std::size_t sizeA = lda * n;
    std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
    std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(float));
    clap_hipMalloc((void**)&d_X, sizeX * sizeof(float));
    clap_hipMalloc((void**)&d_Y, sizeY * sizeof(float));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_Y, Y, sizeY * sizeof(float), hipMemcpyHostToDevice);


    clap_rocblasSsymv(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha,
                d_A, (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

    clap_hipMemcpy(Y, d_Y, sizeY * sizeof(float), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_X);
    clap_hipFree(d_Y);
}

//Double SYMV

void RocBlasBackend::symv(Layout layout, Uplo uplo, int64_t n, const double alpha, const double *A, int64_t lda,
          const double *X, int64_t incx,  const double beta, double *Y, int64_t incy)
{
    double *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

    std::size_t sizeA = lda * n;
    std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
    std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(double));
    clap_hipMalloc((void**)&d_X, sizeX * sizeof(double));
    clap_hipMalloc((void**)&d_Y, sizeY * sizeof(double));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_Y, Y, sizeY * sizeof(double), hipMemcpyHostToDevice);


    clap_rocblasDsymv(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha,
                d_A, (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);


    clap_hipMemcpy(Y, d_Y, sizeY * sizeof(double), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_X);
    clap_hipFree(d_Y);
}

//Float TRMV

void RocBlasBackend::trmv(Layout layout, Uplo uplo,
          Transpose transA, Diag diag,
          int64_t n,
          const float* A, int64_t lda,
          float* x, int64_t incx)
{
    float *d_A = nullptr, *d_x = nullptr;

    int64_t size_x = n;

    clap_hipMalloc((void**)&d_A, lda * n * sizeof(float));
    clap_hipMalloc((void**)&d_x, size_x * incx * sizeof(float));

    clap_hipMemcpy(d_A, A, lda * n * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_x, x, size_x * incx * sizeof(float), hipMemcpyHostToDevice);

    clap_rocblasStrmv(handle, clap::to_rocblas_uplo(uplo), clap::to_rocblas_trans(transA),
                  clap::to_rocblas_diag(diag), (int)n, d_A, (int)lda, d_x, (int)incx);

    clap_hipMemcpy(x, d_x, size_x * incx * sizeof(float), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_x);
}

//Double TRMV

void RocBlasBackend::trmv(Layout layout, Uplo uplo,
          Transpose transA, Diag diag, int64_t n,
          const double* A, int64_t lda,
          double* x, int64_t incx)
{
    double *d_A = nullptr, *d_x = nullptr;

    int size_x = n;

    clap_hipMalloc((void**)&d_A, lda * n * sizeof(double));
    clap_hipMalloc((void**)&d_x, size_x * incx * sizeof(double));

    clap_hipMemcpy(d_A, A, lda * n * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_x, x, size_x * incx * sizeof(double), hipMemcpyHostToDevice);

    clap_rocblasDtrmv(handle, clap::to_rocblas_uplo(uplo), clap::to_rocblas_trans(transA),
             clap::to_rocblas_diag(diag), (int)n, d_A, (int)lda, d_x, (int)incx);

    clap_hipMemcpy(x, d_x, size_x * incx * sizeof(double), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_x);
}

//Float TRSV

void RocBlasBackend::trsv(Layout layout, Uplo uplo,
          Transpose transA, Diag diag, int64_t n,
          const float* A, int64_t lda,
          float* x, int64_t incx)
{
    int size_A = lda * n;
    int size_x = n;

    float *d_A = nullptr, *d_x = nullptr;

    clap_hipMalloc((void**)&d_A, size_A * sizeof(float));
    clap_hipMalloc((void**)&d_x, size_x * incx * sizeof(float));

    clap_hipMemcpy(d_A, A, size_A * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_x, x, size_x * incx * sizeof(float), hipMemcpyHostToDevice);

    clap_rocblasStrsv(handle, clap::to_rocblas_uplo(uplo), clap::to_rocblas_trans(transA),
              clap::to_rocblas_diag(diag), (int)n, d_A, (int)lda, d_x, (int)incx);

    clap_hipMemcpy(x, d_x, size_x * incx * sizeof(float), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_x);
}

//Double TRSV

void RocBlasBackend::trsv(Layout layout, Uplo uplo,
          Transpose transA, Diag diag, int64_t n,
          const double* A, int64_t lda,
          double* x, int64_t incx)
{
    int size_A = lda * n;
    int size_x = n;

    double *d_A = nullptr, *d_x = nullptr;

    clap_hipMalloc((void**)&d_A, size_A * sizeof(double));
    clap_hipMalloc((void**)&d_x, size_x * incx * sizeof(double));

    clap_hipMemcpy(d_A, A, size_A * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_x, x, size_x * incx * sizeof(double), hipMemcpyHostToDevice);

    clap_rocblasDtrsv(handle, clap::to_rocblas_uplo(uplo), clap::to_rocblas_trans(transA),
                    clap::to_rocblas_diag(diag), (int)n, d_A, (int)lda, d_x, (int)incx);

    clap_hipMemcpy(x, d_x, size_x * incx * sizeof(double), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_x);
}
//Float SBMV

void RocBlasBackend::sbmv(Layout layout, Uplo uplo,
          int64_t n, int64_t k,
          const float alpha,
          const float *A, int64_t lda,
          const float *X, int64_t incx,
          const float beta,
          float *Y, int64_t incy)
{
    float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

    std::size_t sizeA = lda * n;
    std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
    std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(float));
    clap_hipMalloc((void**)&d_X, sizeX * sizeof(float));
    clap_hipMalloc((void**)&d_Y, sizeY * sizeof(float));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_Y, Y, sizeY * sizeof(float), hipMemcpyHostToDevice);


    clap_rocblasSsbmv(handle, clap::to_rocblas_uplo(uplo), (int)n, (int)k,  &alpha, d_A, (int)lda,
                      d_X, (int)incx,  &beta, d_Y, (int)incy);

    clap_hipMemcpy(Y, d_Y, sizeY * sizeof(float), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_X);
    clap_hipFree(d_Y);
}

//Double SBMV

void RocBlasBackend::sbmv(Layout layout, Uplo uplo,
          int64_t n, int64_t k,
          const double alpha,
          const double *A, int64_t lda,
          const double *X, int64_t incx,
          const double beta,
          double *Y, int64_t incy)
{
    double *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

    std::size_t sizeA = lda * n;
    std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
    std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(double));
    clap_hipMalloc((void**)&d_X, sizeX * sizeof(double));
    clap_hipMalloc((void**)&d_Y, sizeY * sizeof(double));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_Y, Y, sizeY * sizeof(double), hipMemcpyHostToDevice);


    clap_rocblasDsbmv(handle, clap::to_rocblas_uplo(uplo), (int)n, (int)k, &alpha, d_A, (int)lda,
                      d_X, (int)incx, &beta, d_Y, (int)incy);


    clap_hipMemcpy(Y, d_Y, sizeY * sizeof(double), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_X);
    clap_hipFree(d_Y);
}

// Float GER

void RocBlasBackend::ger(Layout layout, int m, int n, const float alpha,
                         const float *X, int incx, const float *Y, int incy,
                         float *A, int lda) {
  float *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (m - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(float));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));

  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasSger(handle, (int)m, (int)n, &alpha, d_X, (int)incx, d_Y,
                   (int)incy, d_A, (int)lda);

  clap_hipMemcpy(A, d_A, sizeA * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_A);
}

}
