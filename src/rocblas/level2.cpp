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

#include "clap/rocblas_backend.hpp"
#include "clap/rocblas_utils.hpp"
namespace clap {

// Float GEMV
void RocBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                          const float alpha, const float *A, int lda,
                          const float *x, int incx, const float beta, float *y,
                          int incy) {
  float *d_A = nullptr;
  float *d_x = nullptr;
  float *d_y = nullptr;

  rocblas_operation roc_trans = clap::to_rocblas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t sizex = (roc_trans == rocblas_operation_none)
                          ? (1 + (n - 1) * std::abs(incx))
                          : (1 + (m - 1) * std::abs(incx));

  std::size_t sizey = (roc_trans == rocblas_operation_none)
                          ? (1 + (m - 1) * std::abs(incy))
                          : (1 + (n - 1) * std::abs(incy));

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_x, sizex * sizeof(float));
  clap_hipMalloc((void **)&d_y, sizey * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_x, x, sizex * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, sizey * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasSgemv(handle, roc_trans, (int)m, (int)n, &alpha, d_A, (int)lda,
                    d_x, (int)incx, &beta, d_y, (int)incy);

  clap_hipMemcpy(y, d_y, sizey * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Double GEMV
void RocBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                          const double alpha, const double *A, int lda,
                          const double *x, int incx, const double beta,
                          double *y, int incy) {
  double *d_A = nullptr, *d_x = nullptr, *d_y = nullptr;

  rocblas_operation roc_trans = clap::to_rocblas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t sizex = (roc_trans == rocblas_operation_none)
                          ? (1 + (n - 1) * std::abs(incx))
                          : (1 + (m - 1) * std::abs(incx));

  std::size_t sizey = (roc_trans == rocblas_operation_none)
                          ? (1 + (m - 1) * std::abs(incy))
                          : (1 + (n - 1) * std::abs(incy));

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_x, sizex * sizeof(double));
  clap_hipMalloc((void **)&d_y, sizey * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_x, x, sizex * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, sizey * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDgemv(handle, roc_trans, (int)m, (int)n, &alpha, d_A, (int)lda,
                    d_x, (int)incx, &beta, d_y, (int)incy);

  clap_hipMemcpy(y, d_y, sizey * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Complex Float GEMV
void RocBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                          const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          const std::complex<float> *X, int incx,
                          const std::complex<float> *beta,
                          std::complex<float> *Y, int incy) {

  const roc_complex *alpha_roc = reinterpret_cast<const roc_complex *>(alpha);
  const roc_complex *beta_roc = reinterpret_cast<const roc_complex *>(beta);
  const roc_complex *A_roc = reinterpret_cast<const roc_complex *>(A);
  const roc_complex *X_roc = reinterpret_cast<const roc_complex *>(X);

  roc_complex *Y_roc = reinterpret_cast<roc_complex *>(Y);
  roc_complex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  rocblas_operation roc_trans = clap::to_rocblas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t sizeX = (roc_trans == rocblas_operation_none)
                          ? (1 + (n - 1) * std::abs(incx))
                          : (1 + (m - 1) * std::abs(incx));

  std::size_t sizeY = (roc_trans == rocblas_operation_none)
                          ? (1 + (m - 1) * std::abs(incy))
                          : (1 + (n - 1) * std::abs(incy));

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_complex));

  clap_hipMemcpy(d_A, A_roc, sizeA * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_roc, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_roc, sizeY * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasCgemv(handle, roc_trans, (int)m, (int)n, alpha_roc, d_A, (int)lda,
                    d_X, (int)incx, beta_roc, d_Y, (int)incy);

  clap_hipMemcpy(Y_roc, d_Y, sizeY * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Complex Double GEMV
void RocBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                          const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          const std::complex<double> *X, int incx,
                          const std::complex<double> *beta,
                          std::complex<double> *Y, int incy) {

  const roc_DoubleComplex *alpha_roc =
      reinterpret_cast<const roc_DoubleComplex *>(alpha);

  const roc_DoubleComplex *beta_roc =
      reinterpret_cast<const roc_DoubleComplex *>(beta);

  const roc_DoubleComplex *A_roc =
      reinterpret_cast<const roc_DoubleComplex *>(A);

  const roc_DoubleComplex *X_roc =
      reinterpret_cast<const roc_DoubleComplex *>(X);

  roc_DoubleComplex *Y_roc = reinterpret_cast<roc_DoubleComplex *>(Y);

  roc_DoubleComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  rocblas_operation roc_trans = clap::to_rocblas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t sizeX = (roc_trans == rocblas_operation_none)
                          ? (1 + (n - 1) * std::abs(incx))
                          : (1 + (m - 1) * std::abs(incx));

  std::size_t sizeY = (roc_trans == rocblas_operation_none)
                          ? (1 + (m - 1) * std::abs(incy))
                          : (1 + (n - 1) * std::abs(incy));

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_DoubleComplex));

  clap_hipMemcpy(d_A, A_roc, sizeA * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_roc, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_roc, sizeY * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZgemv(handle, roc_trans, (int)m, (int)n, alpha_roc, d_A, (int)lda,
                    d_X, (int)incx, beta_roc, d_Y, (int)incy);

  clap_hipMemcpy(Y_roc, d_Y, sizeY * sizeof(roc_DoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Float SYMV
void RocBlasBackend::symv(Layout layout, Uplo uplo, int n, const float alpha,
                          const float *A, int lda, const float *X, int incx,
                          const float beta, float *Y, int incy) {
  float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasSsymv(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_A,
                    (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_hipMemcpy(Y, d_Y, sizeY * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Double SYMV
void RocBlasBackend::symv(Layout layout, Uplo uplo, int n, const double alpha,
                          const double *A, int lda, const double *X, int incx,
                          const double beta, double *Y, int incy) {
  double *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDsymv(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_A,
                    (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_hipMemcpy(Y, d_Y, sizeY * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Complex Float HEMV
void RocBlasBackend::hemv(Layout layout, Uplo uplo, int n,
                          const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          const std::complex<float> *X, int incx,
                          const std::complex<float> *beta,
                          std::complex<float> *Y, int incy) {
  const roc_complex *alpha_cuda = reinterpret_cast<const roc_complex *>(alpha);
  const roc_complex *A_cuda = reinterpret_cast<const roc_complex *>(A);
  const roc_complex *X_cuda = reinterpret_cast<const roc_complex *>(X);
  const roc_complex *beta_cuda = reinterpret_cast<const roc_complex *>(beta);
  roc_complex *Y_cuda = reinterpret_cast<roc_complex *>(Y);

  roc_complex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_complex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasChemv(handle, clap::to_rocblas_uplo(uplo), (int)n, alpha_cuda,
                    d_A, (int)lda, d_X, (int)incx, beta_cuda, d_Y, (int)incy);

  clap_hipMemcpy(Y_cuda, d_Y, sizeY * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Complex Double HEMV
void RocBlasBackend::hemv(Layout layout, Uplo uplo, int n,
                          const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          const std::complex<double> *X, int incx,
                          const std::complex<double> *beta,
                          std::complex<double> *Y, int incy) {
  const roc_DoubleComplex *alpha_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(alpha);
  const roc_DoubleComplex *A_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(A);
  const roc_DoubleComplex *X_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(X);
  const roc_DoubleComplex *beta_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(beta);
  roc_DoubleComplex *Y_cuda = reinterpret_cast<roc_DoubleComplex *>(Y);

  roc_DoubleComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_DoubleComplex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZhemv(handle, clap::to_rocblas_uplo(uplo), (int)n, alpha_cuda,
                    d_A, (int)lda, d_X, (int)incx, beta_cuda, d_Y, (int)incy);

  clap_hipMemcpy(Y_cuda, d_Y, sizeY * sizeof(roc_DoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Float TRMV
void RocBlasBackend::trmv(Layout layout, Uplo uplo, Transpose transA, Diag diag,
                          int n, const float *A, int lda, float *x, int incx) {
  float *d_A = nullptr, *d_x = nullptr;

  int size_x = n;

  clap_hipMalloc((void **)&d_A, lda * n * sizeof(float));
  clap_hipMalloc((void **)&d_x, size_x * incx * sizeof(float));

  clap_hipMemcpy(d_A, A, lda * n * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_x, x, size_x * incx * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasStrmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(transA), clap::to_rocblas_diag(diag),
                    (int)n, d_A, (int)lda, d_x, (int)incx);

  clap_hipMemcpy(x, d_x, size_x * incx * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_x);
}

// Double TRMV
void RocBlasBackend::trmv(Layout layout, Uplo uplo, Transpose transA, Diag diag,
                          int n, const double *A, int lda, double *x,
                          int incx) {
  double *d_A = nullptr, *d_x = nullptr;

  int size_x = n;

  clap_hipMalloc((void **)&d_A, lda * n * sizeof(double));
  clap_hipMalloc((void **)&d_x, size_x * incx * sizeof(double));

  clap_hipMemcpy(d_A, A, lda * n * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_x, x, size_x * incx * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDtrmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(transA), clap::to_rocblas_diag(diag),
                    (int)n, d_A, (int)lda, d_x, (int)incx);

  clap_hipMemcpy(x, d_x, size_x * incx * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_x);
}

// Complex Float TRMV
void RocBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const std::complex<float> *A, int lda,
                          std::complex<float> *X, int incx) {
  const roc_complex *A_cuda = reinterpret_cast<const roc_complex *>(A);
  roc_complex *X_cuda = reinterpret_cast<roc_complex *>(X);

  roc_complex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasCtrmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X_cuda, d_X, sizeX * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
}

// Complex Double TRMV
void RocBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const std::complex<double> *A, int lda,
                          std::complex<double> *X, int incx) {
  const roc_DoubleComplex *A_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(A);
  roc_DoubleComplex *X_cuda = reinterpret_cast<roc_DoubleComplex *>(X);

  roc_DoubleComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_DoubleComplex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZtrmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X_cuda, d_X, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
}

// Float TRSV
void RocBlasBackend::trsv(Layout layout, Uplo uplo, Transpose transA, Diag diag,
                          int n, const float *A, int lda, float *x, int incx) {
  int size_A = lda * n;
  int size_x = n;

  float *d_A = nullptr, *d_x = nullptr;

  clap_hipMalloc((void **)&d_A, size_A * sizeof(float));
  clap_hipMalloc((void **)&d_x, size_x * incx * sizeof(float));

  clap_hipMemcpy(d_A, A, size_A * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_x, x, size_x * incx * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasStrsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(transA), clap::to_rocblas_diag(diag),
                    (int)n, d_A, (int)lda, d_x, (int)incx);

  clap_hipMemcpy(x, d_x, size_x * incx * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_x);
}

// Double TRSV
void RocBlasBackend::trsv(Layout layout, Uplo uplo, Transpose transA, Diag diag,
                          int n, const double *A, int lda, double *x,
                          int incx) {
  int size_A = lda * n;
  int size_x = n;

  double *d_A = nullptr, *d_x = nullptr;

  clap_hipMalloc((void **)&d_A, size_A * sizeof(double));
  clap_hipMalloc((void **)&d_x, size_x * incx * sizeof(double));

  clap_hipMemcpy(d_A, A, size_A * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_x, x, size_x * incx * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDtrsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(transA), clap::to_rocblas_diag(diag),
                    (int)n, d_A, (int)lda, d_x, (int)incx);

  clap_hipMemcpy(x, d_x, size_x * incx * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_x);
}

// Complex Float TRSV
void RocBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const std::complex<float> *A, int lda,
                          std::complex<float> *X, int incx) {
  const cuComplex *A_cuda = reinterpret_cast<const cuComplex *>(A);
  cuComplex *X_cuda = reinterpret_cast<cuComplex *>(X);

  cuComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(cuComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(cuComplex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex), hipMemcpyHostToDevice);

  clap_rocblasCtrsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X_cuda, d_X, sizeX * sizeof(cuComplex), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
}

// Complex Double TRSV
void RocBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const std::complex<double> *A, int lda,
                          std::complex<double> *X, int incx) {
  const cuDoubleComplex *A_cuda = reinterpret_cast<const cuDoubleComplex *>(A);
  cuDoubleComplex *X_cuda = reinterpret_cast<cuDoubleComplex *>(X);

  cuDoubleComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZtrsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X_cuda, d_X, sizeX * sizeof(cuDoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
}

// Complex Float HPMV
void RocBlasBackend::hpmv(Layout layout, Uplo uplo, int n,
                          const std::complex<float> *alpha,
                          const std::complex<float> *Ap,
                          const std::complex<float> *X, int incx,
                          const std::complex<float> *beta,
                          std::complex<float> *Y, int incy) {
  const roc_complex *alpha_roc = reinterpret_cast<const roc_complex *>(alpha);
  const roc_complex *Ap_roc = reinterpret_cast<const roc_complex *>(Ap);
  const roc_complex *X_roc = reinterpret_cast<const roc_complex *>(X);
  const roc_complex *beta_roc = reinterpret_cast<const roc_complex *>(beta);
  roc_complex *Y_roc = reinterpret_cast<roc_complex *>(Y);

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  roc_complex *d_Ap = nullptr, *d_X = nullptr, *d_Y = nullptr;

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_complex));

  clap_hipMemcpy(d_Ap, Ap_roc, sizeAP * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_roc, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_roc, sizeY * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasChpmv(handle, clap::to_rocblas_uplo(uplo), (int)n, alpha_roc,
                    d_Ap, d_X, (int)incx, beta_roc, d_Y, (int)incy);

  clap_hipMemcpy(Y_roc, d_Y, sizeY * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Complex Double HPMV
void RocBlasBackend::hpmv(Layout layout, Uplo uplo, int n,
                          const std::complex<double> *alpha,
                          const std::complex<double> *Ap,
                          const std::complex<double> *X, int incx,
                          const std::complex<double> *beta,
                          std::complex<double> *Y, int incy) {
  const roc_DoubleComplex *alpha_roc =
      reinterpret_cast<const roc_DoubleComplex *>(alpha);
  const roc_DoubleComplex *Ap_roc =
      reinterpret_cast<const roc_DoubleComplex *>(Ap);
  const roc_DoubleComplex *X_roc =
      reinterpret_cast<const roc_DoubleComplex *>(X);
  const roc_DoubleComplex *beta_roc =
      reinterpret_cast<const roc_DoubleComplex *>(beta);
  roc_DoubleComplex *Y_roc = reinterpret_cast<roc_DoubleComplex *>(Y);

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  roc_DoubleComplex *d_Ap = nullptr, *d_X = nullptr, *d_Y = nullptr;

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_DoubleComplex));

  clap_hipMemcpy(d_Ap, Ap_roc, sizeAP * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_roc, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_roc, sizeY * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZhpmv(handle, clap::to_rocblas_uplo(uplo), (int)n, alpha_roc,
                    d_Ap, d_X, (int)incx, beta_roc, d_Y, (int)incy);

  clap_hipMemcpy(Y_roc, d_Y, sizeY * sizeof(roc_DoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Float SBMV
void RocBlasBackend::sbmv(Layout layout, Uplo uplo, int n, int k,
                          const float alpha, const float *A, int lda,
                          const float *X, int incx, const float beta, float *Y,
                          int incy) {
  float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasSsbmv(handle, clap::to_rocblas_uplo(uplo), (int)n, (int)k, &alpha,
                    d_A, (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_hipMemcpy(Y, d_Y, sizeY * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Double SBMV
void RocBlasBackend::sbmv(Layout layout, Uplo uplo, int n, int k,
                          const double alpha, const double *A, int lda,
                          const double *X, int incx, const double beta,
                          double *Y, int incy) {
  double *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDsbmv(handle, clap::to_rocblas_uplo(uplo), (int)n, (int)k, &alpha,
                    d_A, (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_hipMemcpy(Y, d_Y, sizeY * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Float TBMV
void RocBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, int k, const float *A, int lda, float *X,
                          int incx) {
  float *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasStbmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X, d_X, sizeX * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
}

// Double TBMV
void RocBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, int k, const double *A, int lda, double *X,
                          int incx) {
  double *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDtbmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X, d_X, sizeX * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
}

// Complex Float TBMV
void RocBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, int k, const std::complex<float> *A, int lda,
                          std::complex<float> *X, int incx) {
  const roc_complex *A_roc = reinterpret_cast<const roc_complex *>(A);

  roc_complex *X_roc = reinterpret_cast<roc_complex *>(X);

  roc_complex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));

  clap_hipMemcpy(d_A, A_roc, sizeA * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_roc, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasCtbmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X_roc, d_X, sizeX * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
}

// Complex Double TBMV
void RocBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, int k, const std::complex<double> *A, int lda,
                          std::complex<double> *X, int incx) {
  const roc_DoubleComplex *A_roc =
      reinterpret_cast<const roc_DoubleComplex *>(A);

  roc_DoubleComplex *X_roc = reinterpret_cast<roc_DoubleComplex *>(X);

  roc_DoubleComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_DoubleComplex));

  clap_hipMemcpy(d_A, A_roc, sizeA * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_roc, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZtbmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X_roc, d_X, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
}

// Float TBSV
void RocBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, int k, const float *A, int lda, float *X,
                          int incx) {
  float *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasStbsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X, d_X, sizeX * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
}

// Double TBSV
void RocBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, int k, const double *A, int lda, double *X,
                          int incx) {
  double *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDtbsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X, d_X, sizeX * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
}

// Complex Float TBSV
void RocBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, int k, const std::complex<float> *A, int lda,
                          std::complex<float> *X, int incx) {
  const roc_complex *A_cuda = reinterpret_cast<const roc_complex *>(A);
  roc_complex *X_cuda = reinterpret_cast<roc_complex *>(X);

  roc_complex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasCtbsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X_cuda, d_X, sizeX * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
}

// Complex Double TBSV
void RocBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, int k, const std::complex<double> *A, int lda,
                          std::complex<double> *X, int incx) {
  const roc_DoubleComplex *A_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(A);
  roc_DoubleComplex *X_cuda = reinterpret_cast<roc_DoubleComplex *>(X);

  roc_DoubleComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_DoubleComplex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZtbsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_hipMemcpy(X_cuda, d_X, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
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

// Double GER
void RocBlasBackend::ger(Layout layout, int m, int n, const double alpha,
                         const double *X, int incx, const double *Y, int incy,
                         double *A, int lda) {
  double *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (m - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(double));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));

  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDger(handle, (int)m, (int)n, &alpha, d_X, (int)incx, d_Y,
                   (int)incy, d_A, (int)lda);

  clap_hipMemcpy(A, d_A, sizeA * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_A);
}

// Complex Float GERC
void RocBlasBackend::gerc(Layout layout, int m, int n,
                          const std::complex<float> *alpha,
                          const std::complex<float> *X, int incx,
                          const std::complex<float> *Y, int incy,
                          std::complex<float> *A, int lda) {
  const roc_complex *alpha_cuda = reinterpret_cast<const roc_complex *>(alpha);
  const roc_complex *X_cuda = reinterpret_cast<const roc_complex *>(X);
  const roc_complex *Y_cuda = reinterpret_cast<const roc_complex *>(Y);
  roc_complex *A_cuda = reinterpret_cast<roc_complex *>(A);

  roc_complex *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (m - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_complex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasCgerc(handle, (int)m, (int)n, alpha_cuda, d_X, (int)incx, d_Y,
                    (int)incy, d_A, (int)lda);

  clap_hipMemcpy(A_cuda, d_A, sizeA * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_A);
}

// Complex Double GERC
void RocBlasBackend::gerc(Layout layout, int m, int n,
                          const std::complex<double> *alpha,
                          const std::complex<double> *X, int incx,
                          const std::complex<double> *Y, int incy,
                          std::complex<double> *A, int lda) {
  const roc_DoubleComplex *alpha_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(alpha);
  const roc_DoubleComplex *X_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(X);
  const roc_DoubleComplex *Y_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(Y);
  roc_DoubleComplex *A_cuda = reinterpret_cast<roc_DoubleComplex *>(A);

  roc_DoubleComplex *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (m - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_DoubleComplex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZgerc(handle, (int)m, (int)n, alpha_cuda, d_X, (int)incx, d_Y,
                    (int)incy, d_A, (int)lda);

  clap_hipMemcpy(A_cuda, d_A, sizeA * sizeof(roc_DoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_A);
}

// Complex Float GERU
void RocBlasBackend::geru(Layout layout, int m, int n,
                          const std::complex<float> *alpha,
                          const std::complex<float> *X, int incx,
                          const std::complex<float> *Y, int incy,
                          std::complex<float> *A, int lda) {
  const roc_complex *alpha_cuda = reinterpret_cast<const roc_complex *>(alpha);
  const roc_complex *X_cuda = reinterpret_cast<const roc_complex *>(X);
  const roc_complex *Y_cuda = reinterpret_cast<const roc_complex *>(Y);
  roc_complex *A_cuda = reinterpret_cast<roc_complex *>(A);

  roc_complex *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (m - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_complex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasCgeru(handle, (int)m, (int)n, alpha_cuda, d_X, (int)incx, d_Y,
                    (int)incy, d_A, (int)lda);

  clap_hipMemcpy(A_cuda, d_A, sizeA * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_A);
}

// Complex Double GERU
void RocBlasBackend::geru(Layout layout, int m, int n,
                          const std::complex<double> *alpha,
                          const std::complex<double> *X, int incx,
                          const std::complex<double> *Y, int incy,
                          std::complex<double> *A, int lda) {
  const cuDoubleComplex *alpha_cuda =
      reinterpret_cast<const cuDoubleComplex *>(alpha);
  const cuDoubleComplex *X_cuda = reinterpret_cast<const cuDoubleComplex *>(X);
  const cuDoubleComplex *Y_cuda = reinterpret_cast<const cuDoubleComplex *>(Y);
  cuDoubleComplex *A_cuda = reinterpret_cast<cuDoubleComplex *>(A);

  cuDoubleComplex *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (m - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZgeru(handle, (int)m, (int)n, alpha_cuda, d_X, (int)incx, d_Y,
                    (int)incy, d_A, (int)lda);

  clap_hipMemcpy(A_cuda, d_A, sizeA * sizeof(cuDoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_A);
}

// Float SYR
void RocBlasBackend::syr(Layout layout, Uplo uplo, int n, const float alpha,
                         const float *X, int incx, float *A, int lda) {
  float *d_X = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));

  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasSsyr(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_A, (int)lda);

  clap_hipMemcpy(A, d_A, sizeA * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_A);
}

// Double SYR
void RocBlasBackend::syr(Layout layout, Uplo uplo, int n, const double alpha,
                         const double *X, int incx, double *A, int lda) {
  double *d_X = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));

  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDsyr(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_A, (int)lda);

  clap_hipMemcpy(A, d_A, sizeA * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_A);
}

// Float SYR2
void RocBlasBackend::syr2(Layout layout, Uplo uplo, int n, const float alpha,
                          const float *X, int incx, const float *Y, int incy,
                          float *A, int lda) {
  float *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(float));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));

  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasSsyr2(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                    (int)incx, d_Y, (int)incy, d_A, (int)lda);

  clap_hipMemcpy(A, d_A, sizeA * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_A);
}

// Double SYR2
void RocBlasBackend::syr2(Layout layout, Uplo uplo, int n, const double alpha,
                          const double *X, int incx, const double *Y, int incy,
                          double *A, int lda) {
  double *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(double));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));

  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDsyr2(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                    (int)incx, d_Y, (int)incy, d_A, (int)lda);

  clap_hipMemcpy(A, d_A, sizeA * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_A);
}

// Complex Float HER
void RocBlasBackend::her(Layout layout, Uplo uplo, int n, const float alpha,
                         const std::complex<float> *X, int incx,
                         std::complex<float> *A, int lda) {
  const roc_complex *X_cuda = reinterpret_cast<const roc_complex *>(X);
  roc_complex *A_cuda = reinterpret_cast<roc_complex *>(A);

  roc_complex *d_X = nullptr;
  roc_complex *d_A = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_complex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasCher(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_A, (int)lda);

  clap_hipMemcpy(A_cuda, d_A, sizeA * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_A);
}

// Complex Double HER
void RocBlasBackend::her(Layout layout, Uplo uplo, int n, const double alpha,
                         const std::complex<double> *X, int incx,
                         std::complex<double> *A, int lda) {
  const roc_DoubleComplex *X_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(X);
  roc_DoubleComplex *A_cuda = reinterpret_cast<roc_DoubleComplex *>(A);

  roc_DoubleComplex *d_X = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_DoubleComplex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZher(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_A, (int)lda);

  clap_hipMemcpy(A_cuda, d_A, sizeA * sizeof(roc_DoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_A);
}

// Complex Float HER2
void RocBlasBackend::her2(Layout layout, Uplo uplo, int n,
                          const std::complex<float> *alpha,
                          const std::complex<float> *X, int incx,
                          const std::complex<float> *Y, int incy,
                          std::complex<float> *A, int lda) {
  const roc_complex *alpha_cuda = reinterpret_cast<const roc_complex *>(alpha);
  const roc_complex *X_cuda = reinterpret_cast<const roc_complex *>(X);
  const roc_complex *Y_cuda = reinterpret_cast<const roc_complex *>(Y);
  roc_complex *A_cuda = reinterpret_cast<roc_complex *>(A);

  roc_complex *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_complex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasCher2(handle, clap::to_rocblas_uplo(uplo), (int)n, alpha_cuda,
                    d_X, (int)incx, d_Y, (int)incy, d_A, (int)lda);

  clap_hipMemcpy(A_cuda, d_A, sizeA * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_A);
}

// Complex Double HER2
void RocBlasBackend::her2(Layout layout, Uplo uplo, int n,
                          const std::complex<double> *alpha,
                          const std::complex<double> *X, int incx,
                          const std::complex<double> *Y, int incy,
                          std::complex<double> *A, int lda) {
  const cuDoubleComplex *alpha_cuda =
      reinterpret_cast<const cuDoubleComplex *>(alpha);
  const cuDoubleComplex *X_cuda = reinterpret_cast<const cuDoubleComplex *>(X);
  const cuDoubleComplex *Y_cuda = reinterpret_cast<const cuDoubleComplex *>(Y);
  cuDoubleComplex *A_cuda = reinterpret_cast<cuDoubleComplex *>(A);

  cuDoubleComplex *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));
  clap_hipMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZher2(handle, clap::to_rocblas_uplo(uplo), (int)n, alpha_cuda,
                    d_X, (int)incx, d_Y, (int)incy, d_A, (int)lda);

  clap_hipMemcpy(A_cuda, d_A, sizeA * sizeof(cuDoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_A);
}

// Complex Float HPR
void RocBlasBackend::hpr(Layout layout, Uplo uplo, int n, const float alpha,
                         const std::complex<float> *X, int incx,
                         std::complex<float> *AP) {
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  cuComplex *AP_cuda = reinterpret_cast<cuComplex *>(AP);

  int Ps = n * (n + 1) / 2;

  cuComplex *d_X = nullptr, *d_AP = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_hipMalloc((void **)&d_AP, Ps * sizeof(cuComplex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_AP, AP_cuda, Ps * sizeof(cuComplex), hipMemcpyHostToDevice);

  clap_rocblasChpr(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_AP);

  clap_hipMemcpy(AP_cuda, d_AP, Ps * sizeof(cuComplex), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_AP);
}

// Complex Double HPR
void RocBlasBackend::hpr(Layout layout, Uplo uplo, int n, const double alpha,
                         const std::complex<double> *X, int incx,
                         std::complex<double> *AP) {

  const cuDoubleComplex *X_cuda = reinterpret_cast<const cuDoubleComplex *>(X);
  cuDoubleComplex *AP_cuda = reinterpret_cast<cuDoubleComplex *>(AP);

  int Ps = n * (n + 1) / 2;

  cuDoubleComplex *d_X = nullptr, *d_AP = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_hipMalloc((void **)&d_AP, Ps * sizeof(cuDoubleComplex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_AP, AP_cuda, Ps * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZhpr(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_AP);

  clap_hipMemcpy(AP_cuda, d_AP, Ps * sizeof(cuDoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_AP);
}

// Complex Float HPR2
void RocBlasBackend::hpr2(Layout layout, Uplo uplo, int n,
                          const std::complex<float> *alpha,
                          const std::complex<float> *X, int incx,
                          const std::complex<float> *Y, int incy,
                          std::complex<float> *AP) {
  const cuComplex *alpha_cuda = reinterpret_cast<const cuComplex *>(alpha);
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  const cuComplex *Y_cuda = reinterpret_cast<const cuComplex *>(Y);
  cuComplex *AP_cuda = reinterpret_cast<cuComplex *>(AP);

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  int Ps = n * (n + 1) / 2;

  cuComplex *d_X = nullptr, *d_Y = nullptr, *d_AP = nullptr;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(cuComplex));
  clap_hipMalloc((void **)&d_AP, Ps * sizeof(cuComplex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuComplex), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_AP, AP_cuda, Ps * sizeof(cuComplex), hipMemcpyHostToDevice);

  clap_rocblasChpr2(handle, clap::to_rocblas_uplo(uplo), (int)n, alpha_cuda,
                    d_X, (int)incx, d_Y, (int)incy, d_AP);

  clap_hipMemcpy(AP_cuda, d_AP, Ps * sizeof(cuComplex), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_AP);
}

// Complex Double HPR2
void RocBlasBackend::hpr2(Layout layout, Uplo uplo, int n,
                          const std::complex<double> *alpha,
                          const std::complex<double> *X, int incx,
                          const std::complex<double> *Y, int incy,
                          std::complex<double> *AP) {
  const cuDoubleComplex *alpha_cuda =
      reinterpret_cast<const cuDoubleComplex *>(alpha);
  const cuDoubleComplex *X_cuda = reinterpret_cast<const cuDoubleComplex *>(X);
  const cuDoubleComplex *Y_cuda = reinterpret_cast<const cuDoubleComplex *>(Y);
  cuDoubleComplex *AP_cuda = reinterpret_cast<cuDoubleComplex *>(AP);

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  int Ps = n * (n + 1) / 2;

  cuDoubleComplex *d_X = nullptr, *d_Y = nullptr, *d_AP = nullptr;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));
  clap_hipMalloc((void **)&d_AP, Ps * sizeof(cuDoubleComplex));

  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_AP, AP_cuda, Ps * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZhpr2(handle, clap::to_rocblas_uplo(uplo), (int)n, alpha_cuda,
                    d_X, (int)incx, d_Y, (int)incy, d_AP);

  clap_hipMemcpy(AP_cuda, d_AP, Ps * sizeof(cuDoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_AP);
}

// Float GBMV
void RocBlasBackend::gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                          int KU, const float alpha, const float *A, int lda,
                          const float *X, int incx, const float beta, float *Y,
                          int incy) {
  float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  rocblas_operation roc_trans = clap::to_rocblas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t lenX = (roc_trans == rocblas_operation_none) ? n : m;
  std::size_t lenY = (roc_trans == rocblas_operation_none) ? m : n;

  std::size_t sizeX = 1 + (lenX - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (lenY - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasSgbmv(handle, roc_trans, (int)m, (int)n, (int)KL, (int)KU, &alpha,
                    d_A, (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_hipMemcpy(Y, d_Y, sizeY * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Double GBMV
void RocBlasBackend::gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                          int KU, const double alpha, const double *A, int lda,
                          const double *X, int incx, const double beta,
                          double *Y, int incy) {
  double *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  rocblas_operation roc_trans = clap::to_rocblas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t lenX = (roc_trans == rocblas_operation_none) ? n : m;
  std::size_t lenY = (roc_trans == rocblas_operation_none) ? m : n;

  std::size_t sizeX = 1 + (lenX - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (lenY - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDgbmv(handle, roc_trans, (int)m, (int)n, (int)KL, (int)KU, &alpha,
                    d_A, (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_hipMemcpy(Y, d_Y, sizeY * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Complex Float GBMV
void RocBlasBackend::gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                          int KU, const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          const std::complex<float> *X, int incx,
                          const std::complex<float> *beta,
                          std::complex<float> *Y, int incy) {
  const roc_complex *alpha_cuda = reinterpret_cast<const roc_complex *>(alpha);
  const roc_complex *A_cuda = reinterpret_cast<const roc_complex *>(A);
  const roc_complex *X_cuda = reinterpret_cast<const roc_complex *>(X);
  const roc_complex *beta_cuda = reinterpret_cast<const roc_complex *>(beta);
  roc_complex *Y_cuda = reinterpret_cast<roc_complex *>(Y);

  roc_complex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  rocblas_operation roc_trans = clap::to_rocblas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t lenX = (roc_trans == rocblas_operation_none) ? n : m;
  std::size_t lenY = (roc_trans == rocblas_operation_none) ? m : n;

  std::size_t sizeX = 1 + (lenX - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (lenY - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_complex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasCgbmv(handle, roc_trans, (int)m, (int)n, (int)KL, (int)KU,
                    alpha_cuda, d_A, (int)lda, d_X, (int)incx, beta_cuda, d_Y,
                    (int)incy);

  clap_hipMemcpy(Y_cuda, d_Y, sizeY * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Complex Double GBMV
void RocBlasBackend::gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                          int KU, const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          const std::complex<double> *X, int incx,
                          const std::complex<double> *beta,
                          std::complex<double> *Y, int incy) {
  const roc_DoubleComplex *alpha_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(alpha);
  const roc_DoubleComplex *A_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(A);
  const roc_DoubleComplex *X_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(X);
  const roc_DoubleComplex *beta_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(beta);
  roc_DoubleComplex *Y_cuda = reinterpret_cast<roc_DoubleComplex *>(Y);

  roc_DoubleComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  rocblas_operation roc_trans = clap::to_rocblas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t lenX = (roc_trans == rocblas_operation_none) ? n : m;
  std::size_t lenY = (roc_trans == rocblas_operation_none) ? m : n;

  std::size_t sizeX = 1 + (lenX - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (lenY - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_DoubleComplex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZgbmv(handle, roc_trans, (int)m, (int)n, (int)KL, (int)KU,
                    alpha_cuda, d_A, (int)lda, d_X, (int)incx, beta_cuda, d_Y,
                    (int)incy);

  clap_hipMemcpy(Y_cuda, d_Y, sizeY * sizeof(roc_DoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Complex Float HBMV
void RocBlasBackend::hbmv(Layout layout, Uplo uplo, int n, int k,
                          const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          const std::complex<float> *X, int incx,
                          const std::complex<float> *beta,
                          std::complex<float> *Y, int incy) {
  const roc_complex *alpha_cuda = reinterpret_cast<const roc_complex *>(alpha);
  const roc_complex *A_cuda = reinterpret_cast<const roc_complex *>(A);
  const roc_complex *X_cuda = reinterpret_cast<const roc_complex *>(X);
  const roc_complex *beta_cuda = reinterpret_cast<const roc_complex *>(beta);
  roc_complex *Y_cuda = reinterpret_cast<roc_complex *>(Y);

  roc_complex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_complex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasChbmv(handle, clap::to_rocblas_uplo(uplo), (int)n, (int)k,
                    alpha_cuda, d_A, (int)lda, d_X, (int)incx, beta_cuda, d_Y,
                    (int)incy);

  clap_hipMemcpy(Y_cuda, d_Y, sizeY * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Complex Double HBMV
void RocBlasBackend::hbmv(Layout layout, Uplo uplo, int n, int k,
                          const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          const std::complex<double> *X, int incx,
                          const std::complex<double> *beta,
                          std::complex<double> *Y, int incy) {
  const roc_DoubleComplex *alpha_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(alpha);
  const roc_DoubleComplex *A_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(A);
  const roc_DoubleComplex *X_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(X);
  const roc_DoubleComplex *beta_cuda =
      reinterpret_cast<const roc_DoubleComplex *>(beta);
  roc_DoubleComplex *Y_cuda = reinterpret_cast<roc_DoubleComplex *>(Y);

  roc_DoubleComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_DoubleComplex));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(roc_DoubleComplex));

  clap_hipMemcpy(d_A, A_cuda, sizeA * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y_cuda, sizeY * sizeof(roc_DoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZhbmv(handle, clap::to_rocblas_uplo(uplo), (int)n, (int)k,
                    alpha_cuda, d_A, (int)lda, d_X, (int)incx, beta_cuda, d_Y,
                    (int)incy);

  clap_hipMemcpy(Y_cuda, d_Y, sizeY * sizeof(roc_DoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_X);
  clap_hipFree(d_Y);
}

// Float SPMV
void RocBlasBackend::spmv(Layout layout, Uplo uplo, int n, const float alpha,
                          const float *Ap, const float *x, int incX,
                          const float beta, float *y, int incY) {
  float *d_Ap = nullptr, *d_x = nullptr, *d_y = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizex = 1 + (n - 1) * std::abs(incX);
  std::size_t sizey = 1 + (n - 1) * std::abs(incY);

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(float));
  clap_hipMalloc((void **)&d_x, sizex * sizeof(float));
  clap_hipMalloc((void **)&d_y, sizey * sizeof(float));

  clap_hipMemcpy(d_Ap, Ap, sizeAP * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_x, x, sizex * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, sizey * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasSspmv(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_Ap,
                    d_x, (int)incX, &beta, d_y, (int)incY);

  clap_hipMemcpy(y, d_y, sizey * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Double SPMV
void RocBlasBackend::spmv(Layout layout, Uplo uplo, int n, const double alpha,
                          const double *Ap, const double *x, int incX,
                          const double beta, double *y, int incY) {
  double *d_Ap = nullptr, *d_x = nullptr, *d_y = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizex = 1 + (n - 1) * std::abs(incX);
  std::size_t sizey = 1 + (n - 1) * std::abs(incY);

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(double));
  clap_hipMalloc((void **)&d_x, sizex * sizeof(double));
  clap_hipMalloc((void **)&d_y, sizey * sizeof(double));

  clap_hipMemcpy(d_Ap, Ap, sizeAP * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_x, x, sizex * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, sizey * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDspmv(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_Ap,
                    d_x, (int)incX, &beta, d_y, (int)incY);

  clap_hipMemcpy(y, d_y, sizey * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Float TPMV
void RocBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const float *Ap, float *X, int incx) {
  float *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(float));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));

  clap_hipMemcpy(d_Ap, Ap, sizeAP * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasStpmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_Ap, d_X, (int)incx);

  clap_hipMemcpy(X, d_X, sizeX * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_X);
}

// Double TPMV
void RocBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const double *Ap, double *X, int incx) {
  double *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(double));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));

  clap_hipMemcpy(d_Ap, Ap, sizeAP * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDtpmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_Ap, d_X, (int)incx);

  clap_hipMemcpy(X, d_X, sizeX * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_X);
}

// Complex Float TPMV
void RocBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const std::complex<float> *Ap,
                          std::complex<float> *X, int incx) {
  const roc_complex *Ap_roc = reinterpret_cast<const roc_complex *>(Ap);
  roc_complex *X_roc = reinterpret_cast<roc_complex *>(X);

  roc_complex *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(roc_complex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(roc_complex));

  clap_hipMemcpy(d_Ap, Ap_roc, sizeAP * sizeof(roc_complex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_roc, sizeX * sizeof(roc_complex),
                 hipMemcpyHostToDevice);

  clap_rocblasCtpmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_Ap, d_X, (int)incx);

  clap_hipMemcpy(X_roc, d_X, sizeX * sizeof(roc_complex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_X);
}

// Complex Double TPMV
void RocBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const std::complex<double> *Ap,
                          std::complex<double> *X, int incx) {
  const cuDoubleComplex *Ap_roc = reinterpret_cast<const cuDoubleComplex *>(Ap);
  cuDoubleComplex *X_roc = reinterpret_cast<cuDoubleComplex *>(X);

  cuDoubleComplex *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(cuDoubleComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));

  clap_hipMemcpy(d_Ap, Ap_roc, sizeAP * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_roc, sizeX * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZtpmv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_Ap, d_X, (int)incx);

  clap_hipMemcpy(X_roc, d_X, sizeX * sizeof(cuDoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_X);
}

// Float TPSV
void RocBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const float *Ap, float *X, int incx) {
  float *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(float));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));

  clap_hipMemcpy(d_Ap, Ap, sizeAP * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasStpsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_Ap, d_X, (int)incx);

  clap_hipMemcpy(X, d_X, sizeX * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_X);
}

// Double TPSV
void RocBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const double *Ap, double *X, int incx) {
  double *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(double));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));

  clap_hipMemcpy(d_Ap, Ap, sizeAP * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDtpsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_Ap, d_X, (int)incx);

  clap_hipMemcpy(X, d_X, sizeX * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_X);
}

// Complex Float TPSV
void RocBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const std::complex<float> *Ap,
                          std::complex<float> *X, int incx) {
  const cuComplex *Ap_cuda = reinterpret_cast<const cuComplex *>(Ap);
  cuComplex *X_cuda = reinterpret_cast<cuComplex *>(X);

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  cuComplex *d_Ap = nullptr, *d_X = nullptr;

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(cuComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(cuComplex));

  clap_hipMemcpy(d_Ap, Ap_cuda, sizeAP * sizeof(cuComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex), hipMemcpyHostToDevice);

  clap_rocblasCtpsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_Ap, d_X, (int)incx);

  clap_hipMemcpy(X_cuda, d_X, sizeX * sizeof(cuComplex), hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_X);
}

// Complex Double TPSV
void RocBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                          int n, const std::complex<double> *Ap,
                          std::complex<double> *X, int incx) {
  const cuDoubleComplex *Ap_cuda =
      reinterpret_cast<const cuDoubleComplex *>(Ap);
  cuDoubleComplex *X_cuda = reinterpret_cast<cuDoubleComplex *>(X);

  cuDoubleComplex *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(cuDoubleComplex));
  clap_hipMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));

  clap_hipMemcpy(d_Ap, Ap_cuda, sizeAP * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                 hipMemcpyHostToDevice);

  clap_rocblasZtpsv(handle, clap::to_rocblas_uplo(uplo),
                    clap::to_rocblas_trans(trans), clap::to_rocblas_diag(diag),
                    (int)n, d_Ap, d_X, (int)incx);

  clap_hipMemcpy(X_cuda, d_X, sizeX * sizeof(cuDoubleComplex),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_Ap);
  clap_hipFree(d_X);
}

// Float SPR
void RocBlasBackend::spr(Layout layout, Uplo uplo, int n, const float alpha,
                         const float *X, int incx, float *Ap) {
  float *d_X = nullptr, *d_Ap = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeAP = n * (n + 1) / 2;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(float));

  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Ap, Ap, sizeAP * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasSspr(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_Ap);

  clap_hipMemcpy(Ap, d_Ap, sizeAP * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Ap);
}

// Double SPR
void RocBlasBackend::spr(Layout layout, Uplo uplo, int n, const double alpha,
                         const double *X, int incx, double *Ap) {
  double *d_X = nullptr, *d_Ap = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeAP = n * (n + 1) / 2;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_hipMalloc((void **)&d_Ap, sizeAP * sizeof(double));

  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Ap, Ap, sizeAP * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDspr(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_Ap);

  clap_hipMemcpy(Ap, d_Ap, sizeAP * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Ap);
}

// Float SPR2
void RocBlasBackend::spr2(Layout layout, Uplo uplo, int n, const float alpha,
                          const float *X, int incx, const float *Y, int incy,
                          float *Ap) {
  float *d_X = nullptr, *d_Y = nullptr, *d_Ap = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeAp = n * (n + 1) / 2;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(float));
  clap_hipMalloc((void **)&d_Ap, sizeAp * sizeof(float));

  clap_hipMemcpy(d_X, X, sizeX * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Ap, Ap, sizeAp * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblasSspr2(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                    (int)incx, d_Y, (int)incy, d_Ap);

  clap_hipMemcpy(Ap, d_Ap, sizeAp * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_Ap);
}

// Double SPR2
void RocBlasBackend::spr2(Layout layout, Uplo uplo, int n, const double alpha,
                          const double *X, int incx, const double *Y, int incy,
                          double *Ap) {
  double *d_X = nullptr, *d_Y = nullptr, *d_Ap = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeAp = n * (n + 1) / 2;

  clap_hipMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_hipMalloc((void **)&d_Y, sizeY * sizeof(double));
  clap_hipMalloc((void **)&d_Ap, sizeAp * sizeof(double));

  clap_hipMemcpy(d_X, X, sizeX * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Y, Y, sizeY * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_Ap, Ap, sizeAp * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblasDspr2(handle, clap::to_rocblas_uplo(uplo), (int)n, &alpha, d_X,
                    (int)incx, d_Y, (int)incy, d_Ap);

  clap_hipMemcpy(Ap, d_Ap, sizeAp * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_X);
  clap_hipFree(d_Y);
  clap_hipFree(d_Ap);
}

}

