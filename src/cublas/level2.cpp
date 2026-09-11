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

#include "clap/cublas_backend.hpp"
#include "clap/cublas_utils.hpp"
#include <chrono>
using namespace std::chrono;

namespace clap {

// Float GEMV
void CuBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                         const float alpha, const float *A, int lda,
                         const float *x, int incx, const float beta, float *y,
                         int incy) {
  float *d_A = nullptr, *d_x = nullptr, *d_y = nullptr;

  cublasOperation_t cu_trans = clap::to_cublas_trans(trans);

  std::size_t sizeA = lda * n;
  std::size_t sizex = (cu_trans == CUBLAS_OP_N)
                          ? (1 + (n - 1) * std::abs(incx))
                          : (1 + (m - 1) * std::abs(incx));
  std::size_t sizey = (cu_trans == CUBLAS_OP_N)
                          ? (1 + (m - 1) * std::abs(incy))
                          : (1 + (n - 1) * std::abs(incy));

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_cudaMalloc((void **)&d_x, sizex * sizeof(float));
  clap_cudaMalloc((void **)&d_y, sizey * sizeof(float));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_x, x, sizex * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, sizey * sizeof(float), cudaMemcpyHostToDevice);

  auto start = high_resolution_clock::now();

  clap_cublasSgemv(handle, cu_trans, (int)m, (int)n, &alpha, d_A, (int)lda, d_x,
                   (int)incx, &beta, d_y, (int)incy);

  auto stop = high_resolution_clock::now();

  double sec = duration<double>(stop - start).count();

  std::cout << "Compute Time: " << sec << " sec\n";

  clap_cudaMemcpy(y, d_y, sizey * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
}

// Double GEMV
void CuBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                         const double alpha, const double *A, int lda,
                         const double *x, int incx, const double beta,
                         double *y, int incy) {
  double *d_A = nullptr, *d_x = nullptr, *d_y = nullptr;

  cublasOperation_t cu_trans = clap::to_cublas_trans(trans);

  std::size_t sizeA = lda * n;
  std::size_t sizex = (cu_trans == CUBLAS_OP_N)
                          ? (1 + (n - 1) * std::abs(incx))
                          : (1 + (m - 1) * std::abs(incx));
  std::size_t sizey = (cu_trans == CUBLAS_OP_N)
                          ? (1 + (m - 1) * std::abs(incy))
                          : (1 + (n - 1) * std::abs(incy));

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_cudaMalloc((void **)&d_x, sizex * sizeof(double));
  clap_cudaMalloc((void **)&d_y, sizey * sizeof(double));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_x, x, sizex * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, sizey * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDgemv(handle, cu_trans, (int)m, (int)n, &alpha, d_A, (int)lda, d_x,
                   (int)incx, &beta, d_y, (int)incy);

  clap_cudaMemcpy(y, d_y, sizey * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
}

// Complex Float GEMV
void CuBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                         const std::complex<float> *alpha,
                         const std::complex<float> *A, int lda,
                         const std::complex<float> *X, int incx,
                         const std::complex<float> *beta,
                         std::complex<float> *Y, int incy) {

  const cuComplex *alpha_cuda = reinterpret_cast<const cuComplex *>(alpha);
  const cuComplex *beta_cuda = reinterpret_cast<const cuComplex *>(beta);
  const cuComplex *A_cuda = reinterpret_cast<const cuComplex *>(A);
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  cuComplex *Y_cuda = reinterpret_cast<cuComplex *>(Y);

  cuComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  cublasOperation_t cu_trans = clap::to_cublas_trans(trans);

  std::size_t sizeA = lda * n;
  std::size_t sizeX = (cu_trans == CUBLAS_OP_N)
                          ? (1 + (n - 1) * std::abs(incx))
                          : (1 + (m - 1) * std::abs(incx));
  std::size_t sizeY = (cu_trans == CUBLAS_OP_N)
                          ? (1 + (m - 1) * std::abs(incy))
                          : (1 + (n - 1) * std::abs(incy));

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasCgemv(handle, cu_trans, (int)m, (int)n, alpha_cuda, d_A, (int)lda,
                   d_X, (int)incx, beta_cuda, d_Y, (int)incy);

  clap_cudaMemcpy(Y_cuda, d_Y, sizeY * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Complex Double GEMV
void CuBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                         const std::complex<double> *alpha,
                         const std::complex<double> *A, int lda,
                         const std::complex<double> *X, int incx,
                         const std::complex<double> *beta,
                         std::complex<double> *Y, int incy) {
  const cuDoubleComplex *alpha_cuda =
      reinterpret_cast<const cuDoubleComplex *>(alpha);
  const cuDoubleComplex *beta_cuda =
      reinterpret_cast<const cuDoubleComplex *>(beta);
  const cuDoubleComplex *A_cuda = reinterpret_cast<const cuDoubleComplex *>(A);
  const cuDoubleComplex *X_cuda = reinterpret_cast<const cuDoubleComplex *>(X);
  cuDoubleComplex *Y_cuda = reinterpret_cast<cuDoubleComplex *>(Y);

  cuDoubleComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  cublasOperation_t cu_trans = clap::to_cublas_trans(trans);

  std::size_t sizeA = lda * n;
  std::size_t sizeX = (cu_trans == CUBLAS_OP_N)
                          ? (1 + (n - 1) * std::abs(incx))
                          : (1 + (m - 1) * std::abs(incx));
  std::size_t sizeY = (cu_trans == CUBLAS_OP_N)
                          ? (1 + (m - 1) * std::abs(incy))
                          : (1 + (n - 1) * std::abs(incy));

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZgemv(handle, cu_trans, (int)m, (int)n, alpha_cuda, d_A, (int)lda,
                   d_X, (int)incx, beta_cuda, d_Y, (int)incy);

  clap_cudaMemcpy(Y_cuda, d_Y, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Float SYMV
void CuBlasBackend::symv(Layout layout, Uplo uplo, int n, const float alpha,
                         const float *A, int lda, const float *X, int incx,
                         const float beta, float *Y, int incy) {
  float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(float));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSsymv(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_A,
                   (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_cudaMemcpy(Y, d_Y, sizeY * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Double SYMV
void CuBlasBackend::symv(Layout layout, Uplo uplo, int n, const double alpha,
                         const double *A, int lda, const double *X, int incx,
                         const double beta, double *Y, int incy) {
  double *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(double));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(double), cudaMemcpyHostToDevice);

  auto start = high_resolution_clock::now();

  clap_cublasDsymv(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_A,
                   (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  auto stop = high_resolution_clock::now();

  double sec = duration<double>(stop - start).count();

  std::cout << "Compute Time: " << sec << " sec\n";

  clap_cudaMemcpy(Y, d_Y, sizeY * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Complex Float HEMV
void CuBlasBackend::hemv(Layout layout, Uplo uplo, int n,
                         const std::complex<float> *alpha,
                         const std::complex<float> *A, int lda,
                         const std::complex<float> *X, int incx,
                         const std::complex<float> *beta,
                         std::complex<float> *Y, int incy) {
  const cuComplex *alpha_cuda = reinterpret_cast<const cuComplex *>(alpha);
  const cuComplex *A_cuda = reinterpret_cast<const cuComplex *>(A);
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  const cuComplex *beta_cuda = reinterpret_cast<const cuComplex *>(beta);
  cuComplex *Y_cuda = reinterpret_cast<cuComplex *>(Y);

  cuComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasChemv(handle, clap::to_cublas_uplo(uplo), (int)n, alpha_cuda, d_A,
                   (int)lda, d_X, (int)incx, beta_cuda, d_Y, (int)incy);

  clap_cudaMemcpy(Y_cuda, d_Y, sizeY * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Complex Double HEMV
void CuBlasBackend::hemv(Layout layout, Uplo uplo, int n,
                         const std::complex<double> *alpha,
                         const std::complex<double> *A, int lda,
                         const std::complex<double> *X, int incx,
                         const std::complex<double> *beta,
                         std::complex<double> *Y, int incy) {
  const cuDoubleComplex *alpha_cuda =
      reinterpret_cast<const cuDoubleComplex *>(alpha);
  const cuDoubleComplex *A_cuda = reinterpret_cast<const cuDoubleComplex *>(A);
  const cuDoubleComplex *X_cuda = reinterpret_cast<const cuDoubleComplex *>(X);
  const cuDoubleComplex *beta_cuda =
      reinterpret_cast<const cuDoubleComplex *>(beta);
  cuDoubleComplex *Y_cuda = reinterpret_cast<cuDoubleComplex *>(Y);

  cuDoubleComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZhemv(handle, clap::to_cublas_uplo(uplo), (int)n, alpha_cuda, d_A,
                   (int)lda, d_X, (int)incx, beta_cuda, d_Y, (int)incy);

  clap_cudaMemcpy(Y_cuda, d_Y, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Float TRMV
void CuBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const float *A, int lda, float *X, int incx) {
  float *d_A = nullptr;
  float *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasStrmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Double TRMV
void CuBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const double *A, int lda, double *X, int incx) {
  double *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDtrmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Complex Float TRMV
void CuBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const std::complex<float> *A, int lda,
                         std::complex<float> *X, int incx) {
  const cuComplex *A_cuda = reinterpret_cast<const cuComplex *>(A);
  cuComplex *X_cuda = reinterpret_cast<cuComplex *>(X);

  cuComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasCtrmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Complex Double TRMV
void CuBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const std::complex<double> *A, int lda,
                         std::complex<double> *X, int incx) {
  const cuDoubleComplex *A_cuda = reinterpret_cast<const cuDoubleComplex *>(A);
  cuDoubleComplex *X_cuda = reinterpret_cast<cuDoubleComplex *>(X);

  cuDoubleComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZtrmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Float TRSV
void CuBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const float *A, int lda, float *X, int incx) {
  float *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasStrsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Double TRSV
void CuBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const double *A, int lda, double *X, int incx) {
  double *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDtrsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Complex Float TRSV
void CuBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const std::complex<float> *A, int lda,
                         std::complex<float> *X, int incx) {
  const cuComplex *A_cuda = reinterpret_cast<const cuComplex *>(A);
  cuComplex *X_cuda = reinterpret_cast<cuComplex *>(X);

  cuComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasCtrsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Complex Double TRSV
void CuBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const std::complex<double> *A, int lda,
                         std::complex<double> *X, int incx) {
  const cuDoubleComplex *A_cuda = reinterpret_cast<const cuDoubleComplex *>(A);
  cuDoubleComplex *X_cuda = reinterpret_cast<cuDoubleComplex *>(X);

  cuDoubleComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZtrsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Complex Float GERC
void CuBlasBackend::gerc(Layout layout, int m, int n,
                         const std::complex<float> *alpha,
                         const std::complex<float> *X, int incx,
                         const std::complex<float> *Y, int incy,
                         std::complex<float> *A, int lda) {
  const cuComplex *alpha_cuda = reinterpret_cast<const cuComplex *>(alpha);
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  const cuComplex *Y_cuda = reinterpret_cast<const cuComplex *>(Y);
  cuComplex *A_cuda = reinterpret_cast<cuComplex *>(A);

  cuComplex *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (m - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasCgerc(handle, (int)m, (int)n, alpha_cuda, d_X, (int)incx, d_Y,
                   (int)incy, d_A, (int)lda);

  clap_cudaMemcpy(A_cuda, d_A, sizeA * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_A);
}

// Complex Double GERC
void CuBlasBackend::gerc(Layout layout, int m, int n,
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

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZgerc(handle, (int)m, (int)n, alpha_cuda, d_X, (int)incx, d_Y,
                   (int)incy, d_A, (int)lda);

  clap_cudaMemcpy(A_cuda, d_A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_A);
}

// Float GER
void CuBlasBackend::ger(Layout layout, int m, int n, const float alpha,
                        const float *X, int incx, const float *Y, int incy,
                        float *A, int lda) {
  float *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (m - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(float));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(float));

  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSger(handle, (int)m, (int)n, &alpha, d_X, (int)incx, d_Y,
                  (int)incy, d_A, (int)lda);

  clap_cudaMemcpy(A, d_A, sizeA * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_A);
}

// Double GER
void CuBlasBackend::ger(Layout layout, int m, int n, const double alpha,
                        const double *X, int incx, const double *Y, int incy,
                        double *A, int lda) {
  double *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (m - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(double));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(double));

  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDger(handle, (int)m, (int)n, &alpha, d_X, (int)incx, d_Y,
                  (int)incy, d_A, (int)lda);

  clap_cudaMemcpy(A, d_A, sizeA * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_A);
}

// Complex Float GERU
void CuBlasBackend::geru(Layout layout, int m, int n,
                         const std::complex<float> *alpha,
                         const std::complex<float> *X, int incx,
                         const std::complex<float> *Y, int incy,
                         std::complex<float> *A, int lda) {
  const cuComplex *alpha_cuda = reinterpret_cast<const cuComplex *>(alpha);
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  const cuComplex *Y_cuda = reinterpret_cast<const cuComplex *>(Y);
  cuComplex *A_cuda = reinterpret_cast<cuComplex *>(A);

  cuComplex *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (m - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasCgeru(handle, (int)m, (int)n, alpha_cuda, d_X, (int)incx, d_Y,
                   (int)incy, d_A, (int)lda);

  clap_cudaMemcpy(A_cuda, d_A, sizeA * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_A);
}

// Complex Double GERU
void CuBlasBackend::geru(Layout layout, int m, int n,
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

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZgeru(handle, (int)m, (int)n, alpha_cuda, d_X, (int)incx, d_Y,
                   (int)incy, d_A, (int)lda);

  clap_cudaMemcpy(A_cuda, d_A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_A);
}

// Float SYR
void CuBlasBackend::syr(Layout layout, Uplo uplo, int n, const float alpha,
                        const float *X, int incx, float *A, int lda) {
  float *d_X = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeA = lda * n;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(float));

  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSsyr(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                  (int)incx, d_A, (int)lda);

  clap_cudaMemcpy(A, d_A, sizeA * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_A);
}

// Double SYR
void CuBlasBackend::syr(Layout layout, Uplo uplo, int n, const double alpha,
                        const double *X, int incx, double *A, int lda) {
  double *d_X = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeA = lda * n;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(double));

  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDsyr(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                  (int)incx, d_A, (int)lda);

  clap_cudaMemcpy(A, d_A, sizeA * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_A);
}

// Float SYR2
void CuBlasBackend::syr2(Layout layout, Uplo uplo, int n, const float alpha,
                         const float *X, int incx, const float *Y, int incy,
                         float *A, int lda) {
  float *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(float));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(float));

  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSsyr2(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_Y, (int)incy, d_A, (int)lda);

  clap_cudaMemcpy(A, d_A, sizeA * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_A);
}

// Double SYR2
void CuBlasBackend::syr2(Layout layout, Uplo uplo, int n, const double alpha,
                         const double *X, int incx, const double *Y, int incy,
                         double *A, int lda) {
  double *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(double));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(double));

  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDsyr2(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_Y, (int)incy, d_A, (int)lda);

  clap_cudaMemcpy(A, d_A, sizeA * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_A);
}

// Complex Float HER
void CuBlasBackend::her(Layout layout, Uplo uplo, int n, const float alpha,
                        const std::complex<float> *X, int incx,
                        std::complex<float> *A, int lda) {
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  cuComplex *A_cuda = reinterpret_cast<cuComplex *>(A);

  cuComplex *d_X = nullptr;
  cuComplex *d_A = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeA = lda * n;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasCher(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                  (int)incx, d_A, (int)lda);

  clap_cudaMemcpy(A_cuda, d_A, sizeA * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_A);
}

// Complex Double HER
void CuBlasBackend::her(Layout layout, Uplo uplo, int n, const double alpha,
                        const std::complex<double> *X, int incx,
                        std::complex<double> *A, int lda) {
  const cuDoubleComplex *X_cuda = reinterpret_cast<const cuDoubleComplex *>(X);
  cuDoubleComplex *A_cuda = reinterpret_cast<cuDoubleComplex *>(A);

  cuDoubleComplex *d_X = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeA = lda * n;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZher(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                  (int)incx, d_A, (int)lda);

  clap_cudaMemcpy(A_cuda, d_A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_A);
}

// Complex Float HER2
void CuBlasBackend::her2(Layout layout, Uplo uplo, int n,
                         const std::complex<float> *alpha,
                         const std::complex<float> *X, int incx,
                         const std::complex<float> *Y, int incy,
                         std::complex<float> *A, int lda) {
  const cuComplex *alpha_cuda = reinterpret_cast<const cuComplex *>(alpha);
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  const cuComplex *Y_cuda = reinterpret_cast<const cuComplex *>(Y);
  cuComplex *A_cuda = reinterpret_cast<cuComplex *>(A);

  cuComplex *d_X = nullptr, *d_Y = nullptr, *d_A = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeA = lda * n;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasCher2(handle, clap::to_cublas_uplo(uplo), (int)n, alpha_cuda, d_X,
                   (int)incx, d_Y, (int)incy, d_A, (int)lda);

  clap_cudaMemcpy(A_cuda, d_A, sizeA * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_A);
}

// Complex Double HER2
void CuBlasBackend::her2(Layout layout, Uplo uplo, int n,
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

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZher2(handle, clap::to_cublas_uplo(uplo), (int)n, alpha_cuda, d_X,
                   (int)incx, d_Y, (int)incy, d_A, (int)lda);

  clap_cudaMemcpy(A_cuda, d_A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_A);
}

// Complex Float HPR
void CuBlasBackend::hpr(Layout layout, Uplo uplo, int n, const float alpha,
                        const std::complex<float> *X, int incx,
                        std::complex<float> *AP) {
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  cuComplex *AP_cuda = reinterpret_cast<cuComplex *>(AP);

  int Ps = n * (n + 1) / 2;

  cuComplex *d_X = nullptr, *d_AP = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_AP, Ps * sizeof(cuComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_AP, AP_cuda, Ps * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasChpr(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                  (int)incx, d_AP);

  clap_cudaMemcpy(AP_cuda, d_AP, Ps * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_AP);
}

// Complex Double HPR
void CuBlasBackend::hpr(Layout layout, Uplo uplo, int n, const double alpha,
                        const std::complex<double> *X, int incx,
                        std::complex<double> *AP) {

  const cuDoubleComplex *X_cuda = reinterpret_cast<const cuDoubleComplex *>(X);
  cuDoubleComplex *AP_cuda = reinterpret_cast<cuDoubleComplex *>(AP);

  int Ps = n * (n + 1) / 2;

  cuDoubleComplex *d_X = nullptr, *d_AP = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_AP, Ps * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_AP, AP_cuda, Ps * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZhpr(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                  (int)incx, d_AP);

  clap_cudaMemcpy(AP_cuda, d_AP, Ps * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_AP);
}

// Complex Float HPR2
void CuBlasBackend::hpr2(Layout layout, Uplo uplo, int n,
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

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_AP, Ps * sizeof(cuComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_AP, AP_cuda, Ps * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasChpr2(handle, clap::to_cublas_uplo(uplo), (int)n, alpha_cuda, d_X,
                   (int)incx, d_Y, (int)incy, d_AP);

  clap_cudaMemcpy(AP_cuda, d_AP, Ps * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_AP);
}

// Complex Double HPR2
void CuBlasBackend::hpr2(Layout layout, Uplo uplo, int n,
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

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_AP, Ps * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_AP, AP_cuda, Ps * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZhpr2(handle, clap::to_cublas_uplo(uplo), (int)n, alpha_cuda, d_X,
                   (int)incx, d_Y, (int)incy, d_AP);

  clap_cudaMemcpy(AP_cuda, d_AP, Ps * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_AP);
}

// Float GBMV
void CuBlasBackend::gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                         int KU, const float alpha, const float *A, int lda,
                         const float *X, int incx, const float beta, float *Y,
                         int incy) {
  float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  cublasOperation_t cu_trans = clap::to_cublas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t lenX = (cu_trans == CUBLAS_OP_N) ? n : m;
  std::size_t lenY = (cu_trans == CUBLAS_OP_N) ? m : n;

  std::size_t sizeX = 1 + (lenX - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (lenY - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(float));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSgbmv(handle, cu_trans, (int)m, (int)n, (int)KL, (int)KU, &alpha,
                   d_A, (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_cudaMemcpy(Y, d_Y, sizeY * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Double GBMV
void CuBlasBackend::gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                         int KU, const double alpha, const double *A, int lda,
                         const double *X, int incx, const double beta,
                         double *Y, int incy) {
  double *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  cublasOperation_t cu_trans = clap::to_cublas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t lenX = (cu_trans == CUBLAS_OP_N) ? n : m;
  std::size_t lenY = (cu_trans == CUBLAS_OP_N) ? m : n;

  std::size_t sizeX = 1 + (lenX - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (lenY - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(double));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDgbmv(handle, cu_trans, (int)m, (int)n, (int)KL, (int)KU, &alpha,
                   d_A, (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_cudaMemcpy(Y, d_Y, sizeY * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Complex Float GBMV
void CuBlasBackend::gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                         int KU, const std::complex<float> *alpha,
                         const std::complex<float> *A, int lda,
                         const std::complex<float> *X, int incx,
                         const std::complex<float> *beta,
                         std::complex<float> *Y, int incy) {
  const cuComplex *alpha_cuda = reinterpret_cast<const cuComplex *>(alpha);
  const cuComplex *A_cuda = reinterpret_cast<const cuComplex *>(A);
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  const cuComplex *beta_cuda = reinterpret_cast<const cuComplex *>(beta);
  cuComplex *Y_cuda = reinterpret_cast<cuComplex *>(Y);

  cuComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  cublasOperation_t cu_trans = clap::to_cublas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t lenX = (cu_trans == CUBLAS_OP_N) ? n : m;
  std::size_t lenY = (cu_trans == CUBLAS_OP_N) ? m : n;

  std::size_t sizeX = 1 + (lenX - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (lenY - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasCgbmv(handle, cu_trans, (int)m, (int)n, (int)KL, (int)KU,
                   alpha_cuda, d_A, (int)lda, d_X, (int)incx, beta_cuda, d_Y,
                   (int)incy);

  clap_cudaMemcpy(Y_cuda, d_Y, sizeY * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Complex Double GBMV
void CuBlasBackend::gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                         int KU, const std::complex<double> *alpha,
                         const std::complex<double> *A, int lda,
                         const std::complex<double> *X, int incx,
                         const std::complex<double> *beta,
                         std::complex<double> *Y, int incy) {
  const cuDoubleComplex *alpha_cuda =
      reinterpret_cast<const cuDoubleComplex *>(alpha);
  const cuDoubleComplex *A_cuda = reinterpret_cast<const cuDoubleComplex *>(A);
  const cuDoubleComplex *X_cuda = reinterpret_cast<const cuDoubleComplex *>(X);
  const cuDoubleComplex *beta_cuda =
      reinterpret_cast<const cuDoubleComplex *>(beta);
  cuDoubleComplex *Y_cuda = reinterpret_cast<cuDoubleComplex *>(Y);

  cuDoubleComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  cublasOperation_t cu_trans = clap::to_cublas_trans(trans);

  std::size_t sizeA = lda * n;

  std::size_t lenX = (cu_trans == CUBLAS_OP_N) ? n : m;
  std::size_t lenY = (cu_trans == CUBLAS_OP_N) ? m : n;

  std::size_t sizeX = 1 + (lenX - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (lenY - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZgbmv(handle, cu_trans, (int)m, (int)n, (int)KL, (int)KU,
                   alpha_cuda, d_A, (int)lda, d_X, (int)incx, beta_cuda, d_Y,
                   (int)incy);

  clap_cudaMemcpy(Y_cuda, d_Y, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Complex Float HBMV
void CuBlasBackend::hbmv(Layout layout, Uplo uplo, int n, int k,
                         const std::complex<float> *alpha,
                         const std::complex<float> *A, int lda,
                         const std::complex<float> *X, int incx,
                         const std::complex<float> *beta,
                         std::complex<float> *Y, int incy) {
  const cuComplex *alpha_cuda = reinterpret_cast<const cuComplex *>(alpha);
  const cuComplex *A_cuda = reinterpret_cast<const cuComplex *>(A);
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  const cuComplex *beta_cuda = reinterpret_cast<const cuComplex *>(beta);
  cuComplex *Y_cuda = reinterpret_cast<cuComplex *>(Y);

  cuComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasChbmv(handle, clap::to_cublas_uplo(uplo), (int)n, (int)k,
                   alpha_cuda, d_A, (int)lda, d_X, (int)incx, beta_cuda, d_Y,
                   (int)incy);

  clap_cudaMemcpy(Y_cuda, d_Y, sizeY * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Complex Double HBMV
void CuBlasBackend::hbmv(Layout layout, Uplo uplo, int n, int k,
                         const std::complex<double> *alpha,
                         const std::complex<double> *A, int lda,
                         const std::complex<double> *X, int incx,
                         const std::complex<double> *beta,
                         std::complex<double> *Y, int incy) {
  const cuDoubleComplex *alpha_cuda =
      reinterpret_cast<const cuDoubleComplex *>(alpha);
  const cuDoubleComplex *A_cuda = reinterpret_cast<const cuDoubleComplex *>(A);
  const cuDoubleComplex *X_cuda = reinterpret_cast<const cuDoubleComplex *>(X);
  const cuDoubleComplex *beta_cuda =
      reinterpret_cast<const cuDoubleComplex *>(beta);
  cuDoubleComplex *Y_cuda = reinterpret_cast<cuDoubleComplex *>(Y);

  cuDoubleComplex *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZhbmv(handle, clap::to_cublas_uplo(uplo), (int)n, (int)k,
                   alpha_cuda, d_A, (int)lda, d_X, (int)incx, beta_cuda, d_Y,
                   (int)incy);

  clap_cudaMemcpy(Y_cuda, d_Y, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Float SBMV
void CuBlasBackend::sbmv(Layout layout, Uplo uplo, int n, int k,
                         const float alpha, const float *A, int lda,
                         const float *X, int incx, const float beta, float *Y,
                         int incy) {
  float *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(float));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSsbmv(handle, clap::to_cublas_uplo(uplo), (int)n, (int)k, &alpha,
                   d_A, (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_cudaMemcpy(Y, d_Y, sizeY * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Double SBMV
void CuBlasBackend::sbmv(Layout layout, Uplo uplo, int n, int k,
                         const double alpha, const double *A, int lda,
                         const double *X, int incx, const double beta,
                         double *Y, int incy) {
  double *d_A = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(double));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(double), cudaMemcpyHostToDevice);

  clap_cudaDeviceSynchronize();

  auto start = high_resolution_clock::now();

  clap_cublasDsbmv(handle, clap::to_cublas_uplo(uplo), (int)n, (int)k, &alpha,
                   d_A, (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_cudaDeviceSynchronize();

  auto stop = high_resolution_clock::now();

  double sec = duration<double>(stop - start).count();

  std::cout << "Compute Time: " << sec << " sec\n";

  clap_cudaMemcpy(Y, d_Y, sizeY * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Float TBMV
void CuBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, int k, const float *A, int lda, float *X,
                         int incx) {
  float *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasStbmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Double TBMV
void CuBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, int k, const double *A, int lda, double *X,
                         int incx) {
  double *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDtbmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Complex Float TBMV
void CuBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, int k, const std::complex<float> *A, int lda,
                         std::complex<float> *X, int incx) {
  const cuComplex *A_cuda = reinterpret_cast<const cuComplex *>(A);
  cuComplex *X_cuda = reinterpret_cast<cuComplex *>(X);

  cuComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cudaDeviceSynchronize();

  auto start = high_resolution_clock::now();

  clap_cublasCtbmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_cudaDeviceSynchronize();

  auto stop = high_resolution_clock::now();

  double sec = duration<double>(stop - start).count();

  std::cout << "Compute Time: " << sec << " sec\n";

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Complex Double TBMV
void CuBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, int k, const std::complex<double> *A, int lda,
                         std::complex<double> *X, int incx) {
  const cuDoubleComplex *A_cuda = reinterpret_cast<const cuDoubleComplex *>(A);
  cuDoubleComplex *X_cuda = reinterpret_cast<cuDoubleComplex *>(X);

  cuDoubleComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZtbmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);
  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Float TBSV
void CuBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, int k, const float *A, int lda, float *X,
                         int incx) {
  float *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasStbsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Double TBSV
void CuBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, int k, const double *A, int lda, double *X,
                         int incx) {
  double *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));

  clap_cudaMemcpy(d_A, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDtbsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Complex Float TBSV
void CuBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, int k, const std::complex<float> *A, int lda,
                         std::complex<float> *X, int incx) {
  const cuComplex *A_cuda = reinterpret_cast<const cuComplex *>(A);
  cuComplex *X_cuda = reinterpret_cast<cuComplex *>(X);

  cuComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasCtbsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Complex Double TBSV
void CuBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, int k, const std::complex<double> *A, int lda,
                         std::complex<double> *X, int incx) {
  const cuDoubleComplex *A_cuda = reinterpret_cast<const cuDoubleComplex *>(A);
  cuDoubleComplex *X_cuda = reinterpret_cast<cuDoubleComplex *>(X);

  cuDoubleComplex *d_A = nullptr, *d_X = nullptr;

  std::size_t sizeA = lda * n;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_A, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_A, A_cuda, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZtbsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, (int)k, d_A, (int)lda, d_X, (int)incx);

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
}

// Complex Float HPMV
void CuBlasBackend::hpmv(Layout layout, Uplo uplo, int n,
                         const std::complex<float> *alpha,
                         const std::complex<float> *Ap,
                         const std::complex<float> *X, int incx,
                         const std::complex<float> *beta,
                         std::complex<float> *Y, int incy) {
  const cuComplex *alpha_cuda = reinterpret_cast<const cuComplex *>(alpha);
  const cuComplex *Ap_cuda = reinterpret_cast<const cuComplex *>(Ap);
  const cuComplex *X_cuda = reinterpret_cast<const cuComplex *>(X);
  const cuComplex *beta_cuda = reinterpret_cast<const cuComplex *>(beta);
  cuComplex *Y_cuda = reinterpret_cast<cuComplex *>(Y);

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  cuComplex *d_Ap = nullptr, *d_X = nullptr, *d_Y = nullptr;

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuComplex));

  clap_cudaMemcpy(d_Ap, Ap_cuda, sizeAP * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cudaDeviceSynchronize();

  auto start = high_resolution_clock::now();

  clap_cublasChpmv(handle, clap::to_cublas_uplo(uplo), (int)n, alpha_cuda, d_Ap,
                   d_X, (int)incx, beta_cuda, d_Y, (int)incy);

  clap_cudaDeviceSynchronize();

  auto stop = high_resolution_clock::now();

  double sec = duration<double>(stop - start).count();

  std::cout << "Compute Time: " << sec << " sec\n";

  clap_cudaMemcpy(Y_cuda, d_Y, sizeY * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Complex Double HPMV
void CuBlasBackend::hpmv(Layout layout, Uplo uplo, int n,
                         const std::complex<double> *alpha,
                         const std::complex<double> *Ap,
                         const std::complex<double> *X, int incx,
                         const std::complex<double> *beta,
                         std::complex<double> *Y, int incy) {
  const cuDoubleComplex *alpha_cuda =
      reinterpret_cast<const cuDoubleComplex *>(alpha);
  const cuDoubleComplex *Ap_cuda =
      reinterpret_cast<const cuDoubleComplex *>(Ap);
  const cuDoubleComplex *X_cuda = reinterpret_cast<const cuDoubleComplex *>(X);
  const cuDoubleComplex *beta_cuda =
      reinterpret_cast<const cuDoubleComplex *>(beta);
  cuDoubleComplex *Y_cuda = reinterpret_cast<cuDoubleComplex *>(Y);

  cuDoubleComplex *d_Ap = nullptr, *d_X = nullptr, *d_Y = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_Ap, Ap_cuda, sizeAP * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y_cuda, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZhpmv(handle, clap::to_cublas_uplo(uplo), (int)n, alpha_cuda, d_Ap,
                   d_X, (int)incx, beta_cuda, d_Y, (int)incy);

  clap_cudaMemcpy(Y_cuda, d_Y, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Float SPMV
void CuBlasBackend::spmv(Layout layout, Uplo uplo, int n, const float alpha,
                         const float *Ap, const float *x, int incX,
                         const float beta, float *y, int incY) {
  float *d_Ap = nullptr, *d_x = nullptr, *d_y = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizex = 1 + (n - 1) * std::abs(incX);
  std::size_t sizey = 1 + (n - 1) * std::abs(incY);

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(float));
  clap_cudaMalloc((void **)&d_x, sizex * sizeof(float));
  clap_cudaMalloc((void **)&d_y, sizey * sizeof(float));

  clap_cudaMemcpy(d_Ap, Ap, sizeAP * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_x, x, sizex * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, sizey * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSspmv(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_Ap,
                   d_x, (int)incX, &beta, d_y, (int)incY);

  clap_cudaMemcpy(y, d_y, sizey * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
}

// Double SPMV
void CuBlasBackend::spmv(Layout layout, Uplo uplo, int n, const double alpha,
                         const double *Ap, const double *x, int incX,
                         const double beta, double *y, int incY) {
  double *d_Ap = nullptr, *d_x = nullptr, *d_y = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizex = 1 + (n - 1) * std::abs(incX);
  std::size_t sizey = 1 + (n - 1) * std::abs(incY);

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(double));
  clap_cudaMalloc((void **)&d_x, sizex * sizeof(double));
  clap_cudaMalloc((void **)&d_y, sizey * sizeof(double));

  clap_cudaMemcpy(d_Ap, Ap, sizeAP * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_x, x, sizex * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, sizey * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDspmv(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_Ap,
                   d_x, (int)incX, &beta, d_y, (int)incY);

  clap_cudaMemcpy(y, d_y, sizey * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
}

// Float TPMV
void CuBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const float *Ap, float *X, int incx) {
  float *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(float));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));

  clap_cudaMemcpy(d_Ap, Ap, sizeAP * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasStpmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_Ap, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_X);
}

// Double TPMV
void CuBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const double *Ap, double *X, int incx) {
  double *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(double));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));

  clap_cudaMemcpy(d_Ap, Ap, sizeAP * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDtpmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_Ap, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_X);
}

// Complex Float TPMV
void CuBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const std::complex<float> *Ap,
                         std::complex<float> *X, int incx) {
  const cuComplex *Ap_cuda = reinterpret_cast<const cuComplex *>(Ap);
  cuComplex *X_cuda = reinterpret_cast<cuComplex *>(X);

  cuComplex *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));

  clap_cudaMemcpy(d_Ap, Ap_cuda, sizeAP * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasCtpmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_Ap, d_X, (int)incx);

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_X);
}

// Complex Double TPMV
void CuBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const std::complex<double> *Ap,
                         std::complex<double> *X, int incx) {
  const cuDoubleComplex *Ap_cuda =
      reinterpret_cast<const cuDoubleComplex *>(Ap);
  cuDoubleComplex *X_cuda = reinterpret_cast<cuDoubleComplex *>(X);

  cuDoubleComplex *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_Ap, Ap_cuda, sizeAP * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZtpmv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_Ap, d_X, (int)incx);

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_X);
}

// Float TPSV
void CuBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const float *Ap, float *X, int incx) {
  float *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(float));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));

  clap_cudaMemcpy(d_Ap, Ap, sizeAP * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasStpsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_Ap, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_X);
}

// Double TPSV
void CuBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const double *Ap, double *X, int incx) {
  double *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(double));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));

  clap_cudaMemcpy(d_Ap, Ap, sizeAP * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDtpsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_Ap, d_X, (int)incx);

  clap_cudaMemcpy(X, d_X, sizeX * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_X);
}

// Complex Float TPSV
void CuBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const std::complex<float> *Ap,
                         std::complex<float> *X, int incx) {
  const cuComplex *Ap_cuda = reinterpret_cast<const cuComplex *>(Ap);
  cuComplex *X_cuda = reinterpret_cast<cuComplex *>(X);

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  cuComplex *d_Ap = nullptr, *d_X = nullptr;

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuComplex));

  clap_cudaMemcpy(d_Ap, Ap_cuda, sizeAP * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasCtpsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_Ap, d_X, (int)incx);

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_X);
}

// Complex Double TPSV
void CuBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int n, const std::complex<double> *Ap,
                         std::complex<double> *X, int incx) {
  const cuDoubleComplex *Ap_cuda =
      reinterpret_cast<const cuDoubleComplex *>(Ap);
  cuDoubleComplex *X_cuda = reinterpret_cast<cuDoubleComplex *>(X);

  cuDoubleComplex *d_Ap = nullptr, *d_X = nullptr;

  std::size_t sizeAP = n * (n + 1) / 2;
  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);

  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(d_Ap, Ap_cuda, sizeAP * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_X, X_cuda, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZtpsv(handle, clap::to_cublas_uplo(uplo),
                   clap::to_cublas_trans(trans), clap::to_cublas_diag(diag),
                   (int)n, d_Ap, d_X, (int)incx);

  clap_cudaMemcpy(X_cuda, d_X, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(d_Ap);
  clap_cudaFree(d_X);
}

// Float SPR
void CuBlasBackend::spr(Layout layout, Uplo uplo, int n, const float alpha,
                        const float *X, int incx, float *Ap) {
  float *d_X = nullptr, *d_Ap = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeAP = n * (n + 1) / 2;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(float));

  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Ap, Ap, sizeAP * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSspr(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                  (int)incx, d_Ap);

  clap_cudaMemcpy(Ap, d_Ap, sizeAP * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Ap);
}

// Double SPR
void CuBlasBackend::spr(Layout layout, Uplo uplo, int n, const double alpha,
                        const double *X, int incx, double *Ap) {
  double *d_X = nullptr, *d_Ap = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeAP = n * (n + 1) / 2;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_cudaMalloc((void **)&d_Ap, sizeAP * sizeof(double));

  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Ap, Ap, sizeAP * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDspr(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                  (int)incx, d_Ap);

  clap_cudaMemcpy(Ap, d_Ap, sizeAP * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Ap);
}

// Float SPR2
void CuBlasBackend::spr2(Layout layout, Uplo uplo, int n, const float alpha,
                         const float *X, int incx, const float *Y, int incy,
                         float *Ap) {
  float *d_X = nullptr, *d_Y = nullptr, *d_Ap = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeAp = n * (n + 1) / 2;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(float));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(float));
  clap_cudaMalloc((void **)&d_Ap, sizeAp * sizeof(float));

  clap_cudaMemcpy(d_X, X, sizeX * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Ap, Ap, sizeAp * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSspr2(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_Y, (int)incy, d_Ap);

  clap_cudaMemcpy(Ap, d_Ap, sizeAp * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_Ap);
}

// Double SPR2
void CuBlasBackend::spr2(Layout layout, Uplo uplo, int n, const double alpha,
                         const double *X, int incx, const double *Y, int incy,
                         double *Ap) {
  double *d_X = nullptr, *d_Y = nullptr, *d_Ap = nullptr;

  std::size_t sizeX = 1 + (n - 1) * std::abs(incx);
  std::size_t sizeY = 1 + (n - 1) * std::abs(incy);
  std::size_t sizeAp = n * (n + 1) / 2;

  clap_cudaMalloc((void **)&d_X, sizeX * sizeof(double));
  clap_cudaMalloc((void **)&d_Y, sizeY * sizeof(double));
  clap_cudaMalloc((void **)&d_Ap, sizeAp * sizeof(double));

  clap_cudaMemcpy(d_X, X, sizeX * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Y, Y, sizeY * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_Ap, Ap, sizeAp * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDspr2(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_X,
                   (int)incx, d_Y, (int)incy, d_Ap);

  clap_cudaMemcpy(Ap, d_Ap, sizeAp * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
  clap_cudaFree(d_Ap);
}
}

