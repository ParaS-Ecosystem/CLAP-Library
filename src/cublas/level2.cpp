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

namespace clap {

// Float GEMV

void CuBlasBackend::gemv(Layout layout, Transpose trans, int64_t m, int64_t n,
                         const float alpha, const float *A, int64_t lda,
                         const float *x, int64_t incx, const float beta,
                         float *y, int64_t incy) {
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

  clap_cublasSgemv(handle, cu_trans, (int)m, (int)n, &alpha, d_A, (int)lda, d_x,
                   (int)incx, &beta, d_y, (int)incy);

  clap_cudaMemcpy(y, d_y, sizey * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
}

// Double GEMV

void CuBlasBackend::gemv(Layout layout, Transpose trans, int64_t m, int64_t n,
                         const double alpha, const double *A, int64_t lda,
                         const double *x, int64_t incx, const double beta,
                         double *y, int64_t incy) {
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

// Float SYMV

void CuBlasBackend::symv(Layout layout, Uplo uplo, int64_t n, const float alpha,
                         const float *A, int64_t lda, const float *X,
                         int64_t incx, const float beta, float *Y,
                         int64_t incy) {
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

void CuBlasBackend::symv(Layout layout, Uplo uplo, int64_t n,
                         const double alpha, const double *A, int64_t lda,
                         const double *X, int64_t incx, const double beta,
                         double *Y, int64_t incy) {
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

  clap_cublasDsymv(handle, clap::to_cublas_uplo(uplo), (int)n, &alpha, d_A,
                   (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_cudaMemcpy(Y, d_Y, sizeY * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

// Float TRMV

void CuBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int64_t n, const float *A, int64_t lda, float *X,
                         int64_t incx) {
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
                         int64_t n, const double *A, int64_t lda, double *X,
                         int64_t incx) {
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

// Float TRSV

void CuBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                         int64_t n, const float *A, int64_t lda, float *X,
                         int64_t incx) {
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
                         int64_t n, const double *A, int64_t lda, double *X,
                         int64_t incx) {
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

// Float SBMV

void CuBlasBackend::sbmv(Layout layout, Uplo uplo, int64_t n, int64_t k,
                         const float alpha, const float *A, int64_t lda,
                         const float *X, int64_t incx, const float beta,
                         float *Y, int64_t incy) {
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

void CuBlasBackend::sbmv(Layout layout, Uplo uplo, int64_t n, int64_t k,
                         const double alpha, const double *A, int64_t lda,
                         const double *X, int64_t incx, const double beta,
                         double *Y, int64_t incy) {
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

  clap_cublasDsbmv(handle, clap::to_cublas_uplo(uplo), (int)n, (int)k, &alpha,
                   d_A, (int)lda, d_X, (int)incx, &beta, d_Y, (int)incy);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(Y, d_Y, sizeY * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_A);
  clap_cudaFree(d_X);
  clap_cudaFree(d_Y);
}

} // namespace clap
