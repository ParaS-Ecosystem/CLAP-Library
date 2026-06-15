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

namespace clap {

// level 1

// Float COPY

void CuBlasBackend::copy(int64_t n, const float *x, int64_t incx, float *y,
                         int64_t incy) {
  float *d_x = nullptr, *d_y = nullptr;
  clap_cudaMalloc((void **)&d_x, n * sizeof(float));
  clap_cudaMalloc((void **)&d_y, n * sizeof(float));
  clap_cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cublasScopy(handle, n, d_x, incx, d_y, incy);
  clap_cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
}

// Double COPY

void CuBlasBackend::copy(int64_t n, const double *x, int64_t incx, double *y,
                         int64_t incy) {
  double *d_x = nullptr, *d_y = nullptr;
  clap_cudaMalloc((void **)&d_x, n * sizeof(double));
  clap_cudaMalloc((void **)&d_y, n * sizeof(double));
  clap_cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cublasDcopy(handle, n, d_x, incx, d_y, incy);
  clap_cudaMemcpy(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
}

// Float SCAL

void CuBlasBackend::scal(int64_t n, float alpha, float *x, int64_t incx) {
  float *d_x = nullptr;
  clap_cudaMalloc((void **)&d_x, n * sizeof(float));
  clap_cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cublasSscal(handle, n, &alpha, d_x, incx);
  clap_cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaFree(d_x);
}

// Double SCAL

void CuBlasBackend::scal(int64_t n, double alpha, double *x, int64_t incx) {
  double *d_x = nullptr;
  clap_cudaMalloc((void **)&d_x, n * sizeof(double));
  clap_cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cublasDscal(handle, n, &alpha, d_x, incx);
  clap_cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaFree(d_x);
}

// Float AXPY

void CuBlasBackend::axpy(int64_t n, float a, const float *x, int64_t incx,
                         float *y, int64_t incy) {
  float *d_x = nullptr, *d_y = nullptr;
  clap_cudaMalloc((void **)&d_x, n * sizeof(float));
  clap_cudaMalloc((void **)&d_y, n * sizeof(float));
  clap_cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cublasSaxpy(handle, n, &a, d_x, incx, d_y, incy);
  clap_cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
}

// Doublae AXPY

void CuBlasBackend::axpy(int64_t n, double a, const double *x, int64_t incx,
                         double *y, int64_t incy) {
  double *d_x = nullptr, *d_y = nullptr;
  clap_cudaMalloc((void **)&d_x, n * sizeof(double));
  clap_cudaMalloc((void **)&d_y, n * sizeof(double));
  clap_cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cublasDaxpy(handle, n, &a, d_x, incx, d_y, incy);
  clap_cudaMemcpy(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
}

// Float SWAP

void CuBlasBackend::swap(int64_t n, float *x, int64_t incx, float *y,
                         int64_t incy) {
  float *d_x = nullptr, *d_y = nullptr;
  clap_cudaMalloc((void **)&d_x, n * sizeof(float));
  clap_cudaMalloc((void **)&d_y, n * sizeof(float));
  clap_cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cublasSswap(handle, n, d_x, incx, d_y, incy);
  clap_cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
}

// Double SWAP

void CuBlasBackend::swap(int64_t n, double *x, int64_t incx, double *y,
                         int64_t incy) {
  double *d_x = nullptr, *d_y = nullptr;
  clap_cudaMalloc((void **)&d_x, n * sizeof(double));
  clap_cudaMalloc((void **)&d_y, n * sizeof(double));
  clap_cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cublasDswap(handle, n, d_x, incx, d_y, incy);
  clap_cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
}

// Float DOT

float CuBlasBackend::dot(int64_t n, const float *x, int64_t incx,
                         const float *y, int64_t incy) {
  float *d_x = nullptr, *d_y = nullptr, result = 0.0f;
  clap_cudaMalloc((void **)&d_x, n * sizeof(float));
  clap_cudaMalloc((void **)&d_y, n * sizeof(float));
  clap_cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cublasSdot(handle, n, d_x, incx, d_y, incy, &result);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
  return result;
}

// Double DOT

double CuBlasBackend::dot(int64_t n, const double *x, int64_t incx,
                          const double *y, int64_t incy) {
  double *d_x = nullptr, *d_y = nullptr, result = 0.0;
  clap_cudaMalloc((void **)&d_x, n * sizeof(double));
  clap_cudaMalloc((void **)&d_y, n * sizeof(double));
  clap_cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cublasDdot(handle, n, d_x, incx, d_y, incy, &result);
  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
  return result;
}

} // namespace clap
