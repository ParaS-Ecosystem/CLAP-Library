// -----------------------------------------------------------------------------
// This file is part of the CLAP library — an implementation of
// high-performance linear algebra routines based on the BLAS standard.
//
// CLAP provides a unified math interface across architectures,
// enabling portable and optimized performance on CPUs.
//
// Copyright (C) The CLAP Contributors
//
// CLAP is released under the GNU Lesser General Public License (LGPL) v3.
// See the file LICENSE in the project root for full license details.
// -----------------------------------------------------------------------------

#include "clap/rocblas_backend.hpp"

namespace clap {

// Float AXPY

void RocBlasBackend::axpy(int64_t n, const float alpha, const float *x,
                          int64_t incx, float *y, int64_t incy) {
  float *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(float));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(float));
  clap_hipMemcpy(d_x, x, size_x * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_saxpy(handle, (rocblas_int)n, &alpha, d_x, (rocblas_int)incx,
                     d_y, (rocblas_int)incy);
  clap_hipMemcpy(y, d_y, size_y * sizeof(float), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Double AXPY

void RocBlasBackend::axpy(int64_t n, const double alpha, const double *x,
                          int64_t incx, double *y, int64_t incy) {
  double *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(double));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(double));
  clap_hipMemcpy(d_x, x, size_x * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_daxpy(handle, (rocblas_int)n, &alpha, d_x, (rocblas_int)incx,
                     d_y, (rocblas_int)incy);
  clap_hipMemcpy(y, d_y, size_y * sizeof(double), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Float COPY

void RocBlasBackend::copy(int64_t n, const float *x, int64_t incx, float *y,
                          int64_t incy) {
  float *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(float));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(float));
  clap_hipMemcpy(d_x, x, size_x * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_scopy(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy);
  clap_hipMemcpy(y, d_y, size_y * sizeof(float), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Double COPY

void RocBlasBackend::copy(int64_t n, const double *x, int64_t incx, double *y,
                          int64_t incy) {
  double *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(double));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(double));
  clap_hipMemcpy(d_x, x, size_x * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_dcopy(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy);
  clap_hipMemcpy(y, d_y, size_y * sizeof(double), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Float DOT

float RocBlasBackend::dot(int64_t n, const float *x, int64_t incx,
                          const float *y, int64_t incy) {
  float *d_x = nullptr, *d_y = nullptr;
  float result = 0.0f;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(float));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(float));
  clap_hipMemcpy(d_x, x, size_x * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_sdot(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                    (rocblas_int)incy, &result);
  clap_hipFree(d_x);
  clap_hipFree(d_y);

  return result;
}

// Double DOT

double RocBlasBackend::dot(int64_t n, const double *x, int64_t incx,
                           const double *y, int64_t incy) {
  double *d_x = nullptr, *d_y = nullptr;
  double result = 0.0;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(double));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(double));
  clap_hipMemcpy(d_x, x, size_x * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_ddot(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                    (rocblas_int)incy, &result);
  clap_hipFree(d_x);
  clap_hipFree(d_y);

  return result;
}

// Float SCAL

void RocBlasBackend::scal(int64_t n, float alpha, float *x, int64_t incx) {
  float *d_x = nullptr, *d_alpha = nullptr;

  size_t size_x = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(float));
  clap_hipMalloc((void **)&d_alpha, sizeof(float));
  clap_hipMemcpy(d_x, x, size_x * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_alpha, &alpha, sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_sscal(handle, (rocblas_int)n, d_alpha, d_x, (rocblas_int)incx);
  clap_hipMemcpy(x, d_x, size_x * sizeof(float), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_alpha);
}

// Double SCAL

void RocBlasBackend::scal(int64_t n, double alpha, double *x, int64_t incx) {
  double *d_x = nullptr, *d_alpha = nullptr;

  size_t size_x = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(double));
  clap_hipMalloc((void **)&d_alpha, sizeof(double));
  clap_hipMemcpy(d_x, x, size_x * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_alpha, &alpha, sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_dscal(handle, (rocblas_int)n, d_alpha, d_x, (rocblas_int)incx);
  clap_hipMemcpy(x, d_x, size_x * sizeof(double), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_alpha);
}

// Float SWAP

void RocBlasBackend::swap(int64_t n, float *x, int64_t incx, float *y,
                          int64_t incy) {
  float *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(float));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(float));
  clap_hipMemcpy(d_x, x, size_x * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_sswap(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy);
  clap_hipMemcpy(x, d_x, size_x * sizeof(float), hipMemcpyDeviceToHost);
  clap_hipMemcpy(y, d_y, size_y * sizeof(float), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Double SWAP

void RocBlasBackend::swap(int64_t n, double *x, int64_t incx, double *y,
                          int64_t incy) {
  double *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(double));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(double));
  clap_hipMemcpy(d_x, x, size_x * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_dswap(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy);
  clap_hipMemcpy(x, d_x, size_x * sizeof(double), hipMemcpyDeviceToHost);
  clap_hipMemcpy(y, d_y, size_y * sizeof(double), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

} // namespace clap
