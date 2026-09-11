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

namespace clap {

// Float IAMAX
int RocBlasBackend::iamax(int n, const float *x, int incx) {
  float *d_x = nullptr;
  int *d_result = nullptr;
  int h_result = 0;

  clap_hipMalloc((void **)&d_x, n * sizeof(float));
  clap_hipMalloc((void **)&d_result, sizeof(int));
  clap_hipMemcpy(d_x, x, n * sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_isamax(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_result);
  clap_hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_result);

  return (int)h_result;
}

// Double IAMAX
int RocBlasBackend::iamax(int n, const double *x, int incx) {
  double *d_x = nullptr;
  int *d_result = nullptr;
  int h_result = 0;

  clap_hipMalloc((void **)&d_x, n * sizeof(double));
  clap_hipMalloc((void **)&d_result, sizeof(int));
  clap_hipMemcpy(d_x, x, n * sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_idamax(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_result);
  clap_hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_result);

  return (int)h_result;
}

// Complex Float IAMAX
int RocBlasBackend::iamax(int n, const std::complex<float> *x, int incx) {
  using roc_cfloat = rocblas_float_complex;

  roc_cfloat *d_x = nullptr;
  int *d_result = nullptr;
  int h_result = 0;

  size_t size = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size * sizeof(roc_cfloat));
  clap_hipMalloc((void **)&d_result, sizeof(int));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cfloat *>(x),
                 size * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_rocblas_icamax(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_result);
  clap_hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_result);

  return (int)h_result;
}

// Complex Double IAMAX
int RocBlasBackend::iamax(int n, const std::complex<double> *x, int incx) {
  using roc_cdouble = rocblas_double_complex;

  roc_cdouble *d_x = nullptr;
  int *d_result = nullptr;
  int h_result = 0;

  size_t size = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size * sizeof(roc_cdouble));
  clap_hipMalloc((void **)&d_result, sizeof(int));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cdouble *>(x),
                 size * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_rocblas_izamax(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_result);
  clap_hipMemcpy(&h_result, d_result, sizeof(int), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_result);

  return (int)h_result;
}

// Float ASUM
float RocBlasBackend::asum(int n, const float *x, int incx) {
  float *d_x = nullptr;
  float result = 0.0f;

  size_t size = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size * sizeof(float));
  clap_hipMemcpy(d_x, x, size * sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_sasum(handle, (rocblas_int)n, d_x, (rocblas_int)incx, &result);
  clap_hipFree(d_x);

  return result;
}

// Double ASUM
double RocBlasBackend::asum(int n, const double *x, int incx) {
  double *d_x = nullptr;
  double result = 0.0;

  size_t size = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size * sizeof(double));
  clap_hipMemcpy(d_x, x, size * sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_dasum(handle, (rocblas_int)n, d_x, (rocblas_int)incx, &result);
  clap_hipFree(d_x);

  return result;
}

// Complex Float ASUM
float RocBlasBackend::asum(int n, const std::complex<float> *x, int incx) {
  using roc_cfloat = rocblas_float_complex;

  roc_cfloat *d_x = nullptr;
  float result = 0.0f;

  size_t size = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size * sizeof(roc_cfloat));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cfloat *>(x),
                 size * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_rocblas_scasum(handle, (rocblas_int)n, d_x, (rocblas_int)incx, &result);
  clap_hipFree(d_x);

  return result;
}

// Complex Double ASUM
double RocBlasBackend::asum(int n, const std::complex<double> *x, int incx) {
  using roc_cdouble = rocblas_double_complex;

  roc_cdouble *d_x = nullptr;
  double result = 0.0;

  size_t size = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size * sizeof(roc_cdouble));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cdouble *>(x),
                 size * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_rocblas_dzasum(handle, (rocblas_int)n, d_x, (rocblas_int)incx, &result);
  clap_hipFree(d_x);

  return result;
}

// Float AXPY
void RocBlasBackend::axpy(int n, const float alpha, const float *x, int incx,
                          float *y, int incy) {
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
void RocBlasBackend::axpy(int n, const double alpha, const double *x, int incx,
                          double *y, int incy) {
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

// Complex Float AXPY
void RocBlasBackend::axpy(int n, const std::complex<float> *alpha,
                          const std::complex<float> *x, int incx,
                          std::complex<float> *y, int incy) {
  using roc_cfloat = rocblas_float_complex;

  roc_cfloat *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cfloat));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(roc_cfloat));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cfloat *>(x),
                 size_x * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, reinterpret_cast<const roc_cfloat *>(y),
                 size_y * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_rocblas_caxpy(handle, (rocblas_int)n,
                     reinterpret_cast<const roc_cfloat *>(alpha), d_x,
                     (rocblas_int)incx, d_y, (rocblas_int)incy);
  clap_hipMemcpy(y, d_y, size_y * sizeof(roc_cfloat), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Complex Double AXPY
void RocBlasBackend::axpy(int n, const std::complex<double> *alpha,
                          const std::complex<double> *x, int incx,
                          std::complex<double> *y, int incy) {
  using roc_cdouble = rocblas_double_complex;

  roc_cdouble *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cdouble));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(roc_cdouble));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cdouble *>(x),
                 size_x * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, reinterpret_cast<const roc_cdouble *>(y),
                 size_y * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_rocblas_zaxpy(handle, (rocblas_int)n,
                     reinterpret_cast<const roc_cdouble *>(alpha), d_x,
                     (rocblas_int)incx, d_y, (rocblas_int)incy);
  clap_hipMemcpy(y, d_y, size_y * sizeof(roc_cdouble), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Float COPY
void RocBlasBackend::copy(int n, const float *x, int incx, float *y, int incy) {
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
void RocBlasBackend::copy(int n, const double *x, int incx, double *y,
                          int incy) {
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

// Complex Float COPY
void RocBlasBackend::copy(int n, const std::complex<float> *x, int incx,
                          std::complex<float> *y, int incy) {
  using roc_cfloat = rocblas_float_complex;

  roc_cfloat *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cfloat));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(roc_cfloat));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cfloat *>(x),
                 size_x * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, reinterpret_cast<const roc_cfloat *>(y),
                 size_y * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_rocblas_ccopy(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy);
  clap_hipMemcpy(y, d_y, size_y * sizeof(roc_cfloat), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Complex Double COPY
void RocBlasBackend::copy(int n, const std::complex<double> *x, int incx,
                          std::complex<double> *y, int incy) {
  using roc_cdouble = rocblas_double_complex;

  roc_cdouble *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cdouble));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(roc_cdouble));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cdouble *>(x),
                 size_x * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, reinterpret_cast<const roc_cdouble *>(y),
                 size_y * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_rocblas_zcopy(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy);
  clap_hipMemcpy(y, d_y, size_y * sizeof(roc_cdouble), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Float DOT
float RocBlasBackend::dot(int n, const float *x, int incx, const float *y,
                          int incy) {
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
double RocBlasBackend::dot(int n, const double *x, int incx, const double *y,
                           int incy) {
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

// Complex Float DOTU_SUB
void RocBlasBackend::dotu_sub(int n, const std::complex<float> *x, int incx,
                              const std::complex<float> *y, int incy,
                              std::complex<float> *dotu) {
  using roc_cfloat = rocblas_float_complex;

  roc_cfloat *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cfloat));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(roc_cfloat));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cfloat *>(x),
                 size_x * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, reinterpret_cast<const roc_cfloat *>(y),
                 size_y * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_rocblas_cdotu(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy, reinterpret_cast<roc_cfloat *>(dotu));
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Complex Float DOTC_SUB
void RocBlasBackend::dotc_sub(int n, const std::complex<float> *x, int incx,
                              const std::complex<float> *y, int incy,
                              std::complex<float> *dotc) {
  using roc_cfloat = rocblas_float_complex;

  roc_cfloat *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cfloat));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(roc_cfloat));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cfloat *>(x),
                 size_x * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, reinterpret_cast<const roc_cfloat *>(y),
                 size_y * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_rocblas_cdotc(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy, reinterpret_cast<roc_cfloat *>(dotc));
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Complex Double DOTU_SUB
void RocBlasBackend::dotu_sub(int n, const std::complex<double> *x, int incx,
                              const std::complex<double> *y, int incy,
                              std::complex<double> *dotu) {
  using roc_cdouble = rocblas_double_complex;

  roc_cdouble *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cdouble));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(roc_cdouble));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cdouble *>(x),
                 size_x * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, reinterpret_cast<const roc_cdouble *>(y),
                 size_y * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_rocblas_zdotu(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy, reinterpret_cast<roc_cdouble *>(dotu));
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Complex Double DOTC_SUB
void RocBlasBackend::dotc_sub(int n, const std::complex<double> *x, int incx,
                              const std::complex<double> *y, int incy,
                              std::complex<double> *dotc) {
  using roc_cdouble = rocblas_double_complex;

  roc_cdouble *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cdouble));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(roc_cdouble));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cdouble *>(x),
                 size_x * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, reinterpret_cast<const roc_cdouble *>(y),
                 size_y * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_rocblas_zdotc(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy, reinterpret_cast<roc_cdouble *>(dotc));
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Float NRM2
float RocBlasBackend::nrm2(int n, const float *x, int incx) {
  float *d_x = nullptr;
  float *d_result = nullptr;
  float h_result = 0.0f;

  size_t size_x = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(float));
  clap_hipMalloc((void **)&d_result, sizeof(float));
  clap_hipMemcpy(d_x, x, size_x * sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_snrm2(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_result);
  clap_hipMemcpy(&h_result, d_result, sizeof(float), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_result);

  return h_result;
}

// Double NRM2
double RocBlasBackend::nrm2(int n, const double *x, int incx) {
  double *d_x = nullptr;
  double *d_result = nullptr;
  double h_result = 0.0;

  size_t size_x = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(double));
  clap_hipMalloc((void **)&d_result, sizeof(double));
  clap_hipMemcpy(d_x, x, size_x * sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_dnrm2(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_result);
  clap_hipMemcpy(&h_result, d_result, sizeof(double), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_result);

  return h_result;
}

// Complex Float NRM2
float RocBlasBackend::nrm2(int n, const std::complex<float> *x, int incx) {
  using roc_cfloat = rocblas_float_complex;

  roc_cfloat *d_x = nullptr;
  float *d_result = nullptr;
  float h_result = 0.0f;

  size_t size_x = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cfloat));
  clap_hipMalloc((void **)&d_result, sizeof(float));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cfloat *>(x),
                 size_x * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_rocblas_scnrm2(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_result);
  clap_hipMemcpy(&h_result, d_result, sizeof(float), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_result);

  return h_result;
}

// Complex Double NRM2
double RocBlasBackend::nrm2(int n, const std::complex<double> *x, int incx) {
  using roc_cdouble = rocblas_double_complex;

  roc_cdouble *d_x = nullptr;
  double *d_result = nullptr;
  double h_result = 0.0;

  size_t size_x = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cdouble));
  clap_hipMalloc((void **)&d_result, sizeof(double));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cdouble *>(x),
                 size_x * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_rocblas_dznrm2(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_result);
  clap_hipMemcpy(&h_result, d_result, sizeof(double), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_result);

  return h_result;
}

// Float ROT
void RocBlasBackend::rot(int n, float *x, int incx, float *y, int incy, float c,
                         float s) {
  float *d_x = nullptr, *d_y = nullptr;
  float *d_c = nullptr, *d_s = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(float));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(float));
  clap_hipMalloc((void **)&d_c, sizeof(float));
  clap_hipMalloc((void **)&d_s, sizeof(float));
  clap_hipMemcpy(d_x, x, size_x * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_c, &c, sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_s, &s, sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_srot(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                    (rocblas_int)incy, d_c, d_s);
  clap_hipMemcpy(x, d_x, size_x * sizeof(float), hipMemcpyDeviceToHost);
  clap_hipMemcpy(y, d_y, size_y * sizeof(float), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
  clap_hipFree(d_c);
  clap_hipFree(d_s);
}

// Double ROT
void RocBlasBackend::rot(int n, double *x, int incx, double *y, int incy,
                         double c, double s) {
  double *d_x = nullptr, *d_y = nullptr;
  double *d_c = nullptr, *d_s = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(double));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(double));
  clap_hipMalloc((void **)&d_c, sizeof(double));
  clap_hipMalloc((void **)&d_s, sizeof(double));
  clap_hipMemcpy(d_x, x, size_x * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_c, &c, sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_s, &s, sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_drot(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                    (rocblas_int)incy, d_c, d_s);
  clap_hipMemcpy(x, d_x, size_x * sizeof(double), hipMemcpyDeviceToHost);
  clap_hipMemcpy(y, d_y, size_y * sizeof(double), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
  clap_hipFree(d_c);
  clap_hipFree(d_s);
}

// Float ROTG
void RocBlasBackend::rotg(float *a, float *b, float *c, float *s) {
  float *d_a = nullptr, *d_b = nullptr;
  float *d_c = nullptr, *d_s = nullptr;

  clap_hipMalloc((void **)&d_a, sizeof(float));
  clap_hipMalloc((void **)&d_b, sizeof(float));
  clap_hipMalloc((void **)&d_c, sizeof(float));
  clap_hipMalloc((void **)&d_s, sizeof(float));
  clap_hipMemcpy(d_a, a, sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_b, b, sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_srotg(handle, d_a, d_b, d_c, d_s);
  clap_hipMemcpy(a, d_a, sizeof(float), hipMemcpyDeviceToHost);
  clap_hipMemcpy(b, d_b, sizeof(float), hipMemcpyDeviceToHost);
  clap_hipMemcpy(c, d_c, sizeof(float), hipMemcpyDeviceToHost);
  clap_hipMemcpy(s, d_s, sizeof(float), hipMemcpyDeviceToHost);
  clap_hipFree(d_a);
  clap_hipFree(d_b);
  clap_hipFree(d_c);
  clap_hipFree(d_s);
}

// Double ROTG
void RocBlasBackend::rotg(double *a, double *b, double *c, double *s) {
  double *d_a = nullptr, *d_b = nullptr;
  double *d_c = nullptr, *d_s = nullptr;

  clap_hipMalloc((void **)&d_a, sizeof(double));
  clap_hipMalloc((void **)&d_b, sizeof(double));
  clap_hipMalloc((void **)&d_c, sizeof(double));
  clap_hipMalloc((void **)&d_s, sizeof(double));
  clap_hipMemcpy(d_a, a, sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_b, b, sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_drotg(handle, d_a, d_b, d_c, d_s);
  clap_hipMemcpy(a, d_a, sizeof(double), hipMemcpyDeviceToHost);
  clap_hipMemcpy(b, d_b, sizeof(double), hipMemcpyDeviceToHost);
  clap_hipMemcpy(c, d_c, sizeof(double), hipMemcpyDeviceToHost);
  clap_hipMemcpy(s, d_s, sizeof(double), hipMemcpyDeviceToHost);
  clap_hipFree(d_a);
  clap_hipFree(d_b);
  clap_hipFree(d_c);
  clap_hipFree(d_s);
}

// Float ROTM
void RocBlasBackend::rotm(int n, float *x, int incx, float *y, int incy,
                          const float *param) {
  float *d_x = nullptr, *d_y = nullptr;
  float *d_param = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(float));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(float));
  clap_hipMalloc((void **)&d_param, 5 * sizeof(float));
  clap_hipMemcpy(d_x, x, size_x * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_param, param, 5 * sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_srotm(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy, d_param);
  clap_hipMemcpy(x, d_x, size_x * sizeof(float), hipMemcpyDeviceToHost);
  clap_hipMemcpy(y, d_y, size_y * sizeof(float), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
  clap_hipFree(d_param);
}

// Double ROTM
void RocBlasBackend::rotm(int n, double *x, int incx, double *y, int incy,
                          const double *param) {
  double *d_x = nullptr, *d_y = nullptr;
  double *d_param = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(double));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(double));
  clap_hipMalloc((void **)&d_param, 5 * sizeof(double));
  clap_hipMemcpy(d_x, x, size_x * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, y, size_y * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_param, param, 5 * sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_drotm(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy, d_param);
  clap_hipMemcpy(x, d_x, size_x * sizeof(double), hipMemcpyDeviceToHost);
  clap_hipMemcpy(y, d_y, size_y * sizeof(double), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
  clap_hipFree(d_param);
}

// Float ROTMG
void RocBlasBackend::rotmg(float *d1, float *d2, float *b1, const float b2,
                           float *P) {
  float *d_d1 = nullptr, *d_d2 = nullptr;
  float *d_b1 = nullptr, *d_b2 = nullptr;
  float *d_P = nullptr;

  clap_hipMalloc((void **)&d_d1, sizeof(float));
  clap_hipMalloc((void **)&d_d2, sizeof(float));
  clap_hipMalloc((void **)&d_b1, sizeof(float));
  clap_hipMalloc((void **)&d_b2, sizeof(float));
  clap_hipMalloc((void **)&d_P, 5 * sizeof(float));
  clap_hipMemcpy(d_d1, d1, sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_d2, d2, sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_b1, b1, sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_b2, &b2, sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_srotmg(handle, d_d1, d_d2, d_b1, d_b2, d_P);
  clap_hipMemcpy(d1, d_d1, sizeof(float), hipMemcpyDeviceToHost);
  clap_hipMemcpy(d2, d_d2, sizeof(float), hipMemcpyDeviceToHost);
  clap_hipMemcpy(b1, d_b1, sizeof(float), hipMemcpyDeviceToHost);
  clap_hipMemcpy(P, d_P, 5 * sizeof(float), hipMemcpyDeviceToHost);
  clap_hipFree(d_d1);
  clap_hipFree(d_d2);
  clap_hipFree(d_b1);
  clap_hipFree(d_b2);
  clap_hipFree(d_P);
}

// Double ROTMG
void RocBlasBackend::rotmg(double *d1, double *d2, double *x1, const double y1,
                           double *param) {
  double *d_d1 = nullptr, *d_d2 = nullptr;
  double *d_x1 = nullptr, *d_y1 = nullptr;
  double *d_param = nullptr;

  clap_hipMalloc((void **)&d_d1, sizeof(double));
  clap_hipMalloc((void **)&d_d2, sizeof(double));
  clap_hipMalloc((void **)&d_x1, sizeof(double));
  clap_hipMalloc((void **)&d_y1, sizeof(double));
  clap_hipMalloc((void **)&d_param, 5 * sizeof(double));
  clap_hipMemcpy(d_d1, d1, sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_d2, d2, sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_x1, x1, sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y1, &y1, sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_drotmg(handle, d_d1, d_d2, d_x1, d_y1, d_param);
  clap_hipMemcpy(d1, d_d1, sizeof(double), hipMemcpyDeviceToHost);
  clap_hipMemcpy(d2, d_d2, sizeof(double), hipMemcpyDeviceToHost);
  clap_hipMemcpy(x1, d_x1, sizeof(double), hipMemcpyDeviceToHost);
  clap_hipMemcpy(param, d_param, 5 * sizeof(double), hipMemcpyDeviceToHost);
  clap_hipFree(d_d1);
  clap_hipFree(d_d2);
  clap_hipFree(d_x1);
  clap_hipFree(d_y1);
  clap_hipFree(d_param);
}

// Float SCAL
void RocBlasBackend::scal(int n, float alpha, float *x, int incx) {
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
void RocBlasBackend::scal(int n, double alpha, double *x, int incx) {
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

// Complex Float SCAL
void RocBlasBackend::scal(int n, const std::complex<float> *alpha,
                          std::complex<float> *x, int incx) {
  using roc_cfloat = rocblas_float_complex;

  roc_cfloat *d_x = nullptr, *d_alpha = nullptr;

  size_t size_x = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cfloat));
  clap_hipMalloc((void **)&d_alpha, sizeof(roc_cfloat));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cfloat *>(x),
                 size_x * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_alpha, reinterpret_cast<const roc_cfloat *>(alpha),
                 sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_rocblas_cscal(handle, (rocblas_int)n, d_alpha, d_x, (rocblas_int)incx);
  clap_hipMemcpy(x, d_x, size_x * sizeof(roc_cfloat), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_alpha);
}

// Complex Double SCAL
void RocBlasBackend::scal(int n, const std::complex<double> *alpha,
                          std::complex<double> *x, int incx) {
  using roc_cdouble = rocblas_double_complex;

  roc_cdouble *d_x = nullptr, *d_alpha = nullptr;

  size_t size_x = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cdouble));
  clap_hipMalloc((void **)&d_alpha, sizeof(roc_cdouble));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cdouble *>(x),
                 size_x * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_alpha, reinterpret_cast<const roc_cdouble *>(alpha),
                 sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_rocblas_zscal(handle, (rocblas_int)n, d_alpha, d_x, (rocblas_int)incx);
  clap_hipMemcpy(x, d_x, size_x * sizeof(roc_cdouble), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_alpha);
}

// Complex Float SCAL
void RocBlasBackend::scal(int n, float alpha, std::complex<float> *x,
                          int incx) {
  using roc_cfloat = rocblas_float_complex;

  roc_cfloat *d_x = nullptr;
  float *d_alpha = nullptr;

  size_t size_x = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cfloat));
  clap_hipMalloc((void **)&d_alpha, sizeof(float));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cfloat *>(x),
                 size_x * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_alpha, &alpha, sizeof(float), hipMemcpyHostToDevice);
  clap_rocblas_csscal(handle, (rocblas_int)n, d_alpha, d_x, (rocblas_int)incx);
  clap_hipMemcpy(x, d_x, size_x * sizeof(roc_cfloat), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_alpha);
}

// Complex Double SCAL
void RocBlasBackend::scal(int n, double alpha, std::complex<double> *x,
                          int incx) {
  using roc_cdouble = rocblas_double_complex;

  roc_cdouble *d_x = nullptr;
  double *d_alpha = nullptr;

  size_t size_x = (n - 1) * incx + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cdouble));
  clap_hipMalloc((void **)&d_alpha, sizeof(double));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cdouble *>(x),
                 size_x * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_alpha, &alpha, sizeof(double), hipMemcpyHostToDevice);
  clap_rocblas_zdscal(handle, (rocblas_int)n, d_alpha, d_x, (rocblas_int)incx);
  clap_hipMemcpy(x, d_x, size_x * sizeof(roc_cdouble), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_alpha);
}

// Float SWAP
void RocBlasBackend::swap(int n, float *x, int incx, float *y, int incy) {
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
void RocBlasBackend::swap(int n, double *x, int incx, double *y, int incy) {
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

// Complex Float SWAP
void RocBlasBackend::swap(int n, std::complex<float> *x, int incx,
                          std::complex<float> *y, int incy) {
  using roc_cfloat = rocblas_float_complex;

  roc_cfloat *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cfloat));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(roc_cfloat));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cfloat *>(x),
                 size_x * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, reinterpret_cast<const roc_cfloat *>(y),
                 size_y * sizeof(roc_cfloat), hipMemcpyHostToDevice);
  clap_rocblas_cswap(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy);
  clap_hipMemcpy(x, d_x, size_x * sizeof(roc_cfloat), hipMemcpyDeviceToHost);
  clap_hipMemcpy(y, d_y, size_y * sizeof(roc_cfloat), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

// Complex Double SWAP
void RocBlasBackend::swap(int n, std::complex<double> *x, int incx,
                          std::complex<double> *y, int incy) {
  using roc_cdouble = rocblas_double_complex;

  roc_cdouble *d_x = nullptr, *d_y = nullptr;

  size_t size_x = (n - 1) * incx + 1;
  size_t size_y = (n - 1) * incy + 1;

  clap_hipMalloc((void **)&d_x, size_x * sizeof(roc_cdouble));
  clap_hipMalloc((void **)&d_y, size_y * sizeof(roc_cdouble));
  clap_hipMemcpy(d_x, reinterpret_cast<const roc_cdouble *>(x),
                 size_x * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_y, reinterpret_cast<const roc_cdouble *>(y),
                 size_y * sizeof(roc_cdouble), hipMemcpyHostToDevice);
  clap_rocblas_zswap(handle, (rocblas_int)n, d_x, (rocblas_int)incx, d_y,
                     (rocblas_int)incy);
  clap_hipMemcpy(x, d_x, size_x * sizeof(roc_cdouble), hipMemcpyDeviceToHost);
  clap_hipMemcpy(y, d_y, size_y * sizeof(roc_cdouble), hipMemcpyDeviceToHost);
  clap_hipFree(d_x);
  clap_hipFree(d_y);
}

}

