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

// Float COPY
void CuBlasBackend::copy(int n, const float *x, int incx, float *y, int incy) {
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
void CuBlasBackend::copy(int n, const double *x, int incx, double *y,
                         int incy) {
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
void CuBlasBackend::scal(int n, float alpha, float *x, int incx) {
  float *d_x = nullptr;
  clap_cudaMalloc((void **)&d_x, n * sizeof(float));
  clap_cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cublasSscal(handle, n, &alpha, d_x, incx);
  clap_cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaFree(d_x);
}

// Double SCAL
void CuBlasBackend::scal(int n, double alpha, double *x, int incx) {
  double *d_x = nullptr;
  clap_cudaMalloc((void **)&d_x, n * sizeof(double));
  clap_cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cublasDscal(handle, n, &alpha, d_x, incx);
  clap_cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaFree(d_x);
}

// Float AXPY
void CuBlasBackend::axpy(int n, float a, const float *x, int incx, float *y,
                         int incy) {
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

// Double AXPY
void CuBlasBackend::axpy(int n, double a, const double *x, int incx, double *y,
                         int incy) {
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

// Float IAMAX
int CuBlasBackend::iamax(int n, const float *x, int incx) {
  float *d_x = nullptr;
  int result = 0;
  clap_cudaMalloc((void **)&d_x, n * sizeof(float));
  clap_cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cublasIsamax(handle, n, d_x, incx, &result);
  clap_cudaFree(d_x);
  return result - 1;
}

// Double IAMAX
int CuBlasBackend::iamax(int n, const double *x, int incx) {
  double *d_x = nullptr;
  int result = 0;
  clap_cudaMalloc((void **)&d_x, n * sizeof(double));
  clap_cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cublasIdamax(handle, n, d_x, incx, &result);
  clap_cudaFree(d_x);
  return result - 1;
}

// Float SWAP
void CuBlasBackend::swap(int n, float *x, int incx, float *y, int incy) {
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
void CuBlasBackend::swap(int n, double *x, int incx, double *y, int incy) {
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
float CuBlasBackend::dot(int n, const float *x, int incx, const float *y,
                         int incy) {
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
double CuBlasBackend::dot(int n, const double *x, int incx, const double *y,
                          int incy) {
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

// Float NRM2
float CuBlasBackend::nrm2(int n, const float *x, int incx) {
  float *d_x = nullptr, result = 0.0f;
  clap_cudaMalloc((void **)&d_x, n * sizeof(float));
  clap_cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSnrm2(handle, n, d_x, incx, &result);
  clap_cudaFree(d_x);
  return result;
}

// Double NRM2
double CuBlasBackend::nrm2(int n, const double *x, int incx) {
  double *d_x = nullptr, result = 0.0;
  clap_cudaMalloc((void **)&d_x, n * sizeof(double));
  clap_cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  std::cout << "Calling cublasDnrm2" << std::endl;
  clap_cublasDnrm2(handle, n, d_x, incx, &result);
  clap_cudaFree(d_x);
  return result;
}

// Float ASUM
float CuBlasBackend::asum(int n, const float *x, int incx) {
  float *d_x = nullptr, result = 0.0f;
  clap_cudaMalloc((void **)&d_x, n * sizeof(float));
  clap_cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cublasSasum(handle, n, d_x, incx, &result);
  clap_cudaFree(d_x);
  return result;
}

// Double ASUM
double CuBlasBackend::asum(int n, const double *x, int incx) {
  double *d_x = nullptr, result = 0.0;
  clap_cudaMalloc((void **)&d_x, n * sizeof(double));
  clap_cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cublasDasum(handle, n, d_x, incx, &result);
  clap_cudaFree(d_x);
  return result;
}

// Double ROT
void CuBlasBackend::rot(int n, double *x, int incx, double *y, int incy,
                        double c, double s) {
  double *d_x = nullptr, *d_y = nullptr;
  double *d_c = nullptr, *d_s = nullptr;

  clap_cudaMalloc((void **)&d_x, n * sizeof(double));
  clap_cudaMalloc((void **)&d_y, n * sizeof(double));
  clap_cudaMalloc((void **)&d_c, sizeof(double));
  clap_cudaMalloc((void **)&d_s, sizeof(double));

  clap_cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_c, &c, sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_s, &s, sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  clap_cublasDrot(handle, n, d_x, incx, d_y, incy, d_c, d_s);
  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
  clap_cudaFree(d_c);
  clap_cudaFree(d_s);
}

// Float ROT
void CuBlasBackend::rot(int n, float *x, int incx, float *y, int incy, float c,
                        float s) {
  float *d_x = nullptr, *d_y = nullptr;
  float *d_c = nullptr, *d_s = nullptr;

  clap_cudaMalloc((void **)&d_x, n * sizeof(float));
  clap_cudaMalloc((void **)&d_y, n * sizeof(float));
  clap_cudaMalloc((void **)&d_c, sizeof(float));
  clap_cudaMalloc((void **)&d_s, sizeof(float));

  clap_cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_c, &c, sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_s, &s, sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  clap_cublasSrot(handle, n, d_x, incx, d_y, incy, d_c, d_s);
  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
  clap_cudaFree(d_c);
  clap_cudaFree(d_s);
}

// Float ROTG
void CuBlasBackend::rotg(float *a, float *b, float *c, float *s) {
  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_s = nullptr;

  clap_cudaMalloc((void **)&d_a, sizeof(float));
  clap_cudaMalloc((void **)&d_b, sizeof(float));
  clap_cudaMalloc((void **)&d_c, sizeof(float));
  clap_cudaMalloc((void **)&d_s, sizeof(float));

  clap_cudaMemcpy(d_a, a, sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_b, b, sizeof(float), cudaMemcpyHostToDevice);

  clap_cudaMemset(d_c, 0, sizeof(float));
  clap_cudaMemset(d_s, 0, sizeof(float));

  clap_cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

  clap_cublasSrotg(handle, d_a, d_b, d_c, d_s);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(a, d_a, sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(b, d_b, sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(c, d_c, sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(s, d_s, sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_a);
  clap_cudaFree(d_b);
  clap_cudaFree(d_c);
  clap_cudaFree(d_s);
}

// Double ROTG
void CuBlasBackend::rotg(double *a, double *b, double *c, double *s) {
  double *d_a = nullptr, *d_b = nullptr;
  double *d_c = nullptr, *d_s = nullptr;

  clap_cudaMalloc((void **)&d_a, sizeof(double));
  clap_cudaMalloc((void **)&d_b, sizeof(double));
  clap_cudaMalloc((void **)&d_c, sizeof(double));
  clap_cudaMalloc((void **)&d_s, sizeof(double));

  clap_cudaMemcpy(d_a, a, sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_b, b, sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  clap_cublasDrotg(handle, d_a, d_b, d_c, d_s);
  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(a, d_a, sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(b, d_b, sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(c, d_c, sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(s, d_s, sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_a);
  clap_cudaFree(d_b);
  clap_cudaFree(d_c);
  clap_cudaFree(d_s);
}

// Float ROTM
void CuBlasBackend::rotm(int n, float *x, int incx, float *y, int incy,
                         const float param[5]) {
  float *d_x = nullptr, *d_y = nullptr, *d_param = nullptr;

  clap_cudaMalloc((void **)&d_x, n * sizeof(float));
  clap_cudaMalloc((void **)&d_y, n * sizeof(float));
  clap_cudaMalloc((void **)&d_param, 5 * sizeof(float));

  clap_cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_param, param, 5 * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  clap_cublasSrotm(handle, n, d_x, incx, d_y, incy, d_param);
  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(y, d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
  clap_cudaFree(d_param);
}

// Double ROTM
void CuBlasBackend::rotm(int n, double *x, int incx, double *y, int incy,
                         const double param[5]) {
  double *d_x = nullptr, *d_y = nullptr, *d_param = nullptr;

  clap_cudaMalloc((void **)&d_x, n * sizeof(double));
  clap_cudaMalloc((void **)&d_y, n * sizeof(double));
  clap_cudaMalloc((void **)&d_param, 5 * sizeof(double));

  clap_cudaMemcpy(d_x, x, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_y, y, n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_param, param, 5 * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  clap_cublasDrotm(handle, n, d_x, incx, d_y, incy, d_param);
  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(x, d_x, n * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(y, d_y, n * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(d_x);
  clap_cudaFree(d_y);
  clap_cudaFree(d_param);
}

// Float ROTMG
void CuBlasBackend::rotmg(float *d1, float *d2, float *x1, float y1,
                          float param[5]) {
  float *dd1 = nullptr, *dd2 = nullptr, *dx1 = nullptr;
  float *dy1 = nullptr, *dparam = nullptr;

  clap_cudaMalloc((void **)&dd1, sizeof(float));
  clap_cudaMalloc((void **)&dd2, sizeof(float));
  clap_cudaMalloc((void **)&dx1, sizeof(float));
  clap_cudaMalloc((void **)&dy1, sizeof(float));
  clap_cudaMalloc((void **)&dparam, 5 * sizeof(float));

  clap_cudaMemcpy(dd1, d1, sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dd2, d2, sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dx1, x1, sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dy1, &y1, sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  clap_cublasSrotmg(handle, dd1, dd2, dx1, dy1, dparam);
  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(d1, dd1, sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(d2, dd2, sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(x1, dx1, sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(param, dparam, 5 * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(dd1);
  clap_cudaFree(dd2);
  clap_cudaFree(dx1);
  clap_cudaFree(dy1);
  clap_cudaFree(dparam);
}

// Double ROTMG
void CuBlasBackend::rotmg(double *d1, double *d2, double *x1, double y1,
                          double param[5]) {
  double *dd1 = nullptr, *dd2 = nullptr, *dx1 = nullptr;
  double *dy1 = nullptr, *dparam = nullptr;

  clap_cudaMalloc((void **)&dd1, sizeof(double));
  clap_cudaMalloc((void **)&dd2, sizeof(double));
  clap_cudaMalloc((void **)&dx1, sizeof(double));
  clap_cudaMalloc((void **)&dy1, sizeof(double));
  clap_cudaMalloc((void **)&dparam, 5 * sizeof(double));

  clap_cudaMemcpy(dd1, d1, sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dd2, d2, sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dx1, x1, sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dy1, &y1, sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  clap_cublasDrotmg(handle, dd1, dd2, dx1, dy1, dparam);
  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(d1, dd1, sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(d2, dd2, sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(x1, dx1, sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(param, dparam, 5 * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(dd1);
  clap_cudaFree(dd2);
  clap_cudaFree(dx1);
  clap_cudaFree(dy1);
  clap_cudaFree(dparam);
}

// Complex Float IAMAX
int CuBlasBackend::iamax(int n, const std::complex<float> *x, int incx) {
  cuComplex *dx = nullptr;
  int result = 0;

  size_t sizeX = n * std::abs(incx);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuComplex));
  clap_cudaMemcpy(dx, x, sizeX * sizeof(cuComplex), cudaMemcpyHostToDevice);

  clap_cublasIcamax(handle, (int)n, dx, (int)incx, &result);

  clap_cudaDeviceSynchronize();
  clap_cudaFree(dx);

  printf("Index = %d\n", result);
  return (result - 1);
}

// Complex Double IAMAX
int CuBlasBackend::iamax(int n, const std::complex<double> *x, int incx) {
  cuDoubleComplex *dx = nullptr;
  int result = 0;

  size_t sizeX = n * std::abs(incx);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMemcpy(dx, x, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasIzamax(handle, (int)n, dx, (int)incx, &result);

  clap_cudaDeviceSynchronize();
  clap_cudaFree(dx);

  printf("Index = %d\n", result);
  return (result - 1);
}

// Complex Float ASUM
float CuBlasBackend::asum(int n, const std::complex<float> *x, int incx) {
  cuComplex *dx = nullptr;
  float result;

  size_t sizeX = n * std::abs(incx);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuComplex));
  clap_cudaMemcpy(dx, x, sizeX * sizeof(cuComplex), cudaMemcpyHostToDevice);

  clap_cublasScasum(handle, (int)n, dx, (int)incx, &result);

  clap_cudaDeviceSynchronize();
  clap_cudaFree(dx);

  return result;
}

// Complex Double ASUM
double CuBlasBackend::asum(int n, const std::complex<double> *x, int incx) {
  cuDoubleComplex *dx = nullptr;
  double result;

  size_t sizeX = n * std::abs(incx);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMemcpy(dx, x, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasDzasum(handle, (int)n, dx, (int)incx, &result);

  clap_cudaDeviceSynchronize();
  clap_cudaFree(dx);

  return result;
}

// Complex Float AXPY
void CuBlasBackend::axpy(int n, const std::complex<float> *alpha,
                         const std::complex<float> *x, int incx,
                         std::complex<float> *y, int incy) {
  cuComplex *dx, *dy;

  clap_cudaMalloc((void **)&dx, n * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dy, n * sizeof(cuComplex));

  clap_cudaMemcpy(dx, x, n * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dy, y, n * sizeof(cuComplex), cudaMemcpyHostToDevice);

  cuComplex alpha_c = make_cuComplex(alpha->real(), alpha->imag());

  clap_cublasCaxpy(handle, (int)n, &alpha_c, dx, (int)incx, dy, (int)incy);

  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(y, dy, n * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dx);
  clap_cudaFree(dy);
}

// Complex Double AXPY
void CuBlasBackend::axpy(int n, const std::complex<double> *alpha,
                         const std::complex<double> *x, int incx,
                         std::complex<double> *y, int incy) {
  cuDoubleComplex *dx, *dy;

  clap_cudaMalloc((void **)&dx, n * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dy, n * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dx, x, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dy, y, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  cuDoubleComplex alpha_c = make_cuDoubleComplex(alpha->real(), alpha->imag());

  clap_cublasZaxpy(handle, (int)n, &alpha_c, dx, (int)incx, dy, (int)incy);

  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(y, dy, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dx);
  clap_cudaFree(dy);
}

// Complex Float COPY
void CuBlasBackend::copy(int n, const std::complex<float> *x, int incx,
                         std::complex<float> *y, int incy) {
  cuComplex *dx = nullptr, *dy = nullptr;

  size_t sizeX = n * std::abs(incx);
  size_t sizeY = n * std::abs(incy);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dy, sizeY * sizeof(cuComplex));

  clap_cudaMemcpy(dx, x, sizeX * sizeof(cuComplex), cudaMemcpyHostToDevice);

  clap_cublasCcopy(handle, (int)n, dx, (int)incx, dy, (int)incy);

  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(y, dy, sizeY * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dx);
  clap_cudaFree(dy);
}

// Complex Double COPY
void CuBlasBackend::copy(int n, const std::complex<double> *x, int incx,
                         std::complex<double> *y, int incy) {
  cuDoubleComplex *dx = nullptr, *dy = nullptr;

  size_t sizeX = n * std::abs(incx);
  size_t sizeY = n * std::abs(incy);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dy, sizeY * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dx, x, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZcopy(handle, (int)n, dx, (int)incx, dy, (int)incy);

  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(y, dy, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(dx);
  clap_cudaFree(dy);
}

// Complex Float DOTU_SUB
void CuBlasBackend::dotu_sub(int n, const std::complex<float> *x, int incx,
                             const std::complex<float> *y, int incy,
                             std::complex<float> *result) {
  cuComplex *dx, *dy;
  cuComplex h_result;

  clap_cudaMalloc((void **)&dx, n * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dy, n * sizeof(cuComplex));

  clap_cudaMemcpy(dx, x, n * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dy, y, n * sizeof(cuComplex), cudaMemcpyHostToDevice);

  clap_cublasCdotu(handle, (int)n, dx, (int)incx, dy, (int)incy, &h_result);

  clap_cudaDeviceSynchronize();
  *result = std::complex<float>(cuCrealf(h_result), cuCimagf(h_result));

  clap_cudaFree(dx);
  clap_cudaFree(dy);
}

// Complex Float DOTC_SUB
void CuBlasBackend::dotc_sub(int n, const std::complex<float> *x, int incx,
                             const std::complex<float> *y, int incy,
                             std::complex<float> *dotc) {
  cuComplex *dx = nullptr, *dy = nullptr;

  size_t sizeX = n * std::abs(incx);
  size_t sizeY = n * std::abs(incy);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dy, sizeY * sizeof(cuComplex));

  clap_cudaMemcpy(dx, x, sizeX * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dy, y, sizeY * sizeof(cuComplex), cudaMemcpyHostToDevice);

  clap_cublasCdotc(handle, (int)n, dx, (int)incx, dy, (int)incy,
                   reinterpret_cast<cuComplex *>(dotc));

  clap_cudaDeviceSynchronize();
  clap_cudaFree(dx);
  clap_cudaFree(dy);
}

// Complex Double DOTU_SUB
void CuBlasBackend::dotu_sub(int n, const std::complex<double> *x, int incx,
                             const std::complex<double> *y, int incy,
                             std::complex<double> *result) {
  cuDoubleComplex *dx, *dy, *d_result;
  cuDoubleComplex h_result;

  clap_cudaMalloc((void **)&dx, n * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dy, n * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_result, sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dx, x, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dy, y, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  clap_cublasZdotu(handle, (int)n, dx, (int)incx, dy, (int)incy, d_result);

  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(&h_result, d_result, sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  *result = std::complex<double>(cuCreal(h_result), cuCimag(h_result));

  clap_cudaFree(dx);
  clap_cudaFree(dy);
  clap_cudaFree(d_result);
}

// Complex Double DOTC_SUB
void CuBlasBackend::dotc_sub(int n, const std::complex<double> *x, int incx,
                             const std::complex<double> *y, int incy,
                             std::complex<double> *dotc) {
  cuDoubleComplex *dx = nullptr, *dy = nullptr;

  size_t sizeX = n * std::abs(incx);
  size_t sizeY = n * std::abs(incy);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dy, sizeY * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dx, x, sizeX * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dy, y, sizeY * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  clap_cublasZdotc(handle, (int)n, dx, (int)incx, dy, (int)incy,
                   reinterpret_cast<cuDoubleComplex *>(dotc));

  clap_cudaDeviceSynchronize();
  clap_cudaFree(dx);
  clap_cudaFree(dy);
}

// Complex Float NRM2
float CuBlasBackend::nrm2(int n, const std::complex<float> *x, int incx) {
  cuComplex *dx = nullptr;
  float result = 0.0f;

  size_t sizeX = static_cast<size_t>(n) * std::abs(incx);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuComplex));

  clap_cudaMemcpy(dx, reinterpret_cast<const cuComplex *>(x),
                  sizeX * sizeof(cuComplex), cudaMemcpyHostToDevice);

  clap_cublasScnrm2(handle, (int)n, dx, (int)incx, &result);

  clap_cudaDeviceSynchronize();
  clap_cudaFree(dx);

  return result;
}

// Complex Double NRM2
double CuBlasBackend::nrm2(int n, const std::complex<double> *x, int incx) {
  cuDoubleComplex *dx = nullptr;
  double result = 0.0f;

  size_t sizeX = static_cast<size_t>(n) * std::abs(incx);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dx, reinterpret_cast<const cuDoubleComplex *>(x),
                  sizeX * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  clap_cublasDznrm2(handle, (int)n, dx, (int)incx, &result);

  clap_cudaDeviceSynchronize();
  clap_cudaFree(dx);

  return result;
}

// Complex Float SCAL
void CuBlasBackend::scal(int n, const std::complex<float> *alpha,
                         std::complex<float> *x, int incx) {
  cuComplex *dx = nullptr;

  size_t sizeX = static_cast<size_t>(n) * std::abs(incx);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuComplex));

  clap_cudaMemcpy(dx, reinterpret_cast<const cuComplex *>(x),
                  sizeX * sizeof(cuComplex), cudaMemcpyHostToDevice);

  cuComplex alpha_c = make_cuComplex(alpha->real(), alpha->imag());

  clap_cublasCscal(handle, (int)n, &alpha_c, dx, (int)incx);

  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(reinterpret_cast<cuComplex *>(x), dx,
                  sizeX * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dx);
}

// Complex Double SCAL
void CuBlasBackend::scal(int n, const std::complex<double> *alpha,
                         std::complex<double> *x, int incx) {
  cuDoubleComplex *dx = nullptr;

  size_t sizeX = static_cast<size_t>(n) * std::abs(incx);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dx, reinterpret_cast<const cuDoubleComplex *>(x),
                  sizeX * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  cuDoubleComplex alpha_c = make_cuDoubleComplex(alpha->real(), alpha->imag());

  clap_cublasZscal(handle, (int)n, &alpha_c, dx, (int)incx);

  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(reinterpret_cast<cuDoubleComplex *>(x), dx,
                  sizeX * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dx);
}

// Complex Float SCAL
void CuBlasBackend::scal(int n, const float alpha, std::complex<float> *x,
                         int incx) {
  cuComplex *dx = nullptr;

  size_t sizeX = static_cast<size_t>(n) * std::abs(incx);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuComplex));

  clap_cudaMemcpy(dx, reinterpret_cast<const cuComplex *>(x),
                  sizeX * sizeof(cuComplex), cudaMemcpyHostToDevice);

  clap_cublasCsscal(handle, (int)n, &alpha, dx, (int)incx);

  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(reinterpret_cast<cuComplex *>(x), dx,
                  sizeX * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dx);
}

// Complex Double SCAL
void CuBlasBackend::scal(int n, const double alpha, std::complex<double> *x,
                         int incx) {
  cuDoubleComplex *dx = nullptr;

  size_t sizeX = static_cast<size_t>(n) * std::abs(incx);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dx, reinterpret_cast<const cuDoubleComplex *>(x),
                  sizeX * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  clap_cublasZdscal(handle, (int)n,
                    &alpha,
                    dx, (int)incx);

  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(reinterpret_cast<cuDoubleComplex *>(x), dx,
                  sizeX * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dx);
}

// Complex Float SWAP
void CuBlasBackend::swap(int n, std::complex<float> *x, int incx,
                         std::complex<float> *y, int incy) {
  cuComplex *dx = nullptr, *dy = nullptr;

  size_t sizeX = static_cast<size_t>(n) * std::abs(incx);
  size_t sizeY = static_cast<size_t>(n) * std::abs(incy);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dy, sizeY * sizeof(cuComplex));

  clap_cudaMemcpy(dx, reinterpret_cast<cuComplex *>(x),
                  sizeX * sizeof(cuComplex), cudaMemcpyHostToDevice);

  clap_cudaMemcpy(dy, reinterpret_cast<cuComplex *>(y),
                  sizeY * sizeof(cuComplex), cudaMemcpyHostToDevice);

  clap_cublasCswap(handle, (int)n, dx, (int)incx, dy, (int)incy);

  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(reinterpret_cast<cuComplex *>(x), dx,
                  sizeX * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  clap_cudaMemcpy(reinterpret_cast<cuComplex *>(y), dy,
                  sizeY * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dx);
  clap_cudaFree(dy);
}

// Complex Double SWAP
void CuBlasBackend::swap(int n, std::complex<double> *x, int incx,
                         std::complex<double> *y, int incy) {
  cuDoubleComplex *dx = nullptr, *dy = nullptr;

  size_t sizeX = static_cast<size_t>(n) * std::abs(incx);
  size_t sizeY = static_cast<size_t>(n) * std::abs(incy);

  clap_cudaMalloc((void **)&dx, sizeX * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dy, sizeY * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dx, reinterpret_cast<cuDoubleComplex *>(x),
                  sizeX * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  clap_cudaMemcpy(dy, reinterpret_cast<cuDoubleComplex *>(y),
                  sizeY * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

  clap_cublasZswap(handle, (int)n, dx, (int)incx, dy, (int)incy);

  clap_cudaDeviceSynchronize();
  clap_cudaMemcpy(reinterpret_cast<cuDoubleComplex *>(x), dx,
                  sizeX * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

  clap_cudaMemcpy(reinterpret_cast<cuDoubleComplex *>(y), dy,
                  sizeY * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dx);
  clap_cudaFree(dy);
}

}

