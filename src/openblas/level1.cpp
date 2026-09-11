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

#include "clap/openblas_backend.hpp"

namespace clap {

// Float COPY
void OpenBlasBackend::copy(int n, const float *x, int incx,
                                            float *y, int incy) {
  clap_cblas_scopy(n, x, incx, y, incy);
}

// Double COPY
void OpenBlasBackend::copy(int n, const double *x, int incx,
                                            double *y, int incy) {
  clap_cblas_dcopy(n, x, incx, y, incy);
}

// Float SCAL
void OpenBlasBackend::scal(int n, float a, float *x, int incx) {
  clap_cblas_sscal(n, a, x, incx);
}

// Double SCAL
void OpenBlasBackend::scal(int n, double a, double *x, int incx) {
  clap_cblas_dscal(n, a, x, incx);
}

// Float AXPY
void OpenBlasBackend::axpy(int n, float a, const float *x, int incx, float *y,
                           int incy) {
  clap_cblas_saxpy(n, a, x, incx, y, incy);
}

// Double AXPY
void OpenBlasBackend::axpy(int n, double a, const double *x, int incx,
                           double *y, int incy) {
  clap_cblas_daxpy(n, a, x, incx, y, incy);
}

// Float IAMAX
int OpenBlasBackend::iamax(int n, const float *x, int incx) {
  return clap_cblas_isamax(n, x, incx);
}

// Double IAMAX
int OpenBlasBackend::iamax(int n, const double *x, int incx) {
  return clap_cblas_idamax(n, x, incx);
}

// Float SWAP
void OpenBlasBackend::swap(int n, float *x, int incx, float *y, int incy) {
  clap_cblas_sswap(n, x, incx, y, incy);
}

// Double SWAP
void OpenBlasBackend::swap(int n, double *x, int incx, double *y, int incy) {
  clap_cblas_dswap(n, x, incx, y, incy);
}

// Float DOT
float OpenBlasBackend::dot(int n, const float *x, int incx, const float *y,
                           int incy) {
  return clap_cblas_sdot(n, x, incx, y, incy);
}

// Double DOT
double OpenBlasBackend::dot(int n, const double *x, int incx, const double *y,
                            int incy) {
  return clap_cblas_ddot(n, x, incx, y, incy);
}

// Float NRM2
float OpenBlasBackend::nrm2(int n, const float *x, int incx) {
  return clap_cblas_snrm2(n, x, incx);
}

// Double NRM2
double OpenBlasBackend::nrm2(int n, const double *x, int incx) {
  return clap_cblas_dnrm2(n, x, incx);
}

// Float ASUM
float OpenBlasBackend::asum(int n, const float *x, int incx) {
  return clap_cblas_sasum(n, x, incx);
}

// Double ASUM
double OpenBlasBackend::asum(int n, const double *x, int incx) {
  return clap_cblas_dasum(n, x, incx);
}

// Float ROTG
void OpenBlasBackend::rotg(float *a, float *b, float *c, float *s) {
  clap_cblas_srotg(a, b, c, s);
}

// Double ROTG
void OpenBlasBackend::rotg(double *a, double *b, double *c, double *s) {
  clap_cblas_drotg(a, b, c, s);
}

// Float ROT
void OpenBlasBackend::rot(int n, float *X, int incX, float *Y, int incY,
                          const float c, const float s) {
  clap_cblas_srot(n, X, incX, Y, incY, c, s);
}

// Double ROT
void OpenBlasBackend::rot(int n, double *X, int incX, double *Y, int incY,
                          const double c, const double s) {
  clap_cblas_drot(n, X, incX, Y, incY, c, s);
}

// Float ROTMG
void OpenBlasBackend::rotmg(float *d1, float *d2, float *b1, const float b2,
                            float *P) {
  clap_cblas_srotmg(d1, d2, b1, b2, P);
}

// Double ROTMG
void OpenBlasBackend::rotmg(double *d1, double *d2, double *b1, const double b2,
                            double *P) {
  clap_cblas_drotmg(d1, d2, b1, b2, P);
}

// Float ROTM
void OpenBlasBackend::rotm(int n, float *X, int incX, float *Y, int incY,
                           const float *P) {
  clap_cblas_srotm(n, X, incX, Y, incY, P);
}

// Double ROTM
void OpenBlasBackend::rotm(int n, double *X, int incX, double *Y, int incY,
                           const double *P) {
  clap_cblas_drotm(n, X, incX, Y, incY, P);
}

// Complex Float DOTU_SUB
void OpenBlasBackend::dotu_sub(int n, const std::complex<float> *X, int incX,
                               const std::complex<float> *Y, int incY,
                               std::complex<float> *dotu) {
  clap_cblas_cdotu_sub((int)n, X, (int)incX, Y, (int)incY, dotu);
}

// Complex Float DOTC_SUB
void OpenBlasBackend::dotc_sub(int n, const std::complex<float> *X, int incX,
                               const std::complex<float> *Y, int incY,
                               std::complex<float> *dotc) {
  clap_cblas_cdotc_sub((int)n, X, (int)incX, Y, (int)incY, dotc);
}

// Complex Double DOTU_SUB
void OpenBlasBackend::dotu_sub(int n, const std::complex<double> *X, int incX,
                               const std::complex<double> *Y, int incY,
                               std::complex<double> *dotu) {
  clap_cblas_zdotu_sub((int)n, X, (int)incX, Y, (int)incY, dotu);
}

// Complex Double DOTC_SUB
void OpenBlasBackend::dotc_sub(int n, const std::complex<double> *X, int incX,
                               const std::complex<double> *Y, int incY,
                               std::complex<double> *dotc) {
  clap_cblas_zdotc_sub((int)n, X, (int)incX, Y, (int)incY, dotc);
}

// Complex Float SWAP
void OpenBlasBackend::swap(int n, std::complex<float> *X, int incX,
                           std::complex<float> *Y, int incY) {
  clap_cblas_cswap((int)n, X, (int)incX, Y, (int)incY);
}

// Complex Double SWAP
void OpenBlasBackend::swap(int n, std::complex<double> *X, int incX,
                           std::complex<double> *Y, int incY) {
  clap_cblas_zswap((int)n, X, (int)incX, Y, (int)incY);
}

// Complex Float COPY
void OpenBlasBackend::copy(int n, const std::complex<float> *X, int incX,
                           std::complex<float> *Y, int incY) {
  clap_cblas_ccopy((int)n, X, (int)incX, Y, (int)incY);
}

// Complex Double COPY
void OpenBlasBackend::copy(int n, const std::complex<double> *X, int incX,
                           std::complex<double> *Y, int incY) {
  clap_cblas_zcopy((int)n, X, (int)incX, Y, (int)incY);
}

// Complex Float SCAL
void OpenBlasBackend::scal(int n, const std::complex<float> *alpha,
                           std::complex<float> *X, int incX) {
  clap_cblas_cscal((int)n, alpha, X, (int)incX);
}

// Complex Double SCAL
void OpenBlasBackend::scal(int n, const std::complex<double> *alpha,
                           std::complex<double> *X, int incX) {
  clap_cblas_zscal((int)n, alpha, X, (int)incX);
}

// Complex Float SCAL
void OpenBlasBackend::scal(int n, const float alpha, std::complex<float> *X,
                           int incX) {
  clap_cblas_csscal((int)n, alpha, X, (int)incX);
}

// Complex Double SCAL
void OpenBlasBackend::scal(int n, const double alpha, std::complex<double> *X,
                           int incX) {
  clap_cblas_zdscal((int)n, alpha, X, (int)incX);
}

// Complex Float AXPY
void OpenBlasBackend::axpy(int n, const std::complex<float> *alpha,
                           const std::complex<float> *X, int incX,
                           std::complex<float> *Y, int incY) {
  clap_cblas_caxpy((int)n, alpha, X, (int)incX, Y, (int)incY);
}

// Complex Double AXPY
void OpenBlasBackend::axpy(int n, const std::complex<double> *alpha,
                           const std::complex<double> *X, int incX,
                           std::complex<double> *Y, int incY) {
  clap_cblas_zaxpy((int)n, alpha, X, (int)incX, Y, (int)incY);
}

// Complex Float NRM2
float OpenBlasBackend::nrm2(int n, const std::complex<float> *X, int incX) {
  return (clap_cblas_scnrm2((int)n, X, (int)incX));
}

// Complex Float ASUM
float OpenBlasBackend::asum(int n, const std::complex<float> *X, int incX) {
  return (clap_cblas_scasum((int)n, X, (int)incX));
}

// Complex Double NRM2
double OpenBlasBackend::nrm2(int n, const std::complex<double> *X, int incX) {
  return (clap_cblas_dznrm2((int)n, X, (int)incX));
}

// Complex Double ASUM
double OpenBlasBackend::asum(int n, const std::complex<double> *X, int incX) {
  return (clap_cblas_dzasum((int)n, X, (int)incX));
}

// Complex Float IAMAX
int OpenBlasBackend::iamax(int n, const std::complex<float> *X, int incX) {
  return (clap_cblas_icamax((int)n, X, (int)incX));
}

// Complex Double IAMAX
int OpenBlasBackend::iamax(int n, const std::complex<double> *X, int incX) {
  return (clap_cblas_izamax((int)n, X, (int)incX));
}

}

