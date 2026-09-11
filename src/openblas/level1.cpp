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

// Float IAMAX

int OpenBlasBackend::iamax(int n, const float *x, int incx) {
  return clap_cblas_isamax(n, x, incx);
}

// Double IAMAX

int OpenBlasBackend::iamax(int n, const double *x, int incx) {
  return clap_cblas_idamax(n, x, incx);
}

// Float NRM2

float OpenBlasBackend::nrm2(int n, const float *x, int incx) {
  return clap_cblas_snrm2(n, x, incx);
}

// Double NRM2

double OpenBlasBackend::nrm2(int n, const double *x, int incx) {
  return clap_cblas_dnrm2(n, x, incx);
}


} // namespace clap
