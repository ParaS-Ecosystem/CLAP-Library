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

void OpenBlasBackend::OpenBlasBackend::copy(int64_t n, const float *x,
                                            int64_t incx, float *y,
                                            int64_t incy) {
  clap_cblas_scopy(n, x, incx, y, incy);
}

// Double COPY

void OpenBlasBackend::OpenBlasBackend::copy(int64_t n, const double *x,
                                            int64_t incx, double *y,
                                            int64_t incy) {
  clap_cblas_dcopy(n, x, incx, y, incy);
}

// Float SCAL

void OpenBlasBackend::scal(int64_t n, float a, float *x, int64_t incx) {
  clap_cblas_sscal(n, a, x, incx);
}

// Double SCAL

void OpenBlasBackend::scal(int64_t n, double a, double *x, int64_t incx) {
  clap_cblas_dscal(n, a, x, incx);
}

// Float axpy

void OpenBlasBackend::axpy(int64_t n, float a, const float *x, int64_t incx,
                           float *y, int64_t incy) {
  clap_cblas_saxpy(n, a, x, incx, y, incy);
}

// Double axpy

void OpenBlasBackend::axpy(int64_t n, double a, const double *x, int64_t incx,
                           double *y, int64_t incy) {
  clap_cblas_daxpy(n, a, x, incx, y, incy);
}

// Float swap

void OpenBlasBackend::swap(int64_t n, float *x, int64_t incx, float *y,
                           int64_t incy) {
  clap_cblas_sswap(n, x, incx, y, incy);
}

// Double swap

void OpenBlasBackend::swap(int64_t n, double *x, int64_t incx, double *y,
                           int64_t incy) {
  clap_cblas_dswap(n, x, incx, y, incy);
}

// Double dot

float OpenBlasBackend::dot(int64_t n, const float *x, int64_t incx,
                           const float *y, int64_t incy) {
  return clap_cblas_sdot(n, x, incx, y, incy);
}

// Float dot

double OpenBlasBackend::dot(int64_t n, const double *x, int64_t incx,
                            const double *y, int64_t incy) {
  return clap_cblas_ddot(n, x, incx, y, incy);
}

} // namespace clap
