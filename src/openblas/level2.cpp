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
#include "clap/openblas_utils.hpp"
namespace clap {

// Float GEMV
void OpenBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                           const float alpha, const float *A, int lda,
                           const float *x, int incx, const float beta, float *y,
                           int incy) {
  clap_cblas_sgemv(to_cblas_order(layout), to_cblas_trans(trans), (int)m,
                   (int)n, alpha, A, (int)lda, x, (int)incx, beta, y,
                   (int)incy);
}

// Double GEMV
void OpenBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                           const double alpha, const double *A, int lda,
                           const double *x, int incx, const double beta,
                           double *y, int incy) {
  clap_cblas_dgemv(to_cblas_order(layout), to_cblas_trans(trans), (int)m,
                   (int)n, alpha, A, (int)lda, x, (int)incx, beta, y,
                   (int)incy);
}

// Complex Float GEMV
void OpenBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                           const std::complex<float> *alpha,
                           const std::complex<float> *A, int lda,
                           const std::complex<float> *x, int incx,
                           const std::complex<float> *beta,
                           std::complex<float> *y, int incy) {
  clap_cblas_cgemv(to_cblas_order(layout), to_cblas_trans(trans), (int)m,
                   (int)n, alpha, A, (int)lda, x, (int)incx, beta, y,
                   (int)incy);
}

// Complex Double GEMV
void OpenBlasBackend::gemv(Layout layout, Transpose trans, int m, int n,
                           const std::complex<double> *alpha,
                           const std::complex<double> *A, int lda,
                           const std::complex<double> *x, int incx,
                           const std::complex<double> *beta,
                           std::complex<double> *y, int incy) {
  clap_cblas_zgemv(to_cblas_order(layout), to_cblas_trans(trans), (int)m,
                   (int)n, alpha, A, (int)lda, x, (int)incx, beta, y,
                   (int)incy);
}

// Float SYMV
void OpenBlasBackend::symv(Layout layout, Uplo uplo, int n, const float alpha,
                           const float *A, int lda, const float *X, int incx,
                           const float beta, float *Y, int incy) {
  clap_cblas_ssymv(to_cblas_order(layout), to_cblas_uplo(uplo), n, alpha, A,
                   lda, X, incx, beta, Y, incy);
}

// Double SYMV
void OpenBlasBackend::symv(Layout layout, Uplo uplo, int n, const double alpha,
                           const double *A, int lda, const double *X, int incx,
                           const double beta, double *Y, int incy)

{
  clap_cblas_dsymv(to_cblas_order(layout), to_cblas_uplo(uplo), n, alpha, A,
                   lda, X, incx, beta, Y, incy);
}

// Complex Float HEMV
void OpenBlasBackend::hemv(Layout layout, Uplo uplo, int n,
                           const std::complex<float> *alpha,
                           const std::complex<float> *A, int lda,
                           const std::complex<float> *x, int incx,
                           const std::complex<float> *beta,
                           std::complex<float> *y, int incy) {
  clap_cblas_chemv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   A, (int)lda, x, (int)incx, beta, y, (int)incy);
}

// Complex Double HEMV
void OpenBlasBackend::hemv(Layout layout, Uplo uplo, int n,
                           const std::complex<double> *alpha,
                           const std::complex<double> *A, int lda,
                           const std::complex<double> *x, int incx,
                           const std::complex<double> *beta,
                           std::complex<double> *y, int incy) {
  clap_cblas_zhemv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   A, (int)lda, x, (int)incx, beta, y, (int)incy);
}

// Float TRMV
void OpenBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const float *A, int lda, float *X, int incX) {
  clap_cblas_strmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A,
                   (int)lda, X, (int)incX);
}

// Double TRMV
void OpenBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const double *A, int lda, double *X,
                           int incX) {

  clap_cblas_dtrmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A,
                   (int)lda, X, (int)incX);
}

// Complex Float TRMV
void OpenBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const std::complex<float> *A, int lda,
                           std::complex<float> *X, int incx) {
  clap_cblas_ctrmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A, lda,
                   X, (int)incx);
}

// Complex Double TRMV
void OpenBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const std::complex<double> *A, int lda,
                           std::complex<double> *X, int incx) {
  clap_cblas_ztrmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A, lda,
                   X, (int)incx);
}

// Float TRSV
void OpenBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const float *A, int lda, float *X, int incX) {
  clap_cblas_strsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A,
                   (int)lda, X, (int)incX);
}

// Double TRSV
void OpenBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const double *A, int lda, double *X,
                           int incX) {
  clap_cblas_dtrsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A,
                   (int)lda, X, (int)incX);
}

// Complex Float TRSV
void OpenBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const std::complex<float> *A, int lda,
                           std::complex<float> *X, int incX) {
  clap_cblas_ctrsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A,
                   (int)lda, X, (int)incX);
}

// Complex Double TRSV
void OpenBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const std::complex<double> *A, int lda,
                           std::complex<double> *X, int incX) {
  clap_cblas_ztrsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A,
                   (int)lda, X, (int)incX);
}

// Complex Float GERC
void OpenBlasBackend::gerc(Layout layout, int m, int n,
                           const std::complex<float> *alpha,
                           const std::complex<float> *X, int incx,
                           const std::complex<float> *Y, int incy,
                           std::complex<float> *A, int lda) {
  clap_cblas_cgerc(to_cblas_order(layout), (int)m, (int)n, alpha, X, (int)incx,
                   Y, (int)incy, A, (int)lda);
}

// Complex Double GERC
void OpenBlasBackend::gerc(Layout layout, int m, int n,
                           const std::complex<double> *alpha,
                           const std::complex<double> *X, int incx,
                           const std::complex<double> *Y, int incy,
                           std::complex<double> *A, int lda) {
  clap_cblas_zgerc(to_cblas_order(layout), (int)m, (int)n, alpha, X, (int)incx,
                   Y, (int)incy, A, (int)lda);
}

// Float GER
void OpenBlasBackend::ger(Layout layout, int m, int n, const float alpha,
                          const float *X, int incX, const float *Y, int incY,
                          float *A, int lda) {
  clap_cblas_sger(to_cblas_order(layout), (int)m, (int)n, alpha, X, (int)incX,
                  Y, (int)incY, A, (int)lda);
}

//GER Double

void OpenBlasBackend::ger(Layout layout, int m, int n, const double alpha,
                          const double *X, int incX, const double *Y, int incY,
                          double *A, int lda) {
  clap_cblas_dger(to_cblas_order(layout), (int)m, (int)n, alpha, X, (int)incX,
                  Y, (int)incY, A, (int)lda);
}

//SYR Float

void OpenBlasBackend::syr(Layout layout, Uplo uplo, int n, const float alpha,
                          const float *X, int incX, float *A, int lda) {
  clap_cblas_ssyr(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha, X,
                  (int)incX, A, (int)lda);
}

//SYR Double

void OpenBlasBackend::syr(Layout layout, Uplo uplo, int n, const double alpha,
                          const double *X, int incX, double *A, int lda) {
  clap_cblas_dsyr(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha, X,
                  (int)incX, A, (int)lda);
}

//GBMV Float

void OpenBlasBackend::gbmv(Layout layout, Transpose transpose, int m, int n,
                           int KL, int KU, const float alpha, const float *A,
                           int lda, const float *X, int incX, const float beta,
                           float *Y, int incY) {
  clap_cblas_sgbmv(to_cblas_order(layout), to_cblas_trans(transpose), (int)m,
                   (int)n, (int)KL, (int)KU, alpha, A, (int)lda, X, (int)incX,
                   beta, Y, (int)incY);
}

//GBMV Double

void OpenBlasBackend::gbmv(Layout layout, Transpose transpose, int m, int n,
                           int KL, int KU, const double alpha, const double *A,
                           int lda, const double *X, int incX,
                           const double beta, double *Y, int incY) {
  clap_cblas_dgbmv(to_cblas_order(layout), to_cblas_trans(transpose), (int)m,
                   (int)n, (int)KL, (int)KU, alpha, A, (int)lda, X, (int)incX,
                   beta, Y, (int)incY);
}



} // namespace clap
