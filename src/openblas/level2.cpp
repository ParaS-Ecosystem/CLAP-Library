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

// Double GER
void OpenBlasBackend::ger(Layout layout, int m, int n, const double alpha,
                          const double *X, int incX, const double *Y, int incY,
                          double *A, int lda) {
  clap_cblas_dger(to_cblas_order(layout), (int)m, (int)n, alpha, X, (int)incX,
                  Y, (int)incY, A, (int)lda);
}

// Complex Float GERU
void OpenBlasBackend::geru(Layout layout, int m, int n,
                           const std::complex<float> *alpha,
                           const std::complex<float> *X, int incx,
                           const std::complex<float> *Y, int incy,
                           std::complex<float> *A, int lda) {
  clap_cblas_cgeru(to_cblas_order(layout), (int)m, (int)n, alpha, X, (int)incx,
                   Y, (int)incy, A, (int)lda);
}

// Complex Double GERU
void OpenBlasBackend::geru(Layout layout, int m, int n,
                           const std::complex<double> *alpha,
                           const std::complex<double> *X, int incx,
                           const std::complex<double> *Y, int incy,
                           std::complex<double> *A, int lda) {
  clap_cblas_zgeru(to_cblas_order(layout), (int)m, (int)n, alpha, X, (int)incx,
                   Y, (int)incy, A, (int)lda);
}

// Float SYR
void OpenBlasBackend::syr(Layout layout, Uplo uplo, int n, const float alpha,
                          const float *X, int incX, float *A, int lda) {
  clap_cblas_ssyr(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha, X,
                  (int)incX, A, (int)lda);
}

// Double SYR
void OpenBlasBackend::syr(Layout layout, Uplo uplo, int n, const double alpha,
                          const double *X, int incX, double *A, int lda) {
  clap_cblas_dsyr(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha, X,
                  (int)incX, A, (int)lda);
}

// Float SYR2
void OpenBlasBackend::syr2(Layout layout, Uplo uplo, int n, const float alpha,
                           const float *X, int incX, const float *Y, int incY,
                           float *A, int lda) {
  clap_cblas_ssyr2(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   X, (int)incX, Y, (int)incY, A, (int)lda);
}

// Double SYR2
void OpenBlasBackend::syr2(Layout layout, Uplo uplo, int n, const double alpha,
                           const double *X, int incX, const double *Y, int incY,
                           double *A, int lda) {
  clap_cblas_dsyr2(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   X, (int)incX, Y, (int)incY, A, (int)lda);
}

// Complex Float HER
void OpenBlasBackend::her(Layout layout, Uplo uplo, int n, const float alpha,
                          const std::complex<float> *x, int incx,
                          std::complex<float> *A, int lda) {
  clap_cblas_cher(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha, x,
                  (int)incx, A, (int)lda);
}

// Complex Double HER
void OpenBlasBackend::her(Layout layout, Uplo uplo, int n, const double alpha,
                          const std::complex<double> *x, int incx,
                          std::complex<double> *A, int lda) {
  clap_cblas_zher(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha, x,
                  (int)incx, A, (int)lda);
}

// Complex Float HER2
void OpenBlasBackend::her2(Layout layout, Uplo uplo, int n,
                           const std::complex<float> *alpha,
                           const std::complex<float> *x, int incx,
                           const std::complex<float> *y, int incy,
                           std::complex<float> *A, int lda) {
  clap_cblas_cher2(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   x, (int)incx, y, (int)incy, A, (int)lda);
}

// Complex Double HER2
void OpenBlasBackend::her2(Layout layout, Uplo uplo, int n,
                           const std::complex<double> *alpha,
                           const std::complex<double> *x, int incx,
                           const std::complex<double> *y, int incy,
                           std::complex<double> *A, int lda) {
  clap_cblas_zher2(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   x, (int)incx, y, (int)incy, A, (int)lda);
}

// Complex Float HPR
void OpenBlasBackend::hpr(Layout layout, Uplo uplo, int n, const float alpha,
                          const std::complex<float> *X, int incx,
                          std::complex<float> *ap) {
  clap_cblas_chpr(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha, X,
                  (int)incx, ap);
}

// Complex Double HPR
void OpenBlasBackend::hpr(Layout layout, Uplo uplo, int n, const double alpha,
                          const std::complex<double> *X, int incx,
                          std::complex<double> *ap) {
  clap_cblas_zhpr(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha, X,
                  (int)incx, ap);
}

// Complex Float HPR2
void OpenBlasBackend::hpr2(Layout layout, Uplo uplo, int n,
                           const std::complex<float> *alpha,
                           const std::complex<float> *X, int incx,
                           const std::complex<float> *Y, int incy,
                           std::complex<float> *ap) {
  clap_cblas_chpr2(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   X, (int)incx, Y, (int)incy, ap);
}

// Complex Double HPR2
void OpenBlasBackend::hpr2(Layout layout, Uplo uplo, int n,
                           const std::complex<double> *alpha,
                           const std::complex<double> *X, int incx,
                           const std::complex<double> *Y, int incy,
                           std::complex<double> *ap) {
  clap_cblas_zhpr2(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   X, (int)incx, Y, (int)incy, ap);
}

// Float GBMV
void OpenBlasBackend::gbmv(Layout layout, Transpose transpose, int m, int n,
                           int KL, int KU, const float alpha, const float *A,
                           int lda, const float *X, int incX, const float beta,
                           float *Y, int incY) {
  clap_cblas_sgbmv(to_cblas_order(layout), to_cblas_trans(transpose), (int)m,
                   (int)n, (int)KL, (int)KU, alpha, A, (int)lda, X, (int)incX,
                   beta, Y, (int)incY);
}

// Double GBMV
void OpenBlasBackend::gbmv(Layout layout, Transpose transpose, int m, int n,
                           int KL, int KU, const double alpha, const double *A,
                           int lda, const double *X, int incX,
                           const double beta, double *Y, int incY) {
  clap_cblas_dgbmv(to_cblas_order(layout), to_cblas_trans(transpose), (int)m,
                   (int)n, (int)KL, (int)KU, alpha, A, (int)lda, X, (int)incX,
                   beta, Y, (int)incY);
}

// Complex Float GBMV
void OpenBlasBackend::gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                           int KU, const std::complex<float> *alpha,
                           const std::complex<float> *A, int lda,
                           const std::complex<float> *X, int incX,
                           const std::complex<float> *beta,
                           std::complex<float> *Y, int incY) {
  clap_cblas_cgbmv(to_cblas_order(layout), to_cblas_trans(trans), (int)m,
                   (int)n, (int)KL, (int)KU, alpha, A, lda, X, incX, beta, Y,
                   incY);
}

// Complex Double GBMV
void OpenBlasBackend::gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                           int KU, const std::complex<double> *alpha,
                           const std::complex<double> *A, int lda,
                           const std::complex<double> *X, int incX,
                           const std::complex<double> *beta,
                           std::complex<double> *Y, int incY) {
  clap_cblas_zgbmv(to_cblas_order(layout), to_cblas_trans(trans), (int)m,
                   (int)n, (int)KL, (int)KU, alpha, A, lda, X, incX, beta, Y,
                   incY);
}

// Complex Float HBMV
void OpenBlasBackend::hbmv(Layout layout, Uplo uplo, int n, int k,
                           const std::complex<float> *alpha,
                           const std::complex<float> *A, int lda,
                           const std::complex<float> *X, int incx,
                           const std::complex<float> *beta,
                           std::complex<float> *Y, int incy) {
  clap_cblas_chbmv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, (int)k,
                   alpha, A, (int)lda, X, (int)incx, beta, Y, (int)incy);
}

// Complex Double HBMV
void OpenBlasBackend::hbmv(Layout layout, Uplo uplo, int n, int k,
                           const std::complex<double> *alpha,
                           const std::complex<double> *A, int lda,
                           const std::complex<double> *X, int incx,
                           const std::complex<double> *beta,
                           std::complex<double> *Y, int incy) {
  clap_cblas_zhbmv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, (int)k,
                   alpha, A, (int)lda, X, (int)incx, beta, Y, (int)incy);
}

// Float SBMV
void OpenBlasBackend::sbmv(Layout layout, Uplo uplo, int n, int k,
                           const float alpha, const float *A, int lda,
                           const float *x, int incX, const float beta, float *y,
                           int incY) {
  clap_cblas_ssbmv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, (int)k,
                   alpha, A, (int)lda, x, (int)incX, beta, y, (int)incY);
}

// Double SBMV
void OpenBlasBackend::sbmv(Layout layout, Uplo uplo, int n, int k,
                           const double alpha, const double *A, int lda,
                           const double *x, int incX, const double beta,
                           double *y, int incY) {
  clap_cblas_dsbmv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, (int)k,
                   alpha, A, (int)lda, x, (int)incX, beta, y, (int)incY);
}

// Float TBMV
void OpenBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, int K, const float *A, int lda, float *X,
                           int incX) {
  clap_cblas_stbmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, (int)K,
                   A, (int)lda, X, (int)incX);
}

// Double TBMV
void OpenBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, int K, const double *A, int lda, double *X,
                           int incX) {

  clap_cblas_dtbmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, (int)K,
                   A, (int)lda, X, (int)incX);
}

// Complex Float TBMV
void OpenBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, int k, const std::complex<float> *A, int lda,
                           std::complex<float> *X, int incx) {
  clap_cblas_ctbmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, (int)k,
                   A, (int)lda, X, (int)incx);
}

// Complex Double TBMV
void OpenBlasBackend::tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, int k, const std::complex<double> *A, int lda,
                           std::complex<double> *X, int incx) {
  clap_cblas_ztbmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, (int)k,
                   A, (int)lda, X, (int)incx);
}

// Float TBSV
void OpenBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, int K, const float *A, int lda, float *X,
                           int incX) {

  clap_cblas_stbsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, (int)K,
                   A, (int)lda, X, (int)incX);
}

// Double TBSV
void OpenBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, int K, const double *A, int lda, double *X,
                           int incX) {

  clap_cblas_dtbsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, (int)K,
                   A, (int)lda, X, (int)incX);
}

// Complex Float TBSV
void OpenBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, int K, const std::complex<float> *A, int lda,
                           std::complex<float> *X, int incx) {
  clap_cblas_ctbsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, (int)K,
                   A, (int)lda, X, (int)incx);
}

// Complex Double TBSV
void OpenBlasBackend::tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, int K, const std::complex<double> *A, int lda,
                           std::complex<double> *X, int incx) {
  clap_cblas_ztbsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, (int)K,
                   A, (int)lda, X, (int)incx);
}

// Complex Float HPMV
void OpenBlasBackend::hpmv(Layout layout, Uplo uplo, int n,
                           const std::complex<float> *alpha,
                           const std::complex<float> *Ap,
                           const std::complex<float> *X, int incx,
                           const std::complex<float> *beta,
                           std::complex<float> *Y, int incy) {
  clap_cblas_chpmv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   Ap, X, (int)incx, beta, Y, (int)incy);
}

// Complex Double HPMV
void OpenBlasBackend::hpmv(Layout layout, Uplo uplo, int n,
                           const std::complex<double> *alpha,
                           const std::complex<double> *Ap,
                           const std::complex<double> *X, int incx,
                           const std::complex<double> *beta,
                           std::complex<double> *Y, int incy) {
  clap_cblas_zhpmv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   Ap, X, (int)incx, beta, Y, (int)incy);
}

// Float SPMV
void OpenBlasBackend::spmv(Layout layout, Uplo uplo, int n, const float alpha,
                           const float *Ap, const float *x, int incX,
                           const float beta, float *y, int incY) {
  clap_cblas_sspmv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   Ap, x, incX, beta, y, incY);
}

// Double SPMV
void OpenBlasBackend::spmv(Layout layout, Uplo uplo, int n, const double alpha,
                           const double *Ap, const double *x, int incX,
                           const double beta, double *y, int incY) {
  clap_cblas_dspmv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   Ap, x, incX, beta, y, incY);
}

// Float TPMV
void OpenBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const float *Ap, float *X, int incX) {

  clap_cblas_stpmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, Ap, X,
                   (int)incX);
}

// Double TPMV
void OpenBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const double *Ap, double *X, int incX) {

  clap_cblas_dtpmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, Ap, X,
                   (int)incX);
}

// Complex Float TPMV
void OpenBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const std::complex<float> *Ap,
                           std::complex<float> *X, int incx) {
  clap_cblas_ctpmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, Ap, X,
                   (int)incx);
}

// Complex Double TPMV
void OpenBlasBackend::tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const std::complex<double> *Ap,
                           std::complex<double> *X, int incx) {
  clap_cblas_ztpmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, Ap, X,
                   (int)incx);
}

// Float TPSV
void OpenBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const float *Ap, float *X, int incX) {

  clap_cblas_stpsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, Ap, X,
                   (int)incX);
}

// Double TPSV
void OpenBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const double *Ap, double *X, int incX) {

  clap_cblas_dtpsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, Ap, X,
                   (int)incX);
}

// Complex Float TPSV
void OpenBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const std::complex<float> *Ap,
                           std::complex<float> *X, int incX) {
  clap_cblas_ctpsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, Ap, X,
                   (int)incX);
}

// Complex Double TPSV
void OpenBlasBackend::tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int n, const std::complex<double> *Ap,
                           std::complex<double> *X, int incX) {
  clap_cblas_ztpsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, Ap, X,
                   (int)incX);
}

// Float SPR
void OpenBlasBackend::spr(Layout layout, Uplo uplo, int n, const float alpha,
                          const float *X, int incX, float *Ap) {
  clap_cblas_sspr(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha, X,
                  (int)incX, Ap);
}

// Double SPR
void OpenBlasBackend::spr(Layout layout, Uplo uplo, int n, const double alpha,
                          const double *X, int incX, double *Ap) {
  clap_cblas_dspr(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha, X,
                  (int)incX, Ap);
}

// Float SPR2
void OpenBlasBackend::spr2(Layout layout, Uplo uplo, int n, const float alpha,
                           const float *X, int incX, const float *Y, int incY,
                           float *A) {

  clap_cblas_sspr2(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   X, (int)incX, Y, (int)incY, A);
}

// Double SPR2
void OpenBlasBackend::spr2(Layout layout, Uplo uplo, int n, const double alpha,
                           const double *X, int incX, const double *Y, int incY,
                           double *A) {

  clap_cblas_dspr2(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   X, (int)incX, Y, (int)incY, A);
}

}

