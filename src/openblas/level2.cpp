// -----------------------------------------------------------------------------
// This file is part of the CLAP library — an implementation of
// high-performance linear algebra routines based on the BLAS standard.
//
// CLAP provides a unified math interface across architectures,
// enabling portable and optimized performance on CPUs and GPUs.
//
// Copyright (C) The CLAP Contributors
//
// CLAP is released under the GNU Lesser General Public License (LGPL) v3.
// See the file LICENSE in the project root for full license details.
// -----------------------------------------------------------------------------

#include "clap/openblas_backend.hpp"
#include "clap/openblas_utils.hpp"

namespace clap {

// Level 2 Blas routines

// Float gemv

void OpenBlasBackend::gemv(Layout layout, Transpose trans, int64_t m, int64_t n,
                           const float alpha, const float *A, int64_t lda,
                           const float *x, int64_t incx, const float beta,
                           float *y, int64_t incy) {
  clap_cblas_sgemv(to_cblas_order(layout), to_cblas_trans(trans), (int)m,
                   (int)n, alpha, A, (int)lda, x, (int)incx, beta, y,
                   (int)incy);
}

// Double GEMV

void OpenBlasBackend::gemv(Layout layout, Transpose trans, int64_t m, int64_t n,
                           const double alpha, const double *A, int64_t lda,
                           const double *x, int64_t incx, const double beta,
                           double *y, int64_t incy) {
  clap_cblas_dgemv(to_cblas_order(layout), to_cblas_trans(trans), (int)m,
                   (int)n, alpha, A, (int)lda, x, (int)incx, beta, y,
                   (int)incy);
}

// FLOAT SYMV

void OpenBlasBackend::symv(Layout layout, Uplo uplo, int64_t n,
                           const float alpha, const float *A, int64_t lda,
                           const float *X, int64_t incx, const float beta,
                           float *Y, int64_t incy) {
  clap_cblas_ssymv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   A, (int)lda, X, (int)incx, beta, Y, (int)incy);
}

// Double SYMV

void OpenBlasBackend::symv(Layout layout, Uplo uplo, int64_t n,
                           const double alpha, const double *A, int64_t lda,
                           const double *X, int64_t incx, const double beta,
                           double *Y, int64_t incy) {
  clap_cblas_dsymv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, alpha,
                   A, (int)lda, X, (int)incx, beta, Y, (int)incy);
}

// FLOAT trmv

void OpenBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int64_t n, const float *A, int64_t lda, float *X,
                           int64_t incX) {
  clap_cblas_strmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A,
                   (int)lda, X, (int)incX);
}

// Double trmv

void OpenBlasBackend::trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int64_t n, const double *A, int64_t lda, double *X,
                           int64_t incX) {
  clap_cblas_dtrmv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A,
                   (int)lda, X, (int)incX);
}

// FLOAT SYMV

void OpenBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int64_t n, const float *A, int64_t lda, float *X,
                           int64_t incX) {
  clap_cblas_strsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A,
                   (int)lda, X, (int)incX);
}

// FLOAT SYMV

void OpenBlasBackend::trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                           int64_t n, const double *A, int64_t lda, double *X,
                           int64_t incX) {
  clap_cblas_dtrsv(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(trans), to_cblas_diag(diag), (int)n, A,
                   (int)lda, X, (int)incX);
}

// Float SBMV

void OpenBlasBackend::sbmv(Layout layout, Uplo uplo, int64_t n, int64_t k,
                           const float alpha, const float *A, int64_t lda,
                           const float *x, int64_t incX, const float beta,
                           float *y, int64_t incY) {
  clap_cblas_ssbmv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, (int)k,
                   alpha, A, (int)lda, x, (int)incX, beta, y, (int)incY);
}

// Double SBMV

void OpenBlasBackend::sbmv(Layout layout, Uplo uplo, int64_t n, int64_t k,
                           const double alpha, const double *A, int64_t lda,
                           const double *x, int64_t incX, const double beta,
                           double *y, int64_t incY) {
  clap_cblas_dsbmv(to_cblas_order(layout), to_cblas_uplo(uplo), (int)n, (int)k,
                   alpha, A, (int)lda, x, (int)incX, beta, y, (int)incY);
}

} // namespace clap
