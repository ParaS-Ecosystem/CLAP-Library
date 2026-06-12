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

#include "clap/openblas_backend.hpp"
#include "clap/openblas_utils.hpp"
namespace clap {

// Level-3  BLAS Routines

// Float GEMM

void OpenBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                           int64_t m, int64_t n, int64_t k, float alpha,
                           const float *A, int64_t lda, const float *B,
                           int64_t ldb, float beta, float *C, int64_t ldc) {
  clap_cblas_sgemm(to_cblas_order(layout), to_cblas_trans(transA),
                   to_cblas_trans(transB), (int)m, (int)n, (int)k, alpha, A,
                   (int)lda, B, (int)ldb, beta, C, (int)ldc);
}

// Double GEMM

void OpenBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                           int64_t m, int64_t n, int64_t k, double alpha,
                           const double *A, int64_t lda, const double *B,
                           int64_t ldb, double beta, double *C, int64_t ldc) {
  clap_cblas_dgemm(to_cblas_order(layout), to_cblas_trans(transA),
                   to_cblas_trans(transB), (int)m, (int)n, (int)k, alpha, A,
                   (int)lda, B, (int)ldb, beta, C, (int)ldc);
}

// Float SYMM

void OpenBlasBackend::symm(Layout layout, Side side, Uplo uplo, int64_t m,
                           int64_t n, const float alpha, const float *A,
                           int64_t lda, const float *B, int64_t ldb,
                           const float beta, float *C, int64_t ldc) {
  clap_cblas_ssymm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb, beta, C, (int)ldc);
}

// Double SYMM

void OpenBlasBackend::symm(Layout layout, Side side, Uplo uplo, int64_t m,
                           int64_t n, const double alpha, const double *A,
                           int64_t lda, const double *B, int64_t ldb,
                           const double beta, double *C, int64_t ldc) {

  clap_cblas_dsymm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb, beta, C, (int)ldc);
}

// FLoat SYRK

void OpenBlasBackend::syrk(Layout layout, Uplo uplo, Transpose transA,
                           int64_t n, int64_t K, const float alpha,
                           const float *A, int64_t lda, const float beta,
                           float *C, int64_t ldc) {

  clap_cblas_ssyrk(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(transA), (int)n, (int)K, alpha, A, (int)lda,
                   beta, C, (int)ldc);
}

// Double SYRK

void OpenBlasBackend::syrk(Layout layout, Uplo uplo, Transpose transA,
                           int64_t n, int64_t K, const double alpha,
                           const double *A, int64_t lda, const double beta,
                           double *C, int64_t ldc) {

  clap_cblas_dsyrk(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(transA), (int)n, (int)K, alpha, A, (int)lda,
                   beta, C, (int)ldc);
}

// Float TRMM

void OpenBlasBackend::trmm(Layout layout, Side side, Uplo uplo,
                           Transpose transA, Diag diag, int64_t m, int64_t n,
                           const float alpha, const float *A, int64_t lda,
                           float *B, int64_t ldb) {

  clap_cblas_strmm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), to_cblas_trans(transA),
                   to_cblas_diag(diag), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb);
}

// Double TRMM

void OpenBlasBackend::trmm(Layout layout, Side side, Uplo uplo,
                           Transpose transA, Diag diag, int64_t m, int64_t n,
                           const double alpha, const double *A, int64_t lda,
                           double *B, int64_t ldb) {

  clap_cblas_dtrmm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), to_cblas_trans(transA),
                   to_cblas_diag(diag), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb);
}

// Float TRSM

void OpenBlasBackend::trsm(Layout layout, Side side, Uplo uplo,
                           Transpose transA, Diag diag, int64_t m, int64_t n,
                           float alpha, const float *A, int64_t lda, float *B,
                           int64_t ldb) {
  clap_cblas_strsm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), to_cblas_trans(transA),
                   to_cblas_diag(diag), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb);
}

// Double TRSM

void OpenBlasBackend::trsm(Layout layout, Side side, Uplo uplo,
                           Transpose transA, Diag diag, int64_t m, int64_t n,
                           double alpha, const double *A, int64_t lda,
                           double *B, int64_t ldb) {

  clap_cblas_dtrsm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), to_cblas_trans(transA),
                   to_cblas_diag(diag), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb);
}

} // namespace clap
