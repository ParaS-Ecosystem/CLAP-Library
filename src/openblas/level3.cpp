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

// Float GEMM
void OpenBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                           int m, int n, int k, float alpha, const float *A,
                           int lda, const float *B, int ldb, float beta,
                           float *C, int ldc) {
  clap_cblas_sgemm(to_cblas_order(layout), to_cblas_trans(transA),
                   to_cblas_trans(transB), (int)m, (int)n, (int)k, alpha, A,
                   (int)lda, B, (int)ldb, beta, C, (int)ldc);
}

// Double GEMM
void OpenBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                           int m, int n, int k, double alpha, const double *A,
                           int lda, const double *B, int ldb, double beta,
                           double *C, int ldc) {
  clap_cblas_dgemm(to_cblas_order(layout), to_cblas_trans(transA),
                   to_cblas_trans(transB), (int)m, (int)n, (int)k, alpha, A,
                   (int)lda, B, (int)ldb, beta, C, (int)ldc);
}

// Float SYMM
void OpenBlasBackend::symm(Layout layout, Side side, Uplo uplo, int m, int n,
                           const float alpha, const float *A, int lda,
                           const float *B, int ldb, const float beta, float *C,
                           int ldc) {
  clap_cblas_ssymm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb, beta, C, (int)ldc);
}

// Double SYMM
void OpenBlasBackend::symm(Layout layout, Side side, Uplo uplo, int m, int n,
                           const double alpha, const double *A, int lda,
                           const double *B, int ldb, const double beta,
                           double *C, int ldc) {

  clap_cblas_dsymm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb, beta, C, (int)ldc);
}

// Float SYRK
void OpenBlasBackend::syrk(Layout layout, Uplo uplo, Transpose transA, int n,
                           int K, const float alpha, const float *A, int lda,
                           const float beta, float *C, int ldc) {

  clap_cblas_ssyrk(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(transA), (int)n, (int)K, alpha, A, (int)lda,
                   beta, C, (int)ldc);
}

// Double SYRK
void OpenBlasBackend::syrk(Layout layout, Uplo uplo, Transpose transA, int n,
                           int K, const double alpha, const double *A, int lda,
                           const double beta, double *C, int ldc) {

  clap_cblas_dsyrk(to_cblas_order(layout), to_cblas_uplo(uplo),
                   to_cblas_trans(transA), (int)n, (int)K, alpha, A, (int)lda,
                   beta, C, (int)ldc);
}

// Float SYR2K
void OpenBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose transA, int n,
                            int K, const float alpha, const float *A, int lda,
                            const float *B, int ldb, const float beta, float *C,
                            int ldc) {

  clap_cblas_ssyr2k(to_cblas_order(layout), to_cblas_uplo(uplo),
                    to_cblas_trans(transA), (int)n, (int)K, alpha, A, (int)lda,
                    B, (int)ldb, beta, C, (int)ldc);
}

// Double SYR2K
void OpenBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose transA, int n,
                            int K, const double alpha, const double *A, int lda,
                            const double *B, int ldb, const double beta,
                            double *C, int ldc) {
  clap_cblas_dsyr2k(to_cblas_order(layout), to_cblas_uplo(uplo),
                    to_cblas_trans(transA), (int)n, (int)K, alpha, A, (int)lda,
                    B, (int)ldb, beta, C, (int)ldc);
}

// Float TRMM
void OpenBlasBackend::trmm(Layout layout, Side side, Uplo uplo,
                           Transpose transA, Diag diag, int m, int n,
                           const float alpha, const float *A, int lda, float *B,
                           int ldb) {

  clap_cblas_strmm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), to_cblas_trans(transA),
                   to_cblas_diag(diag), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb);
}

// Double TRMM
void OpenBlasBackend::trmm(Layout layout, Side side, Uplo uplo,
                           Transpose transA, Diag diag, int m, int n,
                           const double alpha, const double *A, int lda,
                           double *B, int ldb) {

  clap_cblas_dtrmm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), to_cblas_trans(transA),
                   to_cblas_diag(diag), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb);
}

// Float TRSM
void OpenBlasBackend::trsm(Layout layout, Side side, Uplo uplo,
                           Transpose transA, Diag diag, int m, int n,
                           float alpha, const float *A, int lda, float *B,
                           int ldb) {

  clap_cblas_strsm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), to_cblas_trans(transA),
                   to_cblas_diag(diag), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb);
}

// Double TRSM
void OpenBlasBackend::trsm(Layout layout, Side side, Uplo uplo,
                           Transpose transA, Diag diag, int m, int n,
                           double alpha, const double *A, int lda, double *B,
                           int ldb) {

  clap_cblas_dtrsm(to_cblas_order(layout), to_cblas_side(side),
                   to_cblas_uplo(uplo), to_cblas_trans(transA),
                   to_cblas_diag(diag), (int)m, (int)n, alpha, A, (int)lda, B,
                   (int)ldb);
}

//SYR2K Float

void OpenBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose transA, int n,
                            int K, const float alpha, const float *A, int lda,
                            const float *B, int ldb, const float beta, float *C,
                            int ldc) {

  clap_cblas_ssyr2k(to_cblas_order(layout), to_cblas_uplo(uplo),
                    to_cblas_trans(transA), (int)n, (int)K, alpha, A, (int)lda,
                    B, (int)ldb, beta, C, (int)ldc);
}

//SYR2K Double

void OpenBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose transA, int n,
                            int K, const double alpha, const double *A, int lda,
                            const double *B, int ldb, const double beta,
                            double *C, int ldc) {
  clap_cblas_dsyr2k(to_cblas_order(layout), to_cblas_uplo(uplo),
                    to_cblas_trans(transA), (int)n, (int)K, alpha, A, (int)lda,
                    B, (int)ldb, beta, C, (int)ldc);
}



//Complex Routines

//GEMM Complex float

void OpenBlasBackend::gemm(Layout layout, Transpose trans, Transpose transB,
                           int m, int n, int K,
                           const std::complex<float> *alpha,
                           const std::complex<float> *A, int lda,
                           const std::complex<float> *B, int ldb,
                           const std::complex<float> *beta,
                           std::complex<float> *C, int ldc) {
  clap_cblas_cgemm(to_cblas_order(layout), to_cblas_trans(trans),
                   to_cblas_trans(trans), (int)m, (int)n, (int)K, alpha, A,
                   (int)lda, B, (int)ldb, beta, C, (int)ldc);
}

//GEMM Complex Double

void OpenBlasBackend::gemm(Layout layout, Transpose trans, Transpose transB,
                           int m, int n, int K,
                           const std::complex<double> *alpha,
                           const std::complex<double> *A, int lda,
                           const std::complex<double> *B, int ldb,
                           const std::complex<double> *beta,
                           std::complex<double> *C, int ldc) {
  clap_cblas_zgemm(to_cblas_order(layout), to_cblas_trans(trans),
                   to_cblas_trans(trans), (int)m, (int)n, (int)K, alpha, A,
                   (int)lda, B, (int)ldb, beta, C, (int)ldc);
}




} // namespace clap
