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

#pragma once
#include "./blas_factory.hpp"

namespace clap {

class RocBlasBackend : public IBlasBackend {

private:
  rocblas_handle handle;

public:
  RocBlasBackend();
  ~RocBlasBackend();

  // Level 1

  void copy(int64_t n, const float *x, int64_t incx, float *y,
            int64_t incy) override;
  void copy(int64_t n, const double *x, int64_t incx, double *y,
            int64_t incy) override;

  void scal(int64_t n, float alpha, float *x, int64_t incx) override;
  void scal(int64_t n, double alpha, double *x, int64_t incx) override;

  void axpy(int64_t n, float a, const float *x, int64_t incx, float *y,
            int64_t incy) override;
  void axpy(int64_t n, double a, const double *x, int64_t incx, double *y,
            int64_t incy) override;

  void swap(int64_t n, float *x, int64_t incx, float *y, int64_t incy) override;
  void swap(int64_t n, double *x, int64_t incx, double *y,
            int64_t incy) override;

  float dot(int64_t n, const float *x, int64_t incx, const float *y,
            int64_t incy) override;
  double dot(int64_t n, const double *x, int64_t incx, const double *y,
             int64_t incy) override;

  // Level 2

  void gemv(Layout layout, Transpose trans, int64_t m, int64_t n,
            const float alpha, const float *A, int64_t lda, const float *x,
            int64_t incx, const float beta, float *y, int64_t incy) override;
  void gemv(Layout layout, Transpose trans, int64_t m, int64_t n,
            const double alpha, const double *A, int64_t lda, const double *x,
            int64_t incx, const double beta, double *y, int64_t incy) override;

  void symv(Layout layout, Uplo uplo, int64_t n, const float alpha,
            const float *A, int64_t lda, const float *X, int64_t incx,
            const float beta, float *Y, int64_t incy) override;
  void symv(Layout layout, Uplo uplo, int64_t n, const double alpha,
            const double *A, int64_t lda, const double *X, int64_t incx,
            const double beta, double *Y, int64_t incy) override;

  void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int64_t n,
            const float *A, int64_t lda, float *X, int64_t incx) override;
  void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int64_t n,
            const double *A, int64_t lda, double *X, int64_t incx) override;

  void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int64_t n,
            const float *A, int64_t lda, float *X, int64_t incx) override;
  void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int64_t n,
            const double *A, int64_t lda, double *X, int64_t incx) override;

  void sbmv(Layout layout, Uplo uplo, int64_t n, int64_t k, const float alpha,
            const float *A, int64_t lda, const float *X, int64_t incx,
            const float beta, float *Y, int64_t incy) override;
  void sbmv(Layout layout, Uplo uplo, int64_t n, int64_t k, const double alpha,
            const double *A, int64_t lda, const double *X, int64_t incx,
            const double beta, double *Y, int64_t incy) override;

   void ger(Layout layout, int m, int n, const float alpha, const float *X,
           int incx, const float *Y, int incy, float *A, int lda) override;

  // Level 3

  void gemm(Layout layout, Transpose transA, Transpose transB, int64_t m,
            int64_t n, int64_t k, float alpha, const float *A, int64_t lda,
            const float *B, int64_t ldb, float beta, float *C,
            int64_t ldc) override;
  void gemm(Layout layout, Transpose transA, Transpose transB, int64_t m,
            int64_t n, int64_t k, double alpha, const double *A, int64_t lda,
            const double *B, int64_t ldb, double beta, double *C,
            int64_t ldc) override;

  void symm(Layout layout, Side side, Uplo uplo, int64_t m, int64_t n,
            float alpha, const float *A, int64_t lda, const float *B,
            int64_t ldb, float beta, float *C, int64_t ldc) override;
  void symm(Layout layout, Side side, Uplo uplo, int64_t m, int64_t n,
            double alpha, const double *A, int64_t lda, const double *B,
            int64_t ldb, double beta, double *C, int64_t ldc) override;

  void syrk(Layout layout, Uplo uplo, Transpose trans, int64_t n, int64_t k,
            float alpha, const float *A, int64_t lda, float beta, float *C,
            int64_t ldc) override;
  void syrk(Layout layout, Uplo uplo, Transpose trans, int64_t n, int64_t k,
            double alpha, const double *A, int64_t lda, double beta, double *C,
            int64_t ldc) override;

  void trmm(Layout layout, Side side, Uplo uplo, Transpose trans, Diag diag,
            int64_t m, int64_t n, double alpha, const double *A, int64_t lda,
            double *B, int64_t ldb) override;
  void trmm(Layout layout, Side side, Uplo uplo, Transpose trans, Diag diag,
            int64_t m, int64_t n, float alpha, const float *A, int64_t lda,
            float *B, int64_t ldb) override;

  void trsm(Layout layout, Side side, Uplo uplo, Transpose trans, Diag diag,
            int64_t m, int64_t n, float alpha, const float *A, int64_t lda,
            float *B, int64_t ldb) override;
  void trsm(Layout layout, Side side, Uplo uplo, Transpose trans, Diag diag,
            int64_t m, int64_t n, double alpha, const double *A, int64_t lda,
            double *B, int64_t ldb) override;
};

} // namespace clap
