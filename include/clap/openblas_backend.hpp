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

class OpenBlasBackend : public IBlasBackend {

public:
  OpenBlasBackend();

  ~OpenBlasBackend() override = default;

//Level 1

  void copy(int n, const float *x, int incx, float *y, int incy) override;
  void copy(int n, const double *x, int incx, double *y, int incy) override;
  void scal(int n, float alpha, float *x, int incx) override;
  void scal(int n, double alpha, double *x, int incx) override;
  void axpy(int n, float a, const float *x, int incx, float *y,
            int incy) override;
  void axpy(int n, double a, const double *x, int incx, double *y,
            int incy) override;
  int iamax(int n, const float *x, int incx) override;
  int iamax(int n, const double *x, int incx) override;
  void swap(int n, float *x, int incx, float *y, int incy) override;
  void swap(int n, double *x, int incx, double *y, int incy) override;
  float dot(int n, const float *x, int incx, const float *y, int incy) override;
  double dot(int n, const double *x, int incx, const double *y,
             int incy) override;
  float nrm2(int n, const float *x, int incx) override;
  double nrm2(int n, const double *x, int incx) override;
  float asum(int n, const float *x, int incx) override;
  double asum(int n, const double *x, int incx) override;
  void rot(int n, double *x, int incx, double *y, int incy, double c,
           double s) override;
  void rot(int n, float *x, int incx, float *y, int incy, float c,
           float s) override;
  void rotg(float *a, float *b, float *c, float *s) override;
  void rotg(double *a, double *b, double *c, double *s) override;
  void rotm(int n, float *x, int incx, float *y, int incy,
            const float param[5]) override;
  void rotm(int n, double *x, int incx, double *y, int incy,
            const double param[5]) override;
  void rotmg(float *d1, float *d2, float *x1, float y1,
             float param[5]) override;
  void rotmg(double *d1, double *d2, double *x1, double y1,
             double param[5]) override;

  void dotu_sub(int n, const std::complex<float> *X, int incX,
                const std::complex<float> *Y, int incY,
                std::complex<float> *dotu) override;
  void dotc_sub(int n, const std::complex<float> *X, int incX,
                const std::complex<float> *Y, int incY,
                std::complex<float> *dotc) override;
  void dotu_sub(int n, const std::complex<double> *X, int incX,
                const std::complex<double> *Y, int incY,
                std::complex<double> *dotu) override;
  void dotc_sub(int n, const std::complex<double> *X, int incX,
                const std::complex<double> *Y, int incY,
                std::complex<double> *dotc) override;
  void swap(int n, std::complex<float> *X, int incX, std::complex<float> *Y,
            int incY) override;
  void swap(int n, std::complex<double> *X, int incX, std::complex<double> *Y,
            int incY) override;
  void copy(int n, const std::complex<float> *X, int incX,
            std::complex<float> *Y, int incY) override;
  void copy(int n, const std::complex<double> *X, int incX,
            std::complex<double> *Y, int incY) override;
  void scal(int n, const std::complex<float> *alpha, std::complex<float> *X,
            int incX) override;
  void scal(int n, const std::complex<double> *alpha, std::complex<double> *X,
            int incX) override;
  void scal(int n, const float alpha, std::complex<float> *X,
            int incX) override;
  void scal(int n, const double alpha, std::complex<double> *X,
            int incX) override;
  void axpy(int n, const std::complex<float> *alpha,
            const std::complex<float> *X, int incX, std::complex<float> *Y,
            int incY) override;
  void axpy(int n, const std::complex<double> *alpha,
            const std::complex<double> *X, int incX, std::complex<double> *Y,
            int incY) override;
  float nrm2(int n, const std::complex<float> *X, int incX) override;
  float asum(int n, const std::complex<float> *X, int incX) override;
  double nrm2(int n, const std::complex<double> *X, int incX) override;
  double asum(int n, const std::complex<double> *X, int incX) override;
  int iamax(int n, const std::complex<float> *X, int incX) override;
  int iamax(int n, const std::complex<double> *X, int incX) override;

//Level 2

  void gemv(Layout layout, Transpose trans, int m, int n, const float alpha,
            const float *A, int lda, const float *x, int incx, const float beta,
            float *y, int incy) override;
  void gemv(Layout layout, Transpose trans, int m, int n, const double alpha,
            const double *A, int lda, const double *x, int incx,
            const double beta, double *y, int incy) override;
  void gemv(Layout layout, Transpose trans, int m, int n,
            const std::complex<float> *alpha, const std::complex<float> *A,
            int lda, const std::complex<float> *X, int incx,
            const std::complex<float> *beta, std::complex<float> *Y,
            int incy) override;
  void gemv(Layout layout, Transpose trans, int m, int n,
            const std::complex<double> *alpha, const std::complex<double> *A,
            int lda, const std::complex<double> *X, int incx,
            const std::complex<double> *beta, std::complex<double> *Y,
            int incy) override;
  void symv(Layout layout, Uplo uplo, int n, const float alpha, const float *A,
            int lda, const float *X, int incx, const float beta, float *Y,
            int incy) override;
  void symv(Layout layout, Uplo uplo, int n, const double alpha,
            const double *A, int lda, const double *X, int incx,
            const double beta, double *Y, int incy) override;
  void hemv(Layout layout, Uplo uplo, int n, const std::complex<float> *alpha,
            const std::complex<float> *A, int lda, const std::complex<float> *X,
            int incx, const std::complex<float> *beta, std::complex<float> *Y,
            int incy) override;
  void hemv(Layout layout, Uplo uplo, int n, const std::complex<double> *alpha,
            const std::complex<double> *A, int lda,
            const std::complex<double> *X, int incx,
            const std::complex<double> *beta, std::complex<double> *Y,
            int incy) override;
  void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const float *A, int lda, float *X, int incx) override;
  void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const double *A, int lda, double *X, int incx) override;
  void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const std::complex<float> *A, int lda, std::complex<float> *X,
            int incx) override;
  void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const std::complex<double> *A, int lda, std::complex<double> *X,
            int incx) override;
  void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const float *A, int lda, float *X, int incx) override;
  void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const double *A, int lda, double *X, int incx) override;
  void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const std::complex<float> *A, int lda, std::complex<float> *X,
            int incx) override;
  void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const std::complex<double> *A, int lda, std::complex<double> *X,
            int incx) override;
  void gerc(Layout layout, int m, int n, const std::complex<float> *alpha,
            const std::complex<float> *X, int incx,
            const std::complex<float> *Y, int incy, std::complex<float> *A,
            int lda) override;
  void gerc(Layout layout, int m, int n, const std::complex<double> *alpha,
            const std::complex<double> *X, int incx,
            const std::complex<double> *Y, int incy, std::complex<double> *A,
            int lda) override;
  void ger(Layout layout, int m, int n, const float alpha, const float *X,
           int incx, const float *Y, int incy, float *A, int lda) override;
  void ger(Layout layout, int m, int n, const double alpha, const double *X,
           int incx, const double *Y, int incy, double *A, int lda) override;
  void geru(Layout layout, int m, int n, const std::complex<float> *alpha,
            const std::complex<float> *X, int incx,
            const std::complex<float> *Y, int incy, std::complex<float> *A,
            int lda) override;
  void geru(Layout layout, int m, int n, const std::complex<double> *alpha,
            const std::complex<double> *X, int incx,
            const std::complex<double> *Y, int incy, std::complex<double> *A,
            int lda) override;
  void syr(Layout layout, Uplo uplo, int n, const float alpha, const float *X,
           int incx, float *A, int lda) override;
  void syr(Layout layout, Uplo uplo, int n, const double alpha, const double *X,
           int incx, double *A, int lda) override;
  void syr2(Layout layout, Uplo uplo, int n, const float alpha, const float *X,
            int incx, const float *Y, int incy, float *A, int lda) override;
  void syr2(Layout layout, Uplo uplo, int n, const double alpha,
            const double *X, int incx, const double *Y, int incy, double *A,
            int lda) override;
  void her(Layout layout, Uplo uplo, int n, const float alpha,
           const std::complex<float> *X, int incx, std::complex<float> *A,
           int lda) override;
  void her(Layout layout, Uplo uplo, int n, const double alpha,
           const std::complex<double> *X, int incx, std::complex<double> *A,
           int lda) override;
  void her2(Layout layout, Uplo uplo, int n, const std::complex<float> *alpha,
            const std::complex<float> *X, int incx,
            const std::complex<float> *Y, int incy, std::complex<float> *A,
            int lda) override;
  void her2(Layout layout, Uplo uplo, int n, const std::complex<double> *alpha,
            const std::complex<double> *X, int incx,
            const std::complex<double> *Y, int incy, std::complex<double> *A,
            int lda) override;
  void hpr(Layout layout, Uplo uplo, int n, const float alpha,
           const std::complex<float> *X, int incx,
           std::complex<float> *AP) override;
  void hpr(Layout layout, Uplo uplo, int n, const double alpha,
           const std::complex<double> *X, int incx,
           std::complex<double> *AP) override;
  void hpr2(Layout layout, Uplo uplo, int n, const std::complex<float> *alpha,
            const std::complex<float> *X, int incx,
            const std::complex<float> *Y, int incy,
            std::complex<float> *AP) override;
  void hpr2(Layout layout, Uplo uplo, int n, const std::complex<double> *alpha,
            const std::complex<double> *X, int incx,
            const std::complex<double> *Y, int incy,
            std::complex<double> *AP) override;
  void gbmv(Layout layout, Transpose trans, int m, int n, int KL, int KU,
            const float alpha, const float *A, int lda, const float *X,
            int incx, const float beta, float *Y, int incy) override;
  void gbmv(Layout layout, Transpose trans, int m, int n, int KL, int KU,
            const double alpha, const double *A, int lda, const double *X,
            int incx, const double beta, double *Y, int incy) override;
  void gbmv(Layout layout, Transpose trans, int m, int n, int KL, int KU,
            const std::complex<float> *alpha, const std::complex<float> *A,
            int lda, const std::complex<float> *X, int incx,
            const std::complex<float> *beta, std::complex<float> *Y,
            int incy) override;
  void gbmv(Layout layout, Transpose trans, int m, int n, int KL, int KU,
            const std::complex<double> *alpha, const std::complex<double> *A,
            int lda, const std::complex<double> *X, int incx,
            const std::complex<double> *beta, std::complex<double> *Y,
            int incy) override;
  void hbmv(Layout layout, Uplo uplo, int n, int k,
            const std::complex<float> *alpha, const std::complex<float> *A,
            int lda, const std::complex<float> *X, int incx,
            const std::complex<float> *beta, std::complex<float> *Y,
            int incy) override;
  void hbmv(Layout layout, Uplo uplo, int n, int k,
            const std::complex<double> *alpha, const std::complex<double> *A,
            int lda, const std::complex<double> *X, int incx,
            const std::complex<double> *beta, std::complex<double> *Y,
            int incy) override;
  void sbmv(Layout layout, Uplo uplo, int n, int k, const float alpha,
            const float *A, int lda, const float *X, int incx, const float beta,
            float *Y, int incy) override;
  void sbmv(Layout layout, Uplo uplo, int n, int k, const double alpha,
            const double *A, int lda, const double *X, int incx,
            const double beta, double *Y, int incy) override;
  void tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n, int k,
            const float *A, int lda, float *X, int incx) override;
  void tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n, int k,
            const double *A, int lda, double *X, int incx) override;
  void tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n, int k,
            const std::complex<float> *A, int lda, std::complex<float> *X,
            int incx) override;
  void tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n, int k,
            const std::complex<double> *A, int lda, std::complex<double> *X,
            int incx) override;
  void tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n, int k,
            const float *A, int lda, float *X, int incx) override;
  void tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n, int k,
            const double *A, int lda, double *X, int incx) override;
  void tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n, int k,
            const std::complex<float> *A, int lda, std::complex<float> *X,
            int incx) override;
  void tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n, int k,
            const std::complex<double> *A, int lda, std::complex<double> *X,
            int incx) override;
  void hpmv(Layout layout, Uplo uplo, int n, const std::complex<float> *alpha,
            const std::complex<float> *Ap, const std::complex<float> *X,
            int incx, const std::complex<float> *beta, std::complex<float> *Y,
            int incy) override;
  void hpmv(Layout layout, Uplo uplo, int n, const std::complex<double> *alpha,
            const std::complex<double> *Ap, const std::complex<double> *X,
            int incx, const std::complex<double> *beta, std::complex<double> *Y,
            int incy) override;
  void spmv(Layout layout, Uplo uplo, int n, const float alpha, const float *Ap,
            const float *x, int incX, const float beta, float *y,
            int incY) override;
  void spmv(Layout layout, Uplo uplo, int n, const double alpha,
            const double *Ap, const double *x, int incX, const double beta,
            double *y, int incY) override;
  void tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const float *Ap, float *X, int incx) override;
  void tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const double *Ap, double *X, int incx) override;
  void tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const std::complex<float> *Ap, std::complex<float> *X,
            int incx) override;
  void tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const std::complex<double> *Ap, std::complex<double> *X,
            int incx) override;
  void tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const float *Ap, float *X, int incx) override;
  void tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const double *Ap, double *X, int incx) override;
  void tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const std::complex<float> *Ap, std::complex<float> *X,
            int incx) override;
  void tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
            const std::complex<double> *Ap, std::complex<double> *X,
            int incx) override;
  void spr(Layout layout, Uplo uplo, int n, const float alpha, const float *X,
           int incx, float *Ap) override;
  void spr(Layout layout, Uplo uplo, int n, const double alpha, const double *X,
           int incx, double *Ap) override;
  void spr2(Layout layout, Uplo uplo, int n, const float alpha, const float *X,
            int incx, const float *Y, int incy, float *Ap) override;
  void spr2(Layout layout, Uplo uplo, int n, const double alpha,
            const double *X, int incx, const double *Y, int incy,
            double *Ap) override;




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
            int m, int n, double alpha, const double *A, int lda, double *B,
            int ldb) override;
  void trmm(Layout layout, Side side, Uplo uplo, Transpose trans, Diag diag,
            int m, int n, float alpha, const float *A, int lda, float *B,
            int ldb) override;
  void trsm(Layout layout, Side side, Uplo uplo, Transpose trans, Diag diag,
            int m, int n, float alpha, const float *A, int lda, float *B,
            int ldb) override;
  void trsm(Layout layout, Side side, Uplo uplo, Transpose trans, Diag diag,
            int m, int n, double alpha, const double *A, int lda, double *B,
            int ldb) override;
  void gemm(Layout layout, Transpose transA, Transpose transB, int m, int n,
            int K, const std::complex<float> *alpha,
            const std::complex<float> *A, int lda, const std::complex<float> *B,
            int ldb, const std::complex<float> *beta, std::complex<float> *C,
            int ldc) override;
  void gemm(Layout layout, Transpose transA, Transpose transB, int m, int n,
            int k, const std::complex<double> *alpha,
            const std::complex<double> *A, int lda,
            const std::complex<double> *B, int ldb,
            const std::complex<double> *beta, std::complex<double> *C,
            int ldc) override;
  void symm(Layout layout, Side side, Uplo uplo, int m, int n,
            const std::complex<float> *alpha, const std::complex<float> *A,
            int lda, const std::complex<float> *B, int ldb,
            const std::complex<float> *beta, std::complex<float> *C,
            int ldc) override;
  void symm(Layout layout, Side side, Uplo uplo, int m, int n,
            const std::complex<double> *alpha, const std::complex<double> *A,
            int lda, const std::complex<double> *B, int ldb,
            const std::complex<double> *beta, std::complex<double> *C,
            int ldc) override;
  void syrk(Layout layout, Uplo uplo, Transpose trans, int n, int K,
            const std::complex<float> *alpha, const std::complex<float> *A,
            int lda, const std::complex<float> *beta, std::complex<float> *C,
            int ldc) override;
  void syrk(Layout layout, Uplo uplo, Transpose trans, int n, int k,
            const std::complex<double> *alpha, const std::complex<double> *A,
            int lda, const std::complex<double> *beta, std::complex<double> *C,
            int ldc) override;
  void syr2k(Layout layout, Uplo uplo, Transpose trans, int n, int k,
             const std::complex<float> *alpha, const std::complex<float> *A,
             int lda, const std::complex<float> *B, int ldb,
             const std::complex<float> *beta, std::complex<float> *C,
             int ldc) override;
  void syr2k(Layout layout, Uplo uplo, Transpose trans, int n, int k,
             const std::complex<double> *alpha, const std::complex<double> *A,
             int lda, const std::complex<double> *B, int ldb,
             const std::complex<double> *beta, std::complex<double> *C,
             int ldc) override;

  void trmm(Layout layout, Side side, Uplo uplo, Transpose trans, Diag diag,
            int m, int n, const std::complex<float> *alpha,
            const std::complex<float> *A, int lda, std::complex<float> *B,
            int ldb) override;
  void trmm(Layout layout, Side side, Uplo uplo, Transpose trans, Diag diag,
            int m, int n, const std::complex<double> *alpha,
            const std::complex<double> *A, int lda, std::complex<double> *B,
            int ldb) override;
  void trsm(Layout layout, Side side, Uplo uplo, Transpose trans, Diag diag,
            int m, int n, const std::complex<float> *alpha,
            const std::complex<float> *A, int lda, std::complex<float> *B,
            int ldb) override;
  void trsm(Layout layout, Side side, Uplo uplo, Transpose trans, Diag diag,
            int m, int n, const std::complex<double> *alpha,
            const std::complex<double> *A, int lda, std::complex<double> *B,
            int ldb) override;
  void hemm(Layout layout, Side side, Uplo uplo, int m, int n,
            const std::complex<float> *alpha, const std::complex<float> *A,
            int lda, const std::complex<float> *B, int ldb,
            const std::complex<float> *beta, std::complex<float> *C,
            int ldc) override;
  void hemm(Layout layout, Side side, Uplo uplo, int m, int n,
            const std::complex<double> *alpha, const std::complex<double> *A,
            int lda, const std::complex<double> *B, int ldb,
            const std::complex<double> *beta, std::complex<double> *C,
            int ldc) override;
  void herk(Layout layout, Uplo uplo, Transpose trans, int n, int k,
            float alpha, const std::complex<float> *A, int lda, float beta,
            std::complex<float> *C, int ldc) override;
  void herk(Layout layout, Uplo uplo, Transpose trans, int n, int k,
            double alpha, const std::complex<double> *A, int lda, double beta,
            std::complex<double> *C, int ldc) override;
  void her2k(Layout layout, Uplo uplo, Transpose trans, int n, int k,
             const std::complex<float> *alpha, const std::complex<float> *A,
             int lda, const std::complex<float> *B, int ldb, float beta,
             std::complex<float> *C, int ldc) override;
  void her2k(Layout layout, Uplo uplo, Transpose trans, int n, int k,
             const std::complex<double> *alpha, const std::complex<double> *A,
             int lda, const std::complex<double> *B, int ldb, double beta,
             std::complex<double> *C, int ldc) override;
};

}
