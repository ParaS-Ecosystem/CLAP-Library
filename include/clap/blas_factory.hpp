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

#include "dyn_backends.hpp"
#include <memory>

namespace clap {

enum class BackendType { CPU, CUDA, AMD };

enum class Layout { RowMajor, ColMajor };
enum class Transpose { NoTrans, Trans, ConjTrans };
enum class Uplo { Upper, Lower };

enum class Diag { NonUnit, Unit };
enum class Side { Left, Right };
class IBlasBackend {
public:
  virtual ~IBlasBackend() = default;

  //Level 1

  virtual void copy(int n, const float *x, int incx, float *y, int incy) = 0;
  virtual void copy(int n, const double *x, int incx, double *y, int incy) = 0;
  virtual void scal(int n, float a, float *x, int incx) = 0;
  virtual void scal(int n, double a, double *x, int incx) = 0;
  virtual void axpy(int n, float a, const float *x, int incx, float *y,
                    int incy) = 0;
  virtual void axpy(int n, double a, const double *x, int incx, double *y,
                    int incy) = 0;
  virtual int iamax(int n, const float *x, int incx) = 0;
  virtual int iamax(int n, const double *x, int incx) = 0;
  virtual void swap(int n, float *x, int incx, float *y, int incy) = 0;
  virtual void swap(int n, double *x, int incx, double *y, int incy) = 0;
  virtual float dot(int n, const float *x, int incx, const float *y,
                    int incy) = 0;
  virtual double dot(int n, const double *x, int incx, const double *y,
                     int incy) = 0;
  virtual float nrm2(int n, const float *x, int incx) = 0;
  virtual double nrm2(int n, const double *x, int incx) = 0;
  virtual float asum(int n, const float *x, int incx) = 0;
  virtual double asum(int n, const double *x, int incx) = 0;
  virtual void rot(int n, double *x, int incx, double *y, int incy, double c,
                   double s) = 0;
  virtual void rot(int n, float *x, int incx, float *y, int incy, float c,
                   float s) = 0;
  virtual void rotg(float *a, float *b, float *c, float *s) = 0;
  virtual void rotg(double *a, double *b, double *c, double *s) = 0;
  virtual void rotm(int n, float *x, int incx, float *y, int incy,
                    const float param[5]) = 0;
  virtual void rotm(int n, double *x, int incx, double *y, int incy,
                    const double param[5]) = 0;
  virtual void rotmg(float *d1, float *d2, float *x1, float y1,
                     float param[5]) = 0;
  virtual void rotmg(double *d1, double *d2, double *x1, double y1,
                     double param[5]) = 0;

  virtual void dotu_sub(int n, const std::complex<float> *X, int incX,
                        const std::complex<float> *Y, int incY,
                        std::complex<float> *dotu) = 0;
  virtual void dotc_sub(int n, const std::complex<float> *X, int incX,
                        const std::complex<float> *Y, int incY,
                        std::complex<float> *dotc) = 0;
  virtual void dotu_sub(int n, const std::complex<double> *X, int incX,
                        const std::complex<double> *Y, int incY,
                        std::complex<double> *dotu) = 0;
  virtual void dotc_sub(int n, const std::complex<double> *X, int incX,
                        const std::complex<double> *Y, int incY,
                        std::complex<double> *dotc) = 0;
  virtual void swap(int n, std::complex<float> *X, int incX,
                    std::complex<float> *Y, int incY) = 0;
  virtual void swap(int n, std::complex<double> *X, int incX,
                    std::complex<double> *Y, int incY) = 0;
  virtual void copy(int n, const std::complex<float> *X, int incX,
                    std::complex<float> *Y, int incY) = 0;
  virtual void copy(int n, const std::complex<double> *X, int incX,
                    std::complex<double> *Y, int incY) = 0;
  virtual void scal(int n, const std::complex<float> *alpha,
                    std::complex<float> *X, int incX) = 0;
  virtual void scal(int n, const std::complex<double> *alpha,
                    std::complex<double> *X, int incX) = 0;
  virtual void scal(int n, const float alpha, std::complex<float> *X,
                    int incX) = 0;
  virtual void scal(int n, const double alpha, std::complex<double> *X,
                    int incX) = 0;
  virtual void axpy(int n, const std::complex<float> *alpha,
                    const std::complex<float> *X, int incX,
                    std::complex<float> *Y, int incY) = 0;
  virtual void axpy(int n, const std::complex<double> *alpha,
                    const std::complex<double> *X, int incX,
                    std::complex<double> *Y, int incY) = 0;
  virtual float nrm2(int n, const std::complex<float> *X, int incX) = 0;
  virtual float asum(int n, const std::complex<float> *X, int incX) = 0;
  virtual double nrm2(int n, const std::complex<double> *X, int incX) = 0;
  virtual double asum(int n, const std::complex<double> *X, int incX) = 0;
  virtual int iamax(int n, const std::complex<float> *X, int incX) = 0;
  virtual int iamax(int n, const std::complex<double> *X, int incX) = 0;

  //Level 2

  virtual void gemv(Layout layout, Transpose trans, int m, int n,
                    const float alpha, const float *A, int lda, const float *x,
                    int incx, const float beta, float *y, int incy) = 0;
  virtual void gemv(Layout layout, Transpose trans, int m, int n,
                    const double alpha, const double *A, int lda,
                    const double *x, int incx, const double beta, double *y,
                    int incy) = 0;
  virtual void gemv(Layout layout, Transpose trans, int m, int n,
                    const std::complex<float> *alpha,
                    const std::complex<float> *A, int lda,
                    const std::complex<float> *X, int incx,
                    const std::complex<float> *beta, std::complex<float> *Y,
                    int incy) = 0;
  virtual void gemv(Layout layout, Transpose trans, int m, int n,
                    const std::complex<double> *alpha,
                    const std::complex<double> *A, int lda,
                    const std::complex<double> *X, int incx,
                    const std::complex<double> *beta, std::complex<double> *Y,
                    int incy) = 0;
  virtual void symv(Layout layout, Uplo uplo, int n, const float alpha,
                    const float *A, int lda, const float *X, int incx,
                    const float beta, float *Y, int incy) = 0;
  virtual void symv(Layout layout, Uplo uplo, int n, const double alpha,
                    const double *A, int lda, const double *X, int incx,
                    const double beta, double *Y, int incy) = 0;
  virtual void hemv(Layout layout, Uplo uplo, int n,
                    const std::complex<float> *alpha,
                    const std::complex<float> *A, int lda,
                    const std::complex<float> *X, int incx,
                    const std::complex<float> *beta, std::complex<float> *Y,
                    int incy) = 0;
  virtual void hemv(Layout layout, Uplo uplo, int n,
                    const std::complex<double> *alpha,
                    const std::complex<double> *A, int lda,
                    const std::complex<double> *X, int incx,
                    const std::complex<double> *beta, std::complex<double> *Y,
                    int incy) = 0;
  virtual void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const float *A, int lda, float *X, int incx) = 0;
  virtual void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const double *A, int lda, double *X, int incx) = 0;
  virtual void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const std::complex<float> *A, int lda,
                    std::complex<float> *X, int incx) = 0;
  virtual void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const std::complex<double> *A, int lda,
                    std::complex<double> *X, int incx) = 0;
  virtual void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const float *A, int lda, float *X, int incx) = 0;
  virtual void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const double *A, int lda, double *X, int incx) = 0;
  virtual void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const std::complex<float> *A, int lda,
                    std::complex<float> *X, int incx) = 0;
  virtual void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const std::complex<double> *A, int lda,
                    std::complex<double> *X, int incx) = 0;
  virtual void gerc(Layout layout, int m, int n,
                    const std::complex<float> *alpha,
                    const std::complex<float> *X, int incx,
                    const std::complex<float> *Y, int incy,
                    std::complex<float> *A, int lda) = 0;
  virtual void gerc(Layout layout, int m, int n,
                    const std::complex<double> *alpha,
                    const std::complex<double> *X, int incx,
                    const std::complex<double> *Y, int incy,
                    std::complex<double> *A, int lda) = 0;
  virtual void ger(Layout layout, int m, int n, const float alpha,
                   const float *X, int incx, const float *Y, int incy, float *A,
                   int lda) = 0;
  virtual void ger(Layout layout, int m, int n, const double alpha,
                   const double *X, int incx, const double *Y, int incy,
                   double *A, int lda) = 0;
  virtual void geru(Layout layout, int m, int n,
                    const std::complex<float> *alpha,
                    const std::complex<float> *X, int incx,
                    const std::complex<float> *Y, int incy,
                    std::complex<float> *A, int lda) = 0;
  virtual void geru(Layout layout, int m, int n,
                    const std::complex<double> *alpha,
                    const std::complex<double> *X, int incx,
                    const std::complex<double> *Y, int incy,
                    std::complex<double> *A, int lda) = 0;
  virtual void syr(Layout layout, Uplo uplo, int n, const float alpha,
                   const float *X, int incx, float *A, int lda) = 0;
  virtual void syr(Layout layout, Uplo uplo, int n, const double alpha,
                   const double *X, int incx, double *A, int lda) = 0;
  virtual void syr2(Layout layout, Uplo uplo, int n, const float alpha,
                    const float *X, int incx, const float *Y, int incy,
                    float *A, int lda) = 0;
  virtual void syr2(Layout layout, Uplo uplo, int n, const double alpha,
                    const double *X, int incx, const double *Y, int incy,
                    double *A, int lda) = 0;
  virtual void her(Layout layout, Uplo uplo, int n, const float alpha,
                   const std::complex<float> *X, int incx,
                   std::complex<float> *A, int lda) = 0;
  virtual void her(Layout layout, Uplo uplo, int n, const double alpha,
                   const std::complex<double> *X, int incx,
                   std::complex<double> *A, int lda) = 0;
  virtual void her2(Layout layout, Uplo uplo, int n,
                    const std::complex<float> *alpha,
                    const std::complex<float> *X, int incx,
                    const std::complex<float> *Y, int incy,
                    std::complex<float> *A, int lda) = 0;
  virtual void her2(Layout layout, Uplo uplo, int n,
                    const std::complex<double> *alpha,
                    const std::complex<double> *X, int incx,
                    const std::complex<double> *Y, int incy,
                    std::complex<double> *A, int lda) = 0;
  virtual void hpr(Layout layout, Uplo uplo, int n, const float alpha,
                   const std::complex<float> *X, int incx,
                   std::complex<float> *AP) = 0;
  virtual void hpr(Layout layout, Uplo uplo, int n, const double alpha,
                   const std::complex<double> *X, int incx,
                   std::complex<double> *AP) = 0;
  virtual void hpr2(Layout layout, Uplo uplo, int n,
                    const std::complex<float> *alpha,
                    const std::complex<float> *X, int incx,
                    const std::complex<float> *Y, int incy,
                    std::complex<float> *AP) = 0;
  virtual void hpr2(Layout layout, Uplo uplo, int n,
                    const std::complex<double> *alpha,
                    const std::complex<double> *X, int incx,
                    const std::complex<double> *Y, int incy,
                    std::complex<double> *AP) = 0;
  virtual void gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                    int KU, const float alpha, const float *A, int lda,
                    const float *X, int incx, const float beta, float *Y,
                    int incy) = 0;
  virtual void gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                    int KU, const double alpha, const double *A, int lda,
                    const double *X, int incx, const double beta, double *Y,
                    int incy) = 0;
  virtual void gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                    int KU, const std::complex<float> *alpha,
                    const std::complex<float> *A, int lda,
                    const std::complex<float> *X, int incx,
                    const std::complex<float> *beta, std::complex<float> *Y,
                    int incy) = 0;
  virtual void gbmv(Layout layout, Transpose trans, int m, int n, int KL,
                    int KU, const std::complex<double> *alpha,
                    const std::complex<double> *A, int lda,
                    const std::complex<double> *X, int incx,
                    const std::complex<double> *beta, std::complex<double> *Y,
                    int incy) = 0;
  virtual void hbmv(Layout layout, Uplo uplo, int n, int k,
                    const std::complex<float> *alpha,
                    const std::complex<float> *A, int lda,
                    const std::complex<float> *X, int incx,
                    const std::complex<float> *beta, std::complex<float> *Y,
                    int incy) = 0;
  virtual void hbmv(Layout layout, Uplo uplo, int n, int k,
                    const std::complex<double> *alpha,
                    const std::complex<double> *A, int lda,
                    const std::complex<double> *X, int incx,
                    const std::complex<double> *beta, std::complex<double> *Y,
                    int incy) = 0;
  virtual void sbmv(Layout layout, Uplo uplo, int n, int k, const float alpha,
                    const float *A, int lda, const float *X, int incx,
                    const float beta, float *Y, int incy) = 0;
  virtual void sbmv(Layout layout, Uplo uplo, int n, int k, const double alpha,
                    const double *A, int lda, const double *X, int incx,
                    const double beta, double *Y, int incy) = 0;
  virtual void tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    int k, const float *A, int lda, float *X, int incx) = 0;
  virtual void tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    int k, const double *A, int lda, double *X, int incx) = 0;
  virtual void tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    int k, const std::complex<float> *A, int lda,
                    std::complex<float> *X, int incx) = 0;
  virtual void tbmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    int k, const std::complex<double> *A, int lda,
                    std::complex<double> *X, int incx) = 0;
  virtual void tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    int k, const float *A, int lda, float *X, int incx) = 0;
  virtual void tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    int k, const double *A, int lda, double *X, int incx) = 0;
  virtual void tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    int k, const std::complex<float> *A, int lda,
                    std::complex<float> *X, int incx) = 0;
  virtual void tbsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    int k, const std::complex<double> *A, int lda,
                    std::complex<double> *X, int incx) = 0;
  virtual void hpmv(Layout layout, Uplo uplo, int n,
                    const std::complex<float> *alpha,
                    const std::complex<float> *Ap, const std::complex<float> *X,
                    int incx, const std::complex<float> *beta,
                    std::complex<float> *Y, int incy) = 0;
  virtual void hpmv(Layout layout, Uplo uplo, int n, const std::complex<double> *alpha,
       const std::complex<double> *Ap, const std::complex<double> *X, int incx,
       const std::complex<double> *beta, std::complex<double> *Y, int incy) = 0;
 
  virtual void spmv(Layout layout, Uplo uplo, int n, const float alpha,
                    const float *Ap, const float *x, int incX, const float beta,
                    float *y, int incY) = 0;
  virtual void spmv(Layout layout, Uplo uplo, int n, const double alpha,
                    const double *Ap, const double *x, int incX,
                    const double beta, double *y, int incY) = 0;
  virtual void tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const float *Ap, float *X, int incx) = 0;
  virtual void tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const double *Ap, double *X, int incx) = 0;
  virtual void tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const std::complex<float> *Ap, std::complex<float> *X,
                    int incx) = 0;
  virtual void tpmv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const std::complex<double> *Ap, std::complex<double> *X,
                    int incx) = 0;
  virtual void tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const float *Ap, float *X, int incx) = 0;
  virtual void tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const double *Ap, double *X, int incx) = 0;
  virtual void tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const std::complex<float> *Ap, std::complex<float> *X,
                    int incx) = 0;
  virtual void tpsv(Layout layout, Uplo uplo, Transpose trans, Diag diag, int n,
                    const std::complex<double> *Ap, std::complex<double> *X,
                    int incx) = 0;
  virtual void spr(Layout layout, Uplo uplo, int n, const float alpha,
                   const float *X, int incx, float *Ap) = 0;
  virtual void spr(Layout layout, Uplo uplo, int n, const double alpha,
                   const double *X, int incx, double *Ap) = 0;
  virtual void spr2(Layout layout, Uplo uplo, int n, const float alpha,
                    const float *X, int incx, const float *Y, int incy,
                    float *Ap) = 0;
  virtual void spr2(Layout layout, Uplo uplo, int n, const double alpha,
                    const double *X, int incx, const double *Y, int incy,
                    double *Ap) = 0;

// Level 3

  virtual void gemm(Layout layout, Transpose transA, Transpose transB, int m,
                    int n, int k, float alpha, const float *A, int lda,
                    const float *B, int ldb, float beta, float *C, int ldc) = 0;
  virtual void gemm(Layout layout, Transpose transA, Transpose transB, int m,
                    int n, int k, double alpha, const double *A, int lda,
                    const double *B, int ldb, double beta, double *C,
                    int ldc) = 0;
  virtual void symm(Layout layout, Side side, Uplo uplo, int m, int n,
                    float alpha, const float *A, int lda, const float *B,
                    int ldb, float beta, float *C, int ldc) = 0;
  virtual void symm(Layout layout, Side side, Uplo uplo, int m, int n,
                    double alpha, const double *A, int lda, const double *B,
                    int ldb, double beta, double *C, int ldc) = 0;
  virtual void syrk(Layout layout, Uplo uplo, Transpose trans, int n, int k,
                    float alpha, const float *A, int lda, float beta, float *C,
                    int ldc) = 0;
  virtual void syrk(Layout layout, Uplo uplo, Transpose trans, int n, int k,
                    double alpha, const double *A, int lda, double beta,
                    double *C, int ldc) = 0;
  virtual void syr2k(Layout layout, Uplo uplo, Transpose trans, int n, int k,
                     float alpha, const float *A, int lda, const float *B,
                     int ldb, float beta, float *C, int ldc) = 0;
  virtual void syr2k(Layout layout, Uplo uplo, Transpose trans, int n, int k,
                     double alpha, const double *A, int lda, const double *B,
                     int ldb, double beta, double *C, int ldc) = 0;
  virtual void trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int m, int n, double alpha, const double *A,
                    int lda, double *B, int ldb) = 0;
  virtual void trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int m, int n, float alpha, const float *A,
                    int lda, float *B, int ldb) = 0;
  virtual void trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int m, int n, float alpha, const float *A,
                    int lda, float *B, int ldb) = 0;
  virtual void trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int m, int n, double alpha, const double *A,
                    int lda, double *B, int ldb) = 0;
  virtual void gemm(Layout layout, Transpose transA, Transpose transB, int m,
                    int n, int K, const std::complex<float> *alpha,
                    const std::complex<float> *A, int lda,
                    const std::complex<float> *B, int ldb,
                    const std::complex<float> *beta, std::complex<float> *C,
                    int ldc) = 0;
  virtual void gemm(Layout layout, Transpose transA, Transpose transB, int m,
                    int n, int k, const std::complex<double> *alpha,
                    const std::complex<double> *A, int lda,
                    const std::complex<double> *B, int ldb,
                    const std::complex<double> *beta, std::complex<double> *C,
                    int ldc) = 0;
  virtual void symm(Layout layout, Side side, Uplo uplo, int m, int n,
                    const std::complex<float> *alpha,
                    const std::complex<float> *A, int lda,
                    const std::complex<float> *B, int ldb,
                    const std::complex<float> *beta, std::complex<float> *C,
                    int ldc) = 0;
  virtual void symm(Layout layout, Side side, Uplo uplo, int m, int n,
                    const std::complex<double> *alpha,
                    const std::complex<double> *A, int lda,
                    const std::complex<double> *B, int ldb,
                    const std::complex<double> *beta, std::complex<double> *C,
                    int ldc) = 0;
  virtual void syrk(Layout layout, Uplo uplo, Transpose trans, int n, int K,
                    const std::complex<float> *alpha,
                    const std::complex<float> *A, int lda,
                    const std::complex<float> *beta, std::complex<float> *C,
                    int ldc) = 0;
  virtual void syrk(Layout layout, Uplo uplo, Transpose trans, int n, int k,
                    const std::complex<double> *alpha,
                    const std::complex<double> *A, int lda,
                    const std::complex<double> *beta, std::complex<double> *C,
                    int ldc) = 0;
  virtual void syr2k(Layout layout, Uplo uplo, Transpose trans, int n, int k,
                     const std::complex<float> *alpha,
                     const std::complex<float> *A, int lda,
                     const std::complex<float> *B, int ldb,
                     const std::complex<float> *beta, std::complex<float> *C,
                     int ldc) = 0;
  virtual void syr2k(Layout layout, Uplo uplo, Transpose trans, int n, int k,
                     const std::complex<double> *alpha,
                     const std::complex<double> *A, int lda,
                     const std::complex<double> *B, int ldb,
                     const std::complex<double> *beta, std::complex<double> *C,
                     int ldc) = 0;
  virtual void trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int m, int n, const std::complex<float> *alpha,
                    const std::complex<float> *A, int lda,
                    std::complex<float> *B, int ldb) = 0;
  virtual void trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int m, int n, const std::complex<double> *alpha,
                    const std::complex<double> *A, int lda,
                    std::complex<double> *B, int ldb) = 0;
  virtual void trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int m, int n, const std::complex<float> *alpha,
                    const std::complex<float> *A, int lda,
                    std::complex<float> *B, int ldb) = 0;
  virtual void trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int m, int n, const std::complex<double> *alpha,
                    const std::complex<double> *A, int lda,
                    std::complex<double> *B, int ldb) = 0;
  virtual void hemm(Layout layout, Side side, Uplo uplo, int m, int n,
                    const std::complex<float> *alpha,
                    const std::complex<float> *A, int lda,
                    const std::complex<float> *B, int ldb,
                    const std::complex<float> *beta, std::complex<float> *C,
                    int ldc) = 0;
  virtual void hemm(Layout layout, Side side, Uplo uplo, int m, int n,
                    const std::complex<double> *alpha,
                    const std::complex<double> *A, int lda,
                    const std::complex<double> *B, int ldb,
                    const std::complex<double> *beta, std::complex<double> *C,
                    int ldc) = 0;
  virtual void herk(Layout layout, Uplo uplo, Transpose trans, int n, int k,
                    float alpha, const std::complex<float> *A, int lda,
                    float beta, std::complex<float> *C, int ldc) = 0;
  virtual void herk(Layout layout, Uplo uplo, Transpose trans, int n, int k,
                    double alpha, const std::complex<double> *A, int lda,
                    double beta, std::complex<double> *C, int ldc) = 0;
  virtual void her2k(Layout layout, Uplo uplo, Transpose trans, int n, int k,
                     const std::complex<float> *alpha,
                     const std::complex<float> *A, int lda,
                     const std::complex<float> *B, int ldb, float beta,
                     std::complex<float> *C, int ldc) = 0;
  virtual void her2k(Layout layout, Uplo uplo, Transpose trans, int n, int k,
                     const std::complex<double> *alpha,
                     const std::complex<double> *A, int lda,
                     const std::complex<double> *B, int ldb, double beta,
                     std::complex<double> *C, int ldc) = 0;
};

class BlasFactory {
public:

  static std::unique_ptr<IBlasBackend>
  create(BackendType default_backend = BackendType::CPU);
};

}
