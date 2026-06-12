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

#pragma once
#include "dyn_backends.hpp"
#include <memory>

namespace clap {
enum class BackendType { CPU, CUDA, ROCM };

enum class Layout { RowMajor, ColMajor };
enum class Transpose { NoTrans, Trans, ConjTrans };
enum class Uplo { Upper, Lower };
enum class Diag { NonUnit, Unit };
enum class Side { Left, Right };

class IBlasBackend {
public:
  virtual ~IBlasBackend() = default;

  // level 1

  virtual void copy(int64_t n, const float *x, int64_t incx, float *y,
                    int64_t incy) = 0;
  virtual void copy(int64_t n, const double *x, int64_t incx, double *y,
                    int64_t incy) = 0;

  virtual void scal(int64_t n, float a, float *x, int64_t incx) = 0;
  virtual void scal(int64_t n, double a, double *x, int64_t incx) = 0;

  virtual void axpy(int64_t n, float a, const float *x, int64_t incx, float *y,
                    int64_t incy) = 0;
  virtual void axpy(int64_t n, double a, const double *x, int64_t incx,
                    double *y, int64_t incy) = 0;

  virtual void swap(int64_t n, float *x, int64_t incx, float *y,
                    int64_t incy) = 0;
  virtual void swap(int64_t n, double *x, int64_t incx, double *y,
                    int64_t incy) = 0;

  virtual float dot(int64_t n, const float *x, int64_t incx, const float *y,
                    int64_t incy) = 0;
  virtual double dot(int64_t n, const double *x, int64_t incx, const double *y,
                     int64_t incy) = 0;

  // level 2

  virtual void gemv(Layout layout, Transpose trans, int64_t m, int64_t n,
                    const float alpha, const float *A, int64_t lda,
                    const float *x, int64_t incx, const float beta, float *y,
                    int64_t incy) = 0;
  virtual void gemv(Layout layout, Transpose trans, int64_t m, int64_t n,
                    const double alpha, const double *A, int64_t lda,
                    const double *x, int64_t incx, const double beta, double *y,
                    int64_t incy) = 0;

  virtual void symv(Layout layout, Uplo uplo, int64_t n, const float alpha,
                    const float *A, int64_t lda, const float *X, int64_t incx,
                    const float beta, float *Y, int64_t incy) = 0;
  virtual void symv(Layout layout, Uplo uplo, int64_t n, const double alpha,
                    const double *A, int64_t lda, const double *X, int64_t incx,
                    const double beta, double *Y, int64_t incy) = 0;

  virtual void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                    int64_t n, const float *A, int64_t lda, float *X,
                    int64_t incx) = 0;
  virtual void trmv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                    int64_t n, const double *A, int64_t lda, double *X,
                    int64_t incx) = 0;

  virtual void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                    int64_t n, const float *A, int64_t lda, float *X,
                    int64_t incx) = 0;
  virtual void trsv(Layout layout, Uplo uplo, Transpose trans, Diag diag,
                    int64_t n, const double *A, int64_t lda, double *X,
                    int64_t incx) = 0;

  virtual void sbmv(Layout layout, Uplo uplo, int64_t n, int64_t k,
                    const float alpha, const float *A, int64_t lda,
                    const float *X, int64_t incx, const float beta, float *Y,
                    int64_t incy) = 0;
  virtual void sbmv(Layout layout, Uplo uplo, int64_t n, int64_t k,
                    const double alpha, const double *A, int64_t lda,
                    const double *X, int64_t incx, const double beta, double *Y,
                    int64_t incy) = 0;

  // level3

  virtual void gemm(Layout layout, Transpose transA, Transpose transB,
                    int64_t m, int64_t n, int64_t k, float alpha,
                    const float *A, int64_t lda, const float *B, int64_t ldb,
                    float beta, float *C, int64_t ldc) = 0;
  virtual void gemm(Layout layout, Transpose transA, Transpose transB,
                    int64_t m, int64_t n, int64_t k, double alpha,
                    const double *A, int64_t lda, const double *B, int64_t ldb,
                    double beta, double *C, int64_t ldc) = 0;

  virtual void symm(Layout layout, Side side, Uplo uplo, int64_t m, int64_t n,
                    float alpha, const float *A, int64_t lda, const float *B,
                    int64_t ldb, float beta, float *C, int64_t ldc) = 0;
  virtual void symm(Layout layout, Side side, Uplo uplo, int64_t m, int64_t n,
                    double alpha, const double *A, int64_t lda, const double *B,
                    int64_t ldb, double beta, double *C, int64_t ldc) = 0;

  virtual void syrk(Layout layout, Uplo uplo, Transpose trans, int64_t n,
                    int64_t k, float alpha, const float *A, int64_t lda,
                    float beta, float *C, int64_t ldc) = 0;
  virtual void syrk(Layout layout, Uplo uplo, Transpose trans, int64_t n,
                    int64_t k, double alpha, const double *A, int64_t lda,
                    double beta, double *C, int64_t ldc) = 0;

  virtual void trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int64_t m, int64_t n, double alpha,
                    const double *A, int64_t lda, double *B, int64_t ldb) = 0;
  virtual void trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int64_t m, int64_t n, float alpha,
                    const float *A, int64_t lda, float *B, int64_t ldb) = 0;

  virtual void trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int64_t m, int64_t n, float alpha,
                    const float *A, int64_t lda, float *B, int64_t ldb) = 0;
  virtual void trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                    Diag diag, int64_t m, int64_t n, double alpha,
                    const double *A, int64_t lda, double *B, int64_t ldb) = 0;
};

class BlasFactory {
public:
  static std::unique_ptr<IBlasBackend>
  create(BackendType default_backend = BackendType::CPU);
};

} // namespace clap