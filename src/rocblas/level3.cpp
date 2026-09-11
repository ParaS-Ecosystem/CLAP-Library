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

#include "clap/rocblas_backend.hpp"
#include "clap/rocblas_utils.hpp"

namespace clap {

// Float GEMM
void RocBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                          int m, int n, int k, float alpha, const float *A,
                          int lda, const float *B, int ldb, float beta,
                          float *C, int ldc) {
  size_t sizeA = lda * ((transA == Transpose::NoTrans) ? k : m);
  size_t sizeB = ldb * ((transB == Transpose::NoTrans) ? n : k);
  size_t sizeC = ldc * n;

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(float));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblas_sgemm(handle, to_rocblas_trans(transA), to_rocblas_trans(transB),
                     (rocblas_int)m, (rocblas_int)n, (rocblas_int)k, &alpha,
                     d_A, (rocblas_int)lda, d_B, (rocblas_int)ldb, &beta, d_C,
                     (rocblas_int)ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Double GEMM
void RocBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                          int m, int n, int k, double alpha, const double *A,
                          int lda, const double *B, int ldb, double beta,
                          double *C, int ldc) {
  size_t sizeA = lda * ((transA == Transpose::NoTrans) ? k : m);
  size_t sizeB = ldb * ((transB == Transpose::NoTrans) ? n : k);
  size_t sizeC = ldc * n;

  double *d_A = nullptr;
  double *d_B = nullptr;
  double *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(double));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblas_dgemm(handle, to_rocblas_trans(transA), to_rocblas_trans(transB),
                     (rocblas_int)m, (rocblas_int)n, (rocblas_int)k, &alpha,
                     d_A, (rocblas_int)lda, d_B, (rocblas_int)ldb, &beta, d_C,
                     (rocblas_int)ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Complex Float GEMM
void RocBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                          int m, int n, int k, const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          const std::complex<float> *B, int ldb,
                          const std::complex<float> *beta,
                          std::complex<float> *C, int ldc) {
  size_t sizeA = lda * ((transA == Transpose::NoTrans) ? k : m);
  size_t sizeB = ldb * ((transB == Transpose::NoTrans) ? n : k);
  size_t sizeC = ldc * n;

  std::complex<float> *d_A = nullptr;
  std::complex<float> *d_B = nullptr;
  std::complex<float> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<float>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);

  clap_rocblas_cgemm(
      handle, to_rocblas_trans(transA), to_rocblas_trans(transB),
      (rocblas_int)m, (rocblas_int)n, (rocblas_int)k,
      reinterpret_cast<const rocblas_float_complex *>(alpha),
      reinterpret_cast<const rocblas_float_complex *>(d_A), (rocblas_int)lda,
      reinterpret_cast<const rocblas_float_complex *>(d_B), (rocblas_int)ldb,
      reinterpret_cast<const rocblas_float_complex *>(beta),
      reinterpret_cast<rocblas_float_complex *>(d_C), (rocblas_int)ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Complex Double GEMM
void RocBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                          int m, int n, int k,
                          const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          const std::complex<double> *B, int ldb,
                          const std::complex<double> *beta,
                          std::complex<double> *C, int ldc) {
  size_t sizeA = lda * ((transA == Transpose::NoTrans) ? k : m);
  size_t sizeB = ldb * ((transB == Transpose::NoTrans) ? n : k);
  size_t sizeC = ldc * n;

  std::complex<double> *d_A = nullptr;
  std::complex<double> *d_B = nullptr;
  std::complex<double> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<double>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);

  clap_rocblas_zgemm(
      handle, to_rocblas_trans(transA), to_rocblas_trans(transB),
      (rocblas_int)m, (rocblas_int)n, (rocblas_int)k,
      reinterpret_cast<const rocblas_double_complex *>(alpha),
      reinterpret_cast<const rocblas_double_complex *>(d_A), (rocblas_int)lda,
      reinterpret_cast<const rocblas_double_complex *>(d_B), (rocblas_int)ldb,
      reinterpret_cast<const rocblas_double_complex *>(beta),
      reinterpret_cast<rocblas_double_complex *>(d_C), (rocblas_int)ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Float SYMM
void RocBlasBackend::symm(Layout layout, Side side, Uplo uplo, int m, int n,
                          float alpha, const float *A, int lda, const float *B,
                          int ldb, float beta, float *C, int ldc) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(float));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblas_ssymm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo), m, n,
                     &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Double SYMM
void RocBlasBackend::symm(Layout layout, Side side, Uplo uplo, int m, int n,
                          double alpha, const double *A, int lda,
                          const double *B, int ldb, double beta, double *C,
                          int ldc) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;

  double *d_A = nullptr;
  double *d_B = nullptr;
  double *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(double));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblas_dsymm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo), m, n,
                     &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Complex Float SYMM
void RocBlasBackend::symm(Layout layout, Side side, Uplo uplo, int m, int n,
                          const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          const std::complex<float> *B, int ldb,
                          const std::complex<float> *beta,
                          std::complex<float> *C, int ldc) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;

  std::complex<float> *d_A = nullptr;
  std::complex<float> *d_B = nullptr;
  std::complex<float> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<float>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);

  clap_rocblas_csymm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo), m, n,
                     reinterpret_cast<const rocblas_float_complex *>(alpha),
                     reinterpret_cast<const rocblas_float_complex *>(d_A), lda,
                     reinterpret_cast<const rocblas_float_complex *>(d_B), ldb,
                     reinterpret_cast<const rocblas_float_complex *>(beta),
                     reinterpret_cast<rocblas_float_complex *>(d_C), ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Complex Double SYMM
void RocBlasBackend::symm(Layout layout, Side side, Uplo uplo, int m, int n,
                          const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          const std::complex<double> *B, int ldb,
                          const std::complex<double> *beta,
                          std::complex<double> *C, int ldc) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;

  std::complex<double> *d_A = nullptr;
  std::complex<double> *d_B = nullptr;
  std::complex<double> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<double>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);

  clap_rocblas_zsymm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo), m, n,
                     reinterpret_cast<const rocblas_double_complex *>(alpha),
                     reinterpret_cast<const rocblas_double_complex *>(d_A), lda,
                     reinterpret_cast<const rocblas_double_complex *>(d_B), ldb,
                     reinterpret_cast<const rocblas_double_complex *>(beta),
                     reinterpret_cast<rocblas_double_complex *>(d_C), ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Float SYRK
void RocBlasBackend::syrk(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, float alpha, const float *A, int lda,
                          float beta, float *C, int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeC = ldc * n;

  float *d_A = nullptr;
  float *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblas_ssyrk(handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n,
                     k, &alpha, d_A, lda, &beta, d_C, ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_C);
}

// Double SYRK
void RocBlasBackend::syrk(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, double alpha, const double *A, int lda,
                          double beta, double *C, int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeC = ldc * n;

  double *d_A = nullptr;
  double *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblas_dsyrk(handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n,
                     k, &alpha, d_A, lda, &beta, d_C, ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_C);
}

// Complex Float SYRK
void RocBlasBackend::syrk(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          const std::complex<float> *beta,
                          std::complex<float> *C, int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeC = ldc * n;

  std::complex<float> *d_A = nullptr;
  std::complex<float> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<float>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);

  clap_rocblas_csyrk(handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n,
                     k, reinterpret_cast<const rocblas_float_complex *>(alpha),
                     reinterpret_cast<const rocblas_float_complex *>(d_A), lda,
                     reinterpret_cast<const rocblas_float_complex *>(beta),
                     reinterpret_cast<rocblas_float_complex *>(d_C), ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_C);
}

// Complex Double SYRK
void RocBlasBackend::syrk(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          const std::complex<double> *beta,
                          std::complex<double> *C, int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeC = ldc * n;

  std::complex<double> *d_A = nullptr;
  std::complex<double> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<double>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);

  clap_rocblas_zsyrk(handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n,
                     k, reinterpret_cast<const rocblas_double_complex *>(alpha),
                     reinterpret_cast<const rocblas_double_complex *>(d_A), lda,
                     reinterpret_cast<const rocblas_double_complex *>(beta),
                     reinterpret_cast<rocblas_double_complex *>(d_C), ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_C);
}

// Float SYR2K
void RocBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose trans, int n,
                           int k, float alpha, const float *A, int lda,
                           const float *B, int ldb, float beta, float *C,
                           int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeB = (trans == Transpose::NoTrans) ? ldb * k : ldb * n;
  size_t sizeC = ldc * n;

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(float));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblas_ssyr2k(handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n,
                      k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Double SYR2K
void RocBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose trans, int n,
                           int k, double alpha, const double *A, int lda,
                           const double *B, int ldb, double beta, double *C,
                           int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeB = (trans == Transpose::NoTrans) ? ldb * k : ldb * n;
  size_t sizeC = ldc * n;

  double *d_A = nullptr;
  double *d_B = nullptr;
  double *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(double));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblas_dsyr2k(handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n,
                      k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Complex Float SYR2K
void RocBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose trans, int n,
                           int k, const std::complex<float> *alpha,
                           const std::complex<float> *A, int lda,
                           const std::complex<float> *B, int ldb,
                           const std::complex<float> *beta,
                           std::complex<float> *C, int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeB = (trans == Transpose::NoTrans) ? ldb * k : ldb * n;
  size_t sizeC = ldc * n;

  std::complex<float> *d_A = nullptr;
  std::complex<float> *d_B = nullptr;
  std::complex<float> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<float>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);

  clap_rocblas_csyr2k(handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n,
                      k, reinterpret_cast<const rocblas_float_complex *>(alpha),
                      reinterpret_cast<const rocblas_float_complex *>(d_A), lda,
                      reinterpret_cast<const rocblas_float_complex *>(d_B), ldb,
                      reinterpret_cast<const rocblas_float_complex *>(beta),
                      reinterpret_cast<rocblas_float_complex *>(d_C), ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Complex Double SYR2K
void RocBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose trans, int n,
                           int k, const std::complex<double> *alpha,
                           const std::complex<double> *A, int lda,
                           const std::complex<double> *B, int ldb,
                           const std::complex<double> *beta,
                           std::complex<double> *C, int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeB = (trans == Transpose::NoTrans) ? ldb * k : ldb * n;
  size_t sizeC = ldc * n;

  std::complex<double> *d_A = nullptr;
  std::complex<double> *d_B = nullptr;
  std::complex<double> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<double>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);

  clap_rocblas_zsyr2k(
      handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n, k,
      reinterpret_cast<const rocblas_double_complex *>(alpha),
      reinterpret_cast<const rocblas_double_complex *>(d_A), lda,
      reinterpret_cast<const rocblas_double_complex *>(d_B), ldb,
      reinterpret_cast<const rocblas_double_complex *>(beta),
      reinterpret_cast<rocblas_double_complex *>(d_C), ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Float TRMM
void RocBlasBackend::trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                          Diag diag, int m, int n, float alpha, const float *A,
                          int lda, float *B, int ldb) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(float));
  clap_hipMalloc((void **)&d_C, sizeB * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblas_strmm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo),
                     to_rocblas_trans(trans), to_rocblas_diag(diag),
                     (rocblas_int)m, (rocblas_int)n, &alpha, d_A,
                     (rocblas_int)lda, d_B, (rocblas_int)ldb, d_C,
                     (rocblas_int)ldb);

  clap_hipMemcpy(B, d_C, sizeB * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Double TRMM
void RocBlasBackend::trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                          Diag diag, int m, int n, double alpha,
                          const double *A, int lda, double *B, int ldb) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;

  double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(double));
  clap_hipMalloc((void **)&d_C, sizeB * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblas_dtrmm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo),
                     to_rocblas_trans(trans), to_rocblas_diag(diag),
                     (rocblas_int)m, (rocblas_int)n, &alpha, d_A,
                     (rocblas_int)lda, d_B, (rocblas_int)ldb, d_C,
                     (rocblas_int)ldb);

  clap_hipMemcpy(B, d_C, sizeB * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Complex Float TRMM
void RocBlasBackend::trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                          Diag diag, int m, int n,
                          const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          std::complex<float> *B, int ldb) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;

  std::complex<float> *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_C, sizeB * sizeof(std::complex<float>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);

  clap_rocblas_ctrmm(
      handle, to_rocblas_side(side), to_rocblas_uplo(uplo),
      to_rocblas_trans(trans), to_rocblas_diag(diag), (rocblas_int)m,
      (rocblas_int)n, reinterpret_cast<const rocblas_float_complex *>(alpha),
      reinterpret_cast<const rocblas_float_complex *>(d_A), (rocblas_int)lda,
      reinterpret_cast<const rocblas_float_complex *>(d_B), (rocblas_int)ldb,
      reinterpret_cast<rocblas_float_complex *>(d_C), (rocblas_int)ldb);

  clap_hipMemcpy(B, d_C, sizeB * sizeof(std::complex<float>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Complex Double TRMM
void RocBlasBackend::trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                          Diag diag, int m, int n,
                          const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          std::complex<double> *B, int ldb) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;

  std::complex<double> *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_C, sizeB * sizeof(std::complex<double>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);

  clap_rocblas_ztrmm(
      handle, to_rocblas_side(side), to_rocblas_uplo(uplo),
      to_rocblas_trans(trans), to_rocblas_diag(diag), (rocblas_int)m,
      (rocblas_int)n, reinterpret_cast<const rocblas_double_complex *>(alpha),
      reinterpret_cast<const rocblas_double_complex *>(d_A), (rocblas_int)lda,
      reinterpret_cast<const rocblas_double_complex *>(d_B), (rocblas_int)ldb,
      reinterpret_cast<rocblas_double_complex *>(d_C), (rocblas_int)ldb);

  clap_hipMemcpy(B, d_C, sizeB * sizeof(std::complex<double>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Float TRSM
void RocBlasBackend::trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                          Diag diag, int m, int n, float alpha, const float *A,
                          int lda, float *B, int ldb) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;

  float *d_A = nullptr;
  float *d_B = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(float));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(float));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(float), hipMemcpyHostToDevice);

  clap_rocblas_strsm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo),
                     to_rocblas_trans(trans), to_rocblas_diag(diag), m, n,
                     &alpha, d_A, lda, d_B, ldb);

  clap_hipMemcpy(B, d_B, sizeB * sizeof(float), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
}

// Double TRSM
void RocBlasBackend::trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                          Diag diag, int m, int n, double alpha,
                          const double *A, int lda, double *B, int ldb) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;

  double *d_A = nullptr;
  double *d_B = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(double));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(double));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(double), hipMemcpyHostToDevice);

  clap_rocblas_dtrsm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo),
                     to_rocblas_trans(trans), to_rocblas_diag(diag), m, n,
                     &alpha, d_A, lda, d_B, ldb);

  clap_hipMemcpy(B, d_B, sizeB * sizeof(double), hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
}

// Complex Float TRSM
void RocBlasBackend::trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                          Diag diag, int m, int n,
                          const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          std::complex<float> *B, int ldb) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;

  std::complex<float> *d_A = nullptr;
  std::complex<float> *d_B = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<float>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);

  clap_rocblas_ctrsm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo),
                     to_rocblas_trans(trans), to_rocblas_diag(diag), m, n,
                     reinterpret_cast<const rocblas_float_complex *>(alpha),
                     reinterpret_cast<const rocblas_float_complex *>(d_A), lda,
                     reinterpret_cast<rocblas_float_complex *>(d_B), ldb);

  clap_hipMemcpy(B, d_B, sizeB * sizeof(std::complex<float>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
}

// Complex Double TRSM
void RocBlasBackend::trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                          Diag diag, int m, int n,
                          const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          std::complex<double> *B, int ldb) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;

  std::complex<double> *d_A = nullptr;
  std::complex<double> *d_B = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<double>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);

  clap_rocblas_ztrsm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo),
                     to_rocblas_trans(trans), to_rocblas_diag(diag), m, n,
                     reinterpret_cast<const rocblas_double_complex *>(alpha),
                     reinterpret_cast<const rocblas_double_complex *>(d_A), lda,
                     reinterpret_cast<rocblas_double_complex *>(d_B), ldb);

  clap_hipMemcpy(B, d_B, sizeB * sizeof(std::complex<double>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
}

// Complex Float HEMM
void RocBlasBackend::hemm(Layout layout, Side side, Uplo uplo, int m, int n,
                          const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          const std::complex<float> *B, int ldb,
                          const std::complex<float> *beta,
                          std::complex<float> *C, int ldc) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;

  std::complex<float> *d_A = nullptr;
  std::complex<float> *d_B = nullptr;
  std::complex<float> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<float>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);

  clap_rocblas_chemm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo), m, n,
                     reinterpret_cast<const rocblas_float_complex *>(alpha),
                     reinterpret_cast<const rocblas_float_complex *>(d_A), lda,
                     reinterpret_cast<const rocblas_float_complex *>(d_B), ldb,
                     reinterpret_cast<const rocblas_float_complex *>(beta),
                     reinterpret_cast<rocblas_float_complex *>(d_C), ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Complex Double HEMM
void RocBlasBackend::hemm(Layout layout, Side side, Uplo uplo, int m, int n,
                          const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          const std::complex<double> *B, int ldb,
                          const std::complex<double> *beta,
                          std::complex<double> *C, int ldc) {
  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;

  std::complex<double> *d_A = nullptr;
  std::complex<double> *d_B = nullptr;
  std::complex<double> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<double>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);

  clap_rocblas_zhemm(handle, to_rocblas_side(side), to_rocblas_uplo(uplo), m, n,
                     reinterpret_cast<const rocblas_double_complex *>(alpha),
                     reinterpret_cast<const rocblas_double_complex *>(d_A), lda,
                     reinterpret_cast<const rocblas_double_complex *>(d_B), ldb,
                     reinterpret_cast<const rocblas_double_complex *>(beta),
                     reinterpret_cast<rocblas_double_complex *>(d_C), ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Complex Float HERK
void RocBlasBackend::herk(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, float alpha, const std::complex<float> *A,
                          int lda, float beta, std::complex<float> *C,
                          int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeC = ldc * n;

  std::complex<float> *d_A = nullptr;
  std::complex<float> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<float>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);

  clap_rocblas_cherk(
      handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n, k, &alpha,
      reinterpret_cast<const rocblas_float_complex *>(d_A), lda, &beta,
      reinterpret_cast<rocblas_float_complex *>(d_C), ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_C);
}

// Complex Double HERK
void RocBlasBackend::herk(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, double alpha, const std::complex<double> *A,
                          int lda, double beta, std::complex<double> *C,
                          int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeC = ldc * n;

  std::complex<double> *d_A = nullptr;
  std::complex<double> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<double>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);

  clap_rocblas_zherk(
      handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n, k, &alpha,
      reinterpret_cast<const rocblas_double_complex *>(d_A), lda, &beta,
      reinterpret_cast<rocblas_double_complex *>(d_C), ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_C);
}

// Complex Float HER2K
void RocBlasBackend::her2k(Layout layout, Uplo uplo, Transpose trans, int n,
                           int k, const std::complex<float> *alpha,
                           const std::complex<float> *A, int lda,
                           const std::complex<float> *B, int ldb, float beta,
                           std::complex<float> *C, int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeB = (trans == Transpose::NoTrans) ? ldb * k : ldb * n;
  size_t sizeC = ldc * n;

  std::complex<float> *d_A = nullptr;
  std::complex<float> *d_B = nullptr;
  std::complex<float> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<float>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<float>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyHostToDevice);

  clap_rocblas_cher2k(handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n,
                      k, reinterpret_cast<const rocblas_float_complex *>(alpha),
                      reinterpret_cast<const rocblas_float_complex *>(d_A), lda,
                      reinterpret_cast<const rocblas_float_complex *>(d_B), ldb,
                      &beta, reinterpret_cast<rocblas_float_complex *>(d_C),
                      ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<float>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

// Complex Double HER2K
void RocBlasBackend::her2k(Layout layout, Uplo uplo, Transpose trans, int n,
                           int k, const std::complex<double> *alpha,
                           const std::complex<double> *A, int lda,
                           const std::complex<double> *B, int ldb, double beta,
                           std::complex<double> *C, int ldc) {
  size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
  size_t sizeB = (trans == Transpose::NoTrans) ? ldb * k : ldb * n;
  size_t sizeC = ldc * n;

  std::complex<double> *d_A = nullptr;
  std::complex<double> *d_B = nullptr;
  std::complex<double> *d_C = nullptr;

  clap_hipMalloc((void **)&d_A, sizeA * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_B, sizeB * sizeof(std::complex<double>));
  clap_hipMalloc((void **)&d_C, sizeC * sizeof(std::complex<double>));

  clap_hipMemcpy(d_A, A, sizeA * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_B, B, sizeB * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);
  clap_hipMemcpy(d_C, C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyHostToDevice);

  clap_rocblas_zher2k(
      handle, to_rocblas_uplo(uplo), to_rocblas_trans(trans), n, k,
      reinterpret_cast<const rocblas_double_complex *>(alpha),
      reinterpret_cast<const rocblas_double_complex *>(d_A), lda,
      reinterpret_cast<const rocblas_double_complex *>(d_B), ldb, &beta,
      reinterpret_cast<rocblas_double_complex *>(d_C), ldc);

  clap_hipMemcpy(C, d_C, sizeC * sizeof(std::complex<double>),
                 hipMemcpyDeviceToHost);

  clap_hipFree(d_A);
  clap_hipFree(d_B);
  clap_hipFree(d_C);
}

}

