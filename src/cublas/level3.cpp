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

#include "clap/cublas_backend.hpp"
#include "clap/cublas_utils.hpp"

namespace clap {

// Float GEMM
void CuBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                         int m, int n, int k, float alpha, const float *A,
                         int lda, const float *B, int ldb, float beta, float *C,
                         int ldc) {

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;

  size_t sizeA =
      static_cast<size_t>(lda) * ((transA == Transpose::NoTrans) ? k : m);

  size_t sizeB =
      static_cast<size_t>(ldb) * ((transB == Transpose::NoTrans) ? n : k);

  size_t sizeC = static_cast<size_t>(ldc) * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(float));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(float));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(float));

  clap_cudaMemcpy(dA, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(float), cudaMemcpyHostToDevice);

  cublasOperation_t cuTransA = to_cublas_trans(transA);
  cublasOperation_t cuTransB = to_cublas_trans(transB);

  int cuM = static_cast<int>(m);
  int cuN = static_cast<int>(n);
  int cuK = static_cast<int>(k);
  int cuLda = static_cast<int>(lda);
  int cuLdb = static_cast<int>(ldb);
  int cuLdc = static_cast<int>(ldc);

  if (layout == Layout::RowMajor) {
    std::swap(cuM, cuN);
    std::swap(cuTransA, cuTransB);

    cuLda = (transA == Transpose::NoTrans) ? static_cast<int>(k)
                                           : static_cast<int>(m);

    cuLdb = (transB == Transpose::NoTrans) ? static_cast<int>(n)
                                           : static_cast<int>(k);

    cuLdc = static_cast<int>(n);
  }

  float alpha_d = alpha;
  float beta_d = beta;

  clap_cublasSgemm(handle, cuTransA, cuTransB, cuM, cuN, cuK, &alpha_d, dA,
                   cuLda, dB, cuLdb, &beta_d, dC, cuLdc);

  clap_cudaMemcpy(C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Double GEMM
void CuBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                         int m, int n, int k, double alpha, const double *A,
                         int lda, const double *B, int ldb, double beta,
                         double *C, int ldc) {
  double *dA = nullptr, *dB = nullptr, *dC = nullptr;

  size_t sizeA =
      static_cast<size_t>(lda) * ((transA == Transpose::NoTrans) ? k : m);
  size_t sizeB =
      static_cast<size_t>(ldb) * ((transB == Transpose::NoTrans) ? n : k);
  size_t sizeC = static_cast<size_t>(ldc) * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(double));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(double));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(double));

  clap_cudaMemcpy(dA, A, sizeA * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(double), cudaMemcpyHostToDevice);

  cublasOperation_t cuTransA = to_cublas_trans(transA);
  cublasOperation_t cuTransB = to_cublas_trans(transB);
  int cuM = static_cast<int>(m);
  int cuN = static_cast<int>(n);
  int cuK = static_cast<int>(k);
  int cuLda = static_cast<int>(lda);
  int cuLdb = static_cast<int>(ldb);
  int cuLdc = static_cast<int>(ldc);

  if (layout == Layout::RowMajor) {
    std::swap(cuM, cuN);
    std::swap(cuTransA, cuTransB);

    cuLda = (transA == Transpose::NoTrans) ? static_cast<int>(k)
                                           : static_cast<int>(m);
    cuLdb = (transB == Transpose::NoTrans) ? static_cast<int>(n)
                                           : static_cast<int>(k);
    cuLdc = static_cast<int>(n);
  }

  double alpha_d = alpha;
  double beta_d = beta;
  clap_cublasDgemm(handle, cuTransA, cuTransB, cuM, cuN, cuK, &alpha_d, dA,
                   cuLda, dB, cuLdb, &beta_d, dC, cuLdc);

  clap_cudaMemcpy(C, dC, sizeC * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Float SYMM
void CuBlasBackend::symm(Layout layout, Side side, Uplo uplo, int m, int n,
                         float alpha, const float *A, int lda, const float *B,
                         int ldb, float beta, float *C, int ldc) {

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;

  clap_cudaMalloc((void **)&dA,
                  lda * (side == Side::Left ? m : n) * sizeof(float));
  clap_cudaMalloc((void **)&dB, ldb * n * sizeof(float));
  clap_cudaMalloc((void **)&dC, ldc * n * sizeof(float));

  clap_cudaMemcpy(dA, A, lda * (side == Side::Left ? m : n) * sizeof(float),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, ldb * n * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, ldc * n * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSsymm(handle, to_cublas_side(side), to_cublas_uplo(uplo), (int)m,
                   (int)n, &alpha, dA, (int)lda, dB, (int)ldb, &beta, dC,
                   (int)ldc);

  clap_cudaMemcpy(C, dC, ldc * n * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Double SYMM
void CuBlasBackend::symm(Layout layout, Side side, Uplo uplo, int m, int n,
                         double alpha, const double *A, int lda,
                         const double *B, int ldb, double beta, double *C,
                         int ldc) {

  double *dA = nullptr, *dB = nullptr, *dC = nullptr;

  clap_cudaMalloc((void **)&dA,
                  lda * (side == Side::Left ? m : n) * sizeof(double));
  clap_cudaMalloc((void **)&dB, ldb * n * sizeof(double));
  clap_cudaMalloc((void **)&dC, ldc * n * sizeof(double));

  clap_cudaMemcpy(dA, A, lda * (side == Side::Left ? m : n) * sizeof(double),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, ldb * n * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, ldc * n * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDsymm(handle, to_cublas_side(side), to_cublas_uplo(uplo), (int)m,
                   (int)n, &alpha, dA, (int)lda, dB, (int)ldb, &beta, dC,
                   (int)ldc);

  clap_cudaMemcpy(C, dC, ldc * n * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Float SYRK
void CuBlasBackend::syrk(Layout layout, Uplo uplo, Transpose trans, int n,
                         int k, float alpha, const float *A, int lda,
                         float beta, float *C, int ldc) {

  float *dA = nullptr, *dC = nullptr;

  clap_cudaMalloc((void **)&dA, lda * k * sizeof(float));
  clap_cudaMalloc((void **)&dC, ldc * n * sizeof(float));

  clap_cudaMemcpy(dA, A, lda * k * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, ldc * n * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSsyrk(handle, to_cublas_uplo(uplo), to_cublas_trans(trans), (int)n,
                   (int)k, &alpha, dA, (int)lda, &beta, dC, (int)ldc);

  clap_cudaMemcpy(C, dC, ldc * n * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dC);
}

// Double SYRK
void CuBlasBackend::syrk(Layout layout, Uplo uplo, Transpose trans, int n,
                         int k, double alpha, const double *A, int lda,
                         double beta, double *C, int ldc) {

  double *dA = nullptr, *dC = nullptr;

  clap_cudaMalloc((void **)&dA, lda * k * sizeof(double));
  clap_cudaMalloc((void **)&dC, ldc * n * sizeof(double));

  clap_cudaMemcpy(dA, A, lda * k * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, ldc * n * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDsyrk(handle, to_cublas_uplo(uplo), to_cublas_trans(trans), (int)n,
                   (int)k, &alpha, dA, (int)lda, &beta, dC, (int)ldc);

  clap_cudaMemcpy(C, dC, ldc * n * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dC);
}

// Float SYR2K
void CuBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, float alpha, const float *A, int lda,
                          const float *B, int ldb, float beta, float *C,
                          int ldc) {

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;

  clap_cudaMalloc((void **)&dA, lda * k * sizeof(float));
  clap_cudaMalloc((void **)&dB, ldb * k * sizeof(float));
  clap_cudaMalloc((void **)&dC, ldc * n * sizeof(float));

  clap_cudaMemcpy(dA, A, lda * k * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, ldb * k * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, ldc * n * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasSsyr2k(handle, to_cublas_uplo(uplo), to_cublas_trans(trans),
                    (int)n, (int)k, &alpha, dA, (int)lda, dB, (int)ldb, &beta,
                    dC, (int)ldc);

  clap_cudaMemcpy(C, dC, ldc * n * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Double SYR2K
void CuBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, double alpha, const double *A, int lda,
                          const double *B, int ldb, double beta, double *C,
                          int ldc) {

  double *dA = nullptr, *dB = nullptr, *dC = nullptr;

  clap_cudaMalloc((void **)&dA, lda * k * sizeof(double));
  clap_cudaMalloc((void **)&dB, ldb * k * sizeof(double));
  clap_cudaMalloc((void **)&dC, ldc * n * sizeof(double));

  clap_cudaMemcpy(dA, A, lda * k * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, ldb * k * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, ldc * n * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDsyr2k(handle, to_cublas_uplo(uplo), to_cublas_trans(trans),
                    (int)n, (int)k, &alpha, dA, (int)lda, dB, (int)ldb, &beta,
                    dC, (int)ldc);

  clap_cudaMemcpy(C, dC, ldc * n * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Double TRMM
void CuBlasBackend::trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                         Diag diag, int m, int n, double alpha, const double *A,
                         int lda, double *B, int ldb) {
  if (layout == Layout::RowMajor) {
    std::swap(m, n);

    side = (side == Side::Left) ? Side::Right : Side::Left;
    uplo = (uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;
    trans =
        (trans == Transpose::NoTrans) ? Transpose::Trans : Transpose::NoTrans;
  }

  double *dA = nullptr, *dB = nullptr, *dC = nullptr;

  int k = (side == Side::Left) ? m : n;

  clap_cudaMalloc((void **)&dA, lda * k * sizeof(double));
  clap_cudaMalloc((void **)&dB, ldb * n * sizeof(double));
  clap_cudaMalloc((void **)&dC, ldb * n * sizeof(double));

  clap_cudaMemcpy(dA, A, lda * k * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, ldb * n * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDtrmm(handle, to_cublas_side(side), to_cublas_uplo(uplo),
                   to_cublas_trans(trans), to_cublas_diag(diag), (int)m, (int)n,
                   &alpha, dA, (int)lda, dB, (int)ldb, dC, (int)ldb);

  clap_cudaMemcpy(B, dC, ldb * n * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Float TRMM
void CuBlasBackend::trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                         Diag diag, int m, int n, float alpha, const float *A,
                         int lda, float *B, int ldb) {
  if (layout == Layout::RowMajor) {
    std::swap(m, n);

    side = (side == Side::Left) ? Side::Right : Side::Left;
    uplo = (uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;
    trans =
        (trans == Transpose::NoTrans) ? Transpose::Trans : Transpose::NoTrans;
  }

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;

  int k = (side == Side::Left) ? m : n;

  clap_cudaMalloc((void **)&dA, lda * k * sizeof(float));
  clap_cudaMalloc((void **)&dB, ldb * n * sizeof(float));
  clap_cudaMalloc((void **)&dC, ldb * n * sizeof(float));

  clap_cudaMemcpy(dA, A, lda * k * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, ldb * n * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasStrmm(handle, to_cublas_side(side), to_cublas_uplo(uplo),
                   to_cublas_trans(trans), to_cublas_diag(diag), (int)m, (int)n,
                   &alpha, dA, (int)lda, dB, (int)ldb, dC, (int)ldb);

  clap_cudaMemcpy(B, dC, ldb * n * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Float TRSM
void CuBlasBackend::trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                         Diag diag, int m, int n, float alpha, const float *A,
                         int lda, float *B, int ldb) {
  if (layout == Layout::RowMajor) {
    std::swap(m, n);

    side = (side == Side::Left) ? Side::Right : Side::Left;
    uplo = (uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;

    trans =
        (trans == Transpose::NoTrans) ? Transpose::Trans : Transpose::NoTrans;
  }

  float *dA, *dB;

  clap_cudaMalloc((void **)&dA, lda * lda * sizeof(float));
  clap_cudaMalloc((void **)&dB, ldb * n * sizeof(float));

  clap_cudaMemcpy(dA, A, lda * lda * sizeof(float), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, ldb * n * sizeof(float), cudaMemcpyHostToDevice);

  clap_cublasStrsm(handle, to_cublas_side(side), to_cublas_uplo(uplo),
                   to_cublas_trans(trans), to_cublas_diag(diag), (int)m, (int)n,
                   &alpha, dA, (int)lda, dB, (int)ldb);

  clap_cudaMemcpy(B, dB, ldb * n * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
}

// Double TRSM
void CuBlasBackend::trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                         Diag diag, int m, int n, double alpha, const double *A,
                         int lda, double *B, int ldb) {
  if (layout == Layout::RowMajor) {
    std::swap(m, n);

    side = (side == Side::Left) ? Side::Right : Side::Left;
    uplo = (uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;

    trans =
        (trans == Transpose::NoTrans) ? Transpose::Trans : Transpose::NoTrans;
  }

  double *dA, *dB;

  clap_cudaMalloc((void **)&dA, lda * lda * sizeof(double));
  clap_cudaMalloc((void **)&dB, ldb * n * sizeof(double));

  clap_cudaMemcpy(dA, A, lda * lda * sizeof(double), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, ldb * n * sizeof(double), cudaMemcpyHostToDevice);

  clap_cublasDtrsm(handle, to_cublas_side(side), to_cublas_uplo(uplo),
                   to_cublas_trans(trans), to_cublas_diag(diag), (int)m, (int)n,
                   &alpha, dA, (int)lda, dB, (int)ldb);

  clap_cudaMemcpy(B, dB, ldb * n * sizeof(double), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
}

// Complex Float GEMM
void CuBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                         int m, int n, int K, const std::complex<float> *alpha,
                         const std::complex<float> *A, int lda,
                         const std::complex<float> *B, int ldb,
                         const std::complex<float> *beta,
                         std::complex<float> *C, int ldc) {

  cuComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;

  size_t sizeA = lda * K;
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuComplex));

  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuComplex), cudaMemcpyHostToDevice);

  cuComplex alpha_c = make_cuComplex(alpha->real(), alpha->imag());
  cuComplex beta_c = make_cuComplex(beta->real(), beta->imag());

  clap_cublasCgemm(handle, to_cublas_trans(transA), to_cublas_trans(transB),
                   (int)m, (int)n, (int)K, &alpha_c, dA, (int)lda, dB, (int)ldb,
                   &beta_c, dC, (int)ldc);

  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Complex Double GEMM
void CuBlasBackend::gemm(Layout layout, Transpose transA, Transpose transB,
                         int m, int n, int k, const std::complex<double> *alpha,
                         const std::complex<double> *A, int lda,
                         const std::complex<double> *B, int ldb,
                         const std::complex<double> *beta,
                         std::complex<double> *C, int ldc) {

  cuDoubleComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;

  size_t sizeA = lda * k;
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  cuDoubleComplex alpha_c = make_cuDoubleComplex(alpha->real(), alpha->imag());
  cuDoubleComplex beta_c = make_cuDoubleComplex(beta->real(), beta->imag());

  clap_cublasZgemm(handle, to_cublas_trans(transA), to_cublas_trans(transB),
                   (int)m, (int)n, (int)k, &alpha_c, dA, (int)lda, dB, (int)ldb,
                   &beta_c, dC, (int)ldc);

  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Complex Float SYMM
void CuBlasBackend::symm(Layout layout, Side side, Uplo uplo, int m, int n,
                         const std::complex<float> *alpha,
                         const std::complex<float> *A, int lda,
                         const std::complex<float> *B, int ldb,
                         const std::complex<float> *beta,
                         std::complex<float> *C, int ldc) {

  cuComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;

  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuComplex));

  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuComplex), cudaMemcpyHostToDevice);

  cuComplex alpha_c = make_cuComplex(alpha->real(), alpha->imag());
  cuComplex beta_c = make_cuComplex(beta->real(), beta->imag());

  clap_cublasCsymm(handle, to_cublas_side(side), to_cublas_uplo(uplo), (int)m,
                   (int)n, &alpha_c, dA, (int)lda, dB, (int)ldb, &beta_c, dC,
                   (int)ldc);

  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Complex Double SYMM
void CuBlasBackend::symm(Layout layout, Side side, Uplo uplo, int m, int n,
                         const std::complex<double> *alpha,
                         const std::complex<double> *A, int lda,
                         const std::complex<double> *B, int ldb,
                         const std::complex<double> *beta,
                         std::complex<double> *C, int ldc) {

  cuDoubleComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;

  size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  cuDoubleComplex alpha_c = make_cuDoubleComplex(alpha->real(), alpha->imag());
  cuDoubleComplex beta_c = make_cuDoubleComplex(beta->real(), beta->imag());

  clap_cublasZsymm(handle, to_cublas_side(side), to_cublas_uplo(uplo), (int)m,
                   (int)n, &alpha_c, dA, (int)lda, dB, (int)ldb, &beta_c, dC,
                   (int)ldc);

  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Complex Float SYRK
void CuBlasBackend::syrk(Layout layout, Uplo uplo, Transpose trans, int n,
                         int K, const std::complex<float> *alpha,
                         const std::complex<float> *A, int lda,
                         const std::complex<float> *beta,
                         std::complex<float> *C, int ldc) {

  cuComplex *dA = nullptr, *dC = nullptr;

  size_t sizeA = lda * ((trans == Transpose::NoTrans) ? K : n);
  size_t sizeC = ldc * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuComplex));

  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuComplex), cudaMemcpyHostToDevice);

  cuComplex alpha_c = make_cuComplex(alpha->real(), alpha->imag());
  cuComplex beta_c = make_cuComplex(beta->real(), beta->imag());

  clap_cublasCsyrk(handle, to_cublas_uplo(uplo), to_cublas_trans(trans), (int)n,
                   (int)K, &alpha_c, dA, (int)lda, &beta_c, dC, (int)ldc);

  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dC);
}

// Complex Double SYRK
void CuBlasBackend::syrk(Layout layout, Uplo uplo, Transpose trans, int n,
                         int k, const std::complex<double> *alpha,
                         const std::complex<double> *A, int lda,
                         const std::complex<double> *beta,
                         std::complex<double> *C, int ldc) {
  cuDoubleComplex *dA = nullptr, *dC = nullptr;

  size_t sizeA = lda * ((trans == Transpose::NoTrans) ? k : n);
  size_t sizeC = ldc * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  cuDoubleComplex alpha_c = make_cuDoubleComplex(alpha->real(), alpha->imag());
  cuDoubleComplex beta_c = make_cuDoubleComplex(beta->real(), beta->imag());

  clap_cublasZsyrk(handle, to_cublas_uplo(uplo), to_cublas_trans(trans), (int)n,
                   (int)k, &alpha_c, dA, (int)lda, &beta_c, dC, (int)ldc);

  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dC);
}

// Complex Float SYR2K
void CuBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          const std::complex<float> *B, int ldb,
                          const std::complex<float> *beta,
                          std::complex<float> *C, int ldc) {
  cuComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;

  size_t sizeA = lda * ((trans == Transpose::NoTrans) ? k : n);
  size_t sizeB = ldb * ((trans == Transpose::NoTrans) ? k : n);
  size_t sizeC = ldc * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuComplex));

  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuComplex), cudaMemcpyHostToDevice);

  cuComplex alpha_c = make_cuComplex(alpha->real(), alpha->imag());
  cuComplex beta_c = make_cuComplex(beta->real(), beta->imag());

  clap_cublasCsyr2k(handle, to_cublas_uplo(uplo), to_cublas_trans(trans),
                    (int)n, (int)k, &alpha_c, dA, (int)lda, dB, (int)ldb,
                    &beta_c, dC, (int)ldc);

  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Complex Double SYR2K
void CuBlasBackend::syr2k(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          const std::complex<double> *B, int ldb,
                          const std::complex<double> *beta,
                          std::complex<double> *C, int ldc) {

  cuDoubleComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;

  size_t sizeA = lda * ((trans == Transpose::NoTrans) ? k : n);
  size_t sizeB = ldb * ((trans == Transpose::NoTrans) ? k : n);
  size_t sizeC = ldc * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuDoubleComplex));

  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);

  cuDoubleComplex alpha_c = make_cuDoubleComplex(alpha->real(), alpha->imag());
  cuDoubleComplex beta_c = make_cuDoubleComplex(beta->real(), beta->imag());

  clap_cublasZsyr2k(handle, to_cublas_uplo(uplo), to_cublas_trans(trans),
                    (int)n, (int)k, &alpha_c, dA, (int)lda, dB, (int)ldb,
                    &beta_c, dC, (int)ldc);

  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Complex Float TRMM
void CuBlasBackend::trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                         Diag diag, int m, int n,
                         const std::complex<float> *alpha,
                         const std::complex<float> *A, int lda,
                         std::complex<float> *B, int ldb) {

  cuComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;

  size_t sizeA = lda * ((side == Side::Left) ? m : n);
  size_t sizeB = ldb * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dC, sizeB * sizeof(cuComplex));

  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuComplex), cudaMemcpyHostToDevice);

  cuComplex alpha_c = make_cuComplex(alpha->real(), alpha->imag());

  clap_cublasCtrmm(handle, to_cublas_side(side), to_cublas_uplo(uplo),
                   to_cublas_trans(trans), to_cublas_diag(diag), (int)m, (int)n,
                   &alpha_c, dA, (int)lda, dB, (int)ldb, dC, (int)ldb);

  clap_cudaMemcpy(B, dC, sizeB * sizeof(cuComplex), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Complex Double TRMM
void CuBlasBackend::trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                         Diag diag, int m, int n,
                         const std::complex<double> *alpha,
                         const std::complex<double> *A, int lda,
                         std::complex<double> *B, int ldb) {
  cuDoubleComplex *dA = nullptr, *dB = nullptr, *dC = nullptr;
  size_t sizeA = lda * ((side == Side::Left) ? m : n);
  size_t sizeB = ldb * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dC,
                  sizeB * sizeof(cuDoubleComplex));
  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  cuDoubleComplex alpha_c = make_cuDoubleComplex(alpha->real(), alpha->imag());
  clap_cublasZtrmm(handle, to_cublas_side(side), to_cublas_uplo(uplo),
                   to_cublas_trans(trans), to_cublas_diag(diag), (int)m, (int)n,
                   &alpha_c, dA, (int)lda, dB, (int)ldb, dC, (int)ldb);
  clap_cudaMemcpy(B, dC, sizeB * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);
  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Complex Float TRSM
void CuBlasBackend::trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                         Diag diag, int m, int n,
                         const std::complex<float> *alpha,
                         const std::complex<float> *A, int lda,
                         std::complex<float> *B, int ldb) {
  cuComplex *dA = nullptr, *dB = nullptr;

  size_t sizeA = lda * ((side == Side::Left) ? m : n);
  size_t sizeB = ldb * n;

  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuComplex));
  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuComplex), cudaMemcpyHostToDevice);
  cuComplex alpha_c = make_cuComplex(alpha->real(), alpha->imag());
  clap_cublasCtrsm(handle, to_cublas_side(side), to_cublas_uplo(uplo),
                   to_cublas_trans(trans), to_cublas_diag(diag), (int)m, (int)n,
                   &alpha_c, dA, (int)lda, dB, (int)ldb);
  clap_cudaMemcpy(B, dB, sizeB * sizeof(cuComplex), cudaMemcpyDeviceToHost);
  clap_cudaFree(dA);
  clap_cudaFree(dB);
}

// Complex Double TRSM
void CuBlasBackend::trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                         Diag diag, int m, int n,
                         const std::complex<double> *alpha,
                         const std::complex<double> *A, int lda,
                         std::complex<double> *B, int ldb) {
  cuDoubleComplex *dA = nullptr, *dB = nullptr;
  size_t sizeA = lda * ((side == Side::Left) ? m : n);
  size_t sizeB = ldb * n;
  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuDoubleComplex));
  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  cuDoubleComplex alpha_c = make_cuDoubleComplex(alpha->real(), alpha->imag());
  clap_cublasZtrsm(handle, to_cublas_side(side), to_cublas_uplo(uplo),
                   to_cublas_trans(trans), to_cublas_diag(diag), (int)m, (int)n,
                   &alpha_c, dA, (int)lda, dB, (int)ldb);
  clap_cudaMemcpy(B, dB, sizeB * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
}

// Complex Float HEMM
void CuBlasBackend::hemm(Layout layout, Side side, Uplo uplo, int m, int n,
                         const std::complex<float> *alpha,
                         const std::complex<float> *A, int lda,
                         const std::complex<float> *B, int ldb,
                         const std::complex<float> *beta,
                         std::complex<float> *C, int ldc) {
  cuComplex *dA, *dB, *dC;
  size_t sizeA = lda * ((side == Side::Left) ? m : n);
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;
  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuComplex));
  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuComplex), cudaMemcpyHostToDevice);
  cuComplex alpha_c = make_cuComplex(alpha->real(), alpha->imag());
  cuComplex beta_c = make_cuComplex(beta->real(), beta->imag());
  clap_cublasChemm(handle, to_cublas_side(side), to_cublas_uplo(uplo), (int)m,
                   (int)n, &alpha_c, dA, (int)lda, dB, (int)ldb, &beta_c, dC,
                   (int)ldc);
  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuComplex), cudaMemcpyDeviceToHost);
  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Complex Double HEMM
void CuBlasBackend::hemm(Layout layout, Side side, Uplo uplo, int m, int n,
                         const std::complex<double> *alpha,
                         const std::complex<double> *A, int lda,
                         const std::complex<double> *B, int ldb,
                         const std::complex<double> *beta,
                         std::complex<double> *C, int ldc) {
  cuDoubleComplex *dA, *dB, *dC;
  size_t sizeA = lda * ((side == Side::Left) ? m : n);
  size_t sizeB = ldb * n;
  size_t sizeC = ldc * n;
  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuDoubleComplex));
  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  cuDoubleComplex alpha_c = make_cuDoubleComplex(alpha->real(), alpha->imag());
  cuDoubleComplex beta_c = make_cuDoubleComplex(beta->real(), beta->imag());
  clap_cublasZhemm(handle, to_cublas_side(side), to_cublas_uplo(uplo), (int)m,
                   (int)n, &alpha_c, dA, (int)lda, dB, (int)ldb, &beta_c, dC,
                   (int)ldc);
  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);
  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Complex Float HERK
void CuBlasBackend::herk(Layout layout, Uplo uplo, Transpose trans, int n,
                         int k, float alpha, const std::complex<float> *A,
                         int lda, float beta, std::complex<float> *C, int ldc) {

  cuComplex *dA, *dC;
  size_t sizeA = lda * ((trans == Transpose::NoTrans) ? k : n);
  size_t sizeC = ldc * n;
  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuComplex));
  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuComplex), cudaMemcpyHostToDevice);
  cublasFillMode_t cuUplo = to_cublas_uplo(uplo);
  cublasOperation_t cuTrans = to_cublas_trans(trans);
  clap_cublasCherk(handle, cuUplo, cuTrans, (int)n, (int)k, &alpha, dA,
                   (int)lda, &beta, dC, (int)ldc);
  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuComplex), cudaMemcpyDeviceToHost);
  clap_cudaFree(dA);
  clap_cudaFree(dC);
}

// Complex Double HERK
void CuBlasBackend::herk(Layout layout, Uplo uplo, Transpose trans, int n,
                         int k, double alpha, const std::complex<double> *A,
                         int lda, double beta, std::complex<double> *C,
                         int ldc) {
  cuDoubleComplex *dA, *dC;
  size_t sizeA = lda * ((trans == Transpose::NoTrans) ? k : n);
  size_t sizeC = ldc * n;
  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuDoubleComplex));
  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  cublasFillMode_t cuUplo = to_cublas_uplo(uplo);
  cublasOperation_t cuTrans = to_cublas_trans(trans);
  clap_cublasZherk(handle, cuUplo, cuTrans, (int)n, (int)k, &alpha, dA,
                   (int)lda, &beta, dC, (int)ldc);
  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dC);
}

// Complex Float HER2K
void CuBlasBackend::her2k(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, const std::complex<float> *alpha,
                          const std::complex<float> *A, int lda,
                          const std::complex<float> *B, int ldb, float beta,
                          std::complex<float> *C, int ldc) {

  cuComplex *dA, *dB, *dC;

  size_t sizeA = lda * k;
  size_t sizeB = ldb * k;
  size_t sizeC = ldc * n;
  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuComplex));
  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuComplex), cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuComplex), cudaMemcpyHostToDevice);
  cuComplex alpha_c = make_cuComplex(alpha->real(), alpha->imag());
  clap_cublasCher2k(handle, to_cublas_uplo(uplo), to_cublas_trans(trans),
                    (int)n, (int)k, &alpha_c, dA, (int)lda, dB, (int)ldb, &beta,
                    dC, (int)ldc);
  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuComplex), cudaMemcpyDeviceToHost);
  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Complex Double HER2K
void CuBlasBackend::her2k(Layout layout, Uplo uplo, Transpose trans, int n,
                          int k, const std::complex<double> *alpha,
                          const std::complex<double> *A, int lda,
                          const std::complex<double> *B, int ldb, double beta,
                          std::complex<double> *C, int ldc) {
  cuDoubleComplex *dA, *dB, *dC;
  size_t sizeA = lda * k;
  size_t sizeB = ldb * k;
  size_t sizeC = ldc * n;
  clap_cudaMalloc((void **)&dA, sizeA * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dB, sizeB * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&dC, sizeC * sizeof(cuDoubleComplex));
  clap_cudaMemcpy(dA, A, sizeA * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, sizeB * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dC, C, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyHostToDevice);
  cuDoubleComplex alpha_c = make_cuDoubleComplex(alpha->real(), alpha->imag());
  clap_cublasZher2k(handle, to_cublas_uplo(uplo), to_cublas_trans(trans),
                    (int)n, (int)k, &alpha_c, dA, (int)lda, dB, (int)ldb, &beta,
                    dC, (int)ldc);
  clap_cudaMemcpy(C, dC, sizeC * sizeof(cuDoubleComplex),
                  cudaMemcpyDeviceToHost);
  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

}


