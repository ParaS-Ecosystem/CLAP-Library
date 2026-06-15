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
                         int64_t m, int64_t n, int64_t k, float alpha,
                         const float *A, int64_t lda, const float *B,
                         int64_t ldb, float beta, float *C, int64_t ldc) {

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
                         int64_t m, int64_t n, int64_t k, double alpha,
                         const double *A, int64_t lda, const double *B,
                         int64_t ldb, double beta, double *C, int64_t ldc)

{
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
    cuLdc = static_cast<int>(n); // output is row-major → columns = n
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

void CuBlasBackend::symm(Layout layout, Side side, Uplo uplo, int64_t m,
                         int64_t n, float alpha, const float *A, int64_t lda,
                         const float *B, int64_t ldb, float beta, float *C,
                         int64_t ldc) {

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

void CuBlasBackend::symm(Layout layout, Side side, Uplo uplo, int64_t m,
                         int64_t n, double alpha, const double *A, int64_t lda,
                         const double *B, int64_t ldb, double beta, double *C,
                         int64_t ldc) {

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

void CuBlasBackend::syrk(Layout layout, Uplo uplo, Transpose trans, int64_t n,
                         int64_t k, float alpha, const float *A, int64_t lda,
                         float beta, float *C, int64_t ldc) {

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

void CuBlasBackend::syrk(Layout layout, Uplo uplo, Transpose trans, int64_t n,
                         int64_t k, double alpha, const double *A, int64_t lda,
                         double beta, double *C, int64_t ldc) {

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

// Double TRMM

void CuBlasBackend::trmm(Layout layout, Side side, Uplo uplo, Transpose trans,
                         Diag diag, int64_t m, int64_t n, double alpha,
                         const double *A, int64_t lda, double *B, int64_t ldb) {
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
                         Diag diag, int64_t m, int64_t n, float alpha,
                         const float *A, int64_t lda, float *B, int64_t ldb) {
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
                   &alpha, dA, (int)lda, dB, (int)ldb, dC,
                   (int)ldb); // ldc = ldb

  clap_cudaMemcpy(B, dC, ldb * n * sizeof(float), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dC);
}

// Float TRSM

void CuBlasBackend::trsm(Layout layout, Side side, Uplo uplo, Transpose trans,
                         Diag diag, int64_t m, int64_t n, float alpha,
                         const float *A, int64_t lda, float *B, int64_t ldb) {
  if (layout == Layout::RowMajor) {
    std::swap(m, n);

    side = (side == Side::Left) ? Side::Right : Side::Left;
    uplo = (uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;

    trans =
        (trans == Transpose::NoTrans) ? Transpose::Trans : Transpose::NoTrans;
  }

  float *dA = nullptr, *dB = nullptr;

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
                         Diag diag, int64_t m, int64_t n, double alpha,
                         const double *A, int64_t lda, double *B, int64_t ldb) {
  if (layout == Layout::RowMajor) {
    std::swap(m, n);

    side = (side == Side::Left) ? Side::Right : Side::Left;
    uplo = (uplo == Uplo::Upper) ? Uplo::Lower : Uplo::Upper;

    trans =
        (trans == Transpose::NoTrans) ? Transpose::Trans : Transpose::NoTrans;
  }

  double *dA = nullptr, *dB = nullptr;

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

} // namespace clap
