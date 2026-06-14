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


#include "clap/rocblas_backend.hpp"
#include "clap/rocblas_utils.hpp"

namespace clap {

//Level 3 Rocblas routines

//Float GEMM

void RocBlasBackend::gemm(Layout layout,
                          Transpose transA,
                          Transpose transB,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          float alpha,
                          const float* A, int64_t lda,
                          const float* B, int64_t ldb,
                          float beta,
                          float* C, int64_t ldc)
{
    size_t sizeA = lda * ((transA == Transpose::NoTrans) ? k : m);
    size_t sizeB = ldb * ((transB == Transpose::NoTrans) ? n : k);
    size_t sizeC = ldc * n;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(float));
    clap_hipMalloc((void**)&d_B, sizeB * sizeof(float));
    clap_hipMalloc((void**)&d_C, sizeC * sizeof(float));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_B, B, sizeB * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_C, C, sizeC * sizeof(float), hipMemcpyHostToDevice);

    clap_rocblas_sgemm(handle,
                       to_rocblas_trans(transA),
                       to_rocblas_trans(transB),
                       (rocblas_int)m, (rocblas_int)n, (rocblas_int)k,
                       &alpha,
                       d_A, (rocblas_int)lda,
                       d_B, (rocblas_int)ldb,
                       &beta,
                       d_C, (rocblas_int)ldc);

    clap_hipMemcpy(C, d_C, sizeC * sizeof(float), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_B);
    clap_hipFree(d_C);
}

//Double GEMM

void RocBlasBackend::gemm(Layout layout,
                          Transpose transA,
                          Transpose transB,
                          int64_t m,
                          int64_t n,
                          int64_t k,
                          double alpha,
                          const double* A, int64_t lda,
                          const double* B, int64_t ldb,
                          double beta,
                          double* C, int64_t ldc)
{
    size_t sizeA = lda * ((transA == Transpose::NoTrans) ? k : m);
    size_t sizeB = ldb * ((transB == Transpose::NoTrans) ? n : k);
    size_t sizeC = ldc * n;

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(double));
    clap_hipMalloc((void**)&d_B, sizeB * sizeof(double));
    clap_hipMalloc((void**)&d_C, sizeC * sizeof(double));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_B, B, sizeB * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_C, C, sizeC * sizeof(double), hipMemcpyHostToDevice);

    clap_rocblas_dgemm(handle,
                       to_rocblas_trans(transA),
                       to_rocblas_trans(transB),
                       (rocblas_int)m, (rocblas_int)n, (rocblas_int)k,
                       &alpha,
                       d_A, (rocblas_int)lda,
                       d_B, (rocblas_int)ldb,
                       &beta,
                       d_C, (rocblas_int)ldc);

    clap_hipMemcpy(C, d_C, sizeC * sizeof(double), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_B);
    clap_hipFree(d_C);
}


//Float SYMM

void RocBlasBackend::symm(Layout layout,
                          Side side,
                          Uplo uplo,
                          int64_t m,
                          int64_t n,
                          float alpha,
                          const float* A, int64_t lda,
                          const float* B, int64_t ldb,
                          float beta,
                          float* C, int64_t ldc)
{
    size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(float));
    clap_hipMalloc((void**)&d_B, sizeB * sizeof(float));
    clap_hipMalloc((void**)&d_C, sizeC * sizeof(float));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_B, B, sizeB * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_C, C, sizeC * sizeof(float), hipMemcpyHostToDevice);

    clap_rocblas_ssymm(handle,
                       to_rocblas_side(side),
                       to_rocblas_uplo(uplo),
                       m, n,
                       &alpha,
                       d_A, lda,
                       d_B, ldb,
                       &beta,
                       d_C, ldc);

    clap_hipMemcpy(C, d_C, sizeC * sizeof(float), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_B);
    clap_hipFree(d_C);
}

//Double SYMM

void RocBlasBackend::symm(Layout layout,
                          Side side,
                          Uplo uplo,
                          int64_t m,
                          int64_t n,
                          double alpha,
                          const double* A, int64_t lda,
                          const double* B, int64_t ldb,
                          double beta,
                          double* C, int64_t ldc)
{
    size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
    size_t sizeB = ldb * n;
    size_t sizeC = ldc * n;

    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(double));
    clap_hipMalloc((void**)&d_B, sizeB * sizeof(double));
    clap_hipMalloc((void**)&d_C, sizeC * sizeof(double));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_B, B, sizeB * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_C, C, sizeC * sizeof(double), hipMemcpyHostToDevice);

    clap_rocblas_dsymm(handle,
                       to_rocblas_side(side),
                       to_rocblas_uplo(uplo),
                       m, n,
                       &alpha,
                       d_A, lda,
                       d_B, ldb,
                       &beta,
                       d_C, ldc);

    clap_hipMemcpy(C, d_C, sizeC * sizeof(double), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_B);
    clap_hipFree(d_C);
}


//Float SYRK

void RocBlasBackend::syrk(Layout layout,
                          Uplo uplo,
                          Transpose trans,
                          int64_t n,
                          int64_t k,
                          float alpha,
                          const float* A, int64_t lda,
                          float beta,
                          float* C, int64_t ldc)
{
    size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
    size_t sizeC = ldc * n;

    float* d_A = nullptr;
    float* d_C = nullptr;

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(float));
    clap_hipMalloc((void**)&d_C, sizeC * sizeof(float));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_C, C, sizeC * sizeof(float), hipMemcpyHostToDevice);

    clap_rocblas_ssyrk(handle,
                       to_rocblas_uplo(uplo),
                       to_rocblas_trans(trans),
                       n, k,
                       &alpha,
                       d_A, lda,
                       &beta,
                       d_C, ldc);

    clap_hipMemcpy(C, d_C, sizeC * sizeof(float), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_C);
}

//Double SYRK

void RocBlasBackend::syrk(Layout layout,
                          Uplo uplo,
                          Transpose trans,
                          int64_t n,
                          int64_t k,
                          double alpha,
                          const double* A, int64_t lda,
                          double beta,
                          double* C, int64_t ldc)
{
    size_t sizeA = (trans == Transpose::NoTrans) ? lda * k : lda * n;
    size_t sizeC = ldc * n;

    double* d_A = nullptr;
    double* d_C = nullptr;

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(double));
    clap_hipMalloc((void**)&d_C, sizeC * sizeof(double));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_C, C, sizeC * sizeof(double), hipMemcpyHostToDevice);

    clap_rocblas_dsyrk(handle,
                       to_rocblas_uplo(uplo),
                       to_rocblas_trans(trans),
                       n, k,
                       &alpha,
                       d_A, lda,
                       &beta,
                       d_C, ldc);

    clap_hipMemcpy(C, d_C, sizeC * sizeof(double), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_C);
}


//Float TRMM

void RocBlasBackend::trmm(Layout layout,
                          Side side, Uplo uplo, Transpose trans, Diag diag,
                          int64_t m, int64_t n,
                          float alpha,
                          const float* A, int64_t lda,
                          float* B,       int64_t ldb)
{
    size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
    size_t sizeB = ldb * n;

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(float));
    clap_hipMalloc((void**)&d_B, sizeB * sizeof(float));
    clap_hipMalloc((void**)&d_C, sizeB * sizeof(float));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_B, B, sizeB * sizeof(float), hipMemcpyHostToDevice);

    clap_rocblas_strmm(handle,
                       to_rocblas_side(side), to_rocblas_uplo(uplo),
                       to_rocblas_trans(trans), to_rocblas_diag(diag),
                       (rocblas_int)m, (rocblas_int)n,
                       &alpha, d_A, (rocblas_int)lda,
                               d_B, (rocblas_int)ldb,
                               d_C, (rocblas_int)ldb);

    clap_hipMemcpy(B, d_C, sizeB * sizeof(float), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_B);
    clap_hipFree(d_C);
}

//Double TRMM

void RocBlasBackend::trmm(Layout layout,
                          Side side, Uplo uplo, Transpose trans, Diag diag,
                          int64_t m, int64_t n,
                          double alpha,
                          const double* A, int64_t lda,
                          double* B,       int64_t ldb)
{
    size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
    size_t sizeB = ldb * n;

    double *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(double));
    clap_hipMalloc((void**)&d_B, sizeB * sizeof(double));
    clap_hipMalloc((void**)&d_C, sizeB * sizeof(double));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_B, B, sizeB * sizeof(double), hipMemcpyHostToDevice);

    clap_rocblas_dtrmm(handle,
                       to_rocblas_side(side), to_rocblas_uplo(uplo),
                       to_rocblas_trans(trans), to_rocblas_diag(diag),
                       (rocblas_int)m, (rocblas_int)n,
                       &alpha, d_A, (rocblas_int)lda,
                               d_B, (rocblas_int)ldb,
                               d_C, (rocblas_int)ldb);

    clap_hipMemcpy(B, d_C, sizeB * sizeof(double), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_B);
    clap_hipFree(d_C);
}


//Float TRSM

void RocBlasBackend::trsm(Layout layout,
                          Side side, Uplo uplo, Transpose trans, Diag diag,
                          int64_t m, int64_t n,
                          float alpha,
                          const float* A, int64_t lda,
                          float* B, int64_t ldb)
{
    size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
    size_t sizeB = ldb * n;

    float* d_A = nullptr;
    float* d_B = nullptr;

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(float));
    clap_hipMalloc((void**)&d_B, sizeB * sizeof(float));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(float), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_B, B, sizeB * sizeof(float), hipMemcpyHostToDevice);

    clap_rocblas_strsm(handle,
                       to_rocblas_side(side), to_rocblas_uplo(uplo),
                       to_rocblas_trans(trans), to_rocblas_diag(diag),
                       m, n,
                       &alpha,
                       d_A, lda,
                       d_B, ldb);

    clap_hipMemcpy(B, d_B, sizeB * sizeof(float), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_B);
}

//Double TRSM

void RocBlasBackend::trsm(Layout layout,
                          Side side, Uplo uplo, Transpose trans, Diag diag,
                          int64_t m, int64_t n,
                          double alpha,
                          const double* A, int64_t lda,
                          double* B, int64_t ldb)
{
    size_t sizeA = (side == Side::Left) ? lda * m : lda * n;
    size_t sizeB = ldb * n;

    double* d_A = nullptr;
    double* d_B = nullptr;

    clap_hipMalloc((void**)&d_A, sizeA * sizeof(double));
    clap_hipMalloc((void**)&d_B, sizeB * sizeof(double));

    clap_hipMemcpy(d_A, A, sizeA * sizeof(double), hipMemcpyHostToDevice);
    clap_hipMemcpy(d_B, B, sizeB * sizeof(double), hipMemcpyHostToDevice);

    clap_rocblas_dtrsm(handle,
                       to_rocblas_side(side), to_rocblas_uplo(uplo),
                       to_rocblas_trans(trans), to_rocblas_diag(diag),
                       m, n,
                       &alpha,
                       d_A, lda,
                       d_B, ldb);

    clap_hipMemcpy(B, d_B, sizeB * sizeof(double), hipMemcpyDeviceToHost);

    clap_hipFree(d_A);
    clap_hipFree(d_B);
}

} // namespace clap
