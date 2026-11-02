#pragma once

#include "clap/blas/rocblas/rocblas_utils.hpp"

namespace clap {
namespace rocblas_gemm_select {

void gemm_fp32(
    rocblas_handle handle,
    Layout layout,
    Transpose transA,
    Transpose transB,
    int m, int n, int k,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float beta,
    float* C, int ldc,
    void* stream);

void gemm_bf16(
    rocblas_handle handle,
    Layout layout,
    Transpose transA,
    Transpose transB,
    int m, int n, int k,
    float alpha,
    const void* A, int lda,
    const void* B, int ldb,
    float beta,
    void* C, int ldc,
    void* stream);

} // namespace rocblas_gemm_select
} // namespace clap
