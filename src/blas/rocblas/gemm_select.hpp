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

void gemm_fp32_strided_batched(
    rocblas_handle handle,
    Layout layout,
    Transpose transA,
    Transpose transB,
    int m, int n, int k,
    float alpha,
    const float* A, int lda, long long strideA,
    const float* B, int ldb, long long strideB,
    float beta,
    float* C, int ldc, long long strideC,
    int batch_count,
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


// Scaled FP8 device GEMM through hipBLASLt.
// Returns false when the datatype/shape has no supported algorithm.
bool gemm_scaled_fp8(
    Layout layout,
    Transpose transA,
    Transpose transB,
    int m, int n, int k,
    float alpha,
    const void* A, BlasDataType typeA, int lda,
    const float* scaleA,
    const void* B, BlasDataType typeB, int ldb,
    const float* scaleB,
    float beta,
    void* C, BlasDataType typeC, int ldc,
    void* stream);

} // namespace rocblas_gemm_select
} // namespace clap
