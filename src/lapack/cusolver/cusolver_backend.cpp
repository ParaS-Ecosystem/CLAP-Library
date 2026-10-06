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

#define CLAP_DISABLE_BLAS_MACROS
#include "clap/lapack/cusolver/cusolver_backend.hpp"
#include "clap/lapack_dyn.hpp"
#include "clap/dyn_backends.hpp"
#include "clap/detail/checked_call.hpp"
#include <stdexcept>
#include <vector>
#include <iostream>

namespace clap {

using detail::checked_call;

// ── cuSOLVER fill mode / operation helpers ────────────────────────────────
static int to_cusolver_uplo(Uplo u) {
    return (u == Uplo::Upper) ? 1 : 0;   // 1=UPPER 0=LOWER
}

static int to_cusolver_trans(Transpose t) {
    switch (t) {
        case Transpose::NoTrans:   return 0;
        case Transpose::Trans:     return 1;
        case Transpose::ConjTrans: return 2;
    }
    return 0;
}

static int to_cusolver_job(Job j) {
    return (j == Job::Vec) ? 1 : 0;   // CUSOLVER_EIG_MODE_VECTOR=1
}

// ── Constructor ───────────────────────────────────────────────────────────
CuSolverBackend::CuSolverBackend() : m_handle(nullptr) {
    std::cout << "Using cuSOLVER backend\n";

    // Load cuSOLVER library
    if (!dyn_lapack::loadCuSolver())
        throw std::runtime_error("cuSOLVER load failed");

    if (!dyn::loadCudaAndCublas())
        throw std::runtime_error("CUDA runtime load failed for cuSOLVER");

    // Create cuSOLVER handle
    cusolverStatus_t st = dyn_lapack::p_cusolverDnCreate(&m_handle);
    if (st != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error("cusolverDnCreate failed");
}

CuSolverBackend::~CuSolverBackend() {
    if (m_handle && dyn_lapack::p_cusolverDnDestroy)
        dyn_lapack::p_cusolverDnDestroy(m_handle);
}

static void check_cuda(cudaError_t status, const char *operation) {
    if (status != 0)
        throw std::runtime_error(std::string(operation) + " failed with CUDA status " +
                                 std::to_string(static_cast<int>(status)));
}

static void check_cusolver(cusolverStatus_t status, const char *operation) {
    if (status != CUSOLVER_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + " failed with cuSOLVER status " +
                                 std::to_string(static_cast<int>(status)));
}


const auto clap_cudaMalloc = checked_call(dyn::p_cudaMalloc, check_cuda, "cudaMalloc");
const auto clap_cudaFree = checked_call(dyn::p_cudaFree, check_cuda, "cudaFree");
const auto clap_cudaMemcpy = checked_call(dyn::p_cudaMemcpy, check_cuda, "cudaMemcpy");
const auto clap_cudaDeviceSynchronize = checked_call(dyn::p_cudaDeviceSynchronize, check_cuda, "cudaDeviceSynchronize");

const auto clap_cusolverDnSgetrf_bufferSize = checked_call(dyn_lapack::p_cusolverDnSgetrf_bufferSize, check_cusolver, "cusolverDnSgetrf_bufferSize");
const auto clap_cusolverDnDgetrf_bufferSize = checked_call(dyn_lapack::p_cusolverDnDgetrf_bufferSize, check_cusolver, "cusolverDnDgetrf_bufferSize");
const auto clap_cusolverDnSgetrf = checked_call(dyn_lapack::p_cusolverDnSgetrf, check_cusolver, "cusolverDnSgetrf");
const auto clap_cusolverDnDgetrf = checked_call(dyn_lapack::p_cusolverDnDgetrf, check_cusolver, "cusolverDnDgetrf");
const auto clap_cusolverDnSgetrs = checked_call(dyn_lapack::p_cusolverDnSgetrs, check_cusolver, "cusolverDnSgetrs");
const auto clap_cusolverDnDgetrs = checked_call(dyn_lapack::p_cusolverDnDgetrs, check_cusolver, "cusolverDnDgetrs");
const auto clap_cusolverDnSpotrf_bufferSize = checked_call(dyn_lapack::p_cusolverDnSpotrf_bufferSize, check_cusolver, "cusolverDnSpotrf_bufferSize");
const auto clap_cusolverDnDpotrf_bufferSize = checked_call(dyn_lapack::p_cusolverDnDpotrf_bufferSize, check_cusolver, "cusolverDnDpotrf_bufferSize");
const auto clap_cusolverDnSpotrf = checked_call(dyn_lapack::p_cusolverDnSpotrf, check_cusolver, "cusolverDnSpotrf");
const auto clap_cusolverDnDpotrf = checked_call(dyn_lapack::p_cusolverDnDpotrf, check_cusolver, "cusolverDnDpotrf");
const auto clap_cusolverDnSpotri_bufferSize = checked_call(dyn_lapack::p_cusolverDnSpotri_bufferSize, check_cusolver, "cusolverDnSpotri_bufferSize");
const auto clap_cusolverDnDpotri_bufferSize = checked_call(dyn_lapack::p_cusolverDnDpotri_bufferSize, check_cusolver, "cusolverDnDpotri_bufferSize");
const auto clap_cusolverDnSpotri = checked_call(dyn_lapack::p_cusolverDnSpotri, check_cusolver, "cusolverDnSpotri");
const auto clap_cusolverDnDpotri = checked_call(dyn_lapack::p_cusolverDnDpotri, check_cusolver, "cusolverDnDpotri");
const auto clap_cusolverDnSpotrs = checked_call(dyn_lapack::p_cusolverDnSpotrs, check_cusolver, "cusolverDnSpotrs");
const auto clap_cusolverDnDpotrs = checked_call(dyn_lapack::p_cusolverDnDpotrs, check_cusolver, "cusolverDnDpotrs");
const auto clap_cusolverDnSgesvd_bufferSize = checked_call(dyn_lapack::p_cusolverDnSgesvd_bufferSize, check_cusolver, "cusolverDnSgesvd_bufferSize");
const auto clap_cusolverDnDgesvd_bufferSize = checked_call(dyn_lapack::p_cusolverDnDgesvd_bufferSize, check_cusolver, "cusolverDnDgesvd_bufferSize");
const auto clap_cusolverDnSgesvd = checked_call(dyn_lapack::p_cusolverDnSgesvd, check_cusolver, "cusolverDnSgesvd");
const auto clap_cusolverDnDgesvd = checked_call(dyn_lapack::p_cusolverDnDgesvd, check_cusolver, "cusolverDnDgesvd");
const auto clap_cusolverDnSsyevd_bufferSize = checked_call(dyn_lapack::p_cusolverDnSsyevd_bufferSize, check_cusolver, "cusolverDnSsyevd_bufferSize");
const auto clap_cusolverDnDsyevd_bufferSize = checked_call(dyn_lapack::p_cusolverDnDsyevd_bufferSize, check_cusolver, "cusolverDnDsyevd_bufferSize");
const auto clap_cusolverDnSsyevd = checked_call(dyn_lapack::p_cusolverDnSsyevd, check_cusolver, "cusolverDnSsyevd");
const auto clap_cusolverDnDsyevd = checked_call(dyn_lapack::p_cusolverDnDsyevd, check_cusolver, "cusolverDnDsyevd");

// ══════════════════════════════════════════════════════════════════════════
// INTERNAL TEMPLATE HELPERS
// Each allocates device memory, copies H→D, runs kernel, copies D→H, frees
// ══════════════════════════════════════════════════════════════════════════

// ── getrf_impl ────────────────────────────────────────────────────────────
template<>
void CuSolverBackend::getrf_impl<float>(lapack_int m, lapack_int n,
                                         float *A, lapack_int lda,
                                         lapack_int *ipiv, lapack_int *info) {
    
    size_t szA  = (size_t)lda * n * sizeof(float);

    // Device allocations
    float *dA       = nullptr;
    int   *d_ipiv   = nullptr;
    int   *d_info   = nullptr;
    float *d_work   = nullptr;
    int    lwork    = 0;

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&d_ipiv, (size_t)std::min(m,n) * sizeof(int));
    clap_cudaMalloc((void**)&d_info, sizeof(int));

    // Copy A host → device
    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    // Query workspace size
    clap_cusolverDnSgetrf_bufferSize(m_handle, m, n, dA, lda, &lwork);
    clap_cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(float));

    // Factorize
    clap_cusolverDnSgetrf(m_handle, m, n, dA, lda, d_work, d_ipiv, d_info);
    clap_cudaDeviceSynchronize();

    // Copy results back
    clap_cudaMemcpy(A,    dA,     szA,                              cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(ipiv, d_ipiv, (size_t)std::min(m,n)*sizeof(int), cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int),                       cudaMemcpyDeviceToHost);

    clap_cudaFree(dA);
    clap_cudaFree(d_ipiv);
    clap_cudaFree(d_info);
    clap_cudaFree(d_work);
}

template<>
void CuSolverBackend::getrf_impl<double>(lapack_int m, lapack_int n,
                                          double *A, lapack_int lda,
                                          lapack_int *ipiv, lapack_int *info) {
    
    size_t szA = (size_t)lda * n * sizeof(double);

    double *dA     = nullptr;
    int    *d_ipiv = nullptr;
    int    *d_info = nullptr;
    double *d_work = nullptr;
    int     lwork  = 0;

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&d_ipiv, (size_t)std::min(m,n) * sizeof(int));
    clap_cudaMalloc((void**)&d_info, sizeof(int));

    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    clap_cusolverDnDgetrf_bufferSize(m_handle, m, n, dA, lda, &lwork);
    clap_cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(double));

    clap_cusolverDnDgetrf(m_handle, m, n, dA, lda, d_work, d_ipiv, d_info);
    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(A,    dA,     szA,                              cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(ipiv, d_ipiv, (size_t)std::min(m,n)*sizeof(int), cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int),                       cudaMemcpyDeviceToHost);

    clap_cudaFree(dA); clap_cudaFree(d_ipiv);
    clap_cudaFree(d_info); clap_cudaFree(d_work);
}

// ── getrs_impl ────────────────────────────────────────────────────────────
template<>
void CuSolverBackend::getrs_impl<float>(Transpose trans,
                                         lapack_int n, lapack_int nrhs,
                                         const float *A, lapack_int lda,
                                         const lapack_int *ipiv,
                                         float *B, lapack_int ldb,
                                         lapack_int *info) {
    
    size_t szA = (size_t)lda * n    * sizeof(float);
    size_t szB = (size_t)ldb * nrhs * sizeof(float);

    float *dA   = nullptr; int *d_ipiv = nullptr;
    float *dB   = nullptr; int *d_info = nullptr;

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&dB,     szB);
    clap_cudaMalloc((void**)&d_ipiv, (size_t)n * sizeof(int));
    clap_cudaMalloc((void**)&d_info, sizeof(int));

    clap_cudaMemcpy(dA,    A,    szA,                   cudaMemcpyHostToDevice);
    clap_cudaMemcpy(dB,    B,    szB,                   cudaMemcpyHostToDevice);
    clap_cudaMemcpy(d_ipiv,ipiv, (size_t)n*sizeof(int), cudaMemcpyHostToDevice);

    clap_cusolverDnSgetrs(m_handle, to_cusolver_trans(trans),
                          n, nrhs, dA, lda, d_ipiv, dB, ldb, d_info);
    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(B,    dB,     szB,        cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    clap_cudaFree(dA); clap_cudaFree(dB);
    clap_cudaFree(d_ipiv); clap_cudaFree(d_info);
}

template<>
void CuSolverBackend::getrs_impl<double>(Transpose trans,
                                          lapack_int n, lapack_int nrhs,
                                          const double *A, lapack_int lda,
                                          const lapack_int *ipiv,
                                          double *B, lapack_int ldb,
                                          lapack_int *info) {
   
    size_t szA = (size_t)lda * n    * sizeof(double);
    size_t szB = (size_t)ldb * nrhs * sizeof(double);

    double *dA = nullptr; int *d_ipiv = nullptr;
    double *dB = nullptr; int *d_info = nullptr;

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&dB,     szB);
    clap_cudaMalloc((void**)&d_ipiv, (size_t)n * sizeof(int));
    clap_cudaMalloc((void**)&d_info, sizeof(int));

    clap_cudaMemcpy(dA,    A,    szA,                   cudaMemcpyHostToDevice);
    clap_cudaMemcpy(dB,    B,    szB,                   cudaMemcpyHostToDevice);
    clap_cudaMemcpy(d_ipiv,ipiv, (size_t)n*sizeof(int), cudaMemcpyHostToDevice);

    clap_cusolverDnDgetrs(m_handle, to_cusolver_trans(trans),
                          n, nrhs, dA, lda, d_ipiv, dB, ldb, d_info);
    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(B,    dB,     szB,        cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    clap_cudaFree(dA); clap_cudaFree(dB);
    clap_cudaFree(d_ipiv); clap_cudaFree(d_info);
}

// ── potrf_impl ────────────────────────────────────────────────────────────
template<>
void CuSolverBackend::potrf_impl<float>(Uplo uplo, lapack_int n,
                                         float *A, lapack_int lda,
                                         lapack_int *info) {
    
    size_t szA = (size_t)lda * n * sizeof(float);

    float *dA     = nullptr; int *d_info = nullptr;
    float *d_work = nullptr; int  lwork  = 0;

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&d_info, sizeof(int));
    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    clap_cusolverDnSpotrf_bufferSize(m_handle,
        to_cusolver_uplo(uplo), n, dA, lda, &lwork);
    clap_cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(float));

    clap_cusolverDnSpotrf(m_handle,
        to_cusolver_uplo(uplo), n, dA, lda, d_work, lwork, d_info);
    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(A,    dA,     szA,        cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    clap_cudaFree(dA); clap_cudaFree(d_info); clap_cudaFree(d_work);
}

template<>
void CuSolverBackend::potrf_impl<double>(Uplo uplo, lapack_int n,
                                          double *A, lapack_int lda,
                                          lapack_int *info) {
    
    size_t szA = (size_t)lda * n * sizeof(double);

    double *dA     = nullptr; int *d_info = nullptr;
    double *d_work = nullptr; int  lwork  = 0;

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&d_info, sizeof(int));
    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    clap_cusolverDnDpotrf_bufferSize(m_handle,
        to_cusolver_uplo(uplo), n, dA, lda, &lwork);
    clap_cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(double));

    clap_cusolverDnDpotrf(m_handle,
        to_cusolver_uplo(uplo), n, dA, lda, d_work, lwork, d_info);
    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(A,    dA,     szA,        cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    clap_cudaFree(dA); clap_cudaFree(d_info); clap_cudaFree(d_work);
}
// ── potri_impl ────────────────────────────────────────────────────────────
template<>
void CuSolverBackend::potri_impl<float>(Uplo uplo, lapack_int n, float *A, lapack_int lda,
                                         lapack_int *info) {
    
    
    size_t szA = (size_t)lda * n * sizeof(float);

    float *dA     = nullptr; int *d_info = nullptr;
    float *d_work = nullptr; int  lwork  = 0;

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&d_info, sizeof(int));
    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    clap_cusolverDnSpotri_bufferSize(m_handle,
        to_cusolver_uplo(uplo), n, dA, lda, &lwork);
    clap_cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(float));

    clap_cusolverDnSpotri(m_handle,
        to_cusolver_uplo(uplo), n, dA, lda, d_work, lwork, d_info);
    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(A,    dA,     szA,        cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    clap_cudaFree(dA); clap_cudaFree(d_info); clap_cudaFree(d_work);                          
}

template<>
void CuSolverBackend::potri_impl<double>(Uplo uplo, lapack_int n, double *A, lapack_int lda,
                                         lapack_int *info) {
   
    
    size_t szA = (size_t)lda * n * sizeof(double);

    double *dA     = nullptr; int *d_info = nullptr;
    double *d_work = nullptr; int  lwork  = 0;

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&d_info, sizeof(int));
    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    clap_cusolverDnDpotri_bufferSize(m_handle,
        to_cusolver_uplo(uplo), n, dA, lda, &lwork);
    clap_cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(double));

    clap_cusolverDnDpotri(m_handle,
        to_cusolver_uplo(uplo), n, dA, lda, d_work, lwork, d_info);
    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(A,    dA,     szA,        cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    clap_cudaFree(dA); clap_cudaFree(d_info); clap_cudaFree(d_work);                          
}

// ── potrs_impl ────────────────────────────────────────────────────────────
template<>
void CuSolverBackend::potrs_impl<float>(Uplo uplo, lapack_int n, lapack_int nrhs,
    									const float *A, lapack_int lda,float *B, 
										lapack_int ldb,lapack_int *info) {
    

    size_t szA = (size_t)lda * n * sizeof(float);
    size_t szB = (size_t)ldb * nrhs * sizeof(float);

    float *dA = nullptr;
    float *dB = nullptr;
    int *d_info = nullptr;

    clap_cudaMalloc((void**)&dA, szA);
    clap_cudaMalloc((void**)&dB, szB);
    clap_cudaMalloc((void**)&d_info, sizeof(int));

    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

    clap_cusolverDnSpotrs(m_handle, to_cusolver_uplo(uplo), n,
        				nrhs, dA, lda, dB, ldb, d_info);

    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);

    clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    clap_cudaFree(dA);
    clap_cudaFree(dB);
    clap_cudaFree(d_info);
}

template<>
void CuSolverBackend::potrs_impl<double>(Uplo uplo, lapack_int n, lapack_int nrhs,
    									const double *A, lapack_int lda, double *B, 
										lapack_int ldb, lapack_int *info) {
    

    size_t szA = (size_t)lda * n * sizeof(double);
    size_t szB = (size_t)ldb * nrhs * sizeof(double);

    double *dA = nullptr;
    double *dB = nullptr;
    int *d_info = nullptr;

    clap_cudaMalloc((void**)&dA, szA);
    clap_cudaMalloc((void**)&dB, szB);
    clap_cudaMalloc((void**)&d_info, sizeof(int));

    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

    clap_cusolverDnDpotrs(m_handle, to_cusolver_uplo(uplo),
        				n, nrhs, dA, lda, dB, ldb, d_info);

    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);

    clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    clap_cudaFree(dA);
    clap_cudaFree(dB);
    clap_cudaFree(d_info);
}

// ── gesvd_impl ────────────────────────────────────────────────────────────
template<>
void CuSolverBackend::gesvd_impl<float>(Job jobu, Job jobvt,
                                         lapack_int m, lapack_int n,
                                         float *A, lapack_int lda, float *s,
                                         float *U, lapack_int ldu,
                                         float *VT, lapack_int ldvt,
                                         lapack_int *info) {
    
    int k = std::min(m, n);
    size_t szA = (size_t)lda * n * sizeof(float);
    size_t szU = (size_t)ldu * m * sizeof(float);
    size_t szV = (size_t)ldvt* n * sizeof(float);

    float *dA=nullptr, *dU=nullptr, *dVT=nullptr, *dS=nullptr;
    float *d_work=nullptr; int *d_info=nullptr; int lwork=0;

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&dU,     szU);
    clap_cudaMalloc((void**)&dVT,    szV);
    clap_cudaMalloc((void**)&dS,     (size_t)k * sizeof(float));
    clap_cudaMalloc((void**)&d_info, sizeof(int));

    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    clap_cusolverDnSgesvd_bufferSize(m_handle, m, n, &lwork);
    clap_cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(float));

    char jobu_c  = static_cast<char>(jobu);
    char jobvt_c = static_cast<char>(jobvt);

    clap_cusolverDnSgesvd(m_handle, jobu_c, jobvt_c, m, n,
                          dA, lda, dS, dU, ldu, dVT, ldvt,
                          d_work, lwork, nullptr, d_info);
    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(A,    dA,     szA,                    cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(s,    dS,     (size_t)k*sizeof(float), cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(U,    dU,     szU,                    cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(VT,   dVT,    szV,                    cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int),             cudaMemcpyDeviceToHost);

    clap_cudaFree(dA); clap_cudaFree(dU); clap_cudaFree(dVT);
    clap_cudaFree(dS); clap_cudaFree(d_work); clap_cudaFree(d_info);
}

template<>
void CuSolverBackend::gesvd_impl<double>(Job jobu, Job jobvt,
                                          lapack_int m, lapack_int n,
                                          double *A, lapack_int lda, double *s,
                                          double *U, lapack_int ldu,
                                          double *VT, lapack_int ldvt,
                                          lapack_int *info) {
    
    int k = std::min(m, n);
    size_t szA = (size_t)lda * n * sizeof(double);
    size_t szU = (size_t)ldu * m * sizeof(double);
    size_t szV = (size_t)ldvt* n * sizeof(double);

    double *dA=nullptr, *dU=nullptr, *dVT=nullptr, *dS=nullptr;
    double *d_work=nullptr; int *d_info=nullptr; int lwork=0;

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&dU,     szU);
    clap_cudaMalloc((void**)&dVT,    szV);
    clap_cudaMalloc((void**)&dS,     (size_t)k * sizeof(double));
    clap_cudaMalloc((void**)&d_info, sizeof(int));

    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    clap_cusolverDnDgesvd_bufferSize(m_handle, m, n, &lwork);
    clap_cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(double));

    char jobu_c  = static_cast<char>(jobu);
    char jobvt_c = static_cast<char>(jobvt);

    clap_cusolverDnDgesvd(m_handle, jobu_c, jobvt_c, m, n,
                          dA, lda, dS, dU, ldu, dVT, ldvt,
                          d_work, lwork, nullptr, d_info);
    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(s,    dS,     (size_t)k*sizeof(double), cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(U,    dU,     szU,                     cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(VT,   dVT,    szV,                     cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int),              cudaMemcpyDeviceToHost);

    clap_cudaFree(dA); clap_cudaFree(dU); clap_cudaFree(dVT);
    clap_cudaFree(dS); clap_cudaFree(d_work); clap_cudaFree(d_info);
}

// ── syev_impl ─────────────────────────────────────────────────────────────
template<>
void CuSolverBackend::syev_impl<float>(Job jobz, Uplo uplo, lapack_int n,
                                        float *A, lapack_int lda, float *w,
                                        lapack_int *info) {
    
    size_t szA = (size_t)lda * n * sizeof(float);

    float *dA=nullptr, *dW=nullptr, *d_work=nullptr; int *d_info=nullptr;
    int lwork=0;
    int jobz_i = to_cusolver_job(jobz);
    int uplo_i = to_cusolver_uplo(uplo);

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&dW,     (size_t)n * sizeof(float));
    clap_cudaMalloc((void**)&d_info, sizeof(int));
    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    clap_cusolverDnSsyevd_bufferSize(m_handle, jobz_i, uplo_i,
                                     n, dA, lda, dW, &lwork);
    clap_cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(float));

    clap_cusolverDnSsyevd(m_handle, jobz_i, uplo_i,
                          n, dA, lda, dW, d_work, lwork, d_info);
    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(A,    dA,     szA,                   cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(w,    dW,     (size_t)n*sizeof(float), cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int),             cudaMemcpyDeviceToHost);

    clap_cudaFree(dA); clap_cudaFree(dW);
    clap_cudaFree(d_work); clap_cudaFree(d_info);
}

template<>
void CuSolverBackend::syev_impl<double>(Job jobz, Uplo uplo, lapack_int n,
                                         double *A, lapack_int lda, double *w,
                                         lapack_int *info) {
     
    size_t szA = (size_t)lda * n * sizeof(double);

    double *dA=nullptr, *dW=nullptr, *d_work=nullptr; int *d_info=nullptr;
    int lwork=0;
    int jobz_i = to_cusolver_job(jobz);
    int uplo_i = to_cusolver_uplo(uplo);

    clap_cudaMalloc((void**)&dA,     szA);
    clap_cudaMalloc((void**)&dW,     (size_t)n * sizeof(double));
    clap_cudaMalloc((void**)&d_info, sizeof(int));
    clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

    clap_cusolverDnDsyevd_bufferSize(m_handle, jobz_i, uplo_i,
                                     n, dA, lda, dW, &lwork);
    clap_cudaMalloc((void**)&d_work, (size_t)lwork * sizeof(double));

    clap_cusolverDnDsyevd(m_handle, jobz_i, uplo_i,
                          n, dA, lda, dW, d_work, lwork, d_info);
    clap_cudaDeviceSynchronize();

    clap_cudaMemcpy(A,    dA,     szA,                    cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(w,    dW,     (size_t)n*sizeof(double), cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(info, d_info, sizeof(int),              cudaMemcpyDeviceToHost);

    clap_cudaFree(dA); clap_cudaFree(dW);
    clap_cudaFree(d_work); clap_cudaFree(d_info);
}

// ══════════════════════════════════════════════════════════════════════════
// PUBLIC API — dispatch to template implementations
// ══════════════════════════════════════════════════════════════════════════

void CuSolverBackend::sgetrf(Layout, lapack_int m, lapack_int n,
    float *A, lapack_int lda, lapack_int *ipiv, lapack_int *info) {
    getrf_impl<float>(m, n, A, lda, ipiv, info);
}
void CuSolverBackend::dgetrf(Layout, lapack_int m, lapack_int n,
    double *A, lapack_int lda, lapack_int *ipiv, lapack_int *info) {
    getrf_impl<double>(m, n, A, lda, ipiv, info);
}
void CuSolverBackend::sgetrs(Layout, Transpose t, lapack_int n, lapack_int nrhs,
    const float *A, lapack_int lda, const lapack_int *ipiv,
    float *B, lapack_int ldb, lapack_int *info) {
    getrs_impl<float>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void CuSolverBackend::dgetrs(Layout, Transpose t, lapack_int n, lapack_int nrhs,
    const double *A, lapack_int lda, const lapack_int *ipiv,
    double *B, lapack_int ldb, lapack_int *info) {
    getrs_impl<double>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void CuSolverBackend::sgesv(lapack_int n, lapack_int nrhs,
    float *A, lapack_int lda, lapack_int *ipiv,
    float *B, lapack_int ldb, lapack_int *info) {
   
    getrf_impl<float>(n, n, A, lda, ipiv, info);
    if (*info == 0)
        getrs_impl<float>(Transpose::NoTrans, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void CuSolverBackend::dgesv(lapack_int n, lapack_int nrhs,
    double *A, lapack_int lda, lapack_int *ipiv,
    double *B, lapack_int ldb, lapack_int *info) {
    
    getrf_impl<double>(n, n, A, lda, ipiv, info);
    if (*info == 0)
        getrs_impl<double>(Transpose::NoTrans, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void CuSolverBackend::sgetri(lapack_int, float*, lapack_int,
    const lapack_int*, lapack_int *info) {
    // cuSOLVER has no direct getri — use trtri + CPU fallback
    std::cerr << "[CLAP] sgetri not supported in cuSOLVER, use CPU backend\n";
    *info = -1;
}
void CuSolverBackend::dgetri(lapack_int, double*, lapack_int,
    const lapack_int*, lapack_int *info) {
    std::cerr << "[CLAP] dgetri not supported in cuSOLVER, use CPU backend\n";
    *info = -1;
}
void CuSolverBackend::spotrf(Layout, Uplo u, lapack_int n,
    float *A, lapack_int lda, lapack_int *info) {
    potrf_impl<float>(u, n, A, lda, info);
}
void CuSolverBackend::dpotrf(Layout, Uplo u, lapack_int n,
    double *A, lapack_int lda, lapack_int *info) {
    potrf_impl<double>(u, n, A, lda, info);
}
void CuSolverBackend::spotri(Layout, Uplo u, lapack_int n,
    float* A, lapack_int lda, lapack_int *info) {
    potri_impl<float>(u, n, A, lda, info);
}
void CuSolverBackend::dpotri(Layout, Uplo u, lapack_int n,
    double* A, lapack_int lda, lapack_int *info) {
    potri_impl<double>(u, n, A, lda, info);
}
void CuSolverBackend::spotrs(Layout, Uplo u, lapack_int n, lapack_int nrhs,
    const float *A, lapack_int lda, float *B, lapack_int ldb, lapack_int *info) {
    potrs_impl<float>(u, n, nrhs, A, lda, B, ldb, info);
}
void CuSolverBackend::dpotrs(Layout, Uplo u, lapack_int n, lapack_int nrhs,
    const double *A, lapack_int lda, double* B, lapack_int ldb, lapack_int *info) {
    potrs_impl<double>(u, n, nrhs, A, lda, B, ldb, info);
}
void CuSolverBackend::sposv(Layout l, Uplo u, lapack_int n, lapack_int nrhs,
    float *A, lapack_int lda, float *B, lapack_int ldb, lapack_int *info) {
    potrf_impl<float>(u, n, A, lda, info);
    if (*info == 0) spotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
void CuSolverBackend::dposv(Layout l, Uplo u, lapack_int n, lapack_int nrhs,
    double *A, lapack_int lda, double *B, lapack_int ldb, lapack_int *info) {
    potrf_impl<double>(u, n, A, lda, info);
    if (*info == 0) dpotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
// QR — cuSOLVER has geqrf via cusolverDnSgeqrf
void CuSolverBackend::sgeqrf(Layout, lapack_int, lapack_int,
    float*, lapack_int, float*, lapack_int *info) {
    std::cerr << "[CLAP] cuSOLVER geqrf not yet wired\n"; *info = -1;
}
void CuSolverBackend::dgeqrf(Layout, lapack_int, lapack_int,
    double*, lapack_int, double*, lapack_int *info) {
    std::cerr << "[CLAP] cuSOLVER geqrf not yet wired\n"; *info = -1;
}
void CuSolverBackend::sorgqr(Layout, lapack_int, lapack_int, lapack_int,
    float*, lapack_int, const float*, lapack_int *info) { *info=-1; }
void CuSolverBackend::dorgqr(Layout, lapack_int, lapack_int, lapack_int,
    double*, lapack_int, const double*, lapack_int *info) { *info=-1; }
void CuSolverBackend::sgels(Layout, Transpose, lapack_int, lapack_int, lapack_int,
    float*, lapack_int, float*, lapack_int, lapack_int *info) { *info=-1; }
void CuSolverBackend::dgels(Layout, Transpose, lapack_int, lapack_int, lapack_int,
    double*, lapack_int, double*, lapack_int, lapack_int *info) { *info=-1; }

void CuSolverBackend::ssyev(Layout, Job j, Uplo u, lapack_int n,
    float *A, lapack_int lda, float *w, lapack_int *info) {
    syev_impl<float>(j, u, n, A, lda, w, info);
}
void CuSolverBackend::dsyev(Layout, Job j, Uplo u, lapack_int n,
    double *A, lapack_int lda, double *w, lapack_int *info) {
    syev_impl<double>(j, u, n, A, lda, w, info);
}
void CuSolverBackend::sgeev(Layout, Job, Job, lapack_int,
    float*, lapack_int, float*, float*, float*, lapack_int, float*, lapack_int,
    lapack_int *info) {
    std::cerr << "[CLAP] cuSOLVER geev not supported\n"; *info=-1;
}
void CuSolverBackend::dgeev(Layout, Job, Job, lapack_int,
    double*, lapack_int, double*, double*, double*, lapack_int, double*, lapack_int,
    lapack_int *info) {
    std::cerr << "[CLAP] cuSOLVER geev not supported\n"; *info=-1;
}
void CuSolverBackend::sgesvd(Layout, Job ju, Job jv, lapack_int m, lapack_int n,
    float *A, lapack_int lda, float *s,
    float *U, lapack_int ldu, float *VT, lapack_int ldvt,
    float*, lapack_int *info) {
    gesvd_impl<float>(ju, jv, m, n, A, lda, s, U, ldu, VT, ldvt, info);
}
void CuSolverBackend::dgesvd(Layout, Job ju, Job jv, lapack_int m, lapack_int n,
    double *A, lapack_int lda, double *s,
    double *U, lapack_int ldu, double *VT, lapack_int ldvt,
    double*, lapack_int *info) {
    gesvd_impl<double>(ju, jv, m, n, A, lda, s, U, ldu, VT, ldvt, info);
}
void CuSolverBackend::strtrs(Layout, Uplo, Transpose, Diag,
    lapack_int, lapack_int, const float*, lapack_int, float*, lapack_int,
    lapack_int *info) {
    std::cerr << "[CLAP] cuSOLVER trtrs not supported\n"; *info=-1;
}
void CuSolverBackend::dtrtrs(Layout, Uplo, Transpose, Diag,
    lapack_int, lapack_int, const double*, lapack_int, double*, lapack_int,
    lapack_int *info) {
    std::cerr << "[CLAP] cuSOLVER trtrs not supported\n"; *info=-1;
}

} // namespace clap
