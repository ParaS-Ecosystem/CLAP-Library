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
#include "clap/lapack/rocsolver/rocsolver_backend.hpp"
#include "clap/lapack_dyn.hpp"
#include "clap/dyn_backends.hpp"
#include "clap/detail/checked_call.hpp"
#include <stdexcept>
#include <vector>
#include <iostream>
#include <algorithm>

namespace clap {

using detail::checked_call;

// ── rocSOLVER enum helpers ─────────────────────────────────────────────────
static int to_roc_uplo(Uplo u) {
    // rocblas_fill_upper=121, rocblas_fill_lower=122
    return (u == Uplo::Upper) ? 121 : 122;
}

static int to_roc_trans(Transpose t) {
    // rocblas_operation_none=111, transpose=112, conjugate=113
    switch (t) {
        case Transpose::NoTrans:   return 111;
        case Transpose::Trans:     return 112;
        case Transpose::ConjTrans: return 113;
    }
    return 111;
}

static int to_roc_evect(Job j) {
    return (j == Job::Vec) ? rocblas_evect_original_solver
                            : rocblas_evect_none_solver;
}

static int to_roc_svect(Job j) {
    switch (j) {
        case Job::All:   return rocblas_svect_all_solver;
        case Job::Some:  return rocblas_svect_singular_solver;
        case Job::Over:  return rocblas_svect_overwrite_solver;
        case Job::NoVec: return rocblas_svect_none_solver;
        default: throw std::invalid_argument("rocSOLVER gesvd requires All, Some, Over, or NoVec");
    }
}

static void check_hip(hipError_t status, const char *operation) {
    if (status != 0)
        throw std::runtime_error(std::string(operation) + " failed with HIP status " +
                                 std::to_string(static_cast<int>(status)));
}

static void check_rocsolver(int status, const char *operation) {
    if (status != rocblas_status_success_solver)
        throw std::runtime_error(std::string(operation) + " failed with rocSOLVER status " +
                                 std::to_string(status));
}


const auto clap_hipMalloc = checked_call(dyn::p_hipMalloc, check_hip, "hipMalloc");
const auto clap_hipFree = checked_call(dyn::p_hipFree, check_hip, "hipFree");
const auto clap_hipMemcpy = checked_call(dyn::p_hipMemcpy, check_hip, "hipMemcpy");
const auto clap_rocsolver_sgetrf = checked_call(dyn_lapack::p_rocsolver_sgetrf, check_rocsolver, "rocsolver_sgetrf");
const auto clap_rocsolver_dgetrf = checked_call(dyn_lapack::p_rocsolver_dgetrf, check_rocsolver, "rocsolver_dgetrf");
const auto clap_rocsolver_sgetrs = checked_call(dyn_lapack::p_rocsolver_sgetrs, check_rocsolver, "rocsolver_sgetrs");
const auto clap_rocsolver_dgetrs = checked_call(dyn_lapack::p_rocsolver_dgetrs, check_rocsolver, "rocsolver_dgetrs");
const auto clap_rocsolver_spotrf = checked_call(dyn_lapack::p_rocsolver_spotrf, check_rocsolver, "rocsolver_spotrf");
const auto clap_rocsolver_dpotrf = checked_call(dyn_lapack::p_rocsolver_dpotrf, check_rocsolver, "rocsolver_dpotrf");
const auto clap_rocsolver_spotri = checked_call(dyn_lapack::p_rocsolver_spotri, check_rocsolver, "rocsolver_spotri");
const auto clap_rocsolver_dpotri = checked_call(dyn_lapack::p_rocsolver_dpotri, check_rocsolver, "rocsolver_dpotri");
const auto clap_rocsolver_spotrs = checked_call(dyn_lapack::p_rocsolver_spotrs, check_rocsolver, "rocsolver_spotrs");
const auto clap_rocsolver_dpotrs = checked_call(dyn_lapack::p_rocsolver_dpotrs, check_rocsolver, "rocsolver_dpotrs");
const auto clap_rocsolver_sgesvd = checked_call(dyn_lapack::p_rocsolver_sgesvd, check_rocsolver, "rocsolver_sgesvd");
const auto clap_rocsolver_dgesvd = checked_call(dyn_lapack::p_rocsolver_dgesvd, check_rocsolver, "rocsolver_dgesvd");
const auto clap_rocsolver_ssyev = checked_call(dyn_lapack::p_rocsolver_ssyev, check_rocsolver, "rocsolver_ssyev");
const auto clap_rocsolver_dsyev = checked_call(dyn_lapack::p_rocsolver_dsyev, check_rocsolver, "rocsolver_dsyev");

// ── Constructor ───────────────────────────────────────────────────────────
RocSolverBackend::RocSolverBackend() : m_handle(nullptr) {
    std::cout << "Using rocSOLVER backend\n";

    if (!dyn_lapack::loadRocSolver())
        throw std::runtime_error("rocSOLVER load failed");

    if (!dyn::loadHipAndRocblas())
        throw std::runtime_error("HIP/rocBLAS runtime load failed for rocSOLVER");

    // rocSOLVER uses the rocBLAS handle
    if (dyn::p_rocblas_create_handle(&m_handle) != 0)
        throw std::runtime_error("rocblas_create_handle failed for rocSOLVER");
}

RocSolverBackend::~RocSolverBackend() {
    if (m_handle && dyn::p_rocblas_destroy_handle)
        dyn::p_rocblas_destroy_handle(m_handle);
}

// ══════════════════════════════════════════════════════════════════════════
// INTERNAL TEMPLATE HELPERS
// ══════════════════════════════════════════════════════════════════════════

template<>
void RocSolverBackend::getrf_impl<float>(lapack_int m, lapack_int n,
                                          float *A, lapack_int lda,
                                          lapack_int *ipiv, lapack_int *info) {
    
    int k = std::min(m, n);
    size_t szA = (size_t)lda * n * sizeof(float);

    float *dA     = nullptr;
    int   *d_ipiv = nullptr;
    int   *d_info = nullptr;

    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&d_ipiv, (size_t)k * sizeof(int));
    clap_hipMalloc((void**)&d_info, sizeof(int));

    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_rocsolver_sgetrf(m_handle, m, n, dA, lda, d_ipiv, d_info);

    // Synchronize via rocblas handle
    clap_hipMemcpy(A,    dA,     szA,                  hipMemcpyDeviceToHost);
    clap_hipMemcpy(ipiv, d_ipiv, (size_t)k*sizeof(int), hipMemcpyDeviceToHost);
    int h_info = 0;
    clap_hipMemcpy(&h_info, d_info, sizeof(int),        hipMemcpyDeviceToHost);
    *info = h_info;

    clap_hipFree(dA); clap_hipFree(d_ipiv); clap_hipFree(d_info);
}

template<>
void RocSolverBackend::getrf_impl<double>(lapack_int m, lapack_int n,
                                           double *A, lapack_int lda,
                                           lapack_int *ipiv, lapack_int *info) {
    
    int k = std::min(m, n);
    size_t szA = (size_t)lda * n * sizeof(double);

    double *dA    = nullptr;
    int *d_ipiv   = nullptr;
    int *d_info   = nullptr;

    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&d_ipiv, (size_t)k * sizeof(int));
    clap_hipMalloc((void**)&d_info, sizeof(int));

    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_rocsolver_dgetrf(m_handle, m, n, dA, lda, d_ipiv, d_info);

    clap_hipMemcpy(A,    dA,     szA,                  hipMemcpyDeviceToHost);
    clap_hipMemcpy(ipiv, d_ipiv, (size_t)k*sizeof(int), hipMemcpyDeviceToHost);
    int h_info = 0;
    clap_hipMemcpy(&h_info, d_info, sizeof(int),        hipMemcpyDeviceToHost);
    *info = h_info;

    clap_hipFree(dA); clap_hipFree(d_ipiv); clap_hipFree(d_info);
}

template<>
void RocSolverBackend::getrs_impl<float>(Transpose trans,
                                          lapack_int n, lapack_int nrhs,
                                          const float *A, lapack_int lda,
                                          const lapack_int *ipiv,
                                          float *B, lapack_int ldb,
                                          lapack_int *info) {
    
    size_t szA = (size_t)lda * n    * sizeof(float);
    size_t szB = (size_t)ldb * nrhs * sizeof(float);

    float *dA = nullptr; float *dB = nullptr;
    int *d_ipiv = nullptr;

    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&dB,     szB);
    clap_hipMalloc((void**)&d_ipiv, (size_t)n * sizeof(int));

    clap_hipMemcpy(dA,    A,    szA,                   hipMemcpyHostToDevice);
    clap_hipMemcpy(dB,    B,    szB,                   hipMemcpyHostToDevice);
    clap_hipMemcpy(d_ipiv,ipiv, (size_t)n*sizeof(int), hipMemcpyHostToDevice);

    clap_rocsolver_sgetrs(m_handle, to_roc_trans(trans),
                          n, nrhs, dA, lda, d_ipiv, dB, ldb);

    clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
    *info = 0;

    clap_hipFree(dA); clap_hipFree(dB); clap_hipFree(d_ipiv);
}

template<>
void RocSolverBackend::getrs_impl<double>(Transpose trans,
                                           lapack_int n, lapack_int nrhs,
                                           const double *A, lapack_int lda,
                                           const lapack_int *ipiv,
                                           double *B, lapack_int ldb,
                                           lapack_int *info) {
    
    size_t szA = (size_t)lda * n    * sizeof(double);
    size_t szB = (size_t)ldb * nrhs * sizeof(double);

    double *dA = nullptr; double *dB = nullptr;
    int *d_ipiv = nullptr;

    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&dB,     szB);
    clap_hipMalloc((void**)&d_ipiv, (size_t)n * sizeof(int));

    clap_hipMemcpy(dA,    A,    szA,                   hipMemcpyHostToDevice);
    clap_hipMemcpy(dB,    B,    szB,                   hipMemcpyHostToDevice);
    clap_hipMemcpy(d_ipiv,ipiv, (size_t)n*sizeof(int), hipMemcpyHostToDevice);

    clap_rocsolver_dgetrs(m_handle, to_roc_trans(trans),
                          n, nrhs, dA, lda, d_ipiv, dB, ldb);

    clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
    *info = 0;

    clap_hipFree(dA); clap_hipFree(dB); clap_hipFree(d_ipiv);
}

template<>
void RocSolverBackend::potrf_impl<float>(Uplo uplo, lapack_int n,
                                          float *A, lapack_int lda,
                                          lapack_int *info) {
    
    size_t szA = (size_t)lda * n * sizeof(float);
    float *dA = nullptr; int *d_info = nullptr;

    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&d_info, sizeof(int));
    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_rocsolver_spotrf(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

    clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
    int h_info = 0;
    clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
    *info = h_info;

    clap_hipFree(dA); clap_hipFree(d_info);
}

template<>
void RocSolverBackend::potrf_impl<double>(Uplo uplo, lapack_int n,
                                           double *A, lapack_int lda,
                                           lapack_int *info) {
    
    size_t szA = (size_t)lda * n * sizeof(double);
    double *dA = nullptr; int *d_info = nullptr;

    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&d_info, sizeof(int));
    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_rocsolver_dpotrf(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

    clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
    int h_info = 0;
    clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
    *info = h_info;

    clap_hipFree(dA); clap_hipFree(d_info);
}

template<>
void RocSolverBackend::potri_impl<float>(Uplo uplo, lapack_int n, 
                                         float *A, lapack_int lda,
                                         lapack_int *info) {
        
    size_t szA = (size_t)lda * n * sizeof(float);

    float *dA = nullptr;
    int *d_info = nullptr;

    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&d_info, sizeof(int));
    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_rocsolver_spotri(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

    clap_hipMemcpy(A,    dA,     szA,        hipMemcpyDeviceToHost);
    clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

    clap_hipFree(dA); 
    clap_hipFree(d_info);                          
}

template<>
void RocSolverBackend::potri_impl<double>(Uplo uplo, lapack_int n, 
                                         double *A, lapack_int lda,
                                         lapack_int *info) {
       
    size_t szA = (size_t)lda * n * sizeof(double);

    double *dA = nullptr; 
    int *d_info = nullptr;
    
    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&d_info, sizeof(int));
    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_rocsolver_dpotri(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

    clap_hipMemcpy(A,    dA,     szA,        hipMemcpyDeviceToHost);
    clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

    clap_hipFree(dA);
    clap_hipFree(d_info);                          
}


template<>
void RocSolverBackend::potrs_impl<float>(Uplo uplo, lapack_int n,
    									lapack_int nrhs, const float *A,
    									lapack_int lda, float *B,
  							lapack_int ldb, lapack_int *info) {
    
    size_t szA = (size_t)lda * n * sizeof(float);
    size_t szB = (size_t)ldb * nrhs * sizeof(float);

    float *dA = nullptr;
    float *dB = nullptr;

    clap_hipMalloc((void**)&dA, szA);
    clap_hipMalloc((void**)&dB, szB);

    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

    clap_rocsolver_spotrs(m_handle, to_roc_uplo(uplo), n, nrhs, dA, lda,
        						dB, ldb);

    clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);

    *info = 0;

    clap_hipFree(dA);
    clap_hipFree(dB);
}

template<>
void RocSolverBackend::potrs_impl<double>(Uplo uplo, lapack_int n,
    									lapack_int nrhs, const double *A,
    									lapack_int lda, double *B,
							lapack_int ldb, lapack_int *info) {
    

    size_t szA = (size_t)lda * n * sizeof(double);
    size_t szB = (size_t)ldb * nrhs * sizeof(double);

    double *dA = nullptr;
    double *dB = nullptr;

    clap_hipMalloc((void**)&dA, szA);
    clap_hipMalloc((void**)&dB, szB);

    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

    clap_rocsolver_dpotrs(m_handle, to_roc_uplo(uplo), n, nrhs,
        			dA, lda, dB, ldb);

    clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);

    *info = 0;

    clap_hipFree(dA);
    clap_hipFree(dB);
}

template<>
void RocSolverBackend::gesvd_impl<float>(Job jobu, Job jobvt,
                                          lapack_int m, lapack_int n,
                                          float *A, lapack_int lda, float *s,
                                          float *U, lapack_int ldu,
                                          float *VT, lapack_int ldvt,
                                          lapack_int *info) {
    
    int k = std::min(m, n);
    size_t szA = (size_t)lda * n * sizeof(float);
    size_t szU = (size_t)ldu * m * sizeof(float);
    size_t szV = (size_t)ldvt* n * sizeof(float);

    float *dA=nullptr, *dU=nullptr, *dVT=nullptr, *dS=nullptr, *dE=nullptr;
    int *d_info=nullptr;

    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&dU,     szU);
    clap_hipMalloc((void**)&dVT,    szV);
    clap_hipMalloc((void**)&dS,     (size_t)k * sizeof(float));
    clap_hipMalloc((void**)&dE,     (size_t)k * sizeof(float));
    clap_hipMalloc((void**)&d_info, sizeof(int));

    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_rocsolver_sgesvd(m_handle, to_roc_svect(jobu), to_roc_svect(jobvt), m, n,
                          dA, lda, dS, dU, ldu, dVT, ldvt,
                          dE, rocblas_outofplace_solver, d_info);

    clap_hipMemcpy(s,    dS,     (size_t)k*sizeof(float), hipMemcpyDeviceToHost);
    clap_hipMemcpy(U,    dU,     szU,                     hipMemcpyDeviceToHost);
    clap_hipMemcpy(VT,   dVT,    szV,                     hipMemcpyDeviceToHost);
    int h_info = 0;
    clap_hipMemcpy(&h_info, d_info, sizeof(int),           hipMemcpyDeviceToHost);
    *info = h_info;

    clap_hipFree(dA); clap_hipFree(dU); clap_hipFree(dVT);
    clap_hipFree(dS); clap_hipFree(dE); clap_hipFree(d_info);
}

template<>
void RocSolverBackend::gesvd_impl<double>(Job jobu, Job jobvt,
                                           lapack_int m, lapack_int n,
                                           double *A, lapack_int lda, double *s,
                                           double *U, lapack_int ldu,
                                           double *VT, lapack_int ldvt,
                                           lapack_int *info) {
    
    int k = std::min(m, n);
    size_t szA = (size_t)lda * n * sizeof(double);
    size_t szU = (size_t)ldu * m * sizeof(double);
    size_t szV = (size_t)ldvt* n * sizeof(double);

    double *dA=nullptr, *dU=nullptr, *dVT=nullptr, *dS=nullptr, *dE=nullptr;
    int *d_info=nullptr;

    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&dU,     szU);
    clap_hipMalloc((void**)&dVT,    szV);
    clap_hipMalloc((void**)&dS,     (size_t)k * sizeof(double));
    clap_hipMalloc((void**)&dE,     (size_t)k * sizeof(double));
    clap_hipMalloc((void**)&d_info, sizeof(int));

    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_rocsolver_dgesvd(m_handle, to_roc_svect(jobu), to_roc_svect(jobvt), m, n,
                          dA, lda, dS, dU, ldu, dVT, ldvt,
                          dE, rocblas_outofplace_solver, d_info);

    clap_hipMemcpy(s,    dS,     (size_t)k*sizeof(double), hipMemcpyDeviceToHost);
    clap_hipMemcpy(U,    dU,     szU,                      hipMemcpyDeviceToHost);
    clap_hipMemcpy(VT,   dVT,    szV,                      hipMemcpyDeviceToHost);
    int h_info = 0;
    clap_hipMemcpy(&h_info, d_info, sizeof(int),            hipMemcpyDeviceToHost);
    *info = h_info;

    clap_hipFree(dA); clap_hipFree(dU); clap_hipFree(dVT);
    clap_hipFree(dS); clap_hipFree(dE); clap_hipFree(d_info);
}

template<>
void RocSolverBackend::syev_impl<float>(Job jobz, Uplo uplo, lapack_int n,
                                         float *A, lapack_int lda, float *w,
                                         lapack_int *info) {
    
    size_t szA = (size_t)lda * n * sizeof(float);
    float *dA=nullptr, *dD=nullptr, *dE=nullptr; int *d_info=nullptr;

    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&dD,     (size_t)n * sizeof(float));
    clap_hipMalloc((void**)&dE,     (size_t)n * sizeof(float));
    clap_hipMalloc((void**)&d_info, sizeof(int));

    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_rocsolver_ssyev(m_handle,
                         to_roc_evect(jobz),
                         to_roc_uplo(uplo),
                         n, dA, lda, dD, dE, d_info);

    clap_hipMemcpy(A, dA, szA,                    hipMemcpyDeviceToHost);
    clap_hipMemcpy(w, dD, (size_t)n*sizeof(float), hipMemcpyDeviceToHost);
    int h_info = 0;
    clap_hipMemcpy(&h_info, d_info, sizeof(int),   hipMemcpyDeviceToHost);
    *info = h_info;

    clap_hipFree(dA); clap_hipFree(dD); clap_hipFree(dE); clap_hipFree(d_info);
}

template<>
void RocSolverBackend::syev_impl<double>(Job jobz, Uplo uplo, lapack_int n,
                                          double *A, lapack_int lda, double *w,
                                          lapack_int *info) {
    
    size_t szA = (size_t)lda * n * sizeof(double);
    double *dA=nullptr, *dD=nullptr, *dE=nullptr; int *d_info=nullptr;

    clap_hipMalloc((void**)&dA,     szA);
    clap_hipMalloc((void**)&dD,     (size_t)n * sizeof(double));
    clap_hipMalloc((void**)&dE,     (size_t)n * sizeof(double));
    clap_hipMalloc((void**)&d_info, sizeof(int));

    clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

    clap_rocsolver_dsyev(m_handle,
                         to_roc_evect(jobz),
                         to_roc_uplo(uplo),
                         n, dA, lda, dD, dE, d_info);

    clap_hipMemcpy(A, dA, szA,                     hipMemcpyDeviceToHost);
    clap_hipMemcpy(w, dD, (size_t)n*sizeof(double), hipMemcpyDeviceToHost);
    int h_info = 0;
    clap_hipMemcpy(&h_info, d_info, sizeof(int),    hipMemcpyDeviceToHost);
    *info = h_info;

    clap_hipFree(dA); clap_hipFree(dD); clap_hipFree(dE); clap_hipFree(d_info);
}

// ══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ══════════════════════════════════════════════════════════════════════════

void RocSolverBackend::sgetrf(Layout, lapack_int m, lapack_int n,
    float *A, lapack_int lda, lapack_int *ipiv, lapack_int *info) {
    getrf_impl<float>(m, n, A, lda, ipiv, info);
}
void RocSolverBackend::dgetrf(Layout, lapack_int m, lapack_int n,
    double *A, lapack_int lda, lapack_int *ipiv, lapack_int *info) {
    getrf_impl<double>(m, n, A, lda, ipiv, info);
}
void RocSolverBackend::sgetrs(Layout, Transpose t, lapack_int n, lapack_int nrhs,
    const float *A, lapack_int lda, const lapack_int *ipiv,
    float *B, lapack_int ldb, lapack_int *info) {
    getrs_impl<float>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::dgetrs(Layout, Transpose t, lapack_int n, lapack_int nrhs,
    const double *A, lapack_int lda, const lapack_int *ipiv,
    double *B, lapack_int ldb, lapack_int *info) {
    getrs_impl<double>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::sgesv(lapack_int n, lapack_int nrhs,
    float *A, lapack_int lda, lapack_int *ipiv,
    float *B, lapack_int ldb, lapack_int *info) {
    getrf_impl<float>(n, n, A, lda, ipiv, info);
    if (*info == 0)
        getrs_impl<float>(Transpose::NoTrans, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::dgesv(lapack_int n, lapack_int nrhs,
    double *A, lapack_int lda, lapack_int *ipiv,
    double *B, lapack_int ldb, lapack_int *info) {
    getrf_impl<double>(n, n, A, lda, ipiv, info);
    if (*info == 0)
        getrs_impl<double>(Transpose::NoTrans, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::sgetri(lapack_int, float*, lapack_int,
    const lapack_int*, lapack_int *info) {
    std::cerr << "[CLAP] rocSOLVER getri not supported\n"; *info=-1;
}
void RocSolverBackend::dgetri(lapack_int, double*, lapack_int,
    const lapack_int*, lapack_int *info) {
    std::cerr << "[CLAP] rocSOLVER getri not supported\n"; *info=-1;
}
void RocSolverBackend::spotrf(Layout, Uplo u, lapack_int n,
    float *A, lapack_int lda, lapack_int *info) {
    potrf_impl<float>(u, n, A, lda, info);
}
void RocSolverBackend::dpotrf(Layout, Uplo u, lapack_int n,
    double *A, lapack_int lda, lapack_int *info) {
    potrf_impl<double>(u, n, A, lda, info);
}
void RocSolverBackend::spotri(Layout, Uplo u, lapack_int n,
    float *A, lapack_int lda, lapack_int *info) {
    potri_impl<float>(u, n, A, lda, info);
}
void RocSolverBackend::dpotri(Layout, Uplo u, lapack_int n,
    double *A, lapack_int lda, lapack_int *info) {
    potri_impl<double>(u, n, A, lda, info);
}
void RocSolverBackend::spotrs(Layout, Uplo u, lapack_int n, lapack_int nrhs,
    const float* A, lapack_int lda, float* B, lapack_int ldb, lapack_int *info) { 
	potrs_impl<float>(u, n, nrhs, A, lda, B, ldb, info); 
}
void RocSolverBackend::dpotrs(Layout, Uplo u, lapack_int n, lapack_int nrhs,
    const double* A, lapack_int lda, double* B, lapack_int ldb, lapack_int *info) { 
	potrs_impl<double>(u, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::sposv(Layout l, Uplo u, lapack_int n, lapack_int nrhs,
    float *A, lapack_int lda, float *B, lapack_int ldb, lapack_int *info) {
    potrf_impl<float>(u, n, A, lda, info);
    if (*info == 0) spotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::dposv(Layout l, Uplo u, lapack_int n, lapack_int nrhs,
    double *A, lapack_int lda, double *B, lapack_int ldb, lapack_int *info) {
    potrf_impl<double>(u, n, A, lda, info);
    if (*info == 0) dpotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::sgeqrf(Layout, lapack_int, lapack_int,
    float*, lapack_int, float*, lapack_int *info) {
    std::cerr << "[CLAP] rocSOLVER geqrf not yet wired\n"; *info=-1;
}
void RocSolverBackend::dgeqrf(Layout, lapack_int, lapack_int,
    double*, lapack_int, double*, lapack_int *info) {
    std::cerr << "[CLAP] rocSOLVER geqrf not yet wired\n"; *info=-1;
}
void RocSolverBackend::sorgqr(Layout, lapack_int, lapack_int, lapack_int,
    float*, lapack_int, const float*, lapack_int *info) { *info=-1; }
void RocSolverBackend::dorgqr(Layout, lapack_int, lapack_int, lapack_int,
    double*, lapack_int, const double*, lapack_int *info) { *info=-1; }
void RocSolverBackend::sgels(Layout, Transpose, lapack_int, lapack_int, lapack_int,
    float*, lapack_int, float*, lapack_int, lapack_int *info) { *info=-1; }
void RocSolverBackend::dgels(Layout, Transpose, lapack_int, lapack_int, lapack_int,
    double*, lapack_int, double*, lapack_int, lapack_int *info) { *info=-1; }
void RocSolverBackend::ssyev(Layout, Job j, Uplo u, lapack_int n,
    float *A, lapack_int lda, float *w, lapack_int *info) {
    syev_impl<float>(j, u, n, A, lda, w, info);
}
void RocSolverBackend::dsyev(Layout, Job j, Uplo u, lapack_int n,
    double *A, lapack_int lda, double *w, lapack_int *info) {
    syev_impl<double>(j, u, n, A, lda, w, info);
}
void RocSolverBackend::sgeev(Layout, Job, Job, lapack_int,
    float*, lapack_int, float*, float*, float*, lapack_int, float*, lapack_int,
    lapack_int *info) {
    std::cerr << "[CLAP] rocSOLVER geev not supported\n"; *info=-1;
}
void RocSolverBackend::dgeev(Layout, Job, Job, lapack_int,
    double*, lapack_int, double*, double*, double*, lapack_int, double*, lapack_int,
    lapack_int *info) {
    std::cerr << "[CLAP] rocSOLVER geev not supported\n"; *info=-1;
}
void RocSolverBackend::sgesvd(Layout, Job ju, Job jv, lapack_int m, lapack_int n,
    float *A, lapack_int lda, float *s,
    float *U, lapack_int ldu, float *VT, lapack_int ldvt,
    float*, lapack_int *info) {
    gesvd_impl<float>(ju, jv, m, n, A, lda, s, U, ldu, VT, ldvt, info);
}
void RocSolverBackend::dgesvd(Layout, Job ju, Job jv, lapack_int m, lapack_int n,
    double *A, lapack_int lda, double *s,
    double *U, lapack_int ldu, double *VT, lapack_int ldvt,
    double*, lapack_int *info) {
    gesvd_impl<double>(ju, jv, m, n, A, lda, s, U, ldu, VT, ldvt, info);
}
void RocSolverBackend::strtrs(Layout, Uplo, Transpose, Diag,
    lapack_int, lapack_int, const float*, lapack_int, float*, lapack_int,
    lapack_int *info) {
    std::cerr << "[CLAP] rocSOLVER trtrs not supported\n"; *info=-1;
}
void RocSolverBackend::dtrtrs(Layout, Uplo, Transpose, Diag,
    lapack_int, lapack_int, const double*, lapack_int, double*, lapack_int,
    lapack_int *info) {
    std::cerr << "[CLAP] rocSOLVER trtrs not supported\n"; *info=-1;
}

} // namespace clap
