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
#include "clap/detail/checked_call.hpp"
#include "clap/dyn_backends.hpp"
#include "clap/lapack_dyn.hpp"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace clap {

using detail::checked_call;

static int to_cusolver_uplo(Uplo u) { return (u == Uplo::Upper) ? 1 : 0; }

static int to_cusolver_trans(Transpose t) {
  switch (t) {
  case Transpose::NoTrans:
    return 0;
  case Transpose::Trans:
    return 1;
  case Transpose::ConjTrans:
    return 2;
  }
  return 0;
}

static int to_cusolver_job(Job j) { return (j == Job::Vec) ? 1 : 0; }

CuSolverBackend::CuSolverBackend() : m_handle(nullptr) {
  std::cout << "Using cuSOLVER backend\n";

  if (!dyn_lapack::loadCuSolver())
    throw std::runtime_error("cuSOLVER load failed");

  if (!dyn::loadCudaAndCublas())
    throw std::runtime_error("CUDA runtime load failed for cuSOLVER");

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
    throw std::runtime_error(std::string(operation) +
                             " failed with CUDA status " +
                             std::to_string(static_cast<int>(status)));
}

static void check_cusolver(cusolverStatus_t status, const char *operation) {
  if (status != CUSOLVER_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) +
                             " failed with cuSOLVER status " +
                             std::to_string(static_cast<int>(status)));
}

const auto clap_cudaMalloc =
    checked_call(dyn::p_cudaMalloc, check_cuda, "cudaMalloc");
const auto clap_cudaFree =
    checked_call(dyn::p_cudaFree, check_cuda, "cudaFree");
const auto clap_cudaMemcpy =
    checked_call(dyn::p_cudaMemcpy, check_cuda, "cudaMemcpy");
const auto clap_cudaDeviceSynchronize = checked_call(
    dyn::p_cudaDeviceSynchronize, check_cuda, "cudaDeviceSynchronize");

const auto clap_cusolverDnSgetrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnSgetrf_bufferSize, check_cusolver,
                 "cusolverDnSgetrf_bufferSize");
const auto clap_cusolverDnDgetrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnDgetrf_bufferSize, check_cusolver,
                 "cusolverDnDgetrf_bufferSize");
const auto clap_cusolverDnCgetrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnCgetrf_bufferSize, check_cusolver,
                 "cusolverDnCgetrf_bufferSize");
const auto clap_cusolverDnZgetrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnZgetrf_bufferSize, check_cusolver,
                 "cusolverDnZgetrf_bufferSize");
const auto clap_cusolverDnSgetrf = checked_call(
    dyn_lapack::p_cusolverDnSgetrf, check_cusolver, "cusolverDnSgetrf");
const auto clap_cusolverDnDgetrf = checked_call(
    dyn_lapack::p_cusolverDnDgetrf, check_cusolver, "cusolverDnDgetrf");
const auto clap_cusolverDnCgetrf = checked_call(
    dyn_lapack::p_cusolverDnCgetrf, check_cusolver, "cusolverDnCgetrf");
const auto clap_cusolverDnZgetrf = checked_call(
    dyn_lapack::p_cusolverDnZgetrf, check_cusolver, "cusolverDnZgetrf");
const auto clap_cusolverDnSgetrs = checked_call(
    dyn_lapack::p_cusolverDnSgetrs, check_cusolver, "cusolverDnSgetrs");
const auto clap_cusolverDnDgetrs = checked_call(
    dyn_lapack::p_cusolverDnDgetrs, check_cusolver, "cusolverDnDgetrs");
const auto clap_cusolverDnCgetrs = checked_call(
    dyn_lapack::p_cusolverDnCgetrs, check_cusolver, "cusolverDnCgetrs");
const auto clap_cusolverDnZgetrs = checked_call(
    dyn_lapack::p_cusolverDnZgetrs, check_cusolver, "cusolverDnZgetrs");
const auto clap_cusolverDnSpotrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnSpotrf_bufferSize, check_cusolver,
                 "cusolverDnSpotrf_bufferSize");
const auto clap_cusolverDnDpotrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnDpotrf_bufferSize, check_cusolver,
                 "cusolverDnDpotrf_bufferSize");
const auto clap_cusolverDnCpotrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnCpotrf_bufferSize, check_cusolver,
                 "cusolverDnCpotrf_bufferSize");
const auto clap_cusolverDnZpotrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnZpotrf_bufferSize, check_cusolver,
                 "cusolverDnZpotrf_bufferSize");
const auto clap_cusolverDnSpotrf = checked_call(
    dyn_lapack::p_cusolverDnSpotrf, check_cusolver, "cusolverDnSpotrf");
const auto clap_cusolverDnDpotrf = checked_call(
    dyn_lapack::p_cusolverDnDpotrf, check_cusolver, "cusolverDnDpotrf");
const auto clap_cusolverDnCpotrf = checked_call(
    dyn_lapack::p_cusolverDnCpotrf, check_cusolver, "cusolverDnCpotrf");
const auto clap_cusolverDnZpotrf = checked_call(
    dyn_lapack::p_cusolverDnZpotrf, check_cusolver, "cusolverDnZpotrf");
const auto clap_cusolverDnSpotri_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnSpotri_bufferSize, check_cusolver,
                 "cusolverDnSpotri_bufferSize");
const auto clap_cusolverDnDpotri_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnDpotri_bufferSize, check_cusolver,
                 "cusolverDnDpotri_bufferSize");
const auto clap_cusolverDnCpotri_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnCpotri_bufferSize, check_cusolver,
                 "cusolverDnCpotri_bufferSize");
const auto clap_cusolverDnZpotri_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnZpotri_bufferSize, check_cusolver,
                 "cusolverDnZpotri_bufferSize");
const auto clap_cusolverDnSpotri = checked_call(
    dyn_lapack::p_cusolverDnSpotri, check_cusolver, "cusolverDnSpotri");
const auto clap_cusolverDnDpotri = checked_call(
    dyn_lapack::p_cusolverDnDpotri, check_cusolver, "cusolverDnDpotri");
const auto clap_cusolverDnCpotri = checked_call(
    dyn_lapack::p_cusolverDnCpotri, check_cusolver, "cusolverDnCpotri");
const auto clap_cusolverDnZpotri = checked_call(
    dyn_lapack::p_cusolverDnZpotri, check_cusolver, "cusolverDnZpotri");
const auto clap_cusolverDnSpotrs = checked_call(
    dyn_lapack::p_cusolverDnSpotrs, check_cusolver, "cusolverDnSpotrs");
const auto clap_cusolverDnDpotrs = checked_call(
    dyn_lapack::p_cusolverDnDpotrs, check_cusolver, "cusolverDnDpotrs");
const auto clap_cusolverDnCpotrs = checked_call(
    dyn_lapack::p_cusolverDnCpotrs, check_cusolver, "cusolverDnCpotrs");
const auto clap_cusolverDnZpotrs = checked_call(
    dyn_lapack::p_cusolverDnZpotrs, check_cusolver, "cusolverDnZpotrs");
const auto clap_cusolverDnSgesvd_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnSgesvd_bufferSize, check_cusolver,
                 "cusolverDnSgesvd_bufferSize");
const auto clap_cusolverDnDgesvd_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnDgesvd_bufferSize, check_cusolver,
                 "cusolverDnDgesvd_bufferSize");
const auto clap_cusolverDnCgesvd_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnCgesvd_bufferSize, check_cusolver,
                 "cusolverDnCgesvd_bufferSize");
const auto clap_cusolverDnZgesvd_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnZgesvd_bufferSize, check_cusolver,
                 "cusolverDnZgesvd_bufferSize");
const auto clap_cusolverDnSgesvd = checked_call(
    dyn_lapack::p_cusolverDnSgesvd, check_cusolver, "cusolverDnSgesvd");
const auto clap_cusolverDnDgesvd = checked_call(
    dyn_lapack::p_cusolverDnDgesvd, check_cusolver, "cusolverDnDgesvd");
const auto clap_cusolverDnCgesvd = checked_call(
    dyn_lapack::p_cusolverDnCgesvd, check_cusolver, "cusolverDnCgesvd");
const auto clap_cusolverDnZgesvd = checked_call(
    dyn_lapack::p_cusolverDnZgesvd, check_cusolver, "cusolverDnZgesvd");
const auto clap_cusolverDnSsyevd_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnSsyevd_bufferSize, check_cusolver,
                 "cusolverDnSsyevd_bufferSize");
const auto clap_cusolverDnDsyevd_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnDsyevd_bufferSize, check_cusolver,
                 "cusolverDnDsyevd_bufferSize");
const auto clap_cusolverDnCheevd_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnCheevd_bufferSize, check_cusolver,
                 "cusolverDnCheevd_bufferSize");
const auto clap_cusolverDnZheevd_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnZheevd_bufferSize, check_cusolver,
                 "cusolverDnZheevd_bufferSize");
const auto clap_cusolverDnSsyevd = checked_call(
    dyn_lapack::p_cusolverDnSsyevd, check_cusolver, "cusolverDnSsyevd");
const auto clap_cusolverDnDsyevd = checked_call(
    dyn_lapack::p_cusolverDnDsyevd, check_cusolver, "cusolverDnDsyevd");
const auto clap_cusolverDnCheevd = checked_call(
    dyn_lapack::p_cusolverDnCheevd, check_cusolver, "cusolverDnCheevd");
const auto clap_cusolverDnZheevd = checked_call(
    dyn_lapack::p_cusolverDnZheevd, check_cusolver, "cusolverDnZheevd");
const auto clap_cusolverDnSgeqrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnSgeqrf_bufferSize, check_cusolver,
                 "cusolverDnSgeqrf_bufferSize");
const auto clap_cusolverDnDgeqrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnDgeqrf_bufferSize, check_cusolver,
                 "cusolverDnDgeqrf_bufferSize");
const auto clap_cusolverDnCgeqrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnCgeqrf_bufferSize, check_cusolver,
                 "cusolverDnCgeqrf_bufferSize");
const auto clap_cusolverDnZgeqrf_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnZgeqrf_bufferSize, check_cusolver,
                 "cusolverDnZgeqrf_bufferSize");
const auto clap_cusolverDnSgeqrf = checked_call(
    dyn_lapack::p_cusolverDnSgeqrf, check_cusolver, "cusolverDnSgeqrf");
const auto clap_cusolverDnDgeqrf = checked_call(
    dyn_lapack::p_cusolverDnDgeqrf, check_cusolver, "cusolverDnDgeqrf");
const auto clap_cusolverDnCgeqrf = checked_call(
    dyn_lapack::p_cusolverDnCgeqrf, check_cusolver, "cusolverDnCgeqrf");
const auto clap_cusolverDnZgeqrf = checked_call(
    dyn_lapack::p_cusolverDnZgeqrf, check_cusolver, "cusolverDnZgeqrf");
const auto clap_cusolverDnSorgqr_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnSorgqr_bufferSize, check_cusolver,
                 "cusolverDnSorgqr_bufferSize");
const auto clap_cusolverDnDorgqr_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnDorgqr_bufferSize, check_cusolver,
                 "cusolverDnDorgqr_bufferSize");
const auto clap_cusolverDnCungqr_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnCungqr_bufferSize, check_cusolver,
                 "cusolverDnCungqr_bufferSize");
const auto clap_cusolverDnZungqr_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnZungqr_bufferSize, check_cusolver,
                 "cusolverDnZungqr_bufferSize");
const auto clap_cusolverDnSorgqr = checked_call(
    dyn_lapack::p_cusolverDnSorgqr, check_cusolver, "cusolverDnSorgqr");
const auto clap_cusolverDnDorgqr = checked_call(
    dyn_lapack::p_cusolverDnDorgqr, check_cusolver, "cusolverDnDorgqr");
const auto clap_cusolverDnCungqr = checked_call(
    dyn_lapack::p_cusolverDnCungqr, check_cusolver, "cusolverDnCungqr");
const auto clap_cusolverDnZungqr = checked_call(
    dyn_lapack::p_cusolverDnZungqr, check_cusolver, "cusolverDnZungqr");
const auto clap_cusolverDnSSgels_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnSSgels_bufferSize, check_cusolver,
                 "cusolverDnSSgels_bufferSize");
const auto clap_cusolverDnDDgels_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnDDgels_bufferSize, check_cusolver,
                 "cusolverDnDDgels_bufferSize");
const auto clap_cusolverDnCCgels_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnCCgels_bufferSize, check_cusolver,
                 "cusolverDnCCgels_bufferSize");
const auto clap_cusolverDnZZgels_bufferSize =
    checked_call(dyn_lapack::p_cusolverDnZZgels_bufferSize, check_cusolver,
                 "cusolverDnZZgels_bufferSize");
const auto clap_cusolverDnSSgels = checked_call(
    dyn_lapack::p_cusolverDnSSgels, check_cusolver, "cusolverDnSSgels");
const auto clap_cusolverDnDDgels = checked_call(
    dyn_lapack::p_cusolverDnDDgels, check_cusolver, "cusolverDnDDgels");
const auto clap_cusolverDnCCgels = checked_call(
    dyn_lapack::p_cusolverDnCCgels, check_cusolver, "cusolverDnCCgels");
const auto clap_cusolverDnZZgels = checked_call(
    dyn_lapack::p_cusolverDnZZgels, check_cusolver, "cusolverDnZZgels");

template <>
void CuSolverBackend::getrf_impl<float>(int m, int n, float *A, int lda,
                                        int *ipiv, int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);

  float *dA = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;
  float *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)std::min(m, n) * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnSgetrf_bufferSize(m_handle, m, n, dA, lda, &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(float));

  clap_cusolverDnSgetrf(m_handle, m, n, dA, lda, d_work, d_ipiv, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(ipiv, d_ipiv, (size_t)std::min(m, n) * sizeof(int),
                  cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::getrf_impl<double>(int m, int n, double *A, int lda,
                                         int *ipiv, int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);

  double *dA = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;
  double *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)std::min(m, n) * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnDgetrf_bufferSize(m_handle, m, n, dA, lda, &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double));

  clap_cusolverDnDgetrf(m_handle, m, n, dA, lda, d_work, d_ipiv, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(ipiv, d_ipiv, (size_t)std::min(m, n) * sizeof(int),
                  cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::getrf_impl<cuComplex>(int m, int n, cuComplex *A, int lda,
                                            int *ipiv, int *info) {

  size_t szA = (size_t)lda * n * sizeof(cuComplex);

  cuComplex *dA = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;
  cuComplex *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)std::min(m, n) * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnCgetrf_bufferSize(m_handle, m, n, dA, lda, &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(cuComplex));

  clap_cusolverDnCgetrf(m_handle, m, n, dA, lda, d_work, d_ipiv, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(ipiv, d_ipiv, (size_t)std::min(m, n) * sizeof(int),
                  cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::getrf_impl<cuDoubleComplex>(int m, int n,
                                                  cuDoubleComplex *A, int lda,
                                                  int *ipiv, int *info) {

  size_t szA = (size_t)lda * n * sizeof(cuDoubleComplex);

  cuDoubleComplex *dA = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;
  cuDoubleComplex *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)std::min(m, n) * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnZgetrf_bufferSize(m_handle, m, n, dA, lda, &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(cuDoubleComplex));

  clap_cusolverDnZgetrf(m_handle, m, n, dA, lda, d_work, d_ipiv, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(ipiv, d_ipiv, (size_t)std::min(m, n) * sizeof(int),
                  cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::getrs_impl<float>(Transpose trans, int n, int nrhs,
                                        const float *A, int lda,
                                        const int *ipiv, float *B, int ldb,
                                        int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szB = (size_t)ldb * nrhs * sizeof(float);

  float *dA = nullptr;
  int *d_ipiv = nullptr;
  float *dB = nullptr;
  int *d_info = nullptr;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int),
                  cudaMemcpyHostToDevice);

  clap_cusolverDnSgetrs(m_handle, to_cusolver_trans(trans), n, nrhs, dA, lda,
                        d_ipiv, dB, ldb, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::getrs_impl<double>(Transpose trans, int n, int nrhs,
                                         const double *A, int lda,
                                         const int *ipiv, double *B, int ldb,
                                         int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szB = (size_t)ldb * nrhs * sizeof(double);

  double *dA = nullptr;
  int *d_ipiv = nullptr;
  double *dB = nullptr;
  int *d_info = nullptr;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int),
                  cudaMemcpyHostToDevice);

  clap_cusolverDnDgetrs(m_handle, to_cusolver_trans(trans), n, nrhs, dA, lda,
                        d_ipiv, dB, ldb, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::getrs_impl<cuComplex>(Transpose trans, int n, int nrhs,
                                            const cuComplex *A, int lda,
                                            const int *ipiv, cuComplex *B,
                                            int ldb, int *info) {

  size_t szA = (size_t)lda * n * sizeof(cuComplex);
  size_t szB = (size_t)ldb * nrhs * sizeof(cuComplex);

  cuComplex *dA = nullptr;
  int *d_ipiv = nullptr;
  cuComplex *dB = nullptr;
  int *d_info = nullptr;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int),
                  cudaMemcpyHostToDevice);

  clap_cusolverDnCgetrs(m_handle, to_cusolver_trans(trans), n, nrhs, dA, lda,
                        d_ipiv, dB, ldb, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::getrs_impl<cuDoubleComplex>(
    Transpose trans, int n, int nrhs, const cuDoubleComplex *A, int lda,
    const int *ipiv, cuDoubleComplex *B, int ldb, int *info) {

  size_t szA = (size_t)lda * n * sizeof(cuDoubleComplex);
  size_t szB = (size_t)ldb * nrhs * sizeof(cuDoubleComplex);

  cuDoubleComplex *dA = nullptr;
  int *d_ipiv = nullptr;
  cuDoubleComplex *dB = nullptr;
  int *d_info = nullptr;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int),
                  cudaMemcpyHostToDevice);

  clap_cusolverDnZgetrs(m_handle, to_cusolver_trans(trans), n, nrhs, dA, lda,
                        d_ipiv, dB, ldb, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::gesv_impl<float>(Transpose trans, int n, int nrhs,
                                       float *A, int lda, int *ipiv, float *B,
                                       int ldb, int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szB = (size_t)ldb * nrhs * sizeof(float);

  float *dA = nullptr;
  float *dB = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;
  float *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnSgetrf_bufferSize(m_handle, n, n, dA, lda, &lwork);

  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(float));

  clap_cusolverDnSgetrf(m_handle, n, n, dA, lda, d_work, d_ipiv, d_info);

  clap_cudaDeviceSynchronize();

  clap_cusolverDnSgetrs(m_handle, to_cusolver_trans(trans), n, nrhs, dA, lda,
                        d_ipiv, dB, ldb, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(ipiv, d_ipiv, (size_t)n * sizeof(int),
                  cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::gesv_impl<double>(Transpose trans, int n, int nrhs,
                                        double *A, int lda, int *ipiv,
                                        double *B, int ldb, int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szB = (size_t)ldb * nrhs * sizeof(double);

  double *dA = nullptr;
  double *dB = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;
  double *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnDgetrf_bufferSize(m_handle, n, n, dA, lda, &lwork);

  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double));

  clap_cusolverDnDgetrf(m_handle, n, n, dA, lda, d_work, d_ipiv, d_info);

  clap_cudaDeviceSynchronize();

  clap_cusolverDnDgetrs(m_handle, to_cusolver_trans(trans), n, nrhs, dA, lda,
                        d_ipiv, dB, ldb, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(ipiv, d_ipiv, (size_t)n * sizeof(int),
                  cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::gesv_impl<cuComplex>(Transpose trans, int n, int nrhs,
                                           cuComplex *A, int lda, int *ipiv,
                                           cuComplex *B, int ldb, int *info) {

  size_t szA = (size_t)lda * n * sizeof(cuComplex);
  size_t szB = (size_t)ldb * nrhs * sizeof(cuComplex);

  cuComplex *dA = nullptr;
  cuComplex *dB = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;
  cuComplex *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnCgetrf_bufferSize(m_handle, n, n, dA, lda, &lwork);

  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(cuComplex));

  clap_cusolverDnCgetrf(m_handle, n, n, dA, lda, d_work, d_ipiv, d_info);

  clap_cudaDeviceSynchronize();

  clap_cusolverDnCgetrs(m_handle, to_cusolver_trans(trans), n, nrhs, dA, lda,
                        d_ipiv, dB, ldb, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(ipiv, d_ipiv, (size_t)n * sizeof(int),
                  cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::gesv_impl<cuDoubleComplex>(Transpose trans, int n,
                                                 int nrhs, cuDoubleComplex *A,
                                                 int lda, int *ipiv,
                                                 cuDoubleComplex *B, int ldb,
                                                 int *info) {

  size_t szA = (size_t)lda * n * sizeof(cuDoubleComplex);
  size_t szB = (size_t)ldb * nrhs * sizeof(cuDoubleComplex);

  cuDoubleComplex *dA = nullptr;
  cuDoubleComplex *dB = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;
  cuDoubleComplex *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnZgetrf_bufferSize(m_handle, n, n, dA, lda, &lwork);

  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(cuDoubleComplex));

  clap_cusolverDnZgetrf(m_handle, n, n, dA, lda, d_work, d_ipiv, d_info);

  clap_cudaDeviceSynchronize();

  clap_cusolverDnZgetrs(m_handle, to_cusolver_trans(trans), n, nrhs, dA, lda,
                        d_ipiv, dB, ldb, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(ipiv, d_ipiv, (size_t)n * sizeof(int),
                  cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_ipiv);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::potrf_impl<float>(Uplo uplo, int n, float *A, int lda,
                                        int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);

  float *dA = nullptr;
  int *d_info = nullptr;
  float *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_info, sizeof(int));
  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnSpotrf_bufferSize(m_handle, to_cusolver_uplo(uplo), n, dA, lda,
                                   &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(float));

  clap_cusolverDnSpotrf(m_handle, to_cusolver_uplo(uplo), n, dA, lda, d_work,
                        lwork, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::potrf_impl<double>(Uplo uplo, int n, double *A, int lda,
                                         int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);

  double *dA = nullptr;
  int *d_info = nullptr;
  double *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_info, sizeof(int));
  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnDpotrf_bufferSize(m_handle, to_cusolver_uplo(uplo), n, dA, lda,
                                   &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double));

  clap_cusolverDnDpotrf(m_handle, to_cusolver_uplo(uplo), n, dA, lda, d_work,
                        lwork, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::potrf_impl<cuComplex>(Uplo uplo, int n, cuComplex *A,
                                            int lda, int *info) {
  size_t szA = (size_t)lda * n * sizeof(cuComplex);
  cuComplex *dA = nullptr;
  int *d_info = nullptr;
  cuComplex *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_info, sizeof(int));
  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnCpotrf_bufferSize(m_handle, to_cusolver_uplo(uplo), n, dA, lda,
                                   &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(cuComplex));

  clap_cusolverDnCpotrf(m_handle, to_cusolver_uplo(uplo), n, dA, lda, d_work,
                        lwork, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::potrf_impl<cuDoubleComplex>(Uplo uplo, int n,
                                                  cuDoubleComplex *A, int lda,
                                                  int *info) {
  size_t szA = (size_t)lda * n * sizeof(cuDoubleComplex);
  cuDoubleComplex *dA = nullptr;
  int *d_info = nullptr;
  cuDoubleComplex *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_info, sizeof(int));
  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnZpotrf_bufferSize(m_handle, to_cusolver_uplo(uplo), n, dA, lda,
                                   &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(cuDoubleComplex));

  clap_cusolverDnZpotrf(m_handle, to_cusolver_uplo(uplo), n, dA, lda, d_work,
                        lwork, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::potri_impl<float>(Uplo uplo, int n, float *A, int lda,
                                        int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);

  float *dA = nullptr;
  int *d_info = nullptr;
  float *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_info, sizeof(int));
  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnSpotri_bufferSize(m_handle, to_cusolver_uplo(uplo), n, dA, lda,
                                   &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(float));

  clap_cusolverDnSpotri(m_handle, to_cusolver_uplo(uplo), n, dA, lda, d_work,
                        lwork, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::potri_impl<double>(Uplo uplo, int n, double *A, int lda,
                                         int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);

  double *dA = nullptr;
  int *d_info = nullptr;
  double *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_info, sizeof(int));
  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnDpotri_bufferSize(m_handle, to_cusolver_uplo(uplo), n, dA, lda,
                                   &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double));

  clap_cusolverDnDpotri(m_handle, to_cusolver_uplo(uplo), n, dA, lda, d_work,
                        lwork, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::potri_impl<cuComplex>(Uplo uplo, int n, cuComplex *A,
                                            int lda, int *info) {

  size_t szA = (size_t)lda * n * sizeof(cuComplex);

  cuComplex *dA = nullptr;
  int *d_info = nullptr;
  cuComplex *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_info, sizeof(int));
  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnCpotri_bufferSize(m_handle, to_cusolver_uplo(uplo), n, dA, lda,
                                   &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(cuComplex));

  clap_cusolverDnCpotri(m_handle, to_cusolver_uplo(uplo), n, dA, lda, d_work,
                        lwork, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::potri_impl<cuDoubleComplex>(Uplo uplo, int n,
                                                  cuDoubleComplex *A, int lda,
                                                  int *info) {

  size_t szA = (size_t)lda * n * sizeof(cuDoubleComplex);

  cuDoubleComplex *dA = nullptr;
  int *d_info = nullptr;
  cuDoubleComplex *d_work = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_info, sizeof(int));
  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnZpotri_bufferSize(m_handle, to_cusolver_uplo(uplo), n, dA, lda,
                                   &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(cuDoubleComplex));

  clap_cusolverDnZpotri(m_handle, to_cusolver_uplo(uplo), n, dA, lda, d_work,
                        lwork, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_info);
  clap_cudaFree(d_work);
}

template <>
void CuSolverBackend::potrs_impl<float>(Uplo uplo, int n, int nrhs,
                                        const float *A, int lda, float *B,
                                        int ldb, int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szB = (size_t)ldb * nrhs * sizeof(float);

  float *dA = nullptr;
  float *dB = nullptr;
  int *d_info = nullptr;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnSpotrs(m_handle, to_cusolver_uplo(uplo), n, nrhs, dA, lda, dB,
                        ldb, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);

  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::potrs_impl<double>(Uplo uplo, int n, int nrhs,
                                         const double *A, int lda, double *B,
                                         int ldb, int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szB = (size_t)ldb * nrhs * sizeof(double);

  double *dA = nullptr;
  double *dB = nullptr;
  int *d_info = nullptr;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnDpotrs(m_handle, to_cusolver_uplo(uplo), n, nrhs, dA, lda, dB,
                        ldb, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);

  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::potrs_impl<cuComplex>(Uplo uplo, int n, int nrhs,
                                            const cuComplex *A, int lda,
                                            cuComplex *B, int ldb, int *info) {

  size_t szA = (size_t)lda * n * sizeof(cuComplex);
  size_t szB = (size_t)ldb * nrhs * sizeof(cuComplex);

  cuComplex *dA = nullptr;
  cuComplex *dB = nullptr;
  int *d_info = nullptr;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnCpotrs(m_handle, to_cusolver_uplo(uplo), n, nrhs, dA, lda, dB,
                        ldb, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);

  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::potrs_impl<cuDoubleComplex>(Uplo uplo, int n, int nrhs,
                                                  const cuDoubleComplex *A,
                                                  int lda, cuDoubleComplex *B,
                                                  int ldb, int *info) {

  size_t szA = (size_t)lda * n * sizeof(cuDoubleComplex);
  size_t szB = (size_t)ldb * nrhs * sizeof(cuDoubleComplex);

  cuDoubleComplex *dA = nullptr;
  cuDoubleComplex *dB = nullptr;
  int *d_info = nullptr;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnZpotrs(m_handle, to_cusolver_uplo(uplo), n, nrhs, dA, lda, dB,
                        ldb, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);

  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::posv_impl<float>(Uplo uplo, int n, int nrhs, float *A,
                                       int lda, float *B, int ldb, int *info) {

  potrf_impl<float>(uplo, n, A, lda, info);

  if (*info != 0)
    return;

  potrs_impl<float>(uplo, n, nrhs, A, lda, B, ldb, info);
}

template <>
void CuSolverBackend::posv_impl<double>(Uplo uplo, int n, int nrhs, double *A,
                                        int lda, double *B, int ldb,
                                        int *info) {

  potrf_impl<double>(uplo, n, A, lda, info);

  if (*info != 0)
    return;

  potrs_impl<double>(uplo, n, nrhs, A, lda, B, ldb, info);
}

template <>
void CuSolverBackend::posv_impl<cuComplex>(Uplo uplo, int n, int nrhs,
                                           cuComplex *A, int lda, cuComplex *B,
                                           int ldb, int *info) {

  potrf_impl<cuComplex>(uplo, n, A, lda, info);

  if (*info != 0)
    return;

  potrs_impl<cuComplex>(uplo, n, nrhs, A, lda, B, ldb, info);
}

template <>
void CuSolverBackend::posv_impl<cuDoubleComplex>(Uplo uplo, int n, int nrhs,
                                                 cuDoubleComplex *A, int lda,
                                                 cuDoubleComplex *B, int ldb,
                                                 int *info) {

  potrf_impl<cuDoubleComplex>(uplo, n, A, lda, info);

  if (*info != 0)
    return;

  potrs_impl<cuDoubleComplex>(uplo, n, nrhs, A, lda, B, ldb, info);
}

template <>
void CuSolverBackend::gesvd_impl<float, float>(Job jobu, Job jobvt, int m,
                                               int n, float *A, int lda,
                                               float *s, float *U, int ldu,
                                               float *VT, int ldvt, int *info) {

  int k = std::min(m, n);
  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szU = (size_t)ldu * m * sizeof(float);
  size_t szV = (size_t)ldvt * n * sizeof(float);

  float *dA = nullptr, *dU = nullptr, *dVT = nullptr, *dS = nullptr;
  float *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dU, szU);
  clap_cudaMalloc((void **)&dVT, szV);
  clap_cudaMalloc((void **)&dS, (size_t)k * sizeof(float));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnSgesvd_bufferSize(m_handle, m, n, &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(float));

  char jobu_c = static_cast<char>(jobu);
  char jobvt_c = static_cast<char>(jobvt);

  clap_cusolverDnSgesvd(m_handle, jobu_c, jobvt_c, m, n, dA, lda, dS, dU, ldu,
                        dVT, ldvt, d_work, lwork, nullptr, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(s, dS, (size_t)k * sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(U, dU, szU, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(VT, dVT, szV, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dU);
  clap_cudaFree(dVT);
  clap_cudaFree(dS);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::gesvd_impl<double, double>(Job jobu, Job jobvt, int m,
                                                 int n, double *A, int lda,
                                                 double *s, double *U, int ldu,
                                                 double *VT, int ldvt,
                                                 int *info) {

  int k = std::min(m, n);
  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szU = (size_t)ldu * m * sizeof(double);
  size_t szV = (size_t)ldvt * n * sizeof(double);

  double *dA = nullptr, *dU = nullptr, *dVT = nullptr, *dS = nullptr;
  double *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dU, szU);
  clap_cudaMalloc((void **)&dVT, szV);
  clap_cudaMalloc((void **)&dS, (size_t)k * sizeof(double));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnDgesvd_bufferSize(m_handle, m, n, &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double));

  char jobu_c = static_cast<char>(jobu);
  char jobvt_c = static_cast<char>(jobvt);

  clap_cusolverDnDgesvd(m_handle, jobu_c, jobvt_c, m, n, dA, lda, dS, dU, ldu,
                        dVT, ldvt, d_work, lwork, nullptr, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(s, dS, (size_t)k * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(U, dU, szU, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(VT, dVT, szV, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dU);
  clap_cudaFree(dVT);
  clap_cudaFree(dS);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::gesvd_impl<cuComplex, float>(Job jobu, Job jobvt, int m,
                                                   int n, cuComplex *A, int lda,
                                                   float *s, cuComplex *U,
                                                   int ldu, cuComplex *VT,
                                                   int ldvt, int *info) {

  int k = std::min(m, n);
  size_t szA = (size_t)lda * n * sizeof(cuComplex);
  size_t szU = (size_t)ldu * m * sizeof(cuComplex);
  size_t szV = (size_t)ldvt * n * sizeof(cuComplex);

  cuComplex *dA = nullptr, *dU = nullptr, *dVT = nullptr;
  float *dS = nullptr;
  cuComplex *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dU, szU);
  clap_cudaMalloc((void **)&dVT, szV);
  clap_cudaMalloc((void **)&dS, (size_t)k * sizeof(float));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnCgesvd_bufferSize(m_handle, m, n, &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(cuComplex));

  char jobu_c = static_cast<char>(jobu);
  char jobvt_c = static_cast<char>(jobvt);

  clap_cusolverDnCgesvd(m_handle, jobu_c, jobvt_c, m, n, dA, lda, dS, dU, ldu,
                        dVT, ldvt, d_work, lwork, nullptr, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(s, dS, (size_t)k * sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(U, dU, szU, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(VT, dVT, szV, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dU);
  clap_cudaFree(dVT);
  clap_cudaFree(dS);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::gesvd_impl<cuDoubleComplex, double>(
    Job jobu, Job jobvt, int m, int n, cuDoubleComplex *A, int lda, double *s,
    cuDoubleComplex *U, int ldu, cuDoubleComplex *VT, int ldvt, int *info) {

  int k = std::min(m, n);
  size_t szA = (size_t)lda * n * sizeof(cuDoubleComplex);
  size_t szU = (size_t)ldu * m * sizeof(cuDoubleComplex);
  size_t szV = (size_t)ldvt * n * sizeof(cuDoubleComplex);

  cuDoubleComplex *dA = nullptr, *dU = nullptr, *dVT = nullptr;
  double *dS = nullptr;
  cuDoubleComplex *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dU, szU);
  clap_cudaMalloc((void **)&dVT, szV);
  clap_cudaMalloc((void **)&dS, (size_t)k * sizeof(double));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnZgesvd_bufferSize(m_handle, m, n, &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(cuDoubleComplex));

  char jobu_c = static_cast<char>(jobu);
  char jobvt_c = static_cast<char>(jobvt);

  clap_cusolverDnZgesvd(m_handle, jobu_c, jobvt_c, m, n, dA, lda, dS, dU, ldu,
                        dVT, ldvt, d_work, lwork, nullptr, d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(s, dS, (size_t)k * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(U, dU, szU, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(VT, dVT, szV, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dU);
  clap_cudaFree(dVT);
  clap_cudaFree(dS);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::geqrf_impl<float>(int m, int n, float *A, int lda,
                                        float *tau, int *info) {

  if (m < 0) {
    *info = -1;
    return;
  }
  if (n < 0) {
    *info = -2;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -4;
    return;
  }
  if (m == 0 || n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szTau = (size_t)std::min(m, n) * sizeof(float);

  float *dA = nullptr;
  float *d_tau = nullptr;
  float *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_tau, szTau);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnSgeqrf_bufferSize(m_handle, m, n, dA, lda, &lwork);

  clap_cudaMalloc((void **)&d_work, (size_t)std::max(1, lwork) * sizeof(float));

  clap_cusolverDnSgeqrf(m_handle, m, n, dA, lda, d_tau, d_work, lwork, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(tau, d_tau, szTau, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_tau);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::geqrf_impl<double>(int m, int n, double *A, int lda,
                                         double *tau, int *info) {

  if (m < 0) {
    *info = -1;
    return;
  }
  if (n < 0) {
    *info = -2;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -4;
    return;
  }
  if (m == 0 || n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szTau = (size_t)std::min(m, n) * sizeof(double);

  double *dA = nullptr;
  double *d_tau = nullptr;
  double *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_tau, szTau);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnDgeqrf_bufferSize(m_handle, m, n, dA, lda, &lwork);

  clap_cudaMalloc((void **)&d_work,
                  (size_t)std::max(1, lwork) * sizeof(double));

  clap_cusolverDnDgeqrf(m_handle, m, n, dA, lda, d_tau, d_work, lwork, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(tau, d_tau, szTau, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_tau);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::geqrf_impl<cuComplex>(int m, int n, cuComplex *A, int lda,
                                            cuComplex *tau, int *info) {

  if (m < 0) {
    *info = -1;
    return;
  }
  if (n < 0) {
    *info = -2;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -4;
    return;
  }
  if (m == 0 || n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(cuComplex);
  size_t szTau = (size_t)std::min(m, n) * sizeof(cuComplex);

  cuComplex *dA = nullptr;
  cuComplex *d_tau = nullptr;
  cuComplex *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_tau, szTau);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnCgeqrf_bufferSize(m_handle, m, n, dA, lda, &lwork);

  clap_cudaMalloc((void **)&d_work,
                  (size_t)std::max(1, lwork) * sizeof(cuComplex));

  clap_cusolverDnCgeqrf(m_handle, m, n, dA, lda, d_tau, d_work, lwork, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(tau, d_tau, szTau, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_tau);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::geqrf_impl<cuDoubleComplex>(int m, int n,
                                                  cuDoubleComplex *A, int lda,
                                                  cuDoubleComplex *tau,
                                                  int *info) {

  if (m < 0) {
    *info = -1;
    return;
  }
  if (n < 0) {
    *info = -2;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -4;
    return;
  }
  if (m == 0 || n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(cuDoubleComplex);
  size_t szTau = (size_t)std::min(m, n) * sizeof(cuDoubleComplex);

  cuDoubleComplex *dA = nullptr;
  cuDoubleComplex *d_tau = nullptr;
  cuDoubleComplex *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_tau, szTau);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnZgeqrf_bufferSize(m_handle, m, n, dA, lda, &lwork);

  clap_cudaMalloc((void **)&d_work,
                  (size_t)std::max(1, lwork) * sizeof(cuDoubleComplex));

  clap_cusolverDnZgeqrf(m_handle, m, n, dA, lda, d_tau, d_work, lwork, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(tau, d_tau, szTau, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_tau);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::orgqr_impl<float>(int m, int n, int k, float *A, int lda,
                                        const float *tau, int *info) {

  if (m < 0) {
    *info = -1;
    return;
  }
  if (n < 0 || n > m) {
    *info = -2;
    return;
  }
  if (k < 0 || k > n) {
    *info = -3;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -5;
    return;
  }
  if (n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szTau = (size_t)k * sizeof(float);

  float *dA = nullptr;
  float *d_tau = nullptr;
  float *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_tau, (size_t)std::max(1, k) * sizeof(float));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  if (k > 0)
    clap_cudaMemcpy(d_tau, tau, szTau, cudaMemcpyHostToDevice);

  clap_cusolverDnSorgqr_bufferSize(m_handle, m, n, k, dA, lda, d_tau, &lwork);

  clap_cudaMalloc((void **)&d_work, (size_t)std::max(1, lwork) * sizeof(float));

  clap_cusolverDnSorgqr(m_handle, m, n, k, dA, lda, d_tau, d_work, lwork,
                        d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_tau);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::orgqr_impl<double>(int m, int n, int k, double *A,
                                         int lda, const double *tau,
                                         int *info) {

  if (m < 0) {
    *info = -1;
    return;
  }
  if (n < 0 || n > m) {
    *info = -2;
    return;
  }
  if (k < 0 || k > n) {
    *info = -3;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -5;
    return;
  }
  if (n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szTau = (size_t)k * sizeof(double);

  double *dA = nullptr;
  double *d_tau = nullptr;
  double *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_tau, (size_t)std::max(1, k) * sizeof(double));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  if (k > 0)
    clap_cudaMemcpy(d_tau, tau, szTau, cudaMemcpyHostToDevice);

  clap_cusolverDnDorgqr_bufferSize(m_handle, m, n, k, dA, lda, d_tau, &lwork);

  clap_cudaMalloc((void **)&d_work,
                  (size_t)std::max(1, lwork) * sizeof(double));

  clap_cusolverDnDorgqr(m_handle, m, n, k, dA, lda, d_tau, d_work, lwork,
                        d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_tau);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::ungqr_impl<cuComplex>(int m, int n, int k, cuComplex *A,
                                            int lda, const cuComplex *tau,
                                            int *info) {

  if (m < 0) {
    *info = -1;
    return;
  }
  if (n < 0 || n > m) {
    *info = -2;
    return;
  }
  if (k < 0 || k > n) {
    *info = -3;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -5;
    return;
  }
  if (n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(cuComplex);
  size_t szTau = (size_t)k * sizeof(cuComplex);

  cuComplex *dA = nullptr;
  cuComplex *d_tau = nullptr;
  cuComplex *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_tau, (size_t)std::max(1, k) * sizeof(cuComplex));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  if (k > 0)
    clap_cudaMemcpy(d_tau, tau, szTau, cudaMemcpyHostToDevice);

  clap_cusolverDnCungqr_bufferSize(m_handle, m, n, k, dA, lda, d_tau, &lwork);

  clap_cudaMalloc((void **)&d_work,
                  (size_t)std::max(1, lwork) * sizeof(cuComplex));

  clap_cusolverDnCungqr(m_handle, m, n, k, dA, lda, d_tau, d_work, lwork,
                        d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_tau);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::ungqr_impl<cuDoubleComplex>(int m, int n, int k,
                                                  cuDoubleComplex *A, int lda,
                                                  const cuDoubleComplex *tau,
                                                  int *info) {

  if (m < 0) {
    *info = -1;
    return;
  }
  if (n < 0 || n > m) {
    *info = -2;
    return;
  }
  if (k < 0 || k > n) {
    *info = -3;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -5;
    return;
  }
  if (n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(cuDoubleComplex);
  size_t szTau = (size_t)k * sizeof(cuDoubleComplex);

  cuDoubleComplex *dA = nullptr;
  cuDoubleComplex *d_tau = nullptr;
  cuDoubleComplex *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&d_tau,
                  (size_t)std::max(1, k) * sizeof(cuDoubleComplex));
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  if (k > 0)
    clap_cudaMemcpy(d_tau, tau, szTau, cudaMemcpyHostToDevice);

  clap_cusolverDnZungqr_bufferSize(m_handle, m, n, k, dA, lda, d_tau, &lwork);

  clap_cudaMalloc((void **)&d_work,
                  (size_t)std::max(1, lwork) * sizeof(cuDoubleComplex));

  clap_cusolverDnZungqr(m_handle, m, n, k, dA, lda, d_tau, d_work, lwork,
                        d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(d_tau);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::gels_impl<float>(Transpose trans, int m, int n, int nrhs,
                                       float *A, int lda, float *B, int ldb,
                                       float *X, int ldx, int *info) {

  if (trans != Transpose::NoTrans) {
    *info = -1;
    return;
  }
  if (m < 0 || m < n) {
    *info = -2;
    return;
  }
  if (n < 0) {
    *info = -3;
    return;
  }
  if (nrhs < 0) {
    *info = -4;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -6;
    return;
  }
  if (ldb < std::max(1, m)) {
    *info = -8;
    return;
  }
  if (ldx < std::max(1, n)) {
    *info = -10;
    return;
  }
  if (m == 0 || n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szB = (size_t)ldb * nrhs * sizeof(float);
  size_t szX = (size_t)ldx * nrhs * sizeof(float);

  float *dA = nullptr;
  float *dB = nullptr;
  float *dX = nullptr;
  void *d_work = nullptr;
  int *d_info = nullptr;
  int niter = 0;
  size_t lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&dX, szX);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnSSgels_bufferSize(m_handle, m, n, nrhs, dA, lda, dB, ldb, dX,
                                   ldx, nullptr, &lwork);

  clap_cudaMalloc(&d_work, std::max((size_t)1, lwork));

  clap_cusolverDnSSgels(m_handle, m, n, nrhs, dA, lda, dB, ldb, dX, ldx, d_work,
                        lwork, &niter, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  if (*info == 0) {
    clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(X, dX, szX, cudaMemcpyDeviceToHost);
  }

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dX);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::gels_impl<double>(Transpose trans, int m, int n, int nrhs,
                                        double *A, int lda, double *B, int ldb,
                                        double *X, int ldx, int *info) {

  if (trans != Transpose::NoTrans) {
    *info = -1;
    return;
  }
  if (m < 0 || m < n) {
    *info = -2;
    return;
  }
  if (n < 0) {
    *info = -3;
    return;
  }
  if (nrhs < 0) {
    *info = -4;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -6;
    return;
  }
  if (ldb < std::max(1, m)) {
    *info = -8;
    return;
  }
  if (ldx < std::max(1, n)) {
    *info = -10;
    return;
  }
  if (m == 0 || n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szB = (size_t)ldb * nrhs * sizeof(double);
  size_t szX = (size_t)ldx * nrhs * sizeof(double);

  double *dA = nullptr;
  double *dB = nullptr;
  double *dX = nullptr;
  void *d_work = nullptr;
  int *d_info = nullptr;
  int niter = 0;
  size_t lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&dX, szX);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnDDgels_bufferSize(m_handle, m, n, nrhs, dA, lda, dB, ldb, dX,
                                   ldx, nullptr, &lwork);

  clap_cudaMalloc(&d_work, std::max((size_t)1, lwork));

  clap_cusolverDnDDgels(m_handle, m, n, nrhs, dA, lda, dB, ldb, dX, ldx, d_work,
                        lwork, &niter, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  if (*info == 0) {
    clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(X, dX, szX, cudaMemcpyDeviceToHost);
  }

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dX);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::gels_impl<cuComplex>(Transpose trans, int m, int n,
                                           int nrhs, cuComplex *A, int lda,
                                           cuComplex *B, int ldb, cuComplex *X,
                                           int ldx, int *info) {

  if (trans != Transpose::NoTrans) {
    *info = -1;
    return;
  }
  if (m < 0 || m < n) {
    *info = -2;
    return;
  }
  if (n < 0) {
    *info = -3;
    return;
  }
  if (nrhs < 0) {
    *info = -4;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -6;
    return;
  }
  if (ldb < std::max(1, m)) {
    *info = -8;
    return;
  }
  if (ldx < std::max(1, n)) {
    *info = -10;
    return;
  }
  if (m == 0 || n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(cuComplex);
  size_t szB = (size_t)ldb * nrhs * sizeof(cuComplex);
  size_t szX = (size_t)ldx * nrhs * sizeof(cuComplex);

  cuComplex *dA = nullptr;
  cuComplex *dB = nullptr;
  cuComplex *dX = nullptr;
  void *d_work = nullptr;
  int *d_info = nullptr;
  int niter = 0;
  size_t lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&dX, szX);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnCCgels_bufferSize(m_handle, m, n, nrhs, dA, lda, dB, ldb, dX,
                                   ldx, nullptr, &lwork);

  clap_cudaMalloc(&d_work, std::max((size_t)1, lwork));

  clap_cusolverDnCCgels(m_handle, m, n, nrhs, dA, lda, dB, ldb, dX, ldx, d_work,
                        lwork, &niter, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  if (*info == 0) {
    clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(X, dX, szX, cudaMemcpyDeviceToHost);
  }

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dX);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::gels_impl<cuDoubleComplex>(Transpose trans, int m, int n,
                                                 int nrhs, cuDoubleComplex *A,
                                                 int lda, cuDoubleComplex *B,
                                                 int ldb, cuDoubleComplex *X,
                                                 int ldx, int *info) {

  if (trans != Transpose::NoTrans) {
    *info = -1;
    return;
  }
  if (m < 0 || m < n) {
    *info = -2;
    return;
  }
  if (n < 0) {
    *info = -3;
    return;
  }
  if (nrhs < 0) {
    *info = -4;
    return;
  }
  if (lda < std::max(1, m)) {
    *info = -6;
    return;
  }
  if (ldb < std::max(1, m)) {
    *info = -8;
    return;
  }
  if (ldx < std::max(1, n)) {
    *info = -10;
    return;
  }
  if (m == 0 || n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(cuDoubleComplex);
  size_t szB = (size_t)ldb * nrhs * sizeof(cuDoubleComplex);
  size_t szX = (size_t)ldx * nrhs * sizeof(cuDoubleComplex);

  cuDoubleComplex *dA = nullptr;
  cuDoubleComplex *dB = nullptr;
  cuDoubleComplex *dX = nullptr;
  void *d_work = nullptr;
  int *d_info = nullptr;
  int niter = 0;
  size_t lwork = 0;

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dB, szB);
  clap_cudaMalloc((void **)&dX, szX);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);
  clap_cudaMemcpy(dB, B, szB, cudaMemcpyHostToDevice);

  clap_cusolverDnZZgels_bufferSize(m_handle, m, n, nrhs, dA, lda, dB, ldb, dX,
                                   ldx, nullptr, &lwork);

  clap_cudaMalloc(&d_work, std::max((size_t)1, lwork));

  clap_cusolverDnZZgels(m_handle, m, n, nrhs, dA, lda, dB, ldb, dX, ldx, d_work,
                        lwork, &niter, d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  if (*info == 0) {
    clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(B, dB, szB, cudaMemcpyDeviceToHost);
    clap_cudaMemcpy(X, dX, szX, cudaMemcpyDeviceToHost);
  }

  clap_cudaFree(dA);
  clap_cudaFree(dB);
  clap_cudaFree(dX);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::syev_impl<float>(Job jobz, Uplo uplo, int n, float *A,
                                       int lda, float *w, int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);

  float *dA = nullptr, *dW = nullptr, *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;
  int jobz_i = to_cusolver_job(jobz);
  int uplo_i = to_cusolver_uplo(uplo);

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dW, (size_t)n * sizeof(float));
  clap_cudaMalloc((void **)&d_info, sizeof(int));
  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnSsyevd_bufferSize(m_handle, jobz_i, uplo_i, n, dA, lda, dW,
                                   &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(float));

  clap_cusolverDnSsyevd(m_handle, jobz_i, uplo_i, n, dA, lda, dW, d_work, lwork,
                        d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(w, dW, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dW);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::syev_impl<double>(Job jobz, Uplo uplo, int n, double *A,
                                        int lda, double *w, int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);

  double *dA = nullptr, *dW = nullptr, *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;
  int jobz_i = to_cusolver_job(jobz);
  int uplo_i = to_cusolver_uplo(uplo);

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dW, (size_t)n * sizeof(double));
  clap_cudaMalloc((void **)&d_info, sizeof(int));
  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnDsyevd_bufferSize(m_handle, jobz_i, uplo_i, n, dA, lda, dW,
                                   &lwork);
  clap_cudaMalloc((void **)&d_work, (size_t)lwork * sizeof(double));

  clap_cusolverDnDsyevd(m_handle, jobz_i, uplo_i, n, dA, lda, dW, d_work, lwork,
                        d_info);
  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(w, dW, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost);
  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dW);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::heev_impl<cuComplex, float>(Job jobz, Uplo uplo, int n,
                                                  cuComplex *A, int lda,
                                                  float *w, int *info) {

  if (jobz != Job::NoVec && jobz != Job::Vec) {
    *info = -1;
    return;
  }

  if (uplo != Uplo::Upper && uplo != Uplo::Lower) {
    *info = -2;
    return;
  }

  if (n < 0) {
    *info = -3;
    return;
  }

  if (lda < std::max(1, n)) {
    *info = -5;
    return;
  }

  if (n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(cuComplex);
  size_t szW = (size_t)n * sizeof(float);

  cuComplex *dA = nullptr;
  float *dW = nullptr;
  cuComplex *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  int jobz_i = to_cusolver_job(jobz);
  int uplo_i = to_cusolver_uplo(uplo);

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dW, szW);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnCheevd_bufferSize(m_handle, jobz_i, uplo_i, n, dA, lda, dW,
                                   &lwork);

  clap_cudaMalloc((void **)&d_work,
                  (size_t)std::max(1, lwork) * sizeof(cuComplex));

  clap_cusolverDnCheevd(m_handle, jobz_i, uplo_i, n, dA, lda, dW, d_work, lwork,
                        d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);

  clap_cudaMemcpy(w, dW, szW, cudaMemcpyDeviceToHost);

  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dW);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

template <>
void CuSolverBackend::heev_impl<cuDoubleComplex, double>(Job jobz, Uplo uplo,
                                                         int n,
                                                         cuDoubleComplex *A,
                                                         int lda, double *w,
                                                         int *info) {

  if (jobz != Job::NoVec && jobz != Job::Vec) {
    *info = -1;
    return;
  }

  if (uplo != Uplo::Upper && uplo != Uplo::Lower) {
    *info = -2;
    return;
  }

  if (n < 0) {
    *info = -3;
    return;
  }

  if (lda < std::max(1, n)) {
    *info = -5;
    return;
  }

  if (n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(cuDoubleComplex);
  size_t szW = (size_t)n * sizeof(double);

  cuDoubleComplex *dA = nullptr;
  double *dW = nullptr;
  cuDoubleComplex *d_work = nullptr;
  int *d_info = nullptr;
  int lwork = 0;

  int jobz_i = to_cusolver_job(jobz);
  int uplo_i = to_cusolver_uplo(uplo);

  clap_cudaMalloc((void **)&dA, szA);
  clap_cudaMalloc((void **)&dW, szW);
  clap_cudaMalloc((void **)&d_info, sizeof(int));

  clap_cudaMemcpy(dA, A, szA, cudaMemcpyHostToDevice);

  clap_cusolverDnZheevd_bufferSize(m_handle, jobz_i, uplo_i, n, dA, lda, dW,
                                   &lwork);

  clap_cudaMalloc((void **)&d_work,
                  (size_t)std::max(1, lwork) * sizeof(cuDoubleComplex));

  clap_cusolverDnZheevd(m_handle, jobz_i, uplo_i, n, dA, lda, dW, d_work, lwork,
                        d_info);

  clap_cudaDeviceSynchronize();

  clap_cudaMemcpy(A, dA, szA, cudaMemcpyDeviceToHost);

  clap_cudaMemcpy(w, dW, szW, cudaMemcpyDeviceToHost);

  clap_cudaMemcpy(info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

  clap_cudaFree(dA);
  clap_cudaFree(dW);
  clap_cudaFree(d_work);
  clap_cudaFree(d_info);
}

void CuSolverBackend::sgetrf(Layout, int m, int n, float *A, int lda, int *ipiv,
                             int *info) {
  getrf_impl<float>(m, n, A, lda, ipiv, info);
}
void CuSolverBackend::dgetrf(Layout, int m, int n, double *A, int lda,
                             int *ipiv, int *info) {
  getrf_impl<double>(m, n, A, lda, ipiv, info);
}
void CuSolverBackend::cgetrf(Layout, int m, int n, cuComplex *A, int lda,
                             int *ipiv, int *info) {
  getrf_impl<cuComplex>(m, n, A, lda, ipiv, info);
}
void CuSolverBackend::zgetrf(Layout, int m, int n, cuDoubleComplex *A, int lda,
                             int *ipiv, int *info) {
  getrf_impl<cuDoubleComplex>(m, n, A, lda, ipiv, info);
}
void CuSolverBackend::sgetrs(Layout, Transpose t, int n, int nrhs,
                             const float *A, int lda, const int *ipiv, float *B,
                             int ldb, int *info) {
  getrs_impl<float>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void CuSolverBackend::dgetrs(Layout, Transpose t, int n, int nrhs,
                             const double *A, int lda, const int *ipiv,
                             double *B, int ldb, int *info) {
  getrs_impl<double>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void CuSolverBackend::cgetrs(Layout, Transpose t, int n, int nrhs,
                             const cuComplex *A, int lda, const int *ipiv,
                             cuComplex *B, int ldb, int *info) {
  getrs_impl<cuComplex>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void CuSolverBackend::zgetrs(Layout, Transpose t, int n, int nrhs,
                             const cuDoubleComplex *A, int lda, const int *ipiv,
                             cuDoubleComplex *B, int ldb, int *info) {
  getrs_impl<cuDoubleComplex>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void CuSolverBackend::sgesv(int n, int nrhs, float *A, int lda, int *ipiv,
                            float *B, int ldb, int *info) {
  gesv_impl<float>(Transpose::NoTrans, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void CuSolverBackend::dgesv(int n, int nrhs, double *A, int lda, int *ipiv,
                            double *B, int ldb, int *info) {
  gesv_impl<double>(Transpose::NoTrans, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void CuSolverBackend::cgesv(int n, int nrhs, cuComplex *A, int lda, int *ipiv,
                            cuComplex *B, int ldb, int *info) {
  gesv_impl<cuComplex>(Transpose::NoTrans, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void CuSolverBackend::zgesv(int n, int nrhs, cuDoubleComplex *A, int lda,
                            int *ipiv, cuDoubleComplex *B, int ldb, int *info) {
  gesv_impl<cuDoubleComplex>(Transpose::NoTrans, n, nrhs, A, lda, ipiv, B, ldb,
                             info);
}
void CuSolverBackend::sgetri(int, float *, int, const int *, int *info) {
  std::cerr << "[CLAP] sgetri not supported in cuSOLVER, use CPU backend\n";
  *info = -1;
}
void CuSolverBackend::dgetri(int, double *, int, const int *, int *info) {
  std::cerr << "[CLAP] dgetri not supported in cuSOLVER, use CPU backend\n";
  *info = -1;
}
void CuSolverBackend::cgetri(int, cuComplex *, int, const int *, int *info) {
  std::cerr << "[CLAP] cgetri not supported in cuSOLVER, use CPU backend\n";
  *info = -1;
}
void CuSolverBackend::zgetri(int, cuDoubleComplex *, int, const int *,
                             int *info) {
  std::cerr << "[CLAP] zgetri not supported in cuSOLVER, use CPU backend\n";
  *info = -1;
}
void CuSolverBackend::spotrf(Layout, Uplo u, int n, float *A, int lda,
                             int *info) {
  potrf_impl<float>(u, n, A, lda, info);
}
void CuSolverBackend::dpotrf(Layout, Uplo u, int n, double *A, int lda,
                             int *info) {
  potrf_impl<double>(u, n, A, lda, info);
}
void CuSolverBackend::cpotrf(Layout, Uplo u, int n, cuComplex *A, int lda,
                             int *info) {
  potrf_impl<cuComplex>(u, n, A, lda, info);
}
void CuSolverBackend::zpotrf(Layout, Uplo u, int n, cuDoubleComplex *A, int lda,
                             int *info) {
  potrf_impl<cuDoubleComplex>(u, n, A, lda, info);
}
void CuSolverBackend::spotri(Layout, Uplo u, int n, float *A, int lda,
                             int *info) {
  potri_impl<float>(u, n, A, lda, info);
}
void CuSolverBackend::dpotri(Layout, Uplo u, int n, double *A, int lda,
                             int *info) {
  potri_impl<double>(u, n, A, lda, info);
}
void CuSolverBackend::cpotri(Layout, Uplo u, int n, cuComplex *A, int lda,
                             int *info) {
  potri_impl<cuComplex>(u, n, A, lda, info);
}
void CuSolverBackend::zpotri(Layout, Uplo u, int n, cuDoubleComplex *A, int lda,
                             int *info) {
  potri_impl<cuDoubleComplex>(u, n, A, lda, info);
}
void CuSolverBackend::spotrs(Layout, Uplo u, int n, int nrhs, const float *A,
                             int lda, float *B, int ldb, int *info) {
  potrs_impl<float>(u, n, nrhs, A, lda, B, ldb, info);
}
void CuSolverBackend::dpotrs(Layout, Uplo u, int n, int nrhs, const double *A,
                             int lda, double *B, int ldb, int *info) {
  potrs_impl<double>(u, n, nrhs, A, lda, B, ldb, info);
}
void CuSolverBackend::cpotrs(Layout, Uplo u, int n, int nrhs,
                             const cuComplex *A, int lda, cuComplex *B, int ldb,
                             int *info) {
  potrs_impl<cuComplex>(u, n, nrhs, A, lda, B, ldb, info);
}
void CuSolverBackend::zpotrs(Layout, Uplo u, int n, int nrhs,
                             const cuDoubleComplex *A, int lda,
                             cuDoubleComplex *B, int ldb, int *info) {
  potrs_impl<cuDoubleComplex>(u, n, nrhs, A, lda, B, ldb, info);
}
void CuSolverBackend::sposv(Layout l, Uplo u, int n, int nrhs, float *A,
                            int lda, float *B, int ldb, int *info) {
  potrf_impl<float>(u, n, A, lda, info);
  if (*info == 0)
    spotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
void CuSolverBackend::dposv(Layout l, Uplo u, int n, int nrhs, double *A,
                            int lda, double *B, int ldb, int *info) {
  potrf_impl<double>(u, n, A, lda, info);
  if (*info == 0)
    dpotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
void CuSolverBackend::cposv(Layout l, Uplo u, int n, int nrhs, cuComplex *A,
                            int lda, cuComplex *B, int ldb, int *info) {
  potrf_impl<cuComplex>(u, n, A, lda, info);
  if (*info == 0)
    cpotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
void CuSolverBackend::zposv(Layout l, Uplo u, int n, int nrhs,
                            cuDoubleComplex *A, int lda, cuDoubleComplex *B,
                            int ldb, int *info) {
  potrf_impl<cuDoubleComplex>(u, n, A, lda, info);
  if (*info == 0)
    zpotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
void CuSolverBackend::sgeqrf(Layout, int m, int n, float *A, int lda,
                             float *tau, int *info) {
  geqrf_impl<float>(m, n, A, lda, tau, info);
}
void CuSolverBackend::dgeqrf(Layout, int m, int n, double *A, int lda,
                             double *tau, int *info) {
  geqrf_impl<double>(m, n, A, lda, tau, info);
}
void CuSolverBackend::cgeqrf(Layout layout, int m, int n, cuComplex *A, int lda,
                             cuComplex *tau, int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  geqrf_impl<cuComplex>(m, n, A, lda, tau, info);
}
void CuSolverBackend::zgeqrf(Layout layout, int m, int n, cuDoubleComplex *A,
                             int lda, cuDoubleComplex *tau, int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  geqrf_impl<cuDoubleComplex>(m, n, A, lda, tau, info);
}
void CuSolverBackend::sorgqr(Layout, int m, int n, int k, float *A, int lda,
                             const float *tau, int *info) {
  orgqr_impl<float>(m, n, k, A, lda, tau, info);
}
void CuSolverBackend::dorgqr(Layout, int m, int n, int k, double *A, int lda,
                             const double *tau, int *info) {
  orgqr_impl<double>(m, n, k, A, lda, tau, info);
}
void CuSolverBackend::cungqr(Layout layout, int m, int n, int k, cuComplex *A,
                             int lda, const cuComplex *tau, int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  ungqr_impl<cuComplex>(m, n, k, A, lda, tau, info);
}

void CuSolverBackend::zungqr(Layout layout, int m, int n, int k,
                             cuDoubleComplex *A, int lda,
                             const cuDoubleComplex *tau, int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  ungqr_impl<cuDoubleComplex>(m, n, k, A, lda, tau, info);
}
void CuSolverBackend::sgels(Layout layout, Transpose trans, int m, int n,
                            int nrhs, float *A, int lda, float *B, int ldb,
                            int *info) {

  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }

  if (trans != Transpose::NoTrans) {
    *info = -1;
    return;
  }

  if (m < 0) {
    *info = -3;
    return;
  }
  if (n < 0) {
    *info = -4;
    return;
  }
  if (nrhs < 0) {
    *info = -5;
    return;
  }

  if (lda < std::max(1, m)) {
    *info = -7;
    return;
  }

  if (ldb < std::max(1, std::max(m, n))) {
    *info = -9;
    return;
  }

  if (m < n) {
    throw std::runtime_error("cuSOLVER GELS: m < n not implemented");
  }

  if (n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  int ldx = std::max(1, n);

  std::vector<float> X((size_t)ldx * nrhs);

  gels_impl<float>(trans, m, n, nrhs, A, lda, B, ldb, X.data(), ldx, info);

  if (*info != 0)
    return;

  for (int j = 0; j < nrhs; ++j) {
    for (int i = 0; i < n; ++i) {
      B[(size_t)j * ldb + i] = X[(size_t)j * ldx + i];
    }
  }
}

void CuSolverBackend::dgels(Layout layout, Transpose trans, int m, int n,
                            int nrhs, double *A, int lda, double *B, int ldb,
                            int *info) {

  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }

  if (trans != Transpose::NoTrans) {
    *info = -1;
    return;
  }

  if (m < 0) {
    *info = -3;
    return;
  }
  if (n < 0) {
    *info = -4;
    return;
  }
  if (nrhs < 0) {
    *info = -5;
    return;
  }

  if (lda < std::max(1, m)) {
    *info = -7;
    return;
  }

  if (ldb < std::max(1, std::max(m, n))) {
    *info = -9;
    return;
  }

  if (m < n) {
    throw std::runtime_error("cuSOLVER GELS: m < n not implemented");
  }

  if (n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  int ldx = std::max(1, n);

  std::vector<double> X((size_t)ldx * nrhs);

  gels_impl<double>(trans, m, n, nrhs, A, lda, B, ldb, X.data(), ldx, info);

  if (*info != 0)
    return;

  for (int j = 0; j < nrhs; ++j) {
    for (int i = 0; i < n; ++i) {
      B[(size_t)j * ldb + i] = X[(size_t)j * ldx + i];
    }
  }
}

void CuSolverBackend::cgels(Layout layout, Transpose trans, int m, int n,
                            int nrhs, cuComplex *A, int lda, cuComplex *B,
                            int ldb, int *info) {

  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }

  if (trans != Transpose::NoTrans) {
    *info = -1;
    return;
  }

  if (m < 0) {
    *info = -3;
    return;
  }
  if (n < 0) {
    *info = -4;
    return;
  }
  if (nrhs < 0) {
    *info = -5;
    return;
  }

  if (lda < std::max(1, m)) {
    *info = -7;
    return;
  }

  if (ldb < std::max(1, std::max(m, n))) {
    *info = -9;
    return;
  }

  if (m < n) {
    throw std::runtime_error("cuSOLVER GELS: m < n not implemented");
  }

  if (n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  int ldx = std::max(1, n);

  std::vector<cuComplex> X((size_t)ldx * nrhs);

  gels_impl<cuComplex>(trans, m, n, nrhs, A, lda, B, ldb, X.data(), ldx, info);

  if (*info != 0)
    return;

  for (int j = 0; j < nrhs; ++j) {
    for (int i = 0; i < n; ++i) {
      B[(size_t)j * ldb + i] = X[(size_t)j * ldx + i];
    }
  }
}

void CuSolverBackend::zgels(Layout layout, Transpose trans, int m, int n,
                            int nrhs, cuDoubleComplex *A, int lda,
                            cuDoubleComplex *B, int ldb, int *info) {

  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }

  if (trans != Transpose::NoTrans) {
    *info = -1;
    return;
  }

  if (m < 0) {
    *info = -3;
    return;
  }
  if (n < 0) {
    *info = -4;
    return;
  }
  if (nrhs < 0) {
    *info = -5;
    return;
  }

  if (lda < std::max(1, m)) {
    *info = -7;
    return;
  }

  if (ldb < std::max(1, std::max(m, n))) {
    *info = -9;
    return;
  }

  if (m < n) {
    throw std::runtime_error("cuSOLVER GELS: m < n not implemented");
  }

  if (n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  int ldx = std::max(1, n);

  std::vector<cuDoubleComplex> X((size_t)ldx * nrhs);

  gels_impl<cuDoubleComplex>(trans, m, n, nrhs, A, lda, B, ldb, X.data(), ldx,
                             info);

  if (*info != 0)
    return;

  for (int j = 0; j < nrhs; ++j) {
    for (int i = 0; i < n; ++i) {
      B[(size_t)j * ldb + i] = X[(size_t)j * ldx + i];
    }
  }
}

void CuSolverBackend::ssyev(Layout, Job j, Uplo u, int n, float *A, int lda,
                            float *w, int *info) {
  syev_impl<float>(j, u, n, A, lda, w, info);
}
void CuSolverBackend::dsyev(Layout, Job j, Uplo u, int n, double *A, int lda,
                            double *w, int *info) {
  syev_impl<double>(j, u, n, A, lda, w, info);
}
void CuSolverBackend::cheev(Layout, Job j, Uplo u, int n, cuComplex *A, int lda,
                            float *w, int *info) {
  heev_impl<cuComplex, float>(j, u, n, A, lda, w, info);
}
void CuSolverBackend::zheev(Layout, Job j, Uplo u, int n, cuDoubleComplex *A,
                            int lda, double *w, int *info) {
  heev_impl<cuDoubleComplex, double>(j, u, n, A, lda, w, info);
}
void CuSolverBackend::sgeev(Layout, Job, Job, int, float *, int, float *,
                            float *, float *, int, float *, int, int *info) {
  std::cerr << "[CLAP] cuSOLVER geev not supported\n";
  *info = -1;
}
void CuSolverBackend::dgeev(Layout, Job, Job, int, double *, int, double *,
                            double *, double *, int, double *, int, int *info) {
  std::cerr << "[CLAP] cuSOLVER geev not supported\n";
  *info = -1;
}
void CuSolverBackend::cgeev(Layout, Job, Job, int, cuComplex *, int,
                            cuComplex *, cuComplex *, int, cuComplex *, int,
                            int *info) {
  std::cerr << "[CLAP] cuSOLVER geev not supported\n";
  *info = -1;
}
void CuSolverBackend::zgeev(Layout, Job, Job, int, cuDoubleComplex *, int,
                            cuDoubleComplex *, cuDoubleComplex *, int,
                            cuDoubleComplex *, int, int *info) {
  std::cerr << "[CLAP] cuSOLVER geev not supported\n";
  *info = -1;
}
void CuSolverBackend::sgesvd(Layout, Job ju, Job jv, int m, int n, float *A,
                             int lda, float *s, float *U, int ldu, float *VT,
                             int ldvt, float *, int *info) {
  gesvd_impl<float, float>(ju, jv, m, n, A, lda, s, U, ldu, VT, ldvt, info);
}
void CuSolverBackend::dgesvd(Layout, Job ju, Job jv, int m, int n, double *A,
                             int lda, double *s, double *U, int ldu, double *VT,
                             int ldvt, double *, int *info) {
  gesvd_impl<double, double>(ju, jv, m, n, A, lda, s, U, ldu, VT, ldvt, info);
}
void CuSolverBackend::cgesvd(Layout, Job ju, Job jv, int m, int n, cuComplex *A,
                             int lda, float *s, cuComplex *U, int ldu,
                             cuComplex *VT, int ldvt, float *, int *info) {
  gesvd_impl<cuComplex, float>(ju, jv, m, n, A, lda, s, U, ldu, VT, ldvt, info);
}
void CuSolverBackend::zgesvd(Layout, Job ju, Job jv, int m, int n,
                             cuDoubleComplex *A, int lda, double *s,
                             cuDoubleComplex *U, int ldu, cuDoubleComplex *VT,
                             int ldvt, double *, int *info) {
  gesvd_impl<cuDoubleComplex, double>(ju, jv, m, n, A, lda, s, U, ldu, VT, ldvt,
                                      info);
}
void CuSolverBackend::strtrs(Layout, Uplo, Transpose, Diag, int, int,
                             const float *, int, float *, int, int *info) {
  std::cerr << "[CLAP] cuSOLVER trtrs not supported\n";
  *info = -1;
}
void CuSolverBackend::dtrtrs(Layout, Uplo, Transpose, Diag, int, int,
                             const double *, int, double *, int, int *info) {
  std::cerr << "[CLAP] cuSOLVER trtrs not supported\n";
  *info = -1;
}
void CuSolverBackend::ctrtrs(Layout, Uplo, Transpose, Diag, int, int,
                             const cuComplex *, int, cuComplex *, int,
                             int *info) {
  std::cerr << "[CLAP] cuSOLVER trtrs not supported\n";
  *info = -1;
}
void CuSolverBackend::ztrtrs(Layout, Uplo, Transpose, Diag, int, int,
                             const cuDoubleComplex *, int, cuDoubleComplex *,
                             int, int *info) {
  std::cerr << "[CLAP] cuSOLVER trtrs not supported\n";
  *info = -1;
}
} // namespace clap