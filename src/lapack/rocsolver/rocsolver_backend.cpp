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
#include "clap/detail/checked_call.hpp"
#include "clap/dyn_backends.hpp"
#include "clap/lapack_dyn.hpp"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace clap {

using detail::checked_call;

static int to_roc_uplo(Uplo u) { return (u == Uplo::Upper) ? 121 : 122; }

static rocblas_operation to_roc_trans(Transpose trans) {
  switch (trans) {
  case Transpose::NoTrans:
    return rocblas_operation_none;

  case Transpose::Trans:
    return rocblas_operation_transpose;

  case Transpose::ConjTrans:
    return rocblas_operation_conjugate_transpose;
  }

  throw std::invalid_argument("Invalid transpose operation");
}

static int to_roc_evect(Job j) {
  return (j == Job::Vec) ? rocblas_evect_original_solver
                         : rocblas_evect_none_solver;
}

static int to_roc_svect(Job j) {
  switch (j) {
  case Job::All:
    return rocblas_svect_all_solver;
  case Job::Some:
    return rocblas_svect_singular_solver;
  case Job::Over:
    return rocblas_svect_overwrite_solver;
  case Job::NoVec:
    return rocblas_svect_none_solver;
  default:
    throw std::invalid_argument(
        "rocSOLVER gesvd requires All, Some, Over, or NoVec");
  }
}

static void check_hip(hipError_t status, const char *operation) {
  if (status != 0)
    throw std::runtime_error(std::string(operation) +
                             " failed with HIP status " +
                             std::to_string(static_cast<int>(status)));
}

static void check_rocsolver(int status, const char *operation) {
  if (status != rocblas_status_success_solver)
    throw std::runtime_error(std::string(operation) +
                             " failed with rocSOLVER status " +
                             std::to_string(status));
}

const auto clap_hipMalloc =
    checked_call(dyn::p_hipMalloc, check_hip, "hipMalloc");
const auto clap_hipFree = checked_call(dyn::p_hipFree, check_hip, "hipFree");
const auto clap_hipMemcpy =
    checked_call(dyn::p_hipMemcpy, check_hip, "hipMemcpy");
const auto clap_rocsolver_sgetrf = checked_call(
    dyn_lapack::p_rocsolver_sgetrf, check_rocsolver, "rocsolver_sgetrf");
const auto clap_rocsolver_dgetrf = checked_call(
    dyn_lapack::p_rocsolver_dgetrf, check_rocsolver, "rocsolver_dgetrf");
const auto clap_rocsolver_cgetrf = checked_call(
    dyn_lapack::p_rocsolver_cgetrf, check_rocsolver, "rocsolver_cgetrf");
const auto clap_rocsolver_zgetrf = checked_call(
    dyn_lapack::p_rocsolver_zgetrf, check_rocsolver, "rocsolver_zgetrf");
const auto clap_rocsolver_sgetrs = checked_call(
    dyn_lapack::p_rocsolver_sgetrs, check_rocsolver, "rocsolver_sgetrs");
const auto clap_rocsolver_dgetrs = checked_call(
    dyn_lapack::p_rocsolver_dgetrs, check_rocsolver, "rocsolver_dgetrs");
const auto clap_rocsolver_cgetrs = checked_call(
    dyn_lapack::p_rocsolver_cgetrs, check_rocsolver, "rocsolver_cgetrs");
const auto clap_rocsolver_zgetrs = checked_call(
    dyn_lapack::p_rocsolver_zgetrs, check_rocsolver, "rocsolver_zgetrs");
const auto clap_rocsolver_sgesv = checked_call(
    dyn_lapack::p_rocsolver_sgesv, check_rocsolver, "rocsolver_sgesv");
const auto clap_rocsolver_dgesv = checked_call(
    dyn_lapack::p_rocsolver_dgesv, check_rocsolver, "rocsolver_dgesv");
const auto clap_rocsolver_cgesv = checked_call(
    dyn_lapack::p_rocsolver_cgesv, check_rocsolver, "rocsolver_cgesv");
const auto clap_rocsolver_zgesv = checked_call(
    dyn_lapack::p_rocsolver_zgesv, check_rocsolver, "rocsolver_zgesv");
const auto clap_rocsolver_sgetri = checked_call(
    dyn_lapack::p_rocsolver_sgetri, check_rocsolver, "rocsolver_sgetri");
const auto clap_rocsolver_dgetri = checked_call(
    dyn_lapack::p_rocsolver_dgetri, check_rocsolver, "rocsolver_dgetri");
const auto clap_rocsolver_cgetri = checked_call(
    dyn_lapack::p_rocsolver_cgetri, check_rocsolver, "rocsolver_cgetri");
const auto clap_rocsolver_zgetri = checked_call(
    dyn_lapack::p_rocsolver_zgetri, check_rocsolver, "rocsolver_zgetri");
const auto clap_rocsolver_spotrf = checked_call(
    dyn_lapack::p_rocsolver_spotrf, check_rocsolver, "rocsolver_spotrf");
const auto clap_rocsolver_dpotrf = checked_call(
    dyn_lapack::p_rocsolver_dpotrf, check_rocsolver, "rocsolver_dpotrf");
const auto clap_rocsolver_cpotrf = checked_call(
    dyn_lapack::p_rocsolver_cpotrf, check_rocsolver, "rocsolver_cpotrf");
const auto clap_rocsolver_zpotrf = checked_call(
    dyn_lapack::p_rocsolver_zpotrf, check_rocsolver, "rocsolver_zpotrf");
const auto clap_rocsolver_spotri = checked_call(
    dyn_lapack::p_rocsolver_spotri, check_rocsolver, "rocsolver_spotri");
const auto clap_rocsolver_dpotri = checked_call(
    dyn_lapack::p_rocsolver_dpotri, check_rocsolver, "rocsolver_dpotri");
const auto clap_rocsolver_cpotri = checked_call(
    dyn_lapack::p_rocsolver_cpotri, check_rocsolver, "rocsolver_cpotri");
const auto clap_rocsolver_zpotri = checked_call(
    dyn_lapack::p_rocsolver_zpotri, check_rocsolver, "rocsolver_zpotri");
const auto clap_rocsolver_spotrs = checked_call(
    dyn_lapack::p_rocsolver_spotrs, check_rocsolver, "rocsolver_spotrs");
const auto clap_rocsolver_dpotrs = checked_call(
    dyn_lapack::p_rocsolver_dpotrs, check_rocsolver, "rocsolver_dpotrs");
const auto clap_rocsolver_cpotrs = checked_call(
    dyn_lapack::p_rocsolver_cpotrs, check_rocsolver, "rocsolver_cpotrs");
const auto clap_rocsolver_zpotrs = checked_call(
    dyn_lapack::p_rocsolver_zpotrs, check_rocsolver, "rocsolver_zpotrs");
const auto clap_rocsolver_sgeqrf = checked_call(
    dyn_lapack::p_rocsolver_sgeqrf, check_rocsolver, "rocsolver_sgeqrf");
const auto clap_rocsolver_dgeqrf = checked_call(
    dyn_lapack::p_rocsolver_dgeqrf, check_rocsolver, "rocsolver_dgeqrf");
const auto clap_rocsolver_cgeqrf = checked_call(
    dyn_lapack::p_rocsolver_cgeqrf, check_rocsolver, "rocsolver_cgeqrf");
const auto clap_rocsolver_zgeqrf = checked_call(
    dyn_lapack::p_rocsolver_zgeqrf, check_rocsolver, "rocsolver_zgeqrf");
const auto clap_rocsolver_sorgqr = checked_call(
    dyn_lapack::p_rocsolver_sorgqr, check_rocsolver, "rocsolver_sorgqr");
const auto clap_rocsolver_dorgqr = checked_call(
    dyn_lapack::p_rocsolver_dorgqr, check_rocsolver, "rocsolver_dorgqr");
const auto clap_rocsolver_cungqr = checked_call(
    dyn_lapack::p_rocsolver_cungqr, check_rocsolver, "rocsolver_cungqr");
const auto clap_rocsolver_zungqr = checked_call(
    dyn_lapack::p_rocsolver_zungqr, check_rocsolver, "rocsolver_zungqr");
const auto clap_rocsolver_sgels = checked_call(
    dyn_lapack::p_rocsolver_sgels, check_rocsolver, "rocsolver_sgels");
const auto clap_rocsolver_dgels = checked_call(
    dyn_lapack::p_rocsolver_dgels, check_rocsolver, "rocsolver_dgels");
const auto clap_rocsolver_cgels = checked_call(
    dyn_lapack::p_rocsolver_cgels, check_rocsolver, "rocsolver_cgels");
const auto clap_rocsolver_zgels = checked_call(
    dyn_lapack::p_rocsolver_zgels, check_rocsolver, "rocsolver_zgels");
const auto clap_rocsolver_sgesvd = checked_call(
    dyn_lapack::p_rocsolver_sgesvd, check_rocsolver, "rocsolver_sgesvd");
const auto clap_rocsolver_dgesvd = checked_call(
    dyn_lapack::p_rocsolver_dgesvd, check_rocsolver, "rocsolver_dgesvd");
const auto clap_rocsolver_cgesvd = checked_call(
    dyn_lapack::p_rocsolver_cgesvd, check_rocsolver, "rocsolver_cgesvd");
const auto clap_rocsolver_zgesvd = checked_call(
    dyn_lapack::p_rocsolver_zgesvd, check_rocsolver, "rocsolver_zgesvd");
const auto clap_rocsolver_ssyev = checked_call(
    dyn_lapack::p_rocsolver_ssyev, check_rocsolver, "rocsolver_ssyev");
const auto clap_rocsolver_dsyev = checked_call(
    dyn_lapack::p_rocsolver_dsyev, check_rocsolver, "rocsolver_dsyev");
const auto clap_rocsolver_cheev = checked_call(
    dyn_lapack::p_rocsolver_cheev, check_rocsolver, "rocsolver_cheev");
const auto clap_rocsolver_zheev = checked_call(
    dyn_lapack::p_rocsolver_zheev, check_rocsolver, "rocsolver_zheev");

RocSolverBackend::RocSolverBackend() : m_handle(nullptr) {
  std::cout << "Using rocSOLVER backend\n";

  if (!dyn_lapack::loadRocSolver())
    throw std::runtime_error("rocSOLVER load failed");

  if (!dyn::loadHipAndRocblas())
    throw std::runtime_error("HIP/rocBLAS runtime load failed for rocSOLVER");

  if (dyn::p_rocblas_create_handle(&m_handle) != 0)
    throw std::runtime_error("rocblas_create_handle failed for rocSOLVER");
}

RocSolverBackend::~RocSolverBackend() {
  if (m_handle && dyn::p_rocblas_destroy_handle)
    dyn::p_rocblas_destroy_handle(m_handle);
}

template <>
void RocSolverBackend::getrf_impl<float>(rocblas_int m, rocblas_int n, float *A,
                                         rocblas_int lda, rocblas_int *ipiv,
                                         rocblas_int *info) {

  int k = std::min(m, n);
  size_t szA = (size_t)lda * n * sizeof(float);

  float *dA = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_ipiv, (size_t)k * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_sgetrf(m_handle, m, n, dA, lda, d_ipiv, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(ipiv, d_ipiv, (size_t)k * sizeof(int), hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::getrf_impl<double>(rocblas_int m, rocblas_int n,
                                          double *A, rocblas_int lda,
                                          rocblas_int *ipiv,
                                          rocblas_int *info) {

  int k = std::min(m, n);
  size_t szA = (size_t)lda * n * sizeof(double);

  double *dA = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_ipiv, (size_t)k * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_dgetrf(m_handle, m, n, dA, lda, d_ipiv, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(ipiv, d_ipiv, (size_t)k * sizeof(int), hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::getrf_impl<rocblas_float_complex>(
    rocblas_int m, rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
    rocblas_int *ipiv, rocblas_int *info) {

  int k = std::min(m, n);
  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);

  rocblas_float_complex *dA = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_ipiv, (size_t)k * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_cgetrf(m_handle, m, n, dA, lda, d_ipiv, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(ipiv, d_ipiv, (size_t)k * sizeof(int), hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::getrf_impl<rocblas_double_complex>(
    rocblas_int m, rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
    rocblas_int *ipiv, rocblas_int *info) {

  int k = std::min(m, n);
  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);

  rocblas_double_complex *dA = nullptr;
  int *d_ipiv = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_ipiv, (size_t)k * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_zgetrf(m_handle, m, n, dA, lda, d_ipiv, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(ipiv, d_ipiv, (size_t)k * sizeof(int), hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::getrs_impl<float>(Transpose trans, rocblas_int n,
                                         rocblas_int nrhs, const float *A,
                                         rocblas_int lda,
                                         const rocblas_int *ipiv, float *B,
                                         rocblas_int ldb, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szB = (size_t)ldb * nrhs * sizeof(float);

  float *dA = nullptr;
  float *dB = nullptr;
  int *d_ipiv = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);
  clap_hipMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int), hipMemcpyHostToDevice);

  clap_rocsolver_sgetrs(m_handle, to_roc_trans(trans), n, nrhs, dA, lda, d_ipiv,
                        dB, ldb);

  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_ipiv);
}

template <>
void RocSolverBackend::getrs_impl<double>(Transpose trans, rocblas_int n,
                                          rocblas_int nrhs, const double *A,
                                          rocblas_int lda,
                                          const rocblas_int *ipiv, double *B,
                                          rocblas_int ldb, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szB = (size_t)ldb * nrhs * sizeof(double);

  double *dA = nullptr;
  double *dB = nullptr;
  int *d_ipiv = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);
  clap_hipMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int), hipMemcpyHostToDevice);

  clap_rocsolver_dgetrs(m_handle, to_roc_trans(trans), n, nrhs, dA, lda, d_ipiv,
                        dB, ldb);

  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_ipiv);
}

template <>
void RocSolverBackend::getrs_impl<rocblas_float_complex>(
    Transpose trans, rocblas_int n, rocblas_int nrhs,
    const rocblas_float_complex *A, rocblas_int lda, const rocblas_int *ipiv,
    rocblas_float_complex *B, rocblas_int ldb, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);
  size_t szB = (size_t)ldb * nrhs * sizeof(rocblas_float_complex);

  rocblas_float_complex *dA = nullptr;
  rocblas_float_complex *dB = nullptr;
  int *d_ipiv = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);
  clap_hipMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int), hipMemcpyHostToDevice);

  clap_rocsolver_cgetrs(m_handle, to_roc_trans(trans), n, nrhs, dA, lda, d_ipiv,
                        dB, ldb);

  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_ipiv);
}

template <>
void RocSolverBackend::getrs_impl<rocblas_double_complex>(
    Transpose trans, rocblas_int n, rocblas_int nrhs,
    const rocblas_double_complex *A, rocblas_int lda, const rocblas_int *ipiv,
    rocblas_double_complex *B, rocblas_int ldb, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);
  size_t szB = (size_t)ldb * nrhs * sizeof(rocblas_double_complex);

  rocblas_double_complex *dA = nullptr;
  rocblas_double_complex *dB = nullptr;
  int *d_ipiv = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);
  clap_hipMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int), hipMemcpyHostToDevice);

  clap_rocsolver_zgetrs(m_handle, to_roc_trans(trans), n, nrhs, dA, lda, d_ipiv,
                        dB, ldb);

  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_ipiv);
}

template <>
void RocSolverBackend::gesv_impl<float>(rocblas_int n, rocblas_int nrhs,
                                        float *A, rocblas_int lda,
                                        rocblas_int *ipiv, float *B,
                                        rocblas_int ldb, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szB = (size_t)ldb * nrhs * sizeof(float);

  float *dA = nullptr, *dB = nullptr;
  int *d_ipiv = nullptr, *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_sgesv(m_handle, n, nrhs, dA, lda, d_ipiv, dB, ldb, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  clap_hipMemcpy(ipiv, d_ipiv, (size_t)n * sizeof(int), hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::gesv_impl<double>(rocblas_int n, rocblas_int nrhs,
                                         double *A, rocblas_int lda,
                                         rocblas_int *ipiv, double *B,
                                         rocblas_int ldb, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szB = (size_t)ldb * nrhs * sizeof(double);

  double *dA = nullptr, *dB = nullptr;
  int *d_ipiv = nullptr, *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_dgesv(m_handle, n, nrhs, dA, lda, d_ipiv, dB, ldb, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  clap_hipMemcpy(ipiv, d_ipiv, (size_t)n * sizeof(int), hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::gesv_impl<rocblas_float_complex>(
    rocblas_int n, rocblas_int nrhs, rocblas_float_complex *A, rocblas_int lda,
    rocblas_int *ipiv, rocblas_float_complex *B, rocblas_int ldb,
    rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);
  size_t szB = (size_t)ldb * nrhs * sizeof(rocblas_float_complex);

  rocblas_float_complex *dA = nullptr, *dB = nullptr;
  int *d_ipiv = nullptr, *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_cgesv(m_handle, n, nrhs, dA, lda, d_ipiv, dB, ldb, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  clap_hipMemcpy(ipiv, d_ipiv, (size_t)n * sizeof(int), hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::gesv_impl<rocblas_double_complex>(
    rocblas_int n, rocblas_int nrhs, rocblas_double_complex *A, rocblas_int lda,
    rocblas_int *ipiv, rocblas_double_complex *B, rocblas_int ldb,
    rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);
  size_t szB = (size_t)ldb * nrhs * sizeof(rocblas_double_complex);

  rocblas_double_complex *dA = nullptr, *dB = nullptr;
  int *d_ipiv = nullptr, *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_zgesv(m_handle, n, nrhs, dA, lda, d_ipiv, dB, ldb, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  clap_hipMemcpy(ipiv, d_ipiv, (size_t)n * sizeof(int), hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::getri_impl<float>(rocblas_int n, float *A,
                                         rocblas_int lda,
                                         const rocblas_int *ipiv,
                                         rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);

  float *dA = nullptr;
  int *d_ipiv = nullptr, *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int), hipMemcpyHostToDevice);

  clap_rocsolver_sgetri(m_handle, n, dA, lda, d_ipiv, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::getri_impl<double>(rocblas_int n, double *A,
                                          rocblas_int lda,
                                          const rocblas_int *ipiv,
                                          rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);

  double *dA = nullptr;
  int *d_ipiv = nullptr, *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int), hipMemcpyHostToDevice);

  clap_rocsolver_dgetri(m_handle, n, dA, lda, d_ipiv, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::getri_impl<rocblas_float_complex>(
    rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
    const rocblas_int *ipiv, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);

  rocblas_float_complex *dA = nullptr;
  int *d_ipiv = nullptr, *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int), hipMemcpyHostToDevice);

  clap_rocsolver_cgetri(m_handle, n, dA, lda, d_ipiv, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::getri_impl<rocblas_double_complex>(
    rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
    const rocblas_int *ipiv, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);

  rocblas_double_complex *dA = nullptr;
  int *d_ipiv = nullptr, *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_ipiv, (size_t)n * sizeof(int));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(d_ipiv, ipiv, (size_t)n * sizeof(int), hipMemcpyHostToDevice);

  clap_rocsolver_zgetri(m_handle, n, dA, lda, d_ipiv, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(d_ipiv);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::potrf_impl<float>(Uplo uplo, rocblas_int n, float *A,
                                         rocblas_int lda, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);
  float *dA = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_info, sizeof(int));
  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_spotrf(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::potrf_impl<double>(Uplo uplo, rocblas_int n, double *A,
                                          rocblas_int lda, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);
  double *dA = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_info, sizeof(int));
  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_dpotrf(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::potrf_impl<rocblas_float_complex>(
    Uplo uplo, rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
    rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);
  rocblas_float_complex *dA = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_info, sizeof(int));
  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_cpotrf(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::potrf_impl<rocblas_double_complex>(
    Uplo uplo, rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
    rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);
  rocblas_double_complex *dA = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_info, sizeof(int));
  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_zpotrf(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::potri_impl<float>(Uplo uplo, rocblas_int n, float *A,
                                         rocblas_int lda, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);

  float *dA = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_info, sizeof(int));
  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_spotri(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::potri_impl<double>(Uplo uplo, rocblas_int n, double *A,
                                          rocblas_int lda, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);

  double *dA = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_info, sizeof(int));
  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_dpotri(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::potri_impl<rocblas_float_complex>(
    Uplo uplo, rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
    rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);

  rocblas_float_complex *dA = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_info, sizeof(int));
  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_cpotri(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::potri_impl<rocblas_double_complex>(
    Uplo uplo, rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
    rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);

  rocblas_double_complex *dA = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_info, sizeof(int));
  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_zpotri(m_handle, to_roc_uplo(uplo), n, dA, lda, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::potrs_impl<float>(Uplo uplo, rocblas_int n,
                                         rocblas_int nrhs, const float *A,
                                         rocblas_int lda, float *B,
                                         rocblas_int ldb, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szB = (size_t)ldb * nrhs * sizeof(float);

  float *dA = nullptr;
  float *dB = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_spotrs(m_handle, to_roc_uplo(uplo), n, nrhs, dA, lda, dB, ldb);

  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(dB);
}

template <>
void RocSolverBackend::potrs_impl<double>(Uplo uplo, rocblas_int n,
                                          rocblas_int nrhs, const double *A,
                                          rocblas_int lda, double *B,
                                          rocblas_int ldb, rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szB = (size_t)ldb * nrhs * sizeof(double);

  double *dA = nullptr;
  double *dB = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_dpotrs(m_handle, to_roc_uplo(uplo), n, nrhs, dA, lda, dB, ldb);

  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(dB);
}

template <>
void RocSolverBackend::potrs_impl<rocblas_float_complex>(
    Uplo uplo, rocblas_int n, rocblas_int nrhs, const rocblas_float_complex *A,
    rocblas_int lda, rocblas_float_complex *B, rocblas_int ldb,
    rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);
  size_t szB = (size_t)ldb * nrhs * sizeof(rocblas_float_complex);

  rocblas_float_complex *dA = nullptr;
  rocblas_float_complex *dB = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_cpotrs(m_handle, to_roc_uplo(uplo), n, nrhs, dA, lda, dB, ldb);

  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(dB);
}

template <>
void RocSolverBackend::potrs_impl<rocblas_double_complex>(
    Uplo uplo, rocblas_int n, rocblas_int nrhs, const rocblas_double_complex *A,
    rocblas_int lda, rocblas_double_complex *B, rocblas_int ldb,
    rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);
  size_t szB = (size_t)ldb * nrhs * sizeof(rocblas_double_complex);

  rocblas_double_complex *dA = nullptr;
  rocblas_double_complex *dB = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_zpotrs(m_handle, to_roc_uplo(uplo), n, nrhs, dA, lda, dB, ldb);

  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(dB);
}

template <>
void RocSolverBackend::geqrf_impl<float>(rocblas_int m, rocblas_int n, float *A,
                                         rocblas_int lda, float *tau,
                                         rocblas_int *info) {

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

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_tau, szTau);

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_sgeqrf(m_handle, m, n, dA, lda, d_tau);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(tau, d_tau, szTau, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(d_tau);
}

template <>
void RocSolverBackend::geqrf_impl<double>(rocblas_int m, rocblas_int n,
                                          double *A, rocblas_int lda,
                                          double *tau, rocblas_int *info) {

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

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_tau, szTau);

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_dgeqrf(m_handle, m, n, dA, lda, d_tau);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(tau, d_tau, szTau, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(d_tau);
}

template <>
void RocSolverBackend::geqrf_impl<rocblas_float_complex>(
    rocblas_int m, rocblas_int n, rocblas_float_complex *A, rocblas_int lda,
    rocblas_float_complex *tau, rocblas_int *info) {

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

  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);
  size_t szTau = (size_t)std::min(m, n) * sizeof(rocblas_float_complex);

  rocblas_float_complex *dA = nullptr;
  rocblas_float_complex *d_tau = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_tau, szTau);

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_cgeqrf(m_handle, m, n, dA, lda, d_tau);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(tau, d_tau, szTau, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(d_tau);
}

template <>
void RocSolverBackend::geqrf_impl<rocblas_double_complex>(
    rocblas_int m, rocblas_int n, rocblas_double_complex *A, rocblas_int lda,
    rocblas_double_complex *tau, rocblas_int *info) {

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

  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);
  size_t szTau = (size_t)std::min(m, n) * sizeof(rocblas_double_complex);

  rocblas_double_complex *dA = nullptr;
  rocblas_double_complex *d_tau = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_tau, szTau);

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_zgeqrf(m_handle, m, n, dA, lda, d_tau);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(tau, d_tau, szTau, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(d_tau);
}

template <>
void RocSolverBackend::orgqr_impl<float>(rocblas_int m, rocblas_int n,
                                         rocblas_int k, float *A,
                                         rocblas_int lda, const float *tau,
                                         rocblas_int *info) {

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

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_tau, (size_t)std::max(1, k) * sizeof(float));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  if (k > 0)
    clap_hipMemcpy(d_tau, tau, szTau, hipMemcpyHostToDevice);

  clap_rocsolver_sorgqr(m_handle, m, n, k, dA, lda, d_tau);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(d_tau);
}

template <>
void RocSolverBackend::orgqr_impl<double>(rocblas_int m, rocblas_int n,
                                          rocblas_int k, double *A,
                                          rocblas_int lda, const double *tau,
                                          rocblas_int *info) {

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

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_tau, (size_t)std::max(1, k) * sizeof(double));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  if (k > 0)
    clap_hipMemcpy(d_tau, tau, szTau, hipMemcpyHostToDevice);

  clap_rocsolver_dorgqr(m_handle, m, n, k, dA, lda, d_tau);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(d_tau);
}

template <>
void RocSolverBackend::ungqr_impl<rocblas_float_complex>(
    rocblas_int m, rocblas_int n, rocblas_int k, rocblas_float_complex *A,
    rocblas_int lda, const rocblas_float_complex *tau, rocblas_int *info) {

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

  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);
  size_t szTau = (size_t)k * sizeof(rocblas_float_complex);

  rocblas_float_complex *dA = nullptr;
  rocblas_float_complex *d_tau = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_tau,
                 (size_t)std::max(1, k) * sizeof(rocblas_float_complex));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  if (k > 0)
    clap_hipMemcpy(d_tau, tau, szTau, hipMemcpyHostToDevice);

  clap_rocsolver_cungqr(m_handle, m, n, k, dA, lda, d_tau);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(d_tau);
}

template <>
void RocSolverBackend::ungqr_impl<rocblas_double_complex>(
    rocblas_int m, rocblas_int n, rocblas_int k, rocblas_double_complex *A,
    rocblas_int lda, const rocblas_double_complex *tau, rocblas_int *info) {

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

  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);
  size_t szTau = (size_t)k * sizeof(rocblas_double_complex);

  rocblas_double_complex *dA = nullptr;
  rocblas_double_complex *d_tau = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&d_tau,
                 (size_t)std::max(1, k) * sizeof(rocblas_double_complex));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  if (k > 0)
    clap_hipMemcpy(d_tau, tau, szTau, hipMemcpyHostToDevice);

  clap_rocsolver_zungqr(m_handle, m, n, k, dA, lda, d_tau);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);

  *info = 0;

  clap_hipFree(dA);
  clap_hipFree(d_tau);
}

template <>
void RocSolverBackend::gels_impl<float>(Transpose trans, rocblas_int m,
                                        rocblas_int n, rocblas_int nrhs,
                                        float *A, rocblas_int lda, float *B,
                                        rocblas_int ldb, rocblas_int *info) {

  if (m < 0) {
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
  if (ldb < std::max(1, std::max(m, n))) {
    *info = -8;
    return;
  }
  if (m == 0 || n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szB = (size_t)ldb * nrhs * sizeof(float);

  float *dA = nullptr;
  float *dB = nullptr;
  rocblas_int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_info, sizeof(rocblas_int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_sgels(m_handle, to_roc_trans(trans), m, n, nrhs, dA, lda, dB,
                       ldb, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::gels_impl<double>(Transpose trans, rocblas_int m,
                                         rocblas_int n, rocblas_int nrhs,
                                         double *A, rocblas_int lda, double *B,
                                         rocblas_int ldb, rocblas_int *info) {

  if (m < 0) {
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
  if (ldb < std::max(1, std::max(m, n))) {
    *info = -8;
    return;
  }
  if (m == 0 || n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szB = (size_t)ldb * nrhs * sizeof(double);

  double *dA = nullptr;
  double *dB = nullptr;
  rocblas_int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_info, sizeof(rocblas_int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_dgels(m_handle, to_roc_trans(trans), m, n, nrhs, dA, lda, dB,
                       ldb, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::gels_impl<rocblas_float_complex>(
    Transpose trans, rocblas_int m, rocblas_int n, rocblas_int nrhs,
    rocblas_float_complex *A, rocblas_int lda, rocblas_float_complex *B,
    rocblas_int ldb, rocblas_int *info) {

  if (m < 0) {
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
  if (ldb < std::max(1, std::max(m, n))) {
    *info = -8;
    return;
  }
  if (m == 0 || n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);
  size_t szB = (size_t)ldb * nrhs * sizeof(rocblas_float_complex);

  rocblas_float_complex *dA = nullptr;
  rocblas_float_complex *dB = nullptr;
  rocblas_int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_info, sizeof(rocblas_int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_cgels(m_handle, to_roc_trans(trans), m, n, nrhs, dA, lda, dB,
                       ldb, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::gels_impl<rocblas_double_complex>(
    Transpose trans, rocblas_int m, rocblas_int n, rocblas_int nrhs,
    rocblas_double_complex *A, rocblas_int lda, rocblas_double_complex *B,
    rocblas_int ldb, rocblas_int *info) {

  if (m < 0) {
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
  if (ldb < std::max(1, std::max(m, n))) {
    *info = -8;
    return;
  }
  if (m == 0 || n == 0 || nrhs == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);
  size_t szB = (size_t)ldb * nrhs * sizeof(rocblas_double_complex);

  rocblas_double_complex *dA = nullptr;
  rocblas_double_complex *dB = nullptr;
  rocblas_int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dB, szB);
  clap_hipMalloc((void **)&d_info, sizeof(rocblas_int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);
  clap_hipMemcpy(dB, B, szB, hipMemcpyHostToDevice);

  clap_rocsolver_zgels(m_handle, to_roc_trans(trans), m, n, nrhs, dA, lda, dB,
                       ldb, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(B, dB, szB, hipMemcpyDeviceToHost);
  clap_hipMemcpy(info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

  clap_hipFree(dA);
  clap_hipFree(dB);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::gesvd_impl<float, float>(
    Job jobu, Job jobvt, rocblas_int m, rocblas_int n, float *A,
    rocblas_int lda, float *s, float *U, rocblas_int ldu, float *VT,
    rocblas_int ldvt, rocblas_int *info) {

  int k = std::min(m, n);
  size_t szA = (size_t)lda * n * sizeof(float);
  size_t szU = (size_t)ldu * m * sizeof(float);
  size_t szV = (size_t)ldvt * n * sizeof(float);

  float *dA = nullptr, *dU = nullptr, *dVT = nullptr, *dS = nullptr,
        *dE = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dU, szU);
  clap_hipMalloc((void **)&dVT, szV);
  clap_hipMalloc((void **)&dS, (size_t)k * sizeof(float));
  clap_hipMalloc((void **)&dE, (size_t)k * sizeof(float));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_sgesvd(m_handle, to_roc_svect(jobu), to_roc_svect(jobvt), m, n,
                        dA, lda, dS, dU, ldu, dVT, ldvt, dE,
                        rocblas_outofplace_solver, d_info);

  clap_hipMemcpy(s, dS, (size_t)k * sizeof(float), hipMemcpyDeviceToHost);
  clap_hipMemcpy(U, dU, szU, hipMemcpyDeviceToHost);
  clap_hipMemcpy(VT, dVT, szV, hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(dU);
  clap_hipFree(dVT);
  clap_hipFree(dS);
  clap_hipFree(dE);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::gesvd_impl<double, double>(
    Job jobu, Job jobvt, rocblas_int m, rocblas_int n, double *A,
    rocblas_int lda, double *s, double *U, rocblas_int ldu, double *VT,
    rocblas_int ldvt, rocblas_int *info) {

  int k = std::min(m, n);
  size_t szA = (size_t)lda * n * sizeof(double);
  size_t szU = (size_t)ldu * m * sizeof(double);
  size_t szV = (size_t)ldvt * n * sizeof(double);

  double *dA = nullptr, *dU = nullptr, *dVT = nullptr, *dS = nullptr,
         *dE = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dU, szU);
  clap_hipMalloc((void **)&dVT, szV);
  clap_hipMalloc((void **)&dS, (size_t)k * sizeof(double));
  clap_hipMalloc((void **)&dE, (size_t)k * sizeof(double));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_dgesvd(m_handle, to_roc_svect(jobu), to_roc_svect(jobvt), m, n,
                        dA, lda, dS, dU, ldu, dVT, ldvt, dE,
                        rocblas_outofplace_solver, d_info);

  clap_hipMemcpy(s, dS, (size_t)k * sizeof(double), hipMemcpyDeviceToHost);
  clap_hipMemcpy(U, dU, szU, hipMemcpyDeviceToHost);
  clap_hipMemcpy(VT, dVT, szV, hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(dU);
  clap_hipFree(dVT);
  clap_hipFree(dS);
  clap_hipFree(dE);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::gesvd_impl<rocblas_float_complex, float>(
    Job jobu, Job jobvt, rocblas_int m, rocblas_int n, rocblas_float_complex *A,
    rocblas_int lda, float *s, rocblas_float_complex *U, rocblas_int ldu,
    rocblas_float_complex *VT, rocblas_int ldvt, rocblas_int *info) {

  if (jobu != Job::All && jobu != Job::Some && jobu != Job::NoVec) {
    *info = -1;
    return;
  }

  if (jobvt != Job::All && jobvt != Job::Some && jobvt != Job::NoVec) {
    *info = -2;
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

  if (lda < std::max(1, m)) {
    *info = -6;
    return;
  }

  const rocblas_int k = std::min(m, n);

  const rocblas_int ucols = (jobu == Job::All)    ? m
                            : (jobu == Job::Some) ? k
                                                  : 0;

  const rocblas_int vtrows = (jobvt == Job::All)    ? n
                             : (jobvt == Job::Some) ? k
                                                    : 0;

  if (ldu < std::max(1, jobu == Job::NoVec ? 1 : m)) {
    *info = -9;
    return;
  }

  if (ldvt < std::max(1, vtrows)) {
    *info = -11;
    return;
  }

  if (m == 0 || n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);

  size_t szU = (size_t)ldu * std::max(1, ucols) * sizeof(rocblas_float_complex);

  size_t szVT = (size_t)ldvt * std::max(1, n) * sizeof(rocblas_float_complex);

  size_t szS = (size_t)k * sizeof(float);
  size_t szE = (size_t)std::max(1, k) * sizeof(float);

  rocblas_float_complex *dA = nullptr;
  rocblas_float_complex *dU = nullptr;
  rocblas_float_complex *dVT = nullptr;

  float *dS = nullptr;
  float *dE = nullptr;
  rocblas_int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dU, szU);
  clap_hipMalloc((void **)&dVT, szVT);
  clap_hipMalloc((void **)&dS, szS);
  clap_hipMalloc((void **)&dE, szE);
  clap_hipMalloc((void **)&d_info, sizeof(rocblas_int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_cgesvd(m_handle, to_roc_svect(jobu), to_roc_svect(jobvt), m, n,
                        dA, lda, dS, dU, ldu, dVT, ldvt, dE,
                        rocblas_outofplace_solver, d_info);

  clap_hipMemcpy(info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

  if (*info == 0) {
    clap_hipMemcpy(s, dS, szS, hipMemcpyDeviceToHost);

    if (jobu != Job::NoVec) {
      clap_hipMemcpy(U, dU, (size_t)ldu * ucols * sizeof(rocblas_float_complex),
                     hipMemcpyDeviceToHost);
    }

    if (jobvt != Job::NoVec) {
      clap_hipMemcpy(VT, dVT, (size_t)ldvt * n * sizeof(rocblas_float_complex),
                     hipMemcpyDeviceToHost);
    }
  }

  clap_hipFree(dA);
  clap_hipFree(dU);
  clap_hipFree(dVT);
  clap_hipFree(dS);
  clap_hipFree(dE);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::gesvd_impl<rocblas_double_complex, double>(
    Job jobu, Job jobvt, rocblas_int m, rocblas_int n,
    rocblas_double_complex *A, rocblas_int lda, double *s,
    rocblas_double_complex *U, rocblas_int ldu, rocblas_double_complex *VT,
    rocblas_int ldvt, rocblas_int *info) {

  if (jobu != Job::All && jobu != Job::Some && jobu != Job::NoVec) {
    *info = -1;
    return;
  }

  if (jobvt != Job::All && jobvt != Job::Some && jobvt != Job::NoVec) {
    *info = -2;
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

  if (lda < std::max(1, m)) {
    *info = -6;
    return;
  }

  const rocblas_int k = std::min(m, n);

  const rocblas_int ucols = (jobu == Job::All)    ? m
                            : (jobu == Job::Some) ? k
                                                  : 0;

  const rocblas_int vtrows = (jobvt == Job::All)    ? n
                             : (jobvt == Job::Some) ? k
                                                    : 0;

  if (ldu < std::max(1, jobu == Job::NoVec ? 1 : m)) {
    *info = -9;
    return;
  }

  if (ldvt < std::max(1, vtrows)) {
    *info = -11;
    return;
  }

  if (m == 0 || n == 0) {
    *info = 0;
    return;
  }

  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);

  size_t szU =
      (size_t)ldu * std::max(1, ucols) * sizeof(rocblas_double_complex);

  size_t szVT = (size_t)ldvt * std::max(1, n) * sizeof(rocblas_double_complex);

  size_t szS = (size_t)k * sizeof(double);
  size_t szE = (size_t)std::max(1, k) * sizeof(double);

  rocblas_double_complex *dA = nullptr;
  rocblas_double_complex *dU = nullptr;
  rocblas_double_complex *dVT = nullptr;

  double *dS = nullptr;
  double *dE = nullptr;
  rocblas_int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dU, szU);
  clap_hipMalloc((void **)&dVT, szVT);
  clap_hipMalloc((void **)&dS, szS);
  clap_hipMalloc((void **)&dE, szE);
  clap_hipMalloc((void **)&d_info, sizeof(rocblas_int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_zgesvd(m_handle, to_roc_svect(jobu), to_roc_svect(jobvt), m, n,
                        dA, lda, dS, dU, ldu, dVT, ldvt, dE,
                        rocblas_outofplace_solver, d_info);

  clap_hipMemcpy(info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

  if (*info == 0) {
    clap_hipMemcpy(s, dS, szS, hipMemcpyDeviceToHost);

    if (jobu != Job::NoVec) {
      clap_hipMemcpy(U, dU,
                     (size_t)ldu * ucols * sizeof(rocblas_double_complex),
                     hipMemcpyDeviceToHost);
    }

    if (jobvt != Job::NoVec) {
      clap_hipMemcpy(VT, dVT, (size_t)ldvt * n * sizeof(rocblas_double_complex),
                     hipMemcpyDeviceToHost);
    }
  }

  clap_hipFree(dA);
  clap_hipFree(dU);
  clap_hipFree(dVT);
  clap_hipFree(dS);
  clap_hipFree(dE);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::syev_impl<float>(Job jobz, Uplo uplo, rocblas_int n,
                                        float *A, rocblas_int lda, float *w,
                                        rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(float);
  float *dA = nullptr, *dD = nullptr, *dE = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dD, (size_t)n * sizeof(float));
  clap_hipMalloc((void **)&dE, (size_t)n * sizeof(float));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_ssyev(m_handle, to_roc_evect(jobz), to_roc_uplo(uplo), n, dA,
                       lda, dD, dE, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(w, dD, (size_t)n * sizeof(float), hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(dD);
  clap_hipFree(dE);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::syev_impl<double>(Job jobz, Uplo uplo, rocblas_int n,
                                         double *A, rocblas_int lda, double *w,
                                         rocblas_int *info) {

  size_t szA = (size_t)lda * n * sizeof(double);
  double *dA = nullptr, *dD = nullptr, *dE = nullptr;
  int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dD, (size_t)n * sizeof(double));
  clap_hipMalloc((void **)&dE, (size_t)n * sizeof(double));
  clap_hipMalloc((void **)&d_info, sizeof(int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_dsyev(m_handle, to_roc_evect(jobz), to_roc_uplo(uplo), n, dA,
                       lda, dD, dE, d_info);

  clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  clap_hipMemcpy(w, dD, (size_t)n * sizeof(double), hipMemcpyDeviceToHost);
  int h_info = 0;
  clap_hipMemcpy(&h_info, d_info, sizeof(int), hipMemcpyDeviceToHost);
  *info = h_info;

  clap_hipFree(dA);
  clap_hipFree(dD);
  clap_hipFree(dE);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::heev_impl<rocblas_float_complex, float>(
    Job jobz, Uplo uplo, rocblas_int n, rocblas_float_complex *A,
    rocblas_int lda, float *w, rocblas_int *info) {

  if (jobz != Job::Vec && jobz != Job::NoVec) {
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

  size_t szA = (size_t)lda * n * sizeof(rocblas_float_complex);

  size_t szW = (size_t)n * sizeof(float);
  size_t szE = (size_t)n * sizeof(float);

  rocblas_float_complex *dA = nullptr;
  float *dD = nullptr;
  float *dE = nullptr;
  rocblas_int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dD, szW);
  clap_hipMalloc((void **)&dE, szE);
  clap_hipMalloc((void **)&d_info, sizeof(rocblas_int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_cheev(m_handle, to_roc_evect(jobz), to_roc_uplo(uplo), n, dA,
                       lda, dD, dE, d_info);

  clap_hipMemcpy(info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

  if (*info == 0) {
    clap_hipMemcpy(w, dD, szW, hipMemcpyDeviceToHost);

    clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  }

  clap_hipFree(dA);
  clap_hipFree(dD);
  clap_hipFree(dE);
  clap_hipFree(d_info);
}

template <>
void RocSolverBackend::heev_impl<rocblas_double_complex, double>(
    Job jobz, Uplo uplo, rocblas_int n, rocblas_double_complex *A,
    rocblas_int lda, double *w, rocblas_int *info) {

  if (jobz != Job::Vec && jobz != Job::NoVec) {
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

  size_t szA = (size_t)lda * n * sizeof(rocblas_double_complex);

  size_t szW = (size_t)n * sizeof(double);
  size_t szE = (size_t)n * sizeof(double);

  rocblas_double_complex *dA = nullptr;
  double *dD = nullptr;
  double *dE = nullptr;
  rocblas_int *d_info = nullptr;

  clap_hipMalloc((void **)&dA, szA);
  clap_hipMalloc((void **)&dD, szW);
  clap_hipMalloc((void **)&dE, szE);
  clap_hipMalloc((void **)&d_info, sizeof(rocblas_int));

  clap_hipMemcpy(dA, A, szA, hipMemcpyHostToDevice);

  clap_rocsolver_zheev(m_handle, to_roc_evect(jobz), to_roc_uplo(uplo), n, dA,
                       lda, dD, dE, d_info);

  clap_hipMemcpy(info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost);

  if (*info == 0) {
    clap_hipMemcpy(w, dD, szW, hipMemcpyDeviceToHost);

    clap_hipMemcpy(A, dA, szA, hipMemcpyDeviceToHost);
  }

  clap_hipFree(dA);
  clap_hipFree(dD);
  clap_hipFree(dE);
  clap_hipFree(d_info);
}

void RocSolverBackend::sgetrf(Layout, rocblas_int m, rocblas_int n, float *A,
                              rocblas_int lda, rocblas_int *ipiv,
                              rocblas_int *info) {
  getrf_impl<float>(m, n, A, lda, ipiv, info);
}
void RocSolverBackend::dgetrf(Layout, rocblas_int m, rocblas_int n, double *A,
                              rocblas_int lda, rocblas_int *ipiv,
                              rocblas_int *info) {
  getrf_impl<double>(m, n, A, lda, ipiv, info);
}
void RocSolverBackend::cgetrf(Layout, rocblas_int m, rocblas_int n,
                              rocblas_float_complex *A, rocblas_int lda,
                              rocblas_int *ipiv, rocblas_int *info) {
  getrf_impl<rocblas_float_complex>(m, n, A, lda, ipiv, info);
}
void RocSolverBackend::zgetrf(Layout, rocblas_int m, rocblas_int n,
                              std::complex<double> *A, rocblas_int lda,
                              rocblas_int *ipiv, rocblas_int *info) {
  getrf_impl<std::complex<double>>(m, n, A, lda, ipiv, info);
}
void RocSolverBackend::sgetrs(Layout, Transpose t, rocblas_int n,
                              rocblas_int nrhs, const float *A, rocblas_int lda,
                              const rocblas_int *ipiv, float *B,
                              rocblas_int ldb, rocblas_int *info) {
  getrs_impl<float>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::dgetrs(Layout, Transpose t, rocblas_int n,
                              rocblas_int nrhs, const double *A,
                              rocblas_int lda, const rocblas_int *ipiv,
                              double *B, rocblas_int ldb, rocblas_int *info) {
  getrs_impl<double>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::cgetrs(Layout, Transpose t, rocblas_int n,
                              rocblas_int nrhs, const rocblas_float_complex *A,
                              rocblas_int lda, const rocblas_int *ipiv,
                              rocblas_float_complex *B, rocblas_int ldb,
                              rocblas_int *info) {
  getrs_impl<rocblas_float_complex>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::zgetrs(Layout, Transpose t, rocblas_int n,
                              rocblas_int nrhs, const rocblas_double_complex *A,
                              rocblas_int lda, const rocblas_int *ipiv,
                              rocblas_double_complex *B, rocblas_int ldb,
                              rocblas_int *info) {
  getrs_impl<rocblas_double_complex>(t, n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::sgesv(rocblas_int n, rocblas_int nrhs, float *A,
                             rocblas_int lda, rocblas_int *ipiv, float *B,
                             rocblas_int ldb, rocblas_int *info) {
  gesv_impl<float>(n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::dgesv(rocblas_int n, rocblas_int nrhs, double *A,
                             rocblas_int lda, rocblas_int *ipiv, double *B,
                             rocblas_int ldb, rocblas_int *info) {
  gesv_impl<double>(n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::cgesv(rocblas_int n, rocblas_int nrhs,
                             rocblas_float_complex *A, rocblas_int lda,
                             rocblas_int *ipiv, rocblas_float_complex *B,
                             rocblas_int ldb, rocblas_int *info) {
  gesv_impl<rocblas_float_complex>(n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::zgesv(rocblas_int n, rocblas_int nrhs,
                             rocblas_double_complex *A, rocblas_int lda,
                             rocblas_int *ipiv, rocblas_double_complex *B,
                             rocblas_int ldb, rocblas_int *info) {
  gesv_impl<rocblas_double_complex>(n, nrhs, A, lda, ipiv, B, ldb, info);
}
void RocSolverBackend::sgetri(rocblas_int n, float *A, rocblas_int lda,
                              const rocblas_int *ipiv, rocblas_int *info) {
  getri_impl<float>(n, A, lda, ipiv, info);
}
void RocSolverBackend::dgetri(rocblas_int n, double *A, rocblas_int lda,
                              const rocblas_int *ipiv, rocblas_int *info) {
  getri_impl<double>(n, A, lda, ipiv, info);
}
void RocSolverBackend::cgetri(rocblas_int n, rocblas_float_complex *A,
                              rocblas_int lda, const rocblas_int *ipiv,
                              rocblas_int *info) {
  getri_impl<rocblas_float_complex>(n, A, lda, ipiv, info);
}
void RocSolverBackend::zgetri(rocblas_int n, rocblas_double_complex *A,
                              rocblas_int lda, const rocblas_int *ipiv,
                              rocblas_int *info) {
  getri_impl<rocblas_double_complex>(n, A, lda, ipiv, info);
}
void RocSolverBackend::spotrf(Layout, Uplo u, rocblas_int n, float *A,
                              rocblas_int lda, rocblas_int *info) {
  potrf_impl<float>(u, n, A, lda, info);
}
void RocSolverBackend::dpotrf(Layout, Uplo u, rocblas_int n, double *A,
                              rocblas_int lda, rocblas_int *info) {
  potrf_impl<double>(u, n, A, lda, info);
}
void RocSolverBackend::cpotrf(Layout, Uplo u, rocblas_int n,
                              rocblas_float_complex *A, rocblas_int lda,
                              rocblas_int *info) {
  potrf_impl<rocblas_float_complex>(u, n, A, lda, info);
}
void RocSolverBackend::zpotrf(Layout, Uplo u, rocblas_int n,
                              rocblas_double_complex *A, rocblas_int lda,
                              rocblas_int *info) {
  potrf_impl<rocblas_double_complex>(u, n, A, lda, info);
}
void RocSolverBackend::spotri(Layout, Uplo u, rocblas_int n, float *A,
                              rocblas_int lda, rocblas_int *info) {
  potri_impl<float>(u, n, A, lda, info);
}
void RocSolverBackend::dpotri(Layout, Uplo u, rocblas_int n, double *A,
                              rocblas_int lda, rocblas_int *info) {
  potri_impl<double>(u, n, A, lda, info);
}
void RocSolverBackend::cpotri(Layout, Uplo u, rocblas_int n,
                              rocblas_float_complex *A, rocblas_int lda,
                              rocblas_int *info) {
  potri_impl<rocblas_float_complex>(u, n, A, lda, info);
}
void RocSolverBackend::zpotri(Layout, Uplo u, rocblas_int n,
                              rocblas_double_complex *A, rocblas_int lda,
                              rocblas_int *info) {
  potri_impl<rocblas_double_complex>(u, n, A, lda, info);
}
void RocSolverBackend::spotrs(Layout, Uplo u, rocblas_int n, rocblas_int nrhs,
                              const float *A, rocblas_int lda, float *B,
                              rocblas_int ldb, rocblas_int *info) {
  potrs_impl<float>(u, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::dpotrs(Layout, Uplo u, rocblas_int n, rocblas_int nrhs,
                              const double *A, rocblas_int lda, double *B,
                              rocblas_int ldb, rocblas_int *info) {
  potrs_impl<double>(u, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::cpotrs(Layout, Uplo u, rocblas_int n, rocblas_int nrhs,
                              const rocblas_float_complex *A, rocblas_int lda,
                              rocblas_float_complex *B, rocblas_int ldb,
                              rocblas_int *info) {
  potrs_impl<rocblas_float_complex>(u, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::zpotrs(Layout, Uplo u, rocblas_int n, rocblas_int nrhs,
                              const rocblas_double_complex *A, rocblas_int lda,
                              rocblas_double_complex *B, rocblas_int ldb,
                              rocblas_int *info) {
  potrs_impl<rocblas_double_complex>(u, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::sposv(Layout l, Uplo u, rocblas_int n, rocblas_int nrhs,
                             float *A, rocblas_int lda, float *B,
                             rocblas_int ldb, rocblas_int *info) {
  potrf_impl<float>(u, n, A, lda, info);
  if (*info == 0)
    spotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::dposv(Layout l, Uplo u, rocblas_int n, rocblas_int nrhs,
                             double *A, rocblas_int lda, double *B,
                             rocblas_int ldb, rocblas_int *info) {
  potrf_impl<double>(u, n, A, lda, info);
  if (*info == 0)
    dpotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::cposv(Layout l, Uplo u, rocblas_int n, rocblas_int nrhs,
                             rocblas_float_complex *A, rocblas_int lda,
                             rocblas_float_complex *B, rocblas_int ldb,
                             rocblas_int *info) {
  potrf_impl<rocblas_float_complex>(u, n, A, lda, info);
  if (*info == 0)
    cpotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::zposv(Layout l, Uplo u, rocblas_int n, rocblas_int nrhs,
                             rocblas_double_complex *A, rocblas_int lda,
                             rocblas_double_complex *B, rocblas_int ldb,
                             rocblas_int *info) {
  potrf_impl<rocblas_double_complex>(u, n, A, lda, info);
  if (*info == 0)
    zpotrs(l, u, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::sgeqrf(Layout layout, rocblas_int m, rocblas_int n,
                              float *A, rocblas_int lda, float *tau,
                              rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  geqrf_impl<float>(m, n, A, lda, tau, info);
}
void RocSolverBackend::dgeqrf(Layout layout, rocblas_int m, rocblas_int n,
                              double *A, rocblas_int lda, double *tau,
                              rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  geqrf_impl<double>(m, n, A, lda, tau, info);
}
void RocSolverBackend::cgeqrf(Layout layout, rocblas_int m, rocblas_int n,
                              rocblas_float_complex *A, rocblas_int lda,
                              rocblas_float_complex *tau, rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  geqrf_impl<rocblas_float_complex>(m, n, A, lda, tau, info);
}
void RocSolverBackend::zgeqrf(Layout layout, rocblas_int m, rocblas_int n,
                              rocblas_double_complex *A, rocblas_int lda,
                              rocblas_double_complex *tau, rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  geqrf_impl<rocblas_double_complex>(m, n, A, lda, tau, info);
}
void RocSolverBackend::sorgqr(Layout layout, rocblas_int m, rocblas_int n,
                              rocblas_int k, float *A, rocblas_int lda,
                              const float *tau, rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  orgqr_impl<float>(m, n, k, A, lda, tau, info);
}
void RocSolverBackend::dorgqr(Layout layout, rocblas_int m, rocblas_int n,
                              rocblas_int k, double *A, rocblas_int lda,
                              const double *tau, rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  orgqr_impl<double>(m, n, k, A, lda, tau, info);
}
void RocSolverBackend::cungqr(Layout layout, rocblas_int m, rocblas_int n,
                              rocblas_int k, rocblas_float_complex *A,
                              rocblas_int lda, const rocblas_float_complex *tau,
                              rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  ungqr_impl<rocblas_float_complex>(m, n, k, A, lda, tau, info);
}
void RocSolverBackend::zungqr(Layout layout, rocblas_int m, rocblas_int n,
                              rocblas_int k, rocblas_double_complex *A,
                              rocblas_int lda,
                              const rocblas_double_complex *tau,
                              rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  ungqr_impl<rocblas_double_complex>(m, n, k, A, lda, tau, info);
}
void RocSolverBackend::sgels(Layout layout, Transpose trans, rocblas_int m,
                             rocblas_int n, rocblas_int nrhs, float *A,
                             rocblas_int lda, float *B, rocblas_int ldb,
                             rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  gels_impl<float>(trans, m, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::dgels(Layout layout, Transpose trans, rocblas_int m,
                             rocblas_int n, rocblas_int nrhs, double *A,
                             rocblas_int lda, double *B, rocblas_int ldb,
                             rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  gels_impl<double>(trans, m, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::cgels(Layout layout, Transpose trans, rocblas_int m,
                             rocblas_int n, rocblas_int nrhs,
                             rocblas_float_complex *A, rocblas_int lda,
                             rocblas_float_complex *B, rocblas_int ldb,
                             rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  gels_impl<rocblas_float_complex>(trans, m, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::zgels(Layout layout, Transpose trans, rocblas_int m,
                             rocblas_int n, rocblas_int nrhs,
                             rocblas_double_complex *A, rocblas_int lda,
                             rocblas_double_complex *B, rocblas_int ldb,
                             rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  gels_impl<rocblas_double_complex>(trans, m, n, nrhs, A, lda, B, ldb, info);
}
void RocSolverBackend::ssyev(Layout, Job j, Uplo u, rocblas_int n, float *A,
                             rocblas_int lda, float *w, rocblas_int *info) {
  syev_impl<float>(j, u, n, A, lda, w, info);
}
void RocSolverBackend::dsyev(Layout, Job j, Uplo u, rocblas_int n, double *A,
                             rocblas_int lda, double *w, rocblas_int *info) {
  syev_impl<double>(j, u, n, A, lda, w, info);
}
void RocSolverBackend::cheev(Layout layout, Job jobz, Uplo uplo, rocblas_int n,
                             rocblas_float_complex *A, rocblas_int lda,
                             float *w, rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  heev_impl(jobz, uplo, n, A, lda, w, info);
}
void RocSolverBackend::zheev(Layout layout, Job jobz, Uplo uplo, rocblas_int n,
                             rocblas_double_complex *A, rocblas_int lda,
                             double *w, rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  heev_impl(jobz, uplo, n, A, lda, w, info);
}
void RocSolverBackend::sgeev(Layout, Job ju, Job jv, rocblas_int n, float *A,
                             rocblas_int lda, float *B, float *C, float *D,
                             rocblas_int E, float *F, rocblas_int G,
                             rocblas_int *info) {
  std::cerr << "[CLAP] rocSOLVER geev not supported\n";
  *info = -1;
}
void RocSolverBackend::dgeev(Layout, Job ju, Job jv, rocblas_int n, double *A,
                             rocblas_int lda, double *B, double *C, double *D,
                             rocblas_int E, double *F, rocblas_int G,
                             rocblas_int *info) {
  std::cerr << "[CLAP] rocSOLVER geev not supported\n";
  *info = -1;
}
void RocSolverBackend::cgeev(Layout, Job ju, Job jv, rocblas_int n,
                             rocblas_float_complex *A, rocblas_int lda,
                             rocblas_float_complex *B, rocblas_float_complex *D,
                             rocblas_int E, rocblas_float_complex *F,
                             rocblas_int G, rocblas_int *info) {
  std::cerr << "[CLAP] rocSOLVER geev not supported\n";
  *info = -1;
}
void RocSolverBackend::zgeev(Layout, Job ju, Job jv, rocblas_int n,
                             rocblas_double_complex *A, rocblas_int lda,
                             rocblas_double_complex *B,
                             rocblas_double_complex *D, rocblas_int E,
                             rocblas_double_complex *F, rocblas_int G,
                             rocblas_int *info) {
  std::cerr << "[CLAP] rocSOLVER geev not supported\n";
  *info = -1;
}
void RocSolverBackend::sgesvd(Layout, Job ju, Job jv, rocblas_int m,
                              rocblas_int n, float *A, rocblas_int lda,
                              float *s, float *U, rocblas_int ldu, float *VT,
                              rocblas_int ldvt, float *, rocblas_int *info) {
  gesvd_impl<float>(ju, jv, m, n, A, lda, s, U, ldu, VT, ldvt, info);
}
void RocSolverBackend::dgesvd(Layout, Job ju, Job jv, rocblas_int m,
                              rocblas_int n, double *A, rocblas_int lda,
                              double *s, double *U, rocblas_int ldu, double *VT,
                              rocblas_int ldvt, double *, rocblas_int *info) {
  gesvd_impl<double>(ju, jv, m, n, A, lda, s, U, ldu, VT, ldvt, info);
}
void RocSolverBackend::cgesvd(Layout layout, Job jobu, Job jobvt, rocblas_int m,
                              rocblas_int n, rocblas_float_complex *A,
                              rocblas_int lda, float *s,
                              rocblas_float_complex *U, rocblas_int ldu,
                              rocblas_float_complex *VT, rocblas_int ldvt,
                              float *superb, rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  gesvd_impl<rocblas_float_complex, float>(jobu, jobvt, m, n, A, lda, s, U, ldu,
                                           VT, ldvt, info);
}
void RocSolverBackend::zgesvd(Layout layout, Job jobu, Job jobvt, rocblas_int m,
                              rocblas_int n, rocblas_double_complex *A,
                              rocblas_int lda, double *s,
                              rocblas_double_complex *U, rocblas_int ldu,
                              rocblas_double_complex *VT, rocblas_int ldvt,
                              double *superb, rocblas_int *info) {
  if (layout != Layout::ColMajor) {
    *info = -1;
    return;
  }
  gesvd_impl<rocblas_double_complex, double>(jobu, jobvt, m, n, A, lda, s, U,
                                             ldu, VT, ldvt, info);
}

void RocSolverBackend::strtrs(Layout, Uplo, Transpose, Diag, rocblas_int,
                              rocblas_int, const float *, rocblas_int, float *,
                              rocblas_int, rocblas_int *info) {
  std::cerr << "[CLAP] rocSOLVER trtrs not supported\n";
  *info = -1;
}
void RocSolverBackend::dtrtrs(Layout, Uplo, Transpose, Diag, rocblas_int,
                              rocblas_int, const double *, rocblas_int,
                              double *, rocblas_int, rocblas_int *info) {
  std::cerr << "[CLAP] rocSOLVER trtrs not supported\n";
  *info = -1;
}
void RocSolverBackend::ctrtrs(Layout, Uplo, Transpose, Diag, rocblas_int,
                              rocblas_int, const rocblas_float_complex *,
                              rocblas_int, rocblas_float_complex *, rocblas_int,
                              rocblas_int *info) {
  std::cerr << "[CLAP] rocSOLVER trtrs not supported\n";
  *info = -1;
}
void RocSolverBackend::ztrtrs(Layout, Uplo, Transpose, Diag, rocblas_int,
                              rocblas_int, const rocblas_double_complex *,
                              rocblas_int, rocblas_double_complex *,
                              rocblas_int, rocblas_int *info) {
  std::cerr << "[CLAP] rocSOLVER trtrs not supported\n";
  *info = -1;
}
} // namespace clap
